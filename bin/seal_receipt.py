#!/usr/bin/env python3
"""Turn a finished run into a sealed `submission-receipt.v1`.

Runs on the instance (cloud lane) or on the operator's machine (local lane),
from the same code, so both recipes emit the SAME receipt schema.  The derived
fields are computed with `registry/tools/registry_lib.py` -- the registry's own
implementation, imported, not re-typed -- so the validator's recomputation and
ours cannot silently disagree.

    python3 bin/seal_receipt.py --job job.json --receipts <dir> --out receipt.json

Determinism evidence is the sha256 of the TOKENWISE-KLD TENSOR, never of a
report file or an archive.  Report files embed run indices, paths and timings,
so five identical measurements produce five different report hashes; the tensor
hash is the thing that actually says the numbers matched.  The registry
enforces this (DET-001) and rejects the alternative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fidelity.common import Console, read_json, sha256_file, write_json  # noqa: E402
from fidelity.receipt import (                          # noqa: E402
    build_submission, host_environment, produced_by_block, validate_locally,
)


def collect_runs(receipts: Path) -> List[Dict[str, Any]]:
    """Read every per-run summary the engine left behind."""
    runs = []
    for d in sorted(receipts.glob("run-*")):
        for name in ("kld-report.json", "capture-receipt.json", "summary.json"):
            p = d / name
            if p.is_file():
                try:
                    runs.append({"dir": str(d), "file": name, "doc": read_json(str(p))})
                except ValueError:
                    pass
                break
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-json",
                    help="explicit measured values, when the engine writes them "
                         "somewhere this script does not know how to read")
    ap.add_argument("--suite-root", default=str(HERE.parent))
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    con = Console()
    suite_root = Path(args.suite_root).resolve()
    job = read_json(args.job)
    receipts = Path(args.receipts).resolve()

    if args.metrics_json:
        m = read_json(args.metrics_json)
    else:
        m = _aggregate(receipts, con)
        if m is None:
            runs = collect_runs(receipts)
            if not runs:
                con.err(
                    "no per-run summaries under %s (looked for run-*/kld-report.json, "
                    "capture-receipt.json, summary.json). Nothing measured, nothing to "
                    "seal. Pass --metrics-json to supply the values explicitly."
                    % receipts)
                return 2
            m = _rollup(runs)
        if m.get("value") is None:
            con.err(
                "the runs under %s carry no mean tokenwise KLD (the capture stage "
                "writes LOGITS; the `score` stage runs k6_kld_report.py over them). "
                "Run `stage_measure.sh score` before sealing, or pass --metrics-json."
                % receipts)
            return 2

    run_means = list(m.get("run_means") or [])
    evidence_hashes = sorted(set(m.get("evidence_hashes") or []))
    identical = len(evidence_hashes) == 1 and len(run_means) >= 1

    target = job.get("target", {})
    panel = job.get("panel", {})
    lane = job.get("lane", "streaming")
    bits = target.get("bits")

    scope = (job.get("scope")
             or _scope_from_receipts(receipts, con)
             or _scope_from_registry(suite_root, target, con)
             or _default_scope(target))

    doc = build_submission(
        suite_root=suite_root,
        lane=lane,
        measurer=job.get("measurer") or {
            "name": "unknown", "handle": "unknown", "url": None,
            "is_artifact_author": False,
        },
        artifact={
            "repository": target.get("repo_id"),
            "revision": target.get("revision"),
            "url": "https://huggingface.co/%s" % target.get("repo_id"),
            "container": target.get("container", "exl3"),
            "precision_label": target.get("precision_label")
                               or ("%g bpw" % bits if bits else None),
            "size_bytes": target.get("size_bytes"),
            "index_sha256": target.get("index_sha256"),
            "config_sha256": target.get("config_sha256"),
            "shard_hash_verification": target.get("shard_hash_verification", "none"),
            "codec": {
                "family": target.get("codec") or "exl3-mcg",
                "bits_per_weight_nominal": bits,
                "bits_per_weight_effective": None,
                "group_size": None,
                "quantizer_tool": "exllamav3 EXL3",
                # the storage-ABI pin when there is one (TR3), else the
                # quantizer version the release's own config states (stock
                # exllamav3 ships no ABI file but does say "1.4.4")
                "quantizer_version": (target.get("exllamav3_pin")
                                      or target.get("quantizer_version")),
            },
            "scope": scope,
            "producer": job.get("producer") or {
                "name": (target.get("repo_id") or "/").split("/")[0],
                "handle": (target.get("repo_id") or "/").split("/")[0],
                "url": "https://huggingface.co/%s"
                       % (target.get("repo_id") or "/").split("/")[0],
            },
            "tensor_parallel_pre_sliced": bool(target.get("tp_sliced")),
            "tensor_parallel_world_size": target.get("tp_world_size"),
        },
        panel={
            "panel_ref": panel.get("panel_ref"),
            "panel_token_sha256": panel.get("panel_token_sha256"),
            "panel_receipt_sha256": panel.get("panel_receipt_sha256"),
            "contexts": panel.get("contexts"),
            "scored_positions_total": panel.get("scored_positions"),
        },
        reference=job.get("reference") or {
            "reference_ref": panel.get("reference_ref"),
            "teacher_receipt_sha256": None,
            "teacher_backend_identity_sha256": None,
        },
        metric={
            "name": m.get("metric_name", "mean_of_run_means_tokenwise_kld"),
            "value": m["value"],
            "units": "nats",
            "direction": m.get("direction", "reference_to_candidate"),
        },
        auxiliary_metrics={"top1_agreement": m.get("top1_agreement")},
        estimator={
            "accumulation_dtype": m.get("accumulation_dtype", "float64"),
            "logits_dtype": m.get("logits_dtype", "fp32"),
            "two_pass": False,
            "vocab_chunk": None,
            "stack_relation": m.get("stack_relation", "same_stack"),
            "head_policy": m.get("head_policy", "native_head"),
            "zero_handling": None,
        },
        determinism={
            "run_count": len(run_means),
            "cold_start_per_run": True,
            "run_means": run_means,
            "identical_across_runs": identical,
            "evidence_kind": "tokenwise_kld_sha256",
            "evidence_hashes": evidence_hashes,
            "distinct_evidence_hash_count": len(evidence_hashes),
            "per_run_report_sha256": m.get("per_run_report_sha256") or [],
            # never None: the schema types this as a string.
            "note": m.get("determinism_note") or (
                "%d cold run(s); determinism evidence is the tokenwise-KLD "
                "tensor digest, not any report file" % len(run_means)),
        },
        measurement_scope={
            "scored_positions": panel.get("scored_positions"),
            "contexts": panel.get("contexts"),
            "positions_per_context": panel.get("positions_per_context"),
            "covers_full_panel": bool(m.get("covers_full_panel", True)),
            "subset_detail": m.get("subset_detail"),
            "position_filter": "all",
        },
        # Provenance about the RUNNER is known on the machine the runner lives
        # on. The cloud controller computes it on the caller's laptop (where
        # there is a git checkout) and ships it in job.json; the fallback below
        # only fires for a local run or a hand-assembled job.
        produced_by=job.get("produced_by") or produced_by_block(
            suite_root,
            "bin/measure_cloud.py" if job.get("recipe") == "cloud"
            else "bin/measure_local.py",
            dependencies=job.get("dependencies") or {}),
        environment=host_environment(job.get("environment")),
        cost=job.get("cost") or {"usd": None, "basis": None},
        evidence=job.get("evidence") or [],
        extra_disclosures=(job.get("disclosures") or []) + _lineage_disclosures(target),
    )

    write_json(args.out, doc)
    con.ok("receipt sealed", "%s  sha256 %s" % (args.out, doc["receipt_sha256"][:16]))

    if not args.no_validate:
        proc = validate_locally(suite_root, Path(args.out))
        if proc is None:
            con.warn("registry validator not found; receipt written but unchecked")
        else:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            if proc.returncode != 0:
                con.err("the registry validator rejected this receipt "
                        "(exit %d). It is written to %s so you can inspect it, "
                        "but do not submit it as-is." % (proc.returncode, args.out))
                return proc.returncode
    return 0


def _run_mean(doc: Dict[str, Any]) -> Optional[float]:
    """The run's mean tokenwise KLD, wherever this engine puts it.

    k6_kld_report's per-run `kld-report.json` nests it as summary.mean -- a
    flat-key-only search found nothing there and sealed a NULL metric after a
    fully paid measurement.
    """
    for key in ("mean_tokenwise_kld", "mean_kld", "measured_mean_kld", "value"):
        if isinstance(doc.get(key), (int, float)):
            return float(doc[key])
    summary = doc.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("mean"), (int, float)):
        return float(summary["mean"])
    return None


def _rollup(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    means, hashes, reports, top1 = [], [], [], []
    for r in runs:
        doc = r["doc"]
        mean = _run_mean(doc)
        if mean is not None:
            means.append(mean)
        for key in ("tokenwise_kld_sha256", "tokenwise_sha256"):
            if doc.get(key):
                hashes.append(doc[key])
                break
        if isinstance(doc.get("top1_agreement"), (int, float)):
            top1.append(float(doc["top1_agreement"]))
        p = Path(r["dir"]) / r["file"]
        reports.append(sha256_file(str(p)))
    return {
        "value": sum(means) / len(means) if means else None,
        "run_means": means,
        "evidence_hashes": hashes,
        "per_run_report_sha256": reports,
        "top1_agreement": sum(top1) / len(top1) if top1 else None,
    }


def _aggregate(receipts: Path, con: Console) -> Optional[Dict[str, Any]]:
    """The lane scorer's OWN aggregate receipt, when the score stage ran.

    `k6_kld_report.py --out <profile>-packed-kld.json` already computed the
    mean of run means, the distinct tokenwise-kld hashes and the determinism
    verdict in fp64.  Recomputing them here from the per-run files would be a
    second implementation of the same arithmetic; read the sealed one instead
    and fall back to the per-run rollup only when it is absent.
    """
    candidates = sorted(receipts.glob("*-packed-kld.json"))
    if not candidates:
        return None
    if len(candidates) > 1:
        con.err("more than one aggregate KLD receipt under %s (%s) -- refusing "
                "to guess which one this measurement is"
                % (receipts, ", ".join(c.name for c in candidates)))
        raise SystemExit(2)
    doc = read_json(str(candidates[0]))
    mean = doc.get("measured_mean_kld")
    run_means = list(doc.get("run_means") or [])
    if not isinstance(mean, (int, float)) or not run_means:
        return None
    con.ok("aggregate KLD receipt", candidates[0].name)
    return {
        "value": float(mean),
        "run_means": [float(x) for x in run_means],
        "evidence_hashes": list(doc.get("distinct_tokenwise_kld_sha256") or []),
        "per_run_report_sha256": list(doc.get("kld_report_sha256") or []),
        "top1_agreement": doc.get("top1_agreement"),
        # The schema requires determinism.note to be a STRING, and a receipt
        # sealed from the aggregate had nothing to put there, so every cloud
        # run ended in "REJECTED: /determinism/note: expected type string, got
        # null" -- after the measurement. Say what was actually observed.
        "determinism_note": (
            "%d cold runs, %d distinct kld_report_sha256, %d distinct "
            "tokenwise_kld_sha256. The report-file digests differ per run and "
            "prove nothing; the tokenwise digest is the determinism evidence.%s"
            % (len(run_means),
               len(set(doc.get("kld_report_sha256") or [])),
               len(set(doc.get("distinct_tokenwise_kld_sha256") or [])),
               (" cold_run_deviation (verbatim): %s" % doc["cold_run_deviation"])
               if doc.get("cold_run_deviation") else "")),
        "aggregate_receipt": candidates[0].name,
        "aggregate_receipt_schema": doc.get("schema"),
        "aggregate_receipt_sha256": sha256_file(str(candidates[0])),
    }


def _scope_from_receipts(receipts: Path, con: Console) -> Optional[Dict[str, Any]]:
    """The scope a SURFACE read off the artifact itself, during this run.

    Ranked above the registry's record and far above the pessimistic default,
    because it is the only one derived from the bytes this measurement actually
    consumed.  A surface that can read its release's published recipe writes
    `artifact-scope.json` into the receipts tree (see stage_measure.sh); one
    that cannot, writes nothing and the chain falls through unchanged.
    """
    path = Path(receipts) / "artifact-scope.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(str(path))
    except Exception as exc:                              # noqa: BLE001
        con.warn("artifact-scope.json unreadable (%s); falling through" % exc)
        return None
    scope = doc.get("scope") if isinstance(doc, dict) else None
    if not isinstance(scope, dict) or not scope.get("assignments"):
        con.warn("artifact-scope.json carries no assignments; falling through")
        return None
    con.ok("scope read from the artifact itself",
           "%d tensor classes, source %s"
           % (len(scope["assignments"]), scope.get("schema", "?")))
    return scope


def _scope_from_registry(suite_root: Path, target: Dict[str, Any],
                         con: Console) -> Optional[Dict[str, Any]]:
    """Reuse the registry's existing description of this artifact, if it has one.

    Measuring an artifact somebody has already catalogued is the COMMON case --
    it is how a second party independently verifies a number.  Inventing a
    fresh scope for it is wrong twice over: the registry refuses the ingest
    ("artifact already exists with different content"), and even if it did not,
    two rows would disagree about what the same weights are.  The quantizer's
    published recipe is a property of the artifact, not of whoever measured it,
    so adopt it.
    """
    path = suite_root / "registry" / "data" / "artifacts.jsonl"
    if not path.is_file():
        return None
    repo = (target.get("repo_id") or "").lower()
    if not repo:
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                identity = row.get("identity") or {}
                candidates = {
                    str(row.get("repository") or "").lower(),
                    str(identity.get("repository") or "").lower(),
                    str((row.get("availability") or {}).get("uri") or "").lower(),
                }
                if repo in candidates or any(
                    c.endswith("/" + repo) or c.endswith(repo) for c in candidates if c
                ):
                    if row.get("scope"):
                        con.ok("scope adopted from the registry",
                               row.get("id", "?"))
                        return row["scope"]
    except (OSError, ValueError):
        return None
    return None


def _lineage_disclosures(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A quant OF a quant is not a quant of the reference.

    Stock exllamav3 records the checkpoint it consumed in
    `original_quantization_config`. When that says the parent was itself
    quantized -- turboderp's GLM-5.3-Flash releases were made from the FP8
    e4m3 release, not from BF16 -- the divergence this row measures includes
    whatever the parent already cost, and a reader comparing it against a row
    quantized straight from BF16 is comparing two different lineages. Read
    from the release's own metadata by the sniffer; never assumed.
    """
    parent = target.get("quantized_from")
    if not parent or str(parent).lower() in ("bf16", "fp32", "unknown", "none"):
        return []
    return [{
        "code": "quantized_from_quantized_parent",
        "severity": "caveat",
        "affects_comparability": True,
        "detail": (
            "The release's own quantization_config declares "
            "original_quantization_config.fmt = %s: this artifact was quantized "
            "from an already-quantized checkpoint, not from the BF16 reference "
            "this row is scored against. Its divergence therefore includes the "
            "parent's, and it is not lineage-comparable with a row quantized "
            "directly from BF16 at the same bit rate." % parent)
    }]


def _default_scope(target: Dict[str, Any]) -> Dict[str, Any]:
    """Honest default: routed experts quantized, everything else UNKNOWN.

    Guessing that attention or the head is native because it usually is would
    put a confident wrong claim into a permanent record. `unknown` shows in the
    scope_digest and costs the row `strict` -- which is the correct price for
    not knowing.
    """
    bits = target.get("bits")
    fam = target.get("codec") or "exl3-mcg"
    quantized = ["mlp.gate", "mlp.up", "mlp.down", "moe.experts"]
    unknown = ["embed_tokens", "attn.qkv", "attn.o", "lm_head"]
    assignments = [
        {"tensor_class": c, "treatment": "quantized", "format": fam,
         "bits_per_weight": bits, "layer_range": "all"} for c in quantized
    ] + [
        {"tensor_class": c, "treatment": "unknown", "format": "unknown",
         "bits_per_weight": None, "layer_range": "all",
         "note": "the release does not declare a per-tensor-class recipe for "
                 "this class; recorded as unknown rather than guessed"}
        for c in unknown
    ]
    return {
        "policy": "mixed", "head_policy": "unknown", "kv_cache_dtype": "unknown",
        "mtp_included": None, "activation_quantization": None,
        "assignments": assignments,
    }


if __name__ == "__main__":
    raise SystemExit(main())
