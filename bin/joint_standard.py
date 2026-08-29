#!/usr/bin/env python3
"""joint-standard -- the adopted rigor from brandonmusic's proposed community
standard, wired into our receipts.

    bin/joint-standard protocol
    bin/joint-standard overlap-scan --panel PANEL.json --arrays DIR --out SEL.json
    bin/joint-standard canary --teacher window.safetensors [--tokens t.npy]
    bin/joint-standard analyze --report kld-report.json --scope-file SEL.json
    bin/joint-standard paired --a A.json --b B.json --label-a K6 --label-b K8
    bin/joint-standard mcnemar --a-only 1629 --b-only 963
    bin/joint-standard stamp --in receipt.json --out stamped.json

Every output carries the protocol stamp (schema + file hash + scoring hash) and
``not_submittable: true``: these are ANALYSIS receipts, not measurements.  A
measurement row still has to come through registry/tools/registry_add.py.

Stdlib only for everything except ``canary`` (which needs numpy to hold a logits
tensor) and the optional numpy bootstrap backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jointstd import canary as canary_mod          # noqa: E402
from jointstd import ngram as ngram_mod            # noqa: E402
from jointstd import oracle as oracle_mod          # noqa: E402
from jointstd import protocol as protocol_mod      # noqa: E402
from jointstd import stats as stats_mod            # noqa: E402

EXIT_OK, EXIT_REFUSED, EXIT_CANARY = 0, 3, 2

ANALYSIS_SCHEMA = "malaiwah.glm53-joint-standard-analysis.v1"
SELECTION_SCHEMA = "malaiwah.glm53-joint-standard-window-selection.v1"
CANARY_SCHEMA = "malaiwah.glm53-joint-standard-r0-canary.v1"
PAIRED_SCHEMA = "malaiwah.glm53-joint-standard-paired.v1"


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _emit(doc: Dict[str, Any], proto: protocol_mod.Protocol,
          out: Optional[str]) -> None:
    doc["not_submittable"] = True
    doc["emitted_by"] = "bin/joint_standard.py"
    proto.stamp_into(doc)
    protocol_mod.require_stamp(doc, proto)
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if out:
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        sys.stderr.write("wrote %s\n" % out)
    else:
        sys.stdout.write(text)


# ------------------------------------------------------------------ loaders
def load_per_window(path: str) -> List[Dict[str, Any]]:
    """Accept every per-window shape this repository and his repository emit."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    # our kld-report.json:  per_window[] = {window_id, domain, summary{...}}
    pw = doc.get("per_window")
    if isinstance(pw, list) and pw and isinstance(pw[0], dict) and "summary" in pw[0]:
        out = []
        for w in pw:
            s = w["summary"]
            out.append({
                "window_id": w["window_id"],
                "domain": w.get("domain"),
                "document_id": w.get("document_id"),
                "count": int(s["count"]),
                "mean": float(s["mean"]),
                "std": (float(s["std"]) if s.get("std") is not None else None),
            })
        return out

    # flat per_window[] = {window_id, domain, mean_kld|mean, count?}
    if isinstance(pw, list) and pw and isinstance(pw[0], dict):
        out = []
        for w in pw:
            out.append({
                "window_id": w["window_id"],
                "domain": w.get("domain"),
                "document_id": w.get("document_id"),
                "count": int(w.get("count", w.get("prediction_positions", 2047))),
                "mean": float(w.get("mean_kld", w.get("mean"))),
                "std": (float(w["std"]) if w.get("std") is not None else None),
            })
        return out

    # his run-<id>.json:  windows{wid: {domain, mean_kld, ...}}
    wins = doc.get("windows")
    if isinstance(wins, dict) and wins:
        out = []
        for wid, w in wins.items():
            out.append({
                "window_id": wid,
                "domain": w.get("domain"),
                "document_id": w.get("document_id"),
                "count": int(w.get("prediction_positions", 2047)),
                "mean": float(w["mean_kld"]),
                "std": None,
            })
        return out

    # bare mapping {window_id: mean}
    if all(isinstance(v, (int, float)) for v in doc.values()):
        return [{"window_id": k, "domain": None, "count": 2047,
                 "mean": float(v), "std": None} for k, v in doc.items()]

    raise SystemExit("cannot find per-window data in %s" % path)


def load_scope(path: Optional[str], scope: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if scope in (None, "selected", "clean"):
        sel = doc.get("selected_windows")
        if isinstance(sel, list):
            return [w if isinstance(w, str) else w["window_id"] for w in sel]
    if scope == "panel":
        pw = doc.get("per_window") or doc.get("calibration_overlap_scan", {}).get("per_window")
        if pw:
            return [w["window_id"] for w in pw]
    raise SystemExit("scope %r not found in %s" % (scope, path))


# ------------------------------------------------------------------- verbs
def cmd_protocol(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    doc = {
        "schema": "malaiwah.glm53-joint-standard-protocol-print.v1",
        "path": proto.path,
        "bytes": len(proto.raw),
        "protocol": proto.doc if args.full else {
            "scoring": proto.doc["scoring"],
            "selection": proto.doc["selection"],
            "uncertainty": proto.doc["uncertainty"],
            "canary_r0": proto.doc["canary_r0"],
        },
        "oracle": oracle_mod.probe(),
    }
    _emit(doc, proto, args.out)
    return EXIT_OK


def cmd_overlap_scan(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    with open(args.panel, "r", encoding="utf-8") as fh:
        panel = json.load(fh)
    windows = panel["windows"] if isinstance(panel, dict) else panel
    finals = sorted([w for w in windows if w.get("role") == "final"],
                    key=lambda w: w["window_id"])
    cals = [w for w in windows if w.get("role") != "final"]
    if not finals:
        raise SystemExit("no role==final windows in %s" % args.panel)

    n = args.ngram or proto.ngram_n
    thr = args.threshold if args.threshold is not None else proto.ngram_threshold

    def tok_path(wid: str) -> str:
        return os.path.join(args.arrays, "%s.tokens.npy" % wid)

    cal_grams = set()
    missing = 0
    for w in cals:
        p = tok_path(w["window_id"])
        if not os.path.exists(p):
            missing += 1
            continue
        cal_grams |= ngram_mod.token_ngrams(ngram_mod.load_tokens(p), n)
    cal_docs = {w.get("document_id") for w in cals}

    prepared = []
    for w in finals:
        p = tok_path(w["window_id"])
        if not os.path.exists(p):
            raise SystemExit("missing token array for sealed window %s: %s"
                             % (w["window_id"], p))
        toks = ngram_mod.load_tokens(p)
        prepared.append({
            "window_id": w["window_id"],
            "domain": w.get("domain"),
            "document_id": w.get("document_id"),
            # carried through so a downstream scope record can pin its own token
            # identity without refetching the panel
            "token_ids_sha256": w.get("token_ids_sha256"),
            "prediction_positions": w.get("prediction_positions"),
            "tokens": toks,
        })

    res = ngram_mod.scan(prepared, cal_grams, cal_docs, n=n, threshold=thr)
    res["calibration_windows_scanned"] = len(cals) - missing
    res["calibration_windows_missing_arrays"] = missing
    res["threshold_sensitivity"] = ngram_mod.threshold_sensitivity(res["per_window"])
    doc = {
        "schema": SELECTION_SCHEMA,
        # digests, not local paths: this file is meant to be published, and an
        # absolute path on somebody's laptop is not provenance.
        "panel_file": os.path.basename(args.panel),
        "panel_sha256": _sha256_file(args.panel),
        "arrays_dir": os.path.basename(os.path.normpath(args.arrays)),
        "arrays_token_files": sum(
            1 for f in os.listdir(args.arrays) if f.endswith(".tokens.npy")),
    }
    doc.update(res)
    rc = EXIT_OK
    if args.expect:
        cross, rc = _compare_selection(res, args.expect)
        doc["cross_check"] = cross
    _emit(doc, proto, args.out)
    return rc


def _compare_selection(res: Dict[str, Any], expect_path: str):
    """Pin our scan against a published selection file, window by window."""
    with open(expect_path, "r", encoding="utf-8") as fh:
        exp = json.load(fh)
    per = exp.get("calibration_overlap_scan", {}).get("per_window") or exp.get("per_window")
    if not per:
        sys.stderr.write("no per_window block in %s\n" % expect_path)
        return {"available": False, "reason": "no per_window block"}, EXIT_REFUSED
    expd = {e["window_id"]: e for e in per}
    bad, checked = [], 0
    for row in res["per_window"]:
        e = expd.get(row["window_id"])
        if not e:
            continue
        checked += 1
        ec = e.get("shared_13gram_count", e.get("shared_ngram_count"))
        ef = e.get("shared_13gram_fraction", e.get("shared_ngram_fraction"))
        if row["shared_ngram_count"] != ec or abs(row["shared_ngram_fraction"] - ef) > 5e-7:
            bad.append({"window_id": row["window_id"],
                        "ours": [row["shared_ngram_count"], row["shared_ngram_fraction"]],
                        "theirs": [ec, ef]})
            sys.stderr.write("MISMATCH %s: ours %d/%.6f vs theirs %s/%s\n"
                             % (row["window_id"], row["shared_ngram_count"],
                                row["shared_ngram_fraction"], ec, ef))
    sys.stderr.write("cross-check against %s: %d windows, %d mismatches\n"
                     % (os.path.basename(expect_path), checked, len(bad)))
    cross = {
        "against": os.path.basename(expect_path),
        "against_sha256": _sha256_file(expect_path),
        "windows_checked": checked,
        "mismatches": bad,
        "identical": not bad,
        "note": ("Independent reimplementation of brandonmusic's 13-gram "
                 "calibration-overlap scan, run on the real published token "
                 "arrays. Every window's shared-gram count and fraction match "
                 "his published window_selection.json exactly."
                 if not bad else "MISMATCHES PRESENT -- do not use this scope"),
    }
    return cross, (EXIT_OK if not bad else EXIT_REFUSED)


def cmd_canary(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    try:
        import numpy as np
    except Exception:
        sys.stderr.write("canary needs numpy (it has to hold a logits tensor)\n")
        return EXIT_REFUSED

    if args.teacher:
        if args.teacher.endswith(".safetensors"):
            try:
                from safetensors.numpy import load_file
                t = load_file(args.teacher)[args.key]
            except Exception:
                t = _read_safetensors_fp32(args.teacher, args.key)
        else:
            t = np.load(args.teacher, allow_pickle=False)
        source = os.path.abspath(args.teacher)
    else:
        rng = np.random.default_rng(args.seed)
        t = rng.normal(0.0, 4.0, size=(args.rows, args.vocab)).astype(np.float32)
        source = "synthetic(rows=%d, vocab=%d, seed=%d)" % (args.rows, args.vocab, args.seed)

    if args.rows_limit:
        t = t[: args.rows_limit]

    tokens = None
    if args.tokens:
        tokens = ngram_mod.load_tokens(args.tokens)[1:]

    vocab_limit = args.vocab_limit
    if vocab_limit is None and t.shape[-1] == proto.stored_vocab:
        vocab_limit = proto.vocab_limit

    doc: Dict[str, Any] = {"schema": CANARY_SCHEMA, "teacher_source": source}
    try:
        res = canary_mod.run_r0(
            t, vocab_limit=vocab_limit, stored_vocab=t.shape[-1],
            shift_ratio_min=proto.shift_ratio_min, tag=args.tag,
            realized_tokens=tokens, alignment_band=proto.alignment_band,
        )
        doc.update(res)
        _emit(doc, proto, args.out)
        return EXIT_OK
    except canary_mod.CanaryFailure as exc:
        doc.update({"verdict": "FAIL", "failure": str(exc)})
        _emit(doc, proto, args.out)
        sys.stderr.write("R0 CANARY FAILED: %s\n" % exc)
        sys.stderr.write("protocol says: abort the session; never shift the alignment\n")
        return EXIT_CANARY


def _read_safetensors_fp32(path: str, key: str):
    """Minimal safetensors reader for one F32 tensor -- no safetensors package."""
    import numpy as np
    import struct

    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen).decode("utf-8"))
        meta = header[key]
        if meta["dtype"] != "F32":
            raise SystemExit("%s[%s] is %s, expected F32" % (path, key, meta["dtype"]))
        start, end = meta["data_offsets"]
        fh.seek(8 + hlen + start)
        buf = fh.read(end - start)
    return np.frombuffer(buf, dtype="<f4").reshape(meta["shape"])


def cmd_analyze(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    per_window = load_per_window(args.report)
    keep = load_scope(args.scope_file, args.scope)
    scope_name = args.scope or ("selected" if keep else "panel")
    if keep:
        missing = [w for w in keep if w not in {x["window_id"] for x in per_window}]
        if missing and not args.allow_partial:
            sys.stderr.write(
                "REFUSED: the scope names %d windows this receipt does not carry "
                "(%s...). A scope mean computed over a different window set is not "
                "the scope's mean. Re-run with --allow-partial only if you intend a "
                "documented subset.\n" % (len(missing), ", ".join(missing[:4])))
            return EXIT_REFUSED
        per_window = [w for w in per_window if w["window_id"] in set(keep)]
    if len(per_window) < 2:
        sys.stderr.write("need at least 2 windows, got %d\n" % len(per_window))
        return EXIT_REFUSED

    counts = sorted({w["count"] for w in per_window})
    equal_windows = len(counts) == 1

    summary = stats_mod.se_from_window_summaries(per_window)
    means = {w["window_id"]: w["mean"] for w in per_window}
    bs = stats_mod.window_block_bootstrap(
        means, b=args.bootstrap_b, seed=args.seed, backend=args.backend)

    oracle_bs = None
    if args.oracle and equal_windows:
        oracle_bs = oracle_mod.block_bootstrap_via_kld_eval(
            means, b=args.bootstrap_b, seed=args.seed,
            positions_per_window=counts[0])

    sr = stats_mod.sigma_run(args.run_mean or [])
    quad = stats_mod.combine_quadrature(
        summary.get("se_clustered_window", float("nan")),
        sr.get("sigma_run"), gate=proto.sigma_run_gate)

    doc: Dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "source_report": os.path.abspath(args.report),
        "label": args.label,
        "scope": {
            "name": scope_name,
            "scope_file": (os.path.abspath(args.scope_file) if args.scope_file else None),
            "windows": sorted(means),
            "n_windows": len(means),
            "scored_positions": summary["n"],
            "equal_window_sizes": equal_windows,
            "positions_per_window": (counts[0] if equal_windows else counts),
        },
        "summary": summary,
        "bootstrap": bs,
        "bootstrap_oracle_kld_eval": oracle_bs,
        "sigma_run": sr,
        "se_quadrature": quad,
        "by_domain": stats_mod.domain_table(
            per_window, b=args.domain_bootstrap_b, seed=args.seed,
            backend=args.backend),
        "percentile_guard": stats_mod.percentile_guard(
            summary["n"], min_exceedances=proto.min_exceedances),
        "pooled_percentiles": stats_mod.guard_pooled_percentiles(per_window),
        "oracle": oracle_mod.probe(),
    }
    if oracle_bs:
        doc["oracle_agreement"] = {
            "mean_abs_diff_bca": max(
                abs(a - b) for a, b in zip(bs["ci95_bca"], oracle_bs["ci95_bca"])),
            "mean_abs_diff_percentile": max(
                abs(a - b) for a, b in zip(bs["ci95_percentile"],
                                           oracle_bs["ci95_percentile"])),
        }
    _emit(doc, proto, args.out)
    return EXIT_OK


def cmd_paired(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    a = {w["window_id"]: w["mean"] for w in load_per_window(args.a)}
    b = {w["window_id"]: w["mean"] for w in load_per_window(args.b)}
    keep = load_scope(args.scope_file, args.scope)
    if keep:
        ks = set(keep)
        a = {k: v for k, v in a.items() if k in ks}
        b = {k: v for k, v in b.items() if k in ks}
    res = stats_mod.paired_windows(a, b, args.label_a, args.label_b,
                                   boot_b=args.bootstrap_b, seed=args.seed,
                                   backend=args.backend)
    doc: Dict[str, Any] = {"schema": PAIRED_SCHEMA,
                           "scope": args.scope or ("selected" if keep else "panel"),
                           "source_a": os.path.abspath(args.a),
                           "source_b": os.path.abspath(args.b)}
    doc.update(res)
    if args.a_only_correct is not None and args.b_only_correct is not None:
        doc["mcnemar"] = stats_mod.mcnemar(args.a_only_correct, args.b_only_correct)
    else:
        doc["mcnemar"] = {
            "available": False,
            "reason": "McNemar needs per-position top-1 agreement for BOTH runs; "
                      "per-window means cannot supply a contingency table. Pass "
                      "--a-only-correct/--b-only-correct from a run that emits them.",
        }
    _emit(doc, proto, args.out)
    return EXIT_OK


def cmd_mcnemar(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    doc = {"schema": "malaiwah.glm53-joint-standard-mcnemar.v1"}
    doc.update(stats_mod.mcnemar(args.a_only, args.b_only,
                                 continuity=not args.no_continuity))
    _emit(doc, proto, args.out)
    return EXIT_OK


def cmd_stamp(args: argparse.Namespace) -> int:
    proto = protocol_mod.load(args.protocol)
    with open(args.inp, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        sys.stderr.write("only a JSON object can be stamped\n")
        return EXIT_REFUSED
    out = args.out or (args.inp + ".stamped.json")
    if os.path.abspath(out) == os.path.abspath(args.inp) and not args.in_place:
        sys.stderr.write("REFUSED: stamping in place would rewrite a receipt whose "
                         "own digest may already be recorded elsewhere. Pass --out, "
                         "or --in-place if you truly mean it.\n")
        return EXIT_REFUSED
    _emit(doc, proto, out)
    return EXIT_OK


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="joint-standard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default=None, help="path to the frozen protocol file")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("protocol", help="print the frozen protocol and both hashes")
    q.add_argument("--full", action="store_true")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_protocol)

    q = sub.add_parser("overlap-scan", help="n-gram calibration-overlap scan")
    q.add_argument("--panel", required=True)
    q.add_argument("--arrays", required=True)
    q.add_argument("--ngram", type=int, default=None)
    q.add_argument("--threshold", type=float, default=None)
    q.add_argument("--expect", help="cross-check against a published selection file")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_overlap_scan)

    q = sub.add_parser("canary", help="R0: self-KLD == 0.0 AND shift explosion")
    q.add_argument("--teacher", help=".safetensors or .npy of teacher logits")
    q.add_argument("--key", default="logits")
    q.add_argument("--tokens", help="tokens .npy, enables the alignment band check")
    q.add_argument("--vocab-limit", type=int, default=None)
    q.add_argument("--rows-limit", type=int, default=None)
    q.add_argument("--rows", type=int, default=64, help="synthetic rows")
    q.add_argument("--vocab", type=int, default=1024, help="synthetic vocab")
    q.add_argument("--seed", type=int, default=20260829)
    q.add_argument("--tag", default="start")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_canary)

    q = sub.add_parser("analyze", help="clustered SE + BCa + per-domain + sigma_run")
    q.add_argument("--report", required=True)
    q.add_argument("--label", default=None)
    q.add_argument("--scope-file")
    q.add_argument("--scope", choices=["selected", "panel"], default=None)
    q.add_argument("--allow-partial", action="store_true")
    q.add_argument("--run-mean", type=float, action="append",
                   help="repeat once per cold run to get sigma_run")
    q.add_argument("--bootstrap-b", type=int, default=5000)
    q.add_argument("--domain-bootstrap-b", type=int, default=1000)
    q.add_argument("--seed", type=int, default=20260829)
    q.add_argument("--backend", choices=["auto", "numpy", "stdlib"], default="auto")
    q.add_argument("--oracle", action="store_true",
                   help="also run the bootstrap through brandonmusic's kld_eval")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_analyze)

    q = sub.add_parser("paired", help="paired per-window ranking of two runs")
    q.add_argument("--a", required=True)
    q.add_argument("--b", required=True)
    q.add_argument("--label-a", default="a")
    q.add_argument("--label-b", default="b")
    q.add_argument("--scope-file")
    q.add_argument("--scope", choices=["selected", "panel"], default=None)
    q.add_argument("--a-only-correct", type=int, default=None)
    q.add_argument("--b-only-correct", type=int, default=None)
    q.add_argument("--bootstrap-b", type=int, default=2000)
    q.add_argument("--seed", type=int, default=20260829)
    q.add_argument("--backend", choices=["auto", "numpy", "stdlib"], default="auto")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_paired)

    q = sub.add_parser("mcnemar", help="McNemar from a contingency table")
    q.add_argument("--a-only", type=int, required=True)
    q.add_argument("--b-only", type=int, required=True)
    q.add_argument("--no-continuity", action="store_true")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_mcnemar)

    q = sub.add_parser("stamp", help="add the protocol stamp to a JSON receipt")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--out")
    q.add_argument("--in-place", action="store_true")
    q.set_defaults(fn=cmd_stamp)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except protocol_mod.ProtocolError as exc:
        sys.stderr.write("PROTOCOL REFUSAL: %s\n" % exc)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
