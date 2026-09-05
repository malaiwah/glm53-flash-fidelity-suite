#!/usr/bin/env python3
"""Render (and, only when asked, open) the standard fidelity-measurement post on
the measured model's Hub discussion page.

    bin/fidelity-post render  --result <extracted result dir> --out post.md
    bin/fidelity-post publish --result <extracted result dir> --token-file ~/.hf_token
                              [--receipt post-receipt.json]

The result directory is a verified candidate result (`result/` beside a
`measure-cloud --candidate-scope` run): `job.json`,
`receipts/root-qualification.json`,
`receipts/reference-comparison/comparison-receipt.json` and, when the
candidate dataset was published, `receipts/publish-root.json`.

Every number in the post is read from a sealed receipt; nothing is typed in.
The post names the reference root dataset and revision, the candidate
dataset and revision, the panel, the estimator, the qualification (two
fresh processes reproduced the capture bitwise), every disclosure, and the
receipts by digest, so a reader can refetch and recompute. It is a
third-party measurement and says so: `measured_by` is the measurer named in
the job, not the model's author. It also states what the number does NOT
do: a same-lane root does not retroactively upgrade rows measured against
another teacher.

`publish` opens exactly one discussion on the CANDIDATE model repository
(the quantized artifact that was measured), never a pull request, and
writes a sealed receipt naming the discussion URL and the body digest.
Stdlib only for `render`; `publish` needs huggingface_hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fidelity import common  # noqa: E402

POST_SCHEMA = "fidelity-suite/model-discussion-post.v1"
TITLE = "Fidelity measurement: KL(reference || this quant) on frozen tokens, receipt-backed"


def _read(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return common.parse_json(handle.read())


def _sealed(path: str, label: str):
    doc = _read(path)
    if not isinstance(doc, dict) or not common.verify_seal(doc):
        raise SystemExit("REFUSED: %s is not a valid self-sealed receipt" % label)
    return doc


def load_result(result_dir: str) -> dict:
    job = _read(os.path.join(result_dir, "job.json"))
    candidate = ((job.get("capture") or {}).get("candidate")
                 if isinstance(job.get("capture"), dict) else None)
    if not isinstance(candidate, dict):
        raise SystemExit("REFUSED: job.json carries no capture.candidate; this post is for "
                         "candidate measurements only")
    qualification = _sealed(
        os.path.join(result_dir, "receipts", "root-qualification.json"), "qualification")
    comparison = _sealed(
        os.path.join(result_dir, "receipts", "reference-comparison", "comparison-receipt.json"),
        "reference comparison")
    publication = None
    publication_path = os.path.join(result_dir, "receipts", "publish-root.json")
    if os.path.isfile(publication_path):
        publication = _sealed(publication_path, "publication")
    return {"job": job, "candidate": candidate, "qualification": qualification,
            "comparison": comparison, "publication": publication}


def _method_line(decode: dict) -> str:
    """The decode the capture applied, said in that surface's own terms.

    A trellis row rendered through the FP8 sentence read "(None block None)"
    on a public post; each decodable surface gets its own sentence and an
    unknown method is rendered by name rather than through the wrong one.
    """
    method = str(decode.get("method"))
    qc = decode.get("quantization_config") or {}
    if method == "fp8-block-dequant-to-bf16":
        return ("dequantize-and-run, weights only: `%s` per tensor from the checkpoint's "
                "`quantization_config` (%s block %s), same engine, schedule and device as "
                "the reference capture" % (method, qc.get("fmt"), qc.get("weight_block_size")))
    if method == "exl3-trellis-decode-to-bf16":
        codebook = qc.get("codebook") or "per-module (read from each payload's own marker)"
        mixed = decode.get("mixed_fp8")
        rest = ("non-routed tensors kept in the source's block-scaled FP8 and dequantized "
                "the same way" if mixed else "non-routed tensors carried as shipped")
        return ("decode-and-run, weights only: `%s` -- each exl3 payload group "
                "(`trellis`/`suh`/`svh`/codebook marker) decoded to bf16 per module on "
                "the capture device via exllamav3's transcribed codebooks (codebook %s, "
                "declared %s bits), %s; same engine, schedule and device as the reference "
                "capture" % (method, codebook, qc.get("bits"), rest))
    if method == "exl3-trellis-tp-compose-to-bf16":
        codebook = qc.get("codebook") or "per-module (read from each payload's own marker)"
        return ("decode-and-run, weights only: `%s` -- each routed-expert module stored as "
                "tensor-parallel rank shards (the artifact's own hybrid_tr3_tail declares tp and "
                "the slicing axis) decoded to bf16 per rank on the capture device via "
                "exllamav3's transcribed codebooks (codebook %s, declared %s bits average) and "
                "composed into the whole weight in ascending rank order; non-routed tensors "
                "carried as shipped; same engine, schedule and device as the reference capture"
                % (method, codebook, qc.get("bits")))
    return "`%s` (decode recorded in the sealed runtime receipt)" % method


def render(loaded: dict) -> str:
    job, candidate = loaded["job"], loaded["candidate"]
    comparison, qualification = loaded["comparison"], loaded["qualification"]
    publication = loaded["publication"]
    target = job["target"]
    reference = candidate["reference"]
    metric = comparison["metric"]
    kl = comparison.get("kl") or {}
    captures = qualification.get("captures") or {}
    canonical = captures.get("canonical") or {}
    repeat = captures.get("repeat") or {}
    measurer = (job.get("measurer") or {}).get("name") or (job.get("capture") or {}).get("author")
    scope = candidate["scope"]
    decode = candidate["weights_decode"]
    lines = [
        "**What this is.** A third-party fidelity measurement of `%s` @ `%s` "
        "(%s, %g bits per weight as declared) against its unquantized reference, made "
        "with [quant-fidelity-suite](https://github.com/malaiwah/quant-fidelity-suite). "
        "Measured by `%s`, not by the model's author. Every number below is read from a "
        "sealed receipt named at the bottom." % (
            target["repo_id"], target["revision"], candidate["codec"],
            float(candidate["declared_bits"]), measurer),
        "",
        "| | |",
        "|---|---|",
        "| **KL(reference ‖ candidate), mean tokenwise, nats** | **%r** |" % metric["value"],
        "| top-1 agreement | %r |" % comparison.get("top1_agreement"),
        "| KL median / p95 / p99 / max | %r / %r / %r / %r |" % (
            kl.get("median"), kl.get("p95"), kl.get("p99"), kl.get("max")),
        "| reference root dataset | `%s` @ `%s` (capture `%s…`) |" % (
            reference["repository"], reference["revision"],
            reference["capture_content_digest"][:16]),
        "| panel | `%s`, %s contexts, %s scored positions |" % (
            reference["panel_id"],
            (comparison.get("panel") or {}).get("contexts"),
            (comparison.get("panel") or {}).get("scored_positions")),
        "| direction / vocabulary / accumulation | %s / full / %s |" % (
            metric.get("direction_label", metric.get("direction")),
            (comparison.get("estimator") or {}).get("accumulation_dtype")),
        "| method | %s |" % _method_line(decode),
        "| determinism | two fresh processes captured the candidate; both sealed captures "
        "carry content digest `%s…` (self-comparison %r) |" % (
            canonical.get("capture_content_digest", "")[:16],
            ((qualification.get("comparison") or {}).get("mean_kld")
             if isinstance(qualification.get("comparison"), dict) else "recorded in the qualification receipt")),
        "| comparability class | %s |" % (comparison.get("comparability") or {}).get("class"),
        "",
        "**Scope** (`scope_digest`): `%s`" % scope["scope_digest"],
        "",
    ]
    # HEAD-1d: the receipt names a head per side and the reproduction must ask
    # for the same rule, or the comparator refuses (HEAD-1b) when the heads
    # differ and reports shared_reference_head when they do not.
    own_heads = ((comparison.get("estimator") or {}).get("head_policy") == "native_head"
                 and (comparison.get("comparator") or {}).get(
                     "head_applied_reference_tensor_content_sha256") is not None)
    disclosures = comparison.get("disclosures") or []
    if disclosures:
        lines.append("**Disclosures on the comparison receipt:**")
        lines.append("")
        for item in disclosures:
            lines.append("- `%s` (%s): %s" % (
                item.get("code"), item.get("severity"), item.get("detail")))
        lines.append("")
    lines += [
        "**What this number does not do.** It is a same-lane distance from one reference "
        "capture on one panel. It does not rank this artifact against numbers measured on "
        "another panel, lane or reference, and a same-lane root does not retroactively "
        "upgrade rows measured against another teacher. Per-window scatter exceeds the gap "
        "between adjacent bit-widths; compare only within a group whose comparability keys "
        "match.",
        "",
        "**Receipts.**",
        "",
        "- comparison receipt `receipt_sha256` `%s`" % comparison["receipt_sha256"],
        "- root-qualification receipt `receipt_sha256` `%s` (canonical dataset_sha256 `%s`, "
        "repeat `%s`)" % (qualification["receipt_sha256"],
                          canonical.get("dataset_sha256"), repeat.get("dataset_sha256")),
        "- job `job_id_full` `%s`" % job.get("job_id_full"),
    ]
    if publication is not None:
        lines.append("- candidate capture dataset published: `%s` @ `%s`" % (
            publication.get("repository"), publication.get("revision")))
    lines += [
        "",
        "Reproduce: fetch the two datasets named above and run "
        "`fidelity-dataset compare --reference <root> --candidate <this>%s`; "
        "the receipt's estimator block is the exact recipe. Questions and corrections are "
        "welcome here; the registry files this as a third-party row (`measured_by` "
        "enumerated, never conflated with author-reported numbers)."
        % (" --own-heads" if own_heads else " --force-compute"),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="write the post body from the receipts")
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("publish", help="open the discussion on the candidate model repo")
    p.add_argument("--result", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--receipt", default=None)
    p.add_argument("--title", default=TITLE)
    args = parser.parse_args(argv)

    loaded = load_result(args.result)
    body = render(loaded)
    if args.command == "render":
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body)
        print("rendered %d bytes -> %s" % (len(body.encode("utf-8")), args.out))
        return 0

    from huggingface_hub import HfApi

    with open(args.token_file, "r", encoding="utf-8") as handle:
        token = handle.read().strip()
    repo = loaded["job"]["target"]["repo_id"]
    api = HfApi(token=token)
    discussion = api.create_discussion(
        repo_id=repo, title=args.title, description=body, repo_type="model",
        pull_request=False)
    receipt = common.seal({
        "schema": POST_SCHEMA,
        "receipt_sha256": "",
        "posted_at": common.utcnow(),
        "repository": repo,
        "discussion_num": discussion.num,
        "discussion_url": discussion.url,
        "title": args.title,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "comparison_receipt_sha256": loaded["comparison"]["receipt_sha256"],
        "qualification_receipt_sha256": loaded["qualification"]["receipt_sha256"],
        "job_id_full": loaded["job"].get("job_id_full"),
    })
    if args.receipt:
        with open(args.receipt, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("posted %s" % discussion.url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
