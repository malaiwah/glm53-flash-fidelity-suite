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


def _exl3_layout_clause(qc: dict) -> str:
    """The rotation layout an exl3 decode resolved, when the contract names one.

    Stock per-module vectors say nothing; a layer-shared layout (willfalco /
    jpsequeira `shared_h_v1`, brandonmusic `r7_shared`) names itself and its
    shared-vector count, and non-routed exl3 modules (o_proj, q_b_proj,
    indexer.wq_b, an exl3 lm_head) are named with their declared bits.
    """
    layout = qc.get("rotation_layout")
    parts = []
    if layout and layout != "per_module":
        shared = qc.get("shared_vectors") or {}
        parts.append("rotation layout `%s`: each routed expert's hidden-side rotation vector "
                     "resolved by name from its layer's shared tensor (%s shared vector(s))"
                     % (layout, shared.get("count")))
    nonrouted = qc.get("nonrouted_exl3") or {}
    if nonrouted.get("count"):
        parts.append("except %s non-routed exl3 module(s) decoded to bf16 by the same "
                     "function (declared bits %s)"
                     % (nonrouted["count"],
                        ", ".join("%s x%d" % kv for kv in sorted(
                            (nonrouted.get("declared_bits") or {}).items()))
                        or "undeclared"))
    return ("; " + "; ".join(parts)) if parts else ""


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
                "the capture device via exllamav3's transcribed codebooks (codebook %s as "
                "declared -- the capture reads each module's own marker -- at %s declared "
                "bits), %s%s; same engine, schedule and device as the reference capture"
                % (method, codebook, qc.get("bits"), rest, _exl3_layout_clause(qc)))
    if method == "exl3-trellis-tp-compose-to-bf16":
        codebook = qc.get("codebook") or "per-module (read from each payload's own marker)"
        return ("decode-and-run, weights only: `%s` -- each routed-expert module stored as "
                "tensor-parallel rank shards (the artifact's own hybrid_tr3_tail declares tp and "
                "the slicing axis) decoded to bf16 per rank on the capture device via "
                "exllamav3's transcribed codebooks (codebook %s, declared %s bits average) and "
                "composed into the whole weight in ascending rank order; non-routed tensors "
                "carried as shipped%s; same engine, schedule and device as the reference capture"
                % (method, codebook, qc.get("bits"), _exl3_layout_clause(qc)))
    if method == "nvfp4-modelopt-dequant-to-bf16":
        producer = qc.get("producer") or {}
        return ("dequantize-and-run, weights only: `%s` -- the ModelOpt NVFP4 dialect "
                "(quant_algo %s, producer %s): each routed-expert projection's packed e2m1 "
                "nibbles (group %s along the input axis) are decoded to exact fp32 as "
                "e2m1 x weight_scale.f32 x weight_scale_2 on the capture device and cast once "
                "to bf16 under the official tensor name, bitwise the compressed-tensors "
                "reference on real fetched rows; routed experts only -- every non-routed "
                "tensor is carried as shipped (plain bf16 under the official names); the "
                "per-tensor `input_scale` is an activation quantity and is NOT applied "
                "(%s), so the measurement is weights-only; same engine, schedule and device "
                "as the reference capture"
                % (method, qc.get("quant_algo"),
                   ("%s %s" % (producer.get("name"), producer.get("version"))
                    if producer else "undeclared"),
                   qc.get("group_size"), qc.get("activation_scheme")))
    if method == "gguf-dequant-to-bf16":
        census = qc.get("type_census") or {}
        types = ", ".join("%s x%d" % (t, census[t]) for t in sorted(census))
        general = qc.get("general") or {}
        return ("dequantize-and-run, weights only: `%s` -- the llama.cpp GGUF build `%s` "
                "(%d tensors; ggml types %s; quantized_by %s, quantization_version %s%s): "
                "EVERY tensor is block-dequantized to fp32 on the capture device by kernels "
                "proven bitwise against gguf-py's reference `dequantize` on real fetched blocks, "
                "then cast once to bf16 under the official tensor name (attn_k_b/attn_v_b "
                "composed into kv_b_proj, fused experts sliced per expert, both proven exact "
                "against the official BF16 release); the Q8_0 token embeddings and the "
                "artifact's OWN output head are decoded the same way, so this row runs "
                "own-heads (HEAD-1d) rather than the reference's head; same engine, schedule "
                "and device as the reference capture"
                % (method, qc.get("build"), qc.get("tensor_count") or 0, types or "none",
                   general.get("general.quantized_by") or "undeclared",
                   general.get("general.quantization_version"),
                   ("; imatrix-calibrated on %s" % general["quantize.imatrix.dataset"]
                    if general.get("quantize.imatrix.dataset") else "")))
    return "`%s` (decode recorded in the sealed runtime receipt)" % method


def _caveat_line(decode: dict):
    """One fixed sentence for a decode that is not the served artifact.

    The class row reads the receipt; this row says WHY a reconstructed row is
    advisory in words a model-page reader can use.  Six posts went out saying
    `strict` beside an author's own served-kernel figure on a different panel
    and teacher, with nothing on the page to explain the gap (review-science
    S1-2).  Same wording as the comparator's `weights_reconstructed` /
    `activation_quantization_not_captured` disclosures, shortened.
    """
    method = str(decode.get("method"))
    schemes = {(block.get("quantization_config") or {}).get("activation_scheme")
               for block in (decode, decode.get("mixed_fp8") or {}) if isinstance(block, dict)}
    if method.startswith("exl3-trellis-"):
        declared = sorted(str(s) for s in schemes if s not in (None, "", "none"))
        overlay = ((" The checkpoint also declares activation quantization "
                    "(`activation_scheme: %s`) that a weights-only capture does not apply, "
                    "so a served deployment quantizes activations at runtime and that term "
                    "is not in this number either." % ", ".join(declared))
                   if declared else "")
        return ("weights-only reconstruction on the HF `transformers` stack (the trellis "
                "payloads decoded to bf16 weights by this suite's transcription of exllamav3's "
                "codebooks, not by exllamav3 itself); the served kernel's fp16 activations and "
                "on-the-fly dequant are not in this number; a different panel and teacher than "
                "the author's own figure, so the two are not comparable.%s The registry files "
                "this row as advisory" % overlay)
    if method == "fp8-block-dequant-to-bf16" and "dynamic" in schemes:
        return ("weights-only dequantization on the HF `transformers` stack; the checkpoint "
                "declares `activation_scheme: dynamic`, so a served W8A8 deployment also "
                "quantizes activations per token and that term is not in this number, which "
                "is expected to understate the served divergence (not a mathematical bound). "
                "The registry files this row as advisory")
    return None


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
    caveat = _caveat_line(decode)
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
    ] + (["| caveat | %s |" % caveat] if caveat else []) + [
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
