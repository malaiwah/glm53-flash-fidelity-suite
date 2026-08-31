#!/usr/bin/env python3
"""Assemble reports/hidden-replay-equivalence.json from the on-box artifacts.

Inputs (all produced on box 486679 by engines/tools/hidden_replay_stage.sh):
  comparator      receipts/hidden-replay-comparator.json
  reproduction    receipts/reproduction-check.json
  three_run       receipts/stream-k6-kld-3run.json
  fetch           receipts/nonrouted-sparse-fetch.json
  selftest        receipts/hidden-replay-selftest.json
  env             receipts/env-versions.txt, receipts/nvidia-smi.txt
  captures        runs/hidden-run{1,2,3}/hidden-capture.json

Prior art (verified from primary sources at pinned revisions, not quoted from
memory): festr2/kimi-k3-distribution-fidelity-1024x2048-v1 @ 402919ae...,
validation/hidden-replay-qualification.json; and the doc
local-inference-lab/rtx6kpro@master models/kimi-k3/distribution-fidelity-1024x2048.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

SCHEMA = "malaiwah.glm53-hidden-replay-equivalence.v1"

SEALED_STREAM_MEAN = 0.013714888822596553

# on-box-only fetch helpers, named by content sha (they are not in git)
ON_BOX_HELPERS = {
    "par_fetch.sh": "4cec05270c2ca3ff5e2f28614bb68268b54e433e62290d6a90f152f83a6984db",
    "fetch_fast.sh": "c742c30ff3a0b658a51af739318243e26ba38d8a4144cb60ecac8a66a28098c7",
}

# ---- prior art, transcribed from the primary artifacts ---------------------
FESTR = {
    "who": "Festr (festr2)",
    "model": "kimi-k3 (official MXFP4 checkpoint, canonical BF16 LM head)",
    "stack": "vLLM tensor-parallel 16, RTX 6000 Pro cluster",
    "doc": "github.com/local-inference-lab/rtx6kpro, models/kimi-k3/distribution-fidelity-1024x2048.md",
    "dataset": "festr2/kimi-k3-distribution-fidelity-1024x2048-v1",
    "dataset_revision": "402919ae70d61396087571b63fe9185d95491afb",
    "receipt_file": "validation/hidden-replay-qualification.json",
    "receipt_status": "qualified",
    "suite": "32-context live-logit qualification suite (2048-token contexts)",
    # canonical_result.kl_reference_to_replay, i.e. KL(live || replayed)
    "mean_replay_kld": 1.2293254239455558e-06,
    "max_token_replay_kld": 0.0019517888166065158,
    "p99_replay_kld": 3.490941640881102e-06,
    "p99_9_replay_kld": 0.0002741868622527322,
    "median_replay_kld": 0.0,
    "top1_agreement_live_vs_replayed": 0.9999542012701514,
    "chunk_invariance_delta_of_means": 1.4895590939580136e-09,
    "chunk_invariance_definition": (
        "canonical (vocab_chunk 10240, position_block 128) vs alternative "
        "(vocab_chunk 8192, position_block 64); BOTH knobs vary, comparator on CPU "
        "with deterministic_algorithms=True and tf32=False"
    ),
    "comparator_device": "cpu",
    "runtime_repeat_sentinels": {
        "definition": (
            "the SAME runtime captured three times on 64 stratified sentinel contexts; "
            "pairwise mean KLD between identical serving runs"
        ),
        "pair_00_vs_01": {"mean_kld": 0.0032166686, "ci95": [0.00269235, 0.00379022],
                          "top1_agreement": 0.98326056},
        "pair_00_vs_02": {"mean_kld": 0.0031814546, "ci95": [0.00267033, 0.00371788],
                          "top1_agreement": 0.98338269},
        "pair_01_vs_02": {"mean_kld": 0.0031337795, "ci95": [0.00261258, 0.00368719],
                          "top1_agreement": 0.98348192},
    },
    "interpretation_rule": (
        "Festr's own scope rule: a KLD threshold measured on one model, corpus, "
        "tokenizer, vocabulary or serving runtime does not transfer to another; the "
        "metric ranks candidates only within that artifact's frozen identities"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="local dir holding the downloaded on-box receipts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--box", default="486679")
    args = ap.parse_args()

    root = args.artifacts.resolve()
    comparator = load(root / "hidden-replay-comparator.json")
    reproduction = load(root / "reproduction-check.json")
    three_run = load(root / "stream-k6-kld-3run.json")
    fetch = load(root / "nonrouted-sparse-fetch.json")
    selftest = load(root / "hidden-replay-selftest.json")
    env_text = (root / "env-versions.txt").read_text(encoding="utf-8").strip()
    smi_text = (root / "nvidia-smi.txt").read_text(encoding="utf-8").strip()

    captures = []
    for n in (1, 2, 3):
        p = root / f"hidden-capture-run{n}.json"
        if p.is_file():
            captures.append(load(p))

    # backend.json carries the streaming lane's own stack identity (the
    # campaign's lane_identity / backend_identity hashes).  Cite it by digest.
    backends = []
    for n in (1, 2, 3):
        p = root / f"backend-run{n}.json"
        if p.is_file():
            b = load(p)
            backends.append({
                "cold_run": n,
                "torch_version": b.get("torch_version"),
                "cuda_runtime_version": b.get("cuda_runtime_version"),
                "device_name": b.get("device_name"),
                "grouped_mm_kernel": b.get("grouped_mm_kernel"),
                "attention_backend": b.get("attention_backend"),
                "experts_implementation": b.get("experts_implementation"),
                "parallelism": b.get("parallelism"),
                "numeric_policy": b.get("numeric_policy"),
                "lane_identity": b.get("lane_identity"),
                "lane_identity_sha256": b.get("lane_identity_sha256"),
                "backend_identity_sha256": b.get("backend_identity_sha256"),
            })

    runs = comparator["runs"]
    cross = comparator["cross_run_determinism"]
    invariance = comparator["vocab_chunk_invariance"]

    # ---- our metrics, per run and pooled -------------------------------
    per_run = []
    for row in runs:
        per_run.append({
            "run_dir": row["run_dir"],
            "cold_run": row["cold_run"],
            "windows": row["window_count"],
            "positions": row["positions"],
            "replay_kld_mean": row["replay_kld"]["mean"],
            "replay_kld_max": row["replay_kld"]["max"],
            "replay_kld_p99": row["replay_kld"]["p99"],
            "replay_kld_p99_9": row["replay_kld"]["p99_9"],
            "top1_agreement_live_vs_replayed": row["top1_agreement_live_vs_replayed"],
            "top1_matches": row["top1_matches"],
            "panel_mean_kld_via_live_logits": row["panel_mean_kld_via_live_logits"],
            "panel_mean_kld_via_replayed_logits": row["panel_mean_kld_via_replayed_logits"],
            "panel_delta_replayed_minus_live": row["panel_delta_replayed_minus_live"],
            "panel_via_live_bitwise_matches_kld_report": row["panel_via_live_bitwise_matches_kld_report"],
            "logits_bytes_stored": row["logits_bytes_stored"],
            "hiddens_bytes_stored": row["hiddens_bytes_stored"],
            "logits_payload_digest_sha256": row["logits_payload_digest_sha256"],
            "hiddens_payload_digest_sha256": row["hiddens_payload_digest_sha256"],
            "stream_capture_receipt_sha256": row["stream_capture_receipt_sha256"],
            "hidden_capture_receipt_sha256": row["hidden_capture_receipt_sha256"],
        })

    logits_bytes = per_run[0]["logits_bytes_stored"]
    hiddens_bytes = per_run[0]["hiddens_bytes_stored"]

    body: Dict[str, Any] = {
        "schema": SCHEMA,
        "title": "Hidden-replay equivalence for the GLM-5.3-Flash streaming lane",
        "protocol": {
            "name": "Phaelon's sign-off protocol",
            "cold_runs": len(runs),
            "one_forward_two_paths": (
                "each window is forwarded ONCE and emits BOTH the full fp32 logits "
                "(path A, byte-identical to a plain stream_score run) and the "
                "post-final-RMSNorm bf16 hidden state (path B)"
            ),
            "aligned_with": "Festr's kimi-k3 hidden-replay qualification (see prior_art)",
        },
        "cut_point": comparator["cut_point"],
        "cut_statement": comparator["cut_statement"],
        "cut_matches_festr": True,
        "cut_note": (
            "our EARLIER head-extraction artifact (head/head-extraction.json) shipped "
            "final_norm.safetensors NEXT TO the head for a norm+head replay; THIS protocol "
            "does not apply final_norm at replay time because the capture already sits "
            "after it -- the cut is stated here explicitly so the two artifacts are not confused"
        ),
        "dtypes": {
            "hidden": comparator["hidden_dtype"],
            "hidden_lossless": (
                "GLM-5.3-Flash's residual/norm output is natively torch.bfloat16; the "
                "capture hook ASSERTS the dtype at every window, so storing bf16 "
                "introduces no rounding"
            ),
            "logits": comparator["logits_dtype"],
            "replay_arithmetic": comparator["replay_definition"],
            "kld_arithmetic": comparator["kld_definition"],
            "panel_kld_arithmetic": comparator["panel_kld_definition"],
        },
        "vocab_size": comparator["vocab_size"],
        "hidden_width": comparator["hidden_width"],
        "head": comparator["head"],
        "head_provenance": {
            "source_repo": fetch.get("repo"),
            "source_revision": fetch.get("revision"),
            "inventory_sha256": fetch.get("inventory_sha256"),
            "fetch_receipt_schema": fetch.get("schema"),
            "sparse_disclosure": fetch.get("sparse_disclosure"),
            "head": fetch.get("head"),
            "published_head_sha_matches_receipt": (fetch.get("head") or {}).get(
                "published_head_sha_matches_receipt"),
            "published_head_content_equals_live_fetch": (fetch.get("head") or {}).get(
                "published_head_content_equals_live_fetch"),
            "note": (
                "the head used for replay is the lm_head from the same BF16 tree the "
                "student was built from; its sha256 was re-verified on fetch against our "
                "published head-extraction receipt"
            ),
        },
        "teacher_receipt_sha256": comparator["teacher_receipt_sha256"],
        "metrics_per_run": per_run,
        "cross_run_determinism": cross,
        "vocab_chunk_invariance": invariance,
        "path_a_reproduction": {
            "what": (
                "path A is the standard streaming scorer; its panel mean must reproduce "
                "the sealed K6 streaming number BITWISE -- a free lane-integrity check"
            ),
            "sealed_stream_mean": SEALED_STREAM_MEAN,
            "measured_mean": reproduction.get("measured_mean"),
            "mean_reproduced_exactly": reproduction.get("mean_reproduced_exactly"),
            "sealed_tokenwise_sha256": reproduction.get("sealed_tokenwise_sha256"),
            "distinct_tokenwise_kld_sha256": reproduction.get("distinct_tokenwise_kld_sha256"),
            "tokenwise_sha_matches_sealed": reproduction.get("tokenwise_sha_matches_sealed"),
            "scored_the_sealed_k6_surface": reproduction.get("scored_the_sealed_k6_surface"),
            "student_checkpoint_identity_sha256": reproduction.get("student_checkpoint_identity_sha256"),
            "bitwise_deterministic_across_runs": reproduction.get("bitwise_deterministic_across_runs"),
            "run_means": reproduction.get("run_means"),
        },
        "storage": {
            "logits_bytes_per_run": logits_bytes,
            "hiddens_bytes_per_run": hiddens_bytes,
            "logits_gib": round(logits_bytes / (1 << 30), 4),
            "hiddens_gib": round(hiddens_bytes / (1 << 30), 4),
            "hiddens_mib": round(hiddens_bytes / (1 << 20), 2),
            "shrink_factor": round(logits_bytes / hiddens_bytes, 2) if hiddens_bytes else None,
            "measured_how": "actual stored file sizes on disk, summed over the 25 panel windows",
            "why": (
                "this is the number that licenses a hidden-form same-lane teacher: the same "
                "panel carried as hiddens instead of logits"
            ),
        },
        "stack": {
            "box": args.box,
            "gpu": smi_text,
            "env_versions": env_text,
            "torch_version": comparator["torch_version"],
            "cuda_runtime_version": comparator["cuda_runtime_version"],
            "device": comparator["device"],
            "device_name": comparator["device_name"],
            "numeric_policy": comparator["numeric_policy"],
            "code_identity": comparator["code_identity"],
            "code_identity_note": (
                "hidden_replay.py, hidden_replay_selftest.py and fetch_nonrouted_sparse.py "
                "are NEW files; stream_score.py was NOT edited (another workflow owns "
                "in-flight changes there). The on-box stream_score.py is the committed "
                "origin/main copy, named by sha in code_identity"
            ),
            "selftest": selftest,
            "streaming_backend_per_run": backends,
            "lane_identity_sha256_distinct": sorted(
                {b["lane_identity_sha256"] for b in backends if b.get("lane_identity_sha256")}),
            "on_box_only_helpers": {
                "why": (
                    "hf download stalls on the 98,878-file content-addressed packed store "
                    "(it plans every file before writing one byte); these two helpers fetch "
                    "the pinned path list directly from the resolve endpoint. They move "
                    "bytes only -- no measurement code path touches them"
                ),
                "files": ON_BOX_HELPERS,
            },
        },
        "prior_art": FESTR,
        "comparison_table": [
            {
                "artifact": "Festr, kimi-k3 (vLLM TP16)",
                "mean_replay_kld": FESTR["mean_replay_kld"],
                "max_token_replay_kld": FESTR["max_token_replay_kld"],
                "p99_9_replay_kld": FESTR["p99_9_replay_kld"],
                "top1_agreement": FESTR["top1_agreement_live_vs_replayed"],
                "chunk_invariance_delta": FESTR["chunk_invariance_delta_of_means"],
                "runtime_repeat_between_identical_runs": FESTR["runtime_repeat_sentinels"]["pair_00_vs_01"]["mean_kld"],
                "source": f"{FESTR['dataset']}@{FESTR['dataset_revision']}",
            },
            {
                "artifact": "this receipt, GLM-5.3-Flash (streaming reference-forward lane)",
                "mean_replay_kld": per_run[0]["replay_kld_mean"],
                "max_token_replay_kld": per_run[0]["replay_kld_max"],
                "p99_9_replay_kld": per_run[0]["replay_kld_p99_9"],
                "top1_agreement": per_run[0]["top1_agreement_live_vs_replayed"],
                "chunk_invariance_delta": invariance["delta_of_means"],
                "runtime_repeat_between_identical_runs": cross.get(
                    "runtime_repeat_sentinel_mean_kld_between_runs"),
                "source": "this file",
            },
        ],
        "licenses": {
            "yes": [
                "a hidden-form SAME-LANE teacher for this lane: the panel carried as "
                "post-final-RMSNorm bf16 hiddens instead of fp32 logits, replayed through "
                "the published head at scoring time",
                "community hidden-capture submissions scored on this lane, provided the "
                "submitter's capture is qualified the same way on their own stack",
            ],
            "no": [
                "cross-stack replay without a per-stack qualification -- Festr's own "
                "interpretation rule: a replay qualified on one serving runtime says "
                "nothing about another. A different stack must re-run this protocol.",
                "any claim about capability, long-context, tool-use or free-running "
                "generation: this measures teacher-forced distribution fidelity only",
            ],
        },
        "credits": [
            "Festr (festr2) -- the kimi-k3 hidden-replay qualification whose conventions "
            "and metric set this receipt adopts so the two read side by side",
            "luke -- review and the hidden-capture direction",
            "Phaelon -- the sign-off protocol (three cold runs; one forward, two paths; "
            "state the cut) and the stack-disclosure standard this lane now follows",
        ],
        "artifacts_on_box": {
            "box": args.box,
            "note": "raw artifacts remain on the box for the verifier",
        },
    }

    if captures:
        body["hidden_captures"] = [
            {
                "cold_run": c.get("cold_run"),
                "receipt_sha256": c.get("receipt_sha256"),
                "hiddens_payload_digest_sha256": c.get("hiddens_payload_digest_sha256"),
                "window_count": c.get("window_count"),
                "lm_head": c.get("lm_head"),
                "stream_capture_receipt_sha256": c.get("stream_capture_receipt_sha256"),
                "student_label": c.get("student_label"),
                "elapsed_seconds": c.get("elapsed_seconds"),
            }
            for c in captures
        ]

    # self-seal: sha256 over the canonical body without the seal field
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "receipt_sha256": body["receipt_sha256"],
        "mean_replay_kld_run1": per_run[0]["replay_kld_mean"],
        "top1_run1": per_run[0]["top1_agreement_live_vs_replayed"],
        "shrink_factor": body["storage"]["shrink_factor"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
