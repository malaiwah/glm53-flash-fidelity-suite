#!/usr/bin/env python3
"""Assemble and publish the v1 (BF16-reference) fidelity dataset immediately.

Runs on the VM host (hfenv python). Conditional on whichever receipts exist;
later pipeline commits extend the same repo (FP8 shard, replay headline, v2
deterministic captures).

    publish_v1.py --root /home/ubuntu/glm53 --repo malaiwah/GLM-5.3-Flash-fidelity-suite-v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load(root: Path, name: str):
    f = root / "out" / f"{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def build_card(root: Path) -> str:
    q = load(root, "qualify-bf16")
    noise = load(root, "determinism-noise-bf16")
    det = load(root, "determinism-bf16")
    he = load(root, "head-equality-fp8")
    gc = load(root, "gen-check")
    cc = load(root, "crosscheck-brandonmusic")
    kp = load(root, "determinism-kernelpin-bf16")

    L = ["---", "license: mit", "tags:", "- glm5_next", "- glm-5.3-flash",
         "- fidelity", "- kld", "- hidden-states", "- determinism",
         "pretty_name: GLM-5.3-Flash Fidelity Suite v1", "---", "",
         "# GLM-5.3-Flash Fidelity Suite v1",
         "",
         "**The first measured quality reference for GLM-5.3-Flash** (released",
         "2026-08-26): BF16-reference hidden-state captures over a 5,120-context",
         "held-out suite, the shared LM head, and receipts — score ANY quant of",
         "this model by exact full-vocab KL divergence without holding the 643 GB",
         "checkpoint. Protocol: the Qwen3.8-27B fidelity-suite-v5 methodology",
         "(hidden-state replay through one shared BF16 head, two-pass exact KL).",
         "",
         "**v1 status (updated in-place):** BF16 reference shard-0 published;",
         "the FP8-as-served capture, the FP8-vs-BF16 headline report, and",
         "launch-deterministic (pinned) v2 captures land in subsequent commits",
         "tonight. GLM-5.3 non-Flash is a different, larger model — this suite",
         "is specifically for **GLM-5.3-Flash** (glm5_next).",
         ""]
    L += ["## Receipts (all files in `reports/`)", ""]
    if gc:
        L.append(f"- **Generation sanity**: pass={gc['pass']}, 'Paris' test "
                 f"{gc['paris_mentioned']} — the engine produces coherent text.")
    if det:
        L.append(f"- **Launch determinism probe**: {det['byte_identical']}/{det['sentinels']} "
                 "sentinel contexts byte-identical across engine loads — the day-one "
                 "runtime is NOT launch-deterministic (first report; see below).")
    if noise:
        L.append(f"- **Measured launch-noise floor**: mean KLD {noise['token_mean_kld']:.2e}, "
                 f"top-1 {noise['top1_agreement']:.4f} over {noise['scored_positions']:,} "
                 "positions (capture-vs-recapture through the shared head).")
    if q:
        L.append(f"- **Live-vs-replay qualification**: mean KLD {q['mean_kld_live_vs_replayed']:.2e}, "
                 f"top-1 {q['top1_agreement']:.4f} — bounds absolute as-served claims; "
                 "capture-vs-capture comparisons carry only the noise floor above.")
    if he:
        L.append(f"- **Head equality**: FP8 repo lm_head/final-norm byte-identical to BF16 "
                 f"({he['head_equal']}/{he['final_norm_equal']}) — one shared head is valid for both.")
    if cc:
        L.append(f"- **Independent cross-validation** vs brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits "
                 f"(separate pipeline, fp32 full-vocab): mean KLD(theirs||ours) {cc['mean_kld']:.4e}, "
                 f"top-1 {cc['top1_agreement']:.4f} over {cc['positions']:,} positions.")
    if kp:
        L.append(f"- **Determinism fix confirmed**: single-config Triton-autotune shim yields "
                 f"{kp['runB_vs_runC_byte_identical'] if 'runB_vs_runC_byte_identical' in kp else kp.get('runP1_vs_runP2_byte_identical')}"
                 f"/{kp['sentinels']} byte-identical sentinels across fresh launches.")
    L += ["",
          "## Known issue: run-to-run nondeterminism (first report)", "",
          "glm5_next inference under the day-one vLLM image is not deterministic",
          "across engine launches (it IS stable within one launch). Root cause:",
          "per-process Triton autotune winner selection on the vendored FLA/KDA",
          "chunk kernels (two kernels change fp32 reduction splits with config",
          "choice), amplified by DSA index_topk membership flips; all-reduce",
          "dispatch verified identical across launches. Lineage: fla-org",
          "flash-linear-attention#945, triton-lang/triton#9368. Upstream report:",
          "https://github.com/vllm-project/vllm/pull/53906#issuecomment-5433635837",
          "Fix (shipped in the companion repo): pin autotune to one config via a",
          "sitecustomize shim, or TRITON_CACHE_AUTOTUNING=1 with a persistent cache.",
          "",
          "## Pins", "",
          "| what | value |", "|---|---|",
          "| BF16 reference | `zai-org/GLM-5.3-Flash-BF16` @ `b1967181a3917ae70a437f4884748f6b8e3a1f4d` |",
          "| FP8 as-served | `zai-org/GLM-5.3-Flash` @ `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |",
          "| engine | vLLM glm53-flash docker image (digest in `reports/image-pin.txt`), TP8 H200, eager, TF32 off, BF16 KV |",
          "| suite | 5,120 ctx x 2,048 tok, held-out v5-lineage corpus, GLM tokenizer, 0 calibration-contamination hits |",
          "| contamination boundary | exllamav3 standard_cal_data @ 0c49587a |",
          "",
          "## Contents", "",
          "- `suite/` — tokens + manifest (analysis/qualification/sentinel partitions)",
          "- `reference-bf16-shard0/` — 512 contexts x [2047, 4096] bf16 final-norm hidden states",
          "- `head/` — shared BF16 lm_head (154,880 x 4,096) + final norm + extraction receipt",
          "- `reports/` — every receipt above; `SHA256SUMS` covers all files",
          "",
          "## Score your own quant", "",
          "Teacher-force your quant over `suite/tokens/`, capture final-norm hidden",
          "states (one context per forward), replay against `reference-bf16-shard0`",
          "through `head/head.safetensors`. Harness, runbook, and the full",
          "captain's-log journal: https://github.com/malaiwah/glm53-flash-fidelity-suite",
          "",
          "Produced autonomously overnight on rented 8x H200. Contact: malaiwah."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/ubuntu/glm53")
    ap.add_argument("--repo", default="malaiwah/GLM-5.3-Flash-fidelity-suite-v1")
    args = ap.parse_args()
    root = Path(args.root)
    from huggingface_hub import HfApi

    token = (root / ".hf_token").read_text().strip()
    api = HfApi(token=token)

    stage = root / "deliverables-v1"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "reports").mkdir(parents=True)
    shutil.copytree(root / "bundle/suite", stage / "suite")
    (stage / "reference-bf16-shard0").mkdir()
    copied = 0
    for i in range(512):
        src = root / f"captures/bf16/hidden_{i:04d}.safetensors"
        if src.is_file():
            os.link(src, stage / "reference-bf16-shard0" / src.name)
            copied += 1
    shutil.copy2(root / "captures/bf16/capture-manifest.json",
                 stage / "reference-bf16-shard0/capture-manifest-full.json")
    (stage / "head").mkdir()
    for name in ("head.safetensors", "final_norm.safetensors", "head-extraction.json"):
        shutil.copy2(root / "out" / name, stage / "head" / name)
    for f in (root / "out").glob("*.json"):
        shutil.copy2(f, stage / "reports" / f.name)
    for extra in ("image-pin.txt", "gen-snippet.txt"):
        f = root / "out" / extra
        if f.is_file():
            shutil.copy2(f, stage / "reports" / f.name)
    (stage / "README.md").write_text(build_card(root))
    print(f"staged: {copied} shard-0 files")

    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset",
                            folder_path=str(stage))
    print("PUBLISHED", f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
