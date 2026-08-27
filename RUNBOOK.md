# GLM-5.3-Flash Fidelity Suite v1 — capture session runbook (post-review r2)

Goal: BF16 reference + FP8-as-served hidden-state captures of `zai-org/GLM-5.3-Flash`
over a 5,120-context held-out suite (v5-corpus lineage, GLM tokenizer), the shared
BF16 LM head, qualification/determinism/head-equality receipts, and the first
FP8-vs-BF16 KLD report — packaged for a HF dataset release.

## Pins (recorded 2026-08-26)

| What | Value |
|---|---|
| BF16 checkpoint | `zai-org/GLM-5.3-Flash-BF16` @ `b1967181a3917ae70a437f4884748f6b8e3a1f4d` (643 GB, 120 shards) |
| FP8 checkpoint | `zai-org/GLM-5.3-Flash` @ `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` (328 GB, 62 shards) |
| vLLM runtime | docker `vllm/vllm-openai@sha256:2c6da6c6f16e...` (`:glm53-flash`) — glm5_next exists ONLY here + PR #53906 |
| v5 corpus | `malaiwah/qwen38-27b-fidelity-suite-v5` `corpus/` (byte-identical archival copy) |
| Contamination boundary | exllamav3 `standard_cal_data` @ `0c49587a` — 0 hits |
| Suite | 5,120 ctx x 2,048 tok; 3,807/1,313/32 analysis/qual/sentinel; seed 20260826; `suite_token_sha256 2e0ea096...` |
| Tensors | head `lm_head.weight` (shard 00001), final norm `model.language_model.norm.weight` (shard 00120) |
| Determinism | `NVIDIA_TF32_OVERRIDE=0`, enforce_eager (hard-fail otherwise), max_num_seqs=1, sequential, chunk-accumulate, sentinel receipts |
| Engine kwargs (in capture contract) | TP=8, `enable_flashinfer_autotune=false`, `limit_mm_per_prompt={"image":0,"video":0}`, BF16 KV (auto) |

## Review status

5-reviewer adversarial audit complete; all 3 blockers + majors fixed in r2:
docker `-i`, qualify shard-hash mirroring, contract records TP/engine-kwargs,
free_bf16 numeric gate (KLD ≤ 1e-5 AND top1 ≥ 0.999 AND determinism receipt),
head-equality check before FP8 capture, multimodal profiler disabled, rank-0-only
IPC, eager hard-fail, chunk-accumulate, suite manifest GLM geometry, AppleDouble
excludes, python3-venv install. Qualify API verified present in v0.27/v0.28-era
vLLM source (max_logprobs=-1, prompt_logprobs=-1, FlatLogprobs).
Expected hook line: `hooked ['language_model.model.norm', ... x8]` — abort if
anything vision/MTP/indexer.

## Sequence (each `jl run` is polled to completion + gate-checked before the next)

```
bash make_bundle.sh                                             # BEFORE create — no paid idle
jl create --gpu H200 --num-gpus 8 --vm --storage 1200 --region IN2 --name glm53-fidelity --yes --json
jl upload MID bundle.tar.gz /home/ubuntu/
jl exec MID -- sh -lc 'mkdir -p /home/ubuntu/glm53 && tar -xzf /home/ubuntu/bundle.tar.gz -C /home/ubuntu/glm53 && mv /home/ubuntu/glm53/bundle/bundle/* /home/ubuntu/glm53/bundle 2>/dev/null; ls /home/ubuntu/glm53/bundle'
jl run --on MID --json --yes -- bash /home/ubuntu/glm53/bundle/remote/vm_setup.sh          # G1
jl exec MID -- sh -lc 'nohup bash /home/ubuntu/glm53/bundle/remote/notify_heartbeat.sh 900 >/home/ubuntu/glm53/logs/heartbeat.log 2>&1 & echo heartbeat-started'
jl run --on MID --json --yes -- bash /home/ubuntu/glm53/bundle/remote/download_model_vm.sh \
    zai-org/GLM-5.3-Flash-BF16 b1967181a3917ae70a437f4884748f6b8e3a1f4d /home/ubuntu/glm53/models/bf16
S=/home/ubuntu/glm53/bundle/remote/stage.sh
jl run --on MID --json --yes -- bash $S extract_head
jl run --on MID --json --yes -- bash $S capture_bf16                                       # G2 pace probe at 64 ctx
jl run --on MID --json --yes -- bash $S sentinel_bf16
jl run --on MID --json --yes -- bash $S qualify_bf16                                       # G3
jl run --on MID --json --yes -- bash /home/ubuntu/glm53/bundle/remote/download_model_vm.sh \
    zai-org/GLM-5.3-Flash 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a /home/ubuntu/glm53/models/fp8
jl run --on MID --json --yes -- bash $S free_bf16          # numeric gates enforced in-script
jl run --on MID --json --yes -- bash $S head_check_fp8     # byte-equality of FP8 repo's head/norm
jl run --on MID --json --yes -- bash $S capture_fp8                                        # G4
jl run --on MID --json --yes -- bash $S sentinel_fp8
jl run --on MID --json --yes -- bash $S qualify_fp8                                        # G5
jl run --on MID --json --yes -- bash $S replay
jl run --on MID --json --yes -- bash $S env_receipt
jl run --on MID --json --yes -- bash $S package
jl download MID /home/ubuntu/glm53/deliverables ./deliverables -r      # ~19 GB
jl pause MID --yes --json
```

## Retention policy

Deliverables ship shard-0 (512 ctx) per variant + all receipts/reports (~19 GB).
The full 5,120-context captures (~172 GB) REMAIN on the paused VM's disk until the
user decides: download more shards, publish more, or destroy. Destroy only after
local SHA256SUMS verification AND the user's explicit call on the full captures.

## Gates

- G1 `vm_setup`: `glm5_next in registry: True`, 8 GPUs, driver r580+ (else cu129 tag).
- G2 pace probe: measured s/context at 64 ctx -> projected leg cost; >2.5 s/ctx
  triggers trim decision (rebuild suite at 2,560) or abort (~$50 sunk).
- G3 `qualify_bf16`: mean KLD(live‖replayed) ≤ 1e-5, top1 ≥ 0.999 (enforced by
  free_bf16 script gate). Red = capture protocol invalid for this arch: stop.
- G4 budget check before FP8 leg: projected remaining ≥ $70 else defer FP8 leg
  (BF16-only dataset ships; FP8 later on 4x H200 @ $15.96/h).
- G5 `qualify_fp8` red -> publish BF16-only, document the FP8 attempt.
- Balance floor $25: instant pause, captures survive on disk.

## Fallbacks

- 8x H200 capacity gone -> wait/retry; last resort 8x on another region/day.
- FP8 head-equality fails -> replay with `--candidate-head` (decision + receipt).
- vLLM day-one bug in KDA/DSA on Hopper -> captures fail loudly at context 0
  (cheap); debug ≤ 30 min else abort leg.
- Do NOT publish to HF from the instance; deliverables come home first.
