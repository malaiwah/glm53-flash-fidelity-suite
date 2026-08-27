# GLM-5.3-Flash Fidelity Suite — session journal

Append-only ledger of the capture campaign. Each entry is written at the
milestone by the supervising Claude session; entries are never edited after the
fact. Times UTC; `~` marks reconstructed times. Local operator: malaiwah
(Michel Belleau). Supervisor: Claude Code (Fable 5) on the operator's Mac.

---

## 2026-08-26 ~23:00 — Mission decided
GLM-5.3-Flash (zai-org, 321.3B total / 18B active, glm5_next, released this
morning) has no quality reference of any kind. Decision: rent 8x GPUs, capture
BF16-reference + FP8-as-served hidden states over a 5,120-context held-out
suite (Qwen3.8-27B fidelity-suite-v5 protocol), publish dataset + receipts +
shared LM head so anyone can KLD-score any quant without the 643 GB checkpoint.
Research (12-agent sweep, then 3-agent prereq pass) established the hard facts:
vLLM support unmerged (PR #53906), day-one deployment is the docker image
`vllm/vllm-openai:glm53-flash` only; SM120 (RTX PRO 6000) broken day-one
("pe_dim must be 64 for fp8_ds_mla") -> Hopper it is; BF16 needs 772 GB VRAM ->
8x H200 VM, region IN2. Pins recorded: BF16 @ b1967181, FP8 @ 3f1971b7 (both
post-template-fix HEADs), image digest 2c6da6c6f16e, exllamav3 cal data @
0c49587a.

## 2026-08-26 ~23:20 — Suite built
v5 archival corpus (byte-identical from the HF dataset) re-tokenized with the
GLM tokenizer (needed transformers>=4.57 / TokenizersBackend; py3.9 venv
rebuilt as py3.12). 5,120 contexts x 2,048 tokens, 5 strata x 1,024, 837 source
clusters, 3,807/1,313/32 analysis/qualification/sentinel, **zero contamination
hits** against exllamav3 standard_cal_data. suite_token_sha256 2e0ea096. Suite
manifest initially carried Qwen geometry (hidden 5120/vocab 248320) — caught in
review, fixed to 4096/154880.

## 2026-08-26 ~23:45 — Blocker: balance $2.03
JarvisLabs balance $2.03 with the operator's qwen38-27b VM (481678) burning
$1.89/h. Reported; operator topped $200 first, later went to $699.

## 2026-08-27 ~00:15 — Adversarial review (5 reviewers): 3 blockers
Pre-spend audit of the kit found: (1) missing `docker run -i` -> every heredoc
stage would silently no-op, including the determinism receipts; (2) qualify
guaranteed SystemExit (capture --no-hash-shards vs qualify hash_shards=True);
(3) runbook chained detached jl runs with no polling. Plus majors: engine knobs
(TP, engine-kwargs) absent from the capture contract; multimodal profiler could
abort a loaded TP8 engine (fix: limit_mm_per_prompt=0); free_bf16 gated on file
existence, not the KLD value; FP8-repo head equality assumed, not verified;
suite manifest geometry wrong. All fixed (r2). Qualify API (max_logprobs=-1,
prompt_logprobs=-1, FlatLogprobs) verified present in v0.28-era vLLM source.

## 2026-08-27 ~00:40 — Skip-gates settled with live data
Official FP8 exists (we only measure it); real-weight NVFP4s already on HF
(LibertAIDAI 194.7 GB, axiomofmind W4A16 205.1 GB — unmeasured). Therefore:
never produce FP8; NVFP4 production skipped unless measured candidates fail.
Measurement-first is the strategy; this dataset is the yardstick.

## 2026-08-27 ~01:00 — ntfy + autonomy armed
Operator's ntfy topic wired (test ping delivered). Two notification layers: VM
posts stage events + 15-min heartbeats (outbound HTTP, immune to flaky inbound
SSH); supervisor posts analytics (balance, projections, gate verdicts).
Balance watcher armed for auto-launch at >= $250. Mac caffeinated 12h.

## 2026-08-27 ~01:10 — Cheap-prep architecture (operator's idea)
2 TB shared filesystem (fs 3393 -> 3394 after resize; region IN2 — filesystems
do NOT cross regions). L4 prep VM 482867 ($0.44/h) with fs attached —
fs-on-VM confirmed empirically (2.0T at /home/jl_fs, writable). Driver
580.126.20 -> default cu130 image OK; cu129 auto-fallback added anyway.

## 2026-08-27 ~01:20 — Publishing armed
HF token verified: malaiwah, write role (rotation after session mandatory —
token transited chat). GitHub repo created and pushed:
github.com/malaiwah/glm53-flash-fidelity-suite (pre-flight r4). Publish stage
generates the dataset card from live receipts at publish time.

## 2026-08-27 ~23:52–00:11 — Prep downloads (L4, $0.44/h)
BF16 599 GiB in **17 min (~600 MB/s)**; FP8 328 GB after it; smoke model
(Qwen3-0.6B); docker image pulled and tarballed to the fs; `glm5_next in
registry: True` confirmed inside the image. First live-fire bug: python3-venv
missing (predicted by review; guard was too quiet) — fixed.

## 2026-08-27 ~00:20 — Activation capture + cross-check built
Operator asked for MoE expert activations while the big box is rented.
Built activation_capture.py: per-layer attn_in/mlp_in block inputs (bf16) +
router_logits (fp32, natural top-8 ground truth) over a 92x2048 calibration
suite from exllamav3 standard_cal_data (contamination boundary preserved:
calibration corpus, NOT the eval suite). Research agent confirmed the operator's
memory precisely: brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians-Canonical (1.05M
tokens, hidden.bf16.bin router inputs + topk + derived H13 Hessians),
madeby561's hybrid quants, lukealonso's NVFP4 lineage. Stock exllamav3 ingests
tokens only (activate_all_experts Hessians) — our captures serve the custom
SQG/BMM-Law-style pipelines, allocation research, and the MLX adventure.
Bonus: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits (51,175 positions, fp32,
qualification-only) exists -> cross_check.py captures his exact 25 token
windows and compares our replayed distributions to his — independent-pipeline
cross-validation for pennies.

## 2026-08-27 00:21–00:47 — The smoke earns its keep (5 attempts)
Prep chain failures, each found at $0.44/h instead of $32/h:
1. **unpinned smoke model** — harness fail-closed check working as designed;
   fixed with revision.txt (and pinned in the script for posterity).
2. **collective_rpc msgspec refusal** — the real vLLM-version gap; the v0.28-era
   image won't serialize function objects without
   VLLM_ALLOW_INSECURE_SERIALIZATION=1. Added to both drun()s.
3. **my own bug**: the review-fix for qualify had swallowed the comparison
   guard — `for key in identity_keys:` straight into an unconditional raise.
   Diagnosed via in-container instrumentation after both identity dicts printed
   byte-identical. Guard restored. (The smoke caught the reviewer's fix's fix.)
4. **teardown SIGABRT**: qualify completes, writes its receipt, then the engine
   aborts during interpreter finalization (PyEval_SaveThread GIL bug). Fixed by
   receipt-verified tolerance everywhere: exit codes lie on this build,
   receipts don't.
5. **GREEN.** SMOKE_SUMMARY: captures byte-identical across engine loads;
   replay cap1-vs-cap2 KLD exactly 0.0; qualify floor 2.29e-3 bf16 / 2.04e-3
   fp32 on the 0.6B — fp32 barely moves it, so the floor is the engine's live
   logprobs path, not capture rounding; the FP8-vs-BF16 headline cancels it by
   construction (shared replay path). G3 gate on the real model converted to a
   supervisor HOLD instead of a hard death. ACT_SMOKE_GREEN (56 modules).
   fsbench: 1.3 GB/s from the filesystem.

## 2026-08-27 ~00:50 — PREP COMPLETE + top-off landed
All prep green; Brandon's teacher logits staged on the fs. Balance watcher
fired: **$699.41**. Both launch conditions met.

## 2026-08-27 00:57 — 8x H200 VM up (482877, $31.92/h)
8x NVIDIA H200 confirmed, fs mounted (947 GB of prep artifacts visible). SSH
slow to boot (operator's warning about flaky inbound SSH validated; also
learned macOS has no `timeout`). ControlMaster configured on the Mac (jl-h200 /
jl-prep aliases) — all subsequent connections instant. tmux `watch` session
with nvtop running for the operator: `ssh -t jl-h200 tmux attach -t watch`.
Bundle + HF token (mode 600, never in bundle/git) staged. apt update +
dist-upgrade with nvidia/kernel packages held (operator request; driver bump
mid-session would be fatal).

## 2026-08-27 01:05 — gen_check gate added (operator's idea)
Before any extraction, the engine must SPEAK: greedy completions on two raw
prompts + one chat-templated prompt, hard-fail on degenerate output, snippet
ntfy'd to the operator's phone ("GLM53 speaks"). Four independent evidence
legs now: machinery (L4 smoke), model sanity (gen_check), replay-represents-
served (qualify), independent pipeline agreement (Brandon cross-check).

## 2026-08-27 01:07–01:25 — vm_setup on the 8x: two more live-fire fixes
(1) docker load restores the image by ID but strips repo digests -> inspect by
digest failed; now runs by loaded image ID. (2) nonexistent python3.12-venv
poisoned the whole apt transaction on 22.04 -> per-package tolerant installs.
Attempt 3 in flight. Every fix committed to the repo as it happened.

## NEXT
vm_setup green -> heartbeat -> pipeline.sh (BUDGET_USD=450): restore 643 GB at
1.3 GB/s -> gen_check -> pace probe -> BF16 capture -> sentinels -> qualify ->
activations -> Brandon cross-check -> free_bf16 (numeric gates) -> FP8 leg ->
replay (the headline number) -> package -> publish to
malaiwah/glm53-flash-fidelity-suite-v1 + calibration-activations-v1 ->
deliverables home -> pause. Spend so far: prep ~$1.5, H200 ~$15.

## 2026-08-27 01:32 — vm_setup GREEN; PIPELINE LAUNCH
torch 2.13.0+cu130, 8x H200, glm5_next in registry inside the loaded image.
Heartbeat started; pipeline.sh launched with BUDGET_USD=450. L4 prep VM paused
(job done). From here the box drives itself; supervisor on 30-min cadence with
tight watch at phase transitions. Captain's-log format adopted at operator's
request: every entry now records status AND what to do better next time.

---

# Do better next time (running ledger)

1. **Smoke the exact image on a cheap box first — always.** Four real bugs
   (pinning, serialization, my own patch, teardown SIGABRT) cost ~$2 at L4
   prices instead of ~$100+ at 8x prices. Make it standing SOP, budget 1-2 h.
2. **Receipts over exit codes.** Day-one engines abort after success
   (finalization GIL bug). Design every stage receipt-verified from the start,
   not retrofitted.
3. **A fix needs its own verification.** My review-fix dropped a comparison
   guard; nothing re-checked the fix itself. Re-run the reviewer (or a unit
   probe) on every hand-applied patch.
4. **Pin everything at first sight, including throwaway models.** Day-one HF
   repos mutate hourly; the harness's fail-closed pinning is right — script
   revision.txt into every download path.
5. **docker save/load strips repo digests.** Pin by image ID when moving
   images through tarballs.
6. **apt transactions are all-or-nothing.** One nonexistent package name kills
   the install of everything else. Per-package tolerant loops for maybe-missing
   packages.
7. **Portability check helper scripts.** macOS has no `timeout`; a poller
   silently degraded. Test the watchdog before trusting the watchdog.
8. **JarvisLabs specifics:** 8-GPU VMs take minutes to accept SSH — build the
   wait in; inbound SSH is flaky (operator was right) so the pipeline must
   live ON the box with outbound ntfy; ControlMaster from minute one;
   filesystems are region-locked and resize rotates fs_id; check balance
   before planning anything.
9. **Shared-filesystem prep makes the expensive box stateless.** Downloads,
   image tarball, heads, cross-check data all pre-staged — the 8x went from
   create to pipeline in ~35 min, most of it boot + my own fixes.
10. **Start the captain's log at hour zero, not hour five.** Backfilling is
    lossy; the discipline is the point. (This ledger exists because the
    operator asked — next campaign it exists from entry one.)

## 2026-08-27 01:44 — Utilization review (operator's question)
Restore flowing at ~0.85 GB/s. Expected GPU utilization during capture legs:
10-20% BY DESIGN — max_num_seqs=1 sequential eager capture is the v5 protocol;
batching would multiply throughput but change bf16 numerics and break byte-
reproducibility + v5 parity. The 8x H200 is rented for its 1.13 TB VRAM (85%
utilized by BF16), not FLOPs. Optimized: prep off-meter, rank-0-only IPC,
FP8 pre-staged, piggybacked activations/cross-check. Declined (risk > reward
tonight): writer-thread overlap, dual-engine sharding, 4x downsize for FP8 leg.
Protocol purity costs ~$100-150 vs a hypothetical optimized harness; the
receipts are the product.
-> Next time (lesson 11): build a BATCHED capture mode for candidate-scoring
   campaigns — once the reference exists, scoring many quants is throughput-
   bound and the protocol can relax (document the numerics delta once,
   batch forever after).

## 2026-08-27 01:42 — GLM-5.3-Flash speaks; capture pace 10x better than planned
gen_check GREEN: "The capital of France is" -> Paris + landmarks; correct
iterative Fibonacci; coherent KL-divergence reasoning under Reasoning Effort:
Max (the model's first sanctioned thoughts were about the metric measuring it).
BF16 capture running at **~0.22 s/context (4.5 ctx/s)** vs the 0.5-2.5 s
planning band — full leg ~19 min, both legs ~40 min. Revised completion
estimate: ~04:30 UTC, total 8x cost ~$105-120 vs the $240-380 budgeted.
Engine loads are now the dominant cost, not capture.
-> Next time (lesson 12): benchmark one real capture context during the cheap
   smoke (load the big model once on the prep box? impossible at 24 GB — so
   accept the band, but tighten it with a published tok/s reference for the
   engine+hardware before writing cost projections).

## 2026-08-27 02:05–02:20 — Sentinel gate trips: glm5_next is not run-deterministic
BF16 capture completed (5,120 ctx, ~19 min as projected). Sentinel recapture:
**12/32 contexts NOT byte-identical across engine loads** — the first real
scientific finding of the night. The Qwen smoke was byte-identical on the same
stack, so this is glm5_next-specific (KDA/DSA kernels: atomics or top-k
tie-breaks suspected). Measured the run-to-run noise floor through the shared
head: **8.7e-4 mean KLD, top-1 0.9946, p999 0.072** over 65,504 positions
(the 20 identical sentinels contribute exactly 0). Decision per pre-encoded
rule (proceed if <=1e-3, documented): PROCEED — the FP8-vs-BF16 headline is a
paired comparison read against a published noise floor, exactly what v5's
sentinels are for when bitwise fails. Sentinel stage converted from byte-assert
to measured-noise gate; receipts ship both. Pipeline relaunched with resume
guards (capture/gen_check skip on existing receipts).
-> Next time (lesson 13): treat byte-determinism as a HYPOTHESIS per
   architecture, not an assumption — design the sentinel stage as
   measure-then-gate from day one, and try deterministic-kernel env knobs
   (a v2 rerun with them would tighten the floor).
