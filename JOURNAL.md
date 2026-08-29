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

## 2026-08-27 02:35 — Attribution correction + divergence investigation launched
Correction to the 02:05 entry: the byte-identical control (Qwen3-0.6B) ran at
TP=1 with no KDA, no DSA, no MoE — so "glm5_next-specific" was over-attributed.
The confounded variables: architecture kernels AND tensor-parallel AND model
scale. New ranked suspects for CROSS-LAUNCH divergence: (1) Triton @autotune on
the KDA/FLA chunk kernels — autotuning benchmarks per process, so two engine
loads can select different kernel configs -> different accumulation order;
(2) NCCL algorithm/protocol selection per communicator init at TP=8;
(3) cuBLAS/cuBLASLt heuristic algo selection per launch; (4) DSA indexer top-k
tie-breaks as an AMPLIFIER of upstream bf16 noise rather than a root cause.
Two agents dispatched: community-sightings sweep + source analysis of the
PR-branch kernels, with a deterministic-rerun env recipe as the deliverable.
-> Next time (lesson 14): controls must vary ONE thing — run the smoke model
   at the same TP as the subject (a 0.6B at TP=8 is silly but free, and it
   would have isolated TP from architecture tonight).

## 2026-08-27 02:50 — Divergence research: we are first; mechanism identified in lineage
Community sweep: NO prior report of run-to-run divergence for GLM-5.3-Flash —
our sentinel measurement is the first. But the kernel ancestry carries a
smoking gun: **fla-org/flash-linear-attention#945** — the chunked-delta-rule
forward kernel family (shared by KDA) returns bitwise-different outputs when
Triton autotune selects num_warps=4 configs (racy; num_warps=2 is clean), and
autotune picks configs PER PROCESS from timing benchmarks -> different config
per engine load -> bitwise-different but internally-consistent numerics per
load. That is EXACTLY our signature (stable within load, 12/32 divergent
across loads). Reinforced by triton#9368 (autotune cache does not restore
cross-restart bitwise determinism). Also learned: vLLM's VLLM_BATCH_INVARIANT
hard-fails on KDA/GDN models (#42960) — deterministic mode is structurally
unavailable for glm5_next — but v0.28's own override_envs_for_invariance()
env set (NCCL algo/proto/channel pins, CUBLAS_WORKSPACE_CONFIG, custom-AR off)
applies standalone. DSA stack has its own open nondeterminism issue (#53257,
concurrency-driven, not our batch-1 case).
Verdict on the operator's question: PLAUSIBLE (mechanism matches signature),
PARTIALLY FIXABLE tonight (env pin Set 1, single-digit % cost), FULLY fixable
with a one-config pin of the vendored fla autotune lists (num_warps=2).
v2 deterministic sentinel probe queued post-session with
TRITON_PRINT_AUTOTUNING=1 + NCCL_DEBUG=INFO to catch the mechanism red-handed.

## 2026-08-27 02:55 — Root cause converges; qualify lands at 1.49e-2
Source dossier (PR-branch code): KDA prefill is 100% Triton (FlashKDA ext not
even called); ~9 autotuned kernels chain per prefill, two provably numerics-
changing (chunk_gla_fwd_kernel_o: 36 configs, BK splits an fp32 reduction;
scaled_dot_kkt: 24 configs). Winners chosen per-process by timing benchmark,
8 TP ranks tune independently -> per-launch winner vector; frozen within a
load. Free diagnostic run on tonight's four engine loads: all-reduce backend
dispatch IDENTICAL every load -> backend-flip excluded; **Triton autotune is
the root cause, DSA top-k reselection the amplifier**. Fix for v2: 
TRITON_CACHE_AUTOTUNING=1 + persistent TRITON_CACHE_DIR (~0% steady-state) or
single-config patch of the vendored fla kernels. NOT applied tonight: FP8 leg
must run the same env as the BF16 leg (paired comparison integrity).
QUALIFY-BF16: mean KLD(live||replayed) = 1.49e-2, top-1 0.957 — 17x the
sentinel floor, because the live pass is a THIRD engine load measured through
the model (autotune variance amplified by indexer top-k membership flips).
Interpretation: replay-vs-served is bounded by ~1.5e-2 on this runtime; the
paired FP8-vs-BF16 headline carries only the 8.7e-4 capture floor. free_bf16
will HOLD as designed; release decision awaits the Brandon cross-check
(independent fp32 teacher = external replay validation).
-> Next time (lesson 15): on a new architecture, run the sentinel pass BEFORE
   the main capture (cheap early warning) and log TRITON_PRINT_AUTOTUNING=1
   from load one — tonight's forensics would have been one grep.

## 2026-08-27 03:10 — Evidence published upstream + card section wired
Nondeterminism dossier posted on the vLLM PR:
https://github.com/vllm-project/vllm/pull/53906#issuecomment-5433635837
(365 words: signature, magnitude, exclusions, root cause with code specifics,
lineage links, mitigations, offer to confirm). Dataset card generator now
carries a "Known issue: run-to-run nondeterminism (first report)" section with
the receipts, the paired-vs-absolute interpretation, and the v2 deterministic
recipe — the finding ships with the data, reproducibly.

## 2026-08-27 03:35 — Self-inflicted halt: torn read of a live script (data unharmed)
Activations stage "failed" rc=2 — but the capture itself was PERFECT (92/92
contexts, 147 GB, 478 s). Root cause: supervisor error — I scp'd stage.sh over
the same inode while the pipeline's bash was executing it; bash reads scripts
incrementally by byte offset and hit a torn old/new hybrid ("syntax error near
card2"). Every edit had passed bash -n locally; the file was fine, the RUNNING
READER wasn't. The prep phase's bundle.new atomic swap existed precisely for
this and I bypassed it on the 8x all night without consequence — until now.
Fixed: all script syncs now scp-to-.new + mv (new inode; running bash keeps
its old fd). Pipeline launch 3 (r_d592c1bc) with full resume guards — resumes
at cross_check with zero recompute lost. Cost of the lesson: ~25 idle minutes,
~$13.
-> Next time (lesson 16): NEVER overwrite a script a live shell may be
   executing — atomic rename only, from the first sync of the campaign. The
   rule existed in prep; carry it everywhere.

## 2026-08-27 04:20 — v1 PUBLIC; cross-pipeline validation lands perfectly
**https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1 is
live** — first quality reference for the model, day one, with suite, BF16
shard-0, shared head, and every receipt. Cross-check v2 (hash-verified pairing
against brandonmusic's independent fp32 teacher): **mean KLD 1.27e-2, top-1
0.9665, offset audit 0.966@0 vs 0.016@±1** (alignment perfect). The three
noise numbers nest exactly as the autotune mechanism predicts:
8.7e-4 (same-pipeline recapture) < 1.27e-2 (cross-pipeline) ~ 1.49e-2
(live-vs-replay) — the runtime's launch variance explains everything; both
pipelines exonerated. Also: second unpinned launch-pair measured 25/32
identical (vs 20/32) — winner-lottery rate wobbles as predicted. Shim v1
was shadowed by Ubuntu's own /usr/lib/python3.12/sitecustomize.py; PYTHONPATH
delivery verified, pinned det_kpatch v2 running. Next: release HOLD -> v1 FP8
leg -> replay -> repo update; then pinned v2 recaptures per operator mandate.
-> Next time (lesson 17): sitecustomize is shadowed by distro copies — deliver
   interpreter shims via PYTHONPATH dir, and assert the shim's banner in logs
   as part of the experiment's validity check (we did; it caught the miss).

## 2026-08-27 04:45 — Intervention test FALSIFIES single-cause autotune story
det_kpatch v2 ran with the shim verifiably active in every process (11 banner
prints: main + workers) — and launches STILL diverge: 20/32 byte-identical,
same rate as unpinned pairs. The autotune winner lottery is at most a partial
contributor. Promoted suspects: NVLS/symm_mem in-switch reduction internals
per communicator init; cuBLAS heuristic selection. Next intervention (queued
behind the v1 FP8 leg): stacked pins — shim + VLLM_ALLREDUCE_USE_SYMM_MEM=0 +
NCCL algo/proto/channel pins + CUBLAS_WORKSPACE_CONFIG. The upstream follow-up
will carry this correction; a falsified mechanism reported honestly is worth
more than a defended one. HOLD released (evidence-based gates green): v1 FP8
leg now running.
-> Next time (lesson 18): circumstantial code-reading converges fast but only
   an intervention test settles causation — budget for the intervention pair
   from the start, and never publish "root cause" before it (our PR comment
   said "likely root cause" — the hedge just earned its keep).

## 2026-08-27 05:05 — THE HEADLINE: official FP8 costs 0.0281 nats vs BF16
Replay complete over all 10,480,640 positions:
**FP8-vs-BF16 mean KLD 0.028104 nats** (macro mean 0.028104, CI95
[0.027205, 0.028982]), median 4.93e-3, p99 0.354, p999 1.374 (heavy tail),
top-1 0.9427, JSD 0.0092 bits. Per-stratum 0.0223 (encyclopedic) to 0.0354
(scientific). llama.cpp-comparable geometry (positions 1024+): 0.0188 /
top-1 0.9512. Signal-to-noise-floor 32x (vs 8.7e-4) — clean measurement.
Comparative note: Qwen3.8-27B's official FP8 measured 0.0053 on the same
protocol family — zai's block-FP8 costs ~5x more divergence against its own
BF16. qualify_fp8 root-caused (comparator OOM: FP8 engine pads KV to the util
target, GPU0 at 136 GiB); retry queued at util 0.80. Package+publish of the
completed v1 running. Remaining: qualify-fp8 receipt, stacked-pin
intervention, pinned v2 recaptures, activations publish, closeout.

## 2026-08-27 08:20 — PR-thread hint + two-step extraction (operator's calls)
Operator's suggestions, both right: (1) PR thread re-read — Zek-Takai reports
DeepGEMM's JIT compiles GLM's mHC hyper-connection kernel and is sensitive to
CUDA package mixes. If the mHC JIT is not gated by VLLM_USE_DEEP_GEMM, our
DeepGEMM-off peel never touched it — **new prime candidate for the residual
1/32**; first ablation for any follow-up session. (2) Two-step extraction:
direct VM->Mac download (residential speed, meter running) replaced by
VM -> private HF scratch (datacenter pipe, minutes) -> Mac pulls off-meter.
Full deliverables + intervention logs uploading to
malaiwah/glm53-session-scratch (private); VM pauses on completion.
-> Next time (lesson 19): egress via the fastest pipe FIRST (HF/S3), pause the
   meter, THEN sync to slow endpoints; and re-read active upstream threads
   before closing an investigation — fresh eyes drop hints hourly on day-one
   PRs.

## 2026-08-27 ~10:30 — Final science: drift certified, KV question closed, endgame ledger
Guarded+stacked pair: 28/32 — the OOB guard does not drive the residual either.
Final intervention ledger (all by experiment): autotune pin = no effect;
collective/cuBLAS pins = ~10x flip-rate reduction; DeepGEMM env = untestable
(model calls DeepGEMM directly, ungated); OOB guard = no additional effect.
Residual unidentified; surviving suspects: indexer/mHC DeepGEMM JIT.
Drift/config-sensitivity: pinned+guarded captures vs day-one reference =
1.32e-2 / top-1 0.964 — the FP8 headline (2.81e-2) stands 2.2x above the full
config envelope and 32x above same-config noise. v1 certified.
KV matrix (operator's last-minute ask): FP8 KV REFUSES INIT on Hopper for both
weight variants — exact assert "pe_dim must be 64 for fp8_ds_mla"
(cache_kernels.cu:866) — empirically confirming the recipe's Blackwell-only
note; the NoPE arch breaks fp8 MLA cache writes off-B200. NVFP4 KV does not
exist in vLLM. Receipts written for both refusals.
Also extracted per operator's last call: per-tensor stats sweep (38,770 BF16 +
FP8 tensors: norms/absmax/row-col spreads) for future EXL3/MLX bit-allocation
design, and the complete FP8 weight_scale_inv map (zai's quantization recipe,
~80 MB) — the key to explaining WHERE the 0.0281 lives.

## 2026-08-27 ~12:30 — Community handshake + K6 program pivot
Lab daily-summary intel: brandonmusic shipped the FIRST EXL3 of glm5_next
(4bpw, KLD 0.0245 on his sealed windows, five bitwise-identical cold runs on
his transformers TP2 stack — deterministic where vLLM is not) with the full
pipeline PUBLIC; a working SM120 image exists (chriswritescode-dev) — the
RTX-6000-Pro path is alive; the "DERISKED" NVFP4 was confirmed grift.
Posted discussion #1 on his quant page: cross-stack validation numbers, all
links, and the co-credited proposal (his 4bpw scored on our 10.48M suite +
a K6 via his pipeline). K6-on-AIBeast math: K6 weights ~246 GiB on TP4's
384 GiB leaves ~106 GiB; the 11-MLA/34-linear design needs only ~3.3 GiB of
fp8 KV for 512K context — K6 fits with ~30x margin; K5 unnecessary; fp8 KV is
the Blackwell-native path. Forge (spot, preempted once and resumed) staging
Brandon's pipeline + pinned exllamav3 + SM120 ref; port workflow continues as
the stock-ecosystem track.

## 2026-08-27 ~13:40 — Same-panel verdict: FP8 0.0206 vs 4bpw 0.0246; comment posted
The settling experiment landed: official FP8 replayed over Brandon's 25 sealed
windows (51,175 positions, token ids sha-verified) scores KLD(teacher‖FP8) =
0.020615 / top-1 95.63%, offset audit clean (0.956@0 vs 0.016@±1), per-window
0.0096–0.0457. Verdict on HIS yardstick: FP8 edges the 4bpw (0.0246) — "beats
FP8" not supported — but our FP8 row is biased UP by the 0.0127 cross-stack
floor, so the true gap is larger; 4bpw within ~1.2x of FP8 at 176 GB vs 328 GB
remains a strong showing, and the K6 thesis (well under FP8 at ~246 GiB) is
intact. Cross-suite reversal explained: his panel is FP8-friendlier than our
10.48M mix (0.0206 vs 0.0281) — same-suite comparison was the right call.
Follow-up comment posted on his discussion #1 (2026-08-27T11:36Z) with the
three-row table + receipts; baseline receipt crosscheck-brandonmusic.json
added to public reports/ (commit 5daa6c52).
Port design bundle persisted + pushed (port/ in this repo): 7-agent workflow
delivered blueprint (exl3 v1.4.4 already has ~80%: dsv4 mHC verified
numerically identical to vLLM's, glm_moe_dsa skeleton, GDN cache machinery;
new: KimiDeltaAttention, kpool indexer mode, NoPE guards, mean ContractStreams,
sigmoid GatedRMSNorm), syntax-checked draft arch, smoke-tested parity harness,
adversarial review (1 blocker: fla version floor could silently accept a fla
without the SAFE gate — **kwargs swallow safe_gate; pin + assert at import).
Rehearsal vehicle found: inference-optimization/GLM-5.3-Flash-0.1B-A0.1B tiny
fixture is architecturally COMPLETE (KDA+DSA+kpool, dense→sparse @3, mHC,
NoPE, vision, full 154,880 vocab, exact model.language_model + hc_*_fn tensor
naming) — stock-transformers-loadable, so it gives the port a true
cross-implementation oracle at toy scale, something our synthetic mini-ckpt
can't. Caveats: random weights (no quality signal), F32 dtype (won't exercise
bf16-load patch), no MTP tensors, tiny dims miss real kernel shape paths; use
T≥512 so kpool selection actually engages (kpool 4 × topk 64 = 256).
MLX prior art (orcarouter/GLM-5.3-Flash-MLX, OrcaSAQ calibration-free mixed
precision): their own numbers kill the Mac path — only 2bit-lite (102 GB) fits
128 GB and it's 0.346 nats / 77% top-1. Operator call: Mac Metal/MLX DROPPED
for this model; keep their allocation policy as convergent prior art for the
AIBeast multi-precision EXL3 (down_proj +1, shared experts +2, never-FP8 set
stays BF16 — their quantized set = exactly the 37,338 scale_inv tensors we
extracted). Their 4-bit vs-FP8-reference KLD 0.0131 is NOT comparable to
Brandon's 0.0246 vs-BF16-teacher (different reference, suite, stack).
Upstream check: both zai repos' 1h-ago pushes were README-only (diffed trees);
pins unaffected; chat-template notes irrelevant to teacher-forced replay.
LESSON 20: capture contracts should embed upstream repo+revision, not just
container paths — the pins lived only in the card/download receipts this time.
LESSON 21: when two quants are compared across different suites/references and
the ordering matters, run the same-panel experiment before repeating the
claim — the cross-suite ordering REVERSED on the shared yardstick.

## 2026-08-27 ~13:55 — CAMPAIGN CLOSED: H200 paused, freight verified, ~$352 spent
482877 (8x H200) confirmed Paused after freight verification: private scratch
holds captures-bf16-full + captures-fp8-full (5,121 files / 85.9 GB each),
crosscheck-suite, jl-run-logs, deliverables, detpin — 16,466 files / 190 GB.
Nothing of value lives only on the VM. fs 3394 (2 TB, IN2) remains the second
copy (checkpoints + captures + activations) pending operator retention call.
Cost: balance $201 → topped to ~$701 → $349.57 remaining ≈ $352 all-in for
the campaign (~10 h of 8x H200 dominates). Zero instances running; paused
storage (H200 1.2 TB, forge 1.2 TB, L4 100 GB, fs 2 TB) still bills.
Open watches: forge 483634 stuck "Resuming" (destroy+recreate at K6 session
if unchanged); Brandon discussion/dataset watch.
Morning to-dos handed to operator: (1) ROTATE the HF write token pasted in
chat — top priority; (2) fs 3394: keep until the K6 session (saves ~970 GB of
re-downloads), destroy after; (3) qwen38 VM 481678 untouched by choice;
(4) remove the glm53-session block from ~/.ssh/config once K6 work concludes.
LESSON 22: instance "cost" in jl get is the live hourly rate, not cumulative
spend — reconstruct campaign cost from balance snapshots; log the balance at
every phase boundary so the accounting is one grep away.

## 2026-08-27 ~14:30 — K6 PROGRAM LAUNCHED (operator greenlight: full autonomy)
Mission: first K6-uniform + K6K8-mixed EXL3/TR3-MCG quants of GLM-5.3-Flash
via brandonmusic's pipeline; score on his 25 sealed windows (targets to beat:
FP8 0.0206, his 4bpw 0.0246); publish weights+receipts+tools. K6K8 fit math:
routed gate/up 203B @6bpw + down 101.5B @8bpw + ~34GB native ≈ 288 GB ≈
268 GiB → 67 GiB/GPU on AIBeast TP4, ~29 GiB/GPU headroom, KV 512K ≈ 3.3 GiB
— AMPLE. Recon rewrote the infra plan: (1) Brandon's ENTIRE pipeline ships in
his 4bpw repo (runtime/src/quant_pipeline, 72 files — campaign runners,
global_dp allocator, MCG codecs, materializer, qualify scripts); his k4 recipe
is schema-versioned with routed_expert_bits parameterized, global allocator
present-but-unused (exactly what K6K8 needs); TP target is a materialization
parameter (his: TP2; ours: TP4). (2) GPU market: PRO6000 effectively ONE
device (IN1, spot) — dropped; IN2 has 8x H200 ($1.99 spot) + 8x H100 VM
($1.19 spot) AND fs 3394 is IN2 → conversion reads BF16 from fs, no 643GB
re-download; Blackwell only needed for AIBeast serving, not convert/score.
(3) Old forge 483634 already reclaimed by platform (stuck-Resuming limbo died
on its own) — clean slate. In flight: 6-agent design workflow (4x source
anatomy → runbook+stage-scripts synthesis → adversarial pre-spend review,
last night's winning pattern) + L4 prep box resumed (482867→484453, $0.44/hr)
running env smoke: fs free space, transformers 5.16.1 + fixture forward,
quant_pipeline imports, exllamav3 @ c5d9c657 build feasibility on a bare VM.
Budget: $349.57; program estimate ~$100-150 all-in.

## 2026-08-27 ~17:30 — G0: design GO_WITH_FIXES, closure hunt, engineering fan-out
Design workflow verdict GO_WITH_FIXES (reviewer fixed 9 defects in place: venv
torchrun, disk ledger, qualify-gated publish, setup guards, input asserts,
byte-count assert, license inheritance). Cost: K6 leg $274 w/30% margin ≤ $349
GO; K6K8 add-on $143 gated on ≥$140 after K6. Sizes receipt-exact: K6
253,536,370,680 B (236.1 GiB), K6K8 279.5 GB (260.3 GiB) — both fit TP4.
G0 fetches rewrote the plan again: his GITHUB repo (brandonmmusic-max/
glm-5.3-flash-exl3-4bpw) is richer than the HF mirror — ships the 7 KLD/
runtime driver scripts AND glm53_uniform_k6.py AND bits-parameterized
preparation/backend (patches 0003/0004/0005 dissolve). shapleymcg repo public
with rev 9d83e7d0 present, run_qwen_fast_encode.py sha MATCHES his seal;
bmmlaw_r7_encoder package found in glm52-sqg-mcg-experiments. Still missing
everywhere public: r7_encoder/r10_codec.py + encode_tr3_v31.py (the sealed
numeric core) → filed github issue #1 on his code repo asking to publish
(fallback: disclosed reconstruction around exllamav3's own trellis ops,
designed in parallel, operator-gated). His campaign attests 4x B200 SM100;
we run H200 SM90 as a disclosed deviation (fat 9.0;10.0 ext build satisfies
the capability check honestly; worker-slot patch discloses the rest).
Calibration: his published captures reusable (~475 GB download, 4 independent
final-window contamination guards) — no self-capture (EP4 hard-pinned,
184.8 GiB/rank > H200). L4 smokes 1-5 delivered the proven env recipe
(torch 2.11.0+cu130 / fa 2.8.3 cu13torch2.10 wheel / formatron 0.5.0 +
pydantic 2.5.3 / exllamav3 ext GREEN on SM89+CUDA13; fixture forward works
stock on torch 2.11 — the scatter bug was torch-2.6-era). G0 engineering
workflow launched (patch rebase onto GitHub base + 4 driver tools adapted
from his scripts + fallback codec + adversarial re-review). Paid P0 waits on
that verdict + closure resolution.
LESSON 23: the HF mirror of a pipeline is not the pipeline — check the
author's GitHub before writing patches; three of seven dissolved on fetch.
LESSON 24: sealed "external closure" deps (content-hash-pinned local files)
are the real long pole — hunt them across ALL the author's repos before
designing reconstruction, and just ask the author early.

## 2026-08-27 ~19:00 — Reconstruction ACTIVATED (operator), publication sweep
Operator decisions: (1) keep Brandon's calibration+teacher for this program
(comparability is the product; our activations dataset stays the base for
future native-exl3/MLX paths; K6 also gets scored on OUR 10.48M suite as a
second yardstick); (2) reconstruction ACCEPTED — RECONSTRUCTION-ACCEPTED.json
authored on verbatim operator instruction, fallback staged to fs, issue #1
updated transparently with the public code link; (3) "publish all our work"
→ k6/ bundle pushed to the repo, front-page README written (repo had NONE —
findability hole), HF card gained a related-work index. Prestage download
(~505 GB calibration+teacher → fs) running on the L4.
Public map now: GitHub repo (tools/remote/k6/port/JOURNAL+README), HF
fidelity dataset + activations dataset, vLLM PR comments, HF discussion #1,
GitHub issue #1. P0 rental launches when prestage lands.

## 2026-08-27 ~21:30 — P0 GREEN: encode projected 2.2h (was 14h planned); P1 launched
P0 rehearsal on 1x H200 (484789) took five setup attempts, each stopped by a
designed guard: (1) template python 3.10, (2) container CUDA 12.6 can't emit
sm_100, (3) missing pydantic/formatron/kbnf + flash-attn in the setup dep set,
(4) pipefail silent-exit on a find over a nonexistent torch_extensions dir,
(5) fixture not staged at $ROOT/fixture/<name>. All five fixes are now IN the
stage script (incl. container self-bootstrap: deadsnakes py3.12 + CUDA 13.0),
pushed public. Verdict: closure gate reconstruction OK (5 staged files),
k6_roundtrip_exact=true, bench 0.84 s/full-size-matrix K6 → projected
main+MTP encode 2.16 h on 4 GPUs — 6.5x under plan, 11x under the abort
gate; P1 encode cost collapses ~$111 → ~$20. K8 probe red AT THE ADAPTER
(codec-side K8 proven on L4; declared-extension patch = the P2 work item;
K6K8 descoped until it lands, exactly per runbook). Operator supervision
directive in force: 10-min watchdog caught the idle box within 30 min of the
pydantic failure (~$1 idle cost). P0 box destroyed; P1 fleet 484853
(4x H200 spot IN2, fs attached) created; chain running: self-bootstrap setup
→ shared_vector_ab (down_suh A/B, operator directive) → convert_k6.
LESSON 25: guards that fail fast are cheap; the expensive failure is the one
that exits SILENTLY — audit every `cmd | tee` under set -euo pipefail.
LESSON 26: chain launchers on `jl run status --json`, not log-footer greps —
a footer grep zombie nearly double-launched a stage.

## 2026-08-27 ~19:50 — Brandon v44 drop recalibrates the FP8 bar
His new commit (0b2f8fea) publishes SM120 TP2 runtime + qualification: WITHIN
his stack, FP8-as-served = 0.02463 mean KLD / top-1 93.8% (5 runs, 2,047
positions each) vs his EXL3 4bpw 0.02455 — A WASH at 54% of the bytes; NVFP4
= 0.0605 / 91.5% (2.5x worse, bitwise-deterministic). Also: 500k needle
tests + decode/prefill benchmarks (dcp2+mtp3) for the SM120 serving stack —
directly relevant to AIBeast. Note his v44 KLD set is a single window, not
the 25-window panel (our cross-stack FP8 0.0206 was full-panel; different
position sets). No closure files pushed; issue #1 unanswered; disclosed
reconstruction remains the campaign path. K6 target: land well under
0.0246-class FP8 on his panel. Campaign state: GSS prep parallelized to all
4 GPUs (contract loop + 3 prepare workers on disjoint ranges) after operator
spotted GPU0-only; prep ~10/42 at parallelization.

## 2026-08-28 ~02:45 — K6 MATERIALIZED: byte-exact, world's first
convert_k6 completed at 02:38:21Z after the encode (42/42 layers, all-worker),
main receipt c65c162b, MTP adapter receipt 1159d61a, calibration deleted per
ledger, checkpoint materialized: output_logical_bytes 253,536,370,680 —
EXACTLY the pre-campaign derivation. Receipt: bits 6, complete,
main_and_mtp_complete, nonrouted_native_exact all true, qualified_tp_sizes []
+ serving_reader_qualified false (topology-neutral form as demanded).
THREE-LANE FAN-OUT: (1) convert_k8 launched on the 4x fleet (calibration
re-download + eviction guards + --overlap-seal per DECISIONS; A/B completes
from receipts vs the banked serial control unit); (2) 8x H200 VM 485017
rented ON-DEMAND (container spot exhausted by our own fleet) for qualify_k6
— 5 cold runs on the sealed panel, ~$65 premium accepted for the morning
publication timestamp; (3) L4 freight box (484453→485016) uploading the
254 GB checkpoint PRIVATE to malaiwah/GLM-5.3-Flash-TR3-6bpw
(QP_PUBLISH_UNQUALIFIED staging; flips public only on green panel receipt).
Overlap-smoke postscript: the disk-ledger calibration deletion preempted the
A/B's overlap leg (control banked); flag enabled on verified correctness,
gain measured from campaign receipts instead. TR3 naming live everywhere.

## 2026-08-28 ~04:50 — Overnight contract (operator handing off for the night)
Operator: keep K8 only under close supervision; unattended autonomy granted.
CONTRACT (armed as overnight_supervisor.sh + 2-min ntfy reporter):
process-level checks every 5 min (stack dumps over exit codes); K8 abort rule
— if payload store <1GB by 06:30 UTC, pause fleet, park K8 for spot tomorrow;
budget guard at $150 (pause all but K6 publication); idle-box guard; K6 lane
completes autonomously (qualify → card → public flip → receipts → discussion
post). Night state at handoff: K6 weights private on HF (259 files);
qualify take-3 in receipt-walk (~39/44); K8 contract take-5 prepping with
gated worker chain. Friction ledger tonight, all mine, all fixed+pushed:
CUDA_VISIBLE_DEVICES-vs-preflight (twice), nice-env ordering, prep/contract
doc race, symlink-farm machine-locality, port collisions, VM sudo bootstrap,
ext rebuild clobber. The K8 path from here reuses the exact chain K6 proved.

## 2026-08-28 ~05:30 — K6 PUBLIC (operator call on the preview strength)
Preview (run 1/5, window-0000, unofficial): KLD 0.0168 / top-1 95.5% vs FP8
0.0265 on the SAME window — 1.6x better at 77% of FP8's bytes. Operator:
publish now, update card after the aggregate. Done:
malaiwah/GLM-5.3-Flash-TR3-6bpw public with provisional-flagged card
(TR3 naming, codec-vs-runtime, provenance + disclosed deviations, family
table, co-credits). Qualify runs 2-5 continue; card + discussion update on
the sealed aggregate. K8 prep continues on the fleet in parallel.

## 2026-08-28 ~09:20 — K6 SEALED: 0.013723 nats, five bitwise-identical runs
The headline the campaign was built for: mean KLD(teacher‖K6) = 0.013723
nats over the full sealed panel (25 windows × 51,175 positions × 5 cold
runs, population stddev EXACTLY 0.0 — the determinism property transfers to
our stack). Quality gate passed. 1.5× better than official FP8 at 77% of its
bytes; 1.8× better than the 4bpw; 4.4× better than NVFP4. Card updated with
receipts; reports in the fidelity suite; final table posted on the
collaboration thread. TP-runtime serving smoke disclosed as not-run (SM90 box
vs SM120 kernels; serving validation = AIBeast). 8× VM destroyed on
completion. BUDGET DRAMA: balance hit $24 (the VM's overnight qualify burn);
supervisor guard fought the operator's K8-must-finish directive — supervisor
stopped, K8 encode racing the wire (~$16 needed, 16/42 layers at the check).
Every number above is free-published; only K8's tail is money-gated.

## 2026-08-28 ~13:00 — K8 MATERIALIZED + Q4 base measurement sealed
K8: 331,449,761,784 bytes (308.7 GiB), bits 8, complete, main_and_mtp_complete,
qualified_tp_sizes [] — the parts-bin sibling exists. Uploading private to
malaiwah/GLM-5.3-Flash-TR3-8bpw from the L4 freight box; 4x fleet paused.
Patch 0011 was needed: build_materialization_plan had a THIRD MTP-schema
ternary 0007 missed, so K8 rejected its own valid receipt as "foreign".
Q4 (0xSero/Dione) SEALED on our panel: 0.027262784814670614, 5 cold runs
bitwise identical, 187.6 GB, receipt published to the fidelity suite and a
base-measurement discussion opened on their model page. LADDER (same panel,
teacher, reader): K6 0.013723 (254GB) < FP8 0.020615 (328GB) < 4bpw 0.024555
(176GB) < Dione Q4 0.027263 (188GB) < NVFP4 0.060535. Headline finding:
brandonmusic's ShapleyMCG pipeline beats Dione's calibration-free selective
map by ~11% at the same nominal rate and 12 GB less — a clean
pipeline-vs-pipeline result at fixed bit-width.
LESSON 27: hash CONTENT not CONTAINERS. Two false "nondeterminism" alarms in
one hour: capture receipts embed elapsed_seconds; safetensors embed
__metadata__ (cold_run, backend identity). Tensor bytes proved Q4 bit-exact
(max_abs_diff 0.0 over 2047x154,880 logits).
LESSON 28: single-window extrapolation does NOT transfer across quantizers.
Window-0000 ran 1.22-1.28x HARDER than the panel for FP8/K6 but EASIER for
Q4 (0.0256 vs 0.0273 panel) — my ~0.020 preview extrapolation was wrong by
36%. Previews are fine; label them, and never let one stand in for the panel.

## 2026-08-28 ~17:40 — The K8 "anomaly" was an underpowered comparison
Single-window (final-0000) streaming numbers said K8 0.018200 > K6 0.016829 —
an 8-bit quant apparently worse than 6-bit. Investigation (lane-matched, so
not a harness artifact) excluded, with evidence: wrong payloads (plan.json
bits 8 / packed_root out-k8); mixed extension binaries (all 43 preps one hash
per campaign); encoder correctness (120/120 byte-identical vs brandonmusic's
sealed core); reader K8 decode (BITWISE identical to exllamav3 native across 6
payloads, both transform stages, K6 controls clean); profile mismatch (K6 and
K8 used a BYTE-IDENTICAL default profile); scope (receipts match field for
field: 1618 native tensors, 37152 routed choices, nonrouted_native_exact both).
The weight-space test was blocked by cos(|W_hat|,|W|) = 0.633 ~ 2/pi — the
signature of a PERMUTATION, not noise. Confirmed: an INTERMEDIATE-CHANNEL
permutation (the expert-MLP symmetry), recovered empirically as a perfect
bijection (2048/2048, mean cosine 0.9998, zero identity matches), consistent
across gate/up/down within an expert, and identical between K6 and K8 (shared
transform seed). Serving is unaffected — the permutation is self-consistent.
Unpermuted, the SHIPPED stores say what they should: rel Frobenius K6 0.021490
vs K8 0.005916, NMSE 4.624e-4 vs 3.505e-5 — K8 13.2x tighter, better in 30/30
matrices.
THE REAL EXPLANATION: per-window KLD sd is 1.73e-3 against a K6-vs-K8 effect
of 1.22e-3 — a single window has NO POWER to separate two rates. Pooled over
the 11 windows both runs had captured: K6 0.013873 vs K8 0.012655, K8 winning
9 of 11 windows, top-1 96.34% vs 96.12%. window-0000 was an unlucky draw.
LESSON 29: NEVER quote a single-window KLD as a rate comparison. The noise
between windows exceeds the effect between adjacent bit-widths. Previews are
fine for "is the pipeline alive"; they are not evidence about which quant is
better. (Lesson 28 said extrapolation doesn't transfer; this says why,
quantitatively.) Corollary for the registry: single-window panels belong in
their own comparability group — which they already do, now empirically
justified.
LESSON 30: a cos(|a|,|b|) ~ 2/pi with matching sorted spectra means PERMUTED,
not broken. Weight-space audits of this pipeline must undo the
intermediate-channel permutation first or they will lose a day.

## 2026-08-28 ~20:00 — Second disk-full: measurement logits were not in the ledger
Both cold run 2s died with SafetensorError "Disk quota exceeded" and sat idle
36 min (~$2.4 wasted) before the check caught it. Cause: the fs ledger was
written for the ENCODE campaign and never accounted for MEASUREMENT output —
each streaming cold run writes ~32-44 GB of fp32 student logits, and two
concurrent campaigns on one 2 TB fs left 1 GB free. Freed 456 GB by deleting
what was already published or re-downloadable: ckpt-k8 (309 GB, uploaded to
HF), glm53/activations (147 GB, published as the activations dataset),
glm53/image (docker tarball), calibration/mtp45-ep4-full (encode-only).
Note the fs still carries TWO campaign trees (glm53 787 GB from the overnight
capture run, glm53-k6 1261 GB) — the old one is now down to models/bf16
(599 GB, still needed by the streaming scorer for non-routed tensors) plus
crosscheck receipts.
LESSON 31: the disk ledger must cover the MEASUREMENT phase, not just encode.
Rule of thumb per streaming panel run: positions x vocab x 4 bytes = 51,175 x
154,880 x 4 = 31.7 GB per cold run, per model, and runs are kept for the
determinism check. Two models x two runs = ~127 GB that no encode-era ledger
predicted.
LESSON 32: a failed jl run leaves the box IDLE but RUNNING. Exit-code watchers
catch it; window-count watchers do not (the count simply stops advancing and
looks like slow progress). Watch the run STATE, not just its output.

## 2026-08-29 ~00:15 — K6 streaming lane SEALED; the cheap lane is validated
stream_mean_kld 0.013714888822596553 vs sealed 0.013723384665701147 —
delta -8.4958e-06 (0.06%), worst single window 2.87e-4. cold_runs 2,
cross_run_payload_bitwise_identical TRUE (the determinism property transfers
from the 8xH200 sealed lane to a single GPU), quality_gate_passed TRUE. The
tool correctly refuses to overclaim: tokenwise_kld_sha256_matches_sealed FALSE
and publishable_as_reproduction FALSE, because a different expert-combine
order is an INDEPENDENT measurement that agrees closely, not a bitwise
reproduction. Cost: ~$6/model vs ~$50 for the sealed 8xH200 protocol.
K6 box destroyed on completion (receipts live on the shared fs).

## 2026-08-29 ~05:25 — The BF16 FLOOR of the streaming lane: 0.011506
`stream_score.py --source native --profile native-bf16` — the identical
streaming capture with the 36,288 routed expert matrices read straight out of
the official BF16 checkpoint by their released tensor names, no codec in the
path — scores **0.011505922619330299** nats on the sealed 25-window panel
(51,175 positions, fp64, teacher receipt 2ae08117…). Same panel, same teacher,
same estimator, same non-routed view, same EP8 emulation, same
`--reduce-order fp32`, same `grouped_mm` kernel, same H200 spot box class as
K6 and K8. The only difference is where the expert weights come from.

What it does to the story:

    student   panel mean     floor          quantization-attributable   floor share
    K6        0.013714889    0.011505923     0.002208966                 83.9%
    K8        0.012384191    0.011505923     0.000878268                 92.9%

K6/K8 is **1.107x on the raw panel mean and 2.515x on quantization-attributable
error** — K8 removes 60.2% of K6's quantization error. That is the number that
belongs next to "K8's shipped store is 13.2x tighter in weight-space NMSE";
"11% better" was never a statement about the codec, it was mostly a statement
about the lane. Note the floor is LANE-specific: the independently measured
cross-stack floor on a different lane was 0.012712, above ours.

Peak device memory 47.08 GB — byte-for-byte the K6/K8 figure, which is the
cheapest possible evidence that nothing about the schedule changed.

Validation before spending: L1 ladder a–f green, including a NEW rung **L1.f**
that proves `NativeCheckpointSource` + `fuse_gate_up` rebuild the stacked
expert parameters transformers' own checkpoint conversion produces, BITWISE, on
the 0.1B fixture (16 experts, max_abs delta 0.0). Negative controls: `--source
native --profile k6` is refused; a K6 packed-store `--dry-run` on the modified
tool still resolves `checkpoint_identity_sha256 a8668be3…`, the sealed one, so
default behaviour is unchanged. `runtime_reader_sha256` moves 0582ba57… →
c1112843… by construction (it hashes stream_score.py) and that is disclosed.

Cost, instrumented because it was asked for: 1× H200 spot IN2, $1.99/GPU-h
list. Cold run 1 12,514.5 s (3.48 h), window 1 678 s cold, steady state
467–549 s, 9.31 TB read off CephFS at ~1.05 GB/s with 28 threads. The A100-80GB
at $0.89/h was tried FIRST and rejected in 28 minutes and ~$0.42: its image
ships NVIDIA driver 12080 and the proven venv is torch 2.11.0+cu130, so
`torch._C._cuda_init()` refuses. Not an SM80 problem — `_can_use_grouped_mm`
has no compute-capability check at all. RTX-PRO6000 at $0.99/h in IN1 could not
be tested: 0 free spot CONTAINERS (the free devices were VM-only rows, and
`--spot` is container-only).

Also corrected a number this journal got wrong: the streaming lane is **~$12
per model for two cold runs** (K6 5.97 GPU-h = $11.89; K8 7.78 GPU-h = $15.48,
from the capture receipts), not the ~$6 recorded on 2026-08-29 ~00:15 — that
figure was one cold run, not the pair.

LESSON 21: the GPU generation is rarely the gate, the DRIVER is; a
shared-filesystem venv is a hard pin on it.
LESSON 22: the account balance is not your bill when another session is
renting on the same account — and `jl destroy` erases the cost record.
LESSON 23: a panel mean is a floor plus an error, and only one of those is the
codec. Measure the floor once per lane, early.
LESSON 24: a cache that can refuse should refuse before it allocates.

## 2026-08-29 (evening) — four asks landed: accuracy, portability, performance, one command

All four operator asks shipped as working tools, selftested green on the M4
Max (`bin/selftest_all.sh`: 30 passed, 0 failed, 3 skipped — two reaper tests
guarded by `SELFTEST_SKIP_ACCOUNT` because another session is renting on the
account, one fixture test pending `transformers` under FIDELITY_PYTHON).

**A. Accuracy.** `stream_score.py --capture-role teacher` emits a SAME-LANE
teacher (role flips to `bf16_teacher`, schema unchanged — exactly what
`_find_teacher_receipt` keys on — plus a sealed `teacher_provenance` block,
`malaiwah.glm53-same-lane-teacher-provenance.v1`). Against such a teacher the
lane's floor is exactly 0 with T1 hash evidence (per-window logit sha256
identity; the all-zeros tokenwise npy has the fixed sha256 3ffddc61…be17,
asserted by `bin/selftest_zero_floor.py`); the $6 recipe, the T1/T2 ladder and
the paste-ready reference row live in `k6/SAME-LANE-TEACHER.md` (the GPU run
itself deliberately not executed — no renting in this change).
`bin/fidelity-stats attributable` reproduces the sealed attributables from
receipts (K6 0.013715−0.011506=+0.002209; K8 0.012384−0.011506=+0.000878,
verified live via `--from-registry` against the public dataset) and REFUSES
cross-lane floors with the arithmetic in the message (0.012384−0.012712=
−0.000328, a negative attributable for an 8-bit quant); `paired-delta` gives
the honest CI (paired t via incomplete beta + BCa over windows + sign test +
Wilcoxon; on the committed K8-ANOMALY 11 windows: d̄=−1.2177e-3, s_d=1.7335e-3,
t=−2.33 — the file's own numbers, reproduced). `k6_kld_report.py` now
propagates teacher_source/teacher_label into reports and summaries, groups
comparison tables per teacher (never mixing them), remaps moved teacher trees
with sha256 verification in the fallback path only, and pre-refuses preview
captures.

**B. Portability.** `bin/registry-view` (stock py3.9, stdlib): `check` tiers
artifacts EXACT/UNPINNED/STALE/PINNED-UNVERIFIED against the live head and
prints rows + receipt links (zai FP8 is a STALE hit by default — correct: main
moved past the measured 3f1971b7); `rows` filters by
model/artifact/panel/lane/measured-by/metric/codec/bpw/class and NEVER merges
comparability groups (grouping is by RECOMPUTED key via registry_lib,
anti-tamper); `lineage` walks base_model tags (more complete than cardData —
verified live) to the registry's model — both zai roots land on
model--zai-org.glm-5.3-flash — and picks the panel/teacher precedent with
printed alternatives. Local clone and public HF dataset give identical group
output (verified; snapshots printed in the footer). The live tripwire
(`--selftest-live`) asserts the two published streaming values never move.

**C. Performance.** Honesty first: position sampling is a STORAGE/teacher-
bandwidth knob, not a compute knob (the causal trunk runs all positions
regardless; fp64 scoring is 0.15 ms/position CPU measured — 8 s/panel).
`--store-positions per-window:<m>` + `--sample-seed` produce PREVIEW captures
(schema `malaiwah.glm53-logit-capture-preview.v1`); `bin/kld-preview` scores
them with the stratified estimator + FPC (`fidelity/previewstats.py`, pure
stdlib, T3-certified: unbiased, z-coverage 93.5% where its assumptions hold,
and the measured KNOWN failure — 77.5% coverage on the extreme tail at m=64,
improving to 89% at m=512 — is printed as a tail-dominance warning, since the
estimate and its SE are positively correlated on heavy tails). The 25-window
gate is structural (sd 1.73e-3 > effect 1.22e-3, lessons 28/29). CENSUS mode
(exact, ~8 s/panel once logits exist) is the default local path. The planner
now prices the REAL engine (`window_major_cost`: 36,288×18 ms = 10.9 min/pass;
×25 uncached ≈ 4.5 h; ram caches 7/42 layers on 128 GB; disk = one decode +
46 min of re-reads at 5.5 GB/s assumed-and-labeled) and marks the legacy
layer-outer schedule as a hypothesis no engine implements. The KDA/MPS trunk
time stays null-with-instructions (fixture: `bin/measure-local --fixture
fetch`). MLX stays out of scope: zero MLX code exists in the stack; MPS is the
Apple lane.

**D. One command.** `bin/measure <hf-link>`: parse → registry (published truth
first) → live revision → already-measured gate (rows + receipts, exit 0) →
lineage → panel/teacher pick → surface sniff (tr3-published/MLX refused for
$0.00 with the missing reader named) → lane pick → `measure-local --execute`,
whose preflight lists ALL missing prerequisites with remedies at once
(demonstrated on this Mac: transformers, quant_pipeline, teacher tree,
artifact path, tr3 reader — rc 3, zero tracebacks). measure-local and
measure-cloud grew the same front gate (`--force` /
`--accept-measured-revision` / `--skip-registry-check`).

**Pinning reconciliation.** streaming/local-mps/local-cuda-budget are now
`pinned: true` in `bin/engines.json`, every flag verified against the probed
CLI (AST scrape + `--probe-engines`, all five lanes green). Planner-only knobs
demoted to "(planner cost model only; not an engine flag)"; `--vram-budget`
maps to `--vram-budget-gb`; `--reduce-order native` refused at invocation
build; local lanes are `receipt_class: preview`; the old divergence findings
moved verbatim to bin/README.md "History". The local lanes' minutes_per_window
is now null — the old 0.6/0.25 figures priced a schedule no engine implements
and are withdrawn.

**Schema strings introduced** (all structurally unsubmittable, refused on two
independent axes — bin-side denylist in `fidelity/receipt.build_submission`
AND the registry's const/adapter gates, the latter demonstrated live:
`registry_add.py` exits 3 naming the string):
`malaiwah.glm53-logit-capture-preview.v1`,
`malaiwah.glm53-census-kld-preview.v1`,
`malaiwah.glm53-sampled-kld-preview.v1`,
`malaiwah.glm53-same-lane-teacher-provenance.v1` (teacher: a REFERENCE, never
a measurement), `malaiwah.glm53-floor-attributable-report.v1`,
`malaiwah.glm53-paired-window-delta.v1`.

Open items: the $6 same-lane teacher pair-run (T1 verification + publishing);
the KDA/MPS trunk per-window time (fixture datum after `pip install
transformers` under FIDELITY_PYTHON; then one real window); the local lane's
floor (needs a local native pass over ~630 GB); σ_w/σ_dpos re-anchoring from
the first full local census pass (the 0.05/0.028 design numbers are estimates
— sealed tokenwise arrays died with the box); `load_capture_receipt`'s exact
validation set (L1.g's reimplemented predicate is the guard until
quant_pipeline is cloned); layer-major preview scheduling (gated on
fixture-proven bitwise equivalence + ≥1 real window).

LESSON 25: a lane's floor is a property of (panel, teacher, lane) — the
tooling now refuses the subtraction instead of documenting that you shouldn't.
LESSON 26: on heavy-tailed data the estimate and its own SE are positively
correlated, so a z-interval that "has the right SE on average" still
under-covers — quote the wider interval, disclose the tail, and fix it with
more positions, not more runs.

## 2026-08-29 (night) — three-review closeout: every blocker/major fixed, suite 33/0/0

Three independent reviews ran against the evening's work (adversarial
correctness, statistical validity, stranger usability). All three returned
GO_WITH_FIXES; this entry closes them out. Final battery after every fix:
`bash bin/selftest_all.sh` — **33 passed, 0 failed, 0 skipped, rc 0**,
including the 0.1B fixture ladder (b,c,f,g,h,i,j all ok on MPS, bitwise_equal
true) and the reaper tests now safe-by-default.

**Fixed during the reviews themselves** (files were left in the tree by the
reviewers; shipped with this commit): the front gate no longer silently
replaces an unresolvable explicit revision with live main (warns, tiers
PINNED-UNVERIFIED — same rule as `check`); renamed HF repos (307-redirects)
are canonicalized before registry matching so an old name cannot false-negative
into a duplicate paid measurement; `invoke_engine.py` composes the streaming
argv with `--source` + the lane's fixed_flags (parses clean through
stream_score's real argparse); `measure` tiers only against a resolved 40-hex
sha; the preview position sampler is FRACTIONAL-step systematic in both copies
(the integer-step design gave every position ≥ k·m inclusion probability ZERO
— +6.5% to +16.4% measured bias; new design: inclusion exactly m/N, bias
−0.03%, coverage at default m=256 96.8% on the observed tail shape); census
PanelGateError degrades to diagnostics instead of a traceback; windows_total
pins to ≥25 whenever the sealed EP8 teacher sha is claimed; paired window-set
equality is checked with clean messages; README's first example now answers
for $0.00 and the pip lines carry the PEP 668 escape.

**Fixed in this closeout session:**
- `bin/measure --accept-measured-revision` on a STALE artifact now re-runs the
  tier match at the measured commit and takes the ALREADY-MEASURED exit-0
  branch (rows printed at the accepted revision; verified live on
  zai-org/GLM-5.3-Flash-BF16). Measuring anyway stays behind `--force`. The
  old behavior dead-ended a stranger in a 4-prerequisite preflight refusal.
- `selftest_all.sh` teardown: `reaper --sweep` runs with `--dry-run` (new
  plumbed flag on the reaper subcommand — reports, destroys NOTHING;
  destruction is never a side effect of "run the selftests"), and machines
  without the `jl` CLI SKIP with the install remedy instead of failing.
  `SELFTEST_SKIP_ACCOUNT=1` still skips the section entirely.
- The viewer's "(single row — nothing to rank against)" note now counts the
  group against the WHOLE snapshot: filtered/artifact-scoped views print
  "(1 of N rows in this comparability group shown …)" — verified on 0xSero
  (1 of 6) and the BF16 check (1 of 6 / 1 of 2).
- UNDISCLOSED panels get an explicit CAVEAT line (sibling of the subset
  caveat; keyed on the `undisclosed_panel` disclosure code) — the orcarouter
  0.0063 row can no longer be scanning-read as the best number on the page.
  The no-declared-lane sub-table is annotated "(sealed rows land here: class
  strict is the sealed number)" when it holds strict-class rows.
- `registry-view lineage --lane L` now prefers the floor row whose PIPELINE
  declares lane L (verified: streaming intent names
  measurement--glm53.bf16-stream-floor…, and explicitly flags the reference's
  self_consistency floor as a DIFFERENT lane's). The data-side fix (per-lane
  floor in reference self_consistency) stays with the registry agent.
- Same-teacher floor forgery hardening: `attributable` gate 2b requires the
  floor summary to carry a `profile` naming its lane, and any floor claiming
  the sealed streaming teacher must be profile `native-bf16-stream` (the
  cross-stack 0.012712 value exists in no receipt carrying that profile).
  New selftest case [9b]; sealed attributables still reproduce to the last
  digit (+0.00220896620326625423 / +0.00087826840410656741, live via
  `--from-registry`). A forgery of the profile field TOO remains out of scope
  — receipts are unsigned.
- LANE-ONLY identity (the stats review's follow-up, implemented forward-
  looking): stream_score's backend.json now carries `lane_identity` +
  `lane_identity_sha256` (schema malaiwah.glm53-streaming-lane-identity.v1 —
  sha256 over torch/cuda/device/kernel/numeric-policy/attention/experts-impl/
  parallelism/ep/reduce-order and NOTHING artifact-specific), k6_kld_report
  copies it into reports as `student_lane_identity_sha256` when present (
  conditionally — historical reports reproduce byte-identically), and
  fidelity-stats gates paired-delta and attributable on ITS equality when
  both sides carry it (equality VERIFIES the lane; inequality refuses with
  both hashes). Receipts predating today lack the field and keep the
  disclosure-warning behavior. The capture receipt's top-level key set is
  UNCHANGED (L1.j golden keys still pass) — the sealed layout is not touched.
- Expected-refusal output hygiene: the h-rung captures the sealed scorer's
  intentional refusal stderr and re-emits it inside its [ok] record; the
  fixture driver disables HF progress bars in the replayed subprocess; the
  vacuous `unexpected_keys_are_exactly…: false` field is emitted only when
  unexpected_keys > 0 (`stray` stays the load-bearing gate). Sha-pair
  refusal displays print FULL hashes when truncations would collide.
- PEP 668: every printed pip remedy (engines preflight, selftest skip text)
  carries the --break-system-packages / venv+FIDELITY_PYTHON escape.

**Residuals, documented not fixed:** a renamed repo queried WITH an explicit
40-hex sha now gets ONE tolerant canonicalization attempt when the alias
matches nothing (silent on failure — pinned-sha flows stay network-optional);
a hand-built unsigned 1-window teacher+student pair still yields a
windows_used=1/windows_total=1 preview (legitimate for fixture panels; full
closure needs receipt-seal verification, unsafe while quant_pipeline's exact
canonical_json cannot be probed locally); sampled CIs on comparisons where
any sampled value exceeds ~5 nats should be treated as suspect and m raised
(coverage sim: 91% in that unobserved regime, 96.8% on the real tail).

LESSON 27b (reviews as instruments): three reviewers attacking the same tree
concurrently found one bias the author's own synthetic test could not (its
population had exchangeable positions — no positional trend, no bias to see).
Selftest populations must contain the structure the estimator is allowed to
get wrong.

## 2026-08-29 (late night) — the stack fingerprint: Phaelon's kernel question becomes a receipt field

A Discord reviewer (Phaelon) put it plainly: "What vLLM runner is used, do
you record what specific kernels are used? Do you enable enforce-eager
(which disables CUDA graphs)? This automation pipeline is nice, but
obfuscates way too much" — and, fairly, "if you capture all of that, totally
rad." The audit that followed confirmed the sting: the facts EXISTED
(environment.json with the exact vllm dev sha and full pip freeze, the image
digest, the determinism receipt family that fed vLLM PR #53906), but the
measurement summary receipts linked none of it by digest, and
enforce_eager/attention-backend were established only by code defaults at a
pinned commit plus per-boot engine logs sitting in the PRIVATE scratch
dataset.

**Shipped tonight:**

- `bin/fidelity/stackprint.py` (`malaiwah.stack-fingerprint.v1`): stdlib-only
  at import, probes lazily, NEVER guesses — engine build+git sha,
  enforce_eager/compilation/cudagraph state queried from the live
  `vllm_config`, attention backend requested AND selected (each with its
  source), kernel-config echo, the determinism-relevant env pins, container
  image digest, GPU inventory, pip-freeze sha256 (freeze written alongside).
  Canonical hash excludes timestamps/paths, so identical stacks hash
  identically (the lane-identity trick). Unqueryable facts record the reason.
  `python3 bin/selftest_stackprint.py` (T9, wired into selftest_all) proves
  determinism, engine-absent handling, and MPS/CUDA-absent handling.
- Serving lane wired: `fidelity.py capture` embeds the fingerprint verbatim
  in the capture manifest (NOT in the capture contract — reuse gating is
  unchanged) and refuses to run without the module; `qualify` embeds its own
  fingerprint + the operand's; `replay` and `cross_check compare` now name
  their operand manifests BY DIGEST (lesson 20 closed) and fingerprint the
  comparator host with engine kind "none"; `gen_check` and
  `activation_capture` fingerprint too. `make_bundle.sh` + `bin/BUNDLE.txt`
  ship the module to both kinds of instance.
- Registry: `registry_add.py --stack-fingerprint-sha256` (+ optional
  `--stack-fingerprint-uri`) records it under provenance
  (schema: optional nullable property + a receipt_file source row).
  Provenance-only for now — it does NOT enter the comparability key; whether
  two rows with different fingerprints stay comparable needs real thought,
  not a flag. `make check` 0 errors; negative controls verified (typo
  property and bad hex are refused by the mini validator).
- `WHAT-WE-MEASURE.md` section 7 + checklist item 8: what each lane records,
  where the sealed rows' evidence lives, and the rule that a fingerprint-less
  future receipt is refusable.
- RETRO-DISCLOSURE published: `reports/stack-provenance-retro.json` in the
  suite dataset (and mirrored here) maps every sealed row BY RECEIPT DIGEST
  to its environment evidence BY FILE DIGEST plus the established
  enforce_eager=True / CUDAGraphMode.NONE / FLASH_ATTN_MLA_SPARSE /
  TritonExperts-vs-FlashInferFp8DeepGEMM facts — each fact labeled
  receipt_field | code_default_at_pinned_commit | log_evidence (six launch
  logs pinned by sha256), and what cannot be established is listed as
  unknown, plainly (Triton autotune winners of the sealed launches — bounded
  by the measured 8.7e-4 noise floor; DeepGEMM mHC JIT identity; the full
  40-char vllm commit). Self-sealed; seal verifies.

**TODO (checkpoint lane, deliberately NOT done tonight):** wiring the
fingerprint into `k6/tools/stream_score.py` waits for the in-flight
format-adapters merge that owns that file — landing it now would manufacture
a conflict. The adapter is ready (`stackprint.from_backend_json(backend)`,
selftested against the published teacher backend.json shape): after their
merge lands on origin/main and a rebase, call it right after `backend` is
assembled and store `backend["stack_fingerprint"]` +
`backend["stack_fingerprint_sha256"]`; same call in `k6_student_capture.py`.

**URGENT (data preservation, for the operator):** the per-run checkpoint-lane
preimages (kld-report.json, backend.json, reader-identity.json, plan.json,
student capture receipts for the three streaming rows and the sealed EP8
student) exist ONLY on JarvisLabs fs 3394, which is slated for destruction
after the K6 session. The retro receipt marks every such digest
"private-fs-only, at risk". Freeze them into the suite dataset (and ideally
the six cited launch logs, ~2 MB) BEFORE the fs goes away, or those chain
links become permanently digest-only.

LESSON 30 (transparency): "we could reconstruct it if asked" is not
disclosure. The reviewer was right — a pipeline that records everything but
links nothing has the epistemics of a pipeline that records nothing. The fix
was not more capture; it was making every receipt NAME its stack by digest,
and publishing the retroactive map for the rows that predate the rule.

## 2026-08-29 — Lesson 33: a dependency guard that does not list every dependency
`bin/bootstrap_measure.sh` installed `hf_transfer` in its wheel block, and both
fetch stages export `HF_HUB_ENABLE_HF_TRANSFER=1`. Correct in isolation — but
the whole block was guarded by
`import torch, transformers, safetensors, huggingface_hub`, and the JarvisLabs
pytorch template already ships all four. On a template box the guard
short-circuits, the block never runs, and `hf_transfer` alone is missing while
the fetch stages still request it. Modern huggingface_hub errors when the flag
is set without the package; older versions fall back silently to a much slower
path. Caught on the M1 turboderp box by an operator question ("did you give them
the token so downloads are efficient?") — the answer was that the token was not
the issue, the accelerator was simply absent. Measured contrast on the two live
boxes at that moment: the A/B box, fetching with a custom 64-way parallel range
fetcher, sustained 111 MB/s; the M1 box had no accelerated path at all.
RULE: an import guard must name EVERY package the block installs, or it silently
becomes a partial install on any host that pre-ships a subset. Fixed three ways:
`hf_transfer` added to the guard, an idempotent single-package ensure step after
it, and a hard fail-closed check (the fetch stages demand the flag, so a box
without the package must not proceed). It now also prints into
`wheel-versions.txt`, so the receipt shows whether the accelerated path existed.

## 2026-08-29 — MLX surface: a third weight-decode surface, built and validated with no GPU

`k6/tools/mlx_surface.py` + `stream_score.py --source mlx --profile mlx` score
community **MLX affine** conversions of GLM-5.3-Flash (orcarouter dialect: HF
tensor names, per-expert `weight`/`scales`/`biases` triplets) on the sealed
25-window panel, through the same streaming capture, the same fp64 estimator
and the same EP8/fp32 lane as K6/K8/Dione/native-BF16. Summary schema
`malaiwah.glm53-mlx-packed-kld-summary.v1`; full write-up in
[`k6/tools/MLX-SURFACE.md`](k6/tools/MLX-SURFACE.md).

**The finding that shaped the design.** This format quantizes PAST the routed
experts. Censused from orcarouter's own index and shard headers (revision
c80f6810, 113,446 stored tensors): 36,288 routed + 864 MTP expert modules, and
also 129 shared-expert (6-bit), 9 dense-MLP and 48 DSA attention modules —
37,338 quantized modules and 1,432 passthrough tensors, together bijecting the
38,770-tensor official BF16 set exactly. So `--bf16` stops being an input of
this source: the non-routed model is a MATERIALIZED DECODED VIEW of the quant
snapshot (passthrough verbatim, quantized non-routed fp32-dequantized and
rounded once to bf16, ~19 GB, hash-stamped and reused, stale views refused),
which the sealed `from_pretrained` then loads with its zero-missing /
zero-stray assertions unchanged. Everything downstream — residency, slab
binding, fuse_gate_up, the single bf16 rounding, the combine — is the lane's
own, untouched.

**Decode proven, not asserted.** Plain-torch byte-level unpack, fp32 accumulate,
no float64 and no uint32 views (so it runs on CUDA, MPS and CPU). Our fp32
dequant rounded ONCE to mlx's own output dtype is BITWISE equal to
`mlx.core.dequantize` (mlx 0.32.2) on six real ranged-fetched orcarouter
tensors — 4-bit `experts.0.gate_proj` [2048,4096], 5-bit `experts.0.down_proj`,
6-bit `shared_experts.gate_proj`, 5-bit dense `layers.0.mlp.down_proj`
[4096,12288], the MTP layer-45 expert, and 4-bit `self_attn.o_proj`
[4096,16384] — and on an 8-bit BF16-scale `embed_tokens` row slice from
pipenetwork's mixed-4_8bit build, which covers the width and scale dtype
orcarouter does not contain. In fp32 the two differ by ≤1 ulp (mlx fuses the
multiply-add), so the claim is bitwise equality AT MLX'S OUTPUT DTYPE with the
fp32 delta reported — never "equal in fp32".

**Bits are derived, not believed.** Per-tensor `(bits, group_size)` come from
the stored shapes against the official BF16 shape census
(`bits = 32*packed_cols/in_features`), cross-checked against config.json's
override map; a disagreement on layers 0–44 refuses. Measured disclosure: 291
layer-45 modules are stored at 5/6-bit while the config override map does not
mention layer 45 at all — recorded, not refused (layer 45 never executes).

**Free integrity check discovered while validating the fetch ledger:** the
index's declared `metadata.total_size` for this snapshot is the ON-DISK total,
and our per-class ledger reconciles with it exactly —
203,976,457,080 tensor bytes + 15,619,216 bytes of safetensors container
headers (62 shards) = 203,992,076,296. Which convention a snapshot used is
now RECORDED (`declared_total_matches`) rather than assumed, since writers
differ (transformers declares tensor bytes only).

**Validation, all offline (`k6/tools/selftest_mlx_offline.py`, 8 rungs, ~8 s,
wired into `bin/selftest_all.sh`):** reference-packer round trip over 18
bits×group-size combos; real-tensor mlx replay from 7 committed fixtures (runs
where mlx cannot be installed — every CUDA box); live `mlx.core.quantize`
round trip over 36 cells (codes EXACTLY mlx's codes, output bitwise equal, f16
AND bf16 scales); the real orcarouter census plus 9 named refusals; the
streaming/decoded-view plumbing at real routed geometry; both dry-runs; and the
registry adapter. `bin/selftest_all.sh` is 32 passed / 0 failed / 2 skipped
(account-gated) with the new rung in.

**Refused by name, never skipped:** the mlx-vlm ("pipenetwork") dialect (fused
`switch_mlp`, renamed modules, no MTP layer), non-affine MLX modes, a census
that does not close, a passthrough tensor differing from the official
dtype/shape, an underivable bit width, a config declaration disagreeing with
the stored shapes, and a capture without a pinned 40-hex revision.
inferencerlabs Q9 (gs32, BF16 scales, no index.json) is decodable by this
kernel but needs a header-glob census: second wave.

**Registry.** `registry_add` gains the family (lane supplied by `--lane`, like
K8 and native-BF16) and, for it alone, refuses a receipt that carries no
`mlx_scope_policy` census — a row that does not say what was quantized would be
read as if only the experts were.

LESSON 33 (scope is part of the measurement): "quantized to 4 bits" names a
codec, not an artifact. Two 4-bit GLM-5.3-Flash conversions can differ by
36,288 vs 37,338 quantized modules, and the difference lands in the same
scalar we publish. Every surface adapter from now on censuses what the
artifact actually quantized, from the artifact's own metadata, and the receipt
carries that census verbatim.

---

## 2026-08-29 — GGUF weight-decode surface (`--source gguf`), built and validated without a GPU

Third scoring surface for the streaming lane, after `dione` and `native`:
community **llama.cpp GGUF** artifacts of GLM-5.3-Flash (unsloth's Q8_0 /
UD-Q4_K_XL / UD-Q5_K_XL / UD-Q6_K_XL) scored on the sealed panel through the
same capture, same teacher, same fp64 estimator, same EP8/fp32 lane. New files:
`k6/tools/gguf_surface.py`, `k6/tools/selftest_gguf_offline.py`,
`k6/tools/gguf-evidence/`. Receipt family
`malaiwah.glm53-gguf-packed-kld-summary.v1`. **No number was measured** — the
first capture is a rental; everything below is what was proven for free.

**The architectural difference that drove the design.** Every other source in
this tool quantizes the routed experts only and runs the official BF16
non-routed parameters untouched. A GGUF quantizes `token_embd`, `output`
(lm_head), every attention/KDA/DSA projection and the shared experts too — at
Q8_0 in the unsloth builds. Scoring those from the official tree would have
measured a model that does not exist. So the lane MATERIALIZES a decoded
non-routed view (every non-routed tensor dequantized once into safetensors
under the official HF names/shapes/dtypes) and `build_streaming_model` grew a
`nonrouted_view=` parameter to accept it. The sealed `from_pretrained` call and
all of its load assertions are unchanged. `--bf16` survives for
config/tokenizer, the inventory binding, and the vision tower — which the main
GGUF genuinely does not carry (it ships as a separate mmproj) and which the
text-only panel never executes.

**Scope is measured, not asserted.** The receipt's `scope_policy` block is read
from the artifact's own tensor table (which tensors carry a quantized ggml
type), never inferred from the format's name — because the assumption "GGUF
quantizes everything, NVFP4 is the same family" is false: the NVFP4 releases of
this model quantize the routed experts only. `registry_add.py` turns the block
into a `quantization_scope_whole_model` disclosure and REFUSES a GGUF summary
that arrives without it, or without `gguf_files`. `WHAT-WE-MEASURE.md` gained
§5a as the worked example.

**Two layout assumptions were settled numerically, not by reading the C.** Both
are the same species of hazard: a wrong answer decodes cleanly, closes every
census, and measures the wrong model.

- `kv_b_proj` does not exist in a GGUF — llama.cpp stores `attn_k_b` (per-head
  TRANSPOSED) and `attn_v_b`. Four candidate reconstructions were scored
  against the official BF16 tensor: the shipped one lands at rel-L2 **0.0054**
  (the Q8_0 error), every other at **>= 1.40**.
- The fused expert tensor's slot `e` was ASSUMED to be HF expert `e`.
  `audit_expert_placement` proves it: slot 0 of `blk.3.ffn_gate_exps.weight`
  reproduces official `experts.0.gate_proj` at rel-L2 **0.0714** (the Q4_K
  error) against **1.42** for every row-shifted control — which settles the slot
  ordering, the reversed-dims orientation and the projection mapping at once.
  This check did not exist before today and cost nothing: the official payload
  was already committed in `dione-evidence/`.

**LESSON 28 (scale-free audit criteria).** The MLA audit originally passed on a
cosine MARGIN (`shipped > runner_up + 0.5`). Running it on a 2-head window
instead of all 64 heads failed it — not because the layout was wrong but
because two arrangements sharing a leading block have a cosine gap that shrinks
as `1/(2*heads)`: 0.013 over 64 heads, 0.546 over 2. The criterion was
window-size dependent, i.e. it would have passed or failed depending on how
much of the tensor an operator chose to fetch. Replaced with rel-L2, which does
not move: the right arrangement scores the QUANTIZATION error and every wrong
one scores O(1), at either size. An audit threshold that depends on the sample
size is not a threshold.

**LESSON 29 (don't trust your own dtype list).** The view's dtype policy started
as a hardcoded suffix list of the tensors the official tree stores float32.
That is a claim about a released checkpoint that can go stale silently and
would leave the view NOT dtype-identical to a native build. It is now
cross-checked: `verify_official_dtypes` reads the real dtypes out of the
official safetensors headers wherever those shards are present and refuses on
any disagreement, counting (never assuming away) the shards that are absent.

**Validation, all four gates, no GPU and no rental.**
1. *Reference cross-check.* Q4_K, Q5_K, Q6_K and Q8_0 are BITWISE equal to
   gguf-py 0.19.0's `dequantize` on real ranged-fetched bytes of the live
   artifact — and identically so under python3.9/torch2.8 and
   python3.14/torch2.13. A scalar transliteration of llama.cpp's own
   `get_scale_min_k4` independently reproduces the Q4_K sub-block scales, which
   is the check a same-code-twice comparison cannot make.
2. *Shape/name census.* All 1,412 GGUF tensors consumed (1,259 one-to-one + 129
   fused + 24 MLA halves); the resulting 1,271 official names EXACTLY biject the
   real BF16 index (38,770 − 37,152 routed − 347 vision). ddh0's different
   convert vintage maps 1,412/1,412 names (its `indexer.kpool_*` alias spellings
   are covered) and is refused only by TYPE.
3. *Offline selftest.* `selftest_gguf_offline.py`, nine rungs, ~2 s, wired into
   `bin/selftest_all.sh`. Includes a minimal GGUF WRITER so the refusal rungs
   can build the malformed artifacts they must refuse — eight of them, each
   required to name the offending tensor or key.
4. *Dry-run.* `stream_score.py --source gguf --dry-run` plans the real
   6-part unsloth UD-Q4_K_XL over HTTP ranges (headers only, no weights):
   1,412 tensors, 36,288 streamed routed modules, 185.48 GB of routed bytes out
   of the artifact's 199.71 GB, the imatrix provenance keys, and clean refusals
   for a partial split, an unpinned revision, a mismatched profile, and an
   https location without `--dry-run`.

**Named v1 exclusions (refused, not skipped), enumerated from the real repo.**
A type census of ALL TWELVE unsloth builds (each build's own 1,412-tensor
table, `gguf-evidence/unsloth-build-census.json`) says the supported set scores
BF16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and refuses the other seven. The
refusals are NOT predictable from the directory names, which is the finding
worth keeping: unsloth's "Dynamic" recipe mixes IQ2_XS/IQ3_XXS/IQ4_XS into
UD-Q2_K_XL and IQ3_XXS/IQ4_XS into UD-Q3_K_XL, so those two are gated on IQ
kernels, not on the Q2_K/Q3_K ones their names imply — adding Q2_K and Q3_K
alone would unlock nothing. Any unsupported type is refused BY NAME AND TYPE at
census time, before a byte is decoded. (Also noted: the repo's own BF16 GGUF is
in the supported set, so this lane can measure a GGUF of *unquantized* weights
— its own container floor — without a second surface.)

**Fetch-ledger honesty fix.** `routed_tensor_census` originally reported one
layer's per-expert byte cost as if it were the artifact's. The unsloth XL builds
deliberately mix types across layers (Q4_K gate/up with one Q5_K layer each;
Q5_K down with three Q6_K layers), so that understated the ledger. It now
reports the distinct sizes per projection plus the exact streamed total.

**Guard that fired, correctly.** Adding provenance fields to the capture receipt
tripped `stream_score_selftest` rung L1.j, which asserts a default receipt is
field-identical to the sealed golden shape and that every later assignment is
flag-gated. The additions were correctly gated; the rung's allowlist of
*permitted* gated keys was extended deliberately, which is exactly the review
this guard exists to force.

---

## 2026-08-29 — three community-quant surfaces merged into one tree, reviewed adversarially

MLX, GGUF and NVFP4 were built in three separate worktrees against three
different bases. This entry is the INTEGRATION: one `--source` dispatch, one
selftest list, one registry adapter table — and an adversarial pass that
re-derived every claim rather than reading the three reports.

### What the three lanes are

| lane | artifact family | reference implementation the decode is proven against | measured scope |
|---|---|---|---|
| `--source mlx` | Apple-silicon MLX affine (u32-packed codes + per-group scale/bias) | `mlx.core.dequantize` 0.32.2 | routed experts + dense MLPs + shared experts + 4 DSA projections; embeddings, `lm_head`, the whole KDA path and the vision tower PASS THROUGH |
| `--source gguf` | llama.cpp GGUF K-quants (Q4_K/Q5_K/Q6_K/Q8_0) | `gguf-py` 0.19.0 `dequantize()` | everything — `token_embd` and `output` are Q8_0 |
| `--source nvfp4` | compressed-tensors NVFP4 (e2m1 + per-16 f8e4m3 scale + fp32 global) | `compressed-tensors` 0.18.0 | routed experts ONLY — same scope as K6/K8 |

**The scope column is the point.** The brief assumed all three quantize
everything. Only GGUF does. Three "community 4-bit" artifacts of the same model
draw three different sensitivity boundaries, and none of it is predictable from
the format name — so scope is CENSUSED from each artifact's own index/tensor
table, carried in the receipt, and `registry_add.py` refuses a summary of any of
these families that arrives without its census.

### Validation evidence, re-derived here rather than trusted

Every number below was reproduced in this session, on this Mac, no GPU rented,
no HF token, no full download.

- **Reference cross-check, live, over HTTP ranges, with an independent fetcher
  and an independent GGUF header parser** (neither borrowed from the adapters):
  - MLX `layers.3.mlp.experts.0.gate_proj` @ `c80f6810` — bitwise equal at
    mlx's own output dtype, max |fp32 Δ| 3.05e-05 (one f16 ulp; mlx fuses the
    multiply-add, we do not). Worked value `[0.007671356201171875,
    -0.01534271240234375, -0.0306854248046875, -0.0306854248046875]`.
  - GGUF `blk.3.ffn_gate_exps` (Q4_K), `blk.3.ffn_down_exps` (Q5_K),
    `blk.11.ffn_down_exps` (Q6_K), `token_embd` (Q8_0) @ `2975ab41` — BITWISE
    equal to `gguf-py`, max |Δ| exactly 0, on 72 KB of fetched payload.
  - NVFP4 `layers.3.mlp.experts.0.gate_proj` from BOTH dialects (RedHatAI
    @`36c184c6` compressed-tensors, LibertAIDAI @`357b45cc` modelopt) — bit
    pattern equal, signed zeros included, max |Δ| 0.0.
- **Every selftest re-run**: `bin/selftest_all.sh` 37 passed / 0 failed / 0
  skipped; `make check` in `registry/` 62 passed / 0 failed.
- **The `stream_score --dry-run` leg that all three agents had to SKIP is now
  closed**: a real `quant_pipeline` tree exists on this machine, so
  `selftest_{mlx,gguf,nvfp4}_offline.py --pipeline-root` runs it for real. MLX
  goes 8/8 with 0 skips; GGUF 9/9; NVFP4 10/10.
- **Cross-format refusals**: each surface fed each other's artifact refuses by
  name — MLX on an NVFP4 index names the extra/absent tensors, NVFP4 on an MLX
  config names `config_groups/group_0`, GGUF on a safetensors file named `.gguf`
  says "not a GGUF file (magic differs)". All exit non-zero.
- **`--source` × `--profile` matrix**: all 30 combinations exercised; the
  pairing gate is exactly diagonal.
- **No float64 on any decode path**: a `Tensor.to` tripwire over 42 kernel cells
  (6 bit-widths × 3 group sizes for MLX, 4 ggml types × 4 trials, 2 NVFP4
  conventions × 4) created ZERO float64 tensors, and every cell is bit-pattern
  identical on MPS and CPU. The only `float64` in the three adapters lives in
  `gguf_surface`'s two CLI-only placement audits, which `stream_score` never
  calls.
- **Receipt families are distinct** (`…-mlx-…` / `…-gguf-…` / `…-nvfp4-…`), all
  three carry `lane: None` + `requires_lane: True` (the K8 contract: `--lane`
  must state it), and a well-formed synthetic summary of each adapts to a row
  while **ten** provenance-violating variants refuse with the right exit codes
  (4 missing, 5 inconsistent, 7 identity clash).

### The bug the merge caught

Rebasing the mlx surface onto the concurrently-shipped exl3hf one had turned
the checkpoint-identity dispatch from one chain into two:

```
if  args.source == "exl3hf":   ...          # sets identity
if  args.source == "mlx":      ...          # a NEW chain
elif args.source == "native":  ...
else:                          ...          # surface.contract_sha256
```

`surface` is `None` on the exl3hf path, so **every `--source exl3hf` capture
would have died with an `AttributeError` after building its identity** — a lane
shipped by the other workflow, broken by a merge nobody had reason to re-read.
Merged into one chain (`exl3hf | mlx | gguf | native | nvfp4 | else`), and a new
static rung **L1.k** now proves there is exactly ONE `args.source` chain per
dispatched variable and that it ends in a catch-all. Verified by mutation:
re-splitting the chain turns L1.k red and names both halves.

LESSON 34 (dispatch shape is a testable property). A chain ending in `else:` is
a trap for the next surface: appending `if args.source == "<new>":` instead of
`elif` is invisible in review, passes every existing test, and silently runs the
catch-all for every earlier source. Three agents each appended a branch; two
appended safely and one did not. What caught it was reading the merged AST, not
reading the diff — so the AST reading is now a rung.

LESSON 35 (a cross-check is only as good as its operation order). The first
independent NVFP4 check reported a 7.45e-09 mismatch, and the tempting reading
was "the adapter is 1 ulp off". It was the reference that was wrong: the adapter
does `scale/global` first and then multiplies, which is exactly what
`compressed_tensors._dequantize` does; the naive `values * scale / global` is a
different rounding. Reproduce the reference's ORDER, not just its formula —
otherwise an adversarial check manufactures the defect it claims to find.

### Merge decisions worth knowing

- The streamer gained named `gguf_source` / `nvfp4_source` parameters on the
  shared producer/consumer loop. The gguf branch had ridden in through
  `native_source` and the nvfp4 branch through a generic `decoded_source`; both
  would have made the "cannot serve two routed sources at once" refusal name the
  wrong source. All five are now named.
- `build_streaming_model` carries BOTH ways a non-official non-routed set
  reaches the forward, documented together: `nonrouted_view` (mlx/gguf hand it a
  MATERIALIZED decoded view) and `view_name`/`config_strip_keys` (nvfp4 points
  the ordinary symlink view at the quant snapshot with `quantization_config`
  stripped from the VIEW's config copy).
- The GGUF summary's `profile` field said `gguf-tp4`. A single-file llama.cpp
  container is not TP4-sliced; it now says `gguf-stream`, like the mlx and
  nvfp4 lanes.
- `registry_add.py` gained ONE adapter table keyed on the schema string instead
  of three sequential `if sch == …` blocks, and the gguf seal disclosure moved
  onto the coded channel the mlx/dione families already use.
- `WHAT-WE-MEASURE.md` §2 claimed "the lm_head weights are never quantized in
  any artifact measured here". That was already false when the stock-exllamav3
  (turbo) lane landed — it quantizes the head at 6 bits — and GGUF makes it
  emphatically false. Corrected, with the scope table moved into a new §5a.
- `bin/BUNDLE.txt` did not list `gguf_surface.py`, `nvfp4_surface.py` or the
  nvfp4 official-name evidence. On the instance that is a crash after the
  receipt is sealed. Added; the bundle-only seal now stages 55 files and still
  validates.
- `k6/STREAMING.md` had two sections numbered 13 and a stranded 11: three agents
  appended lanes blind to each other. Renumbered 12–16 with the MLX lane
  pointing at its own `MLX-SURFACE.md`.

### What a paid measurement of each will cost

Nothing here has been measured yet — no capture has run against real weights on
any of the three lanes. The shapes, stated as expectation and not as
measurement:

- **NVFP4** is the cheapest: routed-only scope means NO decoded non-routed view
  is materialized, the snapshot's own BF16 tensors are symlinked, and the read
  is ~4.08 GB/layer against the BF16 floor lane's measured 14.50 GB/layer. The
  decode is a LUT gather plus one multiply. Two cold runs on the streaming lane
  is the unit of work; whether the lane is IO- or decode-bound is itself a
  measurement, which is why the receipt records `nvfp4_payload_bytes_read` and
  `nvfp4_shards_read`.
- **GGUF** adds a one-time ~19 GB write of the materialized non-routed view plus
  a full decode pass on cold run 1 (reused after, via a fingerprint stamp), on
  top of streaming 185,478,414,336 B of routed payload out of a
  199,707,321,347 B artifact. Budget the disk and the first-run wall clock.
- **MLX** has the same ~19 GB decoded-view cost as GGUF and streams a
  203,992,076,296 B artifact whose ledger reconciles exactly with the index's
  declared total.

Before the first paid capture on any NEW artifact of these families, run that
family's preflight: `gguf_surface.py audit-mla` and `audit-expert`,
`nvfp4_surface.py probe` and `verify-nonrouted`, `mlx_surface.py crosscheck`.
They are cheap, they are offline, and each one guards a layout assumption that
would decode cleanly while measuring the wrong model.
