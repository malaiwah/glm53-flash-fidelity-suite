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
github.com/malaiwah/quant-fidelity-suite (pre-flight r4). Publish stage
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
malaiwah/quant-fidelity-suite-v1 + calibration-activations-v1 ->
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

## 2026-08-29 — the operator architecture: capture and comparison split into two steps

The measuring stack has always fused two different jobs. `stream_score.py` runs
a model over a panel; `k6_kld_report.py` scores that run against a teacher. The
only durable output was a number plus receipts pointing at filesystem paths.
Today that becomes three separable steps behind one tool — `bin/fidelity-dataset
capture | verify | compare` — with a versioned on-disk format
(`malaiwah.fidelity-dataset.v1`), a comparison receipt, an HF card annotation,
and 94 selftest cases.

### The lesson that forced it: we lost a filesystem, and with it a capability

A JarvisLabs filesystem holding our sealed `layers/*.json` and `experts/*.json`
receipt trees was destroyed after being wrongly declared redundant. What that
cost is precise and worth stating without softening:

- The published K6/K8 checkpoints are still self-contained **for serving** —
  `exl3-mcg-storage-abi.json` present, payloads inline, readable through
  `stream_score --source exl3hf`, and `lm_head.weight` still a plain BF16
  tensor in the index.
- But `stream_score --source checkpoint` does
  `packed_root = Path(materialization["packed_root"]).resolve()` and fails if it
  is not a directory, **with no override flag**. That value is
  `/home/jl_fs/glm53-k6/out-k6`. `--source payload-store` needs `contract.json`,
  `inventory.json`, `mtp-adapter-receipt.json` and the `payload-store/` trees,
  none of which are published. Both packed reading paths are therefore
  unreachable from public artifacts, and the published materialization receipt
  still names the dead path.

The registry already had the field for this condition:
`reference.logits_available`, documented as *"false means a number against this
reference can never be re-derived, only re-run."* We had been running with it
false and calling that fine.

**A fidelity dataset would have made the loss irrelevant.** A sealed capture of
the reference is a downloadable public good; losing the machine that produced it
costs nothing, because the thing anyone needs is the capture, not the box. That
is the whole argument for the refactor, and it is an argument from damage
already taken rather than from principle.

### What the split buys, concretely

- **A root capture is paid for once.** Every measurement used to re-pay for
  capture — scoring quant *N* re-ran the reference or depended on a teacher tree
  somebody was still holding.
- **A quant author can contribute a capture with no access to our
  infrastructure**, and publish it *before* any comparison exists. That is a
  hard requirement on the format, and it is the thing the only serious prior art
  (Festr's kimi-k3 artifact) cannot do: he embeds the panel inside the reference
  artifact, so his candidate captures have nowhere to live and only the compare
  receipts survive.
- **The same-lane floor largely stops existing.** Our published cross-stack
  floor is 0.012712 nats — comparable in magnitude to K6's entire 0.013723. That
  is comparison overhead, not quantization. Two captures on one lane compared
  offline in fp64 remove it structurally instead of by subtraction, which
  BIAS-006 forbids across lanes anyway.

### Lesson 34: a container digest is not an identity, and we had both conventions live

The single most load-bearing rule in the format is that head identity is the
**tensor content** digest, never the file digest. We were carrying both
conventions in published receipts without noticing:
`head-extraction.json` and `head-equality-fp8.json` record `47eaf729…` (the
file), while `k6/hidden-replay-evidence/nonrouted-sparse-fetch.json` records
`aa21c427…` (the tensor). The adapter recomputes the second from the published
`head/head.safetensors` with a 20-line pure-stdlib reader and gets `aa21c427…`
exactly, confirming both values are right and that they answer different
questions. v1 requires both, names content normative, and makes comparing one
convention to the other a hard error (case H11).

The same distinction is why `determinism.evidence_hashes` may never contain a
file digest: `stream_score` writes `cold_run` into the safetensors
`__metadata__`, so **bitwise-identical runs produce different file digests**.
Case F15 proves it on constructed bytes; `reports/k6-five-run-kld.json` proves
it on real ones — five different receipt digests, one tokenwise digest,
population stddev exactly 0.0.

### Lesson 35: "shared head" means shared APPLICATION, not shared WEIGHTS

If a quant quantizes its own `lm_head` — stock EXL3 at `head_bits` 6–8, GGUF's
`output.weight`, MLX — then replaying its hidden states through the **reference**
head erases its head-quantization error and flatters it. kimi-k3's comparator
takes one `--lm-head` and applies it to both sides; nothing refuses, warns, or
records a mismatch. **Our own code has the same default live**:
`tools/fidelity.py cmd_replay` takes a required `--head` and an *optional*
`--candidate-head` that defaults to `None`.

The comparator now refuses that condition outright (HEAD-1b, exit 3). The
override exists but is expensive on purpose: `--disclose-head-substitution`
forces `class: advisory`, a bias block with `direction: downward`, and a
**blocking** disclosure, which under DISC-003 forces `status: pending`. A
head-substituted number is not publishable, which is the correct price.

### Validated without a GPU, and against real bytes where they exist

96 cases across three selftests, all offline, all on the system python3:

- `bin/selftest_fidelity_dataset.py` — **66 passed, 0 failed**. F1–F15 format
  and seal, P1–P9 panel binding, H1–H11 head identity, L1–L5 lane/stack,
  C1–C4 coverage, X1–X2 lossy, I1–I15 interop, R1/R4 real artifacts.
- `bin/selftest_fidelity_compare.py` — **16 passed, 0 failed**. Known-answer KLD
  against an independent plain-python oracle (agrees to fp64 epsilon), the
  self-compare exactness assertions, and the T1 constant.
- `bin/selftest_fidelity_card.py` — **14 passed, 0 failed**, including the live
  Hub `validate-yaml` axis on six cards.

Three of those deserve naming because they used real data, not fixtures:

1. **The superset proof.** `adapt --source malaiwah-serving-v2` turns our own
   published `glm53flash-fidelity-capture/2` into a conformant v1 dataset that
   passes `validate --verify-tensors` with 0 errors, and fixes rather than
   copies its three defects: the undeclared cut point (it now declares
   `after_final_rmsnorm_before_lm_head`, which is what the code actually
   implements), the manifest claiming `complete: true` over 5,120 captures in a
   512-file repository (honest coverage plus `shard_of`), and records of
   `{index, sha256, shape}` (full records with payload and content digests).
2. **A real self-compare.** The adapted BF16 root compared against itself, with
   `--force-compute` so the 2 × 2047 × 154,880 fp64 matmul and softmax actually
   ran through the real 154,880 × 4,096 head: **exactly 0.0**, top-1 exactly
   1.0, and the computed array bitwise identical to the hash proof.
3. **A real measurement.** The same two adapted captures — BF16 reference and
   the FP8 as-served capture — scored **0.035262 nats**, top-1 0.9257, on two
   contexts. The published full-suite headline over all 5,120 is 0.028104 /
   0.9427, and the context-depth buckets reproduce the known shape (0.147 at
   depth 0–255 falling to 0.0146 at 1536–2046, against the published
   positions-1024+ figure of 0.018794). The comparator's own gates put it at
   `class: advisory` because neither adapted capture carries a
   `lane_identity_sha256` — correct, and the kind of thing that used to be a
   footnote.

### Lesson 36: our adopted digest preimages are Festr's, and we can prove it

The spec adopts kimi-k3's two token preimages verbatim — compact
`separators=(",",":")` per record, `"\n".join(...)` for the aggregate. That
claim is now checked, not asserted: `adapt --source k3v1` reads his real
published `suite-manifest.json`, recomputes the aggregate from his own
per-record digests under our adopted rule, and gets his declared
`70cd72175fcb…` exactly. The window-form predecessor reproduces `a6856e1d…` the
same way.

And the divergence is equally real: our historical preimage (`json.dumps`
defaults) reproduces our published `token_sha256` `f26a50ad…` exactly, while the
adopted compact preimage gives `9f2fa28a…` for the same token array. Same
tokens, different hashes — a preimage divergence, not a naming one, which is why
v1 ends it by adopting his and carrying ours as `*_legacy`.

### Lesson 37: `huggingface_hub` silently drops nulls inside `args`

Found by the round-trip axis on a card we generated, not by reading docs. A
`metrics[].args` entry with a `null` value survives the Hub's validator but is
**dropped** by `EvalResult` on any library round-trip, so the card that ships is
not the card the Hub re-serves. The generator now omits null keys entirely
(GEN-9) and the validator refuses them.

The second card-level find was scope collision: the registry carries `panel25`
and `clean17` rows for the same artifact, panel and lane, which share
`huggingface_hub`'s five-tuple merge key `(task.type, dataset.type,
dataset.config, dataset.split, dataset.revision)`. Merging them would silently
discard one row's args — the same failure mode BIAS-006 forbids for lanes. Lane
lives in `split`; measurement scope now lives in `config`.

### What was NOT done, and why

- **`registry/` was not touched.** The three additive changes a step-3 receipt
  needs — two disclosure codes, a `registry_add` adapter, four invariants — are
  specified in [`docs/REGISTRY-INTEGRATION.md`](docs/REGISTRY-INTEGRATION.md)
  and deliberately not applied: the sequential measurement workflow holds
  `registry/schema/invariants.json`, `registry/data/*.jsonl` and
  `registry/tools/seed_registry.py` open in the working tree, and `make check`
  passed through an intermediate 11-error state while this was being built
  before returning to 62 passed / 0 failed. Editing a 90-invariant file another
  workflow is editing is how you get a merge conflict in the one place
  correctness is enforced. `git status registry/` shows none of this work.
- **The five reserved files and `stream_score.py` were not edited.** Every
  capture path is wrapped, imported or shelled out to, following the precedent
  `k6/tools/hidden_replay.py` set. The fp64 estimator in the comparator IS
  `k6_kld_report._token_kld`, imported and called, so a number here equals a
  number from the sealed pipeline.
- **Nothing was published.** The tooling can `--publish`, and it refuses to
  unless `verify --verify-tensors` passes first, the dataset is not `draft`, and
  the fetched copy re-verifies. Publishing the annotated K6/K8 cards is the Ship
  phase's job, and it should be preceded by one push to a private scratch model
  repo — live *rendering* of the eval widget is the one axis we cannot check
  from here.
- **A fifth invariant, DS-005, is specified but not applied.** A floor from a
  different measurement SCOPE is not this row's zero-point, for the same reason
  a floor from a different lane is not. This was live in the registry while the
  work was in flight and the other workflow has since resolved it; nothing in
  the schema prevents it recurring. The guard exists at card level today
  (`attributable_refusal`, case K8b) and withholds the number rather than
  printing an unverifiable one.

---

## 2026-08-29 ~20:30 — Joint standard with brandonmusic: adopted, quantified, and one lesson we did not want

brandonmusic proposed a **community standard for quantization-fidelity
measurement** and published the whole harness — protocol, receipts, plots, CLI,
and the sealed 25-window panel plus BF16 teacher logits our entire campaign is
measured against. He already cites our K6 number (0.0137) as a reference point.
The operator asked for alignment, not competition. This entry records what we
took, what we measured, and the one finding that cost us a number.

### He is ahead of us, and the honest count is eight of eighteen

Element-by-element in [`docs/PROTOCOL-ALIGNMENT.md`](docs/PROTOCOL-ALIGNMENT.md):
**eight elements adopted from him** (they are the core of the standard), four
already equivalent, five ours, three genuine divergences. Adopted: the R0 canary
in both halves, per-domain stratification, the 13-gram calibration-overlap scan,
`sigma_run` combined with the statistical SE in quadrature, the window-block
bootstrap with BCa intervals, McNemar on paired top-1, the percentile-exceedance
guard, and one frozen protocol file whose hash is stamped into every emitted
receipt. Implemented clean-room in `bin/jointstd/` (his licence is
source-available with a named exclusion; no `kld_eval` source is vendored), and
validated against his own published endpoints as the known-answer test — plus,
where importable, against `kld_eval` itself as an oracle.

Two of his findings we can now confirm rather than take on faith. His
`sigma_run` is **exactly 0.0** across three cold boots on a deterministic kernel
path, and **not** zero on the NVFP4 MoE path (0/25 windows bitwise, 94 % of
tokens changed, single tokens swinging 4.7 nats while the mean barely moves).
That is the same bitwise-determinism property our reference lane has, measured
independently on different hardware. **Determinism is a property of the kernel
path, not of the quant.**

### Divergence 1: the padded lm_head columns — bounded, not argued

He masks the 24 padded columns of the 154,880-wide head on both sides; we never
have. We downloaded one 1.27 GB teacher window (sha256 verified) and measured
it. The padded rows are not dead — norm ≈ 0.4795 against 1.209 for a typical
real row, all 24 mutually cosine-0.999998, one untrained direction repeated —
but they hold **~1.6e-8 of the teacher's probability mass**. The exact identity
makes every term proportional to that mass, so the general cap is **order 1e-8**
and, where teacher and student share the head (every malaiwah row on his panel),
it collapses to `KLD × mass` = **1.0e-10 nats**. Our published values move at
their **9th significant figure**. For scale: our own sealed-vs-streaming bridge
is 8.5e-6 and his window-clustered SE is 3.19e-3 — 83,000× and 31,000,000×
larger. **No correction and no bias disclosure; a protocol-policy disclosure
only.** We adopt masking anyway, as a zero-cost convergence.

The first version of that section shipped results with no script and no receipt,
in a document whose thesis is receipts over assertions. That was caught in
review and is now fixed: `bin/padded_column_study.py` plus receipts in
`docs/joint-standard/padded-column/`. The half that carries the conclusion needs
**only his published window** — no `lm_head`, no reconstruction, seven seconds —
so anyone can check the teacher side without trusting us.

### Divergence 2 — and this is the lesson: calibration bleed survives document dedup

His 13-gram overlap scan found that **an entire domain of the sealed FINAL
windows shares 37–39 % of its 13-grams with calibration-role windows**, despite
the panel being clean at the *document-hash* level. Our panel-building has
always checked document hashes. **Document-hash dedup is not enough, and we had
no n-gram check at all.** We had flagged the absence of one as a disclosure; we
had not run one, and we did not predict what it would find. The credit for the
finding is entirely his.

Our scan reproduces his selection **17/17 exactly**, 0 mismatches on shared-gram
count and fraction across all 25 windows. Every number we published on that
panel used all 25 windows and therefore carries the same contamination.
Recomputed on his clean scope from our own published per-window arrays — no GPU,
no re-measurement, pure arithmetic on data already public:

| | panel25 | clean17 | move |
|---|---:|---:|---:|
| K6 sealed | 0.013723 | 0.011677 | −14.91 % |
| K8 | 0.012384 | 0.010829 | −12.55 % |
| official FP8 | 0.020615 | 0.018665 | −9.46 % |
| BF16 floor (x-stack) | 0.012712 | 0.010648 | −16.24 % |
| his 4bpw | 0.024555 | 0.024949 | **+1.61 %** |

**Fifteen percent is not a rounding error.** The conclusions survive: K6 beats
the official FP8 on 17/17 clean windows and the margin *widens* (1.50× → 1.60×).
K8-over-K6 survives but weakens — the paired BCa lower bound falls from +0.000695
to +0.000153, sign test p 0.0041 → 0.049. **We will not restate "K8 is better
than K6" without naming the scope.** Both model cards now carry a prominent
scope disclosure; `reports/clean-scope-recompute.json` is published.

The result worth keeping: FP8 falls 9.46 % and the floor falls 16.24 %, but
**FP8 minus the floor rises 1.44 %**. The subtraction is the stable quantity and
the raw means are the unstable ones — a measured answer to his §5.3 objection to
publishing subtracted numbers.

### What we offer back

The measured **BF16 floor** and attributable-error framing; a schema-enforced
**registry** with mechanical refusals (90 invariants, stock interpreter, no pip
install); **eight decode surfaces** (checkpoint, payload-store, dione, native,
exl3hf, mlx, gguf, nvfp4) so one yardstick spans formats; a measured cross-lane
bridge; and R0-b — the shift half of the canary, which R0-a provably cannot
replace (a consistently-shifted pair scores exactly 0.0 on R0-a).

### Lessons

23. **Document-hash dedup does not detect calibration bleed.** Overlapping
    n-grams survive it. Every future panel gets a 13-gram scan against its own
    calibration material *before* anything is measured on it, not after.
24. **A scope is part of a number.** A different window set is a different
    panel: `clean17` now has its own comparability key so a clean17-vs-panel25
    difference is structurally impossible rather than merely discouraged.
25. **A document with no mechanical tie to its receipts drifts.** Five wrong
    numbers reached the alignment documents while `make check` was green,
    because nothing checked them. `bin/check_doc_numbers.py` now re-derives 201
    anchored claims from the committed receipts and runs in `selftest_all.sh`.
    On its first run against the *repaired* documents it found a sixth (a
    per-domain ratio printed as 1.303× where the receipt says 1.302×).
26. **Write the receipt for your own side-study too.** The padded-column section
    argued for receipts while having none.

### What was NOT done, and why

- **The registry mirror was not pushed.** No row data changed (`git diff
  registry/data/` is empty and `render-check` passes), so the only delta is the
  JOINT-009 declaration — while a concurrent workflow holds
  `registry/schema/measurement.schema.json`, `submission.schema.json`,
  `registry_add.py` and `registry_validate.py` open mid-change. Pushing the
  mirror now would publish someone else's half-finished work. It goes after that
  workflow lands.
- **`uncertainty.ci95_total` was not added to the registry schema.** A live
  `sigma_run` needs a total-uncertainty interval, because the block bootstrap
  resamples windows *within* one run and can never contain run-to-run spread.
  It is implemented in the analysis receipts, but the schema field was not added
  — same concurrently-held file. Instead **JOINT-009 fails closed**: every
  `sigma_run` in the registry is exactly 0.0 today, so it passes vacuously, and
  the first row measured on a nondeterministic path fails until the field
  exists. An enforced TODO beats a comment.
- **No LICENSE file was added.** The repo still ships none, so every offer we
  make him is legally unusable as it stands. Choosing a licence interacts with
  his SHAPLEYMCG-1.0 terms and its named exclusion, and it is the operator's
  call, not a supervising session's. §10 lays out the analysis; the reply admits
  the gap. **This is the one blocking action item before the reply is sent.**
- **Nothing was posted.** `docs/ALIGNMENT-REPLY.md` is ten paste-ready Discord
  messages (longest 1,993 of 2,000 characters, checked mechanically). The
  operator relays personally.

## 2026-08-29 ~16:30 — Three-step architecture shipped: review findings closed, cards published

Three adversarial reviews returned `GO_WITH_FIXES` on the fidelity-dataset work:
one attacking the comparator with its own fixtures through the real CLI, one
testing interop against Festr's real artifact and the live Hub, one walking the
path as a stranger. Six blockers and eight majors. All fixed; the two that could
not be fixed here are descoped in writing below.

### The head trap was still open — HEAD-1c

The best finding, and the one that most vindicates the format's own premise.
`classify()` decided `reproduction_confirmation` on `capture_content_digest`
equality **alone**. A quant that changes only `lm_head` — stock EXL3 `head_bits`
6–8 does exactly this — changes nothing before the final norm, so its post-norm
hiddens are *bitwise identical* to the reference's and the content digests
match. The comparator then replayed both sides through one head, subtracted a
quantity from itself, and reported **0.0 nats at top-1 1.0, labelled an exact
reproduction**. `--force-compute` "agreed" vacuously, because `compute()` builds
one `head32_t` and replays both sides through it: the recomputed array is also
all zeros.

The fine print stayed honest the whole time (`head_digest_equal: false`,
`usable_as_floor: false`, a blocking disclosure, SC-3 keeping it out of the
registry). The headline was wrong, and the headline is what gets quoted. That is
precisely the flattering erasure §8.1 exists to prevent, arriving through the one
door HEAD-1b left open — because HEAD-1b's override is right when the two
captures *differ*, and this is the case where they cannot.

`head_substitution_vacuous` now refuses it with **no override at all**. Bitwise-
equal hiddens under different heads means the head IS the whole difference, and
hidden replay erases exactly it; there is no reading under which the number means
anything. A head-only quantization must publish **logit form**, where each side
runs its own head (HEAD-2). Case N12.

### The tokenizer was outside panel identity — PANEL-D6

`suite_token_hash_sha256` hashes token **ids**, which are integers. Two
tokenizers can emit the same ids from different text; one applying a chat
template has scored a different corpus with the same numbers. A candidate
declaring `tokenizer.id: "a-completely-different-tokenizer"`, a different
repository, revision, `add_special_tokens: true` and `chat_template_applied:
true` — sealed honestly, `validate` 0 errors — sailed through at
`class: strict`, and the receipt then published the *reference's* tokenizer block
as if it were the comparison's.

The panel gate now compares tokenizer identity, treats a null on either side as
*unknown* rather than agreement, records **both** sides in the receipt, and — the
part worth noting — **found a real defect in our own adapter on its first real
run**. `adapt_serving_v2` was filling `tokenizer.revision` from the captured
artifact's *model* revision, so our BF16 root and our FP8 candidate, captured on
the same panel, declared different tokenizers. The tokenizer belongs to the
panel; it now comes from the suite manifest's own tokenizer-snapshot pin.

### `--emit-submission` wrote a file the registry rejected, and exited 0

The "registry-submittable proof package" deliverable was not met, for three
independent reasons inside the new code: `artifact`, `panel` and `reference` were
hard-coded to `{}` with no flags to fill them; `evidence[]` used a `kind` that is
not in the source enum, an additional `role` property, and no `uri`; and the
determinism block carried three keys `submission.schema.json` forbids. The
author's own selftest validated receipts against the *new* receipt schema and
never against the registry's submission schema — that was the coverage gap.

Fixed on all three axes, plus `provenance-template` for the identities a dataset
genuinely cannot know (a 40-hex artifact revision, `panel_ref`/`reference_ref`
that must already exist), plus **SC-4** refusing to write empty blocks, plus the
CLI running `registry_validate.py --submission` on its own output before telling
anyone the file is submittable. Case N16 runs that gate in the selftest: it is
offline and costs nothing, which is why it should have been there from the start.

`comparability.bias` and `usable_as_floor` were also being dropped on the way
across. A row derived from a head-substituted comparison arrived with
`bias: null` for a comparison whose own receipt declared it biased **downward**,
because `registry_add` synthesises a bias from `stack_relation` alone and
`stack_relation` cannot see a head substitution. An optional, additive
`comparability` block on the submission now carries the comparator's verdict, and
`registry_add` prefers it when present.

### A documented protection that did not exist

`docs/REGISTRY-INTEGRATION.md` said a receipt carrying `head_substituted` was
"currently unsubmittable — which is the safe direction to be wrong in", and used
that to justify deferring the registry changes. It was **false**. DISC-003 and
DISC-004 live in `check_disclosures`, which iterates the registry *collections*;
`--submission` runs `check_submission`, which calls neither. A structurally valid
submission carrying a blocking disclosure is ACCEPTED at exit 0 by the gate
`CONTRIBUTING.md` tells submitters to run.

The response is not to lean harder on a downstream gate: `emit_submission` now
refuses a blocking disclosure itself (**SC-5**), and the document says what the
registry-side protection actually is and when it fires. A deferral justified on a
protection that does not exist is worse than no deferral.

### Tensor verification is the default now

`verify` and `compare` recomputed `checksums.txt` and the manifest seal but not
per-tensor `tensor_content_sha256` unless asked. A byte flipped inside a tensor,
with `checksums.txt` and the seal refreshed honestly afterwards — which is what
re-running finalize after an edit does — was **scored silently**: 2.098 nats
where the honest answer is 3.688, a 43% error at `class: strict`. Worse, the
receipt hard-coded `source_file_hashes_verified: true` on every run regardless.
Verification is on by default in `verify`, `validate` and `compare`
(`--no-verify-tensors` opts out for 86 GB suites), and the receipt now records
the boolean that actually ran. Case N17.

### Interop: two claims made true, one number worth keeping

`--emit-k3-compat` was specified in the spec and did not exist; `adapt --source
k3v1` had exactly one return path and never built a dataset even at 3-of-3
tensors present. Both now ship, and both were proven against Festr's *unmodified*
comparator rather than described:

- A kimi-k3 artifact → sealed v1 dataset → self-compare **exactly 0.0 nats**
  under `--force-compute`. The full round trip, foreign artifact to answer.
- `compat/` costs **zero duplicated bytes**. The reference shim written during
  review hardlinked every tensor and token (86 GB → 172 GB); his loader resolves
  `directory / record["file"]` with pathlib, so relative aliases onto the one
  real tensor work. Three JSON files at any panel size, written before the seal
  and listed in `checksums.txt` like anything else.

Two honesty constraints the emission forced, both refusals a reader would not
predict: **`--role root` is refused for a k3 translation** (ROOT-1 asserts
`head.quantized: false`, which that artifact never records — that is D-1 — and
its own checkpoint string says "official MXFP4 routed experts"), and
`weights.quantized` is read out of that string with the string named in
`inferred_fields`, because the schema wants a boolean and there is no honest null.

And the number: same bytes, same panel, same head, same direction, two careful
implementations.

| | mean KL(ref‖cand), nats | top-1 |
|---|---|---|
| his comparator, fp32 log-probs, `clamp_min_(0)` | 0.03564599129280951 | 0.925256472887152 |
| ours, fp64 throughout | 0.03526219355348638 | 0.9257449926722032 |
| difference | **3.84e-4 (+1.09 %)** | 4.9e-4 |

That gap is **larger than the entire quantization-attributable signal we publish
for K6** (0.00221 nats). It is the empirical case for D-5:
`estimator.accumulation_dtype` in the comparability key is not pedantry.

### The card generator could not reproduce its own output

Re-running the documented command regenerated the K6 card with five fields null
that the committed card carries, because a human had passed five extra flags
whose correct values were not discoverable from the tool. For an outside quant
author that is the single largest adoption friction: the documented command
silently produces a weaker card than ours. `reference_model` and
`reference_revision` are now derived by walking measurement → `reference_ref` →
`artifact_ref` → `huggingface.{repository, revision}`, and both cards now
regenerate **byte for byte** from the registry alone. Fields nothing supplies are
warned about by name with the flag that would supply them, instead of written as
silent nulls.

`annotate` also always validates its own output and exits non-zero rather than
writing an invalid card — it used to write an invalid `--role fidelity-dataset`
card and exit 0. That role is now built *from the dataset*
(`--fidelity-dataset-root DIR`), which is the one card a standalone capture
publisher needs and the one the generator could not emit.

### Registry: applied, not deferred

`registry/` was clean when this ran, so the three changes previously specified
are now in: two disclosure codes, an optional `comparability` block on the
submission (plus `usable_as_floor` on the measurement), and **BIAS-007** — a row
stamped `usable_as_floor: false` may not be named as any other row's
`floor_measurement_ref`, the registry honouring a verdict the comparator already
reached. `make check`: 62 passed, 0 failed, 433 joint checks, 0 errors.

One real bug surfaced there too: a `cross_stack` comparison emitted the bias
block `measurement.schema.json` rule 4 requires but **not** the
`cross_stack_capture` disclosure it also requires, so every cross-stack row was
schema-invalid on arrival. Fixed in the comparator, not the schema — the schema
was right.

### Deferred, and said plainly

- **measure-cloud integration is out of scope and stays out.** The five reserved
  files (`measure_cloud.py`, `stage_measure.sh`, `hfmeta.py`, `engines.json`,
  `invoke_engine.py`) and `k6/tools/stream_score.py` were not edited; `git diff`
  against `origin/main` is empty for all six. Every capture path is wrapped,
  imported or shelled out to, and the fp64 estimator IS
  `k6_kld_report._token_kld`, imported and called.
- **A `registry_add` adapter for the comparison receipt is still deferred**, now
  with a better reason than a workflow conflict: the submission path works end to
  end and is the supported one. The receipt is the *evidence*; the submission is
  the *claim*. A second ingest path would be two places mapping a number onto a
  row.
- **No root fidelity dataset and no token panel are published**, so steps 1 and 2
  cannot be started from a clean checkout. This is the honest state and it is now
  stated at the point of use (`bin/README.md`) and in `WHAT-WE-MEASURE.md` §8
  rather than buried in an out-of-scope table. The 85.9 GB is the easy part;
  deciding what a canonical root *is* is an operator decision.

### Cards: verified and staged, NOT pushed

The K6 and K8 cards are ready to publish and everything that can be checked
without pushing has been:

```
body byte-identical to the LIVE card ..... yes, both (only the YAML changes)
live Hub POST /api/validate-yaml ......... PASS, both
huggingface_hub 1.29.0 round-trip ........ PASS, both (6 and 3 eval results)
our XC-1..XC-5 against the registry ...... PASS, both
HOSTPATH-1 scan .......................... clean, both
GLM-5.3-Flash-TR3-6bpw.README.md ......... sha256 6a4a0f2f46d1edc1…
GLM-5.3-Flash-TR3-8bpw.README.md ......... sha256 08abe1de095f91d8…
```

**The push itself was not performed.** Publishing to a public repository is a
permissioned act and the permission has to come from the operator, not from the
workflow that scheduled the work — so this session prepared the exact bytes,
verified them against the live Hub, and stopped. One command finishes it:
`python3 <scratchpad>/ship/publish/publish_cards.py --push`, which re-checks
each file against the digest above, refuses if the bytes are not the ones that
were validated, and re-fetches both cards afterwards to confirm.

Both cards had leaked `x_fidelity.registry.snapshot.root: /Users/mbelleau/…` —
the same defect class as the dead `packed_root: /home/jl_fs/…` that motivated
this entire format, and the Hub's own `validate-yaml` accepts it, so no external
check would ever have caught it. `HOSTPATH-1` now walks the entire front matter
against an anchored host-path regex. No capture was published; nothing large was
pushed anywhere.

Battery: `selftest_fidelity_dataset` 69/0, `selftest_fidelity_compare` 25/0,
`selftest_fidelity_card` 16/0, `selftest_all.sh` 42 passed / 0 failed / 3
skipped, registry `make check` 62 passed / 0 failed + 433 joint checks / 0
errors. On real data: BF16-vs-FP8 **0.03526219355348638** nats at top-1
**0.9257449926722032** over 4,094 positions through the real `[154880, 4096]`
head, tensors verified; self-compare exactly **0.0** with `--force-compute`
agreeing bitwise (tokenwise sha256 `d54931c81433dc5d…`).

## 2026-08-29 — measurement 1 on the turnkey runner: turboderp 4.05bpw, and the
## eleven things that stopped `bin/measure-cloud` from ever reaching a GPU

`bin/measure-cloud` was built, reviewed three times, and had never once been run
for real. This is the entry where it was, against
`turboderp/GLM-5.3-Flash-exl3` @ `2a30229e` (branch 4.05bpw) on the sealed
25-window panel. Everything below is a defect the runner had, found in the order
a stranger would have found it, and each one is now closed.

**THE RESULT** — turboderp/GLM-5.3-Flash-exl3 @ 2a30229e (branch 4.05bpw), sealed
25-window / 51,175-position panel, streaming lane, 1x H200 spot, EP8 emulated,
--reduce-order fp32:

    mean tokenwise KLD (teacher||student, fp64)   0.025526426915472484 nats
    2 cold runs, run means IDENTICAL, 1 distinct tokenwise_kld_sha256
    attributable vs the streaming BF16 floor      +0.014020504296142185 nats
    907,200 K4 expert matrices decoded per run

Same-lane, same-panel, same-teacher company (comparability key
cmp--202b717f3219c414, byte-identical to K8's):

    BF16 floor      0.011506
    K8   8bpw       0.012384    attributable 0.000878
    K6   6bpw       0.013715    attributable 0.002209
    turbo 4.05bpw   0.025526    attributable 0.014021   <- this row

Its attributable error is 6.3x K6's at two thirds the bit rate -- and it is not
lineage-comparable with a 4-bpw quant made from BF16, because this one was made
from the FP8 release (its own config says so) and carries that parent's
divergence too.

THE SECOND BRANCH WAS REFUSED, FOR $0.00. turboderp's 3.05bpw branch
(332ab457) omits 22 tensors the official model and its own 4.05bpw sibling both
carry: `self_attn.indexer.index_kpool_compress_{ape,gate}` on all 11 MLA
layers. transformers would have randomly initialised the sparse-attention
indexer's k-pool compression and the number would have described a model nobody
has. The planner reads it out of two index files in seconds.

**Found before a single instance existed** (dry-run only, $0.00):

1. `sniff_surface` classified a stock-exllamav3 release only inside the
   `elif "config.json"` arm. turboderp ships BOTH an inline block and a
   standalone 47.9 MB `quantization_config.json`, so the first arm ran, the
   codec parsed correctly, and the surface stayed `unknown` — the runner
   refused an artifact it could read. Where a publisher puts the block is not
   a format property; classification now happens after parsing, whichever file
   it came from, and the 86 KB inline block is preferred over the 47.9 MB file.
2. `invoke_engine` had no `--source` spelling for the exl3hf surface.
   `build_invocation` drops a flag whose value is empty, so the engine would
   have run with its default source and died on argparse an hour into the
   rental. An unmapped surface is now a refusal, not a default.
3. The same file pointed `--bf16` at the official BF16 metadata skeleton
   instead of the tree the materialize stage writes, and passed none of the
   `--exl3hf-*` pins.
4. `invoke_engine` never prefixed an interpreter. The engine scripts are mode
   644 with an `env python3` shebang, so the measure stage could not exec at
   all — and executing, would have used the system python without torch.
   `measure_local` had this line; the cloud path did not.
5. There was no scoring stage. `stream_score` CAPTURES logits; the divergence
   comes from `k6_kld_report` across the cold runs. `seal` looked for
   `run-*/kld-report.json`, found none, and exits 2 — with the whole rental
   spent. Added `bin/invoke_scorer.py` and the `score` stage.
6. `seal_receipt`'s rollup searched flat keys only, and `k6_kld_report` nests
   the per-run mean at `summary.mean`. It would have sealed a NULL metric.
7. `setup` delegated to `k6/stage_k6.sh`, which was never in `BUNDLE.txt` — so
   the stage ran a file that does not exist on a cold box — and whose setup
   hard-stops on an ENCODER closure gate (ShapleyMCG's r10 codec) that a
   decode-only measurement has no business depending on.
8. The bundle shipped neither `exl3hf_surface.py` nor the `patches-v2` series.
9. `census.storage_need` counted the students' fp32 logits only with
   `--keep-student-logits`, but two cold runs hold both trees before the report
   seals. The proof-target sizes to 400 GB, not 300 (lesson 31 again).
10. `k6_kld_report` suffixed every profile `-tp4`. Stock releases are canonical
    HF shards, not TP4-sliced — a false storage claim in the headline receipt.
11. `exl3-mul1` was not in the registry's `numeric_format` vocabulary, so the
    correct codec string would have been rejected at submission time, after the
    run.

**Found by spending money** (each one cost a restart, none cost a measurement):

12. `jl run <command> --on <id>` is not how `jl run` takes a command: the first
    positional is a TARGET (a file to upload), and a bare command goes after
    `--`. So EVERY stage of every cloud run had always died the instant it was
    launched — `Target does not exist` — with the instance already billing.
13. The double-spend guard adopted an existing instance only on an EXACT name
    match, but the name embeds a deadline computed from "now". A restarted
    controller therefore created a SECOND instance and a SECOND 400 GB
    filesystem and left the first billing. It matches the job-id prefix now.
14. **The done-marker probe was matching its own command text.** The probe is
    `test -f <marker> && echo DONE || echo PENDING`, and the result was tested
    with `"DONE" in str(response)` — but `jl exec --json` echoes the command
    back in its payload. The test searched its own words and could never answer
    anything but yes. Observed live: `stage setup ok 2m07s` against an empty
    receipts directory, while setup was still installing torch, and the
    controller went on to launch `fetch_target` CONCURRENTLY with the setup it
    depends on. Left alone it would have marched through measure and seal and
    produced a receipt from a box that had done nothing.
15. `_await_stage` decided a stage's fate by grepping the last 40 log lines for
    "Traceback". A stage that exits non-zero quietly looked exactly like one
    still working, and the controller would wait until `--max-runtime` and pay
    for the whole window. It reads `jl run status --json` now.
16. Cheapest-that-fits silently swapped an A100-80GB in for the H200 the
    streaming lane's rows were all measured on, and only WARNED. Both constants
    the plan runs on — minutes/window and the observed VRAM peak — are measured
    on H200, and bf16 kernels differ across architectures, so the row would not
    have been same-lane either. Caught by eye one minute into a paid run;
    now a refusal.
17. The lease named only the instance. On an ADOPTED box the filesystem id was
    never learned, so the reaper could have stopped the compute bill and not
    the 400 GB volume behind it, which bills on its own.
18. The 49-file bundle re-uploaded in full on every adoption — ~10 minutes of a
    billing H200 for files already byte-identical on the box. One `sha256sum`
    over the remote paths answers it in a single call.

**Two things that went right and are worth keeping:**

* `--hold-on-failure` (added after defect 12 destroyed a box with a finished
  bundle on it) turned every later failure into a resume. The setup work,
  the fetch and the materialized tree survived three controller restarts.
* The bootstrap probes instead of assuming: exllamav3 was NOT built, because
  the pipeline imports cleanly without it. Neither `stream_score` nor
  `k6_kld_report` imports the package. That is ~20 minutes of CUDA toolkit and
  extension build the measurement never needed — and the probe, not a belief,
  is what decided.

**The artifact, honestly described.** turboderp's release is FULL-scope: the
attention, the dense MLPs, the shared experts, the vision tower and the lm_head
are all EXL3, at per-class rates the release publishes itself
(experts K4, attention K6, shared experts K6, the three dense MLPs K5, head K6; embed, norms, router and the small KDA projections native). Only `embed_tokens`, the norms, the router and the small KDA
projections stay native. So the measured function is the ARTIFACT ALONE: the
materialize stage dequantizes its own non-routed tensors — including that 6-bit
head — into an official-layout BF16 tree, and no official-release weight enters
the path. `head_policy` on the measurement stays `native_head` because that
field says how the head is APPLIED (natively, from the artifact's own weights,
no shared replay); the artifact record carries `head_policy: quantized` and a
`quantized_head` disclosure so the two are never confused.

It is also quantized FROM the FP8 release, not from BF16
(`original_quantization_config.fmt = e4m3`), which the artifact record
discloses: the number includes whatever the FP8 parent already cost
(the campaign's own FP8 row is 0.0281 nats on a different panel/lane).

LESSON 36 (probes, not beliefs): a status probe whose OUTPUT can be confused
with its own COMMAND is not a probe. `jl exec --json` echoes the command it
ran; any test that stringifies the whole response is reading its own words
back. Read stdout, compare the last line exactly, and let the thing under test
be the only thing that can answer.

LESSON 37 (a warning is not a control): the GPU-selection warning was correct,
prominent, and completely ineffective — the run proceeded on the wrong silicon
and only a human reading the scroll stopped it. When a choice invalidates the
numbers the plan is built from, it has to refuse and name the flag that
overrides it.

LESSON 38 (a bootstrap belongs to the lane that needs it): reusing the
encoding campaign's setup for a decode-only measurement imported a closure
gate, two extra clones and a hard stop, none of which have anything to do with
the number. Copy the proven pins; do not inherit the proven program.

**Cost, four ways.** 1x H200 spot at $1.99/h. The measurement box lived 4.03 h
(bootstrap 4m13s, artifact fetch 6m22s for 165 GB at ~430 MB/s, materialize
2m06s, two cold runs 2h38m, score 4m21s, seal) = $8.02, plus $0.86 of earlier
boxes killed while proving the path and ~$0.3 of filesystem, so ~$9.2 all in
against a $14.95 point estimate. The estimate was high for one reason worth
recording: the planner prices with the K6 payload store's MEASURED 7.35
min/window, and this surface ran at 3.12 -- K4 payloads read straight from the
artifact's own HF shards. The planner keeps 7.35 until a second exl3hf artifact
confirms the figure; over-estimating the clock is the safe direction.

**Balance** $175.97 -> $124.41 over the session, of which ours is ~$9.2; the
rest is other sessions' boxes on the shared account.

## 2026-08-29 — measurement 2: the tr3-published reader, and an artifact that turned out to be a mirror

**The target was not what the mission said it was, and that was findable for
$0.00.** Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw was scheduled as "the direct A/B
against brandonmusic's 4bpw — same nominal rate, different quantizer". It is
not a different quantizer. Its `MIRROR.json` says
`"Byte-identical redistribution. Not an original quantization."`, and two
independent checks agree: all 120 `*.safetensors` have the same LFS oid as
`brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8` (only README.md differs among
their 142 shared files), and the mirror republishes brandonmusic's own
`SHA256SUMS` verbatim — against which all 120 downloaded shards verified
byte-wise on the instance.

That does not make the measurement less valuable; it makes it a different, and
better, measurement:

* it is the first row for these weights on OUR streaming lane, and the first
  measured by anyone other than their author — brandonmusic's own 0.024555 is
  an author-reported five-run number on his sealed EP8 stack;
* it is the campaign's only cross-lane bridge over IDENTICAL BYTES; every other
  lane comparison changes the weights too;
* the real same-rate A/B is against M1's turboderp 4.05bpw (0.025526, same
  lane, same panel, same teacher) — same nominal rate, different quantizer,
  opposite ends of the scope axis, and different lineage.

**THE NUMBER.** 0.02550342763436377 nats, mean of two cold runs whose run means
are identical to the last bit and whose tokenwise KL arrays share one sha256
(31177f24...). Top-1 agreement 0.9531411822178798, also identical across runs.
Against this lane's own BF16 floor (0.011505922619330299) the
quantization-attributable error is 0.013997505015033470 nats.

Two comparisons follow, and they are different questions.

SAME WEIGHTS, DIFFERENT LANE AND STACK. brandonmusic measured these exact bytes
at 0.024554564249958208 on his own sealed EP8 stack over five cold runs. The
bytes are provably the same, so the +0.000948863 nats between us is a
LANE-PLUS-STACK offset — his reader digest is 1fb3be87, ours 1ccce446 — and it
BOUNDS the lane term rather than measuring it. It is the only pair in the
registry where the weights are held fixed across lanes; every other lane
comparison changes the weights too.

SAME LANE, SAME RATE, DIFFERENT QUANTIZER. turboderp's stock-exllamav3 4.05bpw
reads 0.025526426915472484 on this exact lane, panel and teacher. TR3 is
0.000023 nats tighter — and on a panel whose window-block SE for a 4-bpw row is
about 0.0038, that is not a difference at all. That is the result worth stating
plainly: two 4-bpw quants built on opposite scope policies (routed-experts-only
with a native BF16 head versus full-scope with a 6-bit head), different
codebooks (mcg versus mul1) and different lineages (from BF16 versus from the
FP8 release) land indistinguishably close on this panel. Anyone choosing between
them on these numbers is choosing on noise. The scope and lineage disclosures on
the two rows are the real difference, and both rows carry them.

**The reader.** `tr3-published` had been refused at plan time since the suite
existed. The missing piece was never decode math: a TR3 release stores
`<module>.{trellis,suh,svh,mcg}` in canonical HF shards — the same objects
`exl3hf_surface` already reads, with the same frozen campaign MCG LUT the K6/K8
rows were measured through. `k6/tools/tr3_surface.py` composes that module for
every byte of the codec path and owns only three things: the scope
(routed-experts-only, native BF16 head), the names (official throughout — none
of the exl3hf fusion remaps apply), and the SEAL.

**The seal is the part worth keeping.** This is the only third-party surface in
the suite whose publisher seals it, and the seal REPRODUCES: the materialization
receipt's own `receipt_sha256` recomputes from its canonical content, its
`config_sha256`/`index_sha256` match the published files, the ABI's
`output_tensor_names_sha256` recomputes as sha256 of the canonical JSON of all
150,226 sorted tensor names, the plan digests agree, the count algebra closes
(4 x 37,152 = 148,608 payload objects + 1,618 natives = 150,226), and the 1,618
non-routed names are EXACTLY the official release's. Twelve claims, all
checkable for a few hundred kilobytes — so `measure-cloud` checks them at PLAN
time, before renting anything, and the box re-checks them against the downloaded
bytes at minute ten. Those rows carry a `sealed_source_verified` disclosure
where the exl3hf and Dione rows carry `unsealed_source`.

**Four defects, all caught before they could reach a number.**

1. `sha256sum -c` is the wrong instrument for a mirror. The mirror trims
   brandonmusic's 192 `.materialization/shards/*.json` sidecars and writes its
   own README/LICENSE while copying his SHA256SUMS verbatim, so `-c` reported
   122 failures — none of them a weight — and exited 1. Under `set -o pipefail`
   that killed `fetch_target` after a 175 GB download and a full checksum pass.
   `bin/verify_published_sums.py` asks the narrower, stronger question instead:
   every present file the list covers must match (fail-closed for weights),
   every weight on disk must be COVERED by the list (a hole `-c` leaves open),
   and entries naming files this repo does not publish are REPORTED by name.
2. A TR3 release's non-routed tensors cannot serve transformers from the
   artifact's own shards. They ARE the official ones — but they are interleaved
   with 148,608 routed payload objects across the same 120 shards, and
   transformers derives its checkpoint key set from the shard FILES, not from
   the index. The symlink view therefore reported 54,272 routed payload tensors
   as unloaded and the load gate refused. `--source tr3` now takes `--bf16`
   from the same materialize stage exl3hf uses; for a TR3 release the
   materializer decodes NOTHING and re-shards the natives verbatim
   (`native_copied: 1618`, `decoded: 0`, dtypes preserved).
3. My own: a multi-edit script failed partway and I trusted its summary instead
   of the file. Seven edits had not landed — including the one that pointed
   `model_root` at the materialized tree — so the runner composed the right
   argv, the materializer wrote the right tree, and the engine built its view
   over the artifact anyway. Re-applied with a per-edit applied/failed ledger,
   then grepped for.
4. Hiding behind (3): `prepare_nonrouted_view` reused its view directory across
   runs but only rewrote the INDEX, leaving symlinks from whatever root built
   it first. A view built from root A and reused against root B is A's shards
   with B's index, and it surfaces as an inexplicable load error several stages
   downstream — exactly where it surfaced. The view now carries a
   `.view-source.json` stamp and is rebuilt when the source changes.

**A registry correction that came free with the reading.** Three artifact
records — our K6 and K8, and brandonmusic's 4bpw — shared a scope helper that
said `attn.qkv`, `attn.o` and `mlp.{gate,up,down}` were quantized at the nominal
rate. They are not. All three are routed-experts-only, and the evidence had been
sitting in the artifacts' own metadata (`scope: glm53_routed_experts_only`,
`non_routed_dtype_policy: official_source_native`, `native_tensor_count: 1618`)
and in our own `k6/K8-ANOMALY.json` test_6_scope. The wrong digest mattered: it
made those rows look scope-comparable with the stock-exllamav3 rows on the same
panel, which really do quantize attention at K6 and the head at K6. Corrected,
with a `scope_record_corrected` disclosure on each saying what it used to say.

LESSON 39 (rehearse the last stage first). Three of M2's four expensive defects
sat in stages that run AFTER the money is spent -- the load gate at the start of
`measure`, and the schema gate inside `seal`. Two of them were found by
REHEARSING those stages offline against real artifact metadata and a synthetic
summary, before the runs finished: the scope object carried two keys
artifact.schema.json forbids, and its kv_cache_dtype disagreed with the
registry's existing record for the same bytes (exit 7). A stage you have never
run is not a stage you have; the cheapest place to run it is on the laptop, with
the real inputs and a fake number.

LESSON 40 (a multi-edit script must be verified against the file). An edit
script that applies eight substitutions and asserts on each one either applies
all eight or, if it raises before writing, applies NONE -- and its console
output looks the same either way if you only read the last line. Seven edits
silently did not land, including the one that pointed the model tree at the
materialized shards, and the result was the same failure twice with the same
message. Every edit pass now prints an applied/failed ledger AND is grepped for
in the file afterwards.

LESSON 41 (do not launch a long-lived controller from a blocking tool call).
`nohup controller & ; wait-loop` in one shell means the harness's timeout kills
the whole process group -- controller included. The controller took SIGTERM
mid-measure and entered its guaranteed teardown. It held (--hold-on-failure
covers `not completed`, not just failure), so nothing was lost, but that was
luck rather than design: run the controller as a managed background task, and
let the waiting happen in a different one.

LESSON 42 (identical runs need identical code). Two edits to the surface module
landed while cold run 1 was in flight. The measured NUMBER was unaffected, but
`scope_census_sha256` -- which feeds `checkpoint_identity_sha256` -- is computed
from the surface's own scope, so run 1 would have identified the artifact
differently from run 2. Both restarts cost about five minutes each. A pair of
cold runs is determinism evidence only if the two ran the same program.

LESSON 43 (the controller supervises, it must not own). A two-hour capture died
at window 22 of 25 because the LOCAL process watching it was killed. The
instance log ends mid-run with no error and no traceback: the remote process
group simply went with the session that started it. Sixty-five minutes of a
rented H200 had to be bought again for a reason with nothing to do with the
measurement. Stages now launch through `nohup setsid` -- own session, orphaned
to init -- and the controller re-attaches by done-marker. That also changes what
the managed run MEANS: it is the LAUNCHER, so "succeeded" means "launched", and
the verdict falls back to the marker plus a liveness probe whose two answers
cannot be confused with its own command text.

LESSON 44 (lesson 36 wears more than one hat). The liveness probe added for
lesson 43 was `pgrep -f 'stage_measure.sh <stage>'`. Exercised against the live
box it reported a stage that has never existed as running, because `pgrep -f`
matches full command lines and the probe's own shell carries the pattern in its
own. Every stage would have read "alive" forever and a dead one would never have
been detected. It is M1's lesson 36 -- a probe whose output can be produced by
its own command text -- in different clothes, found the same way: by running the
new path before trusting it. `[s]tage_measure.sh` fixes it; the three cases
(nonexistent, real, naive) were checked on the instance before it was believed.

**Cost, four ways.** 1x H200 spot at $1.99/h, IN2, one box (486969) adopted
across five controller lifetimes. (1) The planner's point estimate was $14.38,
band $14.38–$20.13, ceiling $20.58 at --max-runtime 10h. (2) Measured
wall-clock: the box lived 21:51→03:29 UTC = 5.63 h = $11.20, plus a 400 GB
filesystem for the same span at an inferred rate, ~$0.4. (3) Account balance is
not usable as ground truth this session: it moved $122.12 → $164.52 across a
top-up and three other sessions' boxes on the shared account. (4) Attributable
to the measurement itself: about $11.6.

That is MORE than M1's ~$9.2, and the mission asked for less. The overrun is
entirely restarts, and it is worth being exact about who paid for what: two
restarts were mine, freezing the code so both cold runs would be produced by the
same program (~10 min), one was the external kill at window 22 of 25 (~65 min),
and one was the SHA256SUMS refusal after a full 175 GB fetch and checksum pass
(~25 min). The measurement work itself — bootstrap 2m07s, fetch 5m42s at ~510
MB/s, materialize 2m06s, two cold runs at 4221.7 s and 4219.5 s, score, seal —
is 2.6 h, and the planner now prices it at $6.61 rather than $14.38 because the
7.35 min/window constant finally had its second data point and could be retired.
M3 should cost about half of M1.

## 2026-08-30 — measurement 4: vcruz305's K2, and the first gate this panel has failed

**THE NUMBER.** 0.15520955491423008 nats, mean of two cold runs whose run means
are identical to the last bit and whose tokenwise KL arrays share one sha256
(a75500a9). Top-1 agreement 0.8726526624328286, also identical across runs.
Against this lane's own BF16 floor (0.011505922619330299) the
quantization-attributable error is 0.143703632294899769 nats. The panel's
quality gate is mean tokenwise KLD < 0.06, and this is the first row in this
registry to FAIL it, at 2.6x the threshold.

**Say it plainly: a 2-bit routed-expert quantization of GLM-5.3-Flash diverges
by 0.155 nats from its own BF16 source, and that is a lot.** It is 3.07x the
0.050501241465423556 of 0xSero's 3.0bpw and 6.09x the 0.025503427634363770 of
the 4-bpw rung, for 35 % and 44 % fewer bytes respectively. Top-1 falls from
95.31 % at 4 bpw to 93.00 % at 3 bpw to 87.27 % here: one token in eight now
disagrees with the BF16 model's argmax. The curve between 4 and 3 bpw was
already steep at 2x; between 3 and 2 it steepens again to 3x. Two rungs of
measured evidence say the same thing, which is more than either says alone.

None of that is a verdict on the run, and the run is not in doubt: two cold
processes produced one tokenwise KL digest, `run_mean_spread` is exactly 0.0,
and all 907,200 decoded expert matrices were K2. A failing gate is a fact about
the artifact. It is published with the gate status on the row, a
`quality_gate_failed` disclosure beside it, and no softening.

**The single mean hides a 3.7x spread.** Per domain: axis2_legal 0.2509,
axis1_general 0.1727, axis3_code_agentic 0.1272, axis4_reasoning_termination
0.0671. The panel mean is the number that ranks, and the per-domain block is
the number that tells you where the damage went. Both are in the receipt.

**What the ladder now says.** On this panel, this lane, this teacher, one
comparability key (`cmp--202b717f3219c414`):

| artifact | bpw | size | mean KLD | attributable | top-1 | gate |
|---|---:|---:|---|---|---|---|
| BF16 floor | 16 | -- | 0.011505922619330299 | -- | -- | -- |
| malaiwah TR3 K8 | 8 | 331.4 GB | 0.012384191023436866 | 0.000878 | -- | PASS |
| malaiwah TR3 K6 | 6 | 253.5 GB | 0.013714888822596553 | 0.002209 | 0.9656 | PASS |
| Mia-AiLab TR3 4bpw | 4 | 175.7 GB | 0.025503427634363770 | 0.013998 | 0.9531 | PASS |
| turboderp exl3 4.05bpw | 4.05 | 165.2 GB | 0.025526426915472484 | 0.014021 | 0.9510 | PASS |
| 0xSero Dione Q4 (sealed lane) | 4.0 | 187.6 GB | 0.027262784814670614 | 0.015757 | -- | PASS |
| 0xSero Dione 3.0bpw | 3.0 | 149.6 GB | 0.050501241465423556 | 0.038995 | 0.9300 | PASS |
| **vcruz305 EXL3 K2** | **2.0** | **97.8 GB** | **0.15520955491423008** | **0.143704** | **0.8727** | **FAIL** |

**A storage layout is not a producer, and a scope is not a storage layout.** This
artifact is read by the `exl3hf` surface, which existed because turboderp
publishes stock-exllamav3 HF shards -- canonical index, per-module
`{trellis,suh,svh,<codebook>}`, official tensor names. vcruz305's pack has
exactly that storage and almost nothing else in common with turboderp's: the
codebook is MCG rather than mul1, the scope is routed-experts-ONLY rather than
full-scope, the head is native BF16 rather than K6, the KDA attention is stored
UNFUSED under the official q/k/v names rather than as a fused `qkv_proj`, and
the MTP layer's 864 routed experts are quantized INTO THE MAIN INDEX rather than
into a side `mtp.safetensors`. It is, in scope, a 0xSero-shaped release wearing
turboderp's storage. Three places in this tree had quietly assumed the two
questions were one question, and all three would have answered after the money.

**The name census is the scope.** The release's own `config.json` states
`bits 2, codebook mcg, head_bits 16, quant_method exl3, scope
glm53_routed_experts_only, non_routed_dtype_policy official_source_native`, and
its 150,226-entry index closes on exactly that: 148,608 routed payload tensors
(43 layers x 288 experts x 3 projections x 4 objects) and precisely the official
BF16 release's 1,618 non-routed names, unfused, no strays either way. Not one
field on the registry row says `unknown`. That is M1's lesson applied before the
fact rather than retro-fixed after it.

It also has a defect worth naming, because it is the kind a reader would
otherwise trust: the standalone `quantization_config.json` sidecar carries a
`tensor_storage` map that covers 4,180 of the 37,152 quantized modules -- layers
10 through 13 in full and layer 14 in part, and then it stops. Its
`serving_reader_qualified` also reads `true` where the inline block in
`config.json` reads `false`. The header fields agree and are what this
measurement used; nothing on the record is read from the sidecar. A partial map
that looks complete is worse than no map, so it is disclosed on the artifact.

**No producer digest list at all, so we brought our own.** Every other
third-party artifact measured on this panel publishes something to check the
bytes against: brandonmusic and his mirrors ship `SHA256SUMS`, 0xSero ships an
`EXL3_MANIFEST.json`. This release ships neither, and `fetch_target` correctly
recorded "no SHA256SUMS in release" and verified nothing. The only digest list
that exists for these bytes is the Hub's own per-file LFS content digests, so
they were captured from the models API BEFORE the rental -- 122 entries, a
manifest digest of `43a16228...` -- uploaded to the instance, and recomputed
there against the downloaded tree: 122/122 verified, 97,764,515,699 bytes in
88 s, zero absent, zero `.safetensors` on disk uncovered by the list. It ran
concurrently with a GPU-bound stage and cost nothing. It is a weaker anchor than
a producer seal and a stronger one than nothing, and the disclosure says exactly
which it is.

**The two teardown layers had opposite theories of the same event, and it took
an hour of a live capture to notice.** At 00:59 the harness reaped the local
controller -- lesson 41's exact shape, on the very run that cites it. Lesson 43's
fix worked perfectly: the measure stage had been launched `nohup setsid`, was
orphaned to init, and kept capturing through the controller's death and through
its interrupted teardown. Nineteen windows in, the capture was healthy, the GPU
was at 60 %, and the box was fine.

LESSON 51 (a liveness signal designed around one failure becomes a hazard when
another one is fixed). The on-instance watchdog has two triggers: the absolute
deadline, and a heartbeat the controller touches every 60 s. The heartbeat
trigger encodes a theory -- "if the controller is dead, nobody is watching, so
stop the work" -- that was TRUE when it was written and stopped being true the
day lesson 43 made the work survive the controller by design. So the watchdog
was seven minutes from `pkill -f 'stage_measure.sh'` on a capture that was
running exactly as intended, and it would have destroyed 19 of 25 windows to
protect against a condition the other fix had already removed. Two safety
mechanisms, each correct alone, disagreeing about what a dead controller means.
The heartbeat was kept fresh by hand -- bounded to 160 iterations so it can
expire, and deliberately NOT touching the deadline, which is the trigger that
still means what it says.

LESSON 52 (a resume that launches is not a resume that attaches). `run_stage`
calls `jl.run_job("nohup setsid bash stage_measure.sh <stage> ...")` and THEN
polls; the stage's own guard is its done-marker, which by definition does not
exist while the stage is running. So re-running the controller during a live
stage starts a SECOND copy of that stage: two capture processes writing
`receipts/run-1/logits/` at once, which is not a crash, it is a corrupted
measurement that looks finished. The liveness probe that would answer this
already exists -- lesson 44 built it, carefully, with the `[s]tage_measure.sh`
bracket -- but it is consulted only after the launch, as a way to interpret a
launcher that already returned. The resume here therefore waited for the marker
rather than trusting the controller to attach, and the probe now runs BEFORE the
launch.

LESSON 53 (an interrupted teardown leaves the secret behind). The killed
controller reached `pulling receipts` and stopped. `--hold-on-failure` did the
right thing and kept the box, but `_shred_secrets` never ran, so the 0600 HF
token file was still on a rented instance an hour later -- and because `done`
was set at the TOP of `Teardown.run`, nothing would have retried it. That is
CLI-02(b), which had been sitting in `docs/REVIEW-DEFERRED.md` as
defence-in-depth, arriving as a live consequence: the fix that makes a second
`run()` retry instead of no-op is the same fix that gets the token shredded.

**Three defects between "the recon says exl3hf" and a number, all of them in
stages that run after the money.**

The lane had no profile for a 2.0-bpw exl3hf artifact and REFUSED at plan time
for $0.00, which is the behaviour lesson 47 bought. Adding `vcruz-k2-2bpw` was
the easy half. The hard half was that `k6_kld_report` republished the capture's
provenance pins under `profile.startswith(("dione", "turbo"))` — correct for the
three profiles that existed when it was written, and wrong for the fourth, which
is captured by the SAME `--source exl3hf` front end and seals the same fields
and starts with neither prefix. The headline summary would have carried no
`artifact_repo`, no `artifact_revision`, no codebook and no seal disclosure: a
registry row citing an artifact it cannot name. That is LESSON 48 recurring one
profile later, and the answer is the same one M3 reached for the profile map —
make the mapping DATA (`PROFILE_SURFACE_FAMILY`), not a string test.

`registry_add` had the same shape a layer down: its `TURBO_SUMMARIES` family was
named after a producer when what it actually keys on is a STORAGE layout. It is
`EXL3HF_SUMMARIES` now. Its adapter already read `declared_head_bits` off the
receipt instead of asserting a head policy, so a native-BF16-head release was
described correctly by the same code — the docstring was the only thing that was
wrong, and a docstring that lies about which fact is load-bearing is how the next
person gets it wrong.

And a third, found by the seal rehearsal rather than by reading: an ingested row
whose quality gate FAILED recorded that in `/quality_gate/passed` and nowhere
else, so every rendered disclosure list showed it as clean. On the one
measurement in this registry whose gate actually fails. `registry_add` emits
`quality_gate_failed` now, and REG-24 probes both outcomes.

**K2 had never been exercised, and a rate nobody has exercised is a rate nobody
may publish.** Every codec rung in this tree pinned K3/K4/K6/K8. Before any
money: the dione selftest's numpy transliteration of exllamav3's `quant/pack.cu`
now packs K2 as well, and the exl3hf selftest asserts that our `anybits` unpack
inverts that packer BITWISE at K2, agrees with the dione copy, and produces a
pinned golden digest. A K2-only defect — one lag dropped from the unpack loop
when `bits == 2` — fails the new rung and passes every pre-existing one. Both
ran green on the instance during setup, where the campaign reader is importable,
before the fetch. A real K2 payload was also decoded locally over HTTP range
requests from the release's own shard 57 before renting anything: marker
-877912083 as declared, orientation (2048, 4096), std 0.0193, finite.

**Cost, four ways.** 1x H200 spot at $1.99/h, IN2, one box (487502) across two
controller lifetimes. (1) The planner's point estimate was $6.84, band
$6.84–$9.57, ceiling $12.25 at `--max-runtime 6h`. (2) Measured wall clock: the
box lived 08:11 → 11:00 UTC = 2.82 h = $5.61, plus a 300 GB filesystem for the
same span at the inferred rate, ~$0.19. (3) The runner's own reconciliation:
`billed_usd` 5.411 from `jl get .cost`, which spans the whole instance lifetime
and is the honest figure; its `computed_usd` of $0.58 is the SECOND lifetime's
17 minutes and must not be quoted as the cost of the measurement. The balance
delta is again unusable — another session's 4x RTX-PRO6000 billed on the same
account throughout. (4) Attributable to this measurement: about $5.6.

That is 81 % of M3's $6.9, 48 % of M2's ~$11.6 and 61 % of M1's ~$9.2, and the
second consecutive measurement to land inside its own point estimate. The
capture was also the fastest per window this lane has seen on any surface:
3552.81 s and 3645.22 s for 25 windows, 2.368 and 2.430 min/window, against
3.12 for the same exl3hf surface at K4. Same storage layout, same fill loop,
half the trellis bytes per matrix. `minutes_per_window_by_surface` keeps 3.12 —
the conservative direction — because one artifact is not a rate.

Stage timings, for the next planner: setup 6m21s (cold apt/pip cache; M3 saw
2m07s), fetch_target 4m14s (97.8 GB), materialize 2m07s, fetch_panel 4m14s,
measure 2h00m for two cold runs, score 4m14s, seal 2m07s. Note that every
already-done stage still costs 2m07s on a resume, because `_await_stage` sleeps
its full 120 s poll before consulting the marker for the first time; five
skipped stages cost about ten minutes of rental to skip.

**For M5.** The five oldest open criticals in `docs/REVIEW-DEFERRED.md` are
closed — SEC-01, CLI-01, CLI-02(b), CLI-11/SEC-08 and CC-07's predicate — each
with a regression test that fails against the unpatched tree, and
`bin/selftest_teardown.py` exists because the class that guarantees a rented GPU
is destroyed had no test at all. Two things to watch. The Mia-AiLab row was
re-based to `repo_all_files` so the four measured rungs share one size axis;
brandonmusic's own row keeps `repo_weight_files` deliberately, because his repo
is unpinned and an all-files sum over a moving tree is not a fact. And
`--source nvfp4` still builds its non-routed view through `prepare_nonrouted_view`
rather than a materialize stage — M2 flagged it, M3 flagged it, and three
surfaces have now needed materialize instead.

## 2026-08-30 — measurement 3: 0xSero's 3.0bpw, and a scope that was published all along

**THE NUMBER.** 0.050501241465423556 nats, mean of two cold runs whose run means are
identical to the last bit and whose tokenwise KL arrays share one sha256
(845617b3). Top-1 agreement 0.9299658036150464, also identical across runs. Against
this lane's own BF16 floor (0.011505922619330299) the quantization-attributable
error is 0.038995318846093259 nats.

**The first point below 4 bpw on this panel, and it costs more than the bytes
it saves.** Every other GLM-5.3-Flash quant measured here sits at 4 bpw or
above. This one is 149.6 GB against 165.2 GB for turboderp's 4.05bpw and 175.6
GB for the TR3 4bpw — 9.4 % and 14.8 % fewer bytes — and it pays 1.98x the
divergence for them (0.050501 against 0.025526 and 0.025503), 2.79x the
attributable error (0.038995 against 0.014021 and 0.013998), and 2.3 points of
top-1 agreement (93.00 % against 95.10 % and 95.31 %). It still clears the
panel's 0.06 gate, and it is still bitwise deterministic through this path,
which is a real property and not every artifact has it. But the shape of the
curve between 4 and 3 bpw is now measured rather than assumed, and it is steep:
about a tenth of the bytes for about twice the divergence.

**Their number and ours disagree by 3x, and that is the point of a panel.**
0xSero publishes their own held-out figure in `RELEASE_STATUS.json` — forward KL
0.15251, top-1 0.87285 over 65,504 positions — and marks the release quality:
FAIL against their own threshold. Ours is 0.050501 at 93.00 %. Neither is wrong.
Different corpus, different position count, different estimator, different
reference. That is exactly why the registry records their verdict on the
ARTIFACT record, where it describes the artifact, and keeps it out of the
measurement row, where it would be mistaken for a second opinion on the same
question. Publishing a failing self-assessment at all is more than most
producers do, and it should be said plainly: this release is honest about
itself.

**What the ladder now says.** On this panel, this lane, this teacher:

| artifact | bpw | mean KLD | attributable |
|---|---|---|---|
| BF16 floor | 16 | 0.011505922619330299 | -- |
| malaiwah TR3 K8 | 8 | 0.012384191023436866 | 0.000878 |
| malaiwah TR3 K6 | 6 | 0.013714888822596553 | 0.002209 |
| Mia-AiLab TR3 4bpw (brandonmusic's bytes) | 4 | 0.025503427634363770 | 0.013998 |
| turboderp exl3 4.05bpw | 4.05 | 0.025526426915472484 | 0.014021 |
| **0xSero Dione 3.0bpw** | **3.0** | **0.050501241465423556** | **0.038995318846093259** |

**The scope was published all along; we had not read it.** 0xSero's Q4 row has
carried `unknown` for embed_tokens, attn.qkv, attn.o and lm_head since it was
added, under the note that the release "declares a scope policy that was not
parsed into this registry". It did not need a new source to fix. The release's
own `config.json` states
`quantized_scope = model.language_model.layers.3..44.mlp.experts.0..287.{gate_proj,up_proj,down_proj}.weight`
and `retained_dtype = source_precision`, and a name census of the published
583,090-entry index closes on exactly that: 580,608 routed payload tensors and a
non-routed set that bijects the official BF16 release's 1,618 names, with no
strays either way. So the head is native BF16, not unknown, and the artifact is
routed-experts-only. Both rungs of the ladder now share one table, the Q4 record
carries a `scope_record_corrected` disclosure, and the worked example in
`docs/examples/` was re-sealed over the corrected scope. This is M1's lesson
turned on a record we wrote ourselves: recording `unknown` when the producer
published the answer is the same failure as guessing.

The same reading also corrected a claim about WHERE the TP4 slicing lives. The
old disclosure said "per-layer part-0..part-3 side files". It is not in the
files: the slices are TENSOR names, `<module>.rank0..rank3.{trellis,suh,svh,mcg}`,
and the full HF matrix is the rank-ordered concatenation. The
`layers/layer-NN-part-K` files are a parallel-encoding artifact — in the 3.0bpw
release layers 3-5 have ONE part holding all 288 experts while layers 6-44 split
even and odd experts across two, and every part carries all four ranks.

**Eight defects between "the recon says compatible" and a running measurement.**

LESSON 45 (one producer, two releases, two manifest schemas). The recon listed
"sniff EXL3_MANIFEST.json name" as a small gap. The name was the smaller half.
0xSero's Q4 ships `exl3-manifest.json` with `quantized_shards`/`retained_shards`
arrays keyed on `name`; the 3.0bpw ships `EXL3_MANIFEST.json` with
`schema_version: 1`, one flat `files` array keyed on `path`, and its source
repo, revision, tp_size and bit rate all moved into nested objects. They share
no key that matters. Sniffing one name and one shape would have refused the
3.0bpw release as "not a Dione tree" — after downloading 149 GB of it.
`find_manifest` folds case and underscores; `parse_manifest` normalizes both and
REFUSES a third rather than guessing at its shape.

LESSON 46 (a fill loop that hashes what nobody reads). `load_decoded_module`
defaults to `hash_payloads=True`, and the streaming fill called it that way —
sha256 over every trellis block, 3.2 MB per matrix, 907,200 matrices per cold
run. The census it feeds is recorded ONCE PER LAYER: 42 rows. So the run was
about to hash ~2.9 TB to publish 42 rows' worth of it. The exl3hf loop had
already got this right (`hash_payload=record_census`); the dione loop, written
earlier, had not. Splitting `load_decoded_module` into a read half and a decode
half fixed it and made the loop threadable at the same time — `DioneShardReader`
kept ONE dict of safetensors handles, which is why that fill was "serial by
design" while every other surface reads through a pool.

LESSON 47 (a bits-only profile key stops working the moment two surfaces publish
at the same rate). The runner resolved the engine profile from the sniffed BIT
RATE alone, and fell back to `"k6"` when the rate was unmapped. A 4.0-bpw TR3
release and a 4.0-bpw Dione release are the same number and a different codec, a
different scope and a different receipt family; and `"k6"` is not a safe default
— it is a real profile that names a real receipt family, so the fallback does not
fail loudly, it publishes a wrong label. `profile_map_by_surface` resolves
(surface, bits) at PLAN time and REFUSES for $0.00 when it cannot.

LESSON 48 (the summary republishes what the capture sealed, so the capture has
to seal it). `k6_kld_report`'s dione branch reads `dione_repo`,
`dione_revision`, `dione_shard_hash_verification` and the source pins off the
CAPTURE receipt — fields the sealed-lane capture writes and the streaming one did
not. The headline summary would have been full of nulls, and the registry row
would have cited an artifact it could not name, four stages after the money. The
capture seals them now, plus the config/index digests, the manifest name and
schema, the materialization receipt and the scope digests.

**Two rehearsals that fired, and one gate that failed on the box for four
minutes.** The seal and registry stages were rehearsed offline against the real
artifact metadata and a synthetic number (M2's lesson 39), and both found
something: `published_scope` asserted `policy: "mixed"` beside a table that says
otherwise — SCOPE-003 reads the word as "do the QUANTIZED classes share one
(format, bits)", and exactly one class is quantized here, so the answer is
`uniform` and the ingest would have been an ERROR; and four disclosure codes the
enriched adapter emits were not in the known-code list (DISC-004). Both would
have fired after both cold runs were paid for.

What the rehearsal could NOT catch was a gate that only exists on the instance.
`selftest_dione_offline.py` — which predates this measurement and had never been
wired into the bootstrap — created its scratch directory inside
`k6/tools/dione-evidence/`, a local evidence directory that does not travel in
the measurement bundle. So it could not run anywhere the source tree is a subset,
which is every measurement instance. And the two manifest-identity fields added
to `DioneSurface` were required and sat mid-record, breaking the same selftest's
hand-built surface. Cost: four minutes and a held box, fixed in two ~30-second
cycles by running the selftest ON the held instance rather than guessing. Both
now pass there, which is the only place the rungs that matter can run:
pack-layout equivalence against the exllamav3 `pack.cu` transliteration at
K3/K4/K6, decode identity against the campaign reader, and — in the new
47-rung stream selftest — `decode_module_payload` bitwise equal to
`load_decoded_module` with `hash_payloads` changing the census and never the
tensor. That is the K3 codec evidence this measurement needed before it was
allowed to cost anything.

LESSON 49 (a launcher that swallows its child's output makes a two-hour stage
unobservable). `invoke_engine.py` runs the engine with captured stdout/stderr and
writes them only after the process exits, so during a 77-minute capture the log
holds exactly one line: the argv. There is no way to tell a healthy run from a
hung one from that file — the progress had to be read out of band, by counting
`receipts/run-N/logits/window-*.safetensors` and timestamping them. That is how
3.08 min/window was measured here at all. A stage you cannot watch is a stage
whose liveness probe is the only thing standing between a stall and the deadline.

LESSON 50 (lesson 36 is a family, and two more members turned up in my own
monitors). The first watcher's exit condition grepped the controller log for
`TEARDOWN` — and matched the plan's own `TEARDOWN PLAN` heading, declaring the
run finished before the box was created. The second grepped for `HELD` over the
WHOLE log, which is appended to across controller lifetimes, so it matched the
PREVIOUS lifetime's held banner and declared the resumed run finished about
fifteen seconds after it started. Both are the M1 shape exactly: a probe whose
output can be produced by text that is not the answer. The third watcher reads
only the lines after this lifetime's `adopting existing instance` marker, and its
done-marker probe was exercised against a file that exists and a file that does
not before it was believed.

**Cost, four ways.** 1x H200 spot at $1.99/h, IN2, one box (487243)
across two controller lifetimes; no filesystem survived it. (1) The planner's
point estimate was $6.53, band $6.53–$9.15, ceiling $18.52 at --max-runtime 9h.
(2) Measured wall clock: the box lived 04:07 → 07:26 UTC = 3.32 h = $6.60, plus
a 400 GB filesystem for the same span at an inferred rate, ~$0.22. (3) The
runner's own reconciliation: computed $6.17, billed $6.65; account balance is
again not usable as ground truth — it moved $164.46 → $127.89 while another
session's 4x RTX-PRO6000 on-demand box billed continuously on the same account.
(4) Attributable to this measurement: about $6.9.

That is 40 % of M2's ~$11.6 and 75 % of M1's ~$9.2, and it is the first
measurement in this campaign to land inside its own point estimate. The reason
is that nothing went wrong after the money started: the single failure cost four
minutes on a HELD box and was fixed by running the failing gate on that box
rather than guessing at it. Stage timings, for the next planner: setup 2m07s,
fetch_target 8m26s (149.6 GB plus 135 s hashing all 130 shards), materialize
2m07s (of which 32 s is the materializer itself; the rest is stage overhead),
fetch_panel 4m13s, measure 2h42m, score 4m14s, seal 2m07s.

**For M4.** Three fixes shipped with this entry, and one thing to watch.

`invoke_engine.py` streams its child now (lesson 49), so the next 79-minute
stage is readable while it runs instead of after it ends.

`minutes_per_window` moved from the lane to the SURFACE. The dione figure is
3.19, tr3's is 2.82 and exl3hf's is 3.12, all on the same lane, the same panel
and the same teacher — the spread is the storage layout, not the bit rate, and
one constant cannot say that. The planner prints which basis it used.

The dione scope table is shared by both rungs of 0xSero's ladder, so a fourth
Dione release needs only its bit rate. The manifest normalizer handles both
published schemas and REFUSES a third rather than guessing — if 0xSero ships a
third shape, that refusal is the thing that will fire, for $0.00, at plan time.

To watch: `--source nvfp4` still builds its non-routed view through
`prepare_nonrouted_view` over the artifact's own snapshot, the path M2 flagged
as never having been run end to end against a real snapshot. Both surfaces that
have since been exercised (tr3, dione) needed a materialize stage instead. The
next NVFP4 measurement should assume it needs one too, and find out at plan time
rather than at load time.

---

## 2026-08-30 — the layer-outer engine: the last blocker on GLM-5.3, and the $35–95 it is worth

Stage A ended on one sentence (`docs/GLM53-ROOT-FEASIBILITY.md` §9.7): *"the
layer-outer, window-inner schedule that reads the tree once per run instead of
once per window is still the difference between a $3 stage C and a $38–96 one,
and it is still unwritten."* It is now written, and proven.

`k6/tools/layer_outer.py` + `hf_capture.py --schedule layer-outer`. The default
schedule is unchanged and the old path is untouched.

### The design decision that made it safe

The naive layer-outer loop re-implements the model's forward — embeddings,
position ids, mask mapping, rotary tables, per-layer kwargs, carried state,
final norm — and every one of those is a chance to differ from `transformers`
by a detail. Two of them bite exactly here: `GlmMoeDsaModel.forward` threads a
**second** value between layers (`hidden_states, topk_indices = layer(...,
prev_topk_indices=topk_indices)` — the DSA indexer's shared top-k, recomputed
only by `full` layers), and `Glm5NextTextModel.forward` carries a hyper-channel
dimension. A re-implementation that knows about hidden states and not about
those is silently wrong, and silently wrong is the worst outcome available.

So the engine re-implements **nothing**. It runs the model's own `forward` once
per (layer, window) and proxies only the decoder layers: layers below the one
being computed return the previous outer iteration's return value *verbatim*
(whatever its shape — the carried state rides along untouched), the layer being
computed calls the real layer and memoises, layers above raise a suspend
exception that unwinds the pass. On the last layer there is no proxy left above,
so the model runs straight on into its own final norm and head — that is the
epilogue, executed by the model's own code, with the existing head pre-hook
firing exactly as before. The only thing the file decides is *when* each layer
runs. The prologue is recomputed once per (layer, window); that is an embedding
gather and a mask build against a layer of a 753B MoE, and it is paid on purpose
to buy an implementation that cannot drift.

Same principle on the loading side: the streamer builds on meta through
`cls.get_init_context(...)` and materialises one layer at a time through
`transformers`' own `convert_and_load_state_dict_in_model`. Re-deriving the
256→1 expert fusion by hand would have been the same category of mistake.
Byte-equality against `from_pretrained`, parameter by parameter, is asserted
(L4/L6), not assumed.

### What it bought

| | before | after |
|---|---:|---:|
| Fruit peak VRAM (L4, measured) | 10.409 GB | **2.167 GB** |
| Fruit peak resident weights | 9.144 GB | **1.471 GB** |
| GLM-5.3 projected peak VRAM | 81.7 GB | **~47–51 GB** |
| GLM-5.3 min/window | 13–26 | **0.4–1.6** |
| Stage C | $38–96 | **$1.08–3.52** |

The GLM-5.3 peak drops below §2's projection because the engine's residency
split is finer than that projection assumed: §2 kept the whole 37.78 GB
non-routed set resident, whereas the engine streams *whole layers* and leaves
only `embed + lm_head + norm` (3.81 GB) permanently resident.

### Four things this cost me, worth remembering

**Lesson 50 — RSS is the wrong instrument for an mmap loader, and it is wrong
in the flattering direction on the old path and the damning direction on the
new one.** The first Fruit run reported 19.98 GB window-outer vs 11.60 GB
layer-outer and I nearly wrote that down as a 1.7× win. It is not: `safetensors`
mmaps the shards, so both runs carry ~10 GB of evictable page cache in
`ru_maxrss`. The real figures are 9.14 vs 1.47 GB of materialised weights — a
6.2× — and I only got them by adding an explicit `resident_parameter_bytes`
high-water mark. Stage A had already flagged this ("an unknown part of that
22.3 GB is evictable file-backed page cache") and I still walked into it.
`ru_maxrss` is also **bytes on Darwin and kilobytes on Linux**; the report names
its units for that reason.

**Lesson 51 — a CPU proof of a GPU engine is half a proof.** Everything passed
on the Mac. The streamed loader's `device_map={"": "cuda:0"}` path had never
run. One L4 spot instance, created and destroyed inside the hour, closed that —
and the CUDA digests matched on both schedules. Shipping "proven" without it
would have left the hole in exactly the place the engine exists for.

**Lesson 52 — `bin/selftest_hf_capture.py` was never wired into
`bin/selftest_all.sh`.** 25 cases, green, and nothing ran them. Stage A's own
regression tests (A17–A22) were in that file. Both it and the new
`selftest_layer_outer.py` are wired in now. A test that no runner invokes is a
document, not a test.

**Lesson 53 — record the schedule where it does not cost comparability.** The
schedule is in `runtime.capture_tool`, deliberately *not* in
`stack_fingerprint`: `dscompare` reads the fingerprint to decide
`stack_relation`, and a `cross_stack` verdict stamps `usable_as_floor: false`
and attaches a 1e-2-class bias block. Charging a capture that penalty for a loop
order the digests prove is bit-identical would be asserting a difference that is
not there. Disclosure and penalty are different things and they go in different
fields.

### Fail-without-fix

`bin/selftest_layer_outer.py` against `git archive HEAD` of the parent commit
(new test file copied in, everything else the pre-change tree): **1 passed, 19
failed**. After: **20 passed, 0 failed**. The one that passes before passes
vacuously — it asserts a refusal, and the old tree refuses the whole invocation
on the unrecognised flag. Named in `docs/GLM53-LAYER-OUTER.md` §9 so the 1 is
not read as coverage.

### To watch

The engine's least-measured term is the **expert-fusion transient** during a
layer load. Stage A upper-bounded it at 22.3 GB (1.15× one sparse layer's routed
experts) from a CPU RSS reading; on Fruit's CUDA run it is small enough to hide
under the epilogue's logits buffer, but Fruit's routed set per layer is 24×
smaller than GLM-5.3's at the same vocabulary, so the happy reading must not be
extrapolated. Second: Fruit's per-layer load shows ~0.13 ms per checkpoint
tensor of size-independent overhead, and a GLM-5.3 sparse layer has 76,800
source expert tensors — extrapolating to ~10 s/layer, ~12.5 min/run, the same
order as the IO. **Both are answered by the first sparse layer's `layer_load`
log line in a Stage B run.** Measure them; do not trust this paragraph.

**Stage B was not taken.** No GLM-5.3 capture was run. That remains a separate,
budgeted decision.

## 2026-09-02 — the safety drill cleaned up correctly, but its proof did not qualify

The first GLM-5.3 root-capture prerequisite drill used checkout
`e3d28f28bc7e9911896fe61ca393b88ad2fc0e6f` and created exactly one secure,
on-demand RunPod L4 (`2vokciv9zqczb2`). The autonomous user-systemd reaper
requested exact-id destruction at `2026-09-02T01:19:57Z`; complete provider
inventory proved exact absence one second later. Billing became available at
`2026-09-02T02:15:42Z` and reconciled to exactly
`$0.06642962130717933` (`$0.0659666582942009` GPU plus
`$0.00046296301297843456` disk). No provider resource remained.

The drill still produced no accepted `proof.json`. Its validator reported only
that lifecycle/billing times were misordered or outside the authored bound, then
deleted its staging evidence. The destroy and absence observations were within
the 15-minute provider-lag bound; billing's 55-minute publication delay was
allowed by the controller. The exact failed conjunct is therefore not
recoverable. The operator fingerprint prompt consumed material time before
controller loss and is the leading explanation, not a proved postmortem.

The prompt is gone. Both the drill and measurement controller now read one
full-line ED25519 fingerprint from the fresh pod's authenticated RunPod v2
container-log stream before uploading code, with a 5,000-line request tail, a
2 MiB response ceiling and a 64 KiB event-line ceiling. The API key remains in
the request header. The controller compares that independently retrieved value
to the untrusted network keyscan and seals both values in
`runpod-ssh-host-key-proof.v2`; any mismatch refuses. Host authentication that
uses the drill workload deadline now requests cleanup instead of producing a
late controller-loss claim. The proof validator reports each timestamp
invariant separately, and an offline regression proves that billing may
legitimately stabilize after the 15-minute lifecycle bound.

This was one paid create and one failed qualification attempt. No GLM-5.3 model
capture ran, and nothing was published.

## 2026-09-02 — the replacement drill found a live-stream deadline bug before measurement

The authorized replacement proof used checkout
`f8535b56b1d4ff4afffb560959dc08fb8e4cd828` and created exactly one secure,
on-demand RunPod L4 (`o8lprocjxzct9c`) at `2026-09-02T11:09:00Z`. The
controller had not published ready evidence when its 300-second workload
deadline arrived. Cleanup requested exact-id destruction at
`2026-09-02T11:13:55Z`; complete provider inventory proved exact absence ten
seconds later. Billing reconciled at `2026-09-02T12:20:57Z` to exactly
`$0.04223442170768976` (`$0.04130849568173289` GPU plus
`$0.0009259260259568691` disk). The two failed prerequisite attempts therefore
cost exactly `$0.10866404301486909` in total. Every lease is terminal and the
reaper is healthy.

The child left no stage-specific error before the supervisor's deliberate
`SIGKILL`, so the durable record does not prove which call occupied the final
instant. Source inspection and a fail-without-fix regression did expose a
matching defect: `urllib`'s timeout bounded each socket operation, not the whole
authenticated event stream. A live stream of valid non-key events could keep
resetting that per-read bound until the controller's outer deadline killed it.
Against the old source the regression consumed the 2 MiB response allowance
instead of obeying its requested global deadline.

Commit `2402f5e` adds an independent monotonic 15-minute whole-stream deadline,
retains the 60-second per-I/O bound and 2 MiB cumulative response ceiling, and
retries bounded finite streams without accepting any weaker source or key
syntax. The default proof envelope is now 20 minutes to ready evidence and 22
minutes to provider/reaper destruction, leaving the same 120-second terminal
reserve. The regression passes with the fix; the full local battery reports
84 passed, 0 failed, 0 skipped, and `registry make check` reports 0 errors with
the existing 41 warnings.

This replacement still produced no accepted `proof.json`. No H200 was created,
no GLM-5.3 capture ran, and nothing was published. The remaining H200
authorization was explicitly conditional on an accepted proof, so another
proof create requires a new operator decision.

## 2026-09-02 — batch every spend-free failure before the final paid proof

The third authorized prerequisite attempt used checkout
`18d44987ce0fdbc93019435b663f941a2e64ccee` and created exactly one secure,
on-demand RunPod L4 (`pd1mvzqts58ihj`) at `2026-09-02T13:54:52Z`. The
supervisor intentionally removed its controller. The autonomous installed
reaper requested exact-id destruction at `2026-09-02T14:16:49Z`; complete
GraphQL and REST inventory proved exact absence one second later. The final
provider bill reconciled at `2026-09-02T15:21:10Z` to exactly
`$0.18164147599600255` (`$0.17932666093111038` GPU plus
`$0.002314815064892173` disk as provider aggregates). All leases are terminal,
the campaign reservation is released with zero remaining liability, and the
installed reaper is healthy. The three prerequisite attempts cost exactly
`$0.29030551901087164` in total.

This attempt exposed one more control-plane defect instead of producing an
accepted `proof.json`. RunPod's GPU aggregate differed from the exact two-bucket
sum by `$0.000000000000000006`; the parser allowed only
`$0.000000000000000001`. The raw bucket and aggregate decimals already remained
unchanged, but that harmless representational difference kept the lease at
`ABSENCE_CONFIRMED`. The bound is now one femtodollar
(`$0.000000000000001`): enough for the observed provider aggregation, still
fourteen decimal places below one cent. The new regression fails against the
old parser and passes with the fix. Reinstalling the final control snapshot then
reconciled the same durable lease without weakening identity, absence, or
billing evidence. The stopped old controller could not seal a proof under its
bound parser, and its checkout is now stale in any case.

Running the exact H200 dry-run early also caught two independent preflight
errors before a large rental. The authored model size was the safetensors
index's tensor-payload total (`1506659919872` bytes), but the storage and receipt
contract requires the 282 shard **file** bytes (`1506667387408`), a
`7467536`-byte difference. The corrected evidence binds shard-manifest SHA-256
`4500ebd01844457a106ed6031a67ff581d77406e8d2872ce43f2abd51a65ba2b`.
Second, Hugging Face returns authenticated `404` but anonymous `401` for this
absent dataset/model/space id. Publication preflight now accepts anonymous
`401` or `404` only after the owner-authenticated exact-id query proves `404`;
it still refuses every ambiguous authenticated result.

The final spend-free command now passes target identity, the 25-window panel,
engine probes, source checkout, publication ownership and non-overwrite checks,
safe H200 capacity, fit, storage, timing, cost, and account/reaper gates. At the
observed secure on-demand H200 rate of `$4.59/h`, the 3 h 30 min workload plus
the 14,400-second retrieval/delete reserve yields a maximum liability of
`$39.41123015873015873015873017`, below the explicit `$40` attempt cap. It
refuses only because the checkout-bound safety-proof file does not exist.

Both available local Hugging Face credentials carry write capability. The
measurement still requires a separate owner-readable `0600` read-scope token
for the remote target fetch; the publication token stays on the controller.

No fourth L4 proof will run in this batch. The next paid sequence is fixed:
after all spend-free changes are committed, run exactly one final L4
controller-loss proof, and only if it qualifies proceed immediately to the
H200 capture. No H200 was created, no GLM-5.3 capture ran, and nothing was
published.

### Additive arithmetic correction

The immediately preceding entry calls one femtodollar "fourteen decimal places
below one cent." The correct relation is `$10^{-15}` USD = `$10^{-13}` cents:
**thirteen** orders of magnitude below one cent. The bound itself is unchanged.
The H200 liability sentence also abbreviates the quote: its
`$39.41123015873015873015873017` includes 12,600 workload seconds, 14,400
retrieval/delete seconds, 600 timer/API-lag seconds, and the authored storage
tariffs. It is not compute for only the first two durations.

## 2026-09-02 — the fourth proof tore down perfectly and lost to a floored second

The authorized fourth prerequisite attempt used clean checkout
`37764a571ebe6ff60b34e04b3297c5ce04c47204` and created exactly one secure,
on-demand RunPod L4 (`hxcekqeaai3dch`) at `2026-09-02T19:16:31Z`. The
supervisor removed its controller. The autonomous installed reaper requested
exact-id destruction at `19:39:13Z`; complete GraphQL and REST inventory proved
exact absence one second later. Billing was retrieved at `20:24:53Z` and the
lease reconciled terminal at `20:24:55Z` to exactly
`$0.058409954188391566` (`$0.05794699117541313` GPU plus
`$0.00046296301297843456` disk). Four prerequisite attempts have now cost
exactly `$0.348715473199263206`. Every lease is terminal, the campaign
reservation is released with zero remaining liability, and the reaper is
healthy.

The femtodollar billing bound from earlier today did its job: this provider
aggregate matched its exact bucket sum with no rounding difference at all, and
the reconciliation that the third attempt could not complete completed here
without touching a single provider decimal.

The proof was still refused, by one conjunct: `provider observations do not
remain ordered through proof issue`. Nothing was out of order. `issued_at` is
written by `time.strftime`, so it is **floored to a whole second**, while the
deadline observations and the reaper health stamps are raw `time.time()`
floats. Seal the proof in the same wall second as the final poll — which is
what happens, because the loop breaks on that poll and the seal follows
immediately — and `last_observation <= issued` compares `T.877` against
`T.000` and refuses a lifecycle that was exactly right. The same
mixed-resolution hazard sat in three sibling chains: `destroy_started <=
destroy_at`, `invocation_started <= billed_at`, and `reaped_at <= issued`.

`_whole_second` now truncates the fractional side wherever it is ordered
against an authored whole-second timestamp. The guarantee these chains state
only ever existed at one-second resolution, so it is checked there; an
observation or reaper invocation a full second late still refuses.

The reason 84 green rungs never caught this is worth naming. The drill
fixture's clock advanced by *exactly* the requested sleep, so every poll
snapped onto the authored deadline second and no timestamp in the battery ever
carried a fraction. The new case models a real sleep, which returns no earlier
than asked and leaves the wall clock slightly past it. Against the pre-fix
validator that case does not merely fail its assertion — `execute_drill` cannot
finish at all, refusing on the sibling billing/health chain, which is the
production failure reproduced offline.

This attempt produced no accepted `proof.json`. No H200 was created, no
GLM-5.3 capture ran, and nothing was published. The remaining H200
authorization was explicitly conditional on an accepted proof, so the next
paid create requires a fresh operator decision.

## 2026-09-03 — the fifth proof lost to one 503 on a read it had already earned

The authorized retry ran on checkout
`0e93522` and created exactly one secure, on-demand RunPod L4
(`y2wmt4dnr41lss`) at `2026-09-02T22:39:56Z`. The one-second-resolution fix
held: nothing in the ordering chains refused. The autonomous installed reaper
requested exact-id destruction at `23:05:04Z` and complete GraphQL and REST
inventory proved exact absence one second later. An independent authenticated
`myself { pods }` read afterwards returned zero pods.

Then, while polling for billing to publish, RunPod answered a **read** with
`HTTP 503: Service Unavailable` and the drill aborted. The pod was already
destroyed and its absence already proven; the run threw away a correct,
paid lifecycle over one transient status on an idempotent query. There was no
retry anywhere in the RunPod transport: every `HTTPError` became a refusal on
the first response.

Idempotent reads now ride out a bounded outage — statuses `{429, 500, 502,
503, 504}`, at most three attempts, 2s then 4s, inside a 30-second budget so a
retried poll stays well within the 120-second `DEADLINE_POLL_DURATION_MAX`
the proof validator enforces. Only timeouts are excluded, because a timeout
consumes the budget it would need. **Mutations are never repeated**: the
GraphQL path retries only a document beginning `query`, the create path is
untouched, and an ambiguous create is still never retried. The regression
drives a 503-then-recover opener and asserts three attempts on a read, exactly
one attempt on a `mutation`, and no key material in either message. Against
the pre-fix transport the same read aborts after one attempt with exactly the
production string.

Billing then refused settlement four more times with `RunPod billing records
omit the end of the resolved window`. That was the validator being right:
RunPod snaps the query to hour boundaries, so a pod that died at `23:05`
resolves a window ending at `00:00`, and the `23:00–00:00` bucket cannot be
published while that hour is still open. It settled at `00:04:42Z`, four
minutes after the hour closed, to exactly `$0.19510014518164098`
(`$0.1927853301167488` GPU plus `$0.002314815064892173` disk) across the two
expected buckets. Five prerequisite attempts have now cost exactly
`$0.543815618380904186`. All ten leases are terminal.

Changing `bin/fidelity/runpodapi.py` also made the installed reaper report
`not healthy`, because that file is inside the reaper's control closure and
the health check compares the installed snapshot against the current
checkout. That is the drift mechanism working, not a fault; reinstalling
restored `ok`. Worth remembering: **any** edit to the control closure requires
a reinstall before the next paid create.

This attempt produced no accepted `proof.json`. No H200 was created, no
GLM-5.3 capture ran, and nothing was published.

## 2026-09-03 — the first accepted proof, and the capacity refusal that closed the gate

The sixth prerequisite attempt, on checkout `166215c` with
`--runpod-drill-billing-wait-seconds 7200`, **produced the campaign's first
accepted `proof.json`**. One secure on-demand L4 (`k8zqoisv9aezz6`) was created
at `01:12:40Z`; the autonomous installed reaper requested exact-id destruction
at `01:37:53Z` and complete GraphQL and REST inventory proved exact absence one
second later. The resolved billing window `01:00–02:00Z` published after the
hour closed and the lease reconciled terminal at `02:18:39Z` to exactly
`$0.048252253560349345`. The proof is
`fidelity-suite/runpod-safety-proof.v2`, sealed
`cfe861c64d7592416306d018538744b58afb93685f3489b535eccb9310a2edc7`, binding 12
artifacts. Six prerequisite attempts cost exactly
`$0.592067871941253531` in total.

The extended billing wait was the difference. It is not a weakened gate: the
bound is the authored `300..86400` range, and the settlement it waited for is
the same complete two-bucket window the validator has always demanded.

With the proof accepted, the exact H200 plan passed all eighteen gates —
`current-paid-fault-drill` among them — at a calculated maximum of
`$39.41123015873015873015873017` against the `$40` cap. The create POST then
came back `SUPPLY_CONSTRAINT`: **no H200 capacity at that instant**. Nothing
was created; an independent authenticated `myself { pods }` read returned zero
pods.

And that refusal closed the campaign's paid admission gate permanently. The
lease protocol had exactly one classification for a create that raised:
"response lost". With zero matches it parks in `CREATING` forever by design
(`LOST_CREATE_RESPONSE_RECONCILED_ZERO_WINDOW_CLOSED_UNRESOLVED`, and the
state graph offered `CREATING` no terminal edge at all), and the ledger's
`cancel_before_create` refuses any release once POST intent is fsynced. So one
capacity refusal left an unresolved lease holding a `$40` reservation, and
`validate_unresolved_lease_scope(require_empty=True)` then refused every
further paid create. That is a fail-closed design meeting a case it had never
been shown.

A create the provider **refused by name** is not a create whose response was
lost. `SUPPLY_CONSTRAINT` on a parseable response carrying no id anywhere is
positive evidence that nothing was ever accepted. That distinction is now
explicit end to end: `RunPodCreateRejectedError` is raised only for an
enumerated code on an id-free response, the lease gains
`PROVIDER_REJECTED_CREATE_NO_RESOURCE` (`CREATING -> TERMINAL`) whose evidence
must name its codes and leave nothing attributable, and the ledger accepts
`PROVIDER_REJECTED_CREATE` as durable no-resource proof. Every ambiguous
failure keeps the old path: a timeout, a transport error, an unparseable body,
any response mentioning an id, any incomplete listing. A refusal contradicted
by a real pod still binds that pod for cleanup and never closes.

The stranded lease was then resolved through that new path rather than by
hand: its own history carried the provider's refusal, a fresh complete
inventory showed zero pods and zero volumes, and it closed `TERMINAL` with a
`provider_rejected_create` proof and no provider ids. The reservation released
to `$0` and paid admission reopened.

Two operational lessons, both cheap and both paid for twice today. Editing any
file in the reaper's control closure — which includes `cloudlease.py`,
`campaign.py` and `runpodapi.py` — makes the installed reaper report `not
healthy` until it is reinstalled; and the installer refuses a **group-writable**
source, which a `umask 002` checkout produces by default. Second, and more
expensive: a safety proof is bound to the controller bytes that produced it, so
**every** controller change invalidates it. The accepted proof above no longer
matches checkout `83d3aa4`. The sequence that works is: freeze the controller,
prove, capture — with nothing in between.

No H200 was created, no GLM-5.3 capture ran, and nothing was published.

## 2026-09-03 — the controller was the campaign; making strictness opt-in

The operator's verdict, verbatim in spirit: the project's purpose is
measuring quantization fidelity and publishing root datasets, and the last
several days went into a cloud-safety controller instead. The review that
followed was read-only and ran while the H200 capture loop waited for
capacity. Its findings, each cited in the review artifacts:

- Last scientific output: **2026-08-30**. Since then, seven paid RunPod
  drills (`$0.592067871941253531`) and zero measurements.
- Commit `607d207` (2026-09-01, +43,065 lines, never journaled) turned a
  ten-flag, one-paid-step recipe into **37 flags, 15 environment variables
  and two paid steps**, and refused the race mode the plan of record assumed.
  Twelve of the 37 flags admit exactly one value.
- **63 gates** ran before the single create POST, several of them twice or
  three times; the safety layer was ~10,450 lines plus ~3,750 in the
  controller, with ~1,900 lines of JarvisLabs-era code unreachable.
- The only leaked instance in the project's history was defect 13 on
  JarvisLabs, fixed by the era-2 lease and reaper. No mechanism added on
  2026-09-01 has ever prevented a leak; each has only refused a correct
  lifecycle.
- On RunPod the systemd reaper is the **only** autonomous destroy after a
  controller dies: the on-pod watchdog cannot destroy a pod and
  `terminateAfter` is ignored. The drill and its proof are a paid regression
  test of that reaper, read by nothing in the runtime teardown path.
- Two latent defects on the never-exercised post-create path: one failed
  ssh probe aborted a multi-hour capture, and billing was read once at
  +300 s when RunPod needs up to ~64 minutes, so a successful capture would
  have refused its own publication and exited 90 with the pod already gone.
  The pinned storage tariffs also expired on 2026-09-07 — a four-day time
  bomb that would have refused every paid run.

The principle adopted is **layered strictness**. The default path enforces
exactly the four things that keep money and machines safe — `--max-cost`
as an all-in cap, `--max-runtime` as an absolute deadline in the lease, the
watchdog and the provider, teardown on every controller exit path, and the
installed reaper as backstop — and nothing else. Everything layered on top
is preserved and opt-in.

| mechanism | before | after |
|---|---|---|
| safety proof | required; bound to ~216 files; 7-day expiry | `--runpod-safety-proof`, validated unchanged when given |
| campaign ledger | required, pre-existing, drill-proven | per-run ledger auto-created (ceiling = `--max-cost`, foreign pods tolerated); `--campaign-ledger` keeps the strict posture |
| billing settlement | gated the lease, publication and the next admission | advisory: publish at proven absence; the reaper settles later without failing health |
| reaper health | checkout ≠ snapshot ⇒ unhealthy AND the timer refused to sweep | control seals the snapshot only (v3); drift is a warning naming the reinstall command |
| install source perms | group-writable checkout refused | read once, copied to a 0600 snapshot |
| tariffs | literals inside a seven-day window | flag defaults with an age reminder; GPU rate stays live |
| target allowlist | four hardcoded tuples | gone; identity still proven from bytes |
| GPU / capacity | per-repo literals | timing evidence or `--gpu`; capacity from model bytes with overrides |
| single-value flags | twelve typed by hand | defaulted or derived |
| exit codes | 90 for any exception, even at $0 | 3 nothing created; 1 failed but gone; 90 only when a pod may remain |
| ssh probe | one `unknown` aborted | tolerated for 300 s; `failed` still aborts at once |
| clean checkout | zero untracked files | tracked files clean |
| lease sinks | CREATING-forever and ambiguous-without-candidates failed every sweep | expire TERMINAL 900 s after the window; reported for an operator |
| help | twelve flags with no help; nine legacy flags | grouped, every flag explained, examples in the epilog |

The recipe is now `reaper --install` once, then thirteen flags (fourteen with
`--dry-run` for the rehearsal). Nothing in the
four guarantees changed. Three tests that claimed to refuse writable source
files were found to refuse for the snapshot's world-writable `/tmp` ancestor
instead, and now test what they claim. Race mode is unchanged this pass: the
engine-level race works, and the gap is the stage wiring, which is named in
`docs/RACE-MODE.md`.

The branch merged and pushed (`131fd18..893cce8`) while the H200 loop was
still refused for capacity, and the loop was switched to the new controller:
a live dry-run passed all nineteen gates on thirteen flags, and the first
paid attempt on the new code ran plan, job binding, per-attempt ledger
reservation and a real create POST, then closed the lease `TERMINAL`,
released the reservation and exited 3 at `$0.00` on `SUPPLY_CONSTRAINT`.

Twelve consecutive refusals later, a read-only `lowestPrice` probe found the
cause. H200 secure on-demand stock read **Medium**; H200 with the 1.9 TB disk
read Medium; H200 with **28 vCPU and 300 GB host memory together** returned
nothing at all. Those two numbers had been copied from the K6 quant-lane
profile, which keeps a whole model resident on a JarvisLabs host. The
layer-outer root engine keeps one decoder layer plus the embeddings, head and
norms resident — for GLM-5.3 roughly 19 GB plus 4 GB — so the host contract
now derives from that residency: (16 vCPU, 128 GB) for GLM-5.3, which the
same probe read as Medium stock. The controller had been asking for a
machine that does not exist, for a reason that no longer applied.

## 2026-09-03 — the RunPod SSH transport reproduces the Fruit root bitwise

`bin/measure-cloud --provider runpod --role root` on `malaiwah/GLM-5.2-SIQ-Fruit-bf16`
(L4, `--sanity-expect ''`, `$5` cap) ran every stage — bootstrap, authenticated
fetch, panel, two cold captures, verify, `compare_root`, `qualify_root`, a 770 MB
result archive retrieved, pod proven absent — and both cold runs sealed

```
capture_content_digest b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c
```

the digest of the published root and of the 2026-08-31 container acceptance.
`terminal-receipt.json`: `qualified-unpublished`, zero operational errors.
Eleven paid attempts to get there, each a few minutes of L4 (one 15-minute
pod that never surfaced its host key), roughly `$1.6` by wall time; the
reaper's hour-bucket reconciliation lags and reads `$0.22` at time of writing.

What each attempt paid for, in order (every one is now a regression test that
was proven failing on the pre-fix code, or a gathered piece of evidence):

1. `extract-bundle` on the pod assumed `/workspace/fidelity/<job>/` existed.
2. `jobcontract.validate_execution_job` imports `.campaign` at call time;
   `campaign.py` was not in `BUNDLE.txt`, and `selftest_bundle_complete` only
   ran the setup-time selftests. It now resolves every intra-package import
   of every bundled `fidelity` module, at every scope, in the staged tree.
3. `sshbase.exec` kept the first 400 bytes of remote stderr — the head of a
   traceback, never the exception.
4. "stage setup ended in failed" was the whole diagnosis of a pod destroyed
   four seconds later. The poller now reads the stage log, the launch
   wrapper's output and state, the watchdog log and `ABANDONED.json` first.
5. That evidence showed exit 1 with an empty log: `readlink -f` on an engine
   root whose parent does not exist exits 1 silently under `set -e`.
   Provision creates the engine root; the preamble names the failure.
6. The same listing showed `/workspace` (the pod volume) ignoring modes:
   `heartbeat` 0666 after `chmod 600`, `.secrets` 0777 after `mkdir -m 700`.
   The HF token now lives under `/root/.fidelity-secrets/<attempt-hash>` on
   the container disk; the install reads both modes back and refuses if
   they did not hold; the stage driver honours `FIDELITY_SECRETS_DIR`.
7. The exact unexpected-tensor allowlist was authored from the pre-streaming
   aggregate (the 791-tensor MTP block). Fruit's checkpoint also carries 5
   DSA indexer tensors on each of layers 3..12, whose `indexer_types` entry
   is `shared`; transformers builds no indexer there and the streamed loader
   reports them unused as each layer lands. The exact check ran on the early
   report and the seal read that same copy, so the **published Fruit root
   discloses 791 unused tensors where the loader left 841** — the numbers are
   untouched (unused weights take no part in the forward pass; the digest
   above is the proof), the disclosure is incomplete. Registry-session
   correction item. `hf_capture` now runs the exact check on the streamed
   union after the capture; `engines/tools/derive_unexpected_allowlist.py`
   authors allowlists from the loader itself (Fruit: 841 names, with a
   provenance sidecar). GLM-5.3-BF16 and GLM-5.3-Flash-BF16 carry indexer
   tensors only on `full` layers, so their over-index-only lists stand.
8. The controller pinned `--sanity-expect Paris` for every root; the 5B
   fixture answers " the". `''` (recorded, not enforced, plan warning) is
   admissible now; anything but those two is refused.
9. A refactor had dropped `dataset.repository` from the sealed manifest
   while `qualify_root` binds it to the job. Restored (it is in the schema).
10. `resultsink` binds the attestation's `gpu_model` to `environment.gpu`;
    the RunPod job environment only had `provider_gpu_display`.
11. The fetch ran anonymously with the verified token beside it:
    `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` from the stage env also disables
    `HF_TOKEN_PATH`. `load_token` re-enables it for its own stage.

Also fixed on the way: `Console.warn` arity at three call sites, the Fruit
timing row's `model_bytes` (whole-repository bytes where the controller
derives shard-file bytes), the never-attested teardown path (no `EXITED`
from `CREATING`, no archive fetch without an authenticated host key), and
`--dataset-repository` defaulting to `<measurer>/<dataset-id>` for an
unpublished root.

Measured on the L4 pod: 10.08 GB fetched and censused in 44 s anonymously
(≥230 MB/s), bootstrap 13 min, one Fruit cold capture 86 s, `compare_root`
383 s for 16 windows. The GLM-5.3 H200 loop is running again on this
controller; its dry-run calculates `$39.41` under the `$40` cap.

## 2026-09-04 — GLM-5.3 root on an H200: one sealed cold capture, salvaged under a documented override

First H200 attempt on the proven controller bound on the first try
(`2g4emitafqvka7`, RunPod secure `us-co-1`, 16 vCPU / 128 GB / 1.8 TB pod
volume, created 19:46:34Z, hard cap `$40`, `--max-runtime 3h30m` from the
timing row). Setup 7 min. Authenticated fetch of the 282 shards: 1,506,667,387,408
bytes in 92 min, **274 MB/s** sustained (network-bound; the volume writes at
784 MB/s direct). `/workspace` on this pod type is **MooseFS**
(`mfs#us-co-1.runpod.net:9421`, FUSE) — which is also why it ignored file modes
earlier in the day. The container disk on the same host measured 3.5 GB/s
write / **5.9 GB/s** direct read.

Cold run 1 started 21:23:42Z. Layer loads streamed from MooseFS at a median
**89.7 s per layer** (max 124.9 s; 78 layers, 6,789 s of loads in a 9,720 s
run — 19.3 GB per layer at ~215 MB/s on a host shared with other tenants),
plus ~38 s of forward per layer for 26 windows. The 3.5 h bound — one-hour
fetch, 37-minute runs — was going to expire at 23:16:31Z with run 1 at layer
~30 of 78. Michel chose to salvage one capture over a clean `$18` failure.

The override, in full, because it is the only time this campaign has done it:
the on-pod watchdog (pid 284, deadline epoch 1788477391) was SIGKILLed at
22:34:32Z; the controller (pid 1193434) was SIGSTOPped at 22:34:40Z; the
capture ran on; a read-only rsync over the controller's own authenticated
channel copied `dataset/` the moment `capture.done` appeared; the controller
was SIGCONTed at 00:09:57Z, saw the deadline past, and tore down normally —
destroy 00:09:58Z, absence confirmed 00:10:00Z, billing reconciled `$16.76`
(GPU `$15.83` + disk `$0.92`; ≈4 h 23 min of H200 by wall time). No live
pods remain. The provider-side `terminateAfter` (03:16:31Z) and the reaper's
lease deadline were never touched. The disarm step at teardown refused
("recorded watchdog process is not live") exactly as it should, so the
controller's failed archive was not retrieved; `receipts/`, `logs/` and
`job.json` were copied from the pod beforehand.

What the run produced (`~/fidelity-runs/glm53-root-sidecopy/dataset`,
2,532,593,132 bytes, 84/84 checksums, `fidelity-dataset verify` VERIFIED with
tensors recomputed):

```
capture_content_digest  9eba97dddb4ff2e2a1d1fad8fdac1a57ec22963f1f85345451fcadcfc42682b8
dataset_sha256          01b09bac3f1a5169ff0175a899f9c3846dc511f2ee7e18633b22fe8241fd621c
checkpoint_identity     02963bc5f9012be8a597a29737adcb9204fe5a7513f66e9b1f85075449f4dda1
head tensor content     864f488a0074d236062a4f24800940df7ba611dba76238d4f9f847d9659e11cd
```

`zai-org/GLM-5.3-BF16@304b8051…`, `panel--glm53.malaiwah.corpus5x5-v1`
(25 windows, 51,175 positions), hidden form, layer-outer stream, H200,
generation probe " Paris" p=0.245 enforced and passed, 791 unused MTP-block
tensors exactly matching the pinned allowlist on the post-streaming union.
Disclosures: `reduced_run_count` (one cold run). It is **sealed and
unqualified**: `fidelity-dataset publish` requires the qualification receipt,
both captures and the result archive, and nothing here argues for bypassing
that. Its value is as the reproduction target: the next run's two captures
either match this digest bitwise across pods and days, or they say something
worth knowing.

Lessons, in the order they cost money: the timing bound must come from
measured rates on the storage the pod actually has (row re-authored to 8 h on
MooseFS; `bin/engines.json`); a large time buffer is not optional on a shared
host; the pod volume is the wrong place for weights the capture streams twice —
the container disk is ~10x faster to read, and a 1.9 TB container-disk layout
is the next controller change; and a failed run must be able to hand back a
sealed capture it finished (the failed archive excludes `dataset/`).

## 2026-09-04 — GLM-5.3-BF16 root published; FP8 candidate route built; four paid defects

**Published.** `malaiwah/glm53-fidelity-root-v1` @ `9c4a29ee10f393ed2fdbdb9262c1192ddb1507b4`,
dataset_sha256 `6b8d3a7bdf934f18fc819cc72d85c5385c3351fa50a8c9c2612dd9a29172a4a4`,
capture_content_digest `9eba97dddb4ff2e2a1d1fad8fdac1a57ec22963f1f85345451fcadcfc42682b8`.
Cold run 1 is the 2026-09-03 H200 capture salvaged by hand, **re-sealed** (owner
decision, `fidelity-dataset reseal`: the validator's sealed verdict named the pod's
`/workspace/...` path, which the publisher refuses; one field rewritten, tensors
untouched, origin seal `01b09bac…` named in `dataset.resealed` and
`validation/reseal-receipt.json`). Cold run 2, captured fresh on a second H200
(container disk), reproduced it bitwise: self-compare 0.0, top-1 1.0. Same-lane root:
it does not retroactively upgrade any row measured against another teacher.

**What the container disk changed.** 1.5 TB fetched in 12 min (2.36 GB/s); cold run
~10 min of forward (vs 162 min from the MooseFS pod volume). The checkpoint identity
hash was 22 of the 33-minute cold run; it is now parallel (same value).

**Paid defects, each fixed with a regression that fails on the previous commit:**
1. `validation/structural-validation.json` sealed the output directory as its subject
   (every capture before today) — unpublishable under the private-path rule.
2. `--` is not a legal Hub repo id; a $1 Fruit run reached its final step to learn it.
3. The streaming archive builder dropped `LICENSE` bodies: the first non-MIT root to
   qualify (run 3, ~$9) refused its own archive after the science had passed.
4. The RunPod host-key log reader followed one SSE session for the whole 30-minute
   bound; a fresh request returns the line at once. Two L40S pods and the "never
   surfaced" L4 pods were this.
Also: pre-POST lease cancellation now shares the reaper's protocol; the resumed
dataset is frozen before the admission snapshots; upload bound = archive size.

**Built.** Block-scaled FP8 decode in the layer-outer streamer (bitwise transformers
`Fp8Dequantize` on 367/367 real GLM-5.3 tensors incl. the 576-row partial block),
the candidate route (`--candidate-scope/--reference-dataset`: two fresh captures,
qualification, KLD(root‖candidate) on the pod), scopes and allowlists authored from
bytes, `fidelity-post` for the model-page discussion. GLM-5.3 FP8 candidate launched
at 13:12Z against the published root.

## 2026-09-04 - GLM-5.3 FP8 measured against the published root; the datacenter was the fetch

**Published.** `malaiwah/glm53-fidelity-fp8-v1` @ `44eb57a8852d745e3ac9c026e65fcd214f948de3`,
dataset_sha256 `ce1c873497d8f935d62d317d42c29bdabcaf3a88b816b3ec95ef34cf222d9b43`,
capture_content_digest `e0102a154bebba73de98643941e043d36c948735ae804d65e87786b59e42b379`
(cold run 2 reproduced cold run 1 bitwise; self-compare 0.0 / top-1 1.0).
Candidate `zai-org/GLM-5.3` @ `187fb9fff6319062325ff825627ef6db084d9bc6`, block-scaled
FP8 e4m3 [128,128] decoded to bf16 per tensor by the layer-outer streamer (541 modules
kept native), scope `engines/scopes/scope--glm53-fp8.json`.

**KLD(root ‖ FP8) = 0.02230513871040026 nats, top-1 0.9564435759648265**, panel
`panel--glm53.malaiwah.corpus5x5-v1`, reference `malaiwah/glm53-fidelity-root-v1`
@ `9c4a29ee` (dataset_sha256 `6b8d3a7bdf934f18…`), same lane, floor measured 0.0.
Percentiles: median 0.00145, p95 0.0978, p99 0.322, p99.9 1.052, max 3.829.
Disclosure: `shared_reference_head` (info, HEAD-1a; lm_head content digest identical on
both sides). Receipt `reference-comparison/comparison-receipt.json` in the sink
bundle (`result_archive_sha256 cd827ae4fd48276d18bbb6384754de529d2852b14d6309767a7ef32ac2067d01`).
Pod `7z2q69tfpl9fcu`, H200 SXM, RunPod secure **US-NC-1**, 1h41m, ≈ $6.45 (balance
191.78 → 185.33). Four earlier attempts today ≈ $15, none producing a number.

**The fetch was the datacenter, not the repo.** Every fast pod was `103.196.86.x`
(Raleigh, US-NC-1: 1.3-2.9 s per 5 GB shard); three slow attempts were one Denver host
`152.236.142.242` (15-28 s per shard). Pinned to US-NC-1 the 750 GB FP8 fetch took
~10 min. `--runpod-datacenter` pins the create; the live attestation now records
`provider_record.data_center_id`; stage lines mirror to the RunPod dashboard log.

**The candidate refusal that cost the fifth attempt.** The panel binding lists the
ROOT's `config.json`; a quantized candidate carries its own, so the capture refused
the binding after a full fetch. `fetch_reference` now fetches the reference root's
model-class files anonymously and the candidate capture verifies a composed tokenizer
root. Fruit FP8 rehearsed the whole route first (`malaiwah/fruit-fidelity-fp8-v1` @
`113d9d0d`, 0.012402918051705871 nats, top-1 0.9364).

Not claimed: this row does not upgrade any GLM-5.3 row measured against another
teacher; admissibility is the registry session's call.
