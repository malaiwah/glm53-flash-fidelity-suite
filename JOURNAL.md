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
