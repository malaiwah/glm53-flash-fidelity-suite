# GLM-5.3-Flash EXL3/TR3-MCG K6 + K8 + K6K8 Program Runbook

Mission: produce the FIRST K6 (uniform routed 6-bit) and K6K8 (routed down_proj 8-bit,
gate/up 6-bit, everything non-routed incl. shared experts native BF16) EXL3/TR3-MCG
quants of GLM-5.3-Flash via brandonmusic's public pipeline, score both on his 25 sealed
final windows against his fp32 teacher logits, publish weights + receipts + tools.
ADDED 2026-08-27 evening (DECISIONS.md 7): a K8-UNIFORM campaign (P1b) on the same
P1 fleet - same calibration/seed/parameters as K6 - as (a) a shippable ~309 GiB
near-BF16 flagship and (b) the complete per-choice parts bin that makes future
multi-precision K6K8 offline assembly instead of re-encode.

All source claims below were re-verified against the mirrored pipeline at
`scratchpad/brandon-pipeline/runtime/src/quant_pipeline` (ground truth), not taken
on faith from the anatomy reports. Where anatomy and source disagreed, source won.

## Operator directives (DECISIONS.md, 2026-08-27) — folded in

1. **K6K8 support is committed engineering, not a maybe**: the K6K8 patch set and
   `malaiwah.*` schema extensions get implemented regardless (free, G0/P0
   timeframe, prior art: operator's Qwen3.8-27B multi-K + GLM-5.2 mixed lineage).
   K6-uniform ships first and **never blocks on the K6K8 patch set**. What stays
   gated is only the PAID K6K8 conversion run (P0 K8 probe green + ≥ $140
   remaining after K6 publishes).
2. **Shared suh/svh per layer — evaluate before committing full conversions.**
   Precise target (verified in source): gate/up `suh` is ALREADY layer-shared and
   down `svh` is ALREADY layer-shared (`vector_topology` in
   `glm53_prepared_backend._projection`); the expert-private vectors are gate/up
   `svh` (output side) and down `suh` (input side). The grouped-gemm win is
   hoisting ONE input-side Hadamard per layer on the down path → the A/B is
   **down_suh shared-per-layer vs expert-private**. Storage feasibility: the
   EXL3 ABI stores suh/svh per module regardless, so a shared choice is just
   identical vectors — checkpoint-compatible; the runtime perf win additionally
   needs a kernel-side hoist (out of scope for this campaign's custom runtime,
   noted for AIBeast serving). Protocol: quantize 2–3 representative layers both
   ways against the captured activations, replay block inputs, compare output
   divergence; adopt shared only if the KLD delta is ~nil. Runs as stage
   `shared_vector_ab` on the P1 instance BEFORE the fleet encode (~1 h, in the
   P1 budget). Operator research link PENDING — fold in when provided; until the
   A/B reads ~nil, the default remains upstream's expert-private topology (a
   shared choice is a disclosed recipe deviation with its own field in the
   launch plan).

## Headline decision (baked in)

**K6-uniform ships first. K6K8 conversion follows conditionally.** Rationale:

* K6 is ~80% pre-plumbed in the pipeline (`SUPPORTED_BITS=(4,6)`, K6 recipe id,
  contract/materialization/runtime/publication schemas all exist) and needs only the
  six small patches in `patches-v2/` (rebased onto the GitHub base `ce1bf970`;
  verified on the L4: clean apply, compile, import, a bits=6 packed-choice
  seal/verify roundtrip, and K4/K6 launch-plan build+verify on a synthetic H200
  preflight all pass). The v1 `patches/` series against the HF mirror is retired.
* K6K8 is a fork, not an edit: the K8 trellis rate is rejected by the pinned codec
  adapter (`codecs/exl3_mcg.py` accepts K3–K6 only), the runtime (trellis words
  64/96 only), the reader, and every receipt schema; the r10 numeric core's 8-bit
  path is unverified. The "native-bf16 down" fallback (357 GiB, 89.2 GiB/rank) does
  NOT fit TP4 x 96 GB, so K6K8 stands or falls with the K8 codec extension.
* Budget forces it too: K6 alone fits $349 with a 30% margin; K6+K6K8 planned
  together does not (see cost table). K6K8 proceeds only if (a) the P0 K8 probe is
  green and (b) ≥ $140 of budget remains after K6 publishes.

## Disclosed deviations from brandonmusic's sealed campaign (put these in every card)

1. **Workers are 4x H200 (SM90), not 4x B200 (SM100).** The sealed schema name
   `...-four-b200-launch-plan.v1` is kept verbatim (his verifier requires the exact
   string); actual devices are recorded truthfully in `scheduler.workers[*]` and a
   `hardware_attestation` block. Patch 0002 widens the worker-slot set to
   `{b200,h200}-{0..3}`.
2. **ExLlamaV3 extension is built with `TORCH_CUDA_ARCH_LIST="9.0;10.0"`** so the
   binary genuinely contains SM100 code objects (his contract requires `"10.0" in
   compute_capabilities`); it executes on SM90.
3. **Student KLD capture runs EP8, not EP4.** His EP4 decoded-BF16 capture peaked
   184.8 GiB/rank on B200-192GB and cannot fit H200-141GB. Patch v2-0006 makes
   `EP_SIZE` env-driven (`QP_GLM53_EP_SIZE=8`); the reconstructed-expert install is
   exact under any divisor of 288, so logits are unchanged.
4. **Hessian artifacts are pruned after each layer receipt seals** (disk: 507 GB if
   kept; ~60 GB peak with pruning). Safe because `seal_layer` is the last consumer
   (`build_materialization_plan` and the materializer verify sealed layer receipts
   only — verified in source), and upstream's own receipts declare the Hessians
   `recomputed_from_sealed_raw_capture_and_sealed_packed_gate_up` (replayable).
   Consequence: never re-enter encode for a sealed layer (the driver skips layers
   whose `layers/layer-NNN.json` exists).
5. **Our transform_seed_sha256 is freshly minted and sealed** (his is not published);
   this is the first-ever K6 so there is no bitwise-reference to match — our own
   five-cold-run receipt establishes determinism of OUR artifact.
6. **K6K8 uses our own schema namespace** (`malaiwah.*`, recipe id
   `malaiwah-shapleymcg-r10-k6k8-...`) — his namespace is never squatted for
   documents his verifiers didn't define.
7. **K6K8 qualification is 3 cold runs, not 5** (budget); K6 gets the full five-run
   receipt matching his protocol.

## What must be authored before any rental (G0 work items)

BASE CHANGE (2026-08-27): the program base is now his GitHub tree
`github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw` @ `ce1bf970` (fetched in
G0), NOT the HF mirror. It is richer than the mirror: `src/quant_pipeline` (no
`runtime/` prefix), `glm53_mcg_preparation` + `glm53_prepared_backend` already
bits-parameterized, `glm53_uniform_k6.py` shipped (K4-KL-gated
`build_launch_plan(inventory, preflight, k4_plan=, k4_authorized_state=)`), and
`scripts/` ships 7 KLD/runtime drivers (student capture EP4, packed-student KLD,
five-run aggregate, TP-runtime qualify/run/receipt) — adapt those for the
qualification tools below instead of writing them fresh. It still ships NO
encode/contract/materialize campaign driver, so write:

| tool | wraps (all published, sealed library calls) |
|---|---|
| `tools/k6_driver.py rehearse` | fixture roundtrip at K6 + K8 codec probe + full-size per-matrix timing bench |
| `tools/k6_driver.py contract` | sealed full-shard inventory builder (`quant-pipeline.glm-release-inventory.v1`), preflight seal, upstream `glm53_uniform_k6.build_launch_plan` (NOTE: K4-KL-gated — needs his sealed K4 launch plan + a `k6_authorized` K4 state receipt as inputs; source them from his published receipts or rebuild via `glm53_uniform_k4` state machine; patches-v2 0003 admits H200 workers, 0004 fixes upstream `state_sha256` KeyError), bits=6 profile-selection receipt, `glm53_direct_k4.build_contract(bits=6)`, work units + hash-chained work state on fs |
| `tools/k6_driver.py encode-worker` | claim → `build_layer_preparation(bits=6)` → `encode_work_unit` → `seal_layer` → prune layer hessians; per-expert receipt resume is native to `encode_work_unit` |
| `tools/k6_driver.py seal-main / release-dead-claims` | main receipt `quant-pipeline.glm53-exl3-mcg-main-k6-receipt.v1` (fields: contract_sha256, complete, matrix_count 36,288 — no upstream builder, driver-authored); requeue dead worker claims |
| `tools/k6_driver.py mtp` | `glm53_mtp_k4.build_contract/build_work_units/claim_next/...`, MTP work-unit **telemetry writer** (`quant-pipeline.glm53-mtp45-exl3-mcg-work-unit-telemetry.v1` — read by `seal_mtp_layer`, written nowhere upstream), `seal_mtp_layer` |
| `tools/k6_driver.py materialize` | reader-ABI receipt (bits 6, `tp_sizes [4]`, exact_reconstruction_checked), `build_materialization_plan`, `materialize_checkpoint` (per-shard resume), `seal_materialization_receipt` |
| `tools/k6_student_capture.py` | EP8 stock-transformers Glm5Next (eager, tf32 off, use_cache off, fp32 logits) + patched offline reader install; capture receipt `quant-pipeline.glm53-logit-capture.v1`; run 1 also dumps the decoded reference parity panel (metadata schema **verbatim** `quant-pipeline.glm53-decoded-k4-tp2-reference-panel.v1` + predeclared tolerances) |
| `tools/k6_kld_report.py` | fp64 `token_kld_chunk` over 25x2047=51,175 positions, report schema `quant-pipeline.glm53-packed-student-kld.v1`, `glm53_k6_postmtp.build_packed_k6_kld_receipt` + `build_five_run_kld_receipt` (patches-v2 0005), comparison table |
| `tools/k6_publish.py` | HF `upload_large_folder`, README cards, `MANIFEST.json` + `SHA256SUMS` closed tree, checkpoint gate receipt, discussion draft |

Estimated authoring effort: the dominant non-GPU work item (~1 focused day) unless
his GitHub repo ships adaptable drivers — **fetch it first**.

## Phase G0 — free preflight (local, $0, 2–4 h; overlaps nothing)

1. DONE (2026-08-27). Fetched and pinned all three source trees (`stage_k6.sh`
   setup clones them on the box):
   `glm-5.3-flash-exl3-4bpw` @ `ce1bf9706b6aa18435e2baccab63bdd72299257c` (new
   base, see BASE CHANGE above); public `shapleymcg` @
   `c9b752b9ea2c1d5bd2b0d63317c4cb8e04a9027c` (HEAD; the `9d83e7d0` closure rev
   is in its history; `scripts/run_qwen_fast_encode.py` sha256
   `ceea8c64…fcbb492` verified); `glm52-sqg-mcg-experiments` @
   `bf37b06691c68525b74bddfa0a1a8216e695c95f` sparse-cloned for
   `bmmlaw_r7_encoder/` (trellis/constants/types/determinism ancestors).
   **STILL OPEN — ABORT-LEVEL**: `r7_encoder/r10_codec.py` (R10TrellisCodec) and
   `encode_tr3_v31.py` are in NO public tree; asked upstream in
   glm-5.3-flash-exl3-4bpw issue #1 (answer pending). **CLOSURE GATE**
   (`stage_k6.sh setup` + every driver invocation): encode proceeds ONLY IF
   (a) his files land upstream (`closure_status.json: closure_source
   upstream`), or (b) the OPERATOR explicitly accepts the disclosed
   reconstruction (`fallback/r10_codec_reconstructed.py`, L4-validated) by
   authoring `$ROOT/RECONSTRUCTION-ACCEPTED.json` — setup then stages the
   fallback package into the shapley tree and `closure_status.json` records
   `closure_source: reconstruction`; every receipt hash-discloses the
   substituted files and bit-identity with his sealed core is explicitly
   NOT claimed (see `fallback/RECONSTRUCTION.md`). Anything else hard-stops
   before any paid encode.
2. Verify on the Hub that his teacher-logits dataset carries `calibration/main-ep4-full/`
   (~464 GB, 42 layers x 1,310,720 rows), `calibration/mtp45-ep4-full/` (~10.8 GB),
   `calibration/panel-v1/`, and `logits/window-0000..0024.safetensors` (31.7 GB).
   **ABORT if the calibration tree is absent** (self-capture needs 4x ~190GB GPUs +
   an unshipped torchrun driver — out of budget).
   **ALSO ABORT-LEVEL (re-review 2026-08-27): his sealed inventory DOCUMENT
   (`quant-pipeline.glm-release-inventory.v1`, sha `f56e9d62…439fea44`) must be
   fetchable.** His capture-manifest binds that sha and `build_contract`
   hard-requires equality; a fresh inventory can never match (it seals our
   local checkpoint path — his declares
   `/workspace/models/zai-org/GLM-5.3-Flash-BF16` per the published capture
   plan). Download it to `calibration/upstream-inventory.json`; `stage_k6.sh
   convert_k6` adopts it verbatim (`--inventory`) and symlinks the declared
   path to the fs BF16 tree. Without the document the captures cannot bind and
   P1 is NO-GO as designed (fallback: self-capture, out of budget).
3. DONE (2026-08-27). Patch series REBASED onto the GitHub base as
   `patches-v2/` (apply order in `patches-v2/SERIES`; `patch -p1` from the repo
   root, verified clean on a pristine clone). What dissolved: v1-0003
   (preparation bits) and v1-0004 (backend bits) are OBSOLETE — upstream
   absorbed them; v1-0005 (our whole `glm53_uniform_k6` module) is SUPERSEDED
   by his shipped module — our disclosed-H200 deviation survives as v2-0003
   (`glm53_uniform_k4._b200_workers` admits a homogeneous H200 fleet, real
   device names recorded in the sealed plan) plus v2-0004 (upstream bug fix:
   `k4_authorized_state["state_sha256"]` KeyError → `state_receipt_sha256`).
   Still needed and carried over: packed-payload bits (v2-0001), direct/MTP
   worker slots + choice bits (v2-0002), K6 post-MTP receipts module
   (v2-0005), reader `QP_GLM53_EP_SIZE` env (v2-0006). Validated on the L4
   (py3.12 venv-b): full-tree py_compile, package imports, bits=6
   packed-choice seal/verify roundtrip (bits=5 rejected), K4+K6
   launch-plan build/verify on a synthetic 4x H200 preflight through a
   synthetic `k6_authorized` K4 gate state (mixed B200/H200 fleet rejected;
   pure-B200 path unchanged), reader EP4/EP8 import + EP3 rejection.
4. Confirm fs 3394 free space (`jl exec ... df -h /home/jl_fs`) and evict FP8
   (328 GB, re-downloadable) → disk ledger below must close.
5. `jl gpus --json`: confirm H200 spot availability in IN2 and that `--num-gpus 4`
   is accepted on the H200 container row (contingency if only 8x granularity: widen
   the worker-slot patches from `range(4)` to `range(8)` and pass 8 workers — the
   scheduler is dynamic so nothing else changes).

Receipts: local pytest-style log + sha256 of fetched closures. Abort criteria above.

## Phase P0 — fixture rehearsal + timing bench (paid, gates everything)

Rental: `jl create --gpu H200 --num-gpus 1 --spot --region IN2 --storage 100 --fs-id 3394 --template pytorch --yes --json`
(spot is fine: the rehearsal is minutes-resumable; container because `--spot`
requires containers).

Commands (via `jl run --on <id> -- bash /home/jl_fs/glm53-k6/stage_k6.sh <stage>`):

1. `stage_k6.sh setup` — venv (torch 2.11.0+cu130, transformers==5.16.1 exact),
   exllamav3 @ c5d9c657 clean-tree in-place build with `TORCH_CUDA_ARCH_LIST="9.0;10.0"`,
   pipeline import smoke, ShapleyMCG pin check.
   In parallel, start the calibration + teacher-panel download to fs (~507 GB; at
   200–400 MB/s ≈ 30–60 min-per-100GB → runs during the rehearsal).
2. `stage_k6.sh fixture_rehearsal` — on `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B`
   (caveats: F32 dtype, no MTP tensors, tiny dims — the driver casts to BF16 and
   synthesizes an MTP-free contract for the roundtrip): contract → preparation →
   encode → seal → materialize → offline-reader decode, bit-exact reconstruction
   check at K6; K8 codec probe (`encode_candidates(bits=8)` through r7_encoder +
   decode vs exllamav3 unpack math); timing bench on 24 full-size (4096x2048)
   synthetic matrices → `seconds_per_full_size_matrix_k6`.

Receipts to verify: `receipts/rehearsal.json` with `k6_roundtrip_exact: true`,
`k8_probe.encode_decode_exact` (bool), `seconds_per_full_size_matrix_k6`.

Abort criteria: roundtrip not exact → stop, debug locally (no more rentals).
Projected main+MTP encode wall `37152 * spm / 4 / 3600` > 24 h → stop, re-plan
(the stage script enforces this gate). K8 probe red → K6K8 is descoped to NO-GO
(K6 continues).

Wall-clock 3–4 h. Cost ≈ **$8**.

## Phase P1 — K6-uniform conversion

Rental: `jl create --gpu H200 --num-gpus 4 --spot --region IN2 --storage 100 --fs-id 3394 --template pytorch --yes --json`

Spot is justified by the pipeline's own resume semantics (all verified in source):
per-expert sealed receipt files (`encode_work_unit` skips existing receipts — at
most one in-flight batch is lost per preemption), idempotent per-layer preparation
(staging dir + `os.replace`), per-shard materialization receipts, hash-chained
work state with dead-claim requeue. Fall back to on-demand ($3.99/GPU-h) only if
preemptions exceed ~2/hour sustained.

Commands: `stage_k6.sh setup` (idempotent re-run on the new instance), then
`stage_k6.sh shared_vector_ab` (operator directive 2: down_suh shared-vs-private
A/B on 2–3 layers against the captured activations, ~1 h on one of the four GPUs;
its receipt fixes the vector topology for the whole campaign BEFORE any fleet
encode), then `stage_k6.sh convert_k6`, which drives:

0. **K4 gate bridge (control session, before the stage runs)**: upstream's
   `glm53_uniform_k6.build_launch_plan` is K4-KL-gated — it demands a sealed
   `glm53_uniform_k4` state receipt in phase `k6_authorized`. The control
   session authors `out-k6/k4-authorized-state.json` as a DISCLOSED bridge:
   directly sealed against the driver-built K4 planning receipt
   (`out-k6/k4-launch-plan.json`, pure planning, `launch_authorized false`),
   with `evidence.k4_packed_kld_receipt_sha256` and friends carrying
   brandonmusic's REAL published K4 receipt hashes plus a
   `malaiwah_bridge_disclosure` block stating that K4 was HIS campaign, not
   re-executed (upstream `verify_state` format-checks evidence hashes; the
   bridge is validated by `tools/selftest_launch_plan.py`). The driver
   NEVER fabricates this document; the stage fail-fasts on its absence.
1. adopt his sealed inventory verbatim (`calibration/upstream-inventory.json`,
   G0 item 2; local shards re-verified against its `shard_sha256` closure;
   cross-check receipt records `binds_to_upstream_receipts` — with the adopted
   document this is true by construction);
2. preflight + K4 planning docs + K4-gated K6 launch plan (upstream module +
   v2-0003 H200 admission + v2-0004 state-field fix) + `build_contract(bits=6)`;
3. 4 workers x `CUDA_VISIBLE_DEVICES={0..3}`, dynamic whole-layer units over
   layers 3–44 (36,288 matrices), candidate-conditioned down Hessians, layer seal,
   hessian prune;
4. MTP45 (864 matrices, 16 units of 18 experts, telemetry receipts, adapter seal) —
   only after the sealed main receipt exists (`main_must_complete_before_mtp`);
5. delete `calibration/main-ep4-full` (disk ledger), then materialize → `ckpt-k6`
   (120 source-aligned shards + config + quantization_config.json +
   exl3-mcg-storage-abi.json + materialization receipt).

Receipts to verify (stage script asserts): `main-receipt.json` complete;
`mtp-adapter-receipt.json` schema `quant-pipeline.glm53-uniform-k6-mtp-adapter-receipt.v1`
qualified; `materialization-receipt.json` schema
`quant-pipeline.glm53-k6-materialization-receipt.v1` with bits 6, `complete`,
`nonrouted_native_exact`, `main_and_mtp_complete` all true, `qualified_tp_sizes []`
and `serving_reader_qualified false` (the verifier hard-requires exactly that);
`output_logical_bytes == 253,536,370,680` (receipt-exact K6 size, 236.12 GiB).

Abort criteria: any contract/seal verification error (fail-closed by design — do
not patch around a red verifier mid-run; stop and diagnose); throughput < 60% of
the P0 projection for > 2 h; disk < 100 GB free on fs.

Wall-clock: **unknown until P0** (no encode timing exists anywhere in the mirror —
verified). Planning window 10–16 h; plan number 14 h. Cost ≈ 14 x $7.96 = **$111**
(range $80–127).

## Phase P1b — K8-uniform conversion (same fleet, right after P1; DECISIONS.md 7)

Runs on the SAME 4x H200 rental as P1, immediately after `convert_k6` seals.
Enablement is `patches-v2/0007-k8-uniform-admission.patch` — admission only
(SUPPORTED_BITS += 8, malaiwah-namespaced K8 recipe id + contract /
materialization / launch-plan / MTP schemas, new `glm53_uniform_k8` plan
module, codec-adapter rate 8, reader 128-word trellis, runtime K8/TP4 serving
admission).  Every single-rate invariant of the sealed uniform contracts is
kept; every K8 identifier that names a sealed artifact is `malaiwah.*`.

**0007 ordering rule (enforced by the stage + setup):** 0007 edits the READER
file whose byte-hash every sealed K6 choice binds (`decoder.reader_abi_sha256`
+ the one-reader-ABI census in `load_complete_surface`).  It lands on the
campaign tree ONLY after `convert_k6.done` (or before any K6 choice seals).
Never mid-K6.

**0008 is REQUIRED for both K6 and K8 (adversarial-review finding):**
`normalization/absolute_v31.py` pins `ALLOWED_BITS = (3,4,5)` at the base
commit and `streaming_v31.FitSampleSpec.from_input` imports it, so
`build_layer_preparation` (the GSS step inside `contract`, and the MTP prep)
refuses bits=6 AND bits=8 — K6's own preparation would crash at layer 3.
`patches-v2/0008-v31-allowed-bits-k6-k8.patch` widens it to (3,4,5,6,8); the
constant is metadata/consistency-only in both v31 modules (audited; no per-K
constants or branches).  Unlike 0007 it touches no reader/closure-hashed file
and may land at ANY time; `stage_k6.sh ensure_0008` applies + receipts it in
setup, convert_k6 and convert_k8.  The P1 fleet fs already carries this
widening as a hot-edit (verified byte-identical to 0008's output on 484853);
its `receipts/patches-v2-applied.txt` line is back-filled by `ensure_0008`.

Commands: `stage_k6.sh convert_k8` then `stage_k6.sh materialize_k8`, driving:

0. patch 0007 top-up + import check; K8 re-probe (`receipts/rehearsal-k8.json`
   — the P0 `rehearsal.json` recorded `k8_probe.admitted=false` pre-0007 BY
   DESIGN) + K8 full-size timing bench (`--bench-bits 8`; K8 trellis edges are
   4x K6's, so seconds/matrix is measured, never assumed) + the
   non-tautological SM90 bit-verify of the reconstructed codec's K8 pack vs
   exllamav3 NATIVE convert (`fallback/probe_native_convert.py`,
   `receipts/k8-native-probe.txt`; VALIDATION.md V9 covered L4/SM89);
1. SAME transform seed: `out-k6/transform-seed.json` is copied to
   `out-k8/` (fail-fast if absent; the driver refuses to mint a fresh seed
   for profile k8 — assembly compatibility is the whole point);
2. SAME K4 gate bridge docs copied from `out-k6/` (the K8 launch plan is
   K4-KL-gated exactly like K6's);
3. K8-specific GSS preparations (bits=8, same calibration captures — re-
   downloaded after `convert_k6` deleted them), contract under
   `malaiwah.glm53-direct-mcg-k8-contract.v1`, 4-worker fleet encode into
   `out-k8/payload-store` (per-expert resume, hessian prune), seal-main, MTP;
4. `materialize_k8` (separate stage, ledger-gated): deletes
   `calibration/main-ep4-full` again, then materializes `ckpt-k8`
   (`malaiwah.glm53-k8-materialization-receipt.v1`, bits 8,
   `output_logical_bytes == 331,449,761,784` = native 19,339,524,984 +
   37,152 x 8,400,900; 308.7 GiB, 77.2 GiB/rank at TP4).

Disk ledger P1b (decimal GB; OBSERVED baseline — the FP8 tree was never
evicted, so P1 started at 513 free, not the ~840 the original ledger assumed):

| moment | delta | free |
|---|---|---|
| convert_k6 start (observed on 484853: df 478 GiB = 513 GB) | — | ~513 |
| after convert_k6 (payload −254, cal main deleted +464, ckpt-k6 −254) | −44 | ~469 |
| **FP8 evicted FIRST** (first `convert_k8` run does this before anything else, receipted in `receipts/fp8-evicted.json`; re-downloadable) | +328 | ~797 |
| cal main re-downloaded for K8 (control session, AFTER the eviction) | −464 | ~333 |
| K8 encode peak (payload −312, hessian transient ≤60 worst-case) | | **−39 worst case** |
| materialize_k8: cal main deleted again | +464 | |
| ckpt-k8 lands | −331 | ~154 |

ORDERING (adversarial-review reorder): the eviction MUST precede the
calibration re-download — the old order re-downloaded 464 GB into ~469 GB
free, a ~5 GB knife-edge.  The stage now runs the receipted FP8 eviction
before its calibration checks (trigger: FP8 present and free < 800, i.e. the
ledger cannot close), then exits asking for the re-download; re-run
`convert_k8` once the captures are back.

The encode peak is the pinch: the worst case goes ~39 GB negative IF the
hessian transient hits its 4-layers-in-flight maximum at the same moment the
payload store completes.  Mitigations, in order: the per-layer hessian prune
keeps the steady-state transient far below 60; the teacher panel (32 GB,
re-downloadable) and any dead scratch can be evicted on warning; encode is
per-expert resumable — if the fs runs dry the workers die NON-destructively,
space is freed, the stage re-runs.  The stage enforces the operator floor
(`require_free_gb 100` before encode) and ntfy-warns when starting under 300
(ledger plans ~333).  `materialize_k8` requires 340 free and tells the
operator to upload+delete `ckpt-k6` (254 GB) first if short.  Keep
`out-k6/payload-store` AND `out-k8/payload-store` — together they ARE the
K6K8 assembly parts bin.

Cost (operator estimate, re-priced by the stage-start bench): **~$26–40**
(3.3–5 h wall on 4x H200 spot at $7.96/h, incl. GSS + MTP; K6's stage actual
was ~2.2 h projected, and K8's Viterbi edge count is 4x — the stage hard-
aborts if the bench projects > 24 h of encode).  Qualification adds
`qualify_k8` (3 cold EP8 runs + TP4 runtime receipt, ~3 h 8x H200 ≈ $48) and
publication ~309 GiB ≈ $7 — both AFTER K6 ships, budget-gated like K6K8.

Abort criteria: any red verifier (fail-closed, never patch around mid-run);
`rehearsal-k8.json` probe not exact; native-probe mismatch on SM90; disk
< 100 GB free; bench projection > 24 h.

**Offline-assembly future note (why P1b exists):** once `out-k6` and
`out-k8` payload stores are both sealed (same seed, same calibration, same
profile policy), a K6K8 "where it counts" checkpoint = pick per-projection
choices from the two stores under a new malaiwah mixed contract + materialize
— zero GPU encode.  DISCLOSED NUANCE: the K8-uniform down_proj payloads were
candidate-conditioned on decoded **K8** gate/up (uniform invariant), while
`recipes/k6k8.json`'s quality-optimal down@8 wants conditioning on decoded
**K6** gate/up.  Assembled-from-parts K6K8 is exactly decodable and valid,
but its down conditioning is K8-based; a fresh conditioned down encode (P2)
remains the quality-optimal path.  Assembly-time work: mixed-rate contract
builder + materializer relaxation of the uniform-rate census (the runtime
already derives per-module K from trellis shape and, post-0007, admits 64/96/
128 words).  Complete list of what the assembly tool consumes and must relax
(verified against the patched tree): (a) inputs — `out-k6/payload-store` +
`out-k8/payload-store` (per-choice payloads are self-contained:
trellis+suh+svh+mcg, choice_id carries the `.K{bits}` suffix), both sealed
contracts, the shared `transform-seed.json`, and either campaign's
`preparation/` permutations (identical by construction: derived from the same
captures + energy_balanced policy, bits-independent); (b) relaxations — a
malaiwah mixed contract schema (uniform contracts pin `allowed_bits == [bits]`
in `verify_contract`), the materializer's per-shard uniform-bits census, and
the runtime's single `qcfg.bits ∈ (4,6,8)` declaration in
`glm53_tp2_exl3.verify_checkpoint` (per-module loading is already
trellis-shape-driven); (c) the packed reader's one-reader-ABI census is
satisfied per-store, but a mixed surface mixes reader_abi values (K6 choices
bind pre-0007 reader bytes, K8 choices post-0007) — the mixed surface loader
must accept exactly that two-value set, receipted.

## Phase P2 — K6K8 conversion (paid run CONDITIONAL — after K6 publishes)

Per operator directive, the K6K8 SUPPORT CODE is implemented unconditionally in
G0/P0 (free) and exercised on the fixture by the P0 K8 probe. Gates for the PAID
conversion: P0 `k8_probe.encode_decode_exact == true` AND remaining budget ≥ $140
AND K6 shipped. If a gate fails the paid run is deferred to the next budget cycle
with the implementation already merged and rehearsed.

Additional code: the K8 codec/reader/runtime enablement LANDED as
`patches-v2/0007-k8-uniform-admission.patch` (P1b) — codec adapter bits tuple,
reader `SUPPORTED_BITS += (8,)`, runtime trellis-words 128.  What P2 still
needs beyond 0007 is only the MIXED-rate relaxation: the `malaiwah.*` K6K8
contract/receipt schemas per `recipes/k6k8.json` and a per-projection-rate
contract/materializer (the uniform modules deliberately refuse profile k6k8).
These are OUR schema extensions — none of his sealed verifiers are weakened;
his verbatim schemas are used only where their checks genuinely pass.  NOTE:
with the P1b parts bin sealed, P2 can also be replaced by offline assembly
(see the P1b note; down conditioning nuance disclosed there).

Rental: same shape as P1 (4x H200 spot, IN2, fs 3394).

Command: `stage_k6.sh convert_k6k8`. Key economy (from the allocation analysis,
justified by upstream's five-run bitwise determinism): gate/up K6 payloads are
**verified-reused** from the sealed K6 payload store (copy + decode + hash
re-verification under the K6K8 contract) — only the 12,384 down_proj matrices are
encoded fresh at K8 with the same candidate-conditioned Hessian flow (evidence
string stays `decoded_k6_candidate_conditioned_...` — gate/up are K6). ~40% of P1
compute. Re-download `calibration/main-ep4-full` first (deleted in P1; ~1 h).

Receipts: same chain under `malaiwah.*` K6K8 schemas; size check
`279,507,501,048 B` (260.31 GiB).

Abort: K8 encode quality anomaly (down-proj proxy error worse than K6 on the same
expert — sanity-checked on layer 3 before the fleet runs); budget gate breach.

Wall-clock 5–8 h (plan 7 h incl. re-download). Cost ≈ 7 x $7.96 ≈ **$56**.

## Phase P3 — qualification on the 25 sealed windows

Rental: `jl create --gpu H200 --num-gpus 8 --spot --region IN2 --storage 100 --fs-id 3394 --template pytorch --yes --json`
(8 GPUs: the EP8 student capture needs all eight ~98 GB/rank; the TP4 runtime
qualify uses 4 of them afterwards. Spot: each cold run is independent and short.)

Commands: `stage_k6.sh qualify_k6` (and later `qualify_k6k8`):

1. 5 cold EP8 student captures (`QP_GLM53_EP_SIZE=8`, eager, tf32 off, fp32
   logits over the 25 windows; run 1 also emits the decoded reference parity
   panel with predeclared tolerances);
2. fp64 tokenwise KLD (exact log-softmax, teacher→student, 51,175 positions,
   mean = his estimator exactly — no weighting, no bf16 anywhere);
3. `glm53_k6_postmtp` packed-KLD receipt (gate: mean < 0.06) + five-cold-run
   receipt (bitwise determinism shows as identical `tokenwise_kld_sha256`);
4. TP4 packed-runtime qualification: `torchrun --nproc-per-node=4
   scripts/qualify_glm53_custom_tp2_runtime.py --model ckpt-k6 --bits 6 ...`
   (script is bits-parameterized: K6→TP4 hardcoded; no SM assumptions in the
   runtime — verified) + generation smoke ≥ 2 tokens.

Receipts: `k6-packed-kld.json` (schema `quant-pipeline.glm53-packed-kld-receipt.v1`,
profile k6-tp4, qualified true), `k6-five-run-kld.json`, `k6-tp4-runtime-receipt.json`
(schema `...-tp4-runtime.v1`, rank logit-sha identity + parity + census
36,288 packed matrices), `comparison-table.md`.

Comparison table skeleton (filled by `k6_kld_report.py`):

| model | routed bpw | size | mean tokenwise KLD vs BF16 teacher (25 sealed windows, 51,175 pos, fp64) | provenance |
|---|---|---|---|---|
| zai-org FP8 (as served) | 8 | 328 GB | 0.020615 | our fidelity-suite baseline |
| brandonmusic K4 (EXL3/TR3-MCG) | 4.01 | 163.6 GiB | 0.024555 (five-run mean, stddev 0) | his sealed receipts |
| **malaiwah K6** | 6.01 | 236.1 GiB | **TBD** (gate < 0.06) | this campaign |
| **malaiwah K6K8** (down@8) | 6.68 | 260.3 GiB | **TBD** (gate < 0.06) | this campaign |

Abort criteria: mean KLD ≥ 0.06 → do NOT publish weights as qualified; publish
receipts + a failure analysis instead (the gate is the release contract). Five-run
means not identical → investigate nondeterminism before publishing (our FP8
campaign's Triton-autotune lesson applies). TP4 parity outside predeclared
tolerances → runtime receipt stays red; weights may still publish with
offline-reader qualification only, deviation disclosed.

Wall-clock: K6 ≈ 5 h (first run pays decode+install; runs 2–5 ride warm page
cache), K6K8 ≈ 3 h. Cost ≈ 5 x $15.92 = **$80** (K6) + 3 x $15.92 = **$48** (K6K8).

## Phase P4 — publication

Rental: `jl create --gpu H200 --num-gpus 1 --spot --region IN2 --storage 100 --fs-id 3394 --template pytorch --yes --json`
(cheapest fs-attached instance; upload is network-bound and resumable —
`upload_large_folder` checkpoints, spot is fine).

Repos (exact names, decided):

* Weights: **`malaiwah/GLM-5.3-Flash-TR3-6bpw`** and **`malaiwah/GLM-5.3-Flash-TR3-6bpwK8-mixed`**
  — each carrying `quantization/recipe.json` (from `recipes/`), `provenance/
  source-model-revision.json` (zai-org/GLM-5.3-Flash-BF16 @ a6c167b6, weight_dtype
  bfloat16, sealed), `receipts/checkpoint.json` publication gate (K6: his schema
  with profile k6; K6K8: `malaiwah.*`), `MANIFEST.json` + `SHA256SUMS` closed tree,
  README card.
* Receipts/metrics: **`malaiwah/GLM-5.3-Flash-fidelity-suite-v1`** under
  `reports/exl3-k6/` (packed-kld, five-run, runtime receipts, tokenwise-KLD
  vectors, comparison table, deviations register, patch series).
* Tools: patch series + driver tools pushed to the existing
  `github.com/malaiwah/quant-fidelity-suite` repo.

README cards MUST carry the upstream model license verbatim: the weight repos
are derivatives of zai-org/GLM-5.3-Flash-BF16 — set the HF `license` metadata
tag and LICENSE file to match the source repo's license exactly (read it off
the pinned revision during G0; do not guess), and state "quantized derivative
of zai-org/GLM-5.3-Flash-BF16 @ a6c167b6" in the card header.

README cards MUST credit brandonmusic prominently: quantization pipeline, recipe,
teacher logits, calibration captures and the sealed-window protocol are his
(links: his HF model `brandonmusic/GLM-5.3-Flash-EXL3-4bpw`-style repo, dataset
`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`, and
`github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw`); our contribution is the
K6/K6K8 rates, the H200 port (deviations register), and the receipts.

Discussion comment draft (posted by the USER, not automation) on his teacher-logits
dataset — `receipts/discussion-comment.md`, roughly: "Using your published pipeline,
calibration captures and sealed final windows, we produced the first K6 (and K6K8)
EXL3/TR3-MCG quants of GLM-5.3-Flash on 4x H200 (deviations disclosed in-repo:
H200 workers, EP8 student capture, fresh transform seed). K6 scored mean tokenwise
KLD <X> vs your fp32 teacher on the same 25 windows (your K4: 0.024555, FP8:
0.020615). Weights + full receipt chain: <links>. Thank you for publishing the
entire evidence chain — it made an independent reproduction at a new rate possible."

Commands: `stage_k6.sh upload_weights`, `stage_k6.sh publish_receipts`
(HF token read from `/home/jl_fs/glm53-k6/.hf_token`, never echoed).

Receipts: HF commit URLs; re-download spot-check of 3 random shards vs SHA256SUMS.

Abort: none destructive; upload retries are idempotent.

Wall-clock 4–6 h (533 GB total for both repos). Cost ≈ 6 x $1.99 = **$12**.

## Disk ledger (fs 3394, 2 TB, IN2 — all phases)

| moment | resident | GB |
|---|---|---|
| start (after evicting FP8 328 GB + old scratch) | BF16 643 | ~650 |
| after G0/P0 downloads | + calibration 475 + teacher panel 32 | ~1,160 |
| P1 encode peak | + payload-store 236 + hessians ≤60 (pruned per layer) + receipts | ~1,460 |
| P1 materialize (calibration deleted first) | − 464 + ckpt-k6 254 | ~1,250 |
| P3 | + logit captures ~13 (5 runs x 2.6 GB fp32) | ~1,265 |
| before P2 | − ckpt-k6 254 (REQUIRED: delete after publication spot-check) | ~1,011 |
| P2 encode peak (cal re-downloaded, coexists with K6K8 payload) | + 464 + payload 280 + hessians ≤60 | ~1,815 peak |
| P2 materialize (cal deleted first) | − 464 + ckpt-k6k8 280 | ~1,630 |

The 2 TB fs closes at every step ONLY IF `ckpt-k6` is deleted after its
publication spot-check and before P2: the K8 down encode needs the calibration
captures resident while the K6K8 payload store grows, so cal (464) + both
payload stores (254 + 280) + hessians (≤60) coexist at the P2 peak. With
ckpt-k6 still resident the peak is ~2,070 GB and busts the fs. `stage_k6.sh
convert_k6k8` enforces this with a 400 GB free-space guard. Delete
`out-k6/payload-store` only AFTER K6K8 finishes (gate/up reuse source).

## Cost table (rates: H200 spot $1.99/GPU-h; on-demand fallback $3.99)

| phase | rental | plan hours | $/h | plan cost | range |
|---|---|---|---|---|---|
| G0 preflight | local | 3 | 0 | $0 | 0 |
| P0 rehearsal + bench | 1x H200 spot cont. | 4 | 1.99 | $8 | 6–10 |
| P1 K6 convert (prep+encode+MTP+materialize) | 4x H200 spot cont. | 14 | 7.96 | $111 | 80–127 |
| P3 K6 qualify (5 cold EP8 + TP4 runtime) | 8x H200 spot cont. | 5 | 15.92 | $80 | 64–96 |
| P4 K6 publish (upload 254 GB + receipts) | 1x H200 spot cont. | 6 | 1.99 | $12 | 8–16 |
| **K6 subtotal** | | | | **$211** | 158–249 |
| **K6 with 30% overrun margin** | | | | **$274** | vs **$349 budget → GO** |
| P1b K8 convert (same fleet, DECISIONS.md 7) | 4x H200 spot cont. | 4 | 7.96 | $32 | 26–40 |
| P3 K8 qualify (3 cold runs + TP4 runtime) | 8x H200 spot cont. | 3 | 15.92 | $48 | 32–64 |
| P4 K8 publish (309 GiB) | 1x H200 spot cont. | 4 | 1.99 | $8 | 5–10 |
| P2 K6K8 convert (down-only + reuse; OR offline assembly from the P1b parts bin ≈ $0 GPU) | 4x H200 spot cont. | 7 | 7.96 | $56 | 40–80 |
| P3 K6K8 qualify (3 cold runs) | 8x H200 spot cont. | 3 | 15.92 | $48 | 32–64 |
| P4 K6K8 publish (280 GB) | 1x H200 spot cont. | 3 | 1.99 | $6 | 4–8 |
| **K6K8 add-on subtotal** | | | | **$110** | 76–152 |
| **K6K8 add-on with 30% margin** | | | | **$143** | |
| **Program total (both), margin-adjusted** | | | | **$417** | **> $349 → K6K8 is gated, not planned** |

Budget rule enacted: run K6 to publication; K6K8 proceeds only if actual remaining
budget ≥ $140 at that point (i.e., K6 actuals ≤ ~$209 — likely if P1 lands ≤ 12 h).

## Wall-clock plan (~24 h target for K6)

T0 G0 (3 h, free, overlaps nothing) → T0+3 P0 (4 h; calibration download runs in
parallel on the same instance) → T0+7 P1 (10–16 h) → T0+17..23 P3 (5 h) → P4 (6 h,
network-bound). **K6 published ≈ T0+28 h (T0+24 h if P1 ≤ 10 h — the P0 bench
decides; the stage script hard-aborts if the projection exceeds 24 h of encode).**
K6K8 adds a second ~12 h window afterwards.

## GO/NO-GO

**GO for K6** — conditional on G0: (1) the public ShapleyMCG closure @ 9d83e7d0
with `r7_encoder` and his GitHub driver repo are fetchable; (2) the calibration
tree exists on the Hub as documented; (3) P0 roundtrip exact and encode projection
< 24 h. Every other blocker found by anatomy is closed by the verified patch series
in `patches-v2/` plus the driver tools enumerated above. Budget fits with the full
30% margin ($274 ≤ $349).

**CONDITIONAL-GO for K6K8** — gated on: P0 K8 probe green (r10 numeric core 8-bit
+ 128-word trellis decode bit-exact on the fixture), K6 shipped, and ≥ $140
remaining. If any gate fails, publish K6 alone and file K6K8 as a designed,
costed follow-up (this document + `recipes/k6k8.json` are the design record).

## Key receipts index (what "done" means)

* `receipts/rehearsal.json` — P0 gate (roundtrip, K8 probe pre-0007, $/h projection)
* `receipts/rehearsal-k8.json` + `receipts/k8-native-probe.txt` — P1b K8 gate
  (post-0007 probe exact + SM90 native bit-verify + K8 $/h projection)
* `receipts/fp8-evicted.json` — P1b ledger eviction note (if taken)
* `ckpt-k8/materialization-receipt.json` — sealed K8 checkpoint (308.7 GiB exact)
* `receipts/k8-packed-kld.json` + `k8-tp4-runtime-receipt.json` — K8 quality + serving
* `ckpt-k6/materialization-receipt.json` — sealed K6 checkpoint (236.12 GiB exact)
* `receipts/k6-packed-kld.json` + `k6-five-run-kld.json` — quality gate < 0.06
* `receipts/k6-tp4-runtime-receipt.json` — TP4 serving qualification
* `receipts/comparison-table.md` — FP8 / K4 / K6 / K6K8 table
* HF: two weight repos + `reports/exl3-k6/` in the fidelity dataset + discussion draft
