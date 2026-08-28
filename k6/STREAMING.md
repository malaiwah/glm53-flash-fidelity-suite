# Streaming single-device scorer for the sealed GLM-5.3-Flash panel

`tools/stream_score.py` scores the sealed 25-window / 51,175-position fidelity
panel on **one** GPU (or a 128 GB Mac, or CPU) instead of the 8× H200 box the
sealed protocol used. It is not a re-implementation of the forward: it is the
same `transformers` model, the same module classes, the same op order, the same
dtypes, with one residency change.

---

## 1. Why one GPU is enough

The sealed scorer (`tools/k6_student_capture.py`) calls
`glm53_packed_k4_reader.load_complete_surface()` and then
`install_local_main_experts()`, which decodes **every** routed expert to BF16 up
front and holds it resident for the whole run. Measured from the sealed
receipts (`receipts/k6-student-run1/backend.json`):

| | sealed EP8 |
|---|---|
| resident per rank | 95,112,933,376 B (88.58 GiB) |
| ranks | 8 |
| install time per rank | 782.34 s for 4,536 matrices (**172 ms/matrix**) |
| capture (25 windows) | 139.58 s |

Those 8 GPUs are rented for **residency**, not compute. The panel is
25 × 2048 tokens; the whole forward is ≈1.7 PFLOP, well under a minute of H200
math, and the sealed run reports ~25 % utilisation.

Byte budget, computed from the released BF16 index and confirmed at runtime:

| component | bytes | GB |
|---|---:|---:|
| non-routed (1,618 checkpoint tensors: KDA, DSA/MLA, indexer, mHC, routers, norms, dense 0–2, shared experts, embed, lm_head, vision tower) | 18,976,485,628 | **18.98** |
| routed experts, **one layer** (288 experts × 3 projections, BF16) | 14,495,514,624 | **14.50** |
| routed experts, all 42 executed layers | 608,811,614,208 | 608.8 |
| activations, **measured** (fp32 eager attention scores + fp32 logit staging + decode transients) | 13,600,000,000 | **13.60** |

Streaming one layer at a time therefore measures at **47.08 GB peak** on CUDA
(`torch.cuda.max_memory_allocated`, 2048-token window). The activation figure
above is derived from that measurement, not guessed, so `--vram-budget-gb` does
not lie to a user with a smaller card.

> **Corrections to the brief.** Non-routed is 18.98 GB, not ~34 GB. The panel
> hidden state at a layer boundary is 4× the naive figure because mHC carries
> `hc_mult = 4` residual streams (`[B, S, 4, 4096]`). And the 1× H200 spot
> container is capped at a **300 GiB cgroup limit** even though the host reports
> 3,019 GB — a 42-layer host-RAM decode cache (609 GB) is OOM-killed at layer
> ~22 if you trust `free`. `stream_score.py` reads the cgroup limit and caches
> only what fits.

---

## 2. Parity anatomy — what actually differs

This was verified against the installed `transformers` 5.16.1 source, not
assumed.

**The sealed run applied the EP plan ONLY, not a TP plan.**
`PreTrainedModel.tp_plan` (`transformers/distributed/mixin.py:87-97`) returns
`self._ep_plan` whenever `distributed_config.enable_expert_parallel` is true;
`apply_tensor_parallelism` consumes that property. The `active_tp_plan` field in
the sealed `backend.json` is the *stored attribute* `_tp_plan`, recorded for
provenance — it was never applied. Independent numerical confirmation:
88.58 GiB/rank = 608.8/8 GB routed + ~19 GB non-routed, which is only consistent
with attention / KDA / DSA / mHC / dense / shared-experts / embed / lm_head being
**fully replicated on every rank**.

⇒ Every op outside the routed-MoE block was already a single-device op.

**The sealed run used `grouped_mm`, not the eager expert loop.** The sealed
capture passes no `experts_implementation`, and
`get_correct_experts_implementation(None)` resolves to `"grouped_mm"`
(`modeling_utils.py:1964-1986`). Verified empirically by rebuilding the model
exactly as the sealed capture did and reading the dispatch:

```
top config._experts_implementation : grouped_mm
DISPATCHED EXPERTS FORWARD         : grouped_mm_experts_forward
```

This matters: the fallback `Glm5NextTextExperts.forward` accumulates with
`index_add_` in **bf16** and is CUDA-nondeterministic. `stream_score.py` forces
and records `grouped_mm`.

**So the only mechanical difference is the routed-expert combine, in 42 layers:**

```
EP8   partial_r = bf16( fp32 sum over the top-k slots rank r owns )     # ~5 of 8 ranks are nonzero
      out       = NCCL bf16 all_reduce( partial_0 … partial_7 )
EP1   out       = bf16( fp32 sum over all 8 top-k slots )
```

`grouped_mm_experts_forward` computes `weighted_out.view(T, 8, H).sum(dim=1)` in
fp32 and rounds **once** to bf16. Under EP8 that rounding happens **per rank**,
and the ~5 nonzero bf16 partials are then summed by NCCL in a topology-dependent
order (ring chunking, or NVLS in-switch reduction on an NVSwitch node).

`stream_score.py --ep-emulate 8` reproduces the EP8 partition exactly on one
device — same `EpRouterParallel` masking, same 36-group `torch._grouped_mm`
launch shape, same per-rank bf16 rounding — leaving **only the reduction order**
as a residual. `--reduce-order {fp32,sequential,reverse,pairwise,rotate:N}`
enumerates the candidates so that residual is *measured*, not asserted.

---

## 3. Design

```
                       ┌──────────────────────────────────────────┐
  payload store  ──►   │ load_payload_cpu()   (24-thread pool)     │  IO + 3 sealed SHA gates
  (content-addressed)  │   store.objects.load_tensor               │
                       │   packed_payload_sha256                   │
                       │   checkpoint_payload_sha256               │
                       └────────────────┬─────────────────────────┘
                                        ▼ bounded queue
                       ┌──────────────────────────────────────────┐
                       │ decode_from_payload()  (device)           │  reader's decode_choice_hf
                       │   unpack_trellis_states → mcg_lut         │  VERBATIM, fp32
                       │   2 Hadamard GEMMs, suh/svh scaling       │
                       └────────────────┬─────────────────────────┘
                                        ▼ fuse_gate_up, one fp32→bf16 rounding
                       ┌──────────────────────────────────────────┐
                       │ ONE reusable BF16 slab  [288,4096,4096]   │  14.5 GB, refilled per layer
                       │                       + [288,4096,2048]   │
                       └────────────────┬─────────────────────────┘
                                        ▼ bound as plain module attributes
   model(input_ids, attention_mask, use_cache=False)   ← the sealed call, verbatim
```

**Model construction.** The sealed capture built the student with
`AutoModelForImageTextToText.from_pretrained(bf16, dtype=bfloat16, …)` and then
overwrote only `mlp.experts.{gate_up_proj,down_proj}` for layers 3–44. Reading
all 599 GB to discard 580 GB of it is precisely the residency cost this tool
removes, so the streaming build calls **the same constructor** over a directory
whose `model.safetensors.index.json` lists only the 1,618 non-routed checkpoint
tensors and whose shards are symlinks to the real ones. The
checkpoint→module key conversion (which fuses `q/k/v_conv1d` → `conv1d` and
renames `hc_attn_*` → `attn_hc.*`), buffer construction, dtype handling and
`post_init` are therefore transformers' own, not re-implemented.

The routed expert `nn.Parameter`s are removed at construction (replaced by
0-element placeholders so `_init_weights` stays a no-op) and re-bound to the
slab. That is the entire delta, and it is asserted from the loader's own report:

```
missing_keys 0   mismatched_keys 0   error_msgs 0
unexpected_keys 84  ==  exactly {layers 3..44} × {gate_up_proj, down_proj}
```

Anything else unloaded is a hard failure.

**Independent cross-check of the routed half.** The streaming run's
`verified_packed_payload_bytes` is **228,750,407,424** — exactly
8 × 28,593,800,928, the per-rank figure in the sealed `backend.json`. The single
device verified byte-for-byte the same total packed payload the 8 sealed ranks
did between them, and `census_closes_main_routed_surface` confirms all
42 × 288 × 3 = 36,288 matrices were installed.

**Streaming.** `install_streaming_experts` replaces `experts.forward` with a
wrapper that (a) refills the slab for the layer about to run and (b) applies the
EP emulation. Because the wrapper sits *inside* the model, the per-window call
is byte-for-byte the sealed one:

```python
model(input_ids=ids, attention_mask=attention_mask, use_cache=False, return_dict=True).logits[:, :-1, :]
```

with batch 1, seq 2048, `attn_implementation="eager"`, tf32 off,
`float32_matmul_precision("highest")`, stored fp32 after
`mask[:-1] & mask[1:]` boolean selection. MTP layer 45 is receipt-gated and
never executed.

**Weight sources.** `--source payload-store` reads the content-addressed store
directly (`out-k6/payload-store`), so **no materialized checkpoint and no 254 GB
download is needed**. `--source checkpoint` takes the materialized path.
Byte-equality with the sealed surface is not an assumption: `load_decoded_choice`
re-verifies `packed_payload_sha256` and `checkpoint_payload_sha256` against the
sealed choice descriptor on **every** load, and the run's own
`checkpoint_identity_sha256` (a hash over inventory + contract + all 42 main
layer receipts + MTP receipts + reader ABI) is compared against the sealed one:

```
stream  checkpoint_identity_sha256 = a8668be3592493035e98a52994e0e3c43548a9757eadb79f7ae939f2f32de1c1
sealed  checkpoint_identity_sha256 = a8668be3592493035e98a52994e0e3c43548a9757eadb79f7ae939f2f32de1c1   ✓
```

This also unlocks scoring a parts-bin assembly with **no materialization at all**.

---

## 4. Memory model and schedules

`--vram-budget-gb` picks the largest schedule that fits; `--slab-experts`
overrides it.

| schedule | slab | device peak | notes |
|---|---:|---:|---|
| whole layer (default) | 288 experts, 14.50 GB | **47.08 GB measured** | required for `--sweep`; allows any `--ep-emulate` |
| one EP group | 36 experts, 1.81 GB | **34.40 GB measured** | memory floor; pairs with `--ep-emulate 8` (the group *is* a rank's shard). Bit-identical to slab 288 (§7.3); 44 % slower decode. |

`--decode-cache {none,ram,disk}` trades host memory for repeated decode. With
`none`, each window re-decodes all 42 layers; with `ram`/`disk`, layer L is
decoded once and paged in per window. The RAM cache is **cgroup-aware** and
refuses to exceed 80 % of the container limit rather than being OOM-killed.

| cache | host cost | per-window cost after the first | fits the 1× H200 spot container? |
|---|---:|---|---|
| `none` | 0 | full re-decode (~7 min) | yes |
| `ram` | 609 GB for all 42 layers | 42 × 14.5 GB H2D | **no** (300 GiB cgroup → ~19 layers) |
| `disk` | 609 GB | 42 × 14.5 GB from NVMe | yes on `/` (702 GB free) |

---

## 5. Measured performance and cost

Box: JarvisLabs machine 485591, 1× H200 (143,771 MiB), 28 vCPU, 300 GiB cgroup,
400 GB local NVMe, `$1.99/h`. Payload store staged to local disk first.

| stage | measured |
|---|---|
| stage 219 GB payload store fs → local NVMe | ~4 min, 16 parallel streams |
| sealed-surface verification (37,152 choices) | 21.0 s |
| model build (1,618 non-routed tensors) + device move | ~16 s |
| **decode, 36,288 matrices (slab 288)** | **397.0 s = 10.94 ms/matrix** |
| decode, 36,288 matrices (slab 36, low-memory schedule) | 570.6 s = 15.72 ms/matrix |
| forward, 1 window (2048 tokens), excluding decode | ~2 s |
| peak device memory, slab 288 / slab 36 | **47.08 GB / 34.40 GB** |
| single-window capture, end to end | **~8 min ≈ $0.27** |
| offline L1 ladder (CPU only) | ~40 s |

**The decode is 15.8× faster per matrix than the sealed run** (10.94 ms vs
172 ms). The sealed install was network-fs-read bound at 36.6 MB/s per rank; a
local store plus a 24-thread IO/SHA pool overlapping GPU decode removes that.

Projected full panel (25 windows):

| configuration | wall clock | cost @ $1.99/h |
|---|---|---|
| `--decode-cache none` (re-decode per window) | ~2.8 h | ~$5.6 |
| `--decode-cache ram`, capped at the 300 GiB cgroup (~19 of 42 layers) | ~1.8 h | ~$3.6 |
| sealed lane, 5 cold runs on 8× H200 | 2.37 h × 8 GPUs | ~$36–57 |

A full-panel `--decode-cache disk` (609 GB) fits on `/` but is **not** faster on
this box: the backing device delivers ~900 MB/s, so paging 609 GB per window
costs ~11 min against ~7 min to re-decode. On a host with a fast local NVMe or
≥700 GB of usable RAM it is the right choice; measure before assuming.

Two cold runs (the L4-justified N) therefore cost roughly **$7–11** on one spot
GPU, against ~$36–57 for the sealed 5-run 8-GPU protocol, and the streaming lane
also needs no materialized checkpoint (saving a 254 GB download or a
materialization pass).

---

## 6. Device matrix

| device | fits | schedule | notes |
|---|---|---|---|
| 1× H200 141 GB (validated) | yes, 3× headroom | slab 288 | 47.08 GB peak measured |
| 1× RTX 6000 Pro 96 GB | yes | slab 288 | ~2–2.5× the decode time (bandwidth-bound) |
| 1× 48 GB (L40S/A6000) | yes | slab 36 + `--ep-emulate 8` | **34.40 GB measured**, bit-identical to slab 288 |
| 128 GB Mac (MPS) | yes | slab 288, `--unpack-device cpu` | see §8 |
| CPU only | ≥32 GB | slab 36 | self-test / fixture only |

---

## 7. Validation ladder — measured results

### L1 offline (no GPU, no large weights) — **PASSED, all five rungs**

`tools/stream_score_selftest.py --packed-root … --fixture …`

| rung | result |
|---|---|
| **L1.a decode parity** | `decode_from_payload(load_payload_cpu(…))` is **bitwise** equal to the reader's `load_decoded_choice(…)` on 12 sampled matrices spanning layers 3–44; the CPU-unpack / device-float split (the MPS path) is **also bitwise** equal |
| **L1.b EP emulation** | `ep_router_remap` matches `transformers.distributed.tensor_parallel.EpRouterParallel.transform_output_post_forward` **exactly** on random routing tables; every `--reduce-order` lands within **0.69 bf16 ULP** of the single-device call (budget 4 ULP) |
| **L1.c forward plumbing** | streaming build vs stock `from_pretrained` on the 0.1B architecturally-complete fixture: **`bitwise_equal: true`, `max_abs_logit_delta: 0.0`**, `missing_keys 0` |
| **L1.d receipt schema** | the emitted `capture-receipt.json` shape is accepted by `quant_pipeline…glm53_logits.load_capture_receipt` |
| **L1.e KLD estimator** | `k6_kld_report._token_kld` vs closed-form fp64 KL: max abs **8.5e-16**; KL(p‖p) exactly **0.0**; the sealed `tokenwise-kld.npy` reshapes to (25, 2047) with per-window means matching the sealed report to **exactly 0.0** |

**L1.c is the load-bearing result: the streaming machinery itself contributes
zero error.** The filtered-index build, the slab binding and the plain-attribute
expert weights reproduce stock `from_pretrained` bit for bit.

### L2 single window on 485591 — **MEASURED**

Window `final-0000`, `--decode-cache none`. Two configurations were run; the
second uses the combine order §7.2 shows is the right model of NCCL.

| quantity | `--ep-emulate 8 --reduce-order sequential` | **`--ep-emulate 8 --reduce-order fp32`** |
|---|---:|---:|
| sealed run-1 window mean KLD | `0.016813833091706077` | `0.016813833091706077` |
| streaming window mean KLD | `0.019810763195244545` | **`0.016828908651190112`** |
| **Δ mean KLD** | `+2.9969e-3` | **`+1.5076e-5`** |
| max abs logit delta vs sealed | 8.375 | **2.000** |
| rms logit delta vs sealed | 0.2797 | **0.0381** |
| argmax agreement vs sealed run-1 | 95.26 % | **99.80 %** (4 of 2047 disagree) |
| top-1 agreement vs teacher | 95.51 % | 95.46 % |
| payload sha256 vs sealed | differs | differs |

**The acceptance rule |Δ mean_kld| ≤ 1e-6 is NOT met at L2.** With the correct
combine model the delta is **1.5e-5**, 15× above the rule — down from 3.0e-3
(3,000×) with a naive bf16 chain.

Structure of the delta under the `sequential` (worst) configuration: the
per-position KLD difference is concentrated at the *start* of the window — the
**top-20 positions of 2047 account for 99.2 % of the mean shift**, and 321
positions are within 1e-4, while the logits differ *everywhere* (median
per-position max-abs delta 1.22, 0 positions below 0.01). That is the signature
of routing flips, not of uniform drift.

### L4 determinism — **PASSED (measured)**

Two independent cold processes (`runs/l2-ep8-seq` cold_run 1 and
`runs/sweep-w0` cold_run 1), same configuration, separate `python` invocations,
separate decode passes:

```
payload sha256 equal: True   (4b29ecefbe7980bf… == 4b29ecefbe7980bf…)
```

The streaming scorer is **bitwise deterministic across cold runs**. Note the
definition: the *tensor payload region* of the safetensors file, not the whole
file — safetensors `__metadata__` carries `cold_run`, so whole-file digests
differ by design in the sealed lane too (the five sealed runs have five distinct
file shas and one identical payload sha). This is the property that lets a
2-cold-run measurement replace the sealed 5-run protocol; `measure_stream`
auto-escalates to 5 runs if it ever fails.

### L3 — runnable by one command, not yet run

```
QP_STREAM_LOCAL_STORE=1 QP_STREAM_CACHE=disk \
QP_STREAM_SWEEP=ep8:reverse,ep8:fp32,ep1:none \
bash /home/jl_fs/glm53-k6/stage_k6.sh measure_stream
```

The fail-closed path is **validated**: pointing `QP_STREAM_PACKED_ROOT` at a
missing directory bootstraps the runtime, detects the missing `contract.json`,
writes `receipts/stream-verdict.json` with `verdict: INPUT_MISSING` and the
reason, and exits **6** without touching a GPU.

That stage bootstraps (no encoder toolchain needed — it needs neither exllamav3
nor nvcc, so it runs on a container where `stage_k6.sh setup` cannot complete),
preflights every input and fails closed, stages the payload store locally, runs the L1 ladder, captures
`QP_STREAM_RUNS` (default 2) cold streaming runs, checks cross-run determinism on
the **tensor payload region** (whole-file sha differs by design: `__metadata__`
carries `cold_run`), auto-escalates to 5 cold runs if they differ, produces the
fp64 report through the unmodified `k6_kld_report.py --profile k6-stream`, writes
`receipts/stream-verdict.json`, and sends one ntfy. It publishes nothing.

---

## 7.1 Diagnosis of the L2 delta — mechanically, not by hand-waving

Everything upstream of the combine is proven exact:

1. **Weights.** The BF16 shards at `/home/jl_fs/models/bf16` were re-hashed
   against the sealed inventory (`seal_mode: full-shard-sha256`) — match.
   `config.json` / `index.json` hashes match the sealed inventory. Every routed
   payload passes the reader's three sealed SHA gates on every load, and
   `stored_encoder_closure` re-derives L3/E0/gate_proj and matches the encoder's
   fp16 reconstruction closure.
2. **Surface identity.** `checkpoint_identity_sha256` equals the sealed run's
   `a8668be3…` exactly.
3. **Build.** `missing_keys 0`, and L1.c shows the build is bitwise exact.
4. **Decode.** L1.a shows the decode is bitwise exact.
5. **Routing.** L1.b shows the EP remap matches upstream exactly.
6. **Kernel.** `grouped_mm_experts_forward` confirmed as the sealed dispatch.

What remains is the single op named in §2: the order and precision of the
routed-expert combine. Under EP8 the top-k sum is split across ranks and each
rank's partial is rounded to bf16 *before* being summed; the sealed run then
summed those bf16 partials with NCCL in an order set by the 8-GPU NVSwitch
topology. A single process cannot reproduce that order.

Why an ULP-scale cause produces an O(1) logit delta: **top-8-of-288 routing is a
discontinuous function of the hidden state.** A one-ULP bf16 difference at layer
L flips marginal routing decisions for some tokens at layer L+1, which changes
those tokens' MoE output by an O(1) *relative* amount, which flips more decisions
downstream. Over 42 routed layers this is a chaotic amplification, not a linear
one — which is exactly why the sealed protocol's five cold runs have population
stddev **exactly 0.0** (same node, same NCCL topology, fully deterministic) while
*any* change to the reduction produces a different, equally valid sample.

`--sweep` measures that sensitivity directly on one decode: it re-runs the window
forward under `ep8:sequential`, `ep8:reverse`, `ep8:fp32` and `ep1:none` using
the **same decoded weights**, so the only variable is the combine.

### 7.2 Sweep result — the delta IS the combine order

One decode of all 36,288 matrices, then the same window re-run through the same
weights under different combine orders (`--sweep`). Deltas are against the
`ep8:sequential` primary, on `final-0000`:

One decode of all 36,288 matrices, then the same window re-run through the
**same decoded weights** under four combine variants (`--sweep`). All numbers are
`final-0000` against the sealed EP8 run:

| combine | window mean KLD | **Δ vs sealed** | max abs logit Δ | rms logit Δ | argmax agreement |
|---|---:|---:|---:|---:|---:|
| **sealed EP8 + NCCL** | `0.016813833091706` | — | 0 | 0 | 100 % |
| **`ep8:fp32`** — partials accumulated in **fp32**, rounded once | `0.016828909` | **+1.508e-5** | **2.000** | **0.0381** | **99.80 %** |
| `ep8:reverse` — bf16 chain, reversed | `0.017404776` | +5.909e-4 | 7.391 | 0.2845 | 94.82 % |
| `ep8:sequential` — bf16 chain, in rank order | `0.019810763` | +2.997e-3 | 8.375 | 0.2797 | 95.26 % |
| `ep1:none` — no EP partition at all | `0.016183480` | −6.304e-4 | 7.672 | 0.2739 | 94.97 % |

Two things fall out of this table, and both are measurements, not inferences.

**(a) The combine order is the whole story, and the amplification is enormous.**
`ep8:sequential` and `ep8:reverse` are the *same arithmetic in a different
order*. They cannot differ by more than one bf16 ULP per layer at the combine.
They diverge by **rms 0.257** in the final logits — the same magnitude as the
worst streaming-vs-sealed gap. The mechanism is amplification through
discontinuity: top-8-of-288 routing is a step function of the hidden state, so a
one-ULP difference at layer L flips marginal routing decisions at layer L+1, each
flip changes that token's MoE output by an O(1) *relative* amount, and 42 routed
layers compound it. This is also why the sealed protocol's five cold runs have
population stddev **exactly 0.0** — same node, same NCCL topology, fully
deterministic — while any change to the reduction lands elsewhere.

**(b) NCCL's bf16 `all_reduce` is not a bf16 chain.** `ep8:fp32` — accumulate
the eight per-rank bf16 partials in fp32 and round **once** — is **7× closer in
rms, 4× closer in max-abs, and 40–200× closer in mean KLD** than any bf16-chain
order, and agrees with the sealed argmax on **99.80 %** of positions (4 of 2047
disagree) versus ~95 % for the chains. On an 8× H200 NVSwitch node NCCL
up-converts for the reduction (in-switch NVLS / SHARP reduce in higher
precision). `--reduce-order fp32` is therefore the **default**, chosen by
measurement rather than by assumption.

**The residual.** With the right combine model, window `final-0000` lands
**1.5e-5** from the sealed mean — still 15× above the 1e-6 acceptance, with
rms 0.0381 in the logits. Since `fp32` accumulation is order-independent, the
residual is *not* reduction order. The leading candidate is `torch._grouped_mm`
tiling: this tool hands the kernel a 36-expert **view** into a 288-expert slab
(`slab[36r : 36r+36]`), whereas each sealed rank held a standalone 36-expert
DTensor with its own base pointer and alignment. `--slab-experts 36` makes every
EP group a base-aligned standalone tensor and is the direct test; results are in
§7.3.

⇒ Under the best available model of the sealed reduction, `|Δ mean_kld| ≤ 1e-6`
is **still not met**, and it is very likely unsatisfiable by construction for any
scorer that does not reproduce the sealed node's exact kernel launches —
including a different 8× H200 box with a different NCCL version or topology. The
sealed number is reproducible *on that machine*; it is not portable across
reduction topologies.

### 7.3 The slab-layout hypothesis is refuted, and the schedule is numerically free

`--slab-experts 36 --ep-emulate 8 --reduce-order fp32` gives every EP group its
own base-aligned 36-expert tensor instead of a view into a 288-expert slab, and
decodes one group at a time:

```
slab-36  payload sha256 : fee81277dba27838f37eb190a5de002d2f4b5a1d086509edbf7fdab97605acc9
slab-288 payload sha256 : fee81277dba27838f37eb190a5de002d2f4b5a1d086509edbf7fdab97605acc9
slab-36 vs slab-288     : BITWISE IDENTICAL   (max abs 0.0, rms 0.0)
peak device memory      : 34.396 GB  (vs 47.08 GB at slab 288)
decode                  : 570.6 s    (vs 397.0 s at slab 288)
```

Two conclusions:

* The `torch._grouped_mm` tiling/alignment hypothesis for the residual is
  **refuted by measurement** — a 36-expert view and a standalone 36-expert
  tensor produce bit-identical output.
* **The memory schedule is numerically free.** Choosing the low-memory schedule
  to fit a 48 GB card costs 44 % more decode time and changes **no bit** of the
  result. `--vram-budget-gb` is therefore safe to use without a parity caveat.

The residual 1.5e-5 / rms 0.0381 therefore remains attributable to the combine
itself: it is **1/3 of one bf16 ULP** at the logit scale (one ULP at |logit|max
29.375 is 0.1147) in rms, with a max of 2.0 on a handful of positions — i.e. a
few surviving routing flips. It is bounded, characterised, and its two leading
alternative explanations (reduction order, kernel layout) have been measured and
eliminated. It is **not** explained down to the bit, and this document does not
claim it is.

### 7.4 Disclosure language for cards

The measured evidence does **not** support calling the streaming number a
reproduction of the sealed number. Use this instead, filling in the panel mean
from `receipts/stream-verdict.json` once L3 has run:

> Measured with a single-device streaming scorer over the identical sealed
> 25-window / 51,175-position panel, the identical fp64 tokenwise-KL estimator,
> and the identical sealed K6 surface (`checkpoint_identity_sha256`
> `a8668be3592493035e98a52994e0e3c43548a9757eadb79f7ae939f2f32de1c1`, verified
> equal to the sealed run's). Mean tokenwise KLD(teacher‖K6) = ⟨panel mean⟩.
> The sealed 8× H200 EP8 lane measured 0.013723384665701147 on the same panel
> and the same weights. The difference is the routed-expert combine: under
> expert parallelism each rank rounds its partial top-k sum to bf16 and the
> partials are reduced over NCCL, whereas a single device keeps one accumulator.
> Because top-8-of-288 routing is a discontinuous function of the hidden state,
> that ULP-scale difference amplifies over 42 routed layers — measured here by
> re-running one window under several combine orders on identical decoded
> weights, which spread by rms 0.26–0.28 in the logits among themselves. Both
> numbers are valid measurements of the same weights; neither is a defect in the
> other, and the streaming lane is bitwise deterministic across cold runs.

Rules for anyone writing a card from this:

* Do **not** write "reproduces" unless `receipts/stream-verdict.json` says
  `EXACT` or `WITHIN_1E-6`.
* Do **not** quote a Mac number without saying it used a different expert kernel
  (`grouped_mm_fallback`) and therefore cannot be bitwise comparable.
* **Do** quote `checkpoint_identity_sha256` — it is the one hash proving both
  lanes scored the same sealed surface, and it matches exactly.
* **Do** state the combine order used (`--reduce-order`), because it changes the
  number by more than the quantisation effect being measured does.

---

## 8. Running it on a 128 GB Mac

```bash
python3.12 -m venv ~/.venvs/glm53-stream && . ~/.venvs/glm53-stream/bin/activate
pip install torch "transformers==5.16.1" safetensors numpy accelerate
export PYTHONPATH=/path/to/pipeline/src

# raise the Metal wired limit so a 38 GB working set is allowed
sudo sysctl -w iogpu.wired_limit_mb=110000

python tools/stream_score.py \
  --source payload-store --packed-root /path/to/out-k6 \
  --bf16 /path/to/GLM-5.3-Flash-BF16 --teacher /path/to/teacher-final \
  --token-panel /path/to/calibration/panel-v1/panel.receipt.json \
  --out ~/stream-run1 --cold-run 1 --profile k6 \
  --device mps --unpack-device cpu \
  --ep-emulate 8 --reduce-order sequential \
  --decode-cache none --decode-threads 8 --windows final-0000
```

Notes, stated plainly:

* `--unpack-device cpu` runs `unpack_trellis_states` (pure int64 bit twiddling)
  on the CPU because MPS int64 coverage is partial. **L1.a proves that split is
  bitwise identical to the reader's own decode**, so it is a placement change,
  not a numerical one.
* `_can_use_grouped_mm` is false on MPS and CPU, so the experts run through
  `torch.ops.transformers.grouped_mm_fallback` — same algebra, different kernel.
  A Mac run therefore **cannot** be bitwise equal to a CUDA run, and cannot
  reproduce the sealed `tokenwise_kl` sha. It is independent corroboration, not
  parity. Say so on any card that cites a Mac number.
* The fp64 report on a Mac must use `--device cpu`; measured deviation from the
  `cuda:0` estimator is ~1e-13 tokenwise and ~1e-15 on window means — nine orders
  below the 1e-6 band, so the *number* is comparable even though the *hash* is not.
* You need ~230 GB of free disk for the payload store and ~40 GB of unified
  memory. `--decode-cache disk` needs a further 609 GB.

Offline, with no weights at all:

```bash
python tools/stream_score_selftest.py --only e        # no pipeline checkout needed
python tools/stream_score_selftest.py --packed-root … --fixture … --pipeline-root …
```

---

## 9. Receipts and disclosure

`stream_score.py` writes the same artefact set as the sealed capture:

```
<out>/plan.json              malaiwah.glm53-streaming-student-logit-capture-plan.v1
<out>/reader-identity.json   quant-pipeline.glm53-packed-k6-offline-reader-identity.v1
<out>/backend.json           malaiwah.glm53-streaming-offline-reader-backend.v1
<out>/logits/window-%04d.safetensors   fp32 [2047, 154880], key "logits"
<out>/capture-receipt.json   quant-pipeline.glm53-logit-capture.v1   (SEALED)
```

`capture-receipt.json` is schema-identical to the sealed one — same
`capture_role`, same `student_label` (`uniform-k6`), same ten-key `logit_files`
rows — so `k6_kld_report.py` consumes a streaming run unmodified. Use
`--profile k6-stream`: it keeps the `uniform-k6` label (so the per-run
`kld-report.json` is directly comparable to the sealed one) and takes the
`malaiwah.*` summary branch, because the sealed K6 receipt chain requires a
materialized checkpoint's `materialization-receipt.json` that a payload-store run
legitimately does not have.

Both `backend.json` and `capture-receipt.json` carry a
`streaming_disclosure` block (`malaiwah.glm53-streaming-disclosure.v1`)
enumerating streaming mode, device, dtype policy, EP semantics, reduce order,
experts implementation, and — explicitly — every difference from and every
identity with the sealed path. `backend.json` additionally records the measured
decode/forward seconds, peak memory, the decode-cache budget, the full choice
census hash, and the `combine_order_sweep` rows.

## 10. Reproducing the numbers in this document

```bash
ROOT=/home/jl_fs/glm53-k6
export PYTHONPATH=$ROOT/pipeline/src QP_PIPELINE_ROOT=$ROOT/pipeline NVIDIA_TF32_OVERRIDE=0
PY=$ROOT/venv/bin/python

# L1, offline, CPU only, ~40 s
$PY $ROOT/tools/stream_score_selftest.py \
    --packed-root /home/glm53-stream/out-k6 \
    --fixture $ROOT/fixture/GLM-5.3-Flash-0.1B-A0.1B \
    --pipeline-root $ROOT/pipeline --json /tmp/selftest.json

# L2, one window, best combine model, low-memory schedule, ~10 min
$PY $ROOT/tools/stream_score.py \
    --source payload-store --packed-root /home/glm53-stream/out-k6 \
    --bf16 /home/jl_fs/models/bf16 --teacher $ROOT/teacher-final \
    --out /home/glm53-stream/runs/l2 --cold-run 1 --profile k6 \
    --windows final-0000 --device cuda:0 \
    --ep-emulate 8 --reduce-order fp32 --slab-experts 36 --decode-cache none

# the combine-order sweep of 7.2 (one decode, four forwards, ~30 min)
#   add:  --slab-experts 288 --sweep "ep8:reverse,ep8:sequential,ep1:none"

# L3 + L4, the whole panel, one command
QP_STREAM_LOCAL_STORE=1 bash $ROOT/stage_k6.sh measure_stream
```

Everything above reads the shared filesystem read-only and writes only under
`/home/glm53-stream` (container-local), `$ROOT/receipts/stream-*` and
`$ROOT/logs/stage-stream.state`.

---

## 11. Constraints honoured

* No HF or GitHub token appears in any code path, argument or log.
* Nothing is published; `measure_stream` has no upload step and writes no
  done-marker.
* Sealed receipts are opened read-only; comparisons hash in place.
* `out-k8` is never read or written; machines 485565 / 485586 / 485016 are
  untouched. The streaming stage writes its state to
  `logs/stage-stream.state` (via `QP_STAGE_STATE`) so the K8 supervisor's
  `logs/stage.state` is never clobbered.
* Bulk output goes to container-local disk, never to the shared `/home/jl_fs`.
