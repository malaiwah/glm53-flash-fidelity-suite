# The layer-outer, window-inner capture engine

**Status: built, and proven bit-identical on two architectures and two devices.**
Run date 2026-08-30. Cost of the proofs: one L4 spot instance, created and
destroyed the same hour (§6). **No GLM-5.3 capture was run** — that is Stage B,
a separate budgeted decision, and nothing in this document is a GLM-5.3
measurement.

> **Amended 2026-08-30.** §7 items 8 and 9 changed after this schedule was run
> against three new architectures: quantized checkpoints are now a REFUSAL
> rather than an untested path (the FP8 reading of it was silent), and layer
> routing now goes through the architecture's conversion renames, which is what
> a VL checkpoint needs. `minimax_m3_vl` joins the bit-identity table below.
> Evidence: `docs/NEW-ARCHITECTURES-FEASIBILITY.md` §2.6.

| gate | result |
|---|---|
| bit-identical capture, 0.1B `glm5_next` fixture | **yes** — same `capture_content_digest`, at 48 and 2048 tokens |
| bit-identical capture, Fruit `glm_moe_dsa` (GLM-5.3's architecture) | **yes** — on CPU *and* on CUDA |
| bit-identical capture, `minimax_m3_vl` 4-layer truncation (added 2026-08-30) | **yes** — same `capture_content_digest` as three window-outer captures |
| `compare --self-compare --force-compute` across schedules | **exactly 0.0 nats**, max 0.0, top-1 1.0 |
| measured peak VRAM, Fruit on an L4 | 10.409 GB → **2.167 GB** (4.80×) |
| regression battery, before → after | **1 passed / 19 failed → 20 / 0** |
| GLM-5.3 peak VRAM, projected on those measurements | 81.7 GB → **~47–51 GB**, one H200 |
| GLM-5.3 Stage C cost | **$1.08–3.52**, against $38–96 for the existing lanes |

`docs/GLM53-ROOT-FEASIBILITY.md` §4 ended on one sentence: *"The layer-outer,
window-inner schedule that reads the tree once per run instead of once per
window is still the difference between a \$3 stage C and a \$38–96 one, and it
is still unwritten."* This document is that sentence closed. It states what was
built, the evidence that the numbers did not move, the memory that was
**measured** rather than projected, and — in §7 — an honest list of what the new
schedule does not handle.

---

## 1. What changed

`engines/tools/hf_capture.py` gains two flags. The default is unchanged and the
existing schedule is untouched.

```
--schedule {window-outer,layer-outer}      default: window-outer
--layer-residency {stream,resident}        default: stream   (layer-outer only)
--memory-report PATH                       write measured peak memory as JSON
```

The engine itself is `engines/tools/layer_outer.py`.

**window-outer** (the old path, still the default): load the model, then for
each window push it through the whole stack. For a checkpoint that does not fit
in memory this pays for the weights **once per window**.

**layer-outer**: `for each layer { load it once; for each window: push that
window through it; free it }`. Each layer's weights are materialised **exactly
once for the whole panel**, and only one layer is resident at a time.

### Windows are never batched

The saving is in weight *loading*, not compute. Windows are pushed through each
layer **sequentially, one at a time**. Stacking them into one matmul would go
faster and would change the reduction order, and therefore the numbers. This
engine exists to make a measurement *possible*; a measurement whose numbers
moved is worth nothing. The refusal to batch is the load-bearing design
decision in the file.

---

## 2. Why it is bit-identical by construction, not by luck

The obvious implementation re-writes the model's forward pass: embeddings,
position ids, the causal-mask mapping, rotary tables, the per-layer kwargs, the
carried state, the final norm. Every one of those is a chance to differ from
`transformers` by a detail, and two of them bite precisely on the architecture
that matters:

* `GlmMoeDsaModel.forward` threads a **second** value between layers —
  `hidden_states, topk_indices = decoder_layer(..., prev_topk_indices=topk_indices)`.
  That is the DSA indexer's shared top-k selection, recomputed only by the
  `full` indexer layers and reused by the `shared` ones. GLM-5.3 has a `full`
  indexer every 4th layer; Fruit has three. A re-implementation that knows
  about "hidden states" and not about this is silently wrong.
* `Glm5NextTextModel.forward` carries a hyper-channel dimension (`hc_mult`, 4 on
  the 0.1B fixture) and builds its masks with a different function.

So `layer_outer.py` **re-implements nothing**. It runs the model's own
`forward`, once per (layer, window), and replaces only the decoder layers with
proxies:

| proxy for layer index | behaviour |
|---|---|
| below the layer being computed | returns, **verbatim**, the value the layer below returned on the previous outer iteration — the whole return value, whatever its shape, so `topk_indices` and any other carried state ride along untouched |
| the layer being computed | calls the real layer, memoises its return value |
| above it | raises `_Suspend`, which unwinds the forward |

The model's own prologue builds the embeddings, position ids, masks and rotary
tables. The model's own loop body computes the per-layer kwargs and threads the
carried state. On the **last** layer there is no proxy left above to suspend, so
the model runs straight on into its own final norm and head — that is the
epilogue, executed by the model's own code, with `hf_capture`'s head pre-hook
firing exactly as it does on the window-outer path. The only thing this file
decides is *when* each layer runs.

The price is that the prologue is recomputed once per (layer, window) rather
than once per window: an embedding gather, a mask build and a rotary table,
against a layer of a 753B-parameter MoE. It is paid on purpose, to buy an
implementation that cannot drift from the model's own code.

### The streaming residency

Reordering the loop saves nothing if the model is fully resident, so the engine
also builds the model on the **meta device** (through `cls.get_init_context(...)`
— the same context managers `from_pretrained` uses) and materialises one layer
at a time through `transformers`' own
`convert_and_load_state_dict_in_model`. Reusing the library's converter rather
than re-deriving the 256→1 expert fusion by hand is what makes the streamed
weights byte-identical to the `from_pretrained` weights. That identity is
asserted directly, parameter by parameter, by `selftest_layer_outer.py` L4 and
L6 — not assumed.

Everything that is not a decoder-layer parameter — embeddings, final norm, head,
and **every buffer, including per-layer ones** — is loaded once and stays
resident. Buffers are rotary tables and router correction biases: kilobytes
against gigabytes, and streaming them would add a way to get a forward pass
wrong for no saving at all.

`--layer-residency resident` runs the new loop order over a fully loaded model.
It buys nothing operationally and exists so that a digest mismatch can be
attributed: `resident` isolates the loop, `stream` adds the loader.

---

## 3. Gate 1 and Gate 2 — the bit-identity proofs

The gate is `capture_content_digest`: the digest over **tensor content**, not
the container. Both schedules, same panel, same checkpoint, same process
version.

### Gate 1 — `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B`

`glm5_next`, 5 layers, hidden 256, `hc_mult` 4, `layer_types` mixing
`linear_attention` and `deepseek_sparse_attention`, 8 routed experts, vocab
154,880. Revision `7c3a6d3dc51732dd8ab230888e06ba8c93a381ac`.

| panel | window-outer | layer-outer (stream) | equal |
|---|---|---|:--:|
| 3 windows × 48 tokens | `4755a01a4712dc251d6ac5a2b87440db403c2241b65fec9e042eba7166423e16` | `4755a01a4712dc251d6ac5a2b87440db403c2241b65fec9e042eba7166423e16` | **yes** |
| 2 windows × 2048 tokens (`panel--glm53.stagea.smoke.2w`) | `efca1227fca0bc934d2b86d74c02c1bfce21fa0eea5d6d28f2ad06394cd9cddb` | `efca1227fca0bc934d2b86d74c02c1bfce21fa0eea5d6d28f2ad06394cd9cddb` | **yes** |

### Gate 2 — `malaiwah/GLM-5.2-SIQ-Fruit-bf16` — the load-bearing one

`glm_moe_dsa`: **the architecture GLM-5.3 uses.** 13 layers, hidden 1024, 256
routed experts, `first_k_dense_replace` 3, `indexer_types` = 3 × `full` then
10 × `shared` (so the `prev_topk_indices` carry is exercised on ten layers),
`num_nextn_predict_layers` 1, vocab 154,880. 10.1 GB, revision
`ef68013aa6e16453cf52b5b77647f72fbe258c3c`. Panel:
`panel--glm53.stagea.smoke.2w`, 2 windows × 2048 tokens, 4,094 scored rows.

| device | window-outer | layer-outer (stream) | equal |
|---|---|---|:--:|
| CPU (M4 Max) | `d2917092d8cc604a547c501747d7b12cde98d0f6545f8a1eab06e729c7fef914` | `d2917092d8cc604a547c501747d7b12cde98d0f6545f8a1eab06e729c7fef914` | **yes** |
| CUDA (L4, torch 2.11.0+cu130) | `aeb896b02afe4565e704f98c30b006dd04cf9285c5875ac09645bdbe8b5e228b` | `aeb896b02afe4565e704f98c30b006dd04cf9285c5875ac09645bdbe8b5e228b` | **yes** |

The CUDA digest differs from the CPU digest for the same model and panel
(`aeb896b0…` vs `d2917092…`) because a CUDA kernel and a CPU kernel are
different stacks — the ordinary cross-stack residual this suite already models
with `stack_fingerprint`. That is not a schedule effect. The gate is that **the
two schedules agree on each device**, and they do, on both.

The CUDA run is not a nicety. The whole point of this engine is to run on an
H200, and the streamed loader's `device_map={"": "cuda:0"}` path is not
exercised by any CPU run. Shipping it CPU-proven only would have left a hole in
exactly the place the engine exists for.

Machine-readable: `engines/tools/layer-outer-evidence/bit-identity.json`.

---

## 4. Gate 3 — the comparator says exactly 0.0

`bin/fidelity-dataset compare --self-compare` on the two schedules' captures,
with `--force-compute` so the arithmetic actually runs rather than
short-circuiting on a digest match:

```
REPRODUCTION CONFIRMATION
  metric              mean_tokenwise_kld = 0.0 nats
  direction           KL(reference || candidate)
  top-1 agreement     1.0
  kl                  {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0, "max": 0.0}
  estimator           full vocabulary, float64, head_policy=shared_reference_head, stack=same_stack
  backend             torch:k6_kld_report._token_kld
  comparability       class=strict same_lane=True usable_as_floor=True
```

Every order statistic is 0.0, not just the mean: a mean of zero over a
tokenwise distribution with a non-zero max would be cancellation, not identity.
`selftest_layer_outer.py` L14b/L14c assert `metric.value`, `kl.mean`, `kl.max`
and `top1_agreement` together, for exactly that reason.

**Note on `stack_fingerprint`.** The schedule is recorded in
`runtime.capture_tool.{schedule,layer_residency,mechanism}` and deliberately
**not** in `stack_fingerprint`. `dscompare` reads the fingerprint to decide
`stack_relation`, and a `cross_stack` verdict stamps `usable_as_floor: false`
and attaches a 1e-2-class bias block. Charging a capture that penalty for a
schedule the digests prove is bit-identical would be asserting a difference that
is not there. It is still written down, in the sealed receipt, where a reader
can see which loop produced their tensors.

---

## 5. Gate 4 — measured memory, and the two numbers that are not the same

### Read this before reading the table

On the CPU path `safetensors` **mmaps** the shards, so every byte the loader
touches becomes file-backed resident memory that the OS is free to evict but
`ru_maxrss` counts anyway. Stage A already flagged this ("safetensors mmaps the
shards, so an unknown part of that 22.3 GB is evictable file-backed page
cache"). A layer-outer run therefore shows a CPU RSS close to the whole
checkpoint size even though it never holds more than one layer of anonymous
weights — the RSS is real, but what it is measuring there is the page cache,
not the schedule.

So the engine reports **two** figures, and a third on CUDA:

* `peak_rss_bytes` — the OS high-water mark. Honest, and on CPU confounded by
  the page cache. Units are named in the report (`ru_maxrss` is bytes on Darwin,
  kilobytes on Linux; a silent 1024× here would rent the wrong machine).
* `peak_resident_weight_bytes` — the maximum, over the run, of *materialised*
  parameter + buffer bytes. Arithmetic, not sampling, and not confounded by
  anything. This is what the schedule controls.
* `peak_cuda_allocated_bytes` — the allocator's own peak on CUDA. **This is the
  authoritative number**: it includes activations and workspace and has no page
  cache to confuse it.

### Fruit (10.1 GB, 13 layers), 2 windows × 2048, CPU (M4 Max)

| schedule | peak RSS | peak resident weights |
|---|---:|---:|
| window-outer | 19.982 GB | **9.144 GB** |
| layer-outer (stream) | 11.603 GB | **1.471 GB** |
| ratio | 1.72× | **6.22×** |

The resident-weight figure is the one to read. 9.144 GB is the whole model
`transformers` builds (10.1 GB of checkpoint minus the MTP layer 13 it does not
build). 1.471 GB is the resident set (embeddings + head + norms, 0.68 GB) plus
one sparse layer (0.79 GB) — measured with that layer loaded, i.e. at the moment
the schedule holds the most it ever holds.

The RSS gap decomposes cleanly and confirms the page-cache reading:
window-outer ≈ 9.1 GB anonymous + ~10.1 GB of mmap page cache ≈ 20 GB;
layer-outer ≈ 1.5 GB anonymous + the same ~10.1 GB of page cache ≈ 11.6 GB.
Both runs read the tree once; only one of them keeps it.

### 0.1B fixture, CPU, 2 windows × 2048

| schedule | peak RSS | peak resident weights |
|---|---:|---:|
| window-outer | 1.678 GB | 0.169 GB |
| layer-outer (stream) | 1.688 GB | 0.163 GB |

**There is nothing here for the schedule to save, and that is expected.** The
fixture is 189 MB and its embedding and head are 79 MB apiece (154,880 × 256 ×
2 B) against ~2 MB of layer weights, so the resident set *is* the model. Stated
explicitly so the fixture's flat numbers are not mistaken for a null result:
**the fixture proves correctness, Fruit proves memory.**

---

## 6. The CUDA run — the authoritative memory numbers

One NVIDIA L4 (24 GB, JarvisLabs IN2 spot, $0.29/h), torch 2.11.0+cu130,
transformers 5.16.1, instance destroyed after the run (verified with `jl list`).
Fruit, `panel--glm53.stagea.smoke.2w`, 2 windows × 2048.

| schedule | **peak CUDA allocated** | peak CUDA reserved | peak resident weights | peak RSS |
|---|---:|---:|---:|---:|
| window-outer | **10.409 GB** | 11.096 GB | 9.144 GB | 17.499 GB |
| layer-outer (stream) | **2.167 GB** | 3.712 GB | 1.471 GB | 11.420 GB |
| ratio | **4.80×** | 2.99× | 6.22× | 1.53× |

`peak_cuda_allocated_bytes` is the number to quote: no page cache, and it
includes activations and workspace. The decomposition is clean:

| term | window-outer | layer-outer |
|---|---:|---:|
| resident weights (embeddings + head + norms) | 0.683 | 0.683 |
| decoder-layer weights held at peak | 8.461 (all 13) | 0.788 (one sparse layer) |
| everything else at peak (activations, logits buffer, loader transient, carried state) | 1.265 | 0.696 |
| **total allocated** | **10.409** | **2.167** |

Two things worth reading out of that last row. First, the layer-outer
non-weight term (0.696 GB) is dominated by the epilogue's logits buffer —
2048 × 154,880 × 2 B is 0.634 GB on its own — which means **on Fruit the
loader's expert-fusion transient never exceeded the epilogue peak**. Second,
that does *not* generalise: Fruit's routed set per sparse layer is 0.805 GB
while GLM-5.3's is 19.33 GB (24×) at the same vocabulary, so on GLM-5.3 the
fusion transient may well be the peak. §8 budgets it conservatively rather than
extrapolating the happy reading.

### Wall clock, as observed (not a projection)

| | window-outer | layer-outer |
|---|---:|---:|
| L4, per window (2048 tokens) | 0.678 s, 0.222 s | 0.016 s, 0.024 s |
| M4 Max CPU, per window | 26.51 s, 26.31 s | 3.32 s, 3.34 s |
| M4 Max CPU, per-layer load (sparse layer, 786 checkpoint tensors) | — | 0.086–0.120 s |

The layer-outer per-window figure is the sum of that window's forward time
across all layers, prologue recomputation included. It is *smaller*, not larger,
on both devices — the reordered loop keeps a layer's weights hot across the
whole panel instead of streaming the entire model past the cache once per
window. This is a welcome side effect and not the point; the point is memory,
and a speed claim from a 2-window panel on a 10 GB model should not be scaled to
a 25-window panel on a 1.5 TB one.

---

## 7. What this schedule does NOT handle

Stated plainly, because a list of limits is part of the deliverable.

1. **It does not compose with `--device-map`, and refuses the combination.**
   `--device-map` hands the model to `accelerate`, which attaches
   `AlignDevicesHook`s that move weights per module call; the layer-outer
   streamer owns residency itself and would be racing those hooks for the same
   parameters. `hf_capture` exits 3 with that explanation rather than letting
   them fight. They are also not complementary: `--device-map` exists (feasibility
   R2) because the window-outer loop cannot fit the model, and layer-outer
   removes that need on a single device.
2. **It is single-device.** There is no multi-GPU layer-outer. For GLM-5.3 that
   is fine — §8 shows one H200 fits — but a model whose *non-routed* set
   exceeded one device would need work that does not exist here.
3. **It does not handle MTP.** Not because it was skipped, but because
   `transformers` 5.16.1 does not build the next-token-prediction layer at all:
   GLM-5.3's layer 78 (791 tensors, 18.5 GiB) and Fruit's layer 13 land in
   `unexpected_keys`. The streamer explicitly counts checkpoint tensors
   addressed to a layer index the model does not build as unexpected, so the
   `checkpoint_tensors_not_loaded` disclosure the window-outer path emits is
   still emitted here. If a future `transformers` *does* build the MTP layer, it
   will appear in the layer list and be streamed like any other — but that is
   untested and must not be assumed.
4. **It assumes the decoder stack is a plain in-order loop, each layer called
   exactly once per forward.** A model that calls a layer twice (depth-tied
   weights, a head reusing a block) is **refused**, not captured
   (`selftest_layer_outer.py` L12b). This is a real restriction, and it is
   enforced rather than documented-and-hoped.
5. **It assumes one decoder stack, findable by structure.** `find_decoder_layers`
   takes the `nn.ModuleList` named `layers` whose parent also owns
   `embed_tokens` — which distinguishes the text stack from a vision tower's
   blocks. Zero or several matches is a refusal, not a guess.
6. **The holes guard has a floor.** `audit_checkpoint_tree` catches a shard
   shorter than its own safetensors header, and a shard-header / index key-set
   disagreement in either direction. It does **not** catch a shard of the right
   length whose bytes were written as zeros (a sparse-file fetch), or any
   corruption that preserves length. Only a content digest catches those, and
   the checkpoint identity `hf_capture` already computes is that digest — for
   the tree as a whole, once.
7. **It depends on private `transformers` API.** `--layer-residency stream`
   needs `core_model_loading.convert_and_load_state_dict_in_model`,
   `modeling_utils.{LoadStateDictConfig,_load_parameter_into_model,patch_output_recorders}`
   and `conversion_mapping.get_model_conversion_mapping`. That coupling is
   deliberate — it is the only way the streamed expert fusion is byte-identical
   rather than re-derived — but it is a pin to a `transformers` version.
   A build that does not expose them gets a refusal naming exactly what is
   missing and pointing at `--layer-residency resident`, not a silent fallback
   to hand-rolled loading.
8. **Quantized checkpoints are REFUSED on this path** (amended 2026-08-30,
   `docs/NEW-ARCHITECTURES-FEASIBILITY.md` §2.6c). This item used to read
   "untested"; running the schedule at `deepseek-ai/DeepSeek-V4-Flash-0731`
   showed that "untested" was too kind. The loader passes `hf_quantizer=None`,
   so the quantizer's module replacement, its `*.scale` ->
   `*.weight_scale_inv` rename and its dequantization op are all absent. For a
   packed format that raises a shape mismatch; **for a plain FP8 E4M3 weight the
   shape MATCHES the bf16 parameter it lands in, the payload is read as bf16 and
   the block scale is never applied** — the M1 Qwen3.8-27B-FP8 defect, whose
   only signal is `unexpected_keys`, behind the flag a truncated tree already
   needs. `build_streamed_model` now refuses on the config's own
   `quantization_config` before any weight is read. `selftest_layer_outer.py`
   L15; on the pre-amendment tree that rung captures and says nothing.
9. **Layer routing is done on the RENAMED checkpoint key** (added 2026-08-30).
   `layer_pattern` comes from the MODEL's stack path; a VL checkpoint may spell
   that path differently (`MiniMaxAI/MiniMax-M3` ships
   `language_model.model.layers.N.` for `model.language_model.layers.N.`), and
   matching the raw key put every layer tensor in the resident load. Each key is
   now passed through the architecture's own conversion RENAMES before the
   pattern is applied — the same mapping `convert_and_load` uses on those keys a
   few lines later. Architectures whose names already match are unaffected by
   construction, and `minimax_m3_vl` now reproduces the window-outer capture
   bit-for-bit (same `capture_content_digest`, `--force-compute` self-compare
   exactly 0.0).
10. **It has not been run on GLM-5.3.** That is Stage B, and it is a separate,
   budgeted decision. Nothing in this document is a GLM-5.3 measurement.

---

## 8. The corrected GLM-5.3 projection

**Nothing here is a GLM-5.3 measurement.** It is the feasibility document's own
census arithmetic — which reconciles to the published `metadata.total_size` with
a delta of **zero bytes** — recombined against the *residency split this engine
actually implements*, plus the non-weight terms measured in §6.

### 8.1 Peak VRAM: 81.7 GB projected → **~47–51 GB**

The §2 projection assumed the whole **non-routed set (37.78 GB) stays
resident** and only the routed experts stream. That is not what this engine
does. It streams *whole layers* — attention/MLA, shared expert, router gate, DSA
indexer, layer norms **and** routed experts — so the permanently resident set is
only `embed_tokens + lm_head + model.norm`:

| term | GB | source |
|---|---:|---|
| resident: embed + lm_head + final norm | **3.81** | census (§2), exact |
| largest single layer (sparse + full indexer, 18.398 GiB) | **19.76** | census (§2), exact |
| = peak resident **weights** | **23.56** | |
| carried state, 25 windows (hidden 2048×6144 bf16 + topk_indices 2048×2048 int64) | 1.47 | arithmetic, int64 assumed |
| epilogue logits buffer, 2048 × 154,880 × 2 B | 0.63 | arithmetic |
| within-layer activations/workspace at hidden 6144, 64 heads, ctx 2048 | 2.0–3.0 | budgeted |
| **expert-fusion transient during a layer load** | **19.3–22.3** | Stage A §9.5, upper-bounded |
| **peak** | **47.0–50.9** | |

The fusion transient is the least-measured term and the one that dominates the
range. Stage A measured 48.2 GB RSS against 25.95 GB resident while loading the
truncated GLM-5.3 tree — a 22.3 GB excess, 1.15× one sparse layer's 19.33 GB of
routed experts — and called it an upper bound because part of it is evictable
page cache. §6 shows the same term on Fruit is small enough to hide under the
logits buffer, but Fruit's routed set per layer is 0.805 GB against GLM-5.3's
19.33 GB (24×) at the same vocabulary, so **the happy reading must not be
extrapolated**. The table budgets the pessimistic one.

| box | VRAM | §2's 81.7 GB projection | **this engine, 47–51 GB** |
|---|---:|:--:|:--:|
| H200 | 141 GB | fits, 59 GB spare | fits, **~90 GB spare** |
| RTX PRO 6000 | 96 GB | fits | fits, ~45 GB spare |
| H100 | 80 GB | **no** | **fits**, ~29 GB spare |
| L4 | 24 GB | no | no |

**Recommendation: still one H200 for Stage B/C.** H100 80 GB now looks
feasible and RTX PRO 6000 spot is half the price of H200 spot ($0.99 vs $1.99),
but the term that decides whether the tightest box works is the one term that
has not been measured at scale. Buy the headroom; it costs about a dollar.

If the transient ever needs removing, the mechanism is known and unimplemented:
load a layer's 256 experts in chunks instead of collecting all 768 source
tensors before stacking. That would take the peak to ~28 GB. It is not needed
for an H200 and was not built.

### 8.2 Minutes per window: 13–26 → **0.4–1.6**

The cost model is unchanged — bytes moved, not FLOPs — but the bytes are now
read **once per run** instead of once per window. §4's table for a layer-outer
schedule stands, and §6 lets two of its assumed terms be replaced by observation.

| term, per COLD RUN | minutes | basis |
|---|---:|---|
| read 1,486.8 GB once | 8.3–23.6 | §4, at 3.0–1.05 GB/s |
| compute, 25-window panel (4.11 PFLOP) | 0.3–0.9 | §4, H200 |
| per-tensor fusion overhead, 75 sparse layers | **0–12.5** | see below |
| **total per cold run** | **~9–37** | |
| **per window at 25 windows** | **0.4–1.6** | vs 13–26 window-outer |

**The fusion-overhead term is new and is a warning, not a reassurance.** Fruit's
per-layer load cost 0.086–0.120 s for 786 checkpoint tensors reading from page
cache — about 0.13 ms per tensor of pure per-tensor overhead, independent of
size. A GLM-5.3 sparse layer has **76,800** source expert tensors (256 experts ×
3 matrices × 100 shards' worth of addressing), which extrapolates to ~10 s per
layer, ~12.5 minutes over 75 layers — the *same order as the IO*. It may
overlap with IO (the loader uses a thread pool) or it may not.

**This is the single thing Stage B should measure and not trust.** It is also
cheap to measure: the first sparse layer's `layer_load` log line reports its own
seconds, so a Stage B run answers it in the first minute.

### 8.3 Cost: Stage C confirmed at **under $4**

At H200 spot $1.99/GPU-h + storage, i.e. **$2.28/h**, and a one-time fetch of
$0.40–0.56:

| stage | runs | h | GPU cost | + fetch | **total** |
|---|---:|---:|---:|---:|---:|
| **B-3** 8-window panel, 2 cold runs | 2 | 0.3–1.3 | $0.68–2.96 | $0.40–0.56 | **$1.08–3.52** |
| **C-3** 25-window panel, 2 cold runs | 2 | 0.3–1.3 | $0.68–2.96 | $0.40–0.56 | **$1.08–3.52** |

**B-3 and C-3 cost the same**, because a layer-outer run's cost does not move
with the window count — the checkpoint is read once either way and the extra
compute is under a minute. That is the whole economic point of the schedule, and
it is now a property of code that exists rather than of a projection.

This confirms §6's C-3 estimate of **$2.45–2.93** as the right order and the
right decision. Against the alternatives in that table — C-2 at $38.23–51.59,
C-1 at $96.32 on a $90.32 balance — the engine is worth roughly **$35–95** on
Stage C alone, and it makes the run preemption-tolerant as a bonus: a 10–37
minute job that loses its spot instance is a cheap retry, a 15–21 hour one is
not.

Disk is also simpler: 1,506.7 GB fetched once, inside the 2,048 GB instance cap,
with **no second copy** (unlike B-2's accelerate disk-offload, which needed
~3.0 TB across an instance disk *and* a shared filesystem) and **no multi-GPU**
(unlike B-1's 8× H200).

### 8.4 What would falsify this

* the per-tensor fusion overhead (§8.2) being IO-serial rather than overlapped —
  would roughly double per-run wall clock, and C-3 would cost ~$5 instead of ~$3;
* the expert-fusion transient exceeding 22.3 GB on real GLM-5.3 layers — would
  push peak past 51 GB, still far inside an H200 but out of an H100;
* `transformers` changing the private loading API §7.7 names — would fall back to
  `--layer-residency resident`, which does not save memory, and the engine would
  need reworking.

None of these is a correctness risk. The numbers this engine produces are the
window-outer numbers, on two architectures and two devices; what is uncertain is
only what the run will cost.

---

## 9. The regression battery

`bin/selftest_layer_outer.py`, 20 cases, offline, no network and no GPU. Wired
into `bin/selftest_all.sh`. `bin/selftest_hf_capture.py` (25 cases, the capture
engine's own battery — including Stage A's own A17–A22 regressions) existed and
was **never wired into `selftest_all.sh`**; nothing ran it. It was wired in at
the same time. A test no runner invokes is a document, not a test.

The battery goes from **50 passed / 0 failed** to **52 passed / 0 failed / 0
skipped** (the two newly wired entries), and `cd registry && make check` stays at
**0 errors** (84 self-tests, 440 joint-standard checks).

**Fail-without-fix, verified by running the battery against `git archive HEAD`
of the parent commit** (the new test file copied in, everything else the
pre-change tree):

| tree | result |
|---|---|
| before the change | **1 passed, 19 failed** |
| after | **20 passed, 0 failed** |

The single case that passes before is L9 ("a streamed layer whose checkpoint
tensors are absent is REFUSED") — it passes vacuously, because the old tree
refuses the whole invocation on the unrecognised `--schedule` flag. Named here
so the 1 is not read as coverage.

The rest fail in two shapes: an argparse refusal of `--schedule` /
`--memory-report` for the CLI cases, and `No module named 'layer_outer'` for the
mechanism cases. The import is guarded on purpose — a bare `ImportError` would
abort the file before a single case reported, which is the one shape of evidence
that cannot be read as "these cases fail without the fix".

---

## 10. Reproducing the gates

```bash
# Gate 1 -- the 0.1B fixture (public, 189 MB)
FIX=$(bin/fixture | tail -1)
for S in window-outer layer-outer; do
  .venv/bin/python engines/tools/hf_capture.py \
    --model "$FIX" --model-revision 7c3a6d3dc51732dd8ab230888e06ba8c93a381ac \
    --panel <a 2048-token panel tree> --out /tmp/ds-$S \
    --role root --lane local-cuda-budget \
    --dataset-id dataset--gate.$S --dataset-name "gate $S" --device cpu \
    --weights-repository inference-optimization/GLM-5.3-Flash-0.1B-A0.1B \
    --schedule $S --memory-report /tmp/mem-$S.json --force
done
# the two capture_content_digests must be EQUAL
python3 -c "import json;print(json.load(open('/tmp/ds-window-outer/fidelity-dataset.json'))['capture']['capture_content_digest'])"

# Gate 3 -- and the comparator must say exactly 0.0
bin/fidelity-dataset compare --reference /tmp/ds-window-outer \
  --candidate /tmp/ds-layer-outer --self-compare --force-compute --out /tmp/cmp

# Gate 5 -- the regression battery
bin/selftest_layer_outer.py            # 20 passed, 0 failed
FIDELITY_PYTHON=$PWD/.venv/bin/python bash bin/selftest_all.sh   # 52 / 0 / 0
cd registry && make check              # 0 errors
```

Gate 2 needs `malaiwah/GLM-5.2-SIQ-Fruit-bf16` (10.1 GB, an HF token) and the
CUDA row needs a GPU; the recipe is identical apart from `--model` and
`--device cuda:0`.
