# Can our capture engine read three architectures it has never seen?

**Stage A, run 2026-08-30 on the operator's M4 Max. $0.00 spent. No GPU was
rented, no JarvisLabs instance was touched, nothing was published, no registry
row was written.** Every model below is truncated to a handful of layers, and a
KLD measured on a truncation is arithmetic about a truncation, not fidelity. The
only KLD reported here is a **self-compare**, whose correct value is exactly 0.0
and whose job is to prove the chain closes.

Method and instruments follow `docs/GLM53-ROOT-FEASIBILITY.md` §9, extended
where these checkpoints do things GLM-5.3 does not.

| | `deepseek-ai/DeepSeek-V4-Flash-0731` | `MiniMaxAI/MiniMax-M3` | `Qwen/Qwen3.8-Flash-Next` |
|---|---|---|---|
| `model_type` | `deepseek_v4` | `minimax_m3_vl` | `qwen4_exp` |
| published root | 166.9 GB, 43L / 4096 / 256E / 129,280 | 869.2 GB, 60L / 6144 / 128E / 200,064 | 360.0 GB, 48L / 2560 / 512E / 248,320 |
| root quantized? | **YES**, FP8 attention + FP4 experts | no, BF16 | no, BF16 |
| unquantized reference exists? | **NO — nowhere, from anyone** | n/a (the root is BF16) | n/a (the root is BF16) |
| loads through `hf_capture.load_model`? | **only after a fix** (§1.2) | yes | yes |
| weights byte-exact after load? | **3,176 / 3,176** | **960 / 960** | **362 / 362** |
| capture + self-compare == 0.0? | yes | yes | yes |
| layer-outer == window-outer, bit for bit? | **cannot run** — the schedule has no quantizer, and now refuses (§2.6c) | **yes**, same digest, 4 captures | not tested (§5) |
| **verdict** | **GO for the engine, NO-GO for a root dataset** | **GO** | **GO-WITH-WORK** |

**Total moved over the network: 48.58 GB** for all three, against 1,396 GB of
published roots — 3.5%. Wall clock, fetch: 7 min 36 s. Peak scratch disk 81 GB,
returned in full afterwards (§4).

---

## 0. What was built to make this possible

Four instruments changed. All four have regression tests that fail on the
pre-change tree, verified by running the new tests inside a `git archive HEAD`
scratch tree.

| change | why | test | on HEAD |
|---|---|---|---|
| `fetch_truncated_ckpt.py` `--layer-key-regex` / `--drop-key-regex` / `--config-node` / `--config-index-list` | three checkpoints, three different decoder-layer key shapes, two different homes for `num_hidden_layers` | `bin/selftest_fetch_truncated.py` F1–F10 | **3 pass, 7 fail** |
| `hf_capture.py` `--drop-parallel-plan` + a named refusal | transformers 5.16.1 cannot load ANY FP8 `deepseek_v4` repo (§1.2) | `bin/selftest_hf_capture.py` A25–A27b | **4 fail** |
| `verify_loaded_weights.py` (new) | the byte check for a root that ships quantized (§1.3) | exercised on all three checkpoints below | did not exist |
| `layer_outer.py` refuses a quantized checkpoint; routes layer keys through the architecture's renames | the schedule silently captured a quantized tree, and could not find a VL decoder stack (§2.6) | `bin/selftest_layer_outer.py` L15 (+ L1–L14 unchanged) | **L15 reads `rc=0`** |

`bin/selftest_all.sh` is **57 passed, 0 failed, 0 skipped**; `registry/ make check` is **87 passed, 0 failed** plus 444 joint-standard checks with 0 errors.

### 0.1 The silent one

The old fetcher hard-coded GLM-5.3's decoder-layer key,
`^model\.layers\.(\d+)\.`. DeepSeek ships `layers.N.attn.wq_a.weight` — no
`model.` prefix. Under the old regex **not one key matched**, so every key
counted as a non-layer key that must be kept, and `--layers 4` would have
planned a fetch of the **entire 166.9 GB checkpoint** while logging
`kept_tensors 72317` as though that were a truncation. It now refuses:

```
--layer-key-regex '^model\\.layers\\.(\\d+)\\.' matched NONE of this checkpoint's
72317 tensor names, so nothing would be truncated and the fetch would be the
WHOLE model.
```

F4 is that case, and it is the rung that reads `rc=0` on the pre-change tree.

### 0.2 CAPTURE-03, fired on real published weights

`bin/selftest_hf_capture.py` A14–A24 already hold the load-report guard on
fixtures. Three of the four ways it can fire were reached here on the real
checkpoints, and each one refused with **no dataset written**:

| guard | how it was reached | outcome |
|---|---|---|
| `unexpected_keys` | Qwen3.8-Flash-Next capture without `--allow-unexpected-tensors` (23 tensors of layers 1 and 9, sharing the fetched shards) | `REFUSED`, naming the keys and the usual cause; `REFUSED [capture_failed]: the capture exited 1; no dataset written` |
| the new FP8 parallel-plan refusal | DeepSeek-V4-Flash-0731 capture without `--drop-parallel-plan` | `REFUSED`, naming `update_tp_plan`, the remedy, and the `deepgemm_megamoe` trap |
| `mismatched_keys` | DeepSeek-V4-Flash-0731 under `--schedule layer-outer`, where the quantizer is absent | `Reinit due to size mismatch`, raised — the failure §2.6(c) turns into a refusal of our own |
| header/index disagreement | MiniMax-M3 under `--schedule layer-outer` on the sparse tree | `REFUSED`: 1,091 tensors named only by a header |

`missing_keys` was not reached: none of the three checkpoints is missing
anything the architecture builds, which is itself the result.

---

## 1. `deepseek-ai/DeepSeek-V4-Flash-0731`

**Verdict: GO for the engine — the load path, the converter, the FP4/FP8
decode and the capture chain are all now proven correct on real published
bytes. NO-GO for a root fidelity dataset, on a cause that has nothing to do
with our engine: this family has no unquantized anchor and cannot be given
one.**

### 1.1 The `quantization_config` question, answered

The root's config carries `quantization_config: {quant_method: "fp8", fmt:
"e4m3", scale_fmt: "ue8m0", weight_block_size: [128,128]}` **and**
`expert_dtype: "fp4"`. Read off the published safetensors dtype census, that
means:

| component | storage | logical params |
|---|---|---:|
| routed + MTP experts | **FP4 E2M1, two values per `int8` byte**, per-row/32-col UE8M0 scale | 296,352,743,424 |
| attention, shared expert, `q/kv/o` projections | **FP8 E4M3**, 128x128 UE8M0 block scale | 6,304,038,912 |
| embeddings, head, norms, router gates | BF16 | 1,483,567,488 |
| hyper-connection tensors, `e_score_correction_bias`, sinks | F32 | 37,741,630 |

**This is not a quantization of a BF16 release. It is the release.** Checked
against every repository on the Hub that could be the missing reference:

| candidate | what it actually is |
|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | FP8 + FP4 experts, `quantization_config` present |
| `deepseek-ai/DeepSeek-V4-Flash-Base` | FP8, `expert_dtype: "fp8"`, `quantization_config` present |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | FP8 + FP4, `quantization_config` present |
| `deepseek-ai/DeepSeek-V4-Pro` (1.599 T), `-Pro-0813` (1.650 T), `-Pro-DSpark` (1.650 T) | `quantization_config` present, `expert_dtype: "fp4"` |
| `deepseek-ai/DeepSeek-V4-Pro-Base` (1.601 T) | `quantization_config` present, `expert_dtype: "fp8"` |
| `mlx-community/DeepSeek-V4-Flash-bf16` | **misnamed**: `quantization_config: {"bits": 4}`, `U8`/`U32` payload — a 4-bit MLX repo |
| `mlx-community/deepseek-ai-DeepSeek-V4-Flash-fp16` | **empty**: one file, `.gitattributes` |

Every DeepSeek-published `deepseek_v4` repository carries a
`quantization_config`. No `-BF16` sibling exists, and the two Hub repositories
whose names promise one do not deliver it.

**What that changes.** GLM-5.3 gave us `zai-org/GLM-5.3-BF16` and
`zai-org/GLM-5.3` — a root and a candidate whose configs are byte-identical
except for the quantization block, so "how far is this quant from the
publisher's own full-precision weights" is a question with an answer. Here
there is no such pair. The 100 quant children of
`DeepSeek-V4-Flash-0731` are quantizations **of an already-quantized artifact**,
and a root dataset captured from it would measure distance-from-the-published-
FP4-root, not distance-from-full-precision. That is a defensible quantity, but
it is a **different** quantity, it is not comparable to any GLM-5.3 or Qwen3.8
row this campaign has published, and calling it a "root" would be the kind of
undeclared difference the dataset format exists to stop.

Three consequences, stated so nobody re-derives them:

1. **A root dataset for this family cannot be produced by anyone**, including
   DeepSeek, without weights DeepSeek has not released.
2. Any dataset we did capture from this root would need a new, loud
   qualification — something like `root_is_itself_quantized` — and its own
   comparability group. That is a spec change, not a capture.
3. A **dequantize-to-BF16 upcast is not a substitute.** It is a deterministic
   function of the FP4 bytes; a KLD against it measures the upcast's rounding,
   not the publisher's lost precision, and it would read as near-zero for
   reasons that have nothing to do with fidelity.

**This is the finding that decides whether to spend anything, and the answer is
no.** Not because the engine cannot read it — §1.2–1.5 show it now can — but
because there is nothing to anchor to.

To be fair to the other reading: "how much worse is this GGUF than what DeepSeek
actually shipped" is a real question that real users have, and the published
FP4 root is exactly the thing they quantize from. If the campaign decides that
`distance_from_published_root` is a quantity worth publishing, this family is
the cheapest place to start doing it (§1.6: under $1.50 for the root and all
100 children). But that is a **specification** decision — a new qualification,
a new comparability group, a new column in the registry — and it has to be
taken before a capture, not defended after one. Nothing in this document should
be read as taking it.

### 1.2 The engine could not load it at all, and the reason was upstream

`transformers` 5.16.1, `quantizers/quantizer_finegrained_fp8.py:195`:

```python
layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
...
updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
```

`_impl_tp_layer_overrides` has exactly one key, `deepgemm_megamoe`, and
`config._experts_implementation` is still `None` when `get_hf_quantizer` runs.
So `layer_overrides` is `None` and the comprehension raises — **for any FP8
config whose parallel plan is non-empty.** `DeepseekV4Config.base_model_ep_plan`
has 7 entries.

```
AttributeError: 'NoneType' object has no attribute 'get'
```

That is what an operator saw, with no repository, no cause and no remedy, on a
model with 4.58 M downloads and 100 quant children — and it happens **before a
single weight is read, on any device, GPU or CPU.**

Two ways around it. Only one is safe, and the difference was measured rather
than reasoned about:

| walk-around | loads? | runs? |
|---|---|---|
| `from_pretrained(..., experts_implementation="deepgemm_megamoe")` | **yes** | **NO** — first forward raises `KeyError: `deepgemm_megamoe` is not a valid experts implementation registered in the `ExpertsInterface`` |
| empty `base_model_tp_plan` / `base_model_ep_plan` on the config | yes | yes — real logits, 27.4 B parameters |

The first is a trap: it buys a model that loads and cannot compute. The second
is inert, because those plans are a map from module path to sharding kind and
are read **only** when the model is being split across ranks — which
`hf_capture.py` never does.

So `hf_capture.py` now (a) recognises this exact failure by its own stack frame,
not by its message, and refuses with the cause and the remedy named, and (b)
offers `--drop-parallel-plan`, which empties those plans and is recorded in the
sealed receipt as `capture_tool.parallel_plan_dropped`. A25–A27b hold both
halves; A25 is deliberately frame-based so that an unrelated
`'NoneType' object has no attribute 'get'` is not swallowed.

### 1.3 The byte check had to be rebuilt, because `memcmp` is the wrong instrument here

`engines/tools/verify_fused_experts.py` closed GLM-5.3's R1 by reading each expert
matrix's raw bytes at its published offset and `memcmp`ing them against the live
fused parameter. That is exactly right when the loader is a byte mover.

Here it is not. With no FP8-capable GPU, `FineGrainedFP8HfQuantizer` sets
`dequantize = True` and the live parameter is the output of an arithmetic
pipeline: unpack two `e2m1` nibbles per `int8` byte, multiply by a UE8M0 block
scale grid, round to bf16 — and then the expert-fusion converter
(`MergeModulelist(dim=0)`, `Concatenate(dim=1)`) collapses 256 per-expert
matrices into `[E, 2I, H]`. A `memcmp` cannot see whether any of that is right,
and a statistic cannot either: a swapped nibble order, a transposed scale grid,
an off-by-one block size and a gate/up swap all produce plausible numbers.

`engines/tools/verify_loaded_weights.py` re-implements the **value** pipeline from
the format, in numpy, with no `transformers` and no `safetensors` in the path:
256-entry E4M3 and E8M0 tables built arithmetically from the bit fields, the
16-entry E2M1 table low-nibble-first, the block size derived from the two
shapes rather than from the config (experts ship a `[1,32]` grid and dense
linears a `[128,128]` one **inside the same checkpoint**), and
round-half-to-even to bf16 written out by hand. The **name** map is a
restatement, not an independent derivation, and is not claimed as one — a wrong
rename cannot corrupt a value quietly, it can only fail loudly, and
`missing_keys == 0` / `unexpected_keys == 0` already checks that the map is
onto.

**The one place where "independent" would be too strong a word, corroborated.**
The E2M1 nibble convention — low nibble first, bit 3 the sign, magnitudes
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`, and index 8 decoding to negative zero — is a
convention, and copying it out of the library under test would make agreement
unfalsifiable. It is not copied: it is the same 16-entry table as
`engines/tools/nvfp4_surface.py::_e2m1_lut16`, which this repository already proved
**bitwise against `compressed-tensors` 0.18.0's own `unpack_fp4_from_uint8`**
on real ranged-fetched tensors (`engines/tools/nvfp4-evidence/`). Two independent
implementations of the same published convention, agreeing, is the strongest
form of this claim available without a third.

### 1.4 R1 for `deepseek_v4` — CLOSED, and it covers the dequant too

Loaded through `hf_capture.load_model` — the production path — with
`--drop-parallel-plan`, on the 4-layer truncation (layers 0–3: three hash-router
layers and one `noaux_tc` top-k layer, all 256 routed experts each).

```
observed true   conversion_errors_visible true
missing_keys 0   mismatched_keys 0   error_msgs 0   conversion_errors 0
unexpected_keys 0
```

| comparison | count | differed |
|---|---:|---:|
| `gate_up_proj[k][0:2048]` vs `experts.k.w1.weight` (FP4, dequantized here) | 1,024 | **0** |
| `gate_up_proj[k][2048:4096]` vs `experts.k.w3.weight` (FP4) | 1,024 | **0** |
| `down_proj[k]` vs `experts.k.w2.weight` (FP4) | 1,024 | **0** |
| FP8 E4M3 projections (attention `q_a/q_b/kv/o_a/o_b`, indexer `q_b`, shared expert `w1/w2/w3`) | 33 | **0** |
| BF16 / F32 / I64 tensors, name-mapped 1:1 | 71 | **0** |
| **total checkpoint tensors compared** | **3,176** | **0** |

3,176 of 3,176 exact — **every tensor in the pruned index**, 3,105 of them
decoded independently from quantized bytes. **54,801,581,172 bytes of live
parameter compared.** 108 of 108 parameters covered; nothing uncovered, nothing
unplanned, no parameter without a checkpoint key. The fetch receipt's per-tensor
sha256 were re-derived from the shard bytes at check time: 3,176 re-checked, 0
mismatched.

**This tree has no holes at all**, which is why `unexpected_keys` is 0 and why
the DeepSeek report is the one of the three written before the byte checker
grew its `shard_tensors_outside_pruned_index` field: its six shards' headers
name **6,281** tensors and the pruned index names the same 6,281 — the audit
line `{"tensors": 6281, "shards": 6, "index_present": true}` from the §2.6 run
is that check, independently. Every one of the 6,281 was planned (3,105 of them
as `.scale` tensors folded into the weight they scale), so the coverage claim
here is over the shard headers, not merely over the index.

**`unexpected_keys` is 0 here and would be 0 on the full root**, because
`DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected` is
`[r"(^|\.)mtp\..*"]` and the MTP subtree is the only thing the architecture does
not build. So `deepseek_v4` needs **no** `--allow-unexpected-tensors` and would
carry **no** blocking disclosure on that account — unlike GLM-5.3, whose MTP
layer is not ignored.

Two facts worth carrying forward:

* **The fusion geometry is confirmed by the comparison, not assumed from the
  converter.** `gate_up_proj` is `[256, 4096, 4096]` with `w1` (gate) in rows
  `0:2048` and `w3` (up) in `2048:4096`; `down_proj` is `[256, 4096, 2048]`.
* **Dequantizing multiplies the checkpoint by 3.34x.** 16.42 GB of fetched
  quantized bytes became **54.802 GB of resident bf16 parameters** — FP4 goes
  to bf16 at 4x, FP8 at 2x, and 2.12 GB of embed+head was already bf16. That
  is **13.17 GB per decoder layer**, so the whole 43-layer model dequantized is
  **~568 GB**. §1.6 prices both regimes; the packed one is the one to use.

### 1.5 The capture chain, on `deepseek_v4`

Two cold captures in two separate processes through
`bin/fidelity-dataset capture --engine hf-transformers`, on
`panel--dsv4.stagea.smoke.2w` (2 windows, 2,048 tokens, literary + scientific,
built by `engines/tools/build_token_panel.py` against DeepSeek's own tokenizer).

| | |
|---|---|
| windows / scored positions | 2 / 4,094 |
| per window (CPU, M4 Max, single-threaded expert loop) | run 1: 365.2 s, 361.3 s; run 2: 368.8 s, 354.9 s |
| head hook | fired exactly once per window |
| head input | `[1, 2048, 4096]`, **bfloat16**, bias `None` |
| hidden form / logit-form equivalent | 33.5 MB / 2.12 GB (**63.1x**) |
| peak resident weights / peak RSS | **54.802 GB** / 75.804 GB (run 2: 54.802 / 74.957) |
| `capture_content_digest`, both runs | `e340db963c473690482a361593f7143a5f67fe8d5ffb22341df6a8866e5c4625` — **identical** |
| `verify --verify-tensors` | VERIFIED, tensors recomputed |
| `validate --strict` | 0 errors, 0 warnings |
| `compare --self-compare` | **exactly 0.0** |
| `compare --self-compare --force-compute` | **exactly 0.0**, top-1 agreement 1.000000 |

The sealed manifests carry `no_known_deviations` and `reduced_run_count` and
nothing else — no missing-weight disclosure, no unexpected-tensor disclosure.
**The datasets were then deleted:** they are a four-layer truncation, they are
not a measurement, and nothing should be able to mistake them for one.

### 1.6 What a root capture would cost, if the anchor existed

Priced with `docs/CAPTURE-SCALING-PLAN.md`'s measured constants (fetch 627 MB/s;
layer-outer 0.4–1.6 min/window at GLM-5.3's 78L/6144 scale;
`C_replay` cuda `1.30e-13 s` per position x vocab x hidden, both sides;
JarvisLabs spot H100 $1.19/GPU-h).

| term | value |
|---|---|
| fetch, 166.9 GB at 627 MB/s | **4.4 min** |
| resident, layer-outer, **FP8/FP4 kept packed** (Hopper+) | embed+head+norm 2.12 GB + one layer 3.57 GB = **5.7 GB** — but see §2.6(c): `--schedule layer-outer` now REFUSES a quantized checkpoint, so this row is a projection for an engine change that has not been made |
| resident, layer-outer, **dequantized to bf16** (no FP8 GPU) | 2.12 GB + one layer ~13.1 GB = **~15.2 GB**, plus the dequant transient |
| whole-model resident, dequantized | **~568 GB** (13.17 GB/layer x 43 + 2.12) — do not |
| capture, 25 windows x 2 cold runs, 43L/4096 | **0.3–1.2 min/window → 15–60 min** |
| **capture cost, 1x H100 spot** | **$0.30–1.19** |
| comparison, per window | 2047 x 129,280 x 4096 x 1.30e-13 = **0.141 s** |
| comparison, 25 windows x 100 quant children | **5.9 min → $0.12** |

**The whole family would cost under $1.50 to measure.** It is the cheapest
large target on the board and the engine is ready for it. The blocker is §1.1
and only §1.1.

---

## 2. `MiniMaxAI/MiniMax-M3`

**Verdict: GO.** Loads clean through the production path, the expert converter
is byte-exact on real weights, the capture chain closes, the self-compare is
exactly 0.0, and the layer-outer schedule reproduces the window-outer capture
**bit for bit**. Two engine defects were found on the way to that last clause
and both are fixed (§2.6); the one remaining caveat is a property of the
truncation, not of the architecture.

### 2.1 What it is, and what surprised

854.2 GiB / 869.2 GB, 23,416 tensors, all BF16, no `quantization_config`. 60
decoder layers: three dense (`moe_layer_freq[0:3] == 0`) then 57 sparse with 128
routed experts and one shared expert; a CLIP vision tower of 32 layers; two
projector MLPs; vocab 200,064 at hidden 6,144.

Two things that would have cost a day each if not checked:

* **The `auto_map` is a red herring.** The repo is tagged `custom_code` and
  ships `configuration_minimax_m3_vl.py`, but `minimax_m3_vl` is a **native**
  `transformers` 5.16.1 architecture, so `AutoConfig` resolves it natively and
  **no `trust_remote_code` is needed.** `MiniMaxM3SparseForConditionalGeneration`
  is a top-level export, which is what `hf_capture.load_model` looks for.
* **`text_config.num_mtp_modules` is 7 and the checkpoint ships no MTP
  tensors at all** — and the model does not build them either. `missing_keys`
  is 0. (`_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` covers the
  other direction.) So a full-root load should report **0 missing and 0
  unexpected**, and would need no override and carry no blocking disclosure.

The layer count lives in `config.text_config.num_hidden_layers`, its per-layer
schedules are lists of **ints** — one of them (`sparse_attention_freq`) nested
inside `sparse_attention_config` — and the vision tower has its own
`...encoder.layers.N.` keys that must survive a text-tower truncation
untouched. All three are why `--config-node` and the recursive schedule
truncator exist.

### 2.2 The fetch

| | |
|---|---:|
| truncation | 4 layers (0–2 dense, 3 sparse with all 128 experts) + the whole vision tower and both projectors |
| shards touched | 5 of 59 |
| **fetched** | **23.495 GB** |
| whole-shard fetching would have moved | 63.99 GB — **40.50 GB avoided** |
| wall clock | **220.4 s** at ~105 MB/s |
| tensors kept | 960 of 23,416 |
| config lists trimmed | `moe_layer_freq`, `sparse_attention_config.sparse_attention_freq`, `sparse_attention_config.sparse_disable_index_value` → `[0,0,0,1]` |

Ranged fetching earns its keep here: 63.3% of the bytes in those five shards
belong to layers we never load.

### 2.3 R1 for `minimax_m3_vl` — CLOSED

`minimax_m3_vl` routes through the same `MergeModulelist(dim=0)` +
`Concatenate(dim=1)` expert fusion as GLM-5.3 and DeepSeek-V4, **plus** a second
converter that folds `gate_proj` + `up_proj` into one `gate_up_proj.weight`
(`Concatenate(dim=0)`) for the dense layers and every shared expert. Both were
checked against the shard bytes.

| comparison | count | differed |
|---|---:|---:|
| `mlp.experts.gate_up_proj[k]` halves vs `experts.k.w1/w3.weight` | 256 | **0** |
| `mlp.experts.down_proj[k]` vs `experts.k.w2.weight` | 128 | **0** |
| `gate_up_proj.weight` halves vs `gate_proj`/`up_proj` (3 dense layers + 1 shared expert) | 8 | **0** |
| vision tower (515), text-tower attention/indexer/norms/router/embed/head (45), projectors (8) | 568 | **0** |
| **total** | **960** | **0** |

960 of 960 exact, **23,480,811,264 bytes** compared, **573 of 573 parameters
covered**, 0 unplanned, 0 uncovered.

### 2.4 The one caveat, and it is the truncation's

`unexpected_keys` is **42**, and they are tensors of layers 25, 26 and 59 —
layers this 4-layer model does not build — that happen to share the five fetched
shards. This is the same mechanism GLM-5.3 Stage A found and named:
**`transformers` enumerates each shard's own safetensors header, not the pruned
index**, so a sparse local tree offers it names whose byte ranges were never
fetched and are holes reading as zeros. Nothing was harmed (they were discarded
as unexpected), but it is why the byte check is scoped to the pruned index and
reports `shard_tensors_outside_pruned_index: 1091` explicitly rather than
comparing against a hole.

Consequence: the **Stage A capture** needs `--allow-unexpected-tensors` and
carries `unexpected_tensors_overridden` at `blocking`. **A full-root capture
would not**, because on a complete tree every shard's tensors belong to a layer
the model builds — and that is not an argument here, it is measured: §2.6's
hole-free repack of the same bytes loads with `unexpected_keys: 0`, needs no
override, and produces the **same `capture_content_digest`**.

### 2.5 The capture chain, on `minimax_m3_vl`

Two cold captures in two separate processes, on `panel--minimaxm3.stagea.smoke.2w`.

| | |
|---|---|
| windows / scored positions | 2 / 4,094 |
| per window (CPU, M4 Max), run 1 / run 2 | 40.56 s, 37.67 s / 36.96 s, 36.26 s (108.1 s and 104.5 s end to end) |
| head hook | fired exactly once per window |
| head input | `[1, 2048, 6144]`, **bfloat16**, bias `None` |
| hidden form / logit-form equivalent | 50.3 MB / 3.28 GB (**65.1x**) |
| peak resident weights / peak RSS | **23.481 GB** / 43.671 GB |
| `capture_content_digest`, both runs | `71a50e53114247f126e0b73cee9b4836b89af23c34484cc6ac2d35c6ddbedcc2` — **identical** |
| `verify --verify-tensors` | VERIFIED, tensors recomputed |
| `validate --strict` | 0 errors, 0 warnings |
| `compare --self-compare` | **exactly 0.0** |
| `compare --self-compare --force-compute` | **exactly 0.0**, top-1 agreement 1.000000 |

Disclosures: `no_known_deviations`, `checkpoint_tensors_not_loaded` (caveat, the
42 of §2.4), `unexpected_tensors_overridden` (**blocking**) and
`reduced_run_count`. **The datasets were then deleted.**

The 43.671 GB peak RSS against 23.481 GB of resident parameters is a **20.2 GB
excess** — the expert-fusion transient plus the safetensors page cache — which
is 1.36x one sparse layer's 14.83 GB of routed experts. That is the same shape
as GLM-5.3 Stage A's finding (1.15x one layer) and is the number §2.6's
"~35 GB peak" is built on.

### 2.6 The layer-outer schedule, which every cost table below assumes

Every price in this document is a layer-outer price, and `layer_outer.py` had
only ever been proven on `glm5_next` and `glm_moe_dsa`. Running it here hit one
designed-in constraint and **two defects, one of them silent** — and then, past
all three, proved the schedule bit-identical on a third architecture.

**(a) The constraint: it refuses a sparse truncation, correctly, which ends the
experiment as originally planned.**
`audit_checkpoint_tree` compares the shard headers against the index and
refuses when they disagree — which is precisely what a ranged-fetch truncation
is (1,091 header-only tensors, §2.4). So **the layer-outer schedule cannot be
tested on a sparse truncation at all**, by design. The bytes were therefore
re-serialised into a hole-free tree — same payload bytes, new offsets, index and
headers in agreement — and the schedule test run on that. The byte check of
§2.3 stays on the SPARSE tree, where the offsets are the published ones.

**(b) It could not find the decoder stack, and said something true about the
wrong name.** `layer_pattern` is built from the MODEL's stack path
(`model.language_model.layers`) and was matched against RAW checkpoint keys,
which MiniMax spells `language_model.model.layers`. Every layer tensor missed,
all of them fell into the resident load, and the schedule refused with *"the
checkpoint holds no tensors for model.language_model.layers.0"*. Fixed by
routing on the key **after** the architecture's own conversion renames — the
same mapping `convert_and_load` applies to those same raw keys a few lines
later. Architectures whose
names already match are unaffected, by construction.

**(c) It accepted a quantized checkpoint, and that one was silent.**
`build_streamed_model` instantiates the architecture directly and loads with
only the MODEL's conversion mapping; it never builds an `HfQuantizer`. So the
quantizer's module replacement, its `*.scale` -> `*.weight_scale_inv` rename and
its dequantization op are all absent. On DeepSeek-V4-Flash-0731 that surfaced
loudly — `Reinit due to size mismatch - ckpt: torch.Size([256, 4096, 2048]) vs
model: torch.Size([256, 4096, 4096])`, raised by `transformers` rather than by
us. **On a plain FP8 E4M3 checkpoint it would not surface at all**: the fp8
weight's shape is IDENTICAL to the bf16 parameter it is loaded into, so the
payload is cast to bf16, the `.scale` tensor falls out as `unexpected`, and the
block scale is never applied. That is numerically the M1 Qwen3.8-27B-FP8 defect,
and its only signal would be `unexpected_keys` — behind the very flag a
truncated tree already needs. `build_streamed_model` now refuses on the config's
own `quantization_config`, before any weight is read. L15 in
`bin/selftest_layer_outer.py`; **on the pre-change tree that rung reads
`rc=0`** — the schedule captures a quantized checkpoint and says nothing.

**And then it worked, bit for bit.**

| capture | schedule | tree | `capture_content_digest` |
|---|---|---|---|
| `mm-run1` | window-outer | sparse | `71a50e53114247f126e0b73cee9b4836b89af23c34484cc6ac2d35c6ddbedcc2` |
| `mm-run2` | window-outer | sparse | same |
| `mm-dense-wo` | window-outer | dense repack | same |
| `mm-dense-lo` | **layer-outer** | dense repack | **same** |

`compare --self-compare --force-compute` between the two dense captures — the
real matmul, not the digest short-circuit — is **exactly 0.0** with top-1
agreement 1.000000. Measured peak resident weights: **21.480 GB** layer-outer
against **23.481 GB** window-outer (a small gap only because this truncation is
three dense layers and one sparse one; on the 60-layer model the resident base
is 7.31 GB and the saving is the other 840 GB).

A bonus fact from the dense run: the window-outer capture of the hole-free tree
reports **`unexpected_keys: 0`** and needs no override, and its digest is
identical to the sparse runs' — so §2.4's 42 unexpected tensors demonstrably
changed nothing about the numbers.

### 2.7 What a root capture would cost

| term | value |
|---|---|
| fetch, 869.2 GB at 627 MB/s | **23.1 min** |
| resident, layer-outer | embed+head+norm **4.92 GB** + one sparse layer **14.83 GB** + expert-fusion transient (~one layer) = **~35 GB peak** |
| device | fits 1x H100 80 GB with 45 GB spare; does **not** fit an L4 |
| capture, 25 windows x 2 cold runs, 60L/6144 | 0.4–1.6 min/window → **20–80 min** |
| **capture cost, 1x H100 spot**, fetch overlapped | **$0.46–1.59** |
| comparison, per window | 2047 x 200,064 x 6,144 x 1.30e-13 = **0.327 s** |
| comparison, 25 windows x 58 quant children | **7.9 min → $0.16** |

**Under $2 for the root plus every child on the Hub today.** This is the
strongest GO of the three: a BF16 root, a clean load, a proven converter, 58
children waiting, and no anchor problem.

---

## 3. `Qwen/Qwen3.8-Flash-Next`

**Verdict: GO-WITH-WORK.** The load, the byte check and the whole capture chain
work today, and the self-compare is exactly 0.0. The work is not in our engine:
it is that this model puts **102.4 GB in a single parameter**, which changes
where a root capture can run and is untested against the layer-outer schedule.

### 3.1 What it is

360.0 GB, 180 B params, all BF16, 1,658 tensors — a very small tensor count for
the size, because **the routed experts are already fused on disk**
(`mlp.experts.gate_up_proj` is one `[512, 1280, 2560]` tensor per layer). That
is why `qwen4_exp` has **no entry in `transformers`' `conversion_mapping.py` at
all**: checkpoint names are already `transformers`' own. There is no expert
converter to be wrong — which is itself the finding.

48 decoder layers on a `[linear_attention x3, full_attention]` cycle, 512
routed experts with top-10, a 27-block vision tower, an MTP block the
architecture ignores (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), and
a PLE (per-layer n-gram embedding) on layer 1.

### 3.2 The 102.4 GB parameter

`model.language_model.layers.1.ple.ple_embedding.ngram_embedding` is shipped as
128 checkpoint shards of `[2,500,012, 160]` bf16 and built by the model as **one
`nn.Embedding` of `[320,001,536, 160]` = 102.4 GB.** That is 28.4% of the whole
model in a single parameter, in a single layer.

`transformers` knows: `Qwen4ExpPreTrainedModel._no_placement_params` lists it,
so `infer_auto_device_map` skips it and the forward moves ids to wherever it
lives (`ngram_ids.to(self.ngram_embedding.weight.device)`). The intended
deployment is **PLE in host RAM, everything else on the accelerator.**

Consequences, in order of how much they cost:

1. A root capture host needs **≥ 128 GB of system RAM** (or ≥ 141 GB VRAM to
   hold it on-device). This is a machine-shape requirement, not a GPU-hours
   one, and it must be checked before renting.
2. **`engines/tools/layer_outer.py` has never seen a `_no_placement_params`
   parameter.** Its residency model streams whole layers on and off the device;
   a 102.4 GB parameter that must stay put is exactly the case it does not
   describe. That is the "work" in GO-WITH-WORK, and it is engine work, not
   discovery.
3. **This is why the Stage A truncation is one layer and not four.** A 4-layer
   truncation includes layer 1, so it costs **126.7 GB to fetch** and ~126 GB
   resident on a 137 GB machine — 5x the GLM-5.3 Stage A budget for one extra
   attention type. A 1-layer truncation is 8.64 GB and answers the questions
   this stage exists to answer — does the architecture instantiate, does the
   loader fill it, does the head hook fire, does the chain seal and self-compare
   to 0.0. It does **not** exercise `full_attention` (layer 3), the DSA-style
   `indexer`, or the PLE itself. Those three are the named gap, and their price
   is the 126.7 GB above.

### 3.3 The config refuses a naive truncation, and says so precisely

`text_config.ple_layer_ids` is `[2]` — a **one-indexed list of layer indices**,
length 1. It is not a per-layer schedule, so no length-based rule can find it,
and `Qwen4ExpTextConfig` validates it:

```
ValueError: ple_layer_ids must contain one-indexed ids in [1, 1], got [2].
```

Hence `--config-index-list ple_layer_ids:1`, which filters out-of-range layer
ids and records what it dropped. F10 is that case.

### 3.4 A second refusal, from the cache, and the fix was already in place

A `qwen4_exp` truncation made only of `linear_attention` layers cannot run a
forward pass with a cache:

```
ValueError: `get_seq_length` can only be called on Attention layers, and the
current Cache seem to only contain LinearAttention layers.
```

`hf_capture.py` already passes `use_cache=False` on its one forward call — for
unrelated reasons, stated in its own comment — and with it the 1-layer
truncation runs and produces real logits. Worth recording because it is a
**capture-only** property: a tool that generated, or that let `use_cache`
default from the config, would hit this and would look like an architecture
failure.

### 3.5 R1 for `qwen4_exp` — CLOSED, and there was no converter to close

```
observed true   conversion_errors_visible true
missing_keys 0   mismatched_keys 0   error_msgs 0   conversion_errors 0
unexpected_keys 23     <- layers 1 and 9, sharing the fetched shards; §2.4's mechanism
```

| comparison | count | differed |
|---|---:|---:|
| every checkpoint tensor in the pruned index, name-mapped 1:1 | **362** | **0** |

362 of 362 exact, **8,641,744,800 bytes** compared, **362 of 362 parameters
covered**, 0 unplanned, 0 uncovered. The fused expert tensors
(`mlp.experts.gate_up_proj` `[512,1280,2560]`, `down_proj` `[512,2560,640]`)
are byte-identical to the shard bytes because they are copied, not built.

### 3.6 The capture chain, on `qwen4_exp`

| | |
|---|---|
| windows / scored positions | 2 / 4,094 |
| per window (CPU, M4 Max), run 1 / run 2 | 8.95 s, 9.04 s / 10.59 s, 9.31 s (25.4 s and 27.3 s end to end) |
| head hook | fired exactly once per window |
| head | `Linear(2560 -> 248320)`, bias `None`; head input `[1, 2048, 2560]` bfloat16 |
| hidden form / logit-form equivalent | 21.0 MB / 4.07 GB (**194.0x**) |
| peak resident weights / peak RSS | **8.642 GB** / 11.622 GB |
| `capture_content_digest`, both runs | `95f96a8cbcf369b84b3114fc05183cf2818143773fc9c13b85f75207b0cebc74` — **identical** |
| `verify --verify-tensors` | VERIFIED, tensors recomputed |
| `validate --strict` | **0 errors, 0 warnings** |
| `compare --self-compare` | **exactly 0.0** |
| `compare --self-compare --force-compute` | **exactly 0.0**, top-1 agreement 1.000000 |

The hidden/logit storage ratio of **194x** is the largest this campaign has
seen — vocab 248,320 against hidden 2,560 — and is worth remembering when
sizing a real Qwen3.8-Flash-Next panel.

Disclosures on the sealed manifests: `no_known_deviations`,
`checkpoint_tensors_not_loaded` (caveat, 23 tensors),
`unexpected_tensors_overridden` (**blocking**, the §2.4 truncation artifact) and
`reduced_run_count`. **The datasets were then deleted.**

### 3.7 What a root capture would cost

| term | value |
|---|---|
| fetch, 360.0 GB at 627 MB/s | **9.6 min** |
| **host RAM floor** | **≥ 128 GB** for the PLE embedding — check the box shape before renting |
| resident on device, layer-outer | embed+head+norm 2.54 GB + one layer 5.19 GB = **~8 GB** (PLE excluded, on the host) |
| resident if the PLE goes on-device | **+102.4 GB** → H200 only |
| capture, 25 windows x 2 cold runs, 48L/2560 | 0.2–0.8 min/window → **10–40 min** |
| **capture cost, 1x H100 spot**, fetch overlapped | **$0.20–0.79** |
| comparison, per window | 2047 x 248,320 x 2,560 x 1.30e-13 = **0.169 s** |
| comparison, 25 windows x 100 quant children | **7.0 min → $0.14** |

The dollars are trivial. The gate is `layer_outer.py` and a host with enough
RAM — engineering and machine shape, not budget.

---

## 4. What this cost, in total

| | |
|---|---:|
| dollars | **$0.00** |
| GPUs rented | **0** |
| JarvisLabs instances touched | **0** |
| bytes fetched, all three | **48.58 GB** (16.42 + 23.50 + 8.66) against 1,396 GB of published roots |
| fetch wall clock | 153.4 s + 220.4 s + 81.7 s = **7 min 36 s** |
| disk, before | 480 GiB used, **1.3 TiB free** |
| disk, high-water | 561 GiB used, **1.2 TiB free** — 81 GB of scratch: 45 GB of sparse truncations, a 23.5 GB hole-free repack for §2.6, and ~13 GB of datasets |
| disk, after cleanup | 480 GiB used, **1.3 TiB free** — back to the byte |
| datasets published | **0** |
| registry rows written | **0** |

### 4.1 Where the evidence is

`engines/tools/newarch-stagea-evidence/` — 32 files, the same shape as
`engines/tools/glm53-stagea-evidence/`:

| file group | what it is |
|---|---|
| `*-fetch-receipt.json` | the pinned revision, the selection flags, and a sha256 for **every** tensor fetched (6,281 / 960 / 362) — the preimage of the byte checks |
| `*-weight-decode-check.json` | the byte-check summaries: counts, coverage both ways, bytes compared, and the load report as read |
| `*-panel.json`, `*-panel.receipt.json` | the three smoke panels and their build receipts (deterministic, no RNG) |
| `*-run*-manifest.json`, `*-memory.json` | the sealed dataset manifests and the measured peak-memory reports |
| `*-sc1-comparison-receipt.json` | the `--self-compare --force-compute` receipts, all exactly 0.0 |
| `minimax-schedule-equivalence-receipt.json` | window-outer vs layer-outer, real matmul, exactly 0.0 |

**The datasets themselves are not here and were deleted.** They are four-layer
(one-layer, for Qwen) truncations; they are not measurements; nothing should be
able to mistake them for one. The manifests are kept because a manifest is a
statement about a run, and the digests in them are what make the run
reproducible.

## 5. What would falsify any of this

* **DeepSeek.** If DeepSeek ever publishes a full-precision `DeepSeek-V4-Flash`
  — or if someone demonstrates that the FP4 release is itself a lossless
  encoding of training weights that exist — §1.1 flips and this becomes the
  cheapest GO on the board. Re-check the Hub before spending anything else on
  the family.
* **All three.** The byte checks are against **truncations**. They prove the
  converter and the decode on layers 0–3 (0 for Qwen); the remaining layers go
  through the identical layer-agnostic code path, which is an argument, not a
  measurement. The first `layer_load` line of a real Stage B run is what turns
  it into one.
* **The expert-fusion / dequant transient is the least-measured term in every
  table above.** It is bounded from CPU RSS readings, not measured on a device:
  MiniMax's peak RSS ran 20.2 GB above its 23.5 GB of resident parameters
  (1.36x one sparse layer), and DeepSeek's dequant expands each layer 3.34x on
  the way in. Both are upper bounds contaminated by the safetensors page cache.
* **Qwen specifically.** The `_no_placement_params` interaction with
  `layer_outer.py` is untested. If it turns out the schedule cannot express
  "one parameter stays on the host", the Qwen capture needs the window-outer
  path and a much larger machine, and §3.7's numbers do not hold.
* **The layer-outer schedule is now proven on `minimax_m3_vl` (§2.6) and on
  nothing else here.** `qwen4_exp` was not tested: a 1-layer truncation makes
  the schedule vacuous, and the interesting case is the 102.4 GB PLE parameter,
  which needs the 4-layer tree §3.2 rules out locally. `deepseek_v4` cannot be
  tested at all until the schedule grows a quantizer — it now refuses, which is
  the right answer and not a passing one. So MiniMax's §2.7 price is
  layer-outer-backed; DeepSeek's §1.6 and Qwen's §3.7 are not.
* **`DeepseekV4PreTrainedModel._is_stateful` is True** — the CSA/HCA compressor
  keeps a running-window state that is not rewindable — and `minimax_m3_vl`'s
  sparse-attention indexer has a per-layer block-selection step. MiniMax's
  bit-identical digest says the indexer survives the schedule. DeepSeek's
  compressor is untested against it, and would be the first thing to check if
  the quantized-checkpoint support is ever built.
* **The per-window capture anchors (0.2-1.6 min) are GLM-5.3's projection
  rescaled by layer count and hidden width, not a measurement on any of these
  three.** The measurements this document does contain are CPU numbers on
  truncations, and the DeepSeek one (365 s/window for 4 layers) is dominated by
  a single-threaded 256-expert Python loop that a GPU does not run. Do not
  extrapolate from it in either direction.
