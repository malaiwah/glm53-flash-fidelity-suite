# Can we capture a root fidelity dataset for GLM-5.3 (the full model)?

**Verdict: GO-WITH-STAGING, and the gate that matters is not a GPU gate.**

> **UPDATE 2026-08-30 — Stage A has been run, for $0.00. See section 9.**
> R1 is CLOSED: 836 of 836 checkpoint tensors, including all 768 fused
> per-expert matrices of a real sparse layer, are byte-for-byte identical
> between the loaded model and the shard bytes. R3's guard was widened to see
> `mismatched_keys`, `error_msgs`, `conversion_errors` and the no-report case.
> R2's mechanism is fixed (`device_map`, proven on `glm_moe_dsa`); R2's
> economics are unchanged. Two corrections to this document's own Stage A
> recipe and to R1's stated tensor shape are in section 9.2.

`zai-org/GLM-5.3-BF16` is our architecture — `glm_moe_dsa`, the same family as
Fruit, which `k6/tools/hf_capture.py` captured end to end. Every config key
Fruit needed is present. The memory arithmetic fits one H200 with room to
spare. The checkpoint fits a JarvisLabs disk. None of that is the problem.

The problem is that **no engine we have can read this checkpoint.** The
portable engine (`hf_capture.py`) materialises the whole model and then calls
`.to(device)`; at 1,486.8 GB that fails on every machine JarvisLabs rents. The
engine that *can* stream a 600 GB BF16 tree on one H200 (`stream_score.py`, via
`k6/tools/hidden_replay.py`) is hard-coded to GLM-5.3-Flash's `glm5_next`
geometry — `HIDDEN_WIDTH = 4096` — and is not in this repository at all.

So the honest shape of the answer is: **the science is ready, the yardstick is
ready, the money is nearly enough, and the capture engine does not exist yet.**
Below is the arithmetic, the three stages, and the one thing most likely to
kill it.

Everything here is metadata fetches, config reads, code reading and local
dry-runs. No GPU was rented and no shard was downloaded.

---

## 1. What the model actually is

Fetched at revision `304b8051cfb2b260b61ce0cbe330e02a98e73639` (public,
ungated, `license: other`, 291 files, no remote code).

| | `zai-org/GLM-5.3-BF16` | `zai-org/GLM-5.3` |
|---|---|---|
| precision | BF16, **no `quantization_config`** | FP8 `e4m3`, block `[128,128]`, dynamic |
| shards | 282 | 141 |
| size | 1,506.7 GB | 755.7 GB |

**The two configs are byte-for-byte identical except that the FP8 one adds
`quantization_config`.** Same 78 layers, same `hidden_size` 6144, same vocab
154,880, same expert layout. That is as clean a root/candidate pair as this
campaign has ever had: the suffix-less repo is a quantisation *of* the `-BF16`
repo, by the publisher's own metadata, exactly as with Flash.

### Config keys, GLM-5.3 vs Fruit

`GLM-5.3-BF16` and `malaiwah/GLM-5.2-SIQ-Fruit-bf16` have **identical key
sets** except that Fruit carries a legacy top-level `rope_theta` alongside
`rope_parameters`. Nothing GLM-5.3 declares is new to us. The differences are
all scale:

| key | GLM-5.3 | Fruit |
|---|---|---|
| `num_hidden_layers` | 78 | 13 |
| `hidden_size` | 6144 | 1024 |
| `intermediate_size` | 12288 | 2048 |
| `moe_intermediate_size` | 2048 | 512 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 64 | 16 / 16 |
| `q_lora_rank` | 2048 | 1024 |
| `max_position_embeddings` | 1,048,576 | 65,536 |
| `rope_parameters.rope_theta` | 8,000,000 | 500,000 |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 8 | 256 / 8 |
| `first_k_dense_replace` | 3 | 3 |
| `num_nextn_predict_layers` | 1 | 1 |
| `tie_word_embeddings` | **false** | false |
| `kv_lora_rank` / `qk_head_dim` / `v_head_dim` | 512 / 256 / 256 | same |
| `index_head_dim` / `index_n_heads` / `index_topk` / `index_topk_freq` | 128 / 32 / 2048 / 4 | same |

**The one structural difference that matters: the DSA indexer schedule.**
Fruit's `indexer_types` is `full` only for layers 0–2 and `shared` for
everything after. GLM-5.3's is `full` every fourth layer — `[0, 1, 2, 6, 10,
14, …, 74]`, 21 layers — with the rest `shared`.

I verified against the safetensors index that indexer tensors exist on exactly
those 21 layers **plus layer 78**, the MTP layer (`index_share_for_mtp_iteration:
true`). No layer that declares `full` is missing its indexer, and no layer that
declares `shared` carries one.

> **Correction to a premise of this investigation.** The belief that
> `transformers` "silently dropped Fruit's DSA indexer for layers 3–13" is not
> what happened. Fruit's config genuinely declares those layers `shared`, and
> `GlmMoeDsaAttention.__init__` sets `self.indexer = None` when
> `config.indexer_types[layer_idx] == "shared"` — by design, not by accident.
> What `transformers` actually reported missing on Fruit, per `hf_capture.py`'s
> own docstring, was `model.layers.{3..12}.mlp.experts.{gate_up,down}_proj` —
> the *routed experts*, because the SIQ artifact ships them as exl3-trellis
> atoms. That is a different failure, and §6 shows it is the one that
> genuinely threatens this run.

`index_topk` is 2048 and the modeling code takes `min(index_topk, key_len)`, so
at a 2048-token window the indexer degenerates to full attention. No sparse
kernel is required: the eager/SDPA path turns the top-k into an additive mask,
and `flash-mla` is optional. `_supports_flash_attn` is `False`,
`_supports_sdpa` is `True`.

---

## 2. The arithmetic of scale

Computed from the safetensors index (59,585 tensors) plus tensor shapes
range-fetched from six shard headers. **The census reconciles to the published
`total_size` with a delta of zero bytes**, which is why the rest of this
document is arithmetic rather than estimate.

### Where the 1,506.7 GB is

| component | params | bytes | GiB | share |
|---|---:|---:|---:|---:|
| routed experts (76 sparse layers) | 734,439,407,616 | 1,468,878,815,232 | 1368.00 | 97.5% |
| attention (MLA, 79 layers) | 13,036,754,432 | 26,073,508,864 | 24.28 | 1.7% |
| shared expert + router gate | 2,988,461,056 | 5,976,961,024 | 5.57 | 0.4% |
| embed + lm_head + final norm | 1,903,171,584 | 3,806,343,168 | 3.54 | 0.25% |
| dense MLP (layers 0–2) | 679,477,248 | 1,358,954,496 | 1.27 | 0.09% |
| DSA indexer (22 layers) | 206,181,888 | 412,363,776 | 0.38 | 0.03% |
| MTP extras (`eh_proj`, norms) | 75,515,904 | 151,031,808 | 0.14 | 0.01% |
| layernorms | 970,752 | 1,941,504 | 0.00 | — |
| **total** | **753,329,940,480** | **1,506,659,919,872** | **1403.19** | |
| published `metadata.total_size` | | 1,506,659,919,872 | | **delta 0** |

**Per-layer routed-expert bytes: 256 × 3 × 6144 × 2048 × 2 = 19,327,352,832 B
= 18.00 GiB = 19.33 GB.** Verified equal to the observed sum for layer 6.
75 sparse layers in the body, plus the MTP layer 78 = 76.

**Non-routed total: 37,781,104,640 B = 35.19 GiB = 37.78 GB.** (Flash's is
19.34 GB — GLM-5.3's is 1.95×.)

Per-layer totals: dense layer 0.747 GiB; sparse layer 18.381 GiB; sparse layer
with a full indexer 18.398 GiB; MTP layer 78 18.539 GiB.

### What `transformers` actually builds

A meta-device instantiation of `GlmMoeDsaForCausalLM` from this exact config
(transformers 5.16.1, torch 2.13.0, run locally, no weights, no GPU) builds
**743,377,000,704 parameters = 1,486.8 GB**, not 753.3 B. The difference is
**layer 78: the model does not build the MTP layer at all**, and its 791
tensors land in `unexpected_keys`. Its 18.5 GiB never has to be resident. It
still has to be *fetched*, because only 3 of the 282 shards (16.1 GB) contain
layer-78 tensors exclusively — skipping them saves 1% and complicates the
checkpoint identity, so don't.

### Peak resident memory for a layer-outer streaming forward

Using the census figures directly (`nonrouted` + N layer buffers + panel state
+ the fp32 logit buffer at vocab 154,880 + 1.5 GB framework/attention floor):

| schedule | 8 win | 16 win | 25 win | H200 141 | RTX PRO 6000 96 | H100 80 | M4 Max ~110 |
|---|---:|---:|---:|:--:|:--:|:--:|:--:|
| non-routed resident + **2** layer buffers | 80.0 | 80.8 | **81.7** | fits | fits | **no** | fits |
| non-routed resident + **1** layer buffer | 60.7 | 61.5 | **62.4** | fits | fits | fits | fits |
| non-routed also streamed + 2 buffers | 45.1 | 45.9 | **46.8** | fits | fits | fits | fits |

(GB. Panel state = `windows × 2047 × 6144 × (2+2+4)` bytes; logit buffer =
`2047 × 154880 × 4` = 1.27 GB.)

**One H200 (141 GB) fits the roomiest schedule with 59 GB of headroom.** This
is consistent with the lane's own history: `k6/STREAMING.md` records a measured
34.40–47.08 GB peak for the Flash streaming lane on one H200, and GLM-5.3's
non-routed is 1.95× Flash's while its per-layer routed set is 1.33× Flash's
14.50 GB.

**No multi-GPU is needed and no CPU offload is needed — for a streaming
schedule.** For a whole-model-resident schedule the picture inverts completely;
see §5.

### `bin/fidelity/census.py`'s fit estimator — yes, it points at this model

`Census.from_config` is architecture-generic and reads every key GLM-5.3
declares (`num_hidden_layers`, `first_k_dense_replace`, `n_routed_experts`,
`hidden_size`, `moe_intermediate_size`, `vocab_size`, `num_nextn_predict_layers`;
`hc_mult` defaults to 1, which is correct here — that knob is Flash's). Run
against the real config + the real `total_size` it reports `census_source:
"hf-blobs"` and reproduces the exact figures above:

```
routed_main 1449.55 GB   routed_mtp 19.33 GB   nonrouted 37.78 GB   per_routed_layer 19.33 GB
solve_local(H200, bits=16, ctx=2047, windows=25) -> peak 85.5 GB, expert_chunk 256, passes 1
minimum_viable_budget(bits=16, ctx=2047)          -> 5.25 GB
```

**Caveat on that 85.5 GB: `local_peak_bytes` models the *packed* lane** — it
counts a decoded chunk *and* a packed chunk. At `bits=16` those are the same
bytes counted twice (38.66 GB each). The honest BF16-root figure is the 81.7 GB
in the table above. The estimator is usable for this model; its answer is
conservative by roughly one layer buffer, which is the safe direction.

---

## 3. Disk and fetch

**JarvisLabs caps an instance disk at 2,048 GB** and a shared filesystem at
50–2,048 GB (IN1/IN2 only). Live from `jl gpus --json` / `jl resources --json`.

| item | GB |
|---|---:|
| full checkpoint | 1,506.7 |
| toolchain / venv / HF cache slack (the planner's own constant) | ~40 |
| capture outputs, 25 windows, hidden form | 0.63 |
| **total** | **~1,548** |
| **headroom to the 2,048 GB cap** | **~500** |

It fits, with real margin. `snapshot_download` uses blobs + symlinks, so there
is no duplication — but an interrupted resume leaves `.incomplete` files, and
the campaign has already lost 36 idle minutes (~$2.40) to "Disk quota exceeded"
once (JOURNAL lesson 31). Budget 1,700 GB, not 1,600.

### Fetch time

Measured throughputs in this tree: **430 MB/s** (M1, 165 GB in 6m22s),
**510 MB/s** (M2, 175 GB in 5m42s), **~600 MB/s** (the L4 prep VM, 599 GiB BF16
in 17 min). Against 1,506.7 GB:

| rate | wall clock |
|---|---|
| 430 MB/s | 0.97 h |
| 510 MB/s | 0.82 h |
| 600 MB/s | 0.70 h |

Roughly **one hour**, and it should be done on the cheapest box that can hold
it (1× L4 spot, $0.29/h) writing to a shared filesystem, not on the GPU that
will do the capture. At $0.29/h + 1,700 GB of storage that fetch costs
**$0.40–0.56**.

### Sparse fetch: does not apply

`k6/tools/fetch_nonrouted_sparse.py` exists and works, but it is the wrong tool
here and would be actively misleading. It materialises **only the non-routed**
byte ranges, because the Flash streaming lane reads routed experts from a
*quantized artifact* and non-routed from BF16. A **root** capture is the BF16
model itself — it needs every routed expert.

And it needs all of them, not most: at 2,047 positions and top-8 of 256, the
probability an expert is never selected in a layer is
`(1 − 8/256)^2047 ≈ e^−64 ≈ 0`. **Every expert in every layer is touched by
every window.** There is no sparse subset. Fetch all 1,506.7 GB.

(A second consequence: `hf_capture.py`'s `checkpoint_identity` hashes every
`*.safetensors` in the directory. A partial tree yields a different identity,
which is another reason to fetch whole.)

---

## 4. Capture cost and time

### The measured anchor — and it is not the one we were reaching for

The per-window figures the campaign quotes (2.37–3.19 min/window) are all
**quantized** surfaces, where the cost is trellis *decode*. The right anchor
for a BF16 root is the **BF16 floor lane**, recorded in `bin/engines.json`:

> `minutes_per_window: 8.3` — "MEASURED on 1× H200 spot (IN2, 28 vCPU, 300 GiB
> cgroup) reading the 599 GB BF16 tree off a CephFS filesystem at ~1.05 GB/s
> with 28 reader threads: window 1 (cold, all 42 layers filled) 678 s;
> steady-state windows 483–549 s with `--decode-cache ram` holding 17 of 42
> layers."

Cross-checked: cold run 1 was 12,514.5 s for 25 windows (8.34 min/window) and
read **9.31 TB** — 372 GB per window, i.e. the 25 of 42 layers that did not fit
the RAM cache, at ~744 MB/s effective. The lane **re-reads the routed set once
per window.** That is the entire cost.

### Scaling that to GLM-5.3, honestly

The dense parts scale as `layers × hidden²` and are irrelevant: total compute
for a 25-window panel is **4.11 PFLOP** at ~40.2 B active parameters per token,
which is 20–51 s of H200 time. **Compute is not the bottleneck and never will
be.** The bottleneck is bytes moved, and those scale with `layers ×
n_routed_experts × moe_intermediate × hidden`:

- GLM-5.3 routed set per window: **75 × 19.327 = 1,450 GB** (2.38× Flash's 609)
- with the same 300 GiB RAM cache it holds **16 of 75** layers (vs 17 of 42),
  so **1,140 GB is re-read per window** — **3.07×** Flash's 372 GB

| effective read rate | min/window | 25-window run |
|---|---:|---:|
| 0.74 GB/s (Flash's measured effective) | 25.5 | 10.64 h |
| 1.05 GB/s (CephFS low) | 18.1 | 7.54 h |
| 1.44 GB/s (CephFS high) | 13.2 | 5.50 h |
| 3.0 GB/s (fast local NVMe) | 6.3 | 2.64 h |

**Range, not a point: 13–26 min/window, 5.5–10.6 h per cold run, on the
per-window-re-read design.** Two cold runs is 11–21 h.

### The 20× that is sitting on the table

A **layer-outer, window-inner** schedule — load layer *i*, run every window
through it, free it, go to layer *i+1* — reads the checkpoint **once per run**
instead of once per window:

| effective read rate | per RUN, any window count |
|---|---:|
| 1.05 GB/s | 23.6 min |
| 1.44 GB/s | 17.2 min |
| 3.0 GB/s | 8.3 min |

Plus 1–2 minutes of compute. **This is bitwise identical to the per-window
forward** — windows do not interact, and running each window separately through
a resident layer performs exactly the same sequence of operations on exactly
the same GEMM shapes as a whole-model forward does; only the loop order across
independent windows changes. (This is the same invariance `solve_local`'s
docstring already asserts for `window_batch`, and it holds here for the
stronger reason that no reduction is shared across windows at all.)

`bin/engines.json:408` already knows this and withdrew an old projection for
it: *"the earlier 0.25 min/window figure was a projection of a layer-outer
schedule NO engine implements and is withdrawn."* That sentence is the whole
feasibility question in one line.

### Capture size: hidden form vs logit form

| windows | scored positions | hidden (bf16, ×6144) | logit (fp32, ×154,880) |
|---:|---:|---:|---:|
| 2 | 4,094 | 50 MB | 2.5 GB |
| 8 | 16,376 | 201 MB | 10.1 GB |
| 16 | 32,752 | 402 MB | 20.3 GB |
| **25** | **51,175** | **629 MB** | **31.7 GB** |

**Hidden form is 50× smaller, at every size.** `hf_capture.py` captures hidden
form only, which is the right default and makes the output storage a rounding
error against 1.5 TB of weights. Logit form would add 31.7 GB per cold run per
model — the exact figure that busted the filesystem in JOURNAL lesson 31.

---

## 5. The panel

**Do not reuse Brandon's sealed 25-window panel.** The token ids are more than
numerically valid: I fetched `zai-org/GLM-5.3-BF16`'s `tokenizer.json` and it is
**byte-identical** to the one this repo already holds for GLM-5.3-Flash-BF16 @
`b1967181` and to Fruit's — all three sha256
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`, 20,217,442
bytes, vocab 154,880. Tokenization is not the objection.

The objection is selection. That panel was built against GLM-5.3-**Flash**'s
corpus and its calibration separation. A number measured on it invites being
ranked against Flash numbers it has no business being ranked against, which is
the cross-model comparison our own rules forbid — and an identical tokenizer
makes that mistake *easier*, not harder, because nothing downstream would
complain. `build_token_panel.py`'s docstring says this better than I can, and it
was written for exactly this situation.

### Recommendation: 25 windows, 5 strata × 5, its own `panel_id`

Build with `k6/tools/build_token_panel.py` — deterministic, no RNG, sorted
traversal and a fixed slice — over the same public corpus Fruit's panel used:

```
corpus     malaiwah/qwen38-27b-fidelity-suite-v5 @ 7797fcce, corpus/text/
strata     code (241 docs), encyclopedic (323), literary (59),
           multilingual (276), scientific (42)      -- verified present
rule       --context-length 2048 --skip-tokens 2048 --windows-per-stratum 5
           => 25 windows x 2047 scored positions = 51,175 scored positions
```

Five strata rather than Fruit's two, because Fruit's own disclosure records
that its two strata differed by nearly 2× on their own (literary 0.0275,
scientific 0.0500) — a two-stratum panel is as much a measure of the mix as of
the model. `scientific` has 42 documents, so 5 per stratum is comfortably
inside eligibility (`≥ skip + context_length = 4096` tokens).

### What it can and cannot resolve

Extrapolated from the Fruit panel's own published disclosure — per-window sd
0.0283 around a mean of 0.0387 over 16 windows, standard error 0.0071, *"cannot
separate artifacts that differ by less than roughly 30 percent"*:

| windows | positions | SE | as % of mean | cannot separate closer than |
|---:|---:|---:|---:|---:|
| 8 | 16,376 | 0.0100 | 25.9% | ~42% |
| 16 | 32,752 | 0.0071 | 18.3% | ~30% (the published figure) |
| **25** | **51,175** | **0.0057** | **14.6%** | **~24%** |
| 36 | 73,692 | 0.0047 | 12.2% | ~20% |
| 64 | 131,008 | 0.0035 | 9.1% | ~15% |
| 144 | 294,768 | 0.0024 | 6.1% | ~10% |

Resolution improves as `1/√n`, so buying a 2× tighter number costs 4× the
panel. **25 windows is the recommendation** because it matches the sealed
panel's statistical weight (51,175 positions exactly), lands at ~24%
separability, and — critically — **on a layer-outer engine the window count is
nearly free**: 25 windows and 8 windows read the same 1,486.8 GB and differ
only by ~35 seconds of compute. On a per-window-re-read engine, 25 windows
costs 3× what 8 does. That asymmetry is a second, independent argument for
building the layer-outer path before spending on the capture.

Two things this panel will *not* do, and both must be disclosed on the row:

- **It cannot resolve FP8's error if FP8 is very good.** Flash's official FP8
  scored 0.0206 against a cross-stack floor of 0.0127. If GLM-5.3's FP8 lands
  similarly close to its floor, a 25-window panel resolves the *presence* of
  the gap, not its size to better than a quarter.
- **It carries no contamination scan.** `build_token_panel.py` records the
  caller's `--separation-note` verbatim and runs no shingle scan. Against a
  frontier model whose pretraining corpus is undisclosed, "held out" is not
  claimable. Say `separation asserted at source level only`, as the Fruit panel
  does.

---

## 6. The staged plan

Three stages, a go/no-go gate between each. Rates are live from `jl gpus
--json`: H200 141 GB spot IN2 **$1.99**/GPU-h (on-demand $3.99), RTX PRO 6000
96 GB spot IN1 **$0.99**, L4 spot **$0.29**, storage **$0.00017**/GB-h (the
repo's own inferred planner constant, ±100%; the vendor page implies $0.00014).
**No B200 is offered on this platform** — the newest silicon is Blackwell RTX
PRO 6000 at 96 GB. Max 8 GPUs per instance.

**Account balance: $90.318.** That number is a hard constraint on stage C and
it is why the staging matters.

### Stage A — the free smoke. Cost: $0.00. Do this first, today.

Two parts, both already possible without renting anything.

**A1 — already done, in this document.** A meta-device instantiation of
`GlmMoeDsaForCausalLM` from the real config, diffed against the real
safetensors index. Results are in §1 and §2 and they answer three of the
questions this stage was meant to answer: the model builds, the indexer
schedule is coherent, layer 78 is not built, `tie_word_embeddings` is false and
the head carries **no bias** (so `hf_capture`'s HEAD gate passes).

**A2 — a truncated real-weights capture on the operator's own M4 Max.**
*(DONE 2026-08-30. Results, and two corrections to the recipe below, in
section 9.)* The
local box has **137 GB unified memory and 1.3 TiB free disk**. Fetch only the
**9 shards (43.2 GB)** that carry `embed_tokens`, `lm_head`, `model.norm` and
layers 0–3, write a pruned index, set `num_hidden_layers: 4`, and run
`bin/fidelity-dataset capture --engine hf-transformers --device cpu` (or `mps`)
on a 2-window panel.

That proves, for **zero dollars**, every mechanism that stage B would otherwise
discover on a metered box:

1. the `WeightConverter` fires and `missing_keys` is empty (see the gate below);
2. layers 0–2 build their own indexer and layer 3 correctly consumes layer 2's
   `prev_topk_indices` through the `shared` path;
3. the `lm_head` forward pre-hook fires exactly once and yields
   `[1, seq, 6144]` in bf16;
4. `build_token_panel.py` produces a panel `hf_capture` accepts, and the
   three-step `capture | verify | compare --self-compare | publish` chain runs
   end to end on `glm_moe_dsa` at hidden width 6144.

**Additionally, and this is the highest-value check in the whole plan:** read
`model.model.layers[3].mlp.experts.gate_up_proj[k]` back out and compare it
byte-for-byte against the raw shard bytes for
`model.layers.3.mlp.experts.{k}.{gate,up}_proj.weight`, for several `k`
including 0 and 255. That proves the 256→1 expert merge is correct **and in the
right order** (`MergeModulelist(dim=0)` then `Concatenate(dim=1)`, gate rows
first, matching `modeling_glm_moe_dsa.py:571`'s `chunk(2, dim=-1)`).

> **Gate A → B.** Stop if `missing_keys`, `mismatched_keys` or `error_msgs` is
> non-empty; if the fused expert bytes do not match the shard bytes exactly; if
> the head hook fires more than once; or if the hidden dtype is not bf16.
> Any of these is a correctness failure, and every one of them is cheaper to
> find here than at 1.5 TB.

### Stage B — a small-panel root capture at scale. Cost: see the table.

Prove the path on real hardware at the real size, with an 8-window panel and
**two cold runs in two separate processes**, verified by
`fidelity-dataset compare --self-compare` (the spec's SC-1). Two runs, because
one is not evidence of determinism — the campaign's own convention.

The cost depends entirely on which engine exists by then:

| lane | $/h | hours | **cost** |
|---|---:|---:|---:|
| **B-3** 1× H200 + a layer-outer engine, fast IO | 2.28 | 1.07 | **$2.45** |
| **B-3** 1× H200 + a layer-outer engine, slow IO | 2.28 | 1.29 | **$2.93** |
| **B-2** 1× H200 + accelerate disk-offload, fast IO | 2.51 | 5.49 | **$13.77** |
| **B-2** 1× H200 + accelerate disk-offload, slow IO | 2.51 | 7.19 | **$18.04** |
| **B-1** 8× H200 + `device_map="auto"` | 16.21 | 3.19 | **$51.76** |

Add **$0.40–0.56** of one-time fetch to every row. Assumptions:

- **B-1** places ~1,050 GiB of 1,385 GiB on eight H200s and spills ~334 GiB to
  CPU RAM (8 × 300 GB = 2,400 GB, so it fits). Per-window that is a 358 GB
  pageable H2D transfer; this repo measured pageable H2D at **1.23 GB/s**
  (`k6/STREAMING.md`: "a 14.50 GB pageable H2D copy (~11.8 s)"), giving **4.9
  min/window**. Every cold run needs a fresh process, hence a full 0.75 h
  reload. This lane needs only a small `hf_capture` patch — pass `device_map` /
  `max_memory` and skip the `model.to(device)` that follows `from_pretrained`,
  because `.to()` on a dispatched model raises.
- **B-2** uses accelerate's disk offload, which **writes a second full copy** of
  the weights as `.dat` files: 1,506.7 GB of source + 1,486.8 GB of offload =
  **~3.0 TB**, which exceeds the 2,048 GB instance-disk cap and therefore needs
  a 2,048 GB instance disk *and* a 2,048 GB shared filesystem. That is the
  hidden cost of this lane, and it is why its storage line is $0.52/h.
- **B-3** is the only lane that reads 1,486.8 GB once per run instead of once
  per window, and the only one whose cost does not move when the panel grows.

> **Gate B → C.** Stop if the two cold runs are not bitwise identical; if peak
> memory exceeds the §2 table by more than the 1.35× headroom factor
> `census.py` already applies; if wall clock per window exceeds 1.5× the range
> in §4 (the IO estimate was wrong and stage C's price is wrong with it); or if
> the instance is preempted more than once, which prices stage C out on its own.

### Stage C — the full root capture and publish

25 windows, two cold runs, then `verify`, `compare --self-compare`, seal, and
the registry path.

| lane | $/h | hours | **cost** | fits $90.32 balance? |
|---|---:|---:|---:|:--:|
| **C-3** 1× H200 + layer-outer engine | 2.28 | 1.1–1.3 | **$2.45–2.93** | yes, 30× over |
| **C-2** 1× H200 + accelerate disk-offload, fast IO | 2.51 | 15.24 | **$38.23** | yes |
| **C-2** 1× H200 + accelerate disk-offload, slow IO | 2.51 | 20.57 | **$51.59** | yes, thin |
| **C-1** 8× H200 + `device_map="auto"` | 16.21 | 5.94 | **$96.32** | **NO** |

Plus fetch, plus whatever stage B consumed.

**This is the decision.** Stage C on the existing-tooling multi-GPU lane costs
more than the account holds. Stage C on the disk-offload lane costs $38–52 and
leaves the balance too thin for a retry after a spot preemption on a
15–21-hour run — and a 15–21-hour spot run *will* be preempted. Stage C on a
layer-outer engine costs under $3 and is preemption-tolerant because a run is
25 minutes long.

**The staging recommendation is therefore: do Stage A now for free; build the
layer-outer capture path; then B-3 and C-3 for a combined GPU spend under
$7.** The engineering is the price, not the GPU-hours — which is exactly what
the BF16-floor measurement already taught us: *"The measurement cost $18.90 —
but most of that was BUILDING the native-BF16 student mode in stream_score.py,
not measuring."*

---

## 7. Risks, ranked

**R1 — the expert-fusion conversion is 98% of this checkpoint, and
`hf_capture`'s guard cannot see it fail.** *(CLOSED by Stage A -- 836/836
tensors byte-exact, and the guard now reads `conversion_errors`. See section 9.3
and 9.4. The `down_proj` shape stated below is wrong; see section 9.2.)* `GlmMoeDsaForCausalLM` holds routed
experts as fused 3-D parameters (`experts.gate_up_proj` `[256, 4096, 6144]`,
`experts.down_proj` `[256, 2048, 6144]`) while the checkpoint ships one matrix
per expert per projection. A naive key-set diff reports **150 missing keys**;
they are saved only by `conversion_mapping.py`'s `glm_moe_dsa` entry
(`MergeModulelist(dim=0)` + `Concatenate(dim=1)`), which I confirmed exists in
transformers 5.16.1 and which is why Fruit — same per-expert layout — loaded.
**57,600 of the checkpoint's 59,585 tensors (96.7%) are consumed by that
converter** and collapse into 150 fused parameters; a further 791 (all of
layer 78) are legitimately unexpected; only 1,194 load by name.

Now the sharp edge: `LoadStateDictInfo.to_dict()` — the exact dict
`output_loading_info=True` hands `hf_capture` — **deliberately omits
`conversion_errors`** ("to be coherent with legacy reporting in the tests"). And
`hf_capture.missing_weight_keys` reads **only** `missing_keys`, ignoring
`mismatched_keys` (whose loading report says *"Reinit due to size mismatch"* —
a randomly-initialised tensor by another name) and `error_msgs`.

*Mitigation:* Stage A2's byte-for-byte comparison of fused expert rows against
raw shard bytes, plus a scoped hardening of `hf_capture` to refuse on
`mismatched_keys` and `error_msgs` and to log `unexpected_keys` (which here will
legitimately contain layer 78's 791 MTP tensors and must be disclosed, not
refused). Reading `conversion_errors` requires the `LoadStateDictInfo` object
rather than its dict.

**R2 — `hf_capture.py` cannot load this model on any machine JarvisLabs
rents.** *(HALF-CLOSED by Stage A: `--device-map` / `--max-memory` /
`--offload-folder` now exist and skip the fatal `.to()`, proven on
`glm_moe_dsa`. The ECONOMICS below stand unchanged. See section 9.5.)* `load_model` calls `from_pretrained(...)` with no `device_map`, then
`model.to(device)`. With `device_map=None`, transformers materialises 1,486.8 GB
on **CPU**; the largest per-GPU RAM on offer is 300 GB (H200 IN2), so a
single-GPU instance is OOM-killed before it ever reaches `.to()`. And `.to()`
would then move 1,486.8 GB onto one 141 GB card. Even the whole box does not
save it: **8 × H200 = 1,128 GB of VRAM < 1,486.8 GB.** No JarvisLabs
configuration holds this model in VRAM.

*Mitigation:* the `device_map` / `max_memory` / `offload_folder` patch described
in §6 B-1, or the layer-outer engine. Either way this is a **tooling change and
a blocker**, not a runtime surprise. It is also the reason the streaming lane
cannot substitute: `k6/tools/hidden_replay.py` hard-codes `HIDDEN_WIDTH = 4096`
and `import stream_score`, and `stream_score.py` is not in this repository —
`fidelity_dataset.py` already refuses `--engine sealed-lane` for exactly that
reason.

**R3 — CAPTURE-03: would the random-initialisation refusal fire here?**
*(Its blind spot is closed; see section 9.4.)* Yes,
for the failure mode it was built for. `hf_capture.py:390` raises
`REFUSED: N parameter(s) were NOT in the checkpoint and were randomly
initialised` unless `--allow-missing-weights`, and it is reached before the
panel loop, so it costs a load and not a capture. Crucially, **if the fusion
converter fails on any single layer** — an incomplete shard, a truncated
`.incomplete` file, a transformers version without the `glm_moe_dsa` mapping —
that layer's `gate_up_proj` and `down_proj` land in `missing_keys` and the
refusal fires. That is exactly the right behaviour and it is the strongest
argument that this run is safe to attempt. Its blind spot is R1's
`conversion_errors` / `mismatched_keys`.

**R4 — spot preemption on a multi-hour run.** All four paid measurements ran
0 preemptions, but they were 2.8–5.6 h. Stage C on the disk-offload lane is
**15–21 h on a spot instance**, which is a different risk class; the campaign
has already lost ~65 minutes to one external kill at window 22 of 25 (M2).
*Mitigation:* prefer the layer-outer lane, where a run is ~25 minutes and a
preemption costs one run, not a day. Failing that, checkpoint per window (the
capture already writes one safetensors file per window, so resume is a
bookkeeping change, not a compute change) and hold `--max-runtime` low.

**R5 — disk quota.** ~1,548 GB against a 2,048 GB cap is comfortable, but the
disk-offload lane needs ~3.0 TB across two volumes and the campaign has hit
"Disk quota exceeded" twice. *Mitigation:* provision 1,700 GB, keep the
checkpoint on a shared filesystem so the fetch survives an instance restart,
and apply the existing free-space guard pattern from `stage_k6.sh`.

**R6 — `hf_capture.py` has only ever run on 0.1B and 5B models.** True, and
the gap to 753 B is four orders of magnitude. But the specific things that
break at scale are enumerable and all appear above: the load path (R2), the
fusion conversion (R1), and the per-window re-read (§4). The panel loop, the
hook, the seal, the manifest and the three-step chain are size-independent —
they move `positions × 6144 × 2` bytes and nothing else. *Mitigation:* Stage
A2 exercises every size-independent mechanism at hidden width 6144 for free.

**R7 — the account balance is smaller than one existing-tooling stage C.**
$90.318 against $96.32 (C-1) or $38–52 (C-2, with no margin for a preempted
retry). *Mitigation:* this is the go/no-go, not a risk to mitigate around.
Either the layer-outer engine gets built or the campaign needs a top-up before
stage C.

**R8 — MPS numerics are not CUDA numerics.** The M4 Max fits the 81.7 GB
schedule and has 1.3 TiB free — tantalisingly close, but **1.3 TiB < 1.507 TB**,
so the full checkpoint does not fit locally, and a root captured on MPS would
not be same-lane comparable to CUDA candidates anyway. Use the Mac for Stage A
and nothing else.

---

## 8. Verdict

**GO-WITH-STAGING.**

The model is our architecture, the config holds no surprises, the census
reconciles to zero bytes, the memory fits one H200 three times over, the
checkpoint fits a JarvisLabs disk with 500 GB to spare, and the fetch is an
hour. The refusal that protects us from measuring randomly-initialised weights
is present and would fire on the most likely load failure.

**The single most likely failure mode: `hf_capture.py` cannot load this
checkpoint at all, because it materialises the whole model on CPU and then
calls `.to(device)` — and 1,486.8 GB exceeds both the largest RAM on offer
(300 GB) and the entire VRAM of a full 8× H200 node (1,128 GB).** That is not a
risk that might materialise; it is a certainty that will halt stage B on the
first attempt unless the engine is changed first. Everything else in this
document is arithmetic; this is the one thing that must be built.

Spend nothing until Stage A is done. It is free, it runs on the operator's own
desk, and it is the difference between finding R1 for $0 and finding it for $50.

---

## 9. Stage A: what it actually proved (run 2026-08-30, $0.00)

Stage A ran end to end on the operator's M4 Max. **No GPU was rented, nothing
was published, no registry row was written, and no number produced here is a
measurement.** The model it ran on is four layers deep; its KLD is arithmetic
about a truncation, not fidelity.

**Verdict: GO for Stage B.** R1 — the risk this stage existed to close — is
closed, with counts rather than a summary. R3's guard was widened to see the
three failures it was blind to. R2's *mechanism* is now fixed and proven on the
real architecture; R2's *economics* are unchanged and remain the gate on
Stage C.

### 9.1 What was fetched, and what it cost

`k6/tools/fetch_truncated_ckpt.py` (new) range-fetches only the byte ranges of
the tensors a truncated model needs, writing them at their published offsets
into **sparse** local shards under the published safetensors headers.

| | planned in §6 | actual |
|---|---:|---:|
| shards touched | 9 | 9 |
| download | 43.2 GB (whole shards) | **25.948 GB** |
| tensors kept | — | **836** |
| wall clock | — | **4m 53s** at 91 MB/s |
| on-disk after | — | 24 GB (sparse; 43.2 GB apparent) |
| dollars | $0.00 | **$0.00** |

Whole-shard fetching would have moved 17.2 GB it never reads: five of the nine
shards carry wanted tensors and unwanted ones side by side (shard 2 holds
**one** layer-1 tensor and 211 layer-10 tensors). Range fetching is the right
tool here even though §3 correctly rules it out for the *full* capture.

Truncated model: 4 layers (0–2 dense, 3 sparse with all 256 routed experts),
**12,973,900,544 parameters = 25,947,801,088 bytes**. The fetch plan's
25,947,802,112 bytes exceeds it by exactly **1,024 bytes** — `mlp.gate.`
`e_score_correction_bias`, 256 fp32 values, a buffer rather than a parameter.
The census reconciles at this scale too.

### 9.2 Two corrections to §6 and §7

**§6's Stage A recipe does not load as written.** Setting only
`num_hidden_layers: 4` is refused by `transformers` 5.16.1:

> `ValueError: num_hidden_layers (4) must be equal to the number of mlp_layer_types (78)`

The per-layer schedule lists must be truncated with it. `fetch_truncated_ckpt.py`
truncates every config value that is a 78-long list of strings —
`mlp_layer_types` and `indexer_types` — which preserves the schedule of the
layers kept: `dense, dense, dense, sparse` and `full, full, full, shared`. So
layer 3 is still a `shared` indexer consuming layer 2's `prev_topk_indices`,
exactly as in the 78-layer model.

**§7 R1 states the fused shape wrong.** `down_proj` is `[256, 6144, 2048]`
(`[E, hidden, moe_intermediate]`), not `[256, 2048, 6144]`. `gate_up_proj` is
`[256, 4096, 6144]` as stated. The confirmed converter is

```
["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"]
    -> mlp.experts.gate_up_proj   via MergeModulelist(dim=0), Concatenate(dim=1)
 "mlp.experts.*.down_proj.weight"
    -> mlp.experts.down_proj      via MergeModulelist(dim=0)
```

and `modeling_glm_moe_dsa.py:571` splits the *output* of
`linear(x, gate_up_proj[k])` with `.chunk(2, dim=-1)`, so rows `0:2048` are
gate and rows `2048:4096` are up.

### 9.3 R1 — CLOSED. The converter is honest, byte for byte

`k6/tools/verify_fused_experts.py` (new) loads the checkpoint through
`hf_capture.load_model` — the production path — then reads each per-expert
matrix's raw bytes straight out of the local shard at its published offset,
with no `safetensors` reader and no `transformers` in the path, and `memcmp`s
it against the corresponding slice of the live fused parameter.

| comparison | count | differed |
|---|---:|---:|
| `gate_up_proj[k][0:2048]` vs `experts.k.gate_proj.weight` | 256 | **0** |
| `gate_up_proj[k][2048:4096]` vs `experts.k.up_proj.weight` | 256 | **0** |
| `down_proj[k]` vs `experts.k.down_proj.weight` | 256 | **0** |
| every other parameter and buffer, name-mapped 1:1 | 68 | **0** |
| **total checkpoint tensors compared** | **836** | **0** |

836 compared, 836 exact, and 836 of 836 is every tensor fetched. The fetch
receipt's per-tensor sha256 were re-derived from the shard bytes at check time:
768 re-checked, 0 mismatched. The two parameters with no checkpoint key are
`model.rotary_emb.inv_freq` and `.original_inv_freq` — computed from the
config, never shipped, correctly absent.

**The same check passes through the `--device-map` path** (`{"": "cpu"}`,
`accelerate` 1.14.0): 836/836 exact. Dispatching does not change the converter's
output.

The load report, read in full:

```
observed true   conversion_errors_visible true
missing_keys 0   mismatched_keys 0   error_msgs 0   conversion_errors 0
unexpected_keys 43
```

**The 43 unexpected keys are not what §7 predicted, and the reason matters.**
They are not layer 78's MTP tensors — they are tensors of layers 9, 10, 19, 20,
29 and 30 that happen to share the nine fetched shards. `transformers` 5.16.1
enumerates each shard's **own safetensors header**, not merely the
`weight_map` of the index, so a pruned index does not stop it from finding —
and running the fusion converter over — tensors it encounters in the file. In
this sparse tree those regions are holes that read as zeros. They were
classified `UNEXPECTED` and discarded, so nothing was harmed. But that is
precisely the mechanism by which a hole could have been served as a tensor of
zeros had a key matched, and it is why the instrument for R1 has to be a byte
comparison and not a key-set diff. On the full Stage B tree the same field will
legitimately carry layer 78's 791 MTP tensors.

### 9.4 R3 — the guard now sees what it needed to see

`hf_capture.py` read `missing_keys` and nothing else. It now reads the whole
report, through `load_report()` and `refuse_on_load_report()`:

| signal | before | now |
|---|---|---|
| `missing_keys` | refuses (`--allow-missing-weights` overrides) | unchanged |
| `mismatched_keys` (*"Reinit due to size mismatch"*) | **ignored** | refuses, same override |
| `error_msgs` | **ignored** | refuses, **not** overridable |
| `conversion_errors` | **not even visible** | refuses, **not** overridable |
| no report at all | read as a clean report | refuses, not overridable |
| `unexpected_keys` | never recorded | `checkpoint_tensors_not_loaded` disclosure |

`conversion_errors` is reachable because `_FullLoadingReport` wraps
`LoadStateDictInfo.to_dict` for the duration of the load — the method whose own
source says it omits the field *"to be coherent with legacy reporting in the
tests"*. If a `transformers` build offers no such class, the wrap does not take
effect and the guard **refuses** rather than reporting a clean bill of health it
never checked.

Two honest qualifications. First, `transformers` 5.16.1 already raises from
inside `from_pretrained` on both `conversion_errors` and `mismatched_keys`, so
on this build the load fails either way — the old code failed closed here, with
a confusing message. The hardening matters because that is the library's
choice, not ours: `ignore_mismatched_sizes=True`, an older build, or a
refactor all hand the report back instead of raising, and then this reader is
the only thing between a reinitialised tensor and a published number. Second,
`_from_pretrained` used to return a bare `{}` on its fallback path, so an
**unexamined** load and a **clean** load were the same value. That one was live.

Regression tests A17–A22 in `bin/selftest_hf_capture.py`; all six fail against
`hf_capture.py` at commit `b3f518c` (verified by running the new selftest
against a `git archive HEAD` tree) and pass after. A17's failure there reads
`missing_weight_keys -> []` on a report carrying a mismatched key: the blind
spot, stated.

### 9.5 R2 — mechanism fixed, economics unchanged

`load_model` now accepts `device_map` / `max_memory` / `offload_folder`, and
**skips the `model.to(device)`** that follows `from_pretrained` when a
`device_map` is given, because `.to()` on a dispatched model raises. Proven on
`glm_moe_dsa` itself, not only on a fixture (§9.3). It refuses with a clear
message when `accelerate` is absent instead of surfacing the library's.

That unblocks the B-1 and B-2 lanes. It does **not** change their prices, and
the layer-outer engine of B-3/C-3 is still unbuilt. §6's cost table stands.

**A new number for the Stage B budget: the converter's transient is one layer,
not three.** Loading the truncation peaked at **48.2 GB RSS** against
25.95 GB of resident parameters — a **22.3 GB** excess, or 1.15x the 19.33 GB
of one sparse layer's routed experts. So the fusion holds roughly the source
copy of the layer it is building and not more. Budget **one extra sparse-layer
buffer (~19.4 GB) of headroom above resident during load**, per layer in
flight. Caveat: safetensors mmaps the shards, so an unknown part of that 22.3 GB
is evictable file-backed page cache rather than anonymous memory — treat 19.4 GB
as an upper bound, not a floor.

### 9.6 The capture chain, on `glm_moe_dsa` at hidden width 6144

Two cold captures in two separate processes, through
`bin/fidelity-dataset capture --engine hf-transformers`:

| | |
|---|---|
| windows / scored positions | 2 / 4,094 |
| per window | 70.8 s, 67.6 s (CPU, M4 Max) |
| total per run | 158.2 s |
| head hook | fired **exactly once** per window |
| head input | `[1, 2048, 6144]`, **bfloat16**, bias `None` |
| hidden form / logit-form equivalent | 50.3 MB / 2.54 GB (50.4x) |
| `verify` | VERIFIED, tensors recomputed |
| `validate --strict` | **0 errors, 0 warnings** |
| `capture_content_digest`, both runs | `faa26e5ac6…` — **identical** |
| `compare --self-compare` | **exactly 0.0** (SC-1) |

Every gate in §6's *Gate A → B* passes: `missing_keys`, `mismatched_keys` and
`error_msgs` all empty; the fused expert bytes match the shard bytes exactly;
the head hook fired once; the hidden dtype is bf16.

The sealed manifests carry three disclosures — `no_known_deviations`,
`reduced_run_count`, and the new `checkpoint_tensors_not_loaded` naming the 43
unused tensors. Evidence lives in `k6/tools/glm53-stagea-evidence/`. **The
datasets themselves were deleted**: they are a truncation, they are not a
measurement, and nothing should be able to mistake them for one.

### 9.7 What Stage B should now cost, and the one thing still unknown

Nothing learned here moves §6's price table: the fetch is still ~1 h and
$0.40–0.56, and B-1/B-2/B-3 still cost $51.76 / $13.77–18.04 / $2.45–2.93. What
changed is *risk*, not price:

* R1 no longer needs paid discovery. The converter was exercised on a real
  sparse layer with all 256 experts and was exact. The remaining 75 sparse
  layers go through the identical layer-agnostic code path.
* R2's blocker is no longer "the engine cannot express this". B-1 and B-2 are
  runnable today with the `device_map` flag.
* Stage B should add `--max-memory` headroom of one sparse-layer buffer
  (~19.4 GB) above its resident plan, per §9.5.
* Stage B must not reuse this panel. `panel--glm53.stagea.smoke.2w` is two
  windows built to exercise a code path; §5's 25-window, 5-stratum panel is
  still the recommendation.

**The single biggest remaining unknown is unchanged and is the one §8 named:
no engine reads this checkpoint economically.** `device_map` makes the load
*expressible*, not cheap — B-1 spills ~334 GiB to CPU RAM and pays a 358 GB
pageable H2D transfer per window; B-2 writes a second full 1,486.8 GB copy to
disk. The layer-outer, window-inner schedule that reads the tree once per run
instead of once per window is still the difference between a $3 stage C and a
$38–96 one, and it is still unwritten. Stage A moved the correctness risk to
zero and left the engineering exactly where it was.
