# M1 (Qwen3.8-27B) — what M2 and M3 inherit

**Status:** rung complete, 2026-08-30. True cost **$5.12** on one on-demand
RTX PRO 6000 (target was $8-15).

## The sealed numbers

Panel `panel--qwen38.malaiwah.suite-v5-shard0-1m` — 512 contexts x 2048 tokens,
**1,048,064 scored positions**, vocab 248,320. Reference
`reference--malaiwah.qwen38-bf16-hf.suite-v5-shard0-1m`, lane
`local-cuda-budget`, engine `hf-transformers`.

| row | mean tokenwise KLD (nats) | top-1 | class |
|---|---:|---:|---|
| **floor** — root vs root, `--force-compute` | **0.0** | **1.000000** | strict |
| FP8 (dequantize-and-run, weights-only) | 0.002989850396847924 | 0.977509961223742 | advisory |
| AWQ-INT4 (cyankiwi), executed natively | 0.022449361029279465 | 0.940180179836346 | advisory |

**THE FLOOR IS EXACTLY 0.0, AND IT WAS MEASURED, NOT ASSUMED.** Every percentile
of the self-compare — median, p95, p99, p99.9, max — is also 0.0. It was run
twice: once answered by the capture-digest short-circuit and once with the
estimator forced to execute (`backend: torch:k6_kld_report._token_kld`), and both
produced the same `tokenwise-kld.npy` digest `8be5dcca...`, so the forced
computation reproduces the hash proof byte for byte rather than merely agreeing
with it.

**Therefore attributable error EQUALS raw KLD for every row above, with nothing
subtracted.** That is the architectural payoff of splitting capture from
comparison, now demonstrated on a real model rather than on the 0.1B fixture.

Determinism: **three** cold captures of the root in three separate processes all
produced `capture_content_digest`
`2376837de2e42561a196a3f33e25ab6e79471bed0f97c5949605656ca97504c3`. The third ran
concurrently with a CPU-saturated comparison and still matched.

Published root: <https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-root-v1>
(`dataset_sha256 8a658364...`, public, re-verified after upload).

**Two of the four named candidates were never capturable on this lane** —
`transformers` has no `exl3` quantizer — so `turboderp/Qwen3.8-27B-exl3` and
`malaiwah/Qwen3.8-27B-EXL3-K6-parity` are not measured here. See learning 8.

---

Numbered so later rungs can cite them. Each says what happened, what it cost,
and what to do differently.

## Cost and timing

**1. The `min/window` anchor is regime-dependent, and for a resident model it is
~250x smaller than the plan assumed.** A 512-window root capture of a 27B model
resident on one RTX PRO 6000 took **335.1 s — 0.0109 min/window**, against the
plan's 2.37–3.12 min/window GLM-5.3-Flash anchor. That anchor describes the
*streaming* lane. When the model fits the device, capture is nearly free and the
cost model should not budget it as the dominant term.

**2. The dominant cost is the COMPARISON, not the capture, and it is CPU-bound.**
`dscompare._replay` applies the head with a **numpy matmul on the CPU**; only the
fp64 KLD reduction goes to the GPU. During a compare the GPU sits at 0% while 20
cores saturate. Per window the replay is
`2 x (positions x vocab x hidden x 2)` FLOPs — for this panel ~10.4 TFLOP/window,
~5.3 PFLOP per compare. **Measured: one 512-window comparison took 60 min 19 s
(0.118 min/window) against a 335 s capture (0.0109 min/window) — the comparison
costs ~10.8x the capture it consumes**, on identical data and the same box.

*Consequence for M2/M3:* budget compares, not captures. A GLM-5.3 root capture is
cheap; each candidate comparison against it is not.

**FIXED in M1.5, and the projection was right.** `compare --replay-device cuda`
runs the head matmul on the device the estimator already uses, one position
block at a time, so the full `[positions x vocab]` fp32 logit array is never
materialised on the host. Same box, same 512-window comparison, end to end
through the CLI: **1,754.71 s (29 min 15 s) -> 173.27 s, 10.13x**, peak **7.13 GB**
of device memory, GPU at 88% where the numpy path leaves it at 0% (watched
live). The comparison is no longer 10.8x the capture; it is
0.52x of it.

It is **opt-in**: an fp32 GEMM accumulates in an order the BLAS chooses, so
switching backends moves the value. Measured on real root data, 32,752
positions, `KLD(numpy replay || cuda replay)` = **5.237e-12 nats** mean,
1.791e-10 max, **top-1 agreement 1.000000** — 1.75e-9 relative to the smallest
published row here, and 2.2e9 below the streaming lane floor. Small, not zero,
so the 16-digit rows above would not reproduce and the default stays numpy.
**The floor is immune and was re-verified**: the same self-compare through the
GPU path returns exactly 0.0 with `tokenwise-kld.npy` digest `8be5dcca...`
byte for byte. See `docs/CAPTURE-SCALING-PLAN.md` §3.

**3. Fetch is not the bottleneck.** 52 GiB pulled in 85 s ≈ **627 MB/s** with
`HF_HUB_ENABLE_HF_TRANSFER=1` on a 28-core IN1 box — at or above the top of the
plan's 430–600 MB/s band. For a 642.7 GB Flash re-capture that projects to
~17 min of fetch, which overlaps with compute under layer-outer.

**4. Captures can run concurrently with compares at no correctness cost.** Root
run 3 was captured while a compare saturated the CPU and still produced a digest
bitwise identical to runs 1 and 2. Capture is GPU-bound, comparison is CPU-bound;
packing them is free wall-clock. Do it.

*Amended by M1.5:* the correctness half stands — a concurrent compare did not
move a capture digest. The **free wall-clock** half was a property of the
comparison being CPU-bound, and `--replay-device cuda` makes it GPU-bound (88%
utilisation). Under the fast path they contend. Pack a comparison against a
*fetch* instead, or leave the comparison on the numpy path while a capture is
running — which is also the right choice when the comparison belongs to an
already-published group.

**5. Spot capacity for a named GPU can be zero even when `jl gpus` shows the row
available.** `jl gpus` showed RTX-PRO6000 IN1 with a green dot; `jl create --spot`
refused with "No free RTX-PRO6000 GPUs". `jl resources --json` gives the real
per-server `num_free_devices` / `effective_num_free_devices` — check that, not the
dot. On-demand at $1.89/h was taken deliberately: a preempted spot instance
mid-capture costs more than the price difference.

## Tooling gaps and traps

**6. `unexpected` tensors in the load report are the silent-corruption case and
nothing gates on them.** `hf_capture` refuses on MISSING weights (behind
`--allow-missing-weights`) but only *logs* unexpected ones. On
`Qwen/Qwen3.8-27B-FP8` that line — `unexpected: 64`, all
`mlp.gate_proj.weight_scale_inv` — was the only signal that the model had loaded
wrong. A number taken from that load would have been published as a quantization
result and would have been a loader artifact.

**FIXED in M1.5.** `refuse_on_load_report` now has a fifth branch: unexpected
tensors REFUSE the capture, naming a few of the keys and saying what it usually
means — a quantization path that silently did not engage, leaving its scale
tensors orphaned. `--allow-unexpected-tensors` is the escape and stamps a
**blocking** `unexpected_tensors_overridden` disclosure, mirroring
`--allow-missing-weights`. The benign reading is indistinguishable from inside
the loader (GLM-5.3-BF16 ships a 791-tensor MTP layer the architecture does not
build), which is exactly why there is an override and why using it is loud.
*Consequence for M2:* the GLM-5.3-Flash root capture will need the flag, and its
sealed dataset will carry a blocking disclosure — a label on the dataset, which
does not propagate into a comparison receipt and does not block a registry row.
Regression: `bin/selftest_hf_capture.py` A21/A23/A24.

**7. Producer exclusion lists collide with prefix matching.**
`transformers.quantizers.quantizers_utils.should_convert_module` tests
`re.match(key, full_name)`, anchored only at the start. The FP8 producer listed
`...layers.N.mlp.gate` (a MoE router that does not exist in this dense
checkpoint), which also matches `...layers.N.mlp.gate_proj`. Result: **65 of 65
`gate_proj` modules excluded from FP8 conversion, 0 of 65 `up_proj`** — fp8
weights loaded into bf16 Linears with the block scale never applied. Verify the
converted/excluded split against the real tensor-name list before trusting any
quantized capture. It is four lines of code and it caught a defect.

**8. Not every artifact is capturable by this engine, and the boundary is
`transformers`, not the budget.** Supported quant methods on this stack:
`aqlm, auto-round, awq, bitnet, bitsandbytes_*, compressed-tensors, eetq,
fbgemm_fp8, fouroversix, fp8, fp_quant, gptq, higgs, hqq, metal, mxfp4, quanto,
quark, sinq, spqr, torchao, vptq`. **`exl3` is not among them**, so
`turboderp/*-exl3` and `malaiwah/*-EXL3-K6-parity` cannot be captured with
`--engine hf-transformers` at all. Same-lane EXL3 numbers need either an EXL3
capture surface on this lane or the dequantize-and-run road (7/9 below). Plan
the candidate list around what the engine can load, and check it *before*
renting.

**9. Dequantize-and-run is the general fallback and it needs validating, not
trusting.** Where the vendor kernel is unavailable or wrong, decode the stored
weights and run densely — the methodology the campaign's GGUF/EXL3/MLX rows
already use. Validate the decode before spending a capture: compare per-tensor
`rel_L2` against the root. FP8 E4M3 came out at **0.0265 uniformly across
gate/up/down/q**, which both confirms the scale convention (multiply by
`weight_scale_inv`) and proves `gate_proj` was fixed — a dropped scale would have
shown a wildly different error for that one projection. **State the limit on the
row:** the checkpoint declares `activation_scheme: "dynamic"`, so a weights-only
number is a LOWER BOUND on the served model's divergence.

**10. Pin optional-kernel packages; an unconstrained upgrade bricks the box.**
`pip install -U kernels` installed 0.16.1 and **broke `import transformers`
outright** (5.8.1 pins `kernels>=0.12.0,<0.13`). Install the constrained range and
re-verify `import transformers` immediately after touching the environment.

**11. Installing kernels changes the lane — prove it did not move the root.**
Adding `kernels` to run a candidate could in principle re-route fused kernels the
root also used, silently breaking same-lane comparability with an already-sealed
root. Control: re-capture 8 root windows post-install and diff per-record
`tensor_content_sha256` against the sealed root. **0 mismatches of 8.** Cheap,
decisive, and it should be standard practice whenever the environment changes
between root and candidate.

**12. A failed capture leaves its output directory and blocks the retry.** The
next attempt refuses `destination_exists`. In a chained script (`a && b && c`) one
mid-chain failure then blocks every later step on the *next* run too. Clear stale
output or pass `--force`.

**13. macOS `tar` injects AppleDouble `._*` files that break the schema loader.**
`registry/tools/_minischema.py` reads every file in `registry/schema/` and died on
`UnicodeDecodeError: 0xa3` from `._measurement.schema.json`. Pack with
`COPYFILE_DISABLE=1 tar ...`, or `find -name '._*' -delete` after extracting.

## Method

**14. A "private" panel may still be publicly recoverable — check before minting a
new one.** `panel--qwen38.malaiwah.suite-v5-shard0-1m` is recorded
`availability.status: "private"`, `uri: null`, its only source a receipt on a
laptop. Its token ids are nonetheless public, in
`malaiwah/qwen38-27b-fidelity-suite-v5` under `suite/tokens/`. Transporting them
(never re-tokenizing) and checking two seals — every context file's sha256 against
the sealed suite manifest, then the concatenated digest against the registry's
`panel_token_sha256` — reproduced `caef8a46…` exactly. **Reuse beats minting**: it
holds the tokens fixed so a new-lane row and an old-lane row differ by the lane
alone.

**15. Same panel does NOT mean rankable, and a same-lane root does NOT
retroactively upgrade anything.** The comparability key binds the reference. The
37 existing Qwen3.8 rows are against a vLLM teacher; these are against a
`transformers` teacher. Same panel makes the difference *interpretable*, not
*comparable*. Say this on every row rather than letting a reader infer otherwise
from the shared `panel_id`.

**The plan said this rung would "retroactively upgrade" the 37 existing rows. It
did not, it could not, and the claim is withdrawn.** Capturing a same-lane root
creates a NEW comparability group beside the old one — here
`cmp--05e16411a5932713` beside `cmp--4a93702ded23e01a` — because the key binds
the reference and the old rows still name the old reference. Their inferred
floors are untouched. The same holds for M2: **the eight existing
GLM-5.3-Flash rows will not be fixed by the Flash re-capture.** M2 creates a
clean group next to them, with a measured 0.0 floor, and the old eight keep
their 0.011506 inferred floor forever unless every one of them is re-measured
against the new root. Budget the re-measurements, or say plainly that the old
rows stay as they are.

**16. Check the architecture before believing the plan's one-line description.**
Qwen3.8-27B is not a dense text model: it is
`Qwen3_5ForConditionalGeneration` — multimodal (333 vision tensors), **hybrid**
attention (48 linear-attention layers, 16 full, `full_attention_interval: 4`),
with an MTP block, and **dense** (zero expert tensors). Two consequences: the
registry's coarse `tensor_class` vocabulary cannot express the hybrid split, so
`attn.qkv`/`attn.o` are genuinely part-quantized and need a disclosure rather than
a rounded verdict; and `native_scope()` emits `moe.experts=native:bf16@16` for a
model with no experts, which is a vocabulary placeholder and not a claim about the
checkpoint. **Recommend `native_scope()` consult the checkpoint's tensor names.**

**17. Derive scope from the artifact's own config plus its weight index, and
handle the pack-quantized case.** A first pass keyed on `.weight` tensors reported
every AWQ class as `native`, because `pack-quantized` emits **no `.weight` at
all** — the payload is `.weight_packed`. Work in terms of module bases (strip
every known parameter suffix, then ask whether that module carries quant state).
Writing `unknown` for a recipe the producer published is a fabricated gap; so is
reporting `native` because the probe looked for the wrong tensor name.

**18. The lane's identity includes the kernels that were NOT installed.**
`transformers` reported the fused linear-attention path unavailable
(`flash-linear-attention` / `causal-conv1d` absent) and fell back to the reference
torch implementation, for the root and every candidate alike. That fallback is
part of what these digests mean; installing those kernels is a different lane and
is not guaranteed to reproduce them. Record it.

**19. `scope.assignments[].format` must come from the registry's
`numeric_format` enum, and a natural-looking string is not in it.** Writing the
obvious `fp8-e4m3` (hyphen) or `pack-quantized-int4-g32` produces a sealed
dataset that validates with warnings and whose scope
`registry_validate.py --submission` REJECTS (`SCOPE-VOCAB`). The enum is:
`awq, bf16, exl3-mcg, exl3-mul1, exl3-trellis, fp16, fp32, fp64, gguf-i-quant,
gguf-k-quant, gptq, hqq, int4, int8, mixed, mlx-affine, mxfp4, nvfp4, fp8_e4m3,
fp8_e5m2, unknown`. Put the exact scheme (block size, group size, observer,
activation scheme) in a `declared_scheme` field and a disclosure. Catch this with
`fidelity-dataset validate` BEFORE the capture, not after — the scope block is
sealed into the dataset, and re-cutting it means re-running the capture and every
comparison whose receipt cites that dataset's `dataset_sha256`.

**20. Exit code 2 from `capture` means "sealed, with warnings", not "failed".**
`jl run` surfaces it as `failed`, which reads like a lost capture. The dataset is
written and valid. Check `validate` output before re-running anything.

## What the same-lane capture actually bought

**21. The lane term is now a measured quantity, not a caveat.** The registry's
older AWQ-INT4 row reads **0.022817869486410007** nats at top-1 **0.939436904616512**,
scored against a vLLM-captured teacher with an unmeasured cross-stack term inside
it. The same-lane row reads **0.022449361029279465** at top-1 **0.940180179836346**,
against a floor measured at exactly 0.0.

The same-lane number is **3.685e-4 nats LOWER**, and its top-1 is *higher* — both
in the direction a cross-stack term predicts, since such a term can only inflate
divergence and depress agreement. That is the first time this campaign can put a
number on what the lane was contributing to a Qwen3.8 row rather than describing
it.

Two cautions on reading it. The two rows name **different artifacts** — the older
one's identity was never established (its receipt records only a local path), so
this is not a controlled A/B on identical bytes and the delta is not certified as
"the lane term for these weights". And the comparison is only legible at all
because the panel was held fixed (learning 14); had a fresh panel been minted, the
two numbers would differ by panel AND lane with no way to separate them. **This is
the payoff argument for reusing a panel, stated as a number.**

**22. Budget the ladder by comparisons, not by gigabytes — and then M1.5 made
the comparison cheap.** This rung's bill was dominated by three comparisons at
60-104 min each, not by the 55.6 GB model. The capture side of a 642.7 GB Flash
re-capture will be roughly 12x this model's weights to fetch (~17 min at the
measured 627 MB/s) and a similar per-window cost if it stays resident.

With `--replay-device cuda` the same 512-window comparison runs in **173.27 s**,
so the term that dominated this rung stops dominating the next one. What is left
to budget is the **capture** — streaming-regime for a 642.7 GB root, and
fetch-overlapped under layer-outer — and the **candidate count**, which is now
bounded by what the engine can load (learning 8) rather than by what the
comparisons cost. `docs/CAPTURE-SCALING-PLAN.md` §3 carries the revised model
and the M2/M3 projections.
