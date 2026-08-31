# Measuring a GGUF is not the same question as measuring an EXL3 quant

`unsloth/GLM-5.3-Flash-GGUF` is, by a wide margin, the largest quant audience
this model has: 45,936 downloads and 290 likes at the revision this document was
written against. Until now the registry could not measure it at all — a
first-time contributor pointing `bin/measure-cloud` at it got

> this artifact cannot be read by any available surface adapter

which was not true. `k6/tools/gguf_surface.py` had been bitwise-proven against
`gguf-py` for months, `stream_score.py --source gguf` existed,
`kld_report.py --profile gguf` existed, and `registry_add.py` already had a
GGUF adapter that refuses a row with no scope census. What was missing was the
lane wiring, so none of it was reachable. A capability nothing can invoke is
indistinguishable from a missing one.

This document is about what a GGUF row *means* once it exists, because the
answer is genuinely different from every other row in the streaming lane, and
the difference is not a caveat you can put in a footnote and then rank across.

## The scope difference, which is the whole thing

Every other third-party artifact this suite measures — turboderp's `exl3hf`
releases, brandonmusic's sealed `tr3-published` release, 0xSero's `dione`
conversion — quantizes the **routed experts** and leaves the rest of the model
alone. Their non-routed tensors are the official release's tensors, byte for
byte or dequantized back to them. When such a row says "0.1216 nats", the
sentence it completes is: *given the reference's embeddings, attention, dense
MLPs and lm_head, replacing the routed experts with this codec at this rate
moves the output distribution this far.*

A GGUF quantizes **everything**. On `UD-Q4_K_XL`, read from the container's own
1,412-tensor table:

| tensor class | what the artifact stores | measured bits/weight |
|---|---|---|
| `embed_tokens` | Q8_0 | 8.5000 |
| `lm_head` | Q8_0 | 8.5000 |
| `attn.qkv` | Q8_0 | 8.5000 |
| `attn.o` | Q8_0 | 8.5000 |
| `mlp.gate` / `mlp.up` / `mlp.down` (dense layers) | Q8_0 | 8.5000 |
| `moe.shared_expert` | Q8_0 | 8.5000 |
| `moe.experts` (routed) | Q4_K ×82, Q5_K ×41, Q6_K ×3 | 4.8745 |
| `moe.router` | F32 | 32 (native) |
| `norm` | F32 | 32 (native) |

So the sentence a GGUF row completes is a different one: *this artifact's own
embeddings, attention, MLPs, experts and head, all of them, move the output
distribution this far from the reference.* Nothing in the measured forward is
the reference's weights except the vision tower, which the panel never
executes.

That is why the comparability question is not "which of these two 4-bit quants
is better". Ranking a GGUF row against a routed-experts-only row at a nominal
4 bpw compares a whole-model quantization against a partial one and reads the
difference as codec quality. The registry refuses to let that happen silently:
`registry_add._apply_gguf_provenance` **requires** the scope census on the
receipt and turns it into a `quantization_scope_whole_model` disclosure, and
refuses the row outright if the census is missing. The comparability key is
computed over `scope_digest`, so a GGUF row and a TR3 row land in different
groups and render in different tables.

If you want the comparison anyway, the honest form of it is not a ranking. It
is: *at ~5 bits/weight overall, a whole-model llama.cpp quantization costs X
nats; at 4 bpw on the routed experts alone, an EXL3 quantization costs Y.*
Those are two different products, and a reader who wants one is usually not
choosing between them — they are choosing between llama.cpp and exllamav3 as
serving stacks, which is a question this lane does not answer at all (see
`llms.txt` Rule 5: this is the *checkpoint* lane; it characterizes weights, not
kernels).

## The rate on the box is not the rate in the file

`UD-Q4_K_XL` measures **4.9806 bits/weight** over every tensor it stores. The
name says 4. Both numbers are on the receipt, and they are different fields for
a reason:

* `bits_per_weight_nominal` = 4.0, read from the build name. This is what a
  reader searching for "the 4-bit one" means.
* `bits_per_weight_effective` = 4.9806, computed from ggml's own block traits —
  elements and bytes — over the container's whole tensor table.

The gap has two independent causes and both matter. First, the non-routed half
of the file is Q8_0, and it is a large half. Second, a K-quant block is wider
than its name: Q8_0 is 34 bytes per 32 weights, i.e. **8.5** bits/weight once
its scale is counted, and Q4_K is 144 bytes per 256 weights, i.e. 4.5.

The practical consequence: **do not read a GGUF build name as a rate.** Which
brings us to the trap that costs money.

## A build name does not tell you what is inside it

unsloth's "UD" (Unsloth Dynamic) recipes mix ggml types across tensor classes.
`UD-Q2_K_XL` contains IQ2_XS, IQ3_XXS and IQ4_XS tensors; `UD-Q3_K_XL` contains
IQ3_XXS and IQ4_XS. `gguf_surface` v1 has no IQ kernels — adding one means
adding it *with* the same bitwise-vs-`gguf-py` proof the K-quant kernels carry,
not skipping the tensors — so both builds are refused.

Neither refusal is predictable from the name. So the planner reads the build's
own headers before renting anything: a GGUF header sits at the front of the
file and `gguf_surface` accepts `https` locations by range request, so
`bin/measure-cloud --dry-run` proves the whole build decodable for a few
hundred kilobytes and $0.00, and refuses by type and by tensor name if it is
not:

```
REFUSE: this GGUF build cannot be decoded by gguf_surface v1
        gguf_surface: REFUSED: 129 tensors use ggml types without a v1 decode
        kernel (IQ2_XS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K), e.g.
        blk.10.ffn_down_exps.weight [IQ3_XXS] ...
```

Of unsloth's twelve builds, v1 scores five: `BF16`, `Q8_0`, `UD-Q4_K_XL`,
`UD-Q5_K_XL`, `UD-Q6_K_XL`.

## A GGUF repo is a shelf, not an artifact

Every other surface here has one artifact per repository. `unsloth/GLM-5.3-Flash-GGUF`
publishes **twelve** at a single commit, 2.55 TB in total, of which one build is
~200 GB. Two consequences the wiring has to carry:

1. **`--path` is required**, and the planner lists the builds rather than
   guessing. `repository + revision` does not identify what was measured here.
   The receipt's `artifact.path` names the build, and the registry schema
   already required exactly that for a repo with more than one catalogued
   artifact.
2. **The identity is the file list.** A community GGUF ships no seal, no
   encoder receipt and no per-file digest manifest, so the fetch stage
   whole-file sha256s every part right after it lands — the only cheap moment to
   hash 200 GB — and the receipt carries name + bytes + sha256 per part.
   `registry_add` refuses a GGUF row whose summary declares `full` verification
   but carries an entry without a digest.

Pricing follows the same rule: the plan sizes the **build**, not the shelf.
Sizing the shelf would refuse, on cost, a run that fits comfortably.

## What `--bf16` is doing in a GGUF run, and what it is not

`--source gguf` still takes `--bf16`, and it means something much narrower than
it does for `exl3hf`/`tr3`/`dione`, where it points at a tree materialized from
the artifact itself. For a GGUF it points at the **official** release, and its
entire job is:

* `config.json` and the tokenizer/processor sidecars — a GGUF carries its own
  tokenizer in llama.cpp's format, not in HF's, and the sealed forward is a
  `transformers` model;
* the **vision tower** (`model.visual.*`, 347 tensors), which the main container
  does not carry at all — llama.cpp ships it as a separate `mmproj` file. On
  GLM-5.3-Flash all 347 live in one shard of 120, so this is ~4.2 GB rather than
  1.4 TB, computed from the index rather than assumed.

No routed expert, no attention projection, no embedding and no head is read from
that tree. The receipt says so (`nonrouted_policy`,
`decoded_from_the_same_gguf_artifact`), and the scope entry for `other` states
the vision tower's absence explicitly rather than leaving a reader to assume it
was quantized like everything else. The text-only sealed panel never executes
the tower, so no vision weight is inside the measured function either way.

The one place the official tree binds the run is identity: `stream_score`'s gate
hashes the `config.json` and index it actually loads against a sealed
`quant-pipeline.glm-release-inventory.v1`. zai publishes no such file and no
materializer produces one here, so the setup stage writes it over the two
official files **at the pinned revision** — which is what makes the gate bind
those bytes to that commit rather than to nothing.

## Two smaller things worth knowing

**Norms come back narrower, not wider.** llama.cpp stores layer norms F32 where
the official release stores them bf16. The materialized view casts them down, so
the constructed model is dtype-identical to a native build. A view that kept
them F32 would be measuring a model slightly *better* than the one anyone runs.
`verify_official_dtypes` reads the real dtypes out of the official safetensors
headers wherever those shards are present and refuses on any disagreement, so
the policy cannot go stale silently.

**Calibration is a comparability fact.** `UD-Q4_K_XL` declares importance-matrix
calibration in its own metadata (`quantize.imatrix.*`: 809 entries, 88 chunks,
`unsloth_calibration_GLM-5.3-Flash.txt`). An imatrix-calibrated build and an
uncalibrated one at the same nominal rate are different artifacts. The
submission schema's `codec` block has no calibration field, so this travels as a
disclosure with its own pinned source — the container's own header at the pinned
revision.

**The source is unsealed.** No community GGUF publishes encoder receipts,
reconstruction closures or a sealed reader ABI. Every GGUF row therefore carries
the `unsealed_source` disclosure, the same treatment the Dione family gets, and
the same one the sealed TR3 family does *not* need.

## What it costs, measured — and why the first attempt did not finish

> **Superseded below.** The 23.7 min/window in this section was real and is kept
> because the diagnosis in it is what led to the fix. The dequant now runs on the
> accelerator and measures **0.98 min/window** on the same box and the same
> artifact, bitwise-identical to the path below. Read this section for *why* it
> was slow; read "The fix" for what it costs today.

The wiring works. The first real run got as far as capturing window 1 of 25 and
was then **stopped deliberately**, because it could not finish inside its
budget. That is a result, not a failure to report, so here is the number and the
reason.

Measured on `unsloth/GLM-5.3-Flash-GGUF` `UD-Q4_K_XL` @ `2975ab41`, one
A100-SXM4-80GB (RunPod secure, $1.59/h), 49 layer-fills spanning window 1 and
the start of window 2:

| quantity | measured |
|---|---|
| cumulative decode | 1656.2 s over 42,336 expert matrices |
| per matrix | **39.12 ms** |
| per window (36,288 matrices) | **23.7 min** of decode alone |
| per cold run (25 windows) | **9.86 h** |
| two cold runs (a submittable receipt needs ≥2) | **19.7 h** |
| peak device memory | 45.7 GB, against 47.07 GB predicted |

For comparison, the same lane measures **3.12 min/window** on `exl3hf` and
**2.82** on `tr3-published`. GGUF is **7.6x** slower per window.

**It is not I/O and it is not the GPU**, and both of those were checked rather
than assumed:

* Window 2 (mean 37.2 s/fill) was **not** faster than window 1 (33.2 s/fill),
  on a host with 1,007 GB of RAM and 919 GB of warm page cache — more than
  enough to hold the whole 185 GB routed payload. So it is not a cold-read
  effect, and no amount of faster storage fixes it.
* `nvidia-smi` read 2–4% utilization with 45.7 GB resident. The process took
  1,055–1,380% CPU on a 128-core host that was 74% idle.

It is the **dequant**. `gguf_surface`'s kernels are deliberately plain,
MPS-safe PyTorch — uint8-level unpack, fp32 accumulation, no float64, no int64
beyond gather indices. That choice is exactly what makes them *bitwise* provable
against `gguf-py` 0.19.0's reference `dequantize()`, which is the property the
whole surface rests on. The same choice costs ~7.5x per matrix against the exl3
path's fused decode (5.2 ms). The proof and the speed are trading against each
other, and v1 chose the proof.

Raw per-fill timings: `k6/tools/gguf-evidence/udq4kxl-decode-timings-a100.jsonl`.

## The fix: decode where the GPU already is

The diagnosis above named the dequant, and it was right. The fix follows from
one observation the diagnosis did not draw out: **the CPU path sends the wrong
thing across the bus.** It decodes a 4.7 MB quantized expert slice into 33.5 MB
of fp32 on the host and then copies *that* to the accelerator — 7.1x more
traffic than the input, to hand over a tensor the accelerator could have
produced itself while it was idle at 2–4%.

`dequant_bytes(..., device=)` moves the raw uint8 buffer first and runs the
kernels there. **Not a rewrite: the same lines, in the same order, on a tensor
that lives somewhere else.** That is what makes it defensible. Every operation
in those kernels is either an integer op on uint8/int8 (shift, mask, or,
reinterpret) or an IEEE 754 binary32 multiply or subtract on an
elementwise-shaped operand. Both classes are exactly specified and
device-independent: no reduction, no matmul (so no TF32 path), and in eager mode
no fusion that could turn the `dd * q - dm` pair into an FMA with a different
rounding.

Measured on the same artifact and the same box class — `UD-Q4_K_XL` @
`2975ab41`, one A100-SXM4-80GB (RunPod secure, $1.59/h), 128 cores, real fused
expert tensors for routed layers 3 and 4 (Q4_K gate/up, Q5_K down), full
288-expert fills:

| | layer 3 | layer 4 | min/window | two cold runs | at $1.59/h |
|---|---|---|---|---|---|
| dequant on **cpu** (the v1 path) | 35.66 ms/matrix | 32.11 ms/matrix | 21.6 / 19.4 | ~18 h | ~$28 |
| dequant on **cuda** | **1.613 ms/matrix** | **1.509 ms/matrix** | **0.98 / 0.91** | **~0.8 h** | **~$1.3** |

**22x**, and the CPU column reproduces the 39.12 ms/matrix of the run that
raised the alarm (this harness runs fills without a forward competing for the
box, which accounts for the difference). Three repeats of the layer-3 fill on
the accelerator path spread 1.718 / 1.913 / 2.021 ms/matrix, so **1.5–2.0
ms/matrix, ~0.9–1.2 min/window** is the honest range rather than the single best
number.

Device memory: a 288-expert accelerator-decoded fill peaked at **19.7 GB** —
the 14.5 GB slab plus ~5.2 GB of decode transients at the default 16 decode
threads. That is new headroom the CPU path did not need, and it is ~5 GB against
the 47.07 GB the fit check already predicts for this lane. The GPU pool sweep
says 8 threads is the knee (1.80 ms/matrix, and less transient memory than 16);
32 is worse on both counts (2.64 ms and 11.7 GB on a 96-expert slab). The
default of 16 is left alone: 16% off the best rate is not worth a second knob.

**The bitwise property held, and that is the acceptance test.** Not "the decoded
fp32 matched" — the whole fill was run twice, once per path, and the two
resulting **BF16 slabs** were compared: 7,247,757,312 bf16 elements per layer,
`torch.equal`, on real UD-Q4_K_XL weights. Those slabs are literally the bytes
the expert forward reads. If every installed weight is bit-identical then the
forward is the same function on the same lane, and the tokenwise KLD tensor it
produces is the same tensor. Harness: `k6/tools/gguf_decode_bench.py --verify`; raw rows, including every sweep below: `k6/tools/gguf-evidence/udq4kxl-decode-device-a100.jsonl`.

The check that guards it from here on is `selftest_gguf_offline.py` rung 1b,
which re-decodes the committed real ranged-fetched bytes on whatever
accelerator the host has and demands `torch.equal` against the CPU output rung 1
proved equal to `gguf-py` 0.19.0. It runs on MPS on a laptop — and, since
`bootstrap_measure.sh` now invokes it, on **CUDA on the rented box, before the
fetch and before any GPU-hour is spent**. A laptop's MPS pass is evidence for
CUDA, not proof of it.

`stream_score --gguf-decode-device {auto,device,cpu}` selects the path; `auto`
(the default) uses the run's device when it is an accelerator. The CPU path is
kept, unchanged, as the reference, and the receipt's backend record carries
`gguf_decode_device` so a reader can see which one produced their number.

### What was refuted along the way, and what the CPU path was actually doing

The commit that measured 23.7 min/window guessed that "the idle 114 cores say
most of the headroom is here". **They do not.** Sweeping the decode pool on the
same box, same layer, same bytes:

| `--decode-threads` | 8 | 16 (the default) | 32 | 64 | 96 |
|---|---|---|---|---|---|
| ms/matrix, cpu path | 35.4 | 35.7 | 41.0 | 54.4 | >1187 † |

† killed after 342 s without finishing 288 matrices, at 12% CPU.

More threads makes it *worse*, catastrophically so at the top. The CPU path is
already at its knee at 8, so "parallelise across the demonstrably idle cores"
was the wrong lever.

The right diagnosis is the opposite one, and it fell out of the same sweep.
`torch.get_num_threads()` on this host is **128**, and `stream_score` runs 16
Python decode threads, each of whose torch ops asks for that 128-wide intra-op
pool: **2,048 threads on 128 cores**, spending most of their time in OMP
barriers. Pinning torch to one intra-op thread, changing nothing else:

| `torch.set_num_threads` | 1 | 4 | 128 (the default) |
|---|---|---|---|
| ms/matrix, cpu path, 16 decode threads | **13.0** | 23.8 | 35.7 |

So the CPU reference path is **2.7x faster** with `OMP_NUM_THREADS=1` in the
environment — 7.9 min/window instead of 21.6 — and the 96-thread collapse above
is the same effect at 12,288 threads. The cores were idle because the work does
not scale onto them; the *time* went to contending for them anyway.

That is documented here rather than hard-coded, deliberately.
`torch.set_num_threads` is global and the model forward shares it, and intra-op
width can move reduction order in CPU kernels — which is exactly the class of
silent change this project refuses to make casually. The dequant itself has no
reduction and could not be affected, but the forward is not this file's to
bargain with. **If you are running the CPU decode path, export
`OMP_NUM_THREADS=1`;** on the accelerator path it is moot.

### What is still on the table

1. `--decode-cache ram` — the routed view is re-decoded every window, and the
   decoded slabs are 14.5 GB/layer, 609 GB for all 42. On a box with the RAM
   (this one had 2 TB) that turns 25 decodes of each layer into one. It is a
   lane knob, not new code, and it is now the *largest* remaining lever because
   the dequant is no longer the dominant term.
2. Batching several experts per dequant call. Worth much less than it was:
   1.6 ms for 8.4M elements is kernel-bound, not launch-bound.
3. A faster GPU now matters where it did not before.

### What the planner does with this

`engines.json` `minutes_per_window_by_surface.gguf` moves from **23.7 to 3.19**,
and the second number is deliberately *not* the measured 0.98. Decode-only was
an honest total only while decode was 7.6x everything else on the lane; it is
not any more, so the per-window remainder — the forward, the logit save — now
dominates and has never been measured for this surface. Rather than invent it,
the planner prices GGUF at the **slowest measured per-window total on this
lane** (dione's 3.19). GGUF's measured decode is now below the decode implied by
every other surface here, so its total cannot reasonably exceed the lane's
slowest. It is a placeholder with a floor under it, not a GGUF measurement, and
it says so in `engines.json`; the first GGUF capture that *finishes* must
replace it with its own `elapsed_seconds / 25`.

At 3.19 min/window a submittable two-cold-run GGUF receipt prices at ~2.7 h and
~$4.20 on this A100 — inside the $6–15 band a routed-experts-only row costs,
where before it was ~20 h and ~$32.

## Where the wiring lives

Adding a surface means several files agreeing, and the refusal text in
`bin/measure_cloud.py` names them. For GGUF specifically:

| file | what it holds |
|---|---|
| `k6/tools/gguf_surface.py` | the reader, the dequant kernels, and `scope` — the per-class recipe measured from the container's own table |
| `k6/tools/gguf_decode_bench.py` | the fill-rate harness and the `--verify` bitwise acceptance test between the two decode paths |
| `k6/tools/stream_score.py` | `--source gguf` / `--profile gguf`, and the view materialization |
| `k6/tools/k6_kld_report.py` | profile `gguf` → student label `gguf-llamacpp` (format-wide, not per-rate) |
| `bin/fidelity/hfmeta.py` | the shelf: build grouping, `--path` selection, nominal rate from the name |
| `bin/engines.json` | `surfaces` + `profile_map_by_surface["gguf"] = {"*": "gguf"}` |
| `bin/invoke_engine.py` | the on-instance argv: every part, the inventory, the official skeleton |
| `bin/stage_measure.sh` | build-scoped fetch, whole-file hashing, the scope pass, the inventory |
| `bin/seal_receipt.py` | container `gguf`, `path`, `llama.cpp` as the quantizer, the effective rate |
| `registry/tools/registry_add.py` | the GGUF summary family and its mandatory scope disclosure |

The `"*"` key in `profile_map_by_surface` is the one deliberate shape difference
from the EXL3 families. `EXL3HF_PROFILES` / `TR3_PROFILES` / `DIONE_PROFILES`
are `{profile: (declared bpw, student label)}` tables because each of those
releases pins one rate and the engine cross-checks it against the release's own
declaration. A GGUF makes no such declaration and has no single rate; one
reader, one receipt family and one format-wide student label serve every build.
Keying the profile on bits would have meant twelve identical entries claiming
twelve different receipt families.

Coverage: `bin/selftest_gguf_lane.py` (T13) walks shelf → plan → argv → fetch →
sealed receipt, and `k6/tools/selftest_gguf_offline.py` rung 7c re-derives the
per-class scope from the real 1,412-tensor table and checks it against the
committed fixture, so the fixture cannot drift from the artifact.
