# Measuring a GGUF is not the same question as measuring an EXL3 quant

`unsloth/GLM-5.3-Flash-GGUF` is, by a wide margin, the largest quant audience
this model has: 45,936 downloads and 290 likes at the revision this document was
written against. Until now the registry could not measure it at all — a
first-time contributor pointing `bin/measure-cloud` at it got

> this artifact cannot be read by any available surface adapter

which was not true. `k6/tools/gguf_surface.py` had been bitwise-proven against
`gguf-py` for months, `stream_score.py --source gguf` existed,
`k6_kld_report.py --profile gguf` existed, and `registry_add.py` already had a
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

So the honest planning number is now in `engines.json`
(`minutes_per_window_by_surface.gguf = 23.7`), which means `measure-cloud` will
price a GGUF run at ~20 h and refuse a shorter `--max-runtime` rather than
discovering it at hour nine. Raw per-fill timings:
`k6/tools/gguf-evidence/udq4kxl-decode-timings-a100.jsonl`.

**What would make a GGUF row affordable**, in the order a future run should try
them:

1. A batched/vectorised dequant for Q4_K/Q5_K/Q6_K/Q8_0 that decodes many
   matrices per call instead of one, keeping the bitwise-vs-`gguf-py` proof as
   the acceptance test. The idle 114 cores say most of the headroom is here.
2. `--decode-cache disk`/`ram` — the view is re-decoded every window today, and
   the box had 900 GB of free RAM. This is a lane knob, not new code.
3. A faster GPU changes almost nothing while utilization is 3%.

Until one of those lands, a GGUF measurement is ~20 GPU-hours (~$32 on an A100
at $1.59/h), which is more than the ~$6–15 a routed-experts-only row costs and
should be budgeted deliberately rather than discovered.

## Where the wiring lives

Adding a surface means several files agreeing, and the refusal text in
`bin/measure_cloud.py` names them. For GGUF specifically:

| file | what it holds |
|---|---|
| `k6/tools/gguf_surface.py` | the reader, the dequant kernels, and `scope` — the per-class recipe measured from the container's own table |
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
