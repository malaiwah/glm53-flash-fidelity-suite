# GLM-5.3 (full) quantized descendants — what the engine side needs

**Status:** gap analysis, 2026-09-04. Nothing below is implemented yet
except where a file is named. Numbers come from the Hub API and the
checkpoint bytes (index censuses, `quantization_config`, shard headers), not
from repo names.

## The one fact that shapes everything

`engines/tools/stream_score.py` — the single-device streaming scorer behind
every quantized row in the registry (K6/K8, exl3hf, dione, gguf, mlx, nvfp4,
the BF16 floor) — is bound to **`Glm5NextForConditionalGeneration`**, i.e.
GLM-5.3-**Flash** (`RELEASED_ARCHITECTURE`, its `quant_pipeline` readers,
its EP8 emulation). The full model is **`GlmMoeDsaForCausalLM`** (78 layers,
256 routed experts, MLA + DSA indexer, one MTP block). Today only the root
engine — `engines/tools/hf_capture.py --schedule layer-outer` streaming
transformers' own modeling one decoder layer at a time — runs it, and that
engine refuses any checkpoint with a `quantization_config`
(`layer_outer.py`, "the layer-outer schedule does not build a quantizer"),
for the right reason: plain FP8 would load as bf16 with its block scale never
applied, silently.

So there is **no engine for any quantized descendant of the full GLM-5.3**,
and the missing piece is one thing, not six: a **decode-to-bf16 weight
source for the layer-outer streamer**. `build_streamed_model` feeds each
layer's tensors to `transformers`' own `convert_and_load` as a
`{key: tensor-or-slice}` subset (`layer_outer.py::do_load`). A weight source
that produces that subset from a quantized surface — bf16 tensors decoded on
the fly, per layer, on the host — leaves the forward pass, the panel, the
capture format, the seal and the comparison exactly as they are for the root.
"Dequantize-and-run, weights-only": the same method M1 used for Qwen3.8 FP8
(`docs/M1-QWEN38-ROOT-LEARNINGS.md`), now under the streaming schedule.

Then the comparison is the dataset route that already exists:
`hf_capture --role quant` writes a hidden-form dataset of the student;
`fidelity-dataset compare --reference <root dataset> --candidate <student>`
computes the full-vocabulary fp64 KLD(root ‖ student) and seals a comparison
receipt; `registry-submit` validates it. Same stack as the root, same
schedule, same H200: a **same-stack** comparison, the strongest kind the
registry knows.

## Per surface

| surface | decode reference (must be bitwise) | decoder we have | gap |
|---|---|---|---|
| **FP8 e4m3, block 128×128** (`zai-org/GLM-5.3`, `weight_scale_inv`) | `transformers` 5.16.1 `FineGrainedFP8HfQuantizer` dequantize | none | dequant hook + parity on real shards; `modules_to_not_convert` handled by the checkpoint carrying those tensors unquantized |
| **exl3 fused experts, `mul1` codebook** (drowzeys `keys-GLM-5.3-EXL3` 3.00 bpw; Blackfrost 2.04) | exllamav3 v1.4.4 `codebook.cuh` cb==2 | `exl3hf_surface.py` (`mul1_lut`, parity on Flash tensors) | route decoded tensors into `GlmMoeDsa` names; parity re-run on the full model's tensors |
| **exl3 fused experts, `mcg` codebook + kept FP8 scales** (wrldsuksgo2mars K4: 1,444 `weight_scale_inv`) | exllamav3 `mcg` + transformers FP8 | `exl3hf_surface.py` (`mcg_lut`) | both hooks at once: FP8 dequant for the non-routed tensors it kept in FP8, trellis for routed |
| **TR3 per-expert atoms, mixed K3/K4** (davidsyoung 3.0 / 3.25 / 3.42 bpw: 233,472 × `mcg/suh/svh/trellis`, 1,119 bf16 non-routed) | exllamav3 `mcg`; the per-expert layout `tr3_surface.py` already reads | `tr3_surface.py` composing `exl3hf_surface.py` | decoded per-expert `gate/up/down` tensors go through transformers' own expert-fusing converter (the loader already fuses per-expert bf16 sources); parity on the full model's tensors; a K3 (3-bit) atom width must be confirmed against the decoder's K-range |

Per-artifact facts (pinned revisions in
`~/fidelity-scratch/glm53-exl3-candidates.json`):

| artifact | parent | bpw | size | note |
|---|---|---:|---:|---|
| `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw@6136d7ac` | `zai-org/GLM-5.3` (FP8) | 3.42 | 355 GB | pure quant of the FP8 release |
| `davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw@7c899410` | `zai-org/GLM-5.3` (FP8) | 3.25 | 340 GB | README reports **0.0240** KLD — the number to measure first |
| `davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw@ffb7770e` | `zai-org/GLM-5.3` (FP8) | 3.0 | 317 GB | pure |
| `drowzeys/keys-GLM-5.3-EXL3@c4f5da04` | `zai-org/GLM-5.3` (FP8) | 3.00 | 330 GB | pure; head 16-bit |
| `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1@dd270970` | `zai-org/GLM-5.3` (FP8) | 4.0 | 394 GB | pure but mixed surface (routed trellis, rest FP8) |
| `brandonmusic/GLM-5.3-EXL3-TR3-3bpw@4576d601` | `zai-org/GLM-5.3-BF16` | 3.0 | 316 GB | DRAFT, no `config.json` — not measurable as shipped |
| `Blackfrost-Research/GLM-5.3-DERISKED-EXL3-2.0bpw-HQ`, `drowzeys/keys-GLM-5.3-EXL3-Abliterated` | derivatives | — | — | not quants of the root; out of scope |

**Every serving-ready Trellis quant of the full model descends from the FP8
release**, not from BF16. Measured against the BF16 root they carry the FP8
step inside their number; the FP8 measurement puts a number on that step and
is therefore the first quant to run.

## Order of work

1. **FP8 dequant weight source** in `layer_outer.build_streamed_model`
   (block-scale dequant to bf16 on the host, per layer), refused unless the
   checkpoint's `quantization_config` is exactly the FineGrainedFP8 form.
   Parity: real `zai-org/GLM-5.3` shards through transformers' own
   dequantize versus ours, bitwise, committed as evidence. Offline selftest
   on a synthetic block-FP8 fixture derived from Fruit.
2. **The quant dataset route in the controller**: `--role quant` with the
   `hf-transformers` engine reuses the root stages (setup, fetch, capture
   once, verify) plus a `fetch_reference` of the published root dataset and a
   `compare_root`-shaped comparison stage; the comparison receipt is the
   deliverable the registry ingests. Profiles and timing rows per artifact,
   admitted by the same gates the root passes.
3. **Trellis weight source**: `exl3hf_surface` / `tr3_surface` decode under
   the streamer, routed into `GlmMoeDsa` parameter names; parity on the full
   model's real tensors (both codebooks); profiles for 3.0 / 3.25 / 3.42 /
   3.00 / 4.0 bpw.
4. Measurements, cheapest evidence first: FP8 (~750 GB fetch ≈ 45 min at
   the measured 274 MB/s, one streamed pass ≈ 1 h on NVMe), then
   davidsyoung 3.25 (the one with a published number), then the rest as
   budget allows — each 200–400 GB, ≈ $15–20 on an on-demand H200.

Each row is a **same-stack** number against `fidelity--glm53.malaiwah.root.bf16`
and does not retroactively touch any row measured against another teacher.
