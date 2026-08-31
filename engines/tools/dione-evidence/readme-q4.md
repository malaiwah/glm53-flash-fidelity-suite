---
license: mit
base_model:
- zai-org/GLM-5.3-Flash-BF16
datasets:
- Salesforce/wikitext
tags:
- exl3
- glm
- mixture-of-experts
- moe
- text-generation
- dione
pipeline_tag: text-generation
---

# GLM-5.3-Flash-EXL3-Q4

> [!WARNING]
> **Not yet run as a full server.** The EXL3 kernel path, tensor-parallel CUDA-graph primitive, and held-out BF16-vs-Q4 quality checks passed. A complete serving integration, endpoint health check, model listing, and generated-response test have **not** been run. Vision and MTP execution are also not validated for this release.

This is a selective **4.0 bpw EXL3** conversion of [Z.AI's GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), pinned to source revision `a6c167b62691b2bac901344b65cb651a70f53e43`.

It is deliberately **not an all-Q4 checkpoint**. Only the routed-expert gate/up/down projections in layers 3–44 are EXL3 Q4. The information-carrying backbone remains at source precision: attention, linear-attention/indexers/mHC, routers and correction biases, shared experts, dense layers 0–2, embeddings, LM head, norms, vision, and MTP.

## Status at a glance

| Claim | Status |
| --- | --- |
| Source downloaded and structurally verified | Complete |
| EXL3 Q4 tensors encoded and independently checked | Complete |
| Artifact assembled | Complete |
| CUDA-graph EXL3 primitive on TP=4 | Validated |
| Held-out BF16-versus-Q4 quality | Validated |
| Full server / endpoint / generation | **Not yet run** |
| Vision and MTP execution | **Not yet run** |

## Quantization layout

The source has 45 language layers: 3 dense layers followed by 42 routed-MoE layers. Each routed layer has 288 experts with top-8 routing and one shared expert.

| Component | Precision | Scope |
| --- | --- | --- |
| Routed expert gate/up/down | EXL3 Q4 | 42 layers × 288 experts × 3 tensors = 580,608 tensors |
| Backbone and shared path | Source BF16 | Attention, linear-attention/indexers/mHC, routers, shared experts, dense layers, embeddings, head, norms, vision, and MTP |

The final artifact contains 583,090 indexed tensors and is 187.45 GB on disk. Quantized expert tensors account for 153.54 GB; 2,482 retained source-precision tensors account for 33.84 GB. The layout is a custom `glm53-selective-exl3-tp4-v1` checkpoint and requires a compatible GLM-5.3 selective-EXL3 loader. Do not assume a stock Transformers loader or a generic EXL3 runtime will load it.

## Dione conversion workflow

This release was produced through the **Dione** conversion workflow: a fail-closed selective-precision map, source-parity pilot, natural-route coverage gate, bounded Hessian/K4 packing pilot, full EXL3 encoding, assembly, and independent quality checks. Dione is credited here as the conversion workflow; it is not a base-model author, training-data source, or serving runtime.

The conversion used [ExLlamaV3](https://github.com/turboderp-org/exllamav3) at commit `5f3c537ca9d89893d771256f5c43c93656553fbb` for the EXL3 path. The release records a 4.0-bpw K4 expert representation with tensor-parallel size 4.

## Calibration and routing coverage

Calibration used 600 sealed rows × 2,048 tokens = **1,228,800 tokens**. It used the pinned ExLlamaV3 standard-calibration bundle, with rows labelled `c4` (102), `code` (172), `multilingual` (33), `technical` (35), `wiki` (144), and `tiny` (22), plus 92 synthetic random-token rows. Corpus text and calibration rows are not included in this repository.

The calibration was a conversion aid only; this release does **not** train or fine-tune the base model. Natural top-8 routing covered every routed expert: 412,876,800 total routes, zero experts with zero hits, and a minimum of 1,655 natural routes for any expert (above the 1,024-route floor).

The pinned calibration row digest is `1cae9bbcd2beb3879a0c459edfca1fd197043ab204b82189c9361de386d0cae1`; the calibration-manifest digest is `179db5d74b865df11734c7ab76cdf1fa68818e6135a1cbbc3104c75cc3df230f`.

## Hardware used

Conversion and validation ran on a local workstation with **4× NVIDIA GeForce RTX 3090 GPUs (24 GB each; 96 GB total VRAM)** and **512 GB DDR4 system memory**. No hostnames, usernames, network details, or internal paths are included in this release.

## Validation

Held-out evaluation used the [Salesforce Wikitext](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-2-raw-v1` test split at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. It scored 65,504 next-token positions in 32 contiguous 2,048-token blocks that were disjoint from calibration rows.

| Metric | BF16 source | EXL3 Q4 | Gate | Result |
| --- | ---: | ---: | ---: | --- |
| Cross-entropy | 1.16296 | 1.18648 | — | — |
| Perplexity | 3.19940 | 3.27554 | Δ ≤ 5% | +2.38% |
| Forward KL, BF16 → Q4 | — | 0.06579 | ≤ 0.15 | Pass |
| Top-1 agreement | — | 91.70% | ≥ 80% | Pass |

The runtime primitive check exercised real EXL3 gate/up/down kernels on all four tensor-parallel ranks with CUDA graph capture and replay. Each rank completed graph replay with zero replay-vs-eager relative L2 difference in that scoped test. This is deliberately narrower than a complete server claim; see the warning at the top.

## Attributions and licenses

- **Z.AI** — base model: [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), revision above. This derivative follows the source **MIT License**; the source `LICENSE` is included here.
- **Dione** — selective EXL3 conversion and validation workflow for this release.
- **ExLlamaV3 / TurboDerp** — EXL3 format and conversion implementation used for this release.
- **Salesforce Research** — held-out quality evaluation via the Wikitext dataset identified above. It was used only for evaluation, not training.
- **ExLlamaV3 standard-calibration bundle** — calibration source categories listed above. The release exposes counts and digests, not corpus material; use the upstream project for its source and licensing context.

## Files and integrity

- `layers/` — Q4 routed-expert EXL3 tensors and per-shard metadata.
- `retained/` — source-precision backbone tensors and manifest.
- `model.safetensors.index.json` — maps all 583,090 tensors to files.
- `exl3-manifest.json` — conversion layout and artifact identity.
- `evidence/` and `validation/` — public-safe calibration, routing, runtime, and quality records.

Artifact manifest SHA-256: `6887012fa7ffee2e5ac5d533c3081abd9df0a9b9163fc6ed1fa983b94584d38b`.

## Intended use

Use this release only with a loader that understands the selective GLM-5.3 EXL3 TP=4 layout and preserves the listed source-precision modules. Treat it as a validated artifact with an unvalidated full serving path until an end-to-end server acceptance run is published.
