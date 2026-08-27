# GLM-5.3-Flash fidelity suite & quantization program

Tools, receipts, and campaign log for measuring — and then beating — the
quality cost of quantizing [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(321B-total / A18B MoE, `glm5_next` hybrid KDA/DSA architecture with mHC
hyper-connections). Everything here was produced within ~48h of the model's
release and is receipt-driven: every published number links to a JSON receipt
with pinned revisions and sha256s.

## Headline results

| Measurement | Value | Where |
|---|---|---|
| Official FP8 vs BF16, mean KLD (10.48M positions) | **0.028104 nats** (CI95 [0.0272, 0.0290], top-1 94.3%) | [fidelity dataset](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1) |
| Official FP8 on brandonmusic's sealed 25-window panel | **0.020615 nats** / top-1 95.6% | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/fp8-on-brandon-panel.json) |
| Cross-stack BF16 floor (our replay vs his teacher) | 0.012712 nats | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/crosscheck-brandonmusic.json) |
| glm5_next launch nondeterminism (first report + interventions) | pins → ~10× flip-rate reduction | [vLLM PR #53906 comments](https://github.com/vllm-project/vllm/pull/53906), `reports/determinism-*.json` |

## What's in this repo

| Path | What it is |
|---|---|
| [`tools/`](tools/) | The fidelity harness (vLLM hidden-state capture → shared-head replay → exact full-vocab KL), activation capture, cross-stack checker, publishers |
| [`remote/`](remote/) | The self-driving on-VM pipeline + stage scripts used for the overnight 8×H200 capture campaign |
| [`k6/`](k6/) | **The K6/K6K8 EXL3 quantization program** (in progress): runbook, stage driver, patch series onto [brandonmusic's pipeline](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw), driver tools, recipes, and the disclosed [r10 codec reconstruction](k6/fallback/) |
| [`port/`](port/) | Design bundle for a native exllamav3 `glm5_next` architecture port (blueprint, draft, parity harness, adversarial review) |
| [`suite/`](suite/), [`calsuite/`](calsuite/) | The held-out evaluation suite (5,120×2,048 ctx) and calibration token sets |
| [`JOURNAL.md`](JOURNAL.md) | The captain's log: every decision, failure, cost, and 24 lessons learned |

## Published datasets

- [malaiwah/GLM-5.3-Flash-fidelity-suite-v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1) — the quality reference: BF16 + FP8-as-served hidden states over 10.48M positions, shared lm_head, all receipts.
- [malaiwah/GLM-5.3-Flash-calibration-activations-v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-calibration-activations-v1) — 147 GB of MoE block-input activations + router logits (natural routing), for calibration-aware quantization work.

## Credits & lineage

Methodology descends from the author's Qwen3.8-27B fidelity/quant work
([malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3)).
The K6 program builds directly on
[brandonmusic](https://huggingface.co/brandonmusic)'s GLM-5.3-Flash EXL3
pipeline and BF16 teacher-logits dataset — see the
[co-credited corroboration thread](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw/discussions/1).
Base model by [Z.ai](https://huggingface.co/zai-org); quant format by
[turboderp's exllamav3](https://github.com/turboderp-org/exllamav3).
