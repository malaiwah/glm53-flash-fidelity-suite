# The bf16 logit-rounding term, measured on the real GLM-5.3 root

Answers review-science S2-1's open question 4: how large is the term a bf16
serving stack adds by rounding every logit to bf16, which the hidden-form rows
(fp32 logits replayed from sealed bf16 hidden states) do not contain?

Everything here is spend-free and reads sealed datasets as published:

| side | dataset | `dataset_sha256` |
|---|---|---|
| root | `malaiwah/glm53-fidelity-root-v1` (`fidelity--glm53.malaiwah.root.bf16`) | `6b8d3a7b…` |
| K4 | `fidelity--glm53.malaiwah.quant.exl3-wrld-k4` (wrldsuksgo2mars K4) | in `result.json` |
| FP8 | `fidelity--glm53.malaiwah.quant.fp8` (zai-org FP8, dequantized) | in `result.json` |

`measure.py` uses the comparator's own replay (`dscompare._replay`, numpy fp32,
vocab chunk 8192) and its own fp64 estimator (`dscompare.token_kld`), then
rounds the replayed fp32 logits to bf16 with round-to-nearest-even and scores
again. One window, `final-0000`, 2,047 positions, workstation CPU
(Xeon X5570, scipy-openblas 0.3.34; the `replay_env` block in `result.json`).

## Result (`result.json`)

| quantity | value |
|---|---|
| max \|logit\| on the window | 46.36 |
| max \|fp32 − bf16(fp32)\| over all logits | 0.1247 (one bf16 ULP at [32, 64)) |
| one-sided KL(fp32 ‖ bf16-rounded), root alone | **1.73e-5 nats** |
| K4 row: KL(ref ‖ cand) fp32 → both sides bf16-rounded | 0.030082 → 0.029956, **Δ −1.26e-4 nats (−0.42 %)** |
| FP8 row: same | 0.012478 → 0.012451, **Δ −2.69e-5 nats (−0.22 %)** |

Reading: a logit-form row captured from a bf16 stack would carry a term of
the 1e-5–1e-4 nats class relative to the hidden-form rows — under 1 % of every
GLM-5.3 row, and negative on both real comparisons here (rounding both sides
slightly *shrinks* the divergence). The review's synthetic estimate
(2.6e-4 one-sided, +4.7 % two-sided) overstated it for real GLM-5.3 logit
magnitudes. One window is a magnitude measurement, not a panel statistic;
`--windows all` on the same script produces the 25-window version in about
three hours on this box.

```
python3 reports/bf16-logit-rounding/measure.py \
    --root <sealed root dataset> --windows 0 \
    --candidate k4=<sealed K4 dataset> --candidate fp8=<sealed FP8 dataset> \
    --out reports/bf16-logit-rounding/result.json
```

Without `--root` the script fetches the needed files (one window ≈ 25 MB, the
head 1.9 GB) from the published root with `huggingface_hub`.
