# K8 fidelity anomaly — resolved

**Date:** 2026-08-28 · **Model:** GLM-5.3-Flash · **Status:** resolved — no defect

## Summary

Our 8-bit quant appeared to measure *worse* than our 6-bit quant. It does not.

The comparison was made on **one window of a 25-window panel**, and per-window mean KL
divergence is heavy-tailed enough that a single window cannot distinguish the two rates.
Across all 11 windows both runs have now captured, **K8 wins 9 of 11** and is better on
pooled mean, pooled median, and top-1 agreement.

| | K6 (6-bit) | K8 (8-bit) | |
|---|---:|---:|---|
| pooled mean KLD (11 windows, 22 517 positions) | 0.013873 | **0.012655** | K8 better by 1.22e-3 |
| pooled median KLD | 0.001846 | **0.001731** | K8 better |
| mean top-1 agreement vs teacher | 96.12 % | **96.34 %** | K8 better |
| windows won | 2 | **9** | |
| shipped-store weight NMSE | 4.62e-4 | **3.50e-5** | K8 **13.2× tighter** |

The K8 checkpoint is **not defective**, the scorer is **not at fault**, and **no re-encode is
needed**. The single adverse window was an unlucky draw.

## Why one window was not enough

Per-window deltas have a standard deviation of **1.73e-3** against a true effect of
**1.22e-3**. The noise is larger than the signal, so a single window has essentially no
power to separate 6-bit from 8-bit.

The reason is that per-position KLD is extremely heavy-tailed. Window medians across the
panel range from 1.5e-5 to 1.2e-2, and single positions reach a KLD of 2.1 — so a window's
*mean* is set by a handful of positions.

On the window where the alarm was raised, the 8 worst positions accounted for **122.7 %** of
the entire K8 excess: remove them and K8 wins that window too. Even on that adverse window,
K8 was **better** at the 95th and 99th percentiles (0.0468 vs 0.0522, and 0.1622 vs 0.2137).

Per-window results:

| window | K6 | K8 | delta | |
|---|---:|---:|---:|---|
| 0000 | 0.016829 | 0.018200 | +0.001371 | K8 loses ← *the reported anomaly* |
| 0001 | 0.018520 | 0.015204 | −0.003315 | K8 wins |
| 0002 | 0.012966 | 0.010361 | −0.002605 | K8 wins |
| 0003 | 0.014703 | 0.011964 | −0.002739 | K8 wins |
| 0004 | 0.023965 | 0.020087 | −0.003878 | K8 wins |
| 0005 | 0.016333 | 0.016053 | −0.000280 | K8 wins |
| 0006 | 0.015793 | 0.014276 | −0.001517 | K8 wins |
| 0007 | 0.007056 | 0.006850 | −0.000205 | K8 wins |
| 0008 | 0.004778 | 0.004006 | −0.000772 | K8 wins |
| 0009 | 0.006585 | 0.006155 | −0.000431 | K8 wins |
| 0010 | 0.015075 | 0.016052 | +0.000977 | K8 loses |

Sanity check: our 11-window pooled K6 mean (0.013873) sits within 1.5e-4 of the sealed
25-window K6 number (0.013723), so the subsample is representative.

## The shipped artifact is sound

Measuring the **actual shipped payload stores** against BF16 — 10 experts, 30 matrices,
layers 8–39, all three projections:

| | K6 | K8 |
|---|---:|---:|
| mean relative Frobenius error | 0.021490 | **0.005916** |
| mean NMSE | 4.624e-4 | **3.505e-5** |
| better in | — | **30 of 30 matrices** |

The shipped K8 store is **13.2× tighter in NMSE** than K6. Eight bits delivered exactly what
it should.

## The permutation that made this hard to see

The weight-space audit was initially blocked: decoded payloads showed *zero* correlation with
the BF16 tensors their own provenance named, with `cos(|Ŵ|,|W|) = 0.6328` — suspiciously
close to `2/π = 0.6366`, the value for two *independent* Gaussians.

That signature is a **permutation**, not noise, and it was: the campaign encodes each expert
in a permuted **intermediate-channel** frame. Permuting `gate_proj`/`up_proj` output rows and
`down_proj` input columns by the same permutation leaves the expert's function unchanged, so
serving is unaffected — which is why both quants work perfectly despite the audit appearing
to show noise.

Evidence:

- **Sorted magnitude spectra match to ~1.1 %** while unsorted cosine is ~0.002 — identical
  value multisets, scrambled positions.
- Matching decoded rows to BF16 rows recovers a **perfect bijection** (2048/2048 unique
  targets, mean cosine 0.9998, **zero** identity matches).
- Unpermuting gives rel err 0.0203 / 0.0202 / 0.0199 for gate/up/down at K6 — precisely the
  expected 6-bit quantization level.
- The permutation recovered from `gate_proj`, applied unchanged to `up_proj` and `down_proj`,
  reproduces the expected error — confirming the symmetry is applied consistently within each
  expert.
- `perm(K6) == perm(K8)` for **10 of 10** experts, consistent with the campaigns sharing one
  transform seed.

**Any future weight-space audit against BF16 must undo this permutation**, or the store will
look like noise. This should be documented in the campaign notes.

## What was ruled out along the way

| Hypothesis | Status | Evidence |
|---|---|---|
| Reader's K8 decode path is wrong *(prime suspect)* | **Eliminated** | Bitwise identical to exllamav3's independent native CUDA kernel, on 6 K8 payloads + 6 K6 controls, both pre- and post-Hadamard. Max abs diff `0.0`. |
| suh/svh sign or scale convention mismatch | **Eliminated** | Decode is the exact algebraic inverse of the encoder; round-trip error 4.2e-7; the reader's Hadamard is bitwise identical to exllamav3's. |
| The codec can't do 8 bits | **Eliminated** | Fresh encode of real experts with real Hessians: K8 NMSE 8.45e-5 vs K6 5.17e-4. |
| Corrupt or pathological Hessian | **Eliminated** | Positive-definite, finite, well-conditioned. |
| K8 got an untuned profile K6 didn't | **Eliminated** | Byte-identical profile settings in both campaigns. |
| Measurement-lane artifact | **Eliminated** | Run plans identical in every measurement-relevant field. |
| Degraded non-routed / native scope | **Eliminated** | Receipts match field for field. |
| Shipped K8 store under-performs its rate | **Eliminated** | 13.2× tighter than K6. |
| Downstream surface-assembly fault | **Eliminated** | No fault to explain — K8 wins the panel. |
| **Underpowered single-window comparison** | **CONFIRMED** | This is the cause. |

The prime suspect deserves a note: K8 support in the packed reader (128-word trellis tiles)
came from our own patch and had never been cross-validated — only K4 and K6 had. It was the
right thing to test first, and it is correct.

## Recommendation

**Publish the K8 artifact normally.** It is a genuine improvement over K6 — tighter weights
and a lower panel KLD with higher top-1 agreement.

1. **Never quote a single-window KLD as a rate comparison.** The per-window noise (sd 1.73e-3)
   exceeds the effect (1.22e-3). Quote the sealed full-panel number.
2. **Let the in-flight K8 run finish all 25 windows** and compare against the sealed 25-window
   K6 figure (0.0137234) for the published number.
3. **Report a trimmed or median KLD alongside the mean.** Per-position KLD is heavy-tailed
   enough that a few positions dominate any window mean.
4. **Document the intermediate-channel permutation** so the next weight-space audit doesn't
   lose a day to it.

---

*All work was done in an isolated workspace at `nice -n 19` alongside two live measurement
runs; no processes were interrupted and no campaign artifact was modified. Full numbers in
`K8-ANOMALY.json`.*
