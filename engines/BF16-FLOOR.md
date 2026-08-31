# The BF16 floor — what a quant's KLD actually costs

**Measured floor: 0.011505922619330299 nats** (mean KLD(teacher / student) where the
"student" is the UNQUANTIZED BF16 model, full 25-window / 51,175-position
sealed panel, fp64, streaming lane, **2 cold runs producing identical means**,
`bitwise_deterministic: true`).

## Why this measurement exists

Our K6 and K8 quants score 0.013715 and 0.012384 on this panel — only **1.11x
apart** — while K8's shipped payload store is **13.2x tighter** than K6's in
weight-space NMSE. Those two facts only reconcile if the KLD we measure is
dominated by something that is NOT quantization error.

It is. Scoring the BF16 weights themselves — no quantization at all — against
the teacher still costs 0.011506 nats on this panel, because the teacher
logits were captured on a different stack (brandonmusic's EP4 runtime) than the
replay lane, and because bf16 arithmetic is not associative across differing
expert-combine orders. That is the FLOOR: the price of the comparison itself.

## Quantization-attributable error

| | panel KLD | minus floor | = attributable |
|---|---:|---:|---:|
| BF16 (floor) | 0.011506 | — | 0 |
| K6 (6 bpw, 254 GB) | 0.013715 | -0.011506 | **0.002209** |
| K8 (8 bpw, 331 GB) | 0.012384 | -0.011506 | **0.000878** |

**K8's quantization error is 2.52x smaller than K6's** — against a raw
panel-mean ratio of only 1.11x. Put the other way: **K8 removes 60% of
the divergence K6 still leaves on the table.** That is the number that belongs
next to "13.2x tighter weights", and it is invisible if you read raw KLD alone.

## How to use this (and how not to)

- The floor is a property of THIS panel + THIS teacher + THIS lane. It is not a
  universal constant. Re-measure it whenever any of those change.
- Subtracting a floor measured on a DIFFERENT lane is invalid. Our official-FP8
  figure (0.020615) was captured cross-stack, and its matching cross-stack floor
  is 0.012712 — so FP8's attributable cost is ~0.0079 against THAT floor, never
  against this one.
- The subtraction is an approximation: KL is not additive, and it is meaningful
  only because both terms are small and share the same reference.
- A quant scoring AT the floor is not "perfect" — it means this panel can no
  longer resolve its error, and a harder panel (or a same-stack teacher) is
  needed to see further.

## Cost, honestly

The measurement cost **$18.90** (1x H200 spot at $1.99/h, 9h21m) — but most of
that was BUILDING the native-BF16 student mode in stream_score.py, not
measuring. A repeat costs ~2 runs x ~3h = ~$12, and a single run ~$6.
Receipts: native-bf16-kld.json (2 cold runs, identical means) and
native-bf16-kld-run1only.json.
