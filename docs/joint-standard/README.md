# `docs/joint-standard/` — emitted analyses

Everything here was produced by `bin/joint-standard` from the per-window means in
`registry/protocol/per-window/`, offline, on this laptop. No GPU, no
re-measurement, no new number: each file is a deterministic function of data this
repository already publishes.

Every file carries `not_submittable: true` and the frozen protocol's two hashes.
These are **analysis receipts**, not measurements — a measurement row still has to
come through `registry/tools/registry_add.py`.

```
analysis/<series>.panel.json        the full 25-window sealed panel
analysis/<series>.selected.json     the 17-window calibration-clean scope
analysis/paired.<A>-vs-<B>.<scope>.json    paired per-window ranking with BCa
```

Each analysis carries: the window-clustered SE, the naive SE and the design
effect where the receipt has per-window `std`; percentile and BCa bootstrap
intervals (B=5000, seed 20260829); the per-domain table with its own intervals;
`sigma_run` and the quadrature; the percentile-exceedance guard; and the refusal
that pooled token percentiles are not derivable from per-window summaries.

## What is deliberately absent

**`paired.K6-vs-BF16floor.*`** — K6 is `same_stack` and the cross-stack BF16
replay floor is `cross_stack`. Pairing them is a cross-lane floor subtraction,
which registry rule BIAS-006 refuses, so the file is not published even though
the tool will happily compute it. The same-lane K6 floor is the *streaming* BF16
floor, whose receipt is scalar-only and therefore cannot be re-read on any scope
other than the full panel.

`paired.FP8-vs-BF16floor.*` **is** published: both sides are `cross_stack`, so
that difference is same-lane and legitimate. It is the attributable-error result
in §4.5 of [`../PROTOCOL-ALIGNMENT.md`](../PROTOCOL-ALIGNMENT.md).

## Reproducing

```bash
bin/joint-standard analyze \
    --report registry/protocol/per-window/k6-sealed.json \
    --scope-file registry/protocol/window-selection.brandonmusic-final25.json \
    --scope selected --oracle
```

`--oracle` additionally runs the same bootstrap through brandonmusic's own
`kld_eval.analysis.stats` when it is importable, and records the agreement
(observed: 7e-18).
