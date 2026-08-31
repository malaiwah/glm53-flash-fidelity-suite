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
intervals (B=5000, seed 20260829); the per-domain table, whose intervals are
Student-t on `log(mean)` rather than BCa because BCa measures 81.3% coverage at
the 5-7 windows a stratum has (see `docs/PUBLISHED-CORRECTIONS.md` §3);
`sigma_run` and the quadrature; the percentile-exceedance guard; and the refusal
that pooled token percentiles are not derivable from per-window summaries.

## The independent unit is the source document, not the window (P1-15/P1-16)

Added 2026-08-31, after an independent peer review. The sealed 25-window panel
derives from **four source documents** — one per axis, 7/6/6/6 windows — and
clean17 from **three** (7/5/5). Windows cut from one document share its topic,
style and register; treating 25 of them as independent observations
pseudoreplicates. Every `paired.*` receipt therefore now carries:

* **`document_level`** — the contrast recomputed at the document unit. For
  K6-vs-K8 the four document means are all positive (ordering survives), and
  the exact sign test is **p = 0.125** (full) / **0.25** (clean17), not the
  window-level 0.0041 / 0.049. The window-level mean, BCa interval and sign
  test remain in the receipt as **descriptions of this fixed panel**;
  `window_stats_are` says so in the receipt itself, and the document block is
  the only inferential statement.
* **`contract_a` / `contract_b` and `cross_lane`** — what each side declares
  about its own lane. `paired` now **refuses a mixed-lane contrast** unless an
  explicit `--bridge` statement is passed (carried verbatim into the receipt;
  a bridge is context, not a correction). Of the historical pairings, only
  FP8-vs-BF16floor was same-lane; K6-vs-K8 mixed the sealed K6 with the
  streaming K8. The same-lane recompute is published beside it as
  `paired.K6stream-vs-K8` — mean 0.001331 (full) / 0.000847 (clean17), same
  ordering, same document-level p.
* **`--document-map`** — window→document provenance from the
  window-selection receipt. Without it a receipt labels itself
  descriptive-only.

The correction is logged in `docs/PUBLISHED-CORRECTIONS.md` §4 and
`docs/PEER-REVIEW-RESPONSE.md` (P1-15/P1-16).

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
