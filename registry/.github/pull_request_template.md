<!--
One measurement per PR. You add exactly ONE file: receipts/<your-hf-handle>/<slug>.json
Do not edit data/, index.json, schema/ or tools/ — CI fails the PR if you do.
Full instructions: CONTRIBUTING.md
-->

## Submission

- **Artifact:** `<hf-repo>` @ `<40-hex commit>`
- **Panel:** `panel--…`
- **Reference:** `reference--…`
- **Metric:** `mean_tokenwise_kld` = `<full float64 value>` nats
- **Lane:** `sealed-ep8` | `streaming` | `local-mps` | `local-cuda-budget`
- **Receipt file:** `receipts/<handle>/<slug>.json`

## Who did what

- **I measured this:** yes / no
- **I made this quant:** yes / no — if no, the quant is by `<handle>`
- **I built this panel or teacher capture:** yes / no

## Checklist

- [ ] The receipt was sealed by a runner (`bin/measure-cloud` / `bin/measure-local`); I did not hand-edit it.
- [ ] `receipt_sha256` verifies (the four-line check in CONTRIBUTING.md §1 prints `True`).
- [ ] `artifact.revision` is the immutable 40-hex commit, not `main` or a tag.
- [ ] `panel_ref` and `reference_ref` already exist in the registry.
- [ ] `metric.value` is the unrounded float64 from the run.
- [ ] `measurement_scope.covers_full_panel` is honest; a subset names its `subset_detail`.
- [ ] `disclosures` is non-empty (use `no_known_deviations` if there is genuinely nothing).
- [ ] This PR touches only `receipts/`.

## Anything unusual about this run?

<!-- Interruptions, retries, an unexpected disclosure, a number that surprised
you, hardware that ran out of memory partway. Write it here even if the receipt
already records it — the prose is what a reviewer reads first. "Nothing" is a
fine answer. -->
