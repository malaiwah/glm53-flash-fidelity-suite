# Registry integration — specified, deliberately not applied

The fidelity-dataset comparison receipt needs three small, **additive** changes
under `registry/` before a step-3 receipt can become a registry row. They are
specified here rather than applied, for one reason found at implementation
time and not at design time:

> **`registry/` is held open by a concurrent workflow.** While this work was
> being built, `registry/data/{measurements,panels,references}.jsonl`,
> `registry/schema/{invariants,measurement,submission}.schema.json`,
> `registry/index.json`, `registry/Makefile`, `registry/README.md` and
> `registry/tools/seed_registry.py` were all modified in the working tree by
> the sequential measurement workflow, and `make check` passed through an
> intermediate state of **11 errors** before returning to `62 passed, 0
> failed`. Editing `schema/invariants.json` — which that workflow is itself
> editing — would have produced a merge conflict in a file whose correctness
> is enforced by a 90-invariant validator.

The constraint was to keep `make check` at 0 errors. Not touching the tree is
the way to honour that while another workflow owns it. **Nothing under
`registry/` was modified by this work** (`git status registry/` shows only that
workflow's changes), and `make check` is green at the time of writing:

```
62 passed, 0 failed
joint-standard checks: 433 run, 0 error(s)
OK: schema + invariants clean, README tables match the data, self-tests pass.
```

Apply the three changes below in a single change, running `cd registry && make
check` after each, once the concurrent workflow has landed.

---

## 1. Two new disclosure codes

`registry/schema/invariants.json` → `known_disclosure_codes`, which
**DISC-004** requires any emitted code to appear in:

```json
"lossy_capture_codec",
"head_substituted"
```

* **`lossy_capture_codec`** — a capture whose stored values are not the model's
  values. Emitted by the comparator when either side's `capture.lossy_codec` is
  non-null; the concrete case is a llama.cpp `.kld` (uint16 log-probs with a
  hard `max_logit − 16` floor). Forces `comparability.class: advisory`.
* **`head_substituted`** — a comparison that applied one artifact's head to
  another artifact's hidden states. Emitted **only** under
  `--disclose-head-substitution` at severity `blocking`, which under
  **DISC-003** forces `status ∈ {pending, retracted}`. A head-substituted number
  is not publishable as a measurement, and that is the intent.

Both codes are already emitted by `bin/fidelity/dscompare.py` today, so a
receipt carrying either is currently unsubmittable — which is the safe
direction to be wrong in.

## 2. A `registry_add` adapter

`registry/tools/registry_add.py`: add the exact string
`malaiwah.fidelity-comparison-receipt.v1` to `OWN_SCHEMAS` and an adapter that
maps a receipt onto a measurement row.

The one rule that adapter must enforce, which nothing else can:

```python
if receipt.get("comparison_kind") != "measurement":
    raise Refusal(...)   # SC-3
```

`bin/fidelity/dscompare.py::emit_submission` already refuses this bin-side, and
`bin/fidelity/receipt.py::_scan_for_unsubmittable` is the second independent
axis. The registry-side check is the third, and it is the one that survives
somebody hand-editing a receipt.

Field mapping (every value is already in the receipt):

| row field | receipt path |
|---|---|
| `metric` | `metric` (name/value/units/direction/higher_is_better) |
| `estimator` | `estimator` |
| `determinism` | `determinism` |
| `measurement_scope` | `measurement_scope` |
| `comparability.key` / `key_inputs` | `comparability.key` / `key_inputs` (computed with `registry_lib.comparability_key`, not recomputed) |
| `comparability.bias` | `comparability.bias` |
| `scope_digest` | `candidate.scope_digest` |
| `disclosures` | `disclosures` |
| `provenance.sources[]` | `reference.dataset_sha256` and `candidate.dataset_sha256`, kind `fidelity_dataset` |

## 3. Four invariants

All mechanically checkable from data already in the rows. Suggested severities
in brackets.

* **DS-001** [error] — a row whose `provenance.sources[]` contains a
  `fidelity_dataset` source carries that dataset's `dataset_sha256` as the
  source digest. A row derived from a dataset must name the dataset by its
  seal, not by its repository.
* **DS-002** [error] — a `head_substituted` disclosure at severity `blocking`
  forces `status ∈ {pending, retracted}`. This is a specialization of DISC-003,
  stated separately so the code cannot be silently downgraded to `caveat` and
  keep its row published.
* **DS-003** [error] — a row may not be derived from a receipt whose
  `comparison_kind` is `reproduction_confirmation` or `run_to_run_floor`.
  Enforced at write time by the adapter above; this recomputes it for
  hand-edited data, exactly as BIAS-006 does for the lane check.
* **DS-004** [warn] — a row citing a fidelity dataset whose `lane` differs from
  the row's own lane carries a `cross_engine_capture` or `non_sealed_lane`
  disclosure.

### A fifth, found in live data

**DS-005** [warn] — `comparability.bias.floor_measurement_ref`, when set, points
at a measurement whose `measurement_scope.scope_name` equals the biased row's
own `scope_name`.

This is the scope analogue of **BIAS-006**'s lane rule, and it was not
hypothetical. During this work the registry briefly carried
`measurement--glm53.k8-8bpw-stream.brandonmusic-final25.clean17` (17 windows,
34,799 positions) citing
`measurement--glm53.bf16-stream-floor.brandonmusic-final25` (25 windows, 51,175
positions) as its floor. A 25-window floor is not a 17-window row's zero-point,
for exactly the reason a streaming floor is not a sealed-lane row's zero-point.
The concurrent workflow resolved it independently — the clean17 rows now carry
`floor_measurement_ref: null` — but nothing in the schema *prevents* it
recurring.

Until the invariant exists, the guard lives at card level:
`bin/fidelity/cardmeta.py::attributable_refusal` withholds the
floor-subtracted number when the floor's lane **or scope** differs from the
row's, and records why in
`x_fidelity.measurements[].quantization_attributable_withheld`. It is exercised
by case **K8b** in `bin/selftest_fidelity_card.py`.

---

## What already exists and needs nothing

* **`reference.logits_available`** — already in `reference.schema.json`,
  documented as *"false means a number against this reference can never be
  re-derived, only re-run."* Publishing a root fidelity dataset is precisely the
  act of flipping it to `true`. No schema change.
* **`estimator.head_policy`** — already `{native_head, shared_reference_head,
  dequantized_head, unknown}`, and **REFC-003** already binds
  `reference.capture.head_source == shared_head_artifact` ⟺
  `head_policy == shared_reference_head` in both directions. The comparator's
  HEAD-1b refusal is the same condition, checked months earlier.
* **`lane`** — the five-value enum stays as it is. A kimi-k3-adapted dataset
  maps to `other` with `lane_inferred: true`. Adding a `serving` value would
  reclassify existing rows and is an operator decision (spec §14).
