# Registry integration — what was applied, and what is still deferred

The fidelity-dataset comparison receipt needed small, **additive** changes under
`registry/` before a step-3 receipt could become a registry row. This document
was originally a specification of changes deliberately *not* applied, because a
concurrent workflow held `registry/` open. That workflow has landed; the changes
below are now **applied**, and `make check` is green with them:

```
62 passed, 0 failed
joint-standard checks: 433 run, 0 error(s)
OK: schema + invariants clean, README tables match the data, self-tests pass.
```

The one piece that is still deferred is named in §4, with the reason.

---

## 1. Two new disclosure codes — APPLIED

`registry/schema/invariants.json` → `known_disclosure_codes`, which **DISC-004**
requires any emitted code to appear in:

```json
"head_substituted",
"lossy_capture_codec"
```

* **`lossy_capture_codec`** — a capture whose stored values are not the model's
  values. Emitted by the comparator when either side's `capture.lossy_codec` is
  non-null; the concrete case is a llama.cpp `.kld` (uint16 log-probs with a
  hard `max_logit − 16` floor). Forces `comparability.class: advisory`.
* **`head_substituted`** — a comparison that applied one artifact's head to
  another artifact's hidden states. Emitted **only** under
  `--disclose-head-substitution` at severity `blocking`, which under
  **DISC-003** forces `status ∈ {pending, retracted}`.

### Correction: the protection is at row-ingest, NOT at submission

An earlier revision of this document said of those two codes:

> *"Both codes are already emitted by `bin/fidelity/dscompare.py` today, so a
> receipt carrying either is currently unsubmittable — which is the safe
> direction to be wrong in."*

**That was false, and it was load-bearing** — it was the justification for
deferring these changes. DISC-003 and DISC-004 live in
`registry_validate.py::check_disclosures`, which iterates over the registry
**collections** (models, artifacts, panels, references, pipelines,
measurements). The `--submission` path is a separate function,
`check_submission`, and calls neither. A structurally valid submission carrying
`head_substituted` at severity `blocking` is **ACCEPTED at exit 0** by the gate
`bin/registry-submit` runs and `CONTRIBUTING.md` tells a submitter to run. It is
caught later, at row ingest — which is real protection, but not the protection
that was claimed, and not at the moment a contributor would find out.

The fix is not to lean harder on a downstream gate.
`bin/fidelity/dscompare.py::emit_submission` now **refuses outright** when any
disclosure carries `severity: blocking` (**SC-5**), alongside its existing SC-3
refusal of a non-measurement `comparison_kind`. The docs already said such a
number is not publishable as a measurement; the tool that mints it is now the
one that says no. Case **N15** in `bin/selftest_fidelity_compare.py`.

## 2. `comparability` on a submission — APPLIED

`submission.schema.json` gains an **optional** `comparability` block:

```json
"comparability": {
  "bias": { "$ref": "measurement.schema.json#/properties/comparability/properties/bias" },
  "usable_as_floor": { "type": ["boolean", "null"] }
}
```

and `measurement.schema.json`'s `comparability` gains an optional
`usable_as_floor`. `registry_add.py::submission_to_records` prefers a declared
bias when the submission carries one, and falls back to the previous derivation
from `estimator.stack_relation` when it does not.

**Why this was necessary.** `emit_submission` forwarded metric, estimator,
determinism, scope and disclosures — and dropped `comparability.bias` and
`usable_as_floor` on the floor. `registry_add` synthesises a bias only for
`stack_relation == cross_stack`, so a row derived from a **head-substituted**
comparison arrived with `bias: null` for a comparison whose own receipt said
`{kind: other, direction: downward, detail: "…the candidate's own
head-quantization error is erased, biasing the number DOWNWARD…"}`. That is
exactly what BIAS-001 exists to prevent, arriving through the submission path.
Cases **N16b**, **N16c**, **N16d**.

## 3. BIAS-007 — APPLIED

```
BIAS-007 [error] — a row whose comparability.usable_as_floor is false may not be
named as any other row's comparability.bias.floor_measurement_ref.
```

Implemented in `registry_validate.py` beside BIAS-006, in the same loop over
`comparability.bias.floor_measurement_ref`. BIAS-006 refuses a floor from
another **lane**; BIAS-007 refuses a floor the producing tool itself stamped
unusable — cross-lane, cross-stack, or head-substituted. It is the registry
honouring a verdict the comparator already reached, rather than re-deriving it
from fields that cannot see the reason.

## 4. Still deferred: a `registry_add` adapter for the receipt itself

`malaiwah.fidelity-comparison-receipt.v1` is **not** in `registry_add.py`'s
`OWN_SCHEMAS`, so `registry_add.py from-receipt` cannot ingest a comparison
receipt directly. This is deliberate and it costs nothing today, because the
submission path is the supported one and it now works end to end:

```
compare --emit-submission --submission-provenance FILE
  → submission-receipt.json
  → registry_validate.py --submission   ACCEPTED
  → copy into registry/receipts/<handle>/ and open a PR
```

The comparison receipt is the **evidence**; the submission is the **claim**.
Adding a second ingest path would mean two places that map a number onto a row,
which is the same "two implementations of one thing" problem the shared
`registry_lib` import exists to avoid. If it is ever added, the one rule it must
enforce that nothing else can is SC-3:

```python
if receipt.get("comparison_kind") != "measurement":
    raise Refusal(...)
```

Three further invariants remain specified-but-unimplemented, all mechanically
checkable from data already in the rows, and none of them blocking:

* **DS-001** [error] — a row whose `provenance.sources[]` names a fidelity
  dataset carries that dataset's `dataset_sha256` as the source digest. (The
  emitter already does this; the invariant would recompute it for hand-edited
  data.)
* **DS-002** [error] — a `head_substituted` disclosure at severity `blocking`
  forces `status ∈ {pending, retracted}`. A specialization of DISC-003, stated
  separately so the code cannot be silently downgraded to `caveat` and keep its
  row published.
* **DS-005** [warn] — `comparability.bias.floor_measurement_ref`, when set,
  points at a measurement whose `measurement_scope.scope_name` equals the biased
  row's own. This is the **scope** analogue of BIAS-006's lane rule, and it was
  not hypothetical: the registry briefly carried a 17-window row citing a
  25-window floor. Until the invariant exists the guard lives at card level, in
  `bin/fidelity/cardmeta.py::attributable_refusal`, exercised by case **K8b**.

---

## What already existed and needed nothing

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
* **`measurement.schema.json` rule 4** — a `cross_stack` row already required
  both a typed bias block **and** a `cross_stack_capture` /
  `cross_engine_capture` disclosure at `affects_comparability: true`. The
  comparator emitted the bias block and not the disclosure, so its rows were
  schema-invalid on arrival. Fixed in `dscompare.py`, not in the schema —
  the schema was right.
