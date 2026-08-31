# The same-lane teacher — driving the streaming lane's floor to zero

**Status: tooling shipped, GPU run NOT executed.** `stream_score.py
--capture-role teacher` exists, is selftested (ladder rungs L1.g–L1.j), and its
output tree is a valid `--teacher` for `k6/tools/k6_kld_report.py` and
`bin/kld-preview`. The ~$6 capture run itself is deliberately not part of this
change (no renting); this document is the complete recipe for whoever runs it.

## Why

The streaming lane's measured floor against the sealed EP8 teacher is
**0.011505922619330299** nats (2 cold runs, identical means,
`k6/native-bf16-kld.json`). Every quant measured on this lane sits on that
floor: K6's panel mean 0.013715 is only 0.002209 of quantization error; K8's
0.012384 is only 0.000878. The floor exists because the teacher was captured
on a DIFFERENT stack (8×H200 EP8 NCCL) than the streaming lane replays.

A teacher captured **by the streaming lane itself** removes that term exactly:
the lane is bitwise deterministic (proven: two cold runs, one distinct
`tokenwise_kld_sha256`), so a native re-run reproduces the teacher's fp32
logit files byte for byte, and the fp64 KLD of bitwise-equal logits is a sum
of exact `+0.0`s. Not epsilon — zero. `bin/selftest_zero_floor.py` proves the
identity end to end on synthetic captures, including the fixed constant below.

Does fp32-logit storage rounding leave residue? **No, not in the floor**: the
`.float()` store is applied identically on both sides, so it cancels
bit-for-bit. What it does do is *define the reference*: the recorded teacher
is "the lane's bf16 forward, logits rounded to fp32" — a property of the
reference (recorded as `capture.logits_dtype`), not a measurement error, and
the same convention the sealed EP8 teacher already uses.

## The capture (the ~$6 recipe)

One H200 spot (≈$1.99/h, the class the floor was measured on). Stage the
official BF16 tree + the sealed release inventory + the sealed token panel.
Then, twice:

```bash
FIDELITY_PYTHON stream_score.py \
    --source native --capture-role teacher \
    --inventory <sealed glm-release-inventory.v1> \
    --bf16 <official BF16 tree> \
    --teacher <panel locator> --token-panel <sealed token-panel receipt> \
    --profile native-bf16 --ep-emulate 8 --reduce-order fp32 \
    --cold-run {1,2} --out runs/teacher-r{1,2}
```

~1.5–2 h per run (IO-bound: 609 GB of routed BF16 re-read per window; the
floor runs measured ~8.3 min/window on CephFS at ~1.05 GB/s), so ≈$6–8 for
the pair plus staging. Peak ≈47 GB VRAM, same as every streaming run.

**Determinism evidence requirement:** the 25 per-window `logits/*.safetensors`
sha256 sets of the two runs must be IDENTICAL. That is `evidence_kind:
logits_tensor_sha256` — never receipt-file or archive hashes (campaign lesson
27; the registry's determinism schema refuses those kinds by design). Either
run's tree is then the teacher: 31.7 GB of fp32 logits + 4 receipts.

What `--capture-role teacher` changes, and only this:

* `capture_role` becomes `bf16_teacher` (the exact predicate
  `kld_report._find_teacher_receipt` discovers teachers by — schema stays
  `quant-pipeline.glm53-logit-capture.v1`, verified by ladder rung L1.g);
* the receipt gains a sealed additive block `teacher_provenance`
  (`schema: malaiwah.glm53-same-lane-teacher-provenance.v1`, carrying
  `teacher_label: native-bf16-streaming-v1`, lane, ep_emulate, reduce_order,
  stream_mode, grouped_mm kernel, device, torch/transformers versions,
  cold_run) — covered by `receipt_sha256`.

Refused by construction: `--capture-role teacher` without `--source native`
(a packed student cannot be a teacher), with `--windows` subsets, or with
`--store-positions` sampling (a subset teacher would silently redefine the
panel every student is scored against).

## The floor ladder (decision rule; tooling enforces it)

* **T1** — a fresh native run's per-window logit sha256s equal the teacher's
  `logit_files[].sha256` → **floor ≡ 0.0 exactly**, no KLD pass needed.
  Every tokenwise value is `+0.0`; the run's `tokenwise-kld.npy` is the
  np.save of 51,175 float64 zeros, whose sha256 is the fixed constant
  `3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17`
  (409,528 bytes — asserted by `bin/selftest_zero_floor.py`).
* **T2** — the hashes differ → **NEVER assume small.** A different grouped_mm
  kernel / GPU class / torch build changes the bf16 forward itself, and that
  class of difference is exactly what produced the 0.0115 cross-topology
  floor: it can be 1e-2-class. Measure the residual floor with the native run
  just made.

Rule enforced by `bin/fidelity-stats attributable`: **"floor = 0" may be
claimed only with T1 hash evidence** (`zero_floor_evidence` of kind
`logits_tensor_sha256` in the floor summary); a claimed zero floor without it
is refused.

**Scope note.** The teacher zeroes the floor for the CUDA streaming lane it
was captured on. A local Apple/MPS pass against it is a *different lane* (the
MPS forward is not bitwise CUDA; only the decode is proven MPS==CPU bitwise)
with its own unknown floor — measuring it needs a local native pass over the
~630 GB checkpoint. Local quant-vs-quant paired deltas are floor-invariant
(the shared floor subtracts out exactly in the difference — observed on the
committed data: raw K6−K8 delta == attributable delta to the last digit).

## Recording it in the registry (paste-ready spec — registry/ is not edited here)

One new `references.jsonl` row:

```json
{"id": "reference--glm-5.3-flash--native-bf16--streaming-v1",
 "panel_ref": "panel--glm53.brandonmusic.final25",
 "artifact_ref": "artifact--zai-org.glm-5.3-flash-bf16.a6c167b6",
 "reference_kind": "native_bf16",
 "logits_available": true,
 "capture": {
   "stack": "glm53-fidelity-suite streaming lane (stream_score.py --source native --capture-role teacher, EP8-emulated single device, reduce-order fp32)",
   "stack_version": "<git rev of the capture checkout>",
   "compute_dtype": "bf16",
   "logits_dtype": "float32",
   "capture_receipt_sha256": "<the run's receipt_sha256>"},
 "self_consistency": {"floor_measurement_ref": "<the T1/T2 floor measurement row's id>"}}
```

Consequences, by the registry's own arithmetic:

* `comparability.key_inputs` includes `reference_id`, so rows against the new
  reference form a **new `cmp--` group** and can never be pooled with
  sealed-teacher rows. That is the anti-conflation mechanism working, not a
  limitation.
* Measurements against it use `estimator.stack_relation: "same_stack"` with
  **no bias block** — that is the accuracy win over today's rows.
* A teacher capture is a REFERENCE record, never a measurement:
  `bin/fidelity/receipt.py` refuses any `capture_role: bf16_teacher` input to
  `build_submission`, and `registry_add.py` has no adapter for a capture
  receipt (both demonstrated by selftests).

## Portability rule (already implemented)

A teacher tree moved off its capture box keeps the receipt's absolute logit
paths. `kld_report.py` and `bin/kld-preview` fall back to
`<teacher_root>/logits/<basename>` and, **in the fallback path only**, verify
the file's sha256 against the receipt row before use — hash content, not
containers. The sealed fast path (recorded path exists) is byte-identical to
the pre-change behaviour.

## Open items

* The $6 GPU pair-run itself (T1 verification + publishing the 31.7 GB tree
  the way the sealed teacher is published).
* The local (Apple) lane's floor against any teacher — needs a local native
  pass; until then local runs are previews and say so.
