#!/usr/bin/env python3
"""T5 -- previews and teachers are STRUCTURALLY unsubmittable.

    python3 bin/selftest_submission_refusal.py

Two independent refusal axes exist; this test exercises the bin-side one
(fidelity/receipt.build_submission's denylist) and the field-shape guarantees
of preview receipts.  The registry-side axis (submission_schema const gate +
adapter schema allowlist) is documented, not executed here, deliberately: the
registry tools are being edited concurrently and this suite must not race
them.  (The live refusal by registry_add is demonstrated separately, as a
read-only invocation.)

Stock python3.9, stdlib, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.receipt import (                              # noqa: E402
    NotSubmittable, assert_submittable, build_submission,
)
from fidelity import previewstats as PS                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def expect_refusal(name, blocks, needle):
    try:
        assert_submittable(blocks)
        check(name, False, "was NOT refused")
    except NotSubmittable as exc:
        check(name, needle in str(exc), str(exc)[:140])


def minimal_submission_kwargs():
    """The smallest coherent build_submission call (a clean one must WORK)."""
    return dict(
        suite_root=ROOT,
        lane="streaming",
        measurer={"name": "t", "handle": "t", "url": None,
                  "is_artifact_author": False},
        artifact={"repo_id": "x/y", "revision": "0" * 40},
        panel={"panel_ref": "panel--x"},
        reference={"reference_ref": "reference--x"},
        metric={"name": "mean_of_run_means_tokenwise_kld", "value": 0.01,
                "units": "nats"},
        estimator={"accumulation_dtype": "float64"},
        determinism={"run_count": 2, "identical_across_runs": True},
        measurement_scope={"covers_full_panel": True, "scored_positions": 51175},
        produced_by={"tool": "selftest"},
    )


def main() -> int:
    print("[1] the denylist refuses each marker, with the reason named")
    expect_refusal(
        "schema containing '-preview.' is refused",
        {"metric": {"schema": "malaiwah.glm53-sampled-kld-preview.v1",
                    "value": 0.01}},
        "PREVIEW")
    expect_refusal(
        "not_submittable: true is refused",
        {"evidence": [{"kind": "receipt", "not_submittable": True}]},
        "not_submittable")
    expect_refusal(
        "capture_role bf16_teacher is refused (teachers are REFERENCES)",
        {"reference": {"capture_role": "bf16_teacher"}},
        "REFERENCE")
    expect_refusal(
        "markers are found NESTED, not just at the top level",
        {"environment": {"stack": {"inner": [
            {"schema": "malaiwah.glm53-census-kld-preview.v1"}]}}},
        "PREVIEW")

    print("\n[2] build_submission itself enforces the same denylist")
    kwargs = minimal_submission_kwargs()
    doc = build_submission(**kwargs)
    check("a clean submission still builds and self-seals",
          doc.get("submission_schema") ==
          "quant-fidelity-registry/submission-receipt.v1" and
          len(doc.get("receipt_sha256", "")) == 64)
    bad = minimal_submission_kwargs()
    bad["metric"] = {"name": "x", "value": 0.01,
                     "source_receipt": {
                         "schema": "malaiwah.glm53-sampled-kld-preview.v1"}}
    try:
        build_submission(**bad)
        check("build_submission refuses a preview-derived metric block", False)
    except NotSubmittable as exc:
        check("build_submission refuses a preview-derived metric block",
              "unsubmittable" in str(exc))
    bad = minimal_submission_kwargs()
    bad["reference"] = {"reference_ref": "reference--x",
                        "capture_role": "bf16_teacher"}
    try:
        build_submission(**bad)
        check("build_submission refuses a teacher capture as reference input", False)
    except NotSubmittable:
        check("build_submission refuses a teacher capture as reference input", True)

    print("\n[3] preview receipts are the wrong SHAPE for a row, by construction")
    per_window = {"final-%04d" % i: {"mean": 0.01} for i in range(25)}
    doc = PS.build_preview_receipt(
        kind="sampled", per_window=per_window, windows_total=25,
        panel_estimate=0.0123,
        ci95_z={"low": 0.011, "high": 0.013},
        ci95_bootstrap={"method": "stratified-position-bootstrap", "B": 2000,
                        "seed": 0, "low": 0.011, "high": 0.0135},
        sampling_design={"scheme": "stratified-systematic",
                         "positions_per_window": 256, "seed": 0},
        tail={"max_sampled_value": 1.2, "top3_share_of_estimate": 0.2},
        lane_disclosure={"lane": "local-preview", "device": "mps"},
        teacher_receipt_sha256="e" * 64)
    check("preview receipt has NO submission_schema key",
          "submission_schema" not in doc)
    check("preview receipt has NO measured_mean_kld key (headline field is "
          "preview_panel_mean_estimate)",
          "measured_mean_kld" not in doc and
          doc.get("preview_panel_mean_estimate") == 0.0123)
    check("preview receipt is sampled:true + not_submittable:true",
          doc["sampled"] is True and doc["not_submittable"] is True)
    check("its schema carries the '-preview.' substring both axes key on",
          "-preview." in doc["schema"])
    try:
        assert_submittable({"receipt": doc})
        check("the denylist refuses the built preview receipt", False)
    except NotSubmittable:
        check("the denylist refuses the built preview receipt", True)

    print("\n[4] the 25-window panel gate is enforced INSIDE receipt assembly")
    subset = {"final-%04d" % i: {"mean": 0.01} for i in range(11)}
    try:
        PS.build_preview_receipt(
            kind="sampled", per_window=subset, windows_total=25,
            panel_estimate=0.0123, ci95_z=None, ci95_bootstrap=None,
            sampling_design=None, tail=None,
            lane_disclosure={}, teacher_receipt_sha256=None)
        check("panel estimate from 11 windows is refused", False)
    except PS.PanelGateError as exc:
        check("panel estimate from 11 windows is refused", True)
        check("the refusal quotes the power arithmetic (1.73e-3 vs 1.22e-3)",
              "1.73e-3" in str(exc) and "1.22e-3" in str(exc))
    doc = PS.build_preview_receipt(
        kind="sampled", per_window=subset, windows_total=25,
        panel_estimate=None, ci95_z=None, ci95_bootstrap=None,
        sampling_design=None, tail=None,
        lane_disclosure={}, teacher_receipt_sha256=None)
    check("per-window diagnostics WITHOUT a panel field are still allowed",
          "preview_panel_mean_estimate" not in doc and doc["windows_used"] == 11)

    print("\n" + "-" * 72)
    print("selftest_submission_refusal: %d passed, %d failed"
          % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
