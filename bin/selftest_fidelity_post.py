#!/usr/bin/env python3
"""Offline selftest for bin/fidelity_post.py's render().

Regression for a 2026-09-04 defect: the panel line read
`comparison["panel"]["scored_positions_total"]`, a key that has never existed
in `fidelity.resultsink`'s comparison-receipt schema (the field is
`scored_positions`), so every rendered post silently said "None scored
positions" -- a rendered public post with a broken number in it, not caught
by any test because none existed for this module.

Runs against real sealed result directories when available (a `result/`
beside a completed `measure-cloud --candidate-scope` run) and always also
runs a minimal synthetic fixture shaped like `render()`'s inputs, so this
suite is not skipped on a fresh checkout.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fidelity_post as FP  # noqa: E402

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
    else:
        FAIL.append(name)
        print("  FAIL  %s%s" % (name, (" -- %s" % detail) if detail else ""))


def _synthetic_loaded():
    return {
        "job": {
            "target": {"repo_id": "example/model", "revision": "a" * 40},
            "job_id_full": "j" * 64,
            "measurer": {"name": "example-measurer"},
        },
        "candidate": {
            "reference": {
                "repository": "example/root", "revision": "b" * 40,
                "capture_content_digest": "c" * 64, "panel_id": "panel--example",
            },
            "codec": "fp8_e4m3", "declared_bits": 8,
            "scope": {"path": "engines/scopes/scope--example.json",
                      "scope_digest": "f" * 16, "sha256": "f" * 64},
            "weights_decode": {"method": "fp8-block-dequant-to-bf16",
                               "quantization_config": {"fmt": "e4m3",
                                                        "weight_block_size": [128, 128]}},
        },
        "qualification": {
            "receipt_sha256": "g" * 64,
            "captures": {
                "canonical": {"capture_content_digest": "d" * 64,
                              "dataset_sha256": "h" * 64},
                "repeat": {"capture_content_digest": "d" * 64,
                           "dataset_sha256": "h" * 64},
            },
        },
        "comparison": {
            "receipt_sha256": "i" * 64,
            "metric": {"value": 0.0123, "direction": "reference_to_candidate",
                       "direction_label": "KL(reference || candidate)"},
            "kl": {"median": 0.001, "p95": 0.05, "p99": 0.1, "max": 1.0},
            "top1_agreement": 0.95,
            "comparability": {"class": "strict"},
            "panel": {"contexts": 25, "scored_positions": 51175},
            "estimator": {"accumulation_dtype": "float64"},
            "disclosures": [],
        },
        "publication": None,
    }


def main():
    real_dirs = sorted(glob.glob(
        os.path.expanduser("~/fidelity-runs/*/result")))
    fixtures = []
    for d in real_dirs:
        try:
            fixtures.append(("real result dir %s" % d, FP.load_result(d)))
        except (SystemExit, OSError):
            continue
    fixtures.append(("synthetic fixture", _synthetic_loaded()))

    exercised_real = False
    for label, loaded in fixtures:
        body = FP.render(loaded)
        panel_line = next(
            (l for l in body.splitlines() if l.startswith("| panel |")), "")
        check("%s: panel line has no unrendered None" % label,
              "None scored positions" not in body
              and ", None contexts" not in body,
              panel_line)
        scored = ((loaded["comparison"].get("panel") or {})
                  .get("scored_positions"))
        check("%s: exact scored-position count appears verbatim" % label,
              str(scored) in body, "expected %r in body" % scored)
        if label.startswith("real result dir"):
            exercised_real = True

    check("at least one real sealed result dir was exercised",
          exercised_real or not real_dirs,
          "no real result dirs under ~/fidelity-runs/*/result found; "
          "synthetic-only run")

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
