#!/usr/bin/env python3
"""P1-07: `identical_across_runs: true` needs one valid digest PER claimed run.

The old ingest reduced the per-run tokenwise digests to a SET before deciding
identity, so a five-run submission with one digest and four missing digests
produced `identical: true` backed by one evidence hash -- "five runs, bitwise
identical" manufactured from "one of five runs supplied a digest", without
falsifying any input field.

Cases per the review: 0, 1, N-1 and N supplied digests, plus one mismatching
digest, on both affected adapters (the five-cold-run family and the foreign
repeated-run family). Verified to FAIL against the pre-fix adapters: with the
helper reverted, the 1-of-5 and 4-of-5 cases return identical=True.

Stock python3, offline, no installs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "registry", "tools"))

import registry_add as A  # noqa: E402

H1 = "a" * 64
H2 = "b" * 64

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def five_run_receipt(digests):
    """A minimal, internally consistent five-cold-run receipt; digests is one
    entry per run (None = the run supplied no digest)."""
    runs = [{"mean_kld": 0.0137, "prediction_positions": 2047,
             **({"tokenwise_kld_sha256": d} if d is not None else {})}
            for d in digests]
    return {
        "schema": A.FIVE_COLD_RUN,
        "runs": runs,
        "run_count": len(runs),
        "mean_of_run_means": 0.0137,
        "population_stddev_of_run_means": 0.0,
        "minimum_run_mean": 0.0137,
        "maximum_run_mean": 0.0137,
        "kld_direction": "teacher_to_student",
        "compute_dtype": "float64",
    }


def adapt5(digests):
    r = five_run_receipt(digests)
    return A.adapt_packed_and_five_run([(r, "five.json", "0" * 64)])


def foreign_receipt(digests):
    runs = [{"mean_kld": 0.0137, "prediction_positions": 2047,
             **({"tokenwise_kld_sha256": d} if d is not None else {})}
            for d in digests]
    return {"schema": A.FOREIGN_REPEATED, "runs": runs,
            "mean_of_run_means": 0.0137, "compute_dtype": "float64"}


def adaptf(digests):
    return A.adapt_foreign(foreign_receipt(digests), "foreign.json")


for label, fn in (("five-cold-run", adapt5), ("foreign repeated", adaptf)):
    # N of N equal digests, N >= 2: the only shape that may claim identity
    out = fn([H1] * 5)
    check("%s: 5/5 equal digests -> identical True" % label, out["identical"] is True)
    check("%s: 5/5 preserves the per-run vector" % label,
          out.get("evidence_hashes_per_run") == [H1] * 5)
    check("%s: 5/5 evidence is the one distinct hash" % label,
          out.get("evidence_hashes") == [H1])

    # 1 of 5: the review's manufactured-identity case
    out = fn([H1, None, None, None, None])
    check("%s: 1/5 digests -> identical is UNKNOWN, never True" % label,
          out["identical"] is None, "got %r" % out["identical"])
    check("%s: 1/5 keeps the partial evidence visible" % label,
          out.get("evidence_hashes") == [H1])
    check("%s: 1/5 carries a missing-evidence disclosure" % label,
          (out.get("det_disclosure") or {}).get("code") == "determinism_evidence_incomplete")
    check("%s: 1/5 per-run vector shows the four holes" % label,
          out.get("evidence_hashes_per_run") == [H1, None, None, None, None])

    # N-1 of N
    out = fn([H1, H1, H1, H1, None])
    check("%s: 4/5 digests -> identical is UNKNOWN, never True" % label,
          out["identical"] is None, "got %r" % out["identical"])
    check("%s: 4/5 carries a missing-evidence disclosure" % label,
          (out.get("det_disclosure") or {}).get("code") == "determinism_evidence_incomplete")

    # 0 of N
    out = fn([None] * 5)
    check("%s: 0/5 digests -> identical is UNKNOWN with no evidence hashes" % label,
          out["identical"] is None and not out.get("evidence_hashes"))
    check("%s: 0/5 evidence_kind degrades to run_mean_equality_only" % label,
          out["evidence_kind"] == "run_mean_equality_only")

    # one mismatching digest refutes identity outright
    out = fn([H1, H1, H1, H1, H2])
    check("%s: one mismatching digest -> identical False" % label,
          out["identical"] is False, "got %r" % out["identical"])

    # a malformed digest is missing evidence, not a wildcard
    out = fn([H1, H1, H1, H1, "not-a-sha256"])
    check("%s: malformed digest counts as missing -> UNKNOWN" % label,
          out["identical"] is None, "got %r" % out["identical"])

    # a single run can never be "identical across runs"
    out = fn([H1])
    check("%s: 1 run with 1 digest -> never True" % label, out["identical"] is not True)

# the decision helper directly: sanity on the empty run list
d = A._determinism_from_run_digests([])
check("helper: zero runs -> unknown, no note", d["identical"] is None and d["note"] is None)

print()
if failures:
    print("selftest_registry_determinism: FAILED: %s" % ", ".join(failures))
    sys.exit(1)
print("selftest_registry_determinism: all checks passed")
