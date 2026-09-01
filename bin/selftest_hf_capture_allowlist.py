#!/usr/bin/env python3
"""Focused exact-set tests for hf_capture unexpected-tensor admission."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "engines" / "tools"))
import hf_capture as HC  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print("  %s  %s%s" % ("PASS" if condition else "FAIL", name,
                           (" -- " + detail) if detail else ""))


def report(keys):
    return {"observed": True, "conversion_errors_visible": True,
            "missing_keys": [], "unexpected_keys": sorted(keys),
            "mismatched_keys": [], "error_msgs": [], "conversion_errors": {}}


def refused(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def identities(raw):
    names = json.loads(raw)
    canonical = json.dumps(sorted(names), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(canonical).hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="hf-allowlist-selftest-") as work:
        path = Path(work) / "keys.json"
        expected = ["model.layers.13.a", "model.layers.13.b", "model.layers.13.c"]
        raw = (json.dumps(expected, indent=2) + "\n").encode()
        path.write_bytes(raw)
        raw_sha, names_sha = identities(raw)
        binding = HC.load_unexpected_tensor_allowlist(str(path), raw_sha, names_sha)
        exact = report(reversed(expected))
        result = HC.refuse_on_load_report(exact, False, binding)
        evidence = exact.get("unexpected_tensor_allowlist") or {}
        check("exact set passes independent of report order", result == []
              and evidence.get("expected_keys") == sorted(expected)
              and evidence.get("observed_keys") == sorted(expected)
              and evidence.get("exact_match") is True)
        check("full equality evidence carries raw and canonical identities",
              evidence.get("artifact_sha256") == raw_sha
              and evidence.get("canonical_sorted_names_sha256") == names_sha
              and evidence.get("missing_keys") == [] and evidence.get("extra_keys") == [])

        duplicate = Path(work) / "duplicate.json"
        duplicate.write_text(json.dumps([expected[0], expected[0]]))
        check("duplicate names refuse", refused(lambda: HC.load_unexpected_tensor_allowlist(
            str(duplicate))))
        check("duplicate observed keys refuse even when their unique set is exact",
              refused(lambda: HC.refuse_on_load_report(
                  report(expected + [expected[0]]), False, binding)))
        check("missing observed name refuses", refused(lambda: HC.refuse_on_load_report(
            report(expected[:-1]), False, binding)))
        check("extra observed name refuses", refused(lambda: HC.refuse_on_load_report(
            report(expected + ["model.layers.13.extra"]), False, binding)))
        renamed = [expected[0], expected[1], expected[2] + ".renamed"]
        check("renamed name refuses as one missing plus one extra", refused(
            lambda: HC.refuse_on_load_report(report(renamed), False, binding)))
        check("allowlist artifact-byte digest mismatch refuses", refused(
            lambda: HC.load_unexpected_tensor_allowlist(str(path), "0" * 64, names_sha)))
        check("canonical sorted-name digest mismatch refuses", refused(
            lambda: HC.load_unexpected_tensor_allowlist(str(path), raw_sha, "f" * 64)))

        reordered_raw = (json.dumps(list(reversed(expected)), indent=4) + "\n").encode()
        reordered = Path(work) / "reordered.json"
        reordered.write_bytes(reordered_raw)
        reordered_raw_sha, reordered_names_sha = identities(reordered_raw)
        check("semantic digest is order invariant but raw identity is not",
              reordered_names_sha == names_sha and reordered_raw_sha != raw_sha)
        check("same names in differently encoded artifact cannot substitute", refused(
            lambda: HC.load_unexpected_tensor_allowlist(str(reordered), raw_sha, names_sha)))

        m2 = REPO / "engines" / "tools" / "dione-evidence" / "m2-layer45-unexpected-keys.json"
        fruit = REPO / "engines" / "tools" / "layer-outer-evidence" / "fruit-layer13-unexpected-keys.json"
        m2_raw, fruit_raw = m2.read_bytes(), fruit.read_bytes()
        check("checked-in M2 list is exact 889-name public layer-45 evidence",
              len(json.loads(m2_raw)) == 889 and identities(m2_raw)[1]
              == "acc1e9f10c0f903c735a7fcf5fd267fc879bce65623f0b850f80016da5e903b7")
        check("checked-in Fruit list is exact 791-name pinned layer-13 evidence",
              len(json.loads(fruit_raw)) == 791 and identities(fruit_raw)[1]
              == "41b825b0045a2e1e90eea8f88bb06022459d26a3957c40c52d65d677d8a17968")
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
