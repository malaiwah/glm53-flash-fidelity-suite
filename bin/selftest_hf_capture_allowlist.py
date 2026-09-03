#!/usr/bin/env python3
"""Focused exact-set tests for hf_capture unexpected-tensor admission."""
from __future__ import annotations

import ast
import hashlib
import inspect
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

        # Layer-outer streaming: the pre-capture report predates every layer
        # load, so names the allowlist correctly carries for streamed layers
        # are not yet observed. The non-exact form tolerates that -- and ONLY
        # that; an extra still refuses -- and the exact form is run on the
        # union after the layers have streamed. Before 2026-09-03 the exact
        # form ran on the early report and the union was never checked or
        # disclosed (the published Fruit root under-reports its unused set).
        resident = ["model.layers.13.a", "model.layers.13.b"]
        streamed = ["model.layers.3.self_attn.indexer.wk.weight"]
        union_path = Path(work) / "union.json"
        union_raw = (json.dumps(resident + streamed, indent=2) + "\n").encode()
        union_path.write_bytes(union_raw)
        union_binding = HC.load_unexpected_tensor_allowlist(
            str(union_path), *identities(union_raw))
        early = report(resident)
        check("non-exact (pre-streaming) check tolerates not-yet-observed names",
              HC.refuse_on_load_report(early, False, union_binding,
                                       require_exact_unexpected=False) == []
              and early["unexpected_tensor_allowlist"]["missing_keys"] == streamed
              and early["unexpected_tensor_allowlist"]["exact_match"] is False)
        check("non-exact check still refuses an extra name", refused(
            lambda: HC.refuse_on_load_report(
                report(resident + ["model.layers.3.extra"]), False, union_binding,
                require_exact_unexpected=False)))
        check("exact (post-streaming) check refuses the pre-streaming subset", refused(
            lambda: HC.refuse_on_load_report(report(resident), False, union_binding)))
        final = report(streamed + resident)
        check("exact check passes on the streamed union",
              HC.refuse_on_load_report(final, False, union_binding) == []
              and final["unexpected_tensor_allowlist"]["exact_match"] is True)
        tree = ast.parse(inspect.getsource(HC.run_capture))
        guard_calls = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call)
                       and getattr(node.func, "id", None) == "refuse_on_load_report"]
        modes = sorted(
            ast.unparse(kw.value)
            for call in guard_calls for kw in call.keywords
            if kw.arg == "require_exact_unexpected")
        exact_calls = [call for call in guard_calls if not any(
            kw.arg == "require_exact_unexpected" for kw in call.keywords)]
        check("run_capture guards the load three times: per streamed layer "
              "(non-exact), before streaming (exact only without a streamer), "
              "and exactly on the union after streaming",
              len(guard_calls) == 3 and modes == ["False", "streamer is None"]
              and len(exact_calls) == 1)

        m2 = REPO / "engines" / "tools" / "dione-evidence" / "m2-layer45-unexpected-keys.json"
        fruit = REPO / "engines" / "tools" / "layer-outer-evidence" / "fruit-unexpected-keys.json"
        m2_raw, fruit_raw = m2.read_bytes(), fruit.read_bytes()
        check("checked-in M2 list is exact 889-name public layer-45 evidence",
              len(json.loads(m2_raw)) == 889 and identities(m2_raw)[1]
              == "acc1e9f10c0f903c735a7fcf5fd267fc879bce65623f0b850f80016da5e903b7")
        # 791 MTP-block names (layer 13) plus 5 DSA indexer tensors on each of
        # layers 3..12 (`indexer_types` says `shared`, so transformers builds no
        # indexer there). Derived by derive_unexpected_allowlist.py from the
        # streamed loader; the 791-only list refused a paid capture at layer 3.
        fruit_names = json.loads(fruit_raw)
        check("checked-in Fruit list is the exact 841-name streamed-load evidence",
              len(fruit_names) == 841 and identities(fruit_raw)[1]
              == "f7a80a42958ad694212db5dd249d32cd55a1ccbca2622713fc3433a718ec3257"
              and sum(1 for n in fruit_names if ".indexer." in n
                      and not n.startswith("model.layers.13.")) == 50)
        provenance = json.loads((fruit.parent / (fruit.name + ".provenance.json")).read_bytes())
        check("Fruit provenance sidecar binds the artifact it describes",
              provenance.get("artifact_sha256") == identities(fruit_raw)[0]
              and provenance.get("canonical_sorted_names_sha256") == identities(fruit_raw)[1]
              and provenance.get("count") == 841)
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
