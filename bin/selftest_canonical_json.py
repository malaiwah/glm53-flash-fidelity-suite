#!/usr/bin/env python3
"""Two canonical_json implementations, one published wire format.

`bin/fidelity/common.py` says its `canonical_json` "must match
`registry_lib.py` exactly" and, until this file, nothing checked it. That
comment is load-bearing: every sealed receipt's `receipt_sha256`, every
`scope_digest`, and every comparability key is a hash over one of these two
serializations, and the two halves of the suite pick different ones --
`bin/` seals a receipt, `registry/tools/` re-verifies it. If they ever
disagreed by a single byte, every receipt sealed on one side would fail
verification on the other, and the failure would read as "this receipt was
tampered with" rather than "two helpers drifted".

They cannot be merged into one module: `registry/tools/` is deliberately
standalone (`registry_validate.py` asserts its import graph reaches no
networking module, which is what makes an offline validator auditable), and
`bin/fidelity/` is the campaign side. Duplication is the price of that
separation, so the duplication gets a test instead of a comment.

What is asserted is the OUTPUT, not the implementation. Either may be rewritten
freely; they may not disagree on a byte.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "registry", "tools"))

from fidelity import common as C                          # noqa: E402
import registry_lib as L                                  # noqa: E402

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


# The inputs where JSON serializers actually diverge, not a happy path.
CASES = [
    ("empty object", {}),
    ("empty array", []),
    ("key order", {"b": 1, "a": 2, "C": 3, "_": 4}),
    ("nested key order", {"z": {"y": 1, "a": 2}, "a": [{"b": 1, "a": 2}]}),
    ("non-ascii value", {"note": "turboderp's mul1 — café naïve"}),
    ("non-ascii KEY", {"clé": "valeur", "a": 1}),
    ("cjk", {"model": "智谱 GLM"}),
    ("emoji", {"tag": "\U0001F9EE fidelity"}),
    ("surrogate-pair-adjacent", {"s": "a�b"}),
    ("float that is not exact", {"kld": 0.12163767673339457}),
    ("float with exponent", {"eps": 1e-30, "big": 1.7976931348623157e308}),
    ("negative zero", {"z": -0.0}),
    ("int vs float", {"a": 1, "b": 1.0}),
    ("large int", {"n": 2 ** 63 + 1}),
    ("booleans and null", {"t": True, "f": False, "n": None}),
    ("string that looks numeric", {"v": "1.0", "w": "01"}),
    ("whitespace in strings", {"s": "a b\tc\nd"}),
    ("quotes and backslashes", {"s": 'he said "\\" once'}),
    ("deep nesting", {"a": {"b": {"c": {"d": [1, {"e": None}]}}}}),
    ("array order is significant", [3, 1, 2]),
    ("duplicate-ish keys after sort", {"a1": 1, "a10": 2, "a2": 3}),
]

print("== the two serializations agree byte for byte ==")
for label, value in CASES:
    a, b = C.canonical_json(value), L.canonical_json(value)
    check("%-30s" % label, a == b)

print("\n== and so do the digests taken over them ==")
for label, value in CASES[:8]:
    check("sha256_hex agrees: %-18s" % label,
          C.sha256_hex(C.canonical_json(value))
          == L.sha256_hex(L.canonical_json(value)))

print("\n== properties the wire format depends on ==")
doc = {"b": 1, "a": {"d": 2, "c": [1, 2]}}
s = C.canonical_json(doc)
check("no whitespace (separators are tight)", " " not in s and "\n" not in s)
check("keys are sorted", s.index('"a"') < s.index('"b"'))
check("non-ascii is NOT escaped (ensure_ascii=False)",
      C.canonical_json({"s": "é"}) == '{"s":"é"}')
check("round-trips through json.loads",
      json.loads(C.canonical_json(doc)) == doc)

print("\n== on real published data, not just synthetic values ==")
root = os.path.join(HERE, "..", "registry", "data")
checked = 0
for name in ("measurements.jsonl", "artifacts.jsonl", "references.jsonl",
             "panels.jsonl"):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        continue
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if C.canonical_json(obj) != L.canonical_json(obj):
            check("row from %s serializes identically" % name, False)
            break
        checked += 1
    else:
        continue
    break
else:
    check("every published registry row serializes identically (%d rows)"
          % checked, checked > 0)

print()
if FAILED:
    print("selftest_canonical_json: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_canonical_json: all passed")
