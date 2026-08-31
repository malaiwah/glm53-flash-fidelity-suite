#!/usr/bin/env python3
"""P1-08 adversarial cases: NaN/Infinity refused at ingest, seal, and render.

Before the fix: the minischema treated any Python float as a number and every
bound check failed open on NaN (all comparisons with NaN are False), so a
measurement with metric.value = NaN validated with ZERO errors; json.load
accepted the non-RFC tokens; and both canonical serializers emitted them,
sealing non-standard bytes under a sha256. Verified to FAIL pre-fix: the
minischema case reports 0 errors and the serializer cases emit tokens there.

Stock python3, offline, no installs.
"""
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "registry", "tools"))

from fidelity import common as C   # noqa: E402
import _minischema                 # noqa: E402
import registry_add as A           # noqa: E402
import registry_lib as L           # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))
    if not ok:
        failures.append(name)


ROOT = os.path.join(HERE, "..", "registry")

# --- 1. schema validation: NaN can no longer pass a bound check ---------------
reg = _minischema.Registry(os.path.join(ROOT, "schema"))
with open(os.path.join(ROOT, "data", "measurements.jsonl"), encoding="utf-8") as fh:
    row = json.loads(fh.readline())
errs = reg.validate(row, "measurement.schema.json")
check("baseline: a published row validates clean", not errs,
      "; ".join(str(e) for e in errs[:3]))

bad = json.loads(json.dumps(row))
bad["metric"]["value"] = float("nan")
errs = reg.validate(bad, "measurement.schema.json")
check("metric.value = NaN is a schema ERROR (used to fail open)", bool(errs))

bad = json.loads(json.dumps(row))
bad["auxiliary_metrics"]["adversarial"] = [1.0, {"deep": float("inf")}]
errs = reg.validate(bad, "measurement.schema.json")
check("Infinity under a permissive subtree is still caught (recursive walk)", bool(errs))

# --- 2. ingestion: the non-RFC tokens are refused at the parse ----------------
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "receipt.json")
    with open(p, "w") as fh:
        fh.write('{"schema": "%s", "mean_kld": NaN, "direction": "teacher_to_student"}'
                 % "glm53flash-crosscheck/2")
    try:
        A.load_receipt(p)
        check("registry_add.load_receipt refuses a NaN token", False, "accepted")
    except A.Refuse as e:
        check("registry_add.load_receipt refuses a NaN token", True)
    except ValueError:
        check("registry_add.load_receipt refuses a NaN token", True, "raw ValueError")

    p2 = os.path.join(td, "rows.jsonl")
    with open(p2, "w") as fh:
        fh.write('{"id": "x", "v": Infinity}\n')
    try:
        L.read_jsonl(p2)
        check("registry_lib.read_jsonl (render path) refuses an Infinity token", False)
    except ValueError as e:
        check("registry_lib.read_jsonl (render path) refuses an Infinity token", True)

    try:
        C.read_json(p)
        check("common.read_json refuses a NaN token", False)
    except ValueError:
        check("common.read_json refuses a NaN token", True)

# --- 3. sealing: a non-finite value cannot be sealed or emitted ---------------
try:
    C.seal({"schema": "x/1", "v": float("nan")})
    check("common.seal refuses NaN", False, "sealed it")
except ValueError:
    check("common.seal refuses NaN", True)

try:
    C.verify_seal({"receipt_sha256": "0" * 64, "v": float("inf")})
    check("common.verify_seal refuses Infinity rather than verifying", False)
except ValueError:
    check("common.verify_seal refuses Infinity rather than verifying", True)

with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "r.json")
    try:
        C.write_json(out, {"v": float("nan")})
        check("common.write_json refuses NaN", False, "wrote it")
    except ValueError:
        check("common.write_json refuses NaN", True)
    check("...and leaves no file behind", not os.listdir(td), repr(os.listdir(td)))

# --- 4. the published data itself is finite (the guard is not vacuous) --------
def all_finite(o):
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(all_finite(v) for v in o.values())
    if isinstance(o, list):
        return all(all_finite(v) for v in o)
    return True

n = 0
for name in ("measurements", "artifacts", "panels", "references", "pipelines", "models"):
    for _, obj, _ in L.read_jsonl(os.path.join(ROOT, "data", name + ".jsonl")):
        n += 1
        if not all_finite(obj):
            check("published row %s is finite" % obj.get("id"), False)
check("every published record parses under the strict rules and is finite (%d)" % n, n > 0)

print()
if failures:
    print("selftest_nonfinite_rejection: FAILED: %s" % ", ".join(failures))
    sys.exit(1)
print("selftest_nonfinite_rejection: all checks passed")
