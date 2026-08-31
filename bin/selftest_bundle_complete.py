#!/usr/bin/env python3
"""Stage ONLY what BUNDLE.txt ships, then run what the instance runs.

`bin/bootstrap_measure.sh` executes five offline selftests during the `setup`
stage, fail-closed under `set -o pipefail`. Those selftests read fixture DATA,
and BUNDLE.txt shipped their code without it. The result was a MiniMax-M3 ROOT
capture whose setup stage died on GGUF test fixtures:

    FileNotFoundError: .../k6/tools/gguf-evidence/manifest.json

with the controller showing nothing but `stage setup` while the instance
billed. The existing checks could not see it: one asserts every BUNDLE.txt
entry EXISTS (they did), another asserts a bundled module's IMPORTS are bundled
(this is data, not an import).

The only honest check is the one this file performs -- build the tree the
instance actually gets, and run in it what the instance actually runs. It costs
seconds locally and it is the difference between finding this here and finding
it after a bootstrap, a fetch and a panel are paid for.

The rule it enforces is NOT "bundle everything": `dione-evidence/` is 187 MB of
fixtures most runs never touch. It is "ship the data, or the reader must
tolerate its absence" -- which is why two small files out of that directory are
bundled and the other 36 are not.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE = Path(__file__).resolve().parent.parent
FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          ("\n        " + detail) if (detail and not ok) else ""))
    if not ok:
        FAILED.append(label)


def bundle_entries():
    text = (SUITE / "bin" / "BUNDLE.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]


def setup_selftests():
    """Exactly what bootstrap_measure.sh runs, read from the script."""
    text = (SUITE / "bin" / "bootstrap_measure.sh").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"(selftest_[a-z0-9_]+\.py)", text)))


print("== the tree an instance actually receives ==")
entries = bundle_entries()
absent = [e for e in entries if not (SUITE / e).is_file()]
check("every BUNDLE.txt entry exists in the repo", not absent, str(absent[:4]))

stage = Path(tempfile.mkdtemp(prefix="fidbundle-"))
try:
    for rel in entries:
        src = SUITE / rel
        if not src.is_file():
            continue
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    staged = sum(1 for _ in stage.rglob("*") if _.is_file())
    check("bundle stages cleanly (%d files)" % staged, staged == len(entries) - len(absent))

    print("\n== and the setup-time selftests RUN in it ==")
    names = setup_selftests()
    check("bootstrap names at least one selftest", bool(names))
    for name in names:
        script = stage / "k6" / "tools" / name
        if not script.is_file():
            # Not bundled at all: bootstrap guards on the file existing, so it
            # is skipped on the instance too. That is consistent, not a gap.
            check("%s is not bundled (bootstrap skips it)" % name, True)
            continue
        proc = subprocess.run([sys.executable, name], cwd=str(script.parent),
                              capture_output=True, text=True, timeout=900)
        out = proc.stdout + proc.stderr
        tail = out.strip().splitlines()[-3:]
        # What this test is for: a file the BUNDLE should have shipped and did
        # not. A missing RUNTIME dependency the instance installs (the
        # quant_pipeline package, passed as --pipeline-root by bootstrap) is a
        # different thing and is not a bundle gap -- conflating the two would
        # make this test unrunnable off an instance, i.e. never run.
        missing = [ln for ln in out.splitlines()
                   if ("FileNotFoundError" in ln or "No such file" in ln)
                   and str(stage) in ln]
        needs_runtime = "--pipeline-root" in out or "quant_pipeline" in out
        if missing:
            check("%s runs from the bundle alone" % name, False,
                  "\n        ".join(missing[:2]))
        elif proc.returncode != 0 and needs_runtime:
            check("%s needs a runtime dep the instance installs (not a bundle gap)"
                  % name, True)
        else:
            check("%s runs from the bundle alone" % name, proc.returncode == 0,
                  "\n        ".join(tail))
finally:
    shutil.rmtree(stage, ignore_errors=True)

print()
if FAILED:
    print("selftest_bundle_complete: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_bundle_complete: all passed")
