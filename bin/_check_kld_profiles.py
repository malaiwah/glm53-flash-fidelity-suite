#!/usr/bin/env python3
"""Every literal `--profile` a shell script hands to k6_kld_report.py must be a choice.

Test-support for bin/selftest_shell_guards.sh (NUM-10). `k6/stage_k6.sh` documented
`QP_STREAM_PROFILE=k6|k8` and composed `--profile "${STREAM_PROFILE}-stream"`, but
`k8-stream` is not one of k6_kld_report's argparse choices -- so a K8 streaming run
exited 2 AFTER the whole multi-hour capture. `k6/tools/measure_dione.sh` passed the
equally non-existent `--profile dione`.

Only `k6_kld_report.py` invocations are inspected: `k6_driver.py` and
`k6_student_capture.py` have their own, different `--profile` vocabularies.

A value that interpolates a shell variable cannot be checked here; it is reported as
UNCHECKABLE so a reader knows the gate did not cover it, and the script itself is
expected to refuse an unsupported value up front.

    _check_kld_profiles.py <suite-root>       -> 0 clean, 1 on a bad profile
"""

import re
import sys
from pathlib import Path

CHOICES = re.compile(r'--profile",\s*required=True,\s*\n?\s*choices=\(([^)]*)\)', re.S)
# Shell line continuations are joined first, so a flag and its value can be on
# different physical lines (they usually are).
CONTINUED = re.compile(r"\\\n\s*")
INVOKE = re.compile(r"k6_kld_report\.py(?P<args>.*)")
PROFILE = re.compile(r"--profile\s+\"?([^\s\"]+)\"?")


def main(argv):
    if len(argv) != 2:
        sys.stderr.write(__doc__)
        return 2
    root = Path(argv[1])
    report = root / "k6" / "tools" / "k6_kld_report.py"
    found = CHOICES.search(report.read_text(encoding="utf-8"))
    if not found:
        sys.stderr.write("could not read --profile choices from %s\n" % report)
        return 2
    choices = set(re.findall(r'"([^"]+)"', found.group(1)))

    bad, uncheckable = [], []
    for script in sorted((root / "k6").rglob("*.sh")) + sorted((root / "bin").rglob("*.sh")):
        text = CONTINUED.sub(" ", script.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            # A comment that NAMES the defect (as stage_k6.sh's NUM-10 refusal does)
            # is not an invocation. Only whole-line comments are dropped: `${x#y}` is
            # a parameter expansion, not a comment, so `#` mid-line stays.
            if line.lstrip().startswith("#"):
                continue
            call = INVOKE.search(line)
            if not call:
                continue
            value = PROFILE.search(call.group("args"))
            if not value:
                continue
            profile = value.group(1)
            if "$" in profile:
                uncheckable.append("%s: %s" % (script.name, profile))
            elif profile not in choices:
                bad.append("%s: --profile %s (choices: %s)"
                           % (script.name, profile, ", ".join(sorted(choices))))
    for row in uncheckable:
        print("UNCHECKABLE (shell variable): %s" % row)
    for row in bad:
        print("NOT A CHOICE: %s" % row)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
