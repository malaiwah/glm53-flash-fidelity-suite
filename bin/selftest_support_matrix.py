#!/usr/bin/env python3
"""Selftest: the README support matrix is generated, current, and drift-detected.

The defect this guards against is the one the contributor-experience review
found: hand-written support claims. README said GGUF was unreachable while
`bin/engines.json` and docs/GGUF-MEASUREMENT.md said it was wired — because the
README table was typed by hand and the data moved. The fix is the registry's
render-drift pattern: `bin/render_support_matrix.py` renders the block from
`bin/engines.json`, README.md carries it between markers, and this test fails
whenever the two disagree.

Rungs:
  M1  README carries exactly one marker pair.
  M2  --check passes against the committed README (no drift).
  M3  --check FAILS on a tampered copy (the drift detector actually bites;
      this rung fails if someone neuters the comparison).
  M4  every lane and every surface in engines.json appears in the block.
  M5  no OTHER document restates a per-lane surface list by hand: the two
      known offenders (bin/README.md, registry/CONTRIBUTING.md) must link to
      the README matrix rather than carry their own copy of the stale claim.

Stock python3, stdlib only, offline, $0.00.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent.parent
BIN = SUITE_ROOT / "bin"
README = SUITE_ROOT / "README.md"
BEGIN = "<!-- BEGIN GENERATED: support-matrix -->"
END = "<!-- END GENERATED: support-matrix -->"

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


def run_check(readme_path):
    return subprocess.run(
        [sys.executable, str(BIN / "render_support_matrix.py"),
         "--check", "--readme", str(readme_path)],
        capture_output=True, text=True)


def main():
    text = README.read_text(encoding="utf-8")

    # M1: exactly one marker pair.
    check("M1: exactly one BEGIN/END marker pair in README",
          text.count(BEGIN) == 1 and text.count(END) == 1,
          "BEGIN x%d END x%d" % (text.count(BEGIN), text.count(END)))

    # M2: no drift.
    proc = run_check(README)
    check("M2: rendered block matches a fresh render (no drift)",
          proc.returncode == 0, (proc.stderr or proc.stdout).strip()[:200])

    # M3: the detector bites. Tamper with one cell inside the generated block
    # of a COPY and assert --check fails. Without this rung, a broken
    # comparison (always-pass) would go unnoticed.
    block_match = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END),
                            text, re.S)
    tampered = None
    if block_match and "✓" in block_match.group(1):
        inner = block_match.group(1).replace("✓", "—", 1)
        tampered = text[:block_match.start(1)] + inner + text[block_match.end(1):]
    if tampered is None:
        check("M3: drift detector fails on a tampered copy", False,
              "could not build a tampered copy (no ✓ cell in the block?)")
    else:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "README.md"
            bad.write_text(tampered, encoding="utf-8")
            proc = run_check(bad)
        check("M3: drift detector fails on a tampered copy",
              proc.returncode != 0, "tampered README passed --check")

    # M4: coverage — every lane and surface in the data appears in the block.
    data = json.loads((BIN / "engines.json").read_text(encoding="utf-8"))
    block = block_match.group(1) if block_match else ""
    missing = [n for n in data["lanes"] if ("`%s`" % n) not in block]
    surfaces = {s for spec in data["lanes"].values()
                for s in (spec.get("surfaces") or [])}
    missing += [s for s in sorted(surfaces) if ("`%s`" % s) not in block]
    check("M4: every lane and surface in engines.json appears in the block",
          not missing, "missing: %s" % ", ".join(missing))

    # M5: single source of truth — the known former offenders must not carry
    # a hand-written per-lane surface list any more. The tell is the exact
    # stale sentence shape both used: a lane name followed by a literal
    # enumeration `packed`, `native-bf16` presented as that lane's full list.
    for rel in ("bin/README.md", "registry/CONTRIBUTING.md"):
        doc = (SUITE_ROOT / rel).read_text(encoding="utf-8")
        # A hand-maintained enumeration of what "the cloud/streaming lane
        # reads" is the drift-prone pattern; a link to the README matrix is
        # the accepted form.
        hand_written = re.search(
            r"(cloud lane|streaming lane)[^.\n]{0,40}reads[^.]{0,120}`packed`",
            doc)
        links = ("support matrix" in doc) or ("support-matrix" in doc) \
            or ("Before you rent" in doc)
        check("M5: %s links to the matrix instead of restating it" % rel,
              (hand_written is None) and links,
              "hand-written lane/surface claim found" if hand_written
              else "no link to the README matrix")

    print()
    if failures:
        print("selftest_support_matrix: %d FAILED" % len(failures))
        return 1
    print("selftest_support_matrix: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
