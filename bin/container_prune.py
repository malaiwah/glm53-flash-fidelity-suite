#!/usr/bin/env python3
"""Reduce a staged tree to exactly what `bin/BUNDLE.txt` lists.

    container_prune.py --stage /tmp/stage --out /opt/fidelity/suite

`bin/BUNDLE.txt` exists so that what lands on rented hardware is auditable and
reviewable rather than "whatever happened to be in the directory" -- which is
how a stray note, a scratch receipt, or somebody's token file ends up on
somebody else's machine.  A container image is a second transport for that
same set, so it reads the same list instead of keeping a parallel one.  This
is not only hygiene: `engines/tools/` is 208 MB in this checkout and 187 MB of that
is one evidence directory nothing in the bundle references.

Two files are added to the set on purpose: the entrypoint and the stage-
sequence rule it imports.  They post-date the SSH-era list and a stage run
from inside the run root would otherwise import a file that is not there.

Missing entries are SKIPPED and the skipping is printed -- BUNDLE.txt's own
stated policy, so a lane whose engine is absent from this checkout does not
break the build.

`--require-all` turns that policy off, and the image build passes it, because
inside a build the two reasons an entry can be missing are NOT the same thing.
Upstream, "absent" means the caller's checkout does not have that lane. In a
build it means the Dockerfile did not COPY the directory the entry lives in --
and fail-open there ships an image that looks fine and dies in the `capture`
stage on rented hardware. That is not hypothetical: docs/schema/ was in
BUNDLE.txt and the Dockerfile COPYed four trees that did not include docs/, so
a containerised root capture got all the way to validating the manifest it had
just written and raised

    FileNotFoundError: .../fidelity/docs/schema

after the bootstrap, the fetch and the capture were all paid for.

Stdlib only, python3.9-clean.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

EXTRA = ("bin/container_entry.py", "bin/container_prune.py",
         "bin/container_manifest.py", "bin/fidelity/stages.py")


def entries(stage: Path):
    text = (stage / "bin" / "BUNDLE.txt").read_text(encoding="utf-8")
    names = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.startswith("#")]
    for extra in EXTRA:
        if extra not in names:
            names.append(extra)
    # BUNDLE.txt is the audited set, and this file is part of the audit trail:
    # an image that dropped it could not be re-pruned or explained.
    if "bin/BUNDLE.txt" not in names:
        names.append("bin/BUNDLE.txt")
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-all", action="store_true",
                    help="refuse if any BUNDLE.txt entry is absent from the "
                         "staged tree (what a container build wants: an absent "
                         "entry there means a missing COPY, not a missing lane)")
    args = ap.parse_args(argv)
    stage, out = Path(args.stage), Path(args.out)

    kept = skipped = 0
    missing = []
    for rel in entries(stage):
        src = stage / rel
        if not src.is_file():
            print("skipped (not in this checkout): %s" % rel)
            missing.append(rel)
            skipped += 1
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        kept += 1
    # The stage scripts are executed, not imported.
    for rel in ("bin/stage_measure.sh", "bin/bootstrap_measure.sh", "bin/watchdog.sh"):
        path = out / rel
        if path.is_file():
            os.chmod(str(path), 0o755)
    print("bundle: %d file(s) kept, %d skipped -> %s" % (kept, skipped, out))
    if missing and args.require_all:
        sys.stderr.write(
            "REFUSED: %d BUNDLE.txt entr%s did not reach the staged tree:\n"
            % (len(missing), "y" if len(missing) == 1 else "ies"))
        for rel in missing:
            sys.stderr.write("  %s\n" % rel)
        sys.stderr.write(
            "  In a container build this means the Dockerfile did not COPY the\n"
            "  directory it lives in (or .dockerignore excluded it). Shipping\n"
            "  anyway produces an image that dies mid-stage on rented hardware.\n")
        return 3
    if kept == 0:
        sys.stderr.write("nothing was kept; --stage %s does not look like a "
                         "suite checkout\n" % stage)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
