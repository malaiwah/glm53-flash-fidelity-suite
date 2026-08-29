#!/usr/bin/env python3
"""Put the token-panel's artifacts where its own receipt says they are.

  stage_panel_paths.py --panel <dir> [--receipt <path>] [--check-only]

The sealed token-panel receipt lists every artifact it depends on by ABSOLUTE
path, byte count and sha256, and the pipeline's `verified_artifacts` requires
each one to exist at that exact path, be a regular file (a symlinked LEAF is
rejected outright), and match both the size and the digest.  Those paths are
the PRODUCER's -- `/workspace/artifacts/dataset/calibration/...` -- and a fresh
instance has nothing there.

On the campaign machines that produced the sealed rows the tree happened to
live at that path, so nobody noticed.  On a cold box the capture gets as far as
loading the panel and dies with `artifact identity mismatch`, after the
165 GB fetch, the materialize and the model load.

This stages the fetched copies into the receipt's own paths and verifies each
by digest as it goes, so a mismatch is reported here -- by name -- instead of
as one opaque line four stages later.  Files are COPIED, not linked: the
verifier rejects a symlinked leaf, and 5.8 MB is not worth being clever about.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, help="the fetched panel tree")
    ap.add_argument("--receipt", help="token-panel receipt (default <panel>/token-panel-receipt.json)")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    panel = Path(args.panel).resolve()
    receipt_path = Path(args.receipt) if args.receipt else panel / "token-panel-receipt.json"
    if not receipt_path.is_file():
        print("stage_panel_paths: no token-panel receipt at %s" % receipt_path, file=sys.stderr)
        return 2
    rows = json.loads(receipt_path.read_text(encoding="utf-8")).get("artifacts")
    if not isinstance(rows, list) or not rows:
        print("stage_panel_paths: receipt has no artifacts list", file=sys.stderr)
        return 2

    staged = already = 0
    unresolved = []
    for row in rows:
        want = Path(row["path"])
        digest = row["sha256"]
        if want.is_file() and not want.is_symlink() \
                and want.stat().st_size == int(row["bytes"]) \
                and sha256_file(want) == digest:
            already += 1
            continue
        # The receipt's paths are rooted at the producer's dataset directory;
        # the fetched repo mirrors everything from `calibration/` down.
        if "/calibration/" not in str(want):
            unresolved.append((str(want), "path is not under a calibration/ root"))
            continue
        src = panel / "calibration" / str(want).split("/calibration/", 1)[1]
        if not src.is_file():
            unresolved.append((str(want), "not in the fetched panel at %s" % src))
            continue
        if sha256_file(src) != digest:
            unresolved.append((str(want), "fetched copy has a different digest"))
            continue
        if args.check_only:
            unresolved.append((str(want), "absent (check-only)"))
            continue
        want.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, want)
        os.chmod(want, 0o644)
        if sha256_file(want) != digest:
            unresolved.append((str(want), "copy did not verify"))
            continue
        staged += 1

    print(json.dumps({"artifacts": len(rows), "already_present": already,
                      "staged": staged, "unresolved": len(unresolved)},
                     sort_keys=True))
    if unresolved:
        for path, why in unresolved[:6]:
            print("  UNRESOLVED %s -- %s" % (path, why), file=sys.stderr)
        if len(unresolved) > 6:
            print("  ... and %d more" % (len(unresolved) - 6), file=sys.stderr)
        print("stage_panel_paths: the panel receipt's artifacts cannot all be "
              "satisfied; the capture would die at load_panel_windows",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
