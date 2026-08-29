#!/usr/bin/env python3
"""Verify a release's own SHA256SUMS against the files it actually published.

Why this is not `sha256sum -c`
------------------------------
`sha256sum -c` answers "does every line of this list check out", and for a
MIRROR that is the wrong question. Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw
republishes brandonmusic's 120 weight shards byte-for-byte, trims his 120
`.materialization/shards/*.json` sidecars, ships its own README and LICENSE --
and copies his SHA256SUMS verbatim. `-c` therefore reports 122 failures, none
of which is a weight, and exits non-zero. Under `set -o pipefail` that killed
the fetch stage after a 175 GB download and a full checksum pass.

The question worth answering is narrower and stronger:

  * every file PRESENT on disk that the list covers must match its digest --
    and for `*.safetensors` that is fail-closed, exit 2;
  * every `*.safetensors` on disk must be COVERED by the list, so a weight
    cannot pass by being unlisted;
  * entries naming files this repo does not publish are REPORTED as
    `listed_absent`, with their names, because "the mirror trimmed the
    producer's sidecars" and "a shard is missing" must never look alike;
  * non-weight files that are present but differ are REPORTED as
    `nonweight_mismatch` and do not fail the stage -- a mirror is expected to
    write its own README.

Exit 0 = every published weight verified. Exit 2 = a weight failed, or a
weight on disk is not covered by the list. Exit 3 = no SHA256SUMS to read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_sums(path: Path):
    """Read a coreutils-style checksum list: '<64 hex>  <name>' (or ' *name')."""
    entries = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        entries[parts[1].strip().lstrip("*")] = parts[0].lower()
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sums", default="SHA256SUMS")
    ap.add_argument("--out", help="write the JSON report here as well as stdout")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sums = root / args.sums
    if not sums.is_file():
        print("no %s in %s" % (args.sums, root), file=sys.stderr)
        return 3
    listed = parse_sums(sums)

    weights_ok, weights_bad = [], []
    other_ok, other_bad, absent = [], [], []
    for name, want in sorted(listed.items()):
        path = root / name
        is_weight = name.endswith(WEIGHT_SUFFIXES)
        if not path.is_file():
            absent.append(name)
            continue
        got = sha256_file(path)
        if got == want:
            (weights_ok if is_weight else other_ok).append(name)
        elif is_weight:
            weights_bad.append({"file": name, "want": want, "got": got})
        else:
            other_bad.append(name)

    on_disk = sorted(p.name for p in root.iterdir()
                     if p.is_file() and p.name.endswith(WEIGHT_SUFFIXES))
    uncovered = [n for n in on_disk if n not in listed]

    report = {
        "schema": "malaiwah.published-sums-verification.v1",
        "root": str(root),
        "sums_file": args.sums,
        "sums_sha256": sha256_file(sums),
        "listed_entries": len(listed),
        "weights_verified": len(weights_ok),
        "weights_failed": weights_bad,
        "weights_on_disk": len(on_disk),
        "weights_not_covered_by_list": uncovered,
        "nonweight_verified": len(other_ok),
        "nonweight_mismatch": other_bad,
        "listed_absent": absent,
        "verdict": ("every published weight matches the release's own SHA256SUMS"
                    if not weights_bad and not uncovered
                    else "WEIGHT VERIFICATION FAILED"),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    if weights_bad:
        print("FAILED: %d weight file(s) do not match their published digest: %s"
              % (len(weights_bad), [b["file"] for b in weights_bad][:3]), file=sys.stderr)
        return 2
    if uncovered:
        print("FAILED: %d weight file(s) on disk are not covered by %s: %s"
              % (len(uncovered), args.sums, uncovered[:3]), file=sys.stderr)
        return 2
    if absent or other_bad:
        print("note: %d listed entr(y/ies) name files this repo does not publish, "
              "and %d non-weight file(s) differ. Neither is a weight failure; both "
              "are recorded in the report." % (len(absent), len(other_bad)),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
