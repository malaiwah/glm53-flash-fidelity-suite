#!/usr/bin/env python3
"""Transport a published token panel subtree into `engines/panels/`, byte-exact.

    engines/tools/transport_token_panel.py \
        --repo brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
        --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa \
        --subtree calibration/panel-v1 \
        --exclude calibration.sealed-corpus.json --exclude corpus.receipt.json \
        --out engines/panels/panel--glm53.brandonmusic.final25

Why this exists: a same-lane root for a model family that already has rows on
somebody else's panel must be captured on THAT panel, or the new number differs
from the old ones by panel *and* lane with no way to separate them
(`docs/M1-QWEN38-ROOT-LEARNINGS.md`, learning 14). The token ids are
transported; nothing is re-tokenized.

What it does, and refuses:
  * lists the subtree at the pinned 40-hex revision through the Hub tree API
    (anonymous), fetches every regular file at that revision, and refuses any
    file whose bytes do not match the listing's digest (LFS sha256, or the git
    blob sha1 of a non-LFS file) and size;
  * writes the files under --out, refusing to overwrite a non-empty directory;
  * excludes only the names given by --exclude (the sealed panel closure that
    `bin/fidelity/panel.resolve_panel` accepts is the artifact receipt's file
    list, and a sibling that is not in it refuses the whole panel);
  * writes `<out>.provenance.json` BESIDE the directory (inside it would be an
    extra file and refuse the closure) with the source repo, revision, every
    file's path/bytes/sha256, the excluded names, and the fetch time.

Stdlib only: this runs on the controller before any spend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

HUB = "https://huggingface.co"
_REVISION40 = re.compile(r"^[0-9a-f]{40}$")


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-fidelity-suite/transport_token_panel"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_subtree(repo: str, revision: str, subtree: str, repo_type: str = "dataset"):
    prefix = {"dataset": "datasets/", "model": ""}[repo_type]
    url = "%s/api/%s%s/tree/%s/%s?recursive=true" % (HUB, prefix, repo, revision, subtree)
    rows = json.loads(_get(url).decode("utf-8"))
    files = []
    for row in rows:
        if row.get("type") != "file":
            continue
        path = PurePosixPath(row["path"])
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise SystemExit("REFUSED: unsafe path in Hub listing: %r" % row["path"])
        lfs = row.get("lfs") or {}
        files.append({
            "path": row["path"],
            "size": int(lfs.get("size", row["size"])),
            "sha256": lfs.get("oid"),
            "git_oid": row.get("oid"),
        })
    return files


def verify(raw: bytes, entry) -> bool:
    if len(raw) != entry["size"]:
        return False
    if entry["sha256"]:
        return hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == entry["git_oid"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="40-hex commit; a branch name is refused")
    parser.add_argument("--repo-type", default="dataset", choices=("dataset", "model"))
    parser.add_argument("--subtree", required=True, help="directory inside the repo, e.g. calibration/panel-v1")
    parser.add_argument("--exclude", action="append", default=[],
                        help="file name (relative to the subtree) NOT to transport; repeatable")
    parser.add_argument("--out", required=True, help="panel directory to create")
    args = parser.parse_args(argv)

    if not _REVISION40.fullmatch(args.revision):
        raise SystemExit("REFUSED: --revision must be a 40-hex commit, got %r" % args.revision)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("REFUSED: %s exists and is not empty" % out)
    subtree = args.subtree.strip("/")
    listing = list_subtree(args.repo, args.revision, subtree, args.repo_type)
    if not listing:
        raise SystemExit("REFUSED: %s@%s has no files under %s" % (args.repo, args.revision, subtree))
    excluded = set(args.exclude)
    seen_excluded = set()
    prefix = {"dataset": "datasets/", "model": ""}[args.repo_type]
    manifest = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for entry in sorted(listing, key=lambda row: row["path"]):
        rel = entry["path"][len(subtree) + 1:] if entry["path"].startswith(subtree + "/") else entry["path"]
        if rel in excluded:
            seen_excluded.add(rel)
            continue
        raw = _get("%s/%s%s/resolve/%s/%s" % (HUB, prefix, args.repo, args.revision, entry["path"]))
        if not verify(raw, entry):
            raise SystemExit("REFUSED: %s does not match the Hub listing's digest/size at %s"
                             % (entry["path"], args.revision))
        target = out.joinpath(*PurePosixPath(rel).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest.append({"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                         "source_path": entry["path"]})
    missing_excludes = sorted(excluded - seen_excluded)
    if missing_excludes:
        raise SystemExit("REFUSED: --exclude names not in the subtree: %s" % ", ".join(missing_excludes))
    provenance = {
        "schema": "malaiwah.transported-token-panel.v1",
        "source": {"repository": args.repo, "repo_type": args.repo_type,
                   "revision": args.revision, "subtree": subtree},
        "transported_at_utc": started,
        "tool": "engines/tools/transport_token_panel.py",
        "verification": ("every file fetched at the pinned revision and checked against the Hub "
                         "tree listing: LFS sha256 + size, or git blob sha1 + size for non-LFS files"),
        "excluded": sorted(seen_excluded),
        "files": manifest,
        "file_count": len(manifest),
        "total_bytes": sum(row["bytes"] for row in manifest),
        "manifest_sha256": hashlib.sha256(json.dumps(
            [{"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]} for r in manifest],
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }
    sidecar = out.parent / (out.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("transported %d files (%d bytes) from %s@%s/%s -> %s; provenance %s"
          % (len(manifest), provenance["total_bytes"], args.repo, args.revision[:8], subtree,
             out, sidecar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
