#!/usr/bin/env python3
"""fixture -- fetch the 0.1B CI fixture (idempotent, cached by commit).

    bin/fixture              # fetch if absent, print the local path
    bin/fixture --print      # print the path without fetching (may not exist)

The fixture (inference-optimization/GLM-5.3-Flash-0.1B-A0.1B) is an
architecturally complete 0.1B GLM-5.3-Flash: 45 layers' worth of structure at
toy width, so the WHOLE chain -- native source, streaming build, capture,
scoring -- runs in minutes on a laptop.  It certifies plumbing and estimator
correctness; it can NOT certify the real model's KLD distribution or wall
clock (its tail is a different animal), and nothing here pretends otherwise.

Cached under FIDELITY_CACHE_DIR (default ~/.cache/glm53-fidelity)/fixture/
<repo>/<40-hex commit>/ so re-runs cost one metadata request.  Unauthenticated
unless the environment already carries an HF token (the repo is public).
The LAST line of stdout is the fixture path -- callers parse exactly that.

Stock python3.9 stdlib.
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import Console, human_bytes         # noqa: E402
from fidelity.hfmeta import HF_ENDPOINT, HFError, repo_meta  # noqa: E402
from fidelity.registry_client import cache_dir           # noqa: E402

FIXTURE_REPO = "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B"
EXIT_OK, EXIT_REFUSED, EXIT_NO_SOURCE = 0, 3, 4


def fixture_root(repo: str, revision: str) -> Path:
    return cache_dir() / "fixture" / repo.replace("/", "__") / revision


def fetch(repo: str, con: Console) -> Path:
    meta = repo_meta(repo, "model", "main")
    dest = fixture_root(repo, meta.revision)
    con.say("fixture %s @ %s" % (repo, meta.revision[:12]))
    con.say("  %d files, %s total" % (len(meta.files),
                                      human_bytes(meta.total_bytes)))
    dest.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    for name, size in meta.files:
        target = dest / name
        if target.is_file() and target.stat().st_size == size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = "%s/%s/resolve/%s/%s" % (
            HF_ENDPOINT, repo, meta.revision, urllib.parse.quote(name))
        req = urllib.request.Request(url, headers={"User-Agent": "fidelity-suite/0.1"})
        tmp = target.with_name(target.name + ".part")
        with urllib.request.urlopen(req, timeout=600) as resp, \
                open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(target)
        got = target.stat().st_size
        if size and got != size:
            target.unlink()
            raise HFError("size mismatch fetching %s (%d != %d)"
                          % (name, got, size))
        con.say("  fetched %-52s %s" % (name, human_bytes(got)))
        fetched += 1
    con.say("  %d fetched, %d already present (idempotent by size at a "
            "pinned commit)" % (fetched, skipped))
    return dest


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fixture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=FIXTURE_REPO)
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="print the newest cached path; fetch nothing")
    args = p.parse_args(argv)
    con = Console(stream=sys.stderr)
    if args.print_only:
        base = cache_dir() / "fixture" / args.repo.replace("/", "__")
        candidates = sorted(base.glob("*")) if base.is_dir() else []
        if not candidates:
            con.err("no cached fixture under %s (run bin/fixture to fetch)" % base)
            return EXIT_REFUSED
        print(candidates[-1])
        return EXIT_OK
    try:
        dest = fetch(args.repo, con)
    except HFError as exc:
        con.err("cannot fetch the fixture: %s" % exc)
        con.err("remedies: restore network access, or copy a fixture tree "
                "manually and pass its path to --fixture")
        return EXIT_NO_SOURCE
    print(dest)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
