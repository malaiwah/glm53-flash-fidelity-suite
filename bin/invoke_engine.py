#!/usr/bin/env python3
"""Run one cold capture, using the lane's pinned engine invocation.

Called on the instance by `stage_measure.sh measure`.  It exists so the argv is
built from `bin/engines.json` in exactly one place -- the same place the local
runner and the cloud planner read -- rather than being spelled out in a shell
script where a drifted flag would only show up at hour three of a rental.

A lane whose engine is unpinned exits 3 with the contract printed, which is the
same refusal `--dry-run` gives on the caller's laptop.  It should never get this
far: the planner refuses before creating anything.  This is the backstop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fidelity.common import Console, run                # noqa: E402
from fidelity.engines import EngineUnpinned, build_invocation, load_engines  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="job.json written by the controller")
    ap.add_argument("--lane", required=True)
    ap.add_argument("--cold-run", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-only", action="store_true",
                    help="print the argv that would run, and exit")
    args = ap.parse_args()

    con = Console()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    engines = load_engines(HERE / "engines.json")
    engine = engines.get(args.lane)
    if engine is None:
        con.err("no engine configured for lane %r" % args.lane)
        return 3

    fs = os.environ.get("FIDELITY_FS_ROOT", "/home/jl_fs/fidelity")
    try:
        argv = build_invocation(
            engine,
            suite_root=Path(os.environ.get("FIDELITY_SUITE_ROOT", fs)),
            checkpoint="%s/models/target" % fs,
            panel_dir="%s/panel" % fs,
            out_dir=args.out,
            surface=job.get("target", {}).get("surface", "packed"),
            profile=job.get("profile", "k6"),
            cold_run=args.cold_run,
            reduce_order=job.get("reduce_order", "fp32"),
            roles=job.get("panel", {}).get("roles", "final"),
            extra={
                "bf16": os.environ.get("BF16", "/home/jl_fs/models/bf16"),
                "pipeline_root": os.environ.get(
                    "QP_PIPELINE_ROOT", "/home/jl_fs/glm53-k6/pipeline"),
            },
        )
    except EngineUnpinned as exc:
        con.err(str(exc))
        return 3

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (engine.env or {}).items()})
    con.say("engine argv: %s" % " ".join(argv))
    if args.print_only:
        return 0
    proc = run(argv, check=False, env=env, timeout=None)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
