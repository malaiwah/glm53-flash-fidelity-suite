#!/usr/bin/env python3
"""Score the finished cold runs into the lane's aggregate KLD receipt.

Called on the instance by `stage_measure.sh score`, between `measure` and
`seal`.  It exists because the cloud recipe used to go straight from capture to
seal -- and `stream_score.py` captures LOGITS, not divergences.  `seal` then
found no `kld-report.json` under `run-*`, said "nothing measured, nothing to
seal", and exited 2 with the whole rental already spent.  The scorer is a
STAGE, with its own marker, so a preemption after it costs nothing.

The argv is composed from `bin/engines.json`'s `scorer` block, in the same way
`invoke_engine.py` composes the capture argv and `measure_local.py` composes
both, so a flag rename is one JSON edit rather than three code edits.
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
from fidelity.engines import load_engines               # noqa: E402
from invoke_engine import engine_python, engine_root    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--receipts", required=True,
                    help="directory holding run-1 .. run-N")
    ap.add_argument("--out", help="aggregate receipt path "
                                  "(default <receipts>/<profile>-packed-kld.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-positions", default="512")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    con = Console()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    engine = load_engines(HERE / "engines.json").get(args.lane)
    if engine is None:
        con.err("no engine configured for lane %r" % args.lane)
        return 3
    scorer = engine.scorer or {}
    if not scorer.get("entrypoint"):
        con.err("lane %r pins no scorer entrypoint in engines.json" % args.lane)
        return 3
    if engine.receipt_class != "submittable":
        con.err("lane %r is not submittable; its preview path is bin/kld_preview.py"
                % args.lane)
        return 3

    fs = os.environ.get("FIDELITY_FS_ROOT", "/home/jl_fs/fidelity")
    suite_root = Path(os.environ.get("FIDELITY_SUITE_ROOT", fs))
    receipts = Path(args.receipts).resolve()
    profile = job.get("profile")
    if not profile:
        con.err("job.json carries no profile -- the scorer's student label "
                "comes from it and must not be guessed")
        return 3

    cold_runs = int(job.get("cold_runs", 1))
    run_dirs = [receipts / ("run-%d" % n) for n in range(1, cold_runs + 1)]
    missing = [str(d) for d in run_dirs
               if not (d / "capture-receipt.json").is_file()]
    if missing:
        con.err("no capture receipt under: %s -- the measure stage did not "
                "finish every cold run" % ", ".join(missing))
        return 2
    if len(run_dirs) < 2:
        con.err("a submittable receipt needs run_count >= 2 (one run cannot "
                "show determinism); job.json asks for %d" % cold_runs)
        return 3

    out = Path(args.out).resolve() if args.out else \
        receipts / ("%s-packed-kld.json" % profile)

    flag = scorer.get("flag_map", {})
    argv = [engine_python(fs), str((suite_root / scorer["entrypoint"]).resolve()),
            flag.get("profile", "--profile"), profile,
            flag.get("panel", "--teacher"), "%s/panel" % fs,
            flag.get("runs", "--runs")] + [str(d) for d in run_dirs] + [
            flag.get("out", "--out"), str(out),
            flag.get("device", "--device"), args.device,
            flag.get("chunk_positions", "--chunk-positions"), args.chunk_positions]
    # Derived from the k6 root the controller sets, not from a JarvisLabs
    # literal. The engine-side twin of this line stalled a paid run at 0% GPU
    # for two hours; this one had the same shape and had simply not been
    # reached yet.
    pipeline_root = os.environ.get("QP_PIPELINE_ROOT") or (
        "%s/pipeline" % engine_root())
    if pipeline_root:
        argv += [flag.get("pipeline_root", "--pipeline-root"), pipeline_root]

    con.say("scorer argv: %s" % " ".join(argv))
    if args.print_only:
        return 0
    proc = run(argv, check=False, env=dict(os.environ), timeout=None)
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode == 0:
        con.ok("aggregate receipt", str(out))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
