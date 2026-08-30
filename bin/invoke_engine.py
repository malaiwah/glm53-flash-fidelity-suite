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
from fidelity.engines import (                          # noqa: E402
    EngineUnpinned, build_invocation, fidelity_python, load_engines,
)


def engine_python(fs: str) -> str:
    """The torch-capable interpreter the engine script runs under.

    build_invocation returns `[<launcher...>, <script.py>, ...]`; the engine
    scripts are mode 644 with an `env python3` shebang, so the argv MUST be
    prefixed with an interpreter, and it must be the venv's -- the stage
    driver calls THIS file with the system python3 (it has to: it runs before
    the venv exists), so sys.executable is the wrong answer here.
    measure_local's execute path does the same thing one line further down.
    """
    env = os.environ.get("FIDELITY_ENGINE_PYTHON")
    if env:
        return env
    for candidate in (
        os.environ.get("VENV") and "%s/bin/python" % os.environ["VENV"],
        "%s/venv/bin/python" % os.environ.get("FIDELITY_K6_ROOT", "/home/jl_fs/glm53-k6"),
        "%s/venv/bin/python" % fs,
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return fidelity_python()


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
    target = job.get("target", {}) or {}
    surface = target.get("surface", "packed")
    # The streaming-family engines spell the input mode --source (vocabulary:
    # checkpoint | payload-store | dione | native | exl3hf), not --surface.
    # measure_local's own execute path sets source="checkpoint" for a
    # materialized target tree; mirror that here so the backstop composes the
    # same argv.  Lanes whose flag_map has no "source" key (sealed-ep8) ignore
    # it.  A surface with no mapping yields "" -- build_invocation then DROPS
    # the flag and the engine falls back to its own default, which is how an
    # unmapped surface used to reach the GPU and die on argparse an hour into
    # a rental.  Refuse here instead: a surface this file cannot spell is a
    # missing mapping, not a default.
    source_by_surface = {
        "packed": "checkpoint",
        "native-bf16": "native",
        "exl3hf": "exl3hf",
        "tr3-published": "tr3",
        "dione": "dione",
    }
    if "source" in (engine.flag_map or {}) and surface not in source_by_surface:
        con.err(
            "no --source spelling for surface %r on lane %r: add it to "
            "invoke_engine.source_by_surface (the engine would otherwise run "
            "with its default source and fail after the fetch)" % (surface, args.lane)
        )
        return 3
    source = source_by_surface.get(surface, "")

    # exl3hf: the measured non-routed function is the ARTIFACT's own, so
    # --bf16 must point at the tree `stage_measure.sh materialize` wrote from
    # this same snapshot -- never at the official BF16 metadata skeleton.
    extra = {
        "source": source,
        "bf16": os.environ.get("BF16", "/home/jl_fs/models/bf16"),
        "pipeline_root": os.environ.get(
            "QP_PIPELINE_ROOT", "/home/jl_fs/glm53-k6/pipeline"),
    }
    if surface == "exl3hf":
        materialized = os.environ.get(
            "EXL3HF_BF16", "%s/models/target-bf16-materialized" % fs)
        extra.update({
            "bf16": materialized,
            "exl3hf_root": "%s/models/target" % fs,
            "exl3hf_repo": target.get("repo_id", ""),
            "exl3hf_revision": target.get("revision", ""),
        })
        missing = [k for k in ("exl3hf_repo", "exl3hf_revision") if not extra[k]]
        if missing:
            con.err("job.json target is missing %s -- an exl3hf capture cannot "
                    "seal its provenance without them" % ", ".join(missing))
            return 3
    elif surface == "tr3-published":
        # A TR3 release quantizes the routed experts ONLY and ships all 1,618
        # non-routed tensors as the OFFICIAL source tensors, in-repo, under
        # their official names.  The engine therefore REFUSES --bf16 outright:
        # a second tree would make it ambiguous which non-routed weights were
        # measured.  Blanking the value is what drops the flag
        # (build_invocation skips None/"" -- see fidelity/engines.py).
        # --bf16 is the tree `stage_measure.sh materialize` wrote from THIS
        # snapshot -- the same contract exl3hf has. For a TR3 release the
        # materializer decodes nothing (routed-experts-only scope); it exists
        # here because transformers keys its checkpoint load off the shard
        # FILES, and the artifact's non-routed tensors share shards with
        # 148,608 routed payload objects.
        materialized = os.environ.get(
            "TR3_BF16", "%s/models/target-bf16-materialized" % fs)
        extra.update({
            "bf16": materialized,
            "tr3_root": "%s/models/target" % fs,
            "tr3_repo": target.get("repo_id", ""),
            "tr3_revision": target.get("revision", ""),
            # the fetch stage verifies the release's published SHA256SUMS
            # byte-wise; `crosscheck` proves that list equals the seal's own
            # shard_sha256 map, which is the binding the receipt claims
            "tr3_verify_shards": os.environ.get("TR3_VERIFY_SHARDS", "crosscheck"),
        })
        missing = [k for k in ("tr3_repo", "tr3_revision") if not extra[k]]
        if missing:
            con.err("job.json target is missing %s -- a tr3 capture cannot "
                    "seal its provenance without them" % ", ".join(missing))
            return 3
    elif surface == "dione":
        # A Dione release quantizes the routed experts of layers 3-44 ONLY and
        # retains everything else -- head included -- at source precision, in
        # its own retained/ shards.  Those shards hold no routed payloads, but
        # they DO hold the 864 MTP-layer expert tensors, and the streaming view
        # filters every `.mlp.experts.N.` name out of the index; transformers
        # keys its load off the shard FILES, so the measured non-routed set
        # still needs shards of its own.  --bf16 is the tree
        # `stage_measure.sh materialize` wrote from THIS snapshot.
        materialized = os.environ.get(
            "DIONE_BF16", "%s/models/target-bf16-materialized" % fs)
        extra.update({
            "bf16": materialized,
            "dione_root": "%s/models/target" % fs,
            "dione_repo": target.get("repo_id", ""),
            "dione_revision": target.get("revision", ""),
            # the fetch stage hashes every shard against the release manifest
            # and writes dione-shards-verified.json; `full` requires that
            # marker, which is the binding the receipt claims
            "dione_verify_shards": os.environ.get("DIONE_VERIFY_SHARDS", "full"),
        })
        missing = [k for k in ("dione_repo", "dione_revision") if not extra[k]]
        if missing:
            con.err("job.json target is missing %s -- a dione capture cannot "
                    "seal its provenance without them" % ", ".join(missing))
            return 3
    try:
        argv = build_invocation(
            engine,
            suite_root=Path(os.environ.get("FIDELITY_SUITE_ROOT", fs)),
            checkpoint="%s/models/target" % fs,
            panel_dir="%s/panel" % fs,
            out_dir=args.out,
            surface=surface,
            profile=job.get("profile", "k6"),
            cold_run=args.cold_run,
            reduce_order=job.get("reduce_order", "fp32"),
            roles=job.get("panel", {}).get("roles", "final"),
            extra=extra,
        )
        # Same rule as measure_local's execute path: a lane's fixed_flags are
        # part of its pinned contract and must be on every composed argv.
        for flag, value in (engine.fixed_flags or {}).items():
            if flag not in argv:
                argv.extend([flag, str(value)])
        # A lane with its own launcher (sealed-ep8: torchrun) already names the
        # program; everything else is a bare script path that needs one.
        if not engine.launcher:
            argv = [engine_python(fs)] + argv
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
