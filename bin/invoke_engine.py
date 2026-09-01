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
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fidelity.common import Console                     # noqa: E402
from fidelity.engines import (                          # noqa: E402
    EngineUnpinned, build_invocation, fidelity_python, load_engines,
)
from fidelity.jobcontract import (                      # noqa: E402
    JobContractError, parse_job_bytes, validate_execution_job,
)


# The engine tree's own default root. It used to be `/home/jl_fs/glm53-k6`:
# one provider AND one model baked into a path on a box that may be measuring
# something else entirely. The provider half cost three paid runs to find (see
# bin/selftest_provider_portability.py); the model half is the same defect one
# axis over, and bin/selftest_naming_sweep.py is the rule that finds the next
# one without renting anything.
#
# FIDELITY_K6_ROOT is the pre-2026-08-31 spelling. It is still READ, because a
# controller and an instance can come from different checkouts (and the
# container image bakes the old name today); it is no longer WRITTEN.
ENGINE_ROOT_ENV = "FIDELITY_ENGINE_ROOT"
ENGINE_ROOT_ENV_LEGACY = "FIDELITY_K6_ROOT"
ENGINE_ROOT_DEFAULT = "/home/jl_fs/fidelity-engine"


def engine_root() -> str:
    """Where the venv, the pipeline clone and the patch series live."""
    return (os.environ.get(ENGINE_ROOT_ENV)
            or os.environ.get(ENGINE_ROOT_ENV_LEGACY)
            or ENGINE_ROOT_DEFAULT)


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
        "%s/venv/bin/python" % engine_root(),
        "%s/venv/bin/python" % fs,
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return fidelity_python()

def canonical_paid_paths(job: dict) -> dict:
    """Derive every execution argv path from the sealed attempt or local roots."""
    attempt = job["execution_attempt"]
    fs = os.environ.get("FIDELITY_FS_ROOT")
    engine = os.environ.get("FIDELITY_ENGINE_ROOT")
    for label, value in (("FIDELITY_FS_ROOT", fs),
                         ("FIDELITY_ENGINE_ROOT", engine)):
        if (not isinstance(value, str) or not value
                or not Path(value).is_absolute()
                or os.path.normpath(value) != value):
            raise JobContractError("%s must be an exact absolute path" % label)
    if attempt["kind"] == "runpod-ssh":
        if fs != attempt["remote_root"]:
            raise JobContractError(
                "FIDELITY_FS_ROOT must equal canonical paid path %s"
                % attempt["remote_root"])
        if engine != attempt["engine_root"]:
            raise JobContractError(
                "FIDELITY_ENGINE_ROOT must equal canonical paid path %s"
                % attempt["engine_root"])
    elif attempt["kind"] != "local-container":
        raise JobContractError("execution_attempt.kind is unsupported")
    path_kind = "paid" if attempt["kind"] == "runpod-ssh" else "local"
    expected = {
        "FIDELITY_SUITE_ROOT": fs,
        "BF16": "%s/models/bf16" % fs,
        "TR3_BF16": "%s/models/target-bf16-materialized" % fs,
        "QP_PIPELINE_ROOT": "%s/pipeline" % engine,
        "FIDELITY_ENGINE_PYTHON": "%s/venv/bin/python" % engine,
    }
    for name, exact in expected.items():
        if os.environ.get(name) != exact:
            raise JobContractError(
                "%s must equal canonical %s path %s"
                % (name, path_kind, exact))
    return dict(expected, FIDELITY_FS_ROOT=fs,
                FIDELITY_ENGINE_ROOT=engine)


def _load_job(path: str) -> dict:
    """Load the exact finalized execution contract without permissive JSON."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise JobContractError("cannot read job.json: %s" % exc) from exc
    job = parse_job_bytes(raw)
    validate_execution_job(job)
    return job


def _invocation_values(job: dict, lane: str, engine) -> dict:
    """Resolve only values explicitly bound by the verified job."""
    if job["role"] != "quant":
        raise JobContractError("invoke_engine only executes quant jobs")
    if lane != job["lane"]:
        raise JobContractError(
            "--lane %r differs from job lane %r" % (lane, job["lane"]))

    profile = job["profile"]
    profile_fields = {"profile_id", "lane", "source", "surface", "bits"}
    if set(profile) != profile_fields:
        raise JobContractError(
            "quant profile fields differ from the canonical execution profile")
    if profile["lane"] != lane:
        raise JobContractError("profile lane differs from job lane")
    if profile["profile_id"] != "tr3-6bpw":
        raise JobContractError(
            "initial paid invoker permits only tr3-6bpw")

    target = job["target"]
    surface = target.get("surface")
    bits = target.get("bits")
    if not isinstance(surface, str) or not surface:
        raise JobContractError("job target lacks an exact surface")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise JobContractError("job target lacks exact numeric bits")
    if profile["surface"] != surface or profile["bits"] != bits:
        raise JobContractError("profile surface/bits differ from target identity")
    expected_profile = {
        "tr3-6bpw": ("tr3", "tr3-published", 6.0, "none"),
    }[profile["profile_id"]]
    if (profile["source"], profile["surface"], float(profile["bits"])) != (
            expected_profile[0], expected_profile[1], expected_profile[2]):
        raise JobContractError(
            "profile source/surface/bits differ from its authored identity")

    runtime = job.get("runtime")
    timing_runtime = job["timing"].get("runtime_profile")
    if not isinstance(runtime, dict) or not isinstance(timing_runtime, dict):
        raise JobContractError(
            "job lacks runtime and timing.runtime_profile contracts")
    for key in ("decode_cache", "decode_threads", "reader_threads"):
        if key not in runtime or key not in timing_runtime:
            raise JobContractError("job lacks exact runtime %s" % key)
        if runtime[key] != timing_runtime[key]:
            raise JobContractError(
                "runtime %s differs from timing evidence" % key)
    if runtime["decode_cache"] != expected_profile[3]:
        raise JobContractError("runtime decode_cache differs from profile evidence")
    if runtime.get("device") != "cuda":
        raise JobContractError("initial paid invoker requires runtime device cuda")
    for key in ("decode_threads", "reader_threads"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise JobContractError("runtime %s must be a positive integer" % key)
    # stream_score's one producer pool performs decode for compressed sources.
    if runtime["decode_threads"] != runtime["reader_threads"]:
        raise JobContractError(
            "engine has one producer pool but decode_threads and reader_threads differ")
    reduce_order = job.get("reduce_order")
    if (not isinstance(reduce_order, str) or not reduce_order
            or runtime.get("reduce_order") != reduce_order):
        raise JobContractError(
            "job reduce_order is absent or differs from runtime contract")
    panel = job["panel"]
    roles = panel.get("roles")
    if roles != "final":
        raise JobContractError(
            "initial paid invoker requires explicit panel roles 'final'")

    required_flags = {"decode_cache", "decode_threads", "device"}
    absent = sorted(required_flags - set(engine.flag_map or {}))
    if absent:
        raise JobContractError(
            "lane cannot express authored runtime fields: %s" % ", ".join(absent))
    return {
        "surface": surface,
        "source": profile["source"],
        "profile_id": profile["profile_id"],
        "roles": roles,
        "reduce_order": reduce_order,
        "decode_cache": runtime["decode_cache"],
        "decode_threads": runtime["decode_threads"],
        "device": runtime["device"],
    }


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
    try:
        job = _load_job(args.job)
        if args.cold_run not in (1, 2):
            raise JobContractError("--cold-run must be 1 or 2 for this job")
        engines = load_engines(HERE / "engines.json")
        engine = engines.get(args.lane)
        paths = canonical_paid_paths(job)
        if engine is None:
            raise JobContractError(
                "no engine configured for lane %r" % args.lane)
        values = _invocation_values(job, args.lane, engine)
    except (JobContractError, EngineUnpinned) as exc:
        con.err(str(exc))
        return 3

    fs = paths["FIDELITY_FS_ROOT"]
    target = job["target"]
    surface = values["surface"]
    extra = {
        "source": values["source"],
        "decode_cache": values["decode_cache"],
        "decode_threads": values["decode_threads"],
        "device": values["device"],
        "bf16": paths["BF16"],
        "pipeline_root": paths["QP_PIPELINE_ROOT"],
    }
    if surface == "tr3-published":
        extra.update({
            "bf16": paths["TR3_BF16"],
            "tr3_root": "%s/models/target" % fs,
            "tr3_repo": target["repo_id"],
            "tr3_revision": target["revision"],
            "tr3_verify_shards": "crosscheck",
        })
    try:
        argv = build_invocation(
            engine,
            suite_root=Path(paths["FIDELITY_SUITE_ROOT"]),
            checkpoint="%s/models/target" % fs,
            panel_dir="%s/panel" % fs,
            out_dir=args.out,
            surface=surface,
            profile=values["profile_id"],
            cold_run=args.cold_run,
            reduce_order=values["reduce_order"],
            roles=values["roles"],
            extra=extra,
        )
        # Same rule as measure_local's execute path: a lane's fixed_flags are
        # part of its pinned contract and must be on every composed argv.
        for flag, value in (engine.fixed_flags or {}).items():
            if flag not in argv:
                argv.extend([flag, str(value)])
        missing_flags = [
            flag for flag in (engine.required_flags or ()) if flag not in argv]
        if missing_flags:
            raise JobContractError(
                "composed invocation lacks required flags: %s"
                % ", ".join(missing_flags))
        # A lane with its own launcher already names the program; everything
        # else is a bare script path that needs one.
        if not engine.launcher:
            argv = [paths["FIDELITY_ENGINE_PYTHON"]] + argv
    except (EngineUnpinned, JobContractError) as exc:
        con.err(str(exc))
        return 3

    env = dict(os.environ)
    env.update({k: str(v) for k, v in (engine.env or {}).items()})
    con.say("engine argv: %s" % " ".join(argv))
    if args.print_only:
        return 0
    # STREAM the engine's output; do not capture it.
    #
    # `run(...)` buffers stdout and stderr and hands them back after the process
    # EXITS, so a 79-minute capture wrote exactly one line to its stage log --
    # the argv -- and nothing else until it was over. There is no way to tell a
    # healthy run from a stalled one from that file, which is precisely the
    # question a supervising controller has to answer (JOURNAL lesson 43). The
    # stage driver already tees this process's own stdout into
    # logs/measure-run-N.log, so inheriting the streams puts the engine's
    # per-layer progress there, live, at the cost of nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.Popen(argv, env=env, stdout=None, stderr=None)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
