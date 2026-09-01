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
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fidelity.common import Console, run                # noqa: E402
from fidelity.engines import load_engines               # noqa: E402
from fidelity.jobcontract import (                      # noqa: E402
    JobContractError, parse_job_bytes, validate_execution_job,
)
from invoke_engine import canonical_paid_paths           # noqa: E402


SCORING_POLICY = {
    "schema": "fidelity-suite/kld-scoring.v1",
    "device": "cuda",
    "chunk_positions": 512,
    "compute_dtype": "float64",
    "direction": "reference_to_candidate",
    "vocabulary": "full",
    "reduction": "mean_of_run_means_tokenwise_kld",
}


def _load_job(path: str) -> dict:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise JobContractError("cannot read job.json: %s" % exc) from exc
    job = parse_job_bytes(raw)
    validate_execution_job(job)
    return job


def _sha256(value) -> bool:
    return (isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _scoring_values(job: dict, lane: str) -> dict:
    if job["role"] != "quant":
        raise JobContractError("invoke_scorer only scores quant jobs")
    if lane != job["lane"]:
        raise JobContractError(
            "--lane %r differs from job lane %r" % (lane, job["lane"]))
    if job["cold_runs"] != 2:
        raise JobContractError("submittable scoring requires exactly two cold runs")
    profile = job["profile"]
    if (set(profile) != {"profile_id", "lane", "source", "surface", "bits"}
            or profile["lane"] != lane
            or profile["profile_id"] not in ("tr3-6bpw", "native-bf16")):
        raise JobContractError("job has no canonical paid quant profile")
    if job.get("scoring") != SCORING_POLICY:
        raise JobContractError(
            "job scoring policy differs from full-vocab fp64 reference||candidate")
    if job.get("reduce_order") != (job.get("runtime") or {}).get("reduce_order"):
        raise JobContractError("job reduce_order differs from runtime contract")

    panel = job["panel"]
    reference = job["reference"]
    if panel.get("roles") != "final":
        raise JobContractError("scoring requires explicit panel roles 'final'")
    bindings = (
        ("reference_ref", "reference_ref"),
        ("teacher_receipt_sha256", "teacher_receipt_sha256"),
        ("teacher_backend_identity_sha256", "teacher_backend_identity_sha256"),
    )
    for panel_key, reference_key in bindings:
        value = panel.get(panel_key)
        valid = ((isinstance(value, str) and bool(value))
                 if panel_key == "reference_ref" else _sha256(value))
        if not valid:
            raise JobContractError("job panel lacks exact %s" % panel_key)
        if reference.get(reference_key) != value:
            raise JobContractError(
                "job panel/reference %s binding differs" % panel_key)
    panel_receipt = panel.get("panel_receipt_sha256")
    if not _sha256(panel_receipt):
        raise JobContractError(
            "job panel lacks exact token-panel receipt identity")
    return {
        "profile_id": profile["profile_id"],
        "device": SCORING_POLICY["device"],
        "chunk_positions": SCORING_POLICY["chunk_positions"],
        "teacher_receipt_sha256": reference["teacher_receipt_sha256"],
        "teacher_backend_identity_sha256":
            reference["teacher_backend_identity_sha256"],
        "token_panel_receipt_sha256": panel_receipt,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--receipts", required=True,
                    help="directory holding run-1 .. run-N")
    ap.add_argument("--out", help="aggregate receipt path "
                                  "(default <receipts>/<profile>-packed-kld.json)")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    con = Console()
    try:
        job = _load_job(args.job)
        engine = load_engines(HERE / "engines.json").get(args.lane)
        paths = canonical_paid_paths(job)
        if engine is None:
            raise JobContractError(
                "no engine configured for lane %r" % args.lane)
        scorer = engine.scorer or {}
        if not scorer.get("entrypoint"):
            raise JobContractError(
                "lane %r pins no scorer entrypoint in engines.json" % args.lane)
        if engine.receipt_class != "submittable":
            raise JobContractError(
                "lane %r is not submittable; use bin/kld_preview.py"
                % args.lane)
        values = _scoring_values(job, args.lane)
        flag = scorer.get("flag_map")
        required = scorer.get("required_flags")
        if not isinstance(flag, dict) or not isinstance(required, list):
            raise JobContractError("scorer registry contract is incomplete")
        required_keys = {
            "profile", "panel", "runs", "out", "device", "chunk_positions",
            "pipeline_root", "expected_teacher_receipt",
            "expected_token_panel_receipt", "expected_teacher_backend",
        }
        missing_keys = sorted(required_keys - set(flag))
        if missing_keys:
            raise JobContractError(
                "scorer cannot express job bindings: %s"
                % ", ".join(missing_keys))
        expected_required = {flag[key] for key in required_keys}
        if set(required) != expected_required or len(required) != len(expected_required):
            raise JobContractError(
                "scorer required_flags differ from its exact flag_map contract")
    except JobContractError as exc:
        con.err(str(exc))
        return 3

    fs = paths["FIDELITY_FS_ROOT"]
    suite_root = Path(paths["FIDELITY_SUITE_ROOT"])
    receipts = Path(args.receipts).resolve()
    run_dirs = [receipts / "run-1", receipts / "run-2"]
    missing = [str(d) for d in run_dirs
               if not (d / "capture-receipt.json").is_file()]
    if missing:
        con.err("no capture receipt under: %s -- the measure stage did not "
                "finish both authored cold runs" % ", ".join(missing))
        return 2

    profile_id = values["profile_id"]
    out = (Path(args.out).resolve() if args.out else
           receipts / ("%s-packed-kld.json" % profile_id))
    argv = [
        paths["FIDELITY_ENGINE_PYTHON"],
        str((suite_root / scorer["entrypoint"]).resolve()),
        flag["profile"], profile_id,
        flag["panel"], "%s/panel" % fs,
        flag["runs"],
    ] + [str(d) for d in run_dirs] + [
        flag["out"], str(out),
        flag["device"], values["device"],
        flag["chunk_positions"], str(values["chunk_positions"]),
        flag["expected_teacher_receipt"],
        values["teacher_receipt_sha256"],
        flag["expected_token_panel_receipt"],
        values["token_panel_receipt_sha256"],
        flag["expected_teacher_backend"],
        values["teacher_backend_identity_sha256"],
        flag["pipeline_root"],
        paths["QP_PIPELINE_ROOT"],
    ]
    missing_flags = [required_flag for required_flag in required
                     if required_flag not in argv]
    if missing_flags:
        con.err("composed scorer invocation lacks required flags: %s"
                % ", ".join(missing_flags))
        return 3

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
