#!/usr/bin/env python3
"""Offline collision and semantic tests for the shared job.v2 identity."""
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.jobcontract import (  # noqa: E402
    JobContractError, finalize_bundle_manifest, finalize_job,
    validate_execution_job, verify_job,
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)


def fixture():
    payload = b"bundle-byte"
    bundle = finalize_bundle_manifest([{
        "path": "bin/stages.py", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }], "BUNDLE.txt")
    control = finalize_bundle_manifest([{
        "path": "bin/measure_cloud.py", "bytes": 1,
        "sha256": "9" * 64,
    }], "authored-control-plane-closure")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    registry = {"path": "bin/BUNDLE.txt", "bytes": 20,
                "sha256": "2" * 64}
    bundle_contract = hashlib.sha256(json.dumps(
        {"bundle": bundle, "registry": registry}, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    shards = [{"path": "model-00001-of-00001.safetensors", "bytes": 123}]
    shard_digest = hashlib.sha256(json.dumps(
        shards, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    download_manifest = [
        {"path": "config.json", "bytes": 1},
        shards[0],
        {"path": "model.safetensors.index.json", "bytes": 1},
    ]
    download_digest = hashlib.sha256(json.dumps(
        download_manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "schema": "fidelity-suite/job.v2", "role": "quant",
        "recipe": "cloud", "lane": "streaming", "cold_runs": 2,
        "target": {
            "repo_id": "owner/model", "revision": "1" * 40,
            "path": None, "surface": "native-bf16",
            "codec": "bf16", "bits": 16,
            "config_sha256": "a" * 64, "index_sha256": "b" * 64,
            "model_bytes": 123, "shards": shards,
            "shard_manifest_sha256": shard_digest,
            "download_manifest": download_manifest,
            "download_bytes_total": 125,
            "download_manifest_sha256": download_digest,
        },
        "bundle": bundle, "bundle_registry": registry,
        "bundle_contract_sha256": bundle_contract,
        "control_plane": control,
        "panel": {
            "panel_id": "p", "roles": "final",
            "resolved_binding": {"revision": "3" * 40},
            "panel_receipt_sha256": "1" * 64,
            "reference_ref": "root@pin",
            "teacher_receipt_sha256": "2" * 64,
            "teacher_backend_identity_sha256": "3" * 64,
        },
        "reference": {
            "reference_ref": "root@pin",
            "teacher_receipt_sha256": "2" * 64,
            "teacher_backend_identity_sha256": "3" * 64,
        },
        "profile": {
            "profile_id": "native-bf16", "lane": "streaming",
            "source": "native", "surface": "native-bf16", "bits": 16},
        "timing": {"cold_runs": 2, "window_count": 25},
        "runtime": {},
        "scoring": {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda", "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        },
        "capture": {"replay_dtype": "float32", "replay_vocab_chunk": 8192,
                    "unexpected_tensor_allowlist": {
                        "path": "engines/evidence/a.json",
                        "artifact_sha256": "4" * 64,
                        "canonical_sorted_names_sha256": "5" * 64}},
        "scope": {"policy": "known"},
        "produced_by": {"name": "suite", "revision": "6" * 40},
        "measurer": {"name": "selftest"},
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 1,
            "min_memory_gb": 1,
            "expected_vram_bytes": 1,
        },
        "environment": {"provider": "runpod", "gpu": "H200",
                        "offer": "on-demand", "secure_cloud": True},
        "execution_attempt": {
            "kind": "runpod-ssh",
            "attempt_id": None,
            "cost_quote": None,
            "engine_root": None,
            "execution_contract_sha256": None,
            "lease_path": None,
            "planned_at": "2026-09-01T00:00:00Z",
            "pre_create_safety": None,
            "prepared_create": None,
            "remote_root": None,
            "provider_terminate_after": None,
            "storage_layout": "container-disk",
            "workload_deadline_utc": None,
        },
    }


def mutate_registry(document):
    document["bundle_registry"]["bytes"] += 1
    document["bundle_contract_sha256"] = hashlib.sha256(json.dumps(
        {"bundle": document["bundle"],
         "registry": document["bundle_registry"]},
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def changed(base, mutator):
    candidate = copy.deepcopy(base)
    mutator(candidate)
    return finalize_job(candidate)["job_id_full"]


def main():
    base = fixture()
    finalized = finalize_job(base)
    check("full identity verifies",
          verify_job(finalized) == finalized["job_id_full"])
    mutations = [
        ("target", lambda d: d["target"].update(config_sha256="c" * 64)),
        ("provider", lambda d: d["environment"].update(gpu="L4")),
        ("bundle registry", mutate_registry),
        ("panel", lambda d: d["panel"].update(panel_id="other")),
        ("profile", lambda d: d["profile"].update(profile_id="other-profile")),
        ("timing", lambda d: d["timing"].update(cold_runs=3)),
        ("replay", lambda d: d["capture"].update(replay_dtype="float64")),
        ("allowlist", lambda d: d["capture"]["unexpected_tensor_allowlist"].update(
            artifact_sha256="7" * 64)),
        ("scope", lambda d: d["scope"].update(policy="other")),
        ("producer", lambda d: d["produced_by"].update(revision="8" * 40)),
        ("unknown top-level",
         lambda d: d.update(future_identity={"x": 1})),
    ]
    for name, mutate in mutations:
        check(name + " moves identity", changed(base, mutate) != finalized["job_id_full"])
    attempt_only = copy.deepcopy(base)
    attempt_only["execution_attempt"].update(
        attempt_id="a" * 24, lease_path="/leases/a.json",
        workload_deadline_utc="2026-09-01T01:00:00Z",
        provider_terminate_after="2026-09-01T01:30:00Z")
    check("attempt and deadlines do not move identity",
          finalize_job(attempt_only)["job_id_full"] == finalized["job_id_full"])
    malformed = copy.deepcopy(finalized)
    malformed["bundle"]["extra"] = True
    try:
        verify_job(malformed)
    except JobContractError:
        pass
    else:
        raise AssertionError("noncanonical bundle accepted")
    missing_panel_receipt = copy.deepcopy(base)
    del missing_panel_receipt["panel"]["panel_receipt_sha256"]
    mismatched_reference = copy.deepcopy(base)
    mismatched_reference["reference"]["teacher_receipt_sha256"] = "9" * 64
    expanded_reference = copy.deepcopy(base)
    expanded_reference["reference"]["unbound"] = True
    for name, invalid in (
            ("missing panel receipt", missing_panel_receipt),
            ("mismatched panel/reference", mismatched_reference),
            ("unbound reference field", expanded_reference)):
        try:
            finalize_job(invalid)
        except JobContractError:
            pass
        else:
            raise AssertionError("%s accepted" % name)
    print("PASS: job.v2 identity collision and semantic checks")
    for rows in ([], [
            {"path": "./bin/stages.py", "bytes": 1, "sha256": "0" * 64}]):
        try:
            finalize_bundle_manifest(rows, "fixture")
        except JobContractError:
            pass
        else:
            raise AssertionError("empty/aliased bundle accepted")
    executed = copy.deepcopy(finalized)
    executed["execution_attempt"].update(
        attempt_id="a" * 24, lease_path="/private/lease.json",
        workload_deadline_utc="2026-09-01T01:00:00Z",
        provider_terminate_after="2026-09-01T01:30:00Z")
    try:
        validate_execution_job(executed)
    except JobContractError:
        pass
    else:
        raise AssertionError("absolute lease path accepted")
    # A vast-container execution_attempt is the same minimal shape as
    # local-container: number 1, kind, 24-hex attempt_id.
    vast_container = copy.deepcopy(base)
    vast_container["execution_attempt"] = {
        "kind": "vast-container", "number": 1, "attempt_id": "a" * 24}
    vast_finalized = finalize_job(vast_container)
    check("vast-container execution_attempt is accepted",
          verify_job(vast_finalized) == vast_finalized["job_id_full"])
    vast_bad = copy.deepcopy(base)
    vast_bad["execution_attempt"] = {
        "kind": "vast-container", "number": 1, "attempt_id": "a" * 24,
        "extra": True}
    try:
        finalize_job(vast_bad)
    except JobContractError:
        pass
    else:
        raise AssertionError("vast-container with extra fields accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
