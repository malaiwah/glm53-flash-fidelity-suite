#!/usr/bin/env python3
"""Offline coherence checks for paid engine profiles and timing admission.

Stock Python only: this checks the probed CLI declarations, report labels,
registry maps, explicit refusals, and fail-closed timing lookup without loading a
model, torch, or provider SDK.  It intentionally performs no network access.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from decimal import Decimal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from fidelity.campaign import CostQuote  # noqa: E402
from fidelity.engines import (  # noqa: E402
    Engine,
    load_engines,
    require_supported_profile,
    resolve_profile_timing,
    resolve_root_timing,
)
from fidelity.jobcontract import (  # noqa: E402
    finalize_bundle_manifest,
    finalize_job,
    job_identity_projection,
    seal_execution_job,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[%s] %s%s" % ("ok" if ok else "FAIL", name,
                         (" - " + detail) if detail else ""))
    if not ok:
        raise SystemExit("selftest_engine_profiles: %s failed: %s" % (name, detail))


def refuses(name, fn, needle):
    try:
        fn()
    except Exception as exc:  # refusal type is part of fidelity.engines
        check(name, needle in str(exc), str(exc))
        return
    check(name, False, "did NOT refuse")


def module_tree(name):
    return ast.parse((HERE / name).read_text(encoding="utf-8"), filename=name)


def top_level_literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError("%s is not a literal top-level assignment" % name)


def argparse_choices(tree, flag):
    choices = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                choices.update(ast.literal_eval(keyword.value))
    return choices


def report_student_labels(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "student_label"
                   for target in node.targets):
            continue
        if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Dict):
            return ast.literal_eval(node.value.value)
    raise AssertionError("kld_report student_label dispatch is no longer a literal map")


stream_tree = module_tree("stream_score.py")
report_tree = module_tree("kld_report.py")
stream_profiles = top_level_literal(stream_tree, "TR3_PROFILES")
report_labels = report_student_labels(report_tree)
stream_choices = argparse_choices(stream_tree, "--profile")
report_choices = argparse_choices(report_tree, "--profile")

check("TR3 scorer admits 4bpw and pinned public 6bpw only",
      stream_profiles == {
          "tr3-4bpw": (4.0, "tr3-exl3-mcg-4bpw"),
          "tr3-6bpw": (6.0, "tr3-exl3-mcg-6bpw"),
      }, repr(stream_profiles))
check("K8 is not an admitted scorer profile",
      "tr3-8bpw" not in stream_profiles and "tr3-8bpw" not in stream_choices)
for profile, (_, label) in sorted(stream_profiles.items()):
    check("%s CLI/report label coherence" % profile,
          profile in stream_choices and profile in report_choices
          and report_labels.get(profile) == label,
          "stream=%s report=%s label=%r" % (
              profile in stream_choices, profile in report_choices,
              report_labels.get(profile)))

lane = load_engines()["streaming"]
scoped = lane.profile_map_by_surface.get("tr3-published") or {}
check("paid TR3 map admits K6 and preserves 4bpw",
      scoped == {"4.0": "tr3-4bpw", "6.0": "tr3-6bpw"}, repr(scoped))
check("generic paid map does not offer K8",
      "8.0" not in lane.profile_map, repr(lane.profile_map))
refuses("K8 has an explicit pre-spend evidence refusal",
        lambda: require_supported_profile(
            lane, surface="tr3-published", bits=8.0),
        "missing_sealed_surface_measurement_bridge")

k6_timing = resolve_profile_timing(
    lane, profile="tr3-6bpw", surface="tr3-published", bits=6.0,
    target_repo="malaiwah/GLM-5.3-Flash-TR3-6bpw",
    target_revision="9ab94105a71708a19c6d960d24b4aa6d459f5623",
    gpu="H200")
native_timing = resolve_profile_timing(
    lane, profile="native-bf16", surface="native-bf16",
    target_repo="zai-org/GLM-5.3-Flash-BF16",
    target_revision="a6c167b62691b2bac901344b65cb651a70f53e43",
    gpu="H200")
check("K6 timing is pinned-target and resource-profile specific",
      k6_timing["minutes_per_window"] == 7.35
      and k6_timing["target_revision"]
      == "9ab94105a71708a19c6d960d24b4aa6d459f5623"
      and k6_timing["runtime_profile"] == {
          "gpu": "H200", "gpu_count": 1, "window_count": 25,
          "decode_cache": "none", "decode_threads": 28, "reader_threads": 28,
          "min_vcpu_count": 28, "min_memory_gb": 300,
          "controller_processes_per_pod": 1,
      })
check("native BF16 timing carries its exact resource profile",
      native_timing["minutes_per_window"] == 11.3
      and native_timing["target_revision"]
      == "a6c167b62691b2bac901344b65cb651a70f53e43"
      and native_timing["runtime_profile"]["decode_cache"] == "ram"
      and native_timing["runtime_profile"]["decode_threads"] == 28
      and native_timing["runtime_profile"]["reader_threads"] == 28
      and native_timing["runtime_profile"]["min_vcpu_count"] == 28
      and native_timing["runtime_profile"]["min_memory_gb"] == 300)
refuses("K6 timing refuses the wrong pinned target",
        lambda: resolve_profile_timing(
            lane, profile="tr3-6bpw", surface="tr3-published", bits=6.0,
            target_repo="other/K6",
            target_revision="9ab94105a71708a19c6d960d24b4aa6d459f5623",
            gpu="H200"),
        "timing_target_mismatch")
refuses("K6 timing refuses the wrong GPU",
        lambda: resolve_profile_timing(
            lane, profile="tr3-6bpw", surface="tr3-published", bits=6.0,
            target_repo="malaiwah/GLM-5.3-Flash-TR3-6bpw",
            target_revision="9ab94105a71708a19c6d960d24b4aa6d459f5623",
            gpu="L4"),
        "timing_gpu_mismatch")
refuses("an untimed paid profile refuses instead of using lane minutes",
        lambda: resolve_profile_timing(
            lane, profile="unmeasured", surface="tr3-published", bits=7.0),
        "timing_evidence_absent")

fruit = resolve_root_timing(
    target_repo="malaiwah/GLM-5.2-SIQ-Fruit-bf16",
    target_revision="ef68013aa6e16453cf52b5b77647f72fbe258c3c",
    gpu="L4", form="hidden", schedule="two-fresh-process-qualification")
m2 = resolve_root_timing(
    target_repo="zai-org/GLM-5.3-Flash-BF16",
    target_revision="a6c167b62691b2bac901344b65cb651a70f53e43",
    gpu="H200", form="hidden", schedule="two-fresh-process-qualification")
check("root timings are named conservative two-process bounds",
      fruit["conservative_upper_hours"] == 1.0
      and m2["conservative_upper_hours"] == 6.0
      and fruit["bound_kind"] == m2["bound_kind"] == "named_conservative_upper")
check("root timing requires separate explicit resource admission",
      fruit["resource_admission"]["required"] is True
      and m2["resource_admission"]["required"] is True
      and fruit["resource_admission"]["mode"]
      == m2["resource_admission"]["mode"]
      == "controller_explicit_safe_resources")
check("root timing rows bind exact target metadata",
      fruit["model_identity"] == {
          "model_bytes": 10102776813,
          "config_sha256":
              "5a19697e555fff140d1b089b852c3ef227114b196f8d76796560feeeb34dc44a",
          "index_sha256":
              "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56",
      }
      and m2["model_identity"] == {
          "model_bytes": 642646653816,
          "config_sha256":
              "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943",
          "index_sha256":
              "e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f",
      })
root_args = {
    "target_repo": "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
    "target_revision": "ef68013aa6e16453cf52b5b77647f72fbe258c3c",
    "gpu": "L4",
    "form": "hidden",
    "schedule": "two-fresh-process-qualification",
}
for dimension, wrong in (
    ("target_repo", "unknown/model"),
    ("gpu", "A100"),
    ("form", "race"),
    ("schedule", "one-process"),
):
    candidate = dict(root_args)
    candidate[dimension] = wrong
    refuses("unknown root %s refuses" % dimension,
            lambda candidate=candidate: resolve_root_timing(**candidate),
            "root_timing_evidence_absent")

with tempfile.TemporaryDirectory(prefix="engine-config-strict-") as config_td:
    duplicate_profile = Path(config_td) / "duplicate-profile.json"
    duplicate_profile.write_text(
        '{"lanes":{"streaming":{"timing":{"profiles":{'
        '"tr3-6bpw":{},"tr3-6bpw":{}}}}}}',
        encoding="utf-8")
    refuses("engine loader rejects duplicate authored profile keys",
            lambda: load_engines(duplicate_profile),
            "duplicate key 'tr3-6bpw'")

    duplicate_timing = Path(config_td) / "duplicate-root-timing.json"
    duplicate_timing.write_text(
        '{"lanes":{},"root_timing_profiles":[{'
        '"target_repo":"owner/a","target_repo":"owner/b"}]}',
        encoding="utf-8")
    refuses("root timing rejects duplicate authored timing keys",
            lambda: resolve_root_timing(
                target_repo="owner/a", target_revision="1" * 40,
                gpu="H200", form="hidden",
                schedule="two-fresh-process-qualification",
                path=duplicate_timing),
            "duplicate key 'target_repo'")

    nonfinite_timing = Path(config_td) / "nonfinite-timing.json"
    nonfinite_timing.write_text(
        '{"lanes":{},"root_timing_profiles":['
        '{"conservative_upper_hours":NaN}]}',
        encoding="utf-8")
    refuses("root timing rejects non-finite authored evidence",
            lambda: resolve_root_timing(
                target_repo="owner/a", target_revision="1" * 40,
                gpu="H200", form="hidden",
                schedule="two-fresh-process-qualification",
                path=nonfinite_timing),
            "non-finite JSON constant NaN")


with tempfile.TemporaryDirectory(prefix="engine-live-probe-") as probe_td:
    probe_root = Path(probe_td)
    engine_script = probe_root / "engine.py"
    scorer_script = probe_root / "scorer.py"
    engine_script.write_text(
        "print('--engine-value --fixed-value')\n", encoding="utf-8")
    scorer_script.write_text(
        "print('--scorer-value')\n", encoding="utf-8")
    probe_engine = Engine(
        lane="fixture", name="fixture", entrypoint="engine.py", pinned=True,
        launcher=[], required_flags=[],
        flag_map={"engine": "--engine-value"},
        scorer={
            "entrypoint": "scorer.py",
            "flag_map": {"scorer": "--scorer-value"},
            "required_flags": ["--scorer-value"],
        },
        notes="", fixed_flags={"--fixed-value": "yes"})
    live_probe = probe_engine.probe(
        probe_root, paid=True, python=sys.executable)
    check("paid probe covers every composed engine and scorer flag",
          live_probe["help_ok"]
          and live_probe["mode"] == "paid-live-help"
          and live_probe["engine"]["expected_flags"]
          == ["--engine-value", "--fixed-value"]
          and live_probe["scorer"]["expected_flags"] == ["--scorer-value"],
          repr(live_probe))
    engine_script.write_text(
        "# --engine-value --fixed-value are source-only\n"
        "raise SystemExit(2)\n", encoding="utf-8")
    failed_live_probe = probe_engine.probe(
        probe_root, paid=True, python=sys.executable)
    check("paid probe never source-regex-falls back after failed help",
          failed_live_probe["help_ok"] is False
          and failed_live_probe["engine"]["found_flags"] == []
          and "never falls back" in " ".join(
              failed_live_probe["engine"]["problems"]),
          repr(failed_live_probe))


def finalize_execution(document):
    prepared = json.loads(json.dumps(document))
    prepared.pop("job_id", None)
    prepared.pop("job_id_full", None)
    attempt = prepared["execution_attempt"]
    attempt.update({
        "attempt_id": "1" * 24,
        "cost_quote": None,
        "engine_root": None,
        "execution_contract_sha256": None,
        "lease_path": "leases/attempt.json",
        "pre_create_safety": None,
        "prepared_create": None,
        "remote_root": None,
        "provider_terminate_after": "2026-09-01T01:10:00Z",
        "workload_deadline_utc": "2026-09-01T01:00:00Z",
    })
    finalized = finalize_job(prepared)
    timing_digest = hashlib.sha256(json.dumps(
        finalized["timing"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    quote = CostQuote(
        reserved_compute_usd_per_hour=Decimal("1"),
        live_compute_usd_per_hour=Decimal("1"),
        container_disk_size_gb=Decimal("1"),
        container_disk_running_usd_per_gb_month=Decimal("0"),
        container_disk_stopped_usd_per_gb_month=Decimal("0"),
        pod_disk_size_gb=Decimal("1"),
        pod_disk_running_usd_per_gb_month=Decimal("0"),
        pod_disk_stopped_usd_per_gb_month=Decimal("0"),
        network_volume_size_gb=Decimal("0"),
        network_volume_usd_per_gb_month=Decimal("0"),
        storage_month_hours=Decimal("672"),
        network_billing_increment_seconds=Decimal("3600"),
        tariff_source="fixture",
        tariff_effective_at="2026-08-31T00:00:00Z",
        quoted_at="2026-09-01T00:00:00Z",
        valid_until="2026-09-01T00:04:00Z",
        target="%s@%s" % (
            finalized["target"]["repo_id"], finalized["target"]["revision"]),
        profile=finalized["profile"]["profile_id"],
        timing_kind="exact-target-profile",
        timing_evidence=timing_digest,
        workload_deadline_seconds=Decimal("3600"),
        provider_termination_deadline_seconds=Decimal("4200"),
        retrieval_delete_reserve_seconds=Decimal("300"),
        timer_api_lag_seconds=Decimal("0"),
        hard_cap_usd=Decimal("10"),
    )
    body = b"{}"
    finalized["execution_attempt"].update({
        "cost_quote": quote.to_dict(),
        "remote_root": "/workspace/fidelity/%s/%s" % (
            finalized["job_id_full"], attempt["attempt_id"]),
        "engine_root": "/workspace/fidelity-engine/%s/%s" % (
            finalized["job_id_full"], attempt["attempt_id"]),
        "prepared_create": {
            "schema": "fidelity-suite/runpod-prepared-create.v1",
            "request_identity": {"fixture": True},
            "graphql_body_sha256": hashlib.sha256(body).hexdigest(),
            "graphql_body_bytes": len(body),
            "graphql_body_base64":
                base64.b64encode(body).decode("ascii"),
        },
        "pre_create_safety": {
            "checked_at": "2026-09-01T00:00:00Z",
            "reaper_health_sha256": "6" * 64,
            "safety_proof_file_sha256": "7" * 64,
            "safety_proof_sha256": "8" * 64,
            "provider_account_id": "fixture-account",
            "provider_gpu_id": "H200",
            "image": "fixture/image@sha256:" + "9" * 64,
            "bundle_contract_sha256":
                finalized["bundle_contract_sha256"],
            "control_manifest_sha256":
                finalized["control_plane"]["manifest_sha256"],
            "server_time": {
                "schema": "fidelity-suite/runpod-server-time.v1",
                "local_minus_server_seconds": 0,
                "evidence_age_seconds": 0,
                "max_clock_delta_seconds": 30,
                "max_evidence_age_seconds": 30,
            },
        },
    })
    return seal_execution_job(finalized)


def invocation_job():
    payload = b"engine-profile-selftest"
    bundle = finalize_bundle_manifest([{
        "path": "bin/invoke_engine.py", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }], "BUNDLE.txt")
    control = finalize_bundle_manifest([{
        "path": "bin/invoke_engine.py", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }], "authored-control-plane-closure")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    registry = {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "2" * 64}
    bundle_contract = hashlib.sha256(json.dumps(
        {"bundle": bundle, "registry": registry}, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    shards = [{"path": "model.safetensors", "bytes": 123}]
    shard_digest = hashlib.sha256(json.dumps(
        shards, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    download_manifest = [
        {"path": "config.json", "bytes": 1},
        shards[0],
        {"path": "model.safetensors.index.json", "bytes": 1},
    ]
    download_digest = hashlib.sha256(json.dumps(
        download_manifest, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    runtime_profile = {
        "gpu": "H200", "gpu_count": 1, "window_count": 25,
        "decode_cache": "none", "decode_threads": 28, "reader_threads": 28,
        "min_vcpu_count": 28, "min_memory_gb": 300,
        "controller_processes_per_pod": 1,
    }
    job = finalize_job({
        "schema": "fidelity-suite/job.v2", "role": "quant", "recipe": "cloud",
        "lane": "streaming", "cold_runs": 2,
        "official_bf16_revision":
            "a6c167b62691b2bac901344b65cb651a70f53e43",
        "profile": {
            "profile_id": "tr3-6bpw", "lane": "streaming", "source": "tr3",
            "surface": "tr3-published", "bits": 6.0,
        },
        "timing": {"runtime_profile": runtime_profile},
        "runtime": dict(
            runtime_profile, device="cuda",
            expert_parallel={"mode": "single_device", "world_size": 1},
            reduce_order="fp32", capacity_basis="authored-profile-measured-host"),
        "reduce_order": "fp32",
        "scoring": {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda", "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate", "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        },
        "target": {
            "repo_id": "malaiwah/GLM-5.3-Flash-TR3-6bpw",
            "revision": "9ab94105a71708a19c6d960d24b4aa6d459f5623",
            "path": None, "surface": "tr3-published",
            "codec": "exl3-mcg", "bits": 6.0,
            "config_sha256": "a" * 64, "index_sha256": "b" * 64,
            "model_bytes": 123, "shards": shards,
            "shard_manifest_sha256": shard_digest,
            "download_manifest": download_manifest,
            "download_bytes_total": 125,
            "download_manifest_sha256": download_digest,
            "official_bf16_identity": {
                "config_sha256": "c" * 64,
                "config_bytes": 1,
                "index_sha256": "d" * 64,
                "index_bytes": 1,
            },
        },
        "bundle": bundle, "bundle_registry": registry,
        "bundle_contract_sha256": bundle_contract, "control_plane": control,
        "panel": {
            "panel_ref": "panel--fixture", "roles": "final",
            "reference_ref": "root@pin",
            "panel_receipt_sha256": "3" * 64,
            "teacher_receipt_sha256": "4" * 64,
            "teacher_backend_identity_sha256": "5" * 64,
        },
        "reference": {
            "reference_ref": "root@pin",
            "teacher_receipt_sha256": "4" * 64,
            "teacher_backend_identity_sha256": "5" * 64,
        },
        "scope": {"policy": "sealed"},
        "environment": {
            "provider": "runpod",
            "provider_account_id": "fixture-account",
            "provider_gpu_id": "H200",
            "gpu": "H200", "gpus": 1,
            "price_per_gpu_hour": "1",
            "hard_cap_usd": "10",
            "image": "fixture/image@sha256:" + "9" * 64,
            "image_reference_mutable": False,
        },
        "produced_by": {"dependencies": {"profile": "tr3-6bpw"}},
        "measurer": {"name": "selftest"},
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 28,
            "min_memory_gb": 300,
            "expected_vram_bytes": 1,
        },
        "post_create_convergence": {
            "schema": "fidelity-suite/runpod-post-create-convergence.v1",
            "timeout_seconds": 180,
            "poll_seconds": 10,
        },
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
            "workload_deadline_utc": None,
        },
    })
    return finalize_execution(job)


with tempfile.TemporaryDirectory(prefix="engine-profile-argv-") as td:
    job_path = Path(td) / "job.json"
    env = dict(os.environ)

    def install_paid_job(document):
        job_path.write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
        fs_root = document["execution_attempt"]["remote_root"]
        test_engine_root = document["execution_attempt"]["engine_root"]
        env.update({
            "FIDELITY_FS_ROOT": fs_root,
            "FIDELITY_SUITE_ROOT": fs_root,
            "FIDELITY_ENGINE_ROOT": test_engine_root,
            "FIDELITY_ENGINE_PYTHON":
                "%s/venv/bin/python" % test_engine_root,
            "BF16": "%s/models/bf16" % fs_root,
            "TR3_BF16": "%s/models/target-bf16-materialized" % fs_root,
            "QP_PIPELINE_ROOT": "%s/pipeline" % test_engine_root,
        })

    install_paid_job(invocation_job())
    command = [
        sys.executable, str(BIN / "invoke_engine.py"),
        "--job", str(job_path), "--lane", "streaming", "--cold-run", "1",
        "--out", str(Path(td) / "out"), "--print-only",
    ]
    printed = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("print-only argv uses canonical profile/runtime/panel values",
          printed.returncode == 0
          and "--profile tr3-6bpw" in printed.stdout
          and "--source tr3" in printed.stdout
          and "--decode-cache none" in printed.stdout
          and "--decode-threads 28" in printed.stdout
          and "--reduce-order fp32" in printed.stdout
          and "--device cuda" in printed.stdout
          and "--roles final" in printed.stdout
          and "{'profile_id'" not in printed.stdout,
          printed.stdout + printed.stderr)
    for variable in (
            "BF16", "TR3_BF16", "QP_PIPELINE_ROOT",
            "FIDELITY_ENGINE_PYTHON", "FIDELITY_SUITE_ROOT",
            "FIDELITY_ENGINE_ROOT"):
        overridden_env = dict(env)
        overridden_env[variable] = "/ambient/override"
        overridden = subprocess.run(
            command, env=overridden_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
        check("paid invoker refuses ambient %s override" % variable,
              overridden.returncode == 3
              and "must equal canonical paid path" in overridden.stderr,
              overridden.stdout + overridden.stderr)
    missing_fs_env = dict(env)
    missing_fs_env.pop("FIDELITY_FS_ROOT")
    missing_fs = subprocess.run(
        command, env=missing_fs_env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("paid invoker refuses filesystem-root fallback",
          missing_fs.returncode == 3
          and "FIDELITY_FS_ROOT must be an exact absolute path"
          in missing_fs.stderr,
          missing_fs.stdout + missing_fs.stderr)
    native = invocation_job()
    native["profile"].update({
        "profile_id": "native-bf16", "source": "native",
        "surface": "native-bf16", "bits": 16.0,
    })
    native["target"].update({
        "repo_id": "zai-org/GLM-5.3-Flash-BF16",
        "revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
        "surface": "native-bf16", "bits": 16.0,
    })
    native["runtime"]["decode_cache"] = "ram"
    native["timing"]["runtime_profile"]["decode_cache"] = "ram"
    install_paid_job(finalize_execution(native))
    native_printed = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("native print-only argv carries authored reader-pool contract",
          native_printed.returncode == 0
          and "--profile native-bf16" in native_printed.stdout
          and "--source native" in native_printed.stdout
          and "--reduce-order fp32" in native_printed.stdout
          and "--decode-cache ram" in native_printed.stdout
          and "--decode-threads 28" in native_printed.stdout
          and "--device cuda" in native_printed.stdout,
          native_printed.stdout + native_printed.stderr)
    missing_role = invocation_job()
    missing_role["panel"].pop("roles")
    projection = job_identity_projection(missing_role)
    identity = hashlib.sha256(json.dumps(
        projection, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    missing_role["job_id_full"] = identity
    missing_role["job_id"] = identity[:16]
    install_paid_job(missing_role)
    absent = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("print-only refuses absent panel roles rather than defaulting final",
          absent.returncode == 3
          and "quant scoring/panel role contract is incomplete" in absent.stderr,
          absent.stdout + absent.stderr)
    runtime_drift = invocation_job()
    runtime_drift["runtime"]["reader_threads"] = 27
    install_paid_job(finalize_execution(runtime_drift))
    drifted = subprocess.run(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("print-only refuses runtime/timing thread drift",
          drifted.returncode == 3
          and "runtime reader_threads differs from timing evidence" in drifted.stderr,
          drifted.stdout + drifted.stderr)
    install_paid_job(invocation_job())
    wrong_lane = subprocess.run(
        command[:command.index("--lane") + 1] + ["sealed-ep8"]
        + command[command.index("--lane") + 2:],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)
    check("print-only refuses CLI/job lane drift",
          wrong_lane.returncode == 3
          and "differs from job lane" in wrong_lane.stderr,
          wrong_lane.stdout + wrong_lane.stderr)
    scorer_receipts = Path(td) / "score-receipts"
    for run in ("run-1", "run-2"):
        run_dir = scorer_receipts / run
        run_dir.mkdir(parents=True)
        (run_dir / "capture-receipt.json").write_text("{}\n", encoding="utf-8")
    scorer_command = [
        sys.executable, str(BIN / "invoke_scorer.py"),
        "--job", str(job_path), "--lane", "streaming",
        "--receipts", str(scorer_receipts), "--print-only",
    ]
    scored = subprocess.run(
        scorer_command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("K6 scorer print-only argv is fully job-bound",
          scored.returncode == 0
          and "--profile tr3-6bpw" in scored.stdout
          and "--device cuda" in scored.stdout
          and "--chunk-positions 512" in scored.stdout
          and ("--expected-teacher-receipt-sha256 %s" % ("4" * 64))
          in scored.stdout
          and ("--expected-token-panel-receipt-sha256 %s" % ("3" * 64))
          in scored.stdout
          and ("--expected-teacher-backend-identity-sha256 %s" % ("5" * 64))
          in scored.stdout
          and "{'profile_id'" not in scored.stdout,
          scored.stdout + scored.stderr)

    native_score_job = invocation_job()
    native_score_job["profile"].update({
        "profile_id": "native-bf16", "source": "native",
        "surface": "native-bf16", "bits": 16.0,
    })
    native_score_job["target"].update({
        "repo_id": "zai-org/GLM-5.3-Flash-BF16",
        "revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
        "surface": "native-bf16", "bits": 16.0,
    })
    native_score_job["runtime"]["decode_cache"] = "ram"
    native_score_job["timing"]["runtime_profile"]["decode_cache"] = "ram"
    install_paid_job(finalize_execution(native_score_job))
    native_scored = subprocess.run(
        scorer_command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    check("native scorer print-only uses profile_id and exact KLD policy",
          native_scored.returncode == 0
          and "--profile native-bf16" in native_scored.stdout
          and "--device cuda" in native_scored.stdout
          and "--chunk-positions 512" in native_scored.stdout
          and "native-bf16-packed-kld.json" in native_scored.stdout,
          native_scored.stdout + native_scored.stderr)

passed = sum(1 for _, ok, _ in RESULTS if ok)
print("selftest_engine_profiles: %d/%d checks passed" % (passed, len(RESULTS)))
raise SystemExit(0 if passed == len(RESULTS) else 1)
