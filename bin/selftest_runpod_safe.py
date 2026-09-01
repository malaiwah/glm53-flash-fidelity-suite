#!/usr/bin/env python3
"""Offline negative-path checks for the initial safe RunPod controller."""
import ast
import hashlib
import json
import inspect
import sys
import tempfile
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import measure_cloud as MC  # noqa: E402
from fidelity.common import Console  # noqa: E402
from fidelity.hfmeta import RepoMeta  # noqa: E402
from fidelity.runpodapi import RunPod, RunPodError  # noqa: E402
from fidelity import resultsink  # noqa: E402
from fidelity import bench as bench_module  # noqa: E402
from fidelity.runpodsafety import SafetyProofError, _artifact  # noqa: E402


def check(name, value):
    if not value:
        raise AssertionError(name)


def refuses(call):
    try:
        call()
    except (SafetyProofError, RunPodError, ValueError):
        return True
    return False


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); proof = root / "proof.json"; artifact = root / "a.json"
        proof.write_text("{}\n", encoding="utf-8"); artifact.write_bytes(b"{}\n")
        record = {"path": "a.json", "bytes": 3,
                  "sha256": hashlib.sha256(b"{}\n").hexdigest()}
        selected, raw = _artifact(proof, record, "fixture")
        check("valid artifact", selected == artifact and raw == b"{}\n")
        check("traversal", refuses(lambda: _artifact(
            proof, dict(record, path="../a.json"), "fixture")))
        class HostProvider:
            def set_known_hosts(self, path):
                self.path = Path(path)

            def verify_host_key(self, provider_id, expected):
                check("exact provider id reaches host verifier",
                      provider_id == "pod-exact")
                check("operator fingerprint reaches host verifier",
                      expected == "SHA256:" + "A" * 43)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    "198.51.100.7 ssh-ed25519 AAAA\n", encoding="utf-8")
                self.path.chmod(0o600)
                return {
                    "algorithm": "ssh-ed25519",
                    "fingerprint": expected,
                    "host": "198.51.100.7",
                    "port": 22022,
                    "known_hosts_sha256": "b" * 64,
                }

        host_provider = HostProvider()
        host_args = type("HostArgs", (), {
            "runpod_host_key_sha256": "SHA256:" + "A" * 43,
        })()
        host_evidence = MC._authenticate_runpod_ssh_host(
            host_args, Console(), host_provider, "pod-exact", root / "run")
        host_proof = json.loads(
            host_evidence["path"].read_text(encoding="utf-8"))
        resultsink._validate_runpod_host_key_proof(
            {"execution_attempt": {"kind": "runpod-ssh"}},
            host_proof, {"provider_id": "pod-exact"})
        check("operator-authenticated host proof is sealed and persisted",
              host_evidence["proof"]["proof_sha256"]
              == host_proof["proof_sha256"]
              and host_evidence["path"].is_file())
    dry = RunPod(dry=True)
    dry._validated_ssh_public_key = lambda: "ssh-ed25519 AAAA"
    terminate_after = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 600))
    created = dry.create(gpu_type="NVIDIA L4", storage_gb=20,
                         container_disk_gb=20, region="secure",
                         name="fidcloud-" + "a" * 32,
                         terminate_after=terminate_after)
    check("dry create", created["dry_run"] is True)
    check("network volume refused", refuses(lambda: dry.create(
        gpu_type="NVIDIA L4", storage_gb=20, container_disk_gb=20,
        network_volume_id="volume", terminate_after=terminate_after)))
    original_repo_meta = MC.repo_meta
    provider_touched = []
    try:
        MC.repo_meta = lambda *_args, **_kwargs: RepoMeta(
            repo_id="malaiwah/GLM-5.3-Flash-TR3-8bpw",
            repo_type="model",
            revision="7199f6f1a211084c240614806f046f11a52dad64",
            requested_revision="7199f6f1a211084c240614806f046f11a52dad64",
            last_modified=None, files=[], author="malaiwah", private=False)

        class UntouchedProvider:
            def __getattr__(self, name):
                provider_touched.append(name)
                raise AssertionError("provider touched during K8 refusal")

        args = type("Args", (), {
            "role": "quant",
            "model": "malaiwah/GLM-5.3-Flash-TR3-8bpw",
            "revision": "7199f6f1a211084c240614806f046f11a52dad64",
            "lane": "streaming",
        })()
        try:
            MC._plan_runpod_anonymous(
                args, Console(), UntouchedProvider(), {})
        except MC.Refusal as exc:
            check("K8 names its pinned missing verdict bridge",
                  "missing_sealed_surface_measurement_bridge" in exc.reason
                  and not provider_touched)
        else:
            raise AssertionError("K8 paid plan was admitted")
    finally:
        MC.repo_meta = original_repo_meta
    check("M2 native quant is outside paid admission",
          not MC._safe_runpod_target_allowed(
              "quant", "zai-org/GLM-5.3-Flash-BF16",
              MC.OFFICIAL_BF16_REVISION))
    check("M2 and Fruit exact roots remain admitted",
          MC._safe_runpod_target_allowed(
              "root", "zai-org/GLM-5.3-Flash-BF16",
              MC.OFFICIAL_BF16_REVISION)
          and MC._safe_runpod_target_allowed(
              "root", "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
              "ef68013aa6e16453cf52b5b77647f72fbe258c3c"))
    check("parser no longer invents maintainer attribution",
          MC.build_parser().parse_args([]).measurer is None)
    placeholder_quant = type("IdentityArgs", (), {
        "role": "quant", "measurer": "YOUR_HF_HANDLE", "spot": False,
    })()
    check("documented measurer placeholder is refused",
          any("--measurer" in item
              for item in MC._runpod_forbidden(placeholder_quant)))
    placeholder_root = type("RootIdentityArgs", (), {
        "role": "root", "measurer": "real-handle", "spot": False,
        "dataset_id": "REPLACE",
        "dataset_name": "REPLACE",
        "dataset_repository": "YOUR_HANDLE/REPLACE",
    })()
    identity_refusals = MC._runpod_forbidden(placeholder_root)
    check("root dataset id/name/repository placeholders are all refused",
          all(any("--%s" % field in item for item in identity_refusals)
              for field in (
                  "dataset-id", "dataset-name", "dataset-repository")))

    original_bench = bench_module.bench_existing
    try:
        bench_module.bench_existing = lambda *_a, **_kw: (_ for _ in ()).throw(
            RuntimeError("benchmark transport failed"))
        bench_args = type("BenchArgs", (), {
            "no_preflight_bench": False,
            "min_h2d_gbps": None,
            "min_gemm_tflops": None,
        })()
        bench_td = type("BenchTD", (), {"machine_id": "pod"})()
        try:
            MC._preflight_bench(
                bench_args, Console(), object(), bench_td, {},
                fail_closed=True, python_executable="/venv/bin/python",
                remote_payload="/sealed/cardbench_payload.py")
            bench_failed_closed = False
        except MC.Refusal:
            bench_failed_closed = True
        check("safe-route benchmark transport errors fail closed",
              bench_failed_closed)
        complete_bench = {
            "gpu": "NVIDIA H200", "torch": "2.11.0", "cuda": "13.0",
            "h2d_GBps": 7.0, "h2d_cold_GBps": 6.0,
            "expert_gemm_TFLOPs": 100.0, "stream_matrix_ms": 1.0,
        }
        bench_module.bench_existing = lambda *_a, **_kw: complete_bench
        bench_args.min_h2d_gbps = 8.0
        try:
            MC._preflight_bench(
                bench_args, Console(), object(), bench_td, {},
                fail_closed=True, python_executable="/venv/bin/python",
                remote_payload="/sealed/cardbench_payload.py")
            threshold_refused = False
        except MC.Refusal:
            threshold_refused = True
        check("configured safe-route benchmark thresholds gate", threshold_refused)
        bench_args.min_h2d_gbps = None
        bench_module.bench_existing = lambda *_a, **_kw: {
            "h2d_GBps": 999.0, "expert_gemm_TFLOPs": 999.0,
        }
        try:
            MC._preflight_bench(
                bench_args, Console(), object(), bench_td, {},
                fail_closed=True, python_executable="/venv/bin/python",
                remote_payload="/sealed/cardbench_payload.py")
            incomplete_refused = False
        except MC.Refusal:
            incomplete_refused = True
        check("safe-route benchmark requires complete measured identity",
              incomplete_refused)
        check("thresholds refuse absent measurements",
              "host->device bandwidth was not measured" in
              str(bench_module.gate({}, min_h2d_gbps=8.0)))
    finally:
        bench_module.bench_existing = original_bench

    class SealedBenchProvider:
        def __init__(self):
            self.uploaded = []
            self.commands = []

        def upload(self, *_args):
            self.uploaded.append(_args)

        def exec_stdout(self, _machine_id, command, timeout):
            self.commands.append((command, timeout))
            return '{"stream_matrix_ms": 1.0}'

    sealed_bench_provider = SealedBenchProvider()
    bench_module.bench_existing(
        sealed_bench_provider, "pod",
        python_executable="/workspace/engine/venv/bin/python",
        remote_payload="/workspace/run/bin/fidelity/cardbench_payload.py")
    check("safe benchmark uses sealed payload and exact venv without upload",
          not sealed_bench_provider.uploaded
          and sealed_bench_provider.commands
          and "/workspace/engine/venv/bin/python" in
          sealed_bench_provider.commands[0][0]
          and "/workspace/run/bin/fidelity/cardbench_payload.py" in
          sealed_bench_provider.commands[0][0])
    def function_calls(function, name):
        tree = ast.parse(inspect.getsource(function))
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == name)
                 or (isinstance(node.func, ast.Attribute)
                     and node.func.attr == name))
        ]

    plan_proof_calls = function_calls(
        MC._plan_runpod_anonymous, "validate_safety_proof")
    execute_proof_calls = function_calls(
        MC.execute_runpod, "validate_safety_proof")
    check("both paid proof callsites receive the current campaign ledger",
          len(plan_proof_calls) == len(execute_proof_calls) == 1
          and len(plan_proof_calls[0].args) == 5
          and "args.campaign_ledger" in ast.unparse(
              plan_proof_calls[0].args[4])
          and len(execute_proof_calls[0].args) == 5
          and ast.unparse(execute_proof_calls[0].args[4]) == "ledger_path")
    check("response-loss handling cannot issue a second provider POST",
          len(function_calls(
              MC.execute_runpod, "submit_prepared_create")) == 1)
    check("live-checkout reaper commands cannot author installed health",
          function_calls(
              MC._runpod_reaper_command, "write_reaper_health") == [])

    from fidelity.campaign import CampaignLedger  # noqa: E402
    with tempfile.TemporaryDirectory() as campaign_td:
        campaign_root = Path(campaign_td)
        missing_path = campaign_root / "missing-campaign.json"
        campaign_args = type("CampaignArgs", (), {
            "campaign_ledger": str(missing_path),
            "campaign_ceiling": "10",
            "campaign_reserve": "1",
            "campaign_reaper_margin": "1",
        })()
        missing_refused = False
        try:
            MC._open_existing_runpod_campaign(
                campaign_args, "account-selftest")
        except MC.Refusal:
            missing_refused = True
        check("normal paid admission never recreates a missing campaign ledger",
              missing_refused and not missing_path.exists())
        CampaignLedger.create(
            str(missing_path), "10", "1", "1",
            max_concurrent_attempts=2, provider="runpod",
            provider_account_id="account-selftest")
        opened_path, opened_ledger = MC._open_existing_runpod_campaign(
            campaign_args, "account-selftest")
        swapped_args = type("SwappedCampaignArgs", (), {
            "campaign_ledger": str(missing_path),
            "campaign_ceiling": "11",
            "campaign_reserve": "1",
            "campaign_reaper_margin": "1",
        })()
        swapped_refused = False
        try:
            MC._open_existing_runpod_campaign(
                swapped_args, "account-selftest")
        except MC.Refusal:
            swapped_refused = True
        check("normal paid admission opens only the exact existing campaign",
              opened_path == str(missing_path.resolve())
              and opened_ledger.snapshot()["provider_account_id"]
                  == "account-selftest"
              and swapped_refused)

    print("PASS: safe RunPod offline guards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
