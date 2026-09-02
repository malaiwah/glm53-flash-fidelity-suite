#!/usr/bin/env python3
"""Offline negative-path checks for the initial safe RunPod controller."""
import ast
import hashlib
import io
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
import fidelity.runpodapi as runpodapi_module  # noqa: E402
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

            def ssh_host_ed25519_fingerprint(self, provider_id):
                check("exact provider id reaches authenticated log API",
                      provider_id == "pod-exact")
                provider_log_line = (
                    "256 SHA256:%s fixture (ED25519)" % ("A" * 43))
                return {
                    "schema":
                        "fidelity-suite/runpod-host-key-log-evidence.v1",
                    "provider": "runpod",
                    "provider_id": provider_id,
                    "endpoint_origin": "https://api.runpod.io",
                    "source": "container",
                    "tail": 5000,
                    "observed_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "line": provider_log_line,
                    "line_sha256": hashlib.sha256(
                        provider_log_line.encode("utf-8")).hexdigest(),
                    "fingerprint": "SHA256:" + "A" * 43,
                }

            def verify_host_key(self, provider_id, expected):
                check("exact provider id reaches host verifier",
                      provider_id == "pod-exact")
                check("provider-log fingerprint reaches host verifier",
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
        host_evidence = MC._authenticate_runpod_ssh_host(
            Console(), host_provider, "pod-exact", root / "run")
        host_proof = json.loads(
            host_evidence["path"].read_text(encoding="utf-8"))
        resultsink._validate_runpod_host_key_proof(
            {"execution_attempt": {"kind": "runpod-ssh"}},
            host_proof, {"provider_id": "pod-exact"})
        check("provider-log-authenticated host proof is sealed and persisted",
              host_evidence["proof"]["proof_sha256"]
              == host_proof["proof_sha256"]
              and host_proof["verification_source"]
                  == "runpod-authenticated-v2-container-log"
              and host_evidence["path"].is_file())
    class LogResponse:
        def __init__(self, lines, content_type="text/event-stream"):
            self.stream = io.BytesIO(b"".join(lines))
            self.headers = {
                "Content-Type": content_type,
                "Date": "Tue, 01 Sep 2026 00:00:00 GMT",
            }

        def __enter__(self):
            return self

        def __exit__(self, *_unused):
            return False

        def readline(self, size=-1):
            return self.stream.readline(size)

    log_requests = []
    log_lines = [(
        b'data:{"source":"container","line":"256 SHA256:'
        + b"A" * 43 + b' fixture (ED25519)"}\n')]
    original_urlopen = runpodapi_module.safe_urlopen
    try:
        def log_urlopen(request, *, timeout):
            log_requests.append((request, timeout))
            return LogResponse(log_lines)

        runpodapi_module.safe_urlopen = log_urlopen
        log_provider = RunPod(dry=False, key_file="/not/read")
        log_provider._key = "fixture-secret"
        log_evidence = log_provider.ssh_host_ed25519_fingerprint("pod-exact")
        request, timeout = log_requests[-1]
        check("authenticated v2 logs yield the exact container ED25519 key",
              log_evidence["fingerprint"] == "SHA256:" + "A" * 43
              and log_evidence["source"] == "container"
              and log_evidence["tail"] == 5000
              and timeout == 60.0)
        check("RunPod API key stays in a request header",
              "fixture-secret" not in request.full_url
              and request.get_header("Authorization")
                  == "Bearer fixture-secret"
              and request.get_header("User-agent")
                  == "quant-fidelity-suite/0.1")
        log_lines[:] = [
            b'data:{"source":"system","line":"256 SHA256:'
            + b"A" * 43 + b' fixture (ED25519)"}\n']
        check("non-container fingerprint logs fail closed", refuses(
            lambda: log_provider.ssh_host_ed25519_fingerprint("pod-exact")))
        log_lines[:] = [
            b'data:{"source":"container","line":"not a host key"}\n']
        check("malformed fingerprint logs fail closed", refuses(
            lambda: log_provider.ssh_host_ed25519_fingerprint("pod-exact")))
        log_lines[:] = [b"x" * (64 * 1024 + 1) + b"\n"]
        check("oversized provider log lines fail before unbounded parsing",
              refuses(lambda:
                  log_provider.ssh_host_ed25519_fingerprint("pod-exact")))
    finally:
        runpodapi_module.safe_urlopen = original_urlopen
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
    check("M2, Fruit, and full GLM exact roots remain admitted",
          MC._safe_runpod_target_allowed(
              "root", "zai-org/GLM-5.3-Flash-BF16",
              MC.OFFICIAL_BF16_REVISION)
          and MC._safe_runpod_target_allowed(
              "root", "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
              "ef68013aa6e16453cf52b5b77647f72fbe258c3c")
          and MC._safe_runpod_target_allowed(*MC._FULL_GLM53_ROOT))
    full_timing = MC.resolve_root_timing(
        target_repo=MC._FULL_GLM53_ROOT[1],
        target_revision=MC._FULL_GLM53_ROOT[2],
        gpu="H200", form="hidden",
        schedule="two-fresh-process-qualification")
    check("full GLM timing is exact-target and identity-bound",
          full_timing["conservative_upper_hours"] == 3.5
          and full_timing["model_identity"] == {
              "model_bytes": 1506659919872,
              "config_sha256":
                  "ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca",
              "index_sha256":
                  "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
          })
    check("full GLM authored source-license pin is exact",
          MC._FULL_GLM53_LICENSE == {
              "source_path": "LICENSE",
              "dataset_path": "LICENSE",
              "bytes": 4263,
              "sha256":
                  "96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c",
          })
    fixture_license = b"fixture source weights license\n"
    fixture_contract = {
        "source_path": "LICENSE", "dataset_path": "LICENSE",
        "bytes": len(fixture_license),
        "sha256": hashlib.sha256(fixture_license).hexdigest(),
    }
    full_meta = RepoMeta(
        repo_id=MC._FULL_GLM53_ROOT[1], repo_type="model",
        revision=MC._FULL_GLM53_ROOT[2],
        requested_revision=MC._FULL_GLM53_ROOT[2],
        last_modified=None, files=[("LICENSE", len(fixture_license))],
        author="zai-org", private=False)
    original_fetch_file = MC.fetch_file
    original_license_contract = MC._FULL_GLM53_LICENSE
    try:
        MC.fetch_file = lambda *_args, **_kwargs: fixture_license
        MC._FULL_GLM53_LICENSE = fixture_contract
        bound_license = MC._root_dataset_license_contract(full_meta)
        check("full GLM root copies the exact source license",
              bound_license == {
                  "dataset_license": "other",
                  "weights_license": fixture_contract,
              })
        MC.fetch_file = lambda *_args, **_kwargs: fixture_license + b"x"
        try:
            MC._root_dataset_license_contract(full_meta)
        except MC.Refusal:
            mismatch_refused = True
        else:
            mismatch_refused = False
        check("source-license byte drift refuses before spend",
              mismatch_refused)
    finally:
        MC.fetch_file = original_fetch_file
        MC._FULL_GLM53_LICENSE = original_license_contract
    parser_defaults = MC.build_parser().parse_args([])
    check("parser no longer invents maintainer attribution",
          parser_defaults.measurer is None)
    check("RunPod download credential has no ambient/default source",
          parser_defaults.hf_download_token_file is None)
    placeholder_quant = type("IdentityArgs", (), {
        "role": "quant", "measurer": "YOUR_HF_HANDLE", "spot": False,
    })()
    check("documented measurer placeholder is refused",
          any("--measurer" in item
              for item in MC._runpod_forbidden(placeholder_quant)))
    check("safe RunPod requires an explicit download-token file",
          any("--hf-download-token-file" in item
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
    check("paid executor installs and scopes the authenticated target token",
          len(function_calls(MC.execute_runpod, "_transport_hf_token")) == 1
          and len(function_calls(
              MC.execute_runpod,
              "_runpod_fetch_target_and_remove_token")) == 1)
    token_install_calls = function_calls(
        MC.execute_runpod, "_transport_hf_token")
    stage_sequence_calls = function_calls(
        MC.execute_runpod, "stage_sequence")
    target_fetch_calls = function_calls(
        MC.execute_runpod, "_runpod_fetch_target_and_remove_token")
    check("token is installed before setup can create .secrets and removed "
          "inside fetch_target",
          len(token_install_calls) == len(stage_sequence_calls)
              == len(target_fetch_calls) == 1
          and token_install_calls[0].lineno < stage_sequence_calls[0].lineno
              < target_fetch_calls[0].lineno)
    check("paid planning validates the download token before provider access",
          len(function_calls(
              MC._plan_runpod_anonymous,
              "_load_required_hf_download_token")) == 1)
    check("paid execution reloads the token immediately before mutation",
          len(function_calls(
              MC._main_runpod, "_load_required_hf_download_token")) == 1)
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
