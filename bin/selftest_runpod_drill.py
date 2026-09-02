#!/usr/bin/env python3
"""Offline failure-closure and evidence fixtures for runpoddrill."""
from __future__ import annotations
import base64, hashlib, json, os, shlex, shutil, signal, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from fidelity import runpoddrill as RD  # noqa: E402
from fidelity.campaign import CampaignLedger  # noqa: E402
from fidelity.cloudlease import LeaseStore, reap_once  # noqa: E402
from fidelity.jobcontract import finalize_bundle_manifest, finalize_job  # noqa: E402
from fidelity.jlapi import GpuOffer  # noqa: E402
from fidelity.resultsink import build_summary, verify_archive, write_archive  # noqa: E402
from fidelity.runpodsafety import SafetyProofError, validate_safety_proof  # noqa: E402


def check(name, value):
    if not value: raise AssertionError(name)

def refuses(fn, kind=Exception):
    try: fn()
    except kind: return True
    return False

class Clock:
    def __init__(self): self.now = 1788220800.0  # 2026-09-01T00:00:00Z
    def time(self): return self.now
    def sleep(self, seconds): self.now += float(seconds)

class Prepared:
    def __init__(self, kw):
        self.kw = dict(kw)
        self.identity = {
            "cloud_type": "SECURE", "is_spot": False, "offer": "on-demand",
            "gpu_type_id": RD.GPU_TYPE, "gpu_count": 1,
            "volume_gb": kw["storage_gb"],
            "container_disk_gb": kw["container_disk_gb"],
            "min_vcpu": kw["min_vcpu"], "min_ram_gb": kw["min_ram_gb"],
            "name": kw["name"], "image_name": kw["image"],
            "terminate_after": kw["terminate_after"], "ports": "22/tcp",
            "volume_mount_path": "/workspace", "network_volume_id": None,
            "public_key_sha256": hashlib.sha256(
                b"fixture-public-key").hexdigest(),
        }
        self.body = json.dumps({
            "query": "mutation { podFindAndDeployOnDemand(input:{}) { id } }"
        }).encode("utf-8")

    def to_dict(self):
        return {
            "schema": "fidelity-suite/runpod-prepared-create.v1",
            "request_identity": dict(self.identity),
            "graphql_body_sha256": hashlib.sha256(self.body).hexdigest(),
            "graphql_body_bytes": len(self.body),
            "graphql_body_base64": base64.b64encode(self.body).decode("ascii"),
        }


class Provider:
    def __init__(self, clock, remote):
        self.clock, self.remote = clock, Path(remote)
        self.key = self.ssh = self.complete = True
        self.balance_value = "10.00"; self.inventory_present = False
        self.create_calls = self.prepare_calls = self.destroy_calls = 0
        self.pod = None; self.termination = None; self.created_at = None
        self.post_create_visibility_delay = 0
        self.post_create_inventory_calls = 0
        self.post_create_wrong_pod = self.post_create_network_volume = False
        self.create_response_wrong_name = False
        self.create_rate = self.graphql_rate = self.rest_rate = "0.44"
        self.termination_offset = 901
        self.lifecycle_observations = []
        self.lifecycle_poll_latency = 0
        self.status_latency = 0
        self.download_calls = []
        self.bounded_download_calls = []
        self.bounded_modes = {}
        self.download_replacements = {}
        self.receipt_padding_bytes = 0
        self.remote_archive_failure = False
        self.billing_delay_after_absence_seconds = 0
        self.gpu_queries = []
    def require(self):
        if not self.key: raise RD.DrillError("missing key")
        return (0, 0, 0)
    def preflight_ssh_key(self): return "fixture-public-key" if self.ssh else ""
    def status(self):
        if self.status_latency:
            self.clock.sleep(self.status_latency)
        return {"id": "fixture-runpod-account",
                "clientBalance": self.balance_value,
                "currentSpendPerHr": "0",
                "observed_at_utc": RD._utc(self.clock.time())}
    def balance(self): return self.balance_value
    def chargeable_inventory(self):
        rows = ([{"id": "existing", "name": "other", "status": "RUNNING"}]
                if self.inventory_present else [])
        volumes = []
        visible = (
            self.pod is not None and self.created_at is not None
            and self.clock.time()
            >= self.created_at + self.post_create_visibility_delay)
        if self.pod is not None:
            self.post_create_inventory_calls += 1
        if visible:
            strict = dict(self.pod)
            strict["cost_per_hr"] = self.rest_rate
            rows.append(strict)
            if self.post_create_wrong_pod:
                rows.append({
                    "id": "foreign-pod", "name": "somebody-else",
                    "status": "RUNNING"})
            if self.post_create_network_volume:
                volumes.append({
                    "id": "foreign-volume", "name": "somebody-volume",
                    "status": "READY"})
        family = lambda resources: {"complete": self.complete, "resources": resources,
                                    "source": "fixture"}
        return {"schema": "fidelity-suite/runpod-chargeable-inventory.v1",
                "provider": "runpod", "complete": self.complete,
                "unknown_families": [],
                "observed_at_utc": RD._utc(self.clock.time()),
                "families": {"pods": family(rows),
                             "network_volumes": family(volumes)}}
    def gpus(self, *, gpu_type=None, secure_only=False):
        self.gpu_queries.append((gpu_type, secure_only))
        if gpu_type != RD.GPU_TYPE or secure_only is not True:
            raise AssertionError("drill did not request its exact secure GPU")
        return [GpuOffer(RD.GPU_TYPE, "secure", 24 * 1024 ** 3, .44,
                         False, 1, "container", {})]
    def list_instances(self):
        if self.lifecycle_poll_latency:
            self.clock.sleep(self.lifecycle_poll_latency)
        if self.inventory_present and self.pod is None:
            return [{"id": "existing", "name": "other", "status": "RUNNING"}]
        if (self.pod and self.clock.time()
                >= self.termination + self.termination_offset):
            self.pod = None
        visible = (
            self.pod is not None and self.created_at is not None
            and self.clock.time()
            >= self.created_at + self.post_create_visibility_delay)
        self.lifecycle_observations.append(
            (self.clock.time(), bool(visible)))
        if not visible:
            return []
        lifecycle = dict(self.pod)
        lifecycle["cost_per_hr"] = self.graphql_rate
        return [lifecycle]
    def server_time_evidence(self, **limits):
        now = self.clock.time()
        return {
            "schema": "fidelity-suite/runpod-server-time.v1",
            "endpoint_origin": "https://api.runpod.io",
            "date_header": "Tue, 01 Sep 2026 00:00:00 GMT",
            "server_epoch": now, "local_received_epoch": now,
            "local_minus_server_seconds": 0.0, "checked_at_epoch": now,
            "evidence_age_seconds": 0.0,
            "max_clock_delta_seconds":
                float(limits["max_clock_delta_seconds"]),
            "max_evidence_age_seconds":
                float(limits["max_evidence_age_seconds"]),
        }
    def prepare_safe_create(self, **kw):
        self.prepare_calls += 1
        check("one outstanding create preparation",
              self.prepare_calls == self.create_calls + 1)
        return Prepared(kw)
    def submit_prepared_create(self, prepared):
        self.create_calls += 1
        check("one create per preparation",
              self.create_calls == self.prepare_calls)
        kw = prepared.kw
        self.termination = RD._utc_epoch(kw["terminate_after"], "terminateAfter")
        self.created_at = self.clock.time()
        self.pod = {
            "id": "pod-1", "name": kw["name"], "status": "RUNNING",
            "cost_per_hr": self.graphql_rate}
        if self.create_response_wrong_name:
            self.pod["name"] = "wrong-response-name"
            raise RD.RunPodCreateResponseError(
                "fixture unqualified response", "pod-1", {
                    "id": "pod-1", "name": self.pod["name"],
                    "cost_per_hr": self.create_rate,
                })
        return {
            "pod_id": "pod-1", "machine_id": "pod-1", "name": kw["name"],
            "cost_per_hr": self.create_rate, "request": dict(prepared.identity),
            "prepared_create": prepared.to_dict(),
        }
    def validate_safe_resource_binding(self, provider_id, **expected):
        observed = self.list_instances()
        check("exact live pod", provider_id == "pod-1" and len(observed) == 1)
        return {"provider_id": provider_id, "passed": True,
                "expected": expected, "observed": observed[0]}
    def destroy(self, provider_id):
        self.destroy_calls += 1; self.pod = None
    def exec(self, provider_id, command, timeout=0):
        if command.startswith("rm -rf"):
            shutil.rmtree(self.remote / "fidelity-drill", ignore_errors=True)
            self.remote.mkdir(parents=True, exist_ok=True)
        elif command.startswith("python3"):
            if self.remote_archive_failure:
                raise RuntimeError("fixture remote archive builder failed")
            root = self.remote / "fidelity-drill" / "result"
            (root / "logs" / "drill.log").write_text(
                "intentional controller-loss drill\n", encoding="utf-8")
            job = json.loads((root / "job.json").read_text())
            summary = build_summary(root, "stage", "abandoned", ["drill"],
                                    failed_stage="drill")
            summary["utc"] = job["execution_attempt"]["planned_at"]
            archive = self.remote / "fidelity-drill" / "result-bundle.tar.gz"
            archive_record = write_archive(root, summary, archive)
            transfer = RD._seal({
                "schema": RD.TRANSFER_SCHEMA, "receipt_sha256": "",
                "path": "result-bundle.tar.gz",
                "bytes": archive_record["bytes"],
                "sha256": archive_record["sha256"],
                "job_id_full": job["job_id_full"]}, "receipt_sha256")
            receipt = self.remote / "fidelity-drill" / "result-transfer.json"
            receipt.write_text(json.dumps(transfer, sort_keys=True) + "\n")
            if self.receipt_padding_bytes:
                with receipt.open("ab") as stream:
                    stream.write(b" " * self.receipt_padding_bytes)
        elif command.startswith("test -f"):
            remote = Path(shlex.split(command)[-1])
            selected = self.remote / "fidelity-drill" / remote.name
            if not selected.is_file() or selected.is_symlink():
                return {"exit_code": 1, "stdout": "", "stderr": "unsafe"}
            return {"exit_code": 0, "stdout": "%d\n" % selected.stat().st_size,
                    "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}
    def upload(self, provider_id, local, remote):
        shutil.copytree(local, self.remote / Path(local).name); return {"ok": True}
    def download(self, provider_id, remote, local, recursive=False, timeout=0):
        self.download_calls.append((remote, local))
        raise AssertionError("unbounded drill download is forbidden")
    def download_bounded(self, provider_id, remote, local, *,
                         expected_bytes, max_bytes, timeout=0):
        name = Path(remote).name
        self.bounded_download_calls.append({
            "name": name, "expected_bytes": expected_bytes,
            "max_bytes": max_bytes, "timeout": timeout})
        if self.bounded_modes.get(name) == "stalled":
            raise TimeoutError("fixture bounded download stalled")
        body = (self.remote / "fidelity-drill" / name).read_bytes()
        replacement = self.download_replacements.get(name)
        if replacement is not None:
            body = replacement(body) if callable(replacement) else replacement
        if self.bounded_modes.get(name) == "short":
            body = body[:-1]
        elif self.bounded_modes.get(name) == "oversized":
            body += b"x"
        if len(body) != expected_bytes or len(body) > max_bytes:
            raise RD.DrillError("fixture bounded download size mismatch")
        Path(local).write_bytes(body)
        return {"ok": True, "bytes": len(body)}
    def reconcile_billing(self, lease):
        pod = lease["provider_resource_ids"][0]
        absence = next(
            item["at"] for item in reversed(lease["history"])
            if item.get("to") == "ABSENCE_CONFIRMED")
        if (self.clock.time()
                < RD._utc_epoch(absence, "absence")
                + self.billing_delay_after_absence_seconds):
            raise RuntimeError("fixture billing remains pending")
        stamp = RD._utc(self.clock.time())
        row = {"podId": pod, "time": stamp, "totalAmount": "0.08",
               "gpuAmount": "0.06", "cpuAmount": "0.01", "diskAmount": "0.01"}
        totals = {key: row[key] for key in
                  ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount")}
        history = {"schema": "fidelity-suite/runpod-billing-evidence.v2",
                   "provider": "runpod", "pod_id": pod,
                   "query": {"podId": pod}, "records": [row],
                   "metadata": {"query": {"podId": pod}, "totals": totals},
                   "validated_record_sums": totals,
                   "validated_bucket_ranges": [{
                       "startTime": stamp, "endTime": stamp}],
                   "retrieved_at_utc": stamp}
        closure = {
            "reconciled": True, "provider": "runpod",
            "provider_resource_ids": [pod], "billing_histories": [history],
            "total_amount": "0.08",
        }
        normalized = json.loads(json.dumps(closure))
        for item in normalized["billing_histories"]:
            item.pop("retrieved_at_utc", None)
        closure["evidence"] = {
            "schema": "fidelity-suite/runpod-billing-stabilization.v1",
            "absence_confirmed_at": absence,
            "minimum_stabilization_seconds": 300,
            "closure_sha256": RD._sha256(normalized),
            "first_retrieval": {
                "schema": "fidelity-suite/runpod-billing-retrieval.v1",
                "retrieval_id": "1" * 24,
                "retrieved_at_utc": stamp,
            },
            "second_retrieval": {
                "schema": "fidelity-suite/runpod-billing-retrieval.v1",
                "retrieval_id": "2" * 24,
                "retrieved_at_utc": stamp,
            },
        }
        return closure

class InlineSupervisor:
    controller_holds = False
    def supervise(self, controller, ready, deadline, clock, prepare):
        startup = prepare(os.getpid())
        controller(startup)
        raw = ready.read_bytes(); state = json.loads(raw)
        return RD._seal({"schema": RD.KILL_EVENT_SCHEMA, "receipt_sha256": "",
                         "controller_pid": state["controller_pid"],
                         "signal": "SIGKILL",
                         "ready_state_sha256": hashlib.sha256(raw).hexdigest(),
                         "killed_at": RD._utc(clock.time()), "wait_status": 9},
                        "receipt_sha256")


class AutonomousTimer:
    def __init__(self, clock, state_dir, lease_dir, healthy=True):
        self.clock = clock
        self.state_dir = Path(state_dir)
        self.lease_dir = Path(lease_dir)
        self.healthy = healthy
        source_files = [
            {"path": logical, "size": len(path.read_bytes()),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for logical, path in (
                ("bin/reap_cloud_leases.py", ROOT / "bin/reap_cloud_leases.py"),
                ("bin/fidelity/__init__.py", ROOT / "bin/fidelity/__init__.py"),
                ("bin/fidelity/cloudlease.py", ROOT / "bin/fidelity/cloudlease.py"),
                ("bin/fidelity/campaign.py", ROOT / "bin/fidelity/campaign.py"),
                ("bin/fidelity/common.py", ROOT / "bin/fidelity/common.py"),
                ("bin/fidelity/runpodapi.py", ROOT / "bin/fidelity/runpodapi.py"),
                ("bin/fidelity/jlapi.py", ROOT / "bin/fidelity/jlapi.py"),
                ("bin/fidelity/sshbase.py", ROOT / "bin/fidelity/sshbase.py"))
        ]
        source_files.sort(key=lambda row: row["path"])
        self.control = {
            "command_sha256": "1" * 64,
            "source_command_sha256": "2" * 64,
            "service_unit": "fidelity-cloud-reaper.service",
            "service_unit_sha256": "3" * 64,
            "timer_unit": "fidelity-cloud-reaper.timer",
            "timer_unit_sha256": "4" * 64,
            "source_files": source_files,
            "runtime_files": json.loads(json.dumps(source_files)),
            "interpreter": {
                "executable_path_sha256": "7" * 64,
                "executable_file_sha256": "8" * 64,
                "version": "3.9",
                "implementation": "cpython",
            },
            "state_dir_sha256": hashlib.sha256(
                str(self.state_dir).encode()).hexdigest(),
            "lease_dir_sha256": hashlib.sha256(
                str(self.lease_dir).encode()).hexdigest(),
            "provider": "runpod",
            "provider_account_id_sha256": hashlib.sha256(
                b"fixture-runpod-account").hexdigest(),
        }
        self.control["control_sha256"] = RD._sha256(self.control)
        self.stamp = self._stamp([], self.clock.time() - 60,
                                 self.clock.time() - 59, "0" * 32)
        self._write()
    def _stamp(self, actions, started, completed, invocation_id):
        stamp = {
            "schema": RD.HEALTH_SCHEMA,
            "invocation_id": invocation_id,
            "invocation_started_at_epoch": started,
            "invocation_started_at_utc": RD._utc(started),
            "completed_at_epoch": completed,
            "completed_at_utc": RD._utc(completed),
            "control": self.control,
            "actions": actions,
            "ok": self.healthy,
            "failure_count": 0 if self.healthy else 1,
            "unresolved_count": 0,
            "result_sha256": "6" * 64,
        }
        stamp["record_sha256"] = hashlib.sha256(
            RD.canonical_bytes(stamp)).hexdigest()
        return stamp
    def _write(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "reaper-health.json").write_text(
            json.dumps(self.stamp, sort_keys=True) + "\n")
    def health(self, **unused):
        return {"ok": self.healthy, "stamp_ok": self.healthy,
                "control_ok": self.healthy, "stamp": self.stamp}
    def tick(self, *, plan, store, provider, now):
        started = self.clock.time()
        self.clock.sleep(1)
        result = reap_once(
            store, {"runpod": provider}, now=self.clock.time(), dry_run=False)
        completed = self.clock.time()
        self.stamp = self._stamp(
            list(result.actions), started, completed,
            ("%032x" % int(completed))[-32:])
        self._write()

def job_fixture():
    bundle = finalize_bundle_manifest([{"path": "bin/fidelity/runpoddrill.py",
        "bytes": 1, "sha256": "7" * 64}], "BUNDLE.txt")
    control = finalize_bundle_manifest([{"path": "bin/measure_cloud.py",
        "bytes": 1, "sha256": "8" * 64}], "authored-control-plane-closure")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    registry = {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "9" * 64}
    contract = hashlib.sha256(json.dumps({"bundle": bundle, "registry": registry},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    shards = [{"path": "tiny.safetensors", "bytes": 1}]
    census = hashlib.sha256(json.dumps(shards, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return finalize_job({"schema": "fidelity-suite/job.v2", "role": "quant",
        "recipe": "runpod-controller-loss-drill",
        "lane": "fault-drill", "cold_runs": 2,
        "target": {"repo_id": "fixture/tiny", "revision": "1" * 40,
          "config_sha256": "a" * 64, "index_sha256": "b" * 64,
          "model_bytes": 1, "shards": shards, "shard_manifest_sha256": census,
          "download_bytes_total": 1, "download_manifest": shards,
          "download_manifest_sha256": census},
        "bundle": bundle, "bundle_registry": registry,
        "bundle_contract_sha256": contract, "control_plane": control,
        "panel": {"id": "drill"}, "reference": {}, "scope": {"kind": "drill"},
        "profile": {"profile_id": "runpod-drill-secure-l4-on-demand",
                    "lane": "fault-drill"},
        "timing": {"seconds": 1},
        "execution_attempt": {"kind": "runpod-ssh", "attempt_id": None,
          "cost_quote": None, "engine_root": None,
          "execution_contract_sha256": None, "lease_path": None,
          "planned_at": "2026-09-01T00:00:00Z",
          "pre_create_safety": None, "prepared_create": None,
          "remote_root": None, "provider_terminate_after": None,
          "workload_deadline_utc": None}})

def fixture(root, healthy=True):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    state_dir = root / "reaper"
    lease_dir = state_dir / "leases"
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    job = job_fixture(); job_path = root / "job.json"
    job_path.write_text(json.dumps(job, sort_keys=True) + "\n")
    ledger_path = state_dir / "campaign.json"
    ledger = CampaignLedger.create(
        str(ledger_path), "10", "0", ".01", 2,
        provider="runpod", provider_account_id="fixture-runpod-account")
    snap = ledger.snapshot()
    check("provider snapshot", ledger.record_provider_snapshot(
      snap["generation"], provider="runpod",
      provider_account_id="fixture-runpod-account",
      balance_available_usd="10", balance_observed_at="2026-09-01T00:00:00Z",
      balance_valid_until="2026-09-03T00:00:00Z", balance_source="fixture",
      provider_resources=[],
      inventory_observed_at="2026-09-01T00:00:00Z",
      inventory_valid_until="2026-09-03T00:00:00Z", inventory_complete=True,
      inventory_source="fixture").applied)
    manifest_refresh = lambda: {
      "bundle_contract_sha256": job["bundle_contract_sha256"],
      "control_manifest_sha256": job["control_plane"]["manifest_sha256"]}
    args = SimpleNamespace(runpod_drill_job_json=str(job_path),
      runpod_drill_manifest_refresh=manifest_refresh,
      runpod_drill_bundle_manifest_sha256=job["bundle_contract_sha256"],
      runpod_drill_control_manifest_sha256=job["control_plane"]["manifest_sha256"],
      reaper_state_dir=str(state_dir), lease_dir=str(lease_dir),
      campaign_ledger=str(ledger_path), campaign_ceiling="10",
      campaign_reserve="0", campaign_reaper_margin=".01", campaign_width=1,
      out=str(root / "proof"), max_cost="1",
      runpod_container_running_tariff=".10", runpod_container_stopped_tariff="0",
      runpod_pod_running_tariff=".10", runpod_pod_stopped_tariff=".20",
      runpod_network_tariff=".07", tariff_effective_at="2026-08-31T00:00:00Z",
      runpod_drill_workload_seconds=300, runpod_drill_terminate_seconds=420,
      runpod_drill_poll_seconds=15, runpod_drill_billing_wait_seconds=3600,
      dry_run=True, yes=False)
    clock = Clock(); provider = Provider(clock, root / "remote")
    timer = AutonomousTimer(clock, state_dir, lease_dir, healthy=healthy)
    checkout_status = lambda mode: {
        "revision": "a" * 40, "untracked_files": mode,
        "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
        "status_bytes": 0, "clean": True,
    }
    def host_key_verifier(_provider, provider_id, stage):
        provider_log_line = (
            "256 SHA256:%s fixture (ED25519)" % ("A" * 43))
        observed_at = RD._utc(provider.clock.time())
        proof = RD._seal({
            "schema": "fidelity-suite/runpod-ssh-host-key-proof.v2",
            "proof_sha256": "",
            "provider": "runpod",
            "provider_id": provider_id,
            "verified_at_utc": observed_at,
            "verification_source": "runpod-authenticated-v2-container-log",
            "provider_log_endpoint_origin": "https://api.runpod.io",
            "provider_log_source": "container",
            "provider_log_tail": 5000,
            "provider_log_observed_at_utc": observed_at,
            "provider_log_line": provider_log_line,
            "provider_log_line_sha256": hashlib.sha256(
                provider_log_line.encode("utf-8")).hexdigest(),
            "provider_log_fingerprint": "SHA256:" + "A" * 43,
            "algorithm": "ssh-ed25519",
            "fingerprint": "SHA256:" + "A" * 43,
            "host": "fixture.runpod.test",
            "port": 22,
            "known_hosts_sha256": "e" * 64,
        }, "proof_sha256")
        RD._atomic_json(
            Path(stage) / "artifacts" / "runpod-ssh-host-key-proof.json",
            proof)
        return proof
    seams = RD.DrillSeams(
      clock=clock, attempt_id_factory=lambda: "d" * 24,
      reaper_health_check=timer.health, autonomous_timer_tick=timer.tick,
      supervisor=InlineSupervisor(), checkout_status=checkout_status,
      host_key_verifier=host_key_verifier)
    return args, provider, seams, ledger

def tree(root):
    return [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(Path(root).rglob("*")) if p.is_file()]

def real_supervisor(root):
    ready = root / "ready.json"
    def child(startup):
        check("child received parent startup", startup["released"] is True)
        RD._atomic_json(ready, RD._seal({
            "schema": RD.CONTROLLER_STATE_SCHEMA, "state_sha256": "",
            "status": "ready", "controller_pid": os.getpid()}, "state_sha256"))
        while True: time.sleep(1)
    event = RD.ForkSupervisor().supervise(
        child, ready, time.time() + 5, RD.RealClock(),
        lambda pid: {"released": pid > 0})
    check("real SIGKILL observed", event["signal"] == "SIGKILL"
          and event["controller_pid"] != os.getpid())


def parent_signal_cleanup(root):
    started = root / "started"
    child_pid = []
    previous = signal.getsignal(signal.SIGTERM)

    class InterruptClock:
        def time(self): return time.time()
        def sleep(self, _seconds):
            deadline = time.time() + 5
            while not started.exists() and time.time() < deadline:
                time.sleep(.01)
            check("signal fixture child started", started.exists())
            os.kill(os.getpid(), signal.SIGTERM)

    def child(_startup):
        started.write_text(str(os.getpid()))
        while True: signal.pause()

    def prepare(pid):
        child_pid.append(pid)
        return {"released": True}

    check("controlled parent signal propagated", refuses(
        lambda: RD.ForkSupervisor().supervise(
            child, root / "never-ready.json", time.time() + 10,
            InterruptClock(), prepare),
        RD._ParentSignal))
    check("parent signal handler restored",
          signal.getsignal(signal.SIGTERM) == previous)
    check("signal child was reaped", len(child_pid) == 1 and refuses(
        lambda: os.waitpid(child_pid[0], os.WNOHANG), ChildProcessError))

def _rewrite_deadline_proof(proof_path, mutate):
    proof_path = Path(proof_path)
    root = proof_path.parent
    proof = json.loads(proof_path.read_text())
    deadline_path = root / proof["artifacts"][
        "provider_deadline_observations"]["path"]
    loss_path = root / proof["artifacts"]["controller_loss"]["path"]
    deadline = json.loads(deadline_path.read_text())
    mutate(deadline)
    deadline = RD._seal(deadline, "record_sha256")
    RD._atomic_json(deadline_path, deadline)
    deadline_raw = deadline_path.read_bytes()
    proof["artifacts"]["provider_deadline_observations"].update({
        "bytes": len(deadline_raw),
        "sha256": hashlib.sha256(deadline_raw).hexdigest()})
    loss = json.loads(loss_path.read_text())
    loss["provider_deadline_observations_sha256"] = deadline["record_sha256"]
    loss = RD._seal(loss, "receipt_sha256")
    RD._atomic_json(loss_path, loss)
    loss_raw = loss_path.read_bytes()
    proof["artifacts"]["controller_loss"].update({
        "bytes": len(loss_raw), "sha256": hashlib.sha256(loss_raw).hexdigest()})
    RD._atomic_json(proof_path, RD._seal(proof, "proof_sha256"))


def _deadline_mutation_refused(proof_path, plan, mutate):
    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text())
    paths = [
        proof_path,
        proof_path.parent / proof["artifacts"][
            "provider_deadline_observations"]["path"],
        proof_path.parent / proof["artifacts"]["controller_loss"]["path"],
    ]
    originals = [path.read_bytes() for path in paths]
    try:
        _rewrite_deadline_proof(proof_path, mutate)
        return refuses(lambda: validate_safety_proof(
            proof_path, plan.bundle_contract_sha256,
            plan.control_manifest_sha256, plan.provider_account_id,
            plan.campaign_ledger), SafetyProofError)
    finally:
        for path, body in zip(paths, originals):
            path.write_bytes(body)


def _destroy_health_mutation_refused(proof_path, plan, mutate):
    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text())
    root = proof_path.parent
    health_path = root / proof["artifacts"]["reaper_destroy_health"]["path"]
    loss_path = root / proof["artifacts"]["controller_loss"]["path"]
    paths = [proof_path, health_path, loss_path]
    originals = [path.read_bytes() for path in paths]
    try:
        health = json.loads(health_path.read_text())
        mutate(health)
        health = RD._seal(health, "record_sha256")
        RD._atomic_json(health_path, health)
        health_raw = health_path.read_bytes()
        proof["artifacts"]["reaper_destroy_health"].update({
            "bytes": len(health_raw),
            "sha256": hashlib.sha256(health_raw).hexdigest(),
        })
        loss = json.loads(loss_path.read_text())
        loss["reaper_destroy_health_sha256"] = health["record_sha256"]
        loss = RD._seal(loss, "receipt_sha256")
        RD._atomic_json(loss_path, loss)
        loss_raw = loss_path.read_bytes()
        proof["artifacts"]["controller_loss"].update({
            "bytes": len(loss_raw),
            "sha256": hashlib.sha256(loss_raw).hexdigest(),
        })
        RD._atomic_json(proof_path, RD._seal(proof, "proof_sha256"))
        return refuses(lambda: validate_safety_proof(
            proof_path, plan.bundle_contract_sha256,
            plan.control_manifest_sha256, plan.provider_account_id,
            plan.campaign_ledger), SafetyProofError)
    finally:
        for target, body in zip(paths, originals):
            target.write_bytes(body)


def _duplicate_receipt_replacement(raw):
    body = b'{"x":1,"x":2}'
    check("duplicate replacement fits observed receipt", len(body) <= len(raw))
    return body + b" " * (len(raw) - len(body))


def main():
    response = {"cost_per_hr": "0.44"}
    binding = {"observed": {"cost_per_hr": "0.44"}}
    strict = {"cost_per_hr": "0.44"}
    check("equal three-source created rate accepted",
          RD._require_exact_created_rate(response, binding, strict)
          == RD.Decimal("0.44"))
    for label, create_rate, graphql_rate, rest_rate in (
            ("create/GraphQL rate mismatch", "0.45", "0.44", "0.44"),
            ("GraphQL/REST rate mismatch", "0.44", "0.45", "0.44"),
            ("REST/create rate mismatch", "0.44", "0.44", "0.45")):
        check(label, refuses(
            lambda create_rate=create_rate, graphql_rate=graphql_rate,
                   rest_rate=rest_rate: RD._require_exact_created_rate(
                       {"cost_per_hr": create_rate},
                       {"observed": {"cost_per_hr": graphql_rate}},
                       {"cost_per_hr": rest_rate}),
            RD.DrillError))
    with tempfile.TemporaryDirectory() as td:
        provider_log_line = (
            "256 SHA256:%s fixture (ED25519)" % ("A" * 43))
        class ProviderLogHost:
            def set_known_hosts(self, path):
                self.known_hosts = Path(path)

            def ssh_host_ed25519_fingerprint(self, provider_id):
                check("drill requests logs for the exact provider id",
                      provider_id == "pod-log")
                return {
                    "endpoint_origin": "https://api.runpod.io",
                    "source": "container",
                    "tail": 5000,
                    "observed_at_utc": RD._utc(time.time()),
                    "line": provider_log_line,
                    "line_sha256": hashlib.sha256(
                        provider_log_line.encode("utf-8")).hexdigest(),
                    "fingerprint": "SHA256:" + "A" * 43,
                }

            def verify_host_key(self, provider_id, expected):
                check("drill compares provider logs to exact network endpoint",
                      provider_id == "pod-log"
                      and expected == "SHA256:" + "A" * 43)
                self.known_hosts.write_text(
                    "fixture ssh-ed25519 AAAA\n", encoding="utf-8")
                self.known_hosts.chmod(0o600)
                return {
                    "algorithm": "ssh-ed25519",
                    "fingerprint": expected,
                    "host": "fixture.runpod.test",
                    "port": 22,
                    "known_hosts_sha256": "e" * 64,
                }

        host_stage = Path(td)
        host_proof = RD._provider_log_host_key_verifier(
            ProviderLogHost(), "pod-log", host_stage)
        check("default drill host authentication is noninteractive provider-log "
              "verification",
              host_proof["schema"]
                  == "fidelity-suite/runpod-ssh-host-key-proof.v2"
              and host_proof["verification_source"]
                  == "runpod-authenticated-v2-container-log"
              and (host_stage / "artifacts"
                   / "runpod-ssh-host-key-proof.json").is_file())
    with tempfile.TemporaryDirectory() as td:
        helper_root = Path(td)
        helpers = RD._snapshot_remote_helpers()
        for name, body, digest in helpers:
            check("remote helper snapshot digest",
                  hashlib.sha256(body).hexdigest() == digest)
            target = helper_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        isolated = subprocess.run(
            [sys.executable, "-I", "-S", "-c",
             "import sys; sys.path.insert(0, %r); "
             "from fidelity import resultsink; "
             "assert callable(resultsink.write_archive)"
             % str(helper_root / "lib")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        check("remote helper closure imports in isolation",
              isolated.returncode == 0)

    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        before = tree(td); plan = RD.plan_drill(args, provider, seams=seams)
        check("dry-run zero mutations", before == tree(td)
              and provider.create_calls == provider.destroy_calls == 0)
        rows = RD._campaign_inventory_rows(
            [{"id": "pod-x", "name": "foreign", "status": "RUNNING"}],
            [{"id": "vol-x", "name": "foreign-volume", "status": "READY"}])
        classification = ledger.classify_provider_resources(rows)
        check("bootstrap classifies every preexisting family as unknown",
              classification["known_pod_ids"] == []
              and classification["unknown_resources"] == [
                  {"family": "network_volumes", "id": "vol-x"},
                  {"family": "pods", "id": "pod-x"}])
        for label, attr, value in (("api key", "key", False), ("ssh key", "ssh", False),
             ("balance", "balance_value", None), ("inventory", "complete", False),
             ("inventory resource", "inventory_present", True)):
            a, p, s, _ = fixture(Path(td) / label.replace(" ", "-")); setattr(p, attr, value)
            check("missing %s refusal" % label,
                  refuses(lambda a=a, p=p, s=s: RD.plan_drill(a, p, seams=s)))
        a, p, s, _ = fixture(Path(td) / "reaper", healthy=False)
        check("missing reaper refusal", refuses(lambda: RD.plan_drill(a, p, seams=s)))
        args.dry_run = False
        before_ledger = ledger.snapshot()
        check("--yes required", refuses(lambda: RD.execute_drill(
            plan, args, provider, seams=seams), RD.DrillError))
        check("authorization pre-mutation", before_ledger == ledger.snapshot()
              and provider.create_calls == 0)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        ordinary_host_verifier = seams.host_key_verifier

        def delayed_host_verifier(*call_args):
            proof = ordinary_host_verifier(*call_args)
            seams.clock.sleep(args.runpod_drill_workload_seconds + 1)
            return proof

        seams.host_key_verifier = delayed_host_verifier
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        try:
            RD.execute_drill(plan, args, provider, seams=seams)
        except RD.DrillError as exc:
            deadline_refused = (
                str(exc)
                == "authenticated host-key retrieval exhausted the drill "
                   "workload deadline")
        else:
            deadline_refused = False
        lease_path = Path(args.lease_dir) / (
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        failed = LeaseStore(Path(args.lease_dir)).read(lease_path)
        check("host authentication cannot delay controller loss past deadline",
              deadline_refused and failed["state"] == "DESTROYING"
              and not Path(args.out).exists())
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        provider.status_latency = 1
        before = ledger.snapshot()
        plan = RD.plan_drill(args, provider, seams=seams)
        check("provider observation after plan start remains admissible",
              plan.planned_at == "2026-09-01T00:00:00Z"
              and ledger.snapshot() == before
              and provider.create_calls == 0)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        provider.remote_archive_failure = True
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        check("pre-loss controller failure refuses drill", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RuntimeError))
        lease_path = Path(args.lease_dir) / (
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        failed = LeaseStore(Path(args.lease_dir)).read(lease_path)
        check("pre-loss controller failure grants immediate cleanup",
              failed["state"] == "DESTROYING"
              and failed["provider_resource_ids"] == ["pod-1"])
        timer = seams.reaper_health_check.__self__
        timer.tick(plan=plan, store=seams.lease_store_factory(
            Path(args.lease_dir)), provider=provider, now=seams.clock.time())
        seams.clock.sleep(300)
        timer.tick(plan=plan, store=seams.lease_store_factory(
            Path(args.lease_dir)), provider=provider, now=seams.clock.time())
        terminal = LeaseStore(Path(args.lease_dir)).read(lease_path)
        closed = ledger.snapshot()["attempts"][plan.attempt_key]
        check("failed drill is fully reconciled before retry",
              terminal["state"] == RD.TERMINAL
              and closed["phase"] == "RECONCILED"
              and closed["released"] is True
              and provider.list_instances() == [])
        provider.remote_archive_failure = False
        seams.attempt_id_factory = lambda: "e" * 24
        args.dry_run = True; args.yes = False
        retry = RD.plan_drill(args, provider, seams=seams)
        check("fully reconciled failed drill remains retryable",
              retry.attempt_id == "e" * 24
              and retry.ledger_generation == ledger.snapshot()["generation"])
        args.dry_run = False; args.yes = True
        retry_proof = RD.execute_drill(retry, args, provider, seams=seams)
        check("reconciled retry reaches accepted paid drill",
              retry_proof.is_file() and provider.create_calls == 2)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        empty_sha = hashlib.sha256(b"").hexdigest()
        seams.checkout_status = lambda mode: {
            "revision": "a" * 40, "untracked_files": mode,
            "status_porcelain_sha256": (
                "b" * 64 if mode == "all" else empty_sha),
            "status_bytes": (1 if mode == "all" else 0),
            "clean": mode != "all",
        }
        before = ledger.snapshot()
        dirty_plan = RD.plan_drill(args, provider, seams=seams)
        check("dirty dry plan reports without mutation",
              dirty_plan.public_dict()["producer_checkout"]["clean"] is False
              and ledger.snapshot() == before)
        args.dry_run = False; args.yes = True
        check("dirty paid producer refused before ledger mutation",
              refuses(lambda: RD.execute_drill(
                  dirty_plan, args, provider, seams=seams), RD.DrillError)
              and ledger.snapshot() == before
              and provider.create_calls == 0)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        empty_sha = hashlib.sha256(b"").hexdigest()
        clean_all = seams.checkout_status
        plan = RD.plan_drill(args, provider, seams=seams)
        checkout_calls = [0]
        def pre_post_drift(mode):
            checkout_calls[0] += 1
            if checkout_calls[0] == 1:
                return clean_all(mode)
            return {
                "revision": "a" * 40, "untracked_files": "all",
                "status_porcelain_sha256": "c" * 64,
                "status_bytes": 1, "clean": False,
            }
        seams.checkout_status = pre_post_drift
        args.dry_run = False; args.yes = True
        check("tracked pre-POST drift refuses sole mutation", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError) and provider.create_calls == 0)
        check("pre-POST checkout refusal creates no campaign liability",
              plan.attempt_key not in ledger.snapshot()["attempts"]
              and not (Path(args.lease_dir) / (
                  "%s.%s.json" % (plan.job_hash, plan.attempt_id))).exists())

    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        Path(args.campaign_ledger).unlink()
        lock = Path(args.campaign_ledger + ".lock")
        if lock.exists():
            lock.unlink()
        before = tree(td)
        absent_plan = RD.plan_drill(args, provider, seams=seams)
        check("absent ledger dry preview is non-mutating",
              absent_plan.ledger_exists is False
              and before == tree(td)
              and not Path(args.campaign_ledger).exists())
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        unresolved_store = LeaseStore(
            Path(args.lease_dir), clock=seams.clock.time)
        unresolved_store.begin_create(
            job_hash="f" * 64, provider="runpod",
            request={"gpu": "fixture"},
            pre_create_resources=[],
            create_deadline_epoch=seams.clock.time() + 60,
            workload_deadline_epoch=seams.clock.time() + 3600)
        timer = seams.reaper_health_check.__self__
        timer.stamp["unresolved_count"] = 1
        timer._write()
        check("bootstrap refuses health-bound unresolved lease",
              refuses(
                  lambda: RD.plan_drill(args, provider, seams=seams),
                  RD.DrillError)
              and provider.create_calls == 0)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        plan = RD.plan_drill(args, provider, seams=seams)
        class InterruptsAfterReservation:
            controller_holds = True
            def supervise(self, controller, ready, deadline, clock, prepare):
                startup = prepare(os.getpid())
                lease_path = Path(args.lease_dir) / startup["lease_name"]
                prepared = LeaseStore(Path(args.lease_dir)).read(lease_path)
                check("PREPARED lease precedes child release",
                      prepared["state"] == "PREPARED"
                      and startup["ledger_generation"] > 0)
                raise KeyboardInterrupt("fixture parent signal")
        seams.supervisor = InterruptsAfterReservation()
        args.dry_run = False; args.yes = True
        check("parent signal propagated after child quiescence", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            KeyboardInterrupt))
        reserved = ledger.snapshot()["attempts"][plan.attempt_key]
        lease_path = Path(args.lease_dir,
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        prepared = LeaseStore(Path(args.lease_dir)).read(lease_path)
        check("signal cannot orphan reservation before visible lease",
              reserved["phase"] == "RESERVED"
              and prepared["state"] == "PREPARED"
              and provider.create_calls == 0)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        plan = RD.plan_drill(args, provider, seams=seams)
        provider.create_response_wrong_name = True
        args.dry_run = False; args.yes = True
        check("unqualified response refuses scientific drill", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError))
        lease_path = Path(args.lease_dir) / (
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        refused_lease = LeaseStore(Path(args.lease_dir)).read(lease_path)
        refused_attempt = ledger.snapshot()["attempts"][plan.attempt_key]
        check("unqualified response id is durable before propagation",
              refused_lease["state"] == "DESTROYING"
              and refused_lease["provider_resource_ids"] == ["pod-1"]
              and any(row["event"] == "CREATE_RESPONSE_BOUND"
                      for row in refused_lease["history"])
              and refused_attempt["provider_ids"] == ["pod-1"]
              and refused_attempt["phase"] == "TERMINATE_REQUIRED")

    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        plan = RD.plan_drill(args, provider, seams=seams)

        class FailingResponseStore(LeaseStore):
            def record_create_success(self, ref, response):
                raise OSError("fixture durable write failure")

        seams.lease_store_factory = lambda path: FailingResponseStore(
            Path(path), clock=seams.clock.time)
        args.dry_run = False; args.yes = True
        check("create response persistence failure refuses drill", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError))
        failed_lease_path = Path(args.lease_dir) / (
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        failed_lease = LeaseStore(Path(args.lease_dir)).read(
            failed_lease_path)
        failed_attempt = ledger.snapshot()["attempts"][plan.attempt_key]
        check("persistence failure leaves exact campaign cleanup authority",
              failed_lease["state"] == "CREATING"
              and failed_lease["provider_resource_ids"] == []
              and failed_attempt["provider_ids"] == ["pod-1"]
              and failed_attempt["phase"] == "TERMINATE_REQUIRED"
              and failed_attempt["cleanup_binding_evidence"] is not None)

    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        plan = RD.plan_drill(args, provider, seams=seams)
        provider.post_create_wrong_pod = True
        provider.post_create_network_volume = True
        args.dry_run = False; args.yes = True
        check("post-create foreign resource delta freezes", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError))
        lease_path = Path(args.lease_dir) / (
            "%s.%s.json" % (plan.job_hash, plan.attempt_id))
        ambiguous = LeaseStore(Path(args.lease_dir)).read(lease_path)
        anomaly = (ambiguous.get("terminal_proof") or {}).get(
            "ambiguous_create") or {}
        check("foreign pod and network volume are blockers, never targets",
              ambiguous["state"] == "AMBIGUOUS"
              and ambiguous["provider_resource_ids"] == ["pod-1"]
              and anomaly.get("wrong_name_new_pod_ids") == ["foreign-pod"]
              and anomaly.get("new_network_volume_ids") == ["foreign-volume"]
              and provider.destroy_calls == 0
              and (ledger.snapshot()["attempts"][plan.attempt_key]
                   ["provider_ids"]) == ["pod-1"])
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        provider.post_create_visibility_delay = 80
        provider.termination_offset = 901
        plan = RD.plan_drill(args, provider, seams=seams)
        provider.status_latency = 31
        provider.lifecycle_poll_latency = 28
        args.dry_run = False; args.yes = True
        proof_path = RD.execute_drill(plan, args, provider, seams=seams)
        proof = json.loads(proof_path.read_text())
        accepted = validate_safety_proof(
          proof_path, plan.bundle_contract_sha256,
          plan.control_manifest_sha256, plan.provider_account_id,
          plan.campaign_ledger,
          now=datetime.fromtimestamp(seams.clock.time(), tz=timezone.utc))
        lease = accepted["lease"]
        history = lease["history"]
        check("one POST", provider.create_calls == 1)
        check("post-create inventory converges across provider lag",
              provider.post_create_inventory_calls >= 3)
        check("autonomous reaper destroy", provider.destroy_calls == 1)
        destroy_events = [
            row for row in history if row["event"] == "DESTROY_REQUESTED"]
        check("one deadline DESTROY_REQUESTED",
              len(destroy_events) == 1
              and destroy_events[0]["evidence"]["reason"]
                  == "absolute reap deadline expired")
        bound = next(row for row in history if row["event"] == "CREATE_RESPONSE_BOUND")
        response = bound["evidence"]["response"]
        check("client deadline echo rejected as evidence",
              "terminate_after" not in response and "requested_terminate_after" not in response
              and bound["evidence"]["submitted_request_sha256"]
                  == lease["create"]["request_sha256"])
        absent = next(row for row in history
                      if row["event"] == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING")
        check("deadline before exact absence within lag",
          RD._utc_epoch(plan.terminate_after, "deadline")
          <= RD._utc_epoch(absent["at"], "absence")
          <= RD._utc_epoch(plan.observation_until, "bound"))
        def artifact(name):
            return json.loads((proof_path.parent / proof["artifacts"][name]["path"]).read_text())
        kill, loss = artifact("controller_kill_event"), artifact("controller_loss")
        check("real-shaped supervisor observation", kill["signal"] == "SIGKILL"
          and loss["controller_exit_observed"] is True
          and loss["kill_event_sha256"] == kill["receipt_sha256"])
        destroy_health = artifact("reaper_destroy_health")
        check("loss binds autonomous destroy health",
              loss["reaper_destroy_health_sha256"]
                  == destroy_health["record_sha256"]
              and any(
                  action.get("action") == "destroy-requested"
                  and action.get("provider_id") == "pod-1"
                  for action in destroy_health["actions"]))
        deadline = artifact("provider_deadline_observations")
        first_absence = next(
            row for row in deadline["observations"]
            if row["exact_present"] is False)
        before_deadline = [
            row for row in deadline["observations"]
            if row["deadline_relation"] == "BEFORE"]
        durations = [
            row["poll_completed_at_epoch"] - row["poll_started_at_epoch"]
            for row in deadline["observations"]]
        check("slow status/list polls remain inside explicit authored maximum",
              first_absence["deadline_relation"] == "AFTER"
              and max(durations) > deadline["poll_interval_seconds"]
              and max(durations) <= deadline["poll_duration_max_seconds"]
              and before_deadline
              and all(row["complete"] is True
                      and row["exact_present"] is True
                      for row in before_deadline)
              and loss["provider_deadline_observations_sha256"]
                  == deadline["record_sha256"])
        check("provider timer is explicitly untrusted",
              proof["drill"]["termination_mechanism"]
                  == "autonomous-systemd-user-reaper"
              and proof["drill"]["provider_timer_trusted"] is False
              and plan.public_dict()["provider_timer_trusted"] is False)
        def forge_destroy_target(document):
            action = next(
                row for row in document["actions"]
                if row.get("action") == "destroy-requested")
            action["provider_id"] = "foreign-pod"
        check("standalone validator rejects forged autonomous destroy target",
              _destroy_health_mutation_refused(
                  proof_path, plan, forge_destroy_target))
        check("poll duration beyond explicit maximum refuses", refuses(
            lambda: RD._persist_deadline_observation(
                Path(td) / "overlong-observation.json", plan, "pod-1",
                [{"id": "pod-1", "name": plan.exact_name,
                  "status": "RUNNING"}],
                0.0, RD.DEADLINE_POLL_DURATION_MAX_SECONDS + 1.0, []),
            RD.DrillError))
        check("exact billing arithmetic", artifact("billing_arithmetic")["total_amount"] == ".08"
              or artifact("billing_arithmetic")["total_amount"] == "0.08")
        campaign = artifact("campaign_release")
        check("ledger released", campaign["released"] is True
          and campaign["maximum_remaining_liability_usd"] == "0"
          and ledger.snapshot()["attempts"][plan.attempt_key]["released"] is True)
        result = verify_archive(proof_path.parent / proof["artifacts"]["result_archive"]["path"])
        check("archive/job/current manifests bound", result["manifest"]["job_id_full"] == plan.job_hash
          and proof["bundle_manifest_sha256"] == plan.bundle_contract_sha256
          and proof["control_manifest_sha256"] == plan.control_manifest_sha256
          and proof["drill"]["create_request_sha256"] == lease["create"]["request_sha256"])
        check("bounded streaming transfer only", provider.download_calls == []
          and [row["name"] for row in provider.bounded_download_calls]
              == ["result-transfer.json", "result-bundle.tar.gz"]
          and provider.bounded_download_calls[0]["expected_bytes"]
              == proof["artifacts"]["result_transfer"]["bytes"]
          and provider.bounded_download_calls[0]["max_bytes"]
              == RD.TRANSFER_RECEIPT_MAX_BYTES
          and provider.bounded_download_calls[1]["max_bytes"]
              == RD.DRILL_ARCHIVE_MAX_BYTES
          and provider.bounded_download_calls[1]["expected_bytes"]
              == proof["artifacts"]["result_archive"]["bytes"]
          and proof["artifacts"]["result_archive"]["bytes"]
              <= RD.DRILL_ARCHIVE_MAX_BYTES)
        unsafe = json.loads(json.dumps(proof)); unsafe["artifacts"]["lease"]["path"] = "../x"
        unsafe = RD._seal(unsafe, "proof_sha256"); unsafe_path = proof_path.parent / "unsafe.json"
        unsafe_path.write_text(json.dumps(unsafe) + "\n")
        check("unsafe artifact refused", refuses(lambda: validate_safety_proof(
          unsafe_path, plan.bundle_contract_sha256,
          plan.control_manifest_sha256, plan.provider_account_id,
          plan.campaign_ledger), SafetyProofError))
        check("secure exact request", lease["create"]["request"]["secure_cloud"] is True
          and lease["create"]["request"]["offer"] == "on-demand"
          and lease["create"]["request"]["network_volume_id"] is None)
        def break_sequence(document):
            document["observations"][1]["sequence"] += 1
        check("standalone validator rejects nonsequential observation chain",
              _deadline_mutation_refused(proof_path, plan, break_sequence))
        def forge_early_absence(document):
            row = document["observations"][0]
            row["resources"] = []
            row["provider_ids"] = []
            row["listing_sha256"] = RD._sha256([])
            row["exact_present"] = False
        check("standalone validator rejects forged predeadline absence",
              _deadline_mutation_refused(proof_path, plan, forge_early_absence))
        def forge_overlong_absence(document):
            rows = document["observations"]
            absent_rows = [row for row in rows if row["exact_present"] is False]
            target = RD._utc_epoch(
                document["provider_deadline_observation_until"], "bound") + 1
            delta = target - absent_rows[0]["poll_completed_at_epoch"]
            for row in absent_rows:
                row["poll_started_at_epoch"] += delta
                row["poll_completed_at_epoch"] += delta
                row["poll_completed_at_utc"] = RD._utc(
                    row["poll_completed_at_epoch"])
                row["deadline_relation"] = "AFTER"
        check("standalone validator rejects absence beyond authored lag",
              _deadline_mutation_refused(
                  proof_path, plan, forge_overlong_absence))
        live_ledger_path = Path(plan.campaign_ledger)
        live_ledger_raw = live_ledger_path.read_bytes()
        try:
            live_ledger_path.write_bytes(live_ledger_raw + b"\n")
            check("later nonsemantic durable ledger bytes remain admissible",
                  validate_safety_proof(
                      proof_path, plan.bundle_contract_sha256,
                      plan.control_manifest_sha256, plan.provider_account_id,
                      plan.campaign_ledger)["proof"]["proof_sha256"]
                  == proof["proof_sha256"])
        finally:
            live_ledger_path.write_bytes(live_ledger_raw)
        copied_dir = live_ledger_path.parent / "different-campaign"
        copied_dir.mkdir()
        copied_ledger = copied_dir / live_ledger_path.name
        copied_ledger.write_bytes(live_ledger_raw)
        extended_ledger = json.loads(live_ledger_raw)
        source_attempt = extended_ledger["attempts"][plan.attempt_key]
        later_attempt = json.loads(json.dumps(source_attempt))
        later_job = "f" * 64
        later_attempt_id = "e" * 24
        later_key = "%s:%s" % (later_job, later_attempt_id)
        later_attempt.update({
            "job_hash": later_job,
            "attempt": later_attempt_id,
            "reservation_kind": "measurement",
            "phase": "CANCELLED_BEFORE_CREATE",
            "provider_ids": [],
            "cleanup_binding_evidence": None,
            "precreate_cancellation": {
                "cancelled_at": RD._utc(seams.clock.time()),
                "campaign_phase_before_cancel": "RESERVED",
                "lease_state": "LEASE_ABSENT",
                "no_create_evidence": "fixture durable no-POST evidence"},
            "actual_quote": None,
            "maximum_remaining_liability_usd": "0",
            "deletion": None,
            "billing": None,
            "released": True,
            "admission_freeze_reason": None,
        })
        extended_ledger["attempts"][later_key] = later_attempt
        extended_ledger["generation"] += 1
        CampaignLedger._validate_document(extended_ledger)
        try:
            live_ledger_path.write_text(
                json.dumps(extended_ledger, sort_keys=True) + "\n")
            check("later campaign attempts do not stale immutable drill proof",
                  validate_safety_proof(
                      proof_path, plan.bundle_contract_sha256,
                      plan.control_manifest_sha256, plan.provider_account_id,
                      plan.campaign_ledger)["proof"]["proof_sha256"]
                  == proof["proof_sha256"])
        finally:
            live_ledger_path.write_bytes(live_ledger_raw)
        check("copied stale ledger cannot replace canonical live campaign",
              refuses(lambda: validate_safety_proof(
                  proof_path, plan.bundle_contract_sha256,
                  plan.control_manifest_sha256, plan.provider_account_id,
                  copied_ledger), SafetyProofError))
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        provider.termination_offset = 901
        provider.billing_delay_after_absence_seconds = 1000
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        proof_path = RD.execute_drill(plan, args, provider, seams=seams)
        accepted = validate_safety_proof(
            proof_path, plan.bundle_contract_sha256,
            plan.control_manifest_sha256, plan.provider_account_id,
            plan.campaign_ledger,
            now=datetime.fromtimestamp(seams.clock.time(), tz=timezone.utc))
        billed_at = RD._utc_epoch(
            accepted["lease"]["billing_reconciliation"][
                "billing_histories"][0]["retrieved_at_utc"],
            "billing retrieval")
        check("billing may stabilize after the lifecycle proof bound",
              billed_at
              > RD._utc_epoch(plan.terminate_after, "deadline")
                  + RD.DRILL_LAG_SECONDS)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        provider.termination_offset = -60
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        check("early exact-pod disappearance refuses immediately", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError)
            and seams.clock.time() < provider.termination
            and not Path(args.out).exists())
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, _ledger = fixture(td)
        provider.termination_offset = 901
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        seams.autonomous_timer_tick = lambda **_kwargs: None
        check("missing autonomous deadline sweep refuses", refuses(
            lambda: RD.execute_drill(plan, args, provider, seams=seams),
            RD.DrillError) and not Path(args.out).exists()
            and provider.pod is None)
    with tempfile.TemporaryDirectory() as td:
        args, provider, seams, ledger = fixture(td)
        plan = RD.plan_drill(args, provider, seams=seams)
        args.dry_run = False; args.yes = True
        original_listing = provider.list_instances
        def misses_provider_deadline():
            if provider.pod is None:
                return []
            return [dict(provider.pod)]
        provider.list_instances = misses_provider_deadline
        ignored_timer_proof = RD.execute_drill(
            plan, args, provider, seams=seams)
        check("ignored provider timer still yields autonomous proof",
              ignored_timer_proof.is_file()
              and provider.create_calls == 1
              and provider.destroy_calls == 1)
        accepted = validate_safety_proof(
            ignored_timer_proof, plan.bundle_contract_sha256,
            plan.control_manifest_sha256, plan.provider_account_id,
            plan.campaign_ledger,
            now=datetime.fromtimestamp(
                seams.clock.time(), tz=timezone.utc))
        attempt = ledger.snapshot()["attempts"][plan.attempt_key]
        check("autonomous cleanup reconciles campaign",
              accepted["proof"]["drill"]["provider_timer_trusted"] is False
              and attempt["released"] is True
              and attempt["phase"] == "RECONCILED")
        provider.list_instances = original_listing
    transfer_faults = (
        ("short archive stream", lambda provider:
            provider.bounded_modes.__setitem__(
                "result-bundle.tar.gz", "short")),
        ("oversized archive stream", lambda provider:
            provider.bounded_modes.__setitem__(
                "result-bundle.tar.gz", "oversized")),
        ("adversarial archive replacement", lambda provider:
            provider.download_replacements.__setitem__(
                "result-bundle.tar.gz",
                lambda raw: raw[:-1] + bytes([raw[-1] ^ 1]))),
        ("stalled archive stream", lambda provider:
            provider.bounded_modes.__setitem__(
                "result-bundle.tar.gz", "stalled")),
        ("adversarial receipt replacement", lambda provider:
            provider.download_replacements.__setitem__(
                "result-transfer.json", _duplicate_receipt_replacement)),
        ("oversized transfer receipt", lambda provider:
            setattr(provider, "receipt_padding_bytes",
                    RD.TRANSFER_RECEIPT_MAX_BYTES)),
    )
    for label, configure in transfer_faults:
        with tempfile.TemporaryDirectory() as td:
            args, provider, seams, _ledger = fixture(td)
            configure(provider)
            plan = RD.plan_drill(args, provider, seams=seams)
            args.dry_run = False
            args.yes = True
            check("%s refuses without publication" % label,
                  refuses(lambda: RD.execute_drill(
                      plan, args, provider, seams=seams), Exception)
                  and provider.download_calls == []
                  and not Path(args.out).exists())
    with tempfile.TemporaryDirectory() as td:
        real_supervisor(Path(td))
        parent_signal_cleanup(Path(td))
    print("PASS: RunPod controller-loss/autonomous-reaper drill")
    return 0

if __name__ == "__main__": raise SystemExit(main())
