#!/usr/bin/env python3
"""Paid RunPod controller-loss/autonomous-reaper drill producer.

Planning is read-only. Execution is an explicitly authorized, single-POST
campaign attempt supervised by a separate process. The provider's accepted
``terminateAfter`` value is retained as an untrusted extra hint; the independent
boot-persistent user-systemd reaper owns the absolute cleanup deadline.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .campaign import (CampaignLedger, CostQuote, MAX_QUOTE_VALIDITY_SECONDS,
                       RUNPOD_TARIFF_SOURCE,
                       _bootstrap_drill_blocked_by_prior_attempts)
from .common import redact
from .cloudlease import (CreateResponsePersistenceError, HEALTH_SCHEMA,
                         TERMINAL, LeaseStore,
                         campaign_cleanup_binding_evidence,
                         campaign_coordinates, systemd_reaper_health, utc_iso,
                         validate_unresolved_lease_scope)
from .runpodapi import (
    RunPodCreateResponseError, _billing_total_matches_record_sum)
from .jobcontract import (finalize_job, seal_execution_job,
                          validate_execution_job, verify_bundle_manifest,
                          verify_job)
from .resultsink import verify_archive
from .runpodsafety import (DRILL_KIND, LEASE_DRILL_KIND, PROOF_SCHEMA,
                           campaign_ledger_coordinate_sha256, canonical_bytes,
                           validate_safety_proof)

GPU_TYPE = "NVIDIA L4"
IMAGE = (
    "runpod/pytorch@sha256:"
    "ab2addc2916ffc72989288bd5048933c69ba6531f1d679c25afbd9eadc5a5fd5")
DRILL_LAG_SECONDS = 900
DEADLINE_POLL_DURATION_MAX_SECONDS = 120
DEADLINE_INTERPOLL_GAP_MAX_SECONDS = 120
DEFAULT_WORKLOAD_SECONDS = 1200
DEFAULT_TERMINATE_SECONDS = 1320
DEFAULT_POLL_SECONDS = 15
DEFAULT_STORAGE_GB = 20
MIN_VCPU = 4
MIN_RAM_GB = 16
DEFAULT_BILLING_WAIT_SECONDS = 3600
TRANSFER_SCHEMA = "fidelity-suite/runpod-drill-result-transfer.v1"
TRANSFER_RECEIPT_MAX_BYTES = 64 * 1024
DRILL_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024
REMOTE_ROOT = "/workspace/fidelity-drill"
KILL_EVENT_SCHEMA = "fidelity-suite/controller-kill-event.v1"
LOSS_SCHEMA = "fidelity-suite/controller-loss-supervisor.v1"
CAMPAIGN_RELEASE_SCHEMA = "fidelity-suite/runpod-drill-campaign-release.v1"
BILLING_SCHEMA = "fidelity-suite/runpod-drill-billing-arithmetic.v1"
CONTROLLER_STATE_SCHEMA = "fidelity-suite/runpod-drill-controller-state.v1"
DEADLINE_OBSERVATION_SCHEMA = (
    "fidelity-suite/runpod-provider-deadline-observations.v1")


class _ParentSignal(BaseException):
    """Controlled parent interruption after a supervised child has started."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__("supervisor parent received %s" %
                         signal.Signals(self.signum).name)




class DrillError(RuntimeError):
    """A prerequisite or a piece of paid drill evidence failed closed."""


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))


def _utc_epoch(text: str, label: str) -> int:
    if not isinstance(text, str):
        raise DrillError("%s is not exact UTC" % label)
    try:
        parsed = time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise DrillError("%s is not exact UTC" % label)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != text:
        raise DrillError("%s is not canonical exact UTC" % label)
    import calendar
    return calendar.timegm(parsed)


def _seal(document: Mapping[str, Any], field: str) -> Dict[str, Any]:
    result = dict(document)
    result[field] = ""
    result[field] = _sha256(result)
    return result

def _verify_seal(document: Mapping[str, Any], field: str) -> bool:
    claimed = document.get(field)
    unsealed = dict(document)
    unsealed[field] = ""
    return (isinstance(claimed, str)
            and secrets.compare_digest(claimed, _sha256(unsealed)))


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".%s." % path.name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        _fsync_dir(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(path: Path, document: Any) -> None:
    _atomic_bytes(path, _json_bytes(document))


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON object key %r" % key)
        document[key] = value
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number %s" % value)
    return parsed


def _read_json_regular(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            raise DrillError("%s must be a regular non-symlink file" % label)
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float)
    except DrillError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise DrillError("cannot read %s: %s" % (label, exc))
    if not isinstance(document, dict):
        raise DrillError("%s must contain a JSON object" % label)
    return document, raw


def _file_sha256_size(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise DrillError(
            "cannot hash artifact %s: %s" % (path, exc.__class__.__name__))
    return digest.hexdigest(), total


def _artifact(root: Path, path: Path) -> Dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in relative):
        raise DrillError("artifact path is not canonical relative POSIX")
    digest, size = _file_sha256_size(path)
    return {"path": relative, "bytes": size, "sha256": digest}


def _require_transition(result: Any, label: str) -> int:
    if not getattr(result, "applied", False):
        raise DrillError("%s refused: %s (%s)" % (
            label, getattr(result, "message", "unknown"),
            getattr(result, "code", "unknown")))
    return int(result.generation)


def _decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise DrillError("%s is not an exact decimal" % label)
    if not parsed.is_finite() or parsed < 0:
        raise DrillError("%s is invalid" % label)
    return parsed

def _producer_checkout_status(untracked_files: str) -> Dict[str, Any]:
    if untracked_files not in ("all", "no"):
        raise DrillError("unsupported checkout status mode")
    root = Path(__file__).resolve().parents[2]
    try:
        head_run = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False)
        status_run = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root),
             "status", "--porcelain",
             "--untracked-files=%s" % untracked_files],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {
            "revision": None, "untracked_files": untracked_files,
            "status_porcelain_sha256": None, "status_bytes": None,
            "clean": False,
        }
    head = head_run.stdout.decode("ascii", "replace").strip() \
        if head_run.returncode == 0 else ""
    status = status_run.stdout
    revision = head if re.fullmatch(r"[0-9a-f]{40}", head) else None
    return {
        "revision": revision, "untracked_files": untracked_files,
        "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "status_bytes": len(status),
        "clean": (
            revision is not None and status_run.returncode == 0
            and status == b""),
    }


class RealClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class DrillPlan:
    job: Dict[str, Any]
    job_hash: str
    attempt_id: str
    bundle_contract_sha256: str
    control_manifest_sha256: str
    provider_account_id: str
    manifest_refresh: Callable[[], Mapping[str, Any]]
    remote_helpers: Tuple[Tuple[str, bytes, str], ...]
    exact_name: str
    terminate_after: str
    observation_until: str
    create_deadline_epoch: float
    workload_deadline_epoch: float
    quote: CostQuote
    offer_rate: Decimal
    inventory: Dict[str, Any]
    pre_create_resources: Tuple[Any, ...]
    balance_available_usd: Decimal
    ledger_generation: int
    attempt_key: str
    lease_dir: Path
    reaper_state_dir: Path
    campaign_ledger: Path
    output: Path
    storage_gb: int
    container_disk_gb: int
    poll_seconds: int
    billing_wait_seconds: int
    campaign_ceiling: Decimal
    campaign_reserve: Decimal
    campaign_reaper_margin: Decimal
    ledger_exists: bool
    reaper_health: Dict[str, Any]
    checkout_initial: Dict[str, Any]
    planned_at: str

    def public_dict(self) -> Dict[str, Any]:
        """Non-secret, path-redacted plan suitable for CLI output."""
        return {
            "schema": "fidelity-suite/runpod-drill-plan.v1",
            "mode": DRILL_KIND,
            "provider": "runpod",
            "provider_account_id": self.provider_account_id,
            "job_id_full": self.job_hash,
            "attempt_id": self.attempt_id,
            "exact_name": self.exact_name,
            "gpu_type": GPU_TYPE,
            "secure_cloud": True,
            "spot": False,
            "offer": "on-demand",
            "gpu_count": 1,
            "network_volume_id": None,
            "storage_gb": self.storage_gb,
            "container_disk_gb": self.container_disk_gb,
            "reap_deadline": self.terminate_after,
            "provider_terminate_after_hint": self.terminate_after,
            "provider_timer_trusted": False,
            "reaper_observation_until": self.observation_until,
            "billing_wait_seconds": self.billing_wait_seconds,
            "live_rate_usd_per_hour": format(self.offer_rate, "f"),
            "maximum_liability_usd": format(self.quote.hard_cap_usd, "f"),
            "campaign_generation": self.ledger_generation,
            "campaign_attempt_key": self.attempt_key,
            "campaign_reservation_kind":
                "bootstrap-controller-loss-drill",
            "campaign_max_concurrent_attempts": 2,
            "campaign_authorized_concurrent_attempts": 1,
            "campaign_width_authorization": None,
            "reaper_healthy": self.reaper_health.get("ok") is True,
            "bundle_contract_sha256": self.bundle_contract_sha256,
            "campaign_ledger_action":
                ("validate-existing" if self.ledger_exists else "would-create"),
            "control_manifest_sha256": self.control_manifest_sha256,
            "producer_checkout": dict(self.checkout_initial),
            "planned_at": self.planned_at,
        }



def _provider_log_host_key_verifier(
        provider: Any, provider_id: str, stage: Path) -> Dict[str, Any]:
    known_hosts = stage / "ssh_known_hosts"
    provider.set_known_hosts(known_hosts)
    try:
        log_evidence = provider.ssh_host_ed25519_fingerprint(provider_id)
        evidence = provider.verify_host_key(
            provider_id, log_evidence["fingerprint"])
    except Exception as exc:
        raise DrillError(
            "RunPod SSH host-key authentication failed: %s" % exc) from exc
    proof = _seal({
        "schema": "fidelity-suite/runpod-ssh-host-key-proof.v2",
        "proof_sha256": "",
        "provider": "runpod",
        "provider_id": provider_id,
        "verified_at_utc": _utc(time.time()),
        "verification_source": "runpod-authenticated-v2-container-log",
        "provider_log_endpoint_origin": log_evidence["endpoint_origin"],
        "provider_log_source": log_evidence["source"],
        "provider_log_tail": log_evidence["tail"],
        "provider_log_observed_at_utc": log_evidence["observed_at_utc"],
        "provider_log_line_sha256": log_evidence["line_sha256"],
        "provider_log_line": log_evidence["line"],
        "provider_log_fingerprint": log_evidence["fingerprint"],
        "algorithm": evidence["algorithm"],
        "fingerprint": evidence["fingerprint"],
        "host": evidence["host"],
        "port": evidence["port"],
        "known_hosts_sha256": evidence["known_hosts_sha256"],
    }, "proof_sha256")
    _atomic_json(
        stage / "artifacts" / "runpod-ssh-host-key-proof.json", proof)
    return proof

@dataclass
class DrillSeams:
    clock: Any = None
    attempt_id_factory: Callable[[], str] = None
    reaper_health_check: Callable[..., Dict[str, Any]] = systemd_reaper_health
    autonomous_timer_tick: Optional[Callable[..., None]] = None
    supervisor: Any = None
    lease_store_factory: Callable[[Path], LeaseStore] = None
    checkout_status: Callable[[str], Dict[str, Any]] = None
    host_key_verifier: Callable[..., Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = RealClock()
        if self.attempt_id_factory is None:
            self.attempt_id_factory = lambda: secrets.token_hex(12)
        if self.supervisor is None:
            self.supervisor = ForkSupervisor()
        if self.lease_store_factory is None:
            self.lease_store_factory = lambda root: LeaseStore(
                root, clock=self.clock.time)
        if self.checkout_status is None:
            self.checkout_status = _producer_checkout_status

        if self.host_key_verifier is None:
            self.host_key_verifier = _provider_log_host_key_verifier

class ForkSupervisor:
    """Fork a controller, prepare its lease in the parent, then SIGKILL it."""

    controller_holds = True

    @staticmethod
    def _terminate_and_wait(pid: int) -> int:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        while True:
            try:
                waited, status = os.waitpid(pid, 0)
                if waited != pid:
                    raise DrillError("supervisor waited for the wrong child")
                return int(status)
            except InterruptedError:
                continue
            except ChildProcessError:
                return 0

    def supervise(self, controller: Callable[[Mapping[str, Any]], None],
                  ready_path: Path, deadline_epoch: float, clock: Any,
                  prepare: Callable[[int], Mapping[str, Any]]) -> Dict[str, Any]:
        startup_read, startup_write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(startup_write)
            try:
                chunks = []
                while True:
                    chunk = os.read(startup_read, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                os.close(startup_read)
                if not chunks:
                    raise DrillError(
                        "supervisor parent did not release controller startup")
                startup = json.loads(b"".join(chunks).decode("utf-8"))
                if not isinstance(startup, Mapping):
                    raise DrillError("controller startup payload is invalid")
                controller(startup)
            except BaseException as exc:  # child must leave a durable diagnosis
                try:
                    _atomic_json(ready_path, {
                        "schema": CONTROLLER_STATE_SCHEMA,
                        "status": "error",
                        "controller_pid": os.getpid(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    })
                finally:
                    os._exit(1)
            os._exit(0)

        os.close(startup_read)
        reaped = False
        old_handlers = {}
        cleaning_up = [False]


        def controlled_interrupt(signum: int, _frame: Any) -> None:
            if not cleaning_up[0]:
                raise _ParentSignal(signum)

        try:
            for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, controlled_interrupt)
            startup = prepare(pid)
            if not isinstance(startup, Mapping):
                raise DrillError("parent startup preparation returned invalid data")
            payload = canonical_bytes(startup)
            written = 0
            while written < len(payload):
                written += os.write(startup_write, payload[written:])
            os.close(startup_write)
            startup_write = -1

            while clock.time() < deadline_epoch:
                if ready_path.exists():
                    state, raw = _read_json_regular(
                        ready_path, "controller state")
                    if state.get("controller_pid") != pid:
                        raise DrillError(
                            "controller state names a different pid")
                    if state.get("status") == "ready":
                        if not _verify_seal(state, "state_sha256"):
                            raise DrillError(
                                "controller ready state seal is invalid")
                        status = self._terminate_and_wait(pid)
                        reaped = True
                        if (not os.WIFSIGNALED(status)
                                or os.WTERMSIG(status) != signal.SIGKILL):
                            raise DrillError(
                                "supervisor did not observe SIGKILL termination")
                        killed_at = _utc(clock.time())
                        return _seal({
                            "schema": KILL_EVENT_SCHEMA,
                            "receipt_sha256": "",
                            "controller_pid": pid,
                            "signal": "SIGKILL",
                            "ready_state_sha256": _sha256_bytes(raw),
                            "killed_at": killed_at,
                            "wait_status": int(status),
                        }, "receipt_sha256")
                    if state.get("status") == "error":
                        status = self._terminate_and_wait(pid)
                        reaped = True
                        raise DrillError(
                            "controller failed before loss drill: %s (status %d)"
                            % (state.get("error", "unknown error"), status))
                    raise DrillError("controller state has an unknown status")
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    reaped = True
                    raise DrillError(
                        "controller exited before supervisor kill (status %d)"
                        % status)
                clock.sleep(0.2)
            raise DrillError(
                "controller did not publish ready evidence before its deadline")
        except BaseException:
            raise
        finally:
            cleaning_up[0] = True
            if startup_write >= 0:
                os.close(startup_write)
            if not reaped:
                self._terminate_and_wait(pid)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def _offer_fields(offer: Any) -> Dict[str, Any]:
    if isinstance(offer, Mapping):
        return dict(offer)
    return {name: getattr(offer, name, None) for name in (
        "gpu_type", "region", "price", "spot", "free_devices", "workload_type")}


def _inventory_resources(inventory: Mapping[str, Any]) -> Tuple[Sequence[Any], Sequence[Any]]:
    families = inventory.get("families")
    if not isinstance(families, Mapping):
        raise DrillError("RunPod chargeable inventory lacks resource families")
    pods = families.get("pods")
    volumes = families.get("network_volumes")
    if (not isinstance(pods, Mapping) or not isinstance(volumes, Mapping)
            or pods.get("complete") is not True or volumes.get("complete") is not True
            or not isinstance(pods.get("resources"), list)
            or not isinstance(volumes.get("resources"), list)):
        raise DrillError("RunPod pod/network-volume inventory is incomplete")
    return pods["resources"], volumes["resources"]

def _campaign_inventory_rows(
        pods: Sequence[Any],
        volumes: Sequence[Any]) -> Sequence[Dict[str, str]]:
    resources = []
    for family, rows in (("pods", pods), ("network_volumes", volumes)):
        for row in rows:
            if not isinstance(row, Mapping):
                raise DrillError("RunPod inventory resource is not an object")
            resource = {
                "family": family, "id": row.get("id"),
                "name": row.get("name"), "status": row.get("status"),
            }
            if any(not isinstance(resource[key], str) or not resource[key]
                   for key in ("id", "name", "status")):
                raise DrillError(
                    "RunPod inventory resource identity is incomplete")
            resources.append(resource)
    resources.sort(key=lambda row: (row["family"], row["id"]))
    return resources


def _lifecycle_resources(provider: Any) -> list:
    """Return one mapping shape for real RunPod and test-provider listings."""
    listing = getattr(provider, "list_lifecycle_resources", None)
    source = (
        listing() if callable(listing)
        else getattr(provider, "list_instances")())
    if not isinstance(source, (list, tuple)):
        raise DrillError("RunPod lifecycle inventory is incomplete")
    normalized = []
    for row in source:
        if isinstance(row, Mapping):
            item = dict(row)
            resource_id = (
                item.get("id") or item.get("pod_id")
                or item.get("machine_id"))
            name = item.get("name")
            status = item.get("status")
        else:
            raw = getattr(row, "raw", None)
            item = dict(raw) if isinstance(raw, Mapping) else {}
            resource_id = getattr(row, "machine_id", None)
            name = getattr(row, "name", None)
            status = getattr(row, "status", None)
        resource_id = str(resource_id or "").strip()
        name = str(name or "").strip()
        status = str(status or "").strip()
        if not resource_id or not name or not status:
            raise DrillError(
                "RunPod lifecycle resource identity is incomplete")
        item.update({"id": resource_id, "name": name, "status": status})
        normalized.append(item)
    return normalized


def _provider_account_id(provider: Any) -> str:
    status = provider.status()
    account_id = status.get("id") if isinstance(status, Mapping) else None
    if (not isinstance(account_id, str) or not account_id
            or account_id.strip() != account_id or len(account_id) > 256
            or any(ord(character) < 0x21 or ord(character) > 0x7e
                   for character in account_id)):
        raise DrillError("RunPod myself.id is unavailable or invalid")
    return account_id


def _require_current_manifests(
        refresh: Callable[[], Mapping[str, Any]], bundle_digest: str,
        control_digest: str) -> None:
    current = refresh()
    if (not isinstance(current, Mapping)
            or current.get("bundle_contract_sha256") != bundle_digest
            or current.get("control_manifest_sha256") != control_digest):
        raise DrillError("current bundle/control manifest bytes changed")


def _build_quote(args: Any, job: Mapping[str, Any], rate: Decimal,
                 planned_at: str, terminate_seconds: int,
                 workload_seconds: int) -> CostQuote:
    hard_cap = getattr(args, "max_cost", None)
    if hard_cap is None:
        raise DrillError("paid drill requires --max-cost")
    quoted = datetime.strptime(planned_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    # The five storage tariffs are flag values whose defaults are RunPod's
    # published rates; the controller warns when the pin is old.  A
    # literal-equality rule plus a seven-day validity window ending
    # 2026-09-07 refused every drill from that date with no override once
    # the flag that extended it was retired.  The GPU rate is always the
    # live offer, and --max-cost still caps the drill.
    for name in (
            "runpod_container_running_tariff",
            "runpod_container_stopped_tariff",
            "runpod_pod_running_tariff", "runpod_pod_stopped_tariff",
            "runpod_network_tariff"):
        value = _decimal(getattr(args, name, "0"), name)
        if not value.is_finite() or value < 0:
            raise DrillError("RunPod drill tariff %s must be non-negative"
                             % name)
    tariff_effective_at = str(getattr(
        args, "tariff_effective_at", "2026-08-31T00:00:00Z"))
    if (_utc_epoch(tariff_effective_at, "tariff effective time")
            > int(quoted.timestamp())):
        raise DrillError("tariff effective time is in the future")
    valid_until = quoted + timedelta(seconds=MAX_QUOTE_VALIDITY_SECONDS)
    quote = CostQuote(
        reserved_compute_usd_per_hour=rate,
        live_compute_usd_per_hour=rate,
        container_disk_size_gb=Decimal(DEFAULT_STORAGE_GB),
        container_disk_running_usd_per_gb_month=Decimal(str(getattr(
            args, "runpod_container_running_tariff", "0.10"))),
        container_disk_stopped_usd_per_gb_month=Decimal(str(getattr(
            args, "runpod_container_stopped_tariff", "0.00"))),
        pod_disk_size_gb=Decimal(DEFAULT_STORAGE_GB),
        pod_disk_running_usd_per_gb_month=Decimal(str(getattr(
            args, "runpod_pod_running_tariff", "0.10"))),
        pod_disk_stopped_usd_per_gb_month=Decimal(str(getattr(
            args, "runpod_pod_stopped_tariff", "0.20"))),
        network_volume_size_gb=Decimal(0),
        network_volume_usd_per_gb_month=Decimal(str(getattr(
            args, "runpod_network_tariff", "0.07"))),
        storage_month_hours=Decimal(672),
        network_billing_increment_seconds=Decimal(3600),
        tariff_source=RUNPOD_TARIFF_SOURCE,
        tariff_effective_at=tariff_effective_at,
        quoted_at=planned_at,
        valid_until=valid_until.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target="%s@%s" % (job["target"]["repo_id"], job["target"]["revision"]),
        profile="runpod-drill-secure-l4-on-demand",
        timing_kind="exact-target-profile",
        timing_evidence=_sha256(job["timing"]),
        workload_deadline_seconds=Decimal(workload_seconds),
        provider_termination_deadline_seconds=Decimal(terminate_seconds),
        retrieval_delete_reserve_seconds=Decimal(terminate_seconds - workload_seconds),
        timer_api_lag_seconds=Decimal(DRILL_LAG_SECONDS),
        hard_cap_usd=Decimal(str(hard_cap)),
    )
    if quote.calculated_maximum_usd() > quote.hard_cap_usd:
        raise DrillError("drill all-in maximum exceeds --max-cost")
    return quote


def plan_drill(args: Any, provider: Any, *, seams: Optional[DrillSeams] = None) -> DrillPlan:
    """Perform every credential/provider/reaper/ledger check without mutation."""
    seams = seams or DrillSeams()
    now = float(seams.clock.time())
    planned_at = _utc(now)
    job_document = getattr(args, "runpod_drill_job_document", None)
    if isinstance(job_document, Mapping):
        job = json.loads(canonical_bytes(job_document).decode("utf-8"))
    else:
        job_path = Path(getattr(args, "runpod_drill_job_json", ""))
        job, _job_raw = _read_json_regular(job_path, "finalized drill job")
    verify_job(job)
    bundle_digest = str(getattr(args, "runpod_drill_bundle_manifest_sha256", ""))
    control_digest = str(getattr(args, "runpod_drill_control_manifest_sha256", ""))
    if (len(bundle_digest) != 64 or len(control_digest) != 64
            or any(c not in "0123456789abcdef" for c in bundle_digest + control_digest)):
        raise DrillError("current bundle/control manifest digests are required")
    verify_bundle_manifest(job["bundle"])
    if (job.get("bundle_contract_sha256") != bundle_digest
            or (job.get("control_plane") or {}).get("manifest_sha256") != control_digest):
        raise DrillError("drill job is not bound to current bundle/control manifests")
    manifest_refresh = getattr(args, "runpod_drill_manifest_refresh", None)
    if not callable(manifest_refresh):
        raise DrillError("current bundle/control manifest refresh is required")
    _require_current_manifests(
        manifest_refresh, bundle_digest, control_digest)

    provider.require()
    ssh_preflight = getattr(provider, "preflight_ssh_key", None)
    if ssh_preflight is None:
        ssh_preflight = getattr(provider, "_validated_ssh_public_key", None)
    if not callable(ssh_preflight) or not ssh_preflight():
        raise DrillError("RunPod SSH key preflight is unavailable")
    provider_account_id = _provider_account_id(provider)
    state_dir = Path(getattr(args, "reaper_state_dir"))
    lease_dir = Path(getattr(args, "lease_dir"))
    if (not state_dir.is_absolute() or not lease_dir.is_absolute()
            or lease_dir.parent.resolve() != state_dir.resolve()):
        raise DrillError(
            "lease dir must be an absolute direct child of reaper state dir")
    health = seams.reaper_health_check(
        state_dir=state_dir, lease_dir=lease_dir, provider="runpod",
        provider_account_id=provider_account_id, now=now)
    if health.get("ok") is not True:
        raise DrillError("healthy installed RunPod reaper is required")
    try:
        validate_unresolved_lease_scope(
            seams.lease_store_factory(lease_dir), health,
            provider="runpod", provider_account_id=provider_account_id,
            require_empty=True)
    except Exception as exc:
        raise DrillError(
            "bootstrap drill requires zero health-bound unresolved leases: %s"
            % exc)

    balance_raw = provider.balance()
    if balance_raw is None:
        raise DrillError("fresh RunPod balance is unavailable")
    balance = _decimal(balance_raw, "RunPod balance")
    inventory = provider.chargeable_inventory()
    if inventory.get("complete") is not True or inventory.get("provider") != "runpod":
        raise DrillError("fresh complete RunPod inventory is unavailable")
    inventory_observed = _utc_epoch(
        inventory.get("observed_at_utc"), "RunPod inventory observation")
    if inventory_observed > now + 5 or now - inventory_observed > 60:
        raise DrillError("RunPod chargeable inventory observation is not fresh")
    pods, volumes = _inventory_resources(inventory)
    inventory_rows = _campaign_inventory_rows(pods, volumes)
    pre_resources = tuple(_lifecycle_resources(provider))
    pod_ids = {row["id"] for row in inventory_rows
               if row["family"] == "pods"}
    listing_ids = {
        str(row.get("id")) for row in pre_resources
        if isinstance(row, Mapping) and row.get("id") is not None}
    if listing_ids != pod_ids:
        raise DrillError(
            "complete RunPod inventory and instance listing disagree")

    offers = [_offer_fields(item) for item in provider.gpus(
        gpu_type=GPU_TYPE, secure_only=True)]
    candidates = [item for item in offers
                  if item.get("gpu_type") == GPU_TYPE
                  and item.get("region") == "secure"
                  and item.get("spot") is False
                  and int(item.get("free_devices") or 0) >= 1
                  and item.get("workload_type") == "container"]
    if len(candidates) != 1:
        raise DrillError("exactly one available secure on-demand L4 offer is required")
    rate = _decimal(candidates[0].get("price"), "live L4 rate")
    workload_seconds = int(getattr(
        args, "runpod_drill_workload_seconds", DEFAULT_WORKLOAD_SECONDS))
    terminate_seconds = int(getattr(
        args, "runpod_drill_terminate_seconds", DEFAULT_TERMINATE_SECONDS))
    poll_seconds = int(getattr(args, "runpod_drill_poll_seconds", DEFAULT_POLL_SECONDS))
    billing_wait_seconds = int(getattr(
        args, "runpod_drill_billing_wait_seconds",
        DEFAULT_BILLING_WAIT_SECONDS))
    if (workload_seconds < 300 or terminate_seconds <= workload_seconds
            or poll_seconds <= 0 or poll_seconds > 60
            or billing_wait_seconds < 300 or billing_wait_seconds > 86400):
        raise DrillError(
            "drill timing requires workload>=300, termination>workload, "
            "poll 1..60, billing wait 300..86400")
    quote = _build_quote(args, job, rate, planned_at,
                         terminate_seconds, workload_seconds)
    job_base = copy.deepcopy(job)
    job_base.pop("job_id", None)
    job_base.pop("job_id_full", None)
    if (job_base.get("profile") or {}).get("profile_id") != (
            "runpod-drill-secure-l4-on-demand"):
        raise DrillError("drill job profile differs from the paid quote profile")
    job_base["environment"] = {
        "provider": "runpod",
        "provider_account_id": provider_account_id,
        "gpu_type": "L4",
        "provider_gpu_id": GPU_TYPE,
        "gpus": 1,
        "offer": "on-demand",
        "secure_cloud": True,
        "image": IMAGE,
        "image_reference_mutable": False,
        "hard_cap_usd": format(quote.hard_cap_usd, "f"),
        "price_per_gpu_hour": format(rate, "f"),
    }
    job_base["post_create_convergence"] = {
        "schema": "fidelity-suite/runpod-post-create-convergence.v1",
        "timeout_seconds": 180,
        "poll_seconds": 10,
    }
    job_base["execution_attempt"]["planned_at"] = planned_at
    job = finalize_job(job_base)
    job_hash = verify_job(job)

    attempt_id = str(seams.attempt_id_factory())
    if len(attempt_id) != 24 or any(c not in "0123456789abcdef" for c in attempt_id):
        raise DrillError("attempt seam did not produce 96-bit lowercase hex")
    from .cloudlease import exact_resource_name
    exact_name = exact_resource_name(job_hash, attempt_id)
    terminate_epoch = int(now) + terminate_seconds
    terminate_after = _utc(terminate_epoch)
    observation_until = _utc(terminate_epoch + DRILL_LAG_SECONDS)
    create_deadline = min(int(now) + 180, int(now) + workload_seconds)
    workload_deadline = int(now) + workload_seconds

    ledger_raw = getattr(args, "campaign_ledger", None)
    if not ledger_raw:
        raise DrillError("RunPod drill requires --campaign-ledger")
    ledger_path = Path(ledger_raw)
    if (not ledger_path.is_absolute()
            or ledger_path.parent.resolve() != state_dir.resolve()
            or ledger_path.name in ("", ".", "..")):
        raise DrillError(
            "campaign ledger must be an absolute direct child of reaper state dir")
    requested_width = int(getattr(args, "campaign_width", 2))
    ceiling = _decimal(getattr(args, "campaign_ceiling", None),
                       "campaign ceiling")
    reserve = _decimal(getattr(args, "campaign_reserve", None),
                       "campaign reserve")
    reaper_margin = _decimal(
        getattr(args, "campaign_reaper_margin", None),
        "campaign reaper margin")
    if requested_width not in (1, 2) or ceiling <= 0:
        raise DrillError("RunPod campaign must have positive ceiling and width at most two")
    configured_width = 2
    ledger_exists = ledger_path.is_file() and not ledger_path.is_symlink()
    if ledger_path.exists() and not ledger_exists:
        raise DrillError("campaign ledger path must be a regular non-symlink file")
    ledger_lock = Path(str(ledger_path) + ".lock")
    if ledger_exists and (
            not ledger_lock.is_file() or ledger_lock.is_symlink()):
        raise DrillError(
            "existing campaign ledger requires its durable regular lock file")
    admission_epoch = float(seams.clock.time())
    if admission_epoch < now:
        raise DrillError("local clock moved backwards during drill planning")
    admission_at = _utc(admission_epoch)
    validity = _utc(now + 300)
    inventory_observed_text = inventory["observed_at_utc"]
    inventory_validity = _utc(inventory_observed + 300)
    if ledger_exists:
        ledger = CampaignLedger(
            str(ledger_path), "runpod", provider_account_id)
        snapshot = ledger.snapshot()
        expected_limits = (
            ceiling, reserve, reaper_margin, configured_width)
        actual_limits = (
            Decimal(snapshot["hard_ceiling_usd"]),
            Decimal(snapshot["reserve_floor_usd"]),
            Decimal(snapshot["cleanup_reaper_margin_usd"]),
            snapshot["max_concurrent_attempts"])
        if (actual_limits != expected_limits
                or snapshot.get("authorized_concurrent_attempts") != 1
                or snapshot.get("width_authorization") is not None):
            raise DrillError(
                "shared campaign ledger is not pre-proof effective width one")
        prior_attempts = list((snapshot.get("attempts") or {}).values())
        if _bootstrap_drill_blocked_by_prior_attempts(prior_attempts):
            raise DrillError(
                "bootstrap drill must precede measurements and unsettled paid drills")
        classified = ledger.classify_provider_resources(inventory_rows)
        expected_unknown = [
            {"family": row["family"], "id": row["id"]}
            for row in inventory_rows]
        if (classified.get("generation") != snapshot["generation"]
                or classified.get("known_pod_ids") != []
                or classified.get("provider_resources") != inventory_rows
                or classified.get("unknown_resources") != expected_unknown):
            raise DrillError(
                "bootstrap inventory classification found ledger-owned resources")
        preview = ledger.preview_reserve_with_provider_snapshot(
            snapshot["generation"], job_hash, attempt_id, quote, admission_at,
            provider="runpod", provider_account_id=provider_account_id,
            provider_resources=inventory_rows,
            balance_available_usd=balance,
            balance_observed_at=planned_at,
            balance_valid_until=validity,
            balance_source="RunPod GraphQL myself.clientBalance",
            inventory_observed_at=inventory_observed_text,
            inventory_valid_until=inventory_validity,
            inventory_complete=True,
            inventory_source="RunPod complete pods+network-volumes inventory")
        ledger_generation = int(snapshot["generation"])
    else:
        preview = CampaignLedger.preview_new_campaign(
            hard_ceiling_usd=ceiling, reserve_floor_usd=reserve,
            cleanup_reaper_margin_usd=reaper_margin,
            max_concurrent_attempts=configured_width,
            provider="runpod", provider_account_id=provider_account_id,
            balance_available_usd=balance,
            provider_resources=inventory_rows,
            balance_observed_at=planned_at,
            balance_valid_until=validity,
            balance_source="RunPod GraphQL myself.clientBalance",
            inventory_observed_at=inventory_observed_text,
            inventory_valid_until=inventory_validity,
            inventory_complete=True,
            inventory_source="RunPod complete pods+network-volumes inventory",
            job_hash=job_hash, attempt=attempt_id, quote=quote,
            now=admission_at)
        ledger_generation = 0
    if not preview.admitted:
        raise DrillError("campaign dry admission refused: %s (%s)" %
                         (preview.message, preview.code))

    output_raw = getattr(args, "out", None)
    if not output_raw:
        raise DrillError("RunPod drill requires an explicit --out directory")
    checkout_initial = seams.checkout_status("all")
    output = Path(output_raw)
    return DrillPlan(
        job=job, job_hash=job_hash, attempt_id=attempt_id,
        bundle_contract_sha256=bundle_digest,
        control_manifest_sha256=control_digest,
        provider_account_id=provider_account_id,
        exact_name=exact_name, terminate_after=terminate_after,
        manifest_refresh=manifest_refresh,
        remote_helpers=_snapshot_remote_helpers(),
        observation_until=observation_until,
        create_deadline_epoch=float(create_deadline),
        workload_deadline_epoch=float(workload_deadline), quote=quote,
        offer_rate=rate, inventory=dict(inventory),
        pre_create_resources=pre_resources, balance_available_usd=balance,
        ledger_generation=ledger_generation,
        attempt_key=str(preview.attempt_key),
        lease_dir=lease_dir, reaper_state_dir=state_dir,
        campaign_ledger=ledger_path, output=output,
        storage_gb=DEFAULT_STORAGE_GB,
        container_disk_gb=DEFAULT_STORAGE_GB,
        poll_seconds=poll_seconds,
        billing_wait_seconds=billing_wait_seconds,
        campaign_ceiling=ceiling, campaign_reserve=reserve,
        campaign_reaper_margin=reaper_margin,
        ledger_exists=ledger_exists, reaper_health=dict(health),
        checkout_initial=checkout_initial,
        planned_at=planned_at)

def _remote_builder_source() -> str:
    return """#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
ROOT = Path('/workspace/fidelity-drill')
sys.path.insert(0, str(ROOT / 'lib'))
from fidelity import resultsink
job = json.loads((ROOT / 'result' / 'job.json').read_text(encoding='utf-8'))
(ROOT / 'result' / 'logs' / 'drill.log').write_text(
    'intentional controller-loss drill\\n', encoding='utf-8')
summary = resultsink.build_summary(ROOT / 'result', 'stage', 'abandoned',
                                   ['drill'], failed_stage='drill')
summary['utc'] = job['execution_attempt']['planned_at']
archive = ROOT / 'result-bundle.tar.gz'
archive_record = resultsink.write_archive(ROOT / 'result', summary, archive)
transfer = {
    'schema': 'fidelity-suite/runpod-drill-result-transfer.v1',
    'receipt_sha256': '',
    'path': 'result-bundle.tar.gz',
    'bytes': archive_record['bytes'],
    'sha256': archive_record['sha256'],
    'job_id_full': job['job_id_full'],
}
canonical = json.dumps(transfer, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False).encode('utf-8')
transfer['receipt_sha256'] = hashlib.sha256(canonical).hexdigest()
(ROOT / 'result-transfer.json').write_text(
    json.dumps(transfer, sort_keys=True, indent=2) + '\\n', encoding='utf-8')
"""


def _snapshot_remote_helpers() -> Tuple[Tuple[str, bytes, str], ...]:
    here = Path(__file__).resolve().parent
    rows = [("build_archive.py", _remote_builder_source().encode("utf-8"))]
    for name in (
            "__init__.py", "common.py", "jobcontract.py", "panel.py",
            "resultsink.py"):
        rows.append(("lib/fidelity/%s" % name, (here / name).read_bytes()))
    return tuple(
        (name, body, _sha256_bytes(body)) for name, body in rows)


def _prepare_remote_payload(
        stage: Path, job: Mapping[str, Any],
        helpers: Tuple[Tuple[str, bytes, str], ...]) -> Path:
    payload = stage / "controller-upload" / "fidelity-drill"
    result = payload / "result"
    (result / "logs").mkdir(mode=0o700, parents=True)
    (payload / "lib" / "fidelity").mkdir(mode=0o700, parents=True)
    _atomic_json(result / "job.json", job)
    _atomic_json(result / "ABANDONED.json", {
        "schema": "fidelity-suite/abandoned.v2",
        "reason": "intentional controller-loss autonomous-reaper drill",
        "job_id_full": job["job_id_full"],
    })
    expected_names = {
        "build_archive.py", "lib/fidelity/__init__.py",
        "lib/fidelity/common.py", "lib/fidelity/jobcontract.py",
        "lib/fidelity/panel.py", "lib/fidelity/resultsink.py",
    }
    if {row[0] for row in helpers} != expected_names:
        raise DrillError("frozen remote helper snapshot is incomplete")
    for name, body, digest in helpers:
        if _sha256_bytes(body) != digest:
            raise DrillError("frozen remote helper snapshot changed")
        _atomic_bytes(payload / Path(name), body)
    return payload


def _actual_quote(plan: DrillPlan, rate: Decimal) -> CostQuote:
    return replace(plan.quote, live_compute_usd_per_hour=rate)


def _require_exact_created_rate(
        response: Mapping[str, Any], binding: Mapping[str, Any],
        strict_row: Mapping[str, Any]) -> Decimal:
    acknowledged = _decimal(
        response.get("cost_per_hr"), "create-response acknowledged rate")
    observed = binding.get("observed")
    if not isinstance(observed, Mapping):
        raise DrillError("GraphQL lifecycle binding lacks observed economics")
    graphql = _decimal(
        observed.get("cost_per_hr"), "GraphQL lifecycle binding rate")
    rest = _decimal(
        strict_row.get("cost_per_hr"), "REST chargeable-inventory strict rate")
    if acknowledged <= 0 or graphql <= 0 or rest <= 0:
        raise DrillError("created pod rates must be positive")
    if acknowledged != graphql or acknowledged != rest or graphql != rest:
        raise DrillError(
            "create acknowledgement, GraphQL binding, and REST inventory "
            "rates differ")
    return acknowledged

def _converge_post_create(
        plan: DrillPlan, provider: Any, provider_id: str,
        clock: Callable[[], float],
        sleep: Callable[[float], None]) -> Tuple[
            Mapping[str, Any], Sequence[Any], Sequence[Any]]:
    """Wait for one exact pod in both identity and chargeable inventories."""
    contract = plan.job.get("post_create_convergence")
    if (not isinstance(contract, Mapping)
            or contract.get("schema")
            != "fidelity-suite/runpod-post-create-convergence.v1"
            or isinstance(contract.get("timeout_seconds"), bool)
            or not isinstance(contract.get("timeout_seconds"), int)
            or contract["timeout_seconds"] <= 0
            or isinstance(contract.get("poll_seconds"), bool)
            or not isinstance(contract.get("poll_seconds"), int)
            or contract["poll_seconds"] <= 0):
        raise DrillError("drill post-create convergence contract is invalid")
    deadline = min(
        float(clock()) + contract["timeout_seconds"],
        plan.workload_deadline_epoch)
    last_reason = "provider has not exposed the created pod"
    terminal = {"EXITED", "FAILED", "STOPPED", "TERMINATED", "DELETED"}
    while True:
        try:
            lifecycle = _lifecycle_resources(provider)
            inventory = provider.chargeable_inventory()
            status = provider.status()
            candidate_account_id = (
                status.get("id") if isinstance(status, Mapping) else None)
            account_id = (
                candidate_account_id
                if isinstance(candidate_account_id, str)
                and candidate_account_id
                and candidate_account_id.strip() == candidate_account_id
                and len(candidate_account_id) <= 256
                and all(0x21 <= ord(character) <= 0x7e
                        for character in candidate_account_id)
                else "")
            if not isinstance(lifecycle, list):
                last_reason = "RunPod lifecycle inventory is incomplete"
            elif not isinstance(inventory, Mapping):
                last_reason = "RunPod chargeable inventory is incomplete"
            else:
                provider_name = str(inventory.get("provider") or "")
                if provider_name and provider_name != "runpod":
                    raise DrillError(
                        "post-create inventory changed provider identity")
                if (account_id and
                        account_id != plan.provider_account_id):
                    raise DrillError(
                        "RunPod account changed after drill create")
                families = inventory.get("families")
                pods_family = (
                    families.get("pods")
                    if isinstance(families, Mapping) else None)
                volumes_family = (
                    families.get("network_volumes")
                    if isinstance(families, Mapping) else None)
                strict_pods = (
                    pods_family.get("resources")
                    if isinstance(pods_family, Mapping) else None)
                strict_volumes = (
                    volumes_family.get("resources")
                    if isinstance(volumes_family, Mapping) else None)
                strict_complete = bool(
                    inventory.get("complete") is True
                    and isinstance(pods_family, Mapping)
                    and pods_family.get("complete") is True
                    and isinstance(volumes_family, Mapping)
                    and volumes_family.get("complete") is True
                    and isinstance(strict_pods, list)
                    and isinstance(strict_volumes, list))

                def parsed_rows(rows: Any, label: str,
                                complete: bool) -> Optional[list]:
                    if not isinstance(rows, list):
                        return None
                    parsed = []
                    for row in rows:
                        if not isinstance(row, Mapping):
                            if complete:
                                raise DrillError(
                                    "%s contains a malformed resource" % label)
                            return None
                        rid = str(row.get("id") or "").strip()
                        name = str(row.get("name") or "").strip()
                        status = str(row.get("status") or "").strip().upper()
                        if not rid or not name or not status:
                            if complete:
                                raise DrillError(
                                    "%s contains incomplete resource identity"
                                    % label)
                            return None
                        parsed.append((rid, name, status, row))
                    return parsed

                lifecycle_rows = parsed_rows(
                    lifecycle, "post-create lifecycle inventory", True)
                pod_rows = parsed_rows(
                    strict_pods, "post-create chargeable pod inventory",
                    strict_complete)
                volume_rows = parsed_rows(
                    strict_volumes,
                    "post-create chargeable network-volume inventory",
                    strict_complete)
                known_pods = (
                    (lifecycle_rows or []) + (pod_rows or []))
                extra_ids = {
                    rid for rid, _name, _status, _row in known_pods
                    if rid != provider_id}
                if extra_ids or volume_rows:
                    combined_pods = {}
                    for rid, _name, _status, row in (
                            lifecycle_rows or []):
                        combined_pods[rid] = dict(row)
                    for rid, _name, _status, row in (pod_rows or []):
                        combined_pods[rid] = dict(row)
                    return (
                        inventory,
                        [combined_pods[rid] for rid in sorted(combined_pods)],
                        list(strict_volumes or []))
                intended = [
                    row for row in known_pods if row[0] == provider_id]
                if any(row[1] != plan.exact_name for row in intended):
                    raise DrillError(
                        "post-create intended pod has the wrong exact name")
                if any(row[2] in terminal for row in intended):
                    raise DrillError(
                        "post-create intended pod entered a terminal state")
                lifecycle_intended = [
                    row for row in (lifecycle_rows or [])
                    if row[0] == provider_id]
                strict_intended = [
                    row for row in (pod_rows or [])
                    if row[0] == provider_id]
                strict_rate_positive = False
                if len(strict_intended) == 1:
                    try:
                        strict_rate = Decimal(str(
                            strict_intended[0][3].get("cost_per_hr")))
                        strict_rate_positive = (
                            strict_rate.is_finite() and strict_rate > 0)
                    except (ValueError, TypeError):
                        pass
                ready = bool(
                    strict_complete and account_id == plan.provider_account_id
                    and len(lifecycle_intended) == 1
                    and len(strict_intended) == 1
                    and len(lifecycle_rows or []) == 1
                    and len(pod_rows or []) == 1
                    and not (volume_rows or [])
                    and lifecycle_intended[0][1] == plan.exact_name
                    and lifecycle_intended[0][2] == "RUNNING"
                    and strict_intended[0][1] == plan.exact_name
                    and strict_intended[0][2] == "RUNNING"
                    and strict_rate_positive)
                if ready:
                    return inventory, strict_pods, strict_volumes
                last_reason = (
                    "exact pod identity, RUNNING state, complete family "
                    "closure, and positive economics have not converged")
        except DrillError:
            raise
        except Exception as exc:
            last_reason = "provider observation failed: %s" % redact(str(exc))
        now = float(clock())
        if now >= deadline:
            raise DrillError(
                "post-create identity/economics convergence timed out: %s"
                % last_reason)
        sleep(min(float(contract["poll_seconds"]), deadline - now))




def _prepare_controller_lease(
        plan: DrillPlan, provider: Any, controller_pid: int,
        lease_store_factory: Callable[[Path], LeaseStore],
        prepared_create: Any, prepared_evidence: Mapping[str, Any],
        checkout_status: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    """Parent-side fallible preparation ending in one durable PREPARED lease."""
    store = lease_store_factory(plan.lease_dir)
    fresh = tuple(_lifecycle_resources(provider))
    if fresh:
        raise DrillError("pre-POST RunPod listing is not empty")
    paid_account_id = _provider_account_id(provider)
    if paid_account_id != plan.provider_account_id:
        raise DrillError("RunPod myself.id changed before the sole paid POST")
    _require_current_manifests(
        plan.manifest_refresh, plan.bundle_contract_sha256,
        plan.control_manifest_sha256)
    if any(_sha256_bytes(body) != digest
           for _name, body, digest in plan.remote_helpers):
        raise DrillError("frozen remote helper snapshot changed before POST")
    if prepared_create.to_dict() != prepared_evidence:
        raise DrillError("prepared create identity changed before lease")
    pre_create_safety = provider.server_time_evidence(
        max_clock_delta_seconds=30, max_evidence_age_seconds=30)
    pre_post_checkout = checkout_status("all")
    if (pre_post_checkout.get("clean") is not True
            or pre_post_checkout.get("revision")
                != plan.checkout_initial.get("revision")):
        raise DrillError(
            "producer HEAD/index/worktree changed before provider POST")
    observation_keys = (
        "untracked_files", "status_porcelain_sha256", "status_bytes", "clean")
    producer_checkout = {
        "schema": "fidelity-suite/producer-checkout.v1",
        "revision": plan.checkout_initial["revision"],
        "initial": {
            key: plan.checkout_initial[key] for key in observation_keys},
        "pre_post": {
            key: pre_post_checkout[key] for key in observation_keys},
    }
    request = {
        "drill_mode": LEASE_DRILL_KIND,
        "provider_account_id": paid_account_id,
        "campaign_ledger": plan.campaign_ledger.name,
        "campaign_attempt_key": plan.attempt_key,
        "secure_cloud": True,
        "offer": "on-demand",
        "spot": False,
        "gpu_type_id": GPU_TYPE,
        "gpu_count": 1,
        "image_name": IMAGE,
        "volume_gb": plan.storage_gb,
        "container_disk_gb": plan.container_disk_gb,
        "min_vcpu": MIN_VCPU,
        "min_ram_gb": MIN_RAM_GB,
        "network_volume_id": None,
        "terminate_after": plan.terminate_after,
        "provider_deadline_observation_until": plan.observation_until,
        "pre_create_safety": pre_create_safety,
        "prepared_create": prepared_evidence,
        "producer_checkout": producer_checkout,
    }
    coordinates = campaign_coordinates(
        {"create": {"request": request}}, plan.lease_dir)
    if (coordinates is None
            or coordinates != (plan.campaign_ledger, plan.attempt_key)):
        raise DrillError("shared lease campaign coordinate differs from drill plan")
    ref = store.begin_create(
        job_hash=plan.job_hash, provider="runpod", request=request,
        pre_create_resources=fresh,
        create_deadline_epoch=plan.create_deadline_epoch,
        workload_deadline_epoch=plan.workload_deadline_epoch,
        attempt_id=plan.attempt_id, controller_pid=controller_pid)
    prepared = store.read(ref)
    if (prepared.get("state") != "PREPARED"
            or prepared["create"].get("controller_pid") != controller_pid):
        raise DrillError("parent did not durably prepare the child-bound lease")
    return {
        "lease_name": ref.path.name,
        "lease_record_sha256": prepared["record_sha256"],
        "pre_create_safety": pre_create_safety,
        "producer_checkout": producer_checkout,
        "paid_account_id": paid_account_id,
    }


def _remote_regular_size(
        provider: Any, provider_id: str, remote: str, *,
        maximum: int, label: str) -> int:
    quoted = shlex.quote(remote)
    response = provider.exec(
        provider_id,
        "test -f %s && test ! -L %s && stat -c %%s -- %s"
        % (quoted, quoted, quoted),
        timeout=30)
    if (not isinstance(response, Mapping)
            or response.get("exit_code") != 0):
        raise DrillError("cannot stat remote %s" % label)
    stdout = response.get("stdout")
    if not isinstance(stdout, str) or re.fullmatch(r"[0-9]+\n?", stdout) is None:
        raise DrillError("remote %s size is not one exact integer" % label)
    size = int(stdout.strip())
    if size <= 0 or size > maximum:
        raise DrillError("remote %s exceeds its fixed byte bound" % label)
    return size


def _controller(plan: DrillPlan, provider: Any, stage: Path,
                ready_path: Path, startup: Mapping[str, Any],
                lease_store_factory: Callable[[Path], LeaseStore],
                prepared_create: Any, prepared_evidence: Mapping[str, Any],
                host_key_verifier: Callable[..., Dict[str, Any]],
                sleep: Callable[[float], None] = time.sleep,
                hold: bool = True) -> None:
    store = lease_store_factory(plan.lease_dir)
    ledger = CampaignLedger(
        str(plan.campaign_ledger), "runpod", plan.provider_account_id)
    lease_path = plan.lease_dir / str(startup.get("lease_name") or "")
    prepared = store.read(lease_path)
    ref = store.ref(lease_path, prepared)
    if (prepared.get("state") != "PREPARED"
            or prepared.get("record_sha256")
                != startup.get("lease_record_sha256")
            or prepared["create"].get("controller_pid") != os.getpid()):
        raise DrillError("controller startup does not bind its PREPARED lease")
    ledger_generation = startup.get("ledger_generation")
    if (isinstance(ledger_generation, bool)
            or not isinstance(ledger_generation, int)
            or ledger_generation <= 0):
        raise DrillError("controller startup campaign generation is invalid")
    pre_create_safety = startup.get("pre_create_safety")
    producer_checkout = startup.get("producer_checkout")
    paid_account_id = startup.get("paid_account_id")
    if (not isinstance(pre_create_safety, Mapping)
            or not isinstance(producer_checkout, Mapping)
            or paid_account_id != plan.provider_account_id):
        raise DrillError("controller startup evidence is invalid")
    try:
        generation = _require_transition(
            ledger.mark_creating(ledger_generation, plan.attempt_key),
            "campaign creating transition")
        # No key read, request formatting, provider read, or other fallible
        # preparation is permitted beyond this point.
        # This fsynced transition is the irrevocable boundary and is kept
        # immediately adjacent to the sole provider POST.
        ref = store.record_post_intent(ref)
    except BaseException:
        prepared = store.read(ref)
        if prepared.get("state") == "PREPARED":
            evidence = _sha256({
                "lease": ref.path.name,
                "state": "PREPARED",
                "request_sha256": prepared["create"]["request_sha256"],
                "no_provider_post": True,
            })
            cancelled = ledger.cancel_before_create(
                ledger.snapshot()["generation"], plan.attempt_key,
                _utc(store.clock()), "PREPARED", evidence)
            _require_transition(cancelled, "campaign pre-create cancellation")
            store.cancel_prepared(ref, {
                "campaign_cancellation_evidence_sha256": evidence,
                "reason": "pre-POST controller preparation failed",
            })
        raise
    try:
        ref, response = store.submit_create_and_record(
            ref, lambda: provider.submit_prepared_create(prepared_create))
    except RunPodCreateResponseError as exc:
        provider_id = str(exc.provider_id)
        ref = getattr(exc, "durable_lease_ref", None)
        if (ref is None
                or store.read(ref).get("provider_resource_ids")
                != [provider_id]):
            raise DrillError(
                "structured create response lacks its durable exact-ID "
                "lease binding") from exc
        snapshot = ledger.snapshot()
        current = snapshot["attempts"].get(plan.attempt_key) or {}
        if not current.get("provider_ids"):
            cleanup = ledger.bind_provider_for_cleanup(
                snapshot["generation"], plan.attempt_key, [provider_id],
                campaign_cleanup_binding_evidence(store.read(ref)))
            _require_transition(
                cleanup, "unqualified create response cleanup binding")
        ref = store.request_destroy(ref, {
            "reason": "unqualified create response; immediate cleanup",
            "provider_ids": [provider_id],
        })
        raise DrillError(
            "provider returned an unqualified create response for durably "
            "bound pod %s; cleanup only" % provider_id) from exc
    except CreateResponsePersistenceError as exc:
        provider_id = str(exc.provider_id or "")
        if not provider_id:
            raise DrillError(
                "committed create response lacks a recoverable exact ID"
            ) from exc
        lease = store.read(ref)
        snapshot = ledger.snapshot()
        current = snapshot["attempts"].get(plan.attempt_key) or {}
        if not current.get("provider_ids"):
            cleanup = ledger.bind_provider_for_cleanup(
                snapshot["generation"], plan.attempt_key, [provider_id],
                campaign_cleanup_binding_evidence(lease, [provider_id]))
            _require_transition(
                cleanup, "create persistence failure cleanup binding")
        raise DrillError(
            "provider committed pod %s but lease response persistence failed; "
            "campaign-bound reaper cleanup is required" % provider_id) from exc
    provider_id = str(response.get("pod_id") or response.get("machine_id") or "")
    if not provider_id:
        raise DrillError("single create POST returned no exact pod id")
    if store.read(ref).get("provider_resource_ids") != [provider_id]:
        raise DrillError("create response differs from its durable lease binding")
    try:
        if (response.get("prepared_create") != prepared_evidence
                or response.get("request")
                    != prepared_evidence.get("request_identity")):
            raise DrillError(
                "provider response does not bind the prepared create bytes")
        post_inventory, post_pods, post_volumes = _converge_post_create(
            plan, provider, provider_id, store.clock, sleep)
        post_inventory_checked_at = float(store.clock())
        post_inventory_epoch = _utc_epoch(
            post_inventory.get("observed_at_utc"),
            "post-create inventory observation")
        if (post_inventory_epoch > post_inventory_checked_at + 5
                or post_inventory_checked_at - post_inventory_epoch > 60):
            raise DrillError("post-create inventory observation is stale")
        ref = store.bind_post_create_inventory(
            ref, post_pods, network_volumes=post_volumes)
        if ref.state != "ACTIVE":
            raise DrillError(
                "post-create full inventory contains an ambiguous resource")
        created_rows = [
            row for row in post_pods
            if str((row or {}).get("id") or "") == provider_id]
        if (len(created_rows) != 1 or len(post_pods) != 1
                or post_volumes
                or str(created_rows[0].get("name") or "") != plan.exact_name
                or str(created_rows[0].get("status") or "").upper()
                != "RUNNING"):
            raise DrillError(
                "post-create campaign inventory is not the exact running pod")
        binding = provider.validate_safe_resource_binding(
            provider_id, expected_name=plan.exact_name, gpu_type_id=GPU_TYPE,
            secure_cloud=True, gpu_count=1, volume_gb=plan.storage_gb,
            container_disk_gb=plan.container_disk_gb, image_name=IMAGE,
            terminate_after=plan.terminate_after)
        if binding.get("passed") is not True:
            raise DrillError("post-create exact resource validation failed")
        actual_rate = _require_exact_created_rate(
            response, binding, created_rows[0])
        post_balance = _decimal(
            provider.balance(), "post-create RunPod balance")
        post_now = float(store.clock())
        post_snapshot = ledger.record_provider_snapshot(
            generation, provider="runpod",
            provider_account_id=plan.provider_account_id,
            balance_available_usd=post_balance,
            balance_observed_at=_utc(post_now),
            balance_valid_until=_utc(post_now + 300),
            balance_source="RunPod GraphQL myself.clientBalance",
            inventory_observed_at=post_inventory["observed_at_utc"],
            inventory_valid_until=_utc(post_inventory_epoch + 300),
            inventory_complete=True,
            provider_resources=[{
                "family": "pods", "id": provider_id,
                "name": plan.exact_name,
                "status": str(created_rows[0]["status"]),
            }],
            inventory_source=post_inventory["schema"])
        generation = _require_transition(
            post_snapshot, "post-create campaign provider snapshot")
        actual_quote = _actual_quote(plan, actual_rate)
        bound = ledger.bind_actual_quote(
            generation, plan.attempt_key, provider_id, actual_quote)
        generation = _require_transition(bound, "actual campaign quote binding")
        if bound.action != "CONTINUE":
            ref = store.request_destroy(ref, {
                "reason": "post-create actual quote refused; immediate cleanup",
                "provider_ids": [provider_id],
            })
            raise DrillError("created pod differed from the reserved live quote")
    except BaseException:
        snapshot = ledger.snapshot()
        current = snapshot["attempts"].get(plan.attempt_key) or {}
        if not current.get("provider_ids"):
            cleanup = ledger.bind_provider_for_cleanup(
                snapshot["generation"], plan.attempt_key, [provider_id],
                campaign_cleanup_binding_evidence(store.read(ref)))
            if getattr(cleanup, "applied", False):
                generation = cleanup.generation
        if store.read(ref)["state"] == "ACTIVE":
            ref = store.request_destroy(ref, {
                "reason": "post-create quote binding failed; immediate cleanup",
                "provider_ids": [provider_id],
            })
        raise
    try:
        submitted = response.get("request")
        submitted_expected = {
            "cloud_type": "SECURE", "is_spot": False,
            "offer": "on-demand", "gpu_type_id": GPU_TYPE,
            "gpu_count": 1, "volume_gb": plan.storage_gb,
            "container_disk_gb": plan.container_disk_gb,
            "min_vcpu": MIN_VCPU, "min_ram_gb": MIN_RAM_GB,
            "name": plan.exact_name, "image_name": IMAGE,
            "terminate_after": plan.terminate_after,
            "ports": "22/tcp", "volume_mount_path": "/workspace",
            "network_volume_id": None,
        }
        if (not isinstance(submitted, Mapping)
                or any(submitted.get(key) != value
                       for key, value in submitted_expected.items())):
            raise DrillError(
                "provider backend submitted request differs from durable lease")
        if binding.get("provider_id") not in (None, provider_id):
            raise DrillError("post-create exact resource binding id changed")
        host_key_proof = host_key_verifier(provider, provider_id, stage)
        if store.clock() >= plan.workload_deadline_epoch:
            raise DrillError(
                "authenticated host-key retrieval exhausted the drill "
                "workload deadline")
    except BaseException:
        if store.read(ref)["state"] == "ACTIVE":
            store.request_destroy(ref, {
                "reason": "post-create resource identity failed; immediate cleanup",
                "provider_ids": [provider_id],
            })
        raise

    job = copy.deepcopy(plan.job)
    job.pop("job_id", None)
    job.pop("job_id_full", None)
    job["execution_attempt"] = {
        "attempt_id": plan.attempt_id,
        "cost_quote": plan.quote.to_dict(),
        "engine_root": REMOTE_ROOT,
        "execution_contract_sha256": None,
        "kind": "runpod-ssh",
        "lease_path": ref.path.name,
        "planned_at": plan.planned_at,
        "pre_create_safety": {
            "checked_at": _utc(store.clock()),
            "reaper_health_sha256": _sha256(plan.reaper_health),
            "provider_account_id": paid_account_id,
            "provider_gpu_id": GPU_TYPE,
            "image": IMAGE,
            "bundle_contract_sha256": plan.bundle_contract_sha256,
            "control_manifest_sha256": plan.control_manifest_sha256,
            "server_time": pre_create_safety,
            "producer_checkout_sha256": _sha256(producer_checkout),
        },
        "prepared_create": prepared_evidence,
        "provider_terminate_after": plan.terminate_after,
        "remote_root": REMOTE_ROOT,
        "workload_deadline_utc": utc_iso(plan.workload_deadline_epoch),
    }
    job = seal_execution_job(finalize_job(job))
    validate_execution_job(job)
    if job["job_id_full"] != plan.job_hash:
        raise DrillError("execution attempt changed drill job identity")
    payload = _prepare_remote_payload(stage, job, plan.remote_helpers)
    provider.exec(provider_id, "rm -rf %s && mkdir -p /workspace" % REMOTE_ROOT,
                  timeout=120)
    provider.upload(provider_id, str(payload), "/workspace/")
    provider.exec(provider_id, "python3 %s/build_archive.py" % REMOTE_ROOT,
                  timeout=180)
    result_path = stage / "artifacts" / "result-bundle.tar.gz"
    transfer_path = stage / "artifacts" / "result-transfer.json"
    result_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    transfer_remote = "%s/result-transfer.json" % REMOTE_ROOT
    transfer_size = _remote_regular_size(
        provider, provider_id, transfer_remote,
        maximum=TRANSFER_RECEIPT_MAX_BYTES,
        label="result transfer receipt")
    provider.download_bounded(
        provider_id, transfer_remote, str(transfer_path),
        expected_bytes=transfer_size, max_bytes=TRANSFER_RECEIPT_MAX_BYTES,
        timeout=180)
    transfer, _transfer_raw = _read_json_regular(
        transfer_path, "on-pod result transfer receipt")
    transfer_keys = {
        "schema", "receipt_sha256", "path", "bytes", "sha256", "job_id_full"}
    transfer_bytes = transfer.get("bytes")
    if (set(transfer) != transfer_keys
            or transfer.get("schema") != TRANSFER_SCHEMA
            or not _verify_seal(transfer, "receipt_sha256")
            or transfer.get("path") != "result-bundle.tar.gz"
            or transfer.get("job_id_full") != plan.job_hash
            or isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or not 0 < transfer_bytes <= DRILL_ARCHIVE_MAX_BYTES
            or re.fullmatch(
                r"[0-9a-f]{64}", str(transfer.get("sha256") or "")) is None):
        raise DrillError("on-pod result transfer receipt is invalid")
    provider.download_bounded(
        provider_id, "%s/result-bundle.tar.gz" % REMOTE_ROOT,
        str(result_path), expected_bytes=transfer_bytes,
        max_bytes=DRILL_ARCHIVE_MAX_BYTES, timeout=180)
    verified = verify_archive(
        result_path, expected_sha256=transfer["sha256"],
        expected_bytes=transfer_bytes)
    if (verified.get("manifest") or {}).get("job_id_full") != plan.job_hash:
        raise DrillError("retrieved archive is bound to a different job")
    state = _seal({
        "schema": CONTROLLER_STATE_SCHEMA,
        "state_sha256": "",
        "status": "ready",
        "controller_pid": os.getpid(),
        "provider_id": provider_id,
        "exact_name": plan.exact_name,
        "lease_name": ref.path.name,
        "campaign_generation": generation,
        "result_archive_sha256": verified["archive_sha256"],
        "result_archive_bytes": verified["archive_bytes"],
        "result_transfer_receipt_sha256": transfer["receipt_sha256"],
        "job_id_full": plan.job_hash,
        "ssh_host_key_proof_sha256": host_key_proof["proof_sha256"],
    }, "state_sha256")
    _atomic_json(ready_path, state)
    if hold:
        while True:
            signal.pause()


def _billing_arithmetic(plan: DrillPlan, lease: Mapping[str, Any],
                        exact_id: str) -> Dict[str, Any]:
    billing = (lease.get("terminal_proof") or {}).get("billing_reconciliation")
    histories = billing.get("billing_histories") if isinstance(billing, dict) else None
    if not isinstance(histories, list) or len(histories) != 1:
        raise DrillError("terminal lease lacks exactly one billing history")
    history = histories[0]
    if history.get("pod_id") != exact_id:
        raise DrillError("billing history pod id differs from drilled pod")
    records = history.get("records")
    totals = (history.get("metadata") or {}).get("totals")
    if not isinstance(records, list) or not records or not isinstance(totals, dict):
        raise DrillError("billing history lacks records/totals")
    fields = ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount")
    sums = {}
    for field in fields:
        sums[field] = sum((_decimal(row.get(field), "billing %s" % field)
                           for row in records), Decimal(0))
        reported = _decimal(totals.get(field), "billing total %s" % field)
        if not _billing_total_matches_record_sum(reported, sums[field]):
            raise DrillError("billing %s arithmetic does not reconcile" % field)
    total = _decimal(billing.get("total_amount"), "lease total billing")
    if not _billing_total_matches_record_sum(total, sums["totalAmount"]):
        raise DrillError("lease total billing differs from bounded record sum")
    return _seal({
        "schema": BILLING_SCHEMA,
        "receipt_sha256": "",
        "pod_id": exact_id,
        "job_id_full": plan.job_hash,
        "attempt_id": plan.attempt_id,
        "lease_record_sha256": lease["record_sha256"],
        "record_count": len(records),
        "validated_sums": {key: format(value, "f") for key, value in sums.items()},
        "total_amount": format(total, "f"),
        "billing_evidence_sha256": _sha256(history),
    }, "receipt_sha256")
def _campaign_release_receipt(
        plan: DrillPlan, state: Mapping[str, Any],
        lease: Mapping[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    ledger = CampaignLedger(
        str(plan.campaign_ledger), "runpod", plan.provider_account_id)
    final = ledger.snapshot()
    item = final["attempts"].get(plan.attempt_key)
    if (not isinstance(item, dict) or item.get("released") is not True
            or item.get("phase") != "RECONCILED"
            or item.get("maximum_remaining_liability_usd") != "0"):
        raise DrillError(
            "autonomous reaper did not release exact campaign liability")
    exact_id = str(state["provider_id"])
    if item.get("provider_ids") != [exact_id]:
        raise DrillError("campaign release does not bind the exact provider id")
    ledger_document, ledger_raw = _read_json_regular(
        plan.campaign_ledger, "current campaign ledger")
    if ledger_document != final:
        raise DrillError("campaign ledger changed during release capture")
    receipt = _seal({
        "schema": CAMPAIGN_RELEASE_SCHEMA,
        "receipt_sha256": "",
        "attempt_key": plan.attempt_key,
        "job_id_full": plan.job_hash,
        "attempt_id": plan.attempt_id,
        "provider": "runpod",
        "provider_account_id": plan.provider_account_id,
        "provider_id": exact_id,
        "ledger_generation": final["generation"],
        "campaign_ledger_sha256": _sha256_bytes(ledger_raw),
        "campaign_ledger_path_sha256":
            campaign_ledger_coordinate_sha256(plan.campaign_ledger),
        "settled_charges_usd": final["settled_charges_usd"],
        "released": True,
        "phase": "RECONCILED",
        "maximum_remaining_liability_usd": "0",
        "final_charge_usd": item["billing"]["final_charge_usd"],
        "reserved_quote": item["reserved_quote"],
        "actual_quote": item["actual_quote"],
        "lease_record_sha256": lease["record_sha256"],
    }, "receipt_sha256")
    return receipt, ledger_raw


def _require_post_loss_health(
        plan: DrillPlan, health: Mapping[str, Any],
        provider_id: str, controller_lost_epoch: int) -> Mapping[str, Any]:
    stamp = health.get("stamp") if isinstance(health, Mapping) else None
    control = stamp.get("control") if isinstance(stamp, Mapping) else None
    initial_stamp = plan.reaper_health.get("stamp") or {}
    initial_control = initial_stamp.get("control") or {}
    started = float((stamp or {}).get("invocation_started_at_epoch", 0))
    completed = float((stamp or {}).get("completed_at_epoch", 0))
    if (health.get("ok") is not True or health.get("stamp_ok") is not True
            or health.get("control_ok") is not True
            or not isinstance(stamp, Mapping)
            or stamp.get("schema") != HEALTH_SCHEMA
            or not isinstance(control, Mapping)
            or started <= controller_lost_epoch
            or completed < started
            or completed < _utc_epoch(plan.terminate_after, "reap deadline")
            or completed > _utc_epoch(
                plan.observation_until, "reaper observation bound")
            or control.get("control_sha256") != initial_control.get("control_sha256")
            or control.get("provider") != "runpod"
            or control.get("provider_account_id_sha256")
                != _sha256_bytes(plan.provider_account_id.encode("utf-8"))):
        raise DrillError(
            "installed reaper lacks a new account-bound deadline invocation")
    actions = stamp.get("actions")
    if (not isinstance(actions, list)
            or not any(
                isinstance(action, Mapping)
                and action.get("action") == "destroy-requested"
                and action.get("provider_id") == provider_id
                and isinstance(action.get("lease_generation"), int)
                and not isinstance(action.get("lease_generation"), bool)
                and re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(action.get("lease_record_sha256") or ""))
                for action in actions)):
        raise DrillError(
            "post-loss systemd sweep does not bind the exact destroy request")
    for row in control.get("control_files") or []:
        path = PurePosixPath(str((row or {}).get("path") or ""))
        if (path.is_absolute() or any(part in ("", ".", "..") for part in path.parts)
                or "\\" in str(path)):
            raise DrillError("reaper health exposes an unsafe control path")
    return stamp

def _persist_deadline_observation(
        path: Path, plan: DrillPlan, provider_id: str,
        resources: Sequence[Any], started_at: float, completed_at: float,
        observations: list) -> Tuple[bool, Mapping[str, Any]]:
    if (not math.isfinite(float(started_at))
            or not math.isfinite(float(completed_at))
            or completed_at < started_at
            or completed_at - started_at
                > DEADLINE_POLL_DURATION_MAX_SECONDS):
        raise DrillError("lifecycle poll timestamps are invalid")
    rows = []
    for resource in resources:
        if not isinstance(resource, Mapping):
            raise DrillError("complete lifecycle inventory contains malformed row")
        row = {
            "id": str(
                resource.get("id") or resource.get("pod_id") or "").strip(),
            "name": str(resource.get("name") or "").strip(),
            "status": str(resource.get("status") or "").strip(),
        }
        if any(not row[key] for key in ("id", "name", "status")):
            raise DrillError(
                "complete lifecycle inventory row lacks exact identity")
        rows.append(row)
    rows.sort(key=lambda item: item["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise DrillError("complete lifecycle inventory repeats an exact id")
    exact_rows = [row for row in rows if row["id"] == provider_id]
    if exact_rows and exact_rows[0]["name"] != plan.exact_name:
        raise DrillError("exact lifecycle pod id changed its provider name")
    exact_present = bool(exact_rows)
    deadline = float(_utc_epoch(plan.terminate_after, "terminateAfter"))
    relation = (
        "BEFORE" if completed_at < deadline
        else "BOUNDARY" if completed_at == deadline
        else "AFTER")
    row = {
        "sequence": len(observations) + 1,
        "poll_started_at_epoch": float(started_at),
        "poll_completed_at_epoch": float(completed_at),
        "poll_completed_at_utc": _utc(completed_at),
        "deadline_relation": relation,
        "complete": True,
        "exact_present": exact_present,
        "provider_ids": [item["id"] for item in rows],
        "resources": rows,
        "listing_sha256": _sha256(rows),
    }
    observations.append(row)
    document = _seal({
        "schema": DEADLINE_OBSERVATION_SCHEMA,
        "record_sha256": "",
        "provider": "runpod",
        "provider_account_id": plan.provider_account_id,
        "exact_pod_id": provider_id,
        "exact_pod_name": plan.exact_name,
        "terminate_after": plan.terminate_after,
        "provider_deadline_observation_until": plan.observation_until,
        "poll_interval_seconds": plan.poll_seconds,
        "poll_duration_max_seconds": DEADLINE_POLL_DURATION_MAX_SECONDS,
        "interpoll_gap_max_seconds": DEADLINE_INTERPOLL_GAP_MAX_SECONDS,
        "observations": list(observations),
    }, "record_sha256")
    _atomic_json(path, document)
    return exact_present, row



def _request_failed_controller_cleanup(
        plan: DrillPlan, ledger: CampaignLedger,
        lease_store_factory: Callable[[Path], LeaseStore]) -> bool:
    """Give the reaper immediate authority after a pre-loss controller failure."""
    lease_path = plan.lease_dir / (
        "%s.%s.json" % (plan.job_hash, plan.attempt_id))
    if not lease_path.exists() and not lease_path.is_symlink():
        return False
    store = lease_store_factory(plan.lease_dir)
    document = store.read(lease_path)
    if document.get("state") == "DESTROYING":
        return True
    if document.get("state") != "ACTIVE":
        return False
    provider_ids = list(document.get("provider_resource_ids") or [])
    if not provider_ids:
        raise DrillError("active failed-controller lease has no cleanup target")
    snapshot = ledger.snapshot()
    item = (snapshot.get("attempts") or {}).get(plan.attempt_key)
    if isinstance(item, Mapping) and not item.get("provider_ids"):
        cleanup = ledger.bind_provider_for_cleanup(
            snapshot["generation"], plan.attempt_key, provider_ids,
            campaign_cleanup_binding_evidence(document))
        _require_transition(
            cleanup, "failed-controller campaign cleanup binding")
    ref = store.ref(lease_path, document)
    store.request_destroy(ref, {
        "reason": "controller failed before intentional loss; immediate cleanup",
        "provider_ids": provider_ids,
    })
    return True


def _cancel_unreserved_prepared_lease(
        plan: DrillPlan, ledger: CampaignLedger,
        lease_store_factory: Callable[[Path], LeaseStore]) -> bool:
    """Close a parent-created PREPARED lease only when no campaign exists."""
    snapshot = ledger.snapshot()
    if (snapshot.get("attempts") or {}).get(plan.attempt_key) is not None:
        return False
    lease_path = plan.lease_dir / (
        "%s.%s.json" % (plan.job_hash, plan.attempt_id))
    if not lease_path.exists() and not lease_path.is_symlink():
        return False
    store = lease_store_factory(plan.lease_dir)
    document = store.read(lease_path)
    if document.get("state") != "PREPARED":
        return False
    ref = store.ref(lease_path, document)
    store.cancel_prepared(ref, {
        "reason": "parent startup failed before campaign reservation",
        "no_provider_post": True,
    })
    return True


def _cancel_lease_absent_reservation(
        plan: DrillPlan, ledger: CampaignLedger, provider: Any,
        clock: Any) -> bool:
    snapshot = ledger.snapshot()
    item = (snapshot.get("attempts") or {}).get(plan.attempt_key)
    lease_path = plan.lease_dir / (
        "%s.%s.json" % (plan.job_hash, plan.attempt_id))
    if not isinstance(item, Mapping) or item.get("phase") != "RESERVED":
        return False
    if lease_path.exists() or lease_path.is_symlink():
        return False
    if _provider_account_id(provider) != plan.provider_account_id:
        return False
    listing = tuple(_lifecycle_resources(provider))
    inventory = provider.chargeable_inventory()
    pods, volumes = _inventory_resources(inventory)
    if (inventory.get("complete") is not True or listing or pods or volumes
            or lease_path.exists() or lease_path.is_symlink()):
        return False
    evidence = _sha256({
        "lease_name": lease_path.name,
        "lease_absent": True,
        "pre_reservation_inventory_sha256": _sha256(plan.inventory),
        "post_failure_inventory_sha256": _sha256(inventory),
        "pre_reservation_provider_listing": list(plan.pre_create_resources),
        "campaign_phase": "RESERVED",
        "provider_account_id": plan.provider_account_id,
        "complete_provider_listing": [],
        "complete_chargeable_inventory": {"pods": [], "network_volumes": []},
        "no_post_authorized": True,
    })
    cancelled = ledger.cancel_before_create(
        snapshot["generation"], plan.attempt_key, _utc(clock.time()),
        "LEASE_ABSENT", evidence)
    _require_transition(cancelled, "lease-absent campaign cancellation")
    return True


def execute_drill(plan: DrillPlan, args: Any, provider: Any, *,
                  seams: Optional[DrillSeams] = None) -> Path:
    """Execute one paid mutation and atomically publish its accepted proof."""
    seams = seams or DrillSeams()
    if getattr(args, "dry_run", False):
        raise DrillError("execute_drill cannot run in dry-run mode")
    if getattr(args, "yes", False) is not True:
        raise DrillError("paid RunPod drill requires explicit --yes")
    checkout_initial = seams.checkout_status("all")
    if (checkout_initial != plan.checkout_initial
            or checkout_initial.get("clean") is not True):
        raise DrillError(
            "paid drill requires unchanged exact HEAD and fully clean checkout")
    execution_now = float(seams.clock.time())
    plan_age = execution_now - _utc_epoch(plan.planned_at, "plan timestamp")
    if plan_age < 0 or plan_age > 60:
        raise DrillError("drill plan/provider facts are stale or time-invalid")
    if _provider_account_id(provider) != plan.provider_account_id:
        raise DrillError("RunPod myself.id changed after dry planning")
    current_health = seams.reaper_health_check(
        state_dir=plan.reaper_state_dir, lease_dir=plan.lease_dir,
        provider="runpod", provider_account_id=plan.provider_account_id,
        now=execution_now)
    if current_health.get("ok") is not True:
        raise DrillError("healthy installed RunPod reaper changed after planning")
    try:
        validate_unresolved_lease_scope(
            seams.lease_store_factory(plan.lease_dir), current_health,
            provider="runpod", provider_account_id=plan.provider_account_id,
            require_empty=True)
    except Exception as exc:
        raise DrillError(
            "paid drill unresolved lease scope changed after planning: %s"
            % exc)
    current_balance_raw = provider.balance()
    if current_balance_raw is None:
        raise DrillError("fresh RunPod balance is unavailable at paid admission")
    current_balance = _decimal(current_balance_raw, "current RunPod balance")
    current_inventory = provider.chargeable_inventory()
    inventory_checked_at = float(seams.clock.time())
    if (current_inventory.get("complete") is not True
            or current_inventory.get("provider") != "runpod"):
        raise DrillError("RunPod inventory became incomplete after planning")
    inventory_at = _utc_epoch(
        current_inventory.get("observed_at_utc"),
        "paid RunPod inventory observation")
    if (inventory_at > inventory_checked_at + 5
            or inventory_checked_at - inventory_at > 60):
        raise DrillError("paid RunPod inventory observation is not fresh")
    current_pods, current_volumes = _inventory_resources(current_inventory)
    current_rows = _campaign_inventory_rows(current_pods, current_volumes)
    current_listing = tuple(_lifecycle_resources(provider))
    listed_ids = {
        str(row.get("id")) for row in current_listing
        if isinstance(row, Mapping) and row.get("id") is not None}
    inventory_ids = {
        row["id"] for row in current_rows if row["family"] == "pods"}
    if listed_ids != inventory_ids:
        raise DrillError(
            "paid RunPod full inventory and instance listing disagree")
    if current_rows:
        raise DrillError(
            "bootstrap drill classifies every existing provider resource "
            "as unknown and refuses spend")
    current_offers = [_offer_fields(item) for item in provider.gpus(
        gpu_type=GPU_TYPE, secure_only=True)]
    current_candidates = [
        item for item in current_offers
        if item.get("gpu_type") == GPU_TYPE
        and item.get("region") == "secure"
        and item.get("spot") is False
        and int(item.get("free_devices") or 0) >= 1
        and item.get("workload_type") == "container"]
    if len(current_candidates) != 1:
        raise DrillError("secure on-demand L4 offer changed after planning")
    current_rate = _decimal(
        current_candidates[0].get("price"), "paid live L4 rate")
    if current_rate != plan.offer_rate:
        raise DrillError("live L4 rate changed after planning; re-plan")
    refreshed_quote = _build_quote(
        args, plan.job, current_rate, _utc(execution_now),
        int(plan.quote.provider_termination_deadline_seconds),
        int(plan.quote.workload_deadline_seconds))
    workload_seconds = int(plan.quote.workload_deadline_seconds)
    terminate_seconds = int(plan.quote.provider_termination_deadline_seconds)
    workload_epoch = int(execution_now) + workload_seconds
    terminate_epoch = int(execution_now) + terminate_seconds
    plan = replace(
        plan, quote=refreshed_quote, offer_rate=current_rate,
        balance_available_usd=current_balance,
        inventory=dict(current_inventory), reaper_health=dict(current_health),
        planned_at=refreshed_quote.quoted_at,
        create_deadline_epoch=float(min(int(execution_now) + 180,
                                        workload_epoch)),
        workload_deadline_epoch=float(workload_epoch),
        terminate_after=_utc(terminate_epoch),
        observation_until=_utc(terminate_epoch + DRILL_LAG_SECONDS))
    if plan.output.exists():
        raise DrillError("drill output already exists; refusing replacement")
    create_kwargs = {
        "gpu_type": GPU_TYPE, "num_gpus": 1, "region": "secure",
        "spot": False, "offer": "on-demand", "name": plan.exact_name,
        "image": IMAGE, "storage_gb": plan.storage_gb,
        "container_disk_gb": plan.container_disk_gb,
        "min_vcpu": MIN_VCPU, "min_ram_gb": MIN_RAM_GB,
        "terminate_after": plan.terminate_after,
    }
    # Key reads, public-key verification, GraphQL formatting, HTTP request
    # construction, and body hashing all finish before campaign reservation.
    prepared_create = provider.prepare_safe_create(**create_kwargs)
    prepared_evidence = prepared_create.to_dict()
    ledger = CampaignLedger.create(
        str(plan.campaign_ledger), plan.campaign_ceiling,
        plan.campaign_reserve, plan.campaign_reaper_margin, 2,
        provider="runpod", provider_account_id=plan.provider_account_id)
    starting_generation = ledger.snapshot()["generation"]
    if plan.ledger_exists and starting_generation != plan.ledger_generation:
        raise DrillError("campaign ledger changed after dry planning; re-plan")
    plan.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        dir=str(plan.output.parent), prefix=".%s.drill-" % plan.output.name))
    ready_path = stage / "controller-state.json"
    try:
        def prepare_controller(controller_pid: int) -> Mapping[str, Any]:
            admission_store = seams.lease_store_factory(plan.lease_dir)
            with admission_store.paid_admission_lock():
                validate_unresolved_lease_scope(
                    admission_store, current_health,
                    provider="runpod",
                    provider_account_id=plan.provider_account_id,
                    require_empty=True)
                if ledger.snapshot()["generation"] != starting_generation:
                    raise DrillError(
                        "campaign ledger changed before atomic paid admission")
                # PREPARED and the matching reservation become visible while
                # every controller using this lease root is globally excluded.
                startup = _prepare_controller_lease(
                    plan, provider, controller_pid,
                    seams.lease_store_factory, prepared_create,
                    prepared_evidence, seams.checkout_status)
                now = float(seams.clock.time())
                now_utc = _utc(now)
                valid_until = _utc(now + 300)
                recorded = ledger.record_provider_snapshot(
                    starting_generation,
                    provider="runpod",
                    provider_account_id=plan.provider_account_id,
                    balance_available_usd=plan.balance_available_usd,
                    balance_observed_at=now_utc,
                    balance_valid_until=valid_until,
                    balance_source="RunPod GraphQL myself.clientBalance",
                    provider_resources=[],
                    inventory_observed_at=current_inventory[
                        "observed_at_utc"],
                    inventory_valid_until=_utc(inventory_at + 300),
                    inventory_complete=True,
                    inventory_source=(
                        "RunPod complete pods+network-volumes inventory"))
                generation = _require_transition(
                    recorded, "fresh campaign provider snapshot")
                admitted = ledger.reserve_bootstrap_drill(
                    generation, plan.job_hash, plan.attempt_id,
                    plan.quote, now_utc)
                if not admitted.admitted or admitted.action != "CREATE":
                    raise DrillError(
                        "atomic campaign reservation refused: %s (%s)" %
                        (admitted.message, admitted.code))
                released = dict(startup)
                released["ledger_generation"] = admitted.generation
                return released

        controller = lambda startup: _controller(
            plan, provider, stage, ready_path, startup,
            seams.lease_store_factory, prepared_create, prepared_evidence,
            seams.host_key_verifier, sleep=seams.clock.sleep,
            hold=bool(getattr(seams.supervisor, "controller_holds", True)))
        try:
            kill_event = seams.supervisor.supervise(
                controller, ready_path, plan.workload_deadline_epoch,
                seams.clock, prepare_controller)
        except BaseException:
            _request_failed_controller_cleanup(
                plan, ledger, seams.lease_store_factory)
            # supervise() does not unwind until its child has been reaped.
            _cancel_unreserved_prepared_lease(
                plan, ledger, seams.lease_store_factory)
            _cancel_lease_absent_reservation(
                plan, ledger, provider, seams.clock)
            raise
        state, state_raw = _read_json_regular(ready_path, "controller ready state")
        if (state.get("status") != "ready"
                or not _verify_seal(state, "state_sha256")
                or not _verify_seal(kill_event, "receipt_sha256")
                or kill_event.get("ready_state_sha256") != _sha256_bytes(state_raw)
                or kill_event.get("controller_pid") != state.get("controller_pid")):
            raise DrillError("supervisor kill event does not bind controller ready state")
        kill_path = stage / "artifacts" / "controller-kill-event.json"
        _atomic_json(kill_path, kill_event)

        store = seams.lease_store_factory(plan.lease_dir)
        lease_path = plan.lease_dir / str(state["lease_name"])
        finish_deadline = (
            _utc_epoch(plan.observation_until, "observation bound")
            + plan.billing_wait_seconds)
        controller_lost_epoch = _utc_epoch(
            kill_event["killed_at"], "controller loss")
        health = None
        terminate_epoch = float(
            _utc_epoch(plan.terminate_after, "terminateAfter"))
        deadline_observations = []
        deadline_observation_path = (
            stage / "artifacts" / "provider-deadline-observations.json")
        deadline_observation_path.parent.mkdir(
            mode=0o700, parents=True, exist_ok=True)
        predeadline_presence_observed = False
        first_absence_observation = None
        bounded_absence_proven = False
        destroy_health = None
        observation_until_epoch = float(
            _utc_epoch(plan.observation_until, "observation bound"))
        exact_provider_id = str(state["provider_id"])
        while seams.clock.time() <= finish_deadline:
            poll_started = float(seams.clock.time())
            if _provider_account_id(provider) != plan.provider_account_id:
                raise DrillError(
                    "RunPod account changed during post-loss observation")
            observed_resources = tuple(_lifecycle_resources(provider))
            poll_completed = float(seams.clock.time())
            exact_present, observation = _persist_deadline_observation(
                deadline_observation_path, plan, exact_provider_id,
                observed_resources, poll_started, poll_completed,
                deadline_observations)
            if poll_completed < terminate_epoch:
                if not exact_present:
                    raise DrillError(
                        "exact pod disappeared before provider terminateAfter")
                predeadline_presence_observed = True
            elif not exact_present and first_absence_observation is None:
                first_absence_observation = observation
                bounded_absence_proven = (
                    predeadline_presence_observed
                    and poll_completed <= observation_until_epoch)
            elif (first_absence_observation is not None
                  and exact_present):
                raise DrillError(
                    "exact pod reappeared after complete provider absence")

            if seams.autonomous_timer_tick is not None:
                seams.autonomous_timer_tick(
                    plan=plan, store=store, provider=provider,
                    now=seams.clock.time())
            lease = store.read(lease_path)
            health = seams.reaper_health_check(
                state_dir=plan.reaper_state_dir, lease_dir=plan.lease_dir,
                provider="runpod",
                provider_account_id=plan.provider_account_id,
                now=seams.clock.time())
            stamp = health.get("stamp") if isinstance(health, Mapping) else None
            if (destroy_health is None and isinstance(stamp, Mapping)
                    and any(
                        isinstance(action, Mapping)
                        and action.get("action") == "destroy-requested"
                        and action.get("provider_id") == exact_provider_id
                        for action in stamp.get("actions") or [])):
                destroy_health = json.loads(
                    canonical_bytes(health).decode("utf-8"))
            if lease["state"] == TERMINAL and not exact_present:
                if destroy_health is None:
                    raise DrillError(
                        "terminal cleanup lacks the autonomous destroy health stamp")
                _require_post_loss_health(
                    plan, destroy_health, exact_provider_id,
                    controller_lost_epoch)
                break
            now = float(seams.clock.time())
            next_poll = poll_started + float(plan.poll_seconds)
            if poll_started < terminate_epoch:
                next_poll = min(next_poll, terminate_epoch)
            delay = max(0.0, next_poll - now)
            if delay:
                seams.clock.sleep(delay)
        else:
            raise DrillError(
                "autonomous installed reaper did not close absence and billing")
        history = lease.get("history") or []
        autonomous_destroy_proven = [
            row for row in history
            if row.get("event") == "DESTROY_REQUESTED"]
        health_source = plan.reaper_state_dir / "reaper-health.json"
        health_stamp, _health_raw = _read_json_regular(
            health_source, "autonomous post-loss reaper health")
        if health_stamp != health.get("stamp"):
            raise DrillError("observed reaper stamp changed before proof capture")
        destroy_health_stamp = destroy_health["stamp"]
        destroy_health_path = (
            stage / "artifacts" / "reaper-destroy-health.json")
        _atomic_json(destroy_health_path, destroy_health_stamp)

        billing_receipt = _billing_arithmetic(
            plan, lease, str(state["provider_id"]))
        campaign_receipt, campaign_ledger_raw = _campaign_release_receipt(
            plan, state, lease)
        artifacts = stage / "artifacts"
        billing_path = artifacts / "billing-arithmetic.json"
        campaign_path = artifacts / "campaign-release.json"
        _atomic_json(billing_path, billing_receipt)
        _atomic_json(campaign_path, campaign_receipt)
        campaign_ledger_copy = artifacts / "campaign-ledger.json"
        _atomic_bytes(campaign_ledger_copy, campaign_ledger_raw)
        if (_artifact(stage, campaign_ledger_copy)["sha256"]
                != campaign_receipt["campaign_ledger_sha256"]):
            raise DrillError("campaign ledger changed during proof capture")
        if len(autonomous_destroy_proven) != 1:
            raise DrillError(
                "controller was lost but the autonomous reaper did not issue "
                "exactly one durable destroy request; no proof issued")
        if not bounded_absence_proven:
            raise DrillError(
                "exact absence was not durably observed inside the authored "
                "autonomous-reaper lag interval; no proof issued")
        deadline_observation, _deadline_observation_raw = _read_json_regular(
            deadline_observation_path, "provider deadline observations")
        if (deadline_observation.get("schema")
                != DEADLINE_OBSERVATION_SCHEMA
                or not _verify_seal(
                    deadline_observation, "record_sha256")
                or deadline_observation.get("exact_pod_id")
                    != str(state["provider_id"])):
            raise DrillError("provider deadline observations changed before proof")


        lease_copy_dir = artifacts / "lease"
        lease_copy_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lease_copy = lease_copy_dir / lease_path.name
        _atomic_json(lease_copy, lease)
        health_copy = artifacts / "reaper-health.json"
        _atomic_bytes(health_copy, _health_raw)
        result_path = artifacts / "result-bundle.tar.gz"
        verified = verify_archive(result_path)
        if (verified.get("manifest") or {}).get("job_id_full") != plan.job_hash:
            raise DrillError("published result archive job binding changed")
        create_events = [
            row for row in lease.get("history") or []
            if row.get("event") == "CREATE_RESPONSE_BOUND"]
        if len(create_events) != 1:
            raise DrillError("terminal lease lacks one exact create acknowledgement")
        create_evidence = create_events[0].get("evidence") or {}
        acknowledged = create_evidence.get("response") or {}
        if (create_evidence.get("provider_id_acknowledged")
                != str(state["provider_id"])
                or create_evidence.get("submitted_request_sha256")
                != lease["create"]["request_sha256"]):
            raise DrillError("provider acknowledgement does not bind request and exact id")
        provider_acknowledgement = {
            "pod_id": str(state["provider_id"]),
            "name": acknowledged.get("name"),
            "cost_per_hr": acknowledged.get("cost_per_hr"),
        }

        loss = _seal({
            "schema": LOSS_SCHEMA,
            "receipt_sha256": "",
            "exact_pod_id": str(state["provider_id"]),
            "exact_pod_name": plan.exact_name,
            "controller_pid": int(state["controller_pid"]),
            "controller_exit_observed": True,
            "controller_lost_at": kill_event["killed_at"],
            "kill_event_sha256": kill_event["receipt_sha256"],
            "lease_record_sha256": lease["record_sha256"],
            "provider_account_id": plan.provider_account_id,
            "result_archive_sha256": verified["archive_sha256"],
            "result_archive_bytes": verified["archive_bytes"],
            "result_transfer_receipt_sha256":
                state["result_transfer_receipt_sha256"],
            "provider_deadline_observations_sha256":
                deadline_observation["record_sha256"],
            "reaper_destroy_health_sha256":
                destroy_health_stamp["record_sha256"],
        }, "receipt_sha256")
        loss_path = artifacts / "controller-loss.json"
        _atomic_json(loss_path, loss)

        issued_epoch = float(seams.clock.time())
        proof = _seal({
            "schema": PROOF_SCHEMA,
            "proof_sha256": "",
            "issued_at": _utc(issued_epoch),
            "expires_at": _utc(issued_epoch + 7 * 86400),
            "bundle_manifest_sha256": plan.bundle_contract_sha256,
            "control_manifest_sha256": plan.control_manifest_sha256,
            "provider_account_id": plan.provider_account_id,
            "drill": {
                "kind": DRILL_KIND,
                "termination_mechanism": "autonomous-systemd-user-reaper",
                "provider_timer_trusted": False,
                "paid": True,
                "provider": "runpod",
                "provider_account_id": plan.provider_account_id,
                "exact_pod_id": str(state["provider_id"]),
                "exact_pod_name": plan.exact_name,
                "job_id_full": plan.job_hash,
                "attempt_id": plan.attempt_id,
                "remote_helper_sha256": {
                    name: digest for name, _body, digest in plan.remote_helpers
                },
                "prepared_create_sha256":
                    _sha256(lease["create"]["request"]["prepared_create"]),
                "graphql_body_sha256":
                    lease["create"]["request"]["prepared_create"][
                        "graphql_body_sha256"],
                "graphql_body_bytes":
                    lease["create"]["request"]["prepared_create"][
                        "graphql_body_bytes"],
                "producer_checkout":
                    lease["create"]["request"]["producer_checkout"],
                "create_request_sha256": lease["create"]["request_sha256"],
                "provider_acknowledgement": provider_acknowledgement,
                "ssh_host_key_proof_sha256":
                    state["ssh_host_key_proof_sha256"],
                "secure_cloud": True,
                "spot": False,
                "offer": "on-demand",
                "gpu_type_id": GPU_TYPE,
                "gpu_count": 1,
                "image_name": IMAGE,
                "volume_gb": plan.storage_gb,
                "container_disk_gb": plan.container_disk_gb,
                "min_vcpu": MIN_VCPU,
                "min_ram_gb": MIN_RAM_GB,
                "network_volume_id": None,
                "live_rate_usd_per_hour": format(plan.offer_rate, "f"),
                "terminate_after": plan.terminate_after,
                "provider_deadline_observation_until": plan.observation_until,
            },
            "artifacts": {
                "lease": _artifact(stage, lease_copy),
                "controller_loss": _artifact(stage, loss_path),
                "controller_kill_event": _artifact(stage, kill_path),
                "provider_deadline_observations": _artifact(
                    stage, deadline_observation_path),
                "reaper_destroy_health": _artifact(
                    stage, destroy_health_path),
                "reaper_health": _artifact(stage, health_copy),
                "ssh_host_key_proof": _artifact(
                    stage,
                    artifacts / "runpod-ssh-host-key-proof.json"),
                "result_archive": _artifact(stage, result_path),
                "result_transfer": _artifact(
                    stage, artifacts / "result-transfer.json"),
                "billing_arithmetic": _artifact(stage, billing_path),
                "campaign_release": _artifact(stage, campaign_path),
                "campaign_ledger": _artifact(stage, campaign_ledger_copy),
            },
        }, "proof_sha256")
        proof_path = stage / "proof.json"
        _atomic_json(proof_path, proof)
        validate_safety_proof(
            proof_path, plan.bundle_contract_sha256,
            plan.control_manifest_sha256, plan.provider_account_id,
            plan.campaign_ledger,
            now=datetime.fromtimestamp(issued_epoch, tz=timezone.utc))
        shutil.rmtree(stage / "controller-upload", ignore_errors=True)
        os.replace(str(stage), str(plan.output))
        _fsync_dir(plan.output.parent)
        return plan.output / "proof.json"
    except BaseException:
        # Durable lease/campaign state remains authoritative.  Normal deadline
        # cleanup is autonomous; the controller requests immediate cleanup only
        # for a post-create identity/rate safety failure.  Never publish partial proof.
        shutil.rmtree(stage, ignore_errors=True)
        raise


def run_drill(args: Any, con: Any, provider: Any, *,
              seams: Optional[DrillSeams] = None) -> int:
    """Narrow measure_cloud entry point: plan always; mutate only with --yes."""
    try:
        plan = plan_drill(args, provider, seams=seams)
        con.say(json.dumps(plan.public_dict(), sort_keys=True, indent=2))
        if getattr(args, "dry_run", False):
            con.say("dry-run: validated drill plan; zero mutations and zero paid calls")
            return 0
        if getattr(args, "yes", False) is not True:
            raise DrillError("paid RunPod drill requires explicit --yes")
        proof = execute_drill(plan, args, provider, seams=seams)
        con.say("RunPod controller-loss proof: %s" % proof)
        return 0
    except Exception as exc:
        con.warn("RunPod drill refused (%s): %s" % (
            type(exc).__name__, redact(str(exc))))
        return 2


__all__ = [
    "DrillError", "DrillPlan", "DrillSeams", "ForkSupervisor",
    "execute_drill", "plan_drill", "run_drill",
]
