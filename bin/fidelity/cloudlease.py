#!/usr/bin/env python3
"""Crash-safe provider leases and the provider-dispatched reaper.

A lease is an authorization and an audit trail, not a cache.  Creation intent is
written before a provider POST, every update is generation checked while the
full job hash is flocked, and terminal records are retained.  In particular,
"delete requested" and a provider status such as EXITED are never treated as
absence: only a complete provider listing which omits the exact id is evidence
that a resource is gone.

This module is intentionally Python 3.9 stdlib only.  Provider objects are
small duck types: ``list_instances()``, ``destroy(id)``, and optionally
``reconcile_billing(lease_dict)``.
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import calendar
import fcntl
import hashlib
import json
import math
import os
import secrets
import subprocess
import re
import tempfile
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .common import redact

SCHEMA = "fidelity-suite/cloud-lease.v2"
HEALTH_SCHEMA = "fidelity-suite/reaper-health.v2"
# v3: control seals the runtime snapshot only; the checkout it was copied
# from is an advisory drift probe.  A v2 manifest is refused with a reinstall
# remedy rather than silently re-verified under the new rule.
CONTROL_SCHEMA = "fidelity-suite/reaper-control.v3"
CONTROL_CLOSURE_PATHS = frozenset({
    "bin/reap_cloud_leases.py",
    "bin/fidelity/__init__.py",
    "bin/fidelity/cloudlease.py",
    "bin/fidelity/campaign.py",
    "bin/fidelity/common.py",
    "bin/fidelity/runpodapi.py",
    "bin/fidelity/jlapi.py",
    "bin/fidelity/sshbase.py",
})
DEFAULT_STATE_DIR = Path.home() / ".fidelity-cloud"
DEFAULT_LEASE_DIR = DEFAULT_STATE_DIR / "leases-v2"
ATTEMPT_BYTES = 12                    # 96 bits
# A lost create response with nothing attributable across complete listings
# for this long after its create window closed is expired to TERMINAL by the
# reaper.  RunPod answers a create synchronously; fifteen minutes is the same
# provider-lag bound the proof validator applies to destroy and absence.
LOST_CREATE_EXPIRY_SECONDS = 900
PROVIDER_DEADLINE_DRILL_MODE = "paid-controller-loss-provider-deadline"
_PROCESS_STARTED_EPOCH = time.time()
_PROCESS_INVOCATION_ID = secrets.token_hex(16)
MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS = 900
BILLING_STABILIZATION_SECONDS = 300

PREPARED = "PREPARED"
CREATING = "CREATING"
ACTIVE = "ACTIVE"
DESTROYING = "DESTROYING"
AMBIGUOUS = "AMBIGUOUS"
ABSENCE_CONFIRMED = "ABSENCE_CONFIRMED"
TERMINAL = "TERMINAL"
UNRESOLVED_STATES = frozenset((PREPARED, CREATING, ACTIVE, DESTROYING,
                               AMBIGUOUS, ABSENCE_CONFIRMED))
_ALLOWED = {
    CREATING: frozenset(
        (CREATING, ACTIVE, AMBIGUOUS, ABSENCE_CONFIRMED, TERMINAL)),
    PREPARED: frozenset((PREPARED, CREATING, TERMINAL)),
    ACTIVE: frozenset((ACTIVE, DESTROYING, AMBIGUOUS, ABSENCE_CONFIRMED)),
    DESTROYING: frozenset((DESTROYING, ABSENCE_CONFIRMED)),
    AMBIGUOUS: frozenset((AMBIGUOUS, DESTROYING)),
    ABSENCE_CONFIRMED: frozenset(
        (ABSENCE_CONFIRMED, DESTROYING, TERMINAL)),
    TERMINAL: frozenset(),
}
_IMMUTABLE_TOP = frozenset(("schema", "job_hash", "attempt_id", "create"))


class LeaseError(RuntimeError):
    """Base class for fail-closed lease errors."""


class LeaseConflict(LeaseError):
    """A job already has an unresolved attempt, or a name already existed."""


class GenerationConflict(LeaseError):
    """The caller attempted to update a stale view of a lease."""


class InvalidLease(LeaseError):
    """A lease is malformed, unsealed, or inconsistent with its filename."""

class CreateResponsePersistenceError(LeaseError):
    """A committed provider response could not be bound to its lease."""

    def __init__(self, response: Mapping[str, Any],
                 cause: BaseException) -> None:
        self.response = dict(response)
        raw_id = self.response.get(
            "id", self.response.get(
                "machine_id", self.response.get("pod_id")))
        self.provider_id = str(raw_id or "")
        super().__init__(
            "committed create response could not be durably bound: %s"
            % redact(str(cause)))



@dataclass(frozen=True)
class LeaseRef:
    path: Path
    job_hash: str
    attempt_id: str
    generation: int
    state: str


@dataclass(frozen=True)
class ReaperResult:
    ok: bool
    actions: Tuple[Dict[str, Any], ...]
    failures: Tuple[Dict[str, Any], ...]
    unresolved: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "actions": list(self.actions),
            "failures": list(self.failures),
            "unresolved": list(self.unresolved),
        }


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LeaseError("lease evidence is not canonical JSON: %s" % exc)
    return text.encode("utf-8")

def _strict_lease_loads(raw: bytes, path: Path) -> Dict[str, Any]:
    def _pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise InvalidLease(
                    "duplicate JSON key %r in lease %s" % (key, path))
            out[key] = value
        return out

    def _constant(value):
        raise InvalidLease(
            "non-finite JSON number %s in lease %s" % (value, path))

    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs,
            parse_constant=_constant)
    except InvalidLease:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidLease("cannot parse lease %s: %s" % (path, exc))
    if not isinstance(document, dict):
        raise InvalidLease("lease %s is not a JSON object" % path)
    return document


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now(now: Optional[float] = None) -> str:
    instant = time.time() if now is None else float(now)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(instant))


def utc_iso(epoch: float) -> str:
    """Return an exact whole-second UTC ISO timestamp."""
    value = float(epoch)
    if not value > 0:
        raise LeaseError("deadline epoch must be positive")
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def _validate_hash(job_hash: str) -> str:
    value = str(job_hash)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LeaseError("job_hash must be the full lowercase SHA-256 hex digest")
    return value


def _validate_attempt(attempt: str) -> str:
    value = str(attempt)
    if len(value) != ATTEMPT_BYTES * 2 or any(c not in "0123456789abcdef" for c in value):
        raise InvalidLease("attempt_id must be 96-bit lowercase hex")
    return value


def exact_resource_name(job_hash: str, attempt_id: str) -> str:
    """Provider name containing both complete collision-resistant identities."""
    return "fidcloud-%s-a%s" % (_validate_hash(job_hash), _validate_attempt(attempt_id))


def _seal(document: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(document)
    out.pop("record_sha256", None)
    out["record_sha256"] = _sha256(out)
    return out


def _safe_directory_fd(path: Path, *, create: bool) -> int:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise LeaseError("lease state directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in target.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, flags | nofollow, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise LeaseError("lease state path is not a directory")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise LeaseError("lease state directory is not owned by current user")
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise LeaseError("lease state directory must have mode 0700")
        result, fd = fd, -1
        return result
    except OSError as exc:
        raise LeaseError("unsafe lease state directory %s: %s" % (target, exc))
    finally:
        if fd >= 0:
            os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = _safe_directory_fd(path, create=False)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_temp(dir_fd: int, basename: str, data: bytes) -> str:
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | getattr(os, "O_NOFOLLOW", 0))
    for unused in range(100):
        name = ".%s.%s" % (basename, secrets.token_hex(12))
        try:
            fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
            break
        except FileExistsError:
            continue
    else:
        raise LeaseError("could not allocate atomic lease temporary file")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(name, dir_fd=dir_fd)
        except OSError:
            pass
        raise
    return name


def _atomic_replace(path: Path, document: Dict[str, Any]) -> None:
    data = _canonical_bytes(_seal(document)) + b"\n"
    dir_fd = _safe_directory_fd(path.parent, create=True)
    tmp = None
    try:
        tmp = _atomic_temp(dir_fd, path.name, data)
        os.replace(tmp, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        tmp = None
        os.fsync(dir_fd)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)


def _atomic_create(path: Path, document: Dict[str, Any]) -> None:
    """Publish a complete file without ever replacing a collision."""
    data = _canonical_bytes(_seal(document)) + b"\n"
    dir_fd = _safe_directory_fd(path.parent, create=True)
    tmp = None
    try:
        tmp = _atomic_temp(dir_fd, path.name, data)
        try:
            os.link(tmp, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd,
                    follow_symlinks=False)
        except FileExistsError:
            raise LeaseConflict("lease filename collision: %s" % path.name)
        os.unlink(tmp, dir_fd=dir_fd)
        tmp = None
        os.fsync(dir_fd)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)


def _resource_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise LeaseError("provider resource id must be an exact string or integer")
    text = str(value)
    if not text or text != text.strip():
        raise LeaseError("provider resource id must be non-empty without padding")
    return text


def _resource(resource: Any) -> Tuple[str, str, str]:
    if isinstance(resource, Mapping):
        rid = resource.get("id", resource.get("machine_id", resource.get("pod_id")))
        name = resource.get("name")
        status_value = resource.get("status", resource.get("desiredStatus"))
    else:
        rid = getattr(resource, "machine_id", getattr(resource, "id", None))
        name = getattr(resource, "name", None)
        status_value = getattr(resource, "status", None)
    return _resource_id(rid), str(name or ""), str(status_value or "")

_LEASE_TOP_KEYS = {
    "schema", "job_hash", "attempt_id", "generation", "state", "create",
    "provider_resource_ids", "history", "billing_reconciliation",
    "terminal_proof", "record_sha256",
}
_CREATE_KEYS = {
    "provider", "exact_name", "pre_create_provider_ids",
    "pre_create_network_volume_ids", "pre_create_observed_at",
    "request", "request_sha256",
    "create_deadline_epoch", "create_deadline_utc",
    "workload_deadline_epoch", "workload_deadline_utc",
    "reap_deadline_epoch", "reap_deadline_utc", "controller_pid",
    "evidence_sha256",
}
_NORMAL_RUNPOD_REQUEST_KEYS = {
    "attempt_key", "campaign_attempt_key", "campaign_ledger", "provider",
    "provider_account_id", "gpu_type", "normalized_gpu", "num_gpus",
    "secure_cloud", "storage_gb", "remote_root", "engine_root",
    "container_disk_gb", "image", "min_vcpu_count", "min_memory_gb",
    "workload_contract", "offer", "network_volume", "terminate_after", "quote",
    "pre_create_safety", "execution_contract_sha256", "grounding_bundle",
    "prepared_create",
}
_DRILL_RUNPOD_REQUEST_KEYS = {
    "drill_mode", "provider_account_id", "campaign_ledger",
    "campaign_attempt_key", "secure_cloud", "offer", "spot", "gpu_type_id",
    "gpu_count", "image_name", "volume_gb", "container_disk_gb",
    "min_vcpu", "min_ram_gb", "network_volume_id", "terminate_after",
    "provider_deadline_observation_until",
    "pre_create_safety", "prepared_create", "producer_checkout",
}
_BILLING_KEYS = {
    "reconciled", "provider", "provider_resource_ids", "billing_histories",
    "total_amount", "evidence",
}
_EVENT_TRANSITIONS = {
    "LEASE_PREPARED_NO_PROVIDER_POST": {(None, PREPARED)},
    "PROVIDER_POST_INTENT_FSYNCED": {(PREPARED, CREATING)},
    "PREPARED_CANCELLATION_INTENT": {(PREPARED, PREPARED)},
    "PREPARED_CANCELLED_NO_PROVIDER_POST": {(PREPARED, TERMINAL)},
    "CREATE_RESPONSE_BOUND": {(CREATING, ACTIVE)},
    "LOST_CREATE_RESPONSE_RECONCILED_ONE": {(CREATING, ACTIVE)},
    "RESOURCE_IDENTITY_ATTESTED": {(ACTIVE, ACTIVE)},
    "LOST_CREATE_RESPONSE_RECONCILED_MULTIPLE": {(CREATING, AMBIGUOUS)},
    "LOST_CREATE_RESPONSE_RECONCILED_AMBIGUOUS":
        {(CREATING, AMBIGUOUS)},
    "POST_CREATE_FAMILY_DELTA_AMBIGUOUS": {(ACTIVE, AMBIGUOUS)},
    "WORKLOAD_EXITED": {(ACTIVE, ACTIVE)},
    "LOST_CREATE_RESPONSE_RECONCILED_ZERO_PENDING":
        {(CREATING, CREATING)},
    "LOST_CREATE_RESPONSE_RECONCILED_ZERO_WINDOW_CLOSED_UNRESOLVED":
        {(CREATING, CREATING)},
    "PROVIDER_REJECTED_CREATE_NO_RESOURCE": {(CREATING, TERMINAL)},
    "LOST_CREATE_RESPONSE_EXPIRED_NO_RESOURCE": {(CREATING, TERMINAL)},
    "DESTROY_REQUESTED": {
        (CREATING, DESTROYING), (ACTIVE, DESTROYING),
        (DESTROYING, DESTROYING), (AMBIGUOUS, DESTROYING),
    },
    "ABSENCE_PROOF_REVOKED": {
        (ABSENCE_CONFIRMED, DESTROYING),
    },
    "EXACT_IDS_STILL_LISTED": {
        (ACTIVE, ACTIVE), (DESTROYING, DESTROYING),
        (AMBIGUOUS, AMBIGUOUS),
    },
    "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING": {
        (ACTIVE, ABSENCE_CONFIRMED), (DESTROYING, ABSENCE_CONFIRMED),
        (AMBIGUOUS, ABSENCE_CONFIRMED),
    },
    "BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN":
        {(ABSENCE_CONFIRMED, ABSENCE_CONFIRMED)},
    "BILLING_RECONCILED_TERMINAL":
        {(ABSENCE_CONFIRMED, TERMINAL)},
}


def _exact_keys(value: Any, required: Iterable[str],
                optional: Iterable[str], label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidLease("%s must be an object" % label)
    required_set, optional_set = set(required), set(optional)
    keys = set(value)
    if not required_set <= keys or not keys <= required_set | optional_set:
        raise InvalidLease(
            "%s keys differ: missing=%s unexpected=%s"
            % (label, sorted(required_set - keys),
               sorted(keys - required_set - optional_set)))
    return value


def _exact_utc_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise InvalidLease("%s must be an exact UTC string" % label)
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise InvalidLease("%s must be an exact UTC string" % label)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != value:
        raise InvalidLease("%s is not canonical UTC" % label)
    return value


def _epoch(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidLease("%s must be a finite epoch" % label)
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise InvalidLease("%s must be a positive finite epoch" % label)
    return parsed


def _sorted_ids(value: Any, label: str) -> List[str]:
    if not isinstance(value, list):
        raise InvalidLease("%s must be a sorted id list" % label)
    try:
        normalized = [_resource_id(item) for item in value]
    except LeaseError as exc:
        raise InvalidLease("%s is invalid: %s" % (label, exc))
    if normalized != sorted(set(normalized)):
        raise InvalidLease("%s must contain sorted unique exact ids" % label)
    return normalized


def _validate_pre_create_safety(value: Any) -> Dict[str, Any]:
    evidence = _exact_keys(
        value,
        ("schema", "endpoint_origin", "date_header", "server_epoch",
         "local_received_epoch", "local_minus_server_seconds",
         "checked_at_epoch", "evidence_age_seconds",
         "max_clock_delta_seconds", "max_evidence_age_seconds"),
        (), "RunPod pre-create server-time evidence")
    if (evidence["schema"] != "fidelity-suite/runpod-server-time.v1"
            or evidence["endpoint_origin"] != "https://api.runpod.io"
            or not isinstance(evidence["date_header"], str)):
        raise InvalidLease("RunPod pre-create server-time identity is invalid")
    for key in (
            "server_epoch", "local_received_epoch", "checked_at_epoch",
            "max_clock_delta_seconds", "max_evidence_age_seconds"):
        _epoch(evidence[key], "RunPod server-time " + key)
    for key in ("local_minus_server_seconds", "evidence_age_seconds"):
        value = evidence[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise InvalidLease(
                "RunPod server-time %s must be finite" % key)
    if (evidence["max_clock_delta_seconds"] > 30
            or evidence["max_evidence_age_seconds"] > 30
            or evidence["evidence_age_seconds"] < -1
            or evidence["evidence_age_seconds"]
            > evidence["max_evidence_age_seconds"]
            or abs(evidence["local_minus_server_seconds"])
            > evidence["max_clock_delta_seconds"]):
        raise InvalidLease("RunPod pre-create server-time proof is unsafe")
    return evidence

def _validate_producer_checkout(value: Any) -> Dict[str, Any]:
    checkout = _exact_keys(
        value, ("schema", "revision", "initial", "pre_post"), (),
        "producer checkout")
    if checkout["schema"] != "fidelity-suite/producer-checkout.v1":
        raise InvalidLease("producer checkout schema differs")
    revision = checkout["revision"]
    if (not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise InvalidLease("producer checkout revision is not exact")
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for label, expected_untracked in (
            ("initial", "all"), ("pre_post", "all")):
        observation = _exact_keys(
            checkout[label],
            ("untracked_files", "status_porcelain_sha256",
             "status_bytes", "clean"),
            (), "producer checkout %s" % label)
        if (observation["untracked_files"] != expected_untracked
                or observation["status_porcelain_sha256"] != empty_sha256
                or isinstance(observation["status_bytes"], bool)
                or observation["status_bytes"] != 0
                or observation["clean"] is not True):
            raise InvalidLease(
                "producer checkout %s is not proven clean" % label)
    return checkout


def _validate_prepared_create(value: Any) -> Dict[str, Any]:
    prepared = _exact_keys(
        value,
        ("schema", "request_identity", "graphql_body_sha256",
         "graphql_body_bytes", "graphql_body_base64"),
        (), "RunPod prepared create")
    if prepared["schema"] != "fidelity-suite/runpod-prepared-create.v1":
        raise InvalidLease("RunPod prepared create schema is invalid")
    identity = _exact_keys(
        prepared["request_identity"],
        ("cloud_type", "is_spot", "offer", "gpu_type_id", "gpu_count",
         "volume_gb", "container_disk_gb", "min_vcpu", "min_ram_gb",
         "name", "image_name", "terminate_after", "ports",
         "volume_mount_path", "network_volume_id", "public_key_sha256"),
        (), "RunPod prepared request identity")
    if (identity["cloud_type"] != "SECURE"
            or identity["is_spot"] is not False
            or identity["offer"] != "on-demand"
            or identity["ports"] != "22/tcp"
            or identity["volume_mount_path"] != "/workspace"
            or identity["network_volume_id"] is not None
            or not isinstance(identity["public_key_sha256"], str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", identity["public_key_sha256"])):
        raise InvalidLease("RunPod prepared request violates safe profile")
    try:
        body = base64.b64decode(
            prepared["graphql_body_base64"].encode("ascii"), validate=True)
    except (AttributeError, UnicodeError, ValueError, binascii.Error):
        raise InvalidLease("RunPod prepared GraphQL body is invalid base64")
    if (isinstance(prepared["graphql_body_bytes"], bool)
            or not isinstance(prepared["graphql_body_bytes"], int)
            or prepared["graphql_body_bytes"] != len(body)
            or hashlib.sha256(body).hexdigest()
            != prepared["graphql_body_sha256"]):
        raise InvalidLease("RunPod prepared GraphQL body identity differs")
    try:
        body_doc = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise InvalidLease("RunPod prepared GraphQL body is invalid JSON")
    if (not isinstance(body_doc, dict) or set(body_doc) != {"query"}
            or not isinstance(body_doc["query"], str)
            or "podFindAndDeployOnDemand" not in body_doc["query"]):
        raise InvalidLease("RunPod prepared GraphQL mutation is invalid")
    return prepared

def _prepared_matches_request(
        prepared: Mapping[str, Any], request: Mapping[str, Any],
        *, drill: bool) -> None:
    identity = prepared["request_identity"]
    fields = (
        (("gpu_type_id", "gpu_type_id"), ("gpu_count", "gpu_count"),
         ("volume_gb", "volume_gb"),
         ("container_disk_gb", "container_disk_gb"),
         ("min_vcpu", "min_vcpu"), ("min_ram_gb", "min_ram_gb"),
         ("image_name", "image_name"),
         ("terminate_after", "terminate_after"))
        if drill else
        (("gpu_type_id", "gpu_type"), ("gpu_count", "num_gpus"),
         ("volume_gb", "storage_gb"),
         ("container_disk_gb", "container_disk_gb"),
         ("min_vcpu", "min_vcpu_count"),
         ("min_ram_gb", "min_memory_gb"),
         ("image_name", "image"),
         ("terminate_after", "terminate_after")))
    if any(identity[identity_key] != request[request_key]
           for identity_key, request_key in fields):
        raise InvalidLease(
            "prepared create differs from immutable provider request")



def _validate_request(request: Any, provider: str) -> Dict[str, Any]:
    if not isinstance(request, dict):
        raise InvalidLease("lease request must be an object")
    if request.get("drill_mode") == PROVIDER_DEADLINE_DRILL_MODE:
        if set(request) != _DRILL_RUNPOD_REQUEST_KEYS or provider != "runpod":
            raise InvalidLease("paid drill request keys/provider differ from policy")
        _validate_pre_create_safety(request["pre_create_safety"])
        prepared = _validate_prepared_create(request["prepared_create"])
        _prepared_matches_request(prepared, request, drill=True)
        _validate_producer_checkout(request["producer_checkout"])
        if (request["secure_cloud"] is not True
                or request["offer"] != "on-demand"
                or request["spot"] is not False
                or request["network_volume_id"] is not None):
            raise InvalidLease("paid drill request violates exact safe profile")
        for key in (
                "gpu_count", "volume_gb", "container_disk_gb",
                "min_vcpu", "min_ram_gb"):
            if (isinstance(request[key], bool)
                    or not isinstance(request[key], int) or request[key] <= 0):
                raise InvalidLease("paid drill %s is not a positive integer" % key)
        for key in ("provider_account_id", "gpu_type_id", "image_name"):
            if (not isinstance(request[key], str) or not request[key]
                    or request[key] != request[key].strip()):
                raise InvalidLease("paid drill %s is not exact" % key)
        _exact_utc_string(request["terminate_after"], "drill terminate_after")
        _exact_utc_string(
            request["provider_deadline_observation_until"],
            "drill provider deadline observation")
        campaign = (
            request["campaign_ledger"], request["campaign_attempt_key"])
        if campaign != (None, None):
            _campaign_coordinates({"create": {"request": request}})
    elif request.get("campaign_ledger") is not None:
        if set(request) != _NORMAL_RUNPOD_REQUEST_KEYS or provider != "runpod":
            raise InvalidLease("paid RunPod request keys/provider differ from policy")
        _validate_pre_create_safety(request["pre_create_safety"])
        prepared = _validate_prepared_create(request["prepared_create"])
        _prepared_matches_request(prepared, request, drill=False)
        if (request["provider"] != provider
                or request["secure_cloud"] is not True
                or request["offer"] != "on-demand"
                or request["network_volume"] is not None):
            raise InvalidLease("paid RunPod request violates exact safe profile")
        _campaign_coordinates({"create": {"request": request}})
        for key in (
                "num_gpus", "storage_gb", "container_disk_gb",
                "min_vcpu_count", "min_memory_gb"):
            if (isinstance(request[key], bool)
                    or not isinstance(request[key], int) or request[key] <= 0):
                raise InvalidLease("paid RunPod %s is not a positive integer" % key)
        for key in (
                "attempt_key", "provider_account_id", "gpu_type",
                "normalized_gpu", "remote_root", "engine_root", "image"):
            if (not isinstance(request[key], str) or not request[key]
                    or request[key] != request[key].strip()):
                raise InvalidLease("paid RunPod %s is not exact" % key)
        if (not os.path.isabs(request["remote_root"])
                or not os.path.isabs(request["engine_root"])
                or not isinstance(request["workload_contract"], dict)
                or not isinstance(request["quote"], dict)):
            raise InvalidLease("paid RunPod immutable workload policy is invalid")
        if (not isinstance(request["execution_contract_sha256"], str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", request["execution_contract_sha256"])):
            raise InvalidLease(
                "paid RunPod execution contract digest is invalid")
        bundle = _exact_keys(
            request["grounding_bundle"],
            ("schema", "archive_sha256", "archive_bytes",
             "manifest_sha256"), (), "RunPod grounding bundle")
        if (bundle["schema"] != "fidelity-suite/grounding-bundle.v1"
                or isinstance(bundle["archive_bytes"], bool)
                or not isinstance(bundle["archive_bytes"], int)
                or bundle["archive_bytes"] <= 0
                or any(not isinstance(bundle[key], str)
                       or not re.fullmatch(r"[0-9a-f]{64}", bundle[key])
                       for key in ("archive_sha256", "manifest_sha256"))):
            raise InvalidLease("paid RunPod grounding bundle is invalid")
        _exact_utc_string(request["terminate_after"], "terminate_after")
    elif any(key in request for key in (
            "campaign_attempt_key", "provider_account_id", "drill_mode")):
        raise InvalidLease("partial paid request policy is forbidden")
    else:
        if set(request) not in ({"gpu"}, {"gpu", "count"}):
            raise InvalidLease("unpaid lease request keys differ from policy")
        if (not isinstance(request["gpu"], str) or not request["gpu"]
                or request["gpu"] != request["gpu"].strip()):
            raise InvalidLease("unpaid lease gpu identity is invalid")
        if "count" in request and (
                isinstance(request["count"], bool)
                or not isinstance(request["count"], int)
                or request["count"] <= 0):
            raise InvalidLease("unpaid lease count is invalid")
    return request

def _validate_billing(
        value: Any, provider: str, *,
        require_stabilized: bool = True) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    billing = _exact_keys(
        value, ("reconciled", "provider"), _BILLING_KEYS, "billing")
    if billing["reconciled"] is not True or billing["provider"] != provider:
        raise InvalidLease("billing provider/reconciliation identity is invalid")
    if "provider_resource_ids" in billing:
        _sorted_ids(billing["provider_resource_ids"],
                    "billing provider_resource_ids")
    if ("billing_histories" in billing
            and not isinstance(billing["billing_histories"], list)):
        raise InvalidLease("billing_histories must be a list")
    if "total_amount" in billing:
        amount = billing["total_amount"]
        if not isinstance(amount, str):
            raise InvalidLease("billing total_amount must be an exact decimal string")
        try:
            from decimal import Decimal, InvalidOperation
            parsed = Decimal(amount)
        except (InvalidOperation, ValueError):
            raise InvalidLease("billing total_amount is not decimal")
        if not parsed.is_finite() or parsed < 0:
            raise InvalidLease("billing total_amount is invalid")
    if provider == "runpod" and require_stabilized:
        stabilization = _exact_keys(
            billing.get("evidence"),
            ("schema", "absence_confirmed_at",
             "minimum_stabilization_seconds", "closure_sha256",
             "first_retrieval", "second_retrieval"),
            (), "RunPod billing stabilization")
        stabilization_seconds = stabilization["minimum_stabilization_seconds"]
        if (stabilization["schema"]
                != "fidelity-suite/runpod-billing-stabilization.v1"
                or isinstance(stabilization_seconds, bool)
                or not isinstance(stabilization_seconds, int)
                or stabilization_seconds < BILLING_STABILIZATION_SECONDS):
            raise InvalidLease("RunPod billing stabilization bound is invalid")
        _exact_utc_string(
            stabilization["absence_confirmed_at"],
            "RunPod billing absence time")
        retrievals = []
        for name in ("first_retrieval", "second_retrieval"):
            retrieval = _exact_keys(
                stabilization[name],
                ("schema", "retrieval_id", "retrieved_at_utc"), (),
                "RunPod billing retrieval")
            if (retrieval["schema"]
                    != "fidelity-suite/runpod-billing-retrieval.v1"
                    or not isinstance(retrieval["retrieval_id"], str)
                    or not re.fullmatch(
                        r"[0-9a-f]{24}", retrieval["retrieval_id"])):
                raise InvalidLease(
                    "RunPod billing retrieval identity is invalid")
            _exact_utc_string(
                retrieval["retrieved_at_utc"],
                "RunPod billing retrieval time")
            retrievals.append(retrieval)
        if retrievals[0]["retrieval_id"] == retrievals[1]["retrieval_id"]:
            raise InvalidLease("RunPod billing retrievals are not independent")
        closure = json.loads(
            _canonical_bytes(dict(billing)).decode("utf-8"))
        closure.pop("evidence", None)
        for history in closure.get("billing_histories") or []:
            if isinstance(history, dict):
                history.pop("retrieved_at_utc", None)
        if stabilization["closure_sha256"] != _sha256(closure):
            raise InvalidLease("RunPod billing closure seal mismatch")
        if (not isinstance(stabilization["closure_sha256"], str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", stabilization["closure_sha256"])):
            raise InvalidLease("RunPod billing closure digest is invalid")
    if not any(key in billing for key in (
            "evidence", "total_amount", "billing_histories")):
        raise InvalidLease("billing contains no provider evidence")
    return billing


def _validate_identity_attestation(
        evidence: Any, document: Mapping[str, Any]) -> Dict[str, Any]:
    attestation = _exact_keys(
        evidence,
        ("schema", "provider", "provider_id", "observed_at_utc", "clock",
         "expected", "observed", "transport_error", "checks", "failures",
         "ok", "attestation_sha256"), (), "resource identity attestation")
    if attestation["schema"] != "fidelity-suite/runpod-live-attestation.v2":
        raise InvalidLease("resource identity attestation schema differs")
    seal = attestation["attestation_sha256"]
    unsealed = dict(attestation)
    unsealed.pop("attestation_sha256")
    if (not isinstance(seal, str)
            or not secrets.compare_digest(seal, _sha256(unsealed))):
        raise InvalidLease("resource identity attestation seal mismatch")
    if (attestation["provider"] != document["create"]["provider"]
            or attestation["provider_id"]
            not in document["provider_resource_ids"]):
        raise InvalidLease("resource identity attestation targets another lease")
    _resource_id(attestation["provider_id"])
    _exact_utc_string(attestation["observed_at_utc"],
                      "resource attestation timestamp")
    clock = _exact_keys(
        attestation["clock"],
        ("controller_send_epoch", "controller_send_utc",
         "controller_receive_epoch", "controller_receive_utc",
         "round_trip_seconds", "remote_time_epoch", "remote_time_utc",
         "clock_skew_seconds", "allowed_skew_seconds", "within_bound"),
        (), "resource attestation clock")
    for field in (
            "controller_send_epoch", "controller_receive_epoch",
            "round_trip_seconds", "allowed_skew_seconds"):
        value = clock[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise InvalidLease(
                "resource attestation %s is not finite" % field)
    sent = float(clock["controller_send_epoch"])
    received = float(clock["controller_receive_epoch"])
    round_trip = float(clock["round_trip_seconds"])
    allowed = float(clock["allowed_skew_seconds"])
    _exact_utc_string(
        clock["controller_send_utc"], "attestation controller send time")
    _exact_utc_string(
        clock["controller_receive_utc"],
        "attestation controller receive time")
    if (received < sent or round_trip < 0
            or not math.isclose(
                round_trip, received - sent, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                allowed, 30.0 + round_trip,
                rel_tol=0.0, abs_tol=1e-6)
            or attestation["observed_at_utc"]
            != clock["controller_receive_utc"]):
        raise InvalidLease("resource attestation controller clock is inconsistent")
    remote_epoch = clock["remote_time_epoch"]
    remote_utc = clock["remote_time_utc"]
    skew = clock["clock_skew_seconds"]
    if remote_epoch is None or remote_utc is None or skew is None:
        if not (remote_epoch is None and remote_utc is None and skew is None
                and clock["within_bound"] is False):
            raise InvalidLease(
                "resource attestation missing remote clock is inconsistent")
    else:
        if (isinstance(remote_epoch, bool)
                or not isinstance(remote_epoch, int)
                or isinstance(skew, bool)
                or not isinstance(skew, (int, float))
                or not math.isfinite(float(skew))):
            raise InvalidLease("resource attestation remote clock is invalid")
        remote_text = _exact_utc_string(
            remote_utc, "attestation remote time")
        remote_parsed = calendar.timegm(time.strptime(
            remote_text, "%Y-%m-%dT%H:%M:%SZ"))
        expected_skew = abs(remote_epoch - (sent + round_trip / 2.0))
        within = expected_skew <= allowed
        if (remote_parsed != remote_epoch
                or not math.isclose(
                    float(skew), expected_skew,
                    rel_tol=0.0, abs_tol=1e-6)
                or clock["within_bound"] is not within):
            raise InvalidLease(
                "resource attestation remote clock proof is inconsistent")
    expected_resources = _exact_keys(
        attestation["expected"],
        ("expected_vram_bytes", "min_vcpu", "min_ram_gb", "volume_gb",
         "container_disk_gb", "workspace_available_bytes_minimum",
         "container_available_bytes_minimum", "gpu_model"),
        (), "attestation expectations")
    for field in (
            "expected_vram_bytes", "min_vcpu", "min_ram_gb", "volume_gb",
            "container_disk_gb", "workspace_available_bytes_minimum",
            "container_available_bytes_minimum"):
        value = expected_resources[field]
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0):
            raise InvalidLease(
                "resource attestation %s is not a positive integer" % field)
    if (not isinstance(expected_resources["gpu_model"], str)
            or not expected_resources["gpu_model"]):
        raise InvalidLease("resource attestation gpu_model is invalid")
    if (attestation["observed"] is not None
            and not isinstance(attestation["observed"], dict)):
        raise InvalidLease("resource attestation observation is invalid")
    if (isinstance(attestation["observed"], dict)
            and (attestation["observed"].get("remote_time_epoch")
                 != remote_epoch
                 or attestation["observed"].get("remote_time_utc")
                 != remote_utc)):
        raise InvalidLease(
            "resource attestation observed clock differs from proof")
    if (attestation["transport_error"] is not None
            and not isinstance(attestation["transport_error"], str)):
        raise InvalidLease("resource attestation transport error is invalid")
    checks = attestation["checks"]
    failures = attestation["failures"]
    if (not isinstance(checks, dict)
            or any(not isinstance(key, str) or not isinstance(value, bool)
                   for key, value in checks.items())
            or not isinstance(failures, list)
            or failures != sorted(set(failures))
            or any(not isinstance(item, str) or not item for item in failures)):
        raise InvalidLease("resource attestation checks/failures are invalid")
    if checks.get("remote_clock") is not clock["within_bound"]:
        raise InvalidLease(
            "resource attestation remote clock check is inconsistent")
    expected_ok = bool(
        attestation["transport_error"] is None
        and not failures and checks and all(checks.values()))
    if attestation["ok"] is not expected_ok:
        raise InvalidLease("resource attestation verdict is inconsistent")
    return attestation




def _validate_event_evidence(event: str, evidence: Any,
                             document: Mapping[str, Any]) -> None:
    evidence = _exact_keys(evidence, (), (), "history evidence") if (
        event == "LEASE_PREPARED_NO_PROVIDER_POST") else evidence
    if event == "LEASE_PREPARED_NO_PROVIDER_POST":
        return
    if event == "POST_CREATE_FAMILY_DELTA_AMBIGUOUS":
        evidence = _exact_keys(
            evidence,
            ("complete_listing", "listed_resource_count",
             "listed_network_volume_count", "intended_provider_id",
             "intended_present", "intended_name_matches_exact",
             "new_pod_ids", "new_exact_name_ids",
             "wrong_name_new_pod_ids", "authorized_sibling_pod_ids",
             "unattributable_wrong_name_pod_ids",
             "new_network_volume_ids"),
            (), "post-create family delta")
        if (evidence["complete_listing"] is not True
                or any(isinstance(evidence[key], bool)
                       or not isinstance(evidence[key], int)
                       or evidence[key] < 0
                       for key in (
                           "listed_resource_count",
                           "listed_network_volume_count"))
                or not isinstance(evidence["intended_present"], bool)
                or not isinstance(
                    evidence["intended_name_matches_exact"], bool)):
            raise InvalidLease("post-create family listing is invalid")
        intended = _resource_id(evidence["intended_provider_id"])
        new_pods = _sorted_ids(
            evidence["new_pod_ids"], "post-create new pod ids")
        exact_ids = _sorted_ids(
            evidence["new_exact_name_ids"],
            "post-create exact-name pod ids")
        wrong_ids = _sorted_ids(
            evidence["wrong_name_new_pod_ids"],
            "post-create wrong-name pod ids")
        volumes = _sorted_ids(
            evidence["new_network_volume_ids"],
            "post-create network-volume ids")
        authorized = _sorted_ids(
            evidence["authorized_sibling_pod_ids"],
            "post-create authorized sibling ids")
        blockers = _sorted_ids(
            evidence["unattributable_wrong_name_pod_ids"],
            "post-create unattributable pod ids")
        expected_new = set(exact_ids) | set(wrong_ids)
        if evidence["intended_present"]:
            expected_new.add(intended)
        candidates = sorted(set(exact_ids) | {intended})
        if (set(exact_ids) & set(wrong_ids)
                or intended in set(wrong_ids)
                or set(new_pods) != expected_new
                or evidence["intended_present"] != (intended in new_pods)
                or not set(authorized).issubset(wrong_ids)
                or set(blockers) != set(wrong_ids) - set(authorized)
                or not (set(exact_ids) - {intended} or blockers or volumes
                        or evidence["intended_present"] is False
                        or evidence["intended_name_matches_exact"] is False)
                or candidates != document["provider_resource_ids"]):
            raise InvalidLease("post-create family delta proof is inconsistent")
        return
    if event == "PROVIDER_POST_INTENT_FSYNCED":
        evidence = _exact_keys(
            evidence, ("submitted_request_sha256", "exact_name"), (),
            "POST intent evidence")
        if (evidence["submitted_request_sha256"]
                != document["create"]["request_sha256"]
                or evidence["exact_name"] != document["create"]["exact_name"]):
            raise InvalidLease("POST intent differs from immutable create")
        return
    if event == "PREPARED_CANCELLATION_INTENT":
        evidence = _exact_keys(evidence, ("reason",), (),
                               "prepared cancellation intent")
        if not isinstance(evidence["reason"], str) or not evidence["reason"]:
            raise InvalidLease("prepared cancellation reason is invalid")
        return
    if event == "WORKLOAD_EXITED":
        evidence = _exact_keys(
            evidence, ("failed_stage", "run_error", "stages_done"), (),
            "workload exit")
        if (evidence["failed_stage"] is not None
                and not isinstance(evidence["failed_stage"], str)):
            raise InvalidLease("failed_stage must be a string or null")
        if (evidence["run_error"] is not None
                and not isinstance(evidence["run_error"], str)):
            raise InvalidLease("run_error must be a string or null")
        if (not isinstance(evidence["stages_done"], list)
                or any(not isinstance(item, str)
                       for item in evidence["stages_done"])):
            raise InvalidLease("stages_done must be a string list")
        return
    if event == "RESOURCE_IDENTITY_ATTESTED":
        _validate_identity_attestation(evidence, document)
        return
    if event == "PREPARED_CANCELLED_NO_PROVIDER_POST":
        evidence = _exact_keys(
            evidence, ("reason", "no_provider_post"),
            ("campaign_projection",), "prepared cancellation")
        if evidence["no_provider_post"] is not True:
            raise InvalidLease("prepared cancellation lacks no-POST proof")
        return
    if event == "CREATE_RESPONSE_BOUND":
        evidence = _exact_keys(
            evidence, ("provider_id_acknowledged", "submitted_request_sha256",
                       "response", "identity_validation_pending"), (),
            "create response")
        _resource_id(evidence["provider_id_acknowledged"])
        if (evidence["submitted_request_sha256"]
                != document["create"]["request_sha256"]
                or evidence["identity_validation_pending"] is not True):
            raise InvalidLease("create response differs from immutable request")
        response = _exact_keys(
            evidence["response"],
            ("id", "machine_id", "pod_id", "name", "cost_per_hr",
             "name_matches_exact"), (), "provider create response")
        if not isinstance(response["name_matches_exact"], bool):
            raise InvalidLease("create response name match must be boolean")
        if any(value is not None and not isinstance(value, str)
               for key, value in response.items()
               if key != "name_matches_exact"):
            raise InvalidLease("create response fields must be strings or null")
        if evidence["provider_id_acknowledged"] not in document[
                "provider_resource_ids"]:
            raise InvalidLease("create response id differs from lease provider ids")
        observed_name = response["name"] or ""
        if response["name_matches_exact"] != (
                observed_name == document["create"]["exact_name"]):
            raise InvalidLease("create response name comparison is inconsistent")
        return
    if (event.startswith("LOST_CREATE_RESPONSE_RECONCILED_")
            or event in ("PROVIDER_REJECTED_CREATE_NO_RESOURCE",
                         "LOST_CREATE_RESPONSE_EXPIRED_NO_RESOURCE")):
        evidence = _exact_keys(
            evidence,
            ("complete_listing", "listed_resource_count",
             "listed_network_volume_count", "exact_name",
             "new_exact_name_ids", "new_pod_ids",
             "wrong_name_new_pod_ids", "authorized_sibling_pod_ids",
             "unattributable_wrong_name_pod_ids",
             "new_network_volume_ids", "response_provider_id",
             "create_window_closed"),
            ("response_error_redacted", "provider_rejection_codes",
             "expired_after_seconds"),
            "response-loss evidence")
        if event == "LOST_CREATE_RESPONSE_EXPIRED_NO_RESOURCE":
            expired = evidence.get("expired_after_seconds")
            if (isinstance(expired, bool) or not isinstance(expired, int)
                    or expired < LOST_CREATE_EXPIRY_SECONDS
                    or evidence["create_window_closed"] is not True
                    or evidence["new_pod_ids"]
                    or evidence["new_network_volume_ids"]
                    or evidence["response_provider_id"] is not None):
                raise InvalidLease(
                    "lost-create expiry must follow a closed window with "
                    "nothing attributable")
        if event == "PROVIDER_REJECTED_CREATE_NO_RESOURCE":
            codes = evidence.get("provider_rejection_codes")
            if (not isinstance(codes, list) or not codes
                    or any(not isinstance(code, str) or not code
                           for code in codes)
                    or codes != sorted(set(codes))
                    or evidence["new_pod_ids"]
                    or evidence["new_network_volume_ids"]
                    or evidence["response_provider_id"] is not None):
                raise InvalidLease(
                    "provider create refusal must name its codes and leave "
                    "nothing attributable")
        if (evidence["complete_listing"] is not True
                or any(isinstance(evidence[key], bool)
                       or not isinstance(evidence[key], int)
                       or evidence[key] < 0
                       for key in (
                           "listed_resource_count",
                           "listed_network_volume_count"))
                or evidence["exact_name"] != document["create"]["exact_name"]
                or not isinstance(evidence["create_window_closed"], bool)):
            raise InvalidLease("response-loss listing evidence is invalid")
        if ("response_error_redacted" in evidence
                and (not isinstance(evidence["response_error_redacted"], str)
                     or not evidence["response_error_redacted"])):
            raise InvalidLease("response-loss error evidence is invalid")
        exact_ids = _sorted_ids(
            evidence["new_exact_name_ids"], "response-loss exact-name ids")
        pod_ids = _sorted_ids(
            evidence["new_pod_ids"], "response-loss new pod ids")
        wrong_ids = _sorted_ids(
            evidence["wrong_name_new_pod_ids"],
            "response-loss wrong-name pod ids")
        authorized = _sorted_ids(
            evidence["authorized_sibling_pod_ids"],
            "response-loss authorized sibling ids")
        blockers = _sorted_ids(
            evidence["unattributable_wrong_name_pod_ids"],
            "response-loss unattributable pod ids")
        volume_ids = _sorted_ids(
            evidence["new_network_volume_ids"],
            "response-loss new network-volume ids")
        response_id = evidence["response_provider_id"]
        if response_id is not None:
            response_id = _resource_id(response_id)
        expected_blockers = set(wrong_ids) - set(authorized) - (
            {response_id} if response_id is not None else set())
        if (set(exact_ids) | set(wrong_ids) != set(pod_ids)
                or set(exact_ids) & set(wrong_ids)
                or not set(authorized).issubset(wrong_ids)
                or set(blockers) != expected_blockers):
            raise InvalidLease("response-loss pod family partition is invalid")
        candidate_ids = sorted(set(exact_ids) | (
            {response_id} if response_id is not None else set()))
        safe_one = (
            len(candidate_ids) == 1 and exact_ids == candidate_ids
            and not blockers and not volume_ids)
        ambiguous = bool(candidate_ids or blockers or volume_ids)
        if ((event == "LOST_CREATE_RESPONSE_RECONCILED_ONE"
             and not safe_one)
                or (event in (
                    "LOST_CREATE_RESPONSE_RECONCILED_MULTIPLE",
                    "LOST_CREATE_RESPONSE_RECONCILED_AMBIGUOUS")
                    and (safe_one or not ambiguous))
                or ("_ZERO_" in event and ambiguous)):
            raise InvalidLease(
                "response-loss event conflicts with family-scoped delta")
        if candidate_ids and candidate_ids != document["provider_resource_ids"]:
            raise InvalidLease(
                "response-loss candidate ids differ from lease provider ids")
        return
    if event == "DESTROY_REQUESTED":
        evidence = _exact_keys(
            evidence, ("reason",),
            ("provider_id", "provider_ids", "ambiguous_create",
             "listed_statuses"), "destroy request")
        if not isinstance(evidence["reason"], str) or not evidence["reason"]:
            raise InvalidLease("destroy request reason is invalid")
        if ("ambiguous_create" in evidence
                and evidence["ambiguous_create"] is not True):
            raise InvalidLease("destroy ambiguity marker must be true")
        if "provider_id" in evidence:
            _resource_id(evidence["provider_id"])
        if "provider_ids" in evidence:
            _sorted_ids(evidence["provider_ids"], "destroy provider ids")
        if ("provider_id" in evidence
                and evidence["provider_id"]
                not in document["provider_resource_ids"]):
            raise InvalidLease("destroy provider_id differs from lease ids")
        if ("provider_ids" in evidence
                and not set(evidence["provider_ids"])
                <= set(document["provider_resource_ids"])):
            raise InvalidLease("destroy provider_ids differ from lease ids")
        if "listed_statuses" in evidence:
            statuses = evidence["listed_statuses"]
            if (not isinstance(statuses, dict)
                    or any(_resource_id(key)
                           not in document["provider_resource_ids"]
                           or not isinstance(value, str)
                           for key, value in statuses.items())):
                raise InvalidLease("destroy listed statuses are invalid")
        return
    if event in (
            "EXACT_IDS_STILL_LISTED",
            "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING",
            "ABSENCE_PROOF_REVOKED"):
        required = (
            "complete_listing", "listed_resource_count",
            "target_provider_ids", "still_present_ids")
        optional = ("listed_statuses", "authoritative_inventory")
        if event == "ABSENCE_PROOF_REVOKED":
            required += ("revoked_absence_sha256",)
        evidence = _exact_keys(
            evidence, required, optional, "provider absence evidence")
        if (evidence["complete_listing"] is not True
                or isinstance(evidence["listed_resource_count"], bool)
                or not isinstance(evidence["listed_resource_count"], int)
                or evidence["listed_resource_count"] < 0):
            raise InvalidLease("provider absence listing is invalid")
        targets = _sorted_ids(
            evidence["target_provider_ids"], "absence target ids")
        present = _sorted_ids(
            evidence["still_present_ids"], "absence present ids")
        if "listed_statuses" in evidence:
            statuses = evidence["listed_statuses"]
            if (not isinstance(statuses, dict)
                    or any(_resource_id(key) not in targets
                           or not isinstance(value, str)
                           for key, value in statuses.items())):
                raise InvalidLease("absence listed statuses are invalid")
        if targets != document["provider_resource_ids"]:
            raise InvalidLease("absence targets differ from lease provider ids")
        if (event in ("EXACT_IDS_STILL_LISTED", "ABSENCE_PROOF_REVOKED")
                and not present):
            raise InvalidLease("still-listed event has no present ids")
        if (event == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING" and present):
            raise InvalidLease("absence event still contains present ids")
        if ("listed_statuses" in evidence
                and set(evidence["listed_statuses"]) != set(present)):
            raise InvalidLease("listed statuses differ from present ids")
        authoritative = evidence.get("authoritative_inventory")
        if authoritative is not None:
            authoritative = _exact_keys(
                authoritative,
                ("schema", "observed_at_utc",
                 "provider_account_id_sha256", "graphql_ids",
                 "rest_pod_ids", "inventory_sha256"),
                (), "authoritative RunPod inventory")
            if (authoritative["schema"]
                    != "fidelity-suite/runpod-absence-inventory.v1"):
                raise InvalidLease(
                    "authoritative RunPod inventory schema differs")
            _exact_utc_string(
                authoritative["observed_at_utc"],
                "authoritative RunPod inventory time")
            account_digest = authoritative["provider_account_id_sha256"]
            if (account_digest is not None
                    and (not isinstance(account_digest, str)
                         or not re.fullmatch(r"[0-9a-f]{64}", account_digest))):
                raise InvalidLease(
                    "authoritative RunPod inventory account digest is invalid")
            graphql_ids = _sorted_ids(
                authoritative["graphql_ids"],
                "authoritative GraphQL ids")
            rest_ids = _sorted_ids(
                authoritative["rest_pod_ids"],
                "authoritative REST pod ids")
            union = set(graphql_ids) | set(rest_ids)
            if (evidence["listed_resource_count"] != len(union)
                    or present != sorted(set(targets) & union)
                    or not isinstance(authoritative["inventory_sha256"], str)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        authoritative["inventory_sha256"])):
                raise InvalidLease(
                    "authoritative RunPod inventory proof is inconsistent")
        if event == "ABSENCE_PROOF_REVOKED":
            digest = evidence["revoked_absence_sha256"]
            if (not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise InvalidLease("revoked absence digest is invalid")
        return
    if event in (
            "BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN",
            "BILLING_RECONCILED_TERMINAL"):
        if (event == "BILLING_RECONCILED_TERMINAL"
                and evidence != document["billing_reconciliation"]):
            raise InvalidLease("terminal billing differs from canonical billing")
        _validate_billing(evidence, document["create"]["provider"])
        return
    raise InvalidLease("unknown lease history event %r" % event)


def _validate_lease_schema(document: Dict[str, Any]) -> None:
    _exact_keys(document, _LEASE_TOP_KEYS, (), "lease")
    if document["schema"] != SCHEMA:
        raise InvalidLease("unsupported lease schema")
    create = _exact_keys(document["create"], _CREATE_KEYS, (), "create")
    provider = create["provider"]
    if not isinstance(provider, str) or not provider or provider != provider.strip():
        raise InvalidLease("create provider must be an exact non-empty string")
    if not isinstance(create["exact_name"], str):
        raise InvalidLease("create exact_name must be a string")
    _sorted_ids(create["pre_create_provider_ids"], "pre-create ids")
    _sorted_ids(
        create["pre_create_network_volume_ids"],
        "pre-create network-volume ids")
    _exact_utc_string(create["pre_create_observed_at"], "pre-create observation")
    request = _validate_request(create["request"], provider)
    if create["request_sha256"] != _sha256(request):
        raise InvalidLease("lease request hash mismatch")
    if "prepared_create" in request:
        prepared_identity = request["prepared_create"]["request_identity"]
        if (prepared_identity["name"] != create["exact_name"]
                or prepared_identity["terminate_after"]
                != request["terminate_after"]):
            raise InvalidLease(
                "prepared create identity differs from immutable lease")
    create_deadline = _epoch(
        create["create_deadline_epoch"], "create deadline")
    workload_deadline = _epoch(
        create["workload_deadline_epoch"], "workload deadline")
    reap_deadline = _epoch(
        create["reap_deadline_epoch"], "reap deadline")
    if workload_deadline < create_deadline:
        raise InvalidLease("workload deadline precedes create deadline")
    if reap_deadline < workload_deadline:
        raise InvalidLease("reap deadline precedes workload deadline")
    if "terminate_after" in request:
        terminate_epoch = _exact_utc_epoch(
            request["terminate_after"], "RunPod terminate_after")
        if (reap_deadline != terminate_epoch
                or reap_deadline <= workload_deadline):
            raise InvalidLease(
                "RunPod reap deadline must exactly equal terminate_after "
                "and follow workload deadline")
    if (create["create_deadline_utc"] != utc_iso(create_deadline)
            or create["workload_deadline_utc"] != utc_iso(workload_deadline)
            or create["reap_deadline_utc"] != utc_iso(reap_deadline)):
        raise InvalidLease("lease deadline UTC fields differ from epochs")
    if (isinstance(create["controller_pid"], bool)
            or not isinstance(create["controller_pid"], int)
            or create["controller_pid"] <= 0):
        raise InvalidLease("controller_pid is invalid")
    if create["evidence_sha256"] != _sha256({
            key: value for key, value in create.items()
            if key != "evidence_sha256"}):
        raise InvalidLease("immutable create evidence seal mismatch")
    ids = _sorted_ids(document["provider_resource_ids"],
                      "provider_resource_ids")
    if set(ids) & set(create["pre_create_provider_ids"]):
        raise InvalidLease("lease binds a pre-existing provider id")
    generation = document["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise InvalidLease("lease generation must be an integer")
    history = document["history"]
    if not isinstance(history, list) or len(history) != generation + 1:
        raise InvalidLease("history length differs from generation")
    prior_state = None
    for index, row in enumerate(history):
        row = _exact_keys(
            row, ("generation", "at", "from", "to", "event", "evidence"),
            (), "history row")
        if row["generation"] != index or row["from"] != prior_state:
            raise InvalidLease("history generation/state chain is inconsistent")
        if row["to"] not in _ALLOWED:
            raise InvalidLease("history contains unknown state")
        if index == 0:
            if row["to"] != PREPARED:
                raise InvalidLease("lease history must begin PREPARED")
        elif row["to"] not in _ALLOWED[prior_state]:
            raise InvalidLease("history contains an invalid state transition")
        _exact_utc_string(row["at"], "history timestamp")
        if not isinstance(row["event"], str):
            raise InvalidLease("history event must be a string")
        if ((row["from"], row["to"])
                not in _EVENT_TRANSITIONS.get(row["event"], set())):
            raise InvalidLease(
                "history event conflicts with its state transition")
        _validate_event_evidence(row["event"], row["evidence"], document)
        prior_state = row["to"]
    if prior_state != document["state"]:
        raise InvalidLease("final history state differs from lease state")
    if document["state"] in (PREPARED, CREATING) and (
            ids or document["terminal_proof"] is not None
            or document["billing_reconciliation"] is not None):
        raise InvalidLease("pre-resource lease carries cleanup evidence")
    if document["state"] == ACTIVE and len(ids) != 1:
        raise InvalidLease("ACTIVE lease must bind exactly one provider id")
    if document["state"] in (DESTROYING, ABSENCE_CONFIRMED) and not ids:
        raise InvalidLease("cleanup lease lacks exact provider ids")
    if document["state"] == AMBIGUOUS and not (
            ids or any((document.get("terminal_proof") or {}).get(
                "ambiguous_create", {}).get(key)
                for key in ("new_network_volume_ids",
                            "unattributable_wrong_name_pod_ids"))):
        raise InvalidLease(
            "AMBIGUOUS lease lacks attributable ids or blocking resources")
    if (document["billing_reconciliation"] is not None
            and document["state"] not in (ABSENCE_CONFIRMED, TERMINAL)):
        raise InvalidLease("billing appears before provider absence")
    billing = _validate_billing(document["billing_reconciliation"], provider)
    if ("provider_resource_ids" in (billing or {})
            and billing["provider_resource_ids"] != ids):
        raise InvalidLease("billing provider ids differ from lease provider ids")
    terminal = document["terminal_proof"]
    if terminal is not None:
        terminal = _exact_keys(
            terminal, (), ("prepared_cancellation", "ambiguous_create",
                           "provider_rejected_create", "lost_create_expired",
                           "provider_absence", "billing_reconciliation",
                           "closed_at"), "terminal proof")
        if not terminal:
            raise InvalidLease("terminal proof cannot be empty")
        if "billing_reconciliation" in terminal:
            if terminal["billing_reconciliation"] != billing:
                raise InvalidLease("terminal billing differs from canonical billing")
            _exact_utc_string(terminal.get("closed_at"), "terminal closed_at")
        elif "closed_at" in terminal:
            raise InvalidLease("terminal closed_at lacks billing proof")
        if "ambiguous_create" in terminal:
            ambiguous_event = next(
                (item for item in reversed(history)
                 if item["event"] in (
                     "LOST_CREATE_RESPONSE_RECONCILED_MULTIPLE",
                     "LOST_CREATE_RESPONSE_RECONCILED_AMBIGUOUS",
                     "POST_CREATE_FAMILY_DELTA_AMBIGUOUS")), None)
            if (ambiguous_event is None
                    or terminal["ambiguous_create"]
                    != ambiguous_event["evidence"]):
                raise InvalidLease("ambiguous terminal proof is inconsistent")
        if "provider_absence" in terminal:
            absence_event = next(
                (item for item in reversed(history)
                 if item["event"]
                 == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"), None)
            if (absence_event is None
                    or terminal["provider_absence"]
                    != absence_event["evidence"]):
                raise InvalidLease("provider absence proof is inconsistent")
    if document["state"] == TERMINAL:
        if terminal is None:
            raise InvalidLease("terminal lease lacks terminal proof")
        if "prepared_cancellation" in terminal:
            if (set(terminal) != {"prepared_cancellation"}
                    or terminal["prepared_cancellation"]
                    != history[-1]["evidence"]):
                raise InvalidLease("prepared terminal proof is inconsistent")
            if ids or billing is not None:
                raise InvalidLease(
                    "prepared terminal lease carries provider cleanup evidence")
        elif "provider_rejected_create" in terminal:
            # The provider refused the POST: no id was ever accepted, so
            # there is nothing to delete and nothing to be billed for.
            if (set(terminal) != {"provider_rejected_create"}
                    or terminal["provider_rejected_create"]
                    != history[-1]["evidence"]):
                raise InvalidLease(
                    "provider-refusal terminal proof is inconsistent")
            if ids or billing is not None:
                raise InvalidLease(
                    "refused-create terminal lease carries cleanup evidence")
        elif "lost_create_expired" in terminal:
            # The POST's response was lost and nothing ever appeared for it
            # across complete listings long after the window closed.
            if (set(terminal) != {"lost_create_expired"}
                    or terminal["lost_create_expired"]
                    != history[-1]["evidence"]):
                raise InvalidLease(
                    "lost-create expiry terminal proof is inconsistent")
            if ids or billing is not None:
                raise InvalidLease(
                    "expired-create terminal lease carries cleanup evidence")
        else:
            if set(terminal) not in (
                    {"provider_absence", "billing_reconciliation", "closed_at"},
                    {"ambiguous_create", "provider_absence",
                     "billing_reconciliation", "closed_at"}):
                raise InvalidLease("terminal cleanup proof keys are incomplete")
            if not ids or billing is None:
                raise InvalidLease(
                    "cleanup terminal lease lacks provider ids or billing")
    elif document["state"] == ABSENCE_CONFIRMED:
        if (terminal is None or "provider_absence" not in terminal
                or set(terminal) - {"ambiguous_create"}
                != {"provider_absence"}):
            raise InvalidLease("absence-confirmed proof keys are inconsistent")
    elif document["state"] == AMBIGUOUS:
        if terminal is None or set(terminal) != {"ambiguous_create"}:
            raise InvalidLease("AMBIGUOUS lease lacks exact ambiguity proof")
    elif document["state"] == DESTROYING and terminal is not None:
        if set(terminal) != {"ambiguous_create"}:
            raise InvalidLease("DESTROYING lease carries conflicting proof")
    elif terminal is not None:
        raise InvalidLease("nonterminal lease carries terminal proof")

def _exact_utc_epoch(value: Any, field: str) -> int:
    text = str(value)
    try:
        parsed = time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise LeaseError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != text:
        raise LeaseError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    return calendar.timegm(parsed)


def _provider_deadline_observation_epoch(
        provider: str, request: Mapping[str, Any],
        workload_deadline_epoch: Optional[float] = None) -> Optional[int]:
    mode = request.get("drill_mode")
    observation = request.get("provider_deadline_observation_until")
    if mode is None and observation is None:
        return None
    if mode != PROVIDER_DEADLINE_DRILL_MODE:
        raise LeaseError("provider deadline observation requires exact drill_mode")
    if str(provider).strip().lower() != "runpod":
        raise LeaseError("provider deadline drill is supported only for RunPod")
    if request.get("secure_cloud") is not True:
        raise LeaseError("provider deadline drill requires secure_cloud exactly true")
    if request.get("offer") != "on-demand":
        raise LeaseError("provider deadline drill requires offer exactly on-demand")
    termination = _exact_utc_epoch(
        request.get("terminate_after"), "request terminate_after")
    observed_until = _exact_utc_epoch(
        observation, "request provider_deadline_observation_until")
    lag = observed_until - termination
    if lag != MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS:
        raise LeaseError(
            "provider deadline observation lag must be exactly %d seconds"
            % MAX_PROVIDER_DEADLINE_OBSERVATION_LAG_SECONDS)
    if (workload_deadline_epoch is not None
            and termination <= float(workload_deadline_epoch)):
        raise LeaseError(
            "provider deadline drill terminate_after must follow workload deadline")
    return observed_until


class LeaseStore:
    """A directory of immutable-evidence, generation-checked lease attempts."""

    def __init__(self, root: Path = DEFAULT_LEASE_DIR,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = Path(os.path.abspath(os.path.expanduser(str(root))))
        self.clock = clock

    def _path(self, job_hash: str, attempt_id: str) -> Path:
        return self.root / ("%s.%s.json" % (_validate_hash(job_hash),
                                             _validate_attempt(attempt_id)))

    def _lock_path(self, job_hash: str) -> Path:
        return self.root / ("%s.lock" % _validate_hash(job_hash))

    @contextlib.contextmanager
    def _exclusive_lock(self, name: str, label: str) -> Iterator[None]:
        dir_fd = _safe_directory_fd(self.root, create=True)
        fd = None
        try:
            try:
                fd = os.open(
                    name,
                    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                    0o600, dir_fd=dir_fd)
            except OSError as exc:
                raise LeaseError(
                    "%s cannot be opened as an owned regular file: %s"
                    % (label, exc)) from exc
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
                raise LeaseError(
                    "%s must be owner-owned regular mode 0600" % label)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            os.close(dir_fd)

    @contextlib.contextmanager
    def job_lock(self, job_hash: str) -> Iterator[None]:
        name = "%s.lock" % _validate_hash(job_hash)
        with self._exclusive_lock(name, "lease lock"):
            yield

    @contextlib.contextmanager
    def create_submission_lock(self, ref: LeaseRef) -> Iterator[None]:
        name = "%s.submit.lock" % _validate_hash(ref.job_hash)
        with self._exclusive_lock(name, "create submission lock"):
            yield
    @contextlib.contextmanager
    def paid_admission_lock(self) -> Iterator[None]:
        with self._exclusive_lock(
                ".runpod-paid-admission.lock", "paid admission lock"):
            yield


    def _read_unlocked(self, path: Path) -> Dict[str, Any]:
        path = Path(path)
        if path.parent != self.root:
            raise InvalidLease("lease path is outside canonical state directory")
        dir_fd = _safe_directory_fd(self.root, create=False)
        fd = None
        try:
            fd = os.open(
                path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=dir_fd)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
                raise InvalidLease(
                    "lease file must be owner-owned regular mode 0600")
            with os.fdopen(fd, "rb") as fh:
                fd = None
                raw = fh.read()
        except InvalidLease:
            raise
        except OSError as exc:
            raise InvalidLease("cannot read lease %s: %s" % (path, exc))
        finally:
            if fd is not None:
                os.close(fd)
            os.close(dir_fd)
        document = _strict_lease_loads(raw, path)
        if not isinstance(document, dict) or document.get("schema") != SCHEMA:
            raise InvalidLease("unsupported lease schema in %s" % path)
        seal = document.get("record_sha256")
        unsealed = dict(document)
        unsealed.pop("record_sha256", None)
        if not isinstance(seal, str) or not secrets.compare_digest(seal, _sha256(unsealed)):
            raise InvalidLease("lease self-seal mismatch in %s" % path)
        _validate_lease_schema(document)
        job_hash = _validate_hash(document.get("job_hash", ""))
        attempt = _validate_attempt(document.get("attempt_id", ""))
        if path.name != "%s.%s.json" % (job_hash, attempt):
            raise InvalidLease("lease identity does not match filename %s" % path.name)
        create = document.get("create")
        if not isinstance(create, dict) or create.get("exact_name") != exact_resource_name(job_hash, attempt):
            raise InvalidLease("lease has invalid immutable create evidence")
        if create.get("evidence_sha256") != _sha256({k: v for k, v in create.items()
                                                     if k != "evidence_sha256"}):
            raise InvalidLease("immutable create evidence seal mismatch")
        request = create.get("request")
        if (not isinstance(request, dict)
                or create.get("request_sha256") != _sha256(request)):
            raise InvalidLease("lease request hash mismatch")
        try:
            _provider_deadline_observation_epoch(
                create.get("provider"), request,
                create.get("workload_deadline_epoch"))
        except LeaseError as exc:
            raise InvalidLease("invalid lease drill policy: %s" % exc)
        if document.get("state") not in _ALLOWED:
            raise InvalidLease("unknown lease state %r" % document.get("state"))
        try:
            _campaign_coordinates(document)
            _expected_provider_account(document)
        except LeaseError as exc:
            raise InvalidLease("invalid lease campaign reference: %s" % exc)
        generation = document.get("generation")
        if not isinstance(generation, int) or generation < 0:
            raise InvalidLease("invalid lease generation")
        return document

    def read(self, ref_or_path: Any) -> Dict[str, Any]:
        path = ref_or_path.path if isinstance(ref_or_path, LeaseRef) else Path(ref_or_path)
        return self._read_unlocked(path)

    def _read_current(self, ref: LeaseRef) -> Dict[str, Any]:
        """Read a mutation source while preserving generation semantics."""
        document = self.read(ref)
        expected_path = self._path(ref.job_hash, ref.attempt_id)
        if (ref.path != expected_path
                or document["job_hash"] != ref.job_hash
                or document["attempt_id"] != ref.attempt_id):
            raise InvalidLease("lease reference identity differs from its path")
        if document["generation"] != ref.generation:
            raise GenerationConflict(
                "expected generation %d, found %d"
                % (ref.generation, document["generation"]))
        return document

    @staticmethod
    def ref(path: Path, document: Dict[str, Any]) -> LeaseRef:
        return LeaseRef(path=path, job_hash=document["job_hash"],
                        attempt_id=document["attempt_id"],
                        generation=document["generation"], state=document["state"])

    def list(self, *, include_terminal: bool = True) -> List[Tuple[LeaseRef, Dict[str, Any]]]:
        try:
            dir_fd = _safe_directory_fd(self.root, create=False)
        except LeaseError:
            if not os.path.lexists(str(self.root)):
                return []
            raise
        try:
            names = sorted(
                name for name in os.listdir(dir_fd)
                if name.endswith(".json") and not name.startswith("."))
        finally:
            os.close(dir_fd)
        out = []
        for name in names:
            path = self.root / name
            document = self._read_unlocked(path)
            if include_terminal or document["state"] != TERMINAL:
                out.append((self.ref(path, document), document))
        return out

    def begin_create(self, *, job_hash: str, provider: str,
                     request: Mapping[str, Any], pre_create_resources: Iterable[Any],
                     create_deadline_epoch: float, workload_deadline_epoch: float,
                     attempt_id: Optional[str] = None,
                     pre_create_network_volumes: Iterable[Any] = (),
                     controller_pid: Optional[int] = None) -> LeaseRef:
        """Write PREPARED (proving no POST yet) and return the attempt.

        ``pre_create_resources`` must be the provider's complete listing made
        immediately before this call.  Its ids become immutable evidence; its
        names prevent even an astronomically unlikely generated-name collision.
        A caller which must bind the attempt into another pre-POST contract may
        supply an independently random 96-bit ``attempt_id``; otherwise this
        method generates it.
        """
        job_hash = _validate_hash(job_hash)
        provider_name = str(provider).strip().lower()
        if not provider_name:
            raise LeaseError("provider is required")
        if controller_pid is None:
            durable_controller_pid = os.getpid()
        elif (isinstance(controller_pid, bool)
              or not isinstance(controller_pid, int)
              or controller_pid <= 0):
            raise LeaseError("controller_pid must be a positive exact integer")
        else:
            durable_controller_pid = controller_pid
        create_deadline = float(create_deadline_epoch)
        workload_deadline = float(workload_deadline_epoch)
        if create_deadline <= self.clock():
            raise LeaseError("create deadline must be in the future")
        if workload_deadline < create_deadline:
            raise LeaseError("workload deadline cannot precede create deadline")
        reap_deadline = workload_deadline
        if "terminate_after" in request:
            reap_deadline = float(_exact_utc_epoch(
                request["terminate_after"], "RunPod terminate_after"))
            if reap_deadline <= workload_deadline:
                raise LeaseError(
                    "RunPod terminate_after must follow workload deadline")
        resources = [_resource(item) for item in pre_create_resources]
        pre_ids = sorted({rid for rid, _, _ in resources})
        pre_names = {name for _, name, _ in resources}
        pre_network_volume_ids = sorted({
            _resource(item)[0] for item in pre_create_network_volumes})
        request_copy = json.loads(_canonical_bytes(dict(request)).decode("utf-8"))
        observation_until = _provider_deadline_observation_epoch(
            provider_name, request_copy, workload_deadline)
        if observation_until is not None and observation_until <= self.clock():
            raise LeaseError(
                "provider deadline observation bound must be in the future")

        campaign = _campaign_coordinates({
            "create": {"request": request_copy},
        })
        if (campaign is not None
                or request_copy.get("drill_mode")
                == PROVIDER_DEADLINE_DRILL_MODE):
            account = request_copy.get("provider_account_id")
            if (not isinstance(account, str) or not account.strip()
                    or account != account.strip()):
                raise LeaseError(
                    "paid RunPod lease requires exact provider_account_id")

        with self.job_lock(job_hash):
            for path in sorted(self.root.glob("%s.*.json" % job_hash)):
                old = self._read_unlocked(path)
                if old["state"] in UNRESOLVED_STATES:
                    raise LeaseConflict(
                        "job %s already has unresolved attempt %s in %s"
                        % (job_hash, old["attempt_id"], old["state"]))
            if attempt_id is not None:
                attempt = _validate_attempt(attempt_id)
                exact_name = exact_resource_name(job_hash, attempt)
                path = self._path(job_hash, attempt)
                if exact_name in pre_names or path.exists():
                    raise LeaseConflict(
                        "supplied attempt collides with pre-create evidence")
            else:
                for unused in range(32):
                    attempt = secrets.token_hex(ATTEMPT_BYTES)
                    exact_name = exact_resource_name(job_hash, attempt)
                    path = self._path(job_hash, attempt)
                    if exact_name not in pre_names and not path.exists():
                        break
                else:
                    raise LeaseConflict(
                        "could not allocate a collision-free 96-bit attempt")
            request_attempt = request_copy.get("attempt_key")
            if (request_attempt is not None
                    and str(request_attempt) != attempt):
                raise LeaseError(
                    "request attempt_key does not match lease attempt_id")
            now = self.clock()
            create = {
                "provider": provider_name,
                "exact_name": exact_name,
                "pre_create_provider_ids": pre_ids,
                "pre_create_network_volume_ids": pre_network_volume_ids,
                "pre_create_observed_at": _utc_now(now),
                "request": request_copy,
                "request_sha256": _sha256(request_copy),
                "create_deadline_epoch": create_deadline,
                "create_deadline_utc": utc_iso(create_deadline),
                "workload_deadline_epoch": workload_deadline,
                "workload_deadline_utc": utc_iso(workload_deadline),
                "controller_pid": durable_controller_pid,
                "reap_deadline_epoch": reap_deadline,
                "reap_deadline_utc": utc_iso(reap_deadline),
            }
            create["evidence_sha256"] = _sha256(create)
            document = {
                "schema": SCHEMA,
                "job_hash": job_hash,
                "attempt_id": attempt,
                "generation": 0,
                "state": PREPARED,
                "create": create,
                "provider_resource_ids": [],
                "history": [{
                    "generation": 0,
                    "at": _utc_now(now),
                    "from": None,
                    "to": PREPARED,
                    "event": "LEASE_PREPARED_NO_PROVIDER_POST",
                    "evidence": {},
                }],
                "billing_reconciliation": None,
                "terminal_proof": None,
            }
            _validate_lease_schema(_seal(document))
            _atomic_create(path, document)
            return self.ref(path, _seal(document))

    def record_post_intent(self, ref: LeaseRef) -> LeaseRef:
        """Fsync the irrevocable POST boundary immediately before create."""
        document = self._read_current(ref)
        if document["state"] != PREPARED:
            raise LeaseError("provider POST intent requires a PREPARED lease")
        return self.transition(
            ref, to_state=CREATING, event="PROVIDER_POST_INTENT_FSYNCED",
            evidence={
                "submitted_request_sha256":
                    document["create"]["request_sha256"],
                "exact_name": document["create"]["exact_name"],
            })

    def cancel_prepared(self, ref: LeaseRef,
                        evidence: Mapping[str, Any]) -> LeaseRef:
        """Close a PREPARED lease; this state proves no provider POST occurred."""
        proof = json.loads(_canonical_bytes(dict(evidence)).decode("utf-8"))
        proof["no_provider_post"] = True
        return self.transition(
            ref, to_state=TERMINAL, event="PREPARED_CANCELLED_NO_PROVIDER_POST",
            evidence=proof, terminal_proof={"prepared_cancellation": proof})

    def transition(self, ref: LeaseRef, *, to_state: str, event: str,
                   evidence: Mapping[str, Any],
                   provider_resource_ids: Optional[Iterable[Any]] = None,
                   billing_reconciliation: Optional[Mapping[str, Any]] = None,
                   terminal_proof: Optional[Mapping[str, Any]] = None,
                   clear_billing_reconciliation: bool = False,
                   clear_terminal_proof: bool = False) -> LeaseRef:
        if to_state not in _ALLOWED:
            raise LeaseError("unknown target state %r" % to_state)
        evidence_copy = json.loads(_canonical_bytes(dict(evidence)).decode("utf-8"))
        with self.job_lock(ref.job_hash):
            document = self._read_unlocked(ref.path)
            expected_path = self._path(ref.job_hash, ref.attempt_id)
            if ref.path != expected_path or document["job_hash"] != ref.job_hash:
                raise InvalidLease(
                    "lease reference is not keyed by its locked job hash")
            if document["attempt_id"] != ref.attempt_id:
                raise InvalidLease("lease attempt changed under its filename")
            if document["generation"] != ref.generation:
                raise GenerationConflict("expected generation %d, found %d"
                                         % (ref.generation, document["generation"]))
            old_state = document["state"]
            if (old_state == PREPARED and to_state == CREATING
                    and event != "PROVIDER_POST_INTENT_FSYNCED"):
                raise LeaseError(
                    "PREPARED may enter CREATING only through POST intent")
            if to_state not in _ALLOWED[old_state]:
                raise LeaseError("invalid lease transition %s -> %s" % (old_state, to_state))
            immutable = {key: document[key] for key in _IMMUTABLE_TOP}
            generation = ref.generation + 1
            updated = dict(document)
            updated["generation"] = generation
            updated["state"] = to_state
            if provider_resource_ids is not None:
                ids = sorted({_resource_id(value)
                              for value in provider_resource_ids})
                pre = set(document["create"]["pre_create_provider_ids"])
                if any(value in pre for value in ids):
                    raise LeaseError("cannot bind a provider id present before create")
                updated["provider_resource_ids"] = ids
            if clear_billing_reconciliation:
                updated["billing_reconciliation"] = None
            elif billing_reconciliation is not None:
                updated["billing_reconciliation"] = json.loads(
                    _canonical_bytes(dict(billing_reconciliation)).decode("utf-8"))
            if clear_terminal_proof:
                updated["terminal_proof"] = None
            elif terminal_proof is not None:
                updated["terminal_proof"] = json.loads(
                    _canonical_bytes(dict(terminal_proof)).decode("utf-8"))
            history = list(document.get("history") or [])
            history.append({
                "generation": generation,
                "at": _utc_now(self.clock()),
                "from": old_state,
                "to": to_state,
                "event": str(event),
                "evidence": evidence_copy,
            })

            updated["history"] = history
            if any(updated[key] != value for key, value in immutable.items()):
                raise LeaseError("immutable create evidence changed")
            _validate_lease_schema(_seal(updated))
            _atomic_replace(ref.path, updated)
            sealed = _seal(updated)
            return self.ref(ref.path, sealed)
    def record_identity_attestation(
            self, ref: LeaseRef,
            evidence: Mapping[str, Any]) -> LeaseRef:
        document = self._read_current(ref)
        if document["state"] != ACTIVE:
            raise LeaseError("resource identity attestation requires ACTIVE lease")
        attestation = json.loads(
            _canonical_bytes(dict(evidence)).decode("utf-8"))
        _validate_identity_attestation(attestation, document)
        return self.transition(
            ref, to_state=ACTIVE, event="RESOURCE_IDENTITY_ATTESTED",
            evidence=attestation)

    def submit_create_and_record(
            self, ref: LeaseRef,
            submit: Callable[[], Mapping[str, Any]]
            ) -> Tuple[LeaseRef, Mapping[str, Any]]:
        """Serialize the sole provider POST with reaper reconciliation."""
        with self.create_submission_lock(ref):
            current = self._read_current(ref)
            if current["state"] != CREATING:
                raise LeaseError(
                    "create submission requires unchanged fsynced POST intent")
            try:
                response = submit()
            except BaseException as exc:
                acknowledged = getattr(exc, "response", None)
                provider_id = getattr(exc, "provider_id", None)
                if isinstance(acknowledged, Mapping) and provider_id is not None:
                    try:
                        bound = self.record_create_success(ref, acknowledged)
                    except BaseException as persistence_exc:
                        raise CreateResponsePersistenceError(
                            acknowledged, persistence_exc) from exc
                    try:
                        setattr(exc, "durable_lease_ref", bound)
                    except BaseException as attribute_exc:
                        raise LeaseError(
                            "structured create exception cannot retain its "
                            "durable lease reference") from attribute_exc
                raise
            if not isinstance(response, Mapping):
                raise LeaseError("provider create response is not an object")
            try:
                bound = self.record_create_success(ref, response)
            except BaseException as persistence_exc:
                raise CreateResponsePersistenceError(
                    response, persistence_exc) from persistence_exc
            return bound, response
    def record_create_success(self, ref: LeaseRef,
                              response: Mapping[str, Any]) -> LeaseRef:
        """Durably retain the paid resource id before identity qualification.

        A provider success response authorizes cleanup of its exact returned id
        even when later name/GPU/disk validation fails.  Therefore this method
        deliberately does not reject an observed-name mismatch; it records the
        mismatch for the separate post-create binding check after the ACTIVE
        lease containing the id is fsynced.
        """
        document = self._read_current(ref)
        if document["state"] != CREATING:
            raise LeaseError("create success requires fsynced POST intent")


        response_id = response.get(
            "id", response.get("machine_id", response.get("pod_id")))
        provider_id = _resource_id(response_id)
        if provider_id in set(document["create"]["pre_create_provider_ids"]):
            raise LeaseError("create response returned a pre-existing provider id")
        observed_name = response.get("name")
        evidence_response = {
            key: (None if response.get(key) is None else str(response.get(key)))
            for key in (
                "id", "machine_id", "pod_id", "name", "cost_per_hr")
        }
        evidence_response["name_matches_exact"] = (
            str(observed_name or "") == document["create"]["exact_name"])
        return self.transition(
            ref, to_state=ACTIVE, event="CREATE_RESPONSE_BOUND",
            evidence={
                "provider_id_acknowledged": provider_id,
                "submitted_request_sha256":
                    document["create"]["request_sha256"],
                "response": evidence_response,
                "identity_validation_pending": True,
            },
            provider_resource_ids=[provider_id])

    def bind_post_create_inventory(
            self, ref: LeaseRef, resources: Iterable[Any], *,
            network_volumes: Iterable[Any] = (),
            authorized_sibling_pod_ids: Iterable[Any] = ()) -> LeaseRef:
        """Freeze any full-family post-create anomaly before qualification."""
        document = self._read_current(ref)
        if document["state"] != ACTIVE:
            raise LeaseError("post-create inventory binding requires ACTIVE lease")
        intended_ids = list(document["provider_resource_ids"])
        if len(intended_ids) != 1:
            raise LeaseError("post-create inventory lacks one intended pod id")
        intended = intended_ids[0]
        pre_ids = set(document["create"]["pre_create_provider_ids"])
        pre_volumes = set(
            document["create"]["pre_create_network_volume_ids"])
        parsed = [_resource(item) for item in resources]
        parsed_volumes = [_resource(item) for item in network_volumes]
        exact_name = document["create"]["exact_name"]
        new_pods = sorted({
            rid for rid, unused_name, unused_status in parsed
            if rid not in pre_ids})
        exact_ids = sorted({
            rid for rid, name, unused_status in parsed
            if rid not in pre_ids and name == exact_name})
        intended_present = intended in new_pods
        intended_name_matches = intended in exact_ids
        wrong_ids = sorted(
            set(new_pods) - set(exact_ids) - {intended})
        authorized = sorted({
            _resource_id(value) for value in authorized_sibling_pod_ids})
        if not set(authorized).issubset(wrong_ids):
            raise LeaseError(
                "authorized sibling ids must be wrong-name post-create pods")
        blockers = sorted(set(wrong_ids) - set(authorized))
        new_volumes = sorted({
            rid for rid, unused_name, unused_status in parsed_volumes
            if rid not in pre_volumes})
        extra_exact = sorted(set(exact_ids) - {intended})
        if (intended_present and intended_name_matches
                and not extra_exact and not blockers and not new_volumes):
            return ref
        candidates = sorted(set(exact_ids) | {intended})
        evidence = {
            "complete_listing": True,
            "listed_resource_count": len(parsed),
            "listed_network_volume_count": len(parsed_volumes),
            "intended_provider_id": intended,
            "intended_present": intended_present,
            "intended_name_matches_exact": intended_name_matches,
            "new_pod_ids": new_pods,
            "new_exact_name_ids": exact_ids,
            "wrong_name_new_pod_ids": wrong_ids,
            "authorized_sibling_pod_ids": authorized,
            "unattributable_wrong_name_pod_ids": blockers,
            "new_network_volume_ids": new_volumes,
        }
        return self.transition(
            ref, to_state=AMBIGUOUS,
            event="POST_CREATE_FAMILY_DELTA_AMBIGUOUS",
            evidence=evidence, provider_resource_ids=candidates,
            terminal_proof={"ambiguous_create": evidence})

    def reconcile_response_lost(
            self, ref: LeaseRef, resources: Iterable[Any], *,
            network_volumes: Iterable[Any] = (),
            response_provider_id: Optional[Any] = None,
            authorized_sibling_pod_ids: Iterable[Any] = (),
            create_window_closed: bool = False,
            response_error: Optional[str] = None,
            provider_rejection_codes: Iterable[Any] = (),
            seconds_since_window_closed: Optional[float] = None) -> LeaseRef:
        """Bind only provider-attributable IDs from a response-loss window.

        A sole exact-name new pod is the only delta eligible to become ACTIVE.
        Exact-name pods and an explicit POST response ID are cleanup targets.
        Wrong-name pods and new network volumes remain unresolved blockers
        unless the campaign ledger authorizes a wrong-name sibling pod.
        """
        document = self._read_current(ref)
        if document["state"] != CREATING:
            raise LeaseError(
                "response-loss reconciliation requires fsynced POST intent")
        exact_name = document["create"]["exact_name"]
        pre_ids = set(document["create"]["pre_create_provider_ids"])
        pre_volume_ids = set(
            document["create"]["pre_create_network_volume_ids"])
        parsed = [_resource(item) for item in resources]
        parsed_volumes = [_resource(item) for item in network_volumes]
        new_pods = sorted({rid for rid, unused_name, unused_status in parsed
                           if rid not in pre_ids})
        exact_ids = sorted({rid for rid, name, unused_status in parsed
                            if rid not in pre_ids and name == exact_name})
        wrong_ids = sorted(set(new_pods) - set(exact_ids))
        new_volumes = sorted({
            rid for rid, unused_name, unused_status in parsed_volumes
            if rid not in pre_volume_ids})
        response_id = (
            None if response_provider_id is None
            else _resource_id(response_provider_id))
        if response_id is not None and response_id in pre_ids:
            raise LeaseError(
                "create response returned a pre-existing provider id")
        authorized = sorted({
            _resource_id(value) for value in authorized_sibling_pod_ids})
        if not set(authorized).issubset(wrong_ids):
            raise LeaseError(
                "authorized sibling ids must be wrong-name post-create pods")
        blockers = sorted(
            set(wrong_ids) - set(authorized) - (
                {response_id} if response_id is not None else set()))
        candidates = sorted(set(exact_ids) | (
            {response_id} if response_id is not None else set()))
        evidence = {
            "complete_listing": True,
            "listed_resource_count": len(parsed),
            "listed_network_volume_count": len(parsed_volumes),
            "exact_name": exact_name,
            "new_exact_name_ids": exact_ids,
            "new_pod_ids": new_pods,
            "wrong_name_new_pod_ids": wrong_ids,
            "authorized_sibling_pod_ids": authorized,
            "unattributable_wrong_name_pod_ids": blockers,
            "new_network_volume_ids": new_volumes,
            "response_provider_id": response_id,
            "create_window_closed": bool(create_window_closed),
        }
        rejection = tuple(
            str(code) for code in provider_rejection_codes if str(code))
        if rejection:
            evidence["provider_rejection_codes"] = sorted(set(rejection))
        if response_error is not None:
            error = redact(str(response_error).strip())[:1000]
            if not error:
                raise LeaseError("response_error evidence cannot be empty")
            evidence["response_error_redacted"] = error
        safe_one = (
            len(candidates) == 1 and exact_ids == candidates
            and not blockers and not new_volumes)
        if safe_one:
            return self.transition(
                ref, to_state=ACTIVE,
                event="LOST_CREATE_RESPONSE_RECONCILED_ONE",
                evidence=evidence, provider_resource_ids=candidates)
        if candidates or blockers or new_volumes:
            return self.transition(
                ref, to_state=AMBIGUOUS,
                event="LOST_CREATE_RESPONSE_RECONCILED_AMBIGUOUS",
                evidence=evidence, provider_resource_ids=candidates,
                terminal_proof={"ambiguous_create": evidence})
        if rejection:
            # The provider REFUSED this create and named an enumerated
            # no-resource code, on a response that carried no id, and a
            # complete listing shows nothing attributable. Nothing was ever
            # accepted, so there is no liability to retain -- and leaving it
            # in CREATING would close paid admission for the whole campaign
            # permanently.
            return self.transition(
                ref, to_state=TERMINAL,
                event="PROVIDER_REJECTED_CREATE_NO_RESOURCE",
                evidence=evidence,
                terminal_proof={"provider_rejected_create": evidence})
        if (create_window_closed
                and seconds_since_window_closed is not None
                and seconds_since_window_closed
                >= LOST_CREATE_EXPIRY_SECONDS):
            # The create window closed long ago and every complete listing
            # since has shown nothing attributable: no exact-name pod, no
            # wrong-name blocker, no new volume, no acknowledged id.  RunPod
            # answers a create synchronously and never queues one, so a pod
            # cannot still be on its way.  Leaving the lease CREATING kept
            # paid admission closed for the whole campaign after one lost
            # response; that liability was zero the entire time.
            evidence["expired_after_seconds"] = int(
                seconds_since_window_closed)
            return self.transition(
                ref, to_state=TERMINAL,
                event="LOST_CREATE_RESPONSE_EXPIRED_NO_RESOURCE",
                evidence=evidence,
                terminal_proof={"lost_create_expired": evidence})
        event = (
            "LOST_CREATE_RESPONSE_RECONCILED_ZERO_WINDOW_CLOSED_UNRESOLVED"
            if create_window_closed else
            "LOST_CREATE_RESPONSE_RECONCILED_ZERO_PENDING")
        return self.transition(
            ref, to_state=CREATING, event=event, evidence=evidence)

    def request_destroy(self, ref: LeaseRef, evidence: Mapping[str, Any]) -> LeaseRef:
        return self.transition(ref, to_state=DESTROYING,
                               event="DESTROY_REQUESTED", evidence=evidence)

    def confirm_exact_absence(
            self, ref: LeaseRef, resources: Iterable[Any], *,
            authoritative_inventory: Optional[Mapping[str, Any]] = None
            ) -> LeaseRef:
        document = self._read_current(ref)
        surprise_volumes = (
            (document.get("terminal_proof") or {})
            .get("ambiguous_create", {})
            .get("new_network_volume_ids") or [])
        wrong_name_blockers = (
            (document.get("terminal_proof") or {})
            .get("ambiguous_create", {})
            .get("unattributable_wrong_name_pod_ids") or [])
        if wrong_name_blockers:
            raise LeaseError(
                "unattributable pod delta remains an unresolved blocker: "
                + ",".join(wrong_name_blockers))
        if surprise_volumes:
            raise LeaseError(
                "network-volume delta remains an unreleased chargeable blocker: "
                + ",".join(surprise_volumes))
        targets = set(document.get("provider_resource_ids") or [])
        if not targets:
            raise LeaseError("cannot confirm target absence without an exact provider id")
        parsed = [_resource(item) for item in resources]
        listed_ids = {rid for rid, _, _ in parsed}
        present = sorted(targets & listed_ids)
        evidence = {
            "complete_listing": True,
            "listed_resource_count": len(listed_ids),
            "target_provider_ids": sorted(targets),
            "still_present_ids": present,
        }
        if authoritative_inventory is not None:
            evidence["authoritative_inventory"] = json.loads(
                _canonical_bytes(dict(authoritative_inventory)).decode("utf-8"))
        if present:
            # EXITED, TERMINATED, and every other listed status remain live.
            statuses = {rid: status for rid, _, status in parsed if rid in targets}
            evidence["listed_statuses"] = statuses
            return self.transition(ref, to_state=document["state"],
                                   event="EXACT_IDS_STILL_LISTED", evidence=evidence)
        terminal = dict(document.get("terminal_proof") or {})
        terminal["provider_absence"] = evidence
        return self.transition(ref, to_state=ABSENCE_CONFIRMED,
                               event="EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING",
                               evidence=evidence, terminal_proof=terminal)
    def revoke_exact_absence(
            self, ref: LeaseRef, resources: Iterable[Any], *,
            authoritative_inventory: Mapping[str, Any]) -> LeaseRef:
        document = self._read_current(ref)
        if document["state"] != ABSENCE_CONFIRMED:
            raise LeaseError(
                "absence revocation requires an ABSENCE_CONFIRMED lease")
        targets = set(document.get("provider_resource_ids") or [])
        parsed = [_resource(item) for item in resources]
        listed = {rid: status for rid, _, status in parsed}
        present = sorted(targets & set(listed))
        if not present:
            raise LeaseError("absence revocation requires a listed exact id")
        old_absence = (
            (document.get("terminal_proof") or {}).get("provider_absence"))
        if not isinstance(old_absence, dict):
            raise LeaseError("absence revocation lacks prior durable proof")
        evidence = {
            "complete_listing": True,
            "listed_resource_count": len(set(listed)),
            "target_provider_ids": sorted(targets),
            "still_present_ids": present,
            "listed_statuses": {
                provider_id: listed[provider_id] for provider_id in present},
            "authoritative_inventory": json.loads(
                _canonical_bytes(dict(authoritative_inventory)).decode("utf-8")),
            "revoked_absence_sha256": _sha256(old_absence),
        }
        ambiguity = (document.get("terminal_proof") or {}).get(
            "ambiguous_create")
        return self.transition(
            ref, to_state=DESTROYING, event="ABSENCE_PROOF_REVOKED",
            evidence=evidence,
            terminal_proof=(
                {"ambiguous_create": ambiguity}
                if ambiguity is not None else None),
            clear_billing_reconciliation=True,
            clear_terminal_proof=ambiguity is None)

    def stage_billing_reconciliation(self, ref: LeaseRef,
                                     evidence: Mapping[str, Any]) -> LeaseRef:
        """Persist billing before cross-ledger projection and terminal close."""
        document = self._read_current(ref)
        if document["state"] != ABSENCE_CONFIRMED:
            raise LeaseError(
                "billing can be staged only after exact provider absence")
        if document.get("billing_reconciliation") is not None:
            raise LeaseError("billing reconciliation is already staged")
        billing = json.loads(_canonical_bytes(dict(evidence)).decode("utf-8"))
        if billing.get("reconciled") is not True:
            raise LeaseError("billing evidence must explicitly say reconciled=true")
        return self.transition(
            ref, to_state=ABSENCE_CONFIRMED,
            event="BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN",
            evidence=billing, billing_reconciliation=billing)

    def record_billing_reconciled(self, ref: LeaseRef,
                                  evidence: Mapping[str, Any]) -> LeaseRef:
        document = self._read_current(ref)
        if document["state"] != ABSENCE_CONFIRMED:
            raise LeaseError("billing can close only after exact provider absence")
        billing = json.loads(_canonical_bytes(dict(evidence)).decode("utf-8"))
        if billing.get("reconciled") is not True:
            raise LeaseError("billing evidence must explicitly say reconciled=true")
        existing = document.get("billing_reconciliation")
        if existing is not None and existing != billing:
            raise LeaseError("terminal billing differs from staged reconciliation")
        terminal = dict(document.get("terminal_proof") or {})
        terminal["billing_reconciliation"] = billing
        terminal["closed_at"] = _utc_now(self.clock())
        return self.transition(ref, to_state=TERMINAL,
                               event="BILLING_RECONCILED_TERMINAL",
                               evidence=billing,
                               billing_reconciliation=billing,
                               terminal_proof=terminal)



def _campaign_coordinates(document: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    request = (document.get("create") or {}).get("request") or {}
    ledger = request.get("campaign_ledger")
    attempt = request.get("campaign_attempt_key")
    if ledger is None and attempt is None:
        return None
    if (not isinstance(ledger, str) or not ledger
            or Path(ledger).name != ledger or ledger in (".", "..")
            or "\\" in ledger or "\x00" in ledger
            or not isinstance(attempt, str) or not attempt
            or attempt != attempt.strip()):
        raise LeaseError(
            "campaign_ledger must be a safe direct state-dir child leaf paired "
            "with an exact campaign_attempt_key")
    return ledger, attempt


LIVE_LIABILITY_STATES = (PREPARED, CREATING, ACTIVE, DESTROYING, AMBIGUOUS)


def validate_lease_liability_scope(
        store: LeaseStore, *, provider: str, provider_account_id: str,
        allow_live: bool = False) -> Dict[str, Any]:
    """Refuse new spend only while an earlier lease may still hold a pod.

    This is the default-path replacement for `validate_unresolved_lease_scope`,
    which additionally binds every unresolved lease to one campaign ledger
    and to the reaper's last sealed count.  Those bindings are what a
    multi-attempt campaign wants; for one measurement with its own per-run
    ledger they made a previous run's pod-gone-billing-pending lease refuse
    the next run.  ABSENCE_CONFIRMED carries no liability and is ignored.
    A lease in any live-liability state is named with its deadline; the
    installed reaper destroys it at that deadline regardless, and
    ``allow_live`` lets an operator proceed beside it deliberately.
    """
    live = []
    for ref, document in store.list(include_terminal=False):
        if document["create"]["provider"] != provider:
            continue
        if _expected_provider_account(document) != provider_account_id:
            continue
        if document["state"] not in LIVE_LIABILITY_STATES:
            continue
        live.append({
            "lease": ref.path.name,
            "state": document["state"],
            "provider_ids": sorted(
                document.get("provider_resource_ids") or []),
            "reap_deadline_utc": _utc_now(
                document["create"]["reap_deadline_epoch"]),
        })
    live.sort(key=lambda item: item["lease"])
    if live and not allow_live:
        detail = "; ".join(
            "%s is %s (pods %s, reaper destroys by %s)" % (
                item["lease"][:24], item["state"],
                ",".join(item["provider_ids"]) or "none yet",
                item["reap_deadline_utc"])
            for item in live)
        raise LeaseError(
            "an earlier lease may still hold a pod: %s. Wait for the reaper, "
            "run `measure-cloud reaper --provider runpod --sweep`, or pass "
            "--allow-unresolved-leases to proceed beside it" % detail)
    return {
        "live_liability_count": len(live),
        "live_liability_leases": live,
        "scope_sha256": _sha256(live),
    }



def campaign_coordinates(
        document: Mapping[str, Any],
        lease_root: Path) -> Optional[Tuple[Path, str]]:
    """Resolve a validated campaign leaf beside the canonical lease root."""
    coordinates = _campaign_coordinates(document)
    if coordinates is None:
        return None
    ledger_leaf, attempt_key = coordinates
    root = Path(lease_root).expanduser()
    if not root.is_absolute():
        raise LeaseError("campaign lease root must be absolute")
    root = Path(os.path.abspath(str(root)))
    return root.parent / ledger_leaf, attempt_key


def _cancel_prepared_campaign(
        document: Mapping[str, Any], lease_root: Path) -> Optional[Dict[str, Any]]:
    coordinates = campaign_coordinates(document, lease_root)
    if coordinates is None:
        return None
    from .campaign import CampaignLedger
    ledger_path, attempt_key = coordinates
    intent = next(
        (event for event in reversed(document.get("history") or [])
         if event.get("event") == "PREPARED_CANCELLATION_INTENT"),
        None)
    if intent is None:
        raise LeaseError("prepared campaign cancellation lacks durable intent")
    evidence = _sha256({
        "lease": "%s.%s.json" % (
            document["job_hash"], document["attempt_id"]),
        "lease_state": document["state"],
        "lease_generation": document["generation"],
        "lease_record_sha256": document["record_sha256"],
        "prepared_cancellation_intent": intent,
    })
    ledger = CampaignLedger(
        str(ledger_path), document["create"]["provider"],
        _expected_provider_account(document))
    result = ledger.cancel_before_create(
        ledger.snapshot()["generation"], attempt_key,
        intent["at"], "PREPARED", evidence)
    if not result.applied and result.code != "ATTEMPT_UNKNOWN":
        raise LeaseError(
            "campaign prepared cancellation failed [%s]: %s"
            % (result.code, result.message))
    return result.to_dict()

def _release_expired_create_campaign(
        document: Mapping[str, Any], lease_root: Path) -> Optional[Dict[str, Any]]:
    """Release the reservation behind a lost create that expired to TERMINAL."""
    coordinates = campaign_coordinates(document, lease_root)
    if coordinates is None:
        return None
    from .campaign import CampaignLedger
    ledger_path, attempt_key = coordinates
    expiry = next(
        (event for event in reversed(document.get("history") or [])
         if event.get("event") == "LOST_CREATE_RESPONSE_EXPIRED_NO_RESOURCE"),
        None)
    if expiry is None:
        raise LeaseError("expired-create campaign release lacks the expiry event")
    evidence = _sha256({
        "lease": "%s.%s.json" % (
            document["job_hash"], document["attempt_id"]),
        "lease_state": document["state"],
        "lease_generation": document["generation"],
        "lease_record_sha256": document["record_sha256"],
        "lost_create_expiry": expiry,
    })
    ledger = CampaignLedger(
        str(ledger_path), document["create"]["provider"],
        _expected_provider_account(document))
    result = ledger.cancel_before_create(
        ledger.snapshot()["generation"], attempt_key,
        expiry["at"], "LOST_CREATE_EXPIRED", evidence)
    if not result.applied and result.code not in (
            "ATTEMPT_UNKNOWN", "CANCELLATION_ALREADY_RECORDED"):
        raise LeaseError(
            "campaign expired-create release failed [%s]: %s"
            % (result.code, result.message))
    return result.to_dict()

def _expected_provider_account(document: Mapping[str, Any]) -> Optional[str]:
    request = (document.get("create") or {}).get("request") or {}
    value = request.get("provider_account_id")
    if value is None:
        if _campaign_coordinates(document) is not None:
            raise LeaseError("campaign lease lacks provider_account_id")
        return None
    if (not isinstance(value, str) or not value.strip()
            or value != value.strip()):
        raise LeaseError(
            "provider_account_id must be an exact non-empty string")
    return value
def validate_unresolved_lease_scope(
        store: LeaseStore, health: Mapping[str, Any], *,
        provider: str, provider_account_id: str,
        campaign_ledger_path: Optional[Path] = None,
        require_empty: bool = False) -> Dict[str, Any]:
    """Bind every unresolved lease to health and one canonical campaign."""
    stamp = health.get("stamp") if isinstance(health, Mapping) else None
    sealed_count = (
        stamp.get("unresolved_count") if isinstance(stamp, Mapping) else None)
    if (isinstance(sealed_count, bool) or not isinstance(sealed_count, int)
            or sealed_count < 0):
        raise LeaseError(
            "reaper health lacks an exact unresolved lease count")
    unresolved = store.list(include_terminal=False)
    if sealed_count != len(unresolved):
        raise LeaseError(
            "current unresolved lease count differs from reaper health")
    if require_empty and unresolved:
        raise LeaseError(
            "bootstrap paid admission requires zero unresolved leases")
    if require_empty:
        return {
            "unresolved_count": 0,
            "scope_sha256": _sha256([]),
        }
    if campaign_ledger_path is None:
        raise LeaseError(
            "unresolved lease scope requires a canonical campaign ledger")
    from .campaign import CampaignLedger
    ledger_path = Path(campaign_ledger_path).resolve(strict=True)
    ledger = CampaignLedger(
        str(ledger_path), provider, provider_account_id)
    campaign = ledger.snapshot()
    rows = []
    for ref, document in unresolved:
        if (document["create"]["provider"] != provider
                or _expected_provider_account(document)
                    != provider_account_id):
            raise LeaseError(
                "unresolved lease belongs to another provider account")
        coordinates = campaign_coordinates(document, store.root)
        if (coordinates is None
                or coordinates[0].resolve(strict=True) != ledger_path):
            raise LeaseError(
                "unresolved lease is outside the canonical campaign")
        attempt_key = coordinates[1]
        attempt = (campaign.get("attempts") or {}).get(attempt_key)
        if (not isinstance(attempt, dict) or attempt.get("released") is True
                or attempt.get("job_hash") != document["job_hash"]
                or attempt.get("attempt") != document["attempt_id"]
                or sorted(attempt.get("provider_ids") or [])
                    != sorted(document.get("provider_resource_ids") or [])):
            raise LeaseError(
                "unresolved lease lacks one live canonical campaign attempt")
        rows.append({
            "lease_record_sha256": document["record_sha256"],
            "state": document["state"],
            "attempt_key": attempt_key,
        })
    rows.sort(key=lambda item: item["attempt_key"])
    return {
        "unresolved_count": len(rows),
        "scope_sha256": _sha256(rows),
    }




def _assert_provider_account(provider: Any,
                             documents: Iterable[Mapping[str, Any]]) -> Optional[str]:
    expected = {
        value for value in (
            _expected_provider_account(document) for document in documents)
        if value is not None
    }
    if not expected:
        return None
    if len(expected) != 1:
        raise LeaseError("lease group has conflicting provider_account_id values")
    status = getattr(provider, "status", None)
    if status is None:
        raise LeaseError("provider cannot prove current account identity")
    observed = status()
    if (not isinstance(observed, Mapping)
            or not isinstance(observed.get("id"), str)
            or not observed["id"] or observed["id"] != observed["id"].strip()):
        raise LeaseError("provider returned no exact account identity")
    actual = observed["id"]
    wanted = next(iter(expected))
    if not actual or not secrets.compare_digest(actual, wanted):
        raise LeaseError(
            "provider account mismatch: expected %r, observed %r"
            % (wanted, actual or None))
    return actual


def campaign_cleanup_binding_evidence(
        document: Mapping[str, Any],
        exact_provider_ids: Optional[Iterable[Any]] = None) -> str:
    """Bind immutable lease identity to one canonical exact cleanup-ID set."""
    source_ids = (
        document.get("provider_resource_ids") or []
        if exact_provider_ids is None else exact_provider_ids)
    if isinstance(source_ids, (str, bytes)):
        raise LeaseError("campaign cleanup binding IDs must be an iterable set")
    try:
        ids = sorted({_resource_id(value) for value in source_ids})
    except TypeError:
        raise LeaseError("campaign cleanup binding IDs must be iterable")
    if not ids:
        raise LeaseError("campaign cleanup binding requires exact provider IDs")
    return _sha256({
        "job_hash": document["job_hash"],
        "attempt_id": document["attempt_id"],
        "provider_ids": ids,
        "request_sha256": document["create"]["request_sha256"],
    })
def campaign_bound_response_id(
        document: Mapping[str, Any], lease_root: Path) -> Optional[str]:
    """Recover one exact acknowledged ID durably bound only in its campaign."""
    if document.get("provider_resource_ids"):
        return None
    coordinates = campaign_coordinates(document, lease_root)
    if coordinates is None:
        return None
    from .campaign import CampaignLedger
    ledger_path, attempt_key = coordinates
    ledger = CampaignLedger(
        str(ledger_path), document["create"]["provider"],
        _expected_provider_account(document))
    item = (ledger.snapshot().get("attempts") or {}).get(attempt_key)
    if item is None or not item.get("provider_ids"):
        return None
    ids = sorted(item["provider_ids"])
    if (len(ids) != 1 or item.get("released") is True
            or item.get("actual_quote") is not None
            or item.get("phase") not in (
                "TERMINATE_REQUIRED", "TERMINATE_REQUESTED")):
        raise LeaseError(
            "campaign-only create response binding is not cleanup-safe")
    expected = campaign_cleanup_binding_evidence(document, ids)
    observed = item.get("cleanup_binding_evidence")
    if (not isinstance(observed, str)
            or not secrets.compare_digest(observed, expected)):
        raise LeaseError(
            "campaign-only create response binding evidence differs")
    return ids[0]




def _project_terminal_campaign(
        document: Mapping[str, Any], lease_root: Path, *,
        provider_snapshot: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    coordinates = campaign_coordinates(document, lease_root)
    if coordinates is None:
        return None
    from .campaign import CampaignLedger
    ledger_path, attempt_key = coordinates
    ids = sorted(document.get("provider_resource_ids") or [])
    if not ids:
        raise LeaseError("campaign projection requires exact provider IDs")
    terminal = document.get("terminal_proof") or {}
    absence = terminal.get("provider_absence")
    billing = document.get("billing_reconciliation")
    if not isinstance(absence, dict) or not isinstance(billing, dict):
        raise LeaseError(
            "campaign projection requires staged absence and billing evidence")
    absence_event = next(
        (event for event in reversed(document.get("history") or [])
         if event.get("event") == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"),
        None)
    billing_event = next(
        (event for event in reversed(document.get("history") or [])
         if event.get("event")
         == "BILLING_RECONCILIATION_STAGED_FOR_CAMPAIGN"),
        None)
    if absence_event is None or billing_event is None:
        raise LeaseError("campaign projection history is incomplete")
    absence_digest = _sha256(absence)
    absence_proofs = {
        provider_id: {
            "deleted_at": absence_event["at"],
            "proof": absence_digest,
        }
        for provider_id in ids
    }
    binding_evidence = campaign_cleanup_binding_evidence(document)
    final_charge = billing.get("total_amount")
    if final_charge is None:
        raise LeaseError("campaign projection billing lacks total_amount")
    ledger = CampaignLedger(
        str(ledger_path), document["create"]["provider"],
        _expected_provider_account(document))
    result = ledger.project_terminal_lease(
        ledger.snapshot()["generation"], attempt_key, ids,
        binding_evidence, absence_proofs, billing_event["at"],
        final_charge, _sha256(billing),
        provider_snapshot=dict(provider_snapshot))
    if not result.applied:
        raise LeaseError(
            "campaign terminal projection failed [%s]: %s"
            % (result.code, result.message))
    return result.to_dict()


def _provider_for(providers: Mapping[str, Any], name: str) -> Any:
    provider = providers.get(name)
    if provider is None:
        raise LeaseError("no reaper provider registered for %s" % name)
    return provider() if callable(provider) and not hasattr(provider, "list_instances") else provider


def runpod_authoritative_listing(
        provider: Any, lifecycle_resources: Iterable[Any],
        provider_account_id: Optional[str],
        inventory: Optional[Mapping[str, Any]] = None
        ) -> Tuple[List[Any], Dict[str, Any]]:
    """Validate complete REST families and conservatively union pod views."""
    if (provider_account_id is not None
            and (not isinstance(provider_account_id, str)
                 or not provider_account_id
                 or provider_account_id != provider_account_id.strip())):
        raise LeaseError(
            "RunPod authoritative inventory account identity is invalid")
    lifecycle = list(lifecycle_resources)
    graphql_rows = [_resource(item) for item in lifecycle]
    if any(not name or not status for unused_id, name, status in graphql_rows):
        raise LeaseError("RunPod GraphQL lifecycle listing is malformed")
    graphql_by_id = {row[0]: row for row in graphql_rows}
    if len(graphql_by_id) != len(graphql_rows):
        raise LeaseError("RunPod GraphQL lifecycle listing has duplicate ids")
    if inventory is None:
        inventory_method = getattr(provider, "chargeable_inventory", None)
        if inventory_method is None:
            raise LeaseError(
                "RunPod absence requires authoritative chargeable inventory")
        inventory = inventory_method()
    if (not isinstance(inventory, Mapping)
            or inventory.get("schema")
            != "fidelity-suite/runpod-chargeable-inventory.v1"
            or inventory.get("provider") != "runpod"
            or inventory.get("complete") is not True
            or inventory.get("unknown_families") != []):
        raise LeaseError(
            "RunPod authoritative chargeable inventory is incomplete")
    families = inventory.get("families")
    if (not isinstance(families, Mapping)
            or set(families) != {"pods", "network_volumes"}):
        raise LeaseError("RunPod authoritative inventory families differ")
    parsed_families: Dict[str, List[Tuple[str, str, str]]] = {}
    for family_name in ("pods", "network_volumes"):
        family = families[family_name]
        if (not isinstance(family, Mapping)
                or family.get("complete") is not True
                or not isinstance(family.get("resources"), list)):
            raise LeaseError(
                "RunPod authoritative %s inventory is incomplete"
                % family_name)
        rows = []
        for item in family["resources"]:
            parsed = _resource(item)
            if (not parsed[1]
                    or (family_name == "pods" and not parsed[2])):
                raise LeaseError(
                    "RunPod authoritative %s inventory is malformed"
                    % family_name)
            rows.append(parsed)
        ids = [row[0] for row in rows]
        if len(ids) != len(set(ids)):
            raise LeaseError(
                "RunPod authoritative %s inventory has duplicate ids"
                % family_name)
        parsed_families[family_name] = rows
    observed = _exact_utc_string(
        inventory.get("observed_at_utc"),
        "RunPod authoritative inventory time")
    rest_resources = families["pods"]["resources"]
    rest_rows = parsed_families["pods"]
    rest_by_id = {row[0]: row for row in rest_rows}
    for provider_id in sorted(set(graphql_by_id) & set(rest_by_id)):
        if graphql_by_id[provider_id] != rest_by_id[provider_id]:
            raise LeaseError(
                "RunPod GraphQL/REST pod identity conflicts for %s"
                % provider_id)
    union: Dict[str, Any] = {}
    for item, parsed in zip(lifecycle, graphql_rows):
        union[parsed[0]] = item
    for item, parsed in zip(rest_resources, rest_rows):
        union.setdefault(parsed[0], item)
    proof = {
        "schema": "fidelity-suite/runpod-absence-inventory.v1",
        "observed_at_utc": observed,
        "provider_account_id_sha256": (
            None if provider_account_id is None else hashlib.sha256(
                provider_account_id.encode("utf-8")).hexdigest()),
        "graphql_ids": sorted(graphql_by_id),
        "rest_pod_ids": sorted(rest_by_id),
        "inventory_sha256": _sha256(inventory),
    }
    return [union[key] for key in sorted(union)], proof


def _authoritative_listing(
        provider_name: str, provider: Any, lifecycle_resources: Iterable[Any],
        provider_account_id: Optional[str]
        ) -> Tuple[List[Any], Optional[Dict[str, Any]], List[Any]]:
    lifecycle = list(lifecycle_resources)
    if provider_name != "runpod":
        return lifecycle, None, []
    inventory = provider.chargeable_inventory()
    listing, proof = runpod_authoritative_listing(
        provider, lifecycle, provider_account_id, inventory=inventory)
    return (
        listing, proof,
        list(inventory["families"]["network_volumes"]["resources"]))




def _billing_closure(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    closure = json.loads(_canonical_bytes(dict(evidence)).decode("utf-8"))
    closure.pop("evidence", None)
    for history in closure.get("billing_histories") or []:
        if isinstance(history, dict):
            history.pop("retrieved_at_utc", None)
    return closure


def finalize_campaign_after_absence(
        provider: Any, document: Mapping[str, Any],
        lease_root: Path) -> Optional[Dict[str, Any]]:
    coordinates = campaign_coordinates(document, lease_root)
    if coordinates is None:
        return None
    inventory_method = getattr(provider, "chargeable_inventory", None)
    if inventory_method is None:
        raise LeaseError(
            "campaign cleanup requires full-family chargeable inventory")
    inventory = inventory_method()
    if (not isinstance(inventory, dict)
            or inventory.get("complete") is not True
            or inventory.get("provider") != document["create"]["provider"]):
        raise LeaseError(
            "campaign cleanup full-family inventory is incomplete")
    families = inventory.get("families")
    if (not isinstance(families, dict)
            or set(families) != {"pods", "network_volumes"}):
        raise LeaseError("campaign cleanup inventory families differ")
    lifecycle_method = getattr(provider, "list_lifecycle_resources", None)
    if lifecycle_method is None:
        lifecycle_method = getattr(provider, "list_instances", None)
    if lifecycle_method is None:
        raise LeaseError(
            "campaign cleanup requires complete lifecycle pod inventory")
    status = provider.status()
    account = _expected_provider_account(document)
    if (not isinstance(status, dict) or status.get("id") != account
            or status.get("clientBalance") is None):
        raise LeaseError(
            "campaign cleanup balance/account snapshot is unavailable")
    lifecycle_resources = lifecycle_method()
    authoritative_pods, unused_absence_proof = runpod_authoritative_listing(
        provider, lifecycle_resources, account, inventory=inventory)
    from .campaign import CampaignLedger
    ledger_path, attempt_key = coordinates
    ledger = CampaignLedger(
        str(ledger_path), document["create"]["provider"], account)
    target_ids = sorted(document.get("provider_resource_ids") or [])
    terminal = document.get("terminal_proof") or {}
    absence = terminal.get("provider_absence")
    absence_event = next((
        item for item in reversed(document.get("history") or [])
        if item.get("event") == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"
    ), None)
    if not target_ids or not isinstance(absence, dict) or absence_event is None:
        raise LeaseError(
            "campaign inventory refresh lacks exact absence proof")
    snapshot = ledger.snapshot()
    attempt = snapshot["attempts"].get(attempt_key)
    if attempt is None:
        raise LeaseError("campaign inventory refresh attempt is absent")
    if (attempt["provider_ids"]
            and attempt["provider_ids"] != target_ids):
        raise LeaseError(
            "campaign cleanup target ids differ from ledger")
    provider_resources = []
    family_resources = {
        "pods": authoritative_pods,
        "network_volumes": families["network_volumes"].get("resources"),
    }
    for family_name, family in sorted(families.items()):
        if (not isinstance(family, dict)
                or family.get("complete") is not True
                or not isinstance(family_resources[family_name], list)):
            raise LeaseError(
                "campaign cleanup inventory family is incomplete")
        for row in family_resources[family_name]:
            if not isinstance(row, dict):
                raise LeaseError(
                    "campaign cleanup inventory resource is malformed")
            resource_id = str(row.get("id") or "")
            resource = {
                "family": family_name,
                "id": resource_id,
                "name": str(row.get("name") or resource_id),
                "status": str(row.get("status") or "PRESENT"),
            }
            if not resource_id:
                raise LeaseError(
                    "campaign cleanup inventory resource lacks id")
            provider_resources.append(resource)
    classification = ledger.classify_provider_resources(provider_resources)
    provider_resources = classification["provider_resources"]
    refreshed_target_ids = {
        item["id"] for item in provider_resources
        if item["family"] == "pods" and item["id"] in set(target_ids)
    }
    if refreshed_target_ids:
        raise LeaseError(
            "current lease target reappeared in fresh full-family inventory: "
            + ",".join(sorted(refreshed_target_ids)))
    tolerate_foreign = (
        ledger.foreign_resources_policy(snapshot) == "tolerate")
    if classification["unknown_resources"] and not tolerate_foreign:
        raise LeaseError(
            "fresh full-family inventory contains unknown chargeable resources")
    inventory_observed = _exact_utc_string(
        inventory.get("observed_at_utc"),
        "campaign cleanup inventory time")
    balance_observed = _exact_utc_string(
        status.get("observed_at_utc"),
        "campaign cleanup balance time")
    inventory_valid_until = _utc_now(
        _exact_utc_epoch(
            inventory_observed, "campaign cleanup inventory time") + 60)
    balance_valid_until = _utc_now(
        _exact_utc_epoch(
            balance_observed, "campaign cleanup balance time") + 60)
    provider_snapshot = {
        "provider": document["create"]["provider"],
        "provider_account_id": account,
        "balance_available_usd": status["clientBalance"],
        "balance_observed_at": balance_observed,
        "balance_valid_until": balance_valid_until,
        "balance_source": "RunPod myself.clientBalance after exact absence",
        "inventory_observed_at": inventory_observed,
        "inventory_valid_until": inventory_valid_until,
        "inventory_complete": True,
        "provider_resources": provider_resources,
        "inventory_source": (
            inventory["schema"] + " classified after exact absence"),
    }
    campaign_projection = _project_terminal_campaign(
        document, lease_root, provider_snapshot=provider_snapshot)
    if campaign_projection is None:
        raise LeaseError("campaign inventory refresh lost campaign coordinates")
    persisted = ledger.snapshot()
    persisted_inventory = persisted.get("inventory")
    if (not isinstance(persisted_inventory, dict)
            or persisted_inventory.get("provider_resources")
            != sorted(provider_resources,
                      key=lambda item: (item["family"], item["id"]))):
        raise LeaseError(
            "campaign inventory changed before cleanup classification")
    persisted_target_ids = {
        item["id"] for item in persisted_inventory["provider_resources"]
        if item["family"] == "pods" and item["id"] in set(target_ids)
    }
    if persisted_target_ids:
        raise LeaseError(
            "current lease target remains in full-family inventory: "
            + ",".join(sorted(persisted_target_ids)))
    if persisted_inventory["unknown_resources"] and not tolerate_foreign:
        raise LeaseError(
            "campaign inventory retains unknown chargeable resources")
    return campaign_projection


def _stabilized_billing(
        provider: Any, document: Mapping[str, Any],
        instant: float) -> Optional[Dict[str, Any]]:
    reconcile = getattr(provider, "reconcile_billing", None)
    if reconcile is None:
        raise LeaseError("provider exposes no billing reconciliation")
    if document["create"]["provider"] != "runpod":
        return reconcile(dict(document))
    absence = next((
        item for item in reversed(document.get("history") or [])
        if item.get("event") == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"
    ), None)
    if absence is None:
        raise LeaseError("RunPod billing stabilization lacks absence evidence")
    try:
        absence_epoch = calendar.timegm(time.strptime(
            absence["at"], "%Y-%m-%dT%H:%M:%SZ"))
    except (KeyError, TypeError, ValueError, OverflowError):
        raise LeaseError("RunPod absence timestamp is invalid")
    if instant - absence_epoch < BILLING_STABILIZATION_SECONDS:
        return None
    first = reconcile(dict(document))
    if ((first.get("evidence") or {}).get("schema")
            == "fidelity-suite/runpod-billing-stabilization.v1"):
        _validate_billing(first, "runpod")
        if first["evidence"]["absence_confirmed_at"] != absence["at"]:
            raise LeaseError(
                "RunPod billing stabilization binds another absence event")
        return first
    second = reconcile(dict(document))
    _validate_billing(first, "runpod", require_stabilized=False)
    _validate_billing(second, "runpod", require_stabilized=False)
    first_retrieval = first.get("evidence")
    second_retrieval = second.get("evidence")
    required = {"schema", "retrieval_id", "retrieved_at_utc"}
    if (not isinstance(first_retrieval, dict)
            or set(first_retrieval) != required
            or not isinstance(second_retrieval, dict)
            or set(second_retrieval) != required
            or first_retrieval.get("schema")
            != "fidelity-suite/runpod-billing-retrieval.v1"
            or second_retrieval.get("schema")
            != "fidelity-suite/runpod-billing-retrieval.v1"
            or not isinstance(first_retrieval.get("retrieval_id"), str)
            or not isinstance(second_retrieval.get("retrieval_id"), str)
            or first_retrieval["retrieval_id"]
            == second_retrieval["retrieval_id"]):
        raise LeaseError(
            "RunPod billing retrievals lack independent identities")
    _exact_utc_string(
        first_retrieval.get("retrieved_at_utc"),
        "first billing retrieval time")
    _exact_utc_string(
        second_retrieval.get("retrieved_at_utc"),
        "second billing retrieval time")
    first_closure = _billing_closure(first)
    second_closure = _billing_closure(second)
    if first_closure != second_closure:
        raise LeaseError(
            "RunPod billing changed between stabilization retrievals")
    stabilized = json.loads(_canonical_bytes(second).decode("utf-8"))
    stabilized["evidence"] = {
        "schema": "fidelity-suite/runpod-billing-stabilization.v1",
        "absence_confirmed_at": absence["at"],
        "minimum_stabilization_seconds": BILLING_STABILIZATION_SECONDS,
        "closure_sha256": _sha256(second_closure),
        "first_retrieval": first_retrieval,
        "second_retrieval": second_retrieval,
    }
    return stabilized


def _same_uid_process_alive(pid: Any) -> Optional[bool]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        process = os.stat("/proc/%d" % pid, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    if hasattr(os, "getuid") and process.st_uid != os.getuid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    return True


def reap_once(store: LeaseStore, providers: Mapping[str, Any], *,
              now: Optional[float] = None, dry_run: bool = False) -> ReaperResult:
    """Reap all providers independently; never convert an outage to absence."""
    instant = time.time() if now is None else float(now)
    actions: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Tuple[LeaseRef, Dict[str, Any]]]] = {}
    for initial_ref, initial_document in store.list(include_terminal=False):
        if initial_document["state"] != PREPARED:
            grouped.setdefault(
                initial_document["create"]["provider"], []).append(
                    (initial_ref, initial_document))
            continue
        ref, document = initial_ref, initial_document
        owner_alive = _same_uid_process_alive(
            document["create"].get("controller_pid"))
        if owner_alive is not False:
            actions.append({
                "lease": ref.path.name,
                "action": "prepared-controller-active-deferred",
                "controller_pid": document["create"].get("controller_pid"),
            })
            continue
        try:
            if dry_run:
                actions.append({
                    "lease": ref.path.name,
                    "action": "would-cancel-prepared-no-post",
                })
                continue
            cancellation_intent = next(
                (event for event in reversed(document.get("history") or [])
                 if event.get("event") == "PREPARED_CANCELLATION_INTENT"),
                None)
            if cancellation_intent is None:
                ref = store.transition(
                    ref, to_state=PREPARED,
                    event="PREPARED_CANCELLATION_INTENT",
                    evidence={
                        "reason": "reaper found prepared lease with no POST intent",
                    })
                document = store.read(ref)
            campaign = _cancel_prepared_campaign(document, store.root)
            ref = store.cancel_prepared(ref, {
                "reason": "reaper cancelled prepared lease with no POST intent",
                "campaign_projection": campaign,
            })
            actions.append({
                "lease": ref.path.name,
                "action": "prepared-cancelled-no-post",
            })
        except Exception as exc:
            failures.append({
                "lease": ref.path.name,
                "provider": document["create"]["provider"],
                "operation": "cancel-prepared",
                "error": str(exc),
            })

    for provider_name in sorted(grouped):
        try:
            provider = _provider_for(providers, provider_name)
            provider_account_id = _assert_provider_account(
                provider, [document for _, document in grouped[provider_name]])
            lifecycle_listing = list(provider.list_instances())
            listing, authoritative_inventory, authoritative_volumes = (
                _authoritative_listing(
                    provider_name, provider, lifecycle_listing,
                    provider_account_id))
            # Parse every union entry now: malformed/incomplete inventory is an
            # outage, never evidence that an omitted exact id is absent.
            parsed_listing = [_resource(item) for item in listing]
        except Exception as exc:  # provider families do not share an exception base
            failures.append({"provider": provider_name, "operation": "list",
                             "error": str(exc)})
            continue

        for initial_ref, initial_doc in grouped[provider_name]:
            ref, document = initial_ref, initial_doc
            try:
                if document["state"] == ABSENCE_CONFIRMED:
                    targets = set(document.get("provider_resource_ids") or [])
                    listed_ids = {rid for rid, _, unused in parsed_listing}
                    rediscovered = targets & listed_ids
                    if rediscovered:
                        if dry_run:
                            actions.append({
                                "lease": ref.path.name,
                                "action": "would-revoke-absence-and-destroy",
                                "provider_ids": sorted(rediscovered),
                            })
                            continue
                        if authoritative_inventory is None:
                            raise LeaseError(
                                "absence recovery lacks authoritative inventory")
                        ref = store.revoke_exact_absence(
                            ref, listing,
                            authoritative_inventory=authoritative_inventory)
                        document = store.read(ref)
                        actions.append({
                            "lease": ref.path.name,
                            "action": "absence-proof-revoked",
                            "provider_ids": sorted(rediscovered),
                        })
                if document["state"] == CREATING:
                    with store.create_submission_lock(ref):
                        refreshed_document = store.read(ref)
                        if (refreshed_document["generation"] != ref.generation
                                or refreshed_document["state"] != CREATING):
                            actions.append({
                                "lease": ref.path.name,
                                "action": "create-submission-race-deferred",
                            })
                            continue
                        document = refreshed_document
                        ref = store.ref(ref.path, document)
                        if dry_run:
                            actions.append({
                                "lease": ref.path.name,
                                "action": "would-reconcile-create",
                            })
                            continue
                        response_provider_id = campaign_bound_response_id(
                            document, store.root)
                        create_deadline = document["create"][
                            "create_deadline_epoch"]
                        ref = store.reconcile_response_lost(
                            ref, listing,
                            network_volumes=authoritative_volumes,
                            response_provider_id=response_provider_id,
                            create_window_closed=(
                                instant >= create_deadline),
                            response_error=(
                                "campaign-bound create response persistence "
                                "recovery"
                                if response_provider_id is not None else None),
                            seconds_since_window_closed=max(
                                0.0, instant - float(create_deadline)))
                        document = store.read(ref)
                        action = {
                            "lease": ref.path.name,
                            "action": "reconciled-create",
                            "state": ref.state,
                        }
                        if (ref.state == TERMINAL
                                and "lost_create_expired"
                                in (document.get("terminal_proof") or {})):
                            action["action"] = "lost-create-expired"
                            action["campaign_release"] = (
                                _release_expired_create_campaign(
                                    document, store.root))
                        actions.append(action)

                if document["state"] == AMBIGUOUS:
                    candidates = sorted(document.get("provider_resource_ids") or [])
                    if not candidates:
                        # Wrong-name pods or new volumes appeared in the
                        # create window and none is attributable to this
                        # lease.  There is nothing this reaper may delete,
                        # and raising here every sweep marked the whole
                        # reaper unhealthy, which blocked every other lease
                        # and every new admission behind one operator call.
                        actions.append({
                            "lease": ref.path.name,
                            "action": "ambiguous-needs-operator",
                            "blockers": (
                                (document.get("terminal_proof") or {})
                                .get("ambiguous_create") or {}),
                        })
                        continue
                    if dry_run:
                        actions.append({
                            "lease": ref.path.name,
                            "action": "would-destroy-ambiguous-candidates",
                            "provider_ids": candidates,
                        })
                        continue
                    ref = store.request_destroy(ref, {
                        "reason": "ambiguous create; delete every attributable candidate",
                        "ambiguous_create": True,
                        "provider_ids": candidates,
                    })
                    document = store.read(ref)

                if document["state"] in (ACTIVE, DESTROYING):
                    targets = set(document.get("provider_resource_ids") or [])
                    listed = {rid: status for rid, _, status in parsed_listing}
                    present = targets & set(listed)
                    if not targets:
                        raise LeaseError("active lease has no exact provider id")
                    if not present:
                        if dry_run:
                            actions.append({"lease": ref.path.name,
                                            "action": "would-confirm-exact-absence"})
                            continue
                        ref = store.confirm_exact_absence(
                            ref, listing,
                            authoritative_inventory=authoritative_inventory)
                        document = store.read(ref)
                        actions.append({"lease": ref.path.name,
                                        "action": "exact-absence-confirmed"})
                    else:
                        # The provider accepted terminateAfter but has been
                        # observed ignoring it.  Drill leases therefore use
                        # the same timestamp as an independently enforced
                        # local reaper deadline; the later observation bound
                        # limits proof collection, never delays cleanup.
                        _provider_deadline_observation_epoch(
                            document["create"]["provider"],
                            document["create"]["request"],
                            document["create"]["workload_deadline_epoch"])
                        immediate_statuses = {
                            "EXITED", "TERMINATED", "STOPPED", "FAILED",
                            "FAILURE", "ERROR", "DEAD",
                        }
                        terminal_present = sorted(
                            target for target in present
                            if str(listed[target]).strip().upper()
                            in immediate_statuses)
                        terminal_status_requires_destroy = bool(terminal_present)
                        reap_deadline_expired = (
                            instant >= document["create"]["reap_deadline_epoch"])
                        destroy_now = (
                            document["state"] == DESTROYING
                            or terminal_status_requires_destroy
                            or reap_deadline_expired)
                        if not destroy_now:
                            continue
                        reason = (
                            "listed terminal/stopped/failure status"
                            if terminal_status_requires_destroy else
                            "destroy already pending"
                            if document["state"] == DESTROYING else
                            "absolute reap deadline expired")
                        if dry_run:
                            actions.append({
                                "lease": ref.path.name,
                                "action": "would-destroy",
                                "provider_ids": sorted(present),
                                "reason": reason,
                            })
                            continue
                        if document["state"] == ACTIVE:
                            ref = store.request_destroy(ref, {
                                "reason": reason,
                                "provider_ids": sorted(targets),
                                "listed_statuses": {
                                    target: listed[target]
                                    for target in sorted(present)},
                            })
                            document = store.read(ref)
                        # Destroy only candidates still present.  A partially
                        # completed prior sweep must not fail on an already
                        # absent id before reaching the remaining live id.
                        for target in sorted(present):
                            provider.destroy(target)
                            actions.append({
                                "lease": ref.path.name,
                                "action": "destroy-requested",
                                "provider_id": target,
                            })
                        # A fresh complete listing is required.  No status,
                        # including EXITED or TERMINATED, proves absence.
                        refreshed_account_id = _assert_provider_account(
                            provider, [document])
                        refreshed_lifecycle = list(provider.list_instances())
                        refreshed, refreshed_inventory, unused_volumes = (
                            _authoritative_listing(
                                provider_name, provider, refreshed_lifecycle,
                                refreshed_account_id))
                        [_resource(item) for item in refreshed]
                        ref = store.confirm_exact_absence(
                            ref, refreshed,
                            authoritative_inventory=refreshed_inventory)
                        document = store.read(ref)
                        if ref.state != ABSENCE_CONFIRMED:
                            failures.append({
                                "lease": ref.path.name,
                                "provider": provider_name,
                                "operation": "confirm-destroy",
                                "error": "exact id remains listed",
                            })
                        else:
                            actions.append({
                                "lease": ref.path.name,
                                "action": "exact-absence-confirmed",
                            })

                if document["state"] == ABSENCE_CONFIRMED:
                    if dry_run:
                        actions.append({
                            "lease": ref.path.name,
                            "action": "would-reconcile-billing-and-campaign",
                        })
                        continue
                    billing = document.get("billing_reconciliation")
                    if billing is None:
                        # The pod is proven absent; nothing is billing.  The
                        # provider publishes its hour bucket up to an hour
                        # and some minutes later, and a 503 on the read is
                        # routine.  Neither is a reaper failure: record it
                        # and try again next sweep.  Failing the sweep here
                        # deleted the health stamp and refused every new
                        # admission for the whole time the bucket was late.
                        try:
                            billing = _stabilized_billing(
                                provider, document, instant)
                        except Exception as exc:  # noqa: BLE001
                            actions.append({
                                "lease": ref.path.name,
                                "action": "billing-pending",
                                "reason": str(exc)[:300],
                            })
                            continue
                        if billing is None:
                            actions.append({
                                "lease": ref.path.name,
                                "action": "billing-stabilization-waiting",
                            })
                            continue
                        ref = store.stage_billing_reconciliation(ref, billing)
                        document = store.read(ref)
                    campaign = finalize_campaign_after_absence(
                        provider, document, store.root)
                    ref = store.record_billing_reconciled(ref, billing)
                    actions.append({
                        "lease": ref.path.name,
                        "action": "terminal-retained",
                        "campaign_projection": campaign,
                    })
            except Exception as exc:
                # Isolate both provider and per-lease failures.  No lease is
                # removed; the generation history records every completed step.
                failures.append({"lease": ref.path.name, "provider": provider_name,
                                 "operation": "reap", "error": str(exc)})

    if not dry_run:
        for action in actions:
            lease_name = action.get("lease")
            if not lease_name or str(action.get("action", "")).startswith("would-"):
                continue
            try:
                action_lease = store.read(store.root / lease_name)
                action["lease_record_sha256"] = action_lease["record_sha256"]
                action["lease_generation"] = action_lease["generation"]
            except LeaseError as exc:
                failures.append({
                    "lease": lease_name,
                    "operation": "bind-action-lease-record",
                    "error": str(exc),
                })
    unresolved = tuple(ref.path.name for ref, _ in store.list(include_terminal=False))
    return ReaperResult(ok=not failures, actions=tuple(actions),
                        failures=tuple(failures), unresolved=unresolved)


def _systemd_arg(value: str) -> str:
    text = str(value)
    if any(char in text for char in ("\x00", "\r", "\n")):
        raise LeaseError("systemd argument contains a control character")
    return '"%s"' % text.replace(
        "%", "%%").replace("\\", "\\\\").replace('"', '\\"')


def _systemd_path(value: str) -> str:
    text = str(value)
    if re.fullmatch(r"/[A-Za-z0-9_./-]+", text) is None:
        raise LeaseError(
            "systemd working directory must be an absolute path containing "
            "only letters, digits, '/', '.', '_' or '-'")
    return text

def _service_text(command: Sequence[str], state: Path) -> str:
    exec_start = " ".join(_systemd_arg(str(item)) for item in command)
    return (
        "[Unit]\nDescription=Fidelity cloud lease reaper\n\n"
        "[Service]\nType=oneshot\nExecStart=%s\n"
        "WorkingDirectory=%s\nUMask=0077\n"
        "Environment=PYTHONPATH=\nEnvironment=PYTHONNOUSERSITE=1\n"
        "Environment=PYTHONSAFEPATH=1\nUnsetEnvironment=PYTHONHOME\n"
        % (exec_start, _systemd_path(str(state))))

def _timer_text(interval: int) -> str:
    return (
        "[Unit]\nDescription=Run Fidelity cloud lease reaper\n\n"
        "[Timer]\nOnBootSec=1min\nOnUnitActiveSec=%ds\nPersistent=true\n"
        "AccuracySec=15s\nUnit=fidelity-cloud-reaper.service\n\n"
        "[Install]\nWantedBy=timers.target\n" % interval)


def _logical_control_name(path: Path) -> str:
    if path.name == "reap_cloud_leases.py":
        value = "bin/reap_cloud_leases.py"
    elif path.parent.name == "fidelity":
        value = "bin/fidelity/" + path.name
    else:
        value = path.name
    if (not value or value.startswith("/") or ".." in value.split("/")
            or "\\" in value):
        raise LeaseError("unsafe logical control file name")
    return value


def _verified_file_fd(path: Path, *, strict: bool = True) -> int:
    """Open a trusted file without following any source path component.

    ``strict`` additionally refuses any group/other write bit on the file or
    its ancestors.  That is the right rule for the installed runtime snapshot,
    which the systemd timer executes unattended with the provider credential.
    It is the wrong rule for the checkout the snapshot is copied FROM: a
    ``umask 002`` clone is group-writable by default, and refusing to read it
    blocked every reinstall until the operator chmod'ed eight files by hand.
    Ownership, regular-file and no-symlink rules apply in both modes.
    """
    target = Path(os.path.abspath(str(path)))
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    trusted_owners = {0}
    if hasattr(os, "getuid"):
        trusted_owners.add(os.getuid())
    writable_mask = 0o022 if strict else 0
    fd = os.open("/", directory_flags)
    try:
        root_info = os.fstat(fd)
        if (root_info.st_uid not in trusted_owners
                or root_info.st_mode & writable_mask):
            raise LeaseError("source root directory is writable or untrusted")
        for component in target.parts[1:-1]:
            next_fd = os.open(
                component, directory_flags | nofollow, dir_fd=fd)
            info = os.fstat(next_fd)
            if (not stat.S_ISDIR(info.st_mode)
                    or info.st_uid not in trusted_owners
                    or info.st_mode & writable_mask):
                os.close(next_fd)
                raise LeaseError(
                    "source path parent is writable or untrusted: %s"
                    % target)
            os.close(fd)
            fd = next_fd
        file_fd = os.open(
            target.name, os.O_RDONLY | nofollow, dir_fd=fd)
        info = os.fstat(file_fd)
        if (not stat.S_ISREG(info.st_mode)
                or info.st_uid not in trusted_owners
                or info.st_mode & writable_mask):
            os.close(file_fd)
            raise LeaseError(
                "source file must be trusted-owner regular and non-writable: %s"
                % target)
        return file_fd
    except LeaseError:
        raise
    except OSError as exc:
        raise LeaseError("unsafe source path %s: %s" % (target, exc))
    finally:
        os.close(fd)


def _verified_file_bytes(path: Path, *, strict: bool = True) -> bytes:
    fd = _verified_file_fd(path, strict=strict)
    try:
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _control_file_rows(paths: Iterable[Path], *,
                       strict: bool = True) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for path in sorted(
            (Path(os.path.abspath(str(item))) for item in paths),
            key=lambda item: str(item)):
        raw = _verified_file_bytes(path, strict=strict)
        logical = _logical_control_name(path)
        if logical in seen:
            raise LeaseError("duplicate logical control file name %s" % logical)
        seen.add(logical)
        rows.append({
            "path": logical,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return sorted(rows, key=lambda row: row["path"])


def _source_closure(entrypoint: Path) -> List[Tuple[str, Path]]:
    entry = Path(os.path.abspath(str(entrypoint)))
    if entry.name != "reap_cloud_leases.py":
        raise LeaseError(
            "installed reaper requires the dedicated reap_cloud_leases.py")
    fidelity = entry.parent / "fidelity"
    names = (
        "__init__.py", "cloudlease.py", "campaign.py", "common.py",
        "runpodapi.py", "jlapi.py", "sshbase.py")
    return [("reap_cloud_leases.py", entry)] + [
        ("fidelity/" + name, fidelity / name) for name in names]


def _write_snapshot_file(directory_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise LeaseError("short write while installing reaper snapshot")
            view = view[written:]
        os.fchmod(fd, 0o600)
        os.fsync(fd)
    finally:
        os.close(fd)


def _install_runtime_snapshot(
        state: Path, closure: Sequence[Tuple[str, Path]]
        ) -> Tuple[Path, List[Path], List[Path]]:
    # Read every byte from the already verified descriptor before publishing
    # any runtime name. A rename in the source checkout cannot redirect a copy.
    source_paths = [path for unused, path in closure]
    # The checkout is read without the write-bit rule; the snapshot copies
    # written below are 0600 inside a 0700 directory and are what the timer
    # executes, so they are the bytes that must stay immutable.
    payloads = [(logical, _verified_file_bytes(path, strict=False))
                for logical, path in closure]
    digest = hashlib.sha256()
    for logical, raw in payloads:
        digest.update(logical.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(raw).digest())
    state_fd = _safe_directory_fd(state, create=True)
    leaf = "reaper-runtime-%s-%s" % (
        digest.hexdigest()[:16], secrets.token_hex(8))
    runtime_paths = []
    try:
        os.mkdir(leaf, 0o700, dir_fd=state_fd)
        runtime_fd = os.open(
            leaf, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0), dir_fd=state_fd)
        try:
            os.mkdir("fidelity", 0o700, dir_fd=runtime_fd)
            fidelity_fd = os.open(
                "fidelity", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0), dir_fd=runtime_fd)
            try:
                for logical, raw in payloads:
                    if logical.startswith("fidelity/"):
                        _write_snapshot_file(
                            fidelity_fd, logical.split("/", 1)[1], raw)
                    else:
                        _write_snapshot_file(runtime_fd, logical, raw)
                    runtime_paths.append(state / leaf / logical)
                os.fsync(fidelity_fd)
            finally:
                os.close(fidelity_fd)
            os.fsync(runtime_fd)
        finally:
            os.close(runtime_fd)
        os.fsync(state_fd)
    finally:
        os.close(state_fd)
    return state / leaf, source_paths, runtime_paths


def _system_python_identity() -> Dict[str, str]:
    executable = Path(os.path.realpath("/usr/bin/python3"))
    raw = _verified_file_bytes(executable)
    try:
        completed = subprocess.run(
            [str(executable), "-I", "-S", "-c",
             "import sys;print('%d.%d' % sys.version_info[:2])"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LeaseError("trusted system Python is unavailable: %s" % exc)
    version = completed.stdout.strip()
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", version)
    if (completed.returncode != 0 or match is None
            or (int(match.group(1)), int(match.group(2))) < (3, 9)):
        raise LeaseError("trusted system Python 3.9+ is unavailable")
    return {
        "path": str(executable),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "version": version,
        "implementation": "cpython",
    }


def _control_documents(
        command: Sequence[str], source_command: Sequence[str],
        state_dir: Path, lease_dir: Path,
        provider: str, provider_account_id: str, service: Path,
        timer: Path, interval: int, source_paths: Iterable[Path],
        runtime_paths: Iterable[Path],
        interpreter: Mapping[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sources = tuple(source_paths)
    runtime = tuple(runtime_paths)
    argv = [str(item) for item in command]
    original_argv = [str(item) for item in source_command]
    if (len(argv) < 4 or not os.path.isabs(argv[0])
            or argv[0] != interpreter["path"]
            or argv[1:3] != ["-I", "-S"]
            or not os.path.isabs(argv[3])):
        raise LeaseError("reaper command must use isolated trusted Python")
    state = Path(os.path.abspath(str(state_dir)))
    leases = Path(os.path.abspath(str(lease_dir)))
    if (not isinstance(provider_account_id, str)
            or not provider_account_id
            or provider_account_id != provider_account_id.strip()):
        raise LeaseError("reaper control requires exact provider_account_id")
    if (not isinstance(provider, str) or not provider
            or provider != provider.strip()):
        raise LeaseError("reaper control requires exact provider")
    service_raw = _service_text(argv, state).encode("utf-8")
    timer_raw = _timer_text(interval).encode("utf-8")
    verified_interpreter = _control_file_rows([Path(interpreter["path"])])[0]
    if verified_interpreter["sha256"] != interpreter["sha256"]:
        raise LeaseError("trusted system Python changed during installation")
    # Control binds the RUNTIME snapshot: those are the bytes the timer
    # executes.  The checkout they were copied from is recorded for the
    # advisory drift probe (`reaper_source_drift`) and is deliberately not
    # part of the sealed control, so editing the checkout never makes an
    # installed, unchanged reaper report itself unhealthy or refuse to sweep.
    runtime_rows = _control_file_rows(runtime)
    if {row["path"] for row in runtime_rows} != CONTROL_CLOSURE_PATHS:
        raise LeaseError(
            "reaper control runtime closure is noncanonical")
    public = {
        "command_sha256": _sha256(argv),
        "source_command_sha256": _sha256(original_argv),
        "service_unit": service.name,
        "service_unit_sha256": hashlib.sha256(service_raw).hexdigest(),
        "timer_unit": timer.name,
        "timer_unit_sha256": hashlib.sha256(timer_raw).hexdigest(),
        "runtime_files": runtime_rows,
        "interpreter": {
            "executable_path_sha256": hashlib.sha256(
                interpreter["path"].encode("utf-8")).hexdigest(),
            "executable_file_sha256": interpreter["sha256"],
            "version": interpreter["version"],
            "implementation": interpreter["implementation"],
        },
        "state_dir_sha256": hashlib.sha256(
            str(state).encode("utf-8")).hexdigest(),
        "lease_dir_sha256": hashlib.sha256(
            str(leases).encode("utf-8")).hexdigest(),
        "provider": provider,
        "provider_account_id_sha256": hashlib.sha256(
            provider_account_id.encode("utf-8")).hexdigest(),
    }
    public["control_sha256"] = _sha256(public)
    internal = {
        "schema": CONTROL_SCHEMA,
        "command_argv": argv,
        "source_command_argv": original_argv,
        "state_dir": str(state),
        "lease_dir": str(leases),
        "provider": provider,
        "provider_account_id": provider_account_id,
        "service_unit_path": str(service),
        "timer_unit_path": str(timer),
        "timer_interval_seconds": interval,
        "source_paths": sorted(str(Path(os.path.abspath(str(path))))
                               for path in sources),
        "runtime_paths": sorted(str(Path(os.path.abspath(str(path))))
                                for path in runtime),
        "runtime_root": str(Path(argv[3]).parent),
        "interpreter": dict(interpreter),
        "public_control": public,
    }
    return internal, public


def _read_sealed_document(path: Path, schema: str) -> Dict[str, Any]:
    parent = Path(os.path.abspath(str(path.parent)))
    dir_fd = _safe_directory_fd(parent, create=False)
    fd = None
    try:
        fd = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=dir_fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
            raise LeaseError("sealed state file must be regular owner mode 0600")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            document = _strict_lease_loads(stream.read(), path)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dir_fd)
    seal = document.get("record_sha256")
    unsealed = dict(document)
    unsealed.pop("record_sha256", None)
    observed_schema = document.get("schema")
    if (observed_schema == "fidelity-suite/reaper-control.v2"
            and schema == CONTROL_SCHEMA):
        raise LeaseError(
            "installed reaper control is the older v2 schema; run "
            "`measure-cloud reaper --provider runpod --install` once to "
            "re-snapshot it")
    if (observed_schema != schema or not isinstance(seal, str)
            or not secrets.compare_digest(seal, _sha256(unsealed))):
        raise LeaseError("sealed state file is invalid: %s" % path.name)
    return document


def _verified_control(
        state_dir: Path, *, lease_dir: Optional[Path] = None,
        provider: Optional[str] = None,
        provider_account_id: Optional[str] = None) -> Dict[str, Any]:
    state = Path(os.path.abspath(str(state_dir)))
    manifest = _read_sealed_document(
        state / "reaper-control.json", CONTROL_SCHEMA)
    if str(state) != manifest.get("state_dir"):
        raise LeaseError("reaper control state directory mismatch")
    if (lease_dir is not None
            and str(Path(os.path.abspath(str(lease_dir))))
            != manifest.get("lease_dir")):
        raise LeaseError("reaper control lease directory mismatch")
    if provider is not None and str(provider) != manifest.get("provider"):
        raise LeaseError("reaper control provider mismatch")
    if (provider_account_id is not None
            and not secrets.compare_digest(
                str(provider_account_id),
                str(manifest.get("provider_account_id") or ""))):
        raise LeaseError("reaper control provider account mismatch")
    service = Path(manifest["service_unit_path"])
    timer = Path(manifest["timer_unit_path"])
    rebuilt, public = _control_documents(
        manifest["command_argv"], manifest["source_command_argv"],
        state, Path(manifest["lease_dir"]),
        manifest["provider"], manifest["provider_account_id"], service,
        timer, manifest["timer_interval_seconds"],
        [Path(value) for value in manifest["source_paths"]],
        [Path(value) for value in manifest["runtime_paths"]],
        manifest["interpreter"])
    if rebuilt["public_control"] != manifest.get("public_control"):
        raise LeaseError("reaper control files or command changed")
    for unit, digest_key in (
            (service, "service_unit_sha256"),
            (timer, "timer_unit_sha256")):
        fd = os.open(
            str(unit), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            unit_info = os.fstat(fd)
            if (not stat.S_ISREG(unit_info.st_mode)
                    or stat.S_IMODE(unit_info.st_mode) != 0o600
                    or (hasattr(os, "getuid")
                        and unit_info.st_uid != os.getuid())):
                raise LeaseError("reaper systemd unit is unsafe")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                unit_raw = stream.read()
        finally:
            if fd >= 0:
                os.close(fd)
        if hashlib.sha256(unit_raw).hexdigest() != public[digest_key]:
            raise LeaseError("installed reaper systemd unit differs from control")
    return public


def verify_reaper_runtime_invocation(
        state_dir: Path, *, lease_dir: Path,
        provider: str) -> Dict[str, Any]:
    """Require this process to be the exact installed isolated reaper command."""
    state = Path(os.path.abspath(str(state_dir)))
    control = _verified_control(
        state, lease_dir=lease_dir, provider=provider)
    manifest = _read_sealed_document(
        state / "reaper-control.json", CONTROL_SCHEMA)
    command = manifest.get("command_argv")
    if (not isinstance(command, list) or len(command) < 4
            or command[1:3] != ["-I", "-S"]):
        raise LeaseError("installed reaper command is invalid")
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if not isinstance(main_file, str) or not main_file:
        raise LeaseError("reaper process has no exact entrypoint")
    entry = Path(os.path.realpath(main_file))
    interpreter = Path(os.path.realpath(sys.executable))
    module = Path(os.path.realpath(__file__))
    expected_entry = Path(os.path.realpath(command[3]))
    expected_interpreter = Path(os.path.realpath(command[0]))
    expected_module = Path(
        os.path.realpath(
            str(Path(manifest["runtime_root"]) / "fidelity"
                / "cloudlease.py")))
    if entry != expected_entry:
        raise LeaseError(
            "reaper process is not the installed snapshot entrypoint")
    if interpreter != expected_interpreter:
        raise LeaseError(
            "reaper process is not using the installed trusted interpreter")
    if module != expected_module:
        raise LeaseError(
            "reaper process imported cloudlease outside the installed snapshot")
    if not sys.flags.isolated or not sys.flags.no_site:
        raise LeaseError("reaper process is not isolated with -I -S")
    observed_command = [str(entry)] + [str(item) for item in sys.argv[1:]]
    expected_command = [str(expected_entry)] + [
        str(item) for item in command[4:]]
    if observed_command != expected_command:
        raise LeaseError(
            "reaper process arguments differ from the installed service command")
    return control


def verify_reaper_control_account(
        state_dir: Path, *, lease_dir: Path, provider: str,
        provider_account_id: str) -> Dict[str, Any]:
    """Bind the exact installed reaper process to its configured account."""
    verify_reaper_runtime_invocation(
        state_dir, lease_dir=lease_dir, provider=provider)
    return _verified_control(
        state_dir, lease_dir=lease_dir, provider=provider,
        provider_account_id=provider_account_id)

def write_reaper_health(
        state_dir: Path, result: ReaperResult, *, lease_dir: Path,
        provider: str, provider_account_id: str,
        now: Optional[float] = None) -> Path:
    """Write a sweep stamp only from the exact installed reaper process."""
    state = Path(os.path.abspath(str(state_dir)))
    control = verify_reaper_control_account(
        state, lease_dir=lease_dir, provider=provider,
        provider_account_id=provider_account_id)
    completed = time.time() if now is None else float(now)
    document = {
        "schema": HEALTH_SCHEMA,
        "invocation_id": _PROCESS_INVOCATION_ID,
        "invocation_started_at_epoch": _PROCESS_STARTED_EPOCH,
        "invocation_started_at_utc": _utc_now(_PROCESS_STARTED_EPOCH),
        "completed_at_epoch": completed,
        "completed_at_utc": _utc_now(completed),
        "control": control,
        "actions": list(result.actions),
        "ok": result.ok,
        "failure_count": len(result.failures),
        "unresolved_count": len(result.unresolved),
        "result_sha256": _sha256(result.to_dict()),
    }
    path = state / "reaper-health.json"
    _atomic_replace(path, document)
    return path


def _login_linger_status(loginctl: str) -> Dict[str, Any]:
    argv = [loginctl, "show-user", str(os.getuid()),
            "-p", "Linger", "--value"]
    try:
        completed = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "value": None, "error": str(exc)}
    value = completed.stdout.strip()
    return {
        "ok": completed.returncode == 0 and value == "yes",
        "value": value if completed.returncode == 0 else None,
        "error": (None if completed.returncode == 0
                  else completed.stderr.strip() or "loginctl failed"),
    }


def _atomic_replace_text(path: Path, text: str) -> None:
    dir_fd = _safe_directory_fd(path.parent, create=True)
    tmp = None
    try:
        tmp = _atomic_temp(dir_fd, path.name, text.encode("utf-8"))
        os.replace(tmp, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        tmp = None
        os.fsync(dir_fd)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp, dir_fd=dir_fd)
            except OSError:
                pass
        os.close(dir_fd)


def install_systemd_user_timer(
        command: Sequence[str], *, lease_dir: Path,
        provider: str, provider_account_id: str,
        state_dir: Path = DEFAULT_STATE_DIR, interval_seconds: int = 300,
        systemctl: str = "systemctl", loginctl: str = "loginctl",
        unit_dir: Optional[Path] = None) -> Dict[str, str]:
    if isinstance(interval_seconds, bool):
        raise LeaseError("reaper interval must be an integer")
    try:
        interval = int(interval_seconds)
    except (TypeError, ValueError):
        raise LeaseError("reaper interval must be an integer")
    if interval < 30:
        raise LeaseError("reaper interval must be at least 30 seconds")
    linger = _login_linger_status(loginctl)
    if not linger["ok"]:
        raise LeaseError(
            "boot-persistent user manager is not proven; remedy: "
            "loginctl enable-linger $USER")
    state = Path(os.path.abspath(str(state_dir)))
    unit_dir = Path(os.path.abspath(str(
        Path.home() / ".config" / "systemd" / "user"
        if unit_dir is None else unit_dir)))
    service = unit_dir / "fidelity-cloud-reaper.service"
    timer = unit_dir / "fidelity-cloud-reaper.timer"
    if len(command) < 2:
        raise LeaseError("reaper source command is incomplete")
    source_command = [str(item) for item in command]
    closure = _source_closure(Path(source_command[1]))
    interpreter = _system_python_identity()
    runtime_root, source_paths, runtime_paths = _install_runtime_snapshot(
        state, closure)
    runtime_command = [
        interpreter["path"], "-I", "-S",
        str(runtime_root / "reap_cloud_leases.py"),
    ] + source_command[2:]
    internal, unused_public = _control_documents(
        runtime_command, source_command, state, lease_dir,
        provider, provider_account_id, service, timer, interval,
        source_paths, runtime_paths, interpreter)
    _atomic_replace_text(service, _service_text(runtime_command, state))
    _atomic_replace_text(timer, _timer_text(interval))
    _atomic_replace(state / "reaper-control.json", internal)
    for argv in (
            [systemctl, "--user", "daemon-reload"],
            [systemctl, "--user", "enable", "--now", timer.name],
            [systemctl, "--user", "start", service.name]):
        try:
            completed = subprocess.run(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LeaseError("%s failed: %s" % (" ".join(argv), exc))
        if completed.returncode != 0:
            raise LeaseError("%s failed: %s"
                             % (" ".join(argv), completed.stderr.strip()))
    return {"service": str(service), "timer": str(timer),
            "control": str(state / "reaper-control.json"),
            "health_stamp": str(state / "reaper-health.json")}


def _systemd_reaper_service_result(
        systemctl: str) -> Dict[str, Any]:
    argv = [
        systemctl, "--user", "show", "fidelity-cloud-reaper.service",
        "--property=Result", "--property=ExecMainStatus",
    ]
    try:
        completed = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False, "result": None, "exec_main_status": None,
            "error": str(exc),
        }
    properties: Dict[str, str] = {}
    malformed = False
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if (not separator or key not in ("Result", "ExecMainStatus")
                or key in properties):
            malformed = True
            continue
        properties[key] = value
    result = properties.get("Result")
    status = properties.get("ExecMainStatus")
    ok = (
        completed.returncode == 0 and not malformed
        and set(properties) == {"Result", "ExecMainStatus"}
        and result == "success" and status == "0")
    return {
        "ok": ok,
        "result": result,
        "exec_main_status": status,
        "error": (
            None if completed.returncode == 0
            else completed.stderr.strip() or "systemctl show failed"),
    }


def reaper_source_drift(state_dir: Path) -> Dict[str, Any]:
    """Report whether the checkout differs from the installed snapshot.

    Advisory only.  The timer executes the snapshot, which is what control
    seals, so drift never makes the reaper unhealthy and never stops a
    sweep; it means a newer reaper exists in the checkout than the one
    installed, and `reaper --install` would pick it up.  Any failure to read
    the checkout is reported as drift with its reason rather than raised.
    """
    state = Path(os.path.abspath(str(state_dir)))
    try:
        manifest = _read_sealed_document(
            state / "reaper-control.json", CONTROL_SCHEMA)
        installed = {
            row["path"]: row["sha256"]
            for row in manifest["public_control"]["runtime_files"]}
        checkout = {
            row["path"]: row["sha256"]
            for row in _control_file_rows(
                [Path(value) for value in manifest["source_paths"]],
                strict=False)}
    except (OSError, ValueError, KeyError, TypeError, LeaseError) as exc:
        return {"drift": True, "changed": [], "reason": str(exc)}
    changed = sorted(
        path for path in set(installed) | set(checkout)
        if installed.get(path) != checkout.get(path))
    return {"drift": bool(changed), "changed": changed, "reason": None}


def systemd_reaper_health(
        *, state_dir: Path, lease_dir: Path, provider: str,
        provider_account_id: str, max_age_seconds: int = 900,
        systemctl: str = "systemctl", loginctl: str = "loginctl",
        now: Optional[float] = None) -> Dict[str, Any]:
    checks = {}
    for operation in ("is-enabled", "is-active"):
        try:
            completed = subprocess.run(
                [systemctl, "--user", operation,
                 "fidelity-cloud-reaper.timer"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=30)
            wanted = "enabled" if operation == "is-enabled" else "active"
            checks[operation] = (
                completed.returncode == 0
                and completed.stdout.strip() == wanted)
        except (OSError, subprocess.TimeoutExpired):
            checks[operation] = False
    service_result = _systemd_reaper_service_result(systemctl)
    linger = _login_linger_status(loginctl)
    try:
        control = _verified_control(
            state_dir, lease_dir=lease_dir, provider=provider,
            provider_account_id=provider_account_id)
        stamp = _read_sealed_document(
            Path(os.path.abspath(str(state_dir))) / "reaper-health.json",
            HEALTH_SCHEMA)
        age = ((time.time() if now is None else float(now))
               - float(stamp["completed_at_epoch"]))
        control_ok = stamp.get("control") == control
        invocation = stamp.get("invocation_id")
        started = float(stamp["invocation_started_at_epoch"])
        completed_at = float(stamp["completed_at_epoch"])
        stamp_ok = (
            stamp.get("ok") is True and control_ok
            and isinstance(invocation, str) and len(invocation) == 32
            and all(char in "0123456789abcdef" for char in invocation)
            and 0 < started <= completed_at
            and 0 <= age <= int(max_age_seconds))
    except (OSError, ValueError, KeyError, TypeError, LeaseError):
        stamp, age, stamp_ok, control_ok = None, None, False, False
    result = {
        "ok": (
            all(checks.values()) and service_result["ok"]
            and linger["ok"] and stamp_ok),
        "timer": checks,
        "service_last_result": service_result,
        "user_manager_persistence": linger,
        "stamp_ok": stamp_ok,
        "control_ok": control_ok,
        "stamp_age_seconds": age,
        "stamp": stamp,
    }
    result["source_drift"] = reaper_source_drift(state_dir)
    return result
