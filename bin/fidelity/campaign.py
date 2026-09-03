#!/usr/bin/env python3
"""Fail-closed campaign admission and liability accounting.

The provider lifecycle and this ledger deliberately have different jobs.  The
lifecycle records what the provider did; this file answers whether the
campaign can afford one more attempt.  An attempt's liability remains in the
sum from reservation until BOTH exact deletion and billing reconciliation
have durable proof.

All money is serialized as base-ten strings and computed with ``Decimal``.
Every mutation holds one process lock and compares the caller's generation
before replacing the complete ledger atomically.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import json
import os
import re
import secrets
import stat
from typing import Any, Dict, Iterable, Optional, Tuple


SCHEMA = "campaign-ledger-v1"
CURRENCY = "USD"
RUNPOD_TARIFF_SOURCE = "https://docs.runpod.io/pods/pricing"
MAX_QUOTE_VALIDITY_SECONDS = 300
_JOB_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^[0-9a-f]{24}$")
_ALLOWED_PHASES = {
    "RESERVED", "CANCELLED_BEFORE_CREATE", "CREATING", "LIVE", "RUNNING",
    "EXITED", "TERMINATE_REQUESTED", "TERMINATE_REQUIRED", "DELETED",
    "RECONCILED",
}
_RESOURCE_FAMILIES = {"pods", "network_volumes"}


def _bootstrap_drill_blocked_by_prior_attempts(
        attempts: Iterable[Dict[str, Any]]) -> bool:
    """Return whether prior paid work forbids the bootstrap drill."""
    for item in attempts:
        if item.get("phase") == "CANCELLED_BEFORE_CREATE":
            # Nothing was ever created and the reservation is released, so
            # this attempt did no paid work and cannot order anything. The
            # exemption used to apply only to drills, purely because the
            # measurement test came first: on 2026-09-03 a measurement whose
            # create RunPod refused with SUPPLY_CONSTRAINT -- $0.00 spent --
            # permanently blocked re-proving the controller.
            continue
        if item.get("reservation_kind") == "measurement":
            return True
        if (item.get("reservation_kind")
                != "bootstrap-controller-loss-drill"):
            continue
        if (item.get("phase") == "RECONCILED"
                and item.get("released") is True
                and item.get("maximum_remaining_liability_usd") == "0"
                and isinstance(item.get("billing"), dict)
                and isinstance(item.get("deletion"), dict)):
            continue
        return True
    return False


class CampaignLedgerError(RuntimeError):
    """The durable ledger is missing, malformed, or inconsistent."""

def _reject_nonfinite_token(value: str) -> None:
    raise ValueError("non-finite JSON number: %s" % value)

def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: %s" % key)
        result[key] = value
    return result



def _decimal(value: Any, name: str, *, allow_none: bool = False) -> Optional[Decimal]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError("%s must be a decimal, not boolean" % name)
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("%s is not a decimal" % name)
    if not result.is_finite() or result < 0:
        raise ValueError("%s must be finite and non-negative" % name)
    return result


def _positive(value: Any, name: str) -> Decimal:
    result = _decimal(value, name)
    if result <= 0:
        raise ValueError("%s must be positive" % name)
    return result


def _money(value: Decimal) -> str:
    """Canonical non-exponent decimal spelling used by the on-disk schema."""
    value = _decimal(value, "money")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _timestamp(value: str, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty RFC3339 timestamp" % name)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError("%s must be an RFC3339 timestamp" % name)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("%s must include a timezone" % name)
    return parsed.astimezone(timezone.utc)


def _require_exact_keys(doc: Dict[str, Any], expected: Iterable[str], name: str) -> None:
    if not isinstance(doc, dict):
        raise ValueError("%s must be an object" % name)
    expected = set(expected)
    actual = set(doc)
    if actual != expected:
        raise ValueError("%s keys differ (missing=%s unexpected=%s)" % (
            name, sorted(expected - actual), sorted(actual - expected)))


def attempt_key(job_hash: str, attempt: str) -> str:
    """Return the canonical durable key for one random campaign attempt."""
    if not isinstance(job_hash, str) or not _JOB_HASH_RE.fullmatch(job_hash):
        raise ValueError("job_hash must be the full lowercase SHA-256")
    if not isinstance(attempt, str) or not _ATTEMPT_RE.fullmatch(attempt):
        raise ValueError("attempt must be a lowercase random 96-bit hex value")
    return "%s:%s" % (job_hash, attempt)


@dataclass(frozen=True)
class CostQuote:
    """Immutable, all-in upper-bound evidence for one provider attempt.

    RunPod publishes storage prices per GB-month.  ``storage_month_hours`` is
    therefore part of the quote rather than an implicit 730-hour convention;
    callers should use a conservative denominator (for example 672 hours).
    Compute and local storage accrue per second.  Network volume would accrue
    in hourly increments, but the first safe profile refuses any non-zero
    network volume explicitly.
    """

    reserved_compute_usd_per_hour: Decimal
    live_compute_usd_per_hour: Decimal
    container_disk_size_gb: Optional[Decimal]
    container_disk_running_usd_per_gb_month: Optional[Decimal]
    container_disk_stopped_usd_per_gb_month: Optional[Decimal]
    pod_disk_size_gb: Optional[Decimal]
    pod_disk_running_usd_per_gb_month: Optional[Decimal]
    pod_disk_stopped_usd_per_gb_month: Optional[Decimal]
    network_volume_size_gb: Optional[Decimal]
    network_volume_usd_per_gb_month: Optional[Decimal]
    storage_month_hours: Decimal
    network_billing_increment_seconds: Decimal
    tariff_source: str
    tariff_effective_at: str
    quoted_at: str
    valid_until: str
    target: str
    profile: str
    timing_kind: str
    timing_evidence: str
    workload_deadline_seconds: Decimal
    provider_termination_deadline_seconds: Decimal
    retrieval_delete_reserve_seconds: Decimal
    timer_api_lag_seconds: Decimal
    hard_cap_usd: Decimal

    _FIELDS = (
        "reserved_compute_usd_per_hour", "live_compute_usd_per_hour",
        "container_disk_size_gb", "container_disk_running_usd_per_gb_month",
        "container_disk_stopped_usd_per_gb_month", "pod_disk_size_gb",
        "pod_disk_running_usd_per_gb_month", "pod_disk_stopped_usd_per_gb_month",
        "network_volume_size_gb", "network_volume_usd_per_gb_month",
        "storage_month_hours", "network_billing_increment_seconds",
        "tariff_source", "tariff_effective_at", "quoted_at", "valid_until",
        "target", "profile", "timing_kind", "timing_evidence",
        "workload_deadline_seconds", "provider_termination_deadline_seconds",
        "retrieval_delete_reserve_seconds", "timer_api_lag_seconds",
        "hard_cap_usd",
    )
    _DECIMAL_FIELDS = {
        "reserved_compute_usd_per_hour", "live_compute_usd_per_hour",
        "container_disk_size_gb", "container_disk_running_usd_per_gb_month",
        "container_disk_stopped_usd_per_gb_month", "pod_disk_size_gb",
        "pod_disk_running_usd_per_gb_month", "pod_disk_stopped_usd_per_gb_month",
        "network_volume_size_gb", "network_volume_usd_per_gb_month",
        "storage_month_hours", "network_billing_increment_seconds",
        "workload_deadline_seconds", "provider_termination_deadline_seconds",
        "retrieval_delete_reserve_seconds", "timer_api_lag_seconds",
        "hard_cap_usd",
    }
    _OPTIONAL_DECIMAL_FIELDS = {
        "container_disk_size_gb", "container_disk_running_usd_per_gb_month",
        "container_disk_stopped_usd_per_gb_month", "pod_disk_size_gb",
        "pod_disk_running_usd_per_gb_month", "pod_disk_stopped_usd_per_gb_month",
        "network_volume_size_gb", "network_volume_usd_per_gb_month",
    }

    def __post_init__(self) -> None:
        for field in self._DECIMAL_FIELDS:
            converted = _decimal(getattr(self, field), field,
                                 allow_none=field in self._OPTIONAL_DECIMAL_FIELDS)
            object.__setattr__(self, field, converted)
        if self.storage_month_hours <= 0:
            raise ValueError("storage_month_hours must be positive")
        if self.network_billing_increment_seconds <= 0:
            raise ValueError("network_billing_increment_seconds must be positive")
        for name in ("tariff_source", "target", "profile", "timing_evidence"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("%s must be non-empty" % name)
        if self.timing_kind not in ("exact-target-profile", "named-conservative-bound"):
            raise ValueError("timing_kind must name exact evidence or a conservative bound")
        effective = _timestamp(self.tariff_effective_at, "tariff_effective_at")
        quoted = _timestamp(self.quoted_at, "quoted_at")
        valid = _timestamp(self.valid_until, "valid_until")
        if effective > quoted:
            raise ValueError("tariff_effective_at is after quoted_at")
        if valid < quoted:
            raise ValueError("valid_until is before quoted_at")
        if (valid - quoted).total_seconds() > MAX_QUOTE_VALIDITY_SECONDS:
            raise ValueError(
                "quote validity exceeds %d seconds"
                % MAX_QUOTE_VALIDITY_SECONDS)
        if self.workload_deadline_seconds <= 0:
            raise ValueError("workload_deadline_seconds must be positive")
        if self.provider_termination_deadline_seconds <= 0:
            raise ValueError("provider_termination_deadline_seconds must be positive")
        if (self.provider_termination_deadline_seconds
                < self.workload_deadline_seconds
                  + self.retrieval_delete_reserve_seconds):
            raise ValueError(
                "provider_termination_deadline_seconds must cover workload "
                "deadline plus retrieval/delete reserve")
        if self.hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be positive")
        if self.storage_known and self.hard_cap_usd < self.calculated_maximum_usd():
            raise ValueError("hard_cap_usd is below the all-in calculated maximum")

    @property
    def storage_known(self) -> bool:
        groups = (
            (self.container_disk_size_gb,
             self.container_disk_running_usd_per_gb_month,
             self.container_disk_stopped_usd_per_gb_month),
            (self.pod_disk_size_gb, self.pod_disk_running_usd_per_gb_month,
             self.pod_disk_stopped_usd_per_gb_month),
            (self.network_volume_size_gb, self.network_volume_usd_per_gb_month),
        )
        return all(all(value is not None for value in group) for group in groups)

    @property
    def duration_seconds(self) -> Decimal:
        # The lease reap deadline is absolute from create, not a phase duration;
        # the same timestamp is also sent as an untrusted provider timer hint.
        # Workload and retrieval/delete fit inside it.  API/reaper settlement
        # lag is the conservative exposure beyond the enforced deadline.
        return self.provider_termination_deadline_seconds + self.timer_api_lag_seconds

    def all_in_hourly_rate(self) -> Decimal:
        if not self.storage_known:
            raise ValueError("storage sizes/rates are unknown")
        storage = (
            self.container_disk_size_gb * max(
                self.container_disk_running_usd_per_gb_month,
                self.container_disk_stopped_usd_per_gb_month)
            + self.pod_disk_size_gb * max(
                self.pod_disk_running_usd_per_gb_month,
                self.pod_disk_stopped_usd_per_gb_month)
            + self.network_volume_size_gb * self.network_volume_usd_per_gb_month
        ) / self.storage_month_hours
        return max(self.reserved_compute_usd_per_hour,
                   self.live_compute_usd_per_hour) + storage

    def calculated_maximum_usd(self) -> Decimal:
        if not self.storage_known:
            return Decimal("Infinity")
        compute_and_local = (
            max(self.reserved_compute_usd_per_hour,
                self.live_compute_usd_per_hour)
            + (self.container_disk_size_gb * max(
                   self.container_disk_running_usd_per_gb_month,
                   self.container_disk_stopped_usd_per_gb_month)
               + self.pod_disk_size_gb * max(
                   self.pod_disk_running_usd_per_gb_month,
                   self.pod_disk_stopped_usd_per_gb_month))
              / self.storage_month_hours
        ) * self.duration_seconds / Decimal(3600)
        network_hourly = (self.network_volume_size_gb
                          * self.network_volume_usd_per_gb_month
                          / self.storage_month_hours)
        network_hours = (self.duration_seconds / self.network_billing_increment_seconds
                         ).to_integral_value(rounding=ROUND_CEILING)
        network = network_hourly * (
            self.network_billing_increment_seconds / Decimal(3600)) * network_hours
        return compute_and_local + network

    def stale_at(self, now: str) -> bool:
        return _timestamp(now, "now") > _timestamp(self.valid_until, "valid_until")

    def future_at(self, now: str) -> bool:
        return _timestamp(now, "now") < _timestamp(self.quoted_at, "quoted_at")

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for field in self._FIELDS:
            value = getattr(self, field)
            result[field] = (_money(value) if field in self._DECIMAL_FIELDS
                             and value is not None else value)
        return result

    @classmethod
    def from_dict(cls, doc: Dict[str, Any]) -> "CostQuote":
        _require_exact_keys(doc, cls._FIELDS, "cost quote")
        return cls(**doc)


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    code: str
    message: str
    generation: int
    attempt_key: Optional[str]
    action: str
    maximum_committed_usd: Decimal
    admission_limit_usd: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admitted": self.admitted,
            "code": self.code,
            "message": self.message,
            "generation": self.generation,
            "attempt_key": self.attempt_key,
            "action": self.action,
            "maximum_committed_usd": _money(self.maximum_committed_usd),
            "admission_limit_usd": _money(self.admission_limit_usd),
        }


@dataclass(frozen=True)
class TransitionResult:
    applied: bool
    code: str
    message: str
    generation: int
    attempt_key: str
    action: str = "NONE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "code": self.code,
            "message": self.message,
            "generation": self.generation,
            "attempt_key": self.attempt_key,
            "action": self.action,
        }


class CampaignLedger:
    """Locked file-backed campaign ledger.

    Use :meth:`create` once, then record a fresh balance/inventory snapshot
    before asking for admission.  Callers must pass the generation they read;
    a stale controller receives ``GENERATION_CONFLICT`` and must re-read.
    """

    _TOP_KEYS = {
        "schema", "generation", "currency", "provider", "provider_account_id",
        "hard_ceiling_usd", "reserve_floor_usd", "cleanup_reaper_margin_usd",
        "max_concurrent_attempts", "authorized_concurrent_attempts",
        "width_authorization", "settled_charges_usd",
        "balance", "inventory", "attempts",
    }

    def __init__(self, path: str, provider: str, provider_account_id: str):
        if (not isinstance(path, str) or not os.path.isabs(path)
                or os.path.normpath(path) != path):
            raise CampaignLedgerError(
                "campaign ledger path must be canonical and absolute")
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be non-empty")
        if not isinstance(provider_account_id, str) or not provider_account_id:
            raise ValueError("provider_account_id must be non-empty")
        self.path = path
        self.directory = os.path.dirname(path)
        self.leaf = os.path.basename(path)
        self.lock_leaf = self.leaf + ".lock"
        self.provider = provider
        self.provider_account_id = provider_account_id
        if not self.leaf or self.leaf in (".", ".."):
            raise CampaignLedgerError("campaign ledger filename is invalid")
        parent_fd = self._open_safe_parent(create=True)
        os.close(parent_fd)

    def _open_safe_parent(self, create: bool) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        fd = os.open("/", flags)
        try:
            for component in self.directory.split(os.sep):
                if not component:
                    continue
                if create:
                    try:
                        os.mkdir(component, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(component, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            info = os.fstat(fd)
            if (not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o022):
                raise CampaignLedgerError(
                    "campaign ledger parent must be owner-controlled and not writable "
                    "by group/other")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _validate_money_file(fd: int, label: str, mode: int = 0o600) -> None:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != mode):
            raise CampaignLedgerError(
                "%s must be an owner-controlled regular file with mode %04o"
                % (label, mode))

    def _ledger_exists_unlocked(self) -> bool:
        parent_fd = self._open_safe_parent(create=False)
        try:
            try:
                info = os.stat(self.leaf, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600):
                raise CampaignLedgerError(
                    "existing campaign ledger is unsafe (symlink/owner/mode)")
            return True
        finally:
            os.close(parent_fd)

    def _write_unlocked(self, doc: Dict[str, Any]) -> None:
        parent_fd = self._open_safe_parent(create=False)
        temp_leaf = None
        handle = None
        try:
            if self._ledger_exists_unlocked():
                # `_ledger_exists_unlocked` performs the no-follow owner/mode check.
                pass
            for _unused in range(32):
                candidate = ".campaign-%s.tmp" % secrets.token_hex(12)
                try:
                    handle = os.open(
                        candidate,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | os.O_CLOEXEC | os.O_NOFOLLOW,
                        0o600, dir_fd=parent_fd)
                    temp_leaf = candidate
                    break
                except FileExistsError:
                    continue
            if handle is None:
                raise CampaignLedgerError("could not allocate campaign temporary file")
            self._validate_money_file(handle, "campaign temporary file")
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                handle = None
                json.dump(doc, fh, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(
                temp_leaf, self.leaf,
                src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            temp_leaf = None
            os.fsync(parent_fd)
        finally:
            if handle is not None:
                os.close(handle)
            if temp_leaf is not None:
                try:
                    os.unlink(temp_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)

    @classmethod
    def _new_campaign_document(
            cls, hard_ceiling_usd: Any, reserve_floor_usd: Any,
            cleanup_reaper_margin_usd: Any, max_concurrent_attempts: int,
            provider: str, provider_account_id: str,
            currency: str = CURRENCY) -> Dict[str, Any]:
        ceiling = _positive(hard_ceiling_usd, "hard_ceiling_usd")
        floor = _decimal(reserve_floor_usd, "reserve_floor_usd")
        margin = _decimal(cleanup_reaper_margin_usd, "cleanup_reaper_margin_usd")
        if (isinstance(max_concurrent_attempts, bool)
                or not isinstance(max_concurrent_attempts, int)
                or max_concurrent_attempts not in (1, 2)):
            raise ValueError("max_concurrent_attempts must be 1 or 2")
        if currency != CURRENCY:
            raise ValueError("campaign-ledger-v1 supports USD only")
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be non-empty")
        if not isinstance(provider_account_id, str) or not provider_account_id:
            raise ValueError("provider_account_id must be non-empty")
        doc = {
            "schema": SCHEMA,
            "generation": 0,
            "currency": currency,
            "provider": provider,
            "provider_account_id": provider_account_id,
            "hard_ceiling_usd": _money(ceiling),
            "reserve_floor_usd": _money(floor),
            "cleanup_reaper_margin_usd": _money(margin),
            "max_concurrent_attempts": max_concurrent_attempts,
            "authorized_concurrent_attempts": 1,
            "width_authorization": None,
            "settled_charges_usd": "0",
            "balance": None,
            "inventory": None,
            "attempts": {},
        }
        cls._validate_document(doc)
        return doc

    @classmethod
    def create(cls, path: str, hard_ceiling_usd: Any, reserve_floor_usd: Any,
               cleanup_reaper_margin_usd: Any, max_concurrent_attempts: int,
               provider: str, provider_account_id: str,
               currency: str = CURRENCY) -> "CampaignLedger":
        ledger = cls(path, provider, provider_account_id)
        new_doc = cls._new_campaign_document(
            hard_ceiling_usd, reserve_floor_usd, cleanup_reaper_margin_usd,
            max_concurrent_attempts, provider, provider_account_id, currency)
        expected = (
            Decimal(new_doc["hard_ceiling_usd"]),
            Decimal(new_doc["reserve_floor_usd"]),
            Decimal(new_doc["cleanup_reaper_margin_usd"]),
            new_doc["max_concurrent_attempts"], new_doc["currency"],
            new_doc["provider"], new_doc["provider_account_id"])
        with ledger._locked(exclusive=True):
            if ledger._ledger_exists_unlocked():
                current = ledger._read_unlocked()
                actual = (Decimal(current["hard_ceiling_usd"]),
                          Decimal(current["reserve_floor_usd"]),
                          Decimal(current["cleanup_reaper_margin_usd"]),
                          current["max_concurrent_attempts"],
                          current["currency"], current["provider"],
                          current["provider_account_id"])
                if actual != expected:
                    raise CampaignLedgerError(
                        "existing ledger campaign identity does not match")
            else:
                ledger._write_unlocked(new_doc)
        return ledger

    @contextmanager
    def _locked(self, exclusive: bool):
        parent_fd = self._open_safe_parent(create=False)
        lock_fd = None
        try:
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
            try:
                lock_fd = os.open(
                    self.lock_leaf, flags, 0o600, dir_fd=parent_fd)
            except OSError as exc:
                raise CampaignLedgerError("unsafe campaign lock path: %s" % exc)
            self._validate_money_file(lock_fd, "campaign lock")
            os.fsync(parent_fd)
            fcntl.flock(
                lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(parent_fd)

    def _read_unlocked(self) -> Dict[str, Any]:
        parent_fd = self._open_safe_parent(create=False)
        fd = None
        try:
            try:
                fd = os.open(
                    self.leaf, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd)
            except OSError as exc:
                raise CampaignLedgerError(
                    "campaign ledger cannot be opened safely: %s" % exc)
            self._validate_money_file(fd, "campaign ledger")
            with os.fdopen(fd, "r", encoding="utf-8") as fh:
                fd = None
                doc = json.load(
                    fh, parse_constant=_reject_nonfinite_token,
                    object_pairs_hook=_reject_duplicate_pairs)
            self._validate_document(doc)
            if (doc["provider"] != self.provider
                    or doc["provider_account_id"] != self.provider_account_id):
                raise CampaignLedgerError(
                    "campaign provider/account identity mismatch")
            return doc
        except CampaignLedgerError:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CampaignLedgerError("invalid campaign ledger: %s" % exc)
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    @classmethod
    def _validate_document(cls, doc: Dict[str, Any]) -> None:
        _require_exact_keys(doc, cls._TOP_KEYS, "campaign ledger")
        if doc["schema"] != SCHEMA or doc["currency"] != CURRENCY:
            raise ValueError("unsupported campaign ledger schema or currency")
        if not isinstance(doc["provider"], str) or not doc["provider"]:
            raise ValueError("campaign provider is empty")
        if (not isinstance(doc["provider_account_id"], str)
                or not doc["provider_account_id"]):
            raise ValueError("campaign provider_account_id is empty")
        if isinstance(doc["generation"], bool) or not isinstance(doc["generation"], int):
            raise ValueError("ledger generation must be an integer")
        if doc["generation"] < 0:
            raise ValueError("ledger generation is negative")
        _positive(doc["hard_ceiling_usd"], "hard_ceiling_usd")
        _decimal(doc["reserve_floor_usd"], "reserve_floor_usd")
        _decimal(doc["cleanup_reaper_margin_usd"], "cleanup_reaper_margin_usd")
        maximum_width = doc["max_concurrent_attempts"]
        authorized_width = doc["authorized_concurrent_attempts"]
        if (isinstance(maximum_width, bool) or maximum_width not in (1, 2)):
            raise ValueError("max_concurrent_attempts must be 1 or 2")
        if (isinstance(authorized_width, bool)
                or not isinstance(authorized_width, int)
                or authorized_width not in (1, 2)
                or authorized_width > maximum_width):
            raise ValueError("authorized_concurrent_attempts is invalid")
        authorization = doc["width_authorization"]
        if authorization is None:
            if authorized_width != 1:
                raise ValueError("campaign width exceeds one without authorization")
        else:
            _require_exact_keys(
                authorization,
                ("authorized_at", "from_width", "to_width",
                 "fruit_public_archive_sha256", "fruit_proof_sha256"),
                "width_authorization")
            _timestamp(authorization["authorized_at"], "width_authorization.authorized_at")
            if (authorization["from_width"] != 1
                    or authorization["to_width"] != 2
                    or maximum_width != 2 or authorized_width != 2):
                raise ValueError("width authorization is not the monotonic 1-to-2 transition")
            for field in ("fruit_public_archive_sha256", "fruit_proof_sha256"):
                value = authorization[field]
                if (not isinstance(value, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", value)):
                    raise ValueError("%s is not a SHA-256 digest" % field)
        _decimal(doc["settled_charges_usd"], "settled_charges_usd")
        if doc["balance"] is not None:
            _require_exact_keys(
                doc["balance"],
                ("provider", "provider_account_id", "available_usd",
                 "observed_at", "valid_until", "source"),
                "balance")
            if (doc["balance"]["provider"] != doc["provider"]
                    or doc["balance"]["provider_account_id"]
                    != doc["provider_account_id"]):
                raise ValueError("balance provider/account identity mismatch")
            _decimal(doc["balance"]["available_usd"], "available_usd")
            observed = _timestamp(doc["balance"]["observed_at"], "balance.observed_at")
            valid = _timestamp(doc["balance"]["valid_until"], "balance.valid_until")
            if valid < observed:
                raise ValueError("balance validity predates observation")
            if not isinstance(doc["balance"]["source"], str) or not doc["balance"]["source"]:
                raise ValueError("balance source is empty")
        resource_keys = set()
        unknown_keys = set()
        if doc["inventory"] is not None:
            _require_exact_keys(
                doc["inventory"],
                ("provider", "provider_account_id", "observed_at", "valid_until",
                 "complete", "provider_resources", "unknown_resources", "source"),
                "inventory")
            if (doc["inventory"]["provider"] != doc["provider"]
                    or doc["inventory"]["provider_account_id"]
                    != doc["provider_account_id"]):
                raise ValueError("inventory provider/account identity mismatch")
            observed = _timestamp(
                doc["inventory"]["observed_at"], "inventory.observed_at")
            valid = _timestamp(
                doc["inventory"]["valid_until"], "inventory.valid_until")
            if valid < observed:
                raise ValueError("inventory validity predates observation")
            if not isinstance(doc["inventory"]["complete"], bool):
                raise ValueError("inventory complete must be boolean")
            resources = doc["inventory"]["provider_resources"]
            if not isinstance(resources, list):
                raise ValueError("provider_resources must be a list")
            prior_key = None
            for index, resource in enumerate(resources):
                _require_exact_keys(
                    resource, ("family", "id", "name", "status"),
                    "provider resource %d" % index)
                if resource["family"] not in _RESOURCE_FAMILIES:
                    raise ValueError("provider resource family is unsupported")
                if any(not isinstance(resource[field], str) or not resource[field]
                       for field in ("id", "name", "status")):
                    raise ValueError("provider resource fields must be non-empty strings")
                resource_key = (resource["family"], resource["id"])
                if prior_key is not None and resource_key <= prior_key:
                    raise ValueError(
                        "provider_resources must be sorted by unique family/ID")
                prior_key = resource_key
                resource_keys.add(resource_key)
            unknown = doc["inventory"]["unknown_resources"]
            if not isinstance(unknown, list):
                raise ValueError("unknown_resources must be a list")
            prior_unknown = None
            unknown_keys = set()
            for index, item in enumerate(unknown):
                _require_exact_keys(
                    item, ("family", "id"), "unknown resource %d" % index)
                if (item["family"] not in _RESOURCE_FAMILIES
                        or not isinstance(item["id"], str) or not item["id"]):
                    raise ValueError("unknown resource family/ID is invalid")
                unknown_key = (item["family"], item["id"])
                if prior_unknown is not None and unknown_key <= prior_unknown:
                    raise ValueError(
                        "unknown_resources must be sorted by unique family/ID")
                prior_unknown = unknown_key
                unknown_keys.add(unknown_key)
            if not unknown_keys.issubset(resource_keys):
                raise ValueError("unknown_resources contains an unobserved provider key")
            if not isinstance(doc["inventory"]["source"], str) or not doc["inventory"]["source"]:
                raise ValueError("inventory source is empty")
        if not isinstance(doc["attempts"], dict):
            raise ValueError("attempts must be an object")
        active_provider_ids = set()
        known_provider_ids = set()
        for key, attempt in doc["attempts"].items():
            cls._validate_attempt(key, attempt)
            if attempt["released"]:
                continue
            known_provider_ids.update(attempt["provider_ids"])
            deleted_ids = set(
                (attempt["deletion"] or {"proofs": {}})["proofs"])
            active_provider_ids.update(
                provider_id for provider_id in attempt["provider_ids"]
                if provider_id not in deleted_ids)
        expected_unknown_keys = {
            resource_key for resource_key in resource_keys
            if resource_key[0] != "pods"
            or resource_key[1] not in known_provider_ids
        }
        if unknown_keys != expected_unknown_keys:
            raise ValueError(
                "unknown_resources is not the canonical campaign classification")
        if (doc["inventory"] is not None
                and not {("pods", provider_id)
                         for provider_id in active_provider_ids}.issubset(
                             resource_keys)):
            raise ValueError(
                "active campaign pod IDs are absent from inventory snapshot")

    @staticmethod
    def _validate_attempt(key: str, attempt: Dict[str, Any]) -> None:
        fields = {
            "job_hash", "attempt", "reservation_kind", "phase", "provider_ids",
            "cleanup_binding_evidence", "precreate_cancellation",
            "reserved_at", "reserved_quote", "actual_quote",
            "maximum_remaining_liability_usd", "deletion", "billing",
            "released", "admission_freeze_reason",
        }
        _require_exact_keys(attempt, fields, "attempt %s" % key)
        if key != attempt_key(attempt["job_hash"], attempt["attempt"]):
            raise ValueError("attempt key does not match its identities")
        if attempt["reservation_kind"] not in (
                "measurement", "bootstrap-controller-loss-drill"):
            raise ValueError("attempt reservation_kind is invalid")
        if attempt["reservation_kind"] == "bootstrap-controller-loss-drill":
            bootstrap_quote = CostQuote.from_dict(attempt["reserved_quote"])
            if (bootstrap_quote.profile != "runpod-drill-secure-l4-on-demand"
                    or bootstrap_quote.timing_kind != "exact-target-profile"
                    or bootstrap_quote.tariff_source != RUNPOD_TARIFF_SOURCE
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", bootstrap_quote.timing_evidence)
                    or bootstrap_quote.live_compute_usd_per_hour
                       != bootstrap_quote.reserved_compute_usd_per_hour):
                raise ValueError("bootstrap drill quote identity is invalid")
        if attempt["phase"] not in _ALLOWED_PHASES:
            raise ValueError("unknown attempt phase")
        _timestamp(attempt["reserved_at"], "reserved_at")
        reserved_quote = CostQuote.from_dict(attempt["reserved_quote"])
        actual_quote = None
        if attempt["actual_quote"] is not None:
            actual_quote = CostQuote.from_dict(attempt["actual_quote"])
        provider_ids = attempt["provider_ids"]
        if (not isinstance(provider_ids, list)
                or any(not isinstance(value, str) or not value
                       for value in provider_ids)
                or provider_ids != sorted(set(provider_ids))):
            raise ValueError("provider_ids must be sorted unique non-empty strings")
        cleanup_evidence = attempt["cleanup_binding_evidence"]
        if cleanup_evidence is not None and (
                not isinstance(cleanup_evidence, str) or not cleanup_evidence):
            raise ValueError("cleanup_binding_evidence must be null or non-empty")
        if cleanup_evidence is not None and not provider_ids:
            raise ValueError("cleanup binding has no provider IDs")
        if attempt["actual_quote"] is not None and (
                len(provider_ids) != 1 or cleanup_evidence is not None):
            raise ValueError("normal actual quote must bind exactly one provider ID")
        if provider_ids and attempt["actual_quote"] is None and cleanup_evidence is None:
            raise ValueError("bound provider IDs lack normal or cleanup binding evidence")
        if cleanup_evidence is not None and attempt["phase"] not in (
                "TERMINATE_REQUIRED", "TERMINATE_REQUESTED",
                "DELETED", "RECONCILED"):
            raise ValueError("cleanup-only provider binding entered a runnable phase")
        if attempt["phase"] in ("LIVE", "RUNNING", "EXITED") and (
                attempt["actual_quote"] is None):
            raise ValueError("runnable attempt lacks an actual quote")
        remaining = _decimal(attempt["maximum_remaining_liability_usd"],
                             "maximum_remaining_liability_usd")
        if not isinstance(attempt["released"], bool):
            raise ValueError("released must be boolean")
        if attempt["released"] != (remaining == 0):
            raise ValueError("released flag and remaining liability disagree")
        if attempt["admission_freeze_reason"] is not None and (
                not isinstance(attempt["admission_freeze_reason"], str)
                or not attempt["admission_freeze_reason"]):
            raise ValueError("admission_freeze_reason must be null or non-empty")
        cancelled = attempt["precreate_cancellation"] is not None
        if cancelled:
            _require_exact_keys(
                attempt["precreate_cancellation"],
                ("cancelled_at", "campaign_phase_before_cancel",
                 "lease_state", "no_create_evidence"),
                "precreate_cancellation")
            cancellation = attempt["precreate_cancellation"]
            _timestamp(
                cancellation["cancelled_at"],
                "precreate_cancellation.cancelled_at")
            origin = cancellation["campaign_phase_before_cancel"]
            lease_state = cancellation["lease_state"]
            if origin not in ("RESERVED", "CREATING"):
                raise ValueError("invalid campaign cancellation origin")
            if (lease_state not in (
                    "LEASE_ABSENT", "PREPARED", "PROVIDER_REJECTED_CREATE")
                    or (origin == "CREATING"
                        and lease_state not in (
                            "PREPARED", "PROVIDER_REJECTED_CREATE"))):
                raise ValueError(
                    "cancellation evidence does not prove that no provider "
                    "resource was ever accepted")
            cancellation_evidence = cancellation["no_create_evidence"]
            if not isinstance(cancellation_evidence, str) or not cancellation_evidence:
                raise ValueError("pre-create cancellation evidence is empty")
            if (attempt["phase"] != "CANCELLED_BEFORE_CREATE" or provider_ids
                    or attempt["actual_quote"] is not None
                    or cleanup_evidence is not None
                    or attempt["deletion"] is not None
                    or attempt["billing"] is not None):
                raise ValueError(
                    "pre-create cancellation is valid only before provider activity")

        deletion_ids = set()
        if attempt["deletion"] is not None:
            _require_exact_keys(attempt["deletion"], ("proofs",), "deletion")
            proofs = attempt["deletion"]["proofs"]
            if not isinstance(proofs, dict):
                raise ValueError("deletion proofs must be an object")
            deletion_ids = set(proofs)
            if not deletion_ids.issubset(set(provider_ids)):
                raise ValueError("deletion proof names an unbound provider ID")
            for provider_id, proof_doc in proofs.items():
                _require_exact_keys(
                    proof_doc, ("deleted_at", "proof"),
                    "deletion proof %s" % provider_id)
                _timestamp(proof_doc["deleted_at"], "deleted_at")
                if not isinstance(proof_doc["proof"], str) or not proof_doc["proof"]:
                    raise ValueError("deletion proof is empty")

        billing_complete = False
        if attempt["billing"] is not None:
            _require_exact_keys(
                attempt["billing"],
                ("provider_ids", "reconciled_at", "final_charge_usd", "proof"),
                "billing")
            billing_ids = attempt["billing"]["provider_ids"]
            if (not isinstance(billing_ids, list)
                    or billing_ids != sorted(set(billing_ids))
                    or any(not isinstance(value, str) or not value
                           for value in billing_ids)):
                raise ValueError("billing provider_ids must be sorted unique strings")
            if billing_ids != provider_ids:
                raise ValueError("billing evidence must reconcile the full provider ID set")
            _timestamp(attempt["billing"]["reconciled_at"], "reconciled_at")
            final_charge = _decimal(
                attempt["billing"]["final_charge_usd"], "final_charge_usd")
            if (final_charge < 0
                    or final_charge > reserved_quote.hard_cap_usd
                    or (actual_quote is not None
                        and final_charge > actual_quote.hard_cap_usd)):
                raise ValueError(
                    "final charge is negative or exceeds an attempt hard cap")
            if (not isinstance(attempt["billing"]["proof"], str)
                    or not attempt["billing"]["proof"]):
                raise ValueError("billing proof is empty")
            billing_complete = True

        complete = (bool(provider_ids)
                    and deletion_ids == set(provider_ids)
                    and billing_complete)
        releasable = complete or cancelled
        if attempt["released"] != releasable:
            raise ValueError(
                "liability may release only after pre-create cancellation or "
                "every bound provider ID has deletion and billing proof")
        if complete and attempt["phase"] != "RECONCILED":
            raise ValueError("fully reconciled attempt has the wrong phase")

    def snapshot(self) -> Dict[str, Any]:
        with self._locked(exclusive=False):
            return self._read_unlocked()

    @staticmethod
    def _conflict(doc: Dict[str, Any], key: str = "") -> TransitionResult:
        return TransitionResult(False, "GENERATION_CONFLICT",
                                "ledger changed; re-read before retrying",
                                doc["generation"], key)

    def _commit(self, doc: Dict[str, Any]) -> int:
        doc["generation"] += 1
        self._validate_document(doc)
        self._write_unlocked(doc)
        return doc["generation"]

    @staticmethod
    def _known_pod_ids(doc: Dict[str, Any]) -> set:
        return {
            provider_id
            for attempt in doc["attempts"].values()
            if not attempt["released"]
            for provider_id in attempt["provider_ids"]
        }

    @staticmethod
    def _canonical_provider_resources(
            provider_resources: Iterable[Dict[str, str]],
            known_pod_ids: Iterable[str]) -> Tuple[list, list]:
        if isinstance(provider_resources, (str, bytes, dict)):
            raise ValueError("provider_resources must be an iterable of objects")
        resources = []
        for index, resource in enumerate(provider_resources):
            if not isinstance(resource, dict):
                raise ValueError("provider resource %d must be an object" % index)
            _require_exact_keys(
                resource, ("family", "id", "name", "status"),
                "provider resource %d" % index)
            copied = dict(resource)
            if copied["family"] not in _RESOURCE_FAMILIES:
                raise ValueError("provider resource family is unsupported")
            if any(not isinstance(copied[field], str) or not copied[field]
                   for field in ("id", "name", "status")):
                raise ValueError("provider resource fields must be non-empty strings")
            resources.append(copied)
        resources.sort(key=lambda item: (item["family"], item["id"]))
        resource_keys = {(item["family"], item["id"]) for item in resources}
        if len(resource_keys) != len(resources):
            raise ValueError("provider resource family/ID keys must be unique")
        if isinstance(known_pod_ids, (str, bytes, dict)):
            raise ValueError("known_pod_ids must be an iterable of pod IDs")
        known = set(known_pod_ids)
        if any(not isinstance(item, str) or not item for item in known):
            raise ValueError("known_pod_ids must contain non-empty strings")
        unknown = [
            {"family": item["family"], "id": item["id"]}
            for item in resources
            if item["family"] != "pods" or item["id"] not in known
        ]
        return resources, unknown

    @classmethod
    def _reclassify_inventory(cls, doc: Dict[str, Any]) -> None:
        inventory = doc["inventory"]
        if inventory is None:
            return
        resources, unknown = cls._canonical_provider_resources(
            inventory["provider_resources"], cls._known_pod_ids(doc))
        inventory["provider_resources"] = resources
        inventory["unknown_resources"] = unknown

    @staticmethod
    def _provider_snapshot_documents(
            *, provider: str, provider_account_id: str,
            balance_available_usd: Any, balance_observed_at: str,
            balance_valid_until: str, balance_source: str,
            inventory_observed_at: str, inventory_valid_until: str,
            inventory_complete: bool,
            provider_resources: Iterable[Dict[str, str]],
            known_pod_ids: Iterable[str],
            inventory_source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        available = _decimal(balance_available_usd, "balance_available_usd")
        observed = _timestamp(balance_observed_at, "balance_observed_at")
        valid = _timestamp(balance_valid_until, "balance_valid_until")
        if valid < observed:
            raise ValueError("balance_valid_until predates balance_observed_at")
        inventory_observed = _timestamp(
            inventory_observed_at, "inventory_observed_at")
        inventory_valid = _timestamp(
            inventory_valid_until, "inventory_valid_until")
        if inventory_valid < inventory_observed:
            raise ValueError("inventory_valid_until predates inventory_observed_at")
        if not isinstance(inventory_complete, bool):
            raise ValueError("inventory_complete must be boolean")
        resources, unknown = CampaignLedger._canonical_provider_resources(
            provider_resources, known_pod_ids)
        if (not isinstance(balance_source, str) or not balance_source
                or not isinstance(inventory_source, str) or not inventory_source):
            raise ValueError("provider snapshot sources must be non-empty strings")
        if not isinstance(provider, str) or not provider:
            raise ValueError("snapshot provider must be non-empty")
        if not isinstance(provider_account_id, str) or not provider_account_id:
            raise ValueError("snapshot provider_account_id must be non-empty")
        return ({
            "provider": provider,
            "provider_account_id": provider_account_id,
            "available_usd": _money(available),
            "observed_at": balance_observed_at,
            "valid_until": balance_valid_until,
            "source": balance_source,
        }, {
            "provider": provider,
            "provider_account_id": provider_account_id,
            "observed_at": inventory_observed_at,
            "valid_until": inventory_valid_until,
            "complete": inventory_complete,
            "provider_resources": resources,
            "unknown_resources": unknown,
            "source": inventory_source,
        })

    def record_provider_snapshot(
            self, expected_generation: int, *, provider: str,
            provider_account_id: str, balance_available_usd: Any,
            balance_observed_at: str, balance_valid_until: str, balance_source: str,
            inventory_observed_at: str, inventory_valid_until: str,
            inventory_complete: bool,
            provider_resources: Iterable[Dict[str, str]],
            inventory_source: str) -> TransitionResult:
        if (provider != self.provider
                or provider_account_id != self.provider_account_id):
            raise CampaignLedgerError(
                "provider snapshot identity differs from campaign")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc)
            balance_doc, inventory_doc = self._provider_snapshot_documents(
                provider=provider,
                provider_account_id=provider_account_id,
                balance_available_usd=balance_available_usd,
                balance_observed_at=balance_observed_at,
                balance_valid_until=balance_valid_until,
                balance_source=balance_source,
                inventory_observed_at=inventory_observed_at,
                inventory_valid_until=inventory_valid_until,
                inventory_complete=inventory_complete,
                provider_resources=provider_resources,
                known_pod_ids=self._known_pod_ids(doc),
                inventory_source=inventory_source)
            doc["balance"] = balance_doc
            doc["inventory"] = inventory_doc
            generation = self._commit(doc)
            return TransitionResult(True, "SNAPSHOT_RECORDED",
                                    "provider balance and inventory classified",
                                    generation, "")

    def classify_provider_resources(
            self, provider_resources: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        """Classify exact live resources from durable unreleased attempt IDs."""
        with self._locked(exclusive=False):
            doc = self._read_unlocked()
            resources, unknown = self._canonical_provider_resources(
                provider_resources, self._known_pod_ids(doc))
            return {
                "generation": doc["generation"],
                "known_pod_ids": sorted(self._known_pod_ids(doc)),
                "provider_resources": resources,
                "unknown_resources": unknown,
            }

    def authorize_concurrent_width_two(
            self, expected_generation: int, authorized_at: str,
            fruit_public_archive_sha256: str,
            fruit_proof_sha256: str) -> TransitionResult:
        _timestamp(authorized_at, "authorized_at")
        for name, value in (
                ("fruit_public_archive_sha256", fruit_public_archive_sha256),
                ("fruit_proof_sha256", fruit_proof_sha256)):
            if (not isinstance(value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError("%s must be a lowercase SHA-256 digest" % name)
        authorization = {
            "authorized_at": authorized_at,
            "from_width": 1,
            "to_width": 2,
            "fruit_public_archive_sha256": fruit_public_archive_sha256,
            "fruit_proof_sha256": fruit_proof_sha256,
        }
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc)
            existing = doc["width_authorization"]
            if existing is not None:
                if existing == authorization:
                    return TransitionResult(
                        True, "WIDTH_ALREADY_AUTHORIZED",
                        "campaign width is already authorized by this Fruit proof",
                        doc["generation"], "")
                return TransitionResult(
                    False, "WIDTH_AUTHORIZATION_MISMATCH",
                    "campaign width is bound to different Fruit evidence",
                    doc["generation"], "")
            if doc["max_concurrent_attempts"] != 2:
                return TransitionResult(
                    False, "WIDTH_TWO_NOT_CONFIGURED",
                    "campaign immutable maximum width is one",
                    doc["generation"], "")
            if doc["authorized_concurrent_attempts"] != 1:
                raise CampaignLedgerError(
                    "campaign authorized width changed without evidence")
            doc["authorized_concurrent_attempts"] = 2
            doc["width_authorization"] = authorization
            generation = self._commit(doc)
            return TransitionResult(
                True, "WIDTH_TWO_AUTHORIZED",
                "Fruit public archive and proof authorize campaign width two",
                generation, "")

    @staticmethod
    def _remaining_liability(doc: Dict[str, Any]) -> Decimal:
        return sum((Decimal(item["maximum_remaining_liability_usd"])
                    for item in doc["attempts"].values()), Decimal(0))

    @staticmethod
    def _limit(doc: Dict[str, Any]) -> Decimal:
        ceiling = Decimal(doc["hard_ceiling_usd"])
        if doc["balance"] is None:
            return Decimal(0)
        # Provider available balance is current, after settled charges.  Add
        # settled spend back only for this comparison with the campaign-wide
        # ceiling; `_committed_with` independently includes it as historical
        # spend.
        provider_capacity = (Decimal(doc["settled_charges_usd"])
                             + Decimal(doc["balance"]["available_usd"]))
        return min(ceiling, provider_capacity)

    @classmethod
    def _committed_with(cls, doc: Dict[str, Any], new_liability: Decimal) -> Decimal:
        return (Decimal(doc["settled_charges_usd"])
                + Decimal(doc["reserve_floor_usd"])
                + cls._remaining_liability(doc)
                + new_liability
                + Decimal(doc["cleanup_reaper_margin_usd"]))

    @classmethod
    def _admission_refusal(cls, doc: Dict[str, Any], code: str, message: str,
                           key: Optional[str], new_liability: Decimal = Decimal(0)
                           ) -> AdmissionResult:
        return AdmissionResult(False, code, message, doc["generation"], key, "REFUSE",
                               cls._committed_with(doc, new_liability), cls._limit(doc))

    @classmethod
    def _admission_decision(
            cls, doc: Dict[str, Any], expected_generation: int, key: str,
            quote: CostQuote, now: str, effective_width: int) -> AdmissionResult:
        if (isinstance(effective_width, bool)
                or not isinstance(effective_width, int)
                or effective_width not in (1, 2)):
            raise ValueError("effective_width must be 1 or 2")
        if doc["generation"] != expected_generation:
            return cls._admission_refusal(
                doc, "GENERATION_CONFLICT", "ledger changed; re-read before retrying", key)
        if key in doc["attempts"]:
            return cls._admission_refusal(
                doc, "ATTEMPT_EXISTS", "attempt identity is already reserved", key)
        if doc["balance"] is None:
            return cls._admission_refusal(
                doc, "BALANCE_UNKNOWN", "no provider balance snapshot", key)
        now_time = _timestamp(now, "now")
        if now_time < _timestamp(
                doc["balance"]["observed_at"], "balance.observed_at"):
            return cls._admission_refusal(
                doc, "BALANCE_FUTURE",
                "provider balance observation is future-dated", key)
        if now_time > _timestamp(
                doc["balance"]["valid_until"], "balance.valid_until"):
            return cls._admission_refusal(
                doc, "BALANCE_STALE", "provider balance snapshot is stale", key)
        if quote.future_at(now):
            return cls._admission_refusal(
                doc, "QUOTE_FUTURE", "cost quote is future-dated", key,
                quote.hard_cap_usd)
        if quote.stale_at(now):
            return cls._admission_refusal(
                doc, "QUOTE_STALE", "cost quote is stale", key, quote.hard_cap_usd)
        if not quote.storage_known:
            return cls._admission_refusal(
                doc, "UNKNOWN_STORAGE", "storage size or tariff is unknown", key)
        if quote.network_volume_size_gb != 0:
            return cls._admission_refusal(
                doc, "NETWORK_VOLUME_REFUSED",
                "safe campaign profile does not permit a network volume", key,
                quote.hard_cap_usd)
        inventory = doc["inventory"]
        if inventory is None:
            return cls._admission_refusal(
                doc, "INVENTORY_UNKNOWN", "no provider resource inventory", key)
        if now_time < _timestamp(
                inventory["observed_at"], "inventory.observed_at"):
            return cls._admission_refusal(
                doc, "INVENTORY_FUTURE",
                "provider resource inventory is future-dated", key)
        if now_time > _timestamp(
                inventory["valid_until"], "inventory.valid_until"):
            return cls._admission_refusal(
                doc, "INVENTORY_STALE", "provider resource inventory is stale", key)
        if not inventory["complete"]:
            return cls._admission_refusal(
                doc, "INVENTORY_UNKNOWN", "provider resource inventory is incomplete", key)
        if inventory["unknown_resources"]:
            return cls._admission_refusal(
                doc, "UNKNOWN_RESOURCES", "provider inventory contains unknown resources",
                key)
        for old_key, old in doc["attempts"].items():
            if old["released"]:
                continue
            if old["admission_freeze_reason"] is not None:
                return cls._admission_refusal(
                    doc, "ATTEMPT_FROZEN",
                    "unresolved attempt %s freezes admission: %s" % (
                        old_key, old["admission_freeze_reason"]), key)
            if old["phase"] == "CREATING" and not old["provider_ids"]:
                return cls._admission_refusal(
                    doc, "AMBIGUOUS_CREATE",
                    "CREATING attempt %s has no exact provider ID" % old_key, key)
            if old["deletion"] is not None and old["billing"] is None:
                return cls._admission_refusal(
                    doc, "BILLING_UNRECONCILED",
                    "deleted attempt %s lacks billing reconciliation" % old_key, key)
        outstanding = sum(
            1 for old in doc["attempts"].values() if not old["released"])
        admitted_width = min(
            doc["authorized_concurrent_attempts"], effective_width)
        if outstanding + 1 > admitted_width:
            return cls._admission_refusal(
                doc, "WIDTH_EXCEEDED",
                "campaign concurrency width is already fully reserved",
                key, quote.hard_cap_usd)
        committed = cls._committed_with(doc, quote.hard_cap_usd)
        limit = cls._limit(doc)
        if committed > limit:
            return cls._admission_refusal(
                doc, "CEILING_EXCEEDED",
                "maximum campaign liability exceeds available ceiling", key,
                quote.hard_cap_usd)
        return AdmissionResult(
            True, "ADMISSIBLE", "admission checks pass without mutation",
            doc["generation"], key, "NONE", committed, limit)

    @staticmethod
    def _reservation_inputs(
            job_hash: str, attempt: str, quote: CostQuote, now: str) -> str:
        key = attempt_key(job_hash, attempt)
        if not isinstance(quote, CostQuote):
            raise TypeError("quote must be CostQuote")
        _timestamp(now, "now")
        return key

    @classmethod
    def preview_new_campaign(
            cls, *, hard_ceiling_usd: Any, reserve_floor_usd: Any,
            cleanup_reaper_margin_usd: Any, max_concurrent_attempts: int,
            provider: str, provider_account_id: str,
            balance_available_usd: Any, balance_observed_at: str,
            balance_valid_until: str, balance_source: str,
            inventory_observed_at: str, inventory_valid_until: str,
            inventory_complete: bool,
            provider_resources: Iterable[Dict[str, str]],
            inventory_source: str,
            job_hash: str, attempt: str,
            quote: CostQuote, now: str, currency: str = CURRENCY,
            effective_width: int = 1) -> AdmissionResult:
        """Preview a brand-new campaign entirely in memory.

        This is the absent-ledger dry-plan path: it builds and validates the
        same v1 document and fresh provider snapshot used by paid admission,
        then invokes the shared decision without creating a lock or ledger
        file.
        """
        doc = cls._new_campaign_document(
            hard_ceiling_usd, reserve_floor_usd, cleanup_reaper_margin_usd,
            max_concurrent_attempts, provider, provider_account_id, currency)
        balance_doc, inventory_doc = cls._provider_snapshot_documents(
            provider=provider,
            provider_account_id=provider_account_id,
            balance_available_usd=balance_available_usd,
            balance_observed_at=balance_observed_at,
            balance_valid_until=balance_valid_until,
            balance_source=balance_source,
            inventory_observed_at=inventory_observed_at,
            inventory_valid_until=inventory_valid_until,
            inventory_complete=inventory_complete,
            provider_resources=provider_resources,
            known_pod_ids=(),
            inventory_source=inventory_source)
        doc["balance"] = balance_doc
        doc["inventory"] = inventory_doc
        cls._validate_document(doc)
        key = cls._reservation_inputs(job_hash, attempt, quote, now)
        return cls._admission_decision(
            doc, 0, key, quote, now, effective_width)

    def preview_reserve(
            self, expected_generation: int, job_hash: str, attempt: str,
            quote: CostQuote, now: str, *,
            effective_width: int = 1) -> AdmissionResult:
        """Run the exact locked admission decision without changing the ledger."""
        key = self._reservation_inputs(job_hash, attempt, quote, now)
        with self._locked(exclusive=False):
            doc = self._read_unlocked()
            return self._admission_decision(
                doc, expected_generation, key, quote, now, effective_width)

    def preview_reserve_with_provider_snapshot(
            self, expected_generation: int, job_hash: str, attempt: str,
            quote: CostQuote, now: str, *, effective_width: int = 1,
            provider: str, provider_account_id: str,
            balance_available_usd: Any, balance_observed_at: str,
            balance_valid_until: str, balance_source: str,
            inventory_observed_at: str, inventory_valid_until: str,
            inventory_complete: bool,
            provider_resources: Iterable[Dict[str, str]],
            inventory_source: str) -> AdmissionResult:
        """Preview an existing campaign against fresh in-memory provider facts.

        Attempts, settled spend, limits, generation, and width come from the
        durable ledger.  Only the balance/inventory snapshot is overlaid on the
        read document, so stale stored facts neither reject nor get refreshed
        by a dry plan.
        """
        key = self._reservation_inputs(job_hash, attempt, quote, now)
        if (provider != self.provider
                or provider_account_id != self.provider_account_id):
            raise CampaignLedgerError(
                "preview provider snapshot identity differs from campaign")
        with self._locked(exclusive=False):
            doc = self._read_unlocked()
            balance_doc, inventory_doc = self._provider_snapshot_documents(
                provider=provider,
                provider_account_id=provider_account_id,
                balance_available_usd=balance_available_usd,
                balance_observed_at=balance_observed_at,
                balance_valid_until=balance_valid_until,
                balance_source=balance_source,
                inventory_observed_at=inventory_observed_at,
                inventory_valid_until=inventory_valid_until,
                inventory_complete=inventory_complete,
                provider_resources=provider_resources,
                known_pod_ids=self._known_pod_ids(doc),
                inventory_source=inventory_source)
            doc["balance"] = balance_doc
            doc["inventory"] = inventory_doc
            self._validate_document(doc)
            return self._admission_decision(
                doc, expected_generation, key, quote, now, effective_width)

    @staticmethod
    def _validate_bootstrap_drill_quote(quote: CostQuote) -> None:
        if (quote.profile != "runpod-drill-secure-l4-on-demand"
                or quote.timing_kind != "exact-target-profile"
                or quote.tariff_source != RUNPOD_TARIFF_SOURCE
                or not re.fullmatch(r"[0-9a-f]{64}", quote.timing_evidence)
                or quote.live_compute_usd_per_hour
                   != quote.reserved_compute_usd_per_hour):
            raise ValueError("bootstrap controller-loss drill quote is not exact")
    def _reserve_kind(
            self, expected_generation: int, job_hash: str, attempt: str,
            quote: CostQuote, now: str, reservation_kind: str,
            effective_width: int) -> AdmissionResult:
        key = self._reservation_inputs(job_hash, attempt, quote, now)
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if reservation_kind == "bootstrap-controller-loss-drill":
                self._validate_bootstrap_drill_quote(quote)
                prior = list(doc["attempts"].values())
                if (doc["width_authorization"] is not None
                        or _bootstrap_drill_blocked_by_prior_attempts(prior)):
                    return self._admission_refusal(
                        doc, "BOOTSTRAP_DRILL_NOT_FIRST",
                        "bootstrap drill must precede measurements and prior paid drills",
                        key, quote.hard_cap_usd)
            decision = self._admission_decision(
                doc, expected_generation, key, quote, now, effective_width)
            if not decision.admitted:
                return decision
            doc["attempts"][key] = {
                "job_hash": job_hash,
                "attempt": attempt,
                "reservation_kind": reservation_kind,
                "phase": "RESERVED",
                "provider_ids": [],
                "cleanup_binding_evidence": None,
                "precreate_cancellation": None,
                "reserved_at": now,
                "reserved_quote": quote.to_dict(),
                "actual_quote": None,
                "maximum_remaining_liability_usd": _money(quote.hard_cap_usd),
                "deletion": None,
                "billing": None,
                "released": False,
                "admission_freeze_reason": None,
            }
            generation = self._commit(doc)
            return AdmissionResult(
                True, "ADMITTED", "attempt liability reserved",
                generation, key, "CREATE", decision.maximum_committed_usd,
                decision.admission_limit_usd)

    def reserve(
            self, expected_generation: int, job_hash: str, attempt: str,
            quote: CostQuote, now: str, *,
            effective_width: int = 1) -> AdmissionResult:
        return self._reserve_kind(
            expected_generation, job_hash, attempt, quote, now, "measurement",
            effective_width)

    def reserve_bootstrap_drill(
            self, expected_generation: int, job_hash: str, attempt: str,
            quote: CostQuote, now: str) -> AdmissionResult:
        """Reserve the one pre-measurement paid controller-loss drill."""
        return self._reserve_kind(
            expected_generation, job_hash, attempt, quote, now,
            "bootstrap-controller-loss-drill", 1)

    def cancel_before_create(
            self, expected_generation: int, attempt_key: str,
            cancelled_at: str, lease_state: str,
            no_create_evidence: str) -> TransitionResult:
        """Release only with durable no-resource proof.

        `LEASE_ABSENT`/`PREPARED` prove the POST was never authorized.
        `PROVIDER_REJECTED_CREATE` is different and equally definitive: the
        POST went out and the provider refused it by name, returning an
        enumerated no-resource code and no id, with a complete listing showing
        nothing attributable. Without this the reservation stays held and paid
        admission closes for the whole campaign, as it did on
        2026-09-03T02:35Z for a `SUPPLY_CONSTRAINT` refusal.
        """
        _timestamp(cancelled_at, "cancelled_at")
        if lease_state not in (
                "LEASE_ABSENT", "PREPARED", "PROVIDER_REJECTED_CREATE"):
            raise ValueError(
                "lease_state must be LEASE_ABSENT, PREPARED or "
                "PROVIDER_REJECTED_CREATE")
        if not isinstance(no_create_evidence, str) or not no_create_evidence:
            raise ValueError("durable no-create evidence is required")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(
                    False, "ATTEMPT_UNKNOWN", "attempt not found",
                    doc["generation"], attempt_key)
            if item["phase"] == "CANCELLED_BEFORE_CREATE":
                recorded = item["precreate_cancellation"]
                if (recorded["cancelled_at"] == cancelled_at
                        and recorded["lease_state"] == lease_state
                        and recorded["no_create_evidence"] == no_create_evidence):
                    return TransitionResult(
                        True, "CANCELLATION_ALREADY_RECORDED",
                        "matching pre-create cancellation evidence already recorded",
                        doc["generation"], attempt_key)
                return TransitionResult(
                    False, "CANCELLATION_EVIDENCE_MISMATCH",
                    "pre-create cancellation evidence conflicts with campaign",
                    doc["generation"], attempt_key, "FREEZE")
            phase = item["phase"]
            phase_permits_cancel = (
                phase == "RESERVED"
                or (phase == "CREATING"
                    and lease_state in ("PREPARED",
                                        "PROVIDER_REJECTED_CREATE")))
            if (not phase_permits_cancel or item["provider_ids"]
                    or item["actual_quote"] is not None
                    or item["cleanup_binding_evidence"] is not None
                    or item["deletion"] is not None or item["billing"] is not None):
                return TransitionResult(
                    False, "POST_INTENT_OR_CREATE_ALREADY_BEGAN",
                    "attempt lacks durable proof that POST was never authorized",
                    doc["generation"], attempt_key, "FREEZE")
            item["precreate_cancellation"] = {
                "cancelled_at": cancelled_at,
                "campaign_phase_before_cancel": phase,
                "lease_state": lease_state,
                "no_create_evidence": no_create_evidence,
            }
            item["phase"] = "CANCELLED_BEFORE_CREATE"
            item["maximum_remaining_liability_usd"] = "0"
            item["released"] = True
            generation = self._commit(doc)
            return TransitionResult(
                True, "CANCELLED_BEFORE_CREATE",
                "reservation released with durable no-POST evidence",
                generation, attempt_key)


    def mark_creating(self, expected_generation: int, attempt_key: str) -> TransitionResult:
        return self._set_phase(expected_generation, attempt_key, "CREATING",
                               allowed=("RESERVED",), code="CREATING_RECORDED")

    def bind_provider_for_cleanup(
            self, expected_generation: int, attempt_key: str,
            provider_ids: Iterable[str], evidence: str) -> TransitionResult:
        """Bind durable lease IDs when scientific post-create binding cannot run.

        This transition never authorizes workload execution.  It makes every
        exact or ambiguous provider identity projectable into deletion and
        billing proof while retaining the full reserved liability and width.
        """
        if isinstance(provider_ids, (str, bytes)):
            raise ValueError("provider_ids must be an iterable of exact IDs")
        ids = list(provider_ids)
        if (not ids
                or any(not isinstance(value, str) or not value for value in ids)):
            raise ValueError("at least one non-empty provider ID is required")
        ids = sorted(set(ids))
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("durable provider-binding evidence is required")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(
                    False, "ATTEMPT_UNKNOWN", "attempt not found",
                    doc["generation"], attempt_key)
            if (item["phase"] == "TERMINATE_REQUIRED"
                    and item["actual_quote"] is None
                    and item["provider_ids"] == ids
                    and item["cleanup_binding_evidence"] == evidence):
                return TransitionResult(
                    True, "PROVIDER_CLEANUP_BINDING_UNCHANGED",
                    "cleanup provider IDs are already bound",
                    doc["generation"], attempt_key, "TERMINATE_IMMEDIATELY")
            if (item["phase"] != "CREATING" or item["provider_ids"]
                    or item["actual_quote"] is not None):
                return TransitionResult(
                    False, "CLEANUP_BINDING_MISMATCH",
                    "cleanup IDs bind exactly once from unbound CREATING",
                    doc["generation"], attempt_key, "FREEZE")
            item["provider_ids"] = ids
            item["cleanup_binding_evidence"] = evidence
            item["phase"] = "TERMINATE_REQUIRED"
            item["admission_freeze_reason"] = (
                "provider identity bound for cleanup only (%d exact ID%s)"
                % (len(ids), "" if len(ids) == 1 else "s"))
            observed_pod_ids = set()
            if doc["inventory"] is not None:
                observed_pod_ids = {
                    resource["id"]
                    for resource in doc["inventory"]["provider_resources"]
                    if resource["family"] == "pods"
                }
            if not set(ids).issubset(observed_pod_ids):
                # A provider response can durably identify a cleanup target
                # before a fresh complete inventory observes it.  The older
                # snapshot is no longer authoritative after that POST; clear
                # it so admission freezes until a fresh full-family read.
                doc["inventory"] = None
            self._reclassify_inventory(doc)
            generation = self._commit(doc)
            return TransitionResult(
                True, "PROVIDER_BOUND_FOR_CLEANUP",
                "provider identity is cleanup-only; workload remains refused",
                generation, attempt_key, "TERMINATE_IMMEDIATELY")

    def mark_phase(self, expected_generation: int, attempt_key: str,
                   phase: str) -> TransitionResult:
        predecessors = {
            "LIVE": ("LIVE",),
            "RUNNING": ("LIVE", "RUNNING"),
            "EXITED": ("LIVE", "RUNNING", "EXITED"),
            "TERMINATE_REQUESTED": (
                "LIVE", "RUNNING", "EXITED", "TERMINATE_REQUIRED",
                "TERMINATE_REQUESTED"),
        }
        if phase not in predecessors:
            raise ValueError("phase is not a public lifecycle transition")
        return self._set_phase(expected_generation, attempt_key, phase,
                               allowed=predecessors[phase], code="PHASE_RECORDED")

    def _set_phase(self, expected_generation: int, attempt_key: str, phase: str,
                   allowed: Tuple[str, ...], code: str) -> TransitionResult:
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(False, "ATTEMPT_UNKNOWN", "attempt not found",
                                        doc["generation"], attempt_key)
            if item["released"]:
                return TransitionResult(False, "ATTEMPT_RELEASED",
                                        "reconciled attempt cannot transition",
                                        doc["generation"], attempt_key)
            if item["phase"] == phase:
                return TransitionResult(
                    True, "PHASE_UNCHANGED",
                    "attempt already has the requested phase",
                    doc["generation"], attempt_key)
            if item["phase"] not in allowed:
                return TransitionResult(False, "INVALID_TRANSITION",
                                        "%s cannot transition to %s" % (item["phase"], phase),
                                        doc["generation"], attempt_key)
            item["phase"] = phase
            generation = self._commit(doc)
            return TransitionResult(True, code, "attempt phase recorded",
                                    generation, attempt_key)

    def bind_actual_quote(self, expected_generation: int, attempt_key: str,
                          provider_id: str, actual_quote: CostQuote) -> TransitionResult:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("provider_id must be non-empty")
        if not isinstance(actual_quote, CostQuote):
            raise TypeError("actual_quote must be CostQuote")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(False, "ATTEMPT_UNKNOWN", "attempt not found",
                                        doc["generation"], attempt_key)
            if (item["phase"] != "CREATING" or item["provider_ids"]
                    or item["cleanup_binding_evidence"] is not None):
                return TransitionResult(False, "INVALID_TRANSITION",
                                        "actual quote binds exactly once after CREATING",
                                        doc["generation"], attempt_key)
            reserved = CostQuote.from_dict(item["reserved_quote"])
            bound_fields = (
                "reserved_compute_usd_per_hour",
                "container_disk_size_gb",
                "container_disk_running_usd_per_gb_month",
                "container_disk_stopped_usd_per_gb_month",
                "pod_disk_size_gb",
                "pod_disk_running_usd_per_gb_month",
                "pod_disk_stopped_usd_per_gb_month",
                "network_volume_size_gb",
                "network_volume_usd_per_gb_month",
                "storage_month_hours",
                "network_billing_increment_seconds",
                "tariff_source",
                "tariff_effective_at",
                "target",
                "profile",
                "timing_kind",
                "timing_evidence",
                "workload_deadline_seconds",
                "provider_termination_deadline_seconds",
                "retrieval_delete_reserve_seconds",
                "timer_api_lag_seconds",
                "hard_cap_usd",
            )
            identity_changed = any(
                getattr(actual_quote, field) != getattr(reserved, field)
                for field in bound_fields)
            storage_unknown = not actual_quote.storage_known
            network_present = (actual_quote.network_volume_size_gb is None
                               or actual_quote.network_volume_size_gb != 0)
            rate_increased = (
                actual_quote.live_compute_usd_per_hour
                > reserved.reserved_compute_usd_per_hour)
            must_terminate = (identity_changed or storage_unknown
                              or network_present or rate_increased)
            item["provider_ids"] = [provider_id]
            item["actual_quote"] = actual_quote.to_dict()
            item["maximum_remaining_liability_usd"] = _money(max(
                Decimal(item["maximum_remaining_liability_usd"]),
                actual_quote.hard_cap_usd))
            self._reclassify_inventory(doc)
            if must_terminate:
                reasons = []
                if identity_changed:
                    reasons.append(
                        "requested resource/deadline/tariff identity changed")
                if storage_unknown:
                    reasons.append("storage became unknown")
                if network_present:
                    reasons.append("network volume present")
                if rate_increased:
                    reasons.append("live compute rate exceeds reservation")
                item["phase"] = "TERMINATE_REQUIRED"
                item["admission_freeze_reason"] = "; ".join(reasons)
                generation = self._commit(doc)
                return TransitionResult(True, "TERMINATE_IMMEDIATELY",
                                        item["admission_freeze_reason"], generation,
                                        attempt_key, "TERMINATE_IMMEDIATELY")
            item["phase"] = "LIVE"
            generation = self._commit(doc)
            return TransitionResult(True, "ACTUAL_QUOTE_BOUND",
                                    "actual provider quote is within reservation",
                                    generation, attempt_key, "CONTINUE")

    def project_terminal_lease(
            self, expected_generation: int, attempt_key: str,
            provider_ids: Iterable[str], binding_evidence: str,
            absence_proofs: Dict[str, Dict[str, str]],
            billing_reconciled_at: str, final_charge_usd: Any,
            billing_proof: str, *,
            provider_snapshot: Dict[str, Any]) -> TransitionResult:
        """Atomically project a reaped lease into campaign release evidence.

        ``absence_proofs`` must have exactly one
        ``{"deleted_at", "proof"}`` object for every durable provider ID.
        ``billing_proof`` covers the aggregate charge for that complete ID set.
        """
        if isinstance(provider_ids, (str, bytes)):
            raise ValueError("provider_ids must be an iterable of exact IDs")
        ids = list(provider_ids)
        if (not ids
                or any(not isinstance(value, str) or not value for value in ids)):
            raise ValueError("at least one non-empty provider ID is required")
        ids = sorted(set(ids))
        if not isinstance(binding_evidence, str) or not binding_evidence:
            raise ValueError("durable provider-binding evidence is required")
        if not isinstance(absence_proofs, dict) or set(absence_proofs) != set(ids):
            raise ValueError("absence_proofs must cover the exact provider ID set")
        normalized_absence = {}
        for provider_id in ids:
            proof_doc = absence_proofs[provider_id]
            _require_exact_keys(
                proof_doc, ("deleted_at", "proof"),
                "absence proof %s" % provider_id)
            _timestamp(proof_doc["deleted_at"], "deleted_at")
            if not isinstance(proof_doc["proof"], str) or not proof_doc["proof"]:
                raise ValueError("absence proof is empty")
            normalized_absence[provider_id] = dict(proof_doc)
        _timestamp(billing_reconciled_at, "billing_reconciled_at")
        charge = _decimal(final_charge_usd, "final_charge_usd")
        if not isinstance(billing_proof, str) or not billing_proof:
            raise ValueError("aggregate billing proof is required")
        billing_doc = {
            "provider_ids": ids,
            "reconciled_at": billing_reconciled_at,
            "final_charge_usd": _money(charge),
            "proof": billing_proof,
        }

        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(
                    False, "ATTEMPT_UNKNOWN", "attempt not found",
                    doc["generation"], attempt_key)
            if not isinstance(provider_snapshot, dict):
                raise ValueError("terminal provider_snapshot must be an object")
            _require_exact_keys(
                provider_snapshot,
                ("provider", "provider_account_id",
                 "balance_available_usd", "balance_observed_at",
                 "balance_valid_until", "balance_source",
                 "inventory_observed_at", "inventory_valid_until",
                 "inventory_complete", "provider_resources",
                 "inventory_source"),
                "terminal provider_snapshot")
            if (provider_snapshot["provider"] != self.provider
                    or provider_snapshot["provider_account_id"]
                        != self.provider_account_id):
                raise CampaignLedgerError(
                    "terminal provider snapshot identity differs from campaign")
            balance_doc, inventory_doc = self._provider_snapshot_documents(
                provider=provider_snapshot["provider"],
                provider_account_id=provider_snapshot["provider_account_id"],
                balance_available_usd=provider_snapshot[
                    "balance_available_usd"],
                balance_observed_at=provider_snapshot["balance_observed_at"],
                balance_valid_until=provider_snapshot["balance_valid_until"],
                balance_source=provider_snapshot["balance_source"],
                inventory_observed_at=provider_snapshot[
                    "inventory_observed_at"],
                inventory_valid_until=provider_snapshot[
                    "inventory_valid_until"],
                inventory_complete=provider_snapshot["inventory_complete"],
                provider_resources=provider_snapshot["provider_resources"],
                known_pod_ids=self._known_pod_ids(doc),
                inventory_source=provider_snapshot["inventory_source"])
            if (inventory_doc["complete"] is not True
                    or inventory_doc["unknown_resources"]):
                return TransitionResult(
                    False, "TERMINAL_SNAPSHOT_UNSAFE",
                    "terminal provider snapshot is incomplete or unknown",
                    doc["generation"], attempt_key, "FREEZE")
            present_target_ids = sorted({
                row["id"] for row in inventory_doc["provider_resources"]
                if row["family"] == "pods" and row["id"] in set(ids)
            })
            if present_target_ids:
                return TransitionResult(
                    False, "TERMINAL_TARGET_REAPPEARED",
                    "terminal provider snapshot still contains exact target IDs",
                    doc["generation"], attempt_key, "FREEZE")
            reserved_quote = CostQuote.from_dict(item["reserved_quote"])
            actual_quote = (
                None if item["actual_quote"] is None
                else CostQuote.from_dict(item["actual_quote"]))
            if (charge < 0 or charge > reserved_quote.hard_cap_usd
                    or (actual_quote is not None
                        and charge > actual_quote.hard_cap_usd)):
                return TransitionResult(
                    False, "FINAL_CHARGE_EXCEEDS_HARD_CAP",
                    "final charge is negative or exceeds an attempt hard cap",
                    doc["generation"], attempt_key, "FREEZE")
            changed = (
                doc.get("balance") != balance_doc
                or doc.get("inventory") != inventory_doc)
            doc["balance"] = balance_doc
            doc["inventory"] = inventory_doc
            if not item["provider_ids"]:
                if (item["phase"] != "CREATING"
                        or item["precreate_cancellation"] is not None):
                    return TransitionResult(
                        False, "TERMINAL_PROJECTION_FORBIDDEN",
                        "unbound terminal projection requires CREATING",
                        doc["generation"], attempt_key, "FREEZE")
                item["provider_ids"] = ids
                item["cleanup_binding_evidence"] = binding_evidence
                item["phase"] = "TERMINATE_REQUIRED"
                item["admission_freeze_reason"] = (
                    "provider identity bound from terminal lease projection")
                changed = True
            elif item["provider_ids"] != ids:
                return TransitionResult(
                    False, "PROVIDER_ID_SET_MISMATCH",
                    "terminal lease IDs differ from campaign-bound IDs",
                    doc["generation"], attempt_key, "FREEZE")
            elif (item["cleanup_binding_evidence"] is not None
                  and item["cleanup_binding_evidence"] != binding_evidence):
                return TransitionResult(
                    False, "TERMINAL_PROOF_MISMATCH",
                    "terminal lease binding evidence conflicts with campaign",
                    doc["generation"], attempt_key, "FREEZE")

            existing_absence = (
                (item["deletion"] or {"proofs": {}})["proofs"])
            for provider_id, proof_doc in existing_absence.items():
                if normalized_absence.get(provider_id) != proof_doc:
                    return TransitionResult(
                        False, "TERMINAL_PROOF_MISMATCH",
                        "absence evidence conflicts for provider ID %s" % provider_id,
                        doc["generation"], attempt_key, "FREEZE")
            if existing_absence != normalized_absence:
                item["deletion"] = {"proofs": normalized_absence}
                changed = True
            if item["billing"] is not None and item["billing"] != billing_doc:
                return TransitionResult(
                    False, "TERMINAL_PROOF_MISMATCH",
                    "aggregate billing evidence conflicts with campaign",
                    doc["generation"], attempt_key, "FREEZE")
            if item["billing"] is None:
                item["billing"] = billing_doc
                doc["settled_charges_usd"] = _money(
                    Decimal(doc["settled_charges_usd"]) + charge)
                changed = True
            item["phase"] = "DELETED"
            self._release_if_complete(item)
            self._reclassify_inventory(doc)
            if not item["released"]:
                return TransitionResult(
                    False, "TERMINAL_PROJECTION_INCOMPLETE",
                    "terminal lease did not provide complete release evidence",
                    doc["generation"], attempt_key, "FREEZE")
            if not changed:
                return TransitionResult(
                    True, "TERMINAL_LEASE_ALREADY_PROJECTED",
                    "terminal lease evidence already released campaign liability",
                    doc["generation"], attempt_key)
            generation = self._commit(doc)
            return TransitionResult(
                True, "TERMINAL_LEASE_PROJECTED",
                "terminal lease released campaign liability",
                generation, attempt_key)

    def mark_deleted(self, expected_generation: int, attempt_key: str,
                     provider_id: str, deleted_at: str, proof: str) -> TransitionResult:
        _timestamp(deleted_at, "deleted_at")
        if not provider_id or not proof:
            raise ValueError("exact provider ID and deletion proof are required")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(False, "ATTEMPT_UNKNOWN", "attempt not found",
                                        doc["generation"], attempt_key)
            if provider_id not in item["provider_ids"]:
                return TransitionResult(False, "PROVIDER_ID_MISMATCH",
                                        "deletion proof is for an unbound provider ID",
                                        doc["generation"], attempt_key)
            if item["deletion"] is None:
                item["deletion"] = {"proofs": {}}
            proofs = item["deletion"]["proofs"]
            if provider_id in proofs:
                return TransitionResult(False, "DELETION_ALREADY_RECORDED",
                                        "deletion proof already exists for provider ID",
                                        doc["generation"], attempt_key)
            proofs[provider_id] = {
                "deleted_at": deleted_at,
                "proof": proof,
            }
            if set(proofs) == set(item["provider_ids"]):
                item["phase"] = "DELETED"
            self._release_if_complete(item)
            self._reclassify_inventory(doc)
            generation = self._commit(doc)
            return TransitionResult(
                True, "DELETION_RECORDED",
                "liability retained pending all deletion/billing proofs"
                if not item["released"] else "liability released",
                generation, attempt_key)

    def reconcile_billing(self, expected_generation: int, attempt_key: str,
                          provider_id: str, reconciled_at: str,
                          final_charge_usd: Any, proof: str) -> TransitionResult:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("exact provider ID is required")
        return self.reconcile_billing_set(
            expected_generation, attempt_key, [provider_id], reconciled_at,
            final_charge_usd, proof)

    def reconcile_billing_set(
            self, expected_generation: int, attempt_key: str,
            provider_ids: Iterable[str], reconciled_at: str,
            final_charge_usd: Any, proof: str) -> TransitionResult:
        """Record one aggregate billing proof for the complete bound ID set."""
        if isinstance(provider_ids, (str, bytes)):
            raise ValueError("provider_ids must be an iterable of exact IDs")
        ids = list(provider_ids)
        if (not ids
                or any(not isinstance(value, str) or not value for value in ids)):
            raise ValueError("at least one non-empty provider ID is required")
        ids = sorted(set(ids))
        _timestamp(reconciled_at, "reconciled_at")
        charge = _decimal(final_charge_usd, "final_charge_usd")
        if not isinstance(proof, str) or not proof:
            raise ValueError("billing proof is required")
        with self._locked(exclusive=True):
            doc = self._read_unlocked()
            if doc["generation"] != expected_generation:
                return self._conflict(doc, attempt_key)
            item = doc["attempts"].get(attempt_key)
            if item is None:
                return TransitionResult(False, "ATTEMPT_UNKNOWN", "attempt not found",
                                        doc["generation"], attempt_key)
            reserved_quote = CostQuote.from_dict(item["reserved_quote"])
            actual_quote = (
                None if item["actual_quote"] is None
                else CostQuote.from_dict(item["actual_quote"]))
            if (charge < 0 or charge > reserved_quote.hard_cap_usd
                    or (actual_quote is not None
                        and charge > actual_quote.hard_cap_usd)):
                return TransitionResult(
                    False, "FINAL_CHARGE_EXCEEDS_HARD_CAP",
                    "final charge is negative or exceeds an attempt hard cap",
                    doc["generation"], attempt_key, "FREEZE")
            if ids != item["provider_ids"]:
                return TransitionResult(
                    False, "BILLING_ID_SET_MISMATCH",
                    "billing proof must name the complete bound provider ID set",
                    doc["generation"], attempt_key)
            if item["billing"] is not None:
                return TransitionResult(False, "BILLING_ALREADY_RECONCILED",
                                        "aggregate billing proof already exists",
                                        doc["generation"], attempt_key)
            item["billing"] = {
                "provider_ids": ids,
                "reconciled_at": reconciled_at,
                "final_charge_usd": _money(charge),
                "proof": proof,
            }
            doc["settled_charges_usd"] = _money(
                Decimal(doc["settled_charges_usd"]) + charge)
            self._release_if_complete(item)
            self._reclassify_inventory(doc)
            generation = self._commit(doc)
            return TransitionResult(
                True, "BILLING_RECONCILED",
                "liability retained pending all deletion proofs"
                if not item["released"] else "liability released",
                generation, attempt_key)

    @staticmethod
    def _release_if_complete(item: Dict[str, Any]) -> None:
        provider_ids = set(item["provider_ids"])
        deletion_ids = set(
            (item["deletion"] or {"proofs": {}})["proofs"])
        billing_ids = set((item["billing"] or {}).get("provider_ids", []))
        if provider_ids and deletion_ids == provider_ids and billing_ids == provider_ids:
            item["maximum_remaining_liability_usd"] = "0"
            item["released"] = True
            item["phase"] = "RECONCILED"
            item["admission_freeze_reason"] = None


__all__ = [
    "AdmissionResult", "CampaignLedger", "CampaignLedgerError", "CostQuote",
    "MAX_QUOTE_VALIDITY_SECONDS", "RUNPOD_TARIFF_SOURCE", "SCHEMA",
    "TransitionResult",
]
