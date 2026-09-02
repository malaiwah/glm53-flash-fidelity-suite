#!/usr/bin/env python3
"""Fail-closed validation for paid RunPod evidence and uploaded panels."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, tarfile, tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional

PROOF_SCHEMA = "fidelity-suite/runpod-safety-proof.v2"
DRILL_KIND = "paid-controller-loss-autonomous-reaper"
LEASE_DRILL_KIND = "paid-controller-loss-provider-deadline"
DEADLINE_POLL_DURATION_MAX_SECONDS = 120
DEADLINE_INTERPOLL_GAP_MAX_SECONDS = 120
DRILL_GPU_TYPE = "NVIDIA L4"
DRILL_IMAGE = (
    "runpod/pytorch@sha256:"
    "ab2addc2916ffc72989288bd5048933c69ba6531f1d679c25afbd9eadc5a5fd5")
FRUIT_REPO = "malaiwah/GLM-5.2-SIQ-Fruit-bf16"
FRUIT_REVISION = "ef68013aa6e16453cf52b5b77647f72fbe258c3c"
FRUIT_PANEL_ID = "panel--fruit.malaiwah.heldout-v1"
FRUIT_PANEL_CONTEXTS = 16
FRUIT_PANEL_SCORED_POSITIONS = 32752
FRUIT_PANEL_SUITE_TOKEN_SHA256 = (
    "a6d367cc3ba448800372dee435d2bb4f536d23ca68843628832fa3b122ceabe1")
FRUIT_PANEL_FILE_SHA256 = (
    "8a53fedd3a7ca67ec8d8026a0e76ca0c85547d2020cd2ae983b2b7a7b8264ea7")
FRUIT_RECEIPT_DECLARED_SHA256 = (
    "3372cb0b55681457a4d1dc860dcd1a9772d09190c4ce97ea6063ebc57e03cf6e")
FRUIT_RECEIPT_FILE_SHA256 = (
    "6c195ef252305d0e647c2f33b99e60d927c306a4e86ff3a055173297bc9a403c")
FRUIT_TOKENIZER_IDENTITY_SHA256 = (
    "cf53e420fedbaf37bf9749fb1fc874322f0df23a3cd69a3dedc9834ced850ef0")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWLISTS = {
    (FRUIT_REPO, FRUIT_REVISION): {
        "path": "engines/tools/layer-outer-evidence/fruit-layer13-unexpected-keys.json",
        "artifact_sha256": "5731fac8f82305c6d3b41fa9fbbd87db3f81a33e27533a0acdf6b28bed1566a2",
        "canonical_sorted_names_sha256": "41b825b0045a2e1e90eea8f88bb06022459d26a3957c40c52d65d677d8a17968",
        "count": 791,
    },
    ("zai-org/GLM-5.3-Flash-BF16",
     "a6c167b62691b2bac901344b65cb651a70f53e43"): {
        "path": "engines/tools/dione-evidence/m2-layer45-unexpected-keys.json",
        "artifact_sha256": "35b7f1bd8d693d92bed089c0784a30c0b2d7b859a65a80c37a3fc03ab565f61b",
        "canonical_sorted_names_sha256": "acc1e9f10c0f903c735a7fcf5fd267fc879bce65623f0b850f80016da5e903b7",
        "count": 889,
    },
    ("zai-org/GLM-5.3-BF16",
     "304b8051cfb2b260b61ce0cbe330e02a98e73639"): {
        "path": "engines/tools/layer-outer-evidence/glm53-layer78-unexpected-keys.json",
        "artifact_sha256": "714d95eef084e00cb8d579ba789fea80d4405160e437f5ef91b1b9c67c98e7df",
        "canonical_sorted_names_sha256": "61e5f26aed8bca408c5de5347d8e1668b0c5716237dad1fc98c47bc108f4ae57",
        "count": 791,
    },
}

class SafetyProofError(ValueError):
    pass

def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SafetyProofError("evidence is not finite canonical JSON: %s" % exc)

def campaign_ledger_coordinate_sha256(path):
    try:
        canonical = str(Path(path).resolve(strict=True))
    except (OSError, TypeError, ValueError) as exc:
        raise SafetyProofError(
            "current durable campaign ledger path is invalid: %s"
            % exc.__class__.__name__)
    if not os.path.isabs(canonical) or "\x00" in canonical:
        raise SafetyProofError(
            "current durable campaign ledger path is not canonical absolute")
    body = (
        b"fidelity-suite/campaign-ledger-coordinate.v1\x00"
        + canonical.encode("utf-8"))
    return hashlib.sha256(body).hexdigest()


def _reject_duplicate_pairs(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("duplicate JSON object key %r" % key)
        document[key] = value
    return document

def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant %s" % value)

def _finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number %s" % value)
    return parsed

def _strict_json(raw, label):
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except (UnicodeError, TypeError, ValueError) as exc:
        raise SafetyProofError("%s is not strict UTF-8 JSON: %s" % (label, exc))


def _json_regular(path, label):
    selected = Path(path)
    if not selected.is_file() or selected.is_symlink():
        raise SafetyProofError("%s must be a readable regular file" % label)
    try:
        raw = selected.read_bytes()
    except OSError as exc:
        raise SafetyProofError("%s is not readable: %s" % (label, exc))
    document = _strict_json(raw, label)
    if not isinstance(document, dict):
        raise SafetyProofError("%s must contain an object" % label)
    return document, raw

def _utc(text, label):
    if not isinstance(text, str) or not text.endswith("Z"):
        raise SafetyProofError("%s must be exact UTC ending in Z" % label)
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise SafetyProofError("%s must use YYYY-MM-DDTHH:MM:SSZ" % label)

def _whole_second(value):
    """Truncate a fractional instant to the second before ordering it.

    Every authored proof timestamp -- `issued_at`, `controller_lost_at`,
    `retrieved_at_utc`, `terminate_after`, the lease history -- is a
    whole-second ISO string, while the reaper health stamps and the provider
    deadline observations are raw `time.time()` floats. Ordering a fractional
    instant against a floored one refused a correctly ordered lifecycle
    whenever the proof was sealed inside the same wall second as the final
    poll: the drill of 2026-09-02T20:24Z tore down and reconciled exactly, and
    lost its proof to that truncation alone. The guarantee these chains state
    only ever existed at one-second resolution, so it is checked there; an
    observation a full second late still refuses.
    """
    return value.replace(microsecond=0)

def _verify_blank_seal(document, field, label):
    declared = document.get(field)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise SafetyProofError("%s must carry lowercase SHA-256 %s" % (label, field))
    unsealed = dict(document); unsealed[field] = ""
    if declared != hashlib.sha256(canonical_bytes(unsealed)).hexdigest():
        raise SafetyProofError("%s self-seal mismatch" % label)
    return declared

def _artifact(proof_path, record, label):
    if not isinstance(record, dict):
        raise SafetyProofError("%s artifact binding must be an object" % label)
    relative = record.get("path")
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if (pure is None or not relative or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)):
        raise SafetyProofError(
            "%s artifact path must be canonical and relative" % label)
    proof = Path(proof_path)
    if not proof.is_file() or proof.is_symlink():
        raise SafetyProofError("RunPod safety proof must be a regular file")
    base = proof.resolve().parent
    candidate = base.joinpath(*pure.parts)
    cursor = base
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SafetyProofError("%s artifact path contains a symlink" % label)
    selected = candidate.resolve()
    try:
        selected.relative_to(base)
    except ValueError:
        raise SafetyProofError("%s artifact escapes proof directory" % label)
    if not selected.is_file() or selected.is_symlink():
        raise SafetyProofError("%s artifact is not a regular file" % label)
    raw = selected.read_bytes()
    if (record.get("bytes") != len(raw)
            or record.get("sha256") != hashlib.sha256(raw).hexdigest()):
        raise SafetyProofError("%s artifact transfer identity mismatch" % label)
    return selected, raw

def _strict_archive_json_members(archive, label):
    parsed = {}
    for member in archive.getmembers():
        if not member.isfile() or not member.name.endswith(".json"):
            continue
        stream = archive.extractfile(member)
        if stream is None:
            raise SafetyProofError(
                "%s JSON member %s cannot be read" % (label, member.name))
        parsed[member.name] = _strict_json(
            stream.read(), "%s member %s" % (label, member.name))
    return parsed

def _validate_deadline_observations(
        proof_path, artifact_record, *, provider_account_id, exact_id,
        exact_name, terminate_after, observation_until):
    selected, _raw = _artifact(
        proof_path, artifact_record, "provider deadline observations")
    document, _unused = _json_regular(
        selected, "provider deadline observations")
    document_keys = {
        "schema", "record_sha256", "provider", "provider_account_id",
        "exact_pod_id", "exact_pod_name", "terminate_after",
        "provider_deadline_observation_until", "poll_interval_seconds",
        "poll_duration_max_seconds", "interpoll_gap_max_seconds",
        "observations"}
    if (set(document) != document_keys
            or document.get("schema")
                != "fidelity-suite/runpod-provider-deadline-observations.v1"
            or document.get("provider") != "runpod"
            or document.get("provider_account_id") != provider_account_id
            or document.get("exact_pod_id") != exact_id
            or document.get("exact_pod_name") != exact_name
            or document.get("terminate_after") != terminate_after
            or document.get("provider_deadline_observation_until")
                != observation_until):
        raise SafetyProofError(
            "provider deadline observations do not bind the drill identity")
    _verify_blank_seal(
        document, "record_sha256", "provider deadline observations")
    deadline = _utc(terminate_after, "provider observation terminate_after")
    bound = _utc(
        observation_until, "provider deadline observation bound")
    poll_interval = document.get("poll_interval_seconds")
    poll_duration_max = document.get("poll_duration_max_seconds")
    interpoll_gap_max = document.get("interpoll_gap_max_seconds")
    if (isinstance(poll_interval, bool)
            or not isinstance(poll_interval, int)
            or not 1 <= poll_interval <= 60
            or poll_duration_max != DEADLINE_POLL_DURATION_MAX_SECONDS
            or interpoll_gap_max != DEADLINE_INTERPOLL_GAP_MAX_SECONDS):
        raise SafetyProofError(
            "provider deadline observation poll bounds are invalid")
    observations = document.get("observations")
    if not isinstance(observations, list) or not observations:
        raise SafetyProofError(
            "provider deadline observations require a nonempty complete chain")
    row_keys = {
        "sequence", "poll_started_at_epoch", "poll_completed_at_epoch",
        "poll_completed_at_utc", "deadline_relation", "complete",
        "exact_present", "provider_ids", "resources", "listing_sha256"}
    previous_completed = None
    previous_present = None
    first_started = None
    first_after_deadline = False
    predeadline_presence = False
    first_absence = None
    for sequence, row in enumerate(observations, 1):
        if not isinstance(row, dict) or set(row) != row_keys:
            raise SafetyProofError(
                "provider deadline observation row schema is not exact")
        started = row.get("poll_started_at_epoch")
        completed = row.get("poll_completed_at_epoch")
        prior_completed = previous_completed
        if (row.get("sequence") != sequence
                or isinstance(row.get("sequence"), bool)
                or not isinstance(row.get("sequence"), int)
                or isinstance(started, bool)
                or not isinstance(started, (int, float))
                or isinstance(completed, bool)
                or not isinstance(completed, (int, float))
                or not math.isfinite(float(started))
                or not math.isfinite(float(completed))
                or float(completed) < float(started)
                or float(completed) - float(started) > poll_duration_max
                or (previous_completed is not None
                    and (float(started) < previous_completed
                         or float(started) - previous_completed
                            > interpoll_gap_max
                         or float(completed) <= previous_completed))
                or row.get("complete") is not True
                or not isinstance(row.get("exact_present"), bool)):
            raise SafetyProofError(
                "provider deadline observation timestamps/sequence are invalid")
        started = float(started)
        if first_started is None:
            first_started = started
        completed = float(completed)
        previous_completed = completed
        try:
            completed_utc = datetime.fromtimestamp(
                completed, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            raise SafetyProofError(
                "provider deadline observation timestamp is out of range")
        relation = (
            "BEFORE" if completed < deadline.timestamp()
            else "BOUNDARY" if completed == deadline.timestamp()
            else "AFTER")
        resources = row.get("resources")
        if not isinstance(resources, list):
            raise SafetyProofError(
                "provider deadline observation resources are incomplete")
        normalized = []
        for resource in resources:
            if (not isinstance(resource, dict)
                    or set(resource) != {"id", "name", "status"}
                    or any(
                        not isinstance(resource.get(key), str)
                        or not resource[key]
                        or resource[key].strip() != resource[key]
                        for key in ("id", "name", "status"))):
                raise SafetyProofError(
                    "provider deadline observation resource is malformed")
            normalized.append(dict(resource))
        if (normalized != sorted(normalized, key=lambda item: item["id"])
                or len({item["id"] for item in normalized}) != len(normalized)
                or row.get("provider_ids")
                    != [item["id"] for item in normalized]
                or row.get("listing_sha256")
                    != hashlib.sha256(canonical_bytes(normalized)).hexdigest()
                or row.get("poll_completed_at_utc") != completed_utc
                or row.get("deadline_relation") != relation):
            raise SafetyProofError(
                "provider deadline observation listing/timestamp binding is false")
        exact_rows = [item for item in normalized if item["id"] == exact_id]
        present = bool(exact_rows)
        if ((exact_rows and exact_rows[0]["name"] != exact_name)
                or row["exact_present"] is not present):
            raise SafetyProofError(
                "provider deadline observation exact-pod presence is false")
        if completed >= deadline.timestamp() and not first_after_deadline:
            if (prior_completed is None
                    or prior_completed >= deadline.timestamp()
                    or previous_present is not True):
                raise SafetyProofError(
                    "provider deadline lacks immediate predeadline presence")
            first_after_deadline = True
        if completed < deadline.timestamp():
            if not present:
                raise SafetyProofError(
                    "provider disappeared before authored terminateAfter")
            predeadline_presence = True
        elif not present and first_absence is None:
            first_absence = completed
        elif first_absence is not None and present:
            raise SafetyProofError(
                "provider pod reappeared after complete observed absence")
        previous_present = present
    if (not predeadline_presence or first_absence is None
            or first_absence < deadline.timestamp()
            or first_absence > bound.timestamp()):
        raise SafetyProofError(
            "first complete provider absence is outside the authored API-lag interval")
    return (
        document,
        datetime.fromtimestamp(first_absence, tz=timezone.utc),
        datetime.fromtimestamp(first_started, tz=timezone.utc),
        datetime.fromtimestamp(previous_completed, tz=timezone.utc))


def validate_safety_proof(path, bundle_manifest_sha256,
                          control_manifest_sha256, provider_account_id,
                          campaign_ledger_path, *, now=None):
    """Aggregate real drill artifacts; this function never generates evidence."""
    from .cloudlease import LeaseStore, TERMINAL, HEALTH_SCHEMA
    from .resultsink import verify_archive

    for label, value in (("bundle", bundle_manifest_sha256),
                         ("control-plane", control_manifest_sha256)):
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            raise SafetyProofError("current %s manifest digest is invalid" % label)
    if (not isinstance(provider_account_id, str) or not provider_account_id
            or provider_account_id.strip() != provider_account_id
            or len(provider_account_id) > 256
            or any(ord(character) < 0x21 or ord(character) > 0x7e
                   for character in provider_account_id)):
        raise SafetyProofError("expected live RunPod myself.id is invalid")
    proof, _raw = _json_regular(path, "RunPod safety proof")
    if proof.get("schema") != PROOF_SCHEMA:
        raise SafetyProofError("unsupported RunPod safety proof schema")
    _verify_blank_seal(proof, "proof_sha256", "RunPod safety proof")
    if (proof.get("bundle_manifest_sha256") != bundle_manifest_sha256
            or proof.get("control_manifest_sha256") != control_manifest_sha256
            or proof.get("provider_account_id") != provider_account_id):
        raise SafetyProofError(
            "safety proof is bound to different controller bytes or RunPod account")
    drill = proof.get("drill")
    if (not isinstance(drill, dict) or drill.get("kind") != DRILL_KIND
            or drill.get("paid") is not True or drill.get("provider") != "runpod"
            or drill.get("provider_account_id") != provider_account_id
            or drill.get("termination_mechanism")
                != "autonomous-systemd-user-reaper"
            or drill.get("provider_timer_trusted") is not False):
        raise SafetyProofError(
            "proof is not an explicit paid drill on the expected RunPod account")
    try:
        from .runpoddrill import _snapshot_remote_helpers
        current_helpers = {
            name: digest
            for name, _body, digest in _snapshot_remote_helpers()
        }
    except (OSError, ValueError) as exc:
        raise SafetyProofError("cannot hash current drill helper bytes: %s" % exc)
    if drill.get("remote_helper_sha256") != current_helpers:
        raise SafetyProofError(
            "drill remote helpers differ from current frozen control bytes")
    issued = _utc(proof.get("issued_at"), "issued_at")
    expires = _utc(proof.get("expires_at"), "expires_at")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if issued > current or current > expires or (expires - issued).total_seconds() > 7 * 86400:
        raise SafetyProofError("RunPod safety proof is stale or has an overlong validity window")
    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SafetyProofError("safety proof lacks machine-verifiable artifacts")

    lease_path, _lease_raw = _artifact(path, artifacts.get("lease"), "lease")
    lease = LeaseStore(lease_path.parent).read(lease_path)
    ids = lease.get("provider_resource_ids") or []
    create = lease.get("create") or {}
    exact_id = str(drill.get("exact_pod_id") or "")
    exact_name = str(drill.get("exact_pod_name") or "")
    job_id = str(drill.get("job_id_full") or "")
    attempt_id = str(drill.get("attempt_id") or "")
    if (lease.get("state") != TERMINAL or ids != [exact_id]
            or create.get("exact_name") != exact_name
            or create.get("provider") != "runpod"
            or lease.get("job_hash") != job_id
            or lease.get("attempt_id") != attempt_id):
        raise SafetyProofError(
            "terminal lease does not bind the exact drill job/attempt/pod")
    request = create.get("request") or {}
    request_keys = {
        "drill_mode", "provider_account_id", "campaign_ledger",
        "campaign_attempt_key", "secure_cloud", "offer", "spot", "gpu_type_id",
        "gpu_count", "image_name", "volume_gb", "container_disk_gb",
        "min_vcpu", "min_ram_gb", "network_volume_id",
        "terminate_after", "provider_deadline_observation_until",
        "pre_create_safety", "prepared_create", "producer_checkout",
    }
    expected_resource = {
        "drill_mode": LEASE_DRILL_KIND,
        "provider_account_id": provider_account_id,
        "secure_cloud": True,
        "offer": "on-demand",
        "spot": False,
        "gpu_type_id": DRILL_GPU_TYPE,
        "gpu_count": 1,
        "image_name": DRILL_IMAGE,
        "volume_gb": 20,
        "container_disk_gb": 20,
        "min_vcpu": 4,
        "min_ram_gb": 16,
        "network_volume_id": None,
    }
    if (not isinstance(request, dict) or set(request) != request_keys
            or any(request.get(key) != value
                   for key, value in expected_resource.items())):
        raise SafetyProofError(
            "lease create request is not the exact secure L4 drill profile")
    prepared_create = request.get("prepared_create")
    prepared_identity = (
        prepared_create.get("request_identity")
        if isinstance(prepared_create, dict) else None)
    expected_provider_identity = {
        "cloud_type": "SECURE", "is_spot": False, "offer": "on-demand",
        "gpu_type_id": DRILL_GPU_TYPE, "gpu_count": 1,
        "volume_gb": 20, "container_disk_gb": 20,
        "min_vcpu": 4, "min_ram_gb": 16, "name": exact_name,
        "image_name": DRILL_IMAGE, "terminate_after":
            request.get("terminate_after"),
        "ports": "22/tcp", "volume_mount_path": "/workspace",
        "network_volume_id": None,
    }
    if (not isinstance(prepared_create, dict)
            or set(prepared_create) != {
                "schema", "request_identity", "graphql_body_sha256",
                "graphql_body_bytes", "graphql_body_base64"}
            or prepared_create.get("schema")
                != "fidelity-suite/runpod-prepared-create.v1"
            or not isinstance(prepared_identity, dict)
            or set(prepared_identity)
                != set(expected_provider_identity) | {"public_key_sha256"}
            or any(prepared_identity.get(key) != value
                   for key, value in expected_provider_identity.items())
            or _HEX64.fullmatch(str(
                prepared_identity.get("public_key_sha256", ""))) is None):
        raise SafetyProofError(
            "lease does not bind the exact prebuilt secure create mutation")
    prepared_create_sha256 = hashlib.sha256(
        canonical_bytes(prepared_create)).hexdigest()
    if (drill.get("prepared_create_sha256") != prepared_create_sha256
            or drill.get("graphql_body_sha256")
                != prepared_create.get("graphql_body_sha256")
            or drill.get("graphql_body_bytes")
                != prepared_create.get("graphql_body_bytes")):
        raise SafetyProofError(
            "proof does not bind the prepared GraphQL mutation body")
    producer_checkout = request.get("producer_checkout")
    initial_checkout = (
        producer_checkout.get("initial")
        if isinstance(producer_checkout, dict) else None)
    pre_post_checkout = (
        producer_checkout.get("pre_post")
        if isinstance(producer_checkout, dict) else None)
    checkout_keys = {
        "untracked_files", "status_porcelain_sha256", "status_bytes", "clean"}
    empty_status_sha = hashlib.sha256(b"").hexdigest()
    if (not isinstance(producer_checkout, dict)
            or set(producer_checkout)
                != {"schema", "revision", "initial", "pre_post"}
            or producer_checkout.get("schema")
                != "fidelity-suite/producer-checkout.v1"
            or _HEX40.fullmatch(str(
                producer_checkout.get("revision", ""))) is None
            or not isinstance(initial_checkout, dict)
            or set(initial_checkout) != checkout_keys
            or not isinstance(pre_post_checkout, dict)
            or set(pre_post_checkout) != checkout_keys
            or initial_checkout != {
                "untracked_files": "all",
                "status_porcelain_sha256": empty_status_sha,
                "status_bytes": 0, "clean": True}
            or pre_post_checkout != {
                "untracked_files": "all",
                "status_porcelain_sha256": empty_status_sha,
                "status_bytes": 0, "clean": True}
            or drill.get("producer_checkout") != producer_checkout):
        raise SafetyProofError(
            "proof does not bind clean exact producer HEAD/status evidence")
    campaign_ledger_ref = request.get("campaign_ledger")
    if (not isinstance(campaign_ledger_ref, str) or not campaign_ledger_ref
            or "/" in campaign_ledger_ref or "\\" in campaign_ledger_ref
            or campaign_ledger_ref in (".", "..")
            or request.get("campaign_attempt_key")
                != "%s:%s" % (job_id, attempt_id)):
        raise SafetyProofError(
            "lease campaign coordinate is not private-safe or attempt-exact")
    proof_resource = dict(expected_resource)
    proof_resource.pop("drill_mode")
    if any(drill.get(key) != value
           for key, value in proof_resource.items()):
        raise SafetyProofError(
            "proof drill resource fields differ from immutable create request")
    create_request_sha256 = hashlib.sha256(
        canonical_bytes(request)).hexdigest()
    if (drill.get("create_request_sha256") != create_request_sha256
            or create.get("request_sha256") != create_request_sha256):
        raise SafetyProofError(
            "proof does not bind the exact submitted create request")
    provider_ack = drill.get("provider_acknowledgement")
    if (not isinstance(provider_ack, dict)
            or set(provider_ack) != {"pod_id", "name", "cost_per_hr"}
            or provider_ack.get("pod_id") != exact_id
            or provider_ack.get("name") != exact_name):
        raise SafetyProofError(
            "proof lacks the exact provider id/name/cost acknowledgement")
    host_key_path, _host_key_raw = _artifact(
        path, artifacts.get("ssh_host_key_proof"), "SSH host-key proof")
    host_key_proof, _ = _json_regular(
        host_key_path, "SSH host-key proof")
    host_key_keys = {
        "schema", "proof_sha256", "provider", "provider_id",
        "verified_at_utc", "verification_source", "algorithm",
        "fingerprint", "host", "port", "known_hosts_sha256",
        "provider_log_endpoint_origin", "provider_log_source",
        "provider_log_tail", "provider_log_observed_at_utc",
        "provider_log_line", "provider_log_line_sha256",
        "provider_log_fingerprint",
    }
    provider_log_line = host_key_proof.get("provider_log_line")
    provider_log_line_match = (
        re.fullmatch(
            r"256\s+(SHA256:[A-Za-z0-9+/]{43})\s+\S+\s+\(ED25519\)",
            provider_log_line)
        if isinstance(provider_log_line, str) else None)
    if (set(host_key_proof) != host_key_keys
            or host_key_proof.get("schema")
                != "fidelity-suite/runpod-ssh-host-key-proof.v2"
            or host_key_proof.get("provider") != "runpod"
            or host_key_proof.get("provider_id") != exact_id
            or host_key_proof.get("verification_source")
                != "runpod-authenticated-v2-container-log"
            or host_key_proof.get("provider_log_endpoint_origin")
                != "https://api.runpod.io"
            or host_key_proof.get("provider_log_source") != "container"
            or host_key_proof.get("provider_log_tail") != 5000
            or _HEX64.fullmatch(str(
                host_key_proof.get("provider_log_line_sha256") or "")) is None
            or provider_log_line_match is None
            or hashlib.sha256(
                provider_log_line.encode("utf-8")).hexdigest()
                != host_key_proof.get("provider_log_line_sha256")
            or re.fullmatch(
                r"SHA256:[A-Za-z0-9+/]{43}",
                str(host_key_proof.get(
                    "provider_log_fingerprint") or "")) is None
            or provider_log_line_match.group(1)
                != host_key_proof.get("provider_log_fingerprint")
            or host_key_proof.get("provider_log_fingerprint")
                != host_key_proof.get("fingerprint")
            or host_key_proof.get("algorithm") != "ssh-ed25519"
            or re.fullmatch(
                r"SHA256:[A-Za-z0-9+/]{43}",
                str(host_key_proof.get("fingerprint") or "")) is None
            or not isinstance(host_key_proof.get("host"), str)
            or not host_key_proof["host"]
            or isinstance(host_key_proof.get("port"), bool)
            or not isinstance(host_key_proof.get("port"), int)
            or not 1 <= host_key_proof["port"] <= 65535
            or _HEX64.fullmatch(
                str(host_key_proof.get("known_hosts_sha256") or "")) is None):
        raise SafetyProofError(
            "proof lacks authenticated exact-pod ED25519 SSH host identity")
    _verify_blank_seal(
        host_key_proof, "proof_sha256", "SSH host-key proof")
    verified_at = _utc(
        host_key_proof.get("verified_at_utc"), "SSH host-key verified_at_utc")
    provider_log_observed_at = _utc(
        host_key_proof.get("provider_log_observed_at_utc"),
        "SSH host-key provider-log observed_at_utc")
    if not (provider_log_observed_at <= verified_at
            <= provider_log_observed_at + timedelta(minutes=2)):
        raise SafetyProofError(
            "SSH host-key network verification did not promptly follow the "
            "authenticated provider-log observation")
    if drill.get("ssh_host_key_proof_sha256") != host_key_proof["proof_sha256"]:
        raise SafetyProofError(
            "drill does not bind the authenticated SSH host-key proof")
    try:
        acknowledged_rate = Decimal(str(provider_ack.get("cost_per_hr")))
        drill_rate = Decimal(str(drill.get("live_rate_usd_per_hour")))
    except (InvalidOperation, ValueError):
        raise SafetyProofError("provider acknowledgement rate is not decimal")
    if (not acknowledged_rate.is_finite() or acknowledged_rate <= 0
            or acknowledged_rate != drill_rate):
        raise SafetyProofError(
            "provider acknowledgement cost differs from positive live rate")
    terminate_after = request.get("terminate_after")
    termination = _utc(terminate_after, "lease terminate_after")
    observation_until = _utc(
        request.get("provider_deadline_observation_until"),
        "provider deadline observation bound")
    if (drill.get("terminate_after") != terminate_after
            or drill.get("provider_deadline_observation_until")
            != request.get("provider_deadline_observation_until")
            or observation_until != termination + timedelta(minutes=15)):
        raise SafetyProofError(
            "proof/provider deadline fields differ or lag is not 15 minutes")
    workload = datetime.fromtimestamp(float(create["workload_deadline_epoch"]),
                                      tz=timezone.utc)
    if termination <= workload:
        raise SafetyProofError("provider terminateAfter was not later than workload deadline")
    terminal = lease.get("terminal_proof") or {}
    absence = terminal.get("provider_absence")
    billing = terminal.get("billing_reconciliation")
    if (not isinstance(absence, dict) or absence.get("still_present_ids") != []
            or absence.get("target_provider_ids") != [exact_id]):
        raise SafetyProofError("lease lacks exact-id absence evidence")
    histories = billing.get("billing_histories") if isinstance(billing, dict) else None
    if (not isinstance(billing, dict) or billing.get("reconciled") is not True
            or billing.get("provider_resource_ids") != [exact_id]
            or not isinstance(histories, list) or len(histories) != 1
            or billing.get("total_amount") is None):
        raise SafetyProofError(
            "lease lacks complete exact-pod billing reconciliation")
    bill_doc = histories[0]
    if (bill_doc.get("schema")
            != "fidelity-suite/runpod-billing-evidence.v2"
            or bill_doc.get("pod_id") != exact_id):
        raise SafetyProofError("billing evidence targets a different pod")

    loss_path, _loss_raw = _artifact(
        path, artifacts.get("controller_loss"),
        "controller loss supervisor receipt")
    loss, _unused = _json_regular(
        loss_path, "controller loss supervisor receipt")
    if loss.get("schema") != "fidelity-suite/controller-loss-supervisor.v1":
        raise SafetyProofError(
            "unsupported controller-loss supervisor receipt")
    _verify_blank_seal(loss, "receipt_sha256",
                       "controller loss supervisor receipt")
    (deadline_observation, first_provider_absence,
     first_provider_observation, last_provider_observation) = (
        _validate_deadline_observations(
            path, artifacts.get("provider_deadline_observations"),
            provider_account_id=provider_account_id, exact_id=exact_id,
            exact_name=exact_name, terminate_after=terminate_after,
            observation_until=request.get(
                "provider_deadline_observation_until")))
    if (loss.get("provider_deadline_observations_sha256")
            != deadline_observation["record_sha256"]):
        raise SafetyProofError(
            "controller-loss receipt does not bind deadline observations")
    history = lease.get("history") or []
    create_bound = [row for row in history
                    if row.get("event") == "CREATE_RESPONSE_BOUND"]
    destroy_requested = [row for row in history
                         if row.get("event") == "DESTROY_REQUESTED"]
    absent_events = [row for row in history
                     if row.get("event")
                     == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"]
    post_intents = [row for row in history
                    if row.get("event") == "PROVIDER_POST_INTENT_FSYNCED"]
    prepared = [row for row in history
                if row.get("event") == "LEASE_PREPARED_NO_PROVIDER_POST"]
    if (len(create_bound) != 1 or len(absent_events) != 1
            or len(destroy_requested) != 1):
        raise SafetyProofError(
            "autonomous-reaper drill requires one create, one exact destroy "
            "request, and one exact absence event")
    destroy_evidence = destroy_requested[0].get("evidence") or {}
    if (destroy_evidence.get("provider_ids") != [exact_id]
            or destroy_evidence.get("listed_statuses") != {exact_id: "RUNNING"}
            or destroy_evidence.get("reason")
                != "absolute reap deadline expired"):
        raise SafetyProofError(
            "reaper destroy request does not bind the live exact pod at its "
            "absolute deadline")
    if (len(prepared) != 1 or len(post_intents) != 1
            or history.index(prepared[0]) >= history.index(post_intents[0])
            or history.index(post_intents[0]) >= history.index(create_bound[-1])):
        raise SafetyProofError(
            "lease does not prove PREPARED then one POST intent then create binding")
    post_intent_evidence = post_intents[0].get("evidence") or {}
    if (post_intent_evidence.get("submitted_request_sha256")
            != create_request_sha256
            or post_intent_evidence.get("exact_name") != exact_name):
        raise SafetyProofError(
            "POST intent does not bind the exact prepared create request")
    create_evidence = create_bound[-1].get("evidence") or {}
    accepted = create_evidence.get("response") or {}
    acknowledged_ids = {
        str(accepted.get(key)) for key in ("id", "machine_id", "pod_id")
        if accepted.get(key) is not None}
    try:
        event_rate = Decimal(str(accepted.get("cost_per_hr")))
    except (InvalidOperation, ValueError):
        raise SafetyProofError("create event cost acknowledgement is not decimal")
    if (exact_id not in acknowledged_ids
            or accepted.get("name") != exact_name
            or accepted.get("name_matches_exact") is not True
            or event_rate != acknowledged_rate
            or create_evidence.get("provider_id_acknowledged") != exact_id
            or create_evidence.get("submitted_request_sha256")
            != create.get("request_sha256")
            or create.get("request_sha256") != create_request_sha256):
        raise SafetyProofError(
            "create acknowledgement does not bind exact id/request SHA")
    create_bound_at = _utc(create_bound[-1].get("at"),
                           "create response bound")
    absent_at = _utc(absent_events[-1].get("at"), "exact absence")
    destroy_at = _utc(
        destroy_requested[0].get("at"), "autonomous reaper destroy request")
    if (loss.get("exact_pod_id") != exact_id
            or loss.get("exact_pod_name") != exact_name
            or loss.get("lease_record_sha256") != lease.get("record_sha256")
            or loss.get("controller_pid") != create.get("controller_pid")
            or loss.get("controller_exit_observed") is not True):
        raise SafetyProofError(
            "controller-loss receipt does not bind the terminal lease")
    loss_at = _utc(loss.get("controller_lost_at"),
                   "controller_lost_at")
    if not (create_bound_at <= loss_at < termination
            <= destroy_at <= absent_at <= observation_until):
        raise SafetyProofError(
            "controller loss, reaper deadline, destroy and absence are not ordered")
    kill_path, _kill_raw = _artifact(
        path, artifacts.get("controller_kill_event"),
        "controller kill event")
    kill, _unused = _json_regular(kill_path, "controller kill event")
    if kill.get("schema") != "fidelity-suite/controller-kill-event.v1":
        raise SafetyProofError("unsupported controller kill event")
    kill_seal = _verify_blank_seal(
        kill, "receipt_sha256", "controller kill event")
    if (loss.get("kill_event_sha256") != kill_seal
            or kill.get("controller_pid") != create.get("controller_pid")
            or kill.get("signal") != "SIGKILL"
            or _HEX64.fullmatch(str(kill.get("ready_state_sha256", "")))
            is None):
        raise SafetyProofError(
            "controller kill event does not bind the lost controller")
    killed_at = _utc(kill.get("killed_at"), "controller killed_at")
    if killed_at != loss_at:
        raise SafetyProofError(
            "controller loss timestamp differs from durable kill event")


    health_path, _health_raw = _artifact(
        path, artifacts.get("reaper_health"), "reaper health")
    health, _unused = _json_regular(health_path, "reaper health")
    health_seal = health.get("record_sha256")
    health_unsealed = dict(health)
    health_unsealed.pop("record_sha256", None)
    control = health.get("control")
    actions = health.get("actions")
    expected_account_hash = hashlib.sha256(
        provider_account_id.encode("utf-8")).hexdigest()
    control_keys = {
        "command_sha256", "source_command_sha256",
        "service_unit", "service_unit_sha256",
        "timer_unit", "timer_unit_sha256",
        "source_files", "runtime_files", "interpreter",
        "state_dir_sha256", "lease_dir_sha256", "provider",
        "provider_account_id_sha256", "control_sha256"}
    interpreter = control.get("interpreter") if isinstance(control, dict) else None
    interpreter_version = (
        re.fullmatch(r"([0-9]+)\.([0-9]+)", str(
            (interpreter or {}).get("version") or ""))
        if isinstance(interpreter, dict) else None)
    control_unsealed = dict(control) if isinstance(control, dict) else {}
    claimed_control_sha = control_unsealed.pop("control_sha256", None)
    if (health.get("schema") != HEALTH_SCHEMA
            or health.get("ok") is not True
            or health.get("failure_count") != 0
            or _HEX32.fullmatch(str(health.get("invocation_id", ""))) is None
            or not isinstance(control, dict)
            or set(control) != control_keys
            or control.get("provider") != "runpod"
            or control.get("provider_account_id_sha256")
                != expected_account_hash
            or any(_HEX64.fullmatch(str(control.get(field, ""))) is None
                   for field in (
                       "command_sha256", "source_command_sha256",
                       "service_unit_sha256", "timer_unit_sha256",
                       "state_dir_sha256", "lease_dir_sha256"))
            or control.get("service_unit")
                != "fidelity-cloud-reaper.service"
            or control.get("timer_unit")
                != "fidelity-cloud-reaper.timer"
            or not isinstance(interpreter, dict)
            or set(interpreter) != {
                "executable_path_sha256", "executable_file_sha256",
                "version", "implementation"}
            or _HEX64.fullmatch(str(
                interpreter.get("executable_path_sha256", ""))) is None
            or _HEX64.fullmatch(str(
                interpreter.get("executable_file_sha256", ""))) is None
            or interpreter.get("implementation") != "cpython"
            or interpreter_version is None
            or (int(interpreter_version.group(1)),
                int(interpreter_version.group(2))) < (3, 9)
            or _HEX64.fullmatch(str(claimed_control_sha or "")) is None
            or claimed_control_sha != hashlib.sha256(
                canonical_bytes(control_unsealed)).hexdigest()
            or health_seal != hashlib.sha256(
                canonical_bytes(health_unsealed)).hexdigest()):
        raise SafetyProofError(
            "reaper health lacks exact sealed immutable runtime control")
    destroy_health_path, _destroy_health_raw = _artifact(
        path, artifacts.get("reaper_destroy_health"),
        "autonomous reaper destroy health")
    destroy_health, _unused = _json_regular(
        destroy_health_path, "autonomous reaper destroy health")
    destroy_unsealed = dict(destroy_health)
    destroy_seal = destroy_unsealed.pop("record_sha256", None)
    destroy_actions = destroy_health.get("actions")
    exact_destroy_actions = [
        action for action in (
            destroy_actions if isinstance(destroy_actions, list) else [])
        if (isinstance(action, dict)
            and action.get("action") == "destroy-requested"
            and action.get("provider_id") == exact_id)]
    try:
        destroy_started = datetime.fromtimestamp(
            float(destroy_health.get("invocation_started_at_epoch")),
            tz=timezone.utc)
        destroy_completed = datetime.fromtimestamp(
            float(destroy_health.get("completed_at_epoch")),
            tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        raise SafetyProofError(
            "autonomous destroy health timestamps are invalid")
    if (destroy_health.get("schema") != HEALTH_SCHEMA
            or destroy_health.get("ok") is not True
            or destroy_health.get("failure_count") != 0
            or destroy_health.get("control") != control
            or _HEX32.fullmatch(str(
                destroy_health.get("invocation_id", ""))) is None
            or _HEX64.fullmatch(str(destroy_seal or "")) is None
            or destroy_seal != hashlib.sha256(
                canonical_bytes(destroy_unsealed)).hexdigest()
            or len(exact_destroy_actions) != 1
            or _HEX64.fullmatch(str(
                exact_destroy_actions[0].get(
                    "lease_record_sha256", ""))) is None
            or isinstance(
                exact_destroy_actions[0].get("lease_generation"), bool)
            or not isinstance(
                exact_destroy_actions[0].get("lease_generation"), int)
            or loss.get("reaper_destroy_health_sha256") != destroy_seal
            or not (loss_at < destroy_started
                    and _whole_second(destroy_started) <= destroy_at
                    and destroy_at <= destroy_completed
                    and destroy_completed <= first_provider_absence)):
        raise SafetyProofError(
            "autonomous reaper health does not prove the exact post-loss "
            "deadline destroy")
    source_files = control["source_files"]
    runtime_files = control["runtime_files"]
    allowed_control_paths = {
        "bin/reap_cloud_leases.py",
        "bin/fidelity/__init__.py",
        "bin/fidelity/cloudlease.py",
        "bin/fidelity/campaign.py",
        "bin/fidelity/common.py",
        "bin/fidelity/runpodapi.py",
        "bin/fidelity/jlapi.py",
        "bin/fidelity/sshbase.py",
    }
    for label, rows in (
            ("source", source_files), ("runtime", runtime_files)):
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, dict) for row in rows)
                or rows != sorted(rows, key=lambda row: row.get("path", ""))
                or len({row.get("path") for row in rows}) != len(rows)):
            raise SafetyProofError(
                "reaper %s control-file identity is incomplete" % label)
        for row in rows:
            relative = str(row.get("path") or "")
            pure = PurePosixPath(relative)
            if (set(row) != {"path", "size", "sha256"}
                    or pure.is_absolute() or pure.as_posix() != relative
                    or any(part in ("", ".", "..") for part in pure.parts)
                    or "\\" in relative
                    or _HEX64.fullmatch(str(row.get("sha256", ""))) is None
                    or isinstance(row.get("size"), bool)
                    or not isinstance(row.get("size"), int)
                    or row["size"] < 0):
                raise SafetyProofError(
                    "reaper %s control-file identity is unsafe" % label)
        if {row["path"] for row in rows} != allowed_control_paths:
            raise SafetyProofError(
                "reaper %s files are not the exact control closure" % label)
    if source_files != runtime_files:
        raise SafetyProofError(
            "reaper runtime snapshot differs from verified source closure")
    checkout_root = Path(__file__).resolve().parents[2]
    for row in source_files:
        local_control = checkout_root.joinpath(
            *PurePosixPath(row["path"]).parts)
        try:
            local_body = local_control.read_bytes()
        except OSError as exc:
            raise SafetyProofError(
                "current reaper source file is unavailable: %s" % exc)
        if (local_control.is_symlink()
                or len(local_body) != row["size"]
                or hashlib.sha256(local_body).hexdigest() != row["sha256"]):
            raise SafetyProofError(
                "installed reaper source differs from current checkout")
    invocation_started = datetime.fromtimestamp(
        float(health["invocation_started_at_epoch"]), tz=timezone.utc)
    reaped_at = datetime.fromtimestamp(
        float(health["completed_at_epoch"]), tz=timezone.utc)
    billed_at = _utc(
        bill_doc.get("retrieved_at_utc"), "billing retrieved_at_utc")
    api_lag_limit = termination + timedelta(minutes=15)
    if not create_bound_at <= loss_at:
        raise SafetyProofError(
            "controller loss precedes the durable provider-id binding")
    if not loss_at <= first_provider_observation < termination:
        raise SafetyProofError(
            "controller loss was not observed before the absolute reap deadline")
    if not (termination <= destroy_at
            and destroy_at <= first_provider_absence
            and _whole_second(first_provider_absence) <= api_lag_limit):
        raise SafetyProofError(
            "autonomous destroy or first exact absence exceeded the "
            "15-minute provider-lag bound")
    if not destroy_at <= absent_at <= api_lag_limit:
        raise SafetyProofError(
            "durable exact absence exceeded the 15-minute provider-lag bound")
    if ((first_provider_observation - loss_at).total_seconds()
            > deadline_observation["interpoll_gap_max_seconds"]):
        raise SafetyProofError(
            "first post-loss provider observation exceeded the authored "
            "inter-poll bound")
    if not (first_provider_absence <= last_provider_observation
            and _whole_second(last_provider_observation) <= issued):
        raise SafetyProofError(
            "provider observations do not remain ordered through proof issue")
    if not (loss_at < invocation_started
            and _whole_second(invocation_started) <= billed_at
            and billed_at <= reaped_at
            and _whole_second(reaped_at) <= issued):
        raise SafetyProofError(
            "billing reconciliation and final healthy reaper invocation are "
            "not ordered after controller loss")
    billing_path, _billing_raw = _artifact(
        path, artifacts.get("billing_arithmetic"), "billing arithmetic")
    billing_receipt, _unused = _json_regular(
        billing_path, "billing arithmetic")
    billing_seal = _verify_blank_seal(
        billing_receipt, "receipt_sha256", "billing arithmetic")
    amount_fields = ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount")
    try:
        billing_total = Decimal(str(billing_receipt.get("total_amount")))
        lease_total = Decimal(str(billing.get("total_amount")))
        expected_sums = {
            field: sum(
                (Decimal(str(row.get(field)))
                 for row in bill_doc.get("records") or []),
                Decimal(0))
            for field in amount_fields
        }
    except (InvalidOperation, ValueError):
        raise SafetyProofError("billing records/totals contain non-decimal amounts")
    if (billing_receipt.get("schema")
            != "fidelity-suite/runpod-drill-billing-arithmetic.v1"
            or billing_receipt.get("pod_id") != exact_id
            or billing_receipt.get("job_id_full") != job_id
            or billing_receipt.get("attempt_id") != attempt_id
            or billing_receipt.get("lease_record_sha256")
            != lease.get("record_sha256")
            or billing_receipt.get("billing_evidence_sha256")
            != hashlib.sha256(canonical_bytes(bill_doc)).hexdigest()
            or billing_receipt.get("record_count")
            != len(bill_doc.get("records") or [])
            or billing_receipt.get("validated_sums")
            != {key: format(value, "f")
                for key, value in expected_sums.items()}
            or billing_total != lease_total
            or billing_total != expected_sums["totalAmount"]):
        raise SafetyProofError(
            "billing arithmetic does not bind exact lease/pod/job/charge")

    campaign_path, _campaign_raw = _artifact(
        path, artifacts.get("campaign_release"), "campaign release")
    campaign, _unused = _json_regular(campaign_path, "campaign release")
    campaign_seal = _verify_blank_seal(
        campaign, "receipt_sha256", "campaign release")
    actual_quote = campaign.get("actual_quote")
    ledger_path, ledger_raw = _artifact(
        path, artifacts.get("campaign_ledger"), "campaign ledger")
    ledger_document, _unused = _json_regular(
        ledger_path, "campaign ledger")
    try:
        live_ledger_path = Path(campaign_ledger_path)
    except TypeError:
        raise SafetyProofError("current durable campaign ledger path is invalid")
    if (live_ledger_path.name != campaign_ledger_ref
            or campaign.get("campaign_ledger_path_sha256")
                != campaign_ledger_coordinate_sha256(live_ledger_path)):
        raise SafetyProofError(
            "proof campaign ledger differs from configured durable coordinate")
    try:
        from .campaign import CampaignLedger, CampaignLedgerError, CostQuote
        CampaignLedger._validate_document(ledger_document)
        live_ledger_document = CampaignLedger(
            str(live_ledger_path), "runpod", provider_account_id).snapshot()
        settled_sum = sum(
            (Decimal(str((item.get("billing") or {}).get("final_charge_usd")))
             for item in (ledger_document.get("attempts") or {}).values()
             if item.get("billing") is not None),
            Decimal(0))
        settled_total = Decimal(str(ledger_document.get("settled_charges_usd")))
        live_settled_total = Decimal(
            str(live_ledger_document.get("settled_charges_usd")))
    except (CampaignLedgerError, InvalidOperation, ValueError, TypeError) as exc:
        raise SafetyProofError("campaign ledger validation failed: %s" % exc)
    if settled_sum != settled_total:
        raise SafetyProofError(
            "campaign settled total differs from exact attempt billing")
    attempt_key = "%s:%s" % (job_id, attempt_id)
    ledger_attempt = (ledger_document.get("attempts") or {}).get(attempt_key)
    live_ledger_attempt = (
        (live_ledger_document.get("attempts") or {}).get(attempt_key))
    immutable_campaign_fields = {
        "schema", "currency", "provider", "provider_account_id",
        "hard_ceiling_usd", "reserve_floor_usd",
        "cleanup_reaper_margin_usd", "max_concurrent_attempts"}
    if (any(live_ledger_document.get(field) != ledger_document.get(field)
            for field in immutable_campaign_fields)
            or live_ledger_document.get("generation", -1)
                < ledger_document.get("generation", 0)
            or live_settled_total < settled_total
            or live_ledger_attempt != ledger_attempt):
        raise SafetyProofError(
            "current campaign does not retain the exact released drill attempt")
    deletion_proof = (
        (((ledger_attempt or {}).get("deletion") or {}).get("proofs") or {})
        .get(exact_id) or {})
    ledger_billing = (ledger_attempt or {}).get("billing") or {}
    expected_deletion_proof = hashlib.sha256(
        canonical_bytes(
            (lease.get("terminal_proof") or {}).get("provider_absence"))
    ).hexdigest()
    expected_billing_proof = hashlib.sha256(
        canonical_bytes(lease.get("billing_reconciliation"))).hexdigest()
    if (hashlib.sha256(ledger_raw).hexdigest()
            != campaign.get("campaign_ledger_sha256")
            or ledger_document.get("provider") != "runpod"
            or ledger_document.get("provider_account_id") != provider_account_id
            or ledger_document.get("generation") != campaign.get("ledger_generation")
            or ledger_document.get("settled_charges_usd")
                != campaign.get("settled_charges_usd")
            or ledger_document.get("max_concurrent_attempts") != 2
            or ledger_document.get("authorized_concurrent_attempts") != 1
            or ledger_document.get("width_authorization") is not None
            or (ledger_document.get("inventory") or {}).get("complete") is not True
            or (ledger_document.get("inventory") or {}).get(
                "provider_resources") != []
            or (ledger_document.get("inventory") or {}).get(
                "unknown_resources") != []
            or not isinstance(ledger_attempt, dict)
            or ledger_attempt.get("job_hash") != job_id
            or ledger_attempt.get("attempt") != attempt_id
            or ledger_attempt.get("reservation_kind")
                != "bootstrap-controller-loss-drill"
            or ledger_attempt.get("phase") != "RECONCILED"
            or ledger_attempt.get("provider_ids") != [exact_id]
            or ledger_attempt.get("reserved_quote") != campaign.get("reserved_quote")
            or ledger_attempt.get("actual_quote") != campaign.get("actual_quote")
            or ledger_attempt.get("maximum_remaining_liability_usd") != "0"
            or ledger_attempt.get("released") is not True
            or deletion_proof.get("proof") != expected_deletion_proof
            or ledger_billing.get("provider_ids") != [exact_id]
            or ledger_billing.get("final_charge_usd")
                != campaign.get("final_charge_usd")
            or ledger_billing.get("proof") != expected_billing_proof):
        raise SafetyProofError(
            "campaign ledger bytes do not independently prove exact release")
    try:
        reserved_quote = CostQuote.from_dict(campaign.get("reserved_quote"))
        actual_quote_value = CostQuote.from_dict(actual_quote)
        actual_rate = actual_quote_value.live_compute_usd_per_hour
        final_charge = Decimal(str(campaign.get("final_charge_usd")))
    except (InvalidOperation, ValueError, TypeError):
        raise SafetyProofError(
            "campaign quotes/final charge are not valid cost evidence")
    if (final_charge < 0
            or final_charge > reserved_quote.hard_cap_usd
            or final_charge > actual_quote_value.hard_cap_usd):
        raise SafetyProofError(
            "drill final charge is negative or exceeds its hard cap")
    if (campaign.get("schema")
            != "fidelity-suite/runpod-drill-campaign-release.v1"
            or campaign.get("attempt_key") != "%s:%s" % (job_id, attempt_id)
            or campaign.get("job_id_full") != job_id
            or campaign.get("attempt_id") != attempt_id
            or campaign.get("provider") != "runpod"
            or campaign.get("provider_account_id") != provider_account_id
            or campaign.get("campaign_ledger_path_sha256")
                != campaign_ledger_coordinate_sha256(live_ledger_path)
            or campaign.get("provider_id") != exact_id
            or campaign.get("lease_record_sha256")
            != lease.get("record_sha256")
            or campaign.get("released") is not True
            or campaign.get("phase") != "RECONCILED"
            or campaign.get("maximum_remaining_liability_usd") != "0"
            or not isinstance(campaign.get("reserved_quote"), dict)
            or not isinstance(actual_quote, dict)
            or actual_rate != acknowledged_rate
            or final_charge != billing_total):
        raise SafetyProofError(
            "campaign receipt does not prove exact reconciled zero-liability release")

    result_path, result_raw = _artifact(
        path, artifacts.get("result_archive"), "result archive")
    result_record = artifacts["result_archive"]
    verified = verify_archive(
        result_path, expected_sha256=result_record["sha256"],
        expected_bytes=result_record["bytes"])
    transfer_path, _transfer_raw = _artifact(
        path, artifacts.get("result_transfer"), "result transfer")
    transfer, _unused = _json_regular(transfer_path, "result transfer")
    transfer_seal = _verify_blank_seal(
        transfer, "receipt_sha256", "result transfer")
    if (transfer.get("schema")
            != "fidelity-suite/runpod-drill-result-transfer.v1"
            or transfer.get("path") != "result-bundle.tar.gz"
            or transfer.get("bytes") != result_record["bytes"]
            or transfer.get("sha256") != result_record["sha256"]
            or transfer.get("job_id_full") != job_id
            or loss.get("result_transfer_receipt_sha256") != transfer_seal
            or loss.get("result_archive_sha256")
            != hashlib.sha256(result_raw).hexdigest()
            or loss.get("result_archive_bytes") != len(result_raw)):
        raise SafetyProofError(
            "on-pod transfer receipt/off-pod archive identity does not match")
    try:
        with tarfile.open(str(result_path), mode="r:gz") as archive:
            parsed_json = _strict_archive_json_members(
                archive, "drill result archive")
            member = archive.getmember("job.json")
            if not member.isfile() or member.size > 4 * 1024 * 1024:
                raise SafetyProofError("archived drill job is not a bounded regular file")
            archived_job = parsed_json.get("job.json")
            if not isinstance(archived_job, dict):
                raise SafetyProofError("cannot read archived drill job")
    except SafetyProofError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, tarfile.TarError) as exc:
        raise SafetyProofError(
            "cannot verify archived drill job: %s" % exc.__class__.__name__)
    from .jobcontract import JobContractError, verify_job
    archived_attempt = archived_job.get("execution_attempt") or {}
    expected_workload = datetime.fromtimestamp(
        float(create["workload_deadline_epoch"]), tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        archived_job_id = verify_job(archived_job)
    except (JobContractError, TypeError, ValueError) as exc:
        raise SafetyProofError("archived drill job is invalid: %s" % exc)
    if (archived_job_id != job_id
            or (verified.get("manifest") or {}).get("job_id_full") != job_id
            or archived_job.get("bundle_contract_sha256")
            != bundle_manifest_sha256
            or (archived_job.get("control_plane") or {}).get(
                "manifest_sha256") != control_manifest_sha256
            or archived_attempt.get("attempt_id") != attempt_id
            or archived_attempt.get("lease_path") != lease_path.name
            or archived_attempt.get("provider_terminate_after")
            != request["terminate_after"]
            or archived_attempt.get("workload_deadline_utc")
            != expected_workload):
        raise SafetyProofError(
            "verified result job differs from drill/current bundle/control")
    return {
        "proof": _strict_json(canonical_bytes(proof), "canonical proof"),
        "lease": lease, "result_archive": verified,
        "job": archived_job, "billing": billing_receipt,
        "campaign_release": campaign, "campaign_ledger": ledger_document,
    }

def validate_unexpected_tensor_allowlist(path, *, target_repo, target_revision,
                                         suite_root=None):
    expected = _ALLOWLISTS.get((target_repo, target_revision))
    if expected is None:
        raise SafetyProofError("no checked-in authored allowlist exists for exact target revision")
    candidate = Path(path)
    if candidate.is_symlink():
        raise SafetyProofError("allowlist input is a symlink")
    selected = candidate.resolve()
    root = (Path(suite_root).resolve() if suite_root is not None
            else Path(__file__).resolve().parents[2])
    authored = (root / expected["path"]).resolve()
    if selected != authored:
        raise SafetyProofError(
            "allowlist must be the checked-in authored artifact %s"
            % expected["path"])
    if not selected.is_file() or selected.is_symlink():
        raise SafetyProofError("authored allowlist is missing or symlinked")
    raw = selected.read_bytes()
    names = _strict_json(raw, "authored allowlist")
    if (hashlib.sha256(raw).hexdigest() != expected["artifact_sha256"]
            or not isinstance(names, list) or len(names) != expected["count"]
            or names != sorted(set(names))):
        raise SafetyProofError("authored allowlist raw identity or exact-name closure differs")
    names_sha = hashlib.sha256(canonical_bytes(names)).hexdigest()
    if names_sha != expected["canonical_sorted_names_sha256"]:
        raise SafetyProofError("authored allowlist canonical name digest differs")
    return {"path": expected["path"], "artifact_sha256": expected["artifact_sha256"],
            "canonical_sorted_names_sha256": names_sha, "count": len(names)}

def _safe_member(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise SafetyProofError("archive contains unsafe member name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise SafetyProofError("archive member escapes destination: %r" % name)
    return str(path)

def extract_bundle_archive(archive_path, manifest_path, destination,
                           expected_sha256, expected_bytes):
    """Extract an exact regular-file bundle into a new atomic run root."""
    from .jobcontract import verify_bundle_manifest
    archive = Path(archive_path)
    manifest, _raw = _json_regular(
        Path(manifest_path), "bundle manifest")
    verify_bundle_manifest(manifest)
    expected = {row["path"]: row for row in manifest["files"]}
    raw_digest = hashlib.sha256()
    size = 0
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            raw_digest.update(chunk); size += len(chunk)
    if size != int(expected_bytes) or raw_digest.hexdigest() != expected_sha256:
        raise SafetyProofError("bundle archive transfer identity mismatch")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise SafetyProofError("fresh bundle destination must be absent")
    staging = target.with_name(".%s.%d.staging" % (target.name, os.getpid()))
    if staging.exists() or staging.is_symlink():
        raise SafetyProofError("bundle staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        with tarfile.open(str(archive), mode="r:gz") as tar:
            members = tar.getmembers()
            if ([member.name for member in members] != sorted(expected)
                    or set(member.name for member in members) != set(expected)):
                raise SafetyProofError("bundle archive member closure differs")
            for member in members:
                row = expected[member.name]
                if (not member.isfile() or member.issym() or member.islnk()
                        or member.size != row["bytes"]):
                    raise SafetyProofError("unsafe bundle member %s" % member.name)
                source = tar.extractfile(member)
                data = source.read() if source else b""
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise SafetyProofError("bundle member digest mismatch")
                output = staging.joinpath(*PurePosixPath(member.name).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(data)
        os.replace(str(staging), str(target))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return {"path": str(target), "files": len(expected),
            "archive_sha256": expected_sha256, "archive_bytes": int(expected_bytes)}


def extract_bound_panel_archive(archive_path, binding_path, destination):
    """Verify exact regular-file member closure before atomic extraction."""
    binding, _raw = _json_regular(binding_path, "panel binding")
    content = binding.get("content")
    identity = content.get("archive") if isinstance(content, dict) else None
    manifest = content.get("manifest") if isinstance(content, dict) else None
    if not isinstance(identity, dict) or not isinstance(manifest, list):
        raise SafetyProofError("panel binding lacks archive manifest")
    digest, size = hashlib.sha256(), 0
    try:
        with Path(archive_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block); size += len(block)
    except OSError as exc: raise SafetyProofError("cannot read panel archive: %s" % exc)
    if size != identity.get("bytes") or digest.hexdigest() != identity.get("sha256"):
        raise SafetyProofError("panel archive transfer identity differs from binding")
    expected = {}
    for row in manifest:
        if not isinstance(row, dict): raise SafetyProofError("panel manifest entry is not an object")
        name = _safe_member(row.get("path"))
        if (name in expected or not isinstance(row.get("bytes"), int) or row["bytes"] < 0
                or _HEX64.fullmatch(str(row.get("sha256", ""))) is None):
            raise SafetyProofError("invalid or duplicate panel manifest entry %s" % name)
        expected[name] = row
    target = Path(destination)
    if target.exists(): raise SafetyProofError("panel extraction destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=".panel-extract-"))
    try:
        with tarfile.open(str(archive_path), mode="r:", format=tarfile.USTAR_FORMAT) as tar:
            actual = {}
            for member in tar.getmembers():
                name = _safe_member(member.name)
                if not member.isfile() or name in actual:
                    raise SafetyProofError("panel archive has non-regular or duplicate member %s" % name)
                actual[name] = member
            if set(actual) != set(expected):
                raise SafetyProofError("panel archive member closure differs from binding")
            for name in sorted(expected):
                member, row = actual[name], expected[name]
                source = tar.extractfile(member); data = source.read() if source else b""
                if (member.size != row["bytes"] or len(data) != row["bytes"]
                        or hashlib.sha256(data).hexdigest() != row["sha256"]):
                    raise SafetyProofError("panel member identity mismatch for %s" % name)
                output = staging.joinpath(*PurePosixPath(name).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream: stream.write(data)
        os.replace(str(staging), str(target))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return {"path": str(target), "files": len(expected),
            "archive_sha256": digest.hexdigest(), "archive_bytes": size}


def _validate_fruit_panel_binding(binding: Mapping[str, Any]) -> None:
    if not isinstance(binding, dict) or set(binding) != {
            "schema", "panel", "receipt", "tokenizer", "content"}:
        raise SafetyProofError("width 2 Fruit resolved panel keys differ")
    panel = binding.get("panel")
    receipt = binding.get("receipt")
    tokenizer = binding.get("tokenizer")
    content = binding.get("content")
    expected_panel = {
        "id": FRUIT_PANEL_ID,
        "name": "GLM-5.2-SIQ-Fruit held-out fidelity panel v1 -- 16 windows x 2048",
        "role": "final",
        "contexts": FRUIT_PANEL_CONTEXTS,
        "context_length": 2048,
        "positions_per_context": 2047,
        "scored_positions_total": FRUIT_PANEL_SCORED_POSITIONS,
        "suite_token_hash_sha256": FRUIT_PANEL_SUITE_TOKEN_SHA256,
        "file": "panel.json",
        "bytes": 17542,
        "sha256": FRUIT_PANEL_FILE_SHA256,
    }
    expected_receipt = {
        "file": "panel.receipt.json",
        "bytes": 8171,
        "declared_receipt_sha256": FRUIT_RECEIPT_DECLARED_SHA256,
        "receipt_seal_mode": "self-blank",
        "receipt_file_sha256": FRUIT_RECEIPT_FILE_SHA256,
    }
    if (binding.get("schema") != "malaiwah.resolved-panel.v1"
            or panel != expected_panel or receipt != expected_receipt):
        raise SafetyProofError(
            "width 2 proof is not the exact 16-context Fruit panel/receipt")
    if (not isinstance(tokenizer, dict)
            or set(tokenizer) != {
                "id", "repository", "revision", "vocab_size",
                "maximum_token_id_exclusive", "identity_sha256", "files",
                "files_verified", "receipt"}
            or tokenizer.get("id") != FRUIT_REPO
            or tokenizer.get("repository") != FRUIT_REPO
            or tokenizer.get("revision") != FRUIT_REVISION
            or tokenizer.get("vocab_size") != 154820
            or tokenizer.get("maximum_token_id_exclusive") != 154820
            or tokenizer.get("identity_sha256")
                != FRUIT_TOKENIZER_IDENTITY_SHA256
            or tokenizer.get("files_verified") is not True
            or tokenizer.get("receipt") is not None):
        raise SafetyProofError(
            "width 2 Fruit tokenizer pin/verified receipt identity differs")
    files = tokenizer.get("files")
    expected_file_hashes = {
        "tokenizer.json":
            "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config.json":
            "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    }
    if (not isinstance(files, list)
            or [row.get("name") for row in files]
                != sorted(expected_file_hashes)
            or any(
                set(row) != {"name", "bytes", "sha256"}
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int)
                or row["bytes"] <= 0
                or row.get("sha256") != expected_file_hashes[row["name"]]
                for row in files)):
        raise SafetyProofError("width 2 Fruit tokenizer files differ")
    if not isinstance(content, dict) or set(content) != {
            "manifest", "manifest_sha256", "archive"}:
        raise SafetyProofError("width 2 Fruit panel content binding differs")
    rows = content.get("manifest")
    expected_names = {"panel.json", "panel.receipt.json"}
    expected_names.update(
        "arrays/final-%04d.%s.npy" % (index, kind)
        for index in range(FRUIT_PANEL_CONTEXTS)
        for kind in ("mask", "tokens"))
    if (not isinstance(rows, list) or len(rows) != len(expected_names)
            or {row.get("path") for row in rows} != expected_names
            or any(
                set(row) != {"path", "bytes", "sha256"}
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int)
                or row["bytes"] <= 0
                or _HEX64.fullmatch(str(row.get("sha256", ""))) is None
                for row in rows)
            or content.get("manifest_sha256")
                != hashlib.sha256(canonical_bytes(rows)).hexdigest()):
        raise SafetyProofError("width 2 Fruit panel content manifest differs")
    rows_by_name = {row["path"]: row for row in rows}
    if (rows_by_name["panel.json"] != {
            "path": "panel.json", "bytes": 17542,
            "sha256": FRUIT_PANEL_FILE_SHA256}
            or rows_by_name["panel.receipt.json"] != {
                "path": "panel.receipt.json", "bytes": 8171,
                "sha256": FRUIT_RECEIPT_FILE_SHA256}):
        raise SafetyProofError("width 2 Fruit panel metadata digests differ")
    archive = content.get("archive")
    if (not isinstance(archive, dict)
            or set(archive) != {
                "format", "compression", "algorithm", "bytes", "sha256"}
            or archive.get("format") != "ustar"
            or archive.get("compression") != "none"
            or archive.get("algorithm") != (
                "sha256(ustar: sorted regular files; "
                "mode=0644; uid=gid=mtime=0)")
            or isinstance(archive.get("bytes"), bool)
            or not isinstance(archive.get("bytes"), int)
            or archive["bytes"] <= 0
            or _HEX64.fullmatch(str(archive.get("sha256", ""))) is None):
        raise SafetyProofError("width 2 Fruit panel archive binding differs")


def validate_current_public_root(publication: Mapping[str, Any]) -> Dict[str, Any]:
    """Anonymously refetch the exact immutable published root identities."""
    import urllib.error
    import urllib.request
    from . import dshub, jobcontract, panel as panel_contract
    from .dsformat import verify_manifest_seal

    if not isinstance(publication, dict):
        raise SafetyProofError("root publication receipt is unavailable")
    repository = publication.get("repository")
    revision = publication.get("revision")
    if (publication.get("schema") != "fidelity.publish-root-receipt.v2"
            or _verify_blank_seal(
                publication, "receipt_sha256", "root publication receipt")
                != publication.get("receipt_sha256")
            or publication.get("private") is not False
            or publication.get("revision_immutable") is not True
            or publication.get("verified_anonymously") is not True
            or publication.get("verified_after_publish") is not True
            or publication.get("verified_revision") != revision
            or not isinstance(repository, str)
            or re.fullmatch(r"[^\s/]+/[^\s/]+", repository) is None
            or _HEX64.fullmatch(
                str(publication.get("result_archive_sha256", ""))) is None
            or isinstance(publication.get("result_archive_bytes"), bool)
            or not isinstance(publication.get("result_archive_bytes"), int)
            or publication["result_archive_bytes"] <= 0
            or _HEX40.fullmatch(str(revision or "")) is None):
        raise SafetyProofError("root publication receipt is not immutable public proof")

    def anonymous(path_name: str) -> bytes:
        url = dshub.resolve_url(
            repository, revision, path_name, repo_type="datasets")
        request = urllib.request.Request(
            url, headers={"User-Agent": "quant-fidelity-suite/public-proof"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(16 * 1024 * 1024 + 1)
                if response.getcode() != 200:
                    raise SafetyProofError(
                        "anonymous root refetch returned HTTP %s"
                        % response.getcode())
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise SafetyProofError(
                "anonymous root refetch failed for %s: %s"
                % (path_name, exc.__class__.__name__))
        if len(body) > 16 * 1024 * 1024:
            raise SafetyProofError(
                "anonymous root refetch exceeded bounded evidence size")
        return body

    manifest_raw = anonymous("fidelity-dataset.json")
    qualification_raw = anonymous("receipts/root-qualification.json")
    panel_receipt_raw = anonymous("panel/panel-receipt.json")
    manifest = _strict_json(
        manifest_raw, "current public fidelity-dataset.json")
    qualification = _strict_json(
        qualification_raw, "current public root qualification")
    contract = (
        qualification.get("job_contract")
        if isinstance(qualification, dict) else None)
    try:
        jobcontract.validate_root_qualification_contract(contract)
    except jobcontract.JobContractError as exc:
        raise SafetyProofError(
            "current public root job contract is invalid: %s" % exc)
    expected_license = contract.get("weights_license")
    expected_runtime_license = (
        None if expected_license is None else {
            "source_file": "LICENSE",
            "dataset_path": expected_license.get("dataset_path"),
            "bytes": expected_license.get("bytes"),
            "sha256": expected_license.get("sha256"),
        })
    if expected_license is None:
        license_raw = None
        license_matches = contract.get("dataset_license") == "mit"
    else:
        license_raw = anonymous("LICENSE")
        license_matches = (
            contract.get("dataset_license") == "other"
            and len(license_raw) == expected_license.get("bytes")
            and hashlib.sha256(license_raw).hexdigest()
                == expected_license.get("sha256"))
    if not license_matches:
        raise SafetyProofError(
            "current public root source-license bytes differ from qualification")
    dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
    weights = manifest.get("weights") if isinstance(manifest, dict) else None
    panel = manifest.get("panel") if isinstance(manifest, dict) else None
    capture = manifest.get("capture") if isinstance(manifest, dict) else None
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    manifest_tokenizer = (
        panel.get("tokenizer") if isinstance(panel, dict) else None)
    contract_target = contract["target"]
    contract_binding = contract["panel_resolved_binding"]
    if (not isinstance(dataset, dict)
            or not isinstance(weights, dict)
            or not isinstance(panel, dict)
            or not isinstance(capture, dict)
            or not isinstance(runtime, dict)
            or not isinstance(manifest_tokenizer, dict)
            or dataset.get("role") != "root"
            or dataset.get("id") != contract.get("dataset_id")
            or dataset.get("name") != contract.get("dataset_name")
            or (dataset.get("author") or {}).get("name")
                != contract.get("author")
            or dataset.get("repository")
                != contract.get("dataset_repository")
            or dataset.get("license") != contract.get("dataset_license")
            or weights.get("repository") != contract_target.get("repo_id")
            or weights.get("revision") != contract_target.get("revision")
            or panel.get("panel_id") != contract.get("panel_id")
            or panel.get("suite_token_hash_sha256")
                != contract.get("panel_suite_token_hash_sha256")
            or panel.get("panel_receipt_sha256")
                != contract.get("panel_receipt_sha256")
            or panel.get("panel_receipt_file")
                != "panel/panel-receipt.json"
            or manifest_tokenizer.get("identity_sha256")
                != contract.get("tokenizer_identity_sha256")
            or capture.get("form") != contract.get("form")
            or str(capture.get("dtype", "")).lower()
                not in ({str(contract.get("dtype", "")).lower()}
                        | ({"bf16", "bfloat16"}
                           if str(contract.get("dtype", "")).lower()
                           in ("bf16", "bfloat16") else set()))
            or runtime.get("lane") != contract.get("lane")
            or any(
                contract_binding["panel"].get(name)
                != contract.get(contract_name)
                for name, contract_name in (
                    ("id", "panel_id"),
                    ("suite_token_hash_sha256",
                     "panel_suite_token_hash_sha256")))):
        raise SafetyProofError(
            "current public root manifest differs from qualification contract")
    try:
        panel_contract.verify_bound_panel_receipt_bytes(
            contract_binding["receipt"], panel_receipt_raw,
            "current public panel receipt")
    except panel_contract.PanelError as exc:
        raise SafetyProofError(str(exc)) from exc
    captures = qualification.get("captures") if isinstance(
        qualification, dict) else None
    canonical_capture = captures.get("canonical") if isinstance(
        captures, dict) else None
    canonical_panel = (
        canonical_capture.get("panel")
        if isinstance(canonical_capture, dict) else None)
    canonical_allowlist = (
        canonical_capture.get("unexpected_tensor_allowlist")
        if isinstance(canonical_capture, dict) else None)
    contract_allowlist = contract["unexpected_tensor_allowlist"]
    expected_binding_evidence = {
        "binding_file": Path(contract["panel_binding_path"]).name,
        "binding_file_sha256": contract["panel_binding_file_sha256"],
        "binding": contract_binding,
    }
    manifest_file_sha = hashlib.sha256(manifest_raw).hexdigest()
    if (not isinstance(manifest, dict) or not verify_manifest_seal(manifest)
            or not isinstance(canonical_capture, dict)
            or canonical_capture.get("dataset_id")
                != contract.get("dataset_id")
            or canonical_capture.get("dataset_name")
                != contract.get("dataset_name")
            or canonical_capture.get("dataset_author")
                != contract.get("author")
            or canonical_capture.get("dataset_repository")
                != contract.get("dataset_repository")
            or canonical_capture.get("dataset_license")
                != contract.get("dataset_license")
            or canonical_capture.get("weights_license")
                != expected_runtime_license
            or (expected_license is None and (
                canonical_capture.get("weights_license_file_sha256") is not None
                or canonical_capture.get("weights_license_file_bytes") is not None))
            or (expected_license is not None and (
                canonical_capture.get("weights_license_file_sha256")
                    != expected_license.get("sha256")
                or canonical_capture.get("weights_license_file_bytes")
                    != expected_license.get("bytes")))
            or canonical_capture.get("weights_repository")
                != contract_target.get("repo_id")
            or canonical_capture.get("weights_revision")
                != contract_target.get("revision")
            or canonical_capture.get("capture_form") != contract.get("form")
            or str(canonical_capture.get("capture_dtype", "")).lower()
                not in ({str(contract.get("dtype", "")).lower()}
                        | ({"bf16", "bfloat16"}
                           if str(contract.get("dtype", "")).lower()
                           in ("bf16", "bfloat16") else set()))
            or canonical_capture.get("runtime_lane") != contract.get("lane")
            or canonical_capture.get("runtime_device")
                != contract.get("device")
            or canonical_capture.get("runtime_engine")
                != "transformers-eager"
            or canonical_capture.get("capture_tool_file")
                != "engines/tools/hf_capture.py"
            or canonical_capture.get("capture_schedule")
                != contract.get("schedule")
            or not isinstance(canonical_panel, dict)
            or canonical_panel.get("panel_id") != contract.get("panel_id")
            or canonical_panel.get("suite_token_hash_sha256")
                != contract.get("panel_suite_token_hash_sha256")
            or canonical_panel.get("panel_receipt_sha256")
                != contract.get("panel_receipt_sha256")
            or not isinstance(canonical_panel.get("tokenizer"), dict)
            or canonical_panel["tokenizer"].get("identity_sha256")
                != contract.get("tokenizer_identity_sha256")
            or canonical_panel.get("resolved_binding_evidence")
                != expected_binding_evidence
            or not isinstance(canonical_allowlist, dict)
            or canonical_allowlist.get("artifact_sha256")
                != contract_allowlist.get("artifact_sha256")
            or canonical_allowlist.get("canonical_sorted_names_sha256")
                != contract_allowlist.get("canonical_sorted_names_sha256")
            or canonical_allowlist.get("exact_match") is not True
            or canonical_allowlist.get("duplicate_observed_keys") != []
            or canonical_allowlist.get("missing_keys") != []
            or canonical_allowlist.get("extra_keys") != []
            or manifest_file_sha
                != canonical_capture.get("dataset_manifest_file_sha256")
            or manifest.get("dataset_sha256")
                != canonical_capture.get("dataset_sha256")
            or manifest.get("dataset_sha256")
                != publication.get("dataset_sha256")
            or manifest.get("dataset_sha256")
                != publication.get("published_dataset_sha256")):
        raise SafetyProofError(
            "current public fidelity-dataset.json differs from publication")
    qualification_file_sha = hashlib.sha256(qualification_raw).hexdigest()
    if (not isinstance(qualification, dict)
            or qualification.get("schema")
                != "fidelity.root-qualification-receipt.v1"
            or _verify_blank_seal(
                qualification, "receipt_sha256",
                "public root qualification")
                != publication.get("qualification_receipt_sha256")
            or qualification_file_sha
                != publication.get("qualification_file_sha256")
            or qualification_file_sha
                != publication.get("published_qualification_file_sha256")
            or qualification.get("dataset_repository") != repository
            or qualification.get("destination_repository") != repository
            or contract.get("dataset_repository") != repository
            or contract.get("publish_root_to") != repository):
        raise SafetyProofError(
            "current public root qualification differs from publication")
    return {
        "repository": repository,
        "revision": revision,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_manifest_file_sha256": manifest_file_sha,
        "qualification_file_sha256": qualification_file_sha,
        "publicly_accessible": True,
    }

def validate_width_two_root_archive(path, expected_job: Mapping[str, Any]):
    """Validate the immutable published 16-context Fruit root canary."""
    from .engines import resolve_root_timing
    from .jobcontract import JobContractError, verify_job
    from .resultsink import verify_archive
    verified = verify_archive(path)
    try:
        with tarfile.open(str(path), mode="r:gz") as tar:
            parsed_json = _strict_archive_json_members(
                tar, "width 2 root archive")
            job_member = tar.getmember("job.json")
            publication_member = tar.getmember("receipts/publish-root.json")
            if not job_member.isfile() or not publication_member.isfile():
                raise SafetyProofError(
                    "width 2 root archive identity members are not regular files")
            archived = parsed_json.get("job.json")
            publication = parsed_json.get("receipts/publish-root.json")
    except SafetyProofError:
        raise
    except (KeyError, OSError, UnicodeError, ValueError, tarfile.TarError) as exc:
        raise SafetyProofError(
            "width 2 root archive lacks exact publication evidence: %s" % exc)
    manifest = verified.get("manifest") or {}
    if not isinstance(archived, dict) or archived.get("role") != "root":
        raise SafetyProofError("width 2 proof archive is not a root result")
    try:
        archived_job_id = verify_job(archived)
    except (JobContractError, TypeError, ValueError) as exc:
        raise SafetyProofError("width 2 root job contract is invalid: %s" % exc)
    if (manifest.get("status")
            not in ("ok", "complete", "completed", "success", "succeeded")
            or manifest.get("role") != "root"
            or manifest.get("verb") != "capture"
            or manifest.get("job_id_full") != archived_job_id):
        raise SafetyProofError(
            "width 2 root result did not complete the exact archived job")
    target = archived.get("target") or {}
    if (target.get("repo_id") != FRUIT_REPO
            or target.get("revision") != FRUIT_REVISION
            or target.get("model_bytes") != 10102776813
            or target.get("config_sha256")
                != "5a19697e555fff140d1b089b852c3ef227114b196f8d76796560feeeb34dc44a"
            or target.get("index_sha256")
                != "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56"):
        raise SafetyProofError("width 2 proof is not the fixed Fruit root pin")
    for key in (
            "bundle", "bundle_registry", "bundle_contract_sha256",
            "control_plane"):
        if archived.get(key) != expected_job.get(key):
            raise SafetyProofError(
                "width 2 root result differs in current %s binding" % key)
    current_producer = expected_job.get("produced_by") or {}
    archived_producer = archived.get("produced_by") or {}
    for key in (
            "tool", "repository", "revision", "entrypoint",
            "entrypoint_sha256"):
        if archived_producer.get(key) != current_producer.get(key):
            raise SafetyProofError(
                "width 2 root job producer digest differs in %s" % key)
    expected_profile = {
        "profile_id": "root-hf-transformers-bf16",
        "lane": "root",
        "source": "native",
        "surface": "native-bf16",
        "form": "hidden",
        "engine": "hf-transformers",
        "compute_dtype": "bfloat16",
        "device": "cuda",
        "schedule": "two-fresh-process-qualification",
    }
    expected_runtime = {
        "min_vcpu_count": 4,
        "min_memory_gb": 32,
        "gpu_count": 1,
        "device": "cuda",
        "expert_parallel": {"mode": "single_device", "world_size": 1},
        "reduce_order": "fp32",
        "capacity_basis": "controller-conservative-capacity",
    }
    expected_timing = resolve_root_timing(
        target_repo=FRUIT_REPO, target_revision=FRUIT_REVISION,
        gpu="L4", form="hidden",
        schedule="two-fresh-process-qualification")
    if (archived.get("profile") != expected_profile
            or archived.get("timing") != expected_timing
            or archived.get("runtime") != expected_runtime
            or archived.get("reduce_order") != "fp32"
            or "scoring" in archived):
        raise SafetyProofError(
            "width 2 Fruit root profile/timing/runtime/scoring contract differs")
    panel = archived.get("panel") or {}
    binding = panel.get("resolved_binding") or {}
    if (not isinstance(panel, dict) or set(panel) != {
            "resolved_binding", "binding_path", "binding_file_sha256",
            "archive_path", "archive_bytes", "archive_sha256", "content_path"}
            or panel.get("binding_path") != "inputs/panel.binding.json"
            or panel.get("binding_file_sha256")
                != hashlib.sha256(canonical_bytes(binding)).hexdigest()
            or panel.get("archive_path") != "inputs/panel.tar"
            or panel.get("archive_bytes")
                != ((binding.get("content") or {}).get("archive") or {}).get(
                    "bytes")
            or panel.get("archive_sha256")
                != ((binding.get("content") or {}).get("archive") or {}).get(
                    "sha256")
            or panel.get("content_path") != "inputs/panel"):
        raise SafetyProofError("width 2 Fruit outer panel binding differs")
    _validate_fruit_panel_binding(binding)
    fruit = _ALLOWLISTS[(FRUIT_REPO, FRUIT_REVISION)]
    capture = archived.get("capture") or {}
    allowlist = capture.get("unexpected_tensor_allowlist") or {}
    for key in ("path", "artifact_sha256", "canonical_sorted_names_sha256"):
        if allowlist.get(key) != fruit[key]:
            raise SafetyProofError("width 2 Fruit allowlist differs in %s" % key)
    protocol = capture.get("root_protocol") or {}
    replay = capture.get("replay") or {}
    if (protocol.get("schedule") != "two-fresh-process-qualification"
            or protocol.get("fresh_processes") != 2
            or protocol.get("run_count_per_process") != 1
            or protocol.get("exact_self_comparison") is not True
            or protocol.get("qualification_required") is not True
            or protocol.get("publication_mode") != "canonical-public"):
        raise SafetyProofError(
            "width 2 Fruit proof lacks exact two-process qualification")
    if (replay != {
            "device": "numpy", "dtype": "float32", "vocab_chunk": 8192}
            or capture.get("replay_device") != "numpy"
            or capture.get("replay_dtype") != "float32"
            or capture.get("vocab_chunk") != 8192):
        raise SafetyProofError(
            "width 2 Fruit proof lacks exact NumPy float32 replay")
    positions = FRUIT_PANEL_SCORED_POSITIONS
    hidden_size, vocab_size, bytes_per_element = 1024, 154880, 2
    hidden_bytes = positions * hidden_size * bytes_per_element
    shared_head_bytes = vocab_size * hidden_size * bytes_per_element
    capture_bytes = hidden_bytes + shared_head_bytes
    archive_uncompressed = capture_bytes * 2 + 67108864
    archive_transfer = (
        archive_uncompressed
        + ((archive_uncompressed + 16382) // 16383) * 5 + 64)
    expected_storage = {
        "form": "hidden",
        "storage_dtype": "bfloat16",
        "selected_prediction_positions": positions,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "bytes_per_element": bytes_per_element,
        "fresh_processes": 2,
        "hidden_bytes_per_process": hidden_bytes,
        "shared_head_bytes_per_process": shared_head_bytes,
        "bytes_per_process": capture_bytes,
        "capture_bytes_total": capture_bytes * 2,
        "capture_archive_duplicate_upper_bound_bytes": capture_bytes * 2,
        "required_dataset_trees": 2,
        "result_archive_max_members": positions * 2 + 128,
        "result_archive_max_uncompressed_bytes": archive_uncompressed,
        "result_archive_max_transfer_bytes": archive_transfer,
    }
    expected_archive_contract = {
        key: expected_storage[key] for key in (
            "required_dataset_trees", "result_archive_max_members",
            "result_archive_max_uncompressed_bytes",
            "result_archive_max_transfer_bytes")
    }
    if (target.get("root_capture_storage") != expected_storage
            or target.get("result_archive_contract")
                != expected_archive_contract):
        raise SafetyProofError(
            "width 2 Fruit result-archive contract differs")
    dataset_repository = capture.get("dataset_repository")
    if (not isinstance(dataset_repository, str)
            or re.fullmatch(r"[^\s/]+/[^\s/]+", dataset_repository) is None
            or capture.get("publish_root_to") != dataset_repository
            or manifest.get("publication_requested") is not True
            or not isinstance(publication, dict)
            or publication.get("schema")
                != "fidelity.publish-root-receipt.v2"
            or publication.get("repository") != dataset_repository
            or _HEX40.fullmatch(str(publication.get("revision", ""))) is None
            or publication.get("revision_immutable") is not True
            or publication.get("verified_after_publish") is not True
            or publication.get("verified_revision")
                != publication.get("revision")):
        raise SafetyProofError(
            "width 2 Fruit root lacks actual immutable publication")
    result = dict(verified)
    result["publication"] = publication
    return result

def verify_watchdog(fs_root, deadline_epoch, heartbeat_timeout_seconds):
    root = Path(fs_root)
    armed_path = root / "receipts" / "watchdog-armed.json"
    armed, _raw = _json_regular(armed_path, "watchdog arming receipt")
    if (armed.get("schema") != "fidelity-suite/watchdog-armed.v2"
            or armed.get("deadline_epoch") != int(deadline_epoch)
            or armed.get("heartbeat_timeout_seconds")
            != int(heartbeat_timeout_seconds)):
        raise SafetyProofError("watchdog arming receipt differs from controller")
    pid = armed.get("watchdog_pid")
    ticks = armed.get("proc_start_ticks")
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            or isinstance(ticks, bool) or not isinstance(ticks, int)):
        raise SafetyProofError("watchdog receipt has invalid process identity")
    try:
        live_ticks = int(
            Path("/proc/%d/stat" % pid).read_text(encoding="ascii").split()[21])
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        raise SafetyProofError("recorded watchdog process is not live")
    if live_ticks != ticks:
        raise SafetyProofError("watchdog PID was reused")
    return armed


def disarm_watchdog(fs_root, deadline_epoch, heartbeat_timeout_seconds):
    armed = verify_watchdog(
        fs_root, deadline_epoch, heartbeat_timeout_seconds)
    os.killpg(int(armed["watchdog_pgid"]), 15)
    receipt = {
        "schema": "fidelity-suite/watchdog-disarmed.v1",
        "watchdog_pid": armed["watchdog_pid"],
        "watchdog_pgid": armed["watchdog_pgid"],
        "proc_start_ticks": armed["proc_start_ticks"],
        "deadline_epoch": int(deadline_epoch),
        "heartbeat_timeout_seconds": int(heartbeat_timeout_seconds),
        "disarmed_at_epoch": int(datetime.now(timezone.utc).timestamp()),
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    target = Path(fs_root) / "receipts" / "watchdog-disarmed.json"
    temporary = target.with_name(".%s.%d.tmp" % (target.name, os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    os.replace(str(temporary), str(target))
    return receipt


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("extract-panel", "extract-bundle",
                            "verify-watchdog", "disarm-watchdog"))
    parser.add_argument("--archive")
    parser.add_argument("--binding")
    parser.add_argument("--destination")
    parser.add_argument("--fs-root")
    parser.add_argument("--manifest")
    parser.add_argument("--sha256")
    parser.add_argument("--bytes", type=int)
    parser.add_argument("--deadline", type=int)
    parser.add_argument("--heartbeat-timeout", type=int)
    args = parser.parse_args(argv)
    if args.command == "extract-panel":
        if not args.archive or not args.binding or not args.destination:
            parser.error("extract-panel requires archive, binding and destination")
        result = extract_bound_panel_archive(
            args.archive, args.binding, args.destination)
    elif args.command == "extract-bundle":
        result = extract_bundle_archive(
            args.archive, args.manifest, args.destination,
            args.sha256, args.bytes)
    elif args.command == "verify-watchdog":
        result = verify_watchdog(
            args.fs_root, args.deadline, args.heartbeat_timeout)
    else:
        result = disarm_watchdog(
            args.fs_root, args.deadline, args.heartbeat_timeout)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__": raise SystemExit(_main())
