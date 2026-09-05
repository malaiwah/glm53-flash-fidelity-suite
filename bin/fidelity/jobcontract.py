#!/usr/bin/env python3
"""Canonical scientific/execution identity for ``job.json``.

The complete finalized document is the stage contract.  Identity excludes only
three top-level controller bookkeeping fields; every other present field,
including fields unknown to this version, is deliberately hashed.
"""
from __future__ import annotations

import copy
import base64
import binascii
import hashlib
import json
import re
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Any, Dict

# Run-root base per declared RunPod storage layout (measure_cloud
# RUNPOD_STORAGE_LAYOUTS is the authoring side; this is the validating side).
RUNPOD_RUN_BASES = {"container-disk": "/root", "pod-volume": "/workspace"}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_EXCLUDED_TOP_LEVEL = frozenset(("job_id", "job_id_full", "execution_attempt"))
DISPLAY_HEX = 16


class JobContractError(ValueError):
    """A job document is not canonical JSON or carries the wrong identity."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JobContractError("job document is not finite canonical JSON: %s" % exc)


def canonical_relative_path(value: object, label: str) -> PurePosixPath:
    """Validate and return an exact portable relative POSIX path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise JobContractError("%s must be a canonical relative path" % label)
    pure = PurePosixPath(value)
    if (pure.is_absolute() or pure.as_posix() != value
            or any(part in ("", ".", "..") for part in pure.parts)):
        raise JobContractError("%s must be a canonical relative path" % label)
    return pure


def finalize_bundle_manifest(files: Any, source: str) -> Dict[str, Any]:
    """Canonicalize the exact regular-file identities shared by every transport."""
    if not isinstance(source, str) or not source:
        raise JobContractError("bundle source must be non-empty")
    if not isinstance(files, list) or not files:
        raise JobContractError("bundle files must be a non-empty array")
    rows = []
    seen = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise JobContractError("bundle files[%d] must be an object" % index)
        path = row.get("path")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (pure is None or not path or pure.is_absolute()
                or pure.as_posix() != path
                or any(part in ("", ".", "..") for part in pure.parts)
                or "\\" in path or path in seen):
            raise JobContractError(
                "bundle path is unsafe or duplicated: %r" % path)
        size, digest = row.get("bytes"), row.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or not isinstance(digest, str) or _HEX64.fullmatch(digest) is None):
            raise JobContractError("bundle file identity is invalid for %s" % path)
        seen.add(path)
        rows.append({"path": path, "bytes": size, "sha256": digest})
    rows.sort(key=lambda item: item["path"])
    return {
        "schema": "fidelity-suite/bundle-manifest.v1",
        "source": source,
        "files": rows,
        "manifest_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def verify_bundle_manifest(bundle: dict) -> str:
    """Verify and return the shared exact-file manifest digest."""
    if not isinstance(bundle, dict) or bundle.get("schema") != (
            "fidelity-suite/bundle-manifest.v1"):
        raise JobContractError("unsupported bundle manifest schema")
    expected = finalize_bundle_manifest(bundle.get("files"), bundle.get("source"))
    if bundle != expected:
        raise JobContractError(
            "bundle manifest is noncanonical or has unknown fields")
    return expected["manifest_sha256"]


IMPORTED_CAPTURE_SCHEMA = "fidelity-suite/imported-capture.v1"
_IMPORTED_CAPTURE_KEYS = frozenset({
    "schema", "receipt_sha256", "imported_at", "job_id_full", "attempt_id",
    "dataset_sha256", "capture_content_digest",
    "dataset_manifest_file_sha256", "archive_sha256", "archive_bytes",
    "manifest_sha256", "file_count", "origin", "source_path", "resealed_from"})

#: capture.resume_capture on a resumed root. `resealed_from` is null for a
#: dataset imported exactly as its pod sealed it, or the identity of the
#: seal it was re-sealed from (fidelity-dataset reseal: the validator's
#: verdict named a private path; tensors and capture_content_digest are
#: unchanged and the receipt travels inside the tree). The chain is named
#: in the job, the import receipt, the qualification and the archive so no
#: reader has to infer it from the dataset alone.
RESUME_CAPTURE_KEYS = frozenset({
    "dataset_sha256", "capture_content_digest",
    "dataset_manifest_file_sha256", "origin", "resealed_from"})
RESEALED_FROM_KEYS = frozenset({
    "dataset_sha256", "reason", "receipt", "receipt_sha256"})


def valid_resealed_from(value: Any) -> bool:
    """None, or the exact re-seal origin block (see RESUME_CAPTURE_KEYS)."""
    if value is None:
        return True
    return (isinstance(value, dict)
            and set(value) == RESEALED_FROM_KEYS
            and _HEX64.fullmatch(str(value.get("dataset_sha256", ""))) is not None
            and _HEX64.fullmatch(str(value.get("receipt_sha256", ""))) is not None
            and value.get("receipt") == "validation/reseal-receipt.json"
            and isinstance(value.get("reason"), str)
            and bool(value.get("reason")))


def build_imported_capture_receipt(*, job_id_full: str, attempt_id: str,
                                   resume: Dict[str, Any], archive_sha256: str,
                                   archive_bytes: int, manifest_sha256: str,
                                   file_count: int, source_path: str,
                                   imported_at: str) -> Dict[str, Any]:
    """The controller's sealed statement of which sealed dataset it imported.

    Written to receipts/imported-capture.json beside the extracted dataset/;
    the capture stage refuses to treat dataset/ as cold run 1 without it, and
    qualify-root records it as the canonical capture's origin.
    """
    from .common import seal
    doc = {
        "schema": IMPORTED_CAPTURE_SCHEMA,
        "imported_at": imported_at,
        "job_id_full": job_id_full,
        "attempt_id": attempt_id,
        "dataset_sha256": resume["dataset_sha256"],
        "capture_content_digest": resume["capture_content_digest"],
        "dataset_manifest_file_sha256": resume["dataset_manifest_file_sha256"],
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "manifest_sha256": manifest_sha256,
        "file_count": int(file_count),
        "origin": resume.get("origin"),
        "source_path": source_path,
        "resealed_from": resume.get("resealed_from"),
    }
    return seal(doc)


def verify_imported_capture_receipt(doc: Any, *, job: Dict[str, Any],
                                    dataset_sha256: str,
                                    dataset_manifest_file_sha256: str) -> Dict[str, Any]:
    """Refuse unless the receipt, the job and the dataset in dataset/ agree."""
    from .common import verify_seal
    resume = ((job.get("capture") or {}).get("resume_capture")
              if isinstance(job.get("capture"), dict) else None)
    if not isinstance(resume, dict):
        raise JobContractError(
            "job.json does not declare capture.resume_capture; an imported "
            "dataset/ is never adopted as a fresh cold run")
    if not isinstance(doc, dict) or set(doc) != _IMPORTED_CAPTURE_KEYS:
        raise JobContractError("imported-capture receipt keys differ from v1")
    if doc.get("schema") != IMPORTED_CAPTURE_SCHEMA or not verify_seal(doc):
        raise JobContractError("imported-capture receipt schema/seal is invalid")
    attempt = (job.get("execution_attempt") or {})
    expected = {
        "job_id_full": job.get("job_id_full"),
        "attempt_id": attempt.get("attempt_id"),
        "dataset_sha256": resume["dataset_sha256"],
        "capture_content_digest": resume["capture_content_digest"],
        "dataset_manifest_file_sha256": resume["dataset_manifest_file_sha256"],
        "origin": resume.get("origin"),
        "resealed_from": resume.get("resealed_from"),
    }
    for name, value in expected.items():
        if doc.get(name) != value:
            raise JobContractError(
                "imported-capture receipt %s differs from job.json" % name)
    if not valid_resealed_from(doc.get("resealed_from")):
        raise JobContractError("imported-capture receipt resealed_from is invalid")
    if (doc["dataset_sha256"] != dataset_sha256
            or doc["dataset_manifest_file_sha256"]
            != dataset_manifest_file_sha256):
        raise JobContractError(
            "dataset/ is not the sealed dataset the job names for resumption")
    if (isinstance(doc.get("archive_bytes"), bool)
            or not isinstance(doc.get("archive_bytes"), int)
            or doc["archive_bytes"] <= 0
            or _HEX64.fullmatch(str(doc.get("archive_sha256", ""))) is None
            or _HEX64.fullmatch(str(doc.get("manifest_sha256", ""))) is None
            or isinstance(doc.get("file_count"), bool)
            or not isinstance(doc.get("file_count"), int)
            or doc["file_count"] <= 0):
        raise JobContractError("imported-capture receipt transfer identity is invalid")
    return doc

def validate_job(document: dict) -> None:
    """Validate the role-neutral structure common to every job.v2 executor."""
    if not isinstance(document, dict) or document.get("schema") != (
            "fidelity-suite/job.v2"):
        raise JobContractError("job schema must be fidelity-suite/job.v2")
    if document.get("role") not in ("quant", "root"):
        raise JobContractError("job role must be quant or root")
    target = document.get("target")
    if (not isinstance(target, dict)
            or not isinstance(target.get("repo_id"), str)
            or not target["repo_id"]
            or re.fullmatch(r"[0-9a-f]{40}",
                            str(target.get("revision", ""))) is None):
        raise JobContractError(
            "job target requires repo_id and exact lowercase 40-hex revision")
    verify_bundle_manifest(document.get("bundle"))
    for name in ("config_sha256", "index_sha256",
                 "shard_manifest_sha256"):
        if _HEX64.fullmatch(str(target.get(name, ""))) is None:
            raise JobContractError("job target lacks exact %s" % name)
    if (isinstance(target.get("model_bytes"), bool)
            or not isinstance(target.get("model_bytes"), int)
            or target["model_bytes"] <= 0):
        raise JobContractError("job target model_bytes must be positive")
    shards = target.get("shards")
    if not isinstance(shards, list) or not shards:
        raise JobContractError("job target requires a non-empty shard census")
    for shard in shards:
        if (not isinstance(shard, dict)
                or set(shard) != {"path", "bytes"}
                or not isinstance(shard["path"], str)
                or PurePosixPath(shard["path"]).as_posix() != shard["path"]
                or PurePosixPath(shard["path"]).is_absolute()
                or "\\" in shard["path"]
                or any(part in ("", ".", "..")
                       for part in PurePosixPath(shard["path"]).parts)
                or isinstance(shard["bytes"], bool)
                or not isinstance(shard["bytes"], int)
                or shard["bytes"] <= 0):
            raise JobContractError("job target shard census is noncanonical")
    if shards != sorted(shards, key=lambda row: row["path"]):
        raise JobContractError("job target shard census is not sorted")
    if len({row["path"] for row in shards}) != len(shards):
        raise JobContractError("job target shard paths must be unique")
    expected_shard_digest = hashlib.sha256(
        _canonical_bytes(shards)).hexdigest()
    if target["shard_manifest_sha256"] != expected_shard_digest:
        raise JobContractError("job target shard manifest digest mismatch")
    if target["model_bytes"] != sum(row["bytes"] for row in shards):
        raise JobContractError("job target model_bytes differs from shards")
    target_path = target.get("path")
    if target_path is not None:
        canonical_relative_path(target_path, "job target path")
    download_manifest = target.get("download_manifest")
    if (not isinstance(download_manifest, list)
            or download_manifest != sorted(
                download_manifest, key=lambda row: row.get("path", ""))
            or any(
                not isinstance(row, dict)
                or set(row) != {"path", "bytes"}
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int)
                or row["bytes"] < 0
                for row in download_manifest)):
        raise JobContractError(
            "job target download manifest is noncanonical")
    for row in download_manifest:
        canonical_relative_path(row["path"], "download manifest path")
    if (len({row["path"] for row in download_manifest})
            != len(download_manifest)):
        raise JobContractError("job target download paths must be unique")
    if target.get("download_bytes_total") != sum(
            row["bytes"] for row in download_manifest):
        raise JobContractError(
            "job target download byte total differs from manifest")
    if target.get("download_manifest_sha256") != hashlib.sha256(
            _canonical_bytes(download_manifest)).hexdigest():
        raise JobContractError(
            "job target download manifest digest mismatch")
    if document.get("recipe") != "runpod-controller-loss-drill":
        download_by_path = {
            row["path"]: row["bytes"] for row in download_manifest}
        # A GGUF build ships neither file: its model-class config is the
        # reference release's (copied beside the build on the pod) and its
        # tensor table stands where the index stands (target.index_sha256 is
        # the table digest, target.index_source says so).
        if target.get("surface") != "gguf":
            for required_path in (
                    "config.json", "model.safetensors.index.json"):
                if download_by_path.get(required_path, 0) <= 0:
                    raise JobContractError(
                        "job target download manifest lacks positive %s"
                        % required_path)
        elif not all(str(shard["path"]).endswith(".gguf") for shard in shards) \
                or not isinstance(target.get("index_source"), str):
            raise JobContractError(
                "job target of surface gguf must list .gguf shards and name its index_source")
        for shard in shards:
            if download_by_path.get(shard["path"]) != shard["bytes"]:
                raise JobContractError(
                    "job target download manifest differs from shard census")
    if document.get("cold_runs") != 2:
        raise JobContractError("job requires exactly two cold runs")
    if not isinstance(document.get("profile"), dict) or not document["profile"]:
        raise JobContractError("job profile must be a non-empty object")
    if not isinstance(document.get("timing"), dict) or not document["timing"]:
        raise JobContractError("job timing must be a non-empty object")
    if not isinstance(document.get("lane"), str) or not document["lane"]:
        raise JobContractError("job lane must be non-empty")
    profile = document["profile"]
    if profile.get("lane") not in (document["lane"], "root"):
        raise JobContractError("job profile lane differs from job lane")
    if document.get("recipe") != "runpod-controller-loss-drill":
        for name in ("surface", "codec", "bits", "path"):
            if name not in target:
                raise JobContractError(
                    "job target lacks authored %s identity" % name)
        if not isinstance(document.get("runtime"), dict):
            raise JobContractError("job runtime contract must be an object")
        if not isinstance(document.get("environment"), dict):
            raise JobContractError("job environment contract must be an object")
        if not isinstance(document.get("produced_by"), dict):
            raise JobContractError("job produced_by contract must be an object")
        if not isinstance(document.get("measurer"), dict):
            raise JobContractError("job measurer contract must be an object")
        resources = document.get("resource_requirements")
        required_resource_keys = {
            "workspace_available_bytes_minimum",
            "container_available_bytes_minimum",
            "min_vcpu_count", "min_memory_gb", "expected_vram_bytes",
        }
        if (not isinstance(resources, dict)
                or set(resources) != required_resource_keys
                or any(
                    isinstance(resources[name], bool)
                    or not isinstance(resources[name], int)
                    or resources[name] <= 0
                    for name in required_resource_keys)):
            raise JobContractError(
                "job resource requirements are noncanonical")
    control = document.get("control_plane")
    if (not isinstance(control, dict)
            or control.get("schema")
            != "fidelity-suite/control-plane-manifest.v1"):
        raise JobContractError("job control-plane manifest is unsupported")
    canonical_control = finalize_bundle_manifest(
        control.get("files"), control.get("source"))
    canonical_control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    if control != canonical_control:
        raise JobContractError("control-plane manifest is noncanonical")
    registry = document.get("bundle_registry")
    if (not isinstance(registry, dict)
            or set(registry) != {"path", "bytes", "sha256"}
            or registry.get("path") != "bin/BUNDLE.txt"
            or isinstance(registry.get("bytes"), bool)
            or not isinstance(registry.get("bytes"), int)
            or registry["bytes"] <= 0
            or _HEX64.fullmatch(str(registry.get("sha256", ""))) is None):
        raise JobContractError("bundle registry binding is noncanonical")
    binding = hashlib.sha256(_canonical_bytes({
        "bundle": document.get("bundle"), "registry": registry})).hexdigest()
    if document.get("bundle_contract_sha256") != binding:
        raise JobContractError("bundle contract digest mismatch")
    if not isinstance(document.get("scope"), dict) or not document["scope"]:
        raise JobContractError("job scope must be complete")
    panel = document.get("panel")
    if not isinstance(panel, dict) or not panel:
        raise JobContractError("job panel binding must be complete")
    if document["role"] == "root":
        capture = document.get("capture")
        allowlist = (
            capture.get("unexpected_tensor_allowlist")
            if isinstance(capture, dict) else None)
        # The allowlist is authored per pin and absent for a model that
        # carries no tensors beyond its architecture; the pod-side engine
        # refuses unexpected tensors on its own when none is given.
        allowlist_ok = allowlist is None or (
            isinstance(allowlist, dict)
            and set(allowlist) == {
                "path", "artifact_sha256",
                "canonical_sorted_names_sha256"}
            and all(_HEX64.fullmatch(str(allowlist.get(name, ""))) is not None
                    for name in (
                        "artifact_sha256",
                        "canonical_sorted_names_sha256")))
        if (not isinstance(capture, dict)
                or capture.get("engine") != "hf-transformers"
                or capture.get("dtype") != "bfloat16"
                or capture.get("device") != "cuda"
                or capture.get("schedule") != "layer-outer"
                or capture.get("replay_device") != "numpy"
                or capture.get("replay_dtype") != "float32"
                or capture.get("vocab_chunk") != 8192
                or capture.get("replay") != {
                    "device": "numpy", "dtype": "float32",
                    "vocab_chunk": 8192}
                or (capture.get("publish_root_to") is not None
                    and capture.get("publish_root_to")
                    != capture.get("dataset_repository"))
                or not allowlist_ok
                or not valid_candidate(capture.get("candidate"))):
            raise JobContractError("root capture contract is incomplete")
        candidate = capture.get("candidate")
        if candidate is not None:
            if candidate["reference"]["panel_id"] != (
                    ((panel.get("resolved_binding") or {}).get("panel") or {}).get("id")):
                raise JobContractError(
                    "candidate reference dataset was captured on another panel")
            if candidate["reference"]["repository"] == capture.get("dataset_repository"):
                raise JobContractError(
                    "candidate dataset repository must differ from its reference")
        # A resumed root: cold run 1 is a sealed dataset from an earlier
        # attempt of the same recipe, imported into dataset/ before the
        # stages run; cold run 2 is captured fresh. The job names the exact
        # dataset it expects so the pod refuses any other import.
        resume = capture.get("resume_capture")
        if resume is not None:
            origin = resume.get("origin") if isinstance(resume, dict) else None
            if (not isinstance(resume, dict)
                    or set(resume) != RESUME_CAPTURE_KEYS
                    or not valid_resealed_from(resume.get("resealed_from"))
                    or any(_HEX64.fullmatch(str(resume.get(name, ""))) is None
                           for name in ("dataset_sha256",
                                        "capture_content_digest",
                                        "dataset_manifest_file_sha256"))
                    or not (origin is None or (
                        isinstance(origin, dict)
                        and set(origin) == {"job_id_full", "attempt_id",
                                            "job_file_sha256"}
                        and _HEX64.fullmatch(str(origin["job_id_full"])) is not None
                        and _HEX64.fullmatch(str(origin["job_file_sha256"])) is not None
                        and re.fullmatch(r"[0-9a-f]{24}",
                                         str(origin["attempt_id"])) is not None))):
                raise JobContractError(
                    "root capture resume_capture identity is incomplete")
        dataset_license = capture.get("dataset_license")
        weights_license = capture.get("weights_license")
        if dataset_license == "mit":
            if weights_license is not None:
                raise JobContractError(
                    "MIT root capture cannot carry source-license identity")
        elif dataset_license == "other":
            if (not isinstance(weights_license, dict)
                    or set(weights_license) != {
                        "source_path", "dataset_path", "bytes", "sha256"}
                    or weights_license.get("source_path") != "LICENSE"
                    or weights_license.get("dataset_path") != "LICENSE"
                    or isinstance(weights_license.get("bytes"), bool)
                    or not isinstance(weights_license.get("bytes"), int)
                    or not 0 < weights_license["bytes"] <= 1024 * 1024
                    or _HEX64.fullmatch(str(
                        weights_license.get("sha256", ""))) is None):
                raise JobContractError(
                    "non-MIT root capture license identity differs")
        else:
            raise JobContractError(
                "root capture dataset_license must be mit or other")
        if target.get("weights_license") != weights_license:
            raise JobContractError(
                "root target and capture source-license identities differ")
        if (document.get("recipe") != "runpod-controller-loss-drill"
                and weights_license is not None
                and download_by_path.get(weights_license["source_path"])
                != weights_license["bytes"]):
            raise JobContractError(
                "root source-license identity is absent from download manifest")
        dependencies = document["produced_by"].get("dependencies")
        if (not isinstance(dependencies, dict)
                or dependencies.get("profile") != profile.get("profile_id")):
            raise JobContractError(
                "root producing-code profile dependency differs from job profile")
        attempt_value = document.get("execution_attempt")
        attempt_kind = (
            attempt_value.get("kind")
            if isinstance(attempt_value, dict) else None)
        if attempt_kind == "runpod-ssh" and (
                dependencies.get("provider") != "runpod"
                or dependencies.get("lane") != document.get("lane")):
            raise JobContractError(
                "root RunPod producing-code provider/lane dependencies differ")
        if (attempt_kind in ("local-container", "local")
                and "provider" in dependencies
                and dependencies.get("provider") != attempt_kind):
            raise JobContractError(
                "root producing-code provider dependency differs from execution")
        if ("lane" in dependencies
                and dependencies.get("lane") != document.get("lane")):
            raise JobContractError(
                "root producing-code lane dependency differs from job lane")
        protocol = capture.get("root_protocol") or {}
        publication_requested = capture.get("publish_root_to") is not None
        if (set(protocol) != {
                "schedule", "fresh_processes", "run_count_per_process",
                "exact_self_comparison", "qualification_required",
                "canonical_publication_required", "publication_mode"}
                or protocol.get("schedule")
                != "two-fresh-process-qualification"
                or protocol.get("fresh_processes") != 2
                or protocol.get("run_count_per_process") != 1
                or protocol.get("exact_self_comparison") is not True
                or protocol.get("qualification_required") is not True
                or protocol.get("canonical_publication_required")
                is not publication_requested
                or protocol.get("publication_mode") != (
                    "canonical-public" if publication_requested
                    else "qualified-unpublished")):
            raise JobContractError("root two-process protocol is incomplete")
        if (set(profile) != {
                "profile_id", "lane", "source", "surface", "form", "engine",
                "compute_dtype", "device", "schedule"}
                or profile.get("profile_id") != "root-hf-transformers-bf16"
                or profile.get("lane") != "root"
                or profile.get("source") != "native"
                or profile.get("surface") != target.get("surface")
                or profile.get("form") != capture.get("form")
                or profile.get("engine") != "hf-transformers"
                or profile.get("compute_dtype") != "bfloat16"
                or profile.get("device") != "cuda"
                or profile.get("schedule")
                != "two-fresh-process-qualification"):
            raise JobContractError("root profile contract is incomplete")
    elif document.get("recipe") != "runpod-controller-loss-drill":
        reference = document.get("reference")
        reference_fields = (
            "reference_ref", "teacher_receipt_sha256",
            "teacher_backend_identity_sha256")
        reference_complete = (
            isinstance(reference, dict)
            and set(reference) == set(reference_fields)
            and isinstance(reference.get("reference_ref"), str)
            and bool(reference["reference_ref"])
            and all(
                _HEX64.fullmatch(str(reference.get(name, ""))) is not None
                for name in reference_fields[1:])
            and all(panel.get(name) == reference[name]
                    for name in reference_fields)
            and _HEX64.fullmatch(str(
                panel.get("panel_receipt_sha256", ""))) is not None)
        if (not reference_complete
                or set(profile) != {
                    "profile_id", "lane", "source", "surface", "bits"}
                or not profile.get("profile_id")
                or profile.get("lane") != document["lane"]
                or profile.get("source") not in ("tr3", "native")
                or profile.get("surface") != target.get("surface")
                or profile.get("bits") != target.get("bits")):
            raise JobContractError(
                "quant reference/profile contract is incomplete")
        if profile.get("profile_id") == "tr3-6bpw":
            official_revision = document.get("official_bf16_revision")
            official_identity = target.get("official_bf16_identity")
            if (re.fullmatch(r"[0-9a-f]{40}",
                             str(official_revision or "")) is None
                    or not isinstance(official_identity, dict)
                    or any(
                        _HEX64.fullmatch(str(
                            official_identity.get(name, ""))) is None
                        for name in ("config_sha256", "index_sha256"))
                    or any(
                        isinstance(official_identity.get(name), bool)
                        or not isinstance(official_identity.get(name), int)
                        or official_identity[name] <= 0
                        for name in ("config_bytes", "index_bytes"))):
                raise JobContractError(
                    "TR3 bridge lacks exact official BF16 metadata identity")
        scoring = document.get("scoring")
        expected_scoring = {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda", "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        }
        if scoring != expected_scoring or panel.get("roles") != "final":
            raise JobContractError(
                "quant scoring/panel role contract is incomplete")
    attempt = document.get("execution_attempt")
    if not isinstance(attempt, dict):
        raise JobContractError("execution_attempt must be an object")
    if attempt.get("kind") in ("local-container", "local"):
        kind = attempt["kind"]
        if set(attempt) != {"number", "kind", "attempt_id"}:
            raise JobContractError(
                "%s execution_attempt is noncanonical" % kind)
        if (attempt["number"] != 1
                or re.fullmatch(r"[0-9a-f]{24}",
                                str(attempt["attempt_id"])) is None):
            raise JobContractError(
                "%s attempt requires number 1 and 24 lowercase hex" % kind)
        return
    if attempt.get("kind") != "runpod-ssh":
        raise JobContractError("execution_attempt.kind is unsupported")
    fields = {
        "attempt_id", "cost_quote", "engine_root",
        "execution_contract_sha256", "kind", "lease_path",
        "planned_at", "pre_create_safety", "prepared_create", "remote_root",
        "provider_terminate_after", "storage_layout", "workload_deadline_utc",
    }
    if set(attempt) != fields:
        raise JobContractError("runpod-ssh execution_attempt fields differ")
    attempt_id = attempt["attempt_id"]
    if (attempt_id is not None
            and re.fullmatch(r"[0-9a-f]{24}", str(attempt_id)) is None):
        raise JobContractError(
            "runpod attempt_id must be null or 24 lowercase hex")
    if attempt["lease_path"] is not None and not isinstance(
            attempt["lease_path"], str):
        raise JobContractError("runpod lease_path must be null or string")
    remote_root = attempt["remote_root"]
    if remote_root is not None and not isinstance(remote_root, str):
        raise JobContractError("runpod remote_root must be null or string")
    engine_root = attempt["engine_root"]
    if engine_root is not None and not isinstance(engine_root, str):
        raise JobContractError("runpod engine_root must be null or string")
    if (attempt["cost_quote"] is not None
            and not isinstance(attempt["cost_quote"], dict)):
        raise JobContractError("runpod cost_quote must be null or object")
    if (attempt["pre_create_safety"] is not None
            and not isinstance(attempt["pre_create_safety"], dict)):
        raise JobContractError(
            "runpod pre_create_safety must be null or object")
    prepared = attempt["prepared_create"]
    if prepared is not None:
        if (not isinstance(prepared, dict)
                or set(prepared) != {
                    "schema", "request_identity", "graphql_body_sha256",
                    "graphql_body_bytes", "graphql_body_base64"}
                or prepared["schema"]
                != "fidelity-suite/runpod-prepared-create.v1"
                or not isinstance(prepared["request_identity"], dict)
                or _HEX64.fullmatch(
                    str(prepared["graphql_body_sha256"])) is None
                or isinstance(prepared["graphql_body_bytes"], bool)
                or not isinstance(prepared["graphql_body_bytes"], int)
                or prepared["graphql_body_bytes"] <= 0):
            raise JobContractError(
                "runpod prepared_create is not a canonical prepared mutation")
        try:
            graphql_body = base64.b64decode(
                prepared["graphql_body_base64"].encode("ascii"),
                validate=True)
        except (AttributeError, UnicodeError, ValueError, binascii.Error) as exc:
            raise JobContractError(
                "runpod prepared_create body is invalid base64") from exc
        if (len(graphql_body) != prepared["graphql_body_bytes"]
                or hashlib.sha256(graphql_body).hexdigest()
                != prepared["graphql_body_sha256"]):
            raise JobContractError(
                "runpod prepared_create body identity differs")
    seal = attempt["execution_contract_sha256"]
    if seal is not None and (
            not isinstance(seal, str) or _HEX64.fullmatch(seal) is None):
        raise JobContractError(
            "execution_contract_sha256 must be null or 64 lowercase hex")
    for name in ("planned_at", "provider_terminate_after",
                 "workload_deadline_utc"):
        value = attempt[name]
        if (value is not None
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                                 str(value)) is None):
            raise JobContractError("%s must be null or exact UTC" % name)


def parse_job_bytes(data: bytes) -> dict:
    """Parse UTF-8 job JSON while rejecting every duplicate object key."""
    if not isinstance(data, bytes):
        raise JobContractError("job JSON input must be bytes")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise JobContractError(
                    "job JSON contains duplicate key %r" % key)
            result[key] = value
        return result

    def reject_constant(value):
        raise JobContractError(
            "job JSON contains non-finite constant %s" % value)

    try:
        document = json.loads(
            data.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JobContractError("job JSON is not canonical UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise JobContractError("job JSON root must be an object")
    return document


def job_identity_projection(document: dict) -> dict:
    """Return the exact content whose SHA-256 identifies a job.

    Exclusions are deliberately top-level and closed.  In particular, nested
    provider, timing, panel, bundle, scope and provenance data cannot disappear
    merely because this module did not know their field names.
    """
    projected = {key: value for key, value in document.items()
                 if key not in _EXCLUDED_TOP_LEVEL}
    # JSON round-trip both validates and returns a detached plain-data value.
    return json.loads(_canonical_bytes(projected).decode("utf-8"))


def finalize_job(document: dict) -> dict:
    """Copy, hash and finalize a complete job document."""
    if not isinstance(document, dict):
        raise JobContractError("job document must be an object")
    finalized = copy.deepcopy(document)
    validate_job(finalized)
    projection = job_identity_projection(finalized)
    full = hashlib.sha256(_canonical_bytes(projection)).hexdigest()
    finalized["job_id_full"] = full
    finalized["job_id"] = full[:DISPLAY_HEX]
    # Validate the result too: custom Mapping/list subclasses must not leak.
    return json.loads(_canonical_bytes(finalized).decode("utf-8"))


def execution_contract_sha256(document: dict) -> str:
    """Hash the complete job with only its execution seal slot blanked."""
    candidate = copy.deepcopy(document)
    attempt = candidate.get("execution_attempt")
    if not isinstance(attempt, dict):
        raise JobContractError("execution_attempt must be an object")
    attempt["execution_contract_sha256"] = None
    return hashlib.sha256(_canonical_bytes(candidate)).hexdigest()


def seal_execution_job(document: dict) -> dict:
    """Return an execution document whose full-file contract is sealed."""
    sealed = copy.deepcopy(document)
    sealed["execution_attempt"]["execution_contract_sha256"] = None
    validate_job(sealed)
    sealed["execution_attempt"]["execution_contract_sha256"] = (
        execution_contract_sha256(sealed))
    validate_execution_job(sealed)
    return sealed


def validate_execution_job(document: dict) -> None:
    """Validate a finalized, runnable job rather than a null planned attempt."""
    verify_job(document)
    attempt = document["execution_attempt"]
    is_drill = document.get("recipe") == "runpod-controller-loss-drill"
    if attempt["kind"] != "runpod-ssh":
        return
    stored_execution_seal = attempt["execution_contract_sha256"]
    if (stored_execution_seal is None
            or stored_execution_seal != execution_contract_sha256(document)):
        raise JobContractError(
            "execution contract seal does not match the complete job")
    for name in ("attempt_id", "cost_quote", "engine_root", "lease_path",
                 "prepared_create", "remote_root", "planned_at",
                 "provider_terminate_after", "workload_deadline_utc"):
        if attempt[name] is None:
            raise JobContractError("executed RunPod job lacks %s" % name)
    lease = attempt["lease_path"]
    pure = PurePosixPath(lease)
    if (pure.is_absolute() or pure.as_posix() != lease
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in lease):
        raise JobContractError("lease_path must be a canonical relative record id")
    if is_drill:
        if (attempt["remote_root"] != "/workspace/fidelity-drill"
                or attempt["engine_root"] != "/workspace/fidelity-drill"):
            raise JobContractError(
                "drill remote paths differ from its fresh-pod payload root")
    else:
        # The run root's base is declared by the job: the container disk
        # (/root, the host's local NVMe) or the pod volume (/workspace).
        layout = attempt.get("storage_layout")
        run_base = RUNPOD_RUN_BASES.get(layout)
        if run_base is None:
            raise JobContractError(
                "storage_layout must be one of %s"
                % ", ".join(sorted(RUNPOD_RUN_BASES)))
        expected_remote_root = "%s/fidelity/%s/%s" % (
            run_base, document["job_id_full"], attempt["attempt_id"])
        if attempt["remote_root"] != expected_remote_root:
            raise JobContractError(
                "remote_root is not the exact fresh job/attempt directory")
        expected_engine_root = "%s/fidelity-engine/%s/%s" % (
            run_base, document["job_id_full"], attempt["attempt_id"])
        if attempt["engine_root"] != expected_engine_root:
            raise JobContractError(
                "engine_root is not the exact fresh job/attempt directory")
    try:
        from .campaign import CostQuote
        quote = CostQuote.from_dict(attempt["cost_quote"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JobContractError(
            "runpod execution cost_quote is invalid") from exc
    target = document["target"]
    profile = document["profile"]
    environment = document.get("environment")
    safety = attempt["pre_create_safety"]
    if not isinstance(environment, dict) or not isinstance(safety, dict):
        raise JobContractError(
            "executed RunPod job lacks environment/safety binding")
    if quote.target != "%s@%s" % (
            target.get("repo_id"), target.get("revision")):
        raise JobContractError("quote target differs from exact job target")
    if quote.profile != profile.get("profile_id"):
        raise JobContractError("quote profile differs from exact job profile")
    if quote.timing_evidence != hashlib.sha256(
            _canonical_bytes(document["timing"])).hexdigest():
        raise JobContractError("quote timing evidence differs from job timing")
    if quote.hard_cap_usd != Decimal(str(environment.get("hard_cap_usd"))):
        raise JobContractError("quote hard cap differs from job environment")
    if quote.live_compute_usd_per_hour != (
            Decimal(str(environment.get("price_per_gpu_hour")))
            * Decimal(str(environment.get("gpus")))):
        raise JobContractError("quote rate differs from exact offer")
    server_time = safety.get("server_time")
    if is_drill:
        expected_safety_keys = {
            "checked_at", "reaper_health_sha256", "provider_account_id",
            "provider_gpu_id", "image", "bundle_contract_sha256",
            "control_manifest_sha256", "server_time",
            "producer_checkout_sha256",
        }
        if (set(safety) != expected_safety_keys
                or safety.get("provider_account_id")
                    != environment.get("provider_account_id")
                or safety.get("provider_gpu_id")
                    != environment.get("provider_gpu_id")
                or safety.get("image") != environment.get("image")
                or safety.get("bundle_contract_sha256")
                    != document.get("bundle_contract_sha256")
                or safety.get("control_manifest_sha256")
                    != (document.get("control_plane") or {}).get(
                        "manifest_sha256")
                or any(_HEX64.fullmatch(str(safety.get(name, ""))) is None
                       for name in (
                           "reaper_health_sha256",
                           "producer_checkout_sha256",
                           "bundle_contract_sha256",
                           "control_manifest_sha256"))):
            raise JobContractError(
                "drill pre-create safety binding differs from finalized job")
    else:
        expected_safety_keys = {
            "checked_at", "reaper_health_sha256",
            "safety_proof_file_sha256", "safety_proof_sha256",
            "provider_account_id", "provider_gpu_id", "image",
            "bundle_contract_sha256", "control_manifest_sha256",
            "server_time",
        }
        proof_fields = (
            safety.get("safety_proof_file_sha256"),
            safety.get("safety_proof_sha256"))
        # Both proof digests are present (strict campaign mode with a
        # sealed drill proof) or both are absent (default mode); one without
        # the other is a malformed binding.
        proof_binding_ok = (
            all(value is None for value in proof_fields)
            or all(isinstance(value, str) and _HEX64.fullmatch(value)
                   for value in proof_fields))
        if (set(safety) != expected_safety_keys
                or safety.get("provider_account_id")
                    != environment.get("provider_account_id")
                or safety.get("provider_gpu_id")
                    != environment.get("provider_gpu_id")
                or safety.get("image") != environment.get("image")
                or safety.get("bundle_contract_sha256")
                    != document.get("bundle_contract_sha256")
                or safety.get("control_manifest_sha256")
                    != (document.get("control_plane") or {}).get(
                        "manifest_sha256")
                or not proof_binding_ok
                or any(_HEX64.fullmatch(str(safety.get(name, ""))) is None
                       for name in (
                           "reaper_health_sha256", "bundle_contract_sha256",
                           "control_manifest_sha256"))):
            raise JobContractError(
                "pre-create safety binding differs from finalized job")
    try:
        server_delta = abs(float(server_time.get(
            "local_minus_server_seconds", float("inf"))))
        evidence_age = float(server_time.get(
            "evidence_age_seconds", float("inf")))
    except (AttributeError, TypeError, ValueError) as exc:
        raise JobContractError(
            "provider server-time evidence is malformed") from exc
    if (not isinstance(server_time, dict)
            or server_time.get("schema")
            != "fidelity-suite/runpod-server-time.v1"
            or server_delta > 30 or evidence_age > 30
            or server_time.get("max_clock_delta_seconds") != 30
            or server_time.get("max_evidence_age_seconds") != 30):
        raise JobContractError(
            "provider server-time evidence is invalid or stale")
    if (not isinstance(environment.get("image"), str)
            or not environment["image"]
            or environment.get("image_reference_mutable")
            is not ("@sha256:" not in environment["image"])):
        raise JobContractError("RunPod image identity binding is invalid")
    convergence = document.get("post_create_convergence")
    if (not isinstance(convergence, dict)
            or set(convergence) != {
                "schema", "timeout_seconds", "poll_seconds"}
            or convergence.get("schema")
            != "fidelity-suite/runpod-post-create-convergence.v1"
            or convergence.get("timeout_seconds") != 180
            or convergence.get("poll_seconds") != 10):
        raise JobContractError(
            "RunPod post-create convergence contract is noncanonical")
    from datetime import datetime
    planned = datetime.strptime(
        attempt["planned_at"], "%Y-%m-%dT%H:%M:%SZ")
    deadline = datetime.strptime(
        attempt["workload_deadline_utc"], "%Y-%m-%dT%H:%M:%SZ")
    termination = datetime.strptime(
        attempt["provider_terminate_after"], "%Y-%m-%dT%H:%M:%SZ")
    if attempt["planned_at"] != quote.quoted_at:
        raise JobContractError(
            "planned_at differs from the exact quote timestamp")
    if Decimal(str((deadline - planned).total_seconds())) != (
            quote.workload_deadline_seconds):
        raise JobContractError(
            "workload deadline differs from exact quote duration")
    if Decimal(str((termination - planned).total_seconds())) != (
            quote.provider_termination_deadline_seconds):
        raise JobContractError(
            "provider termination differs from exact quote duration")
    if deadline >= termination:
        raise JobContractError(
            "workload deadline must precede provider termination")


def verify_job(document: dict) -> str:
    """Verify both stored identities and return the complete 64-hex hash."""
    if not isinstance(document, dict):
        raise JobContractError("job document must be an object")
    validate_job(document)
    stored_full = document.get("job_id_full")
    stored_display = document.get("job_id")
    if not isinstance(stored_full, str) or _HEX64.fullmatch(stored_full) is None:
        raise JobContractError("job_id_full must be 64 lowercase hexadecimal characters")
    if stored_display != stored_full[:DISPLAY_HEX]:
        raise JobContractError(
            "job_id is not the display prefix of job_id_full")
    expected = hashlib.sha256(
        _canonical_bytes(job_identity_projection(document))).hexdigest()
    if stored_full != expected:
        raise JobContractError(
            "job identity does not match canonical job content")
    return expected


ROOT_QUALIFICATION_CONTRACT_KEYS = frozenset({
    "dataset_id", "dataset_name", "author", "dataset_repository",
    "publish_root_to", "dataset_license", "weights_license",
    "weights_repository", "weights_revision", "lane",
    "form", "schedule", "device", "dtype", "engine", "execution_kind",
    "container_image_reference", "container_image_digest", "panel_id",
    "panel_suite_token_hash_sha256", "panel_receipt_sha256",
    "panel_receipt_file_sha256", "panel_receipt_file_bytes",
    "panel_receipt_seal_mode", "panel_binding_file_sha256",
    "panel_binding_path", "tokenizer_identity_sha256",
    "unexpected_tensor_allowlist", "target", "profile",
    "panel_resolved_binding", "resume_capture", "candidate",
})

#: capture.candidate: the two-fresh-process protocol applied to a QUANTIZED
#: target. Null for a root. When present the pod captures with --role quant
#: under the authored scope, the qualification binds a role=quant dataset,
#: and one more stage (compare_reference) scores the qualified capture
#: against the published root the block names -- the same-stack KLD the
#: registry ingests. The reference identity is exact (repo, revision, seal,
#: content digest) so the pod refuses any other dataset under that name.
CANDIDATE_KEYS = frozenset({
    "scope", "codec", "declared_bits", "weights_decode", "reference"})
CANDIDATE_SCOPE_KEYS = frozenset({"path", "sha256", "scope_digest"})
CANDIDATE_REFERENCE_KEYS = frozenset({
    "repository", "revision", "dataset_sha256", "capture_content_digest",
    "dataset_id", "panel_id", "suite_token_hash_sha256"})
CANDIDATE_DECODE_KEYS = frozenset({"method", "quantization_config"})


def valid_candidate(value: Any) -> bool:
    """None, or the exact candidate block (see CANDIDATE_KEYS)."""
    if value is None:
        return True
    if not isinstance(value, dict) or set(value) != CANDIDATE_KEYS:
        return False
    scope, reference, decode = (
        value.get("scope"), value.get("reference"), value.get("weights_decode"))
    return (
        isinstance(scope, dict) and set(scope) == CANDIDATE_SCOPE_KEYS
        and isinstance(scope.get("path"), str) and bool(scope["path"])
        and _HEX64.fullmatch(str(scope.get("sha256", ""))) is not None
        # The registry's scope digest is a canonical class=treatment:format@bits
        # string (registry/tools scope_digest), not a hash.
        and isinstance(scope.get("scope_digest"), str) and bool(scope["scope_digest"])
        and "=" in scope["scope_digest"]
        and isinstance(value.get("codec"), str) and bool(value["codec"])
        and isinstance(value.get("declared_bits"), (int, float))
        and not isinstance(value.get("declared_bits"), bool)
        and 0 < float(value["declared_bits"]) <= 16
        and isinstance(decode, dict) and set(decode) == CANDIDATE_DECODE_KEYS
        and isinstance(decode.get("method"), str) and bool(decode["method"])
        and isinstance(decode.get("quantization_config"), dict)
        and isinstance(reference, dict)
        and set(reference) == CANDIDATE_REFERENCE_KEYS
        and isinstance(reference.get("repository"), str)
        and re.fullmatch(r"[^\s/]+/[^\s/]+", reference["repository"]) is not None
        and re.fullmatch(r"[0-9a-f]{40}", str(reference.get("revision", ""))) is not None
        and all(_HEX64.fullmatch(str(reference.get(name, ""))) is not None
                for name in ("dataset_sha256", "capture_content_digest",
                             "suite_token_hash_sha256"))
        and isinstance(reference.get("dataset_id"), str) and bool(reference["dataset_id"])
        and isinstance(reference.get("panel_id"), str) and bool(reference["panel_id"]))


def validate_root_qualification_contract(contract: dict) -> None:
    """Validate the closed public projection without requiring its source job."""
    if not isinstance(contract, dict) \
            or set(contract) not in (
                ROOT_QUALIFICATION_CONTRACT_KEYS,
                # Receipts sealed before 2026-09-04 carry no candidate key;
                # a root's candidate is null either way.
                ROOT_QUALIFICATION_CONTRACT_KEYS - {"candidate"}):
        raise JobContractError(
            "root qualification job_contract fields differ")
    if not valid_candidate(contract.get("candidate")):
        raise JobContractError(
            "root qualification job_contract candidate block is invalid")
    dataset_license = contract.get("dataset_license")
    weights_license = contract.get("weights_license")
    if dataset_license == "mit":
        if weights_license is not None:
            raise JobContractError(
                "MIT root qualification cannot carry source-license identity")
    elif dataset_license == "other":
        if (not isinstance(weights_license, dict)
                or set(weights_license) != {
                    "source_path", "dataset_path", "bytes", "sha256"}
                or weights_license.get("source_path") != "LICENSE"
                or weights_license.get("dataset_path") != "LICENSE"
                or isinstance(weights_license.get("bytes"), bool)
                or not isinstance(weights_license.get("bytes"), int)
                or not 0 < weights_license["bytes"] <= 1024 * 1024
                or _HEX64.fullmatch(str(
                    weights_license.get("sha256", ""))) is None):
            raise JobContractError(
                "non-MIT root qualification license identity differs")
    else:
        raise JobContractError(
            "root qualification dataset_license must be mit or other")
    target = contract.get("target")
    profile = contract.get("profile")
    binding = contract.get("panel_resolved_binding")
    panel_identity = (
        binding.get("panel") if isinstance(binding, dict) else None)
    receipt = (
        binding.get("receipt") if isinstance(binding, dict) else None)
    tokenizer = (
        binding.get("tokenizer") if isinstance(binding, dict) else None)
    allowlist = contract.get("unexpected_tensor_allowlist")
    candidate = contract.get("candidate")
    # A root is the unquantized release; a candidate is the one quantized
    # form the streaming loader decodes, and its target identity must say
    # exactly that (surface, codec and bits agree with the candidate block).
    if candidate is None:
        expected_target = ("native-bf16", "bf16", 16)
    else:
        # The surface follows the decode the candidate block declares, rather
        # than being assumed FP8: the streaming loader decodes block-scaled
        # FP8, stock-exllamav3 trellis, modelopt NVFP4 and llama.cpp GGUF
        # builds; a trellis artifact's sniffed surface is `exl3hf`, a modelopt
        # NVFP4 one's is `nvfp4`, a GGUF's is `gguf`. Anything else is a decode
        # this contract has never seen and refuses by name rather than
        # defaulting.
        decode = candidate.get("weights_decode") or {}
        surfaces = {
            "fp8-block-dequant-to-bf16": "fp8-block",
            "exl3-trellis-decode-to-bf16": "exl3hf",
            "exl3-trellis-tp-compose-to-bf16": "exl3hf",
            "nvfp4-modelopt-dequant-to-bf16": "nvfp4",
            "gguf-dequant-to-bf16": "gguf",
        }
        surface = surfaces.get(str(decode.get("method")))
        if surface is None:
            raise JobContractError(
                "root qualification candidate declares decode method %r, which "
                "maps to no known target surface (known: %s)"
                % (decode.get("method"), ", ".join(sorted(surfaces))))
        expected_target = (surface, candidate["codec"], candidate["declared_bits"])
    # A GGUF repo is a shelf of builds; the target names the ONE build directory
    # in `path` (the same string the decode contract's `build` carries). Every
    # other surface is a whole repository and `path` must be null.
    gguf_build = (expected_target[0] == "gguf")
    path_ok = (
        (isinstance(target.get("path"), str) and target["path"]
         and "/" not in target["path"].strip("/") and target["path"] not in (".", "..")
         and target["path"] == (((candidate or {}).get("weights_decode") or {})
                                .get("quantization_config") or {}).get("build"))
        if gguf_build else target.get("path") is None) if isinstance(target, dict) else False
    if (not isinstance(target, dict)
            or set(target) != {
                "repo_id", "revision", "surface", "codec", "bits", "path"}
            or target.get("repo_id") != contract.get("weights_repository")
            or target.get("revision") != contract.get("weights_revision")
            or not isinstance(target.get("repo_id"), str)
            or not target["repo_id"]
            or _HEX40.fullmatch(str(target.get("revision", ""))) is None
            or target.get("surface") != expected_target[0]
            or target.get("codec") != expected_target[1]
            or target.get("bits") != expected_target[2]
            or not path_ok):
        raise JobContractError(
            "root qualification target contract differs")
    for name in (
            "dataset_id", "dataset_name", "author", "dataset_repository",
            "weights_repository", "weights_revision", "lane", "form",
            "schedule", "device", "dtype", "engine", "execution_kind"):
        if not isinstance(contract.get(name), str) or not contract[name]:
            raise JobContractError(
                "root qualification %s is incomplete" % name)
    if (not isinstance(profile, dict)
            or set(profile) != {
                "profile_id", "lane", "source", "surface", "form", "engine",
                "compute_dtype", "device", "schedule"}
            or profile.get("profile_id") != "root-hf-transformers-bf16"
            or profile.get("lane") != "root"
            or profile.get("source") != "native"
            or profile.get("surface") != target.get("surface")
            or profile.get("form") != contract.get("form")
            or profile.get("engine") != contract.get("engine")
            or profile.get("compute_dtype") != contract.get("dtype")
            or profile.get("device") != contract.get("device")
            or profile.get("schedule")
                != "two-fresh-process-qualification"):
        raise JobContractError(
            "root qualification profile contract differs")
    if (not isinstance(panel_identity, dict)
            or not isinstance(receipt, dict)
            or not isinstance(tokenizer, dict)
            or panel_identity.get("id") != contract.get("panel_id")
            or panel_identity.get("suite_token_hash_sha256")
                != contract.get("panel_suite_token_hash_sha256")
            or receipt.get("declared_receipt_sha256")
                != contract.get("panel_receipt_sha256")
            or receipt.get("receipt_file_sha256")
                != contract.get("panel_receipt_file_sha256")
            or receipt.get("bytes")
                != contract.get("panel_receipt_file_bytes")
            or receipt.get("receipt_seal_mode")
                != contract.get("panel_receipt_seal_mode")
            or receipt.get("receipt_seal_mode")
                not in ("self-blank", "legacy-field-absent")
            or tokenizer.get("identity_sha256")
                != contract.get("tokenizer_identity_sha256")
            or any(_HEX64.fullmatch(str(contract.get(name, ""))) is None
                   for name in (
                       "panel_suite_token_hash_sha256",
                       "panel_receipt_sha256",
                       "panel_receipt_file_sha256",
                       "panel_binding_file_sha256",
                       "tokenizer_identity_sha256"))
            or isinstance(contract.get("panel_receipt_file_bytes"), bool)
            or not isinstance(contract.get("panel_receipt_file_bytes"), int)
            or contract["panel_receipt_file_bytes"] <= 0):
        raise JobContractError(
            "root qualification panel contract differs")
    canonical_relative_path(
        contract.get("panel_binding_path"),
        "root qualification panel binding_path")
    if (not isinstance(allowlist, dict)
            or set(allowlist) != {
                "path", "artifact_sha256",
                "canonical_sorted_names_sha256"}
            or any(_HEX64.fullmatch(str(allowlist.get(name, ""))) is None
                   for name in (
                       "artifact_sha256",
                       "canonical_sorted_names_sha256"))):
        raise JobContractError(
            "root qualification allowlist contract differs")
    canonical_relative_path(
        allowlist.get("path"), "root qualification allowlist path")


def _root_execution_identity(document: dict):
    environment = document.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    produced_by = document.get("produced_by")
    dependencies = (
        produced_by.get("dependencies")
        if isinstance(produced_by, dict) else None)
    provider = (
        dependencies.get("provider")
        if isinstance(dependencies, dict) else None)
    if provider == "runpod":
        execution_kind = "runpod-ssh"
        image_reference = environment.get("image")
        image_digest = (
            image_reference.rsplit("@", 1)[1]
            if isinstance(image_reference, str) and "@" in image_reference
            else None)
    elif provider == "local" or document.get("recipe") == "local":
        # A job written by `fidelity-dataset qualify-root --local` from two
        # captures' own binding evidence: no pod, no image, no attestation.
        execution_kind = "local"
        image_reference = None
        image_digest = None
    elif provider == "local-container" \
            or document.get("recipe") == "local-container":
        execution_kind = "local-container"
        image_reference = environment.get("container_image")
        image_digest = environment.get("container_digest")
    else:
        raise JobContractError(
            "root execution identity is not derivable from job identity")
    execution = document.get("execution_attempt")
    observed_kind = (
        execution.get("kind") if isinstance(execution, dict) else None)
    if observed_kind is not None and observed_kind != execution_kind:
        raise JobContractError(
            "root execution attempt differs from identity-bound provider")
    return execution_kind, image_reference, image_digest


def _root_qualification_contract(document: dict) -> dict:
    """Build the closed public projection after its job identity is verified."""
    if document.get("role") != "root":
        raise JobContractError(
            "root qualification contract requires role=root")
    target = document.get("target")
    profile = document.get("profile")
    capture = document.get("capture")
    panel = document.get("panel")
    binding = (
        panel.get("resolved_binding") if isinstance(panel, dict) else None)
    if (not isinstance(target, dict) or not isinstance(profile, dict)
            or not isinstance(capture, dict) or not isinstance(binding, dict)
            or not isinstance(panel.get("binding_file_sha256"), str)
            or _HEX64.fullmatch(panel["binding_file_sha256"]) is None):
        raise JobContractError(
            "root qualification requires an exact resolved panel binding")
    binding_path = canonical_relative_path(
        panel.get("binding_path"), "root panel binding_path").as_posix()
    receipt = binding.get("receipt")
    tokenizer = binding.get("tokenizer")
    panel_identity = binding.get("panel")
    if (not isinstance(receipt, dict) or not isinstance(tokenizer, dict)
            or not isinstance(panel_identity, dict)):
        raise JobContractError(
            "root qualification panel binding is incomplete")
    if target.get("weights_license") != capture.get("weights_license"):
        raise JobContractError(
            "root target and capture source-license identities differ")
    execution_kind, image_reference, image_digest = (
        _root_execution_identity(document))
    target_identity = {
        key: target.get(key)
        for key in ("repo_id", "revision", "surface", "codec", "bits", "path")
    }
    contract = {
        "dataset_id": capture.get("dataset_id"),
        "dataset_name": capture.get("dataset_name"),
        "author": capture.get("author"),
        # Null for a two-fresh-process root; the imported cold run 1's exact
        # identity and origin for a resumed one (validate_job checks it).
        "resume_capture": capture.get("resume_capture"),
        # Null for a root; for a candidate, the authored scope, codec, bits,
        # the decode the loader must apply and the exact reference dataset.
        "candidate": capture.get("candidate"),
        "dataset_repository": capture.get("dataset_repository"),
        "publish_root_to": capture.get("publish_root_to"),
        "dataset_license": capture.get("dataset_license"),
        "weights_license": capture.get("weights_license"),
        "weights_repository": target.get("repo_id"),
        "weights_revision": target.get("revision"),
        "lane": document.get("lane"),
        "form": capture.get("form"),
        "schedule": capture.get("schedule"),
        "device": capture.get("device"),
        "dtype": capture.get("dtype"),
        "engine": capture.get("engine"),
        "execution_kind": execution_kind,
        "container_image_reference": image_reference,
        "container_image_digest": image_digest,
        "panel_id": panel_identity.get("id"),
        "panel_suite_token_hash_sha256":
            panel_identity.get("suite_token_hash_sha256"),
        "panel_receipt_sha256":
            receipt.get("declared_receipt_sha256"),
        "panel_receipt_file_sha256":
            receipt.get("receipt_file_sha256"),
        "panel_receipt_file_bytes": receipt.get("bytes"),
        "panel_receipt_seal_mode": receipt.get("receipt_seal_mode"),
        "panel_binding_file_sha256": panel.get("binding_file_sha256"),
        "panel_binding_path": binding_path,
        "tokenizer_identity_sha256": tokenizer.get("identity_sha256"),
        "unexpected_tensor_allowlist":
            capture.get("unexpected_tensor_allowlist"),
        "target": target_identity,
        "profile": profile,
        "panel_resolved_binding": binding,
    }
    contract = json.loads(_canonical_bytes(contract).decode("utf-8"))
    validate_root_qualification_contract(contract)
    return contract

def root_qualification_contract(document: dict) -> dict:
    """Return the closed public projection of one verified root job."""
    verify_job(document)
    return _root_qualification_contract(document)


__all__ = [
    "JobContractError", "ROOT_QUALIFICATION_CONTRACT_KEYS",
    "canonical_relative_path", "finalize_bundle_manifest", "finalize_job",
    "job_identity_projection", "parse_job_bytes",
    "root_qualification_contract", "validate_execution_job", "validate_job",
    "validate_root_qualification_contract", "verify_bundle_manifest",
    "verify_job",
]
