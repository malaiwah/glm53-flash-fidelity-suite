#!/usr/bin/env python3
"""Focused offline checks for exact width-two Fruit panel admission."""
from __future__ import annotations

import copy
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from fidelity import common, dsformat, jobcontract, runpodsafety as safety  # noqa: E402


def check(name, condition):
    if not condition:
        raise AssertionError(name)


def fruit_binding():
    rows = [
        {"path": "panel.json", "bytes": 17542,
         "sha256": safety.FRUIT_PANEL_FILE_SHA256},
        {"path": "panel.receipt.json", "bytes": 8171,
         "sha256": safety.FRUIT_RECEIPT_FILE_SHA256},
    ]
    rows.extend(
        {"path": "arrays/final-%04d.%s.npy" % (index, kind),
         "bytes": 1, "sha256": "%064x" % (index * 2 + offset + 1)}
        for index in range(safety.FRUIT_PANEL_CONTEXTS)
        for offset, kind in enumerate(("mask", "tokens"))
    )
    panel = {
        "id": safety.FRUIT_PANEL_ID,
        "name": "GLM-5.2-SIQ-Fruit held-out fidelity panel v1 -- 16 windows x 2048",
        "role": "final", "contexts": 16, "context_length": 2048,
        "positions_per_context": 2047, "scored_positions_total": 32752,
        "suite_token_hash_sha256": safety.FRUIT_PANEL_SUITE_TOKEN_SHA256,
        "file": "panel.json", "bytes": 17542,
        "sha256": safety.FRUIT_PANEL_FILE_SHA256,
    }
    receipt = {
        "file": "panel.receipt.json", "bytes": 8171,
        "declared_receipt_sha256": safety.FRUIT_RECEIPT_DECLARED_SHA256,
        "receipt_seal_mode": "self-blank",
        "receipt_file_sha256": safety.FRUIT_RECEIPT_FILE_SHA256,
    }
    tokenizer = {
        "id": safety.FRUIT_REPO, "repository": safety.FRUIT_REPO,
        "revision": safety.FRUIT_REVISION, "vocab_size": 154820,
        "maximum_token_id_exclusive": 154820,
        "identity_sha256": safety.FRUIT_TOKENIZER_IDENTITY_SHA256,
        "files": [
            {"name": "tokenizer.json", "bytes": 1,
             "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"},
            {"name": "tokenizer_config.json", "bytes": 1,
             "sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        ],
        "files_verified": True, "receipt": None,
    }
    content = {
        "manifest": rows,
        "manifest_sha256": hashlib.sha256(
            safety.canonical_bytes(rows)).hexdigest(),
        "archive": {
            "format": "ustar", "compression": "none",
            "algorithm": "sha256(ustar: sorted regular files; mode=0644; uid=gid=mtime=0)",
            "bytes": 1, "sha256": "f" * 64,
        },
    }
    return {"schema": "malaiwah.resolved-panel.v1", "panel": panel,
            "receipt": receipt, "tokenizer": tokenizer, "content": content}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.body[:limit]

    def getcode(self):
        return 200


def public_refetch_fixture():
    repository = "example/root-proof"
    weights_repository = "example/root-weights"
    weights_revision = "b" * 40
    panel_id = "panel--selftest.public-root"
    suite_sha = "1" * 64
    receipt_doc = common.seal({
        "schema": "fidelity.selftest-panel-receipt.v1",
        "panel_id": panel_id,
    })
    panel_receipt_raw = safety.canonical_bytes(receipt_doc)
    receipt_sha = receipt_doc["receipt_sha256"]
    receipt_file_sha = hashlib.sha256(panel_receipt_raw).hexdigest()
    tokenizer_sha = "4" * 64
    binding_file_sha = "5" * 64
    allowlist_sha = "6" * 64
    allowlist_names_sha = "7" * 64
    binding = {
        "panel": {
            "id": panel_id,
            "suite_token_hash_sha256": suite_sha,
        },
        "receipt": {
            "file": "panel.receipt.json",
            "declared_receipt_sha256": receipt_sha,
            "receipt_file_sha256": receipt_file_sha,
            "bytes": len(panel_receipt_raw),
            "receipt_seal_mode": "self-blank",
        },
        "tokenizer": {
            "identity_sha256": tokenizer_sha,
        },
    }
    allowlist = {
        "path": "allowlist.json",
        "artifact_sha256": allowlist_sha,
        "canonical_sorted_names_sha256": allowlist_names_sha,
    }
    bundle = jobcontract.finalize_bundle_manifest([{
        "path": "bin/fidelity_dataset.py",
        "bytes": 1,
        "sha256": "8" * 64,
    }], "public-root-selftest")
    control = jobcontract.finalize_bundle_manifest([{
        "path": "bin/fidelity/jobcontract.py",
        "bytes": 1,
        "sha256": "9" * 64,
    }], "public-root-control-selftest")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    registry = {
        "path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "a" * 64,
    }
    bundle_contract_sha = hashlib.sha256(safety.canonical_bytes({
        "bundle": bundle, "registry": registry})).hexdigest()
    shards = [{"path": "model.safetensors", "bytes": 17}]
    download_manifest = [
        {"path": "config.json", "bytes": 1},
        shards[0],
        {"path": "model.safetensors.index.json", "bytes": 1},
    ]
    target = {
        "repo_id": weights_repository,
        "revision": weights_revision,
        "surface": "native-bf16",
        "codec": "bf16",
        "bits": 16,
        "path": None,
        "config_sha256": "b" * 64,
        "index_sha256": "c" * 64,
        "shards": shards,
        "shard_manifest_sha256": hashlib.sha256(
            safety.canonical_bytes(shards)).hexdigest(),
        "model_bytes": 17,
        "download_manifest": download_manifest,
        "download_bytes_total": 19,
        "download_manifest_sha256": hashlib.sha256(
            safety.canonical_bytes(download_manifest)).hexdigest(),
    }
    profile = {
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
    capture = {
        "dataset_id": "fidelity--selftest.root.hidden",
        "dataset_name": "selftest public root",
        "author": "selftest",
        "dataset_repository": repository,
        "publish_root_to": repository,
        "form": "hidden",
        "schedule": "layer-outer",
        "device": "cuda",
        "dtype": "bfloat16",
        "engine": "hf-transformers",
        "replay_device": "numpy",
        "replay_dtype": "float32",
        "vocab_chunk": 8192,
        "replay": {
            "device": "numpy", "dtype": "float32", "vocab_chunk": 8192,
        },
        "unexpected_tensor_allowlist": allowlist,
        "root_protocol": {
            "schedule": "two-fresh-process-qualification",
            "fresh_processes": 2,
            "run_count_per_process": 1,
            "exact_self_comparison": True,
            "qualification_required": True,
            "canonical_publication_required": True,
            "publication_mode": "canonical-public",
        },
    }
    job = jobcontract.finalize_job({
        "schema": "fidelity-suite/job.v2",
        "role": "root",
        "recipe": "local-container",
        "lane": "sealed-ep8",
        "cold_runs": 2,
        "target": target,
        "profile": profile,
        "capture": capture,
        "panel": {
            "binding_path": "panel-binding.json",
            "binding_file_sha256": binding_file_sha,
            "resolved_binding": binding,
        },
        "bundle": bundle,
        "control_plane": control,
        "bundle_registry": registry,
        "bundle_contract_sha256": bundle_contract_sha,
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 1,
            "min_memory_gb": 1,
            "expected_vram_bytes": 1,
        },
        "timing": {"kind": "public-root-selftest"},
        "scope": {"kind": "public-root-selftest"},
        "runtime": {},
        "environment": {
            "container_image": "example/image@sha256:" + "8" * 64,
            "container_digest": "sha256:" + "8" * 64,
        },
        "measurer": {"name": "selftest"},
        "produced_by": {
            "revision": "d" * 40,
            "dependencies": {
                "profile": "root-hf-transformers-bf16",
                "lane": "sealed-ep8",
                "provider": "local-container",
            },
        },
        "execution_attempt": {
            "number": 1,
            "kind": "local-container",
            "attempt_id": "e" * 24,
        },
    })
    contract = jobcontract.root_qualification_contract(job)
    manifest = dsformat.seal_manifest({
        "schema": dsformat.DATASET_SCHEMA,
        "dataset_sha256": "",
        "dataset": {
            "id": contract["dataset_id"],
            "name": contract["dataset_name"],
            "role": "root",
            "author": {"name": contract["author"]},
            "repository": repository,
        },
        "weights": {
            "repository": weights_repository,
            "revision": weights_revision,
        },
        "panel": {
            "panel_id": panel_id,
            "suite_token_hash_sha256": suite_sha,
            "panel_receipt_sha256": receipt_sha,
            "panel_receipt_file": "panel/panel-receipt.json",
            "tokenizer": {"identity_sha256": tokenizer_sha},
        },
        "capture": {"form": "hidden", "dtype": "BF16"},
        "runtime": {"lane": "sealed-ep8"},
    })
    manifest_raw = safety.canonical_bytes(manifest)
    canonical_capture = {
        "dataset_id": contract["dataset_id"],
        "dataset_name": contract["dataset_name"],
        "dataset_author": contract["author"],
        "dataset_repository": repository,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_manifest_file_sha256":
            hashlib.sha256(manifest_raw).hexdigest(),
        "weights_repository": weights_repository,
        "weights_revision": weights_revision,
        "capture_form": "hidden",
        "capture_dtype": "BF16",
        "runtime_lane": "sealed-ep8",
        "runtime_device": "cuda",
        "runtime_engine": "transformers-eager",
        "capture_tool_file": "engines/tools/hf_capture.py",
        "capture_schedule": "layer-outer",
        "panel": {
            "panel_id": panel_id,
            "suite_token_hash_sha256": suite_sha,
            "panel_receipt_sha256": receipt_sha,
            "tokenizer": {"identity_sha256": tokenizer_sha},
            "resolved_binding_evidence": {
                "binding_file": "panel-binding.json",
                "binding_file_sha256": binding_file_sha,
                "binding": binding,
            },
        },
        "unexpected_tensor_allowlist": {
            "artifact_sha256": allowlist_sha,
            "canonical_sorted_names_sha256": allowlist_names_sha,
            "exact_match": True,
            "duplicate_observed_keys": [],
            "missing_keys": [],
            "extra_keys": [],
        },
    }
    qualification = common.seal({
        "schema": "fidelity.root-qualification-receipt.v1",
        "qualified_at": "2026-01-01T00:00:00Z",
        "canonical_job_sha256": job["job_id_full"],
        "job_file_sha256": hashlib.sha256(
            safety.canonical_bytes(job)).hexdigest(),
        "dataset_repository": repository,
        "destination_repository": repository,
        "job_contract": contract,
        "captures": {"canonical": canonical_capture},
    })
    qualification_raw = safety.canonical_bytes(qualification)
    publication = common.seal({
        "schema": "fidelity.publish-root-receipt.v2",
        "repository": repository, "revision": "a" * 40,
        "revision_immutable": True, "private": False,
        "dataset_sha256": manifest["dataset_sha256"],
        "published_dataset_sha256": manifest["dataset_sha256"],
        "qualification_receipt_sha256":
            qualification["receipt_sha256"],
        "qualification_file_sha256":
            hashlib.sha256(qualification_raw).hexdigest(),
        "published_qualification_file_sha256":
            hashlib.sha256(qualification_raw).hexdigest(),
        "verified_after_publish": True,
        "result_archive_sha256": "f" * 64,
        "result_archive_bytes": 1024,
        "verified_anonymously": True, "verified_revision": "a" * 40,
    })
    return publication, manifest_raw, qualification_raw, panel_receipt_raw


def check_public_refetch():
    publication, manifest_raw, qualification_raw, panel_receipt_raw = (
        public_refetch_fixture())
    real_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout):
        check("anonymous current refetch has no authorization",
              request.get_header("Authorization") is None and timeout == 60)
        if request.full_url.endswith("receipts/root-qualification.json"):
            body = qualification_raw
        elif request.full_url.endswith("panel/panel-receipt.json"):
            body = panel_receipt_raw
        else:
            body = manifest_raw
        return FakeResponse(body)

    try:
        urllib.request.urlopen = fake_urlopen
        observed = safety.validate_current_public_root(publication)
        check("current public refetch accepted exact identities",
              observed["publicly_accessible"] is True)
        original_panel_receipt = panel_receipt_raw
        panel_receipt_raw = b"{}"
        try:
            safety.validate_current_public_root(publication)
        except safety.SafetyProofError:
            pass
        else:
            raise AssertionError("substituted public panel receipt was accepted")
        panel_receipt_raw = original_panel_receipt
        drifted = bytearray(qualification_raw)
        drifted[-2] = ord(" ")
        qualification_raw = bytes(drifted)
        try:
            safety.validate_current_public_root(publication)
        except safety.SafetyProofError:
            pass
        else:
            raise AssertionError("drifted current public root was accepted")
    finally:
        urllib.request.urlopen = real_urlopen


def main():
    canonical = fruit_binding()
    safety._validate_fruit_panel_binding(canonical)
    wrong = copy.deepcopy(canonical)
    wrong["panel"]["contexts"] = 25
    try:
        safety._validate_fruit_panel_binding(wrong)
    except safety.SafetyProofError:
        pass
    else:
        raise AssertionError("25-context wrong panel was accepted")
    for label, raw in (
            ("nested duplicate", b'{"outer":{"x":1,"x":2}}'),
            ("NaN constant", b'{"outer":{"x":NaN}}'),
            ("overflowed number", b'{"outer":{"x":1e999}}')):
        try:
            safety._strict_json(raw, label)
        except safety.SafetyProofError:
            pass
        else:
            raise AssertionError("%s safety JSON was accepted" % label)
    check_public_refetch()
    check("canonical Fruit contexts", canonical["panel"]["contexts"] == 16)
    print("PASS: exact Fruit width-two binding and strict nested safety JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
