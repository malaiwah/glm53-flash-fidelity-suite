#!/usr/bin/env python3
"""T26  the answer gets off the box -- for every verb, not just published roots.

The defect this covers was found by renting: a container-native run ended with
"receipts under /workspace/fidelity/receipts", on a RunPod pod whose volume is
pod-scoped, whose image runs no sshd, and whose REST API exposes no logs and no
files. ROOT-1's --publish-root-to covered a multi-GB root capture and nothing
else -- not `measure`, whose 4-40 KB receipt IS the submission object, not
`stage`, and not a FAILED run, whose evidence is the hardest to reach and the
most wanted.

Every rung here runs offline: the http rungs drive a stub server on loopback.
"""
from __future__ import annotations

import hashlib
import gzip
import http.server
import io
import json
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fidelity import resultsink as RS          # noqa: E402
from fidelity import jobcontract as JC              # noqa: E402
import fidelity_dataset as FD                  # noqa: E402

PASS = FAIL = 0
FAILED = []


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  PASS  %s" % label)
    else:
        FAIL += 1
        FAILED.append(label)
        print("  FAIL  %s%s" % (label, ("  -- " + detail) if detail else ""))


def _sealed(schema, **fields):
    doc = {"schema": schema, "receipt_sha256": ""}
    doc.update(fields)
    return RS.seal(doc)


def _run_root(tmp, *, receipt_bytes=200, failed=False, role="quant",
              publish=False, abandoned=False, weights_license_body=None):
    root = Path(tmp) / "run"
    (root / "receipts" / "done").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / ".secrets").mkdir(parents=True)
    (root / "receipts" / ".stream-work").mkdir(parents=True)
    bundle_payload = b"result-sink-selftest"
    bundle = JC.finalize_bundle_manifest([{
        "path": "bin/fidelity/resultsink.py",
        "bytes": len(bundle_payload),
        "sha256": hashlib.sha256(bundle_payload).hexdigest(),
    }], "result-sink-selftest")
    control = JC.finalize_bundle_manifest([{
        "path": "bin/fidelity/jobcontract.py",
        "bytes": 1,
        "sha256": "8" * 64,
    }], "result-sink-control")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    registry = {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "7" * 64}
    bundle_contract = hashlib.sha256(json.dumps(
        {"bundle": bundle, "registry": registry},
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    shards = [{"path": "model-00001-of-00001.safetensors", "bytes": 123}]
    shard_manifest_sha256 = hashlib.sha256(json.dumps(
        shards, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    target = {
        "repo_id": "owner/model",
        "revision": "1" * 40,
        "path": None,
        "surface": "tr3-published",
        "codec": "exl3-mcg",
        "bits": 4.0,
        "config_sha256": "2" * 64,
        "index_sha256": "3" * 64,
        "shard_manifest_sha256": shard_manifest_sha256,
        "model_bytes": 123,
        "shards": shards,
        "download_manifest": list(shards),
        "download_bytes_total": 123,
        "download_manifest_sha256": shard_manifest_sha256,
        "size_bytes": 123,
        "precision_label": "4 bpw",
        "container": "exl3",
        "quantizer_tool": "exl3-mcg",
        "quantizer_version": "test-quantizer-v1",
        "exllamav3_pin": None,
        "bits_per_weight_effective": None,
        "group_size": None,
        "shard_hash_verification": "full",
        "result_archive_contract": {
            "retained_content": [
                "receipts", "reports", "bounded-log-tails", "control"],
            "result_archive_max_members": 2048,
            "result_archive_max_uncompressed_bytes": 2 * 1024 ** 3,
            "result_archive_max_transfer_bytes": (
                2 * 1024 ** 3
                + (((2 * 1024 ** 3) + 16382) // 16383) * 5 + 64),
        },
    }
    panel = {
        "panel_ref": "panel--test",
        "panel_token_sha256": "4" * 64,
        "panel_receipt_sha256": "5" * 64,
        "contexts": 25,
        "scored_positions": 51175,
    }
    reference = {
        "reference_ref": "reference--test",
        "teacher_receipt_sha256": "6" * 64,
        "teacher_backend_identity_sha256": "7" * 64,
    }
    measurer = {
        "name": "tester", "handle": "tester",
        "url": "https://huggingface.co/tester",
        "is_artifact_author": False,
    }
    produced_by = {
        "pipeline": "selftest", "revision": "8" * 40,
        "dependencies": {"profile": "test-profile"},
    }
    job = {
        "schema": "fidelity-suite/job.v2",
        "role": role,
        "recipe": "runpod-controller-loss-drill",
        "lane": "streaming",
        "cold_runs": 2,
        "profile": {"profile_id": "test-profile", "lane": "streaming"},
        "timing": {"kind": "result-sink-selftest"},
        "target": target,
        "bundle": bundle,
        "control_plane": control,
        "bundle_registry": registry,
        "bundle_contract_sha256": bundle_contract,
        "panel": panel,
        "reference": reference,
        "measurer": measurer,
        "produced_by": produced_by,
        "scoring": {
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "compute_dtype": "float64",
            "reduction": "mean_of_run_means_tokenwise_kld",
        },
        "scope": {"kind": "selftest", "layers": "all"},
        "execution_attempt": {
            "number": 1,
            "kind": "local-container",
            "attempt_id": "1" * 24,
        },
    }
    if role == "root":
        panel_receipt_doc = _sealed(
            "selftest.panel-receipt.v1",
            panel_id="panel--test-root",
            tokenizer={"repository": "owner/tokenizer",
                       "revision": "9" * 40})
        panel_receipt_body = (
            json.dumps(panel_receipt_doc, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")
        tokenizer_binding = {
            "id": "owner/tokenizer",
            "repository": "owner/tokenizer",
            "revision": "9" * 40,
            "vocab_size": 16,
            "maximum_token_id_exclusive": 16,
            "identity_sha256": "e" * 64,
            "files": [{
                "name": "tokenizer.json", "bytes": 1, "sha256": "f" * 64,
            }],
            "files_verified": True,
            "receipt": {
                "declared_receipt_sha256": "0" * 64,
                "receipt_seal_mode": "self-blank",
                "receipt_file_sha256": "1" * 64,
                "receipt_file_bytes": 1,
            },
        }
        resolved_panel_binding = {
            "schema": "malaiwah.resolved-panel.v1",
            "panel": {
                "id": "panel--test-root", "name": "selftest panel",
                "role": "final", "contexts": 25, "context_length": 2048,
                "positions_per_context": 2047,
                "scored_positions_total": 51175,
                "suite_token_hash_sha256": "4" * 64,
                "file": "panel.json", "bytes": 1, "sha256": "2" * 64,
            },
            "receipt": {
                "file": "panel.receipt.json",
                "bytes": len(panel_receipt_body),
                "declared_receipt_sha256":
                    panel_receipt_doc["receipt_sha256"],
                "receipt_seal_mode": "self-blank",
                "receipt_file_sha256":
                    hashlib.sha256(panel_receipt_body).hexdigest(),
            },
            "tokenizer": tokenizer_binding,
            "content": {
                "manifest": [], "manifest_sha256": "3" * 64,
                "archive": {
                    "format": "ustar", "compression": "none",
                    "algorithm": "selftest", "bytes": 1,
                    "sha256": "4" * 64,
                },
            },
        }
        panel_binding_body = (
            json.dumps(
                resolved_panel_binding, sort_keys=True,
                separators=(",", ":")) + "\n").encode("utf-8")
        job["panel"] = {
            "panel_ref": "panel--test",
            "panel_token_sha256": "4" * 64,
            "panel_receipt_sha256": panel_receipt_doc["receipt_sha256"],
            "contexts": 25, "scored_positions": 51175,
            "binding_path": "inputs/panel.binding.json",
            "binding_file_sha256":
                hashlib.sha256(panel_binding_body).hexdigest(),
            "resolved_binding": resolved_panel_binding,
        }
        allowlist_names = ["unused.weight"]
        allowlist_body = (
            json.dumps(allowlist_names, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        allowlist_names_sha = hashlib.sha256(
            json.dumps(
                allowlist_names, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if weights_license_body is None:
            source_license = None
            runtime_license = None
            dataset_license = "mit"
        else:
            if not isinstance(weights_license_body, bytes) or not weights_license_body:
                raise AssertionError("weights_license_body must be nonempty bytes")
            source_license = {
                "source_path": "LICENSE",
                "dataset_path": "LICENSE",
                "bytes": len(weights_license_body),
                "sha256": hashlib.sha256(weights_license_body).hexdigest(),
            }
            runtime_license = {
                "source_file": "LICENSE",
                "dataset_path": "LICENSE",
                "bytes": source_license["bytes"],
                "sha256": source_license["sha256"],
            }
            dataset_license = "other"
        job["target"].update({
            "surface": "native-bf16", "codec": "bf16", "bits": 16})
        job["target"]["weights_license"] = source_license
        job["profile"] = {
            "profile_id": "root-hf-transformers-bf16",
            "lane": "root", "source": "native",
            "surface": "native-bf16", "form": "hidden",
            "engine": "hf-transformers",
            "compute_dtype": "bfloat16", "device": "cuda",
            "schedule": "two-fresh-process-qualification",
        }
        job["produced_by"] = dict(produced_by, dependencies={
            "profile": "root-hf-transformers-bf16",
            "lane": "streaming", "provider": "local-container"})
        job["capture"] = {
            "dataset_repository": "owner/dataset",
            "publish_root_to": "owner/dataset" if publish else None,
            "dataset_id": "dataset--selftest",
            "dataset_name": "selftest root",
            "author": "selftest",
            "dataset_license": dataset_license,
            "weights_license": source_license,
            "form": "hidden", "schedule": "layer-outer",
            "engine": "hf-transformers", "dtype": "bfloat16",
            "device": "cuda",
            "replay_device": "numpy", "replay_dtype": "float32",
            "vocab_chunk": 8192,
            "replay": {
                "device": "numpy", "dtype": "float32",
                "vocab_chunk": 8192},
            "root_protocol": {
                "schedule": "two-fresh-process-qualification",
                "fresh_processes": 2,
                "run_count_per_process": 1,
                "exact_self_comparison": True,
                "qualification_required": True,
                "canonical_publication_required": publish,
                "publication_mode":
                    "canonical-public" if publish else "qualified-unpublished",
            },
            "unexpected_tensor_allowlist": {
                "path": "inputs/allowlist.json",
                "artifact_sha256":
                    hashlib.sha256(allowlist_body).hexdigest(),
                "canonical_sorted_names_sha256": allowlist_names_sha,
            },
        }
        archive_uncompressed = 2048 + RS.ARCHIVE_MARGIN_BYTES
        job["target"]["root_capture_storage"] = {
            "form": "hidden",
            "storage_dtype": "bfloat16",
            "selected_prediction_positions": 51175,
            "vocab_size": 16,
            "hidden_size": 16,
            "bytes_per_element": 2,
            "fresh_processes": 2,
            "hidden_bytes_per_process": 512,
            "shared_head_bytes_per_process": 512,
            "bytes_per_process": 1024,
            "capture_bytes_total": 2048,
            "capture_archive_duplicate_upper_bound_bytes": 2048,
            "required_dataset_trees": 2,
            "result_archive_max_members": 102478,
            "result_archive_max_uncompressed_bytes": archive_uncompressed,
            "result_archive_max_transfer_bytes": (
                archive_uncompressed
                + ((archive_uncompressed + 16382) // 16383) * 5 + 64),
        }
    job = JC.finalize_job(job)
    job_bytes = json.dumps(job).encode("utf-8")
    census = _sealed(
        RS.TARGET_CENSUS_SCHEMA,
        verified_at="2026-01-01T00:00:00Z",
        job_id_full=job["job_id_full"],
        job_file_sha256=hashlib.sha256(job_bytes).hexdigest(),
        repository=target["repo_id"],
        revision=target["revision"],
        config_sha256=target["config_sha256"],
        index_sha256=target["index_sha256"],
        shard_manifest_sha256=target["shard_manifest_sha256"],
        model_bytes=target["model_bytes"],
        shards=target["shards"],
        index_shards=["model-00001-of-00001.safetensors"])
    (root / "receipts" / "fetch-target-census.json").write_text(
        json.dumps(census), encoding="utf-8")
    run_report_hashes = []
    for cold_run in (1, 2):
        report_path = root / "reports" / ("cold-run-%d.json" % cold_run)
        report_path.write_text(json.dumps({
            "schema": "fidelity-suite/selftest-kld-run.v1",
            "cold_run": cold_run,
            "profile": job["profile"]["profile_id"],
            "lane": job["lane"],
            "scoring": job["scoring"],
            "mean_tokenwise_kld": 0.125,
            "tokenwise_kld_sha256": "a" * 64,
        }, sort_keys=True), encoding="utf-8")
        run_report_hashes.append(RS.sha256_file(str(report_path)))
    measurement = RS.seal({
        "submission_schema": RS.QUANT_RECEIPT_SCHEMA,
        "receipt_sha256": "",
        "measured_at": "2026-01-01T00:00:00Z",
        "lane": job["lane"],
        "measurer": measurer,
        "produced_by": produced_by,
        "artifact": {
            "url": "https://huggingface.co/owner/model",
            "container": target["container"],
            "precision_label": target["precision_label"],
            "size_bytes": target["size_bytes"],
            "shard_hash_verification": target["shard_hash_verification"],
            "repository": target["repo_id"],
            "revision": target["revision"],
            "path": target["path"],
            "codec": {
                "family": target["codec"],
                "bits_per_weight_nominal": target["bits"],
                "bits_per_weight_effective":
                    target["bits_per_weight_effective"],
                "group_size": target["group_size"],
                "quantizer_tool": target["quantizer_tool"],
                "quantizer_version": target["quantizer_version"],
            },
            "config_sha256": target["config_sha256"],
            "index_sha256": target["index_sha256"],
            "scope": job["scope"],
        },
        "panel": {
            "panel_ref": panel["panel_ref"],
            "panel_token_sha256": panel["panel_token_sha256"],
            "panel_receipt_sha256": panel["panel_receipt_sha256"],
            "contexts": panel["contexts"],
            "scored_positions_total": panel["scored_positions"],
        },
        "metric": {
            "name": "mean_of_run_means_tokenwise_kld",
            "value": 0.125,
            "units": "nats",
            "direction": "reference_to_candidate",
        },
        "auxiliary_metrics": {},
        "estimator": {
            "accumulation_dtype": "float64",
            "logits_dtype": "fp32",
        },
        "measurement_scope": {
            "scored_positions": 51175,
            "contexts": 25,
            "positions_per_context": 2047,
            "covers_full_panel": True,
            "subset_detail": None,
            "position_filter": "all",
        },
        "determinism": {
            "run_count": 2,
            "cold_start_per_run": True,
            "run_means": [0.125, 0.125],
            "identical_across_runs": True,
            "evidence_kind": "tokenwise_kld_sha256",
            "evidence_hashes": ["a" * 64],
            "distinct_evidence_hash_count": 1,
            "per_run_report_sha256": run_report_hashes,
        },
        "environment": {},
        "cost": {"usd": None, "basis": None},
        "evidence": [],
        "disclosures": [{
            "code": "selftest_padding",
            "severity": "info",
            "affects_comparability": False,
            "detail": "x" * receipt_bytes,
        }],
        "reference": reference,
    })
    (root / "receipts" / "measurement-receipt.json").write_text(
        json.dumps(measurement), encoding="utf-8")
    (root / "receipts" / ".stream-work" / "huge.bin").write_text(
        "z" * 4096, encoding="utf-8")
    (root / ".secrets" / "hf_token").write_text(
        "hf_TOKEN_MUST_NEVER_LEAVE", encoding="utf-8")
    if role == "root":
        capture_identities = {}
        for tree, label in (("dataset", "canonical"),
                            ("dataset-repeat", "repeat")):
            process_label = (
                "root-cold-1" if label == "canonical" else "root-cold-2")
            tree_root = root / tree
            capture_dir = tree_root / "capture"
            runtime_dir = tree_root / "runtime"
            capture_dir.mkdir(parents=True)
            runtime_dir.mkdir()
            panel_dir = tree_root / "panel"
            panel_dir.mkdir()
            (panel_dir / "panel-receipt.json").write_bytes(
                panel_receipt_body)
            runtime_container = {"kind": "selftest"}
            binding_evidence = {
                "binding_file": "panel.binding.json",
                "binding_file_sha256":
                    job["panel"]["binding_file_sha256"],
                "binding": resolved_panel_binding,
            }
            allowlist_evidence = {
                "schema":
                    "malaiwah.unexpected-tensor-allowlist-binding.v1",
                "artifact_file": "allowlist.json",
                "artifact_bytes": len(allowlist_body),
                "artifact_sha256":
                    job["capture"]["unexpected_tensor_allowlist"][
                        "artifact_sha256"],
                "canonical_sorted_names_sha256": allowlist_names_sha,
                "expected_count": 1,
                "expected_keys": list(allowlist_names),
                "observed_count": 1,
                "observed_keys": list(allowlist_names),
                "duplicate_observed_keys": [],
                "missing_keys": [],
                "extra_keys": [],
                "exact_match": True,
            }
            if weights_license_body is not None:
                (tree_root / "LICENSE").write_bytes(weights_license_body)
            tensor_body = b"independent-capture"
            (capture_dir / "context-0000.bin").write_bytes(tensor_body)
            content_digest = hashlib.sha256(tensor_body).hexdigest()
            capture_doc = {
                "schema": "test-capture.v1",
                "run_name": process_label,
                "form": "hidden",
                "dtype": "BF16",
                "capture_content_digest": content_digest,
            }
            capture_body = (json.dumps(
                capture_doc, sort_keys=True, separators=(",", ":"))
                + "\n").encode("utf-8")
            (capture_dir / "manifest.json").write_bytes(capture_body)
            runtime_doc = {
                "schema": "test-runtime.v1",
                "lane": job["lane"],
                "stack_fingerprint": {
                    "device": "cuda", "engine": "transformers-eager",
                },
                "stack_fingerprint_sha256": "c" * 64,
                "lane_identity_sha256": "d" * 64,
                "weights": {
                    "repository": job["target"]["repo_id"],
                    "revision": job["target"]["revision"],
                    "model_revision": job["target"]["revision"],
                },
                "runtime_environment": {"cold_run": process_label},
                "container": runtime_container,
                "capture_tool": {
                    "file": "engines/tools/hf_capture.py",
                    "schedule": "layer-outer",
                    "resolved_panel_binding": binding_evidence,
                    "unexpected_tensor_allowlist": allowlist_evidence,
                    "weights_license": runtime_license,
                },
            }
            runtime_body = (json.dumps(
                runtime_doc, sort_keys=True, separators=(",", ":"))
                + "\n").encode("utf-8")
            (runtime_dir / "capture-runtime.json").write_bytes(runtime_body)
            checksum_rows = []
            checksum_paths = [
                "capture/context-0000.bin", "capture/manifest.json",
                "runtime/capture-runtime.json", "panel/panel-receipt.json",
            ]
            if weights_license_body is not None:
                checksum_paths.append("LICENSE")
            for relative in sorted(checksum_paths):
                body = (tree_root / relative).read_bytes()
                checksum_rows.append(
                    "%s  %s\n" % (hashlib.sha256(body).hexdigest(), relative))
            checksums_body = "".join(checksum_rows).encode("utf-8")
            (tree_root / "checksums.txt").write_bytes(checksums_body)
            dataset_doc = {
                "schema": "malaiwah.fidelity-dataset.v1",
                "dataset_sha256": "",
                "seal": {
                    "checksums_file": "checksums.txt",
                    "checksums_sha256":
                        hashlib.sha256(checksums_body).hexdigest(),
                },
                "dataset": {
                    "id": job["capture"]["dataset_id"],
                    "name": job["capture"]["dataset_name"],
                    "role": "root",
                    "author": {"name": job["capture"]["author"]},
                    "repository": job["capture"]["dataset_repository"],
                    "license": dataset_license,
                },
                "weights": {
                    "repository": job["target"]["repo_id"],
                    "revision": job["target"]["revision"],
                    "model_revision": job["target"]["revision"],
                    "quantized": False,
                    "codec": None,
                    "declared_bits": None,
                },
                "panel": {
                    "panel_id": resolved_panel_binding["panel"]["id"],
                    "suite_token_hash_sha256":
                        resolved_panel_binding["panel"][
                            "suite_token_hash_sha256"],
                    "panel_receipt_sha256":
                        resolved_panel_binding["receipt"][
                            "declared_receipt_sha256"],
                    "panel_receipt_file": "panel/panel-receipt.json",
                    "tokenizer": tokenizer_binding,
                },
                "capture": {
                    "manifest_file": "capture/manifest.json",
                    "manifest_file_sha256":
                        hashlib.sha256(capture_body).hexdigest(),
                    "capture_content_digest": content_digest,
                    "form": "hidden",
                    "dtype": "BF16",
                },
                "runtime": {
                    "file": "runtime/capture-runtime.json",
                    "file_sha256": hashlib.sha256(runtime_body).hexdigest(),
                    "stack_fingerprint_sha256": "c" * 64,
                    "lane_identity_sha256": "d" * 64,
                    "lane": job["lane"],
                    "source": "native",
                },
            }
            dataset_doc["dataset_sha256"] = hashlib.sha256(
                RS.canonical_json(dataset_doc).encode("utf-8")).hexdigest()
            dataset_body = (json.dumps(
                dataset_doc, sort_keys=True, separators=(",", ":"))
                + "\n").encode("utf-8")
            (tree_root / "fidelity-dataset.json").write_bytes(dataset_body)
            capture_identities[label] = {
                "process_label": process_label,
                "dataset_id": "dataset--selftest",
                "dataset_name": "selftest root",
                "dataset_author": "selftest",
                "dataset_repository": "owner/dataset",
                "dataset_license": dataset_license,
                "weights_license": runtime_license,
                "weights_license_file_sha256": (
                    source_license["sha256"] if source_license else None),
                "weights_license_file_bytes": (
                    source_license["bytes"] if source_license else None),
                "dataset_sha256": dataset_doc["dataset_sha256"],
                "dataset_manifest_file_sha256":
                    hashlib.sha256(dataset_body).hexdigest(),
                "capture_manifest": "capture/manifest.json",
                "capture_manifest_sha256":
                    hashlib.sha256(capture_body).hexdigest(),
                "capture_content_digest": content_digest,
                "capture_form": "hidden",
                "capture_dtype": "BF16",
                "runtime_manifest": "runtime/capture-runtime.json",
                "runtime_manifest_sha256":
                    hashlib.sha256(runtime_body).hexdigest(),
                "runtime_lane": job["lane"],
                "runtime_device": "cuda",
                "runtime_engine": "transformers-eager",
                "runtime_container": runtime_container,
                "capture_tool_file": "engines/tools/hf_capture.py",
                "capture_schedule": "layer-outer",
                "panel": {
                    "panel_id": resolved_panel_binding["panel"]["id"],
                    "suite_token_hash_sha256":
                        resolved_panel_binding["panel"][
                            "suite_token_hash_sha256"],
                    "panel_receipt_sha256":
                        resolved_panel_binding["receipt"][
                            "declared_receipt_sha256"],
                    "tokenizer": tokenizer_binding,
                    "resolved_binding_evidence": binding_evidence,
                },
                "unexpected_tensor_allowlist": allowlist_evidence,
                "stack_fingerprint_sha256": "c" * 64,
                "lane_identity_sha256": "d" * 64,
                "weights_repository": target["repo_id"],
                "weights_revision": target["revision"],
                "determinism_run_count": 1,
            }
        qualification = _sealed(
            RS.ROOT_QUALIFICATION_SCHEMA,
            qualified_at="2026-01-01T00:00:00Z",
            canonical_job_sha256=job["job_id_full"],
            job_file_sha256=hashlib.sha256(job_bytes).hexdigest(),
            dataset_repository=job["capture"]["dataset_repository"],
            destination_repository=job["capture"]["publish_root_to"],
            job_contract=JC.root_qualification_contract(job),
            captures=capture_identities,
            comparison={
                "path": "comparison-receipt.json",
                "file_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
                "comparison_kind": "reproduction_confirmation",
                "mean_kld": 0.0,
                "max_kld": 0.0,
                "top1_agreement": 1.0,
            },
            comparator={
                "requested_replay_device": "numpy",
                "requested_replay_dtype": "float32",
                "requested_vocab_chunk": 8192,
                "device": "cpu",
                "replay_backend": "numpy:cpu:float32",
                "estimator_backend": "numpy-streaming-softmax-kld",
                "accumulation_dtype": "float64",
                "vocab_chunk": 8192,
                "force_compute_agreed": True,
            },
            verification={
                "canonical": {
                    "receipt_sha256": "3" * 64,
                    "file_sha256": "4" * 64,
                },
                "repeat": {
                    "receipt_sha256": "5" * 64,
                    "file_sha256": "6" * 64,
                },
            },
            reproduction_confirmation={
                "two_fresh_processes": True,
                "distinct_dataset_roots": True,
                "both_independently_verified": True,
                "exact_zero_comparison": True,
                "canonical_dataset_only": True,
            })
        (root / "receipts" / "root-qualification.json").write_text(
            json.dumps(qualification), encoding="utf-8")
        if publish:
            qualification_file_sha = hashlib.sha256(
                (root / "receipts" / "root-qualification.json").read_bytes()
            ).hexdigest()
            canonical_dataset_sha = capture_identities[
                "canonical"]["dataset_sha256"]
            publication = _sealed(
                RS.ROOT_PUBLICATION_SCHEMA,
                repository=job["capture"]["dataset_repository"],
                revision="a" * 40,
                private=False,
                verified_anonymously=True,
                revision_immutable=True,
                dataset_sha256=canonical_dataset_sha,
                published_dataset_sha256=canonical_dataset_sha,
                qualification_receipt_sha256=qualification["receipt_sha256"],
                qualification_file_sha256=qualification_file_sha,
                published_qualification_file_sha256=qualification_file_sha,
                verified_after_publish=True,
                verified_revision="a" * 40,
                result_archive_sha256="b" * 64,
                result_archive_bytes=123)
            (root / "receipts" / "publish-root.json").write_text(
                json.dumps(publication), encoding="utf-8")
    (root / "job.json").write_bytes(job_bytes)
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "setup.log").write_text("setup fine\n", encoding="utf-8")
    (root / "logs" / "capture.log").write_text(
        "b" * (RS.LOG_TAIL_BYTES + 5000)
        + "\nREFUSED [capture_failed]: the reason\n", encoding="utf-8")
    (root / "receipts" / "done" / "setup.done").write_text(
        "stage=setup\n", encoding="utf-8")
    if abandoned:
        (root / "ABANDONED.json").write_text(json.dumps({
            "schema": "fidelity-suite/abandoned.v2",
            "reason": "test abandonment",
            "stage_process_group_stopped": True,
        }) + "\n", encoding="utf-8")
    return root


def _doctor_root(tmp, status="ok"):
    root = Path(tmp) / "doctor"
    (root / "receipts").mkdir(parents=True)
    (root / "receipts" / "doctor.json").write_text(json.dumps({
        "schema": "malaiwah.fidelity-doctor.v1",
        "status": status,
        "report": ["offline doctor evidence"],
    }) + "\n", encoding="utf-8")
    return root


def con(_text):
    pass


class _Collector(http.server.BaseHTTPRequestHandler):
    received = []

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Collector.received.append({
            "method": "PUT", "body": self.rfile.read(n),
            "status": self.headers.get("X-Fidelity-Status"),
            "auth": self.headers.get("Authorization"),
        })
        self.send_response(200); self.end_headers()

    do_POST = do_PUT

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/collect" % srv.server_address[1]


def rung_parse():
    print("[T26.1] the sink list is parsed before anything is spent")
    s = RS.parse_sinks([])
    check("R1 stdout is unconditional and needs no flag",
          len(s) == 1 and s[0].scheme == "stdout")
    s = RS.parse_sinks(["file:/tmp/a", "https://h/u"])
    check("R2 stdout stays FIRST, so a later sink's failure cannot eat the "
          "answer", [x.scheme for x in s] == ["stdout", "file", "http"])
    s = RS.parse_sinks([], env={"FIDELITY_RESULT_SINK": "file:/tmp/a,https://h/u"})
    check("R3 the environment is a sink channel -- the one providers do NOT "
          "echo back in their console", [x.scheme for x in s[1:]] == ["file", "http"])
    s = RS.parse_sinks(["file:/tmp/a"], env={"FIDELITY_RESULT_SINK": "file:/tmp/a"})
    check("R4 the same sink named twice is delivered once", len(s) == 2)
    for bad, why in (("ftp://h/x", "unknown scheme"),
                     ("/tmp/plain", "a bare path is not a URI")):
        try:
            RS.parse_sinks([bad])
            check("R5 %s is refused (%s)" % (bad, why), False, "accepted it")
        except RS.SinkError:
            check("R5 %s is refused (%s)" % (bad, why), True)
    try:
        RS.parse_sinks(["hf://me/repo"])
        check("R6 hf:// is refused, naming --publish-root-to instead", False)
    except RS.SinkError as exc:
        check("R6 hf:// is refused, naming --publish-root-to instead",
              "publish-root-to" in str(exc))


def rung_content():
    print("[T26.2] what leaves the box, and what never does")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["setup", "seal"])
        paths = [f["path"] for f in summary["files"]]
        check("R7 the receipt and the job document are carried",
              "receipts/measurement-receipt.json" in paths and "job.json" in paths)
        check("R8 .secrets/ is NEVER in the manifest",
              not any(".secrets" in p for p in paths), "%s" % paths)
        check("R9 .stream-work/ (the multi-GB scratch tree) is not either",
              not any(".stream-work" in p for p in paths), "%s" % paths)
        check("R10 every carried file is identified by sha256",
              all(len(f["sha256"]) == 64 for f in summary["files"]))
        blob = RS._bundle(root, summary)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            names = tar.getnames()
        check("R11 the tar.gz carries the summary alongside the receipts",
              "result-summary.json" in names)
        check("R12 ... and no secret rides along in the tar either",
              not any(".secrets" in n for n in names), "%s" % names)


def rung_logs():
    print("[T26.7] the logs travel, because a failure report without one is not "
          "a report")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        summary = RS.build_summary(root, "capture", "failed",
                                   ["setup", "capture"], None, "capture")
        paths = [f["path"] for f in summary["files"]]
        check("R28 stage logs are carried, not just receipts",
              "logs/capture.log" in paths and "logs/setup.log" in paths,
              "%s" % paths)
        blob = RS._bundle(root, summary)
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            body = tar.extractfile("logs/capture.log").read()
        check("R29 an oversize log is TAIL-capped, keeping the end where the "
              "reason is", len(body) < RS.LOG_TAIL_BYTES + 500
              and b"REFUSED [capture_failed]: the reason" in body)
        check("R30 ... and the truncation is announced in the bytes, so nobody "
              "reads a capped log as a whole one",
              b"earlier bytes omitted" in body)
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            RS._deliver_stdout(root, summary, con)
        text = buf.getvalue()
        check("R31 stdout shows the FAILING stage's log inline -- the one thing "
              "a pod-scoped volume takes to the grave",
              "logs/capture.log (the stage that failed)" in text
              and "REFUSED [capture_failed]: the reason" in text)

    # A workload deadline that expires during cold run 2 must not cost the
    # sealed cold run 1 (GLM-5.3, 2026-09-04: rescued by hand from a pod that
    # was about to be destroyed). A failed root capture archive carries every
    # dataset tree a capture finished SEALING, and never a partial one.
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        shutil.rmtree(root / "dataset-repeat")
        partial = root / "dataset-repeat"
        partial.mkdir()
        (partial / "fidelity-dataset.json").write_text(
            json.dumps({"schema": "malaiwah.fidelity-dataset.v1",
                        "dataset_sha256": ""}), encoding="utf-8")
        summary = RS.build_summary(
            root, "capture", "failed", ["setup", "capture", "verify"],
            None, "capture_repeat")
        paths = [f["path"] for f in summary["files"]]
        check("R31b a failed root capture archive salvages the SEALED cold run 1",
              "dataset/fidelity-dataset.json" in paths
              and "dataset/checksums.txt" in paths, "%s" % paths[:12])
        check("R31c ... and never a partial, unsealed dataset tree",
              not any(p.startswith("dataset-repeat/") for p in paths))
        blob = RS.build_archive(root, summary)
        verified = RS.verify_archive(blob)
        check("R31d the salvaged archive verifies as a failed capture result",
              verified["manifest"]["status"] == "failed")
        with tempfile.TemporaryDirectory() as out:
            RS.extract_verified_archive(blob, Path(out) / "result")
            check("R31e ... and extracts the sealed dataset for a later resume",
                  (Path(out) / "result" / "dataset" / "fidelity-dataset.json").is_file()
                  and not (Path(out) / "result" / "dataset-repeat").exists())


def rung_http():
    print("[T26.3] the https sink, against a real server on loopback")
    srv, url = _serve()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(tmp)
            summary = RS.build_summary(root, "measure", "ok", ["seal"])
            res = RS.deliver(root, RS.parse_sinks([url]), summary, con)
            got = [r for r in res if r["scheme"] == "http"]
            check("R13 the bundle is PUT and the endpoint answers 200",
                  got and got[0].get("ok") and got[0].get("code") == 200,
                  "%s" % got)
            body = _Collector.received[-1]["body"] if _Collector.received else b""
            check("R14 the body is a readable gzip tar of the receipts",
                  b"hf_TOKEN_MUST_NEVER_LEAVE" not in body and len(body) > 0)
            with tarfile.open(fileobj=io.BytesIO(body)) as tar:
                names = tar.getnames()
            check("R15 ... carrying the receipt the registry ingests",
                  "receipts/measurement-receipt.json" in names, "%s" % names)
            check("R16 the status rides in a header a collector can route on",
                  _Collector.received[-1]["status"] == "ok")
            os.environ["FIDELITY_RESULT_SINK_AUTH"] = "Bearer TESTVALUE"
            try:
                RS.deliver(root, RS.parse_sinks([url]), summary, con)
                check("R17 an Authorization header comes from the environment, "
                      "never argv",
                      _Collector.received[-1]["auth"] == "Bearer TESTVALUE")
            finally:
                os.environ.pop("FIDELITY_RESULT_SINK_AUTH", None)
    finally:
        srv.shutdown()

    print("[T26.4] a sink that fails does not become a measurement result")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        # A port nothing listens on: the connection is refused, not hung.
        s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()
        res = RS.deliver(root, RS.parse_sinks(["http://127.0.0.1:%d/x" % dead]),
                         summary, con)
        http_r = [r for r in res if r["scheme"] == "http"][0]
        stdout_r = [r for r in res if r["scheme"] == "stdout"][0]
        check("R18 the unreachable sink is reported as failed", not http_r["ok"])
        check("R19 ... and stdout still delivered the answer anyway",
              stdout_r["ok"])
        check("R20 the failure names the host but NOT the query string, which "
              "on a presigned URL is the credential",
              "127.0.0.1" in http_r["error"] and "?" not in http_r["error"])


def rung_cap():
    print("[T26.5] a payload too big for a log frame says so")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, receipt_bytes=RS.STDOUT_CAP_BYTES + 10)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            RS._deliver_stdout(root, summary, con)


        text = buf.getvalue()
        check("R21 the oversize receipt is withheld, not dumped",
              "WITHHELD" in text)
        receipt_sha = next(
            item["sha256"] for item in summary["files"]
            if item["path"] == "receipts/measurement-receipt.json")
        check("R22 ... and its sha256 is in the frame regardless, so the "
              "artifact is still identifiable", receipt_sha in text)
        check("R23 the frame markers are present and greppable",
              RS.BEGIN in text and RS.END in text)


def _with_duplicate(raw, key, value):
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return ("{%s:%s,%s" % (
        json.dumps(key), json.dumps(value), text[1:])).encode("utf-8")


def _repack(blob, *, omit=None, replace=None):
    omit = set(omit or ())
    replace = dict(replace or {})
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as source:
        parts = []
        for member in source.getmembers():
            if member.name in omit:
                continue
            body = source.extractfile(member).read()
            parts.append((member.name, replace.get(member.name, body)))
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, body in parts:
                archive.addfile(RS._tar_info(name, len(body)), io.BytesIO(body))
    return output.getvalue()


def _rewrite_consistent(blob, replacements, manifest_updates):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as source:
        bodies = {
            member.name: source.extractfile(member).read()
            for member in source.getmembers()
        }
    manifest = json.loads(bodies.pop(RS.RESULT_MANIFEST_NAME).decode("utf-8"))
    bodies.update(replacements)
    for record in manifest["files"]:
        if record["path"] in replacements:
            body = replacements[record["path"]]
            record["bytes"] = len(body)
            record["source_bytes"] = len(body)
            record["sha256"] = hashlib.sha256(body).hexdigest()
    manifest.update(manifest_updates)
    bodies[RS.RESULT_MANIFEST_NAME] = (
        json.dumps(RS.seal(manifest, field="manifest_sha256"),
                   indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, body in sorted(bodies.items()):
                archive.addfile(RS._tar_info(name, len(body)), io.BytesIO(body))
    return output.getvalue()

def _coherent_root_substitution(blob, kind):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as source:
        bodies = {
            member.name: source.extractfile(member).read()
            for member in source.getmembers()
        }
    prefix = "dataset"
    manifest_name = prefix + "/fidelity-dataset.json"
    checksums_name = prefix + "/checksums.txt"
    qualification_name = "receipts/root-qualification.json"
    manifest = json.loads(bodies[manifest_name].decode("utf-8"))
    qualification = json.loads(bodies[qualification_name].decode("utf-8"))
    identity = qualification["captures"]["canonical"]
    replacements = {}

    def replace_checksum(relative, body):
        rows = []
        for line in bodies[checksums_name].decode("utf-8").splitlines():
            digest, path = line.split("  ", 1)
            if path == relative:
                digest = hashlib.sha256(body).hexdigest()
            rows.append("%s  %s\n" % (digest, path))
        checksums = "".join(rows).encode("utf-8")
        bodies[checksums_name] = checksums
        replacements[checksums_name] = checksums

    if kind == "weights_revision":
        manifest["weights"]["revision"] = "0" * 40
        manifest["weights"]["model_revision"] = "0" * 40
    elif kind == "runtime_lane":
        runtime_name = prefix + "/runtime/capture-runtime.json"
        runtime = json.loads(bodies[runtime_name].decode("utf-8"))
        runtime["lane"] = "substituted-lane"
        runtime_body = (
            json.dumps(runtime, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")
        bodies[runtime_name] = runtime_body
        replacements[runtime_name] = runtime_body
        replace_checksum("runtime/capture-runtime.json", runtime_body)
        runtime_sha = hashlib.sha256(runtime_body).hexdigest()
        manifest["runtime"]["file_sha256"] = runtime_sha
        identity["runtime_manifest_sha256"] = runtime_sha
    elif kind == "allowlist_names":
        runtime_name = prefix + "/runtime/capture-runtime.json"
        runtime = json.loads(bodies[runtime_name].decode("utf-8"))
        evidence = runtime["capture_tool"]["unexpected_tensor_allowlist"]
        evidence["expected_keys"] = ["model.substituted"]
        evidence["observed_keys"] = ["model.substituted"]
        evidence["expected_count"] = 1
        evidence["observed_count"] = 1
        identity["unexpected_tensor_allowlist"] = json.loads(
            json.dumps(evidence))
        runtime_body = (
            json.dumps(runtime, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")
        bodies[runtime_name] = runtime_body
        replacements[runtime_name] = runtime_body
        replace_checksum("runtime/capture-runtime.json", runtime_body)
        runtime_sha = hashlib.sha256(runtime_body).hexdigest()
        manifest["runtime"]["file_sha256"] = runtime_sha
        identity["runtime_manifest_sha256"] = runtime_sha
    elif kind == "panel_receipt":
        receipt_name = prefix + "/panel/panel-receipt.json"
        receipt_body = b'{"receipt_sha256":"%s","substituted":true}\n' % (
            b"0" * 64)
        bodies[receipt_name] = receipt_body
        replacements[receipt_name] = receipt_body
        replace_checksum("panel/panel-receipt.json", receipt_body)
    elif kind == "profile_surface":
        qualification["job_contract"]["target"]["surface"] = (
            "substituted-surface")
        qualification["job_contract"]["profile"]["surface"] = (
            "substituted-surface")
    else:
        raise AssertionError("unknown coherent substitution %r" % kind)

    if kind in ("runtime_lane", "allowlist_names", "panel_receipt"):
        manifest["seal"]["checksums_sha256"] = hashlib.sha256(
            bodies[checksums_name]).hexdigest()
    if kind != "profile_surface":
        manifest["dataset_sha256"] = ""
        manifest["dataset_sha256"] = hashlib.sha256(
            RS.canonical_json(manifest).encode("utf-8")).hexdigest()
        manifest_body = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            + "\n").encode("utf-8")
        replacements[manifest_name] = manifest_body
        identity["dataset_sha256"] = manifest["dataset_sha256"]
        identity["dataset_manifest_file_sha256"] = hashlib.sha256(
            manifest_body).hexdigest()
    qualification = RS.seal(qualification)
    replacements[qualification_name] = (
        json.dumps(qualification, sort_keys=True, separators=(",", ":"))
        + "\n").encode("utf-8")
    return _rewrite_consistent(blob, replacements, {})


def _unsafe_tar(name, *, link=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside"
            archive.addfile(info)
        else:
            body = b"escape"
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return output.getvalue()


def _refused(label, operation):
    try:
        operation()
    except RS.ArchiveError:
        check(label, True)
    else:
        check(label, False, "unsafe/incomplete archive was accepted")


def rung_archive():
    print("[T26.8] deterministic self-verifying off-pod result archive")
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["setup", "seal"])
        first = RS.build_archive(root, summary)
        second = RS.build_archive(root, summary)
        check("R32 rebuilding unchanged evidence is byte-for-byte deterministic",
              first == second)
        tar_bytes = gzip.decompress(first)
        check("R83 result archive uses stored DEFLATE, not paid CPU compression",
              first[8] == 0 and len(first) > len(tar_bytes))
        identity = RS.verify_archive(
            first, expected_sha256=hashlib.sha256(first).hexdigest(),
            expected_bytes=len(first))
        check("R33 transfer size/hash and every manifested member verify",
              identity["archive_bytes"] == len(first)
              and identity["manifest"]["schema"] == RS.RESULT_MANIFEST_SCHEMA)
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
            manifest_raw = archive.extractfile(RS.RESULT_MANIFEST_NAME).read()
            job_raw = archive.extractfile("job.json").read()
            receipt_raw = archive.extractfile(
                "receipts/measurement-receipt.json").read()
            census_raw = archive.extractfile(RS.TARGET_CENSUS_PATH).read()
            delivered_log = archive.extractfile("logs/capture.log").read()
        duplicate_manifest = _repack(first, replace={
            RS.RESULT_MANIFEST_NAME: _with_duplicate(
                manifest_raw, "schema", RS.RESULT_MANIFEST_SCHEMA)})
        _refused("R64 duplicate result-manifest key is refused",
                 lambda: RS.verify_archive(duplicate_manifest))
        duplicate_job = _rewrite_consistent(first, {
            "job.json": _with_duplicate(job_raw, "role", "quant")}, {})
        _refused("R65 duplicate job.json key is refused",
                 lambda: RS.verify_archive(duplicate_job))
        duplicate_receipt = _rewrite_consistent(first, {
            "receipts/measurement-receipt.json": _with_duplicate(
                receipt_raw, "submission_schema", RS.QUANT_RECEIPT_SCHEMA)}, {})
        _refused("R66 duplicate measurement receipt key is refused",
                 lambda: RS.verify_archive(duplicate_receipt))
        census_doc = json.loads(census_raw.decode("utf-8"))
        census_doc["repository"] = "owner/wrong-model"
        census_doc = RS.seal(census_doc)
        cross_target_census = _rewrite_consistent(first, {
            RS.TARGET_CENSUS_PATH:
                RS.canonical_json(census_doc).encode("utf-8")}, {})
        _refused("R67 target census cannot cross target identity",
                 lambda: RS.verify_archive(cross_target_census))
        log_record = next(
            item for item in identity["manifest"]["files"]
            if item["path"] == "logs/capture.log")
        check("R34 log records hash the delivered tail bytes, not discarded bytes",
              log_record["delivery"] == "tail"
              and log_record["bytes"] == len(delivered_log)
              and log_record["sha256"]
              == hashlib.sha256(delivered_log).hexdigest())

        extracted = Path(tmp) / "verified"
        RS.extract_verified_archive(
            first, extracted,
            expected_sha256=identity["archive_sha256"],
            expected_bytes=identity["archive_bytes"])
        check("R35 safe extraction occurs only after full verification",
              (extracted / "job.json").read_bytes()
              == (root / "job.json").read_bytes()
              and (extracted / RS.RESULT_MANIFEST_NAME).is_file())
        events = []
        original_fsync_directory = RS._fsync_directory
        original_write_durable = RS._stream_exclusive_durable

        def traced_fsync_directory(path):
            events.append(("fsync-directory", str(Path(path))))
            return original_fsync_directory(path)

        def traced_write_durable(path, source):
            events.append(("write-start", str(Path(path))))
            original_write_durable(path, source)
            events.append(("write-durable", str(Path(path))))

        RS._fsync_directory = traced_fsync_directory
        RS._stream_exclusive_durable = traced_write_durable
        try:
            durable = Path(tmp) / "durable"
            RS.extract_verified_archive(first, durable)
            extracted_order = list(events)
            events[:] = []
            archive_path = Path(tmp) / "written" / RS.ARCHIVE_NAME
            RS.write_archive(root, summary, archive_path)
            archive_order = list(events)
        finally:
            RS._fsync_directory = original_fsync_directory
            RS._stream_exclusive_durable = original_write_durable
        last_extracted_parent = (
            "fsync-directory", str((Path(tmp) / "durable").parent))
        last_archive_parent = (
            "fsync-directory", str((Path(tmp) / "written")))
        check("R63 extraction and archive publication fsync files before the "
              "final parent-directory durability barrier",
              extracted_order[-1:] == [last_extracted_parent]
              and archive_order[-1:] == [last_archive_parent]
              and any(kind == "write-durable"
                      for kind, _path in extracted_order))


        _refused("R36 a truncated transfer is a hard verification error",
                 lambda: RS.verify_archive(first[:len(first) // 2]))
        _refused("R37 a tampered member is a hard verification error",
                 lambda: RS.verify_archive(_repack(
                     first, replace={"logs/setup.log": b"setup evil\n"})))
        _refused("R38 a missing manifested member is a hard verification error",
                 lambda: RS.verify_archive(_repack(first, omit={"job.json"})))
        _refused("R39 a wrong remote transfer digest is a hard error",
                 lambda: RS.verify_archive(first, expected_sha256="0" * 64))
        _refused("R62 uppercase transfer SHA-256 is not a canonical identity",
                 lambda: RS.verify_archive(
                     first,
                     expected_sha256=hashlib.sha256(first).hexdigest().upper()))
        _refused("R40 tar traversal is rejected before extraction",
                 lambda: RS.verify_archive(_unsafe_tar("../outside")))
        _refused("R41 tar links are rejected before extraction",
                 lambda: RS.verify_archive(_unsafe_tar("receipts/link", link=True)))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root", publish=True)
        summary = RS.build_summary(
            root, "capture", "ok", ["qualify_root", "publish_root"])
        root_blob = RS.build_archive(root, summary)
        root_verified = RS.verify_archive(root_blob)
        check("R42 a completed published root carries qualification and immutable "
              "publish/refetch evidence",
              root_verified["manifest"]["role"] == "root"
              and root_verified["manifest"]["publication_requested"])
        extracted_root = Path(tmp) / "extracted-root"
        RS.extract_verified_archive(root_blob, extracted_root)
        check("R68 both independently qualified dataset payloads survive extraction",
              (extracted_root / "dataset" / "capture"
               / "context-0000.bin").read_bytes() == b"independent-capture"
              and (extracted_root / "dataset-repeat" / "capture"
                   / "context-0000.bin").read_bytes()
              == b"independent-capture")
        tampered_dataset = _rewrite_consistent(root_blob, {
            "dataset/capture/context-0000.bin": b"tampered-capture"}, {})
        _refused("R69 re-manifested dataset tamper still fails qualification proof",
                 lambda: RS.verify_archive(tampered_dataset))

    with tempfile.TemporaryDirectory() as tmp:
        source_license_body = b"exact upstream license\n"
        root = _run_root(
            tmp, role="root", weights_license_body=source_license_body)
        summary = RS.build_summary(
            root, "capture", "qualified-unpublished",
            ["qualify_root", "publish_root"], failed_stage="publish_root")
        licensed_blob = RS.build_archive(root, summary)
        licensed = RS.verify_archive(licensed_blob)
        with tarfile.open(
                fileobj=io.BytesIO(licensed_blob), mode="r:gz") as archive:
            canonical_license = archive.extractfile("dataset/LICENSE").read()
            repeat_license = archive.extractfile(
                "dataset-repeat/LICENSE").read()
        check("R91 non-MIT root archive binds both exact source-license copies",
              licensed["manifest"]["role"] == "root"
              and canonical_license == source_license_body
              and repeat_license == source_license_body)
        tampered_license = _rewrite_consistent(
            licensed_blob, {"dataset/LICENSE": b"wrong upstream license\n"}, {})
        _refused("R92 source-license byte drift is refused off-pod",
                 lambda: RS.verify_archive(tampered_license))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        summary = RS.build_summary(
            root, "capture", "qualified-unpublished",
            ["qualify_root", "publish_root"], failed_stage="publish_root")
        bound_root_blob = RS.build_archive(root, summary)
        for label, kind in (
                ("R84 coherently resealed wrong weights revision is refused",
                 "weights_revision"),
                ("R85 coherently resealed wrong runtime lane is refused",
                 "runtime_lane"),
                ("R86 coherently resealed wrong profile/surface is refused",
                 "profile_surface"),
                ("R87 substituted raw panel receipt is refused",
                 "panel_receipt"),
                ("R88 substituted allowlist names are refused",
                 "allowlist_names")):
            substituted = _coherent_root_substitution(
                bound_root_blob, kind)
            _refused(label, lambda value=substituted:
                     RS.verify_archive(value))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root", publish=True)
        (root / "receipts" / "publish-root.json").unlink()
        summary = RS.build_summary(
            root, "capture", "qualified-unpublished",
            ["qualify_root", "publish_root"], failed_stage="publish_root")
        unpublished_blob = RS.build_archive(root, summary)
        unpublished = RS.verify_archive(unpublished_blob)
        destination = Path(tmp) / "qualified-unpublished"
        RS.extract_verified_archive(unpublished_blob, destination)
        check("R70 qualified-unpublished root remains usable after pod deletion",
              unpublished["manifest"]["publication_requested"]
              and unpublished["manifest"]["status"] == "qualified-unpublished"
              and (destination / "dataset" / "fidelity-dataset.json").is_file()
              and (destination / "dataset-repeat"
                   / "fidelity-dataset.json").is_file())
        archive_bound_publication = FD._verify_publish_source_archive(
            unpublished_blob,
            hashlib.sha256(unpublished_blob).hexdigest(),
            len(unpublished_blob),
            destination / "dataset",
            destination / "receipts" / "root-qualification.json",
            destination / "job.json")
        check("R89 publication source binds exact archive and canonical tree",
              archive_bound_publication["archive_sha256"]
              == hashlib.sha256(unpublished_blob).hexdigest()
              and archive_bound_publication["canonical_dataset_records"]
              and archive_bound_publication["qualification_record"]["bytes"]
              > 0)
        qualification_path = (
            destination / "receipts" / "root-qualification.json")
        qualification_path.write_bytes(
            qualification_path.read_bytes() + b"\n")
        try:
            FD._verify_publish_source_archive(
                unpublished_blob,
                hashlib.sha256(unpublished_blob).hexdigest(),
                len(unpublished_blob),
                destination / "dataset", qualification_path,
                destination / "job.json")
        except FD.RootQualificationError:
            local_substitution_refused = True
        else:
            local_substitution_refused = False
        check("R90 post-extraction qualification substitution is refused",
              local_substitution_refused)
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        sparse = root / "receipts" / "large-payload.bin"
        with sparse.open("wb") as stream:
            stream.truncate(2 * 1024 * 1024)
        huge_log = root / "logs" / "huge.log"
        with huge_log.open("wb") as stream:
            stream.truncate(300 * 1024 * 1024)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        original_memory_cap = RS.MAX_IN_MEMORY_ARCHIVE_BYTES
        RS.MAX_IN_MEMORY_ARCHIVE_BYTES = 1024 * 1024
        try:
            _refused("R71 large bundle refuses the in-memory compatibility API",
                     lambda: RS.build_archive(root, summary))
            archive_path = Path(tmp) / "streamed.tar.gz"
            streamed = RS.write_archive(root, summary, archive_path)
            delivered_dir = Path(tmp) / "streamed-delivery"
            delivered_result = RS.deliver(
                root, RS.parse_sinks(["file:%s" % delivered_dir]),
                summary, lambda message: None)
        finally:
            RS.MAX_IN_MEMORY_ARCHIVE_BYTES = original_memory_cap
        verified = RS.verify_archive(
            archive_path, expected_sha256=streamed["sha256"],
            expected_bytes=streamed["bytes"])
        extracted = Path(tmp) / "streamed-extract"
        RS.extract_verified_archive(archive_path, extracted)
        huge_log_record = next(
            row for row in verified["manifest"]["files"]
            if row["path"] == "logs/huge.log")
        check("R72 sparse payload streams through write/verify/extract/delivery "
              "while huge logs remain bounded tails",
              verified["archive_sha256"] == streamed["sha256"]
              and any(result.get("scheme") == "file" and result.get("ok")
                      for result in delivered_result)
              and (delivered_dir / RS.ARCHIVE_NAME).is_file()
              and (extracted / "receipts"
                   / "large-payload.bin").stat().st_size == 2 * 1024 * 1024
              and huge_log_record["delivery"] == "tail"
              and huge_log_record["omitted_prefix_bytes"] > 0
              and (extracted / "logs" / "huge.log").stat().st_size
              == huge_log_record["bytes"]
              and huge_log_record["bytes"] <= RS.LOG_TAIL_BYTES + 128)

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(
            root, "measure", "failed", ["setup", "measure"],
            failed_stage="measure")
        failed_verified = RS.verify_archive(RS.build_archive(root, summary))
        check("R43 a failed bundle remains valid with job, logs and state evidence",
              failed_verified["manifest"]["status"] == "failed")

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, abandoned=True)
        summary = RS.build_summary(
            root, "measure", "abandoned", ["setup"],
            failed_stage="setup")
        abandoned = RS.verify_archive(RS.build_archive(root, summary))
        check("R44 an abandoned bundle binds its valid ABANDONED.json state",
              abandoned["manifest"]["status"] == "abandoned")

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, abandoned=True)
        (root / "ABANDONED.json").write_text("{", encoding="utf-8")
        summary = RS.build_summary(
            root, "measure", "abandoned", ["setup"],
            failed_stage="setup")
        _refused("R47 malformed ABANDONED.json is refused",
                 lambda: RS.build_archive(root, summary))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        summary = RS.build_summary(
            root, "measure", "failed", ["measure"], failed_stage="measure")
        _refused("R54 failed measure cannot use a root job",
                 lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(
            root, "capture", "failed", ["capture"], failed_stage="capture")
        _refused("R55 failed capture cannot use a quant job",
                 lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(
            root, "measure", "failed", ["measure"], failed_stage="measure")
        valid = RS.build_archive(root, summary)
        wrong_root = _run_root(Path(tmp) / "wrong-job", role="root")
        wrong_job = (wrong_root / "job.json").read_bytes()
        wrong_job_doc = json.loads(wrong_job.decode("utf-8"))
        state = {
            "schema": "result-state-v1", "verb": "measure", "role": "root",
            "status": "failed", "failed_stage": "measure",
            "stages": ["measure"], "publication_requested": False,
            "job_id_full": wrong_job_doc["job_id_full"],
            "measurement_receipt_sha256": None,
        }
        forged = _rewrite_consistent(
            valid,
            {"job.json": wrong_job,
             RS.RUN_STATE_NAME: (json.dumps(
                 state, indent=2, sort_keys=True) + "\n").encode("utf-8")},
            {"role": "root", "job_id_full": wrong_job_doc["job_id_full"]})
        _refused("R56 off-box verifier rejects a self-consistent failed "
                 "measure/root role mismatch",
                 lambda: RS.verify_archive(forged))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        valid = RS.build_archive(root, summary)
        tampered_job = json.loads(
            (root / "job.json").read_text(encoding="utf-8"))
        tampered_job["target"]["repo_id"] = "attacker/other-model"
        forged = _rewrite_consistent(
            valid,
            {"job.json": json.dumps(tampered_job).encode("utf-8")},
            {})
        _refused("R57 off-box verifier rejects job content whose stored "
                 "job_id_full was not recomputed",
                 lambda: RS.verify_archive(forged))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        receipt_path = root / "receipts" / "measurement-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifact"]["repository"] = "attacker/other-model"
        receipt_path.write_text(
            json.dumps(RS.seal(receipt)), encoding="utf-8")
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        _refused("R58 individually sealed measurement evidence from another "
                 "artifact cannot satisfy the job",
                 lambda: RS.build_archive(root, summary))
    for label, field, value in (
            ("run_count differs from job cold_runs", "run_count", 1),
            ("cold_start_per_run is false", "cold_start_per_run", False),
            ("identical_across_runs is false", "identical_across_runs", False)):
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(tmp)
            receipt_path = root / "receipts" / "measurement-receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["determinism"][field] = value
            receipt_path.write_text(
                json.dumps(RS.seal(receipt)), encoding="utf-8")
            summary = RS.build_summary(root, "measure", "ok", ["seal"])
            _refused("R61 completed quant determinism refused when %s" % label,
                     lambda: RS.build_archive(root, summary))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        (root / "reports" / "cold-run-2.json").unlink()
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        _refused("R79 completed quant requires both exact per-run report bytes",
                 lambda: RS.build_archive(root, summary))

    for label, block, field, value in (
            ("wrong KLD direction", "metric", "direction",
             "candidate_to_reference"),
            ("top-k position filter", "measurement_scope",
             "position_filter", "top-k")):
        with tempfile.TemporaryDirectory() as tmp:
            root = _run_root(tmp)
            receipt_path = root / "receipts" / "measurement-receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt[block][field] = value
            receipt_path.write_text(
                json.dumps(RS.seal(receipt)), encoding="utf-8")
            summary = RS.build_summary(root, "measure", "ok", ["seal"])
            _refused("R73 completed quant refuses %s" % label,
                     lambda: RS.build_archive(root, summary))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        receipt_path = root / "receipts" / "measurement-receipt.json"
        raw = receipt_path.read_text(encoding="utf-8")
        receipt_path.write_text(
            raw.replace('"value": 0.125', '"value": NaN', 1),
            encoding="utf-8")
        summary = RS.build_summary(root, "measure", "ok", ["seal"])
        _refused("R74 completed quant refuses non-finite KLD",
                 lambda: RS.build_archive(root, summary))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        job = json.loads((root / "job.json").read_text(encoding="utf-8"))
        contract = job["target"]["result_archive_contract"]
        _refused("R75 quant archive refuses one byte above job bound",
                 lambda: RS._enforce_quant_archive_caps(
                     job, 1,
                     contract["result_archive_max_uncompressed_bytes"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        summary = RS.build_summary(
            root, "measure", "completed-operational-failure",
            ["seal"], failed_stage="token_cleanup")
        preserved = RS.verify_archive(RS.build_archive(root, summary))
        check("R76 completed science survives a later operational failure",
              preserved["manifest"]["status"]
              == "completed-operational-failure"
              and preserved["manifest"]["measurement_receipt_sha256"]
              is not None)




    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        wrong_role = (root / "receipts" / "measurement-receipt.json").read_bytes()
        (root / "receipts" / "root-qualification.json").write_bytes(wrong_role)
        summary = RS.build_summary(root, "capture", "ok", ["qualify_root"])
        _refused("R45 a measurement receipt cannot satisfy the root role",
                 lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        qualification_path = root / "receipts" / "root-qualification.json"
        qualification = json.loads(
            qualification_path.read_text(encoding="utf-8"))
        qualification["canonical_job_sha256"] = "d" * 64
        qualification_path.write_text(
            json.dumps(RS.seal(qualification)), encoding="utf-8")
        summary = RS.build_summary(root, "capture", "ok", ["qualify_root"])
        _refused("R49 an individually sealed qualification from another job "
                 "is refused", lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root")
        valid_summary = RS.build_summary(
            root, "capture", "ok", ["qualify_root"])
        valid = RS.build_archive(root, valid_summary)
        job_path = root / "job.json"
        job = json.loads(job_path.read_text(encoding="utf-8"))
        job["execution_attempt"]["attempt_id"] = "2" * 24
        job = JC.finalize_job(job)
        changed_job_bytes = json.dumps(job).encode("utf-8")
        forged = _rewrite_consistent(
            valid, {"job.json": changed_job_bytes}, {})
        _refused("R60 off-box verifier rejects qualification from another "
                 "execution attempt with the same canonical job id",
                 lambda: RS.verify_archive(forged))
        job_path.write_bytes(changed_job_bytes)
        summary = RS.build_summary(root, "capture", "ok", ["qualify_root"])
        _refused("R59 qualification from the same science but a different "
                 "execution attempt is refused",
                 lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root", publish=True)
        publish_path = root / "receipts" / "publish-root.json"
        publication = json.loads(publish_path.read_text(encoding="utf-8"))
        publication["verified_after_publish"] = False
        publish_path.write_text(
            json.dumps(RS.seal(publication)), encoding="utf-8")
        summary = RS.build_summary(root, "capture", "ok", ["publish_root"])
        _refused("R48 a sealed publish receipt without v2 refetch proof is refused",
                 lambda: RS.build_archive(root, summary))

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp, role="root", publish=True)
        (root / "receipts" / "publish-root.json").unlink()
        summary = RS.build_summary(root, "capture", "ok", ["publish_root"])
        _refused("R46 publication requested without immutable publish/refetch "
                 "receipt is refused", lambda: RS.build_archive(root, summary))


    with tempfile.TemporaryDirectory() as tmp:
        root = _doctor_root(tmp)
        summary = RS.build_summary(root, "doctor", "ok", [])
        doctor = RS.verify_archive(RS.build_archive(root, summary))
        check("R50 doctor archives valid status evidence without job.json",
              doctor["manifest"]["verb"] == "doctor"
              and doctor["manifest"]["role"] == "doctor")

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        (root / "receipts" / "measurement-receipt.json").unlink()
        summary = RS.build_summary(root, "stage", "ok", ["setup"])
        staged = RS.verify_archive(RS.build_archive(root, summary))
        check("R51 completed stage needs its named log/done but no final receipt",
              staged["manifest"]["verb"] == "stage")

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        (root / "receipts" / "measurement-receipt.json").unlink()
        summary = RS.build_summary(
            root, "stage", "failed", ["capture"], failed_stage="capture")
        failed_stage = RS.verify_archive(RS.build_archive(root, summary))
        check("R52 failed stage needs named failed-stage log but no done/final receipt",
              failed_stage["manifest"]["status"] == "failed")

    with tempfile.TemporaryDirectory() as tmp:
        root = _run_root(tmp)
        (root / "receipts" / "measurement-receipt.json").unlink()
        (root / "receipts" / "done" / "setup.done").unlink()
        summary = RS.build_summary(root, "stage", "ok", ["setup"])
        _refused("R53 completed stage without its named done evidence is refused",
                 lambda: RS.build_archive(root, summary))

    resource_job = {
        "environment": {"gpu": "NVIDIA H100 80GB HBM3"},
        "resource_requirements": {
            "workspace_available_bytes_minimum": 4096,
            "container_available_bytes_minimum": 2048,
            "min_vcpu_count": 8,
            "min_memory_gb": 32,
            "expected_vram_bytes": 80 * 1024 ** 3,
        },
        "execution_attempt": {"kind": "runpod-ssh"},
    }
    attestation = {
        "schema": RS.RUNPOD_ATTESTATION_SCHEMA,
        "provider": "runpod",
        "provider_id": "pod-123",
        "observed_at_utc": "2026-01-01T00:00:01Z",
        "clock": {
            "controller_send_epoch": 1767225600.0,
            "controller_send_utc": "2026-01-01T00:00:00Z",
            "controller_receive_epoch": 1767225601.0,
            "controller_receive_utc": "2026-01-01T00:00:01Z",
            "round_trip_seconds": 1.0,
            "remote_time_epoch": 1767225601,
            "remote_time_utc": "2026-01-01T00:00:01Z",
            "clock_skew_seconds": 0.5,
            "allowed_skew_seconds": 31.0,
            "within_bound": True,
        },
        "expected": {
            "expected_vram_bytes": 80 * 1024 ** 3,
            "min_vcpu": 8,
            "min_ram_gb": 32,
            "volume_gb": 100,
            "container_disk_gb": 20,
            "workspace_available_bytes_minimum": 4096,
            "container_available_bytes_minimum": 2048,
            "gpu_model": "NVIDIA H100 80GB HBM3",
        },
        "observed": {
            "remote_time_epoch": 1767225601,
            "remote_time_utc": "2026-01-01T00:00:01Z",
            "filesystems": {
                "workspace": {"available_bytes": 8192},
                "container": {"available_bytes": 4096},
            },
        },
        "transport_error": None,
        "checks": {
            "container_available_bytes": True,
            "gpu_model": True,
            "remote_clock": True,
            "storage": True,
            "workspace_available_bytes": True,
        },
        "failures": [],
        "ok": True,
    }
    attestation["attestation_sha256"] = hashlib.sha256(
        json.dumps(attestation, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    RS._validate_runpod_attestation(resource_job, attestation)
    check("R77 sealed live RunPod attestation binds exact job resources", True)
    provider_log_line = (
        "256 SHA256:%s fixture (ED25519)" % ("A" * 43))
    host_key_proof = {
        "schema": RS.RUNPOD_HOST_KEY_PROOF_SCHEMA,
        "provider": "runpod",
        "provider_id": "pod-123",
        "verified_at_utc": "2026-01-01T00:00:00Z",
        "verification_source": "runpod-authenticated-v2-container-log",
        "provider_log_endpoint_origin": "https://api.runpod.io",
        "provider_log_source": "container",
        "provider_log_tail": 5000,
        "provider_log_observed_at_utc": "2026-01-01T00:00:00Z",
        "provider_log_line": provider_log_line,
        "provider_log_line_sha256": hashlib.sha256(
            provider_log_line.encode("utf-8")).hexdigest(),
        "provider_log_fingerprint": "SHA256:" + "A" * 43,
        "algorithm": "ssh-ed25519",
        "fingerprint": "SHA256:" + "A" * 43,
        "host": "198.51.100.7",
        "port": 22022,
        "known_hosts_sha256": "9" * 64,
        "proof_sha256": "",
    }
    host_key_proof["proof_sha256"] = hashlib.sha256(
        RS.canonical_json(host_key_proof).encode("utf-8")).hexdigest()
    RS._validate_runpod_host_key_proof(
        resource_job, host_key_proof, attestation)
    check("R81 sealed SSH proof binds authenticated provider logs to the "
          "exact pod key", True)
    bad_host_key = dict(host_key_proof)
    bad_host_key["verification_source"] = "network-keyscan-tofu"
    bad_host_key["proof_sha256"] = ""
    bad_host_key["proof_sha256"] = hashlib.sha256(
        RS.canonical_json(bad_host_key).encode("utf-8")).hexdigest()
    _refused("R82 self-sealed unauthenticated first-hop proof is refused",
             lambda: RS._validate_runpod_host_key_proof(
                 resource_job, bad_host_key, attestation))
    mismatched_log_key = dict(host_key_proof)
    mismatched_log_key["provider_log_fingerprint"] = "SHA256:" + "B" * 43
    mismatched_log_key["proof_sha256"] = ""
    mismatched_log_key["proof_sha256"] = hashlib.sha256(
        RS.canonical_json(mismatched_log_key).encode("utf-8")).hexdigest()
    _refused("R93 self-sealed provider-log/network key mismatch is refused",
             lambda: RS._validate_runpod_host_key_proof(
                 resource_job, mismatched_log_key, attestation))
    bad_log_line_hash = dict(host_key_proof)
    bad_log_line_hash["provider_log_line_sha256"] = "7" * 64
    bad_log_line_hash["proof_sha256"] = ""
    bad_log_line_hash["proof_sha256"] = hashlib.sha256(
        RS.canonical_json(bad_log_line_hash).encode("utf-8")).hexdigest()
    _refused("R94 self-sealed provider-log line/hash mismatch is refused",
             lambda: RS._validate_runpod_host_key_proof(
                 resource_job, bad_log_line_hash, attestation))
    bad_attestation = dict(attestation)
    bad_attestation["checks"] = dict(attestation["checks"])
    bad_attestation["checks"]["storage"] = False
    bad_attestation["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in bad_attestation.items()
             if key != "attestation_sha256"},
            sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _refused("R78 sealed live RunPod attestation with a failed check is refused",
             lambda: RS._validate_runpod_attestation(
                 resource_job, bad_attestation))
    stale_clock = json.loads(json.dumps(attestation))
    stale_clock["clock"]["remote_time_epoch"] -= 3600
    stale_clock["clock"]["remote_time_utc"] = "2025-12-31T23:00:01Z"
    stale_clock["observed"]["remote_time_epoch"] -= 3600
    stale_clock["observed"]["remote_time_utc"] = "2025-12-31T23:00:01Z"
    stale_clock["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in stale_clock.items()
             if key != "attestation_sha256"},
            sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _refused("R79 sealed RunPod attestation with stale remote clock is refused",
             lambda: RS._validate_runpod_attestation(
                 resource_job, stale_clock))
    low_free = json.loads(json.dumps(attestation))
    low_free["observed"]["filesystems"]["workspace"]["available_bytes"] = 4095
    low_free["attestation_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in low_free.items()
             if key != "attestation_sha256"},
            sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _refused("R80 sealed RunPod attestation with low free space is refused",
             lambda: RS._validate_runpod_attestation(
                 resource_job, low_free))


def rung_wired():
    print("[T26.6] the entrypoint actually uses it, and the image ships it")
    entry = (HERE / "container_entry.py").read_text(encoding="utf-8")
    check("R24 --result-sink is on the common parser", "--result-sink" in entry)
    # Anchored on the LAST delivery site, not the first: `doctor` also
    # delivers, and it is defined earlier in main(), so a naive index() reads
    # the wrong one and both rungs go green for the wrong reason.
    check("R25 the stage run's delivery is in the finally, so a FAILED run "
          "still reports",
          "RS.deliver" in entry
          and entry.rindex("RS.deliver") > entry.rindex("finally:"))
    cleanup = entry.rindex("clear_stale_token(fs_root, con)")
    delivery = entry.rindex("RS.deliver")
    check("R26 the token is shredded BEFORE any result leaves the box",
          entry.rindex("finally:") < cleanup < delivery,
          "final cleanup/delivery order differs")
    check("R27b doctor takes the common flags, so 'rent a pod and check the "
          "image sees the GPU' has an answer you can retrieve",
          '"doctor", help=' in entry and "add_common(d)" in entry)
    bundle = (HERE / "BUNDLE.txt").read_text(encoding="utf-8").split()
    check("R27 bin/fidelity/resultsink.py is bundled -- an unbundled module is "
          "an image that dies at the last line of a paid run",
          "bin/fidelity/resultsink.py" in bundle)


def main():
    print("== T26 result sinks: getting the answer off the box ==")
    for rung in (rung_parse, rung_content, rung_logs, rung_http, rung_cap,
                 rung_archive, rung_wired):
        rung()
    print("\nT26: %d passed, %d failed" % (PASS, FAIL))
    for f in FAILED:
        print("  - %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
