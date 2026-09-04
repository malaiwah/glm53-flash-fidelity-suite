#!/usr/bin/env python3
"""T6 -- the fidelity-dataset format, seal and refusal matrix.

Every fixture is built here, in a temp dir, from pure JSON plus hand-written
safetensors bytes.  No network, no GPU, no torch: this runs on the system
python3 (3.9) with numpy and nothing else, which is the same promise
`registry/Makefile` makes.

Each case names the spec rule it exercises, so a failure points at a sentence in
docs/FIDELITY-DATASET-SPEC.md rather than at an opinion.

    F1-F15   format and seal          spec 5
    P1-P9    panel binding            spec 7
    H1-H11   head identity            spec 8
    L1-L5    lane and stack           spec 9, 10.1
    C1-C4    coverage                 spec 6.3
    X1-X2    lossy / dtype            spec 4.1, 12.5 D-8
    SV1-SV2  scope vocabulary         SCOPE-VOCAB (the registry's numeric_format enum)
    R1-R5    real published artifacts (metadata only)

Exit 0 = all pass.
"""

from __future__ import annotations
import argparse

import json
from pathlib import Path
import hashlib
import os
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fidelity_dataset as CLI  # noqa: E402
from fidelity import common, jobcontract  # noqa: E402
from fidelity import dsformat as F  # noqa: E402
from fidelity import dsadapt, dscompare, dsmanifest, dsvalidate  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def expect_errors(name, report, code=None, rule=None):
    codes = [e["code"] for e in report.errors]
    rules = [e["rule"] for e in report.errors]
    ok = bool(report.errors)
    if code:
        ok = ok and code in codes
    if rule:
        ok = ok and any(r.startswith(rule) for r in rules)
    check(name, ok, "codes=%s rules=%s" % (codes[:4], rules[:4]))


def expect_refusal(name, fn, code=None, gate=None):
    try:
        fn()
    except dscompare.Refusal as exc:
        ok = (code is None or exc.code == code) and (gate is None or exc.gate == gate)
        check(name, ok, "got code=%s gate=%s" % (exc.code, exc.gate))
        return
    except F.FormatError as exc:
        check(name, code is None or exc.code == code, "FormatError %s" % exc.code)
        return
    check(name, False, "no refusal raised")


# ---------------------------------------------------------------------------
# safetensors, by hand
# ---------------------------------------------------------------------------


def st_bytes(tensors, metadata=None):
    """tensors: {name: (dtype_str, shape, raw_bytes)}"""
    header = {}
    blob = b""
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + blob


def bf16_bytes(array):
    """Truncate fp32 to bf16 (round toward zero) and return the raw LE bytes."""
    wide = np.ascontiguousarray(array, dtype="<f4").view("<u4")
    return (wide >> 16).astype("<u2").tobytes()


def bf16_roundtrip(array):
    """The exact fp32 values a bf16 store/load round-trip yields."""
    raw = np.frombuffer(bf16_bytes(array), dtype="<u2").astype(np.uint32)
    raw = raw << 16
    return raw.view(np.float32).reshape(np.asarray(array).shape)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

VOCAB = 16
HIDDEN = 4
ROWS = 3
RECORDS = 2


def build_dataset(root, *, role="root", form="hidden", lane="sealed-ep8", seed=1,
                  head_seed=7, head_present=True, head_content=None,
                  vocab=VOCAB, hidden=HIDDEN, rows=ROWS, records=RECORDS,
                  capture_indices=None,
                  token_offset=0, score_from=0, declared_records=None,
                  shard_of=None, subset_detail=None, lossy_codec=None,
                  dtype_lossless=True, stack="stack-a", lane_identity="lane-a",
                  mask_salt=0, model_revision="a" * 40, checkpoint_identity="b" * 64,
                  quantized=False, structural_status="sealed",
                  head_applied_in_capture=None, final_norm_applied_at_replay=False,
                  tokenizer=None, emit_k3_compat=False, run_name=None,
                  cold_run=None, dataset_repository=None,
                  weights_repository="selftest/weights",
                  qualification_contract=False,
                  panel_receipt_sha256=None,
                  resolved_panel_binding=None,
                  panel_binding_file_sha256="2" * 64):
    """Build a complete, sealed, conformant dataset.  Every knob is a test axis."""
    panel_receipt_raw = None
    panel_receipt_file_sha256 = None
    if qualification_contract:
        receipt_doc = common.seal({
            "schema": "fidelity.selftest-panel-receipt.v1",
            "panel_id": "panel--selftest.tiny",
        })
        panel_receipt_raw = (
            common.canonical_json(receipt_doc) + "\n").encode("utf-8")
        generated_receipt_sha256 = receipt_doc["receipt_sha256"]
        if (panel_receipt_sha256 is not None
                and panel_receipt_sha256 != generated_receipt_sha256):
            raise ValueError("qualification fixture receipt identity mismatch")
        panel_receipt_sha256 = generated_receipt_sha256
        panel_receipt_file_sha256 = hashlib.sha256(
            panel_receipt_raw).hexdigest()
        tokenizer = tokenizer or {
            "id": "selftest-tokenizer",
            "repository": weights_repository,
            "revision": model_revision,
            "vocab_size": vocab,
            "files": [{
                "path": "tokenizer.json",
                "bytes": 17,
                "sha256": "4" * 64,
            }],
            "identity_sha256": "3" * 64,
            "add_special_tokens": False,
            "chat_template_applied": False,
        }
    writer = dsmanifest.DatasetWriter(root)
    if panel_receipt_raw is not None:
        writer.add_file("panel/panel-receipt.json", panel_receipt_raw)
    rng = np.random.RandomState(seed)

    # -- head ---------------------------------------------------------------
    head_rng = np.random.RandomState(head_seed)
    head = head_rng.normal(size=(vocab, hidden)).astype(np.float32)
    head_payload = st_bytes({"lm_head.weight": ("BF16", [vocab, hidden], bf16_bytes(head))})
    head_rel = writer.add_head_payload(head_payload) if head_present else None
    head_full = os.path.join(root, head_rel) if head_rel else None
    head_content_digest = head_content
    if head_content_digest is None and head_full:
        head_content_digest = F.tensor_content_sha256(head_full, "lm_head.weight")

    tensor_key = F.TENSOR_KEY_HIDDEN if form == "hidden" else F.TENSOR_KEY_LOGIT
    width = hidden if form == "hidden" else vocab

    panel_records = []
    capture_records = []
    for index in range(records):
        ids = [token_offset + index * 100 + i for i in range(rows + 1)]
        token_rel = writer.add_token_file(index, ids)
        mask = np.ones(rows + 1, dtype=np.int64)
        mask[0] = 1 + mask_salt
        mask_rel, mask_sha = writer.add_mask_file(index, mask.tobytes())
        panel_records.append(dsmanifest.panel_record(
            index=index, token_file=token_rel, token_ids=ids,
            prediction_positions=rows, window_id="final-%04d" % index,
            attention_mask_file=mask_rel, attention_mask_sha256=mask_sha,
            role="final", domain="axis1_general", document_id="doc-%d" % index,
            allocation_stratum="encyclopedic", source_cluster_id="cluster-%d" % index))
        if capture_indices is not None and index not in capture_indices:
            continue
        values = rng.normal(size=(rows, width)).astype(np.float32)
        if form == "hidden":
            payload = st_bytes({tensor_key: ("BF16", [rows, width], bf16_bytes(values))},
                               metadata={"cold_run": str(seed)})
            dtype = "BF16"
        else:
            payload = st_bytes({tensor_key: ("F32", [rows, width],
                                             np.ascontiguousarray(values, "<f4").tobytes())},
                               metadata={"cold_run": str(seed)})
            dtype = "F32"
        rel = writer.add_capture_tensor(index, payload, form)
        capture_records.append(dsmanifest.tensor_record(
            index=index, filename=os.path.basename(rel),
            abs_path=os.path.join(root, rel), key=tensor_key, dtype=dtype,
            shape=[rows, width], scored_rows=rows,
            token_ids_json_sha256=F.token_ids_json_sha256(ids),
            token_ids_sha256_legacy=F.token_ids_json_sha256_legacy(ids),
            attention_mask_sha256=mask_sha, window_id="final-%04d" % index,
            role="final", domain="axis1_general", document_id="doc-%d" % index,
            allocation_stratum="encyclopedic", source_cluster_id="cluster-%d" % index))

    panel_doc = dsmanifest.panel_binding(
        panel_id="panel--selftest.tiny", name="selftest tiny panel",
        records=panel_records, context_length=rows + 1,
        tokenizer=tokenizer or {"id": "selftest", "repository": None, "revision": None,
                                "vocab_size": vocab, "add_special_tokens": False,
                                "chat_template_applied": False},
        scoring_window={"score_from": score_from, "windowed": score_from > 0,
                        "min_left_context_tokens": 1, "dropped_positions_total": 0,
                        "policy": "selftest"},
        panel_receipt_sha256=panel_receipt_sha256)

    coverage = dsmanifest.coverage_block(
        capture_records, declared_records if declared_records is not None else records,
        shard_of=shard_of, subset_detail=subset_detail)

    process_label = run_name or ("selftest-%s" % role)
    capture_doc = dsmanifest.capture_manifest(
        run_name=process_label, form=form,
        semantic_point=("after_final_rmsnorm_before_lm_head" if form == "hidden"
                        else "lm_head_output_before_sampling"),
        tensor_key=tensor_key, dtype=("BF16" if form == "hidden" else "F32"),
        dtype_lossless=dtype_lossless, vocab_size=vocab, context_length=rows + 1,
        records=capture_records, hidden_width=hidden if form == "hidden" else None,
        coverage=coverage)

    applied_in_capture = (form == "logit") if head_applied_in_capture is None \
        else head_applied_in_capture
    head_doc = dsmanifest.head_identity(
        present=head_present, tensor_key="lm_head.weight", shape=[vocab, hidden],
        dtype="BF16", file_sha256=F.sha256_file(head_full) if head_full else None,
        tensor_content_sha256=head_content_digest, quantized=quantized,
        source="native" if not quantized else "artifact_dequantized",
        applied_in_capture=applied_in_capture, file=head_rel, bits=16 if not quantized else 6,
        final_norm={"file": None, "tensor_key": "model.norm.weight", "shape": [hidden],
                    "dtype": "BF16", "file_sha256": None, "tensor_content_sha256": None,
                    "applied_in_capture": True,
                    "applied_at_replay": final_norm_applied_at_replay})

    if qualification_contract and resolved_panel_binding is None:
        resolved_panel_binding = {
            "panel": {
                "id": panel_doc["panel_id"],
                "suite_token_hash_sha256":
                    panel_doc["suite_token_hash_sha256"],
            },
            "receipt": {
                "file": "panel.receipt.json",
                "declared_receipt_sha256": panel_receipt_sha256,
                "receipt_file_sha256": panel_receipt_file_sha256,
                "bytes": len(panel_receipt_raw),
                "receipt_seal_mode": "self-blank",
            },
            "tokenizer": {
                "repository": panel_doc["tokenizer"]["repository"],
                "revision": panel_doc["tokenizer"]["revision"],
                "vocab_size": panel_doc["tokenizer"]["vocab_size"],
                "files": panel_doc["tokenizer"]["files"],
                "files_verified": True,
                "identity_sha256": panel_doc["tokenizer"]["identity_sha256"],
                "id": panel_doc["tokenizer"]["id"],
            },
        }
    runtime_engine = "transformers-eager" if qualification_contract else stack
    fingerprint = {
        "schema": "malaiwah.stack-fingerprint.v1",
        "engine": runtime_engine,
    }
    if qualification_contract:
        fingerprint["device"] = "cuda"
    capture_tool = {
        "file": ("engines/tools/hf_capture.py" if qualification_contract
                 else "bin/fidelity_dataset.py"),
        "sha256": F.sha256_hex("tool"),
        "wraps": ["engines/tools/hidden_replay.py"],
        "mechanism": "selftest fixture",
    }
    if qualification_contract:
        capture_tool.update({
            "schedule": "layer-outer",
            "resolved_panel_binding": {
                "binding_file": "selftest.binding.json",
                "binding_file_sha256": panel_binding_file_sha256,
                "binding": resolved_panel_binding,
            },
        })
        allow_names = ["model.unused"]
        allow_names_sha = common.sha256_hex(json.dumps(
            allow_names, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False))
        capture_tool["unexpected_tensor_allowlist"] = {
            "artifact_sha256": "5" * 64,
            "canonical_sorted_names_sha256": allow_names_sha,
            "expected_keys": allow_names,
            "expected_count": 1,
            "observed_keys": allow_names,
            "observed_count": 1,
            "duplicate_observed_keys": [],
            "missing_keys": [],
            "extra_keys": [],
            "exact_match": True,
        }
    runtime_doc = dsmanifest.capture_runtime(
        lane=lane, stack_fingerprint=fingerprint,
        stack_fingerprint_sha256=F.sha256_hex(stack),
        lane_identity_sha256=F.sha256_hex(lane_identity),
        weights={"repository": weights_repository, "revision": model_revision,
                 "model_revision": model_revision,
                 "checkpoint_identity_sha256": checkpoint_identity},
        runtime_environment={"cold_run": cold_run or process_label},
        source_files={"engines/tools/stream_score.py": F.sha256_hex("selftest")},
        capture_tool=capture_tool)

    scope = (dsmanifest.native_scope() if not quantized else dsmanifest.scope_block(
        [{"tensor_class": name, "treatment": "quantized", "format": "exl3-mcg",
          "bits_per_weight": 6, "layer_range": None}
         for name in ("moe.experts",)]
        + [{"tensor_class": name, "treatment": "native", "format": "bf16",
            "bits_per_weight": 16, "layer_range": None}
           for name in ("embed_tokens", "attn.qkv", "attn.o", "mlp.gate", "mlp.up",
                        "mlp.down", "norm", "lm_head")],
        head_policy="native", kv_cache_dtype="bf16", policy="mixed"))

    manifest = dsmanifest.top_manifest(
        dataset={"id": "fidelity--selftest.%s.%s" % (role, form),
                 "name": "selftest %s %s" % (role, form), "role": role,
                 "structural_status": structural_status, "qualification": None,
                 "author": {"name": "selftest", "role": "capture-author",
                            "handle": None, "url": None,
                            "is_registry_maintainer": False},
                 "license": "mit", "repository": dataset_repository, "revision": None,
                 "base_capture": None},
        weights={"repository": weights_repository, "revision": model_revision,
                 "model_revision": model_revision, "quantized": quantized,
                 "checkpoint_identity_sha256": checkpoint_identity,
                 "config_sha256": None, "index_sha256": None, "artifact_ref": None,
                 "model_ref": None, "codec": None, "declared_bits": None,
                 "declared_head_bits": None},
        scope=scope,
        panel={"panel_id": "panel--selftest.tiny", "panel_file": "panel/panel.json",
               "panel_file_sha256": "0" * 64,
               "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
               "panel_token_sha256_legacy": panel_doc["panel_token_sha256_legacy"],
               "panel_receipt_sha256": panel_receipt_sha256,
               "panel_receipt_file": (
                   "panel/panel-receipt.json"
                   if qualification_contract else None),
               "repository": None, "revision": None,
               "contexts": len(panel_records), "context_length": rows + 1,
               "scored_positions_total": rows * len(panel_records),
               "scoring_window": panel_doc["scoring_window"],
               "tokenizer": panel_doc["tokenizer"], "remap_file": None,
               "contamination": panel_doc["contamination"]},
        capture={"manifest_file": "capture/manifest.json",
                 "manifest_file_sha256": "0" * 64,
                 "capture_content_digest": capture_doc["capture_content_digest"],
                 "form": form, "semantic_point": capture_doc["semantic_point"],
                 "tensor_key": tensor_key, "dtype": capture_doc["dtype"],
                 "dtype_lossless": dtype_lossless,
                 "hidden_width": hidden if form == "hidden" else None,
                 "vocab_size": vocab, "head_separable": True,
                 "head_not_separable_reason": None,
                 "records_count": len(capture_records),
                 "scored_rows_total": rows * len(capture_records),
                 "total_size_bytes": capture_doc["total_size_bytes"],
                 "lossy_codec": lossy_codec},
        head={"present": head_present, "file": head_rel, "head_json": "head/head.json",
              "tensor_key": "lm_head.weight", "compat_tensor_key": "weight",
              "shape": [vocab, hidden], "dtype": "BF16", "bias": None,
              "file_sha256": head_doc["file_sha256"],
              "raw_tensor_sha256": head_content_digest,
              "tensor_content_sha256": head_content_digest,
              "quantized": quantized, "bits": 16 if not quantized else 6,
              "source": head_doc["source"],
              "applied_in_capture": applied_in_capture,
              "final_norm": head_doc["final_norm"], "equality_receipt": None},
        runtime={"file": "runtime/capture-runtime.json", "file_sha256": "0" * 64,
                 "lane": lane, "lane_inferred": False,
                 "lane_identity_sha256": runtime_doc["lane_identity_sha256"],
                 "stack_fingerprint_sha256": runtime_doc["stack_fingerprint_sha256"],
                 "backend_identity_sha256": None, "runtime_reader_sha256": None,
                 "source": "native"},
        determinism={"run_count": 1 if role == "root" else 2,
                     "cold_start_per_run": True,
                     "evidence_kind": ("hidden_state_tensor_sha256" if form == "hidden"
                                       else "logits_tensor_sha256"),
                     "evidence_hashes": [capture_doc["capture_content_digest"]],
                     "distinct_evidence_hash_count": 1,
                     "identical_across_runs": None if role == "root" else True,
                     "repeats": [], "repeat_noise": None,
                     "note": ("one independent cold capture" if role == "root"
                              else "selftest fixture")},
        coverage=coverage,
        disclosures=([{"code": "no_known_deviations", "severity": "info",
                       "affects_comparability": False, "detail": "selftest fixture"}]
                     + ([{"code": "reduced_run_count", "severity": "caveat",
                          "affects_comparability": False,
                          "detail": "one independent root capture; exact reproduction "
                                    "is established by the outer comparison"}]
                        if role == "root" else [])))

    if emit_k3_compat:
        from fidelity import k3compat                                # noqa: WPS433

        manifest["interop"].update(k3compat.emit(
            writer, panel_doc=panel_doc, capture_doc=capture_doc,
            manifest_capture=manifest["capture"], head_relpath=head_rel,
            dataset_name="selftest fidelity dataset"))
    writer.add_readme("---\nlicense: mit\n---\n\n# selftest fidelity dataset\n")
    report = dsvalidate.Report(root)
    report.ok("pre-seal")
    return writer.finish(manifest, panel_doc, capture_doc, head_doc, runtime_doc,
                         validation_report=report.to_dict())


def reseal(root):
    """Recompute checksums.txt and the manifest seal after an edit."""
    manifest = F.read_json(os.path.join(root, F.MANIFEST_NAME))
    return dsmanifest.finalize(root, manifest)


# ---------------------------------------------------------------------------
# F -- format and seal
# ---------------------------------------------------------------------------


def section_format(tmp):
    print("\n== F: format and seal (spec 5) ==")
    root = os.path.join(tmp, "f-base")
    manifest = build_dataset(root)
    report = dsvalidate.validate_dataset(root, verify_tensors=True)
    check("F1  round-trip: build -> seal -> verify", report.passed,
          json.dumps(report.errors[:3]))

    # F2 flip one byte in a capture tensor
    root2 = os.path.join(tmp, "f2")
    shutil.copytree(root, root2)
    victim = os.path.join(root2, "capture/hidden_0000.safetensors")
    with open(victim, "r+b") as handle:
        handle.seek(os.path.getsize(victim) - 1)
        last = handle.read(1)
        handle.seek(os.path.getsize(victim) - 1)
        handle.write(bytes([last[0] ^ 0xFF]))
    expect_errors("F2  flipped tensor byte -> refused",
                  dsvalidate.validate_dataset(root2, verify_tensors=True))

    # F3 flip a character in checksums.txt
    root3 = os.path.join(tmp, "f3")
    shutil.copytree(root, root3)
    path = os.path.join(root3, F.CHECKSUMS_NAME)
    text = open(path).read()
    open(path, "w").write(("b" if text[0] != "b" else "c") + text[1:])
    expect_errors("F3  edited checksums.txt -> seal_failed",
                  dsvalidate.validate_dataset(root3), code="seal_failed")

    # F4 re-serialize the manifest with different key order
    root4 = os.path.join(tmp, "f4")
    shutil.copytree(root, root4)
    doc = F.read_json(os.path.join(root4, F.MANIFEST_NAME))
    with open(os.path.join(root4, F.MANIFEST_NAME), "w") as handle:
        json.dump(doc, handle, indent=4, sort_keys=False)
    check("F4  reordered manifest keys -> seal still verifies",
          dsvalidate.validate_dataset(root4).passed)

    # F5 unknown top-level key (additive rule 1.3)
    root5 = os.path.join(tmp, "f5")
    shutil.copytree(root, root5)
    doc = F.read_json(os.path.join(root5, F.MANIFEST_NAME))
    doc["x_future_extension"] = {"invented": "by v1.1"}
    F.write_json(os.path.join(root5, F.MANIFEST_NAME), F.seal_manifest(doc))
    check("F5  unknown top-level key -> accepted (additive rule)",
          dsvalidate.validate_dataset(root5).passed)

    # F6 extra file not in checksums.txt
    root6 = os.path.join(tmp, "f6")
    shutil.copytree(root, root6)
    open(os.path.join(root6, "capture/stowaway.safetensors"), "wb").write(b"x")
    expect_errors("F6  unlisted file -> refused",
                  dsvalidate.validate_dataset(root6), code="unlisted_file")

    # F7 delete a listed file
    root7 = os.path.join(tmp, "f7")
    shutil.copytree(root, root7)
    os.remove(os.path.join(root7, "capture/hidden_0001.safetensors"))
    expect_errors("F7  missing listed file -> refused",
                  dsvalidate.validate_dataset(root7), code="missing_file")

    # F8 absolute path in a record
    root8 = os.path.join(tmp, "f8")
    shutil.copytree(root, root8)
    cm = F.read_json(os.path.join(root8, "capture/manifest.json"))
    cm["records"][0]["file"] = "/etc/passwd"
    F.write_json(os.path.join(root8, "capture/manifest.json"), F.seal_receipt(cm))
    reseal(root8)
    expect_errors("F8  absolute path in a record -> path_escape",
                  dsvalidate.validate_dataset(root8), code="path_escape")

    # F9 `..` escaping the root
    root9 = os.path.join(tmp, "f9")
    shutil.copytree(root, root9)
    cm = F.read_json(os.path.join(root9, "capture/manifest.json"))
    cm["records"][0]["file"] = "../../../etc/passwd"
    F.write_json(os.path.join(root9, "capture/manifest.json"), F.seal_receipt(cm))
    reseal(root9)
    expect_errors("F9  '..' escaping the root -> path_escape",
                  dsvalidate.validate_dataset(root9), code="path_escape")

    # F10 a symlink in the tree
    root10 = os.path.join(tmp, "f10")
    shutil.copytree(root, root10)
    os.symlink("/etc/hosts", os.path.join(root10, "capture/link.safetensors"))
    expect_errors("F10 symlink in the tree -> refused (PATH-4)",
                  dsvalidate.validate_dataset(root10), code="symlink")

    # F11 compat/ may use `..`
    root11 = os.path.join(tmp, "f11")
    shutil.copytree(root, root11)
    F.write_json(os.path.join(root11, "compat/reference-hidden/manifest.json"),
                 {"contexts": [{"context_index": 0,
                                "file": "../../capture/hidden_0000.safetensors",
                                "key": "hidden_states"}]})
    reseal(root11)
    check("F11 compat/ '..' -> permitted (PATH-3)",
          dsvalidate.validate_dataset(root11).passed,
          json.dumps(dsvalidate.validate_dataset(root11).errors[:2]))

    # F12 old seal fails, new seal passes
    root12 = os.path.join(tmp, "f12")
    shutil.copytree(root, root12)
    doc = F.read_json(os.path.join(root12, F.MANIFEST_NAME))
    doc["dataset"]["name"] = "tampered"
    F.write_json(os.path.join(root12, F.MANIFEST_NAME), doc)
    before = dsvalidate.validate_dataset(root12)
    F.write_json(os.path.join(root12, F.MANIFEST_NAME), F.seal_manifest(doc))
    after = dsvalidate.validate_dataset(root12)
    check("F12 edited manifest: old seal fails, resealed passes",
          (not before.passed) and after.passed)

    # F13/F14 capture_content_digest
    cm = F.read_json(os.path.join(root, "capture/manifest.json"))
    forward = F.capture_content_digest(cm["records"])
    backward = F.capture_content_digest(list(reversed(cm["records"])))
    check("F13 capture_content_digest is order-independent", forward == backward)
    mutated = json.loads(json.dumps(cm["records"]))
    mutated[0]["tensor_content_sha256"] = "0" * 64
    check("F14 capture_content_digest changes with content",
          F.capture_content_digest(mutated) != forward)

    # F15 metadata-only rewrite (DET-D2)
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    a = st_bytes({"hidden_states": ("BF16", [2, 3], bf16_bytes(values))},
                 metadata={"cold_run": "1"})
    b = st_bytes({"hidden_states": ("BF16", [2, 3], bf16_bytes(values))},
                 metadata={"cold_run": "2", "checkpoint_identity_sha256": "x" * 64})
    pa, pb = os.path.join(tmp, "m_a.st"), os.path.join(tmp, "m_b.st")
    open(pa, "wb").write(a)
    open(pb, "wb").write(b)
    check("F15 metadata rewrite: file digest differs, payload+content do NOT (DET-D2)",
          F.sha256_file(pa) != F.sha256_file(pb)
          and F.payload_sha256(pa) == F.payload_sha256(pb)
          and F.tensor_content_sha256(pa, "hidden_states")
          == F.tensor_content_sha256(pb, "hidden_states"))
    return root


# ---------------------------------------------------------------------------
# P -- panel binding
# ---------------------------------------------------------------------------


def section_panel(tmp, base):
    print("\n== P: panel binding (spec 7) ==")
    twin = os.path.join(tmp, "p-twin")
    build_dataset(twin, seed=1)
    gates, findings = dscompare.run_gates(
        dscompare.load_dataset(base), dscompare.load_dataset(twin), {})
    check("P1  matching panels -> panel gate passes", gates["panel"]["passed"])

    other = os.path.join(tmp, "p2")
    build_dataset(other, token_offset=9000)
    expect_refusal("P2  different suite_token_hash_sha256 -> panel_mismatch",
                   lambda: dscompare.run_gates(dscompare.load_dataset(base),
                                               dscompare.load_dataset(other), {}),
                   code="panel_mismatch", gate="panel")

    # P3 same aggregate, one record's token digest edited in the CAPTURE manifest
    root3 = os.path.join(tmp, "p3")
    shutil.copytree(base, root3)
    cm = F.read_json(os.path.join(root3, "capture/manifest.json"))
    cm["records"][0]["token_ids_json_sha256"] = "f" * 64
    F.write_json(os.path.join(root3, "capture/manifest.json"), F.seal_receipt(cm))
    reseal(root3)
    expect_errors("P3  record token digest differs from panel -> BIND-2",
                  dsvalidate.validate_dataset(root3), code="panel_binding")

    # P4 attention mask digest differs
    mask_variant = os.path.join(tmp, "p4")
    build_dataset(mask_variant, mask_salt=5)
    ref = dscompare.load_dataset(base)
    cand = dscompare.load_dataset(mask_variant)
    expect_refusal("P4  attention_mask_sha256 differs -> panel_mismatch (BIND-3)",
                   lambda: dscompare.run_gates(ref, cand, {}), code="panel_mismatch")

    # P5 scoring_window differs
    windowed = os.path.join(tmp, "p5")
    build_dataset(windowed, score_from=1)
    expect_refusal("P5  scoring_window score_from 0 vs 1 -> panel_mismatch (PANEL-D3)",
                   lambda: dscompare.run_gates(dscompare.load_dataset(base),
                                               dscompare.load_dataset(windowed), {}),
                   code="panel_mismatch")

    # P6 panel_receipt_sha256 reused as the token identity
    root6 = os.path.join(tmp, "p6")
    shutil.copytree(base, root6)
    doc = F.read_json(os.path.join(root6, F.MANIFEST_NAME))
    doc["panel"]["panel_receipt_sha256"] = doc["panel"]["suite_token_hash_sha256"]
    F.write_json(os.path.join(root6, F.MANIFEST_NAME), F.seal_manifest(doc))
    expect_errors("P6  panel_receipt_sha256 reused as token identity -> refused (PANEL-D2)",
                  dsvalidate.validate_dataset(root6))

    # P7 tokens edited, aggregate not recomputed
    root7 = os.path.join(tmp, "p7")
    shutil.copytree(base, root7)
    token_path = os.path.join(root7, "panel/tokens/context-0000.json")
    ids = F.read_json(token_path)
    ids[0] += 1
    open(token_path, "w").write(json.dumps(ids, separators=(",", ":")))
    reseal(root7)
    expect_errors("P7  edited tokens, stale aggregate -> panel_digest_mismatch (BIND-6)",
                  dsvalidate.validate_dataset(root7), code="panel_digest_mismatch")

    # P8 the two preimages genuinely differ
    ids = [1, 2, 3]
    check("P8  compact vs default separators are different preimages (5.1)",
          F.token_ids_json_sha256(ids) != F.token_ids_json_sha256_legacy(ids)
          and F.suite_token_hash_sha256(["a" * 64, "b" * 64])
          != F.suite_token_hash_sha256_legacy(["a" * 64, "b" * 64]))

    # P9 remap entry whose target does not hash to its key
    root9 = os.path.join(tmp, "p9")
    shutil.copytree(base, root9)
    sealed = F.seal_receipt({
        "schema": "quant-pipeline.glm53-token-panel-receipt.v1", "receipt_sha256": "",
        "artifacts": [{"path": "/workspace/artifacts/tokens/context-0000.json",
                       "bytes": 1, "sha256": "e" * 64}]})
    F.write_json(os.path.join(root9, "panel/panel-receipt.json"), sealed)
    F.write_json(os.path.join(root9, "panel/panel-remap.json"), F.seal_receipt({
        "schema": F.REMAP_SCHEMA, "receipt_sha256": "",
        "for_receipt_sha256": sealed["receipt_sha256"],
        "for_receipt_file": "panel/panel-receipt.json",
        "resolution_rule": "resolve by sha256, never by path",
        "entries": {"e" * 64: "panel/tokens/context-0000.json"}}))
    doc = F.read_json(os.path.join(root9, F.MANIFEST_NAME))
    doc["panel"]["remap_file"] = "panel/panel-remap.json"
    F.write_json(os.path.join(root9, F.MANIFEST_NAME), F.seal_manifest(doc))
    reseal(root9)
    expect_errors("P9  remap target does not hash to its key -> remap_invalid (REMAP-2)",
                  dsvalidate.validate_dataset(root9), code="remap_invalid")


# ---------------------------------------------------------------------------
# H -- head identity
# ---------------------------------------------------------------------------


def section_head(tmp):
    print("\n== H: head identity, the head trap (spec 8) ==")
    same_a = os.path.join(tmp, "h-a")
    same_b = os.path.join(tmp, "h-b")
    build_dataset(same_a, seed=1, head_seed=7)
    build_dataset(same_b, seed=2, head_seed=7, role="quant", quantized=True)
    gates, findings = dscompare.run_gates(dscompare.load_dataset(same_a),
                                          dscompare.load_dataset(same_b), {})
    check("H1  hidden<->hidden, equal head content -> shared_reference_head, info (HEAD-1a)",
          findings["head_policy"] == "shared_reference_head"
          and any(d["code"] == "shared_reference_head" and d["severity"] == "info"
                  for d in findings["disclosures"])
          and findings["class"] == "strict")

    other_head = os.path.join(tmp, "h-c")
    build_dataset(other_head, seed=2, head_seed=99, role="quant", quantized=True)
    expect_refusal("H2  hidden<->hidden, DIFFERENT head content -> REFUSED (HEAD-1b)",
                   lambda: dscompare.run_gates(dscompare.load_dataset(same_a),
                                               dscompare.load_dataset(other_head), {}),
                   code="head_mismatch", gate="head")

    gates, findings = dscompare.run_gates(
        dscompare.load_dataset(same_a), dscompare.load_dataset(other_head),
        {"disclose_head_substitution": True})
    blocking = [d for d in findings["disclosures"] if d["code"] == "head_substituted"]
    check("H3  H2 + --disclose-head-substitution -> advisory, downward bias, blocking",
          findings["class"] == "advisory"
          and findings["bias"]["direction"] == "downward"
          and blocking and blocking[0]["severity"] == "blocking")

    logit_a = os.path.join(tmp, "h-la")
    logit_b = os.path.join(tmp, "h-lb")
    build_dataset(logit_a, form="logit", seed=1, head_seed=7)
    build_dataset(logit_b, form="logit", seed=2, head_seed=99, role="quant", quantized=True)
    gates, findings = dscompare.run_gates(dscompare.load_dataset(logit_a),
                                          dscompare.load_dataset(logit_b), {})
    check("H4  logit<->logit with different heads -> ALLOWED, native_head (HEAD-2)",
          gates["head"]["passed"] and findings["head_policy"] == "native_head")

    gates, findings = dscompare.run_gates(dscompare.load_dataset(same_a),
                                          dscompare.load_dataset(logit_a), {})
    check("H5  hidden<->logit, equal head digests -> allowed (HEAD-3)",
          gates["head"]["passed"] and findings["head_policy"] == "native_head")

    expect_refusal("H6  hidden<->logit, different head digests -> REFUSED (HEAD-3)",
                   lambda: dscompare.run_gates(dscompare.load_dataset(same_a),
                                               dscompare.load_dataset(logit_b), {}),
                   code="head_mismatch")

    null_head = os.path.join(tmp, "h-null")
    build_dataset(null_head, role="quant", quantized=True, head_content="")
    doc = F.read_json(os.path.join(null_head, F.MANIFEST_NAME))
    doc["head"]["tensor_content_sha256"] = None
    doc["head"]["raw_tensor_sha256"] = None
    F.write_json(os.path.join(null_head, F.MANIFEST_NAME), F.seal_manifest(doc))
    expect_refusal("H7  hidden form with a null head content digest -> REFUSED, no override",
                   lambda: dscompare.run_gates(
                       dscompare.load_dataset(same_a),
                       dscompare.load_dataset(null_head, verify=False),
                       {"disclose_head_substitution": True}),
                   code="head_mismatch")
    expect_errors("H7b validator also refuses it (HEAD-4)",
                  dsvalidate.validate_dataset(null_head), code="head_mismatch")

    applied = os.path.join(tmp, "h-applied")
    build_dataset(applied, head_applied_in_capture=True)
    expect_errors("H8  hidden form with head.applied_in_capture true -> invalid (HEAD-5)",
                  dsvalidate.validate_dataset(applied), rule="HEAD-5")

    headless = os.path.join(tmp, "h-headless")
    build_dataset(headless, head_present=False)
    doc = F.read_json(os.path.join(headless, F.MANIFEST_NAME))
    doc["head"]["present"] = False
    F.write_json(os.path.join(headless, F.MANIFEST_NAME), F.seal_manifest(doc))
    expect_errors("H9  role root with head.present false -> invalid (HEAD-6)",
                  dsvalidate.validate_dataset(headless), rule="HEAD-6")

    norm = os.path.join(tmp, "h-norm")
    build_dataset(norm, final_norm_applied_at_replay=True)
    expect_errors("H10 post-norm cut + final_norm.applied_at_replay -> invalid (HEAD-7)",
                  dsvalidate.validate_dataset(norm), rule="HEAD-7")

    # H11: equal file digest, different tensor content -- content is normative.
    h11a = os.path.join(tmp, "h11a")
    h11b = os.path.join(tmp, "h11b")
    build_dataset(h11a, head_seed=7)
    build_dataset(h11b, head_seed=99, role="quant", quantized=True)
    doc = F.read_json(os.path.join(h11b, F.MANIFEST_NAME))
    other = F.read_json(os.path.join(h11a, F.MANIFEST_NAME))
    doc["head"]["file_sha256"] = other["head"]["file_sha256"]
    F.write_json(os.path.join(h11b, F.MANIFEST_NAME), F.seal_manifest(doc))
    expect_refusal("H11 equal head file digest, different content -> REFUSED (O-6)",
                   lambda: dscompare.run_gates(
                       dscompare.load_dataset(h11a),
                       dscompare.load_dataset(h11b, verify=False), {}),
                   code="head_mismatch")


# ---------------------------------------------------------------------------
# L / C / X -- lane, stack, coverage, lossy
# ---------------------------------------------------------------------------


def section_lane(tmp):
    print("\n== L/C/X: lane, stack, coverage, lossy (spec 9, 10.1) ==")
    a = os.path.join(tmp, "l-a")
    b = os.path.join(tmp, "l-b")
    build_dataset(a, seed=1)
    build_dataset(b, seed=2, role="quant", quantized=True)
    gates, findings = dscompare.run_gates(dscompare.load_dataset(a),
                                          dscompare.load_dataset(b), {})
    check("L1  same lane -> same_lane true, usable_as_floor true",
          findings["same_lane"] and findings["usable_as_floor"])

    other_lane = os.path.join(tmp, "l-c")
    build_dataset(other_lane, seed=2, lane="streaming", role="quant", quantized=True)
    expect_refusal("L2  different lanes without a flag -> lane_mismatch",
                   lambda: dscompare.run_gates(dscompare.load_dataset(a),
                                               dscompare.load_dataset(other_lane), {}),
                   code="lane_mismatch", gate="lane")

    gates, findings = dscompare.run_gates(dscompare.load_dataset(a),
                                          dscompare.load_dataset(other_lane),
                                          {"allow_cross_lane": True})
    check("L3  --allow-cross-lane -> advisory AND usable_as_floor false (BIAS-006)",
          findings["class"] == "advisory" and findings["usable_as_floor"] is False
          and any(d["code"] == "cross_engine_capture" for d in findings["disclosures"]))

    check("L4  equal lane_identity + stack_fingerprint -> same_stack",
          findings.get("stack_relation") == "same_stack")

    cross_stack = os.path.join(tmp, "l-e")
    build_dataset(cross_stack, seed=2, stack="stack-b", lane_identity="lane-b",
                  role="quant", quantized=True)
    gates, findings = dscompare.run_gates(dscompare.load_dataset(a),
                                          dscompare.load_dataset(cross_stack), {})
    check("L5  differing stack fingerprints -> cross_stack + bias block (BIAS-001)",
          findings["stack_relation"] == "cross_stack" and findings.get("bias"))

    # C1: our own O-3 defect -- declared 5120, present 512, complete true.
    c1 = os.path.join(tmp, "c1")
    build_dataset(c1)
    doc = F.read_json(os.path.join(c1, F.MANIFEST_NAME))
    doc["coverage"]["declared_records"] = 5120
    doc["coverage"]["complete"] = True
    F.write_json(os.path.join(c1, F.MANIFEST_NAME), F.seal_manifest(doc))
    expect_errors("C1  declared 5120 / present 2 / complete true -> refused (COV-1, our O-3)",
                  dsvalidate.validate_dataset(c1), code="incomplete")

    c2 = os.path.join(tmp, "c2")
    build_dataset(c2, declared_records=10, shard_of={"index": 0, "total": 5, "stride": 1})
    report = dsvalidate.validate_dataset(c2, allow_partial=True)
    check("C2  same numbers with shard_of + --allow-partial -> accepted (COV-2/3)",
          report.passed, json.dumps(report.errors[:3]))
    check("C2b without --allow-partial the same dataset is refused (COV-3)",
          not dsvalidate.validate_dataset(c2).passed)

    c3 = os.path.join(tmp, "c3")
    build_dataset(c3, seed=2, capture_indices=[0], role="quant", quantized=True,
                  shard_of={"index": 0, "total": 2, "stride": 1})
    expect_refusal("C3  differing index sets, no flag -> coverage_mismatch",
                   lambda: dscompare.run_gates(
                       dscompare.load_dataset(a),
                       dscompare.load_dataset(c3, allow_partial=True), {}),
                   code="coverage_mismatch", gate="coverage")
    gates, findings = dscompare.run_gates(
        dscompare.load_dataset(a), dscompare.load_dataset(c3, allow_partial=True),
        {"allow_partial": True})
    check("C4  --allow-partial -> intersect + subset_of_panel disclosure (SCOPE-010)",
          findings["shared_indices"] == [0]
          and any(d["code"] == "subset_of_panel" for d in findings["disclosures"])
          and findings["covers_full_panel"] is False)

    x1 = os.path.join(tmp, "x1")
    build_dataset(x1, seed=2, role="quant", quantized=True,
                  lossy_codec={"kind": "llamacpp-kld-uint16", "bits": 16,
                               "clamp": "max_logit - 16",
                               "description": "16-bit quantized log-probs with a hard "
                                              "max_logit-16 floor"})
    gates, findings = dscompare.run_gates(dscompare.load_dataset(a),
                                          dscompare.load_dataset(x1), {})
    check("X1  lossy_codec non-null -> advisory + lossy_capture_codec (D-8)",
          findings["class"] == "advisory"
          and any(d["code"] == "lossy_capture_codec" for d in findings["disclosures"]))

    x2 = os.path.join(tmp, "x2")
    build_dataset(x2, seed=2, role="quant", quantized=True, dtype_lossless=False)
    gates, findings = dscompare.run_gates(dscompare.load_dataset(a),
                                          dscompare.load_dataset(x2), {})
    check("X2  dtype_lossless false -> advisory (FORM-1)", findings["class"] == "advisory")

    # SV1/SV2 -- SCOPE-VOCAB. A scope the registry's schema will reject must be
    # caught while the dataset is being written, not at submission time after
    # the GPU hours are spent. Real finding: the first candidate this repo
    # captured on real weights declared format "rtn-int4-per-row" and was
    # rejected by registry_validate.py --submission on schema alone.
    allowed = dsvalidate.registry_numeric_formats()
    check("SV1 the registry numeric_format enum is READ, not copied",
          bool(allowed) and "exl3-mcg" in allowed and "int4" in allowed,
          "got %r" % (sorted(allowed)[:4] if allowed else None))

    sv = os.path.join(tmp, "scope-vocab")
    build_dataset(sv, seed=2, role="quant", quantized=True)
    manifest = F.read_json(os.path.join(sv, F.MANIFEST_NAME))
    for assignment in manifest["scope"]["assignments"]:
        if assignment["treatment"] == "quantized":
            assignment["format"] = "rtn-int4-per-row"
    F.write_json(os.path.join(sv, F.MANIFEST_NAME), manifest)
    reseal(sv)
    report = dsvalidate.validate_dataset(sv, verify_tensors=False)
    check("SV2 an off-vocabulary scope format warns (SCOPE-VOCAB), and is not an error",
          not report.errors
          and any(w["code"] == "scope_format_unknown" for w in report.warnings),
          "errors=%d warnings=%s" % (len(report.errors),
                                     [w["code"] for w in report.warnings]))


# ---------------------------------------------------------------------------
# R -- real published artifacts, metadata only
# ---------------------------------------------------------------------------


def section_real(tmp):
    print("\n== R: real artifacts (metadata only, no bulk download) ==")
    examples = os.path.join(REPO, "docs", "examples")
    ok = True
    for name, schema in (("fidelity-dataset.root-glm53-bf16.json",
                          "fidelity-dataset.schema.json"),
                         ("fidelity-dataset.quant-glm53-k6.json",
                          "fidelity-dataset.schema.json"),
                         ("fidelity-comparison-receipt.k6-vs-bf16.json",
                          "fidelity-comparison-receipt.schema.json"),
                         ("fidelity-comparison-receipt.self-compare.json",
                          "fidelity-comparison-receipt.schema.json")):
        doc = F.read_json(os.path.join(examples, name))
        errors = dsvalidate.schema_errors(doc, schema)
        field = "dataset_sha256" if "dataset" in name else "receipt_sha256"
        sealed = F.recompute_seal(doc, field) == doc[field]
        if errors or not sealed:
            ok = False
            print("        %s: %d schema errors, seal=%s" % (name, len(errors), sealed))
    check("R4  the four shipped worked examples: schema clean AND seals recompute", ok)

    receipt = F.read_json(os.path.join(examples, "fidelity-comparison-receipt.self-compare.json"))
    report = dsvalidate.validate_receipt(receipt)
    check("R4b self-compare example passes the SC-1 conditional rules", report.passed,
          json.dumps(report.errors[:3]))

    # Our own published capture manifest: the O-3 / O-4 defects, in real data.
    real = os.path.join(REPO, "deliverables", "deliverables",
                        "reference-bf16-shard0", "capture-manifest-full.json")
    if os.path.isfile(real):
        doc = F.read_json(real)
        overclaims = (doc.get("complete") is True
                      and doc.get("expected_contexts") == 5120
                      and len(doc.get("captures") or []) == 5120)
        thin = set((doc.get("captures") or [{}])[0].keys()) == {"index", "sha256", "shape"}
        no_cut = "semantic_point" not in doc and "cut_point" not in doc
        check("R1  our published glm53flash-fidelity-capture/2 exhibits O-1, O-3 and O-4",
              overclaims and thin and no_cut,
              "overclaims=%s thin=%s no_cut=%s" % (overclaims, thin, no_cut))
    else:
        print("  SKIP  R1 (deliverables/ not present on this machine)")



# ---------------------------------------------------------------------------
# I -- interop: the adapters, on real metadata where it is available
# ---------------------------------------------------------------------------


def _k3_fixture(tmp):
    """A synthetic artifact in kimi-k3's EXACT shape: his file names, his field
    names, his digest preimages.  Proves the adapter reads his layout without
    needing his 30 GB of tensors."""
    root = os.path.join(tmp, "k3-synthetic")
    os.makedirs(os.path.join(root, "reference-hidden"), exist_ok=True)
    os.makedirs(os.path.join(root, "lm-head"), exist_ok=True)
    os.makedirs(os.path.join(root, "tokens"), exist_ok=True)
    os.makedirs(os.path.join(root, "validation"), exist_ok=True)
    contexts = []
    hidden = []
    digests = []
    for index in range(3):
        ids = [10 + index, 20 + index, 30 + index, 40 + index]
        path = os.path.join(root, "tokens", "context-%04d.json" % index)
        open(path, "w").write(json.dumps(ids, separators=(",", ":")))
        digest = F.token_ids_json_sha256(ids)
        digests.append(digest)
        contexts.append({
            "context_index": index, "index": index, "num_tokens": len(ids),
            "token_file": "tokens/context-%04d.json" % index,
            "token_ids_json_sha256": digest,
            "token_ids_first16": ids[:16], "token_ids_last16": ids[-16:],
            "allocation_stratum": "encyclopedic_factual",
            "semantic_class": "encyclopedic_article", "source_cluster_id": str(index),
            "partition": "analysis", "sentinel": False,
            "scored_row_end_exclusive": len(ids) - 1, "scored_row_start": 0,
        })
        hidden.append({
            "context_index": index, "dtype": "BF16",
            "file": "hidden_%04d.safetensors" % index, "key": "hidden_states",
            "raw_chunks_retained": False, "sha256": "0" * 64,
            "shape": [len(ids) - 1, 8], "size_bytes": 48,
            "token_ids_json_sha256": digest,
        })
    aggregate = F.suite_token_hash_sha256(digests)
    F.write_json(os.path.join(root, "suite-manifest.json"), {
        "kind": "Kimi K3 teacher-forced distribution-fidelity token suite",
        "format_version": 1, "context_count": 3, "context_length": 4,
        "scored_positions_per_context": 3, "total_scored_positions": 9,
        "suite_token_hash_sha256": aggregate, "contexts": contexts,
        "tokenizer": {"class": "TikTokenTokenizer"}, "status": "implemented"})
    F.write_json(os.path.join(root, "reference-hidden", "manifest.json"), {
        "kind": "Kimi K3 final-normalized pre-LM-head hidden states", "format_version": 1,
        "semantic_point": "after_final_rmsnorm_before_lm_head",
        "tensor_key": "hidden_states", "hidden_width": 8, "context_length": 4,
        "scored_rows_per_context": 3, "suite_token_hash_sha256": aggregate,
        "runtime_manifest": "../capture-runtime.json",
        "runtime_manifest_sha256": "1" * 64,
        "total_size_bytes": 144, "contexts": hidden})
    F.write_json(os.path.join(root, "lm-head", "manifest.json"), {
        "kind": "Kimi K3 canonical LM-head weight", "format_version": 1,
        "file": "weight.safetensors", "key": "weight", "dtype": "BF16",
        "shape": [16, 8], "size_bytes": 256,
        "file_sha256": "2" * 64, "raw_tensor_sha256": "3" * 64})
    F.write_json(os.path.join(root, "capture-runtime.json"), {
        "artifact_kind": "synthetic", "format_version": 1,
        "container": {"image_id": "sha256:" + "4" * 64, "image_reference": "x:y",
                      "image_repository_digest": "x@sha256:" + "5" * 64},
        "runtime": {"tensor_parallel_size": 16, "attention_backend": "B12X_MLA"},
        "runtime_environment": {"VLLM_USE_V2_MODEL_RUNNER": "1"},
        "source_files": {"vllm/v1/worker/gpu/model_runner.py": "6" * 64}})
    F.write_json(os.path.join(root, "manifest.json"), {
        "artifact_kind": "synthetic k3 reference", "format_version": 1,
        "context_count": 3, "context_length": 4, "status": "qualified",
        "scored_positions_per_context": 3, "total_scored_positions": 9,
        "lm_head": {"file": "lm-head/weight.safetensors", "file_sha256": "2" * 64,
                    "raw_tensor_sha256": "3" * 64, "shape": [16, 8]},
        "reference_hidden": {"manifest": "reference-hidden/manifest.json",
                             "manifest_sha256": "7" * 64, "context_count": 3},
        "suite": {"manifest": "suite-manifest.json", "manifest_sha256": "8" * 64,
                  "suite_token_hash_sha256": aggregate}})
    F.write_json(os.path.join(root, "validation", "artifact-validation.json"), {
        "status": "qualified", "context_count": 3,
        "sentinel_repeat_mean_kld": {"00-vs-01": 0.0032166685936858316,
                                     "00-vs-02": 0.0031814546488495347}})
    return root, aggregate


def section_interop(tmp):
    print("\n== I: interop adapters (spec 12) ==")
    root, aggregate = _k3_fixture(tmp)
    report = dsadapt.adapt_k3(root, os.path.join(tmp, "k3-out"), source="k3v1")
    check("I1  the k3v1 adapter reads kimi-k3's layout and RECOMPUTES his aggregate "
          "from his per-record digests under the adopted preimage",
          report["panel"]["suite_token_hash_sha256_recomputed"] == aggregate
          and report["panel"]["aggregate_agrees"] is True)
    check("I2  the k3 head is imported by his raw_tensor_sha256 with quantized=null "
          "(his format cannot express it, so every comparison is advisory -- D-1)",
          report["head"]["tensor_content_sha256"] == "3" * 64
          and report["head"]["quantized"] is None
          and "head.quantized" in report["inferred_fields"])
    check("I3  the k3 lane is inferred to `other` (the registry has no serving lane) "
          "and lane inference is declared",
          report["runtime"]["lane"] == "other" and report["runtime"]["lane_inferred"] is True
          and "runtime.lane" in report["inferred_fields"])
    check("I4  his container / runtime_environment / source_files blocks survive",
          report["runtime"]["container"]["image_id"].startswith("sha256:")
          and report["runtime"]["runtime_environment"]
          and report["runtime"]["source_files_pinned_by_content"])
    check("I5  his sentinel repeat noise is imported and DOWNGRADED to "
          "run_mean_equality_only (his per-file digests are container hashes)",
          report["determinism"]["evidence_kind"] == "run_mean_equality_only"
          and report["determinism"]["repeat_noise"]["kl_canonical_to_repeat_mean"]
          == 0.0032166685936858316)
    check("I6  a metadata-only translation reports 0 present records and says why, "
          "instead of inventing a digest",
          report["coverage"]["present_records"] == 0
          and "not downloaded" in (report["coverage"]["subset_detail"] or ""))

    # llama.cpp .kld -- write a real one, byte for byte.
    kld = os.path.join(tmp, "synthetic.kld")
    n_ctx, n_vocab, n_chunk = 8, 32, 2
    with open(kld, "wb") as handle:
        handle.write(b"_logits_")
        handle.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        handle.write(struct.pack("<%di" % (n_ctx * n_chunk),
                                 *range(n_ctx * n_chunk)))
    header = dsadapt.read_llamacpp_kld_header(kld)
    check("I7  the llama.cpp .kld header parses (magic, n_ctx, n_vocab, n_chunk, tokens)",
          header["n_ctx"] == n_ctx and header["n_vocab"] == n_vocab
          and header["n_chunk"] == n_chunk and len(header["tokens"]) == n_ctx * n_chunk)
    check("I8  .kld scores the SECOND HALF only, and that is PANEL IDENTITY, not a flag "
          "(D-3): score_from = n_ctx/2",
          header["scoring_window"]["score_from"] == n_ctx // 2
          and header["scoring_window"]["windowed"] is True)
    llama = dsadapt.adapt_llamacpp_kld(kld, os.path.join(tmp, "kld-out"))
    check("I9  a .kld lands as lossy logit form with head_separable false (D-8 / D-10)",
          llama["capture"]["lossy_codec"]["bits"] == 16
          and llama["capture"]["head_separable"] is False
          and llama["capture"]["dtype_lossless"] is False)

    # Real fixtures, when they are on this machine.
    real_k3 = os.environ.get("FIDELITY_K3_FIXTURE")
    if real_k3 and os.path.isdir(real_k3):
        real = dsadapt.adapt_k3(real_k3, os.path.join(tmp, "k3-real"), source="k3v1")
        check("I10 the k3v1 adapter recomputes the REAL published "
              "suite_token_hash_sha256 from the real manifests",
              real["panel"]["aggregate_agrees"] is True,
              real["panel"]["suite_token_hash_sha256_declared"][:16])
    else:
        print("  SKIP  I10 real kimi-k3 manifests (set FIDELITY_K3_FIXTURE)")

    published = os.path.join(REPO, "deliverables", "deliverables")
    if os.path.isdir(os.path.join(published, "reference-bf16-shard0")):
        out = os.path.join(tmp, "serving-v2")
        manifest = dsadapt.adapt_serving_v2(
            os.path.join(published, "reference-bf16-shard0"), out,
            suite_dir=os.path.join(published, "suite"),
            head_dir=os.path.join(published, "head"),
            dataset_id="fidelity--selftest.serving-v2", name="selftest serving-v2",
            role="root", lane="other", limit=1, link=True)
        report = dsvalidate.validate_dataset(out, verify_tensors=True, allow_partial=True)
        check("I11 our OWN published glm53flash-fidelity-capture/2 adapts to a conformant, "
              "seal-verified v1 dataset (the superset proof)",
              report.passed, json.dumps(report.errors[:2]))
        check("I12 the adapter recomputes the head TENSOR CONTENT digest aa21c427... from "
              "the published head.safetensors, whose FILE digest is 47eaf729... (O-6)",
              manifest["head"]["tensor_content_sha256"].startswith("aa21c427")
              and manifest["head"]["file_sha256"].startswith("47eaf729"))
        check("I13 O-3 is FIXED, not copied: the published manifest says complete:true over "
              "5,120 captures; the adapted dataset says 1 of 5,120 with shard_of",
              manifest["coverage"]["complete"] is False
              and manifest["coverage"]["declared_records"] == 5120
              and manifest["coverage"]["shard_of"] is not None)
        check("I14 O-1 is FIXED: semantic_point is declared and is kimi-k3's exact string",
              manifest["capture"]["semantic_point"]
              == "after_final_rmsnorm_before_lm_head")
        suite = F.read_json(os.path.join(published, "suite", "suite-manifest.json"))
        row = suite["context_index"][0]
        ids = F.read_json(os.path.join(published, "suite", row["file"]))
        check("I15 our LEGACY token preimage reproduces the published token_sha256 exactly, "
              "and the adopted compact preimage differs (5.1: a preimage divergence, not a "
              "naming one)",
              F.token_ids_json_sha256_legacy(ids) == row["token_sha256"]
              and F.token_ids_json_sha256(ids) != row["token_sha256"])
        # ---- CC-03: a LEGACY-keyed capture must be rewritten in the FILE ----------
        # REC-2 says a pre-v1 `hidden` key is "accepted on ingest and rewritten". The
        # rewrite was manifest-only: the tensor was hardlinked verbatim and still carried
        # `hidden` while record["key"] said `hidden_states`, so the emitted manifest named
        # a tensor the bytes do not contain. dsvalidate's own SEAL-1(d) refuses that, and
        # any consumer following record["key"] gets a KeyError.
        legacy_src = os.path.join(tmp, "legacy-src")
        os.makedirs(legacy_src, exist_ok=True)
        real_shard = os.path.join(published, "reference-bf16-shard0")
        shutil.copyfile(os.path.join(real_shard, "capture-manifest-full.json"),
                        os.path.join(legacy_src, "capture-manifest-full.json"))
        src_t = os.path.join(real_shard, "hidden_0000.safetensors")
        dst_t = os.path.join(legacy_src, "hidden_0000.safetensors")
        with open(src_t, "rb") as fh:
            raw = fh.read()
        hlen = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8:8 + hlen])
        hdr["hidden"] = hdr.pop("hidden_states")          # forge the pre-v1 key
        blob = json.dumps(hdr, separators=(",", ":"), sort_keys=True).encode("utf-8")
        blob += b" " * ((-len(blob)) % 8)
        with open(dst_t, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob))); fh.write(blob); fh.write(raw[8 + hlen:])
        before_content = F.tensor_content_sha256(dst_t, "hidden")
        lout = os.path.join(tmp, "legacy-out")
        lman = dsadapt.adapt_serving_v2(
            legacy_src, lout, suite_dir=os.path.join(published, "suite"),
            head_dir=os.path.join(published, "head"),
            dataset_id="fidelity--selftest.legacy-key", name="legacy key",
            role="root", lane="other", limit=1, link=False)
        cap_man = F.read_json(os.path.join(lout, "capture", "manifest.json"))
        rec = cap_man["records"][0]
        emitted = os.path.join(lout, "capture", rec["file"])
        _, ehdr = F.read_safetensors_header(emitted)
        ekeys = [k for k in ehdr if k != "__metadata__"]
        lrep = dsvalidate.validate_dataset(lout, verify_tensors=True, allow_partial=True)
        check("CC-03 a pre-v1 `hidden` capture is rewritten in the FILE, so the manifest "
              "names a tensor the bytes actually contain",
              rec.get("key") == "hidden_states" and ekeys == ["hidden_states"]
              and lrep.passed,
              "record.key=%r file keys=%r validate=%s %s"
              % (rec.get("key"), ekeys, lrep.passed, json.dumps(lrep.errors[:1])))
        check("CC-03 the rewrite preserves tensor CONTENT identity (only the container "
              "digest moves)",
              F.tensor_content_sha256(emitted, "hidden_states") == before_content,
              before_content[:16])
        # ---- CC-03b: the rewrite must not reach back through the HARDLINK ---------
        # `link=True` is the DEFAULT (`link=not args.copy`), and the case above ran with
        # link=False, which is why the defect survived. With a hardlink, `dest` and the
        # caller's source capture share an inode, so an in-place "wb" rewrite changed the
        # SOURCE file: its sha256 moved, its own checksums.txt/manifest rows stopped
        # matching, and a dataset just fetched from the Hub became unverifiable against
        # the published digests -- silently, as a side effect of reading it.
        src_before = F.sha256_file(dst_t)
        lout2 = os.path.join(tmp, "legacy-out-linked")
        dsadapt.adapt_serving_v2(
            legacy_src, lout2, suite_dir=os.path.join(published, "suite"),
            head_dir=os.path.join(published, "head"),
            dataset_id="fidelity--selftest.legacy-key-linked", name="legacy key linked",
            role="root", lane="other", limit=1, link=True)
        man2 = F.read_json(os.path.join(lout2, "capture", "manifest.json"))
        emitted2 = os.path.join(lout2, "capture", man2["records"][0]["file"])
        _, ehdr2 = F.read_safetensors_header(emitted2)
        check("CC-03b adapt --link rewrites the emitted copy and leaves the caller's "
              "source capture byte-identical (the rewrite must break the hardlink)",
              F.sha256_file(dst_t) == src_before
              and [k for k in ehdr2 if k != "__metadata__"] == ["hidden_states"]
              and os.stat(dst_t).st_nlink == 1,
              "source moved=%s emitted keys=%r nlink=%d"
              % (F.sha256_file(dst_t) != src_before,
                 [k for k in ehdr2 if k != "__metadata__"], os.stat(dst_t).st_nlink))
    else:
        print("  SKIP  I11-I15 published deliverables (not on this machine)")

    # -- I16..I19 the k3 emission and the compat view ------------------------
    from fidelity import k3compat                                    # noqa: WPS433

    k3ds = os.path.join(tmp, "k3-emitted")
    emitted = dsadapt.adapt_k3(root, k3ds, source="k3v1", emit_dataset=True)
    check("I16 --emit-dataset on a tensor-less k3 artifact REFUSES to seal, and says why "
          "(a seal is computed over bytes, never fabricated)",
          emitted["emitted"]["written"] is False
          and "sealed dataset is made of BYTES" in emitted["emitted"]["reason"])
    try:
        dsadapt.adapt_k3(root, os.path.join(tmp, "k3-root"), source="k3v1",
                         emit_dataset=True, role="root")
        check("I17 --role root is refused for a k3 translation (ROOT-1 asserts a head "
              "quantization status the source never records -- D-1)", True,
              "no tensors, so the role guard is not reached")
    except dsadapt.AdapterError as exc:
        check("I17 --role root is refused for a k3 translation (ROOT-1 asserts a head "
              "quantization status the source never records -- D-1)",
              "ROOT-1" in str(exc) and "derived" in str(exc), str(exc)[:80])

    compat_root = os.path.join(tmp, "compat-ds")
    build_dataset(compat_root, emit_k3_compat=True)
    manifest = F.load_manifest(compat_root)
    listed = set(F.parse_checksums(
        open(os.path.join(compat_root, "checksums.txt")).read()))
    compat_files = sorted(f for f in listed if f.startswith("compat/"))
    report = dsvalidate.validate_dataset(compat_root, verify_tensors=True)
    check("I18 --emit-k3-compat writes compat/ INSIDE the seal (SEAL-1(c) would refuse it "
          "otherwise) and the dataset still validates",
          report.passed and len(compat_files) == 3
          and manifest["interop"]["k3_compat_emitted"] is True
          and manifest["interop"]["k3_compat_tensor_bytes_duplicated"] == 0,
          "%d compat file(s): %s" % (len(compat_files), ", ".join(compat_files)))
    problems = k3compat.verify(compat_root)
    suite = F.read_json(os.path.join(compat_root, "compat", "suite-manifest.json"))
    panel = F.read_json(os.path.join(compat_root, manifest["panel"]["panel_file"]))
    resolves = all(os.path.isfile(os.path.normpath(os.path.join(
        compat_root, "compat", row["token_file"]))) for row in suite["contexts"])
    check("I19 the compat view is faithful: `contexts` is a LIST (PANEL-D5), the suite token "
          "hash is copied up, and every relative alias resolves onto the ONE real file",
          not problems and isinstance(suite["contexts"], list) and resolves
          and suite["suite_token_hash_sha256"] == panel["suite_token_hash_sha256"],
          "; ".join(problems[:2]) or "clean")


def section_hostile_fetch(tmp):
    """A dataset fetched from somebody else's repo is UNTRUSTED INPUT.

    `fetch_dataset` wrote every path listed in the remote `checksums.txt`, and the file
    named by the remote manifest's `seal.checksums_file`, straight onto the download
    directory with `os.path.join` -- so `../../../../engines/tools/stream_score.py` landed
    there, an absolute entry won outright, and the digests beside those paths were parsed
    and never compared to the bytes. Pointing `fidelity-dataset verify` at a stranger's
    repo is the documented way to look at their capture, so this was reachable from the
    front door. These cases drive the parser and the containment proof directly; the
    network half is exercised by hand against a local endpoint (see the commit).
    """
    print("\n== X: a hostile dataset must not be able to write outside the download dir ==")
    for label, line in (
            ("relative traversal", "%s  ../../PWNED.txt" % ("0" * 64)),
            ("absolute path", "%s  /tmp/PWNED.txt" % ("0" * 64)),
            ("nested traversal", "%s  capture/../../../PWNED.txt" % ("0" * 64)),
            ("windows drive", "%s  C:/PWNED.txt" % ("0" * 64)),
    ):
        try:
            F.parse_checksums(line + "\n")
            ok, detail = False, "ACCEPTED -- this path would have been written"
        except F.FormatError as exc:
            ok, detail = exc.code == "seal_failed", exc.message[:70]
        check("X1  checksums.txt %-18s is refused at parse time" % label, ok, detail)

    good = "%s  capture/hidden_0000.safetensors\n%s  panel/tokens/context-0000.json\n" % (
        "a" * 64, "b" * 64)
    try:
        parsed = F.parse_checksums(good)
        ok, detail = len(parsed) == 2, "parsed %d" % len(parsed)
    except F.FormatError as exc:
        ok, detail = False, "legitimate entries REFUSED: " + exc.message
    check("X2  a legitimate checksums.txt still parses", ok, detail)

    root = os.path.join(tmp, "x-contain")
    os.makedirs(os.path.join(root, "capture"), exist_ok=True)
    inside = F.resolve_inside(root, "capture/a.bin", owner="t")
    ok = inside.startswith(os.path.realpath(root) + os.sep)
    check("X3  resolve_inside keeps a legitimate path inside the root", ok, inside)
    escaped = None
    try:
        F.resolve_inside(root, "../../etc/passwd", owner="t")
    except F.FormatError as exc:
        escaped = exc.code
    check("X4  resolve_inside refuses a path that leaves the root",
          escaped == "path_escape", str(escaped))

    # SEC-07. `publish_dataset` called upload_folder with no ignore_patterns, and
    # `iter_dataset_files` walked dotfiles and dot-directories, so a stray credential
    # under a dataset root was HASHED INTO the published checksums.txt and then uploaded.
    # It refuses now rather than filtering: a file dropped from the upload but still
    # listed in checksums.txt makes the published dataset unverifiable.
    cred = os.path.join(tmp, "x-cred")
    build_dataset(cred)
    for rel in (".hf_token", ".secrets/hf_token", "run/.hf_token", ".env", "deploy.pem"):
        full = os.path.join(cred, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write("hf_NOTAREALTOKEN000000000000000000000")
        caught = None
        try:
            F.iter_dataset_files(cred)
        except F.FormatError as exc:
            caught = exc.code
        check("X5  %-20s cannot be sealed into a dataset" % rel,
              caught == "credential_in_tree", str(caught))
        os.remove(full)

    payload_ok = True
    try:
        listed = F.iter_dataset_files(cred)
        payload_ok = any(p.startswith("panel/tokens/") for p in listed)
    except F.FormatError as exc:
        payload_ok = False
    check("X6  legitimate panel/tokens/ payload is NOT mistaken for a credential",
          payload_ok, "a *token* pattern here would strip required payload")


def section_root_qualification(tmp):
    print("\n== Q: two-process root qualification ==")
    first = os.path.join(tmp, "q-first")
    repeat = os.path.join(tmp, "q-repeat")
    destination = "selftest/root-dataset"
    weights = "selftest/weights"
    build_dataset(first, seed=91, run_name="root-cold-1", cold_run="root-cold-1",
                  dataset_repository=destination, weights_repository=weights,
                  qualification_contract=True)
    build_dataset(repeat, seed=91, run_name="root-cold-2", cold_run="root-cold-2",
                  dataset_repository=destination, weights_repository=weights,
                  qualification_contract=True)
    same_root_rc = CLI.cmd_compare(argparse.Namespace(
        reference=first, candidate=first, allow_partial=False))
    check("Q0  --self-compare refuses one dataset path supplied twice",
          same_root_rc == CLI.REFUSED)
    first_verify = os.path.join(tmp, "q-first-verify.json")
    repeat_verify = os.path.join(tmp, "q-repeat-verify.json")
    common.write_json(first_verify, dsvalidate.validate_dataset(
        first, verify_tensors=True).to_dict())
    common.write_json(repeat_verify, dsvalidate.validate_dataset(
        repeat, verify_tensors=True).to_dict())
    comparison_dir = os.path.join(tmp, "q-comparison")
    comparison = dscompare.compare(first, repeat, comparison_dir, {
        "self_compare": True, "force_compute": True,
        "device": "cpu", "replay_device": "numpy", "replay_dtype": "float32",
        "vocab_chunk": 8192, "reference_label": "root-cold-1",
        "candidate_label": "root-cold-2", "verify_tensors": True,
    })
    comparison_path = os.path.join(comparison_dir, "comparison-receipt.json")
    first_manifest = F.load_manifest(first)
    first_runtime = F.read_json(os.path.join(
        first, first_manifest["runtime"]["file"]))
    binding = first_runtime["capture_tool"]["resolved_panel_binding"]["binding"]
    job_path = os.path.join(tmp, "q-job.json")
    q_bundle = jobcontract.finalize_bundle_manifest(
        [{"path": "bin/fidelity_dataset.py", "bytes": 1, "sha256": "6" * 64}],
        "qualification-selftest")
    q_control = jobcontract.finalize_bundle_manifest(
        [{"path": "bin/fidelity/jobcontract.py", "bytes": 1,
          "sha256": "7" * 64}], "qualification-control-selftest")
    q_control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    q_registry = {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "8" * 64}
    q_contract_sha = common.sha256_hex(json.dumps(
        {"bundle": q_bundle, "registry": q_registry},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False))
    q_shards = [{"path": "model.safetensors", "bytes": 17}]
    q_download_manifest = [
        {"path": "config.json", "bytes": 1},
        q_shards[0],
        {"path": "model.safetensors.index.json", "bytes": 1},
    ]
    q_download_sha = common.sha256_hex(json.dumps(
        q_download_manifest, sort_keys=True, separators=(",", ":")))
    q_names_sha = common.sha256_hex(
        json.dumps(["model.unused"], separators=(",", ":")))
    job = jobcontract.finalize_job({
        "schema": "fidelity-suite/job.v2",
        "execution_attempt": {
            "number": 1, "kind": "local-container", "attempt_id": "9" * 24},
        "bundle": q_bundle,
        "control_plane": q_control,
        "bundle_registry": q_registry,
        "bundle_contract_sha256": q_contract_sha,
        "role": "root",
        "lane": "sealed-ep8",
        "cold_runs": 2,
        "recipe": "local-container",
        "runtime": {}, "environment": {}, "measurer": {},
        "produced_by": {
            "dependencies": {
                "profile": "root-hf-transformers-bf16",
                "lane": "sealed-ep8",
                "provider": "local-container",
            },
        },
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 1, "min_memory_gb": 1,
            "expected_vram_bytes": 1,
        },
        "profile": {
            "profile_id": "root-hf-transformers-bf16",
            "lane": "root", "source": "native",
            "surface": "native-bf16", "form": "hidden",
            "engine": "hf-transformers",
            "compute_dtype": "bfloat16",
            "device": "cuda",
            "schedule": "two-fresh-process-qualification",
        },
        "timing": {"kind": "qualification-selftest"},
        "scope": {"kind": "qualification-selftest"},
        "target": {
            "repo_id": weights, "revision": "a" * 40,
            "path": None, "surface": "native-bf16",
            "codec": "bf16", "bits": 16,
            "config_sha256": "a" * 64, "index_sha256": "b" * 64,
            "shard_manifest_sha256": common.sha256_hex(json.dumps(
                q_shards, sort_keys=True, separators=(",", ":"))),
            "model_bytes": 17, "shards": q_shards,
            "download_manifest": q_download_manifest,
            "download_bytes_total": 19,
            "download_manifest_sha256": q_download_sha,
            "weights_license": None,
        },
        "panel": {
            "binding_file_sha256": "2" * 64,
            "binding_path": "panel-binding.json",
            "resolved_binding": binding,
        },
        "capture": {
            "dataset_id": "fidelity--selftest.root.hidden",
            "panel_id": binding["panel"]["id"],
            "dataset_name": "selftest root hidden",
            "author": "selftest",
            "dataset_repository": destination,
            "publish_root_to": destination,
            "dataset_license": "mit",
            "weights_license": None,
            "form": "hidden",
            "schedule": "layer-outer",
            "device": "cuda",
            "dtype": "bfloat16",
            "engine": "hf-transformers",
            "preview_of": None,
            "race": False,
            "replay_device": "numpy",
            "replay_dtype": "float32",
            "vocab_chunk": 8192,
            "replay": {
                "device": "numpy", "dtype": "float32",
                "vocab_chunk": 8192,
            },
            "root_protocol": {
                "schedule": "two-fresh-process-qualification",
                "fresh_processes": 2,
                "run_count_per_process": 1,
                "exact_self_comparison": True,
                "qualification_required": True,
                "canonical_publication_required": True,
                "publication_mode": "canonical-public",
            },
            "unexpected_tensor_allowlist": {
                "path": "allowlist.json",
                "artifact_sha256": "5" * 64,
                "canonical_sorted_names_sha256": q_names_sha,
            },
        },
    })
    common.write_json(job_path, job)

    def qualify(**overrides):
        values = {
            "job": job_path, "first": first, "repeat": repeat,
            "comparison": comparison_path, "first_verify": first_verify,
            "repeat_verify": repeat_verify, "first_label": "root-cold-1",
            "repeat_label": "root-cold-2",
            "imported_canonical": None,
            "out": os.path.join(tmp, "q-qualification.json"),
        }
        values.update(overrides)
        return CLI.cmd_qualify_root(argparse.Namespace(**values))

    rc = qualify()
    receipt = F.read_json(os.path.join(tmp, "q-qualification.json"))
    check("Q1  two separately sealed run_count=1 datasets + forced exact-zero "
          "comparison produce a self-sealed outer receipt",
          rc == CLI.OK and common.verify_seal(receipt)
          and receipt["captures"]["canonical"]["determinism_run_count"] == 1
          and receipt["captures"]["repeat"]["determinism_run_count"] == 1
          and receipt["comparison"]["mean_kld"] == 0.0
          and receipt["comparison"]["max_kld"] == 0.0
          and receipt["comparison"]["top1_agreement"] == 1.0
          and not np.signbit(receipt["comparison"]["mean_kld"])
          and not np.signbit(receipt["comparison"]["max_kld"])
          and receipt["dataset_repository"] == destination
          and receipt["destination_repository"] == destination
          and receipt["job_contract"]["dataset_id"]
          == "fidelity--selftest.root.hidden"
          and receipt["comparator"]["replay_backend"] == "numpy:cpu:float32"
          and receipt["comparator"]["force_compute_agreed"] is True,
          json.dumps(receipt)[:300])
    forged = json.loads(json.dumps(receipt))
    alternate_job = json.loads(json.dumps(job))
    alternate_job["target"]["revision"] = "0" * 40
    alternate_job = jobcontract.finalize_job(alternate_job)
    forged["canonical_job_sha256"] = alternate_job["job_id_full"]
    forged["job_contract"] = jobcontract.root_qualification_contract(
        alternate_job)
    for identity in forged["captures"].values():
        identity["weights_revision"] = "0" * 40
    forged = common.seal(forged)
    forged_path = os.path.join(tmp, "q-forged-job.json")
    common.write_json(forged_path, forged)
    try:
        CLI._load_qualification(
            forged_path, job_path=job_path,
            dataset=first, repository=destination)
    except CLI.RootQualificationError:
        forged_refused = True
    else:
        forged_refused = False
    check("Q1b a coherently resealed alternate job cannot publish the "
          "unchanged capture", forged_refused)

    # A resumed root: cold run 1 imported from a prior attempt of the same
    # recipe. The job names the exact dataset, the controller's sealed
    # receipt proves what landed, and the qualification records the origin.
    first_manifest_raw = Path(os.path.join(first, "fidelity-dataset.json")).read_bytes()
    first_manifest = json.loads(first_manifest_raw)
    resume_identity = {
        "dataset_sha256": first_manifest["dataset_sha256"],
        "capture_content_digest": first_manifest["capture"]["capture_content_digest"],
        "dataset_manifest_file_sha256": hashlib.sha256(first_manifest_raw).hexdigest(),
        "origin": {"job_id_full": "a" * 64, "attempt_id": "b" * 24,
                   "job_file_sha256": "c" * 64},
    }
    resumed_job = json.loads(json.dumps(job))
    resumed_job["capture"]["resume_capture"] = resume_identity
    resumed_job = jobcontract.finalize_job(resumed_job)
    resumed_job_path = os.path.join(tmp, "q-resumed-job.json")
    common.write_json(resumed_job_path, resumed_job)
    import_receipt = jobcontract.build_imported_capture_receipt(
        job_id_full=resumed_job["job_id_full"], attempt_id="9" * 24,
        resume=resume_identity, archive_sha256="d" * 64, archive_bytes=4096,
        manifest_sha256="e" * 64, file_count=7, source_path="/prior/dataset",
        imported_at="2026-09-04T01:00:00Z")
    import_path = os.path.join(tmp, "q-imported-capture.json")
    common.write_json(import_path, import_receipt)
    resumed_out = os.path.join(tmp, "q-resumed-qualification.json")
    rc = qualify(job=resumed_job_path, imported_canonical=import_path,
                 out=resumed_out)
    resumed = F.read_json(resumed_out) if rc == CLI.OK else {}
    check("Q1c a resumed root qualifies with the imported cold run 1 recorded "
          "as the canonical capture's origin",
          rc == CLI.OK and common.verify_seal(resumed)
          and resumed["captures"]["canonical"]["imported_from"]["origin"]
          == resume_identity["origin"]
          and resumed["captures"]["canonical"]["imported_from"]["receipt_sha256"]
          == import_receipt["receipt_sha256"]
          and resumed["job_contract"]["resume_capture"] == resume_identity
          and "imported_from" not in resumed["captures"]["repeat"])
    check("Q1d the import receipt alone, without the job naming a resume, refuses",
          qualify(imported_canonical=import_path,
                  out=os.path.join(tmp, "q-x1.json")) != CLI.OK)
    check("Q1e the job naming a resume without the receipt refuses",
          qualify(job=resumed_job_path, out=os.path.join(tmp, "q-x2.json"))
          != CLI.OK)
    wrong = dict(import_receipt)
    wrong["dataset_sha256"] = "f" * 64
    wrong = common.seal({k: v for k, v in wrong.items() if k != "receipt_sha256"})
    wrong_path = os.path.join(tmp, "q-wrong-import.json")
    common.write_json(wrong_path, wrong)
    check("Q1f a resealed receipt naming a different dataset refuses",
          qualify(job=resumed_job_path, imported_canonical=wrong_path,
                  out=os.path.join(tmp, "q-x3.json")) != CLI.OK)
    tampered = dict(import_receipt)
    tampered["origin"] = None
    tampered_path = os.path.join(tmp, "q-tampered-import.json")
    common.write_json(tampered_path, tampered)
    check("Q1g a receipt whose bytes no longer match its seal refuses",
          qualify(job=resumed_job_path, imported_canonical=tampered_path,
                  out=os.path.join(tmp, "q-x4.json")) != CLI.OK)
    check("Q1h the resumed qualification reloads under the strict loader",
          CLI._load_qualification(resumed_out, job_path=resumed_job_path,
                                  dataset=first, repository=destination)
          is not None)
    source_license = {
        "source_path": "LICENSE", "dataset_path": "LICENSE",
        "bytes": 17, "sha256": "3" * 64,
    }
    licensed_job = json.loads(json.dumps(job))
    licensed_job["capture"]["dataset_license"] = "other"
    licensed_job["capture"]["weights_license"] = source_license
    licensed_job["target"]["weights_license"] = source_license
    licensed_job["target"]["download_manifest"].append({
        "path": "LICENSE", "bytes": source_license["bytes"]})
    licensed_job["target"]["download_manifest"].sort(
        key=lambda row: row["path"])
    licensed_job["target"]["download_bytes_total"] = sum(
        row["bytes"] for row in licensed_job["target"]["download_manifest"])
    licensed_job["target"]["download_manifest_sha256"] = common.sha256_hex(
        json.dumps(
            licensed_job["target"]["download_manifest"],
            sort_keys=True, separators=(",", ":")))
    licensed_job = jobcontract.finalize_job(licensed_job)
    licensed_job_path = os.path.join(tmp, "q-licensed-job.json")
    common.write_json(licensed_job_path, licensed_job)
    check("Q1c MIT captures cannot satisfy a source-license-bound root job",
          qualify(
              job=licensed_job_path,
              out=os.path.join(tmp, "q-license-mismatch.json"))
          == CLI.REFUSED)
    check("Q2  one path supplied twice refuses",
          qualify(repeat=first) == CLI.REFUSED)
    check("Q3  a missing second independent verification receipt refuses",
          qualify(repeat_verify=os.path.join(tmp, "absent-repeat-verify.json"))
          == CLI.REFUSED)

    nonzero = dict(comparison)

    unpublished_capture = dict(job["capture"], publish_root_to=None)
    unpublished_capture["root_protocol"] = dict(
        unpublished_capture["root_protocol"],
        canonical_publication_required=False,
        publication_mode="qualified-unpublished")
    unpublished_job = jobcontract.finalize_job(dict(
        job, capture=unpublished_capture))

    null_tokenizer = os.path.join(tmp, "q-null-tokenizer-revision")
    shutil.copytree(repeat, null_tokenizer)
    null_manifest = F.load_manifest(null_tokenizer)
    null_panel_path = os.path.join(
        null_tokenizer, null_manifest["panel"]["panel_file"])
    null_panel = F.read_json(null_panel_path)
    null_panel["tokenizer"]["revision"] = None
    F.write_json(null_panel_path, null_panel)
    null_manifest["panel"]["tokenizer"]["revision"] = None
    null_manifest["panel"]["panel_file_sha256"] = F.sha256_file(null_panel_path)
    F.write_json(os.path.join(null_tokenizer, F.MANIFEST_NAME), null_manifest)
    reseal(null_tokenizer)
    null_verify = os.path.join(tmp, "q-null-tokenizer-verify.json")
    common.write_json(null_verify, dsvalidate.validate_dataset(
        null_tokenizer, verify_tensors=True).to_dict())
    check("Q3f a captured null tokenizer revision cannot satisfy a bound panel",
          qualify(repeat=null_tokenizer, repeat_verify=null_verify)
          == CLI.REFUSED)
    unpublished_path = os.path.join(tmp, "q-job-unpublished.json")
    common.write_json(unpublished_path, unpublished_job)
    unpublished_out = os.path.join(tmp, "q-unpublished-qualification.json")
    unpublished_rc = qualify(job=unpublished_path, out=unpublished_out)
    unpublished_receipt = F.read_json(unpublished_out)
    check("Q3b a root may qualify without authorizing publication",
          unpublished_rc == CLI.OK
          and unpublished_receipt["dataset_repository"] == destination
          and unpublished_receipt["destination_repository"] is None)

    wrong_panel = json.loads(json.dumps(job))
    wrong_panel["panel"]["resolved_binding"]["panel"]["id"] = "panel--wrong"
    common.write_json(job_path, jobcontract.finalize_job(wrong_panel))
    check("Q3c a job-bound wrong panel identity refuses",
          qualify() == CLI.REFUSED)

    wrong_dataset = json.loads(json.dumps(job))
    wrong_dataset["capture"]["dataset_id"] = "fidelity--other.root.hidden"
    common.write_json(job_path, jobcontract.finalize_job(wrong_dataset))
    check("Q3d a dataset id different from the canonical job refuses",
          qualify() == CLI.REFUSED)

    tampered_job = json.loads(json.dumps(job))
    tampered_job["target"]["revision"] = "b" * 40
    common.write_json(job_path, tampered_job)
    check("Q3e a self-identity-tampered job refuses qualification",
          qualify() == CLI.REFUSED)
    common.write_json(job_path, job)
    nonzero["metric"] = dict(nonzero["metric"], value=0.0001)
    nonzero = F.seal_receipt(nonzero)
    nonzero_path = os.path.join(tmp, "q-nonzero.json")
    F.write_json(nonzero_path, nonzero)
    check("Q4  a nonzero reproduction comparison refuses qualification",
          qualify(comparison=nonzero_path) == CLI.REFUSED)

    mismatch = os.path.join(tmp, "q-one-bit-different")
    build_dataset(mismatch, seed=92, run_name="root-cold-2", cold_run="root-cold-2",
                  dataset_repository=destination, weights_repository=weights,
                  qualification_contract=True)
    expect_refusal(
        "Q4b --self-compare refuses distinct captures with changed content",
        lambda: dscompare.compare(first, mismatch,
                                  os.path.join(tmp, "q-mismatch-comparison"), {
                                      "self_compare": True,
                                      "force_compute": True,
                                      "device": "cpu",
                                      "replay_device": "numpy",
                                      "replay_dtype": "float32",
                                      "vocab_chunk": 8192,
                                      "reference_label": "root-cold-1",
                                      "candidate_label": "root-cold-2",
                                      "verify_tensors": True,
                                  }),
        code="not_a_self_compare")
    mismatch_verify = os.path.join(tmp, "q-mismatch-verify.json")
    common.write_json(mismatch_verify, dsvalidate.validate_dataset(
        mismatch, verify_tensors=True).to_dict())
    one_bit = os.path.join(tmp, "q-one-bit")
    shutil.copytree(repeat, one_bit)
    victim = os.path.join(one_bit, "capture", "hidden_0000.safetensors")
    with open(victim, "r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        value = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([value[0] ^ 0x01]))
    one_bit_verify = os.path.join(tmp, "q-one-bit-verify.json")
    common.write_json(one_bit_verify, dsvalidate.validate_dataset(
        one_bit, verify_tensors=True).to_dict())
    check("Q5b one flipped payload bit cannot qualify for publication",
          qualify(repeat=one_bit, repeat_verify=one_bit_verify) == CLI.REFUSED)

    check("Q5  changed capture content cannot borrow the exact-zero receipt",
          qualify(repeat=mismatch, repeat_verify=mismatch_verify) == CLI.REFUSED)

    swapped = os.path.join(tmp, "q-swapped-repositories")
    build_dataset(swapped, seed=91, run_name="root-cold-2", cold_run="root-cold-2",
                  dataset_repository=weights, weights_repository=destination,
                  qualification_contract=True)
    swapped_verify = os.path.join(tmp, "q-swapped-verify.json")
    common.write_json(swapped_verify, dsvalidate.validate_dataset(
        swapped, verify_tensors=True).to_dict())
    check("Q6  swapped target/destination repository identities refuse",
          qualify(repeat=swapped, repeat_verify=swapped_verify) == CLI.REFUSED)


def main():
    tmp = tempfile.mkdtemp(prefix="fidelity-dataset-selftest-")
    try:
        base = section_format(tmp)
        section_panel(tmp, base)
        section_head(tmp)
        section_lane(tmp)
        section_interop(tmp)
        section_real(tmp)
        section_hostile_fetch(tmp)
        section_root_qualification(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nselftest_fidelity_dataset: %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
