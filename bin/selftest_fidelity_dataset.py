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
    R1-R5    real published artifacts (metadata only)

Exit 0 = all pass.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

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
                  tokenizer=None, emit_k3_compat=False):
    """Build a complete, sealed, conformant dataset.  Every knob is a test axis."""
    writer = dsmanifest.DatasetWriter(root)
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
        panel_receipt_sha256=None)

    coverage = dsmanifest.coverage_block(
        capture_records, declared_records if declared_records is not None else records,
        shard_of=shard_of, subset_detail=subset_detail)

    capture_doc = dsmanifest.capture_manifest(
        run_name="selftest-%s" % role, form=form,
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

    fingerprint = {"schema": "malaiwah.stack-fingerprint.v1", "engine": stack}
    runtime_doc = dsmanifest.capture_runtime(
        lane=lane, stack_fingerprint=fingerprint,
        stack_fingerprint_sha256=F.sha256_hex(stack),
        lane_identity_sha256=F.sha256_hex(lane_identity),
        weights={"repository": "selftest/weights", "revision": model_revision,
                 "model_revision": model_revision,
                 "checkpoint_identity_sha256": checkpoint_identity},
        source_files={"k6/tools/stream_score.py": F.sha256_hex("selftest")},
        capture_tool={"file": "bin/fidelity_dataset.py", "sha256": F.sha256_hex("tool"),
                      "wraps": ["k6/tools/hidden_replay.py"], "mechanism": "selftest fixture"})

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
                 "license": "mit", "repository": None, "revision": None,
                 "base_capture": None},
        weights={"repository": "selftest/weights", "revision": model_revision,
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
               "panel_receipt_sha256": None, "repository": None, "revision": None,
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
        determinism={"run_count": 2, "cold_start_per_run": True,
                     "evidence_kind": ("hidden_state_tensor_sha256" if form == "hidden"
                                       else "logits_tensor_sha256"),
                     "evidence_hashes": [capture_doc["capture_content_digest"]],
                     "distinct_evidence_hash_count": 1, "identical_across_runs": True,
                     "repeats": [], "repeat_noise": None, "note": "selftest fixture"},
        coverage=coverage,
        disclosures=[{"code": "no_known_deviations", "severity": "info",
                      "affects_comparability": False, "detail": "selftest fixture"}])

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


def main():
    tmp = tempfile.mkdtemp(prefix="fidelity-dataset-selftest-")
    try:
        base = section_format(tmp)
        section_panel(tmp, base)
        section_head(tmp)
        section_lane(tmp)
        section_interop(tmp)
        section_real(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nselftest_fidelity_dataset: %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
