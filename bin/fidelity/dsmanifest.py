"""Builders for a conformant `malaiwah.fidelity-dataset.v1` tree.

This module lays out the directory, computes every digest, writes the four
sealed sub-manifests, writes `checksums.txt`, and seals the top-level manifest
last -- in that order, because the seal chain is acyclic only in that order
(spec section 5.4):

    sub-manifests -> their file digests -> checksums.txt -> checksums digest
    -> self-blanked dataset_sha256.

Stdlib only.  No torch, no numpy at import: a dataset is buildable and
CI-checkable on a stock py3.9 interpreter with no GPU.

The one thing this module deliberately does NOT do is run a model.  Capture is
`k6/tools/stream_score.py` (frozen) wrapped by `k6/tools/hidden_replay.py`; this
module turns the tree those produce into a portable dataset.
"""

from __future__ import annotations

import datetime
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import dsformat as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_REGISTRY_TOOLS = os.path.join(_REPO, "registry", "tools")


def registry_lib():
    """The registry's OWN scope_digest / comparability key, imported never copied."""
    if _REGISTRY_TOOLS not in sys.path:
        sys.path.insert(0, _REGISTRY_TOOLS)
    import registry_lib  # noqa: WPS433

    return registry_lib


def utc_now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

NATIVE_TENSOR_CLASSES = (
    "embed_tokens", "attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.down",
    "moe.experts", "norm", "lm_head",
)


def native_scope(kv_cache_dtype: str = "bf16", dtype: str = "bf16",
                 bits: int = 16) -> Dict[str, Any]:
    """ROOT-1's all-native scope, with the registry's own digest."""
    assignments = [
        {"tensor_class": name, "treatment": "native", "format": dtype,
         "bits_per_weight": bits, "layer_range": None}
        for name in NATIVE_TENSOR_CLASSES
    ]
    scope = {
        "policy": "native",
        "head_policy": "native",
        "kv_cache_dtype": kv_cache_dtype,
        "assignments": assignments,
    }
    scope["scope_digest"] = registry_lib().scope_digest(scope)
    return scope


def scope_block(assignments: Sequence[Dict[str, Any]], head_policy: str,
                kv_cache_dtype: str, policy: str) -> Dict[str, Any]:
    scope = {
        "policy": policy,
        "head_policy": head_policy,
        "kv_cache_dtype": kv_cache_dtype,
        "assignments": [dict(a) for a in assignments],
    }
    scope["scope_digest"] = registry_lib().scope_digest(scope)
    return scope


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def tensor_record(
    *,
    index: int,
    filename: str,
    abs_path: str,
    key: str,
    dtype: str,
    shape: Sequence[int],
    scored_rows: int,
    token_ids_json_sha256: str,
    token_ids_sha256_legacy: Optional[str] = None,
    attention_mask_sha256: Optional[str] = None,
    window_id: Optional[str] = None,
    role: Optional[str] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    allocation_stratum: Optional[str] = None,
    semantic_class: Optional[str] = None,
    source_cluster_id: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One `capture/manifest.json` record (spec section 6.2).

    Three index aliases on purpose: kimi-k3's comparator resolves a record index
    by trying `context_index`, then `window_index`, then `index`.  Emitting all
    three costs a few bytes and makes our capture readable by his unmodified
    tool.
    """
    return {
        "index": int(index),
        "context_index": int(index),
        "window_index": int(index),
        "window_id": window_id,
        "file": filename,
        "key": key,
        "dtype": dtype,
        "shape": [int(v) for v in shape],
        "size_bytes": os.path.getsize(abs_path),
        "sha256": F.file_sha256(abs_path),
        "payload_sha256": F.payload_sha256(abs_path),
        "tensor_content_sha256": F.tensor_content_sha256(abs_path, key),
        "token_ids_json_sha256": token_ids_json_sha256,
        "token_ids_sha256_legacy": token_ids_sha256_legacy,
        "attention_mask_sha256": attention_mask_sha256,
        "prediction_positions": int(scored_rows),
        "scored_rows": int(scored_rows),
        "role": role,
        "domain": domain,
        "document_id": document_id,
        "allocation_stratum": allocation_stratum,
        "semantic_class": semantic_class,
        "source_cluster_id": source_cluster_id,
        "elapsed_seconds": elapsed_seconds,
        "request_id": request_id,
        # REC-4: false means the host-local chunk keys were STRIPPED (PATH-2),
        # not merely absent.  Field name adopted from kimi-k3.
        "raw_chunks_retained": False,
    }


def panel_record(
    *,
    index: int,
    token_file: str,
    token_ids: Sequence[int],
    prediction_positions: int,
    window_id: Optional[str] = None,
    attention_mask_file: Optional[str] = None,
    attention_mask_sha256: Optional[str] = None,
    role: Optional[str] = None,
    domain: Optional[str] = None,
    document_id: Optional[str] = None,
    allocation_stratum: Optional[str] = None,
    semantic_class: Optional[str] = None,
    source_cluster_id: Optional[str] = None,
    partition: Optional[str] = None,
    sentinel: bool = False,
) -> Dict[str, Any]:
    ids = list(token_ids)
    return {
        "index": int(index),
        "context_index": int(index),
        "window_index": int(index),
        "window_id": window_id,
        "token_file": token_file,
        "token_ids_json_sha256": F.token_ids_json_sha256(ids),
        "token_ids_sha256_legacy": F.token_ids_json_sha256_legacy(ids),
        # PANEL-D4: a costless eyeball check that does not require downloading
        # tokens/.  Adopted from Festr's 32x2048 suite.
        "token_ids_first16": ids[:16],
        "token_ids_last16": ids[-16:],
        "num_tokens": len(ids),
        "prediction_positions": int(prediction_positions),
        "attention_mask_file": attention_mask_file,
        "attention_mask_sha256": attention_mask_sha256,
        "role": role,
        "domain": domain,
        "document_id": document_id,
        "allocation_stratum": allocation_stratum,
        "semantic_class": semantic_class,
        "source_cluster_id": source_cluster_id,
        "partition": partition,
        "sentinel": bool(sentinel),
    }


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def panel_binding(
    *,
    panel_id: Optional[str],
    name: str,
    records: Sequence[Dict[str, Any]],
    context_length: int,
    tokenizer: Dict[str, Any],
    repository: Optional[str] = None,
    revision: Optional[str] = None,
    panel_receipt_sha256: Optional[str] = None,
    panel_token_sha256_legacy: Optional[str] = None,
    scoring_window: Optional[Dict[str, Any]] = None,
    contamination: Optional[Dict[str, Any]] = None,
    strata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ordered = sorted(records, key=lambda r: int(r["index"]))
    per_record = [r["token_ids_json_sha256"] for r in ordered]
    doc = {
        "schema": F.PANEL_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "panel_id": panel_id,
        "name": name,
        "suite_token_hash_sha256": F.suite_token_hash_sha256(per_record),
        "panel_token_sha256_legacy": panel_token_sha256_legacy
        or F.suite_token_hash_sha256_legacy(per_record),
        "token_digest_algorithm": {
            "per_record": "sha256(json.dumps(ids, separators=(',',':')).encode('utf-8'))",
            "aggregate": "sha256('\\n'.join(per_record_hex).encode('ascii'))",
            "legacy_per_record": "sha256(json.dumps(ids).encode('utf-8'))",
            "legacy_aggregate": "sha256(''.join(per_record_hex).encode('utf-8'))",
        },
        "repository": repository,
        "revision": revision,
        "panel_receipt_sha256": panel_receipt_sha256,
        "contexts": len(ordered),
        "context_length": int(context_length),
        "positions_per_context": int(ordered[0]["prediction_positions"]) if ordered else 0,
        "scored_positions_total": sum(int(r["prediction_positions"]) for r in ordered),
        # PANEL-D3: part of panel IDENTITY, not a comparator flag.
        "scoring_window": scoring_window or {
            "score_from": 0,
            "windowed": False,
            "min_left_context_tokens": 1,
            "dropped_positions_total": 0,
            "policy": "every prediction position of every window is scored; nothing is dropped",
        },
        "tokenizer": dict(tokenizer),
        "contamination": contamination or {
            "checked": False,
            "method": "not established",
            "hits": None,
            "receipt": None,
        },
        "strata": strata or {},
        "records": ordered,
    }
    return F.seal_receipt(doc)


def capture_manifest(
    *,
    run_name: str,
    form: str,
    semantic_point: str,
    tensor_key: str,
    dtype: str,
    dtype_lossless: bool,
    vocab_size: int,
    context_length: int,
    records: Sequence[Dict[str, Any]],
    hidden_width: Optional[int] = None,
    coverage: Optional[Dict[str, Any]] = None,
    runtime_manifest: str = "../runtime/capture-runtime.json",
    runtime_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    ordered = sorted(records, key=lambda r: int(r["index"]))
    doc = {
        "schema": F.CAPTURE_MANIFEST_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "created_utc": utc_now(),
        "run_name": run_name,
        "form": form,
        "semantic_point": semantic_point,
        "tensor_key": tensor_key,
        "dtype": dtype,
        "dtype_lossless": bool(dtype_lossless),
        "hidden_width": hidden_width,
        "vocab_size": int(vocab_size),
        "context_length": int(context_length),
        "scored_rows_per_context": int(ordered[0]["scored_rows"]) if ordered else 0,
        "total_scored_rows": sum(int(r["scored_rows"]) for r in ordered),
        "total_size_bytes": sum(int(r["size_bytes"]) for r in ordered),
        "capture_content_digest": F.capture_content_digest(ordered),
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "coverage": coverage,
        "records": ordered,
    }
    return F.seal_receipt(doc)


def head_identity(
    *,
    present: bool,
    tensor_key: str,
    shape: Sequence[int],
    dtype: str,
    file_sha256: Optional[str],
    tensor_content_sha256: Optional[str],
    quantized: Optional[bool],
    source: str,
    applied_in_capture: bool,
    file: Optional[str] = None,
    bits: Optional[int] = None,
    final_norm: Optional[Dict[str, Any]] = None,
    equality_receipt: Optional[Dict[str, Any]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """HEAD-IDENT: content is normative; `raw_tensor_sha256` is the required
    byte-equal alias carrying Festr's field name so his tooling reads our head."""
    doc = {
        "schema": F.HEAD_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "present": bool(present),
        "file": file,
        "tensor_key": tensor_key,
        "compat_tensor_key": "weight",
        "shape": [int(v) for v in shape],
        "dtype": dtype,
        "bias": None,
        "file_sha256": file_sha256,
        "raw_tensor_sha256": tensor_content_sha256,
        "tensor_content_sha256": tensor_content_sha256,
        "quantized": quantized,
        "bits": bits,
        "source": source,
        "applied_in_capture": bool(applied_in_capture),
        "final_norm": final_norm,
        "equality_receipt": equality_receipt,
        "note": note,
    }
    return F.seal_receipt(doc)


def capture_runtime(
    *,
    lane: str,
    stack_fingerprint: Dict[str, Any],
    stack_fingerprint_sha256: str,
    weights: Dict[str, Any],
    lane_identity_sha256: Optional[str] = None,
    container: Optional[Dict[str, Any]] = None,
    runtime_environment: Optional[Dict[str, str]] = None,
    source_files: Optional[Dict[str, str]] = None,
    capture_tool: Optional[Dict[str, Any]] = None,
    upstream_receipts: Optional[Sequence[Dict[str, Any]]] = None,
    lane_inferred: bool = False,
) -> Dict[str, Any]:
    doc = {
        "schema": F.RUNTIME_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "lane": lane,
        "lane_inferred": bool(lane_inferred),
        "lane_identity_sha256": lane_identity_sha256,
        "lane_identity_inputs": [
            "torch_version", "cuda_runtime_version", "device_name", "grouped_mm_kernel",
            "numeric_policy", "attention_backend", "experts_implementation",
            "parallelism", "ep_emulate", "reduce_order",
        ],
        "stack_fingerprint": stack_fingerprint,
        "stack_fingerprint_sha256": stack_fingerprint_sha256,
        # Three blocks adopted from kimi-k3 (spec section 9.2).
        "container": container or {
            "image_digest": None, "image_reference": None, "image_repository_digest": None,
        },
        "runtime_environment": runtime_environment or {},
        "source_files": source_files or {},
        "capture_tool": capture_tool,
        "weights": weights,
        "upstream_receipts": list(upstream_receipts or []),
    }
    return F.seal_receipt(doc)


def coverage_block(records: Sequence[Dict[str, Any]], declared_records: int,
                   shard_of: Optional[Dict[str, int]] = None,
                   subset_detail: Optional[str] = None) -> Dict[str, Any]:
    indices = sorted(int(r["index"]) for r in records)
    complete = (len(indices) == declared_records and indices == list(range(declared_records)))
    missing = []
    if not complete:
        have = set(indices)
        missing = [i for i in range(declared_records) if i not in have][:8]
    return {
        "declared_records": int(declared_records),
        "present_records": len(indices),
        "complete": complete,
        "index_range": [indices[0], indices[-1]] if indices else [0, -1],
        "shard_of": shard_of,
        "subset_detail": subset_detail,
        "missing_indices_sample": missing,
    }


DIVERGENCES = [
    {"id": "D-1", "field": "head",
     "reason": "every capture declares its OWN head identity by tensor content digest; the "
               "comparator refuses a hidden-form comparison across differing heads."},
    {"id": "D-2", "field": "runtime.lane",
     "reason": "lane is required and gated at compare time so registry BIAS-006 is checked "
               "before a number exists, not at submission."},
    {"id": "D-3", "field": "panel.scoring_window",
     "reason": "scoring_window is part of panel identity, not a comparator flag: a "
               "score_from=1024 number is a different panel, not a variant."},
    {"id": "D-4", "field": "comparison_kind",
     "reason": "self-compare is a first-class mode asserting exactly 0.0."},
    {"id": "D-5", "field": "estimator.accumulation_dtype",
     "reason": "required, fp64 default; an fp32 estimator lands in a different comparability "
               "class on the same data and that must be visible."},
    {"id": "D-6", "field": "coverage",
     "reason": "declared vs present record counts with shard_of; fixes our own manifest that "
               "claimed 5,120 captures in a 512-file repository."},
    {"id": "D-7", "field": "panel",
     "reason": "the panel is a separately referenceable object, which is what makes a candidate "
               "capture publishable standalone."},
    {"id": "D-8", "field": "capture.lossy_codec",
     "reason": "lets a llama.cpp .kld be ingested without laundering a lossy capture as exact."},
    {"id": "D-9", "field": "capture.records[].attention_mask_sha256",
     "reason": "our packed and streaming lanes vary mask construction; a single-request capture "
               "path makes it invariant and has no equivalent field."},
    {"id": "D-10", "field": "capture.head_separable",
     "reason": "hidden form is the default; logit form must declare head_separable:false with a "
               "reason."},
]


def interop_block(k3_compat: bool = False,
                  adapted_from: Optional[Dict[str, Any]] = None,
                  note: Optional[str] = None) -> Dict[str, Any]:
    return {
        "compatible_with": ["kimi-k3-distribution-fidelity/1"],
        "k3_compat_emitted": bool(k3_compat),
        "adapted_from": adapted_from,
        "divergences": list(DIVERGENCES),
        "note": note,
    }


def seal_block(checksums_sha256: str, external_anchor: Optional[str] = None) -> Dict[str, Any]:
    return {
        "method": F.SEAL_METHOD,
        "seal_field": F.SEAL_FIELD,
        "checksums_file": F.CHECKSUMS_NAME,
        "checksums_format": "sha256sum",
        "checksums_sha256": checksums_sha256,
        "covers": [
            "every published file except checksums.txt and fidelity-dataset.json, "
            "via checksums.txt",
            "fidelity-dataset.json itself, via the self-blanked seal",
            "every capture tensor a second time by CONTENT, via capture.capture_content_digest",
        ],
        "excludes": list(F.SEAL_EXCLUDES),
        "external_anchor": external_anchor,
    }


# ---------------------------------------------------------------------------
# The finalizer
# ---------------------------------------------------------------------------


def finalize(root: str, manifest: Dict[str, Any],
             external_anchor: Optional[str] = None) -> Dict[str, Any]:
    """Write checksums.txt over the tree, then seal and write the manifest.

    MUST be called last: `checksums.txt` covers every file except itself and the
    manifest, so anything written afterwards is an `unlisted_file` refusal.
    """
    checksums_sha = F.write_checksums(root)
    manifest = dict(manifest)
    manifest["seal"] = seal_block(checksums_sha, external_anchor)
    manifest[F.SEAL_FIELD] = ""
    manifest = F.seal_manifest(manifest)
    F.write_json(os.path.join(root, F.MANIFEST_NAME), manifest)
    return manifest


def write_sub(root: str, relpath: str, doc: Dict[str, Any]) -> Tuple[str, str]:
    """Write a sealed sub-manifest; return (relpath, its file digest)."""
    full = os.path.join(root, relpath)
    F.write_json(full, doc)
    return relpath, F.sha256_file(full)


# ---------------------------------------------------------------------------
# Streaming tap (the suite-scale work item)
# ---------------------------------------------------------------------------


def flushing_tap(write_one: Callable[[int, Any], None]) -> Callable[[int, Any], None]:
    """Flush-per-window tap factory.

    `k6/tools/hidden_replay.py`'s tap accumulates every window in CPU RAM before
    writing: 419 MiB for the 25-window panel is fine, but a 400-context shard is
    ~6.6 GiB and the full 10.48M-position suite is ~86 GB.  Anything past panel
    scale must write and drop.  Passing this as the tap's `flush_fn` keeps that
    change out of `stream_score.py`, which is frozen.
    """

    def tap(index: int, tensor: Any) -> None:
        write_one(index, tensor)
        del tensor

    return tap


# ---------------------------------------------------------------------------
# DatasetWriter -- lays out the tree in seal-chain order
# ---------------------------------------------------------------------------


class DatasetWriter(object):
    """Assemble a conformant dataset directory.

    Usage is deliberately linear, because the seal chain is:

        add files -> write sub-manifests -> write checksums.txt -> seal manifest

    Anything written after `finish()` is an `unlisted_file` refusal, which is
    the point: `validation/structural-validation.json` is therefore written
    BEFORE the seal (by `finish`), not by a later `validate` run.  A third party
    validating a downloaded dataset writes their report outside it.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.tensor_records: List[Dict[str, Any]] = []
        self.panel_records: List[Dict[str, Any]] = []

    # -- files ---------------------------------------------------------------
    def _write_bytes(self, relpath: str, payload: bytes) -> str:
        full = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(payload)
        return full

    def add_token_file(self, index: int, token_ids: Sequence[int]) -> str:
        """Compact JSON, so the file itself is the preimage of the normative digest."""
        import json as _json

        relpath = "panel/tokens/context-%04d.json" % index
        self._write_bytes(relpath, _json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8"))
        return relpath

    def add_mask_file(self, index: int, payload: bytes) -> Tuple[str, str]:
        relpath = "panel/masks/context-%04d.npy" % index
        full = self._write_bytes(relpath, payload)
        return relpath, F.sha256_file(full)

    def add_capture_tensor(self, index: int, payload: bytes, form: str = "hidden") -> str:
        name = ("hidden_%04d.safetensors" if form == "hidden" else "logits_%04d.safetensors") % index
        relpath = "capture/" + name
        self._write_bytes(relpath, payload)
        return relpath

    def copy_capture_tensor(self, index: int, src: str, form: str = "hidden") -> str:
        with open(src, "rb") as handle:
            return self.add_capture_tensor(index, handle.read(), form)

    def add_head_payload(self, payload: bytes) -> str:
        relpath = "head/weight.safetensors"
        self._write_bytes(relpath, payload)
        return relpath

    def add_file(self, relpath: str, payload: bytes) -> str:
        F.check_relpath(relpath, owner="DatasetWriter.add_file",
                        allow_parent=relpath.startswith("compat/"))
        self._write_bytes(relpath, payload)
        return relpath

    def add_readme(self, text: str) -> str:
        return self.add_file("README.md", text.encode("utf-8"))

    # -- finish --------------------------------------------------------------
    def finish(self, manifest: Dict[str, Any], panel_doc: Dict[str, Any],
               capture_doc: Dict[str, Any], head_doc: Optional[Dict[str, Any]],
               runtime_doc: Dict[str, Any],
               external_anchor: Optional[str] = None,
               validation_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _, panel_sha = write_sub(self.root, manifest["panel"]["panel_file"], panel_doc)
        manifest["panel"]["panel_file_sha256"] = panel_sha
        _, runtime_sha = write_sub(self.root, manifest["runtime"]["file"], runtime_doc)
        manifest["runtime"]["file_sha256"] = runtime_sha
        if head_doc is not None and manifest["head"].get("head_json"):
            write_sub(self.root, manifest["head"]["head_json"], head_doc)
        # The capture manifest names the runtime manifest by relative path AND
        # digest -- adopted from kimi-k3, including his rule that an absolute
        # path is refused -- so it must be sealed after the runtime file exists.
        capture_doc = dict(capture_doc)
        capture_doc["runtime_manifest_sha256"] = runtime_sha
        capture_doc["receipt_sha256"] = ""
        capture_doc = F.seal_receipt(capture_doc)
        _, capture_sha = write_sub(self.root, manifest["capture"]["manifest_file"], capture_doc)
        manifest["capture"]["manifest_file_sha256"] = capture_sha
        if validation_report is not None:
            F.write_json(os.path.join(self.root, "validation/structural-validation.json"),
                         validation_report)
        return finalize(self.root, manifest, external_anchor)


def top_manifest(
    *,
    dataset: Dict[str, Any],
    weights: Dict[str, Any],
    scope: Dict[str, Any],
    panel: Dict[str, Any],
    capture: Dict[str, Any],
    head: Dict[str, Any],
    runtime: Dict[str, Any],
    determinism: Dict[str, Any],
    coverage: Dict[str, Any],
    interop: Optional[Dict[str, Any]] = None,
    disclosures: Optional[Sequence[Dict[str, Any]]] = None,
    upstream_receipts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "schema": F.DATASET_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "dataset_sha256": "",
        "created_utc": utc_now(),
        "dataset": dataset,
        "weights": weights,
        "scope": scope,
        "panel": panel,
        "capture": capture,
        "head": head,
        "runtime": runtime,
        "determinism": determinism,
        "coverage": coverage,
        "seal": seal_block("0" * 64),
        "interop": interop or interop_block(),
        "upstream_receipts": list(upstream_receipts or []),
        "disclosures": list(disclosures or []),
    }
