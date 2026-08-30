"""Adapters: foreign capture artifacts -> `malaiwah.fidelity-dataset.v1`.

Four sources, all real:

    k3v1                 festr2/kimi-k3-distribution-fidelity-1024x2048-v1
    k3v0-window          festr2/kimi-k3-full-mxfp4-kld-reference-32x2048
    malaiwah-serving-v2  our own `glm53flash-fidelity-capture/2`
    llamacpp-kld         llama.cpp `--kl-divergence-base` binary

Two rules the adapters never break:

  * **No tensor rewriting.**  BF16 `[rows, hidden]` safetensors with tensor key
    `hidden_states` is already our native form, which is exactly why the k3
    adapter is pure metadata translation.
  * **No fabrication.**  Every field the adapter had to synthesize is named in
    `interop.adapted_from.inferred_fields[]`, and every entry in that array
    forces `comparability.class = advisory` at compare time.

When the tensors are not present locally (the normal case for a 30 GB foreign
artifact) the adapter emits a TRANSLATION REPORT instead of a sealed dataset:
the panel binding, the head identity, the lane inference, the coverage truth and
the outstanding work.  A sealed dataset requires the bytes; saying so is better
than inventing a digest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import dsformat as F
from . import dsmanifest, dsvalidate, k3compat

ADAPTER_REPORT_SCHEMA = "malaiwah.fidelity-adapter-report.v1"

SOURCES = ("k3v1", "k3v0-window", "malaiwah-serving-v2", "llamacpp-kld")


class AdapterError(Exception):
    pass


def _report(source: str, inferred: Sequence[str], **extra: Any) -> Dict[str, Any]:
    doc = {
        "schema": ADAPTER_REPORT_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "source_format": source,
        "adapter": "bin/fidelity/dsadapt.py",
        "inferred_fields": sorted(set(inferred)),
        "consequence": "every entry in inferred_fields forces comparability.class = advisory "
                       "at compare time; the adapter never fabricates a digest.",
    }
    doc.update(extra)
    return F.seal_receipt(doc)


# ---------------------------------------------------------------------------
# kimi-k3 (both generations)
# ---------------------------------------------------------------------------


def _k3_paths(root: str) -> Dict[str, Optional[str]]:
    """Locate the k3 metadata files, tolerating both repo layouts."""
    found: Dict[str, Optional[str]] = {}
    for key, candidates in (
        ("manifest", ("manifest.json",)),
        ("suite", ("suite-manifest.json",)),
        ("runtime", ("capture-runtime.json",)),
        ("hidden", ("reference-hidden/manifest.json",)),
        ("logit", ("ref/manifest.json",)),
        ("head", ("lm-head/manifest.json",)),
        ("validation", ("validation/artifact-validation.json",)),
        ("qualification", ("validation/hidden-replay-qualification.json",)),
        ("checksums", ("checksums.txt",)),
    ):
        found[key] = None
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                found[key] = path
                break
    return found


def adapt_k3(root: str, out_dir: str, source: str = "k3v1",
             tokens_dir: Optional[str] = None,
             recompute_content_digests: bool = False,
             emit_dataset: bool = False,
             emit_k3_compat: bool = False,
             dataset_id: str = "fidelity--adapted.kimi-k3",
             name: str = "adapted kimi-k3 capture",
             role: str = "root", lane: str = "other",
             limit: Optional[int] = None, link: bool = True) -> Dict[str, Any]:
    """Translate a kimi-k3 artifact.  Metadata only unless the tensors are local."""
    paths = _k3_paths(root)
    if not paths["suite"]:
        raise AdapterError("no suite-manifest.json under %s; this is not a kimi-k3 artifact"
                           % root)
    suite = F.read_json(paths["suite"])
    top = F.read_json(paths["manifest"]) if paths["manifest"] else {}
    inferred: List[str] = []

    # -- record list: contexts[] (k3v1) or windows[] (k3v0) ------------------
    records = suite.get("contexts") or suite.get("windows") or []
    if not records:
        raise AdapterError("suite manifest carries neither contexts[] nor windows[]")
    tensor_manifest_path = paths["hidden"] or paths["logit"]
    tensor_manifest = F.read_json(tensor_manifest_path) if tensor_manifest_path else {}
    tensor_records = (tensor_manifest.get("contexts") or tensor_manifest.get("windows") or [])
    by_index = {}
    for row in tensor_records:
        # His comparator resolves an index by trying context_index, then
        # window_index, then index -- so we read it the same way.
        index = row.get("context_index")
        if index is None:
            index = row.get("window_index")
        if index is None:
            index = row.get("index")
        by_index[int(index)] = row

    form = "hidden" if paths["hidden"] else "logit"
    semantic_point = tensor_manifest.get("semantic_point")
    if not semantic_point:
        semantic_point = ("after_final_rmsnorm_before_lm_head" if form == "hidden"
                          else "lm_head_output_before_sampling")
        inferred.append("capture.semantic_point")

    # -- panel ---------------------------------------------------------------
    tokens_root = tokens_dir or root
    panel_rows = []
    token_digest_source = "recomputed_from_tokens"
    for row in records:
        index = row.get("context_index", row.get("index"))
        token_file = row.get("token_file")
        digest = row.get("token_ids_json_sha256")
        local = os.path.join(tokens_root, token_file) if token_file else None
        if local and os.path.isfile(local):
            ids = F.read_json(local)
            digest = F.token_ids_json_sha256(ids)
            legacy = F.token_ids_json_sha256_legacy(ids)
        else:
            legacy = None
            token_digest_source = "carried_from_source_manifest"
        panel_rows.append({
            "index": int(index),
            "context_index": int(index),
            "window_index": int(index),
            "window_id": row.get("window_id"),
            "token_file": token_file,
            "token_ids_json_sha256": digest,
            "token_ids_sha256_legacy": legacy,
            "token_ids_first16": row.get("token_ids_first16"),
            "token_ids_last16": row.get("token_ids_last16"),
            "num_tokens": row.get("num_tokens"),
            "prediction_positions": row.get("scored_row_end_exclusive")
            or suite.get("scored_positions_per_context")
            or suite.get("scored_positions_per_window"),
            "attention_mask_file": None,
            # D-9: his single-request-per-context path makes the mask invariant
            # and he has no equivalent field.  We do not invent one.
            "attention_mask_sha256": None,
            "role": None,
            "domain": row.get("domain"),
            "document_id": row.get("source_content_sha256"),
            "allocation_stratum": row.get("allocation_stratum"),
            "semantic_class": row.get("semantic_class"),
            "source_cluster_id": row.get("source_cluster_id"),
            "partition": row.get("partition"),
            "sentinel": bool(row.get("sentinel", False)),
        })
    inferred.append("panel.records[].attention_mask_sha256")
    if token_digest_source == "carried_from_source_manifest":
        inferred.append("panel.records[].token_ids_sha256_legacy")

    aggregate = None
    if all(r["token_ids_json_sha256"] for r in panel_rows):
        aggregate = F.suite_token_hash_sha256(
            [r["token_ids_json_sha256"] for r in sorted(panel_rows, key=lambda x: x["index"])])
    declared = suite.get("suite_token_hash_sha256")
    aggregate_agrees = (aggregate == declared) if aggregate else None

    # -- head: HIS ARTIFACT RECORDS NO PER-CAPTURE HEAD (D-1) ----------------
    head_source = top.get("lm_head") or {}
    if not head_source and paths["head"]:
        head_source = F.read_json(paths["head"])
    head_block = {
        "present": bool(head_source),
        "file": None,
        "tensor_key": head_source.get("key") or head_source.get("tensor_key") or "weight",
        "compat_tensor_key": "weight",
        "shape": head_source.get("shape"),
        "dtype": head_source.get("dtype") or head_source.get("tensor_dtype") or "BF16",
        "bias": None,
        "file_sha256": head_source.get("file_sha256"),
        # His `raw_tensor_sha256` IS a tensor-content digest, which is why we
        # adopted his field name: it maps straight onto ours.
        "raw_tensor_sha256": head_source.get("raw_tensor_sha256"),
        "tensor_content_sha256": head_source.get("raw_tensor_sha256"),
        "quantized": None,
        "bits": None,
        "source": "unknown",
        "applied_in_capture": form == "logit",
        "final_norm": None,
        "equality_receipt": None,
        "note": "kimi-k3 records ONE artifact-level lm_head and no per-capture head identity. "
                "quantized is null (unknown), so any comparison against a non-k3 candidate is "
                "advisory (D-1).",
    }
    inferred += ["head.quantized", "head.source"]

    # -- lane ----------------------------------------------------------------
    runtime = F.read_json(paths["runtime"]) if paths["runtime"] else {}
    lane_note = "vLLM serving (TP%s) derived from capture-runtime.runtime; the registry has no " \
                "`serving` lane value, so this maps to `other` with lane_inferred true" \
                % (runtime.get("runtime", {}).get("tensor_parallel_size"))
    inferred.append("runtime.lane")

    # -- determinism ---------------------------------------------------------
    validation = F.read_json(paths["validation"]) if paths["validation"] else {}
    repeats = validation.get("sentinel_repeat_mean_kld") or {}
    determinism = {
        "run_count": 3 if repeats else 1,
        "cold_start_per_run": None,
        # OPEN ITEM 5: his per-file sha256 is a CONTAINER hash, so the imported
        # evidence cannot reach our required strength without reading tensors.
        "evidence_kind": ("hidden_state_tensor_sha256" if recompute_content_digests
                          else "run_mean_equality_only"),
        "evidence_hashes": [],
        "distinct_evidence_hash_count": None,
        "identical_across_runs": False if repeats else None,
        "repeats": [{"name": key, "dir": None, "capture_content_digest": None}
                    for key in sorted(repeats)],
        "repeat_noise": {
            "kl_canonical_to_repeat_mean": (max(repeats.values()) if repeats else None),
            "kl_repeat_to_canonical_mean": (min(repeats.values()) if repeats else None),
            "js_mean": None,
            "interpretation": "Sentinel repeats of IDENTICAL weights on the same serving stack "
                              "disagree at this magnitude; treat changes at or below it as "
                              "runtime noise unless confirmed with repeated captures.",
        },
        "note": "imported from validation/artifact-validation.json.sentinel_repeat_mean_kld. "
                "The per-file digests in the source are CONTAINER hashes, so the evidence kind "
                "is downgraded to run_mean_equality_only unless --recompute-content-digests "
                "pays for one local pass over the sentinel tensors.",
    }
    if not recompute_content_digests:
        inferred.append("determinism.evidence_kind")

    # -- coverage ------------------------------------------------------------
    declared_records = (top.get("context_count") or suite.get("context_count")
                        or suite.get("window_count") or len(panel_rows))
    present = 0
    tensor_dir = os.path.dirname(tensor_manifest_path) if tensor_manifest_path else None
    if tensor_dir:
        for row in tensor_records:
            if row.get("file") and os.path.isfile(os.path.join(tensor_dir, row["file"])):
                present += 1

    translation = _report(
        source, inferred,
        source_root=os.path.abspath(root),
        source_files={key: (os.path.relpath(value, root) if value else None)
                      for key, value in sorted(paths.items())},
        capture={
            "form": form,
            "semantic_point": semantic_point,
            "tensor_key": tensor_manifest.get("tensor_key")
            or ("hidden_states" if form == "hidden" else "logits"),
            "dtype": tensor_manifest.get("dtype", "BF16"),
            "hidden_width": tensor_manifest.get("hidden_width"),
            "vocab_size": tensor_manifest.get("vocab_size") or (head_block.get("shape") or [None])[0],
            "declared_records": declared_records,
            "records_with_local_tensors": present,
            "total_size_bytes": tensor_manifest.get("total_size_bytes"),
        },
        panel={
            "panel_id": None,
            "suite_token_hash_sha256_declared": declared,
            "suite_token_hash_sha256_recomputed": aggregate,
            "aggregate_agrees": aggregate_agrees,
            "token_digest_source": token_digest_source,
            "contexts": len(panel_rows),
            "context_length": suite.get("context_length"),
            "scored_positions_total": suite.get("total_scored_positions"),
            # PANEL-D3: derived, not assumed -- his scored rows are
            # context_length - 1 starting at row 0.
            "scoring_window": {
                "score_from": 0, "windowed": False, "min_left_context_tokens": 1,
                "dropped_positions_total": 0,
                "policy": "derived from scored_positions_per_context == context_length - 1",
            },
            "tokenizer": suite.get("tokenizer") or {"class": suite.get("tokenizer_class")},
            "records": panel_rows[:4],
            "records_total": len(panel_rows),
        },
        head=head_block,
        runtime={
            "lane": "other",
            "lane_inferred": True,
            "lane_note": lane_note,
            "container": runtime.get("container"),
            "runtime_environment": runtime.get("runtime_environment"),
            "source_files_pinned_by_content": runtime.get("source_files"),
            "stack_fingerprint": {"origin": "%s-capture-runtime" % source,
                                  "raw": runtime.get("runtime")},
        },
        determinism=determinism,
        coverage={
            "declared_records": declared_records,
            "present_records": present,
            "complete": present == declared_records and present > 0,
            "subset_detail": "metadata-only translation: the tensors were not downloaded"
            if present == 0 else None,
        },
        outstanding=([
            "download the capture tensors (or point --in at a local copy) so "
            "capture_content_digest and checksums.txt can be computed; a sealed dataset "
            "requires the bytes"] if present < declared_records else []) + [
            "the head's quantization status is not recorded anywhere in the source artifact; "
            "until a human establishes it, every comparison is advisory (D-1)",
            "no candidate captures exist in the source artifact -- it publishes compare "
            "receipts only -- so the realistic adapter target is reference-side only (12.4.9)",
        ],
    )
    os.makedirs(out_dir, exist_ok=True)
    if emit_dataset and present:
        # The report travels INSIDE the dataset, added before the seal is
        # computed. Writing it afterwards would leave a file present but not in
        # checksums.txt, which is SEAL-1(c) -- the dataset would refuse itself.
        emitted = _emit_k3_dataset(
            root, out_dir, translation=translation, suite=suite, top=top,
            tensor_manifest=tensor_manifest, by_index=by_index, panel_rows=panel_rows,
            tokens_root=tokens_root, tensor_dir=tensor_dir, head_source=head_source,
            head_paths=paths, runtime=runtime, form=form, inferred=inferred,
            dataset_id=dataset_id, name=name, role=role, lane=lane, link=link,
            limit=limit, report_name="%s-translation.json" % source,
            emit_k3_compat=emit_k3_compat)
        translation["emitted"] = emitted
    elif emit_dataset:
        F.write_json(os.path.join(out_dir, "%s-translation.json" % source), translation)
        translation["emitted"] = {
            "written": False,
            "reason": "0 of %d declared capture tensors are present locally. A sealed dataset "
                      "is made of BYTES: checksums.txt and capture_content_digest are computed "
                      "over the tensors, and the format never fabricates a digest. Point --in "
                      "at a local copy of the artifact (or fetch reference-hidden/) and re-run."
                      % declared_records,
        }
        F.write_json(os.path.join(out_dir, "%s-translation.json" % source), translation)
    else:
        F.write_json(os.path.join(out_dir, "%s-translation.json" % source), translation)
    return translation


#: Format tokens that mean "these weights were quantized".  Read out of the
#: source's OWN `checkpoint.tensor_format` string; the schema wants a boolean
#: and there is no honest null, so the inference is made explicitly and recorded
#: in inferred_fields rather than guessed silently.
QUANTIZED_FORMAT_TOKENS = ("mxfp4", "nvfp4", "fp8", "int8", "int4", "awq", "gptq",
                           "exl3", "exl2", "gguf", "q4", "q5", "q6", "q8", "bnb")


def _k3_weights_quantized(checkpoint: Dict[str, Any],
                          inferred: List[str]) -> bool:
    fmt = (checkpoint.get("tensor_format") or "").lower()
    inferred.append("weights.quantized (read from checkpoint.tensor_format %r)"
                    % (checkpoint.get("tensor_format") or None))
    return any(token in fmt for token in QUANTIZED_FORMAT_TOKENS)


def _k3_tokenizer(suite: Dict[str, Any], checkpoint: Dict[str, Any],
                  head_source: Dict[str, Any], inferred: List[str]) -> Dict[str, Any]:
    """His tokenizer block onto ours, without inventing anything.

    He names the CLASS and the vocabulary size and pins four tokenizer files by
    digest; he has no notion of `add_special_tokens` or `chat_template_applied`,
    so those stay null -- and the compare-time tokenizer gate treats a null on
    either side as unknown, never as agreement.
    """
    block = suite.get("tokenizer") or {}
    identifier = block.get("class") or checkpoint.get("repository")
    if not block:
        inferred.append("panel.tokenizer (the source names none; id from "
                        "checkpoint.repository, vocab_size from the lm_head shape)")
    vocab = block.get("vocabulary_size") or block.get("vocab_size")
    if not vocab:
        vocab = (head_source.get("shape") or [None])[0]
    return {
        "id": identifier or "unknown",
        "repository": checkpoint.get("repository"),
        "revision": (str(block.get("checkpoint_revision") or checkpoint.get("revision"))
                     if (block.get("checkpoint_revision") or checkpoint.get("revision"))
                     else None),
        "vocab_size": int(vocab or 0) or 1,
        # PANEL-D6: null is UNKNOWN, not false.  The source records neither.
        "add_special_tokens": None,
        "chat_template_applied": (False if (suite.get("chat_rendering") or {})
                                  .get("add_generation_prompt") is False else None),
        "source_files": block.get("files"),
    }


def _rewrite_tensor_key(path, old_key, new_key):
    """Rename the single tensor key in a safetensors file, in place.

    CC-03. The payload region is untouched, so tensor_content_sha256 and payload_sha256
    are unchanged; only the container digest moves. The header must stay 8-byte aligned
    or the real safetensors loader rejects the file -- dsformat's pure-python reader does
    NOT check that, so a test written only against it would pass on a file torch cannot
    open."""
    import json as _json
    import struct as _struct
    with open(path, "rb") as fh:
        raw = fh.read()
    hlen = _struct.unpack("<Q", raw[:8])[0]
    header = _json.loads(raw[8:8 + hlen])
    if old_key not in header:
        raise AdapterError("%s does not carry %r" % (path, old_key))
    header[new_key] = header.pop(old_key)
    blob = _json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)          # safetensors requires 8-byte alignment
    with open(path, "wb") as fh:
        fh.write(_struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(raw[8 + hlen:])


def _emit_k3_dataset(root, out_dir, *, translation, suite, top, tensor_manifest, by_index,
                     panel_rows, tokens_root, tensor_dir, head_source, head_paths, runtime,
                     form, inferred, dataset_id, name, role, lane, link, limit,
                     report_name, emit_k3_compat=False):
    """Turn a translated kimi-k3 artifact into a SEALED v1 dataset.

    The translation report above already resolved every value this needs; what
    this adds is the bytes -- tokens copied, tensors hardlinked, digests
    recomputed from content, and one seal over the lot.  Anything the source
    could not answer stays null and rides in `interop.adapted_from.inferred_fields`,
    which forces `comparability.class = advisory` at compare time.
    """
    checkpoint = (top.get("checkpoint") or runtime.get("checkpoint") or {})
    if role == "root":
        # ROOT-1 requires head.quantized == false and weights.quantized == false.
        # A kimi-k3 artifact records NEITHER (D-1: one artifact-level head, no
        # quantization status), and its own checkpoint string says the routed
        # experts are MXFP4. Asserting `root` here would be the format telling a
        # lie on the source's behalf.
        raise AdapterError(
            "--role root is refused for a kimi-k3 translation. ROOT-1 requires "
            "head.quantized: false and weights.quantized: false; the source artifact records "
            "no head quantization status at all (D-1) and its checkpoint declares %r. Use "
            "--role derived (the default), which carries the same numbers without asserting "
            "something the source never said."
            % (checkpoint.get("tensor_format") or "an unstated tensor format"))
    writer = dsmanifest.DatasetWriter(out_dir)
    indices = sorted(int(row["index"]) for row in panel_rows
                     if int(row["index"]) in by_index
                     and by_index[int(row["index"])].get("file")
                     and tensor_dir
                     and os.path.isfile(os.path.join(tensor_dir,
                                                     by_index[int(row["index"])]["file"])))
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        raise AdapterError("no capture tensor is readable under %s" % tensor_dir)

    panel_by_index = {int(row["index"]): row for row in panel_rows}
    panel_records, capture_records = [], []
    for index in indices:
        source_row = panel_by_index[index]
        token_rel = source_row.get("token_file")
        local_tokens = os.path.join(tokens_root, token_rel) if token_rel else None
        if not (local_tokens and os.path.isfile(local_tokens)):
            raise AdapterError(
                "context %d has no local token file (%s). The panel binding is token IDS, not a "
                "digest carried from a manifest: without the ids the dataset cannot be sealed."
                % (index, token_rel))
        ids = F.read_json(local_tokens)
        written_tokens = writer.add_token_file(index, ids)
        scored = int(by_index[index].get("shape", [0])[0]
                     or source_row.get("prediction_positions") or 0)
        panel_records.append(dsmanifest.panel_record(
            index=index, token_file=written_tokens, token_ids=ids,
            prediction_positions=scored,
            window_id=source_row.get("window_id") or "context-%04d" % index,
            role="final", domain=source_row.get("domain"),
            document_id=source_row.get("document_id"),
            allocation_stratum=source_row.get("allocation_stratum"),
            semantic_class=source_row.get("semantic_class"),
            source_cluster_id=source_row.get("source_cluster_id"),
            partition=source_row.get("partition"),
            sentinel=bool(source_row.get("sentinel"))))

        src = os.path.join(tensor_dir, by_index[index]["file"])
        dest_rel = "capture/%s_%04d.safetensors" % (
            "hidden" if form == "hidden" else "logits", index)
        dest = os.path.join(out_dir, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _place(src, dest, link)
        _, header = F.read_safetensors_header(dest)
        key = by_index[index].get("key") or tensor_manifest.get("tensor_key")
        if key not in header:
            key = sorted(k for k in header if k != "__metadata__")[0]
        record = dsmanifest.tensor_record(
            index=index, filename=os.path.basename(dest_rel), abs_path=dest, key=key,
            dtype=header[key]["dtype"], shape=header[key]["shape"],
            scored_rows=int(header[key]["shape"][0]),
            token_ids_json_sha256=F.token_ids_json_sha256(ids),
            token_ids_sha256_legacy=F.token_ids_json_sha256_legacy(ids),
            attention_mask_sha256=None,
            window_id=source_row.get("window_id") or "context-%04d" % index,
            role="final", domain=source_row.get("domain"),
            document_id=source_row.get("document_id"),
            allocation_stratum=source_row.get("allocation_stratum"),
            semantic_class=source_row.get("semantic_class"),
            source_cluster_id=source_row.get("source_cluster_id"),
            elapsed_seconds=by_index[index].get("elapsed_seconds"),
            request_id=by_index[index].get("request_id"))
        # His own container digest, carried and cross-checked rather than dropped.
        record["source_manifest_sha256"] = by_index[index].get("sha256")
        record["source_manifest_sha256_agrees"] = (
            by_index[index].get("sha256") == record["sha256"])
        capture_records.append(record)

    # -- head: copied when the payload is local, else identity only -----------
    head_rel = head_file_sha = head_content = None
    head_key = head_source.get("tensor_key") or head_source.get("key") or "weight"
    head_src = None
    declared_file = head_source.get("file") or "weight.safetensors"
    for guess in (os.path.join(root, declared_file),
                  os.path.join(os.path.dirname(head_paths.get("head") or ""),
                               os.path.basename(declared_file))
                  if head_paths.get("head") else None,
                  os.path.join(root, "lm-head", "weight.safetensors")):
        if guess and os.path.isfile(guess):
            head_src = guess
            break
    if head_src:
        head_rel = "head/weight.safetensors"
        dest = os.path.join(out_dir, head_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _place(head_src, dest, link)
        _, header = F.read_safetensors_header(dest)
        head_key = head_key if head_key in header else sorted(header)[0]
        head_content = F.tensor_content_sha256(dest, head_key)
        head_file_sha = F.sha256_file(dest)
        declared = head_source.get("raw_tensor_sha256")
        if declared and declared != head_content:
            raise AdapterError(
                "the head payload hashes to %s but the source manifest declares "
                "raw_tensor_sha256 %s" % (head_content[:16], declared[:16]))
    else:
        # D-1: identity WITHOUT the payload.  Legal, and the comparator will
        # refuse to replay through a head it does not have.
        head_content = head_source.get("raw_tensor_sha256")
        head_file_sha = head_source.get("file_sha256")
        inferred.append("head.file (payload not local)")

    head_doc = dsmanifest.head_identity(
        present=bool(head_rel), tensor_key=head_key,
        shape=head_source.get("shape") or [0, 0],
        dtype=head_source.get("dtype") or head_source.get("tensor_dtype") or "BF16",
        file_sha256=head_file_sha, tensor_content_sha256=head_content,
        # D-1: kimi-k3 records ONE artifact-level head and never says whether it
        # was quantized.  null is the honest value; it is NOT false.
        quantized=None, source="unknown", applied_in_capture=(form == "logit"),
        file=head_rel, bits=None,
        note="imported from the kimi-k3 artifact. quantized is null (unknown), so a "
             "comparison against a non-k3 candidate is advisory (D-1).")

    tokenizer = _k3_tokenizer(suite, checkpoint, head_source, inferred)
    weights_quantized = _k3_weights_quantized(checkpoint, inferred)
    panel_doc = dsmanifest.panel_binding(
        panel_id=None,
        name="kimi-k3 distribution-fidelity token suite (%d of %d contexts)"
             % (len(panel_records), translation["coverage"]["declared_records"]),
        records=panel_records,
        context_length=int(suite.get("context_length") or 0),
        tokenizer=tokenizer,
        repository=None, revision=None,
        scoring_window=translation["panel"]["scoring_window"],
        strata=suite.get("strata"))

    coverage = dsmanifest.coverage_block(
        capture_records, int(translation["coverage"]["declared_records"]),
        subset_detail=("%d of %d declared contexts have local tensors"
                       % (len(capture_records),
                          translation["coverage"]["declared_records"]))
        if len(capture_records) < translation["coverage"]["declared_records"] else None)

    capture_doc = dsmanifest.capture_manifest(
        run_name=tensor_manifest.get("run_name") or "kimi-k3 reference",
        form=form, semantic_point=translation["capture"]["semantic_point"],
        tensor_key=F.TENSOR_KEY_HIDDEN if form == "hidden" else F.TENSOR_KEY_LOGIT,
        dtype=translation["capture"]["dtype"], dtype_lossless=True,
        vocab_size=int(translation["capture"]["vocab_size"] or 0),
        context_length=int(suite.get("context_length") or 0),
        records=capture_records,
        hidden_width=translation["capture"].get("hidden_width"),
        coverage=coverage)

    fingerprint = {"schema": "malaiwah.stack-fingerprint.v1",
                   "origin": "kimi-k3-capture-runtime",
                   "raw": runtime.get("runtime")}
    runtime_doc = dsmanifest.capture_runtime(
        lane=lane, lane_inferred=True, stack_fingerprint=fingerprint,
        stack_fingerprint_sha256=F.sha256_hex(F.canonical_json(fingerprint)),
        lane_identity_sha256=None,
        container=(runtime.get("container") or None),
        runtime_environment=runtime.get("runtime_environment") or {},
        source_files=runtime.get("source_files") or {},
        weights={"repository": checkpoint.get("repository"),
                 "revision": checkpoint.get("revision"),
                 "model_revision": checkpoint.get("revision"),
                 "config_sha256": checkpoint.get("config_sha256"),
                 "index_sha256": checkpoint.get("index_sha256"),
                 "checkpoint_identity_sha256": None},
        upstream_receipts=[{"file": "upstream/%s" % os.path.basename(value),
                            "schema": "kimi-k3/%s" % key,
                            "sha256": F.sha256_file(value), "stripped_fields": []}
                           for key, value in sorted(head_paths.items())
                           if value and os.path.isfile(value)])

    os.makedirs(os.path.join(out_dir, "upstream"), exist_ok=True)
    upstream = []
    for key, value in sorted(head_paths.items()):
        if not (value and os.path.isfile(value) and value.endswith(".json")):
            continue
        stripped, names = F.strip_host_paths(F.read_json(value))
        rel = "upstream/%s" % os.path.basename(value)
        F.write_json(os.path.join(out_dir, rel), stripped)
        upstream.append({"file": rel, "schema": "kimi-k3/%s" % key,
                         "sha256": F.sha256_file(os.path.join(out_dir, rel)),
                         "stripped_fields": names})

    manifest = dsmanifest.top_manifest(
        dataset={"id": dataset_id, "name": name, "role": role,
                 "structural_status": "sealed", "qualification": None,
                 "author": {"name": "adapted from kimi-k3",
                            "role": "capture-author", "handle": None,
                            "url": "https://huggingface.co/datasets/festr2/"
                                   "kimi-k3-distribution-fidelity-1024x2048-v1",
                            "is_registry_maintainer": False},
                 "license": None, "repository": None, "revision": None,
                 # A translation is `derived`, and a derived dataset must name what
                 # it derives FROM. There is no v1 dataset upstream of a kimi-k3
                 # artifact, so the pointer is the source artifact's own manifest
                 # digest -- honest, checkable, and null-free where it can be.
                 "base_capture": {
                     "dataset_sha256": None,
                     "capture_content_digest": None,
                     "repository": "festr2/kimi-k3-distribution-fidelity-1024x2048-v1",
                     "revision": checkpoint.get("revision"),
                     "note": "translated from a kimi-k3-distribution-fidelity/1 artifact, "
                             "which predates this format and therefore has no "
                             "dataset_sha256 of its own. Its manifest.json digest is %s "
                             "and its suite token hash is %s."
                             % (F.sha256_file(head_paths["manifest"])[:16]
                                if head_paths.get("manifest") else "unavailable",
                                (suite.get("suite_token_hash_sha256") or "unavailable")[:16]),
                 }},
        weights={"repository": checkpoint.get("repository"),
                 "revision": checkpoint.get("revision"),
                 "model_revision": checkpoint.get("revision"),
                 "quantized": weights_quantized,
                 "config_sha256": checkpoint.get("config_sha256"),
                 "index_sha256": checkpoint.get("index_sha256"),
                 "checkpoint_identity_sha256": None,
                 "artifact_ref": None, "model_ref": None, "codec": None,
                 "declared_bits": None, "declared_head_bits": None},
        scope=dsmanifest.native_scope(),
        panel={"panel_id": None, "panel_file": "panel/panel.json",
               "panel_file_sha256": "0" * 64,
               "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
               "panel_token_sha256_legacy": panel_doc["panel_token_sha256_legacy"],
               "upstream_suite_token_sha256": suite.get("suite_token_hash_sha256"),
               "panel_receipt_sha256": None, "repository": None, "revision": None,
               "contexts": len(panel_records),
               "context_length": int(suite.get("context_length") or 0),
               "scored_positions_total": panel_doc["scored_positions_total"],
               "scoring_window": panel_doc["scoring_window"],
               "tokenizer": panel_doc["tokenizer"], "remap_file": None,
               "contamination": panel_doc["contamination"]},
        capture={"manifest_file": "capture/manifest.json",
                 "manifest_file_sha256": "0" * 64,
                 "capture_content_digest": capture_doc["capture_content_digest"],
                 "form": form,
                 "semantic_point": translation["capture"]["semantic_point"],
                 "tensor_key": capture_doc["tensor_key"],
                 "dtype": translation["capture"]["dtype"], "dtype_lossless": True,
                 "hidden_width": translation["capture"].get("hidden_width"),
                 "vocab_size": int(translation["capture"]["vocab_size"] or 0),
                 "head_separable": form == "hidden",
                 "head_not_separable_reason": None,
                 "records_count": len(capture_records),
                 "scored_rows_total": capture_doc["total_scored_rows"],
                 "total_size_bytes": capture_doc["total_size_bytes"],
                 "lossy_codec": None},
        head={"present": bool(head_rel), "file": head_rel, "head_json": "head/head.json",
              "tensor_key": head_key, "compat_tensor_key": "weight",
              "shape": head_doc["shape"], "dtype": head_doc["dtype"], "bias": None,
              "file_sha256": head_file_sha, "raw_tensor_sha256": head_content,
              "tensor_content_sha256": head_content, "quantized": None, "bits": None,
              "source": "unknown", "applied_in_capture": (form == "logit"),
              "final_norm": None, "equality_receipt": None},
        runtime={"file": "runtime/capture-runtime.json", "file_sha256": "0" * 64,
                 "lane": lane, "lane_inferred": True, "lane_identity_sha256": None,
                 "stack_fingerprint_sha256": runtime_doc["stack_fingerprint_sha256"],
                 "backend_identity_sha256": None, "runtime_reader_sha256": None,
                 "source": "kimi-k3-capture-runtime"},
        determinism=dict(translation["determinism"],
                         evidence_hashes=[capture_doc["capture_content_digest"]],
                         distinct_evidence_hash_count=1),
        coverage=coverage,
        interop=dsmanifest.interop_block(
            adapted_from={"source_format": translation["source_format"],
                          "adapter": "bin/fidelity/dsadapt.py::adapt_k3",
                          "inferred_fields": sorted(set(inferred)),
                          "source_schema": "kimi-k3-distribution-fidelity/1"},
            note="translated from a kimi-k3 artifact; every inferred field forces "
                 "comparability.class = advisory at compare time."),
        disclosures=[{"code": "subset_of_panel", "severity": "caveat",
                      "affects_comparability": True,
                      "detail": coverage["subset_detail"]}]
        if not coverage["complete"] else
        [{"code": "no_known_deviations", "severity": "info",
          "affects_comparability": False, "detail": "full panel"}],
        upstream_receipts=upstream)

    compat_info = {}
    if emit_k3_compat:
        compat_info = k3compat.emit(
            writer, panel_doc=panel_doc, capture_doc=capture_doc,
            manifest_capture=manifest["capture"], head_relpath=head_rel,
            dataset_name=name)
        manifest["interop"].update(compat_info)
    writer.add_file(report_name, (F.canonical_json(dict(
        translation, inferred_fields=sorted(set(inferred)))) + "\n").encode("utf-8"))
    writer.add_readme(
        "---\nlicense: other\n---\n\n# %s\n\nTranslated from a kimi-k3 "
        "distribution-fidelity artifact by `bin/fidelity-dataset adapt --source k3v1 "
        "--emit-dataset`.\n" % name)
    report = dsvalidate.Report(out_dir)
    report.ok("adapter")
    written = writer.finish(manifest, panel_doc, capture_doc, head_doc, runtime_doc,
                            validation_report=report.to_dict())
    return {"written": True, "root": os.path.abspath(out_dir),
            "dataset_sha256": written[F.SEAL_FIELD],
            "capture_content_digest": written["capture"]["capture_content_digest"],
            "records": len(capture_records),
            "head_payload_present": bool(head_rel)}


def _place(src: str, dest: str, link: bool) -> None:
    if link:
        try:
            os.link(src, dest)
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


# ---------------------------------------------------------------------------
# malaiwah serving v2 -- our own published capture
# ---------------------------------------------------------------------------


_TOKENIZER_SNAPSHOT_RE = re.compile(r"@\s*([0-9a-f]{40})")


def _serving_v2_tokenizer(suite: Dict[str, Any]) -> Dict[str, Any]:
    """The panel's tokenizer identity, out of the suite manifest.

    `suite["model"]` is a string of the form
    `"<repo> @ <40-hex> (tokenizer snapshot)"`, and `model_identity` carries the
    digest of the tokenizer files themselves.  Both are properties of the panel:
    every capture on this panel tokenized with exactly this tokenizer, whatever
    weights it was running.
    """
    label = str(suite.get("model") or "")
    identity = suite.get("model_identity") or {}
    repository = label.split("@")[0].strip().split(" ")[0] or None
    match = _TOKENIZER_SNAPSHOT_RE.search(label)
    return {
        "id": "glm-5.3-flash",
        "repository": repository,
        "revision": match.group(1) if match else None,
        "vocab_size": int(suite["vocab_size"]),
        "add_special_tokens": False,
        "chat_template_applied": False,
        # Stronger than a revision: the digest of the tokenizer FILES, which is
        # what `model_identity` pins and what actually determines the ids.
        "tokenizer_sha256": identity.get("tokenizer_sha256"),
    }


def adapt_serving_v2(
    capture_dir: str,
    out_dir: str,
    *,
    suite_dir: str,
    head_dir: Optional[str] = None,
    dataset_id: str,
    name: str,
    role: str = "root",
    lane: str = "other",
    limit: Optional[int] = None,
    link: bool = True,
    emit_k3_compat: bool = False,
    weights: Optional[Dict[str, Any]] = None,
    scope: Optional[Dict[str, Any]] = None,
    quantized: bool = False,
) -> Dict[str, Any]:
    """Turn `glm53flash-fidelity-capture/2` into a conformant v1 dataset.

    This is the superset proof: every field of the published manifest survives,
    and the three defects it carries are FIXED rather than copied --

      O-1 no declared cut point   -> capture.semantic_point, required
      O-3 complete:true in a 512-file shard -> honest coverage + shard_of
      O-4 records of {index, sha256, shape} -> full records with content digests
    """
    source = F.read_json(os.path.join(capture_dir, "capture-manifest-full.json"))
    if source.get("schema") != "glm53flash-fidelity-capture/2":
        raise AdapterError("expected glm53flash-fidelity-capture/2, got %r"
                           % source.get("schema"))
    suite = F.read_json(os.path.join(suite_dir, "suite-manifest.json"))
    inferred = ["capture.semantic_point", "panel.records[].attention_mask_sha256",
                "runtime.lane", "runtime.stack_fingerprint_sha256"]

    writer = dsmanifest.DatasetWriter(out_dir)
    suite_index = {int(row["index"]): row for row in suite["context_index"]}
    declared = int(source.get("expected_contexts") or source.get("contexts"))

    present_indices = []
    for row in source["captures"]:
        index = int(row["index"])
        tensor_path = os.path.join(capture_dir, "hidden_%04d.safetensors" % index)
        if os.path.isfile(tensor_path):
            present_indices.append(index)
    if limit is not None:
        present_indices = present_indices[:limit]

    panel_records = []
    capture_records = []
    for index in present_indices:
        suite_row = suite_index[index]
        token_src = os.path.join(suite_dir, suite_row["file"])
        ids = F.read_json(token_src)
        token_rel = writer.add_token_file(index, ids)
        panel_records.append(dsmanifest.panel_record(
            index=index, token_file=token_rel, token_ids=ids,
            prediction_positions=int(suite["scored_positions_per_context"]),
            window_id="context-%04d" % index,
            role="final", domain=suite_row.get("stratum"),
            document_id=suite_row.get("source_cluster"),
            allocation_stratum=suite_row.get("stratum"),
            source_cluster_id=suite_row.get("source_cluster"),
            partition=suite_row.get("partition"),
            sentinel=bool(suite_row.get("sentinel", False))))

        tensor_path = os.path.join(capture_dir, "hidden_%04d.safetensors" % index)
        dest_rel = "capture/hidden_%04d.safetensors" % index
        dest = os.path.join(out_dir, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if link:
            try:
                os.link(tensor_path, dest)
            except OSError:
                shutil.copy2(tensor_path, dest)
        else:
            shutil.copy2(tensor_path, dest)
        _, header = F.read_safetensors_header(dest)
        # CC-03. REC-2 says a pre-v1 `hidden` key is "accepted on ingest and rewritten".
        # The rewrite was MANIFEST-ONLY: the tensor file was hardlinked verbatim and still
        # carried `hidden`, while record["key"] was overwritten to `hidden_states`. The
        # emitted manifest then named a tensor the bytes do not contain -- dsvalidate's
        # own SEAL-1(d) refuses it ("tensor key 'hidden_states' absent; file carries
        # ['hidden']") and any consumer following record["key"] gets a KeyError. Rewrite
        # the FILE so the manifest tells the truth. A header-key rename leaves
        # tensor_content_sha256 and payload_sha256 byte-identical (verified) -- only the
        # container digest moves, and dsformat already declares that is never
        # determinism evidence.
        present = [k for k in header if k != "__metadata__"]
        key = "hidden_states" if "hidden_states" in header else None
        if key is None:
            legacy = [k for k in present if k in F.LEGACY_HIDDEN_KEYS]
            if not legacy:
                raise AdapterError(
                    "capture tensor %s carries none of the accepted hidden-state keys "
                    "(%s); it holds %r" % (dest_rel, ", ".join(
                        (F.TENSOR_KEY_HIDDEN,) + tuple(F.LEGACY_HIDDEN_KEYS)), present))
            if len(present) != 1:
                raise AdapterError(
                    "capture tensor %s holds %d tensors (%r); a legacy key can only be "
                    "rewritten safely in a single-tensor file, because renaming a key in "
                    "a multi-tensor header can shift data offsets"
                    % (dest_rel, len(present), present))
            key = legacy[0]
            _rewrite_tensor_key(dest, key, F.TENSOR_KEY_HIDDEN)
            _, header = F.read_safetensors_header(dest)
            inferred.append("capture.records[].key (file rewritten from %r to %r, REC-2)"
                            % (key, F.TENSOR_KEY_HIDDEN))
            key = F.TENSOR_KEY_HIDDEN
        shape = header[key]["shape"]
        record = dsmanifest.tensor_record(
            index=index, filename=os.path.basename(dest_rel), abs_path=dest,
            key=key, dtype=header[key]["dtype"], shape=shape, scored_rows=int(shape[0]),
            token_ids_json_sha256=F.token_ids_json_sha256(ids),
            token_ids_sha256_legacy=F.token_ids_json_sha256_legacy(ids),
            attention_mask_sha256=None, window_id="context-%04d" % index,
            role="final", domain=suite_row.get("stratum"),
            document_id=suite_row.get("source_cluster"),
            allocation_stratum=suite_row.get("stratum"),
            source_cluster_id=suite_row.get("source_cluster"))
        # The published manifest's own container digest, cross-checked.
        published = next((r for r in source["captures"] if int(r["index"]) == index), {})
        record["source_manifest_sha256"] = published.get("sha256")
        record["source_manifest_sha256_agrees"] = (published.get("sha256") == record["sha256"])
        capture_records.append(record)

    # -- head ---------------------------------------------------------------
    head_rel = None
    head_content = None
    head_file_sha = None
    head_shape = None
    head_key = "weight"
    if head_dir:
        head_src = os.path.join(head_dir, "head.safetensors")
        extraction_path = os.path.join(head_dir, "head-extraction.json")
        extraction = F.read_json(extraction_path) if os.path.isfile(extraction_path) else {}
        if os.path.isfile(head_src):
            head_rel = "head/weight.safetensors"
            dest = os.path.join(out_dir, head_rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if link:
                try:
                    os.link(head_src, dest)
                except OSError:
                    shutil.copy2(head_src, dest)
            else:
                shutil.copy2(head_src, dest)
            _, header = F.read_safetensors_header(dest)
            head_key = "weight" if "weight" in header else "lm_head.weight"
            head_shape = header[head_key]["shape"]
            head_content = F.tensor_content_sha256(dest, head_key)
            head_file_sha = F.sha256_file(dest)
            published_file_sha = ((extraction.get("tensors") or {}).get("head") or {}).get("sha256")
            if published_file_sha and published_file_sha != head_file_sha:
                raise AdapterError("head file digest disagrees with head-extraction.json")

    head_doc = dsmanifest.head_identity(
        present=bool(head_rel), tensor_key=head_key, shape=head_shape or [0, 0],
        dtype="BF16", file_sha256=head_file_sha, tensor_content_sha256=head_content,
        quantized=quantized, source="native" if not quantized else "artifact_dequantized",
        applied_in_capture=False, file=head_rel, bits=16 if not quantized else None,
        final_norm={"file": None, "tensor_key": "model.language_model.norm.weight",
                    "shape": [4096], "dtype": "BF16",
                    "file_sha256": ((F.read_json(os.path.join(head_dir, "head-extraction.json"))
                                     .get("tensors") or {}).get("final_norm") or {}).get("sha256")
                    if head_dir and os.path.isfile(os.path.join(head_dir, "head-extraction.json"))
                    else None,
                    "tensor_content_sha256": None,
                    "applied_in_capture": True,
                    # HEAD-7: the capture is ALREADY after the final norm.  The
                    # published artifact ships final_norm.safetensors next to
                    # the head, which implies a norm+head replay it does not
                    # want; v1 says so in a field (O-1).
                    "applied_at_replay": False},
        note="the shared BF16 head the fidelity suite publishes; content digest recomputed "
             "here because the published receipts record only the FILE digest (O-6).")

    panel_doc = dsmanifest.panel_binding(
        panel_id="panel--malaiwah.glm53-flash.suite-v1",
        name="GLM-5.3-Flash fidelity suite v1 token panel",
        records=panel_records, context_length=int(suite["context_length"]),
        # The tokenizer belongs to the PANEL, not to the artifact being captured.
        # This used to read `candidate_identity.model_revision`, which is the
        # revision of the weights under test -- so a BF16 capture and an FP8
        # capture of the SAME panel declared different tokenizers, and PANEL-D6
        # refused a comparison that is perfectly legitimate. The suite manifest
        # names the tokenizer snapshot it actually tokenized with; that is the
        # value, identically on both sides.
        tokenizer=_serving_v2_tokenizer(suite),
        # NOT the published suite_token_sha256: that is the aggregate over all
        # 5,120 contexts, and this dataset may be a shard.  The upstream value
        # travels as a separate, non-normative field instead of being asserted
        # as this panel's legacy digest.
        panel_token_sha256_legacy=None,
        contamination={
            "checked": True,
            "method": (suite.get("contamination_scan") or {}).get("normalization"),
            "hits": (suite.get("contamination_scan") or {}).get("calibration_shingles"),
            "receipt": None,
        },
        strata=suite.get("strata"))

    coverage = dsmanifest.coverage_block(
        capture_records, declared,
        shard_of=({"index": 0, "total": max(1, declared // max(1, len(capture_records))),
                   "stride": 1} if len(capture_records) < declared else None),
        subset_detail=("shard of the published capture: indices %d-%d of %d declared"
                       % (present_indices[0], present_indices[-1], declared)
                       if len(capture_records) < declared else None))

    capture_doc = dsmanifest.capture_manifest(
        run_name=source.get("model", "capture"), form="hidden",
        # O-1: the published manifest declares NO cut point.  This is the value
        # its code actually implements -- a post-hook on the final norm -- and
        # it is byte-for-byte kimi-k3's string.
        semantic_point="after_final_rmsnorm_before_lm_head",
        tensor_key=F.TENSOR_KEY_HIDDEN, dtype="BF16", dtype_lossless=True,
        vocab_size=int(suite["vocab_size"]), context_length=int(suite["context_length"]),
        records=capture_records, hidden_width=int(suite["hidden_size"]),
        coverage=coverage)

    identity = source.get("candidate_identity") or {}
    fingerprint = {"schema": "malaiwah.stack-fingerprint.v1",
                   "origin": "glm53flash-fidelity-capture/2",
                   "capture_contract_sha256": source.get("capture_contract_sha256"),
                   "kv_cache_dtype": identity.get("kv_cache_dtype_resolved"),
                   "quantization": source.get("quantization")}
    runtime_doc = dsmanifest.capture_runtime(
        lane=lane, lane_inferred=True, stack_fingerprint=fingerprint,
        stack_fingerprint_sha256=F.sha256_hex(F.canonical_json(fingerprint)),
        lane_identity_sha256=None,
        weights={"repository": "zai-org/GLM-5.3-Flash-BF16" if not quantized else "unknown",
                 "revision": identity.get("model_revision"),
                 "model_revision": identity.get("model_revision"),
                 "config_sha256": identity.get("config_sha256"),
                 "index_sha256": identity.get("index_sha256"),
                 "checkpoint_identity_sha256": None},
        upstream_receipts=[{
            "file": "upstream/capture-manifest-full.json",
            "schema": "glm53flash-fidelity-capture/2",
            "sha256": F.sha256_file(os.path.join(capture_dir, "capture-manifest-full.json")),
            "stripped_fields": ["model"],
        }])

    # PATH-2: the upstream receipt travels verbatim except for host-local paths,
    # and what was stripped is named.
    stripped, names = F.strip_host_paths(source)
    F.write_json(os.path.join(out_dir, "upstream/capture-manifest-full.json"), stripped)

    weights_block = weights or {
        "repository": "zai-org/GLM-5.3-Flash-BF16",
        "revision": identity.get("model_revision"),
        "model_revision": identity.get("model_revision"),
        "quantized": quantized,
        "config_sha256": identity.get("config_sha256"),
        "index_sha256": identity.get("index_sha256"),
        "checkpoint_identity_sha256": None,
        "artifact_ref": None, "model_ref": None, "codec": None,
        "declared_bits": None, "declared_head_bits": None,
    }
    scope_block = scope or dsmanifest.native_scope()

    manifest = dsmanifest.top_manifest(
        dataset={"id": dataset_id, "name": name, "role": role,
                 "structural_status": "sealed", "qualification": None,
                 "author": {"name": "malaiwah", "role": "capture-author",
                            "handle": "malaiwah",
                            "url": "https://huggingface.co/malaiwah",
                            "is_registry_maintainer": True},
                 "license": "mit",
                 "repository": "malaiwah/GLM-5.3-Flash-fidelity-suite-v1",
                 "revision": None, "base_capture": None},
        weights=weights_block, scope=scope_block,
        panel={"panel_id": panel_doc["panel_id"], "panel_file": "panel/panel.json",
               "panel_file_sha256": "0" * 64,
               "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
               "panel_token_sha256_legacy": panel_doc["panel_token_sha256_legacy"],
               "upstream_suite_token_sha256": suite.get("suite_token_sha256"),
               "panel_receipt_sha256": None,
               "repository": "malaiwah/GLM-5.3-Flash-fidelity-suite-v1",
               "revision": None, "contexts": len(panel_records),
               "context_length": int(suite["context_length"]),
               "scored_positions_total": panel_doc["scored_positions_total"],
               "scoring_window": panel_doc["scoring_window"],
               "tokenizer": panel_doc["tokenizer"], "remap_file": None,
               "contamination": panel_doc["contamination"]},
        capture={"manifest_file": "capture/manifest.json", "manifest_file_sha256": "0" * 64,
                 "capture_content_digest": capture_doc["capture_content_digest"],
                 "form": "hidden",
                 "semantic_point": "after_final_rmsnorm_before_lm_head",
                 "tensor_key": F.TENSOR_KEY_HIDDEN, "dtype": "BF16",
                 "dtype_lossless": True, "hidden_width": int(suite["hidden_size"]),
                 "vocab_size": int(suite["vocab_size"]), "head_separable": True,
                 "head_not_separable_reason": None,
                 "records_count": len(capture_records),
                 "scored_rows_total": capture_doc["total_scored_rows"],
                 "total_size_bytes": capture_doc["total_size_bytes"],
                 "lossy_codec": None},
        head={"present": bool(head_rel), "file": head_rel, "head_json": "head/head.json",
              "tensor_key": head_key, "compat_tensor_key": "weight",
              "shape": head_shape or [0, 0], "dtype": "BF16", "bias": None,
              "file_sha256": head_file_sha, "raw_tensor_sha256": head_content,
              "tensor_content_sha256": head_content, "quantized": quantized,
              "bits": 16 if not quantized else None, "source": head_doc["source"],
              "applied_in_capture": False, "final_norm": head_doc["final_norm"],
              "equality_receipt": None},
        runtime={"file": "runtime/capture-runtime.json", "file_sha256": "0" * 64,
                 "lane": lane, "lane_inferred": True, "lane_identity_sha256": None,
                 "stack_fingerprint_sha256": runtime_doc["stack_fingerprint_sha256"],
                 "backend_identity_sha256": None, "runtime_reader_sha256": None,
                 "source": "vllm-serving"},
        determinism={"run_count": 1, "cold_start_per_run": None,
                     "evidence_kind": "hidden_state_tensor_sha256",
                     "evidence_hashes": [capture_doc["capture_content_digest"]],
                     "distinct_evidence_hash_count": 1,
                     "identical_across_runs": None, "repeats": [],
                     "repeat_noise": {
                         "kl_canonical_to_repeat_mean": None,
                         "kl_repeat_to_canonical_mean": None, "js_mean": None,
                         "interpretation": "the serving lane is NOT bitwise reproducible: "
                                           "reports/determinism-bf16.json records 20 of 32 "
                                           "sentinels byte-identical. A single capture "
                                           "therefore cannot claim identical_across_runs."},
                     "note": "one capture; identical_across_runs is null, not true."},
        coverage=coverage,
        interop=dsmanifest.interop_block(
            adapted_from={"source_format": "malaiwah-serving-v2",
                          "adapter": "bin/fidelity/dsadapt.py::adapt_serving_v2",
                          "inferred_fields": sorted(set(inferred)),
                          "source_schema": "glm53flash-fidelity-capture/2"},
            note="derived from our own published capture; O-1, O-3 and O-4 are fixed here "
                 "rather than copied."),
        disclosures=[
            {"code": "subset_of_panel", "severity": "caveat", "affects_comparability": True,
             "detail": coverage["subset_detail"] or "full panel"},
        ] if not coverage["complete"] else [
            {"code": "no_known_deviations", "severity": "info",
             "affects_comparability": False, "detail": "full panel"}],
        upstream_receipts=[{
            "file": "upstream/capture-manifest-full.json",
            "schema": "glm53flash-fidelity-capture/2",
            "sha256": F.sha256_file(os.path.join(out_dir,
                                                 "upstream/capture-manifest-full.json")),
            "stripped_fields": names,
        }])

    if emit_k3_compat:
        manifest["interop"].update(k3compat.emit(
            writer, panel_doc=panel_doc, capture_doc=capture_doc,
            manifest_capture=manifest["capture"], head_relpath=head_rel,
            dataset_name=name))
    writer.add_readme(
        "---\nlicense: mit\n---\n\n# %s\n\nAdapted from `glm53flash-fidelity-capture/2` by "
        "`bin/fidelity-dataset adapt --source malaiwah-serving-v2`.\n" % name)
    report = dsvalidate.Report(out_dir)
    report.ok("adapter")
    return writer.finish(manifest, panel_doc, capture_doc, head_doc, runtime_doc,
                         validation_report=report.to_dict())


# ---------------------------------------------------------------------------
# llama.cpp .kld
# ---------------------------------------------------------------------------

KLD_MAGIC = b"_logits_"


def read_llamacpp_kld_header(path: str) -> Dict[str, Any]:
    """Parse the llama.cpp `--kl-divergence-base` header.

    Layout: magic `_logits_`, `uint32 n_ctx`, `int32 n_vocab`, `int32 n_chunk`,
    then `n_ctx * n_chunk` int32 tokens, then per scored row
    `2 + 2*((n_vocab+1)/2)` uint16 values whose first four bytes are the two
    fp32 halves of `scale` and `min_log_prob`.

    NO provenance: no hashes, no model id, no tokenizer id, no runtime, and a
    vocab mismatch is only a warning.  Everything below is therefore recorded as
    a lossy_codec plus an all-inferred adapter report.
    """
    with open(path, "rb") as handle:
        magic = handle.read(8)
        if magic != KLD_MAGIC:
            raise AdapterError("not a llama.cpp .kld file (magic %r)" % magic)
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", handle.read(12))
        tokens = list(struct.unpack("<%di" % (n_ctx * n_chunk),
                                    handle.read(4 * n_ctx * n_chunk)))
    return {
        "n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk,
        "tokens": tokens,
        # The panel travels BY VALUE inside the file -- genuinely good, and
        # worth stealing for tiny panels (spec 12.6).
        "panel_by_value": True,
        # It scores the SECOND HALF only.  That is panel identity, not a flag.
        "scoring_window": {"score_from": n_ctx // 2, "windowed": True,
                           "min_left_context_tokens": n_ctx // 2,
                           "dropped_positions_total": (n_ctx // 2) * n_chunk,
                           "policy": "llama.cpp scores positions n_ctx/2 .. n_ctx-1 only"},
        "lossy_codec": {
            "kind": "llamacpp-kld-uint16",
            "bits": 16,
            "clamp": "max_logit - 16",
            "description": "16-bit quantized log-probabilities with a hard max_logit-16 floor; "
                           "the stored values are NOT the model's values (D-8).",
        },
    }


def adapt_llamacpp_kld(path: str, out_dir: str) -> Dict[str, Any]:
    header = read_llamacpp_kld_header(path)
    inferred = ["weights.repository", "weights.revision", "panel.tokenizer",
                "runtime.lane", "runtime.stack_fingerprint_sha256", "head",
                "capture.dtype_lossless", "determinism"]
    translation = _report(
        "llamacpp-kld", inferred,
        source_root=os.path.abspath(path),
        capture={"form": "logit", "semantic_point": "lm_head_output_before_sampling",
                 "tensor_key": "logits", "head_separable": False,
                 "head_not_separable_reason": "llama.cpp emits quantized log-probabilities; "
                                              "there is no separable head in the artifact",
                 "vocab_size": header["n_vocab"],
                 "records_count": header["n_chunk"],
                 "lossy_codec": header["lossy_codec"],
                 "dtype_lossless": False},
        panel={"contexts": header["n_chunk"], "context_length": header["n_ctx"],
               "scoring_window": header["scoring_window"],
               "scored_positions_total": header["n_chunk"]
               * (header["n_ctx"] - header["n_ctx"] // 2),
               "panel_by_value": True},
        head={"present": False, "tensor_content_sha256": None,
              "note": "no head is expressible; HEAD-2 applies (both sides logit form) and a "
                      "null digest makes the comparison advisory."},
        runtime={"lane": "other", "lane_inferred": True,
                 "stack_fingerprint": None,
                 "lane_note": "the format carries no runtime information at all"},
        outstanding=[
            "a llama.cpp number is placeable only against another llama.cpp-geometry number: "
            "score_from = n_ctx/2 is a DIFFERENT PANEL from score_from = 0 (PANEL-D3 / D-3)",
            "lossy_codec is non-null, so every comparison is advisory with a "
            "lossy_capture_codec disclosure (D-8)",
        ],
    )
    os.makedirs(out_dir, exist_ok=True)
    F.write_json(os.path.join(out_dir, "llamacpp-kld-translation.json"), translation)
    return translation
