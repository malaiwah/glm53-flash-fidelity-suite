#!/usr/bin/env python3
"""Regenerate data/*.jsonl for the quant-fidelity-registry from verified sources.

This file IS the provenance of the seeded rows: every number below was read back
from the receipt named in its `sources` block during the seeding pass, and nothing
here was transcribed from a summary. Re-running it must reproduce data/*.jsonl
byte for byte (`make reseed-check`).

Offline: no network. Receipt URIs are recorded, never fetched.

Usage:  python3 tools/seed_registry.py [--out DIR] [--check]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_lib as L  # noqa: E402

V = L.SCHEMA_VERSION

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def disc(code, severity, detail, affects=False):
    return {"code": code, "severity": severity, "detail": detail, "affects_comparability": affects}


NONE_DISC = [disc("no_known_deviations", "info", "No deviation from this registry's default protocol is known for this record.")]


def src(kind, uri, sha256=None, note=None):
    d = {"kind": kind, "uri": uri}
    if sha256:
        d["sha256"] = sha256
    if note:
        d["note"] = note
    return d


def attr(name, role, handle=None, url=None, maintainer=False):
    d = {"name": name, "role": role, "handle": handle, "url": url, "is_registry_maintainer": maintainer}
    return d


# Receipts this repository holds, at receipts/<handle>/<slug>.json. The digest is of the
# committed file, so a row citing one of these is citing bytes any reader can fetch and hash.
STREAM_K6_RECEIPT = "receipts/malaiwah/stream-k6-kld.json"
STREAM_K6_RECEIPT_SHA = "7ee0de697d050ff1aca9b85981a158f57304a46c020408b39742f5f85a0ff969"
STREAM_K6_VERDICT = "receipts/malaiwah/stream-k6-verdict.json"
STREAM_K6_VERDICT_SHA = "e205c14f5700417b32f4cb4a2d6724f3bf416ffc9d4cca3f129c18b0a0e7b005"
STREAM_K8_RECEIPT = "receipts/malaiwah/stream-k8-kld.json"
STREAM_K8_RECEIPT_SHA = "8eab14b0ef3ba042e49735973d91dcc47e470b9331f9e65151635b2862bb05d1"
STREAM_BF16_RECEIPT = "receipts/malaiwah/stream-bf16-kld.json"
STREAM_BF16_RECEIPT_SHA = "8abee678d26fc4be92f3b6327419da25c505de2021043dc2719c1e355290090b"
HF_REGISTRY_RAW = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/"

MAL = lambda role: attr("malaiwah", role, handle="malaiwah", url="https://huggingface.co/malaiwah", maintainer=True)
BRANDON = lambda role: attr("brandonmusic", role, handle="brandonmusic", url="https://huggingface.co/brandonmusic")
SERO = lambda role: attr("0xSero", role, handle="0xSero", url="https://huggingface.co/0xSero")
ORCA = lambda role: attr("orcarouter", role, handle="orcarouter", url="https://huggingface.co/orcarouter")
TURBO = lambda role: attr("turboderp", role, handle="turboderp", url="https://huggingface.co/turboderp")
ZAI = lambda role: attr("Z.ai", role, handle="zai-org", url="https://huggingface.co/zai-org")
QWEN = lambda role: attr("Qwen (Alibaba)", role, handle="Qwen", url="https://huggingface.co/Qwen")
UNSLOTH = lambda role: attr("unsloth", role, handle="unsloth", url="https://huggingface.co/unsloth")
GITTENSOR = lambda role: attr("gittensor-model-hub", role, handle="gittensor-model-hub", url="https://huggingface.co/gittensor-model-hub")


def hf(repo, revision, revision_source="hf_api", status="known", link_type="repository", dataset=False,
       reason=None, path=None):
    if repo is None:
        return {"repository": None, "url": None, "revision": revision, "path": path,
                "revision_source": revision_source, "status": status, "link_type": "none", "reason": reason}
    url = "https://huggingface.co/%s%s" % ("datasets/" if dataset else "", repo)
    return {"repository": repo, "url": url, "revision": revision, "path": path,
            "revision_source": revision_source, "status": status, "link_type": link_type, "reason": reason}


def lair(model_id=None, instance_id=None, url=None, confidence="unverified"):
    """cross_refs into 0xSero/local-ai-registry. Never asserts an unverified match as exact."""
    return {"local_ai_registry": {"model_id": model_id, "model_instance_id": instance_id,
                                  "url": url, "match_confidence": confidence}}


def asg(cls, treatment, fmt, bpw=None, layer_range="all", note=None):
    d = {"tensor_class": cls, "treatment": treatment, "format": fmt,
         "bits_per_weight": bpw, "layer_range": layer_range}
    if note:
        d["note"] = note
    return d


def scope(policy, assignments, head_policy, kv="bf16", act=None, mtp=None):
    return {"policy": policy, "assignments": assignments, "head_policy": head_policy,
            "kv_cache_dtype": kv, "activation_quantization": act, "mtp_included": mtp}


def native_scope(fmt="bf16", kv="bf16", mtp=None):
    return scope("none", [
        asg("embed_tokens", "native", fmt), asg("attn.qkv", "native", fmt), asg("attn.o", "native", fmt),
        asg("mlp.gate", "native", fmt), asg("mlp.up", "native", fmt), asg("mlp.down", "native", fmt),
        asg("moe.experts", "native", fmt), asg("norm", "native", fmt), asg("lm_head", "native", fmt),
    ], "native", kv=kv, mtp=mtp)


def unknown_scope(fmt, bpw, kv="unknown", head="unknown", mtp=None, note=None):
    """For a third-party artifact whose per-tensor-class recipe was never published."""
    return scope("mixed", [
        asg("embed_tokens", "unknown", "unknown", note=note),
        asg("attn.qkv", "unknown", "unknown"),
        asg("attn.o", "unknown", "unknown"),
        asg("mlp.gate", "quantized", fmt, bpw),
        asg("mlp.up", "quantized", fmt, bpw),
        asg("mlp.down", "quantized", fmt, bpw),
        asg("moe.experts", "quantized", fmt, bpw),
        asg("lm_head", "unknown", "unknown"),
    ], head, kv=kv, mtp=mtp)


INCOMPLETE = disc(
    "artifact_identity_incomplete", "caveat",
    "The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments "
    "records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.", True)


def artifact(aid, model_ref, name, kind, huggingface, container, precision_label, size_bytes,
             codec, sc, producer, sources, disclosures, **kw):
    rec = {
        "schema_version": V, "id": aid, "model_ref": model_ref, "name": name, "kind": kind,
        "huggingface": huggingface,
        "weights": {"container": container, "precision_label": precision_label,
                    "size_bytes": size_bytes,
                    "size_gb": (None if size_bytes is None else size_bytes / 1e9)},
        "codec": codec, "scope": sc, "scope_digest": L.scope_digest(sc),
        "producer": producer, "sources": sources, "disclosures": disclosures,
    }
    rec["weights"].update(kw.pop("weights_extra", {}))
    rec.update(kw)
    return rec


def codec(family, nominal, effective=None, tool=None, version=None, calibration=None, group_size=None):
    c = {"family": family, "bits_per_weight_nominal": nominal, "bits_per_weight_effective": effective,
         "group_size": group_size}
    if tool:
        c["quantizer"] = {"tool": tool, "version": version, "revision": None, "pipeline_ref": None}
    c["calibration"] = calibration or {"used": None, "corpus": None, "tokens": None,
                                       "overlaps_any_panel": None, "overlapping_panel_refs": []}
    return c


# ===========================================================================
# 1. MODELS
# ===========================================================================
GLM = "model--zai-org.glm-5.3-flash"
QWN = "model--qwen.qwen3.8-27b"

A_BF16_A6 = "artifact--zai-org.glm-5.3-flash-bf16.a6c167b6"
A_BF16_B1 = "artifact--zai-org.glm-5.3-flash-bf16.b1967181"
Q_BF16 = "artifact--qwen.qwen3.8-27b-bf16"

MODELS = [
    {
        "schema_version": V, "id": GLM, "name": "GLM-5.3-Flash", "family": "glm-5.3",
        "publisher": ZAI("model-publisher"),
        "huggingface": hf("zai-org/GLM-5.3-Flash", "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a", "revision_txt"),
        "architecture": {"kind": "moe-decoder", "total_parameters": None, "active_parameters": None,
                         "num_layers": None, "hidden_size": 4096, "vocab_size": 154880, "has_mtp": True,
                         "note": "hidden_size and vocab_size read from the fidelity reports' own header "
                                 "(hidden_size 4096, vocab_size 154880); parameter counts are not asserted "
                                 "because no receipt in this registry establishes them."},
        "tokenizer": {"id": "glm-5.3-flash", "repository": "zai-org/GLM-5.3-Flash-BF16",
                      "revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
                      "files_sha256": {
                          "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
                          "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
                          "chat_template.jinja": "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3"},
                      "vocab_size": 154880},
        "canonical_weights": {"artifact_ref": A_BF16_B1, "precision": "bf16"},
        "license": None,
        "cross_refs": lair(url="https://huggingface.co/datasets/0xSero/local-ai-registry"),
        "sources": [
            src("hf_file", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/resolve/95f4fdd94bf29989db2e0d1054e4931f55edb6aa/calibration/panel-v1/tokenizer.receipt.json",
                "fd6e407903e7c787f84df361c44d0af945193ade27e953a02dd613ecf9a4c3b2",
                "tokenizer file digests; fetched read-only at the pinned dataset revision"),
            src("model_card", "https://huggingface.co/zai-org/GLM-5.3-Flash"),
        ],
        "disclosures": [disc(
            "estimator_unknown", "info",
            "The tokenizer receipt declares vocab_size 154856 while every scorer in this registry scores "
            "over 154880 logit entries (the padded head width). 154880 is recorded here because that is "
            "the width the measurements actually cover.")],
    },
    {
        "schema_version": V, "id": QWN, "name": "Qwen3.8-27B", "family": "qwen3.8",
        "publisher": QWEN("model-publisher"),
        "huggingface": hf("Qwen/Qwen3.8-27B", None, "none"),
        "architecture": {"kind": "dense-decoder", "total_parameters": None, "active_parameters": None,
                         "num_layers": None, "hidden_size": 5120, "vocab_size": 248320, "has_mtp": True,
                         "note": "hidden_size 5120 / vocab_size 248320 read from the kld5 ladder receipts."},
        "tokenizer": {"id": "qwen3.8", "repository": "Qwen/Qwen3.8-27B", "revision": None,
                      "files_sha256": {"tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"},
                      "vocab_size": 248320},
        "canonical_weights": {"artifact_ref": Q_BF16, "precision": "bf16"},
        "license": None,
        "cross_refs": lair(),
        "sources": [src("receipt_file", "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/kld5-suite-manifest.json",
                        "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019",
                        "qwen38-distribution-fidelity/6 suite manifest; carries model_identity.tokenizer_sha256")],
        "disclosures": [disc(
            "revision_unpinned", "caveat",
            "No receipt in this registry pins a Hub revision for the Qwen3.8-27B BF16 base: every kld5 "
            "receipt records model_revision=null with model_revision_source='none'. Identity rests on "
            "index_sha256 77042094... and config_sha256 191e0af2... instead.", True)],
    },
]

# ===========================================================================
# 2. PANELS
# ===========================================================================
P_B25 = "panel--glm53.brandonmusic.final25"
P_B1W = "panel--glm53.brandonmusic.final-0000"
P_G10M = "panel--glm53.malaiwah.suite-v5-10m"
P_G10M_W1024 = "panel--glm53.malaiwah.suite-v5-10m.scorefrom1024"
P_ORCA = "panel--orcarouter.undisclosed"
P_Q10M = "panel--qwen38.malaiwah.suite-v5-10m"
P_Q1M = "panel--qwen38.malaiwah.suite-v5-shard0-1m"
P_Q2M = "panel--qwen38.malaiwah.suite-v5-shards01-2m"
P_Q1M_W256 = "panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256"
P_Q1M_W1024 = "panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024"

# The 25 final-window token-id digests, read out of brandonmusic's own panel.json
# (sha256 6bafe328..., fetched read-only at dataset revision 95f4fdd9).
FINAL25 = {
    "final-0000": "338027e62f41540f73e38c6f9b4b9a06a50196cbd38cd9c69f11886af9d3cf9f",
    "final-0001": "75e32c0a3c6d478004e63902a3a9a2075ca0b1e583e60bdb9df0d3a4ef65a85e",
    "final-0002": "68cc93c3e99875430ebfec1f60ed399ca0e7484a54bc522eaa4884b022f65b4e",
}
_PANEL_JSON_SRC = src(
    "hf_file",
    "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/resolve/95f4fdd94bf29989db2e0d1054e4931f55edb6aa/calibration/panel-v1/panel.json",
    "6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff",
    "quant-pipeline.glm53-token-panel.v1: 665 windows, each with role, domain, document_id, "
    "prediction_positions and a token_ids_sha256. Downloaded and hashed independently during seeding; "
    "the digest matches the value panel.receipt.json declares as token_panel_artifact_sha256.")

BRANDON_GUARD = disc(
    "weak_contamination_guard", "caveat",
    "This panel's only contamination guard is ROLE SEPARATION: the 25 'final' windows are drawn from the "
    "same packed corpus as the 384 fit / 128 conditional-fit / 64 selection / 64 confirmation windows and "
    "are declared qualification-only. No lexical or n-gram scan is published, and the underlying document "
    "provenance is published only as a digest. This is materially weaker than the malaiwah v5 suites, which "
    "run a 12-word shingle whole-document pre-exclusion and report 0 hits. Do not describe the two guards "
    "as equivalent. It applies equally to every row on this panel, so it does not disturb comparisons "
    "WITHIN the panel.")

MAL_SUITE_CONTAM = ("12-word lexical shingles, stride 1, blake2b-128 digests over Unicode NFKC casefolded "
                    "word tokens, scanned against 859,426 calibration shingles from 6 calibration sources; "
                    "whole-document pre-exclusion on any match, plus a decoded-token rejection pass on every "
                    "emitted context")

PANELS = [
    {
        "schema_version": V, "id": P_B25,
        "name": "brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows",
        "author": BRANDON("panel-author"), "model_scope": [GLM],
        "tokenizer": {"id": "glm-5.3-flash", "repository": "zai-org/GLM-5.3-Flash-BF16",
                      "revision": "a6c167b62691b2bac901344b65cb651a70f53e43", "vocab_size": 154880},
        "structure": {
            "contexts": 25, "context_length": 2048, "positions_per_context": 2047,
            "positions_per_context_min": 2047, "positions_per_context_max": 2047,
            "scored_positions_total": 51175,
            "scoring_window": {"score_from": 0, "windowed": False, "min_left_context_tokens": 1,
                               "dropped_positions_total": 0,
                               "policy": "every prediction position of every window is scored; nothing is dropped"},
            "strata": {"axis1_general": {"contexts": 7}, "axis2_legal": {"contexts": 6},
                       "axis3_code_agentic": {"contexts": 6}, "axis4_reasoning_termination": {"contexts": 6}},
        },
        "corpus": {
            "lineage": "reap-recall-packed.jsonl, an author-built packed calibration corpus (32,420,240 B, "
                       "sha256 f767863e...) over 4 domains: axis1_general, axis2_legal, axis3_code_agentic, "
                       "axis4_reasoning_termination, minimum 5 documents per domain",
            "version": "panel-v1", "build_tool_ref": None, "public": True,
            "sources": [src("dataset_card", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits")],
            "license_note": "Per-document URLs and licences for the packed corpus are not published; the "
                            "corpus is pinned by digest only.",
        },
        "identity": {
            "panel_token_sha256": "6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff",
            "hash_covers": "token_manifest",
            "manifest_sha256": None,
            "panel_receipt_sha256": "0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5",
            "shard_token_sha256": FINAL25,
        },
        "contamination": {"checked": False,
                          "method": "role separation only; no lexical or n-gram scan published",
                          "benchmarks_scanned": [], "hits": None, "receipt": None},
        "sealed": True,
        "derived_from": None, "derivation": None,
        "availability": {"status": "public",
                         "uri": "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"},
        "cross_refs": lair(),
        "sources": [_PANEL_JSON_SRC,
                    src("hf_file", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/resolve/95f4fdd94bf29989db2e0d1054e4931f55edb6aa/calibration/panel-v1/panel.receipt.json",
                        None, "quant-pipeline.glm53-token-panel-receipt.v1; its self-declared receipt_sha256 "
                              "is 0beec577... and it names token_panel_artifact_sha256 6bafe328...")],
        "disclosures": [BRANDON_GUARD],
    },
    {
        "schema_version": V, "id": P_B1W,
        "name": "brandonmusic panel v1, single window final-0000",
        "author": BRANDON("panel-author"), "model_scope": [GLM],
        "tokenizer": {"id": "glm-5.3-flash", "repository": "zai-org/GLM-5.3-Flash-BF16",
                      "revision": "a6c167b62691b2bac901344b65cb651a70f53e43", "vocab_size": 154880},
        "structure": {
            "contexts": 1, "context_length": 2048, "positions_per_context": 2047,
            "positions_per_context_min": 2047, "positions_per_context_max": 2047,
            "scored_positions_total": 2047,
            "scoring_window": {"score_from": 0, "windowed": False, "min_left_context_tokens": 1,
                               "dropped_positions_total": 0,
                               "policy": "every prediction position of the single window is scored"},
            "strata": {"axis1_general": {"contexts": 1}},
        },
        "corpus": {"lineage": "the axis1_general window reap-recall-packed-axis1_general-4 of panel-v1",
                   "version": "panel-v1", "build_tool_ref": None, "public": True,
                   "sources": [src("dataset_card", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits")],
                   "license_note": None},
        "identity": {"panel_token_sha256": FINAL25["final-0000"], "hash_covers": "token_ids",
                     "manifest_sha256": None, "panel_receipt_sha256": None,
                     "shard_token_sha256": {"final-0000": FINAL25["final-0000"]}},
        "contamination": {"checked": False, "method": "role separation only; inherited from panel-v1",
                          "benchmarks_scanned": [], "hits": None, "receipt": None},
        "sealed": True,
        "derived_from": P_B25,
        "derivation": {"kind": "shard_subset",
                       "detail": "window final-0000 alone, 1/25 of the parent panel. 2,047 scored positions "
                                 "instead of 51,175. brandonmusic's runtime receipts score this window only. "
                                 "The same artifact reads 0.022751 here and 0.024555 over the full 25 windows, "
                                 "a 7% swing -- which is why this is a separate panel record."},
        "availability": {"status": "public",
                         "uri": "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"},
        "cross_refs": lair(),
        "sources": [_PANEL_JSON_SRC,
                    src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/nvfp4-dynamic-scale-control-kld-report.json",
                        "e5365075bccd4e27c9e7f002c23e31cc6f8df196c3c7ccf847faae4f007b22f9",
                        "independently corroborates the window's token digest: tokens_sha256 338027e6... "
                        "and window_id final-0000")],
        "disclosures": [BRANDON_GUARD,
                        disc("subset_of_panel", "caveat",
                             "A single 2,047-position window. Numbers on this panel have far wider sampling "
                             "error than the 25-window panel and must never be tabled beside it.", True)],
    },
    {
        "schema_version": V, "id": P_G10M,
        "name": "malaiwah GLM-5.3-Flash distribution-fidelity suite v5 -- 5,120 contexts",
        "author": MAL("panel-author"), "model_scope": [GLM],
        "tokenizer": {"id": "glm-5.3-flash", "repository": "zai-org/GLM-5.3-Flash-BF16",
                      "revision": "b1967181a3917ae70a437f4884748f6b8e3a1f4d", "vocab_size": 154880},
        "structure": {
            "contexts": 5120, "context_length": 2048, "positions_per_context": 2047,
            "positions_per_context_min": 2047, "positions_per_context_max": 2047,
            "scored_positions_total": 10480640,
            "scoring_window": {"score_from": 0, "windowed": False, "min_left_context_tokens": 1,
                               "dropped_positions_total": 0,
                               "policy": "no window: every scored position of every context is included"},
            "strata": {s: {"contexts": 1024} for s in
                       ("code", "encyclopedic", "literary", "multilingual", "scientific")},
        },
        "corpus": {"lineage": "suite v5: 5 strata (code, encyclopedic, literary, multilingual, scientific) "
                              "at 1,024 contexts each, drawn by deterministic sorted-document round-robin from "
                              "941 discovered documents in 837 source clusters; 44 documents excluded for "
                              "calibration overlap before selection, 897 eligible",
                   "version": "v5", "build_tool_ref": None, "public": True,
                   "sources": [src("dataset_card", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1")],
                   "license_note": None},
        "identity": {"panel_token_sha256": "2e0ea09683564554dad9f6e610cb265c5cb86c7350953a83a5ac368c7a475bee",
                     "hash_covers": "token_ids",
                     "manifest_sha256": "0d49ef4b3960e324bebde1b24d448004eb4181d368582852bb9614b1a5a70af6",
                     "panel_receipt_sha256": None, "shard_token_sha256": {}},
        "contamination": {"checked": True, "method": MAL_SUITE_CONTAM,
                          "benchmarks_scanned": [], "hits": 0,
                          "receipt": src("receipt_file", "/Users/mbelleau/Projects/glm53-fidelity-suite/suite/suite-manifest.json",
                                         "0d49ef4b3960e324bebde1b24d448004eb4181d368582852bb9614b1a5a70af6",
                                         "glm53flash-distribution-fidelity/6; document_scan reports 941 scanned, "
                                         "44 excluded, and contamination_scan reports total_hits 0")},
        "sealed": True, "derived_from": None, "derivation": None,
        "availability": {"status": "public",
                         "uri": "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1"},
        "cross_refs": lair(),
        "sources": [src("receipt_file", "/Users/mbelleau/Projects/glm53-fidelity-suite/suite/suite-manifest.json",
                        "0d49ef4b3960e324bebde1b24d448004eb4181d368582852bb9614b1a5a70af6")],
        "disclosures": NONE_DISC,
    },
    {
        "schema_version": V, "id": P_G10M_W1024,
        "name": "malaiwah GLM-5.3-Flash suite v5, scored from position 1024",
        "author": MAL("panel-author"), "model_scope": [GLM],
        "tokenizer": {"id": "glm-5.3-flash", "repository": "zai-org/GLM-5.3-Flash-BF16",
                      "revision": "b1967181a3917ae70a437f4884748f6b8e3a1f4d", "vocab_size": 154880},
        "structure": {
            "contexts": 5120, "context_length": 2048, "positions_per_context": 1023,
            "positions_per_context_min": 1023, "positions_per_context_max": 1023,
            "scored_positions_total": 5237760,
            # 1025, verbatim from the receipt's scored_position_window: the first
            # RETAINED position is index 1024, so it has 1025 tokens of left context.
            "scoring_window": {"score_from": 1024, "windowed": True, "min_left_context_tokens": 1025,
                               "dropped_positions_total": 5242880,
                               "policy": "the first 1024 scored positions of every context were dropped before "
                                         "any statistic was computed"},
            "strata": {s: {"contexts": 1024} for s in
                       ("code", "encyclopedic", "literary", "multilingual", "scientific")},
        },
        "corpus": {"lineage": "identical token content to the parent panel; only the scored-position policy differs",
                   "version": "v5", "build_tool_ref": None, "public": True, "sources": [], "license_note": None},
        "identity": {"panel_token_sha256": "2e0ea09683564554dad9f6e610cb265c5cb86c7350953a83a5ac368c7a475bee",
                     "hash_covers": "token_ids",
                     "manifest_sha256": "0d49ef4b3960e324bebde1b24d448004eb4181d368582852bb9614b1a5a70af6",
                     "panel_receipt_sha256": None, "shard_token_sha256": {}},
        "contamination": {"checked": True, "method": MAL_SUITE_CONTAM, "benchmarks_scanned": [], "hits": 0,
                          "receipt": None},
        "sealed": True, "derived_from": P_G10M,
        "derivation": {"kind": "scoring_window_change",
                       "detail": "score_from 0 -> 1024. Identical tokens, half the scored positions, and a "
                                 "materially different number: 0.028104 becomes 0.018794 on the same artifact "
                                 "and the same teacher. This is the clearest demonstration in the registry that "
                                 "the scored-position policy is part of panel identity."},
        "availability": {"status": "public",
                         "uri": "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1"},
        "cross_refs": lair(),
        "sources": [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16-scorefrom1024.json",
                        None, "glm53flash-fidelity-report/3; declares scored_position_window score_from=1024, "
                              "windowed=true, scored_positions 5,237,760")],
        "disclosures": NONE_DISC,
    },
    {
        "schema_version": V, "id": P_ORCA,
        "name": "orcarouter MLX evaluation set (undisclosed)",
        "author": ORCA("panel-author"), "model_scope": [GLM],
        "tokenizer": {"id": "glm-5.3-flash", "repository": None, "revision": None, "vocab_size": None},
        "structure": {"contexts": None, "context_length": None, "positions_per_context": None,
                      "scored_positions_total": None,
                      "scoring_window": {"score_from": None, "windowed": False,
                                         "min_left_context_tokens": None, "dropped_positions_total": None,
                                         "policy": "not disclosed"}},
        "corpus": {"lineage": "not disclosed on the model card", "version": None, "build_tool_ref": None,
                   "public": False,
                   "sources": [src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX")],
                   "license_note": None},
        "identity": {"panel_token_sha256": None, "hash_covers": "none", "manifest_sha256": None,
                     "panel_receipt_sha256": None, "shard_token_sha256": {}},
        "contamination": {"checked": False, "method": None, "benchmarks_scanned": [], "hits": None, "receipt": None},
        "sealed": False, "derived_from": None, "derivation": None,
        "availability": {"status": "undisclosed", "uri": None},
        "cross_refs": lair(),
        "sources": [src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX",
                        None, "the card reports KLD, p95 KLD, top-1, perplexity and weight-space metrics but "
                              "states no window count, context length or scored-position total")],
        "disclosures": [disc("undisclosed_panel", "caveat",
                             "Neither the token set, the window count nor the scored-position total is "
                             "published. Numbers on this panel can be reported but cannot be compared with "
                             "anything measured on a known panel -- including other rows for the same model.",
                             True)],
    },
]


def _mal_qwen_panel(pid, name, contexts, positions, token_sha, derived=None, derivation=None,
                    score_from=0, windowed=False, ppc=2047, clusters=None, shard_hashes=None,
                    strata=True, sources=None, extra_disc=None):
    return {
        "schema_version": V, "id": pid, "name": name, "author": MAL("panel-author"), "model_scope": [QWN],
        "tokenizer": {"id": "qwen3.8", "repository": "Qwen/Qwen3.8-27B", "revision": None, "vocab_size": 248320},
        "structure": {
            "contexts": contexts, "context_length": 2048, "positions_per_context": ppc,
            "positions_per_context_min": ppc, "positions_per_context_max": ppc,
            "scored_positions_total": positions,
            "scoring_window": {"score_from": score_from, "windowed": windowed,
                               # score_from + 1, verbatim from the receipts' scored_position_window
                               # (from256 -> 257, from1024 -> 1025): the first retained position is
                               # index score_from, so it carries score_from+1 tokens of left context.
                               "min_left_context_tokens": (score_from + 1 if score_from else 1),
                               "dropped_positions_total": (contexts * (2047 - ppc)) if windowed else 0,
                               "policy": ("the first %d scored positions of every context were dropped before "
                                          "any statistic was computed" % score_from) if windowed else
                                         "every shard scored every position of every context; nothing is windowed"},
            "strata": ({s: {"contexts": contexts // 5} for s in
                        ("code", "encyclopedic", "literary", "multilingual", "scientific")} if strata else {}),
        },
        "corpus": {"lineage": "qwen38 suite v5: 5 strata at %d contexts each, 842 source clusters in the "
                              "parent suite; 12-word shingle contamination pre-exclusion" % (contexts // 5)
                              if strata else "shard subset of the qwen38 suite v5 parent",
                   "version": "v5", "build_tool_ref": None, "public": False, "sources": [],
                   "license_note": None},
        "identity": {"panel_token_sha256": token_sha,
                     "hash_covers": "token_ids" if token_sha else "none",
                     "manifest_sha256": "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019",
                     "panel_receipt_sha256": None, "shard_token_sha256": shard_hashes or {}},
        "contamination": {"checked": True, "method": MAL_SUITE_CONTAM, "benchmarks_scanned": [], "hits": 0,
                          "receipt": src("receipt_file",
                                         "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/kld5-suite-manifest.json",
                                         "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019")},
        "sealed": bool(token_sha), "derived_from": derived, "derivation": derivation,
        "availability": {"status": "private",
                         "uri": None},
        "cross_refs": lair(),
        "sources": sources or [src("receipt_file",
                                   "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/kld5-suite-manifest.json",
                                   "c79dfad3767ca5b3015129077f20dbb9282a2e51ca8bca9ed09be8c7a9c73019")],
        "disclosures": (extra_disc or []) + [disc(
            "unsealed_source", "caveat",
            "The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest "
            "c79dfad3..., but the token files themselves are not published, so a third party cannot "
            "reproduce the digest today.", True)],
    }


SH0 = "caef8a4628d6c07c162100895096f890cdf9cafc8e4c48b3d66035d737ee7cf7"
SH1 = "3961604e08636b41f0e263238e888c2940ca49f2ff5ac4a834e46f4c29f902b3"

PANELS += [
    _mal_qwen_panel(P_Q10M, "malaiwah Qwen3.8-27B distribution-fidelity suite v5 -- 5,120 contexts",
                    5120, 10480640, "510541f6861b589d44932db253ec25d96d6daaeeee4ea2ab9b65329209482b88",
                    shard_hashes={"shard-0000": SH0, "shard-0001": SH1}),
    _mal_qwen_panel(P_Q1M, "malaiwah Qwen3.8-27B suite v5, shard 0 -- 512 contexts",
                    512, 1048064, SH0, derived=P_Q10M,
                    derivation={"kind": "shard_subset",
                                "detail": "shard 0 of 10 (512 of 5,120 contexts, 330 of 842 source clusters). "
                                          "Different tokens, therefore a different digest and a different "
                                          "comparability key. K6-parity 0.001634 lives here; the FP8 baseline "
                                          "on this panel is 0.005197, NOT the 10M panel's 0.005294."},
                    strata=False),
    _mal_qwen_panel(P_Q2M, "malaiwah Qwen3.8-27B suite v5, shards 0-1 -- 1,024 contexts",
                    1024, 2096128, None, derived=P_Q10M,
                    derivation={"kind": "shard_subset",
                                "detail": "shards 0 and 1 of 10 (1,024 contexts, 495 source clusters)."},
                    strata=False, shard_hashes={"shard-0000": SH0, "shard-0001": SH1},
                    extra_disc=[disc("unsealed_source", "caveat",
                                     "No combined token digest was published for the two-shard union; the "
                                     "two per-shard digests are recorded instead, which pin the content but "
                                     "are not a single panel identity.", True)]),
    _mal_qwen_panel(P_Q1M_W256, "malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 256",
                    512, 916992, SH0, derived=P_Q1M,
                    derivation={"kind": "scoring_window_change", "detail": "score_from 0 -> 256 on shard 0."},
                    score_from=256, windowed=True, ppc=1791, strata=False),
    _mal_qwen_panel(P_Q1M_W1024, "malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 1024",
                    512, 523776, SH0, derived=P_Q1M,
                    derivation={"kind": "scoring_window_change", "detail": "score_from 0 -> 1024 on shard 0."},
                    score_from=1024, windowed=True, ppc=1023, strata=False),
]

# ===========================================================================
# 3. ARTIFACTS
# ===========================================================================
A_FP8 = "artifact--zai-org.glm-5.3-flash-fp8"
A_FP8_MLAKV = "artifact--brandonmusic.glm-5.3-flash-fp8-mla-kv"
A_NVFP4_BM = "artifact--brandonmusic.glm-5.3-flash-nvfp4-runtime"
A_K6 = "artifact--malaiwah.glm-5.3-flash-tr3-6bpw"
A_K8 = "artifact--malaiwah.glm-5.3-flash-tr3-8bpw"
A_DIONE = "artifact--0xsero.glm-5.3-flash-exl3-q4"
A_B4 = "artifact--brandonmusic.glm-5.3-flash-tr3-4bpw"
A_FP8_DEQ = "artifact--orcarouter.glm-5.3-flash-fp8-dequantized"
ORCA_IDS = {b: "artifact--orcarouter.glm-5.3-flash-mlx-%s" % b.replace("-", "").replace("_", "")
            for b in ("6-bit", "4-bit", "3-bit", "2-bit", "2bit-lite")}

Q_FP8 = "artifact--qwen.qwen3.8-27b-fp8"
Q_K5K6 = "artifact--malaiwah.qwen3.8-27b-exl3-k5k6"
Q_HYD = "artifact--malaiwah.qwen3.8-27b-exl3-k5k6-hydrated"
Q_CTX = "artifact--malaiwah.qwen3.8-27b-exl3-k5k6-context"
Q_K4 = "artifact--malaiwah.qwen3.8-27b-k4"
Q_K6P = "artifact--malaiwah.qwen3.8-27b-exl3-k6-parity"
Q_NVFP4 = "artifact--unsloth.qwen3.8-27b-nvfp4"
Q_GT5090 = "artifact--gittensor-model-hub.qwen3.8-27b-nvfp4-rtx5090"
Q_T5 = "artifact--turboderp.qwen3.8-27b-exl3.5bpw"
Q_T6 = "artifact--turboderp.qwen3.8-27b-exl3.6bpw"
Q_GGUF_Q8 = "artifact--unsloth.qwen3.8-27b-gguf.q8-0"
Q_GGUF_Q6 = "artifact--unsloth.qwen3.8-27b-gguf.q6-k"
Q_GGUF_Q5 = "artifact--unsloth.qwen3.8-27b-gguf.ud-q5-k-xl"
Q_GGUF_BF16 = "artifact--unsloth.qwen3.8-27b-gguf.bf16"
Q_AWQ = "artifact--unattributed.qwen3.8-27b-awq-int4"
Q_MTP = "artifact--unattributed.qwen3.8-27b-mtp-nvfp4"

REV_UNPINNED = lambda what: disc(
    "revision_unpinned", "caveat",
    "No measurement receipt for this artifact records a Hub revision. %s" % what, True)

EXL3_SCOPE_UNIFORM = lambda bpw: scope("uniform", [
    asg("embed_tokens", "native", "bf16"), asg("attn.qkv", "quantized", "exl3-mcg", bpw),
    asg("attn.o", "quantized", "exl3-mcg", bpw), asg("mlp.gate", "quantized", "exl3-mcg", bpw),
    asg("mlp.up", "quantized", "exl3-mcg", bpw), asg("mlp.down", "quantized", "exl3-mcg", bpw),
    asg("moe.experts", "quantized", "exl3-mcg", bpw), asg("norm", "native", "bf16"),
    asg("lm_head", "native", "bf16"),
], "native", kv="bf16")

ARTIFACTS = [
    artifact(A_BF16_A6, GLM, "GLM-5.3-Flash BF16 @a6c167b6", "base",
             hf("zai-org/GLM-5.3-Flash-BF16", "a6c167b62691b2bac901344b65cb651a70f53e43", "revision_txt"),
             "safetensors", "BF16", None, codec("bf16", None), native_scope(),
             ZAI("model-publisher"),
             [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-packed-kld.json",
                  "19766e5e9643dbe940c05deaee7c3085f9ee339553da35ead973c825adddfef2",
                  "quant-pipeline.glm53-packed-kld-receipt.v1 pins source_revision a6c167b6..."),
              src("hf_file", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/resolve/95f4fdd94bf29989db2e0d1054e4931f55edb6aa/backend.json",
                  None, "brandonmusic's teacher capture backend, index_sha256 e6007bd5..., config_sha256 33e63ec7...")],
             [disc("record_note", "info",
                   "The revision the K6 / Dione / brandonmusic chain pins as 'the BF16 teacher weights'.")],
             weights_extra={"index_sha256": "e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f",
                            "config_sha256": "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943",
                            "size_basis": "unknown"},
             availability={"status": "public", "uri": "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(A_BF16_B1, GLM, "GLM-5.3-Flash BF16 @b1967181", "base",
             hf("zai-org/GLM-5.3-Flash-BF16", "b1967181a3917ae70a437f4884748f6b8e3a1f4d", "revision_txt"),
             "safetensors", "BF16", None, codec("bf16", None), native_scope(),
             ZAI("model-publisher"),
             [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json",
                  None, "glm53flash-fidelity-report/2 reference_identity.model_revision = b1967181..., "
                        "index_sha256 e6007bd5..., model_revision_source revision_txt")],
             [disc("record_note", "info",
                   "A DIFFERENT pinned revision of the same repository from artifact--zai-org.glm-5.3-flash-bf16.a6c167b6. "
                   "index_sha256 and config_sha256 agree between the two; the Hub revisions do not. Both are kept as "
                   "separate artifacts because a registry that silently merged them would be asserting an identity "
                   "nobody has verified. They back different panels, so no table mixes them.")],
             weights_extra={"index_sha256": "e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f",
                            "config_sha256": "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943",
                            "size_basis": "unknown"},
             availability={"status": "public", "uri": "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(A_FP8, GLM, "GLM-5.3-Flash official FP8", "quant",
             hf("zai-org/GLM-5.3-Flash", "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a", "revision_txt"),
             "safetensors", "FP8", 328366171529,
             codec("fp8_e4m3", 8.0, 8.0, tool="unknown (publisher's own pipeline)"),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16"), asg("attn.qkv", "quantized", "fp8_e4m3", 8.0),
                 asg("attn.o", "quantized", "fp8_e4m3", 8.0), asg("mlp.gate", "quantized", "fp8_e4m3", 8.0),
                 asg("mlp.up", "quantized", "fp8_e4m3", 8.0), asg("mlp.down", "quantized", "fp8_e4m3", 8.0),
                 asg("moe.experts", "quantized", "fp8_e4m3", 8.0), asg("norm", "native", "bf16"),
                 asg("lm_head", "native", "bf16"),
             ], "native", kv="bf16"),
             ZAI("quantizer"),
             [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json",
                  None, "candidate_identity.model_revision 3f1971b7..., index_sha256 3c3f4036..."),
              src("url", "https://huggingface.co/api/models/zai-org/GLM-5.3-Flash?blobs=true&revision=3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
                  None, "byte total 328,366,171,529 over 72 files at the MEASURED revision, read from the Hub API")],
             [disc("record_note", "info",
                   "The Hub head has moved past the measured revision (04c4e9e9 at mining); the byte total "
                   "recorded here is the one at 3f1971b7, the revision the measurements name.")],
             weights_extra={"shard_count": 72, "size_basis": "repo_all_files",
                            "index_sha256": "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"},
             availability={"status": "public", "uri": "https://huggingface.co/zai-org/GLM-5.3-Flash"},
             cross_refs=lair(model_id="glm-5.3-flash", url="https://huggingface.co/datasets/0xSero/local-ai-registry",
                            confidence="unverified"),
             seal={"sealed": False}),
    artifact(A_K6, GLM, "malaiwah GLM-5.3-Flash TR3 6bpw (K6)", "quant",
             hf("malaiwah/GLM-5.3-Flash-TR3-6bpw", None, "none"),
             "exl3", "6bpw", 253536370680,
             codec("exl3-mcg", 6.0, None, tool="exllamav3"),
             EXL3_SCOPE_UNIFORM(6.0), MAL("quantizer"),
             [src("hf_file", "https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw/blob/main/receipts/materialization-receipt.json",
                  None, "output_logical_bytes 253,536,370,680 and 120 shard sha256 values"),
              src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-packed-kld.json",
                  "19766e5e9643dbe940c05deaee7c3085f9ee339553da35ead973c825adddfef2",
                  "student_checkpoint_identity_sha256 a8668be3...")],
             [REV_UNPINNED("Identity rests on student_checkpoint_identity_sha256 a8668be3... and the "
                           "materialization receipt's 120 shard digests."),
              disc("size_unverified", "info",
                   "253,536,370,680 is the materialization receipt's tensor payload. The Hub safetensors sum is "
                   "253,555,566,680 and the all-files sum 253,691,838,479; the differences are container framing "
                   "and repo metadata, not weights.")],
             weights_extra={"size_basis": "tensor_payload", "shard_count": 120},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw"},
             cross_refs=lair(),
             seal={"sealed": True, "receipts": [
                 src("hf_file", "https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw/blob/main/receipts/k6-packed-kld.json",
                     "19766e5e9643dbe940c05deaee7c3085f9ee339553da35ead973c825adddfef2")],
                 "note": "reader ABI 3d659542..., runtime reader 1ccce446..., reader audit receipt c986a0a9..."}),
    artifact(A_K8, GLM, "malaiwah GLM-5.3-Flash TR3 8bpw (K8)", "quant",
             hf("malaiwah/GLM-5.3-Flash-TR3-8bpw", None, "none", status="unavailable",
                reason="HTTP 401 unauthenticated at seeding time: private, or not yet created."),
             "exl3", "8bpw", 331449761784,
             codec("exl3-mcg", 8.0, None, tool="exllamav3"),
             EXL3_SCOPE_UNIFORM(8.0), MAL("quantizer"),
             [src("private_communication", "operator inventory, 2026-08-28",
                  None, "materialization facts: output_logical_bytes 331,449,761,784; 37,152 routed choices; "
                        "1,618 native (non-routed) checkpoint tensors; bits 8; qualified_tp_sizes []"),
              src("receipt_file", STREAM_K8_RECEIPT, STREAM_K8_RECEIPT_SHA,
                  "malaiwah.glm53-k8-packed-kld-summary.v1: student_label uniform-k8, profile k8-tp4")],
             # 2026-08-28: qualification_pending is GONE because it is no longer true -- this
             # artifact now carries a measurement row. The size is recorded instead of left null:
             # it is the materialization receipt's own output_logical_bytes and it closes on its
             # own arithmetic, 19,339,524,984 native + 37,152 x 8,400,900 routed = 331,449,761,784,
             # which is a check the number either passes or fails. What is still missing is an
             # independent look at the repository, and that is what size_unverified now says.
             [REV_UNPINNED("The repository returns HTTP 401 unauthenticated, so no commit sha could "
                           "be read; identity rests on the materialization receipt's bits=8, 37,152 "
                           "routed choices and 1,618 native tensors, and on the scope below."),
              disc("size_unverified", "caveat",
                   "331,449,761,784 is the materialization receipt's output_logical_bytes and closes on its own "
                   "arithmetic (native 19,339,524,984 + 37,152 routed choices x 8,400,900 bytes). It has NOT been "
                   "confirmed against the repository, which returns HTTP 401 unauthenticated, and it is a tensor "
                   "payload total -- not a safetensors sum and not an all-files sum, both of which would be larger."),
              ],
             weights_extra={"size_basis": "tensor_payload",
                            "tensor_parallel": {"pre_sliced": False, "world_size": None}},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "private", "uri": None}, cross_refs=lair(), seal={"sealed": False}),
    artifact(A_DIONE, GLM, "0xSero GLM-5.3-Flash EXL3 Q4 (Dione, TP4-sliced)", "quant",
             hf("0xSero/GLM-5.3-Flash-EXL3-Q4", "99cccdf0e8741715662c383828a9ea601990c125", "hf_api"),
             "exl3", "Q4", 187607584245,
             codec("exl3-mcg", 4.0, None, tool="exllamav3"),
             unknown_scope("exl3-mcg", 4.0, kv="unknown", head="unknown",
                           note="the release's exl3-manifest.json / PUBLIC_RELEASE_MANIFEST.json declares a scope "
                                "policy that was not parsed into this registry; recorded as unknown rather than guessed"),
             SERO("quantizer"),
             [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json",
                  "d18b37d8ed1ba90ed837d1fb2adca0b90999b2d702613f6730ef87fe23d9f9b7", "malaiwah.glm53-dione-q4-packed-kld-summary.v1: dione_repo, dione_revision 99cccdf0..., "
                        "dione_shard_hash_verification=full"),
              src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json"),
              src("url", "https://huggingface.co/api/models/0xSero/GLM-5.3-Flash-EXL3-Q4?blobs=true",
                  None, "498 files, 217 safetensors; all-files sum 187,607,584,245; safetensors sum 187,453,172,472")],
             [INCOMPLETE,
              disc("unsealed_source", "caveat",
                   "The Dione checkpoint ships no upstream receipts, reconstruction closures or sealed reader ABI. "
                   "The packed surface was decoded WITHOUT seal verification; the immutable repo revision and the "
                   "consumed payload sha256s were recorded instead (dione_shard_hash_verification: full).", True),
              disc("tp_sliced_artifact", "info",
                   "Shipped pre-sliced for TP4 (per-layer part-0..part-3 side files). That is artifact identity, "
                   "not a runtime option.")],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 217,
                            "tensor_parallel": {"pre_sliced": True, "world_size": 4}},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public", "uri": "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4"},
             cross_refs=lair(model_id="glm-5.3-flash", url="https://huggingface.co/datasets/0xSero/local-ai-registry",
                            confidence="probable"),
             seal={"sealed": False, "note": "unsealed source; see the unsealed_source disclosure"}),
    artifact(A_B4, GLM, "brandonmusic GLM-5.3-Flash tr3 4bpw", "quant",
             hf("brandonmusic/GLM-5.3-Flash-tr3-4bpw", None, "none"),
             "exl3", "4bpw", 175642157700,
             codec("exl3-mcg", 4.0, None, tool="exllamav3"),
             EXL3_SCOPE_UNIFORM(4.0), BRANDON("quantizer"),
             [src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json",
                  "d955bfaedad36ad9841c30808c67fc36b72017f87b720fb460d8e1c13fe75e57",
                  "student_checkpoint_identity_sha256 598ce08d..., student_label uniform-k4, profile k4-tp2")],
             [REV_UNPINNED("His receipt pins the checkpoint by student_checkpoint_identity_sha256 598ce08d... "
                           "Our own crosscheck notes his metadata records an earlier repo revision and that the "
                           "weights were never modified post-upload (config/template churn only)."),
              disc("size_unverified", "info",
                   "Byte total is the Hub safetensors sum observed during mining, at a Hub head later than the "
                   "measurement; treat as approximate.")],
             weights_extra={"size_basis": "repo_weight_files"},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public", "uri": "https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw"},
             cross_refs=lair(), seal={"sealed": True, "note": "ships its own five-cold-run receipt and reader digest 1fb3be87..."}),
    artifact(A_FP8_MLAKV, GLM, "GLM-5.3-Flash official FP8 weights served with FP8 MLA KV", "quant",
             hf("zai-org/GLM-5.3-Flash", None, "none"),
             "safetensors", "FP8 + FP8 MLA KV", None,
             codec("fp8_e4m3", 8.0, 8.0),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16"), asg("attn.qkv", "quantized", "fp8_e4m3", 8.0),
                 asg("attn.o", "quantized", "fp8_e4m3", 8.0), asg("mlp.gate", "quantized", "fp8_e4m3", 8.0),
                 asg("mlp.up", "quantized", "fp8_e4m3", 8.0), asg("mlp.down", "quantized", "fp8_e4m3", 8.0),
                 asg("moe.experts", "quantized", "fp8_e4m3", 8.0), asg("norm", "native", "bf16"),
                 asg("lm_head", "native", "bf16"), asg("kv_cache", "quantized", "fp8_e4m3", 8.0),
             ], "native", kv="fp8_e4m3", mtp=False),
             ZAI("quantizer"),
             [src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v75/kld/fp8-five-run-kld.json",
                  "409a3487925a98b40d97c174b5e44e2b3526794d14c5e7ef5a35fd5f669b3209",
                  "regime: 'FP8 MLA NoPE ... eager no-MTP, full 2048-token window'")],
             [REV_UNPINNED("brandonmusic's runtime receipts record no zai-org revision."),
              disc("record_note", "info",
                   "Same published weights as artifact--zai-org.glm-5.3-flash-fp8; the difference is the declared "
                   "serving numerics (FP8 MLA KV cache, no MTP). That is why it is a separate artifact record: its "
                   "scope_digest differs at kv=.")],
             weights_extra={"size_basis": "unknown"},
             derived_from_artifact_ref=A_FP8,
             availability={"status": "public", "uri": "https://huggingface.co/zai-org/GLM-5.3-Flash"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(A_NVFP4_BM, GLM, "brandonmusic GLM-5.3-Flash NVFP4 runtime build", "quant",
             hf(None, None, "none", status="unavailable",
                reason="No published NVFP4 checkpoint repository was located. The artifact exists as a serving "
                       "configuration inside his runtime image, not as downloadable weights."),
             "other", "NVFP4 + NVFP4 MLA KV", None,
             codec("nvfp4", 4.0, None),
             scope("mixed", [
                 asg("embed_tokens", "unknown", "unknown"), asg("attn.qkv", "quantized", "nvfp4", 4.0),
                 asg("attn.o", "quantized", "nvfp4", 4.0), asg("mlp.gate", "quantized", "nvfp4", 4.0),
                 asg("mlp.up", "quantized", "nvfp4", 4.0), asg("mlp.down", "quantized", "nvfp4", 4.0),
                 asg("moe.experts", "quantized", "nvfp4", 4.0), asg("lm_head", "unknown", "unknown"),
                 asg("kv_cache", "quantized", "nvfp4", 4.0),
             ], "unknown", kv="nvfp4", mtp=False),
             BRANDON("quantizer"),
             [src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/nvfp4-five-run-kld-receipt.json",
                  "c01cc32afb1802eaba317edc3c1ef90ae649f368307ff5c8957f37bccac78755")],
             [INCOMPLETE,
              REV_UNPINNED("There is no repository, so there is no revision either."),
              disc("artifact_identity_incomplete", "caveat",
                   "No downloadable checkpoint exists. The artifact identity is 'brandonmusic's NVFP4 build served "
                   "by runtime image vNN' and nothing more; the image version is carried on the pipeline record, "
                   "which is what actually varies between the v44, v71 and v75 rows.", True)],
             weights_extra={"size_basis": "unknown"},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "unknown", "uri": None}, cross_refs=lair(), seal={"sealed": False}),
    artifact(A_FP8_DEQ, GLM, "GLM-5.3-Flash FP8 dequantized to BF16 (orcarouter reference)", "dequantized",
             hf("zai-org/GLM-5.3-Flash", None, "none"),
             "mlx", "FP8 dequantized to BF16", 328000000000,
             codec("bf16", None),
             native_scope(), ORCA("toolchain-author"),
             [src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX",
                  None, "'All three tables compare each build against the full FP8 reference (dequantized to BF16 "
                        "and run through the identical glm5_next forward, so the only variable is the quantization).'")],
             [disc("different_reference_kind", "caveat",
                   "This is NOT a BF16 teacher. It is the official FP8 release dequantized to BF16. A student "
                   "measured against it scores systematically LOWER than the same student measured against true "
                   "BF16, because the FP8 error is in the reference rather than in the student. Every number "
                   "against it is quarantined from the BF16-teacher tables.", True),
              disc("size_unverified", "caveat",
                   "328 GB is the card's own decimal-GB figure for the FP8 reference, not a byte count we read.")],
             weights_extra={"size_basis": "unknown"},
             derived_from_artifact_ref=A_FP8,
             availability={"status": "public", "uri": "https://huggingface.co/zai-org/GLM-5.3-Flash"},
             cross_refs=lair(), seal={"sealed": False}),
]

_ORCA_BYTES = {"6-bit": 295627484591, "4-bit": 204023874943, "3-bit": 184306492535,
               "2-bit": 145006896343, "2bit-lite": 102460072185}
_ORCA_BPW = {"6-bit": 6.0, "4-bit": 4.0, "3-bit": 3.0, "2-bit": 2.0, "2bit-lite": 2.0}
for build, aid in ORCA_IDS.items():
    ARTIFACTS.append(artifact(
        aid, GLM, "orcarouter GLM-5.3-Flash-MLX %s" % build, "quant",
        hf("orcarouter/GLM-5.3-Flash-MLX", "c80f6810b1a95b5be9042761becc6aa78d189782", "hf_api",
           path=build + "/"),
        "mlx", build, _ORCA_BYTES[build],
        codec("mlx-affine", _ORCA_BPW[build], None, tool="OrcaSAQ (mlx-lm derivative)"),
        scope("mixed", [
            asg("embed_tokens", "unknown", "unknown"),
            asg("attn.qkv", "unknown", "unknown"), asg("attn.o", "unknown", "unknown"),
            asg("mlp.gate", "quantized", "mlx-affine", _ORCA_BPW[build]),
            asg("mlp.up", "quantized", "mlx-affine", _ORCA_BPW[build]),
            asg("mlp.down", "quantized", "mlx-affine", _ORCA_BPW[build]),
            asg("moe.experts", "quantized", "mlx-affine", _ORCA_BPW[build]),
            asg("mtp", "quantized", "mlx-affine", _ORCA_BPW[build],
                note="layer 45 is included inside the quantized weights rather than exported separately"),
            asg("lm_head", "unknown", "unknown"),
        ], "unknown", kv="unknown"),
        ORCA("quantizer"),
        [src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX"),
         src("url", "https://huggingface.co/api/models/orcarouter/GLM-5.3-Flash-MLX?blobs=true",
             None, "per-subfolder byte totals read from the Hub API")],
        [INCOMPLETE,
         disc("record_note", "info",
              "Architecture-aware mixed precision: the card publishes a 173-entry per-tensor-pattern override "
              "map but not a class-by-class allocation, so per-class treatments are recorded as unknown. "
              "Quantized from the official FP8 release, NOT from BF16.")],
        weights_extra={"size_basis": "repo_weight_files"},
        derived_from_artifact_ref=A_FP8,
        availability={"status": "public", "uri": "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX"},
        cross_refs=lair(), seal={"sealed": False}))

# --- Qwen3.8-27B artifacts -------------------------------------------------
QREC = lambda f, note=None: src(
    "receipt_file", "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/" + f, None, note)

QWEN_NOREV = REV_UNPINNED(
    "Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on "
    "index_sha256 and the per-shard sha256 map the receipt carries.")

EXL3 = lambda cls, bits: asg(cls, "quantized", "exl3-mcg", bits)

ARTIFACTS += [
    artifact(Q_BF16, QWN, "Qwen3.8-27B BF16", "base", hf("Qwen/Qwen3.8-27B", None, "none"),
             "safetensors", "BF16", None, codec("bf16", None), native_scope(), QWEN("model-publisher"),
             [QREC("kld5-10M-fp8.json", "reference_identity index_sha256 77042094..., config_sha256 191e0af2...")],
             [QWEN_NOREV],
             weights_extra={"size_basis": "unknown",
                            "index_sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
                            "config_sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"},
             availability={"status": "public", "uri": "https://huggingface.co/Qwen/Qwen3.8-27B"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(Q_FP8, QWN, "Qwen3.8-27B FP8 (official)", "quant", hf("Qwen/Qwen3.8-27B-FP8", None, "none"),
             "safetensors", "FP8", 30890049597, codec("fp8_e4m3", 8.0, 8.0),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16"), asg("attn.qkv", "quantized", "fp8_e4m3", 8.0),
                 asg("attn.o", "quantized", "fp8_e4m3", 8.0), asg("mlp.gate", "quantized", "fp8_e4m3", 8.0),
                 asg("mlp.up", "quantized", "fp8_e4m3", 8.0), asg("mlp.down", "quantized", "fp8_e4m3", 8.0),
                 asg("norm", "native", "bf16"), asg("lm_head", "native", "bf16"),
             ], "native", kv="bf16"),
             QWEN("quantizer"),
             [QREC("kld5-10M-fp8.json", "candidate_identity index_sha256 f0838c76..., config_sha256 74227dd6...")],
             [QWEN_NOREV],
             weights_extra={"size_basis": "repo_all_files",
                            "index_sha256": "f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/Qwen/Qwen3.8-27B-FP8"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(Q_K5K6, QWN, "malaiwah Qwen3.8-27B EXL3 K5K6", "quant",
             hf("malaiwah/Qwen3.8-27B-EXL3-K5K6", None, "none"),
             "exl3", "K5/K5/K6", 30597231933, codec("exl3-mcg", 5.0, None, tool="exllamav3"),
             scope("mixed", [
                 asg("embed_tokens", "native", "bf16"), asg("attn.qkv", "native", "bf16"),
                 asg("attn.o", "native", "bf16"), EXL3("mlp.gate", 5.0), EXL3("mlp.up", 5.0),
                 EXL3("mlp.down", 6.0), asg("norm", "native", "bf16"), EXL3("lm_head", 6.0),
                 asg("mtp", "native", "bf16"),
             ], "quantized", kv="bf16", mtp=True),
             MAL("quantizer"),
             [QREC("kld5-10M-k5k6.json", "candidate_index_sha256 f8ca5af9..."),
              src("model_card", "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6",
                  None, "MODEL_CARD-K5K6.md: 'Attention weights ship in BF16'; MLP gate/up K5, down K6, lm_head K6/mcg")],
             [QWEN_NOREV], weights_extra={"size_basis": "tensor_payload"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6"},
             cross_refs=lair(), seal={"sealed": True}),
    artifact(Q_HYD, QWN, "malaiwah Qwen3.8-27B EXL3 K5K6 hydrated", "quant",
             hf("malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated", None, "none"),
             "exl3", "K5/K5/K6 + K6 attention", 21610933884, codec("exl3-mcg", 5.0, None, tool="exllamav3"),
             scope("mixed", [
                 asg("embed_tokens", "native", "bf16"), EXL3("attn.qkv", 6.0), EXL3("attn.o", 6.0),
                 EXL3("mlp.gate", 5.0), EXL3("mlp.up", 5.0), EXL3("mlp.down", 6.0),
                 asg("norm", "native", "bf16"), EXL3("lm_head", 6.0), EXL3("mtp", 5.0),
             ], "quantized", kv="bf16", mtp=True),
             MAL("quantizer"),
             [QREC("kld5-10M-hyd.json"),
              src("model_card", "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated",
                  None, "attention EXL3 K6 serialized on disk (calibrated), quantized MTP")],
             [QWEN_NOREV], weights_extra={"size_basis": "tensor_payload"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated"},
             cross_refs=lair(), seal={"sealed": True}),
    artifact(Q_CTX, QWN, "malaiwah Qwen3.8-27B EXL3 K5K6 context", "quant",
             hf("malaiwah/Qwen3.8-27B-EXL3-K5K6-context", None, "none"),
             "exl3", "K5/K5/K6 + K5 attention", 20696053306, codec("exl3-mcg", 5.0, None, tool="exllamav3"),
             scope("mixed", [
                 asg("embed_tokens", "native", "bf16"), EXL3("attn.qkv", 5.0), EXL3("attn.o", 5.0),
                 EXL3("mlp.gate", 5.0), EXL3("mlp.up", 5.0), EXL3("mlp.down", 6.0),
                 asg("norm", "native", "bf16"), EXL3("lm_head", 6.0), EXL3("mtp", 5.0),
             ], "quantized", kv="bf16", mtp=True),
             MAL("quantizer"),
             [QREC("kld5-10M-ctx.json", "candidate_index_sha256 cd53b8e4..."),
              src("model_card", "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context")],
             [QWEN_NOREV], weights_extra={"size_basis": "tensor_payload"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-context"},
             cross_refs=lair(), seal={"sealed": True}),
    artifact(Q_K4, QWN, "malaiwah Qwen3.8-27B K4", "quant", hf("malaiwah/Qwen3.8-27B-K4", None, "none"),
             "exl3", "K4", 28345369355, codec("exl3-mcg", 4.0, None, tool="exllamav3"),
             scope("mixed", [
                 asg("embed_tokens", "native", "bf16"),
                 asg("attn.qkv", "native", "bf16", note="BF16 on disk, encoded to K6 at load"),
                 asg("attn.o", "native", "bf16", note="BF16 on disk, encoded to K6 at load"),
                 EXL3("mlp.gate", 4.0), EXL3("mlp.up", 4.0), EXL3("mlp.down", 4.0),
                 asg("norm", "native", "bf16"), EXL3("lm_head", 6.0), asg("mtp", "native", "bf16"),
             ], "quantized", kv="bf16", mtp=True),
             MAL("quantizer"),
             [QREC("kld5-10M-k4.json"), src("model_card", "https://huggingface.co/malaiwah/Qwen3.8-27B-K4",
                                            None, "MODEL_CARD-K4.md: MLP all K4, lm_head K6, attention BF16 on disk")],
             [QWEN_NOREV,
              disc("size_unverified", "info",
                   "The K4 release evidence records no disk byte total; the Hub all-files sum is the only figure.")],
             weights_extra={"size_basis": "repo_all_files"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/Qwen3.8-27B-K4"},
             cross_refs=lair(), seal={"sealed": True}),
    artifact(Q_K6P, QWN, "malaiwah Qwen3.8-27B EXL3 K6-parity", "quant",
             hf("malaiwah/Qwen3.8-27B-EXL3-K6-parity", "a34ebcea909e43b3eb5b66b43782d9a509bda14b", "hf_api"),
             "exl3", "K6", 23059333816, codec("exl3-mcg", 6.0, None, tool="exllamav3"),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16"), EXL3("attn.qkv", 6.0), EXL3("attn.o", 6.0),
                 EXL3("mlp.gate", 6.0), EXL3("mlp.up", 6.0), EXL3("mlp.down", 6.0),
                 asg("norm", "native", "bf16"), EXL3("lm_head", 6.0), EXL3("mtp", 6.0),
             ], "quantized", kv="bf16", mtp=True),
             MAL("quantizer"),
             [QREC("kld5-1M-k6parity.json", "candidate_index_sha256 a35eb2fe..."),
              src("model_card", "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K6-parity",
                  None, "MODEL_CARD-K6-parity.md: full_attention and linear_attention K6 serialized+calibrated, "
                        "mlp gate/up/down K6, lm_head K6/mcg, MTP mlp K6/K6/K6")],
             [disc("record_note", "info",
                   "Revision a34ebcea is the publication receipt's; the Hub head has since moved by 40,966 B of "
                   "card and doc edits only.")],
             weights_extra={"size_basis": "repo_all_files"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K6-parity"},
             cross_refs=lair(), seal={"sealed": True}),
    artifact(Q_NVFP4, QWN, "unsloth Qwen3.8-27B NVFP4", "quant",
             hf("unsloth/Qwen3.8-27B-NVFP4", "9c73e2daee1d0fd494ffbd1d8753f2174a953796", "reported_by_author"),
             "safetensors", "NVFP4", None, codec("nvfp4", 4.0, None, tool="llm-compressor (compressed-tensors)"),
             unknown_scope("nvfp4", 4.0, kv="bf16", head="unknown", mtp=True),
             UNSLOTH("quantizer"), [QREC("kld5-10M-nvfp4.json", "candidate shard sha256 c473512c... / 1d8268aa...")],
             [INCOMPLETE], weights_extra={"size_basis": "unknown"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public", "uri": "https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4"},
             cross_refs=lair(), seal={"sealed": False}),
    artifact(Q_GT5090, QWN, "gittensor-model-hub Qwen3.8-27B NVFP4 (RTX5090)", "quant",
             hf("gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090", "69274a0d8dff5dd35bcee8290612f71e03b6e981",
                "reported_by_author"),
             "safetensors", "NVFP4", 20616833355, codec("nvfp4", 4.0, None, tool="ModelOpt"),
             unknown_scope("nvfp4", 4.0, kv="bf16", head="unknown"),
             GITTENSOR("quantizer"), [QREC("kld5-1M-gt5090.json", "3 candidate shard sha256 values recorded")],
             [INCOMPLETE,
              disc("size_unverified", "info",
                   "The receipt's byte total at the measured revision is used; the Hub head has since moved.")],
             weights_extra={"size_basis": "repo_all_files"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public",
                           "uri": "https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"},
             cross_refs=lair(), seal={"sealed": False}),
]

for aid, branch, sha, size, bpw, rec in (
        (Q_T5, "5.00bpw", "a35e75a73baee51da709329d19294245cbeeb5d8", 19925543918, 5.0, "kld5-1M-turbo5.json"),
        (Q_T6, "6.00bpw", "d32ba0bbd17de6bed8d5bbfb8c19f16f228f67ff", 22966414310, 6.0, "kld5-1M-turbo6.json")):
    ARTIFACTS.append(artifact(
        aid, QWN, "turboderp Qwen3.8-27B exl3 %s" % branch, "quant",
        hf("turboderp/Qwen3.8-27B-exl3", sha, "hf_api", path="branch:" + branch),
        "exl3", branch, size, codec("exl3-mcg", bpw, None, tool="exllamav3"),
        unknown_scope("exl3-mcg", bpw, kv="bf16", head="unknown"),
        TURBO("quantizer"), [QREC(rec, "candidate shard sha256 map recorded in the receipt"),
                             src("url", "https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/%s" % branch)],
        [INCOMPLETE,
         disc("revision_unpinned", "caveat",
              "The measurement receipt records no Hub revision. The revision here is the head of the '%s' branch "
              "observed on the Hub, corroborated by the archival mirror name we created at measurement time. "
              "It is a strong but not receipt-sealed link." % branch, True)],
        weights_extra={"size_basis": "repo_all_files"},
        derived_from_artifact_ref=Q_BF16,
        availability={"status": "public", "uri": "https://huggingface.co/turboderp/Qwen3.8-27B-exl3"},
        cross_refs=lair(), seal={"sealed": False}))

for aid, fname, size, fsha, label, bpw, fam, rec in (
        (Q_GGUF_Q8, "Qwen3.8-27B-Q8_0.gguf", 29047086048,
         "a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348", "Q8_0", 8.0, "gguf-k-quant", "gguf-report-q8_0.json"),
        (Q_GGUF_Q6, "Qwen3.8-27B-Q6_K.gguf", 22884408288,
         "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727", "Q6_K", 6.0, "gguf-k-quant", "gguf-report-q6_k.json"),
        (Q_GGUF_Q5, "Qwen3.8-27B-UD-Q5_K_XL.gguf", 20218178624,
         "176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab", "UD-Q5_K_XL", 5.0, "gguf-k-quant", "gguf-report-q5_k_xl.json"),
        (Q_GGUF_BF16, "Qwen3.8-27B-BF16-00001-of-00002.gguf", 54657735616,
         "b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f", "BF16", None, "bf16", "gguf-report-engine-floor.json")):
    is_base = bpw is None
    ARTIFACTS.append(artifact(
        aid, QWN, "unsloth Qwen3.8-27B-GGUF %s" % label, "base" if is_base else "quant",
        hf("unsloth/Qwen3.8-27B-GGUF", "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe", "hf_api", path=fname),
        "gguf", label, size, codec(fam, None if is_base else bpw),
        native_scope("bf16") if is_base else unknown_scope(fam, bpw, kv="bf16", head="unknown"),
        UNSLOTH("quantizer"), [QREC(rec, "candidate_identity.shard_sha256 pins the exact .gguf file")],
        ([disc("record_note", "info",
               "The unquantized BF16 GGUF. It exists in this registry only as the CROSS-ENGINE FLOOR: what "
               "llama.cpp and vLLM disagree by on identical unquantized weights.")]
         if is_base else
         [INCOMPLETE,
          disc("record_note", "info",
               "GGUF k-quants use a per-tensor mixed scheme that the release does not publish class by class.")]),
        weights_extra={"size_basis": "repo_weight_files", "shard_sha256": {fname: fsha},
                       "shard_count": 2 if is_base else 1},
        derived_from_artifact_ref=Q_BF16,
        availability={"status": "public", "uri": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"},
        cross_refs=lair(), seal={"sealed": False}))

for aid, name, path, fam, bpw, rec, shards in (
        (Q_AWQ, "Qwen3.8-27B AWQ-INT4 (upstream unattributed)", "/models/Qwen3.8-27B-AWQ-INT4",
         "awq", 4.0, "kld5-1M-awq.json", None),
        (Q_MTP, "Qwen3.8-27B MTP-NVFP4 (upstream unattributed)", "/models/Qwen3.8-27B-MTP-NVFP4",
         "nvfp4", 4.0, "kld5-1M-saka.json",
         {"model.safetensors": "0e1597fc7835a5a7578243809420f88ae06732733a716e49629392e571a62f76",
          "model-mtp-bf16.safetensors": "90fa0e3eed5a647c035c6df9ecabc416c0f8d573ff84ac12485b085f00a7cdf2"})):
    ARTIFACTS.append(artifact(
        aid, QWN, name, "quant",
        hf(None, None, "none", status="unknown",
           reason="The measurement receipt records only a local path (%s) and model_revision=null. An upstream "
                  "repository id circulates in our own landscape notes but is not recorded by any receipt, so it "
                  "is NOT asserted here." % path),
        "safetensors", "INT4" if fam == "awq" else "NVFP4", None, codec(fam, bpw),
        unknown_scope(fam, bpw, kv="bf16", head="unknown"),
        attr("unknown", "quantizer", handle=None, url=None),
        [QREC(rec, "candidate_identity.model_path %s, model_revision null" % path)],
        [INCOMPLETE,
         disc("revision_unpinned", "caveat",
              "Neither a repository nor a revision is recorded by the measurement receipt. The MEASUREMENT is "
              "ours and real; the upstream artifact identity is not established. Deliberately seeded with "
              "repository=null rather than with a guessed repo id.", True)],
        weights_extra={"size_basis": "unknown", "shard_sha256": shards or {}},
        derived_from_artifact_ref=Q_BF16,
        availability={"status": "unknown", "uri": None}, cross_refs=lair(), seal={"sealed": False}))

# ===========================================================================
# 4. REFERENCES (teacher captures)
# ===========================================================================
R_B25 = "reference--brandonmusic.glm53-bf16-fp32-logits.final25"
R_B1W = "reference--brandonmusic.glm53-bf16-fp32-logits.final-0000"
R_G10M = "reference--malaiwah.glm53-bf16-vllm.suite-v5-10m"
R_G10M_W1024 = "reference--malaiwah.glm53-bf16-vllm.suite-v5-10m.scorefrom1024"
R_ORCA = "reference--orcarouter.glm53-fp8-dequantized.undisclosed"
R_Q10M = "reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m"
R_Q1M = "reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m"
R_Q2M = "reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m"
R_Q1M_W256 = "reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256"
R_Q1M_W1024 = "reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024"

M_FLOOR_GLM = "measurement--glm53.bf16-replay-floor.brandonmusic-final25"
M_FLOOR_GGUF = "measurement--qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m"

BM_CAPTURE = {
    "stack": "transformers", "stack_version": "5.16.1", "pipeline_ref": None,
    "compute_dtype": "bf16", "logits_dtype": "fp32", "kv_cache_dtype": "not_applicable",
    "head_source": "own_head", "head_sha256": None, "batch_invariant": None,
    "capture_receipt_sha256": "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5",
}

REFERENCES = [
    {"schema_version": V, "id": R_B25,
     "name": "brandonmusic BF16 fp32 teacher logits over the 25 final windows",
     "artifact_ref": A_BF16_A6, "panel_ref": P_B25, "reference_kind": "native_bf16",
     "capture": dict(BM_CAPTURE), "author": BRANDON("measurer"), "logits_available": True,
     "self_consistency": {"floor_measurement_ref": M_FLOOR_GLM,
                          "note": "Our same-panel BF16 replay through vLLM scores 0.012712 against these stored "
                                  "logits. That is the floor any cross-stack row on this panel sits on."},
     "sources": [src("dataset_card", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"),
                 src("hf_file", "https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits/resolve/95f4fdd94bf29989db2e0d1054e4931f55edb6aa/backend.json",
                     None, "B200 x4, expert-parallel world size 4, eager attention, torch 2.11.0+cu130, "
                           "allow_tf32 false, use_cache false, stored logits float32; backend identity 85b11599...")],
     "disclosures": [disc("record_note", "info",
                          "Precomputed float32 full-vocabulary logits published as a dataset, so every student "
                          "measured against them is scored against byte-identical teacher values.")]},
    {"schema_version": V, "id": R_B1W,
     "name": "brandonmusic BF16 fp32 teacher logits, window final-0000",
     "artifact_ref": A_BF16_A6, "panel_ref": P_B1W, "reference_kind": "native_bf16",
     "capture": dict(BM_CAPTURE), "author": BRANDON("measurer"), "logits_available": True,
     "self_consistency": {"floor_measurement_ref": None,
                          "note": "No same-stack self-consistency floor was measured on the single-window panel."},
     "sources": [src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/nvfp4-dynamic-scale-control-kld-report.json",
                     "e5365075bccd4e27c9e7f002c23e31cc6f8df196c3c7ccf847faae4f007b22f9",
                     "teacher_path .../window-0000.safetensors, teacher_sha256 9f49af1b...")],
     "disclosures": [disc("subset_of_panel", "caveat",
                          "The same capture as the 25-window reference, restricted to window final-0000. "
                          "teacher_logits sha256 9f49af1b... pins the single window.", True)]},
    {"schema_version": V, "id": R_G10M,
     "name": "malaiwah BF16 hidden-state capture with shared head, GLM suite v5",
     "artifact_ref": A_BF16_B1, "panel_ref": P_G10M, "reference_kind": "native_bf16",
     "capture": {"stack": "vllm", "stack_version": None, "pipeline_ref": None, "compute_dtype": "bf16",
                 "logits_dtype": "fp32", "kv_cache_dtype": "bf16", "head_source": "shared_head_artifact",
                 "head_sha256": "47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0",
                 "batch_invariant": None, "capture_receipt_sha256": None},
     "author": MAL("measurer"), "logits_available": True,
     "self_consistency": {"floor_measurement_ref": None,
                          "note": "Same-stack replay: reference and candidate hidden states are captured by the "
                                  "same vLLM path and scored through one shared head, so no cross-stack floor "
                                  "term applies."},
     "sources": [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json",
                     None, "head /glm53/out/head.safetensors sha256 47eaf729..., candidate_head null")],
     "disclosures": [disc("shared_reference_head", "info",
                          "Hidden states are captured for both sides and ONE head (47eaf729...) is applied to "
                          "both. This removes head numerics from the comparison; it also means the number does "
                          "not include any error in the candidate's own lm_head.")]},
    {"schema_version": V, "id": R_G10M_W1024,
     "name": "malaiwah BF16 shared-head capture, GLM suite v5 scored from 1024",
     "artifact_ref": A_BF16_B1, "panel_ref": P_G10M_W1024, "reference_kind": "native_bf16",
     "capture": {"stack": "vllm", "stack_version": None, "pipeline_ref": None, "compute_dtype": "bf16",
                 "logits_dtype": "fp32", "kv_cache_dtype": "bf16", "head_source": "shared_head_artifact",
                 "head_sha256": "47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0",
                 "batch_invariant": None, "capture_receipt_sha256": None},
     "author": MAL("measurer"), "logits_available": True,
     "self_consistency": {"floor_measurement_ref": None, "note": "same-stack replay"},
     "sources": [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16-scorefrom1024.json")],
     "disclosures": [disc("shared_reference_head", "info", "As R_G10M; only the scored-position policy differs.")]},
    {"schema_version": V, "id": R_ORCA,
     "name": "orcarouter FP8-dequantized reference (undisclosed panel)",
     "artifact_ref": A_FP8_DEQ, "panel_ref": P_ORCA, "reference_kind": "dequantized_from_quant",
     "capture": {"stack": "mlx-lm", "stack_version": None, "pipeline_ref": None, "compute_dtype": "bf16",
                 "logits_dtype": "unknown", "kv_cache_dtype": "unknown", "head_source": "own_head",
                 "head_sha256": None, "batch_invariant": None, "capture_receipt_sha256": None},
     "author": ORCA("measurer"), "logits_available": False,
     "self_consistency": {"floor_measurement_ref": None,
                          "note": "No floor is available: neither the panel nor the capture is disclosed."},
     "sources": [src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX")],
     "disclosures": [disc("different_reference_kind", "caveat",
                          "The teacher is the official FP8 release dequantized to BF16, not a BF16 teacher. "
                          "Numbers against it are systematically smaller than the same numbers against true "
                          "BF16 and must never be ranked against native_bf16 rows -- including the other "
                          "GLM-5.3-Flash rows in this registry.", True),
                     disc("undisclosed_panel", "caveat",
                          "The capture is over an undisclosed evaluation set.", True)]},
]

_QW_CAP = {"stack": "vllm", "stack_version": None, "pipeline_ref": None, "compute_dtype": "bf16",
           "logits_dtype": "fp32", "kv_cache_dtype": "bf16", "head_source": "shared_head_artifact",
           "head_sha256": "25a30fd5f826da0abc4efc4cc71def9f02bcb8085f7175eee284d221dee4cfff",
           "batch_invariant": None, "capture_receipt_sha256": None}

for rid, pid, floor, note in ((R_Q10M, P_Q10M, None, None), (R_Q1M, P_Q1M, M_FLOOR_GGUF,
                                                             "The GGUF rows on this panel are cross-engine and "
                                                             "sit on the llama.cpp-vs-vLLM floor 0.000507."),
                              (R_Q2M, P_Q2M, None, None), (R_Q1M_W256, P_Q1M_W256, None, None),
                              (R_Q1M_W1024, P_Q1M_W1024, None, None)):
    REFERENCES.append({
        "schema_version": V, "id": rid,
        "name": "malaiwah Qwen3.8-27B BF16 shared-head capture over %s" % pid.split("--", 1)[1],
        "artifact_ref": Q_BF16, "panel_ref": pid, "reference_kind": "native_bf16",
        "capture": dict(_QW_CAP), "author": MAL("measurer"), "logits_available": True,
        "self_consistency": {"floor_measurement_ref": floor,
                             "note": note or "Same-stack replay through one shared head (25a30fd5...)."},
        "sources": [QREC("kld5-10M-fp8.json", "head /work/kld2/lm_head.safetensors sha256 25a30fd5..., "
                                              "candidate_head null, reference index_sha256 77042094...")],
        "disclosures": [disc("shared_reference_head", "info",
                             "One head (25a30fd5...) applied to both sides' hidden states."),
                        QWEN_NOREV],
    })

# ===========================================================================
# 5. PIPELINES
# ===========================================================================
PL_K6 = "pipeline--malaiwah.glm53-packed-kld"
PL_STREAM = "pipeline--malaiwah.glm53-stream-packed-kld"
PL_DIONE = "pipeline--malaiwah.glm53-dione-packed-kld"
PL_GSUITE = "pipeline--malaiwah.glm53-fidelity-replay"
PL_XCHECK = "pipeline--malaiwah.glm53-crosscheck"
PL_QLADDER = "pipeline--malaiwah.qwen38-kld-ladder"
PL_QGGUF = "pipeline--malaiwah.qwen38-gguf-cross-engine"
PL_BM_PACKED = "pipeline--brandonmusic.glm53-packed-kld"
PL_BM_TP2 = "pipeline--brandonmusic.glm53-custom-tp2-runtime"
PL_BM_V44 = "pipeline--brandonmusic.sm120-runtime.v44"
PL_BM_V71 = "pipeline--brandonmusic.sm120-runtime.v71"
PL_BM_V75 = "pipeline--brandonmusic.sm120-runtime.v75"
PL_ORCA = "pipeline--orcarouter.mlx-eval"


def pipeline(pid, name, roles, repo, revision, entrypoint, author, disclosures, **kw):
    rec = {"schema_version": V, "id": pid, "name": name, "roles": roles,
           "implementation": {"repository": repo, "revision": revision, "entrypoint": entrypoint,
                              "file_sha256": None, "container_image": None, "container_digest": None,
                              "runtime_reader_sha256": None, "dependencies": {}},
           "author": author, "disclosures": disclosures}
    rec["implementation"].update(kw.pop("impl", {}))
    rec.update(kw)
    return rec


FP64 = {"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
        "determinism_controls": ["cold_process_per_run"]}

PIPELINES = [
    pipeline(PL_K6, "malaiwah GLM-5.3-Flash packed-surface KLD scorer (k6-tp4)",
             ["replay", "scorer", "aggregator"], None, None, "tools/k6_kld_report.py", MAL("toolchain-author"),
             [disc("record_note", "info",
                   "Scores a packed EXL3 surface against brandonmusic's stored fp32 teacher logits over the "
                   "sealed token panel, in float64, five cold processes.")],
             impl={"runtime_reader_sha256": "1ccce44602d4ccf41abe594ede448bf726516ac44f67a54dcd65cc0b5bf9dd14",
                   "dependencies": {"packed_reader_abi_sha256": "3d659542e5acbf1e3436b4b01d04f7f4edbe8def1c3029fbd3a6a1976b573dee",
                                    "reader_audit_receipt_sha256": "c986a0a98d6c34d8a311401f90be24ee87e01d20602583fef5bb37d1ff504cc7"}},
             numerics={"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": ["cold_process_per_run", "fixed_batch_shape"]},
             hardware={"gpu": None, "gpu_count": None, "tensor_parallel": 4, "note": "profile k6-tp4"},
             cost={"usd_per_measurement": None, "basis": None},
             sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-five-run-kld.json",
                          "1611800a1ff37cbae5e8e46a0024fb49d62955efc682c4e609e5a6e43aa714da")],
             cross_refs=lair()),
    # The streaming lane. Everything that makes it a DIFFERENT lane from PL_K6 is a field
    # here, not an adjective: one device instead of eight, the EP8 partition emulated in
    # process rather than run across real ranks, and an fp32 routed-expert combine where the
    # sealed lane had NCCL summing bf16 per-rank partials in a topology-dependent order.
    # `lane.bridge` is what stops that being a story: it is the measured distance between
    # this lane and the sealed one on the same panel, read off the verdict receipt, together
    # with the two flags that say the run is NOT a reproduction of the sealed number.
    pipeline(PL_STREAM,
             "malaiwah GLM-5.3-Flash streaming single-GPU KLD scorer (EP8 emulated, reduce-order fp32)",
             ["replay", "scorer", "aggregator"], None, None,
             "tools/stream_score.py (single-device capture) -> tools/k6_kld_report.py --profile k6-stream "
             "(unmodified fp64 scorer)",
             MAL("toolchain-author"),
             [disc("non_sealed_lane", "caveat",
                   "This is the streaming lane, not the sealed-ep8 lane. It scores the same sealed panel "
                   "against the same stored teacher logits on ONE GPU by streaming one layer of routed "
                   "experts at a time, and it emulates the sealed run's 8-way expert-parallel partition "
                   "inside a single process. Numbers from this lane sit beside the sealed lane's, never "
                   "in place of them.", True),
              disc("local_device_reduction_order", "caveat",
                   "The one op that differs is the routed-expert combine, in each of 42 layers. The sealed "
                   "run rounded each rank's partial to bf16 and let NCCL sum the ~5 nonzero partials in an "
                   "order set by the 8-GPU NVSwitch topology; a single process cannot reproduce that order, "
                   "so this lane sums in fp32 (--reduce-order fp32). Because top-8-of-288 routing is "
                   "discontinuous in the hidden state, an ULP-scale difference there flips marginal routing "
                   "decisions downstream -- which is why the tokenwise KL array differs from the sealed one "
                   "even though the panel mean is within 8.5e-06 nats.", True)],
             impl={"dependencies": {
                 "sealed_checkpoint_identity_sha256":
                     "a8668be3592493035e98a52994e0e3c43548a9757eadb79f7ae939f2f32de1c1"}},
             numerics={"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": ["cold_process_per_run", "fixed_batch_shape"]},
             hardware={"gpu": "H200", "gpu_count": 1, "tensor_parallel": None,
                       "note": "one device; the tp4/tp8 strings in the profile names describe the sealed "
                               "partition being emulated, not a real world size"},
             cost={"usd_per_measurement": None,
                   "basis": "1x H200 spot at $1.99/h. No invoice was captured for these two runs, so no "
                            "single figure is asserted here; the measured decode is 10.94 ms/matrix over "
                            "36,288 matrices and a full-panel cold run projects at ~2.8 h (~$5.6), against "
                            "~2.37 h x 8 GPUs x 5 cold runs for the sealed lane."},
             lane={"name": "streaming", "device_count": 1, "expert_parallel_emulated": True,
                   "expert_parallel_world_size": 8, "reduce_order": "fp32",
                   "bridge": {
                       "compared_to_lane": "sealed-ep8",
                       "panel_ref": P_B25,
                       "sealed_measurement_ref": "measurement--glm53.k6-6bpw.brandonmusic-final25",
                       "delta_mean_kld": -8.495843104593809e-06,
                       "max_abs_per_window_delta": 0.00028735280093581186,
                       "windows_compared": 25,
                       "tokenwise_kld_sha256_matches_sealed": False,
                       "publishable_as_reproduction": False,
                       "verdict": "LARGER_DELTA_SEE_DISCLOSURE",
                       "evidence": [src("receipt_file", STREAM_K6_VERDICT, STREAM_K6_VERDICT_SHA,
                                        "malaiwah.glm53-streaming-measurement-verdict.v1: scored the sealed "
                                        "K6 surface (student and sealed checkpoint_identity_sha256 both "
                                        "a8668be3...), 25 per-window pairs, cross-run payload bitwise "
                                        "identical over 2 cold runs"),
                                    src("hf_file", HF_REGISTRY_RAW + STREAM_K6_VERDICT,
                                        STREAM_K6_VERDICT_SHA, "the same file, published")]}},
             sources=[src("receipt_file", STREAM_K6_RECEIPT, STREAM_K6_RECEIPT_SHA,
                          "malaiwah.glm53-k6-stream-packed-kld-summary.v1, profile k6-stream-tp4"),
                      src("receipt_file", STREAM_K8_RECEIPT, STREAM_K8_RECEIPT_SHA,
                          "malaiwah.glm53-k8-packed-kld-summary.v1, profile k8-tp4"),
                      src("receipt_file", STREAM_BF16_RECEIPT, STREAM_BF16_RECEIPT_SHA,
                          "malaiwah.glm53-native-bf16-packed-kld-summary.v1, profile native-bf16-stream -- "
                          "the reference's own unquantized weights, this lane's measurement floor"),
                      src("hf_file", HF_REGISTRY_RAW + STREAM_K6_RECEIPT, STREAM_K6_RECEIPT_SHA),
                      src("hf_file", HF_REGISTRY_RAW + STREAM_K8_RECEIPT, STREAM_K8_RECEIPT_SHA),
                      src("hf_file", HF_REGISTRY_RAW + STREAM_BF16_RECEIPT, STREAM_BF16_RECEIPT_SHA)],
             cross_refs=lair()),
    pipeline(PL_DIONE, "malaiwah Dione-surface KLD scorer (dione-q4-tp4)",
             ["replay", "scorer", "aggregator"], None, None, "tools/k6_kld_report.py (dione surface adapter)",
             MAL("toolchain-author"),
             [disc("unsealed_source", "caveat",
                   "Decodes an unsealed third-party packed surface: no upstream reader ABI to verify against, so "
                   "the adapter records consumed payload digests and the immutable repo revision instead.", True)],
             impl={"runtime_reader_sha256": "1ccce44602d4ccf41abe594ede448bf726516ac44f67a54dcd65cc0b5bf9dd14"},
             hardware={"gpu": None, "gpu_count": None, "tensor_parallel": 4, "note": "profile dione-q4-tp4"},
             sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json",
                          "d18b37d8ed1ba90ed837d1fb2adca0b90999b2d702613f6730ef87fe23d9f9b7")], cross_refs=lair()),
    pipeline(PL_GSUITE, "malaiwah GLM-5.3-Flash capture + shared-head replay + fp64 scorer",
             ["capture", "replay", "scorer", "aggregator"], None, None, "tools/fidelity_report.py",
             MAL("toolchain-author"),
             [disc("record_note", "info",
                   "Captures hidden states for reference and candidate through one vLLM path, applies one shared "
                   "head, scores in float64 two-pass over 15,488-entry vocabulary chunks, and bootstraps a 95% "
                   "interval over 837 source clusters with 10,000 resamples.")],
             numerics={"accumulation_dtype": "fp64", "two_pass": True, "vocab_chunk": 15488,
                       "determinism_controls": ["fixed_batch_shape"]},
             hardware={"gpu": "cuda", "gpu_count": None, "tensor_parallel": None},
             sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json")],
             cross_refs=lair()),
    pipeline(PL_XCHECK, "malaiwah cross-stack replay against a foreign teacher panel",
             ["capture", "replay", "scorer"], None, None, "tools/crosscheck.py", MAL("toolchain-author"),
             [disc("cross_stack_capture", "caveat",
                   "Replays a model through OUR vLLM stack and scores it against a teacher captured on a "
                   "DIFFERENT stack (transformers/eager on B200). The result contains a stack-difference term "
                   "that can only inflate it. Every measurement from this pipeline must name its floor.", True)],
             numerics={"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": []},
             hardware={"gpu": "cuda", "gpu_count": None, "tensor_parallel": None},
             sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/fp8-on-brandon-panel.json")],
             cross_refs=lair()),
    pipeline(PL_QLADDER, "malaiwah Qwen3.8-27B shared-head replay + fp64 KLD ladder",
             ["capture", "replay", "scorer", "aggregator"], None, None, "tools/kld_aggregate.py",
             MAL("toolchain-author"),
             [disc("record_note", "info",
                   "float64 two-pass over 24,832-entry vocabulary chunks; per-shard reports are recomputed from "
                   "per-context rows and cross-checked against each shard's own summary "
                   "(max relative gap 1.7e-16).")],
             numerics={"accumulation_dtype": "fp64", "two_pass": True, "vocab_chunk": 24832,
                       "determinism_controls": ["fixed_batch_shape"]},
             hardware={"gpu": "cuda", "gpu_count": None, "tensor_parallel": None},
             cost={"usd_per_measurement": None, "basis": None},
             sources=[QREC("kld5-10M-fp8.json")], cross_refs=lair()),
    pipeline(PL_QGGUF, "malaiwah llama.cpp GGUF capture + vLLM-referenced fp64 scorer",
             ["capture", "scorer"], "https://github.com/ggml-org/llama.cpp",
             "ece963f41b0b02d7a0d61436ae365762c073a4c8", "tools/gguf_capture.cpp", MAL("toolchain-author"),
             [disc("cross_engine_capture", "caveat",
                   "GGUF candidates are captured with llama.cpp (reading res->t_embd, post-final-norm) while the "
                   "reference and every EXL3/FP8 row are captured under vLLM. Every number from this pipeline "
                   "carries a llama.cpp-vs-vLLM term on top of quantization error, which can only inflate it. "
                   "It is measured: 0.000507 nats on identical unquantized weights.", True)],
             numerics={"accumulation_dtype": "fp64", "two_pass": True, "vocab_chunk": 24832,
                       "determinism_controls": []},
             hardware={"gpu": "cuda", "gpu_count": None, "tensor_parallel": None},
             sources=[QREC("cross-engine-comparator.json"), QREC("gguf-report-engine-floor.json")],
             cross_refs=lair()),
    pipeline(PL_BM_PACKED, "brandonmusic packed-surface KLD scorer (k4-tp2)",
             ["replay", "scorer", "aggregator"], None, None, None, BRANDON("toolchain-author"),
             [disc("author_reported_only", "caveat",
                   "The author's own scorer. Same token panel and same teacher receipt as ours "
                   "(token_panel_receipt_sha256 0beec577..., teacher_receipt_sha256 2ae08117... are byte-identical "
                   "in his receipt and in ours), but a different reader: 1fb3be87... vs our 1ccce446...")],
             impl={"runtime_reader_sha256": "1fb3be878e0a9445b640565558fc34715891bfd60a63e976002181c620a41a69"},
             numerics={"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": ["cold_process_per_run"]},
             hardware={"gpu": None, "gpu_count": None, "tensor_parallel": 2, "note": "profile k4-tp2"},
             sources=[src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json",
                          "d955bfaedad36ad9841c30808c67fc36b72017f87b720fb460d8e1c13fe75e57")],
             cross_refs=lair()),
    pipeline(PL_BM_TP2, "brandonmusic custom TP2 runtime window scorer",
             ["end-to-end"], None, None, None, BRANDON("toolchain-author"),
             [disc("author_reported_only", "caveat", "The author's own single-window runtime scorer."),
              disc("single_run", "caveat", "One run; no repeatability evidence.", False)],
             numerics={"accumulation_dtype": "fp64", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": []},
             hardware={"gpu": None, "gpu_count": None, "tensor_parallel": 2},
             sources=[src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/tp2-runtime-window-kld.json",
                          "a22aec25c33de1d7a2876e475ff1c45fbe500095ceb8d8f23d681c895b33cc65")],
             cross_refs=lair()),
    pipeline(PL_ORCA, "orcarouter MLX evaluation harness",
             ["end-to-end"], None, None, None, ORCA("toolchain-author"),
             [disc("author_reported_only", "caveat",
                   "The author's own harness. No entrypoint, revision, estimator precision or run count is "
                   "published; the model card gives results only.")],
             numerics={"accumulation_dtype": "unknown", "two_pass": None, "vocab_chunk": None,
                       "determinism_controls": []},
             sources=[src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX")],
             cross_refs=lair()),
]

for pid, ver, regime in (
        (PL_BM_V44, "v44", "TP2 DCP1 eager no-MTP, GPUs 2,3, exact 2048-token window / 2047 prediction positions"),
        (PL_BM_V71, "v71", "MLA NoPE, route128 SMEM, TP2/EP2, DCP2 B12X A2A eager no-MTP, full 2048-token window"),
        (PL_BM_V75, "v75", "release image, MLA NoPE, route128 SMEM/register, TP2/EP2, DCP2 direct symmetric-memory "
                           "A2A, eager no-MTP, full 2048-token window")):
    PIPELINES.append(pipeline(
        pid, "brandonmusic SM120 vLLM/EXL3 runtime image %s" % ver, ["end-to-end"],
        "https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw", None, None,
        BRANDON("toolchain-author"),
        [disc("author_reported_only", "caveat",
              "The author's own serving image at version %s. Regime as published: %s" % (ver, regime))],
        # AUDIT 2026-08-28: "fp64" here was ours, not his. The glm53-r19-runtime-kld-repeated.v1
        # receipts this pipeline produces carry no compute_dtype field, unlike his other two
        # GLM-5.3-Flash receipt families. Matches the estimator_unknown disclosure on the six
        # measurement rows this pipeline backs.
        numerics={"accumulation_dtype": "unknown", "two_pass": None, "vocab_chunk": None,
                  "determinism_controls": ["cold_process_per_run"]},
        hardware={"gpu": "SM120", "gpu_count": 2, "tensor_parallel": 2},
        sources=[src("github_file",
                     "https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/tree/main/runtime-results/%s/kld" % ver)],
        cross_refs=lair()))

# ===========================================================================
# 6. MEASUREMENTS
# ===========================================================================

def measurement(mid, model_ref, artifact_ref, panel_ref, reference_ref, pipeline_ref,
                value, *, metric_name="mean_tokenwise_kld", direction="reference_to_candidate",
                accumulation="float64", stack_relation="same_stack", head_policy="native_head",
                two_pass=None, vocab_chunk=None, top1=None, aux=None,
                ci=None, ci_method="none", clusters=None, samples=None,
                scored_positions=None, contexts=None, covers_full=True, subset_detail=None,
                position_filter="all", runs=1, cold=None, run_means=None, identical=None,
                evidence_kind="none", evidence_hashes=None, det_note=None,
                measured_by="self-measured", measurer=None, verified=False, verification=None,
                sources=None, receipt_schema=None, cls="strict", bias=None,
                gate=None, disclosures=None, status="published", notes=None, artifacts_map=None):
    art = artifacts_map[artifact_ref]
    # AUDIT 2026-08-28: logits_dtype used to be hardcoded "fp32" for every row, including
    # rows whose estimator is otherwise entirely undisclosed (orcarouter's, brandonmusic's
    # runtime series). We know it for our own scorers; for somebody else's we do not, and
    # no third-party receipt in this registry states it. Assert it only where we ran the code.
    logits_dtype = "fp32" if measured_by == "self-measured" else "unknown"
    est = {"accumulation_dtype": accumulation, "logits_dtype": logits_dtype, "two_pass": two_pass,
           "vocab_chunk": vocab_chunk, "stack_relation": stack_relation, "head_policy": head_policy}
    ki = {"panel_id": panel_ref, "reference_id": reference_ref, "metric_name": metric_name,
          "direction": direction, "accumulation_dtype": accumulation,
          "stack_relation": stack_relation, "head_policy": head_policy}
    det = {"run_count": runs, "cold_start_per_run": cold, "identical_across_runs": identical,
           "evidence_kind": evidence_kind, "evidence_hashes": evidence_hashes or [],
           "distinct_evidence_hash_count": (len(evidence_hashes) if evidence_hashes is not None else None)}
    if run_means is not None:
        det["run_means"] = list(run_means)
        det["min_run_mean"] = min(run_means)
        det["max_run_mean"] = max(run_means)
        det["population_stddev_of_run_means"] = L.population_stddev(run_means)
    if det_note:
        det["note"] = det_note
    rec = {
        "schema_version": V, "id": mid, "status": status, "supersedes": None,
        "model_ref": model_ref, "artifact_ref": artifact_ref, "panel_ref": panel_ref,
        "reference_ref": reference_ref, "pipeline_ref": pipeline_ref,
        "scope_digest": art["scope_digest"],
        "metric": {"name": metric_name, "value": value, "units": "nats", "direction": direction,
                   "higher_is_better": False},
        "auxiliary_metrics": dict(aux or {}, top1_agreement=top1),
        "uncertainty": {"method": ci_method,
                        "ci95_low": (ci[0] if ci else None), "ci95_high": (ci[1] if ci else None),
                        "clusters": clusters, "samples": samples},
        "estimator": est, "determinism": det,
        "measurement_scope": {"scored_positions": scored_positions, "contexts": contexts,
                              "positions_per_context": None, "covers_full_panel": covers_full,
                              "subset_detail": subset_detail, "position_filter": position_filter},
        "provenance": {"measured_by": measured_by,
                       "measurer": measurer or MAL("measurer"),
                       "independently_verified": verified, "verification": verification,
                       "sources": sources or [], "receipt_schema": receipt_schema},
        "comparability": {"key": L.comparability_key(ki), "key_inputs": ki, "class": cls, "bias": bias},
        "quality_gate": gate,
        "cross_refs": lair(),
        "disclosures": disclosures or NONE_DISC,
    }
    if notes:
        rec["notes"] = notes
    return rec


def build_measurements(artifacts_map):
    M = lambda *a, **k: measurement(*a, artifacts_map=artifacts_map, **k)
    out = []

    K6_SRC = [src("receipt_file", "scratchpad copy of reports/k6-five-run-kld.json",
                  "1611800a1ff37cbae5e8e46a0024fb49d62955efc682c4e609e5a6e43aa714da",
                  "byte-identical to the published copy; the receipt's own self-declared receipt_sha256 "
                  "field is 57faf356..., which is a digest of its canonical content, not of the file"),
              src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-five-run-kld.json",
                  "1611800a1ff37cbae5e8e46a0024fb49d62955efc682c4e609e5a6e43aa714da",
                  "fetched read-only and hashed during seeding; identical to the local receipt"),
              src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-packed-kld.json",
                  "19766e5e9643dbe940c05deaee7c3085f9ee339553da35ead973c825adddfef2",
                  "quant-pipeline.glm53-packed-kld-receipt.v1; self-declared receipt_sha256 25eea649...")]
    K6 = 0.013723384665701147
    out.append(M("measurement--glm53.k6-6bpw.brandonmusic-final25", GLM, A_K6, P_B25, R_B25, PL_K6, K6,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 scored_positions=51175, contexts=25, runs=5, cold=True, run_means=[K6] * 5,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["52e35723dacd0314acb85bcee86d2faefd5c12ff9d82c6e026e05d35ee15db4b"],
                 det_note="Five cold processes with five DIFFERENT student_backend_identity_sha256 values "
                          "produced one identical tokenwise KL array. The differing backend identities are what "
                          "make the identical tensor digest meaningful; the receipt-file digests also differ "
                          "per run and prove nothing.",
                 sources=K6_SRC, receipt_schema="quant-pipeline.glm53-packed-student-kld-five-cold-run.v1",
                 gate={"metric": "mean_of_five_run_mean_tokenwise_kld", "threshold_lt": 0.06,
                       "threshold_gt": None, "passed": True},
                 disclosures=[disc("no_known_deviations", "info",
                                   "Full 25-window panel, five cold runs, float64, bitwise identical.")]))

    # ---------------------------------------------------------------- streaming lane
    # Same artifacts, same panel, same teacher, DIFFERENT lane: one GPU, the sealed run's
    # 8-way expert-parallel partition emulated in process, and the routed-expert combine
    # summed in fp32 instead of by NCCL over bf16 per-rank partials. The K6 row is the one
    # that can be bridged, because a sealed-lane K6 number exists on this panel to bridge
    # against; the verdict receipt scored both surfaces and the delta is on the row as a
    # measured bias, not as prose. The K8 row has no such bridge and says so.
    M_BF16_FLOOR = "measurement--glm53.bf16-stream-floor.brandonmusic-final25"
    BF16_FLOOR = 0.011505922619330299

    SK6 = 0.013714888822596553
    STREAM_DISC = lambda measured: [
        disc("reduced_run_count", "caveat",
             "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)", True),
        disc("non_sealed_lane", "caveat",
             "Produced by the 'streaming' lane, not the sealed-ep8 lane. %s" % measured, True)]
    out.append(M("measurement--glm53.k6-6bpw-stream.brandonmusic-final25", GLM, A_K6, P_B25, R_B25,
                 PL_STREAM, SK6,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 top1=0.9656277479237909,
                 scored_positions=51175, contexts=25, runs=2, cold=True, run_means=[SK6] * 2,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["9657ede36b9f4b09a2c74916239c6d9a3baebce5f3fa64af7af388b0686aa284"],
                 det_note="2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct "
                          "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                          "nothing; the single tokenwise digest is the determinism evidence.",
                 sources=[src("receipt_file", STREAM_K6_RECEIPT, STREAM_K6_RECEIPT_SHA,
                              "malaiwah.glm53-k6-stream-packed-kld-summary.v1"),
                          src("receipt_file", STREAM_K6_VERDICT, STREAM_K6_VERDICT_SHA,
                              "malaiwah.glm53-streaming-measurement-verdict.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_K6_RECEIPT, STREAM_K6_RECEIPT_SHA),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_K6_VERDICT, STREAM_K6_VERDICT_SHA)],
                 receipt_schema="malaiwah.glm53-k6-stream-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "downward", "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": 8.495843104593809e-06,
                       "detail": "Lane offset, MEASURED not estimated: this 'streaming'-lane run scores "
                                 "0.013714888822596553 against the sealed-ep8 lane's 0.013723384665701147 on "
                                 "the same panel, a signed delta of -8.495843104593809e-06 nats (|max| "
                                 "0.00028735280093581186 on any one of 25 windows). The tokenwise KL array "
                                 "does NOT match the sealed one, and the runner's own verdict is "
                                 "publishable_as_reproduction=False, so this number stands beside the sealed "
                                 "one rather than replacing it. This lane's own measurement floor (%s) is "
                                 "%r nats; netting it out gives an estimated quantization-attributable error "
                                 "of %r nats here -- an estimate, not an identity, because KL is not "
                                 "additive, and it is only meaningful because both terms are small and share "
                                 "the same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, SK6 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": True},
                 disclosures=STREAM_DISC(
                     "On this panel the lane's offset against the sealed lane IS measured: "
                     "-8.495843104593809e-06 nats on the mean (max 0.00028735280093581186 on any one "
                     "window over 25 windows), and the tokenwise KL array is NOT the sealed one, so the "
                     "run is not a reproduction of the sealed number."),
                 notes="Provenance of the fields the summary receipt does not carry. metric.direction and "
                       "estimator.accumulation_dtype: SUPPLIED -- the k6-stream summary states neither, and "
                       "both are recorded as the sealed lane's because the scorer is the same unmodified "
                       "tools/k6_kld_report.py, invoked as --profile k6-stream. measurement_scope.contexts: "
                       "READ from the verdict receipt's 25-entry per_window array, whose streaming means "
                       "average to exactly the summary's measured_mean_kld. scored_positions: SUPPLIED as "
                       "the panel's own 51,175 (25 x 2047), which the equal-weighted window average is "
                       "consistent with. determinism.identical_across_runs: RECOMPUTED from run_means and "
                       "distinct_tokenwise_kld_sha256; the receipt's bitwise_deterministic flag was checked "
                       "against that, not copied. The verdict's sealed_mean_kld is bit-identical to the "
                       "sealed K6 row in this file, which is what makes the delta a comparison of these two "
                       "rows and not of two unrelated numbers. comparability.bias.floor_measurement_ref: "
                       "SUPPLIED by --floor-measurement once the streaming-lane floor row below existed; "
                       "build_row checked it was measured on this SAME lane before writing the reference "
                       "(exit 7 otherwise)."))

    SK8 = 0.012384191023436866
    out.append(M("measurement--glm53.k8-8bpw-stream.brandonmusic-final25", GLM, A_K8, P_B25, R_B25,
                 PL_STREAM, SK8,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 scored_positions=51175, contexts=25, runs=2, cold=True, run_means=[SK8] * 2,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["763bc4a56a371e11a0f96469885b920deb6acb2c7c576d1268fb0907577f0942"],
                 det_note="2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct "
                          "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                          "nothing; the single tokenwise digest is the determinism evidence.",
                 sources=[src("receipt_file", STREAM_K8_RECEIPT, STREAM_K8_RECEIPT_SHA,
                              "malaiwah.glm53-k8-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_K8_RECEIPT, STREAM_K8_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-k8-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown", "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane, whose offset against the sealed-ep8 lane "
                                 "is known to be non-zero but was NOT measured for this artifact: no "
                                 "sealed-lane row for it exists to bridge against. The lane offset measured "
                                 "for a sibling artifact on this panel is not transferable -- it is a "
                                 "property of the routing, not a constant. This lane's own measurement floor "
                                 "(%s) is %r nats; netting it out gives an estimated quantization-"
                                 "attributable error of %r nats here -- an estimate, not an identity, "
                                 "because KL is not additive, and it is only meaningful because both terms "
                                 "are small and share the same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, SK8 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": True},
                 disclosures=STREAM_DISC(
                     "The lane's offset against the sealed lane is NOT measured for this artifact: no "
                     "sealed-lane row for it exists to bridge against."),
                 notes="This receipt does not name its lane. Its schema string is "
                       "malaiwah.glm53-k8-packed-kld-summary.v1 and its profile reads 'k8-tp4' -- neither "
                       "carries the '-stream-' marker the K6 summary's family name does -- so 'streaming' "
                       "here is OPERATOR-ASSERTED (operator inventory, 2026-08-28) and not read off the "
                       "file. It is recorded as the more caveated of the two possibilities on purpose: if "
                       "the assertion is wrong the row is under-claimed, never over-claimed. Also supplied "
                       "rather than read: metric.direction, estimator.accumulation_dtype, "
                       "measurement_scope.scored_positions and contexts -- this family is a scalar summary "
                       "and states none of them, and unlike the K6 row there is no verdict receipt here to "
                       "read the window count from. No top-1 agreement was produced for this run. "
                       "determinism.identical_across_runs is RECOMPUTED from run_means and "
                       "distinct_tokenwise_kld_sha256. comparability.bias.floor_measurement_ref: SUPPLIED by "
                       "--floor-measurement once the streaming-lane floor row below existed; build_row "
                       "checked it was measured on this SAME lane before writing the reference (exit 7 "
                       "otherwise)."))

    # ------------------------------------------------------------ streaming-lane floor
    # 2026-08-29: the UNQUANTIZED BF16 weights, scored as the streaming lane's own
    # "student" against the reference's stored teacher logits, on the SAME panel and
    # the SAME harness as the two rows above (tools/stream_score.py --source native ->
    # tools/k6_kld_report.py --profile native-bf16-stream). Zero quantization is
    # involved: the divergence here is purely the cost of comparing across capture
    # stacks plus bf16 non-associativity across differing expert-combine orders --
    # this streaming lane's zero-point. See k6/BF16-FLOOR.md for the full analysis.
    #
    # It is NOT the cross-stack floor (measurement--glm53.bf16-replay-floor...,
    # 0.012712 nats, a different pipeline and a different comparability key): that
    # number bounds CROSS-STACK rows on this panel and must never be subtracted from a
    # same-stack streaming row, nor this floor from a cross-stack one. BIAS-006 (new)
    # refuses a floor_measurement_ref that crosses lanes; build_row refuses it at
    # write time (exit 7) before a row like that could even be generated.
    out.append(M(M_BF16_FLOOR, GLM, A_BF16_A6, P_B25, R_B25, PL_STREAM, BF16_FLOOR,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 scored_positions=51175, contexts=25, runs=2, cold=True, run_means=[BF16_FLOOR] * 2,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["c033bcd30f0a67c1be972619f46bf18d598a8f6861df384cdf81add9bdc36546"],
                 det_note="2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct "
                          "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                          "nothing; the single tokenwise digest is the determinism evidence.",
                 sources=[src("receipt_file", STREAM_BF16_RECEIPT, STREAM_BF16_RECEIPT_SHA,
                              "malaiwah.glm53-native-bf16-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_BF16_RECEIPT, STREAM_BF16_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-native-bf16-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown", "floor_measurement_ref": None,
                       "estimated_magnitude": None,
                       "detail": "THIS ROW IS THE FLOOR for the 'streaming' lane: it replays the "
                                 "reference's own unquantized weights through the SAME streaming harness "
                                 "that scored every other row on this pipeline, so its divergence against "
                                 "the stored teacher logits is the lane's zero-point, not a quantization "
                                 "result. It is NOT the cross-stack floor recorded elsewhere in this "
                                 "registry (a different pipeline, a different lane, a different "
                                 "comparability key) and is never interchangeable with it: subtracting one "
                                 "lane's floor from another lane's row is exactly the mistake BIAS-006 "
                                 "exists to catch. The lane's offset against the sealed-ep8 lane is NOT "
                                 "measured for this artifact: no sealed-lane counterpart to this profile "
                                 "exists to bridge against."},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": True},
                 disclosures=[
                     disc("reduced_run_count", "caveat",
                          "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; "
                          "disclosed)", True),
                     disc("non_sealed_lane", "caveat",
                          "Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset "
                          "against the sealed lane is NOT measured for this artifact: no sealed-lane row "
                          "for it exists to bridge against. This row is itself the streaming lane's "
                          "measurement floor -- the zero-point the K6-stream and K8-stream rows in this "
                          "same table subtract to obtain their own quantization-attributable error (see "
                          "their bias blocks).", True),
                     disc("third_party_artifact_self_measured", "info",
                          "Someone else's weights, our measurement.")],
                 notes="CONTROL ROW / STREAMING-LANE MEASUREMENT FLOOR. Not the cross-stack floor "
                       "(measurement--glm53.bf16-replay-floor.brandonmusic-final25, 0.012712 nats, "
                       "pipeline--malaiwah.glm53-crosscheck): a different pipeline, a different lane, a "
                       "different comparability key -- BIAS-002 already keeps the two apart by key, and "
                       "BIAS-006 additionally forbids naming one as the other's floor even inside a shared "
                       "key. Provenance of the fields the summary receipt does not carry: metric.direction "
                       "and estimator.accumulation_dtype are SUPPLIED as reference_to_candidate / float64, "
                       "matching every other row on this pipeline, because the scorer is the same "
                       "unmodified tools/k6_kld_report.py. measurement_scope.scored_positions and contexts "
                       "are SUPPLIED as the panel's own 51,175 positions over 25 contexts (25 x 2047) -- "
                       "like the K8-stream row, no verdict receipt exists for this profile to read the "
                       "window count from. determinism.identical_across_runs is RECOMPUTED from run_means "
                       "and distinct_tokenwise_kld_sha256; the receipt's own bitwise_deterministic flag was "
                       "checked against that, not copied. cold_run_count (2) was checked against "
                       "len(run_means) and len(kld_report_sha256), both 2."))

    DQ = 0.027262784814670614
    out.append(M("measurement--glm53.dione-q4.brandonmusic-final25", GLM, A_DIONE, P_B25, R_B25, PL_DIONE, DQ,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 scored_positions=51175, contexts=25, runs=5, cold=True, run_means=[DQ] * 5,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["f4038d07c329e6e8663e8a09509219b99d34ec6d71a9246eeb65daa37755cb5b"],
                 det_note="Five cold runs, five distinct kld_report_sha256 values, one distinct "
                          "tokenwise_kld_sha256.",
                 sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json",
                                   "d18b37d8ed1ba90ed837d1fb2adca0b90999b2d702613f6730ef87fe23d9f9b7",
                                   "fetched read-only and hashed during seeding; byte-identical to our local copy")],
                 receipt_schema="malaiwah.glm53-dione-q4-packed-kld-summary.v1",
                 cls="advisory",
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None, "passed": True},
                 disclosures=[
                     disc("third_party_artifact_self_measured", "info",
                          "Someone else's weights, our measurement. 0xSero produced the artifact; malaiwah "
                          "produced the number. Credit for the artifact is theirs."),
                     disc("unsealed_source", "caveat",
                          "The Dione checkpoint ships no upstream receipts or sealed reader ABI. The packed "
                          "surface was decoded without seal verification; the immutable revision "
                          "99cccdf0... and the consumed payload sha256s were recorded instead "
                          "(dione_shard_hash_verification: full).", True),
                     disc("artifact_identity_incomplete", "caveat",
                          "The release's own scope manifest was not parsed into this registry, so the "
                          "artifact's per-class recipe is recorded as unknown.", True)],
                 notes="The receipt's cold_run_deviation field reads verbatim '5 cold runs, not 5 (budget; "
                       "disclosed)' -- a self-contradictory template string. cold_run_count is 5 and run_means "
                       "has 5 entries, so five runs is what happened; the string is a receipt-generator defect "
                       "and is recorded here rather than copied into a disclosure."))

    B4 = 0.024554564249958208
    out.append(M("measurement--glm53.brandonmusic-4bpw.brandonmusic-final25", GLM, A_B4, P_B25, R_B25,
                 PL_BM_PACKED, B4, metric_name="mean_of_run_means_tokenwise_kld",
                 scored_positions=51175, contexts=25, runs=5, cold=True, run_means=[B4] * 5,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["2a596810dcdd52fc654eb94fffe1cf394b826ea6b25d8f411049d8354e52f562"],
                 det_note="Five cold runs with five distinct student_backend_identity_sha256 values and one "
                          "distinct tokenwise_kld_sha256.",
                 measured_by="author-reported", measurer=BRANDON("measurer"), cls="advisory",
                 sources=[src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json",
                              "d955bfaedad36ad9841c30808c67fc36b72017f87b720fb460d8e1c13fe75e57")],
                 receipt_schema="quant-pipeline.glm53-packed-student-kld-five-cold-run.v1",
                 gate={"metric": "mean_of_five_run_mean_tokenwise_kld", "threshold_lt": 0.06,
                       "threshold_gt": None, "passed": True},
                 disclosures=[
                     disc("author_reported_only", "caveat",
                          "Measured and published by brandonmusic on his own stack. We have not re-run it. It is "
                          "nonetheless unusually well anchored: his receipt's token_panel_receipt_sha256 "
                          "(0beec577...) and teacher_receipt_sha256 (2ae08117...) are byte-identical to ours, so "
                          "the panel and the teacher are provably the same. Only the reader differs "
                          "(1fb3be87... vs our 1ccce446...).", True)],
                 notes="On the single-window sub-panel the same artifact reads 0.022751 -- a 7% swing from "
                       "0.024555 over the full 25 windows."))

    XSRC = [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/crosscheck-brandonmusic.json",
                "30bcb58625f79f6e37ac19b04d20193f728386adc22d8fac4be490cff340f303",
                "glm53flash-crosscheck/2; hashed during seeding")]
    FLOOR_BIAS = {"kind": "cross_stack_capture_replay", "direction": "upward",
                  "floor_measurement_ref": None, "estimated_magnitude": None,
                  "detail": "THIS ROW IS THE FLOOR. It replays the reference's own BF16 weights through our "
                            "vLLM stack and scores them against brandonmusic's stored fp32 teacher logits. "
                            "0.012712 nats is therefore what two stacks disagree by on identical unquantized "
                            "weights -- not a quantization result. No floor is named because none exists "
                            "below it."}
    out.append(M(M_FLOOR_GLM, GLM, A_BF16_A6, P_B25, R_B25, PL_XCHECK, 0.01271159981725071,
                 stack_relation="cross_stack", scored_positions=51175, contexts=25,
                 top1=0.96652663230896,
                 runs=1, evidence_kind="none", det_note="Single replay pass; no repeatability evidence.",
                 sources=XSRC, receipt_schema="glm53flash-crosscheck/2", cls="advisory", bias=FLOOR_BIAS,
                 disclosures=[
                     disc("cross_stack_capture", "caveat",
                          "Teacher captured on transformers/eager (B200 x4); candidate replayed on our vLLM "
                          "stack. The offset audit confirms position alignment: top-1 agreement is 0.9665 at "
                          "offset 0 and 0.0159 / 0.0162 at offsets -1 / +1.", True),
                     disc("single_run", "caveat", "One pass; determinism not established.", False)],
                 notes="CONTROL ROW / MEASUREMENT FLOOR. Every cross-stack row on this panel contains this term."))

    out.append(M("measurement--glm53.official-fp8.brandonmusic-final25.crossstack", GLM, A_FP8, P_B25, R_B25,
                 PL_XCHECK, 0.020615254540417995,
                 stack_relation="cross_stack", scored_positions=51175, contexts=25,
                 top1=0.9563458824157715, runs=1, evidence_kind="none",
                 det_note="Single replay pass; no repeatability evidence.",
                 sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/fp8-on-brandon-panel.json",
                              "f13df1eb8900164d4786b7433c6326d6d94079df0efe82ddec747b0fd6721cca",
                              "glm53flash-crosscheck/2; fetched read-only and hashed during seeding")],
                 receipt_schema="glm53flash-crosscheck/2", cls="advisory",
                 bias={"kind": "cross_stack_capture_replay", "direction": "upward",
                       "floor_measurement_ref": M_FLOOR_GLM, "estimated_magnitude": 0.01271159981725071,
                       "detail": "Teacher captured on brandonmusic's transformers/eager stack, candidate "
                                 "replayed on our vLLM stack. The same-stack BF16 replay floor on this exact "
                                 "panel is 0.012712, so this number is an UPPER BOUND on the FP8 release's own "
                                 "divergence. The naive difference is 0.007904 -- an estimate, not an identity, "
                                 "because KL is not additive. Do not subtract and publish."},
                 disclosures=[
                     disc("cross_stack_capture", "caveat",
                          "This row cannot be ranked against the K6 / Dione / 4bpw rows on the same panel: those "
                          "are same-stack sealed-capture numbers and this is a cross-stack replay. Their "
                          "comparability keys differ, and the registry's tables are grouped by that key.", True),
                     disc("single_run", "caveat", "One pass; determinism not established.", False)]))

    GS = [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json",
              "c1755f773dcd2119d5dba554d93e4cad36ca269eb2f6ff4914d6032e42bbf29e",
              "glm53flash-fidelity-report/2; fetched read-only and hashed during seeding")]
    out.append(M("measurement--glm53.official-fp8.malaiwah-suite-v5-10m", GLM, A_FP8, P_G10M, R_G10M,
                 PL_GSUITE, 0.028103897727130314,
                 head_policy="shared_reference_head", two_pass=True, vocab_chunk=15488,
                 top1=0.9427366076880801,
                 aux={"context_macro_mean_kld": 0.02810389772713031, "max_kld": 26.968090564012527,
                      "mean_jsd_bits": 0.009201327112046149,
                      "strata": {"code": 0.025320324634148437, "encyclopedic": 0.022284586562593495,
                                 "literary": 0.03232413117304593, "multilingual": 0.02515409825278548}},
                 ci=(0.027205316874101864, 0.028982193226993906), ci_method="context_cluster_bootstrap",
                 clusters=837, samples=10000,
                 scored_positions=10480640, contexts=5120, runs=1, evidence_kind="none",
                 det_note="One pass. Repeatability receipts exist for this suite but were not parsed into this "
                          "registry, so no determinism is claimed.",
                 sources=GS, receipt_schema="glm53flash-fidelity-report/2",
                 disclosures=[disc("single_run", "caveat",
                                   "One pass; determinism not established for this row.", False)]))
    out.append(M("measurement--glm53.official-fp8.malaiwah-suite-v5-10m.scorefrom1024", GLM, A_FP8,
                 P_G10M_W1024, R_G10M_W1024, PL_GSUITE, 0.018794284895435484,
                 head_policy="shared_reference_head", two_pass=True, vocab_chunk=15488,
                 top1=0.9512066226783968,
                 aux={"context_macro_mean_kld": 0.018794284895435484, "max_kld": 7.998210341009944,
                      "mean_jsd_bits": 0.006454983909880134},
                 ci=(0.018073872462596716, 0.019494099868760738), ci_method="context_cluster_bootstrap",
                 clusters=837, samples=10000,
                 scored_positions=5237760, contexts=5120, runs=1, evidence_kind="none",
                 sources=[src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16-scorefrom1024.json",
                              "62b4fe08b72ac2756354b144d92384029cc77afce1e6dade74613b89265f0590",
                              "glm53flash-fidelity-report/3; fetched read-only and hashed during seeding")],
                 receipt_schema="glm53flash-fidelity-report/3",
                 disclosures=[disc("single_run", "caveat", "One pass; determinism not established.", False)],
                 notes="Same tokens, same artifact, same teacher as the 0.028104 row. Dropping the first 1024 "
                       "scored positions of every context moves the number by 33%. That is why the scored-position "
                       "policy is part of panel identity."))
    return out

def build_measurements_runtime(artifacts_map):
    """brandonmusic's single-window runtime series, and the orcarouter author reports."""
    M = lambda *a, **k: measurement(*a, artifacts_map=artifacts_map, **k)
    out = []
    GH = "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results"

    RUNTIME = [
        # (slug, artifact, pipeline, file, sha, value, top1, run_means, distinct_tokenwise, gate, regime_note)
        ("official-fp8.v44", A_FP8_MLAKV, PL_BM_V44, "v44/kld/fp8-five-run-kld-receipt.json",
         "8302e72a523189af1fe65a5e2530d45f9532c0efca8f7b4e43eb5cdc3dfd0e1e",
         0.02462857659644576, 0.9379579872984856,
         [0.024566116963587743, 0.02484955747693601, 0.02488293126922237, 0.024016412383556018,
          0.024827864888926656], 5, True,
         "v43 TP2 DCP1 eager no-MTP FP8 MLA KV, GPUs 2,3"),
        ("nvfp4.v44", A_NVFP4_BM, PL_BM_V44, "v44/kld/nvfp4-five-run-kld-receipt.json",
         "c01cc32afb1802eaba317edc3c1ef90ae649f368307ff5c8957f37bccac78755",
         0.06053485053836315, 0.9154860771861261, [0.06053485053836315] * 5, 1, False,
         "v44 TP2 DCP1 eager no-MTP NVFP4 MLA KV, GPUs 2,3"),
        ("official-fp8.v71", A_FP8_MLAKV, PL_BM_V71, "v71/kld/fp8-dcp2-route128-five-run-kld.json",
         "da072d243fbdb231388bfc23b84bdb0cee2cb26c1885d3ec407c4164525b6b6b",
         0.024581652920382186, 0.9362970200293113,
         [0.024520091705208007, 0.024808400792921563, 0.024394565124699296, 0.024728333543250002,
          0.024456873435832055], 5, True,
         "FP8 MLA NoPE, route128 SMEM, TP2/EP2, DCP2 B12X A2A eager no-MTP"),
        ("nvfp4.v71", A_NVFP4_BM, PL_BM_V71, "v71/kld/nvfp4-dcp2-route128-power2-five-run-kld.json",
         "b52b6d7abbcbf1f0bc81f713e4513bc8a376235e2f44cc7f4ba7d368f62e69ca",
         0.05475737222323711, 0.9149975574010746, [0.054757372223237115] * 5, 1, True,
         "NVFP4 MLA NoPE, power-of-two ceil amax scale v2, route128 SMEM, TP2/EP2, DCP2 B12X A2A eager no-MTP"),
        ("official-fp8.v75", A_FP8_MLAKV, PL_BM_V75, "v75/kld/fp8-five-run-kld.json",
         "409a3487925a98b40d97c174b5e44e2b3526794d14c5e7ef5a35fd5f669b3209",
         0.02461059122118168, 0.9372740595994138,
         [0.024265303032851262, 0.024501753730412402, 0.02497251177396559, 0.024478425471418843,
          0.02483496209726031], 5, True,
         "v75 release image, FP8 MLA NoPE, route128 SMEM/register, TP2/EP2, DCP2 direct symmetric-memory A2A"),
        ("nvfp4.v75", A_NVFP4_BM, PL_BM_V75, "v75/kld/nvfp4-five-run-kld.json",
         "416b44704406dcf67f1f6555c8c5ca391f86b74188b684d1daec2d593dc1e9ee",
         0.05475737222323711, 0.9149975574010746, [0.054757372223237115] * 5, 1, True,
         "v75 release image, NVFP4 MLA NoPE calibrated power-of-two 46-layer scales"),
    ]
    TOKENWISE = {
        "nvfp4.v44": "03dc42308d83b9f64e04c101253a5e316dd21f1e55332a9d63c36fabac7b156e",
        "nvfp4.v71": "39091c2a0a8a78bb95643079e866faf48dd12ba18a5413227c2ba8278017f62c",
        "nvfp4.v75": "39091c2a0a8a78bb95643079e866faf48dd12ba18a5413227c2ba8278017f62c",
    }
    for slug, art, pl, path, sha, value, top1, means, distinct, gate_ok, regime in RUNTIME:
        det_ok = distinct == 1
        ds = [disc("author_reported_only", "caveat",
                   "Measured and published by brandonmusic on his own runtime image. Regime as published: %s. "
                   "We have not re-run it." % regime, True),
              # AUDIT 2026-08-28: these six glm53-r19-runtime-kld-repeated.v1 receipts carry NO
              # compute_dtype field (unlike his results/five-cold-run-kld.json and
              # results/tp2-runtime-window-kld.json, which both declare float64). The rows
              # previously asserted float64 anyway. That is the one estimator field the
              # comparability key is built from, so asserting it on his behalf would let a
              # genuinely float64-attested row merge into this group. Recorded as unknown.
              disc("estimator_unknown", "caveat",
                   "This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, "
                   "so the accumulation precision of brandonmusic's scorer is not established for these "
                   "rows and is recorded as unknown. All six rows in this group share that condition, so "
                   "they remain mutually comparable; a row whose receipt attests float64 would not join "
                   "them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely "
                   "but not evidenced here.", True)]
        if not gate_ok:
            ds.append(disc("quality_gate_failed", "caveat",
                           "The author's own gate (mean tokenwise KLD < 0.06) did NOT pass. Recorded because a "
                           "failing gate is a fact about the artifact, not a reason to hide the row.", False))
        if slug == "nvfp4.v75":
            ds.append(disc("value_identical_to_sibling", "info",
                           "Bit-identical to the v71 NVFP4 row: same value, same top-1, and the SAME tokenwise "
                           "KL digest 39091c2a... The two runtime images produce identical NVFP4 KV numerics on "
                           "this window. This is evidence, not a copy-paste error.", False))
        out.append(M("measurement--glm53.%s.brandonmusic-final-0000" % slug, GLM, art, P_B1W, R_B1W, pl, value,
                     metric_name="mean_of_run_means_tokenwise_kld", top1=top1,
                     accumulation="unknown",
                     scored_positions=2047, contexts=1, runs=5, cold=True, run_means=means,
                     identical=(True if det_ok else False),
                     evidence_kind="tokenwise_kld_sha256" if det_ok else "run_mean_equality_only",
                     evidence_hashes=([TOKENWISE[slug]] if det_ok else None),
                     det_note=("One distinct tokenwise_kld_sha256 across 5 runs." if det_ok else
                               "Five DISTINCT per-run tokenwise_kld_sha256 values: this row is NOT bitwise "
                               "reproducible, and its run means differ accordingly."),
                     measured_by="author-reported", measurer=BRANDON("measurer"), cls="advisory",
                     sources=[src("github_file", "%s/%s" % (GH, path), sha)],
                     receipt_schema="glm53-r19-runtime-kld-repeated.v1",
                     gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                           "passed": gate_ok},
                     disclosures=ds))

    out.append(M("measurement--glm53.nvfp4-dynamic-scale-control.brandonmusic-final-0000", GLM, A_NVFP4_BM,
                 P_B1W, R_B1W, PL_BM_V44, 0.0682295794008272, top1=0.9198827552515877,
                 aux={"median_kld": 0.02432948308191232, "p95_kld": 0.17120012547551325,
                      "p99_kld": 0.7212358598263886, "max_kld": 7.168338286003065},
                 scored_positions=2047, contexts=1, runs=1, evidence_kind="none",
                 det_note="Single run; the receipt records one tokenwise_kld_sha256 (1cb25614...) but a single "
                          "digest is not repeatability evidence.",
                 measured_by="author-reported", measurer=BRANDON("measurer"), cls="advisory",
                 sources=[src("github_file", "%s/v44/kld/nvfp4-dynamic-scale-control-kld-report.json" % GH,
                              "e5365075bccd4e27c9e7f002c23e31cc6f8df196c3c7ccf847faae4f007b22f9")],
                 receipt_schema="glm53-r19-runtime-window-kld.v1",
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": False},
                 disclosures=[disc("author_reported_only", "caveat",
                                   "brandonmusic's dynamic-scale CONTROL for the v44 NVFP4 row: same window, "
                                   "same teacher, dynamic instead of calibrated power-of-two scales.", True),
                              disc("single_run", "caveat", "One run.", False),
                              disc("quality_gate_failed", "caveat",
                                   "mean_kld_gate_passed false at threshold 0.06.", False)]))

    out.append(M("measurement--glm53.brandonmusic-4bpw.tp2-runtime.brandonmusic-final-0000", GLM, A_B4, P_B1W,
                 R_B1W, PL_BM_TP2, 0.022750847877671544, top1=0.9384465070835368,
                 aux={"median_kld": 0.00993991401651846, "p95_kld": 0.07140553100728228,
                      "p99_kld": 0.22317846601861877, "max_kld": 1.018581137984496},
                 scored_positions=2047, contexts=1, runs=1, evidence_kind="none",
                 measured_by="author-reported", measurer=BRANDON("measurer"), cls="advisory",
                 sources=[src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/tp2-runtime-window-kld.json",
                              "a22aec25c33de1d7a2876e475ff1c45fbe500095ceb8d8f23d681c895b33cc65")],
                 receipt_schema="quant-pipeline.glm53-custom-tp2-runtime-window-kld.v1",
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None, "passed": True},
                 disclosures=[disc("author_reported_only", "caveat",
                                   "brandonmusic's custom TP2 runtime on the single qualification window. The "
                                   "receipt notes runtime_raw_decoded_parity_passed false with "
                                   "runtime_rank_output_identical true.", True),
                              disc("single_run", "caveat", "One run.", False)],
                 notes="THE PANEL-SCOPE OBJECT LESSON: the same artifact reads 0.022751 here and 0.024555 over "
                       "the full 25 windows, against the same teacher. A 7% swing from window selection alone."))

    ORCA_ROWS = [("6-bit", 0.0063, 0.9776, 0.0142, 2.7864), ("4-bit", 0.0131, 0.9613, 0.0477, 2.8620),
                 ("3-bit", 0.0421, 0.9206, 0.1332, 3.0566), ("2-bit", 0.1647, 0.8656, 0.6528, 4.3622),
                 ("2bit-lite", 0.3456, 0.7719, 1.2617, 6.7018)]
    for build, kld, top1, p95, ppl in ORCA_ROWS:
        out.append(M("measurement--glm53.orcarouter-mlx-%s.undisclosed" % build.replace("-", ""), GLM,
                     ORCA_IDS[build], P_ORCA, R_ORCA, PL_ORCA, kld,
                     accumulation="unknown", head_policy="unknown",
                     top1=top1, aux={"p95_kld": p95},
                     scored_positions=None, contexts=None, covers_full=False,
                     subset_detail="Unknown: the card publishes no window count or scored-position total.",
                     runs=1, evidence_kind="none",
                     measured_by="author-reported", measurer=ORCA("measurer"), cls="advisory",
                     ci_method="unknown",
                     sources=[src("model_card", "https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX",
                                  None, "read from the KL divergence & Top-1 table on the card")],
                     receipt_schema=None,
                     disclosures=[
                         disc("author_reported_only", "caveat",
                              "Reported by orcarouter on their model card. No receipt, no estimator precision, "
                              "no run count.", True),
                         disc("different_reference_kind", "caveat",
                              "Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 "
                              "teacher. Numbers against a quantized reference are systematically smaller. This "
                              "row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's "
                              "panel -- they are not the same quantity.", True),
                         disc("undisclosed_panel", "caveat",
                              "Evaluation set not disclosed: no token digest, window count or position total.", True),
                         disc("subset_of_panel", "caveat",
                              "Panel coverage unknown, so covers_full_panel is false by default.", True),
                         disc("estimator_unknown", "caveat",
                              "Accumulation precision and head policy are not published.", True)],
                     notes="Perplexity reported alongside on the same card: %s (FP8 reference 2.7797)." % ppl))
    return out

QREC_DIR = "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts"

_QART = {"fp8": Q_FP8, "k5k6": Q_K5K6, "hyd": Q_HYD, "ctx": Q_CTX, "k4": Q_K4, "nvfp4": Q_NVFP4,
         "gt5090": Q_GT5090, "awq": Q_AWQ, "saka": Q_MTP, "turbo5": Q_T5, "turbo6": Q_T6,
         "k6parity": Q_K6P}
_QNAME = {"fp8": "official-fp8", "k5k6": "k5k6", "hyd": "k5k6-hydrated", "ctx": "k5k6-context",
          "k4": "k4", "nvfp4": "unsloth-nvfp4", "gt5090": "gittensor-nvfp4", "awq": "awq-int4",
          "saka": "mtp-nvfp4", "turbo5": "turboderp-5bpw", "turbo6": "turboderp-6bpw",
          "k6parity": "k6-parity"}

_QPANEL = [
    (P_Q10M, R_Q10M, "suite-v5-10m",
     [("fp8", "kld5-10M-fp8.json"), ("k5k6", "kld5-10M-k5k6.json"), ("hyd", "kld5-10M-hyd.json"),
      ("ctx", "kld5-10M-ctx.json"), ("k4", "kld5-10M-k4.json"), ("nvfp4", "kld5-10M-nvfp4.json")]),
    (P_Q1M, R_Q1M, "suite-v5-shard0-1m",
     [("fp8", "kld5-1M-tail-fp8.json"), ("k5k6", "kld5-1M-tail-k5k6.json"),
      ("hyd", "kld5-1M-tail-hyd.json"), ("ctx", "kld5-1M-tail-ctx.json"), ("k4", "kld5-1M-tail-k4.json"),
      ("nvfp4", "kld5-1M-nvfp4.json"), ("gt5090", "kld5-1M-gt5090.json"), ("awq", "kld5-1M-awq.json"),
      ("saka", "kld5-1M-saka.json"), ("turbo5", "kld5-1M-turbo5.json"), ("turbo6", "kld5-1M-turbo6.json"),
      ("k6parity", "kld5-1M-k6parity.json")]),
    (P_Q2M, R_Q2M, "suite-v5-shards01-2m",
     [("fp8", "kld5-2M-tail-fp8.json"), ("k5k6", "kld5-2M-tail-k5k6.json"),
      ("hyd", "kld5-2M-tail-hyd.json"), ("ctx", "kld5-2M-tail-ctx.json"), ("k4", "kld5-2M-tail-k4.json")]),
    (P_Q1M_W256, R_Q1M_W256, "suite-v5-shard0-1m.scorefrom256",
     [(c, "kld5-window-%s-from256.json" % c) for c in ("fp8", "k5k6", "hyd", "ctx", "k4")]),
    (P_Q1M_W1024, R_Q1M_W1024, "suite-v5-shard0-1m.scorefrom1024",
     [(c, "kld5-window-%s-from1024.json" % c) for c in ("fp8", "k5k6", "hyd", "ctx", "k4")]),
]

_QGGUF = [("q8-0", Q_GGUF_Q8, "gguf-report-q8_0.json", "Q8_0"),
          ("q6-k", Q_GGUF_Q6, "gguf-report-q6_k.json", "Q6_K"),
          ("ud-q5-k-xl", Q_GGUF_Q5, "gguf-report-q5_k_xl.json", "UD-Q5_K_XL")]


def _read_receipt(fname):
    import json as _json
    path = os.path.join(QREC_DIR, fname)
    if not os.path.exists(path):
        raise SystemExit("seed: receipt not found, refusing to invent its numbers: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return _json.load(fh), path, L.sha256_file(path)


def build_measurements_qwen(artifacts_map):
    """Every Qwen row is read straight out of its receipt -- no transcribed numbers."""
    M = lambda *a, **k: measurement(*a, artifacts_map=artifacts_map, **k)
    out = []
    for panel, ref, pslug, entries in _QPANEL:
        for cand, fname in entries:
            r, path, fsha = _read_receipt(fname)
            cb = r.get("context_bootstrap") or {}
            cmp_ = r.get("comparator") or {}
            ds = [QWEN_NOREV]
            if cand in ("awq", "saka"):
                ds.append(disc("artifact_identity_incomplete", "caveat",
                               "The upstream repository for this artifact is not recorded by the receipt; only "
                               "a local path. The measurement is ours and real, the artifact identity is not "
                               "established.", True))
            if cand in ("nvfp4", "gt5090", "turbo5", "turbo6"):
                ds.append(disc("third_party_artifact_self_measured", "info",
                               "Someone else's weights, our measurement."))
                ds.append(INCOMPLETE)
            ds.append(disc("single_run", "caveat",
                           "One pass. Repeatability was not established for this row.", False))
            ds.append(disc("shared_reference_head", "info",
                           "One head (25a30fd5...) applied to both sides' hidden states."))
            out.append(M(
                "measurement--qwen38.%s.%s" % (_QNAME[cand], pslug), QWN, _QART[cand], panel, ref,
                PL_QLADDER, r["token_mean_kld"],
                head_policy="shared_reference_head",
                two_pass=cmp_.get("two_pass"), vocab_chunk=cmp_.get("vocab_chunk"),
                top1=r.get("top1_agreement"),
                aux={"context_macro_mean_kld": r.get("context_macro_mean_kld"),
                     "max_kld": r.get("max_kld"), "mean_jsd_bits": r.get("mean_jsd_bits")},
                ci=((cb["ci95_low"], cb["ci95_high"]) if cb.get("ci95_low") is not None else None),
                ci_method=("context_cluster_bootstrap" if cb.get("ci95_low") is not None else "none"),
                clusters=cb.get("clusters"), samples=cb.get("samples"),
                scored_positions=r.get("scored_positions"), contexts=r.get("contexts"),
                runs=1, evidence_kind="none",
                sources=[src("receipt_file", path, fsha, "%s, candidate '%s'" % (r.get("schema"), cand))],
                receipt_schema=r.get("schema"), cls="advisory", disclosures=ds))

    # --- GGUF: cross-engine, with a measured floor on the same panel -----------
    fr, fpath, fsha = _read_receipt("gguf-report-engine-floor.json")
    fcb = fr.get("context_bootstrap") or {}
    fcmp = fr.get("comparator") or {}
    GGUF_DISC = lambda extra: [
        disc("cross_engine_capture", "caveat",
             "The candidate was captured with llama.cpp; the reference and every EXL3/FP8 row on this panel "
             "were captured under vLLM. This number therefore contains a llama.cpp-vs-vLLM term on top of "
             "quantization error, which can only inflate it. That term is measured: 0.000507 nats.", True),
        disc("third_party_artifact_self_measured", "info", "unsloth's weights, our measurement."),
        disc("single_run", "caveat", "One pass.", False),
        disc("shared_reference_head", "info", "One head (25a30fd5...) applied to both sides."),
        QWEN_NOREV] + extra
    out.append(M(M_FLOOR_GGUF, QWN, Q_GGUF_BF16, P_Q1M, R_Q1M, PL_QGGUF, fr["token_mean_kld"],
                 stack_relation="cross_stack", head_policy="shared_reference_head",
                 two_pass=fcmp.get("two_pass"), vocab_chunk=fcmp.get("vocab_chunk"),
                 top1=fr.get("top1_agreement"),
                 aux={"max_kld": fr.get("max_kld"), "mean_jsd_bits": fr.get("mean_jsd_bits")},
                 ci=(fcb["ci95_low"], fcb["ci95_high"]), ci_method="context_cluster_bootstrap",
                 clusters=fcb.get("clusters"), samples=fcb.get("samples"),
                 scored_positions=fr.get("scored_positions"), contexts=fr.get("contexts"),
                 runs=1, evidence_kind="none",
                 sources=[src("receipt_file", fpath, fsha, fr.get("schema")),
                          src("receipt_file", os.path.join(QREC_DIR, "cross-engine-comparator.json"),
                              None, "qwen38-cross-engine-comparator/1")],
                 receipt_schema=fr.get("schema"), cls="advisory",
                 bias={"kind": "cross_stack_capture_replay", "direction": "upward",
                       "floor_measurement_ref": None, "estimated_magnitude": None,
                       "detail": "THIS ROW IS THE FLOOR. Unquantized BF16 weights read by llama.cpp and scored "
                                 "against the vLLM BF16 reference: what two engines disagree by on identical "
                                 "weights. 0.000507 nats, 99.07% top-1. Every GGUF row on this panel contains "
                                 "this term; no EXL3 or FP8 row does."},
                 disclosures=GGUF_DISC([]),
                 notes="CONTROL ROW / CROSS-ENGINE FLOOR."))
    for slug, art, fname, label in _QGGUF:
        r, path, fsha = _read_receipt(fname)
        cb = r.get("context_bootstrap") or {}
        cmp_ = r.get("comparator") or {}
        naive = r["token_mean_kld"] - fr["token_mean_kld"]
        out.append(M("measurement--qwen38.unsloth-gguf-%s.suite-v5-shard0-1m" % slug, QWN, art, P_Q1M, R_Q1M,
                     PL_QGGUF, r["token_mean_kld"],
                     stack_relation="cross_stack", head_policy="shared_reference_head",
                     two_pass=cmp_.get("two_pass"), vocab_chunk=cmp_.get("vocab_chunk"),
                     top1=r.get("top1_agreement"),
                     aux={"p999_kld": r.get("p999_kld"), "max_kld": r.get("max_kld"),
                          "mean_jsd_bits": r.get("mean_jsd_bits")},
                     ci=(cb["ci95_low"], cb["ci95_high"]), ci_method="context_cluster_bootstrap",
                     clusters=cb.get("clusters"), samples=cb.get("samples"),
                     scored_positions=r.get("scored_positions"), contexts=r.get("contexts"),
                     runs=1, evidence_kind="none",
                     sources=[src("receipt_file", path, fsha, "%s, %s" % (r.get("schema"), label))],
                     receipt_schema=r.get("schema"), cls="advisory",
                     bias={"kind": "cross_stack_capture_replay", "direction": "upward",
                           "floor_measurement_ref": M_FLOOR_GGUF,
                           "estimated_magnitude": fr["token_mean_kld"],
                           "detail": "llama.cpp candidate capture vs vLLM reference capture. The cross-engine "
                                     "floor on this exact panel is %.6f nats, so this is an UPPER BOUND. Naive "
                                     "net of floor: %r -- an estimate, not an identity, because KL is not "
                                     "additive." % (fr["token_mean_kld"], naive)},
                     disclosures=GGUF_DISC([INCOMPLETE])))
    return out


# ===========================================================================
# 7. MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(L.repo_root(__file__), "data"))
    ap.add_argument("--check", action="store_true", help="fail if the files would change")
    args = ap.parse_args()

    amap = {a["id"]: a for a in ARTIFACTS}
    measurements = (build_measurements(amap) + build_measurements_runtime(amap)
                    + build_measurements_qwen(amap))

    collections_out = [("models", MODELS), ("artifacts", ARTIFACTS), ("panels", PANELS),
                       ("references", REFERENCES), ("pipelines", PIPELINES),
                       ("measurements", measurements)]

    changed = []
    for name, records in collections_out:
        path = os.path.join(args.out, name + ".jsonl")
        new = "".join(L.canonical_json(r) + "\n" for r in sorted(records, key=lambda x: x["id"]))
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if old != new:
            changed.append(name)
            if not args.check:
                os.makedirs(args.out, exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
        print("%-14s %4d records%s" % (name, len(records), "  [changed]" if old != new else ""))

    if args.check and changed:
        print("\nRESEED DRIFT in: %s" % ", ".join(changed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
