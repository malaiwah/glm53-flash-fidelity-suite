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
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_lib as L  # noqa: E402
import joint_enrich  # noqa: E402
import harness_id as H  # noqa: E402

V = L.SCHEMA_VERSION

# ---------------------------------------------------------------------------
# HARNESS IDENTITY (2026-08-30)
# ---------------------------------------------------------------------------
# Which code produced which number. See tools/harness_id.py for the boundary and
# why it is drawn where it is.
#
# These three are PINNED LITERALS, not readings of the current environment, and
# that is deliberate. `make reseed-check` must give the same answer on python 3.9
# and 3.12 -- it is the integrity gate that proves the rows are a function of
# their receipts, and a gate that fails on a different interpreter is a gate
# nobody runs. A harness block is a historical record of the run that produced
# the numbers, exactly like a receipt digest; it is not a probe of whoever is
# running `make check` today. The code digests, by contrast, ARE read from the
# bytes, because those bytes are in the repository and are the same everywhere.
HARNESS_TOOL_VERSIONS = {"python": "3.9.6", "numpy": "2.0.2", "torch": None}
HARNESS_REPOSITORY = {
    "url": "https://github.com/malaiwah/quant-fidelity-suite",
    # The commit whose tree holds these exact closure bytes. Verified, not
    # assumed: `git log <commit>..HEAD -- <each closure path>` is empty and each
    # file is byte-identical to that tree, which is what `dirty: false` asserts
    # (the closure, not the whole worktree -- data/ necessarily differs, since
    # this reseed is what changes it). If the closure is ever edited without
    # being committed, this must go back to commit_role=parent with dirty=true
    # rather than pointing at a tree that does not contain the code that ran.
    "commit": "c562f769e39a6fd74121771ac248a57d53d23029",
    "commit_role": "exact",
    "dirty": False,
}
HARNESS_UNRECORDED_DETAIL = (
    "metric.value on this row was produced before this registry recorded harness "
    "identity (schema/harness-grandfather.json, frozen 2026-08-30), so there is no "
    "content digest of the code that computed it. The row is grandfathered under "
    "HARN-001 and is NOT retroactively invalidated: its receipt is still hashed and "
    "still verifiable. What is missing is the ability to answer 'was this number "
    "produced before or after defect X was fixed' in one field test. Digests are not "
    "reconstructed from today's checkout, because today's files are not the files that "
    "produced this row and a plausible-looking digest set would be a fabricated "
    "provenance record.")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def disc(code, severity, detail, affects=False, provenance=False, sources=None):
    """One disclosure.

    PROC-01. `provenance=True` marks a disclosure that claims HOW an artifact was
    produced or WHERE it came from -- a mechanism, a lineage, a code path, an
    inherited config -- as opposed to describing the record. That marking obliges
    `sources` (PROV-014), each pinned (PROV-015). Until this existed a metric
    needed a hashed receipt and an assertion needed nothing, so a prose provenance
    claim reached two dataset cards and two registry rows uncited and validated
    clean.
    """
    d = {"code": code, "severity": severity, "detail": detail, "affects_comparability": affects}
    if provenance:
        d["asserts_provenance"] = True
        d["sources"] = list(sources or [])
    elif sources:
        d["sources"] = list(sources)
    return d


NONE_DISC = [disc("no_known_deviations", "info", "No deviation from this registry's default protocol is known for this record.")]


def src(kind, uri, sha256=None, note=None, lines=None):
    d = {"kind": kind, "uri": uri}
    if sha256:
        d["sha256"] = sha256
    if lines:
        # Only ever with a COMMIT-pinned uri: an anchor against a branch is worse
        # than none, because it still reads as precision after the lines have moved.
        d["lines"] = lines
    if note:
        d["note"] = note
    return d


def attr(name, role, handle=None, url=None, maintainer=False):
    d = {"name": name, "role": role, "handle": handle, "url": url, "is_registry_maintainer": maintainer}
    return d


# Receipts this repository holds, at receipts/<handle>/<slug>.json. The digest is of the
# committed file, so a row citing one of these is citing bytes any reader can fetch and hash.
# REG-03. These digests used to be hardcoded literals, so `make reseed-check` proved only
# that data/*.jsonl agreed with the literals in THIS file -- two files in the same commit
# -- while the Makefile and this module's docstring both claim the rows are a function of
# their receipts. A published receipt and the row citing it could disagree silently: the
# digest on the row was never recomputed from the bytes. Now they ARE the bytes.
def _receipt_sha(rel):
    """sha256 of a committed receipt, read at seed time.

    The point of the exercise: a row's `sources[].sha256` must be a digest of the file it
    names, not a constant transcribed beside it."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    return L.sha256_file(path)


STREAM_K6_RECEIPT = "receipts/malaiwah/stream-k6-kld.json"
STREAM_K6_VERDICT = "receipts/malaiwah/stream-k6-verdict.json"
STREAM_TURBO405_RECEIPT = "receipts/malaiwah/stream-turbo-4.05bpw-kld.json"
STREAM_TURBO205_RECEIPT = "receipts/malaiwah/stream-turbo-2.05bpw-kld.json"
STREAM_TR34_RECEIPT = "receipts/malaiwah/stream-tr3-4bpw-kld.json"
STREAM_TR34_RECEIPT_SHA = "1e790a0e2a69b1646ffee3c1c1529596bc2a5ac7d4f314039c6950b6e3ae1e6f"
STREAM_DIONE30_RECEIPT = "receipts/malaiwah/stream-dione-3.0bpw-kld.json"
STREAM_VCRUZK2_RECEIPT = "receipts/malaiwah/stream-vcruz-k2-2bpw-kld.json"
STREAM_DIONE30_RECEIPT_SHA = _receipt_sha(STREAM_DIONE30_RECEIPT)
STREAM_VCRUZK2_RECEIPT_SHA = _receipt_sha(STREAM_VCRUZK2_RECEIPT)
STREAM_K8_RECEIPT = "receipts/malaiwah/stream-k8-kld.json"
STREAM_BF16_RECEIPT = "receipts/malaiwah/stream-bf16-kld.json"
STREAM_K6_RECEIPT_SHA = _receipt_sha(STREAM_K6_RECEIPT)
STREAM_K6_VERDICT_SHA = _receipt_sha(STREAM_K6_VERDICT)
STREAM_TURBO405_RECEIPT_SHA = _receipt_sha(STREAM_TURBO405_RECEIPT)
STREAM_TURBO205_RECEIPT_SHA = _receipt_sha(STREAM_TURBO205_RECEIPT)
STREAM_K8_RECEIPT_SHA = _receipt_sha(STREAM_K8_RECEIPT)
STREAM_BF16_RECEIPT_SHA = _receipt_sha(STREAM_BF16_RECEIPT)
HF_REGISTRY_RAW = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/"

MAL = lambda role: attr("malaiwah", role, handle="malaiwah", url="https://huggingface.co/malaiwah", maintainer=True)
BRANDON = lambda role: attr("brandonmusic", role, handle="brandonmusic", url="https://huggingface.co/brandonmusic")
MIA = lambda role: attr("Mia-AiLab", role, handle="Mia-AiLab", url="https://huggingface.co/Mia-AiLab")
SERO = lambda role: attr("0xSero", role, handle="0xSero", url="https://huggingface.co/0xSero")
TURBODERP = lambda role: attr("turboderp", role, handle="turboderp",
                              url="https://huggingface.co/turboderp")
ORCA = lambda role: attr("orcarouter", role, handle="orcarouter", url="https://huggingface.co/orcarouter")
TURBO = lambda role: attr("turboderp", role, handle="turboderp", url="https://huggingface.co/turboderp")
VCRUZ = lambda role: attr("vcruz305", role, handle="vcruz305",
                          url="https://huggingface.co/vcruz305")
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


def scope_from_evidence(rel_path, kv="not_applicable", mtp=None):
    """Read an artifact's scope from the evidence file the GATE also reads.

    `measure-cloud --scope-json` cross-checks that file against the release's
    own published per-module rates before a run starts, and refuses on any
    disagreement. Restating the same assignments here by hand would create a
    second copy that the gate does not check -- which is exactly how the first
    turbo-2.05bpw receipt came to claim the 4.05bpw branch's rates. One file,
    two readers.
    """
    # SUITE root: this file is registry/tools/, so three dirnames up. The seed
    # is a maintainer tool that runs in the suite checkout; the published
    # dataset repo ships registry/ only and does not run it.
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, rel_path)
    if not os.path.isfile(path):
        raise SystemExit("seed_registry: scope evidence not found: %s\n"
                         "  An artifact's scope is READ from the same file "
                         "`measure-cloud --scope-json` verifies against the "
                         "release; it is not restated here." % path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return scope(doc["policy"],
                 [asg(a["tensor_class"], a["treatment"], a["format"],
                      a.get("bits_per_weight"), a.get("layer_range") or "all",
                      a.get("note"))
                  for a in doc["assignments"]],
                 doc["head_policy"],
                 kv=doc.get("kv_cache_dtype", kv),
                 act=doc.get("activation_quantization"),
                 mtp=doc.get("mtp_included", mtp))


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
A_DIONE30 = "artifact--0xsero.glm-5.3-flash-exl3-3.0bpw"
A_B4 = "artifact--brandonmusic.glm-5.3-flash-tr3-4bpw"
A_TURBO405 = "artifact--turboderp.glm-5.3-flash-exl3-4.05bpw"
A_TURBO205 = "artifact--turboderp.glm-5.3-flash-exl3-2.05bpw"
A_VCRUZK2 = "artifact--vcruz305.glm-5.3-flash-exl3-k2"
VCRUZ_SRC = ("read from the release's OWN config.json quantization_config @ 1718dd40 (bits 2, codebook mcg, head_bits 16, quant_method exl3, scope glm53_routed_experts_only, non_routed_dtype_policy official_source_native, version 0.0.43) and confirmed by a name census of its own 150,226-entry model.safetensors.index.json: 148,608 routed payload tensors (43 layers x 288 experts x 3 projections x 4 objects) and exactly the official 1,618 non-routed names, unfused and under official names, no strays either way")
A_TR3MIRROR = "artifact--mia-ailab.glm-5.3-flash-exl3-tr3-4bpw"
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

# CORRECTED 2026-08-29 (M2). This helper used to say attn.qkv / attn.o /
# mlp.{gate,up,down} were quantized at the nominal rate.  They are not.  Every
# TR3 artifact it describes -- our K6 and K8, and brandonmusic's 4bpw -- is
# ROUTED-EXPERTS-ONLY, and the evidence is the artifacts' own published
# metadata, which nobody had read into the seed:
#
#   * brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8 (and its byte-identical
#     mirror Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw @ 024db9f7) declares, in its
#     own config.json, scope="glm53_routed_experts_only",
#     non_routed_dtype_policy="official_source_native", head_bits=16; and in
#     quantization/recipe.json, tensor_policy="uniform-routed-experts",
#     nonrouted_policy="native".
#   * Its materialization-receipt.json (self-seal 092be1ff..., recomputed)
#     states native_tensor_count 1618, packed_tensor_count 148608,
#     routed_choice_count 37152, nonrouted_native_exact true -- i.e. 43 layers
#     x 288 experts x 3 projections of payloads and NOTHING else, with all
#     1,618 non-routed tensors byte-exact official.  Its
#     model.safetensors.index.json carries {trellis,suh,svh,mcg} objects under
#     `.mlp.experts.<n>.` names only; lm_head.weight is a plain BF16
#     [154880, 4096] tensor.
#   * Our own K6/K8 checkpoints have the identical census, recorded in
#     engines/K8-ANOMALY.json test_6_scope: native_tensor_count 1618,
#     output_tensor_count 150226, routed_choice_count 37152,
#     packed_tensor_count 148608, nonrouted_native_exact true.
#
# The old digest mattered: it made these rows look scope-comparable with the
# stock-exllamav3 rows on the same panel, which really do quantize attention
# (at K6) and the head (at K6).  They are at opposite ends of the scope axis.
# ---------------------------------------------------------------------------
# The Dione (0xSero selective-EXL3) per-tensor-class recipe.
#
# Both rungs of 0xSero's ladder ship the SAME scope at different rates, and both
# STATE it in their own config.json: quantized_scope names the exact module
# range, retained_dtype is `source_precision`, and a name census of the
# 583,090-entry index closes on it exactly (580,608 routed payload tensors, and
# a non-routed set that bijects the official release's 1,618 names).  The Q4
# record used to say `unknown` for embed_tokens / attn.qkv / attn.o / lm_head;
# that was a gap in this registry's reading, not in the release's publishing.
# One table serves both records: one place to be wrong.
# ---------------------------------------------------------------------------
DIONE_CITE = lambda bpw, extra: (
    "read from the release's OWN config.json quantization_config "
    "(quant_method=exl3_selective_tp4, format=glm53-selective-exl3-tp4-v1, "
    "trellis_k=%g, bits_per_weight=%g, mcg=true, retained_dtype=source_precision, "
    "quantized_scope=model.language_model.layers.3..44.mlp.experts.0..287."
    "{gate_proj,up_proj,down_proj}.weight)%s, and confirmed by a name census of "
    "its 583,090-entry index: 580,608 routed payload tensors (42 layers x 288 "
    "experts x 3 projections x 4 TP ranks x 4 objects) and exactly the official "
    "1,618 non-routed names, no strays either way "
    "(k6/tools/dione_surface.py census_weight_map)." % (bpw, bpw, extra))

# `uniform`, not `mixed`: SCOPE-003 reads the word as "do the QUANTIZED classes
# share one (format, bits)", and here exactly one class is quantized at all.
DIONE_SCOPE = lambda bpw, extra="": scope("uniform", [
    asg("embed_tokens", "native", "bf16", 16,
        note="retained at source precision in the release's own retained/ shards. "
             + DIONE_CITE(bpw, extra)),
    asg("attn.qkv", "native", "bf16", 16,
        note="NOT quantized: the quantized scope is the routed experts only. "
             + DIONE_CITE(bpw, extra)),
    asg("attn.o", "native", "bf16", 16,
        note="NOT quantized: routed-experts-only scope. " + DIONE_CITE(bpw, extra)),
    asg("attn.other", "native", "mixed", None,
        note="indexers, mHC and the attention norms ship as the official tensors "
             "at their source dtypes (A_log, dt_bias and e_score_correction_bias "
             "are fp32 there and fp32 here). " + DIONE_CITE(bpw, extra)),
    asg("mlp.gate", "native", "bf16", 16, layer_range="0-2",
        note="only the three DENSE layers have an mlp.*; NOT quantized. "
             + DIONE_CITE(bpw, extra)),
    asg("mlp.up", "native", "bf16", 16, layer_range="0-2",
        note="NOT quantized. " + DIONE_CITE(bpw, extra)),
    asg("mlp.down", "native", "bf16", 16, layer_range="0-2",
        note="NOT quantized. " + DIONE_CITE(bpw, extra)),
    asg("moe.router", "native", "fp32", 32,
        note="the routing gate and e_score_correction_bias are retained natively. "
             + DIONE_CITE(bpw, extra)),
    asg("moe.shared_expert", "native", "bf16", 16,
        note="the shared expert is not routed and is NOT quantized. "
             + DIONE_CITE(bpw, extra)),
    asg("moe.experts", "quantized", "exl3-mcg", bpw, layer_range="3-44",
        note="36,288 modules = 42 layers x 288 experts x 3 projections, each "
             "stored as 4 TP-rank slices (<module>.rank{R}.{trellis,suh,svh,mcg}) "
             "at K%g. The ONLY quantized class. " % bpw + DIONE_CITE(bpw, extra)),
    asg("mtp", "native", "bf16", 16, layer_range="45",
        note="layer 45's routed experts are RETAINED at source precision in this "
             "family (they are QUANTIZED in the TR3 releases -- the two are not "
             "interchangeable). Present in the artifact, outside the measured "
             "function: standard-logits scoring never executes the MTP layer. "
             + DIONE_CITE(bpw, extra)),
    asg("norm", "native", "bf16", 16, note=DIONE_CITE(bpw, extra)),
    asg("lm_head", "native", "bf16", 16,
        note="the head is RETAINED at source precision -- unlike stock exllamav3, "
             "which quantizes it (head_bits 6-8). " + DIONE_CITE(bpw, extra)),
    asg("other", "native", "bf16", 16,
        note="the vision tower is retained natively and is never executed by "
             "text-only scoring. " + DIONE_CITE(bpw, extra)),
], "native", kv="bf16", mtp=True)

DIONE_SCOPE_CORRECTED = disc(
    "scope_record_corrected", "info",
    "Superseded record: this artifact's scope previously read embed_tokens, "
    "attn.qkv, attn.o and lm_head as `unknown`, with the note that the release "
    "'declares a scope policy that was not parsed into this registry'. It is "
    "parsed now. The release's own config.json states quantized_scope = "
    "model.language_model.layers.3..44.mlp.experts.0..287."
    "{gate_proj,up_proj,down_proj}.weight and retained_dtype = source_precision, "
    "and a name census of its published index closes on exactly that: 580,608 "
    "routed payload tensors and a non-routed set that bijects the official BF16 "
    "release's 1,618 names. So the head is native BF16, not unknown, and the "
    "artifact is routed-experts-only. Corrected 2026-08-30 from the artifact's "
    "own published metadata. scope_digest changed accordingly; the measured "
    "VALUE is unaffected -- the measurement always ran the artifact as published.")

DIONE_TP_SLICED = disc(
    "tp_sliced_artifact", "info",
    "Shipped pre-sliced for TP4: each routed matrix is stored as four "
    "independently quantized EXL3 payloads under TENSOR names "
    "<module>.rank0..rank3.{trellis,suh,svh,mcg}, and the full HF matrix is the "
    "rank-ordered concatenation (dim 0 for gate/up, dim 1 for down). That is "
    "artifact identity, not a runtime option. NOTE: the layers/layer-NN-part-K "
    "FILES are a parallel-encoding artifact, not the TP slices -- in the 3.0bpw "
    "release layers 3-5 have one part holding all 288 experts while layers 6-44 "
    "split even/odd experts across two, and every one of them carries all four "
    "ranks.")


EXL3_SCOPE_UNIFORM = lambda bpw: scope("uniform", [
    asg("embed_tokens", "native", "bf16", 16,
        note="routed-experts-only scope: not quantized"),
    asg("attn.qkv", "native", "bf16", 16,
        note="NOT quantized. KDA layers ship the official split q/k/v_proj, MLA "
             "layers the official q_a/q_b/kv_a_with_mqa/wq_b, all byte-exact "
             "official (nonrouted_native_exact)"),
    asg("attn.o", "native", "bf16", 16, note="NOT quantized"),
    asg("attn.other", "native", "mixed", None,
        note="b_proj / f_a_proj / f_b_proj / g_a_proj / g_b_proj / conv1d / the "
             "DSA indexer and the attention norms ship as the official tensors; "
             "A_log, dt_bias and e_score_correction_bias are fp32 there and here"),
    asg("mlp.gate", "native", "bf16", 16, layer_range="0-2",
        note="only the three DENSE layers have an mlp.*; NOT quantized"),
    asg("mlp.up", "native", "bf16", 16, layer_range="0-2", note="NOT quantized"),
    asg("mlp.down", "native", "bf16", 16, layer_range="0-2", note="NOT quantized"),
    asg("moe.router", "native", "fp32", 32,
        note="the routing gate and e_score_correction_bias are native"),
    asg("moe.shared_expert", "native", "bf16", 16,
        note="the shared expert is not routed and is NOT quantized"),
    asg("moe.experts", "quantized", "exl3-mcg", bpw, layer_range="3-44",
        note="36,288 modules = 42 executed layers x 288 experts x 3 projections. "
             "The ONLY quantized class."),
    asg("mtp", "quantized", "exl3-mcg", bpw, layer_range="45",
        note="layer 45's routed experts are quantized and present in the release "
             "(routed_choice_count 37152 = 43 layers), and are NOT executed by "
             "standard-logits scoring"),
    asg("norm", "native", "bf16", 16),
    asg("lm_head", "native", "bf16", 16,
        note="head_bits 16: TR3 keeps the head native BF16, unlike stock "
             "exllamav3 which quantizes it"),
    asg("other", "native", "bf16", 16,
        note="the vision tower ships the official fused attn.qkv and is never "
             "executed by text-only scoring"),
], "native", kv="bf16", mtp=True)

SCOPE_CORRECTED = disc(
    "scope_record_corrected", "info",
    "Superseded record: this artifact's scope previously read attn.qkv / attn.o / "
    "mlp.{gate,up,down} as quantized:exl3-mcg at the nominal rate. That was wrong. "
    "The release is routed-experts-only: its own config.json declares "
    "scope=glm53_routed_experts_only, non_routed_dtype_policy=official_source_native "
    "and head_bits=16, and its materialization receipt states native_tensor_count "
    "1618 / packed_tensor_count 148608 / routed_choice_count 37152 / "
    "nonrouted_native_exact true -- expert payloads and nothing else. Corrected "
    "2026-08-29 from the artifacts' own published metadata (and, for the K6/K8 "
    "checkpoints, k6/K8-ANOMALY.json test_6_scope, which records the identical "
    "census). scope_digest changed accordingly; the measured VALUES are unaffected, "
    "since the measurement always ran the artifact as published.")

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
             [SCOPE_CORRECTED,
              REV_UNPINNED("Identity rests on student_checkpoint_identity_sha256 a8668be3... and the "
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
             [SCOPE_CORRECTED,
              REV_UNPINNED("The repository returns HTTP 401 unauthenticated, so no commit sha could "
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
             codec("exl3-mcg", 4.0, None, tool="exllamav3", version=None),
             DIONE_SCOPE(4.0),
             SERO("quantizer"),
             [src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json",
                  "d18b37d8ed1ba90ed837d1fb2adca0b90999b2d702613f6730ef87fe23d9f9b7", "malaiwah.glm53-dione-q4-packed-kld-summary.v1: dione_repo, dione_revision 99cccdf0..., "
                        "dione_shard_hash_verification=full"),
              src("hf_file", "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json"),
              src("url", "https://huggingface.co/api/models/0xSero/GLM-5.3-Flash-EXL3-Q4?blobs=true",
                  None, "498 files, 217 safetensors; all-files sum 187,607,584,245; safetensors sum 187,453,172,472")],
             [DIONE_SCOPE_CORRECTED,
              disc("unsealed_source", "caveat",
                   "The Dione checkpoint ships no upstream receipts, reconstruction closures or sealed reader ABI. "
                   "The packed surface was decoded WITHOUT seal verification; the immutable repo revision and the "
                   "consumed payload sha256s were recorded instead (dione_shard_hash_verification: full).", True),
              DIONE_TP_SLICED],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 217,
                            "tensor_parallel": {"pre_sliced": True, "world_size": 4}},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public", "uri": "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4"},
             cross_refs=lair(model_id="glm-5.3-flash", url="https://huggingface.co/datasets/0xSero/local-ai-registry",
                            confidence="probable"),
             seal={"sealed": False, "note": "unsealed source; see the unsealed_source disclosure"}),
    artifact(A_DIONE30, GLM,
             "0xSero GLM-5.3-Flash EXL3 3.0bpw (Dione, K3, TP4-sliced, native BF16 head)",
             "quant",
             hf("0xSero/GLM-5.3-Flash-EXL3-3.0bpw",
                "8b099bf276507a17faea920deff3f62d5597fb52", "hf_api"),
             "exl3", "3.0 bpw", 149556991042,
             codec("exl3-mcg", 3.0, None, tool="exllamav3",
                   version="git 5f3c537ca9d89893d771256f5c43c93656553fbb",
                   # The release publishes its calibration shape AND the digests
                   # of the six corpus files it drew from. Whether any of them
                   # overlaps this panel cannot be answered from either side:
                   # the panel's own per-document provenance is published as a
                   # digest only. `null` is that answer, not a shrug.
                   calibration={"used": True,
                                "corpus": "0xSero release-calibration-manifest: 600 rows x "
                                          "2048 columns over c4/code/multilingual/technical/"
                                          "tiny/wiki (six utf8 corpus files, each sha256'd in "
                                          "evidence/release-calibration-manifest.json) plus 92 "
                                          "random-token rows; routing_policy natural_top8, "
                                          "route floor 1024, zero_hit_experts 0",
                                "tokens": 1228800,
                                "overlaps_any_panel": None,
                                "overlapping_panel_refs": []}),
             DIONE_SCOPE(3.0, " and its retained_scope string 'attention, indexers, mHC, "
                              "routers, shared experts, dense layers 0-2, embeddings, "
                              "lm_head, norms, vision, MTP'"),
             SERO("quantizer"),
             [src("url",
                  "https://huggingface.co/api/models/0xSero/GLM-5.3-Flash-EXL3-3.0bpw"
                  "?blobs=true&revision=8b099bf276507a17faea920deff3f62d5597fb52",
                  None,
                  "335 files, 130 safetensors; all-files sum 149,556,991,042; safetensors "
                  "sum 149,402,871,912; the index's own metadata.total_size 149,325,518,712"),
              src("hf_file",
                  "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw/resolve/"
                  "8b099bf276507a17faea920deff3f62d5597fb52/config.json",
                  "0f97529bf936f823b84dc77ac0e09190e9439094e29dc8895547df8e17e4a24e",
                  "the release's own quantization_config: bits_per_weight 3.0, trellis_k 3, "
                  "mcg true, tensor_parallel_size 4, retained_dtype source_precision, "
                  "quantized_scope layers 3..44 x experts 0..287 x {gate,up,down}_proj, "
                  "source_revision a6c167b6. Every entry in scope.assignments is read from "
                  "it."),
              src("hf_file",
                  "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw/resolve/"
                  "8b099bf276507a17faea920deff3f62d5597fb52/EXL3_MANIFEST.json",
                  "05e0ff9cc6a3f87fbd8e27b46bb679e114579dfea3bc4afcc2d724b58be3d1ee",
                  "the release manifest (schema_version 1): a sha256 and byte count for "
                  "each of the 130 shards, target_bpw 3.0, indexed_tensor_count 583,090, "
                  "quantized_tensor_count 580,608, retained tensor_count 2,482, source "
                  "zai-org/GLM-5.3-Flash-BF16 @ a6c167b6. All 130 digests were recomputed "
                  "on the measurement instance before any payload was decoded."),
              src("hf_file",
                  "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw/resolve/"
                  "8b099bf276507a17faea920deff3f62d5597fb52/RELEASE_STATUS.json",
                  None,
                  "the producer's OWN gate verdicts: overall_status "
                  "weights_public_validation_incomplete; quality FAIL against their own "
                  "threshold (their forward_kl 0.15251, perplexity_delta_fraction 0.09297, "
                  "top1_agreement 0.87285 over 65,504 held-out positions of THEIR panel); "
                  "serve and mtp pending; structure and public_ungated pass."),
              src("hf_file",
                  "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw/resolve/"
                  "8b099bf276507a17faea920deff3f62d5597fb52/evidence/"
                  "release-calibration-manifest.json",
                  None,
                  "the calibration recorded in codec.calibration: 600 rows x 2048 columns, "
                  "1,228,800 tokens, exllamav3_revision 5f3c537c, six sha256'd corpus "
                  "files, routing_policy natural_top8.")],
             [disc("unsealed_source", "caveat",
                   "The Dione checkpoint ships no upstream receipts, reconstruction "
                   "closures or sealed reader ABI. The packed surface was decoded WITHOUT "
                   "seal verification. What it DOES publish is a per-shard sha256 manifest, "
                   "and all 130 shard digests were recomputed on the measurement instance "
                   "before anything was decoded (dione_shard_hash_verification: full); that "
                   "plus the immutable revision and the consumed-payload sha256 census are "
                   "the provenance anchors.", True),
              DIONE_TP_SLICED,
              disc("producer_quality_gate_failed", "info",
                   "The producer's own RELEASE_STATUS.json marks this release "
                   "weights_public_validation_incomplete with quality: FAIL -- their own "
                   "held-out forward KL is 0.15251 nats at 0.87285 top-1 over 65,504 "
                   "positions, and they publish it rather than hiding it. That is THEIR "
                   "panel and THEIR estimator, so the number is not comparable with this "
                   "registry's; it is recorded because a reader deciding whether to run "
                   "these weights should see the producer's own verdict next to ours.")],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 130,
                            "index_sha256":
                                "0fd35de9b0d5fc9428a45d3b311dc757ea891e4cec7788050b75089593ad3215",
                            "config_sha256":
                                "0f97529bf936f823b84dc77ac0e09190e9439094e29dc8895547df8e17e4a24e",
                            "tensor_parallel": {"pre_sliced": True, "world_size": 4}},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public",
                           "uri": "https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw"},
             cross_refs=lair(model_id="glm-5.3-flash",
                             url="https://huggingface.co/datasets/0xSero/local-ai-registry",
                             confidence="probable"),
             seal={"sealed": False,
                   "note": "unsealed source; the release publishes no seal. Its 130 shard "
                           "sha256s (EXL3_MANIFEST.json, schema_version 1) were recomputed "
                           "in full on the measurement instance."}),
    artifact(A_TURBO405, GLM,
             "turboderp GLM-5.3-Flash EXL3 4.05bpw (stock exllamav3, mul1, quantized head)",
             "quant",
             hf("turboderp/GLM-5.3-Flash-exl3",
                "2a30229e67012798ba9f0cd832bb78abf4c363d5", "hf_api",
                path="branch 4.05bpw"),
             "exl3", "4.05 bpw", 165151543361,
             codec("exl3-mul1", 4.05, None, tool="exllamav3", version="1.4.4",
                   # the release states its own calibration shape: 250 rows x 2048 cols
                   calibration={"used": True, "corpus": None, "tokens": 250 * 2048,
                                "overlaps_any_panel": None, "overlapping_panel_refs": []}),
             # Every rate below was READ, not assumed -- see NOTE.
             scope("mixed", [
                 asg("embed_tokens", "native", "bf16", 16, note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("attn.qkv", "quantized", "exl3-mul1", 6,
                     note="KDA layers ship a FUSED qkv_proj (rows q|k|v); MLA layers ship "
                          "q_a/q_b/kv_a_with_mqa/wq_b. All K6. " + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("attn.o", "quantized", "exl3-mul1", 6, note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("attn.other", "native", "fp16", 16,
                     note="b_proj, f_a/f_b_proj, g_a/g_b_proj, weights_proj, wk, conv1d and the "
                          "attention norms ship as plain fp16/bf16 tensors. " + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("mlp.gate", "quantized", "exl3-mul1", 5, layer_range="0-2",
                     note="only the three DENSE layers have an mlp.*; layers 3-44 are MoE. "
                          + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("mlp.up", "quantized", "exl3-mul1", 5, layer_range="0-2", note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("mlp.down", "quantized", "exl3-mul1", 5, layer_range="0-2", note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("moe.router", "native", "fp32", 32, layer_range="3-44",
                     note="the routing gate ships fp16/fp32, unquantized. " + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("moe.experts", "quantized", "exl3-mul1", 4, layer_range="3-44",
                     note="12,096 modules = 42 layers x 288 experts x 3 projections, all K4. "
                          + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("moe.shared_expert", "quantized", "exl3-mul1", 6, layer_range="3-44",
                     note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("norm", "native", "bf16", 16, note="read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("mtp", "quantized", "exl3-mul1", 4, layer_range="45",
                     note="layer 45 ships in a separate mtp.safetensors and is NOT executed by "
                          "standard-logits scoring; present in the artifact, outside the measured "
                          "function. " + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("lm_head", "quantized", "exl3-mul1", 6,
                     note="stock exllamav3 quantizes the head (head_bits 6), unlike TR3 which "
                          "keeps it native BF16. The FP8 parent had lm_head in "
                          "modules_to_not_convert, so this 6-bit head was quantized from BF16 "
                          "weights. " + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
                 asg("other", "quantized", "exl3-mul1", 6,
                     note="vision tower (model.visual.*): attn q/k/v/o and MLP are EXL3 at K6, "
                          "patch embed and norms native. Never executed by text-only scoring. "
                          + "read from the release's OWN quantization_config.json @ 2a30229e (47.9 MB, one tensor_storage entry per module): bits = trellis.shape[-1]//16, verified identical within each class"),
             ], "quantized", kv="not_applicable", mtp=True),
             TURBODERP("quantizer"),
             [src("url", "https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3?blobs=true",
                  None, "31 files at revision 2a30229e; 19 safetensors shards + mtp.safetensors; "
                        "all-files sum 165,151,543,361"),
              src("hf_file",
                  "https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/resolve/"
                  "2a30229e67012798ba9f0cd832bb78abf4c363d5/quantization_config.json",
                  None,
                  "47.9 MB, one tensor_storage entry per module: every per-tensor-class bit rate "
                  "in scope.assignments was READ from it. config.json sha256 df80c17a68120aeae4c"
                  "1eca8a9aa67866603484f0487cd5458c08dfd3c45156d and "
                  "model.safetensors.index.json sha256 ee1f2dbea800dea0b4225c38193f7ef41180ed42d"
                  "d51f80a0cee58daf44cf606, both verified against the fetched copy on the "
                  "measurement instance"),
              src("hf_file",
                  "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/"
                  "resolve/main/reports/turbo-4.05bpw-packed-kld.json",
                  "68ef836737f9eeb59f62da5107246249fcd30c462ccf25493bab75f917df0706",
                  "malaiwah.glm53-turbo-4.05bpw-packed-kld-summary.v1")],
             [disc("unsealed_source", "caveat",
                   "Stock exllamav3 releases ship no upstream receipts, reconstruction closures "
                   "or sealed reader ABI. The packed surface was decoded WITHOUT seal "
                   "verification; the immutable repo revision, the artifact's own config/index "
                   "sha256 and the consumed payload digests were recorded instead.", True),
              disc("quantized_from_quantized_parent", "caveat",
                   "The release's own quantization_config declares "
                   "original_quantization_config.fmt = e4m3: this artifact was quantized from the "
                   "FP8 release, not from BF16. Its divergence against a BF16 reference therefore "
                   "includes the FP8 parent's, and it is not lineage-comparable with a 4-bpw "
                   "artifact quantized directly from BF16.", True),
              disc("quantized_head", "caveat",
                   "head_bits 6: stock exllamav3 quantizes lm_head, unlike the TR3 artifacts in "
                   "this registry which keep it native BF16. The head is APPLIED natively from "
                   "the artifact's own weights (no shared or replayed head), so measurements of "
                   "it carry head_policy native_head; the quantization is artifact identity, "
                   "recorded here.", True),
              disc("redundant_tensor_representations", "info",
                   "Each of the 24 vision blocks ships BOTH the EXL3-quantized split "
                   "attn.{q,k,v}_proj AND the untouched original fused attn.qkv.{weight,bias} -- "
                   "48 names with two representations. The fused copy was verified to be bitwise "
                   "the official BF16 weight cast to fp16. Any measurement must say which it "
                   "used; ours uses the quantized split. The vision tower is not executed by "
                   "text-only scoring, so it does not affect the published number.")],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 19,
                            "tensor_parallel": {"pre_sliced": False, "world_size": None}},
             derived_from_artifact_ref=A_FP8,
             availability={"status": "public",
                           "uri": "https://huggingface.co/turboderp/GLM-5.3-Flash-exl3"},
             cross_refs=lair(),
             seal={"sealed": False, "note": "unsealed source; see the unsealed_source disclosure"}),
    # turboderp's LOW-BIT rung, and the only artifact in this registry whose
    # number has been reproduced on hardware we do not own. Its scope is READ
    # from the evidence file `measure-cloud --scope-json` verifies against the
    # release, rather than restated here: the FIRST 2.05bpw receipt carried the
    # 4.05bpw branch's rates -- experts 4 where this release publishes 2, head 6
    # where it declares 5 -- because a sibling branch's scope file is a valid
    # file that names the same classes and is wrong in every rate.
    artifact(A_TURBO205, GLM,
             "turboderp GLM-5.3-Flash EXL3 2.05bpw (stock exllamav3, mul1, quantized head at 5 bits)",
             "quant",
             hf("turboderp/GLM-5.3-Flash-exl3",
                "51058cd551c7e570d87bd32a4adee720edce2349", "hf_api",
                path="branch 2.05bpw"),
             "exl3", "2.05 bpw", 85233484348,
             codec("exl3-mul1", 2.05, None, tool="exllamav3", version="1.4.4",
                   calibration={"used": True, "corpus": None, "tokens": 250 * 2048,
                                "overlaps_any_panel": None, "overlapping_panel_refs": []}),
             # A path READ FROM DISK, so it follows the tree: k6/ was renamed
             # engines/ on 2026-08-31. Every other k6/... string in this file
             # is a published FIELD -- a scope note, a pipeline entrypoint, a
             # harness code_digests path -- and those keep the spelling the
             # tree had when the number ran. `make reseed-check` is what tells
             # the two apart: it read this one and could not find it.
             scope_from_evidence("engines/tools/exl3hf-evidence/scope-turbo-2.05bpw.json"),
             TURBODERP("quantizer"),
             [src("url", "https://huggingface.co/api/models/turboderp/GLM-5.3-Flash-exl3?blobs=true",
                  None, "24 files at revision 51058cd5; all-files sum 85,233,484,348"),
              src("hf_file",
                  "https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/resolve/"
                  "51058cd551c7e570d87bd32a4adee720edce2349/quantization_config.json",
                  "22a0eb34458b7a5951a5f20aa30290b6cd330085db3585dd83993bf7bb83ae2c",
                  "47,905,719 bytes, one tensor_storage entry per module stating "
                  "bits_per_weight; the header declares bits 2.05, head_bits 5, "
                  "vision_bits 5, mtp_bits 2. Every rate in scope was read from it "
                  "and is re-verified against it at plan time.")],
             [disc("unsealed_source", "caveat",
                   "Stock exllamav3 releases ship no upstream receipts, reconstruction closures "
                   "or sealed reader ABI. The packed surface was decoded WITHOUT seal "
                   "verification; the immutable repo revision and the artifact's own "
                   "quantization_config sha256 were recorded instead.", True),
              disc("quantized_from_quantized_parent", "caveat",
                   "The release's own quantization_config declares "
                   "original_quantization_config.fmt = e4m3: this artifact was quantized from "
                   "the FP8 release, not from BF16. Its divergence against a BF16 reference "
                   "therefore includes the FP8 parent's.", True),
              disc("quantized_head", "caveat",
                   "head_bits 5 -- LOWER than the 6 that this producer's 4.05bpw and 3.05bpw "
                   "branches declare. The head is APPLIED natively from the artifact's own "
                   "weights, so measurements carry head_policy native_head; the quantization is "
                   "artifact identity, recorded here.", True)],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 19,
                            "tensor_parallel": {"pre_sliced": False, "world_size": None}},
             derived_from_artifact_ref=A_FP8,
             availability={"status": "public",
                           "uri": "https://huggingface.co/turboderp/GLM-5.3-Flash-exl3"},
             cross_refs=lair(),
             seal={"sealed": False, "note": "unsealed source; see the unsealed_source disclosure"}),
    # M4. The fourth rung of the measured GLM-5.3-Flash ladder and the lowest rate
    # on this panel. Storage is turboderp's (stock-exllamav3 HF shards, canonical
    # index, per-module {trellis,suh,svh,mcg}); everything the storage does not
    # decide is 0xSero's shape instead -- MCG codebook, routed experts only,
    # non-routed tensors retained at the official source precision, native BF16
    # head. Reading either half off the other would put a false claim on this row,
    # so every assignment below cites the release's own config and its own index.
    artifact(A_VCRUZK2, GLM,
             "vcruz305 GLM-5.3-Flash EXL3 K2 (stock-exllamav3 HF layout, mcg, routed "
             "experts only, native BF16 head)",
             "quant",
             hf("vcruz305/GLM-5.3-Flash-EXL3-K2",
                "1718dd403534fd369e82d676e55a50dc19630ffc", "hf_api"),
             "exl3", "2.0 bpw", 97767357064,
             codec("exl3-mcg", 2.0, None, tool="exllamav3",
                   version="quantization_config.version 0.0.43 (the release names no "
                           "exllamav3 git revision; its encoder is the producer's own "
                           "glm53_exl3_encode_experts.py writing the stock EXL3/MCG "
                           "payload layout)",
                   # The release publishes no calibration manifest at all. `None` is
                   # "not published", which is not the same as "not used": an EXL3
                   # trellis fit is calibrated by construction. Recording False here
                   # would be an assertion nobody made.
                   calibration={"used": None,
                                "corpus": "not published: the release ships no "
                                          "calibration manifest, corpus digest or "
                                          "token count. An EXL3 fit is calibrated by "
                                          "construction, so this is 'undisclosed', "
                                          "not 'none'.",
                                "tokens": None,
                                "overlaps_any_panel": None,
                                "overlapping_panel_refs": []}),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16", 16,
                     note="NOT quantized: the quantized scope is the routed experts only. "
                          + VCRUZ_SRC),
                 asg("attn.qkv", "native", "bf16", 16,
                     note="NOT quantized, and stored UNFUSED under the official names -- "
                          "q_proj/k_proj/v_proj on the 34 KDA layers and "
                          "q_a/q_b/kv_a_proj_with_mqa/kv_b_proj on the 12 MLA layers -- "
                          "unlike stock exllamav3 releases, which ship a fused "
                          "self_attn.qkv_proj. " + VCRUZ_SRC),
                 asg("attn.o", "native", "bf16", 16, note="NOT quantized. " + VCRUZ_SRC),
                 asg("attn.other", "native", "mixed", None,
                     note="q/k/v_conv1d, b_proj, f_a/f_b_proj, g_a/g_b_proj, the DSA "
                          "indexers and the attention norms ship as the official tensors "
                          "at their source dtypes (A_log and dt_bias are fp32 there and "
                          "fp32 here). " + VCRUZ_SRC),
                 asg("mlp.gate", "native", "bf16", 16, layer_range="0-2",
                     note="only the three DENSE layers have an mlp.{gate,up,down}_proj; "
                          "NOT quantized. " + VCRUZ_SRC),
                 asg("mlp.up", "native", "bf16", 16, layer_range="0-2",
                     note="NOT quantized. " + VCRUZ_SRC),
                 asg("mlp.down", "native", "bf16", 16, layer_range="0-2",
                     note="NOT quantized. " + VCRUZ_SRC),
                 asg("moe.router", "native", "fp32", 32, layer_range="3-45",
                     note="the routing gate and e_score_correction_bias are retained "
                          "natively. " + VCRUZ_SRC),
                 asg("moe.shared_expert", "native", "bf16", 16, layer_range="3-45",
                     note="the shared expert is not routed and is NOT quantized. "
                          + VCRUZ_SRC),
                 asg("moe.experts", "quantized", "exl3-mcg", 2.0, layer_range="3-44",
                     note="36,288 modules = 42 layers x 288 experts x 3 projections, each "
                          "stored as one K2 EXL3/MCG payload {trellis[.,.,32] int16, suh "
                          "fp16, svh fp16, mcg int32 marker} under the official module "
                          "name. The dominant quantized class. " + VCRUZ_SRC),
                 asg("mtp", "quantized", "exl3-mcg", 2.0, layer_range="45",
                     note="layer 45's 864 routed experts ARE quantized here, at the same "
                          "K2 rate, and they live in the MAIN index rather than a separate "
                          "mtp.safetensors. That differs from the Dione family (which "
                          "retains them at source precision) and from turboderp's stock "
                          "releases (which quantize them into a side file). Present in the "
                          "artifact, OUTSIDE the measured function: standard-logits scoring "
                          "never executes the MTP layer. Layer 45's non-routed tensors "
                          "(eh_proj, enorm, hnorm, shared_head.norm, its norms, o_proj, "
                          "router and shared expert) are native. " + VCRUZ_SRC),
                 asg("norm", "native", "bf16", 16, note="NOT quantized. " + VCRUZ_SRC),
                 asg("lm_head", "native", "bf16", 16,
                     note="the head is RETAINED at source precision: head_bits 16 and "
                          "non_routed_dtype_policy official_source_native, and "
                          "lm_head.weight appears in the index as a plain tensor with no "
                          "{trellis,suh,svh,mcg} group. Unlike stock exllamav3, which "
                          "quantizes it (turboderp's 4.05bpw head is K6). " + VCRUZ_SRC),
                 asg("other", "native", "bf16", 16,
                     note="the vision tower (model.visual.*) is retained natively -- its "
                          "attn.qkv is FUSED, which is the official BF16 layout -- and is "
                          "never executed by text-only scoring. " + VCRUZ_SRC),
             ], "native", kv="bf16", mtp=True),
             VCRUZ("quantizer"),
             [src("url",
                  "https://huggingface.co/api/models/vcruz305/GLM-5.3-Flash-EXL3-K2"
                  "?blobs=true&revision=1718dd403534fd369e82d676e55a50dc19630ffc",
                  None,
                  "138 files, 120 safetensors; all-files sum 97,767,357,064; safetensors "
                  "sum 97,728,721,536, which is also the index's own "
                  "metadata.total_size. The weights were uploaded in one commit "
                  "(a618e7ad, 2026-08-28); every later commit on this repo touches only "
                  "the model card and the runtime/ build context."),
              src("hf_file",
                  "https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2/resolve/"
                  "1718dd403534fd369e82d676e55a50dc19630ffc/config.json",
                  None,
                  "the release's own inline quantization_config -- bits 2, codebook mcg, "
                  "head_bits 16, quant_method exl3, scope glm53_routed_experts_only, "
                  "non_routed_dtype_policy official_source_native, version 0.0.43. Every "
                  "entry in scope.assignments is read from it or from the index."),
              src("hf_file",
                  "https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2/resolve/"
                  "1718dd403534fd369e82d676e55a50dc19630ffc/"
                  "model.safetensors.index.json",
                  None,
                  "150,226 entries: 37,152 quantized modules x 4 objects = 148,608 routed "
                  "payload tensors, plus exactly the official BF16 release's 1,618 "
                  "non-routed names. The name census that fixes the scope is this file's, "
                  "not a producer claim; the runner re-ran it before renting anything and "
                  "the materializer's completeness gate re-ran it on the instance "
                  "(1618/1618, 0 duplicates)."),
              src("hf_file",
                  "https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2/resolve/"
                  "1718dd403534fd369e82d676e55a50dc19630ffc/quantization_config.json",
                  None,
                  "the standalone sidecar. Its header fields agree with the inline block, "
                  "but its tensor_storage map is INCOMPLETE -- 4,180 of the 37,152 "
                  "quantized modules, covering layers 10-13 in full and layer 14 in part -- "
                  "and its serving_reader_qualified reads true where the inline block "
                  "reads false. Nothing on this record is read from it; the scope comes "
                  "from the inline block and the index."),
              src("hf_file",
                  "https://huggingface.co/datasets/malaiwah/"
                  "GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/"
                  "vcruz-k2-2bpw-packed-kld.json",
                  STREAM_VCRUZK2_RECEIPT_SHA,
                  "malaiwah.glm53-vcruz-k2-2bpw-packed-kld-summary.v1")],
             [disc("unsealed_source", "caveat",
                   "The release ships no upstream receipts, no reconstruction closures, no "
                   "sealed reader ABI -- and, unlike every other third-party artifact "
                   "measured on this panel, no per-shard digest list of its own either: "
                   "there is no SHA256SUMS and no EXL3_MANIFEST.json. The provenance "
                   "anchors are therefore the immutable 40-hex revision, the Hub's own "
                   "per-file LFS content digests at that revision (a 122-entry list "
                   "captured before the rental and re-verified against the 120 downloaded "
                   "shards on the measurement instance), the artifact's config and index "
                   "sha256 recomputed locally and bound into the materialization receipt, "
                   "and the consumed-payload sha256 census the capture records.", True),
              disc("routed_experts_only_scope", "info",
                   "2-bit MCG trellis on the routed MoE experts of layers 3-45 and nothing "
                   "else: attention, indexers, mHC, routers, shared experts, the three "
                   "dense layers, embeddings, norms, the vision tower and the head are all "
                   "retained at the official source precision. The measured divergence is "
                   "therefore attributable to the routed experts alone, which is what "
                   "makes it comparable with the TR3 and Dione rungs and NOT with "
                   "turboderp's full-scope releases."),
              disc("native_head_retained", "info",
                   "head_bits 16 and non_routed_dtype_policy official_source_native: the "
                   "lm_head is the official BF16 tensor, present in the index as a plain "
                   "weight. Stock exllamav3 quantizes it; this release does not."),
              disc("revision_unpinned", "caveat",
                   "The release names its parent as zai-org/GLM-5.3-Flash-BF16 (model card "
                   "and base_model metadata) but publishes NO source revision, so "
                   "derived_from_artifact_ref is left empty rather than guessed at one of "
                   "the two BF16 records this registry holds. The lineage that IS "
                   "established is the direction: quantized from BF16, not from the FP8 "
                   "release -- unlike turboderp's exl3 artifacts, whose own "
                   "quantization_config declares an FP8 parent."),
              disc("unreadable_by_stock_loader", "info",
                   "The producer states plainly that stock vLLM cannot load these weights: "
                   "it has neither the exl3 quantization method nor the Glm5Next "
                   "architecture, and they ship prebuilt wheels for a GB10 Spark and an "
                   "L40 TP4 build context alongside. Irrelevant to this measurement, which "
                   "decodes the payloads offline, and material to anyone deciding whether "
                   "to run them."),
              disc("record_note", "info",
                   "The producer publishes their own quality evidence, and it is a "
                   "different question from this registry's: sixcat 0.5.1 think-on at 64k "
                   "scores 84.2 overall (knowledge 65, math 100, truth 85, instruct 75, "
                   "code 90, tools 90) with a trunc-in-think flag on the instruct "
                   "category, plus decode throughput on one GB10 and an L40 TP4 text "
                   "smoke. A capability benchmark is not a fidelity divergence and the two "
                   "cannot be netted against each other; it is recorded on the ARTIFACT, "
                   "where it describes the artifact, and kept off the measurement row.")],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 120,
                            "tensor_parallel": {"pre_sliced": False, "world_size": None}},
             derived_from_artifact_ref=None,
             availability={"status": "public",
                           "uri": "https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2"},
             cross_refs=lair(),
             seal={"sealed": False,
                   "note": "unsealed source, and no producer digest list of any kind; see "
                           "the unsealed_source disclosure for what binds the bytes"}),
    artifact(A_B4, GLM, "brandonmusic GLM-5.3-Flash tr3 4bpw", "quant",
             hf("brandonmusic/GLM-5.3-Flash-tr3-4bpw", None, "none"),
             "exl3", "4bpw", 175642157700,
             codec("exl3-mcg", 4.0, None, tool="exllamav3"),
             EXL3_SCOPE_UNIFORM(4.0), BRANDON("quantizer"),
             [src("github_file", "https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json",
                  "d955bfaedad36ad9841c30808c67fc36b72017f87b720fb460d8e1c13fe75e57",
                  "student_checkpoint_identity_sha256 598ce08d..., student_label uniform-k4, profile k4-tp2")],
             [SCOPE_CORRECTED,
              REV_UNPINNED("His receipt pins the checkpoint by student_checkpoint_identity_sha256 598ce08d... "
                           "Our own crosscheck notes his metadata records an earlier repo revision and that the "
                           "weights were never modified post-upload (config/template churn only)."),
              disc("size_unverified", "info",
                   "Byte total is the Hub safetensors sum observed during mining, at a Hub head later than the "
                   "measurement; treat as approximate.")],
             weights_extra={"size_basis": "repo_weight_files"},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public", "uri": "https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw"},
             cross_refs=lair(), seal={"sealed": True, "note": "ships its own five-cold-run receipt and reader digest 1fb3be87..."}),
    # The SAME WEIGHTS as A_B4, redistributed at a PINNED revision. It gets its
    # own record rather than being folded into brandonmusic's for one reason: it
    # is the tree that was actually fetched and measured, and a registry that
    # says "we measured X" must name the bytes it opened. The relationship is
    # not asserted from the mirror's README -- it is proven twice, and both
    # proofs are in the sources below.
    artifact(A_TR3MIRROR, GLM,
             "Mia-AiLab GLM-5.3-Flash EXL3 TR3 4bpw (byte-identical mirror of brandonmusic's)",
             "quant",
             hf("Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
                "024db9f7e9871e8efdf21538ba55af7442be3cd5", "hf_api"),
             # M4 normalization: this is the only rung of the measured GLM-5.3-Flash
             # ladder that reported a WEIGHT-FILE sum where turboderp's, 0xSero's and
             # vcruz305's rows report an all-files sum, so the size column was not
             # like-for-like. It is repo_all_files now (175,715,854,761 over 144 files
             # at 024db9f7); the safetensors sum stays in the note because THAT is the
             # quantity byte-identical to brandonmusic's original, and his row keeps
             # repo_weight_files precisely because his repo is unpinned -- an all-files
             # sum for a moving revision is not a fact.
             "exl3", "4bpw", 175715854761,
             codec("exl3-mcg", 4.0, None, tool="exllamav3", version="0.0.43"),
             EXL3_SCOPE_UNIFORM(4.0), BRANDON("quantizer"),
             [src("url",
                  "https://huggingface.co/api/models/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
                  "?blobs=true&revision=024db9f7e9871e8efdf21538ba55af7442be3cd5",
                  None,
                  "the size on this record: 144 files, all-files sum 175,715,854,761 "
                  "(weights.size_bytes, basis repo_all_files); safetensors sum "
                  "175,642,157,752, which is the quantity byte-identical to "
                  "brandonmusic's original and what this row previously reported. "
                  "Re-based to all-files in M4 so the four MEASURED rungs of this "
                  "ladder share one axis; brandonmusic's own row keeps "
                  "repo_weight_files because his repo is not pinned to a revision and "
                  "an all-files sum for a moving tree is not a fact."),
              src("url",
                  "https://huggingface.co/api/models/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
                  "/revision/024db9f7e9871e8efdf21538ba55af7442be3cd5?blobs=true",
                  None,
                  "PROOF 1 of byte-identity: the LFS oid of all 120 *.safetensors is "
                  "equal, file for file, to brandonmusic/GLM-5.3-Flash-tr3-4bpw @ "
                  "5ab363a8dcf6405955fd5f99671e01a1c9fb124b. Of the 142 files the two "
                  "repos share, only README.md differs; the mirror adds MIRROR.json and "
                  "ORIGINAL_MODEL_CARD.md and omits brandonmusic's 192 "
                  ".materialization/shards/*.json sidecars."),
              src("hf_file",
                  "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/"
                  "024db9f7e9871e8efdf21538ba55af7442be3cd5/SHA256SUMS",
                  None,
                  "PROOF 2, independent of the Hub's own hashing: the mirror republishes "
                  "brandonmusic's SHA256SUMS verbatim, and all 120 downloaded shards were "
                  "verified against it byte-wise on the measurement instance "
                  "(malaiwah.published-sums-verification.v1: weights_verified 120, "
                  "weights_failed 0, weights_not_covered_by_list 0)."),
              src("hf_file",
                  "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/"
                  "024db9f7e9871e8efdf21538ba55af7442be3cd5/MIRROR.json",
                  None,
                  "the mirror's own declaration: mirror_of brandonmusic/"
                  "GLM-5.3-Flash-tr3-4bpw @ 5ab363a8, quant_author 'Brandon M. Music', "
                  "'Byte-identical redistribution. Not an original quantization.' It is "
                  "recorded because the producer said it, and it is BELIEVED because the "
                  "two proofs above check it."),
              src("hf_file",
                  "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/"
                  "024db9f7e9871e8efdf21538ba55af7442be3cd5/config.json",
                  "4f5341e048984459471bfb9c894e6bf87e69b9c67402672af901631d1349f265",
                  "the release's own quantization_config: bits 4, codebook mcg, "
                  "head_bits 16, scope glm53_routed_experts_only, "
                  "non_routed_dtype_policy official_source_native, version 0.0.43. "
                  "Every per-tensor-class entry in scope.assignments is read from it."),
              src("hf_file",
                  "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/resolve/"
                  "024db9f7e9871e8efdf21538ba55af7442be3cd5/materialization-receipt.json",
                  "092be1ffa8db66bf02d4c370d0433a57aa48d4a6e5ce89723ef6a3bb7ca32643",
                  "the release's own seal. Its self-declared receipt_sha256 RECOMPUTES "
                  "from its canonical content; it states native_tensor_count 1618, "
                  "packed_tensor_count 148608, routed_choice_count 37152, "
                  "nonrouted_native_exact true, and it binds config_sha256 / "
                  "index_sha256 to the published files.")],
             [disc("byte_identical_redistribution", "info",
                   "This is NOT an original quantization. Its 120 weight shards are "
                   "byte-identical to brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8 "
                   "(artifact--brandonmusic.glm-5.3-flash-tr3-4bpw), proven two ways: "
                   "equal LFS oids for all 120 files, and a byte-wise verification of the "
                   "downloaded shards against brandonmusic's own published SHA256SUMS, "
                   "which the mirror republishes verbatim. Credit for the quantization is "
                   "brandonmusic's; the mirror's contribution is a pinned, durable fetch "
                   "target -- which is why the measurement targets it: the upstream record "
                   "carries no revision at all.", True),
              disc("sealed_source_verified", "info",
                   "Unlike every other third-party artifact in this registry, this release "
                   "SEALS itself: exl3-mcg-storage-abi.json and "
                   "materialization-receipt.json state digests over the emitted tensor-name "
                   "set, the materialization plan, the config and the index. All 12 claims "
                   "were RECOMPUTED from the published bytes -- once before renting "
                   "anything (a few hundred KB), and again on the instance against the "
                   "downloaded tree. The ABI's own serving_reader_qualified=false and empty "
                   "qualified_tp_sizes concern TP SERVING ('ExLlamaV3 v0.0.43 has no "
                   "audited GLM-5.3 TP model load/inference receipt'), not the offline "
                   "single-device decode this registry's measurement performs; its "
                   "storage_checkpoint_verified is true.")],
             weights_extra={"size_basis": "repo_all_files", "shard_count": 120},
             derived_from_artifact_ref=A_BF16_A6,
             availability={"status": "public",
                           "uri": "https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"},
             cross_refs=lair(),
             seal={"sealed": True,
                   "note": "exl3-mcg-storage-abi.json + materialization-receipt.json "
                           "(receipt 092be1ff..., plan a359003a...); all 12 claims "
                           "recomputed from the published bytes by tr3_surface.verify_seal"}),
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
    # this streaming lane's zero-point. See engines/BF16-FLOOR.md for the analysis.
    #
    # It is NOT the cross-stack floor (measurement--glm53.bf16-replay-floor...,
    # 0.012712 nats, a different pipeline and a different comparability key): that
    # number bounds CROSS-STACK rows on this panel and must never be subtracted from a
    # same-stack streaming row, nor this floor from a cross-stack one. BIAS-006 (new)
    # refuses a floor_measurement_ref that crosses lanes; build_row refuses it at
    # write time (exit 7) before a row like that could even be generated.


    # brandonmusic's TR3 EXL3/MCG 4bpw, fetched from the Mia-AiLab mirror at a
    # PINNED revision and measured 2026-08-29 by bin/measure-cloud. Same lane,
    # same panel, same teacher as the K6/K8/floor/turbo rows above --
    # comparability key cmp--202b717f3219c414.
    #
    # This row answers two DIFFERENT questions, and they must not be blurred:
    #   * SAME WEIGHTS, DIFFERENT LANE AND STACK. brandonmusic's own
    #     author-reported five-run number on his sealed EP8 stack is 0.024554564249958208
    #     (measurement--glm53.brandonmusic-4bpw.brandonmusic-final25). The bytes
    #     are provably identical -- equal LFS oids for all 120 shards, plus a
    #     byte-wise verification against his own published SHA256SUMS -- so the
    #     difference of +0.000948863 nats is lane PLUS stack and nothing
    #     else. It is the only such pair in this registry; every other lane
    #     comparison changes the weights too.
    #   * SAME LANE, SAME RATE, DIFFERENT QUANTIZER. turboderp's stock-exllamav3
    #     4.05bpw reads 0.025526426915472484 on this exact lane. Same panel, same teacher,
    #     same estimator, ~the same nominal rate, and three declared differences:
    #     scope (routed-experts-only with a native BF16 head vs full-scope with a
    #     6-bit head), codebook (mcg vs mul1), and lineage (quantized from BF16
    #     vs from the FP8 release).
    TR34 = 0.02550342763436377
    D30 = 0.050501241465423556
    out.append(M("measurement--glm53.dione-3.0bpw-stream.brandonmusic-final25", GLM,
                 A_DIONE30, P_B25, R_B25, PL_STREAM, D30,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 top1=0.9299658036150464,
                 scored_positions=51175, contexts=25, runs=2, cold=True,
                 run_means=[0.050501241465423556, 0.050501241465423556],
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=['845617b34375f5b47d6b9b40cb19e822c808761322cf859cf9209d59c30a00c8'],
                 det_note=('2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct tokenwise_kld_sha256. The report-file digests differ per run and prove nothing; the tokenwise digest is the determinism evidence.'),
                 sources=[src("receipt_file", STREAM_DIONE30_RECEIPT,
                              STREAM_DIONE30_RECEIPT_SHA,
                              "malaiwah.glm53-dione-3.0bpw-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_DIONE30_RECEIPT,
                              STREAM_DIONE30_RECEIPT_SHA),
                          src("hf_file",
                              "https://huggingface.co/datasets/malaiwah/"
                              "GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/"
                              "dione-3.0bpw-packed-kld.json", STREAM_DIONE30_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-dione-3.0bpw-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown",
                       "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane, whose offset against "
                                 "the sealed-ep8 lane is known to be non-zero and is NOT "
                                 "measured for this artifact: it has no sealed-lane row to "
                                 "bridge against. Its 4-bpw SIBLING does "
                                 "(measurement--glm53.dione-q4.brandonmusic-final25, "
                                 "0.027262784814670614 on the sealed lane), but a lane "
                                 "offset is a property of the routing, not a constant, so "
                                 "it does not transfer between rungs of a ladder. This "
                                 "lane's own measurement floor (%s) is %r nats; netting "
                                 "it out gives an estimated quantization-attributable "
                                 "error of %r nats here -- an estimate, not an identity, "
                                 "because KL is not additive, and it is only meaningful "
                                 "because both terms share the same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, D30 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06,
                       "threshold_gt": None, "passed": True},
                 disclosures=[
                     disc("third_party_artifact_self_measured", "info",
                          "Someone else's weights, our measurement. 0xSero produced the "
                          "artifact; malaiwah produced the number. Credit for the artifact "
                          "is theirs."),
                     disc("unsealed_source", "caveat",
                          "The Dione checkpoint ships no upstream receipts, reconstruction "
                          "closures or sealed reader ABI, so the packed surface was decoded "
                          "WITHOUT seal verification. What the release DOES publish is a "
                          "per-shard sha256 manifest (EXL3_MANIFEST.json), and all 130 shard "
                          "digests were recomputed on the measurement instance before "
                          "anything was decoded (dione_shard_hash_verification: full); "
                          "that, the immutable revision, the local config/index digests and "
                          "the consumed-payload sha256 census are the provenance anchors.",
                          True),
                     disc("reduced_run_count", "caveat",
                          "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)"),
                     # PROV-012: the comparability key carries no lane input, so a
                     # streaming row tabled beside sealed-lane rows must SAY which
                     # machine produced it. registry_add emits this automatically; a
                     # hand-authored seed row does not, and the invariant is the thing
                     # that caught it.
                     disc("non_sealed_lane", "caveat",
                          "Produced by the 'streaming' lane, not the sealed-ep8 lane. The "
                          "lane's offset against the sealed lane is NOT measured for this "
                          "artifact: no sealed-lane row for it exists to bridge against, "
                          "and the offset its 4-bpw sibling would give is a property of "
                          "the routing rather than a constant of the ladder.", True),
                     # Both of these are the OPPOSITE of a caveat, and both are what
                     # registry_add's dione adapter emits from the same receipt.
                     disc("shard_hashes_verified", "info",
                          "dione_shard_hash_verification=full (verbatim from the receipt): "
                          "all 130 shard sha256s were recomputed against the release's own "
                          "EXL3_MANIFEST.json on the measurement instance -- 135 s over "
                          "149.6 GB -- before any payload was decoded, and every weight file "
                          "on disk was covered by that list. The release publishes no seal, "
                          "so this and the immutable revision are the provenance anchors."),
                     disc("native_head_retained", "info",
                          "declared_head_bits 16 (verbatim from the receipt): this release "
                          "retains the lm_head at source precision, unlike a stock-exllamav3 "
                          "release which quantizes it. The head is APPLIED natively from the "
                          "artifact's own weights, which is why estimator.head_policy is "
                          "native_head."),
                 ],
                 notes=("Third rung of 0xSero's ladder measured here. His Q4 reads 0.027262784814670614 on the SEALED lane and this 3.0bpw reads 0.050501241465423556 on the STREAMING lane; the two are not directly comparable (different lane, different comparability key) and the registry refuses to net them. Within this lane the attributable error against the BF16 floor is 0.03899531884609326 nats. The producer's own RELEASE_STATUS.json marks this release quality: FAIL at their own threshold (their held-out forward KL 0.15251, top-1 0.87285 over 65,504 positions of THEIR panel) -- their number, their panel, their estimator, recorded on the artifact record rather than mixed into this one.")))
    # ---- M4 measured values, transcribed from receipts/malaiwah/
    # stream-vcruz-k2-2bpw-kld.json, whose sha256 this row cites and whose
    # bytes `make reseed-check` re-reads.
    M4_VALUE = 0.15520955491423008
    M4_RUN1 = 0.15520955491423008
    M4_RUN2 = 0.15520955491423008
    M4_TOP1 = 0.8726526624328286
    M4_IDENTICAL = True
    M4_EVIDENCE = ["a75500a9d060dacff12023553b2360ac0a29d28ebd95b844bf9b39e312a291ee"]
    M4_GATE_PASSED = False
    M4_DET_NOTE = ("2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct "
                   "tokenwise_kld_sha256. The report-file digests differ per run and "
                   "prove nothing; the tokenwise digest is the determinism evidence.")
    M4_DISCLOSURES = [
        disc("third_party_artifact_self_measured", "info",
             "Someone else's weights, our measurement. vcruz305 produced the artifact; "
             "malaiwah produced the number. Credit for the artifact is theirs."),
        disc("quality_gate_failed", "caveat",
             "The panel's gate is mean tokenwise KLD < 0.06 and this row reads "
             "0.15520955491423008 -- 2.6x the threshold, and 3.07x the 3.0-bpw rung "
             "immediately above it. The gate is the artifact's verdict, not the "
             "measurement's: the run is bitwise deterministic, its two cold runs agree "
             "to the last bit, and the number is published exactly as it came out. "
             "What it says is that a 2-bit routed-expert quantization of this model "
             "diverges by 0.155 nats from its own BF16 source on this panel, at 87.27 % "
             "top-1 agreement.", True),
        disc("unsealed_source", "caveat",
             "The release ships no upstream receipts, no reconstruction closures, no "
             "sealed reader ABI -- and no per-shard digest list of its own: no "
             "SHA256SUMS, no EXL3_MANIFEST.json. What binds the bytes is the immutable "
             "40-hex revision, the Hub's own per-file LFS content digests at that "
             "revision -- a 122-entry list, manifest digest 43a162282c06b19d09802"
             "9afea4bedc77238026bca28fc514e50e33d827a9b66, captured from the models "
             "API BEFORE the rental and recomputed on the instance against the "
             "downloaded tree: 122/122 verified, 97,764,515,699 bytes, 0 absent, 0 "
             "safetensors on disk uncovered by the list -- plus the artifact's "
             "config sha256 163bd0888684f7eaf963ad67cdff3fbdca0749796c0aa5a6e7035816"
             "e503ecfc and index sha256 e9dd7cb2f6358843de334baa40ff537b4914721dbaa9c"
             "7dab42a386562afce19 recomputed locally and bound into the "
             "materialization receipt, and the consumed-payload sha256 census.", True),
        disc("reduced_run_count", "caveat",
             "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 "
             "(budget; disclosed)"),
        disc("non_sealed_lane", "caveat",
             "Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's "
             "offset against the sealed lane is NOT measured for this artifact: it has "
             "no sealed-lane row, and no sibling on either lane, to bridge against.",
             True),
        disc("native_head_retained", "info",
             "declared_head_bits 16 (verbatim from the receipt): this release retains "
             "the lm_head at source precision, unlike a stock-exllamav3 release which "
             "quantizes it. The head is APPLIED natively from the artifact's own "
             "weights, which is why estimator.head_policy is native_head."),
        disc("routed_experts_only_scope", "info",
             "The quantized scope is the routed MoE experts and nothing else, so the "
             "divergence is attributable to them alone -- comparable with the TR3 and "
             "Dione rungs of this ladder, and NOT with turboderp's full-scope "
             "releases, which also quantize attention, the dense layers and the head."),
    ]
    M4_NOTES = (
        "The lowest rate measured on this panel, and the first row here to FAIL the "
        "0.06 gate. 0.15520955491423008 nats at 87.27 % top-1, against 0.050501241465423556 "
        "at 93.00 % for the 3.0-bpw rung and 0.025503427634363770 at 95.31 % for 4 bpw: "
        "3.07x the divergence of 3 bpw for 35 % fewer bytes (97.8 GB against 149.6 GB), "
        "and 6.09x the divergence of 4 bpw for 44 % fewer bytes. Against this lane's own "
        "BF16 floor the quantization-attributable error is 0.143703632294899769 nats. "
        "Both cold runs produced identical run means and ONE tokenwise KL digest, so the "
        "path is bitwise deterministic; the divergence is the codec, not the harness. "
        "Every one of the 907,200 decoded expert matrices was K2 "
        "(routed_bits_decode_histogram {K2: 907200}), which is the decode side confirming "
        "the release's declared routed-experts-only scope. Per-domain the damage is "
        "uneven: axis2_legal 0.2509, axis1_general 0.1727, axis3_code_agentic 0.1272, "
        "axis4_reasoning_termination 0.0671 -- a 3.7x spread across domains that the "
        "single panel mean hides.")

    # M4: the lowest rate on this panel, and the first row here whose quality gate
    # does not necessarily pass. Whatever the number is, it is published: a gate
    # verdict is a finding about the artifact, not a verdict on the measurement.
    VCRUZK2 = M4_VALUE
    out.append(M("measurement--glm53.vcruz-k2-2bpw-stream.brandonmusic-final25", GLM,
                 A_VCRUZK2, P_B25, R_B25, PL_STREAM, VCRUZK2,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 top1=M4_TOP1,
                 scored_positions=51175, contexts=25, runs=2, cold=True,
                 run_means=[M4_RUN1, M4_RUN2],
                 identical=M4_IDENTICAL, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=M4_EVIDENCE,
                 det_note=M4_DET_NOTE,
                 sources=[src("receipt_file", STREAM_VCRUZK2_RECEIPT,
                              STREAM_VCRUZK2_RECEIPT_SHA,
                              "malaiwah.glm53-vcruz-k2-2bpw-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_VCRUZK2_RECEIPT,
                              STREAM_VCRUZK2_RECEIPT_SHA),
                          src("hf_file",
                              "https://huggingface.co/datasets/malaiwah/"
                              "GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/"
                              "vcruz-k2-2bpw-packed-kld.json", STREAM_VCRUZK2_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-vcruz-k2-2bpw-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown",
                       "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane, whose offset against "
                                 "the sealed-ep8 lane is known to be non-zero and is NOT "
                                 "measured for this artifact: it has no sealed-lane row "
                                 "to bridge against, and no sibling of its own on either "
                                 "lane. This lane's own measurement floor (%s) is %r "
                                 "nats; netting it out gives an estimated "
                                 "quantization-attributable error of %r nats here -- an "
                                 "estimate, not an identity, because KL is not additive, "
                                 "and it is only meaningful because both terms share the "
                                 "same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, VCRUZK2 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06,
                       "threshold_gt": None, "passed": M4_GATE_PASSED},
                 disclosures=M4_DISCLOSURES,
                 notes=M4_NOTES))
    out.append(M("measurement--glm53.tr3-4bpw-stream.brandonmusic-final25", GLM,
                 A_TR3MIRROR, P_B25, R_B25, PL_STREAM, TR34,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 top1=0.9531411822178798,
                 scored_positions=51175, contexts=25, runs=2, cold=True,
                 run_means=[0.02550342763436377, 0.02550342763436377],
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=['31177f244e7c2dca7c80863f8c0859596bf0308e3d80d586019d9d9cfb16e09c'],
                 det_note=("2 cold runs, 2 distinct kld_report_sha256 "
                           "values, 1 distinct tokenwise_kld_sha256. The report-file "
                           "digests differ per run and prove nothing; the single "
                           "tokenwise digest is the determinism evidence."),
                 sources=[src("receipt_file", STREAM_TR34_RECEIPT, STREAM_TR34_RECEIPT_SHA,
                              "malaiwah.glm53-tr3-4bpw-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_TR34_RECEIPT,
                              STREAM_TR34_RECEIPT_SHA),
                          src("hf_file",
                              "https://huggingface.co/datasets/malaiwah/"
                              "GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/"
                              "tr3-4bpw-packed-kld.json", STREAM_TR34_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-tr3-4bpw-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown",
                       "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane. Unlike every other "
                                 "streaming row, this artifact HAS a sealed-lane sibling to "
                                 "bridge against: the same bytes read 0.024554564249958208 there "
                                 "(author-reported, brandonmusic's own five-run receipt on "
                                 "his own stack), so the streaming-lane number sits "
                                 "+0.000948863 nats from it -- a LANE-PLUS-STACK offset, "
                                 "not a lane offset, because the reader digests differ too "
                                 "(1fb3be87... vs 1ccce446...). This lane's own measurement "
                                 "floor (%s) is %r nats; netting it out gives an estimated "
                                 "quantization-attributable error of %r nats here -- an "
                                 "estimate, not an identity, because KL is not additive, and "
                                 "it is only meaningful because both terms share the same "
                                 "reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, TR34 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06,
                       "threshold_gt": None, "passed": True},
                 disclosures=[
                     disc("sealed_source_verified", "info",
                          "The release publishes its own storage ABI and materialization "
                          "receipt, and this measurement RECOMPUTED all 12 of their claims "
                          "from the published bytes before decoding: the receipt's own "
                          "self-seal, the config/index digests, the sha256 over all 150,226 "
                          "emitted tensor names, plan_sha256 agreement, the "
                          "packed/native/output count algebra, nonrouted_native_exact, and "
                          "the non-routed name-set bijection against the official release. "
                          "Shard bytes: the receipt's shard_sha256 map equals the published "
                          "SHA256SUMS, against which all 120 downloaded shards verified "
                          "byte-wise. This is the only third-party row here that carries a "
                          "verified seal rather than an unsealed_source caveat.", False),
                     disc("byte_identical_redistribution", "info",
                          "The measured bytes are brandonmusic's, redistributed: all 120 "
                          "shards have the same LFS oid as "
                          "brandonmusic/GLM-5.3-Flash-tr3-4bpw @ 5ab363a8. The mirror was "
                          "measured rather than the upstream because it pins a revision and "
                          "the upstream record carries none. Credit for the quantization is "
                          "brandonmusic's; credit for this number is ours.", True),
                     disc("routed_experts_only_scope", "info",
                          "scope glm53_routed_experts_only, non_routed_dtype_policy "
                          "official_source_native, head_bits 16, read from the release's own "
                          "config. Only the routed experts are quantized; all 1,618 "
                          "non-routed tensors including lm_head are the OFFICIAL ones, "
                          "verified name-set-equal to the official release's. The "
                          "stock-exllamav3 rows on this same panel quantize attention, the "
                          "dense MLPs, the shared experts, the vision tower and the head as "
                          "well: at ~the same nominal bpw they are measuring a different "
                          "amount of model.", True),
                     disc("reduced_run_count", "caveat",
                          "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)",
                          True),
                     disc("non_sealed_lane", "caveat",
                          "Produced by the 'streaming' lane, not the sealed-ep8 lane. Unlike "
                          "every other streaming row here, this artifact HAS a sealed-lane "
                          "sibling to bridge against, because the bytes are provably "
                          "identical to brandonmusic's: the same weights read 0.024554564249958208 "
                          "there. The +0.000948863 nats between them is a "
                          "LANE-PLUS-STACK offset, not a lane offset -- his run used his "
                          "reader (1fb3be87...) and ours uses ours (1ccce446...) -- so it "
                          "bounds the lane term rather than measuring it.", True),
                     disc("third_party_artifact_self_measured", "info",
                          "Someone else's weights, our measurement. brandonmusic produced "
                          "the artifact and Mia-AiLab redistributed it; malaiwah produced "
                          "the number.")],
                 notes="First tr3-published artifact measurable by this suite: the streaming lane gained a reader (stream_score --source tr3, k6/tools/tr3_surface.py) in the same change. The routed decode is the campaign's own -- exl3hf_surface.decode_module over the frozen MCG LUT, proven bitwise identical to calling it directly -- so the codec path is the one the K6/K8 rows on this lane were measured through. The non-routed weights are the ARTIFACT's own, re-sharded VERBATIM by the materializer (1,618 tensors copied, 0 decoded, dtypes preserved) because they share shards with the 148,608 routed payload objects and transformers keys its checkpoint load off the shard files. No official-release weight is in the measured function. 907200 K4 expert matrices were decoded per cold run. Attributable error against this lane's own floor: 0.013997505 nats, versus 0.014020504 for turboderp's 4.05bpw -- the TR3 quant is the tighter of the two at ~the same nominal rate, on a strictly smaller quantized scope."))

    # turboderp's stock-exllamav3 4.05bpw, measured 2026-08-29 by bin/measure-cloud's
    # first end-to-end run. Same lane, same panel, same teacher as the K6/K8/floor rows
    # above -- comparability key cmp--202b717f3219c414, identical to K8's.
    STURBO405 = 0.025526426915472484
    out.append(M("measurement--glm53.turbo-4.05bpw-stream.brandonmusic-final25", GLM,
                 A_TURBO405, P_B25, R_B25, PL_STREAM, STURBO405,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 # identical in both cold runs, read from run-N/kld-report.json
                 top1=0.9509916951636541,
                 scored_positions=51175, contexts=25, runs=2, cold=True,
                 run_means=[STURBO405] * 2,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["68cc5f61c8924c9962bdd446f60dc84a7880d2ab2eefa3020cdc2fa5d3275aa9"],
                 det_note="2 cold runs, 2 distinct kld_report_sha256 values, 1 distinct "
                          "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                          "nothing; the single tokenwise digest is the determinism evidence.",
                 sources=[src("receipt_file", STREAM_TURBO405_RECEIPT, STREAM_TURBO405_RECEIPT_SHA,
                              "malaiwah.glm53-turbo-4.05bpw-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_TURBO405_RECEIPT,
                              STREAM_TURBO405_RECEIPT_SHA),
                          src("hf_file",
                              "https://huggingface.co/datasets/malaiwah/"
                              "GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/"
                              "turbo-4.05bpw-packed-kld.json",
                              "68ef836737f9eeb59f62da5107246249fcd30c462ccf25493bab75f917df0706")],
                 receipt_schema="malaiwah.glm53-turbo-4.05bpw-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown", "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane, whose offset against the sealed-ep8 "
                                 "lane is known to be non-zero but was NOT measured for this artifact: no "
                                 "sealed-lane row for it exists to bridge against. This lane's own "
                                 "measurement floor (%s) is %r nats; netting it out gives an estimated "
                                 "quantization-attributable error of %r nats here -- an estimate, not an "
                                 "identity, because KL is not additive, and it is only meaningful because "
                                 "both terms share the same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, STURBO405 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": True},
                 disclosures=[
                     disc("unsealed_source", "caveat",
                          "seal_disclosure (verbatim from the receipt): unsealed-source scoring: stock "
                          "exllamav3 releases ship no upstream receipts, reconstruction closures or "
                          "sealed reader ABI; the packed surface was decoded WITHOUT seal verification "
                          "(consumed payload sha256s and the immutable repo revision are recorded "
                          "instead)", True),
                     disc("reduced_run_count", "caveat",
                          "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 "
                          "(budget; disclosed)", True),
                     disc("quantized_head", "caveat",
                          "declared_head_bits 6 (verbatim from the receipt): this artifact's lm_head is "
                          "itself quantized by the producer, unlike the TR3 artifacts on this panel "
                          "which keep it native BF16. It is APPLIED natively from the artifact's own "
                          "weights -- no shared or replayed head -- so estimator.head_policy is "
                          "native_head; the quantization is artifact identity.", True),
                     disc("non_sealed_lane", "caveat",
                          "Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset "
                          "against the sealed lane is NOT measured for this artifact: no sealed-lane row "
                          "for it exists to bridge against.", True),
                     disc("third_party_artifact_self_measured", "info",
                          "The artifact is turboderp's; the measurement is ours. Credit for the quant "
                          "and credit for the number are separate.", False)],
                 notes="First artifact measured end to end by bin/measure-cloud. The receipt's family "
                       "name carries no lane marker, so 'streaming' is SUPPLIED by --lane; direction, "
                       "accumulation dtype, scored positions and context count are supplied too (this "
                       "family is a scalar summary). determinism.identical_across_runs is RECOMPUTED "
                       "from run_means and distinct_tokenwise_kld_sha256. The non-routed weights are "
                       "the ARTIFACT's own, dequantized from its shards -- including its 6-bit head -- "
                       "so no official-release weight is in the measured function; the materialization "
                       "receipt is 3653c55f0dc729c3fccc6bbe5d8949b55e27517ade5d8c546fec79de03dd1c81. "
                       "907,200 K4 expert matrices were decoded per cold run. Top-1 agreement "
                       "0.9509916951636541, identical across both cold runs, read from the "
                       "per-run kld-report.json (the scalar summary family did not carry it "
                       "at the time this row was written; k6_kld_report now emits it)."))
    STURBO205 = 0.12163767673339457
    out.append(M("measurement--glm53.turbo-2.05bpw-stream.brandonmusic-final25", GLM,
                 A_TURBO205, P_B25, R_B25, PL_STREAM, STURBO205,
                 metric_name="mean_of_run_means_tokenwise_kld",
                 top1=0.8891841719589644,
                 scored_positions=51175, contexts=25, runs=2, cold=True,
                 run_means=[STURBO205] * 2,
                 identical=True, evidence_kind="tokenwise_kld_sha256",
                 evidence_hashes=["9d27f2cbc2c1f27079ee7e80d0185268ef2d0b0ac4a5cf709ab7fd5dededed4e"],
                 det_note="2 cold runs, 2 distinct kld_report_sha256, 1 distinct "
                          "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                          "nothing; the tokenwise digest is the determinism evidence.",
                 sources=[src("receipt_file", STREAM_TURBO205_RECEIPT, STREAM_TURBO205_RECEIPT_SHA,
                              "malaiwah.glm53-turbo-2.05bpw-packed-kld-summary.v1"),
                          src("hf_file", HF_REGISTRY_RAW + STREAM_TURBO205_RECEIPT,
                              STREAM_TURBO205_RECEIPT_SHA)],
                 receipt_schema="malaiwah.glm53-turbo-2.05bpw-packed-kld-summary.v1",
                 cls="advisory",
                 bias={"kind": "other", "direction": "unknown", "floor_measurement_ref": M_BF16_FLOOR,
                       "estimated_magnitude": None,
                       "detail": "Measured on the 'streaming' lane, whose offset against the sealed-ep8 "
                                 "lane is known to be non-zero but was NOT measured for this artifact. "
                                 "This lane's own measurement floor (%s) is %r nats; netting it out "
                                 "gives an estimated quantization-attributable error of %r nats here -- "
                                 "an estimate, not an identity, because KL is not additive, and it is "
                                 "only meaningful because both terms share the same reference and lane."
                                 % (M_BF16_FLOOR, BF16_FLOOR, STURBO205 - BF16_FLOOR)},
                 gate={"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                       "passed": False},
                 disclosures=[
                     disc("unsealed_source", "caveat",
                          "seal_disclosure (verbatim from the receipt): unsealed-source scoring: stock "
                          "exllamav3 releases ship no upstream receipts, reconstruction closures or "
                          "sealed reader ABI; the packed surface was decoded WITHOUT seal "
                          "verification.", True),
                     disc("reduced_run_count", "caveat",
                          "cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 "
                          "(budget; disclosed)", True),
                     disc("quantized_head", "caveat",
                          "declared_head_bits 5 -- lower than the 6 this producer's 4.05bpw and "
                          "3.05bpw branches declare. Applied natively from the artifact's own "
                          "weights, so estimator.head_policy is native_head.", True),
                     disc("non_sealed_lane", "caveat",
                          "Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's "
                          "offset against the sealed lane is NOT measured for this artifact.", True),
                     disc("cross_hardware", "info",
                          "This value was measured on an H200, as were the other rows in its "
                          "comparability group. It was independently REPRODUCED on A100 hardware at "
                          "two different providers (Vast A100 PCIe and RunPod A100-SXM4), which "
                          "agreed with each other BITWISE -- same tokenwise-KLD tensor hash -- and "
                          "differ from this H200 value by 2.973e-04 nats (0.245%). The GPU model, "
                          "not the provider or the host, is the discriminator; see "
                          "docs/ARCHITECTURE-DETERMINISM.md. That term is larger than the gap "
                          "between some 4-bpw rows in this registry, and it cancels only because "
                          "those rows share this row's hardware.")]))
    # Stamped after the row is built, the way the Qwen rows are: stamp_harness()
    # honours a row that already carries a recorded block. This is the FIRST row
    # in this registry with a recorded harness rather than the grandfather
    # clause -- the allowlist froze on 2026-08-30 and this number came after it.
    out[-1]["harness"] = stream_lane_harness()
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


# ===========================================================================
# 8. GLM-5.2-SIQ-Fruit -- the first family measured with the three-step
#    fidelity-dataset architecture (capture / capture / compare).
#
# Fruit is the registry maintainer's own trained model, so the reference is
# unambiguous: no borrowed teacher, no third-party checkpoint, no lane bridged
# to another lane. That also means nothing here has been independently
# reproduced, which the model row says out loud.
# ===========================================================================
FRUIT = "model--malaiwah.glm-5.2-siq-fruit"
F_BF16 = "artifact--malaiwah.glm-5.2-siq-fruit-bf16"
F_SIQ = "artifact--malaiwah.glm-5.2-siq-fruit.exl3-k3k4"
P_FRUIT = "panel--fruit.malaiwah.heldout-v1"
R_FRUIT = "reference--malaiwah.fruit-bf16-hf.heldout-v1"
PL_FIDDS = "pipeline--malaiwah.fidelity-dataset-hf"

FRUIT_ROOT_DS = "https://huggingface.co/datasets/malaiwah/fruit-fidelity-root-v1"
FRUIT_QUANT_DS = "https://huggingface.co/datasets/malaiwah/fruit-fidelity-quant-siq-v1"
FRUIT_TOKEN_SHA = "a6d367cc3ba448800372dee435d2bb4f536d23ca68843628832fa3b122ceabe1"
FRUIT_PANEL_RECEIPT_SHA = "6c195ef252305d0e647c2f33b99e60d927c306a4e86ff3a055173297bc9a403c"
FRUIT_ROOT_DATASET_SHA = "f56674f9159a68fa4abd5ccb2e727aadf510f35ce1b44b5cc4b825987560e7cf"
FRUIT_ROOT_CAPTURE_SHA = "b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c"
FRUIT_QUANT_DATASET_SHA = "135776882d1ad3b4a1bdd0401035e61905351ad296b90b42396b45412ea270a0"
FRUIT_QUANT_CAPTURE_SHA = "8875fe45cffdb958f39d0e39a3e26a885b26dc48b2509cb4c13276ecdcb9d49e"
FRUIT_HEAD_SHA = "8d0f7d6e35c48a3f6b97f5f5ebc24657a3fd32e1e6c22adc710a9b317b5b5440"
# The exporter commit these line citations were read against. Pinned, because a
# line number against a moving branch is not provenance.
PROXY_FRUIT_PIN = "75b0840fe2ff42181945fab94bd4a81286114422"

MODELS += [
    {"schema_version": V, "id": FRUIT, "name": "GLM-5.2-SIQ-Fruit", "family": "glm5.2",
     "license": "apache-2.0",
     "publisher": MAL("model-publisher"),
     "huggingface": hf("malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5"),
     "architecture": {
         "kind": "moe-decoder", "hidden_size": 1024, "num_layers": 13, "vocab_size": 154880,
         "has_mtp": True, "total_parameters": 5040000000, "active_parameters": 460000000,
         "note": "GlmMoeDsaForCausalLM: 3 dense + 10 sparse layers, 256 routed experts at "
                 "top-8, moe_intermediate_size 512, MLA attention with a DSA lightning "
                 "indexer, and one co-trained MTP draft layer at index 13. A serving proxy "
                 "for the GLM-5.2 family (about 1:150 by total parameters), trained by the "
                 "registry maintainer as a CI fixture and kernel-development vehicle -- not "
                 "an assistant, and not a quality benchmark."},
     "tokenizer": {"id": "glm-5.2-siq-fruit", "repository": "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
                   "revision": "ef68013aa6e16453cf52b5b77647f72fbe258c3c", "vocab_size": 154880,
                   "files_sha256": {"tokenizer.json":
                       "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"}},
     "canonical_weights": {"artifact_ref": F_BF16, "precision": "bf16"},
     "cross_refs": lair(),
     "sources": [src("model_card", "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit",
                     None, "geometry, nine-source pretraining recipe, SIQ artifact description"),
                 src("github_file", "https://github.com/malaiwah/proxy-fruit",
                     None, "trainer, exporter, gauntlets")],
     "disclosures": [disc("record_note", "info",
                          "The model, every artifact of it, the panel, the reference and the "
                          "measurement all have the same author. Stated rather than hidden: it "
                          "makes the reference unambiguous, and it means no third party has "
                          "reproduced any of these rows.")]},
]

PANELS += [
    {"schema_version": V, "id": P_FRUIT,
     "name": "Fruit held-out fidelity panel v1 -- 16 windows x 2048",
     "author": MAL("panel-author"), "model_scope": [FRUIT],
     "tokenizer": {"id": "glm-5.2-siq-fruit", "repository": "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
                   "revision": "ef68013aa6e16453cf52b5b77647f72fbe258c3c", "vocab_size": 154880},
     "structure": {"contexts": 16, "context_length": 2048, "positions_per_context": 2047,
                   "positions_per_context_min": 2047, "positions_per_context_max": 2047,
                   "scored_positions_total": 32752,
                   "scoring_window": {"score_from": 0, "windowed": False,
                                      "min_left_context_tokens": 1,
                                      "dropped_positions_total": 0,
                                      "policy": "no window: every causal prediction position "
                                                "of every context is included"},
                   "strata": {"literary": {"contexts": 8}, "scientific": {"contexts": 8}}},
     "identity": {"hash_covers": "token_ids", "panel_token_sha256": FRUIT_TOKEN_SHA,
                  "panel_receipt_sha256": FRUIT_PANEL_RECEIPT_SHA,
                  "manifest_sha256": None, "shard_token_sha256": {}},
     "corpus": {"public": True, "version": "qwen38-kld5-corpus-text/1", "build_tool_ref": None,
                "lineage": "k6/tools/build_token_panel.py over the published corpus tree "
                           "malaiwah/qwen38-27b-fidelity-suite-v5 @ 7797fcce, corpus/text/. "
                           "Strata sorted ascending; within each, documents sorted by file "
                           "name; each tokenized whole with add_special_tokens=False; "
                           "eligible at >= 4096 tokens; window = tokens[2048:4096] because the "
                           "head of a real document is title pages and boilerplate; first 8 "
                           "eligible documents per stratum. No RNG anywhere. "
                           "panel/panel-receipt.json inside the root dataset carries every "
                           "source document's sha256 and the exact token slice.",
                "license_note": "literary = Project Gutenberg text, public domain in the US; "
                                "scientific = arXiv titles and abstracts under the arXiv API "
                                "terms of use. The panel redistributes token ids, not text.",
                "sources": [src("dataset_card",
                                "https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5")]},
     "contamination": {"checked": False, "hits": None, "benchmarks_scanned": [],
                       "method": "source-level separation only. Fruit's published pretraining "
                                 "recipe names nine sources: FineWeb-Edu, English and Chinese "
                                 "Wikipedia, TinyStories, two GLM-5.2 distillation corpora, "
                                 "REAP calibration text, SPDX licence text, and code. This "
                                 "panel draws only from the literary and scientific strata, "
                                 "neither of which appears in that list.",
                       "receipt": None},
     "sealed": True,
     "availability": {"status": "public", "uri": FRUIT_ROOT_DS},
     "derived_from": None, "derivation": None, "cross_refs": lair(),
     "sources": [src("dataset_card", FRUIT_ROOT_DS, None,
                     "the panel ships inside the root fidelity dataset: panel/panel.json, "
                     "panel/tokens/, panel/masks/, and the byte-verbatim build receipt "
                     "panel/panel-receipt.json")],
     "disclosures": [
         disc("weak_contamination_guard", "caveat",
              "Separation from Fruit's training data is asserted at SOURCE level only: the two "
              "strata used are not among the nine sources Fruit's card names. No shingle or "
              "n-gram scan against the published pretraining shards was run, so incidental "
              "overlap through a web-crawl source such as FineWeb-Edu is not excluded."),
         disc("small_panel", "caveat",
              "16 windows / 32,752 scored positions. On the one artifact measured here so far "
              "the per-window standard deviation is 0.0283 nats around a mean of 0.0387, a "
              "standard error near 0.0071. Numbers on this panel cannot separate artifacts "
              "that differ by less than roughly 30 percent.", True)]},
]

_FRUIT_NATIVE = [
    asg("embed_tokens", "native", "bf16", 16), asg("attn.qkv", "native", "bf16", 16),
    asg("attn.o", "native", "bf16", 16), asg("attn.other", "native", "bf16", 16),
    asg("mlp.gate", "native", "bf16", 16), asg("mlp.up", "native", "bf16", 16),
    asg("mlp.down", "native", "bf16", 16), asg("moe.router", "native", "bf16", 16),
    asg("moe.shared_expert", "native", "bf16", 16), asg("moe.experts", "native", "bf16", 16),
    asg("mtp", "native", "bf16", 16), asg("norm", "native", "bf16", 16),
    asg("lm_head", "native", "bf16", 16),
]

_FRUIT_SIQ_SCOPE = scope("mixed", [
    asg("embed_tokens", "native", "bf16", 16),
    asg("attn.qkv", "native", "bf16", 16, "0-13"),
    asg("attn.o", "native", "bf16", 16, "0-13"),
    asg("attn.other", "native", "bf16", 16, "0-13",
        note="the DSA lightning-indexer tensors"),
    asg("mlp.gate", "native", "bf16", 16, "0-2", note="the three dense MLP layers"),
    asg("mlp.up", "native", "bf16", 16, "0-2"),
    asg("mlp.down", "native", "bf16", 16, "0-2"),
    asg("moe.router", "native", "bf16", 16, "3-13"),
    asg("moe.shared_expert", "native", "bf16", 16, "3-13"),
    asg("moe.experts", "quantized", "exl3-trellis", 3.375, "3-12",
        note="THE ONLY CHANGED CLASS in the ten sparse layers. 96 experts at K4 and 160 at K3 "
             "per layer, from the artifact's own tier_bitmap.json: (96*4 + 160*3)/256 = 3.375. "
             "Stored as .rank0.{trellis,suh,svh,mcg} atoms."),
    asg("mtp", "quantized", "exl3-trellis", 3, "13",
        note="the MTP draft layer's 256 experts, uniform K3"),
    asg("norm", "native", "bf16", 16),
    asg("lm_head", "native", "bf16", 16,
        note="bitwise identical to the reference head: both exports write it through the same "
             "unconditional bf16 path"),
], "native", kv="bf16", mtp=True)

ARTIFACTS += [
    artifact(F_BF16, FRUIT, "GLM-5.2-SIQ-Fruit BF16 (the reference export)", "base",
             hf("malaiwah/GLM-5.2-SIQ-Fruit-bf16", "ef68013aa6e16453cf52b5b77647f72fbe258c3c"),
             "safetensors", "BF16", 10102776813,
             codec("bf16", None),
             scope("none", _FRUIT_NATIVE, "native", kv="bf16", mtp=True),
             MAL("model-publisher"),
             [src("model_card", "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16"),
              src("github_file",
                  "https://github.com/malaiwah/proxy-fruit/blob/%s/export_fruit.py" % PROXY_FRUIT_PIN,
                  None,
                  "pinned to the commit so the line numbers cannot drift: the unconditional "
                  "bf16() helper at 262-266 (`sd[key].to(torch.bfloat16)`), the FRUIT_BF16=1 "
                  "routed-expert branch at 317-333 whose `continue` on 333 skips the trellis "
                  "encoder that resumes on 334, and the config path at 373-378 where "
                  "quantization_config is popped ONLY in BF16 mode")],
             # PROC-01. This is a MECHANISM claim -- "a direct cast, not a
             # dequantization" -- and it decides this artifact's reference_kind,
             # which decides whether a KL number measured against it means what it
             # says. It was published as prose with nothing attached, and the
             # validator had nothing to object to, because only metric rows have
             # ever needed a receipt. Now it carries the two line-anchored,
             # commit-pinned citations it was always resting on.
             [disc("record_note", "info",
                   "Every tensor is bf16 and comes from the trained checkpoint by a direct "
                   "cast: export_fruit.py FRUIT_BF16=1 reads the annealed state dict and "
                   "writes .to(torch.bfloat16) bits, and its `continue` skips the SIQ encoder "
                   "entirely. No dequantization step exists anywhere in the exporter, so this "
                   "is reference_kind native_bf16 and NOT dequantized_from_quant. The "
                   "underlying checkpoint did go through a 500-step QNOISE quantization-aware "
                   "anneal before export; that is a property of the model, not of these bytes.",
                   provenance=True,
                   sources=[
                       src("github_file",
                           "https://github.com/malaiwah/proxy-fruit/blob/%s/export_fruit.py"
                           % PROXY_FRUIT_PIN, None,
                           "the unconditional bf16() helper: `sd[key].to(torch.bfloat16)`. "
                           "There is no dequantize path in the file for it to be an "
                           "alternative to.", lines="262-266"),
                       src("github_file",
                           "https://github.com/malaiwah/proxy-fruit/blob/%s/export_fruit.py"
                           % PROXY_FRUIT_PIN, None,
                           "the FRUIT_BF16=1 routed-expert branch, whose `continue` on 333 "
                           "skips the trellis encoder that resumes on 334", lines="317-333"),
                   ])],
             availability={"status": "public",
                           "uri": "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16"},
             derived_from_artifact_ref=None, cross_refs=lair(),
             weights_extra={"size_basis": "repo_all_files",
                            "index_sha256": "86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56",
                            "config_sha256": "5a19697e555fff140d1b089b852c3ef227114b196f8d76796560feeeb34dc44a"}),
    artifact(F_SIQ, FRUIT, "GLM-5.2-SIQ-Fruit (exl3-trellis K3/K4 routed experts)", "quant",
             hf("malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5"),
             "safetensors", "EXL3 K3/K4 experts, BF16 elsewhere", 3125527019,
             codec("exl3-trellis", 3.375,
                   tool="proxy-fruit export_fruit.py (exl3-trellis encoder)"),
             _FRUIT_SIQ_SCOPE, MAL("quantizer"),
             [src("model_card", "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit"),
              src("hf_file",
                  "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit/blob/"
                  "c1798e3676fa16b4a874381171adab1e3033fbd5/tier_bitmap.json",
                  None, "per-expert K allocation and the encoder's own expert_rel_rt_mse; "
                        "keep_nvfp4 is an empty list for every layer"),
              src("github_file",
                  "https://github.com/malaiwah/proxy-fruit/blob/%s/export_fruit.py" % PROXY_FRUIT_PIN,
                  None,
                  "lines 373-378: `src_cfg = json.loads((SRC/'config.json').read_text())`, then "
                  "`src_cfg.pop('quantization_config', None)` ONLY when FRUIT_BF16=1, then "
                  "`cfg = dict(src_cfg)`. The non-BF16 path never pops it, which is how the "
                  "parent GLM-5.2 NVFP4/modelopt block reaches this artifact unchanged."),
              src("github_file",
                  "https://github.com/malaiwah/proxy-fruit/blob/%s/EXLLAMAV3_SIQ_REVIEW.md" % PROXY_FRUIT_PIN,
                  None,
                  "the producer's own review reaches the same conclusion independently: line "
                  "210 `quant_method=modelopt / quant_algo=NVFP4 does not describe the actual "
                  "routed-expert storage`, line 230 `No experts remain NVFP4: "
                  "nvfp4_keep_per_layer = 0`")],
             # PROC-01. Two provenance claims in one paragraph -- the stored bytes
             # are not what config.json declares, and the declaration is INHERITED
             # rather than authored -- and the second one is the load-bearing half:
             # it is the difference between "the producer mislabelled this" and
             # "a field was copied forward". Both are cited, both by commit.
             [disc("declared_scheme_mismatch", "caveat",
                   "config.json declares quantization_config quant_method=modelopt, "
                   "quant_algo=NVFP4, group_size 16, W4A4, producer "
                   "b300-exl3-modelopt-dispatch-shim. That block does not describe the stored "
                   "bytes: zero tensors are NVFP4 (tier_bitmap.json's keep_nvfp4 is empty for "
                   "every layer) and the routed experts are exl3-trellis K3/K4. The exporter "
                   "copies the block from the parent GLM-5.2 config rather than authoring it "
                   "(export_fruit.py 373-378 pops quantization_config ONLY under FRUIT_BF16, "
                   "then does cfg = dict(src_cfg)) -- which is why its ignore list still names "
                   "model.layers.78.eh_proj, a layer this 13-layer model does not have. The "
                   "scope on this row describes the bytes.",
                   provenance=True,
                   sources=[
                       src("github_file",
                           "https://github.com/malaiwah/proxy-fruit/blob/%s/export_fruit.py"
                           % PROXY_FRUIT_PIN, None,
                           "`src_cfg = json.loads((SRC/'config.json').read_text())`, then "
                           "`src_cfg.pop('quantization_config', None)` ONLY when FRUIT_BF16=1, "
                           "then `cfg = dict(src_cfg)`. The non-BF16 path never pops it: that "
                           "is the inheritance.", lines="373-378"),
                       src("hf_file",
                           "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit/blob/"
                           "c1798e3676fa16b4a874381171adab1e3033fbd5/tier_bitmap.json",
                           None,
                           "keep_nvfp4 is an empty list for every layer: zero tensors are "
                           "NVFP4, which is the claim about the stored bytes"),
                       src("github_file",
                           "https://github.com/malaiwah/proxy-fruit/blob/%s/"
                           "EXLLAMAV3_SIQ_REVIEW.md" % PROXY_FRUIT_PIN, None,
                           "the producer's own review, independently: `quant_method=modelopt "
                           "/ quant_algo=NVFP4 does not describe the actual routed-expert "
                           "storage` (210) and `No experts remain NVFP4: "
                           "nvfp4_keep_per_layer = 0` (230)", lines="210-230"),
                   ]),
              disc("unreadable_by_stock_loader", "caveat",
                   "The routed experts are stored as .rank0.{trellis,suh,svh,mcg} atoms. Stock "
                   "transformers 5.16.1 does not fail on them: it reports "
                   "model.layers.{3..12}.mlp.experts.{gate_up,down}_proj as MISSING, randomly "
                   "initialises them (mean about 0, std 0.0199) and returns a running model. "
                   "Any measurement of this artifact through a stock loader without a "
                   "reconstruction step is a measurement of random weights.", True)],
             availability={"status": "public",
                           "uri": "https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit"},
             derived_from_artifact_ref=F_BF16, cross_refs=lair(),
             weights_extra={"size_basis": "repo_all_files",
                            "index_sha256": "5808a4b3e75c4a949a1ede42e6c6fb2576089ec1544038b77de24076e99bf3da",
                            "config_sha256": "7df3d68ab252ffa0bff636d00f82330f56939f2808915eb7d2d209c98e0b9753"}),
]

REFERENCES += [
    {"schema_version": V, "id": R_FRUIT,
     "name": "malaiwah Fruit BF16 hidden-state capture, transformers lane, held-out panel v1",
     "artifact_ref": F_BF16, "panel_ref": P_FRUIT, "reference_kind": "native_bf16",
     "capture": {"stack": "transformers", "stack_version": "5.16.1", "pipeline_ref": PL_FIDDS,
                 "compute_dtype": "bf16", "logits_dtype": "fp32", "kv_cache_dtype": "bf16",
                 "head_source": "shared_head_artifact", "head_sha256": FRUIT_HEAD_SHA,
                 "batch_invariant": None,
                 "capture_receipt_sha256": FRUIT_ROOT_DATASET_SHA},
     "author": MAL("measurer"), "logits_available": True,
     "self_consistency": {"floor_measurement_ref": None,
                          "note": "Reference and candidate are captured by the SAME engine on "
                                  "the SAME lane and compared offline in fp64, so there is no "
                                  "cross-stack floor term to subtract. Measured, not assumed: "
                                  "two cold captures of these weights agree bitwise "
                                  "(capture_content_digest b417acc2...), and comparing them "
                                  "with --force-compute over all 32,752 x 154,880 logits "
                                  "returns exactly 0.0 nats at top-1 agreement 1.0."},
     "sources": [src("dataset_card", FRUIT_ROOT_DS, None,
                     "malaiwah.fidelity-dataset.v1; dataset_sha256 f56674f9..., "
                     "capture_content_digest b417acc2...")],
     "disclosures": [
         disc("shared_reference_head", "info",
              "Hidden states are captured for both sides and ONE head (8d0f7d6e...) is applied "
              "to both. Legitimate here rather than merely convenient: both exports write "
              "lm_head through the same unconditional bf16 path, so the candidate's own head "
              "is the same tensor."),
         disc("architecture_subset_loaded", "caveat",
              "Stock transformers implements glm_moe_dsa natively but drops Fruit's DSA "
              "lightning indexer for layers 3-13 and the whole MTP draft layer 13, loading "
              "4,572,134,656 of 5.04B parameters. The forward pass is dense MLA attention with "
              "no speculative decoding. Identical for the reference and every candidate on "
              "this lane, so the comparison is sound -- but a number here is the stored "
              "weights' error, not the serving stack's.", True)]},
]

PIPELINES += [
    pipeline(PL_FIDDS,
             "malaiwah three-step fidelity dataset (capture / capture / compare), "
             "hf-transformers engine",
             ["capture", "scorer", "aggregator"],
             "https://github.com/malaiwah/quant-fidelity-suite", None,
             "bin/fidelity_dataset.py + k6/tools/hf_capture.py", MAL("toolchain-author"),
             [disc("record_note", "info",
                   "Capture and comparison are separated: each side runs one transformers "
                   "forward per panel window, taps the lm_head input with a forward pre-hook, "
                   "and seals a portable dataset. The comparison then reads two datasets and "
                   "needs neither set of weights. Because both sides are captured by the same "
                   "engine on the same lane, the floor is structurally zero rather than "
                   "subtracted -- verified at exactly 0.0 nats.")],
             numerics=FP64,
             hardware={"gpu": "NVIDIA L4", "gpu_count": 1, "tensor_parallel": 1,
                       "note": "JarvisLabs spot container, region IN2"},
             cost={"usd_per_measurement": None, "basis": None},
             sources=[src("dataset_card", FRUIT_ROOT_DS),
                      src("dataset_card", FRUIT_QUANT_DS)],
             cross_refs=lair()),
]


# ---------------------------------------------------------------------------
# M1: Qwen3.8-27B same-lane root capture (hf-transformers lane, RTX PRO 6000)
#
# The 37 Qwen3.8-27B rows above are scored against a vLLM-captured teacher, so
# each carries an unmeasured cross-stack term and its attributable error is
# inferred by subtraction. The rows below are scored against a teacher captured
# by the SAME engine on the SAME lane as the candidates, so the floor is 0.0 by
# construction -- MEASURED, not assumed -- and a candidate's raw KLD IS its
# attributable error, with nothing subtracted.
#
# They are NOT rankable against the 37 older rows. The comparability key binds
# the reference and the references differ. The panel is deliberately the SAME
# one, so the two groups differ by the LANE ALONE, which makes the difference
# interpretable without making it comparable.
# ---------------------------------------------------------------------------

Q38_HF_REF = "reference--malaiwah.qwen38-bf16-hf.suite-v5-shard0-1m"
Q38_AWQ_CYAN = "artifact--cyankiwi.qwen3.8-27b-awq-int4"
PL_FIDDS_Q38 = "pipeline--malaiwah.fidelity-dataset-hf.rtxpro6000"

Q38_ROOT_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
Q38_FP8_REV = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
Q38_AWQ_REV = "63768c10df38c0395e12ef49edac1bd539eaeeea"
Q38_SUITE_REV = "7797fcce3ffed62b99871348887f4626dc9b2b3b"

Q38_ROOT_CAPTURE_SHA = "2376837de2e42561a196a3f33e25ab6e79471bed0f97c5949605656ca97504c3"
Q38_ROOT_DS_SHA = "8a65836468ee50585d44a387f2a35cfe8a4a6cdb1508c154704b588ca59b280f"
Q38_HEAD_SHA = "d922b751f014ee1139488cb94c0e164a1eb3da5c14048070ed1b784e00f92723"
Q38_PANEL_AGG_SHA = "8847e99a855fab374c3b5fecc1777eabf61307c36782ed48f9fc77e7aca3e42f"
Q38_FP8_CAPTURE_SHA = "cab1aa85c748ca024870bfdeca9841d53f2cc9744c775a862bc59d6bff0e8bb3"
Q38_FP8_DS_SHA = "dbe0a4462275bcb3de1c1cfeb181fe8f07bc6ec52fb3f8f74c130d4144022709"
Q38_AWQ_CAPTURE_SHA = "d22de49aa8bc31e11fcabc1b9217fd3151b9ef0cfad319bc7c9188db6e55261a"
Q38_AWQ_DS_SHA = "59205aebb69dff05e5d84c51941fd9134a46fd68ed3f98cbf1959d141cfb262e"

Q38_ROOT_DS = "https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-root-v1"
Q38_SUITE_DS = "https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v5"

# The computational closure for a number this campaign PRODUCED. Unlike every
# row above, metric.value here was computed by code this repository ships, at a
# commit we can name; the bytes that ran were verified byte-identical to that
# commit's blobs ON the machine that ran them, so these rows carry a RECORDED
# harness covering metric.value rather than the grandfather clause.
#
# The digests are transcribed from that verification rather than re-read from
# today's checkout ON PURPOSE: `k6/tools/hf_capture.py` has since changed on
# main, and re-digesting it now would stamp these rows with the identity of code
# that did not produce them -- exactly the failure harness_id.py exists to
# prevent. Anyone can re-derive them: `git show <pin>:<path> | sha256sum`.
# The paths below keep their 2026-08 spelling on purpose. k6_kld_report.py was
# renamed kld_report.py (and k6_student_capture.py -> student_capture.py) on
# 2026-08-31; `harness_id` is a sha256 over {boundary, [{role, PATH, sha256}],
# tool_versions}, so the path is INSIDE the hash. Rewriting it here would give
# these rows the identity of a file whose bytes are not the ones that ran --
# the exact failure harness_id.py exists to prevent -- and would change
# published harness ids. bin/selftest_naming_sweep.py freezes them.
FIDELITY_COMPARE_PIN = "9133f4288838a3333b25b575f4d5e4e8ab3b419a"
FIDELITY_COMPARE_DIGESTS = [
    {"role": "capture", "path": "k6/tools/hf_capture.py",
     "sha256": "1a64290a445d437145df69bf6172df557af9e21330b97700c6596efba6947a9b"},
    {"role": "comparator", "path": "bin/fidelity/dscompare.py",
     "sha256": "77544ea17d63c324b67066fc2b750176bee5ac777c323a0b53fe69af043d5a33"},
    {"role": "estimator", "path": "k6/tools/k6_kld_report.py",
     "sha256": "7c2fa04808d595948489a2846e40eb76828f31495767014a426046c4d6501871"},
    {"role": "format", "path": "bin/fidelity/dsformat.py",
     "sha256": "269bee32eb6d786ee0068f57185f08a529fabd49d1c2be652bf29919568fb023"},
    {"role": "front_end", "path": "bin/fidelity_dataset.py",
     "sha256": "7966954ef2774257b1575e22ec34c110cb5530cba813683584c32e73cbd68303"},
    {"role": "manifest", "path": "bin/fidelity/dsmanifest.py",
     "sha256": "8281a18a630e222ae0f96c6e612cd605bff02b82a3464a253e07381fce75153c"},
]
FIDELITY_COMPARE_TOOL_VERSIONS = {
    "python": "3.10.20", "torch": "2.11.0+cu130", "transformers": "5.8.1",
    "numpy": "2.2.6", "safetensors": "0.8.0-rc.0",
}


# The streaming lane's closure, for the FIRST row this registry ever recorded a
# harness for rather than grandfathering. The pin is not guessed: the receipt's
# own `produced_by.entrypoint_sha256` for bin/measure_cloud.py is
# 2d4ccd44f80b3ad9..., and `git show f3b6d823:bin/measure_cloud.py` hashes to
# exactly that -- so the commit whose bytes were uploaded is identified by the
# receipt itself, not by when someone thinks the run happened.
STREAM_LANE_PIN = "f3b6d8234a1ccb9b3fa461f28ebdb9b043c2ae3a"
STREAM_LANE_DIGESTS = [
    {"role": "capture", "path": "k6/tools/stream_score.py",
     "sha256": "f313ba248cd522016444123aa7353372105848a1b3b73f254c8c485c85bf5294"},
    {"role": "estimator", "path": "k6/tools/k6_kld_report.py",
     "sha256": "27c1c4c75ea2136b874b62dfa47c4799539e09029b640ab027a8d81b00c6bb5a"},
    {"role": "front_end", "path": "bin/invoke_engine.py",
     "sha256": "04917772e2e83b3f059160957ee3139f9190ae085a131918a18476c10a7129f5"},
    {"role": "comparator", "path": "bin/invoke_scorer.py",
     "sha256": "77aa5d160857db51411138776748433b74532aeabfd0c2feaf8b6f5ab4ede566"},
    {"role": "format", "path": "bin/seal_receipt.py",
     "sha256": "88a28532a85b0a80ee38b45250a3a8cc701620ff9c049bf010f975f27a988812"},
]
# Read off the measuring instance's own receipts (python-version.txt,
# wheel-versions.txt), not off whoever runs `make reseed`.
STREAM_LANE_TOOL_VERSIONS = {
    "python": "3.12.13", "torch": "2.11.0+cu130", "transformers": "5.16.1",
    "numpy": "2.5.2", "safetensors": "0.8.0",
}


def stream_lane_harness():
    """RECORDED harness for a streaming-lane row this campaign computed."""
    return {
        "harness_id": H.compute_id(STREAM_LANE_DIGESTS, STREAM_LANE_TOOL_VERSIONS),
        "recorded": True,
        "boundary": H.BOUNDARY,
        "covers": ["auxiliary_metrics", "determinism", "metric.value"],
        "repository": {"url": HARNESS_REPOSITORY["url"],
                       "commit": STREAM_LANE_PIN, "commit_role": "exact",
                       "dirty": False},
        "code_digests": STREAM_LANE_DIGESTS,
        "tool_versions": dict(sorted(STREAM_LANE_TOOL_VERSIONS.items())),
        "note": ("Covers metric.value: capture and estimation both ran from this "
                 "commit's bytes on the measuring instance. The commit is "
                 "identified BY THE RECEIPT -- its produced_by.entrypoint_sha256 "
                 "matches `git show %s:bin/measure_cloud.py` -- rather than by "
                 "recollection. Re-derive any row with "
                 "`git show %s:<path> | sha256sum`."
                 % (STREAM_LANE_PIN[:12], STREAM_LANE_PIN[:12])),
    }


def q38_hf_harness():
    """RECORDED harness for the rows this campaign computed end to end."""
    return {
        "harness_id": H.compute_id(FIDELITY_COMPARE_DIGESTS,
                                   FIDELITY_COMPARE_TOOL_VERSIONS),
        "recorded": True,
        "boundary": H.BOUNDARY,
        "covers": ["auxiliary_metrics", "determinism", "metric.value"],
        "repository": {"url": "https://github.com/malaiwah/quant-fidelity-suite",
                       "commit": FIDELITY_COMPARE_PIN, "commit_role": "exact",
                       "dirty": False},
        "code_digests": FIDELITY_COMPARE_DIGESTS,
        "tool_versions": dict(sorted(FIDELITY_COMPARE_TOOL_VERSIONS.items())),
        "note": ("Covers metric.value: capture and comparison both ran from this "
                 "commit's bytes, verified byte-identical on the measuring machine "
                 "before the run. Digests are transcribed from that verification, "
                 "not re-read from a later checkout, because hf_capture.py has since "
                 "changed on main and re-digesting would name code that did not "
                 "produce these numbers. Re-derive with "
                 "`git show %s:<path> | sha256sum`." % FIDELITY_COMPARE_PIN[:12]),
    }


Q38_HF_LANE_DISC = disc(
    "record_note", "info",
    "LANE IDENTITY. transformers reported the fused linear-attention path "
    "unavailable (flash-linear-attention / causal-conv1d absent) and fell back to "
    "the reference torch implementation, identically for the reference and every "
    "candidate. That fallback is part of what these digests mean; installing those "
    "kernels is a different lane and is not guaranteed to reproduce them. The "
    "`kernels` package (0.12.3) WAS installed partway through, between the root "
    "capture and the candidates; a control re-captured 8 root windows afterwards "
    "and reproduced the sealed root's per-record tensor_content_sha256 exactly "
    "(0 mismatches of 8), so the lane did not move.")

Q38_NOT_RANKABLE_DISC = disc(
    "record_note", "info",
    "NOT RANKABLE AGAINST THE 37 OLDER Qwen3.8-27B ROWS. Those are scored against "
    "reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m, a vLLM capture; this "
    "row is scored against a transformers capture on the local-cuda-budget lane. "
    "The comparability key binds the reference, so the two groups are different "
    "keys. The panel is deliberately identical -- same 512 contexts, same "
    "panel_token_sha256 caef8a46... -- so the two groups differ by the LANE ALONE, "
    "which makes the difference interpretable but still not comparable. This is "
    "recorded as INFORMATION, not as a comparability caveat: within its own key the "
    "row is strict, and the separation from the older group is already carried "
    "structurally by the differing key rather than by this note.")

Q38_HYBRID_SCOPE_DISC = disc(
    "record_note", "info",
    "HYBRID ARCHITECTURE, COARSE VOCABULARY. Qwen3.8-27B is "
    "Qwen3_5ForConditionalGeneration: of 64 text layers, 48 use linear attention "
    "and 16 full attention (full_attention_interval 4), plus a vision tower and an "
    "MTP block, and it is DENSE -- the checkpoint contains zero expert tensors. "
    "Two consequences for scope. (a) The producers quantize each layer's main "
    "projection but leave the linear-attention families (conv1d, in_proj_a/b) in "
    "bf16; the registry tensor_class vocabulary cannot express that split, so "
    "attn.qkv/attn.o read 'quantized' with the per-class tensor counts recorded in "
    "the dataset's scope.derivation. (b) The root's scope comes from "
    "native_scope(), which emits moe.experts=native:bf16@16 unconditionally -- for "
    "this model that is a vocabulary placeholder, NOT a claim that experts exist.")


ARTIFACTS += [
    artifact(Q38_AWQ_CYAN, QWN, "cyankiwi Qwen3.8-27B AWQ-INT4", "quant",
             hf("cyankiwi/Qwen3.8-27B-AWQ-INT4", Q38_AWQ_REV, "hf_api"),
             "safetensors", "INT4", 21041255795,
             codec("awq", 4.0, None, group_size=32),
             scope("uniform", [
                 asg("embed_tokens", "native", "bf16"),
                 asg("attn.qkv", "quantized", "int4", 4.0),
                 asg("attn.o", "quantized", "int4", 4.0),
                 asg("mlp.gate", "quantized", "int4", 4.0),
                 asg("mlp.up", "quantized", "int4", 4.0),
                 asg("mlp.down", "quantized", "int4", 4.0),
                 asg("norm", "native", "bf16"), asg("lm_head", "native", "bf16"),
             ], "native", kv="bf16"),
             attr("cyankiwi", "quantizer", handle="cyankiwi",
                  url="https://huggingface.co/cyankiwi"),
             [src("model_card", "https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4", None,
                  "revision 63768c10..., index_sha256 82b1bf79..., 2,396 index entries; "
                  "scope derived from the artifact's own quantization_config + weight index "
                  "by k6/tools/derive_scope.py")],
             [disc("record_note", "info",
                   "Scope derived from the artifact's OWN config and weight index, not guessed. "
                   "quantization_config declares compressed-tensors pack-quantized, num_bits=4, "
                   "group_size=32, observer=mse, and a 313-entry `ignore` list. Verified against "
                   "the real tensor names: every mlp/attention projection carries pack-quantized "
                   "state, while the linear-attention families (conv1d, out_proj, in_proj_a/b) "
                   "and the ENTIRE mtp block stay bf16. The producer labels the repo AWQ; the "
                   "stored container is compressed-tensors pack-quantized, so `format` is the "
                   "registry's numeric `int4` and codec.family is `awq`."),
              Q38_HYBRID_SCOPE_DISC],
             weights_extra={"size_basis": "repo_all_files",
                            "index_sha256": "82b1bf79f5b61333e83da17ec3bf89c9f178e29395a14c6b3ce3bbc474e1ead8"},
             derived_from_artifact_ref=Q_BF16,
             availability={"status": "public",
                           "uri": "https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4"},
             cross_refs=lair(), seal={"sealed": False}),
]

REFERENCES += [
    {"schema_version": V, "id": Q38_HF_REF,
     "name": "malaiwah Qwen3.8-27B BF16 hidden-state capture, transformers lane, "
             "suite-v5 shard-0 panel",
     "artifact_ref": Q_BF16, "panel_ref": P_Q1M, "reference_kind": "native_bf16",
     "capture": {"stack": "transformers", "stack_version": "5.8.1",
                 "pipeline_ref": PL_FIDDS_Q38,
                 "compute_dtype": "bf16", "logits_dtype": "fp32", "kv_cache_dtype": "bf16",
                 "head_source": "shared_head_artifact", "head_sha256": Q38_HEAD_SHA,
                 "batch_invariant": None,
                 "capture_receipt_sha256": Q38_ROOT_DS_SHA},
     "author": MAL("measurer"), "logits_available": True,
     "self_consistency": {
         "floor_measurement_ref": "measurement--qwen38-hf.bf16-selfcompare-floor.suite-v5-shard0-1m",
         "note": "Reference and candidates are captured by the SAME engine on the SAME lane and "
                 "compared offline in fp64, so there is no cross-stack floor term to subtract. "
                 "Measured, not assumed: THREE cold captures of these weights in three separate "
                 "processes agree bitwise (capture_content_digest 2376837d...), and comparing two "
                 "of them with --force-compute over all 1,048,064 x 248,320 logits returns exactly "
                 "0.0 nats at top-1 agreement 1.0, with the forced computation reproducing the "
                 "hash proof's tokenwise-kld digest byte for byte."},
     "sources": [src("dataset_card", Q38_ROOT_DS, None,
                     "malaiwah.fidelity-dataset.v1; dataset_sha256 8a658364..., "
                     "capture_content_digest 2376837d..., model revision 1d4bf0f2...")],
     "disclosures": [
         disc("shared_reference_head", "info",
              "Hidden states are captured for both sides and ONE head (d922b751...) is applied to "
              "both. Legitimate rather than merely convenient here: every candidate measured on "
              "this lane leaves lm_head native bf16 by its own config, so the candidate's own head "
              "is the same tensor."),
         Q38_HF_LANE_DISC,
         Q38_NOT_RANKABLE_DISC,
         disc("architecture_subset_loaded", "info",
              "The checkpoint is multimodal (Qwen3_5ForConditionalGeneration, 333 vision tensors) "
              "and the panel is text-only: no image or video token is ever fed, so the vision "
              "tower does not participate in any scored forward pass. transformers loaded all "
              "1,199 tensors with 0 missing, 0 unexpected and 0 mismatched.")]},
]

PIPELINES += [
    pipeline(PL_FIDDS_Q38,
             "malaiwah three-step fidelity dataset (capture / capture / compare), "
             "hf-transformers engine, RTX PRO 6000",
             ["capture", "scorer", "aggregator"],
             "https://github.com/malaiwah/quant-fidelity-suite", FIDELITY_COMPARE_PIN,
             "bin/fidelity_dataset.py + k6/tools/hf_capture.py", MAL("toolchain-author"),
             [disc("record_note", "info",
                   "Same toolchain as pipeline--malaiwah.fidelity-dataset-hf, different hardware, "
                   "recorded separately so the hardware block stays factual. Capture and "
                   "comparison are separated: each side runs one transformers forward per panel "
                   "window, taps the lm_head input with a forward pre-hook, and seals a portable "
                   "dataset; the comparison then reads two datasets and needs neither set of "
                   "weights. Because both sides are captured by the same engine on the same lane, "
                   "the floor is structurally zero rather than subtracted -- verified here at "
                   "exactly 0.0 nats with the estimator actually executed (--force-compute), not "
                   "answered by the hash short-circuit."),
              disc("record_note", "info",
                   "COST SHAPE, measured on this run and recorded for whoever plans the next one: "
                   "a 512-window capture took 335 s (0.0109 min/window) while ONE 512-window "
                   "comparison took 60 min 19 s (0.118 min/window) -- the comparison costs ~10.8x "
                   "the capture it consumes. dscompare._replay applies the head with a numpy "
                   "matmul on the CPU and only the fp64 KLD reduction runs on the GPU, so a "
                   "comparison pins ~20 cores while the GPU sits at 0%.")],
             numerics=FP64,
             hardware={"gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_count": 1,
                       "tensor_parallel": 1,
                       "note": "JarvisLabs on-demand container, region IN1, 96 GB VRAM, "
                               "driver 595.58.03, 28 vCPU"},
             cost={"usd_per_measurement": None,
                   "basis": "one box served the root (3 cold runs), 2 candidates, 3 comparisons "
                            "and the publish"},
             sources=[src("dataset_card", Q38_ROOT_DS)],
             cross_refs=lair()),
]


def build_measurements_fruit(artifacts_map):
    M = lambda *a, **k: measurement(*a, artifacts_map=artifacts_map, **k)
    GH = "https://github.com/malaiwah/quant-fidelity-suite/blob/main/registry/protocol/fruit/"
    ds_sources = [
        src("dataset_card", FRUIT_ROOT_DS, None,
            "reference capture: dataset_sha256 f56674f9..., capture_content_digest b417acc2..."),
        src("dataset_card", FRUIT_QUANT_DS, None,
            "candidate capture: dataset_sha256 13577688..., capture_content_digest 8875fe45..."),
    ]
    return [
        M("measurement--fruit.bf16-selfcompare-floor.heldout-v1", FRUIT, F_BF16, P_FRUIT,
          R_FRUIT, PL_FIDDS, 0.0,
          head_policy="shared_reference_head", top1=1.0,
          scored_positions=32752, contexts=16,
          runs=2, cold=True, identical=True, evidence_kind="hidden_state_tensor_sha256",
          evidence_hashes=[FRUIT_ROOT_CAPTURE_SHA],
          det_note="Two cold captures of the same bf16 weights in two separate processes on "
                   "one L4 produced the same capture_content_digest. Their dataset_sha256 "
                   "values differ because a manifest embeds timestamps and a cold-run label, "
                   "which is exactly why determinism evidence is taken over tensor CONTENT.",
          sources=ds_sources + [
              src("github_file", GH + "comparison.fruit-bf16-selfcompare-floor.heldout-v1.json",
                  "cf1f2271c68a3c5c497ba6ad9612d8f2ddcd3cda3d7ebbb80cb8e044db04c104",
                  "malaiwah.fidelity-comparison-receipt.v1 for the self-compare of the two "
                  "cold root captures; both sides carry capture_content_digest b417acc2...")],
          disclosures=[
              disc("record_note", "info",
                   "THE FLOOR, measured rather than assumed. `fidelity-dataset compare "
                   "--self-compare --force-compute` over all 32,752 x 154,880 logits in fp64 "
                   "returns mean tokenwise KLD exactly 0.0 nats at top-1 agreement 1.0. This "
                   "is the architectural payoff of separating capture from comparison: when "
                   "both sides are captured by one engine on one lane, comparison overhead is "
                   "structurally zero and never has to be subtracted from anything."),
              disc("shared_reference_head", "info",
                   "One head (8d0f7d6e...) applied to both sides' hidden states.")]),
        M("measurement--fruit.siq-exl3-k3k4.heldout-v1", FRUIT, F_SIQ, P_FRUIT, R_FRUIT,
          PL_FIDDS, 0.03873745371351417,
          head_policy="shared_reference_head", top1=0.8797630679042501,
          aux={"median_kld": 0.01676410508811121, "p95_kld": 0.14771257394235618,
               "p99_kld": 0.3174928513324092, "p999_kld": 0.756304641037244,
               "max_kld": 1.4961663314355835,
               "context_macro_mean_kld": 0.038737453713514176,
               "strata": {"literary": 0.027501507795953492,
                          "scientific": 0.049973399631074854}},
          notes="Per-window mean 0.038737453713514176, population sd 0.028308679654341876, "
                "min 0.012369540015856577 (final-0006, literary), max 0.09151472952402755 "
                "(final-0009, scientific) over 16 windows. The macro mean over contexts equals "
                "the token mean because every window contributes the same 2,047 positions.",
          scored_positions=32752, contexts=16,
          runs=1, cold=True, evidence_kind="hidden_state_tensor_sha256",
          evidence_hashes=[FRUIT_QUANT_CAPTURE_SHA],
          det_note="One cold capture of the candidate. The reference side of this comparison "
                   "is the same two-run-verified capture the floor row uses.",
          cls="advisory",
          sources=ds_sources + [
              src("github_file", GH + "comparison.fruit-siq-exl3-k3k4.heldout-v1.json",
                  "a128a272936eb27fa65057ec0ed04f904f1fc9fac4779e3d8f86b771e89a73a2",
                  "malaiwah.fidelity-comparison-receipt.v1; receipt_sha256 04e052be..., "
                  "per-context and per-domain breakdowns, tokenwise-kld digest 3b381147..."),
              src("github_file", GH + "exl3-reconstruction.fruit-siq.json",
                  "e317be43ef2b7ae8b77142b40982393d490879892ebb7296dac8f829c4f4930f",
                  "malaiwah.exl3-reconstruction-receipt.v1: per-module rel-L2 and cosine for "
                  "all 8,448 decoded expert matrices, and the cross-check against the "
                  "producer's own tier_bitmap expert_rel_rt_mse")],
          disclosures=[
              disc("record_note", "info",
                   "Comparison receipt malaiwah.fidelity-comparison-receipt.v1 "
                   "04e052be94c3e39a2da4c4d9ebbfd18722c728fdd532983c4fa4d2c3e5459317, "
                   "reference dataset_sha256 f56674f9..., candidate dataset_sha256 13577688..."),
              disc("lossy_capture_codec", "caveat",
                   "RECONSTRUCTED, NOT EXECUTED. The artifact's routed experts are exl3-trellis "
                   "atoms that stock transformers cannot read, so the candidate capture ran a "
                   "bf16 reconstruction of them (k6/tools/materialize_exl3_experts.py) rather "
                   "than the vendor kernel. This is the dequantize-and-run methodology the "
                   "GGUF/MLX/EXL3 ecosystems use for KLD: it measures the error of the STORED "
                   "WEIGHTS and isolates it from kernel error. It does not measure Fruit's "
                   "production path (b12x/SparkInfer + vLLM, fp8/nvfp4 KV, MTP). Decode "
                   "evidence: the codebook table is bitwise equal to the campaign's "
                   "independently frozen mcg table on all 65,536 entries; the bit rate read "
                   "off every one of 8,448 payloads agrees with the producer's tier_bitmap; "
                   "and the reconstruction error reproduces the ENCODER's own recorded "
                   "expert_rel_rt_mse with ratio mean 1.00013 over range 0.98902-1.01337. The "
                   "decode has NOT been proven bitwise against a running exllamav3 kernel, "
                   "which is why this row is advisory.", True),
              disc("small_panel", "caveat",
                   "Per-window standard deviation 0.0283 around a mean of 0.0387 over 16 "
                   "windows, i.e. a standard error near 0.0071. Do not rank this against "
                   "anything it differs from by less than roughly 30 percent. The two strata "
                   "differ by nearly 2x on their own (literary 0.0275, scientific 0.0500).",
                   True),
              disc("single_run", "caveat",
                   "One cold capture of the candidate. Repeatability was established for the "
                   "reference side only."),
              disc("shared_reference_head", "info",
                   "One head (8d0f7d6e...) applied to both sides' hidden states. Not a "
                   "substitution: both exports write lm_head through the same bf16 path, so "
                   "the candidate's own head is the same tensor."),
              disc("declared_scheme_mismatch", "caveat",
                   "The artifact's config.json declares NVFP4/modelopt; the stored bytes are "
                   "exl3-trellis K3/K4. scope_digest describes the bytes.")]),
    ]


def build_measurements_qwen38_hf(artifacts_map):
    """The same-lane Qwen3.8-27B rows: a MEASURED 0.0 floor and two candidates."""
    M = lambda *a, **k: measurement(*a, artifacts_map=artifacts_map, **k)
    GH = ("https://github.com/malaiwah/quant-fidelity-suite/blob/main/"
          "registry/protocol/qwen38-hf/")
    ds_root = src("dataset_card", Q38_ROOT_DS, None,
                  "reference capture: dataset_sha256 8a658364..., "
                  "capture_content_digest 2376837d...")
    est = dict(accumulation="float64", head_policy="shared_reference_head",
               vocab_chunk=24832, two_pass=True, stack_relation="same_stack")

    rows = [
        M("measurement--qwen38-hf.bf16-selfcompare-floor.suite-v5-shard0-1m",
          QWN, Q_BF16, P_Q1M, Q38_HF_REF, PL_FIDDS_Q38, 0.0,
          top1=1.0, scored_positions=1048064, contexts=512,
          runs=3, cold=True, identical=True,
          evidence_kind="hidden_state_tensor_sha256",
          evidence_hashes=[Q38_ROOT_CAPTURE_SHA],
          det_note="THREE cold captures of the same bf16 weights, in three separate "
                   "processes on one RTX PRO 6000, produced the same "
                   "capture_content_digest 2376837d... Their dataset_sha256 values "
                   "differ because a manifest embeds timestamps and a cold-run label, "
                   "which is exactly why determinism evidence is taken over tensor "
                   "CONTENT. The third ran concurrently with a CPU-saturated "
                   "comparison and still matched, so host load does not perturb the "
                   "arithmetic.",
          sources=[ds_root,
                   src("github_file", GH + "comparison.qwen38-bf16-selfcompare-floor.json",
                       "b2436077ac6b94b2814657749ef957e6f3087de453f72fd742f358812b99063b",
                       "malaiwah.fidelity-comparison-receipt.v1 for the --force-compute "
                       "self-compare of two cold root captures")],
          disclosures=[
              disc("record_note", "info",
                   "THE FLOOR, MEASURED. `fidelity-dataset compare --self-compare "
                   "--force-compute` over all 1,048,064 x 248,320 logits in fp64 returns "
                   "mean tokenwise KLD exactly 0.0 nats at top-1 agreement 1.0, with every "
                   "percentile (median, p95, p99, p99.9, max) also 0.0. Run twice: once "
                   "answered by the capture-digest short-circuit and once with the "
                   "estimator forced to execute (backend "
                   "torch:k6_kld_report._token_kld). Both produced the same "
                   "tokenwise-kld.npy digest 8be5dcca..., so the forced computation "
                   "reproduces the hash proof byte for byte rather than merely agreeing "
                   "with it. This is the architectural payoff of separating capture from "
                   "comparison: when both sides are captured by one engine on one lane, "
                   "comparison overhead is structurally zero and never has to be "
                   "subtracted from anything. Every candidate row on this reference "
                   "therefore reports attributable error EQUAL to its raw KLD."),
              disc("shared_reference_head", "info",
                   "One head (d922b751...) applied to both sides' hidden states."),
              disc("reduced_run_count", "info",
                   "THREE cold captures, not the campaign's usual five. Three was chosen "
                   "because the evidence here is a CONTENT digest rather than a spread over "
                   "run means: all three processes produced the identical "
                   "capture_content_digest, so a fourth and fifth would restate a bitwise "
                   "identity rather than tighten an estimate. The third was deliberately run "
                   "under a saturated CPU to test whether host load perturbs the arithmetic; "
                   "it did not."),
              Q38_HF_LANE_DISC, Q38_NOT_RANKABLE_DISC, Q38_HYBRID_SCOPE_DISC],
          **est),

        M("measurement--qwen38-hf.fp8-dequantized.suite-v5-shard0-1m",
          QWN, Q_FP8, P_Q1M, Q38_HF_REF, PL_FIDDS_Q38, 0.002989850396847924,
          top1=0.977509961223742,
          aux={"median_kld": 0.0009696961924583243,
               "p95_kld": 0.009972340458292886,
               "p99_kld": 0.032763897796112766,
               "p999_kld": 0.14369528668542408,
               "max_kld": 2.0162679295433397},
          scored_positions=1048064, contexts=512,
          runs=1, cold=True, evidence_kind="hidden_state_tensor_sha256",
          evidence_hashes=[Q38_FP8_CAPTURE_SHA],
          det_note="One cold capture of the candidate. The reference side is the same "
                   "three-run-verified capture the floor row uses.",
          cls="advisory",
          sources=[ds_root,
                   src("model_card", "https://huggingface.co/Qwen/Qwen3.8-27B-FP8", None,
                       "revision 017b9c7a..."),
                   src("github_file", GH + "comparison.qwen38-fp8-dequantized.json",
                       "55fa8d0505c5e2855e2ab04280bd7ac37727ea272d0e947fbf62ddb9958ea084",
                       "malaiwah.fidelity-comparison-receipt.v1; candidate dataset_sha256 "
                       "dbe0a446..., capture_content_digest cab1aa85...")],
          disclosures=[
              disc("lossy_capture_codec", "caveat",
                   "RECONSTRUCTED, NOT EXECUTED. The vendor FP8 path is unavailable on this "
                   "hardware: the fused deep-gemm kernel aborts with 'Unknown recipe' on "
                   "Blackwell. The candidate was therefore captured from a bf16 "
                   "materialisation of the stored fp8 weights "
                   "(k6/tools/dequant_fp8.py, w = fp8 * weight_scale_inv over 128x128 "
                   "blocks, accumulated fp32, stored bf16). This is the dequantize-and-run "
                   "methodology the GGUF/EXL3/MLX ecosystems use for KLD: it measures the "
                   "error of the STORED weights, not of the vendor kernel. Validated before "
                   "use: per-tensor rel-L2 against the root is 0.0265 uniformly across "
                   "gate/up/down/q projections, which is FP8 E4M3's expected error and "
                   "confirms the scale convention.",
                   True),
              disc("estimator_scope_narrower_than_artifact", "caveat",
                   "WEIGHT-ONLY, THEREFORE A LOWER BOUND. The checkpoint declares "
                   "activation_scheme: 'dynamic', i.e. the served model also quantizes "
                   "activations per-token at runtime. That term is absent from this "
                   "measurement, so this value is a LOWER BOUND on the served model's "
                   "divergence, not the served model's divergence. It is in particular NOT "
                   "the same quantity as measurement--qwen38.fp8.suite-v5-shard0-1m "
                   "(0.005197), which ran the real kernel on the vLLM lane.",
                   True),
              disc("record_note", "caveat",
                   "UPSTREAM LOADER DEFECT, ROUTED AROUND. Capturing this artifact through "
                   "stock transformers silently loads it WRONG. The producer's "
                   "modules_to_not_convert lists '...layers.N.mlp.gate' -- a MoE router that "
                   "does not exist in this dense checkpoint -- and "
                   "transformers.quantizers.quantizers_utils.should_convert_module tests "
                   "re.match(key, full_name), which is anchored only at the START, so that "
                   "pattern ALSO matches '...layers.N.mlp.gate_proj'. Verified against the "
                   "real tensor names: 65 of 65 gate_proj modules excluded from fp8 "
                   "conversion, 0 of 65 up_proj. Their fp8 weights load into plain bf16 "
                   "Linears with the block scale never applied, and the 65 "
                   "gate_proj.weight_scale_inv tensors drop out of the load as 'unexpected' "
                   "-- the only signal, and nothing refuses on it. The dequantisation used "
                   "here applies all 407 block scales, and the resulting checkpoint loads "
                   "with 0 unexpected / 0 missing / 0 mismatched.",
                   True),
              disc("single_run", "caveat",
                   "One cold capture of the candidate. Repeatability was not established "
                   "for the candidate side. The REFERENCE side is the three-run "
                   "bitwise-identical capture the floor row uses, and the comparison itself "
                   "is deterministic offline arithmetic over sealed tensors, so the "
                   "unrepeated term is the candidate forward pass alone."),
              disc("shared_reference_head", "info",
                   "One head (d922b751...) applied to both sides' hidden states."),
              Q38_HF_LANE_DISC, Q38_NOT_RANKABLE_DISC, Q38_HYBRID_SCOPE_DISC],
          **est),

        M("measurement--qwen38-hf.awq-int4-cyankiwi.suite-v5-shard0-1m",
          QWN, Q38_AWQ_CYAN, P_Q1M, Q38_HF_REF, PL_FIDDS_Q38, 0.022449361029279465,
          top1=0.9401801798363458,
          aux={"median_kld": 0.007379024173383942,
               "p95_kld": 0.07445505811118572,
               "p99_kld": 0.25099987912793037,
               "p999_kld": 1.1358863146676335,
               "max_kld": 9.553774200094734},
          scored_positions=1048064, contexts=512,
          runs=1, cold=True, evidence_kind="hidden_state_tensor_sha256",
          evidence_hashes=[Q38_AWQ_CAPTURE_SHA],
          det_note="One cold capture of the candidate. The reference side is the same "
                   "three-run-verified capture the floor row uses.",
          cls="advisory",
          sources=[ds_root,
                   src("model_card", "https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4",
                       None, "revision 63768c10..."),
                   src("github_file", GH + "comparison.qwen38-awq-int4-cyankiwi.json",
                       "e7d6e50ef432bcb133af62280778a109b7aeda918a2d1845cb5e82aadafc7ce3",
                       "malaiwah.fidelity-comparison-receipt.v1; candidate dataset_sha256 "
                       "59205aeb..., capture_content_digest d22de49a...")],
          disclosures=[
              disc("record_note", "info",
                   "EXECUTED, NOT RECONSTRUCTED. transformers loaded the compressed-tensors "
                   "pack-quantized checkpoint natively with 0 missing, 0 unexpected and 0 "
                   "mismatched tensors, so this is the artifact as the loader runs it. "
                   "Attributable error EQUALS this value: the floor on this reference is a "
                   "measured 0.0, so nothing is subtracted."),
              disc("third_party_artifact_self_measured", "info",
                   "cyankiwi's weights, our measurement."),
              disc("record_note", "info",
                   "DISTINCT FROM artifact--unattributed.qwen3.8-27b-awq-int4. That row's "
                   "artifact identity was never established (its receipt records only a "
                   "local path) and its scope claims moe.experts=quantized:awq@4 for a "
                   "checkpoint that contains no expert tensors at all. This row names a "
                   "pinned public repository and derives its scope from that repository's "
                   "own config and weight index. The two are NOT asserted to be the same "
                   "bytes."),
              disc("single_run", "caveat",
                   "One cold capture of the candidate. Repeatability was not established "
                   "for the candidate side. The REFERENCE side is the three-run "
                   "bitwise-identical capture the floor row uses, and the comparison itself "
                   "is deterministic offline arithmetic over sealed tensors, so the "
                   "unrepeated term is the candidate forward pass alone."),
              disc("shared_reference_head", "info",
                   "One head (d922b751...) applied to both sides' hidden states."),
              Q38_HF_LANE_DISC, Q38_NOT_RANKABLE_DISC, Q38_HYBRID_SCOPE_DISC],
          **est),
    ]
    # Stamp at BUILD time, from the digests of the code that actually ran.
    # stamp_harness() honours a row that already carries a recorded block.
    for rec in rows:
        rec["harness"] = q38_hf_harness()
    return rows


def stamp_harness(measurements):
    """Attach the harness block, and mark what predates it.

    Two populations, and the difference between them is the entire point:

    * the 12 rows whose `uncertainty` / `by_domain` / `protocol` blocks are
      DERIVED here, from per-window means this repository publishes, by code
      this repository ships. Those get a RECORDED harness covering exactly those
      three fields -- and deliberately NOT metric.value, which came from a GPU
      run that predates the mechanism. Claiming metric.value would be the precise
      failure the block exists to prevent;
    * every row's metric.value, which is grandfathered and says so, on the row.
    """
    closure = H.digests(os.path.dirname(L.repo_root(__file__)),
                        H.JOINT_DERIVATION_CLOSURE)
    hid = H.compute_id(closure, HARNESS_TOOL_VERSIONS)
    for rec in measurements:
        # A row that already carries a RECORDED harness produced its own stamp at
        # build time, from the digests of the code that actually ran. Re-stamping
        # it here would overwrite that with THIS checkout's joint-derivation
        # closure -- naming code that did not compute the row's metric.value,
        # which is the exact failure harness_id.py exists to prevent.
        if (rec.get("harness") or {}).get("recorded"):
            rec["disclosures"] = [d for d in rec["disclosures"]
                                  if d["code"] != "no_known_deviations"] or NONE_DISC
            continue
        covers = []
        if rec.get("by_domain"):
            covers += ["by_domain", "uncertainty"]
        if rec.get("protocol"):
            covers.append("protocol")
        # The clean17 rows are the only ones whose HEADLINE this code computed:
        # `_clean_row` re-reduces the published per-window means over the 17-window
        # scope and writes the result into metric.value. The panel25 headline came
        # off a GPU and is only CHECKED here, so it stays uncovered -- the
        # difference between "this code produced the number" and "this code agrees
        # with the number" is the whole distinction the block exists to carry.
        derived_value = rec["id"].endswith(joint_enrich.CLEAN_SUFFIX)
        if derived_value:
            covers.append("metric.value")
        if covers:
            rec["harness"] = {
                "harness_id": hid,
                "recorded": True,
                "boundary": H.BOUNDARY,
                "covers": sorted(covers),
                "repository": dict(HARNESS_REPOSITORY),
                "code_digests": closure,
                "tool_versions": dict(sorted(HARNESS_TOOL_VERSIONS.items())),
                "note": (("Covers metric.value: this scope's headline is the equal-weight "
                          "mean of the published per-window means over the clean17 window "
                          "set, computed by the code digested below. The per-window means "
                          "themselves come from the receipts named in provenance.sources."
                          if derived_value else
                          "Covers the LOCALLY DERIVED blocks only. metric.value came from a "
                          "measurement run that predates harness recording and is "
                          "grandfathered; see the harness_unrecorded disclosure.")
                         + " Two rows share this code exactly when harness_id is equal; a "
                           "differing id means read code_digests, because the boundary errs "
                           "toward over-sensitivity on purpose."),
            }
        else:
            rec["harness"] = {
                "harness_id": None,
                "recorded": False,
                "covers": ["metric.value"],
                "note": "No harness was recorded for the code that produced this value.",
            }
        ds = [d for d in rec["disclosures"] if d["code"] != "no_known_deviations"]
        if not derived_value:
            ds.append(disc("harness_unrecorded", "info", HARNESS_UNRECORDED_DETAIL))
        rec["disclosures"] = ds or NONE_DISC
    return measurements


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(L.repo_root(__file__), "data"))
    ap.add_argument("--check", action="store_true", help="fail if the files would change")
    args = ap.parse_args()

    amap = {a["id"]: a for a in ARTIFACTS}
    measurements = (build_measurements(amap) + build_measurements_runtime(amap)
                    + build_measurements_qwen(amap) + build_measurements_fruit(amap)
                    + build_measurements_qwen38_hf(amap))
    # Joint fidelity standard (2026-08-29): window-clustered BCa intervals, the
    # per-domain table, sigma_run in quadrature, the protocol stamp, and the
    # calibration-clean scope siblings. Implemented in tools/joint_enrich.py so
    # this generator keeps one job; every value it writes is re-derived from
    # registry/protocol/per-window/*.json, so `make reseed-check` still proves
    # the rows are a function of their receipts.
    measurements = joint_enrich.apply(measurements)
    measurements = stamp_harness(measurements)

    # The clean17 scope is a derived PANEL with a derived REFERENCE, so it gets
    # its own comparability key and can never be tabled beside the parent panel.
    panels_out = PANELS + joint_enrich.panels(PANELS)
    references_out = REFERENCES + joint_enrich.references(REFERENCES)

    collections_out = [("models", MODELS), ("artifacts", ARTIFACTS), ("panels", panels_out),
                       ("references", references_out), ("pipelines", PIPELINES),
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


def _require_numpy():
    """STAT-03. The CI endpoints this tool re-derives come from a PCG64 stream that
    reproduces the joint standard's reference implementation bit-for-bit. The stdlib
    fallback draws a different stream from the same seed, so reseeding without numpy
    would answer with different endpoints -- which is what made `make check` report a
    fabricated RESEED DRIFT on the stock interpreter the Makefile advertises."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "seed_registry: numpy is required to RE-DERIVE the rows.\n"
            "  The uncertainty block is a BCa bootstrap whose resample stream must match\n"
            "  the joint standard's reference implementation; the stdlib fallback draws a\n"
            "  different stream from the same seed and would move published CI endpoints\n"
            "  by up to 1.2%. Validation, rendering and the schema checks need no numpy;\n"
            "  only `make reseed` and `make reseed-check` do.\n")
        raise SystemExit(4)


if __name__ == "__main__":
    _require_numpy()
    sys.exit(main())
