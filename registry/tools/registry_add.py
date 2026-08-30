#!/usr/bin/env python3
"""Turn a sealed receipt into a conformant measurement row -- refusing to invent provenance.

Dispatch is on the receipt's own declared `schema` string, exact match. An unknown
receipt family is an error naming the observed string, never a guess at its shape.
Every field the row carries is either READ from the receipt (and recorded in
`field_provenance` with a JSON Pointer into it), or SUPPLIED by an explicit flag.
Nothing is defaulted, inferred or averaged into existence. Booleans and aggregates
that the receipt asserts are RECOMPUTED from the underlying arrays, and a
disagreement is an error rather than a warning.

Offline by construction: no networking module is imported anywhere in this module's
graph, and `--offline-selftest` proves it. `--*-url` flags record strings; nothing is
ever fetched.

  registry_add.py from-receipt   --receipt R [--receipt R2] --artifact A --panel P \
                                 --reference REF --pipeline PL --model M [flags]
  registry_add.py from-report    --report R  ... [--reference-revision SHA --reference-revision-evidence REF]
  registry_add.py from-crosscheck --report R ... (--floor-measurement ID | --floor-pending)
  registry_add.py from-foreign   --receipt R --reported-by HANDLE --source-url URL ...
  registry_add.py schemas        # list the receipt families this tool understands
  registry_add.py offline-selftest  # prove no networking module is reachable

Exit codes (stable; CONTRIBUTING.md cites them; argparse itself exits 2 on an unknown
subcommand or a malformed flag, which is not in this table because it never reaches
this tool's own logic):
  0 row written, or unchanged (idempotent re-run)
  3 unrecognized receipt `schema` string
  4 required provenance missing and no flag supplied
  5 receipt internally inconsistent (a recomputation disagreed with the receipt)
  6 provenance void: a flag asserts something the receipt contradicts, with no --disclosure
  7 identity clash: --panel / --reference / --artifact does not match the receipt's, or
    --floor-measurement names a floor measured on a different lane than this row's own
  8 attribution conflict
  9 id collision with differing content
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_lib as L  # noqa: E402

E_SCHEMA, E_MISSING, E_INCONSISTENT, E_VOID, E_IDENTITY, E_ATTRIB, E_COLLISION = 3, 4, 5, 6, 7, 8, 9


class Refuse(Exception):
    def __init__(self, code, message, remedy=None):
        Exception.__init__(self, message)
        self.code = code
        self.remedy = remedy


# --- receipt families -------------------------------------------------------
# The keys are the exact `schema` strings our tooling writes. OWN_SCHEMAS are the
# families we produce ourselves; only those may back a self-measured row.

PACKED_RECEIPT = "quant-pipeline.glm53-packed-kld-receipt.v1"
FIVE_COLD_RUN = "quant-pipeline.glm53-packed-student-kld-five-cold-run.v1"
DIONE_SUMMARY = "malaiwah.glm53-dione-q4-packed-kld-summary.v1"
# One producer, one format, one profile per bit rate -- so the family is a SET,
# not a string.  0xSero publishes a ladder (Q4, 3.0bpw, ...) and each rung's
# summary names its own profile in its schema; matching only the Q4 spelling
# would send the next rung to adapt_packed_and_five_run, whose arithmetic is a
# different receipt's.
DIONE_SUMMARIES = (
    DIONE_SUMMARY,
    "malaiwah.glm53-dione-3.0bpw-packed-kld-summary.v1",
)
# Releases stored in the STOCK-exllamav3 HF layout, scored through the SAME
# streaming harness via stream_score.py --source exl3hf.  Shape-identical to the
# Dione summary; what differs is where the artifact pins live (artifact_repo /
# artifact_revision rather than dione_repo / dione_revision).
#
# The family is a SET and it is keyed on the STORAGE layout, which is what picks
# the reader -- not on the producer and not on the scope.  turboderp's releases
# are full-scope with a 6-bit head; vcruz305's K2 pack is the same storage with
# the MCG codebook, a routed-experts-only scope and a NATIVE BF16 head.  The
# adapter already reads `declared_head_bits` off the receipt rather than
# asserting a head policy, so both are described correctly by the same code and
# neither inherits the other's identity.
EXL3HF_SUMMARIES = (
    "malaiwah.glm53-turbo-4.05bpw-packed-kld-summary.v1",
    "malaiwah.glm53-turbo-3.05bpw-packed-kld-summary.v1",
    "malaiwah.glm53-vcruz-k2-2bpw-packed-kld-summary.v1",
)
# TR3-published releases (brandonmusic's EXL3/MCG layout and its byte-identical
# mirrors) scored through the SAME streaming harness, via stream_score.py
# --source tr3.  Shape-identical to the turbo summary, and different from every
# other third-party family here in one way that must NOT be flattened: these
# releases SEAL themselves, and the summary carries the VERIFICATION of that
# seal (which claims were recomputed from the published bytes, and how the shard
# bytes were bound) rather than an unsealed-source caveat.  Coding that as
# `unsealed_source` -- which is what reusing the turbo adapter would do -- would
# put the opposite of the truth on the row.
TR3_SUMMARIES = (
    "malaiwah.glm53-tr3-4bpw-packed-kld-summary.v1",
)
FOREIGN_REPEATED = "glm53-r19-runtime-kld-repeated.v1"
FOREIGN_WINDOW = "glm53-r19-runtime-window-kld.v1"
FOREIGN_TP2 = "quant-pipeline.glm53-custom-tp2-runtime-window-kld.v1"

# The single-GPU streaming lane. Three summary families and one verdict.
#
# The K6 family's own name carries `-stream-`: that string is written by
# tools/stream_score.py's aggregator and by nothing else, so dispatching on it IS
# a statement about which lane produced the file. The K8 and native-BF16 families'
# names carry no such marker -- K8's `profile` reads "k8-tp4" and the native-BF16
# receipt's own `profile` reads "native-bf16-stream", but this tool dispatches lane
# identity off the `schema` string alone, never off a content field a receipt could
# spell differently -- so neither receipt says which lane it came from and the lane
# has to be supplied by --lane. The difference is recorded per-family in
# LANE_STATED_BY_SCHEMA rather than sniffed out of strings at read time.
#
# The native-BF16 family is shaped exactly like the other two (measured_mean_kld /
# run_means / distinct_tokenwise_kld_sha256 / cold_run_count / bitwise_deterministic)
# but its artifact is not a quant at all: it is the reference's OWN unquantized
# weights, scored through this same streaming harness (tools/stream_score.py
# --source native). That makes a row built from it the streaming lane's measurement
# floor -- see build_row's is_floor test and BIAS-004/005/006.
#
# The MLX family is the same shape again, produced by the same aggregator from a
# stream_score.py --source mlx capture.  Two things make it different from every
# other family here and both end up on the row as disclosures rather than as
# assumptions: its artifact is an UNSEALED third-party conversion (no contract,
# no payload digests -- the pins are the repo revision and the config/index
# sha256), and its quantization SCOPE reaches past the routed experts into dense
# MLPs, shared experts and attention projections, so the row must not be read as
# "experts quantized, everything else official".  The receipt states the measured
# scope policy; this tool copies it onto the row verbatim.
# The GGUF family is the same shape again, produced by the same aggregator
# (k6_kld_report.py --profile gguf) over stream_score.py --source gguf. What
# makes it different from every other row in this lane is SCOPE: a community
# llama.cpp GGUF quantizes the embeddings, the attention/KDA/DSA path and
# lm_head as well as the routed experts, and the measured forward therefore runs
# the artifact's own non-routed weights rather than the reference's. That is not
# a caveat this tool may infer -- the summary carries a scope_policy block read
# from the artifact's own tensor table, and the adapter below turns it into a
# disclosure on the row so a reader never compares a whole-model quant against a
# routed-experts-only one without being told.
STREAM_K6_SUMMARY = "malaiwah.glm53-k6-stream-packed-kld-summary.v1"
K8_SUMMARY = "malaiwah.glm53-k8-packed-kld-summary.v1"
NATIVE_BF16_SUMMARY = "malaiwah.glm53-native-bf16-packed-kld-summary.v1"
# The NVFP4 family is the same shape again, produced by tools/stream_score.py
# --source nvfp4 (community NVFP4 snapshots of the reference, decoded e2m1/gs16
# in exact fp32 and scored on the same panel by the same estimator). Two things
# make it different from every other family here and both are disclosed on the
# row rather than left to a reader: the artifact is SOMEONE ELSE'S and carries no
# encoder-side seal, and its measured scope is weights-only, so any activation
# quantization the artifact declares or ships scales for is NOT in the number.
# Those come out of the summary's own scope_policy / activations / seal_disclosure
# blocks, which stream_score copies verbatim from the sealed capture receipt --
# see _apply_nvfp4_disclosure, which REQUIRES them rather than tolerating a
# summary that dropped them.
# The three community-quant weight-decode families.  Each schema string names
# its FORMAT, so a registry row can never be read as if it came from another
# one, and none of them carries a "-stream-" marker: the lane is stated with
# --lane, exactly as for K8, because the lane is a fact about the measurement
# and the schema is a fact about the artifact.
MLX_SUMMARY = "malaiwah.glm53-mlx-packed-kld-summary.v1"
GGUF_SUMMARY = "malaiwah.glm53-gguf-packed-kld-summary.v1"
NVFP4_SUMMARY = "malaiwah.glm53-nvfp4-packed-kld-summary.v1"
STREAM_VERDICT = "malaiwah.glm53-streaming-measurement-verdict.v1"
STREAM_SUMMARIES = (STREAM_K6_SUMMARY, K8_SUMMARY, NATIVE_BF16_SUMMARY,
                    MLX_SUMMARY, GGUF_SUMMARY, NVFP4_SUMMARY)
# The three community families are None for the same reason K8 is: their family
# strings carry no `-stream-` marker, and this tool refuses to infer a lane from
# anything but the string.  Only one runner writes them today, but that is a fact
# about content, not about the name, so the lane still has to come from --lane.
LANE_STATED_BY_SCHEMA = {STREAM_K6_SUMMARY: "streaming", K8_SUMMARY: None,
                         NATIVE_BF16_SUMMARY: None, MLX_SUMMARY: None,
                         GGUF_SUMMARY: None, NVFP4_SUMMARY: None}

REPORT_FAMILIES = ("glm53flash-fidelity-report/2", "glm53flash-fidelity-report/3",
                   "qwen38-kld-ladder-cumulative/2", "qwen38-kld-ladder-cumulative/3",
                   "qwen38-fidelity-report/2", "qwen38-fidelity-report/3")
CROSSCHECK_FAMILIES = ("glm53flash-crosscheck/2",)
OWN_SCHEMAS = set([PACKED_RECEIPT, FIVE_COLD_RUN, STREAM_VERDICT]
                  + list(DIONE_SUMMARIES)
                  + list(STREAM_SUMMARIES) + list(EXL3HF_SUMMARIES)
                  + list(TR3_SUMMARIES)
                  + list(REPORT_FAMILIES) + list(CROSSCHECK_FAMILIES))
FOREIGN_SCHEMAS = {FOREIGN_REPEATED, FOREIGN_WINDOW, FOREIGN_TP2}
KNOWN = OWN_SCHEMAS | FOREIGN_SCHEMAS


def load_receipt(path):
    if not os.path.exists(path):
        raise Refuse(E_MISSING, "receipt not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        try:
            r = json.load(fh)
        except ValueError as exc:
            raise Refuse(E_MISSING, "receipt is not valid JSON: %s (%s)" % (path, exc))
    schema = r.get("schema")
    if schema not in KNOWN:
        raise Refuse(E_SCHEMA,
                     "unrecognized receipt family. The receipt declares schema=%r.\n"
                     "This tool dispatches on that exact string and will not guess at an unknown "
                     "receipt's shape.\nKnown families:\n  %s" % (schema, "\n  ".join(sorted(KNOWN))),
                     "add an adapter keyed on that schema string, or convert the receipt")
    return r, path, L.sha256_file(path)


# --- adapters ---------------------------------------------------------------

def _need(receipt, pointer, path, ctx=""):
    node, cur = receipt, ""
    for part in [p for p in pointer.split("/") if p]:
        cur += "/" + part
        if not isinstance(node, dict) or part not in node:
            raise Refuse(E_MISSING,
                         "%s does not carry %s%s. The tool will not substitute a default for a "
                         "provenance-bearing field." % (os.path.basename(path), pointer,
                                                        (" (%s)" % ctx) if ctx else ""))
        node = node[part]
    return node


def adapt_packed_and_five_run(receipts):
    """F1 + F2 fused: the packed receipt supplies the value and the seals, the five-run
    receipt supplies the determinism evidence. The two must agree or it is exit 5."""
    packed = next((r for r in receipts if r[0].get("schema") == PACKED_RECEIPT), None)
    five = next((r for r in receipts if r[0].get("schema") == FIVE_COLD_RUN), None)
    if five is None:
        raise Refuse(E_MISSING, "no %s receipt supplied: without it there is no determinism evidence "
                                "and no per-run array to recompute against." % FIVE_COLD_RUN)
    fr, fpath, fsha = five
    runs = _need(fr, "/runs", fpath)
    means = [x["mean_kld"] for x in runs]
    value = _need(fr, "/mean_of_run_means", fpath)
    if not L.close(value, sum(means) / len(means)):
        raise Refuse(E_INCONSISTENT, "mean_of_run_means %r != mean of the per-run means %r"
                     % (value, sum(means) / len(means)))
    for field, want in (("population_stddev_of_run_means", L.population_stddev(means)),
                        ("minimum_run_mean", min(means)), ("maximum_run_mean", max(means))):
        if field in fr and not L.close(fr[field], want):
            raise Refuse(E_INCONSISTENT, "%s = %r but recomputing from runs[] gives %r"
                         % (field, fr[field], want))
    if _need(fr, "/run_count", fpath) != len(runs):
        raise Refuse(E_INCONSISTENT, "run_count != len(runs)")
    positions = {x.get("prediction_positions") for x in runs}
    if len(positions) != 1:
        raise Refuse(E_INCONSISTENT, "runs disagree on prediction_positions: %s" % sorted(positions))
    digests = sorted({x.get("tokenwise_kld_sha256") for x in runs if x.get("tokenwise_kld_sha256")})
    if len(digests) != len(runs) and len(digests) != 1:
        pass  # partial digests: reported below as not-identical
    identical = len(digests) == 1 and len(digests) == len({d for d in digests})
    if packed:
        pr, ppath, _ = packed
        pv = _need(pr, "/measured_mean_kld", ppath)
        if not L.close(pv, value):
            raise Refuse(E_INCONSISTENT, "the packed receipt says %r and the five-run receipt says %r; "
                                         "they are not the same measurement." % (pv, value))
        for field in ("token_panel_receipt_sha256", "teacher_receipt_sha256"):
            if field in pr and field in fr and pr[field] != fr[field]:
                raise Refuse(E_INCONSISTENT, "the two receipts disagree on %s" % field)
    out = {
        "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
        "direction": _dir(_need(fr, "/kld_direction", fpath)),
        "accumulation": _acc(_need(fr, "/compute_dtype", fpath)),
        "scored_positions": positions.pop() * len(runs) if False else sum(
            x["prediction_positions"] for x in runs) // len(runs),
        "runs": len(runs), "run_means": means, "cold": True,
        "identical": bool(identical and len(runs) >= 2),
        "evidence_kind": "tokenwise_kld_sha256" if digests else "run_mean_equality_only",
        "evidence_hashes": digests if identical else (digests or None),
        "panel_digest": fr.get("token_panel_receipt_sha256"),
        "teacher_digest": fr.get("teacher_receipt_sha256"),
        "gate": _gate(fr),
        "field_provenance": {"value": "#/mean_of_run_means", "direction": "#/kld_direction",
                             "accumulation": "#/compute_dtype", "runs": "#/run_count",
                             "run_means": "#/runs[]/mean_kld",
                             "evidence_hashes": "#/runs[]/tokenwise_kld_sha256"},
        "receipt_schema": FIVE_COLD_RUN,
        "stack_relation": "same_stack", "head_policy": "native_head",
    }
    if not identical and len(digests) > 1:
        out["det_note"] = ("%d DISTINCT per-run tokenwise digests: this measurement is not bitwise "
                           "reproducible." % len(digests))
    return out


def adapt_dione(receipt, path):
    value = _need(receipt, "/measured_mean_kld", path)
    means = _need(receipt, "/run_means", path)
    digests = _need(receipt, "/distinct_tokenwise_kld_sha256", path)
    recomputed = len(set(digests)) == 1 and len(set(means)) == 1
    declared = receipt.get("bitwise_deterministic")
    if declared is not None and bool(declared) != recomputed:
        raise Refuse(E_INCONSISTENT,
                     "the receipt declares bitwise_deterministic=%r but recomputing from run_means "
                     "(%d distinct) and distinct_tokenwise_kld_sha256 (%d entries) gives %r"
                     % (declared, len(set(means)), len(set(digests)), recomputed))
    if not L.close(value, sum(means) / len(means)):
        raise Refuse(E_INCONSISTENT, "measured_mean_kld != mean(run_means)")
    # CODED disclosures.  The untyped `verbatim_disclosure` channel stamps
    # `unsealed_source` on EVERY entry it holds, so this adapter used to publish
    # a reduced run count under the code for a missing seal -- two different
    # facts, one wrong code.  Same fix the turbo and tr3 adapters already got.
    coded = []
    if receipt.get("seal_disclosure"):
        coded.append({"code": "unsealed_source",
                      "detail": "seal_disclosure (verbatim from the receipt): %s"
                                % receipt["seal_disclosure"]})
    if receipt.get("cold_run_deviation"):
        coded.append({"code": "reduced_run_count",
                      "detail": "cold_run_deviation (verbatim from the receipt): %s"
                                % receipt["cold_run_deviation"]})
    verification = receipt.get("dione_shard_hash_verification")
    if verification == "full":
        # Not a caveat: the release publishes a sha256 for every shard and the
        # measurement hashed all of them before decoding anything.  It is the
        # strongest binding an unsealed release admits, and coding it as a
        # caveat would say the opposite of what happened.
        coded.append({
            "code": "shard_hashes_verified",
            "severity": "info", "affects_comparability": False,
            "detail": ("dione_shard_hash_verification=full (verbatim from the "
                       "receipt): every shard's sha256 was recomputed and matched "
                       "the release's own manifest (%s) before any payload was "
                       "decoded. The release publishes no seal, so this and the "
                       "immutable revision are the provenance anchors."
                       % (receipt.get("exl3_manifest_name") or "exl3-manifest.json"))})
    elif verification is not None:
        coded.append({
            "code": "shard_hashes_unverified",
            "detail": ("dione_shard_hash_verification=%r (verbatim from the "
                       "receipt): the shard bytes were NOT bound to the release's "
                       "published per-shard digests." % verification)})
    head_bits = receipt.get("declared_head_bits")
    if head_bits is not None and float(head_bits) >= 16:
        coded.append({
            "code": "native_head_retained",
            "severity": "info", "affects_comparability": False,
            "detail": ("declared_head_bits %s (verbatim from the receipt): this "
                       "release retains the lm_head at source precision, unlike a "
                       "stock-exllamav3 release which quantizes it. The head is "
                       "applied natively from the artifact's own weights."
                       % head_bits)})
    return {
        "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
        "direction": "reference_to_candidate", "accumulation": "float64",
        "runs": len(means), "run_means": list(means), "cold": True,
        "identical": recomputed and len(means) >= 2,
        "evidence_kind": "tokenwise_kld_sha256", "evidence_hashes": list(digests),
        "scored_positions": None, "gate": _gate(receipt),
        # A KL number without top-1 agreement does not say WHICH divergence it
        # is; the Dione adapter was the one family still dropping it.
        "top1": receipt.get("top1_agreement"),
        "artifact_repo": receipt.get("dione_repo"), "artifact_revision": receipt.get("dione_revision"),
        "teacher_digest": receipt.get("teacher_receipt_sha256"),
        "verbatim_disclosure_coded": coded,
        "field_provenance": {"value": "#/measured_mean_kld", "run_means": "#/run_means",
                             "evidence_hashes": "#/distinct_tokenwise_kld_sha256",
                             "artifact_repo": "#/dione_repo",
                             "artifact_revision": "#/dione_revision",
                             "head_policy": ("SUPPLIED: how the head is APPLIED "
                                             "(natively, from the artifact's own "
                                             "retained weights). #/declared_head_bits "
                                             "says what the artifact's head IS.")},
        # The receipt's OWN schema string, not the Q4 constant: two rungs of one
        # ladder are two families, and copying the first one's name onto the
        # second would make the row cite a receipt that does not exist.
        "receipt_schema": receipt.get("schema") or DIONE_SUMMARY,
        "stack_relation": "same_stack", "head_policy": "native_head",
    }


def adapt_turbo(receipt, path):
    """F7: a stock-exllamav3 release on the streaming lane (--source exl3hf).

    Recomputed exactly like the Dione summary -- the asserted mean is re-derived
    from run_means and the asserted bitwise_deterministic flag from the per-run
    means and the distinct tokenwise digests; a disagreement is exit 5, never a
    warning.

    Two things this family states that the Dione one does not, and that the row
    must therefore carry rather than lose:

      * declared_head_bits.  READ, never assumed: turboderp's stock releases
        quantize their own lm_head, TR3 does not, and vcruz305's K2 pack
        declares head_bits 16 and keeps it native too.  head_policy stays
        "native_head" because
        that field describes how the head is APPLIED -- natively, from the
        artifact's own weights, with no shared replay -- and the fact that
        those weights are themselves quantized is ARTIFACT identity. It is
        disclosed here so no reader has to infer it from a bit count.
      * the codebook -- mul1 on turboderp's releases, mcg on vcruz305's -- and
        the exllamav3-compatible quantizer version that wrote it.

    The lane is not read from the schema string: like K8 and native-BF16, this
    family's name carries no lane marker, so --lane supplies it.
    """
    value = _need(receipt, "/measured_mean_kld", path)
    means = _need(receipt, "/run_means", path)
    digests = _need(receipt, "/distinct_tokenwise_kld_sha256", path)
    recomputed = len(set(digests)) == 1 and len(set(means)) == 1
    declared = receipt.get("bitwise_deterministic")
    if declared is not None and bool(declared) != recomputed:
        raise Refuse(E_INCONSISTENT,
                     "the receipt declares bitwise_deterministic=%r but recomputing from run_means "
                     "(%d distinct) and distinct_tokenwise_kld_sha256 (%d entries) gives %r"
                     % (declared, len(set(means)), len(set(digests)), recomputed))
    if not L.close(value, sum(means) / len(means)):
        raise Refuse(E_INCONSISTENT, "measured_mean_kld != mean(run_means)")
    # CODED disclosures, not a bag of strings: the untyped `verbatim_disclosure`
    # list is stamped `unsealed_source` for every entry it holds, so three
    # different facts came out as three identical codes on one row.
    coded = []
    if receipt.get("seal_disclosure"):
        coded.append({"code": "unsealed_source",
                      "detail": "seal_disclosure (verbatim from the receipt): %s"
                                % receipt["seal_disclosure"]})
    if receipt.get("cold_run_deviation"):
        coded.append({"code": "reduced_run_count",
                      "detail": "cold_run_deviation (verbatim from the receipt): %s"
                                % receipt["cold_run_deviation"]})
    head_bits = receipt.get("declared_head_bits")
    if head_bits is not None and float(head_bits) < 16:
        coded.append({
            "code": "quantized_head",
            "detail": ("declared_head_bits %s (verbatim from the receipt): this "
                       "artifact's lm_head is itself quantized by the producer, "
                       "unlike the TR3 artifacts on this panel which keep it native "
                       "BF16. It is APPLIED natively from the artifact's own weights "
                       "-- no shared or replayed head -- so estimator.head_policy is "
                       "native_head; the quantization is artifact identity."
                       % head_bits)})
    return {
        "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
        "direction": "reference_to_candidate", "accumulation": "float64",
        "runs": len(means), "run_means": list(means), "cold": True,
        "identical": recomputed and len(means) >= 2,
        "evidence_kind": "tokenwise_kld_sha256", "evidence_hashes": list(digests),
        "scored_positions": None, "gate": _gate(receipt),
        "top1": receipt.get("top1_agreement"),
        "artifact_repo": receipt.get("artifact_repo"),
        "artifact_revision": receipt.get("artifact_revision"),
        "teacher_digest": receipt.get("teacher_receipt_sha256"),
        "verbatim_disclosure_coded": coded,
        "field_provenance": {"value": "#/measured_mean_kld", "run_means": "#/run_means",
                             "evidence_hashes": "#/distinct_tokenwise_kld_sha256",
                             "artifact_repo": "#/artifact_repo",
                             "artifact_revision": "#/artifact_revision",
                             "head_policy": ("SUPPLIED: how the head is APPLIED "
                                             "(natively, from the artifact's own "
                                             "weights). #/declared_head_bits says "
                                             "what the artifact's head IS.")},
        "receipt_schema": receipt.get("schema"),
        "stack_relation": "same_stack", "head_policy": "native_head",
    }


def adapt_tr3(receipt, path):
    """F8: a SEALED TR3-published release on the streaming lane (--source tr3).

    The arithmetic is the turbo adapter's, recomputed the same way: the asserted
    mean is re-derived from run_means and the asserted bitwise_deterministic
    flag from the per-run means and the distinct tokenwise digests; a
    disagreement is exit 5, never a warning.

    What this family states that no other third-party family does, and what the
    row must therefore carry:

      * A PUBLISHER SEAL, and its verification.  exl3-mcg-storage-abi.json and
        materialization-receipt.json state digests over the emitted name set,
        the materialization plan, the config and the index; the measurement
        recomputed every one of them from the published bytes before decoding.
        This adapter REQUIRES seal_verified to be true and refuses a summary
        that merely claims a seal without saying what was checked -- an
        unverified seal is a word, and the whole point of coding it is that a
        reader can tell the two apart.
      * SCOPE.  scope=glm53_routed_experts_only with head_bits 16: the routed
        experts are quantized and every other tensor -- lm_head included -- is
        the OFFICIAL one.  That is the opposite end of the scope axis from the
        stock-exllamav3 rows on this same panel, and a reader must not have to
        infer it from a bit count.

    The lane is not read from the schema string: like K8, native-BF16 and the
    turbo families, this name carries no lane marker, so --lane supplies it.
    """
    value = _need(receipt, "/measured_mean_kld", path)
    means = _need(receipt, "/run_means", path)
    digests = _need(receipt, "/distinct_tokenwise_kld_sha256", path)
    recomputed = len(set(digests)) == 1 and len(set(means)) == 1
    declared = receipt.get("bitwise_deterministic")
    if declared is not None and bool(declared) != recomputed:
        raise Refuse(E_INCONSISTENT,
                     "the receipt declares bitwise_deterministic=%r but recomputing from run_means "
                     "(%d distinct) and distinct_tokenwise_kld_sha256 (%d entries) gives %r"
                     % (declared, len(set(means)), len(set(digests)), recomputed))
    if not L.close(value, sum(means) / len(means)):
        raise Refuse(E_INCONSISTENT, "measured_mean_kld != mean(run_means)")
    if receipt.get("seal_verified") is not True:
        raise Refuse(E_INCONSISTENT,
                     "a %s receipt must carry seal_verified=true: this family exists "
                     "because the release seals itself and the measurement recomputed "
                     "that seal. The receipt says seal_verified=%r."
                     % (receipt.get("schema"), receipt.get("seal_verified")),
                     "re-run the measurement with a stream_score build that verifies "
                     "the seal, or submit it through a family that claims no seal")
    checks = receipt.get("seal_check_names") or []
    passed = receipt.get("seal_checks_passed")
    if not checks or passed != len(checks):
        raise Refuse(E_INCONSISTENT,
                     "seal_verified is true but the receipt names %d checks and reports "
                     "%r passed. A seal is only evidence when the row can say WHICH "
                     "claims were recomputed." % (len(checks), passed))
    coded = [{
        "code": "sealed_source_verified",
        "detail": ("The release publishes its own storage ABI and materialization "
                   "receipt, and this measurement RECOMPUTED all %d of their claims "
                   "from the published bytes before decoding (%s). Shard bytes: %s. "
                   "seal_disclosure (verbatim from the receipt): %s"
                   % (len(checks), ", ".join(checks),
                      receipt.get("shard_verification") or "not stated",
                      receipt.get("seal_disclosure") or "not stated")),
        "severity": "info", "affects_comparability": False,
    }]
    if receipt.get("cold_run_deviation"):
        coded.append({"code": "reduced_run_count",
                      "detail": "cold_run_deviation (verbatim from the receipt): %s"
                                % receipt["cold_run_deviation"]})
    head_bits = receipt.get("declared_head_bits")
    if head_bits is not None and float(head_bits) < 16:
        raise Refuse(E_INCONSISTENT,
                     "this family is the routed-experts-only TR3 scope, whose head is "
                     "native BF16; the receipt declares head_bits=%r. A quantized head "
                     "changes the measured function and belongs on a family that says "
                     "so." % head_bits)
    scope_policy = receipt.get("scope_policy")
    if scope_policy:
        coded.append({
            "code": "routed_experts_only_scope",
            "detail": ("scope_policy (verbatim from the release's own config): %s, "
                       "non_routed_dtype_policy %s, head_bits %s. Only the routed "
                       "experts are quantized; every other tensor including lm_head "
                       "is the OFFICIAL source tensor, verified name-set-equal to the "
                       "official release's 1,618 non-routed names. Rows from "
                       "full-scope artifacts on this same panel are measuring a "
                       "different amount of model."
                       % (scope_policy,
                          receipt.get("nonrouted_policy_declared") or "not stated",
                          head_bits)),
            "severity": "info", "affects_comparability": True})
    return {
        "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
        "direction": "reference_to_candidate", "accumulation": "float64",
        "runs": len(means), "run_means": list(means), "cold": True,
        "identical": recomputed and len(means) >= 2,
        "evidence_kind": "tokenwise_kld_sha256", "evidence_hashes": list(digests),
        "scored_positions": None, "gate": _gate(receipt),
        "top1": receipt.get("top1_agreement"),
        "artifact_repo": receipt.get("artifact_repo"),
        "artifact_revision": receipt.get("artifact_revision"),
        "teacher_digest": receipt.get("teacher_receipt_sha256"),
        "verbatim_disclosure_coded": coded,
        "field_provenance": {"value": "#/measured_mean_kld", "run_means": "#/run_means",
                             "evidence_hashes": "#/distinct_tokenwise_kld_sha256",
                             "artifact_repo": "#/artifact_repo",
                             "artifact_revision": "#/artifact_revision",
                             "seal": "#/seal_verified + #/seal_check_names",
                             "scope": "#/scope_policy + #/declared_head_bits",
                             "head_policy": ("SUPPLIED: how the head is APPLIED "
                                             "(natively, from the artifact's own "
                                             "weights). #/declared_head_bits says "
                                             "what the artifact's head IS -- here, "
                                             "16, i.e. unquantized.")},
        "receipt_schema": receipt.get("schema"),
        "stack_relation": "same_stack", "head_policy": "native_head",
    }


def adapt_stream_summary(receipts):
    """F5: the single-GPU streaming lane -- a packed-KLD summary, optionally with its verdict.

    Shape-wise this is the Dione summary's twin (measured_mean_kld / run_means /
    distinct_tokenwise_kld_sha256 / bitwise_deterministic / cold_run_count), and it is
    recomputed the same way: the asserted mean is re-derived from run_means and the
    asserted `bitwise_deterministic` flag is re-derived from the per-run means and the
    distinct tokenwise digests. The flag is never trusted; a disagreement is exit 5.

    What this family does NOT state, and therefore what this adapter refuses to invent:

      * the KL direction and the accumulation dtype -- neither appears anywhere in the
        file, so --direction and --accumulation must supply them;
      * how many positions and contexts were scored -- the summary is a scalar, so
        --scored-positions and --contexts must supply them, unless a verdict receipt
        is also given, in which case the context count is READ from its per-window
        array and the array's mean is checked against the summary's scalar;
      * which lane produced it, for the K8 and native-BF16 families (see
        LANE_STATED_BY_SCHEMA).

    When the verdict IS supplied it is not carried along unread. Every number it
    asserts about the bridge to the sealed lane -- the per-window deltas, their
    maximum, the mean delta -- is recomputed from its own per_window array, and the
    verdict is required to agree with the summary about the value, the per-run means
    and the tokenwise digests. That is what makes the bias block on the resulting row
    a measurement rather than a claim.
    """
    summaries = [r for r in receipts if r[0].get("schema") in STREAM_SUMMARIES]
    verdicts = [r for r in receipts if r[0].get("schema") == STREAM_VERDICT]
    if len(summaries) != 1:
        raise Refuse(E_MISSING,
                     "expected exactly one %s receipt, got %d. The verdict receipt describes a "
                     "summary; it is not a measurement on its own."
                     % (" / ".join(STREAM_SUMMARIES), len(summaries)))
    rec, path, _ = summaries[0]
    sch = rec.get("schema")

    value = _need(rec, "/measured_mean_kld", path)
    means = _need(rec, "/run_means", path)
    digests = _need(rec, "/distinct_tokenwise_kld_sha256", path)
    runs = _need(rec, "/cold_run_count", path)
    if not L.close(value, sum(means) / len(means)):
        raise Refuse(E_INCONSISTENT, "measured_mean_kld %r != mean(run_means) %r"
                     % (value, sum(means) / len(means)))
    if runs != len(means):
        raise Refuse(E_INCONSISTENT, "cold_run_count is %r but run_means has %d entries"
                     % (runs, len(means)))
    recomputed = len(set(digests)) == 1 and len(set(means)) == 1 and len(means) >= 2
    declared = rec.get("bitwise_deterministic")
    if declared is not None and bool(declared) != recomputed:
        raise Refuse(E_INCONSISTENT,
                     "the receipt declares bitwise_deterministic=%r but recomputing from run_means "
                     "(%d distinct over %d runs) and distinct_tokenwise_kld_sha256 (%d entries) "
                     "gives %r" % (declared, len(set(means)), len(means), len(set(digests)),
                                   recomputed))
    reports = rec.get("kld_report_sha256")
    det_note = None
    if reports is not None:
        if len(reports) != runs:
            raise Refuse(E_INCONSISTENT, "kld_report_sha256 has %d entries but cold_run_count is %r"
                         % (len(reports), runs))
        det_note = ("%d cold runs, %d distinct kld_report_sha256 values, %d distinct "
                    "tokenwise_kld_sha256. The report-file digests differ per run and prove "
                    "nothing; the single tokenwise digest is the determinism evidence."
                    % (runs, len(set(reports)), len(set(digests))))

    out = {
        "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
        # Not stated by this receipt family. Left as None so a flag must supply them.
        "direction": None, "accumulation": None,
        "scored_positions": None, "contexts": None,
        "runs": len(means), "run_means": list(means), "cold": True,
        "identical": recomputed,
        "evidence_kind": "tokenwise_kld_sha256", "evidence_hashes": list(digests),
        "det_note": det_note,
        "gate": _gate(rec),
        "teacher_digest": rec.get("teacher_receipt_sha256"),
        "lane": LANE_STATED_BY_SCHEMA.get(sch),
        "requires_lane": True,
        "verbatim_disclosure_coded": [],
        "field_provenance": {"value": "#/measured_mean_kld", "run_means": "#/run_means",
                             "runs": "#/cold_run_count",
                             "evidence_hashes": "#/distinct_tokenwise_kld_sha256",
                             "identical_across_runs": "RECOMPUTED from #/run_means and "
                                                      "#/distinct_tokenwise_kld_sha256; the "
                                                      "receipt's own bitwise_deterministic flag "
                                                      "was checked against it, not copied"},
        "receipt_schema": sch,
        "stack_relation": "same_stack", "head_policy": "native_head",
    }
    if out["lane"]:
        out["field_provenance"]["lane"] = ("#/schema (this exact family string is written by the "
                                           "streaming runner and by nothing else)")
    if rec.get("cold_run_deviation"):
        out["verbatim_disclosure_coded"].append(
            {"code": "reduced_run_count",
             "detail": "cold_run_deviation (verbatim from the receipt): %s"
                       % rec["cold_run_deviation"]})
    # One dispatch table for the community-quant families.  Each entry binds the
    # artifact and states what that FORMAT quantized; none of them may be
    # inferred from the schema string, and a receipt missing its family's scope
    # census is refused rather than rowed as an experts-only quant.
    adapter = {MLX_SUMMARY: _apply_mlx_provenance,
               GGUF_SUMMARY: _apply_gguf_provenance,
               NVFP4_SUMMARY: _apply_nvfp4_disclosure}.get(sch)
    if adapter is not None:
        adapter(out, rec, path)
    if verdicts:
        _apply_stream_verdict(out, rec, path, verdicts)
    return out


def _apply_mlx_provenance(out, rec, path):
    """The MLX family's extra fields: the artifact pins, the unsealed-source
    disclosure and the MEASURED quantization scope, all verbatim.

    Nothing here is inferred.  A receipt that does not carry the scope census is
    refused rather than rowed as if the scope were the usual experts-only one:
    the whole point of the family is that it is NOT.
    """
    scope = rec.get("mlx_scope_policy")
    if not isinstance(scope, dict) or scope.get("quantized_module_count") is None:
        raise Refuse(E_MISSING,
                     "%s carries no mlx_scope_policy census. This artifact family quantizes "
                     "beyond the routed experts, and a row that does not say what was quantized "
                     "would be read as if only the experts were." % os.path.basename(path))
    out["artifact_repo"] = rec.get("mlx_repo")
    out["artifact_revision"] = rec.get("mlx_revision")
    out["field_provenance"].update({
        "artifact_repo": "#/mlx_repo", "artifact_revision": "#/mlx_revision",
    })
    if rec.get("seal_disclosure"):
        out["verbatim_disclosure_coded"].append(
            {"code": "unsealed_source",
             "detail": "seal_disclosure (verbatim from the receipt): %s" % rec["seal_disclosure"]})
    detail = (
        "mlx_scope_policy (measured from the artifact's own index + shard headers, verbatim "
        "from the receipt): %s -- %d quantized modules = %d routed expert + %d MTP expert + "
        "%d non-routed (%s); %d tensors left at source dtype; bit mix %s. Weights only: %s."
        % (scope.get("policy"), scope.get("quantized_module_count", -1),
           scope.get("routed_expert_modules", -1), scope.get("mtp_expert_modules", -1),
           scope.get("nonrouted_quantized_modules", -1),
           ", ".join("%s x%d" % (k.split(".")[-1], v)
                     for k, v in sorted((scope.get("nonrouted_quantized_kinds") or {}).items()))
           or "none",
           scope.get("passthrough_tensor_count", -1),
           " ".join("%s:%d" % (k, v) for k, v in sorted((scope.get("bits_histogram") or {}).items())),
           scope.get("activations"))
    )
    out["verbatim_disclosure_coded"].append({"code": "quantization_scope", "detail": detail})
    if rec.get("nonrouted_policy"):
        out["verbatim_disclosure_coded"].append(
            {"code": "nonrouted_weights_decoded",
             "detail": "nonrouted_policy (verbatim from the receipt): %s. The non-routed model was "
                       "built from a decoded view of the quant snapshot, NOT from the official "
                       "BF16 tree." % rec["nonrouted_policy"]})
    if rec.get("mlx_shard_hash_verification") == "skipped":
        out["verbatim_disclosure_coded"].append(
            {"code": "shard_hashes_unverified",
             "detail": "mlx_shard_hash_verification is \"skipped\": the shards were read without "
                       "whole-file sha256 verification against the HF manifest; the pins are the "
                       "repo revision and the config/index sha256 only."})
    out["field_provenance"]["quantization_scope"] = "#/mlx_scope_policy (measured, not declared)"


def _apply_gguf_provenance(out, rec, path):
    """Bind the third-party artifact and disclose WHAT IT QUANTIZED.

    Two things must reach the row and neither may be inferred:

      * identity -- the repo, the immutable 40-hex revision and the per-file
        sha256 of every .gguf consumed. A GGUF repo holds a dozen different
        quants under one revision, so the FILE list is the artifact, not the
        repo id; a summary that names files without hashes is refused rather
        than recorded as if it were pinned.
      * scope -- the artifact's own measured scope_policy. Every other row in
        this lane quantizes the routed experts only and runs the reference's
        untouched non-routed parameters; this family does not, and a row that
        omits that is quietly incomparable.
    """
    files = rec.get("gguf_files")
    if not files:
        raise Refuse(E_MISSING,
                     "%s carries no /gguf_files: a GGUF repo holds many different quants at one "
                     "revision, so the file list IS the artifact identity. Re-run the aggregator "
                     "against a capture receipt that records it." % os.path.basename(path))
    names = [f.get("name") for f in files]
    verification = rec.get("gguf_file_hash_verification")
    if verification == "full" and any(not f.get("sha256") for f in files):
        raise Refuse(E_INCONSISTENT,
                     "the summary declares gguf_file_hash_verification='full' but %d of %d entries "
                     "in /gguf_files carry no sha256"
                     % (sum(1 for f in files if not f.get("sha256")), len(files)))
    out["artifact_repo"] = rec.get("gguf_repo")
    out["artifact_revision"] = rec.get("gguf_revision")
    if rec.get("seal_disclosure"):
        # same treatment as the Dione and MLX families: a third-party artifact
        # with no upstream encoder receipts becomes an `unsealed_source` caveat
        out["verbatim_disclosure_coded"].append(
            {"code": "unsealed_source",
             "detail": "seal_disclosure (verbatim from the receipt): %s" % rec["seal_disclosure"]})
    out["field_provenance"].update({
        "artifact_repo": "#/gguf_repo", "artifact_revision": "#/gguf_revision",
        "artifact_files": "#/gguf_files[] (name + bytes + sha256)",
    })
    scope = rec.get("scope_policy") or {}
    if not scope.get("disclosure"):
        raise Refuse(E_MISSING,
                     "%s carries no /scope_policy/disclosure. This family's artifacts quantize the "
                     "non-routed tensors too; a row without that statement invites comparison "
                     "against routed-experts-only rows that are not measuring the same weights."
                     % os.path.basename(path))
    out["verbatim_disclosure_coded"].append(
        {"code": "quantization_scope_whole_model",
         "detail": "scope_policy (verbatim from the receipt): %s | embeddings=%s lm_head=%s "
                   "attention_quantized=%s routed_expert_types=%s"
                   % (scope["disclosure"], scope.get("embeddings_type"),
                      scope.get("lm_head_type"), scope.get("attention_kda_dsa_quantized"),
                      ",".join(scope.get("routed_expert_types") or []))})
    if verification != "full":
        out["verbatim_disclosure_coded"].append(
            {"code": "artifact_files_unhashed",
             "detail": "gguf_file_hash_verification is %r: the measured files were NOT whole-file "
                       "hashed, so this row pins the artifact by repo+revision+name only (%s)"
                       % (verification, ", ".join(str(n) for n in names))})


def _apply_nvfp4_disclosure(out, rec, path):
    """Fold the NVFP4 family's mandatory provenance and caveats into the row.

    Nothing here is invented or paraphrased: every string is lifted verbatim from
    the summary, which lifted it from the sealed capture receipt's
    streaming_disclosure.nvfp4 block, which stream_score built from the artifact's
    OWN config and index. What this function adds is refusal -- a summary of this
    family that lost its scope or activation block cannot become a row, because the
    row would then read like a whole-artifact measurement of a sealed quant, and it
    is neither.
    """
    repo = rec.get("nvfp4_repo")
    revision = rec.get("nvfp4_revision")
    seal = rec.get("seal_disclosure")
    scope = rec.get("scope_policy") or {}
    activations = rec.get("activations") or {}
    for name, value in (("nvfp4_revision", revision), ("seal_disclosure", seal),
                        ("scope_policy", scope), ("activations", activations)):
        if not value:
            raise Refuse(E_MISSING,
                         "%s is an %s receipt but carries no /%s. This family measures a "
                         "third-party unsealed artifact; a row without its scope and seal "
                         "disclosure would misrepresent what was measured."
                         % (os.path.basename(path), NVFP4_SUMMARY, name))
    if (not isinstance(revision, str) or len(revision) != 40
            or not all(c in "0123456789abcdef" for c in revision)):
        raise Refuse(E_IDENTITY,
                     "nvfp4_revision %r is not an immutable 40-hex repo commit. A community "
                     "snapshot measured at a moving ref cannot be pinned to a row." % (revision,))
    # Pinning these makes build_row's existing revision gate fire against the
    # registry's artifact record: a receipt that measured a different commit than
    # the row claims is exit 5, not a footnote.
    out["artifact_repo"] = repo
    out["artifact_revision"] = revision

    quantized = scope.get("quantized_scope")
    nonrouted = scope.get("nonrouted_policy")
    if not quantized or not nonrouted:
        raise Refuse(E_MISSING,
                     "scope_policy carries no quantized_scope/nonrouted_policy; the row cannot "
                     "state what part of the artifact the number covers.")
    out["verbatim_disclosure_coded"].append(
        {"code": "unsealed_source",
         "detail": "seal_disclosure (verbatim from the receipt): %s" % seal})
    out["verbatim_disclosure_coded"].append(
        {"code": "quantization_scope",
         "detail": "scope_policy (verbatim from the receipt): %s | %s" % (quantized, nonrouted)})
    # Only stated when it is TRUE of this artifact. A genuine W4A16 snapshot -- no
    # declared input_activations and no activation scale tensors in the index --
    # is fully captured by a weights-only decode and gets no caveat it has not
    # earned; the surface measures which case it is instead of assuming.
    captured = activations.get("weights_only_decode_captures_artifact_fully")
    if captured is None:
        raise Refuse(E_MISSING,
                     "activations block does not state weights_only_decode_captures_artifact_"
                     "fully; whether the number covers the whole artifact is exactly what this "
                     "disclosure exists to answer.")
    if not captured:
        detail = activations.get("disclosure")
        if not detail:
            raise Refuse(E_MISSING, "activations.disclosure is missing on a receipt that says "
                                    "the weights-only decode does NOT capture the artifact fully.")
        out["verbatim_disclosure_coded"].append(
            {"code": "activation_quantization_not_captured",
             "detail": "activations (verbatim from the receipt): %s" % detail})
    out["field_provenance"].update({
        "artifact_revision": "#/nvfp4_revision",
        "disclosures.quantization_scope": "#/scope_policy (measured from the artifact's own "
                                          "index by nvfp4_surface, not read off its README)",
        "disclosures.activation_quantization_not_captured":
            "#/activations (present only when #/activations/weights_only_decode_captures_"
            "artifact_fully is false)",
    })


def _apply_stream_verdict(out, summary, spath, verdicts):
    """Fold a streaming verdict into the adapted row, recomputing everything it asserts."""
    if len(verdicts) != 1:
        raise Refuse(E_MISSING, "expected at most one %s receipt, got %d" % (STREAM_VERDICT,
                                                                            len(verdicts)))
    v, vpath, _ = verdicts[0]
    stream = _need(v, "/stream_mean_kld", vpath)
    if not L.close(stream, out["value"]):
        raise Refuse(E_INCONSISTENT,
                     "the verdict says the streaming mean is %r and %s says %r; they are not the "
                     "same measurement." % (stream, os.path.basename(spath), out["value"]))
    for field in ("run_means", "distinct_tokenwise_kld_sha256"):
        if field in v and field in summary and list(v[field]) != list(summary[field]):
            raise Refuse(E_INCONSISTENT, "the verdict and the summary disagree on %s" % field)

    windows = _need(v, "/per_window", vpath)
    got = sum(w["stream_mean"] for w in windows) / len(windows)
    if not L.close(got, stream, rel=1e-9):
        raise Refuse(E_INCONSISTENT,
                     "the verdict's %d per-window streaming means average to %r, but it declares a "
                     "streaming mean of %r. Either the windows are not equally weighted or one of "
                     "the two numbers is wrong." % (len(windows), got, stream))
    for w in windows:
        if not L.close(w["stream_mean"] - w["sealed_mean"], w["delta"], rel=1e-6, abs_=1e-18):
            raise Refuse(E_INCONSISTENT, "window %s: stream_mean - sealed_mean != delta"
                         % w.get("window_id"))
    max_abs = max(abs(w["delta"]) for w in windows)
    if "max_abs_per_window_delta" in v and not L.close(v["max_abs_per_window_delta"], max_abs):
        raise Refuse(E_INCONSISTENT, "max_abs_per_window_delta %r but recomputing over per_window "
                                     "gives %r" % (v["max_abs_per_window_delta"], max_abs))
    sealed = _need(v, "/sealed_mean_kld", vpath)
    delta = stream - sealed
    if "delta_mean_kld" in v and not L.close(v["delta_mean_kld"], delta):
        raise Refuse(E_INCONSISTENT, "delta_mean_kld %r but stream_mean_kld - sealed_mean_kld is %r"
                     % (v["delta_mean_kld"], delta))
    if "abs_delta_mean_kld" in v and not L.close(v["abs_delta_mean_kld"], abs(delta)):
        raise Refuse(E_INCONSISTENT, "abs_delta_mean_kld disagrees with |delta_mean_kld|")

    out["contexts"] = len(windows)
    out["top1"] = v.get("top1_agreement")
    out["stream_bridge"] = {
        "stream_mean_kld": stream, "sealed_mean_kld": sealed, "delta_mean_kld": delta,
        "max_abs_per_window_delta": max_abs, "windows_compared": len(windows),
        "tokenwise_kld_sha256_matches_sealed": v.get("tokenwise_kld_sha256_matches_sealed"),
        "publishable_as_reproduction": v.get("publishable_as_reproduction"),
        "scored_the_sealed_surface": v.get("scored_the_sealed_k6_surface"),
        "sealed_top1_agreement": v.get("sealed_top1_agreement"),
        "verdict": v.get("verdict"),
        "checkpoint_identity_sha256": v.get("student_checkpoint_identity_sha256"),
        "sealed_checkpoint_identity_sha256": v.get("sealed_checkpoint_identity_sha256"),
    }
    out["field_provenance"].update({
        "contexts": "#/per_window[] (length), verdict receipt",
        "top1": "#/top1_agreement, verdict receipt",
        "bias.estimated_magnitude": "RECOMPUTED as #/stream_mean_kld - #/sealed_mean_kld, verdict "
                                    "receipt; checked against its own #/delta_mean_kld",
    })


def adapt_report(receipt, path, position_selector=None):
    """F4: shared-head replay reports (GLM suite and the Qwen ladder)."""
    cmp_ = receipt.get("comparator") or {}
    sw = receipt.get("scored_position_window") or {}
    ref_rev = ((receipt.get("reference_identity") or {}).get("model_revision"))
    sel = "all"
    if sw.get("windowed"):
        sel = "score_from:%s" % sw.get("score_from")
    if position_selector and position_selector != sel:
        raise Refuse(E_IDENTITY, "--position-selector %r but the receipt declares %r"
                     % (position_selector, sel))
    cb = receipt.get("context_bootstrap") or {}
    return {
        "value": _need(receipt, "/token_mean_kld", path),
        "metric_name": "mean_tokenwise_kld", "direction": "reference_to_candidate",
        "accumulation": _acc(cmp_.get("accumulation")),
        "two_pass": cmp_.get("two_pass"), "vocab_chunk": cmp_.get("vocab_chunk"),
        "top1": receipt.get("top1_agreement"),
        "scored_positions": receipt.get("scored_positions"), "contexts": receipt.get("contexts"),
        "ci": ((cb.get("ci95_low"), cb.get("ci95_high")) if cb.get("ci95_low") is not None else None),
        "clusters": cb.get("clusters"), "samples": cb.get("samples"),
        "runs": 1, "identical": None, "evidence_kind": "none", "evidence_hashes": [],
        "position_selector": sel,
        "reference_revision": ref_rev,
        "reference_revision_source": (receipt.get("reference_identity") or {}).get("model_revision_source"),
        "head_sha256": receipt.get("head_sha256"),
        "candidate_head": receipt.get("candidate_head"),
        "panel_digest": receipt.get("suite_token_sha256") or
                        ((receipt.get("suite") or {}).get("parent") or {}).get("manifest_sha256"),
        "field_provenance": {"value": "#/token_mean_kld", "top1": "#/top1_agreement",
                             "accumulation": "#/comparator/accumulation",
                             "scored_positions": "#/scored_positions",
                             "ci": "#/context_bootstrap/ci95_low..ci95_high"},
        "receipt_schema": receipt.get("schema"),
        "stack_relation": "same_stack",
        "head_policy": "shared_reference_head" if receipt.get("head_sha256") else "native_head",
    }


def adapt_crosscheck(receipt, path):
    return {
        "value": _need(receipt, "/mean_kld", path), "metric_name": "mean_tokenwise_kld",
        "direction": _dir(_need(receipt, "/direction", path)),
        "direction_source_text": receipt.get("direction"),
        "accumulation": "float64", "top1": receipt.get("top1_agreement"),
        "scored_positions": receipt.get("positions"), "contexts": receipt.get("windows"),
        "runs": 1, "identical": None, "evidence_kind": "none", "evidence_hashes": [],
        "offset_audit": receipt.get("offset_audit_mean_top1"),
        "field_provenance": {"value": "#/mean_kld", "top1": "#/top1_agreement",
                             "scored_positions": "#/positions", "contexts": "#/windows"},
        "receipt_schema": receipt.get("schema"),
        "stack_relation": "cross_stack", "head_policy": "native_head",
    }


def adapt_foreign(receipt, path):
    sch = receipt.get("schema")
    if sch == FOREIGN_REPEATED:
        runs = _need(receipt, "/runs", path)
        means = [x["mean_kld"] for x in runs]
        value = _need(receipt, "/mean_of_run_means", path)
        if not L.close(value, sum(means) / len(means)):
            raise Refuse(E_INCONSISTENT, "mean_of_run_means disagrees with runs[]")
        sd = receipt.get("population_stddev_of_run_means")
        if sd is not None and not L.close(sd, L.population_stddev(means)):
            raise Refuse(E_INCONSISTENT, "population_stddev_of_run_means %r but recomputing gives %r"
                         % (sd, L.population_stddev(means)))
        digests = sorted({x.get("tokenwise_kld_sha256") for x in runs if x.get("tokenwise_kld_sha256")})
        identical = len(digests) == 1
        pos = {x.get("prediction_positions") for x in runs}
        return {
            "value": value, "metric_name": "mean_of_run_means_tokenwise_kld",
            "direction": "reference_to_candidate",
            # This receipt family carries no compute_dtype. It is somebody else's
            # estimator; asserting float64 on their behalf would be inventing the one
            # field that is load-bearing for the comparability key. Record what the
            # receipt says (nothing) and let --accumulation supply it explicitly.
            "accumulation": _acc_optional(receipt.get("compute_dtype")),
            "top1": receipt.get("mean_top1_agreement"),
            "scored_positions": (pos.pop() if len(pos) == 1 else None),
            "contexts": 1, "runs": len(runs), "run_means": means, "cold": True,
            "identical": identical and len(runs) >= 2,
            "evidence_kind": "tokenwise_kld_sha256" if digests else "run_mean_equality_only",
            "evidence_hashes": digests if identical else None,
            "det_note": ("%d distinct per-run tokenwise digests" % len(digests)) if not identical else None,
            "gate": {"metric": "mean_tokenwise_kld", "threshold_lt": 0.06, "threshold_gt": None,
                     "passed": bool(receipt.get("all_quality_gates_passed"))},
            "regime": receipt.get("regime"),
            "field_provenance": {"value": "#/mean_of_run_means", "top1": "#/mean_top1_agreement",
                                 "run_means": "#/runs[]/mean_kld"},
            "receipt_schema": sch, "stack_relation": "same_stack", "head_policy": "native_head",
        }
    summary = receipt.get("summary") or {}
    return {
        "value": _need(receipt, "/summary/mean", path), "metric_name": "mean_tokenwise_kld",
        "direction": "reference_to_candidate", "accumulation": _acc(receipt.get("compute_dtype")),
        "top1": receipt.get("top1_agreement"),
        "scored_positions": receipt.get("prediction_positions"), "contexts": 1,
        "runs": 1, "identical": None, "evidence_kind": "none", "evidence_hashes": [],
        "aux": {"median_kld": summary.get("p50"), "p95_kld": summary.get("p95"),
                "p99_kld": summary.get("p99"), "max_kld": summary.get("max")},
        "gate": {"metric": "mean_tokenwise_kld", "threshold_lt": receipt.get("max_mean_kld"),
                 "threshold_gt": None, "passed": bool(receipt.get("mean_kld_gate_passed"))},
        "token_ids_sha256": receipt.get("tokens_sha256") or receipt.get("token_ids_sha256"),
        "field_provenance": {"value": "#/summary/mean", "top1": "#/top1_agreement",
                             "scored_positions": "#/prediction_positions"},
        "receipt_schema": sch, "stack_relation": "same_stack", "head_policy": "native_head",
    }


def _dir(text):
    """Map a receipt's free-text KL direction onto the registry enum.

    `direction` is one of the seven comparability-key inputs, and KL(P||Q) != KL(Q||P), so
    a wrong answer here files the row in the wrong ranked table. The old test asked
    "teacher_to_student first, student_to_teacher second", which made the answer depend on
    substring ORDER: a string naming both tokens returned whichever was tested first, and
    the result was not even symmetric in the two operands. Text that names both is
    ambiguous by construction and now refuses instead of picking."""
    t = (text or "").lower()
    fwd = ("teacher_to_student" in t or t.startswith("kld(brandonmusic_teacher")
           or "bf16_teacher_to" in t)
    rev = "student_to_teacher" in t
    if fwd and rev:
        raise Refuse(E_MISSING,
                     "the receipt's KL direction %r names BOTH teacher_to_student and "
                     "student_to_teacher. Which one the number is cannot be decided by reading "
                     "it." % text,
                     "supply --direction explicitly")
    if fwd:
        return "reference_to_candidate"
    if rev:
        return "candidate_to_reference"
    raise Refuse(E_MISSING, "cannot map the receipt's KL direction %r onto the registry's enum without "
                            "guessing. Supply --direction explicitly." % text)


def _acc(text):
    t = (text or "").lower()
    if t in ("float64", "fp64", "double"):
        return "float64"
    if t in ("float32", "fp32"):
        return "float32"
    if not t:
        raise Refuse(E_MISSING, "the receipt does not state its accumulation dtype; supply "
                                "--accumulation explicitly rather than assuming float64.")
    return "mixed"


def _acc_optional(text):
    """Like _acc, but a silent receipt yields None instead of refusing.

    None means "the receipt does not say". The caller must then either be given
    --accumulation, or the row is built with accumulation_dtype 'unknown' -- which is
    an honest statement about a third party's estimator and keeps such rows in their
    own comparability group instead of merging them with float64-attested ones.
    """
    return _acc(text) if text else None


def _stream_bias(adapted, lane, is_floor=False, floor_ref=None, floor_value=None):
    """The bias block for a non-sealed-lane row.

    A lane offset is not a cross-stack capture replay and must not be filed as one: the
    reference and the candidate came off the same runtime, and the offset here is the
    routed-expert combine, measured against the sealed lane's own number rather than
    bounded by a floor row. `estimated_magnitude` is that measured delta -- the only
    place in this file where the field is a measurement instead of a null.

    `is_floor` marks a row whose artifact IS the reference's own unquantized weights
    (build_row's test: --artifact equals the reference's artifact_ref), replayed through
    this SAME streaming pipeline. Such a row is the lane's own zero-point, not a
    quantization result, and never carries a floor_measurement_ref of its own.

    `floor_ref` / `floor_value` let a DIFFERENT row on the same lane name that zero-point.
    build_row has already checked, before calling this, that the floor was measured on
    the SAME lane as this row (E_IDENTITY otherwise) -- this function only ever renders
    the netted-out estimate it is handed, it does not itself decide whether the pairing
    is legal. The estimate is prose, not a new field: KL is not additive, so "value minus
    floor" is recorded as words a reader can question, the same way the cross-stack floor's
    "naive difference" already is (see adapt_crosscheck's callers) -- never as a number the
    schema asserts is exact.

    Returns None when the lane is the sealed one, or when the row is not from a lane-
    bearing receipt family at all.
    """
    if not lane or lane == "sealed-ep8":
        return None
    bridge = adapted.get("stream_bridge") or {}
    delta = bridge.get("delta_mean_kld")
    no_bridge_text = (
        "The lane's offset against the sealed-ep8 lane is NOT measured for this artifact: "
        "no sealed-lane counterpart to this profile exists to bridge against."
        if is_floor else
        "Measured on the %r lane, whose offset against the sealed-ep8 lane is known to be "
        "non-zero but was NOT measured for this artifact: no sealed-lane row for it exists "
        "to bridge against. The lane offset measured for a sibling artifact on this panel "
        "is not transferable -- it is a property of the routing, not a constant." % lane)
    floor_sentence = ""
    if floor_ref is not None and floor_value is not None:
        attributable = adapted["value"] - floor_value
        floor_sentence = (
            " This lane's own measurement floor (%s) is %r nats; netting it out gives an "
            "estimated quantization-attributable error of %r nats here -- an estimate, not "
            "an identity, because KL is not additive, and it is only meaningful because both "
            "terms are small and share the same reference and lane."
            % (floor_ref, floor_value, attributable))
    if is_floor:
        detail = (
            "THIS ROW IS THE FLOOR for the %r lane: it replays the reference's own "
            "unquantized weights through the SAME streaming harness that scored every other "
            "row on this pipeline, so its divergence against the stored teacher logits is "
            "the lane's zero-point, not a quantization result. It is NOT the cross-stack "
            "floor recorded elsewhere in this registry (a different pipeline, a different "
            "lane, a different comparability key) and is never interchangeable with it: "
            "subtracting one lane's floor from another lane's row is exactly the mistake "
            "BIAS-006 exists to catch." % lane)
        if delta is None:
            return {"kind": "other", "direction": "unknown", "floor_measurement_ref": None,
                    "estimated_magnitude": None, "detail": detail + " " + no_bridge_text}
        reproduces = bridge.get("tokenwise_kld_sha256_matches_sealed")
        detail += (" Bridge to the sealed-ep8 lane, measured: signed delta %r nats against %r "
                  "(|max| %r on any one of %d windows); tokenwise KL array %s the sealed one."
                  % (delta, bridge.get("sealed_mean_kld"), bridge.get("max_abs_per_window_delta"),
                     bridge.get("windows_compared"), "matches" if reproduces else "does NOT match"))
        return {"kind": "other",
                "direction": "downward" if delta < 0 else ("upward" if delta > 0 else "unknown"),
                "floor_measurement_ref": None, "estimated_magnitude": abs(delta), "detail": detail}
    if delta is None:
        return {"kind": "other", "direction": "unknown", "floor_measurement_ref": floor_ref,
                "estimated_magnitude": None, "detail": no_bridge_text + floor_sentence}
    reproduces = bridge.get("tokenwise_kld_sha256_matches_sealed")
    return {
        "kind": "other",
        "direction": "downward" if delta < 0 else ("upward" if delta > 0 else "unknown"),
        "floor_measurement_ref": floor_ref,
        "estimated_magnitude": abs(delta),
        "detail": ("Lane offset, MEASURED not estimated: this %r-lane run scores %r against the "
                  "sealed-ep8 lane's %r on the same panel, a signed delta of %r nats "
                  "(|max| %r on any one of %d windows). The tokenwise KL array %s the sealed "
                  "one, and the runner's own verdict is publishable_as_reproduction=%r, so this "
                  "number stands beside the sealed one rather than replacing it."
                  % (lane, bridge.get("stream_mean_kld"), bridge.get("sealed_mean_kld"), delta,
                     bridge.get("max_abs_per_window_delta"), bridge.get("windows_compared"),
                     "matches" if reproduces else "does NOT match",
                     bridge.get("publishable_as_reproduction"))) + floor_sentence,
    }


def _gate(receipt):
    """The receipt's own quality gate, or None.

    `bool(None)` is False, so a receipt that stated a `quality_gate` block but no verdict
    published `passed: false` -- a failed gate the receipt never asserted. This module's
    contract is that nothing is defaulted or inferred into existence; silence about a
    verdict is not a negative verdict, it is a refusal."""
    q = receipt.get("quality_gate")
    if not isinstance(q, dict):
        return None
    verdict = None
    for key in ("quality_gate_passed", "qualified"):
        if receipt.get(key) is not None:
            verdict = bool(receipt[key])
            break
    if verdict is None:
        raise Refuse(E_MISSING,
                     "the receipt states a quality_gate (%r) but no verdict: neither "
                     "quality_gate_passed nor qualified is present. A missing verdict is not a "
                     "failed gate." % (q.get("metric") or "unnamed"),
                     "add quality_gate_passed to the receipt, or drop the quality_gate block")
    return {"metric": q.get("metric"), "threshold_lt": q.get("threshold_lt"),
            "threshold_gt": q.get("threshold_gt"), "passed": verdict}


# --- row assembly -----------------------------------------------------------

def build_row(args, adapted, receipt_sources, registry):
    panels = registry["panels"]
    refs = registry["references"]
    arts = registry["artifacts"]

    for name, coll, val in (("--artifact", arts, args.artifact), ("--panel", panels, args.panel),
                            ("--reference", refs, args.reference)):
        if val not in coll:
            raise Refuse(E_MISSING, "%s %r does not exist in the registry. Declare the record first; "
                                    "this tool will not create one from a measurement receipt."
                         % (name, val))
    art, pan, ref = arts[args.artifact], panels[args.panel], refs[args.reference]

    # A receipt family that does not state its scope leaves these None; the flags then
    # supply them and `field_provenance` says so. A flag may not quietly restate what a
    # receipt already carries as something else -- that is exit 7, not a merge.
    for flag_name, flag_val, key_name in (("--scored-positions", args.scored_positions,
                                           "scored_positions"),
                                          ("--contexts", args.contexts, "contexts")):
        stated = adapted.get(key_name)
        if flag_val is None:
            continue
        if stated is not None and stated != flag_val:
            raise Refuse(E_IDENTITY,
                         "%s %r contradicts the receipt, which carries %r for %s."
                         % (flag_name, flag_val, stated, key_name))
        if stated is None:
            adapted[key_name] = flag_val
            adapted.setdefault("field_provenance", {})[key_name] = (
                "SUPPLIED by %s; this receipt family does not carry it" % flag_name)

    if ref.get("panel_ref") != args.panel:
        raise Refuse(E_IDENTITY, "reference %s was captured on panel %s, not %s"
                     % (args.reference, ref.get("panel_ref"), args.panel))

    pd = adapted.get("panel_digest")
    if pd:
        ident = pan.get("identity") or {}
        known = {ident.get("panel_token_sha256"), ident.get("panel_receipt_sha256"),
                 ident.get("manifest_sha256")} | set((ident.get("shard_token_sha256") or {}).values())
        if pd not in known:
            raise Refuse(E_IDENTITY,
                         "the receipt pins panel digest %s, which panel %s does not carry. Either the "
                         "wrong --panel was given or this is a different panel."
                         % (pd[:16] + "...", args.panel))
    # The teacher is half the identity of a fidelity number. REFC-005 checks this on the
    # submission path; a receipt that names its teacher deserves the same check here,
    # because a number measured against a different capture is a different quantity.
    td = adapted.get("teacher_digest")
    known_teacher = (ref.get("capture") or {}).get("capture_receipt_sha256")
    if td and known_teacher and td != known_teacher:
        raise Refuse(E_IDENTITY,
                     "the receipt was scored against teacher capture %s..., but reference %s is the "
                     "capture %s.... A number measured against a different teacher cannot share a "
                     "table with rows measured against this one."
                     % (td[:16], args.reference, known_teacher[:16]))

    sp = adapted.get("scored_positions")
    total = (pan.get("structure") or {}).get("scored_positions_total")
    if adapted.get("artifact_revision"):
        have = (art.get("huggingface") or {}).get("revision")
        if have and have != adapted["artifact_revision"]:
            raise Refuse(E_IDENTITY, "the receipt pins artifact revision %s, the registry record has %s"
                         % (adapted["artifact_revision"], have))

    schema = adapted.get("receipt_schema")
    if args.attribution == "self-measured" and schema not in OWN_SCHEMAS:
        raise Refuse(E_ATTRIB, "receipt family %r is not one this registry produces, so a row derived "
                               "from it cannot be marked self-measured." % schema)
    owner = ((art.get("huggingface") or {}).get("repository") or "/").split("/")[0]
    if args.attribution == "self-measured" and owner and owner != L.MAINTAINER and not args.third_party_artifact:
        raise Refuse(E_ATTRIB, "artifact %s belongs to %r, not to the registry maintainer. A row that "
                               "measures someone else's weights must pass --third-party-artifact so the "
                               "table can say whose artifact it is." % (args.artifact, owner))
    if args.attribution != "self-measured" and not (args.reported_by and args.source_url):
        raise Refuse(E_MISSING, "a %s row requires --reported-by and --source-url" % args.attribution)

    # Resolve coverage instead of asserting it. When both counts are known the answer is
    # arithmetic and no flag is needed; when either is unknown the operator must say,
    # because a full-panel claim against a panel of unknown size cannot be checked by
    # anyone -- and that is exactly the row SCOPE-007 lets through today, since it skips
    # when scored_positions_total is null.
    covers = args.covers_full_panel
    if covers is None:
        if sp and total:
            # Derivable, so no flag is needed -- and a count that does not match the panel
            # is a WRONG PANEL, not an undeclared subset. Saying "you forgot
            # --subset-detail" to someone who filed a single-window receipt under the
            # 25-window panel points them away from their actual mistake.
            if sp != total:
                raise Refuse(E_IDENTITY,
                             "the receipt scored %d positions but panel %s has %d. Use the panel "
                             "record that matches, or pass --no-covers-full-panel with "
                             "--subset-detail if this is deliberately a subset."
                             % (sp, args.panel, total))
            covers = True
        else:
            raise Refuse(E_MISSING,
                         "panel %s does not declare scored_positions_total (or the receipt does not "
                         "declare scored_positions), so whether this row covers the whole panel "
                         "cannot be derived -- and a full-panel claim against a panel of unknown "
                         "size is unverifiable by anyone." % args.panel,
                         "pass --covers-full-panel, or --no-covers-full-panel with --subset-detail")
    elif covers and sp and total and sp != total:
        raise Refuse(E_IDENTITY,
                     "--covers-full-panel was passed, but the receipt scored %d positions and panel "
                     "%s has %d." % (sp, args.panel, total))
    if not covers and not args.subset_detail:
        raise Refuse(E_MISSING,
                     "--no-covers-full-panel was passed but no --subset-detail says which part of "
                     "panel %s this row covers (%s of %s positions)."
                     % (args.panel, sp if sp is not None else "?", total if total is not None else "?"),
                     "pass --subset-detail \"<which windows/positions>\"")
    args.covers_full_panel = covers

    if adapted.get("reference_revision") is None and schema in REPORT_FAMILIES:
        if not (args.reference_revision and args.reference_revision_evidence):
            raise Refuse(E_MISSING,
                         "the report records reference_identity.model_revision = null "
                         "(model_revision_source = %r). This tool will not write a null revision and "
                         "will not invent one."
                         % adapted.get("reference_revision_source"),
                         "pass --reference-revision <sha> --reference-revision-evidence <path|url>; "
                         "the row will record revision_source='operator_asserted'")

    if adapted.get("identical") and adapted.get("evidence_kind") not in (
            "tokenwise_kld_sha256", "logits_tensor_sha256", "hidden_state_tensor_sha256",
            "sealed_tokenwise_digest"):
        raise Refuse(E_INCONSISTENT, "determinism would be claimed on %r evidence, which cannot support it"
                     % adapted.get("evidence_kind"))
    if args.deterministic and not adapted.get("identical"):
        raise Refuse(E_INCONSISTENT, "--deterministic was passed but the receipt's own per-run digests do "
                                     "not support a bitwise-identical claim.")

    # A flag may SUPPLY what the receipt is silent about. It may never quietly
    # overrule what the receipt states -- that is how a cross_stack replay gets
    # relabelled same_stack and loses its bias block, and how a float32 estimator
    # gets promoted into the float64 comparability group. Overruling is allowed
    # only with an explicit --disclosure, which lands on the row for the reader.
    overridden = []
    for flag_name, flag_val, key_name in (("--stack-relation", args.stack_relation, "stack_relation"),
                                          ("--head-policy", args.head_policy, "head_policy"),
                                          ("--accumulation", args.accumulation, "accumulation"),
                                          ("--direction", args.direction, "direction"),
                                          ("--lane", args.lane, "lane")):
        stated = adapted.get(key_name)
        if flag_val is None or stated is None or flag_val == stated:
            continue
        if not args.disclosure:
            raise Refuse(E_VOID,
                         "%s %r contradicts the receipt, which states %r (%s). A flag may supply what "
                         "a receipt omits; it may not overrule what a receipt states."
                         % (flag_name, flag_val, stated, key_name),
                         "drop the flag to use the receipt's value, or pass --disclosure explaining "
                         "on what evidence the receipt's own value is being overruled.")
        overridden.append((key_name, stated, flag_val))

    stack = args.stack_relation or adapted.get("stack_relation")
    head = args.head_policy or adapted.get("head_policy")
    direction = args.direction or adapted.get("direction")
    if direction is None:
        raise Refuse(E_MISSING,
                     "this receipt does not state the KL direction and no --direction was given. "
                     "Direction is a comparability key input: KLD(teacher||student) and "
                     "KLD(student||teacher) are different numbers.",
                     "pass --direction reference_to_candidate or --direction candidate_to_reference")
    lane = args.lane or adapted.get("lane")
    if adapted.get("requires_lane") and not lane:
        raise Refuse(E_MISSING,
                     "this receipt family does not name the measurement lane it came from, and no "
                     "--lane was given. Lanes are not interchangeable: a non-sealed lane carries a "
                     "disclosed offset against the sealed lane on the same panel.",
                     "pass --lane sealed-ep8 | streaming | local-mps | local-cuda-budget | other")
    if lane and lane not in ("sealed-ep8", "streaming", "local-mps", "local-cuda-budget", "other"):
        raise Refuse(E_MISSING, "unknown lane %r" % lane)
    for flag_name, flag_val, key_name in (("--direction", args.direction, "direction"),
                                          ("--lane", args.lane, "lane")):
        if flag_val is not None and adapted.get(key_name) is None:
            adapted.setdefault("field_provenance", {})[key_name] = (
                "SUPPLIED by %s; this receipt family does not carry it" % flag_name)
    if stack == "cross_stack" and not (args.floor_measurement or args.floor_pending):
        raise Refuse(E_MISSING, "a cross-stack row must name its measurement floor "
                                "(--floor-measurement ID) or declare that none exists yet "
                                "(--floor-pending with --disclosure).")
    if args.floor_pending and not args.disclosure:
        raise Refuse(E_VOID, "--floor-pending requires --disclosure explaining why no floor exists.")

    # A floor is the zero-point for ONE lane. The comparability key carries no lane input
    # (PROV-012), so a row on one lane and a floor measured on another can share a key
    # without sharing a lane -- BIAS-002 (same key) is not enough to catch that. Checked
    # here, at write time, rather than only at validate time, because "refuse what it
    # cannot substantiate" should stop a bad row before it is ever written, not just flag
    # one that already was.
    floor_value = None
    if args.floor_measurement:
        floor_row = registry["measurements"].get(args.floor_measurement)
        if floor_row is None:
            raise Refuse(E_MISSING,
                         "--floor-measurement %r does not exist in the registry. A floor must be a "
                         "row already on file, not one invented for this occasion." % args.floor_measurement)
        floor_value = floor_row["metric"]["value"]
        floor_lane = (registry["pipelines"].get(floor_row.get("pipeline_ref")) or {}).get(
            "lane") or {}
        floor_lane_name = floor_lane.get("name") or "sealed-ep8"
        this_lane_name = lane or "sealed-ep8"
        if floor_lane_name != this_lane_name:
            raise Refuse(E_IDENTITY,
                         "--floor-measurement %s was measured on lane %r, but this row is on lane "
                         "%r. A floor measured on one lane is not the zero-point for a different "
                         "lane, even when the two rows share a comparability key: the key carries "
                         "no lane input." % (args.floor_measurement, floor_lane_name, this_lane_name))
    # The floor itself: this row's own artifact IS the reference's unquantized weights,
    # replayed through this same pipeline. Mirrors BIAS-004/005's structural test.
    is_floor = (args.artifact == ref.get("artifact_ref"))

    ki = {"panel_id": args.panel, "reference_id": args.reference,
          "metric_name": adapted["metric_name"], "direction": direction,
          # "unknown" when neither the receipt nor a flag states it -- never a guess.
          "accumulation_dtype": args.accumulation or adapted["accumulation"] or "unknown",
          "stack_relation": stack, "head_policy": head}
    key = L.comparability_key(ki)

    disclosures = []
    for text in (adapted.get("verbatim_disclosure") or []):
        disclosures.append({"code": "unsealed_source", "severity": "caveat", "detail": text,
                            "affects_comparability": True})
    # A receipt that discloses its own deviation in its own words: the code says which
    # deviation, the detail keeps the receipt's wording rather than a paraphrase of it.
    for d in (adapted.get("verbatim_disclosure_coded") or []):
        # The severity was hard-coded "caveat" and affects_comparability
        # hard-coded True, which is the right DEFAULT (an adapter that says
        # nothing is saying "this is a caveat") but wrong as a law: the TR3
        # family's `sealed_source_verified` is the good news -- the publisher
        # sealed the release and the measurement recomputed the seal -- and
        # stamping it a comparability-affecting caveat would put the opposite
        # of the truth on the row. An adapter that states either field wins.
        disclosures.append({
            "code": d["code"],
            "severity": d.get("severity", "caveat"),
            "detail": d["detail"],
            "affects_comparability": bool(d.get("affects_comparability", True))})
    if lane and lane != "sealed-ep8":
        bridge = adapted.get("stream_bridge") or {}
        measured = ("On this panel the lane's offset against the sealed lane IS measured: "
                    "%r nats on the mean (max %r on any one window over %d windows), and the "
                    "tokenwise KL array is NOT the sealed one, so the run is not a reproduction "
                    "of the sealed number."
                    % (bridge["delta_mean_kld"], bridge["max_abs_per_window_delta"],
                       bridge["windows_compared"])
                    if bridge.get("delta_mean_kld") is not None else
                    "The lane's offset against the sealed lane is NOT measured for this artifact: "
                    "no sealed-lane row for it exists to bridge against.")
        disclosures.append({
            "code": "non_sealed_lane", "severity": "caveat",
            "detail": "Produced by the %r lane, not the sealed-ep8 lane. %s" % (lane, measured),
            "affects_comparability": True})
    if args.attribution != "self-measured":
        disclosures.append({"code": "author_reported_only", "severity": "caveat",
                            "detail": "Measured and published by %s. We have not re-run it.%s"
                                      % (args.reported_by,
                                         (" Regime as published: %s" % adapted["regime"])
                                         if adapted.get("regime") else ""),
                            "affects_comparability": True})
    if stack == "cross_stack":
        disclosures.append({"code": "cross_stack_capture", "severity": "caveat",
                            "detail": "Reference and candidate logits were not produced by the same "
                                      "runtime and code path; the result carries a stack-difference term.",
                            "affects_comparability": True})
    if adapted.get("runs", 1) == 1:
        disclosures.append({"code": "single_run", "severity": "caveat",
                            "detail": "One pass; repeatability was not established.",
                            "affects_comparability": False})
    if args.third_party_artifact:
        disclosures.append({"code": "third_party_artifact_self_measured", "severity": "info",
                            "detail": "Someone else's weights, our measurement.",
                            "affects_comparability": False})
    # A FAILED gate stated only in /quality_gate/passed is a fact that survives in
    # the record and vanishes from every rendered disclosure list. seed_registry has
    # emitted this disclosure by hand since the runtime rows; the ingest path did
    # not, so an externally submitted failing row would read as clean. The gate is
    # a fact about the ARTIFACT, not a reason to hide the row -- and not a reason to
    # let it go unsaid either.
    _gate_block = adapted.get("gate") or {}
    if _gate_block.get("passed") is False:
        _thr = _gate_block.get("threshold_lt")
        disclosures.append({
            "code": "quality_gate_failed", "severity": "caveat",
            "detail": ("The gate this receipt declares (%s%s) did NOT pass: the measured "
                       "value is %r. Recorded because a failing gate is a fact about the "
                       "artifact, not a reason to hide the row."
                       % (_gate_block.get("metric") or "quality gate",
                          (" < %r" % _thr) if _thr is not None else "",
                          adapted.get("value"))),
            "affects_comparability": False})
    for key_name, stated, flag_val in overridden:
        disclosures.append({"code": "estimator_overridden", "severity": "caveat",
                            "detail": "%s is recorded as %r although the receipt states %r. "
                                      "Reason given at generation time: %s"
                                      % (key_name, flag_val, stated, args.disclosure),
                            "affects_comparability": True})
    if args.disclosure:
        disclosures.append({"code": args.disclosure_code, "severity": "caveat",
                            "detail": args.disclosure, "affects_comparability": True})
    if not disclosures:
        disclosures.append({"code": "no_known_deviations", "severity": "info",
                            "detail": "No deviation from this registry's default protocol is known "
                                      "for this row.", "affects_comparability": False})

    det = {"run_count": adapted.get("runs", 1), "cold_start_per_run": adapted.get("cold"),
           "identical_across_runs": adapted.get("identical"),
           "evidence_kind": adapted.get("evidence_kind", "none"),
           "evidence_hashes": adapted.get("evidence_hashes") or [],
           "distinct_evidence_hash_count": len(adapted.get("evidence_hashes") or [])
           if adapted.get("evidence_hashes") is not None else None}
    if adapted.get("run_means"):
        rm = adapted["run_means"]
        det.update({"run_means": rm, "min_run_mean": min(rm), "max_run_mean": max(rm),
                    "population_stddev_of_run_means": L.population_stddev(rm)})
    if adapted.get("det_note"):
        det["note"] = adapted["det_note"]

    sources = list(receipt_sources)
    if args.source_url:
        sources.append({"kind": "url", "uri": args.source_url})
    if args.reference_revision_evidence:
        sources.append({"kind": "url", "uri": args.reference_revision_evidence,
                        "note": "operator-supplied evidence for --reference-revision"})

    # Stack fingerprint (malaiwah.stack-fingerprint.v1): the digest of the
    # stack-fingerprint.json the run's harness wrote -- engine build,
    # enforce_eager/cudagraph state, attention backend, kernel knobs, env pins,
    # image digest, pip freeze.  Provenance-recorded ONLY for now: it does not
    # enter the comparability key or any invariant (whether two rows with
    # different fingerprints stay comparable needs more thought than a flag).
    fingerprint_sha = getattr(args, "stack_fingerprint_sha256", None)
    fingerprint_uri = getattr(args, "stack_fingerprint_uri", None)
    if fingerprint_sha is not None:
        fingerprint_sha = fingerprint_sha.strip().lower()
        if len(fingerprint_sha) != 64 or any(c not in "0123456789abcdef"
                                             for c in fingerprint_sha):
            raise Refuse(E_SCHEMA, "--stack-fingerprint-sha256 must be 64 lowercase hex "
                                   "chars, got %r" % args.stack_fingerprint_sha256)
        if fingerprint_uri:
            sources.append({"kind": "receipt_file", "uri": fingerprint_uri,
                            "sha256": fingerprint_sha, "note": "stack_fingerprint"})
    elif fingerprint_uri:
        raise Refuse(E_MISSING, "--stack-fingerprint-uri names a fingerprint file but no "
                                "--stack-fingerprint-sha256 pins its content; a pointer "
                                "without a digest is an anecdote.")

    ci = adapted.get("ci")
    lane_bias = _stream_bias(adapted, lane, is_floor=is_floor,
                             floor_ref=args.floor_measurement, floor_value=floor_value)
    row = {
        "schema_version": L.SCHEMA_VERSION,
        "id": args.id or _mint_id(args, ki, adapted),
        "status": "published", "supersedes": None,
        "model_ref": art["model_ref"], "artifact_ref": args.artifact, "panel_ref": args.panel,
        "reference_ref": args.reference, "pipeline_ref": args.pipeline,
        "scope_digest": art["scope_digest"],
        "metric": {"name": adapted["metric_name"], "value": adapted["value"], "units": "nats",
                   "direction": direction, "higher_is_better": False},
        "auxiliary_metrics": dict(adapted.get("aux") or {}, top1_agreement=adapted.get("top1")),
        "uncertainty": {"method": "context_cluster_bootstrap" if ci else "none",
                        "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None,
                        "clusters": adapted.get("clusters"), "samples": adapted.get("samples")},
        "estimator": {"accumulation_dtype": ki["accumulation_dtype"], "logits_dtype": "fp32",
                      "two_pass": adapted.get("two_pass"), "vocab_chunk": adapted.get("vocab_chunk"),
                      "stack_relation": stack, "head_policy": head},
        "determinism": det,
        "measurement_scope": {"scored_positions": adapted.get("scored_positions"),
                              "contexts": adapted.get("contexts"), "positions_per_context": None,
                              "covers_full_panel": bool(args.covers_full_panel),
                              "subset_detail": args.subset_detail,
                              "position_filter": adapted.get("position_selector", "all")},
        "provenance": {
            "measured_by": args.attribution,
            "measurer": ({"name": L.MAINTAINER, "role": "measurer", "handle": L.MAINTAINER,
                          "url": "https://huggingface.co/%s" % L.MAINTAINER,
                          "is_registry_maintainer": True}
                         if args.attribution == "self-measured" else
                         {"name": args.reported_by, "role": "measurer", "handle": args.reported_by,
                          "url": None, "is_registry_maintainer": False}),
            "independently_verified": False, "verification": None,
            "sources": sources, "receipt_schema": schema,
            # Present only when supplied: rows predating the fingerprint (and
            # reseeded historical rows) keep their exact shape.
            **({"stack_fingerprint_sha256": fingerprint_sha}
               if fingerprint_sha is not None else {})},
        "comparability": {"key": key, "key_inputs": ki,
                          "class": "strict" if (args.attribution == "self-measured"
                                                and stack == "same_stack"
                                                and pan.get("sealed")
                                                and not any(d.get("affects_comparability")
                                                            for d in disclosures)) else "advisory",
                          "bias": (lane_bias if lane_bias
                                   else {"kind": "cross_stack_capture_replay", "direction": "upward",
                                    "floor_measurement_ref": args.floor_measurement,
                                    "estimated_magnitude": None,
                                    "detail": args.disclosure or
                                              "Cross-stack replay; see the named floor."}
                                   if stack == "cross_stack" else None)},
        "quality_gate": adapted.get("gate"),
        "cross_refs": {"local_ai_registry": {"model_id": None, "model_instance_id": None,
                                             "url": None, "match_confidence": "unverified"}},
        "disclosures": disclosures,
    }
    # field_provenance is the reader's audit trail: every entry claims "this field
    # came from that JSON Pointer in the receipt". A field the operator overrode no
    # longer came from there, so its pointer is replaced rather than left to lie.
    fp = dict(adapted.get("field_provenance") or {})
    for key_name, _stated, flag_val in overridden:
        fp[key_name] = "OVERRIDDEN by flag (%r); receipt pointer no longer applies" % flag_val
    if fingerprint_sha is not None:
        fp["stack_fingerprint_sha256"] = ("SUPPLIED by --stack-fingerprint-sha256"
                                          + ("; file named by --stack-fingerprint-uri"
                                             if fingerprint_uri else
                                             "; no file pointer supplied"))
    row["notes"] = "field_provenance: " + json.dumps(fp, sort_keys=True)
    return row


def _mint_id(args, ki, adapted):
    """Content-addressed so re-running on the same receipt with the same flags is idempotent."""
    payload = L.canonical_json({"ki": ki, "artifact": args.artifact, "pipeline": args.pipeline,
                                "attribution": args.attribution, "value": adapted["value"]})
    return "measurement--auto." + L.sha256_hex(payload)[:16]


# --- CLI --------------------------------------------------------------------

def add_common(p):
    p.add_argument("--registry", default=L.repo_root(__file__))
    p.add_argument("--out", default=None)
    p.add_argument("--artifact", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--pipeline", required=True)
    p.add_argument("--id", default=None)
    p.add_argument("--attribution", default="self-measured",
                   choices=("self-measured", "author-reported", "third-party-reported"))
    p.add_argument("--reported-by", default=None)
    p.add_argument("--source-url", default=None)
    p.add_argument("--third-party-artifact", action="store_true")
    p.add_argument("--accumulation", default=None, choices=(None, "float64", "float32", "mixed"))
    p.add_argument("--direction", default=None,
                   choices=(None, "reference_to_candidate", "candidate_to_reference"),
                   help="supply the KL direction when the receipt does not state it")
    p.add_argument("--lane", default=None,
                   choices=(None, "sealed-ep8", "streaming", "local-mps", "local-cuda-budget",
                            "other"),
                   help="supply the measurement lane when the receipt does not name it")
    p.add_argument("--scored-positions", type=int, default=None,
                   help="supply the scored-position count when the receipt is a scalar summary")
    p.add_argument("--contexts", type=int, default=None,
                   help="supply the context count when the receipt is a scalar summary")
    p.add_argument("--stack-relation", default=None, choices=(None, "same_stack", "cross_stack"))
    p.add_argument("--head-policy", default=None,
                   choices=(None, "native_head", "shared_reference_head", "dequantized_head"))
    # `action="store_true", default=True` can never be False: the flag was inert and every
    # row this tool wrote claimed full-panel coverage, so --subset-detail produced a
    # self-contradictory row and the guard below ("... and args.covers_full_panel") was an
    # unconditional test wearing a conditional. Coverage is now INFERRED when it is
    # derivable -- scored_positions against the panel's own total -- and must be stated
    # otherwise, because a full-panel claim against a panel of unknown size is
    # unverifiable by construction.
    p.add_argument("--covers-full-panel", dest="covers_full_panel", default=None,
                   action="store_const", const=True,
                   help="assert full-panel coverage (only needed when the panel does not "
                        "declare scored_positions_total)")
    p.add_argument("--no-covers-full-panel", dest="covers_full_panel",
                   action="store_const", const=False,
                   help="declare a SUBSET measurement; requires --subset-detail")
    p.add_argument("--subset-detail", default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--disclosure", default=None)
    p.add_argument("--disclosure-code", default="record_note")
    p.add_argument("--floor-measurement", default=None)
    p.add_argument("--floor-pending", action="store_true")
    p.add_argument("--reference-revision", default=None)
    p.add_argument("--reference-revision-evidence", default=None)
    p.add_argument("--position-selector", default=None)
    p.add_argument("--stack-fingerprint-sha256", default=None,
                   help="sha256 of the run's stack-fingerprint.json "
                        "(malaiwah.stack-fingerprint.v1: engine build, enforce_eager/"
                        "cudagraph state, attention backend, kernel knobs, env pins, "
                        "image digest, pip freeze). Recorded in provenance; NOT part "
                        "of the comparability key yet.")
    p.add_argument("--stack-fingerprint-uri", default=None,
                   help="where that stack-fingerprint.json lives (recorded as a "
                        "receipt_file source with the digest above; requires "
                        "--stack-fingerprint-sha256)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")


def _identity_conflict(existing, generated):
    """Does a receipt CONTRADICT an already-catalogued artifact, or just say less?

    Returns a human-readable description of the first genuine contradiction, or
    None when the two records are compatible.

    Only fields that answer "what are these weights" count.  A receipt that
    omits our curated `derived_from_artifact_ref`, or spells the quantizer tool
    'exllamav3 EXL3' where we wrote 'exllamav3', or reports a slightly larger
    byte total because it summed all repo files rather than only the weight
    files, is not disagreeing with us about the artifact -- it simply carries
    less context than a curated row. Treating that as a collision would block
    every independent verification, which is the one thing this registry most
    wants to encourage.

    A DIFFERENT scope_digest, revision, codec family or bit width is a real
    contradiction: the same id would then name two different sets of weights,
    and no amount of merging can make that safe.
    """
    def dig(record, *path):
        cur = record
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    checks = (
        ("scope_digest", ("scope_digest",)),
        ("Hub revision", ("huggingface", "revision")),
        ("codec family", ("codec", "family")),
        ("bits per weight", ("codec", "bits_per_weight_nominal")),
    )
    for label, path in checks:
        old, new = dig(existing, *path), dig(generated, *path)
        # A receipt that says nothing about a field cannot contradict it.
        if new in (None, "", "unknown") or old in (None, "", "unknown"):
            continue
        if old != new:
            return "%s differs (catalogued %r, receipt %r)" % (label, old, new)
    return None


MEASUREMENT_IDENTITY = (
    ("metric", "the measured value"),
    ("panel_ref", "the panel"),
    ("reference_ref", "the reference capture"),
    ("artifact_ref", "the artifact measured"),
    ("estimator", "the estimator"),
    ("determinism", "the determinism evidence"),
    ("measurement_scope", "the measurement scope"),
)


def _measurement_conflict(old, new):
    """Does this receipt describe a DIFFERENT measurement than the catalogued row?

    `_identity_conflict` answers the artifact question -- are these the same weights --
    and every field it reads is absent from a measurement row, so it returned None for two
    rows reporting different numbers. This asks the measurement question. Provenance
    bookkeeping (`provenance.sources[].uri`, `measured_at`) is deliberately NOT compared:
    the same receipt ingested from a local path and from a CI checkout differs there, and
    refusing on it would break re-ingest."""
    for field, label in MEASUREMENT_IDENTITY:
        a, b = old.get(field), new.get(field)
        if L.canonical_json(a) != L.canonical_json(b):
            if field == "metric":
                return "%s differs (catalogued %r, receipt %r)" % (
                    label, (a or {}).get("value"), (b or {}).get("value"))
            return "%s differs" % label
    return None


def ingest_submissions(args):
    """The documented contributor path: sealed submission receipts -> registry records."""
    registry = L.load_registry(os.path.join(args.registry, "data"))
    generated = {"measurements": [], "artifacts": [], "pipelines": [], "receipts": []}
    rows, extras = [], []
    try:
        for path in args.submissions:
            sub, path, fsha = load_submission(path)
            row, new = submission_to_records(
                sub, path, fsha, registry,
                maintainer_attribution=bool(getattr(args, "maintainer_attribution", False)))
            rows.append(row)
            extras.extend(new)
            for rec in new:
                registry[L.collection_of_id(rec["id"])][rec["id"]] = rec
            generated["receipts"].append({"path": path, "sha256": fsha,
                                          "receipt_sha256": sub["receipt_sha256"]})
    except Refuse as exc:
        print("REFUSED (exit %d): %s" % (exc.code, exc), file=sys.stderr)
        if exc.remedy:
            print("  -> %s" % exc.remedy, file=sys.stderr)
        return exc.code

    for row in rows:
        generated["measurements"].append(row["id"])
    for rec in extras:
        generated[L.collection_of_id(rec["id"])].append(rec["id"])

    # A generated measurement id is handle.repo-tail.panel, so every quant in one HF repo
    # collapses to ONE id. `_identity_conflict` only inspects artifact-shaped fields
    # (scope_digest, revision, codec family, bpw) -- all absent from a measurement row --
    # so a DIFFERENT metric.value was never a contradiction: the second row was discarded
    # with "= kept existing record (receipt agrees)" and the tool then printed
    # "wrote <id> / value <NEW>" and exited 0. The operator, and the PR comment built from
    # --report, were told a number was recorded that was not.
    kept = set()
    if args.write:
        for coll, recs in (("measurements", rows),
                           ("artifacts", [r for r in extras if r["id"].startswith("artifact--")]),
                           ("pipelines", [r for r in extras if r["id"].startswith("pipeline--")])):
            if not recs:
                continue
            path = os.path.join(args.registry, "data", coll + ".jsonl")
            existing = {r["id"]: r for _, r, _ in L.read_jsonl(path)}
            for r in recs:
                if coll == "measurements" and r["id"] in existing:
                    contradiction = _measurement_conflict(existing[r["id"]], r)
                    if contradiction:
                        print("REFUSED (exit %d): %s already exists and this receipt reports a "
                              "different measurement: %s"
                              % (E_COLLISION, r["id"], contradiction), file=sys.stderr)
                        print("  -> the generated id is <handle>.<repo>.<panel>, so two quants from "
                              "one repository collide. Pass --id to declare a distinct row, or "
                              "name the artifact's path so the id can be disambiguated.",
                              file=sys.stderr)
                        return E_COLLISION
                    if L.canonical_json(existing[r["id"]]) != L.canonical_json(r):
                        # Same measurement, different bookkeeping (the receipt's own path
                        # differs between a local run and a CI checkout). Keep the
                        # catalogued row and SAY that nothing was written.
                        kept.add(r["id"])
                        print("  = kept existing record  %s (same measurement; only provenance "
                              "bookkeeping differs)" % r["id"])
                    continue
                if r["id"] in existing and L.canonical_json(existing[r["id"]]) != L.canonical_json(r):
                    # An independent measurement of an artifact we ALREADY
                    # catalogue is the flagship case for this registry, and the
                    # collision fires every time: a contributor's receipt
                    # describes the same weights more thinly than our curated
                    # row does (no derived_from, no seal note, no link to the
                    # producer's own receipt). Refusing on any textual
                    # difference makes independent verification impossible;
                    # letting the receipt overwrite would let a second
                    # measurement quietly delete the first one's provenance.
                    #
                    # So: refuse only on a CONTRADICTION about what the weights
                    # ARE, and otherwise keep the existing, richer record.
                    conflict = _identity_conflict(existing[r["id"]], r)
                    if conflict:
                        print("REFUSED (exit %d): %s already exists and this "
                              "receipt contradicts it: %s"
                              % (E_COLLISION, r["id"], conflict), file=sys.stderr)
                        print("  -> the same artifact cannot be two different "
                              "things. Check the revision and the quantization "
                              "scope in your receipt against the existing row.",
                              file=sys.stderr)
                        return E_COLLISION
                    print("  = kept existing record  %s "
                          "(receipt agrees; catalogued row is more complete)"
                          % r["id"])
                    continue
                existing[r["id"]] = r
            L.write_jsonl(path, list(existing.values()))
    if kept:
        generated["kept_existing"] = sorted(kept)
        generated["measurements"] = [m for m in generated["measurements"] if m not in kept]
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(generated, fh, indent=2, sort_keys=True)
    for row in rows:
        if row["id"] in kept:
            verb = "NOT WRITTEN (kept catalogued row) "
        else:
            verb = "wrote " if args.write else "would write "
        print("%s%s" % (verb, row["id"]))
        print("  value               %r %s" % (row["metric"]["value"], row["metric"]["units"]))
        print("  comparability key   %s  (class %s)"
              % (row["comparability"]["key"], row["comparability"]["class"]))
        print("  attribution         %s by %s" % (row["provenance"]["measured_by"],
                                                  row["provenance"]["measurer"]["name"]))
    for rec in extras:
        print("  + new record        %s" % rec["id"])
    if not args.write:
        print("\n(dry run: pass --write to record these)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, files in (("from-receipt", "--receipt"), ("from-report", "--report"),
                        ("from-crosscheck", "--report"), ("from-foreign", "--receipt")):
        p = sub.add_parser(name)
        p.add_argument(files, action="append", required=True, dest="receipts")
        add_common(p)
    sub.add_parser("schemas")
    sub.add_parser("offline-selftest")
    # The contributor path: a sealed submission receipt, no subcommand.
    # `registry_add.py --receipt R [--receipt R2 ...] [--write] [--report FILE]`
    ap.add_argument("--receipt", action="append", dest="submissions", default=None,
                    help="sealed submission receipt (quant-fidelity-registry/submission-receipt.v1)")
    ap.add_argument("--write", action="store_true", help="write the generated records into data/")
    ap.add_argument("--maintainer-attribution", action="store_true", dest="maintainer_attribution",
                    help="record a maintainer-identity submission as self-measured. This is an "
                         "OPERATOR assertion made at the command line on a machine you control; "
                         "it is deliberately not a field in the receipt, so no pull request and "
                         "no CI job can mint self-measured attribution. Without it a submission "
                         "claiming the maintainer's identity is recorded as third-party-reported "
                         "with an attribution_downgraded disclosure.")
    ap.add_argument("--report", default=None, help="write a JSON summary of what was generated")
    ap.add_argument("--registry", default=L.repo_root(__file__))
    args = ap.parse_args()

    if args.cmd is None and args.submissions:
        return ingest_submissions(args)

    if args.cmd == "schemas":
        print("receipt families this tool understands (dispatch is on the exact string):")
        for s in sorted(OWN_SCHEMAS):
            print("  OWN     %s" % s)
        for s in sorted(FOREIGN_SCHEMAS):
            print("  FOREIGN %s" % s)
        return 0
    if args.cmd == "offline-selftest":
        bad = sorted(m for m in sys.modules
                     if m.split(".")[0] in ("requests", "urllib", "http", "socket", "ssl",
                                            "huggingface_hub", "aiohttp", "httpx"))
        print("networking modules loaded: %s" % (bad or "none"))
        return 0 if not bad else 1
    if not args.cmd:
        ap.print_help()
        return 4

    try:
        loaded = [load_receipt(p) for p in args.receipts]
        schemas = {r[0].get("schema") for r in loaded}
        if args.cmd == "from-receipt":
            if schemas & set(STREAM_SUMMARIES):
                adapted = adapt_stream_summary(loaded)
            elif schemas & set(DIONE_SUMMARIES):
                adapted = adapt_dione(loaded[0][0], loaded[0][1])
            elif schemas & set(EXL3HF_SUMMARIES):
                adapted = adapt_turbo(loaded[0][0], loaded[0][1])
            elif schemas & set(TR3_SUMMARIES):
                adapted = adapt_tr3(loaded[0][0], loaded[0][1])
            elif STREAM_VERDICT in schemas:
                raise Refuse(E_MISSING,
                             "a %s receipt describes a summary; it carries no measurement of its "
                             "own. Pass the summary receipt alongside it." % STREAM_VERDICT)
            else:
                adapted = adapt_packed_and_five_run(loaded)
        elif args.cmd == "from-report":
            adapted = adapt_report(loaded[0][0], loaded[0][1], args.position_selector)
            if args.reference_revision:
                adapted["reference_revision"] = args.reference_revision
                adapted["reference_revision_source"] = "operator_asserted"
        elif args.cmd == "from-crosscheck":
            adapted = adapt_crosscheck(loaded[0][0], loaded[0][1])
        else:
            if loaded[0][0].get("schema") in OWN_SCHEMAS:
                raise Refuse(E_ATTRIB, "from-foreign was given one of OUR receipt families (%s). Use "
                                       "from-receipt / from-report instead."
                             % loaded[0][0].get("schema"))
            adapted = adapt_foreign(loaded[0][0], loaded[0][1])
            if args.attribution == "self-measured":
                raise Refuse(E_ATTRIB, "from-foreign rows are never self-measured.")
        sources = [{"kind": "receipt_file", "uri": p, "sha256": sha,
                    "note": r.get("schema")} for r, p, sha in loaded]
        registry = L.load_registry(os.path.join(args.registry, "data"))
        row = build_row(args, adapted, sources, registry)
    except Refuse as exc:
        print("REFUSED (exit %d): %s" % (exc.code, exc), file=sys.stderr)
        if exc.remedy:
            print("  -> %s" % exc.remedy, file=sys.stderr)
        return exc.code

    line = L.canonical_json(row)
    if args.dry_run:
        print(line if args.json else json.dumps(row, indent=2, sort_keys=True))
        return 0
    out = args.out or os.path.join(args.registry, "data", "measurements.jsonl")
    existing = {r["id"]: (r, ln) for _, r, ln in L.read_jsonl(out)}
    if row["id"] in existing:
        if existing[row["id"]][1] == line:
            print("unchanged: %s" % row["id"])
            return 0
        print("REFUSED (exit %d): id %s already exists with different content. Pass --id to declare a "
              "distinct row." % (E_COLLISION, row["id"]), file=sys.stderr)
        return E_COLLISION
    rows = [r for r, _ in existing.values()] + [row]
    L.write_jsonl(out, rows)
    print("wrote %s (%d rows) -> %s" % (row["id"], len(rows), out))
    print("  comparability key %s" % row["comparability"]["key"])
    print("  run tools/registry_validate.py and tools/registry_render.py next")
    return 0



# ===========================================================================
# Submission receipts (the contributor path documented in CONTRIBUTING.md)
# ===========================================================================
SUBMISSION_SCHEMA = "quant-fidelity-registry/submission-receipt.v1"


def _norm_path(value):
    """Compare artifact file selectors without tripping over a trailing slash.

    The registry stores MLX builds as directory prefixes ("2-bit/") and GGUF builds as
    bare filenames ("Qwen3.8-27B-Q8_0.gguf"), so a contributor writing "2-bit" must not be
    refused for a cosmetic difference."""
    if value is None:
        return None
    value = str(value).strip().strip("/")
    return value or None


def _claims_maintainer(measurer):
    """Which field, if any, claims the registry maintainer's identity.

    Returns `(field, value)` or None.  Case-folded, and it looks at every field a
    reader sees: `registry_render.badge()` prints `measurer.name`, so a submission
    with `name: malaiwah` and a throwaway handle rendered as "reported by malaiwah"
    in the published Attribution column while every handle-keyed check stayed
    silent.  The url is checked too, because a hf.co/<maintainer> link is a claim.
    """
    m = L.MAINTAINER.lower()
    for field in ("handle", "name"):
        value = (measurer.get(field) or "").strip()
        if value.lower() == m:
            return field, value
    url = (measurer.get("url") or "").strip().lower().rstrip("/")
    if url and ("huggingface.co/" + m) in url:
        return "url", (measurer.get("url") or "").strip()
    return None


def verify_seal(sub):
    """A submission seals itself: sha256 over its canonical form with receipt_sha256 blanked."""
    claimed = sub.get("receipt_sha256")
    if not claimed:
        raise Refuse(E_MISSING, "the submission carries no receipt_sha256")
    probe = dict(sub)
    probe["receipt_sha256"] = ""
    actual = L.sha256_hex(L.canonical_json(probe))
    if actual != claimed:
        raise Refuse(E_INCONSISTENT,
                     "the seal does not verify: the file was edited after the run.\n"
                     "  claimed    %s\n  recomputed %s" % (claimed, actual),
                     "re-run the measurement rather than patching the receipt")
    return actual


def load_submission(path):
    if not os.path.exists(path):
        raise Refuse(E_MISSING, "submission not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        try:
            sub = json.load(fh)
        except ValueError as exc:
            raise Refuse(E_MISSING, "submission is not valid JSON: %s (%s)" % (path, exc))
    if sub.get("submission_schema") != SUBMISSION_SCHEMA:
        raise Refuse(E_SCHEMA, "submission_schema is %r, expected %r"
                     % (sub.get("submission_schema"), SUBMISSION_SCHEMA))
    verify_seal(sub)
    want = L.scope_digest(sub["artifact"]["scope"])
    if sub["artifact"].get("scope_digest") != want:
        raise Refuse(E_INCONSISTENT,
                     "artifact.scope_digest does not match the scope it describes.\n"
                     "  declared   %s\n  recomputed %s" % (sub["artifact"].get("scope_digest"), want))
    det = sub.get("determinism") or {}
    rm = det.get("run_means")
    if rm:
        if len(rm) != det.get("run_count"):
            raise Refuse(E_INCONSISTENT, "run_means has %d entries but run_count is %s"
                         % (len(rm), det.get("run_count")))
        if not L.close(sub["metric"]["value"], sum(rm) / len(rm)) and \
                sub["metric"]["name"] == "mean_of_run_means_tokenwise_kld":
            raise Refuse(E_INCONSISTENT, "metric.value %r != mean(run_means) %r"
                         % (sub["metric"]["value"], sum(rm) / len(rm)))
    if det.get("identical_across_runs") and det.get("evidence_kind") not in (
            "tokenwise_kld_sha256", "logits_tensor_sha256", "hidden_state_tensor_sha256",
            "sealed_tokenwise_digest"):
        raise Refuse(E_INCONSISTENT,
                     "identical_across_runs is claimed on evidence_kind=%r. Only a tensor-content "
                     "digest can support that: report files embed run indices, paths and timings and "
                     "differ across bit-identical runs." % det.get("evidence_kind"))
    return sub, path, L.sha256_file(path)


def _slug(text):
    out = []
    for ch in (text or "").lower():
        out.append(ch if (ch.isalnum() or ch in ".-") else "-")
    s = "".join(out).strip("-.")
    while "--" in s:
        s = s.replace("--", "-")
    return s or "unknown"


def submission_to_records(sub, path, fsha, registry, strict_new=False,
                          maintainer_attribution=False):
    """Return (measurement_row, new_records) -- new artifact/pipeline records the
    submission implies. Panels and references must already exist: a contributor cannot
    introduce a panel through a measurement (CONTRIBUTING.md section 6)."""
    art_in = sub["artifact"]
    pan_ref = sub["panel"]["panel_ref"]
    ref_ref = sub["reference"]["reference_ref"]

    # A submission may not mint a row attributed to the registry maintainer on the strength
    # of anything INSIDE the submission. `measured_by: self-measured` is the registry's
    # highest trust level and `class: strict` follows from it; the old gate compared two
    # strings the submitter types (measurer.handle and produced_by.repository) against each
    # other, so anyone who wrote "malaiwah" in both fields minted a strict, self-measured row
    # in the flagship comparability group. PROV-008 cannot catch it either, because the
    # ingest mints the pipeline record from the same handle it is meant to check against.
    #
    # Two changes. The identity CLAIM is now recognised however it is spelled -- handle, name
    # or url, case-folded -- so `name: malaiwah` with a throwaway handle no longer renders as
    # "reported by malaiwah" in the published Attribution column. And the claim is only
    # HONOURED when the operator asserts it at the command line (`--maintainer-attribution`),
    # which is a property of the invocation, not of the file: CI never passes it, so no pull
    # request can mint self-measured no matter what it contains. An unasserted claim is
    # DOWNGRADED rather than refused -- the file may be perfectly honest, and the safe
    # direction is to understate provenance -- and the downgrade is disclosed on the row.
    claim = _claims_maintainer(sub.get("measurer") or {})
    attribution_downgraded = None
    if claim:
        pb_repo = (sub.get("produced_by") or {}).get("repository") or ""
        if not pb_repo.lower().startswith(L.MAINTAINER.lower() + "/"):
            raise Refuse(E_ATTRIB,
                         "this submission claims the registry maintainer's identity (measurer.%s=%r) "
                         "but produced_by.repository is %r, which is not a repository of theirs. A row "
                         "attributed to us is a row that ran on our stack; a submission cannot assert "
                         "that on our behalf."
                         % (claim[0], claim[1], pb_repo or None),
                         "set measurer.name/handle/url to your own identity; your row will be "
                         "recorded as author-reported or third-party-reported and credited to you")
        if not maintainer_attribution:
            attribution_downgraded = (
                "the submission claims the registry maintainer's identity (measurer.%s=%r), but "
                "self-measured attribution is asserted by the operator at ingest, never by the "
                "receipt: recorded as third-party-reported. Re-run registry_add with "
                "--maintainer-attribution on a machine you control to record it as self-measured."
                % (claim[0], claim[1]))

    if pan_ref not in registry["panels"]:
        raise Refuse(E_MISSING,
                     "panel %s is not in the registry. A measurement cannot introduce a panel; open a "
                     "'panel: <name>' discussion with its token digest, context and position counts, "
                     "scoring window and tokenizer first." % pan_ref)
    if ref_ref not in registry["references"]:
        raise Refuse(E_MISSING, "reference %s is not in the registry." % ref_ref)
    pan, ref = registry["panels"][pan_ref], registry["references"][ref_ref]
    if ref.get("panel_ref") != pan_ref:
        raise Refuse(E_IDENTITY, "reference %s was captured on panel %s, not %s"
                     % (ref_ref, ref.get("panel_ref"), pan_ref))

    ident = pan.get("identity") or {}
    declared = sub["panel"].get("panel_token_sha256")
    if declared and declared != ident.get("panel_token_sha256"):
        raise Refuse(E_IDENTITY,
                     "the submission pins panel token digest %s but panel %s carries %s. Either the "
                     "wrong panel_ref was named or these are different token sets."
                     % (declared[:16] + "...", pan_ref,
                        (ident.get("panel_token_sha256") or "none")[:16] + "..."))
    # The teacher is half the identity of a fidelity number, and the submission is REQUIRED
    # to declare the capture it scored against. Here that declaration is actually checked
    # against the reference record, instead of being carried along unread.
    known_teacher = (ref.get("capture") or {}).get("capture_receipt_sha256")
    declared_teacher = sub["reference"].get("teacher_receipt_sha256")
    teacher_unverified = not known_teacher
    if known_teacher and declared_teacher != known_teacher:
        raise Refuse(E_IDENTITY,
                     "the submission was scored against teacher capture %s, but reference %s is the "
                     "capture %s. A number measured against a different teacher is a different "
                     "quantity: it cannot share a table with rows measured against this one."
                     % (declared_teacher[:16] + "...", ref_ref, known_teacher[:16] + "..."),
                     "name the reference your teacher actually is, or open a 'reference: <name>' "
                     "discussion to register the capture you used")

    total = (pan.get("structure") or {}).get("scored_positions_total")
    ms = sub["measurement_scope"]
    if ms.get("covers_full_panel") and total and ms.get("scored_positions") != total:
        raise Refuse(E_IDENTITY, "covers_full_panel is true but %s of %s positions were scored"
                     % (ms.get("scored_positions"), total))
    if total and (ms.get("scored_positions") or 0) > total:
        raise Refuse(E_IDENTITY,
                     "this row scores %s positions but panel %s only has %s. Whatever was scored, it "
                     "was not this panel." % (ms.get("scored_positions"), pan_ref, total))
    if not ms.get("covers_full_panel") and not any(
            d.get("code") == "subset_of_panel" for d in (sub.get("disclosures") or [])):
        raise Refuse(E_MISSING,
                     "covers_full_panel is false -- %s of panel %s's %s positions were scored -- but "
                     "no subset_of_panel disclosure says which subset. Without it the row would be "
                     "tabled beside full-panel rows with nothing marking the difference."
                     % (ms.get("scored_positions"), pan_ref, total),
                     'add {"code": "subset_of_panel", "severity": "caveat", '
                     '"affects_comparability": true, "detail": "<which positions, and why>"} '
                     "to disclosures")

    new = []
    art_id = None
    # A submission is bound to an existing artifact by (repository, revision) and, when a
    # repo publishes more than one artifact at one revision, by PATH. Taking the first
    # dict-order hit bound a measurement of orcarouter's `2bit-lite/` weights to the
    # `2-bit/` record -- different name, different size (145.0 GB vs 102.5 GB) -- and the
    # scope_digest guard could not catch it, because those two artifacts carry a
    # byte-identical scope_digest. `_apply_gguf_provenance` already says it out loud: "a
    # GGUF repo holds many different quants at one revision, so the file list IS the
    # artifact identity". The submission schema had no way to say which one.
    candidates = [aid for aid, a in registry["artifacts"].items()
                  if (a.get("huggingface") or {}).get("repository") == art_in.get("repository")
                  and (a.get("huggingface") or {}).get("revision") == art_in.get("revision")]
    if len(candidates) == 1:
        art_id = candidates[0]
    elif len(candidates) > 1:
        want = _norm_path(art_in.get("path"))
        matched = [aid for aid in candidates
                   if _norm_path(((registry["artifacts"][aid].get("huggingface") or {})
                                  .get("path"))) == want]
        if want is None or len(matched) != 1:
            raise Refuse(E_IDENTITY,
                         "%s@%s holds %d catalogued artifacts and this submission does not name "
                         "which one: %s. Their weights differ; binding to the first would file a "
                         "measurement against bytes it never read."
                         % (art_in.get("repository"), (art_in.get("revision") or "")[:12],
                            len(candidates),
                            ", ".join("%s (path %r)"
                                      % (aid, (registry["artifacts"][aid].get("huggingface")
                                               or {}).get("path"))
                                      for aid in sorted(candidates))),
                         "add \"path\" to the artifact block, exactly as the registry records it "
                         "(a directory prefix like \"4-bit/\" or a filename like \"model-Q8_0.gguf\")")
        art_id = matched[0]
    if art_id is None:
        owner = (art_in.get("repository") or "unknown/x").split("/")[0]
        art_id = "artifact--%s.%s" % (_slug(owner), _slug((art_in.get("repository") or "x").split("/")[-1]))
        model_ref = None
        for mid, m in registry["models"].items():
            if (m.get("tokenizer") or {}).get("id") and mid.split("--")[1].split(".")[-1] in \
                    (art_in.get("repository") or "").lower():
                model_ref = mid
        if model_ref is None:
            raise Refuse(E_MISSING,
                         "cannot tell which model %s is a quantization of. Register the model and the "
                         "artifact first, or name an existing artifact." % art_in.get("repository"))
        cd = art_in.get("codec") or {}
        new.append({
            "schema_version": L.SCHEMA_VERSION, "id": art_id, "model_ref": model_ref,
            "name": art_in.get("precision_label") or art_in.get("repository"),
            "kind": "quant",
            "huggingface": {"repository": art_in.get("repository"),
                            "url": art_in.get("url"), "revision": art_in.get("revision"),
                            "path": None, "revision_source": "reported_by_author",
                            "status": "known", "link_type": "repository", "reason": None},
            "weights": {"container": art_in.get("container"),
                        "precision_label": art_in.get("precision_label"),
                        "size_bytes": art_in.get("size_bytes"),
                        "size_gb": (art_in["size_bytes"] / 1e9) if art_in.get("size_bytes") else None,
                        "size_basis": "repo_all_files",
                        "index_sha256": art_in.get("index_sha256"),
                        "config_sha256": art_in.get("config_sha256")},
            "codec": {"family": cd.get("family"),
                      "bits_per_weight_nominal": cd.get("bits_per_weight_nominal"),
                      "bits_per_weight_effective": cd.get("bits_per_weight_effective"),
                      "group_size": cd.get("group_size"),
                      "quantizer": {"tool": cd.get("quantizer_tool") or "unknown",
                                    "version": cd.get("quantizer_version"), "revision": None,
                                    "pipeline_ref": None},
                      "calibration": {"used": None, "corpus": None, "tokens": None,
                                      "overlaps_any_panel": None, "overlapping_panel_refs": []}},
            "scope": art_in["scope"], "scope_digest": art_in["scope_digest"],
            "producer": {"name": (art_in.get("producer") or {}).get("name") or "unknown",
                         "role": "quantizer",
                         "handle": (art_in.get("producer") or {}).get("handle"),
                         "url": (art_in.get("producer") or {}).get("url"),
                         "is_registry_maintainer":
                             (art_in.get("producer") or {}).get("handle") == L.MAINTAINER},
            "availability": {"status": "public", "uri": art_in.get("url")},
            "seal": {"sealed": False},
            "cross_refs": {"local_ai_registry": {"model_id": None, "model_instance_id": None,
                                                 "url": None, "match_confidence": "unverified"}},
            "sources": [{"kind": "url", "uri": art_in.get("url") or "unknown"}],
            "disclosures": ([{"code": "artifact_identity_incomplete", "severity": "caveat",
                              "detail": "Per-class recipe recorded as unknown where the release does "
                                        "not declare one.", "affects_comparability": True}]
                            if any(x.get("treatment") == "unknown" for x in art_in["scope"]["assignments"])
                            else [{"code": "no_known_deviations", "severity": "info",
                                   "detail": "Declared by the submission receipt.",
                                   "affects_comparability": False}]),
        })
        art = new[-1]
    else:
        art = registry["artifacts"][art_id]
        if art.get("scope_digest") != art_in.get("scope_digest"):
            raise Refuse(E_IDENTITY,
                         "artifact %s is already registered with scope_digest\n  %s\nbut the submission "
                         "declares\n  %s\nThose are different artifacts, or one of the two scopes is wrong."
                         % (art_id, art.get("scope_digest"), art_in.get("scope_digest")))

    pb = sub.get("produced_by") or {}
    measurer = sub["measurer"]
    pl_id = "pipeline--%s.%s" % (_slug(measurer.get("handle") or measurer.get("name")),
                                 _slug(pb.get("tool") or sub.get("lane") or "stack"))
    if pl_id not in registry["pipelines"]:
        new.append({
            "schema_version": L.SCHEMA_VERSION, "id": pl_id,
            "name": "%s -- %s (lane %s)" % (pb.get("tool") or "contributed stack",
                                            measurer.get("name"), sub.get("lane")),
            "roles": ["end-to-end"],
            # PROV-012, BIAS-006 and registry_render.lane_of all read the lane off the
            # PIPELINE. The submission declares one and the minted record dropped it, so
            # _row_lane defaulted every contributed row to "sealed-ep8": a streaming-lane
            # submission was validated as sealed and rendered inside the primary sealed
            # table, while our own streaming rows sat correctly in the lane sub-table
            # below it. The same omission made BIAS-006 refuse a CORRECT streaming floor.
            "lane": {"name": sub["lane"]},
            "implementation": {"repository": pb.get("repository"), "revision": pb.get("revision"),
                               "entrypoint": pb.get("entrypoint"),
                               "file_sha256": pb.get("entrypoint_sha256"),
                               "container_image": pb.get("container_image"),
                               "container_digest": pb.get("container_digest"),
                               "runtime_reader_sha256": pb.get("runtime_reader_sha256"),
                               "dependencies": pb.get("dependencies") or {}},
            "numerics": {"accumulation_dtype": ("fp64" if sub["estimator"]["accumulation_dtype"]
                                                == "float64" else "fp32"),
                         "two_pass": sub["estimator"].get("two_pass"),
                         "vocab_chunk": sub["estimator"].get("vocab_chunk"),
                         "determinism_controls": (["cold_process_per_run"]
                                                  if (sub.get("determinism") or {}).get("cold_start_per_run")
                                                  else [])},
            "hardware": {"gpu": (sub.get("environment") or {}).get("gpu"),
                         "gpu_count": (sub.get("environment") or {}).get("gpu_count"),
                         "tensor_parallel": (sub.get("environment") or {}).get("tensor_parallel")},
            "cost": {"usd_per_measurement": (sub.get("cost") or {}).get("usd"),
                     "basis": (sub.get("cost") or {}).get("basis")},
            "author": {"name": measurer.get("name"), "role": "toolchain-author",
                       "handle": measurer.get("handle"), "url": measurer.get("url"),
                       "is_registry_maintainer": bool(claim)
                       and maintainer_attribution},
            "cross_refs": {"local_ai_registry": {"model_id": None, "model_instance_id": None,
                                                 "url": None, "match_confidence": "unverified"}},
            "sources": [{"kind": "receipt_file", "uri": path, "sha256": fsha}],
            "disclosures": [{"code": "record_note", "severity": "info",
                             "detail": "Declared by the submission receipt; lane %s."
                                       % sub.get("lane"), "affects_comparability": False}],
        })

    is_ours = bool(claim) and attribution_downgraded is None
    est = sub["estimator"]
    ki = {"panel_id": pan_ref, "reference_id": ref_ref, "metric_name": sub["metric"]["name"],
          "direction": sub["metric"]["direction"],
          "accumulation_dtype": est["accumulation_dtype"],
          "stack_relation": est["stack_relation"], "head_policy": est["head_policy"]}
    key = L.comparability_key(ki)
    declared_cmp = sub.get("comparability") or {}
    declared_bias = declared_cmp.get("bias")
    declared_floor = declared_cmp.get("usable_as_floor")
    disclosures = [dict(d) for d in sub["disclosures"]]
    for d in disclosures:
        d.setdefault("affects_comparability", False)
    if attribution_downgraded:
        disclosures.append({"code": "attribution_downgraded", "severity": "caveat",
                            "detail": attribution_downgraded, "affects_comparability": True})
    if not is_ours and not any(d["code"] == "author_reported_only" for d in disclosures):
        disclosures.append({"code": "author_reported_only", "severity": "caveat",
                            "detail": "Measured and published by %s; we have not re-run it."
                                      % measurer.get("name"), "affects_comparability": True})
    # submission.schema.json says the lane "forces the matching disclosure code", but nothing
    # was forcing it: a lane=streaming receipt generated a row that landed in the same
    # comparability key as the sealed-lane rows with no mention of the lane at all. The lane
    # is a property of the estimator, so a non-sealed lane is a caveat on the row itself,
    # not just a line in the pipeline record.
    lane = sub.get("lane")
    if lane and lane != "sealed-ep8" and not any(d["code"] == "non_sealed_lane" for d in disclosures):
        disclosures.append({
            "code": "non_sealed_lane", "severity": "caveat",
            "detail": "Produced by the %r lane, not the sealed-ep8 lane that the other rows in this "
                      "comparability group used. Lanes are not interchangeable: this row carries an "
                      "undisclosed offset against the sealed lane on the same panel until that offset "
                      "is itself measured and recorded here." % lane,
            "affects_comparability": True})
    if teacher_unverified:
        disclosures.append({
            "code": "teacher_capture_unverified", "severity": "caveat",
            "detail": "Reference %s carries no capture_receipt_sha256, so the teacher digest this "
                      "submission declares (%s) could not be checked against the registry's record "
                      "of that capture. The teacher is asserted here, not verified."
                      % (ref_ref, (declared_teacher or "none")[:16] + "..."),
            "affects_comparability": True})
    # A record that discloses something is not a record with nothing to disclose (DISC-002).
    if len(disclosures) > 1:
        stripped = [d for d in disclosures if d.get("code") != "no_known_deviations"]
        if stripped:
            disclosures = stripped
    det = dict(sub.get("determinism") or {})
    det.pop("per_run_report_sha256", None)
    if det.get("run_means"):
        rm = det["run_means"]
        det["min_run_mean"], det["max_run_mean"] = min(rm), max(rm)
        det["population_stddev_of_run_means"] = L.population_stddev(rm)
    det.setdefault("evidence_hashes", [])
    det.setdefault("distinct_evidence_hash_count", len(det["evidence_hashes"]))

    sources = [{"kind": "receipt_file", "uri": path, "sha256": fsha,
                "note": "sealed submission receipt %s" % SUBMISSION_SCHEMA}]
    sources += [dict(e) for e in (sub.get("evidence") or [])]

    row = {
        "schema_version": L.SCHEMA_VERSION,
        "id": "measurement--%s.%s.%s" % (_slug(measurer.get("handle")),
                                         _slug((art_in.get("repository") or "x").split("/")[-1]),
                                         _slug(pan_ref.split("--", 1)[1])),
        "status": "published", "supersedes": None,
        "model_ref": art["model_ref"], "artifact_ref": art_id, "panel_ref": pan_ref,
        "reference_ref": ref_ref, "pipeline_ref": pl_id,
        "scope_digest": art_in["scope_digest"],
        "metric": dict(sub["metric"], higher_is_better=False),
        "auxiliary_metrics": dict(sub.get("auxiliary_metrics") or {}),
        "uncertainty": {"method": "none", "ci95_low": None, "ci95_high": None,
                        "clusters": None, "samples": None},
        "estimator": {"accumulation_dtype": est["accumulation_dtype"],
                      "logits_dtype": est.get("logits_dtype") or "fp32",
                      "two_pass": est.get("two_pass"), "vocab_chunk": est.get("vocab_chunk"),
                      "stack_relation": est["stack_relation"], "head_policy": est["head_policy"],
                      "zero_handling": est.get("zero_handling")},
        "determinism": det,
        "measurement_scope": dict(ms),
        "provenance": {
            "measured_by": "self-measured" if is_ours else (
                "author-reported" if measurer.get("is_artifact_author") else "third-party-reported"),
            "measurer": {"name": measurer.get("name"), "role": "measurer",
                         "handle": measurer.get("handle"), "url": measurer.get("url"),
                         "is_registry_maintainer": bool(is_ours)},
            "measured_at": sub.get("measured_at"),
            "independently_verified": False, "verification": None,
            "sources": sources, "receipt_schema": SUBMISSION_SCHEMA},
        "comparability": {
            "key": key, "key_inputs": ki,
            "class": "strict" if (is_ours and est["stack_relation"] == "same_stack"
                                  and pan.get("sealed")
                                  and not any(d.get("affects_comparability") for d in disclosures))
            else "advisory",
            # A submission MAY declare its own bias and floor usability. When it
            # does, that wins: the tool that computed the number knows things
            # stack_relation cannot express -- a head substitution biases the
            # number DOWNWARD and is not a cross-stack effect at all, so deriving
            # the bias from stack_relation alone would file it as `bias: null`.
            "bias": (declared_bias if declared_bias is not None else
                     ({"kind": "cross_stack_capture_replay", "direction": "upward",
                       "floor_measurement_ref": None, "estimated_magnitude": None,
                       "detail": "Cross-stack capture declared by the submission; a floor measurement "
                                 "on this panel must be named before this row can be published."}
                      if est["stack_relation"] == "cross_stack" else None)),
            "usable_as_floor": declared_floor},
        "quality_gate": None,
        "cross_refs": {"local_ai_registry": {"model_id": None, "model_instance_id": None,
                                             "url": None, "match_confidence": "unverified"}},
        "disclosures": disclosures,
    }
    return row, new

if __name__ == "__main__":
    sys.exit(main())
