#!/usr/bin/env python3
"""Prove the guarantees by breaking them: build deliberately-invalid registries and
assert the validator rejects each one with the expected check, then assert registry_add
refuses the provenance cases it is supposed to refuse.

A guarantee nobody has tried to violate is a comment. Every case below is a real
mutation of the real data, run through the real tools.

  python3 tools/registry_selftest.py [--root DIR] [--verbose] [--keep]

Stdlib only, offline, no installs.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_lib as L  # noqa: E402
import registry_validate as RV  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# --- mutations: each takes the loaded collections and breaks exactly one thing ----

def m_forge_key(C):
    """A row's comparability key is copied from a row on another panel."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    b = C["measurements"]["measurement--glm53.official-fp8.malaiwah-suite-v5-10m"]
    a["comparability"]["key"] = b["comparability"]["key"]
    return "CMP-001", "a forged comparability key cannot smuggle a number into another table"


def m_forge_key_inputs(C):
    """key_inputs is edited to claim a different panel than the row's own panel_ref."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["comparability"]["key_inputs"]["panel_id"] = "panel--glm53.malaiwah.suite-v5-10m"
    return "CMP-002", "key_inputs is an expansion of the row, never an override"


def m_determinism_from_receipt_hash(C):
    """Determinism claimed on a receipt-file digest instead of tensor content."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["determinism"]["evidence_kind"] = "receipt_file_sha256"
    return "L1.SCHEMA", "a receipt-file hash can never back a determinism claim"


def m_determinism_single_run(C):
    a = C["measurements"]["measurement--glm53.official-fp8.malaiwah-suite-v5-10m"]
    a["determinism"]["identical_across_runs"] = True
    return "L1.SCHEMA", "one run cannot be 'identical across runs'"


def m_zero_spread_lie(C):
    """Run means that differ, with a stddev of 0.0 asserted."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["determinism"]["run_means"] = [0.0137, 0.0138, 0.0139, 0.0140, 0.0141]
    return "DET-002", "the recomputed spread must match the declared one"


def m_author_row_marked_strict(C):
    a = C["measurements"]["measurement--glm53.brandonmusic-4bpw.brandonmusic-final25"]
    a["comparability"]["class"] = "strict"
    return "L1.SCHEMA", "a reported row is never strict"


def m_self_measured_without_receipt(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["provenance"]["sources"] = [{"kind": "url", "uri": "https://example.invalid/trust-me"}]
    return "PROV-001", "you cannot claim to have measured something without a hashed receipt"


def m_third_party_marked_ours(C):
    a = C["measurements"]["measurement--glm53.brandonmusic-4bpw.brandonmusic-final25"]
    a["provenance"]["measured_by"] = "self-measured"
    a["provenance"]["measurer"] = {"name": "malaiwah", "role": "measurer", "handle": "malaiwah",
                                   "url": "https://huggingface.co/malaiwah",
                                   "is_registry_maintainer": True}
    return "PROV-007", "somebody else's measurement can never be relabelled as ours"


def m_self_verified_by_self(C):
    a = C["measurements"]["measurement--glm53.official-fp8.malaiwah-suite-v5-10m"]
    a["provenance"]["independently_verified"] = True
    a["provenance"]["verification"] = {"verified_by": {"name": "malaiwah", "role": "measurer",
                                                       "handle": "malaiwah", "url": None,
                                                       "is_registry_maintainer": True},
                                       "method": "independent_rerun",
                                       "verification_measurement_ref": None}
    return "PROV-003", "verification means somebody else reproduced it"


def m_cross_stack_without_bias(C):
    a = C["measurements"]["measurement--glm53.official-fp8.brandonmusic-final25.crossstack"]
    a["comparability"]["bias"] = None
    return "L1.SCHEMA", "a cross-stack number without its floor is not publishable"


def m_floor_from_another_panel(C):
    a = C["measurements"]["measurement--glm53.official-fp8.brandonmusic-final25.crossstack"]
    a["comparability"]["bias"]["floor_measurement_ref"] = \
        "measurement--qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m"
    return "BIAS-002", "a floor from a different panel is not a floor"


def m_teacher_from_another_panel(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["reference_ref"] = "reference--malaiwah.glm53-bf16-vllm.suite-v5-10m"
    return "REF-004", "a teacher captured on another panel can never back a row"


def m_positions_under_wrong_panel(C):
    """A 51,175-position row filed under the 10.48M panel."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["panel_ref"] = "panel--glm53.malaiwah.suite-v5-10m"
    a["comparability"]["key_inputs"]["panel_id"] = "panel--glm53.malaiwah.suite-v5-10m"
    a["comparability"]["key"] = L.comparability_key(a["comparability"]["key_inputs"])
    return "SCOPE-007", "a subset silently presented as the whole panel"


def m_mlx_row_promoted(C):
    """An MLX row measured against a dequantized reference, stripped of its disclosure."""
    a = C["measurements"]["measurement--glm53.orcarouter-mlx-6bit.undisclosed"]
    a["disclosures"] = [d for d in a["disclosures"] if d["code"] != "different_reference_kind"]
    return "REFC-001", "a dequantized-reference row must say so"


def _make_proxy(C):
    """Turn the dequantized reference into a DESIGNATED quantized proxy.

    The corpus has no quantized_proxy reference yet -- no family we measure
    lacks an unquantized release -- so the fixture builds one. A proxy is the
    most-faithful PUBLISHED artifact of a family that ships no unquantized
    weights at all, designated as the reference so its children are measurable.
    """
    r = C["references"]["reference--orcarouter.glm53-fp8-dequantized.undisclosed"]
    r["reference_kind"] = "quantized_proxy"
    r["artifact_ref"] = "artifact--0xsero.glm-5.3-flash-exl3-3.0bpw"
    return C["measurements"]["measurement--glm53.orcarouter-mlx-6bit.undisclosed"]


def m_proxy_reference_undisclosed(C):
    """A row against a designated proxy that does not say the reference is one."""
    a = _make_proxy(C)
    a["disclosures"] = [d for d in a["disclosures"]
                        if d["code"] != "different_reference_kind"]
    return "REFC-006", "a designated-proxy row must say its number is not a floor"


def m_proxy_reference_marked_strict(C):
    """A designated proxy is not a measured floor, so no row against one is strict."""
    a = _make_proxy(C)
    a["comparability"]["class"] = "strict"
    return "REFC-006", "a designated-proxy row cannot be strict"


def m_proxy_pointing_at_base(C):
    """A proxy pointing at a BASE artifact is a native reference wearing the wrong label."""
    _make_proxy(C)
    C["references"]["reference--orcarouter.glm53-fp8-dequantized.undisclosed"][
        "artifact_ref"] = "artifact--malaiwah.glm-5.2-siq-fruit-bf16"
    return "REFC-006", "a quantized_proxy must point at an actually-quantized artifact"


def m_remote_code_unrecorded_harness(C):
    """A row measured by executing repo-shipped modeling code, with no digest of it.

    The approved policy (2026-09-01) is that trust_remote_code is acceptable in
    the checkpoint lane ONLY revision-pinned and content-digested: the shipped
    .py files enter harness.code_digests exactly like the suite's own estimator
    closure. A remote_code row whose harness is unrecorded asserts "we ran code
    we did not hash", which is the one claim this registry exists to refuse.
    """
    a = C["measurements"]["measurement--glm53.turbo-4.05bpw-stream.brandonmusic-final25"]
    a["disclosures"].append({"code": "remote_code", "severity": "caveat",
                             "affects_comparability": True,
                             "detail": "executed modeling_kimi_k3.py from the repo"})
    return "RC-001", "remote code without a recorded harness must be refused"


def m_scope_digest_edited(C):
    a = C["artifacts"]["artifact--malaiwah.glm-5.3-flash-tr3-6bpw"]
    a["scope"]["assignments"][3]["bits_per_weight"] = 4.0
    return "SCOPE-002", "a scope edit nobody restated cannot slip through"


def m_dangling_panel(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["panel_ref"] = "panel--does.not.exist"
    a["comparability"]["key_inputs"]["panel_id"] = "panel--does.not.exist"
    a["comparability"]["key"] = L.comparability_key(a["comparability"]["key_inputs"])
    return "REF-001", "every ref must resolve"


def m_panel_cycle(C):
    p = C["panels"]["panel--glm53.brandonmusic.final25"]
    p["derived_from"] = "panel--glm53.brandonmusic.final-0000"
    p["derivation"] = {"kind": "shard_subset", "detail": "deliberate cycle for the self-test"}
    return "REF-008", "a derived_from cycle is rejected"


def m_receipt_hash_as_panel_identity(C):
    p = C["panels"]["panel--glm53.brandonmusic.final25"]
    p["identity"]["panel_token_sha256"] = p["identity"]["panel_receipt_sha256"]
    return "PANEL-002", "a hash of the file that DESCRIBES a panel is not a hash of its tokens"


def m_subset_panel_shares_digest(C):
    p = C["panels"]["panel--qwen38.malaiwah.suite-v5-shard0-1m"]
    parent = C["panels"]["panel--qwen38.malaiwah.suite-v5-10m"]
    p["identity"]["panel_token_sha256"] = parent["identity"]["panel_token_sha256"]
    return "PANEL-008", "a shard subset contains different tokens and must not share the digest"


def m_negative_value(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["metric"]["value"] = -0.001
    a["determinism"]["run_means"] = [-0.001] * 5
    return "STAT-004", "a negative mean KL is an estimator bug, not a result"


def m_truncated_value(C):
    """A value rounded for display: the per-run array still carries the true numbers,
    so the aggregate recomputation catches it."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["metric"]["value"] = 0.0137
    return "DET-003", "the data file keeps full float64; the README rounds"


def m_ci_excludes_value(C):
    a = C["measurements"]["measurement--glm53.official-fp8.malaiwah-suite-v5-10m"]
    a["uncertainty"]["ci95_high"] = 0.0271
    return "STAT-001", "a value outside its own interval"


def m_unknown_disclosure_code(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["disclosures"] = [{"code": "looks_fine_to_me", "severity": "info", "detail": "trust me",
                         "affects_comparability": False}]
    return "DISC-004", "codes come from a closed list so they stay groupable"


def m_no_known_deviations_plus_caveat(C):
    # Both entries are set here rather than appended to whatever the row happens
    # to carry: this case tests DISC-002, not the current disclosure population,
    # and it silently stopped testing anything the moment a real row gained a
    # second disclosure and no longer had `no_known_deviations` to contradict.
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["disclosures"] = [
        {"code": "no_known_deviations", "severity": "info",
         "detail": "nothing to disclose", "affects_comparability": False},
        {"code": "reduced_run_count", "severity": "caveat",
         "detail": "and also this", "affects_comparability": False},
    ]
    return "DISC-002", "'nothing to disclose' cannot coexist with a disclosure"


def m_pipeline_not_ours(C):
    """A self-measured row that ran on somebody else's stack."""
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["pipeline_ref"] = "pipeline--brandonmusic.glm53-packed-kld"
    return "PROV-008", "a number we claim to have produced must have run on a stack we own"


def m_duplicate_row(C):
    a = copy.deepcopy(C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"])
    a["id"] = "measurement--glm53.k6-6bpw.brandonmusic-final25.copy"
    C["measurements"][a["id"]] = a
    return "CMP-004", "an undeclared duplicate of an existing row"


def m_pending_row_with_a_value(C):
    a = copy.deepcopy(C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"])
    a["id"] = "measurement--glm53.k8-8bpw.pending"
    a["status"] = "pending"
    a["artifact_ref"] = "artifact--malaiwah.glm-5.3-flash-tr3-8bpw"
    a["scope_digest"] = C["artifacts"]["artifact--malaiwah.glm-5.3-flash-tr3-8bpw"]["scope_digest"]
    C["measurements"][a["id"]] = a
    return "L1.SCHEMA", "a pending row must not carry a number"


def m_stream_row_loses_its_lane(C):
    """The failure this whole lane split exists to stop: a streaming-lane number filed with
    nothing on it that says so, sitting in the sealed lane's table one line above the sealed
    row for the SAME weights."""
    a = C["measurements"]["measurement--glm53.k6-6bpw-stream.brandonmusic-final25"]
    a["disclosures"] = [d for d in a["disclosures"] if d["code"] != "non_sealed_lane"]
    return "PROV-012", "a streaming-lane row must say it is a streaming-lane row"


def m_stream_row_loses_its_bias(C):
    a = C["measurements"]["measurement--glm53.k6-6bpw-stream.brandonmusic-final25"]
    a["comparability"]["bias"] = None
    return "PROV-012", "a lane offset is measured or unknown, never absent"


def m_lane_is_its_own_baseline(C):
    p = C["pipelines"]["pipeline--malaiwah.glm53-stream-packed-kld"]
    p["lane"]["bridge"]["sealed_measurement_ref"] = \
        "measurement--glm53.k6-6bpw-stream.brandonmusic-final25"
    return "PROV-012", "a lane cannot bridge to a row it produced itself"


def m_floor_measured_on_a_different_lane(C):
    """Exactly the mistake engines/BF16-FLOOR.md warns about, made mechanically: a floor gets
    re-pointed at a row measured on a DIFFERENT lane than the row citing it, without
    touching either row's artifact, panel, teacher or comparability key -- so BIAS-002
    (same key) and BIAS-004 (floor measures unquantized weights) still pass, and only
    BIAS-006 stands between this and a published cross-lane subtraction. The floor's
    pipeline_ref is swapped to the SEALED lane's pipeline (no `lane` object at all, so it
    defaults to sealed-ep8); the K6-stream and K8-stream rows that cite it stay on the
    streaming lane."""
    floor = C["measurements"]["measurement--glm53.bf16-stream-floor.brandonmusic-final25"]
    floor["pipeline_ref"] = "pipeline--malaiwah.glm53-packed-kld"
    return "BIAS-006", "a floor measured on one lane is not the zero-point for a different lane"


def m_row_below_its_floor(C):
    """A published row reporting LESS divergence than unquantized weights.

    The forged-submission case (a `receipts/malaiwah/` file claiming self-measured, at
    0.009 nats) rendered at the TOP of the flagship ranked table with the validator
    reporting zero errors, because nothing compared a row against the measurement floor
    sitting in its own comparability group.  This mutation is INTERNALLY CONSISTENT --
    the CI, the run means and the per-domain table all move with the headline -- so every
    other invariant is satisfied and only FLOOR-001 can catch it.  A 6bpw quant cannot be
    more faithful to the reference than the bf16 weights it was quantized from."""
    m = C["measurements"]["measurement--glm53.k6-6bpw-stream.brandonmusic-final25"]
    new = 0.0090001
    delta = new - m["metric"]["value"]
    m["metric"]["value"] = new
    det = m["determinism"]
    det["run_means"] = [new] * len(det["run_means"])
    det["min_run_mean"] = det["max_run_mean"] = new
    u = m["uncertainty"]
    for k in ("ci95_low", "ci95_high"):
        if u.get(k) is not None:
            u[k] += delta
    for d in m.get("by_domain") or []:
        d["mean"] += delta
        for k in ("ci95_low", "ci95_high"):
            if d.get(k) is not None:
                d[k] += delta
    return "FLOOR-001", ("a quantized row below the unquantized floor of its own group is not a "
                         "ranking, it is a defect")




# --- A. harness identity (HARN-*) ------------------------------------------
# A row's number is a function of some code. Until 2026-08-30 nothing said WHICH
# code, so a defect in the estimator put every published row equally under
# suspicion and no row could be cleared. These prove the stamp cannot be skipped,
# forged, or faked from a later checkout.

_ENRICHED = "measurement--glm53.k6-6bpw.brandonmusic-final25"
_PLAIN = "measurement--glm53.official-fp8.malaiwah-suite-v5-10m"


def m_harness_missing(C):
    del C["measurements"][_ENRICHED]["harness"]
    return "HARN-001", "a row must say which code produced it, or say that it does not know"


def m_harness_new_row_unstamped(C):
    """THE case: a brand-new row, not on the frozen grandfather list, with no harness.

    The grandfather list exists so 70+ published rows are not retroactively
    invalidated. It must not become a door: an id that is not on it has to carry
    a real stamp, and the list is never appended to.
    """
    new = copy.deepcopy(C["measurements"][_PLAIN])
    new["id"] = "measurement--brandnew.unstamped.suite-v5-10m"
    new["harness"] = {"harness_id": None, "recorded": False, "covers": ["metric.value"],
                      "note": "not recorded"}
    new["disclosures"] = [d for d in new["disclosures"]
                          if d["code"] != "harness_unrecorded"] or [
        {"code": "record_note", "severity": "info", "detail": "x",
         "affects_comparability": False}]
    C["measurements"][new["id"]] = new
    return "HARN-001", "a NEW row cannot skip the harness by not being on the frozen list"


def m_harness_id_forged(C):
    h = C["measurements"][_ENRICHED]["harness"]
    h["harness_id"] = "harness--0000000000000000"
    return "HARN-003", "harness_id is recomputed from the digests, never trusted"


def m_harness_digest_swapped(C):
    """The digest set is edited while the id is left alone -- the shape a silent
    provenance edit actually takes."""
    h = C["measurements"][_ENRICHED]["harness"]
    h["code_digests"][0]["sha256"] = "0" * 64
    return "HARN-003", "editing a digest without the id must not validate"


def m_harness_unrecorded_with_digests(C):
    h = C["measurements"][_PLAIN]["harness"]
    h["code_digests"] = [{"role": "estimator", "path": "bin/jointstd/stats.py",
                          "sha256": "1" * 64}]
    return "HARN-002", "an unrecorded harness must not carry digests invented later"


def m_harness_grandfathered_undisclosed(C):
    m = C["measurements"][_PLAIN]
    m["disclosures"] = [d for d in m["disclosures"] if d["code"] != "harness_unrecorded"]
    return "HARN-004", "the gap must be readable on the row, not only in schema/"


# --- B. the per-domain interval (STAT-007/008/009) -------------------------

def m_domain_shared_seed(C):
    """STAT-17 as it was: two strata drawing one resample stream."""
    bd = C["measurements"][_ENRICHED]["by_domain"]
    bd[0]["interval_method"] = bd[1]["interval_method"] = "window_block_bootstrap_bca"
    bd[0]["bootstrap_b"] = bd[1]["bootstrap_b"] = 1000
    bd[0]["bootstrap_seed"] = bd[1]["bootstrap_seed"] = 20260829
    return "STAT-008", "two domains sharing a seed share their Monte-Carlo error"


def m_domain_no_coverage(C):
    """STAT-01 as it was: a five-window interval labelled 95% and never measured."""
    for cell in C["measurements"][_ENRICHED]["by_domain"]:
        cell.pop("coverage_measured", None)
    return "STAT-007", "a small-g interval must state the coverage it actually has"


def m_domain_negative_lower(C):
    C["measurements"][_ENRICHED]["by_domain"][0]["ci95_low"] = -0.009475
    return "STAT-009", "a negative lower bound on a KL divergence is an artifact"


def m_domain_method_unstated(C):
    C["measurements"][_ENRICHED]["by_domain"][0]["interval_method"] = None
    return "STAT-009", "an interval must name the procedure that produced it"


# --- C. provenance assertions and source portability (PROV-014..017) -------
# PROC-01: metric rows have always needed a hashed receipt; an ASSERTION about
# mechanism or lineage needed nothing, so a prose provenance claim reached two
# dataset cards and two registry rows uncited and validated clean.

_FRUIT = "artifact--malaiwah.glm-5.2-siq-fruit.exl3-k3k4"


def m_provenance_uncited(C):
    for d in C["artifacts"][_FRUIT]["disclosures"]:
        if d.get("asserts_provenance"):
            d.pop("sources", None)
    return "PROV-014", "a provenance assertion with no source is what PROC-01 was"


def m_provenance_cited_by_branch(C):
    for d in C["artifacts"][_FRUIT]["disclosures"]:
        if d.get("asserts_provenance"):
            d["sources"][0]["uri"] = ("https://github.com/malaiwah/proxy-fruit/blob/main/"
                                      "export_fruit.py")
    return "PROV-015", "cite by commit; a line anchor against a branch stops being true"


def m_provenance_unmarked(C):
    for d in C["artifacts"][_FRUIT]["disclosures"]:
        if d.get("asserts_provenance"):
            d["asserts_provenance"] = False
    return "PROV-016", "an author who does not think of a mechanism claim as one is the case"


def m_host_absolute_source(C):
    model = C["models"]["model--qwen.qwen3.8-27b"]
    model["sources"][0]["uri"] = "/home/reviewer/private-receipt.json"
    return "PROV-017", "published evidence must resolve beyond its author's workstation"


def m_non_canonical_line(C):
    return None  # handled specially below


def m_remote_code_without_remote_digest(C):
    """A remote_code disclosure beside a harness that digests only the suite's own
    closure. RC-001's tightened form: the disclosure and the digests must corroborate
    each other -- the modeling .py files the disclosure warns about must appear in
    code_digests with role=remote_model_code, or 'we hashed what ran' is false for
    exactly the code that makes the row need the disclosure."""
    a = C["measurements"]["measurement--glm53.k6-6bpw-stream.brandonmusic-final25"]
    assert (a.get("harness") or {}).get("recorded"), "fixture row must have a recorded harness"
    assert not any(d.get("role") == "remote_model_code"
                   for d in a["harness"]["code_digests"])
    a.setdefault("disclosures", []).append({
        "code": "remote_code",
        "detail": "executed repository-shipped modeling code (trust_remote_code)",
        "severity": "caveat",
        "affects_comparability": True,
    })
    return "RC-001", "a remote_code disclosure must be corroborated by a remote_model_code digest"


MUTATIONS = [
    ("forged-comparability-key", m_forge_key),
    ("remote-code-without-remote-digest", m_remote_code_without_remote_digest),
    ("forged-key-inputs", m_forge_key_inputs),
    ("determinism-from-receipt-hash", m_determinism_from_receipt_hash),
    ("determinism-single-run", m_determinism_single_run),
    ("zero-spread-lie", m_zero_spread_lie),
    ("author-row-marked-strict", m_author_row_marked_strict),
    ("self-measured-without-receipt", m_self_measured_without_receipt),
    ("third-party-row-relabelled-ours", m_third_party_marked_ours),
    ("self-verified-by-self", m_self_verified_by_self),
    ("cross-stack-without-bias", m_cross_stack_without_bias),
    ("floor-from-another-panel", m_floor_from_another_panel),
    ("teacher-from-another-panel", m_teacher_from_another_panel),
    ("positions-under-wrong-panel", m_positions_under_wrong_panel),
    ("dequantized-reference-undisclosed", m_mlx_row_promoted),
    ("remote-code-unrecorded-harness", m_remote_code_unrecorded_harness),
    ("proxy-reference-undisclosed", m_proxy_reference_undisclosed),
    ("proxy-reference-marked-strict", m_proxy_reference_marked_strict),
    ("proxy-reference-on-base-artifact", m_proxy_pointing_at_base),
    ("scope-digest-not-restated", m_scope_digest_edited),
    ("dangling-panel-ref", m_dangling_panel),
    ("panel-derivation-cycle", m_panel_cycle),
    ("receipt-hash-as-panel-identity", m_receipt_hash_as_panel_identity),
    ("subset-panel-shares-parent-digest", m_subset_panel_shares_digest),
    ("negative-kl", m_negative_value),
    ("value-truncated-for-display", m_truncated_value),
    ("relabelled-via-pipeline", m_pipeline_not_ours),
    ("value-outside-its-own-ci", m_ci_excludes_value),
    ("unknown-disclosure-code", m_unknown_disclosure_code),
    ("no-deviations-plus-a-caveat", m_no_known_deviations_plus_caveat),
    ("undeclared-duplicate-row", m_duplicate_row),
    ("pending-row-carrying-a-value", m_pending_row_with_a_value),
    ("stream-row-without-its-lane", m_stream_row_loses_its_lane),
    ("stream-row-without-its-bias", m_stream_row_loses_its_bias),
    ("lane-bridged-to-itself", m_lane_is_its_own_baseline),
    ("floor-measured-on-a-different-lane", m_floor_measured_on_a_different_lane),
    ("row-below-its-own-floor", m_row_below_its_floor),
    ("harness-block-missing", m_harness_missing),
    ("new-row-with-no-harness", m_harness_new_row_unstamped),
    ("forged-harness-id", m_harness_id_forged),
    ("harness-digest-swapped", m_harness_digest_swapped),
    ("unrecorded-harness-with-digests", m_harness_unrecorded_with_digests),
    ("grandfathered-row-not-disclosed", m_harness_grandfathered_undisclosed),
    ("two-domains-one-bootstrap-seed", m_domain_shared_seed),
    ("small-g-interval-without-coverage", m_domain_no_coverage),
    ("negative-lower-bound-on-a-kl", m_domain_negative_lower),
    ("interval-without-its-method", m_domain_method_unstated),
    ("provenance-assertion-with-no-source", m_provenance_uncited),
    ("provenance-cited-against-a-branch", m_provenance_cited_by_branch),
    ("mechanism-claim-not-marked", m_provenance_unmarked),
    ("host-absolute-source-uri", m_host_absolute_source),
]


def build_case(root, tmp, name, mutate):
    dest = os.path.join(tmp, name)
    shutil.copytree(os.path.join(root, "schema"), os.path.join(dest, "schema"))
    C = L.load_registry(os.path.join(root, "data"))
    expected, why = mutate(C)
    os.makedirs(os.path.join(dest, "data"))
    for coll, _, _ in L.COLLECTIONS:
        L.write_jsonl(os.path.join(dest, "data", coll + ".jsonl"), list(C[coll].values()))
    return dest, expected, why


def run_validator(root, extra=()):
    out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"), "--root", root,
                          "--json", "--jsonschema-lib", "mini"] + list(extra),
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout), out.returncode
    except ValueError:
        return {"findings": [], "_stderr": out.stderr, "_stdout": out.stdout}, out.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=L.repo_root(__file__))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="qfr-selftest-")
    passed = failed = 0

    print("=" * 78)
    print("A. the real registry must be clean")
    print("=" * 78)
    rep, code = run_validator(args.root)
    n_err = len([f for f in rep.get("findings", []) if f["severity"] == "error"])
    ok = code == 0 and n_err == 0
    print("  %-58s %s (%d errors, %d warnings)"
          % ("data/ validates", "PASS" if ok else "FAIL", n_err,
             len([f for f in rep.get("findings", []) if f["severity"] == "warn"])))
    passed += ok
    failed += not ok


    print()
    print("=" * 78)
    print("B. deliberately-invalid registries must be REJECTED, each by the right check")
    print("=" * 78)
    for name, mutate in MUTATIONS:
        dest, expected, why = build_case(args.root, tmp, name, mutate)
        rep, code = run_validator(dest)
        errs = [f for f in rep.get("findings", []) if f["severity"] == "error"]
        hit = [f for f in errs if f["check"] == expected]
        ok = bool(hit) and code == 1
        print("  %-42s %-16s %s" % (name, expected, "PASS" if ok else "FAIL"))
        if args.verbose or not ok:
            print("      why it must fail: %s" % why)
            if hit:
                print("      caught: %s" % hit[0]["message"].split("\n")[0][:150])
            else:
                print("      NOT CAUGHT. errors seen: %s" % sorted({f["check"] for f in errs}))
        passed += ok
        failed += not ok

    # non-canonical serialization
    dest = os.path.join(tmp, "non-canonical-line")
    shutil.copytree(os.path.join(args.root, "schema"), os.path.join(dest, "schema"))
    shutil.copytree(os.path.join(args.root, "data"), os.path.join(dest, "data"))
    p = os.path.join(dest, "data", "measurements.jsonl")
    lines = open(p, encoding="utf-8").read().splitlines()
    obj = json.loads(lines[0])
    lines[0] = json.dumps(obj, indent=None, sort_keys=False)
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    rep, code = run_validator(dest)
    ok = any(f["check"] in ("L0.CANONICAL", "L0.SORTED") for f in rep.get("findings", []))
    print("  %-42s %-16s %s" % ("non-canonical-serialization", "L0.CANONICAL", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    # CMP-007: a mixed-lane group's like-for-like predicate hand-promoted to
    # comparable=true must be rejected. The key is a necessary partition, not a
    # certificate (P1-01); the predicate is the machine-readable rest of the
    # contract, and it is recomputed, never trusted -- same doctrine as CMP-001.
    dest = os.path.join(tmp, "predicate-promoted-by-hand")
    shutil.copytree(os.path.join(args.root, "schema"), os.path.join(dest, "schema"))
    shutil.copytree(os.path.join(args.root, "data"), os.path.join(dest, "data"))
    with open(os.path.join(args.root, "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    promoted = 0
    for entry in idx.get("comparability_keys", []):
        pred = entry.get("comparability") or {}
        if pred.get("comparable") == "false":
            pred["comparable"] = "true"
            pred["reasons"] = []
            promoted += 1
    with open(os.path.join(dest, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2, sort_keys=True, ensure_ascii=False)
    rep, code = run_validator(dest)
    errs = [f for f in rep.get("findings", []) if f["severity"] == "error"]
    ok = promoted > 0 and code == 1 and any(f["check"] == "CMP-007" for f in errs)
    print("  %-42s %-16s %s" % ("predicate-promoted-by-hand", "CMP-007", "PASS" if ok else "FAIL"))
    if not ok:
        print("      promoted %d groups; errors seen: %s"
              % (promoted, sorted({f["check"] for f in errs})))
    passed += ok
    failed += not ok

    print()
    print("=" * 78)
    print("C. registry_add must REFUSE, with the documented exit code")
    print("=" * 78)
    scratch = os.path.join(tmp, "receipts")
    os.makedirs(scratch)
    with open(os.path.join(scratch, "unknown-family.json"), "w") as fh:
        json.dump({"schema": "someone-elses-kld/1", "mean": 0.01}, fh)
    base = [PY, os.path.join(HERE, "registry_add.py")]
    common = ["--registry", args.root, "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
              "--panel", "panel--glm53.brandonmusic.final25",
              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
              "--pipeline", "pipeline--malaiwah.glm53-packed-kld", "--dry-run"]
    receipts = _find_receipts()
    cases = [
        ("unknown receipt family", 3,
         base + ["from-receipt", "--receipt", os.path.join(scratch, "unknown-family.json")] + common),
    ]
    if receipts.get("k6_five"):
        cases.append(("panel digest does not match --panel", 7,
                      base + ["from-receipt", "--receipt", receipts["k6_five"]] + common[:2]
                      + ["--panel", "panel--glm53.malaiwah.suite-v5-10m",
                         "--reference", "reference--malaiwah.glm53-bf16-vllm.suite-v5-10m",
                         "--pipeline", "pipeline--malaiwah.glm53-packed-kld", "--dry-run",
                         "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw"]))
    if receipts.get("qwen_report"):
        cases.append(("report with reference_revision=null", 4,
                      base + ["from-report", "--report", receipts["qwen_report"],
                              "--registry", args.root,
                              "--artifact", "artifact--qwen.qwen3.8-27b-fp8",
                              "--panel", "panel--qwen38.malaiwah.suite-v5-10m",
                              "--reference", "reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m",
                              "--pipeline", "pipeline--malaiwah.qwen38-kld-ladder",
                              "--third-party-artifact", "--dry-run"]))
    if receipts.get("crosscheck"):
        cases.append(("cross-stack row with no floor", 4,
                      base + ["from-crosscheck", "--report", receipts["crosscheck"],
                              "--registry", args.root,
                              "--artifact", "artifact--zai-org.glm-5.3-flash-fp8",
                              "--panel", "panel--glm53.brandonmusic.final25",
                              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                              "--pipeline", "pipeline--malaiwah.glm53-crosscheck",
                              "--third-party-artifact", "--dry-run"]))
    if receipts.get("v44_fp8"):
        cases.append(("foreign receipt claimed as self-measured", 8,
                      base + ["from-foreign", "--receipt", receipts["v44_fp8"], "--registry", args.root,
                              "--artifact", "artifact--brandonmusic.glm-5.3-flash-fp8-mla-kv",
                              "--panel", "panel--glm53.brandonmusic.final-0000",
                              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final-0000",
                              "--pipeline", "pipeline--brandonmusic.sm120-runtime.v44",
                              "--attribution", "self-measured", "--dry-run"]))
        cases.append(("foreign receipt filed under the 25-window panel", 7,
                      base + ["from-foreign", "--receipt", receipts["v44_fp8"], "--registry", args.root,
                              "--artifact", "artifact--brandonmusic.glm-5.3-flash-fp8-mla-kv",
                              "--panel", "panel--glm53.brandonmusic.final25",
                              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                              "--pipeline", "pipeline--brandonmusic.sm120-runtime.v44",
                              "--attribution", "author-reported", "--reported-by", "brandonmusic",
                              "--source-url", "https://example.org/r", "--dry-run"]))
    if receipts.get("dione"):
        cases.append(("third-party artifact claimed without the flag", 8,
                      base + ["from-receipt", "--receipt", receipts["dione"], "--registry", args.root,
                              "--artifact", "artifact--0xsero.glm-5.3-flash-exl3-q4",
                              "--panel", "panel--glm53.brandonmusic.final25",
                              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                              "--pipeline", "pipeline--malaiwah.glm53-dione-packed-kld", "--dry-run"]))
    stream_common = ["--registry", args.root, "--panel", "panel--glm53.brandonmusic.final25",
                     "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                     "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                     "--direction", "reference_to_candidate", "--accumulation", "float64",
                     "--scored-positions", "51175", "--dry-run"]
    if receipts.get("stream_k8"):
        cases.append(("streaming summary that does not name its lane", 4,
                      base + ["from-receipt", "--receipt", receipts["stream_k8"],
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-8bpw",
                              "--contexts", "25"] + stream_common))
        # The write-time half of BIAS-006: this registry's CROSS-STACK floor
        # (measurement--glm53.bf16-replay-floor...) was measured on the sealed-ep8 lane
        # (pipeline--malaiwah.glm53-crosscheck declares no `lane` object at all); naming it
        # as a streaming-lane row's floor is exactly the cross-lane subtraction
        # engines/BF16-FLOOR.md warns against, and must never even reach a written row.
        cases.append(("a floor measured on a different lane", 7,
                      base + ["from-receipt", "--receipt", receipts["stream_k8"],
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-8bpw",
                              "--contexts", "25", "--lane", "streaming", "--floor-measurement",
                              "measurement--glm53.bf16-replay-floor.brandonmusic-final25"]
                      + stream_common))
    if receipts.get("stream_k6"):
        cases.append(("a lane flag contradicting the receipt's own family", 6,
                      base + ["from-receipt", "--receipt", receipts["stream_k6"],
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                              "--contexts", "25", "--lane", "sealed-ep8"] + stream_common))
        cases.append(("streaming summary with no --direction", 4,
                      base + ["from-receipt", "--receipt", receipts["stream_k6"],
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                              "--registry", args.root, "--panel", "panel--glm53.brandonmusic.final25",
                              "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                              "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                              "--accumulation", "float64", "--scored-positions", "51175",
                              "--contexts", "25", "--dry-run"]))
        # A receipt whose asserted determinism flag disagrees with its own arrays.
        tampered = os.path.join(scratch, "stream-k6-flag-lie.json")
        with open(receipts["stream_k6"], encoding="utf-8") as fh:
            bad = json.load(fh)
        bad["run_means"] = [bad["run_means"][0], bad["run_means"][0] + 1e-9]
        with open(tampered, "w") as fh:
            json.dump(bad, fh)
        cases.append(("bitwise_deterministic contradicted by run_means", 5,
                      base + ["from-receipt", "--receipt", tampered,
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                              "--contexts", "25"] + stream_common))
    if receipts.get("stream_k6_verdict"):
        cases.append(("a verdict receipt offered as a measurement", 4,
                      base + ["from-receipt", "--receipt", receipts["stream_k6_verdict"],
                              "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                              "--contexts", "25"] + stream_common))

    for label, want, cmd in cases:
        out = subprocess.run(cmd, capture_output=True, text=True)
        ok = out.returncode == want
        print("  %-58s exit %d  %s" % (label, want, "PASS" if ok else "FAIL (got %d)" % out.returncode))
        if args.verbose or not ok:
            print("      %s" % (out.stderr.strip().split("\n")[0][:160] or out.stdout[:160]))
        passed += ok
        failed += not ok

    print()
    print("=" * 78)
    print("D. registry_add must ACCEPT the real receipts and reproduce the seeded rows")
    print("=" * 78)
    if receipts.get("k6_five") and receipts.get("k6_packed"):
        out = subprocess.run(base + ["from-receipt", "--receipt", receipts["k6_packed"],
                                     "--receipt", receipts["k6_five"]] + common,
                             capture_output=True, text=True)
        ok = False
        if out.returncode == 0:
            row = json.loads(out.stdout)
            seeded = L.load_registry(os.path.join(args.root, "data"))["measurements"][
                "measurement--glm53.k6-6bpw.brandonmusic-final25"]
            same = (row["metric"]["value"] == seeded["metric"]["value"]
                    and row["comparability"]["key"] == seeded["comparability"]["key"]
                    and row["determinism"]["evidence_hashes"] == seeded["determinism"]["evidence_hashes"]
                    and row["measurement_scope"]["scored_positions"]
                    == seeded["measurement_scope"]["scored_positions"])
            ok = same
        print("  %-58s %s" % ("K6 rebuilt from receipts matches the seeded row",
                              "PASS" if ok else "FAIL"))
        passed += ok
        failed += not ok

    for label, mid, cmd in (
            ("K6 streaming row rebuilt from its receipt + verdict",
             "measurement--glm53.k6-6bpw-stream.brandonmusic-final25",
             (base + ["from-receipt", "--receipt", receipts.get("stream_k6", ""),
                      "--receipt", receipts.get("stream_k6_verdict", ""),
                      "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                      "--registry", args.root, "--panel", "panel--glm53.brandonmusic.final25",
                      "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                      "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                      "--direction", "reference_to_candidate", "--accumulation", "float64",
                      "--scored-positions", "51175",
                      "--floor-measurement", "measurement--glm53.bf16-stream-floor.brandonmusic-final25",
                      "--dry-run"])
             if receipts.get("stream_k6") and receipts.get("stream_k6_verdict") else None),
            ("K8 streaming row rebuilt from its receipt + an asserted lane",
             "measurement--glm53.k8-8bpw-stream.brandonmusic-final25",
             (base + ["from-receipt", "--receipt", receipts.get("stream_k8", ""),
                      "--artifact", "artifact--malaiwah.glm-5.3-flash-tr3-8bpw",
                      "--registry", args.root, "--panel", "panel--glm53.brandonmusic.final25",
                      "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                      "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                      "--direction", "reference_to_candidate", "--accumulation", "float64",
                      "--scored-positions", "51175", "--contexts", "25",
                      "--lane", "streaming",
                      "--floor-measurement", "measurement--glm53.bf16-stream-floor.brandonmusic-final25",
                      "--dry-run"])
             if receipts.get("stream_k8") else None),
            ("native-BF16 streaming floor row rebuilt from its receipt",
             "measurement--glm53.bf16-stream-floor.brandonmusic-final25",
             (base + ["from-receipt", "--receipt", receipts.get("stream_bf16", ""),
                      "--artifact", "artifact--zai-org.glm-5.3-flash-bf16.a6c167b6",
                      "--registry", args.root, "--panel", "panel--glm53.brandonmusic.final25",
                      "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                      "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                      "--direction", "reference_to_candidate", "--accumulation", "float64",
                      "--scored-positions", "51175", "--contexts", "25",
                      "--lane", "streaming", "--third-party-artifact", "--dry-run"])
             if receipts.get("stream_bf16") else None)):
        if cmd is None:
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        ok = False
        detail = out.stderr.strip()[:200]
        if out.returncode == 0:
            row = json.loads(out.stdout)
            seeded = L.load_registry(os.path.join(args.root, "data"))["measurements"][mid]
            checks = {
                "value": row["metric"]["value"] == seeded["metric"]["value"],
                "key": row["comparability"]["key"] == seeded["comparability"]["key"],
                "evidence": (row["determinism"]["evidence_hashes"]
                             == seeded["determinism"]["evidence_hashes"]),
                "positions": (row["measurement_scope"]["scored_positions"]
                              == seeded["measurement_scope"]["scored_positions"]),
                "contexts": row["measurement_scope"]["contexts"] == seeded["measurement_scope"]["contexts"],
                "bias": row["comparability"]["bias"] == seeded["comparability"]["bias"],
                "disclosure codes": (sorted(d["code"] for d in row["disclosures"])
                                     == sorted(d["code"] for d in seeded["disclosures"])),
                "top1": (row["auxiliary_metrics"]["top1_agreement"]
                         == seeded["auxiliary_metrics"]["top1_agreement"]),
            }
            ok = all(checks.values())
            detail = "differs on: %s" % ", ".join(k for k, v in checks.items() if not v)
        print("  %-58s %s" % (label, "PASS" if ok else "FAIL"))
        if not ok:
            print("      %s" % detail)
        passed += ok
        failed += not ok

    print()
    print("=" * 78)
    print("G. the submission path (CONTRIBUTING.md) must work, and must refuse a broken seal")
    print("=" * 78)
    ex = os.path.join(args.root, "docs", "examples", "dione-q4.submission.json")
    if os.path.exists(ex):
        out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"), "--root", args.root,
                              "--submission", ex], capture_output=True, text=True)
        ok = out.returncode == 0 and "cmp--202b717f3219c414" in out.stdout
        print("  %-58s %s" % ("the worked example validates and lands in the right group",
                              "PASS" if ok else "FAIL"))
        if not ok and args.verbose:
            print("      %s" % (out.stdout + out.stderr)[:400])
        passed += ok
        failed += not ok

        seeded = L.load_registry(os.path.join(args.root, "data"))["measurements"][
            "measurement--glm53.dione-q4.brandonmusic-final25"]
        out = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                              "--receipt", ex], capture_output=True, text=True)
        ok = (out.returncode == 0
              and seeded["comparability"]["key"] in out.stdout
              and repr(seeded["metric"]["value"]) in out.stdout)
        print("  %-58s %s" % ("ingesting it reproduces the seeded Dione row's key and value",
                              "PASS" if ok else "FAIL"))
        passed += ok
        failed += not ok

        broken = os.path.join(tmp, "broken-seal.json")
        with open(ex, encoding="utf-8") as fh:
            sub = json.load(fh)
        sub["metric"]["value"] = 0.001
        with open(broken, "w", encoding="utf-8") as fh:
            json.dump(sub, fh)
        out = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                              "--receipt", broken], capture_output=True, text=True)
        ok = out.returncode == 5 and "seal does not verify" in out.stderr
        print("  %-58s %s" % ("a value edited inside a submission breaks its seal",
                              "PASS" if ok else "FAIL (exit %d)" % out.returncode))
        passed += ok
        failed += not ok

        nopanel = os.path.join(tmp, "unknown-panel.json")
        with open(ex, encoding="utf-8") as fh:
            sub = json.load(fh)
        sub["panel"]["panel_ref"] = "panel--nobody.has.this"
        sub["receipt_sha256"] = ""
        sub["receipt_sha256"] = L.sha256_hex(L.canonical_json(sub))
        with open(nopanel, "w", encoding="utf-8") as fh:
            json.dump(sub, fh)
        out = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                              "--receipt", nopanel], capture_output=True, text=True)
        ok = out.returncode == 4 and "cannot introduce a panel" in out.stderr
        print("  %-58s %s" % ("a measurement cannot introduce an unknown panel",
                              "PASS" if ok else "FAIL (exit %d)" % out.returncode))
        passed += ok
        failed += not ok

        # The red-team cases. Each of these was ACCEPTED before the check that stops it
        # existed. A submission is a bundle of CLAIMS about the panel, the teacher, the
        # scope and the measurer; a claim is worth nothing until it is checked against the
        # record it names.
        def redteam(label, want_exit, want_text, edits):
            path = os.path.join(
                tmp, "redteam-%s.json" % L.sha256_hex(label)[:16])
            with open(ex, encoding="utf-8") as fh:
                s = json.load(fh)
            for dotted, value in edits.items():
                node = s
                parts = dotted.split(".")
                for p in parts[:-1]:
                    node = node[p]
                node[parts[-1]] = value
            s["receipt_sha256"] = ""
            s["receipt_sha256"] = L.sha256_hex(L.canonical_json(s))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(s, fh)
            r = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                               "--receipt", path], capture_output=True, text=True)
            good = r.returncode == want_exit and want_text in r.stderr
            print("  %-58s exit %d  %s" % (label, want_exit,
                                           "PASS" if good else "FAIL (exit %d)" % r.returncode))
            if not good and args.verbose:
                print("      %s" % (r.stdout + r.stderr)[:500])
            return good

        # The forgery the red-team cases above could NOT reach: an inbound file that types
        # the maintainer's handle AND a maintainer-owned produced_by.repository. Both
        # strings are supplied by the submitter, so the old gate compared a claim against
        # itself and minted `self-measured` + `class: strict` in the flagship group. The
        # trust level now comes from the INVOCATION (--maintainer-attribution), which CI
        # never passes, so the same file can only ever land as third-party-reported.
        forged = os.path.join(tmp, "forged-attribution.json")
        with open(ex, encoding="utf-8") as fh:
            s_forge = json.load(fh)
        s_forge["measurer"] = {"name": "Evil Quants", "handle": L.MAINTAINER, "url": None,
                               "is_artifact_author": False}
        s_forge["produced_by"]["repository"] = L.MAINTAINER + "/glm53-fidelity-suite"
        s_forge["artifact"]["repository"] = "evilquant/GLM-5.3-Flash-SUPER"
        s_forge["receipt_sha256"] = ""
        s_forge["receipt_sha256"] = L.sha256_hex(L.canonical_json(s_forge))
        with open(forged, "w", encoding="utf-8") as fh:
            json.dump(s_forge, fh)
        out = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                              "--receipt", forged], capture_output=True, text=True)
        ok = out.returncode == 0 and "third-party-reported" in out.stdout \
            and "self-measured" not in out.stdout
        print("  %-58s %s" % ("a typed maintainer handle cannot mint self-measured",
                              "PASS" if ok else "FAIL"))
        if not ok and args.verbose:
            print("      %s" % (out.stdout + out.stderr)[:400])
        passed += ok
        failed += not ok

        out = subprocess.run([PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                              "--receipt", ex, "--maintainer-attribution"],
                             capture_output=True, text=True)
        ok = out.returncode == 0 and "self-measured" in out.stdout
        print("  %-58s %s" % ("...but the operator may assert it at the command line",
                              "PASS" if ok else "FAIL"))
        passed += ok
        failed += not ok

        for label, code, text, edits in (
            ("a row scored against a different teacher capture", 7, "different teacher",
             {"reference.teacher_receipt_sha256": "d" * 64}),
            ("an outsider claiming the maintainer's attribution", 8, "not a repository of theirs",
             {"produced_by.repository": "somebody-else/their-eval"}),
            ("the maintainer's NAME on a throwaway handle", 8, "identity",
             {"measurer.name": "malaiwah", "measurer.handle": "totally-not-malaiwah",
              "measurer.url": "https://huggingface.co/malaiwah",
              "produced_by.repository": "somebody-else/their-eval"}),
            ("a subset row with no subset_of_panel disclosure", 4, "no subset_of_panel disclosure",
             {"measurement_scope.covers_full_panel": False,
              "measurement_scope.scored_positions": 4094,
              "measurement_scope.subset_detail": "two windows"}),
            ("a row scoring more positions than the panel holds", 7, "was not this panel",
             {"measurement_scope.scored_positions": 51176,
              "measurement_scope.covers_full_panel": False,
              "measurement_scope.subset_detail": "n/a"}),
        ):
            ok = redteam(label, code, text, edits)
            passed += ok
            failed += not ok
    else:
        print("  (docs/examples/dione-q4.submission.json absent; skipped)")

    print()
    print("=" * 78)
    print("I. a withdrawn row must not rank, and a subset must be recordable")
    print("=" * 78)
    # REG-05. The renderer filtered nothing on `status`: a retracted row was tabled in rank
    # order like a live one and the word "retracted" appeared nowhere in README.md.
    rend = os.path.join(tmp, "retracted")
    shutil.copytree(os.path.join(args.root, "schema"), os.path.join(rend, "schema"))
    shutil.copytree(os.path.join(args.root, "data"), os.path.join(rend, "data"))
    shutil.copytree(os.path.join(args.root, "tables"), os.path.join(rend, "tables"),
                    dirs_exist_ok=True) if os.path.isdir(os.path.join(args.root, "tables")) else None
    for name in ("README.head.md",):
        src = os.path.join(args.root, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(rend, name))
    mp = os.path.join(rend, "data", "measurements.jsonl")
    rows = [json.loads(x) for x in open(mp, encoding="utf-8") if x.strip()]
    base = [r for r in rows if r["id"] == "measurement--glm53.k6-6bpw.brandonmusic-final25"][0]
    dead = json.loads(json.dumps(base))
    dead["id"] = "measurement--glm53.aaa-retracted-demo"
    dead["status"] = "retracted"
    d = 0.0011 - dead["metric"]["value"]
    dead["metric"]["value"] = 0.0011
    det = dead["determinism"]
    det["run_means"] = [0.0011] * len(det["run_means"])
    det["min_run_mean"] = det["max_run_mean"] = 0.0011
    for k in ("ci95_low", "ci95_high"):
        if dead["uncertainty"].get(k) is not None:
            dead["uncertainty"][k] += d
    for dd in dead.get("by_domain") or []:
        dd["mean"] += d
        for k in ("ci95_low", "ci95_high"):
            if dd.get(k) is not None:
                dd[k] += d
    dead.setdefault("disclosures", []).append(
        {"code": "record_note", "severity": "blocking", "affects_comparability": True,
         "detail": "WITHDRAWN: scorer bug found after publication."})
    rows.append(dead)
    L.write_jsonl(mp, rows)
    out = subprocess.run([PY, os.path.join(HERE, "registry_render.py"), "--root", rend],
                         capture_output=True, text=True)
    md = ""
    try:
        md = open(os.path.join(rend, "README.md"), encoding="utf-8").read()
    except IOError:
        pass
    ranked = [l for l in md.splitlines()
              if l.startswith("| ") and "**0.0011**" in l]
    ok = bool(md) and not ranked and "retracted" in md.lower() \
        and "WITHDRAWN: scorer bug" in md and "1148%" not in md
    print("  %-58s %s" % ("a retracted row is struck through, never ranked",
                          "PASS" if ok else "FAIL"))
    if not ok and args.verbose:
        print("      ranked=%r  has_retracted=%s  render_rc=%d"
              % (ranked[:1], "retracted" in md.lower(), out.returncode))
    passed += ok
    failed += not ok

    print()
    print("=" * 78)
    print("H. the CI diff gate must actually refuse a row no receipt generates")
    print("=" * 78)
    # `--assert-only-touched` is the gate that is supposed to stop a hand-edited published
    # number reaching data/. It built its `allowed` set and then compared it to nothing:
    # the only outcome for a changed data/ file was a warning, and main() exits non-zero on
    # errors only. Editing the K6 headline by hand passed it with exit 0.
    gitrepo = os.path.join(tmp, "ci-gate")
    os.makedirs(gitrepo)
    shutil.copytree(os.path.join(args.root, "schema"), os.path.join(gitrepo, "schema"))
    shutil.copytree(os.path.join(args.root, "data"), os.path.join(gitrepo, "data"))
    genp = os.path.join(tmp, "generated-none.json")
    with open(genp, "w", encoding="utf-8") as fh:
        json.dump({"measurements": [], "artifacts": [], "pipelines": []}, fh)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    quiet = dict(capture_output=True, text=True, cwd=gitrepo, env=env)
    have_git = subprocess.run(["git", "init", "-q"], **quiet).returncode == 0
    if have_git:
        subprocess.run(["git", "add", "-A"], **quiet)
        subprocess.run(["git", "commit", "-qm", "baseline"], **quiet)

        out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"), "--root", gitrepo,
                              "--assert-only-touched", genp], capture_output=True, text=True)
        ok = out.returncode == 0
        print("  %-58s %s" % ("an untouched checkout passes", "PASS" if ok else "FAIL"))
        passed += ok
        failed += not ok

        mp = os.path.join(gitrepo, "data", "measurements.jsonl")
        rows = [json.loads(x) for x in open(mp, encoding="utf-8") if x.strip()]
        target = "measurement--glm53.k6-6bpw.brandonmusic-final25"
        for r in rows:
            if r["id"] == target:
                new = 0.0100000000000001
                d = new - r["metric"]["value"]
                r["metric"]["value"] = new
                u = r["uncertainty"]
                for k in ("ci95_low", "ci95_high"):
                    if u.get(k) is not None:
                        u[k] += d
                det = r["determinism"]
                det["run_means"] = [new] * len(det["run_means"])
                det["min_run_mean"] = det["max_run_mean"] = new
        L.write_jsonl(mp, rows)
        out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"), "--root", gitrepo,
                              "--assert-only-touched", genp], capture_output=True, text=True)
        ok = out.returncode == 1 and target in out.stdout and "CI.GENERATED" in out.stdout
        print("  %-58s %s" % ("a hand-edited published value is REFUSED and named",
                              "PASS" if ok else "FAIL (exit %d)" % out.returncode))
        if not ok and args.verbose:
            print("      %s" % (out.stdout + out.stderr)[:400])
        passed += ok
        failed += not ok

        # A gate that cannot run must not read as a gate that passed.
        nogit = os.path.join(tmp, "ci-gate-nogit")
        shutil.copytree(gitrepo, nogit)
        shutil.rmtree(os.path.join(nogit, ".git"))
        out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"), "--root", nogit,
                              "--assert-only-touched", genp], capture_output=True, text=True)
        ok = out.returncode == 1 and "cannot diff" in out.stdout
        print("  %-58s %s" % ("outside a git checkout it fails CLOSED",
                              "PASS" if ok else "FAIL (exit %d)" % out.returncode))
        passed += ok
        failed += not ok
    else:
        print("  (git unavailable; CI diff gate cases skipped)")

    print()
    print("=" * 78)
    print("E2. the tools work in the shape a CONTRIBUTOR clones, not only in ours")
    print("=" * 78)
    # CONTRIBUTING §1 tells an outside contributor to clone the PUBLISHED dataset
    # repo and run `python tools/registry_validate.py --submission <receipt>`.
    # That repo has schema/, data/ and tools/ at its ROOT -- there is no
    # `registry/` directory above them. INGEST_CLOSURE names
    # `registry/tools/registry_add.py`, resolved against the parent of the
    # registry root, so in that shape it resolved to a path outside the clone and
    # the documented command died with an IOError traceback instead of printing
    # ACCEPTED or a named failure. Every outside submission check hit it.
    #
    # The invariant that must hold while fixing it: the harness_id is a function
    # of the CODE, not of where somebody cloned it. Same bytes -> same digests ->
    # same id, in both shapes.
    flat = os.path.join(tmp, "as-published")
    os.makedirs(flat, exist_ok=True)
    for sub in ("tools", "schema", "data"):
        src = os.path.join(args.root, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(flat, sub))
    probe = (
        "import sys, json; sys.path.insert(0, %r); "
        "import registry_add as A; print(json.dumps(A._ingest_digests()))"
        % os.path.join(flat, "tools"))
    out = subprocess.run([PY, "-c", probe], capture_output=True, text=True,
                         cwd=os.path.join(flat, "tools"))
    ok = out.returncode == 0
    print("  %-58s %s" % ("the ingest closure resolves in the published shape",
                          "PASS" if ok else "FAIL (%s)"
                          % (out.stderr.strip().splitlines() or [""])[-1][:70]))
    passed += ok
    failed += not ok
    if ok:
        here = subprocess.run(
            [PY, "-c", "import sys, json; sys.path.insert(0, %r); "
                       "import registry_add as A; print(json.dumps(A._ingest_digests()))"
                       % os.path.join(args.root, "tools")],
            capture_output=True, text=True, cwd=os.path.join(args.root, "tools"))
        same = (here.returncode == 0
                and json.loads(here.stdout) == json.loads(out.stdout))
        print("  %-58s %s" % ("...and yields the SAME digests as our own shape",
                              "PASS" if same else "FAIL"))
        passed += same
        failed += not same
        paths = [d["path"] for d in json.loads(out.stdout)]
        stable = all(p.startswith("registry/tools/") for p in paths)
        print("  %-58s %s" % ("...recording the suite-relative path either way",
                              "PASS" if stable else "FAIL (%s)" % paths))
        passed += stable
        failed += not stable

    print()
    print("=" * 78)
    print("F. a hand-edited value is caught by re-deriving the rows from their receipts")
    print("=" * 78)
    out = subprocess.run([PY, os.path.join(HERE, "seed_registry.py"), "--check"],
                         capture_output=True, text=True, cwd=args.root)
    ok = out.returncode == 0
    print("  %-58s %s" % ("seed_registry.py --check on the committed data", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok
    qwen_override = os.path.join(tmp, "tampered-qwen-receipts")
    os.mkdir(qwen_override)
    qwen_name = "kld5-10M-fp8.json"
    qwen_source = os.path.join(
        args.root, "protocol", "qwen38-receipts-public-8558b8c", qwen_name)
    qwen_tampered = os.path.join(qwen_override, qwen_name)
    shutil.copyfile(qwen_source, qwen_tampered)
    with open(qwen_tampered, "ab") as fh:
        fh.write(b"\n")
    qwen_env = dict(os.environ)
    qwen_env["FIDELITY_QWEN_RECEIPTS_DIR"] = qwen_override
    out = subprocess.run(
        [PY, os.path.join(HERE, "seed_registry.py"), "--check"],
        capture_output=True, text=True, cwd=args.root, env=qwen_env)
    ok = out.returncode != 0 and "differs from public pin" in (out.stdout + out.stderr)
    print("  %-58s %s" % (
        "a byte-edited frozen receipt is rejected by its public pin",
        "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    tampered = os.path.join(tmp, "tampered-data")
    shutil.copytree(os.path.join(args.root, "data"), tampered)
    tp = os.path.join(tampered, "measurements.jsonl")
    rows = [json.loads(l) for l in open(tp, encoding="utf-8")]
    for r in rows:
        if r["id"] == "measurement--glm53.k6-6bpw.brandonmusic-final25":
            r["metric"]["value"] = 0.0137
    open(tp, "w", encoding="utf-8").write("".join(L.canonical_json(r) + "\n" for r in rows))
    out = subprocess.run([PY, os.path.join(HERE, "seed_registry.py"), "--check", "--out", tampered],
                         capture_output=True, text=True)
    ok = out.returncode != 0 and "measurements" in (out.stdout + out.stderr)
    print("  %-58s %s" % ("a value edited by hand is caught as reseed drift", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    # ================================================================== REG batch
    print()
    print("=" * 78)
    print("R. attribution, digests and schema-walk edge cases")
    print("=" * 78)

    # REG-11: _owner did a SUBSTRING search, so any URL merely CONTAINING
    # "huggingface.co/malaiwah/" was attributed to us, and _ours used startswith(),
    # so a prefix squat passed too.
    spoofs = [
        ("https://evil.example.com/?ref=huggingface.co/malaiwah/x", False),
        ("https://huggingface.co/malaiwah-impostor/repo/r.json", False),
        ("https://github.com/evil/repo?x=huggingface.co/malaiwah/y", False),
        ("https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw/resolve/main/r.json", True),
        ("https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/raw/main/x.json", True),
        ("receipts/malaiwah/stream-k6-kld.json", True),
    ]
    bad = [(u, w, RV._ours(u)) for u, w in spoofs if RV._ours(u) != w]
    ok = not bad
    print("  %-58s %s" % ("REG-11 attribution parses the netloc, not a substring",
                          "PASS" if ok else "FAIL %r" % (bad,)))
    failed += not ok

    # REG-20: Python's `$` also matches before a trailing newline, so a digest or id
    # with one satisfied the anchored patterns and never byte-compared equal.
    ok = (not L.SHA256_RE.match("a" * 64 + "\n")
          and not L.ID_RE.match("measurement--x\n")
          and bool(L.SHA256_RE.match("a" * 64))
          and bool(L.ID_RE.match("measurement--x")))
    print("  %-58s %s" % ("REG-20 a trailing newline is not a valid id or digest",
                          "PASS" if ok else "FAIL"))
    failed += not ok

    # REG-23: _assert_supported recursed into DATA positions, so a spec-legal
    # object-valued const/default/enum member raised "unsupported keyword" and took
    # the whole validator down with exit 4.
    import _minischema as MS
    ok = True
    for schema in ({"const": {"a": 1}}, {"default": {"x": 2}}, {"enum": [{"a": 1}]}):
        try:
            MS._assert_supported(schema, "t")
        except MS.SchemaError:
            ok = False
    guard = 0
    for schema in ({"unevaluatedProperties": False}, {"items": {"bogusKw": 1}},
                   {"properties": {"p": {"bogusKw": 1}}}):
        try:
            MS._assert_supported(schema, "t")
        except MS.SchemaError:
            guard += 1
    print("  %-58s %s" % ("REG-23 object-valued const/default/enum load; unknown "
                          "keywords still raise", "PASS" if (ok and guard == 3) else
                          "FAIL (data ok=%s, guard %d/3)" % (ok, guard)))
    failed += not (ok and guard == 3)

    # REG-03: nothing recomputed a cited receipt digest, so a published receipt and
    # the row citing it could disagree in silence.
    import shutil as _sh
    rcp = os.path.join(args.root, "receipts", "malaiwah", "stream-k6-kld.json")
    if os.path.isfile(rcp):
        tamp = os.path.join(tmp, "receipt-tamper")
        for sub in ("schema", "data", "receipts"):
            _sh.copytree(os.path.join(args.root, sub), os.path.join(tamp, sub))
        vic = os.path.join(tamp, "receipts", "malaiwah", "stream-k6-kld.json")
        doc = json.load(open(vic))
        doc["measured_mean_kld"] = 0.001
        with open(vic, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        out = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"),
                              "--root", tamp], capture_output=True, text=True)
        ok = out.returncode != 0 and "RECEIPT-001" in out.stdout
        print("  %-58s %s" % ("REG-03 a row whose receipt was edited is REFUSED",
                              "PASS" if ok else "FAIL (exit %d)" % out.returncode))
        failed += not ok
        # and a root with no receipts/ at all must NOT error: the CI diff gate
        # validates a synthetic root holding only schema/ and data/.
        partial = os.path.join(tmp, "receipt-partial")
        for sub in ("schema", "data"):
            _sh.copytree(os.path.join(args.root, sub), os.path.join(partial, sub))
        out2 = subprocess.run([PY, os.path.join(HERE, "registry_validate.py"),
                               "--root", partial], capture_output=True, text=True)
        ok2 = "RECEIPT-001" not in out2.stdout
        print("  %-58s %s" % ("REG-03 a data-only root does not fire RECEIPT-001",
                              "PASS" if ok2 else "FAIL"))
        failed += not ok2

    # REG-24: a FAILED quality gate lived only in /quality_gate/passed on an
    # ingested row, so every rendered disclosure list showed it as clean.
    # seed_registry has emitted `quality_gate_failed` by hand since the runtime
    # rows; the ingest path did not. Probe BOTH outcomes -- a passing gate must
    # NOT produce the disclosure, or the check is just a constant.
    if receipts.get("stream_turbo405") or receipts.get("stream_k8"):
        srcpath = receipts.get("stream_turbo405") or receipts["stream_k8"]
        with open(srcpath, encoding="utf-8") as fh:
            base_receipt = json.load(fh)
        seen = {}
        for want_pass in (True, False):
            doc = dict(base_receipt)
            doc["quality_gate_passed"] = want_pass
            if isinstance(doc.get("quality_gate"), dict):
                doc["quality_gate"] = dict(doc["quality_gate"], passed=want_pass)
            path = os.path.join(tmp, "gate-%s.json" % want_pass)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            out = subprocess.run(
                [PY, os.path.join(HERE, "registry_add.py"), "--registry", args.root,
                 "from-receipt", "--receipt", path,
                 "--artifact", "artifact--turboderp.glm-5.3-flash-exl3-4.05bpw"
                 if receipts.get("stream_turbo405")
                 else "artifact--malaiwah.glm-5.3-flash-tr3-8bpw"] +
                (["--third-party-artifact"] if receipts.get("stream_turbo405") else []) +
                [
                 "--panel", "panel--glm53.brandonmusic.final25",
                 "--reference", "reference--brandonmusic.glm53-bf16-fp32-logits.final25",
                 "--pipeline", "pipeline--malaiwah.glm53-stream-packed-kld",
                 "--lane", "streaming", "--scored-positions", "51175", "--contexts", "25",
                 "--direction", "reference_to_candidate", "--dry-run"],
                capture_output=True, text=True)
            codes = []
            if out.returncode == 0 and "{" in out.stdout:
                row = json.loads(out.stdout[out.stdout.index("{"):])
                codes = [d["code"] for d in row["disclosures"]]
            seen[want_pass] = codes
        ok = ("quality_gate_failed" in seen[False]
              and "quality_gate_failed" not in seen[True])
        print("  %-58s %s" % ("REG-24 a failed gate becomes a disclosure, a passed one does not",
                              "PASS" if ok else "FAIL"))
        if not ok and args.verbose:
            print("      passed-gate codes: %s" % seen[True])
            print("      failed-gate codes: %s" % seen[False])
        passed += ok
        failed += not ok

    print()
    print("=" * 78)
    print("S. an all-native format census is never a quantized assignment (SCOPE-011)")
    print("=" * 78)
    # review-science S1-1: the published GLM-5.3 scopes carried `quantized:mixed` rows
    # whose own census named nothing but native groups (bf16 router weights beside an
    # fp32 correction bias). fp8_scope.assignments_from_census has written those as
    # treatment=native since 56ff020; the validator must refuse the old rows so the
    # contradiction cannot be republished. Fixtures are the OLD published strings.
    _src_drowzeys = (
        "read from drowzeys/keys-GLM-5.3-EXL3@ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5 "
        "model.safetensors.index.json + shard headers: a class is exl3 trellis when its weights "
        "are stored as trellis/suh/svh payload groups (codebook from the object each module "
        "carries: mcg:768, mul1:56832; declared bits 3 by quantization_config), fp8_e4m3 when a "
        "_scale_inv sibling exists, native otherwise (stored dtype from the shard headers).")
    _src_wrld = (
        "read from wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1@47af23347db743b4666d952e2eb48f2b01c3fede "
        "model.safetensors.index.json + shard headers: a class is exl3 trellis when its weights "
        "are stored as trellis/suh/svh payload groups (codebook from the object each module "
        "carries: mcg:57600; declared bits 4 by quantization_config), fp8_e4m3 when a "
        "_scale_inv sibling exists, native otherwise (stored dtype from the shard headers).")
    _mixes = "class mixes formats on the same layers (SCOPE-004 admits one row per class and layer_range): "
    all_native = {
        "artifact--drowzeys.keys-glm-5.3-exl3": [
            {"tensor_class": "attn.other", "layer_range": "0-77", "treatment": "quantized",
             "format": "mixed", "bits_per_weight": None,
             "note": _mixes + "156 x native:bf16@16 (model.layers.N.self_attn.kv_a_layernorm.weight, "
                     "model.layers.N.self_attn.q_a_layernorm.weight); 105 x native:fp16@16 "
                     "(model.layers.N.self_attn.indexer.k_norm.bias, "
                     "model.layers.N.self_attn.indexer.k_norm.weight, "
                     "model.layers.N.self_attn.indexer.weights_proj.weight). " + _src_drowzeys},
            {"tensor_class": "moe.router", "layer_range": "3-77", "treatment": "quantized",
             "format": "mixed", "bits_per_weight": None,
             "note": _mixes + "75 x native:fp16@16 (model.layers.N.mlp.gate.weight); 75 x native:fp32@32 "
                     "(model.layers.N.mlp.gate.e_score_correction_bias). " + _src_drowzeys},
            {"tensor_class": "mtp", "layer_range": "78", "treatment": "quantized",
             "format": "mixed", "bits_per_weight": None,
             "note": _mixes + "7 x native:bf16@16 (model.layers.N.enorm.weight, model.layers.N.hnorm.weight, "
                     "model.layers.N.input_layernorm.weight); 783 x native:fp16@16 "
                     "(model.layers.N.eh_proj.weight, model.layers.N.mlp.experts.E.down_proj.weight, "
                     "model.layers.N.mlp.experts.E.gate_proj.weight); 1 x native:fp32@32 "
                     "(model.layers.N.mlp.gate.e_score_correction_bias). " + _src_drowzeys},
        ],
        "artifact--wrldsuksgo2mars.glm-5.3-exl3-k4-v1": [
            {"tensor_class": "moe.router", "layer_range": "3-77", "treatment": "quantized",
             "format": "mixed", "bits_per_weight": None,
             "note": _mixes + "75 x native:bf16@16 (model.layers.N.mlp.gate.weight); 75 x native:fp32@32 "
                     "(model.layers.N.mlp.gate.e_score_correction_bias). " + _src_wrld},
        ],
    }

    def _scope011(assignments):
        C = {"artifacts": {aid: {"scope": {"policy": "mixed", "head_policy": "native",
                                           "kv_cache_dtype": "bf16", "assignments": rows}}
                           for aid, rows in assignments.items()},
             "measurements": {}}
        rep = RV.Report()
        RV.check_scope(C, rep)
        return [f for f in rep.errors if f["check"] == "SCOPE-011"]

    hits = _scope011(all_native)
    hit_ids = sorted(f["id"] for f in hits)
    want_ids = ["artifact--drowzeys.keys-glm-5.3-exl3"] * 3 + ["artifact--wrldsuksgo2mars.glm-5.3-exl3-k4-v1"]
    ok = hit_ids == want_ids and all(
        "native:" in f["message"] and "quantized:" not in f["message"] for f in hits)
    print("  %-58s %s" % ("SCOPE-011 the OLD quantized:mixed all-native rows are REFUSED (3+1)",
                          "PASS" if ok else "FAIL (%s)" % hit_ids))
    passed += ok
    failed += not ok

    # The same censuses with the treatment fp8_scope writes today must be clean.
    relabelled = {aid: [dict(x, treatment="native") for x in rows] for aid, rows in all_native.items()}
    ok = not _scope011(relabelled)
    print("  %-58s %s" % ("SCOPE-011 treatment=native with the same census is clean", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    # A census with ANY quantized group is a genuinely mixed class: silent.
    mixed = {"artifact--x.mixed": [
        {"tensor_class": "mtp", "layer_range": "78", "treatment": "quantized",
         "format": "mixed", "bits_per_weight": None,
         "note": _mixes + "12 x native:bf16@16 (model.layers.N.eh_proj.weight); 1 x native:fp32@32 "
                 "(model.layers.N.mlp.gate.e_score_correction_bias); 75 x quantized:exl3-mcg@4 "
                 "(model.layers.N.mlp.experts.E.down_proj.weight). " + _src_wrld},
        {"tensor_class": "attn.other", "layer_range": "0-77", "treatment": "quantized",
         "format": "mixed", "bits_per_weight": None,
         "note": _mixes + "219 x native:bf16@16 (model.layers.N.self_attn.indexer.k_norm.bias); "
                 "42 x quantized:fp8_e4m3@8 (model.layers.N.self_attn.indexer.wk.weight). " + _src_wrld},
    ]}
    ok = not _scope011(mixed)
    print("  %-58s %s" % ("SCOPE-011 a census with a quantized group is silent", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    # A prose note carries no census: nothing to contradict, so the rule stays silent
    # even when the prose mentions native storage.
    prose = {"artifact--x.prose": [
        {"tensor_class": "moe.experts", "layer_range": "all", "treatment": "quantized",
         "format": "exl3-mcg", "bits_per_weight": 4.0,
         "note": "57600 tensors: model.layers.N.mlp.experts.E.down_proj.weight. " + _src_wrld},
        {"tensor_class": "attn.o", "layer_range": "all", "treatment": "quantized",
         "format": "fp8_e4m3", "bits_per_weight": 8,
         "note": "native bf16 elsewhere; 78 tensors quantized to fp8_e4m3 with native:bf16 scales"},
    ]}
    ok = not _scope011(prose)
    print("  %-58s %s" % ("SCOPE-011 a prose-only note is silent", "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    # The catalogue must know the code, or the finding is unexplainable to a reader.
    with open(os.path.join(args.root, "schema", "invariants.json"), encoding="utf-8") as fh:
        _inv = json.load(fh)
    _entry = [i for i in _inv.get("invariants", []) if i.get("id") == "SCOPE-011"]
    ok = len(_entry) == 1 and _entry[0].get("severity") == "error"
    print("  %-58s %s" % ("SCOPE-011 is catalogued in schema/invariants.json as an error",
                          "PASS" if ok else "FAIL"))
    passed += ok
    failed += not ok

    print()
    print("=" * 78)
    print("E. the tools import no networking library")
    print("=" * 78)
    for tool in ("registry_validate.py", "registry_add.py"):
        flag = "--offline-selftest" if tool == "registry_validate.py" else "offline-selftest"
        out = subprocess.run([PY, os.path.join(HERE, tool), flag], capture_output=True, text=True)
        ok = out.returncode == 0 and "none" in out.stdout
        print("  %-58s %s" % (tool, "PASS" if ok else "FAIL"))
        if not ok:
            print("      %s" % out.stdout.strip()[:200])
        passed += ok
        failed += not ok

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("\nfixtures kept in %s" % tmp)
    print("\n%s\n%d passed, %d failed" % ("-" * 78, passed, failed))
    return 0 if failed == 0 else 1


def _find_receipts():
    """Locate the real receipts if they are still on this machine. The self-test degrades
    gracefully rather than inventing stand-ins."""
    out = {}
    scratch = os.environ.get("QFR_RECEIPT_DIR", "")
    cands = [scratch] if scratch else []
    cands += ["/private/tmp/claude-501/-Users-mbelleau-Projects-GLM/"
              "c1546622-1c41-4561-ba68-92b6b9cb9811/scratchpad"]
    for d in cands:
        for key, fn in (("k6_five", "k6-five-run-kld.json"), ("k6_packed", "k6-packed-kld.json"),
                        ("dione", "dione-q4-kld.json"), ("crosscheck", "crosscheck-brandonmusic.json")):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                out.setdefault(key, p)
        for key, fn in (("v44_fp8", "regbuild/brandon/v44-fp8.json"),):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                out.setdefault(key, p)
    # The streaming lane's receipts are committed to this repository, so these are always
    # present -- for everyone, not only on the machine that ran them.
    here = os.path.join(L.repo_root(__file__), "receipts", "malaiwah")
    for key, fn in (("stream_k6", "stream-k6-kld.json"),
                    ("stream_k6_verdict", "stream-k6-verdict.json"),
                    ("stream_k8", "stream-k8-kld.json"),
                    ("stream_bf16", "stream-bf16-kld.json"),
                    ("stream_turbo405", "stream-turbo-4.05bpw-kld.json"),
                    ("stream_tr34", "stream-tr3-4bpw-kld.json"),
                    ("stream_dione30", "stream-dione-3.0bpw-kld.json")):
        fp = os.path.join(here, fn)
        if os.path.exists(fp):
            out[key] = fp
    q = "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/kld5-10M-fp8.json"
    if os.path.exists(q):
        out["qwen_report"] = q
    return out


if __name__ == "__main__":
    sys.exit(main())
