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
    a = C["measurements"]["measurement--glm53.k6-6bpw.brandonmusic-final25"]
    a["disclosures"].append({"code": "reduced_run_count", "severity": "caveat",
                             "detail": "and also this", "affects_comparability": False})
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


def m_non_canonical_line(C):
    return None  # handled specially below


MUTATIONS = [
    ("forged-comparability-key", m_forge_key),
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
                      "--scored-positions", "51175", "--dry-run"])
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
                      "--lane", "streaming", "--dry-run"])
             if receipts.get("stream_k8") else None)):
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
            path = os.path.join(tmp, "redteam-%d.json" % abs(hash(label)))
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

        for label, code, text, edits in (
            ("a row scored against a different teacher capture", 7, "different teacher",
             {"reference.teacher_receipt_sha256": "d" * 64}),
            ("an outsider claiming the maintainer's attribution", 8, "not a repository of theirs",
             {"produced_by.repository": "somebody-else/their-eval"}),
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
    print("F. a hand-edited value is caught by re-deriving the rows from their receipts")
    print("=" * 78)
    out = subprocess.run([PY, os.path.join(HERE, "seed_registry.py"), "--check"],
                         capture_output=True, text=True, cwd=args.root)
    ok = out.returncode == 0
    print("  %-58s %s" % ("seed_registry.py --check on the committed data", "PASS" if ok else "FAIL"))
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
                    ("stream_k8", "stream-k8-kld.json")):
        fp = os.path.join(here, fn)
        if os.path.exists(fp):
            out[key] = fp
    q = "/Users/mbelleau/Projects/qwen38-27b-exl3/receipts/kld5-10M-fp8.json"
    if os.path.exists(q):
        out["qwen_report"] = q
    return out


if __name__ == "__main__":
    sys.exit(main())
