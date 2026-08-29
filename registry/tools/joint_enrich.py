"""Joint-standard enrichment of the seeded measurement rows.

WHAT THIS ADDS, and why each piece is here.

Before 2026-08-29 every measurement row in this registry carried
``uncertainty: {"method": "none"}`` -- a point estimate with no interval.
brandonmusic's proposed community standard asks for a window-clustered interval,
a per-domain table, run-to-run spread combined in quadrature, and a protocol
hash stamped into the output.  All of that is recoverable from data we already
published, with no GPU: our receipts carry per-window means, every window is
exactly 2047 scored positions, and the panel's domain assignment is public.

So this module, from ``registry/protocol/per-window/*.json``:

  * computes the window-block bootstrap (percentile + BCa, B=5000, seed
    20260829) and the cluster-robust SE, and writes them into ``uncertainty``;
  * computes ``sigma_run`` from the row's OWN ``determinism.run_means`` -- never
    from a hardcoded constant -- and combines it with the clustered SE in
    quadrature;
  * writes the per-domain table into ``by_domain``;
  * stamps the frozen protocol's two hashes into ``protocol``;
  * names the scope (``panel25``) and records the calibration-overlap scan that
    defines the alternative scope;
  * emits a SECOND row per series on the calibration-clean ``clean17`` scope.

The clean17 rows are not a correction and do not supersede anything.  They are
the same measurement read over a different, smaller window set, and they exist
because brandonmusic's 13-gram scan showed the sealed panel is not
calibration-clean.  The two scopes disagree in DIRECTION between contributors --
every malaiwah row falls 12.6-16.2% on the clean scope while his own 4bpw row
rises 1.6% -- so neither scope can stand in for the other and both are published.

No number here is invented: every value is a deterministic function of
per-window means this registry already publishes, and ``make check`` re-derives
all of it from those files.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REGISTRY = os.path.dirname(_HERE)
_REPO = os.path.dirname(_REGISTRY)
PROTOCOL_DIR = os.path.join(_REGISTRY, "protocol")
PER_WINDOW_DIR = os.path.join(PROTOCOL_DIR, "per-window")
PROTOCOL_FILE = os.path.join(PROTOCOL_DIR, "glm53-joint-kld-protocol.v1.json")
SELECTION_FILE = os.path.join(PROTOCOL_DIR, "window-selection.brandonmusic-final25.json")

# bin/jointstd is the implementation; importing it keeps ONE bootstrap in the
# repository instead of two that can drift.
sys.path.insert(0, os.path.join(_REPO, "bin"))
import registry_lib as _L                    # noqa: E402
from jointstd import protocol as _proto      # noqa: E402
from jointstd import stats as _stats         # noqa: E402

BOOTSTRAP_B = 5000
DOMAIN_BOOTSTRAP_B = 1000
SEED = 20260829

# measurement id -> per-window file basename.  Only rows whose receipts actually
# carry per-window arrays appear here; the two that do not (the BF16 streaming
# floor and the Dione Q4, both scalar-only receipts) are deliberately absent and
# are named in NOT_RECOMPUTABLE so the omission is visible rather than silent.
SERIES = {
    "measurement--glm53.k6-6bpw.brandonmusic-final25": "k6-sealed",
    "measurement--glm53.k6-6bpw-stream.brandonmusic-final25": "k6-streaming",
    "measurement--glm53.k8-8bpw-stream.brandonmusic-final25": "k8-streaming",
    "measurement--glm53.official-fp8.brandonmusic-final25.crossstack": "fp8-crossstack",
    "measurement--glm53.bf16-replay-floor.brandonmusic-final25": "bf16-floor-crossstack",
    "measurement--glm53.brandonmusic-4bpw.brandonmusic-final25": "brandonmusic-4bpw",
}

NOT_RECOMPUTABLE = {
    "measurement--glm53.bf16-stream-floor.brandonmusic-final25":
        "receipt registry/receipts/malaiwah/stream-bf16-kld.json is scalar-only "
        "(run_means + a tokenwise digest, no per_window block)",
    "measurement--glm53.dione-q4.brandonmusic-final25":
        "receipt reports/dione-q4-packed-kld.json is scalar-only (no per_window block)",
}

CLEAN_SUFFIX = ".clean17"


# --------------------------------------------------------------------------
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round(x: Optional[float], sig: int = 15) -> Optional[float]:
    """Round to SIGNIFICANT digits, not decimal places.

    A double carries about 15.95 significant digits, so trimming to 15 discards
    exactly the bits that differ between interpreters and library builds and
    keeps everything a reader could ever use. This is what makes
    `make reseed-check` give the same answer under python 3.9 and 3.12.
    """
    if x is None:
        return None
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    exp = math.floor(math.log10(abs(x)))
    return float(round(x, sig - 1 - exp))


class _Context:
    """Everything the enrichment needs, loaded once."""

    def __init__(self) -> None:
        self.proto = _proto.load(PROTOCOL_FILE)
        self.selection = _load_json(SELECTION_FILE)
        self.selection_sha = _sha256_file(SELECTION_FILE)
        self.clean = list(self.selection["selected_windows"])
        self.per_window: Dict[str, List[Dict[str, Any]]] = {}
        for mid, slug in SERIES.items():
            doc = _load_json(os.path.join(PER_WINDOW_DIR, slug + ".json"))
            self.per_window[mid] = doc["per_window"]
        scan = {
            "method": self.selection["method"],
            "ngram_n": self.selection["ngram_n"],
            "threshold": self.selection["threshold"],
            "windows_scanned": len(self.selection["per_window"]),
            "windows_excluded": len(self.selection["excluded_windows"]),
            "max_retained_fraction": max(
                w["shared_ngram_fraction"] for w in self.selection["per_window"]
                if w["window_id"] in set(self.clean)),
            "excluded_windows": [e["window_id"] for e in self.selection["excluded_windows"]],
            "note": "Reproduced independently from the published token arrays; every "
                    "window's shared-gram count and fraction matches brandonmusic's "
                    "window_selection.json exactly.",
        }
        self.scan = scan

    def protocol_block(self) -> Dict[str, Any]:
        b = self.proto.stamp()
        return {
            "schema": b["protocol_schema"],
            "file": b["protocol_file"],
            "file_sha256": b["protocol_file_sha256"],
            "scoring_sha256": b["protocol_scoring_sha256"],
            "note": "Two rows are protocol-comparable when scoring_sha256 matches; a "
                    "differing file_sha256 with a matching scoring hash means only that "
                    "identity metadata was edited.",
        }


def _uncertainty(per_window: List[Dict[str, Any]], run_means: Optional[List[float]],
                 gate: float, sigma_override: Optional[float] = None,
                 sigma_runs: Optional[int] = None,
                 sigma_note: Optional[str] = None) -> Dict[str, Any]:
    summary = _stats.se_from_window_summaries(per_window)
    means = {w["window_id"]: float(w["mean"]) for w in per_window}
    bs = _stats.window_block_bootstrap(means, b=BOOTSTRAP_B, seed=SEED, backend="auto")
    unc: Dict[str, Any] = {
        "method": "window_block_bootstrap_bca",
        "interval_kind": "bca",
        "cluster_unit": "window",
        "ci95_low": _round(bs["ci95_bca"][0]),
        "ci95_high": _round(bs["ci95_bca"][1]),
        "clusters": summary["n_clusters_window"],
        "samples": summary["n"],
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_seed": SEED,
        "se_clustered": _round(summary["se_clustered_window"]),
    }
    if "se_naive" in summary:
        unc["se_naive"] = _round(summary["se_naive"])
        unc["deff"] = _round(summary["deff_window"])
    note = [
        "Window-block bootstrap, B=%d, seed=%d; BCa endpoints quoted, percentile "
        "endpoints [%.9f, %.9f]." % (BOOTSTRAP_B, SEED,
                                     bs["ci95_percentile"][0], bs["ci95_percentile"][1]),
    ]
    if sigma_override is not None and sigma_runs and sigma_runs >= 2:
        q = _stats.combine_quadrature(summary["se_clustered_window"],
                                      sigma_override, gate=gate)
        unc["sigma_run"] = sigma_override
        unc["sigma_run_runs"] = int(sigma_runs)
        unc["se_total"] = _round(q["se_total"])
        note.append(sigma_note or "")
    elif sigma_override is None and sigma_note and not run_means:
        note.append(sigma_note)
    elif run_means and len(run_means) >= 2:
        sr = _stats.sigma_run(run_means)
        q = _stats.combine_quadrature(summary["se_clustered_window"],
                                      sr["sigma_run"], gate=gate)
        unc["sigma_run"] = _round(sr["sigma_run"])
        unc["sigma_run_runs"] = sr["runs"]
        unc["se_total"] = _round(q["se_total"])
        note.append("sigma_run over %d cold runs = %.3e; SE_total = hypot(SE_clustered, "
                    "sigma_run) = %.6e (%s)."
                    % (sr["runs"], sr["sigma_run"], q["se_total"],
                       "run term negligible" if q["gate_ok"]
                       else "RUN TERM NOT NEGLIGIBLE"))
        if sr["runs"] == 2:
            note.append("Two cold runs give sigma = |delta|/sqrt(2) with one degree of "
                        "freedom; the joint protocol asks for three.")
    else:
        note.append("sigma_run not estimable: fewer than two cold runs.")
    note.append("Percentiles of the per-token distribution are NOT quoted: they are not "
                "derivable from per-window summaries.")
    unc["note"] = " ".join(note)
    return unc


def _by_domain(per_window: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in _stats.domain_table(per_window, b=DOMAIN_BOOTSTRAP_B, seed=SEED,
                                 backend="auto"):
        row = {
            "domain": r["domain"],
            "windows": r["windows"],
            "scored_positions": r["n"],
            "mean": _round(r["mean"]),
        }
        if r.get("se_clustered_window") is not None:
            row["se_clustered"] = _round(r["se_clustered_window"])
        if r.get("ci95_bca"):
            row["ci95_low"] = _round(r["ci95_bca"][0])
            row["ci95_high"] = _round(r["ci95_bca"][1])
            row["interval_kind"] = "bca"
            row["note"] = "BCa over %d window resamples (B=%d)." % (r["windows"],
                                                                    DOMAIN_BOOTSTRAP_B)
        rows.append(row)
    return rows


def _clean_row(row: Dict[str, Any], ctx: _Context,
               per_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The same measurement read over the calibration-clean window set."""
    clean = [w for w in per_window if w["window_id"] in set(ctx.clean)]
    summary = _stats.se_from_window_summaries(clean)
    new = json.loads(json.dumps(row))          # deep copy of the enriched panel row
    new["id"] = row["id"] + CLEAN_SUFFIX
    new["metric"] = dict(row["metric"], value=_round(summary["mean"]))
    new["auxiliary_metrics"] = {k: None for k in row.get("auxiliary_metrics", {})}
    det = new["determinism"]
    bitwise = det.get("identical_across_runs") is True
    for k in ("run_means", "min_run_mean", "max_run_mean",
              "population_stddev_of_run_means"):
        det.pop(k, None)
    det["note"] = ((det.get("note", "") + " ") if det.get("note") else "") + (
        "Per-run means are dropped on this scope: the published run_means are the "
        "panel25 means and would not be this row's per-run means. The determinism "
        "EVIDENCE is unaffected -- a content hash of the per-token array is a property "
        "of the arrays, not of which windows you average.")
    # Bitwise-identical runs give sigma_run = 0 exactly on EVERY subset of windows,
    # because every per-run per-token array is the same array. Anything else is not
    # recoverable from panel-scope run means, so it is omitted rather than guessed.
    sigma = 0.0 if bitwise else None
    new["uncertainty"] = _uncertainty(
        clean, None, ctx.proto.sigma_run_gate,
        sigma_override=sigma, sigma_runs=det.get("run_count") if bitwise else None,
        sigma_note=("sigma_run is exactly 0.0 on any window subset because the cold runs "
                    "are bitwise identical (%s, one distinct content hash)."
                    % det.get("evidence_kind")) if bitwise else
                   ("sigma_run is not quoted on this scope: the runs are not bitwise "
                    "identical and per-run clean-scope means are not recoverable from "
                    "the published panel-scope run means."))
    new["by_domain"] = _by_domain(clean)
    ms = dict(new["measurement_scope"])
    ms.update({
        "scored_positions": summary["n"],
        "contexts": len(clean),
        "covers_full_panel": False,
        "scope_name": "clean17",
        "subset_detail":
            "The %d of 25 sealed windows that survive the 13-gram calibration-overlap "
            "scan at threshold %.2f. Six axis4_reasoning_termination windows share "
            "37-39%% of their 13-grams with calibration-role windows, and final-0021 "
            "(7.1%%) and final-0022 (5.8%%) also exceed the threshold, so this is NOT a "
            "whole-domain drop. Recomputed from the same per-window means as the "
            "panel25 row; no new measurement was made."
            % (len(clean), ctx.selection["threshold"]),
    })
    new["measurement_scope"] = ms
    new["panel_ref"] = CLEAN_PANEL
    new["reference_ref"] = CLEAN_REFERENCE
    cmp_ = json.loads(json.dumps(row["comparability"]))
    cmp_["key_inputs"]["panel_id"] = CLEAN_PANEL
    cmp_["key_inputs"]["reference_id"] = CLEAN_REFERENCE
    cmp_["key"] = _L.comparability_key(cmp_["key_inputs"])
    cmp_["class"] = "advisory"
    if cmp_.get("bias") and cmp_["bias"].get("floor_measurement_ref"):
        floor = cmp_["bias"]["floor_measurement_ref"]
        if floor in SERIES:
            # the floor has a clean sibling, so point at the SCOPE-MATCHED one
            cmp_["bias"]["floor_measurement_ref"] = floor + CLEAN_SUFFIX
            cmp_["bias"]["detail"] = (
                cmp_["bias"]["detail"] + " Scope-matched: this row's floor reference is "
                "the clean17 floor, not the panel25 one. Subtracting a floor measured on "
                "a different WINDOW SET is the same class of error as subtracting one "
                "measured on a different LANE, and this registry refuses both.")
        else:
            cmp_["bias"]["floor_measurement_ref"] = None
            cmp_["bias"]["detail"] = (
                cmp_["bias"]["detail"] + " NO FLOOR ON THIS SCOPE: the same-lane floor "
                "(%s) has a scalar-only receipt with no per-window array, so it cannot be "
                "recomputed on the calibration-clean window set. Rather than borrow the "
                "panel25 floor -- a cross-scope subtraction -- this row carries no floor "
                "reference at all." % floor)
    new["comparability"] = cmp_
    delta = summary["mean"] - row["metric"]["value"]
    rel = 100.0 * delta / row["metric"]["value"] if row["metric"]["value"] else float("nan")
    new["notes"] = (
        "Calibration-clean scope recompute of %s. panel25 %.15f -> clean17 %.15f "
        "(%+.15f, %+.2f%%). Not a correction and not a supersession: the two scopes "
        "answer different questions and move different contributors' rows in opposite "
        "directions. Never compare a clean17 value against a panel25 value."
        % (row["id"], row["metric"]["value"], summary["mean"], delta, rel))
    codes = {d["code"] for d in new.get("disclosures", [])}
    disc = [d for d in new.get("disclosures", []) if d["code"] != "no_known_deviations"]
    if "subset_of_panel" not in codes:
        disc.append({
            "code": "subset_of_panel",
            "severity": "caveat",
            "affects_comparability": True,
            "detail": "17 of the panel's 25 sealed windows (34,799 of 51,175 scored "
                      "positions). The excluded 8 are the windows the calibration-overlap "
                      "scan flags; see measurement_scope.calibration_overlap_scan.",
        })
    if "calibration_panel_overlap" not in codes:
        disc.append({
            "code": "calibration_panel_overlap",
            "severity": "info",
            "detail": "This row EXISTS because of calibration overlap in the panel: the "
                      "13-gram scan found one whole domain sharing 37-39% of its 13-grams "
                      "with calibration-role windows despite clean document-level "
                      "separation. The panel25 sibling includes those windows.",
        })
    new["disclosures"] = disc
    new["cross_refs"] = row.get("cross_refs")
    return new


# --------------------------------------------------------------------------
def apply(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich the seeded rows in place and append the clean-scope siblings."""
    ctx = _Context()
    proto_block = ctx.protocol_block()
    out: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []

    for row in rows:
        est = row.get("estimator") or {}
        # A property of the SCORER, so it follows the same rule logits_dtype does:
        # asserted only where we ran the code, unknown for somebody else's number.
        if "vocab_masking_policy" not in est:
            if row.get("provenance", {}).get("measured_by") == "self-measured":
                est["vocab_masking_policy"] = "full_stored_vocab"
                est["padded_columns_masked"] = 0
            else:
                est["vocab_masking_policy"] = "unknown"
                est["padded_columns_masked"] = None

        mid = row["id"]
        if mid in SERIES:
            pw = ctx.per_window[mid]
            got = sum(w["count"] for w in pw)
            want = row["measurement_scope"]["scored_positions"]
            if want is not None and got != want:
                raise SystemExit(
                    "joint_enrich: %s per-window positions %d != row scored_positions %d"
                    % (mid, got, want))
            recomputed = _stats.se_from_window_summaries(pw)["mean"]
            if abs(recomputed - row["metric"]["value"]) > 5e-15:
                raise SystemExit(
                    "joint_enrich: %s per-window mean %.17g does not reproduce the "
                    "published value %.17g" % (mid, recomputed, row["metric"]["value"]))
            row["uncertainty"] = _uncertainty(
                pw, row["determinism"].get("run_means"), ctx.proto.sigma_run_gate)
            row["by_domain"] = _by_domain(pw)
            row["protocol"] = dict(proto_block)
            ms = row["measurement_scope"]
            ms["scope_name"] = "panel25"
            ms["scope_selection_file"] = os.path.relpath(SELECTION_FILE, _REGISTRY)
            ms["scope_selection_sha256"] = ctx.selection_sha
            ms["calibration_overlap_scan"] = dict(ctx.scan)
            out.append(row)
            clean_rows.append(_clean_row(row, ctx, pw))
        else:
            if mid in NOT_RECOMPUTABLE:
                ms = row["measurement_scope"]
                ms["scope_name"] = "panel25"
                row["protocol"] = dict(proto_block)
                row["notes"] = ((row.get("notes", "") + " ") if row.get("notes") else "") + (
                    "No clean17 sibling: %s, so the calibration-clean scope cannot be "
                    "recomputed without re-running the measurement."
                    % NOT_RECOMPUTABLE[mid])
            out.append(row)
    return out + clean_rows


# ==========================================================================
# The clean17 scope is a DIFFERENT PANEL, not a footnote on the same one.
#
# The registry's CMP-003 invariant caught this the moment the clean-scope rows
# were first written under the parent panel's comparability key: two rows that
# score a different number of positions must not sit in one comparison group.
# That is the correct answer, and the registry already had a precedent for it
# (panel--glm53.brandonmusic.final-0000). So the clean scope gets its own
# derived panel record and its own derived reference record, which gives it its
# own comparability key and makes a clean17-vs-panel25 table structurally
# impossible rather than merely discouraged.
# ==========================================================================
CLEAN_PANEL = "panel--glm53.brandonmusic.final25-clean17"
CLEAN_REFERENCE = "reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17"
PARENT_PANEL = "panel--glm53.brandonmusic.final25"
PARENT_REFERENCE = "reference--brandonmusic.glm53-bf16-fp32-logits.final25"


def _clean_windows_meta() -> List[Dict[str, Any]]:
    sel = _load_json(SELECTION_FILE)
    keep = set(sel["selected_windows"])
    return [w for w in sel["per_window"] if w["window_id"] in keep]


def panels(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The derived calibration-clean panel record."""
    import hashlib

    parent = next(p for p in existing if p["id"] == PARENT_PANEL)
    sel = _load_json(SELECTION_FILE)
    meta = _clean_windows_meta()
    shard = {w["window_id"]: w["token_ids_sha256"] for w in meta
             if w.get("token_ids_sha256")}
    if len(shard) != len(meta):
        raise SystemExit("joint_enrich: the selection file is missing token digests")
    # identity over TOKEN CONTENT: a manifest digest of the member windows'
    # own token-id digests, in window order. Never over a report file.
    manifest = "\n".join("%s %s" % (k, shard[k]) for k in sorted(shard)).encode("utf-8")
    positions = sum(int(w["prediction_positions"]) for w in meta)
    strata: Dict[str, Dict[str, Any]] = {}
    for w in meta:
        strata.setdefault(w["domain"], {"contexts": 0})["contexts"] += 1
    dropped = [e["window_id"] for e in sel["excluded_windows"]]
    rec = {
        "schema_version": parent["schema_version"],
        "id": CLEAN_PANEL,
        "name": "brandonmusic panel v1, calibration-clean subset -- %d of 25 final windows"
                % len(meta),
        "author": parent["author"],
        "model_scope": list(parent["model_scope"]),
        "tokenizer": dict(parent["tokenizer"]),
        "structure": {
            "contexts": len(meta), "context_length": 2048, "positions_per_context": 2047,
            "positions_per_context_min": 2047, "positions_per_context_max": 2047,
            "scored_positions_total": positions,
            "scoring_window": dict(parent["structure"]["scoring_window"]),
            "strata": strata,
        },
        "corpus": dict(parent["corpus"]),
        "identity": {
            "panel_token_sha256": hashlib.sha256(manifest).hexdigest(),
            "hash_covers": "token_manifest",
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "panel_receipt_sha256": None,
            "shard_token_sha256": shard,
        },
        "contamination": {
            "checked": True,
            "method": "brandonmusic's %d-token-gram calibration-overlap scan at threshold "
                      "%.2f, reproduced independently by bin/joint-standard overlap-scan "
                      "against all %d non-final panel windows; every window's shared-gram "
                      "count and fraction matches his published window_selection.json "
                      "exactly. Document-level separation was ALREADY clean for all 25 "
                      "windows -- document_id_in_calibration is false everywhere -- so "
                      "every exclusion here is driven by the n-gram test alone."
                      % (sel["ngram_n"], sel["threshold"], sel["calibration_windows_scanned"]),
            "benchmarks_scanned": [],
            "hits": len(dropped),
            "receipt": {
                "kind": "receipt_file",
                "uri": "registry/protocol/window-selection.brandonmusic-final25.json",
                "sha256": _sha256_file(SELECTION_FILE),
                "note": "malaiwah.glm53-joint-standard-window-selection.v1, emitted by "
                        "bin/joint-standard overlap-scan and cross-checked window by "
                        "window against brandonmusic's published window_selection.json "
                        "(0 mismatches on all 25).",
            },
        },
        "sealed": True,
        "derived_from": PARENT_PANEL,
        "derivation": {
            "kind": "shard_subset",
            "detail": "The %d of 25 sealed windows whose %d-gram overlap with the "
                      "calibration-role windows is at or below %.2f. Dropped: %s. Six of "
                      "the eight are axis4_reasoning_termination at 37-39%% overlap, which "
                      "removes that domain entirely; the other two (final-0021 at 7.1%%, "
                      "final-0022 at 5.8%%) are legal and code-agentic, so this is NOT a "
                      "whole-domain drop and a 19-window axis4-only exclusion is a "
                      "DIFFERENT scope. The highest overlap among the retained windows is "
                      "%.2f%% (final-0014), so the %.0f%% threshold separates cleanly but "
                      "not by much -- it is inherited from brandonmusic and is an open "
                      "joint decision, not a derived constant."
                      % (len(meta), sel["ngram_n"], sel["threshold"], ", ".join(dropped),
                         100.0 * max(w["shared_ngram_fraction"] for w in meta),
                         100.0 * sel["threshold"]),
        },
        "availability": dict(parent["availability"]),
        "cross_refs": parent.get("cross_refs"),
        "sources": list(parent["sources"]),
        "disclosures": [
            {"code": "subset_of_panel", "severity": "caveat", "affects_comparability": True,
             "detail": "%d of the parent panel's 25 windows, %d of 51,175 scored positions. "
                       "Rows on this panel must never be tabled beside rows on the parent "
                       "panel: excluding the contaminated windows moves different "
                       "contributors' numbers in OPPOSITE directions (every malaiwah row "
                       "falls 12.6-16.2%%, brandonmusic's own 4bpw row rises 1.6%%)."
                       % (len(meta), positions)},
            {"code": "calibration_panel_overlap", "severity": "info",
             "affects_comparability": False,
             "detail": "This panel exists because the parent's contamination guard was "
                       "role separation only. The n-gram scan that produced it found one "
                       "whole domain sharing 37-39% of its 13-grams with calibration-role "
                       "windows despite clean document-level separation -- the finding is "
                       "brandonmusic's and the reproduction is ours."},
        ],
    }
    return [rec]


def references(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The same teacher capture, restricted to the calibration-clean windows."""
    parent = next(r for r in existing if r["id"] == PARENT_REFERENCE)
    meta = _clean_windows_meta()
    rec = json.loads(json.dumps(parent))
    rec["id"] = CLEAN_REFERENCE
    rec["name"] = ("brandonmusic BF16 fp32 teacher logits over the %d "
                   "calibration-clean windows" % len(meta))
    rec["panel_ref"] = CLEAN_PANEL
    rec["self_consistency"] = {
        "floor_measurement_ref":
            "measurement--glm53.bf16-replay-floor.brandonmusic-final25" + CLEAN_SUFFIX,
        "note": "The same cross-stack BF16 replay, read over the calibration-clean "
                "windows: 0.010648 instead of 0.012712. The floor moves with the scope, "
                "which is precisely why a floor must never be borrowed across scopes.",
    }
    rec["disclosures"] = [
        {"code": "subset_of_panel", "severity": "caveat", "affects_comparability": True,
         "detail": "The identical teacher capture as the 25-window reference, restricted "
                   "to the %d windows that survive the calibration-overlap scan. No new "
                   "capture was made and no teacher value changed." % len(meta)},
    ]
    return [rec]
