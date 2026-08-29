#!/usr/bin/env python3
"""Joint-standard invariants JSON Schema cannot express.

    python3 tools/registry_joint_check.py [--root .] [-v]

JSON Schema can say "se_total must be a number".  It cannot say "se_total must
equal hypot(se_clustered, sigma_run)", or "the per-domain positions must add up
to the panel positions", or "this row's value must be the equal-weight mean of
exactly the windows its scope names".  Those are the ones that catch a real
mistake, so they live here.  JOINT-001..008 in schema/invariants.json are the
published statements; this file is their implementation.

Exit 0 clean, 1 on any error.  Stdlib only; no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bin"))

REL = 1e-9


class Report:
    def __init__(self, verbose: bool = False) -> None:
        self.errors = []
        self.checks = 0
        self.verbose = verbose

    def check(self, cond: bool, code: str, msg: str) -> bool:
        self.checks += 1
        if not cond:
            self.errors.append("%-11s %s" % (code, msg))
        elif self.verbose:
            print("  ok  %-11s %s" % (code, msg))
        return cond


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _close(a, b, rel=REL):
    if a is None or b is None:
        return False
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) <= rel * scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rep = Report(args.verbose)

    data = os.path.join(args.root, "data")
    rows = _load_jsonl(os.path.join(data, "measurements.jsonl"))
    panels = {p["id"]: p for p in _load_jsonl(os.path.join(data, "panels.jsonl"))}
    proto_dir = os.path.join(args.root, "protocol")
    proto_path = os.path.join(proto_dir, "glm53-joint-kld-protocol.v1.json")
    sel_path = os.path.join(proto_dir, "window-selection.brandonmusic-final25.json")

    # ---------------------------------------------------------------- protocol
    have_proto = os.path.exists(proto_path)
    rep.check(have_proto, "JOINT-008",
              "the frozen protocol file exists at protocol/glm53-joint-kld-protocol.v1.json")
    proto_file_sha = proto_scoring_sha = None
    if have_proto:
        from jointstd import protocol as protomod

        p = protomod.load(proto_path)
        proto_file_sha, proto_scoring_sha = p.file_sha256, p.scoring_sha256
        rep.check(proto_file_sha == _sha256_file(proto_path), "JOINT-008",
                  "protocol file hash is a hash of the file")

    sel = None
    if os.path.exists(sel_path):
        sel = json.load(open(sel_path, encoding="utf-8"))
        sel_sha = _sha256_file(sel_path)
        rep.check(sel.get("cross_check", {}).get("identical") is True, "JOINT-005",
                  "the committed window selection records a clean cross-check against "
                  "brandonmusic's published window_selection.json")
        kept = set(sel["selected_windows"])
        worst_kept = max(w["shared_ngram_fraction"] for w in sel["per_window"]
                         if w["window_id"] in kept)
        rep.check(worst_kept <= sel["threshold"], "JOINT-005",
                  "no retained window exceeds the scan threshold (worst kept %.4f <= %.2f)"
                  % (worst_kept, sel["threshold"]))
        dropped = {e["window_id"] for e in sel["excluded_windows"]}
        best_dropped = min(w["shared_ngram_fraction"] for w in sel["per_window"]
                           if w["window_id"] in dropped)
        rep.check(best_dropped > sel["threshold"], "JOINT-005",
                  "every dropped window exceeds the threshold (best dropped %.4f > %.2f)"
                  % (best_dropped, sel["threshold"]))
    else:
        sel_sha = None

    # per-window inputs, so a row's value can be re-derived from its own scope
    pw_dir = os.path.join(proto_dir, "per-window")
    per_window = {}
    if os.path.isdir(pw_dir):
        for f in sorted(os.listdir(pw_dir)):
            if f.endswith(".json"):
                d = json.load(open(os.path.join(pw_dir, f), encoding="utf-8"))
                per_window[f[:-5]] = d["per_window"]

    try:
        from joint_enrich import SERIES, CLEAN_SUFFIX, CLEAN_PANEL
    except Exception as exc:                                  # pragma: no cover
        print("cannot import joint_enrich: %s" % exc, file=sys.stderr)
        return 1

    by_id = {r["id"]: r for r in rows}

    # ------------------------------------------------------------------- rows
    for row in rows:
        mid = row["id"]
        unc = row.get("uncertainty") or {}
        scope = row.get("measurement_scope") or {}
        det = row.get("determinism") or {}

        # JOINT-002: the quadrature arithmetic
        if unc.get("se_total") is not None:
            rep.check(_close(unc["se_total"],
                             math.hypot(unc.get("se_clustered") or 0.0,
                                        unc.get("sigma_run") or 0.0)),
                      "JOINT-002",
                      "%s: se_total == hypot(se_clustered, sigma_run)" % mid)

        # JOINT-003: sigma_run cannot claim more runs than were made
        if unc.get("sigma_run") is not None:
            rep.check((unc.get("sigma_run_runs") or 0) >= 2, "JOINT-003",
                      "%s: sigma_run carries at least 2 runs" % mid)
            rep.check((unc.get("sigma_run_runs") or 0) <= (det.get("run_count") or 0),
                      "JOINT-003",
                      "%s: sigma_run_runs (%s) <= determinism.run_count (%s)"
                      % (mid, unc.get("sigma_run_runs"), det.get("run_count")))
            if unc["sigma_run"] == 0.0:
                rep.check(det.get("identical_across_runs") is True, "JOINT-003",
                          "%s: sigma_run == 0 is only legal with bitwise-identical runs"
                          % mid)

        # JOINT-001: an interval must bracket the point estimate
        if unc.get("ci95_low") is not None and row["metric"].get("value") is not None:
            v = row["metric"]["value"]
            rep.check(unc["ci95_low"] <= v <= unc["ci95_high"], "JOINT-001",
                      "%s: the 95%% interval brackets the point estimate" % mid)
            rep.check((unc.get("clusters") or 0) >= 2, "JOINT-001",
                      "%s: a clustered interval needs at least 2 clusters" % mid)
        if unc.get("deff") is not None and unc.get("se_naive"):
            rep.check(_close(unc["deff"], (unc["se_clustered"] / unc["se_naive"]) ** 2, 1e-6),
                      "JOINT-001", "%s: deff == (se_clustered/se_naive)^2" % mid)

        # JOINT-006: the per-domain table must partition the scored positions
        bd = row.get("by_domain")
        if bd:
            tot = sum(d["scored_positions"] for d in bd)
            rep.check(tot == scope.get("scored_positions"), "JOINT-006",
                      "%s: per-domain positions sum to measurement_scope.scored_positions "
                      "(%d vs %s)" % (mid, tot, scope.get("scored_positions")))
            if scope.get("contexts") and all(d.get("windows") for d in bd):
                rep.check(sum(d["windows"] for d in bd) == scope["contexts"], "JOINT-006",
                          "%s: per-domain windows sum to contexts" % mid)
            rep.check(len({d["domain"] for d in bd}) == len(bd), "JOINT-006",
                      "%s: no duplicate domain in by_domain" % mid)
            for d in bd:
                if d.get("ci95_low") is not None:
                    rep.check(d["ci95_low"] <= d["mean"] <= d["ci95_high"], "JOINT-006",
                              "%s/%s: the domain interval brackets the domain mean"
                              % (mid, d["domain"]))
                    rep.check(bool(d.get("interval_kind")), "JOINT-006",
                              "%s/%s: an interval states its kind" % (mid, d["domain"]))

        # JOINT-008: the protocol stamp must resolve to the committed file
        pro = row.get("protocol")
        if pro:
            rep.check(pro.get("file_sha256") == proto_file_sha, "JOINT-008",
                      "%s: protocol.file_sha256 matches the committed protocol file" % mid)
            rep.check(pro.get("scoring_sha256") == proto_scoring_sha, "JOINT-008",
                      "%s: protocol.scoring_sha256 matches the recomputed scoring hash" % mid)

        # JOINT-004/005: the scope block
        if scope.get("scope_selection_sha256"):
            rep.check(scope["scope_selection_sha256"] == sel_sha, "JOINT-005",
                      "%s: scope_selection_sha256 matches the committed selection file" % mid)
        if scope.get("scope_name") and scope["scope_name"] != "panel25":
            rep.check(scope.get("covers_full_panel") is False, "JOINT-004",
                      "%s: a named sub-scope does not claim full panel coverage" % mid)
            rep.check(bool(scope.get("subset_detail")), "JOINT-004",
                      "%s: a named sub-scope carries a subset_detail" % mid)
            rep.check(any(d["code"] == "subset_of_panel" for d in row.get("disclosures", [])),
                      "JOINT-004", "%s: a named sub-scope discloses subset_of_panel" % mid)

        # JOINT-007: the masking policy must be stated
        rep.check("vocab_masking_policy" in (row.get("estimator") or {}), "JOINT-007",
                  "%s: estimator states a vocab_masking_policy" % mid)

    # --------------------------- the value must BE the mean of its own scope
    for mid, slug in SERIES.items():
        pw = per_window.get(slug)
        if pw is None:
            rep.check(False, "JOINT-006", "missing per-window input %s.json" % slug)
            continue
        for suffix, keep in ((None, None), (CLEAN_SUFFIX, set(sel["selected_windows"]) if sel else None)):
            rid = mid + (suffix or "")
            row = by_id.get(rid)
            if row is None:
                rep.check(False, "JOINT-006", "missing row %s" % rid)
                continue
            sub = [w for w in pw if keep is None or w["window_id"] in keep]
            n = sum(w["count"] for w in sub)
            mean = sum(w["count"] * w["mean"] for w in sub) / n
            rep.check(_close(mean, row["metric"]["value"], 1e-12), "JOINT-006",
                      "%s: value is the equal-weight mean of exactly its scope's windows "
                      "(%.15g)" % (rid, mean))
            rep.check(row["measurement_scope"]["scored_positions"] == n, "JOINT-006",
                      "%s: scored_positions == the scope's positions (%d)" % (rid, n))

    # ------------------- a scope must never share a comparability key with another
    keys = {}
    for row in rows:
        k = row["comparability"]["key"]
        keys.setdefault(k, set()).add((row["measurement_scope"].get("scope_name"),
                                       row["measurement_scope"].get("scored_positions")))
    for k, members in sorted(keys.items()):
        rep.check(len({m[0] for m in members if m[0]}) <= 1, "JOINT-004",
                  "comparability key %s holds exactly one scope_name" % k)

    # ------------------- the derived clean panel must be a real subset
    if CLEAN_PANEL in panels and sel:
        p = panels[CLEAN_PANEL]
        parent = panels.get(p.get("derived_from"))
        rep.check(parent is not None, "JOINT-004", "the clean panel names its parent")
        rep.check(p["structure"]["contexts"] == len(sel["selected_windows"]), "JOINT-004",
                  "the clean panel holds exactly the selected windows")
        rep.check(set(p["identity"]["shard_token_sha256"]) == set(sel["selected_windows"]),
                  "JOINT-004", "the clean panel's shard digests are exactly its windows")
        rep.check(p["contamination"]["checked"] is True, "JOINT-004",
                  "the clean panel records that contamination WAS checked")
        rep.check(sum(s["contexts"] for s in p["structure"]["strata"].values())
                  == p["structure"]["contexts"], "JOINT-004",
                  "the clean panel's strata sum to its context count")

    print()
    print("joint-standard checks: %d run, %d error(s)" % (rep.checks, len(rep.errors)))
    for e in rep.errors:
        print("  ERROR  %s" % e)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
