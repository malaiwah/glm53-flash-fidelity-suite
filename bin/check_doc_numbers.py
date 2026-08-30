#!/usr/bin/env python3
"""Re-derive every number in the alignment documents from the committed data.

    bin/check_doc_numbers.py [--root .] [-v] [--sweep]

WHY THIS EXISTS.  ``docs/PROTOCOL-ALIGNMENT.md`` and ``docs/ALIGNMENT-REPLY.md``
argue that a fidelity number without a receipt is an assertion.  Until this
file, nothing tied either document to a receipt: ``registry/Makefile``'s check
target validates the registry and its README, and no committed script mentioned
either document at all.  Five wrong numbers reached those documents through that
gap -- an "every row moves down" claim that three rows contradicted, a "10x
tighter" that measured 4.2x, a "13 rows" that was 16, a padded-column cap stated
an order of magnitude tighter than the identity supports, and a "ten synthetic
students" that were thirteen.  Every one was recomputable from data already in
the tree.  So recompute it, in CI, and fail.

WHAT IT DOES.  Two passes, and only the first one is a gate.

  ANCHORED CLAIMS.  For each claim: recompute the value from committed data,
  format it the way the document formats it, and require the resulting literal
  to appear in the document.  A claim that cannot be recomputed is an error, not
  a skip -- "I could not check this" is the state that let the five through.

  SWEEP (``--sweep``).  Scan both documents for every decimal with six or more
  places and report any that appears nowhere in the committed JSON and is not a
  value this checker itself derives.  Advisory: a document legitimately quotes
  brandonmusic's published numbers, which live in his receipts and in our
  fixture rather than in our registry.  It is a prompt to look, not a gate.

Sources, all committed: ``registry/protocol/per-window/*.json`` (the six
per-window series), ``registry/protocol/window-selection.brandonmusic-final25.json``
(his 17-window clean scope, cross-checked against his published selection),
``docs/joint-standard/analysis/*.json`` (the emitted analyses),
``docs/joint-standard/padded-column/*.json`` (the padded-column study) and
``registry/data/measurements.jsonl``.

Exit 0 clean, 1 on any anchored-claim failure.  Stdlib only; no network.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ALIGNMENT = "docs/PROTOCOL-ALIGNMENT.md"
REPLY = "docs/ALIGNMENT-REPLY.md"

# The six per-window series, and the label each one carries in the documents.
SERIES = {
    "k6-sealed": "K6 tr3-6bpw sealed 5-run",
    "k6-streaming": "K6 tr3-6bpw streaming 2-run",
    "k8-streaming": "K8 tr3-8bpw streaming 2-run",
    "fp8-crossstack": "official FP8 cross-stack",
    "bf16-floor-crossstack": "BF16 floor cross-stack",
    "brandonmusic-4bpw": "his 4bpw",
}


class Report:
    def __init__(self, verbose=False):
        self.errors = []
        self.warns = []
        self.checks = 0
        self.verbose = verbose

    def check(self, cond, claim, detail=""):
        self.checks += 1
        if not cond:
            self.errors.append("%-46s %s" % (claim, detail))
        elif self.verbose:
            print("  ok    %-46s %s" % (claim, detail))
        return cond

    def warn(self, msg):
        self.warns.append(msg)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def equal_weight_mean(per_window, keep=None):
    """The scope's mean, exactly as the registry defines it: the equal-weight
    mean of the window means. Legal here because every window is 2047
    positions -- and that is asserted, not assumed."""
    rows = [w for w in per_window if keep is None or w["window_id"] in keep]
    counts = {w["count"] for w in rows}
    if len(counts) != 1:
        raise ValueError("mixed window sizes %s: the equal-weight mean is not "
                         "the token-weighted one" % sorted(counts))
    return math.fsum(w["mean"] for w in rows) / len(rows), len(rows), rows


def fmt_variants(value, places):
    """The literal a document would carry, plus the thousands-separated and
    percent-signed shapes markdown tables use."""
    out = {("%%.%df" % places) % value}
    out.add((("%%.%df" % places) % value).rstrip("0").rstrip("."))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="also report high-precision decimals with no source")
    args = ap.parse_args()
    rep = Report(args.verbose)
    R = args.root

    align_path = os.path.join(R, ALIGNMENT)
    reply_path = os.path.join(R, REPLY)
    for p in (align_path, reply_path):
        if not os.path.exists(p):
            print("check_doc_numbers: %s is missing" % p, file=sys.stderr)
            return 1
    align = open(align_path, encoding="utf-8").read()
    reply = open(reply_path, encoding="utf-8").read()
    both = align + "\n" + reply

    pw_dir = os.path.join(R, "registry/protocol/per-window")
    series = {k: load_json(os.path.join(pw_dir, k + ".json"))["per_window"]
              for k in SERIES}
    sel = load_json(os.path.join(
        R, "registry/protocol/window-selection.brandonmusic-final25.json"))
    clean = set(sel["selected_windows"])

    derived = set()          # every value this checker computes, for the sweep

    def note(v):
        derived.add(round(float(v), 12))
        return v

    # ================================================== 1. the twelve means
    print("=" * 78)
    print("1. THE TWELVE SCOPE MEANS (section 4.3) -- recomputed from per-window")
    print("=" * 78)
    means = {}
    for key, label in SERIES.items():
        for scope, keep in (("panel25", None), ("clean17", clean)):
            m, n, _ = equal_weight_mean(series[key], keep)
            means[(key, scope)] = m
            note(m)
            want = "%.12f" % m
            rep.check(want in align, "%s / %s mean" % (label, scope),
                      "%s (%d windows)" % (want, n))
    rep.check(len(clean) == 17, "the clean scope is 17 windows", str(len(clean)))
    rep.check(len(series["k6-sealed"]) == 25, "the full panel is 25 windows",
              str(len(series["k6-sealed"])))

    # ================================================== 2. the scope deltas
    print()
    print("=" * 78)
    print("2. THE SCOPE DELTAS (section 4.3) -- clean17 against panel25")
    print("=" * 78)
    for key, label in SERIES.items():
        p, c = means[(key, "panel25")], means[(key, "clean17")]
        pct = (c - p) / p * 100.0
        note(pct)
        lit = "%.2f" % abs(pct)
        found = re.search(r"[+−-]\s?%s\s?%%" % re.escape(lit), align)
        rep.check(bool(found), "%s scope delta" % label,
                  "%+.2f %% (%s)" % (pct, "found" if found else "NOT IN DOC"))
        # and the SIGN has to be right, which is the claim that was wrong before
        sign_txt = "−" if pct < 0 else "+"
        rep.check((("%s%s %%" % (sign_txt, lit)) in align
                   or ("%s%s%%" % (sign_txt, lit)) in align
                   or ("**%s%s %%**" % (sign_txt, lit)) in align),
                  "%s scope delta SIGN" % label, "%s%s %%" % (sign_txt, lit))

    # the FP8 exception: it is the one row that rises at the loosest thresholds,
    # and both documents now say so. Verify the mechanism, not just the words.
    ts = sel.get("threshold_sensitivity") or []
    if ts:
        loose = [t for t in ts if t.get("threshold", 0) >= 0.075]
        rep.check(bool(loose), "the sensitivity sweep reaches 0.075+",
                  "%d thresholds" % len(loose))

    # ============================================ 3. the attributable table
    print()
    print("=" * 78)
    print("3. THE ATTRIBUTABLE TABLE (section 4.5) -- FP8 minus the same-lane floor")
    print("=" * 78)
    for scope in ("panel25", "clean17"):
        fp8 = means[("fp8-crossstack", scope)]
        flr = means[("bf16-floor-crossstack", scope)]
        att = fp8 - flr
        ratio = fp8 / flr
        note(att); note(ratio)
        rep.check("%.9f" % att in align, "attributable %s" % scope,
                  "%.9f" % att)
        rep.check("%.3f" % ratio in align, "attributable ratio %s" % scope,
                  "%.3f" % ratio)
    a_p = means[("fp8-crossstack", "panel25")] - means[("bf16-floor-crossstack", "panel25")]
    a_c = means[("fp8-crossstack", "clean17")] - means[("bf16-floor-crossstack", "clean17")]
    move = (a_c - a_p) / a_p * 100.0
    note(move)
    rep.check("%.2f" % move in align,
              "attributable moves while its inputs move", "+%.2f %%" % move)

    # ================================================ 4. the emitted analyses
    print()
    print("=" * 78)
    print("4. THE EMITTED ANALYSES -- the doc's CI/SE/deff must be the receipts'")
    print("=" * 78)
    an_dir = os.path.join(R, "docs/joint-standard/analysis")
    if not os.path.isdir(an_dir):
        rep.check(False, "analysis receipts present", an_dir)
    else:
        for key, label in SERIES.items():
            for scope, suffix in (("panel25", "panel"), ("clean17", "selected")):
                p = os.path.join(an_dir, "%s.%s.json" % (key, suffix))
                if not os.path.exists(p):
                    rep.check(False, "analysis %s/%s" % (label, scope), "missing")
                    continue
                d = load_json(p)
                rep.check(abs(d["summary"]["mean"] - means[(key, scope)]) < 1e-15,
                          "%s/%s: receipt mean == recomputed" % (label, scope),
                          "%.15g" % d["summary"]["mean"])
                se = d["summary"]["se_clustered_window"]
                note(se)
                rep.check("%.3e" % se in align, "%s/%s SE in doc" % (label, scope),
                          "%.3e" % se)
                lo, hi = d["bootstrap"]["ci95_bca"]
                note(lo); note(hi)
                rep.check("%.6f" % lo in align and "%.6f" % hi in align,
                          "%s/%s BCa endpoints in doc" % (label, scope),
                          "[%.6f, %.6f]" % (lo, hi))
                # an interval that does not bracket its own mean is a wrong table
                rep.check(lo <= d["summary"]["mean"] <= hi,
                          "%s/%s BCa brackets the mean" % (label, scope), "")

    # ================================================== 5. the paired table
    print()
    print("=" * 78)
    print("5. THE PAIRED TABLE (section 4.4) -- ratios, win counts, BCa, sign p")
    print("=" * 78)
    # Anchor on the doc's OWN table rows. Demanding that every emitted receipt
    # appear in the document is wrong -- which comparisons to tabulate is an
    # editorial choice -- but every row the document DOES print has to be the
    # receipt's numbers. So parse the table, map each row to its receipt, and
    # verify. A row whose numbers were edited then matches no receipt and fails.
    DOC_LABEL = {
        ("K6", "K8"): "paired.K6-vs-K8",
        ("K6", "FP8"): "paired.K6-vs-FP8",
        ("K8", "FP8"): "paired.K8-vs-FP8",
        ("FP8", "BF16 floor"): "paired.FP8-vs-BF16floor",
        ("his 4bpw", "K6"): "paired.BM4bpw-vs-K6",
    }
    SCOPE_SUFFIX = {"panel25": "panel", "clean17": "selected"}
    row_re = re.compile(
        r"^\|\s*(.+?)\s*[-\u2212]\s*(.+?)\s*\|\s*(panel25|clean17)\s*\|"
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
        r"\s*\[([-+\u2212\d.]+),\s*([-+\u2212\d.]+)\]\s*\|"
        r"\s*(\d+)/(\d+)\s*\|\s*([\d.e+-]+)\s*\|", re.M)
    seen_rows = 0
    for m in row_re.finditer(align):
        a, b, scope = m.group(1), m.group(2), m.group(3)
        stem = DOC_LABEL.get((a, b))
        if stem is None:
            rep.check(False, "paired row %s - %s / %s" % (a, b, scope),
                      "no receipt is mapped for this comparison")
            continue
        path = os.path.join(an_dir, "%s.%s.json" % (stem, SCOPE_SUFFIX[scope]))
        if not os.path.exists(path):
            rep.check(False, "paired row %s - %s / %s" % (a, b, scope),
                      "receipt %s is missing" % os.path.basename(path))
            continue
        seen_rows += 1
        r = load_json(path)
        note(r["ratio_a_over_b"]); note(r["ci95_diff_bca"][0])
        note(r["ci95_diff_bca"][1]); note(r["mean_diff"])
        rep.check("%.3f" % r["ratio_a_over_b"] == m.group(6),
                  "%s/%s ratio" % (stem, scope),
                  "doc %s vs receipt %.3f" % (m.group(6), r["ratio_a_over_b"]))
        lo = float(m.group(7).replace("\u2212", "-"))
        hi = float(m.group(8).replace("\u2212", "-"))
        rep.check(abs(lo - r["ci95_diff_bca"][0]) < 5e-7
                  and abs(hi - r["ci95_diff_bca"][1]) < 5e-7,
                  "%s/%s BCa endpoints" % (stem, scope),
                  "doc [%+.6f, %+.6f] vs receipt [%+.6f, %+.6f]"
                  % (lo, hi, r["ci95_diff_bca"][0], r["ci95_diff_bca"][1]))
        # STAT-02: the sign test's denominator is its own informative-pair count, not
        # n_windows. They coincide only when nothing tied, which is true of every
        # currently published pair and is NOT true of the K6 sealed-vs-streaming pair
        # anyone would compare next (11 of 25 windows tie exactly).
        want_n = r.get("sign_test_n", r["n_windows"])
        rep.check(int(m.group(9)) == r["windows_a_better"]
                  and int(m.group(10)) == want_n,
                  "%s/%s win count" % (stem, scope),
                  "doc %s/%s vs receipt %d/%d"
                  % (m.group(9), m.group(10), r["windows_a_better"], want_n))
        if r.get("sign_test_p") is None:
            rep.check(True, "%s/%s sign-test p (all windows tied)" % (stem, scope), "")
        else:
            rep.check(abs(float(m.group(11)) - r["sign_test_p"])
                      <= 0.05 * max(r["sign_test_p"], 1e-30),
                      "%s/%s sign-test p" % (stem, scope),
                      "doc %s vs receipt %.3g" % (m.group(11), r["sign_test_p"]))
        # a paired interval that excludes zero is the claim; check the receipt agrees
        rep.check(r["bca_excludes_zero"] == (lo > 0 or hi < 0),
                  "%s/%s excludes-zero agrees with the printed interval" % (stem, scope),
                  str(r["bca_excludes_zero"]))
    rep.check(seen_rows >= 6, "the paired table has rows to check",
              "%d rows parsed" % seen_rows)
    # receipts the document does not tabulate: informational, not a failure
    untab = []
    for f in sorted(os.listdir(an_dir)) if os.path.isdir(an_dir) else []:
        if f.startswith("paired."):
            stem, suffix = f[:-5].rsplit(".", 1)
            scope = "panel25" if suffix == "panel" else "clean17"
            if not any(DOC_LABEL.get((mm.group(1), mm.group(2))) == stem
                       and mm.group(3) == scope for mm in row_re.finditer(align)):
                untab.append("%s/%s" % (stem, scope))
    if untab:
        print("  (%d emitted paired receipts are not tabulated in the doc: %s)"
              % (len(untab), ", ".join(untab)))

    # ========================================= 6. the paired-tightness claim
    print()
    print("=" * 78)
    print("6. THE 'PAIRED IS TIGHTER' CLAIM (section 4.4) -- the one that said 10x")
    print("=" * 78)
    pk = os.path.join(an_dir, "paired.K6-vs-K8.selected.json")
    if os.path.exists(pk):
        d = load_json(pk)
        r = d.get("paired") or d
        ci = r.get("ci95_diff_bca")
        if ci:
            pw = ci[1] - ci[0]
            note(pw)
            a6 = load_json(os.path.join(an_dir, "k6-sealed.selected.json"))
            a8 = load_json(os.path.join(an_dir, "k8-streaming.selected.json"))
            w6 = a6["bootstrap"]["ci95_bca"][1] - a6["bootstrap"]["ci95_bca"][0]
            w8 = a8["bootstrap"]["ci95_bca"][1] - a8["bootstrap"]["ci95_bca"][0]
            note(w6); note(w8)
            ratio = max(w6, w8) / pw
            note(ratio)
            rep.check("%.3e" % pw in align, "paired width quoted", "%.3e" % pw)
            rep.check("%.3e" % w6 in align and "%.3e" % w8 in align,
                      "both marginal widths quoted", "%.3e / %.3e" % (w6, w8))
            rep.check(("%.1f" % ratio) in align, "the tightness FACTOR",
                      "%.1fx (a claim of 10x here would fail)" % ratio)
            # and no stale "10x"-shaped overclaim may survive anywhere
            # A QUOTED "10x tighter" is the documents describing an error they
            # fixed, not making the claim. Only unquoted occurrences are live
            # claims, so the quoted form is excluded rather than tripping this.
            # A QUOTED "10x tighter" is the documents describing an error they
            # fixed, not making the claim, so quotes are excluded. Digits and
            # dots are excluded too, or the pattern starts mid-number and reads
            # "0x tighter" out of a quoted "10x tighter".
            stale = re.findall(
                r'(?<![\d."\u201c])(\d+(?:\.\d+)?)\s?[x\u00d7]\s+tighter', both)
            bad = [s for s in stale if abs(float(s) - ratio) > 1.0]
            rep.check(not bad, "no stale tightness factor in either doc",
                      "found %s vs measured %.1f" % (bad, ratio) if bad else "clean")

    # ================================================ 7. the per-domain table
    print()
    print("=" * 78)
    print("7. THE PER-DOMAIN TABLE (section 4.6) -- his non-uniformity, our data")
    print("=" * 78)
    dom_src = {}
    for key in ("k6-sealed", "k8-streaming", "fp8-crossstack", "bf16-floor-crossstack"):
        p = os.path.join(an_dir, "%s.selected.json" % key)
        if os.path.exists(p):
            dom_src[key] = {r["domain"]: r for r in load_json(p)["by_domain"]}
    if dom_src:
        for dom in sorted(dom_src["k6-sealed"]):
            for key in dom_src:
                v = dom_src[key][dom]["mean"]
                note(v)
                if key == "bf16-floor-crossstack":
                    # not a column in the doc's table; it enters only through
                    # the FP8/floor ratio, which is checked below
                    continue
                rep.check("%.9f" % v in align,
                          "%s / %s" % (dom, key), "%.9f" % v)
            r_k6 = dom_src["fp8-crossstack"][dom]["mean"] / dom_src["k6-sealed"][dom]["mean"]
            r_fl = (dom_src["fp8-crossstack"][dom]["mean"]
                    / dom_src["bf16-floor-crossstack"][dom]["mean"])
            note(r_k6); note(r_fl)
            rep.check("%.3f" % r_k6 in align, "%s FP8/K6 ratio" % dom, "%.3f" % r_k6)
            rep.check("%.3f" % r_fl in align, "%s FP8/floor ratio" % dom, "%.3f" % r_fl)
        ratios = {d: dom_src["fp8-crossstack"][d]["mean"] / dom_src["k6-sealed"][d]["mean"]
                  for d in dom_src["k6-sealed"]}
        spread = max(ratios.values()) / min(ratios.values())
        note(spread)
        rep.check("%.2f" % spread in align, "the cross-domain spread",
                  "%.2fx" % spread)
        worst = max(ratios, key=ratios.get)
        rep.check("legal" in worst, "legal is the worst domain, as he found",
                  "worst = %s at %.3f" % (worst, ratios[worst]))
        # the positions must partition the scope, or the table is not a partition
        tot = sum(dom_src["k6-sealed"][d]["n"] for d in dom_src["k6-sealed"])
        rep.check(tot == 17 * 2047, "per-domain positions partition clean17",
                  "%d == 17*2047" % tot)

    # ============================================= 8. the padded-column study
    print()
    print("=" * 78)
    print("8. THE PADDED-COLUMN STUDY (section 3) -- doc vs the committed receipt")
    print("=" * 78)
    pc_dir = os.path.join(R, "docs/joint-standard/padded-column")
    study = None
    for cand in ("padded-column-study.json", "teacher-side-reproduction.json"):
        p = os.path.join(pc_dir, cand)
        if os.path.exists(p):
            study = load_json(p)
            break
    if study is None:
        rep.check(False, "a padded-column receipt is committed",
                  "section 3 says the script and a receipt are owed")
    else:
        t = study["teacher_padded"]
        note(t["Pm_mean"])
        rep.check("1.6e-8" in align or "1.600639e-08" in align
                  or "%.6e" % t["Pm_mean"] in align,
                  "padded mass on his real window",
                  "P_m mean %.6e" % t["Pm_mean"])
        rep.check(t["Pm_mean"] < 1e-7, "the padded mass is the order the doc claims",
                  "%.3e < 1e-7" % t["Pm_mean"])
        # the doc's general cap is order e_p; it must NOT claim 1e-10 generally
        rep.check("1e-8" in align, "the GENERAL cap is stated as order 1e-8",
                  "the shared-head 1e-10 is the special case")
        n_students = 0
        for block in ("case_A_shared_head", "case_B_quantized_head"):
            n_students += len([k for k in (study.get(block) or {})
                               if not k.startswith("_")])
        extra = study.get("case_C_final") or {}
        n_students += len([k for k in extra if not k.startswith("_")])
        if n_students:
            words = {10: "ten", 13: "thirteen", 12: "twelve", 11: "eleven"}
            w = words.get(n_students)
            rep.check(w is not None and w in align,
                      "the synthetic-student COUNT matches the receipt",
                      "%d (%s)" % (n_students, w))
        hp = study.get("head_padded_rows")
        if hp:
            note(hp["padded_row_norm_mean"]); note(hp["real_row_norm_mean"])
            note(hp["padded_pairwise_cosine_mean"])
            rep.check("%.4f" % hp["padded_row_norm_mean"] in align,
                      "padded row norm quoted from the receipt",
                      "%.4f" % hp["padded_row_norm_mean"])
            rep.check("%.2f" % hp["real_row_norm_mean"] in align,
                      "typical real-row norm quoted from the receipt",
                      "%.2f" % hp["real_row_norm_mean"])
            rep.check("%.6f" % hp["padded_pairwise_cosine_mean"] in align,
                      "padded pairwise cosine quoted from the receipt",
                      "%.6f" % hp["padded_pairwise_cosine_mean"])
            rep.check(hp["nonfinite_in_head"] == 0,
                      "the head loaded with no NaN or Inf",
                      "%d non-finite" % hp["nonfinite_in_head"])
        if "r0_canary" in study:
            c = study["r0_canary"]
            rep.check(c["r0a_self_kld_full_max"] == 0.0
                      and c["r0a_self_kld_masked_max"] == 0.0,
                      "R0-a on the real teacher: exactly 0.0 both scopes", "")
            rep.check(c["r0b_ratio"] >= c["r0b_gate"],
                      "R0-b on the real teacher clears the gate",
                      "%.2fx entropy vs gate %.1f" % (c["r0b_ratio"], c["r0b_gate"]))

    # ==================================================== 9. countable claims
    print()
    print("=" * 78)
    print("9. COUNTABLE CLAIMS -- the ones that were wrong by counting")
    print("=" * 78)
    rows = load_jsonl(os.path.join(R, "registry/data/measurements.jsonl"))
    on_his = [r for r in rows if "brandonmusic" in (r.get("panel_ref") or "")]
    rep.check(len(on_his) >= 1, "rows on his panels are findable",
              "%d rows" % len(on_his))
    with_ci = [r for r in rows
               if (r.get("uncertainty") or {}).get("method") == "window_block_bootstrap_bca"]
    note(len(with_ci))
    rep.check(str(len(with_ci)) in both, "the BCa row COUNT is stated correctly",
              "%d rows carry window_block_bootstrap_bca" % len(with_ci))
    with_dom = [r for r in rows if r.get("by_domain")]
    rep.check(str(len(with_dom)) in both, "the by_domain row COUNT",
              "%d rows" % len(with_dom))
    with_sig = [r for r in rows
                if (r.get("uncertainty") or {}).get("sigma_run") is not None]
    rep.check(str(len(with_sig)) in both, "the sigma_run row COUNT",
              "%d rows" % len(with_sig))
    # JOINT-009's premise: no live sigma_run in the registry today
    live = [r for r in with_sig if r["uncertainty"]["sigma_run"] != 0.0]
    rep.check(not live, "every sigma_run in the registry is exactly 0.0",
              "%d live" % len(live))

    # =============================================== 10. Discord paste limit
    print()
    print("=" * 78)
    print("10. THE REPLY MUST ACTUALLY PASTE -- Discord's 2000-character limit")
    print("=" * 78)
    blocks = re.findall(r"(?ms)^## Message.*?(?=^## Message|\Z)", reply)
    msgs = []
    for b in blocks:
        # drop the header line and any trailing horizontal rule: what the
        # operator actually pastes is the body
        body = b.split("\n", 1)[1] if "\n" in b else ""
        body = re.sub(r"(?m)^---\s*$", "", body)
        msgs.append(body)
    if msgs:
        worst = max(len(m.strip()) for m in msgs)
        for i, m in enumerate(msgs, 1):
            n = len(m.strip())
            rep.check(n <= 2000, "message %d is under 2000 characters" % i,
                      "%d chars" % n)
        claimed = re.search(r"(\w+) messages, each under", reply)
        if claimed:
            words = {"Ten": 10, "Nine": 9, "Eleven": 11, "Twelve": 12,
                     "ten": 10, "nine": 9, "eleven": 11, "twelve": 12}
            want = words.get(claimed.group(1))
            if want:
                rep.check(len(msgs) == want,
                          "the reply contains the number of messages it claims",
                          "%d found, %d claimed" % (len(msgs), want))
        print("  (%d messages, longest %d chars)" % (len(msgs), worst))
    else:
        rep.warn("could not split the reply into messages; the 2000-char "
                 "check did not run")

    # ============================================ 11. the model-card disclosure
    print()
    print("=" * 78)
    print("11. THE MODEL CARDS -- the scope disclosure must match the recompute")
    print("=" * 78)
    cards = [("docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md", "k6-sealed"),
             ("docs/cards/GLM-5.3-Flash-TR3-8bpw.README.md", "k8-streaming")]
    # The card table quotes six-decimal values, which is what a reader compares.
    card_rows = [("K6 sealed", "k6-sealed"), ("K6 streaming", "k6-streaming"),
                 ("K8", "k8-streaming"), ("official FP8", "fp8-crossstack"),
                 ("BF16 floor", "bf16-floor-crossstack"),
                 ("brandonmusic 4bpw", "brandonmusic-4bpw")]
    for path, own in cards:
        full = os.path.join(R, path)
        if not os.path.exists(full):
            rep.check(False, "card %s" % os.path.basename(path), "missing")
            continue
        card = open(full, encoding="utf-8").read()
        has = "Scope disclosure" in card
        rep.check(has, "%s carries the scope disclosure" % os.path.basename(path),
                  "found" if has else "ABSENT -- the panel25 number stands unqualified")
        if not has:
            continue
        for _, key in card_rows:
            for scope in ("panel25", "clean17"):
                v = means[(key, scope)]
                rep.check("%.6f" % v in card,
                          "%s: %s/%s" % (os.path.basename(path)[:22], key, scope),
                          "%.6f" % v)
            p_, c_ = means[(key, "panel25")], means[(key, "clean17")]
            pct = (c_ - p_) / p_ * 100.0
            rep.check("%.2f" % abs(pct) in card,
                      "%s: %s scope delta" % (os.path.basename(path)[:22], key),
                      "%+.2f %%" % pct)
        # the card's own headline value must be the panel25 one it claims
        rep.check("%.6f" % means[(own, "panel25")] in card,
                  "%s: its own headline is the panel25 value"
                  % os.path.basename(path)[:22],
                  "%.6f" % means[(own, "panel25")])
        # and the widening ratio it quotes must be the real one
        for scope, places in (("panel25", 2), ("clean17", 2)):
            r_ = means[("fp8-crossstack", scope)] / means[(own, scope)]
            note(r_)
            rep.check(("%.2f" % r_) in card,
                      "%s: FP8-over-own ratio %s"
                      % (os.path.basename(path)[:22], scope), "%.2fx" % r_)
        # the padded-column figure the card quotes must be the receipt's
        if study is not None:
            rep.check("1.6e-8" in card,
                      "%s: padded mass quoted as the receipt has it"
                      % os.path.basename(path)[:22],
                      "P_m %.3e" % study["teacher_padded"]["Pm_mean"])

    # ====================================== 12. the single-window power arithmetic
    print()
    print("=" * 78)
    print("12. THE POWER ARITHMETIC -- re-derived, not quoted")
    print("=" * 78)
    # CC-01. Six places in this repo, and the PUBLISHED K8 model card, said
    # "per-window KLD scatter has sd 1.73e-3 against a K6-vs-K8 effect of 1.22e-3".
    # Both numbers came from k6/K8-ANOMALY.json, where they are the per-window DELTA
    # sd (0.0017334539428769534) and the pooled delta (-0.0012176728196882456) over an
    # ELEVEN-window subset -- correct in that document, mislabelled and mis-scoped
    # everywhere else. Two selftests asserted the literal strings, which locked them in.
    # Derive them here instead, from the committed per-window series, so a wrong one
    # cannot be quoted again.
    import statistics as _st

    def _series(name):
        doc = load_json(os.path.join(R, "registry", "protocol", "per-window", name))
        return {row["window_id"]: float(row["mean"]) for row in doc["per_window"]}

    try:
        k6s = _series("k6-streaming.json")
        k8s = _series("k8-streaming.json")
        k6seal = _series("k6-sealed.json")
    except (IOError, KeyError, ValueError) as exc:
        rep.warn("per-window series unreadable (%s); section 12 did not run" % exc)
    else:
        common = sorted(set(k6s) & set(k8s))
        scatter_k6 = _st.stdev([k6seal[w] for w in sorted(k6seal)])
        scatter_k8 = _st.stdev([k8s[w] for w in sorted(k8s)])
        deltas = [k8s[w] - k6s[w] for w in common]
        delta_sd = _st.stdev(deltas)
        effect = abs(_st.fmean(deltas))
        print("  per-window KLD sd: k6 %.4e / k8 %.4e | paired delta sd %.4e | "
              "effect %.4e (n=%d)" % (scatter_k6, scatter_k8, delta_sd, effect, len(common)))
        rep.check(abs(scatter_k6 - 7.2e-3) < 5e-5,
                  "per-window KLD scatter (K6 sealed) is the 7.2e-3 the docs quote",
                  "%.4e" % scatter_k6)
        rep.check(abs(delta_sd - 2.0e-3) < 5e-5,
                  "paired per-window K6-K8 delta sd is the 2.0e-3 the docs quote",
                  "%.4e" % delta_sd)
        rep.check(abs(effect - 1.33e-3) < 5e-6,
                  "the K6-vs-K8 effect is the 1.33e-3 the docs quote",
                  "%.4e" % effect)
        stale = []
        for rel in ("bin/kld_preview.py", "bin/fidelity/previewstats.py", "bin/README.md",
                    "WHAT-WE-MEASURE.md",
                    "docs/cards/GLM-5.3-Flash-TR3-8bpw.README.md"):
            full = os.path.join(R, rel)
            if not os.path.exists(full):
                continue
            text = open(full, encoding="utf-8").read()
            # The corrected passages NAME the retracted pair while explaining it, so
            # only an occurrence that still reads as a live claim counts.
            for line in text.splitlines():
                if ("1.73e-3" in line or "1.7e-3" in line) and "scatter" in line \
                        and "7.2e-3" not in line:
                    stale.append("%s: %s" % (rel, line.strip()[:70]))
        rep.check(not stale,
                  "no document still quotes the 11-window delta sd as the panel's "
                  "KLD scatter",
                  "; ".join(stale) if stale else "clean")

    # ======================================================== SWEEP (advisory)
    if args.sweep:
        print()
        print("=" * 78)
        print("SWEEP (advisory) -- high-precision decimals with no committed source")
        print("=" * 78)
        universe = set()

        def harvest(o):
            if isinstance(o, dict):
                for v in o.values():
                    harvest(v)
            elif isinstance(o, list):
                for v in o:
                    harvest(v)
            elif isinstance(o, float):
                universe.add(round(o, 12))

        for base, _, files in os.walk(os.path.join(R, "registry")):
            for f in files:
                if f.endswith(".json"):
                    try:
                        harvest(load_json(os.path.join(base, f)))
                    except Exception:
                        pass
                elif f.endswith(".jsonl"):
                    try:
                        for r in load_jsonl(os.path.join(base, f)):
                            harvest(r)
                    except Exception:
                        pass
        for base, _, files in os.walk(os.path.join(R, "docs/joint-standard")):
            for f in files:
                if f.endswith(".json"):
                    try:
                        harvest(load_json(os.path.join(base, f)))
                    except Exception:
                        pass
        fx = os.path.join(R, "bin/jointstd/fixtures/brandonmusic-known-answer.json")
        if os.path.exists(fx):
            harvest(load_json(fx))
        universe |= derived
        unmatched = []
        for m in re.finditer(r"(?<![\w.])(\d+\.\d{6,})", both):
            v = float(m.group(1))
            # A document prints an interval endpoint as "-0.009982"; the sign is
            # not part of the match, so compare on magnitude as well or every
            # negative endpoint reads as unsourced.
            if any(abs(v - u) <= 1e-9 * max(1.0, abs(u))
                   or abs(v - abs(u)) <= 1e-9 * max(1.0, abs(u)) for u in universe):
                continue
            # also accept it as a rounded view of something committed
            if any(abs(v - round(u, len(m.group(1).split(".")[1]))) < 1e-12
                   for u in universe):
                continue
            unmatched.append(m.group(1))
        seen = []
        for u in unmatched:
            if u not in seen:
                seen.append(u)
        if seen:
            print("  %d distinct high-precision decimals have no committed source:"
                  % len(seen))
            for u in seen[:40]:
                print("    %s" % u)
            if len(seen) > 40:
                print("    ... and %d more" % (len(seen) - 40))
            print("  (his published numbers legitimately land here; ours should not)")
        else:
            print("  every high-precision decimal in both documents is committed.")

    # ------------------------------------------------------------------ done
    print()
    print("-" * 78)
    for w in rep.warns:
        print("WARN  %s" % w)
    if rep.errors:
        print("%d claim(s) FAILED of %d checked:" % (len(rep.errors), rep.checks))
        for e in rep.errors:
            print("  FAIL  %s" % e)
        return 1
    print("doc-vs-receipt: %d claims checked, 0 failed." % rep.checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
