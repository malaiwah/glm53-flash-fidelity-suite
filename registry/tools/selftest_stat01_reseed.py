#!/usr/bin/env python3
"""The STAT-01 / STAT-17 reseed, tested against the real 42 cells.

The validator selftest (`registry_selftest.py`) proves the registry REFUSES a
by_domain block that shares a seed, omits its measured coverage or publishes a
negative lower bound.  This proves the arithmetic underneath it: that the new
interval is what it says, that the OLD numbers are still regenerable so the
change is attributable rather than merely different, and that the coverage
figures the rows publish are the ones the committed simulation produced.

Every case here fails against the pre-2026-08-30 code.  T1/T2 fail because
`derive_seed` does not exist and every domain drew the same stream; T3 fails
because `delta_t_log` does not exist; T4 fails because `domain_table` had no
`interval` parameter to ask for the old procedure by name; T6/T7 fail because
`coverage_measured` was never computed for anything.  T8 fails against the
pre-2026-09-05 code because `joint_enrich` read only the Flash per-window files
and left every GLM-5.3 row at `uncertainty.method: none`.

Needs numpy, for the same reason `make reseed` does: the resample stream must be
PCG64.  Run with `make stat-selftest`.
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REGISTRY = os.path.dirname(_HERE)
_REPO = os.path.dirname(_REGISTRY)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "bin"))

from jointstd import stats as S      # noqa: E402
import coverage_sim as CS            # noqa: E402
import joint_enrich as JE            # noqa: E402

DOMAINS = ("axis1_general", "axis2_legal", "axis3_code_agentic",
           "axis4_reasoning_termination")

_passed = []
_failed = []


def check(ok, name, detail=""):
    (_passed if ok else _failed).append(name)
    print("  %-64s %s" % (name, "PASS" if ok else "FAIL"))
    if not ok and detail:
        print("      %s" % detail)


def real_cells():
    """(label, window means) for each of the 42 published per-domain cells."""
    sel = json.load(open(CS.SELECTION_FILE, encoding="utf-8"))["selected_windows"]
    clean = set(sel)
    for slug in CS.SERIES:
        pw = json.load(open(os.path.join(CS.PER_WINDOW_DIR, slug + ".json"),
                            encoding="utf-8"))["per_window"]
        for scope, rows in (("panel25", pw),
                            ("clean17", [w for w in pw if w["window_id"] in clean])):
            by = {}
            for w in rows:
                by.setdefault(w["domain"], []).append(w)
            for dom in sorted(by):
                if len(by[dom]) >= 2:
                    yield (slug, scope, dom), by[dom]


def _receipt_row(mid, value, scored_positions=51175, identical=True):
    """The minimum a row needs for joint_enrich.apply; the receipt supplies the rest."""
    return {
        "id": mid,
        "estimator": {"vocab_masking_policy": "full_stored_vocab"},
        "provenance": {"measured_by": "self-measured"},
        "metric": {"name": "mean_tokenwise_kld", "value": value},
        "measurement_scope": {"scored_positions": scored_positions, "contexts": 25},
        "determinism": {"run_count": 2, "identical_across_runs": identical,
                        "evidence_kind": "hidden_state_tensor_sha256"},
        "comparability": {"key": "cmp--selftest"},
        "uncertainty": {"method": "none"},
    }


def receipt_series_section():
    """T8: a receipt's per_context is a window source, and it is checked, not trusted."""
    print()
    print("=" * 78)
    print("T8. GLM-5.3 rows get the interval from their receipt's per_context")
    print("=" * 78)
    fp8 = "measurement--glm-5.3.fp8-dequantized.corpus5x5-v1"
    k4 = "measurement--glm-5.3.exl3-k4-wrldsuksgo2mars.corpus5x5-v1"
    try:
        series = JE.SERIES[fp8]
        pw = series.load()
        n = sum(w["count"] for w in pw)
        value = S.se_from_window_summaries(pw)["mean"]
        rows = JE.apply([_receipt_row(fp8, value, n)])
    except Exception as exc:                                  # the pre-fix tree
        check(False, "receipt-sourced series is declared and loads", repr(exc))
        return
    check(series.source == "receipt-per-context" and not series.panel,
          "the series is declared receipt-per-context with panel enrichment OFF")
    check(len(pw) == 25 and n == 51175 and len({w["window_id"] for w in pw}) == 25,
          "25 distinct windows, 51,175 scored positions, read from the receipt")
    check(len(rows) == 1, "no clean17 sibling is appended (%d rows out)" % len(rows))
    row = rows[0]
    u = row.get("uncertainty") or {}
    check(u.get("method") == "window_block_bootstrap_bca" and u.get("interval_kind") == "bca",
          "uncertainty.method is window_block_bootstrap_bca (bca)", repr(u.get("method")))
    check(u.get("clusters") == 25 and u.get("samples") == 51175
          and u.get("cluster_unit") == "window",
          "clusters 25, samples 51175, cluster_unit window")
    check(u.get("bootstrap_b") == JE.BOOTSTRAP_B and u.get("bootstrap_seed") == JE.SEED,
          "B=%d, seed=%d" % (JE.BOOTSTRAP_B, JE.SEED))
    check(u.get("ci95_low") is not None and u["ci95_low"] <= value <= u["ci95_high"],
          "the BCa interval brackets the value")
    check(u.get("se_clustered") is not None and "se_naive" not in u and "deff" not in u,
          "se_clustered is quoted; se_naive/deff are not (the receipt carries no std)")
    check(u.get("sigma_run") == 0.0 and u.get("sigma_run_runs") == 2
          and u.get("se_total") == u.get("se_clustered"),
          "bitwise-identical cold runs give sigma_run 0.0 over 2 runs")
    check("coverage_measured" not in u and "NOT measured" in u.get("note", ""),
          "no coverage_measured; the note says coverage is not measured for this panel")
    check("by_domain" not in row and "protocol" not in row
          and "scope_name" not in row["measurement_scope"],
          "no by_domain, protocol stamp or scope naming on a panel=False series")
    # A per_context that does not reproduce the headline is a refusal, not a footnote.
    for label, bad in (("value off by 1e-9", _receipt_row(fp8, value + 1e-9, n)),
                       ("scored_positions off by one", _receipt_row(fp8, value, n - 1))):
        try:
            JE.apply([bad])
            check(False, "refuses a row whose %s" % label)
        except SystemExit as exc:
            check("joint_enrich" in str(exc), "refuses a row whose %s" % label, str(exc))
    # Two rows in one group get the paired ordering footnote.
    pw_k4 = JE.SERIES[k4].load()
    v_k4 = S.se_from_window_summaries(pw_k4)["mean"]
    both = JE.apply([_receipt_row(k4, v_k4, n), _receipt_row(fp8, value, n)])
    pairs = JE.orderings(both)
    check(len(pairs) == 1 and pairs[0]["lower"] == fp8 and pairs[0]["higher"] == k4
          and pairs[0]["windows"] == 25 and pairs[0]["wins"] + pairs[0]["ties"] <= 25
          and 0.0 < pairs[0]["sign_test_p"] <= 1.0,
          "orderings() pairs the two rows lowest-first with a sign-test p in (0, 1]")
    check(all("Ordering vs" in r["uncertainty"]["note"] for r in both),
          "each row's note carries the ordering sentence")
    check(sum(1 for r in both if "Ordering vs" in r["uncertainty"]["note"]) == 2
          and "Ordering vs" not in u["note"],
          "a row alone in its group gets no ordering sentence")


def main():
    import numpy as np

    print("=" * 78)
    print("T0. the estimator under test actually carries the fix")
    print("=" * 78)
    # Without this the whole file dies on an AttributeError against a pre-fix tree,
    # and a traceback is a worse answer than a named failure: "reverted and it
    # crashed" does not distinguish a missing fix from a broken test.
    missing = [n for n in ("derive_seed", "delta_t_log", "bootstrap_t_log",
                           "DOMAIN_BOOTSTRAP_B", "SMALL_G")
               if not hasattr(S, n)]
    check(not missing, "jointstd.stats exposes the post-2026-08-30 estimator",
          "missing: %r -- this tree predates the STAT-01/STAT-17 reseed" % missing)
    if missing:
        print()
        print("-" * 78)
        print("%d passed, %d failed" % (len(_passed), len(_failed)))
        return 1

    print()
    print("=" * 78)
    print("T1. the per-domain seeds are distinct, derived and reproducible (STAT-17)")
    print("=" * 78)
    seeds = {d: S.derive_seed(S.BOOTSTRAP_SEED, d) for d in DOMAINS}
    check(len(set(seeds.values())) == len(DOMAINS),
          "four domains, four distinct seeds", repr(seeds))
    check(all(S.derive_seed(S.BOOTSTRAP_SEED, d) == seeds[d] for d in DOMAINS),
          "derive_seed is a pure function of (seed, domain)")
    check(all(v != S.BOOTSTRAP_SEED for v in seeds.values()),
          "no domain reuses the base seed")

    print()
    print("=" * 78)
    print("T2. with the OLD shared seed the strata really did share their stream")
    print("=" * 78)
    # The defect, demonstrated rather than described: two domains with the same
    # window count and the same seed draw the SAME resample index stream, so their
    # Monte-Carlo error is common and it pairs unrelated windows.
    g = 6
    a = np.random.default_rng(S.BOOTSTRAP_SEED).integers(0, g, size=(200, g))
    b = np.random.default_rng(S.BOOTSTRAP_SEED).integers(0, g, size=(200, g))
    check(np.array_equal(a, b), "shared seed => byte-identical index streams (the defect)")
    c = np.random.default_rng(seeds["axis2_legal"]).integers(0, g, size=(200, g))
    d = np.random.default_rng(seeds["axis3_code_agentic"]).integers(0, g, size=(200, g))
    check(not np.array_equal(c, d), "derived seeds => different index streams (the fix)")

    print()
    print("=" * 78)
    print("T3. the published interval is non-negative, brackets the mean, and is sane")
    print("=" * 78)
    neg = bad_brack = 0
    worst_delta = worst_boot = 0.0
    worst_boot_cell = None
    for (slug, scope, dom), ws in real_cells():
        means = {w["window_id"]: float(w["mean"]) for w in ws}
        dt = S.delta_t_log(means)
        lo, hi = dt["ci95_delta_t_log"]
        if lo < 0:
            neg += 1
        if not (lo <= dt["observed"] <= hi):
            bad_brack += 1
        worst_delta = max(worst_delta, hi / dt["observed"])
        # The rejected candidate, on the same data. Measured here rather than
        # asserted in a comment: the reason delta_t_log is published instead of
        # bootstrap-t is a number, and this is where the number comes from.
        bt = S.bootstrap_t_log(means, b=20000,
                               seed=S.derive_seed(S.BOOTSTRAP_SEED, dom), backend="numpy")
        r = bt["ci95_t_log"][1] / bt["observed"]
        if r > worst_boot:
            worst_boot, worst_boot_cell = r, (slug, scope, dom)
    check(neg == 0, "no cell has a negative lower bound on a KL divergence",
          "%d cells went negative" % neg)
    check(bad_brack == 0, "every cell's interval brackets its own mean",
          "%d cells did not" % bad_brack)
    check(worst_delta < 5.0,
          "the published upper endpoint stays within 5x the estimate (worst %.1fx)"
          % worst_delta)
    check(worst_boot > 10.0,
          "bootstrap-t, the rejected candidate, reaches %.0fx on %s -- which is why it "
          "is not what ships" % (worst_boot, "/".join(worst_boot_cell or ())))

    print()
    print("=" * 78)
    print("T4. the OLD endpoints are still regenerable, so the delta is attributable")
    print("=" * 78)
    # A change to published numbers that cannot reproduce the numbers it replaced
    # is not a correction, it is a replacement nobody can audit. `interval="bca"`
    # with the old shared seed and B=1000 IS the pre-reseed procedure.
    old_path = os.path.join(_REGISTRY, "protocol", "coverage",
                            "pre-reseed-by-domain-endpoints.json")
    if not os.path.exists(old_path):
        check(False, "pre-reseed endpoint record exists", old_path)
    else:
        want = json.load(open(old_path, encoding="utf-8"))["endpoints"]
        miss = 0
        for (slug, scope, dom), ws in real_cells():
            rows = S.domain_table(ws, b=1000, seed=S.BOOTSTRAP_SEED, backend="numpy",
                                  interval="bca")
            row = next(r for r in rows if r["domain"] == dom)
            key = "%s|%s|%s" % (slug, scope, dom)
            if key not in want:
                miss += 1
                continue
            # domain_table now derives the seed, so regenerating the OLD number
            # means asking for the old procedure at the old SHARED seed.
            bs = S.window_block_bootstrap({w["window_id"]: float(w["mean"]) for w in ws},
                                          b=1000, seed=S.BOOTSTRAP_SEED, backend="numpy")
            got = [JE._round(bs["ci95_bca"][0]), JE._round(bs["ci95_bca"][1])]
            if got != want[key]:
                miss += 1
                print("      %s regenerated %r, published %r" % (key, got, want[key]))
        check(miss == 0, "all 42 pre-reseed BCa cells regenerate bit-for-bit",
              "%d did not" % miss)

    print()
    print("=" * 78)
    print("T5. the coverage simulator measures the SHIPPED code, not a lookalike")
    print("=" * 78)
    bad_bca = bad_t = bad_d = 0
    for (slug, scope, dom), ws in real_cells():
        vals = [float(w["mean"]) for w in sorted(ws, key=lambda x: x["window_id"])]
        means = {w["window_id"]: float(w["mean"]) for w in ws}
        seed = S.derive_seed(S.BOOTSTRAP_SEED, dom)
        ref = S.window_block_bootstrap(means, b=1000, seed=seed, backend="numpy")
        rng = np.random.default_rng(seed)
        if list(CS._bca(np.asarray(vals), 1000, rng, np)) != list(ref["ci95_bca"]):
            bad_bca += 1
        ref_t = S.bootstrap_t_log(means, b=2000, seed=seed, backend="numpy")
        rng = np.random.default_rng(seed)
        if list(CS._t_log(np.asarray(vals), 2000, rng, np)) != list(ref_t["ci95_t_log"]):
            bad_t += 1
        ref_d = S.delta_t_log(means)
        if list(CS._delta_t_log(np.asarray(vals), 1, None, np)) != list(ref_d["ci95_delta_t_log"]):
            bad_d += 1
    check(bad_bca == 0, "coverage_sim._bca == jointstd window_block_bootstrap on 42 cells",
          "%d differed" % bad_bca)
    check(bad_t == 0, "coverage_sim._t_log == jointstd bootstrap_t_log on 42 cells",
          "%d differed" % bad_t)
    check(bad_d == 0, "coverage_sim._delta_t_log == jointstd delta_t_log on 42 cells",
          "%d differed" % bad_d)

    print()
    print("=" * 78)
    print("T5b. the Student-t quantile is exact, not a normal approximation")
    print("=" * 78)
    from jointstd import chi2 as _c
    table = {1: 12.70620474, 2: 4.30265273, 4: 2.77644511, 5: 2.57058184,
             6: 2.44691185, 10: 2.22813885, 30: 2.04227246, 100: 1.98397152}
    worst = max(abs(_c.student_t_ppf(0.975, df) - v) for df, v in table.items())
    check(worst < 1e-7, "student_t_ppf matches the textbook table (worst %.2e)" % worst)
    # The reason it has to be exact: at g=5 the normal understates by 42%.
    check(abs(_c.student_t_ppf(0.975, 4) / 1.959963985 - 1.0) > 0.4,
          "at df=4 the t quantile is >40%% above the normal one, so approximating it "
          "would understate the published interval")

    print()
    print("=" * 78)
    print("T6. the published coverage figures come from the committed simulation")
    print("=" * 78)
    cov = json.load(open(JE.COVERAGE_FILE, encoding="utf-8"))
    index = {(c["series"], c["scope"], c["domain"]): c["procedures"] for c in cov["cells"]}
    rows = [json.loads(l) for l in open(os.path.join(_REGISTRY, "data",
                                                     "measurements.jsonl"),
                                        encoding="utf-8")]
    slug_of = {mid: s.slug for mid, s in JE.SERIES.items()}
    checked = mismatch = 0
    for r in rows:
        for cell in r.get("by_domain") or []:
            cm = cell.get("coverage_measured")
            if not cm:
                continue
            base = r["id"][:-len(JE.CLEAN_SUFFIX)] if r["id"].endswith(JE.CLEAN_SUFFIX) else r["id"]
            scope = "clean17" if r["id"].endswith(JE.CLEAN_SUFFIX) else "panel25"
            key = (slug_of.get(base), scope, cell["domain"])
            checked += 1
            want = index.get(key, {}).get(JE.DOMAIN_PROCEDURE, {}).get("coverage")
            if want != cm["measured"]:
                mismatch += 1
                print("      %s/%s published %r, simulation says %r"
                      % (r["id"], cell["domain"], cm["measured"], want))
    check(checked == 42, "all 42 per-domain cells publish a measured coverage",
          "found %d" % checked)
    check(mismatch == 0, "every published coverage equals the committed simulation",
          "%d mismatched" % mismatch)

    print()
    print("=" * 78)
    print("T7. the correction is real: new coverage beats old on every cell")
    print("=" * 78)
    worse = []
    olds, news = [], []
    for c in cov["cells"]:
        if c["block"] != "domain":
            continue
        o = c["procedures"]["old_bca_b1000"]["coverage"]
        n = c["procedures"][JE.DOMAIN_PROCEDURE]["coverage"]
        olds.append(o)
        news.append(n)
        if n <= o:
            worse.append((c["series"], c["scope"], c["domain"], o, n))
    check(not worse, "no cell got worse", repr(worse[:3]))
    check(sum(olds) / len(olds) < 0.85,
          "the OLD procedure measures below 85%% (mean %.1f%%)" % (100 * sum(olds) / len(olds)))
    check(sum(news) / len(news) > 0.90,
          "the NEW procedure measures above 90%% (mean %.1f%%)" % (100 * sum(news) / len(news)))
    # And the honest half: it is NOT 95%, and the rows say so rather than implying it.
    check(sum(news) / len(news) < 0.95,
          "the NEW procedure is still not nominal, which is why coverage_measured is "
          "published (mean %.1f%%)" % (100 * sum(news) / len(news)))
    bad_dir = [c for c in cov["cells"] if c["block"] == "domain"
               and c["procedures"]["old_bca_b1000"]["miss_low"]
               <= c["procedures"]["old_bca_b1000"]["miss_high"]]
    check(not bad_dir,
          "the OLD interval missed HIGH on every cell: it understated divergence",
          "%d cells did not" % len(bad_dir))

    receipt_series_section()

    print()
    print("-" * 78)
    print("%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
