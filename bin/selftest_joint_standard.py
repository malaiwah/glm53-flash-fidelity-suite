#!/usr/bin/env python3
"""Validation for bin/jointstd -- the adopted joint-standard rigor.

Nothing here is a smoke test.  Every case is one of:

  KNOWN ANSWER   an endpoint brandonmusic published, reproduced from his
                 per-window means alone by our independent implementation, or
                 a value computable by hand and written out in the assertion.
  ORACLE         our result versus HIS code (``kld_eval``) on the same input,
                 when his package is importable.  SKIPs loudly otherwise.
  FIRE           a gate handed input it must reject.  A canary nobody has seen
                 fire is decoration.

Run:  python3 bin/selftest_joint_standard.py [-v]
Exit: 0 all passed, 1 a case failed.

Optional inputs, each SKIPping rather than failing when absent:
  JOINTSTD_PANEL     brandonmusic's panel.json
  JOINTSTD_ARRAYS    directory of <window>.tokens.npy from his teacher dataset
  JOINTSTD_TEACHER   one teacher logits .safetensors (1.27 GB) for the real R0
  PYTHONPATH=<his eval/kld>   makes the kld_eval oracle available
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from jointstd import canary as canary_mod      # noqa: E402
from jointstd import chi2 as chi2_mod          # noqa: E402
from jointstd import ngram as ngram_mod        # noqa: E402
from jointstd import oracle as oracle_mod      # noqa: E402
from jointstd import protocol as protocol_mod  # noqa: E402
from jointstd import stats as stats_mod        # noqa: E402

FIXTURE = os.path.join(HERE, "jointstd", "fixtures", "brandonmusic-known-answer.json")
SELECTION = os.path.join(ROOT, "registry", "protocol",
                         "window-selection.brandonmusic-final25.json")

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
PASS = FAIL = SKIP = 0
_SECTION = ""


def section(name: str) -> None:
    global _SECTION
    _SECTION = name
    print()
    print("=" * 78)
    print(name)
    print("=" * 78)


def ok(name: str, detail: str = "") -> None:
    global PASS
    PASS += 1
    print("  PASS  %-56s %s" % (name, detail))


def bad(name: str, detail: str) -> None:
    global FAIL
    FAIL += 1
    print("  FAIL  %-56s %s" % (name, detail))


def skip(name: str, why: str) -> None:
    global SKIP
    SKIP += 1
    print("  SKIP  %-56s (%s)" % (name, why))


def close(name: str, got: float, want: float, tol: float, fmt: str = "%.15g") -> None:
    d = abs(got - want)
    if d <= tol:
        ok(name, ("got " + fmt + "  |d|=%.2e") % (got, d))
    else:
        bad(name, ("got " + fmt + " want " + fmt + "  |d|=%.2e > %.1e")
            % (got, want, d, tol))


def load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ======================================================================== 1
def t_protocol() -> None:
    section("1. THE FROZEN PROTOCOL FILE AND ITS TWO HASHES")
    p = protocol_mod.load()
    ok("protocol loads", "%s" % os.path.relpath(p.path, ROOT))
    ok("file sha256", p.file_sha256)
    ok("scoring sha256", p.scoring_sha256)

    p2 = protocol_mod.load()
    if p2.file_sha256 == p.file_sha256 and p2.scoring_sha256 == p.scoring_sha256:
        ok("both hashes are stable across loads", "")
    else:
        bad("hash stability", "two loads disagree")

    # The drift fix: an identity-only edit must move the file hash and NOT the
    # scoring hash.  This is exactly the edit that broke his campaign.
    doc = json.loads(p.raw.decode("utf-8"))
    doc["derived_from"] = dict(doc["derived_from"])
    doc["derived_from"]["note"] = doc["derived_from"]["note"] + " (edited)"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh, indent=2)
        tmp = fh.name
    try:
        q = protocol_mod.load(tmp)
        if q.file_sha256 != p.file_sha256 and q.scoring_sha256 == p.scoring_sha256:
            ok("identity-only edit: file hash MOVES, scoring hash HOLDS",
               "%s -> %s" % (p.file_sha256[:12], q.file_sha256[:12]))
        else:
            bad("scoring-hash invariance",
                "file %s scoring %s" % (q.file_sha256 == p.file_sha256,
                                        q.scoring_sha256 == p.scoring_sha256))
        # ... and a scoring edit must move BOTH
        doc["scoring"]["padded_column_policy"] = "no_mask"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        r = protocol_mod.load(tmp)
        if r.scoring_sha256 != p.scoring_sha256:
            ok("a scoring edit MOVES the scoring hash",
               "%s -> %s" % (p.scoring_sha256[:12], r.scoring_sha256[:12]))
        else:
            bad("scoring-hash sensitivity", "a scoring edit did not move it")
    finally:
        os.unlink(tmp)

    # FIRE: require_stamp must refuse
    for label, doc_, exc in (
        ("unstamped receipt", {}, True),
        ("foreign protocol schema",
         {"protocol_schema": "someone-elses.v9",
          "protocol_file_sha256": "x", "protocol_scoring_sha256": "y"}, True),
        ("stale scoring hash",
         {"protocol_schema": protocol_mod.PROTOCOL_SCHEMA,
          "protocol_file_sha256": p.file_sha256,
          "protocol_scoring_sha256": "0" * 64}, True),
        ("correctly stamped receipt", p.stamp(), False),
    ):
        try:
            protocol_mod.require_stamp(dict(doc_), p)
            raised = False
        except protocol_mod.ProtocolError:
            raised = True
        if raised == exc:
            ok("require_stamp: %s" % label, "refused" if exc else "accepted")
        else:
            bad("require_stamp: %s" % label, "raised=%s expected=%s" % (raised, exc))


# ======================================================================== 2
def t_chi2_and_mcnemar() -> None:
    section("2. McNEMAR -- HAND-COMPUTED TABLE, THEN HIS FIVE PUBLISHED p-VALUES")

    # ---- hand-computed 2x2 --------------------------------------------
    # A right / B wrong: 12.  B right / A wrong: 3.  Concordant pairs drop out.
    #   chi2 = (|12-3| - 1)^2 / (12+3) = 8^2/15 = 64/15 = 4.2666666666666666
    #   exact two-sided binomial at p0=1/2, n=15, k=3:
    #     2 * (C(15,0)+C(15,1)+C(15,2)+C(15,3)) / 2^15
    #   = 2 * (1 + 15 + 105 + 455) / 32768 = 1152/32768 = 0.03515625   EXACT
    m = stats_mod.mcnemar(12, 3)
    close("hand table chi2 = 64/15", m["chi2"], 64.0 / 15.0, 1e-15)
    close("hand table exact binomial = 1152/32768", m["p_exact"], 1152.0 / 32768.0, 0.0)
    if m["favours"] == "a" and m["discordant"] == 15:
        ok("hand table bookkeeping", "discordant=15 favours=a")
    else:
        bad("hand table bookkeeping", repr(m))

    # the chi2 tail, two independent algorithms inside our own module:
    # the df=1 closed form (erfc) versus the general regularized upper
    # incomplete gamma Q(1/2, x/2) continued fraction.
    worst = 0.0
    for x in (0.5, 1.0, 3.841458820694124, 4.2666666666666666, 10.0, 25.0, 170.611497):
        a = chi2_mod.chi2_sf(x, 1)
        b = chi2_mod._gamma_cf(0.5, x / 2.0) if x / 2.0 >= 1.5 else \
            1.0 - chi2_mod._gamma_series(0.5, x / 2.0)
        rel = abs(a - b) / max(b, 1e-300)
        worst = max(worst, rel)
    if worst < 1e-12:
        ok("chi2_sf: erfc form == incomplete-gamma form", "worst rel %.2e" % worst)
    else:
        bad("chi2_sf cross-algorithm", "worst rel %.2e" % worst)
    # the 5% critical value of chi2_1 is 3.841458820694124
    close("chi2_sf(3.8414588207, 1) == 0.05", chi2_mod.chi2_sf(3.841458820694124, 1),
          0.05, 1e-12)

    # ---- his five published p-values ----------------------------------
    fix = load_fixture()
    cases = [("nvfp4-vs-exl3 clean", fix["expected_paired"]["clean"]["mcnemar"]),
             ("nvfp4-vs-exl3 panel", fix["expected_paired"]["panel"]["mcnemar"])]
    for k, v in sorted(fix["expected_mcnemar_extra"].items()):
        cases.append(("paired_%s" % k, v["mcnemar"]))
    for label, cell in cases:
        got = stats_mod.mcnemar(cell["a_only_correct"], cell["b_only_correct"])
        want = cell["p"]
        rel = abs(got["p"] - want) / want if want else abs(got["p"])
        if rel < 1e-10:
            ok("published p reproduced: %s" % label,
               "%d/%d  p=%.6e  rel=%.1e" % (cell["a_only_correct"],
                                            cell["b_only_correct"], got["p"], rel))
        else:
            bad("published p: %s" % label, "ours %.6e his %.6e rel %.2e"
                % (got["p"], want, rel))

    # FIRE: a table with no discordant pairs carries no evidence
    z = stats_mod.mcnemar(0, 0)
    if z["p"] == 1.0 and z["chi2"] is None:
        ok("no discordant pairs -> p=1, no chi2", z["note"])
    else:
        bad("empty McNemar table", repr(z))


# ======================================================================== 3
def t_stats_refusals() -> None:
    section("2b. REFUSALS AND EDGE CASES THAT USED TO ANSWER ANYWAY")

    # STAT-14: the exact binomial was abandoned above 2000 discordant pairs, which
    # excluded this tool's OWN worked example (--a-only 1629 --b-only 963, n=2592)
    # and both McNemar tables in the known-answer fixture.
    r = stats_mod.mcnemar(1629, 963)
    if r.get("p_exact") is not None and abs(r["p_exact"] - 2.0803213154515343e-39) < 1e-52:
        ok("STAT-14: the documented worked example has an exact p",
           "n=%d p_exact=%.6e (pre-fix: None, 'skipped above 2000')"
           % (r["discordant"], r["p_exact"]))
    else:
        bad("STAT-14: exact p at n=2592", "p_exact=%r" % r.get("p_exact"))
    r2 = stats_mod.mcnemar(1273, 2120)
    if r2.get("p_exact") is not None:
        ok("STAT-14: fixture panel table has an exact p",
           "n=%d p_exact=%.3e" % (r2["discordant"], r2["p_exact"]))
    else:
        bad("STAT-14: exact p at n=3393", "p_exact=%r" % r2.get("p_exact"))

    # STAT-18: all() over an EMPTY sequence is vacuously True, so the refusal helper
    # claimed pooled percentiles WERE derivable when handed no data at all.
    g = stats_mod.guard_pooled_percentiles([])
    if g.get("available") is False and "no windows" in (g.get("reason") or ""):
        ok("STAT-18: no windows is a refusal, not availability", g["reason"])
    else:
        bad("STAT-18: empty per_window", repr(g))
    g2 = stats_mod.guard_pooled_percentiles([{"window_id": "a", "mean": 0.1}])
    if g2.get("available") is False:
        ok("STAT-18: per-window summaries still refuse", "unchanged")
    else:
        bad("STAT-18: summaries", repr(g2))

    # STAT-19: deff was gated on TRUTHINESS, so a zero-variance panel silently lost
    # the key and joint_enrich's unconditional lookup raised KeyError.
    z = stats_mod.se_from_window_summaries([
        {"window_id": "a", "count": 10, "mean": 1.0, "std": 0.0},
        {"window_id": "b", "count": 10, "mean": 1.0, "std": 0.0}])
    if "deff_window" in z and z["deff_window"] is None:
        ok("STAT-19: zero-variance panel says deff is UNDEFINED, not missing",
           "se_naive=%r deff_window=None (pre-fix: key absent -> KeyError)"
           % z.get("se_naive"))
    else:
        bad("STAT-19: zero-variance deff",
            "deff_window present=%r value=%r" % ("deff_window" in z, z.get("deff_window")))
    nz = stats_mod.se_from_window_summaries([
        {"window_id": "a", "count": 10, "mean": 1.0, "std": 0.1},
        {"window_id": "b", "count": 10, "mean": 2.0, "std": 0.2}])
    if isinstance(nz.get("deff_window"), float):
        ok("STAT-19: a real panel still gets a numeric deff", "%.4f" % nz["deff_window"])
    else:
        bad("STAT-19: numeric deff", repr(nz.get("deff_window")))

    # STAT-13: a window shorter than the n-gram width was reported as perfectly
    # clean. The fixture below is a VERBATIM PREFIX of the calibration corpus.
    cal = ngram_mod.token_ngrams(list(range(100)), 13)
    fins = [
        {"window_id": "SHORT", "document_id": "C", "domain": "d",
         "tokens": list(range(12))},                       # 12 < 13 -> no grams at all
        {"window_id": "CLEAN", "document_id": "A", "domain": "d",
         "tokens": list(range(500, 700))},
    ]
    sc = ngram_mod.scan(fins, cal, set(), n=13, threshold=0.05)
    short = [w for w in sc["per_window"] if w["window_id"] == "SHORT"][0]
    if (short.get("scannable") is False
            and short.get("shared_ngram_fraction") is None
            and "SHORT" not in sc["selected_windows"]):
        ok("STAT-13: an unscannable window is refused, not certified clean",
           "12 tokens < 13-gram width -> excluded (pre-fix: fraction 0.0, SELECTED)")
    else:
        bad("STAT-13: short window",
            "scannable=%r frac=%r selected=%r" % (short.get("scannable"),
                                                  short.get("shared_ngram_fraction"),
                                                  "SHORT" in sc["selected_windows"]))
    if "CLEAN" in sc["selected_windows"]:
        ok("STAT-13: a genuinely clean window is still selected", "no false refusal")
    else:
        bad("STAT-13: clean window", "was excluded")


def t_clustered_se() -> None:
    section("3. CLUSTER-ROBUST SE -- HAND ARITHMETIC, THEN HIS PUBLISHED VALUES")

    # ---- hand-computed ------------------------------------------------
    # Two clusters of two.  values = [1,3 | 10,12], mean = 6.5, N = 4, g = 2.
    #   T_A = 4,  n_A*mean = 13  -> resid -9
    #   T_B = 22, n_B*mean = 13  -> resid  9
    #   se = sqrt(2/1 * (81+81)) / 4 = sqrt(324)/4 = 18/4 = 4.5   EXACT
    r = stats_mod.clustered_se([1.0, 3.0, 10.0, 12.0], ["A", "A", "B", "B"])
    close("hand 2-cluster se = 18/4", r["se"], 4.5, 1e-15)
    # naive se = stdev([1,3,10,12])/2 ; stdev = sqrt(((5.5^2)+(3.5^2)+(3.5^2)+(5.5^2))/3)
    naive = math.sqrt((5.5 ** 2 + 3.5 ** 2 + 3.5 ** 2 + 5.5 ** 2) / 3.0) / 2.0
    close("hand naive se", r["se_naive"], naive, 1e-15)
    close("hand deff = (se/naive)^2", r["deff"], (4.5 / naive) ** 2, 1e-13)

    # ---- reproduce his published clustered SEs from window means -------
    fix = load_fixture()
    sel = set(json.load(open(SELECTION))["selected_windows"])
    per = {"exl3": {w: v["mean_kld"] for w, v in fix["per_window"]["exl3_run1"].items()},
           "nvfp4": {w: v["mean_kld"] for w, v in fix["per_window"]["nvfp4_run1"].items()}}
    for tag in ("exl3", "nvfp4"):
        for scope, keep in (("selected", sel), ("panel", set(per[tag]))):
            m = {k: v for k, v in per[tag].items() if k in keep}
            pw = [{"window_id": w, "count": 2047, "mean": v} for w, v in m.items()]
            s = stats_mod.se_from_window_summaries(pw)
            e = fix["expected"]["%s_%s" % (tag, scope)]["summary"]
            close("se_clustered_window %s/%s" % (tag, scope),
                  s["se_clustered_window"], e["se_clustered_window"], 1e-15, "%.12e")
            close("mean %s/%s" % (tag, scope), s["mean"], e["mean"], 1e-15)

    # the same block from OUR receipts, where std is present -> se_naive/deff
    p = os.path.join(ROOT, "registry/protocol/per-window/k8-streaming.json")
    if os.path.exists(p):
        pw = json.load(open(p))["per_window"]
        s = stats_mod.se_from_window_summaries(pw)
        # pooled std must reproduce the panel std the same receipt reports
        want_std = 0.06827937096765291   # k8 summary.std, 25 windows / 51175 tokens
        close("pooled std from per-window (n,mean,std) == receipt panel std",
              s["pooled_std"], want_std, 5e-12)
        ok("deff from our own receipt", "deff_window=%.2f (naive SE understates by %.1fx)"
           % (s["deff_window"], math.sqrt(s["deff_window"])))
    else:
        skip("pooled std from our receipts", "per-window inputs not present")

    # ORACLE
    st = oracle_mod.probe()
    if st["available"]:
        o = oracle_mod.clustered_se_via_kld_eval([1.0, 3.0, 10.0, 12.0],
                                                 ["A", "A", "B", "B"])
        close("oracle kld_eval.clustered_se agrees", o["se"], r["se"], 1e-15)
    else:
        skip("oracle clustered_se", st["reason"] or "kld_eval not importable")


# ======================================================================== 4
def t_bootstrap() -> None:
    section("4. WINDOW BLOCK BOOTSTRAP -- PERCENTILE + BCa")

    # ---- quantile helper against numpy's default ----------------------
    try:
        import numpy as np
        import random as _r

        rnd = _r.Random(7)
        worst = 0.0
        for _ in range(200):
            n = rnd.randint(2, 60)
            xs = sorted(rnd.uniform(-5, 5) for _ in range(n))
            for p in (0.0, 0.025, 0.5, 0.9, 0.975, 1.0):
                worst = max(worst, abs(stats_mod.quantile_linear(xs, p)
                                       - float(np.quantile(np.asarray(xs), p))))
        if worst < 1e-12:
            ok("quantile_linear == numpy default quantile", "worst |d| %.2e" % worst)
        else:
            bad("quantile_linear", "worst |d| %.2e" % worst)
    except Exception as exc:
        skip("quantile_linear vs numpy", repr(exc)[:60])

    # ---- degenerate known answer: all windows equal -------------------
    same = {"w%02d" % i: 0.25 for i in range(10)}
    bs = stats_mod.window_block_bootstrap(same, b=500, seed=1)
    if bs["ci95_percentile"] == [0.25, 0.25] and bs["ci95_bca"] == [0.25, 0.25]:
        ok("degenerate input -> CI collapses to the point", "[0.25, 0.25]")
    else:
        bad("degenerate bootstrap", repr(bs["ci95_percentile"]))

    # ---- HIS FOUR PUBLISHED ENDPOINT SETS -----------------------------
    fix = load_fixture()
    sel = set(json.load(open(SELECTION))["selected_windows"])
    per = {"exl3": {w: v["mean_kld"] for w, v in fix["per_window"]["exl3_run1"].items()},
           "nvfp4": {w: v["mean_kld"] for w, v in fix["per_window"]["nvfp4_run1"].items()}}
    try:
        import numpy  # noqa: F401
        have_np = True
    except Exception:
        have_np = False

    for tag in ("exl3", "nvfp4"):
        for scope, keep in (("selected", sel), ("panel", set(per[tag]))):
            m = {k: v for k, v in per[tag].items() if k in keep}
            e = fix["expected"]["%s_%s" % (tag, scope)]
            if have_np:
                b = stats_mod.window_block_bootstrap(
                    m, b=e["bootstrap_b"], seed=e["bootstrap_seed"], backend="numpy")
                for kind, mine, want in (
                        ("percentile", b["ci95_percentile"], e["ci95_percentile_mean"]),
                        ("BCa", b["ci95_bca"], e["ci95_bca_mean"])):
                    d = max(abs(x - y) for x, y in zip(mine, want))
                    if d < 1e-15:
                        ok("published %s CI %s/%s" % (kind, tag, scope),
                           "[%.12f, %.12f] max|d|=%.1e" % (mine[0], mine[1], d))
                    else:
                        bad("published %s CI %s/%s" % (kind, tag, scope),
                            "ours %s his %s max|d|=%.2e" % (mine, want, d))
            else:
                skip("published CIs %s/%s" % (tag, scope), "numpy absent")

    # ---- stdlib fallback must agree within Monte Carlo error ----------
    m = {k: v for k, v in per["exl3"].items() if k in sel}
    s_ = stats_mod.window_block_bootstrap(m, b=5000, seed=20260829, backend="stdlib")
    if have_np:
        n_ = stats_mod.window_block_bootstrap(m, b=5000, seed=20260829, backend="numpy")
        width = n_["ci95_percentile"][1] - n_["ci95_percentile"][0]
        d = max(abs(a - b) for a, b in zip(s_["ci95_bca"], n_["ci95_bca"]))
        if d < 0.05 * width:
            ok("stdlib backend agrees with numpy backend within MC error",
               "max|d|=%.2e = %.2f%% of the CI width" % (d, 100.0 * d / width))
        else:
            bad("stdlib vs numpy backend", "max|d|=%.2e vs width %.2e" % (d, width))
    else:
        skip("stdlib vs numpy backend", "numpy absent")

    # ---- ORACLE: his own block_bootstrap on the same input ------------
    st = oracle_mod.probe()
    if st["available"] and have_np:
        o = oracle_mod.block_bootstrap_via_kld_eval(m, b=5000, seed=20260829)
        n_ = stats_mod.window_block_bootstrap(m, b=5000, seed=20260829, backend="numpy")
        d = max(abs(a - b) for a, b in zip(o["ci95_bca"], n_["ci95_bca"]))
        if d < 1e-15:
            ok("ORACLE: kld_eval.block_bootstrap == ours", "max|d|=%.2e" % d)
        else:
            bad("oracle bootstrap", "max|d|=%.2e" % d)
    else:
        skip("ORACLE kld_eval.block_bootstrap", st["reason"] or "numpy absent")

    # ---- FIRE: one window cannot be block-bootstrapped ----------------
    try:
        stats_mod.window_block_bootstrap({"only": 0.1})
        bad("single-window bootstrap", "did not refuse")
    except ValueError:
        ok("FIRE: single window refuses to bootstrap", "ValueError")


# ======================================================================== 5
def t_paired() -> None:
    section("5. PAIRED RANKING -- HIS PUBLISHED diff/ratio ENDPOINTS")
    fix = load_fixture()
    sel = set(json.load(open(SELECTION))["selected_windows"])
    ex = {w: v["mean_kld"] for w, v in fix["per_window"]["exl3_run1"].items()}
    nv = {w: v["mean_kld"] for w, v in fix["per_window"]["nvfp4_run1"].items()}
    try:
        import numpy  # noqa: F401
    except Exception:
        skip("paired endpoints", "numpy absent")
        return
    for scope, keep in (("clean", sel), ("panel", set(ex))):
        a = {k: v for k, v in nv.items() if k in keep}
        b = {k: v for k, v in ex.items() if k in keep}
        r = stats_mod.paired_windows(a, b, "nvfp4", "exl3", boot_b=2000, seed=20260829,
                                     backend="numpy")
        e = fix["expected_paired"][scope]
        close("paired mean_diff (%s)" % scope, r["mean_diff"], e["mean_diff"], 1e-15)
        close("paired ratio (%s)" % scope, r["ratio_a_over_b"], e["ratio_a_over_b"], 1e-12)
        d = max(abs(x - y) for x, y in zip(r["ci95_diff_percentile"],
                                           e["mean_diff_ci95_bootstrap"]))
        if d < 1e-15:
            ok("paired diff CI (%s)" % scope,
               "[%.12f, %.12f] max|d|=%.1e" % (r["ci95_diff_percentile"][0],
                                               r["ci95_diff_percentile"][1], d))
        else:
            bad("paired diff CI (%s)" % scope, "max|d|=%.2e" % d)
        d = max(abs(x - y) for x, y in zip(r["ci95_ratio_percentile"],
                                           e["ratio_ci95_bootstrap"]))
        if d < 1e-12:
            ok("paired ratio CI (%s)" % scope,
               "[%.9f, %.9f] max|d|=%.1e" % (r["ci95_ratio_percentile"][0],
                                             r["ci95_ratio_percentile"][1], d))
        else:
            bad("paired ratio CI (%s)" % scope, "max|d|=%.2e" % d)

    # the pairing is the point: it must be tighter than the marginals
    a = {k: v for k, v in nv.items() if k in sel}
    b = {k: v for k, v in ex.items() if k in sel}
    r = stats_mod.paired_windows(a, b, "nvfp4", "exl3", boot_b=2000, seed=20260829)
    ma = stats_mod.window_block_bootstrap(a, b=2000, seed=20260829)
    mb = stats_mod.window_block_bootstrap(b, b=2000, seed=20260829)
    paired_w = r["ci95_diff_percentile"][1] - r["ci95_diff_percentile"][0]
    marginal_w = ((ma["ci95_percentile"][1] - ma["ci95_percentile"][0]) +
                  (mb["ci95_percentile"][1] - mb["ci95_percentile"][0]))
    if paired_w < marginal_w and r["excludes_zero"]:
        ok("paired CI is tighter than the two marginals and excludes 0",
           "paired %.6f vs marginals %.6f" % (paired_w, marginal_w))
    else:
        bad("paired vs marginal", "paired %.6f marginal %.6f" % (paired_w, marginal_w))

    # ---- STAT-03: a pinned backend must refuse, never silently substitute ------
    # The numpy and stdlib backends draw DIFFERENT resample streams from the same
    # seed (measured: published endpoints move up to 1.20%). A caller that pins the
    # backend is pinning the numbers, so an unavailable backend has to refuse.
    # Pre-fix this fell through to stdlib and answered with different endpoints.
    import builtins as _b
    _real_import = _b.__import__

    def _no_numpy(name, *a, **k):
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("numpy blocked for this test")
        return _real_import(name, *a, **k)

    _m = {"a": 0.011, "b": 0.012, "c": 0.013, "d": 0.014}
    _b.__import__ = _no_numpy
    try:
        try:
            stats_mod.window_block_bootstrap(_m, b=64, seed=20260829, backend="numpy")
            bad("STAT-03: pinned numpy backend with numpy absent",
                "returned an answer from the OTHER backend instead of refusing")
        except RuntimeError as exc:
            ok("STAT-03: pinned numpy backend refuses when numpy is absent",
               str(exc).split(".")[0][:64])
        except ImportError:
            bad("STAT-03: pinned numpy backend with numpy absent",
                "raised bare ImportError; the refusal must say WHY falling back "
                "would move published endpoints")
        # the stdlib backend must still work with numpy absent -- that is the whole
        # point of having it, and a refusal here would break the offline contract.
        r = stats_mod.window_block_bootstrap(_m, b=64, seed=20260829, backend="stdlib")
        if r and r.get("ci95_percentile"):
            ok("STAT-03: stdlib backend still runs with numpy absent",
               "offline path intact")
        else:
            bad("STAT-03: stdlib backend with numpy absent", "no result")
    finally:
        _b.__import__ = _real_import

    # FIRE: fewer than two common windows
    try:
        stats_mod.paired_windows({"a": 1.0}, {"b": 2.0})
        bad("paired with no common windows", "did not refuse")
    except ValueError:
        ok("FIRE: no common windows refuses to pair", "ValueError")

    # ---- STAT-02: exact ties carry no sign -------------------------------------
    # The sign test counted ties as wins for B and left them in the binomial
    # denominator, which made a SYMMETRIC test argument-order dependent. Each of
    # these three assertions fails on the pre-fix tool.
    selfcmp = stats_mod.paired_windows(dict(b), dict(b), "X", "X",
                                       boot_b=200, seed=20260829)
    if (selfcmp.get("windows_tied") == selfcmp["n_windows"]
            and selfcmp.get("windows_a_better") == 0
            and selfcmp.get("windows_b_better") == 0
            and selfcmp.get("sign_test_p", "absent") is None
            and selfcmp.get("sign_test_n") == 0):
        ok("STAT-02: a series against ITSELF is all ties, no p",
           "%d/%d tied, sign_test_p None (pre-fix: 25-0 and p=5.96e-08)"
           % (selfcmp["windows_tied"], selfcmp["n_windows"]))
    else:
        bad("STAT-02: self-comparison",
            "tied=%r a=%r b=%r p=%r" % (selfcmp.get("windows_tied"),
                                        selfcmp.get("windows_a_better"),
                                        selfcmp.get("windows_b_better"),
                                        selfcmp.get("sign_test_p")))

    # the real pair a reader would compare next: K6 sealed vs K6 streaming share
    # 11 EXACT per-window ties, so the tie handling is not hypothetical here.
    k6s = {w["window_id"]: w["mean"] for w in
           json.load(open(os.path.join(ROOT, "registry/protocol/per-window/k6-sealed.json")))["per_window"]}
    k6t = {w["window_id"]: w["mean"] for w in
           json.load(open(os.path.join(ROOT, "registry/protocol/per-window/k6-streaming.json")))["per_window"]}
    rp = stats_mod.paired_windows(k6s, k6t, "sealed", "stream", boot_b=200, seed=20260829)
    want_p = chi2_mod.binom_sf_two_sided(9, 14)   # 9 informative wins of 14, NOT of 25
    if (rp.get("windows_tied") == 11 and rp.get("sign_test_n") == 14
            and rp.get("sign_test_p") is not None
            and abs(rp["sign_test_p"] - want_p) < 1e-15):
        ok("STAT-02: K6 sealed vs streaming, 11 exact ties excluded",
           "9/14 p=%.10f (pre-fix: 9/25 p=%.10f)"
           % (rp["sign_test_p"], chi2_mod.binom_sf_two_sided(9, 25)))
    else:
        bad("STAT-02: K6 sealed vs streaming",
            "tied=%r n=%r p=%r" % (rp.get("windows_tied"), rp.get("sign_test_n"),
                                   rp.get("sign_test_p")))

    # a symmetric test may not depend on which series is called A
    rev = stats_mod.paired_windows(k6t, k6s, "stream", "sealed", boot_b=200, seed=20260829)
    if rp.get("sign_test_p") == rev.get("sign_test_p"):
        ok("STAT-02: sign test is argument-order invariant",
           "p=%.10f both ways (pre-fix: 0.2295 vs 0.0041, across 0.05)"
           % (rp.get("sign_test_p") or float("nan")))
    else:
        bad("STAT-02: argument-order invariance",
            "a,b -> %r ; b,a -> %r" % (rp.get("sign_test_p"), rev.get("sign_test_p")))


# ======================================================================== 6
def t_sigma_run() -> None:
    section("6. sigma_run AND SE IN QUADRATURE -- HIS PUBLISHED SWEEPS")
    fix = load_fixture()
    for label, case in sorted(fix["expected_sigma_run"].items()):
        means = list(case["per_run_mean_kld"].values())
        got = stats_mod.sigma_run(means)
        close("sigma_run %s (%d runs)" % (label, len(means)),
              got["sigma_run"], case["sigma_run"], 1e-18, "%.12e")
        if len(means) == 2 and "two-run" not in (got.get("note") or ""):
            bad("two-run disclosure %s" % label, "no 1-dof flag")
        elif len(means) == 2:
            ok("two-run sigma carries its 1-dof flag", got["note"][:52] + "...")

    # a 2-run sigma is |delta|/sqrt(2): check it algebraically
    case = fix["expected_sigma_run"]["nvfp4_2cold_25w"]
    a, b = list(case["per_run_mean_kld"].values())
    close("2-run sigma == |delta|/sqrt(2)", case["sigma_run"],
          abs(a - b) / math.sqrt(2.0), 1e-18, "%.12e")

    # his quadrature: SE_total = hypot(5.990897411031214e-3, 3.3320411112846096e-4)
    se_stat = fix["expected"]["nvfp4_panel"]["summary"]["se_clustered_window"]
    sig = case["sigma_run"]
    q = stats_mod.combine_quadrature(se_stat, sig)
    close("SE_total = hypot(SE_clust, sigma_run)", q["se_total"],
          math.hypot(se_stat, sig), 0.0, "%.12e")
    close("his published SE_total 6.00e-3", q["se_total"], 6.000156e-3, 1e-9, "%.9e")
    if abs(q["ratio"] - 0.05562) < 1e-4 and q["gate_ok"]:
        ok("sigma_run/SE ratio inside his 0.20 gate", "ratio=%.5f" % q["ratio"])
    else:
        bad("quadrature gate", "ratio=%.6f gate_ok=%s" % (q["ratio"], q["gate_ok"]))

    # FIRE: a run term that is NOT negligible must trip the gate
    q2 = stats_mod.combine_quadrature(1e-3, 5e-4)
    if not q2["gate_ok"]:
        ok("FIRE: sigma_run/SE = 0.5 trips the 0.20 gate", q2["note"][:48] + "...")
    else:
        bad("quadrature gate", "0.5 did not trip")
    z = stats_mod.sigma_run([0.1])
    if z["sigma_run"] is None:
        ok("one run -> sigma_run is not estimable", z["note"])
    else:
        bad("one-run sigma", repr(z))

    # ---- ci95_total: a live run term has to produce an INTERVAL, not just an SE.
    # The complaint this answers: the receipt said "quote SE_total" and then gave
    # the reader no interval built from SE_total, because the BCa endpoints
    # resample windows within ONE run and cannot contain sigma_run.
    m = 0.0305
    q3 = stats_mod.combine_quadrature(1e-3, 5e-4, mean=m)
    if q3["ci95_total"] is None:
        bad("ci95_total on a live sigma_run", "not emitted")
    else:
        lo, hi = q3["ci95_total"]
        close("ci95_total low  == mean - 1.96*SE_total", lo,
              m - 1.96 * math.hypot(1e-3, 5e-4), 0.0, "%.15e")
        close("ci95_total high == mean + 1.96*SE_total", hi,
              m + 1.96 * math.hypot(1e-3, 5e-4), 0.0, "%.15e")
        if q3["interval_kind"] == "z" and "NOT BCa" not in q3["note"] and "not BCa" in q3["note"]:
            ok("ci95_total is labelled a z-interval, not BCa", q3["note"][-46:])
        elif q3["interval_kind"] == "z":
            ok("ci95_total is labelled a z-interval, not BCa",
               "interval_kind=%s" % q3["interval_kind"])
        else:
            bad("ci95_total kind", repr(q3["interval_kind"]))
        # it must be WIDER than the statistical half alone, or it is pointless
        if (hi - lo) > 2 * 1.96 * 1e-3:
            ok("ci95_total is wider than the statistical SE alone",
               "%.4e vs %.4e" % (hi - lo, 2 * 1.96 * 1e-3))
        else:
            bad("ci95_total width", "%.6e" % (hi - lo))

    # sigma_run == 0.0 must NOT emit a second, worse-shaped copy of the BCa interval
    q4 = stats_mod.combine_quadrature(1e-3, 0.0, mean=m)
    if q4["ci95_total"] is None and q4["se_total"] == 1e-3:
        ok("sigma_run == 0 -> no ci95_total; BCa already is the total",
           q4["note"][-44:])
    else:
        bad("ci95_total at sigma_run=0", repr(q4["ci95_total"]))
    # and neither should the not-estimable case
    q5 = stats_mod.combine_quadrature(1e-3, None, mean=m)
    if q5["ci95_total"] is None:
        ok("sigma_run not estimable -> no ci95_total", q5["note"])
    else:
        bad("ci95_total when sigma_run is None", repr(q5["ci95_total"]))
    # the old 3-positional-arg call must still work unchanged
    q6 = stats_mod.combine_quadrature(1e-3, 5e-4, 0.2)
    if q6["se_total"] == math.hypot(1e-3, 5e-4) and q6["ci95_total"] is None:
        ok("mean is optional: the pre-existing call site is unchanged",
           "se_total=%.9e" % q6["se_total"])
    else:
        bad("combine_quadrature back-compat", repr(q6))


# ======================================================================== 7
def t_percentile_guard() -> None:
    section("7. PERCENTILE-EXCEEDANCE GUARD")
    cases = [
        (34799, 0.90, True), (34799, 0.95, True), (34799, 0.99, True),
        (34799, 0.999, False),          # 34.8 exceedances -- his suppressed case
        (51175, 0.999, False),          # 51.2 exceedances
        (1000, 0.90, True),             # exactly 100 -- boundary is inclusive
        (999, 0.90, False),             # 99.9
        (100000, 0.999, True),          # 100
    ]
    for n, q, want in cases:
        got = stats_mod.percentile_ok(n, q)
        if got == want:
            ok("guard n=%d q=%.3f" % (n, q), "%s (%.1f exceedances)" % (got, n * (1 - q)))
        else:
            bad("guard n=%d q=%.3f" % (n, q), "got %s want %s" % (got, want))

    g = stats_mod.percentile_guard(34799)
    admitted = [r["q"] for r in g["quantiles"] if r["ok"]]
    if admitted == [0.90, 0.95, 0.99]:
        ok("clean scope admits p90/p95/p99, suppresses p99.9", "N=34799")
    else:
        bad("clean-scope guard", repr(admitted))

    # the refusal that our own receipts need
    r = stats_mod.guard_pooled_percentiles([{"window_id": "a", "mean": 1.0, "p95": 2.0}])
    if not r["available"]:
        ok("FIRE: pooled percentile refused from per-window summaries",
           r["reason"][:50] + "...")
    else:
        bad("pooled percentile guard", "did not refuse")


# ======================================================================== 8
def t_ngram() -> None:
    section("8. 13-GRAM CALIBRATION-OVERLAP SCANNER")

    # ---- synthetic corpus, answer exact by construction ---------------
    # calibration = 1000 distinct ids.  final = 100 fresh ids, then a verbatim
    # 30-token span lifted out of calibration, then 100 more fresh ids.
    # A 13-gram lies inside the planted span iff it starts at one of its first
    # 30-13+1 = 18 offsets, so exactly 18 grams are shared.  The final window
    # has 230 tokens -> 230-13+1 = 218 grams, all distinct (all ids distinct).
    cal = list(range(1000, 2000))
    final = list(range(5000, 5100)) + cal[500:530] + list(range(6000, 6100))
    cg = ngram_mod.token_ngrams(cal, 13)
    fg = ngram_mod.token_ngrams(final, 13)
    hits = len(fg & cg)
    if len(fg) == 218 and hits == 18:
        ok("planted overlap: 18 shared of 218 grams", "fraction %.6f" % (18 / 218))
    else:
        bad("planted overlap", "grams=%d hits=%d (want 218/18)" % (len(fg), hits))

    res = ngram_mod.scan(
        [{"window_id": "planted", "domain": "d", "document_id": "F", "tokens": final},
         {"window_id": "clean", "domain": "d", "document_id": "G",
          "tokens": list(range(9000, 9230))}],
        cg, {"C"}, n=13, threshold=0.05)
    got = {w["window_id"]: w["shared_ngram_fraction"] for w in res["per_window"]}
    if abs(got["planted"] - round(18 / 218, 6)) < 1e-9 and got["clean"] == 0.0:
        ok("scan flags the planted window and clears the clean one",
           "planted %.6f / clean %.6f" % (got["planted"], got["clean"]))
    else:
        bad("synthetic scan", repr(got))
    if [e["window_id"] for e in res["excluded_windows"]] == ["planted"]:
        ok("threshold 0.05 excludes exactly the planted window", "")
    else:
        bad("synthetic exclusion", repr(res["excluded_windows"]))

    # ---- deduplicated denominator: the easy thing to get wrong --------
    rep = list(range(100, 120)) * 20            # 400 tokens, heavy repetition
    g = ngram_mod.token_ngrams(rep, 13)
    if len(g) == 20 and len(rep) - 13 + 1 == 388:
        ok("deduplicated denominator", "20 distinct grams from 388 slices")
    else:
        bad("dedup denominator", "%d distinct from %d slices" % (len(g), len(rep) - 12))

    # ---- document-level check catches what n-grams would miss ---------
    res2 = ngram_mod.scan(
        [{"window_id": "docdup", "domain": "d", "document_id": "C",
          "tokens": list(range(9000, 9230))}],
        cg, {"C"}, n=13, threshold=0.05)
    if res2["excluded_windows"] and "document" in res2["excluded_windows"][0]["reason"]:
        ok("document-id check excludes a 0%-ngram window", "belt and braces")
    else:
        bad("document-id check", repr(res2["excluded_windows"]))

    # ---- ORACLE: his token_ngrams on the same input -------------------
    st = oracle_mod.probe()
    o = oracle_mod.token_ngrams_via_kld_eval(final, 13)
    if o is None:
        skip("ORACLE kld_eval.token_ngrams", st["reason"] or "not importable")
    elif o == fg:
        ok("ORACLE: kld_eval.token_ngrams produces the identical digest set",
           "%d grams" % len(o))
    else:
        bad("oracle token_ngrams", "%d vs %d grams, %d common"
            % (len(o), len(fg), len(o & fg)))

    # ---- REAL PANEL: reproduce his published window_selection.json ----
    panel = os.environ.get("JOINTSTD_PANEL")
    arrays = os.environ.get("JOINTSTD_ARRAYS")
    expect = os.environ.get("JOINTSTD_EXPECT")
    if panel and arrays and os.path.isdir(arrays):
        rc = subprocess.call(
            [sys.executable, os.path.join(HERE, "joint_standard.py"), "overlap-scan",
             "--panel", panel, "--arrays", arrays] +
            (["--expect", expect] if expect else []) +
            ["--out", os.path.join(tempfile.gettempdir(), "js-scan-selftest.json")],
            stdout=subprocess.DEVNULL)
        if rc == 0:
            ok("REAL PANEL scan runs and cross-checks clean", "rc=0")
        else:
            bad("real panel scan", "rc=%d" % rc)
    else:
        skip("REAL PANEL 13-gram scan",
             "set JOINTSTD_PANEL and JOINTSTD_ARRAYS (665 files, 5.5 MB)")

    # the committed selection file IS the recorded result of that scan
    if os.path.exists(SELECTION):
        d = json.load(open(SELECTION))
        cc = d.get("cross_check", {})
        if (len(d["selected_windows"]) == 17 and len(d["excluded_windows"]) == 8
                and cc.get("identical")):
            ok("committed selection: 17 kept / 8 dropped, matched his file",
               "%d windows cross-checked" % cc.get("windows_checked", 0))
        else:
            bad("committed selection", "kept=%d dropped=%d cross_check=%s"
                % (len(d["selected_windows"]), len(d["excluded_windows"]), cc))
        drops = {e["window_id"] for e in d["excluded_windows"]}
        if {"final-0021", "final-0022"} <= drops:
            ok("the two NON-axis4 exclusions are present",
               "final-0021 (7.1%) and final-0022 (5.8%) -- a whole-axis4 drop is NOT his scope")
        else:
            bad("non-axis4 exclusions", repr(sorted(drops)))
    else:
        skip("committed selection file", "not present")


# ======================================================================== 9
def t_canary() -> None:
    section("9. R0 CANARY -- MUST PASS ON TRUTH AND FIRE ON A LIE")
    try:
        import numpy as np
    except Exception as exc:
        skip("R0 canary", "numpy absent (%r)" % exc)
        return

    rng = np.random.default_rng(20260829)
    t = rng.normal(0.0, 4.0, size=(256, 2048)).astype(np.float32)

    # PASS on a real-shaped teacher
    try:
        r = canary_mod.run_r0(t, vocab_limit=2024, shift_ratio_min=3.0, tag="synthetic")
        ok("R0 passes on a well-formed teacher",
           "self-KLD 0.0 both scopes; shift %.3f nats = %.1fx entropy"
           % (r["shift"]["mean_kld_nats"], r["shift"]["ratio_to_entropy"]))
    except canary_mod.CanaryFailure as exc:
        bad("R0 on a good teacher", str(exc)[:70])

    # FIRE 1 -- R0-a: student is the teacher with ONE logit nudged
    s = t.copy()
    s[7, 11] += np.float32(1e-3)
    try:
        canary_mod.run_r0_on_pair(t, s, vocab_limit=2024, tag="one-logit-nudged")
        bad("FIRE R0-a: one nudged logit", "canary did NOT fire")
    except canary_mod.CanaryFailure as exc:
        ok("FIRE R0-a: one nudged logit out of 524288", str(exc)[:56] + "...")

    # FIRE 2 -- R0-a: the classic off-by-one, teacher vs teacher-shifted
    try:
        canary_mod.run_r0_on_pair(t[1:], t[:-1], vocab_limit=2024, tag="off-by-one")
        bad("FIRE R0-a: off-by-one pair", "canary did NOT fire")
    except canary_mod.CanaryFailure as exc:
        ok("FIRE R0-a: deliberately misaligned pair", str(exc)[:56] + "...")

    # FIRE 3 -- R0-b, the half his harness only unit-tests.  A degenerate
    # teacher whose rows barely differ (what a left-on prefix cache looks like)
    # passes R0-a at exactly 0.0 and is still broken.  R0-b is what catches it.
    base = rng.normal(0.0, 4.0, size=(1, 2048)).astype(np.float32)
    degen = np.repeat(base, 256, axis=0)
    degen = degen + (np.arange(256, dtype=np.float32)[:, None] * np.float32(1e-6))
    self_ok = np.all(canary_mod.kld_rows(degen, degen, 2024) == 0.0)
    try:
        canary_mod.run_r0(degen, vocab_limit=2024, shift_ratio_min=3.0, tag="degenerate")
        bad("FIRE R0-b: degenerate teacher", "canary did NOT fire")
    except canary_mod.CanaryFailure as exc:
        if self_ok and "R0-b" in str(exc):
            ok("FIRE R0-b: near-constant rows pass R0-a at 0.0 and still fail",
               str(exc)[:52] + "...")
        else:
            bad("FIRE R0-b", "fired for the wrong reason: %s" % str(exc)[:60])

    # FIRE 4 -- the alignment band
    try:
        canary_mod.run_r0(t, vocab_limit=2024, shift_ratio_min=3.0, tag="band",
                          realized_tokens=[0] * 255, alignment_band=(0.20, 0.995))
        bad("FIRE R0-c: alignment band", "canary did NOT fire")
    except canary_mod.CanaryFailure as exc:
        ok("FIRE R0-c: teacher-top1 agreement outside the band", str(exc)[:56] + "...")

    # masking must not perturb the self-KLD identity
    padded = np.concatenate([t, rng.normal(0, 4, size=(256, 24)).astype(np.float32)], axis=1)
    d_mask = canary_mod.kld_rows(padded, padded, 2048)
    d_full = canary_mod.kld_rows(padded, padded, None)
    if np.all(d_mask == 0.0) and np.all(d_full == 0.0):
        ok("self-KLD is exactly 0.0 in BOTH scopes with padded columns present", "")
    else:
        bad("padded-scope self-KLD", "masked %s full %s"
            % (float(np.max(d_mask)), float(np.max(d_full))))

    # ---- ORACLE: his score_window on the same pair --------------------
    st = oracle_mod.probe()
    tok = np.zeros(t.shape[0] + 1, dtype=np.int64)
    o = oracle_mod.score_window_via_kld_eval(t, s, tok, 2024)
    if o is None:
        skip("ORACLE kld_eval.score_window", st["reason"] or "not importable")
    else:
        ours = canary_mod.kld_rows(t, s, 2024)
        d = float(np.max(np.abs(ours - o.kld)))
        if d < 1e-12:
            ok("ORACLE: kld_eval.score_window == our fp64 kernel", "max|d|=%.2e" % d)
        else:
            bad("oracle score_window", "max|d|=%.2e" % d)

    # ---- REAL TEACHER WINDOW ------------------------------------------
    tw = os.environ.get("JOINTSTD_TEACHER")
    if tw and os.path.exists(tw):
        rc = subprocess.call(
            [sys.executable, os.path.join(HERE, "joint_standard.py"), "canary",
             "--teacher", tw, "--rows-limit", "256",
             "--out", os.path.join(tempfile.gettempdir(), "js-canary-selftest.json")],
            stdout=subprocess.DEVNULL)
        if rc == 0:
            r = json.load(open(os.path.join(tempfile.gettempdir(),
                                            "js-canary-selftest.json")))
            ok("REAL teacher window passes R0",
               "shift %.3f nats vs entropy %.3f (%.1fx)"
               % (r["shift"]["mean_kld_nats"], r["teacher_mean_entropy_nats"],
                  r["shift"]["ratio_to_entropy"]))
        else:
            bad("real teacher R0", "rc=%d" % rc)
    else:
        skip("REAL teacher window R0", "set JOINTSTD_TEACHER to a 1.27 GB window")


# ======================================================================= 10
def t_cli() -> None:
    section("10. THE CLI: EVERY EMITTED RECEIPT CARRIES THE PROTOCOL STAMP")
    proto = protocol_mod.load()
    tmp = tempfile.mkdtemp(prefix="js-selftest-")
    per = os.path.join(ROOT, "registry/protocol/per-window/k6-sealed.json")
    per8 = os.path.join(ROOT, "registry/protocol/per-window/k8-streaming.json")
    runs = [
        ("protocol", ["protocol"]),
        ("mcnemar", ["mcnemar", "--a-only", "1629", "--b-only", "963"]),
    ]
    if os.path.exists(per):
        runs.append(("analyze", ["analyze", "--report", per, "--scope-file", SELECTION,
                                 "--scope", "selected", "--bootstrap-b", "400",
                                 "--domain-bootstrap-b", "200"]))
    if os.path.exists(per) and os.path.exists(per8):
        runs.append(("paired", ["paired", "--a", per, "--b", per8,
                                "--label-a", "K6", "--label-b", "K8",
                                "--scope-file", SELECTION, "--scope", "selected",
                                "--bootstrap-b", "400"]))
    for name, argv in runs:
        out = os.path.join(tmp, name + ".json")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "joint_standard.py")]
                             + argv + ["--out", out],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            bad("verb %s" % name, "rc=%d" % rc)
            continue
        d = json.load(open(out))
        try:
            protocol_mod.require_stamp(d, proto)
        except protocol_mod.ProtocolError as exc:
            bad("verb %s stamp" % name, str(exc))
            continue
        if d.get("not_submittable") is True:
            ok("verb %s: stamped and marked not_submittable" % name,
               "scoring %s" % d["protocol_scoring_sha256"][:12])
        else:
            bad("verb %s" % name, "missing not_submittable")

    # FIRE: analyze must refuse a scope the receipt cannot cover
    if os.path.exists(per):
        partial = os.path.join(tmp, "partial.json")
        d = json.load(open(per))
        d["per_window"] = d["per_window"][:5]
        json.dump(d, open(partial, "w"))
        rc = subprocess.call([sys.executable, os.path.join(HERE, "joint_standard.py"),
                              "analyze", "--report", partial, "--scope-file", SELECTION,
                              "--scope", "selected", "--out", os.path.join(tmp, "x.json")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc == 3:
            ok("FIRE: analyze refuses a scope the receipt cannot cover", "exit 3")
        else:
            bad("scope refusal", "rc=%d expected 3" % rc)

    # FIRE: analyze must refuse a scope that MIXES WINDOW SIZES.
    #
    # This is the check whose absence made the stats module's docstring a lie:
    # window_block_bootstrap sees only means, so its point estimate is the
    # equal-weight mean, while se_from_window_summaries weights by token count.
    # On unequal windows the receipt would carry a BCa interval around a
    # different number than its own headline. Numbers below are chosen so the
    # two means disagree by ~6x, which no tolerance could hide.
    if os.path.exists(per):
        uneq = os.path.join(tmp, "unequal.json")
        d = json.load(open(per))
        base = d["per_window"][0]
        d["per_window"] = [
            dict(base, window_id="w-big", count=10000, mean=0.010),
            dict(base, window_id="w-s1", count=100, mean=0.100),
            dict(base, window_id="w-s2", count=100, mean=0.100),
        ]
        json.dump(d, open(uneq, "w"))
        argv = [sys.executable, os.path.join(HERE, "joint_standard.py"), "analyze",
                "--report", uneq, "--bootstrap-b", "200",
                "--domain-bootstrap-b", "100", "--out", os.path.join(tmp, "u.json")]
        p = subprocess.run(argv, capture_output=True)
        if p.returncode == 3 and b"REFUSED" in p.stderr and b"mixes window sizes" in p.stderr:
            ok("FIRE: analyze refuses a scope that mixes window sizes",
               "exit 3, both means named")
        else:
            bad("unequal-window refusal", "rc=%d stderr=%s"
                % (p.returncode, p.stderr.decode()[:120]))
        # and the override must restore the capability, not just silence it
        p2 = subprocess.run(argv + ["--allow-unequal-windows"], capture_output=True)
        if p2.returncode == 0:
            u = json.load(open(os.path.join(tmp, "u.json")))
            if u["scope"]["equal_window_sizes"] is False:
                ok("--allow-unequal-windows runs and records equal_window_sizes=false",
                   "positions_per_window=%s" % (u["scope"]["positions_per_window"],))
            else:
                bad("unequal override", "scope did not record the inequality")
        else:
            bad("unequal override", "rc=%d" % p2.returncode)
        # equal windows must be untouched by all of this
        p3 = subprocess.run(
            [sys.executable, os.path.join(HERE, "joint_standard.py"), "analyze",
             "--report", per, "--bootstrap-b", "200", "--domain-bootstrap-b", "100",
             "--out", os.path.join(tmp, "eq.json")], capture_output=True)
        if p3.returncode == 0:
            e = json.load(open(os.path.join(tmp, "eq.json")))
            if e["scope"]["equal_window_sizes"] is True:
                ok("equal-window analyze is unaffected by the new guard",
                   "%d windows x %s positions"
                   % (e["scope"]["n_windows"], e["scope"]["positions_per_window"]))
            else:
                bad("equal-window regression", "guard changed the equal case")
        else:
            bad("equal-window analyze", "rc=%d" % p3.returncode)

    # FIRE: stamp must refuse to rewrite a receipt in place
    src = os.path.join(tmp, "r.json")
    json.dump({"schema": "x"}, open(src, "w"))
    rc = subprocess.call([sys.executable, os.path.join(HERE, "joint_standard.py"),
                          "stamp", "--in", src, "--out", src],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 3:
        ok("FIRE: stamp refuses an in-place rewrite without --in-place", "exit 3")
    else:
        bad("stamp in-place refusal", "rc=%d expected 3" % rc)

    # FIRE: a missing protocol file is a refusal, not a traceback
    env = dict(os.environ, JOINTSTD_PROTOCOL=os.path.join(tmp, "nope.json"))
    p = subprocess.run([sys.executable, os.path.join(HERE, "joint_standard.py"),
                        "protocol"], env=env, capture_output=True)
    if p.returncode == 3 and b"PROTOCOL REFUSAL" in p.stderr and b"Traceback" not in p.stderr:
        ok("FIRE: missing protocol file refuses cleanly", "exit 3, no traceback")
    else:
        bad("missing protocol", "rc=%d" % p.returncode)


# ======================================================================= 11
def t_cli_refusals() -> None:
    section("10b. THE CLI REFUSES WHAT IT CANNOT HONESTLY ANSWER")
    tmp = tempfile.mkdtemp(prefix="js-refuse-")
    per = os.path.join(ROOT, "registry/protocol/per-window/k6-sealed.json")
    per8 = os.path.join(ROOT, "registry/protocol/per-window/k8-streaming.json")
    cli = os.path.join(ROOT, "bin", "joint_standard.py")

    def run(argv):
        r = subprocess.run([sys.executable, cli] + argv, capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    # ---- STAT-07: a report that declares no window sizes must not get an invented one
    rows = json.load(open(per))["per_window"]
    nocount = os.path.join(tmp, "nocount.json")
    json.dump({"per_window": [{"window_id": w["window_id"], "domain": w["domain"],
                               "mean_kld": w["mean"]} for w in rows]},
              open(nocount, "w"))
    out = os.path.join(tmp, "a.json")
    rc, _ = run(["analyze", "--report", nocount, "--out", out, "--bootstrap-b", "200",
                 "--domain-bootstrap-b", "100"])
    if rc == 0 and os.path.exists(out):
        d = json.load(open(out))
        sc, pg = d["scope"], d.get("percentile_guard") or {}
        if (sc.get("scored_positions") is None
                and sc.get("window_sizes_declared") is False
                and pg.get("available") is False):
            ok("STAT-07: undeclared window sizes are null, not 2047 each",
               "scored_positions None, guard refuses "
               "(pre-fix: 51175 fabricated, guard answered on it)")
        else:
            bad("STAT-07: undeclared sizes",
                "scored_positions=%r declared=%r guard=%r"
                % (sc.get("scored_positions"), sc.get("window_sizes_declared"),
                   pg.get("available")))
    else:
        bad("STAT-07: undeclared sizes", "analyze exited %d" % rc)

    # a half-declared panel is neither weighting and must refuse outright
    mixed = os.path.join(tmp, "mixed.json")
    pw = [{"window_id": w["window_id"], "domain": w["domain"], "mean_kld": w["mean"]}
          for w in rows]
    for w in pw[:3]:
        w["count"] = 2047
    json.dump({"per_window": pw}, open(mixed, "w"))
    rc, txt = run(["analyze", "--report", mixed, "--out", os.path.join(tmp, "m.json"),
                   "--bootstrap-b", "200", "--domain-bootstrap-b", "100"])
    if rc != 0 and "declare a scored-position count" in txt:
        ok("STAT-07: a half-declared panel refuses", "exit %d" % rc)
    else:
        bad("STAT-07: half-declared panel", "exit %d" % rc)

    # ---- CLI-20: pairing two reports that cover different windows
    if os.path.exists(per8):
        short = os.path.join(tmp, "short.json")
        json.dump({"per_window": json.load(open(per8))["per_window"][:3]},
                  open(short, "w"))
        rc, txt = run(["paired", "--a", per, "--b", short, "--out",
                       os.path.join(tmp, "p.json"), "--bootstrap-b", "200"])
        if rc != 0 and "do not cover the same windows" in txt:
            ok("CLI-20: pairing 25 windows against 3 refuses",
               "exit %d (pre-fix: exit 0, ranked over the 3 that survived)" % rc)
        else:
            bad("CLI-20: partial pairing", "exit %d" % rc)
        # and with the flag it proceeds, but says what it dropped
        pout = os.path.join(tmp, "p2.json")
        rc, _ = run(["paired", "--a", per, "--b", short, "--allow-partial",
                     "--out", pout, "--bootstrap-b", "200"])
        if rc == 0 and os.path.exists(pout):
            d = json.load(open(pout))
            if d.get("scope_complete") is False and len(d.get("windows_dropped_a") or []) == 22:
                ok("CLI-20: --allow-partial ranks and DISCLOSES the 22 dropped windows",
                   "scope_complete false")
            else:
                bad("CLI-20: --allow-partial disclosure",
                    "complete=%r dropped_a=%r" % (d.get("scope_complete"),
                                                  len(d.get("windows_dropped_a") or [])))
        else:
            bad("CLI-20: --allow-partial", "exit %d" % rc)

    # ---- CLI-07: an overlap scan of ZERO calibration windows is not evidence
    panel = {"windows": [
        {"window_id": "cal-0", "role": "calibration", "document_id": "c0"},
        {"window_id": "final-0", "role": "final", "document_id": "f0"},
        {"window_id": "final-1", "role": "final", "document_id": "f1"}]}
    ppath = os.path.join(tmp, "panel.json")
    json.dump(panel, open(ppath, "w"))
    arrays = os.path.join(tmp, "arrays")
    os.makedirs(arrays)
    try:
        import numpy as _np
        for wid, toks in (("final-0", range(0, 60)), ("final-1", range(500, 700))):
            _np.save(os.path.join(arrays, "%s.tokens.npy" % wid),
                     _np.asarray(list(toks), dtype="<i8"))
        rc, txt = run(["overlap-scan", "--panel", ppath, "--arrays", arrays,
                       "--out", os.path.join(tmp, "sel.json")])
        if rc != 0 and "ZERO calibration windows" in txt:
            ok("CLI-07: a scan with no readable calibration arrays refuses",
               "exit %d (pre-fix: exit 0, every window SELECTED as clean)" % rc)
        else:
            bad("CLI-07: zero-calibration scan",
                "exit %d -- %s" % (rc, txt.strip()[:80]))
    except ImportError:
        skip("CLI-07: zero-calibration scan", "numpy absent")


def t_registry_joint_check() -> None:
    section("11. THE REGISTRY'S JOINT INVARIANTS MUST FIRE ON TAMPERED DATA")
    reg = os.path.join(ROOT, "registry")
    checker = os.path.join(reg, "tools", "registry_joint_check.py")
    if not os.path.exists(checker):
        skip("registry joint check", "tools/registry_joint_check.py absent")
        return
    p = subprocess.run([sys.executable, checker, "--root", reg], capture_output=True)
    line = p.stdout.decode().strip().splitlines()[-1] if p.stdout else ""
    if p.returncode == 0:
        ok("committed registry passes every joint invariant", line)
    else:
        bad("committed registry", p.stdout.decode()[-300:])
        return

    # Every tamper below is a real mistake somebody could make, and each must be
    # caught by a NAMED invariant rather than by a schema type error.
    import shutil

    tampers = [
        ("JOINT-002 se_total no longer equals hypot(se_clustered, sigma_run)",
         lambda r: r["uncertainty"].__setitem__("se_total", r["uncertainty"]["se_total"] * 1.5)),
        ("JOINT-001 the interval no longer brackets the point estimate",
         lambda r: r["uncertainty"].__setitem__("ci95_low", r["metric"]["value"] * 1.01)),
        ("JOINT-006 a per-domain row's positions were edited",
         lambda r: r["by_domain"][0].__setitem__("scored_positions",
                                                 r["by_domain"][0]["scored_positions"] + 1)),
        ("JOINT-006 the headline value was edited away from its window mean",
         lambda r: r["metric"].__setitem__("value", r["metric"]["value"] * 1.000001)),
        ("JOINT-008 the protocol stamp points at a different protocol",
         lambda r: r["protocol"].__setitem__("scoring_sha256", "b" * 64)),
        ("JOINT-003 sigma_run claims more runs than were made",
         lambda r: r["uncertainty"].__setitem__("sigma_run_runs",
                                                r["determinism"]["run_count"] + 1)),
        ("JOINT-007 the masking policy was removed",
         lambda r: r["estimator"].pop("vocab_masking_policy", None)),
        ("JOINT-005 the scope selection digest was edited",
         lambda r: r["measurement_scope"].__setitem__("scope_selection_sha256", "c" * 64)),
    ]
    target = "measurement--glm53.k6-6bpw.brandonmusic-final25"
    for label, mutate in tampers:
        tmp = tempfile.mkdtemp(prefix="js-reg-")
        try:
            shutil.copytree(reg, os.path.join(tmp, "registry"),
                            ignore=shutil.ignore_patterns("__pycache__", ".git"))
            mpath = os.path.join(tmp, "registry", "data", "measurements.jsonl")
            rows = [json.loads(x) for x in open(mpath, encoding="utf-8") if x.strip()]
            hit = False
            for r in rows:
                if r["id"] == target:
                    mutate(r)
                    hit = True
            if not hit:
                skip(label, "target row %s absent" % target)
                continue
            with open(mpath, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
            q = subprocess.run([sys.executable, checker, "--root",
                                os.path.join(tmp, "registry")], capture_output=True)
            code = label.split()[0]
            out = q.stdout.decode()
            if q.returncode == 1 and code in out:
                first = [l for l in out.splitlines() if "ERROR" in l and code in l][0]
                ok("FIRE: " + label[:52], first.strip()[:66])
            else:
                bad("FIRE: " + label, "rc=%d, %s not raised" % (q.returncode, code))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ======================================================================= 12
def t_no_network() -> None:
    section("12. THE PACKAGE IMPORTS NO NETWORKING LIBRARY")
    src = []
    for base, _, files in os.walk(os.path.join(HERE, "jointstd")):
        for f in files:
            if f.endswith(".py"):
                src.append(os.path.join(base, f))
    src.append(os.path.join(HERE, "joint_standard.py"))
    banned = ("urllib", "requests", "httpx", "socket", "http.client", "aiohttp",
              "huggingface_hub")
    hits = []
    for p in src:
        text = open(p, "r", encoding="utf-8").read()
        for b in banned:
            for line in text.splitlines():
                ls = line.strip()
                if ls.startswith(("import %s" % b, "from %s " % b)):
                    hits.append("%s: %s" % (os.path.basename(p), ls))
    if not hits:
        ok("no networking import in %d files" % len(src), ", ".join(banned[:4]) + ", ...")
    else:
        bad("networking import", "; ".join(hits))


def main() -> int:
    print("joint-standard selftest -- %s" % ROOT)
    st = oracle_mod.probe()
    print("oracle (brandonmusic kld_eval): %s%s" % (
        "AVAILABLE " + json.dumps(st.get("modules", {})) if st["available"]
        else "not importable", "" if st["available"] else " -- %s" % st["reason"]))
    for fn in (t_protocol, t_chi2_and_mcnemar, t_stats_refusals, t_clustered_se, t_bootstrap,
               t_paired, t_sigma_run, t_percentile_guard, t_ngram, t_canary,
               t_cli, t_cli_refusals, t_registry_joint_check, t_no_network):
        fn()
    print()
    print("-" * 78)
    print("%d passed, %d failed, %d skipped" % (PASS, FAIL, SKIP))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
