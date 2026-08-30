#!/usr/bin/env python3
"""Measured coverage of this registry's published intervals.

WHAT THIS ANSWERS
-----------------
Every interval in this registry was labelled 95%.  That number is a property of
a method's asymptotics, and this panel is not asymptotic: a domain has 5 to 7
windows.  STAT-01 found that the 42 per-domain intervals actually cover 78-83%,
and undercover in the harmful direction -- truth lands ABOVE the interval far
more often than below, so the intervals understate divergence and manufacture
separations between domains that are not there.

Rather than assert a corrected number, this simulates it.

DESIGN
------
For each published cell we take that cell's OWN window means, fit a lognormal on
their logs, and declare that population's mean to be TRUTH.  Then `reps` fresh
panels of g iid windows are drawn from it, each is run through the interval
procedure, and we count how often TRUTH lands inside.  The lognormal is a choice
and it is stated: it is the shape a per-window KL distribution actually has, it
is what STAT-01 used, and the deficit is not an artifact of it -- a normal
population at g=7 measures 87.2%, so no population shape rescues small-g BCa.

Both procedures see the SAME simulated panels.  We are measuring two procedures
on one set of draws, not two seeds.

REPRODUCIBILITY
---------------
Every stream is seeded from `--seed` and a stable digest of the cell and method
names.  `hash()` is NOT used anywhere: python randomises string hashing per
process, so a simulation seeded from it is unreproducible by construction, which
would make a published `coverage_measured` an anecdote.

Needs numpy, like `make reseed` and for the same reason: the resample stream has
to be PCG64.  Everything else in registry/ stays stock-python3.9.

Usage:
    python3 tools/coverage_sim.py --reps 4000 --out protocol/coverage/domain-interval-coverage.v1.json
    python3 tools/coverage_sim.py --check protocol/coverage/domain-interval-coverage.v1.json --reps 200
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REGISTRY = os.path.dirname(_HERE)
_REPO = os.path.dirname(_REGISTRY)
sys.path.insert(0, os.path.join(_REPO, "bin"))

from jointstd import chi2  # noqa: E402
from jointstd import stats as _stats  # noqa: E402

PER_WINDOW_DIR = os.path.join(_REGISTRY, "protocol", "per-window")
SELECTION_FILE = os.path.join(_REGISTRY, "protocol",
                              "window-selection.brandonmusic-final25.json")

SERIES = ("k6-sealed", "k6-streaming", "k8-streaming", "fp8-crossstack",
          "bf16-floor-crossstack", "brandonmusic-4bpw")

SCHEMA = "quant-fidelity-registry/interval-coverage.v1"
POPULATION = "lognormal fitted by moments to this cell's own window means"


def _substream(seed, *labels):
    """A seed derived from `seed` and a STABLE digest of the labels."""
    h = hashlib.sha256("\x1f".join(labels).encode("utf-8")).digest()[:8]
    return (int(seed) ^ int.from_bytes(h, "big")) & 0x7FFFFFFFFFFFFFFF


# --------------------------------------------------------------- the two procedures
def _bca(x, b, rng, np, alpha=0.05):
    """BCa on the raw mean: jointstd.stats.window_block_bootstrap's numpy path,
    vectorised over B.

    Asserted bit-identical to the shipped function on all 42 real cells by
    registry/tools/selftest_stat01_reseed.py; a coverage number produced by a
    lookalike of the code under test would measure the lookalike.

    `statistics.fmean`, NOT `x.mean()`.  They differ in the last ULP -- fmean is
    exactly rounded, numpy's is pairwise -- and that ULP is not cosmetic here:
    a small-g bootstrap of the mean lands EXACTLY on the observed value on ~0.5%
    of resamples, so which side of the strict `<` those replicates fall on moves
    the bias correction z0 (measured on axis4_reasoning_termination: prop 0.484
    vs 0.493, and a visibly different endpoint). That sensitivity is itself part
    of why BCa is the wrong interval at g=6."""
    g = len(x)
    observed = statistics.fmean(x.tolist())
    boots = x[rng.integers(0, g, size=(b, g))].mean(axis=1)
    prop = float(np.clip((boots < observed).mean(), 1.0 / (b + 1), 1.0 - 1.0 / (b + 1)))
    z0 = chi2.norm_ppf(prop)
    jack = np.array([np.delete(x, k).mean() for k in range(g)])
    jbar = jack.mean()
    num = float(np.sum((jbar - jack) ** 3))
    den = 6.0 * float(np.sum((jbar - jack) ** 2)) ** 1.5
    a = num / den if den > 0 else 0.0
    out = []
    for z in (chi2.norm_ppf(alpha / 2.0), chi2.norm_ppf(1 - alpha / 2.0)):
        w = z0 + z
        q = min(max(chi2.norm_cdf(z0 + w / (1 - a * w)), 0.0), 1.0)
        out.append(float(np.quantile(boots, q)))
    return out[0], out[1]


def _t_log(x, b, rng, np, alpha=0.05):
    """Bootstrap-t on log(mean), delta-method SE.  jointstd.stats.bootstrap_t_log,
    vectorised over B."""
    g = len(x)
    xl = x.tolist()
    # Both the mean AND the sd come from `statistics`, not numpy: the shipped
    # estimator uses statistics.fmean/statistics.stdev, and numpy's pairwise
    # reductions differ from them in the last ULP -- enough to move a published
    # endpoint by 1 ULP, which is enough for an exact-equality selftest to be
    # worth something.
    m = statistics.fmean(xl)
    se_log = (statistics.stdev(xl) / math.sqrt(g)) / m
    xb = x[rng.integers(0, g, size=(b, g))]
    mb = xb.mean(axis=1)
    seb = xb.std(axis=1, ddof=1) / math.sqrt(g)
    good = (mb > 0) & (seb > 0)
    t = (np.log(mb[good]) - math.log(m)) / (seb[good] / mb[good])
    tlo, thi = np.quantile(t, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (math.exp(math.log(m) - float(thi) * se_log),
            math.exp(math.log(m) - float(tlo) * se_log))


def _delta_t_log(x, b, rng, np, alpha=0.05):
    """Student-t on log(mean) with the delta-method SE: jointstd.stats.delta_t_log.

    No resample stream at all, so `b` and `rng` are ignored -- which is the point:
    an interval with no Monte-Carlo error cannot share Monte-Carlo error with the
    domain next to it (STAT-17), and cannot blow up on a degenerate resample the
    way bootstrap-t does at g=5.
    """
    g = len(x)
    xl = x.tolist()
    m = statistics.fmean(xl)
    se_log = (statistics.stdev(xl) / math.sqrt(g)) / m
    t = chi2.student_t_ppf(1.0 - alpha / 2.0, g - 1)
    return math.exp(math.log(m) - t * se_log), math.exp(math.log(m) + t * se_log)


def _t_raw(x, b, rng, np, alpha=0.05):
    """Bootstrap-t on the RAW mean.  Not a candidate -- measured only so the
    reason for rejecting it (negative lower endpoints on a KL divergence) is a
    counted number rather than a preference."""
    g = len(x)
    xl = x.tolist()
    m = statistics.fmean(xl)
    se = statistics.stdev(xl) / math.sqrt(g)
    xb = x[rng.integers(0, g, size=(b, g))]
    mb = xb.mean(axis=1)
    seb = xb.std(axis=1, ddof=1) / math.sqrt(g)
    good = seb > 0
    t = (mb[good] - m) / seb[good]
    tlo, thi = np.quantile(t, [alpha / 2.0, 1.0 - alpha / 2.0])
    return m - float(thi) * se, m - float(tlo) * se


PROCEDURES = {
    "old_bca_b1000": (_bca, 1000),
    "bca_b20000": (_bca, 20000),
    "boott_log_b20000": (_t_log, 20000),
    "t_raw_b20000": (_t_raw, 20000),
    "new_delta_t_log": (_delta_t_log, 1),
    "panel_bca_b5000": (_bca, 5000),
}
DOMAIN_PROCEDURES = ("old_bca_b1000", "bca_b20000", "boott_log_b20000", "t_raw_b20000",
                     "new_delta_t_log")
PANEL_PROCEDURES = ("panel_bca_b5000",)


# ------------------------------------------------------------------------- cells
def cells():
    """Every published cell: 42 per-domain and 12 panel-level."""
    clean = set(json.load(open(SELECTION_FILE, encoding="utf-8"))["selected_windows"])
    out = []
    for slug in SERIES:
        with open(os.path.join(PER_WINDOW_DIR, slug + ".json"), encoding="utf-8") as fh:
            pw = json.load(fh)["per_window"]
        for scope, sel in (("panel25", pw),
                           ("clean17", [w for w in pw if w["window_id"] in clean])):
            out.append({"block": "panel", "series": slug, "scope": scope,
                        "domain": None,
                        "values": [float(w["mean"]) for w in sel]})
            by = {}
            for w in sel:
                by.setdefault(w["domain"], []).append(float(w["mean"]))
            for dom in sorted(by):
                if len(by[dom]) >= 2:
                    out.append({"block": "domain", "series": slug, "scope": scope,
                                "domain": dom, "values": by[dom]})
    return out


def run(reps, seed, only=None):
    import numpy as np

    result = {"schema": SCHEMA, "reps": reps, "seed": seed,
              "population": POPULATION,
              "nominal": 0.95,
              "note": ("Measured, not asserted. Each cell's own window means fix a "
                       "lognormal; TRUTH is that population's mean; `reps` fresh panels "
                       "of g windows are drawn and the interval procedure is run on "
                       "each. miss_low counts reps where TRUTH fell ABOVE the interval "
                       "-- the direction that understates divergence."),
              "cells": []}
    for c in cells():
        if only and c["block"] != only:
            continue
        v = np.asarray(c["values"], dtype=np.float64)
        g = len(v)
        logs = np.log(v)
        mu, sd = float(logs.mean()), float(logs.std(ddof=1))
        truth = math.exp(mu + sd * sd / 2.0)
        label = "%s|%s|%s" % (c["series"], c["scope"], c["domain"] or "-")
        data_rng = np.random.default_rng(_substream(seed, "data", label))
        panels = np.exp(data_rng.normal(mu, sd, size=(reps, g)))
        row = {"block": c["block"], "series": c["series"], "scope": c["scope"],
               "domain": c["domain"], "g": g, "truth": truth,
               "lognormal_mu": mu, "lognormal_sigma": sd, "procedures": {}}
        procs = PANEL_PROCEDURES if c["block"] == "panel" else DOMAIN_PROCEDURES
        for pname in procs:
            fn, b = PROCEDURES[pname]
            rng = np.random.default_rng(_substream(seed, pname, label))
            inside = miss_low = miss_high = neg = 0
            width = 0.0
            worst_hi_over_mean = 0.0
            for r in range(reps):
                lo, hi = fn(panels[r], b, rng, np)
                if lo < 0:
                    neg += 1
                width += hi - lo
                mr = float(panels[r].mean())
                if mr > 0:
                    worst_hi_over_mean = max(worst_hi_over_mean, hi / mr)
                if truth > hi:
                    miss_low += 1
                elif truth < lo:
                    miss_high += 1
                else:
                    inside += 1
            row["procedures"][pname] = {
                "b": b,
                "coverage": inside / float(reps),
                "miss_low": miss_low / float(reps),
                "miss_high": miss_high / float(reps),
                "mean_width": width / float(reps),
                "reps_with_negative_lower": neg,
                # The failure mode that coverage alone cannot see: an interval can
                # cover 92% of the time and still return an upper endpoint eighteen
                # times the estimate on a five-window stratum.
                "worst_high_over_mean": worst_hi_over_mean,
            }
        result["cells"].append(row)
        print("%-22s %-8s %-28s g=%2d  %s"
              % (c["series"], c["scope"], c["domain"] or "(panel)", g,
                 "  ".join("%s=%.1f%%" % (p, 100 * row["procedures"][p]["coverage"])
                           for p in procs)), flush=True)
    result["summary"] = summarize(result)
    return result


def summarize(result):
    out = {}
    for block in ("domain", "panel"):
        rows = [c for c in result["cells"] if c["block"] == block]
        if not rows:
            continue
        procs = sorted({p for c in rows for p in c["procedures"]})
        out[block] = {"cells": len(rows)}
        for p in procs:
            cov = [c["procedures"][p]["coverage"] for c in rows if p in c["procedures"]]
            neg = sum(1 for c in rows
                      if c["procedures"].get(p, {}).get("reps_with_negative_lower", 0) > 0)
            out[block][p] = {
                "mean_coverage": sum(cov) / len(cov),
                "min_coverage": min(cov), "max_coverage": max(cov),
                "cells_with_a_negative_lower_endpoint": neg,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--only", choices=("domain", "panel"), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, metavar="FILE",
                    help="re-run at --reps and assert the same seed reproduces the same "
                         "coverage for the cells it recomputes")
    args = ap.parse_args()

    if args.check:
        with open(args.check, encoding="utf-8") as fh:
            have = json.load(fh)
        if args.reps != have["reps"]:
            print("--check needs --reps %d to compare against %s"
                  % (have["reps"], args.check), file=sys.stderr)
            return 2
        got = run(have["reps"], have["seed"], only=args.only)
        a = {(c["series"], c["scope"], c["domain"]): c["procedures"] for c in got["cells"]}
        bad = 0
        for c in have["cells"]:
            k = (c["series"], c["scope"], c["domain"])
            if k not in a:
                continue
            for p, v in c["procedures"].items():
                if a[k][p]["coverage"] != v["coverage"]:
                    bad += 1
                    print("DRIFT %s %s: %r != %r" % (k, p, a[k][p]["coverage"], v["coverage"]))
        print("coverage --check: %d drifted" % bad)
        return 1 if bad else 0

    result = run(args.reps, args.seed, only=args.only)
    print()
    for block, s in sorted(result["summary"].items()):
        for p, v in sorted(s.items()):
            if p == "cells":
                continue
            print("%-7s %-18s mean %.1f%%  (min %.1f%%, max %.1f%%)  cells ever negative: %d"
                  % (block, p, 100 * v["mean_coverage"], 100 * v["min_coverage"],
                     100 * v["max_coverage"], v["cells_with_a_negative_lower_endpoint"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
