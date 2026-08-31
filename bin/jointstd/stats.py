"""The inference layer of the joint standard.

Everything brandonmusic's standard asks for and our GLM-5.3 publications did
not carry: cluster-robust SEs, a window-level block bootstrap with BCa
intervals, sigma_run combined with the statistical SE in quadrature, McNemar on
paired top-1, and a percentile-exceedance guard that refuses to print a
quantile it cannot support.

THREE BACKENDS, and the receipt says which one produced the number.

  ``kld_eval``  his own implementation, imported not copied, used when
                numpy+scipy+pandas are importable.  Reproduces his published
                endpoints exactly because it IS his code.
  ``numpy``     our implementation driving numpy's PCG64 with his seed and his
                draw pattern.  Also reproduces his published endpoints exactly
                (verified: 24/24 BCa and percentile endpoints on his run1
                clean and panel scopes).
  ``stdlib``    our implementation on ``random.Random``.  Agrees within Monte
                Carlo error; used on interpreters with no numpy, which is the
                only environment ``registry/ make check`` is allowed to assume.

EQUAL-WEIGHT WINDOWS, AND THE CONDITION THAT MAKES THAT LEGAL.  His bootstrap
resamples windows and concatenates their per-TOKEN arrays.  Every window in this
panel is exactly 2047 scored positions, so the token-weighted mean of a resample
equals the plain mean of the resampled window means, and a bootstrap over window
means is the same estimator.  Our published receipts carry per-window means but
not per-token arrays, so this equivalence is what makes our existing rows
analysable at all.

``window_block_bootstrap`` below takes ONLY window means: it never sees a token
count and therefore always computes the EQUAL-WEIGHT mean, while
``se_from_window_summaries`` computes the TOKEN-WEIGHTED one.  On equal windows
those coincide exactly; on unequal windows they do not, and a receipt carrying
both would quote a BCa interval around a different point estimate than its own
headline mean.  The equivalence is therefore a precondition, not a property:
``bin/joint_standard.py::cmd_analyze`` checks the window sizes and REFUSES when
they differ (``--allow-unequal-windows`` to override deliberately), and
``registry/tools/joint_enrich.py`` runs only on 2047-position panels.
"""

from __future__ import annotations

import math
import hashlib
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import chi2

MIN_EXCEEDANCES = 100
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260829
DOMAIN_BOOTSTRAP_B = 20000
# Below this many windows a BCa interval on the raw mean does not deliver its
# nominal level and is not published as though it did.  Measured, not assumed:
# registry/tools/coverage_sim.py, 4000 reps per cell against a lognormal fitted
# to each cell's own windows.
SMALL_G = 10
POSITION_BUCKETS = ((0, 256), (256, 1024), (1024, 4096), (4096, 1 << 30))


def derive_seed(seed: int, label: str) -> int:
    """A per-stratum seed that is a deterministic function of the stratum name.

    STAT-17.  Bootstrapping every domain from ONE seed gives identical resample
    index streams wherever the window counts are equal, so the domains' intervals
    share their Monte-Carlo error -- measured on the real panel at |r| up to 0.57.
    That correlation is not merely unwanted, it is arbitrary: it pairs domain A's
    k-th window with domain B's k-th, and those are unrelated windows.

    Derived rather than drawn, so the value stays reproducible from the row: the
    published cell records the seed it actually used.
    """
    h = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    return int(seed ^ (h & 0x7FFFFFFF))


# ============================================================ cluster-robust
def clustered_se(
    values: Sequence[float], clusters: Sequence[Any]
) -> Dict[str, Any]:
    """Liang-Zeger cluster-robust SE of the mean, with the finite-cluster factor.

        se = sqrt( g/(g-1) * sum_c (T_c - n_c*mean)^2 ) / N
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        raise ValueError("no values")
    mean = statistics.fmean(vals)
    naive = statistics.stdev(vals) / math.sqrt(n) if n > 1 else float("nan")
    totals: Dict[Any, float] = {}
    counts: Dict[Any, int] = {}
    for v, c in zip(vals, clusters):
        totals[c] = totals.get(c, 0.0) + v
        counts[c] = counts.get(c, 0) + 1
    g = len(totals)
    if g < 2:
        return {"se": float("nan"), "deff": float("nan"), "n_clusters": g,
                "se_naive": naive, "mean": mean, "n": n}
    ssq = math.fsum((totals[c] - counts[c] * mean) ** 2 for c in totals)
    se = math.sqrt(g / (g - 1.0) * ssq) / n
    deff = (se / naive) ** 2 if naive and naive > 0 else float("nan")
    return {"se": se, "deff": deff, "n_clusters": g, "se_naive": naive,
            "mean": mean, "n": n}


def se_from_window_summaries(
    per_window: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """His full ``summary`` block from per-window (count, mean, std) triples.

    Our published receipts carry exactly those three fields per window, so the
    naive SE, the window-clustered SE and the design effect are all recoverable
    with no per-token data.  Pooled variance is the standard decomposition

        (N-1) s^2 = sum_w (n_w - 1) s_w^2 + sum_w n_w (m_w - M)^2
    """
    # STAT-07. A window may legitimately declare no size (a third-party report that
    # carries only means). Weighting is then undefined, so rather than inventing a count
    # -- which is what produced a fabricated scope.scored_positions and a percentile
    # guard that failed open -- weight every window equally and OMIT the token-count
    # quantities. `n` is a token total; if there is no token total there is no `n`.
    unsized = any(w.get("count") is None for w in per_window)
    if unsized:
        if not all(w.get("count") is None for w in per_window):
            raise ValueError(
                "some windows declare a scored-position count and others do not; a "
                "half-declared panel has neither a token-weighted nor an equal-weight "
                "mean that can honestly be called the panel mean")
        counts = [1] * len(per_window)
    else:
        counts = [int(w["count"]) for w in per_window]
    means = [float(w["mean"]) for w in per_window]
    n = sum(counts)
    if n == 0:
        raise ValueError("empty panel")
    # math.fsum, not sum(): CPython 3.12 switched builtin sum() to Neumaier
    # compensated summation for floats, so the same data reduced by 3.9 and by
    # 3.12 differed in the last ULP -- enough to make `make reseed-check` report
    # drift on a different interpreter. fsum is exactly rounded on every version.
    grand = math.fsum(c * m for c, m in zip(counts, means)) / n
    out: Dict[str, Any] = {"n": (None if unsized else n), "mean": grand,
                           "n_clusters_window": len(counts)}
    if unsized:
        out["weighting"] = "equal_per_window (no window declares a position count)"
    if (not unsized) and all("std" in w and w["std"] is not None for w in per_window):
        within = math.fsum((c - 1) * float(w["std"]) ** 2 for c, w in zip(counts, per_window))
        between = math.fsum(c * (m - grand) ** 2 for c, m in zip(counts, means))
        var = (within + between) / (n - 1) if n > 1 else float("nan")
        out["pooled_std"] = math.sqrt(var)
        out["se_naive"] = math.sqrt(var / n)
    ssq = math.fsum((c * m - c * grand) ** 2 for c, m in zip(counts, means))
    g = len(counts)
    if g >= 2:
        se = math.sqrt(g / (g - 1.0) * ssq) / n
        out["se_clustered_window"] = se
        # STAT-19. Gated on TRUTHINESS, so a panel whose per-window stds are all
        # exactly 0.0 (se_naive == 0.0) silently dropped the key rather than saying the
        # design effect is undefined. joint_enrich then emitted se_naive and looked up
        # deff_window unconditionally -> KeyError. None serialises to JSON null, which is
        # valid RFC-8259; inf/nan would not be, and json.dumps writes them bare.
        naive = out.get("se_naive")
        if naive is not None:
            out["deff_window"] = (se / naive) ** 2 if naive > 0 else None
    return out


# ================================================================= bootstrap
def quantile_linear(sorted_values: Sequence[float], p: float) -> float:
    """numpy's default ('linear') quantile, on an already-sorted sequence.

    Matching the interpolation matters: with B=5000 the index-only quantile can
    sit a whole order statistic away from numpy's, which shows up in the fourth
    significant figure of a CI endpoint.
    """
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_values[0])
    pos = min(max(p, 0.0), 1.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _bca_endpoints(
    boots: Sequence[float], observed: float, jack: Sequence[float], alpha: float
) -> Tuple[float, float, float, float]:
    """Returns (lo, hi, z0, acceleration)."""
    b = len(boots)
    ordered = sorted(boots)
    prop = sum(1 for x in boots if x < observed) / float(b)
    # STAT-22. 1e-9 asserts a bias correction three orders finer than B resamples can
    # resolve; the finest proportion the data supports is 1/B. Endpoints do not move on
    # any reachable input (the clamp binds only at prop 0.0/1.0, and prop == 1.0 is
    # impossible for a bootstrap of the mean since E[boot] == observed), so this is a
    # statement of what the resamples support, not a change to a published number.
    prop = min(max(prop, 1.0 / (b + 1)), 1.0 - 1.0 / (b + 1))
    z0 = chi2.norm_ppf(prop)
    jbar = statistics.fmean(jack)
    num = sum((jbar - j) ** 3 for j in jack)
    den = 6.0 * (sum((jbar - j) ** 2 for j in jack)) ** 1.5
    a = num / den if den > 0 else 0.0

    out = []
    for z in (chi2.norm_ppf(alpha / 2.0), chi2.norm_ppf(1.0 - alpha / 2.0)):
        w = z0 + z
        out.append(quantile_linear(ordered, chi2.norm_cdf(z0 + w / (1.0 - a * w))))
    return out[0], out[1], z0, a


def window_block_bootstrap(
    window_means: Dict[str, float],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    backend: str = "auto",
) -> Dict[str, Any]:
    """Window-level block bootstrap of the mean, percentile and BCa intervals.

    ``window_means`` must be keyed by window id; the draw order is the sorted
    window id order, which is what his implementation uses.
    """
    wids = sorted(window_means)
    vals = [float(window_means[w]) for w in wids]
    g = len(vals)
    if g < 2:
        raise ValueError("need at least 2 windows to bootstrap")
    observed = statistics.fmean(vals)

    chosen = backend
    if backend in ("auto", "numpy"):
        try:
            import numpy as np
        except Exception:
            if backend == "numpy":
                # STAT-03. The two backends draw DIFFERENT resample index streams from the
                # same seed, so a silent fallback changes published CI endpoints by up to
                # 1.2%. A caller that pins the backend is pinning the numbers; answering
                # with the other one is worse than not answering.
                raise RuntimeError(
                    "backend='numpy' was requested and numpy is not importable. The "
                    "stdlib backend draws a different resample stream from the same "
                    "seed, so falling back would silently change the CI endpoints this "
                    "registry publishes.")
            chosen = "stdlib"
        else:
            chosen = "numpy"
    if chosen == "numpy":
        import numpy as np

        arr = np.asarray(vals, dtype=np.float64)
        rng = np.random.default_rng(seed)
        boots = np.empty(b, dtype=np.float64)
        for i in range(b):
            idx = rng.integers(0, g, size=g)
            boots[i] = arr[idx].mean()
        lo, hi = (float(x) for x in np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0]))
        jack = [float(np.delete(arr, k).mean()) for k in range(g)]
        # STAT-22, numpy path: see _bca_endpoints. 1/(B+1) is the finest resolvable
        # proportion; 1e-9 claimed precision B resamples do not have.
        prop = float(np.clip(np.mean(boots < observed),
                             1.0 / (b + 1), 1.0 - 1.0 / (b + 1)))
        z0 = chi2.norm_ppf(prop)
        jbar = float(np.mean(jack))
        jj = np.asarray(jack)
        num = float(np.sum((jbar - jj) ** 3))
        den = 6.0 * float(np.sum((jbar - jj) ** 2)) ** 1.5
        a = num / den if den > 0 else 0.0
        adj = []
        for z in (chi2.norm_ppf(alpha / 2.0), chi2.norm_ppf(1 - alpha / 2.0)):
            w = z0 + z
            adj.append(float(np.quantile(boots, min(max(chi2.norm_cdf(z0 + w / (1 - a * w)), 0.0), 1.0))))
        bca = (adj[0], adj[1])
    else:
        rnd = random.Random(seed)
        boots = [statistics.fmean(vals[rnd.randrange(g)] for _ in range(g)) for _ in range(b)]
        ordered = sorted(boots)
        lo = quantile_linear(ordered, alpha / 2.0)
        hi = quantile_linear(ordered, 1.0 - alpha / 2.0)
        jack = [statistics.fmean(vals[:k] + vals[k + 1:]) for k in range(g)]
        blo, bhi, z0, a = _bca_endpoints(boots, observed, jack, alpha)
        bca = (blo, bhi)

    return {
        "statistic": "mean_kld",
        "observed": observed,
        "b": b,
        "seed": seed,
        "n_windows": g,
        "backend": chosen,
        "ci95_percentile": [lo, hi],
        "ci95_bca": list(bca),
        "bca_z0": z0,
        "bca_acceleration": a,
        "windows": wids,
    }


def delta_t_log(
    window_means: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Student-t interval on log(mean) with the delta-method SE, exponentiated.

    THE PUBLISHED PER-DOMAIN INTERVAL, and the reason it is this and not
    ``bootstrap_t_log``, measured rather than argued:

      * coverage is the same to within Monte-Carlo error (92.0% vs 92.2% over the
        42 real cells, 4000 reps each) -- both fix STAT-01's 81.3%;
      * ``bootstrap_t_log`` blows up at g=5. On the real
        k8-8bpw-stream/clean17/axis2_legal cell -- five ordinary windows, cv 0.47,
        nothing pathological -- it returns an upper endpoint of 0.187 nats around
        a mean of 0.0103, eighteen times the estimate. That is not the data: it is
        resamples that draw four copies of one window, whose studentizing
        denominator collapses and whose t is enormous. Publishing it would replace
        an interval that is too NARROW with one that is absurd, and a reader would
        take 0.187 nats as a real possibility;
      * it has NO resample stream, so STAT-17 cannot recur here. A derived
        per-domain seed stops the strata from sharing Monte-Carlo error; having no
        Monte-Carlo error is strictly better than sharing none of it, and it also
        removes the 6.10%-at-B=1000 seed noise that made two honest reseeds
        disagree;
      * it is non-negative by construction, like the bootstrap-t on log and unlike
        the bootstrap-t on the raw mean, which puts a negative lower bound on a KL
        divergence on 5 of these 42 cells.

    It is NOT nominal. 92.0% is what it measures and 92.0% is what the row says,
    which is the actual content of the STAT-01 fix.
    """
    wids = sorted(window_means)
    vals = [float(window_means[w]) for w in wids]
    g = len(vals)
    if g < 2:
        raise ValueError("need at least 2 windows for an interval")
    m = statistics.fmean(vals)
    if m <= 0.0:
        raise ValueError("delta_t_log needs a positive mean; got %r" % m)
    se = statistics.stdev(vals) / math.sqrt(g)
    se_log = se / m
    t = chi2.student_t_ppf(1.0 - alpha / 2.0, g - 1)
    log_m = math.log(m)
    return {
        "statistic": "log_mean_kld",
        "observed": m,
        "n_windows": g,
        "df": g - 1,
        "se": se,
        "se_log": se_log,
        "t_critical": t,
        "ci95_delta_t_log": [math.exp(log_m - t * se_log), math.exp(log_m + t * se_log)],
        "windows": wids,
    }


def bootstrap_t_log(
    window_means: Dict[str, float],
    b: int = DOMAIN_BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    backend: str = "auto",
) -> Dict[str, Any]:
    """Bootstrap-t on log(mean), studentized with the delta-method SE.

    NOT the published per-domain interval -- ``delta_t_log`` is.  Kept because it
    is the right interval at moderate g, because the review that opened STAT-01
    recommended it and the record should show what it actually does, and because
    ``registry/tools/selftest_stat01_reseed.py`` uses it to demonstrate the
    failure that ruled it out: at g=5 a resample that draws four copies of one
    window has a studentizing denominator near zero and an enormous t, and the
    exponentiated upper endpoint follows it.  On the real
    k8-8bpw-stream/clean17/axis2_legal cell -- five ordinary windows, cv 0.47 --
    that produces an upper endpoint of 0.187 nats around a mean of 0.0103.

    WHY NOT BCa EITHER.  BCa is the right interval on this panel's 25 windows and
    it is what the panel block publishes.  On a DOMAIN it is not: g runs 5 to 7,
    and at that size BCa on the raw mean measures 81.3% coverage while claiming
    95% (registry/tools/coverage_sim.py, 4000 reps per cell).  Raising B does not
    touch it -- measured 81.5% at B=20000 -- because the deficit is small-g, not
    Monte Carlo.  It also fails in the harmful direction: truth lands ABOVE the
    interval far more often than below, so the intervals systematically
    understate divergence and produce false SEPARATIONS between domains.

    WHY log AND NOT THE RAW MEAN.  Bootstrap-t on the raw mean measures the same
    coverage but puts a NEGATIVE lower endpoint on every one of the 42 cells at
    least once across the simulation, and on 5 cells at their published values.
    A negative lower bound on a KL divergence is a worse publication defect than
    the undercoverage it fixes.  On the log scale the interval is non-negative by
    construction.

    Resamples that are degenerate (all g draws the same window, hence zero
    within-resample spread) carry no studentized value and are excluded from the
    t distribution rather than being given an infinite one; the count is
    reported so an unusable cell is visible instead of silently thin.
    """
    wids = sorted(window_means)
    vals = [float(window_means[w]) for w in wids]
    g = len(vals)
    if g < 2:
        raise ValueError("need at least 2 windows to bootstrap")
    observed = statistics.fmean(vals)
    if observed <= 0.0:
        raise ValueError("bootstrap_t_log needs a positive mean; got %r" % observed)
    se = statistics.stdev(vals) / math.sqrt(g)
    se_log = se / observed
    log_obs = math.log(observed)

    chosen = backend
    if backend in ("auto", "numpy"):
        try:
            import numpy as np  # noqa: F401
        except Exception:
            if backend == "numpy":
                raise RuntimeError(
                    "backend='numpy' was requested and numpy is not importable. The "
                    "stdlib backend draws a different resample stream from the same "
                    "seed, so falling back would silently change the CI endpoints this "
                    "registry publishes.")
            chosen = "stdlib"
        else:
            chosen = "numpy"

    if chosen == "numpy":
        import numpy as np

        arr = np.asarray(vals, dtype=np.float64)
        rng = np.random.default_rng(seed)
        # One (b, g) block draw, NOT b calls of size g. Verified equal: PCG64 fills
        # row-major, so the two consume the identical stream and this function stays
        # reproducible against window_block_bootstrap's draw pattern from the same
        # seed. Vectorised rather than looped because the coverage simulator must
        # measure THIS code -- a per-row `.mean()` and a `mean(axis=1)` differ in the
        # last ULP (pairwise summation over a different shape), which is enough to
        # move a t at the order statistic the quantile lands on.
        xb = arr[rng.integers(0, g, size=(b, g))]
        mb = xb.mean(axis=1)
        seb = xb.std(axis=1, ddof=1) / math.sqrt(g)
        keep = (mb > 0.0) & (seb > 0.0)
        t = (np.log(mb[keep]) - log_obs) / (seb[keep] / mb[keep])
        tlo, thi = (float(x) for x in np.quantile(t, [alpha / 2.0, 1.0 - alpha / 2.0]))
        n_used = int(keep.sum())
    else:
        rnd = random.Random(seed)
        tvals = []
        for _ in range(b):
            samp = [vals[rnd.randrange(g)] for _ in range(g)]
            mb = statistics.fmean(samp)
            seb = statistics.stdev(samp) / math.sqrt(g)
            if mb > 0.0 and seb > 0.0:
                tvals.append((math.log(mb) - log_obs) / (seb / mb))
        tvals.sort()
        tlo = quantile_linear(tvals, alpha / 2.0)
        thi = quantile_linear(tvals, 1.0 - alpha / 2.0)
        n_used = len(tvals)

    return {
        "statistic": "log_mean_kld",
        "observed": observed,
        "b": b,
        "seed": seed,
        "n_windows": g,
        "backend": chosen,
        "se_log": se_log,
        "t_low": tlo,
        "t_high": thi,
        "resamples_used": n_used,
        # thi goes to the LOW endpoint: the bootstrap-t interval is
        # [theta - t_{1-a/2} se, theta - t_{a/2} se], which is the pivot
        # reversal that makes it second-order accurate.
        "ci95_t_log": [math.exp(log_obs - thi * se_log),
                       math.exp(log_obs - tlo * se_log)],
        "windows": wids,
    }


# ================================================================= sigma_run
def sigma_run(run_means: Sequence[float]) -> Dict[str, Any]:
    """Run-to-run spread, and the honest statement of how many runs made it.

    A two-run sigma is |delta|/sqrt(2) with one degree of freedom.  It is a
    legal number and we report it, flagged, because pretending it is a 3-run
    estimate is how a live tail gets buried.
    """
    vals = [float(v) for v in run_means]
    n = len(vals)
    if n == 0:
        return {"runs": 0, "sigma_run": None, "dof": 0, "note": "no runs"}
    if n == 1:
        return {"runs": 1, "sigma_run": None, "dof": 0,
                "note": "one run: sigma_run is not estimable"}
    s = statistics.stdev(vals)
    out = {
        "runs": n,
        "sigma_run": s,
        "dof": n - 1,
        "min_run_mean": min(vals),
        "max_run_mean": max(vals),
        "spread": max(vals) - min(vals),
        "all_equal": len(set(vals)) == 1,
    }
    if n == 2:
        out["note"] = ("two-run sigma = |delta|/sqrt(2), 1 degree of freedom; "
                       "the joint protocol asks for >= 3 cold runs")
    return out


def combine_quadrature(se_stat: float, sigma: Optional[float],
                       gate: float = 0.2,
                       mean: Optional[float] = None,
                       z: float = 1.96) -> Dict[str, Any]:
    """SE_total = hypot(SE_stat, sigma_run), with the gate that decides whether
    the run term has to appear in the headline.

    AND, when ``mean`` is given and the run term is real, the interval that
    goes with it.  Without that the receipt tells its reader to quote SE_total
    and then hands them no interval built from it -- the BCa endpoints come
    from the bootstrap, which resamples windows within ONE run and therefore
    cannot see run-to-run spread at all.  A consumer following the instruction
    literally had nothing to follow.

    ``ci95_total`` is deliberately a plain z-interval, ``mean +- 1.96*SE_total``,
    and is labelled ``interval_kind: "z"``.  It is NOT a BCa interval and must
    not be presented as one: sigma_run has no bootstrap distribution to be
    bias-corrected or accelerated against, so there is nothing for BCa to
    correct.  Quote it BESIDE the BCa interval, not instead of it -- the BCa
    endpoints remain the better statement of the statistical half, and on a
    skewed panel they are visibly asymmetric where this one cannot be.

    It is emitted only when ``sigma_run > 0``.  At exactly 0.0 -- every
    bitwise-deterministic path, which is every malaiwah row published so far --
    SE_total == SE_stat and the BCa interval already IS the total interval;
    emitting a second, worse-shaped copy of it would invite someone to quote
    the z-interval when the BCa one was available.
    """
    if sigma is None:
        return {"se_stat": se_stat, "sigma_run": None, "se_total": se_stat,
                "ratio": None, "gate": gate, "gate_ok": True,
                "ci95_total": None, "interval_kind": None,
                "note": "sigma_run not estimable; SE_total = SE_stat"}
    total = math.hypot(se_stat, sigma)
    ratio = (sigma / se_stat) if se_stat else float("inf")
    out = {
        "se_stat": se_stat,
        "sigma_run": sigma,
        "se_total": total,
        "ratio": ratio,
        "gate": gate,
        "gate_ok": ratio <= gate,
        "ci95_total": None,
        "interval_kind": None,
        "note": ("run-to-run term is negligible against the statistical SE"
                 if ratio <= gate else
                 "run-to-run term is NOT negligible: quote SE_total, not SE_stat"),
    }
    if sigma > 0.0 and mean is not None:
        out["ci95_total"] = [mean - z * total, mean + z * total]
        out["interval_kind"] = "z"
        out["z"] = z
        out["note"] += ("; ci95_total = mean +- %.2f*SE_total is a z-interval, "
                        "not BCa -- quote it beside the BCa endpoints, which "
                        "remain the better statement of the statistical half"
                        % z)
    elif sigma == 0.0:
        out["note"] += ("; sigma_run is exactly 0.0, so SE_total == SE_stat and "
                        "the BCa interval already is the total interval")
    return out


# =================================================================== McNemar
def mcnemar(b01: int, b10: int, continuity: bool = True) -> Dict[str, Any]:
    """Paired top-1 test.  b01 = A right where B wrong; b10 = B right where A wrong.

    Concordant pairs carry no information and are excluded by construction.
    """
    b01, b10 = int(b01), int(b10)
    n = b01 + b10
    out: Dict[str, Any] = {
        "a_only_correct": b01,
        "b_only_correct": b10,
        "discordant": n,
        "continuity_correction": continuity,
    }
    if n == 0:
        out.update({"chi2": None, "p": 1.0, "p_exact": 1.0,
                    "note": "no discordant pairs"})
        return out
    diff = abs(b01 - b10) - (1 if continuity else 0)
    diff = max(diff, 0)
    x2 = diff * diff / float(n)
    out["chi2"] = x2
    out["p"] = chi2.chi2_sf(x2, 1)
    # STAT-14. The exact binomial used to be abandoned above 2000 discordant pairs --
    # a cutoff that excluded this tool's OWN worked example (`mcnemar --a-only 1629
    # --b-only 963`, n = 2592) and both McNemar tables in the known-answer fixture, so
    # the number a careful reader wants was withheld precisely where it was documented.
    # binom_sf_two_sided is now an O(kk) Decimal recurrence rather than a sum of kk
    # exact bignums, identical to the last ULP and fast enough that no cutoff is needed.
    out["p_exact"] = chi2.binom_sf_two_sided(b01, n)
    out["favours"] = "a" if b01 > b10 else ("b" if b10 > b01 else "tie")
    return out


# =============================================== percentile-exceedance guard
def percentile_ok(n: int, q: float, min_exceedances: int = MIN_EXCEEDANCES) -> bool:
    """n*(1-q) >= min_exceedances, evaluated without the float-boundary artifact.

    His ``_percentile_ok`` is ``n * (1.0 - q) >= MIN_EXCEEDANCES``.  At an exact
    boundary that is wrong by one ULP in the unhelpful direction: 1 - 0.9 is
    0.09999999999999998, so n=1000, q=0.90 evaluates to 99.99999999999998 and
    the guard suppresses a quantile with exactly 100 exceedances.  The tolerance
    below removes that and changes NO decision in his campaign -- his two panel
    sizes are 34799 and 51175 against quantiles .90/.95/.99/.999, none of which
    lands on a boundary.  Flagged to him rather than silently diverged.
    """
    return n * (1.0 - q) >= min_exceedances - 1e-9 * max(1.0, float(min_exceedances))


def percentile_guard(n: int, quantiles: Sequence[float] = (0.90, 0.95, 0.99, 0.999),
                     min_exceedances: int = MIN_EXCEEDANCES) -> Dict[str, Any]:
    rows = []
    for q in quantiles:
        exc = n * (1.0 - q)
        # percentile_ok(), not a second inline copy of the comparison: written
        # inline this used the bare `exc >= min_exceedances` and so disagreed
        # with percentile_ok() at exactly the boundary percentile_ok() exists to
        # fix (n=1000, q=0.90 -> ok here, refused there). Decision-neutral on
        # every real panel size (51175 / 34799 / 2047), but two functions in one
        # module must not answer the same question differently.
        rows.append({"q": q, "exceedances": exc,
                     "ok": percentile_ok(n, q, min_exceedances)})
    return {"n": n, "min_exceedances": min_exceedances, "quantiles": rows}


def guard_pooled_percentiles(per_window: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Refuse a pooled percentile that per-window summaries cannot support.

    A panel p95 is NOT the mean of per-window p95s and is not recoverable from
    them.  Our published receipts carry per-window percentiles; quoting them as
    a panel percentile would be wrong, so this returns a refusal instead of a
    number.  It is the same shape of rule as the registry's cross-lane refusal.
    """
    # STAT-18. all() over an EMPTY sequence is vacuously True, so this refusal helper
    # answered "yes, pooled percentiles are derivable" when handed no data at all --
    # the opposite of what it exists to say. The empty case gets its own reason rather
    # than blaming per-window summaries that were never supplied.
    have_tokens = bool(per_window) and all("values" in w for w in per_window)
    if have_tokens:
        return {"available": True}
    reason = ("no windows supplied" if not per_window else
              ("pooled token percentiles are not derivable from per-window "
               "summaries; a panel p95 is not a function of per-window p95s"))
    return {
        "available": False,
        "reason": reason,
        "remedy": "publish the per-token KL array (or per-window sufficient "
                  "statistics: n, sum d, sum d^2, and a quantile sketch)",
    }


# ================================================================ per-domain
def domain_table(
    per_window: Sequence[Dict[str, Any]],
    b: int = DOMAIN_BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    backend: str = "auto",
    interval: str = "delta_t_log",
) -> List[Dict[str, Any]]:
    """Per-domain stratified means with window-clustered SE and a bootstrap CI.

    ``per_window`` items need ``window_id``, ``domain``, ``count``, ``mean``
    and optionally ``std``.

    STAT-01 + STAT-17, fixed together because they are one decision.

    *The seed.*  Each domain now bootstraps from ``derive_seed(seed, domain)``
    instead of the shared ``seed``, so the strata stop sharing Monte-Carlo error,
    and the seed it used is returned on the row so the endpoints stay
    reproducible from the published record.

    *The interval.*  ``interval="t_log"`` (the default, and what this registry
    publishes) is bootstrap-t on log(mean): 92.3% measured coverage at g=5-7
    against BCa's 81.6%.  ``interval="bca"`` reproduces the pre-2026-08-30
    procedure and is kept so the old numbers can be regenerated and diffed, not
    because it is a supported choice for a small stratum.

    Neither change makes the interval nominal.  ``coverage_measured`` on the
    published cell is what closes STAT-01: the row states what it measures.
    """
    if interval not in ("delta_t_log", "t_log", "bca"):
        raise ValueError("interval must be 'delta_t_log', 't_log' or 'bca', got %r" % interval)
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for w in per_window:
        by_domain.setdefault(str(w.get("domain", "unknown")), []).append(w)
    rows = []
    for dom in sorted(by_domain):
        ws = by_domain[dom]
        summ = se_from_window_summaries(ws)
        row: Dict[str, Any] = {"domain": dom, "windows": len(ws)}
        row.update(summ)
        if len(ws) >= 2:
            dseed = derive_seed(seed, dom)
            means = {w["window_id"]: float(w["mean"]) for w in ws}
            row["bootstrap_b"] = b
            row["bootstrap_seed"] = dseed
            row["small_g"] = len(ws) < SMALL_G
            if interval == "delta_t_log":
                dt = delta_t_log(means)
                row["ci95_t_log"] = dt["ci95_delta_t_log"]
                row["ci95_bca"] = None
                row["ci95_percentile"] = None
                row["interval_method"] = "delta_t_log"
                row["interval_kind"] = "t"
                row["t_critical"] = dt["t_critical"]
                row["df"] = dt["df"]
                # No resample stream, so no seed and no B to report. Reported as
                # null rather than omitted: "this interval used no bootstrap" is a
                # fact about it, and a missing key would read as an oversight.
                row["bootstrap_b"] = None
                row["bootstrap_seed"] = None
            elif interval == "t_log":
                bt = bootstrap_t_log(means, b=b, seed=dseed, backend=backend)
                row["ci95_t_log"] = bt["ci95_t_log"]
                row["ci95_bca"] = None
                row["ci95_percentile"] = None
                row["interval_method"] = "window_block_bootstrap_t_log"
                row["interval_kind"] = "t"
                row["resamples_used"] = bt["resamples_used"]
            else:
                bs = window_block_bootstrap(means, b=b, seed=dseed, backend=backend)
                row["ci95_percentile"] = bs["ci95_percentile"]
                row["ci95_bca"] = bs["ci95_bca"]
                row["ci95_t_log"] = None
                row["interval_method"] = "window_block_bootstrap_bca"
                row["interval_kind"] = "bca"
        else:
            row["ci95_percentile"] = None
            row["ci95_bca"] = None
            row["ci95_t_log"] = None
            row["interval_method"] = "none"
            row["interval_kind"] = "none"
            row["bootstrap_b"] = None
            row["bootstrap_seed"] = None
            row["note"] = "a single window cannot be block-bootstrapped"
        # Every consumer of this table gets the caveat, not just the registry.
        # bin/joint_standard.py calls this function, so a contributor running the
        # public CLI would otherwise print an interval at g=5 with nothing beside
        # it saying what that interval is worth -- which is the state STAT-01
        # found the registry in.
        if row.get("small_g") and row.get("interval_kind") not in (None, "none"):
            row["coverage_note"] = (
                "%d windows. A 95%%-labelled interval on this few clusters does not "
                "deliver 95%%: measured 81.3%% for BCa and 92.0%% for this one over the "
                "42 real cells in registry/tools/coverage_sim.py. Quote it as what it "
                "measures, or quote se_clustered_window and build your own."
                % row["windows"])
        rows.append(row)
    return rows


# =================================================== paired window comparison
def document_level_paired(diffs: Sequence[float], windows: Sequence[str],
                          documents: Dict[str, str], alpha: float = 0.05) -> Dict[str, Any]:
    """The paired contrast at the actual independent unit: the SOURCE DOCUMENT.

    P1-15 (peer review, confirmed by recomputation). The sealed 25-window panel
    is not 25 independently sourced texts: its four axes each come from ONE
    source document (7/6/6/6 windows), and clean17 holds three documents
    (7/5/5). Windows cut from one document share its topic, style and register;
    resampling them as exchangeable units, or sign-testing 25 of them, treats
    pseudoreplicates as evidence. Splitting the same four documents into MORE
    windows would shrink the window-level interval without adding one bit of
    independent textual evidence.

    Published values this replaces as the INFERENTIAL statement (K6 vs K8):
    window-level sign test p=0.004077 (full) / 0.0490 (clean17); document-level
    exact sign test p=0.125 (4 of 4 documents positive) / 0.25 (3 of 3).
    The window-level mean and BCa spread remain correct DESCRIPTIONS of this
    fixed panel; what is withdrawn is their reading as population inference.

    Sign test excludes exact ties (Dixon-Mood), matching the window-level test.
    The t interval weights documents equally; with 3-4 documents it is
    illustrative, not calibrated, and the receipt says so.
    """
    by_doc: Dict[str, List[float]] = {}
    for w, d in zip(windows, diffs):
        by_doc.setdefault(documents[w], []).append(d)
    doc_ids = sorted(by_doc)
    doc_means = [statistics.fmean(by_doc[d]) for d in doc_ids]
    g = len(doc_ids)
    wins_a = sum(1 for m in doc_means if m < 0)   # lower KLD is better for A
    wins_b = sum(1 for m in doc_means if m > 0)
    ties = g - wins_a - wins_b
    sign_n = wins_a + wins_b
    out: Dict[str, Any] = {
        "unit": "source_document",
        "n_documents": g,
        "per_document": [
            {"document_id": d, "n_windows": len(by_doc[d]),
             "mean_diff": statistics.fmean(by_doc[d])}
            for d in doc_ids],
        "documents_a_better": wins_a,
        "documents_b_better": wins_b,
        "documents_tied": ties,
        "sign_test_n": sign_n,
        "sign_test_p": (None if sign_n == 0
                        else chi2.binom_sf_two_sided(wins_a, sign_n)),
        "note": ("the panel's windows derive from %d source document%s; the document is "
                 "the independent sampling unit, and this block is the inferential "
                 "statement. Window-level intervals and sign tests above it are "
                 "DESCRIPTIVE of this fixed panel only." % (g, "" if g == 1 else "s")),
    }
    if g >= 2:
        mean = statistics.fmean(doc_means)
        se = statistics.stdev(doc_means) / math.sqrt(g)
        t = chi2.student_t_ppf(1.0 - alpha / 2.0, g - 1)
        out.update({
            "mean_diff_equal_documents": mean,
            "se_equal_documents": se,
            "df": g - 1,
            "t_critical": t,
            "ci95_diff_t": [mean - t * se, mean + t * se],
            "t_interval_note": ("equal-document-weight Student-t; at %d documents this is "
                                "illustrative, not calibrated -- a domain-population "
                                "interval is not estimable from one document per domain"
                                % g),
        })
    return out


def paired_windows(
    a: Dict[str, float],
    b: Dict[str, float],
    label_a: str = "a",
    label_b: str = "b",
    boot_b: int = 2000,
    seed: int = BOOTSTRAP_SEED,
    backend: str = "auto",
    documents: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Rank two students on the SAME windows by their paired difference.

    Never rank by eyeballing two overlapping marginal CIs: the windows are
    common to both, so the pairing removes the window variance that dominates
    the marginals.  The paired CI here is routinely an order of magnitude
    tighter than the two marginal CIs it sits between.
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        raise ValueError("need at least 2 common windows")
    da = [float(a[w]) for w in common]
    db = [float(b[w]) for w in common]
    diffs = [x - y for x, y in zip(da, db)]
    mean_a = statistics.fmean(da)
    mean_b = statistics.fmean(db)
    # STAT-02. The sign test counted EXACT TIES as wins for B: wins_a counted d < 0 and
    # the emitted windows_b_better was `n - wins_a`, with n = every common window. Ties
    # carry no sign and must leave BOTH the count and the binomial denominator (Dixon-Mood
    # zero-exclusion). Comparing a series with itself reported 25-0 and p = 5.96e-08
    # alongside a mean difference of exactly 0.0; the two headline K6 numbers -- sealed
    # 0.013723384665701147 and streaming 0.013714888822596553 -- share 11 EXACT per-window
    # ties, so the single most natural next comparison gave p = 0.2295 instead of 0.4240.
    # Worse, the bug made the test ARGUMENT-ORDER DEPENDENT, which no symmetric test may
    # be: paired(sealed, stream) gave 0.2295 and paired(stream, sealed) gave 0.0041 on the
    # same data -- the same comparison crossing the 0.05 line depending on which series
    # was called A. bin/fidelity_stats.py:631 already excluded ties; the repo disagreed
    # with itself.
    wins_a = sum(1 for d in diffs if d < 0)   # lower KLD is better
    wins_b = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0.0)
    n = len(common)
    sign_n = wins_a + wins_b

    # paired bootstrap over the SAME resampled window index for both series
    chosen = backend
    if backend in ("auto", "numpy"):
        try:
            import numpy as np  # noqa: F401
            chosen = "numpy"
        except Exception:
            chosen = "stdlib"
    if chosen == "numpy":
        import numpy as np

        A = np.asarray(da); B = np.asarray(db)
        rng = np.random.default_rng(seed)
        bd = np.empty(boot_b); br = np.empty(boot_b)
        for i in range(boot_b):
            idx = rng.integers(0, n, size=n)
            ma, mb = A[idx].mean(), B[idx].mean()
            bd[i] = ma - mb
            br[i] = ma / mb if mb > 0 else float("nan")
        dlo, dhi = (float(x) for x in np.quantile(bd, [0.025, 0.975]))
        rlo, rhi = (float(x) for x in np.quantile(br[~np.isnan(br)], [0.025, 0.975]))
        jack = [float((np.delete(A, k) - np.delete(B, k)).mean()) for k in range(n)]
        obs = float((A - B).mean())
        blo, bhi, z0, acc = _bca_endpoints(list(bd), obs, jack, 0.05)
    else:
        rnd = random.Random(seed)
        bd, br = [], []
        for _ in range(boot_b):
            idx = [rnd.randrange(n) for _ in range(n)]
            ma = statistics.fmean(da[i] for i in idx)
            mb = statistics.fmean(db[i] for i in idx)
            bd.append(ma - mb)
            if mb > 0:
                br.append(ma / mb)
        sd = sorted(bd); sr = sorted(br)
        dlo, dhi = quantile_linear(sd, 0.025), quantile_linear(sd, 0.975)
        rlo, rhi = quantile_linear(sr, 0.025), quantile_linear(sr, 0.975)
        jack = [statistics.fmean([diffs[i] for i in range(n) if i != k]) for k in range(n)]
        blo, bhi, z0, acc = _bca_endpoints(bd, statistics.fmean(diffs), jack, 0.05)

    cse = clustered_se(diffs, common)
    out = {
        "label_a": label_a,
        "label_b": label_b,
        "n_windows": n,
        "windows": common,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": statistics.fmean(diffs),
        "mean_diff_se": cse["se_naive"],
        "ci95_diff_percentile": [dlo, dhi],
        "ci95_diff_bca": [blo, bhi],
        "bca_z0": z0,
        "bca_acceleration": acc,
        "ratio_a_over_b": mean_a / mean_b if mean_b else float("nan"),
        "ci95_ratio_percentile": [rlo, rhi],
        "windows_a_better": wins_a,
        "windows_b_better": wins_b,
        "windows_tied": ties,
        # The denominator the published p must be reproducible FROM. Emitted explicitly so
        # a reader is never left to infer it from n_windows, which is not it.
        "sign_test_n": sign_n,
        "sign_test_p": (None if sign_n == 0
                        else chi2.binom_sf_two_sided(wins_a, sign_n)),
        "sign_test_note": ("every window tied exactly; the sign test has no informative "
                           "pairs" if sign_n == 0 else None),
        "excludes_zero": (dlo > 0) or (dhi < 0),
        "bca_excludes_zero": (blo > 0) or (bhi < 0),
        "bootstrap_b": boot_b,
        "seed": seed,
        "backend": chosen,
    }
    covered = documents is not None and all(w in documents for w in common)
    if covered:
        doc_block = document_level_paired(diffs, common, documents)
        out["document_level"] = doc_block
        out["inference_unit"] = "source_document"
        out["window_stats_are"] = (
            "descriptive of this fixed panel only: %d windows derive from %d source "
            "document(s), so window-level bootstrap intervals and the window-level "
            "sign test treat pseudoreplicates as independent evidence (P1-15). The "
            "inferential statement is document_level."
            % (n, doc_block["n_documents"]))
    else:
        out["document_level"] = {
            "available": False,
            "reason": ("no document map supplied%s; window-to-document provenance is "
                       "REQUIRED to read any of the interval or sign-test fields above "
                       "as inference rather than description -- windows cut from one "
                       "source document are pseudoreplicates"
                       % ("" if documents is None else
                          " for every common window")),
        }
        out["inference_unit"] = "none"
        out["window_stats_are"] = (
            "descriptive of this fixed panel only; without window-to-document "
            "provenance no inferential reading is supported (P1-15)")
    return out
