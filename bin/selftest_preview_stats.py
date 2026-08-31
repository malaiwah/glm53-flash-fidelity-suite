#!/usr/bin/env python3
"""T3 -- the stratified preview estimator is unbiased, covered, and seeded.

    python3 bin/selftest_preview_stats.py

Builds a synthetic heavy-tailed 25x2047 population with a fixed seed, then
certifies exactly what the fixture CI can certify (and nothing it cannot):
estimator unbiasedness, empirical 95% coverage, the FPC known answer, the
<25-window refusal, and same-seed determinism of the systematic design.
Stock python3.9, stdlib only, offline; runtime a few seconds.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import previewstats as PS                     # noqa: E402

PASS, FAIL = [], []
WINDOWS = 25
N_PER_WINDOW = 2047
M = 64
REPLICATES = 200


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def build_population(sigma, spike_rate, spike_max, seed=20260829):
    """Heavy-tailed KLD-shaped values: lognormal body + rare spikes, window
    means spread like the real panel's (between-window structure).  sigma
    controls how heavy; the selftest uses TWO populations on purpose (see
    [3] and [3b])."""
    rnd = random.Random(seed)
    population = {}
    for j in range(WINDOWS):
        mu_j = -6.5 + rnd.gauss(0.0, 0.6)          # window difficulty shift
        values = []
        for _ in range(N_PER_WINDOW):
            x = math.exp(rnd.gauss(mu_j, sigma))   # lognormal tail
            if rnd.random() < spike_rate:
                x += rnd.uniform(0.5, spike_max)   # the 2.13-class spikes
            values.append(x)
        population["final-%04d" % j] = values
    return population


def replicate(population, n_positions, true_mean, m, replicates):
    estimates, covered = [], 0
    for r in range(replicates):
        samples = {}
        for w, values in population.items():
            idx = PS.systematic_indices(r, w, N_PER_WINDOW, m)
            samples[w] = [values[i] for i in idx]
        mu = PS.stratified_mean(samples, n_positions)
        var = PS.stratified_variance(samples, n_positions)
        lo, hi = PS.z_interval(mu, var)
        estimates.append(mu)
        if lo <= true_mean <= hi:
            covered += 1
    return estimates, covered / replicates


def main() -> int:
    print("[0] population: %d windows x %d positions, fixed seed" %
          (WINDOWS, N_PER_WINDOW))
    # sigma=0.7: a moderately heavy tail, the regime where the z-interval's
    # assumptions hold -- [3b] then measures what happens when they do not
    population = build_population(sigma=0.7, spike_rate=0.0, spike_max=0.0)
    n_positions = {w: N_PER_WINDOW for w in population}
    true_mean = PS.stratified_mean(population, n_positions)
    pooled = sum(sum(v) for v in population.values()) / (WINDOWS * N_PER_WINDOW)
    check("stratified mean of the FULL population == pooled mean (identity "
          "when m_j == N_j, rel 1e-12)",
          abs(true_mean - pooled) <= 1e-12 * abs(pooled),
          "%.9e vs %.9e" % (true_mean, pooled))

    print("\n[1] FPC known answers")
    check("full census -> Var_hat == 0 exactly (FPC kills the last term)",
          PS.stratified_variance(population, n_positions) == 0.0)
    tiny = {"w": [1.0, 3.0]}
    check("hand case: N=4, sample [1,3] -> (1)(1-2/4)(2)/2 == 0.5",
          PS.stratified_variance(tiny, {"w": 4}) == 0.5,
          "%r" % PS.stratified_variance(tiny, {"w": 4}))

    print("\n[2] same-seed determinism of the systematic design")
    a = PS.systematic_indices(7, "final-0003", 2047, 256)
    b = PS.systematic_indices(7, "final-0003", 2047, 256)
    c = PS.systematic_indices(8, "final-0003", 2047, 256)
    check("same seed -> identical indices", a == b)
    check("different seed -> a different start", a != c)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engines" / "tools"))
        import stream_score
        same = all(
            PS.systematic_indices(s, w, 2047, m) ==
            stream_score.preview_position_indices(s, w, 2047, m)
            for s in (0, 7) for w in ("final-0000", "final-0019")
            for m in (64, 256))
        check("previewstats and stream_score implement the SAME design "
              "(cross-checked)", same)
    except ImportError as exc:
        print("  skip cross-check (stream_score not importable here: %s)" % exc)

    print("\n[3] unbiasedness + 95%% coverage over R=%d seeded replicates "
          "(m=%d/window, systematic; moderate tail sigma=0.7)"
          % (REPLICATES, M))
    estimates, coverage = replicate(population, n_positions, true_mean,
                                    M, REPLICATES)
    bias = statistics.fmean(estimates) - true_mean
    se_single = statistics.stdev(estimates)
    check("|mean bias| < SE/5 (SE = one estimate's sd; MC noise alone is "
          "SE/sqrt(200) = SE/14)", abs(bias) < se_single / 5.0,
          "bias %.3e vs SE %.3e" % (bias, se_single))
    check("empirical 95% coverage in [92%, 98%] (the +/-2sd binomial band "
          "at R=200 is +/-3.1%)", 0.92 <= coverage <= 0.98,
          "%.1f%%" % (100 * coverage))

    print("\n[3b] KNOWN ANSWER: on an EXTREME tail (sigma=1.6 + 2.13-class "
          "spikes, the real panel's regime: mean/median ~ 9) the plain "
          "z-interval UNDER-COVERS at small m -- measured here, not asserted. "
          "Cause: the estimate and its SE are positively correlated on "
          "heavy-tailed data (miss the spikes -> low mean AND low SE). This "
          "is why kld-preview quotes the WIDER of z/bootstrap, reports "
          "tail_disclosure, and defaults to m=256; the remedy that actually "
          "works is more positions, and that is asserted too.")
    extreme = build_population(sigma=1.6, spike_rate=0.001, spike_max=2.13)
    extreme_true = PS.stratified_mean(extreme, n_positions)
    ex_estimates, ex_coverage = replicate(extreme, n_positions, extreme_true,
                                          M, REPLICATES)
    check("z-interval coverage drops below 92% on the extreme tail at m=64 "
          "(anti-conservatism is REAL, not hypothetical)",
          ex_coverage < 0.92, "%.1f%%" % (100 * ex_coverage))
    ex_bias = statistics.fmean(ex_estimates) - extreme_true
    check("the ESTIMATOR stays unbiased even there (|bias| < SE/5)",
          abs(ex_bias) < statistics.stdev(ex_estimates) / 5.0,
          "bias %.3e vs SE %.3e" % (ex_bias, statistics.stdev(ex_estimates)))
    _, cov512 = replicate(extreme, n_positions, extreme_true, 512, 100)
    check("coverage improves with m (m=512 > m=64): more positions is the "
          "remedy, exactly as the sample-size table says",
          cov512 > ex_coverage, "m=512: %.1f%% vs m=64: %.1f%%"
          % (100 * cov512, 100 * ex_coverage))

    print("\n[3c] POSITIONAL-TREND population: KLD falls with context depth, "
          "so a sampler that cannot reach late positions is biased. The old "
          "integer-step design (k = floor(N/m)) made positions >= k*m "
          "unreachable at ANY seed -- 12.5% of every window at m=256 -- and "
          "would fail this check by ~+11% relative bias; the fractional-step "
          "design reaches every position with probability exactly m/N.")
    trend_rnd = random.Random(4)
    trend = {}
    for j in range(WINDOWS):
        base = 0.01 * math.exp(trend_rnd.gauss(0.0, 0.3))
        trend["final-%04d" % j] = [
            base * math.exp(-3.0 * pos / N_PER_WINDOW) *
            math.exp(trend_rnd.gauss(0.0, 0.2))
            for pos in range(N_PER_WINDOW)]
    trend_true = PS.stratified_mean(trend, n_positions)
    t_estimates, _ = replicate(trend, n_positions, trend_true, 256, 100)
    t_bias = statistics.fmean(t_estimates) - trend_true
    t_se = statistics.stdev(t_estimates)
    check("m=256 estimator unbiased on the trend population "
          "(|bias| < 3*SE/sqrt(R): MC noise only; the integer-step design "
          "sat ~40 MC-sd away)", abs(t_bias) < 3.0 * t_se / math.sqrt(100),
          "bias %.3e (rel %.2f%%) vs MC bound %.3e"
          % (t_bias, 100 * t_bias / trend_true, 3.0 * t_se / math.sqrt(100)))
    check("all %d positions reachable across seeds (m=256)" % N_PER_WINDOW,
          len({i for s in range(60)
               for i in PS.systematic_indices(s, "final-0000",
                                              N_PER_WINDOW, 256)}) ==
          N_PER_WINDOW)

    print("\n[4] the bootstrap: seeded, and it never resamples windows")
    samples = {w: [population[w][i] for i in PS.systematic_indices(1, w, N_PER_WINDOW, M)]
               for w in population}
    b1 = PS.stratified_position_bootstrap(samples, n_positions, 500, 11)
    b2 = PS.stratified_position_bootstrap(samples, n_positions, 500, 11)
    check("same seed -> identical bootstrap interval",
          (b1["low"], b1["high"]) == (b2["low"], b2["high"]))
    mu = PS.stratified_mean(samples, n_positions)
    check("bootstrap interval brackets the point estimate",
          b1["low"] <= mu <= b1["high"])

    print("\n[5] the <25-window panel gate")
    subset = {w: samples[w] for w in sorted(samples)[:11]}
    try:
        PS.require_all_windows(len(subset), WINDOWS)
        check("11-window panel estimate refused", False)
    except PS.PanelGateError as exc:
        check("11-window panel estimate refused", True)
        # CC-01. This used to assert the literals "1.73e-3" and "1.22e-3", which
        # LOCKED IN a wrong pair: they are K8-ANOMALY.json's per-window DELTA sd and
        # pooled delta over an 11-window subset, quoted in the refusal as if they were
        # the full-panel KLD scatter and effect. bin/check_doc_numbers.py now re-derives
        # the real values from registry/protocol/per-window/, so this asserts the SHAPE
        # of the refusal and the derivation gate owns the numbers.
        check("refusal text carries the power arithmetic (scatter, paired delta, effect)",
              "7.2e-3" in str(exc) and "2.0e-3" in str(exc) and "1.33e-3" in str(exc)
              and "lessons 28/29" in str(exc))

    print("\n[6] tail disclosure fields exist and are sane")
    tail = PS.tail_disclosure(samples, n_positions)
    check("max_sampled_value and top3_share present",
          tail["max_sampled_value"] > 0 and 0 < tail["top3_share_of_estimate"] < 1,
          tail)

    print("\n" + "-" * 72)
    print("selftest_preview_stats: %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
