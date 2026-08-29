"""Stratified-sampling estimator for preview KLD scoring.  PURE stdlib.

This is the unit-testable math core of bin/kld-preview: no torch, no numpy,
so its correctness (unbiasedness, FPC, coverage, seeding) is certified by a
stock-python selftest and the torch tool only feeds it numbers.

THE DESIGN RULE THIS MODULE ENFORCES.  Per-window KLD scatter (sd 1.73e-3)
exceeds the K6-vs-K8 effect (1.22e-3), so a single window has NO power to
compare quants (campaign lessons 28/29).  Every estimator here therefore
stratifies over ALL panel windows, and the panel-mean gate refuses to emit an
estimate unless every window contributed.

TWO BOOTSTRAPS, DELIBERATELY KEPT APART.  The position bootstrap here
resamples positions WITHIN windows and never resamples windows: the estimand
is the fixed sealed-panel mean, and all windows are in the design by
construction -- its CI is SAMPLING error.  The window-cluster (BCa) bootstrap
in bin/fidelity_stats.py resamples windows and answers the different
GENERALIZATION question for full-census deltas.  Confusing the two would
misstate what a CI means, so each lives in exactly one module.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

Z_975 = 1.959963984540054           # Phi^-1(0.975)


def systematic_indices(seed: int, window_id: str, n_positions: int,
                       per_window: int) -> List[int]:
    """The same systematic-with-seeded-start design stream_score.py implements
    (k6/tools must stay self-contained for upload to rented boxes, so the two
    copies exist on purpose; selftest_preview_stats.py cross-checks equality
    whenever stream_score is importable, which is what keeps them from
    drifting).

    FRACTIONAL step, not integer: step = N/m and start u ~ U[0, step), so
    every position has inclusion probability exactly m/N.  An integer step
    k = floor(N/m) would make positions >= k*m unreachable at ANY seed --
    12.5%% of every window at m=256, 50%% at m=1024 -- biasing the estimate
    whenever KLD trends with context depth (it does).  Indices are distinct
    and strictly increasing because step > 1 whenever m < N.
    """
    if per_window >= n_positions:
        return list(range(n_positions))
    step = n_positions / float(per_window)
    u = random.Random("%s:%s" % (seed, window_id)).random() * step
    return [min(n_positions - 1, int(u + i * step)) for i in range(per_window)]

CENSUS_PREVIEW_SCHEMA = "malaiwah.glm53-census-kld-preview.v1"
SAMPLED_PREVIEW_SCHEMA = "malaiwah.glm53-sampled-kld-preview.v1"

PANEL_GATE_TEXT = (
    "REFUSED: panel estimate requires all %d windows (got %d). Per-window "
    "scatter (sd 1.73e-3) exceeds the K6-vs-K8 effect (1.22e-3): a single "
    "window has no power to compare quants (campaign lessons 28/29). "
    "Printing per-window diagnostics only."
)


class PanelGateError(RuntimeError):
    """Raised when a panel-mean estimate is requested from a window subset."""


def require_all_windows(windows_used: int, windows_total: int) -> None:
    if windows_used != windows_total:
        raise PanelGateError(PANEL_GATE_TEXT % (windows_total, windows_used))


# --------------------------------------------------------------------------
# Stratified estimator with finite-population correction
# --------------------------------------------------------------------------


def stratified_mean(samples: Dict[str, Sequence[float]],
                    n_positions: Dict[str, int]) -> float:
    """mu_hat = sum_j (N_j/N) * xbar_j  (exact panel mean when m_j == N_j)."""
    total = float(sum(n_positions[w] for w in samples))
    return sum((n_positions[w] / total) * statistics.fmean(samples[w])
               for w in samples)


def stratified_variance(samples: Dict[str, Sequence[float]],
                        n_positions: Dict[str, int]) -> float:
    """Var_hat(mu_hat) = sum_j (N_j/N)^2 (1 - m_j/N_j) s_j^2 / m_j  (FPC).

    Windows with m_j < 2 contribute 0 (their s_j is undefined); callers gate
    m_j >= 8 upstream so this is a formality, not a loophole.
    """
    total = float(sum(n_positions[w] for w in samples))
    var = 0.0
    for w, xs in samples.items():
        m = len(xs)
        n = n_positions[w]
        if m < 2:
            continue
        s2 = statistics.variance(xs)
        var += (n / total) ** 2 * (1.0 - m / float(n)) * s2 / m
    return var


def z_interval(mu: float, var: float) -> Tuple[float, float]:
    half = Z_975 * (var ** 0.5)
    return (mu - half, mu + half)


def stratified_position_bootstrap(samples: Dict[str, Sequence[float]],
                                  n_positions: Dict[str, int],
                                  B: int, seed: int) -> Dict[str, Any]:
    """Percentile CI resampling positions WITHIN each window; never windows."""
    rnd = random.Random(seed)
    total = float(sum(n_positions[w] for w in samples))
    weights = {w: n_positions[w] / total for w in samples}
    ordered = sorted(samples)
    boots: List[float] = []
    for _ in range(B):
        mu = 0.0
        for w in ordered:
            xs = samples[w]
            m = len(xs)
            mu += weights[w] * (sum(xs[rnd.randrange(m)] for _ in range(m)) / m)
        boots.append(mu)
    boots.sort()

    def q(p: float) -> float:
        idx = min(max(int(p * B), 0), B - 1)
        return boots[idx]

    return {"method": "stratified-position-bootstrap", "B": B, "seed": seed,
            "low": q(0.025), "high": q(0.975)}


def wider_of(z_ci: Tuple[float, float], boot_ci: Dict[str, Any]) -> Dict[str, Any]:
    """Quote the WIDER interval: heavy tails make each individually
    anti-conservative in opposite regimes."""
    z_width = z_ci[1] - z_ci[0]
    b_width = boot_ci["high"] - boot_ci["low"]
    if b_width >= z_width:
        return {"low": boot_ci["low"], "high": boot_ci["high"],
                "source": "bootstrap"}
    return {"low": z_ci[0], "high": z_ci[1], "source": "z"}


def tail_disclosure(samples: Dict[str, Sequence[float]],
                    n_positions: Dict[str, int]) -> Dict[str, Any]:
    """How tail-dependent the estimate is -- a reader must see this.

    top3_share_of_estimate: the three largest sampled values' contribution to
    mu_hat (each weighted by its window's expansion factor), as a fraction of
    the estimate.  At m=128 a single 2.13-class position moves the estimate by
    ~6.7e-4, about one half-width -- which is why m=256 is the default.
    """
    total = float(sum(n_positions[w] for w in samples))
    contributions: List[float] = []
    max_value = None
    for w, xs in samples.items():
        weight = (n_positions[w] / total) / len(xs)
        for x in xs:
            contributions.append(weight * x)
            if max_value is None or x > max_value:
                max_value = x
    mu = sum(contributions)
    top3 = sum(sorted(contributions, reverse=True)[:3])
    return {
        "max_sampled_value": max_value,
        "top3_share_of_estimate": (top3 / mu) if mu else None,
    }


def sigma_hat_per_window(samples: Dict[str, Sequence[float]]) -> Dict[str, float]:
    return {w: (statistics.stdev(xs) if len(xs) >= 2 else float("nan"))
            for w, xs in samples.items()}


# --------------------------------------------------------------------------
# Paired common-position delta (two artifacts sampled with the SAME seed)
# --------------------------------------------------------------------------


def paired_delta(samples_a: Dict[str, Sequence[float]],
                 samples_b: Dict[str, Sequence[float]],
                 n_positions: Dict[str, int],
                 B: int, seed: int) -> Dict[str, Any]:
    """Estimator + CI for mean(b - a) over COMMON sampled positions.

    Callers must have verified position_indices identity per window first --
    pairing different positions is not a paired design, it is two noisy
    estimates in a trench coat.
    """
    if set(samples_a) != set(samples_b):
        only_a = sorted(set(samples_a) - set(samples_b))[:3]
        only_b = sorted(set(samples_b) - set(samples_a))[:3]
        raise ValueError("paired windows differ (only-a: %s; only-b: %s) -- "
                         "pairing requires identical window sets"
                         % (only_a, only_b))
    deltas: Dict[str, List[float]] = {}
    for w in samples_a:
        xa, xb = samples_a[w], samples_b[w]
        if len(xa) != len(xb):
            raise ValueError("window %s has %d vs %d sampled positions -- not "
                             "common-position pairs" % (w, len(xa), len(xb)))
        deltas[w] = [b - a for a, b in zip(xa, xb)]
    mu = stratified_mean(deltas, n_positions)
    var = stratified_variance(deltas, n_positions)
    boot = stratified_position_bootstrap(deltas, n_positions, B, seed)
    z_ci = z_interval(mu, var)
    return {
        "delta_mean": mu,
        "se": var ** 0.5,
        "ci95_z": {"low": z_ci[0], "high": z_ci[1]},
        "ci95_bootstrap": boot,
        "quoted_interval": wider_of(z_ci, boot),
    }


# --------------------------------------------------------------------------
# Receipt assembly (pure, so refusability is testable without torch)
# --------------------------------------------------------------------------


def build_preview_receipt(*, kind: str, per_window: Dict[str, Dict[str, Any]],
                          windows_total: int,
                          panel_estimate: Optional[float],
                          ci95_z: Optional[Dict[str, float]],
                          ci95_bootstrap: Optional[Dict[str, Any]],
                          sampling_design: Optional[Dict[str, Any]],
                          tail: Optional[Dict[str, Any]],
                          lane_disclosure: Dict[str, Any],
                          teacher_receipt_sha256: Optional[str],
                          extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A preview receipt, structurally unsubmittable BY FIELD SHAPE:

      * schema contains "-preview." (bin-side denylist + no adapter accepts it);
      * headline field is preview_panel_mean_estimate, never measured_mean_kld;
      * not_submittable: true, always;
      * no submission_schema key ever appears.
    """
    if kind not in ("census", "sampled"):
        raise ValueError("kind must be census|sampled")
    doc: Dict[str, Any] = {
        "schema": CENSUS_PREVIEW_SCHEMA if kind == "census" else SAMPLED_PREVIEW_SCHEMA,
        "sampled": kind == "sampled",
        "not_submittable": True,
        "windows_used": len(per_window),
        "windows_total": windows_total,
        "per_window": per_window,
        "lane_disclosure": lane_disclosure,
        "teacher_receipt_sha256": teacher_receipt_sha256,
    }
    if panel_estimate is not None:
        require_all_windows(len(per_window), windows_total)
        doc["preview_panel_mean_estimate"] = panel_estimate
    if ci95_z is not None:
        doc["ci95_z"] = ci95_z
    if ci95_bootstrap is not None:
        doc["ci95_bootstrap"] = ci95_bootstrap
        doc["quoted_interval"] = "wider_of_z_and_bootstrap"
    if sampling_design is not None:
        doc["sampling_design"] = sampling_design
    if tail is not None:
        doc["tail_disclosure"] = tail
    for key, value in (extra or {}).items():
        doc[key] = value
    forbidden = {"submission_schema", "measured_mean_kld"}
    leaked = forbidden & set(doc)
    if leaked:
        raise ValueError("a preview receipt must never carry %s" % sorted(leaked))
    if doc["schema"] not in (CENSUS_PREVIEW_SCHEMA, SAMPLED_PREVIEW_SCHEMA) \
            or doc["not_submittable"] is not True:
        raise ValueError("extra keys must not overwrite the preview receipt's "
                         "schema or not_submittable fields")
    return doc


# Sample-size arithmetic for the help text (sigma_w = 0.05 DESIGN number from
# quantile integration of the K8-ANOMALY tail forensics; the tool always
# reports the ACHIEVED CI from its own s_j, never these planning values).
SAMPLE_SIZE_TABLE = (
    ("m=32   (N=800)",   "SE 1.75e-3, half-width 3.4e-3  -- coarse triage only"),
    ("m=64   (N=1600)",  "SE 1.23e-3, half-width 2.4e-3"),
    ("m=128  (N=3200)",  "SE 8.6e-4,  half-width 1.7e-3"),
    ("m=256  (N=6400)",  "SE 5.9e-4,  half-width 1.15e-3  -- recommended default"),
    ("m=512  (N=12800)", "SE 3.8e-4,  half-width 7.5e-4"),
    ("m=1024 (N=25600)", "SE 2.2e-4,  half-width 4.3e-4"),
)

DELTA_HONESTY_TEXT = (
    "Sampled previews can separate K4-vs-K6-class gaps (>= 2e-3) at m=128; "
    "K6-vs-K8-class gaps (1.2e-3) need m >= 512 WITH common-position pairing "
    "(--student2, same --sample-seed); without pairing they cannot separate "
    "close quants at any affordable N."
)
