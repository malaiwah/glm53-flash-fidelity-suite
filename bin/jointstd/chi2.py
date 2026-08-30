"""Chi-square and normal tails, stdlib only.

scipy is not importable under the stock interpreter this repository targets,
and McNemar needs exactly one number out of it: P(chi2_1 > x).  For df=1 that
is closed form, so there is no approximation anywhere in this module:

    P(chi2_1 > x) = erfc(sqrt(x/2))

which ``math.erfc`` evaluates to full double precision including deep in the
tail (erfc does not underflow to 0 until ~1e-308, i.e. x ~ 1400).  The general
df path uses the regularized upper incomplete gamma Q(k/2, x/2) via the
standard Lentz continued fraction / series split, and is only there so a caller
that wants a 2xC table is not silently wrong.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
import statistics

_NORM = statistics.NormalDist()


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF."""
    return _NORM.inv_cdf(p)


def norm_cdf(z: float) -> float:
    return _NORM.cdf(z)


def _gamma_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by series (good for x < a+1)."""
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int = 1) -> float:
    """Upper tail P(chi2_df > x)."""
    if x <= 0:
        return 1.0
    if df == 1:
        # exact, and tail-accurate: erfc does not lose the exponent
        return math.erfc(math.sqrt(x / 2.0))
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        return 1.0 - _gamma_series(a, z)
    return _gamma_cf(a, z)


def binom_sf_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0=0.5 -- the exact McNemar test.

    The division is int/int ON PURPOSE.  ``float(2 ** n)`` raises OverflowError
    for n >= 1024, which crashed every contingency table with 1024..2000
    discordant pairs -- squarely inside the range McNemar is used on.  CPython's
    ``int.__truediv__`` divides two arbitrary-precision integers with a single
    correct rounding and never materialises either side as a float, so this is
    both crash-free and exact (checked against scipy.stats.binomtest to 9e-14
    relative out to n = 2592, including tails down to 1e-299).

    The summation is an O(kk) Decimal recurrence rather than a sum of kk exact
    bignums of ~n bits each: ``term *= (n - i) / (i + 1)`` from ``2**-n``.  That is
    identical to the bignum form to the last ULP on every table checked (worst case
    1 ULP over 600 randomised (k, n) pairs; 0.0 relative on the fixtures and on the
    published McNemar tables) and 300x-20000x faster, which is what lets the exact
    test run at ANY n instead of being abandoned above a threshold.
    """
    if n == 0:
        return float("nan")
    kk = min(k, n - k)
    # localcontext(), not a module-level getcontext() assignment: this must not
    # mutate global Decimal state for the caller.
    with localcontext() as ctx:
        ctx.prec = 40
        term = Decimal(2) ** (-n)          # C(n,0) / 2**n
        total = term
        for i in range(kk):
            term = term * Decimal(n - i) / Decimal(i + 1)
            total += term
        return min(1.0, float(2 * total))


# ============================================================== Student's t
# Added 2026-08-30 for the per-domain interval. It needs an EXACT quantile, not
# a normal approximation: at g=5 the 97.5% point is 2.7764 against the normal's
# 1.9600, so approximating it would understate the interval by 42% -- the same
# direction and roughly the same size as the undercoverage the interval is being
# rebuilt to fix.

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, by modified Lentz."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                          + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = float(df) / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: int) -> float:
    """Inverse CDF, by bisection on the exact CDF.

    Bisection rather than an asymptotic expansion (Hill's) because this value
    multiplies a published endpoint: 200 halvings of a bracket that starts at
    +-1e4 leave nothing at double precision, and the cost is microseconds on the
    42 cells that use it. Verified against the textbook table in
    registry/tools/selftest_stat01_reseed.py.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1e4, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
