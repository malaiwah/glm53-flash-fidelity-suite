#!/usr/bin/env python3
"""selftest for the arch-determinism analysis -- the parts that carry a claim.

`docs/ARCHITECTURE-DETERMINISM.md` says, in print, that no variant digest key
is a function of compute capability alone, that one is a function of SM count
alone only vacuously, and that the cross-card KLD spread is such-and-such a
number. Three helpers decide all of that: `explained_by`, `partition` and
`kld_fp64`. If any of them is wrong the document is wrong, so they are tested
here rather than trusted.

Each case is written to FAIL against a plausible wrong implementation:
`explained_by` must reject BOTH failure directions (same attribute different
result, and same result different attribute -- a version checking only the
first passes case 2 and fails case 3), `kld_fp64` must be exactly 0.0 on
identical inputs (a naive softmax without the max-shift is merely close), and
`ulp_f32` must count across the sign boundary (an implementation that subtracts
raw int32 bit patterns gets that wrong by 2^31).

    python3 reports/arch-determinism/selftest_arch_determinism.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                          # noqa: E402

import analyse                                              # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        print("  PASS  %s" % name)
        PASS += 1
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAIL += 1


def groups(*gs):
    """A partition in the shape `partition()` returns: digest -> [labels]."""
    return {"d%d" % i: list(g) for i, g in enumerate(gs)}


def main():
    print("selftest_arch_determinism")

    # -- explained_by ------------------------------------------------------
    cap = {"a": "sm_90", "b": "sm_90", "c": "sm_120", "d": "sm_120"}
    check("[1] a partition that IS the capability partition is explained by it",
          analyse.explained_by(groups("ab", "cd"), cap) is True)
    check("[2] same capability, different result -> not explained",
          analyse.explained_by(groups("a", "bcd"), cap) is False)
    check("[3] same result, different capability -> not explained "
          "(the direction a one-sided check misses)",
          analyse.explained_by(groups("abcd"), cap) is False)
    smc = {"a": 58, "b": 84, "c": 108, "d": 132}
    check("[4] all-singletons is vacuously a function of a bijective attribute",
          analyse.explained_by(groups("a", "b", "c", "d"), smc) is True)
    check("[5] one group over a bijective attribute -> not explained",
          analyse.explained_by(groups("abcd"), smc) is False)

    # -- partition ---------------------------------------------------------
    boxes = {
        "x": {"lane": {"digests": {"K": "aa", "J": "zz"}}},
        "y": {"lane": {"digests": {"K": "aa", "J": "ww"}}},
        "z": {"lane": {"digests": {"K": "bb", "J": "zz"}}},
    }
    check("[6] partition groups labels by digest value",
          analyse.partition(boxes, "K") == {"aa": ["x", "y"], "bb": ["z"]})
    check("[7] a key absent from a box groups under None, it is not dropped",
          analyse.partition(boxes, "MISSING") == {None: ["x", "y", "z"]})

    # -- kld_fp64 ----------------------------------------------------------
    rng = np.random.Generator(np.random.PCG64(7))
    a = rng.standard_normal((16, 512), dtype=np.float32)
    same = analyse.kld_fp64(a, a)
    check("[8] KLD of a distribution with itself is EXACTLY 0.0, not merely small",
          bool((same == 0.0).all()), "max=%r" % float(np.abs(same).max()))
    # a hand-computable two-class case: logits [0, t] vs [0, s]
    t, s = 1.0, 0.25
    tv = np.array([[0.0, t]], dtype=np.float32)
    sv = np.array([[0.0, s]], dtype=np.float32)
    pt = 1.0 / (1.0 + math.exp(-t))
    ps = 1.0 / (1.0 + math.exp(-s))
    want = pt * math.log(pt / ps) + (1 - pt) * math.log((1 - pt) / (1 - ps))
    got = float(analyse.kld_fp64(tv, sv)[0])
    check("[9] KLD matches the closed form on a two-class case",
          abs(got - want) < 1e-12, "got %.17g want %.17g" % (got, want))
    # overflow guard: without the max-shift these logits produce inf/nan
    big = np.array([[900.0, 0.0, -900.0]], dtype=np.float32)
    big2 = np.array([[900.0, 1.0, -900.0]], dtype=np.float32)
    v = analyse.kld_fp64(big, big2)
    check("[10] extreme logits stay finite (the max-shift is present)",
          bool(np.isfinite(v).all()), "got %r" % v)

    # -- ulp_f32 -----------------------------------------------------------
    one = np.array([1.0], dtype=np.float32)
    nxt = np.nextafter(one, np.float32(2.0)).astype(np.float32)
    check("[11] adjacent floats are 1 ULP apart",
          int(analyse.ulp_f32(one, nxt)[0]) == 1,
          "got %d" % int(analyse.ulp_f32(one, nxt)[0]))
    check("[12] a float and itself are 0 ULP apart",
          int(analyse.ulp_f32(one, one)[0]) == 0)
    pz = np.array([0.0], dtype=np.float32)
    nz = np.array([-0.0], dtype=np.float32)
    check("[13] +0.0 and -0.0 are 0 ULP apart, not 2^31 "
          "(the sign-boundary case raw int subtraction gets wrong)",
          int(analyse.ulp_f32(pz, nz)[0]) == 0,
          "got %d" % int(analyse.ulp_f32(pz, nz)[0]))
    neg = np.array([-1.0], dtype=np.float32)
    check("[14] ULP distance across zero is small, not 2^32-scale",
          int(analyse.ulp_f32(neg, one)[0]) < 2 ** 31,
          "got %d" % int(analyse.ulp_f32(neg, one)[0]))

    print("selftest_arch_determinism: %d passed, %d failed" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
