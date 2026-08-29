#!/usr/bin/env python3
"""T8 -- the comparator's numerics: known answers and exactness.

Runs on any interpreter with numpy.  When torch is importable the estimator
under test IS `k6_kld_report._token_kld` (imported, not copied); otherwise the
numpy fp64 fallback runs and the receipt says so.  Both paths are asserted
against the SAME analytic answers, which is the point: a backend swap must not
move a number.

    N1   known-answer KLD, analytic, 1e-15
    N2   KL(x||x) on a random capture is all-zero, exactly
    N3   self-compare A == B by digest: exactly 0.0, top-1 exactly 1.0, +0.0 maxima
    N4   N3 with --force-compute: the computed array is bitwise identical
    N5   the T1 constant: 51,175 float64 zeros -> 409,528 bytes, 3ffddc61...be17
    N6   same weights identity, different capture content -> run_to_run_floor
    N7   vocab-chunk invariance
    N8   a --vocab-chunk that does not divide vocab_size is refused with the hint
    N9   a NaN in one capture -> hard refusal, never a clamp
    N10  a permuted head applied at replay -> large KLD (the estimator has teeth)
    N11  a reproduction-confirmation receipt fed to the submission builder -> refused
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from fidelity import dsformat as F  # noqa: E402
from fidelity import dscompare, dsvalidate  # noqa: E402

import selftest_fidelity_dataset as fixtures  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def analytic_kl(p_logits, q_logits):
    """KL(P||Q) from logits, in plain python floats -- an independent oracle."""
    out = []
    for row_p, row_q in zip(p_logits, q_logits):
        mp = max(row_p)
        mq = max(row_q)
        zp = sum(math.exp(v - mp) for v in row_p)
        zq = sum(math.exp(v - mq) for v in row_q)
        total = 0.0
        for vp, vq in zip(row_p, row_q):
            p = math.exp(vp - mp) / zp
            lp = (vp - mp) - math.log(zp)
            lq = (vq - mq) - math.log(zq)
            total += p * (lp - lq)
        out.append(total)
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="fidelity-compare-selftest-")
    backend_note = "torch" if dscompare._torch_available() else "numpy fp64 fallback"
    print("== N: comparator numerics (estimator backend: %s) ==" % backend_note)
    try:
        # -- N1 known-answer -------------------------------------------------
        rng = np.random.RandomState(11)
        p = rng.normal(size=(2, 8)).astype(np.float32) * 3.0
        q = rng.normal(size=(2, 8)).astype(np.float32) * 3.0
        values, matches, backend = dscompare.token_kld(p, q, "cpu")
        want = analytic_kl(p.astype(np.float64).tolist(), q.astype(np.float64).tolist())
        worst = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(values.tolist(), want))
        check("N1  known-answer KLD (2 x 8, analytic) agrees to fp64 epsilon (<1e-15 rel)",
              worst < 1e-15, "worst relative delta = %.3e, backend=%s" % (worst, backend))

        # -- N2 KL(x||x) -----------------------------------------------------
        values, matches, _ = dscompare.token_kld(p, p, "cpu")
        check("N2  KL(x||x) is exactly zero everywhere and top-1 agrees",
              bool(np.all(values == 0.0)) and matches == p.shape[0],
              "max=%r" % float(values.max()))

        # -- N3/N4 self-compare ---------------------------------------------
        a = os.path.join(tmp, "a")
        b = os.path.join(tmp, "b")
        fixtures.build_dataset(a, seed=1)
        shutil.copytree(a, b)
        out = os.path.join(tmp, "sc")
        receipt = dscompare.compare(a, b, out, {"self_compare": True, "vocab_chunk": 8})
        zeros_ok = (
            receipt["comparison_kind"] == "reproduction_confirmation"
            and receipt["metric"]["value"] == 0.0
            and receipt["top1_agreement"] == 1.0
            and all(v == 0.0 for v in receipt["kl"].values())
            and all(not math.copysign(1, c["max"]) < 0 for c in receipt["per_context"])
            and receipt["comparator"]["short_circuited"] is True
        )
        check("N3  A == B by digest -> exactly 0.0, top-1 1.0, every per-window max +0.0",
              zeros_ok, json.dumps(receipt["kl"]))
        report = dsvalidate.validate_receipt(receipt)
        check("N3b the reproduction receipt passes its own schema and SC-1 rules",
              report.passed, json.dumps(report.errors[:3]))

        out2 = os.path.join(tmp, "sc-forced")
        forced = dscompare.compare(a, b, out2, {"self_compare": True, "force_compute": True,
                                                "vocab_chunk": 8})
        check("N4  --force-compute agrees bitwise with the hash proof",
              forced["self_compare"]["force_compute_agreed"] is True
              and forced["metric"]["value"] == 0.0
              and forced["comparator"]["estimator_backend"] is not None,
              json.dumps(forced["self_compare"]))

        # -- N5 the T1 constant ---------------------------------------------
        path = os.path.join(tmp, "tokenwise-51175.npy")
        meta = dscompare.save_tokenwise(path, dscompare.zero_tokenwise(51175))
        check("N5  51,175 float64 zeros -> 409,528 bytes and the published sha256",
              meta["bytes"] == F.ZERO_TOKENWISE_BYTES_51175
              and meta["sha256"] == F.ZERO_TOKENWISE_SHA256_51175,
              "%d bytes, %s" % (meta["bytes"], meta["sha256"][:16]))

        # -- N6 run-to-run floor --------------------------------------------
        c = os.path.join(tmp, "c")
        fixtures.build_dataset(c, seed=2)  # same weights identity, different content
        out3 = os.path.join(tmp, "floor")
        floor = dscompare.compare(a, c, out3, {"vocab_chunk": 8})
        check("N6  same weights, different capture content -> run_to_run_floor, never a reproduction",
              floor["comparison_kind"] == "run_to_run_floor"
              and floor["metric"]["value"] > 0.0
              and floor["self_compare"]["capture_content_digest_equal"] is False,
              floor["comparison_kind"])

        # -- N7 vocab-chunk invariance --------------------------------------
        out4 = os.path.join(tmp, "chunk4")
        out16 = os.path.join(tmp, "chunk16")
        r4 = dscompare.compare(a, c, out4, {"vocab_chunk": 4})
        r16 = dscompare.compare(a, c, out16, {"vocab_chunk": 16})
        delta = abs(r4["metric"]["value"] - r16["metric"]["value"])
        check("N7  vocab-chunk invariance: two chunk sizes agree to < 1e-12",
              delta < 1e-12, "delta = %.3e" % delta)

        # -- N8 bad vocab chunk ---------------------------------------------
        try:
            dscompare.compare(a, c, os.path.join(tmp, "bad"), {"vocab_chunk": 7})
            check("N8  a --vocab-chunk that does not divide vocab_size is refused", False,
                  "no refusal")
        except dscompare.Refusal as exc:
            check("N8  a --vocab-chunk that does not divide vocab_size is refused, with a hint",
                  exc.code == "bad_vocab_chunk" and "working values" in exc.message,
                  exc.message[:90])
        check("N8b the divisor hint for GLM-5.3-Flash names 9680, not kimi-k3's 10240",
              154880 % 10240 != 0 and 9680 in F.divisors_hint(154880, limit=12))

        # -- N9 NaN -> hard refusal -----------------------------------------
        bad_p = p.copy()
        bad_p[0, 0] = np.nan
        try:
            dscompare.token_kld(bad_p, q, "cpu")
            check("N9  a NaN in a capture is a hard refusal, never a clamp", False,
                  "no refusal")
        except (dscompare.Refusal, Exception) as exc:
            check("N9  a NaN in a capture is a hard refusal, never a clamp",
                  "finite" in str(exc).lower(), str(exc)[:110])

        # -- N10 permuted head ----------------------------------------------
        ref = dscompare.load_dataset(a)
        cand = dscompare.load_dataset(b)
        gates, findings = dscompare.run_gates(ref, cand, {})
        head_path = ref.head_path()
        head = dscompare.load_tensor(head_path, "lm_head.weight")
        permuted = head[::-1].copy()
        hidden = dscompare.load_tensor(ref.record_path(ref.records[0]), "hidden_states")
        straight = hidden @ np.ascontiguousarray(head.T)
        crooked = hidden @ np.ascontiguousarray(permuted.T)
        big, _, _ = dscompare.token_kld(straight, crooked, "cpu")
        small, _, _ = dscompare.token_kld(straight, straight, "cpu")
        check("N10 a permuted head at replay produces a large KLD (the estimator has teeth)",
              float(big.mean()) > 1.0 and float(small.mean()) == 0.0,
              "permuted mean = %.4f" % float(big.mean()))

        # -- N11 SC-3 --------------------------------------------------------
        try:
            dscompare.emit_submission(
                receipt, os.path.join(tmp, "submission.json"),
                measurer={"name": "selftest", "handle": "selftest", "url": None,
                          "is_artifact_author": False},
                artifact={}, panel={}, reference={})
            check("N11 a reproduction-confirmation receipt is refused a submission (SC-3)",
                  False, "no refusal")
        except dscompare.NotAMeasurement as exc:
            check("N11 a reproduction-confirmation receipt is refused a submission (SC-3)",
                  "reproduction_confirmation" in str(exc), str(exc)[:80])
        try:
            dscompare.emit_submission(
                floor, os.path.join(tmp, "submission2.json"),
                measurer={}, artifact={}, panel={}, reference={})
            check("N11b a run_to_run_floor receipt is refused a submission too", False,
                  "no refusal")
        except dscompare.NotAMeasurement:
            check("N11b a run_to_run_floor receipt is refused a submission too", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nselftest_fidelity_compare: %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
