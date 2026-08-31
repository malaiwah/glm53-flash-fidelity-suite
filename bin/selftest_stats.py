#!/usr/bin/env python3
"""Known-answer tests for fidelity-stats, from COMMITTED receipts.

    python3 bin/selftest_stats.py

Every number asserted below is either in a committed file (engines/K8-ANOMALY.json,
engines/native-bf16-kld.json, engines/BF16-FLOOR.json) or is arithmetic redoable on
paper.  Stock python3.9, stdlib, offline.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fidelity_stats as FS                                # noqa: E402
from fidelity.common import Console                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def rel_close(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b))


def main() -> int:
    anomaly = json.loads((ROOT / "engines" / "K8-ANOMALY.json").read_text())
    per = anomaly["per_window"]
    k6 = [row["k6"] for row in per]
    k8 = [row["k8"] for row in per]
    deltas = [b - a for a, b in zip(k6, k8)]
    n = len(deltas)

    print("[1] paired-delta reproduces the sealed K8-ANOMALY numbers (11 windows)")
    d_bar = statistics.fmean(deltas)
    s_d = statistics.stdev(deltas)
    se = s_d / math.sqrt(n)
    t = d_bar / se
    check("d_bar == -1.2177e-3 (rel 1e-4)", rel_close(d_bar, -1.2176728196882489e-3, 1e-4),
          "%.10e" % d_bar)
    check("s_d == 1.7335e-3 (rel 1e-4)", rel_close(s_d, 1.7334539428769534e-3, 1e-4),
          "%.10e" % s_d)
    check("t == -2.33 (rel 1e-2 -- the file rounds to 2 decimals)",
          rel_close(t, -2.33, 1e-2), "%.4f" % t)
    # the per_window block stores 9-decimal ROUNDED values; the headline
    # stdev was computed upstream from unrounded data, so 1e-6 rel is the
    # honest agreement bound here, not 1e-12
    check("the file's own delta stats agree (its stdev field, rel 1e-6)",
          rel_close(s_d, anomaly["headline_numbers"]
                    ["all_windows_captured_by_both_runs"]["per_window_delta_stdev"],
                    1e-6))

    print("\n[2] paired identity: s_d^2 == s_A^2 + s_B^2 - 2 rho s_A s_B")
    s_a = statistics.stdev(k6)
    s_b = statistics.stdev(k8)
    mean_a, mean_b = statistics.fmean(k6), statistics.fmean(k8)
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(k6, k8)) / (n - 1)
    rho = cov / (s_a * s_b)
    lhs = s_d ** 2
    rhs = s_a ** 2 + s_b ** 2 - 2.0 * rho * s_a * s_b
    check("identity holds to 1e-12 rel", rel_close(lhs, rhs, 1e-12),
          "lhs %.15e rhs %.15e (rho %.4f)" % (lhs, rhs, rho))
    check("rho ~ 0.95 (windows hard for the lane are hard for every quant)",
          0.93 <= rho <= 0.97, "%.4f" % rho)

    print("\n[3] the t machinery: quantile inversion matches the printed table value")
    check("t_{24,0.975} == 2.0639 (tol 1e-3)",
          abs(FS.t_quantile_975(24) - 2.0639) <= 1e-3,
          "%.4f" % FS.t_quantile_975(24))
    check("t_{10,0.975} == 2.2281 (tol 1e-3)",
          abs(FS.t_quantile_975(10) - 2.2281) <= 1e-3,
          "%.4f" % FS.t_quantile_975(10))
    check("p(t=2.0639, df=24) == 0.05 (tol 1e-4)",
          abs(FS.t_two_sided_p(2.0639, 24) - 0.05) <= 1e-4,
          "%.6f" % FS.t_two_sided_p(2.0639, 24))

    print("\n[4] exact sign test known value")
    p_sign = FS.sign_test_two_sided(9, 11)
    check("wins 9/11 -> p == 2*(1+11+55)/2048 == 0.065430 (tol 1e-6)",
          abs(p_sign - 2.0 * (1 + 11 + 55) / 2048.0) <= 1e-6, "%.6f" % p_sign)

    print("\n[5] seeded bootstrap reproducibility (same seed -> identical interval)")
    b1 = FS.bca_interval(deltas, 2000, 42)
    b2 = FS.bca_interval(deltas, 2000, 42)
    b3 = FS.bca_interval(deltas, 2000, 43)
    check("seed 42 twice -> byte-identical interval",
          (b1["low"], b1["high"]) == (b2["low"], b2["high"]))
    check("seed 43 -> a different resample (interval moves)",
          (b1["low"], b1["high"]) != (b3["low"], b3["high"]))
    check("BCa interval brackets d_bar", b1["low"] < d_bar < b1["high"],
          "[%.3e, %.3e]" % (b1["low"], b1["high"]))

    print("\n[6] the CLI end-to-end on the committed anomaly file")
    rc = FS.main(["paired-delta", "--report-a",
                  str(ROOT / "engines" / "K8-ANOMALY.json"), "--anomaly-format",
                  "--bootstrap-b", "500", "--seed", "0"])
    check("paired-delta --anomaly-format exits 0", rc == 0)

    print("\n[7] cross-lane floor refusal, with the arithmetic in the message")
    con = Console()
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        # A synthetic cross-stack 'floor' (different teacher) against the REAL
        # published K8 streaming mean: the refusal must print -0.000328.
        quant = {
            "schema": "malaiwah.glm53-k8-packed-kld-summary.v1",
            "student_label": "uniform-k8",
            "measured_mean_kld": 0.012384191023436866,
            "teacher_receipt_sha256": FS.SEALED_STREAM_TEACHER,
        }
        wrong_floor = {
            "schema": "malaiwah.glm53-native-bf16-packed-kld-summary.v1",
            "student_label": "native-bf16",
            "measured_mean_kld": 0.01271159981725071,
            "teacher_receipt_sha256": "f" * 64,   # a DIFFERENT teacher
        }
        (tmpd / "quant.json").write_text(json.dumps(quant))
        (tmpd / "wrong-floor.json").write_text(json.dumps(wrong_floor))
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(tmpd / "wrong-floor.json")])
        out = buf.getvalue()
        check("cross-lane floor is REFUSED (rc 3)", rc == 3)
        check("refusal shows the actual subtraction -0.000327",
              "-0.000327" in out or "-0.000328" in out, out.strip()[-160:])
        check("refusal carries the canonical worked example (-0.000328)",
              "0.012384 - 0.012712 = -0.000328" in out)
        check("refusal names the same-lane arithmetic (+0.000878)",
              "0.012384 - 0.011506 = +0.000878" in out)

        print("\n[8] BF16-FLOOR.json (the analysis) is refused as a floor input")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(ROOT / "engines" / "BF16-FLOOR.json")])
        out = buf.getvalue()
        check("analysis-as-floor is REFUSED (rc 3)", rc == 3)
        check("refusal quotes cross_stack_floor_do_not_mix by name",
              "cross_stack_floor_do_not_mix" in out)
        check("refusal names the real floor summary path",
              "native-bf16-kld.json" in out)

        print("\n[9] the REAL floor summary yields the sealed attributables")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(ROOT / "engines" / "native-bf16-kld.json"),
                          "--out", str(tmpd / "attr.json")])
        out = buf.getvalue()
        check("K8 vs same-lane floor exits 0", rc == 0)
        attr = json.loads((tmpd / "attr.json").read_text())
        check("attributable == 0.0008782684041065674 (the sealed value, exact)",
              attr["attributable"] == 0.012384191023436866 - 0.011505922619330299,
              "%.18g" % attr["attributable"])
        check("output is structurally unsubmittable (not_submittable true, "
              "no measured_mean_kld key)",
              attr.get("not_submittable") is True and
              "measured_mean_kld" not in attr)

        print("\n[9b] same-teacher floor FORGERY: sealed teacher sha + "
              "native-bf16 label but a foreign/absent profile is refused")
        forged = {
            "schema": "malaiwah.glm53-native-bf16-packed-kld-summary.v1",
            "student_label": "native-bf16",
            "measured_mean_kld": 0.01271159981725071,   # the cross-stack value
            "teacher_receipt_sha256": FS.SEALED_STREAM_TEACHER,
        }
        (tmpd / "forged.json").write_text(json.dumps(forged))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(tmpd / "forged.json")])
        check("profile-less floor with the sealed teacher sha -> rc 3", rc == 3)
        check("refusal names the missing 'profile' field",
              "profile" in buf.getvalue())
        forged["profile"] = "native-bf16-vllm-crosscheck"   # a foreign lane
        (tmpd / "forged.json").write_text(json.dumps(forged))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(tmpd / "forged.json")])
        check("foreign-profile floor with the sealed teacher sha -> rc 3",
              rc == 3)
        check("refusal names the streaming lane's own profile",
              "native-bf16-stream" in buf.getvalue())

        print("\n[10] a claimed zero floor without T1 hash evidence is refused")
        fake_zero = {
            "schema": "malaiwah.glm53-native-bf16-packed-kld-summary.v1",
            "student_label": "native-bf16",
            "profile": "native-bf16-stream",
            "measured_mean_kld": 0.0,
            "teacher_receipt_sha256": FS.SEALED_STREAM_TEACHER,
        }
        (tmpd / "zero.json").write_text(json.dumps(fake_zero))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(tmpd / "zero.json")])
        check("zero floor without logits_tensor_sha256 evidence -> rc 3",
              rc == 3)
        check("refusal quotes the T1 rule",
              "logits_tensor_sha256" in buf.getvalue())
        fake_zero["zero_floor_evidence"] = [
            {"evidence_kind": "logits_tensor_sha256",
             "detail": "25/25 per-window logit sha256 identical to teacher"}]
        (tmpd / "zero.json").write_text(json.dumps(fake_zero))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = FS.main(["attributable", "--quant-summary", str(tmpd / "quant.json"),
                          "--floor-summary", str(tmpd / "zero.json"),
                          "--out", str(tmpd / "attr0.json")])
        attr0 = json.loads((tmpd / "attr0.json").read_text())
        check("with T1 evidence: floor 0 accepted, attributable == panel mean",
              rc == 0 and attr0["attributable"] == quant["measured_mean_kld"]
              and attr0["zero_floor"] is True)

    _ = con
    print("\n" + "-" * 72)
    print("selftest_stats: %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
