#!/usr/bin/env python3
"""Offline regression tests for k6/tools/k6_kld_report.py.  No GPU, no network, no
`quant_pipeline` checkout.

`selftest_offline.py` needs a real quant_pipeline tree, which a public clone does not
have (`k6/.patchwork/*/runtime/src/quant_pipeline` ships without an `__init__.py`), so
the aggregation and provenance logic in the summary branch had NO test that runs on this
machine at all.  Every defect below was live in a tool that produces published numbers.

The trick that makes this cheap: `_measure_run` RESUMES from an existing
`kld-report.json` without recomputing, so a run directory holding a hand-written report
exercises the whole aggregation path with no logits and no torch.  The few
`quant_pipeline` symbols the module imports at call time are stubbed.

  NUM-01  a resumed report measured against a DIFFERENT teacher must refuse
  NUM-02  the headline of an N-run summary is the aggregate, not run 1
  NUM-03  two run dirs holding ONE capture are not two cold runs
  NUM-06  the summary carries per_window, or a published row can never be rescoped
  NUM-07  no profile claims a storage layout it does not have
  NUM-09  a resumed report scored at a different position_block must refuse
  NUM-13  --five-run-out must not be accepted and silently ignored
  NUM-14  a malformed sibling receipt must not crash the comparison table
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PASS: list = []
FAIL: list = []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name if condition else (name, detail))
    print("  %s  %s%s" % ("PASS" if condition else "FAIL", name,
                          ("  -- " + detail) if (detail and not condition) else ""))


def _stub_pipeline(root):
    """The minimum of `quant_pipeline` that k6_kld_report imports at call time."""
    import hashlib

    def canonical_json(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    def sha256_bytes(b):
        return hashlib.sha256(b).hexdigest()

    def sha256_file(p):
        return hashlib.sha256(open(str(p), "rb").read()).hexdigest()

    def write_json(p, obj):
        with open(str(p), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)

    def summarize(v):
        import statistics
        v = list(map(float, v))
        return {"count": len(v), "mean": statistics.fmean(v) if v else 0.0,
                "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "cvar95": 0.0, "max": 0.0}

    def load_capture_receipt(path, *a, **k):
        return json.loads(open(str(path), encoding="utf-8").read())

    for name in ("quant_pipeline", "quant_pipeline.core", "quant_pipeline.core.artifacts",
                 "quant_pipeline.evaluation", "quant_pipeline.evaluation.glm53_logits",
                 "quant_pipeline.publication",
                 "quant_pipeline.publication.glm53_k6_postmtp"):
        sys.modules.setdefault(name, types.ModuleType(name))
    art = sys.modules["quant_pipeline.core.artifacts"]
    def atomic_write(path, data, **kw):
        with open(str(path), "wb" if isinstance(data, bytes) else "w") as fh:
            fh.write(data)

    art.canonical_json, art.sha256_bytes = canonical_json, sha256_bytes
    art.sha256_file, art.write_json = sha256_file, write_json
    art.atomic_write = atomic_write
    art.__spec__ = None
    lg = sys.modules["quant_pipeline.evaluation.glm53_logits"]
    lg.summarize, lg.load_capture_receipt = summarize, load_capture_receipt
    lg.CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"
    lg.sealed_json = lambda p, *a, **k: json.loads(open(str(p), encoding="utf-8").read())


def _report(teacher_sha, panel_sha, mean, tokenwise, student_sha, label, block=16):
    per_window = [{"window_id": "final-%04d" % i, "document_id": "doc-%d" % i,
                   "domain": "axis1_general", "role": "final",
                   "summary": {"count": 2047, "mean": mean, "std": 0.19, "p50": 0.0,
                               "p95": 0.0, "p99": 0.0, "cvar95": 0.0, "max": 1.0}}
                  for i in range(25)]
    return {"schema": "quant-pipeline.glm53-packed-student-kld.v2",
            "teacher_receipt_sha256": teacher_sha,
            "token_panel_receipt_sha256": panel_sha,
            "student_receipt_sha256": student_sha,
            "student_label": label,
            "summary": {"mean": mean, "count": 51175},
            "per_window": per_window,
            "per_domain": {"axis1_general": {"count": 51175, "mean": mean}},
            "qualification_window_count": 25,
            "position_block": block,
            "top1_agreement": 0.99,
            "tokenwise_kld_sha256": tokenwise}


def main():
    _stub_pipeline(HERE)
    sys.path.insert(0, HERE)
    import k6_kld_report as K

    tmp = tempfile.mkdtemp(prefix="kld-report-selftest-")
    try:
        T1, T2 = "a" * 64, "b" * 64
        PANEL = "p" * 64
        teacher = {"receipt_sha256": T1, "token_panel_receipt_sha256": PANEL,
                   "vocab_size": 154880, "backend_identity_sha256": "x" * 64,
                   "logit_files": []}

        def run_dir(name, **kw):
            d = os.path.join(tmp, name)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "kld-report.json"), "w", encoding="utf-8") as fh:
                json.dump(_report(**kw), fh)
            return d

        print("\n== NUM-01: a resumed report is bound to the teacher on the command line ==")
        d = run_dir("r1", teacher_sha=T1, panel_sha=PANEL, mean=0.0137, tokenwise="d" * 64,
                    student_sha="1" * 64, label="uniform-k6")
        try:
            K._measure_run(run_dir=__import__("pathlib").Path(d), teacher=teacher,
                           student_label="uniform-k6", chunk_positions=16, device="cpu")
            ok, detail = True, ""
        except SystemExit as exc:
            ok, detail = False, "matching teacher was REFUSED: %s" % exc
        check("NUM-01  a matching teacher still resumes", ok, detail)

        other = dict(teacher, receipt_sha256=T2)
        refused = _stderr_of(lambda: K._measure_run(
            run_dir=__import__("pathlib").Path(d), teacher=other,
            student_label="uniform-k6", chunk_positions=16, device="cpu"))
        check("NUM-01  a DIFFERENT teacher refuses instead of resuming",
              refused is not None and "DIFFERENT reference" in (refused or ""),
              "resumed silently -- the summary would publish the new teacher's sha over "
              "means computed against the old one" if refused is None else "")

        print("\n== NUM-09: the position_block that can change the sealed digest ==")
        refused = _stderr_of(lambda: K._measure_run(
            run_dir=__import__("pathlib").Path(d), teacher=teacher,
            student_label="uniform-k6", chunk_positions=512, device="cpu"))
        check("NUM-09  a different position_block refuses instead of resuming",
              refused is not None and "position_block" in (refused or ""),
              "resumed" if refused is None else "")

        legacy = os.path.join(tmp, "legacy")
        os.makedirs(legacy, exist_ok=True)
        rep = _report(T1, PANEL, 0.0137, "d" * 64, "9" * 64, "uniform-k6")
        rep.pop("position_block")
        with open(os.path.join(legacy, "kld-report.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh)
        ok = True
        try:
            K._measure_run(run_dir=__import__("pathlib").Path(legacy), teacher=teacher,
                           student_label="uniform-k6", chunk_positions=512, device="cpu")
        except SystemExit:
            ok = False
        check("NUM-09  a report written before the field still resumes", ok,
              "a missing value is unknown, not a mismatch")

        print("\n== NUM-02 / NUM-03 / NUM-06: the summary branch ==")
        a = run_dir("a", teacher_sha=T1, panel_sha=PANEL, mean=0.010, tokenwise="1" * 64,
                    student_sha="a" * 64, label="uniform-k8")
        b = run_dir("b", teacher_sha=T1, panel_sha=PANEL, mean=0.020, tokenwise="2" * 64,
                    student_sha="b" * 64, label="uniform-k8")
        out = os.path.join(tmp, "k8-summary.json")

        def cli(*argv):
            return subprocess.run(
                [sys.executable, os.path.join(HERE, "k6_kld_report.py")] + list(argv),
                capture_output=True, text=True)

        # driven in-process: the CLI would need a real teacher tree
        summary = K_summary(K, teacher, [a, b], "uniform-k8", out)
        check("NUM-02  the headline is the mean of run means, not run 1",
              summary and abs(summary["measured_mean_kld"] - 0.015) < 1e-15,
              "got %r (run 1 is 0.010)" % (summary or {}).get("measured_mean_kld"))
        check("NUM-02  the spread of a nondeterministic set is published",
              summary and abs(summary.get("run_mean_spread", 0) - 0.010) < 1e-15,
              str((summary or {}).get("run_mean_spread")))
        check("NUM-06  the summary carries per_window",
              bool(summary and summary.get("per_window")
                   and len(summary["per_window"]) == 25),
              "absent: a published row with no per_window can never be rescoped without "
              "a GPU, which is why the streaming BF16 floor and Dione Q4 have no CI")
        check("NUM-06  and says which run it describes when the runs disagree",
              bool(summary and "run-1 ONLY" in (summary.get("per_window_source") or "")),
              str((summary or {}).get("per_window_source")))
        check("NUM-06  cold_run_deviation does not contradict itself",
              bool(summary and "not 5" in summary["cold_run_deviation"]),
              summary and summary["cold_run_deviation"])

        dup = K_summary(K, teacher, [a, a], "uniform-k8", out, expect_fail=True)
        check("NUM-03  the same run directory twice is refused",
              dup is not None and "same capture directory" in dup, str(dup))

        clone = os.path.join(tmp, "a-clone")
        shutil.rmtree(clone, ignore_errors=True)
        shutil.copytree(a, clone)
        dup2 = K_summary(K, teacher, [a, clone], "uniform-k8", out, expect_fail=True)
        check("NUM-03  a copied run directory is refused (same capture receipt)",
              dup2 is not None and "distinct cold captures" in dup2, str(dup2))

        print("\n== NUM-07: no profile claims a storage layout it does not have ==")
        for profile, want_absent in (("k6-stream", "tp4"), ("k8", "tp4"),
                                     ("k6k8", "tp4")):
            label = K._profile_storage_label(profile)
            check("NUM-07  %-12s does not claim TP4 slicing" % profile,
                  want_absent not in label, label)
        check("NUM-07  dione DOES claim TP4 (it really is sliced)",
              K._profile_storage_label("dione-q4") == "dione-q4-tp4",
              K._profile_storage_label("dione-q4"))
        unknown = _stderr_of(lambda: K._profile_storage_label("some-new-profile"))
        check("NUM-07  an unknown profile refuses rather than inheriting -tp4",
              unknown is not None and "STORAGE label" in (unknown or ""), str(unknown))

        print("\n== NUM-13: an evidence flag must not be silently ignored ==")
        r = cli("--profile", "k8", "--teacher", "/nonexistent", "--runs", "/nonexistent",
                "--out", os.path.join(tmp, "x.json"), "--five-run-out",
                os.path.join(tmp, "five.json"))
        check("NUM-13  --five-run-out on a non-k6 profile refuses",
              r.returncode != 0 and "five-run-out is implemented only" in (r.stderr + r.stdout),
              (r.stderr + r.stdout)[:120])

        print("\n== NUM-14: a malformed sibling must not crash the comparison table ==")
        rd = os.path.join(tmp, "receipts")
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "k8-packed-kld.json"), "w", encoding="utf-8") as fh:
            json.dump({"schema": "x", "quality_gate_passed": True}, fh)   # no mean
        with open(os.path.join(rd, "k6k8-packed-kld.json"), "w", encoding="utf-8") as fh:
            fh.write('{"measured_mean_kld": 0.0272')                      # truncated
        crashed = None
        try:
            K._comparison_table(__import__("pathlib").Path(os.path.join(tmp, "t.md")),
                                0.020615, 0.024555,
                                __import__("pathlib").Path(rd))
        except Exception as exc:                                          # noqa: BLE001
            crashed = "%s: %s" % (type(exc).__name__, exc)
        check("NUM-14  a receipt with no measured_mean_kld is skipped, not fatal",
              crashed is None, str(crashed))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nselftest_kld_report_offline: %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


def _stderr_of(fn):
    """`_fail` PRINTS to stderr and RETURNS a SystemExit for the caller to raise, so the
    message is never in the exception. Capture the stream."""
    import io, contextlib
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn()
    except SystemExit:
        pass
    except Exception as exc:                                              # noqa: BLE001
        err.write(str(exc))
    return err.getvalue() or None


def K_summary(K, teacher, run_dirs, label, out, expect_fail=False):
    """Drive main()'s summary branch in-process with argv, capturing the refusal."""
    import pathlib
    argv = ["k6_kld_report.py", "--profile", "k8", "--teacher", "/unused",
            "--runs"] + [str(d) for d in run_dirs] + ["--out", out, "--device", "cpu"]
    import io, contextlib
    saved_argv, saved_find = sys.argv, K._find_teacher_receipt
    sys.argv = argv
    # _find_teacher_receipt returns the PATH; main() loads the receipt from it.
    tpath = pathlib.Path(out).parent / "teacher-capture-receipt.json"
    with open(tpath, "w", encoding="utf-8") as fh:
        json.dump(teacher, fh)
    K._find_teacher_receipt = lambda root: tpath
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            K.main()
    except SystemExit:
        pass
    except Exception as exc:                                              # noqa: BLE001
        err.write(str(exc))
    finally:
        sys.argv, K._find_teacher_receipt = saved_argv, saved_find
    if expect_fail:
        return err.getvalue() or None
    try:
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
    except IOError:
        return None


if __name__ == "__main__":
    sys.exit(main())
