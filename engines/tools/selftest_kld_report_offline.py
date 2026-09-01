#!/usr/bin/env python3
"""Offline regression tests for engines/tools/kld_report.py.  No GPU, no network, no
`quant_pipeline` checkout.

`selftest_offline.py` needs a real quant_pipeline tree, which a public clone does not
have (`engines/.patchwork/*/runtime/src/quant_pipeline` ships without an `__init__.py`), so
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
  NUM-15  the provenance branch dispatches on the capture SURFACE, not on a
          profile-name prefix (LESSON 48 recurring on a fourth profile)
  NUM-17  per-window top-1 integer counts make subset rescoring exact
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


def _declared_capture_profiles() -> dict:
    """{profile: surface family} read from stream_score's OWN tables.

    stream_score.py is the file that decides which profiles exist; this test
    asserts kld_report can describe every one of them.  Importing the tables
    instead of restating them is what makes that assertion stay true when a
    profile is added.  stream_score imports heavy optionals at call time only,
    so the module-level tables are readable without torch -- but if that ever
    changes, parse them out of the source rather than skipping the check.
    """
    import ast

    src = open(os.path.join(HERE, "stream_score.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    tables = {"EXL3HF_PROFILES": "exl3hf", "TR3_PROFILES": "tr3",
              "DIONE_PROFILES": "dione"}
    out: dict = {}
    seen = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, "id", None)
            if name in tables:
                seen.add(name)
                for key in ast.literal_eval(node.value):
                    out[key] = tables[name]
    missing = set(tables) - seen
    if missing:
        raise AssertionError("stream_score.py no longer defines %s -- this "
                             "test's derivation is stale, fix it rather than "
                             "letting it pass vacuously" % sorted(missing))
    return out
PASS: list = []
FAIL: list = []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name if condition else (name, detail))
    print("  %s  %s%s" % ("PASS" if condition else "FAIL", name,
                          ("  -- " + detail) if (detail and not condition) else ""))


def _stub_pipeline(root):
    """The minimum of `quant_pipeline` that kld_report imports at call time."""
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
                               "p95": 0.0, "p99": 0.0, "cvar95": 0.0, "max": 1.0},
                   "top1_matches": 2026, "positions": 2047}
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
            "top1_agreement": 2026 / 2047,
            "tokenwise_kld_sha256": tokenwise}


def _measured_top1_report(K, root, teacher_sha, panel_sha):
    """Exercise the compute path cheaply: two tiny windows, no torch or logits."""
    from pathlib import Path

    import numpy as np

    run_dir = Path(root) / "top1-counts"
    run_dir.mkdir()
    specs = (("final-0000", 3), ("final-0001", 2))

    def rows(side):
        return [
            {
                "window_id": window_id,
                "document_id": "doc-" + window_id,
                "domain": "axis1_general",
                "role": "final",
                "token_ids_sha256": window_id + "-tokens",
                "attention_mask_sha256": window_id + "-mask",
                "prediction_positions": count,
                "path": str(run_dir / ("%s-%s.safetensors" % (side, window_id))),
                "sha256": side[0] * 64,
            }
            for window_id, count in specs
        ]

    teacher = {
        "receipt_sha256": teacher_sha,
        "token_panel_receipt_sha256": panel_sha,
        "vocab_size": 3,
        "backend_identity_sha256": "t" * 64,
        "logit_files": rows("teacher"),
    }
    student = {
        "schema": "quant-pipeline.glm53-logit-capture.v1",
        "receipt_sha256": "s" * 64,
        "token_panel_receipt_sha256": panel_sha,
        "vocab_size": 3,
        "runtime_reader_sha256": "r" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "backend_identity_sha256": "b" * 64,
        "logit_files": rows("student"),
    }
    with open(run_dir / "capture-receipt.json", "w", encoding="utf-8") as fh:
        json.dump(student, fh)

    saved = (
        K.FINAL_WINDOW_IDS,
        K.FINAL_PREDICTION_POSITIONS,
        K._resolve_teacher_paths,
        K._load_slice,
        K._token_kld,
    )
    missing = object()
    saved_torch = sys.modules.get("torch", missing)
    fake_torch = types.ModuleType("torch")
    fake_torch.get_num_threads = lambda: 1

    def fake_load(path, start, stop):
        marker = 1.0 if "final-0001" in str(path) else 0.0
        return np.full((stop - start, 3), marker, dtype=np.float32)

    def fake_kld(teacher_logits, student_logits, device):
        del student_logits, device
        positions = teacher_logits.shape[0]
        matches = 0 if teacher_logits[0, 0] else positions
        values = np.full(positions, 0.2 + teacher_logits[0, 0], dtype=np.float64)
        return values, int(matches)

    K.FINAL_WINDOW_IDS = tuple(window_id for window_id, _ in specs)
    K.FINAL_PREDICTION_POSITIONS = sum(count for _, count in specs)
    K._resolve_teacher_paths = lambda mapped, root_, sha: {
        window_id: Path(row["path"]) for window_id, row in mapped.items()
    }
    K._load_slice = fake_load
    K._token_kld = fake_kld
    sys.modules["torch"] = fake_torch
    try:
        report_path = K._measure_run(
            run_dir=run_dir,
            teacher=teacher,
            student_label="uniform-k8",
            chunk_positions=2,
            device="cpu",
        )
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        (
            K.FINAL_WINDOW_IDS,
            K.FINAL_PREDICTION_POSITIONS,
            K._resolve_teacher_paths,
            K._load_slice,
            K._token_kld,
        ) = saved
        if saved_torch is missing:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch


def main():
    _stub_pipeline(HERE)
    sys.path.insert(0, HERE)
    import kld_report as K

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

        print("\n== NUM-17: exact per-window top-1 subset rescoring ==")
        measured = _measured_top1_report(K, tmp, T1, PANEL)
        measured_windows = measured["per_window"]
        counts = [
            (row.get("top1_matches"), row.get("positions"))
            for row in measured_windows
        ]
        check("NUM-17  compute emits per-window top-1 matches and positions",
              counts == [(3, 3), (0, 2)], str(counts))
        subset_top1 = None
        if all(isinstance(value, int) for pair in counts for value in pair):
            subset_top1 = sum(row["top1_matches"] for row in measured_windows[1:]) / sum(
                row["positions"] for row in measured_windows[1:])
        check("NUM-17  a window subset recomputes top-1 exactly from integers",
              subset_top1 == 0.0, str(subset_top1))
        check("NUM-17  panel top-1 equals the ratio of emitted window counts",
              measured["top1_agreement"] == 3 / 5,
              str(measured["top1_agreement"]))

        print("\n== NUM-02 / NUM-03 / NUM-06: the summary branch ==")
        a = run_dir("a", teacher_sha=T1, panel_sha=PANEL, mean=0.010, tokenwise="1" * 64,
                    student_sha="a" * 64, label="uniform-k8")
        b = run_dir("b", teacher_sha=T1, panel_sha=PANEL, mean=0.020, tokenwise="2" * 64,
                    student_sha="b" * 64, label="uniform-k8")
        out = os.path.join(tmp, "k8-summary.json")

        def cli(*argv):
            return subprocess.run(
                [sys.executable, os.path.join(HERE, "kld_report.py")] + list(argv),
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
        check("NUM-06  the summary preserves exact per-window top-1 counts",
              bool(summary and all(
                  row.get("top1_matches") == 2026
                  and row.get("positions") == row["summary"]["count"]
                  for row in summary.get("per_window", []))),
              "the run report had counts but the published summary dropped them")
        check("NUM-06  and says which run it describes when the runs disagree",
              bool(summary and "run-1 ONLY" in (summary.get("per_window_source") or "")),
              str((summary or {}).get("per_window_source")))

        c = run_dir("c", teacher_sha=T1, panel_sha=PANEL, mean=0.010,
                    tokenwise="3" * 64, student_sha="c" * 64, label="uniform-k8")
        d = run_dir("d", teacher_sha=T1, panel_sha=PANEL, mean=0.010,
                    tokenwise="3" * 64, student_sha="d" * 64, label="uniform-k8")
        e = run_dir("e", teacher_sha=T1, panel_sha=PANEL, mean=0.010,
                    tokenwise="3" * 64, student_sha="e" * 64, label="uniform-k8")
        d_path = os.path.join(d, "kld-report.json")
        with open(d_path, encoding="utf-8") as fh:
            d_report = json.load(fh)
        d_report["per_window"][0]["top1_matches"] -= 1
        check("NUM-17  contradictory panel and window top-1 counts are invalid",
              K._per_window_top1_signature(d_report) is None,
              "the provenance gate accepted counts that do not reproduce top1_agreement")
        with open(d_path, "w", encoding="utf-8") as fh:
            json.dump(d_report, fh)
        counts_differ = K_summary(
            K, teacher, [c, d], "uniform-k8",
            os.path.join(tmp, "top1-counts-differ.json"))
        check("NUM-17  a KLD digest cannot claim differing top-1 counts agree",
              bool(counts_differ and counts_differ["bitwise_deterministic"]
                   and "run-1 ONLY" in counts_differ["per_window_source"]
                   and "counts are absent or differ" in counts_differ["per_window_source"]),
              str((counts_differ or {}).get("per_window_source")))
        e_path = os.path.join(e, "kld-report.json")
        with open(e_path, encoding="utf-8") as fh:
            e_report = json.load(fh)
        e_report["per_window"].reverse()
        with open(e_path, "w", encoding="utf-8") as fh:
            json.dump(e_report, fh)
        counts_match = K_summary(
            K, teacher, [c, e], "uniform-k8",
            os.path.join(tmp, "top1-counts-match.json"))
        check("NUM-17  exact cross-run counts are order-independent",
              bool(counts_match and "top-1 counts identical across runs"
                   in counts_match["per_window_source"]),
              str((counts_match or {}).get("per_window_source")))
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

        print("\n== NUM-15: the provenance branch dispatches on SURFACE, not name prefix ==")
        # LESSON 48, one profile later. `startswith(("dione","turbo"))` answered
        # correctly for the three profiles that existed when it was written and
        # would have answered "not a third-party artifact" for vcruz-k2-2bpw --
        # an exl3hf capture whose headline receipt would then carry no
        # artifact_repo, no artifact_revision, no codebook and no
        # seal_disclosure. Probe both outcomes, not just the one that passes.
        # The pairs are DERIVED from stream_score's own profile tables, not
        # retyped here.  A hand-written list is a list that drifts: adding a
        # profile to stream_score and forgetting this file used to leave the new
        # profile with no NUM-15 coverage at all, and the suite still went green.
        # Deriving it means a profile that stream_score can capture but
        # kld_report cannot describe fails HERE, before it can seal a receipt.
        for profile, want in sorted(_declared_capture_profiles().items()):
            got = K._profile_surface_family(profile)
            check("NUM-15  %-14s -> surface %s" % (profile, want), got == want, str(got))
        check("NUM-15  a lane-native profile has no surface family",
              K._profile_surface_family("k6") is None
              and K._profile_surface_family("native-bf16") is None)
        # the prefix probe and the declared map must not be confused for each
        # other: assert the map disagrees with the old prefix rule exactly where
        # the old rule was wrong.
        check("NUM-15  the old prefix rule would have missed vcruz-k2-2bpw",
              not "vcruz-k2-2bpw".startswith(("dione", "turbo"))
              and K._profile_surface_family("vcruz-k2-2bpw") == "exl3hf")
        # every profile stream_score can capture a third-party artifact with must
        # be declared here, or its provenance silently vanishes
        for profile in sorted(_declared_capture_profiles()):
            check("NUM-15  %-14s has a storage label too" % profile,
                  isinstance(K._profile_storage_label(profile), str))

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
    argv = ["kld_report.py", "--profile", "k8", "--teacher", "/unused",
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
