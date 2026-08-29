#!/usr/bin/env python3
"""kld-preview -- score capture trees locally, honestly labeled as previews.

    bin/kld-preview --teacher DIR --student DIR --out preview.json
    bin/kld-preview --teacher DIR --student DIR --student2 DIR2 --out d.json

Needs torch + safetensors (run under FIDELITY_PYTHON); deliberately needs
NEITHER quant_pipeline NOR transformers -- it re-implements only receipt
reading and the exact _token_kld math (fp64 log_softmax, teacher->student),
with the fp64 accumulation PINNED TO CPU because MPS cannot represent float64
at all (torch raises TypeError).

Mode is auto-detected from the student capture's schema:
  * sealed full capture (quant-pipeline.glm53-logit-capture.v1) -> CENSUS:
    every position scored exactly; the panel mean is exact FOR THIS LANE and
    the receipt is preview-labeled ONLY because the lane differs from the
    teacher's (malaiwah.glm53-census-kld-preview.v1).  This is the default
    local path: scoring costs ~0.15 ms/position on CPU (~8 s/panel), so once
    a window's logits exist there is no reason to sample locally.
  * preview capture (malaiwah.glm53-logit-capture-preview.v1) -> SAMPLED:
    teacher rows are sliced at the student's stored position_indices (no full
    1.27 GB reads); the stratified estimator + FPC from fidelity/previewstats
    produce the estimate, and the quoted CI is the WIDER of z and bootstrap.

PANEL-ESTIMATE GATE: no panel mean is emitted unless ALL panel windows
contributed -- per-window scatter (sd 1.73e-3) exceeds the K6-vs-K8 effect
(1.22e-3), so window subsets get per-window diagnostics only (lessons 28/29).

Sample sizes (sigma_w = 0.05 DESIGN number; the tool reports the ACHIEVED CI
from its own s_j, never the plan):
%s

%s

Preview receipts are structurally unsubmittable: schema contains "-preview.",
the headline field is preview_panel_mean_estimate (never measured_mean_kld),
not_submittable: true, and no submission_schema key exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import previewstats as PS                    # noqa: E402
from fidelity.common import Console, sha256_file, write_json  # noqa: E402

__doc__ = __doc__ % (
    "\n".join("  %-18s %s" % row for row in PS.SAMPLE_SIZE_TABLE),
    PS.DELTA_HONESTY_TEXT,
)

EXIT_OK, EXIT_REFUSED = 0, 3

SEALED_CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"
PREVIEW_CAPTURE_SCHEMA = "malaiwah.glm53-logit-capture-preview.v1"

# The one same-lane CUDA floor that exists today, quoted in the disclosure
# when the teacher is the sealed EP8 one.
KNOWN_CUDA_STREAM_FLOOR = 0.011505922619330299

# The sealed EP8 teacher's receipt sha and its panel's true window count.
# windows_total normally comes from the teacher tree itself (a fixture panel
# may honestly have fewer windows), but when the teacher CLAIMS to be the
# sealed one, a truncated tree (fewer logit_files rows than the sealed 25)
# must not shrink the panel gate: pinning 25 here keeps a hand-trimmed
# sealed-teacher tree from turning a window subset into a "panel mean".
SEALED_STREAM_TEACHER_RECEIPT = (
    "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5")
SEALED_PANEL_WINDOWS = 25


def _panel_windows_total(teacher: Dict[str, Any], declared: int) -> int:
    if teacher.get("receipt_sha256") == SEALED_STREAM_TEACHER_RECEIPT:
        return max(declared, SEALED_PANEL_WINDOWS)
    return declared


class Refusal(RuntimeError):
    def __init__(self, reason: str, advice: Optional[List[str]] = None) -> None:
        self.reason, self.advice = reason, list(advice or [])
        super().__init__(reason)


def _find_capture_receipt(root: Path, *, role: Optional[str],
                          schemas: Tuple[str, ...]) -> Tuple[Path, Dict[str, Any]]:
    direct = root / "capture-receipt.json"
    candidates = [direct] if direct.is_file() else sorted(root.glob("**/*.json"))
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") not in schemas:
            continue
        if role is not None and doc.get("capture_role") != role:
            continue
        return path, doc
    raise Refusal(
        "no capture receipt with schema in %s%s under %s"
        % (list(schemas), (" and role %r" % role) if role else "", root))


def _rows(receipt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["window_id"]: row for row in receipt["logit_files"]}


def _resolve_row_path(row: Dict[str, Any], root: Path, con: Console,
                      side: str) -> Path:
    recorded = Path(row["path"])
    if recorded.is_file():
        return recorded
    fallback = root / "logits" / recorded.name
    if not fallback.is_file():
        raise Refusal(
            "%s logits for %s not found at the recorded path %s nor at the "
            "portable fallback %s" % (side, row["window_id"], recorded, fallback))
    digest = sha256_file(str(fallback))
    if digest != row["sha256"]:
        raise Refusal(
            "%s fallback %s has sha256 %s..., receipt row says %s... -- the "
            "remapped file is NOT the sealed row (content hash rules identity)"
            % (side, fallback, digest[:12], str(row["sha256"])[:12]))
    con.say("  %s path remapped: %s -> %s (sha256 verified)"
            % (side, row["window_id"], fallback))
    return fallback


def _gate_pair(teacher: Dict[str, Any], student: Dict[str, Any],
               label: str) -> None:
    if teacher.get("token_panel_receipt_sha256") != student.get("token_panel_receipt_sha256"):
        raise Refusal(
            "teacher and %s use different sealed token panels (%s... vs %s...)"
            % (label, str(teacher.get("token_panel_receipt_sha256"))[:12],
               str(student.get("token_panel_receipt_sha256"))[:12]))
    if teacher.get("vocab_size") != student.get("vocab_size"):
        raise Refusal("teacher and %s vocabularies differ (%s vs %s)"
                      % (label, teacher.get("vocab_size"), student.get("vocab_size")))
    trows, srows = _rows(teacher), _rows(student)
    if set(trows) != set(srows):
        raise Refusal(
            "teacher and %s window sets differ (teacher %d windows, %s %d; "
            "first missing: %s)" % (
                label, len(trows), label, len(srows),
                sorted(set(trows) ^ set(srows))[:3]))
    for window_id, left in trows.items():
        right = srows[window_id]
        for field in ("document_id", "token_ids_sha256", "attention_mask_sha256",
                      "role", "prediction_positions"):
            if left.get(field) != right.get(field):
                raise Refusal("%s relabels window %s field %s (%r vs %r)"
                              % (label, window_id, field,
                                 left.get(field), right.get(field)))


def _token_kld_cpu(teacher_logits, student_logits):
    """The exact sealed math: fp64 log_softmax, teacher->student, on CPU."""
    import torch

    t64 = teacher_logits.to(device="cpu", dtype=torch.float64)
    s64 = student_logits.to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(t64).all() or not torch.isfinite(s64).all():
        raise Refusal("teacher/student logits must be finite")
    tlp = torch.log_softmax(t64, dim=-1)
    slp = torch.log_softmax(s64, dim=-1)
    values = torch.sum(torch.exp(tlp) * (tlp - slp), dim=-1)
    matches = int((t64.argmax(-1) == s64.argmax(-1)).sum())
    return values, matches


def _lane_disclosure(teacher: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    same_lane = isinstance(teacher.get("teacher_provenance"), dict)
    if same_lane:
        floor_context = (
            "teacher is a SAME-LANE capture (%s); against it the capture "
            "lane's floor is 0 with T1 hash evidence -- but THIS scorer runs "
            "on a different local lane whose floor against that teacher is "
            "unmeasured" % teacher["teacher_provenance"].get("teacher_label"))
    else:
        floor_context = (
            "local lane floor unmeasured against this teacher; the known "
            "same-lane CUDA streaming floor is %.18g nats (k6/native-bf16-"
            "kld.json) and must NOT be subtracted from local-lane numbers"
            % KNOWN_CUDA_STREAM_FLOOR)
    return {
        "lane": "local-preview",
        "device": "cpu (fp64 accumulation pinned to cpu: MPS cannot hold float64)",
        "torch": torch.__version__,
        "transformers": None,
        "floor_context": floor_context,
    }


# --------------------------------------------------------------------------
# CENSUS mode
# --------------------------------------------------------------------------


def _iter_full(path: Path, chunk: int):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        sl = handle.get_slice("logits")
        n = sl.get_shape()[0]
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            yield start, stop, sl[start:stop]


def score_census(teacher: Dict[str, Any], teacher_root: Path,
                 student: Dict[str, Any], student_root: Path,
                 args, con: Console) -> Dict[str, Any]:
    con.say("mode: CENSUS (full positions, exact for this lane)")
    trows, srows = _rows(teacher), _rows(student)
    per_window: Dict[str, Dict[str, Any]] = {}
    samples: Dict[str, List[float]] = {}
    n_positions: Dict[str, int] = {}
    matches_total, count_total = 0, 0
    started = time.monotonic()
    for window_id in sorted(trows):
        tpath = _resolve_row_path(trows[window_id], teacher_root, con, "teacher")
        spath = _resolve_row_path(srows[window_id], student_root, con, "student")
        count = int(trows[window_id]["prediction_positions"])
        values: List[float] = []
        titer = _iter_full(tpath, args.chunk_positions)
        siter = _iter_full(spath, args.chunk_positions)
        for (tstart, tstop, tchunk), (sstart, sstop, schunk) in zip(titer, siter):
            if (tstart, tstop) != (sstart, sstop) or tchunk.shape != schunk.shape:
                raise Refusal("logit geometry mismatch in %s" % window_id)
            vals, m = _token_kld_cpu(tchunk, schunk)
            values.extend(float(v) for v in vals)
            matches_total += m
        if len(values) != count:
            raise Refusal("window %s scored %d of %d positions"
                          % (window_id, len(values), count))
        count_total += count
        samples[window_id] = values
        n_positions[window_id] = count
        mean = sum(values) / len(values)
        per_window[window_id] = {"mean": mean, "positions": count,
                                 "max": max(values)}
        con.say("  %s: mean %.6f" % (window_id, mean))
    windows_total = _panel_windows_total(teacher, len(trows))
    panel_mean: Optional[float] = None
    gate_error: Optional[str] = None
    try:
        PS.require_all_windows(len(per_window), windows_total)
        panel_mean = PS.stratified_mean(samples, n_positions)
        con.say("  panel mean (exact for this lane): %.18g   top-1 agreement "
                "%.4f" % (panel_mean, matches_total / float(count_total)))
    except PS.PanelGateError as exc:
        # A truncated tree of a full panel (e.g. a hand-trimmed sealed
        # teacher): per-window diagnostics only, exactly like sampled mode.
        gate_error = str(exc)
        con.say("")
        con.say(gate_error)
    con.say("  scored %d positions in %.1f s"
            % (count_total, time.monotonic() - started))
    extra: Dict[str, Any] = {
        "top1_agreement": matches_total / float(count_total),
        "student_receipt_sha256": student.get("receipt_sha256"),
        "student_label": student.get("student_label"),
        "scored_positions": count_total}
    if gate_error:
        extra["panel_estimate_refused"] = gate_error
    return PS.build_preview_receipt(
        kind="census", per_window=per_window,
        windows_total=windows_total,
        panel_estimate=panel_mean, ci95_z=None, ci95_bootstrap=None,
        sampling_design=None,
        tail=PS.tail_disclosure(samples, n_positions),
        lane_disclosure=_lane_disclosure(teacher),
        teacher_receipt_sha256=teacher.get("receipt_sha256"),
        extra=extra)


# --------------------------------------------------------------------------
# SAMPLED mode
# --------------------------------------------------------------------------


def _read_sampled(path: Path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        logits = handle.get_tensor("logits")
        indices = handle.get_tensor("position_indices")
    return logits, [int(i) for i in indices]


def _teacher_rows_at(path: Path, indices: List[int]):
    import torch
    from safetensors import safe_open

    rows = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        sl = handle.get_slice("logits")
        n = sl.get_shape()[0]
        for i in indices:
            if not (0 <= i < n):
                raise Refusal("position index %d out of range 0..%d" % (i, n - 1))
            rows.append(sl[i:i + 1])
    return torch.cat(rows, dim=0)


def _score_sampled_windows(teacher: Dict[str, Any], teacher_root: Path,
                           student: Dict[str, Any], student_root: Path,
                           con: Console, side: str):
    trows, srows = _rows(teacher), _rows(student)
    samples: Dict[str, List[float]] = {}
    indices_by_window: Dict[str, List[int]] = {}
    n_positions: Dict[str, int] = {}
    for window_id in sorted(srows):
        tpath = _resolve_row_path(trows[window_id], teacher_root, con, "teacher")
        spath = _resolve_row_path(srows[window_id], student_root, con, side)
        logits, indices = _read_sampled(spath)
        teacher_rows = _teacher_rows_at(tpath, indices)
        if teacher_rows.shape != logits.shape:
            raise Refusal("sampled geometry mismatch in %s (%s vs %s)"
                          % (window_id, tuple(teacher_rows.shape),
                             tuple(logits.shape)))
        values, _ = _token_kld_cpu(teacher_rows, logits)
        samples[window_id] = [float(v) for v in values]
        indices_by_window[window_id] = indices
        n_positions[window_id] = int(trows[window_id]["prediction_positions"])
    return samples, indices_by_window, n_positions


def score_sampled(teacher: Dict[str, Any], teacher_root: Path,
                  student: Dict[str, Any], student_root: Path,
                  student2: Optional[Dict[str, Any]], student2_root: Optional[Path],
                  args, con: Console) -> Dict[str, Any]:
    con.say("mode: SAMPLED (stratified positions; teacher sliced at the "
            "student's stored indices)")
    design = student.get("sampling_design") or {}
    windows_total = _panel_windows_total(
        teacher, int(design.get("windows_total") or len(_rows(teacher))))
    samples, indices, n_positions = _score_sampled_windows(
        teacher, teacher_root, student, student_root, con, "student")
    per_window = {}
    for w, xs in samples.items():
        per_window[w] = {"mean": sum(xs) / len(xs), "sampled_positions": len(xs),
                         "max": max(xs)}
        con.say("  %s: sampled mean %.6f (m=%d of %d)"
                % (w, per_window[w]["mean"], len(xs), n_positions[w]))

    panel_estimate = None
    ci_z = None
    boot = None
    tail = PS.tail_disclosure(samples, n_positions)
    if (tail.get("top3_share_of_estimate") or 0) > 0.15:
        con.warn(
            "TAIL-DOMINATED SAMPLE: the top 3 sampled positions carry %.0f%% "
            "of the estimate (max sampled value %.4g). On heavy tails the "
            "estimate and its SE are positively correlated, so BOTH intervals "
            "below are likely anti-conservative -- raise m (the remedy that "
            "works) before trusting a close comparison."
            % (100 * tail["top3_share_of_estimate"], tail["max_sampled_value"]))
    gate_error: Optional[str] = None
    try:
        PS.require_all_windows(len(samples), windows_total)
        panel_estimate = PS.stratified_mean(samples, n_positions)
        var = PS.stratified_variance(samples, n_positions)
        z_lo, z_hi = PS.z_interval(panel_estimate, var)
        ci_z = {"low": z_lo, "high": z_hi}
        boot = PS.stratified_position_bootstrap(samples, n_positions,
                                                args.bootstrap_b, args.seed)
        quoted = PS.wider_of((z_lo, z_hi), boot)
        con.say("  panel estimate %.6f   quoted CI95 [%.6f, %.6f] (%s of z/bootstrap)"
                % (panel_estimate, quoted["low"], quoted["high"], quoted["source"]))
        con.say("  ACHIEVED half-width %.3e -- from this sample's own s_j, not "
                "the planning table" % ((quoted["high"] - quoted["low"]) / 2.0))
    except PS.PanelGateError as exc:
        gate_error = str(exc)
        con.say("")
        con.say(gate_error)

    extra: Dict[str, Any] = {
        "student_receipt_sha256": student.get("receipt_sha256"),
        "student_label": student.get("student_label"),
        "sigma_hat_per_window": PS.sigma_hat_per_window(samples),
    }
    if gate_error:
        extra["panel_estimate_refused"] = gate_error
    if student2 is not None:
        con.say("  paired mode: --student2 present; requiring COMMON positions")
        samples2, indices2, _ = _score_sampled_windows(
            teacher, teacher_root, student2, student2_root, con, "student2")
        if set(indices) != set(indices2):
            raise Refusal(
                "paired preview deltas require identical window sets "
                "(student-only: %s; student2-only: %s)"
                % (sorted(set(indices) - set(indices2))[:3],
                   sorted(set(indices2) - set(indices))[:3]))
        for w in indices:
            if indices.get(w) != indices2.get(w):
                raise Refusal(
                    "paired preview deltas require common positions (same "
                    "--sample-seed); indices differ in window %s." % w)
        extra["paired_delta"] = PS.paired_delta(
            samples, samples2, n_positions,
            args.bootstrap_b, args.seed)
        extra["paired_delta"]["label_a"] = student.get("student_label")
        extra["paired_delta"]["label_b"] = student2.get("student_label")
        pd = extra["paired_delta"]
        con.say("  paired delta (student2 - student): %+.6e  quoted CI95 "
                "[%+.6e, %+.6e]" % (pd["delta_mean"],
                                    pd["quoted_interval"]["low"],
                                    pd["quoted_interval"]["high"]))
        con.say("  " + PS.DELTA_HONESTY_TEXT)
    sampling_design = dict(design) if design else {
        "scheme": "stratified-systematic (from capture)",
    }
    sampling_design["achieved_positions_per_window"] = {
        w: len(xs) for w, xs in samples.items()}
    return PS.build_preview_receipt(
        kind="sampled", per_window=per_window, windows_total=windows_total,
        panel_estimate=panel_estimate, ci95_z=ci_z, ci95_bootstrap=boot,
        sampling_design=sampling_design, tail=tail,
        lane_disclosure=_lane_disclosure(teacher),
        teacher_receipt_sha256=teacher.get("receipt_sha256"),
        extra=extra)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kld-preview", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--student2", type=Path,
                   help="second student for a paired common-position delta "
                        "(must share --sample-seed with --student)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunk-positions", type=int, default=512,
                   help="census-mode read chunk (512 selftest-proven vs the "
                        "sealed scorer's conservative 16)")
    p.add_argument("--bootstrap-b", type=int, default=None,
                   help="default 2000; 10000 when --student2 is given")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()
    if args.bootstrap_b is None:
        args.bootstrap_b = 10000 if args.student2 else 2000
    try:
        try:
            import torch                                   # noqa: F401
            import safetensors                             # noqa: F401
        except ImportError as exc:
            raise Refusal(
                "kld-preview needs torch + safetensors (missing: %s). Run it "
                "under FIDELITY_PYTHON (e.g. FIDELITY_PYTHON=/opt/homebrew/"
                "bin/python3.14 bin/kld-preview ...) or install them for this "
                "interpreter." % exc)
        con.say("fp64 accumulation pinned to cpu: MPS cannot hold float64")
        teacher_path, teacher = _find_capture_receipt(
            args.teacher.resolve(), role="bf16_teacher",
            schemas=(SEALED_CAPTURE_SCHEMA,))
        teacher_root = teacher_path.parent
        student_path, student = _find_capture_receipt(
            args.student.resolve(), role=None,
            schemas=(SEALED_CAPTURE_SCHEMA, PREVIEW_CAPTURE_SCHEMA))
        student_root = student_path.parent
        if student.get("capture_role") == "bf16_teacher":
            raise Refusal(
                "--student %s is itself a bf16_teacher capture; scoring a "
                "teacher against a teacher is the floor measurement, which "
                "needs the sealed scorer's determinism chain (or T1 hash "
                "identity -- see k6/SAME-LANE-TEACHER.md)" % args.student)
        _gate_pair(teacher, student, "student")
        sampled = student.get("schema") == PREVIEW_CAPTURE_SCHEMA
        student2 = student2_root = None
        if args.student2:
            s2_path, student2 = _find_capture_receipt(
                args.student2.resolve(), role=None,
                schemas=(PREVIEW_CAPTURE_SCHEMA,) if sampled
                else (SEALED_CAPTURE_SCHEMA,))
            student2_root = s2_path.parent
            _gate_pair(teacher, student2, "student2")
        if sampled:
            doc = score_sampled(teacher, teacher_root, student, student_root,
                                student2, student2_root, args, con)
        else:
            if student2 is not None:
                raise Refusal(
                    "--student2 with full-census captures: score each with "
                    "the sealed scorer and use bin/fidelity-stats "
                    "paired-delta on the two kld-report.json files -- the "
                    "full-census delta deserves the window-cluster CI, not "
                    "the position bootstrap.")
            doc = score_census(teacher, teacher_root, student, student_root,
                               args, con)
        write_json(str(args.out), doc)
        con.say("")
        con.say("PREVIEW receipt written: %s" % args.out)
        con.say("  schema %s -- structurally unsubmittable (no "
                "submission_schema, no measured_mean_kld, not_submittable: "
                "true); the sealed lane (engines.json 'streaming') is the "
                "submittable path." % doc["schema"])
        if args.json:
            print(json.dumps(doc, indent=2, sort_keys=True))
        return EXIT_OK
    except Refusal as exc:
        con.say("")
        con.say("REFUSED: %s" % exc.reason)
        for line in exc.advice:
            con.say("         %s" % line)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
