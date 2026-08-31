#!/usr/bin/env python3
"""fp64 tokenwise KLD vs the fp32 teacher + five-cold-run aggregation (K6/K6K8).

Adapted from brandonmusic's measure_glm53_packed_student_kld.py and
aggregate_glm53_five_run_kld.py, joined into the single CLI stage_campaign.sh pins:

  kld_report.py --profile k6 --teacher <tree> --runs run1 ... run5 \
      --fp8-baseline 0.020615 --k4-baseline 0.024555 \
      --out k6-packed-kld.json --five-run-out k6-five-run-kld.json \
      --comparison-out comparison-table.md

Per run it computes (or resumes) the sealed per-run KLD report
(quant-pipeline.glm53-packed-student-kld.v2): exact fp64 log-softmax,
teacher->student direction, the sealed 25-window final panel only
(25 x 2047 = 51,175 positions), mean = upstream's estimator exactly (no
weighting, no bf16 anywhere).  For profile k6 it then seals the packed-KLD
receipt chain through the patched glm53_k6_postmtp module (patches-v2 0005) and,
with five runs, the five-cold-run receipt (bitwise determinism shows as one
identical tokenwise_kld_sha256).  Profile k6k8 gets a malaiwah.* summary
receipt (its sealed schema set ships with the K6K8 support module).
"""

from __future__ import annotations

import argparse
import io
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FINAL_WINDOW_COUNT = 25
FINAL_WINDOW_IDS = tuple(f"final-{index:04d}" for index in range(25))
FINAL_PREDICTION_POSITIONS = 25 * 2047
# --profile mlx: mlx_surface.MlxSurface.student_label() builds
# "mlx-affine-b<bits>-gs<group>[-mixed-<hash of the bit histogram>]", so the
# report gates the FAMILY here and the exact string across runs.
MLX_STUDENT_LABEL_PREFIX = "mlx-affine-"


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"kld_report: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise _fail(f"no quant_pipeline package under {pipeline_root}")


def _import_pipeline(pipeline_root: Optional[str]) -> None:
    if pipeline_root:
        src = str(_pipeline_src(Path(pipeline_root)))
    elif os.environ.get("QP_PIPELINE_ROOT"):
        src = str(_pipeline_src(Path(os.environ["QP_PIPELINE_ROOT"])))
    else:
        try:
            import quant_pipeline  # noqa: F401

            return
        except ImportError:
            raise _fail(
                "quant_pipeline not importable: pass --pipeline-root, set "
                "QP_PIPELINE_ROOT, or export PYTHONPATH to the patched tree"
            )
    if src not in sys.path:
        sys.path.insert(0, src)


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".new-{os.getpid()}")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def _find_teacher_receipt(teacher_root: Path) -> Path:
    from quant_pipeline.evaluation.glm53_logits import CAPTURE_SCHEMA

    direct = teacher_root / "capture-receipt.json"
    candidates = [direct] if direct.is_file() else sorted(teacher_root.glob("**/*.json"))
    previews = 0
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "-preview." in str(value.get("schema", "")):
            previews += 1
            continue
        if (
            isinstance(value, dict)
            and value.get("schema") == CAPTURE_SCHEMA
            and value.get("capture_role") == "bf16_teacher"
        ):
            return path
    hint = ""
    if previews:
        hint = (
            f" ({previews} PREVIEW capture receipt(s) were seen and refused: a "
            "preview is position-sampled and can never be a teacher; capture a "
            "teacher with stream_score.py --capture-role teacher --store-positions all)"
        )
    raise _fail(
        f"no sealed bf16_teacher capture receipt found under {teacher_root} "
        f"(expected the downloaded teacher final-window tree){hint}"
    )


def _refuse_preview_capture(run_dir: Path) -> None:
    """Friendly pre-check: a preview capture must never reach the sealed scorer.

    Deliberately needs only json+pathlib (no quant_pipeline), so selftests can
    exercise it on a laptop.
    """
    capture = run_dir / "capture-receipt.json"
    if not capture.is_file():
        return
    try:
        schema = str(json.loads(capture.read_text(encoding="utf-8")).get("schema", ""))
    except (OSError, json.JSONDecodeError, AttributeError):
        return
    if "-preview." in schema:
        raise _fail(
            f"REFUSED: {run_dir} is a PREVIEW capture (schema {schema}). The "
            "sealed scorer only accepts full-census captures "
            "(quant-pipeline.glm53-logit-capture.v1). Score previews with "
            "bin/kld-preview."
        )


def _teacher_source_of(teacher: Dict[str, Any]) -> "tuple[str, str]":
    """(teacher_source, teacher_label) from the teacher receipt.

    A same-lane teacher carries the additive teacher_provenance block; the
    sealed EP8 teacher predates it and carries none -- backward compatible by
    construction.
    """
    provenance = teacher.get("teacher_provenance")
    if isinstance(provenance, dict):
        return "same_lane_native_bf16", str(provenance.get("teacher_label")
                                            or "same-lane-native-bf16")
    return "sealed_ep8_bf16_teacher", "sealed-ep8"


def _resolve_teacher_paths(
    teacher_rows: Dict[str, Dict[str, Any]], teacher_root: Optional[Path],
    sha256_file,
) -> Dict[str, Path]:
    """Sealed fast path: the recorded absolute path exists and is used as-is
    (byte-identical behaviour to before this function existed).  Portability
    fallback: a teacher tree moved to another machine keeps its receipt's
    absolute paths from the capture box; remap to <teacher_root>/logits/<name>
    and, in the fallback path ONLY, verify the file's sha256 against the
    receipt row before use -- hash content, not containers."""
    resolved: Dict[str, Path] = {}
    for window_id in sorted(teacher_rows):
        row = teacher_rows[window_id]
        recorded = Path(row["path"])
        if recorded.is_file():
            resolved[window_id] = recorded
            continue
        if teacher_root is None:
            raise _fail(f"teacher logits missing: {recorded}")
        fallback = teacher_root / "logits" / recorded.name
        if not fallback.is_file():
            raise _fail(
                f"teacher logits for {window_id} not found at the sealed path "
                f"{recorded} nor at the portable fallback {fallback}"
            )
        digest = sha256_file(fallback)
        if digest != row["sha256"]:
            raise _fail(
                f"teacher fallback {fallback} has sha256 {digest[:12]}..., but "
                f"the sealed receipt row says {str(row['sha256'])[:12]}... -- "
                "the remapped file is NOT the sealed teacher row (content "
                "hash rules identity, never the filename)"
            )
        print(
            f"teacher path remapped: {window_id}: {recorded} -> {fallback} "
            "(sha256 verified)",
            flush=True,
        )
        resolved[window_id] = fallback
    return resolved


def _record_map(receipt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["window_id"]: row for row in receipt["logit_files"]}


#: How the artifact this profile measures is STORED. Not the lane, and not the world size.
#: The old expression suffixed every unrecognised profile "-tp4", so the two published
#: streaming receipts carry `k6-stream-tp4` and `k8-tp4` -- a claim that the artifact ships
#: TP4-sliced ranks. k6/DECISIONS.md forbids exactly that ("the checkpoint must not bake in
#: a TP/EP topology"), and k6/stage_campaign.sh hard-asserts `qualified_tp_sizes == []` on those
#: very releases before publishing them. It is the same defect JOURNAL lesson #10 already
#: recorded and fixed for `gguf-tp4`, left live in two more receipts.
PROFILE_STORAGE_LABEL = {
    "k6": "k6-hf-sharded",
    "k6-stream": "k6-stream",
    "k8": "k8-hf-sharded",
    "k6k8": "k6k8-hf-sharded",
    "native-bf16": "native-bf16-stream",
    "mlx": "mlx-stream",
    "gguf": "gguf-stream",
    "nvfp4": "nvfp4-stream",
    "turbo-4.05bpw": "turbo-4.05bpw-hf-sharded",
    "turbo-3.05bpw": "turbo-3.05bpw-hf-sharded",
    "turbo-2.05bpw": "turbo-2.05bpw-hf-sharded",
    "vcruz-k2-2bpw": "vcruz-k2-2bpw-hf-sharded",
    "tr3-4bpw": "tr3-4bpw-hf-sharded",
    # genuinely TP4-sliced: dione_surface pins glm53-selective-exl3-tp4-v1
    "dione-q4": "dione-q4-tp4",
    "dione-3.0bpw": "dione-3.0bpw-tp4",
}


#: Which capture SURFACE each profile's run receipts came from.  Declared, never
#: inferred from the profile NAME.  The provenance-republishing block below used
#: to dispatch on `profile.startswith(("dione", "turbo"))`, which is a probe whose
#: answer happens to be right for the three profiles that existed when it was
#: written and silently wrong for the fourth: `vcruz-k2-2bpw` is captured by the
#: SAME `stream_score --source exl3hf` front end as the turbo-* profiles and seals
#: the same receipt fields, and it starts with neither prefix.  Under the old
#: dispatch its headline summary would have carried no artifact_repo, no
#: artifact_revision, no codebook and no seal_disclosure -- a registry row citing
#: an artifact it cannot name, discovered after the money.  That is JOURNAL
#: LESSON 48 recurring one profile later, so the mapping is now data.
PROFILE_SURFACE_FAMILY = {
    "turbo-4.05bpw": "exl3hf",
    "turbo-3.05bpw": "exl3hf",
    "turbo-2.05bpw": "exl3hf",
    "vcruz-k2-2bpw": "exl3hf",
    "dione-q4": "dione",
    "dione-3.0bpw": "dione",
    "tr3-4bpw": "tr3",
}


def _profile_surface_family(profile):
    """The capture surface for this profile, or None for a lane-native profile.

    A profile that names a THIRD-PARTY artifact and is missing here would seal a
    headline receipt with no provenance, so an unknown profile whose run receipts
    turn out to carry surface pins is caught at seal time by the required-field
    check, not by this lookup returning None.
    """
    return PROFILE_SURFACE_FAMILY.get(profile)


def _profile_storage_label(profile):
    """Refuse rather than guess. A new profile must not inherit a storage claim by
    falling through a default."""
    try:
        return PROFILE_STORAGE_LABEL[profile]
    except KeyError:
        raise _fail(
            f"profile {profile!r} has no declared STORAGE label. The old default appended "
            f"'-tp4' to anything unrecognised, which put a false 'ships TP4-sliced ranks' "
            f"claim into two published receipts. Add an entry to PROFILE_STORAGE_LABEL "
            f"saying how this artifact is actually stored."
        )


def _torch_threads():
    """Recorded beside position_block: the divergent reduction path is thread-count
    sensitive, so the block size alone does not pin the accumulation order."""
    try:
        import torch
        return int(torch.get_num_threads())
    except Exception:                                             # noqa: BLE001
        return None


def _load_slice(path: Path, start: int, stop: int) -> Any:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_slice("logits")[start:stop]


def _token_kld(teacher: Any, student: Any, device: str) -> "tuple[np.ndarray, int]":
    import torch

    if teacher.shape != student.shape or teacher.ndim != 2:
        raise _fail("teacher/student logit geometry mismatch")
    teacher64 = teacher.to(device=device, dtype=torch.float64)
    student64 = student.to(device=device, dtype=torch.float64)
    if not torch.isfinite(teacher64).all() or not torch.isfinite(student64).all():
        raise _fail("teacher/student logits must be finite")
    teacher_logp = torch.log_softmax(teacher64, dim=-1)
    student_logp = torch.log_softmax(student64, dim=-1)
    values = torch.sum(torch.exp(teacher_logp) * (teacher_logp - student_logp), dim=-1)
    matches = int(
        torch.count_nonzero(
            torch.argmax(teacher64, dim=-1) == torch.argmax(student64, dim=-1)
        )
    )
    return values.cpu().numpy(), matches


def _measure_run(
    *,
    run_dir: Path,
    teacher: Dict[str, Any],
    student_label: str,
    chunk_positions: int,
    device: str,
    expected_capture_role: str = "packed_student",
    teacher_root: Optional[Path] = None,
) -> Path:
    """Compute or resume one sealed per-run KLD report; returns its path."""

    from quant_pipeline.core.artifacts import (
        atomic_write,
        canonical_json,
        sha256_bytes,
        sha256_file,
        write_json,
    )
    from quant_pipeline.evaluation.glm53_logits import load_capture_receipt, summarize

    report_path = run_dir / "kld-report.json"
    if report_path.is_file():
        resumed = json.loads(report_path.read_text(encoding="utf-8"))
        if resumed.get("student_label") != student_label:
            raise _fail(
                f"resumed {report_path} is labeled {resumed.get('student_label')!r}, "
                f"not the profile's expected {student_label!r} - wrong --runs/--profile pair"
            )
        # The resume branch validated ONLY the label, then returned. Every teacher, panel
        # and window cross-check lives below it and was skipped, so re-scoring an existing
        # run dir against a DIFFERENT teacher silently did nothing and the summary then
        # published the new teacher's receipt_sha256 over means computed against the old
        # one. teacher_receipt_sha256 is the registry's comparability key, so the row would
        # be filed in the wrong ranked table -- exactly the conflation _comparison_table's
        # docstring says the key exists to prevent. The sealed K6 branch was immune only
        # because build_packed_k6_kld_receipt checks it explicitly.
        #
        # The field names are ASYMMETRIC: the report calls it teacher_receipt_sha256, the
        # teacher receipt calls it receipt_sha256. The obvious symmetric loop raises
        # KeyError on every resume.
        for report_field, teacher_field in (("teacher_receipt_sha256", "receipt_sha256"),
                                            ("token_panel_receipt_sha256",
                                             "token_panel_receipt_sha256")):
            if resumed.get(report_field) != teacher.get(teacher_field):
                raise _fail(
                    f"resumed {report_path} carries {report_field} "
                    f"{str(resumed.get(report_field))[:12]}..., but --teacher resolves to "
                    f"{str(teacher.get(teacher_field))[:12]}... - the resumed number was "
                    f"measured against a DIFFERENT reference; delete the stale "
                    f"kld-report.json to re-score"
                )
        if chunk_positions is not None and resumed.get("position_block") not in (
                None, chunk_positions):
            # NUM-09. Absent on reports written before the field existed, so a missing
            # value is "unknown" and only an explicit mismatch refuses.
            raise _fail(
                f"resumed {report_path} was scored with position_block "
                f"{resumed.get('position_block')}, not the requested {chunk_positions}. "
                f"The tokenwise digest is the suite's determinism evidence and the block "
                f"size can change it bitwise; delete the stale report to re-score"
            )
        return report_path
    _refuse_preview_capture(run_dir)
    student = load_capture_receipt(
        run_dir / "capture-receipt.json", expected_role=expected_capture_role
    )
    declared_label = student.get("student_label")
    if declared_label is not None and declared_label != student_label:
        raise _fail(
            f"capture receipt declares student_label {declared_label!r}, but "
            f"--profile expects {student_label!r} - wrong --runs/--profile pair"
        )
    if teacher["token_panel_receipt_sha256"] != student["token_panel_receipt_sha256"]:
        raise _fail("teacher and student captures use different sealed token panels")
    if teacher["vocab_size"] != student["vocab_size"]:
        raise _fail("teacher and student vocabularies differ")
    if not student.get("runtime_reader_sha256"):
        raise _fail("student capture is not bound to an exact runtime reader/source identity")
    teacher_rows = _record_map(teacher)
    student_rows = _record_map(student)
    if set(teacher_rows) != set(student_rows):
        raise _fail("teacher and student window sets differ")
    if (
        tuple(sorted(teacher_rows)) != FINAL_WINDOW_IDS
        or any(row.get("role") != "final" for row in teacher_rows.values())
        or any(row.get("role") != "final" for row in student_rows.values())
        or sum(int(row.get("prediction_positions", -1)) for row in teacher_rows.values())
        != FINAL_PREDICTION_POSITIONS
    ):
        raise _fail("KLD qualification requires the sealed 25-window final panel only")
    for window_id, left in teacher_rows.items():
        right = student_rows[window_id]
        fields = (
            "document_id",
            "domain",
            "role",
            "token_ids_sha256",
            "attention_mask_sha256",
            "prediction_positions",
        )
        if any(left[field] != right[field] for field in fields):
            raise _fail(f"student capture relabels sealed window {window_id}")

    started = time.monotonic()
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise _fail(
            "CUDA device requested but unavailable; pass --device cpu for the "
            "(slow) exact fp64 CPU path"
        )
    teacher_paths = _resolve_teacher_paths(teacher_rows, teacher_root, sha256_file)
    teacher_source, teacher_label = _teacher_source_of(teacher)
    token_values: List[np.ndarray] = []
    top1_matches = 0
    per_window: List[Dict[str, Any]] = []
    for window_id in sorted(teacher_rows):
        teacher_row = teacher_rows[window_id]
        student_row = student_rows[window_id]
        count = int(teacher_row["prediction_positions"])
        values = np.empty(count, dtype=np.float64)
        for start in range(0, count, chunk_positions):
            stop = min(start + chunk_positions, count)
            teacher_logits = _load_slice(teacher_paths[window_id], start, stop)
            student_logits = _load_slice(Path(student_row["path"]), start, stop)
            if teacher_logits.shape != (stop - start, int(teacher["vocab_size"])) or (
                student_logits.shape != teacher_logits.shape
            ):
                raise _fail(f"logit geometry mismatch in {window_id}")
            chunk_values, chunk_matches = _token_kld(teacher_logits, student_logits, device)
            values[start:stop] = chunk_values
            top1_matches += chunk_matches
        token_values.append(values)
        per_window.append(
            {
                "window_id": window_id,
                "document_id": teacher_row["document_id"],
                "domain": teacher_row["domain"],
                "role": teacher_row["role"],
                "summary": summarize(values),
            }
        )
        print(f"{window_id}: mean {per_window[-1]['summary']['mean']:.6f}", flush=True)
    all_values = np.concatenate(token_values)
    buffer = io.BytesIO()
    np.save(buffer, all_values, allow_pickle=False)
    token_path = run_dir / "tokenwise-kld.npy"
    atomic_write(token_path, buffer.getvalue())
    domains: Dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in per_window}):
        indices = [index for index, row in enumerate(per_window) if row["domain"] == domain]
        domains[domain] = summarize(
            np.concatenate([token_values[index] for index in indices])
        )
    overall = summarize(all_values)
    # Lane-ONLY identity (additive, forward-looking): copied from the run's
    # backend.json when the capture emitted one.  backend_identity_sha256
    # pins artifact+lane together and cannot answer "same lane?"; this hash
    # can, so fidelity-stats gates paired deltas and attributables on its
    # equality when both sides carry it.  Absent for captures made before
    # 2026-08-29, so historical reports reproduce byte-identically.
    student_lane_identity = None
    try:
        backend_json = run_dir / "backend.json"
        if backend_json.is_file():
            student_lane_identity = json.loads(
                backend_json.read_text(encoding="utf-8")
            ).get("lane_identity_sha256")
    except (OSError, ValueError):
        student_lane_identity = None
    report = {
        # v2: adds position_block and torch_threads (NUM-09). Both are inside
        # report_sha256, so a report written by this version cannot be byte-compared with
        # a v1 report of the same run -- the schema string says which is which rather than
        # leaving a reader to discover it from a hash mismatch. The MEASURED quantities
        # (summary.mean, tokenwise_kld_sha256) are unchanged; no published number moves,
        # and the already-written v1 reports on disk are not rewritten.
        "schema": "quant-pipeline.glm53-packed-student-kld.v2",
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "student_receipt_sha256": student["receipt_sha256"],
        "student_label": student_label,
        "student_checkpoint_identity_sha256": student["checkpoint_identity_sha256"],
        "runtime_reader_sha256": student["runtime_reader_sha256"],
        "token_panel_receipt_sha256": teacher["token_panel_receipt_sha256"],
        "teacher_backend_identity_sha256": teacher["backend_identity_sha256"],
        "teacher_source": teacher_source,
        "teacher_label": teacher_label,
        "student_backend_identity_sha256": student["backend_identity_sha256"],
        "qualification_panel_final_only": True,
        "qualification_window_count": len(per_window),
        "kld_direction": "teacher_to_student",
        "metric": "tokenwise KL over exact jointly-valid causal prediction positions",
        "compute_device": device,
        "compute_dtype": "float64",
        # NUM-09. --chunk-positions changes the tokenwise values bitwise in degenerate
        # configurations (a single-row batch splits the 154,880-wide reduction instead of
        # the rows, so the accumulation order changes; measured, and thread-count
        # sensitive too), and tokenwise_kld_sha256 is the suite's headline determinism
        # evidence. It was recorded nowhere, so a recompute at a different block size
        # produced a bare hash mismatch that was unattributable to any artifact. Named
        # position_block to match docs/FIDELITY-DATASET-SPEC.md's mandatory comparator
        # field and bin/fidelity_dataset.py, which already emit exactly that.
        "position_block": chunk_positions,
        "torch_threads": _torch_threads(),
        "summary": overall,
        "per_domain": domains,
        "per_window": per_window,
        "top1_agreement": top1_matches / int(all_values.size),
        "mean_kld_lt_0_06": bool(overall["mean"] < 0.06),
        "tokenwise_kld_path": str(token_path.resolve()),
        "tokenwise_kld_bytes": token_path.stat().st_size,
        "tokenwise_kld_sha256": sha256_file(token_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    if student_lane_identity:
        report["student_lane_identity_sha256"] = student_lane_identity
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(report_path, report)
    return report_path


# The teacher every historical baseline in the comparison table was measured
# against (brandonmusic's sealed EP8 fp32-logits capture).  Receipts naming a
# DIFFERENT teacher_receipt_sha256 are tabled apart and never ranked against
# these rows.
SEALED_EP8_TEACHER_SHA = (
    "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5"
)

_TABLE_HEADER = (
    "| model | routed bpw | size | mean tokenwise KLD vs BF16 teacher "
    "(25 sealed windows, 51,175 pos, fp64) | provenance |"
)


def _comparison_table(
    out_path: Path,
    fp8_baseline: float,
    k4_baseline: float,
    receipts_dir: Path,
) -> None:
    """One ranked table PER TEACHER.  Rows measured against different
    teacher_receipt_sha256 values are different estimands; putting them in one
    ranked list would be exactly the cross-reference conflation the registry's
    comparability key exists to prevent."""
    baseline_rows = [
        f"| zai-org FP8 (as served) | 8 | 328 GB | {fp8_baseline:.6f} "
        "| our fidelity-suite baseline |",
        f"| brandonmusic K4 (EXL3/TR3-MCG) | 4.01 | 163.6 GiB | {k4_baseline:.6f} "
        "(five-run mean, stddev 0) | his sealed receipts |",
    ]
    groups: "Dict[str, Dict[str, Any]]" = {}

    def _group(sha: str, label: str) -> Dict[str, Any]:
        return groups.setdefault(sha, {"label": label, "rows": []})

    for profile, bpw, size in (
        ("k6", "6.01", "236.1 GiB"),
        ("k8", "8.01", "308.7 GiB"),
        ("k6k8", "6.68", "260.3 GiB"),
        ("dione-q4", "4.0 (TP4-sliced)", "174.5 GiB"),
        ("dione-3.0bpw", "3.0 (TP4-sliced)", "~139 GiB"),
        ("turbo-4.05bpw", "4.05 (full-scope, head 6)", "150.2 GiB"),
        ("turbo-3.05bpw", "3.05 (full-scope, head 6)", "116.6 GiB"),
        ("turbo-2.05bpw", "2.05 (full-scope, head 5)", "79.4 GiB"),
        ("vcruz-k2-2bpw", "2.0 (routed experts only, native head)", "91.0 GiB"),
        # community MLX affine snapshots: bit mix and size are properties of the
        # artifact, so both are READ from the receipt instead of hardcoded here
        ("mlx", None, None),
    ):
        receipt_path = receipts_dir / f"{profile}-packed-kld.json"
        if receipt_path.is_file():
            # NUM-14. $RCPT is a long-lived shared receipts directory and this table
            # deliberately scans eight profiles measured in different lanes and different
            # sessions, so the siblings arrive by hand/scp/hf-download. A truncated
            # download raised JSONDecodeError, and a receipt with no measured_mean_kld
            # raised `TypeError: unsupported format string passed to NoneType.__format__`
            # -- both AFTER --out was already written, which is the half-committed state
            # somebody later picks up by hand. A skipped row must be LOUD: a silently
            # dropped artifact in a public comparison table is worse than the crash.
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"comparison-table: skipping unreadable {receipt_path.name}: {exc}",
                      file=sys.stderr)
                continue
            mean = receipt.get("measured_mean_kld")
            if isinstance(mean, bool) or not isinstance(mean, (int, float)) \
                    or not math.isfinite(mean):
                # bool first: isinstance(True, int) is True and f"{True:.6f}" renders
                # "1.000000", i.e. a silently bogus row.
                print(f"comparison-table: skipping {receipt_path.name}: "
                      f"measured_mean_kld is not a finite number ({mean!r})",
                      file=sys.stderr)
                continue
            gate = "GREEN" if receipt.get("quality_gate_passed") else "RED"
            label = {
                "k6": "malaiwah K6",
                "k8": "malaiwah K8 (uniform)",
                "k6k8": "malaiwah K6K8 (down@8)",
                "dione-q4": "0xSero Dione Q4 (EXL3 K4, unsealed source)",
                "dione-3.0bpw": "0xSero Dione 3.0bpw (EXL3 K3, unsealed source)",
                "turbo-4.05bpw": "turboderp 4.05bpw (stock EXL3 mul1, quantized head, unsealed source)",
                "turbo-3.05bpw": "turboderp 3.05bpw (stock EXL3 mul1, quantized head, unsealed source)",
                "turbo-2.05bpw": "turboderp 2.05bpw (stock EXL3 mul1, quantized head at 5 bits, unsealed source)",
                "vcruz-k2-2bpw": "vcruz305 K2 2bpw (EXL3 mcg, routed experts only, native head, unsealed source)",
                "mlx": "%s (MLX affine, unsealed source, quantized BEYOND the routed experts)"
                       % (receipt.get("mlx_repo") or "community MLX"),
            }[profile]
            if profile == "mlx":
                histogram = receipt.get("mlx_bits_histogram") or {}
                bpw = " ".join(f"{key}:{value}" for key, value in sorted(histogram.items())) or "?"
                artifact_bytes = receipt.get("mlx_artifact_bytes")
                size = (
                    f"{artifact_bytes / (1 << 30):.1f} GiB"
                    if isinstance(artifact_bytes, int)
                    else "?"
                )
            teacher_sha = str(
                receipt.get("teacher_receipt_sha256") or SEALED_EP8_TEACHER_SHA
            )
            teacher_label = str(receipt.get("teacher_label") or "sealed-ep8")
            _group(teacher_sha, teacher_label)["rows"].append(
                f"| **{label}** | {bpw} | {size} | **{mean:.6f}** (gate < 0.06 {gate}) "
                "| this campaign |"
            )
    # the historical baselines belong to the sealed EP8 teacher's table
    _group(SEALED_EP8_TEACHER_SHA, "sealed-ep8")["rows"] = (
        baseline_rows + _group(SEALED_EP8_TEACHER_SHA, "sealed-ep8")["rows"]
    )
    ordered = sorted(groups, key=lambda sha: (sha != SEALED_EP8_TEACHER_SHA, sha))
    lines: List[str] = []
    for sha in ordered:
        group = groups[sha]
        lines.append(f"#### teacher: {group['label']} ({sha[:12]}...)")
        lines.append("")
        lines.append(_TABLE_HEADER)
        lines.append("|---|---|---|---|---|")
        lines.extend(group["rows"])
        lines.append("")
    if len(ordered) > 1:
        pairs = " vs ".join(sha[:12] + "..." for sha in ordered)
        lines.append(
            "rows above and below are NOT comparable: different reference "
            f"(teacher_receipt_sha256 {pairs})"
        )
    out_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True,
                        choices=("k6", "k6-stream", "k8", "k6k8", "dione-q4", "dione-3.0bpw",
                                 "turbo-4.05bpw", "turbo-3.05bpw", "turbo-2.05bpw",
                                 "vcruz-k2-2bpw", "tr3-4bpw",
                                 "native-bf16", "mlx", "gguf", "nvfp4"))
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--fp8-baseline", type=float, default=0.020615)
    parser.add_argument("--k4-baseline", type=float, default=0.024555)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--five-run-out", type=Path)
    parser.add_argument("--comparison-out", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="materialized checkpoint (default <root>/ckpt-<profile>)")
    parser.add_argument("--output-root", type=Path,
                        help="encode output root (default <root>/out-<profile>)")
    parser.add_argument("--pipeline-root")
    parser.add_argument("--chunk-positions", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    # NUM-13. `--five-run-out` is honoured only on the k6 branch; every other profile
    # accepted the flag, never referenced it, wrote no file and exited 0. Silent-no-op on
    # an evidence-producing flag is the same failure class as a fetch that matches zero
    # files. Refused at parse time, before the teacher load, so it costs milliseconds on a
    # laptop rather than a wasted run.
    if getattr(args, "five_run_out", None) and args.profile != "k6":
        raise _fail(
            f"--five-run-out is implemented only for --profile k6 (the sealed five-run "
            f"receipt builder is K6-specific and stamps a K6 identity); --profile "
            f"{args.profile} would write nothing"
        )
    _import_pipeline(args.pipeline_root)

    from quant_pipeline.core.artifacts import sha256_file
    from quant_pipeline.evaluation.glm53_logits import load_capture_receipt

    bits = {"k6": 6, "k6-stream": 6, "k8": 8, "nvfp4": 4, "tr3-4bpw": 4}.get(args.profile)
    student_label = {
        "k6": "uniform-k6",
        # single-device STREAMING capture of the same sealed K6 surface: the
        # per-run kld-report.json is produced by the identical estimator, so the
        # numbers are directly comparable to the sealed k6 lane.  It takes the
        # malaiwah summary branch below because the sealed K6 receipt chain
        # requires a materialized checkpoint's materialization-receipt.json,
        # which a payload-store run legitimately does not have.
        "k6-stream": "uniform-k6",
        "k8": "uniform-k8",
        "k6k8": "mixed-k6k8",
        # third-party Dione (0xSero) selective-EXL3 TP4 snapshots, scored on
        # the same sealed panel through --surface dione (unsealed source,
        # disclosed in the capture receipt's seal_disclosure)
        "dione-q4": "dione-exl3-k4-tp4",
        "dione-3.0bpw": "dione-exl3-k3-tp4",
        # third-party stock-exllamav3 releases (turboderp), scored on the same
        # sealed panel through stream_score --source exl3hf (mul1 codebook,
        # FULL-scope quant incl. the 6-bit head, unsealed source, disclosed).
        # Labels must match stream_score.EXL3HF_PROFILES.
        "turbo-4.05bpw": "turboderp-exl3-mul1-4.05bpw",
        "turbo-3.05bpw": "turboderp-exl3-mul1-3.05bpw",
        "turbo-2.05bpw": "turboderp-exl3-mul1-2.05bpw",
        # vcruz305's K2 pack shares turboderp's STORAGE layout (stock-exllamav3
        # HF shards, read by --source exl3hf) and nothing else: MCG codebook,
        # routed-experts-only scope, native BF16 head.  Separate label because a
        # label that said "turboderp-exl3-mul1" would be false on all three.
        "vcruz-k2-2bpw": "vcruz305-exl3-mcg-2bpw",
        # a SEALED TR3-published EXL3/MCG release (brandonmusic's layout and its
        # byte-identical mirrors), scored through stream_score --source tr3.
        # Routed experts only, native BF16 head, and -- unlike every other
        # third-party surface here -- a publisher seal this lane recomputes.
        # Label must match stream_score.TR3_PROFILES.
        "tr3-4bpw": "tr3-exl3-mcg-4bpw",
        # the BF16 FLOOR of the streaming lane: the identical capture with the
        # routed experts read straight from the official checkpoint and no codec
        # in the path (stream_score.py --source native).  Subtracting this mean
        # from a quant's mean leaves that quant's quantization-attributable error.
        "native-bf16": "native-bf16",
        # community MLX affine snapshots (stream_score.py --source mlx).  The
        # label is DERIVED from the artifact's own bit mix, so it cannot be a
        # constant here; it is read from the first run's capture receipt below,
        # gated on the family prefix, and then required of every other run.
        "mlx": None,
        # community llama.cpp GGUF artifacts (unsloth, ...) scored through
        # stream_score.py --source gguf.  The label is deliberately FORMAT-wide,
        # not per-file: which quant it was (repo, revision, per-file sha256, the
        # measured ggml type census and the scope policy) is carried in the
        # summary's provenance block below, where a registry row reads it.  A
        # per-file label would put a mutable string in the equality gate.
        "gguf": "gguf-llamacpp",
        # community NVFP4 snapshots (RedHatAI / LibertAIDAI), scored through
        # stream_score.py --source nvfp4 (exact-fp32 e2m1/gs16 dequant, unsealed
        # source disclosed in the capture receipt's streaming_disclosure.nvfp4).
        # One label for both dialects: the summary below carries repo+revision.
        "nvfp4": "nvfp4-e2m1-gs16",
    }[args.profile]
    # a native run is not a packed student and does not claim to be one; nor is
    # a GGUF run, whose non-routed weights are the artifact's as well
    expected_capture_role = {
        "native-bf16": "native_bf16_student",
        "gguf": "gguf_student",
    }.get(args.profile, "packed_student")
    runs = [path.resolve() for path in args.runs]
    # NUM-03. `--runs` was never deduplicated and the per-run student_receipt_sha256 values
    # were never required to be distinct, so passing one run dir twice produced
    # cold_run_count=2, two identical run_means, ONE distinct tokenwise digest, and
    # `bitwise_deterministic: true` -- fabricated determinism evidence from a single
    # capture, which is the strongest claim this suite makes. The sealed five-run builder
    # gates this; the summary branch did not. Checked BEFORE the measure loop: after it
    # would burn hours of fp64 GPU scoring before refusing.
    if len(set(runs)) != len(runs):
        raise _fail(
            "--runs lists the same capture directory more than once (after resolving "
            "symlinks). Two cold runs are two captures; one capture named twice is not "
            "determinism evidence."
        )
    if args.profile == "mlx":
        first = runs[0] / "capture-receipt.json"
        if not first.is_file():
            raise _fail(f"--profile mlx needs the first run's capture receipt: {first}")
        declared = json.loads(first.read_text(encoding="utf-8")).get("student_label")
        if not isinstance(declared, str) or not declared.startswith(MLX_STUDENT_LABEL_PREFIX):
            raise _fail(
                f"--profile mlx expects a student_label starting with "
                f"{MLX_STUDENT_LABEL_PREFIX!r} (mlx_surface derives it from the artifact's "
                f"bit mix); {first} declares {declared!r}"
            )
        # every remaining run must now carry this EXACT label, which is the
        # cross-run gate the fixed-label profiles get for free
        student_label = declared
    campaign_root = runs[0].parent.parent
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else campaign_root / f"ckpt-{args.profile}"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else campaign_root / f"out-{args.profile}"
    )

    teacher_path = _find_teacher_receipt(args.teacher.resolve())
    teacher = load_capture_receipt(teacher_path, expected_role="bf16_teacher")
    teacher_source, teacher_label = _teacher_source_of(teacher)
    print(
        json.dumps(
            {
                "teacher_source": teacher_source,
                "teacher_label": teacher_label,
                "teacher_receipt_sha256": teacher["receipt_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    report_paths: List[Path] = []
    for run_dir in runs:
        report_paths.append(
            _measure_run(
                run_dir=run_dir,
                teacher=teacher,
                student_label=student_label,
                expected_capture_role=expected_capture_role,
                chunk_positions=args.chunk_positions,
                device=args.device,
                teacher_root=teacher_path.parent,
            )
        )
    reports = [
        (json.loads(path.read_text(encoding="utf-8")), sha256_file(path))
        for path in report_paths
    ]
    # NUM-03, second half. Path dedup alone is not sufficient: `cp -r run1 run2` gives two
    # distinct resolved paths carrying the same capture receipt, and only this catches it.
    # `.get()` plus a shape check rather than a subscript -- a KeyError is a crash, not a
    # refusal, and the sibling five-run builder validates the shape too.
    _receipts = [str(report.get("student_receipt_sha256")) for report, _ in reports]
    if len(set(_receipts)) != len(_receipts) or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in _receipts):
        raise _fail(
            f"cold-run evidence does not contain {len(reports)} distinct cold captures "
            f"(student_receipt_sha256: {_receipts!r}). Two run directories holding one "
            f"capture are one measurement, and reporting them as two fabricates the "
            f"determinism claim."
        )
    means = [float(report["summary"]["mean"]) for report, _ in reports]
    kld_shas = sorted({str(report["tokenwise_kld_sha256"]) for report, _ in reports})
    # One capture can never evidence determinism, however many times it is hashed. The
    # summary used to write `bitwise_deterministic: true` from a single run; DET-001 caught
    # it downstream, but the JSON artifact itself asserted determinism from one run.
    bitwise_deterministic = len(reports) >= 2 and len(kld_shas) == 1
    print(
        json.dumps(
            {
                "runs": len(reports),
                "run_means": means,
                "distinct_tokenwise_kld_sha256": kld_shas,
                "bitwise_deterministic": bitwise_deterministic,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.profile == "k6":
        from quant_pipeline.publication.glm53_k6_postmtp import (
            build_native_copy_bridge,
            build_packed_k6_kld_receipt,
            build_five_run_kld_receipt,
        )

        materialization = _read_json(
            checkpoint / "materialization-receipt.json",
            "materialization receipt (pass --checkpoint if the layout differs)",
        )
        plan = _read_json(
            output_root / "materialization-plan.json",
            "materialization plan (pass --output-root if the layout differs)",
        )
        contract = _read_json(output_root / "contract.json", "direct contract")
        inventory = _read_json(output_root / "inventory.json", "inventory")
        reader_abi = _read_json(
            output_root / "reader-abi-receipt.json", "reader ABI receipt"
        )
        shards_verified = (output_root / "inventory-shards-verified.json").is_file()
        if not shards_verified:
            raise _fail(
                "full-shard hash verification marker absent "
                f"({output_root / 'inventory-shards-verified.json'}) - the native "
                "copy bridge requires it; re-run the contract stage"
            )
        native_copy = build_native_copy_bridge(
            materialization=materialization,
            plan=plan,
            contract=contract,
            inventory=inventory,
            materialization_file_sha256=sha256_file(
                checkpoint / "materialization-receipt.json"
            ),
            full_shard_hash_verification=True,
        )
        student_backend = _read_json(runs[0] / "backend.json", "run-1 student backend")
        packed = build_packed_k6_kld_receipt(
            materialization=materialization,
            native_copy=native_copy,
            reader_abi=reader_abi,
            student_backend=student_backend,
            teacher_receipt_path=teacher_path,
            student_receipt_path=runs[0] / "capture-receipt.json",
            kld_report=reports[0][0],
            kld_report_file_sha256=reports[0][1],
        )
        _atomic_json(args.out.resolve(), packed)
        _atomic_json(args.out.resolve().with_name("k6-native-copy-bridge.json"), native_copy)
        print(
            json.dumps(
                {
                    "measured_mean_kld": packed["measured_mean_kld"],
                    "quality_gate_passed": packed["quality_gate_passed"],
                },
                sort_keys=True,
            )
        )
        if args.five_run_out:
            if len(reports) != 5:
                raise _fail(
                    f"--five-run-out requires exactly five runs, got {len(reports)}"
                )
            five = build_five_run_kld_receipt(reports)
            _atomic_json(args.five_run_out.resolve(), five)
            print(
                json.dumps(
                    {
                        "mean_of_run_means": five["mean_of_run_means"],
                        "population_stddev_of_run_means": five[
                            "population_stddev_of_run_means"
                        ],
                        "five_run_qualified": five["qualified"],
                    },
                    sort_keys=True,
                )
            )
    else:
        # K8/K6K8: the sealed receipt builders in glm53_k6_postmtp are
        # K6-specific; these profiles get a transparent malaiwah summary (3
        # cold runs, disclosed) that satisfies the stage gate fields.
        # NUM-02. `mean = means[0]` published RUN 1 as the headline of a receipt that
        # presents itself as an N-cold-run summary (cold_run_count, run_means,
        # cold_run_deviation), and `bitwise_deterministic` was computed, printed, and
        # allowed to gate nothing. Every published receipt happens to be bitwise
        # deterministic so no live number moves -- but the registry's own metric is NAMED
        # mean_of_run_means_tokenwise_kld and registry_add recomputes it to 1e-12, so the
        # emitter was the side that was out of step. The identical-runs special case is
        # required, not cosmetic: plain sum/len shifts a deterministic n=3 or n=5 value by
        # 1 ULP (0.027262784814670614 -> 0.02726278481467061), which would gratuitously
        # move a published digit in a project whose docs advertise agreement "to the last
        # digit".
        mean = means[0] if len(set(means)) == 1 else sum(means) / len(means)
        run_mean_spread = (max(means) - min(means)) if means else 0.0
        report_top1 = [report.get("top1_agreement") for report, _ in reports]
        summary = {
            "schema": f"malaiwah.glm53-{args.profile}-packed-kld-summary.v1",
            # STORAGE layout, not lane: the dione conversions ship TP4-sliced
            # ranks, the stock-exllamav3 (turbo) releases ship canonical HF
            # shards, the MLX conversions ship per-expert HF-named tensors, a
            # GGUF is a single-file (or split) llama.cpp container with fused
            # per-layer expert tensors, and the native lane is single-device
            # (EP8-emulated), and an NVFP4 snapshot ships canonical HF shards.
            # Suffixing every profile "-tp4" put a false storage claim in the
            # headline receipt of an artifact that is not sliced at all.
            "profile": _profile_storage_label(args.profile),
            "student_label": student_label,
            "cold_run_count": len(reports),
            # "5 cold runs, not 5 (budget; disclosed)" is what the hardcoded 5 emitted for
            # a genuine five-run campaign, and registry_add copies this string VERBATIM
            # into a published disclosure.
            "cold_run_deviation": (
                f"{len(reports)} cold runs"
                if len(reports) >= 5
                else f"{len(reports)} cold runs, not 5 (budget; disclosed)"),
            "run_means": means,
            "run_mean_spread": run_mean_spread,
            "distinct_tokenwise_kld_sha256": kld_shas,
            "bitwise_deterministic": bitwise_deterministic,
            "measured_mean_kld": mean,
            "quality_gate": {"metric": "mean_tokenwise_kld", "threshold_lt": 0.06},
            "quality_gate_passed": bool(mean < 0.06),
            "kld_report_sha256": [sha for _, sha in reports],
            # Every per-run report computes top-1 agreement; the scalar summary
            # was dropping it, and a KL number with no top-1 leaves the reader
            # unable to tell WHICH kind of divergence it is (the registry warns
            # STAT-005 about exactly that). Carried when the runs agree on it,
            # and left null rather than averaged when they do not.
            "top1_agreement": (
                report_top1[0] if len(set(report_top1)) == 1 else None),
            "per_run_top1_agreement": report_top1,
            "teacher_receipt_sha256": teacher["receipt_sha256"],
            "teacher_source": teacher_source,
            "teacher_label": teacher_label,
            # NUM-06. The per-run report computes a full per_window block -- window_id,
            # document_id, domain, role, and count/mean/std/p50/p95/p99/cvar95/max -- and
            # the summary dropped it, keeping only scalars. That block is the ONLY thing
            # that makes a published row rescoreable on a different window scope without a
            # GPU: bin/jointstd/stats.py's cluster-robust SE reads per_window[].count/
            # .mean/.std, and the BCa block bootstrap and the domain table read
            # window_id/domain/count/mean. Its absence is why the streaming BF16 floor and
            # the Dione Q4 rows carry uncertainty.method "none" with null endpoints and
            # why their calibration-clean recompute is permanently impossible -- the run
            # dirs died with the rented box. Verbatim (nested "summary" shape), because
            # bin/joint_standard.py::load_per_window normalises it and flattening here
            # would fork the shape away from the report whose sha this receipt pins.
            "per_window": reports[0][0].get("per_window"),
            "per_domain": reports[0][0].get("per_domain"),
            "qualification_window_count": reports[0][0].get("qualification_window_count"),
            "token_panel_receipt_sha256": reports[0][0].get("token_panel_receipt_sha256"),
            "position_block": reports[0][0].get("position_block"),
            # Asserting that the runs AGREE on per_window when they are bitwise identical
            # is a tautology (one distinct tokenwise digest implies identical per-window
            # means). The case that matters is the inverse: shipping run 1's block as
            # though it described the campaign when the runs are NOT identical.
            "per_window_source": (
                "run-1; identical across runs (one distinct tokenwise_kld_sha256)"
                if bitwise_deterministic else
                "run-1 ONLY; the runs are not bitwise identical and this block describes "
                "one of them"),
        }
        surface_family = _profile_surface_family(args.profile)
        if surface_family in ("dione", "exl3hf"):
            # the headline number's own receipt must carry the unsealed-source
            # disclosure and the immutable provenance pins, not only the
            # capture receipt underneath it
            from quant_pipeline.evaluation.glm53_logits import (
                CAPTURE_SCHEMA,
                sealed_json,
            )

            student_receipt = sealed_json(
                runs[0] / "capture-receipt.json", CAPTURE_SCHEMA, "receipt_sha256"
            )
            if surface_family == "exl3hf":
                for field in ("exl3hf_repo", "exl3hf_revision"):
                    if student_receipt.get(field) is None:
                        raise _fail(
                            f"--profile {args.profile}: the capture receipt carries no "
                            f"{field}; it was not produced by stream_score.py "
                            "--source exl3hf, and the headline receipt would name no "
                            "artifact"
                        )
                summary.update(
                    {
                        "student_receipt_sha256": student_receipt["receipt_sha256"],
                        "artifact_repo": student_receipt.get("exl3hf_repo"),
                        "artifact_revision": student_receipt.get("exl3hf_revision"),
                        "artifact_config_sha256": student_receipt.get("artifact_config_sha256"),
                        "artifact_index_sha256": student_receipt.get("artifact_index_sha256"),
                        "codebook": student_receipt.get("codebook"),
                        "exllamav3_version": student_receipt.get("exllamav3_version"),
                        "declared_bits": student_receipt.get("declared_bits"),
                        "declared_head_bits": student_receipt.get("declared_head_bits"),
                        "materialization_receipt_sha256": student_receipt.get(
                            "materialization_receipt_sha256"
                        ),
                        "routed_bits_decode_histogram": student_receipt.get(
                            "routed_bits_decode_histogram"
                        ),
                        "seal_disclosure": student_receipt.get("seal_disclosure"),
                    }
                )
            else:
                # Every pin the CAPTURE sealed, republished here so the
                # headline number carries its own provenance chain.  The four
                # original keys are the sealed lane's; the rest are what the
                # streaming capture adds and are simply absent (None) on a
                # sealed-lane receipt, which is why they are read with .get.
                summary.update(
                    {
                        "student_receipt_sha256": student_receipt["receipt_sha256"],
                        "dione_repo": student_receipt.get("dione_repo"),
                        "dione_revision": student_receipt.get("dione_revision"),
                        "dione_shard_hash_verification": student_receipt.get(
                            "dione_shard_hash_verification"
                        ),
                        "source_repo": student_receipt.get("source_repo"),
                        "source_revision": student_receipt.get("source_revision"),
                        "artifact_config_sha256": student_receipt.get(
                            "artifact_config_sha256"),
                        "artifact_index_sha256": student_receipt.get(
                            "artifact_index_sha256"),
                        "exl3_manifest_name": student_receipt.get("exl3_manifest_name"),
                        "exl3_manifest_sha256": student_receipt.get("exl3_manifest_sha256"),
                        "exl3_manifest_schema": student_receipt.get("exl3_manifest_schema"),
                        "codebook": student_receipt.get("codebook"),
                        "codec_family": student_receipt.get("codec_family"),
                        "declared_bits": student_receipt.get("declared_bits"),
                        "declared_head_bits": student_receipt.get("declared_head_bits"),
                        "tp_size": student_receipt.get("tp_size"),
                        "materialization_receipt_sha256": student_receipt.get(
                            "materialization_receipt_sha256"),
                        "scope_digest": student_receipt.get("scope_digest"),
                        "scope_census_sha256": student_receipt.get("scope_census_sha256"),
                        "seal_disclosure": student_receipt.get("seal_disclosure"),
                    }
                )
        elif surface_family == "tr3":
            # Same rule as Dione/turbo -- the headline receipt carries the
            # provenance pins itself -- with one difference that matters: a TR3
            # release SEALS itself, so what travels here is the VERIFICATION
            # (which claims were recomputed, and how the shard bytes were bound),
            # not an unsealed-source caveat.  The scope policy travels too,
            # because routed-experts-only with a native BF16 head is artifact
            # identity a registry row must state.
            from quant_pipeline.evaluation.glm53_logits import (
                CAPTURE_SCHEMA,
                sealed_json,
            )

            student_receipt = sealed_json(
                runs[0] / "capture-receipt.json", CAPTURE_SCHEMA, "receipt_sha256"
            )
            for field in ("tr3_repo", "tr3_revision", "seal_verification"):
                if student_receipt.get(field) is None:
                    raise _fail(
                        f"--profile {args.profile}: the capture receipt carries no {field}; "
                        "it was not produced by stream_score.py --source tr3"
                    )
            seal = student_receipt.get("seal_verification") or {}
            summary.update(
                {
                    "student_receipt_sha256": student_receipt["receipt_sha256"],
                    "artifact_repo": student_receipt.get("tr3_repo"),
                    "artifact_revision": student_receipt.get("tr3_revision"),
                    "artifact_config_sha256": student_receipt.get("artifact_config_sha256"),
                    "artifact_index_sha256": student_receipt.get("artifact_index_sha256"),
                    "codebook": student_receipt.get("codebook"),
                    "codec_family": student_receipt.get("codec_family"),
                    "exllamav3_version": student_receipt.get("exllamav3_version"),
                    "exllamav3_pin": student_receipt.get("exllamav3_pin"),
                    "declared_bits": student_receipt.get("declared_bits"),
                    "declared_head_bits": student_receipt.get("declared_head_bits"),
                    "scope_policy": student_receipt.get("scope_policy"),
                    "nonrouted_policy_declared": student_receipt.get(
                        "nonrouted_policy_declared"
                    ),
                    # TWO receipts, and conflating them would lose the one that
                    # matters: the ARTIFACT's own published seal, and the
                    # non-routed tree this run materialized from it.
                    "artifact_materialization_receipt_sha256": student_receipt.get(
                        "artifact_materialization_receipt_sha256"
                    ),
                    "materialization_receipt_sha256": student_receipt.get(
                        "materialization_receipt_sha256"
                    ),
                    "scope_census_sha256": student_receipt.get("scope_census_sha256"),
                    "routed_bits_decode_histogram": student_receipt.get(
                        "routed_bits_decode_histogram"
                    ),
                    "seal_verified": bool(seal.get("verified")),
                    "seal_checks_passed": sum(
                        1 for c in (seal.get("checks") or []) if c.get("passed")),
                    "seal_check_names": [c.get("check") for c in (seal.get("checks") or [])],
                    "shard_verification": (
                        student_receipt.get("shard_verification") or {}
                    ).get("verification"),
                    "seal_disclosure": student_receipt.get("seal_disclosure"),
                }
            )
        elif args.profile == "mlx":
            # Same rule as Dione: the headline receipt carries the unsealed-source
            # disclosure and the immutable pins itself.  It additionally carries
            # the SCOPE POLICY, because this artifact family quantizes beyond the
            # routed experts and a registry row must disclose that.
            from quant_pipeline.evaluation.glm53_logits import (
                CAPTURE_SCHEMA,
                sealed_json,
            )

            student_receipt = sealed_json(
                runs[0] / "capture-receipt.json", CAPTURE_SCHEMA, "receipt_sha256"
            )
            for field in ("mlx_repo", "mlx_revision", "mlx_scope_policy"):
                if student_receipt.get(field) is None:
                    raise _fail(
                        f"--profile mlx: the capture receipt carries no {field}; it was not "
                        "produced by stream_score.py --source mlx"
                    )
            summary.update(
                {
                    "student_receipt_sha256": student_receipt["receipt_sha256"],
                    "mlx_repo": student_receipt.get("mlx_repo"),
                    "mlx_revision": student_receipt.get("mlx_revision"),
                    "mlx_format": student_receipt.get("mlx_format"),
                    "mlx_default_bits": student_receipt.get("mlx_default_bits"),
                    "mlx_default_group_size": student_receipt.get("mlx_default_group_size"),
                    "mlx_bits_histogram": student_receipt.get("mlx_bits_histogram"),
                    "mlx_config_sha256": student_receipt.get("mlx_config_sha256"),
                    "mlx_index_sha256": student_receipt.get("mlx_index_sha256"),
                    "mlx_shard_hash_verification": student_receipt.get(
                        "mlx_shard_hash_verification"
                    ),
                    "mlx_scope_policy": student_receipt.get("mlx_scope_policy"),
                    "mlx_artifact_bytes": (
                        student_receipt.get("mlx_fetch_ledger") or {}
                    ).get("on_disk_total_bytes"),
                    "nonrouted_policy": "decoded_bf16_view_materialized_from_the_quant_snapshot",
                    "source_repo": student_receipt.get("source_repo"),
                    "source_revision": student_receipt.get("source_revision"),
                    "seal_disclosure": student_receipt.get("seal_disclosure"),
                }
            )
        elif args.profile == "gguf":
            # Same rule again: a community GGUF quantized EVERYTHING, so the
            # headline receipt has to carry the artifact identity AND the scope
            # policy -- a registry row that does not disclose "the embeddings and
            # lm_head are quantized too" is not comparable to a
            # routed-experts-only row.  A GGUF repo holds many quants at ONE
            # revision, so the FILE LIST is part of that identity.
            from quant_pipeline.evaluation.glm53_logits import (
                CAPTURE_SCHEMA,
                sealed_json,
            )

            student_receipt = sealed_json(
                runs[0] / "capture-receipt.json", CAPTURE_SCHEMA, "receipt_sha256"
            )
            summary.update(
                {
                    "student_receipt_sha256": student_receipt["receipt_sha256"],
                    "gguf_repo": student_receipt.get("gguf_repo"),
                    "gguf_revision": student_receipt.get("gguf_revision"),
                    "gguf_files": student_receipt.get("gguf_files"),
                    "gguf_file_hash_verification": student_receipt.get(
                        "gguf_file_hash_verification"
                    ),
                    "gguf_architecture": student_receipt.get("gguf_architecture"),
                    "gguf_type_census": student_receipt.get("gguf_type_census"),
                    "gguf_quant_metadata": student_receipt.get("gguf_quant_metadata"),
                    "scope_policy": student_receipt.get("scope_policy"),
                    "source_repo": student_receipt.get("source_repo"),
                    "source_revision": student_receipt.get("source_revision"),
                    "seal_disclosure": student_receipt.get("seal_disclosure"),
                }
            )
        elif args.profile == "nvfp4":
            # same rule as dione: the headline receipt itself carries the
            # provenance pins, the MEASURED scope policy (what the artifact
            # actually quantizes) and the activation caveat, all lifted from
            # the sealed capture receipt's streaming_disclosure.nvfp4 block -
            # never re-asserted here.
            from quant_pipeline.evaluation.glm53_logits import (
                CAPTURE_SCHEMA,
                sealed_json,
            )

            student_receipt = sealed_json(
                runs[0] / "capture-receipt.json", CAPTURE_SCHEMA, "receipt_sha256"
            )
            block = (student_receipt.get("streaming_disclosure") or {}).get("nvfp4")
            if not isinstance(block, dict):
                raise _fail(
                    "run-1 capture receipt carries no streaming_disclosure.nvfp4 block - "
                    "these runs were not produced by stream_score.py --source nvfp4"
                )
            scope = block.get("scope_policy") or {}
            activations = block.get("activations") or {}
            summary.update(
                {
                    "student_receipt_sha256": student_receipt["receipt_sha256"],
                    "nvfp4_repo": block.get("nvfp4_repo"),
                    "nvfp4_revision": block.get("nvfp4_revision"),
                    "nvfp4_layout": block.get("layout"),
                    "nvfp4_config_format": block.get("config_format"),
                    "nvfp4_producer": block.get("producer"),
                    "nvfp4_quant_weights": block.get("quant_weights"),
                    "nvfp4_config_sha256": block.get("config_sha256"),
                    "nvfp4_index_sha256": block.get("index_sha256"),
                    "nvfp4_shard_hash_verification": block.get("shard_hash_verification"),
                    "scope_policy": scope,
                    "scope_policy_note": "%s | %s" % (
                        scope.get("quantized_scope"), scope.get("nonrouted_policy")),
                    "activations": activations,
                    "activations_disclosure": activations.get("disclosure"),
                    "seal_disclosure": block.get("seal_disclosure"),
                }
            )
        _atomic_json(args.out.resolve(), summary)
        print(
            json.dumps(
                {
                    "measured_mean_kld": mean,
                    "quality_gate_passed": summary["quality_gate_passed"],
                },
                sort_keys=True,
            )
        )

    if args.comparison_out:
        _comparison_table(
            args.comparison_out.resolve(),
            args.fp8_baseline,
            args.k4_baseline,
            args.out.resolve().parent,
        )
        print(f"comparison table written: {args.comparison_out}")
    _ = bits
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
