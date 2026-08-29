#!/usr/bin/env python3
"""fp64 tokenwise KLD vs the fp32 teacher + five-cold-run aggregation (K6/K6K8).

Adapted from brandonmusic's measure_glm53_packed_student_kld.py and
aggregate_glm53_five_run_kld.py, joined into the single CLI stage_k6.sh pins:

  k6_kld_report.py --profile k6 --teacher <tree> --runs run1 ... run5 \
      --fp8-baseline 0.020615 --k4-baseline 0.024555 \
      --out k6-packed-kld.json --five-run-out k6-five-run-kld.json \
      --comparison-out comparison-table.md

Per run it computes (or resumes) the sealed per-run KLD report
(quant-pipeline.glm53-packed-student-kld.v1): exact fp64 log-softmax,
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
    print(f"k6_kld_report: ERROR: {message}", file=sys.stderr, flush=True)
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
        "schema": "quant-pipeline.glm53-packed-student-kld.v1",
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
        # community MLX affine snapshots: bit mix and size are properties of the
        # artifact, so both are READ from the receipt instead of hardcoded here
        ("mlx", None, None),
    ):
        receipt_path = receipts_dir / f"{profile}-packed-kld.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            mean = receipt.get("measured_mean_kld")
            gate = "GREEN" if receipt.get("quality_gate_passed") else "RED"
            label = {
                "k6": "malaiwah K6",
                "k8": "malaiwah K8 (uniform)",
                "k6k8": "malaiwah K6K8 (down@8)",
                "dione-q4": "0xSero Dione Q4 (EXL3 K4, unsealed source)",
                "dione-3.0bpw": "0xSero Dione 3.0bpw (EXL3 K3, unsealed source)",
                "turbo-4.05bpw": "turboderp 4.05bpw (stock EXL3 mul1, quantized head, unsealed source)",
                "turbo-3.05bpw": "turboderp 3.05bpw (stock EXL3 mul1, quantized head, unsealed source)",
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
                                 "turbo-4.05bpw", "turbo-3.05bpw",
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
    _import_pipeline(args.pipeline_root)

    from quant_pipeline.core.artifacts import sha256_file
    from quant_pipeline.evaluation.glm53_logits import load_capture_receipt

    bits = {"k6": 6, "k6-stream": 6, "k8": 8, "nvfp4": 4}.get(args.profile)
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
    means = [float(report["summary"]["mean"]) for report, _ in reports]
    kld_shas = sorted({str(report["tokenwise_kld_sha256"]) for report, _ in reports})
    print(
        json.dumps(
            {
                "runs": len(reports),
                "run_means": means,
                "distinct_tokenwise_kld_sha256": kld_shas,
                "bitwise_deterministic": len(kld_shas) == 1,
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
        mean = means[0]
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
            "profile": (
                "native-bf16-stream" if args.profile == "native-bf16"
                else "mlx-stream" if args.profile == "mlx"
                else "gguf-stream" if args.profile == "gguf"
                else "nvfp4-stream" if args.profile == "nvfp4"
                else f"{args.profile}-hf-sharded" if args.profile.startswith("turbo")
                else f"{args.profile}-tp4"
            ),
            "student_label": student_label,
            "cold_run_count": len(reports),
            "cold_run_deviation": f"{len(reports)} cold runs, not 5 (budget; disclosed)",
            "run_means": means,
            "distinct_tokenwise_kld_sha256": kld_shas,
            "bitwise_deterministic": len(kld_shas) == 1,
            "measured_mean_kld": mean,
            "quality_gate": {"metric": "mean_tokenwise_kld", "threshold_lt": 0.06},
            "quality_gate_passed": bool(mean < 0.06),
            "kld_report_sha256": [sha for _, sha in reports],
            "teacher_receipt_sha256": teacher["receipt_sha256"],
            "teacher_source": teacher_source,
            "teacher_label": teacher_label,
        }
        if args.profile.startswith(("dione", "turbo")):
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
            if args.profile.startswith("turbo"):
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
