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
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(value, dict)
            and value.get("schema") == CAPTURE_SCHEMA
            and value.get("capture_role") == "bf16_teacher"
        ):
            return path
    raise _fail(
        f"no sealed bf16_teacher capture receipt found under {teacher_root} "
        "(expected the downloaded teacher final-window tree)"
    )


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
            teacher_logits = _load_slice(Path(teacher_row["path"]), start, stop)
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
    report = {
        "schema": "quant-pipeline.glm53-packed-student-kld.v1",
        "teacher_receipt_sha256": teacher["receipt_sha256"],
        "student_receipt_sha256": student["receipt_sha256"],
        "student_label": student_label,
        "student_checkpoint_identity_sha256": student["checkpoint_identity_sha256"],
        "runtime_reader_sha256": student["runtime_reader_sha256"],
        "token_panel_receipt_sha256": teacher["token_panel_receipt_sha256"],
        "teacher_backend_identity_sha256": teacher["backend_identity_sha256"],
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
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    write_json(report_path, report)
    return report_path


def _comparison_table(
    out_path: Path,
    fp8_baseline: float,
    k4_baseline: float,
    receipts_dir: Path,
) -> None:
    rows = [
        "| model | routed bpw | size | mean tokenwise KLD vs BF16 teacher "
        "(25 sealed windows, 51,175 pos, fp64) | provenance |",
        "|---|---|---|---|---|",
        f"| zai-org FP8 (as served) | 8 | 328 GB | {fp8_baseline:.6f} "
        "| our fidelity-suite baseline |",
        f"| brandonmusic K4 (EXL3/TR3-MCG) | 4.01 | 163.6 GiB | {k4_baseline:.6f} "
        "(five-run mean, stddev 0) | his sealed receipts |",
    ]
    for profile, bpw, size in (
        ("k6", "6.01", "236.1 GiB"),
        ("k8", "8.01", "308.7 GiB"),
        ("k6k8", "6.68", "260.3 GiB"),
        ("dione-q4", "4.0 (TP4-sliced)", "174.5 GiB"),
        ("dione-3.0bpw", "3.0 (TP4-sliced)", "~139 GiB"),
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
            }[profile]
            rows.append(
                f"| **{label}** | {bpw} | {size} | **{mean:.6f}** (gate < 0.06 {gate}) "
                "| this campaign |"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True,
                        choices=("k6", "k6-stream", "k8", "k6k8", "dione-q4", "dione-3.0bpw",
                                 "native-bf16"))
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

    bits = {"k6": 6, "k6-stream": 6, "k8": 8}.get(args.profile)
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
        # the BF16 FLOOR of the streaming lane: the identical capture with the
        # routed experts read straight from the official checkpoint and no codec
        # in the path (stream_score.py --source native).  Subtracting this mean
        # from a quant's mean leaves that quant's quantization-attributable error.
        "native-bf16": "native-bf16",
    }[args.profile]
    # a native run is not a packed student and does not claim to be one
    expected_capture_role = (
        "native_bf16_student" if args.profile == "native-bf16" else "packed_student"
    )
    runs = [path.resolve() for path in args.runs]
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
            # the native lane is single-device (EP8-emulated), never TP4
            "profile": (
                "native-bf16-stream" if args.profile == "native-bf16" else f"{args.profile}-tp4"
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
        }
        if args.profile.startswith("dione"):
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
