#!/usr/bin/env python3
"""Turn a native-BF16 floor run into the quantization-attributable-error receipt.

The streaming lane's panel mean for a quant is NOT that quant's quantization
error.  It is

    KLD(teacher || our-stack-running-the-quant)
      = floor  +  quantization-attributable error

where the FLOOR is what it costs to compare OUR stack's forward against the
TEACHER's logits with no quantization at all: a different process topology, a
different expert-combine order, a different box.  ``stream_score.py
--source native`` measures that floor directly by running the identical capture
with the routed experts read straight from the official BF16 checkpoint.

This tool subtracts it, on the panel mean and window by window, and seals the
result so a model card or a forum post can quote a number with a receipt behind
it instead of an inference.

    bf16_floor_summary.py \
        --floor-kld  <native-bf16-kld.json> \
        --floor-run  <floor-run1> [--floor-run <floor-run2> ...] \
        --quant k6:<stream-k6-kld.json>:<k6 run1 kld-report.json> \
        --quant k8:<stream-k8-kld.json>:<k8 run1 kld-report.json> \
        --cost <cost.json> --out-json BF16-FLOOR.json --out-md BF16-FLOOR.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCHEMA = "malaiwah.glm53-bf16-floor-attribution.v1"


def _fail(message: str) -> "SystemExit":
    print(f"bf16_floor_summary: ERROR: {message}", file=sys.stderr)
    return SystemExit(1)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def seal(body: Dict[str, Any], field: str = "receipt_sha256") -> Dict[str, Any]:
    body = dict(body)
    body[field] = ""
    body[field] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def window_means(report: Dict[str, Any]) -> Dict[str, float]:
    return {row["window_id"]: float(row["summary"]["mean"]) for row in report["per_window"]}


def domain_means(report: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(value["mean"]) for name, value in report["per_domain"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floor-kld", type=Path, required=True,
                        help="the native-bf16 profile summary from k6_kld_report.py")
    parser.add_argument("--floor-run", type=Path, action="append", required=True,
                        help="a cold-run directory (kld-report.json, capture-receipt.json, "
                             "backend.json, plan.json); repeat per cold run")
    parser.add_argument("--quant", action="append", default=[],
                        help="label:<summary json>:<one run's kld-report.json>")
    parser.add_argument("--cost", type=Path, help="cost accounting json to embed verbatim")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    floor_summary = read_json(args.floor_kld)
    if floor_summary.get("student_label") != "native-bf16":
        raise _fail(
            f"--floor-kld is labelled {floor_summary.get('student_label')!r}, not 'native-bf16'"
        )
    runs: List[Dict[str, Any]] = []
    for run_dir in args.floor_run:
        runs.append(
            {
                "run_dir": str(run_dir),
                "kld": read_json(run_dir / "kld-report.json"),
                "capture": read_json(run_dir / "capture-receipt.json"),
                "backend": read_json(run_dir / "backend.json"),
                "plan": read_json(run_dir / "plan.json"),
            }
        )
    run_means = [float(row["kld"]["summary"]["mean"]) for row in runs]
    if len(set(run_means)) != 1:
        print(f"WARNING: cold runs disagree: {run_means}", file=sys.stderr)
    floor = float(floor_summary["measured_mean_kld"])
    if abs(floor - run_means[0]) > 0:
        raise _fail("summary mean does not equal the per-run mean")

    reference = runs[0]["kld"]
    floor_windows = window_means(reference)
    floor_domains = domain_means(reference)

    quants: List[Dict[str, Any]] = []
    for spec in args.quant:
        label, summary_path, report_path = spec.split(":", 2)
        summary = read_json(Path(summary_path))
        report = read_json(Path(report_path))
        mean = float(summary["measured_mean_kld"])
        if report["token_panel_receipt_sha256"] != reference["token_panel_receipt_sha256"]:
            raise _fail(f"{label} was scored on a different token panel")
        if report["teacher_receipt_sha256"] != reference["teacher_receipt_sha256"]:
            raise _fail(f"{label} was scored against a different teacher")
        per_window = window_means(report)
        if set(per_window) != set(floor_windows):
            raise _fail(f"{label} window set differs from the floor run")
        attributable = mean - floor
        quants.append(
            {
                "label": label,
                "student_label": summary.get("student_label"),
                "panel_mean_kld": mean,
                "floor_subtracted": floor,
                "quantization_attributable_kld": attributable,
                "fraction_of_panel_mean_that_is_floor": floor / mean,
                "windows_above_floor": sum(
                    1 for window, value in per_window.items() if value > floor_windows[window]
                ),
                "window_count": len(per_window),
                "per_window_attributable": {
                    window: per_window[window] - floor_windows[window]
                    for window in sorted(per_window)
                },
                "per_domain_attributable": {
                    name: value - floor_domains[name]
                    for name, value in domain_means(report).items()
                },
                "source_summary_sha256": hashlib.sha256(
                    Path(summary_path).read_bytes()
                ).hexdigest(),
                "source_report_sha256": report["report_sha256"],
                "run_means": summary.get("run_means"),
                "bitwise_deterministic": summary.get("bitwise_deterministic"),
            }
        )

    ratios = {}
    for left in quants:
        for right in quants:
            if left is right:
                continue
            denominator = right["quantization_attributable_kld"]
            key = f"{left['label']}_over_{right['label']}"
            ratios[key] = {
                "panel_mean_ratio": left["panel_mean_kld"] / right["panel_mean_kld"],
                "attributable_ratio": (
                    left["quantization_attributable_kld"] / denominator
                    if denominator
                    else None
                ),
            }

    backend = runs[0]["backend"]
    plan = runs[0]["plan"]
    receipt = {
        "schema": SCHEMA,
        "what_this_is": (
            "the measurement floor of the single-device streaming fidelity lane, and each "
            "quant's panel mean with that floor subtracted"
        ),
        "floor": {
            "mean_tokenwise_kld": floor,
            "student_label": "native-bf16",
            "cold_runs": len(runs),
            "run_means": run_means,
            "bitwise_deterministic": floor_summary.get("bitwise_deterministic"),
            "distinct_tokenwise_kld_sha256": floor_summary.get("distinct_tokenwise_kld_sha256"),
            "summary": reference["summary"],
            "per_domain": reference["per_domain"],
            "per_window": {window: floor_windows[window] for window in sorted(floor_windows)},
            "top1_agreement_with_teacher": reference["top1_agreement"],
        },
        "quants": quants,
        "ratios": ratios,
        "measurement": {
            "panel": "sealed 25-window final panel, 51,175 jointly-valid causal positions",
            "token_panel_receipt_sha256": reference["token_panel_receipt_sha256"],
            "teacher_receipt_sha256": reference["teacher_receipt_sha256"],
            "estimator": "tokenwise KL(teacher || student), float64",
            "kld_direction": reference["kld_direction"],
            "lane": "single-device streaming, EP8-emulated, --reduce-order fp32",
            "student_checkpoint_identity_sha256": reference["student_checkpoint_identity_sha256"],
            "runtime_reader_sha256": reference["runtime_reader_sha256"],
            "model_revision": plan.get("model_revision"),
            "inventory_sha256": plan.get("inventory_sha256"),
            "routed_tensor_count": (plan.get("native_routed_layout") or {}).get(
                "routed_tensor_count"
            ),
            "routed_shard_count": (plan.get("native_routed_layout") or {}).get(
                "routed_shard_count"
            ),
            "device": backend.get("device_name"),
            "torch_version": backend.get("torch_version"),
            "transformers_version": backend.get("transformers_version"),
            "grouped_mm_kernel": backend.get("grouped_mm_kernel"),
            "numeric_policy": backend.get("numeric_policy"),
            "peak_device_allocated_bytes": backend.get("peak_device_allocated_bytes"),
            "floor_run_kld_report_sha256": [row["kld"]["report_sha256"] for row in runs],
            "floor_capture_receipt_sha256": [
                row["capture"]["receipt_sha256"] for row in runs
            ],
            "floor_backend_identity_sha256": [
                row["backend"]["backend_identity_sha256"] for row in runs
            ],
        },
        "disclosure": {
            "publishable_as_reproduction": False,
            "why": (
                "the floor and the quants were measured on the SAME lane, the same box class, "
                "the same torch/transformers build and the same panel, so the subtraction is "
                "apples to apples; but this lane is not the sealed 8xH200 EP8 protocol, and its "
                "own offset against that protocol was measured separately (-8.5e-06 on K6)"
            ),
            "floor_is_lane_specific": (
                "a floor measured on this lane bounds THIS lane. Another stack's floor against "
                "the same teacher will differ; a cross-stack floor of 0.012712 was measured "
                "independently on a different lane"
            ),
            "no_decode": (plan.get("streaming_disclosure") or {}).get("no_decode"),
            "routed_weight_source": (plan.get("streaming_disclosure") or {}).get(
                "routed_weight_source"
            ),
        },
    }
    if args.cost:
        receipt["cost"] = read_json(args.cost)
    receipt = seal(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- markdown -------------------------------------------------------
    lines: List[str] = []
    lines.append("# The BF16 floor of the GLM-5.3-Flash streaming fidelity lane")
    lines.append("")
    lines.append(
        f"**Floor = {floor:.9f} nats** (mean tokenwise KL against the sealed BF16 teacher, "
        f"25-window panel, 51,175 positions, fp64, {len(runs)} cold "
        f"run{'s' if len(runs) != 1 else ''}, "
        f"cross-run bitwise identical: "
        f"{str(floor_summary.get('bitwise_deterministic')).lower()})."
    )
    lines.append("")
    lines.append(
        "This is what our stack costs *with no quantization at all*: the routed experts are "
        "the official BF16 checkpoint tensors, read by name, with no codec in the path. "
        "Everything else -- panel, teacher, estimator, EP8 emulation, reduce order, "
        "grouped_mm kernel, fp32 logit storage -- is the identical code the K6 and K8 "
        "measurements ran. The only difference is where the expert weights come from."
    )
    lines.append("")
    lines.append("## Quantization-attributable error")
    lines.append("")
    lines.append("| student | panel mean KLD | floor | quantization-attributable | floor share |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| native BF16 (the floor) | {floor:.9f} | {floor:.9f} | 0 (by construction) | 100% |"
    )
    for quant in quants:
        lines.append(
            f"| {quant['label'].upper()} ({quant['student_label']}) "
            f"| {quant['panel_mean_kld']:.9f} | {floor:.9f} "
            f"| **{quant['quantization_attributable_kld']:.9f}** "
            f"| {quant['fraction_of_panel_mean_that_is_floor'] * 100:.1f}% |"
        )
    lines.append("")
    for key, value in sorted(ratios.items()):
        left, right = key.split("_over_")
        if value["attributable_ratio"] is None:
            continue
        lines.append(
            f"- **{left.upper()} / {right.upper()}**: {value['panel_mean_ratio']:.3f}x on the raw "
            f"panel mean, **{value['attributable_ratio']:.2f}x** on quantization-attributable "
            f"error."
        )
    lines.append("")
    lines.append("## Why this matters")
    lines.append("")
    lines.append(
        "Raw panel means of adjacent bit-widths look nearly identical because most of the "
        "number is the floor, not the quantizer. Subtracting a measured floor is what turns "
        "\"11% better\" into a statement about the codec."
    )
    lines.append("")
    lines.append("## Per-domain")
    lines.append("")
    header = "| domain | floor |" + "".join(f" {q['label'].upper()} attributable |" for q in quants)
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(quants))
    for domain in sorted(floor_domains):
        row = f"| {domain} | {floor_domains[domain]:.6f} |"
        for quant in quants:
            row += f" {quant['per_domain_attributable'][domain]:.6f} |"
        lines.append(row)
    lines.append("")
    lines.append("## Per-window")
    lines.append("")
    header = "| window | floor |" + "".join(f" {q['label'].upper()} attributable |" for q in quants)
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(quants))
    for window in sorted(floor_windows):
        row = f"| {window} | {floor_windows[window]:.6f} |"
        for quant in quants:
            row += f" {quant['per_window_attributable'][window]:.6f} |"
        lines.append(row)
    lines.append("")
    if "cost" in receipt:
        lines.append("## What this measurement cost")
        lines.append("")
        cost = receipt["cost"]
        for key in sorted(cost):
            value = cost[key]
            if isinstance(value, (dict, list)):
                lines.append(f"- **{key}**: `{json.dumps(value, sort_keys=True)}`")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- model revision `{plan.get('model_revision')}`")
    lines.append(f"- release inventory `{plan.get('inventory_sha256')}`")
    lines.append(f"- token panel receipt `{reference['token_panel_receipt_sha256']}`")
    lines.append(f"- teacher receipt `{reference['teacher_receipt_sha256']}`")
    lines.append(
        f"- floor student identity `{reference['student_checkpoint_identity_sha256']}`"
    )
    lines.append(f"- floor source identity `{reference['runtime_reader_sha256']}`")
    lines.append(
        f"- device `{backend.get('device_name')}`, torch `{backend.get('torch_version')}`, "
        f"transformers `{backend.get('transformers_version')}`"
    )
    lines.append(f"- receipt seal `{receipt['receipt_sha256']}`")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The floor is **lane-specific**. It bounds this stack against this teacher. A "
        "cross-stack floor measured on a different lane came out at 0.012712."
    )
    lines.append(
        "- `publishable_as_reproduction: false`. This lane agrees with the sealed 8xH200 "
        "protocol to -8.5e-06 on K6 but is an independent measurement, not a bitwise "
        "reproduction."
    )
    lines.append(
        "- The subtraction assumes the floor and the quant errors add. They are measured, not "
        "modelled: a quant whose panel mean fell *below* the floor would falsify that."
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "floor": floor,
                "attributable": {
                    quant["label"]: quant["quantization_attributable_kld"] for quant in quants
                },
                "receipt_sha256": receipt["receipt_sha256"],
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
