#!/usr/bin/env python3
"""Hidden-replay equivalence driver for the GLM-5.3-Flash streaming lane.

Phaelon's sign-off protocol, aligned with Festr's kimi-k3 hidden-replay
qualification (github.com/local-inference-lab/rtx6kpro,
models/kimi-k3/distribution-fidelity-1024x2048.md; dataset
festr2/kimi-k3-distribution-fidelity-1024x2048-v1):

  capture   one streaming forward per window emits BOTH the full fp32 logits
            (path A -- byte-identical to a plain stream_score run: this driver
            wraps stream_score.main() and changes NOTHING in its path) and the
            post-final-RMSNorm bf16 hidden states (path B), captured as the
            lm_head module's INPUT via a torch forward pre-hook.
  compare   offline comparator: replay logits' = hidden @ head^T in fp32
            (both sides upcast from bf16), per-token KL(live || replayed) over
            the full vocabulary in fp64, top-1 agreement, vocab-chunk
            invariance, panel mean KLD vs the teacher through BOTH paths, and
            cross-run determinism evidence on tensor content.

THE CUT (state it, never assume it): the captured tensor is the bf16 hidden
state AFTER the text model's final RMSNorm (model.language_model.norm) and
immediately BEFORE the lm_head matmul -- i.e. the lm_head input.  Replay
applies the head ONLY.  This matches Festr's kimi-k3 convention.  Our earlier
head-extraction artifact (head/head-extraction.json in
malaiwah/GLM-5.3-Flash-fidelity-suite-v1) shipped final_norm.safetensors
NEXT TO the head for a norm+head replay; this protocol does NOT apply
final_norm at replay time because the capture already sits after it.

GLM-5.3-Flash's residual/norm output is natively torch.bfloat16 (the model is
built and run in bf16; the hook asserts the dtype at every window), so the
bf16 capture is LOSSLESS -- no rounding is introduced by storing it.

This file is deliberately a NEW module: k6/tools/stream_score.py is not
edited (another workflow owns large in-flight changes there).  The wrapper
monkeypatches stream_score.build_streaming_model at run time to attach the
hook; the sealed capture path itself is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOLS_DIR = Path(__file__).resolve().parent

HIDDEN_CAPTURE_SCHEMA = "malaiwah.glm53-hidden-capture.v1"
COMPARATOR_SCHEMA = "malaiwah.glm53-hidden-replay-comparator.v1"

# NUM-11 / CC-18. Two problems, one line. The value was "post_final_rmsnorm_pre_lm_head",
# which is not in the suite's own vocabulary -- docs/schema/fidelity-dataset.schema.json
# enumerates semantic_point as after_final_rmsnorm_before_lm_head |
# lm_head_output_before_sampling | live_lm_head_output_before_sampling, and dsvalidate
# REFUSES the old string. And the field was named `cut_point`, which no consumer reads.
# Same cut, spelled the way the schema and every reader expect.
SEMANTIC_POINT = "after_final_rmsnorm_before_lm_head"
CUT_POINT = SEMANTIC_POINT   # retained: existing receipts and the prose cut_statement
CUT_STATEMENT = (
    "the bf16 tensor after the text model's final RMSNorm "
    "(model.language_model.norm) and immediately before the lm_head matmul, "
    "captured as the lm_head module's input via a forward pre-hook; replay "
    "applies the head ONLY (no final_norm at replay time -- the capture "
    "already sits after it), matching Festr's kimi-k3 hidden-replay cut"
)

HIDDEN_WIDTH = 4096
EXPECTED_VOCAB = 154880


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"hidden_replay: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(path: Path) -> str:
    """sha256 of the safetensors TENSOR REGION only (header excluded).

    Whole-file hashes differ between bit-identical runs because __metadata__
    carries cold_run and backend identity; determinism is defined on tensor
    content (campaign lesson 27).  Same convention as stage_campaign.sh's
    payload-shas.json.
    """
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_len)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_content_sha256(tensor) -> str:
    """sha256 over the tensor's raw little-endian storage bytes (bf16 -> uint16 view)."""
    import torch

    flat = tensor.detach().contiguous()
    if flat.dtype == torch.bfloat16:
        flat = flat.view(torch.uint16)
    return hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()


def _mini_flag(argv: List[str], flag: str) -> Optional[str]:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def run_capture(args: argparse.Namespace, stream_argv: List[str]) -> int:
    if _mini_flag(stream_argv, "--sweep") is not None:
        raise _fail("hidden capture refuses --sweep: the sweep re-runs the forward and the "
                    "lm_head hook would interleave sweep hiddens with primary ones")
    store_positions = _mini_flag(stream_argv, "--store-positions")
    if store_positions not in (None, "all"):
        raise _fail("hidden capture requires --store-positions all: a sampled hidden set "
                    "cannot serve as a replay teacher")
    out_value = _mini_flag(stream_argv, "--out")
    if out_value is None:
        raise _fail("could not find --out in the stream_score argv")
    out_dir = Path(out_value).resolve()
    token_panel = _mini_flag(stream_argv, "--token-panel")
    if token_panel is None:
        raise _fail("hidden capture requires --token-panel in the stream_score argv "
                    "(the mask npy paths are resolved from the sealed panel receipt)")

    sys.path.insert(0, str(TOOLS_DIR))
    import stream_score  # noqa: E402  (the sealed streaming engine, unmodified)
    import torch  # noqa: E402

    tap: List[Any] = []
    head_facts: Dict[str, Any] = {}
    original_build = stream_score.build_streaming_model

    def tapped_build(**kwargs):
        model, record = original_build(**kwargs)
        head = model.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            raise _fail("model.get_output_embeddings() did not return the lm_head module")
        weight = head.weight
        if tuple(weight.shape) != (EXPECTED_VOCAB, HIDDEN_WIDTH):
            raise _fail(f"lm_head weight shape {tuple(weight.shape)} != "
                        f"({EXPECTED_VOCAB}, {HIDDEN_WIDTH})")
        if weight.dtype != torch.bfloat16:
            raise _fail(f"lm_head weight dtype {weight.dtype} != torch.bfloat16")
        if getattr(head, "bias", None) is not None:
            raise _fail("lm_head carries a bias; the replay contract assumes none")
        head_facts.update(
            {
                "module_class": type(head).__name__,
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
                "bias": None,
                "weight_content_sha256": tensor_content_sha256(weight),
                "hook": "torch.nn.Module.register_forward_pre_hook on model.get_output_embeddings()",
            }
        )

        def pre_hook(module, hook_args):
            hidden = hook_args[0]
            if not torch.is_tensor(hidden):
                raise _fail("lm_head pre-hook received a non-tensor input")
            if hidden.dtype != torch.bfloat16:
                raise _fail(f"lm_head input dtype {hidden.dtype} != torch.bfloat16 -- the "
                            "'capture is lossless' claim would be false; refusing")
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != HIDDEN_WIDTH:
                raise _fail(f"lm_head input shape {tuple(hidden.shape)} is not [1, seq, {HIDDEN_WIDTH}]")
            tap.append(hidden.detach().squeeze(0).to("cpu", copy=True))

        head.register_forward_pre_hook(pre_hook)
        return model, record

    stream_score.build_streaming_model = tapped_build
    saved_argv = sys.argv
    sys.argv = ["stream_score.py"] + list(stream_argv)
    started = time.monotonic()
    try:
        rc = stream_score.main()
    except SystemExit as exit_:  # stream_score raises SystemExit on failure paths
        rc = int(exit_.code or 0)
    finally:
        sys.argv = saved_argv
        stream_score.build_streaming_model = original_build
    if rc != 0:
        raise _fail(f"stream_score.main() exited {rc}; no hiddens are written for a failed capture", rc)

    receipt_path = out_dir / "capture-receipt.json"
    if not receipt_path.is_file():
        raise _fail(f"stream capture receipt missing at {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt.get("logit_files") or []
    if len(tap) != len(rows):
        raise _fail(f"hook fired {len(tap)} times but the capture wrote {len(rows)} windows -- "
                    "an extra or missing forward would misalign hiddens; refusing")

    panel_receipt = json.loads(Path(token_panel).read_text(encoding="utf-8"))
    mask_by_digest = {row["sha256"]: row["path"] for row in panel_receipt.get("artifacts", [])}

    import numpy as np  # noqa: E402
    from safetensors.torch import save_file  # noqa: E402

    hiddens_dir = out_dir / "hiddens"
    hiddens_dir.mkdir(exist_ok=True)
    hidden_rows: List[Dict[str, Any]] = []
    inference_guard = torch.inference_mode()
    inference_guard.__enter__()  # the tapped tensors are inference tensors; stay in the mode
    for index, row in enumerate(rows):
        mask_path = mask_by_digest.get(row["attention_mask_sha256"])
        if mask_path is None:
            raise _fail(f"window {row['window_id']}: attention-mask digest not in the token panel receipt")
        mask = np.load(mask_path, allow_pickle=False)
        causal = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        hidden_full = tap[index]
        if hidden_full.shape[0] != mask.shape[0]:
            raise _fail(f"window {row['window_id']}: hidden seq {hidden_full.shape[0]} != mask {mask.shape[0]}")
        selected = hidden_full[:-1][torch.from_numpy(causal)]
        if tuple(selected.shape) != (int(row["prediction_positions"]), HIDDEN_WIDTH):
            raise _fail(f"window {row['window_id']}: selected hidden shape {tuple(selected.shape)} != "
                        f"({row['prediction_positions']}, {HIDDEN_WIDTH})")
        selected = selected.contiguous()
        # mirror the logits file naming: logits/window-%04d.safetensors -> hiddens/window-%04d.safetensors
        file_name = Path(row["path"]).name
        hidden_path = (hiddens_dir / file_name).resolve()
        save_file(
            # NUM-11. v1's normative key is `hidden_states` (dsformat.TENSOR_KEY_HIDDEN);
            # `hidden` is only tolerated as a pre-v1 legacy on ingest. Writing the legacy
            # name meant our own captures needed the legacy path, and the dataset adapter
            # then declared a key the file did not carry (CC-03). _load_hidden still
            # accepts both, so trees captured before this change keep working.
            {"hidden_states": selected},
            hidden_path,
            metadata={
                "capture_role": "hidden_states_pre_lm_head",
                "cut_point": CUT_POINT,
                "semantic_point": SEMANTIC_POINT,
                "window_id": row["window_id"],
                "cold_run": str(receipt.get("cold_run")),
                "token_ids_sha256": row["token_ids_sha256"],
                "attention_mask_sha256": row["attention_mask_sha256"],
                "dtype": "bfloat16",
            },
        )
        hidden_rows.append(
            {
                "window_id": row["window_id"],
                "path": str(hidden_path),
                "bytes": hidden_path.stat().st_size,
                "sha256": sha256_file(hidden_path),
                "payload_sha256": payload_sha256(hidden_path),
                "tensor_content_sha256": tensor_content_sha256(selected),
                "prediction_positions": int(row["prediction_positions"]),
                "logit_file_sha256": row["sha256"],
            }
        )
        print(json.dumps({"hidden_window": row["window_id"], "bytes": hidden_rows[-1]["bytes"]},
                         sort_keys=True), flush=True)
    inference_guard.__exit__(None, None, None)

    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes  # noqa: E402

    run_digest = hashlib.sha256(
        "".join(row["payload_sha256"] for row in hidden_rows).encode("ascii")
    ).hexdigest()
    body: Dict[str, Any] = {
        "schema": HIDDEN_CAPTURE_SCHEMA,
        "cut_point": CUT_POINT,
        "semantic_point": SEMANTIC_POINT,
        "cut_statement": CUT_STATEMENT,
        "hidden_dtype": "bfloat16",
        "hidden_dtype_lossless": (
            "the model computes and hands the lm_head a bf16 tensor; storing bf16 "
            "introduces no rounding (asserted per window by the hook)"
        ),
        "hidden_width": HIDDEN_WIDTH,
        "cold_run": receipt.get("cold_run"),
        "window_count": len(hidden_rows),
        "hidden_files": hidden_rows,
        "hiddens_payload_digest_sha256": run_digest,
        "lm_head": head_facts,
        "stream_capture_receipt_sha256": receipt.get("receipt_sha256"),
        "stream_capture_receipt_path": str(receipt_path),
        "student_label": receipt.get("student_label"),
        "capture_wrapper": {
            "file": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "stream_score_sha256": sha256_file(TOOLS_DIR / "stream_score.py"),
            "mechanism": "monkeypatched stream_score.build_streaming_model; forward pre-hook on lm_head; "
                         "the stream_score capture path itself is unmodified",
        },
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    body["receipt_sha256"] = sha256_bytes(canonical_json(body))
    receipt_out = out_dir / "hidden-capture.json"
    receipt_out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"hidden_capture": str(receipt_out),
                      "windows": len(hidden_rows),
                      "hiddens_payload_digest_sha256": run_digest}, sort_keys=True), flush=True)
    return 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def _load_hidden(path: Path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        # accept both: trees captured before NUM-11 carry the legacy `hidden`.
        for key in ("hidden_states", "hidden"):
            if key in handle.keys():
                return handle.get_tensor(key)
        raise _fail("hidden capture %s carries neither 'hidden_states' nor 'hidden'"
                    % path)


def _replay_logits(hidden_bf16, head32_t, device, vocab_chunk: Optional[int] = None):
    """logits' = hidden @ head^T in fp32 (both sides upcast from bf16).

    ``head32_t`` is the fp32 head transposed to [hidden, vocab].  With
    ``vocab_chunk`` set, the matmul runs in vocab-dimension chunks and the
    pieces are concatenated -- the invariance probe.
    """
    import torch

    hidden32 = hidden_bf16.to(device).float()
    if vocab_chunk is None:
        return hidden32 @ head32_t
    pieces = []
    for start in range(0, head32_t.shape[1], vocab_chunk):
        pieces.append(hidden32 @ head32_t[:, start:start + vocab_chunk])
    return torch.cat(pieces, dim=1)


def summarize_tokenwise(values) -> Dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "positions": int(array.size),
        "mean": float(array.mean()),
        "max": float(array.max()),
        "p99": float(np.quantile(array, 0.99)),
        "p99_9": float(np.quantile(array, 0.999)),
        "quantile_method": "numpy.quantile linear interpolation",
    }


def run_compare(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(TOOLS_DIR))
    import numpy as np
    import torch

    import stream_score  # for resolve_device / apply_numeric_policy (reuse, not reimplementation)
    import kld_report  # for _token_kld / teacher resolution (the sealed estimator, reused)

    device = stream_score.resolve_device(args.device)
    numeric_policy = stream_score.apply_numeric_policy(device)
    device_str = str(device)

    # ---- head ------------------------------------------------------------
    head_path = args.head.resolve()
    from safetensors import safe_open

    with safe_open(head_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise _fail(f"head file must carry exactly one tensor, has {keys}")
        head = handle.get_tensor(keys[0])
    if tuple(head.shape) != (EXPECTED_VOCAB, HIDDEN_WIDTH):
        raise _fail(f"head shape {tuple(head.shape)} != ({EXPECTED_VOCAB}, {HIDDEN_WIDTH})")
    if head.dtype != torch.bfloat16:
        raise _fail(f"head dtype {head.dtype} != torch.bfloat16")
    head_record = {
        "path": str(head_path),
        "tensor_name": keys[0],
        "file_sha256": sha256_file(head_path),
        "payload_sha256": payload_sha256(head_path),
        "tensor_content_sha256": tensor_content_sha256(head),
        "shape": list(head.shape),
        "dtype": "bfloat16",
    }
    head32_t = head.to(device).float().t().contiguous()  # [hidden, vocab] fp32

    # ---- teacher ---------------------------------------------------------
    teacher_root = args.teacher.resolve()
    teacher_receipt_path = kld_report._find_teacher_receipt(teacher_root)
    from quant_pipeline.evaluation.glm53_logits import load_capture_receipt

    teacher = load_capture_receipt(teacher_receipt_path, expected_role="bf16_teacher")
    teacher_rows = kld_report._record_map(teacher)
    teacher_paths = kld_report._resolve_teacher_paths(teacher_rows, teacher_root, sha256_file)
    vocab = int(teacher["vocab_size"])
    if vocab != EXPECTED_VOCAB:
        raise _fail(f"teacher vocab {vocab} != {EXPECTED_VOCAB}")

    chunk = int(args.chunk_positions)
    runs_out: List[Dict[str, Any]] = []
    per_run_replay_vectors: List[np.ndarray] = []
    invariance_record: Optional[Dict[str, Any]] = None

    for run_index, run_dir in enumerate(args.runs, start=1):
        run_dir = run_dir.resolve()
        stream_receipt = json.loads((run_dir / "capture-receipt.json").read_text(encoding="utf-8"))
        hidden_receipt = json.loads((run_dir / "hidden-capture.json").read_text(encoding="utf-8"))
        student_rows = {row["window_id"]: row for row in stream_receipt["logit_files"]}
        hidden_rows = {row["window_id"]: row for row in hidden_receipt["hidden_files"]}
        if set(student_rows) != set(teacher_rows) or set(hidden_rows) != set(teacher_rows):
            raise _fail(f"{run_dir}: window sets differ between teacher/logits/hiddens")

        report_path = run_dir / "kld-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        tokenwise_path = run_dir / "tokenwise-kld.npy"
        report_tokenwise = np.load(tokenwise_path, allow_pickle=False) if tokenwise_path.is_file() else None

        replay_values: List[np.ndarray] = []
        live_vs_teacher: List[np.ndarray] = []
        replay_vs_teacher: List[np.ndarray] = []
        top1_matches = 0
        positions_total = 0
        per_window: List[Dict[str, Any]] = []
        logits_payload_shas: Dict[str, str] = {}
        hiddens_payload_shas: Dict[str, str] = {}

        for window_id in sorted(teacher_rows):
            teacher_row = teacher_rows[window_id]
            student_path = Path(student_rows[window_id]["path"])
            if not student_path.is_file():
                student_path = run_dir / "logits" / student_path.name
            hidden_path = Path(hidden_rows[window_id]["path"])
            if not hidden_path.is_file():
                hidden_path = run_dir / "hiddens" / hidden_path.name
            count = int(teacher_row["prediction_positions"])
            logits_payload_shas[window_id] = payload_sha256(student_path)
            hiddens_payload_shas[window_id] = payload_sha256(hidden_path)
            if hiddens_payload_shas[window_id] != hidden_rows[window_id]["payload_sha256"]:
                raise _fail(f"{run_dir} {window_id}: hidden payload sha drifted since capture")

            hidden = _load_hidden(hidden_path)
            if tuple(hidden.shape) != (count, HIDDEN_WIDTH):
                raise _fail(f"{run_dir} {window_id}: hidden shape {tuple(hidden.shape)}")
            replayed = _replay_logits(hidden, head32_t, device)  # fp32 [count, vocab]

            window_replay = np.empty(count, dtype=np.float64)
            window_live_teacher = np.empty(count, dtype=np.float64)
            window_replay_teacher = np.empty(count, dtype=np.float64)
            window_top1 = 0
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                live = kld_report._load_slice(student_path, start, stop)
                teacher_logits = kld_report._load_slice(teacher_paths[window_id], start, stop)
                replay_chunk = replayed[start:stop]
                values, matches = kld_report._token_kld(live, replay_chunk, device_str)
                window_replay[start:stop] = values
                window_top1 += matches
                values_a, _ = kld_report._token_kld(teacher_logits, live, device_str)
                window_live_teacher[start:stop] = values_a
                values_b, _ = kld_report._token_kld(teacher_logits, replay_chunk, device_str)
                window_replay_teacher[start:stop] = values_b
            del replayed

            replay_values.append(window_replay)
            live_vs_teacher.append(window_live_teacher)
            replay_vs_teacher.append(window_replay_teacher)
            top1_matches += window_top1
            positions_total += count
            per_window.append(
                {
                    "window_id": window_id,
                    "replay_kld_mean": float(window_replay.mean()),
                    "replay_kld_max": float(window_replay.max()),
                    "top1_matches": int(window_top1),
                    "positions": count,
                    "panel_mean_via_live": float(window_live_teacher.mean()),
                    "panel_mean_via_replayed": float(window_replay_teacher.mean()),
                }
            )
            print(json.dumps({"run": run_dir.name, "window": window_id,
                              "replay_kld_mean": float(window_replay.mean())}, sort_keys=True),
                  flush=True)

        replay_all = np.concatenate(replay_values)
        live_teacher_all = np.concatenate(live_vs_teacher)
        replay_teacher_all = np.concatenate(replay_vs_teacher)
        per_run_replay_vectors.append(replay_all)

        replay_npy = run_dir / "replay-kld.npy"
        buffer = io.BytesIO()
        np.save(buffer, replay_all, allow_pickle=False)
        replay_npy.write_bytes(buffer.getvalue())

        path_a_bitwise = (
            bool(np.array_equal(live_teacher_all, report_tokenwise))
            if report_tokenwise is not None
            else None
        )
        logits_digest = hashlib.sha256(
            "".join(logits_payload_shas[w] for w in sorted(logits_payload_shas)).encode("ascii")
        ).hexdigest()
        hiddens_digest = hashlib.sha256(
            "".join(hiddens_payload_shas[w] for w in sorted(hiddens_payload_shas)).encode("ascii")
        ).hexdigest()

        logits_bytes = sum(Path(student_rows[w]["path"]).stat().st_size
                           if Path(student_rows[w]["path"]).is_file()
                           else (run_dir / "logits" / Path(student_rows[w]["path"]).name).stat().st_size
                           for w in student_rows)
        hiddens_bytes = sum((run_dir / "hiddens" / Path(hidden_rows[w]["path"]).name).stat().st_size
                            for w in hidden_rows)

        runs_out.append(
            {
                "run_dir": str(run_dir),
                "cold_run": stream_receipt.get("cold_run"),
                "window_count": len(per_window),
                "positions": int(positions_total),
                "replay_kld": summarize_tokenwise(replay_all),
                "top1_agreement_live_vs_replayed": float(top1_matches / positions_total),
                "top1_matches": int(top1_matches),
                "panel_mean_kld_via_live_logits": float(live_teacher_all.mean()),
                "panel_mean_kld_via_replayed_logits": float(replay_teacher_all.mean()),
                "panel_delta_replayed_minus_live": float(replay_teacher_all.mean() - live_teacher_all.mean()),
                "panel_via_live_bitwise_matches_kld_report": path_a_bitwise,
                "kld_report_mean": (report or {}).get("summary", {}).get("mean") if report else None,
                "per_window": per_window,
                "logits_payload_sha256": logits_payload_shas,
                "hiddens_payload_sha256": hiddens_payload_shas,
                "logits_payload_digest_sha256": logits_digest,
                "hiddens_payload_digest_sha256": hiddens_digest,
                "logits_bytes_stored": int(logits_bytes),
                "hiddens_bytes_stored": int(hiddens_bytes),
                "replay_kld_npy": str(replay_npy),
                "replay_kld_npy_sha256": sha256_file(replay_npy),
                "stream_capture_receipt_sha256": stream_receipt.get("receipt_sha256"),
                "hidden_capture_receipt_sha256": hidden_receipt.get("receipt_sha256"),
            }
        )

        # ---- vocab-chunk invariance (designated run only) ------------------
        if run_index == int(args.invariance_run):
            alt_chunk = int(args.alt_vocab_chunk)
            alt_values: List[np.ndarray] = []
            bitwise_equal = 0
            elements_total = 0
            for window_id in sorted(teacher_rows):
                hidden_path = run_dir / "hiddens" / Path(hidden_rows[window_id]["path"]).name
                if not hidden_path.is_file():
                    hidden_path = Path(hidden_rows[window_id]["path"])
                student_path = Path(student_rows[window_id]["path"])
                if not student_path.is_file():
                    student_path = run_dir / "logits" / student_path.name
                count = int(teacher_rows[window_id]["prediction_positions"])
                hidden = _load_hidden(hidden_path)
                replay_default = _replay_logits(hidden, head32_t, device)
                replay_alt = _replay_logits(hidden, head32_t, device, vocab_chunk=alt_chunk)
                bitwise_equal += int((replay_default == replay_alt).sum().item())
                elements_total += replay_default.numel()
                window_alt = np.empty(count, dtype=np.float64)
                for start in range(0, count, chunk):
                    stop = min(start + chunk, count)
                    live = kld_report._load_slice(student_path, start, stop)
                    values, _ = kld_report._token_kld(live, replay_alt[start:stop], device_str)
                    window_alt[start:stop] = values
                alt_values.append(window_alt)
                del replay_default, replay_alt
            alt_all = np.concatenate(alt_values)
            invariance_record = {
                "run_dir": str(run_dir),
                "default_vocab_chunk": "monolithic (one fp32 GEMM per window)",
                "alt_vocab_chunk": alt_chunk,
                "mean_replay_kld_default": float(replay_all.mean()),
                "mean_replay_kld_alt_chunk": float(alt_all.mean()),
                "delta_of_means": float(abs(alt_all.mean() - replay_all.mean())),
                "max_token_abs_delta": float(np.abs(alt_all - replay_all).max()),
                "replayed_logits_bitwise_equal_fraction": float(bitwise_equal / elements_total),
            }

    # ---- cross-run determinism -------------------------------------------
    logits_digests = sorted({run["logits_payload_digest_sha256"] for run in runs_out})
    hiddens_digests = sorted({run["hiddens_payload_digest_sha256"] for run in runs_out})
    replay_vectors_equal = all(
        bool(np.array_equal(per_run_replay_vectors[0], vec)) for vec in per_run_replay_vectors[1:]
    )
    cross_run = {
        "runs": len(runs_out),
        "logits_bitwise_identical_across_runs": len(logits_digests) == 1,
        "hiddens_bitwise_identical_across_runs": len(hiddens_digests) == 1,
        "distinct_logits_payload_digests": logits_digests,
        "distinct_hiddens_payload_digests": hiddens_digests,
        "replay_kld_vectors_identical_across_runs": bool(replay_vectors_equal),
        "runtime_repeat_sentinel_mean_kld_between_runs": (
            0.0 if len(logits_digests) == 1 else None
        ),
        "note": (
            "KL between bitwise-identical logit sets is a sum of exact +0.0s; Festr's "
            "runtime-repeat sentinels on a vLLM serving stack measured ~3.2e-3 nats between "
            "identical runs -- the reference-forward streaming lane eliminates that term"
        ),
    }

    import torch as _torch

    result = {
        "schema": COMPARATOR_SCHEMA,
        "cut_point": CUT_POINT,
        "semantic_point": SEMANTIC_POINT,
        "cut_statement": CUT_STATEMENT,
        "kld_definition": "per-token KL(live_logits || replayed_logits) over the full vocabulary, fp64",
        "panel_kld_definition": "KL(teacher || student) over the full vocabulary, fp64 -- "
                                "the sealed estimator (kld_report._token_kld), reused not reimplemented",
        "replay_definition": "logits' = hidden @ head^T computed in fp32 (bf16 hidden and bf16 head "
                             "upcast to fp32; TF32 off; float32_matmul_precision highest)",
        "hidden_dtype": "bfloat16",
        "logits_dtype": "float32",
        "vocab_size": EXPECTED_VOCAB,
        "hidden_width": HIDDEN_WIDTH,
        "chunk_positions": chunk,
        "head": head_record,
        "teacher_receipt_sha256": teacher.get("receipt_sha256"),
        "runs": runs_out,
        "cross_run_determinism": cross_run,
        "vocab_chunk_invariance": invariance_record,
        "numeric_policy": numeric_policy,
        "device": device_str,
        "device_name": (_torch.cuda.get_device_name(device) if device.type == "cuda" else device.type),
        "torch_version": _torch.__version__,
        "cuda_runtime_version": getattr(_torch.version, "cuda", None),
        "code_identity": {
            "hidden_replay_sha256": sha256_file(Path(__file__).resolve()),
            "stream_score_sha256": sha256_file(TOOLS_DIR / "stream_score.py"),
            # The KEY keeps its 2026-08 spelling: it is a field of the
            # sealed malaiwah.glm53-hidden-replay-equivalence.v1 receipt, whose
            # receipt_sha256 covers it. Only the FILE moved (kld_report.py).
            "k6_kld_report_sha256": sha256_file(TOOLS_DIR / "kld_report.py"),
        },
    }
    out_path = args.out.resolve()
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    headline = {
        "out": str(out_path),
        "mean_replay_kld_run1": runs_out[0]["replay_kld"]["mean"],
        "max_token_replay_kld_run1": runs_out[0]["replay_kld"]["max"],
        "top1_run1": runs_out[0]["top1_agreement_live_vs_replayed"],
        "panel_delta_run1": runs_out[0]["panel_delta_replayed_minus_live"],
        "cross_run_bitwise": cross_run["logits_bitwise_identical_across_runs"]
        and cross_run["hiddens_bitwise_identical_across_runs"],
    }
    print(json.dumps(headline, sort_keys=True), flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="run one streaming cold capture with the lm_head hidden tap; "
                                         "pass the full stream_score argv after --")
    cap.add_argument("stream_argv", nargs=argparse.REMAINDER,
                     help="-- followed by the exact stream_score.py argv")

    cmp_ = sub.add_parser("compare", help="offline hidden-replay comparator over N captured runs")
    cmp_.add_argument("--runs", type=Path, nargs="+", required=True)
    cmp_.add_argument("--head", type=Path, required=True,
                      help="safetensors file carrying the bf16 lm_head weight [154880, 4096]")
    cmp_.add_argument("--teacher", type=Path, required=True)
    cmp_.add_argument("--device", default="auto")
    cmp_.add_argument("--chunk-positions", type=int, default=16,
                      help="fp64 KL chunk (16 = the sealed estimator's own)")
    cmp_.add_argument("--alt-vocab-chunk", type=int, default=8192)
    cmp_.add_argument("--invariance-run", type=int, default=1)
    cmp_.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "capture":
        argv = list(args.stream_argv)
        if argv and argv[0] == "--":
            argv = argv[1:]
        if not argv:
            raise _fail("capture needs the stream_score argv after --")
        return run_capture(args, argv)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
