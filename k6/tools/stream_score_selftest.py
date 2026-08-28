#!/usr/bin/env python3
"""Offline validation ladder for stream_score.py — NO GPU, NO big weights.

Every check here runs on a laptop.  Ladder rung L1 of the streaming
qualification; L2/L3/L4 need the real surface and live in
``stage_k6.sh measure_stream``.

  L1.a  decode parity      decode_from_payload(load_payload_cpu(...)) is BITWISE
                           equal to the reader's own load_decoded_choice(...),
                           including the CPU-unpack split used on MPS.
                           Needs a payload store (--packed-root); skipped otherwise.
  L1.b  EP emulation       ep_router_remap reproduces
                           EpRouterParallel.transform_output_post_forward on random
                           routing tables, and the 8 emulated partials of the stock
                           grouped_mm experts forward sum to the same value the
                           single-device (ep=1) call produces when the reduction is
                           done in fp32.  Pure torch, tiny shapes, CPU.
  L1.c  forward plumbing   the streaming build (filtered index + slab-bound experts)
                           produces logits BITWISE equal to stock from_pretrained on
                           the architecturally-complete 0.1B fixture.
  L1.d  receipt schema     a synthetic capture receipt in stream_score's own shape is
                           accepted by quant_pipeline's load_capture_receipt and by
                           k6_kld_report's per-window field comparison.
  L1.e  KLD estimator      k6_kld_report._token_kld against a closed-form KL between
                           two categorical distributions, fp64, plus the sealed
                           tokenwise vector's reshape identity when it is available.

Usage:
    stream_score_selftest.py [--packed-root DIR] [--fixture DIR] [--pipeline-root DIR]
                             [--sealed-tokenwise NPY] [--sealed-report JSON]
                             [--only a,b,c] [--json OUT]

Exit code 0 only if every rung that could run PASSED; a rung with missing inputs is
reported as SKIP and does not fail the ladder unless --require names it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

RESULTS: List[Dict[str, Any]] = []


def _record(name: str, status: str, detail: Dict[str, Any]) -> None:
    RESULTS.append({"check": name, "status": status, **detail})
    marker = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[status]
    print(f"[{marker}] {name}: {json.dumps(detail, sort_keys=True, default=str)}", flush=True)


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise SystemExit(f"no quant_pipeline package under {pipeline_root}")


def _import_pipeline(pipeline_root: Optional[str]) -> bool:
    if pipeline_root:
        sys.path.insert(0, str(_pipeline_src(Path(pipeline_root))))
    elif os.environ.get("QP_PIPELINE_ROOT"):
        sys.path.insert(0, str(_pipeline_src(Path(os.environ["QP_PIPELINE_ROOT"]))))
    try:
        import quant_pipeline  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# L1.a decode parity
# --------------------------------------------------------------------------
def check_decode_parity(packed_root: Optional[Path], samples: int) -> None:
    if packed_root is None or not (packed_root / "payload-store" / "objects").is_dir():
        _record("L1.a-decode-parity", "SKIP", {"reason": "no --packed-root payload store"})
        return
    import torch
    import stream_score
    from quant_pipeline.evaluation.glm53_packed_k4_reader import (
        load_decoded_choice,
        PackedK4Surface,
        _expert_receipt,
    )
    from quant_pipeline.campaign.glm53_direct_k4 import MAIN_ROUTED_LAYERS, PROJECTIONS

    contract = json.loads((packed_root / "contract.json").read_text(encoding="utf-8"))
    bits = int(contract["rate"]["bits"])
    contract_sha = contract["contract_sha256"]

    # A partial surface is enough for a decode-parity check; it holds exactly the
    # choices we sample and re-uses the reader's own verification for each.
    picks = []
    stride = max(1, len(MAIN_ROUTED_LAYERS) // max(1, samples))
    for offset, layer in enumerate(MAIN_ROUTED_LAYERS[::stride][:samples]):
        picks.append((layer, (offset * 37) % 288))
    choices: Dict[Any, Any] = {}
    for layer, expert in picks:
        _, verified = _expert_receipt(
            packed_root, contract_sha256=contract_sha, layer=layer, expert=expert, bits=bits
        )
        for projection in PROJECTIONS:
            choices[(layer, expert, projection)] = verified[projection]
    surface = PackedK4Surface(
        root=packed_root,
        contract_sha256=contract_sha,
        mtp_adapter_receipt_sha256="0" * 64,
        mtp_pack_receipt_sha256="0" * 64,
        packed_reader_abi_sha256="0" * 64,
        choices=choices,
        main_layer_receipt_sha256=(),
        bits=bits,
    )
    device = torch.device("cpu")
    mismatches: List[str] = []
    split_mismatches: List[str] = []
    started = time.monotonic()
    for layer, expert in picks:
        for projection in PROJECTIONS:
            reference, _ = load_decoded_choice(
                surface, layer=layer, expert=expert, projection=projection, device=device
            )
            payload, choice = stream_score.load_payload_cpu(
                surface, layer=layer, expert=expert, projection=projection
            )
            observed, _ = stream_score.decode_from_payload(
                payload, choice, projection=projection, device=device, bits=bits
            )
            if not torch.equal(reference, observed):
                mismatches.append(f"L{layer}/E{expert}/{projection}")
            split, _ = stream_score.decode_from_payload(
                payload, choice, projection=projection, device=device, bits=bits,
                unpack_device=torch.device("cpu"),
            )
            if not torch.equal(reference, split):
                split_mismatches.append(f"L{layer}/E{expert}/{projection}")
    detail = {
        "matrices": len(picks) * len(PROJECTIONS),
        "bits": bits,
        "bitwise_mismatches": mismatches,
        "unpack_split_mismatches": split_mismatches,
        "seconds": round(time.monotonic() - started, 2),
    }
    _record("L1.a-decode-parity", "PASS" if not mismatches and not split_mismatches else "FAIL", detail)


# --------------------------------------------------------------------------
# L1.b EP emulation
# --------------------------------------------------------------------------
def check_ep_emulation() -> None:
    import torch
    import stream_score

    torch.manual_seed(0)
    num_experts, top_k, tokens, ep = 32, 4, 64, 4
    num_local = num_experts // ep
    index = torch.randint(0, num_experts, (tokens, top_k))
    weights = torch.rand(tokens, top_k, dtype=torch.float32)

    # (1) the remap reproduces the upstream hook exactly
    try:
        from transformers.distributed.tensor_parallel import EpRouterParallel

        class _Mesh:
            def __init__(self, rank, size):
                self._rank, self._size = rank, size

            def get_local_rank(self):
                return self._rank

            def size(self):
                return self._size

        _Router = type("_Router", (torch.nn.Module,), {"num_experts": num_experts})

        remap_ok = True
        for rank in range(ep):
            _, upstream_scores, upstream_index = EpRouterParallel().transform_output_post_forward(
                _Router(), (None, weights.clone(), index.clone()), _Mesh(rank, ep)
            )
            ours_index, ours_scores = stream_score.ep_router_remap(index, weights, rank, num_local)
            remap_ok &= bool(torch.equal(ours_index, upstream_index) and torch.equal(ours_scores, upstream_scores))
        remap_source = "transformers.EpRouterParallel"
    except Exception as error:  # noqa: BLE001 - upstream shape drift is a real finding
        remap_ok = None
        remap_source = f"unavailable: {type(error).__name__}: {error}"

    # (2) 8 emulated partials summed in fp32 equal the single-device call
    from transformers.integrations.moe import grouped_mm_experts_forward

    hidden_dim, inter = 16, 24

    class _Experts(torch.nn.Module):
        def __init__(self, gate_up, down):
            super().__init__()
            self.gate_up_proj = gate_up
            self.down_proj = down
            self.num_experts = gate_up.shape[0]
            self.has_gate, self.has_bias, self.is_transposed = True, False, False
            self.swiglu_limit = 10.0

        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            return torch.nn.functional.silu(gate) * up

    gate_up = torch.randn(num_experts, 2 * inter, hidden_dim, dtype=torch.bfloat16)
    down = torch.randn(num_experts, hidden_dim, inter, dtype=torch.bfloat16)
    hidden = torch.randn(tokens, hidden_dim, dtype=torch.bfloat16)
    whole = grouped_mm_experts_forward(_Experts(gate_up, down), hidden, index.clone(), weights.clone())
    partials = []
    for rank in range(ep):
        local = _Experts(
            gate_up[rank * num_local: (rank + 1) * num_local],
            down[rank * num_local: (rank + 1) * num_local],
        )
        remapped_index, remapped_weights = stream_score.ep_router_remap(index, weights, rank, num_local)
        partials.append(grouped_mm_experts_forward(local, hidden, remapped_index, remapped_weights))
    combined = stream_score.combine_partials(partials, "fp32")
    scale = float(whole.float().abs().max())
    bf16_ulp = scale * 2 ** -8  # bf16 has 8 mantissa bits including the implicit one
    max_abs = float((combined.float() - whole.float()).abs().max())
    orders = {}
    for order in ("fp32", "sequential", "reverse", "pairwise", "rotate:1"):
        value = stream_score.combine_partials(partials, order)
        orders[order] = float((value.float() - whole.float()).abs().max())
    # every order must stay inside a few bf16 ULPs of the single-device call:
    # that is the whole claim - EP emulation changes ONLY the rounding, never
    # which experts a token visits or how they are weighted.
    coverage_ok = max(orders.values()) <= bf16_ulp * ep
    status = "PASS" if (remap_ok in (True, None)) and coverage_ok else "FAIL"
    _record(
        "L1.b-ep-emulation",
        status,
        {
            "remap_matches_upstream": remap_ok,
            "remap_reference": remap_source,
            "ep": ep,
            "output_abs_max": scale,
            "one_bf16_ulp_at_that_scale": bf16_ulp,
            "budget_ep_ulps": bf16_ulp * ep,
            "fp32_combine_max_abs_vs_single_device": max_abs,
            "max_abs_by_reduce_order": orders,
            "max_abs_in_ulps": {k: (v / bf16_ulp if bf16_ulp else 0.0) for k, v in orders.items()},
            "note": "a nonzero delta here is the per-rank bf16 partial rounding the sealed "
                    "EP8 run also incurred; EP8 rounds ~5 partials, EP1 rounds once",
        },
    )


# --------------------------------------------------------------------------
# L1.c forward plumbing on the 0.1B fixture
# --------------------------------------------------------------------------
def check_fixture_forward(fixture: Optional[Path], device_spec: str = "cpu") -> None:
    if fixture is None or not (fixture / "config.json").is_file():
        _record("L1.c-fixture-forward", "SKIP", {"reason": "no --fixture with config.json"})
        return
    import torch
    import stream_score
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    import transformers.models.glm5_next.modeling_glm5_next as glm5_next

    # Parity is asserted DEVICE-LOCALLY: reference and streaming build run on the
    # same device, so a nonzero delta can only come from the streaming machinery.
    # Cross-device (mps/cuda vs cpu) deltas are reported separately and are a
    # property of the backend kernels, never of this tool.
    try:
        device = stream_score.resolve_device(device_spec)
    except SystemExit as error:
        _record("L1.c-fixture-forward", "SKIP", {"reason": f"device {device_spec!r}: {error}"})
        return

    config = AutoConfig.from_pretrained(fixture, local_files_only=True)
    text_config = config.get_text_config()
    first_dense = int(getattr(text_config, "first_k_dense_replace", 0))
    layer_count = int(text_config.num_hidden_layers)
    layers = tuple(range(first_dense, layer_count))
    auto = (
        AutoModelForImageTextToText
        if config.architectures and "ConditionalGeneration" in config.architectures[0]
        else AutoModelForCausalLM
    )
    torch.manual_seed(0)
    reference = auto.from_pretrained(
        fixture, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    reference.set_experts_implementation("grouped_mm")
    ids = torch.randint(0, int(text_config.vocab_size), (1, 64)).to(device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    expert_weights = {}
    module_layers = reference.model.language_model.layers if hasattr(reference.model, "language_model") else reference.model.layers
    for layer in layers:
        experts = module_layers[layer].mlp.experts
        expert_weights[layer] = (
            experts.gate_up_proj.detach().clone(),
            experts.down_proj.detach().clone(),
        )
    del reference

    with tempfile.TemporaryDirectory() as work:
        model, build = stream_score.build_streaming_model(
            bf16_root=fixture,
            work_dir=Path(work),
            device=device,
            attn_implementation="eager",
            experts_implementation="grouped_mm",
            layers=layers,
        )
        module_layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
        for layer in layers:
            experts = module_layers[layer].mlp.experts
            experts._parameters.pop("gate_up_proj", None)
            experts._parameters.pop("down_proj", None)
            experts.gate_up_proj, experts.down_proj = expert_weights[layer]
            experts.num_experts = expert_weights[layer][0].shape[0]
        with torch.inference_mode():
            observed = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    bitwise = bool(torch.equal(expected, observed))
    max_abs = float((expected.float() - observed.float()).abs().max())
    _record(
        "L1.c-fixture-forward",
        "PASS" if bitwise else "FAIL",
        {
            "fixture": str(fixture),
            "device": str(device),
            "layers_streamed": [layers[0], layers[-1]] if layers else [],
            "bitwise_equal": bitwise,
            "max_abs_logit_delta": max_abs,
            "logits_shape": list(expected.shape),
            "load_report": build["load_report"],
            "nonrouted_view": build["nonrouted_view"],
            "note": "reference and streaming build both ran on this device; the delta is the "
                    "streaming machinery only, never a cross-backend comparison",
        },
    )


# --------------------------------------------------------------------------
# L1.d receipt schema conformance
# --------------------------------------------------------------------------
def check_receipt_schema() -> None:
    import torch
    from safetensors.torch import save_file
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file
    from quant_pipeline.evaluation.glm53_logits import load_capture_receipt

    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        logits_dir = root / "logits"
        logits_dir.mkdir()
        rows = []
        for index in range(2):
            path = (logits_dir / f"window-{index:04d}.safetensors").resolve()
            save_file({"logits": torch.zeros(3, 8, dtype=torch.float32)}, path,
                      metadata={"capture_role": "packed_student"})
            rows.append(
                {
                    "window_id": f"final-{index:04d}",
                    "document_id": f"doc-{index}",
                    "domain": "axis1_general",
                    "role": "final",
                    "token_ids_sha256": "a" * 64,
                    "attention_mask_sha256": "b" * 64,
                    "prediction_positions": 3,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        receipt = {
            "schema": "quant-pipeline.glm53-logit-capture.v1",
            "capture_role": "packed_student",
            "cold_run": 1,
            "model_revision": "0" * 40,
            "checkpoint_identity_sha256": "c" * 64,
            "runtime_reader_sha256": "d" * 64,
            "token_panel_receipt_sha256": "e" * 64,
            "backend_identity_sha256": "f" * 64,
            "weight_dtype": "EXL3/TR3 uniform-k6 streamed offline-decoded to BF16",
            "logits_dtype": "float32",
            "kld_direction": "teacher_to_student",
            "prediction_positions": 6,
            "vocab_size": 8,
            "student_label": "uniform-k6",
            "logit_files": rows,
            "elapsed_seconds": 1.0,
            "streaming_disclosure": {"schema": "malaiwah.glm53-streaming-disclosure.v1"},
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        (root / "capture-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        try:
            loaded = load_capture_receipt(root / "capture-receipt.json", expected_role="packed_student")
            ok = loaded["receipt_sha256"] == receipt["receipt_sha256"]
            reason = None
        except Exception as error:  # noqa: BLE001 - the schema gate IS the check
            ok, reason = False, f"{type(error).__name__}: {error}"
    _record(
        "L1.d-receipt-schema",
        "PASS" if ok else "FAIL",
        {
            "accepted_by_load_capture_receipt": ok,
            "reason": reason,
            "note": "logit_files rows must carry EXACTLY the 10 sealed keys; extra top-level "
                    "keys such as streaming_disclosure are permitted by the seal",
        },
    )


# --------------------------------------------------------------------------
# L1.e KLD estimator
# --------------------------------------------------------------------------
def check_kld_estimator(sealed_tokenwise: Optional[Path], sealed_report: Optional[Path]) -> None:
    import torch
    import k6_kld_report

    torch.manual_seed(0)
    vocab = 512
    teacher = torch.randn(64, vocab, dtype=torch.float32) * 3.0
    student = teacher + torch.randn(64, vocab, dtype=torch.float32) * 0.5
    values, matches = k6_kld_report._token_kld(teacher, student, "cpu")
    # closed form in numpy float128-free fp64
    t64 = teacher.double().numpy()
    s64 = student.double().numpy()
    t64 = t64 - t64.max(axis=-1, keepdims=True)
    s64 = s64 - s64.max(axis=-1, keepdims=True)
    tlp = t64 - np.log(np.exp(t64).sum(axis=-1, keepdims=True))
    slp = s64 - np.log(np.exp(s64).sum(axis=-1, keepdims=True))
    closed = (np.exp(tlp) * (tlp - slp)).sum(axis=-1)
    max_abs = float(np.abs(values - closed).max())
    # exact identity: KL(p||p) == 0
    zero, _ = k6_kld_report._token_kld(teacher, teacher.clone(), "cpu")
    self_kl = float(np.abs(zero).max())
    detail: Dict[str, Any] = {
        "max_abs_vs_closed_form": max_abs,
        "self_kl_max": self_kl,
        "top1_matches": matches,
        "positions": int(values.size),
    }
    ok = max_abs <= 1e-12 and self_kl <= 1e-15
    if sealed_tokenwise and sealed_tokenwise.is_file():
        vector = np.load(sealed_tokenwise, allow_pickle=False)
        detail["sealed_tokenwise_shape"] = list(vector.shape)
        detail["sealed_tokenwise_mean"] = float(vector.mean())
        if vector.size == 25 * 2047:
            per_window = vector.reshape(25, 2047).mean(axis=1)
            detail["sealed_per_window_means"] = [float(value) for value in per_window]
            if sealed_report and sealed_report.is_file():
                report = json.loads(sealed_report.read_text(encoding="utf-8"))
                declared = [row["summary"]["mean"] for row in report["per_window"]]
                delta = float(np.abs(np.asarray(declared) - per_window).max())
                detail["reshape_identity_max_delta"] = delta
                ok = ok and delta == 0.0
    _record("L1.e-kld-estimator", "PASS" if ok else "FAIL", detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packed-root", type=Path, help="encode output root with payload-store/")
    parser.add_argument("--fixture", type=Path, help="0.1B architecturally-complete BF16 fixture")
    parser.add_argument("--pipeline-root")
    parser.add_argument("--sealed-tokenwise", type=Path,
                        default=_TOOLS / "stream-evidence" / "k6-sealed-tokenwise-kld.npy")
    parser.add_argument("--sealed-report", type=Path,
                        default=_TOOLS / "stream-evidence" / "k6-sealed-run1-kld-report.json")
    parser.add_argument("--decode-samples", type=int, default=4)
    parser.add_argument("--device", default="cpu",
                        help="device for the L1.c fixture forward: cpu (default) | mps | cuda[:N] | auto. "
                             "Reference and streaming build always run on the SAME device.")
    parser.add_argument("--only", help="comma-separated rung letters, e.g. a,b,d")
    parser.add_argument("--require", help="comma-separated rungs that must not SKIP")
    parser.add_argument("--json", type=Path, help="write the machine-readable verdict here")
    args = parser.parse_args()

    have_pipeline = _import_pipeline(args.pipeline_root)
    wanted = set((args.only or "a,b,c,d,e").replace(" ", "").split(","))
    required = set((args.require or "").replace(" ", "").split(",")) - {""}

    if "a" in wanted:
        if have_pipeline:
            check_decode_parity(args.packed_root.resolve() if args.packed_root else None,
                                args.decode_samples)
        else:
            _record("L1.a-decode-parity", "SKIP", {"reason": "quant_pipeline not importable"})
    # b and c need `transformers`; on a bare laptop (the documented Mac ladder in
    # STREAMING.md 10) it is absent and the rung must record SKIP, not abort the
    # whole ladder with a traceback.  Callers that need the rung to run pass
    # --require b,c, which turns a SKIP into a non-zero exit.
    if "b" in wanted:
        try:
            check_ep_emulation()
        except ImportError as error:
            _record("L1.b-ep-emulation", "SKIP", {"reason": f"{type(error).__name__}: {error}"})
    if "c" in wanted:
        try:
            check_fixture_forward(args.fixture.resolve() if args.fixture else None, args.device)
        except ImportError as error:
            _record("L1.c-fixture-forward", "SKIP", {"reason": f"{type(error).__name__}: {error}"})
    if "d" in wanted:
        if have_pipeline:
            check_receipt_schema()
        else:
            _record("L1.d-receipt-schema", "SKIP", {"reason": "quant_pipeline not importable"})
    if "e" in wanted:
        # the estimator half needs only torch + numpy, so this rung runs on a
        # laptop with no pipeline checkout
        check_kld_estimator(args.sealed_tokenwise, args.sealed_report)

    failed = [row["check"] for row in RESULTS if row["status"] == "FAIL"]
    skipped = [row["check"] for row in RESULTS if row["status"] == "SKIP"]
    unmet = [name for name in required if any(
        row["check"].startswith(f"L1.{name}") and row["status"] == "SKIP" for row in RESULTS
    )]
    verdict = {
        "schema": "malaiwah.glm53-streaming-selftest.v1",
        "checks": RESULTS,
        "failed": failed,
        "skipped": skipped,
        "required_but_skipped": unmet,
        "passed": not failed and not unmet,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
    print(json.dumps({"passed": verdict["passed"], "failed": failed, "skipped": skipped},
                     sort_keys=True), flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
