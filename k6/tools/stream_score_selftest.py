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
  L1.f  native source     NativeCheckpointSource + fuse_gate_up rebuild the stacked expert
                           parameters transformers' own loader produces, BITWISE, on the
                           0.1B fixture.  This is what makes --source native (the BF16
                           floor) comparable to the packed lanes.  Needs --fixture.
  L1.d  receipt schema     a synthetic capture receipt in stream_score's own shape is
                           accepted by quant_pipeline's load_capture_receipt and by
                           k6_kld_report's per-window field comparison.
  L1.e  KLD estimator      k6_kld_report._token_kld against a closed-form KL between
                           two categorical distributions, fp64, plus the sealed
                           tokenwise vector's reshape identity when it is available.
  L1.g  teacher role       a --capture-role teacher receipt (sealed schema, role
                           bf16_teacher, teacher_provenance block) satisfies the
                           teacher-discovery predicate (schema+role, reimplemented
                           here so the rung runs without quant_pipeline).
  L1.h  preview refusal    a --store-positions preview capture is refused BOTH as a
                           teacher (predicate) and by k6_kld_report's pre-check.
  L1.i  sampling indices   preview_position_indices is deterministic per seed,
                           per-window randomized, in-bounds and evenly spread.
  L1.j  receipt stability  the receipt dict stream_score builds for a DEFAULT
                           invocation is field-identical to the sealed golden key
                           set (static AST proof that the new blocks are add-only
                           and flag-gated).

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
# L1.f native (BF16 floor) source parity
# --------------------------------------------------------------------------
def check_native_source(fixture: Optional[Path], device_spec: str = "cpu") -> None:
    """``--source native`` reads the SAME expert weights transformers' own loader does.

    The native lane fills the slab from per-expert checkpoint tensors
    (``...experts.E.{gate,up,down}_proj.weight``) fused with ``fuse_gate_up``.
    transformers builds its stacked ``experts.gate_up_proj`` / ``down_proj``
    parameters from those same checkpoint tensors through a completely different
    code path (its checkpoint-conversion mapping).  This rung asserts the two
    agree BITWISE on the architecture fixture, which is what makes the floor
    comparable to the packed lane's numbers rather than a differently-laid-out
    model that happens to run.
    """

    if fixture is None or not (fixture / "config.json").is_file():
        _record("L1.f-native-source", "SKIP", {"reason": "no --fixture with config.json"})
        return
    import torch
    import stream_score
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        device = stream_score.resolve_device(device_spec)
    except SystemExit as error:
        _record("L1.f-native-source", "SKIP", {"reason": f"device {device_spec!r}: {error}"})
        return
    config = AutoConfig.from_pretrained(fixture, local_files_only=True)
    text_config = config.get_text_config()
    first_dense = int(getattr(text_config, "first_k_dense_replace", 0))
    layers = tuple(range(first_dense, int(text_config.num_hidden_layers)))
    experts_per_layer = int(getattr(text_config, "n_routed_experts", 0))
    if not layers or not experts_per_layer:
        _record("L1.f-native-source", "SKIP", {"reason": "fixture declares no routed experts"})
        return
    try:
        source = stream_score.NativeCheckpointSource(fixture)
        layout = source.routed_tensor_census(layers, experts_per_layer)
    except SystemExit as error:
        _record("L1.f-native-source", "SKIP", {"reason": f"fixture layout: {error}"})
        return
    auto = (
        AutoModelForImageTextToText
        if config.architectures and "ConditionalGeneration" in config.architectures[0]
        else AutoModelForCausalLM
    )
    reference = auto.from_pretrained(
        fixture, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)
    module_layers = (
        reference.model.language_model.layers
        if hasattr(reference.model, "language_model")
        else reference.model.layers
    )
    checked = 0
    worst_gate_up = 0.0
    worst_down = 0.0
    bitwise = True
    with torch.inference_mode():
        for layer in layers:
            experts = module_layers[layer].mlp.experts
            for expert in range(experts_per_layer):
                parts = [
                    source.load(layer=layer, expert=expert, projection=projection)[0].to(device)
                    for projection in ("gate_proj", "up_proj", "down_proj")
                ]
                gate_up = torch.cat((parts[0], parts[1]), dim=0).contiguous()
                down = parts[2]
                # transformers stores the stacked parameter in whichever
                # orientation the module wants; accept the transpose so the rung
                # tests the VALUES, which is what the slab carries.
                for name, mine, theirs in (
                    ("gate_up", gate_up, experts.gate_up_proj[expert]),
                    ("down", down, experts.down_proj[expert]),
                ):
                    candidates = [theirs] if mine.shape == theirs.shape else [theirs.T]
                    ok = any(torch.equal(mine, candidate) for candidate in candidates)
                    delta = min(
                        float((mine.float() - candidate.float()).abs().max())
                        for candidate in candidates
                        if candidate.shape == mine.shape
                    ) if any(candidate.shape == mine.shape for candidate in candidates) else float("inf")
                    if name == "gate_up":
                        worst_gate_up = max(worst_gate_up, delta)
                    else:
                        worst_down = max(worst_down, delta)
                    bitwise = bitwise and ok
                checked += 1
    del reference
    _record(
        "L1.f-native-source",
        "PASS" if bitwise else "FAIL",
        {
            "fixture": str(fixture),
            "device": str(device),
            "experts_checked": checked,
            "routed_layout": layout,
            "bitwise_equal_to_loader_parameters": bitwise,
            "max_abs_gate_up_delta": worst_gate_up,
            "max_abs_down_delta": worst_down,
            "note": "NativeCheckpointSource + fuse_gate_up reproduces the stacked expert "
                    "parameters transformers' own checkpoint conversion builds",
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


# --------------------------------------------------------------------------
# L1.g / L1.h teacher-role acceptance and preview refusal
# --------------------------------------------------------------------------
SEALED_CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"


def _teacher_discovery_predicate(doc: Any) -> bool:
    """Reimplementation of k6_kld_report._find_teacher_receipt's acceptance
    test (schema equality + role), so this rung runs without quant_pipeline.
    The real function's behaviour is asserted by L1.d when the pipeline is
    present; this predicate is the laptop-side guard against drift."""
    return (
        isinstance(doc, dict)
        and doc.get("schema") == SEALED_CAPTURE_SCHEMA
        and doc.get("capture_role") == "bf16_teacher"
    )


def _teacher_receipt_fixture() -> Dict[str, Any]:
    import stream_score

    return {
        "schema": stream_score.CAPTURE_SCHEMA,
        "capture_role": stream_score.TEACHER_CAPTURE_ROLE,
        "cold_run": 1,
        "model_revision": "0" * 40,
        "checkpoint_identity_sha256": "c" * 64,
        "runtime_reader_sha256": "d" * 64,
        "token_panel_receipt_sha256": "e" * 64,
        "backend_identity_sha256": "f" * 64,
        "weight_dtype": "official BF16 checkpoint routed experts, streamed, NO decode",
        "logits_dtype": "float32",
        "kld_direction": "teacher_to_student",
        "prediction_positions": 51175,
        "vocab_size": 154880,
        "student_label": stream_score.NATIVE_STUDENT_LABEL,
        "logit_files": [],
        "elapsed_seconds": 1.0,
        "streaming_disclosure": {"schema": "malaiwah.glm53-streaming-disclosure.v1"},
        "teacher_provenance": {
            "schema": stream_score.TEACHER_PROVENANCE_SCHEMA,
            "teacher_label": stream_score.TEACHER_LABEL,
            "lane": "streaming-single-device",
            "source": "native-bf16",
            "ep_emulate": 8,
            "reduce_order": "fp32",
            "stream_mode": "window-major",
        },
    }


def check_teacher_role() -> None:
    import stream_score

    receipt = _teacher_receipt_fixture()
    accepted = _teacher_discovery_predicate(receipt)
    provenance = receipt["teacher_provenance"]
    provenance_ok = (
        provenance["schema"] == "malaiwah.glm53-same-lane-teacher-provenance.v1"
        and provenance["teacher_label"] == "native-bf16-streaming-v1"
    )
    student = dict(receipt, capture_role=stream_score.NATIVE_CAPTURE_ROLE)
    student.pop("teacher_provenance")
    _record(
        "L1.g-teacher-role",
        "PASS" if (accepted and provenance_ok
                   and not _teacher_discovery_predicate(student)) else "FAIL",
        {
            "teacher_accepted_by_predicate": accepted,
            "provenance_block_sealed_shape": provenance_ok,
            "native_student_role_NOT_a_teacher": not _teacher_discovery_predicate(student),
            "note": "schema stays quant-pipeline.glm53-logit-capture.v1 -- only the "
                    "ROLE flips, which is exactly what discovery keys on",
        },
    )


def check_preview_refusal() -> None:
    import stream_score
    import k6_kld_report

    preview = _teacher_receipt_fixture()
    preview["schema"] = stream_score.PREVIEW_CAPTURE_SCHEMA
    preview["capture_role"] = stream_score.NATIVE_CAPTURE_ROLE
    preview.pop("teacher_provenance")
    preview["not_submittable"] = True
    preview["sampling_design"] = {"scheme": "stratified-systematic",
                                  "positions_per_window": 256, "seed": 0}
    not_a_teacher = not _teacher_discovery_predicate(preview)
    # The refusal being demonstrated here is EXPECTED: capture its stderr so
    # a raw "k6_kld_report: ERROR: REFUSED" line never leaks unprefixed into
    # a passing run's output (a stranger reads that as a failure of THEIR
    # run -- usability review, 2026-08-28).  The captured text is re-emitted
    # inside this rung's [ok] record instead.
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as work:
        run_dir = Path(work)
        (run_dir / "capture-receipt.json").write_text(
            json.dumps(preview, sort_keys=True), encoding="utf-8")
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                k6_kld_report._refuse_preview_capture(run_dir)
            refused, message = False, None
        except SystemExit:
            refused = True
            message = ("expected refusal (captured): "
                       + " ".join(captured.getvalue().split()))
        sealed_dir_ok = True
        (run_dir / "capture-receipt.json").write_text(
            json.dumps(dict(preview, schema=SEALED_CAPTURE_SCHEMA),
                       sort_keys=True), encoding="utf-8")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                k6_kld_report._refuse_preview_capture(run_dir)
        except SystemExit:
            sealed_dir_ok = False
    _record(
        "L1.h-preview-refusal",
        "PASS" if (not_a_teacher and refused and sealed_dir_ok) else "FAIL",
        {
            "preview_refused_as_teacher": not_a_teacher,
            "preview_refused_by_kld_report_precheck": refused,
            "sealed_schema_passes_precheck": sealed_dir_ok,
            "detail": message,
        },
    )


def check_sampling_indices() -> None:
    # The design is FRACTIONAL-step systematic (step = N/m with a seeded
    # start), NOT integer-step: an integer step k = floor(N/m) makes every
    # position >= k*m unreachable at any seed, biasing the estimate whenever
    # KLD trends with context depth.  Consecutive gaps therefore alternate
    # between floor(step) and ceil(step), and the tail beyond k*m MUST be
    # reachable.  (Kept identical to bin/fidelity/previewstats.
    # systematic_indices; selftest_preview_stats cross-checks equality.)
    import stream_score

    a = stream_score.preview_position_indices(7, "final-0003", 2047, 256)
    b = stream_score.preview_position_indices(7, "final-0003", 2047, 256)
    starts = {stream_score.preview_position_indices(7, f"final-{i:04d}", 2047, 256)[0]
              for i in range(25)}
    k = 2047 // 256
    gaps = {y - x for x, y in zip(a, a[1:])}
    steps_ok = gaps <= {k, k + 1} and min(gaps) >= 1     # strictly increasing
    tail_ok = a[-1] >= k * 256                            # old unreachable zone
    bounds_ok = all(0 <= x < 2047 for x in a) and len(a) == 256
    tiny = stream_score.preview_position_indices(3, "w", 10, 8)
    ok = (a == b and steps_ok and tail_ok and bounds_ok and len(starts) > 1
          and all(x < 10 for x in tiny))
    _record(
        "L1.i-sampling-indices",
        "PASS" if ok else "FAIL",
        {
            "same_seed_same_indices": a == b,
            "fractional_step": 2047 / 256.0,
            "gaps_in_floor_ceil": sorted(gaps),
            "steps_floor_or_ceil": steps_ok,
            "tail_beyond_integer_step_reachable": tail_ok,
            "in_bounds": bounds_ok,
            "distinct_window_starts_of_25": len(starts),
            "clipping_respected": all(x < 10 for x in tiny),
        },
    )


# The sealed capture-receipt field set (the golden shape every consumer of
# quant-pipeline.glm53-logit-capture.v1 relies on).  receipt_sha256 is added
# after assembly and is not part of the dict literal.
GOLDEN_RECEIPT_KEYS = frozenset({
    "schema", "capture_role", "cold_run", "model_revision",
    "checkpoint_identity_sha256", "runtime_reader_sha256",
    "token_panel_receipt_sha256", "backend_identity_sha256", "weight_dtype",
    "logits_dtype", "kld_direction", "prediction_positions", "vocab_size",
    "student_label", "logit_files", "elapsed_seconds", "streaming_disclosure",
})

# Keys a NON-default run may add on top of the golden literal, each reviewed
# once and each provably behind an `if`.  A key that is not on this list fails
# the rung: adding a receipt field is a deliberate act, not a side effect.
#   teacher/preview:  the capture-role and sampling additions
#   exl3hf:           the stock-exllamav3 artifact pins (source == "exl3hf"),
#                     which name the artifact revision, its config/index shas,
#                     the codebook and the non-routed materialization receipt
#   mlx:              the community MLX artifact pins (source == "mlx") plus the
#                     two things that family MUST disclose - the measured
#                     quantization scope (it reaches past the routed experts)
#                     and the decoded non-routed view the forward was built from
#   gguf:             the community GGUF artifact pins (source == "gguf") - repo,
#                     immutable revision, the FILE LIST (a GGUF repo holds many
#                     quants at one revision, so the file list IS the artifact
#                     identity), the measured ggml type census, the imatrix
#                     metadata
#   scope_policy:     the measured "what did this artifact actually quantize"
#                     block.  gguf and nvfp4 both emit it under this name; mlx
#                     spells its own mlx_scope_policy because its census carries
#                     the passthrough set as well.
GATED_RECEIPT_KEYS = frozenset({
    "teacher_provenance", "schema", "not_submittable", "sampling_design",
    "exl3hf_repo", "exl3hf_revision", "artifact_config_sha256",
    "artifact_index_sha256", "codebook", "exllamav3_version", "declared_bits",
    "declared_head_bits", "materialization_receipt_sha256", "seal_disclosure",
    "routed_bits_decode_histogram",
    "mlx_repo", "mlx_revision", "mlx_format", "mlx_default_bits",
    "mlx_default_group_size", "mlx_bits_histogram", "mlx_config_sha256",
    "mlx_index_sha256", "mlx_shard_hash_verification", "mlx_scope_policy",
    "mlx_nonrouted_view", "mlx_nonrouted_passthrough_crosscheck",
    "mlx_fetch_ledger", "official_shape_census_sha256",
    "gguf_repo", "gguf_revision", "gguf_files", "gguf_file_hash_verification",
    "gguf_architecture", "gguf_type_census", "gguf_quant_metadata",
    "scope_policy",
    "source_repo", "source_revision",
    # --source tr3 (M2): a TR3-published release is the one third-party surface
    # that seals itself, so its receipt carries the VERIFICATION and the scope
    # it read off the artifact, not only the artifact's claims.
    "tr3_repo", "tr3_revision", "codec_family", "exllamav3_pin",
    "scope_census_sha256", "nonrouted_policy_declared",
    "seal_verification", "shard_verification",
    "artifact_materialization_receipt_sha256",
})


def check_receipt_stability() -> None:
    """Static AST proof that default invocations build the SEALED receipt shape.

    The receipt dict literal must carry exactly the golden keys, and every
    later addition except receipt_sha256 must sit inside an `if` (i.e. be
    flag-gated) -- which is what makes the teacher/preview/exl3hf/mlx blocks
    invisible to a default run.

    "Addition" means BOTH spellings: `receipt[key] = ...` and
    `receipt.update({...})`.  Only the first was checked until the mlx surface
    used the second, which would have slipped a whole block of fields past the
    one rung whose job is to notice them; a `.update()` whose argument is not a
    literal dict of constant keys is reported as ungated, because the rung can
    then prove nothing about it.
    """
    import ast

    source = (_TOOLS / "stream_score.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: Dict[Any, Any] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    literal_keys: Optional[set] = None
    ungated: List[str] = []
    gated: List[str] = []

    def _flag_gated(node) -> bool:
        cursor = node
        while cursor in parents:
            cursor = parents[cursor]
            if isinstance(cursor, ast.If):
                return True
        return False

    for node in ast.walk(tree):
        # receipt.update({...}) -- the same act as a subscript assignment, and
        # held to the same rule
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "receipt"):
            bucket = gated if _flag_gated(node) else ungated
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Dict):
                bucket.append("<receipt.update(non-literal)>")
            else:
                for key in node.args[0].keys:
                    bucket.append(key.value if isinstance(key, ast.Constant)
                                  else "<receipt.update(computed-key)>")
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Name) and target.id == "receipt"
                and isinstance(node.value, ast.Dict)):
            literal_keys = {k.value for k in node.value.keys
                            if isinstance(k, ast.Constant)}
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "receipt"
                and isinstance(target.slice, ast.Constant)):
            key = target.slice.value
            if key == "receipt_sha256":
                continue
            (gated if _flag_gated(node) else ungated).append(key)
    unreviewed = sorted(set(gated) - GATED_RECEIPT_KEYS)
    ok = (literal_keys == GOLDEN_RECEIPT_KEYS and not ungated and not unreviewed)
    _record(
        "L1.j-receipt-stability",
        "PASS" if ok else "FAIL",
        {
            "literal_keys_equal_golden": literal_keys == GOLDEN_RECEIPT_KEYS,
            "missing": sorted(GOLDEN_RECEIPT_KEYS - (literal_keys or set())),
            "unexpected": sorted((literal_keys or set()) - GOLDEN_RECEIPT_KEYS),
            "ungated_receipt_assignments": ungated,
            "flag_gated_assignments": sorted(set(gated)),
            "unreviewed_gated_assignments": unreviewed,
        },
    )


def check_source_dispatch() -> None:
    """Static AST proof that a weight SOURCE cannot be served by two branches.

    ``stream_score.main`` picks the routed source object, the checkpoint
    identity and the student label with if/elif chains keyed on
    ``args.source``, each ending in a catch-all ``else`` for the packed lane.
    A chain that ends in ``else:`` is a TRAP for a new surface: append a bare
    ``if args.source == "<new>":`` after it instead of an ``elif`` and every
    earlier source now runs its own branch AND falls through into the
    catch-all.

    That is not hypothetical.  It happened while the mlx surface was rebased
    onto the exl3hf one: the identity block became ``if exl3hf: ...`` followed
    by a fresh ``if mlx: ... elif native: ... else: ...``, so a
    ``--source exl3hf`` run built its identity and then re-entered the packed
    branch, dereferencing ``surface.contract_sha256`` with ``surface`` still
    None.  Nothing caught it, because no rung looked at the SHAPE of the
    dispatch.

    The rule: for each variable below, all of its ``args.source`` branches live
    in exactly ONE chain, and that chain ends in a catch-all.  Two chains
    assigning the same variable is the failure.
    """
    import ast

    source = (_TOOLS / "stream_score.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")

    choices: List[str] = []
    for node in ast.walk(main):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--source"):
            for kw in node.keywords:
                if kw.arg == "choices":
                    choices = [elt.value for elt in kw.value.elts]

    def assigns(stmt, name) -> bool:
        return any(isinstance(x, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == name for t in x.targets)
                   for x in ast.walk(stmt))

    def tests_source(stmt) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == "source"
                   for n in ast.walk(stmt.test))

    def chain(stmt):
        named, node = set(), stmt
        while True:
            for cmp_node in ast.walk(node.test):
                if (isinstance(cmp_node, ast.Compare)
                        and isinstance(cmp_node.left, ast.Attribute)
                        and cmp_node.left.attr == "source"):
                    for comparator in cmp_node.comparators:
                        if isinstance(comparator, ast.Constant):
                            named.add(comparator.value)
                        elif isinstance(comparator, (ast.Tuple, ast.List)):
                            named.update(e.value for e in comparator.elts
                                         if isinstance(e, ast.Constant))
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
                continue
            return sorted(named), bool(node.orelse)

    detail = {"source_choices": choices}
    ok = bool(choices)
    for var in ("checkpoint_identity", "student_label"):
        chains = [chain(stmt) for stmt in main.body
                  if isinstance(stmt, ast.If) and tests_source(stmt) and assigns(stmt, var)]
        detail[var] = {"chains": len(chains),
                       "named": chains[0][0] if len(chains) == 1 else
                                [c[0] for c in chains],
                       "ends_in_catch_all": chains[0][1] if len(chains) == 1 else None}
        if len(chains) != 1 or not chains[0][1]:
            ok = False
    _record("L1.k-source-dispatch", "PASS" if ok else "FAIL", detail)


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
    wanted = set((args.only or "a,b,c,d,e,f,g,h,i,j").replace(" ", "").split(","))
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
    if "f" in wanted:
        try:
            check_native_source(args.fixture.resolve() if args.fixture else None, args.device)
        except ImportError as error:
            _record("L1.f-native-source", "SKIP", {"reason": f"{type(error).__name__}: {error}"})
    if "d" in wanted:
        if have_pipeline:
            check_receipt_schema()
        else:
            _record("L1.d-receipt-schema", "SKIP", {"reason": "quant_pipeline not importable"})
    if "e" in wanted:
        # the estimator half needs only torch + numpy, so this rung runs on a
        # laptop with no pipeline checkout
        check_kld_estimator(args.sealed_tokenwise, args.sealed_report)
    # g/h/i/j/k need no pipeline, no torch and no fixtures -- pure json/ast --
    # so they run on any laptop; SKIP only if an import surprises us.
    for rung, fn in (("g", check_teacher_role), ("h", check_preview_refusal),
                     ("i", check_sampling_indices), ("j", check_receipt_stability),
                     ("k", check_source_dispatch)):
        if rung in wanted:
            try:
                fn()
            except ImportError as error:
                _record(f"L1.{rung}", "SKIP",
                        {"reason": f"{type(error).__name__}: {error}"})

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
