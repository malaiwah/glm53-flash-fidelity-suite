#!/usr/bin/env python3
"""Single-device STREAMING packed-K6/K8 student logit capture over the sealed panel.

Why this exists
---------------
The sealed measurement (``tools/k6_student_capture.py``) calls
``glm53_packed_k4_reader.load_complete_surface`` and then
``install_local_main_experts``, which decodes EVERY routed expert to BF16 up
front (~609 GB across 42 executed layers) and runs an ``torchrun`` EP8 forward
on 8x H200.  Those 8 GPUs are rented for RESIDENCY, not compute: the panel is
25 windows x 2048 tokens and the forward is ~1.7 PFLOP, well under a minute of
H200 math.

This tool keeps the *same* model, the *same* module classes, the *same* op
order and the *same* dtypes, and replaces only the residency strategy: the 288
routed experts of one layer (13.5 GiB BF16) are decoded immediately before that
layer's expert block runs and released immediately after.  Peak device memory
drops from 88.6 GiB/rank x 8 to ~26-38 GiB on ONE device.

It is deliberately NOT a new transformer forward.  In the default
``--stream-mode window-major`` the per-window call is the byte-identical
``model(input_ids=..., attention_mask=..., use_cache=False)`` the sealed capture
makes; the only injected code is a pre-forward hook on
``layers[L].mlp.experts`` that fills a reusable slab.

Parity anatomy (verified against transformers 5.16.1 source, not assumed)
-------------------------------------------------------------------------
``DistributedConfig(enable_expert_parallel=True)`` makes
``PreTrainedModel.tp_plan`` return ``self._ep_plan`` ONLY
(``transformers/distributed/mixin.py:87-97``).  The TP plan recorded in the
sealed ``backend.json`` under ``active_tp_plan`` is the *stored attribute*, not
an applied plan: ``apply_tensor_parallelism`` consumes the ``tp_plan``
PROPERTY.  Independent numerical confirmation from the sealed receipt:
``rank_installs[0].allocated_bytes = 95_112_933_376`` = 76.1 GB routed
(608.8 GB / 8) + ~19.0 GB non-routed, i.e. attention / KDA / DSA / mHC / dense /
shared-experts / embed / lm_head were FULLY REPLICATED on every rank.

Therefore every op outside the routed-MoE block is a single-device op already,
and the ONLY mechanical difference between the sealed EP8 run and this one is
the routed-expert combine in the 42 routed layers:

    EP8   : partial_r = bf16( fp32_sum over the top-k slots rank r owns )
            out       = NCCL bf16 all_reduce( partial_0 .. partial_7 )
    EP1   : out       = bf16( fp32_sum over all 8 top-k slots )

``--ep-emulate 8`` reproduces the EP8 partition exactly on one device (same
``EpRouterParallel`` masking, same 36-group ``torch._grouped_mm`` launch shape,
same per-rank bf16 rounding), leaving ONLY the NCCL reduction ORDER as a
residual; ``--reduce-order`` enumerates the candidate orders so the residual can
be measured rather than asserted.

The sealed run also did not pass ``experts_implementation``, so
``get_correct_experts_implementation(None)`` resolved to ``"grouped_mm"``
(``transformers/modeling_utils.py:1964-1986``); this tool forces and records it,
because the fallback ``Glm5NextTextExperts.forward`` accumulates with
``index_add_`` in bf16 and is CUDA-nondeterministic.

The BF16 floor (``--source native``)
------------------------------------
``--source {checkpoint,payload-store,dione}`` all score a QUANTIZED routed
surface.  ``--source native`` scores the un-quantized one: the routed experts
are read straight from the official BF16 checkpoint shards by their own tensor
names (``glm53_direct_k4.tensor_name``) with NO decode, and everything else --
panel, teacher, non-routed tensors, slab, EP8 emulation, reduce order,
grouped_mm kernel, fp32 logit storage, receipt schema -- is byte-for-byte the
same code path.  The resulting KLD is therefore the FLOOR of this lane: what it
costs to compare our stack's forward against the teacher's logits with zero
quantization error.  Subtracting it from a quant's panel mean leaves that
quant's quantization-ATTRIBUTABLE error.  The receipt is labelled
``native-bf16`` and its disclosure block records ``no_decode``.

Outputs
-------
``<out>/{plan,reader-identity,backend,capture-receipt}.json`` and
``<out>/logits/window-%04d.safetensors`` in the SAME schema family as
``k6_student_capture.py``, so ``k6_kld_report.py`` consumes this run unmodified
(use ``--profile k6-stream``; the ``student_label`` stays ``uniform-k6`` so the
per-run ``kld-report.json`` is directly comparable to the sealed one).
``backend.json`` and ``capture-receipt.json`` additionally carry a
``streaming_disclosure`` block naming every deviation from the sealed path.

Community MLX surfaces (``--source mlx``)
-----------------------------------------
``--source mlx`` scores an MLX affine-quantized community conversion
(orcarouter dialect: HF tensor names, per-expert ``weight/scales/biases``
triplets) through ``mlx_surface.py``.  Two structural differences from every
other source, both disclosed in the receipt:

  * SCOPE: the artifact quantizes beyond the routed experts (dense MLPs,
    shared experts, DSA attention projections).  The non-routed model is
    therefore built from a MATERIALIZED DECODED VIEW of the quant snapshot
    itself (passthrough tensors verbatim, quantized non-routed tensors
    decoded fp32 and rounded once to bf16), and ``--bf16`` is NOT an input of
    this source - passing it only enables an optional byte-identity
    cross-check of the passthrough tensors.  The receipt carries the measured
    ``scope_policy`` census.
  * PROVENANCE: community checkpoints are unsealed (no contract, no payload
    hashes); identity is the immutable repo revision + config/index sha256 +
    the official-BF16 shape-census binding, and the receipt carries
    ``seal_disclosure`` saying so (same policy as the Dione lane).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

# stdlib-only, ~60 lines, and in bin/BUNDLE.txt beside this file: a fill takes
# 27 s on the GGUF lane and a window up to 24 min, and both of those used to be
# unbroken silence in `logs/stage-measure.log`.  See progress.py's docstring for
# why this is not tqdm and why it renders differently into a file.
import progress as progress_meter  # noqa: E402

STREAM_PLAN_SCHEMA = "malaiwah.glm53-streaming-student-logit-capture-plan.v1"
STREAM_BACKEND_SCHEMA = "malaiwah.glm53-streaming-offline-reader-backend.v1"
# Lane-ONLY identity: a sha256 over just the fields that name the lane
# (torch/device/kernel/numeric policy/parallelism/reduce order) and NOTHING
# artifact-specific.  backend_identity_sha256 hashes checkpoint identity and
# per-quant dtypes alongside the lane, so it differs between two quants on
# the SAME lane and cannot answer "same lane?" -- this hash can (statistical
# review, 2026-08-28).  Emitted in backend.json; k6_kld_report copies it
# into future reports so fidelity-stats can gate on its equality.
STREAM_LANE_IDENTITY_SCHEMA = "malaiwah.glm53-streaming-lane-identity.v1"
CAPTURE_SCHEMA = "quant-pipeline.glm53-logit-capture.v1"
# A position-sampled capture is a PREVIEW: a different schema string on
# purpose, so the sealed teacher discovery (which matches CAPTURE_SCHEMA
# exactly) and the sealed scorer both refuse it structurally.
PREVIEW_CAPTURE_SCHEMA = "malaiwah.glm53-logit-capture-preview.v1"
DISCLOSURE_SCHEMA = "malaiwah.glm53-streaming-disclosure.v1"
NATIVE_IDENTITY_SCHEMA = "malaiwah.glm53-native-bf16-source-identity.v1"
NATIVE_STUDENT_IDENTITY_SCHEMA = "malaiwah.glm53-native-bf16-student-identity.v1"
NATIVE_STUDENT_LABEL = "native-bf16"
NATIVE_CAPTURE_ROLE = "native_bf16_student"
# stock-exllamav3 (exl3hf) profiles: profile -> (declared bpw, student label).
# The label must match k6_kld_report's map for the same profile.
EXL3HF_PROFILES = {
    "turbo-4.05bpw": (4.05, "turboderp-exl3-mul1-4.05bpw"),
    "turbo-3.05bpw": (3.05, "turboderp-exl3-mul1-3.05bpw"),
    # 2.05bpw is the same producer, storage layout and codebook as the two
    # above, at a lower rate AND a lower head: the release declares
    # head_bits 5, not the 6 that 4.05/3.05 declare.  Nothing here encodes
    # that -- declared_head_bits is READ off the artifact -- but the display
    # strings in k6_kld_report do, so they must say 5 for this profile.
    "turbo-2.05bpw": (2.05, "turboderp-exl3-mul1-2.05bpw"),
    # vcruz305's K2 pack: the same STOCK-exllamav3 HF storage layout (canonical
    # index, per-module {trellis,suh,svh,<codebook>}, official unfused tensor
    # names) as turboderp's releases, and therefore the same reader -- but the
    # MCG codebook and a routed-experts-ONLY scope with a native BF16 head, so
    # the label may not borrow turboderp's producer or the mul1 codec.  The
    # surface is a storage question; the scope is an artifact question.
    "vcruz-k2-2bpw": (2.0, "vcruz305-exl3-mcg-2bpw"),
}
# TR3-published (sealed EXL3/MCG, routed-experts-only) profiles.  Same shape as
# EXL3HF_PROFILES and the same rule: the label must match k6_kld_report's map.
TR3_PROFILES = {
    "tr3-4bpw": (4.0, "tr3-exl3-mcg-4bpw"),
}
# Dione (0xSero selective-EXL3, TP4-sliced, routed experts only) profiles.
# Same rule again: the label must match k6_kld_report's map for the profile,
# and the profile names the receipt family
# malaiwah.glm53-<profile>-packed-kld-summary.v1.
DIONE_PROFILES = {
    "dione-q4": (4.0, "dione-exl3-k4-tp4"),
    "dione-3.0bpw": (3.0, "dione-exl3-k3-tp4"),
}
TEACHER_CAPTURE_ROLE = "bf16_teacher"
# --source gguf: a community llama.cpp artifact that quantizes EVERYTHING, so
# the whole forward (not only the routed experts) is the artifact's weights.
GGUF_STUDENT_IDENTITY_SCHEMA = "malaiwah.glm53-gguf-student-identity.v1"
GGUF_STUDENT_LABEL = "gguf-llamacpp"
GGUF_CAPTURE_ROLE = "gguf_student"
TEACHER_PROVENANCE_SCHEMA = "malaiwah.glm53-same-lane-teacher-provenance.v1"
TEACHER_LABEL = "native-bf16-streaming-v1"
RELEASE_INVENTORY_SCHEMA = "quant-pipeline.glm-release-inventory.v1"

RELEASED_ARCHITECTURE = "Glm5NextForConditionalGeneration"
RELEASED_MODEL_TYPE = "glm5_next"
RELEASED_TEXT_MODEL_TYPE = "glm5_next_text"

# Routed expert parameter suffixes (the ONLY tensors this tool streams).
ROUTED_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"stream_score: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def preview_position_indices(seed: int, window_id: str, n_positions: int,
                             per_window: int) -> List[int]:
    """Systematic per-window position sample with a seeded random start.

    FRACTIONAL step = n_positions / per_window; start u ~ U[0, step) seeded
    by (seed, window_id) so every window gets its own start, the whole design
    is reproducible from one integer, and two artifacts sampled with the same
    seed share positions exactly (which is what makes paired preview deltas
    possible).  The step must be fractional, not floor(N/m): an integer step
    k makes positions >= k*m unreachable at ANY seed (12.5% of every window
    at m=256, 50% at m=1024), biasing the estimate whenever KLD trends with
    context depth.  Fractional step gives every position inclusion
    probability exactly m/N; indices are distinct and strictly increasing
    because step > 1 whenever m < N.  MUST stay identical to
    bin/fidelity/previewstats.systematic_indices (cross-checked by
    selftest_preview_stats.py).
    """
    if per_window >= n_positions:
        return list(range(n_positions))
    step = n_positions / float(per_window)
    u = random.Random(f"{seed}:{window_id}").random() * step
    return [min(n_positions - 1, int(u + i * step)) for i in range(per_window)]


# --------------------------------------------------------------------------
# pipeline import (identical resolution order to k6_student_capture.py)
# --------------------------------------------------------------------------
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


def _sealed_json(path: Path, schema: str, field: str) -> Dict[str, Any]:
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes

    value = _read_json(path, schema)
    body = dict(value)
    seal = body.pop(field, None)
    if (
        value.get("example_only") is True
        or value.get("schema") != schema
        or seal != sha256_bytes(canonical_json(body))
    ):
        raise _fail(f"invalid sealed {schema}: {path}")
    return value


# --------------------------------------------------------------------------
# device / dtype policy
# --------------------------------------------------------------------------
def resolve_device(spec: str):
    import torch

    if spec == "auto":
        if torch.cuda.is_available():
            spec = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            spec = "mps"
        else:
            spec = "cpu"
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise _fail("--device cuda requested but torch.cuda is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise _fail("--device mps requested but the MPS backend is unavailable")
    elif device.type != "cpu":
        raise _fail(f"unsupported device type {device.type!r} (cuda | mps | cpu)")
    return device


def probe_grouped_mm_kernel(device) -> Dict[str, Any]:
    """MEASURE which grouped-MM kernel this build/device actually dispatches.

    ``set_experts_implementation("grouped_mm")`` only picks the *forward*; the
    kernel underneath is chosen per call by
    ``transformers.integrations.moe._can_use_grouped_mm``, whose answer depends on
    the torch version and the device.  On torch 2.11/CUDA it is the native
    ``grouped_mm``; on torch <= 2.10 CPU it is
    ``torch.ops.transformers.grouped_mm_fallback``; on torch 2.13 MPS it is native
    again.  A receipt must never *assert* which one ran, so this probes it with a
    tiny call of the real dtypes and records the answer.
    """

    import torch

    record: Dict[str, Any] = {"probe": "transformers.integrations.moe._can_use_grouped_mm"}
    try:
        from transformers.integrations import moe as moe_module

        weight = torch.zeros(2, 8, 4, dtype=torch.bfloat16, device=device)
        hidden = torch.zeros(4, 4, dtype=torch.bfloat16, device=device)
        offs = torch.tensor([2, 4], dtype=torch.int32, device=device)
        can_use = bool(moe_module._can_use_grouped_mm(hidden, weight.transpose(-2, -1), offs))
        if can_use and hasattr(torch.nn.functional, "grouped_mm"):
            kernel = "torch.nn.functional.grouped_mm"
        elif can_use and hasattr(torch, "_grouped_mm"):
            kernel = "torch._grouped_mm"
        else:
            kernel = "torch.ops.transformers.grouped_mm_fallback"
        record.update({"can_use_native_grouped_mm": can_use, "dispatched_kernel": kernel})
    except Exception as error:  # noqa: BLE001 - an unprobeable kernel is a disclosure fact
        record.update({"can_use_native_grouped_mm": None, "dispatched_kernel": None,
                       "probe_error": f"{type(error).__name__}: {error}"})
    return record


def apply_numeric_policy(device) -> Dict[str, Any]:
    """Reproduce k6_student_capture.py:381-383 plus the environment it ran in."""

    import torch

    record: Dict[str, Any] = {
        "float32_matmul_precision": "highest",
        "nvidia_tf32_override_env": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record["allow_tf32_cuda_matmul"] = False
        record["allow_tf32_cudnn"] = False
    else:
        record["allow_tf32_cuda_matmul"] = None
        record["allow_tf32_cudnn"] = None
    return record


# --------------------------------------------------------------------------
# payload load / decode split (a literal split of reader.load_decoded_choice)
# --------------------------------------------------------------------------
def load_payload_cpu(surface, *, layer: int, expert: int, projection: str):
    """IO + the three sealed hash gates of ``load_decoded_choice``, on CPU.

    Split out of the reader so a thread pool can overlap the network/disk read
    and the SHA-256 gates with GPU decode.  ``stream_score_selftest.py`` proves
    ``decode_from_payload(load_payload_cpu(...))`` is BITWISE equal to
    ``load_decoded_choice(...)``.
    """

    from quant_pipeline.checkpoint.exact_payload import packed_payload_sha256
    from quant_pipeline.checkpoint.packed_payload import (
        MCG_MARKER_SIGNED_INT32,
        checkpoint_payload_sha256,
    )
    from quant_pipeline.campaign.glm53_direct_k4 import tensor_name

    choice = surface.choice(layer, expert, projection)
    store = surface.store
    payload_cpu = {
        name: store.objects.load_tensor(choice["objects"][name])
        for name in ("trellis", "suh", "svh", "mcg")
    }
    if (
        int(payload_cpu["mcg"].reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32
        or packed_payload_sha256({name: payload_cpu[name] for name in ("trellis", "suh", "svh")})
        != choice["packed_sha256"]
        or checkpoint_payload_sha256(payload_cpu) != choice["checkpoint_payload_sha256"]
    ):
        raise ValueError(f"packed payload hash differs: {tensor_name(layer, expert, projection)}")
    return payload_cpu, choice


def decode_from_payload(payload_cpu, choice, *, projection: str, device, bits: int, unpack_device=None):
    """Device move + ``decode_choice_hf`` (the sealed decode contract, verbatim).

    ``unpack_device`` splits ``unpack_trellis_states`` (pure int64 bit twiddling,
    bit-exact on any backend) onto a different device than the float half.  It
    exists for MPS, whose int64 coverage is partial; the CUDA path never uses it
    and calls ``decode_choice_hf`` unmodified.
    """

    import torch
    from quant_pipeline.campaign.glm53_direct_k4 import projection_shape
    from quant_pipeline.evaluation.glm53_packed_k4_reader import (
        decode_choice_hf,
        mcg_lut,
        unpack_trellis_states,
        _hadamard,
        _permutation,
    )

    payload = {name: value.to(device) for name, value in payload_cpu.items() if name != "mcg"}
    if unpack_device is None or torch.device(unpack_device) == torch.device(device):
        decoded = decode_choice_hf(payload["trellis"], payload["suh"], payload["svh"], bits=bits)
    else:
        # verbatim decode_choice_hf body with the integer unpack hoisted onto
        # `unpack_device`; the float half is bit-for-bit the same expression.
        states = unpack_trellis_states(payload_cpu["trellis"].to(unpack_device), bits=bits).to(device)
        indices = (states.to(torch.int64) & 0xFFFF).long()
        values = mcg_lut(states.device).index_select(0, indices.flatten()).reshape_as(states).float()
        values = values.index_select(-1, torch.argsort(_permutation(states.device)))
        k_tiles, n_tiles, _ = values.shape
        exl = (
            values.reshape(k_tiles, n_tiles, 16, 16)
            .permute(0, 2, 1, 3)
            .reshape(k_tiles * 16, n_tiles * 16)
        )
        had = _hadamard(exl.device, exl.dtype)
        exl = torch.matmul(had, exl.reshape(-1, 128, exl.shape[1])).reshape_as(exl)
        exl *= payload["suh"].to(device=exl.device, dtype=exl.dtype).reshape(-1, 1)
        exl = torch.matmul(exl.reshape(exl.shape[0], -1, 128), had).reshape_as(exl)
        exl *= payload["svh"].to(device=exl.device, dtype=exl.dtype).reshape(1, -1)
        decoded = exl.T.contiguous()
    if tuple(decoded.shape) != projection_shape(projection):
        raise ValueError("decoded projection orientation/shape differs from official HF tensor")
    return decoded, choice


# --------------------------------------------------------------------------
# native (un-quantized) routed surface: the BF16 checkpoint itself
# --------------------------------------------------------------------------
def native_tensor_name(layer: int, expert: int, projection: str) -> str:
    """The official per-expert checkpoint tensor name.

    Identical to ``quant_pipeline.campaign.glm53_direct_k4.tensor_name`` (which
    is what the ENCODER read to build the packed store), restated here without
    the GLM-5.3-Flash layer/expert bounds so the same reader also serves the
    0.1B architecture fixture in the offline ladder.
    """

    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


class NativeCheckpointSource:
    """Routed experts read straight from the official BF16 shards -- NO decode.

    This is the surface the packed codec approximates.  It deliberately mirrors
    ``load_payload_cpu``: one call returns the CPU tensor for one
    (layer, expert, projection) plus a census row, and the CALLER does the
    device move, ``fuse_gate_up``, the single bf16 rounding, the ``copy_`` into
    the slab and the ``torch.equal`` close check -- i.e. the packed lane's own
    installation algebra, unmodified.  The checkpoint tensors are already
    bfloat16, so that rounding is the identity and the slab holds the released
    bytes exactly.

    safetensors handles are cached PER THREAD: the streamer reads with a pool
    and the handles are not documented thread-safe (the Dione adapter serialises
    for the same reason; here a per-thread handle keeps the parallel IO that a
    599 GB network tree needs).
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: Dict[str, str] = dict(index["weight_map"])
            self.single_file: Optional[str] = None
        elif (self.root / "model.safetensors").is_file():
            self.weight_map = {}
            self.single_file = "model.safetensors"
        else:
            raise _fail(
                f"--source native needs {self.root}/model.safetensors.index.json "
                "or model.safetensors"
            )
        self._local = threading.local()
        self._lock = threading.Lock()
        self.shards_read: set = set()
        self.bytes_read = 0

    # -- layout -----------------------------------------------------------
    def shard_for(self, name: str) -> str:
        if self.single_file is not None:
            return self.single_file
        shard = self.weight_map.get(name)
        if shard is None:
            raise _fail(f"BF16 index has no routed tensor named {name}")
        return shard

    def routed_tensor_census(self, layers: Tuple[int, ...], num_experts: int) -> Dict[str, Any]:
        """Prove, from the index alone, that every routed tensor exists."""

        projections = ("gate_proj", "up_proj", "down_proj")
        expected = [
            native_tensor_name(layer, expert, projection)
            for layer in layers
            for expert in range(num_experts)
            for projection in projections
        ]
        if self.single_file is None:
            absent = [name for name in expected if name not in self.weight_map]
            shards = sorted({self.weight_map[name] for name in expected if name in self.weight_map})
        else:
            absent = []
            shards = [self.single_file]
        if absent:
            raise _fail(
                f"BF16 checkpoint is missing {len(absent)} routed expert tensors "
                f"(first: {absent[0]}) - wrong tree for --source native"
            )
        return {
            "routed_tensor_count": len(expected),
            "routed_shard_count": len(shards),
            "routed_shards": shards if len(shards) <= 200 else shards[:200],
            "layers": [layers[0], layers[-1]] if layers else [],
            "experts_per_layer": num_experts,
            "single_file_checkpoint": self.single_file is not None,
        }

    # -- reads ------------------------------------------------------------
    def _handle(self, shard: str):
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = self._local.handles = {}
        handle = cache.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.root / shard), framework="pt", device="cpu")
            enter = getattr(handle, "__enter__", None)
            if enter is not None:
                handle = enter()
            cache[shard] = handle
        return handle

    def load(self, *, layer: int, expert: int, projection: str):
        name = native_tensor_name(layer, expert, projection)
        shard = self.shard_for(name)
        tensor = self._handle(shard).get_tensor(name)
        nbytes = int(tensor.numel() * tensor.element_size())
        with self._lock:
            self.shards_read.add(shard)
            self.bytes_read += nbytes
        return tensor, {"tensor": name, "shard": shard, "bytes": nbytes,
                        "dtype": str(tensor.dtype).replace("torch.", "")}


def native_source_identity(module_path: Path, runner_path: Path) -> Dict[str, Any]:
    """``reader_identity``'s shape for a run that decodes NOTHING.

    Same two file hashes (the reader module still supplies ``fuse_gate_up`` and
    ``resolve_main_layers``, and this file is still the runner), so a native
    receipt and a packed receipt are comparable field by field; the mode string
    and the absent codebook are what say "no codec ran".
    """

    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file

    body = {
        "schema": NATIVE_IDENTITY_SCHEMA,
        "mode": "official_bf16_checkpoint_routed_experts_no_decode_for_logit_measurement",
        "serving_kernel": False,
        "final_tp2_kernel": False,
        "bits": 16,
        "codebook": None,
        "decode_executed": False,
        "module_sha256": sha256_file(Path(module_path)),
        "runner_sha256": sha256_file(Path(runner_path)),
    }
    body["runtime_reader_sha256"] = sha256_bytes(canonical_json(body))
    return body


# --------------------------------------------------------------------------
# EP emulation
# --------------------------------------------------------------------------
def ep_router_remap(top_k_index, top_k_weights, rank: int, num_local_experts: int):
    """``EpRouterParallel.transform_output_post_forward`` verbatim.

    transformers/distributed/tensor_parallel.py (5.16.1), the EP router hook the
    sealed run installed on ``layers.*.mlp.gate``.
    """

    import torch

    non_local_mask = (top_k_index // num_local_experts) != rank
    scores = top_k_weights.masked_fill(non_local_mask, 0.0)
    indices = top_k_index.masked_fill(non_local_mask, -1)
    if num_local_experts > 1:
        indices = torch.fmod(indices, num_local_experts)
    else:
        indices = indices.masked_fill(indices > 0, 0).masked_fill(indices < 0, -1)
    indices = indices.masked_fill(indices == -1, num_local_experts)
    return indices, scores


def combine_partials(partials: List[Any], order: str):
    """Sum per-EP-rank bf16 partials.

    The sealed run summed them with ``dist.all_reduce`` over NCCL in bf16, whose
    per-element order is topology dependent and NOT reproducible from a single
    process.  Every order here is a candidate for that reduction; ``fp32`` is the
    order-free upper bound (accumulate in fp32, round once).
    """

    import torch

    if len(partials) == 1:
        return partials[0]
    if order == "fp32":
        # MEASURED to be the closest model of NCCL's bf16 all_reduce on an
        # 8x H200 NVSwitch node (see STREAMING.md 7.2): accumulate the partials
        # in fp32 and round ONCE, rather than chaining bf16 additions.
        acc = partials[0].to(torch.float32)
        for item in partials[1:]:
            acc = acc + item.to(torch.float32)
        return acc.to(partials[0].dtype)
    if order == "sequential":
        acc = partials[0]
        for item in partials[1:]:
            acc = acc + item
        return acc
    if order == "reverse":
        acc = partials[-1]
        for item in reversed(partials[:-1]):
            acc = acc + item
        return acc
    if order == "pairwise":
        level = list(partials)
        while len(level) > 1:
            nxt = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return level[0]
    if order.startswith("rotate:"):
        start = int(order.split(":", 1)[1]) % len(partials)
        rotated = partials[start:] + partials[:start]
        acc = rotated[0]
        for item in rotated[1:]:
            acc = acc + item
        return acc
    raise _fail(f"unknown --reduce-order {order!r}")


REDUCE_ORDERS = ("fp32", "sequential", "reverse", "pairwise") + tuple(f"rotate:{i}" for i in range(8))


LAYER_SLAB_BYTES = 288 * (4096 * 4096 + 4096 * 2048) * 2  # 14,495,514,624


def host_memory_budget_bytes(reserve: float = 0.80) -> int:
    """Usable host RAM for the decode cache, cgroup-aware.

    ``free``/``MemAvailable`` report the HOST on a container; the 1x H200 spot
    image used for the streaming validation reports 3,019 GB of host RAM behind
    a 300 GiB cgroup limit, and a 42-layer RAM cache (609 GB) is OOM-killed at
    layer ~22 if you believe ``free``.  cgroup v2 and v1 are both consulted.
    """

    candidates: List[int] = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw and raw != "max":
            try:
                value = int(raw)
            except ValueError:
                continue
            if 0 < value < (1 << 62):
                candidates.append(value)
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except OSError:
        pass
    if not candidates:
        return 1 << 62
    return int(min(candidates) * reserve)


# --------------------------------------------------------------------------
# the expert streamer
# --------------------------------------------------------------------------
class ExpertStreamer:
    """Owns one reusable BF16 expert slab and fills it per (layer, ep-group).

    Fill semantics are ``install_local_main_experts`` verbatim: three
    ``load_decoded_choice`` calls per expert, ``fuse_gate_up`` = ``cat((gate,up),0)``,
    a single fp32->bf16 rounding, ``copy_`` into the slab, and a ``torch.equal``
    close check.
    """

    def __init__(
        self,
        *,
        surface,
        device,
        bits: Optional[int],
        layers: Tuple[int, ...],
        slab_experts: int,
        decode_threads: int,
        unpack_device=None,
        cache_mode: str = "none",
        cache_dir: Optional[Path] = None,
        progress: bool = True,
        progress_interval: float = progress_meter.DEFAULT_INTERVAL_SECONDS,
        dione_shards: Any = None,
        native_source: Optional[NativeCheckpointSource] = None,
        exl3hf_source: Optional[Tuple[Any, Any]] = None,
        mlx_source: Any = None,
        gguf_source: Any = None,
        nvfp4_source: Any = None,
        cache_identity: Optional[Dict[str, Any]] = None,
    ):
        import torch

        self.progress = progress
        self.progress_interval = float(progress_interval)
        #: The in-fill meter, rebound by ``_begin_fill`` and ticked by whichever
        #: of the three fill loops is running.  ``None`` outside a fill.
        self._fill_meter: Optional[progress_meter.Progress] = None
        # when set, the routed surface is a Dione-style selective-EXL3 tree and
        # each matrix is assembled by dione_surface.load_decoded_module (decode
        # per TP slice, then rank-ordered concat) instead of the packed store.
        self.dione_shards = dione_shards
        # when set, the routed surface is the official BF16 checkpoint itself:
        # the slab is filled by reading the released per-expert tensors, with no
        # codec in the path at all (the measurement floor of this lane).
        self.native_source = native_source
        # when set, the routed surface is a stock-exllamav3 HF-sharded release:
        # (exl3hf_surface.Exl3HfSurface, Exl3HfShardReader).  The fill loop is
        # the packed lane's own producer/consumer with the payload IO and the
        # decode call swapped for exl3hf_surface's.
        self.exl3hf_source = exl3hf_source
        # when set, the routed surface is an MLX affine snapshot: the pool
        # threads run mlx_surface.dequant_affine on CPU (IO + fp32 decode per
        # tensor), and the consumer does the same device move / fuse_gate_up /
        # single bf16 rounding / copy_ / torch.equal close as every other lane.
        self.mlx_source = mlx_source
        # when set, the routed surface is a community llama.cpp GGUF: the pool
        # threads slice one expert out of the fused blk.L.ffn_*_exps tensor and
        # block-dequantize it on CPU, and the consumer does the same device move
        # / fuse_gate_up / single bf16 rounding / copy_ / torch.equal close as
        # every other lane.
        self.gguf_source = gguf_source
        # when set, the routed surface is a community NVFP4 snapshot: the pool
        # threads return FULLY-DECODED exact-fp32 CPU tensors (e2m1 LUT gather
        # + per-block scale), so the fp32->bf16 rounding downstream is the
        # lane's single rounding rather than the identity it is for native
        # bf16 bytes.
        self.nvfp4_source = nvfp4_source
        routed_sources = [name for name, value in (
            ("dione", dione_shards), ("native", native_source),
            ("exl3hf", exl3hf_source), ("mlx", mlx_source),
            ("gguf", gguf_source), ("nvfp4", nvfp4_source),
        ) if value is not None]
        if len(routed_sources) > 1:
            raise _fail(
                "ExpertStreamer cannot serve two routed sources at once: %s"
                % ", ".join(routed_sources)
            )
        if 288 % slab_experts:
            raise _fail("--slab-experts must divide 288")
        self.surface = surface
        self.device = device
        self.bits = bits
        self.layers = layers
        self.slab_experts = slab_experts
        self.decode_threads = decode_threads
        self.unpack_device = unpack_device
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir
        self.ram_cache: Dict[int, Tuple[Any, Any]] = {}
        self.ram_cache_bytes = 0
        self.ram_cache_budget = host_memory_budget_bytes()
        self.cache_refusals = 0
        self.resident: Optional[Tuple[int, int]] = None  # (layer, group)
        self.gate_up = torch.empty(slab_experts, 4096, 4096, dtype=torch.bfloat16, device=device)
        self.down = torch.empty(slab_experts, 4096, 2048, dtype=torch.bfloat16, device=device)
        self.decode_seconds = 0.0
        self.decoded_matrices = 0
        self.cache_hits = 0
        self.cache_fills = 0
        self.payload_bytes = 0
        self.census: List[Dict[str, Any]] = []
        self._census_layers: set = set()
        self.cache_served_layers: set = set()
        if cache_mode == "disk":
            if cache_dir is None:
                raise _fail("--decode-cache disk requires --decode-cache-dir")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._bind_disk_cache(cache_identity)

    # -- cache layout -----------------------------------------------------
    #: What a `--decode-cache disk` directory must agree about before its bytes may be
    #: installed as this run's expert weights.  The cache is keyed ONLY on the layer
    #: index (`layer-%03d.{gate_up,down}.bf16.bin`) and `_cache_load_into_slab` reads
    #: raw int16 with nothing but a size check, so pointing a second run at the same
    #: --decode-cache-dir installed the FIRST artifact's decoded experts and published
    #: them under the second artifact's identity: a wrong number wearing correct
    #: provenance.  Every other reusable artifact in this file is identity-stamped
    #: (`prepare_nonrouted_view` writes .view-source.json; mlx binds config/index/
    #: census/adapter sha; gguf's non-routed view is fingerprinted) -- this was the
    #: one that was not.
    CACHE_STAMP_NAME = "decode-cache-identity.json"

    def _bind_disk_cache(self, identity: Optional[Dict[str, Any]]) -> None:
        import hashlib

        stamp_path = self.cache_dir / self.CACHE_STAMP_NAME
        want = dict(identity or {})
        want["decode_cache_layout"] = "layer-%03d.{gate_up,down}.bf16.bin"
        blob = json.dumps(want, sort_keys=True, separators=(",", ":"))
        want["identity_sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        existing_payload = sorted(self.cache_dir.glob("layer-*.bf16.bin"))
        if stamp_path.is_file():
            try:
                have = json.loads(stamp_path.read_text())
            except (OSError, ValueError) as exc:
                raise _fail(
                    f"decode cache stamp {stamp_path} is unreadable ({exc}). Delete the "
                    f"cache directory and re-decode; its bytes cannot be attributed."
                )
            if have.get("identity_sha256") != want["identity_sha256"]:
                differing = sorted(
                    k for k in set(have) | set(want)
                    if k != "identity_sha256" and have.get(k) != want.get(k))
                raise _fail(
                    f"--decode-cache-dir {self.cache_dir} was filled by a DIFFERENT run "
                    f"(differs on: {', '.join(differing) or 'identity_sha256'}). Its "
                    f"decoded experts would be installed as this run's weights and "
                    f"published under this run's identity. Use a fresh directory, or "
                    f"delete this one."
                )
            return
        if existing_payload:
            raise _fail(
                f"--decode-cache-dir {self.cache_dir} already holds "
                f"{len(existing_payload)} decoded layer file(s) but no "
                f"{self.CACHE_STAMP_NAME}, so there is nothing that says which artifact "
                f"they were decoded from. Delete the directory and re-decode."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = stamp_path.with_name(stamp_path.name + ".new")
        tmp.write_text(json.dumps(want, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, stamp_path)

    def _cache_path(self, layer: int, which: str) -> Path:
        return self.cache_dir / f"layer-{layer:03d}.{which}.bf16.bin"

    def _cache_ready(self, layer: int) -> bool:
        if self.cache_mode == "ram":
            return layer in self.ram_cache
        if self.cache_mode == "disk":
            return self._cache_path(layer, "gate_up").is_file() and self._cache_path(layer, "down").is_file()
        return False

    def _cache_store(self, layer: int, gate_up_cpu, down_cpu) -> None:
        import torch

        if self.cache_mode == "ram":
            if self.ram_cache_bytes + LAYER_SLAB_BYTES > self.ram_cache_budget:
                # Fail SOFT, not OOM: a container cgroup limit (measured 300 GiB on
                # the 1x H200 spot image, NOT the host's 3 TB) holds ~19 of the 42
                # layers.  Cache what fits and re-decode the rest.
                self.cache_refusals += 1
                return
            self.ram_cache[layer] = (gate_up_cpu, down_cpu)
            self.ram_cache_bytes += LAYER_SLAB_BYTES
        elif self.cache_mode == "disk":
            for which, tensor in (("gate_up", gate_up_cpu), ("down", down_cpu)):
                path = self._cache_path(layer, which)
                staging = path.with_name(path.name + ".new")
                tensor.contiguous().view(torch.int16).numpy().tofile(str(staging))
                os.replace(staging, path)

    def _cache_load_into_slab(self, layer: int, group: int) -> None:
        import torch

        lo, hi = group * self.slab_experts, (group + 1) * self.slab_experts
        if self.cache_mode == "ram":
            gate_up_cpu, down_cpu = self.ram_cache[layer]
            self.gate_up.copy_(gate_up_cpu[lo:hi], non_blocking=False)
            self.down.copy_(down_cpu[lo:hi], non_blocking=False)
            return
        for which, slab, cols in (("gate_up", self.gate_up, 4096), ("down", self.down, 2048)):
            path = self._cache_path(layer, which)
            per_expert = 4096 * cols
            raw = np.fromfile(
                str(path), dtype=np.int16, count=self.slab_experts * per_expert, offset=lo * per_expert * 2
            )
            if raw.size != self.slab_experts * per_expert:
                raise _fail(f"decode cache truncated: {path}")
            slab.copy_(
                torch.from_numpy(raw).view(torch.bfloat16).reshape(self.slab_experts, 4096, cols)
            )

    # -- in-fill progress --------------------------------------------------
    #
    # A fill is the unit that takes real time: ~4.5 s on the exl3/tr3 lanes,
    # ~27 s on the GGUF lane.  The per-fill JSON record below `_fill_range`
    # only appears once the fill is OVER, so a stage log went quiet for the
    # whole of it -- and for a 24-minute window that is indistinguishable from
    # a hang.  These two methods are the only thing the hot loops call: an
    # integer add, and a clock read at most once per `check_every` matrices.
    # No tensor is touched, nothing synchronizes, no reduction order moves.
    _FILL_CHECK_EVERY = 8

    def _begin_fill(self, layer: int, group: int, count: int) -> None:
        self._fill_meter = progress_meter.Progress(
            count * len(("gate_proj", "up_proj", "down_proj")),
            label="fill L%03d/g%d matrices" % (layer, group),
            interval=self.progress_interval,
            enabled=self.progress,
            check_every=self._FILL_CHECK_EVERY,
        )

    def _end_fill(self) -> None:
        meter, self._fill_meter = self._fill_meter, None
        if meter is not None:
            meter.close()

    def _tick(self, step: int = 3) -> None:
        meter = self._fill_meter
        if meter is not None:
            meter.update(step)

    # -- fill -------------------------------------------------------------
    def ensure(self, layer: int, group: int = 0) -> None:
        import torch

        key = (layer, group)
        if self.resident == key:
            return
        started = time.monotonic()
        if self._cache_ready(layer):
            self._cache_load_into_slab(layer, group)
            self.cache_hits += 1
            # A cache hit returns BEFORE `record_census` below, so a fully warm cache
            # left `self.census` empty -- and the receipt still published
            # `installed_choice_census_sha256`, which then reduces to the digest of the
            # empty list (37517e5f3dc66819...), the same 64 hex digits for every warm
            # run of every artifact.  A constant that looks like evidence is worse than
            # an absent field, so record which layers were served from cache and let the
            # receipt say the census is partial.
            self.cache_served_layers.add(layer)
            self.resident = key
            self.decode_seconds += time.monotonic() - started
            return
        want_cache = self.cache_mode != "none" and self.slab_experts == 288
        if (
            want_cache
            and self.cache_mode == "ram"
            and self.ram_cache_bytes + LAYER_SLAB_BYTES > self.ram_cache_budget
        ):
            # The budget is already spent (cgroup-aware).  Decide that BEFORE
            # allocating and filling the 14.5 GB host mirror: _cache_store would
            # only throw it away, and filling it costs a full device->host copy
            # of every expert, once per layer, once per window.
            self.cache_refusals += 1
            want_cache = False
        cpu_gate_up = cpu_down = None
        if want_cache:
            cpu_gate_up = torch.empty(288, 4096, 4096, dtype=torch.bfloat16, device="cpu")
            cpu_down = torch.empty(288, 4096, 2048, dtype=torch.bfloat16, device="cpu")
        lo = group * self.slab_experts
        record_census = key not in self._census_layers
        self._begin_fill(layer, group, self.slab_experts)
        try:
            self._fill_range(layer, lo, self.slab_experts, cpu_gate_up, cpu_down, record_census)
        finally:
            self._end_fill()
        if record_census:
            self._census_layers.add(key)
        if want_cache:
            self._cache_store(layer, cpu_gate_up, cpu_down)
            self.cache_fills += 1
            del cpu_gate_up, cpu_down
        self.resident = key
        elapsed = time.monotonic() - started
        self.decode_seconds += elapsed
        if self.progress:
            print(
                json.dumps(
                    {
                        "fill": f"L{layer:03d}/g{group}",
                        "seconds": round(elapsed, 2),
                        "matrices": self.decoded_matrices,
                        "cumulative_decode_seconds": round(self.decode_seconds, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def _fill_range_dione(self, layer: int, lo: int, count: int, cpu_gate_up, cpu_down,
                          record_census: bool) -> None:
        """Dione surface: one matrix = tp_size decoded slices concatenated.

        The packed lane's producer/consumer shape with the IO half replaced by
        DioneShardReader.payload (thread-local safetensors handles) and the
        decode half by dione_surface.decode_module_payload -- the same math
        load_decoded_module and install_local_main_experts_dione run, split so
        the reads can overlap the decodes.  The install algebra after decode
        (fuse_gate_up, one fp32->bf16 rounding, copy_ into the slab, the
        torch.equal close) is shared verbatim with every other surface.

        ``hash_payloads=record_census`` matters: a census is recorded once per
        layer, but sha256 over every trellis block is ~3.2 MB of hashing per
        matrix and a cold run decodes 907,200 matrices.  Hashing all of them
        would add hours of pure CPU for 42 layers' worth of recorded rows.
        """
        import torch
        import dione_surface as ds
        from quant_pipeline.evaluation.glm53_packed_k4_reader import fuse_gate_up

        reader = self.dione_shards
        jobs: "queue.Queue[Any]" = queue.Queue(maxsize=max(2, self.decode_threads * 2))
        error: List[BaseException] = []

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.decode_threads) as pool:
                    pending: List[Any] = []
                    for offset in range(count):
                        expert = lo + offset
                        pending.append(
                            (
                                offset,
                                expert,
                                [
                                    pool.submit(reader.payload, layer, expert, proj)
                                    for proj in ("gate_proj", "up_proj", "down_proj")
                                ],
                            )
                        )
                        while len(pending) > self.decode_threads:
                            jobs.put(pending.pop(0))
                    for item in pending:
                        jobs.put(item)
            except BaseException as exc:  # noqa: BLE001 - propagated to the consumer
                error.append(exc)
            finally:
                jobs.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        with torch.inference_mode():
            while True:
                item = jobs.get()
                if item is None:
                    break
                offset, expert, futures = item
                slices = [future.result() for future in futures]
                decoded = []
                census = []
                for payloads, projection in zip(slices, ("gate_proj", "up_proj", "down_proj")):
                    tensor, row = ds.decode_module_payload(
                        self.surface, payloads, layer=layer, expert=expert,
                        projection=projection, device=self.device,
                        hash_payloads=record_census,
                    )
                    decoded.append(tensor)
                    census.append(row)
                    self.payload_bytes += sum(
                        int(slice_payload[name].numel() * slice_payload[name].element_size())
                        for slice_payload in payloads
                        for name in ("trellis", "suh", "svh")
                    )
                gate_up_bf16 = fuse_gate_up(decoded[0], decoded[1]).to(dtype=torch.bfloat16)
                down_bf16 = decoded[2].to(dtype=torch.bfloat16)
                self.gate_up[offset].copy_(gate_up_bf16)
                self.down[offset].copy_(down_bf16)
                if not torch.equal(self.gate_up[offset], gate_up_bf16) or not torch.equal(
                    self.down[offset], down_bf16
                ):
                    raise RuntimeError("BF16 streamed expert installation did not close exactly")
                if cpu_gate_up is not None:
                    cpu_gate_up[expert].copy_(gate_up_bf16.to("cpu"))
                    cpu_down[expert].copy_(down_bf16.to("cpu"))
                self.decoded_matrices += 3
                self._tick(3)
                if record_census:
                    self.census.append(
                        {"layer": layer, "global_expert": expert, "local_expert": offset,
                         "slices": census}
                    )
                del decoded, gate_up_bf16, down_bf16, slices
        thread.join()
        if error:
            raise error[0]

    def _fill_range_exl3hf(self, layer: int, lo: int, count: int, cpu_gate_up, cpu_down,
                           record_census: bool) -> None:
        """Stock-exllamav3 surface: threaded payload IO + on-device decode.

        The packed lane's producer/consumer shape, with load_payload_cpu
        replaced by Exl3HfShardReader.payload (thread-local safetensors
        handles) and decode_from_payload by exl3hf_surface.decode_module (the
        campaign decode ABI with the artifact's own codebook LUT).  The
        install algebra after decode - fuse_gate_up, single fp32->bf16
        rounding, copy_ into the slab, torch.equal close - is shared verbatim.
        """
        import torch
        import exl3hf_surface as xs3
        from quant_pipeline.evaluation.glm53_packed_k4_reader import fuse_gate_up

        xsurface, xreader = self.exl3hf_source
        jobs: "queue.Queue[Any]" = queue.Queue(maxsize=max(2, self.decode_threads * 2))
        error: List[BaseException] = []

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.decode_threads) as pool:
                    pending: List[Any] = []
                    for offset in range(count):
                        expert = lo + offset
                        pending.append(
                            (
                                offset,
                                expert,
                                [
                                    pool.submit(
                                        xreader.payload,
                                        xs3.routed_module_name(layer, expert, proj),
                                    )
                                    for proj in ("gate_proj", "up_proj", "down_proj")
                                ],
                            )
                        )
                        while len(pending) > self.decode_threads:
                            jobs.put(pending.pop(0))
                    for item in pending:
                        jobs.put(item)
            except BaseException as exc:  # noqa: BLE001 - propagated to the consumer
                error.append(exc)
            finally:
                jobs.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        with torch.inference_mode():
            while True:
                item = jobs.get()
                if item is None:
                    break
                offset, expert, futures = item
                payloads = [future.result() for future in futures]
                decoded = []
                rows = []
                for payload, projection in zip(payloads, ("gate_proj", "up_proj", "down_proj")):
                    module = xs3.routed_module_name(layer, expert, projection)
                    tensor, row = xs3.decode_module(
                        xsurface,
                        payload,
                        module=module,
                        device=self.device,
                        expected_shape=xs3.PROJECTION_SHAPE[projection],
                        unpack_device=self.unpack_device,
                        hash_payload=record_census,
                    )
                    key = f"K{row['bits']}"
                    xsurface.routed_bits_histogram[key] = (
                        xsurface.routed_bits_histogram.get(key, 0) + 1
                    )
                    decoded.append(tensor)
                    rows.append(row)
                    self.payload_bytes += sum(
                        int(payload[name].numel() * payload[name].element_size())
                        for name in ("trellis", "suh", "svh")
                    )
                gate_up_bf16 = fuse_gate_up(decoded[0], decoded[1]).to(dtype=torch.bfloat16)
                down_bf16 = decoded[2].to(dtype=torch.bfloat16)
                self.gate_up[offset].copy_(gate_up_bf16)
                self.down[offset].copy_(down_bf16)
                if not torch.equal(self.gate_up[offset], gate_up_bf16) or not torch.equal(
                    self.down[offset], down_bf16
                ):
                    raise RuntimeError("BF16 streamed expert installation did not close exactly")
                if cpu_gate_up is not None:
                    cpu_gate_up[expert].copy_(gate_up_bf16.to("cpu"))
                    cpu_down[expert].copy_(down_bf16.to("cpu"))
                self.decoded_matrices += 3
                self._tick(3)
                if record_census:
                    self.census.append(
                        {"layer": layer, "global_expert": expert, "local_expert": offset,
                         "payloads": rows}
                    )
                del decoded, gate_up_bf16, down_bf16
        thread.join()
        if error:
            raise error[0]

    def _fill_range(self, layer: int, lo: int, count: int, cpu_gate_up, cpu_down, record_census: bool) -> None:
        if self.dione_shards is not None:
            return self._fill_range_dione(layer, lo, count, cpu_gate_up, cpu_down, record_census)
        if self.exl3hf_source is not None:
            return self._fill_range_exl3hf(layer, lo, count, cpu_gate_up, cpu_down, record_census)
        import torch
        from quant_pipeline.evaluation.glm53_packed_k4_reader import fuse_gate_up

        jobs: "queue.Queue[Any]" = queue.Queue(maxsize=max(2, self.decode_threads * 2))
        error: List[BaseException] = []
        # ONE loop serves every per-tensor surface.  The packed lane submits
        # load_payload_cpu (IO + the three sealed hash gates) and the consumer
        # runs decode_from_payload; the native lane submits a checkpoint read,
        # the mlx lane a checkpoint read + CPU fp32 dequant, the gguf lane a
        # fused-tensor slice + CPU block dequant, and the nvfp4 lane an e2m1
        # LUT gather + per-block scale in exact fp32.  For all four the
        # consumer only moves the returned CPU tensor to the device.  Every
        # line after that -- fuse_gate_up, the single bf16 rounding, copy_ into
        # the slab, the torch.equal close check, the host cache -- is shared.
        source = next(
            (item for item in (self.native_source, self.mlx_source,
                               self.gguf_source, self.nvfp4_source)
             if item is not None),
            None,
        )

        def submit(pool, expert: int, projection: str):
            if source is not None:
                return pool.submit(source.load, layer=layer, expert=expert, projection=projection)
            return pool.submit(load_payload_cpu, self.surface, layer=layer,
                               expert=expert, projection=projection)

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=self.decode_threads) as pool:
                    pending: List[Any] = []
                    for offset in range(count):
                        expert = lo + offset
                        pending.append(
                            (
                                offset,
                                expert,
                                [
                                    submit(pool, expert, proj)
                                    for proj in ("gate_proj", "up_proj", "down_proj")
                                ],
                            )
                        )
                        while len(pending) > self.decode_threads:
                            jobs.put(pending.pop(0))
                    for item in pending:
                        jobs.put(item)
            except BaseException as exc:  # noqa: BLE001 - propagated to the consumer
                error.append(exc)
            finally:
                jobs.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        with torch.inference_mode():
            while True:
                item = jobs.get()
                if item is None:
                    break
                offset, expert, futures = item
                payloads = [future.result() for future in futures]
                decoded = []
                for (payload_cpu, choice), projection in zip(
                    payloads, ("gate_proj", "up_proj", "down_proj")
                ):
                    if source is not None:
                        # payload_cpu is the RELEASED bf16 tensor (native) or a
                        # fully-decoded fp32 CPU tensor (mlx / gguf / nvfp4);
                        # `choice` is its census row.  The adapter did all the
                        # codec work on the pool thread; here is only the
                        # device move.  No sealed hash gate exists for any of
                        # them: the binding is the inventory's index_sha256
                        # (native) or the artifact's checkpoint identity
                        # (revision + config/index sha256, per-component
                        # sha256 where the format has them), disclosed in the
                        # receipt.
                        decoded.append(payload_cpu.to(self.device))
                        continue
                    tensor, _ = decode_from_payload(
                        payload_cpu,
                        choice,
                        projection=projection,
                        device=self.device,
                        bits=self.bits,
                        unpack_device=self.unpack_device,
                    )
                    decoded.append(tensor)
                gate_up_bf16 = fuse_gate_up(decoded[0], decoded[1]).to(dtype=torch.bfloat16)
                down_bf16 = decoded[2].to(dtype=torch.bfloat16)
                self.gate_up[offset].copy_(gate_up_bf16)
                self.down[offset].copy_(down_bf16)
                if not torch.equal(self.gate_up[offset], gate_up_bf16) or not torch.equal(
                    self.down[offset], down_bf16
                ):
                    raise RuntimeError("BF16 streamed expert installation did not close exactly")
                if cpu_gate_up is not None:
                    cpu_gate_up[expert].copy_(gate_up_bf16.to("cpu"))
                    cpu_down[expert].copy_(down_bf16.to("cpu"))
                self.decoded_matrices += 3
                self._tick(3)
                choices = [row[1] for row in payloads]
                if source is not None:
                    self.payload_bytes += sum(int(row["bytes"]) for row in choices)
                    if record_census:
                        census_row = {
                            "layer": layer,
                            "global_expert": expert,
                            "local_expert": offset,
                            "tensors": [row["tensor"] for row in choices],
                            "shards": [row["shard"] for row in choices],
                        }
                        if any("quant" in row for row in choices):
                            census_row["quant"] = [row.get("quant") for row in choices]
                        if any("format" in row for row in choices):
                            census_row["formats"] = [row.get("format") for row in choices]
                        if any("components" in row for row in choices):
                            # per-matrix component provenance (e.g. NVFP4's
                            # packed/scale/global sha256s) - receipt-grade,
                            # folded into installed_choice_census_sha256
                            census_row["components"] = [row.get("components")
                                                        for row in choices]
                        self.census.append(census_row)
                else:
                    self.payload_bytes += sum(int(row["logical_payload_bytes"]) for row in choices)
                    if record_census:
                        self.census.append(
                            {
                                "layer": layer,
                                "global_expert": expert,
                                "local_expert": offset,
                                "choice_sha256": [row["choice_sha256"] for row in choices],
                                "packed_sha256": [row["packed_sha256"] for row in choices],
                            }
                        )
                del decoded, gate_up_bf16, down_bf16
        thread.join()
        if error:
            raise error[0]

    def release(self) -> None:
        self.resident = None


# --------------------------------------------------------------------------
# experts forward wiring
# --------------------------------------------------------------------------
def install_streaming_experts(model, *, streamer: ExpertStreamer, layers: Tuple[int, ...],
                              ep_emulate: int, reduce_order: str, stats: Dict[str, Any],
                              runtime: Optional[Dict[str, Any]] = None):
    """Bind the shared slab into every routed layer and install the EP wrapper.

    The expert weights are held as PLAIN module attributes rather than
    ``nn.Parameter`` so the slab can be re-bound per layer and per EP rank
    without allocating under ``inference_mode``.  Values, shapes, dtypes and the
    consuming kernel are unchanged.

    ``runtime`` is a live dict ({"ep": int, "order": str}); mutating it between
    forwards switches EP semantics without rebuilding anything, which is what the
    reduce-order sweep uses to measure combine-order sensitivity on ONE decode.
    """

    from quant_pipeline.evaluation.glm53_packed_k4_reader import resolve_main_layers

    module_layers = resolve_main_layers(model)
    if runtime is None:
        runtime = {}
    runtime.setdefault("ep", ep_emulate)
    runtime.setdefault("order", reduce_order)
    if streamer.slab_experts not in (288, 288 // ep_emulate):
        raise _fail(
            f"--slab-experts {streamer.slab_experts} is incompatible with --ep-emulate "
            f"{ep_emulate}: use 288 (whole layer) or {288 // ep_emulate} (one EP group)"
        )
    groups_per_layer = 288 // streamer.slab_experts

    for layer_index in layers:
        experts = module_layers[layer_index].mlp.experts
        stock_forward = experts.forward  # bound; dispatches to grouped_mm_experts_forward
        for name in ("gate_up_proj", "down_proj"):
            experts._parameters.pop(name, None)
        experts.gate_up_proj = streamer.gate_up
        experts.down_proj = streamer.down

        def make(layer_index=layer_index, experts=experts, stock_forward=stock_forward):
            def forward(hidden_states, top_k_index, top_k_weights):
                started = time.monotonic()
                ep = int(runtime["ep"])
                order = str(runtime["order"])
                if ep == 1:
                    streamer.ensure(layer_index, 0)
                    experts.num_experts = 288
                    experts.gate_up_proj = streamer.gate_up
                    experts.down_proj = streamer.down
                    out = stock_forward(hidden_states, top_k_index, top_k_weights)
                    stats["expert_forward_seconds"] += time.monotonic() - started
                    return out
                num_local = 288 // ep
                if streamer.slab_experts not in (288, num_local):
                    raise _fail(f"slab of {streamer.slab_experts} cannot serve ep={ep}")
                experts.num_experts = num_local
                partials = []
                for rank in range(ep):
                    group = (rank * num_local) // streamer.slab_experts
                    streamer.ensure(layer_index, group)
                    offset = (rank * num_local) % streamer.slab_experts
                    experts.gate_up_proj = streamer.gate_up[offset: offset + num_local]
                    experts.down_proj = streamer.down[offset: offset + num_local]
                    index, weights = ep_router_remap(top_k_index, top_k_weights, rank, num_local)
                    partials.append(stock_forward(hidden_states, index, weights))
                out = combine_partials(partials, order)
                stats["expert_forward_seconds"] += time.monotonic() - started
                return out

            return forward

        experts.forward = make()
    return {
        "routed_layers": list(layers),
        "slab_experts": streamer.slab_experts,
        "groups_per_layer": groups_per_layer,
        "ep_emulate": ep_emulate,
        "reduce_order": reduce_order,
        "expert_weights_as_plain_attributes": True,
    }


# --------------------------------------------------------------------------
# model construction: from_pretrained over a NON-ROUTED VIEW of the BF16 tree
# --------------------------------------------------------------------------
# The sealed capture built the student with
#     AutoModelForImageTextToText.from_pretrained(bf16, dtype=bfloat16,
#         distributed_config=..., local_files_only=True, low_cpu_mem_usage=True,
#         attn_implementation="eager")
# and then overwrote ONLY mlp.experts.{gate_up_proj,down_proj} for layers 3..44.
# Reading all 599 GB to throw 580 GB of it away is the residency cost this tool
# exists to remove, so the streaming build calls the SAME constructor over a
# directory whose model.safetensors.index.json lists only the 1,618 non-routed
# checkpoint tensors and whose shards are symlinks to the real ones.  The
# checkpoint->module key conversion, buffer construction, dtype handling and
# post-init are therefore transformers' own, not re-implemented here.
#
# The routed expert Parameters are removed from Glm5NextTextExperts at
# construction (replaced by 0-element placeholders so `_init_weights` stays a
# no-op) and re-bound to the streaming slab afterwards.  That is the entire
# delta, and it is asserted: the load report must show ZERO missing keys and
# exactly the two routed expert patterns as unexpected.
EXPERT_CHECKPOINT_KEY = re.compile(r"\.mlp\.experts\.\d+\.")


def prepare_nonrouted_view(bf16_root: Path, work_dir: Path, *,
                           view_name: str = "bf16-nonrouted-view",
                           config_strip_keys: Tuple[str, ...] = ()) -> Tuple[Path, Dict[str, Any]]:
    """Build (or reuse) a symlink view of the model tree without routed experts.

    ``config_strip_keys`` removes top-level config.json keys from the VIEW's
    copy only (the source file is never touched): a quantized snapshot's
    ``quantization_config`` would otherwise make ``from_pretrained`` engage a
    quantized-loading integration instead of the sealed plain-BF16 build.
    The default arguments reproduce the original behaviour byte-for-byte.
    """

    index_path = bf16_root / "model.safetensors.index.json"
    if not index_path.is_file():
        # single-file checkpoint (the 0.1B parity fixture): rewrite it without the
        # routed experts instead of symlinking shards.  Same contract, small file.
        return _prepare_nonrouted_view_single_file(
            bf16_root, work_dir, view_name=view_name, config_strip_keys=config_strip_keys
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index["weight_map"]
    keep = {name: shard for name, shard in weight_map.items()
            if EXPERT_CHECKPOINT_KEY.search(name) is None}
    dropped = len(weight_map) - len(keep)
    if not keep or dropped <= 0:
        raise _fail("the BF16 index carries no routed expert tensors to filter - wrong checkpoint?")
    shards = sorted(set(keep.values()))
    view = work_dir / view_name
    # A view is REUSED across cold runs on purpose -- rebuilding 1,618 symlinks
    # twice is waste. But "reuse" must mean "the same source", and it did not:
    # only the index was rewritten, so a view built from one root and reused
    # against another kept the first root's symlinks beside the second's index.
    # Stamp the source and rebuild from scratch when it changes; a stale view is
    # the kind of thing that surfaces as an inexplicable load error three stages
    # later, which is exactly where it did surface.
    stamp_path = view / ".view-source.json"
    stamp = {"bf16_root": str(bf16_root.resolve()),
             "index_sha256": __import__("hashlib").sha256(
                 index_path.read_bytes()).hexdigest(),
             "config_strip_keys": sorted(config_strip_keys)}
    if view.is_dir():
        try:
            current = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if current != stamp:
            shutil.rmtree(view)
    view.mkdir(parents=True, exist_ok=True)
    for entry in bf16_root.iterdir():
        if entry.name == "model.safetensors.index.json" or entry.is_dir():
            continue
        if entry.suffix in (".json", ".jinja", ".txt", ".model"):
            target = view / entry.name
            if not target.exists():
                target.write_bytes(_view_file_bytes(entry, config_strip_keys))
    for shard in shards:
        link = view / shard
        if not link.exists():
            link.symlink_to(bf16_root / shard)
    (view / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": dict(index.get("metadata", {})), "weight_map": keep}),
        encoding="utf-8",
    )
    stamp_path.write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")
    return view, {
        "checkpoint_tensor_count": len(weight_map),
        "nonrouted_tensor_count": len(keep),
        "routed_tensor_count_filtered": dropped,
        "shards_referenced": len(shards),
        "shards_total": len(set(weight_map.values())),
        "config_keys_stripped_in_view": sorted(config_strip_keys),
        "view_path": str(view),
    }


def _view_file_bytes(entry: Path, config_strip_keys: Tuple[str, ...]) -> bytes:
    """The bytes the non-routed view stores for one metadata file."""

    if config_strip_keys and entry.name == "config.json":
        config = json.loads(entry.read_text(encoding="utf-8"))
        for key in config_strip_keys:
            config.pop(key, None)
        return (json.dumps(config, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return entry.read_bytes()


def _prepare_nonrouted_view_single_file(bf16_root: Path, work_dir: Path, *,
                                        view_name: str = "bf16-nonrouted-view",
                                        config_strip_keys: Tuple[str, ...] = ()) -> Tuple[Path, Dict[str, Any]]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = bf16_root / "model.safetensors"
    if not source.is_file():
        raise _fail(f"{bf16_root} has neither model.safetensors.index.json nor model.safetensors")
    view = work_dir / view_name
    view.mkdir(parents=True, exist_ok=True)
    for entry in bf16_root.iterdir():
        if entry.is_dir() or entry.name == "model.safetensors":
            continue
        if entry.suffix in (".json", ".jinja", ".txt", ".model"):
            target = view / entry.name
            if not target.exists():
                target.write_bytes(_view_file_bytes(entry, config_strip_keys))
    kept = {}
    total = 0
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            total += 1
            if EXPERT_CHECKPOINT_KEY.search(name) is None:
                kept[name] = handle.get_tensor(name)
    if len(kept) == total:
        raise _fail("single-file checkpoint carries no routed expert tensors to filter")
    save_file(kept, view / "model.safetensors")
    return view, {
        "checkpoint_tensor_count": total,
        "nonrouted_tensor_count": len(kept),
        "routed_tensor_count_filtered": total - len(kept),
        "shards_referenced": 1,
        "shards_total": 1,
        "single_file_rewrite": True,
        "config_keys_stripped_in_view": sorted(config_strip_keys),
        "view_path": str(view),
    }


def build_streaming_model(*, bf16_root: Path, work_dir: Path, device, attn_implementation: str,
                          experts_implementation: str, layers: Tuple[int, ...],
                          nonrouted_view: Optional[Tuple[Path, Dict[str, Any]]] = None,
                          view_name: str = "bf16-nonrouted-view",
                          config_strip_keys: Tuple[str, ...] = ()):
    """Construct the sealed model over a routed-expert-free view of the weights.

    There are two ways a non-official non-routed set reaches the forward, and
    the three community formats split across both:

    ``nonrouted_view`` lets a SURFACE supply a MATERIALIZED view instead of the
    official BF16 tree.  --source mlx and --source gguf both do: those formats
    quantize embeddings / attention / lm_head as well as the routed experts, so
    their non-routed tensors are the ARTIFACT'S OWN, decoded into the official
    names, shapes and dtypes, and scoring them from the official tree would
    measure the wrong weights.

    ``view_name`` / ``config_strip_keys`` instead point the ordinary symlink
    view at a different root: --source nvfp4 quantizes the routed experts ONLY
    and ships its non-routed tensors as plain BF16 in-repo under the official
    names, so the snapshot itself is the non-routed source -- with the
    snapshot's ``quantization_config`` stripped from the VIEW's config copy
    (never the source file) so ``from_pretrained`` builds the sealed plain-BF16
    model rather than engaging a quantized-loading integration.

    The constructor call and every load assertion below are unchanged in both
    cases -- the view is a directory of safetensors with an index either way.
    """

    import torch
    from transformers import AutoModelForImageTextToText
    import transformers.models.glm5_next.modeling_glm5_next as glm5_next

    if nonrouted_view is not None:
        # --source mlx / gguf: the view was MATERIALIZED (decoded) by the
        # surface (mlx_surface.prepare_nonrouted_view_decoded,
        # gguf_surface.materialize_nonrouted_view) and already contains only the
        # non-routed tensors, so the symlink/filter step does not apply.
        view, view_record = nonrouted_view
    else:
        view, view_record = prepare_nonrouted_view(
            bf16_root, work_dir, view_name=view_name, config_strip_keys=config_strip_keys
        )
    started = time.monotonic()
    experts_class = glm5_next.Glm5NextTextExperts
    original_init = experts_class.__init__

    def streaming_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        # drop the 9.7 GB + 4.8 GB per-layer placeholders; the streaming slab is
        # bound in install_streaming_experts.  0-element stand-ins keep
        # Glm5NextPreTrainedModel._init_weights a no-op instead of an error.
        self._parameters.pop("gate_up_proj", None)
        self._parameters.pop("down_proj", None)
        self.gate_up_proj = torch.empty(0, dtype=torch.bfloat16)
        self.down_proj = torch.empty(0, dtype=torch.bfloat16)

    experts_class.__init__ = streaming_init
    try:
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            view,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        model = model.eval()
    finally:
        experts_class.__init__ = original_init
    load_seconds = time.monotonic() - started

    # The load report is the proof that filtering the index changed NOTHING except
    # the routed experts: every other tensor came from the checkpoint, none was
    # left at its _init_weights value, and nothing was reshaped.
    missing = sorted(loading_info.get("missing_keys") or [])
    unexpected = sorted(loading_info.get("unexpected_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    errors = list(loading_info.get("error_msgs") or [])
    if missing or mismatched or errors:
        raise _fail(
            "non-routed load is not exact: "
            f"missing={missing[:4]} mismatched={mismatched[:4]} errors={errors[:2]}"
        )
    routed_expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{suffix}"
        for layer in layers
        for suffix in ("gate_up_proj", "down_proj")
    }
    stray = sorted(set(unexpected) - routed_expected)
    if stray:
        raise _fail(
            f"filtered load left {len(stray)} non-routed tensors unloaded: {stray[:6]}"
        )
    # `unexpected` is the loader's view of targets with no source.  Sharded
    # checkpoints report exactly the streamed routed experts here; the
    # single-file rewrite path reports none (the modules carry no parameters at
    # all by then).  Either is fine - `stray` above is the load-bearing check.
    unexpected_is_exactly_routed = set(unexpected) == routed_expected

    text_config = model.config.get_text_config()
    model.set_experts_implementation(experts_implementation)
    module_layers = model.model.language_model.layers
    resolved = module_layers[layers[0]].mlp.experts.config._experts_implementation
    if resolved != experts_implementation:
        raise _fail(
            f"experts implementation resolved to {resolved!r}, not the requested "
            f"{experts_implementation!r}; the sealed run resolved to grouped_mm"
        )
    if text_config._attn_implementation != attn_implementation:
        raise _fail(
            f"attention backend is {text_config._attn_implementation!r}, expected "
            f"{attn_implementation!r}"
        )

    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    still_meta = sorted(name for name, value in parameters + buffers if value.device.type == "meta")
    if still_meta:
        raise _fail(f"non-routed build left tensors on meta: {still_meta[:6]}")
    parameter_bytes = sum(value.numel() * value.element_size() for _, value in parameters)
    buffer_bytes = sum(value.numel() * value.element_size() for _, value in buffers)
    move_started = time.monotonic()
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    off_device = sorted(
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if value.device.type != device.type
    )
    if off_device:
        raise _fail(f"streaming build left tensors off {device}: {off_device[:6]}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "constructor": "AutoModelForImageTextToText.from_pretrained over a non-routed view",
        "nonrouted_view": view_record,
        "nonrouted_parameter_count": len(parameters),
        "nonrouted_buffer_count": len(buffers),
        "nonrouted_parameter_bytes": int(parameter_bytes),
        "nonrouted_buffer_bytes": int(buffer_bytes),
        "load_report": {
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            # Emitted only when the loader actually reported unexpected keys:
            # on the single-file rewrite path unexpected == [] and an
            # assertion-named field reading "false" inside an [ok] rung looks
            # like a failed check when it is merely vacuous (usability
            # review, 2026-08-28).  `stray` above stays the load-bearing gate.
            **({"unexpected_keys_are_exactly_the_streamed_routed_experts":
                unexpected_is_exactly_routed} if unexpected else {}),
            "mismatched_keys": len(mismatched),
            "error_msgs": len(errors),
        },
        "load_seconds": load_seconds,
        "device_move_seconds": time.monotonic() - move_started,
        "experts_implementation": resolved,
        "attn_implementation": attn_implementation,
        "routed_expert_parameters_replaced_by_slab": True,
    }


# --------------------------------------------------------------------------
# memory budget
# --------------------------------------------------------------------------
NONROUTED_BYTES = 18_976_485_628  # measured on the released BF16 tree (1,618 tensors)


def plan_memory(vram_budget_gb: Optional[float], ep_emulate: int, slab_experts: Optional[int],
                cache_mode: str) -> Dict[str, Any]:
    """Pick a slab size that fits the budget; fail closed when nothing fits."""

    # MEASURED, not estimated: a 2048-token window on CUDA peaked at
    # 47.08 GB with an 18.98 GB non-routed model and a 14.50 GB slab, so the
    # eager fp32 attention scores + fp32 logit staging + decode transients cost
    # 13.60 GB.  Budgeting less than this makes --vram-budget-gb lie.
    activations = 13_600_000_000
    per_expert = (4096 * 4096 + 4096 * 2048) * 2
    options = [288 // ep_emulate] if ep_emulate > 1 else [288]
    if ep_emulate > 1:
        options = [288, 288 // ep_emulate]
    chosen = None
    if slab_experts is not None:
        chosen = slab_experts
    elif vram_budget_gb is None:
        chosen = options[0]
    else:
        budget = vram_budget_gb * 1e9
        for candidate in options:
            if NONROUTED_BYTES + candidate * per_expert + activations <= budget:
                chosen = candidate
                break
        if chosen is None:
            raise _fail(
                f"--vram-budget-gb {vram_budget_gb} cannot hold the smallest schedule "
                f"({(NONROUTED_BYTES + options[-1] * per_expert + activations) / 1e9:.1f} GB "
                f"needed with --ep-emulate {ep_emulate}); raise the budget or the EP size"
            )
    estimate = NONROUTED_BYTES + chosen * per_expert + activations
    return {
        "slab_experts": chosen,
        "estimated_peak_device_bytes": int(estimate),
        "estimated_peak_device_gb": round(estimate / 1e9, 3),
        "nonrouted_bytes_model": NONROUTED_BYTES,
        "slab_bytes": int(chosen * per_expert),
        "activation_headroom_bytes": activations,
        "decode_cache_mode": cache_mode,
        "decode_cache_host_bytes": (42 * LAYER_SLAB_BYTES) if cache_mode in ("ram", "disk") else 0,
    }




# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_argument_group("weight source")
    src.add_argument("--source", choices=("checkpoint", "payload-store", "dione", "native",
                                          "exl3hf", "tr3", "mlx", "gguf", "nvfp4"),
                     default="payload-store",
                     help="checkpoint = materialized shards (its receipt names the packed root); "
                          "payload-store = the content-addressed store directly (default); "
                          "dione = a Dione-style selective-EXL3 tree (0xSero) via "
                          "dione_surface: routed experts only, stored as tp_size EXL3/MCG "
                          "slices per matrix; the artifact's own retained non-routed tensors "
                          "serve the rest, materialized VERBATIM into --bf16 by "
                          "dione_surface materialize; "
                          "native = the official BF16 checkpoint's own routed experts, NO decode "
                          "(the measurement floor of this lane); "
                          "exl3hf = a stock-exllamav3 HF-sharded release (mul1/mcg codebook, "
                          "full-scope quant) via exl3hf_surface -- --bf16 must point at its "
                          "materialized non-routed tree; "
                          "tr3 = a SEALED TR3-published EXL3/MCG release via tr3_surface "
                          "(routed experts only; the artifact's own official-native non-routed "
                          "tensors serve the rest, materialized VERBATIM into --bf16 by "
                          "exl3hf_surface materialize because they share shards with the routed "
                          "payloads); "
                          "mlx = a community MLX affine snapshot via mlx_surface (EVERY tensor "
                          "from the quant repo: routed experts streamed-decoded, non-routed "
                          "decoded into a materialized bf16 view); "
                          "gguf = a community llama.cpp GGUF via gguf_surface: ALL tensors "
                          "(embeddings, attention, lm_head included) are decoded from the "
                          "artifact itself and --bf16 supplies only config/tokenizer/vision; "
                          "nvfp4 = a community NVFP4 snapshot via nvfp4_surface (routed experts "
                          "decoded e2m1/gs16 in exact fp32; the snapshot's own BF16 non-routed "
                          "tensors serve the rest, --bf16 is refused)")
    src.add_argument("--checkpoint", type=Path, help="--source checkpoint: materialized checkpoint root")
    src.add_argument("--packed-root", type=Path,
                     help="--source payload-store: encode output root (out-k6) carrying "
                          "contract.json, inventory.json, mtp-adapter-receipt.json and payload-store/")
    src.add_argument("--dione-root", type=Path,
                     help="--source dione: local snapshot of the Dione release "
                          "(config.json + index + layers/ + retained/ + EXL3_MANIFEST.json)")
    src.add_argument("--dione-repo",
                     help="--source dione: the HF repo id the snapshot came from")
    src.add_argument("--dione-revision",
                     help="--source dione: the immutable 40-hex revision of that snapshot")
    src.add_argument("--dione-verify-shards", choices=("full", "skip"), default="full",
                     help="--source dione: how the shard BYTES are bound to the release "
                          "manifest. full (default) requires the dione-shards-verified.json "
                          "marker written by `dione_surface.py verify-shards`, which hashes "
                          "every shard against EXL3_MANIFEST.json; skip is disclosed in the "
                          "receipt as shard_hash_verification=skipped")
    src.add_argument("--exl3hf-root", type=Path,
                     help="--source exl3hf: local snapshot of the stock-exllamav3 checkpoint")
    src.add_argument("--exl3hf-repo",
                     help="--source exl3hf: the HF repo id the snapshot came from")
    src.add_argument("--exl3hf-revision",
                     help="--source exl3hf: the immutable 40-hex revision of that snapshot")
    src.add_argument("--tr3-root", type=Path,
                     help="--source tr3: local snapshot of the TR3-published release "
                          "(config.json + index + shards + exl3-mcg-storage-abi.json + "
                          "materialization-receipt.json)")
    src.add_argument("--tr3-repo",
                     help="--source tr3: the HF repo id the snapshot came from")
    src.add_argument("--tr3-revision",
                     help="--source tr3: the immutable 40-hex revision of that snapshot")
    src.add_argument("--tr3-verify-shards", choices=("crosscheck", "full", "skip"),
                     default="crosscheck",
                     help="--source tr3: how the shard BYTES are bound to the published seal. "
                          "crosscheck (default) proves the receipt's shard_sha256 map equals the "
                          "release's own SHA256SUMS, which the fetch stage verifies byte-wise; "
                          "full re-hashes every shard here (~176 GB); skip is disclosed")
    src.add_argument("--mlx-root", type=Path,
                     help="--source mlx: MLX snapshot root (config.json + index + shards; a "
                          "fetch-meta metadata root is enough for --dry-run)")
    src.add_argument("--mlx-repo", help="--source mlx: HF repo id, recorded in the identity")
    src.add_argument("--mlx-revision",
                     help="--source mlx: immutable 40-hex repo commit (REQUIRED unless --dry-run)")
    src.add_argument("--mlx-official-census", type=Path,
                     help="--source mlx: official BF16 shape census "
                          "(default k6/tools/mlx-evidence/bf16-shape-census.json.gz)")
    src.add_argument("--mlx-skip-shard-hashes", action="store_true",
                     help="--source mlx: skip the whole-shard sha256 gate (disclosed in the receipt)")
    src.add_argument("--gguf-file", action="append", dest="gguf_files", metavar="GGUF",
                     help="--source gguf: every .gguf of the artifact (repeat for split parts; "
                          "an https URL is accepted for --dry-run metadata validation only)")
    src.add_argument("--gguf-repo", help="--source gguf: the artifact's HF repo id (recorded)")
    src.add_argument("--gguf-revision",
                     help="--source gguf: the immutable 40-hex repo commit (required)")
    src.add_argument("--skip-gguf-hashes", action="store_true",
                     help="--source gguf: skip the whole-file sha256 marker requirement "
                          "(recorded as file_hash_verification=skipped; a disclosed read)")
    src.add_argument("--nvfp4-root", type=Path,
                     help="--source nvfp4: local snapshot of the NVFP4 repo "
                          "(config.json + model.safetensors.index.json + shards)")
    src.add_argument("--nvfp4-repo",
                     help="--source nvfp4: HF repo id (e.g. RedHatAI/GLM-5.3-Flash-NVFP4), "
                          "recorded in the checkpoint identity")
    src.add_argument("--nvfp4-revision",
                     help="--source nvfp4: immutable 40-hex repo commit (required)")
    src.add_argument("--nvfp4-skip-shard-hashes", action="store_true",
                     help="--source nvfp4: accept a snapshot without the "
                          "nvfp4-shards-verified.json marker (disclosed in the receipt)")
    src.add_argument("--inventory", type=Path,
                     help="--source native: sealed quant-pipeline.glm-release-inventory.v1 that "
                          "binds the BF16 config/index (defaults to <packed-root>/inventory.json). "
                          "It is the only provenance anchor a native run has, since there is no "
                          "contract and no payload store")
    src.add_argument("--bf16", type=Path,
                     help="official BF16 checkpoint (non-routed source). REQUIRED for "
                          "checkpoint/payload-store/dione/native/exl3hf/gguf and REFUSED for "
                          "nvfp4. --source mlx supplies every tensor from the quant snapshot "
                          "itself, so there the flag only enables an optional passthrough "
                          "byte-identity cross-check; --source gguf still needs it but its role "
                          "NARROWS to config/tokenizer plus the vision tower (the main GGUF "
                          "carries none), every measured weight being decoded from the artifact; "
                          "--source nvfp4 quantizes the routed experts ONLY and ships its own "
                          "BF16 non-routed set in-repo, so a second tree would be ambiguous; "
                          "--source tr3 also quantizes the routed experts only, but its "
                          "non-routed tensors share shards with the routed payloads, so it "
                          "REQUIRES the materialized tree here")

    parser.add_argument("--teacher", type=Path, required=True,
                        help="teacher final-window tree (panel receipt search root)")
    parser.add_argument("--token-panel", type=Path, help="explicit sealed token-panel receipt path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cold-run", type=int, required=True)
    parser.add_argument("--profile", default="k6",
                        choices=("k6", "k8", "k6k8", "native-bf16", "mlx", "gguf", "nvfp4",
                                 "turbo-4.05bpw", "turbo-3.05bpw", "turbo-2.05bpw",
                                 "vcruz-k2-2bpw",
                                 "tr3-4bpw", "dione-q4", "dione-3.0bpw"))
    parser.add_argument("--roles", default="final")
    parser.add_argument("--windows", help="comma-separated window ids to score (default: all)")
    parser.add_argument("--pipeline-root")
    parser.add_argument("--attention-backend", default="eager")

    stream = parser.add_argument_group("streaming")
    stream.add_argument("--device", default="auto", help="auto | cuda | cuda:N | mps | cpu")
    stream.add_argument("--decode-device", default="same",
                        help="same (default) | cpu — where decode_choice_hf runs")
    stream.add_argument("--unpack-device", default="same",
                        help="same (default) | cpu — where unpack_trellis_states runs (MPS int64)")
    stream.add_argument("--stream-mode", choices=("window-major",), default="window-major",
                        help="window-major replays the sealed per-window model() call verbatim; "
                             "it is the only mode, because any layer-major driver would have to "
                             "re-implement Glm5NextTextModel.forward and lose exactly the parity "
                             "this tool exists to keep (use --decode-cache to amortise instead)")
    stream.add_argument(
        "--progress-seconds", type=float, default=progress_meter.interval_from_env(),
        help="how often to print a progress line when stdout is a FILE (default 30; "
             "0 disables). On a TTY the meter updates in place instead and this is "
             "ignored. Env override: FIDELITY_PROGRESS_SECONDS.")
    stream.add_argument(
        "--gguf-decode-device", choices=("auto", "device", "cpu"), default="auto",
        help="--source gguf only: where the ggml block dequant runs. auto (default) "
             "= the run's --device when it is an accelerator, else cpu; `cpu` forces "
             "the reference path. The two are BITWISE identical (same kernels, same "
             "op order, elementwise fp32 only) and selftest_gguf_offline rung 1b "
             "proves it on real bytes on whatever accelerator the host has. cpu costs "
             "~39 ms/matrix with the GPU at 2-4%%; see docs/GGUF-MEASUREMENT.md.")
    stream.add_argument("--decode-cache", choices=("none", "ram", "disk"), default="none")
    stream.add_argument("--decode-cache-dir", type=Path)
    stream.add_argument("--work-dir", type=Path,
                        help="scratch for the non-routed BF16 symlink view "
                             "(default <out>/../.stream-work); never written to by the forward")
    stream.add_argument("--vram-budget-gb", type=float)
    stream.add_argument("--slab-experts", type=int)
    stream.add_argument("--decode-threads", type=int, default=min(16, (os.cpu_count() or 8)))
    stream.add_argument("--ep-emulate", type=int, default=8, choices=(1, 2, 4, 8),
                        help="reproduce the sealed EP partition on one device (default 8 = the "
                             "sealed world size); 1 = plain single-device combine")
    stream.add_argument("--reduce-order", default="fp32",
                        help="how the EP partials are summed: " + " | ".join(REDUCE_ORDERS) +
                             ". DEFAULT fp32, chosen by measurement: on an 8x H200 NVSwitch node "
                             "NCCL's bf16 all_reduce behaves like an fp32 accumulate rounded once, "
                             "not a bf16 chain. On window final-0000 fp32 lands within 1.5e-5 of "
                             "the sealed mean KLD with 99.80%% argmax agreement, while every bf16 "
                             "chain order (sequential/reverse) lands 40-200x further away.")
    stream.add_argument("--experts-implementation", default="grouped_mm")
    stream.add_argument("--sweep",
                        help="extra forward passes per window on the SAME decoded weights, "
                             "e.g. 'ep1:none,ep8:fp32,ep8:pairwise,ep8:rotate:3' - measures how "
                             "much of any delta vs the sealed run is EP combine ORDER "
                             "(needs --slab-experts 288; cheap with --decode-cache ram)")
    stream.add_argument("--capture-role", choices=("student", "teacher"),
                        default="student",
                        help="teacher: emit this run's capture as a SAME-LANE bf16 teacher "
                             "(capture_role bf16_teacher + a sealed teacher_provenance block). "
                             "Legal only with --source native, full census. The output tree is "
                             "then a valid --teacher for k6_kld_report.py, and the lane's floor "
                             "against it is exactly 0 once T1 hash evidence exists "
                             "(k6/SAME-LANE-TEACHER.md)")
    stream.add_argument("--store-positions", default="all",
                        help="all (default, sealed) | per-window:<m> -- store only m "
                             "systematically-sampled positions per window. PREVIEW ONLY: the "
                             "receipt schema switches to %s and no sealed consumer accepts it. "
                             "NOTE: this saves logit STORAGE/teacher bandwidth only -- the trunk "
                             "forward still runs every position of every window (causality); "
                             "the lm_head saving (m/2047) is real but modest"
                             % PREVIEW_CAPTURE_SCHEMA)
    stream.add_argument("--sample-seed", type=int,
                        help="REQUIRED with --store-positions per-window:<m>; one integer "
                             "reproduces the whole design, and two artifacts sampled with the "
                             "same seed share positions (paired preview deltas need that)")
    stream.add_argument("--dry-run", action="store_true",
                        help="validate every input, seal and layout, print the plan, and exit "
                             "without touching weights or a GPU")
    args = parser.parse_args()
    _import_pipeline(args.pipeline_root)

    from quant_pipeline.core.artifacts import (
        canonical_json,
        prepare_empty_destination,
        sha256_bytes,
        sha256_file,
        write_json,
    )
    from quant_pipeline.campaign.glm53_direct_k4 import contract_bits, contract_schema_for_bits
    from quant_pipeline.campaign.glm53_mtp_k4 import _mtp_adapter_schema
    from quant_pipeline.evaluation.glm53_logits import load_panel_windows
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader_module
    from quant_pipeline.evaluation.glm53_packed_k4_reader import (
        MAIN_ROUTED_LAYERS,
        load_complete_surface,
        reader_identity,
        stored_encoder_closure,
    )

    import k6_student_capture as sealed_capture  # helper reuse, not a re-implementation

    if args.reduce_order not in REDUCE_ORDERS:
        raise _fail(f"--reduce-order must be one of {REDUCE_ORDERS}")
    if (args.source == "dione") != (args.profile in DIONE_PROFILES):
        raise _fail(
            "--source dione and a dione profile (%s) must be used together: the profile names "
            "the receipt family (malaiwah.glm53-<profile>-packed-kld-summary.v1) and selects "
            "the student label the KLD report expects"
            % "/".join(sorted(DIONE_PROFILES))
        )

    if (args.source == "native") != (args.profile == "native-bf16"):
        raise _fail(
            "--source native and --profile native-bf16 must be used together: the profile "
            "selects the student label the KLD report expects, and a native run is not a K6/K8 one"
        )
    if (args.source == "exl3hf") != (args.profile in EXL3HF_PROFILES):
        raise _fail(
            "--source exl3hf and an exl3hf profile (%s) must be used together: the profile "
            "selects the student label the KLD report expects"
            % "/".join(sorted(EXL3HF_PROFILES))
        )
    if (args.source == "tr3") != (args.profile in TR3_PROFILES):
        raise _fail(
            "--source tr3 and a tr3 profile (%s) must be used together: the profile names the "
            "receipt family (malaiwah.glm53-<profile>-packed-kld-summary.v1) and selects the "
            "student label the KLD report expects"
            % "/".join(sorted(TR3_PROFILES))
        )
    if (args.source == "mlx") != (args.profile == "mlx"):
        raise _fail(
            "--source mlx and --profile mlx must be used together: the profile names the "
            "receipt family (malaiwah.glm53-mlx-packed-kld-summary.v1) and an MLX surface "
            "is not a K6/K8 one"
        )
    if (args.source == "gguf") != (args.profile == "gguf"):
        raise _fail(
            "--source gguf and --profile gguf must be used together: the profile names the "
            "receipt family (malaiwah.glm53-gguf-packed-kld-summary.v1) and selects the "
            "student label the KLD report expects; a GGUF run is not a K6/K8/native one"
        )
    if (args.source == "nvfp4") != (args.profile == "nvfp4"):
        raise _fail(
            "--source nvfp4 and --profile nvfp4 must be used together: the profile names the "
            "receipt family (malaiwah.glm53-nvfp4-packed-kld-summary.v1) and selects the "
            "student label the KLD report expects; an NVFP4 run is not a K6/K8 one"
        )
    if args.source not in ("mlx", "nvfp4") and args.bf16 is None:
        raise _fail(
            "--bf16 is required for --source checkpoint/payload-store/dione/native/exl3hf/gguf "
            "(with --source gguf only for config/tokenizer/vision; every measured weight is "
            "decoded from the artifact). --source mlx supplies every tensor, config included, "
            "from the quant snapshot itself, and --source nvfp4 refuses --bf16 outright. "
            "--source tr3 and --source dione need it for the same mechanical reason exl3hf "
            "does: the non-routed tensors must reach transformers in shards that hold "
            "nothing else"
        )

    # ---- teacher role and preview sampling (both flag-gated; defaults are
    # byte-identical to the sealed behaviour) ------------------------------
    if args.capture_role == "teacher":
        if args.source != "native":
            raise _fail(
                "REFUSED: --capture-role teacher requires --source native. The teacher is the "
                f"lane's own bf16 forward; a packed student (profile {args.profile}) cannot be "
                "a teacher."
            )
        if args.windows or args.store_positions != "all":
            got = f"--windows {args.windows!r}" if args.windows else \
                f"--store-positions {args.store_positions!r}"
            raise _fail(
                "REFUSED: a teacher must be a full-census capture of all 25 windows / all "
                f"positions (got {got}). A subset teacher would silently redefine the panel "
                "every student is scored against."
            )
    preview_positions: Optional[int] = None
    if args.store_positions != "all":
        match = re.fullmatch(r"per-window:(\d+)", args.store_positions)
        if match is None:
            raise _fail(
                f"--store-positions must be 'all' or 'per-window:<m>' (got {args.store_positions!r})"
            )
        preview_positions = int(match.group(1))
        if args.sample_seed is None:
            raise _fail(
                "--store-positions per-window:<m> requires --sample-seed: the seed is the whole "
                "reproducibility story of a sampled preview, and paired previews of two "
                "artifacts only work when both used the same seed"
            )
        if preview_positions < 8:
            raise _fail(
                f"per-window:{preview_positions} is below the minimum of 8: with fewer than 8 "
                "positions per window the within-window variance estimate s_j is meaningless "
                "and no honest CI can be quoted"
            )
        if preview_positions >= 2047:
            raise _fail(
                f"sampling {preview_positions} of 2047 positions is not a preview, run "
                "--store-positions all"
            )

    # ---- resolve and verify the packed surface (fail closed) -------------
    materialization: Optional[Dict[str, Any]] = None
    packed_root: Optional[Path] = None
    surface = None
    contract = None
    mtp_adapter = None
    bits: Optional[int] = None
    mlx_surface_obj = None
    gguf_surface_obj = None
    nvfp4_module = None
    nvfp4 = None
    tr3_module = None
    tr3 = None
    dione_module = None
    dione = None
    dione_shards = None
    inventory: Optional[Dict[str, Any]] = None
    if args.source == "mlx":
        import mlx_surface as mlxs

        if args.mlx_root is None:
            raise _fail("--source mlx requires --mlx-root (the MLX snapshot root)")
        if not args.dry_run and (
            args.mlx_revision is None
            or mlxs._REVISION.fullmatch(args.mlx_revision) is None
        ):
            raise _fail(
                "--source mlx requires --mlx-revision <40-hex immutable commit> for a real "
                "capture: an unsealed community artifact's ONLY provenance anchor is the "
                "pinned revision plus the config/index sha256 (--dry-run may omit it)"
            )
        try:
            mlx_surface_obj = mlxs.load_mlx_surface(
                args.mlx_root,
                repo=args.mlx_repo,
                revision=args.mlx_revision,
                official_census_path=args.mlx_official_census,
                require_shard_hashes=not args.mlx_skip_shard_hashes,
            )
        except ValueError as error:
            raise _fail(str(error))
        model_revision = mlx_surface_obj.revision
    elif args.source == "dione":
        # A Dione release publishes NO seal: no upstream receipts, no
        # reconstruction closures, no reader ABI.  What it does publish is a
        # manifest with a sha256 for every shard, an index whose 583,090 names
        # close exactly against the declared scope, and a config that states the
        # recipe in full.  Those are the provenance anchors, and the receipt
        # says so rather than implying a seal that does not exist.
        if args.dione_root is None or not args.dione_repo or not args.dione_revision:
            raise _fail("--source dione requires --dione-root, --dione-repo and --dione-revision")
        if sealed_capture.REVISION.fullmatch(args.dione_revision) is None:
            raise _fail("--dione-revision must be the immutable 40-hex commit")
        import dione_surface as dione_module  # noqa: F811 - sibling tool module

        try:
            dione = dione_module.load_dione_surface(
                args.dione_root,
                repo=args.dione_repo,
                revision=args.dione_revision,
                require_shard_hashes=(args.dione_verify_shards == "full"),
            )
        except ValueError as error:
            raise _fail(str(error))
        want_bits = DIONE_PROFILES[args.profile][0]
        if abs(float(dione.bits) - want_bits) > 1e-6:
            raise _fail(
                f"--profile {args.profile} names a {want_bits} bpw release, but the "
                f"release declares bits={dione.bits}"
            )
        # --bf16 points at the MATERIALIZED non-routed tree for the same
        # mechanical reason it does for tr3 and exl3hf.  A Dione release keeps
        # its non-routed tensors in retained/ shards that hold no routed
        # payloads -- but those shards ALSO carry the 864 MTP-layer expert
        # tensors, transformers derives its checkpoint key set from the shard
        # FILES, and the streaming build filters every `.mlp.experts.N.` name
        # out of the index.  The materializer copies the measured non-routed
        # set into shards of its own, decoding NOTHING.
        inventory_path = (
            args.inventory.resolve() if args.inventory
            else (args.bf16.resolve() / "inventory.json")
        )
        inventory = _sealed_json(inventory_path, RELEASE_INVENTORY_SCHEMA, "inventory_sha256")
        model_revision = str(inventory.get("model_revision", ""))
        if model_revision != args.dione_revision:
            raise _fail(
                "the materialized tree's inventory binds revision "
                f"{model_revision!r}, not --dione-revision {args.dione_revision!r}"
            )
        if inventory.get("seal_mode") != "full-shard-sha256":
            raise _fail("streaming capture requires the exact full-shard-hash inventory")
        materialization = _sealed_json(
            args.bf16.resolve() / "materialization-receipt.json",
            dione_module.DIONE_MATERIALIZATION_SCHEMA,
            "receipt_sha256",
        )
        if (
            materialization.get("source_repo") != args.dione_repo
            or materialization.get("source_revision") != args.dione_revision
            or materialization.get("source_config_sha256") != dione.config_sha256
            or materialization.get("source_index_sha256") != dione.index_sha256
            or materialization.get("inventory_sha256") != inventory["inventory_sha256"]
        ):
            raise _fail(
                "the --bf16 tree was not materialized from THIS dione snapshot; "
                "re-run `dione_surface.py materialize` against the same --dione-root"
            )
        bits = int(dione.bits)
    elif args.source == "tr3":
        # A TR3-published release is the one third-party surface in this suite
        # that SEALS itself: exl3-mcg-storage-abi.json and
        # materialization-receipt.json state digests over the emitted name set,
        # the plan, the config and the index, and tr3_surface.verify_seal
        # RECOMPUTES every one of them from the published bytes before a single
        # payload is read.
        if args.tr3_root is None or not args.tr3_repo or not args.tr3_revision:
            raise _fail("--source tr3 requires --tr3-root, --tr3-repo and --tr3-revision")
        if sealed_capture.REVISION.fullmatch(args.tr3_revision) is None:
            raise _fail("--tr3-revision must be the immutable 40-hex commit")
        import exl3hf_surface as xs3  # noqa: F811 - the shared decode + materializer ABI
        import tr3_surface as tr3_module  # noqa: F811 - sibling tool module

        tr3 = tr3_module.load_tr3_surface(
            args.tr3_root,
            repo=args.tr3_repo,
            revision=args.tr3_revision,
            verify_shards=args.tr3_verify_shards,
        )
        want_bits = TR3_PROFILES[args.profile][0]
        if abs(tr3.declared_bits - want_bits) > 1e-6:
            raise _fail(
                f"--profile {args.profile} names a {want_bits} bpw release, but the "
                f"release declares bits={tr3.declared_bits}"
            )
        # --bf16 points at the MATERIALIZED non-routed tree, exactly as it does
        # for exl3hf -- and for a reason that is mechanical, not about scope.
        # A TR3 release's non-routed tensors ARE the official ones, so morally
        # the snapshot could serve as the model tree. It cannot, because they
        # are INTERLEAVED with routed payloads in the same 120 shards, and
        # transformers derives its checkpoint key set from the shard FILES, not
        # from the index: a symlink view whose index lists only the 1,618
        # non-routed names still reports the 54,272 routed payload tensors
        # living in those same files as unloaded, and the load gate -- rightly
        # -- refuses. The materializer writes the non-routed set into clean
        # shards of its own, VERBATIM for a TR3 release: nothing is decoded,
        # bf16 stays bf16 and the fp32 natives stay fp32.
        inventory_path = (
            args.inventory.resolve() if args.inventory
            else (args.bf16.resolve() / "inventory.json")
        )
        inventory = _sealed_json(inventory_path, RELEASE_INVENTORY_SCHEMA, "inventory_sha256")
        model_revision = str(inventory.get("model_revision", ""))
        if model_revision != args.tr3_revision:
            raise _fail(
                "the materialized tree's inventory binds revision "
                f"{model_revision!r}, not --tr3-revision {args.tr3_revision!r}"
            )
        if inventory.get("seal_mode") != "full-shard-sha256":
            raise _fail("streaming capture requires the exact full-hash inventory")
        materialization = _sealed_json(
            args.bf16.resolve() / "materialization-receipt.json",
            xs3.EXL3HF_MATERIALIZATION_SCHEMA,
            "receipt_sha256",
        )
        if (
            materialization.get("source_repo") != args.tr3_repo
            or materialization.get("source_revision") != args.tr3_revision
            or materialization.get("source_config_sha256") != tr3.config_sha256
            or materialization.get("source_index_sha256") != tr3.index_sha256
            or materialization.get("inventory_sha256") != inventory["inventory_sha256"]
        ):
            raise _fail(
                "the --bf16 tree was not materialized from THIS tr3 snapshot; "
                "re-run exl3hf_surface materialize against the same --tr3-root"
            )
        bits = int(round(tr3.declared_bits))
    elif args.source == "nvfp4":
        # There is no contract, no payload store, no sealed inventory and no
        # official tree in the path.  The provenance anchors are the quant
        # repo's immutable 40-hex revision plus the local config/index hashes
        # (and, unless --nvfp4-skip-shard-hashes, the fetched HF LFS manifest
        # the shard bytes were verified against).  nvfp4_surface fail-closed
        # censuses the index: every routed module NVFP4-complete, the
        # non-routed set exactly the official 1,618 names.
        if args.nvfp4_root is None:
            raise _fail("--source nvfp4 requires --nvfp4-root (local snapshot of the quant repo)")
        if args.bf16 is not None:
            raise _fail(
                "REFUSED: --bf16 plays no role in an NVFP4 run. The artifact itself ships "
                "every non-routed tensor as plain BF16 under the official names (measured "
                "scope, enforced by the surface census), and the non-routed view is built "
                "from the quant snapshot. Drop --bf16; to compare trees byte-for-byte run "
                "`nvfp4_surface.py verify-nonrouted` / `probe` instead."
            )
        if not args.nvfp4_revision or sealed_capture.REVISION.fullmatch(args.nvfp4_revision) is None:
            raise _fail("--source nvfp4 requires --nvfp4-revision <immutable 40-hex repo commit>")
        import nvfp4_surface as nvfp4_module  # noqa: F811 - sibling tool module

        nvfp4 = nvfp4_module.load_nvfp4_surface(
            args.nvfp4_root,
            repo=args.nvfp4_repo,
            revision=args.nvfp4_revision,
            require_shard_hashes=not args.nvfp4_skip_shard_hashes,
        )
        model_revision = args.nvfp4_revision
        bits = 4
    elif args.source in ("native", "gguf"):
        # There is no contract, no payload store and no MCG codec.  The ONE
        # provenance anchor for the OFFICIAL side is the sealed release
        # inventory, which binds the BF16 config.json and
        # model.safetensors.index.json this run consumes: ALL of them for
        # native (the floor reads its routed experts straight from that tree,
        # so the floor and the quants are stated against the same weights),
        # only config/tokenizer/vision for gguf, whose measured weights come
        # decoded from the artifact and are identified separately below.
        inventory_path = (
            args.inventory.resolve()
            if args.inventory
            else (args.packed_root.resolve() / "inventory.json" if args.packed_root else None)
        )
        if inventory_path is None or not inventory_path.is_file():
            raise _fail(
                f"--source {args.source} requires --inventory <sealed glm-release-inventory.v1> "
                "(or --packed-root, whose inventory.json is used)"
            )
        inventory = _sealed_json(inventory_path, RELEASE_INVENTORY_SCHEMA, "inventory_sha256")
        model_revision = str(inventory.get("model_revision", ""))
        if sealed_capture.REVISION.fullmatch(model_revision) is None:
            raise _fail("inventory model revision is not an immutable 40-hex commit")
        if inventory.get("seal_mode") != "full-shard-sha256":
            raise _fail("streaming capture requires the exact full-hash BF16 inventory")
        if args.source == "gguf":
            import gguf_surface as gguf_module

            if not args.gguf_files:
                raise _fail("--source gguf requires --gguf-file (every part of the artifact)")
            if not args.gguf_revision:
                raise _fail(
                    "--source gguf requires --gguf-revision <40-hex immutable repo commit>: "
                    "an unpinned community artifact cannot be identified in a receipt"
                )
            remote = any(str(item).startswith(("http://", "https://"))
                         for item in args.gguf_files)
            if remote and not args.dry_run:
                raise _fail(
                    "https --gguf-file locations are metadata-only: a real capture needs the "
                    "local files (plus gguf_surface.py verify-files). Use --dry-run to "
                    "validate the plan against the live repo."
                )
            gguf_surface_obj = gguf_module.load_gguf_surface(
                [str(item) for item in args.gguf_files],
                repo=args.gguf_repo,
                revision=args.gguf_revision,
                require_file_hashes=not (args.skip_gguf_hashes or remote),
            )
    elif args.source == "checkpoint":
        if args.checkpoint is None:
            raise _fail("--source checkpoint requires --checkpoint")
        checkpoint_root = args.checkpoint.resolve()
        materialization = _read_json(
            checkpoint_root / "materialization-receipt.json",
            "materialization receipt (materialize the checkpoint first)",
        )
        packed_root = Path(str(materialization.get("packed_root", ""))).resolve()
        if not packed_root.is_dir():
            raise _fail(f"packed root from the materialization receipt is absent: {packed_root}")
    elif args.source == "exl3hf":
        # The measured function is the stock-exllamav3 artifact ALONE: routed
        # experts stream-decode from its shards, and everything else comes from
        # the --bf16 tree that exl3hf_surface.materialize_nonrouted dequantized
        # from the SAME snapshot.  The provenance chain is: artifact revision
        # -> materialization receipt (binds the artifact's config/index shas)
        # -> local inventory (binds the materialized tree the model loads).
        import exl3hf_surface as xs3

        if args.exl3hf_root is None or not args.exl3hf_repo or not args.exl3hf_revision:
            raise _fail(
                "--source exl3hf requires --exl3hf-root, --exl3hf-repo and --exl3hf-revision"
            )
        if sealed_capture.REVISION.fullmatch(args.exl3hf_revision) is None:
            raise _fail("--exl3hf-revision must be the immutable 40-hex commit")
        exl3hf = xs3.load_surface(args.exl3hf_root)
        want_bits = EXL3HF_PROFILES[args.profile][0]
        if abs(exl3hf.declared_bits - want_bits) > 1e-6:
            raise _fail(
                f"profile {args.profile} expects a {want_bits}-bpw artifact, but the "
                f"checkpoint declares bits={exl3hf.declared_bits}"
            )
        inventory_path = (
            args.inventory.resolve() if args.inventory
            else (args.bf16.resolve() / "inventory.json")
        )
        inventory = _sealed_json(inventory_path, RELEASE_INVENTORY_SCHEMA, "inventory_sha256")
        model_revision = str(inventory.get("model_revision", ""))
        if model_revision != args.exl3hf_revision:
            raise _fail(
                "the materialized tree's inventory binds revision "
                f"{model_revision!r}, not --exl3hf-revision {args.exl3hf_revision!r}"
            )
        if inventory.get("seal_mode") != "full-shard-sha256":
            raise _fail("streaming capture requires the exact full-hash inventory")
        materialization = _sealed_json(
            args.bf16.resolve() / "materialization-receipt.json",
            xs3.EXL3HF_MATERIALIZATION_SCHEMA,
            "receipt_sha256",
        )
        if (
            materialization.get("source_repo") != args.exl3hf_repo
            or materialization.get("source_revision") != args.exl3hf_revision
            or materialization.get("source_config_sha256") != exl3hf.config_sha256
            or materialization.get("source_index_sha256") != exl3hf.index_sha256
            or materialization.get("inventory_sha256") != inventory["inventory_sha256"]
        ):
            raise _fail(
                "the materialized non-routed tree does not bind this exact artifact "
                "snapshot (repo/revision/config/index/inventory mismatch) - "
                "re-run exl3hf_surface materialize against the same --exl3hf-root"
            )
    else:
        if args.packed_root is None:
            raise _fail("--source payload-store requires --packed-root (the encode output root)")
        packed_root = args.packed_root.resolve()
    if packed_root is not None:
        for required in ("contract.json", "inventory.json", "mtp-adapter-receipt.json"):
            if not (packed_root / required).is_file():
                raise _fail(f"packed root {packed_root} lacks {required}")
        for required in ("payload-store/objects", "payload-store/choices"):
            if not (packed_root / required).is_dir():
                raise _fail(f"packed root {packed_root} lacks {required}")

        inventory = _sealed_json(
            packed_root / "inventory.json", RELEASE_INVENTORY_SCHEMA, "inventory_sha256"
        )
        model_revision = str(inventory.get("model_revision", ""))
        if sealed_capture.REVISION.fullmatch(model_revision) is None:
            raise _fail("inventory model revision is not an immutable 40-hex commit")
        if inventory.get("seal_mode") != "full-shard-sha256":
            raise _fail("streaming capture requires the exact full-hash BF16 inventory")
        if materialization is not None and materialization.get(
            "source_inventory_sha256"
        ) != inventory["inventory_sha256"]:
            raise _fail("materialized checkpoint binds a different inventory")

        raw_contract = _read_json(packed_root / "contract.json", "direct contract")
        bits = int(raw_contract.get("rate", {}).get("bits", -1))
        expected_bits = {"k6": 6, "k8": 8}.get(args.profile)
        if expected_bits is not None and bits != expected_bits:
            raise _fail(f"profile {args.profile} expects K{expected_bits}, contract is K{bits}")
        contract = _sealed_json(
            packed_root / "contract.json", contract_schema_for_bits(bits), "contract_sha256"
        )
        if contract_bits(contract) != bits:
            raise _fail("packed student contract rate differs")
        if contract.get("inventory_sha256") != inventory["inventory_sha256"]:
            raise _fail("direct MCG contract targets another BF16 inventory")
        mtp_adapter = _sealed_json(
            packed_root / "mtp-adapter-receipt.json", _mtp_adapter_schema(bits), "receipt_sha256"
        )

    # ---- model-tree identity (official BF16, or the mlx/nvfp4/tr3 snapshot) --
    if args.source == "mlx":
        # The quant snapshot IS the model: geometry was gated by
        # load_mlx_surface against the same released constants, and identity
        # is the repo revision + config/index sha256 + official-census binding
        # (there is no sealed inventory for a community artifact).
        model_root = mlx_surface_obj.root
        config = dict(mlx_surface_obj.config)
        text_config = config.get("text_config", {})
    else:
        if args.source == "nvfp4":
            # the quant snapshot IS the model tree here too: its config carries
            # the same released geometry (gated below AND inside
            # load_nvfp4_surface), and its 1,618 non-routed tensors are the
            # official name set (census).  Unlike mlx it needs no decoded view,
            # because those tensors are already plain BF16 under the official
            # names -- so it goes through the ordinary config/index gate below.
            model_root = args.nvfp4_root.resolve()
        else:
            if args.bf16 is None:
                raise _fail(
                    f"--bf16 (official BF16 checkpoint) is required for --source {args.source}"
                )
            model_root = args.bf16.resolve()
        config_path = model_root / "config.json"
        index_path = model_root / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise _fail("official model requires config.json and model.safetensors.index.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = config.get("text_config", {})
        if (
            config.get("architectures") != [RELEASED_ARCHITECTURE]
            or config.get("model_type") != RELEASED_MODEL_TYPE
            or text_config.get("model_type") != RELEASED_TEXT_MODEL_TYPE
            or text_config.get("num_hidden_layers") != 45
            or text_config.get("num_nextn_predict_layers") != 1
            or text_config.get("n_routed_experts") != 288
            or text_config.get("hidden_size") != 4096
            or text_config.get("moe_intermediate_size") != 2048
        ):
            raise _fail("official GLM5Next main/MTP geometry differs")
        if args.source == "nvfp4":
            # no sealed inventory exists for a community snapshot; the binding
            # is the surface's own config/index hashes, re-checked against the
            # files this run actually reads (they enter
            # checkpoint_identity_sha256)
            if nvfp4.config_sha256 != sha256_file(config_path) or (
                nvfp4.index_sha256 != sha256_file(index_path)
            ):
                raise _fail("nvfp4 surface does not bind the local config/index")
        elif inventory.get("config_sha256") != sha256_file(config_path) or inventory.get(
            "index_sha256"
        ) != sha256_file(index_path):
            raise _fail("BF16 inventory does not bind the local config/index")

    # ---- surface + panel --------------------------------------------------
    surface_started = time.monotonic()
    native_source: Optional[NativeCheckpointSource] = None
    native_layout: Optional[Dict[str, Any]] = None
    exl3hf_reader = None
    exl3hf_layout: Optional[Dict[str, Any]] = None
    mlx_expert_source = None
    mlx_layout: Optional[Dict[str, Any]] = None
    gguf_source = None
    gguf_layout: Optional[Dict[str, Any]] = None
    gguf_summary: Optional[Dict[str, Any]] = None
    nvfp4_expert_source = None
    tr3_reader = None
    tr3_layout: Optional[Dict[str, Any]] = None
    dione_layout: Optional[Dict[str, Any]] = None
    if args.source == "mlx":
        import mlx_surface as mlxs

        mlx_layout = {
            "routed_expert_modules": len(mlx_surface_obj.census["routed_modules"]),
            "mtp_expert_modules": len(mlx_surface_obj.census["routed_mtp_modules"]),
            "nonrouted_quantized_modules": len(
                mlx_surface_obj.census["nonrouted_quantized_modules"]
            ),
            "passthrough_tensor_count": len(mlx_surface_obj.census["passthrough_tensors"]),
            "bits_histogram": dict(mlx_surface_obj.census["bits_histogram"]),
            "fetch_ledger": mlx_surface_obj.fetch_ledger(),
            "config_agreement": dict(mlx_surface_obj.config_agreement),
            "metadata_only": mlx_surface_obj.metadata_only,
        }
        if not args.dry_run:
            if mlx_surface_obj.metadata_only:
                raise _fail(
                    "--source mlx capture needs the real shards on disk; this root is a "
                    "metadata-only fetch-meta snapshot (fine for --dry-run only)"
                )
            try:
                mlx_expert_source = mlxs.MlxExpertSource(mlx_surface_obj)
            except ValueError as error:
                raise _fail(str(error))
    elif args.source == "native":
        native_source = NativeCheckpointSource(model_root)
        native_layout = native_source.routed_tensor_census(
            tuple(MAIN_ROUTED_LAYERS), int(text_config["n_routed_experts"])
        )
    elif args.source == "exl3hf":
        exl3hf_reader = xs3.Exl3HfShardReader(exl3hf)
        exl3hf_layout = xs3.routed_census(
            exl3hf, tuple(MAIN_ROUTED_LAYERS), int(text_config["n_routed_experts"])
        )
    elif args.source == "gguf":
        import gguf_surface as gguf_module

        gguf_source = gguf_module.GgufExpertSource(gguf_surface_obj)
        gguf_layout = gguf_source.routed_tensor_census(tuple(MAIN_ROUTED_LAYERS))
        # the artifact's non-routed map must EXACTLY biject the official
        # non-routed non-vision set of the LOCAL bf16 index the inventory binds
        gguf_layout["nonrouted_bijection"] = gguf_module.verify_nonrouted_bijection(
            gguf_surface_obj.census,
            json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].keys(),
        )
        gguf_summary = gguf_module.surface_summary(gguf_surface_obj)
    elif args.source == "tr3":
        # The routed payload layout and the decode are IDENTICAL to exl3hf's,
        # so the streamer gets the same (surface, reader) pair and runs the
        # same fill loop.  A second fill implementation would be a second thing
        # to keep correct, for no gain: what differs about TR3 is the seal, the
        # scope and the non-routed source, none of which the fill loop touches.
        exl3hf, tr3_reader = tr3_module.expert_source(tr3)
        tr3_layout = tr3_module.routed_census(tr3)
    elif args.source == "dione":
        # The census already closed inside load_dione_surface (every routed
        # slice present, none stray, MTP native, non-routed name set exactly
        # the official 1,618).  The reader only adds thread-local safetensors
        # handles so the fill loop can read payloads from a pool.
        dione_shards = dione_module.DioneShardReader(dione)
        dione_layout = dione_module.routed_census(dione)
    elif args.source == "nvfp4":
        # census/geometry already closed inside load_nvfp4_surface; the
        # source object only adds the thread-safe read+decode machinery.
        nvfp4_expert_source = nvfp4_module.Nvfp4ExpertSource(nvfp4)
    else:
        surface = load_complete_surface(
            root=packed_root, contract=contract, mtp_adapter_receipt=mtp_adapter
        )
    surface_seconds = time.monotonic() - surface_started
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    panel_path = sealed_capture._find_token_panel_receipt(
        [
            args.token_panel,
            args.teacher.resolve(),
            args.teacher.resolve().parent / "panel-v1",
            args.teacher.resolve().parent / "calibration" / "panel-v1",
            args.out.resolve().parent.parent / "calibration" / "panel-v1",
        ]
    )
    panel_receipt, _, all_windows = load_panel_windows(
        panel_path, roles=roles, vocab_size=int(text_config["vocab_size"])
    )
    if args.windows:
        wanted = [item.strip() for item in args.windows.split(",") if item.strip()]
        by_id = {window.window_id: window for window in all_windows}
        unknown = [item for item in wanted if item not in by_id]
        if unknown:
            raise _fail(f"--windows names ids absent from the panel: {unknown}")
        # positional index in the FULL panel is the sealed file-name convention
        selection = [(list(by_id).index(item), by_id[item]) for item in wanted]
    else:
        selection = list(enumerate(all_windows))

    if args.source == "exl3hf":
        identity = xs3.reader_identity(
            Path(__file__).resolve(),
            codebook=exl3hf.codebook,
            bits_note=(
                f"per-module (declared {exl3hf.declared_bits} bpw, "
                f"head_bits {exl3hf.declared_head_bits})"
            ),
        )
        checkpoint_identity = sha256_bytes(
            canonical_json(
                {
                    "schema": xs3.EXL3HF_IDENTITY_SCHEMA,
                    "inventory_sha256": inventory["inventory_sha256"],
                    "artifact_repo": args.exl3hf_repo,
                    "artifact_revision": args.exl3hf_revision,
                    "artifact_config_sha256": exl3hf.config_sha256,
                    "artifact_index_sha256": exl3hf.index_sha256,
                    "materialization_receipt_sha256": materialization["receipt_sha256"],
                    "codebook": exl3hf.codebook.upper(),
                    "declared_bits": exl3hf.declared_bits,
                    "declared_head_bits": exl3hf.declared_head_bits,
                    "routed_policy": "streamed_offline_decode_stock_exl3_full_matrices",
                    "nonrouted_policy": "artifact_dequantized_bf16_materialized_tree",
                }
            )
        )
    elif args.source == "mlx":
        import mlx_surface as mlxs

        identity = mlxs.mlx_reader_identity(Path(__file__).resolve(), mlx_surface_obj)
        checkpoint_identity = mlx_surface_obj.checkpoint_identity_sha256()
    elif args.source == "gguf":
        import gguf_surface as gguf_module

        identity = gguf_module.gguf_reader_identity(
            Path(__file__).resolve(), surface=gguf_surface_obj
        )
        # The artifact IS the student: every tensor of the measured forward comes
        # from it.  The identity therefore binds the GGUF side (repo, immutable
        # revision, per-file sha256, the measured type census and scope policy)
        # AND the official side, which here supplies only config/tokenizer and
        # the vision tower the artifact does not carry.
        checkpoint_identity = sha256_bytes(
            canonical_json(
                {
                    "schema": GGUF_STUDENT_IDENTITY_SCHEMA,
                    "gguf_identity_sha256": gguf_surface_obj.checkpoint_identity_sha256(),
                    "inventory_sha256": inventory["inventory_sha256"],
                    "config_sha256": inventory["config_sha256"],
                    "index_sha256": inventory["index_sha256"],
                    "bits": None,
                    "codebook": "ggml-block-quants",
                    "routed_policy": "gguf_fused_expert_tensors_block_dequantized_to_bf16",
                    "routed_tensor_count": gguf_layout["routed_tensor_count"],
                    "nonrouted_policy": "decoded_from_the_same_gguf_artifact",
                    "nonrouted_tensor_count": gguf_layout["nonrouted_bijection"][
                        "nonrouted_mapped_tensors"
                    ],
                    "vision_policy": "official_bf16_not_in_artifact_never_executed",
                }
            )
        )
    elif args.source == "native":
        identity = native_source_identity(
            Path(reader_module.__file__).resolve(), Path(__file__).resolve()
        )
        checkpoint_identity = sha256_bytes(
            canonical_json(
                {
                    "schema": NATIVE_STUDENT_IDENTITY_SCHEMA,
                    "inventory_sha256": inventory["inventory_sha256"],
                    "config_sha256": inventory["config_sha256"],
                    "index_sha256": inventory["index_sha256"],
                    "bits": 16,
                    "codebook": None,
                    "routed_policy": "official_bf16_checkpoint_experts_no_decode",
                    "routed_tensor_count": native_layout["routed_tensor_count"],
                    "nonrouted_policy": "official_source_native",
                }
            )
        )
    elif args.source == "tr3":
        identity = tr3_module.tr3_reader_identity(Path(__file__).resolve(), tr3)
        checkpoint_identity = tr3.checkpoint_identity_sha256()
    elif args.source == "dione":
        identity = dione_module.dione_reader_identity(Path(__file__).resolve(), bits=bits)
        checkpoint_identity = dione.checkpoint_identity_sha256()
    elif args.source == "nvfp4":
        identity = nvfp4_module.nvfp4_reader_identity(Path(__file__).resolve(), nvfp4)
        checkpoint_identity = nvfp4.checkpoint_identity_sha256()
    else:
        identity = reader_identity(
            Path(reader_module.__file__).resolve(), Path(__file__).resolve(), bits=bits
        )
        checkpoint_identity = sha256_bytes(
            canonical_json(
                {
                    "schema": f"quant-pipeline.glm53-packed-k{bits}-student-identity.v1",
                    "inventory_sha256": inventory["inventory_sha256"],
                    "contract_sha256": surface.contract_sha256,
                    "main_layer_receipt_sha256": list(surface.main_layer_receipt_sha256),
                    "mtp_adapter_receipt_sha256": surface.mtp_adapter_receipt_sha256,
                    "mtp_pack_receipt_sha256": surface.mtp_pack_receipt_sha256,
                    "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
                    "bits": bits,
                    "codebook": "MCG",
                    "nonrouted_policy": "official_source_native",
                }
            )
        )

    memory = plan_memory(args.vram_budget_gb, args.ep_emulate, args.slab_experts, args.decode_cache)
    disclosure = {
        "schema": DISCLOSURE_SCHEMA,
        "streaming_mode": args.stream_mode,
        "decode_cache": args.decode_cache,
        "slab_experts": memory["slab_experts"],
        "ep_semantics": (
            "single_device"
            if args.ep_emulate == 1
            else f"single_device_emulating_expert_parallel_world_size_{args.ep_emulate}"
        ),
        "ep_emulate": args.ep_emulate,
        "reduce_order": args.reduce_order,
        "experts_implementation": args.experts_implementation,
        "attention_backend": args.attention_backend,
        "dtype_policy": {
            "weights": "bfloat16 decoded from the packed payload (fp32 decode, one rounding)",
            "nonrouted": "official bfloat16 shard bytes, untouched",
            "router": "float32 (upstream)",
            "expert_combine": "float32 slot sum inside grouped_mm_experts_forward, bf16 per EP partial",
            "stored_logits": "float32",
        },
        "sealed_path_differences": [
            "residency: 288 experts of ONE layer at a time instead of 36 experts of every "
            "layer resident on each of 8 ranks",
            "process topology: 1 process, 1 device instead of torchrun world size 8",
            "expert weights are plain module attributes rather than nn.Parameter so the slab "
            "can be rebound per layer / per EP group (values and dtypes unchanged)",
            "model construction: the SAME from_pretrained call, but over a directory whose "
            "model.safetensors.index.json lists only the 1,618 non-routed checkpoint tensors "
            "(shards symlinked), instead of reading all 599 GB to discard 580 GB of it; the "
            "routed expert Parameters are removed at construction and re-bound to the slab. "
            "The loader reports missing_keys 0 / mismatched_keys 0 / error_msgs 0, and every "
            "unexpected key is one of the streamed routed experts (the sharded path reports "
            "exactly those; the single-file rewrite path reports none, because those modules "
            "carry no parameters by then - load_report records which). A single NON-routed "
            "tensor left unloaded is a hard failure",
            "the routed-expert combine is summed locally; the sealed run summed 8 bf16 per-rank "
            "partials with dist.all_reduce over NCCL, whose element order is not reproducible "
            "from one process (--reduce-order enumerates the candidates)",
        ],
        "sealed_path_identical": [
            "decode contract: glm53_packed_k4_reader.decode_choice_hf and the three sealed "
            "payload hash gates, called unmodified",
            "expert install algebra: fuse_gate_up + single fp32->bf16 rounding + torch.equal close",
            "forward: the model's own module classes and op order; window-major replays the "
            "sealed per-window model(...) call with batch 1, seq 2048, use_cache=False",
            "attention backend eager, tf32 off, float32_matmul_precision highest",
            "stored logits: logits[:, :-1, :] then boolean select on mask[:-1] & mask[1:], float32",
            "MTP layer 45 is receipt-gated and never executed",
        ],
        "mtp_standard_logits_executed": False,
    }
    if args.source == "native":
        # The native lane differs from a K6/K8 streaming run in exactly ONE
        # place: where the routed expert bytes come from.  Everything the
        # disclosure above asserts about residency, topology, slab binding,
        # model construction and the expert combine is unchanged and still true.
        disclosure["no_decode"] = True
        disclosure["routed_weight_source"] = (
            "official BF16 checkpoint shards, read per expert by the released tensor names "
            "(model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight) - the same "
            "names the ENCODER read to build the packed store"
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 released checkpoint bytes, unmodified (no codec; the packed lane's single "
            "fp32->bf16 rounding is the identity on values that are already bf16)"
        )
        disclosure["sealed_path_identical"] = [
            item
            for item in disclosure["sealed_path_identical"]
            if not item.startswith(("decode contract:", "expert install algebra:"))
        ]
        disclosure["sealed_path_identical"].insert(
            0,
            "expert install algebra: fuse_gate_up + the single fp32->bf16 rounding (the identity "
            "here, the inputs are already bf16) + torch.equal close - the packed lane's own "
            "installation code with only the decode call removed",
        )
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: the UN-QUANTIZED BF16 experts. No payload store, no contract, no "
            "MCG codebook and no hash-gated decode are in the path; the provenance anchor is the "
            "sealed release inventory's config_sha256/index_sha256, which bind the tree these "
            "tensors are read from. Shard bytes are not re-hashed by this tool (neither are they "
            "on the packed lane, whose non-routed tensors come from the same shards)",
        ]
        disclosure["native_routed_layout"] = native_layout
    if args.source == "exl3hf":
        # Same lane, two disclosed differences from a packed K6/K8 run: where
        # the routed bytes come from (stock exl3 payloads, variable K, no seal
        # to verify) and where the non-routed weights come from (the artifact's
        # own tensors, dequantized to BF16 by the materializer -- NOT the
        # official release).  Both are artifact identity, not lane identity.
        disclosure["routed_weight_source"] = (
            "stock exllamav3 payloads (<module>.{trellis,suh,svh,%s}) read per expert "
            "from the artifact's own HF shards and offline-decoded with the campaign "
            "decode ABI (fp32 hadamards, %s LUT, one bf16 rounding)"
            % (exl3hf.codebook, exl3hf.codebook.upper())
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 decoded from stock exl3 payloads (fp32 decode, one rounding)"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "bfloat16 MATERIALIZED from the artifact's own (quantized or native) "
            "tensors by exl3hf_surface.materialize_nonrouted - the lm_head included; "
            "no official-release weight is part of the measured function"
        )
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: a stock-exllamav3 release (codebook %s, per-module bit "
            "rate). No payload store, no contract and no encoder closure exist; the "
            "provenance anchors are the immutable artifact revision, the materialization "
            "receipt and the consumed-payload sha256 census (seal_disclosure records the "
            "same unsealed-source deviation the Dione rows carry)" % exl3hf.codebook,
            "non-routed weights: dequantized from the SAME artifact snapshot (its "
            "attention, shared experts, dense MLPs, vision tower and 6-bit lm_head are "
            "part of the measured function; the official BF16 release contributes "
            "nothing). The head is applied natively from those dequantized weights",
        ]
        disclosure["exl3hf_routed_layout"] = exl3hf_layout
        disclosure["seal_disclosure"] = xs3.SEAL_DISCLOSURE
    elif args.source == "tr3":
        # Two disclosed differences from a packed K6/K8 run, and NEITHER is
        # "unsealed": where the routed bytes come from (a TR3-published
        # release's own shards, whose seal this run recomputed) and where the
        # non-routed weights come from (the artifact's own tensors, which the
        # publisher declares and this run verifies to be the OFFICIAL ones).
        disclosure["routed_weight_source"] = (
            "TR3-published EXL3/MCG payloads (<module>.{trellis,suh,svh,mcg}) read per "
            "expert from the artifact's own HF shards and offline-decoded with the "
            "campaign decode ABI (fp32 hadamards, frozen MCG LUT, one bf16 rounding) - "
            "the identical codec the K6/K8 rows on this lane were measured through"
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 decoded from TR3 EXL3/MCG payloads (fp32 decode, one rounding)"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "the ARTIFACT's own non-routed tensors, copied VERBATIM: the release quantizes "
            "the routed experts only and ships all %d non-routed tensors as the official "
            "source tensors (non_routed_dtype_policy official_source_native; dtype census "
            "%s). The materializer re-shards them without decoding anything -- bf16 stays "
            "bf16, the fp32 natives stay fp32 -- because they share shards with the routed "
            "payloads and transformers keys its load off the shard FILES. lm_head is native "
            "BF16 (head_bits 16) and no official-release weight is in the path"
            % (tr3.nonrouted_tensor_count,
               json.dumps(tr3.dtype_census, sort_keys=True))
        )
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: a TR3-published release (codebook mcg, uniform K%d). There "
            "is no payload store and no encoder closure in the path, but unlike the "
            "stock-exllamav3 and Dione surfaces this release SEALS itself: the "
            "provenance anchors are the immutable artifact revision plus "
            "exl3-mcg-storage-abi.json and materialization-receipt.json, every claim of "
            "which tr3_surface.verify_seal recomputed from the published bytes before "
            "decoding (%d checks)"
            % (int(round(tr3.declared_bits)), len(tr3.seal.get("checks") or [])),
            "non-routed weights: the artifact's OWN tensors, re-sharded verbatim by "
            "exl3hf_surface.materialize_nonrouted (no decode -- this release quantizes "
            "nothing outside the routed experts). They are byte-exact official "
            "(nonrouted_native_exact, and the name set was verified equal to the official "
            "release's %d non-routed names). The head is applied natively from those "
            "weights and is not quantized"
            % tr3.nonrouted_tensor_count,
        ]
        disclosure["tr3_routed_layout"] = tr3_layout
        disclosure["tr3"] = tr3_module.surface_summary(tr3)
        disclosure["seal_disclosure"] = tr3_module.SEAL_DISCLOSURE
    elif args.source == "dione":
        # Two disclosed differences from a packed K6/K8 run: where the routed
        # bytes come from (an UNSEALED third-party release, decoded per TP
        # slice and concatenated) and where the non-routed weights come from
        # (that same artifact's retained tensors, copied verbatim).
        disclosure["routed_weight_source"] = (
            "Dione selective-EXL3 payloads, %d TP-rank slices per matrix "
            "(<module>.rank{R}.{trellis,suh,svh,mcg}) read from the artifact's own "
            "layers/ shards, decoded with the campaign decode ABI (fp32 hadamards, "
            "frozen MCG LUT) and concatenated in rank order (dim %s), then one bf16 "
            "rounding - the identical codec the K6/K8 rows on this lane were measured "
            "through, at K%d"
            % (dione.tp_size, json.dumps(dione_module.CONCAT_DIM, sort_keys=True), bits)
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 decoded from Dione EXL3/MCG TP slices (fp32 decode + rank concat, "
            "one rounding)"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "the ARTIFACT's own retained tensors, copied VERBATIM: the release quantizes "
            "the routed experts of layers 3-44 only and retains everything else at source "
            "precision (retained_dtype: source_precision), including the lm_head. Its "
            "non-routed name set was verified equal to the official release's 1,618 names "
            "before anything was read; the materializer re-shards them without decoding "
            "or casting anything. No official-release weight is in the measured path"
        )
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: a Dione (0xSero) selective-EXL3 release, uniform K%d, stored "
            "TP%d-sliced. There is no payload store, no contract, no encoder closure and "
            "NO publisher seal; the provenance anchors are the immutable artifact "
            "revision, the release manifest's per-shard sha256 (%s), the local "
            "config/index digests and the consumed-payload sha256 census"
            % (bits, dione.tp_size, dione.shard_hash_verification),
            "non-routed weights: the artifact's OWN retained tensors, re-sharded verbatim "
            "by dione_surface.materialize_nonrouted (no decode -- this release quantizes "
            "nothing outside the routed experts of layers 3-44). The head is applied "
            "natively from those weights and is NOT quantized (head_bits 16), unlike a "
            "stock-exllamav3 release",
        ]
        disclosure["dione_routed_layout"] = dione_layout
        disclosure["dione"] = dione_module.surface_summary(dione)
        disclosure["seal_disclosure"] = dione_module.SEAL_DISCLOSURE
    elif args.source == "mlx":
        import mlx_surface as mlxs

        # The MLX lane differs from a K6/K8 streaming run in TWO places, both
        # named here: where the routed expert bytes come from (an unsealed
        # community snapshot, affine-decoded) and where the non-routed model
        # comes from (a DECODED bf16 view of that same snapshot, because this
        # format quantizes beyond the routed experts).  Residency, topology,
        # slab binding, the from_pretrained constructor and the expert combine
        # are unchanged and still true above.
        disclosure["routed_weight_source"] = (
            "MLX affine u32-packed community snapshot, read per expert by the released "
            "tensor names and dequantized in fp32 (q * scale + bias, groups of "
            f"{mlx_surface_obj.default_group_size} along the input axis; per-tensor bits "
            "derived from shapes against the official BF16 shape census)"
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 decoded from the MLX affine payload (fp32 dequant, one rounding); "
            "proven bitwise-equal to mlx.core.dequantize at f16 on real tensors"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "materialized decoded view of the quant snapshot: passthrough tensors verbatim "
            "(byte-identical dtype/shape to the official tree), quantized non-routed tensors "
            "fp32-dequantized and rounded once to bf16"
        )
        disclosure["sealed_path_identical"] = [
            item
            for item in disclosure["sealed_path_identical"]
            if not item.startswith("decode contract:")
        ]
        # The generic "model construction" difference above says the non-routed
        # index lists symlinked OFFICIAL shards.  That is false for this source
        # and a receipt must not carry a false clause next to a true one, so the
        # clause is rewritten in place rather than contradicted further down.
        disclosure["sealed_path_differences"] = [
            item.replace(
                "(shards symlinked), instead of reading all 599 GB to discard 580 GB of it",
                "(shards MATERIALIZED by mlx_surface as decoded bf16, NOT symlinked official "
                "shards - see nonrouted_policy), instead of reading the whole artifact to "
                "discard the routed part of it",
            )
            for item in disclosure["sealed_path_differences"]
        ]
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: an UNSEALED community MLX artifact. No payload store, no "
            "contract, no MCG codebook and no hash-gated decode are in the path; the "
            "provenance anchor is the immutable repo revision plus config/index sha256 "
            "plus the official-BF16 shape-census binding (see seal_disclosure)",
            "non-routed model: built from a materialized DECODED view of the quant snapshot "
            "instead of the official BF16 tree, because this artifact quantizes dense MLPs, "
            "shared experts and DSA attention projections too (see scope_policy); the "
            "sealed from_pretrained constructor and its zero-missing/zero-stray load "
            "assertions run over that view unchanged",
        ]
        disclosure["scope_policy"] = mlx_surface_obj.scope_policy()
        disclosure["seal_disclosure"] = mlxs.SEAL_DISCLOSURE
        disclosure["dtype_disclosure"] = mlxs.DTYPE_DISCLOSURE
        disclosure["mlx_routed_layout"] = mlx_layout
    elif args.source == "gguf":
        import gguf_surface as gguf_module

        # The GGUF lane differs from a K6/K8 streaming run in TWO places: where
        # the routed expert bytes come from, and -- unlike every other source --
        # where the NON-ROUTED bytes come from.  Residency, topology, slab
        # binding and the expert combine are unchanged and still true.
        disclosure["routed_weight_source"] = (
            "the GGUF's fused blk.L.ffn_{gate,up,down}_exps.weight tensors, sliced per expert "
            "and block-dequantized (ggml Q4_K/Q5_K/Q6_K/Q8_0 -> fp32) by k6/tools/gguf_surface.py, "
            "whose kernels are bitwise equal to gguf-py's reference dequantize()"
        )
        disclosure["nonrouted_weight_source"] = (
            "the SAME GGUF artifact, decoded once into a materialized safetensors view under the "
            "official HF tensor names -- token_embd, output (lm_head), every attention/KDA/DSA "
            "projection, the shared experts and the dense MLPs. The official BF16 tree supplies "
            "ONLY config/tokenizer files and the vision tower, which the main GGUF does not carry "
            "and which the text-only sealed panel never executes"
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 dequantized from the GGUF blocks (fp32 dequant, one rounding)"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "bfloat16 dequantized from the SAME GGUF (fp32 dequant, one rounding); the 291 "
            "tensors the official tree stores float32 are stored float32 in the GGUF too and "
            "pass through byte-exactly"
        )
        disclosure["sealed_path_identical"] = [
            item
            for item in disclosure["sealed_path_identical"]
            if not item.startswith("decode contract:")
        ]
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "routed surface: a third-party llama.cpp GGUF. No payload store, no contract, no MCG "
            "codebook and no hash-gated decode are in the path; provenance is the immutable repo "
            "revision plus a whole-file sha256 of every consumed .gguf",
            "NON-ROUTED surface: also the artifact's, decoded. Every other source in this tool "
            "scores the official BF16 non-routed parameters untouched; this one cannot, because "
            "the artifact quantized them too. A GGUF number and a K6/K8/Dione number are "
            "therefore NOT measuring the same set of weights, and the scope_policy block below "
            "says what each artifact actually quantized (read from its own tensor table)",
            "kv_b_proj: llama.cpp stores the MLA projection pre-split as attn_k_b (per-head "
            "transposed) and attn_v_b; the official tensor is reconstructed from the two halves "
            "in the arrangement proven against the official BF16 weight (see mla_placement)",
        ]
        disclosure["gguf_routed_layout"] = gguf_layout
        disclosure["gguf_surface"] = gguf_summary
        disclosure["scope_policy"] = gguf_surface_obj.scope_policy
        disclosure["seal_disclosure"] = gguf_module.SEAL_DISCLOSURE
    elif args.source == "nvfp4":
        # Same residency/topology/combine story as every streaming run; what
        # changes is the codec and its provenance.  The full surface block
        # (repo, revision, layout, quant config, measured scope policy,
        # activation caveat, seal disclosure) rides inside the disclosure so
        # plan/backend/receipt all carry it without changing the sealed
        # receipt's key set (stream_score_selftest rung L1.j).
        disclosure["nvfp4"] = nvfp4_module.surface_summary(nvfp4)
        disclosure["routed_weight_source"] = (
            "community NVFP4 snapshot, decoded per expert on the read-pool CPU threads: "
            "e2m1 nibble LUT x per-16-group f8e4m3 scale, one fp32 global per tensor "
            f"({nvfp4.layout} convention), exact fp32 throughout, then the lane's single "
            "fp32->bf16 rounding at slab install"
        )
        disclosure["dtype_policy"] = dict(disclosure["dtype_policy"])
        disclosure["dtype_policy"]["weights"] = (
            "bfloat16 rounded ONCE from the exact-fp32 NVFP4 dequant (e2m1 values, f8e4m3 "
            "scales and fp32 globals are all exact in fp32; one divide-or-multiply per "
            "group scale, one multiply per element)"
        )
        disclosure["dtype_policy"]["nonrouted"] = (
            "the snapshot's own bfloat16 shard bytes, untouched (the artifact ships the "
            "official 1,618-name non-routed set unquantized)"
        )
        disclosure["sealed_path_identical"] = [
            item
            for item in disclosure["sealed_path_identical"]
            if not item.startswith("decode contract:")
        ]
        disclosure["sealed_path_differences"] = list(disclosure["sealed_path_differences"]) + [
            "decode contract: nvfp4_surface.dequant_nvfp4 (proven bitwise against "
            "compressed-tensors' unpack+dequant in fp32 on real tensors) replaces the "
            "hash-gated MCG decode; the NVFP4 checkpoint ships no encoder-side seals, so "
            "component sha256s + the immutable repo revision are recorded instead "
            "(seal_disclosure)",
            "non-routed view: built from the QUANT snapshot, not the official tree "
            "(measured scope: the artifact quantizes routed experts only and ships the "
            "official non-routed set as plain BF16; name-set equality enforced)",
            "weights-only lane: any activation quantization the artifact declares or "
            "ships scales for is NOT captured (see streaming_disclosure.nvfp4.activations)",
        ]

    plan = {
        "schema": STREAM_PLAN_SCHEMA,
        "model": str(model_root),
        "model_revision": model_revision,
        "weight_source": args.source,
        "packed_root": str(packed_root) if packed_root is not None else None,
        "checkpoint_root": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "materialization_receipt_sha256": (materialization or {}).get("receipt_sha256"),
        "inventory_sha256": inventory["inventory_sha256"] if inventory is not None else None,
        "contract_sha256": surface.contract_sha256 if surface is not None else None,
        "checkpoint_identity_sha256": checkpoint_identity,
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "packed_reader_abi_sha256": (
            surface.packed_reader_abi_sha256 if surface is not None else None
        ),
        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
        "roles": list(roles),
        "windows": [window.window_id for _, window in selection],
        "window_count": len(selection),
        "panel_window_count": len(all_windows),
        "prediction_positions": sum(window.prediction_positions for _, window in selection),
        "parallelism": disclosure["ep_semantics"],
        "reader_mode": identity["mode"],
        "final_tp2_serving_kernel": False,
        "cold_run": args.cold_run,
        "bits": bits,
        "main_routed_policy": (
            "streamed_official_bf16_checkpoint_experts_no_decode_one_layer_resident"
            if args.source == "native"
            else "streamed_decoded_mlx_affine_u32_to_bf16_one_layer_resident"
            if args.source == "mlx"
            else "streamed_gguf_block_dequant_to_bf16_one_layer_resident"
            if args.source == "gguf"
            else "streamed_exact_fp32_nvfp4_e2m1_gs16_decode_to_bf16_one_layer_resident"
            if args.source == "nvfp4"
            else f"streamed_decode_seal_verified_tr3_published_k{bits}_mcg_to_bf16_one_layer_resident"
            if args.source == "tr3"
            else (f"streamed_decode_hash_verified_dione_tp{dione.tp_size}_slices_k{bits}_mcg"
                  "_concat_to_bf16_one_layer_resident")
            if args.source == "dione"
            else f"streamed_decode_hash_verified_packed_k{bits}_mcg_to_bf16_one_layer_resident"
        ),
        "native_routed_layout": native_layout,
        "mlx_routed_layout": mlx_layout,
        "gguf_routed_layout": gguf_layout,
        "gguf_surface": gguf_summary,
        "mtp_policy": (
            "mtp_layer_45_present_in_the_checkpoint_but_not_executed_by_standard_logits"
            if args.source in ("native", "mlx", "gguf")
            else (
                "mtp_layer_45_experts_present_as_%s_identity_covered_but_not_executed_by_"
                "standard_logits" % nvfp4.scope["mtp_expert_format"].replace("-", "_")
            )
            if args.source == "nvfp4"
            else (
                "mtp_layer_45_experts_present_as_exl3_mcg_k%d_and_seal_covered_but_not_"
                "executed_by_standard_logits" % int(round(tr3.declared_bits))
            )
            if args.source == "tr3"
            else (
                "mtp_layer_45_experts_retained_at_source_precision_bf16_but_not_executed_"
                "by_standard_logits"
            )
            if args.source == "dione"
            else "complete_and_receipt_required_but_not_executed_by_standard_logits"
        ),
        "nonrouted_policy": (
            # the three sources whose non-routed tensors are not the official
            # tree's: mlx and gguf quantize past the routed experts, so theirs
            # are DECODED from the artifact; nvfp4 quantizes the routed experts
            # only but ships its own unquantized copy of the official
            # non-routed set, which is what the forward reads.
            "decoded_bf16_view_materialized_from_the_quant_snapshot"
            if args.source == "mlx"
            else "decoded_from_the_same_gguf_artifact_vision_from_official_bf16"
            if args.source == "gguf"
            else "quant_snapshot_bf16_parameters_official_name_set_unquantized_in_artifact"
            if args.source == "nvfp4"
            else "artifact_own_official_source_native_tensors_materialized_verbatim_no_decode"
            if args.source == "tr3"
            else "untouched_official_checkpoint_parameters"
        ),
        "stored_logits_dtype": "float32",
        "surface_load_seconds": surface_seconds,
        "memory_plan": memory,
        "streaming_disclosure": disclosure,
        "output": str(args.out.resolve()),
        "dry_run": args.dry_run,
    }
    if args.source == "exl3hf":
        plan.update(
            {
                "exl3hf_repo": args.exl3hf_repo,
                "exl3hf_revision": args.exl3hf_revision,
                "exl3hf_root": str(args.exl3hf_root.resolve()),
                "artifact_config_sha256": exl3hf.config_sha256,
                "artifact_index_sha256": exl3hf.index_sha256,
                "codebook": exl3hf.codebook,
                "exllamav3_version": exl3hf.exllamav3_version,
                "declared_bits": exl3hf.declared_bits,
                "declared_head_bits": exl3hf.declared_head_bits,
                "materialization_receipt_sha256": materialization["receipt_sha256"],
                "seal_disclosure": xs3.SEAL_DISCLOSURE,
                "main_routed_policy": (
                    "streamed_decode_stock_exl3_%s_per_module_bits_to_bf16_one_layer_resident"
                    % exl3hf.codebook
                ),
                "nonrouted_policy": "artifact_dequantized_bf16_materialized_tree",
                "mtp_policy": (
                    "mtp_layer_45_nonrouted_materialized_from_artifact_mtp_file_"
                    "but_not_executed_by_standard_logits"
                ),
                "exl3hf_routed_layout": exl3hf_layout,
            }
        )
    if args.source == "dione":
        plan.update(
            {
                "dione_repo": args.dione_repo,
                "dione_revision": args.dione_revision,
                "dione_root": str(args.dione_root.resolve()),
                "artifact_config_sha256": dione.config_sha256,
                "artifact_index_sha256": dione.index_sha256,
                "codebook": "mcg",
                "codec_family": "exl3-mcg",
                "declared_bits": float(dione.bits),
                "declared_head_bits": 16,
                "tp_size": dione.tp_size,
                "source_repo": dione.source_repo,
                "source_revision": dione.source_revision,
                "exl3_manifest_name": dione.exl3_manifest_name,
                "exl3_manifest_sha256": dione.exl3_manifest_sha256,
                "exl3_manifest_schema": dione.exl3_manifest_schema,
                "dione_shard_hash_verification": dione.shard_hash_verification,
                "materialization_receipt_sha256": materialization["receipt_sha256"],
                "scope_digest": dione_module.scope_digest(dione),
                "scope_census_sha256":
                    dione_module.surface_summary(dione)["scope_census_sha256"],
                "seal_disclosure": dione_module.SEAL_DISCLOSURE,
                "nonrouted_policy": "artifact_own_retained_natives_rematerialized_verbatim",
                "dione_routed_layout": dione_layout,
            }
        )
    if args.source == "tr3":
        plan.update(
            {
                "tr3_repo": args.tr3_repo,
                "tr3_revision": args.tr3_revision,
                "tr3_root": str(args.tr3_root.resolve()),
                "artifact_config_sha256": tr3.config_sha256,
                "artifact_index_sha256": tr3.index_sha256,
                "codebook": tr3.codebook,
                "codec_family": "exl3-mcg",
                "exllamav3_version": tr3.quantizer_version,
                "exllamav3_pin": tr3.exllamav3_pin,
                "declared_bits": tr3.declared_bits,
                "declared_head_bits": tr3.declared_head_bits,
                "scope_policy": tr3.scope_policy,
                "nonrouted_policy_declared": tr3.nonrouted_policy,
                "artifact_materialization_receipt_sha256":
                    tr3.seal["materialization"]["receipt_sha256"],
                "materialization_receipt_sha256": materialization["receipt_sha256"],
                "seal_verification": tr3.seal,
                "shard_verification": tr3.shard_verification,
                "scope_census_sha256": tr3.scope_census_sha256(),
                "seal_disclosure": tr3_module.SEAL_DISCLOSURE,
                "tr3_routed_layout": tr3_layout,
            }
        )
    if args.source == "dione":
        plan["dione"] = dione_module.surface_summary(dione)
        plan["dione"]["scope_digest"] = dione_module.scope_digest(dione)
    if args.source == "mlx":
        plan["mlx"] = {
            "mlx_repo": mlx_surface_obj.repo,
            "mlx_revision": mlx_surface_obj.revision,
            "config_sha256": mlx_surface_obj.config_sha256,
            "index_sha256": mlx_surface_obj.index_sha256,
            "official_census_sha256": mlx_surface_obj.official_census_sha256,
            "official_source_repo": mlx_surface_obj.official_source_repo,
            "official_source_revision": mlx_surface_obj.official_source_revision,
            "default_bits": mlx_surface_obj.default_bits,
            "default_group_size": mlx_surface_obj.default_group_size,
            "student_label": mlx_surface_obj.student_label(),
            "scope_policy": mlx_surface_obj.scope_policy(),
            "fetch_ledger": mlx_surface_obj.fetch_ledger(),
            "config_agreement": dict(mlx_surface_obj.config_agreement),
            "shard_hash_verification": mlx_surface_obj.shard_hash_verification,
            "bf16_passthrough_crosscheck_planned": args.bf16 is not None,
        }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    # ---- runtime ----------------------------------------------------------
    import torch
    from safetensors.torch import save_file
    from transformers import __version__ as transformers_version

    if tuple(int(part) for part in transformers_version.split(".")[:2]) < (5, 16):
        raise _fail("packed GLM5Next reader requires transformers>=5.16")
    device = resolve_device(args.device)
    numeric = apply_numeric_policy(device)
    decode_device = device if args.decode_device == "same" else resolve_device(args.decode_device)
    unpack_device = None if args.unpack_device == "same" else resolve_device(args.unpack_device)
    # --- where the GGUF block dequant runs --------------------------------
    # The source object was built before the device was resolved (it has to be:
    # its census gates the plan, which --dry-run prints without touching a
    # GPU), so the decision is bound here, once, before the first fill.
    #
    # This is the ONLY knob in this file that moves arithmetic between devices,
    # and it is safe to move because the arithmetic it moves has no
    # device-dependent operation in it: integer unpack plus elementwise fp32
    # multiply and subtract, no reduction, no matmul, no fusion.  Everything
    # AFTER the decode -- fuse_gate_up, the single fp32->bf16 rounding, the
    # copy_ into the slab, the torch.equal close -- already ran on `device` and
    # is untouched.  The receipt records which path ran.
    gguf_decode_device = None
    if args.source == "gguf":
        if args.gguf_decode_device == "cpu" or decode_device.type == "cpu":
            gguf_decode_device = None
        else:
            gguf_decode_device = decode_device
        if gguf_source is not None:
            gguf_source.decode_device = gguf_decode_device
    if device.type == "cuda":
        torch.cuda.set_device(device)  # sealed capture: torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)

    output_root = args.out.resolve()
    prepare_empty_destination(output_root)
    (output_root / "logits").mkdir()
    write_json(output_root / "plan.json", plan | {"dry_run": False})
    write_json(output_root / "reader-identity.json", identity)

    layers = tuple(MAIN_ROUTED_LAYERS)
    load_started = time.monotonic()
    work_dir = (args.work_dir or (output_root.parent / ".stream-work")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    nonrouted_view = None
    mlx_bf16_crosscheck: Optional[Dict[str, Any]] = None
    if args.source == "mlx":
        import mlx_surface as mlxs

        try:
            nonrouted_view = mlxs.prepare_nonrouted_view_decoded(mlx_surface_obj, work_dir)
            if args.bf16 is not None:
                # OPTIONAL: the quantizer left the passthrough tensors
                # untouched, so a deterministic sample must be byte-identical
                # to the official tree when the operator supplies one.
                mlx_bf16_crosscheck = mlxs.verify_nonrouted_passthrough(
                    mlx_surface_obj, args.bf16.resolve()
                )
        except ValueError as error:
            raise _fail(str(error))
    elif args.source == "gguf":
        # A GGUF quantizes the non-routed tensors too, so they cannot come from
        # the official tree: decode them ONCE into a safetensors view carrying
        # the official names/shapes/dtypes.  The view is fingerprinted by the
        # artifact identity + this adapter's hash and reused across cold runs.
        view_path, view_record = gguf_module.materialize_nonrouted_view(
            gguf_surface_obj, model_root, work_dir
        )
        view_record["value_spot_check"] = gguf_module.verify_view_nonrouted_values(
            gguf_surface_obj, view_path
        )
        write_json(output_root / "gguf-nonrouted-view.json", view_record)
        print(json.dumps({"gguf_nonrouted_view": view_record["tensor_count"],
                          "reused": view_record["reused"],
                          "bytes": view_record["tensor_bytes"]}, sort_keys=True), flush=True)
        nonrouted_view = (view_path, view_record)
    model, build = build_streaming_model(
        bf16_root=model_root,
        work_dir=work_dir,
        device=device,
        attn_implementation=args.attention_backend,
        experts_implementation=args.experts_implementation,
        layers=layers,
        nonrouted_view=nonrouted_view,
        # an nvfp4 / tr3 view is a DIFFERENT artifact than a bf16 view sharing
        # the same work_dir, and its config must lose quantization_config so
        # from_pretrained builds the sealed plain-BF16 model instead of
        # engaging a quantized-loading integration
        view_name=("nvfp4-nonrouted-view" if args.source == "nvfp4"
                   else "tr3-nonrouted-view" if args.source == "tr3"
                   else "bf16-nonrouted-view"),
        config_strip_keys=(("quantization_config",)
                           if args.source in ("nvfp4", "tr3") else ()),
    )
    streamer = ExpertStreamer(
        # the dione fill loop reads geometry (bits, tp_size) off the surface it
        # is given, so for that source the surface IS the DioneSurface
        surface=(dione if args.source == "dione" else surface),
        device=decode_device,
        bits=bits,
        layers=layers,
        slab_experts=memory["slab_experts"],
        decode_threads=args.decode_threads,
        progress_interval=args.progress_seconds,
        unpack_device=unpack_device,
        cache_mode=args.decode_cache,
        cache_dir=args.decode_cache_dir.resolve() if args.decode_cache_dir else None,
        # native, mlx, gguf and nvfp4 all satisfy the same one-call contract
        # the streamer's non-packed branch expects: load(layer=, expert=,
        # projection=) -> (CPU tensor, census row with tensor/shard/bytes).  The
        # caller still does fuse_gate_up, the single bf16 rounding and the
        # torch.equal close.
        native_source=native_source,
        # dione has its own fill loop: one matrix is tp_size independent EXL3
        # payloads that must be decoded and concatenated in rank order, which
        # is not the one-payload-per-matrix shape every other surface has.
        dione_shards=dione_shards,
        # tr3 rides the exl3hf fill loop deliberately: identical payload
        # objects, identical decode ABI, and one implementation to keep correct
        exl3hf_source=((exl3hf, exl3hf_reader) if args.source == "exl3hf"
                       else (exl3hf, tr3_reader) if args.source == "tr3"
                       else None),
        mlx_source=mlx_expert_source,
        gguf_source=gguf_source,
        nvfp4_source=nvfp4_expert_source,
        # NUM-04. What the decoded bytes in --decode-cache-dir are OF. Without it the
        # cache is keyed on the layer index alone, so a second run over a different
        # artifact reuses the first artifact's decoded experts and publishes them under
        # its own identity.
        cache_identity={
            "checkpoint_identity_sha256": checkpoint_identity,
            "source": args.source,
            "bits": bits,
            "slab_experts": memory["slab_experts"],
            "routed_layers": list(layers),
            "reader_identity": identity,
        },
    )
    if decode_device != device:
        raise _fail(
            "--decode-device must currently equal --device: the slab is allocated on the decode "
            "device and consumed by the forward (use --unpack-device cpu for the MPS int64 split)"
        )
    stats: Dict[str, Any] = {"expert_forward_seconds": 0.0}
    runtime_ep: Dict[str, Any] = {"ep": args.ep_emulate, "order": args.reduce_order}
    wiring = install_streaming_experts(
        model,
        streamer=streamer,
        layers=layers,
        ep_emulate=args.ep_emulate,
        reduce_order=args.reduce_order,
        stats=stats,
        runtime=runtime_ep,
    )
    closure = (
        None
        if surface is None
        else stored_encoder_closure(
            surface, layer=layers[0], expert=0, projection="gate_proj", device=device
        )
    )

    backend = {
        "schema": STREAM_BACKEND_SCHEMA,
        "architecture": RELEASED_ARCHITECTURE,
        "model_revision": model_revision,
        "inventory_sha256": inventory["inventory_sha256"] if inventory is not None else None,
        "checkpoint_identity_sha256": checkpoint_identity,
        "contract_sha256": surface.contract_sha256 if surface is not None else None,
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "packed_reader_abi_sha256": (
            surface.packed_reader_abi_sha256 if surface is not None else None
        ),
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.platform()
        ),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "numeric_policy": numeric,
        "attention_backend": args.attention_backend,
        "experts_implementation": build["experts_implementation"],
        "grouped_mm_kernel": probe_grouped_mm_kernel(device),
        "parallelism": disclosure["ep_semantics"],
        "reader_mode": identity["mode"],
        "final_tp2_serving_kernel": False,
        "main_routed_runtime_dtype": (
            "bfloat16 read straight from the official checkpoint shards (no decode)"
            if args.source == "native"
            else (
                "bfloat16 streamed-decoded from stock exl3 %s payloads (per-module bits)"
                % exl3hf.codebook
            )
            if args.source == "exl3hf"
            else "bfloat16 streamed-decoded from MLX affine u32 payload (fp32 dequant, "
                 "one rounding)"
            if args.source == "mlx"
            else "bfloat16 streamed-dequantized from the GGUF's fused expert tensors"
            if args.source == "gguf"
            else "bfloat16 streamed-decoded from NVFP4 e2m1 gs16 (exact fp32 decode, one rounding)"
            if args.source == "nvfp4"
            else f"bfloat16 streamed-decoded from TR3-published EXL3/MCG K{bits} payload"
            if args.source == "tr3"
            else f"bfloat16 streamed-decoded from packed K{bits} payload"
        ),
        "nonrouted_runtime_dtype": (
            "bfloat16 materialized from the artifact's own tensors (dequantized/cast)"
            if args.source == "exl3hf"
            else "decoded bf16 view of the quant snapshot (passthrough verbatim, quantized "
                 "non-routed fp32-dequantized, one bf16 rounding)"
            if args.source == "mlx"
            else "bfloat16 dequantized from the SAME GGUF artifact (float32 passthrough for the "
                 "tensors the official tree also stores float32); vision copied from official BF16"
            if args.source == "gguf"
            else "quant-snapshot bfloat16 bytes, untouched (official non-routed name set, "
                 "unquantized in the artifact)"
            if args.source == "nvfp4"
            else "the artifact's OWN official-source-native bytes, re-sharded verbatim (no "
                 "decode; routed-experts-only scope, name set verified equal to the official "
                 "non-routed set)"
            if args.source == "tr3"
            else "official source dtype, untouched"
        ),
        "mtp_standard_logits_executed": False,
        "mtp_pack_receipt_sha256": (
            surface.mtp_pack_receipt_sha256 if surface is not None else None
        ),
        "native_routed_layout": native_layout,
        "mlx_routed_layout": mlx_layout,
        "mlx_nonrouted_passthrough_crosscheck": mlx_bf16_crosscheck,
        "gguf_routed_layout": gguf_layout,
        "stored_encoder_closure": closure,
        "model_build": build,
        "streaming_wiring": wiring,
        "memory_plan": memory,
        "declared_tp_plan_not_applied": getattr(model, "_tp_plan", None),
        "declared_ep_plan_not_applied": getattr(model, "_ep_plan", None),
        "streaming_disclosure": disclosure,
        "load_seconds": time.monotonic() - load_started,
    }
    # Lane-only identity (see STREAM_LANE_IDENTITY_SCHEMA above): exactly the
    # fields that name the lane, none that name the artifact.  Two quants
    # measured on the same machine/config hash IDENTICALLY here even though
    # their backend_identity_sha256 values differ.
    lane_fields = {
        "schema": STREAM_LANE_IDENTITY_SCHEMA,
        "torch_version": backend["torch_version"],
        "cuda_runtime_version": backend["cuda_runtime_version"],
        "device_name": backend["device_name"],
        "grouped_mm_kernel": backend["grouped_mm_kernel"],
        "numeric_policy": backend["numeric_policy"],
        "attention_backend": backend["attention_backend"],
        "experts_implementation": backend["experts_implementation"],
        "parallelism": backend["parallelism"],
        "ep_emulate": args.ep_emulate,
        "reduce_order": args.reduce_order,
    }
    if args.source == "gguf":
        # WHERE the ggml block dequant ran.  Recorded because it is a measurable
        # difference in COST (39 ms/matrix on cpu while the GPU sat at 2-4%) and
        # a claimed non-difference in VALUE -- the two paths produce
        # bitwise-identical tensors.  A field a reader can check beats a
        # property a document asserts, and a run that took the fast path should
        # say so on its own receipt rather than in a commit message.
        #
        # Added under the source test rather than as an unconditional key so
        # that every OTHER lane's backend dict -- and therefore its
        # backend_identity_sha256 -- is byte-identical to what it was before
        # this flag existed.  A new field is a new hash, and a hash that moved
        # for a lane that did not is a false signal waiting to be investigated.
        backend["gguf_decode_device"] = (
            str(gguf_decode_device) if gguf_decode_device is not None else "cpu"
        )
    backend["lane_identity"] = lane_fields
    backend["lane_identity_sha256"] = sha256_bytes(canonical_json(lane_fields))

    # ---- capture ----------------------------------------------------------
    logit_records: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    sweep_specs: List[Tuple[int, str]] = []
    if args.sweep:
        for item in args.sweep.split(","):
            item = item.strip()
            if not item:
                continue
            head, _, tail = item.partition(":")
            if not head.startswith("ep") or not tail:
                raise _fail(f"--sweep entry {item!r} must look like ep8:sequential or ep1:none")
            sweep_specs.append((int(head[2:]), tail))
        for sweep_ep, sweep_order in sweep_specs:
            if sweep_ep != 1 and sweep_order not in REDUCE_ORDERS:
                raise _fail(f"--sweep reduce order {sweep_order!r} is not one of {REDUCE_ORDERS}")
            if 288 % sweep_ep:
                raise _fail(f"--sweep ep {sweep_ep} does not divide 288")
            if memory["slab_experts"] not in (288, 288 // sweep_ep):
                raise _fail(
                    f"--sweep needs --slab-experts 288 to serve ep={sweep_ep} "
                    f"(have {memory['slab_experts']})"
                )
    if args.source == "mlx":
        # DERIVED from the artifact, not from the profile: an MLX repo's bit mix is
        # part of what is being measured, so the label carries it (and the KLD report
        # gates the family prefix + cross-run equality instead of one fixed string).
        student_label = mlx_surface_obj.student_label()
    elif args.source == "exl3hf":
        student_label = EXL3HF_PROFILES[args.profile][1]
    elif args.source == "tr3":
        student_label = TR3_PROFILES[args.profile][1]
    elif args.source == "dione":
        student_label = DIONE_PROFILES[args.profile][1]
    elif args.source == "nvfp4":
        student_label = nvfp4_module.NVFP4_STUDENT_LABEL
    else:
        student_label = {
            "native": NATIVE_STUDENT_LABEL,
            "gguf": GGUF_STUDENT_LABEL,
        }.get(args.source, f"uniform-k{bits}")
    capture_role = {
        "native": NATIVE_CAPTURE_ROLE,
        "gguf": GGUF_CAPTURE_ROLE,
    }.get(args.source, "packed_student")
    if args.capture_role == "teacher":
        # the label stays native-bf16 (it IS the native forward); only the ROLE
        # flips, which is exactly what k6_kld_report's teacher discovery keys on
        capture_role = TEACHER_CAPTURE_ROLE
    capture_started = time.monotonic()
    forward_seconds = 0.0
    stored_positions_total = 0
    # The OUTER meter: how far through the panel, and when this capture ends.
    # A fill meter answers "is it alive"; this one answers "will it finish
    # inside --max-runtime", which is the question that decided the first GGUF
    # run.  `every=1` because a window is minutes long, so every completed one
    # is worth a line even when the 30 s throttle has not elapsed.
    window_meter = progress_meter.Progress(
        len(selection), label="windows",
        interval=args.progress_seconds, every=1,
    )
    for index, window in selection:
        tokens = np.load(window.token_path, allow_pickle=False)
        mask = np.load(window.attention_mask_path, allow_pickle=False)
        causal_mask = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(device)
        attention_mask = torch.from_numpy(np.asarray(mask, dtype=np.int64)).unsqueeze(0).to(device)
        window_started = time.monotonic()
        with torch.inference_mode():
            output_logits = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[:, :-1, :]
        selected = output_logits[0, torch.from_numpy(causal_mask).to(device)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds += time.monotonic() - window_started
        if selected.shape != (window.prediction_positions, int(text_config["vocab_size"])):
            raise _fail("streamed student logits differ from sealed panel geometry")
        stored = selected.float().cpu().contiguous()
        logit_path = (output_root / "logits" / f"window-{index:04d}.safetensors").resolve()
        save_tensors = {"logits": stored}
        save_metadata = {
            "capture_role": capture_role,
            "student_label": student_label,
            "cold_run": str(args.cold_run),
            "window_id": window.window_id,
            "token_ids_sha256": window.token_sha256,
            "attention_mask_sha256": window.attention_mask_sha256,
            "checkpoint_identity_sha256": checkpoint_identity,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "streaming_mode": args.stream_mode,
            "ep_emulate": str(args.ep_emulate),
            "reduce_order": args.reduce_order,
        }
        if preview_positions is not None:
            # PREVIEW: keep only the sampled rows.  Note the compute has
            # already happened -- position sampling is a storage/bandwidth
            # knob, never a compute knob (the causal trunk needed every
            # prefix token regardless).
            indices = preview_position_indices(
                args.sample_seed, window.window_id,
                int(window.prediction_positions), preview_positions,
            )
            index_tensor = torch.tensor(indices, dtype=torch.int64)
            save_tensors = {
                "logits": stored.index_select(0, index_tensor).contiguous(),
                "position_indices": index_tensor,
            }
            save_metadata["store_positions"] = args.store_positions
            save_metadata["sample_seed"] = str(args.sample_seed)
            stored_positions_total += len(indices)
        else:
            stored_positions_total += int(window.prediction_positions)
        save_file(
            save_tensors,
            logit_path,
            metadata=save_metadata,
        )
        # ---- combine-order sweep (L5): the ONLY residual between this run and
        # the sealed EP8 one is the order in which the 8 bf16 per-rank partials
        # are summed, which NCCL chooses by topology.  Re-running the forward
        # under each candidate order on the SAME decoded weights measures that
        # sensitivity directly instead of asserting it.
        for spec in sweep_specs:
            sweep_ep, sweep_order = spec
            runtime_ep["ep"], runtime_ep["order"] = sweep_ep, sweep_order
            sweep_started = time.monotonic()
            with torch.inference_mode():
                sweep_logits = model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits[:, :-1, :]
            sweep_selected = sweep_logits[0, torch.from_numpy(causal_mask).to(device)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tag = f"ep{sweep_ep}-{sweep_order.replace(':', '')}"
            sweep_dir = output_root / "sweep" / tag
            sweep_dir.mkdir(parents=True, exist_ok=True)
            sweep_path = (sweep_dir / f"window-{index:04d}.safetensors").resolve()
            save_file({"logits": sweep_selected.float().cpu().contiguous()}, sweep_path,
                      metadata={"window_id": window.window_id, "ep_emulate": str(sweep_ep),
                                "reduce_order": sweep_order})
            delta = (sweep_selected.float() - selected.float())
            sweep_rows.append(
                {
                    "window_id": window.window_id,
                    "ep_emulate": sweep_ep,
                    "reduce_order": sweep_order,
                    "path": str(sweep_path),
                    "max_abs_vs_primary": float(delta.abs().max()),
                    "rms_vs_primary": float(delta.double().pow(2).mean().sqrt()),
                    "argmax_agreement_vs_primary": float(
                        (sweep_selected.argmax(-1) == selected.argmax(-1)).double().mean()
                    ),
                    "seconds": round(time.monotonic() - sweep_started, 2),
                }
            )
            print(json.dumps(sweep_rows[-1], sort_keys=True), flush=True)
            del sweep_logits, sweep_selected, delta
        runtime_ep["ep"], runtime_ep["order"] = args.ep_emulate, args.reduce_order

        logit_records.append(
            {
                "window_id": window.window_id,
                "document_id": window.document_id,
                "domain": window.domain,
                "role": window.role,
                "token_ids_sha256": window.token_sha256,
                "attention_mask_sha256": window.attention_mask_sha256,
                "prediction_positions": window.prediction_positions,
                "path": str(logit_path),
                "bytes": logit_path.stat().st_size,
                "sha256": sha256_file(logit_path),
            }
        )
        print(
            json.dumps(
                {
                    "window": window.window_id,
                    "index": index,
                    "forward_seconds": round(time.monotonic() - window_started, 3),
                    "decode_seconds_cumulative": round(streamer.decode_seconds, 1),
                    "decoded_matrices": streamer.decoded_matrices,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        window_meter.update(1)
        del ids, attention_mask, output_logits, selected, stored
    window_meter.close(suffix="capture complete")

    expected_matrices = len(layers) * 288 * 3
    census_matrices = len(streamer.census) * 3
    backend.update(
        {
            "decode_seconds": streamer.decode_seconds,
            "decoded_matrix_count": streamer.decoded_matrices,
            "distinct_matrix_census": census_matrices,
            "census_closes_main_routed_surface": census_matrices == expected_matrices,
            # "verified" means hash-gated by the sealed payload store.  Only the
            # sealed packed lanes have such bytes.  The native lane has no
            # payload at all, and the mlx / gguf / nvfp4 lanes are unsealed
            # artifacts with no per-payload digest to gate on -- their bytes are
            # sha256-RECORDED in the installed-choice census, never
            # seal-CHECKED.  The artifact bytes each of them actually read are
            # reported separately, unqualified.
            "verified_packed_payload_bytes": (
                # "verified" here means hash-gated PER PAYLOAD by a sealed
                # payload store.  A TR3 release is sealed at the SHARD level
                # (its receipt's shard_sha256 map, cross-checked against the
                # published SHA256SUMS the fetch verified byte-wise), which is
                # a real seal but not this field's claim -- so tr3 reports its
                # bytes under tr3_payload_bytes_read like the other
                # artifact-shard readers.
                None if args.source in ("native", "mlx", "gguf", "nvfp4", "tr3")
                else streamer.payload_bytes
            ),
            "tr3_payload_bytes_read": (
                int(tr3_reader.bytes_read) if tr3_reader is not None else None
            ),
            "tr3_shards_read": (
                len(tr3_reader.shards_read) if tr3_reader is not None else None
            ),
            "mlx_artifact_bytes_read": (
                int(mlx_expert_source.bytes_read) if mlx_expert_source is not None else None
            ),
            "mlx_shards_read": (
                len(mlx_expert_source.shards_read) if mlx_expert_source is not None else None
            ),
            "native_checkpoint_bytes_read": (
                int(native_source.bytes_read) if native_source is not None else None
            ),
            "native_shards_read": (
                len(native_source.shards_read) if native_source is not None else None
            ),
            "gguf_bytes_read": (
                int(gguf_source.bytes_read) if gguf_source is not None else None
            ),
            "gguf_files_read": (
                sorted(gguf_source.files_read) if gguf_source is not None else None
            ),
            "nvfp4_payload_bytes_read": (
                int(nvfp4_expert_source.bytes_read)
                if nvfp4_expert_source is not None
                else None
            ),
            "nvfp4_shards_read": (
                len(nvfp4_expert_source.shards_read)
                if nvfp4_expert_source is not None
                else None
            ),
            # NULL, not the digest of nothing.  Layers served from the decode cache
            # never reach the census recorder, so a warm cache produced an EMPTY census
            # whose digest is the fixed constant 37517e5f3dc66819... -- identical for
            # every warm run of every artifact, and indistinguishable in the receipt
            # from a real census.  Say "partial" out loud instead.
            "installed_choice_census_sha256": (
                sha256_bytes(canonical_json(streamer.census))
                if not streamer.cache_served_layers else None
            ),
            "installed_choice_census_complete": not streamer.cache_served_layers,
            "installed_choice_census_layers_served_from_cache": sorted(
                streamer.cache_served_layers),
            "decode_cache_hits": streamer.cache_hits,
            "decode_cache_fills": streamer.cache_fills,
            "decode_cache_refusals_over_budget": streamer.cache_refusals,
            "decode_cache_budget_bytes": streamer.ram_cache_budget,
            "decode_cache_resident_bytes": streamer.ram_cache_bytes,
            "expert_forward_seconds": stats["expert_forward_seconds"],
            "forward_seconds": forward_seconds,
            "combine_order_sweep": sweep_rows,
            "peak_device_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "peak_device_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        }
    )
    if census_matrices != expected_matrices and not args.windows:
        raise _fail(
            f"streamed census does not close the main routed surface "
            f"({census_matrices} of {expected_matrices})"
        )
    backend["backend_identity_sha256"] = sha256_bytes(canonical_json(backend))
    write_json(output_root / "backend.json", backend)

    receipt = {
        "schema": CAPTURE_SCHEMA,
        "capture_role": capture_role,
        "cold_run": args.cold_run,
        "model_revision": model_revision,
        "checkpoint_identity_sha256": checkpoint_identity,
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
        "backend_identity_sha256": backend["backend_identity_sha256"],
        "weight_dtype": (
            "official BF16 checkpoint routed experts, streamed, NO decode"
            if args.source == "native"
            else (
                "stock EXL3 (%s codebook, per-module bits) streamed offline-decoded to BF16"
                % exl3hf.codebook
            )
            if args.source == "exl3hf"
            else (
                "MLX affine u32-packed weights streamed offline-dequantized to BF16 "
                f"({mlx_surface_obj.student_label() if mlx_surface_obj is not None else ''}); "
                "the artifact also quantizes non-routed tensors, decoded into the "
                "materialized view this forward was built from (see scope_policy)"
            )
            if args.source == "mlx"
            else (
                "llama.cpp GGUF block quants (%s) offline-dequantized to BF16 -- ALL tensors, "
                "non-routed included" % ",".join(
                    sorted(k for k in gguf_surface_obj.type_census
                           if k not in ("F32", "F16", "BF16")))
            )
            if args.source == "gguf"
            else (
                "NVFP4 e2m1 gs16 routed experts streamed exact-fp32-decoded to BF16 "
                "(non-routed BF16 from the artifact itself)"
            )
            if args.source == "nvfp4"
            else (
                "TR3-published EXL3/MCG uniform-k%d routed experts streamed offline-decoded "
                "to BF16 (routed-experts-only scope: the artifact's own official-native "
                "non-routed tensors, native BF16 head, serve the rest); the release's "
                "published seal was recomputed before decoding" % bits
            )
            if args.source == "tr3"
            else f"EXL3/TR3 uniform-k{bits} streamed offline-decoded to BF16"
        ),
        "logits_dtype": "float32",
        "kld_direction": "teacher_to_student",
        "prediction_positions": sum(window.prediction_positions for _, window in selection),
        "vocab_size": int(text_config["vocab_size"]),
        "student_label": student_label,
        "logit_files": logit_records,
        "elapsed_seconds": time.monotonic() - capture_started,
        "streaming_disclosure": disclosure,
    }
    # Every block below is added BEFORE receipt_sha256 is computed, so the seal
    # covers it.  None of them appears in a default invocation, which keeps
    # default receipts field-identical to the sealed layout (asserted by
    # stream_score_selftest rung L1.j).
    if args.source == "exl3hf":
        # The summary receipt (k6_kld_report) republishes these pins; they are
        # sealed here first so the headline number's provenance chain starts in
        # the capture itself.  Default receipts are field-identical to the
        # sealed layout (L1.j) - these keys exist only on exl3hf runs.
        receipt["exl3hf_repo"] = args.exl3hf_repo
        receipt["exl3hf_revision"] = args.exl3hf_revision
        receipt["artifact_config_sha256"] = exl3hf.config_sha256
        receipt["artifact_index_sha256"] = exl3hf.index_sha256
        receipt["codebook"] = exl3hf.codebook
        receipt["exllamav3_version"] = exl3hf.exllamav3_version
        receipt["declared_bits"] = exl3hf.declared_bits
        receipt["declared_head_bits"] = exl3hf.declared_head_bits
        receipt["materialization_receipt_sha256"] = materialization["receipt_sha256"]
        receipt["seal_disclosure"] = xs3.SEAL_DISCLOSURE
        receipt["routed_bits_decode_histogram"] = dict(
            sorted(exl3hf.routed_bits_histogram.items())
        )
    if args.source == "dione":
        # Same rule as exl3hf and tr3: k6_kld_report republishes these pins in
        # the headline summary, so the capture seals them first.  A Dione
        # release publishes NO seal, so what travels is the release's own
        # declarations plus the digests THIS run verified: the per-shard hash
        # verdict, the local config/index digests and the materialization
        # receipt that binds the non-routed tree to this same snapshot.
        receipt["dione_repo"] = args.dione_repo
        receipt["dione_revision"] = args.dione_revision
        receipt["artifact_config_sha256"] = dione.config_sha256
        receipt["artifact_index_sha256"] = dione.index_sha256
        receipt["codebook"] = "mcg"
        receipt["codec_family"] = "exl3-mcg"
        receipt["declared_bits"] = float(dione.bits)
        receipt["declared_head_bits"] = 16
        receipt["tp_size"] = dione.tp_size
        receipt["source_repo"] = dione.source_repo
        receipt["source_revision"] = dione.source_revision
        receipt["exl3_manifest_name"] = dione.exl3_manifest_name
        receipt["exl3_manifest_sha256"] = dione.exl3_manifest_sha256
        receipt["exl3_manifest_schema"] = dione.exl3_manifest_schema
        receipt["dione_shard_hash_verification"] = dione.shard_hash_verification
        receipt["materialization_receipt_sha256"] = materialization["receipt_sha256"]
        receipt["scope_digest"] = dione_module.scope_digest(dione)
        receipt["scope_census_sha256"] = \
            dione_module.surface_summary(dione)["scope_census_sha256"]
        receipt["seal_disclosure"] = dione_module.SEAL_DISCLOSURE
    if args.source == "tr3":
        # Same rule as exl3hf: the summary receipt republishes these, so they are
        # sealed in the capture first.  What is different is that a TR3 release
        # can pin its own seal -- so the receipt carries the verification block,
        # not just the artifact's claims.
        receipt["tr3_repo"] = args.tr3_repo
        receipt["tr3_revision"] = args.tr3_revision
        receipt["artifact_config_sha256"] = tr3.config_sha256
        receipt["artifact_index_sha256"] = tr3.index_sha256
        receipt["codebook"] = tr3.codebook
        receipt["codec_family"] = "exl3-mcg"
        receipt["exllamav3_version"] = tr3.quantizer_version
        receipt["exllamav3_pin"] = tr3.exllamav3_pin
        receipt["declared_bits"] = tr3.declared_bits
        receipt["declared_head_bits"] = tr3.declared_head_bits
        receipt["scope_policy"] = tr3.scope_policy
        receipt["nonrouted_policy_declared"] = tr3.nonrouted_policy
        receipt["artifact_materialization_receipt_sha256"] = \
            tr3.seal["materialization"]["receipt_sha256"]
        receipt["materialization_receipt_sha256"] = materialization["receipt_sha256"]
        receipt["seal_verification"] = tr3.seal
        receipt["shard_verification"] = tr3.shard_verification
        receipt["scope_census_sha256"] = tr3.scope_census_sha256()
        receipt["seal_disclosure"] = tr3_module.SEAL_DISCLOSURE
        receipt["routed_bits_decode_histogram"] = dict(
            sorted(exl3hf.routed_bits_histogram.items())
        )
    if args.source == "mlx":
        # The provenance a community artifact CAN carry, at the top level where
        # k6_kld_report lifts it into the headline summary (mirrors the Dione
        # capture receipt's dione_* block).  There is no contract and no payload
        # digest to record here, which is exactly what seal_disclosure says.
        receipt.update(
            {
                "mlx_repo": mlx_surface_obj.repo,
                "mlx_revision": mlx_surface_obj.revision,
                "mlx_format": mlxs.MLX_FORMAT,
                "mlx_default_bits": mlx_surface_obj.default_bits,
                "mlx_default_group_size": mlx_surface_obj.default_group_size,
                "mlx_bits_histogram": dict(mlx_surface_obj.census["bits_histogram"]),
                "mlx_config_sha256": mlx_surface_obj.config_sha256,
                "mlx_index_sha256": mlx_surface_obj.index_sha256,
                "mlx_shard_hash_verification": mlx_surface_obj.shard_hash_verification,
                "mlx_scope_policy": mlx_surface_obj.scope_policy(),
                "mlx_nonrouted_view": {
                    key: nonrouted_view[1].get(key)
                    for key in (
                        "nonrouted_tensor_count",
                        "decoded_module_count",
                        "passthrough_tensor_count",
                        "shards_written",
                        "total_bytes",
                        "config_quantization_block_stripped",
                    )
                },
                "mlx_nonrouted_passthrough_crosscheck": mlx_bf16_crosscheck,
                "mlx_fetch_ledger": mlx_surface_obj.fetch_ledger(),
                "source_repo": mlx_surface_obj.official_source_repo,
                "source_revision": mlx_surface_obj.official_source_revision,
                "official_shape_census_sha256": mlx_surface_obj.official_census_sha256,
                "seal_disclosure": mlxs.SEAL_DISCLOSURE,
            }
        )
    if args.source == "gguf":
        # What the registry row has to disclose about a third-party artifact
        # that quantized EVERYTHING: which repo at which immutable revision,
        # which files (hashed), what it actually quantized, and that nothing
        # here was sealed by an upstream encoder.
        receipt["gguf_repo"] = gguf_surface_obj.repo
        receipt["gguf_revision"] = gguf_surface_obj.revision
        receipt["gguf_files"] = [dict(row) for row in gguf_surface_obj.file_records]
        receipt["gguf_file_hash_verification"] = gguf_surface_obj.file_hash_verification
        receipt["gguf_architecture"] = gguf_surface_obj.architecture
        receipt["gguf_type_census"] = dict(gguf_surface_obj.type_census)
        receipt["gguf_quant_metadata"] = dict(gguf_surface_obj.quant_metadata)
        receipt["scope_policy"] = gguf_surface_obj.scope_policy
        receipt["source_repo"] = "zai-org/GLM-5.3-Flash-BF16"
        receipt["source_revision"] = model_revision
        receipt["seal_disclosure"] = gguf_module.SEAL_DISCLOSURE
    if args.capture_role == "teacher":
        receipt["teacher_provenance"] = {
            "schema": TEACHER_PROVENANCE_SCHEMA,
            "teacher_label": TEACHER_LABEL,
            "lane": "streaming-single-device",
            "source": "native-bf16",
            "ep_emulate": args.ep_emulate,
            "reduce_order": args.reduce_order,
            "stream_mode": args.stream_mode,
            "grouped_mm_kernel": backend.get("grouped_mm_kernel"),
            "device_name": backend.get("device_name"),
            "torch_version": backend.get("torch_version"),
            "transformers_version": backend.get("transformers_version"),
            "cold_run": args.cold_run,
        }
    if preview_positions is not None:
        receipt["schema"] = PREVIEW_CAPTURE_SCHEMA
        receipt["not_submittable"] = True
        receipt["sampling_design"] = {
            "scheme": "stratified-systematic",
            "windows_used": len(logit_records),
            "windows_total": len(all_windows),
            "positions_per_window": preview_positions,
            "total_positions": stored_positions_total,
            "seed": args.sample_seed,
            "fpc_applied": True,
            "note": "position sampling saves logit storage/teacher bandwidth "
                    "only; the trunk forward still ran all positions of every "
                    "selected window (causality). Score with bin/kld-preview.",
        }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    write_json(output_root / "capture-receipt.json", receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "receipt_sha256": receipt["receipt_sha256"],
                "cold_run": args.cold_run,
                "windows": len(logit_records),
                "decode_seconds": round(streamer.decode_seconds, 1),
                "forward_seconds": round(forward_seconds, 1),
                "peak_device_gb": (
                    round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
                    if device.type == "cuda"
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
