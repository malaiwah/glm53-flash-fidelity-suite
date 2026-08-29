#!/usr/bin/env python3
"""Stock-exllamav3 HF-sharded checkpoint surface adapter ("exl3hf") for the streaming scorer.

Scores stock `exllamav3` releases (turboderp/GLM-5.3-Flash-exl3 and format-identical
repos) on OUR sealed 25-window panel with the SAME single-device EP8-emulated
streaming capture used for K6/K8/the BF16 floor, so the number ranks directly
against the streaming lane's own rows.

Format (verified against the live repo @ 2a30229e / 332ab457, header-level and
payload-level, 2026-08-29):

  * Canonical HF shard layout (model.safetensors.index.json), OFFICIAL tensor
    names (model.language_model.*, model.visual.*, lm_head).  Every quantized
    module M stores  M.{trellis,suh,svh,<codebook>}  where <codebook> is
    `mul1` (exllamav3 >= 1.4 default) or `mcg`; the codebook tensor is an I32
    scalar MARKER equal to the codebook's own multiplier constant.  Native
    tensors (embeddings, norms, gates, conv1d, hc_*, A_log, dt_bias, biases)
    ship as plain tensors.
  * FULL-scope quantization: routed experts AND attention AND shared experts
    AND dense MLPs AND the vision tower AND lm_head are EXL3 (embed_tokens
    stays BF16).  Bit-rate varies per module class (trellis.shape[-1]//16).
  * Two module families are stored FUSED relative to the official BF16 tree
    that transformers 5.16.1 loads (the zai-org/GLM-5.3-Flash-BF16 layout the
    whole lane is proven on):
      - KDA layers: self_attn.qkv_proj (one EXL3 matrix, rows q|k|v) and
        self_attn.conv1d (native, rows q|k|v) versus official split
        q/k/v_proj + q/k/v_conv1d.  Split order PROVEN: the fused conv1d's
        three row-slices are BITWISE equal to the official q/k/v_conv1d.
      - vision tower: attn.{q,k,v}_proj SPLIT versus official fused attn.qkv.
    The MTP layer 45 lives in a separate `mtp.safetensors` under its official
    names (its routed experts are never executed by standard-logits scoring).
  * Decode: verbatim campaign math (glm53_packed_k4_reader.decode_choice_hf
    over the anybits unpack) with the codebook LUT swapped for exllamav3
    v1.4.4's `mul1` codebook (codebook.cuh cb==2):
        x = idx * 0x83DCD12D  (u32 wrap)
        s = u16(0x6400 + bytesum(x))              # dp4a(x, 0x01010101, 0x6400)
        v = hfma(f16_bits(s), f16_bits(0x1eee), f16_bits(0xc931))
    The fused half-precision FMA is emulated EXACTLY: both operands are exact
    in fp32, the product needs <= 22 significand bits and the sum <= 24, so
    one fp32 round-to-fp16 equals the fused round (selftest asserts this
    against an independent fp64 computation).

DISCLOSED DEVIATION - unsealed-source scoring: stock exllamav3 releases ship
no per-choice receipts, no reconstruction closures and no sealed reader ABI.
This adapter decodes WITHOUT seal verification: it records the sha256 of every
payload it consumed and the immutable repo revision, exactly like the Dione
adapter.  Every receipt carries ``seal_disclosure`` saying so.

Validation evidence (real payloads vs the official BF16 weights, 2026-08-29,
exl3hf-evidence/local-decode-audit.json): KDA qkv slice-vs-official cosine
0.99981 (K6) with cross-slice cosine 0.0031; fused conv1d slices bitwise
equal to official q/k/v_conv1d; visual fused-qkv cosine 0.99984; routed
expert K4 cosine 0.99709, rel-L2 0.0762 (the expected 4bpw reconstruction
error, cf. the Dione audit's 0.083-0.089).  All cosines fp64.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

EXL3HF_SURFACE_SCHEMA = "malaiwah.glm53-exl3hf-surface.v1"
EXL3HF_IDENTITY_SCHEMA = "malaiwah.glm53-exl3hf-student-identity.v1"
EXL3HF_READER_IDENTITY_SCHEMA = "malaiwah.glm53-exl3hf-offline-reader-identity.v1"
EXL3HF_MATERIALIZATION_SCHEMA = "malaiwah.glm53-exl3hf-nonrouted-materialization.v1"
RELEASE_INVENTORY_SCHEMA = "quant-pipeline.glm-release-inventory.v1"
SEAL_DISCLOSURE = (
    "unsealed-source scoring: stock exllamav3 releases ship no upstream receipts, "
    "reconstruction closures or sealed reader ABI; the packed surface was decoded "
    "WITHOUT seal verification (consumed payload sha256s and the immutable repo "
    "revision are recorded instead)"
)

MAIN_ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYER = 45
NUM_EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PROJECTION_SHAPE = {
    "gate_proj": (2048, 4096),
    "up_proj": (2048, 4096),
    "down_proj": (4096, 2048),
}

# Codebook constants, verbatim from exllamav3 v1.4.4 exllamav3_ext/quant/codebook.cuh.
MUL1_MULT = 0x83DCD12D
MUL1_MARKER_SIGNED_INT32 = -2082680531  # int32 view of 0x83DCD12D
MCG_MULT = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083  # int32 view of 0xCBAC1FED
CODEBOOK_OBJECTS = {"mul1": MUL1_MARKER_SIGNED_INT32, "mcg": MCG_MARKER_SIGNED_INT32}
PAYLOAD_OBJECTS = ("trellis", "suh", "svh")

_ROUTED = re.compile(r"\.mlp\.experts\.(\d+)\.")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _fail(message: str) -> ValueError:
    return ValueError(f"exl3hf_surface: {message}")


def _canonical_json(value: Any) -> bytes:
    """Byte-identical to quant_pipeline.core.artifacts.canonical_json."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _seal(body: Dict[str, Any], sha_field: str) -> Dict[str, Any]:
    sealed = dict(body)
    sealed[sha_field] = _sha256_bytes(_canonical_json(body))
    return sealed


def _tensor_sha256(value) -> str:
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


# --------------------------------------------------------------------------
# codebooks
# --------------------------------------------------------------------------
_LUT_CACHE: Dict[Tuple[str, str], Any] = {}


def mul1_lut(device="cpu"):
    """The exllamav3 v1.4.4 `mul1` codebook as a 65,536-entry fp16 LUT.

    hfma emulation exactness: h is an integer in [1024, 2620] (11 significand
    bits), k_inv has fp16 precision (11 bits): the fp32 product is exact
    (<= 22 bits); adding the fp16-precision bias needs <= 24 aligned bits, so
    the single fp32->fp16 round equals CUDA's fused __hfma round.
    """
    import torch

    key = ("mul1", str(device))
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    idx = np.arange(1 << 16, dtype=np.uint64)
    prod = ((idx * np.uint64(MUL1_MULT)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    bytesum = (
        (prod & np.uint32(0xFF))
        + ((prod >> np.uint32(8)) & np.uint32(0xFF))
        + ((prod >> np.uint32(16)) & np.uint32(0xFF))
        + ((prod >> np.uint32(24)) & np.uint32(0xFF))
    ).astype(np.uint32)
    s = (np.uint32(0x6400) + bytesum).astype(np.uint16)
    h = s.view(np.float16).astype(np.float32)
    k_inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    k_bias = np.array([0xC931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    values = (h * k_inv + k_bias).astype(np.float16)
    if not np.isfinite(values).all():
        raise _fail("mul1 lookup table is non-finite")
    lut = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    _LUT_CACHE[key] = lut
    return lut


def codebook_lut(codebook: str, device="cpu"):
    if codebook == "mul1":
        return mul1_lut(device)
    if codebook == "mcg":
        # the campaign reader's own frozen MCG table (kept as the single source)
        from quant_pipeline.evaluation.glm53_packed_k4_reader import mcg_lut

        return mcg_lut(device)
    raise _fail(f"unknown codebook {codebook!r} (known: mul1, mcg)")


# --------------------------------------------------------------------------
# decode: verbatim campaign math over the anybits unpack, LUT swapped
# --------------------------------------------------------------------------
def unpack_trellis_states_anybits(packed, bits: int):
    """Verbatim glm53_packed_k4_reader.unpack_trellis_states math, rate 1..8."""
    import torch

    if (
        bits not in range(1, 9)
        or packed.dtype != torch.int16
        or packed.ndim != 3
        or packed.shape[-1] != bits * 16
    ):
        raise _fail("anybits unpack expects int16 trellis words of shape [.,.,bits*16]")
    tiles = packed.reshape(-1, bits * 16).to(torch.int64) & 0xFFFF
    tiles = tiles.reshape(tiles.shape[0], -1, 2).flip(-1).reshape(tiles.shape)
    word_shifts = torch.arange(15, -1, -1, device=packed.device)
    bitstream = ((tiles.reshape(-1, 16, bits)[..., None] >> word_shifts) & 1).reshape(
        -1, 16, bits * 16
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=packed.device)
    edges = (bitstream.reshape(-1, 16, 16, bits) << symbol_shifts).sum(dim=-1)
    edges = edges.reshape(-1, 256)
    states = torch.zeros_like(edges)
    for lag in range(math.ceil(16 / bits)):
        states |= torch.roll(edges, shifts=lag, dims=-1) << (lag * bits)
    return (states & 0xFFFF).to(torch.int16).reshape(*packed.shape[:-1], 256).contiguous()


def _permutation(device):
    import torch

    values = [0] * 256
    for thread in range(32):
        rows = ((thread % 4) * 2, (thread % 4) * 2 + 1, (thread % 4) * 2 + 8, (thread % 4) * 2 + 9)
        columns = (thread // 4, thread // 4 + 8)
        for offset, (row, column) in enumerate(
            ((row, column) for column in columns for row in rows)
        ):
            values[thread * 8 + offset] = row * 16 + column
    return torch.tensor(values, dtype=torch.long, device=device)


def _hadamard(device, dtype):
    import torch

    value = torch.ones((1, 1), dtype=dtype, device=device)
    while value.shape[0] < 128:
        value = torch.cat((torch.cat((value, value), 1), torch.cat((value, -value), 1)), 0)
    return value * (1.0 / math.sqrt(128))


def decode_payload_hf(trellis, suh, svh, *, codebook: str, unpack_device=None):
    """Decode one stored payload to official/HF orientation [out_features, in_features].

    Verbatim decode_choice_hf structure (fp32 hadamards, one final rounding by
    the CALLER); only the LUT differs by codebook.  `unpack_device` mirrors the
    stream_score escape for backends whose int64 shifts are broken.
    """
    import torch

    bits = trellis.shape[-1] // 16
    unpack_input = trellis if unpack_device is None else trellis.to(unpack_device)
    states = unpack_trellis_states_anybits(unpack_input, bits)
    if unpack_device is not None:
        states = states.to(trellis.device)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = (
        codebook_lut(codebook, states.device)
        .index_select(0, indices.flatten())
        .reshape_as(states)
        .float()
    )
    values = values.index_select(-1, torch.argsort(_permutation(states.device)))
    k_tiles, n_tiles, _ = values.shape
    exl = (
        values.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
    )
    had = _hadamard(exl.device, exl.dtype)
    exl = torch.matmul(had, exl.reshape(-1, 128, exl.shape[1])).reshape_as(exl)
    exl *= suh.to(device=exl.device, dtype=exl.dtype).reshape(-1, 1)
    exl = torch.matmul(exl.reshape(exl.shape[0], -1, 128), had).reshape_as(exl)
    exl *= svh.to(device=exl.device, dtype=exl.dtype).reshape(1, -1)
    return exl.T.contiguous()


# --------------------------------------------------------------------------
# surface: index/config resolution
# --------------------------------------------------------------------------
@dataclass
class Exl3HfSurface:
    root: Path
    codebook: str
    exllamav3_version: str
    declared_bits: float
    declared_head_bits: Optional[int]
    config_sha256: str
    index_sha256: str
    weight_map: Dict[str, str]
    quantized_modules: Dict[str, int]  # module -> K
    marker_value: int = 0
    routed_bits_histogram: Dict[str, int] = field(default_factory=dict)

    @property
    def total_quantized_modules(self) -> int:
        return len(self.quantized_modules)


def module_groups(weight_map: Mapping[str, str]) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Group index names into (module -> {object: tensor_name}) plus plain natives."""
    grouped: Dict[str, Dict[str, str]] = {}
    natives: List[str] = []
    payload_suffixes = ("trellis", "suh", "svh", "mul1", "mcg")
    for name in weight_map:
        stem, _, last = name.rpartition(".")
        if stem and last in payload_suffixes:
            grouped.setdefault(stem, {})[last] = name
        else:
            natives.append(name)
    return grouped, natives


def load_surface(root: Path) -> Exl3HfSurface:
    root = Path(root).resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise _fail(f"{root} lacks config.json / model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "exl3":
        raise _fail("config.quantization_config.quant_method is not 'exl3'")
    codebook = str(quant.get("codebook") or "3inst")
    if codebook not in CODEBOOK_OBJECTS:
        raise _fail(f"unsupported exl3 codebook {codebook!r} (this reader speaks mul1/mcg)")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = dict(index["weight_map"])
    grouped, _ = module_groups(weight_map)
    quantized: Dict[str, int] = {}
    for module, objects in grouped.items():
        missing = [o for o in PAYLOAD_OBJECTS + (codebook,) if o not in objects]
        if missing:
            raise _fail(f"quantized module {module} lacks {missing}")
        quantized[module] = -1  # K resolved lazily from the trellis shape
    return Exl3HfSurface(
        root=root,
        codebook=codebook,
        exllamav3_version=str(quant.get("version", "unknown")),
        declared_bits=float(quant.get("bits", 0.0)),
        declared_head_bits=quant.get("head_bits"),
        config_sha256=_sha256_file(config_path),
        index_sha256=_sha256_file(index_path),
        weight_map=weight_map,
        quantized_modules=quantized,
        marker_value=CODEBOOK_OBJECTS[codebook],
    )


def routed_module_name(layer: int, expert: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"


def routed_census(surface: Exl3HfSurface, layers: Tuple[int, ...], num_experts: int) -> Dict[str, Any]:
    """Prove, from the index alone, that every routed payload group exists."""
    absent: List[str] = []
    shards: set = set()
    for layer in layers:
        for expert in range(num_experts):
            for projection in PROJECTIONS:
                module = routed_module_name(layer, expert, projection)
                if module not in surface.quantized_modules:
                    absent.append(module)
                    if len(absent) > 4:
                        break
                else:
                    shards.add(surface.weight_map[module + ".trellis"])
    if absent:
        raise _fail(
            f"index is missing {len(absent)}+ routed payload groups (first: {absent[0]})"
        )
    return {
        "routed_module_count": len(layers) * num_experts * len(PROJECTIONS),
        "routed_shard_count": len(shards),
        "layers": [layers[0], layers[-1]],
        "experts_per_layer": num_experts,
        "codebook": surface.codebook,
    }


class Exl3HfShardReader:
    """Thread-local safetensors handles over the checkpoint shards.

    Mirrors NativeCheckpointSource's caching: handles are cached PER THREAD so
    the streamer's IO pool can read in parallel without sharing a handle.
    """

    def __init__(self, surface: Exl3HfSurface):
        self.surface = surface
        self._local = threading.local()
        self._lock = threading.Lock()
        self.shards_read: set = set()
        self.bytes_read = 0

    def _handle(self, shard: str):
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = self._local.handles = {}
        handle = cache.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.surface.root / shard), framework="pt", device="cpu")
            enter = getattr(handle, "__enter__", None)
            if enter is not None:
                handle = enter()
            cache[shard] = handle
        return handle

    def tensor(self, name: str):
        shard = self.surface.weight_map.get(name)
        if shard is None:
            raise _fail(f"index has no tensor named {name}")
        tensor = self._handle(shard).get_tensor(name)
        with self._lock:
            self.shards_read.add(shard)
            self.bytes_read += int(tensor.numel() * tensor.element_size())
        return tensor

    def payload(self, module: str) -> Dict[str, Any]:
        objects = {name: self.tensor(f"{module}.{name}") for name in PAYLOAD_OBJECTS}
        objects["marker"] = self.tensor(f"{module}.{self.surface.codebook}")
        return objects


def verify_marker(surface: Exl3HfSurface, module: str, marker) -> None:
    value = int(marker.reshape(-1)[0])
    if value != surface.marker_value:
        raise _fail(
            f"{surface.codebook} marker differs on {module}: {value} != {surface.marker_value}"
        )


def payload_bits(payload: Mapping[str, Any]) -> int:
    return int(payload["trellis"].shape[-1]) // 16


def decode_module(
    surface: Exl3HfSurface,
    payload: Mapping[str, Any],
    *,
    module: str,
    device,
    expected_shape: Optional[Tuple[int, int]] = None,
    unpack_device=None,
    hash_payload: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Decode one module's payload to an fp32 [out,in] tensor + census row."""
    verify_marker(surface, module, payload["marker"])
    bits = payload_bits(payload)
    moved = {name: payload[name].to(device) for name in PAYLOAD_OBJECTS}
    decoded = decode_payload_hf(
        moved["trellis"], moved["suh"], moved["svh"],
        codebook=surface.codebook, unpack_device=unpack_device,
    )
    if expected_shape is not None and tuple(decoded.shape) != tuple(expected_shape):
        raise _fail(
            f"decoded {module} has shape {tuple(decoded.shape)}, expected {expected_shape}"
        )
    census = {"module": module, "bits": bits}
    if hash_payload:
        census.update(
            {
                "trellis_sha256": _tensor_sha256(payload["trellis"]),
                "suh_sha256": _tensor_sha256(payload["suh"]),
                "svh_sha256": _tensor_sha256(payload["svh"]),
            }
        )
    return decoded, census


def load_decoded_module(
    surface: Exl3HfSurface,
    shards: Exl3HfShardReader,
    *,
    layer: int,
    expert: int,
    projection: str,
    device,
    unpack_device=None,
    hash_payloads: bool = True,
):
    """Routed-expert accessor with the dione-adapter signature: fp32 [out,in]."""
    module = routed_module_name(layer, expert, projection)
    payload = shards.payload(module)
    decoded, census = decode_module(
        surface,
        payload,
        module=module,
        device=device,
        expected_shape=PROJECTION_SHAPE[projection],
        unpack_device=unpack_device,
        hash_payload=hash_payloads,
    )
    bits = census["bits"]
    key = f"K{bits}"
    surface.routed_bits_histogram[key] = surface.routed_bits_histogram.get(key, 0) + 1
    return decoded, census


def reader_identity(runner_path: Path, *, codebook: str, bits_note: str) -> Dict[str, Any]:
    body = {
        "schema": EXL3HF_READER_IDENTITY_SCHEMA,
        "mode": "stock_exl3_hf_shard_offline_decode_for_logit_measurement",
        "serving_kernel": False,
        "final_tp2_kernel": False,
        "codebook": codebook.upper(),
        "bits": bits_note,
        "decode_executed": True,
        "surface_module_sha256": _sha256_file(Path(__file__).resolve()),
        "runtime_reader_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    body["identity_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


# --------------------------------------------------------------------------
# non-routed materialization -> an official-layout BF16 tree
# --------------------------------------------------------------------------
KDA_QKV = re.compile(r"^(model\.language_model\.layers\.\d+\.self_attn)\.qkv_proj$")
KDA_CONV = re.compile(r"^(model\.language_model\.layers\.\d+\.self_attn)\.conv1d\.weight$")
VISUAL_QKV = re.compile(r"^(model\.visual\.blocks\.\d+\.attn)\.(q|k|v)_proj$")


def _materialize_stream(
    surface: Exl3HfSurface,
    reader: Exl3HfShardReader,
    device,
    stats: Dict[str, Any],
    extra_maps: Optional[List[Tuple[Dict[str, Dict[str, str]], List[str], Any]]] = None,
):
    """Yield (official_name, tensor) for every NON-ROUTED tensor, official layout.

    `extra_maps` carries additional (grouped, natives, reader_fn) sources --
    used for mtp.safetensors, whose tensors are not in the index.
    """
    import torch

    def finalize(name, tensor, *, decoded=False):
        # decoded fp32 gets the lane's single fp32->bf16 rounding; NATIVE fp32
        # tensors (A_log, dt_bias, e_score_correction_bias) stay fp32 verbatim.
        if tensor.dtype == torch.float32 and not decoded:
            return name, tensor.contiguous()
        return name, tensor.to(torch.bfloat16).contiguous()

    grouped, natives = module_groups(surface.weight_map)
    sources = [(grouped, natives, reader.tensor)]
    for extra in extra_maps or []:
        sources.append(extra)

    for grouped_i, natives_i, read in sources:
        native_set = set(natives_i)
        # 1) plain native tensors (skip routed experts entirely; skip biases
        #    that belong to a quantized module -- the quantized pass emits them
        #    next to their dequantized weight, under the official fused name
        #    where fusion applies)
        for name in natives_i:
            if _ROUTED.search(name):
                continue
            stem, _, last = name.rpartition(".")
            if last == "bias" and stem in grouped_i:
                continue
            m = KDA_CONV.match(name)
            if m:
                fused = read(name)
                if fused.shape[0] % 3:
                    raise _fail(f"{name} rows not divisible by 3")
                d = fused.shape[0] // 3
                for i, part in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
                    yield finalize(f"{m.group(1)}.{part}.weight", fused[i * d:(i + 1) * d])
                stats["kda_conv_split"] = stats.get("kda_conv_split", 0) + 1
                continue
            yield finalize(name, read(name))
            stats["native_copied"] = stats.get("native_copied", 0) + 1

        # 2) quantized modules
        visual_parts: Dict[str, Dict[str, Any]] = {}
        for module in sorted(grouped_i):
            if _ROUTED.search(module):
                continue
            payload = {o: read(f"{module}.{o}") for o in PAYLOAD_OBJECTS}
            payload["marker"] = read(f"{module}.{surface.codebook}")
            decoded, census = decode_module(
                surface, payload, module=module, device=device, hash_payload=False
            )
            stats.setdefault("dequantized_bits", {})
            key = f"K{census['bits']}"
            stats["dequantized_bits"][key] = stats["dequantized_bits"].get(key, 0) + 1
            bias_name = f"{module}.bias"
            bias = read(bias_name) if bias_name in native_set else None

            m = KDA_QKV.match(module)
            if m:
                if decoded.shape[0] % 3:
                    raise _fail(f"{module} out-features not divisible by 3")
                d = decoded.shape[0] // 3
                for i, part in enumerate(("q_proj", "k_proj", "v_proj")):
                    yield finalize(f"{m.group(1)}.{part}.weight", decoded[i * d:(i + 1) * d].cpu(), decoded=True)
                stats["kda_qkv_split"] = stats.get("kda_qkv_split", 0) + 1
                continue
            m = VISUAL_QKV.match(module)
            if m:
                slot = visual_parts.setdefault(m.group(1), {})
                slot[m.group(2)] = (decoded.cpu(), bias)
                if len(slot) == 3:
                    fused_w = torch.cat([slot[p][0] for p in ("q", "k", "v")], dim=0)
                    yield finalize(f"{m.group(1)}.qkv.weight", fused_w, decoded=True)
                    if all(slot[p][1] is not None for p in ("q", "k", "v")):
                        fused_b = torch.cat([slot[p][1].float() for p in ("q", "k", "v")])
                        yield finalize(f"{m.group(1)}.qkv.bias", fused_b, decoded=True)
                    visual_parts.pop(m.group(1))
                    stats["visual_qkv_fused"] = stats.get("visual_qkv_fused", 0) + 1
                continue
            yield finalize(f"{module}.weight", decoded.cpu(), decoded=True)
            if bias is not None:
                yield finalize(f"{module}.bias", bias)
        if visual_parts:
            raise _fail(f"visual qkv fusion left incomplete groups: {sorted(visual_parts)[:4]}")


def materialize_nonrouted(
    root: Path,
    out_dir: Path,
    *,
    device: str = "cpu",
    source_repo: str,
    source_revision: str,
    official_index: Optional[Path] = None,
    shard_bytes: int = 8 << 30,
) -> Dict[str, Any]:
    """Materialize the artifact's NON-ROUTED function as an official-layout BF16 tree.

    The routed experts (layers 3..44 and the never-executed MTP layer 45) get
    VIRTUAL index entries pointing at a shard that does not exist; the
    streaming engine's non-routed view filters them before anything reads
    shards, and their presence is what lets that filter prove it dropped the
    routed surface.
    """
    import torch
    from safetensors.torch import save_file

    started = time.monotonic()
    surface = load_surface(root)
    reader = Exl3HfShardReader(surface)
    if not _REVISION.fullmatch(source_revision):
        raise _fail("--source-revision must be the immutable 40-hex commit")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    extra_maps = []
    mtp_path = Path(root) / "mtp.safetensors"
    mtp_handle = None
    if mtp_path.is_file():
        from safetensors import safe_open

        mtp_handle = safe_open(str(mtp_path), framework="pt", device="cpu")
        enter = getattr(mtp_handle, "__enter__", None)
        if enter is not None:
            mtp_handle = enter()
        mtp_names = list(mtp_handle.keys())
        mtp_grouped, mtp_natives = module_groups({n: "mtp.safetensors" for n in mtp_names})
        extra_maps.append((mtp_grouped, mtp_natives, mtp_handle.get_tensor))

    stats: Dict[str, Any] = {}
    shard_index = 0
    current: Dict[str, Any] = {}
    current_bytes = 0
    weight_map: Dict[str, str] = {}
    shard_files: List[str] = []
    total_bytes = 0

    def flush():
        nonlocal current, current_bytes, shard_index
        if not current:
            return
        shard_index += 1
        name = f"model-nonrouted-{shard_index:05d}.safetensors"
        save_file(current, str(out_dir / name))
        for tensor_name in current:
            weight_map[tensor_name] = name
        shard_files.append(name)
        current = {}
        current_bytes = 0

    produced: set = set()
    for name, tensor in _materialize_stream(surface, reader, dev, stats, extra_maps):
        if name in produced:
            raise _fail(f"duplicate materialized tensor {name}")
        produced.add(name)
        nbytes = int(tensor.numel() * tensor.element_size())
        if current_bytes + nbytes > shard_bytes and current:
            flush()
        current[name] = tensor
        current_bytes += nbytes
        total_bytes += nbytes
    flush()

    # completeness gate against the OFFICIAL BF16 index (name-set equality on
    # the non-routed part).  Optional but the cloud recipe always passes it.
    official_check: Dict[str, Any] = {"checked": False}
    if official_index is not None:
        official = json.loads(Path(official_index).read_text(encoding="utf-8"))["weight_map"]
        want = {n for n in official if not _ROUTED.search(n)}
        missing = sorted(want - produced)
        extra = sorted(produced - want)
        if missing or extra:
            raise _fail(
                "materialized non-routed name set differs from the official index: "
                f"missing {len(missing)} (first {missing[:3]}), extra {len(extra)} "
                f"(first {extra[:3]})"
            )
        official_check = {"checked": True, "official_nonrouted_count": len(want),
                          "official_index_sha256": _sha256_file(Path(official_index))}

    # virtual routed entries: every routed expert name from the OFFICIAL layout
    virtual_shard = "model-routed-virtual.safetensors"
    routed_layers = list(MAIN_ROUTED_LAYERS) + [MTP_LAYER]
    for layer in routed_layers:
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                weight_map[f"{routed_module_name(layer, expert, projection)}.weight"] = virtual_shard

    index_doc = {
        "metadata": {"total_size": total_bytes,
                     "note": ("virtual routed-expert entries reference a shard that does not "
                              "exist; the streaming engine's non-routed view filters them "
                              "before any shard is opened")},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_doc, sort_keys=True), encoding="utf-8"
    )

    config = json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
    quant_block = config.pop("quantization_config", None)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    for aux in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = Path(root) / aux
        if src.is_file():
            (out_dir / aux).write_bytes(src.read_bytes())

    shard_hashes = {name: _sha256_file(out_dir / name) for name in shard_files}
    inventory = _seal(
        {
            "schema": RELEASE_INVENTORY_SCHEMA,
            "model_repo": source_repo,
            "model_revision": source_revision,
            "seal_mode": "full-shard-sha256",
            "config_sha256": _sha256_file(out_dir / "config.json"),
            "index_sha256": _sha256_file(out_dir / "model.safetensors.index.json"),
            "shards": shard_hashes,
            "provenance": ("materialized locally from the quantized artifact by "
                           "exl3hf_surface.materialize_nonrouted; NOT an official release "
                           "inventory. It binds the DEQUANTIZED-BF16 non-routed tree the "
                           "streaming engine loads, whose every value derives from the "
                           "artifact revision above."),
        },
        "inventory_sha256",
    )
    (out_dir / "inventory.json").write_bytes(_canonical_json(inventory))

    receipt = _seal(
        {
            "schema": EXL3HF_MATERIALIZATION_SCHEMA,
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_config_sha256": surface.config_sha256,
            "source_index_sha256": surface.index_sha256,
            "source_quantization_config": quant_block,
            "codebook": surface.codebook,
            "exllamav3_version": surface.exllamav3_version,
            "mtp_file_used": mtp_handle is not None,
            "mtp_file_sha256": _sha256_file(mtp_path) if mtp_handle is not None else None,
            "written_tensor_count": len(produced),
            "written_bytes": total_bytes,
            "shard_files": shard_files,
            "shard_sha256": shard_hashes,
            "virtual_routed_entries": len(routed_layers) * NUM_EXPERTS * len(PROJECTIONS),
            "inventory_sha256": inventory["inventory_sha256"],
            "official_index_check": official_check,
            "stats": stats,
            "dtype_policy": ("EXL3 payloads decoded fp32 (campaign decode ABI) then one "
                             "rounding to bf16; native f16/bf16 tensors written bf16 (one "
                             "rounding); native fp32 tensors kept fp32"),
            "fused_module_policy": ("KDA self_attn.qkv_proj/conv1d split to official "
                                    "q|k|v (order proven bitwise on conv1d); visual "
                                    "attn q/k/v fused to official qkv"),
            "seal_disclosure": SEAL_DISCLOSURE,
            "elapsed_seconds": time.monotonic() - started,
        },
        "receipt_sha256",
    )
    (out_dir / "materialization-receipt.json").write_bytes(_canonical_json(receipt))
    return receipt


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    mat = sub.add_parser("materialize", help="materialize the non-routed BF16 tree")
    mat.add_argument("--root", type=Path, required=True)
    mat.add_argument("--out", type=Path, required=True)
    mat.add_argument("--device", default="cpu")
    mat.add_argument("--source-repo", required=True)
    mat.add_argument("--source-revision", required=True)
    mat.add_argument("--official-index", type=Path,
                     help="official BF16 model.safetensors.index.json for the "
                          "non-routed name-set completeness gate")
    probe = sub.add_parser("probe", help="decode one routed module and print stats")
    probe.add_argument("--root", type=Path, required=True)
    probe.add_argument("--layer", type=int, default=3)
    probe.add_argument("--expert", type=int, default=0)
    probe.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.cmd == "materialize":
        receipt = materialize_nonrouted(
            args.root, args.out, device=args.device,
            source_repo=args.source_repo, source_revision=args.source_revision,
            official_index=args.official_index,
        )
        print(json.dumps({"ok": True, "receipt_sha256": receipt["receipt_sha256"],
                          "written_tensor_count": receipt["written_tensor_count"],
                          "written_bytes": receipt["written_bytes"],
                          "stats": receipt["stats"]}, sort_keys=True))
        return 0
    if args.cmd == "probe":
        surface = load_surface(args.root)
        reader = Exl3HfShardReader(surface)
        rows = []
        for projection in PROJECTIONS:
            decoded, census = load_decoded_module(
                surface, reader, layer=args.layer, expert=args.expert,
                projection=projection, device=args.device,
            )
            rows.append({**census, "shape": list(decoded.shape),
                         "std": float(decoded.std())})
        print(json.dumps({"codebook": surface.codebook,
                          "exllamav3_version": surface.exllamav3_version,
                          "declared_bits": surface.declared_bits,
                          "modules": rows}, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
