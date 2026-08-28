#!/usr/bin/env python3
"""Dione (0xSero) selective-EXL3 TP4 checkpoint surface adapter for the K6 scorer.

Scores 0xSero/GLM-5.3-Flash-EXL3-* checkpoints ("Dione" conversion workflow,
exllamav3 @ 5f3c537, format ``glm53-selective-exl3-tp4-v1``) on OUR sealed
25-window panel with the SAME EP8 teacher-forced capture used for K4/K6/K8,
so the number is directly comparable to brandonmusic's 4bpw and our campaign.

Format (verified against the live repo @ 99cccdf0, header-level, 2026-08-28):

  * Routed scope identical to ours: layers 3..44 x 288 experts x
    {gate,up,down}_proj.  Everything else (attention, shared experts, gates,
    dense 0-2, embeddings, lm_head, norms, vision, MTP layer 45 experts) ships
    natively in source BF16 under the official tensor names.
  * Each routed projection is stored TP4-SLICED, four independent EXL3
    quantizations per matrix:
      ...experts.E.{proj}.rank{R}.{trellis,suh,svh,mcg}   R in 0..tp-1
    gate/up: out-feature slices (svh len 512), trellis [in/16, 512/16, 16*bits]
    down:    in-feature  slices (suh len 512), trellis [512/16, out/16, 16*bits]
  * Decode: each slice is EXACTLY a standard EXL3/MCG payload; the campaign
    reader's ``decode_choice_hf`` applies verbatim per slice; the full HF
    [out,in] matrix is the rank-ordered concat (dim 0 for gate/up, dim 1 for
    down).  Placement proven against the official BF16 weights (per-slice
    cosine 0.996 on the identity placement, cross terms < 0.004; see
    dione-evidence/real-payload-placement-audit.json).

DISCLOSED DEVIATION - unsealed-source scoring: the Dione checkpoint ships no
per-choice receipts, no reconstruction closures and no sealed reader ABI
(those are brandonmusic pipeline additions).  This adapter therefore decodes
the surface WITHOUT seal verification: it records the sha256 of every payload
it consumed and the repo revision it came from, and verifies whole-shard
sha256 against the release's own exl3-manifest.json, but there is no
encoder-side closure to close against.  Every receipt this adapter touches
carries ``seal_disclosure`` saying exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

DIONE_FORMAT = "glm53-selective-exl3-tp4-v1"
DIONE_QUANT_METHOD = "exl3_selective_tp4"
DIONE_SURFACE_SCHEMA = "malaiwah.glm53-dione-exl3-surface.v1"
DIONE_IDENTITY_SCHEMA = "malaiwah.glm53-dione-exl3-student-identity.v1"
DIONE_READER_IDENTITY_SCHEMA = "malaiwah.glm53-dione-exl3-offline-reader-identity.v1"
DIONE_SHARDS_VERIFIED_SCHEMA = "malaiwah.glm53-dione-shards-verified.v1"
DIONE_PLAN_SCHEMA = "malaiwah.glm53-dione-student-logit-capture-plan.v1"
SEAL_DISCLOSURE = (
    "unsealed-source scoring: the Dione checkpoint ships no upstream receipts, "
    "reconstruction closures or sealed reader ABI; the packed surface was decoded "
    "WITHOUT seal verification (consumed payload sha256s and the immutable repo "
    "revision are recorded instead; whole-shard sha256 optionally verified against "
    "the release's exl3-manifest.json)"
)

MAIN_ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYER = 45
NUM_EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
OBJECTS = ("trellis", "suh", "svh", "mcg")
CONCAT_DIM = {"gate_proj": 0, "up_proj": 0, "down_proj": 1}
PROJECTION_SHAPE = {
    "gate_proj": (2048, 4096),
    "up_proj": (2048, 4096),
    "down_proj": (4096, 2048),
}
MCG_MARKER_SIGNED_INT32 = -877912083  # int32 view of 0xCBAC1FED
_REVISION = re.compile(r"[0-9a-f]{40}")
_SLICE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.rank(\d+)\.(trellis|suh|svh|mcg)$"
)


def _fail(message: str) -> ValueError:
    return ValueError(f"dione_surface: {message}")


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


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def slice_name(layer: int, expert: int, projection: str, rank: int, obj: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.rank{rank}.{obj}"
    )


def official_name(layer: int, expert: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def expected_slice_geometry(projection: str, *, bits: int, tp_size: int) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """dtype/shape of each stored object for one TP slice of one projection."""
    out_features, in_features = PROJECTION_SHAPE[projection]
    if CONCAT_DIM[projection] == 0:  # gate/up: out-feature (column-parallel) slices
        out_slice, in_slice = out_features // tp_size, in_features
    else:  # down: in-feature (row-parallel) slices
        out_slice, in_slice = out_features, in_features // tp_size
    if out_slice % 128 or in_slice % 128:
        raise _fail(f"TP{tp_size} slice of {projection} is not tile/Hadamard aligned")
    return {
        "trellis": ("I16", (in_slice // 16, out_slice // 16, 16 * bits)),
        "suh": ("F16", (in_slice,)),
        "svh": ("F16", (out_slice,)),
        "mcg": ("I32", ()),
    }


@dataclass(frozen=True)
class DioneSurface:
    root: Path
    repo: Optional[str]
    revision: str
    bits: int
    tp_size: int
    fmt: str
    source_repo: str
    source_revision: str
    config_sha256: str
    index_sha256: str
    exl3_manifest_sha256: Optional[str]
    weight_map: Mapping[str, str]
    retained_names: Tuple[str, ...]
    shard_hash_verification: str  # "full" | "skipped"
    text_vocab_size: int

    def checkpoint_identity_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "schema": DIONE_IDENTITY_SCHEMA,
                    "dione_repo": self.repo,
                    "dione_revision": self.revision,
                    "format": self.fmt,
                    "bits": self.bits,
                    "tp_size": self.tp_size,
                    "config_sha256": self.config_sha256,
                    "index_sha256": self.index_sha256,
                    "exl3_manifest_sha256": self.exl3_manifest_sha256,
                    "source_repo": self.source_repo,
                    "source_revision": self.source_revision,
                    "shard_hash_verification": self.shard_hash_verification,
                    "codebook": "MCG",
                    "nonrouted_policy": "official_source_native",
                    "seal_disclosure": SEAL_DISCLOSURE,
                }
            )
        )


def census_weight_map(weight_map: Mapping[str, str], *, tp_size: int) -> Tuple[List[str], Dict[str, int]]:
    """Fail-closed name census: every routed slice present, none stray.

    Returns (retained_names, counts).  Pure names/topology - shapes are
    checked from shard headers by validate_layout when shards are on disk.
    """
    packed_seen: set = set()
    stray: List[str] = []
    retained: List[str] = []
    for name in weight_map:
        match = _SLICE.match(name)
        if match is None:
            retained.append(name)
            continue
        layer, expert, projection, rank, obj = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            int(match.group(4)),
            match.group(5),
        )
        if layer not in MAIN_ROUTED_LAYERS or expert >= NUM_EXPERTS or rank >= tp_size:
            stray.append(name)
            continue
        packed_seen.add((layer, expert, projection, rank, obj))
    if stray:
        raise _fail(f"packed tensors outside the declared scope: {stray[:5]}")
    missing = [
        slice_name(layer, expert, projection, rank, obj)
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
        for rank in range(tp_size)
        for obj in OBJECTS
        if (layer, expert, projection, rank, obj) not in packed_seen
    ]
    if missing:
        raise _fail(f"{len(missing)} routed payload tensors absent, e.g. {missing[:3]}")
    # routed originals must NOT also ship natively; MTP layer 45 MUST ship natively
    for layer in MAIN_ROUTED_LAYERS:
        if official_name(layer, 0, "gate_proj") in weight_map:
            raise _fail(f"layer {layer} ships both packed and native routed tensors")
    mtp_missing = [
        official_name(MTP_LAYER, expert, projection)
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
        if official_name(MTP_LAYER, expert, projection) not in weight_map
    ]
    if mtp_missing:
        raise _fail(f"MTP layer 45 native experts absent, e.g. {mtp_missing[:3]}")
    counts = {
        "packed_tensors": len(packed_seen),
        "packed_modules": len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * len(PROJECTIONS),
        "retained_tensors": len(retained),
    }
    if counts["packed_tensors"] != counts["packed_modules"] * tp_size * len(OBJECTS):
        raise _fail("packed census does not close")
    return retained, counts


def load_dione_surface(
    root: str | Path,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    require_shard_hashes: bool = True,
) -> DioneSurface:
    root = Path(root).resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = _read_json(config_path, "dione config.json")
    index = _read_json(index_path, "dione model.safetensors.index.json")
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise _fail("config.json has no quantization_config block")
    if (
        quant.get("quant_method") != DIONE_QUANT_METHOD
        or quant.get("format") != DIONE_FORMAT
        or quant.get("mcg") is not True
        or quant.get("requires_custom_loader") is not True
        or quant.get("retained_dtype") != "source_precision"
    ):
        raise _fail(f"quantization_config is not the known Dione TP4 format: {dict(quant)}")
    bits = int(quant.get("trellis_k", -1))
    tp_size = int(quant.get("tensor_parallel_size", -1))
    if bits not in (2, 3, 4, 5, 6, 8) or float(quant.get("bits_per_weight", -1)) != float(bits):
        raise _fail(f"trellis_k/bits_per_weight pair unsupported: {bits}")
    if tp_size != 4:
        raise _fail(f"only the published TP4 slicing is supported, got tp_size={tp_size}")
    text = config.get("text_config", {})
    if (
        config.get("architectures") != ["Glm5NextForConditionalGeneration"]
        or config.get("model_type") != "glm5_next"
        or text.get("model_type") != "glm5_next_text"
        or text.get("num_hidden_layers") != 45
        or text.get("num_nextn_predict_layers") != 1
        or text.get("n_routed_experts") != NUM_EXPERTS
        or text.get("hidden_size") != 4096
        or text.get("moe_intermediate_size") != 2048
    ):
        raise _fail("dione checkpoint does not carry official GLM5Next main/MTP geometry")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise _fail("index has no weight_map")
    metadata = index.get("metadata") or {}
    if metadata.get("format") not in (None, DIONE_FORMAT):
        raise _fail(f"index metadata format differs: {metadata.get('format')}")
    retained, _ = census_weight_map(weight_map, tp_size=tp_size)

    manifest_path = root / "exl3-manifest.json"
    manifest_sha: Optional[str] = None
    source_repo = "zai-org/GLM-5.3-Flash-BF16"
    source_revision = ""
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "exl3-manifest.json")
        manifest_sha = _sha256_file(manifest_path)
        source_repo = str(manifest.get("source_repo", source_repo))
        source_revision = str(manifest.get("source_revision", ""))
        if int(manifest.get("tensor_parallel_size", tp_size)) != tp_size or float(
            manifest.get("bits_per_weight", bits)
        ) != float(bits):
            raise _fail("exl3-manifest.json disagrees with config quantization_config")
    if not source_revision:
        source_revision = str(metadata.get("source_revision", ""))
    if _REVISION.fullmatch(source_revision) is None:
        raise _fail("source (BF16) revision is not an immutable 40-hex commit")
    if revision is not None and _REVISION.fullmatch(revision) is None:
        raise _fail("--dione-revision must be the immutable 40-hex repo commit")

    marker = root / "dione-shards-verified.json"
    if marker.is_file():
        verified = _read_json(marker, "shard verification marker")
        if (
            verified.get("schema") != DIONE_SHARDS_VERIFIED_SCHEMA
            or verified.get("exl3_manifest_sha256") != manifest_sha
            or verified.get("all_verified") is not True
        ):
            raise _fail("stale/foreign dione-shards-verified.json - re-run verify-shards")
        shard_hash_verification = "full"
    elif require_shard_hashes:
        raise _fail(
            "whole-shard sha256 verification marker absent: run "
            f"`python dione_surface.py verify-shards --root {root}` first, or pass "
            "--skip-shard-hashes for a disclosed unverified read"
        )
    else:
        shard_hash_verification = "skipped"

    return DioneSurface(
        root=root,
        repo=repo,
        revision=revision or "unpinned-local-snapshot",
        bits=bits,
        tp_size=tp_size,
        fmt=DIONE_FORMAT,
        source_repo=source_repo,
        source_revision=source_revision,
        config_sha256=_sha256_file(config_path),
        index_sha256=_sha256_file(index_path),
        exl3_manifest_sha256=manifest_sha,
        weight_map=dict(weight_map),
        retained_names=tuple(sorted(retained)),
        shard_hash_verification=shard_hash_verification,
        text_vocab_size=int(text["vocab_size"]),
    )


def verify_shard_hashes(root: str | Path) -> Dict[str, Any]:
    """Hash every shard against exl3-manifest.json; write the marker file."""
    root = Path(root).resolve()
    manifest = _read_json(root / "exl3-manifest.json", "exl3-manifest.json")
    rows: List[Dict[str, Any]] = []
    started = time.monotonic()
    for group, subdir in (("quantized_shards", "layers"), ("retained_shards", "retained")):
        for entry in manifest.get(group, []):
            path = root / subdir / str(entry["name"])
            if not path.is_file():
                raise _fail(f"shard listed in manifest is absent: {path}")
            observed = _sha256_file(path)
            ok = observed == entry["sha256"] and path.stat().st_size == int(entry["bytes"])
            rows.append({"shard": f"{subdir}/{entry['name']}", "ok": ok})
            if not ok:
                raise _fail(f"shard hash differs from exl3-manifest.json: {path}")
    record = {
        "schema": DIONE_SHARDS_VERIFIED_SCHEMA,
        "root": str(root),
        "exl3_manifest_sha256": _sha256_file(root / "exl3-manifest.json"),
        "shards": len(rows),
        "all_verified": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "dione-shards-verified.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


class DioneShardReader:
    """Cached safetensors handles over the snapshot's shard files."""

    def __init__(self, surface: DioneSurface):
        self.surface = surface
        self._handles: Dict[str, Any] = {}

    def _handle(self, shard_rel: str):
        handle = self._handles.get(shard_rel)
        if handle is None:
            from safetensors import safe_open

            path = self.surface.root / shard_rel
            if not path.is_file():
                raise _fail(f"shard absent: {path}")
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._handles[shard_rel] = handle
        return handle

    def tensor(self, name: str):
        shard_rel = self.surface.weight_map.get(name)
        if shard_rel is None:
            raise _fail(f"tensor not in weight_map: {name}")
        return self._handle(shard_rel).get_tensor(name)


# ---------------------------------------------------------------------------
# decode - delegates to the campaign reader for its supported bit-rates so the
# scoring math is BYTE-IDENTICAL to the sealed K4/K6/K8 measurements; the
# anybits copy below (same math, rate gate widened) covers K3/K5 checkpoints.
# ---------------------------------------------------------------------------

def _reader():
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader
    from quant_pipeline.checkpoint.packed_payload import (
        MCG_MARKER_SIGNED_INT32 as pipeline_marker,
    )

    if MCG_MARKER_SIGNED_INT32 != pipeline_marker:
        raise _fail("MCG marker constant drifted from the campaign reader")
    return reader


def _unpack_trellis_states_anybits(packed, bits: int):
    """Verbatim reader.unpack_trellis_states math, rate gate widened to 1..8."""
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


def _decode_choice_hf_anybits(trellis, suh, svh, *, bits: int):
    """Verbatim reader.decode_choice_hf math over the anybits unpack."""
    import torch

    reader = _reader()
    states = _unpack_trellis_states_anybits(trellis, bits)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = (
        reader.mcg_lut(states.device).index_select(0, indices.flatten()).reshape_as(states).float()
    )
    values = values.index_select(-1, torch.argsort(reader._permutation(states.device)))
    k_tiles, n_tiles, _ = values.shape
    exl = (
        values.reshape(k_tiles, n_tiles, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
    )
    had = reader._hadamard(exl.device, exl.dtype)
    exl = torch.matmul(had, exl.reshape(-1, 128, exl.shape[1])).reshape_as(exl)
    exl *= suh.to(device=exl.device, dtype=exl.dtype).reshape(-1, 1)
    exl = torch.matmul(exl.reshape(exl.shape[0], -1, 128), had).reshape_as(exl)
    exl *= svh.to(device=exl.device, dtype=exl.dtype).reshape(1, -1)
    return exl.T.contiguous()


def decode_slice(trellis, suh, svh, *, bits: int):
    reader = _reader()
    if bits in reader.SUPPORTED_BITS:
        return reader.decode_choice_hf(trellis, suh, svh, bits=bits)
    return _decode_choice_hf_anybits(trellis, suh, svh, bits=bits)


def _tensor_sha256(value) -> str:
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def load_decoded_module(
    surface: DioneSurface,
    shards: DioneShardReader,
    *,
    layer: int,
    expert: int,
    projection: str,
    device,
    hash_payloads: bool = True,
):
    """Decode + rank-ordered concat -> official/HF [out,in] fp32 tensor."""
    import torch

    geometry = expected_slice_geometry(projection, bits=surface.bits, tp_size=surface.tp_size)
    dtype_map = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32}
    decoded_slices = []
    census: Dict[str, Any] = {"module": official_name(layer, expert, projection), "slices": []}
    for rank in range(surface.tp_size):
        payload = {}
        for obj in OBJECTS:
            value = shards.tensor(slice_name(layer, expert, projection, rank, obj))
            want_dtype, want_shape = geometry[obj]
            if value.dtype != dtype_map[want_dtype] or tuple(value.shape) != want_shape:
                raise _fail(
                    f"slice geometry differs: L{layer} E{expert} {projection} rank{rank} "
                    f"{obj} {value.dtype} {tuple(value.shape)} != {want_dtype} {want_shape}"
                )
            payload[obj] = value
        if int(payload["mcg"].reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32:
            raise _fail(f"MCG marker differs: L{layer} E{expert} {projection} rank{rank}")
        if hash_payloads:
            census["slices"].append(
                {
                    "rank": rank,
                    "trellis_sha256": _tensor_sha256(payload["trellis"]),
                    "suh_sha256": _tensor_sha256(payload["suh"]),
                    "svh_sha256": _tensor_sha256(payload["svh"]),
                }
            )
        moved = {name: payload[name].to(device) for name in ("trellis", "suh", "svh")}
        decoded_slices.append(
            decode_slice(moved["trellis"], moved["suh"], moved["svh"], bits=surface.bits)
        )
    full = torch.cat(decoded_slices, dim=CONCAT_DIM[projection]).contiguous()
    if tuple(full.shape) != PROJECTION_SHAPE[projection]:
        raise _fail(f"assembled module has wrong official shape: {tuple(full.shape)}")
    return full, census


def install_local_main_experts_dione(
    model, surface: DioneSurface, shards: DioneShardReader, *, rank: int, device
) -> Dict[str, Any]:
    """Mirror of reader.install_local_main_experts over the Dione surface."""
    import torch

    reader = _reader()
    layers = reader.resolve_main_layers(model)
    start, stop = reader.expert_range(rank)
    installed: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for layer_index in MAIN_ROUTED_LAYERS:
            experts = layers[layer_index].mlp.experts
            gate_up_target = experts.gate_up_proj
            down_target = experts.down_proj
            gate_up_target = (
                gate_up_target.to_local() if hasattr(gate_up_target, "to_local") else gate_up_target
            )
            down_target = down_target.to_local() if hasattr(down_target, "to_local") else down_target
            per_rank = NUM_EXPERTS // reader.EP_SIZE
            if tuple(gate_up_target.shape) != (per_rank, 4096, 4096) or tuple(
                down_target.shape
            ) != (per_rank, 4096, 2048):
                raise _fail(f"EP{reader.EP_SIZE} local expert layout differs at layer {layer_index}")
            for global_expert in range(start, stop):
                local_expert = global_expert - start
                gate, gate_census = load_decoded_module(
                    surface, shards, layer=layer_index, expert=global_expert,
                    projection="gate_proj", device=device,
                )
                up, up_census = load_decoded_module(
                    surface, shards, layer=layer_index, expert=global_expert,
                    projection="up_proj", device=device,
                )
                down, down_census = load_decoded_module(
                    surface, shards, layer=layer_index, expert=global_expert,
                    projection="down_proj", device=device,
                )
                gate_up = reader.fuse_gate_up(gate, up)
                gate_up_bf16 = gate_up.to(dtype=torch.bfloat16)
                down_bf16 = down.to(dtype=torch.bfloat16)
                gate_up_target[local_expert].copy_(gate_up_bf16)
                down_target[local_expert].copy_(down_bf16)
                if not torch.equal(gate_up_target[local_expert], gate_up_bf16) or not torch.equal(
                    down_target[local_expert], down_bf16
                ):
                    raise RuntimeError("BF16 local expert installation did not close exactly")
                installed.append(
                    {
                        "layer": layer_index,
                        "global_expert": global_expert,
                        "local_expert": local_expert,
                        "payload_census": [gate_census, up_census, down_census],
                    }
                )
                del gate, up, down, gate_up, gate_up_bf16, down_bf16
    return {
        "schema": DIONE_SURFACE_SCHEMA,
        "rank": rank,
        "global_expert_start": start,
        "global_expert_stop": stop,
        "main_layers": list(MAIN_ROUTED_LAYERS),
        "installed_expert_triplets": len(installed),
        "installed_matrix_count": len(installed) * len(PROJECTIONS),
        "tp_slices_per_matrix": surface.tp_size,
        "installed_payload_census_sha256": _sha256_bytes(_canonical_json(installed)),
        "mutated_parameter_suffixes": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        "nonrouted_parameters_mutated": False,
        "mtp_parameters_mutated": False,
        "seal_disclosure": SEAL_DISCLOSURE,
    }


def audit_slice_placement(
    surface: DioneSurface,
    shards: DioneShardReader,
    bf16_root: str | Path,
    *,
    layer: int = 3,
    expert: int = 0,
    device="cpu",
) -> Dict[str, Any]:
    """Prove the rank->block placement against the official BF16 weights.

    For each projection: decode every TP slice, correlate against every
    candidate contiguous block of the official tensor; the identity placement
    must dominate.  Catches any silent slice-order or axis regression in a
    future Dione release before a single logit is captured.
    """
    import torch
    import torch.nn.functional as F

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    from safetensors import safe_open

    audit: Dict[str, Any] = {"layer": layer, "expert": expert, "projections": {}}
    for projection in PROJECTIONS:
        name = official_name(layer, expert, projection)
        shard = bf16_index["weight_map"].get(name)
        if shard is None:
            raise _fail(f"official BF16 checkpoint lacks {name}")
        with safe_open(str(bf16_root / shard), framework="pt", device="cpu") as handle:
            official = handle.get_tensor(name).float()
        geometry = expected_slice_geometry(projection, bits=surface.bits, tp_size=surface.tp_size)
        slice_width = geometry["svh"][1][0] if CONCAT_DIM[projection] == 0 else geometry["suh"][1][0]
        decoded = []
        for rank in range(surface.tp_size):
            tr = shards.tensor(slice_name(layer, expert, projection, rank, "trellis")).to(device)
            suh = shards.tensor(slice_name(layer, expert, projection, rank, "suh")).to(device)
            svh = shards.tensor(slice_name(layer, expert, projection, rank, "svh")).to(device)
            decoded.append(decode_slice(tr, suh, svh, bits=surface.bits).cpu())
        matrix = []
        for rank in range(surface.tp_size):
            row = []
            for block in range(surface.tp_size):
                if CONCAT_DIM[projection] == 0:
                    blk = official[block * slice_width : (block + 1) * slice_width]
                else:
                    blk = official[:, block * slice_width : (block + 1) * slice_width]
                row.append(
                    float(F.cosine_similarity(decoded[rank].flatten(), blk.flatten(), dim=0))
                )
            matrix.append(row)
        diag = [matrix[i][i] for i in range(surface.tp_size)]
        off = [
            matrix[i][j]
            for i in range(surface.tp_size)
            for j in range(surface.tp_size)
            if i != j
        ]
        ok = min(diag) > 0.90 and min(diag) > max(off) + 0.5
        full = torch.cat(decoded, dim=CONCAT_DIM[projection])
        rel_l2 = float((full - official).norm() / official.norm())
        audit["projections"][projection] = {
            "concat_dim": CONCAT_DIM[projection],
            "cosine_rank_x_block": matrix,
            "identity_placement_dominates": ok,
            "assembled_rel_l2_vs_official_bf16": rel_l2,
        }
        if not ok:
            raise _fail(
                f"slice placement audit FAILED for {projection}: {matrix} - the "
                "checkpoint's TP slicing differs from the proven rank-ordered layout"
            )
    audit["passed"] = True
    return audit


def verify_nonrouted_tensors(
    surface: DioneSurface,
    shards: DioneShardReader,
    bf16_root: str | Path,
    *,
    mode: str = "sample",
    sample_count: int = 64,
) -> Dict[str, Any]:
    """Retained (non-routed + MTP-native) tensors vs the official BF16 checkpoint.

    modes: "full" byte-compares every retained tensor; "sample" byte-compares a
    deterministic subset and shape/dtype-checks the rest; "names" checks the
    bijection only.  The bijection itself is always enforced: official names ==
    dione retained names + the routed originals the packed slices replace.
    """
    import torch

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    official_map: Mapping[str, str] = bf16_index["weight_map"]
    expected_official = set(surface.retained_names) | {
        official_name(layer, expert, projection)
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    if set(official_map) != expected_official:
        extra = sorted(set(official_map) - expected_official)[:3]
        missing = sorted(expected_official - set(official_map))[:3]
        raise _fail(
            f"official/dione tensor bijection differs (extra {extra}, missing {missing})"
        )
    if mode not in ("full", "sample", "names"):
        raise _fail(f"unknown nonrouted verification mode: {mode}")
    result = {
        "mode": mode,
        "retained_tensors": len(surface.retained_names),
        "bijection_ok": True,
    }
    if mode == "names":
        return result

    from safetensors import safe_open

    bf16_handles: Dict[str, Any] = {}

    def _official(name: str):
        shard = official_map[name]
        handle = bf16_handles.get(shard)
        if handle is None:
            handle = safe_open(str(bf16_root / shard), framework="pt", device="cpu")
            bf16_handles[shard] = handle
        return handle.get_tensor(name)

    if mode == "full":
        chosen = list(surface.retained_names)
    else:
        anchors = [
            name
            for name in (
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
                "model.language_model.norm.weight",
            )
            if name in surface.retained_names
        ]
        rng = np.random.default_rng(0x610E)  # fixed seed: deterministic sample
        pool = sorted(set(surface.retained_names) - set(anchors))
        picks = rng.choice(len(pool), size=min(sample_count, len(pool)), replace=False)
        chosen = anchors + [pool[int(index)] for index in sorted(picks)]
    compared = 0
    for name in chosen:
        ours = shards.tensor(name)
        theirs = _official(name)
        if ours.dtype != theirs.dtype or tuple(ours.shape) != tuple(theirs.shape):
            raise _fail(f"retained tensor geometry differs from official: {name}")
        if not torch.equal(ours, theirs):
            raise _fail(f"retained tensor bytes differ from official BF16: {name}")
        compared += 1
        del ours, theirs
    if mode == "sample":
        for name in surface.retained_names:
            ours_slice = shards._handle(surface.weight_map[name]).get_slice(name)
            shard = official_map[name]
            handle = bf16_handles.get(shard)
            if handle is None:
                handle = safe_open(str(bf16_root / shard), framework="pt", device="cpu")
                bf16_handles[shard] = handle
            if list(ours_slice.get_shape()) != list(handle.get_slice(name).get_shape()):
                raise _fail(f"retained tensor shape differs from official: {name}")
    result["byte_compared_tensors"] = compared
    result["all_equal"] = True
    return result


def dione_reader_identity(runner_path: str | Path, *, bits: int) -> Dict[str, Any]:
    """Identity binding this adapter + the campaign reader + the runner."""
    reader = _reader()
    body = {
        "schema": DIONE_READER_IDENTITY_SCHEMA,
        "mode": (
            "offline_dione_tp4_slice_decode_concat_to_bf16_ep_for_logit_measurement"
        ),
        "serving_kernel": False,
        "bits": bits,
        "codebook": "MCG",
        "mcg_multiplier_hex": "0xCBAC1FED",
        "concat_dims": dict(CONCAT_DIM),
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "campaign_reader_sha256": _sha256_file(Path(reader.__file__).resolve()),
        "runner_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    body["runtime_reader_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


# ---------------------------------------------------------------------------
# standalone CLI: layout dry-run / shard hashing / CPU placement probe
# ---------------------------------------------------------------------------

def _add_pipeline_root(path: Optional[str]) -> None:
    if not path:
        path = os.environ.get("QP_PIPELINE_ROOT")
    if path:
        for candidate in ("runtime/src", "src", "."):
            src = Path(path) / candidate
            if (src / "quant_pipeline" / "__init__.py").is_file():
                if str(src.resolve()) not in sys.path:
                    sys.path.insert(0, str(src.resolve()))
                return
        raise _fail(f"no quant_pipeline package under {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dry-run", help="validate a snapshot layout from config+index alone")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--skip-shard-hashes", action="store_true")

    p = sub.add_parser("verify-shards", help="hash every shard against exl3-manifest.json")
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("probe", help="CPU decode + slice placement audit vs official BF16")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--pipeline-root")
    p.add_argument("--skip-shard-hashes", action="store_true")

    args = parser.parse_args()
    if args.command == "verify-shards":
        record = verify_shard_hashes(args.root)
        print(json.dumps(record, sort_keys=True))
        return 0

    surface = load_dione_surface(
        args.root,
        repo=getattr(args, "repo", None),
        revision=getattr(args, "revision", None),
        require_shard_hashes=not args.skip_shard_hashes,
    )
    summary = {
        "schema": DIONE_SURFACE_SCHEMA,
        "root": str(surface.root),
        "bits": surface.bits,
        "tp_size": surface.tp_size,
        "format": surface.fmt,
        "source_repo": surface.source_repo,
        "source_revision": surface.source_revision,
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "exl3_manifest_sha256": surface.exl3_manifest_sha256,
        "shard_hash_verification": surface.shard_hash_verification,
        "packed_modules": len(MAIN_ROUTED_LAYERS) * NUM_EXPERTS * len(PROJECTIONS),
        "retained_tensors": len(surface.retained_names),
        "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    if args.command == "probe":
        _add_pipeline_root(args.pipeline_root)
        shards = DioneShardReader(surface)
        summary["placement_audit"] = audit_slice_placement(
            surface, shards, args.bf16, layer=args.layer, expert=args.expert
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
