#!/usr/bin/env python3
"""NVFP4 (e2m1, group-16) checkpoint surface adapter for the streaming scorer.

Scores community NVFP4 quantizations of GLM-5.3-Flash on OUR sealed 25-window
panel through ``stream_score.py --source nvfp4``, with the SAME measured
function (same panel, same teacher, same fp64 estimator, same EP8/fp32 lane)
used for K4/K6/K8/Dione, so the number lands on the same yardstick.

Two on-disk dialects are supported, both verified against the live repos at
header level (2026-08-29, range requests, no full downloads):

  * ``compressed-tensors`` (RedHatAI/GLM-5.3-Flash-NVFP4 @ 36c184c6):
    per routed expert projection ``weight_packed`` U8 [out, in/2],
    ``weight_scale`` F8_E4M3 [out, in/16], ``weight_global_scale`` F32 [1]
    (plus ``input_global_scale`` F32 [1]: the W4A4 activation scheme's static
    global).  Dequant: W = e2m1(packed) * (scale.f32 / weight_global_scale).
    MTP layer-45 experts ship as a SEPARATE FP8 block-128x128 group
    (``weight`` F8_E4M3 + ``weight_scale`` BF16 [16,32]) - present, hashed
    into the identity, never executed by standard logits.
  * ``modelopt`` (LibertAIDAI/GLM-5.3-Flash-NVFP4 @ 357b45cc):
    ``weight`` U8 (packed, same nibble stream), ``weight_scale`` F8_E4M3,
    ``weight_scale_2`` F32 scalar, ``input_scale`` F32 scalar; all 43 expert
    layers incl MTP-45.  Dequant: W = e2m1(packed) * scale.f32 *
    weight_scale_2 (compressed-tensors' modelopt converter documents the
    equivalence weight_global_scale = 1/weight_scale_2, with vLLM citations).

SCOPE FINDING (measured from the real indexes, not the READMEs): both repos
quantize the ROUTED EXPERTS ONLY.  Every non-expert tensor - embeddings,
attention/KDA/DSA, shared experts, dense MLPs 0-2, vision, norms, lm_head -
ships as plain BF16 under EXACTLY the official tensor names (the 1,618-name
non-routed set of the official BF16 index, verified as a set equality).  The
streaming scorer therefore builds its non-routed view from the QUANT SNAPSHOT
ITSELF; the official BF16 tree plays no role in an NVFP4 run (--bf16 is
refused).  The receipt's scope-policy block states this measured scope so a
registry row can never imply "everything was quantized" for this family.

Nibble stream (proven bitwise against compressed_tensors 0.18.0
``unpack_fp4_from_uint8`` + ``_dequantize`` in fp32 on real fetched tensors):
LOW nibble first; bit 3 is the sign, bits 0-2 index the e2m1 magnitude LUT
[0, 0.5, 1, 1.5, 2, 3, 4, 6].  Nibble 0b1000 decodes to -0.0, exactly as the
reference does.  Groups of 16 along the input (last) axis.

DISCLOSED DEVIATION - decode dtype: compressed-tensors' own ``decompress``
unpacks to bf16 and multiplies in bf16.  This adapter decodes in EXACT fp32
(every e2m1 value, every f8e4m3 scale and every f32 global are exact in fp32;
one divide or multiply per group scale; one multiply per element) and rounds
ONCE to bf16 at slab install - the suite's own installation algebra.  The
bitwise equality claim is against the exact fp32 math, proven by selftest and
by the real-tensor cross-check fixtures in nvfp4-evidence/.

DISCLOSED DEVIATION - unsealed-source scoring: NVFP4 checkpoints ship no
encoder-side receipts and no reconstruction closures.  The surface is decoded
WITHOUT seal verification: consumed component sha256s and the immutable repo
revision are recorded, and whole-shard sha256 can be verified against a
fetched HF LFS manifest (``fetch-manifest`` + ``verify-shards``), but there is
no encoder closure to close against.  Receipts carry ``seal_disclosure``.

Kernel constraints honoured: plain torch, byte-level uint8 unpack, fp32
accumulation, no float64 anywhere, int64 only as gather indices - runs on
CPU, CUDA and Apple MPS (float8 storage dtypes are converted to fp32 via a
256-entry LUT on whatever device holds the bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

NVFP4_SURFACE_SCHEMA = "malaiwah.glm53-nvfp4-surface.v1"
NVFP4_IDENTITY_SCHEMA = "malaiwah.glm53-nvfp4-student-identity.v1"
NVFP4_READER_IDENTITY_SCHEMA = "malaiwah.glm53-nvfp4-offline-reader-identity.v1"
NVFP4_SHARD_MANIFEST_SCHEMA = "malaiwah.glm53-nvfp4-shard-manifest.v1"
NVFP4_SHARDS_VERIFIED_SCHEMA = "malaiwah.glm53-nvfp4-shards-verified.v1"
NVFP4_STUDENT_LABEL = "nvfp4-e2m1-gs16"

LAYOUT_COMPRESSED_TENSORS = "compressed-tensors"
LAYOUT_MODELOPT = "modelopt"

SEAL_DISCLOSURE = (
    "unsealed-source scoring: the NVFP4 checkpoint ships no encoder-side receipts or "
    "reconstruction closures; the packed surface was decoded WITHOUT seal verification "
    "(consumed component sha256s and the immutable repo revision are recorded instead; "
    "whole-shard sha256 optionally verified against a fetched HF LFS manifest)"
)

#: Group size and projection names are the NVFP4 format's; everything else
#: about WHERE the routed experts live is the model family's, held as data so
#: the same fail-closed census serves GLM-5.3-Flash (`glm5_next`, 45 layers,
#: 288 experts, the VL stack at `model.language_model.`) and the GLM-5.3
#: flagship (`glm_moe_dsa`, 78 layers, 256 experts, `model.`). Both are read
#: off the config and cross-checked against the index; a model_type outside
#: this table is refused by name.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GROUP_SIZE = 16


@dataclass(frozen=True)
class Nvfp4Geometry:
    model_type: str
    architectures: Tuple[str, ...]
    stack: str  # "model.language_model." (VL) or "model." (text-only)
    num_hidden_layers: int
    first_dense_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    official_nonrouted_evidence: str  # file under nvfp4-evidence/
    official_nonrouted_count: int
    config_geometry: Tuple[Tuple[str, Any], ...]  # config keys checked verbatim

    @property
    def main_routed_layers(self) -> Tuple[int, ...]:
        return tuple(range(self.first_dense_layers, self.num_hidden_layers))

    @property
    def mtp_layer(self) -> int:
        return self.num_hidden_layers

    @property
    def projection_shape(self) -> Dict[str, Tuple[int, int]]:
        inter, hidden = self.moe_intermediate_size, self.hidden_size
        return {"gate_proj": (inter, hidden), "up_proj": (inter, hidden),
                "down_proj": (hidden, inter)}

    def expert_re(self) -> "re.Pattern[str]":
        return re.compile(
            r"^" + re.escape(self.stack) + r"layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.([a-z0-9_]+)$")

    def official_name(self, layer: int, expert: int, projection: str) -> str:
        return f"{self.stack}layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def component_name(self, layer: int, expert: int, projection: str, component: str) -> str:
        return f"{self.stack}layers.{layer}.mlp.experts.{expert}.{projection}.{component}"

    def text_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return config.get("text_config", {}) if self.stack != "model." else config


GLM5NEXT_GEOMETRY = Nvfp4Geometry(
    model_type="glm5_next",
    architectures=("Glm5NextForConditionalGeneration",),
    stack="model.language_model.",
    num_hidden_layers=45, first_dense_layers=3, num_experts=288,
    hidden_size=4096, moe_intermediate_size=2048,
    official_nonrouted_evidence="official-nonrouted-names.json",
    official_nonrouted_count=1618,
    config_geometry=(("model_type", "glm5_next_text"), ("num_hidden_layers", 45),
                     ("num_nextn_predict_layers", 1), ("n_routed_experts", 288),
                     ("hidden_size", 4096), ("moe_intermediate_size", 2048)),
)
GLM_MOE_DSA_GEOMETRY = Nvfp4Geometry(
    model_type="glm_moe_dsa",
    architectures=("GlmMoeDsaForCausalLM",),
    stack="model.",
    num_hidden_layers=78, first_dense_layers=3, num_experts=256,
    hidden_size=6144, moe_intermediate_size=2048,
    official_nonrouted_evidence="official-glm53-nonrouted-names.json",
    official_nonrouted_count=1217,
    config_geometry=(("model_type", "glm_moe_dsa"), ("num_hidden_layers", 78),
                     ("num_nextn_predict_layers", 1), ("n_routed_experts", 256),
                     ("first_k_dense_replace", 3),
                     ("hidden_size", 6144), ("moe_intermediate_size", 2048)),
)
GEOMETRIES = {g.model_type: g for g in (GLM5NEXT_GEOMETRY, GLM_MOE_DSA_GEOMETRY)}

# The Flash geometry's constants, kept under their historical names: the
# streaming lane's callers and its selftest address them directly.
MAIN_ROUTED_LAYERS = GLM5NEXT_GEOMETRY.main_routed_layers
MTP_LAYER = GLM5NEXT_GEOMETRY.mtp_layer
NUM_EXPERTS = GLM5NEXT_GEOMETRY.num_experts
PROJECTION_SHAPE = GLM5NEXT_GEOMETRY.projection_shape

# Component sets per dialect, measured from the real indexes (148,498 /
# 150,226 tensors).  DECODE names feed the dequant; the activation-scale
# names are acknowledged in the scope census but never read by the decode.
CT_NVFP4_DECODE = ("weight_packed", "weight_scale", "weight_global_scale")
CT_NVFP4_ACTIVATION = ("input_global_scale",)
CT_FP8_COMPONENTS = ("weight", "weight_scale")
MO_NVFP4_DECODE = ("weight", "weight_scale", "weight_scale_2")
MO_NVFP4_ACTIVATION = ("input_scale",)
KNOWN_COMPONENTS = {
    LAYOUT_COMPRESSED_TENSORS: set(CT_NVFP4_DECODE) | set(CT_NVFP4_ACTIVATION)
    | set(CT_FP8_COMPONENTS),
    LAYOUT_MODELOPT: set(MO_NVFP4_DECODE) | set(MO_NVFP4_ACTIVATION),
}
#: modelopt config keys that, when true, declare an ONLINE weight transform
#: (a rotation folded into the activations at serving time) that a
#: decode-and-run measurement would not apply. Inferact's flagship export
#: declares all four false; any true value is refused by name.
MO_ONLINE_TRANSFORM_KEYS = ("rotate", "learned_rotation", "quarot_r1_fold",
                            "expert_block_reorder")

_REVISION = re.compile(r"[0-9a-f]{40}")
_EXPERT = GLM5NEXT_GEOMETRY.expert_re()

_EVIDENCE = Path(__file__).resolve().parent / "nvfp4-evidence"
OFFICIAL_NONROUTED_NAMES = _EVIDENCE / GLM5NEXT_GEOMETRY.official_nonrouted_evidence


def _fail(message: str) -> ValueError:
    return ValueError(f"nvfp4_surface: {message}")


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


def official_name(layer: int, expert: int, projection: str,
                  geometry: Nvfp4Geometry = GLM5NEXT_GEOMETRY) -> str:
    return geometry.official_name(layer, expert, projection)


def component_name(layer: int, expert: int, projection: str, component: str,
                   geometry: Nvfp4Geometry = GLM5NEXT_GEOMETRY) -> str:
    return geometry.component_name(layer, expert, projection, component)


# ---------------------------------------------------------------------------
# decode kernels - plain torch, fp32, byte-level unpack, MPS-safe
# ---------------------------------------------------------------------------

_LUT_CACHE: Dict[Tuple[str, str], Any] = {}


def _e2m1_lut16(device) -> Any:
    """16-entry signed e2m1 LUT indexed by the raw nibble.

    Index 8 (0b1000) is NEGATIVE ZERO, exactly as compressed-tensors'
    ``unpack_fp4_from_uint8`` produces (kE2M1[0] * -1.0 == -0.0).
    """
    import torch

    key = ("e2m1", str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=device,
        )
        _LUT_CACHE[key] = lut
    return lut


def _f8e4m3_lut(device) -> Any:
    """256-entry float8_e4m3fn -> float32 LUT built from the format definition.

    e4m3fn: bias 7, no infinities, NaN only at S.1111.111.  Every finite value
    is exactly representable in fp32, so indexing this LUT is BIT-IDENTICAL to
    torch's native ``.to(torch.float32)`` cast on all 254 finite codes -
    negative zero (0x80) included - while working on devices with no float8
    kernels (MPS).  The two NaN codes are NaN in both but carry different
    payloads (IEEE-754 leaves payloads unspecified and they vary by device), so
    that is where the equality claim stops; selftest_nvfp4_offline rung 1
    asserts exactly this and no more.  No NaN can reach arithmetic anyway:
    dequant_nvfp4 refuses a NaN scale before applying it.
    """
    import torch

    key = ("f8e4m3", str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        values = []
        for byte in range(256):
            sign = -1.0 if (byte >> 7) & 1 else 1.0
            exponent = (byte >> 3) & 0xF
            mantissa = byte & 0x7
            if exponent == 0xF and mantissa == 0x7:
                values.append(float("nan"))
            elif exponent == 0:
                values.append(sign * (mantissa / 8.0) * 2.0 ** -6)
            else:
                values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
        lut = torch.tensor(values, dtype=torch.float32, device=device)
        _LUT_CACHE[key] = lut
    return lut


def f8e4m3_to_float32(value):
    """float8_e4m3fn (or its uint8 byte view) -> exact float32."""
    import torch

    if value.dtype == torch.float32:
        return value
    if str(value.dtype) == "torch.float8_e4m3fn":
        value = value.view(torch.uint8)
    if value.dtype != torch.uint8:
        raise _fail(f"f8e4m3_to_float32 expects float8_e4m3fn/uint8/float32, got {value.dtype}")
    return _f8e4m3_lut(value.device)[value.long()]


def unpack_e2m1(packed):
    """U8 [rows, cols/2] -> fp32 [rows, cols], LOW nibble first (ct order)."""
    import torch

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise _fail(f"packed tensor must be 2-D uint8, got {packed.dtype} {tuple(packed.shape)}")
    rows, half = packed.shape
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(rows, half * 2)
    return _e2m1_lut16(packed.device)[nibbles.long()]


def dequant_nvfp4(packed, scale, *, weight_global_scale=None, weight_scale_2=None):
    """Exact-fp32 NVFP4 dequant, both scale conventions.

    compressed-tensors: W = e2m1 * (scale.f32 / weight_global_scale)   [divide]
    modelopt:           W = e2m1 * (scale.f32 * weight_scale_2)        [multiply]

    Each is a single fp32 rounding per group scale, matching that ecosystem's
    own reference math bitwise (never convert one convention into the other:
    1/x then divide would double-round).
    """
    import torch

    if (weight_global_scale is None) == (weight_scale_2 is None):
        raise _fail(
            "exactly one of weight_global_scale (compressed-tensors) / weight_scale_2 "
            "(modelopt) must be supplied"
        )
    values = unpack_e2m1(packed)
    rows, cols = values.shape
    if cols % GROUP_SIZE:
        raise _fail(f"input width {cols} is not a multiple of the NVFP4 group size {GROUP_SIZE}")
    groups = cols // GROUP_SIZE
    if tuple(scale.shape) != (rows, groups):
        raise _fail(
            f"weight_scale shape {tuple(scale.shape)} does not match packed geometry "
            f"[{rows}, {groups}] (group size {GROUP_SIZE})"
        )
    scale32 = f8e4m3_to_float32(scale)
    if torch.isnan(scale32).any():
        raise _fail("weight_scale contains NaN f8e4m3 codes - corrupt artifact, refusing")
    if weight_global_scale is not None:
        gs32 = weight_global_scale.to(torch.float32).reshape(())
        if torch.isnan(gs32).any() or float(gs32) == 0.0:
            raise _fail("weight_global_scale is NaN or zero - corrupt artifact, refusing")
        effective = scale32 / gs32
    else:
        s2 = weight_scale_2.to(torch.float32).reshape(())
        if torch.isnan(s2).any() or float(s2) == 0.0:
            raise _fail("weight_scale_2 is NaN or zero - corrupt artifact, refusing")
        effective = scale32 * s2
    return (
        values.reshape(rows, groups, GROUP_SIZE) * effective.reshape(rows, groups, 1)
    ).reshape(rows, cols)


def _tensor_sha256(value) -> str:
    """sha256 of the raw storage bytes (works for float8, which numpy lacks)."""
    import torch

    return _sha256_bytes(
        value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )


# ---------------------------------------------------------------------------
# surface: config + index census, fail closed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Nvfp4Surface:
    root: Path
    repo: Optional[str]
    revision: str
    layout: str  # LAYOUT_COMPRESSED_TENSORS | LAYOUT_MODELOPT
    config_format: str  # the config's own format/quant_algo string, verbatim
    producer: Dict[str, Any]
    config_sha256: str
    index_sha256: str
    weight_map: Mapping[str, str]
    retained_names: Tuple[str, ...]
    scope: Dict[str, Any] = field(compare=False)
    quant_weights: Dict[str, Any] = field(compare=False)
    activations: Dict[str, Any] = field(compare=False)
    shard_hash_verification: str  # "full" | "skipped"
    nonrouted_verification: str  # "official-name-bijection" | "structural-census"
    text_vocab_size: int
    geometry: Nvfp4Geometry = GLM5NEXT_GEOMETRY

    def decode_components(self, layer: int) -> Tuple[str, ...]:
        geometry = self.geometry
        if layer not in geometry.main_routed_layers:
            raise _fail(
                f"layer {layer} is not a streamed main routed layer "
                f"({geometry.first_dense_layers}..{geometry.num_hidden_layers - 1}); the MTP "
                f"layer-{geometry.mtp_layer} experts are receipt-covered but never decoded "
                f"or executed"
            )
        return CT_NVFP4_DECODE if self.layout == LAYOUT_COMPRESSED_TENSORS else MO_NVFP4_DECODE

    def scope_census_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.scope))

    def checkpoint_identity_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "schema": NVFP4_IDENTITY_SCHEMA,
                    "nvfp4_repo": self.repo,
                    "nvfp4_revision": self.revision,
                    "layout": self.layout,
                    "config_format": self.config_format,
                    "producer": self.producer,
                    "quant_weights": self.quant_weights,
                    "codebook": "e2m1-lut",
                    "group_size": GROUP_SIZE,
                    "config_sha256": self.config_sha256,
                    "index_sha256": self.index_sha256,
                    "scope_census_sha256": self.scope_census_sha256(),
                    "shard_hash_verification": self.shard_hash_verification,
                    "nonrouted_policy": "quant_snapshot_bf16_native",
                    "nonrouted_verification": self.nonrouted_verification,
                    "seal_disclosure": SEAL_DISCLOSURE,
                }
            )
        )


def census_weight_map(weight_map: Mapping[str, Any], *, layout: str,
                      geometry: Nvfp4Geometry = GLM5NEXT_GEOMETRY) -> Tuple[List[str], Dict[str, Any]]:
    """Fail-closed name census over the quant repo's index.

    Every expert tensor must be a KNOWN component of a KNOWN module; every
    expected module must be complete; every layer must be format-homogeneous.
    Anything else is refused BY NAME, never skipped.
    """
    if layout not in KNOWN_COMPONENTS:
        raise _fail(f"unknown layout {layout!r}")
    known = KNOWN_COMPONENTS[layout]
    expert_re = geometry.expert_re()
    main_layers, mtp_layer, num_experts = (
        geometry.main_routed_layers, geometry.mtp_layer, geometry.num_experts)
    modules: Dict[Tuple[int, int, str], set] = {}
    retained: List[str] = []
    stray: List[str] = []
    for name in weight_map:
        match = expert_re.match(name)
        if match is None:
            retained.append(name)
            continue
        layer, expert = int(match.group(1)), int(match.group(2))
        projection, component = match.group(3), match.group(4)
        if layer not in main_layers and layer != mtp_layer:
            stray.append(f"{name} (unexpected expert layer {layer})")
            continue
        if expert >= num_experts:
            stray.append(f"{name} (expert index {expert} >= {num_experts})")
            continue
        if component not in known:
            stray.append(f"{name} (unknown component {component!r} for layout {layout})")
            continue
        modules.setdefault((layer, expert, projection), set()).add(component)
    if stray:
        raise _fail(
            f"{len(stray)} expert tensors outside the declared {layout} scope, e.g. {stray[:4]}"
        )

    # per-module format from its component-set signature
    signatures = {
        frozenset(CT_NVFP4_DECODE): "nvfp4",
        frozenset(CT_NVFP4_DECODE) | frozenset(CT_NVFP4_ACTIVATION): "nvfp4",
        frozenset(CT_FP8_COMPONENTS): "fp8-scale-pair",
        frozenset(MO_NVFP4_DECODE): "nvfp4",
        frozenset(MO_NVFP4_DECODE) | frozenset(MO_NVFP4_ACTIVATION): "nvfp4",
        frozenset(("weight",)): "plain-weight",
    }
    per_layer_format: Dict[int, str] = {}
    activation_components: set = set()
    missing: List[str] = []
    for layer in main_layers + (mtp_layer,):
        formats = set()
        for expert in range(num_experts):
            for projection in PROJECTIONS:
                comps = modules.get((layer, expert, projection))
                if comps is None:
                    missing.append(geometry.component_name(layer, expert, projection, "<module>"))
                    continue
                fmt = signatures.get(frozenset(comps))
                if fmt is None:
                    raise _fail(
                        f"module {geometry.component_name(layer, expert, projection, '*')} carries an "
                        f"unrecognised component set {sorted(comps)}"
                    )
                formats.add(fmt)
                activation_components |= comps & (
                    set(CT_NVFP4_ACTIVATION) | set(MO_NVFP4_ACTIVATION)
                )
        if missing:
            raise _fail(f"{len(missing)} expert modules absent, e.g. {missing[:3]}")
        if len(formats) != 1:
            raise _fail(f"layer {layer} mixes expert formats {sorted(formats)} - refusing")
        per_layer_format[layer] = formats.pop()
    wrong_main = sorted(
        {layer for layer in main_layers if per_layer_format[layer] != "nvfp4"}
    )
    if wrong_main:
        raise _fail(
            f"main routed layers {wrong_main} are not NVFP4-packed "
            f"({[per_layer_format[l] for l in wrong_main]}) - this is not an NVFP4 "
            f"routed surface"
        )
    # A routed module shipping BOTH packed components and an unquantized
    # original is already refused above, by the component-set signature: the
    # combined set matches nothing in `signatures`, so it raises "unrecognised
    # component set" naming the module. There is deliberately no second check
    # here - `weight` is a legitimate component name in BOTH dialects (modelopt
    # packs INTO `weight`; compressed-tensors uses it for the MTP fp8 pair), so
    # a name-presence test could only ever be dead code pretending to be a gate.
    main_modules = len(main_layers) * num_experts * len(PROJECTIONS)
    mtp_modules = num_experts * len(PROJECTIONS)
    counts = {
        "expert_tensors": len(weight_map) - len(retained),
        "nvfp4_main_modules": main_modules,
        "mtp_modules": mtp_modules,
        "mtp_expert_format": per_layer_format[mtp_layer],
        "retained_tensors": len(retained),
        "activation_scale_components": sorted(activation_components),
        "per_layer_format": {str(k): v for k, v in sorted(per_layer_format.items())},
    }
    return retained, counts


def _verify_nonrouted_names(retained: List[str],
                            geometry: Nvfp4Geometry = GLM5NEXT_GEOMETRY) -> str:
    """Retained names vs the official BF16 non-routed name set (evidence file).

    The evidence file is DERIVED from the official BF16 index (Flash: revision
    a6c167b6, the same index the sealed inventory binds; flagship: 304b8051,
    the root dataset's weights pin); when it is present a strict set equality
    is enforced.  Without it (a stripped copy of this file), a structural
    census still gates count and anchors, and the receipt records which gate
    ran.
    """
    evidence_path = _EVIDENCE / geometry.official_nonrouted_evidence
    if evidence_path.is_file():
        evidence = _read_json(evidence_path, "official non-routed name evidence")
        expected = set(evidence["names"])
        got = set(retained)
        if got != expected:
            extra = sorted(got - expected)[:4]
            missing = sorted(expected - got)[:4]
            raise _fail(
                f"non-routed tensor names differ from the official BF16 set "
                f"(extra {extra}, missing {missing}) - the artifact's non-routed scope "
                f"is not the official one"
            )
        return "official-name-bijection"
    stack = geometry.stack
    anchors = (
        f"{stack}embed_tokens.weight",
        "lm_head.weight",
        f"{stack}norm.weight",
        f"{stack}layers.{geometry.mtp_layer}.eh_proj.weight",
    )
    absent = [name for name in anchors if name not in retained]
    if len(retained) != geometry.official_nonrouted_count or absent:
        raise _fail(
            f"structural non-routed census failed (count {len(retained)} != "
            f"{geometry.official_nonrouted_count}, absent anchors {absent}) and the "
            f"official-name evidence file is missing: {evidence_path}"
        )
    return "structural-census"


def _activation_disclosure(layout: str, declared: Optional[Dict[str, Any]],
                           scale_components: List[str]) -> Dict[str, Any]:
    """The receipt's activation-quantization caveat, stated from measured facts."""
    if declared is not None:
        detail = (
            "the artifact declares quantized input activations (verbatim config above); this "
            "lane decodes and scores the WEIGHTS only, so activation quantization is NOT "
            "captured by the measured KLD (same caveat family as the official FP8 releases)"
        )
        captured = False
    elif scale_components:
        detail = (
            "config declares input_activations null (a W4A16 label), yet per-module "
            "activation scale tensors ship in the artifact "
            f"({', '.join(scale_components)}); both facts recorded verbatim. This lane "
            "decodes weights only - if a serving stack applies activation quantization "
            "using those scales, that is not captured by the measured KLD"
        )
        captured = False
    else:
        detail = (
            "W4A16: the artifact quantizes weights only, so the weights-only decode "
            "captures the artifact fully (activations remain high-precision by declaration)"
        )
        captured = True
    return {
        "declared_input_activations": declared,
        "activation_scale_tensors_present": bool(scale_components),
        "activation_scale_components": list(scale_components),
        "weights_only_decode_captures_artifact_fully": captured,
        "disclosure": detail,
    }


def geometry_for_config(config: Mapping[str, Any]) -> Nvfp4Geometry:
    """The family geometry this config declares, cross-checked key by key.

    `model_type` picks the table row; every geometry key the row names must
    then agree verbatim (top level for a text-only release, `text_config` for
    a VL one), so a config that borrows a known model_type over a different
    stack is refused by the first key that differs, by name.
    """
    model_type = config.get("model_type")
    geometry = GEOMETRIES.get(str(model_type))
    if geometry is None:
        raise _fail(
            f"model_type {model_type!r} is not a family this surface knows "
            f"({', '.join(sorted(GEOMETRIES))}); its expert geometry would be a guess"
        )
    if list(config.get("architectures") or []) != list(geometry.architectures):
        raise _fail(
            f"architectures {config.get('architectures')!r} differ from "
            f"{list(geometry.architectures)} for model_type {model_type!r}"
        )
    text = geometry.text_config(config)
    for key, want in geometry.config_geometry:
        got = text.get(key)
        if got != want:
            raise _fail(
                f"nvfp4 checkpoint does not carry official {geometry.model_type} geometry: "
                f"{key}={got!r}, expected {want!r}"
            )
    return geometry


def modelopt_weight_declaration(quant: Mapping[str, Any]) -> Dict[str, Any]:
    """{num_bits, group_size, declared_by, input_activations} from a modelopt block.

    Two spellings ship: the compressed-tensors-shaped `config_groups.group_0`
    (RadixArk, incoai, LibertAIDAI) and a flat block with a top-level
    `group_size` beside `quant_algo` (Inferact, whose export writes its
    calibration recipe where the groups would be). Both are read; anything
    else is refused by name. NVFP4 is 4-bit by definition, so a flat block
    that states no `num_bits` declares 4.
    """
    groups = quant.get("config_groups")
    if isinstance(groups, Mapping):
        if sorted(groups) != ["group_0"]:
            raise _fail(f"unexpected modelopt config groups {sorted(groups)}")
        group = groups["group_0"] or {}
        weights = group.get("weights") or {}
        if (
            int(weights.get("num_bits", -1)) != 4
            or int(weights.get("group_size", -1)) != GROUP_SIZE
            or weights.get("dynamic") not in (False, None)
            or weights.get("type") != "float"
        ):
            raise _fail(
                f"group_0 weights are not static float 4-bit group-{GROUP_SIZE} NVFP4: "
                f"{dict(weights)}"
            )
        declared_activations = group.get("input_activations")
        return {
            "num_bits": 4, "group_size": GROUP_SIZE,
            "declared_by": "config_groups.group_0.weights",
            "weights": dict(weights),
            "input_activations": (dict(declared_activations)
                                  if isinstance(declared_activations, Mapping)
                                  else declared_activations),
        }
    if groups is not None:
        raise _fail(f"quantization_config.config_groups is {type(groups).__name__}, not a mapping")
    group_size = quant.get("group_size")
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise _fail(
            "modelopt quantization_config declares neither config_groups nor an integer "
            "top-level group_size; the weight format is undeclared"
        )
    if group_size != GROUP_SIZE:
        raise _fail(f"modelopt group_size {group_size} is not the NVFP4 group size {GROUP_SIZE}")
    num_bits = quant.get("num_bits", 4)
    if isinstance(num_bits, bool) or num_bits != 4:
        raise _fail(f"modelopt num_bits {num_bits!r} is not 4 (NVFP4)")
    # A flat block declares activations by `with_input_scale`; when true the
    # per-tensor input_scale ships and the artifact is W4A4 by declaration.
    with_input_scale = quant.get("with_input_scale")
    declared_activations = (
        {"dynamic": False, "num_bits": 4, "type": "float",
         "granularity": quant.get("input_scale_granularity")}
        if with_input_scale is True else None)
    return {
        "num_bits": 4, "group_size": GROUP_SIZE,
        "declared_by": "quantization_config.group_size",
        "weights": {"num_bits": 4, "group_size": GROUP_SIZE, "type": "float",
                    "dynamic": False},
        "input_activations": declared_activations,
    }


def modelopt_online_transforms(quant: Mapping[str, Any]) -> List[str]:
    """The MO_ONLINE_TRANSFORM_KEYS this block declares TRUE (empty = plain weights)."""
    return [key for key in MO_ONLINE_TRANSFORM_KEYS if quant.get(key) is True]


MODELOPT_DEQUANT_METHOD = "nvfp4-modelopt-dequant-to-bf16"
MODELOPT_ACTIVATION_SCHEME = "static-nvfp4-not-applied"


def modelopt_ignore_sha256(ignore: Any) -> str:
    """sha256 of the canonical JSON of the sorted, stringified `ignore` list.

    The list is the artifact's own statement of what it left native (231 to
    1,880 entries across the flagship exports); the CONTRACT carries its hash
    and count so the pod's plan and the controller's stdlib mirror can be
    compared for exact equality without shipping the list twice.
    """
    names = sorted(str(item) for item in (ignore or []))
    return _sha256_bytes(_canonical_json(names))


def modelopt_nvfp4_plan(config: Mapping[str, Any], weight_map: Mapping[str, Any]) -> Dict[str, Any]:
    """The decode-and-run plan for a modelopt NVFP4 checkpoint, from config + index alone.

    Returns ``{"contract": ..., "observed": ...}``.  ``contract`` is the
    `quantization_config` block the sealed runtime receipt records and
    `qualify_root` compares, key for key, against the job's candidate block
    (mirrored by `bin/measure_cloud._candidate_decode_plan`'s modelopt branch
    with stdlib only, so every value here derives from the config text).
    ``observed`` is what the index census found and rides on the decode
    evidence and the log line.

    Refuses by name: a quant_method other than modelopt, a quant_algo other
    than NVFP4, an undeclared or non-16 group size, any declared online weight
    transform, an unknown family geometry, a routed module whose component
    set is not the modelopt {weight, weight_scale, weight_scale_2}
    (+input_scale) layout, an expert layer outside the geometry, and a
    non-routed name set that is not the official BF16 release's.
    """
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping) or not quant:
        raise _fail("config.json has no quantization_config block")
    method = quant.get("quant_method")
    if method != "modelopt":
        raise _fail(f"quant_method {method!r} is not modelopt; this plan decodes the modelopt "
                    "NVFP4 dialect only")
    algo = quant.get("quant_algo")
    if algo != "NVFP4":
        raise _fail(f"modelopt quant_algo {algo!r} is not NVFP4")
    declaration = modelopt_weight_declaration(quant)
    transforms = modelopt_online_transforms(quant)
    if transforms:
        raise _fail(
            f"modelopt quantization_config declares online weight transforms {transforms}; "
            "a decode-and-run measurement would not apply them"
        )
    geometry = geometry_for_config(config)
    retained, counts = census_weight_map(weight_map, layout=LAYOUT_MODELOPT, geometry=geometry)
    nonrouted_verification = _verify_nonrouted_names(retained, geometry)
    producer = quant.get("producer")
    producer = ({"name": str(producer.get("name")), "version": str(producer.get("version"))}
                if isinstance(producer, Mapping) else None)
    ignore = quant.get("ignore") or []
    contract = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "num_bits": 4,
        "group_size": GROUP_SIZE,
        "weights_declared_by": declaration["declared_by"],
        # `input_scale` is an ACTIVATION quantity (the static per-tensor input
        # scale the serving kernel would apply to x, not to W); a weights-only
        # decode never touches it, and the receipt says so.
        "activation_scheme": MODELOPT_ACTIVATION_SCHEME,
        "producer": producer,
        "ignore_count": len(ignore),
        "ignore_sha256": modelopt_ignore_sha256(ignore),
    }
    observed = {
        "geometry": geometry.model_type,
        "layout": LAYOUT_MODELOPT,
        "quantized_modules": counts["nvfp4_main_modules"],
        "routed_layers": f"{geometry.main_routed_layers[0]}-{geometry.main_routed_layers[-1]}",
        "mtp_layer": geometry.mtp_layer,
        "mtp_expert_format": counts["mtp_expert_format"],
        "nonrouted_tensors": counts["retained_tensors"],
        "nonrouted_verification": nonrouted_verification,
        "activation_scale_components": counts["activation_scale_components"],
        "declared_input_activations": declaration["input_activations"],
        "online_transforms_declared": transforms,
    }
    return {"contract": contract, "observed": observed, "geometry": geometry}


def load_nvfp4_surface(
    root,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    require_shard_hashes: bool = True,
) -> Nvfp4Surface:
    root = Path(root).resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = _read_json(config_path, "nvfp4 config.json")
    index = _read_json(index_path, "nvfp4 model.safetensors.index.json")
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise _fail("config.json has no quantization_config block")

    method = quant.get("quant_method")
    groups = quant.get("config_groups")

    if method == "compressed-tensors":
        if not isinstance(groups, Mapping) or "group_0" not in groups:
            raise _fail("quantization_config carries no config_groups/group_0")
        weights = (groups["group_0"] or {}).get("weights") or {}
        declared_activations = (groups["group_0"] or {}).get("input_activations")
        layout = LAYOUT_COMPRESSED_TENSORS
        config_format = str(quant.get("format"))
        producer = {"quant_method": method, "version": quant.get("version")}
        if config_format not in ("nvfp4-pack-quantized", "mixed-precision"):
            raise _fail(
                f"compressed-tensors format {config_format!r} is not a known NVFP4 packing "
                "(expected nvfp4-pack-quantized or mixed-precision)"
            )
        extra_groups = sorted(set(groups) - {"group_0", "group_1"})
        if extra_groups:
            raise _fail(f"unexpected quantization config groups {extra_groups} - refusing")
        if "group_1" in groups:
            g1 = (groups["group_1"] or {}).get("weights") or {}
            if int(g1.get("num_bits", -1)) != 8 or g1.get("type") != "float":
                raise _fail(
                    f"config group_1 is not the known MTP fp8 block scheme: {dict(g1)}"
                )
        if (
            int(weights.get("num_bits", -1)) != 4
            or int(weights.get("group_size", -1)) != GROUP_SIZE
            or weights.get("dynamic") not in (False, None)
            or weights.get("symmetric") is not True
        ):
            raise _fail(
                f"group_0 weights are not static symmetric 4-bit group-{GROUP_SIZE} NVFP4: "
                f"{dict(weights)}"
            )
    elif method == "modelopt":
        layout = LAYOUT_MODELOPT
        config_format = str(quant.get("quant_algo"))
        producer = dict(quant.get("producer") or {})
        producer["quant_method"] = method
        if config_format != "NVFP4":
            raise _fail(f"modelopt quant_algo {config_format!r} is not NVFP4")
        declaration = modelopt_weight_declaration(quant)
        weights = declaration["weights"]
        declared_activations = declaration["input_activations"]
        transforms = modelopt_online_transforms(quant)
        if transforms:
            raise _fail(
                f"modelopt quantization_config declares online weight transforms "
                f"{transforms}; a decode-and-run measurement would not apply them"
            )
    else:
        raise _fail(
            f"quant_method {method!r} is neither compressed-tensors nor modelopt - "
            "not a supported NVFP4 dialect"
        )

    geometry = geometry_for_config(config)
    text = geometry.text_config(config)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise _fail("index has no weight_map")
    retained, counts = census_weight_map(weight_map, layout=layout, geometry=geometry)
    nonrouted_verification = _verify_nonrouted_names(retained, geometry)

    if revision is not None and _REVISION.fullmatch(revision) is None:
        raise _fail("--nvfp4-revision must be the immutable 40-hex repo commit")

    marker = root / "nvfp4-shards-verified.json"
    if marker.is_file():
        verified = _read_json(marker, "shard verification marker")
        manifest_path = root / "nvfp4-shard-manifest.json"
        if (
            verified.get("schema") != NVFP4_SHARDS_VERIFIED_SCHEMA
            or verified.get("all_verified") is not True
            or not manifest_path.is_file()
            or verified.get("manifest_sha256") != _sha256_file(manifest_path)
        ):
            raise _fail("stale/foreign nvfp4-shards-verified.json - re-run verify-shards")
        shard_hash_verification = "full"
    elif require_shard_hashes:
        raise _fail(
            "whole-shard sha256 verification marker absent: run "
            f"`python nvfp4_surface.py fetch-manifest --repo <repo> --revision <rev> --root {root}` "
            f"then `python nvfp4_surface.py verify-shards --root {root}`, or pass "
            "--nvfp4-skip-shard-hashes for a disclosed unverified read"
        )
    else:
        shard_hash_verification = "skipped"

    main = geometry.main_routed_layers
    scope = {
        "quantized_scope": "routed experts only (measured from the index, not the README)",
        "quantized_pattern": (
            f"{geometry.stack}layers.{{{main[0]}..{main[-1]}}}.mlp.experts."
            f"{{0..{geometry.num_experts - 1}}}.{{gate,up,down}}_proj -> nvfp4 e2m1 group-16"
        ),
        "mtp_expert_format": counts["mtp_expert_format"],
        "mtp_policy": (f"layer-{geometry.mtp_layer} experts present and identity-covered, "
                       "never executed"),
        "nonrouted_policy": (
            f"{geometry.official_nonrouted_count:,} non-routed tensors ship as plain BF16 "
            "under the official names in the artifact itself (embeddings, attention, "
            "shared experts, dense MLPs, norms, lm_head are NOT quantized in this artifact)"
        ),
        "nonrouted_verification": nonrouted_verification,
        "counts": {k: v for k, v in counts.items() if k != "per_layer_format"},
        "per_layer_format": counts["per_layer_format"],
    }
    activations = _activation_disclosure(
        layout,
        dict(declared_activations) if isinstance(declared_activations, Mapping) else declared_activations,
        counts["activation_scale_components"],
    )

    return Nvfp4Surface(
        root=root,
        repo=repo,
        revision=revision or "unpinned-local-snapshot",
        layout=layout,
        config_format=config_format,
        producer=producer,
        config_sha256=_sha256_file(config_path),
        index_sha256=_sha256_file(index_path),
        weight_map=dict(weight_map),
        retained_names=tuple(sorted(retained)),
        scope=scope,
        quant_weights=dict(weights),
        activations=activations,
        shard_hash_verification=shard_hash_verification,
        nonrouted_verification=nonrouted_verification,
        text_vocab_size=int(text["vocab_size"]),
        geometry=geometry,
    )


# ---------------------------------------------------------------------------
# expert source for the streaming slab (thread-safe: used from a read pool)
# ---------------------------------------------------------------------------

class Nvfp4ExpertSource:
    """One call returns (decoded fp32 CPU tensor, census row) per module.

    Mirrors ``NativeCheckpointSource``: the streamer's consumer does the
    device move, ``fuse_gate_up``, the SINGLE fp32->bf16 rounding, the
    ``copy_`` into the slab and the ``torch.equal`` close check - the packed
    lane's own installation algebra, unmodified.  safetensors handles are
    cached PER THREAD (the streamer reads with a pool and the handles are not
    documented thread-safe).
    """

    def __init__(self, surface: Nvfp4Surface):
        self.surface = surface
        self._local = threading.local()
        self._lock = threading.Lock()
        self.shards_read: set = set()
        self.bytes_read = 0
        self.decoded_modules = 0

    def _handle(self, shard: str):
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = self._local.handles = {}
        handle = cache.get(shard)
        if handle is None:
            from safetensors import safe_open

            path = self.surface.root / shard
            if not path.is_file():
                raise _fail(f"shard absent: {path}")
            handle = safe_open(str(path), framework="pt", device="cpu")
            enter = getattr(handle, "__enter__", None)
            if enter is not None:
                handle = enter()
            cache[shard] = handle
        return handle

    def _component(self, layer: int, expert: int, projection: str, component: str):
        name = self.surface.geometry.component_name(layer, expert, projection, component)
        shard = self.surface.weight_map.get(name)
        if shard is None:
            raise _fail(f"tensor not in weight_map: {name}")
        return name, shard, self._handle(shard).get_tensor(name)

    def load(self, *, layer: int, expert: int, projection: str):
        import torch

        surface = self.surface
        comps = surface.decode_components(layer)
        out_features, in_features = surface.geometry.projection_shape[projection]
        want = {
            comps[0]: (torch.uint8, (out_features, in_features // 2)),
            comps[1]: (None, (out_features, in_features // GROUP_SIZE)),  # f8, dtype-checked below
            comps[2]: (torch.float32, None),  # scalar: [] or [1]
        }
        loaded: Dict[str, Any] = {}
        components: Dict[str, Any] = {}
        nbytes = 0
        primary_shard = None
        for comp in comps:
            name, shard, tensor = self._component(layer, expert, projection, comp)
            want_dtype, want_shape = want[comp]
            if want_dtype is not None and tensor.dtype != want_dtype:
                raise _fail(f"{name} has dtype {tensor.dtype}, expected {want_dtype}")
            if comp == comps[1] and str(tensor.dtype) != "torch.float8_e4m3fn":
                raise _fail(f"{name} has dtype {tensor.dtype}, expected torch.float8_e4m3fn")
            if want_shape is not None and tuple(tensor.shape) != want_shape:
                raise _fail(f"{name} has shape {tuple(tensor.shape)}, expected {want_shape}")
            if want_shape is None and tuple(tensor.shape) not in ((), (1,)):
                raise _fail(f"{name} has shape {tuple(tensor.shape)}, expected a scalar")
            size = int(tensor.numel() * tensor.element_size())
            nbytes += size
            if primary_shard is None:
                primary_shard = shard
            loaded[comp] = tensor
            components[comp] = {
                "shard": shard,
                "sha256": _tensor_sha256(tensor),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": list(tensor.shape),
                "bytes": size,
            }
        if surface.layout == LAYOUT_COMPRESSED_TENSORS:
            decoded = dequant_nvfp4(
                loaded["weight_packed"], loaded["weight_scale"],
                weight_global_scale=loaded["weight_global_scale"],
            )
        else:
            decoded = dequant_nvfp4(
                loaded["weight"], loaded["weight_scale"],
                weight_scale_2=loaded["weight_scale_2"],
            )
        if tuple(decoded.shape) != (out_features, in_features):
            raise _fail(
                f"decoded {surface.geometry.official_name(layer, expert, projection)} has shape "
                f"{tuple(decoded.shape)}, expected {(out_features, in_features)}"
            )
        with self._lock:
            self.shards_read.update(row["shard"] for row in components.values())
            self.bytes_read += nbytes
            self.decoded_modules += 1
        row = {
            "tensor": surface.geometry.official_name(layer, expert, projection),
            "shard": primary_shard,
            "bytes": nbytes,
            "dtype": "float32",
            "format": f"nvfp4-e2m1-gs{GROUP_SIZE}-{self.surface.layout}",
            "components": components,
        }
        return decoded, row


# ---------------------------------------------------------------------------
# identity + verification helpers
# ---------------------------------------------------------------------------

def nvfp4_reader_identity(runner_path, surface: Nvfp4Surface) -> Dict[str, Any]:
    """Identity binding this adapter + the runner + the decode contract."""
    body = {
        "schema": NVFP4_READER_IDENTITY_SCHEMA,
        "mode": "offline_nvfp4_e2m1_group16_exact_fp32_dequant_to_bf16_for_logit_measurement",
        "serving_kernel": False,
        "bits": 4,
        "codebook": "e2m1-lut",
        "e2m1_magnitudes": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        "group_size": GROUP_SIZE,
        "layout": surface.layout,
        "scale_convention": (
            "weight_scale.f32 / weight_global_scale"
            if surface.layout == LAYOUT_COMPRESSED_TENSORS
            else "weight_scale.f32 * weight_scale_2"
        ),
        "nibble_order": "low_nibble_first",
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    body["runtime_reader_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


def surface_summary(surface: Nvfp4Surface) -> Dict[str, Any]:
    """The provenance block receipts embed (streaming_disclosure['nvfp4'])."""
    return {
        "schema": NVFP4_SURFACE_SCHEMA,
        "nvfp4_repo": surface.repo,
        "nvfp4_revision": surface.revision,
        "layout": surface.layout,
        "config_format": surface.config_format,
        "producer": surface.producer,
        "quant_weights": surface.quant_weights,
        "group_size": GROUP_SIZE,
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "scope_policy": surface.scope,
        "scope_census_sha256": surface.scope_census_sha256(),
        "activations": surface.activations,
        "shard_hash_verification": surface.shard_hash_verification,
        "seal_disclosure": SEAL_DISCLOSURE,
    }


def audit_expert_placement(
    surface: Nvfp4Surface,
    bf16_root,
    *,
    layer: int = 3,
    expert: int = 0,
    cosine_floor: float = 0.98,
) -> Dict[str, Any]:
    """Decoded expert vs the official BF16 weights: orientation/axis audit.

    NVFP4 has no TP slicing, but a silent packed-axis or nibble-order
    regression in a future export would land here before a single logit is
    captured: a correctly decoded 4-bit expert correlates > 0.999 with the
    official tensor, while any axis mistake collapses the cosine.
    """
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    reader = Nvfp4ExpertSource(surface)
    audit: Dict[str, Any] = {"layer": layer, "expert": expert, "projections": {}}
    for projection in PROJECTIONS:
        name = surface.geometry.official_name(layer, expert, projection)
        shard = bf16_index["weight_map"].get(name)
        if shard is None:
            raise _fail(f"official BF16 checkpoint lacks {name}")
        with safe_open(str(bf16_root / shard), framework="pt", device="cpu") as handle:
            official = handle.get_tensor(name).float()
        decoded, _ = reader.load(layer=layer, expert=expert, projection=projection)
        cosine = float(F.cosine_similarity(decoded.flatten(), official.flatten(), dim=0))
        rel_l2 = float((decoded - official).norm() / official.norm())
        ok = cosine > cosine_floor
        audit["projections"][projection] = {
            "cosine_vs_official_bf16": cosine,
            "rel_l2_vs_official_bf16": rel_l2,
            "passed": ok,
        }
        if not ok:
            raise _fail(
                f"placement audit FAILED for {name}: cosine {cosine:.4f} <= {cosine_floor} - "
                "the decode axis convention differs from the proven layout"
            )
    audit["passed"] = True
    return audit


def verify_nonrouted_tensors(
    surface: Nvfp4Surface,
    bf16_root,
    *,
    mode: str = "sample",
    sample_count: int = 64,
) -> Dict[str, Any]:
    """Non-routed tensors vs the official BF16 checkpoint (byte compare).

    modes: "full" byte-compares every retained tensor; "sample" byte-compares
    a deterministic subset and shape-checks the rest; "names" checks the name
    bijection only.  The name bijection is always enforced at surface load
    (against the committed official-name evidence); this adds bytes.
    """
    import numpy as np
    import torch
    from safetensors import safe_open

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    official_map: Mapping[str, str] = bf16_index["weight_map"]
    official_nonrouted = {
        name for name in official_map
        if re.search(r"\.mlp\.experts\.\d+\.", name) is None
    }
    if set(surface.retained_names) != official_nonrouted:
        extra = sorted(set(surface.retained_names) - official_nonrouted)[:3]
        missing = sorted(official_nonrouted - set(surface.retained_names))[:3]
        raise _fail(f"official/nvfp4 non-routed bijection differs (extra {extra}, missing {missing})")
    if mode not in ("full", "sample", "names"):
        raise _fail(f"unknown nonrouted verification mode: {mode}")
    result = {"mode": mode, "retained_tensors": len(surface.retained_names), "bijection_ok": True}
    if mode == "names":
        return result

    ours_handles: Dict[str, Any] = {}
    theirs_handles: Dict[str, Any] = {}

    def _tensor(root: Path, weight_map: Mapping[str, str], handles: Dict[str, Any], name: str):
        shard = weight_map[name]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(root / shard), framework="pt", device="cpu")
            handles[shard] = handle
        return handle.get_tensor(name)

    if mode == "full":
        chosen = list(surface.retained_names)
    else:
        anchors = [
            name
            for name in (
                f"{surface.geometry.stack}embed_tokens.weight",
                "lm_head.weight",
                f"{surface.geometry.stack}norm.weight",
            )
            if name in surface.retained_names
        ]
        rng = np.random.default_rng(0x4FB4)  # fixed seed: deterministic sample
        pool = sorted(set(surface.retained_names) - set(anchors))
        picks = rng.choice(len(pool), size=min(sample_count, len(pool)), replace=False)
        chosen = anchors + [pool[int(index)] for index in sorted(picks)]
    compared = 0
    for name in chosen:
        ours = _tensor(surface.root, surface.weight_map, ours_handles, name)
        theirs = _tensor(bf16_root, official_map, theirs_handles, name)
        if ours.dtype != theirs.dtype or tuple(ours.shape) != tuple(theirs.shape):
            raise _fail(f"non-routed tensor geometry differs from official: {name}")
        if not torch.equal(ours, theirs):
            raise _fail(f"non-routed tensor bytes differ from official BF16: {name}")
        compared += 1
        del ours, theirs
    result["byte_compared_tensors"] = compared
    result["all_equal"] = True
    return result


# ---------------------------------------------------------------------------
# shard pinning: HF LFS manifest fetch + local hash verification
# ---------------------------------------------------------------------------

def fetch_shard_manifest(repo: str, revision: str, root) -> Dict[str, Any]:
    """Pin every repo file's size + LFS sha256 from the public HF tree API.

    No token: the target repos are public.  The manifest is what
    ``verify_shard_hashes`` later checks local bytes against, giving this
    unsealed source the same fail-closed byte pinning the Dione lane gets
    from its in-repo exl3-manifest.json.
    """
    if _REVISION.fullmatch(revision) is None:
        raise _fail("fetch-manifest requires the immutable 40-hex repo commit")
    root = Path(root).resolve()
    files: List[Dict[str, Any]] = []
    url = (
        "https://huggingface.co/api/models/%s/tree/%s?recursive=true&expand=true"
        % (repo, revision)
    )
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": "quant-fidelity-suite"})
        with urllib.request.urlopen(request, timeout=120) as response:
            entries = json.loads(response.read().decode("utf-8"))
            link = response.headers.get("Link") or ""
        for entry in entries:
            if entry.get("type") != "file":
                continue
            lfs = entry.get("lfs") or {}
            files.append(
                {
                    "name": entry["path"],
                    "bytes": int(entry.get("size", -1)),
                    "sha256": lfs.get("oid"),
                }
            )
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
        url = next_url
    if not any(row["name"].endswith(".safetensors") for row in files):
        raise _fail(f"HF tree for {repo}@{revision} lists no safetensors shards")
    manifest = {
        "schema": NVFP4_SHARD_MANIFEST_SCHEMA,
        "repo": repo,
        "revision": revision,
        "generated_unix": int(time.time()),
        "source": "huggingface tree API (?recursive=true&expand=true; lfs.oid is the sha256)",
        "files": sorted(files, key=lambda row: row["name"]),
    }
    (root / "nvfp4-shard-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_shard_hashes(root) -> Dict[str, Any]:
    """Hash every local shard against nvfp4-shard-manifest.json; write marker."""
    root = Path(root).resolve()
    manifest_path = root / "nvfp4-shard-manifest.json"
    manifest = _read_json(manifest_path, "nvfp4-shard-manifest.json")
    if manifest.get("schema") != NVFP4_SHARD_MANIFEST_SCHEMA:
        raise _fail("nvfp4-shard-manifest.json carries the wrong schema")
    rows: List[Dict[str, Any]] = []
    started = time.monotonic()
    hashed = size_only = 0
    for entry in manifest.get("files", []):
        name = str(entry["name"])
        interesting = name.endswith(".safetensors") or name in (
            "config.json", "model.safetensors.index.json",
        )
        if not interesting:
            continue
        path = root / name
        if not path.is_file():
            raise _fail(f"file listed in the manifest is absent locally: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise _fail(f"size differs from the manifest: {path}")
        if entry.get("sha256"):
            observed = _sha256_file(path)
            if observed != entry["sha256"]:
                raise _fail(f"sha256 differs from the pinned LFS oid: {path}")
            hashed += 1
        else:
            size_only += 1
        rows.append({"file": name, "hashed": bool(entry.get("sha256"))})
    if hashed == 0:
        raise _fail("manifest pinned no LFS sha256s - refusing to write a verification marker")
    record = {
        "schema": NVFP4_SHARDS_VERIFIED_SCHEMA,
        "root": str(root),
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "manifest_sha256": _sha256_file(manifest_path),
        "files_hash_verified": hashed,
        "files_size_only": size_only,
        "all_verified": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "nvfp4-shards-verified.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


# ---------------------------------------------------------------------------
# standalone CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dry-run", help="validate a snapshot layout from config+index alone")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--skip-shard-hashes", action="store_true")

    p = sub.add_parser("fetch-manifest", help="pin file sizes + LFS sha256 from the HF tree API")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)

    p = sub.add_parser("verify-shards", help="hash local shards against nvfp4-shard-manifest.json")
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("probe", help="CPU decode + orientation audit vs official BF16")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--skip-shard-hashes", action="store_true")

    p = sub.add_parser("verify-nonrouted", help="byte-compare non-routed tensors vs official BF16")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "sample", "names"), default="sample")
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--skip-shard-hashes", action="store_true")

    args = parser.parse_args()
    if args.command == "fetch-manifest":
        manifest = fetch_shard_manifest(args.repo, args.revision, args.root)
        print(json.dumps({"ok": True, "files": len(manifest["files"]),
                          "repo": manifest["repo"], "revision": manifest["revision"]},
                         sort_keys=True))
        return 0
    if args.command == "verify-shards":
        record = verify_shard_hashes(args.root)
        print(json.dumps(record, sort_keys=True))
        return 0

    surface = load_nvfp4_surface(
        args.root,
        repo=getattr(args, "repo", None),
        revision=getattr(args, "revision", None),
        require_shard_hashes=not args.skip_shard_hashes,
    )
    summary = surface_summary(surface)
    summary["root"] = str(surface.root)
    summary["checkpoint_identity_sha256"] = surface.checkpoint_identity_sha256()
    if args.command == "probe":
        summary["placement_audit"] = audit_expert_placement(
            surface, args.bf16, layer=args.layer, expert=args.expert
        )
    if args.command == "verify-nonrouted":
        summary["nonrouted_verification_bytes"] = verify_nonrouted_tensors(
            surface, args.bf16, mode=args.mode
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    # A refusal is an ANSWER, not a crash: these tools are run by hand before a
    # paid capture, so `_fail`'s named message prints on one line instead of
    # under a traceback -- the same handler gguf_surface uses, and the same exit
    # code (2), so a caller can tell "this artifact is refused" from "the tool
    # broke".
    import sys

    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
