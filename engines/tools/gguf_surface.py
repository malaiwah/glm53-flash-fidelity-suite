#!/usr/bin/env python3
"""GGUF (llama.cpp) checkpoint surface adapter for the streaming K6 scorer.

Scores community GGUF quantizations of GLM-5.3-Flash (unsloth/GLM-5.3-Flash-GGUF,
ddh0/GLM-5.3-Flash-GGUF, ...) on OUR sealed 25-window panel through the SAME
single-device streaming capture (``stream_score.py --source gguf``) used for
K6/K8/native-BF16, so the number lands on the same yardstick.

CRITICAL SCOPE DIFFERENCE from the EXL3/TR3 and Dione families: a GGUF
quantizes (nearly) EVERYTHING -- token_embd, output (lm_head), all attention /
KDA / DSA projections, shared experts, dense MLPs AND the routed experts.  Only
norms, conv1d kernels, router gates and a few scalar tensors are stored F32.
This adapter therefore supplies ALL tensors from the GGUF itself:

  * routed experts (layers 3..44) are sliced per expert out of the fused
    ``blk.L.ffn_{gate,up,down}_exps.weight`` tensors and streamed-decoded to
    BF16 through the shared install algebra (fp32 dequant, ONE bf16 rounding);
  * every non-routed tensor is decoded once into a MATERIALIZED safetensors
    view under the official HF names, and the sealed ``from_pretrained``
    constructor runs over that view unmodified;
  * the official BF16 tree is consulted ONLY for config/tokenizer files and the
    vision tower (``model.visual.*``), which the main GGUF does not carry (it
    ships as a separate mmproj file).  The receipt discloses this.

Format facts (measured from the live repos, 2026-08-29, header-level; see
gguf-evidence/):

  * unsloth UD-Q4_K_XL @ 2975ab41: llama.cpp v3 split (6 files, each a complete
    GGUF with its own tensor table; shard 1 carries all KV metadata and zero
    tensors), arch ``glm5next``, 1412 tensors: Q8_0 645 / F32 638 / Q4_K 84 /
    Q5_K 42 / Q6_K 3.  ddh0 @ be335b19: single file, arch spelled
    ``glm5-next``, IQ3_S/IQ4_XS experts (v1-REFUSED by type, see below).
  * ggml dims are stored fastest-first; the row-major (= official HF) shape is
    ``reversed(dims)``.  Fused expert tensors are [in, out, 288] -> expert e is
    a contiguous [out, in] block-aligned slice.  That slot-e-is-expert-e
    assumption is PROVEN, not assumed: ``audit_expert_placement`` scores the
    decoded slot against the official BF16 expert and gets rel-L2 0.0714 (the
    Q4_K error) vs 1.42 for every row-shifted control, which settles the slot
    ordering, the reversed-dims orientation and the projection mapping at once.
  * MLA split: HF ``self_attn.kv_b_proj.weight`` [32768, 512] does NOT exist in
    the GGUF; llama.cpp stores ``attn_k_b.weight`` (dims [256,512,64] --
    PER-HEAD TRANSPOSED) and ``attn_v_b.weight`` (dims [512,256,64]).  The
    reconstruction (per head h: concat(transpose(k_b[h]), v_b[h]) along the
    row axis, heads stacked) was PROVEN against the official BF16 tensor:
    rel-L2 0.0054 (the Q8_0 error) for this arrangement vs >= 1.40 for every
    other candidate (k/v swapped, non-interleaved).  The full 64-head audit is
    recorded in gguf-evidence/mla-full-audit.json and the offline selftest
    re-runs ``audit_mla_placement`` itself on a real head window.
  * Tensors the official tree stores F32 (hc_*_base/scale, A_log, dt_bias,
    e_score_correction_bias) are stored F32 in the GGUF too and pass through
    BYTE-EXACTLY; every other decoded tensor gets a single fp32 -> bf16
    rounding, so a GGUF that stores norms F32 (llama.cpp's convention) is
    DOWNCAST to the official bf16 rather than widening the model.  The policy
    is not trusted: ``verify_official_dtypes`` reads the official safetensors
    headers wherever those shards are present and refuses any disagreement.

Dequantization is plain PyTorch (uint8-level unpack, fp32 accumulation, no
float64, no int64 beyond gather indices -- MPS-safe) and is BITWISE equal to
gguf-py 0.19.0's reference ``dequantize`` for every supported type, proven on
real ranged-fetched tensors including full expert slices with sub-block scales
(Q4_K/Q5_K) -- the offline selftest re-proves it from committed fixtures.

SUPPORTED ggml types: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K (v1, the Flash
lane) plus Q3_K, IQ4_XS, IQ3_XXS, IQ3_S (2026-09-05, for the GLM-5.3 flagship
UD-Q3_K_XL / UD-IQ4_XS builds; each proven bitwise against gguf-py 0.19.0 on
real ranged-fetched blocks, gguf-evidence/dequant_*_ggufpy_ref.npy).  Measured
against every build in the Flash repo (gguf-evidence/unsloth-build-census.json,
each build's own 1,412-tensor table), that set scores BF16, Q8_0, UD-IQ4_XS,
UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL and UD-Q6_K_XL and refuses the five IQ1/IQ2
builds.  Note the refusals are NOT predictable from the directory names:
unsloth's "Dynamic" recipe mixes IQ2_XS/IQ3_XXS/IQ4_XS into UD-Q2_K_XL and
IQ3_XXS/IQ4_XS into UD-Q3_K_XL.

Any unsupported type is REFUSED BY NAME AND TYPE at census time, before any
decode: adding a type means adding a kernel WITH the same bitwise-vs-gguf-py
proof, not silently skipping tensors.

TWO ARCHITECTURES, ONE DATA TABLE (``GgufArch``): ``glm5next`` (GLM-5.3-Flash,
above) and ``glm-dsa`` (the GLM-5.3 flagship, unsloth/GLM-5.3-GGUF @ 346b3591,
``GlmMoeDsaForCausalLM``): 78 decoder layers + MTP blk.78, dense layers 0-2,
256 routed experts of [2048, 6144], MLA with 64 heads x (nope 192 + v 256) over
kv_lora_rank 512 (kv_b_proj [28672, 512]), and a DSA indexer whose weights the
official tree carries on 22 "full" layers only -- the GGUF ships indexer
tensors on EVERY layer, the extra 285 being value-identical copies of the
preceding full layer's (stored BF16 where the parent is F32, so the proof
compares decoded values, never bytes).  Every mapping and layout choice for
glm-dsa was proven EXACTLY (uint16 bit patterns) against zai-org/GLM-5.3-BF16 @
304b8051 by HTTP range requests -- the composed kv_b_proj over all 64 heads,
q_a/q_b/kv_a/o projections, dense and shared FFNs, routed expert slots 0/128/
255, the indexer, embed/head/norm windows and the MTP block --
gguf-evidence/glmdsa-layout-audit.json; the tokenizer array order equals the
official vocab (glmdsa-tokenizer-order-audit.json).  ``materialize_layer`` is
the per-layer reader the layer-outer streamer (``gguf-dequant-to-bf16``)
calls: one decoder layer's tensors under their official names, decoded on the
capture device, kv_b composed, experts sliced, ONE bf16 rounding.

DISCLOSED DEVIATION - unsealed-source scoring: community GGUFs ship no encoder
receipts, no reconstruction closures and no sealed reader ABI.  The adapter
records whole-file sha256 for every GGUF it consumes plus the immutable repo
revision, and every receipt carries ``seal_disclosure`` saying exactly that
(the Dione precedent, engines/tools/dione_surface.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

GGUF_FORMAT = "glm53-gguf-llamacpp-v1"
GGUF_SURFACE_SCHEMA = "malaiwah.glm53-gguf-surface.v1"
GGUF_IDENTITY_SCHEMA = "malaiwah.glm53-gguf-student-identity.v1"
GGUF_READER_IDENTITY_SCHEMA = "malaiwah.glm53-gguf-offline-reader-identity.v1"
GGUF_FILES_VERIFIED_SCHEMA = "malaiwah.glm53-gguf-files-verified.v1"
GGUF_VIEW_RECEIPT_SCHEMA = "malaiwah.glm53-gguf-nonrouted-view-receipt.v1"
SEAL_DISCLOSURE = (
    "unsealed-source scoring: community GGUF releases ship no upstream encoder "
    "receipts, reconstruction closures or sealed reader ABI; the surface was "
    "decoded WITHOUT seal verification (whole-file sha256 of every consumed "
    "GGUF and the immutable repo revision are recorded instead)"
)
SCOPE_DISCLOSURE = (
    "the GGUF artifact quantizes non-routed tensors too (token_embd, lm_head, "
    "attention/KDA/DSA projections, shared experts, dense MLPs); this lane "
    "decodes ALL of them from the artifact -- the non-routed forward therefore "
    "measures the artifact, not the official BF16 tree. The vision tower is "
    "NOT in the main GGUF (separate mmproj file) and is sourced from the "
    "official BF16 tree; it is never executed by the text-only sealed panel"
)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPS_SUFFIX = {"gate_proj": "ffn_gate_exps.weight", "up_proj": "ffn_up_exps.weight",
               "down_proj": "ffn_down_exps.weight"}
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class GgufArch:
    """Everything this adapter knows about one llama.cpp architecture, as DATA.

    Two entries exist: ``glm5next`` (GLM-5.3-Flash, the original streaming
    lane) and ``glm-dsa`` (the GLM-5.3 flagship, `GlmMoeDsaForCausalLM`).  Every
    number here was read from the artifact headers and the official config and
    then PROVEN against the official BF16 tensors (module docstring;
    gguf-evidence/glmdsa-layout-audit.json for the flagship).  A GGUF whose
    ``general.architecture`` is not one of these keys is refused by name.
    """

    key: str                                  # general.architecture
    accepted_names: Tuple[str, ...]           # spellings seen across convert vintages
    family: str                               # human label
    layer_prefix: str                         # official HF stack path of the decoder layers
    top_level: Mapping[str, str]              # gguf name -> official name, top level
    block_count: int                          # decoder layers INCLUDING the MTP block
    dense_layers: Tuple[int, ...]             # leading dense (non-MoE) layers
    num_experts: int
    projection_shape: Mapping[str, Tuple[int, int]]   # per routed expert, [out, in]
    mla_heads: int
    mla_kv_lora_rank: int
    mla_k_nope: int                           # per-head rows of attn_k_b^T in kv_b_proj
    mla_v_dim: int                            # per-head rows of attn_v_b in kv_b_proj
    geometry_gate: Mapping[str, int]          # <arch>.<key> KV values that must match
    official_f32_suffixes: Tuple[str, ...]    # official-tree float32 tensors (passthrough)
    indexer_shared_copies: bool               # glm-dsa: GGUF ships indexer copies on shared layers

    @property
    def mtp_layer(self) -> int:
        return self.block_count - 1

    @property
    def routed_layers(self) -> Tuple[int, ...]:
        """Main routed layers: after the dense prefix, before the MTP block."""
        return tuple(range(len(self.dense_layers), self.mtp_layer))

    @property
    def kv_b_rows(self) -> int:
        return self.mla_heads * (self.mla_k_nope + self.mla_v_dim)

    def layer_name(self, layer: int, suffix: str) -> str:
        return f"{self.layer_prefix}.{layer}.{suffix}"

    def expert_name(self, layer: int, expert: int, projection: str) -> str:
        return self.layer_name(layer, f"mlp.experts.{expert}.{projection}.weight")

    def kv_b_name(self, layer: int) -> str:
        return self.layer_name(layer, "self_attn.kv_b_proj.weight")

    def official_dtype_for(self, hf_name: str) -> str:
        for suffix in self.official_f32_suffixes:
            if hf_name.endswith(suffix):
                return "float32"
        return "bfloat16"


GLM5NEXT = GgufArch(
    key="glm5next",
    accepted_names=("glm5next", "glm5-next"),
    family="GLM-5.3-Flash",
    layer_prefix="model.language_model.layers",
    top_level={
        "token_embd.weight": "model.language_model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "output_norm.weight": "model.language_model.norm.weight",
    },
    block_count=46,
    dense_layers=(0, 1, 2),
    num_experts=288,
    projection_shape={"gate_proj": (2048, 4096), "up_proj": (2048, 4096),
                      "down_proj": (4096, 2048)},
    mla_heads=64, mla_kv_lora_rank=512, mla_k_nope=256, mla_v_dim=256,
    geometry_gate={
        "block_count": 46, "expert_count": 288, "expert_used_count": 8,
        "leading_dense_block_count": 3, "embedding_length": 4096,
        "expert_feed_forward_length": 2048, "vocab_size": 154880,
        "nextn_predict_layers": 1,
    },
    # measured: 291 of 38,770 official tensors, exactly these suffix families
    official_f32_suffixes=(
        "hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale",
        "mlp.gate.e_score_correction_bias", "self_attn.A_log", "self_attn.dt_bias",
    ),
    indexer_shared_copies=False,
)

# GLM-5.3 flagship (zai-org/GLM-5.3-BF16 @ 304b8051, GlmMoeDsaForCausalLM).
# Geometry from the unsloth/GLM-5.3-GGUF headers (glm-dsa.* KVs) and the
# official config: 78 decoder layers + MTP blk.78, dense 0-2, 256 experts,
# MLA nope 192 / v 256 / 64 heads (attn_k_b dims [192,512,64], attn_v_b
# [512,256,64] -> kv_b_proj [64*(192+256)=28672, 512]).  Official float32
# tensors: only e_score_correction_bias (the audit read every other class as
# bf16, including the router gate, the norms and indexer.weights_proj that the
# GGUF widens to F32 -- all proven exactly representable and bit-equal).
GLM_DSA = GgufArch(
    key="glm-dsa",
    accepted_names=("glm-dsa",),
    family="GLM-5.3",
    layer_prefix="model.layers",
    top_level={
        "token_embd.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "output_norm.weight": "model.norm.weight",
    },
    block_count=79,
    dense_layers=(0, 1, 2),
    num_experts=256,
    projection_shape={"gate_proj": (2048, 6144), "up_proj": (2048, 6144),
                      "down_proj": (6144, 2048)},
    mla_heads=64, mla_kv_lora_rank=512, mla_k_nope=192, mla_v_dim=256,
    geometry_gate={
        "block_count": 79, "expert_count": 256, "expert_used_count": 8,
        "leading_dense_block_count": 3, "embedding_length": 6144,
        "expert_feed_forward_length": 2048, "vocab_size": 154880,
        "nextn_predict_layers": 1, "attention.q_lora_rank": 2048,
        "attention.kv_lora_rank": 512, "attention.key_length_mla": 256,
        "attention.value_length_mla": 256, "attention.head_count": 64,
        "attention.indexer.head_count": 32, "attention.indexer.key_length": 128,
    },
    official_f32_suffixes=("mlp.gate.e_score_correction_bias",),
    indexer_shared_copies=True,
)

ARCHITECTURES: Dict[str, GgufArch] = {name: arch for arch in (GLM5NEXT, GLM_DSA)
                                      for name in arch.accepted_names}


def arch_for(architecture: str) -> GgufArch:
    """The table entry for a ``general.architecture`` value; refuses by name."""
    arch = ARCHITECTURES.get(architecture)
    if arch is None:
        raise _fail(
            f"general.architecture is {architecture!r}, not one of "
            f"{tuple(ARCHITECTURES)} - not a GLM-5.3 GGUF this adapter knows"
        )
    return arch


# glm5next spellings, kept as module constants: the Flash streaming lane
# (stream_score.py, gguf_decode_bench.py, the offline selftest) addresses them.
MAIN_ROUTED_LAYERS = GLM5NEXT.routed_layers
MTP_LAYER = GLM5NEXT.mtp_layer
NUM_EXPERTS = GLM5NEXT.num_experts
PROJECTION_SHAPE = GLM5NEXT.projection_shape
ACCEPTED_ARCHITECTURES = GLM5NEXT.accepted_names

QK_K = 256
# (elements per block, bytes per block) from ggml's own type traits.
GGML_TYPE_IDS = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
                 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
                 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
                 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
                 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
                 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 36: "MXFP4"}
BLOCK_TRAITS = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34),
                "Q6_K": (256, 210), "Q5_K": (256, 176), "Q4_K": (256, 144),
                "Q3_K": (256, 110), "Q2_K": (256, 84), "Q5_0": (32, 22),
                "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_1": (32, 24),
                "IQ4_XS": (256, 136), "IQ4_NL": (32, 18), "IQ3_XXS": (256, 98),
                "IQ3_S": (256, 110), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74),
                "IQ2_S": (256, 82), "IQ1_S": (256, 50), "IQ1_M": (256, 56),
                "MXFP4": (32, 17)}
# Decode support: the types the unsloth Q8_0/UD-Q*_K_XL builds use (Flash and
# the GLM-5.3 flagship UD-Q4_K_XL), plus the four IQ/K types the flagship
# UD-Q3_K_XL and UD-IQ4_XS builds mix in.  Every kernel below is bitwise-equal
# to gguf-py 0.19.0 (selftest-proven on real ranged-fetched blocks).
SUPPORTED_TYPES = ("F32", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K",
                   "Q3_K", "IQ4_XS", "IQ3_XXS", "IQ3_S")

# Tensors the OFFICIAL BF16 tree stores as float32 (measured: 291 of 38,770,
# exactly these suffix families).  The GGUF stores them F32 too; the
# materialized view writes them F32 so the constructed model is dtype-identical
# to a native build.  Everything else is written bfloat16.
OFFICIAL_F32_SUFFIXES = GLM5NEXT.official_f32_suffixes

# PROVEN MLA reconstruction arrangement (see module docstring + selftest):
# per head h: rows [h*R .. h*R+nope-1] = transpose(attn_k_b[h]),
#             rows [h*R+nope .. h*R+R-1] = attn_v_b[h],  R = nope + v_dim.
# glm5next: nope = v = 256 (R 512, rel-L2 audit); glm-dsa: nope 192, v 256
# (R 448, EXACT equality against the official bf16 in the layout audit).
MLA_KV_B_ARRANGEMENT = "per_head_rows_kT_then_v"
MLA_HEADS = GLM5NEXT.mla_heads
MLA_KV_LORA_RANK = GLM5NEXT.mla_kv_lora_rank
MLA_HEAD_DIM = GLM5NEXT.mla_k_nope


def _fail(message: str) -> ValueError:
    return ValueError(f"gguf_surface: {message}")


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


# ---------------------------------------------------------------------------
# dequantization kernels -- plain torch, uint8-level, fp32, MPS-safe.
# Transliterated from gguf-py 0.19.0 gguf/quants.py with the SAME op order
# ((d*sc)*q - dmin*m for Q4_K/Q5_K), which is what makes them bitwise-equal to
# the reference dequantize().  Proven on real ranged-fetched tensors (module
# docstring); the offline selftest re-proves from committed fixtures.
# ---------------------------------------------------------------------------

def _f16cast(b):
    import torch

    return b.contiguous().view(torch.float16).to(torch.float32)


def dequant_q8_0(blocks):
    """uint8 [nb, 34] -> fp32 [nb, 32].  layout: f16 d | int8 qs[32]."""
    import torch

    d = _f16cast(blocks[:, :2])
    x = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    return x * d


def _get_scale_min(scales):
    """The 12-byte 6-bit packed scales/mins of Q4_K/Q5_K -> uint8 [nb, 8] x 2."""
    import torch

    s = scales.reshape(-1, 3, 4)
    d, m, m_d = s[:, 0, :], s[:, 1, :], s[:, 2, :]
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc, mn


def dequant_q4_k(blocks):
    """uint8 [nb, 144] -> fp32 [nb, 256].  f16 d | f16 dmin | 12B scales | 128B q."""
    import torch

    nb = blocks.shape[0]
    d = _f16cast(blocks[:, 0:2])
    dmin = _f16cast(blocks[:, 2:4])
    sc, mn = _get_scale_min(blocks[:, 4:16])
    dd = (d * sc.to(torch.float32)).reshape(nb, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(nb, -1, 1)
    qs = blocks[:, 16:].reshape(nb, 4, 1, 32)
    qs = torch.stack([qs >> 0, qs >> 4], dim=2).reshape(nb, 4, 2, 32)
    qs = (qs & 0x0F).reshape(nb, -1, 32).to(torch.float32)
    return (dd * qs - dm).reshape(nb, QK_K)


def dequant_q5_k(blocks):
    """uint8 [nb, 176] -> fp32 [nb, 256].  adds 32B of high bits over Q4_K."""
    import torch

    nb = blocks.shape[0]
    d = _f16cast(blocks[:, 0:2])
    dmin = _f16cast(blocks[:, 2:4])
    sc, mn = _get_scale_min(blocks[:, 4:16])
    dd = (d * sc.to(torch.float32)).reshape(nb, -1, 1)
    dm = (dmin * mn.to(torch.float32)).reshape(nb, -1, 1)
    qh = blocks[:, 16:48].reshape(nb, 1, 1, 32)
    qs = blocks[:, 48:].reshape(nb, 4, 1, 32)
    ql = torch.stack([qs >> 0, qs >> 4], dim=2).reshape(nb, 4, 2, 32)
    ql = (ql & 0x0F).reshape(nb, -1, 32)
    qh = torch.stack([qh >> i for i in range(8)], dim=2).reshape(nb, 1, 8, 32)
    qh = (qh & 0x01).reshape(nb, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)
    return (dd * q - dm).reshape(nb, QK_K)


def dequant_q6_k(blocks):
    """uint8 [nb, 210] -> fp32 [nb, 256].  128B ql | 64B qh | 16 int8 sc | f16 d."""
    import torch

    nb = blocks.shape[0]
    ql = blocks[:, 0:128].reshape(nb, 2, 1, 64)
    qh = blocks[:, 128:192].reshape(nb, 2, 1, 32)
    scales = blocks[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
    d = _f16cast(blocks[:, 208:210])
    dd = (d * scales).reshape(nb, QK_K // 16, 1)
    ql = torch.stack([ql >> 0, ql >> 4], dim=2).reshape(nb, 2, 2, 64)
    ql = (ql & 0x0F).reshape(nb, -1, 32)
    qh = torch.stack([qh >> 0, qh >> 2, qh >> 4, qh >> 6], dim=2).reshape(nb, 2, 4, 32)
    qh = (qh & 0x03).reshape(nb, -1, 32)
    q = (ql | (qh << 4)).view(torch.int8).to(torch.float32) - 32.0
    q = q.reshape(nb, QK_K // 16, -1)
    return (dd * q).reshape(nb, QK_K)


def dequant_q3_k(blocks):
    """uint8 [nb, 110] -> fp32 [nb, 256].  32B hmask | 64B qs | 12B scales | f16 d.

    The 16 six-bit scales are packed low nibbles first (8 bytes) then the two
    high bits of each (4 bytes); q = ql - 4*(1 - hmask_bit), per gguf-py.
    """
    import torch

    nb = blocks.shape[0]
    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    scales = blocks[:, 96:108]
    d = _f16cast(blocks[:, 108:110])
    lscales = torch.stack([scales[:, 0:8] >> 0, scales[:, 0:8] >> 4], dim=1).reshape(nb, 16)
    hscales = torch.stack([scales[:, 8:12] >> s for s in (0, 2, 4, 6)], dim=1).reshape(nb, 16)
    sc = ((lscales & 0x0F) | ((hscales & 0x03) << 4)).view(torch.int8).to(torch.float32) - 32.0
    dl = (d * sc).reshape(nb, 16, 1)
    ql = torch.stack([qs.reshape(nb, 2, 32) >> s for s in (0, 2, 4, 6)], dim=2)
    ql = (ql.reshape(nb, 16, 16) & 0x03)
    qh = torch.stack([hmask.reshape(nb, 1, 32) >> s for s in range(8)], dim=2)
    qh = ((qh.reshape(nb, 16, 16) & 0x01) ^ 0x01)
    q = (ql.view(torch.int8) - (qh << 2).view(torch.int8)).to(torch.float32)
    return (dl * q).reshape(nb, QK_K)


IQ4_NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def dequant_iq4_xs(blocks):
    """uint8 [nb, 136] -> fp32 [nb, 256].  f16 d | u16 scales_h | 4B scales_l | 128B qs.

    Eight 6-bit sub-block scales (4 low bits per nibble of scales_l, 2 high
    bits per pair in scales_h), 4-bit codes through the IQ4_NL codebook.
    """
    import torch

    nb = blocks.shape[0]
    d = _f16cast(blocks[:, 0:2])
    scales_h = blocks[:, 2:4].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    scales_l = torch.stack([blocks[:, 4:8] >> 0, blocks[:, 4:8] >> 4], dim=2).reshape(nb, 8) & 0x0F
    scales_h = torch.stack([(scales_h.reshape(nb) >> (2 * i)) & 0x03 for i in range(8)], dim=1)
    sc = (scales_l.to(torch.int32) | (scales_h << 4)).to(torch.int8).to(torch.float32) - 32.0
    dl = (d * sc).reshape(nb, 8, 1)
    qs = blocks[:, 8:].reshape(nb, 8, 1, 16)
    qs = torch.stack([qs >> 0, qs >> 4], dim=2).reshape(nb, 8, 32) & 0x0F
    kvalues = torch.tensor(IQ4_NL_KVALUES, dtype=torch.float32, device=blocks.device)
    vals = kvalues[qs.to(torch.long)]
    return (dl * vals).reshape(nb, QK_K)


# ksigns / grid tables copied from gguf-py 0.19.0 (gguf/quants.py, MIT), which
# copies them from ggml; each grid row is 4 codebook values, two per hex byte.
_IQ_KSIGNS = (
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
    b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
    b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
    b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
    b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
    b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
    b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
    b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
)
_IQ3_XXS_GRID_MAP = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)
_IQ3_XXS_GRID_HEX = (
    b"0000020004001100130017002000220031004200730075000101030110011201"
    b"2101250130013201410154017001000202020402110220022202310233023702"
    b"5102570275020103070310031203250370031304370444045704730475040105"
    b"0705320552053506640610071407160743076107011003101010121021102310"
    b"3010321034104710501000110211111120112211011203121012121221123012"
    b"7212001302132013311346136613011405145014201524154615711505162217"
    b"4017002002201120132020202220262031204220012103210521102112212121"
    b"3021632167217021002202221122172220222222372240225522012310231423"
    b"7023742335245324032527254125742501270327162745270130103012302130"
    b"2330503065307230003102312031313144314631013203321032253252327232"
    b"1133333330344734723400350635223555351436363663363337603704401740"
    b"3540374053405740744120423742404260426642074345430444514464442545"
    b"4345704505471047124730471250415070500051065126515551145232527252"
    b"0253535310542354275472540255315550562457425724604460466064602161"
    b"6161176264623063366344640565526533660367216703700570077010703270"
    b"5270267140711272457252720073157333736073217441740075027524753076"
)
_IQ3_S_GRID_MAP = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)
_IQ3_S_GRID_HEX = (
    b"0000010002000500070010001100120014001600200021002500330040004200"
    b"4500470051005300600062007100740077000001010102010401100111011501"
    b"2001230127013101350144016101650172010002010205020702100213021602"
    b"2102250230023402420245024702510253027002730203031103150320032203"
    b"3103330336034403500352036703710375030004130417042104240432044004"
    b"4304510470040205040520052205260533054105450547056605730506061106"
    b"1306310652067106000702070407200722072607330750075407001001100210"
    b"0410101011101310151017102010221031103410361054105610611072100011"
    b"0111031106111011141121113011331141115011521170117611001212121512"
    b"1712201224123212401243125512601272120113041307131013131321132713"
    b"3013341341136213701303140514121414143114331442144614501454140115"
    b"1015131521153015321551152016241627164416461601170317101712172117"
    b"3517411762177017002001200320052007201020122014201620212023202720"
    b"3020322041204320452050205220672070207320752000210221102113211721"
    b"2221252131213421422151210122042207222122232230223722412253225722"
    b"7122742200230223052311232223242331233323422350236623012407242024"
    b"2324322435244124722475240425112522253725402553257025002602260726"
    b"2126552661260527112726273027432750270230113013301530173022303130"
    b"3330353042304430473051306330713001310331053114312131233140316031"
    b"7231763100321232203232323432503201331033143321332333273330334133"
    b"4333473355337333033411341634223431345234603464340135103512352535"
    b"3235443556357335163641360137033720372237353700400440124020402440"
    b"2740324041405040704002410741114113412241304135414341514155410142"
    b"0342104215422142334240425742624270420443114313432043224331433543"
    b"0044024424443744404471440545074521456245134634466046104715473047"
    b"4347514702501050145022504050445047505250665074500151035105511251"
    b"2151325172510052115223523052365253520253075310532753445351536553"
    b"7353015404542054325446541255265551555355425602570457225711601360"
    b"1560316033606060006120612761646112623462426255626262706200631463"
    b"2163406325644364626400650365346560650566406611671367007004700770"
    b"2070227036704070547062700271117124714371457101720472107216722172"
    b"3072517202733273357353730174057413742074507422754275027631760077"
)
_IQ_GRID_CACHE: Dict[Tuple[str, str], Any] = {}


def _iq_grid(kind: str, device) -> Any:
    """The IQ3 codebook as an fp32 [entries, 4] tensor on `device` (cached).

    Decoded from the hex table exactly as gguf-py's ``init_grid`` does: each
    hex byte carries two 3-bit indices (low nibble, then high nibble), mapped through
    ``grid_map``.  256 entries for IQ3_XXS, 512 for IQ3_S.
    """
    import torch

    import numpy as np

    key = (kind, str(device))
    grid = _IQ_GRID_CACHE.get(key)
    if grid is None:
        hex_bytes, grid_map = {"IQ3_XXS": (_IQ3_XXS_GRID_HEX, _IQ3_XXS_GRID_MAP),
                               "IQ3_S": (_IQ3_S_GRID_HEX, _IQ3_S_GRID_MAP)}[kind]
        packed = np.frombuffer(bytes.fromhex(hex_bytes.decode("ascii")), dtype=np.uint8)
        codes = np.stack([packed & 0x07, (packed >> 4) & 0x07], axis=-1).reshape(-1)
        values = np.asarray(grid_map, dtype=np.float32)[codes].reshape(-1, 4)
        grid = torch.from_numpy(values.copy()).to(device)
        _IQ_GRID_CACHE[key] = grid
    return grid


def _iq_signs_from_ksigns(sign_index):
    """7-bit ksigns indices [..., 4] -> +-1.0 fp32 [..., 4, 8] (bit i of the
    looked-up byte is the sign of element i)."""
    import torch

    ksigns = torch.tensor(list(_IQ_KSIGNS), dtype=torch.uint8, device=sign_index.device)
    byte = ksigns[sign_index.to(torch.long)]
    bits = torch.stack([(byte >> i) & 0x01 for i in range(8)], dim=-1)
    return torch.where(bits == 0, torch.tensor(1.0, device=sign_index.device),
                       torch.tensor(-1.0, device=sign_index.device))


def dequant_iq3_xxs(blocks):
    """uint8 [nb, 98] -> fp32 [nb, 256].  f16 d | 64B qs (grid indices) | 8 x u32 scales.

    Per 32-element group one u32: bits 0..27 = four 7-bit ksigns indices, bits
    28..31 = the 4-bit sub-scale; db = d * (0.5 + s) * 0.5; value = db * grid * sign.
    """
    import torch

    nb = blocks.shape[0]
    d = _f16cast(blocks[:, 0:2])
    qs = blocks[:, 2:66]
    scales = blocks[:, 66:98].contiguous().view(torch.int32).reshape(nb, 8)
    # u32 semantics on an int32 view: shift as unsigned via masking
    sub = ((scales >> 28) & 0x0F).to(torch.float32)
    db = ((d * (0.5 + sub)) * 0.5).reshape(nb, 8, 1, 1)
    sign_index = torch.stack([(scales >> s) & 0x7F for s in (0, 7, 14, 21)], dim=-1)
    signs = _iq_signs_from_ksigns(sign_index).reshape(nb, 8, 4, 8)
    grid = _iq_grid("IQ3_XXS", blocks.device)[qs.reshape(nb, 8, 8).to(torch.long)]
    grid = grid.reshape(nb, 8, 4, 8)
    return ((db * grid) * signs).reshape(nb, QK_K)


def dequant_iq3_s(blocks):
    """uint8 [nb, 110] -> fp32 [nb, 256].  f16 d | 64B qs | 8B qh | 32B signs | 4B scales.

    9-bit grid indices (qs byte + one qh bit), raw sign bits, 4-bit sub-scales
    with db = d * (1 + 2*s).
    """
    import torch

    nb = blocks.shape[0]
    d = _f16cast(blocks[:, 0:2])
    qs = blocks[:, 2:66]
    qh = blocks[:, 66:74]
    sign_bytes = blocks[:, 74:106]
    scales = blocks[:, 106:110]
    sc = torch.stack([scales >> 0, scales >> 4], dim=2).reshape(nb, 8) & 0x0F
    db = (d * (1 + 2 * sc.to(torch.int32)).to(torch.float32)).reshape(nb, 8, 1, 1)
    bits = torch.stack([(sign_bytes >> i) & 0x01 for i in range(8)], dim=-1)
    signs = torch.where(bits == 0, torch.tensor(1.0, device=blocks.device),
                        torch.tensor(-1.0, device=blocks.device)).reshape(nb, 8, 4, 8)
    high = torch.stack([(qh.reshape(nb, 8, 1) >> i) & 0x01 for i in range(8)], dim=-1)
    index = qs.reshape(nb, 8, 8).to(torch.long) | (high.reshape(nb, 8, 8).to(torch.long) << 8)
    grid = _iq_grid("IQ3_S", blocks.device)[index].reshape(nb, 8, 4, 8)
    return ((db * grid) * signs).reshape(nb, QK_K)


def dequant_bytes(ggml_type: str, raw: bytes, n_elements: int, device=None):
    """Decode a block-aligned byte string of `ggml_type` to a flat fp32 tensor.

    ``device`` moves the raw UINT8 BUFFER before any kernel runs, so the whole
    decode happens there and the result is already resident.  The kernels above
    are not rewritten for it -- they are the same lines, in the same order, on a
    tensor that happens to live somewhere else.

    Why that keeps the bitwise property.  Every operation in these kernels is
    either an integer op on uint8/int8 (shift, mask, or, reinterpret) or an IEEE
    754 binary32 multiply or subtract on an elementwise-shaped operand.  Both
    classes are exactly specified and device-independent: there is no reduction,
    no matmul (so no TF32 path), and in eager mode no fusion that could turn the
    ``dd * q - dm`` pair into an FMA with a different rounding.  Proven, not
    assumed: `engines/tools/selftest_gguf_offline.py` rung 1b re-decodes the same
    committed REAL ranged-fetched bytes on whatever accelerator the host has and
    demands ``torch.equal`` against the CPU output that rung 1 already proved
    bitwise-equal to gguf-py 0.19.0.  On this laptop that is MPS; on the rented
    box it is CUDA, and it runs there BEFORE the capture, which is the only
    place the check is worth anything.

    Speed is the point.  Measured on the box (docs/GGUF-MEASUREMENT.md), the CPU
    path costs ~39 ms per expert matrix while the GPU sits at 2-4% -- and the
    fp32 result it then has to hand over is 33.5 MB per matrix, 7.1x the 4.7 MB
    of quantized bytes this path sends instead.
    """
    import torch

    if ggml_type not in SUPPORTED_TYPES:
        raise _fail(
            f"ggml type {ggml_type} has no v1 decode kernel (supported: "
            f"{', '.join(SUPPORTED_TYPES)}); adding one requires the same "
            "bitwise-vs-gguf-py proof the shipped kernels carry"
        )
    per_block, block_bytes = BLOCK_TRAITS[ggml_type]
    if n_elements % per_block:
        raise _fail(f"{ggml_type}: {n_elements} elements is not block-aligned ({per_block})")
    expected = n_elements // per_block * block_bytes
    if len(raw) != expected:
        raise _fail(f"{ggml_type}: got {len(raw)} bytes, expected {expected}")
    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    if device is not None and str(device) != "cpu":
        # The reinterpreting `.view(torch.float16)` inside `_f16cast` needs a
        # contiguous source; `.to()` of a contiguous uint8 tensor stays one.
        buf = buf.to(device)
    if ggml_type == "F32":
        return buf.view(torch.float32).clone()
    if ggml_type == "F16":
        return buf.view(torch.float16).to(torch.float32)
    if ggml_type == "BF16":
        return buf.view(torch.bfloat16).to(torch.float32)
    blocks = buf.reshape(-1, block_bytes)
    if ggml_type == "Q8_0":
        return dequant_q8_0(blocks).reshape(-1)
    if ggml_type == "Q4_K":
        return dequant_q4_k(blocks).reshape(-1)
    if ggml_type == "Q5_K":
        return dequant_q5_k(blocks).reshape(-1)
    if ggml_type == "Q6_K":
        return dequant_q6_k(blocks).reshape(-1)
    if ggml_type == "Q3_K":
        return dequant_q3_k(blocks).reshape(-1)
    if ggml_type == "IQ4_XS":
        return dequant_iq4_xs(blocks).reshape(-1)
    if ggml_type == "IQ3_XXS":
        return dequant_iq3_xxs(blocks).reshape(-1)
    if ggml_type == "IQ3_S":
        return dequant_iq3_s(blocks).reshape(-1)
    raise _fail(f"unreachable: {ggml_type}")


# ---------------------------------------------------------------------------
# GGUF container parsing (v2/v3, llama.cpp split convention)
# ---------------------------------------------------------------------------

class _NeedMoreData(Exception):
    pass


class _Reader:
    def __init__(self, buf: bytes):
        self.b, self.o = buf, 0

    def _take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise _NeedMoreData()
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def string(self) -> str:
        return self._take(self.u64()).decode("utf8", "replace")

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
               10: "<Q", 11: "<q", 12: "<d"}.get(t)
        if fmt is None:
            raise _fail(f"unknown GGUF metadata value type {t}")
        return struct.unpack(fmt, self._take(struct.calcsize(fmt)))[0]


def parse_gguf_header(buf: bytes) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Parse one GGUF file's header from its leading bytes.

    Raises _NeedMoreData when `buf` is too short -- callers read progressively.
    Returns (info, kv, tensor_rows); tensor offsets are relative to the file's
    data_start (= align(header_end, general.alignment)).
    """
    if buf[:4] != b"GGUF":
        raise _fail("not a GGUF file (magic differs)")
    r = _Reader(buf)
    r.o = 4
    version = r.u32()
    if version not in (2, 3):
        raise _fail(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    kv: Dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        kv[key] = r.value(vtype)
    tensors: List[Dict[str, Any]] = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        type_id = r.u32()
        offset = r.u64()
        ggml_type = GGML_TYPE_IDS.get(type_id, f"UNKNOWN_{type_id}")
        n = 1
        for d in dims:
            n *= d
        traits = BLOCK_TRAITS.get(ggml_type)
        nbytes = (n // traits[0] * traits[1]) if traits and n % traits[0] == 0 else None
        tensors.append({"name": name, "dims": dims, "type": ggml_type,
                        "elements": n, "offset": offset, "bytes": nbytes})
    alignment = int(kv.get("general.alignment", 32))
    data_start = (r.o + alignment - 1) // alignment * alignment
    info = {"version": version, "kv_count": n_kv, "tensor_count": n_tensors,
            "header_end": r.o, "alignment": alignment, "data_start": data_start}
    return info, kv, tensors


class GgufFile:
    """One GGUF file, local path or https URL (URLs are metadata/audit-only)."""

    _HEADER_STEPS = (1 << 19, 1 << 21, 1 << 23, 1 << 25, 1 << 26)

    def __init__(self, location: str):
        self.location = str(location)
        self.remote = self.location.startswith("http://") or self.location.startswith("https://")
        self.name = self.location.rsplit("/", 1)[-1]
        self._lock = threading.Lock()
        self._local_fd: Optional[int] = None
        self.size = self._probe_size()
        buf = b""
        for step in self._HEADER_STEPS:
            buf = self._read_absolute(0, min(step, self.size))
            try:
                self.info, self.kv, tensor_rows = parse_gguf_header(buf)
                break
            except _NeedMoreData:
                if step >= self.size or step == self._HEADER_STEPS[-1]:
                    raise _fail(f"{self.name}: header larger than {step} bytes or truncated")
        self.tensors: Dict[str, Dict[str, Any]] = {}
        for row in tensor_rows:
            if row["name"] in self.tensors:
                raise _fail(f"{self.name}: duplicate tensor {row['name']}")
            row["file"] = self.name
            self.tensors[row["name"]] = row

    # -- IO ---------------------------------------------------------------
    def _probe_size(self) -> int:
        if not self.remote:
            path = Path(self.location)
            if not path.is_file():
                raise _fail(f"GGUF file absent: {path}")
            return path.stat().st_size
        import urllib.request

        req = urllib.request.Request(self.location, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            content_range = resp.headers.get("Content-Range", "")
        match = re.match(r"bytes \d+-\d+/(\d+)", content_range)
        if match is None:
            raise _fail(f"{self.location}: no Content-Range on a ranged request")
        return int(match.group(1))

    def _read_absolute(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if self.remote:
            import urllib.request

            req = urllib.request.Request(
                self.location, headers={"Range": "bytes=%d-%d" % (start, start + length - 1)}
            )
            data = urllib.request.urlopen(req, timeout=600).read()
        else:
            with self._lock:
                if self._local_fd is None:
                    self._local_fd = os.open(self.location, os.O_RDONLY)
            data = os.pread(self._local_fd, length, start)
        if len(data) != length:
            raise _fail(f"{self.name}: short read at {start} ({len(data)} of {length})")
        return data

    def read_tensor_range(self, tensor_name: str, rel_offset: int, length: int) -> bytes:
        row = self.tensors.get(tensor_name)
        if row is None:
            raise _fail(f"{self.name} has no tensor {tensor_name}")
        if row["bytes"] is not None and rel_offset + length > row["bytes"]:
            raise _fail(f"{tensor_name}: range {rel_offset}+{length} exceeds {row['bytes']}")
        return self._read_absolute(self.info["data_start"] + row["offset"] + rel_offset, length)

    # -- split KVs --------------------------------------------------------
    @property
    def split_no(self) -> Optional[int]:
        value = self.kv.get("split.no")
        return None if value is None else int(value)

    @property
    def split_count(self) -> Optional[int]:
        value = self.kv.get("split.count")
        return None if value is None else int(value)


class GgufContainer:
    """A complete artifact: one single-file GGUF or every part of a v3 split."""

    def __init__(self, files: List[GgufFile]):
        if not files:
            raise _fail("no GGUF files given")
        arch = {f.kv.get("general.architecture") for f in files if "general.architecture" in f.kv}
        arch.discard(None)
        if len(arch) != 1:
            raise _fail(f"files disagree on general.architecture: {sorted(arch)}")
        self.architecture = arch.pop()
        counts = {f.split_count for f in files}
        if counts == {None}:
            if len(files) != 1:
                raise _fail(f"{len(files)} files given but none carries split.count")
            self.files = files
        else:
            if None in counts or len(counts) != 1:
                raise _fail("split.count differs (or is absent) across the given files")
            want = counts.pop()
            if len(files) != want:
                got = sorted(f.name for f in files)
                raise _fail(
                    f"split artifact needs all {want} parts, got {len(files)}: {got} "
                    "(pass every .gguf of the split)"
                )
            by_no = {f.split_no: f for f in files}
            if sorted(by_no) != list(range(want)):
                raise _fail(f"split.no values {sorted(by_no)} do not cover 0..{want - 1}")
            self.files = [by_no[i] for i in range(want)]
        declared = {int(f.kv["split.tensors.count"]) for f in self.files
                    if "split.tensors.count" in f.kv}
        # the union tensor table (names must be disjoint across parts)
        self.tensors: Dict[str, Dict[str, Any]] = {}
        for f in self.files:
            overlap = set(f.tensors) & set(self.tensors)
            if overlap:
                raise _fail(f"tensor(s) present in two split parts: {sorted(overlap)[:3]}")
            self.tensors.update(f.tensors)
        if declared and declared != {len(self.tensors)}:
            raise _fail(
                f"split.tensors.count declares {sorted(declared)} but the union table has "
                f"{len(self.tensors)} tensors"
            )
        # KV lives on the part that carries the full metadata (split.no 0 /
        # single file); every part contributes what it has, first part wins.
        self.kv: Dict[str, Any] = {}
        for f in reversed(self.files):
            self.kv.update(f.kv)
        self._by_file = {f.name: f for f in self.files}
        self.remote = any(f.remote for f in self.files)

    def geometry_value(self, key: str):
        return self.kv.get(f"{self.architecture}.{key}")

    def read_tensor_range(self, tensor_name: str, rel_offset: int, length: int) -> bytes:
        row = self.tensors.get(tensor_name)
        if row is None:
            raise _fail(f"artifact has no tensor {tensor_name}")
        return self._by_file[row["file"]].read_tensor_range(tensor_name, rel_offset, length)

    def read_tensor(self, tensor_name: str) -> bytes:
        row = self.tensors.get(tensor_name)
        if row is None:
            raise _fail(f"artifact has no tensor {tensor_name}")
        if row["bytes"] is None:
            raise _fail(f"{tensor_name}: byte size underivable for type {row['type']}")
        return self.read_tensor_range(tensor_name, 0, row["bytes"])


# ---------------------------------------------------------------------------
# header-level identity + contract (STDLIB ONLY: the controller imports these)
# ---------------------------------------------------------------------------

GGUF_DECODE_METHOD = "gguf-dequant-to-bf16"
#: What the layer-outer `gguf-dequant-to-bf16` lane binds into
#: `weights_decode.quantization_config`: everything a GGUF declares about its
#: own quantization, read from the header bytes.  The controller (range
#: requests) and the pod (local files) MUST compute the identical block from
#: the same headers -- `qualify-root` compares them field for field.
GGUF_CONTRACT_KV = ("general.architecture", "general.file_type",
                    "general.quantization_version", "general.quantized_by",
                    "quantize.imatrix.file", "quantize.imatrix.dataset",
                    "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count")


def tensor_table_sha256(container: GgufContainer) -> str:
    """sha256 over the canonical JSON of every tensor's (name, dims, type, offset,
    file): the GGUF analogue of a safetensors index digest.  Header CONTENT, never
    container bytes, so it is the same on the controller and on the pod."""
    rows = [{"name": n, "dims": [int(d) for d in r["dims"]], "type": r["type"],
             "offset": int(r["offset"]), "file": r["file"]}
            for n, r in sorted(container.tensors.items())]
    return _sha256_bytes(_canonical_json(rows))


def decode_contract(container: GgufContainer, build: str) -> Dict[str, Any]:
    """The `weights_decode` block of the gguf lane, from headers alone.

    ``build`` is the repo-relative directory of the variant (e.g. ``UD-Q4_K_XL``,
    the ``--path`` of the plan), so two builds of one repo revision never share
    a contract.
    """
    type_census: Dict[str, int] = {}
    for row in container.tensors.values():
        type_census[row["type"]] = type_census.get(row["type"], 0) + 1
    kv = {key: container.kv[key] for key in GGUF_CONTRACT_KV if key in container.kv}
    return {
        "method": GGUF_DECODE_METHOD,
        "quantization_config": {
            "container": "gguf",
            "build": build,
            "files": sorted(f.name for f in container.files),
            "general": kv,
            "type_census": dict(sorted(type_census.items())),
            "tensor_count": len(container.tensors),
            "tensor_table_sha256": tensor_table_sha256(container),
            "decode": "every tensor block-dequantized to fp32 by the gguf-py-proven "
                      "kernels on the capture device, then ONE rounding to bfloat16 "
                      "(official-float32 tensors kept fp32); attn_k_b/attn_v_b composed "
                      "into kv_b_proj; fused experts sliced per expert",
        },
    }


def audit_container(container: GgufContainer) -> Dict[str, Any]:
    """Every tensor's bytes lie inside its file, and no two overlap.

    The GGUF analogue of `layer_outer.audit_checkpoint_tree`: a truncated part
    would otherwise read as a short read (refused at read time) or, worse, a
    row whose offset points past the data would decode garbage.  Local files
    only (sizes are stat'ed); remote containers are audited by range failures.
    """
    by_file: Dict[str, List[Tuple[int, int, str]]] = {}
    for name, row in container.tensors.items():
        if row["bytes"] is None:
            raise _fail(f"{name}: byte size underivable for type {row['type']}")
        by_file.setdefault(row["file"], []).append((int(row["offset"]), int(row["bytes"]), name))
    total = 0
    for f in container.files:
        extents = sorted(by_file.get(f.name, []))
        end = f.info["data_start"]
        for offset, nbytes, name in extents:
            start = f.info["data_start"] + offset
            if start < end:
                raise _fail(f"{f.name}: {name} overlaps the previous tensor's bytes")
            end = start + nbytes
            total += nbytes
        # a metadata-only part (llama.cpp's split 1) ends at its header, which
        # may sit BEFORE the aligned data_start; only tensor bytes are bounded
        if extents and end > f.size:
            raise _fail(
                f"{f.name}: tensor extents run to byte {end} but the file has {f.size} "
                "(truncated part?)"
            )
    return {"files": len(container.files), "tensors": len(container.tensors),
            "tensor_bytes": total, "file_bytes": sum(f.size for f in container.files),
            "extents_ok": True}


def tokenizer_matches(kv: Mapping[str, Any], tokenizer_json: bytes) -> Dict[str, Any]:
    """Does the GGUF's embedded vocabulary equal an HF tokenizer.json's, by ID?

    A GGUF ships no tokenizer files: the token strings live in
    ``tokenizer.ggml.tokens`` (index = id) and the BPE merges in
    ``tokenizer.ggml.merges``.  The lane runs the reference root's tokenizer
    files (the panel is already tokenized), so it must be SHOWN that the
    artifact's own vocabulary is the same one: every id the HF vocab (model +
    added tokens) defines must carry the same string, the merges must be the
    same list in the same order, and ids beyond the HF vocab may only be
    llama.cpp's ``[PAD<id>]`` fillers up to the declared vocab_size.  Refuses
    on any other difference, naming the first.
    """
    doc = json.loads(tokenizer_json.decode("utf-8"))
    vocab = dict((doc.get("model") or {}).get("vocab") or {})
    by_id: Dict[int, str] = {int(i): s for s, i in vocab.items()}
    for added in doc.get("added_tokens") or []:
        by_id.setdefault(int(added["id"]), added["content"])
    tokens = list(kv.get("tokenizer.ggml.tokens") or [])
    if not tokens:
        raise _fail("the GGUF carries no tokenizer.ggml.tokens array")
    mismatched = [i for i in range(len(tokens)) if i in by_id and by_id[i] != tokens[i]]
    if mismatched:
        i = mismatched[0]
        raise _fail(
            f"REFUSED: GGUF token id {i} is {tokens[i]!r} but the HF tokenizer says "
            f"{by_id[i]!r} ({len(mismatched)} ids differ)"
        )
    missing = [i for i in by_id if i >= len(tokens)]
    if missing:
        raise _fail(f"REFUSED: HF tokenizer defines id {min(missing)} beyond the GGUF's "
                    f"{len(tokens)} tokens")
    pads = [i for i in range(len(tokens)) if i not in by_id]
    bad_pads = [i for i in pads if tokens[i] != "[PAD%d]" % i]
    if bad_pads:
        raise _fail(f"REFUSED: GGUF token id {bad_pads[0]} = {tokens[bad_pads[0]]!r} is "
                    "absent from the HF tokenizer and is not a [PAD<id>] filler")
    hf_merges = [(m if isinstance(m, str) else " ".join(m))
                 for m in (doc.get("model") or {}).get("merges") or []]
    gg_merges = list(kv.get("tokenizer.ggml.merges") or [])
    if hf_merges != gg_merges:
        raise _fail(f"REFUSED: BPE merges differ (HF {len(hf_merges)}, GGUF {len(gg_merges)})")
    return {"tokens": len(tokens), "hf_ids": len(by_id), "pad_fillers": len(pads),
            "merges": len(gg_merges), "pre": kv.get("tokenizer.ggml.pre"),
            "model": kv.get("tokenizer.ggml.model"), "equal": True}


# ---------------------------------------------------------------------------
# llama.cpp -> HF name map
# ---------------------------------------------------------------------------
# Suffix rules verified by BIJECTION against the official BF16 index (glm5next:
# 38,770 tensors, revision a6c167b6...; glm-dsa: 59,585 tensors, revision
# 304b8051...) in the offline selftest: every GGUF tensor is consumed exactly
# once, every official non-routed non-vision tensor is produced exactly once,
# and reversed(ggml dims) equals the official shape for every 1:1 tensor.  The
# per-layer suffix map is shared by both architectures (glm-dsa simply never
# ships the KDA/mHC/kpool names); the top level and the layer prefix come from
# the arch table.
_TOP_LEVEL = GLM5NEXT.top_level
_LAYER_DIRECT = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "ssm_beta.weight": "self_attn.b_proj.weight",
    "ssm_f_a.weight": "self_attn.f_a_proj.weight",
    "ssm_f_b.weight": "self_attn.f_b_proj.weight",
    "ssm_g_a.weight": "self_attn.g_a_proj.weight",
    "ssm_g_b.weight": "self_attn.g_b_proj.weight",
    "ssm_conv1d_q.weight": "self_attn.q_conv1d.weight",
    "ssm_conv1d_k.weight": "self_attn.k_conv1d.weight",
    "ssm_conv1d_v.weight": "self_attn.v_conv1d.weight",
    "ssm_a": "self_attn.A_log",
    "ssm_dt.bias": "self_attn.dt_bias",
    "ssm_norm.weight": "self_attn.o_norm.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer_compressor_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
    "indexer_compressor_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
    # ddh0's convert vintage spells the same two tensors differently (same
    # dims); one artifact carries one spelling, never both -- build_census
    # refuses a duplicate HF target.
    "indexer.kpool_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
    "indexer.kpool_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.shared_head_norm.weight": "shared_head.norm.weight",
}
_ROUTED = {"ffn_gate_exps.weight": "gate_proj", "ffn_up_exps.weight": "up_proj",
           "ffn_down_exps.weight": "down_proj"}
_MLA = {"attn_k_b.weight": "k_b", "attn_v_b.weight": "v_b"}
_BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def classify_tensor(gguf_name: str, arch: GgufArch = GLM5NEXT) -> Tuple[str, ...]:
    """One GGUF tensor name -> its role.

    Returns one of
      ("top", hf_name)                       -- 1:1, top level
      ("direct", layer, hf_name)             -- 1:1, per layer
      ("routed", layer, projection)          -- fused expert tensor
      ("mla", layer, "k_b"|"v_b")            -- half of kv_b_proj
      ("unmapped",)                          -- census REFUSES these by name
    """
    if gguf_name in arch.top_level:
        return ("top", arch.top_level[gguf_name])
    match = _BLK.match(gguf_name)
    if match is None:
        return ("unmapped",)
    layer, suffix = int(match.group(1)), match.group(2)
    if suffix in _LAYER_DIRECT:
        return ("direct", layer, arch.layer_name(layer, _LAYER_DIRECT[suffix]))
    if suffix in _ROUTED:
        return ("routed", layer, _ROUTED[suffix])
    if suffix in _MLA:
        return ("mla", layer, _MLA[suffix])
    return ("unmapped",)


def routed_tensor_name(layer: int, projection: str) -> str:
    return f"blk.{layer}.{EXPS_SUFFIX[projection]}"


def official_expert_name(layer: int, expert: int, projection: str,
                         arch: GgufArch = GLM5NEXT) -> str:
    return arch.expert_name(layer, expert, projection)


def kv_b_hf_name(layer: int, arch: GgufArch = GLM5NEXT) -> str:
    return arch.kv_b_name(layer)


def hf_shape_of(row: Mapping[str, Any]) -> Tuple[int, ...]:
    """ggml dims are fastest-first; the official row-major shape is the reverse."""
    return tuple(int(d) for d in reversed(row["dims"]))


def expert_slice_range(row: Mapping[str, Any], expert: int,
                       arch: GgufArch = GLM5NEXT) -> Tuple[int, int]:
    """(relative byte offset, byte length) of one expert inside a fused tensor."""
    dims = [int(d) for d in row["dims"]]
    experts = arch.num_experts
    if len(dims) != 3 or dims[2] != experts:
        raise _fail(f"{row['name']}: dims {dims} are not a fused {experts}-expert tensor")
    per_expert_elems = dims[0] * dims[1]
    traits = BLOCK_TRAITS.get(row["type"])
    if traits is None:
        raise _fail(f"{row['name']}: no block traits for type {row['type']}")
    per_block, block_bytes = traits
    if per_expert_elems % per_block:
        raise _fail(f"{row['name']}: expert slice is not block-aligned")
    if expert < 0 or expert >= experts:
        raise _fail(f"expert {expert} out of range")
    per_expert_bytes = per_expert_elems // per_block * block_bytes
    return expert * per_expert_bytes, per_expert_bytes


# ---------------------------------------------------------------------------
# census + surface
# ---------------------------------------------------------------------------

@dataclass
class GgufCensus:
    direct_map: Dict[str, str]            # gguf name -> HF name (1:1)
    routed: Dict[Tuple[int, str], str]    # (layer, projection) -> gguf name
    mla: Dict[Tuple[int, str], str]       # (layer, half) -> gguf name
    mla_layers: Tuple[int, ...]
    unmapped: List[str] = field(default_factory=list)
    unsupported: List[Tuple[str, str]] = field(default_factory=list)
    arch: GgufArch = GLM5NEXT
    # glm-dsa: GGUF indexer tensors on layers whose official indexer is
    # "shared" (no HF module) -> (gguf name, the full layer they copy).  They
    # are NOT in direct_map: a name the model does not build is never loaded.
    shared_indexer_copies: Dict[str, int] = field(default_factory=dict)

    def nonrouted_hf_names(self) -> List[str]:
        return sorted(list(self.direct_map.values())
                      + [kv_b_hf_name(layer, self.arch) for layer in self.mla_layers])


def indexer_full_layers_from_config(config: Any, arch: GgufArch) -> Optional[Tuple[int, ...]]:
    """Layers whose DSA indexer has its OWN weights, from the official config.

    ``indexer_types`` lists ``full``/``shared`` per decoder layer; the MTP block
    (past ``num_hidden_layers``) carries its own indexer in the official tree
    (index_share_for_mtp_iteration is a runtime flag, the weights ship).  None
    when the config carries no ``indexer_types`` (glm5next has none).
    """
    if isinstance(config, Mapping):
        types = config.get("indexer_types")
    else:
        types = getattr(config, "indexer_types", None)
    if types is None:
        return None
    types = list(types)
    if len(types) != arch.mtp_layer:
        raise _fail(
            f"config.indexer_types has {len(types)} entries but {arch.key} has "
            f"{arch.mtp_layer} decoder layers before the MTP block"
        )
    if any(t not in ("full", "shared") for t in types):
        raise _fail(f"config.indexer_types carries an unknown entry: {sorted(set(types))}")
    if types[0] != "full":
        raise _fail("config.indexer_types: layer 0 must be 'full' (a shared layer needs a parent)")
    return tuple(i for i, t in enumerate(types) if t == "full") + (arch.mtp_layer,)


def build_census(container: GgufContainer, arch: Optional[GgufArch] = None,
                 indexer_full_layers: Optional[Sequence[int]] = None) -> GgufCensus:
    """Classify EVERY tensor; refuse unknown names and undecodable types.

    ``indexer_full_layers`` is REQUIRED for an architecture whose GGUF ships
    indexer copies on shared layers (glm-dsa): it is the official config's
    ``indexer_types == "full"`` set (see ``indexer_full_layers_from_config``).
    Indexer tensors on any other layer are recorded as ``shared_indexer_copies``
    and must be proven value-identical to their parent by
    ``verify_shared_indexer_copies`` before a run.
    """
    if arch is None:
        arch = arch_for(container.architecture)
    if arch.indexer_shared_copies and indexer_full_layers is None:
        raise _fail(
            f"{arch.key}: the census needs the official config's indexer_types "
            "(which layers own an indexer); the GGUF carries indexer tensors on "
            "every layer and cannot say which are copies"
        )
    full_set = set(indexer_full_layers or ())
    direct_map: Dict[str, str] = {}
    routed: Dict[Tuple[int, str], str] = {}
    mla: Dict[Tuple[int, str], str] = {}
    shared_copies: Dict[str, int] = {}
    unmapped: List[str] = []
    unsupported: List[Tuple[str, str]] = []
    for name, row in container.tensors.items():
        role = classify_tensor(name, arch)
        if role[0] == "unmapped":
            unmapped.append(name)
            continue
        if row["type"] not in SUPPORTED_TYPES:
            unsupported.append((name, row["type"]))
        if role[0] == "top":
            direct_map[name] = role[1]
        elif role[0] == "direct":
            layer = role[1]
            if (arch.indexer_shared_copies and ".indexer." in name
                    and layer not in full_set):
                parents = [f for f in full_set if f < layer]
                if not parents:
                    raise _fail(f"{name}: shared-indexer layer {layer} has no preceding full layer")
                shared_copies[name] = max(parents)
            else:
                direct_map[name] = role[2]
        elif role[0] == "routed":
            routed[(role[1], role[2])] = name
        elif role[0] == "mla":
            mla[(role[1], role[2])] = name
    if unmapped:
        raise _fail(
            f"{len(unmapped)} tensors have no {arch.key}->HF mapping (first: "
            f"{sorted(unmapped)[:5]}). A tensor this adapter cannot NAME is a "
            "tensor it will not silently skip."
        )
    target_counts: Dict[str, int] = {}
    for hf in direct_map.values():
        target_counts[hf] = target_counts.get(hf, 0) + 1
    duplicate_targets = [hf for hf, count in target_counts.items() if count > 1]
    if duplicate_targets:
        raise _fail(
            f"two GGUF tensors map to the same official tensor: {sorted(duplicate_targets)[:3]} "
            "(alias spellings must not coexist in one artifact)"
        )
    if unsupported:
        listed = ", ".join(f"{n} [{t}]" for n, t in sorted(unsupported)[:6])
        types = sorted({t for _, t in unsupported})
        raise _fail(
            f"REFUSED: {len(unsupported)} tensors use ggml types without a v1 decode "
            f"kernel ({', '.join(types)}), e.g. {listed}. Supported: "
            f"{', '.join(SUPPORTED_TYPES)}. IQ-family GGUFs (ddh0, unsloth UD-IQ*) "
            "are a named exclusion until their kernels land with the same "
            "bitwise-vs-gguf-py proof."
        )
    # closure: routed tensors for every routed layer AND the MTP block, all
    # three projections
    expected_routed_layers = arch.routed_layers + (arch.mtp_layer,)
    missing_routed = [
        routed_tensor_name(layer, projection)
        for layer in expected_routed_layers
        for projection in PROJECTIONS
        if (layer, projection) not in routed
    ]
    if missing_routed:
        raise _fail(f"fused expert tensors absent: {missing_routed[:5]}")
    stray_routed = sorted(set(routed) - {(l, p) for l in expected_routed_layers
                                         for p in PROJECTIONS})
    if stray_routed:
        raise _fail(
            f"fused expert tensors outside layers {expected_routed_layers[0]}.."
            f"{expected_routed_layers[-1]}: {stray_routed[:5]}"
        )
    for (layer, projection), name in routed.items():
        row = container.tensors[name]
        out_features, in_features = arch.projection_shape[projection]
        if [int(d) for d in row["dims"]] != [in_features, out_features, arch.num_experts]:
            raise _fail(
                f"{name}: dims {row['dims']} != expected [in={in_features}, "
                f"out={out_features}, experts={arch.num_experts}]"
            )
    # MLA pairs must be complete per layer
    mla_layers = sorted({layer for layer, _ in mla})
    for layer in mla_layers:
        for half in ("k_b", "v_b"):
            if (layer, half) not in mla:
                raise _fail(f"layer {layer}: attn_{half} present without its pair")
    for (layer, half), name in mla.items():
        row = container.tensors[name]
        want = ([arch.mla_k_nope, arch.mla_kv_lora_rank, arch.mla_heads] if half == "k_b"
                else [arch.mla_kv_lora_rank, arch.mla_v_dim, arch.mla_heads])
        if [int(d) for d in row["dims"]] != want:
            raise _fail(f"{name}: dims {row['dims']} != expected {want}")
    if arch.indexer_shared_copies:
        # every full layer must own a complete indexer; every copy layer must
        # copy exactly the suffixes its parent has
        indexer_suffixes = sorted(s for s in _LAYER_DIRECT if s.startswith("indexer."))
        present = {name for name in container.tensors if ".indexer." in name}
        for layer in range(arch.block_count):
            want = {f"blk.{layer}.{s}" for s in indexer_suffixes
                    if f"blk.{layer}.{s}" in present}
            if layer in full_set and not want:
                raise _fail(f"layer {layer} is a full indexer layer but ships no indexer tensors")
            if layer not in full_set and want:
                parent = shared_copies[next(iter(want))]
                parent_have = {n.split(".", 2)[2] for n in present if n.startswith(f"blk.{parent}.")}
                mine = {n.split(".", 2)[2] for n in want}
                if mine != parent_have:
                    raise _fail(
                        f"layer {layer}: indexer copy set {sorted(mine)} differs from its "
                        f"parent layer {parent} {sorted(parent_have)}"
                    )
    return GgufCensus(direct_map=direct_map, routed=routed, mla=mla,
                      mla_layers=tuple(mla_layers), arch=arch,
                      shared_indexer_copies=shared_copies)


def verify_shared_indexer_copies(container: GgufContainer, census: GgufCensus,
                                 device=None) -> Dict[str, Any]:
    """PROVE every shared-layer indexer tensor equals its parent's, by VALUE.

    The audit (gguf-evidence/glmdsa-layout-audit.json) found the copies stored
    in a different ggml type than their parent in the BF16 build (BF16 vs F32),
    so bytes cannot be compared: both sides are decoded to fp32 and must be
    ``torch.equal``.  Refuses on the first difference, naming the tensor.
    """
    import torch

    compared = 0
    by_parent: Dict[int, int] = {}
    for name, parent in sorted(census.shared_indexer_copies.items()):
        suffix = name.split(".", 2)[2]
        parent_name = f"blk.{parent}.{suffix}"
        mine = load_decoded_tensor(container, name, device=device)
        theirs = load_decoded_tensor(container, parent_name, device=device)
        if mine.shape != theirs.shape or not torch.equal(mine, theirs):
            raise _fail(
                f"REFUSED: {name} is not a value-identical copy of {parent_name} - the "
                "converter's shared-indexer layout differs from the proven one"
            )
        compared += 1
        by_parent[parent] = by_parent.get(parent, 0) + 1
    return {"copies_compared": compared, "copies_by_parent_layer": by_parent,
            "all_value_identical": True}


def verify_nonrouted_bijection(census: GgufCensus, official_names) -> Dict[str, Any]:
    """The mapped HF set must EXACTLY biject the official non-routed non-vision set."""
    arch = census.arch
    official = set(official_names)
    routed_official = {
        official_expert_name(layer, expert, projection, arch)
        for layer in arch.routed_layers + (arch.mtp_layer,)
        for expert in range(arch.num_experts)
        for projection in PROJECTIONS
    }
    vision = {name for name in official if name.startswith("model.visual.")}
    expected = official - routed_official - vision
    produced = set(census.nonrouted_hf_names())
    if produced != expected:
        extra = sorted(produced - expected)[:5]
        missing = sorted(expected - produced)[:5]
        raise _fail(
            f"non-routed name map does not biject the official set "
            f"(maps-to-nothing-official: {extra}, official-but-unmapped: {missing})"
        )
    if len(routed_official - official):
        raise _fail("official index lacks routed expert names -- wrong index")
    return {
        "official_tensors": len(official),
        "official_routed_tensors": len(routed_official),
        "official_vision_tensors": len(vision),
        "nonrouted_mapped_tensors": len(produced),
        "mla_reconstructed_tensors": len(census.mla_layers),
        "shared_indexer_copies_not_loaded": len(census.shared_indexer_copies),
        "bijection_ok": True,
    }


GEOMETRY_GATE = GLM5NEXT.geometry_gate


@dataclass(frozen=True)
class GgufSurface:
    container: GgufContainer
    census: GgufCensus
    repo: Optional[str]
    revision: str
    architecture: str
    file_records: Tuple[Dict[str, Any], ...]
    file_hash_verification: str  # "full" | "skipped"
    type_census: Dict[str, int]
    scope_policy: Dict[str, Any]
    quant_metadata: Dict[str, Any]

    @property
    def arch(self) -> GgufArch:
        return self.census.arch

    def checkpoint_identity_sha256(self) -> str:
        body = {
            "schema": GGUF_IDENTITY_SCHEMA,
            "gguf_repo": self.repo,
            "gguf_revision": self.revision,
            "format": GGUF_FORMAT,
            "architecture": self.architecture,
            "files": list(self.file_records),
            "file_hash_verification": self.file_hash_verification,
            "type_census": dict(self.type_census),
            "scope_policy": self.scope_policy,
            "quant_metadata": self.quant_metadata,
            "mla_kv_b_arrangement": MLA_KV_B_ARRANGEMENT,
            "nonrouted_policy": "decoded_from_the_same_gguf_artifact",
            "seal_disclosure": SEAL_DISCLOSURE,
        }
        if self.census.shared_indexer_copies:
            # glm-dsa only: the identity says how many artifact tensors are
            # proven duplicates that the model never loads (glm5next hashes
            # stay byte-identical to the published Flash rows)
            body["shared_indexer_copies_not_loaded"] = len(self.census.shared_indexer_copies)
        return _sha256_bytes(_canonical_json(body))


def _scope_policy(container: GgufContainer, census: GgufCensus) -> Dict[str, Any]:
    """MEASURED scope: what this artifact quantized, read from its own table."""
    def type_of(name: str) -> Optional[str]:
        row = container.tensors.get(name)
        return None if row is None else row["type"]

    quantized = sorted({row["type"] for row in container.tensors.values()
                        if row["type"] not in ("F32", "F16", "BF16")})
    routed_types = sorted({container.tensors[name]["type"] for name in census.routed.values()})
    attention_names = [name for name in list(census.direct_map) + list(census.mla.values())
                       if ".attn_" in name or ".ssm_" in name or ".indexer" in name]
    attention_quantized = any(
        container.tensors[name]["type"] not in ("F32", "F16", "BF16")
        for name in attention_names
    )
    return {
        "policy": "artifact_quantizes_nonrouted_tensors_too",
        "embeddings_type": type_of("token_embd.weight"),
        "lm_head_type": type_of("output.weight"),
        "attention_kda_dsa_quantized": bool(attention_quantized),
        "routed_expert_types": routed_types,
        "quantized_types_present": quantized,
        "vision_in_artifact": False,
        "vision_source": "official_bf16_tree_not_part_of_the_artifact_never_executed",
        "activations": "not_quantized_by_gguf_weights_only_format",
        "disclosure": SCOPE_DISCLOSURE,
    }


GGUF_SCOPE_SCHEMA = "malaiwah.glm53-gguf-scope.v1"

#: ggml type -> (registry numeric_format, nominal bits/weight).  "Nominal" is
#: the type's name; the MEASURED rate per class is computed from the block
#: traits below and reported in the note, because a K-quant block carries
#: scales and mins that the name does not mention.
_REGISTRY_FORMAT = {
    "F32": ("fp32", 32.0), "F16": ("fp16", 16.0), "BF16": ("bf16", 16.0),
    "Q8_0": ("gguf-k-quant", 8.0), "Q6_K": ("gguf-k-quant", 6.0),
    "Q5_K": ("gguf-k-quant", 5.0), "Q4_K": ("gguf-k-quant", 4.0),
    "Q3_K": ("gguf-k-quant", 3.0), "Q2_K": ("gguf-k-quant", 2.0),
    "Q5_1": ("gguf-k-quant", 5.0), "Q5_0": ("gguf-k-quant", 5.0),
    "Q4_1": ("gguf-k-quant", 4.0), "Q4_0": ("gguf-k-quant", 4.0),
    "IQ4_XS": ("gguf-i-quant", 4.0), "IQ4_NL": ("gguf-i-quant", 4.0),
    "IQ3_S": ("gguf-i-quant", 3.0), "IQ3_XXS": ("gguf-i-quant", 3.0),
    "IQ2_S": ("gguf-i-quant", 2.0), "IQ2_XS": ("gguf-i-quant", 2.0),
    "IQ2_XXS": ("gguf-i-quant", 2.0), "IQ1_M": ("gguf-i-quant", 1.0),
    "IQ1_S": ("gguf-i-quant", 1.0), "MXFP4": ("mxfp4", 4.0),
}
_NATIVE_TYPES = ("F32", "F16", "BF16")


def scope_class_of(hf_name: str, layer: Optional[int],
                   arch: GgufArch = GLM5NEXT) -> Tuple[str, str]:
    """(registry tensor_class, why) for one official name.

    The registry vocabulary is coarse and model-agnostic on purpose, and this
    architecture has three families that do not map onto it by keyword:
    GLM5Next's KDA gates (``f_a/f_b/g_a/g_b/b_proj``, the conv1d kernels,
    ``A_log``/``dt_bias``), the DSA sparse-attention indexer, and the mHC
    hyper-connection coefficients (``hc_*``).  None of them is a q/k/v or an o
    projection, so calling them ``attn.qkv`` would overstate what the qkv row
    covers; they go to ``attn.other`` / ``other`` WITH a note, which the schema
    requires for ``other`` and which is the honest answer anyway.
    """
    if layer == arch.mtp_layer:
        return "mtp", "layer %d is the MTP layer; present in the artifact, never executed" % arch.mtp_layer
    if hf_name.endswith("lm_head.weight"):
        return "lm_head", "the output projection"
    if "embed_tokens" in hf_name:
        return "embed_tokens", "the token embedding table"
    if ".mlp.experts." in hf_name or ".mlp.experts" in hf_name:
        return "moe.experts", "routed experts"
    if ".mlp.shared_experts." in hf_name:
        return "moe.shared_expert", "the always-on shared expert"
    if ".mlp.gate." in hf_name:
        return "moe.router", "the router logits and their score-correction bias"
    if hf_name.endswith(".mlp.gate_proj.weight"):
        return "mlp.gate", "dense (non-MoE) layers only"
    if hf_name.endswith(".mlp.up_proj.weight"):
        return "mlp.up", "dense (non-MoE) layers only"
    if hf_name.endswith(".mlp.down_proj.weight"):
        return "mlp.down", "dense (non-MoE) layers only"
    if ".self_attn." in hf_name:
        tail = hf_name.split(".self_attn.", 1)[1]
        if tail.startswith("o_proj"):
            return "attn.o", "the attention output projection"
        if tail.split(".")[0] in (
                "q_proj", "k_proj", "v_proj", "q_a_proj", "q_b_proj",
                "kv_a_proj_with_mqa", "kv_b_proj"):
            return "attn.qkv", "q/k/v and the MLA down/up projections"
        return "attn.other", ("KDA gates, conv1d kernels, the DSA indexer and "
                              "the attention norms")
    if ".hc_attn" in hf_name:
        return "attn.other", "mHC hyper-connection coefficients on the attention branch"
    if ".hc_ffn" in hf_name:
        return "other", "mHC hyper-connection coefficients on the FFN branch"
    if "norm" in hf_name:
        return "norm", "layer and final norms"
    if ".visual." in hf_name:
        return "other", "the vision tower"
    return "other", "unclassified by the registry's coarse vocabulary"


def measured_scope(surface: "GgufSurface") -> Dict[str, Any]:
    """The per-tensor-class recipe, MEASURED from the container's own table.

    Nothing here is read off a build NAME and nothing is defaulted.  Every
    class reports the ggml types its tensors actually use, so a mixed class
    (unsloth's "Dynamic" recipe puts Q4_K gate/up beside Q6_K down on three
    layers) says `mixed` instead of picking one and calling the other a
    rounding error.

    The MEASURED bits/weight per class is computed from ggml's own block traits
    -- elements and bytes -- and lands in the note, because it is the number the
    type name does not tell you: Q4_K is 4.5 bits/weight once its scales and
    mins are counted, not 4.
    """
    container, census = surface.container, surface.census
    arch = census.arch
    per_class: Dict[str, Dict[str, Any]] = {}

    def account(hf_name: str, layer: Optional[int], row: Mapping[str, Any],
                count: int = 1) -> None:
        cls, why = scope_class_of(hf_name, layer, arch)
        entry = per_class.setdefault(cls, {"types": {}, "why": why,
                                           "elements": 0, "bytes": 0})
        ggml_type = row["type"]
        entry["types"][ggml_type] = entry["types"].get(ggml_type, 0) + count
        elements = int(row["elements"])
        traits = BLOCK_TRAITS.get(ggml_type)
        entry["elements"] += elements
        if traits is not None:
            per_block, block_bytes = traits
            entry["bytes"] += elements // per_block * block_bytes

    for gguf_name, hf_name in census.direct_map.items():
        match = _BLK.match(gguf_name)
        account(hf_name, int(match.group(1)) if match else None,
                container.tensors[gguf_name])
    for (layer, _half), gguf_name in census.mla.items():
        account(kv_b_hf_name(layer, arch), layer, container.tensors[gguf_name])
    for (layer, projection), gguf_name in census.routed.items():
        account(official_expert_name(layer, 0, projection, arch), layer,
                container.tensors[gguf_name])

    assignments = []
    for cls in sorted(per_class):
        entry = per_class[cls]
        types = entry["types"]
        native = all(t in _NATIVE_TYPES for t in types)
        formats = {_REGISTRY_FORMAT.get(t, ("unknown", None))[0] for t in types}
        fmt = formats.pop() if len(formats) == 1 else "mixed"
        nominal = {_REGISTRY_FORMAT.get(t, ("unknown", None))[1] for t in types}
        effective = (8.0 * entry["bytes"] / entry["elements"]) if entry["elements"] else None
        note = "%s. ggml types: %s. MEASURED %.4f bits/weight over %d weights." % (
            entry["why"],
            ", ".join("%s x%d" % (t, types[t]) for t in sorted(types)),
            effective if effective is not None else float("nan"), entry["elements"])
        if native:
            note += (" Stored natively by the artifact; the streaming lane's "
                     "materialized view casts F32 tensors the official release "
                     "stores as bf16 down to bf16, so the constructed model is "
                     "dtype-identical to a native build rather than wider.")
        assignments.append({
            "tensor_class": cls,
            "treatment": "native" if native else "quantized",
            "format": fmt,
            "bits_per_weight": (nominal.pop() if len(nominal) == 1 else None),
            "layer_range": "all",
            "note": note,
        })
    # The vision tower belongs to `other` too, and it is ABSENT. One class
    # cannot be both quantized and not_present, and the registry vocabulary has
    # one `other` slot, so the absence is stated in the note rather than as a
    # second entry with the same tensor_class -- which would double-count the
    # class in scope_digest and read as a contradiction.
    vision = ("The vision tower (model.visual.*) is ALSO `other` and is NOT in "
              "this container -- llama.cpp ships it as a separate mmproj file. "
              "The streaming lane copies it from the official BF16 tree so the "
              "model can be constructed at all; the text-only sealed panel never "
              "executes it, so no vision weight is inside the measured function.")
    other = next((a for a in assignments if a["tensor_class"] == "other"), None)
    if other is None:
        assignments.append({
            "tensor_class": "other", "treatment": "not_present",
            "format": "unknown", "bits_per_weight": None, "layer_range": "all",
            "note": vision,
        })
    else:
        other["note"] += " " + vision
    quantized = {(a["format"], a["bits_per_weight"]) for a in assignments
                 if a["treatment"] == "quantized"}
    policy = "none" if not quantized else ("uniform" if len(quantized) == 1 else "mixed")
    head = next((a for a in assignments if a["tensor_class"] == "lm_head"), None)
    return {
        "policy": policy,
        "head_policy": ("quantized" if head and head["treatment"] == "quantized"
                        else "native" if head else "unknown"),
        # A GGUF is a WEIGHTS container. The KV cache dtype is a llama.cpp
        # runtime flag that the file does not carry, and this measurement does
        # not run llama.cpp at all -- it runs the sealed transformers forward.
        # "unknown" would suggest the artifact declares something we failed to
        # read; it declares nothing, which is a different fact.
        "kv_cache_dtype": "not_applicable",
        "mtp_included": any(a["tensor_class"] == "mtp" for a in assignments),
        "activation_quantization": None,
        "assignments": assignments,
    }


def measured_bits_per_weight(surface: "GgufSurface") -> Optional[float]:
    """The artifact's own bits/weight, over every tensor it stores."""
    elements = 0
    payload = 0
    for row in surface.container.tensors.values():
        traits = BLOCK_TRAITS.get(row["type"])
        if traits is None:
            return None
        per_block, block_bytes = traits
        n = int(row["elements"])
        elements += n
        payload += n // per_block * block_bytes
    return (8.0 * payload / elements) if elements else None


def scope_report(surface: "GgufSurface") -> Dict[str, Any]:
    """What `gguf_surface.py scope` writes: the scope plus its provenance."""
    scope = measured_scope(surface)
    return {
        "schema": GGUF_SCOPE_SCHEMA,
        "scope": scope,
        "measured_bits_per_weight": measured_bits_per_weight(surface),
        "source": {
            "repo": surface.repo,
            "revision": surface.revision,
            "architecture": surface.architecture,
            "files": [row["name"] for row in surface.file_records],
            "file_hash_verification": surface.file_hash_verification,
            "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
            "quant_metadata": surface.quant_metadata,
            "read_from": ("the container's OWN tensor table -- every type, "
                          "element count and block size -- never from the build "
                          "name, which for an unsloth 'UD' build claims one type "
                          "and holds four"),
        },
        "seal_disclosure": SEAL_DISCLOSURE,
        "scope_disclosure": SCOPE_DISCLOSURE,
    }


def load_gguf_surface(
    locations: List[str],
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    require_file_hashes: bool = True,
    indexer_full_layers: Optional[Sequence[int]] = None,
) -> GgufSurface:
    """Open every file of one artifact, gate its geometry and census it.

    ``indexer_full_layers`` is the official config's ``indexer_types == "full"``
    set (``indexer_full_layers_from_config``); required for glm-dsa, ignored
    for glm5next.
    """
    files = [GgufFile(location) for location in locations]
    container = GgufContainer(files)
    arch = arch_for(container.architecture)
    for key, want in arch.geometry_gate.items():
        got = container.geometry_value(key)
        if got is None or int(got) != want:
            raise _fail(
                f"geometry gate: {container.architecture}.{key} is {got!r}, expected {want}"
            )
    census = build_census(container, arch, indexer_full_layers=indexer_full_layers)
    if revision is not None and _REVISION.fullmatch(revision) is None:
        raise _fail("--gguf-revision must be the immutable 40-hex repo commit")

    file_records: List[Dict[str, Any]] = []
    verification = "skipped"
    if container.remote:
        if require_file_hashes:
            raise _fail(
                "remote (https) GGUF locations support metadata dry-runs and audits "
                "only; a measurement needs local files plus `gguf_surface.py "
                "verify-files` (or an explicit --skip-gguf-hashes disclosure)"
            )
        for f in container.files:
            file_records.append({"name": f.name, "bytes": f.size, "sha256": None})
    else:
        marker: Optional[Dict[str, Any]] = None
        marker_path = Path(container.files[0].location).resolve().parent / "gguf-files-verified.json"
        if marker_path.is_file():
            marker = _read_json(marker_path, "gguf-files-verified.json")
            if marker.get("schema") != GGUF_FILES_VERIFIED_SCHEMA or marker.get("all_hashed") is not True:
                raise _fail(f"stale/foreign {marker_path} - re-run verify-files")
            by_name = {row["name"]: row for row in marker.get("files", [])}
            for f in container.files:
                row = by_name.get(f.name)
                if row is None or int(row["bytes"]) != f.size:
                    raise _fail(
                        f"{marker_path} does not cover {f.name} at its current size - "
                        "re-run verify-files"
                    )
                file_records.append({"name": f.name, "bytes": f.size, "sha256": row["sha256"]})
            verification = "full"
        elif require_file_hashes:
            raise _fail(
                "whole-file sha256 marker absent: run `python gguf_surface.py "
                f"verify-files --file ...` first (writes {marker_path.name}), or pass "
                "--skip-gguf-hashes for a disclosed unverified read"
            )
        else:
            for f in container.files:
                file_records.append({"name": f.name, "bytes": f.size, "sha256": None})

    type_census: Dict[str, int] = {}
    for row in container.tensors.values():
        type_census[row["type"]] = type_census.get(row["type"], 0) + 1
    quant_keys = ("general.file_type", "general.quantization_version", "general.quantized_by",
                  "quantize.imatrix.file", "quantize.imatrix.dataset",
                  "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count")
    quant_metadata = {key: container.kv[key] for key in quant_keys if key in container.kv}
    return GgufSurface(
        container=container,
        census=census,
        repo=repo,
        revision=revision or "unpinned-local-snapshot",
        architecture=container.architecture,
        file_records=tuple(file_records),
        file_hash_verification=verification,
        type_census=type_census,
        scope_policy=_scope_policy(container, census),
        quant_metadata=quant_metadata,
    )


def verify_file_hashes(locations: List[str]) -> Dict[str, Any]:
    """Hash every local GGUF; write gguf-files-verified.json next to the first."""
    paths = [Path(location).resolve() for location in locations]
    for path in paths:
        if not path.is_file():
            raise _fail(f"GGUF file absent: {path}")
    started = time.monotonic()
    rows = [{"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in paths]
    record = {
        "schema": GGUF_FILES_VERIFIED_SCHEMA,
        "files": rows,
        "all_hashed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    marker = paths[0].parent / "gguf-files-verified.json"
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record["marker"] = str(marker)
    return record


# ---------------------------------------------------------------------------
# decode: routed experts, MLA reconstruction, whole tensors
# ---------------------------------------------------------------------------

def load_decoded_tensor(container: GgufContainer, gguf_name: str, device=None):
    """Whole tensor -> fp32 in the OFFICIAL (row-major reversed-dims) shape."""
    row = container.tensors[gguf_name]
    flat = dequant_bytes(row["type"], container.read_tensor(gguf_name), int(row["elements"]),
                         device=device)
    return flat.reshape(hf_shape_of(row))


def load_decoded_expert(container: GgufContainer, census: GgufCensus, *,
                        layer: int, expert: int, projection: str, device=None):
    """One routed expert -> (fp32 [out, in] tensor, census row).

    ``device`` is handed straight to ``dequant_bytes``: the quantized slice is
    what crosses the bus, and the decode lands the tensor where the slab
    already is.  Bitwise-identical to ``device=None`` by construction (same
    kernels, same order) and by test.
    """
    arch = census.arch
    name = census.routed.get((layer, projection))
    if name is None:
        raise _fail(f"no fused tensor for layer {layer} {projection}")
    row = container.tensors[name]
    rel, nbytes = expert_slice_range(row, expert, arch)
    out_features, in_features = arch.projection_shape[projection]
    flat = dequant_bytes(row["type"], container.read_tensor_range(name, rel, nbytes),
                         out_features * in_features, device=device)
    tensor = flat.reshape(out_features, in_features)
    return tensor, {
        "tensor": official_expert_name(layer, expert, projection, arch),
        "gguf_tensor": name,
        "shard": row["file"],
        "bytes": nbytes,
        "ggml_type": row["type"],
        "dtype": "float32-decoded",
    }


def compose_kv_b(k, v, arch: GgufArch):
    """Decoded attn_k_b [heads, rank, nope] + attn_v_b [heads, v, rank] ->
    official kv_b_proj [heads*(nope+v), rank].

    PROVEN arrangement (MLA_KV_B_ARRANGEMENT): per head the official rows are
    [transpose(k_b[h]); v_b[h]] -- llama.cpp stores k_b per-head TRANSPOSED
    for its MQA absorb trick.  glm5next: rel-L2 0.0054 (Q8_0 error) vs every
    other candidate >= 1.40; glm-dsa: EXACT equality on the BF16 build
    (gguf-evidence/glmdsa-layout-audit.json), and the offline selftest re-runs
    this composition on committed real windows.
    """
    import torch

    heads, rank = arch.mla_heads, arch.mla_kv_lora_rank
    k = k.reshape(heads, rank, arch.mla_k_nope)
    v = v.reshape(heads, arch.mla_v_dim, rank)
    k_t = k.transpose(1, 2).contiguous()
    return torch.cat([k_t, v], dim=1).reshape(arch.kv_b_rows, rank).contiguous()


def reconstruct_kv_b(container: GgufContainer, census: GgufCensus, layer: int,
                     device=None):
    """attn_k_b + attn_v_b -> official kv_b_proj, fp32 (see ``compose_kv_b``)."""
    arch = census.arch
    k_name = census.mla[(layer, "k_b")]
    v_name = census.mla[(layer, "v_b")]
    k_row, v_row = container.tensors[k_name], container.tensors[v_name]
    k_flat = dequant_bytes(k_row["type"], container.read_tensor(k_name), int(k_row["elements"]),
                           device=device)
    v_flat = dequant_bytes(v_row["type"], container.read_tensor(v_name), int(v_row["elements"]),
                           device=device)
    return compose_kv_b(k_flat, v_flat, arch)


def audit_expert_placement(candidate, official_bf16, *, label: str,
                           shifts: Tuple[int, ...] = (1, 8, 64, 512)) -> Dict[str, Any]:
    """Is the expert we decoded the expert the official checkpoint calls that?

    The fused GGUF tensor is [in, out, 288] and slot `e` is assumed to be HF
    expert `e`.  That assumption is exactly as unproven-by-inspection as the
    kv_b_proj layout was, and exactly as silently wrong if it is off: a
    permuted expert order decodes cleanly, closes every census, and measures
    the wrong model.  So it is checked numerically, the same way.

    `candidate` is the decoded slice (fp32, [rows, in]); `official_bf16` is the
    official expert tensor, of which the leading `rows` rows are compared.  The
    aligned comparison must land at the QUANTIZATION error while every
    row-shifted control lands at O(1) -- which simultaneously proves the slot
    ordering, the row-major (reversed-dims) orientation and the projection
    mapping, since a transposed read or a wrong projection fails it too.
    """
    import torch

    cand = candidate.to(torch.float64)
    official = official_bf16.to(torch.float64)
    rows = int(cand.shape[0])
    if official.shape[0] < rows or official.shape[1] != cand.shape[1]:
        raise _fail(
            f"expert placement audit: candidate {tuple(cand.shape)} does not fit the official "
            f"tensor {tuple(official.shape)}"
        )

    def _score(other):
        cos = float(torch.nn.functional.cosine_similarity(
            cand.flatten(), other.flatten(), dim=0))
        return {"cosine": cos, "rel_l2": float((cand - other).norm() / other.norm())}

    aligned = _score(official[:rows])
    controls: Dict[str, Dict[str, float]] = {}
    for shift in shifts:
        if shift + rows <= official.shape[0]:
            controls[f"official_rows_shifted_by_{shift}"] = _score(
                official[shift:shift + rows])
    if not controls:
        raise _fail("expert placement audit needs at least one row-shifted control")
    best_control = min(row["rel_l2"] for row in controls.values())
    ok = (aligned["rel_l2"] < 0.5 and best_control > 0.5
          and best_control > 4.0 * aligned["rel_l2"])
    audit = {"label": label, "rows_compared": rows, "aligned": aligned,
             "controls": controls, "best_control_rel_l2": best_control,
             "passed": bool(ok)}
    if not ok:
        raise _fail(
            f"expert placement audit FAILED for {label}: aligned rel-L2 "
            f"{aligned['rel_l2']:.4f} vs best control {best_control:.4f} - the fused "
            "tensor's expert slot ordering or orientation is not what this adapter assumes"
        )
    return audit


def audit_mla_placement(container: GgufContainer, census: GgufCensus, *,
                        layer: int, official_bf16) -> Dict[str, Any]:
    """Re-prove the kv_b arrangement against the official BF16 tensor.

    `official_bf16` is the official kv_b_proj tensor (any float dtype).  All
    four candidate arrangements are scored; the shipped one must dominate.
    Cosines are accumulated in float64 on CPU (audits never run on MPS).

    The head count is READ from the k_b row's own dims rather than assumed, so
    the same function audits a full 64-head tensor and a cheap leading-head
    WINDOW of one (which is what the offline selftest replays from committed
    real bytes, and what a bandwidth-limited pre-flight would fetch).
    """
    import torch

    k_name = census.mla[(layer, "k_b")]
    v_name = census.mla[(layer, "v_b")]
    k_row, v_row = container.tensors[k_name], container.tensors[v_name]
    heads = int(k_row["dims"][2])
    if int(v_row["dims"][2]) != heads:
        raise _fail(f"layer {layer}: attn_k_b covers {heads} heads, attn_v_b "
                    f"{int(v_row['dims'][2])} - not the same window")
    if official_bf16.numel() != heads * 2 * MLA_HEAD_DIM * MLA_KV_LORA_RANK:
        raise _fail(
            f"official kv_b_proj has {official_bf16.numel()} elements but the {heads}-head "
            f"window needs {heads * 2 * MLA_HEAD_DIM * MLA_KV_LORA_RANK} "
            "(pass the matching leading rows)"
        )
    official = official_bf16.to(torch.float64).reshape(heads * 2 * MLA_HEAD_DIM,
                                                       MLA_KV_LORA_RANK)
    k = dequant_bytes(k_row["type"], container.read_tensor(k_name),
                      int(k_row["elements"])).reshape(heads, MLA_KV_LORA_RANK, MLA_HEAD_DIM)
    v = dequant_bytes(v_row["type"], container.read_tensor(v_name),
                      int(v_row["elements"])).reshape(heads, MLA_HEAD_DIM, MLA_KV_LORA_RANK)
    k_t = k.transpose(1, 2).contiguous()
    candidates = {
        "per_head_rows_kT_then_v": torch.cat([k_t, v], dim=1).reshape(-1, MLA_KV_LORA_RANK),
        "per_head_rows_v_then_kT": torch.cat([v, k_t], dim=1).reshape(-1, MLA_KV_LORA_RANK),
        "all_k_rows_then_all_v_rows": torch.cat(
            [k_t.reshape(-1, MLA_KV_LORA_RANK), v.reshape(-1, MLA_KV_LORA_RANK)], dim=0),
        "all_v_rows_then_all_k_rows": torch.cat(
            [v.reshape(-1, MLA_KV_LORA_RANK), k_t.reshape(-1, MLA_KV_LORA_RANK)], dim=0),
    }
    scores: Dict[str, Dict[str, float]] = {}
    for name, cand in candidates.items():
        c64 = cand.to(torch.float64)
        cos = float(torch.nn.functional.cosine_similarity(
            c64.flatten(), official.flatten(), dim=0))
        rel = float((c64 - official).norm() / official.norm())
        scores[name] = {"cosine": cos, "rel_l2": rel}
    winner = max(scores, key=lambda name: scores[name]["cosine"])
    # The discriminator is rel-L2, not a cosine MARGIN.  Two arrangements that
    # share a leading block (per-head-k-then-v and all-k-then-all-v agree on
    # head 0's k rows) have a cosine gap that shrinks as 1/(2*heads): measured
    # 0.013 over the full 64 heads but 0.546 over a 2-head window, so an
    # absolute cosine margin passes or fails depending on how much of the
    # tensor was fetched.  rel-L2 does not move: the right arrangement scores
    # the QUANTIZATION error (~0.005 for Q8_0) and every wrong one scores O(1),
    # at either window size.  The gate therefore reads "the shipped arrangement
    # reproduces the official weights and no other arrangement comes near".
    runner_up_rel_l2 = min(scores[name]["rel_l2"] for name in scores
                           if name != MLA_KV_B_ARRANGEMENT)
    shipped_rel_l2 = scores[MLA_KV_B_ARRANGEMENT]["rel_l2"]
    ok = (winner == MLA_KV_B_ARRANGEMENT
          and scores[MLA_KV_B_ARRANGEMENT]["cosine"] > 0.98
          and shipped_rel_l2 < 0.10
          and runner_up_rel_l2 > 0.50
          and runner_up_rel_l2 > 10.0 * shipped_rel_l2)
    audit = {"layer": layer, "heads_audited": heads, "candidates": scores,
             "shipped": MLA_KV_B_ARRANGEMENT, "winner": winner,
             "shipped_rel_l2": shipped_rel_l2, "best_other_rel_l2": runner_up_rel_l2,
             "passed": bool(ok)}
    if not ok:
        raise _fail(
            f"MLA kv_b placement audit FAILED at layer {layer}: {scores} - the "
            "converter's k_b/v_b layout differs from the proven arrangement"
        )
    return audit


# ---------------------------------------------------------------------------
# expert source for stream_score's ExpertStreamer
# ---------------------------------------------------------------------------

class GgufExpertSource:
    """Routed experts decoded per (layer, expert, projection) from the GGUF.

    Mirrors NativeCheckpointSource's contract: ``load`` returns a tensor ready
    for the shared install algebra (device move, fuse_gate_up, ONE bf16
    rounding, torch.equal close) plus a census row.  os.pread-based reads are
    thread-safe without per-thread handles.

    ``decode_device`` decides WHERE the block dequant runs.  ``None`` is the
    reference: the pool thread decodes on CPU and the consumer moves 33.5 MB of
    fp32 per matrix to the accelerator.  Anything else sends the 4.7 MB
    quantized slice instead and decodes there, which is the difference between
    23.7 min/window and something a contributor can afford (see the module
    docstring on `dequant_bytes` and docs/GGUF-MEASUREMENT.md).  The two produce
    bitwise-identical tensors; that is the acceptance test, not a hope.

    The consumer's ``payload_cpu.to(self.device)`` is a no-op when the decode
    already landed on that device, so the fill loop needs no branch.
    """

    def __init__(self, surface: GgufSurface, decode_device=None):
        self.surface = surface
        self.decode_device = decode_device
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.files_read: set = set()

    def routed_tensor_census(self, layers: Tuple[int, ...]) -> Dict[str, Any]:
        arch = self.surface.arch
        rows = []
        for layer in layers:
            for projection in PROJECTIONS:
                name = self.surface.census.routed.get((layer, projection))
                if name is None:
                    raise _fail(f"fused tensor absent for layer {layer} {projection}")
                row = self.surface.container.tensors[name]
                rows.append(row)
        # per-expert byte cost is a function of the LAYER's ggml type, and the
        # unsloth XL builds deliberately mix types across layers (Q4_K gate/up
        # with three Q6_K down layers, ...).  Reporting one layer's number as
        # if it were the artifact's would understate the fetch ledger, so this
        # reports the distinct sizes per projection and the exact total.
        per_expert: Dict[str, Dict[str, Any]] = {}
        streamed_bytes = 0
        for projection in PROJECTIONS:
            sizes: Dict[str, int] = {}
            for layer in layers:
                row = self.surface.container.tensors[self.surface.census.routed[
                    (layer, projection)]]
                size = expert_slice_range(row, 0)[1]
                sizes[row["type"]] = size
                streamed_bytes += size * arch.num_experts
            per_expert[projection] = {"bytes_by_ggml_type": dict(sorted(sizes.items()))}
        return {
            "routed_tensor_count": len(layers) * arch.num_experts * len(PROJECTIONS),
            "fused_gguf_tensors": len(rows),
            "layers": [layers[0], layers[-1]] if layers else [],
            "experts_per_layer": arch.num_experts,
            "per_expert_bytes": per_expert,
            "streamed_routed_bytes_total": streamed_bytes,
            "types": sorted({row["type"] for row in rows}),
            "mtp_layer_%d_fused_tensors_present_not_streamed" % arch.mtp_layer: all(
                (arch.mtp_layer, p) in self.surface.census.routed for p in PROJECTIONS
            ),
        }

    def load(self, *, layer: int, expert: int, projection: str):
        tensor, row = load_decoded_expert(
            self.surface.container, self.surface.census,
            layer=layer, expert=expert, projection=projection,
            device=self.decode_device,
        )
        with self._lock:
            self.bytes_read += int(row["bytes"])
            self.files_read.add(row["shard"])
        return tensor, row


# ---------------------------------------------------------------------------
# per-layer subset reader for the layer-outer streamer (gguf-dequant-to-bf16)
# ---------------------------------------------------------------------------

RESIDENT_LAYER = -1  # the key `layer_subsets` uses for embed/norm/head


def layer_partition(census: GgufCensus) -> Dict[int, List[str]]:
    """layer index -> the GGUF tensor names the streamer loads for that layer.

    RESIDENT_LAYER carries the top-level tensors (token_embd, output_norm,
    output).  Shared-indexer copies are absent by construction (they are not in
    ``direct_map``).  Every other tensor of the container appears exactly once.
    """
    out: Dict[int, List[str]] = {}
    for gguf_name in census.direct_map:
        match = _BLK.match(gguf_name)
        out.setdefault(int(match.group(1)) if match else RESIDENT_LAYER, []).append(gguf_name)
    for (layer, _half), gguf_name in census.mla.items():
        out.setdefault(layer, []).append(gguf_name)
    for (layer, _projection), gguf_name in census.routed.items():
        out.setdefault(layer, []).append(gguf_name)
    return {layer: sorted(names) for layer, names in out.items()}


def materialize_layer(surface: GgufSurface, layer: int, *, torch_dtype=None,
                      device=None, stats: Optional[Dict[str, Any]] = None,
                      only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Every tensor of one decoder layer (or RESIDENT_LAYER) under its OFFICIAL name.

    ``only`` restricts the decode to the named official tensors (the streamer's
    resident load asks one layer for just its router-correction BUFFER); a name
    in ``only`` that the layer does not carry is a refusal.

    Decoded ON ``device`` (the quantized bytes cross the bus, not the fp32
    result): each GGUF tensor is dequantized once to fp32 by the proven kernels,
    the MLA halves are composed into ``kv_b_proj`` (``compose_kv_b``), fused
    expert tensors are sliced into the ``mlp.experts.{e}.{proj}.weight`` names
    the HF converter expects, and the result is cast ONCE to ``torch_dtype``
    (bfloat16 by default) -- except the tensors the official tree stores as
    float32 (``arch.official_f32_suffixes``), which stay fp32.  Nothing here is
    a guess: the name map, the kv_b arrangement, the expert slot order and the
    F32-widened tensors were each proven bitwise against the official BF16
    release (gguf-evidence/glmdsa-layout-audit.json; module docstring).

    ``stats`` (optional) accumulates decoded tensor counts, bytes read and the
    ggml type histogram so the runtime receipt can report them.
    """
    import torch

    torch_dtype = torch_dtype or torch.bfloat16
    arch, container, census = surface.arch, surface.container, surface.census
    partition = layer_partition(census)
    if layer not in partition:
        raise _fail(f"layer {layer} is not a layer of this artifact "
                    f"(known: {sorted(partition)})")
    out: Dict[str, Any] = {}
    counters = stats if stats is not None else {}
    counters.setdefault("tensors_decoded", 0)
    counters.setdefault("official_tensors_produced", 0)
    counters.setdefault("gguf_bytes_read", 0)
    counters.setdefault("ggml_types", {})

    def note(row: Mapping[str, Any]) -> None:
        counters["tensors_decoded"] += 1
        counters["gguf_bytes_read"] += int(row["bytes"])
        counters["ggml_types"][row["type"]] = counters["ggml_types"].get(row["type"], 0) + 1

    def finish(hf_name: str, tensor) -> None:
        if hf_name in out:
            raise _fail(f"{hf_name} produced twice for layer {layer}")
        want = torch.float32 if arch.official_dtype_for(hf_name) == "float32" else torch_dtype
        out[hf_name] = tensor.to(want).contiguous()
        counters["official_tensors_produced"] += 1

    wanted = set(only) if only is not None else None
    mla_seen: Dict[int, Dict[str, Any]] = {}
    for gguf_name in partition[layer]:
        role = classify_tensor(gguf_name, arch)
        row = container.tensors[gguf_name]
        if role[0] in ("top", "direct"):
            hf_name = role[1] if role[0] == "top" else role[2]
            if wanted is not None and hf_name not in wanted:
                continue
            finish(hf_name, load_decoded_tensor(container, gguf_name, device=device))
            note(row)
        elif role[0] == "mla":
            if wanted is not None and kv_b_hf_name(role[1], arch) not in wanted:
                continue
            mla_seen.setdefault(role[1], {})[role[2]] = gguf_name
            note(row)
        elif role[0] == "routed":
            projection = role[2]
            experts = [e for e in range(arch.num_experts)
                       if wanted is None
                       or official_expert_name(layer, e, projection, arch) in wanted]
            if not experts:
                continue
            for expert in experts:
                tensor, _row = load_decoded_expert(container, census, layer=layer,
                                                   expert=expert, projection=projection,
                                                   device=device)
                finish(official_expert_name(layer, expert, projection, arch), tensor)
            note(row)
        else:  # pragma: no cover - the census refused these already
            raise _fail(f"{gguf_name}: unmapped tensor reached materialize_layer")
    for mla_layer, halves in mla_seen.items():
        if set(halves) != {"k_b", "v_b"}:
            raise _fail(f"layer {mla_layer}: MLA halves incomplete ({sorted(halves)})")
        finish(kv_b_hf_name(mla_layer, arch),
               reconstruct_kv_b(container, census, mla_layer, device=device))
    if wanted is not None and set(out) != wanted:
        missing = sorted(wanted - set(out))[:5]
        raise _fail(f"layer {layer} does not carry the requested tensors {missing}")
    return out


def materialize_plan(surface: GgufSurface) -> Dict[str, Any]:
    """What ``materialize_layer`` will do, as a receipt-ready summary (no reads)."""
    arch, census = surface.arch, surface.census
    partition = layer_partition(census)
    return {
        "architecture": arch.key,
        "family": arch.family,
        "layer_prefix": arch.layer_prefix,
        "decoder_layers": arch.mtp_layer,
        "mtp_layer": arch.mtp_layer,
        "layers_with_tensors": sorted(l for l in partition if l != RESIDENT_LAYER),
        "resident_tensors": [census.direct_map[n] for n in partition.get(RESIDENT_LAYER, [])],
        "routed_layers": list(arch.routed_layers),
        "experts_per_layer": arch.num_experts,
        "mla_layers": list(census.mla_layers),
        "kv_b_arrangement": MLA_KV_B_ARRANGEMENT,
        "kv_b_shape": [arch.kv_b_rows, arch.mla_kv_lora_rank],
        "shared_indexer_copies_not_loaded": len(census.shared_indexer_copies),
        "official_f32_suffixes": list(arch.official_f32_suffixes),
        "type_census": dict(surface.type_census),
    }

# ---------------------------------------------------------------------------
# materialized non-routed view (decoded safetensors under official HF names)
# ---------------------------------------------------------------------------

def _official_dtype_for(hf_name: str, arch: GgufArch = GLM5NEXT) -> str:
    return arch.official_dtype_for(hf_name)


def safetensors_header(path: Path) -> Dict[str, Any]:
    """The JSON header of a safetensors file (name -> {dtype, shape, offsets})."""
    with Path(path).open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


_ST_DTYPE_TO_TORCH = {"BF16": "bfloat16", "F32": "float32", "F16": "float16",
                      "F64": "float64", "I64": "int64", "I32": "int32", "I16": "int16",
                      "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def verify_official_dtypes(bf16_root: Path, weight_map: Mapping[str, str],
                           names: List[str], arch: GgufArch = GLM5NEXT) -> Dict[str, Any]:
    """Check OFFICIAL_F32_SUFFIXES against the official tree's ACTUAL dtypes.

    The view has to be dtype-identical to a native build, and the suffix list is
    a claim about the released checkpoint that could go stale.  Wherever the
    official shard is present this reads the real dtype out of its safetensors
    header and refuses on any disagreement; shards that are absent (the GGUF
    lane only strictly needs the vision-carrying ones) are COUNTED, not assumed
    away, and the count lands in the view receipt.
    """
    bf16_root = Path(bf16_root)
    by_shard: Dict[str, List[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if shard is None:
            raise _fail(f"official index has no entry for {name}")
        by_shard.setdefault(shard, []).append(name)
    verified = 0
    unreadable_shards = []
    disagreements = []
    for shard, shard_names in sorted(by_shard.items()):
        path = bf16_root / shard
        if not path.is_file():
            unreadable_shards.append(shard)
            continue
        header = safetensors_header(path)
        for name in shard_names:
            info = header.get(name)
            if info is None:
                raise _fail(f"{shard} does not carry {name} despite the index saying so")
            actual = _ST_DTYPE_TO_TORCH.get(info["dtype"], info["dtype"])
            if actual != _official_dtype_for(name, arch):
                disagreements.append((name, actual, _official_dtype_for(name, arch)))
            verified += 1
    if disagreements:
        listed = ", ".join(f"{n}: official {a}, policy {p}" for n, a, p in disagreements[:5])
        raise _fail(
            f"the official dtype policy is WRONG for {len(disagreements)} tensors ({listed}). "
            "OFFICIAL_F32_SUFFIXES no longer describes the released checkpoint; the view would "
            "not be dtype-identical to a native build"
        )
    return {"tensors_checked_against_official_headers": verified,
            "tensors_unchecked": len(names) - verified,
            "shards_absent": len(unreadable_shards),
            "policy_disagreements": 0}


def materialize_nonrouted_view(
    surface: GgufSurface,
    bf16_root: Path,
    work_dir: Path,
    *,
    shard_bytes: int = 4 << 30,
    progress: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Decode every non-routed tensor into a from_pretrained-able directory.

    The view carries the official HF names/dtypes/shapes: decoded GGUF tensors
    (bf16, or byte-exact F32 for the official-F32 set), the reconstructed
    kv_b_proj per DSA layer, and the vision tower copied VERBATIM from the
    official BF16 tree (the only part the artifact does not carry).  config /
    tokenizer / preprocessor files come from the BF16 tree.  Routed experts are
    ABSENT by construction -- stream_score's loader assertions
    (missing==0, stray==0) apply unchanged.

    Reused across runs via a fingerprint stamp: the view is rebuilt only when
    the artifact identity or this adapter changes.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    official_map: Mapping[str, str] = bf16_index["weight_map"]
    bijection = verify_nonrouted_bijection(surface.census, official_map.keys())
    vision_names = sorted(name for name in official_map if name.startswith("model.visual."))
    dtype_audit = verify_official_dtypes(bf16_root, official_map,
                                         surface.census.nonrouted_hf_names(), surface.arch)

    view = Path(work_dir).resolve() / "gguf-nonrouted-view"
    stamp_path = view / "gguf-view-receipt.json"
    fingerprint = {
        "schema": GGUF_VIEW_RECEIPT_SCHEMA,
        "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "vision_tensor_count": len(vision_names),
    }
    if stamp_path.is_file():
        stamp = _read_json(stamp_path, "gguf view stamp")
        if all(stamp.get(key) == value for key, value in fingerprint.items()):
            record = dict(stamp)
            record["reused"] = True
            return view, record
    view.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    container, census = surface.container, surface.census
    jobs: List[Tuple[str, str]] = []          # (hf_name, kind:gguf_name|mla:layer|vision)
    for gguf_name, hf_name in census.direct_map.items():
        jobs.append((hf_name, "gguf:" + gguf_name))
    for layer in census.mla_layers:
        jobs.append((kv_b_hf_name(layer, surface.arch), "mla:%d" % layer))
    for name in vision_names:
        jobs.append((name, "vision"))
    jobs.sort(key=lambda item: item[0])

    bf16_handles: Dict[str, Any] = {}

    def official_tensor(name: str):
        shard = official_map[name]
        handle = bf16_handles.get(shard)
        if handle is None:
            shard_path = bf16_root / shard
            if not shard_path.is_file():
                raise _fail(
                    f"BF16 shard {shard} absent ({name}); the GGUF view needs only "
                    "the vision-carrying shards of the official tree"
                )
            handle = safe_open(str(shard_path), framework="pt", device="cpu")
            bf16_handles[shard] = handle
        return handle.get_tensor(name)

    shard_index = 0
    current: Dict[str, Any] = {}
    current_bytes = 0
    total_tensor_bytes = 0
    weight_map: Dict[str, str] = {}
    shard_files: List[str] = []
    tensor_manifest: List[Dict[str, Any]] = []
    counts = {"decoded_bf16": 0, "f32_passthrough": 0, "mla_reconstructed": 0,
              "vision_copied": 0}

    def flush() -> None:
        nonlocal shard_index, current, current_bytes
        if not current:
            return
        name = "gguf-view-%05d.safetensors" % shard_index
        save_file(current, str(view / name), metadata={"format": "pt"})
        shard_files.append(name)
        for tensor_name in current:
            weight_map[tensor_name] = name
        shard_index += 1
        current = {}
        current_bytes = 0

    for hf_name, kind in jobs:
        if kind == "vision":
            tensor = official_tensor(hf_name)
            counts["vision_copied"] += 1
            source = "official_bf16_vision"
        elif kind.startswith("mla:"):
            layer = int(kind.split(":", 1)[1])
            tensor = reconstruct_kv_b(container, census, layer).to(torch.bfloat16)
            counts["mla_reconstructed"] += 1
            source = "mla_reconstruction"
        else:
            gguf_name = kind.split(":", 1)[1]
            row = container.tensors[gguf_name]
            decoded = load_decoded_tensor(container, gguf_name)
            if _official_dtype_for(hf_name, surface.arch) == "float32":
                if row["type"] != "F32":
                    raise _fail(
                        f"{hf_name} is float32 in the official tree but {row['type']} "
                        "in the GGUF - dtype policy would not round-trip"
                    )
                tensor = decoded  # byte-exact F32 passthrough
                counts["f32_passthrough"] += 1
            else:
                tensor = decoded.to(torch.bfloat16)
                counts["decoded_bf16"] += 1
            source = "gguf:" + row["type"]
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if current_bytes + nbytes > shard_bytes and current:
            flush()
        current[hf_name] = tensor
        current_bytes += nbytes
        total_tensor_bytes += nbytes
        tensor_manifest.append({"tensor": hf_name, "source": source,
                                "dtype": str(tensor.dtype).replace("torch.", ""),
                                "shape": list(tensor.shape)})
        if progress and len(tensor_manifest) % 400 == 0:
            print(json.dumps({"gguf_view_progress": len(tensor_manifest),
                              "of": len(jobs)}), flush=True)
    flush()
    for handle in bf16_handles.values():
        close = getattr(handle, "__exit__", None)
        if close is not None:
            try:
                close(None, None, None)
            except Exception:  # noqa: BLE001 - handle cleanup is best-effort
                pass

    (view / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_tensor_bytes},
                    "weight_map": weight_map}),
        encoding="utf-8",
    )
    for entry in bf16_root.iterdir():
        if entry.name == "model.safetensors.index.json" or entry.is_dir():
            continue
        if entry.suffix in (".json", ".jinja", ".txt", ".model"):
            target = view / entry.name
            if not target.exists():
                target.write_bytes(entry.read_bytes())

    record = dict(fingerprint)
    record.update({
        "view_path": str(view),
        "tensor_count": len(weight_map),
        "tensor_bytes": total_tensor_bytes,
        "shard_count": len(shard_files),
        "counts": counts,
        "bijection": bijection,
        "official_dtype_audit": dtype_audit,
        "tensor_manifest_sha256": _sha256_bytes(_canonical_json(tensor_manifest)),
        "elapsed_seconds": time.monotonic() - started,
        "reused": False,
    })
    stamp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return view, record


def verify_view_nonrouted_values(
    surface: GgufSurface, view: Path, *, sample: int = 8
) -> Dict[str, Any]:
    """Spot-check the materialized view against a fresh decode (regression net)."""
    import torch
    from safetensors import safe_open

    index = _read_json(Path(view) / "model.safetensors.index.json", "view index")
    weight_map = index["weight_map"]
    names = sorted(surface.census.direct_map.items())
    import numpy as np

    rng = np.random.default_rng(0x66F)
    picks = [names[int(i)] for i in rng.choice(len(names), size=min(sample, len(names)),
                                               replace=False)]
    handles: Dict[str, Any] = {}
    compared = 0
    for gguf_name, hf_name in picks:
        shard = weight_map[hf_name]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(Path(view) / shard), framework="pt", device="cpu")
            handles[shard] = handle
        stored = handle.get_tensor(hf_name)
        fresh = load_decoded_tensor(surface.container, gguf_name)
        if _official_dtype_for(hf_name, surface.arch) == "float32":
            expected = fresh
        else:
            expected = fresh.to(torch.bfloat16)
        if not torch.equal(stored, expected):
            raise _fail(f"materialized view differs from a fresh decode: {hf_name}")
        compared += 1
    return {"spot_checked_tensors": compared, "all_equal": True}


def gguf_reader_identity(runner_path, *, surface: GgufSurface) -> Dict[str, Any]:
    """Identity binding this adapter + the runner (mirrors dione_reader_identity)."""
    body = {
        "schema": GGUF_READER_IDENTITY_SCHEMA,
        "mode": "offline_gguf_block_dequant_to_bf16_all_tensors_for_logit_measurement",
        "serving_kernel": False,
        "bits": None,
        "codebook": "ggml-block-quants",
        "supported_types": list(SUPPORTED_TYPES),
        "type_census": dict(surface.type_census),
        "mla_kv_b_arrangement": MLA_KV_B_ARRANGEMENT,
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    body["runtime_reader_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


def surface_summary(surface: GgufSurface) -> Dict[str, Any]:
    return {
        "schema": GGUF_SURFACE_SCHEMA,
        "gguf_repo": surface.repo,
        "gguf_revision": surface.revision,
        "architecture": surface.architecture,
        "format": GGUF_FORMAT,
        "files": list(surface.file_records),
        "file_hash_verification": surface.file_hash_verification,
        "tensor_count": len(surface.container.tensors),
        "type_census": dict(surface.type_census),
        "quant_metadata": surface.quant_metadata,
        "scope_policy": surface.scope_policy,
        "streamed_routed_modules": (len(surface.arch.routed_layers) * surface.arch.num_experts
                                    * len(PROJECTIONS)),
        "mtp_layer_%d_experts" % surface.arch.mtp_layer:
            "present_in_artifact_identity_never_streamed_or_executed",
        "shared_indexer_copies_not_loaded": len(surface.census.shared_indexer_copies),
        "mla_reconstructed_layers": list(surface.census.mla_layers),
        "mla_kv_b_arrangement": MLA_KV_B_ARRANGEMENT,
        "nonrouted_tensors_from_artifact": len(surface.census.nonrouted_hf_names()),
        "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
        "seal_disclosure": SEAL_DISCLOSURE,
    }


# ---------------------------------------------------------------------------
# standalone CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dry-run", help="validate an artifact from headers alone "
                                       "(local paths or https URLs, no weight reads)")
    p.add_argument("--file", action="append", required=True, dest="files",
                   help="every .gguf of the artifact (repeat; local path or https URL)")
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--bf16-index", type=Path,
                   help="official BF16 model.safetensors.index.json for the "
                        "non-routed bijection census (optional but recommended)")

    p = sub.add_parser("verify-files", help="sha256 every local file; write the marker")
    p.add_argument("--file", action="append", required=True, dest="files")

    p = sub.add_parser("scope", help="emit the per-tensor-class recipe MEASURED "
                                     "from the container's own tensor table")
    p.add_argument("--file", action="append", required=True, dest="files")
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--out", type=Path, help="write here instead of stdout")

    p = sub.add_parser("audit-mla", help="re-prove the kv_b_proj reconstruction "
                                         "arrangement against the official BF16 tensor")
    p.add_argument("--file", action="append", required=True, dest="files")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--bf16", type=Path, required=True,
                   help="official BF16 root (index + the shard carrying kv_b_proj)")
    p.add_argument("--repo")
    p.add_argument("--revision")

    p = sub.add_parser("audit-expert", help="prove the fused tensor's expert SLOT ordering "
                                            "and orientation against the official BF16 expert")
    p.add_argument("--file", action="append", required=True, dest="files")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--projection", default="gate_proj", choices=PROJECTIONS)
    p.add_argument("--bf16", type=Path, required=True,
                   help="official BF16 root (index + the shard carrying that expert)")
    p.add_argument("--repo")
    p.add_argument("--revision")

    args = parser.parse_args()
    if args.command == "verify-files":
        record = verify_file_hashes(args.files)
        print(json.dumps(record, sort_keys=True))
        return 0

    surface = load_gguf_surface(
        args.files, repo=getattr(args, "repo", None),
        revision=getattr(args, "revision", None),
        require_file_hashes=False,
    )
    summary = surface_summary(surface)
    if args.command == "scope":
        report = scope_report(surface)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if getattr(args, "out", None):
            args.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "dry-run":
        if args.bf16_index and args.bf16_index.is_file():
            official = json.loads(args.bf16_index.read_text(encoding="utf-8"))["weight_map"]
            summary["nonrouted_bijection"] = verify_nonrouted_bijection(
                surface.census, official.keys())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command in ("audit-mla", "audit-expert"):
        from safetensors import safe_open

        bf16_root = args.bf16.resolve()
        index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")

        def _official(name: str):
            shard = index["weight_map"].get(name)
            if shard is None:
                raise _fail(f"official index lacks {name}")
            with safe_open(str(bf16_root / shard), framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)

        if args.command == "audit-mla":
            summary["mla_placement_audit"] = audit_mla_placement(
                surface.container, surface.census, layer=args.layer,
                official_bf16=_official(kv_b_hf_name(args.layer)))
        else:
            name = official_expert_name(args.layer, args.expert, args.projection)
            decoded, _ = load_decoded_expert(
                surface.container, surface.census,
                layer=args.layer, expert=args.expert, projection=args.projection)
            summary["expert_placement_audit"] = audit_expert_placement(
                decoded, _official(name), label=name)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise _fail(f"unknown command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
