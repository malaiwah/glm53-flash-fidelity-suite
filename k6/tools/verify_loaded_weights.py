#!/usr/bin/env python3
"""Does a loaded model hold the published bytes -- when the published bytes are QUANTIZED?

What this adds to `k6/tools/verify_fused_experts.py`
----------------------------------------------------
That tool answers the same question for a BF16 root: read the raw little-endian
bytes of each checkpoint tensor at its published offset and `memcmp` them
against the live parameter.  A `memcmp` is the right instrument exactly when the
loader is supposed to be a byte mover.

For `deepseek-ai/DeepSeek-V4-Flash-0731` it is not.  That checkpoint ships

  * attention and shared-expert projections as **FP8 E4M3** payload plus a
    **UE8M0** (`float8_e8m0fnu`) per-128x128-block scale under a sibling
    `.scale` key, and
  * all 256 routed experts per layer as **FP4 E2M1 packed two-per-byte in
    int8**, with a per-row/32-column UE8M0 scale,

and on a machine with no FP8-capable GPU `transformers` DEQUANTIZES both to
bf16 while loading.  So the live parameter is not a copy of any byte range: it
is the output of an arithmetic pipeline -- unpack, scale, round -- laid on top
of the expert-fusion converter (`MergeModulelist` + `Concatenate`) that
`docs/GLM53-ROOT-FEASIBILITY.md` R1 was written about.  A `memcmp` cannot see
whether that pipeline is right, and a statistic ("the mean looks fine") cannot
either: a swapped nibble order, a transposed scale grid, an off-by-one block
size or a gate/up swap all produce plausible-looking numbers.

What is and is not re-implemented here
--------------------------------------
The NAME map (checkpoint key -> live parameter) is a restatement of the
architecture's published renames.  It is not the interesting half and it is not
claimed as independent: a wrong rename cannot silently corrupt a value, it can
only fail loudly, and the load report's `missing_keys == 0` /
`unexpected_keys == 0` is already an independent check that the map is onto.

The VALUE pipeline is re-implemented from the FORMAT, in numpy, with no
`transformers` and no `safetensors` in the path -- because that is the half
that can be wrong quietly:

  * E4M3 and E8M0 are decoded through 256-entry tables built arithmetically
    from the bit fields, not by casting through `torch`'s float8 dtypes;
  * E2M1 through the 16-entry value table, low nibble first;
  * the block scale grid is derived from the two shapes, not from the config;
  * the fp32 product is rounded to bf16 with round-half-to-even, by hand.

  * and the fusion geometry -- which half of `gate_up_proj` is gate, which axis
    the experts stack on -- is asserted by the comparison rather than taken
    from the converter that produced it.

Then it compares the result to the live parameter **bit for bit**.  A pass is
"every published tensor, decoded independently, is exactly the tensor the model
is about to compute with".  A statistic is never reported and never sufficient.

Coverage is stated in both directions, because either half alone is a lie:

  * every CHECKPOINT tensor must land in a live parameter (or be named as
    deliberately unused);
  * every live PARAMETER must be covered by checkpoint tensors (or be named as
    a derived buffer the checkpoint never ships).

The model is loaded through `k6/tools/hf_capture.load_model` -- the exact code
path `bin/fidelity-dataset capture --engine hf-transformers` uses -- so a pass
is a statement about the production engine.

Usage:
  verify_loaded_weights.py --model <ckpt dir> --plan deepseek_v4 \\
      [--receipt <fetch receipt>] [--experts all|0,7,255] [--out report.json] \\
      [--drop-parallel-plan]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hf_capture  # noqa: E402
from verify_fused_experts import ShardReader  # noqa: E402


def die(message: str) -> "SystemExit":
    print("verify_loaded_weights: ERROR: %s" % message, file=sys.stderr, flush=True)
    return SystemExit(2)


def log(**fields: Any) -> None:
    print(json.dumps(fields, sort_keys=True, default=str), flush=True)


# ---------------------------------------------------------------------------
# decode, from the format
# ---------------------------------------------------------------------------


def _e4m3_table():
    """The 256 values of `float8_e4m3fn`, computed from the bit fields.

    1 sign, 4 exponent (bias 7), 3 mantissa. No infinities; exponent 15 with
    mantissa 7 is the only NaN. Subnormals when the exponent field is 0.
    """
    import numpy as np

    out = np.zeros(256, dtype=np.float64)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0x0F
        mantissa = byte & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            out[byte] = np.nan
        elif exponent == 0:
            out[byte] = sign * (mantissa / 8.0) * (2.0 ** -6)
        else:
            out[byte] = sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return out.astype(np.float32)


def _e8m0_table():
    """`float8_e8m0fnu`: a bare 8-bit exponent. value = 2**(byte-127); 255 = NaN."""
    import numpy as np

    out = np.zeros(256, dtype=np.float64)
    for byte in range(256):
        out[byte] = np.nan if byte == 0xFF else 2.0 ** (byte - 127)
    return out.astype(np.float32)


# E2M1, low nibble first. 1 sign, 2 exponent (bias 1), 1 mantissa.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_TABLES: Dict[str, Any] = {}


def table(name: str):
    if name not in _TABLES:
        import numpy as np

        _TABLES[name] = {"e4m3": _e4m3_table, "e8m0": _e8m0_table,
                         "e2m1": lambda: np.array(_E2M1, dtype=np.float32)}[name]()
    return _TABLES[name]


def decode_payload(raw: bytes, dtype: str, shape: Sequence[int]):
    """Raw safetensors bytes -> a float32 numpy array of the LOGICAL shape.

    For FP4 the logical shape is twice the stored width on the last axis:
    two `e2m1` values live in each int8 byte, low nibble first.
    """
    import numpy as np

    buffer = np.frombuffer(raw, dtype=np.uint8)
    if dtype == "F8_E4M3":
        return table("e4m3")[buffer].reshape(tuple(shape))
    if dtype == "F8_E8M0":
        return table("e8m0")[buffer].reshape(tuple(shape))
    if dtype == "I8":
        low = table("e2m1")[buffer & 0x0F]
        high = table("e2m1")[(buffer >> 4) & 0x0F]
        pairs = np.stack([low, high], axis=-1).reshape(-1)
        return pairs.reshape(tuple(shape[:-1]) + (2 * int(shape[-1]),))
    if dtype == "BF16":
        wide = np.zeros(buffer.size // 2, dtype=np.uint32)
        wide |= buffer.view("<u2").astype(np.uint32) << 16
        return wide.view(np.float32).reshape(tuple(shape))
    if dtype == "F32":
        return buffer.view("<f4").reshape(tuple(shape))
    if dtype == "F16":
        return buffer.view("<f2").astype(np.float32).reshape(tuple(shape))
    if dtype == "I64":
        return buffer.view("<i8").reshape(tuple(shape))
    raise die("no decoder for safetensors dtype %r" % dtype)


def apply_block_scale(values, scales):
    """Multiply a matrix by a per-block scale grid derived from the two shapes.

    The block size is `values.shape // scales.shape`, not a config value: MoE
    experts ship a `[1, 32]` grid and dense linears a `[128, 128]` one inside
    the SAME checkpoint, so reading the block size off the config would be
    right for one of them and silently wrong for the other.
    """
    import numpy as np

    rows, cols = values.shape[-2:]
    scale_rows, scale_cols = scales.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise die("weight %r is not divisible by its scale grid %r"
                  % ((rows, cols), (scale_rows, scale_cols)))
    block_m, block_n = rows // scale_rows, cols // scale_cols
    grid = values.reshape(scale_rows, block_m, scale_cols, block_n)
    return (grid * scales.reshape(scale_rows, 1, scale_cols, 1).astype(np.float32)
            ).reshape(rows, cols)


def to_bf16_bits(values):
    """float32 -> the uint16 bf16 bit patterns, round-half-to-even.

    Written out rather than delegated to `torch.Tensor.to(bfloat16)`: the cast
    is part of what is being checked, so borrowing the implementation under
    test would make a pass unfalsifiable.
    """
    import numpy as np

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits.astype(np.uint64) + 0x7FFF + lsb.astype(np.uint64)
    out = (rounded >> 16).astype(np.uint16)
    # NaN must stay NaN: rounding an all-ones payload can carry into the
    # exponent and produce an infinity.
    nan = np.isnan(values)
    if nan.any():
        out = out.copy()
        out[nan] = 0x7FC0
    return out


# ---------------------------------------------------------------------------
# live-parameter bytes
# ---------------------------------------------------------------------------


def live_bits(tensor):
    """A live parameter's bits, as the widest safe integer view."""
    import numpy as np
    import torch

    flat = tensor.detach().to("cpu").contiguous()
    if flat.dtype == torch.bfloat16:
        return "bf16", flat.view(torch.uint16).numpy()
    if flat.dtype == torch.float32:
        return "f32", flat.numpy().view(np.uint32)
    if flat.dtype == torch.float16:
        return "f16", flat.view(torch.uint16).numpy()
    if flat.dtype in (torch.int64, torch.long):
        return "i64", flat.numpy()
    return str(flat.dtype), flat.numpy()


def expected_bits(values, kind: str):
    """The decoded float32 values, re-expressed in the live parameter's dtype."""
    import numpy as np

    if kind == "bf16":
        return to_bf16_bits(values)
    if kind == "f32":
        return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if kind == "f16":
        return np.ascontiguousarray(values, dtype=np.float16).view(np.uint16)
    if kind == "i64":
        return np.ascontiguousarray(values, dtype=np.int64)
    raise die("no expectation builder for live dtype %r" % kind)


# ---------------------------------------------------------------------------
# plans: checkpoint key -> (live parameter, destination slice)
# ---------------------------------------------------------------------------


class Piece(object):
    """One checkpoint tensor's contribution to one live parameter."""

    def __init__(self, key: str, param: str, index: Optional[Tuple] = None,
                 scale_key: Optional[str] = None) -> None:
        self.key = key
        self.param = param
        self.index = index          # numpy index into the live parameter, or None
        self.scale_key = scale_key


DSV4_RENAMES: List[Tuple[str, str]] = [
    # (checkpoint regex, live replacement). Written from the CHECKPOINT's names
    # and the model's module tree, not read out of transformers' mapping table:
    # the mapping table is part of what is under test.
    (r"^embed\.weight$", "model.embed_tokens.weight"),
    (r"^head\.weight$", "lm_head.weight"),
    (r"^norm\.weight$", "model.norm.weight"),
    (r"^hc_head_fn$", "model.hc_head.hc_fn"),
    (r"^hc_head_base$", "model.hc_head.hc_base"),
    (r"^hc_head_scale$", "model.hc_head.hc_scale"),
    (r"^layers\.(\d+)\.attn_norm\.weight$", r"model.layers.\1.input_layernorm.weight"),
    (r"^layers\.(\d+)\.ffn_norm\.weight$",
     r"model.layers.\1.post_attention_layernorm.weight"),
    (r"^layers\.(\d+)\.hc_attn_fn$", r"model.layers.\1.attn_hc.fn"),
    (r"^layers\.(\d+)\.hc_attn_base$", r"model.layers.\1.attn_hc.base"),
    (r"^layers\.(\d+)\.hc_attn_scale$", r"model.layers.\1.attn_hc.scale"),
    (r"^layers\.(\d+)\.hc_ffn_fn$", r"model.layers.\1.ffn_hc.fn"),
    (r"^layers\.(\d+)\.hc_ffn_base$", r"model.layers.\1.ffn_hc.base"),
    (r"^layers\.(\d+)\.hc_ffn_scale$", r"model.layers.\1.ffn_hc.scale"),
    (r"^layers\.(\d+)\.attn\.attn_sink$", r"model.layers.\1.self_attn.sinks"),
    (r"^layers\.(\d+)\.attn\.q_norm\.weight$",
     r"model.layers.\1.self_attn.q_a_norm.weight"),
    (r"^layers\.(\d+)\.attn\.kv_norm\.weight$",
     r"model.layers.\1.self_attn.kv_norm.weight"),
    (r"^layers\.(\d+)\.attn\.wq_a\.weight$", r"model.layers.\1.self_attn.q_a_proj.weight"),
    (r"^layers\.(\d+)\.attn\.wq_b\.weight$", r"model.layers.\1.self_attn.q_b_proj.weight"),
    (r"^layers\.(\d+)\.attn\.wkv\.weight$", r"model.layers.\1.self_attn.kv_proj.weight"),
    (r"^layers\.(\d+)\.attn\.wo_a\.weight$", r"model.layers.\1.self_attn.o_a_proj.weight"),
    (r"^layers\.(\d+)\.attn\.wo_b\.weight$", r"model.layers.\1.self_attn.o_b_proj.weight"),
    (r"^layers\.(\d+)\.attn\.compressor\.wkv\.weight$",
     r"model.layers.\1.self_attn.compressor.kv_proj.weight"),
    (r"^layers\.(\d+)\.attn\.compressor\.wgate\.weight$",
     r"model.layers.\1.self_attn.compressor.gate_proj.weight"),
    (r"^layers\.(\d+)\.attn\.compressor\.norm\.weight$",
     r"model.layers.\1.self_attn.compressor.kv_norm.weight"),
    (r"^layers\.(\d+)\.attn\.compressor\.ape$",
     r"model.layers.\1.self_attn.compressor.position_bias"),
    (r"^layers\.(\d+)\.attn\.indexer\.compressor\.wkv\.weight$",
     r"model.layers.\1.self_attn.compressor.indexer.kv_proj.weight"),
    (r"^layers\.(\d+)\.attn\.indexer\.compressor\.wgate\.weight$",
     r"model.layers.\1.self_attn.compressor.indexer.gate_proj.weight"),
    (r"^layers\.(\d+)\.attn\.indexer\.compressor\.norm\.weight$",
     r"model.layers.\1.self_attn.compressor.indexer.kv_norm.weight"),
    (r"^layers\.(\d+)\.attn\.indexer\.compressor\.ape$",
     r"model.layers.\1.self_attn.compressor.indexer.position_bias"),
    (r"^layers\.(\d+)\.attn\.indexer\.weights_proj\.weight$",
     r"model.layers.\1.self_attn.compressor.indexer.scorer.weights_proj.weight"),
    (r"^layers\.(\d+)\.attn\.indexer\.wq_b\.weight$",
     r"model.layers.\1.self_attn.compressor.indexer.q_b_proj.weight"),
    (r"^layers\.(\d+)\.ffn\.gate\.weight$", r"model.layers.\1.mlp.gate.weight"),
    (r"^layers\.(\d+)\.ffn\.gate\.bias$",
     r"model.layers.\1.mlp.gate.e_score_correction_bias"),
    (r"^layers\.(\d+)\.ffn\.gate\.tid2eid$", r"model.layers.\1.mlp.gate.tid2eid"),
    (r"^layers\.(\d+)\.ffn\.shared_experts\.w1\.weight$",
     r"model.layers.\1.mlp.shared_experts.gate_proj.weight"),
    (r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.weight$",
     r"model.layers.\1.mlp.shared_experts.down_proj.weight"),
    (r"^layers\.(\d+)\.ffn\.shared_experts\.w3\.weight$",
     r"model.layers.\1.mlp.shared_experts.up_proj.weight"),
]

# `layers.L.ffn.experts.E.wN.weight` -> a slice of a fused 3-D parameter.
# gate_up_proj is [E, 2*I, H] with w1 (gate) on top of w3 (up); down_proj is
# [E, H, I]. Both orders are asserted here rather than assumed: getting them
# backwards is precisely the failure a key-set diff cannot see.
DSV4_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


def dsv4_plan(reader: ShardReader, keys: Sequence[str], model, config,
              experts: Optional[Sequence[int]]) -> Tuple[List[Piece], List[str]]:
    pieces: List[Piece] = []
    unplanned: List[str] = []
    intermediate = int(getattr(config, "moe_intermediate_size"))
    compiled = [(re.compile(pattern), target) for pattern, target in DSV4_RENAMES]
    for key in keys:
        if key.endswith(".scale"):
            continue                                  # consumed with its weight
        scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
        if scale_key is not None and not reader.has(scale_key):
            scale_key = None
        match = DSV4_EXPERT.match(key)
        if match is not None:
            layer, expert, which = int(match.group(1)), int(match.group(2)), match.group(3)
            if experts is not None and expert not in experts:
                continue
            if which == "w2":
                pieces.append(Piece(key, "model.layers.%d.mlp.experts.down_proj" % layer,
                                    (expert, slice(None), slice(None)), scale_key))
            else:
                lo = 0 if which == "w1" else intermediate
                pieces.append(Piece(key, "model.layers.%d.mlp.experts.gate_up_proj" % layer,
                                    (expert, slice(lo, lo + intermediate), slice(None)),
                                    scale_key))
            continue
        target = None
        for pattern, replacement in compiled:
            if pattern.match(key):
                target = pattern.sub(replacement, key)
                break
        if target is None:
            unplanned.append(key)
            continue
        pieces.append(Piece(key, target, None, scale_key))
    return pieces, unplanned


# ---------------------------------------------------------------------------
# minimax_m3_vl
# ---------------------------------------------------------------------------

MM3_RENAMES: List[Tuple[str, str]] = [
    (r"^language_model\.lm_head\.(.*)$", r"lm_head.\1"),
    (r"^vision_tower\.vision_model\.embeddings\.patch_embedding\.(.*)$",
     r"model.vision_tower.embeddings.proj.\1"),
    (r"^vision_tower\.vision_model\.encoder\.layers\.(.*)$",
     r"model.vision_tower.layers.\1"),
    (r"^vision_tower\.vision_model\.pre_layrnorm\.(.*)$",
     r"model.vision_tower.pre_layrnorm.\1"),
    (r"^multi_modal_projector\.(linear_[12]\..*)$", r"model.multi_modal_projector.\1"),
    (r"^patch_merge_mlp\.linear_([12])\.(.*)$",
     r"model.multi_modal_projector.merge_linear_\1.\2"),
    (r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.e_score_correction_bias$",
     r"model.language_model.layers.\1.mlp.gate.e_score_correction_bias"),
    (r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.(.*)$",
     r"model.language_model.layers.\1.mlp.\2"),
    (r"^language_model\.model\.layers\.(\d+)\.self_attn\.index_([qk])_(proj|norm)\.(.*)$",
     r"model.language_model.layers.\1.self_attn.indexer.\2_\3.\4"),
    (r"^language_model\.model\.(.*)$", r"model.language_model.\1"),
]

MM3_EXPERT = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.weight$")
# The dense layers (moe_layer_freq == 0) and every shared expert ship gate_proj
# and up_proj separately; the model holds one `gate_up_proj.weight` that is
# `Concatenate(dim=0)` of the two, gate first.
MM3_GATE_UP = re.compile(r"^(.*)\.(gate|up)_proj\.weight$")


def mm3_plan(reader: ShardReader, keys: Sequence[str], model, config,
             experts: Optional[Sequence[int]]) -> Tuple[List[Piece], List[str]]:
    pieces: List[Piece] = []
    unplanned: List[str] = []
    text = getattr(config, "text_config", config)
    compiled = [(re.compile(pattern), target) for pattern, target in MM3_RENAMES]
    params = dict(model.named_parameters())
    for key in keys:
        match = MM3_EXPERT.match(key)
        if match is not None:
            layer, expert, which = int(match.group(1)), int(match.group(2)), match.group(3)
            if experts is not None and expert not in experts:
                continue
            base = "model.language_model.layers.%d.mlp.experts." % layer
            if which == "w2":
                pieces.append(Piece(key, base + "down_proj",
                                    (expert, slice(None), slice(None))))
            else:
                inter = int(getattr(text, "intermediate_size"))
                lo = 0 if which == "w1" else inter
                pieces.append(Piece(key, base + "gate_up_proj",
                                    (expert, slice(lo, lo + inter), slice(None))))
            continue
        target = None
        for pattern, replacement in compiled:
            if pattern.match(key):
                target = pattern.sub(replacement, key)
                break
        if target is None:
            unplanned.append(key)
            continue
        # `gate_proj` + `up_proj` -> one `gate_up_proj`, gate on top. Applied
        # only where the model actually holds the fused parameter, because the
        # same suffix also appears where it does not.
        fused = MM3_GATE_UP.match(target)
        if fused is not None and (fused.group(1) + ".gate_up_proj.weight") in params:
            live = params[fused.group(1) + ".gate_up_proj.weight"]
            half = int(live.shape[0]) // 2
            lo = 0 if fused.group(2) == "gate" else half
            pieces.append(Piece(key, fused.group(1) + ".gate_up_proj.weight",
                                (slice(lo, lo + half), slice(None))))
            continue
        pieces.append(Piece(key, target, None))
    return pieces, unplanned


def direct_plan(reader: ShardReader, keys: Sequence[str], model, config,
                experts: Optional[Sequence[int]]) -> Tuple[List[Piece], List[str]]:
    """Checkpoint key == parameter name, one to one.

    `Qwen/Qwen3.8-Flash-Next` is published this way: the names are already
    `transformers`' own (`model.language_model.layers.N....`) and the routed
    experts are ALREADY FUSED in the checkpoint
    (`mlp.experts.gate_up_proj` is one 3-D tensor on disk), so `qwen4_exp` has
    no entry in `conversion_mapping.py` at all.  There is no converter to be
    wrong -- which is itself the finding, and this plan is how it is checked
    rather than assumed.
    """
    pieces: List[Piece] = []
    unplanned: List[str] = []
    named = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    for key in keys:
        if key in named:
            pieces.append(Piece(key, key, None))
        else:
            unplanned.append(key)
    return pieces, unplanned


PLANS: Dict[str, Callable] = {"deepseek_v4": dsv4_plan, "minimax_m3_vl": mm3_plan,
                              "direct": direct_plan}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="verify_loaded_weights", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True, choices=sorted(PLANS))
    ap.add_argument("--receipt", default=None,
                    help="fetch receipt; its per-tensor sha256 are re-derived from the "
                         "shard bytes read here")
    ap.add_argument("--experts", default="all",
                    help="'all' or a comma-separated list of routed-expert indices")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--drop-parallel-plan", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import hashlib

    import numpy as np

    model, config, info = hf_capture.load_model(
        args.model, args.device, args.dtype,
        drop_parallel_plan=args.drop_parallel_plan)
    text_config = getattr(config, "text_config", config)
    log(stage="loaded", architectures=list(getattr(config, "architectures", None) or []),
        num_hidden_layers=int(getattr(text_config, "num_hidden_layers", -1)))
    report = hf_capture.load_report(info)
    log(stage="load_report", **{k: (len(v) if isinstance(v, (list, set, dict)) else v)
                                for k, v in report.items()})

    reader = ShardReader(args.model)
    # SCOPE. `ShardReader` enumerates each shard's OWN safetensors header, which
    # is the right thing for a whole checkpoint and the wrong thing for a
    # truncation: a sparse local shard still declares every tensor the published
    # shard held, and the ranges we never fetched are HOLES that read as zeros.
    # The pruned `model.safetensors.index.json` is the list of tensors that were
    # actually fetched, so it is the scope of any claim made here. The
    # difference is reported rather than dropped -- it is the same mechanism
    # that put 43 tensors in GLM-5.3 Stage A's `unexpected_keys`, and a tool
    # that quietly compared against a hole would be the exact failure this file
    # exists to prevent.
    present = sorted(reader._index)                     # noqa: SLF001 -- same package
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    outside_index: List[str] = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            named = set(json.load(handle).get("weight_map") or {})
        outside_index = [k for k in present if k not in named]
        keys = [k for k in present if k in named]
    else:
        keys = present
    log(stage="scope", shard_header_tensors=len(present), in_pruned_index=len(keys),
        outside_pruned_index=len(outside_index), outside_sample=outside_index[:4])
    experts = None
    if args.experts != "all":
        experts = [int(v) for v in args.experts.split(",") if v.strip() != ""]
    pieces, unplanned = PLANS[args.plan](reader, keys, model, config, experts)
    log(stage="plan", checkpoint_tensors=len(keys), pieces=len(pieces),
        unplanned=len(unplanned), unplanned_sample=unplanned[:6])

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    digests = (json.load(open(args.receipt)) if args.receipt else {}).get("tensor_digests", {})

    matched = differed = missing_param = 0
    digest_ok = digest_bad = 0
    bytes_compared = 0
    covered_params: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []
    dequantized = 0

    for piece in pieces:
        live = params.get(piece.param)
        if live is None:
            missing_param += 1
            failures.append({"key": piece.key, "status": "NO_SUCH_PARAMETER",
                             "parameter": piece.param})
            continue
        meta = reader.meta(piece.key)
        raw = reader.raw(piece.key)
        if piece.key in digests:
            got = hashlib.sha256(raw).hexdigest()
            if got == digests[piece.key]["sha256"]:
                digest_ok += 1
            else:
                digest_bad += 1
                failures.append({"key": piece.key, "status": "RECEIPT_DIGEST_MISMATCH",
                                 "receipt_sha256": digests[piece.key]["sha256"],
                                 "read_sha256": got})
        values = decode_payload(raw, meta["dtype"], meta["shape"])
        if piece.scale_key is not None:
            scale_meta = reader.meta(piece.scale_key)
            scales = decode_payload(reader.raw(piece.scale_key), scale_meta["dtype"],
                                    scale_meta["shape"])
            values = apply_block_scale(values, scales)
            dequantized += 1
        kind, actual = live_bits(live)
        target = actual if piece.index is None else actual[piece.index]
        want = expected_bits(values, kind).reshape(target.shape)
        bytes_compared += int(target.nbytes)
        if np.array_equal(target, want):
            matched += 1
        else:
            differed += 1
            bad = np.flatnonzero((target != want).reshape(-1))
            failures.append({
                "key": piece.key, "status": "DIFFERS", "parameter": piece.param,
                "live_dtype": kind, "checkpoint_dtype": meta["dtype"],
                "checkpoint_shape": meta["shape"], "compared_shape": list(target.shape),
                "elements_differing": int(bad.size),
                "first_differing_index": int(bad[0]) if bad.size else None,
                "quantized": piece.scale_key is not None})
        covered_params[piece.param] = covered_params.get(piece.param, 0) + 1

    # The other direction. A model whose every checkpoint tensor matched can
    # still hold a parameter nothing filled -- that is the shape of a hole read
    # as zeros -- so the parameters are enumerated too.
    uncovered: List[str] = []
    parameter_names = set(dict(model.named_parameters()))
    for name in sorted(parameter_names):
        if name in covered_params:
            continue
        uncovered.append(name)
    derived = [n for n in uncovered if n.endswith("_inv_freq") or ".inv_freq" in n]
    uncovered = [n for n in uncovered if n not in derived]

    reader.close()
    summary = {
        "schema": "malaiwah.loaded-weight-decode-check.v1",
        "model_dir": os.path.abspath(args.model),
        "plan": args.plan,
        "architectures": list(getattr(config, "architectures", None) or []),
        "checkpoint_tensors_in_pruned_index": len(keys),
        "shard_header_tensors": len(present),
        "shard_tensors_outside_pruned_index": len(outside_index),
        "shard_tensors_outside_pruned_index_sample": outside_index[:8],
        "checkpoint_tensors_compared": matched + differed,
        "checkpoint_tensors_matched_exactly": matched,
        "checkpoint_tensors_differed": differed,
        "checkpoint_tensors_unplanned": unplanned,
        "checkpoint_tensors_dequantized_here": dequantized,
        "parameters_total": len(parameter_names),
        "parameters_covered": len(set(covered_params) & parameter_names),
        "buffers_covered": len(set(covered_params) - parameter_names),
        "parameters_not_covered": uncovered,
        "parameters_derived_from_config": derived,
        "no_such_parameter": missing_param,
        "receipt_digests_rechecked": digest_ok,
        "receipt_digest_mismatches": digest_bad,
        "bytes_compared": bytes_compared,
        "experts_selected": args.experts,
        "parallel_plan_dropped": bool(args.drop_parallel_plan),
        "load_report": {k: sorted(v) if isinstance(v, set) else v
                        for k, v in report.items()},
        "failures": failures[:64],
    }
    summary["verdict"] = ("PASS" if (matched and not differed and not unplanned
                                     and not uncovered and not missing_param
                                     and not digest_bad) else "FAIL")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    log(stage="summary", **{k: v for k, v in summary.items()
                            if k not in ("failures", "load_report")})
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
