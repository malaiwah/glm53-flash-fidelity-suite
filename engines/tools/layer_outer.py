#!/usr/bin/env python3
"""The layer-outer, window-inner capture schedule -- and the streaming residency it needs.

Why this file exists
--------------------
`engines/tools/hf_capture.py` captures a panel the obvious way: load the model, then
push one window at a time through the whole stack.  For a checkpoint that does
not fit in memory that is the wrong loop order, and the cost is not compute --
it is WEIGHT LOADING.  `docs/GLM53-ROOT-FEASIBILITY.md` puts the number on it:
`zai-org/GLM-5.3-BF16` materialises 1,486.8 GB, which is larger than any RAM
configuration we can rent, so the window-outer schedule cannot run at all; the
`--device-map` paths that make it *expressible* pay ~358 GB of host-to-device
traffic per window (B-1) or write a second full 1,486.8 GB copy to disk (B-2).

This module inverts the loop:

    for each layer:  load it once;  for each window: push that window through it;  free it

Every layer's weights are materialised **exactly once for the whole panel**
instead of once per window, and only one layer is resident at a time.

THE NUMBERS DO NOT MOVE
-----------------------
This is a *scheduling* change, never an arithmetic one.  Windows are pushed
through each layer **sequentially, one at a time**, never batched: batching
would change the reduction order of the matmuls and therefore change the
numbers, and a measurement whose numbers moved is worth nothing.  The engine
exists to make a measurement POSSIBLE, not to make it faster.

How bit-identity is obtained, and why it is structural rather than hoped for
---------------------------------------------------------------------------
The naive implementation of a layer-outer loop re-implements the model's
forward: embeddings, position ids, the causal-mask mapping, the rotary
embeddings, the per-layer kwargs, the carried state, the final norm.  Every one
of those is a chance to differ from `transformers` by a detail, and several of
them are architecture-specific in ways that bite exactly on the architecture we
care about.  `GlmMoeDsaModel.forward` threads a SECOND value between layers --

    hidden_states, topk_indices = decoder_layer(..., prev_topk_indices=topk_indices)

-- the DSA indexer's shared top-k selection, which only the `full` indexer
layers recompute; and `Glm5NextTextModel.forward` carries a hyper-channel
dimension (`hc_mult`) plus a different mask builder.  A re-implementation that
knows about "hidden states" and not about those is silently wrong.

So this module re-implements NOTHING.  It runs the model's own
`forward` once per (layer, window) and replaces only the decoder layers with
proxies:

  * a proxy for a layer BELOW the one being computed returns, verbatim, the
    value that layer's successor produced on the previous outer iteration --
    the whole return value, whatever its shape, so `topk_indices` and any other
    carried state ride along untouched;
  * the proxy for the layer being computed calls the real layer and memoises
    its return value;
  * a proxy for a layer ABOVE it raises `_Suspend`, which unwinds the forward.

The model's own prologue therefore builds the embeddings, position ids, masks
and rotary embeddings; the model's own loop body computes the per-layer kwargs
and threads the carried state; the model's own epilogue runs the final norm and
the head.  The only thing this file decides is WHEN each layer runs.  The
per-window arithmetic is the same operations, in the same order, on the same
inputs -- which is why the capture digests compare equal rather than close.

The price is that the prologue is recomputed once per (layer, window) instead
of once per window.  It is an embedding gather, a mask build and a rotary
table: microseconds against a layer of a 753B-parameter MoE.  It is paid on
purpose, to buy an implementation that cannot drift from the model's own code.

Streaming residency
-------------------
Reordering the loop is only half of it.  If the model is fully resident the new
order saves nothing, so this module also builds the model on the meta device
and materialises **one layer at a time** through `transformers`' own
`convert_and_load_state_dict_in_model` -- the same converter, the same
`WeightConverter` chain that fuses 256 per-expert matrices into one tensor, the
same dtype plan.  Reusing it rather than re-deriving the fusion is what makes
the streamed weights byte-identical to the `from_pretrained` weights; that
identity is asserted directly by `selftest_layer_outer.py`, not assumed.

`--layer-residency resident` keeps the new loop order over a fully loaded model.
It buys nothing operationally and exists so that a digest mismatch can be
attributed: `resident` isolates the schedule, `stream` adds the loader.

THE TRAP THIS LOADER WALKS INTO, AND THE GUARD FOR IT
-----------------------------------------------------
Stage A found that `transformers` enumerates each shard's OWN safetensors
header rather than the checkpoint's pruned `model.safetensors.index.json`.  A
per-layer loader reads shard headers directly, so it inherits the same
exposure: against a sparsely-fetched tree a tensor can be *named* by a header
whose bytes were never fetched, and a short read is not an error -- it is
ZEROS, and zeros load without complaint.  `audit_checkpoint_tree` refuses
before the first window on the two signatures that produces: a shard shorter
than its own header requires, and a header/index key-set disagreement.
"""

from __future__ import annotations

import gc
import json
import os
import re
import struct
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEDULE_WINDOW_OUTER = "window-outer"
SCHEDULE_LAYER_OUTER = "layer-outer"
RESIDENCY_STREAM = "stream"
RESIDENCY_RESIDENT = "resident"


class LayerOuterError(Exception):
    """Something about this model or checkpoint the layer-outer engine will not guess at."""


class _Suspend(Exception):
    """Raised by a proxy above the layer being computed, to unwind the forward pass."""


# ---------------------------------------------------------------------------
# locating the decoder stack
# ---------------------------------------------------------------------------


def find_decoder_layers(model) -> Tuple[str, Any]:
    """The text decoder's `nn.ModuleList`, by structure rather than by name.

    Naming alone is not enough: `Glm5NextForConditionalGeneration` carries a
    vision tower whose blocks are also a ModuleList, and its text stack is
    nested two levels down at `model.language_model.layers`.  The structural
    signature of a text decoder stack is that its PARENT also owns the input
    embedding (`embed_tokens`), which the vision tower does not.

    Refuses on zero or several matches rather than picking one: running a
    layer-outer schedule over the wrong ModuleList would produce a capture that
    is wrong in a way no digest of ours would flag.
    """
    import torch

    candidates = []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        parent_name, _, leaf = name.rpartition(".")
        if leaf != "layers":
            continue
        parent = modules.get(parent_name)
        if parent is None or not hasattr(parent, "embed_tokens"):
            continue
        candidates.append((name, module))
    if not candidates:
        raise LayerOuterError(
            "could not find the text decoder's layer list: no `nn.ModuleList` named "
            "'layers' whose parent module also owns `embed_tokens`. The layer-outer "
            "schedule needs to know which modules are the per-layer weights it should "
            "stream, and guessing is worse than refusing. Model class: %s"
            % type(model).__name__)
    if len(candidates) > 1:
        raise LayerOuterError(
            "found %d candidate decoder layer lists (%s); the layer-outer schedule "
            "refuses to pick one. This model needs an explicit selector."
            % (len(candidates), ", ".join(name for name, _ in candidates)))
    return candidates[0]


# ---------------------------------------------------------------------------
# the checkpoint tree audit (the "holes reading as zeros" guard)
# ---------------------------------------------------------------------------


def _safetensors_header(path: str) -> Tuple[Dict[str, Any], int]:
    """(header dict, the byte length the file must have for its own header to be readable)."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise LayerOuterError("%s is shorter than a safetensors header length field "
                                  "(%d bytes)" % (path, len(raw)))
        (header_len,) = struct.unpack("<Q", raw)
        blob = handle.read(header_len)
        if len(blob) != header_len:
            raise LayerOuterError("%s declares a %d-byte header but only %d bytes are "
                                  "present" % (path, header_len, len(blob)))
    header = json.loads(blob.decode("utf-8"))
    end = 0
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") or [0, 0]
        end = max(end, int(offsets[1]))
    return header, 8 + header_len + end


def audit_checkpoint_tree(model_dir: str,
                          shards: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Refuse a checkpoint whose shards can hand back holes instead of weights.

    `shards` restricts the audit to a NAMED SUBSET, for the overlapped fetch of
    `engines/tools/race_fetch.py`: at the moment layer N is about to be loaded, the
    shards for layer N+1 may legitimately still be downloading, and auditing
    them would refuse a tree that is merely incomplete-so-far.  The subset audit
    is not weaker on what it covers -- each named shard is still checked against
    its own header length, and the index/header key-set comparison is still
    exact, restricted on BOTH sides to the named shards.  Every shard is audited
    exactly once, immediately before the first load that reads it.

    Stage A (docs/GLM53-ROOT-FEASIBILITY.md) found that `transformers`
    enumerates each shard's own header, not the pruned index.  A loader that
    reads shards directly -- which this one does, per layer -- can therefore be
    handed a tensor NAME whose BYTES were never fetched.  safetensors does not
    treat a short file as an error at open time; the tensor reads as zeros, the
    load reports nothing, and the capture is a confident measurement of a hole.

    Two signatures are checked, both cheap and both before the first window:

      1. a shard whose on-disk size is smaller than its own header requires --
         the signature of a range-fetched or interrupted download;
      2. a shard header and the checkpoint index disagreeing about which keys
         exist, in either direction -- the signature of a pruned index over a
         complete shard (extra header keys) or a truncated tree (missing ones).

    What it does NOT catch, stated so nobody relies on it: a shard that is the
    right LENGTH but whose bytes were written as zeros (a sparse-file fetch),
    and any corruption that preserves length.  Only a content digest catches
    those, and the checkpoint identity `hf_capture` already computes is that
    digest -- for the tree as a whole, once.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    index_keys: Optional[set] = None
    shard_names: List[str]
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map") or {}
        index_keys = set(weight_map)
        shard_names = sorted(set(weight_map.values()))
        if shards is not None:
            wanted = set(shards)
            unknown = sorted(wanted - set(shard_names))
            if unknown:
                raise LayerOuterError(
                    "REFUSED: asked to audit %d shard(s) the checkpoint index does not "
                    "name: %s. A shard nothing in the index points at holds tensors no "
                    "load will ever ask for by name -- or the audit is being driven from "
                    "a stale plan."
                    % (len(unknown), ", ".join(unknown[:4])))
            shard_names = sorted(wanted)
            index_keys = {key for key, shard in weight_map.items() if shard in wanted}
    else:
        shard_names = sorted(name for name in os.listdir(model_dir)
                             if name.endswith(".safetensors"))
        if shards is not None:
            raise LayerOuterError(
                "REFUSED: a subset audit needs model.safetensors.index.json, which %s "
                "does not have. Without the index there is no map from shard to tensor, "
                "so 'these shards are complete' cannot be stated about a partial tree."
                % model_dir)
    if not shard_names:
        raise LayerOuterError("no *.safetensors shards in %s" % model_dir)

    header_keys: set = set()
    shards: List[Dict[str, Any]] = []
    for name in shard_names:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise LayerOuterError(
                "REFUSED: the checkpoint index names shard %s, which is not present in "
                "%s. A per-layer loader would simply not find that shard's tensors and "
                "the layers they carry would stay unset." % (name, model_dir))
        size = os.path.getsize(path)
        header, required = _safetensors_header(path)
        if size < required:
            raise LayerOuterError(
                "REFUSED: shard %s is %d bytes but its own safetensors header requires "
                "%d. This is the signature of a partially fetched shard, and it is the "
                "dangerous case rather than a loud one: safetensors does not error on a "
                "short file, the missing bytes read as ZEROS, and a capture over them is "
                "a confident number for weights that were never present."
                % (name, size, required))
        keys = {k for k in header if k != "__metadata__"}
        header_keys |= keys
        shards.append({"name": name, "size": size, "tensors": len(keys)})

    if index_keys is not None and index_keys != header_keys:
        only_header = sorted(header_keys - index_keys)
        only_index = sorted(index_keys - header_keys)
        raise LayerOuterError(
            "REFUSED: the shard headers and model.safetensors.index.json disagree about "
            "which tensors this checkpoint holds -- %d named only by a header, %d named "
            "only by the index%s%s. transformers enumerates the HEADERS, so a tensor in "
            "the first group is one a loader will happily read from a region of a file "
            "the index says nothing about. Resolve the tree before capturing."
            % (len(only_header), len(only_index),
               ("; header-only e.g. %s" % ", ".join(only_header[:4])) if only_header else "",
               ("; index-only e.g. %s" % ", ".join(only_index[:4])) if only_index else ""))

    return {"shards": len(shards), "tensors": len(header_keys),
            "index_present": index_keys is not None,
            "bytes": sum(s["size"] for s in shards)}


# ---------------------------------------------------------------------------
# streaming residency
# ---------------------------------------------------------------------------


def _require_transformers_internals():
    """The private loading API this streamer stands on, or a refusal naming it.

    Reusing `transformers`' converter is the whole reason the streamed weights
    are byte-identical to the `from_pretrained` weights -- a MoE checkpoint's
    per-expert matrices are fused by a `WeightConverter`, and re-deriving that
    fusion by hand is precisely the kind of "close enough" this suite exists to
    refuse.  The API is private, so a build that does not offer it gets a
    refusal that names what is missing, not a silent fallback to hand-rolled
    loading.
    """
    missing = []
    try:
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
    except Exception as exc:  # pragma: no cover - depends on the build
        convert_and_load_state_dict_in_model = None
        missing.append("transformers.core_model_loading.convert_and_load_state_dict_in_model (%s)" % exc)
    try:
        from transformers.modeling_utils import (LoadStateDictConfig,
                                                 _load_parameter_into_model,
                                                 patch_output_recorders)
    except Exception as exc:  # pragma: no cover
        LoadStateDictConfig = _load_parameter_into_model = patch_output_recorders = None
        missing.append("transformers.modeling_utils.{LoadStateDictConfig,"
                       "_load_parameter_into_model,patch_output_recorders} (%s)" % exc)
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
    except Exception as exc:  # pragma: no cover
        get_model_conversion_mapping = None
        missing.append("transformers.conversion_mapping.get_model_conversion_mapping (%s)" % exc)
    if missing:
        raise LayerOuterError(
            "REFUSED: --layer-residency stream needs transformers' own weight-conversion "
            "loader so that a streamed layer is BYTE-IDENTICAL to what from_pretrained "
            "would have built (a MoE checkpoint's experts are fused by a WeightConverter; "
            "re-deriving that fusion by hand is exactly the kind of near-enough this suite "
            "refuses). This build does not expose: %s. Use --layer-residency resident, "
            "which reorders the loop over a fully loaded model and needs no private API."
            % "; ".join(missing))
    return (convert_and_load_state_dict_in_model, LoadStateDictConfig,
            _load_parameter_into_model, patch_output_recorders,
            get_model_conversion_mapping)


class StreamedModel(object):
    """A model whose decoder layers are materialised one at a time.

    Everything that is NOT a decoder-layer parameter -- embeddings, the final
    norm, the head, every buffer including the per-layer ones -- is loaded once
    and stays resident: that is the 37.78 GB "non-routed set" of
    `docs/GLM53-ROOT-FEASIBILITY.md` §2, and it is the part a forward pass needs
    at every layer anyway.  Buffers are deliberately never streamed: they are
    rotary tables and router correction biases, kilobytes against gigabytes,
    and streaming them would add a way to get a forward pass wrong for no
    saving at all.
    """

    def __init__(self, model, layers_prefix: str, layers, load_layer_keys,
                 load_call, free_call, report: Dict[str, Any]):
        self.model = model
        self.layers_prefix = layers_prefix
        self.layers = layers
        self._load_layer_keys = load_layer_keys
        self._load_call = load_call
        self._free_call = free_call
        self.report = report
        self.resident_layer: Optional[int] = None

    # -- the two operations the schedule needs --------------------------------

    def load_layer(self, index: int) -> None:
        self._load_call(index)
        self.resident_layer = index

    def free_layer(self, index: int) -> None:
        self._free_call(index)
        if self.resident_layer == index:
            self.resident_layer = None

    def close(self) -> None:
        """Release the safetensors handles once the last layer has been loaded.

        They are held open for the whole layer loop on purpose: the state dict
        is lazy slices over those mmaps, and closing early would break the
        loads that have not happened yet.
        """
        for pointer in getattr(self, "pointers", ()) or ():
            try:
                pointer.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self.pointers = []


class _LayerCounts(object):
    """A live view of "how many checkpoint tensors does layer N have".

    Not a snapshot: under a gate the layer subsets are still being filled in as
    shards land, and a dict comprehension taken at build time would report 0 for
    every layer that had not arrived yet -- in the log line whose whole job is to
    say what was just loaded.
    """

    def __init__(self, layer_subset: Dict[int, Dict[str, Any]]):
        self._subset = layer_subset

    def get(self, index: int, default: Any = None) -> Any:
        subset = self._subset.get(int(index))
        return len(subset) if subset is not None else default

    def __getitem__(self, index: int) -> int:
        return len(self._subset[int(index)])

    def __len__(self) -> int:
        return len(self._subset)

    def items(self):
        return [(index, len(subset)) for index, subset in sorted(self._subset.items())]


def _index_weight_map(model_dir: str) -> Dict[str, str]:
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(path):
        raise LayerOuterError(
            "REFUSED: a gated (race-mode) streamed load needs %s -- it is the only "
            "statement of which tensors the complete checkpoint holds, and without it "
            "a tree that is merely still downloading is indistinguishable from a tree "
            "that is missing tensors." % path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("weight_map") or {}


def _model_device(device: str):
    import torch

    return torch.device(device)


# ---------------------------------------------------------------------------
# FP8 block-scaled checkpoints: decode to bf16 on the host, per tensor
# ---------------------------------------------------------------------------

FP8_DECODE_METHOD = "fp8-block-dequant-to-bf16"
#: The arithmetic this decoder reproduces, and the parity evidence that shows
#: it does so bitwise on real tensors: transformers 5.16.1
#: `integrations.finegrained_fp8.Fp8Dequantize._dequantize_one` -- fp8 -> fp32,
#: one multiply per element by the block's fp32 scale, one cast to the
#: destination dtype. `engines/tools/selftest_fp8_decode_offline.py` asserts
#: equality against that function on synthetic and on real fetched shards.
FP8_DECODE_REFERENCE = "transformers.integrations.finegrained_fp8.Fp8Dequantize._dequantize_one"
FP8_SCALE_SUFFIX = "_scale_inv"


def fp8_checkpoint_plan(config) -> Optional[Dict[str, Any]]:
    """The exact FP8 form this schedule decodes, read from the config, or None.

    Accepts only the FineGrainedFP8 checkpoint form `transformers` itself
    loads with `dequantize=True`: `quant_method: fp8`, `fmt: e4m3`, a 2-D
    `weight_block_size`, dynamic (or unstated) activation scaling. Anything
    else is refused by the caller: a static activation scale is not a
    weights-only artifact, and a packed format has other shapes.
    """
    qc = getattr(config, "quantization_config", None)
    if not qc:
        return None
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(getattr(qc, "__dict__", {}))
    method = qc.get("quant_method")
    fmt = qc.get("fmt")
    block = qc.get("weight_block_size")
    activation = qc.get("activation_scheme")
    ok = (method == "fp8" and fmt == "e4m3"
          and isinstance(block, (list, tuple)) and len(block) == 2
          and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in block)
          and activation in (None, "dynamic"))
    if not ok:
        raise LayerOuterError(
            "REFUSED: quantization_config quant_method=%r fmt=%r weight_block_size=%r "
            "activation_scheme=%r is not the block-scaled FP8 e4m3 weights-only form "
            "this schedule decodes (the form transformers loads with dequantize=True). "
            "Use --schedule window-outer, or author a decoder for this surface."
            % (method, fmt, block, activation))
    return {
        "quant_method": method, "fmt": fmt,
        "weight_block_size": [int(block[0]), int(block[1])],
        "activation_scheme": activation,
        "modules_to_not_convert": sorted(str(m) for m in (qc.get("modules_to_not_convert") or [])),
    }


def dequantize_block_fp8(quantized, scales, output_dtype, block_size=(128, 128)):
    """fp8 e4m3 x fp32 block scale -> output dtype, the reference arithmetic exactly.

    `quantized` is (rows, cols); `scales` is (ceil(rows/bm), ceil(cols/bn)) in
    fp32 -- the DeepSeek-V3 block form `weight_block_size` declares, where the
    LAST block along either axis may be partial (GLM-5.3's kv_a_proj_with_mqa
    is 576 x 6144 under a 128 x 128 block: a 5 x 48 grid). Each element is
    promoted to fp32, multiplied ONCE by its block's scale in fp32, and cast
    once to `output_dtype` (round to nearest even). No accumulation, no fused
    multiply-add, no device-dependent kernel.

    Full blocks: bitwise `transformers` `Fp8Dequantize._dequantize_one`, which
    performs this exact reshape-multiply-cast. Partial blocks: the same
    arithmetic on the zero-padded tensor, cropped -- an element's value never
    depends on its neighbours, so this equals the kernel rule every FP8 server
    (DeepSeek `weight_dequant`, vLLM, DeepGEMM) applies: `s[i // bm, j // bn]`.
    `engines/tools/fp8_parity.py` asserts both on real fetched shards.
    """
    import torch

    if quantized.dtype != torch.float8_e4m3fn:
        raise LayerOuterError(
            "REFUSED: block-scaled FP8 decode was handed a %s tensor" % quantized.dtype)
    if scales.dtype != torch.float32:
        raise LayerOuterError(
            "REFUSED: block-scaled FP8 decode expects fp32 scales, got %s" % scales.dtype)
    if quantized.dim() != 2 or scales.dim() != 2:
        raise LayerOuterError(
            "REFUSED: block-scaled FP8 decode expects 2-D weight and scale, got %s and %s"
            % (tuple(quantized.shape), tuple(scales.shape)))
    block_m, block_n = int(block_size[0]), int(block_size[1])
    rows, cols = quantized.shape
    scale_rows, scale_cols = scales.shape
    if (scale_rows != -(-rows // block_m)) or (scale_cols != -(-cols // block_n)):
        raise LayerOuterError(
            "REFUSED: weight shape (%d, %d) under a (%d, %d) block needs a (%d, %d) "
            "scale grid; the checkpoint carries (%d, %d)"
            % (rows, cols, block_m, block_n, -(-rows // block_m), -(-cols // block_n),
               scale_rows, scale_cols))
    pad_rows = scale_rows * block_m - rows
    pad_cols = scale_cols * block_n - cols
    q = quantized.to(torch.float32)
    if pad_rows or pad_cols:
        q = torch.nn.functional.pad(q, (0, pad_cols, 0, pad_rows))
    q = q.reshape(scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(scale_rows, 1, scale_cols, 1)
    out = (q * s).to(output_dtype).reshape(scale_rows * block_m, scale_cols * block_n)
    if pad_rows or pad_cols:
        out = out[:rows, :cols].contiguous()
    return out


def materialize_fp8_subset(subset: Dict[str, Any], plan: Dict[str, Any], torch_dtype,
                           stats: Dict[str, int]) -> Dict[str, Any]:
    """Replace every (weight, weight_scale_inv) pair in a lazy subset by one decoded tensor.

    Keys keep their order and the weight keeps its name, so the model's own
    conversion mapping sees exactly what it would see for a bf16 checkpoint.
    A scale without its weight, or an fp8 tensor without a scale, is refused:
    the second case is the silent one (the payload would load as bf16 with
    the block scale never applied).
    """
    import torch

    out: Dict[str, Any] = {}
    scale_keys = {key for key in subset if key.endswith(FP8_SCALE_SUFFIX)}
    for key, value in subset.items():
        if key in scale_keys:
            continue
        scale_key = key + FP8_SCALE_SUFFIX
        if scale_key in scale_keys:
            quantized = _eager(value)
            scales = _eager(subset[scale_key])
            out[key] = dequantize_block_fp8(quantized, scales, torch_dtype,
                                            plan["weight_block_size"])
            stats["dequantized"] += 1
            stats["scales_consumed"] += 1
            stats["fp8_bytes"] += int(quantized.numel())
            continue
        dtype = getattr(value, "dtype", None)
        if dtype is None and hasattr(value, "get_dtype"):
            dtype = value.get_dtype()
        if str(dtype) in ("torch.float8_e4m3fn", "F8_E4M3", "float8_e4m3fn"):
            raise LayerOuterError(
                "REFUSED: %s is an fp8 tensor with no %s sibling in the checkpoint; "
                "loading it as bf16 would apply no block scale" % (key, scale_key))
        out[key] = value
    orphans = sorted(key for key in scale_keys
                     if key[:-len(FP8_SCALE_SUFFIX)] not in subset)
    if orphans:
        raise LayerOuterError(
            "REFUSED: %d scale tensor(s) have no weight beside them: %s%s"
            % (len(orphans), ", ".join(orphans[:4]),
               " (+%d more)" % (len(orphans) - 4) if len(orphans) > 4 else ""))
    return out


# ---------------------------------------------------------------------------
# EXL3 trellis checkpoints: decode to bf16 on the host, per module, per layer
# ---------------------------------------------------------------------------

TRELLIS_DECODE_METHOD = "exl3-trellis-decode-to-bf16"
#: Every byte of decode math is `engines/tools/exl3hf_surface.py`'s
#: `decode_payload_hf` -- the exllamav3 v1.4.x `mul1`/`mcg` codebooks
#: transcribed from `exllamav3_ext/quant/codebook.cuh`, whose LUTs and anybits
#: unpack are proven bitwise offline against an independent fp64 route, against
#: `dione_surface`'s copy (K2/K3/K4/K6/K8) and against the campaign reader
#: (`engines/tools/selftest_exl3hf_offline.py`). This module adds NO
#: arithmetic: it groups a checkpoint's payload objects, picks each module's
#: codebook from the object that is actually present, and hands the decoded
#: dense tensor to the same converter a bf16 checkpoint would reach.
TRELLIS_DECODE_REFERENCE = "engines/tools/exl3hf_surface.py::decode_payload_hf"
TRELLIS_PAYLOAD_OBJECTS = ("trellis", "suh", "svh")
TRELLIS_CODEBOOKS = ("mul1", "mcg")
#: TP-SHARDED payloads. davidsyoung's TR3 releases store one projection as
#: `M.rank{r}.{trellis,suh,svh,mcg}`, r in 0..tp-1: the atoms are the
#: tensor-parallel shards their serving stack loads one per GPU, declared in
#: the artifact's own `config.hybrid_tr3_tail` (`tp`, `slicing`, `tensor_schema`)
#: and proven against the BF16 source they were encoded from: decoded per rank
#: and concatenated in ascending rank order along the ONE axis the shapes admit,
#: layer-3/expert-0 of the 3.25bpw release matches zai-org/GLM-5.3-BF16 at
#: cosine 0.9994/0.9992/0.9993 (rel_l2 0.067 = the K4 trellis error measured on
#: Fruit), the reversed order at cosine ~0, and the other axis is a shape
#: mismatch (engines/tools/layer-outer-evidence/glm53-exl3-tp-rank-and-zero-pad-parity.json).
#: The composition is therefore SHAPE-DETERMINED and verified, not read from
#: prose: the axis is the one along which `tp` parts tile the parameter's shape;
#: a declared `slicing` entry must agree when present; missing ranks, a rank
#: count that is not `tp`, or two admissible axes all refuse.
TRELLIS_TP_COMPOSE_METHOD = "exl3-trellis-tp-compose-to-bf16"
TRELLIS_RANK_RE = re.compile(r"^(?P<module>.+)\.rank(?P<rank>\d+)$")
TRELLIS_RANK_SPLIT_RE = re.compile(r"\.rank\d+\.(?:%s)$" % "|".join(
    TRELLIS_PAYLOAD_OBJECTS + TRELLIS_CODEBOOKS))
#: Serving-kernel ZERO PADDING. drowzeys' `kv_a_proj_with_mqa` ships as
#: [640, 6144] where its own config implies [576, 6144]; rows 0..575 are the
#: root's tensor bitwise (after FP8 dequant) and rows 576..639 are exactly zero
#: -- 640 = 5 x 128, an alignment pad for their kernel. Recoverable only under a
#: hard check: every excess row exactly zero, every other dim equal, and the
#: transformation recorded. A non-zero tail is a shape mismatch and refuses.
ZERO_PAD_METHOD = "trailing-zero-rows-truncated"


def _config_dict(value) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(getattr(value, "__dict__", {}))


def trellis_tail_declaration(config) -> Optional[Dict[str, Any]]:
    """`config.hybrid_tr3_tail` when it declares an exl3-trellis artifact, else None.

    davidsyoung's releases carry a leftover ModelOpt/NVFP4 `quantization_config`
    (`quant_method: modelopt`, `num_bits: 4, type: float, group_size: 16`) that
    describes NOTHING in the checkpoint; the exl3 declaration lives in this
    top-level block (`format: exl3-trellis`, `codebook`, `tp`, `tensor_schema`).
    Read from bytes it is the payload keys that decide; this block is what the
    artifact SAYS, and the two must agree.
    """
    tail = getattr(config, "hybrid_tr3_tail", None)
    if tail is None and isinstance(config, dict):
        tail = config.get("hybrid_tr3_tail")
    tail = _config_dict(tail)
    if not tail or tail.get("format") != "exl3-trellis":
        return None
    return tail


def trellis_checkpoint_plan(config, declared_keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The exact EXL3 trellis form this schedule decodes, or None.

    Accepts `quant_method: exl3` whose payload groups are the stock
    exllamav3 object layout `M.{trellis,suh,svh,<codebook>}`, one group per
    quantized module, `<codebook>` in {mul1, mcg} PER MODULE -- drowzeys'
    `keys-GLM-5.3-EXL3` uses `mcg` on layer 3 and `mul1` on layers 4-77, so
    the codebook is read from the object each module actually carries and not
    from `quantization_config.codebook`, which names only one of them.
    """
    qc = _config_dict(getattr(config, "quantization_config", None)
                      if not isinstance(config, dict) else config.get("quantization_config"))
    tail = trellis_tail_declaration(config)
    declared_method = qc.get("quant_method")
    if declared_method != "exl3" and tail is None:
        return None
    groups = trellis_payload_groups(declared_keys)
    if not groups:
        raise LayerOuterError(
            "REFUSED: the config declares an exl3 artifact (quant_method=%r, "
            "hybrid_tr3_tail=%s) but the checkpoint carries no %s payload group; "
            "loading it as-is would read trellis bytes as weights."
            % (declared_method, "present" if tail else "absent",
               "/".join(TRELLIS_PAYLOAD_OBJECTS)))
    codebooks: Dict[str, int] = {}
    for objects in groups.values():
        codebooks[objects["codebook"]] = codebooks.get(objects["codebook"], 0) + 1
    # TP-sharded groups: every rank-stem must belong to a module with EXACTLY
    # tp ranks 0..tp-1, and tp must be declared by the artifact.
    ranked: Dict[str, Set[int]] = {}
    for stem in groups:
        match = TRELLIS_RANK_RE.match(stem)
        if match:
            ranked.setdefault(match.group("module"), set()).add(int(match.group("rank")))
    composition = None
    if ranked:
        tp = (tail or {}).get("tp")
        if isinstance(tp, bool) or not isinstance(tp, int) or tp < 2:
            raise LayerOuterError(
                "REFUSED: %d module(s) store rank-sharded trellis payloads (e.g. %s) "
                "but the config declares no hybrid_tr3_tail.tp >= 2 to compose them "
                "by; this schedule does not guess a shard count."
                % (len(ranked), sorted(ranked)[0]))
        bad = sorted(module for module, ranks in ranked.items() if ranks != set(range(tp)))
        if bad:
            raise LayerOuterError(
                "REFUSED: %d module(s) do not carry exactly ranks 0..%d (e.g. %s: %s)"
                % (len(bad), tp - 1, bad[0], sorted(ranked[bad[0]])))
        composition = {
            "tp": tp,
            "modules": len(ranked),
            "declared_slicing": {str(k): str(v) for k, v in
                                 ((tail or {}).get("slicing") or {}).items()},
            "tensor_schema": (tail or {}).get("tensor_schema"),
            "k_values": [int(k) for k in ((tail or {}).get("k_values") or [])
                         if isinstance(k, int) and not isinstance(k, bool)],
        }
    # MIRRORS `measure_cloud._candidate_decode_plan`'s exl3 branch FIELD FOR
    # FIELD, as the FP8 plan mirrors its own: `qualify_root` compares the
    # capture's recorded quantization_config against the job's candidate block
    # for exact equality, so an extra or renamed key here refuses the whole
    # run after both cold runs and the self-compare have already passed.
    # The observed census (module count, per-module codebook histogram) is
    # NOT part of that contract and rides on the log line and the decode
    # evidence instead.
    if tail is not None:
        # The tail block is the artifact's exl3 declaration; the leftover
        # quantization_config is not. Mirrored by the controller.
        contract = {
            "quant_method": "exl3",
            "codebook": str(tail["codebook"]) if tail.get("codebook") is not None else None,
            "bits": tail.get("bits_avg", tail.get("bits")),
            "head_bits": None,
            "modules_to_not_convert": [],
        }
    else:
        contract = {
            "quant_method": "exl3",
            "codebook": str(qc["codebook"]) if qc.get("codebook") is not None else None,
            "bits": qc.get("bits"),
            "head_bits": qc.get("head_bits"),
            "modules_to_not_convert": sorted(
                str(m) for m in (qc.get("modules_to_not_convert") or [])),
        }
    contract["_observed"] = {
        "quantized_module_count": len(groups),
        "codebook_histogram": dict(sorted(codebooks.items())),
        "quant_method_declared": declared_method,
        "declared_by": "hybrid_tr3_tail" if tail is not None else "quantization_config",
        "composition": composition,
    }
    return contract


def trellis_payload_groups(keys: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Group `<module>.{trellis,suh,svh,<codebook>}` keys by module.

    A group is returned only when all three payload objects AND exactly one
    codebook marker are present; a partial group is a refusal, not a skip,
    because a module whose trellis is loaded without its scales is the silent
    failure this decoder exists to prevent.
    """
    staged: Dict[str, Dict[str, str]] = {}
    for key in keys:
        stem, _, last = key.rpartition(".")
        if not stem:
            continue
        if last in TRELLIS_PAYLOAD_OBJECTS:
            staged.setdefault(stem, {})[last] = key
        elif last in TRELLIS_CODEBOOKS:
            staged.setdefault(stem, {}).setdefault("codebooks", []).append(last)  # type: ignore[union-attr]
    groups: Dict[str, Dict[str, str]] = {}
    partial: List[str] = []
    for module, found in staged.items():
        marks = found.get("codebooks") or []
        missing = [name for name in TRELLIS_PAYLOAD_OBJECTS if name not in found]
        if missing or len(marks) != 1:
            partial.append("%s (missing %s, codebook markers %s)"
                           % (module, missing or "none", sorted(marks) or "none"))
            continue
        groups[module] = {name: found[name] for name in TRELLIS_PAYLOAD_OBJECTS}
        groups[module]["codebook"] = marks[0]
        groups[module]["marker"] = "%s.%s" % (module, marks[0])
    if partial:
        raise LayerOuterError(
            "REFUSED: %d incomplete trellis payload group(s): %s%s"
            % (len(partial), "; ".join(sorted(partial)[:3]),
               " (+%d more)" % (len(partial) - 3) if len(partial) > 3 else ""))
    return groups


def _eager(value):
    """Read a lazy safetensors slice, whatever its rank.

    `PySafeSlice[:]` raises `IndexError: slice() cannot be applied to a 0-dim
    tensor`, and an exl3 codebook marker IS 0-dim -- an I32 scalar equal to the
    codebook's own multiplier. `[...]` reads any rank. Hit at layer 3 of a live
    wrldsuksgo2mars capture, after layers 0-2 (FP8 only, all rank>=1) decoded.
    """
    if hasattr(value, "dtype"):
        return value
    try:
        return value[...]
    except (TypeError, IndexError):
        return value[:]


def materialize_trellis_subset(subset: Dict[str, Any], plan: Dict[str, Any], torch_dtype,
                               stats: Dict[str, int], fp8_plan: Optional[Dict[str, Any]] = None,
                               fp8_stats: Optional[Dict[str, int]] = None,
                               device: str = "cpu",
                               composition: Optional[Dict[str, Any]] = None,
                               expected_shape: Optional[Callable[[str], Optional[Tuple[int, ...]]]] = None
                               ) -> Dict[str, Any]:
    """Replace every trellis payload group in a lazy subset by one decoded `.weight`.

    Decodes on `device` -- the capture device, not the host. The trellis
    decode is matmul-heavy (two 128x128 Hadamard passes plus a 65,536-entry
    LUT gather per module) where the FP8 decode is one elementwise multiply,
    and 768 modules per MoE layer x 75 layers on a CPU extrapolates to ~11 h
    per cold run: measured on a live wrldsuksgo2mars pod, where layer 3 alone
    did not finish loading in 8 minutes while the three FP8 layers before it
    took ~1 s each.

    Composes with the FP8 decoder when the artifact keeps part of itself in
    block-scaled FP8 beside the trellis payloads -- wrldsuksgo2mars'
    `GLM-5.3-EXL3-K4-v1` keeps `shared_experts`/`self_attn` as
    `weight_scale_inv` FP8 and quantizes only the routed experts, so one
    subset carries both surfaces and both hooks must run over it.
    """
    surface = _exl3hf()
    groups = trellis_payload_groups(subset)
    consumed = {key for objects in groups.values()
                for name, key in objects.items() if name != "codebook"}
    passthrough = {key: value for key, value in subset.items() if key not in consumed}
    # The FP8 half counts into ITS OWN counter dict: the two decoders keep
    # separate stats and the log line reads both by name, so handing the
    # trellis dict to the FP8 decoder is a KeyError on the first dequantized
    # tensor (it was, on a live wrldsuksgo2mars pod).
    if fp8_plan is not None:
        counters = fp8_stats if fp8_stats is not None else stats
        for key in ("dequantized", "scales_consumed", "fp8_bytes"):
            counters.setdefault(key, 0)
        out: Dict[str, Any] = materialize_fp8_subset(
            passthrough, fp8_plan, torch_dtype, counters)
    else:
        # No FP8 plan means the index carries no scale tensor at all -- so an
        # fp8 tensor here has NO scale anywhere and would load as bf16 with
        # its block scale never applied: the M1 defect, refused by name in
        # the FP8 path and, until 2026-09-04, silently accepted here.
        for key, value in passthrough.items():
            dtype = getattr(value, "dtype", None)
            if dtype is None and hasattr(value, "get_dtype"):
                dtype = value.get_dtype()
            if str(dtype) in ("torch.float8_e4m3fn", "F8_E4M3", "float8_e4m3fn",
                              "torch.float8_e5m2", "F8_E5M2", "float8_e5m2"):
                raise LayerOuterError(
                    "REFUSED: %s is an fp8 tensor in a checkpoint that carries no "
                    "%s tensor anywhere; loading it as bf16 would apply no block scale"
                    % (key, FP8_SCALE_SUFFIX))
        out = dict(passthrough)
    parts: Dict[str, Dict[int, Any]] = {}
    for module, objects in groups.items():
        payload = {}
        for name in TRELLIS_PAYLOAD_OBJECTS:
            payload[name] = _eager(subset[objects[name]])
        marker = _eager(subset[objects["marker"]])
        expected = surface.CODEBOOK_OBJECTS[objects["codebook"]]
        observed = int(marker.reshape(-1)[0])
        if observed != expected:
            raise LayerOuterError(
                "REFUSED: %s carries a %s marker of %d, not the codebook's own "
                "multiplier %d; the payload was not written by the codebook it names"
                % (module, objects["codebook"], observed, expected))
        decoded = surface.decode_payload_hf(
            payload["trellis"].to(device), payload["suh"].to(device),
            payload["svh"].to(device), codebook=objects["codebook"])
        stats["decoded_modules"] += 1
        bits = int(payload["trellis"].shape[-1]) // 16
        stats["trellis_bits"] += bits
        histogram = stats.setdefault("k_histogram", {})
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
        _check_declared_bits(module, bits, plan, composition)
        ranked = TRELLIS_RANK_RE.match(module)
        weight_key = "%s.weight" % (ranked.group("module") if ranked else module)
        if weight_key in subset:
            raise LayerOuterError(
                "REFUSED: %s exists as a plain weight beside its trellis payload; the "
                "checkpoint carries two versions of one tensor and this schedule will "
                "not pick one" % weight_key)
        if ranked is None:
            out[weight_key] = decoded.to(torch_dtype)
            continue
        if composition is None:
            raise LayerOuterError(
                "REFUSED: %s is a rank-sharded payload but the plan carries no "
                "composition (the config declared no hybrid_tr3_tail.tp)" % module)
        parts.setdefault(ranked.group("module"), {})[int(ranked.group("rank"))] = decoded
    for module, by_rank in sorted(parts.items()):
        out["%s.weight" % module] = _compose_tp_ranks(
            module, by_rank, composition, expected_shape, torch_dtype, stats)
    return out


def _check_declared_bits(module: str, bits: int, plan: Dict[str, Any],
                         composition: Optional[Dict[str, Any]]) -> None:
    """The payload's own K against what the artifact declares.

    A uniform declaration (`bits: 4`) must equal every module's K. A TR3 tail
    declares an AVERAGE (`bits_avg: 3.25`) over per-expert tiers with
    `k_values: [3, 4]`, so the check is membership in k_values. A declaration
    that is neither is not checkable and is recorded, not trusted: the row's
    bit-width label then rests on the K histogram in the decode evidence.
    """
    declared = plan.get("bits")
    k_values = (composition or {}).get("k_values") if composition else None
    if k_values:
        if bits not in {int(k) for k in k_values}:
            raise LayerOuterError(
                "REFUSED: %s is a K%d payload but the artifact declares k_values %s"
                % (module, bits, sorted(k_values)))
        return
    if composition is not None:
        # A TR3 tail declares bits_avg, an AVERAGE over per-expert tiers; with
        # no k_values it is not checkable per module and stays a recorded
        # label backed by the K histogram in the decode evidence.
        return
    if isinstance(declared, bool) or declared is None:
        return
    try:
        declared_value = float(declared)
    except (TypeError, ValueError):
        return
    if declared_value.is_integer() and int(declared_value) != bits:
        raise LayerOuterError(
            "REFUSED: %s is a K%d payload but the artifact declares bits=%r; the row "
            "would be labelled with a bit-width its bytes do not carry"
            % (module, bits, declared))


def _compose_tp_ranks(module: str, by_rank: Dict[int, Any], composition: Dict[str, Any],
                      expected_shape, torch_dtype, stats: Dict[str, int]):
    """Concatenate tp decoded shards along the ONE axis the shapes admit.

    Ascending rank order is the artifact's declared and root-verified order
    (see TRELLIS_TP_COMPOSE_METHOD). The axis is not read from prose: it is
    the axis along which `tp` equal parts tile the parameter's expected shape,
    and exactly one axis may qualify. A declared `slicing` entry for the
    projection must agree when present.
    """
    import torch

    tp = int(composition["tp"])
    if sorted(by_rank) != list(range(tp)):
        raise LayerOuterError(
            "REFUSED: %s carries ranks %s, not 0..%d" % (module, sorted(by_rank), tp - 1))
    shapes = {tuple(t.shape) for t in by_rank.values()}
    if len(shapes) != 1:
        raise LayerOuterError(
            "REFUSED: %s rank shards differ in shape: %s" % (module, sorted(shapes)))
    part = next(iter(shapes))
    if len(part) != 2:
        raise LayerOuterError("REFUSED: %s rank shard is %d-D, not 2-D" % (module, len(part)))
    want = expected_shape("%s.weight" % module) if expected_shape is not None else None
    if want is None or len(want) != 2:
        raise LayerOuterError(
            "REFUSED: %s: no expected 2-D shape is known for the composed weight, so "
            "the concatenation axis cannot be determined" % module)
    axes = [axis for axis in (0, 1)
            if part[axis] * tp == want[axis] and part[1 - axis] == want[1 - axis]]
    if len(axes) != 1:
        raise LayerOuterError(
            "REFUSED: %s: %d shards of %s do not tile %s along exactly one axis "
            "(admissible: %s)" % (module, tp, part, tuple(want), axes))
    axis = axes[0]
    projection = module.rsplit(".", 1)[-1]
    declared = (composition.get("declared_slicing") or {}).get(projection)
    if declared is not None:
        declared_axis = (0 if declared.startswith("N-sliced")
                         else 1 if declared.startswith("K-sliced") else None)
        if declared_axis != axis:
            raise LayerOuterError(
                "REFUSED: %s: the shapes admit concatenation along axis %d but the "
                "artifact declares %r" % (module, axis, declared))
    composed = torch.cat([by_rank[r] for r in range(tp)], dim=axis).to(torch_dtype)
    stats["tp_composed_modules"] = stats.get("tp_composed_modules", 0) + 1
    seen = stats.setdefault("tp_axes", {})
    seen[projection] = axis
    return composed


def truncate_zero_padded_rows(subset: Dict[str, Any], expected_shape, stats: Dict[str, Any],
                              consumed: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Drop all-zero trailing rows a serving kernel padded onto a plain weight.

    Applies only to a plain tensor whose expected parameter shape is known, has
    the same rank, equal dims beyond the first, and FEWER rows than the
    checkpoint's. Every excess row must be exactly zero; one non-zero element
    in the tail refuses by name, because then the tensor is not padded but
    different. Payload objects and scale tensors are never touched. Every
    truncation is counted and named for the decode evidence and the dataset's
    disclosures.
    """
    out: Dict[str, Any] = {}
    for key, value in subset.items():
        if consumed and key in consumed or key.endswith(FP8_SCALE_SUFFIX):
            out[key] = value
            continue
        stem, _, last = key.rpartition(".")
        if last in TRELLIS_PAYLOAD_OBJECTS or last in TRELLIS_CODEBOOKS:
            out[key] = value
            continue
        shape = getattr(value, "shape", None)
        if shape is None and hasattr(value, "get_shape"):
            shape = value.get_shape()
        shape = tuple(int(d) for d in shape) if shape is not None else None
        want = expected_shape(key) if expected_shape is not None else None
        if (shape is None or want is None or len(shape) != len(want) or len(shape) < 1
                or shape[0] <= want[0] or tuple(shape[1:]) != tuple(want[1:])):
            out[key] = value
            continue
        tensor = _eager(value)
        tail = tensor[want[0]:]
        nonzero = int((tail != 0).sum())
        if nonzero:
            raise LayerOuterError(
                "REFUSED: %s is %s where the model expects %s and the %d excess row(s) "
                "carry %d non-zero element(s): not padding, a different tensor"
                % (key, shape, want, shape[0] - want[0], nonzero))
        out[key] = tensor[: want[0]].contiguous()
        record = stats.setdefault("zero_padded_rows_truncated", {"count": 0, "rows": 0, "tensors": []})
        record["count"] += 1
        record["rows"] += shape[0] - want[0]
        if len(record["tensors"]) < 8:
            record["tensors"].append({"name": key, "stored": list(shape), "used": list(want)})
    return out


def _quant_method(config) -> Optional[str]:
    qc = getattr(config, "quantization_config", None)
    if not qc:
        return None
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(getattr(qc, "__dict__", {}))
    method = qc.get("quant_method")
    return str(method) if method is not None else None


def fp8_checkpoint_plan_for_mixed(config) -> Dict[str, Any]:
    """The FP8 half of a mixed trellis+FP8 artifact, defaulted where unstated.

    An EXL3 config declares `quant_method: exl3` and says nothing about the
    tensors the quantizer LEFT in the source's block-scaled FP8, so the block
    geometry cannot come from `quantization_config`. It comes from the source
    release this artifact declares as its base -- GLM-5.3's own 128x128 e4m3 --
    and every decoded tensor is still checked against its own scale grid by
    `dequantize_block_fp8`, which refuses a grid that does not match the
    tensor's shape.
    """
    qc = getattr(config, "quantization_config", None) or {}
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(getattr(qc, "__dict__", {}))
    return {
        "quant_method": "fp8", "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "activation_scheme": None,
        "modules_to_not_convert": sorted(
            str(m) for m in (qc.get("modules_to_not_convert") or [])),
        "block_size_source": "source release (zai-org/GLM-5.3) 128x128 e4m3; "
                             "the exl3 config declares no block geometry",
    }


def _materialized(subset: Dict[str, Any], fp8_plan, trellis_plan, trellis_fp8_plan,
                  torch_dtype, fp8_stats, trellis_stats,
                  device: str = "cpu", expected_shape=None) -> Dict[str, Any]:
    """Whichever decoders this artifact needs, in the one order that is safe."""
    if trellis_plan is not None:
        composition = (trellis_plan.get("_observed") or {}).get("composition")
        if composition is None:
            composition = trellis_stats.get("composition")
        subset = truncate_zero_padded_rows(subset, expected_shape, trellis_stats)
        return materialize_trellis_subset(
            subset, trellis_plan, torch_dtype, trellis_stats,
            fp8_plan=trellis_fp8_plan, fp8_stats=fp8_stats, device=device,
            composition=composition, expected_shape=expected_shape)
    if fp8_plan is not None:
        return materialize_fp8_subset(subset, fp8_plan, torch_dtype, fp8_stats)
    return subset


def _pin_fp32_matmul_policy() -> Dict[str, Any]:
    """TF32 off, highest fp32 matmul precision; returns what was set and seen."""
    import torch

    before = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "pinned": {"allow_tf32_matmul": False, "allow_tf32_cudnn": False,
                   "float32_matmul_precision": "highest"},
        "before_pin": before,
        "note": "the trellis decode's fp32 Hadamard GEMMs run on the capture device; "
                "cuBLAS fp32 without TF32 is deterministic per device but not bitwise "
                "against the CPU fixture proof, which is why per-host bitwise "
                "reproduction is asserted by the two-cold-run qualification, not assumed",
    }


def _exl3hf():
    """Import the decode ABI lazily: torch-heavy, and only a quant run needs it."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import exl3hf_surface

    return exl3hf_surface


def build_streamed_model(model_dir: str, cls, config, dtype_name: str, device: str,
                         log: Callable[..., None],
                         layer_guard: Optional[Callable[[int, Dict[str, Any]], None]] = None,
                         gate: Optional[Any] = None) -> StreamedModel:
    """Instantiate on meta, load everything but the decoder layers, and return the streamer.

    `gate` turns the loader from "the tree is complete" into "the tree arrives
    while I work" -- the overlapped fetch of `engines/tools/race_fetch.py`.  It is any
    object exposing `wait_for_shards(names)`, `wait_for_layer(i)` and `.plan`
    (a `race_fetch.FetchPlan`).  With a gate:

      * only the shards the RESIDENT load will read are opened and audited
        before it -- computed here from the model's own stack prefix and the
        conversion mapping's renames, not taken from the plan's bucket -- and
        the audit is the same audit, restricted to them;
      * layer N's shards are waited on, audited and opened inside `load_layer(N)`
        -- i.e. at the last possible moment, which is the whole point;
      * a buffer belonging to ONE layer rides with that layer rather than with
        the resident set (see the comment on `deferred_buffers`);
      * with `gate=None` the original code path runs untouched.

    Nothing about the ARITHMETIC differs between the two: the same slices go to
    the same converter in the same per-shard, per-header order.  What differs is
    only when the bytes behind those slices arrived -- which
    `bin/selftest_race_mode.py` R6 asserts by digest rather than by argument.
    """
    import copy as _copy

    import torch
    from transformers.utils.generic import ContextManagers

    (convert_and_load, LoadStateDictConfig, load_param_into_model,
     patch_output_recorders, get_conversion_mapping) = _require_transformers_internals()

    from safetensors import safe_open

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype_name]

    if gate is None:
        audit = audit_checkpoint_tree(model_dir)
        log(stage="checkpoint_audit", **audit)
    # With a gate the audit is DEFERRED: which shards the resident load actually
    # reads is not knowable until the module tree exists (it depends on the
    # model's own buffer names and stack prefix), and auditing the whole tree
    # would refuse a checkpoint that is merely still arriving. It happens a few
    # dozen lines below, over exactly the shards the base load will open.

    # QUANTIZED CHECKPOINTS: one form is decoded here, every other is refused,
    # and the reason this is a refusal rather than a comment is that one of
    # the two ways it fails is silent.
    #
    # `from_pretrained` builds an `HfQuantizer`, which (a) replaces the module
    # tree's Linear/Experts with quantized ones and (b) contributes weight
    # conversions -- the `*.scale` -> `*.weight_scale_inv` rename and, on a
    # machine with no FP8 kernel, `Fp8Dequantize`.  `build_streamed_model` does
    # neither: it calls `cls(config)` directly and takes only the MODEL's
    # conversion mapping.  What happens next depends on shapes:
    #
    #   * FP4-packed experts (deepseek-ai/DeepSeek-V4-Flash-0731): the packed
    #     tensor's last dim is half the parameter's, so `transformers` reports
    #     "Reinit due to size mismatch" and raises. Loud, harmless.
    #   * A plain FP8 E4M3 weight: the shape is IDENTICAL to the bf16 parameter
    #     it is loaded into. The fp8 values are cast to bf16, the scale tensor
    #     falls out as `unexpected`, AND THE SCALE IS NEVER APPLIED. That is
    #     numerically the M1 Qwen3.8-27B-FP8 defect -- a confident number for a
    #     projection whose weights are off by a per-block factor.
    #
    # So the block-scaled FP8 e4m3 form is DECODED on the host, per tensor,
    # before the subset reaches the converter (`materialize_fp8_subset`): the
    # weight arrives as bf16 with its scale applied and the scale key never
    # reaches the loader. "Dequantize-and-run, weights-only", the M1 method,
    # under the streaming schedule. Any other quantization_config is refused
    # by `fp8_checkpoint_plan` before anything is instantiated.
    fp8_plan = fp8_checkpoint_plan(config) if _quant_method(config) != "exl3" else None
    fp8_stats = {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0}
    if fp8_plan is not None:
        log(stage="fp8_decode_plan", method=FP8_DECODE_METHOD,
            reference=FP8_DECODE_REFERENCE, **fp8_plan)
    # An EXL3 trellis artifact may ALSO keep part of itself in block-scaled
    # FP8 (wrldsuksgo2mars keeps shared_experts/self_attn that way), so the
    # trellis plan is resolved from the checkpoint's own keys and the FP8 hook
    # runs under it rather than beside it. The index is read ONLY for an exl3
    # artifact: a bf16 or FP8 checkpoint must not acquire a dependency on an
    # index file it may not have (a single-shard tree has none, and under a
    # gate it has not landed yet).
    trellis_plan = None
    trellis_stats = {"decoded_modules": 0, "trellis_bits": 0}
    trellis_fp8_plan = None
    if _quant_method(config) == "exl3" or trellis_tail_declaration(config) is not None:
        # The trellis decode is two fp32 128x128 GEMMs per module and now runs
        # on the capture device: pin the matmul policy exactly as stream_score
        # does and RECORD it, so the sealed receipt can show that TF32 was off
        # rather than assume torch's default (which NVIDIA_TF32_OVERRIDE can
        # flip from the environment without a trace).
        trellis_stats["numeric_policy"] = _pin_fp32_matmul_policy()
        if str(os.environ.get("NVIDIA_TF32_OVERRIDE", "")).strip() == "1":
            raise LayerOuterError(
                "REFUSED: NVIDIA_TF32_OVERRIDE=1 forces TF32 in cuBLAS regardless of the "
                "torch flags; the trellis decode's fp32 GEMMs would not be fp32. Unset it.")
        keys = list(_index_weight_map(model_dir))
        trellis_plan = trellis_checkpoint_plan(config, keys)
        if any(key.endswith(FP8_SCALE_SUFFIX) for key in keys):
            trellis_fp8_plan = fp8_checkpoint_plan_for_mixed(config)
    if trellis_plan is not None:
        observed = trellis_plan.pop("_observed", {})
        trellis_stats["composition"] = observed.get("composition")
        trellis_stats["quant_method_declared"] = observed.get("quant_method_declared")
        trellis_stats["declared_by"] = observed.get("declared_by")
        log(stage="trellis_decode_plan",
            method=(TRELLIS_TP_COMPOSE_METHOD if observed.get("composition")
                    else TRELLIS_DECODE_METHOD),
            reference=TRELLIS_DECODE_REFERENCE,
            mixed_fp8=trellis_fp8_plan is not None, observed=observed,
            **trellis_plan)
        trellis_stats["quantized_module_count"] = observed.get(
            "quantized_module_count", 0)
        trellis_stats["codebook_histogram"] = observed.get("codebook_histogram", {})

    # Build with the SAME context managers `from_pretrained` uses, so the module
    # tree (kernel patches, dtype, tie-weight suppression) is the one the
    # window-outer path would have built -- on meta, so nothing is allocated.
    with ContextManagers(cls.get_init_context(torch_dtype, False, False, None)):
        model = cls(_copy.deepcopy(config))
        patch_output_recorders(model)
    model.eval()

    layers_prefix, layers = find_decoder_layers(model)
    layer_pattern = re.compile(r"^" + re.escape(layers_prefix) + r"\.(\d+)\.")

    # Buffers are never streamed (see the class docstring), so a checkpoint key
    # that targets one must be routed to the resident load.  The checkpoint may
    # address it with or without the base-model prefix, and getting that wrong
    # would strand the buffer on meta and trip the refusal below for no reason,
    # so both spellings are accepted.
    prefix = getattr(model, "base_model_prefix", "") or ""
    buffer_names = set()
    for name, _ in model.named_buffers():
        buffer_names.add(name)
        if prefix:
            buffer_names.add(name.removeprefix(prefix + "."))
            buffer_names.add("%s.%s" % (prefix, name))
    streamed_params = {name for name, _ in model.named_parameters()
                       if layer_pattern.match(name)}
    if not streamed_params:
        raise LayerOuterError("no parameters under %s.<i>. -- nothing to stream"
                              % layers_prefix)

    dtype_plan = model._get_dtype_plan(torch_dtype)
    weight_mapping = get_conversion_mapping(model, None, None)
    load_config = LoadStateDictConfig(
        pretrained_model_name_or_path=model_dir,
        device_map={"": str(_model_device(device))},
        dtype=torch_dtype, dtype_plan=dtype_plan, weight_mapping=weight_mapping)

    # The checkpoint, as lazy safetensors slices: opening every shard costs
    # mmap handles, not bytes.  Materialisation happens per parameter, inside
    # `convert_and_load_state_dict_in_model`.
    #
    # With a gate only the shards that have landed are opened; the rest are
    # opened in `do_load` as the capture reaches the layers that need them.
    # Either way the keys are enumerated from each shard's OWN header, in shard
    # order -- so the subsets handed to the converter carry the same keys in the
    # same order on both paths.
    pointers: List[Any] = []
    opened_shards: Dict[str, Any] = {}

    def _open_shards(names: Sequence[str]) -> Dict[str, Any]:
        """Open shards not yet open; return {key: lazy slice} for the NEW ones only."""
        fresh: Dict[str, Any] = {}
        for name in sorted(names):
            if name in opened_shards:
                continue
            pointer = safe_open(os.path.join(model_dir, name), framework="pt", device="cpu")
            opened_shards[name] = pointer
            pointers.append(pointer)
            for key in pointer.keys():
                fresh[key] = pointer.get_slice(key)
        return fresh

    ungated_shard_names = (sorted(name for name in os.listdir(model_dir)
                                  if name.endswith(".safetensors"))
                           if gate is None else [])

    # ROUTING IS DONE ON THE RENAMED KEY, not the raw one.
    #
    # `layer_pattern` is built from the MODEL's stack path. For GLM-5.3 the
    # checkpoint spells that path the same way and matching the raw key works.
    # For a VL checkpoint it does not: `MiniMaxAI/MiniMax-M3` ships
    # `language_model.model.layers.N.` while the model holds
    # `model.language_model.layers.N.`, so EVERY layer tensor missed the pattern,
    # every one of them fell into the resident load, and the schedule then
    # refused with "the checkpoint holds no tensors for
    # model.language_model.layers.0" -- a true statement about the wrong name.
    #
    # The conversion mapping already knows the answer; `convert_and_load` uses it
    # a few lines below on these same raw keys. Applying only its RENAMES here
    # (converters collapse several sources into one target, which is fine: they
    # all carry the same layer index) puts each key in the right bucket while the
    # subsets still hold the raw names the loader expects.
    #
    # Where a rename cannot be computed the raw key is used, which is exactly the
    # old behaviour -- so an architecture whose names already match is unaffected.
    renames = list(weight_mapping.values() if isinstance(weight_mapping, dict)
                   else (weight_mapping or []))

    def routing_key(key: str) -> str:
        out = key
        for rename in renames:
            renamer = getattr(rename, "rename_source_key", None)
            if renamer is None:
                continue
            try:
                renamed = renamer(out)
            except Exception:
                return key
            out = renamed[0] if isinstance(renamed, tuple) else renamed
            if not isinstance(out, str):
                return key
        return out

    _expert_key = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")

    def expected_shape(key: str) -> Optional[Tuple[int, ...]]:
        """The parameter shape the converter will load `key` into, or None.

        A plain parameter answers from the meta-device module tree. A routed
        expert projection has no per-expert parameter (the model fuses them),
        so its per-expert shape comes from the config's own geometry:
        gate/up [moe_intermediate_size, hidden_size], down the transpose.
        """
        match = _expert_key.search(key)
        if match:
            text = getattr(config, "text_config", None) or config
            hidden = getattr(text, "hidden_size", None)
            inter = getattr(text, "moe_intermediate_size", None)
            if not isinstance(hidden, int) or not isinstance(inter, int):
                return None
            return (inter, hidden) if match.group(1) != "down_proj" else (hidden, inter)
        try:
            return tuple(int(d) for d in model.get_parameter(routing_key(key)).shape)
        except (AttributeError, KeyError, ValueError):
            return None

    base_subset: Dict[str, Any] = {}
    layer_subset: Dict[int, Dict[str, Any]] = {}
    routing_counts = {"routed": 0, "seen": 0}

    # A BUFFER THAT BELONGS TO ONE LAYER IS NOT PART OF THE RESIDENT SET, and
    # under a gate it must not be treated as one.
    #
    # Found by running against the live `malaiwah/GLM-5.2-SIQ-Fruit-bf16`, not by
    # reading the code: `model.layers.N.mlp.gate.e_score_correction_bias` is a
    # router-correction BUFFER, so the ungated loader routes it to the resident
    # load -- but it is four kilobytes living inside that layer's 845 MB shard.
    # Blocking the resident load on it means blocking on ten of the fourteen
    # shards, which serializes almost the whole fetch and deletes the overlap.
    # And it is not a quiet failure either: `transformers` reported all ten
    # MISSING, randomly initialised them, and CAPTURE-03 refused the capture.
    #
    # So under a gate these ride WITH their layer, which is the loop order's own
    # logic -- layer N's everything loads when layer N loads, and the value is
    # read only by layer N's forward. The set is derived from the MODULE TREE
    # rather than from whichever shards happen to be open, because the resident
    # load has to know which buffers it is not responsible for BEFORE it runs.
    #
    # The ungated path is untouched, deliberately: its bit-identity against
    # `from_pretrained` is already proven, and two paths loading the same bytes
    # at different moments is a scheduling difference, not an arithmetic one.
    # `bin/selftest_race_mode.py` R6 asserts the two produce the same
    # capture_content_digest rather than arguing that they must.
    deferred_buffers: Set[str] = (
        {name for name, _ in model.named_buffers() if layer_pattern.match(name)}
        if gate is not None else set())

    def bucket(slices: Dict[str, Any]) -> None:
        """Route freshly opened keys into the resident subset or a layer's subset.

        Called once for the whole tree without a gate, and once per landed
        shard batch with one.
        """
        for key, value in slices.items():
            routing_counts["seen"] += 1
            target = routing_key(key)
            if target != key:
                routing_counts["routed"] += 1
            match = layer_pattern.match(target)
            is_buffer = key in buffer_names or target in buffer_names
            if match is None or (is_buffer and gate is None):
                base_subset[key] = value
            else:
                layer_subset.setdefault(int(match.group(1)), {})[key] = value

    if gate is None:
        bucket(_open_shards(ungated_shard_names))
        resident_shards: List[str] = []
    else:
        # THE RESIDENT SET, computed rather than guessed. `gate.plan` decides the
        # fetch ORDER; this decides what the base load actually blocks on, using
        # the model's own stack prefix and the conversion mapping's renames. The
        # two agree on every checkpoint whose keys the plan's regex matched, and
        # where they do not, this one is right -- so the wait is for exactly
        # these shards, not for the plan's bucket.
        weight_map = _index_weight_map(model_dir)
        resident_keys = [key for key in weight_map
                         if layer_pattern.match(routing_key(key)) is None]
        resident_shards = sorted({weight_map[key] for key in resident_keys})
        if not resident_shards:
            raise LayerOuterError(
                "REFUSED: every tensor in the checkpoint index routes to a decoder "
                "layer, so there is nothing to load resident -- no embeddings, no "
                "final norm, no head. Either the index is partial or %s is not this "
                "model's stack prefix." % layers_prefix)
        waited = gate.wait_for_shards(resident_shards)
        audit = audit_checkpoint_tree(model_dir, shards=resident_shards)
        log(stage="checkpoint_audit", partial=True,
            audited_shards=len(resident_shards), waited_seconds=round(waited, 3),
            **audit)
        bucket(_open_shards(resident_shards))
    if routing_counts["routed"]:
        log(stage="stream_routing", renamed_checkpoint_keys=routing_counts["routed"],
            total_checkpoint_keys=routing_counts["seen"], layers_prefix=layers_prefix)
    if fp8_plan is not None:
        # A config that declares FP8 over a checkpoint with no scale tensors is
        # lying about itself; decoding nothing and capturing as native would be
        # a confident number for an artifact nobody described.
        declared_keys = (list(base_subset) + [key for subset in layer_subset.values()
                                              for key in subset]
                         if gate is None else list(_index_weight_map(model_dir)))
        if not any(key.endswith(FP8_SCALE_SUFFIX) for key in declared_keys):
            raise LayerOuterError(
                "REFUSED: quantization_config declares block-scaled FP8 but the "
                "checkpoint carries no *%s tensor; the payload cannot be decoded and "
                "loading it as-is would apply no block scale. Use --schedule "
                "window-outer with a quantizer that understands this artifact, or "
                "fix the artifact's config." % FP8_SCALE_SUFFIX)

    aggregate = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": [],
                 "error_msgs": [], "conversion_errors": {}}

    # Checkpoint tensors addressed to a layer index the model does not build are
    # never handed to the loader by this schedule, so they would never appear in
    # any per-load `unexpected_keys` and the `checkpoint_tensors_not_loaded`
    # disclosure the window-outer path emits would silently go missing.  This is
    # not hypothetical: GLM-5.3's MTP layer 78 (791 tensors, 18.5 GiB) and
    # Fruit's layer 13 are exactly this case -- `transformers` builds
    # `num_hidden_layers` layers and drops the next-token-prediction layer.
    #
    # With a gate the shards holding those tensors may not have landed yet, so
    # the answer comes from the checkpoint INDEX -- which names every key in the
    # tree without reading a byte of it -- rather than from the shards opened so
    # far. Getting this from the index rather than from "what is on disk right
    # now" is what stops race mode from quietly dropping a disclosure that the
    # fetch-then-capture path would have made.
    if gate is None:
        over_index_keys = {index: set(subset) for index, subset in layer_subset.items()}
    else:
        over_index_keys = {}
        for key in _index_weight_map(model_dir):
            target = routing_key(key)
            match = layer_pattern.match(target)
            # Same buffer test `bucket` uses, so the two cannot disagree about
            # what counts as a layer tensor.
            if match is not None and key not in buffer_names \
                    and target not in buffer_names:
                over_index_keys.setdefault(int(match.group(1)), set()).add(key)
    for index in sorted(over_index_keys):
        if index >= len(layers):
            aggregate["unexpected_keys"] |= over_index_keys[index]

    def _absorb(info) -> None:
        aggregate["unexpected_keys"] |= set(info.unexpected_keys or set())
        for entry in (info.mismatched_keys or []):
            aggregate["mismatched_keys"].append(entry)
        aggregate["error_msgs"].extend(list(info.error_msgs or []))
        aggregate["conversion_errors"].update(dict(info.conversion_errors or {}))

    base_info, _ = convert_and_load(
        model,
        _materialized(base_subset, fp8_plan, trellis_plan, trellis_fp8_plan,
                      torch_dtype, fp8_stats, trellis_stats, device=device,
                      expected_shape=expected_shape),
        load_config)
    _absorb(base_info)

    # Finalisation would otherwise materialise AND randomly initialise every
    # decoder-layer parameter -- exactly the allocation this schedule exists to
    # avoid.  Marking them initialised and dropping them from `missing_keys`
    # confines finalisation to what it is actually needed for here: moving
    # non-persistent buffers off meta (rotary tables), initialising genuinely
    # absent non-layer keys, and tying weights.
    for name in streamed_params:
        model.get_parameter(name)._is_hf_initialized = True
    base_info.missing_keys = {key for key in base_info.missing_keys
                              if key not in streamed_params
                              and key not in deferred_buffers}
    cls._finalize_model_loading(model, load_config, base_info)
    aggregate["missing_keys"] |= set(base_info.missing_keys or set())

    stranded = [name for name, tensor in
                list(model.named_parameters()) + list(model.named_buffers())
                if tensor.device.type == "meta" and name not in streamed_params
                and name not in deferred_buffers]
    if stranded:
        raise LayerOuterError(
            "REFUSED: %d parameter(s)/buffer(s) outside the streamed decoder layers are "
            "still on the meta device after the resident load, so a forward pass would "
            "read them as undefined: %s%s"
            % (len(stranded), ", ".join(stranded[:6]),
               " (+%d more)" % (len(stranded) - 6) if len(stranded) > 6 else ""))

    log(stage="stream_base", resident_tensors=len(base_subset),
        streamed_layers=len(layers), streamed_params=len(streamed_params),
        deferred_buffers=len(deferred_buffers),
        resident_shards=len(resident_shards) or None,
        layers_prefix=layers_prefix)

    def layer_param_names(index: int) -> List[str]:
        head = "%s.%d." % (layers_prefix, index)
        return sorted(name for name in streamed_params if name.startswith(head))

    audited_shards: Set[str] = set(resident_shards)

    def do_load(index: int) -> None:
        if gate is not None:
            # THE BLOCK. Everything above this line ran while layer `index`'s
            # bytes were still on the wire; this is where the capture stops and
            # waits, and `race_fetch.ShardGate` records for how long. The audit
            # runs on the shards that just landed, before a single tensor is
            # read out of them -- a shard that arrived short would otherwise
            # read as zeros, silently.
            waited = gate.wait_for_layer(index)
            wanted = gate.plan.shards_for_layer(index)
            fresh = sorted(wanted - audited_shards)
            if fresh:
                audit_checkpoint_tree(model_dir, shards=fresh)
                audited_shards.update(fresh)
                bucket(_open_shards(fresh))
            if waited > 0.0 or fresh:
                log(stage="race_layer_ready", index=index,
                    waited_seconds=round(waited, 3), audited_shards=len(fresh))
        subset = layer_subset.get(index)
        if not subset:
            raise LayerOuterError(
                "REFUSED: the checkpoint holds no tensors for %s.%d. A layer with no "
                "weights does not fail to run -- it runs on whatever the meta-device "
                "placeholder is replaced by, which is nothing anybody measured."
                % (layers_prefix, index))
        if fp8_plan is not None or trellis_plan is not None:
            # Decoded per layer into a transient dict: the streamer keeps the
            # lazy slices, never the 19 GB of decoded bf16, across layers.
            before = dict(fp8_stats)
            before_trellis = dict(trellis_stats)
            started = time.monotonic()
            decoded = _materialized(subset, fp8_plan, trellis_plan, trellis_fp8_plan,
                                    torch_dtype, fp8_stats, trellis_stats,
                                    device=device, expected_shape=expected_shape)
            decode_seconds = time.monotonic() - started
            info, _ = convert_and_load(model, decoded, load_config)
            del decoded
            log(stage=("trellis_decode_layer" if trellis_plan is not None
                       else "fp8_decode_layer"), index=index,
                dequantized=fp8_stats["dequantized"] - before["dequantized"],
                fp8_elements=fp8_stats["fp8_bytes"] - before["fp8_bytes"],
                decoded_modules=(trellis_stats["decoded_modules"]
                                 - before_trellis["decoded_modules"]) or None,
                decode_seconds=round(decode_seconds, 3))
        else:
            info, _ = convert_and_load(model, subset, load_config)
        _absorb(info)
        names = layer_param_names(index)
        head = "%s.%d." % (layers_prefix, index)
        # CAPTURE-03 is a per-LOAD guard, and this schedule performs one load
        # per layer.  Running it only on the resident set would leave every
        # streamed layer -- i.e. 97.5% of GLM-5.3 by bytes -- unchecked, which
        # is the exact blind spot Stage A closed for the window-outer path.
        if layer_guard is not None:
            layer_guard(index, {
                "_load_report_observed": True,
                "_load_report_has_conversion_errors": True,
                "missing_keys": [],
                "unexpected_keys": sorted(info.unexpected_keys or set()),
                "mismatched_keys": [entry for entry in (info.mismatched_keys or [])
                                    if str(entry[0] if isinstance(entry, (tuple, list))
                                           else entry).startswith(head)],
                "error_msgs": list(info.error_msgs or []),
                "conversion_errors": dict(info.conversion_errors or {}),
            })
        # The guard that closes the hole Stage A found: it is not enough that
        # the load raised nothing, every parameter of this layer must actually
        # have left the meta device.  A key the shard header named but did not
        # deliver lands here, before any window is pushed through it.  This one
        # is NOT overridable: a meta parameter has no contents to disclose.
        stuck = [name for name in names if model.get_parameter(name).device.type == "meta"]
        # A buffer deferred to this layer -- a router correction bias -- must
        # actually have been DELIVERED by this load. Its meta-ness cannot answer
        # that: model finalisation materialises non-persistent buffers, so a
        # buffer nobody supplied is off meta and holding whatever finalisation
        # put there. What the resident path checked by reporting it missing, this
        # path checks by name, here, before a window is pushed through the layer.
        expected = {name for name in deferred_buffers if name.startswith(head)}
        if expected:
            delivered = {routing_key(key) for key in subset}
            undelivered = sorted(expected - delivered)
            if undelivered:
                aggregate["missing_keys"] |= set(undelivered)
                raise LayerOuterError(
                    "REFUSED: layer %d loaded but the checkpoint delivered none of "
                    "%d buffer(s) this layer's forward reads: %s. Under the gated "
                    "(race-mode) loader those ride with their layer instead of with "
                    "the resident set, so an absent one would otherwise be silently "
                    "replaced by whatever model finalisation initialised."
                    % (index, len(undelivered), ", ".join(undelivered[:6])))
        if stuck:
            aggregate["missing_keys"] |= set(stuck)
            raise LayerOuterError(
                "REFUSED: layer %d loaded but %d of its %d parameters are still on the "
                "meta device: %s%s. The checkpoint named them and did not deliver them."
                % (index, len(stuck), len(names), ", ".join(stuck[:6]),
                   " (+%d more)" % (len(stuck) - 6) if len(stuck) > 6 else ""))

    def do_free(index: int) -> None:
        for name in layer_param_names(index):
            param = model.get_parameter(name)
            if param.device.type == "meta":
                continue
            load_param_into_model(model, name, torch.empty_like(param, device="meta"))
        # Freeing must actually free: drop the converter's leftovers and hand
        # the allocator back its blocks, then say so in the log so the claim is
        # a measurement rather than a hope.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    streamer = StreamedModel(model, layers_prefix, layers, layer_param_names,
                             do_load, do_free, aggregate)
    streamer.config = config
    # `pointers` and `layer_subset` are the SAME objects the loader keeps
    # appending to, so a gated build's later shards are closed at `close()` and
    # counted here too. Snapshotting them would have made race mode leak mmap
    # handles and under-report every layer that had not landed yet.
    streamer.pointers = pointers
    streamer.layer_counts = _LayerCounts(layer_subset)
    streamer.fp8_plan = fp8_plan
    streamer.fp8_stats = fp8_stats
    streamer.trellis_plan = trellis_plan
    streamer.trellis_stats = trellis_stats
    streamer.trellis_fp8_plan = trellis_fp8_plan
    return streamer


def weights_decode_evidence(streamer: StreamedModel) -> Optional[Dict[str, Any]]:
    """What the streamer did to the checkpoint's bytes before the forward, for
    the runtime receipt: None for a native checkpoint, else the FP8 plan and
    the counts of tensors decoded and scale tensors consumed."""
    trellis_plan = getattr(streamer, "trellis_plan", None)
    if trellis_plan is not None:
        # The candidate identity the qualification binds: the capture must
        # RECORD the decode it applied, or `qualify_root` refuses with
        # `weights_decode=None` against a job that declares one -- which it
        # did, after both cold runs and the self-compare had already passed.
        stats = dict(getattr(streamer, "trellis_stats", {}))
        mixed = getattr(streamer, "trellis_fp8_plan", None)
        fp8_stats = dict(getattr(streamer, "fp8_stats", {}))
        composition = stats.get("composition")
        evidence = {
            "method": (TRELLIS_TP_COMPOSE_METHOD if composition else TRELLIS_DECODE_METHOD),
            "reference": TRELLIS_DECODE_REFERENCE,
            "output_dtype": "bfloat16",
            "quantization_config": trellis_plan,
            "modules_decoded": int(stats.get("decoded_modules", 0)),
            "trellis_bits_seen": int(stats.get("trellis_bits", 0)),
            "k_histogram": dict(sorted((stats.get("k_histogram") or {}).items())),
            "numeric_policy": dict(stats.get("numeric_policy") or {}),
            "observed": {
                "quantized_module_count": int(stats.get("quantized_module_count", 0)),
                "codebook_histogram": dict(stats.get("codebook_histogram", {})),
                "quant_method_declared": stats.get("quant_method_declared"),
                "declared_by": stats.get("declared_by"),
            },
        }
        if composition:
            evidence["tp_rank_composition"] = {
                "tp": composition.get("tp"),
                "modules_declared": composition.get("modules"),
                "modules_composed": int(stats.get("tp_composed_modules", 0)),
                "axes": dict(stats.get("tp_axes", {})),
                "rank_order": "ascending",
                "declared_slicing": composition.get("declared_slicing"),
                "evidence": "engines/tools/layer-outer-evidence/"
                            "glm53-exl3-tp-rank-and-zero-pad-parity.json",
            }
        if stats.get("zero_padded_rows_truncated"):
            evidence["zero_padded_rows_truncated"] = dict(
                stats["zero_padded_rows_truncated"], method=ZERO_PAD_METHOD)
        if mixed is not None:
            evidence["mixed_fp8"] = {
                "quantization_config": mixed,
                "tensors_dequantized": int(fp8_stats.get("dequantized", 0)),
                "scale_tensors_consumed": int(fp8_stats.get("scales_consumed", 0)),
                "fp8_elements": int(fp8_stats.get("fp8_bytes", 0)),
            }
        return evidence
    plan = getattr(streamer, "fp8_plan", None)
    if plan is None:
        return None
    stats = dict(getattr(streamer, "fp8_stats", {}))
    return {
        "method": FP8_DECODE_METHOD,
        "reference": FP8_DECODE_REFERENCE,
        "output_dtype": "bfloat16",
        "quantization_config": plan,
        "tensors_dequantized": int(stats.get("dequantized", 0)),
        "scale_tensors_consumed": int(stats.get("scales_consumed", 0)),
        "fp8_elements": int(stats.get("fp8_bytes", 0)),
    }


def streamed_loading_info(streamer: StreamedModel) -> Dict[str, Any]:
    """The aggregate load report, in the shape `hf_capture.load_report` reads.

    The streaming loader calls `convert_and_load_state_dict_in_model` once per
    layer plus once for the resident set, so there is no single
    `LoadStateDictInfo` to hand to CAPTURE-03's guards.  This unions them, and
    keeps the two flags those guards refuse on -- `observed` and
    `conversion_errors_visible` -- true only because this path went through the
    library's own converter and read its report object directly, which is a
    stronger position than the wrapped `to_dict()` the window-outer path needs.
    """
    report = streamer.report
    return {
        "_load_report_observed": True,
        "_load_report_has_conversion_errors": True,
        "missing_keys": sorted(report["missing_keys"]),
        "unexpected_keys": sorted(report["unexpected_keys"]),
        "mismatched_keys": list(report["mismatched_keys"]),
        "error_msgs": list(report["error_msgs"]),
        "conversion_errors": dict(report["conversion_errors"]),
    }


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------


class LayerOuterSchedule(object):
    """Proxy the decoder layers so the model's own forward can be run one layer at a time.

    `install()` must be paired with `remove()`; the proxies are instance
    attributes on the layer modules, so `nn.Module.__call__` and every hook it
    runs are untouched -- only the body of the layer is redirected.
    """

    def __init__(self, layers):
        self.layers = layers
        self.count = len(layers)
        self._original: List[Any] = []
        self.active: Optional[int] = None
        self.replay: Any = None
        self.captured: Any = None
        self.calls = 0
        self._installed = False

    def install(self) -> "LayerOuterSchedule":
        if self._installed:
            raise LayerOuterError("the layer proxies are already installed")
        self._original = [layer.forward for layer in self.layers]
        for index, layer in enumerate(self.layers):
            layer.forward = self._proxy(index, self._original[index])
        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            return
        for layer, original in zip(self.layers, self._original):
            try:
                del layer.forward
            except AttributeError:  # pragma: no cover - defensive
                layer.forward = original
        self._original = []
        self._installed = False

    def __enter__(self) -> "LayerOuterSchedule":
        return self.install()

    def __exit__(self, *exc) -> bool:
        self.remove()
        return False

    def _proxy(self, index: int, original):
        def forward(*args, **kwargs):
            active = self.active
            if active is None:  # pragma: no cover - defensive
                raise LayerOuterError("a layer proxy fired with no active layer set")
            if index < active:
                # Whatever the layer below returned last time round, verbatim:
                # a bare tensor, a 2-tuple carrying `topk_indices`, anything.
                # The model's own loop unpacks and re-threads it.
                return self.replay
            if index == active:
                self.calls += 1
                out = original(*args, **kwargs)
                self.captured = out
                return out
            raise _Suspend()
        return forward


def run_panel(model, layers, forward_once: Callable[[int], None], window_count: int,
              log: Callable[..., None],
              on_layer_start: Optional[Callable[[int], None]] = None,
              on_layer_end: Optional[Callable[[int], None]] = None,
              collect: Optional[Callable[[int], Any]] = None) -> List[Any]:
    """for each layer { load it once; for each window: push that window through it; free it }.

    `forward_once(window_index)` runs ONE `model(...)` call for that window --
    it is `hf_capture`'s own call, built from `hf_capture`'s own tensors, so
    that the inputs cannot drift from the window-outer path.

    Windows are pushed through a layer ONE AT A TIME.  They are never stacked
    into a batch: a batched matmul reduces in a different order and would move
    the numbers this engine exists to preserve.

    There is no separate epilogue pass.  On the LAST layer no proxy is left
    above to suspend the forward, so the model runs straight on into its own
    final norm and head -- which is exactly the epilogue, executed by the
    model's own code with the head pre-hook firing as it does on the
    window-outer path.  `collect(window_index)` is called there, once per
    window, and returns whatever the caller wants kept.
    """
    schedule = LayerOuterSchedule(layers)
    memo: List[Any] = [None] * window_count
    results: List[Any] = [None] * window_count
    last = schedule.count - 1
    with schedule:
        for layer_index in range(schedule.count):
            if on_layer_start is not None:
                on_layer_start(layer_index)
            for window_index in range(window_count):
                schedule.active = layer_index
                schedule.replay = memo[window_index]
                schedule.captured = None
                schedule.calls = 0
                try:
                    forward_once(window_index)
                except _Suspend:
                    pass
                if schedule.calls != 1:
                    raise LayerOuterError(
                        "layer %d ran %d time(s) for window %d, expected exactly 1. The "
                        "layer-outer schedule assumes the decoder stack is executed as a "
                        "plain in-order loop, each layer called once per forward; this "
                        "model does something else and must not be captured this way."
                        % (layer_index, schedule.calls, window_index))
                memo[window_index] = schedule.captured
                schedule.captured = None
                if layer_index == last and collect is not None:
                    results[window_index] = collect(window_index)
                    memo[window_index] = None
            schedule.replay = None
            if on_layer_end is not None:
                on_layer_end(layer_index)
            log(stage="layer", index=layer_index, windows=window_count)
    return results


# ---------------------------------------------------------------------------
# measured, not predicted
# ---------------------------------------------------------------------------


def resident_parameter_bytes(model) -> Dict[str, int]:
    """Exactly how many bytes of weights are live right now, by arithmetic.

    Why this exists alongside RSS: on the CPU path `safetensors` mmaps the
    shards, so every byte the loader reads becomes file-backed resident memory
    that the OS is free to evict but `ru_maxrss` counts anyway.  A layer-outer
    run therefore shows an RSS close to the checkpoint size even though it never
    holds more than one layer of anonymous weights -- the RSS is real, but what
    it is measuring there is the page cache, not the schedule.  This figure
    measures the
    schedule: it counts only tensors that are actually materialised, so it is
    the number a "does GLM-5.3 fit in 141 GB" projection has to be built on.
    `torch.cuda.max_memory_allocated` is the same idea for VRAM and has no page
    cache to confuse it, which is why the CUDA numbers are the load-bearing
    ones.
    """
    parameters = 0
    buffers = 0
    for _, tensor in model.named_parameters():
        if tensor.device.type != "meta":
            parameters += tensor.numel() * tensor.element_size()
    for _, tensor in model.named_buffers():
        if tensor.device.type != "meta":
            buffers += tensor.numel() * tensor.element_size()
    return {"parameters": parameters, "buffers": buffers,
            "total": parameters + buffers}


class ResidentWeightPeak(object):
    """A high-water mark over `resident_parameter_bytes`, sampled at layer boundaries."""

    def __init__(self, model):
        self.model = model
        self.peak = 0
        self.detail: Dict[str, int] = {}

    def sample(self) -> int:
        current = resident_parameter_bytes(self.model)
        if current["total"] > self.peak:
            self.peak = current["total"]
            self.detail = current
        return current["total"]


def reset_peak_memory(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: str) -> Dict[str, Any]:
    """What this process actually used, per the OS and the allocator.

    `ru_maxrss` is a high-water mark for the whole process, which is the number
    a rental decision is made on.  It is in BYTES on Darwin and KILOBYTES on
    Linux -- a units bug here would silently misreport by 1024x, so the platform
    is recorded next to the figure.
    """
    import platform
    import resource

    import torch

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system = platform.system()
    rss = int(raw) if system == "Darwin" else int(raw) * 1024
    out: Dict[str, Any] = {"peak_rss_bytes": rss, "peak_rss_gb": round(rss / 1e9, 3),
                           "rss_units_source": "%s ru_maxrss" % system}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        out["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        out["peak_cuda_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        out["peak_cuda_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    elif str(device).startswith("mps") and getattr(torch, "mps", None) is not None:
        # MPS has no peak tracker; on unified memory the RSS figure already
        # covers the weights, so say that rather than emitting a bogus zero.
        out["mps_note"] = ("torch.mps exposes no peak-allocation counter; on unified "
                           "memory peak_rss_bytes already includes the weights")
    return out
