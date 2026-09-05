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
import hashlib
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
# FP8 block-scaled checkpoints: decode to bf16 on the capture device, per
# tensor, under a host-parity gate
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


def _fp8_device_decode(quantized, scales, torch_dtype, block_size, device):
    """The production FP8 decode: fp8 bytes + fp32 scales to `device`, then the
    unchanged reference arithmetic there. Separated so a selftest can stand a
    perturbing stub in its place and watch the parity gate refuse."""
    return dequantize_block_fp8(quantized.to(device, non_blocking=True),
                                scales.to(device), torch_dtype, block_size)


def _fp8_has_partial_block(quantized, block_size) -> bool:
    rows, cols = quantized.shape
    return bool(rows % int(block_size[0]) or cols % int(block_size[1]))


def materialize_fp8_subset(subset: Dict[str, Any], plan: Dict[str, Any], torch_dtype,
                           stats: Dict[str, int], device: str = "cpu",
                           parity_all: bool = False,
                           sink: Optional[Callable[[str, Any], bool]] = None,
                           device_decode: Optional[Callable[..., Any]] = None
                           ) -> Dict[str, Any]:
    """Replace every (weight, weight_scale_inv) pair in a lazy subset by one decoded tensor.

    Keys keep their order and the weight keeps its name, so the model's own
    conversion mapping sees exactly what it would see for a bf16 checkpoint.
    A scale without its weight, or an fp8 tensor without a scale, is refused:
    the second case is the silent one (the payload would load as bf16 with
    the block scale never applied).

    DECODES ON `device`. The arithmetic is one fp8->fp32 promotion (exact),
    one fp32 multiply and one round-to-nearest-even cast per element, none
    of which is a reduction, so CPU and CUDA are expected to agree bitwise --
    and "expected" is not the standard here. Every run RE-DECODES ON THE HOST
    and asserts `torch.equal` for (a) every tensor of the first decoded layer
    (`parity_all`) and (b) every tensor whose shape leaves a partial block
    under the plan's block size (GLM-5.3's 576-row kv_a_proj_with_mqa), on
    every layer. A mismatch REFUSES the run by tensor, dtype, device and
    max_abs_diff; it never falls back to the host result. On the pod this is
    one layer's worth of the old host arithmetic per run instead of 75. The
    counts land in `stats` and the receipt's weights_decode block says so.

    `sink(key, tensor) -> bool` receives each decoded tensor as soon as it
    exists; when it returns True the tensor is NOT held in the returned dict
    (the direct expert fill copies it into its fused slice and drops it, so a
    layer's 19 GB of decoded experts never accumulates anywhere).
    """
    import torch

    decode = device_decode if device_decode is not None else _fp8_device_decode
    on_host = str(device) == "cpu"
    block = plan["weight_block_size"]
    out: Dict[str, Any] = {}
    scale_keys = {key for key in subset if key.endswith(FP8_SCALE_SUFFIX)}
    for key, value in subset.items():
        if key in scale_keys:
            continue
        scale_key = key + FP8_SCALE_SUFFIX
        if scale_key in scale_keys:
            quantized = _eager(value)
            scales = _eager(subset[scale_key])
            if on_host:
                decoded = dequantize_block_fp8(quantized, scales, torch_dtype, block)
            else:
                decoded = decode(quantized, scales, torch_dtype, block, device)
                partial = _fp8_has_partial_block(quantized, block)
                if parity_all or partial:
                    host = dequantize_block_fp8(quantized, scales, torch_dtype, block)
                    mirrored = decoded.to("cpu")
                    if mirrored.shape != host.shape or mirrored.dtype != host.dtype \
                            or not torch.equal(host, mirrored):
                        diff = ((mirrored.to(torch.float32) - host.to(torch.float32))
                                .abs().max().item()
                                if mirrored.shape == host.shape else float("inf"))
                        raise LayerOuterError(
                            "REFUSED: block-scaled FP8 decode of %s (%s, block %s) on %s "
                            "is not bitwise the host decode: max_abs_diff=%r, output "
                            "dtype %s. The receipt would claim %s and the bytes would "
                            "be something else; decode on the host with --device cpu "
                            "or fix the device arithmetic."
                            % (key, tuple(quantized.shape), tuple(block), device, diff,
                               torch_dtype, FP8_DECODE_REFERENCE))
                    stats["device_parity_checked"] = stats.get("device_parity_checked", 0) + 1
                    if partial:
                        stats["device_parity_partial_block_checked"] = (
                            stats.get("device_parity_partial_block_checked", 0) + 1)
                    del host, mirrored
            stats["dequantized"] += 1
            stats["scales_consumed"] += 1
            stats["fp8_bytes"] += int(quantized.numel())
            stats["decode_device"] = "cpu" if on_host else str(device)
            if sink is not None and sink(key, decoded):
                del decoded
                continue
            out[key] = decoded
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
#: Where each rotation layout's resolution rule is grounded -- the reader
#: that serves the artifact, cited file:line (see `exl3_rotation_groups`
#: below and bin/fidelity/hfmeta.py) -- and the real-tensor parity that
#: proves the decode against the BF16 source under each of them.
TRELLIS_LAYOUT_READERS = {
    "per_module": "exllamav3 1.4.2 exllamav3/modules/linear.py:391-407 (load_exl3: "
                  "key+'.suh', key+'.svh', key+'.trellis' per module)",
    "shared_h_v1": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@7c73450f "
                   "runtime/r17-g64-q-only/exl3_overlay.py:353-357,1228-1239,1667-1703 "
                   "(experts.shared_h.{proj}.rank{r}.{suh|svh} -> experts.0.{proj}.{field}, "
                   "broadcast to every expert; a per-expert H-side vector refuses)",
    "r7_shared": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@7c73450f "
                 "runtime/r17-g64-q-only/exl3_overlay.py:1655-1664,2575-2583,2668-2673 "
                 "(r7_shared.gate_up_suh -> experts.0.gate_proj.suh aliased to up_proj; "
                 "r7_shared.down_svh -> experts.0.down_proj.svh) and "
                 "patches/patch_r7_broadcast_rotations.py:7-15",
}
TRELLIS_LAYOUT_EVIDENCE = "engines/tools/layer-outer-evidence/glm52-exl3-layouts-parity.json"
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


def _tail_declared_bits(tail) -> Any:
    """Mirror of `fidelity.hfmeta.tr3_tail_declared_bits` (no bin/ import on the pod)."""
    for key in ("bits_avg", "bits", "expert_bpw_mean"):
        value = tail.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return tail.get("bits_avg", tail.get("bits"))


def _sidecar_declared_bits(doc, key, file, sha256):
    """Mirror of `fidelity.hfmeta._sidecar_declared_bits` (no bin/ import on the
    pod).  The trellis selftest's [18c] rung asserts the two produce the same
    block from the same sidecar bytes."""
    import hashlib  # noqa: F401  -- stdlib
    entries = []
    histogram = {}
    for entry in doc.values():
        rates = entry.get(key) if isinstance(entry, dict) else None
        if not isinstance(rates, list):
            continue
        for rate in rates:
            if isinstance(rate, int) and not isinstance(rate, bool):
                entries.append(rate)
                srate = str(rate)
                histogram[srate] = histogram.get(srate, 0) + 1
    if not entries:
        raise LayerOuterError(
            "REFUSED: sidecar %r key %r carries no per-expert integer bitrates"
            % (file, key))
    mean = sum(entries) / len(entries)
    source = {
        "sidecar": file,
        "key": key,
        "entries": len(entries),
        "histogram": dict(sorted(histogram.items())),
        "sha256": sha256,
    }
    return mean, source


def _read_sidecar_from_dir(model_dir, file, config, tail, skey):
    """Read `<model_dir>/<file>`, validate it against the config's MoE layer
    range and n_routed_experts, and return (doc, sha256).  Refuses by name if
    the file is absent, not strict JSON, has a layer set different from the
    config's moe_layers, or any layer's list under `skey` is not exactly
    n_routed_experts long."""
    import hashlib
    path = os.path.join(model_dir, file)
    if not os.path.isfile(path):
        raise LayerOuterError(
            "REFUSED: hybrid_tr3_tail.bits_per_expert names %r but it is absent "
            "from the checkpoint directory %s" % (file, model_dir))
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        doc = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise LayerOuterError(
            "REFUSED: sidecar %r is not strict JSON (%s)" % (file, exc)) from None
    if not isinstance(doc, dict):
        raise LayerOuterError(
            "REFUSED: sidecar %r is not a JSON object" % (file,))
    moe_layers = (tail or {}).get("moe_layers")
    expected_layers = None
    if isinstance(moe_layers, list) and len(moe_layers) == 2 and all(
            isinstance(x, int) and not isinstance(x, bool) for x in moe_layers):
        expected_layers = set(range(moe_layers[0], moe_layers[1] + 1))
    n_experts = None
    if isinstance(config, dict):
        n_experts = config.get("n_routed_experts")
    if n_experts is None:
        n_experts = getattr(config, "n_routed_experts", None)
    if n_experts is None:
        n_experts = (tail or {}).get("experts_per_layer")
    if not isinstance(n_experts, int) or isinstance(n_experts, bool):
        n_experts = None
    layers_seen = set()
    for layer, entry in doc.items():
        try:
            layers_seen.add(int(layer))
        except (TypeError, ValueError):
            raise LayerOuterError(
                "REFUSED: sidecar %r has a non-integer layer key %r" % (file, layer))
        if not isinstance(entry, dict):
            raise LayerOuterError(
                "REFUSED: sidecar %r layer %r is not an object" % (file, layer))
        rates = entry.get(skey)
        if isinstance(rates, list) and n_experts is not None and len(rates) != n_experts:
            raise LayerOuterError(
                "REFUSED: sidecar %r layer %s key %r has %d entries but the config "
                "declares n_routed_experts=%d"
                % (file, layer, skey, len(rates), n_experts))
    if expected_layers is not None and layers_seen != expected_layers:
        raise LayerOuterError(
            "REFUSED: sidecar %r covers layers %s but hybrid_tr3_tail.moe_layers "
            "declares %s" % (file, sorted(layers_seen), sorted(expected_layers)))
    return doc, hashlib.sha256(raw).hexdigest()


#: The exl3 ROTATION LAYOUTS (per_module / shared_h_v1 / r7_shared): which
#: tensor carries each module's suh and svh. Read from the index names and
#: cross-checked against the declaration; the reader rules are cited in
#: bin/fidelity/hfmeta.py above `exl3_rotation_groups`, and the evidence is
#: engines/tools/layer-outer-evidence/glm52-exl3-layouts-parity.json. The
#: block below is BYTE-IDENTICAL to bin/fidelity/hfmeta.py (no bin/ import on
#: the pod; selftest_trellis_decode_offline rung [19] asserts the two texts).
EXL3_ROTATION_LAYOUTS = ("per_module", "shared_h_v1", "r7_shared")
EXL3_SHARED_H_TENSOR_SCHEMA = (
    "model.layers.{L}.mlp.experts.shared_h.{proj}.rank{r}.{suh|svh}")
_EXL3_OBJECTS = ("trellis", "suh", "svh")
_EXL3_CODEBOOKS = ("mul1", "mcg")
_EXL3_EXPERT_RE = re.compile(
    r"^(?P<experts>.+\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"(?:\.rank(?P<rank>\d+))?$")
_EXL3_SHARED_H_RE = re.compile(
    r"^(?P<experts>.+\.experts)\.shared_h\.(?P<proj>gate_proj|up_proj|down_proj)"
    r"\.rank(?P<rank>\d+)\.(?P<field>suh|svh)$")
_EXL3_R7_SHARED_RE = re.compile(
    r"^(?P<experts>.+\.experts)\.r7_shared\.(?P<field>gate_up_suh|down_svh)$")
_EXL3_RANK_SUFFIX_RE = re.compile(r"\.rank\d+$")


def exl3_rotation_groups(keys):
    """Group `<module>.{trellis,suh,svh,<codebook>}` keys by module, resolving a
    layer-shared H-side rotation vector BY NAME where a module's own group
    omits it.

    Returns (groups, census). `groups[stem]` = {trellis, suh, svh: key,
    codebook, marker, shared: None | (field, key, layout)}. `census` =
    {layout, shared_vectors: sorted shared keys, per_layout: {layout: modules}}.
    A group is complete only when all three objects resolve AND exactly one
    codebook marker is present; a partial group, a module carrying its own
    H-side vector beside a shared one, a shared vector no module resolves, or
    two shared layouts in one checkpoint all raise ValueError.
    """
    staged = {}
    shared_h = {}
    r7 = {}
    for key in keys:
        match = _EXL3_SHARED_H_RE.match(key)
        if match is not None:
            slot = (match.group("experts"), match.group("proj"), int(match.group("rank")))
            shared_h.setdefault(slot, {})[match.group("field")] = key
            continue
        match = _EXL3_R7_SHARED_RE.match(key)
        if match is not None:
            r7.setdefault(match.group("experts"), {})[match.group("field")] = key
            continue
        stem, _, last = key.rpartition(".")
        if not stem:
            continue
        if last in _EXL3_OBJECTS:
            staged.setdefault(stem, {})[last] = key
        elif last in _EXL3_CODEBOOKS:
            staged.setdefault(stem, {}).setdefault("codebooks", []).append(last)
    groups = {}
    partial = []
    consumers = {}
    per_layout = {}
    for stem, found in staged.items():
        marks = found.get("codebooks") or []
        missing = [name for name in _EXL3_OBJECTS if name not in found]
        shared = None
        expert = _EXL3_EXPERT_RE.match(stem)
        if expert is not None:
            proj = expert.group("proj")
            h_side = "svh" if proj == "down_proj" else "suh"
            if expert.group("rank") is not None:
                slot = (expert.group("experts"), proj, int(expert.group("rank")))
                vector = shared_h.get(slot, {}).get(h_side)
                layout = "shared_h_v1"
            else:
                vector = r7.get(expert.group("experts"), {}).get(
                    "down_svh" if proj == "down_proj" else "gate_up_suh")
                layout = "r7_shared"
            if vector is not None:
                if h_side in found:
                    raise ValueError(
                        "%s carries its own %s beside the layer-shared %s; two "
                        "candidates for one rotation vector" % (stem, h_side, vector))
                found[h_side] = vector
                missing = [name for name in missing if name != h_side]
                shared = (h_side, vector, layout)
                consumers[vector] = consumers.get(vector, 0) + 1
        if missing or len(marks) != 1:
            partial.append("%s (missing %s, codebook markers %s)"
                           % (stem, missing or "none", sorted(marks) or "none"))
            continue
        groups[stem] = {name: found[name] for name in _EXL3_OBJECTS}
        groups[stem]["codebook"] = marks[0]
        groups[stem]["marker"] = "%s.%s" % (stem, marks[0])
        groups[stem]["shared"] = shared
        layout = shared[2] if shared is not None else "per_module"
        per_layout[layout] = per_layout.get(layout, 0) + 1
    if partial:
        raise ValueError(
            "%d incomplete trellis payload group(s): %s%s"
            % (len(partial), "; ".join(sorted(partial)[:3]),
               " (+%d more)" % (len(partial) - 3) if len(partial) > 3 else ""))
    vectors = sorted(key for entry in list(shared_h.values()) + list(r7.values())
                     for key in entry.values())
    orphans = [key for key in vectors if key not in consumers]
    if orphans:
        raise ValueError(
            "%d layer-shared rotation vector(s) resolve no module (e.g. %s)"
            % (len(orphans), orphans[0]))
    layouts = sorted(name for name in per_layout if name != "per_module")
    if len(layouts) > 1:
        raise ValueError(
            "two shared rotation layouts in one checkpoint: %s" % ", ".join(layouts))
    census = {"layout": layouts[0] if layouts else "per_module",
              "shared_vectors": vectors, "per_layout": dict(sorted(per_layout.items()))}
    return groups, census


def exl3_declared_module_bits(name, qc, tail):
    """The bits an artifact declares for a NON-ROUTED exl3 module, or None.

    jpsequeira: hybrid_tr3_tail.protected_tensor_policy.tensors[name].bits;
    brandonmusic: quantization_config.tensor_storage[name].bits_per_weight;
    a stock inline exl3 config: quantization_config.head_bits for lm_head.
    """
    entry = (((tail or {}).get("protected_tensor_policy") or {}).get("tensors") or {}).get(name)
    if isinstance(entry, dict):
        bits = entry.get("bits")
        if isinstance(bits, (int, float)) and not isinstance(bits, bool):
            return bits
    entry = ((qc or {}).get("tensor_storage") or {}).get(name)
    if isinstance(entry, dict):
        bits = entry.get("bits_per_weight")
        if isinstance(bits, (int, float)) and not isinstance(bits, bool):
            return bits
    if name == "lm_head":
        bits = (qc or {}).get("head_bits")
        if isinstance(bits, (int, float)) and not isinstance(bits, bool):
            return bits
    return None


def _exl3_names_sha256(names):
    if not names:
        return None
    import hashlib
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def exl3_layout_contract(keys, qc, tail):
    """The rotation-layout half of an exl3 weights_decode contract, from the
    index names and the config alone.

    Returns (contract, detail). `contract` is bound field for field into
    `weights_decode.quantization_config` on both the controller and the pod:
    rotation_layout, shared_vectors {count, names_sha256}, nonrouted_exl3
    {count, names_sha256, declared_bits histogram}, activation_scheme.
    `detail` carries what the pod's decoder needs beyond the contract: the
    groups, the census, the per-module declared bits of the non-routed
    modules and the r7 k_values. Raises ValueError when the names and the
    declaration disagree.
    """
    qc = qc if isinstance(qc, dict) else {}
    tail = tail if isinstance(tail, dict) else {}
    groups, census = exl3_rotation_groups(keys)
    layout = census["layout"]
    declared_layout = tail.get("rotation_layout")
    if layout == "shared_h_v1":
        if (declared_layout != "shared_h_v1"
                or tail.get("shared_h_tensor_schema") != EXL3_SHARED_H_TENSOR_SCHEMA):
            raise ValueError(
                "the index stores layer-shared H-side rotations under experts.shared_h "
                "but hybrid_tr3_tail declares rotation_layout=%r, shared_h_tensor_schema=%r "
                "(the authors' reader requires 'shared_h_v1' and %r)"
                % (declared_layout, tail.get("shared_h_tensor_schema"),
                   EXL3_SHARED_H_TENSOR_SCHEMA))
    elif declared_layout not in (None, "per_expert_v1"):
        raise ValueError(
            "hybrid_tr3_tail declares rotation_layout=%r but the index carries no "
            "experts.shared_h vector" % (declared_layout,))
    r7 = qc.get("r7_routed_experts")
    if layout == "r7_shared":
        if not isinstance(r7, dict) or not r7.get("schema"):
            raise ValueError(
                "the index stores layer-shared rotations under experts.r7_shared but "
                "quantization_config declares no r7_routed_experts block (the authors' "
                "reader keys the r7_shared aliasing on it)")
    nonrouted = sorted({_EXL3_RANK_SUFFIX_RE.sub("", stem) for stem in groups
                        if _EXL3_EXPERT_RE.match(stem) is None})
    module_bits = {name: exl3_declared_module_bits(name, qc, tail) for name in nonrouted}
    histogram = {}
    for bits in module_bits.values():
        if isinstance(bits, float) and bits.is_integer():
            bits = int(bits)
        label = str(bits) if bits is not None else "undeclared"
        histogram[label] = histogram.get(label, 0) + 1
    overlay = tail.get("online_mxfp8_overlay")
    activation = None
    if isinstance(overlay, dict) and overlay:
        activation = overlay.get("activation") or overlay.get("format")
    if activation is None:
        activation = qc.get("activation_scheme")
    contract = {
        "rotation_layout": layout,
        "shared_vectors": {"count": len(census["shared_vectors"]),
                           "names_sha256": _exl3_names_sha256(census["shared_vectors"])},
        "nonrouted_exl3": {"count": len(nonrouted),
                           "names_sha256": _exl3_names_sha256(nonrouted),
                           "declared_bits": dict(sorted(histogram.items()))},
        "activation_scheme": str(activation) if activation is not None else None,
    }
    r7_k_values = sorted({int(k) for k in ((r7 or {}).get("k_values") or [])
                          if isinstance(k, int) and not isinstance(k, bool)}) \
        if isinstance(r7, dict) else []
    detail = {"groups": groups, "census": census, "nonrouted_bits": module_bits,
              "r7_k_values": r7_k_values,
              "r7_declaration": ({k: r7.get(k) for k in ("schema", "feature", "moe_layers",
                                                          "k_values", "bit_map_manifests",
                                                          "loader_implementation_status")}
                                 if isinstance(r7, dict) else None)}
    return contract, detail


def trellis_checkpoint_plan(config, declared_keys: Sequence[str],
                                model_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The exact EXL3 trellis form this schedule decodes, or None.

    Accepts `quant_method: exl3` whose payload groups are the stock
    exllamav3 object layout `M.{trellis,suh,svh,<codebook>}`, one group per
    quantized module, `<codebook>` in {mul1, mcg} PER MODULE -- drowzeys'
    `keys-GLM-5.3-EXL3` uses `mcg` on layer 3 and `mul1` on layers 4-77, so
    the codebook is read from the object each module actually carries and not
    from `quantization_config.codebook`, which names only one of them -- and
    the two layer-shared rotation layouts (`shared_h_v1`, `r7_shared`; see
    `exl3_rotation_groups`), where a routed expert's H-side vector is resolved
    BY NAME from the layer's shared tensor and refused when it cannot be.
    """
    qc = _config_dict(getattr(config, "quantization_config", None)
                      if not isinstance(config, dict) else config.get("quantization_config"))
    tail = trellis_tail_declaration(config)
    declared_method = qc.get("quant_method")
    if declared_method != "exl3" and tail is None:
        return None
    try:
        layout, detail = exl3_layout_contract(declared_keys, qc, tail)
    except ValueError as exc:
        raise LayerOuterError("REFUSED: %s" % exc) from None
    groups = detail["groups"]
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
            # The first NUMERIC of bits_avg / bits / expert_bpw_mean -- willfalco's
            # GLM-5.2 tails declare `bits: "mixed"` beside `expert_bpw_mean`;
            # byte-identical to bin/fidelity/hfmeta.tr3_tail_declared_bits, which
            # the controller mirror uses, so the contract's `bits` agrees.
            "bits": _tail_declared_bits(tail),
            "head_bits": None,
            "modules_to_not_convert": [],
        }
        # jpsequeira's GLM-5.2 TR3 declares `bits: "mixed"` with no numeric
        # and a `bits_per_expert: "<file>:<key>" sidecar shipped in the repo
        # root.  The pod reads `<model_dir>/<file>`, validates its layer set
        # against moe_layers and each list against n_routed_experts, and the
        # declared bits become the exact float mean of every entry.  The
        # `declared_bits_source` block is byte-identical to the controller's
        # mirror (same sha256).  Without model_dir (e.g. the layout-parity
        # tool) the legacy string stays, as today.
        ref = tail.get("bits_per_expert")
        if model_dir is not None and isinstance(ref, str) and ":" in ref:
            sfile, _, skey = ref.partition(":")
            sfile, skey = sfile.strip(), skey.strip()
            if sfile and skey:
                doc, sha = _read_sidecar_from_dir(model_dir, sfile, config, tail, skey)
                mean, source = _sidecar_declared_bits(doc, skey, sfile, sha)
                contract["bits"] = mean
                contract["declared_bits_source"] = source
    else:
        contract = {
            "quant_method": "exl3",
            "codebook": str(qc["codebook"]) if qc.get("codebook") is not None else None,
            "bits": qc.get("bits"),
            "head_bits": qc.get("head_bits"),
            "modules_to_not_convert": sorted(
                str(m) for m in (qc.get("modules_to_not_convert") or [])),
        }
    # The rotation layout, the shared-vector and non-routed name digests and
    # the declared activation overlay are CONTRACT: read from the index names
    # on both sides by the byte-identical `exl3_layout_contract`.
    contract.update(layout)
    contract["_observed"] = {
        "quantized_module_count": len(groups),
        "codebook_histogram": dict(sorted(codebooks.items())),
        "quant_method_declared": declared_method,
        "declared_by": "hybrid_tr3_tail" if tail is not None else "quantization_config",
        "composition": composition,
        "rotation_layout": detail["census"]["layout"],
        "modules_per_layout": detail["census"]["per_layout"],
        "shared_vector_count": len(detail["census"]["shared_vectors"]),
        "r7_declaration": detail["r7_declaration"],
        # What the decoder checks each module's K against beyond the tail's
        # k_values: the declared bits of every non-routed exl3 module and the
        # r7 block's own k_values for the unsharded routed experts.
        "module_bits_policy": {"nonrouted": detail["nonrouted_bits"],
                               "r7_k_values": detail["r7_k_values"]},
    }
    return contract


def trellis_payload_groups(keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Group `<module>.{trellis,suh,svh,<codebook>}` keys by module.

    A group is returned only when all three payload objects AND exactly one
    codebook marker are present; a partial group is a refusal, not a skip,
    because a module whose trellis is loaded without its scales is the silent
    failure this decoder exists to prevent. Under `shared_h_v1` / `r7_shared`
    a routed expert's missing H-side vector resolves BY NAME to the layer's
    shared tensor (`exl3_rotation_groups`); every group then names the exact
    key each of its three objects is read from, and `shared` says which.
    """
    try:
        groups, _ = exl3_rotation_groups(list(keys))
    except ValueError as exc:
        raise LayerOuterError("REFUSED: %s" % exc) from None
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
                               expected_shape: Optional[Callable[[str], Optional[Tuple[int, ...]]]] = None,
                               fp8_parity_all: bool = False,
                               sink: Optional[Callable[[str, Any], bool]] = None
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
    subset carries both surfaces and both hooks must run over it. The FP8
    half decodes on `device` too, under `materialize_fp8_subset`'s parity gate.

    RANK-SHARDED modules are held per rank in `torch_dtype`, not fp32, and
    composed the moment their last rank decodes: a concatenation places
    elements and a cast rounds each element on its own, so cast-then-cat is
    the same bytes as cat-then-cast (`selftest_trellis_decode_offline.py`
    [6c] asserts it), and a 768-module layer no longer parks 38.7 GB of fp32
    shards on the device until the loop ends. `sink` is the direct expert
    fill's hook, as in `materialize_fp8_subset`.
    """
    surface = _exl3hf()
    groups = trellis_payload_groups(subset)
    # Every key a group reads -- including a layer-shared rotation vector
    # several groups resolve to -- is consumed here and never reaches the
    # converter as a stray tensor.
    consumed = {key for objects in groups.values()
                for name, key in objects.items() if name not in ("codebook", "shared")}
    passthrough = {key: value for key, value in subset.items() if key not in consumed}
    policy = stats.get("module_bits_policy") or {}
    # The FP8 half counts into ITS OWN counter dict: the two decoders keep
    # separate stats and the log line reads both by name, so handing the
    # trellis dict to the FP8 decoder is a KeyError on the first dequantized
    # tensor (it was, on a live wrldsuksgo2mars pod).
    if fp8_plan is not None:
        counters = fp8_stats if fp8_stats is not None else stats
        for key in ("dequantized", "scales_consumed", "fp8_bytes"):
            counters.setdefault(key, 0)
        out: Dict[str, Any] = materialize_fp8_subset(
            passthrough, fp8_plan, torch_dtype, counters, device=device,
            parity_all=fp8_parity_all, sink=sink)
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
    tp = int(composition["tp"]) if composition is not None else None

    def emit(weight_key: str, tensor) -> None:
        if sink is not None and sink(weight_key, tensor):
            return
        out[weight_key] = tensor

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
            payload["svh"].to(device), codebook=objects["codebook"]).to(torch_dtype)
        stats["decoded_modules"] += 1
        bits = int(payload["trellis"].shape[-1]) // 16
        stats["trellis_bits"] += bits
        histogram = stats.setdefault("k_histogram", {})
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
        shared = objects.get("shared")
        layouts = stats.setdefault("modules_per_layout", {})
        layout = shared[2] if shared is not None else "per_module"
        layouts[layout] = layouts.get(layout, 0) + 1
        if shared is not None:
            stats["shared_vectors_applied"] = stats.get("shared_vectors_applied", 0) + 1
        _check_declared_bits(module, bits, plan, composition, policy=policy,
                             layout=layout, shared=shared is not None)
        ranked = TRELLIS_RANK_RE.match(module)
        weight_key = "%s.weight" % (ranked.group("module") if ranked else module)
        if weight_key in subset:
            raise LayerOuterError(
                "REFUSED: %s exists as a plain weight beside its trellis payload; the "
                "checkpoint carries two versions of one tensor and this schedule will "
                "not pick one" % weight_key)
        if _EXL3_EXPERT_RE.match(module) is None:
            # A NON-ROUTED exl3 module (o_proj, q_b_proj, indexer.wq_b, lm_head):
            # decoded by the same function wherever it sits, and named with its
            # K in the evidence -- the head's K is what hf_capture seals as the
            # candidate's own dequantized head (HEAD-1d, own heads).
            stats.setdefault("nonrouted_exl3_decoded", {})[weight_key[:-len(".weight")]] = bits
        if ranked is None:
            if layout == "r7_shared":
                # brandonmusic's r7 encoder stores each expert with its
                # INTERMEDIATE channels permuted (gate/up rows, down columns,
                # one permutation per expert: r7_encoder/permutation.py
                # permute_expert_hf); the layer manifest names it and the
                # inverse puts the decoded module in the source's order.
                decoded = _r7_unpermute(module, objects, decoded, stats)
            emit(weight_key, decoded)
            continue
        if composition is None:
            raise LayerOuterError(
                "REFUSED: %s is a rank-sharded payload but the plan carries no "
                "composition (the config declared no hybrid_tr3_tail.tp)" % module)
        by_rank = parts.setdefault(ranked.group("module"), {})
        by_rank[int(ranked.group("rank"))] = decoded
        stats["tp_rank_storage_dtype"] = str(decoded.dtype).replace("torch.", "")
        if len(by_rank) >= tp:
            del parts[ranked.group("module")]
            emit(weight_key, _compose_tp_ranks(ranked.group("module"), by_rank, composition,
                                               expected_shape, torch_dtype, stats))
    # A module still here never reached `tp` ranks; `_compose_tp_ranks` refuses
    # it by name rather than letting a partial projection go missing quietly.
    for module, by_rank in sorted(parts.items()):
        emit("%s.weight" % module, _compose_tp_ranks(
            module, by_rank, composition, expected_shape, torch_dtype, stats))
    return out


def _check_declared_bits(module: str, bits: int, plan: Dict[str, Any],
                         composition: Optional[Dict[str, Any]],
                         policy: Optional[Dict[str, Any]] = None,
                         layout: str = "per_module", shared: bool = False) -> None:
    """The payload's own K against what the artifact declares.

    A uniform declaration (`bits: 4`) must equal every module's K. A TR3 tail
    declares an AVERAGE (`bits_avg: 3.25`) over per-expert tiers with
    `k_values: [3, 4]`, so the check is membership in k_values. A declaration
    that is neither is not checkable and is recorded, not trusted: the row's
    bit-width label then rests on the K histogram in the decode evidence.

    `policy` (the plan's `module_bits_policy`) adds two per-module rules: a
    non-routed exl3 module must carry exactly the bits its artifact declares
    for it (jpsequeira's protected_tensor_policy, brandonmusic's
    tensor_storage, a stock config's head_bits), and an `r7_shared` routed
    expert must carry a K in `quantization_config.r7_routed_experts.k_values`
    -- the tail's k_values describe only the rank-sharded layer it covers.
    """
    policy = policy or {}
    name = _EXL3_RANK_SUFFIX_RE.sub("", module)
    nonrouted = policy.get("nonrouted") or {}
    if name in nonrouted:
        declared_bits = nonrouted[name]
        if declared_bits is not None and int(declared_bits) != bits:
            raise LayerOuterError(
                "REFUSED: %s is a K%d payload but the artifact declares %r bits for it"
                % (module, bits, declared_bits))
        return
    if shared and layout == "r7_shared":
        r7_k = policy.get("r7_k_values") or []
        if r7_k and bits not in set(r7_k):
            raise LayerOuterError(
                "REFUSED: %s is a K%d payload but r7_routed_experts declares k_values %s"
                % (module, bits, sorted(r7_k)))
        return
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


R7_MANIFEST_RE = re.compile(r"^r7-experts-layer-(?P<layer>\d+)\.json$")
R7_UNPERMUTE_REFERENCE = (
    "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@7c73450f r7_encoder/permutation.py:99-114 "
    "(permute_expert_hf: gate/up rows and down columns index_select'ed by the expert's "
    "permutation before encoding), :91-96 (inverse_permutation) and r7_encoder/schema.py:268-272 "
    "(the layer manifest's permutations[E].new_to_old is that permutation)")


class R7PermutationSource:
    """The per-expert intermediate permutations of an `r7_shared` artifact.

    brandonmusic's r7 encoder permutes each expert's 2048 intermediate
    channels (`energy_balanced` or identity, one permutation per expert,
    applied identically to gate rows, up rows and down columns --
    `r7_encoder/permutation.py::permute_expert_hf`) BEFORE the trellis
    encode, and writes `new_to_old` into the layer's manifest
    `r7-experts-layer-{L:03d}.json` (`permutations[E]`), the files
    `quantization_config.r7_routed_experts.bit_map_manifests` lists and the
    repository ships beside its shards. Serving never undoes it (a permutation
    of the intermediate axis is invisible to gate*up@down when all three carry
    it); this decoder does, so the decoded module sits under the official name
    in the SOURCE's channel order and can be proven against the BF16 rows
    (evidence: TRELLIS_LAYOUT_EVIDENCE -- decoded as stored, cosine ~0 against
    the source; inverse-permuted, the K-band). The same manifest's
    `vector_refs[module]` must name exactly the suh/svh keys the name
    resolution chose; a disagreement refuses.
    """

    def __init__(self, model_dir: str, manifests: Sequence[str], intermediate: int):
        self.model_dir = model_dir
        self.intermediate = int(intermediate)
        self.by_layer: Dict[int, str] = {}
        for name in manifests:
            match = R7_MANIFEST_RE.match(str(name))
            if match is None:
                raise LayerOuterError(
                    "REFUSED: r7_routed_experts.bit_map_manifests names %r, not an "
                    "r7-experts-layer-NNN.json manifest" % (name,))
            self.by_layer[int(match.group("layer"))] = str(name)
        self._layers: Dict[int, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {"manifests_read": 0, "experts_unpermuted": 0,
                                      "policies": {}, "manifest_sha256": {}}

    def verify_present(self) -> None:
        missing = [name for name in self.by_layer.values()
                   if not os.path.isfile(os.path.join(self.model_dir, name))]
        if missing:
            raise LayerOuterError(
                "REFUSED: %d r7 layer manifest(s) the config lists are absent from the "
                "checkpoint (e.g. %s); the experts' intermediate permutations cannot be "
                "undone without them" % (len(missing), missing[0]))

    def _layer(self, layer: int) -> Dict[str, Any]:
        if layer in self._layers:
            return self._layers[layer]
        name = self.by_layer.get(layer)
        if name is None:
            raise LayerOuterError(
                "REFUSED: layer %d stores r7_shared experts but r7_routed_experts."
                "bit_map_manifests lists no manifest for it" % layer)
        path = os.path.join(self.model_dir, name)
        with open(path, "rb") as handle:
            raw = handle.read()
        doc = json.loads(raw.decode("utf-8"))
        if int(doc.get("layer", -1)) != layer:
            raise LayerOuterError(
                "REFUSED: %s says layer %r, expected %d" % (name, doc.get("layer"), layer))
        self._layers[layer] = doc
        self.stats["manifests_read"] += 1
        self.stats["manifest_sha256"][name] = hashlib.sha256(raw).hexdigest()
        return doc

    def inverse(self, layer: int, expert: int, module: str, objects: Dict[str, Any]):
        """The index that puts a decoded expert's intermediate axis back in
        source order, after the manifest's vector_refs confirm the name resolution."""
        import torch

        doc = self._layer(layer)
        refs = (doc.get("vector_refs") or {}).get(module)
        chosen = {"suh": objects["suh"], "svh": objects["svh"]}
        if refs != chosen:
            raise LayerOuterError(
                "REFUSED: %s: the layer manifest's vector_refs name %r but the index names "
                "resolved %r" % (module, refs, chosen))
        entry = (doc.get("permutations") or {}).get(str(expert))
        perm = entry.get("new_to_old") if isinstance(entry, dict) else None
        if (not isinstance(perm, list) or len(perm) != self.intermediate
                or sorted(perm) != list(range(self.intermediate))):
            raise LayerOuterError(
                "REFUSED: %s: the layer manifest carries no valid %d-element permutation "
                "for expert %d" % (module, self.intermediate, expert))
        policy = str(entry.get("policy"))
        self.stats["policies"][policy] = self.stats["policies"].get(policy, 0) + 1
        self.stats["experts_unpermuted"] += 1
        inverse = torch.empty(self.intermediate, dtype=torch.long)
        inverse[torch.tensor(perm, dtype=torch.long)] = torch.arange(self.intermediate)
        return inverse


def _r7_unpermute(module: str, objects: Dict[str, Any], decoded, stats: Dict[str, Any]):
    source = stats.get("r7_permutations")
    if not isinstance(source, R7PermutationSource):
        raise LayerOuterError(
            "REFUSED: %s is an r7_shared expert but the decode carries no permutation "
            "source (the layer manifests were not planned)" % module)
    expert = _EXL3_EXPERT_RE.match(module)
    layer_match = re.search(r"\.layers\.(\d+)\.", module)
    if expert is None or layer_match is None:
        raise LayerOuterError("REFUSED: %s is not a routed-expert projection" % module)
    inverse = source.inverse(int(layer_match.group(1)), int(expert.group("expert")), module, objects)
    axis = 1 if expert.group("proj") == "down_proj" else 0
    if decoded.shape[axis] != source.intermediate:
        raise LayerOuterError(
            "REFUSED: %s decodes to %s; its intermediate axis is not %d"
            % (module, tuple(decoded.shape), source.intermediate))
    return decoded.index_select(axis, inverse.to(decoded.device))


def r7_permutation_source(config, model_dir: str, declaration: Dict[str, Any]) -> R7PermutationSource:
    """The permutation source an `r7_shared` decode needs, from the artifact's own
    declaration (`r7_routed_experts.bit_map_manifests`) and geometry; every
    listed manifest must be present in the checkpoint directory."""
    manifests = declaration.get("bit_map_manifests") or []
    if not isinstance(manifests, list) or not manifests:
        raise LayerOuterError(
            "REFUSED: the index stores r7_shared experts but r7_routed_experts declares no "
            "bit_map_manifests; the experts' intermediate permutations live in those files")
    text = _config_dict(getattr(config, "text_config", None)
                        if not isinstance(config, dict) else config.get("text_config"))
    geometry = text if text else _config_dict(config)
    intermediate = geometry.get("moe_intermediate_size")
    if isinstance(intermediate, bool) or not isinstance(intermediate, int) or intermediate <= 0:
        raise LayerOuterError(
            "REFUSED: r7_shared experts need the config's moe_intermediate_size to validate "
            "their permutations; got %r" % (intermediate,))
    source = R7PermutationSource(model_dir, manifests, intermediate)
    source.verify_present()
    return source


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
        # davidsyoung writes "K-sliced: rank r = input cols"; the GLM-5.2 TR3
        # tails (willfalco, jpsequeira, brandonmusic) write "TP4 K-slice: rank r
        # owns input columns [512r,512r+512)". The axis token is what is read;
        # a declaration naming neither, or both, still refuses.
        tokens = set(re.findall(r"\b([NK])-slice[d]?\b", str(declared)))
        declared_axis = (0 if tokens == {"N"} else 1 if tokens == {"K"} else None)
        if declared_axis != axis:
            raise LayerOuterError(
                "REFUSED: %s: the shapes admit concatenation along axis %d but the "
                "artifact declares %r" % (module, axis, declared))
    # The shards arrive already in `torch_dtype` (see materialize_trellis_subset):
    # the cat places bytes and the `.to` is then the identity. Kept so a caller
    # handing fp32 shards still gets the declared dtype.
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


# ---------------------------------------------------------------------------
# NVFP4 (modelopt) checkpoints: decode to bf16 on the capture device, per module
# ---------------------------------------------------------------------------

NVFP4_DECODE_METHOD = "nvfp4-modelopt-dequant-to-bf16"
#: Every byte of decode math is `engines/tools/nvfp4_surface.py`'s
#: `dequant_nvfp4(weight_scale_2=...)`: LOW nibble first, the e2m1 LUT
#: [0, .5, 1, 1.5, 2, 3, 4, 6] (nibble 0b1000 is -0.0), the f8e4m3 scale
#: promoted exactly to fp32 and multiplied ONCE by the fp32 `weight_scale_2`,
#: one multiply per element, one cast to bf16 -- proven bitwise against
#: compressed-tensors 0.18.0 `unpack_fp4_from_uint8` + the modelopt scale
#: convention on real ranged-fetched rows of all three flagship exports
#: (`engines/tools/nvfp4-evidence/glm53-nvfp4-parity.json`, max_abs_diff
#: exactly 0.0 in fp32 and after the bf16 cast; `selftest_nvfp4_offline.py`
#: rung 11 re-derives it live). This module adds NO arithmetic: it groups a
#: routed module's {weight, weight_scale, weight_scale_2} from the layer's
#: lazy subset, decodes on the capture device, and hands the dense tensor
#: under the OFFICIAL name to the same converter a bf16 checkpoint reaches.
#: `input_scale` is an ACTIVATION quantity (the static per-tensor scale a
#: W4A4 kernel applies to x) and is never applied to weights; the plan says
#: `activation_scheme: static-nvfp4-not-applied` so the receipt does too.
NVFP4_DECODE_REFERENCE = "engines/tools/nvfp4_surface.py::dequant_nvfp4"
NVFP4_PARITY_EVIDENCE = "engines/tools/nvfp4-evidence/glm53-nvfp4-parity.json"


def _nvfp4():
    """Import the decode ABI lazily, like `_exl3hf`."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import nvfp4_surface

    return nvfp4_surface


def nvfp4_checkpoint_plan(config, model_dir: str) -> Optional[Dict[str, Any]]:
    """The exact modelopt NVFP4 form this schedule decodes, or None.

    Pure detection from config.json + the index: `quant_method: modelopt`,
    `quant_algo: NVFP4`, a group-16 4-bit weight declaration in either of the
    two spellings modelopt exports use, no declared online transform, a
    known family geometry, every routed module in the modelopt
    {weight U8, weight_scale F8_E4M3, weight_scale_2 F32} (+input_scale)
    layout, and a non-routed name set equal to the official BF16 release's.
    Any other modelopt block is refused BY NAME by `nvfp4_surface`; a config
    that is not modelopt at all returns None so the FP8 gate decides.

    The returned dict is the CONTRACT block (`quantization_config` in the
    sealed receipt, compared field for field against the controller's
    stdlib mirror in `measure_cloud._candidate_decode_plan`), with the index
    census under the private `_observed` key the caller pops.
    """
    if _quant_method(config) != "modelopt":
        return None
    surface = _nvfp4()
    cfg = _config_dict(config)
    cfg = dict(cfg)
    cfg["quantization_config"] = _config_dict(cfg.get("quantization_config"))
    algo = cfg["quantization_config"].get("quant_algo")
    if algo != "NVFP4":
        raise LayerOuterError(
            "REFUSED: quantization_config quant_method='modelopt' quant_algo=%r is not the "
            "NVFP4 form this schedule decodes (modelopt NVFP4: e2m1 group-16 routed experts); "
            "another modelopt form needs its decoder authored and proven bitwise first."
            % (algo,))
    weight_map = _index_weight_map(model_dir)
    try:
        plan = surface.modelopt_nvfp4_plan(cfg, weight_map)
    except ValueError as exc:
        raise LayerOuterError(
            "REFUSED: %s. This schedule decodes the modelopt NVFP4 dialect only; "
            "another modelopt form needs its decoder authored and proven bitwise first."
            % exc) from None
    contract = dict(plan["contract"])
    contract["_observed"] = dict(plan["observed"])
    contract["_geometry"] = plan["geometry"]
    return contract


def materialize_nvfp4_subset(subset: Dict[str, Any], plan: Dict[str, Any], torch_dtype,
                             stats: Dict[str, Any], device: str = "cpu",
                             geometry=None) -> Dict[str, Any]:
    """Replace every modelopt NVFP4 module in a lazy subset by one decoded `.weight`.

    Decodes on `device` -- the capture device, not the host: 768 modules per
    MoE layer x 75 layers, and the host-side FP8 decode was measured at 80 %
    of a cold run. Each module's three components are read from the lazy
    slices, moved to the device as packed bytes (a quarter of the bf16 size),
    decoded to exact fp32 there and cast once to `torch_dtype`. The decoded
    tensor lands under the OFFICIAL per-expert name; the packed components
    and the activation `input_scale` never reach the converter.

    Non-routed tensors pass through untouched when they are the official-
    named bf16/fp32 tensors the plan verified by name; a packed dtype
    (uint8, float8) outside a routed module is refused, because loading it
    as-is would read encoded bytes as weights. A routed module missing a
    component, or a routed name with a component this dialect does not
    ship, is refused by name rather than skipped.
    """
    import torch

    surface = _nvfp4()
    geometry = geometry or plan.get("_geometry") or surface.GLM_MOE_DSA_GEOMETRY
    expert_re = geometry.expert_re()
    decode = tuple(surface.MO_NVFP4_DECODE)
    activation = set(surface.MO_NVFP4_ACTIVATION)
    modules: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    out: Dict[str, Any] = {}
    for key, value in subset.items():
        match = expert_re.match(key)
        if match is None:
            dtype = getattr(value, "dtype", None)
            if dtype is None and hasattr(value, "get_dtype"):
                dtype = value.get_dtype()
            name = str(dtype)
            if name in ("torch.uint8", "U8", "uint8", "torch.float8_e4m3fn", "F8_E4M3",
                        "float8_e4m3fn", "torch.float8_e5m2", "F8_E5M2", "float8_e5m2"):
                raise LayerOuterError(
                    "REFUSED: %s is a %s tensor outside a routed-expert module; the modelopt "
                    "NVFP4 dialect packs routed experts only, so this is a payload this "
                    "schedule has no decoder for" % (key, name))
            census = stats.setdefault("nonrouted_by_dtype", {})
            census[name] = census.get(name, 0) + 1
            if name not in ("torch.bfloat16", "BF16", "bfloat16"):
                examples = stats.setdefault("nonrouted_non_bf16_examples", [])
                if len(examples) < 8:
                    examples.append({"name": key, "dtype": name})
            out[key] = value
            continue
        layer, expert = int(match.group(1)), int(match.group(2))
        projection, component = match.group(3), match.group(4)
        if component in activation:
            stats["input_scales_skipped"] = stats.get("input_scales_skipped", 0) + 1
            continue
        if component not in decode:
            raise LayerOuterError(
                "REFUSED: %s carries component %r, which the modelopt NVFP4 dialect does "
                "not ship (%s)" % (key, component, "/".join(decode + tuple(activation))))
        modules.setdefault((layer, expert, projection), {})[component] = key
    for (layer, expert, projection), keys in sorted(modules.items()):
        weight_key = geometry.official_name(layer, expert, projection)
        if set(keys) == {"weight"}:
            # A routed module shipped WHOLE (the MTP layer's experts, or a
            # producer that left a layer native): only bf16 passes, and only
            # when the plan's census called that layer plain-weight.
            value = subset[keys["weight"]]
            dtype = getattr(value, "dtype", None)
            if dtype is None and hasattr(value, "get_dtype"):
                dtype = value.get_dtype()
            if str(dtype) not in ("torch.bfloat16", "BF16", "bfloat16"):
                raise LayerOuterError(
                    "REFUSED: %s ships as a lone %s `weight` with no weight_scale / "
                    "weight_scale_2 beside it; a packed tensor without its scales cannot "
                    "be decoded and a non-bf16 one is not the official form"
                    % (weight_key, dtype))
            out[weight_key] = value
            stats["plain_modules_passed"] = stats.get("plain_modules_passed", 0) + 1
            continue
        missing = [name for name in decode if name not in keys]
        if missing:
            raise LayerOuterError(
                "REFUSED: %s is missing %s beside %s; a routed module with part of its "
                "NVFP4 payload cannot be decoded and will not be loaded as-is"
                % (weight_key, "/".join(missing), sorted(keys.values())))
        packed = _eager(subset[keys["weight"]])
        scale = _eager(subset[keys["weight_scale"]])
        scale_2 = _eager(subset[keys["weight_scale_2"]])
        if packed.dtype != torch.uint8:
            raise LayerOuterError(
                "REFUSED: %s packed weight is %s, not uint8" % (weight_key, packed.dtype))
        if scale.dtype != torch.float8_e4m3fn:
            raise LayerOuterError(
                "REFUSED: %s weight_scale is %s, not float8_e4m3fn" % (weight_key, scale.dtype))
        if scale_2.dtype != torch.float32 or tuple(scale_2.shape) not in ((), (1,)):
            raise LayerOuterError(
                "REFUSED: %s weight_scale_2 is %s %s, not an fp32 scalar"
                % (weight_key, scale_2.dtype, tuple(scale_2.shape)))
        try:
            decoded = surface.dequant_nvfp4(
                packed.to(device), scale.to(device), weight_scale_2=scale_2.to(device))
        except ValueError as exc:
            raise LayerOuterError("REFUSED: %s: %s" % (weight_key, exc)) from None
        want = geometry.projection_shape[projection]
        if tuple(decoded.shape) != want:
            raise LayerOuterError(
                "REFUSED: %s decodes to %s, not the %s the geometry declares for %s"
                % (weight_key, tuple(decoded.shape), want, projection))
        if not torch.isfinite(decoded).all():
            raise LayerOuterError(
                "REFUSED: %s decodes to a non-finite value; a corrupt scale is never "
                "clamped into plausibility" % weight_key)
        out[weight_key] = decoded.to(torch_dtype)
        stats["decoded_modules"] = stats.get("decoded_modules", 0) + 1
        stats["packed_bytes"] = stats.get("packed_bytes", 0) + int(packed.numel())
        stats["scales_consumed"] = stats.get("scales_consumed", 0) + 2
    return out


#: The llama.cpp GGUF lane. Every byte of decode math is
#: `engines/tools/gguf_surface.py`: block dequantizers transliterated from
#: gguf-py 0.19.0 and proven BITWISE against `gguf.quants.dequantize` on real
#: ranged-fetched blocks of the measured builds (F32/F16/BF16/Q8_0/Q4_K/Q5_K/
#: Q6_K/Q3_K/IQ4_XS/IQ3_XXS/IQ3_S; gguf-evidence/dequant_*_ggufpy_ref.npy,
#: `selftest_gguf_offline.py` rung 1), plus the glm-dsa name map, the per-head
#: `kv_b_proj` composition and the fused-expert slot slicing, each proven EXACT
#: against zai-org/GLM-5.3-BF16 (gguf-evidence/glmdsa-layout-audit.json). This
#: module adds NO arithmetic: per layer it asks the surface for that layer's
#: tensors under their OFFICIAL names -- dequantized on the capture device,
#: composed, cast once to bf16 (official-float32 tensors kept fp32) -- and
#: hands them to the same converter a bf16 checkpoint reaches. There is no
#: safetensors tree at all: the "subset" of a layer is a set of official names
#: the container proves it carries, and the bytes are read at decode time.
GGUF_DECODE_METHOD = "gguf-dequant-to-bf16"
GGUF_DECODE_REFERENCE = "engines/tools/gguf_surface.py::materialize_layer"
GGUF_PARITY_EVIDENCE = ("engines/tools/gguf-evidence/manifest.json#dequant_fixtures + "
                        "engines/tools/gguf-evidence/glmdsa-layout-audit.json")


def _gguf():
    """Import the surface lazily, like `_exl3hf`."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import gguf_surface

    return gguf_surface


def gguf_files_in(model_dir: str) -> List[str]:
    """Every *.gguf under `model_dir` or exactly one build directory below it.

    A GGUF repo is a shelf of builds in subdirectories; the plan downloads one
    build's files under its repo-relative path, so the tree on the pod is
    `<model_dir>/<build>/*.gguf` (plus the sidecars at the top). Two build
    directories is a refusal: a measurement describes ONE artifact.
    """
    top = sorted(os.path.join(model_dir, n) for n in os.listdir(model_dir)
                 if n.endswith(".gguf") and os.path.isfile(os.path.join(model_dir, n)))
    builds = {}
    for name in sorted(os.listdir(model_dir)):
        sub = os.path.join(model_dir, name)
        if os.path.isdir(sub):
            files = sorted(os.path.join(sub, n) for n in os.listdir(sub) if n.endswith(".gguf"))
            if files:
                builds[name] = files
    if top and builds:
        raise LayerOuterError(
            "REFUSED: %s holds .gguf files both at its top level and under %s; which "
            "build is the artifact?" % (model_dir, ", ".join(sorted(builds))))
    if len(builds) > 1:
        raise LayerOuterError(
            "REFUSED: %s holds %d GGUF build directories (%s); a measurement describes "
            "ONE build" % (model_dir, len(builds), ", ".join(sorted(builds))))
    return top or (next(iter(builds.values())) if builds else [])


class _GgufSlot(object):
    """The lazy 'slice' of one official tensor the GGUF lane will decode.

    Stands where a `PySafeSlice` stands in the layer subsets: it names the
    layer whose decode produces the tensor (`RESIDENT_LAYER` for embed/norm/
    head). Nothing is read until `materialize_gguf_subset` decodes the layer.
    """
    __slots__ = ("layer", "name")

    def __init__(self, layer: int, name: str):
        self.layer = layer
        self.name = name


def gguf_checkpoint_plan(config, model_dir: str) -> Optional[Dict[str, Any]]:
    """The exact GGUF form this schedule decodes, or None when there is no GGUF.

    Pure detection from the container headers: `general.architecture` must be
    one the surface's arch table knows (refused by name otherwise), the
    geometry gate must hold, every tensor must be nameable and decodable, the
    whole-file sha256 marker `gguf-files-verified.json` must sit beside the
    build (the fetch stage writes it; without it the identity the receipt
    claims is unverified), and the OFFICIAL config (the tree's config.json,
    copied from the reference release) must say which layers own a DSA
    indexer so the artifact's shared-layer copies can be recognised -- and
    then PROVEN value-identical to their parents (`verify_shared_indexer_copies`)
    before a single layer is decoded.

    Returns the CONTRACT block (`weights_decode` = {method, quantization_config},
    compared field for field against the controller's header-only mirror in
    `measure_cloud._candidate_decode_plan`) with the loaded surface and its
    partition under private `_` keys the caller pops.
    """
    files = gguf_files_in(model_dir)
    if not files:
        return None
    surface_mod = _gguf()
    cfg = _config_dict(config)
    try:
        container = surface_mod.GgufContainer([surface_mod.GgufFile(f) for f in files])
        arch = surface_mod.arch_for(container.architecture)
        full = surface_mod.indexer_full_layers_from_config(cfg, arch)
        if arch.indexer_shared_copies and full is None:
            raise LayerOuterError(
                "REFUSED: %s carries indexer tensors on every layer and the tree's "
                "config.json declares no indexer_types; copy the official release's "
                "config.json beside the build (the candidate stage does)" % arch.key)
        surface = surface_mod.load_gguf_surface(
            files, repo=None, revision=None, require_file_hashes=True,
            indexer_full_layers=full)
        audit = surface_mod.audit_container(surface.container)
        copies = surface_mod.verify_shared_indexer_copies(surface.container, surface.census)
    except ValueError as exc:
        raise LayerOuterError(
            "REFUSED: %s. This schedule decodes the llama.cpp GGUF builds whose "
            "architecture and ggml types gguf_surface has proven; another form needs "
            "its kernel or name map authored and proven bitwise first." % exc) from None
    # the official geometry the model will be built with must be the GGUF's
    layers_declared = cfg.get("num_hidden_layers")
    if layers_declared != arch.mtp_layer:
        raise LayerOuterError(
            "REFUSED: config.json declares %r decoder layers but the %s GGUF carries "
            "%d decoder blocks before its MTP block" % (layers_declared, arch.key, arch.mtp_layer))
    build = os.path.basename(os.path.dirname(files[0])) if os.path.dirname(files[0]) != model_dir.rstrip("/") else ""
    contract = surface_mod.decode_contract(surface.container, build)
    plan = dict(contract["quantization_config"])
    plan["_surface"] = surface
    plan["_partition"] = surface_mod.layer_partition(surface.census)
    plan["_observed"] = {
        "architecture": arch.key,
        "family": arch.family,
        "files_verified": surface.file_hash_verification,
        "file_records": list(surface.file_records),
        "container_audit": audit,
        "shared_indexer_copies": copies,
        "materialize_plan": surface_mod.materialize_plan(surface),
        "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
    }
    return plan


def gguf_subsets(plan: Dict[str, Any]) -> Dict[str, Any]:
    """{official name: _GgufSlot} for every tensor the container carries.

    This is what stands in for opening the safetensors shards: the streamer
    routes these names by layer exactly as it routes checkpoint keys, so the
    MTP block's 791 official names land above the model's layer count and are
    reported as unexpected (the authored allowlist), the resident set is the
    three top-level tensors, and every decoder layer gets its own bucket.
    """
    surface_mod = _gguf()
    surface = plan["_surface"]
    arch, census = surface.arch, surface.census
    slots: Dict[str, Any] = {}
    for layer, names in plan["_partition"].items():
        for gguf_name in names:
            role = surface_mod.classify_tensor(gguf_name, arch)
            if role[0] == "top":
                slots[role[1]] = _GgufSlot(layer, role[1])
            elif role[0] == "direct":
                slots[role[2]] = _GgufSlot(layer, role[2])
            elif role[0] == "mla":
                slots[arch.kv_b_name(layer)] = _GgufSlot(layer, arch.kv_b_name(layer))
            elif role[0] == "routed":
                for expert in range(arch.num_experts):
                    name = arch.expert_name(layer, expert, role[2])
                    slots[name] = _GgufSlot(layer, name)
    return slots


def materialize_gguf_subset(subset: Dict[str, Any], plan: Dict[str, Any], torch_dtype,
                            stats: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """Decode one layer's GGUF tensors into official-named dense tensors.

    `subset` is a bucket of `_GgufSlot`s that all name ONE layer (the streamer
    buckets by layer); the surface decodes that layer on `device` -- the
    quantized bytes cross the bus, not fp32 -- and the result is offered to the
    converter (or the direct expert fill) under the official names. The set of
    names produced must equal the set the bucket named: a tensor the container
    promised and did not deliver is a refusal, not a random initialisation.
    """
    if not subset:
        return {}
    foreign = [key for key, value in subset.items() if not isinstance(value, _GgufSlot)]
    if foreign:
        raise LayerOuterError(
            "REFUSED: a GGUF bucket holds %d keys that are not GGUF slots (e.g. %s)"
            % (len(foreign), foreign[0]))
    # A layer's bucket is that layer; the RESIDENT bucket is the three top-level
    # tensors plus one router-correction BUFFER per MoE layer (the ungated
    # streamer loads buffers resident), so it asks each of those layers for
    # exactly that name and nothing else.
    by_layer: Dict[int, List[str]] = {}
    for key, value in subset.items():
        by_layer.setdefault(value.layer, []).append(key)
    surface_mod = _gguf()
    out: Dict[str, Any] = {}
    try:
        for layer, names in sorted(by_layer.items()):
            out.update(surface_mod.materialize_layer(
                plan["_surface"], layer, torch_dtype=torch_dtype, device=device,
                stats=stats, only=names))
    except ValueError as exc:
        raise LayerOuterError("REFUSED: %s" % exc) from None
    layer = max(by_layer) if len(by_layer) == 1 else -1
    if set(out) != set(subset):
        missing = sorted(set(subset) - set(out))[:5]
        stray = sorted(set(out) - set(subset))[:5]
        raise LayerOuterError(
            "REFUSED: the GGUF decode of layer %d produced %d tensors but the bucket "
            "named %d (missing %s, stray %s)" % (layer, len(out), len(subset), missing, stray))
    stats["layers_decoded"] = stats.get("layers_decoded", 0) + 1
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
                  device: str = "cpu", expected_shape=None,
                  nvfp4_plan=None, nvfp4_stats=None,
                  fp8_parity_all: bool = False, sink=None,
                  gguf_plan=None, gguf_stats=None) -> Dict[str, Any]:
    """Whichever decoders this artifact needs, in the one order that is safe.

    `fp8_parity_all` and `sink` reach the FP8 and trellis decoders (see
    `materialize_fp8_subset`); the NVFP4 decoder keeps its own contract and
    hands back the decoded dict, which `do_load` then offers to the sink.
    """
    if trellis_plan is not None:
        composition = (trellis_plan.get("_observed") or {}).get("composition")
        if composition is None:
            composition = trellis_stats.get("composition")
        subset = truncate_zero_padded_rows(subset, expected_shape, trellis_stats)
        return materialize_trellis_subset(
            subset, trellis_plan, torch_dtype, trellis_stats,
            fp8_plan=trellis_fp8_plan, fp8_stats=fp8_stats, device=device,
            composition=composition, expected_shape=expected_shape,
            fp8_parity_all=fp8_parity_all, sink=sink)
    if nvfp4_plan is not None:
        return materialize_nvfp4_subset(
            subset, nvfp4_plan, torch_dtype,
            nvfp4_stats if nvfp4_stats is not None else {}, device=device)
    if gguf_plan is not None:
        return materialize_gguf_subset(
            subset, gguf_plan, torch_dtype,
            gguf_stats if gguf_stats is not None else {}, device=device)
    if fp8_plan is not None:
        return materialize_fp8_subset(subset, fp8_plan, torch_dtype, fp8_stats,
                                      device=device, parity_all=fp8_parity_all, sink=sink)
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


# ---------------------------------------------------------------------------
# the direct expert fill: routed experts written straight into the fused
# parameter, around the converter's stack-and-cat transient
# ---------------------------------------------------------------------------

EXPERT_FILL_DIRECT = "direct"
EXPERT_FILL_CONVERTER = "converter"
EXPERT_FILL_MODES = (EXPERT_FILL_DIRECT, EXPERT_FILL_CONVERTER)
#: What `transformers` does to a MoE layer's routed experts and what this
#: reproduces byte for byte: `MergeModulelist(dim=0)` stacks the E per-expert
#: sources of one pattern in natural-key order (expert 0..E-1), then
#: `Concatenate(dim=1)` cats the stacks in SOURCE-PATTERN ORDER (gate rows
#: first, then up). Measured on the H200 pod (review-efficiency section 3),
#: the cat is a second 12.9 GB copy of gate_up_proj on the device while the
#: 6.4 GB of down_proj sources are already resident: bf16 and FP8 GLM-5.3
#: captures peak at 37.53 GB and the trellis one at 56.86 GB, so no <=48 GB
#: card can run them. Here the fused parameter is allocated ONCE and every
#: source is copied into its slice -- `gate_up[e, :I]`, `gate_up[e, I:]`,
#: `down[e]` -- through a small pinned staging ring (or straight from the
#: decoder's output tensor). A copy is a copy: `bin/selftest_layer_outer.py`
#: L4/L6 assert the result equals `from_pretrained`'s and the converter's own,
#: parameter by parameter. Every non-expert tensor still goes through the
#: converter, untouched.
EXPERT_FILL_REFERENCE = ("transformers.core_model_loading.MergeModulelist(dim=0) + "
                         "Concatenate(dim=1)")
_SAFETENSORS_DTYPE_NAMES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
_PATTERN_LITERAL_FORBIDDEN = set("[](){}+?^$|")


def _pattern_literal(pattern: str) -> Optional[str]:
    """A conversion pattern as the literal key fragment it matches, or None
    when the pattern carries regex syntax this fill does not model."""
    if any(ch in _PATTERN_LITERAL_FORBIDDEN for ch in pattern):
        return None
    return pattern.replace("\\.", ".")


class _StagingRing(object):
    """A few reusable host buffers a checkpoint slice is read into, then DMA'd out.

    On CUDA the buffers are page-locked and the copy is asynchronous, so the
    NVMe read of slice n+1 overlaps the bus transfer of slice n (the converter
    moves each of the 768 per-expert slices as a synchronous pageable copy,
    measured at 3.0 GB/s against the disk's 5.9 GB/s). A buffer is reused only
    after the event recorded behind its copy has completed. On any other
    device the same code runs with pageable buffers and synchronous copies:
    the bytes are the same bytes, only their timing differs.
    """

    def __init__(self, device, slots: int = 4):
        import torch

        self.device = torch.device(device)
        self.pinned = self.device.type == "cuda" and torch.cuda.is_available()
        self.buffers: List[Any] = [None] * int(slots)
        self.events: List[Any] = [None] * int(slots)
        self.next = 0
        self.bytes = 0
        self.reads = 0

    def copy_into(self, dst, path: str, offset: int, nbytes: int) -> None:
        import torch

        if int(dst.numel()) * dst.element_size() != nbytes:
            raise LayerOuterError(
                "REFUSED: %s bytes at %s+%d do not fill a %s %s slice"
                % (nbytes, os.path.basename(path), offset, tuple(dst.shape), dst.dtype))
        slot = self.next
        self.next = (slot + 1) % len(self.buffers)
        buffer = self.buffers[slot]
        if buffer is None or buffer.numel() < nbytes:
            buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.pinned)
            self.buffers[slot] = buffer
        if self.events[slot] is not None:
            self.events[slot].synchronize()
            self.events[slot] = None
        view = buffer[:nbytes]
        _read_exact(path, offset, view.numpy())
        dst.copy_(view.view(dst.dtype).view(dst.shape), non_blocking=self.pinned)
        if self.pinned:
            event = torch.cuda.Event()
            event.record()
            self.events[slot] = event
        self.bytes += nbytes
        self.reads += 1

    def drain(self) -> None:
        for index, event in enumerate(self.events):
            if event is not None:
                event.synchronize()
                self.events[index] = None


def _read_exact(path: str, offset: int, into) -> None:
    """Fill a writable buffer with exactly its length from `path` at `offset`."""
    view = memoryview(into).cast("B")
    fd = os.open(path, os.O_RDONLY)
    try:
        done = 0
        total = len(view)
        while done < total:
            if hasattr(os, "preadv"):
                got = os.preadv(fd, [view[done:]], offset + done)
            else:  # pragma: no cover - non-Linux hosts
                chunk = os.pread(fd, total - done, offset + done)
                got = len(chunk)
                view[done:done + got] = chunk
            if got <= 0:
                raise LayerOuterError(
                    "REFUSED: %s ended %d bytes short of a tensor the header declares "
                    "at offset %d" % (path, total - done, offset))
            done += got
    finally:
        os.close(fd)


class _ExpertFill(object):
    """One layer's fused expert parameters, allocated once and filled slice by slice.

    `plan` (from `plan_expert_fill`) maps every source key the converter
    would have stacked to its (parameter, expert, part) slot. `offer(key,
    value)` copies a decoded tensor or a lazy checkpoint slice into that slot
    and reports whether it took it; anything it declines goes to the
    converter exactly as before. `finish()` refuses a slot nobody filled --
    the converter would have reported a shape mismatch for the same
    checkpoint, and this schedule does not run a layer on undefined bytes.
    """

    def __init__(self, model, plan: Dict[str, Any], device: str, load_param,
                 locator: Dict[str, Tuple[str, int, int, str, Tuple[int, ...]]],
                 ring: Optional[_StagingRing], stats: Dict[str, Any]):
        import torch

        self.model = model
        self.plan = plan
        self.locator = locator
        self.ring = ring
        self.stats = stats
        self.slots: Dict[str, Tuple[str, int, int]] = {}
        self.filled: Set[str] = set()
        self.params: Dict[str, Any] = {}
        for name, target in plan["targets"].items():
            tensor = torch.empty(tuple(target["shape"]), dtype=getattr(torch, target["dtype"]),
                                 device=device)
            load_param(model, name, tensor)
            param = model.get_parameter(name)
            param._is_hf_initialized = True
            self.params[name] = param
            for key, (expert, part) in target["slots"].items():
                self.slots[key] = (name, expert, part)

    def _slice(self, name: str, expert: int, part: int):
        target = self.plan["targets"][name]
        param = self.params[name]
        if target["parts"] == 1:
            return param.data[expert]
        width = int(target["shape"][1]) // int(target["parts"])
        return param.data[expert, part * width:(part + 1) * width]

    def offer(self, key: str, value) -> bool:
        slot = self.slots.get(key)
        if slot is None:
            return False
        if key in self.filled:
            raise LayerOuterError(
                "REFUSED: %s was handed to the expert fill twice; the checkpoint or a "
                "decoder carries two versions of one expert slice" % key)
        name, expert, part = slot
        dst = self._slice(name, expert, part)
        located = self.locator.get(key) if not hasattr(value, "dtype") else None
        if located is not None:
            path, offset, nbytes, dtype_name, shape = located
            if tuple(shape) != tuple(dst.shape) \
                    or _SAFETENSORS_DTYPE_NAMES.get(dtype_name) != str(dst.dtype).replace("torch.", ""):
                raise LayerOuterError(
                    "REFUSED: %s is %s %s in the checkpoint but its fused slice is %s %s"
                    % (key, dtype_name, tuple(shape), dst.dtype, tuple(dst.shape)))
            self.ring.copy_into(dst, path, offset, nbytes)
            self.stats["staged_slices"] += 1
        else:
            tensor = _eager(value)
            if tuple(tensor.shape) != tuple(dst.shape) or tensor.dtype != dst.dtype:
                raise LayerOuterError(
                    "REFUSED: %s decoded as %s %s but its fused slice is %s %s"
                    % (key, tensor.dtype, tuple(tensor.shape), dst.dtype, tuple(dst.shape)))
            dst.copy_(tensor)
            self.stats["decoded_slices"] += 1
        self.filled.add(key)
        self.stats["slices_filled"] += 1
        self.stats["bytes_filled"] += int(dst.numel()) * dst.element_size()
        return True

    def finish(self) -> None:
        if self.ring is not None:
            self.ring.drain()
        unfilled = sorted(key for key in self.slots if key not in self.filled)
        if unfilled:
            raise LayerOuterError(
                "REFUSED: %d expert slice(s) of %s were never delivered: %s%s. The "
                "checkpoint (or its decoder) did not produce them and the fused "
                "parameter would run on undefined bytes."
                % (len(unfilled), ", ".join(sorted(self.plan["targets"])), ", ".join(unfilled[:6]),
                   " (+%d more)" % (len(unfilled) - 6) if len(unfilled) > 6 else ""))
        self.stats["targets_filled"] += len(self.plan["targets"])
        self.stats["layers_filled"] += 1


def plan_expert_fill(model, converters: Sequence[Any], layer_names: Sequence[str],
                     routing_key: Callable[[str], str], subset_keys: Iterable[str],
                     source_dtype: Callable[[str], Optional[str]],
                     source_shape: Callable[[str], Optional[Tuple[int, ...]]],
                     dtype_plan: Dict[str, Any], torch_dtype,
                     decoders_active: bool) -> Dict[str, Any]:
    """Which of a layer's parameters the direct fill may build, and from which keys.

    Eligibility is derived from the model's OWN conversion mapping, never from
    a name convention: a `WeightConverter` whose operations are exactly
    `[MergeModulelist(dim=0)]` or `[MergeModulelist(dim=0), Concatenate(dim=1)]`
    with one `*` per source pattern. For each such converter and each layer
    parameter ending in its target, the E x parts candidate source keys are
    constructed and each one is ROUND-TRIPPED through the same rename the
    converter applies (`routing_key(candidate) == parameter`); a candidate
    that does not route back, a subset key that routes to the parameter but
    is not a candidate, a source whose stored dtype or shape is not the
    slice's, a dtype-plan override this fill cannot evaluate, or (with no
    decoder active) a candidate absent from the checkpoint, all make the
    parameter INELIGIBLE -- it then goes through the converter exactly as
    before, and the converter reports whatever it reports. Everything about
    `declined` is in the returned plan so the receipt can say what was filled
    directly and what was not.
    """
    import torch

    try:
        from transformers.core_model_loading import (Concatenate, MergeModulelist,
                                                     build_glob_alternation)
    except Exception as exc:  # pragma: no cover - depends on the build
        return {"targets": {}, "declined": {"*": "transformers internals: %s" % exc}}

    plan_alternation = None
    if dtype_plan:
        alternation, by_group, _ = build_glob_alternation(list(dtype_plan.keys()))
        plan_alternation = (alternation, by_group)
    subset_keys = list(subset_keys)
    key_set = set(subset_keys)
    routed = {}
    for key in subset_keys:
        routed.setdefault(routing_key(key), []).append(key)
    targets: Dict[str, Any] = {}
    declined: Dict[str, str] = {}
    for converter in converters:
        operations = list(getattr(converter, "operations", None) or [])
        if not operations or not isinstance(operations[0], MergeModulelist) \
                or getattr(operations[0], "dim", None) != 0:
            continue
        if len(operations) == 2:
            if not isinstance(operations[1], Concatenate) or operations[1].dim != 1:
                continue
        elif len(operations) != 1:
            continue
        if len(converter.target_patterns) != 1:
            continue
        target_literal = _pattern_literal(converter.target_patterns[0])
        source_literals = [_pattern_literal(p) for p in converter.source_patterns]
        if target_literal is None or any(s is None or s.count("*") != 1 for s in source_literals):
            continue
        for name in layer_names:
            if not name.endswith(target_literal):
                continue
            prefix = name[:-len(target_literal)]
            try:
                param = model.get_parameter(name)
            except AttributeError:
                continue
            shape = tuple(int(d) for d in param.shape)
            parts = len(source_literals)
            if len(shape) < 2 or (parts > 1 and (len(shape) < 3 or shape[1] % parts)):
                declined[name] = "shape %s does not split into %d parts" % (shape, parts)
                continue
            experts = shape[0]
            if parts == 1:
                slice_shape = shape[1:]
            else:
                slice_shape = (shape[1] // parts,) + shape[2:]
            target_dtype = str(param.dtype).replace("torch.", "")
            if plan_alternation is not None:
                matched = plan_alternation[0].search(name)
                if matched is not None:
                    planned = dtype_plan[plan_alternation[1][matched.lastgroup]]
                    target_dtype = str(planned).replace("torch.", "")
            slots: Dict[str, Tuple[int, int]] = {}
            reason = None
            for part, literal in enumerate(source_literals):
                for expert in range(experts):
                    candidate = prefix + literal.replace("*", str(expert))
                    if routing_key(candidate) != name:
                        reason = "%s does not route back to %s" % (candidate, name)
                        break
                    slots[candidate] = (expert, part)
                if reason:
                    break
            if reason is None:
                present = routed.get(name, [])
                strangers = [key for key in present if key not in slots]
                if strangers:
                    reason = "%s routes here but is not an expert slice" % strangers[0]
                elif not decoders_active and len(present) != len(slots):
                    reason = "%d of %d expert slices present in the checkpoint" % (
                        len(present), len(slots))
            if reason is None:
                for key in routed.get(name, []):
                    if decoders_active and (key + FP8_SCALE_SUFFIX) in key_set:
                        # A block-scaled FP8 pair: the decoder replaces it by a
                        # `torch_dtype` tensor of the same name, checked for
                        # dtype and shape when it is offered to the slot.
                        continue
                    dtype_name = source_dtype(key)
                    stored = source_shape(key)
                    if _SAFETENSORS_DTYPE_NAMES.get(dtype_name or "") != target_dtype:
                        reason = "%s is stored as %s, the parameter is %s" % (
                            key, dtype_name, target_dtype)
                        break
                    if stored is not None and tuple(stored) != tuple(slice_shape):
                        reason = "%s is stored as %s, the slice is %s" % (
                            key, tuple(stored), tuple(slice_shape))
                        break
            if reason is not None:
                declined[name] = reason
                continue
            targets[name] = {"shape": shape, "dtype": target_dtype, "parts": parts,
                             "experts": experts, "slots": slots,
                             "sources": [prefix + s for s in source_literals]}
    return {"targets": targets, "declined": declined}


def expert_fill_evidence(streamer: StreamedModel) -> Optional[Dict[str, Any]]:
    """How the routed experts reached the device, for the runtime receipt."""
    stats = getattr(streamer, "expert_fill_stats", None)
    if stats is None:
        return None
    return {
        "mode": stats.get("mode"),
        "reference": EXPERT_FILL_REFERENCE,
        "layers_filled": int(stats.get("layers_filled", 0)),
        "targets_filled": int(stats.get("targets_filled", 0)),
        "slices_filled": int(stats.get("slices_filled", 0)),
        "staged_slices": int(stats.get("staged_slices", 0)),
        "decoded_slices": int(stats.get("decoded_slices", 0)),
        "bytes_filled": int(stats.get("bytes_filled", 0)),
        "staging": stats.get("staging"),
        "declined": dict(stats.get("declined") or {}),
        "identity": "a byte copy into the fused parameter's slice; bin/selftest_layer_outer.py "
                    "L4/L6 assert equality with from_pretrained and with the converter path",
    }


def is_trellis_checkpoint(config) -> bool:
    """One answer to "is this an EXL3 trellis artifact?" for EVERY gate.

    Two declarations count: an inline `quantization_config.quant_method: exl3`
    (turboderp layout; drowzeys, wrldsuksgo2mars) and a top-level
    `hybrid_tr3_tail` (davidsyoung, whose `quantization_config` is a leftover
    ModelOpt block). The FP8 gate and the trellis gate MUST consult the same
    predicate: on 2026-09-05 they did not, the FP8 gate saw only
    `quant_method: modelopt` and refused three davidsyoung pods after their
    fetch, while the trellis gate one line below would have accepted them.
    """
    return _quant_method(config) == "exl3" or trellis_tail_declaration(config) is not None


def checkpoint_decode_plans(config, model_dir: str, log: Callable[..., None]):
    """Resolve which host-side decoders this checkpoint needs, before anything is built.

    Returns `(fp8_plan, trellis_plan, trellis_fp8_plan, trellis_stats, nvfp4_plan,
    gguf_plan)`. Exactly one of five shapes is admitted: a native tree (all None), the
    block-scaled FP8 e4m3 weights-only form (`fp8_plan`), an EXL3 trellis
    artifact (`trellis_plan`, with `trellis_fp8_plan` when the checkpoint ALSO
    keeps tensors in block-scaled FP8 -- wrldsuksgo2mars keeps shared_experts
    and self_attn that way), a modelopt NVFP4 artifact (`nvfp4_plan`), or a
    llama.cpp GGUF build (`gguf_plan`: decided by the presence of .gguf files,
    since a GGUF tree carries no config of its own -- the config.json beside it
    is the official release's, copied there by the candidate stage). Any
    other `quantization_config` is refused here by `fp8_checkpoint_plan`,
    which is only consulted when the artifact is NEITHER a trellis nor a
    modelopt one: a trellis artifact's `quantization_config` may be a
    leftover that describes nothing in the checkpoint (see
    `trellis_tail_declaration`), and a modelopt block is judged by
    `nvfp4_checkpoint_plan`, which refuses every modelopt form but NVFP4 by
    name.

    The index is read ONLY for a trellis or modelopt artifact: a bf16 or FP8
    checkpoint must not acquire a dependency on an index file it may not have
    (a single-shard tree has none, and under a race-mode gate it has not
    landed yet).
    """
    gguf_plan = gguf_checkpoint_plan(config, model_dir)
    if gguf_plan is not None:
        observed = gguf_plan.pop("_observed", {})
        log(stage="gguf_decode_plan", method=GGUF_DECODE_METHOD,
            reference=GGUF_DECODE_REFERENCE, parity=GGUF_PARITY_EVIDENCE,
            observed={k: v for k, v in observed.items() if k != "file_records"},
            **{k: v for k, v in gguf_plan.items() if not k.startswith("_")})
        gguf_plan["_observed"] = observed
        trellis_stats = {"decoded_modules": 0, "trellis_bits": 0}
        return None, None, None, trellis_stats, None, gguf_plan
    trellis = is_trellis_checkpoint(config)
    nvfp4_plan = None if trellis else nvfp4_checkpoint_plan(config, model_dir)
    if nvfp4_plan is not None:
        observed = nvfp4_plan.pop("_observed", {})
        log(stage="nvfp4_decode_plan", method=NVFP4_DECODE_METHOD,
            reference=NVFP4_DECODE_REFERENCE, parity=NVFP4_PARITY_EVIDENCE,
            observed=observed,
            **{k: v for k, v in nvfp4_plan.items() if not k.startswith("_")})
        nvfp4_plan["_observed"] = observed
    fp8_plan = None if (trellis or nvfp4_plan is not None) else fp8_checkpoint_plan(config)
    if fp8_plan is not None:
        log(stage="fp8_decode_plan", method=FP8_DECODE_METHOD,
            reference=FP8_DECODE_REFERENCE, **fp8_plan)
    trellis_plan = None
    trellis_stats: Dict[str, Any] = {"decoded_modules": 0, "trellis_bits": 0}
    trellis_fp8_plan = None
    if trellis:
        # The trellis decode is two fp32 128x128 GEMMs per module and runs on
        # the capture device: pin the matmul policy exactly as stream_score
        # does and RECORD it, so the sealed receipt can show that TF32 was off
        # rather than assume torch's default (which NVIDIA_TF32_OVERRIDE can
        # flip from the environment without a trace).
        trellis_stats["numeric_policy"] = _pin_fp32_matmul_policy()
        if str(os.environ.get("NVIDIA_TF32_OVERRIDE", "")).strip() == "1":
            raise LayerOuterError(
                "REFUSED: NVIDIA_TF32_OVERRIDE=1 forces TF32 in cuBLAS regardless of the "
                "torch flags; the trellis decode's fp32 GEMMs would not be fp32. Unset it.")
        keys = list(_index_weight_map(model_dir))
        trellis_plan = trellis_checkpoint_plan(config, keys, model_dir=model_dir)
        if any(key.endswith(FP8_SCALE_SUFFIX) for key in keys):
            trellis_fp8_plan = fp8_checkpoint_plan_for_mixed(config)
    if trellis_plan is not None:
        observed = trellis_plan.pop("_observed", {})
        trellis_stats["composition"] = observed.get("composition")
        trellis_stats["quant_method_declared"] = observed.get("quant_method_declared")
        trellis_stats["declared_by"] = observed.get("declared_by")
        if observed.get("rotation_layout") == "r7_shared":
            trellis_stats["r7_permutations"] = r7_permutation_source(
                config, model_dir, observed.get("r7_declaration") or {})
        trellis_stats["rotation_layout"] = observed.get("rotation_layout")
        trellis_stats["modules_per_layout_planned"] = observed.get("modules_per_layout")
        trellis_stats["r7_declaration"] = observed.get("r7_declaration")
        trellis_stats["module_bits_policy"] = observed.get("module_bits_policy")
        log(stage="trellis_decode_plan",
            method=(TRELLIS_TP_COMPOSE_METHOD if observed.get("composition")
                    else TRELLIS_DECODE_METHOD),
            reference=TRELLIS_DECODE_REFERENCE,
            mixed_fp8=trellis_fp8_plan is not None, observed=observed,
            **trellis_plan)
        trellis_stats["quantized_module_count"] = observed.get(
            "quantized_module_count", 0)
        trellis_stats["codebook_histogram"] = observed.get("codebook_histogram", {})
    return fp8_plan, trellis_plan, trellis_fp8_plan, trellis_stats, nvfp4_plan, None


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
                         gate: Optional[Any] = None,
                         expert_fill: str = EXPERT_FILL_DIRECT) -> StreamedModel:
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

    gguf_tree = gate is None and bool(gguf_files_in(model_dir))
    if gate is None and not gguf_tree:
        audit = audit_checkpoint_tree(model_dir)
        log(stage="checkpoint_audit", **audit)
    # A GGUF tree has no safetensors to audit; `gguf_checkpoint_plan` audits
    # the container's own extents (and the whole-file digests) below.
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
    fp8_plan, trellis_plan, trellis_fp8_plan, trellis_stats, nvfp4_plan, gguf_plan = (
        checkpoint_decode_plans(config, model_dir, log))
    if gguf_plan is not None and gate is not None:
        raise LayerOuterError("REFUSED: the GGUF lane has no gated (race-mode) loader")
    if gguf_plan is not None:
        log(stage="checkpoint_audit", **(gguf_plan.get("_observed") or {}).get("container_audit", {}))
    fp8_stats = {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0}
    gguf_stats: Dict[str, Any] = {"tensors_decoded": 0, "official_tensors_produced": 0,
                                  "gguf_bytes_read": 0, "ggml_types": {}, "layers_decoded": 0}
    nvfp4_stats: Dict[str, Any] = {"decoded_modules": 0, "packed_bytes": 0,
                                   "scales_consumed": 0, "input_scales_skipped": 0,
                                   "plain_modules_passed": 0}

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
    # Where each key's bytes live: (shard path, absolute offset, byte length,
    # stored dtype, shape) from the shard's own header. The direct expert fill
    # reads a slice from here straight into its staging buffer -- the same
    # bytes `PySafeSlice[...]` would hand back, without the intermediate copy.
    locator: Dict[str, Tuple[str, int, int, str, Tuple[int, ...]]] = {}

    def _open_shards(names: Sequence[str]) -> Dict[str, Any]:
        """Open shards not yet open; return {key: lazy slice} for the NEW ones only."""
        fresh: Dict[str, Any] = {}
        for name in sorted(names):
            if name in opened_shards:
                continue
            path = os.path.join(model_dir, name)
            pointer = safe_open(path, framework="pt", device="cpu")
            opened_shards[name] = pointer
            pointers.append(pointer)
            header, _ = _safetensors_header(path)
            with open(path, "rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
            for key in pointer.keys():
                fresh[key] = pointer.get_slice(key)
                entry = header[key]
                start, stop = (int(v) for v in entry["data_offsets"])
                locator[key] = (path, 8 + header_len + start, stop - start,
                                str(entry["dtype"]), tuple(int(d) for d in entry["shape"]))
        return fresh

    if expert_fill not in EXPERT_FILL_MODES:
        raise LayerOuterError("REFUSED: unknown expert_fill mode %r (one of %s)"
                              % (expert_fill, ", ".join(EXPERT_FILL_MODES)))
    ring = _StagingRing(device)
    expert_fill_stats: Dict[str, Any] = {
        "mode": expert_fill, "layers_filled": 0, "targets_filled": 0,
        "slices_filled": 0, "staged_slices": 0, "decoded_slices": 0, "bytes_filled": 0,
        "staging": "pinned" if ring.pinned else "pageable", "declined": {}}
    timing: Dict[str, Any] = {"layers_loaded": 0, "load_seconds": 0.0,
                              "decode_seconds": 0.0, "fill_seconds": 0.0,
                              "checkpoint_bytes_read": 0, "converter_bytes": 0}

    # A GGUF slot has no stored bytes the fill could read; what the decoder
    # hands it is a `torch_dtype` tensor of the slice shape, and saying so here
    # is what lets `plan_expert_fill` admit the routed experts to the direct fill.
    gguf_slot_keys: Set[str] = set()
    gguf_decoded_dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}[dtype_name]

    def _stored_dtype(key: str) -> Optional[str]:
        located = locator.get(key)
        if located is None and key in gguf_slot_keys:
            return gguf_decoded_dtype
        return located[3] if located is not None else None

    def _stored_shape(key: str) -> Optional[Tuple[int, ...]]:
        located = locator.get(key)
        return located[4] if located is not None else None

    def _subset_bytes(keys: Iterable[str]) -> int:
        return sum(locator[key][2] for key in keys if key in locator)

    ungated_shard_names = (sorted(name for name in os.listdir(model_dir)
                                  if name.endswith(".safetensors"))
                           if gate is None and gguf_plan is None else [])

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
        if gguf_plan is not None:
            # the container's official names stand where the shard keys stand
            slots = gguf_subsets(gguf_plan)
            gguf_slot_keys.update(slots)
            bucket(slots)
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

    resident_started = time.monotonic()
    base_info, _ = convert_and_load(
        model,
        _materialized(base_subset, fp8_plan, trellis_plan, trellis_fp8_plan,
                      torch_dtype, fp8_stats, trellis_stats, device=device,
                      expected_shape=expected_shape,
                      nvfp4_plan=nvfp4_plan, nvfp4_stats=nvfp4_stats,
                      fp8_parity_all=str(device) != "cpu",
                      gguf_plan=gguf_plan, gguf_stats=gguf_stats),
        load_config)
    _absorb(base_info)
    timing["resident_load_seconds"] = time.monotonic() - resident_started
    timing["resident_bytes"] = _subset_bytes(base_subset)
    timing["checkpoint_bytes_read"] += timing["resident_bytes"]

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
        load_started = time.monotonic()
        decoders_active = (fp8_plan is not None or trellis_plan is not None
                           or nvfp4_plan is not None or gguf_plan is not None)
        # THE DIRECT EXPERT FILL (see EXPERT_FILL_REFERENCE). Planned per layer
        # from the model's own conversion mapping; the fused parameters are
        # allocated here, once, and every routed-expert source -- a checkpoint
        # slice or a decoder's output -- is copied into its slice as it
        # appears. What the plan declines, and everything that is not a routed
        # expert, reaches the converter exactly as before.
        fill = None
        if expert_fill == EXPERT_FILL_DIRECT:
            fill_plan = plan_expert_fill(
                model, renames, layer_param_names(index), routing_key, subset,
                _stored_dtype, _stored_shape, dtype_plan, torch_dtype, decoders_active)
            expert_fill_stats["declined"].update(fill_plan["declined"])
            if fill_plan["targets"]:
                fill = _ExpertFill(model, fill_plan, device, load_param_into_model,
                                   locator, ring, expert_fill_stats)
        sink = fill.offer if fill is not None else None
        timing["checkpoint_bytes_read"] += _subset_bytes(subset)
        decode_log = None
        if decoders_active:
            # Decoded per layer into a transient dict: the streamer keeps the
            # lazy slices, never the 19 GB of decoded bf16, across layers --
            # and with a fill in place the decoded experts are not even held
            # for the layer: the sink copies each one into its slice and drops it.
            before = dict(fp8_stats)
            before_trellis = dict(trellis_stats)
            before_nvfp4 = dict(nvfp4_stats)
            before_gguf = dict(gguf_stats)
            # S1-2's gate: the FIRST layers decoded on a device re-decode every
            # FP8 tensor on the host and must agree bitwise, until one layer
            # that carries routed experts has passed; partial-block tensors
            # are checked on every layer (materialize_fp8_subset).
            parity_all = (str(device) != "cpu"
                          and not fp8_stats.get("device_parity_sparse_layer_done"))
            started = time.monotonic()
            decoded = _materialized(subset, fp8_plan, trellis_plan, trellis_fp8_plan,
                                    torch_dtype, fp8_stats, trellis_stats,
                                    device=device, expected_shape=expected_shape,
                                    nvfp4_plan=nvfp4_plan, nvfp4_stats=nvfp4_stats,
                                    fp8_parity_all=parity_all, sink=sink,
                                    gguf_plan=gguf_plan, gguf_stats=gguf_stats)
            decode_seconds = time.monotonic() - started
            timing["decode_seconds"] += decode_seconds
            if parity_all and fp8_stats["dequantized"] > before["dequantized"]:
                fp8_stats.setdefault("device_parity_full_layers", []).append(index)
                if fill is not None or expert_fill != EXPERT_FILL_DIRECT:
                    fp8_stats["device_parity_sparse_layer_done"] = True
            decode_log = dict(
                stage=("trellis_decode_layer" if trellis_plan is not None
                       else "nvfp4_decode_layer" if nvfp4_plan is not None
                       else "gguf_decode_layer" if gguf_plan is not None
                       else "fp8_decode_layer"), index=index,
                dequantized=fp8_stats["dequantized"] - before["dequantized"],
                fp8_elements=fp8_stats["fp8_bytes"] - before["fp8_bytes"],
                decoded_modules=((trellis_stats["decoded_modules"]
                                  - before_trellis["decoded_modules"])
                                 + (nvfp4_stats["decoded_modules"]
                                    - before_nvfp4["decoded_modules"])
                                 + (gguf_stats["tensors_decoded"]
                                    - before_gguf["tensors_decoded"])) or None,
                decode_seconds=round(decode_seconds, 3))
            if gguf_plan is not None:
                timing["checkpoint_bytes_read"] += (gguf_stats["gguf_bytes_read"]
                                                    - before_gguf["gguf_bytes_read"])
            loader_subset = decoded
            # The dict is the last holder of any decoded tensor the fill did
            # not consume; the fill's `remaining` takes those over below.
            del decoded
        else:
            loader_subset = subset
        if fill is not None:
            fill_started = time.monotonic()
            remaining: Dict[str, Any] = {}
            for key, value in loader_subset.items():
                if not fill.offer(key, value):
                    remaining[key] = value
            fill.finish()
            loader_subset = remaining
            timing["fill_seconds"] += time.monotonic() - fill_started
        timing["converter_bytes"] += _subset_bytes(loader_subset)
        info, _ = convert_and_load(model, loader_subset, load_config)
        del loader_subset
        if decode_log is not None:
            log(**decode_log)
        _absorb(info)
        timing["layers_loaded"] += 1
        timing["load_seconds"] += time.monotonic() - load_started
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
    streamer.expert_fill_stats = expert_fill_stats
    streamer.timing = timing
    streamer.fp8_plan = fp8_plan
    streamer.fp8_stats = fp8_stats
    streamer.trellis_plan = trellis_plan
    streamer.trellis_stats = trellis_stats
    streamer.trellis_fp8_plan = trellis_fp8_plan
    streamer.nvfp4_plan = nvfp4_plan
    streamer.nvfp4_stats = nvfp4_stats
    streamer.gguf_plan = gguf_plan
    streamer.gguf_stats = gguf_stats
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
        # The ROTATION LAYOUT the decoder resolved (contract keys ride inside
        # quantization_config; this block is the census beside them).
        layout = stats.get("rotation_layout") or trellis_plan.get("rotation_layout")
        evidence["rotation_layout"] = {
            "layout": layout,
            "reader": TRELLIS_LAYOUT_READERS.get(layout),
            "modules_per_layout": dict(sorted((stats.get("modules_per_layout") or {}).items())),
            "shared_vectors_declared": int(
                (trellis_plan.get("shared_vectors") or {}).get("count") or 0),
            "shared_vectors_applied": int(stats.get("shared_vectors_applied", 0)),
            "nonrouted_exl3_modules": [
                {"name": name, "K": int(bits)}
                for name, bits in sorted((stats.get("nonrouted_exl3_decoded") or {}).items())],
            "r7_declaration": stats.get("r7_declaration"),
            "evidence": TRELLIS_LAYOUT_EVIDENCE,
        }
        permutations = stats.get("r7_permutations")
        if isinstance(permutations, R7PermutationSource):
            evidence["rotation_layout"]["r7_intermediate_unpermute"] = dict(
                permutations.stats, intermediate=permutations.intermediate,
                manifests_declared=len(permutations.by_layer),
                reference=R7_UNPERMUTE_REFERENCE)
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
            evidence["mixed_fp8"].update(fp8_device_parity_evidence(fp8_stats))
        return evidence
    nvfp4_plan = getattr(streamer, "nvfp4_plan", None)
    if nvfp4_plan is not None:
        stats = dict(getattr(streamer, "nvfp4_stats", {}))
        observed = dict(nvfp4_plan.get("_observed") or {})
        evidence = {
            "method": NVFP4_DECODE_METHOD,
            "reference": NVFP4_DECODE_REFERENCE,
            "parity_evidence": NVFP4_PARITY_EVIDENCE,
            "output_dtype": "bfloat16",
            # The CONTRACT block only: the census and geometry ride under
            # private keys and are reported beside it, never inside it.
            "quantization_config": {k: v for k, v in nvfp4_plan.items()
                                    if not k.startswith("_")},
            "modules_decoded": int(stats.get("decoded_modules", 0)),
            "packed_elements": int(stats.get("packed_bytes", 0)),
            "scale_tensors_consumed": int(stats.get("scales_consumed", 0)),
            "input_scale_tensors_not_applied": int(stats.get("input_scales_skipped", 0)),
            "plain_bf16_modules_passed_through": int(stats.get("plain_modules_passed", 0)),
            "decode_device": "capture-device",
            "observed": observed,
        }
        # Every non-routed tensor by stored dtype. The official BF16 release
        # keeps the 75 router correction biases in fp32; a producer that
        # rounded them to bf16 shows up here as the ABSENCE of float32, and
        # the scope file authored from the shard headers says the same.
        evidence["nonrouted_by_dtype"] = dict(sorted(
            (stats.get("nonrouted_by_dtype") or {}).items()))
        evidence["nonrouted_non_bf16_examples"] = list(
            stats.get("nonrouted_non_bf16_examples") or [])
        return evidence
    gguf_plan = getattr(streamer, "gguf_plan", None)
    if gguf_plan is not None:
        stats = dict(getattr(streamer, "gguf_stats", {}))
        observed = dict(gguf_plan.get("_observed") or {})
        return {
            "method": GGUF_DECODE_METHOD,
            "reference": GGUF_DECODE_REFERENCE,
            "parity_evidence": GGUF_PARITY_EVIDENCE,
            "output_dtype": "bfloat16",
            # The CONTRACT block only (header-derived, mirrored by the controller);
            # the surface object and partition ride under private keys.
            "quantization_config": {k: v for k, v in gguf_plan.items()
                                    if not k.startswith("_")},
            "tensors_decoded": int(stats.get("tensors_decoded", 0)),
            "official_tensors_produced": int(stats.get("official_tensors_produced", 0)),
            "layers_decoded": int(stats.get("layers_decoded", 0)),
            "gguf_bytes_read": int(stats.get("gguf_bytes_read", 0)),
            "ggml_types_decoded": dict(sorted((stats.get("ggml_types") or {}).items())),
            "decode_device": "capture-device",
            "files_verified": observed.get("files_verified"),
            "file_records": observed.get("file_records"),
            "container_audit": observed.get("container_audit"),
            "shared_indexer_copies": observed.get("shared_indexer_copies"),
            "materialize_plan": observed.get("materialize_plan"),
            "checkpoint_identity_sha256": observed.get("checkpoint_identity_sha256"),
            "tokenizer_source": ("the reference root's tokenizer files; the GGUF's own "
                                 "tokenizer.ggml.tokens/merges were proven equal to that "
                                 "vocabulary by id (gguf_surface.tokenizer_matches, at plan "
                                 "time and in gguf-evidence/glmdsa-tokenizer-order-audit.json)"),
            "head_source": "the artifact's own output.weight, decoded (HEAD-1d own heads)",
        }
    plan = getattr(streamer, "fp8_plan", None)
    if plan is None:
        return None
    stats = dict(getattr(streamer, "fp8_stats", {}))
    evidence = {
        "method": FP8_DECODE_METHOD,
        "reference": FP8_DECODE_REFERENCE,
        "output_dtype": "bfloat16",
        "quantization_config": plan,
        "tensors_dequantized": int(stats.get("dequantized", 0)),
        "scale_tensors_consumed": int(stats.get("scales_consumed", 0)),
        "fp8_elements": int(stats.get("fp8_bytes", 0)),
    }
    evidence.update(fp8_device_parity_evidence(stats))
    return evidence


def head_decode_identity(streamer: StreamedModel) -> Optional[Dict[str, Any]]:
    """How the model's `lm_head.weight` came to be, for the sealed head identity.

    None when the head was loaded as shipped (native). When the trellis
    decoder produced it from an exl3 payload group (jpsequeira's 8-bit
    `lm_head.{trellis,suh,svh,mcg}`), the head hf_capture seals is the
    candidate's OWN dequantized head: `quantized: true`, its payload K as
    `bits`, `source: artifact_dequantized` (docs/FIDELITY-DATASET-SPEC.md,
    head source table) -- and the comparison runs it under HEAD-1d, own
    heads, so the head's quantization error is inside the number.
    """
    if getattr(streamer, "trellis_plan", None) is None:
        return None
    decoded = dict(getattr(streamer, "trellis_stats", {}) or {}).get("nonrouted_exl3_decoded") or {}
    bits = decoded.get("lm_head")
    if bits is None:
        return None
    return {"quantized": True, "bits": int(bits), "source": "artifact_dequantized",
            "method": TRELLIS_DECODE_METHOD, "reference": TRELLIS_DECODE_REFERENCE}


def fp8_device_parity_evidence(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Where the FP8 decode ran and how much of it was re-derived on the host.

    `fp8_decode_device` is the device the bytes were dequantised on;
    `fp8_device_parity_checked_tensors` counts the tensors ALSO decoded on the
    CPU and asserted `torch.equal` (every tensor of the first layers, every
    partial-block tensor of every layer), `..._partial_block` the subset of
    those with a partial last block, `..._full_layers` the layer indices
    checked in full. `fp8_device_parity` is "not-applicable" on a CPU run,
    "passed" once anything was checked -- a failure never reaches a receipt,
    it refuses the run.
    """
    device = stats.get("decode_device")
    checked = int(stats.get("device_parity_checked", 0))
    return {
        "fp8_decode_device": device,
        "fp8_device_parity_checked_tensors": checked,
        "fp8_device_parity_checked_partial_block": int(
            stats.get("device_parity_partial_block_checked", 0)),
        "fp8_device_parity_full_layers": list(stats.get("device_parity_full_layers") or []),
        "fp8_device_parity": ("not-applicable" if device in (None, "cpu")
                              else "passed" if checked else "unchecked"),
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
