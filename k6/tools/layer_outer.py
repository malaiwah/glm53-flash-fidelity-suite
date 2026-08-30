#!/usr/bin/env python3
"""The layer-outer, window-inner capture schedule -- and the streaming residency it needs.

Why this file exists
--------------------
`k6/tools/hf_capture.py` captures a panel the obvious way: load the model, then
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def audit_checkpoint_tree(model_dir: str) -> Dict[str, Any]:
    """Refuse a checkpoint whose shards can hand back holes instead of weights.

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
    else:
        shard_names = sorted(name for name in os.listdir(model_dir)
                             if name.endswith(".safetensors"))
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


def _model_device(device: str):
    import torch

    return torch.device(device)


def build_streamed_model(model_dir: str, cls, config, dtype_name: str, device: str,
                         log: Callable[..., None],
                         layer_guard: Optional[Callable[[int, Dict[str, Any]], None]] = None
                         ) -> StreamedModel:
    """Instantiate on meta, load everything but the decoder layers, and return the streamer."""
    import copy as _copy

    import torch
    from transformers.utils.generic import ContextManagers

    (convert_and_load, LoadStateDictConfig, load_param_into_model,
     patch_output_recorders, get_conversion_mapping) = _require_transformers_internals()

    from safetensors import safe_open

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype_name]

    audit = audit_checkpoint_tree(model_dir)
    log(stage="checkpoint_audit", **audit)

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
    shard_paths = sorted(os.path.join(model_dir, name) for name in os.listdir(model_dir)
                         if name.endswith(".safetensors"))
    pointers = [safe_open(path, framework="pt", device="cpu") for path in shard_paths]
    lazy: Dict[str, Any] = {}
    for pointer in pointers:
        for key in pointer.keys():
            lazy[key] = pointer.get_slice(key)

    base_subset: Dict[str, Any] = {}
    layer_subset: Dict[int, Dict[str, Any]] = {}
    for key, value in lazy.items():
        match = layer_pattern.match(key)
        if match is None or key in buffer_names:
            base_subset[key] = value
        else:
            layer_subset.setdefault(int(match.group(1)), {})[key] = value

    aggregate = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": [],
                 "error_msgs": [], "conversion_errors": {}}

    # Checkpoint tensors addressed to a layer index the model does not build are
    # never handed to the loader by this schedule, so they would never appear in
    # any per-load `unexpected_keys` and the `checkpoint_tensors_not_loaded`
    # disclosure the window-outer path emits would silently go missing.  This is
    # not hypothetical: GLM-5.3's MTP layer 78 (791 tensors, 18.5 GiB) and
    # Fruit's layer 13 are exactly this case -- `transformers` builds
    # `num_hidden_layers` layers and drops the next-token-prediction layer.
    for index in sorted(layer_subset):
        if index >= len(layers):
            aggregate["unexpected_keys"] |= set(layer_subset[index])

    def _absorb(info) -> None:
        aggregate["unexpected_keys"] |= set(info.unexpected_keys or set())
        for entry in (info.mismatched_keys or []):
            aggregate["mismatched_keys"].append(entry)
        aggregate["error_msgs"].extend(list(info.error_msgs or []))
        aggregate["conversion_errors"].update(dict(info.conversion_errors or {}))

    base_info, _ = convert_and_load(model, base_subset, load_config)
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
                              if key not in streamed_params}
    cls._finalize_model_loading(model, load_config, base_info)
    aggregate["missing_keys"] |= set(base_info.missing_keys or set())

    stranded = [name for name, tensor in
                list(model.named_parameters()) + list(model.named_buffers())
                if tensor.device.type == "meta" and name not in streamed_params]
    if stranded:
        raise LayerOuterError(
            "REFUSED: %d parameter(s)/buffer(s) outside the streamed decoder layers are "
            "still on the meta device after the resident load, so a forward pass would "
            "read them as undefined: %s%s"
            % (len(stranded), ", ".join(stranded[:6]),
               " (+%d more)" % (len(stranded) - 6) if len(stranded) > 6 else ""))

    log(stage="stream_base", resident_tensors=len(base_subset),
        streamed_layers=len(layers), streamed_params=len(streamed_params),
        layers_prefix=layers_prefix)

    def layer_param_names(index: int) -> List[str]:
        head = "%s.%d." % (layers_prefix, index)
        return sorted(name for name in streamed_params if name.startswith(head))

    def do_load(index: int) -> None:
        subset = layer_subset.get(index)
        if not subset:
            raise LayerOuterError(
                "REFUSED: the checkpoint holds no tensors for %s.%d. A layer with no "
                "weights does not fail to run -- it runs on whatever the meta-device "
                "placeholder is replaced by, which is nothing anybody measured."
                % (layers_prefix, index))
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
    streamer.pointers = pointers
    streamer.layer_counts = {index: len(subset) for index, subset in layer_subset.items()}
    return streamer


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
