#!/usr/bin/env python3
"""Byte-for-byte: do a loaded MoE model's FUSED expert rows equal the shard bytes?

The risk this closes (docs/GLM53-ROOT-FEASIBILITY.md R1)
--------------------------------------------------------
`GlmMoeDsaForCausalLM` -- and every model `transformers` routes through the
`qwen2_moe` conversion pattern -- holds routed experts as fused 3-D parameters:

    mlp.experts.gate_up_proj   [E, 2*I, H]
    mlp.experts.down_proj      [E,   H, I]

while the checkpoint ships one 2-D matrix per expert per projection.  For
`zai-org/GLM-5.3-BF16` that is 57,600 of 59,585 tensors -- 96.7% of the
checkpoint -- collapsed by

    WeightConverter(["mlp.experts.*.gate_proj.weight",
                     "mlp.experts.*.up_proj.weight"],
                    "mlp.experts.gate_up_proj",
                    [MergeModulelist(dim=0), Concatenate(dim=1)])
    WeightConverter("mlp.experts.*.down_proj.weight",
                    "mlp.experts.down_proj",
                    [MergeModulelist(dim=0)])

A converter that silently produced the wrong ORDER, the wrong HALF, or a
transposed block would yield a model that loads clean, runs confidently, and
is not the published artifact.  No key-set diff can see that, because the key
sets are *supposed* to disagree.

So this tool does not compare key sets and does not compare statistics.  For
each expert it reads the raw little-endian BF16 bytes of
`...experts.{k}.{gate,up,down}_proj.weight` straight out of the local shard --
at the offset the published safetensors header records, with no safetensors
reader and no `transformers` in the path -- and compares them with
`memcmp` against the bytes of the corresponding slice of the live fused
parameter:

    gate_up_proj[k][0:I,   :]   ==  experts.k.gate_proj.weight
    gate_up_proj[k][I:2*I, :]   ==  experts.k.up_proj.weight
    down_proj[k]                ==  experts.k.down_proj.weight

The model is loaded through `k6/tools/hf_capture.load_model`, i.e. the exact
code path `bin/fidelity-dataset capture` uses, so a pass is a statement about
the production engine and not about a bespoke script.

Exit status is 0 only when every compared tensor matched exactly.

Usage:
  verify_fused_experts.py --model <ckpt dir> --receipt <fetch receipt json> \
      [--layer 3] [--experts all|0,1,255] [--dtype bfloat16] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hf_capture  # noqa: E402


def die(message: str) -> "SystemExit":
    print("verify_fused_experts: ERROR: %s" % message, file=sys.stderr, flush=True)
    return SystemExit(2)


def log(**fields: Any) -> None:
    print(json.dumps(fields, sort_keys=True), flush=True)


class ShardReader:
    """Raw byte reader keyed by the shards' own published safetensors headers.

    Deliberately does not use `safetensors`: the point of this check is to be
    independent of the library whose behaviour is under test.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._index: Dict[str, Any] = {}
        self._handles: Dict[str, Any] = {}
        for name in sorted(os.listdir(model_dir)):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(model_dir, name)
            with open(path, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            data_start = 8 + length
            for key, entry in header.items():
                if key == "__metadata__":
                    continue
                begin, end = entry["data_offsets"]
                self._index[key] = {"file": path, "dtype": entry["dtype"],
                                    "shape": entry["shape"],
                                    "start": data_start + begin,
                                    "stop": data_start + end}

    def has(self, key: str) -> bool:
        return key in self._index

    def meta(self, key: str) -> Dict[str, Any]:
        return self._index[key]

    def raw(self, key: str) -> bytes:
        entry = self._index[key]
        handle = self._handles.get(entry["file"])
        if handle is None:
            handle = self._handles[entry["file"]] = open(entry["file"], "rb")
        handle.seek(entry["start"])
        return handle.read(entry["stop"] - entry["start"])

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


def tensor_raw(tensor) -> bytes:
    """The BF16 bytes of a tensor slice, in the same little-endian layout a
    safetensors file stores them in."""
    import torch

    flat = tensor.detach().to("cpu").contiguous()
    if flat.dtype != torch.bfloat16:
        raise die("expected a bfloat16 parameter, got %s -- a dtype cast would make this "
                  "comparison meaningless" % flat.dtype)
    return flat.view(torch.uint16).numpy().tobytes()


def any_raw(tensor) -> bytes:
    """A tensor's bytes in safetensors' little-endian layout, any dtype.

    `torch` has no numpy dtype for bfloat16, so bf16 is reinterpreted as uint16
    first -- a bit-preserving view, never a cast.
    """
    import torch

    flat = tensor.detach().to("cpu").contiguous()
    if flat.dtype == torch.bfloat16:
        return flat.view(torch.uint16).numpy().tobytes()
    return flat.numpy().tobytes()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="verify_fused_experts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local checkpoint directory")
    ap.add_argument("--receipt", default=None,
                    help="fetch receipt; when given, its per-tensor sha256 are re-checked "
                         "against the shard bytes read here")
    ap.add_argument("--layer", type=int, action="append", default=None,
                    help="sparse layer index to check (repeatable; default: every layer "
                         "that has a fused expert parameter)")
    ap.add_argument("--experts", default="all",
                    help="'all' or a comma-separated list of expert indices")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--device-map", default=None,
                    help="load through the dispatched path instead of materialising the "
                         "whole model and calling .to(--device); 'auto', 'cpu', or JSON")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--check-all", action="store_true",
                    help="additionally compare EVERY other parameter and buffer "
                         "1:1 against its checkpoint tensor")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    args = ap.parse_args(argv)

    import torch

    device_map = args.device_map
    if device_map and device_map.strip().startswith("{"):
        device_map = json.loads(device_map)
    model, config, info = hf_capture.load_model(args.model, args.device, args.dtype,
                                                device_map=device_map)
    log(stage="loaded", architectures=list(getattr(config, "architectures", None) or []),
        num_hidden_layers=int(getattr(config, "num_hidden_layers", -1)),
        n_routed_experts=int(getattr(config, "n_routed_experts", -1)))

    # The load report, in full -- including the fields `to_dict()` drops.
    report = hf_capture.load_report(info)
    log(stage="load_report", **{k: (len(v) if isinstance(v, (list, set, dict)) else v)
                                for k, v in report.items()})

    reader = ShardReader(args.model)
    layers = args.layer
    if not layers:
        layers = []
        for name, _ in model.named_parameters():
            if name.endswith(".mlp.experts.gate_up_proj"):
                layers.append(int(name.split("model.layers.")[1].split(".")[0]))
        layers = sorted(set(layers))
    if not layers:
        raise die("this model has no fused `mlp.experts.gate_up_proj` parameter; there is "
                  "nothing for this check to prove")
    log(stage="targets", layers=layers)

    n_experts = int(getattr(config, "n_routed_experts"))
    if args.experts == "all":
        experts = list(range(n_experts))
    else:
        experts = [int(v) for v in args.experts.split(",") if v.strip() != ""]

    params = dict(model.named_parameters())
    results: List[Dict[str, Any]] = []
    matched = 0
    differed = 0
    absent = 0
    receipt = json.load(open(args.receipt)) if args.receipt else None
    digests = (receipt or {}).get("tensor_digests", {})
    digest_ok = 0
    digest_bad = 0

    import hashlib

    for layer in layers:
        gate_up = params["model.layers.%d.mlp.experts.gate_up_proj" % layer]
        down = params["model.layers.%d.mlp.experts.down_proj" % layer]
        inter = int(gate_up.shape[1]) // 2
        if int(gate_up.shape[1]) != 2 * inter:
            raise die("gate_up_proj second dim %d is odd; it cannot be a [gate|up] "
                      "concatenation" % int(gate_up.shape[1]))
        for k in experts:
            plan = (
                ("gate_proj", gate_up[k][0:inter, :]),
                ("up_proj", gate_up[k][inter:2 * inter, :]),
                ("down_proj", down[k]),
            )
            for projection, slice_ in plan:
                key = "model.layers.%d.mlp.experts.%d.%s.weight" % (layer, k, projection)
                if not reader.has(key):
                    absent += 1
                    results.append({"key": key, "status": "absent_from_shards"})
                    continue
                shard_bytes = reader.raw(key)
                live_bytes = tensor_raw(slice_)
                same = shard_bytes == live_bytes
                if same:
                    matched += 1
                else:
                    differed += 1
                    first = next((i for i in range(min(len(shard_bytes), len(live_bytes)))
                                  if shard_bytes[i] != live_bytes[i]), None)
                    results.append({
                        "key": key, "status": "DIFFERS",
                        "shard_bytes": len(shard_bytes), "live_bytes": len(live_bytes),
                        "first_differing_byte": first,
                        "shard_sha256": hashlib.sha256(shard_bytes).hexdigest(),
                        "live_sha256": hashlib.sha256(live_bytes).hexdigest(),
                        "shard_shape": reader.meta(key)["shape"],
                        "live_shape": list(slice_.shape)})
                if key in digests:
                    got = hashlib.sha256(shard_bytes).hexdigest()
                    if got == digests[key]["sha256"]:
                        digest_ok += 1
                    else:
                        digest_bad += 1
                        results.append({"key": key, "status": "RECEIPT_DIGEST_MISMATCH",
                                        "receipt_sha256": digests[key]["sha256"],
                                        "read_sha256": got})
        log(stage="layer_done", layer=layer, matched=matched, differed=differed,
            absent=absent)

    # Every OTHER tensor the model holds, name-mapped 1:1.  The fused experts
    # are the subtle case, but "the experts are right" is not "the model is
    # right": attention, norms, the router gate, the shared expert, the
    # embedding and the head all have to be the published bytes too, and a
    # sparse local tree is exactly the shape of thing that could silently serve
    # a hole as a tensor of zeros.
    direct_matched = direct_differed = direct_unmapped = 0
    unmapped_buffers: List[str] = []
    if args.check_all:
        parameter_names = set(dict(model.named_parameters()))
        named = dict(model.named_parameters())
        named.update(dict(model.named_buffers()))
        for name, tensor in sorted(named.items()):
            if name.endswith(".mlp.experts.gate_up_proj") or \
                    name.endswith(".mlp.experts.down_proj"):
                continue                      # covered above, byte for byte
            if not reader.has(name):
                if name not in parameter_names:
                    # A derived buffer -- `model.rotary_emb.inv_freq` and
                    # friends are COMPUTED from the config, never shipped. Not
                    # having a checkpoint key is correct for these.
                    unmapped_buffers.append(name)
                    continue
                direct_unmapped += 1
                results.append({"key": name, "status": "PARAMETER_WITH_NO_CHECKPOINT_KEY"})
                continue
            shard_bytes = reader.raw(name)
            live = tensor.detach().to("cpu").contiguous()
            live_bytes = any_raw(live)
            if shard_bytes == live_bytes:
                direct_matched += 1
            else:
                direct_differed += 1
                first = next((i for i in range(min(len(shard_bytes), len(live_bytes)))
                              if shard_bytes[i] != live_bytes[i]), None)
                results.append({"key": name, "status": "DIFFERS",
                                "shard_bytes": len(shard_bytes),
                                "live_bytes": len(live_bytes),
                                "first_differing_byte": first,
                                "dtype": str(live.dtype),
                                "shard_shape": reader.meta(name)["shape"],
                                "live_shape": list(live.shape)})
        log(stage="direct_done", matched=direct_matched, differed=direct_differed,
            unmapped_parameters=direct_unmapped,
            derived_buffers_not_in_checkpoint=unmapped_buffers)

    reader.close()
    compared = matched + differed
    summary = {
        "schema": "malaiwah.fused-expert-byte-check.v1",
        "model_dir": os.path.abspath(args.model),
        "architectures": list(getattr(config, "architectures", None) or []),
        "layers_checked": layers,
        "experts_checked": len(experts),
        "n_routed_experts": n_experts,
        "tensors_compared": compared,
        "tensors_matched_exactly": matched,
        "tensors_differed": differed,
        "tensors_absent_from_shards": absent,
        "direct_tensors_matched_exactly": direct_matched,
        "direct_tensors_differed": direct_differed,
        "direct_parameters_without_checkpoint_key": direct_unmapped,
        "derived_buffers_not_in_checkpoint": unmapped_buffers,
        "checked_all_parameters": bool(args.check_all),
        "device_map": args.device_map,
        "receipt_digests_rechecked": digest_ok,
        "receipt_digest_mismatches": digest_bad,
        "bytes_compared": None,
        "load_report": {k: sorted(v) if isinstance(v, (set,)) else v
                        for k, v in report.items()},
        "failures": [r for r in results if r.get("status") != "absent_from_shards"][:64],
        "verdict": ("PASS" if compared and differed == 0 and digest_bad == 0
                    and absent == 0 and direct_differed == 0 and direct_unmapped == 0
                    else "FAIL"),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    log(stage="summary", **{k: v for k, v in summary.items()
                            if k not in ("failures", "load_report")})
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
