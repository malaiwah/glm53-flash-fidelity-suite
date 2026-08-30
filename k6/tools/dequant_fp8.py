#!/usr/bin/env python3
"""Materialise a block-wise FP8 checkpoint as bf16 -- "dequantize and run".

Why this exists
---------------
Capturing `Qwen/Qwen3.8-27B-FP8` through stock transformers on this box fails
twice over, and both failures would silently or loudly corrupt a measurement:

1. **Silent, and the dangerous one.** The producer's
   `modules_to_not_convert` lists `...layers.N.mlp.gate` -- a MoE router that
   does not exist in this dense checkpoint. `should_convert_module` tests
   `re.match(key, full_name)`, which is anchored only at the START, so that
   pattern also matches `...layers.N.mlp.gate_proj`. All 65 `gate_proj` modules
   are therefore excluded from FP8 conversion, their fp8 weights are loaded into
   plain bf16 Linears WITHOUT the block scale ever being applied, and the 65
   `gate_proj.weight_scale_inv` tensors fall out of the load as "unexpected".
   Nothing raises. The model runs and produces confident garbage in that
   projection.

2. **Loud.** The fused `deep-gemm` fp8 kernel aborts with
   `Assertion error ... Unknown recipe` on this RTX PRO 6000 (Blackwell).

So the vendor kernel path is unavailable here regardless. This tool takes the
other accepted road -- the same one the campaign's GGUF/EXL3/MLX rows already
travel: decode the STORED weights exactly, then run them densely. It applies
EVERY block scale, including the 65 that transformers drops, so the arithmetic
is the checkpoint's own.

What that measures, and what it does not
----------------------------------------
This measures the error of the WEIGHTS AS STORED. The published checkpoint also
declares `activation_scheme: "dynamic"`, i.e. the served model additionally
quantizes activations per-token at runtime. That term is NOT present here, so a
number taken over this materialisation is a LOWER BOUND on the served model's
divergence, not the served model's divergence. Say so on the row.

Dequantisation
--------------
`weight_block_size: [128, 128]`, scale shape `[ceil(out/128), ceil(in/128)]`,
and the DeepSeek-style convention that `weight_scale_inv` is the MULTIPLICATIVE
dequantisation scale:

    w[i, j] = fp8[i, j] * scale_inv[i // 128, j // 128]

Accumulated in float32 and stored bf16, which is the reference dtype the root
was captured in.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCALE_SUFFIX = ".weight_scale_inv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.loads(open(os.path.join(args.src, "config.json")).read())
    qc = cfg.get("quantization_config") or {}
    block = tuple(qc.get("weight_block_size") or (128, 128))

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    index = json.loads(open(index_path).read())
    weight_map = index["weight_map"]

    shards = sorted(set(weight_map.values()))
    new_map = {}
    stats = {"dequantized": 0, "copied": 0, "scales_consumed": 0}

    for shard in shards:
        src = os.path.join(args.src, shard)
        tensors = {}
        with safe_open(src, framework="pt") as f:
            keys = list(f.keys())
            scale_keys = {k for k in keys if k.endswith(SCALE_SUFFIX)}
            for k in keys:
                if k in scale_keys:
                    continue
                t = f.get_tensor(k)
                sk = k[: -len(".weight")] + SCALE_SUFFIX if k.endswith(".weight") else None
                if sk and sk in scale_keys and t.dtype == torch.float8_e4m3fn:
                    s = f.get_tensor(sk).to(torch.float32)
                    w = t.to(torch.float32)
                    # expand the per-block scale over the weight, then trim:
                    # the last block is partial whenever a dim is not a multiple
                    # of the block size, so repeat_interleave + slice rather than
                    # reshape, which would silently require exact divisibility.
                    s = s.repeat_interleave(block[0], dim=0).repeat_interleave(block[1], dim=1)
                    s = s[: w.shape[0], : w.shape[1]]
                    if s.shape != w.shape:
                        raise SystemExit("scale/weight shape mismatch on %s: %s vs %s"
                                         % (k, tuple(s.shape), tuple(w.shape)))
                    tensors[k] = (w * s).to(torch.bfloat16)
                    stats["dequantized"] += 1
                    stats["scales_consumed"] += 1
                elif t.dtype == torch.float8_e4m3fn:
                    raise SystemExit("fp8 tensor with no scale: %s" % k)
                else:
                    tensors[k] = t
                    stats["copied"] += 1
        out_shard = os.path.join(args.out, shard)
        save_file(tensors, out_shard, metadata={"format": "pt"})
        for k in tensors:
            new_map[k] = shard
        print(json.dumps({"shard": shard, "tensors": len(tensors)}), flush=True)
        del tensors

    # config without quantization_config: the tree is now genuinely bf16
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    index["weight_map"] = new_map
    index.get("metadata", {}).pop("total_size", None)
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    for name in os.listdir(args.src):
        if name.endswith((".json", ".txt", ".py", ".model")) and \
           name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(os.path.join(args.src, name), os.path.join(args.out, name))

    print(json.dumps({"stage": "done", **stats}))


if __name__ == "__main__":
    main()
