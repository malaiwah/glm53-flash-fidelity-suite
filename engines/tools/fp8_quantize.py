#!/usr/bin/env python3
"""Quantize a bf16 safetensors checkpoint into the block-scaled FP8 e4m3 form.

    engines/tools/fp8_quantize.py --source <bf16 dir> --out <fp8 dir> \
        [--block 128 128] [--source-repo owner/name --source-revision 40hex]

The output is the FineGrainedFP8 checkpoint form `zai-org/GLM-5.3` ships and
`layer_outer` decodes: every eligible 2-D projection weight becomes an fp8
`weight` plus an fp32 `weight_scale_inv` on a ceil-padded block grid (the
last block along either axis may be partial), everything else stays bf16
and is named in `quantization_config.modules_to_not_convert`. Eligibility
mirrors GLM-5.3's own census, read from its index: attention projections,
the DSA indexer's wk/wq_b, dense and expert MLP projections, shared experts;
native: embeddings, head, every norm, the router and its correction bias,
the indexer's weights_proj and k_norm, and the MTP glue (eh_proj, enorm,
hnorm, shared_head.norm).

Quantizer: per block, scale = amax / 448 (the e4m3 maximum), q = round-to-
nearest e4m3 of w / scale, weight_scale_inv = scale. This is a FIXTURE
quantizer for rehearsing the candidate route on a small model (Fruit), not
a production quantization recipe; the produced README says so.

Shards keep their names; the index is rebuilt with the scale keys and the
new total_size. Tokenizer, config, license and generation files are copied.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

QUANTIZED_MODULE_SUFFIXES = (
    ".self_attn.q_a_proj", ".self_attn.q_b_proj", ".self_attn.kv_a_proj_with_mqa",
    ".self_attn.kv_b_proj", ".self_attn.o_proj",
    ".self_attn.indexer.wk", ".self_attn.indexer.wq_b",
    ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
    ".gate_proj", ".up_proj", ".down_proj",  # experts.E.* and shared_experts.*
)
E4M3_MAX = 448.0


def eligible(key: str, ndim: int) -> bool:
    if ndim != 2 or not key.endswith(".weight"):
        return False
    module = key[:-len(".weight")]
    if ".mlp.gate" == module[-len(".mlp.gate"):]:
        return False  # the router
    return any(module.endswith(suffix) for suffix in QUANTIZED_MODULE_SUFFIXES)


def quantize_block(tensor, block):
    import torch

    rows, cols = tensor.shape
    grid_rows, grid_cols = -(-rows // block[0]), -(-cols // block[1])
    w = torch.nn.functional.pad(
        tensor.to(torch.float32),
        (0, grid_cols * block[1] - cols, 0, grid_rows * block[0] - rows))
    w = w.reshape(grid_rows, block[0], grid_cols, block[1])
    amax = w.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / E4M3_MAX
    q = (w / scale).to(torch.float8_e4m3fn)
    q = q.reshape(grid_rows * block[0], grid_cols * block[1])[:rows, :cols]
    return q.contiguous(), scale.reshape(grid_rows, grid_cols).contiguous()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import load_file, save_file

    source, out = os.path.abspath(args.source), os.path.abspath(args.out)
    if os.path.exists(out):
        print("REFUSED: %s exists" % out)
        return 3
    os.makedirs(out)
    shards = sorted(n for n in os.listdir(source) if n.endswith(".safetensors"))
    weight_map, total_size, quantized, native_modules = {}, 0, [], set()
    for name in shards:
        tensors = load_file(os.path.join(source, name))
        result = {}
        for key, tensor in tensors.items():
            if eligible(key, tensor.ndim):
                q, scale = quantize_block(tensor, args.block)
                result[key] = q
                result[key + "_scale_inv"] = scale
                quantized.append(key)
            else:
                result[key] = tensor
                if key.endswith(".weight") or key.endswith(".bias"):
                    native_modules.add(key.rsplit(".", 1)[0])
        save_file(result, os.path.join(out, name), metadata={"format": "pt"})
        for key, tensor in result.items():
            weight_map[key] = name
            total_size += tensor.numel() * tensor.element_size()
        print("%s: %d tensors, %d quantized" % (name, len(tensors),
                                                sum(1 for k in tensors if eligible(k, tensors[k].ndim))))
    for name in os.listdir(source):
        path = os.path.join(source, name)
        if name.endswith(".safetensors") or name in ("model.safetensors.index.json", "README.md",
                                                     "MANIFEST.sha256", ".cache", ".gitattributes"):
            continue
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(out, name))
    config = json.load(open(os.path.join(source, "config.json"), encoding="utf-8"))
    config["quantization_config"] = {
        "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
        "weight_block_size": [int(args.block[0]), int(args.block[1])],
        "modules_to_not_convert": sorted(native_modules),
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": dict(sorted(weight_map.items()))}, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(
            "---\nlicense: mit\nbase_model: %s\ntags:\n- fidelity\n- fp8\n- fixture\n---\n"
            "# %s, block-scaled FP8 e4m3\n\n"
            "A FIXTURE for rehearsing quant-fidelity-suite's candidate route, not a "
            "serving quantization. Produced by `engines/tools/fp8_quantize.py` from "
            "`%s@%s`: every attention/indexer/MLP/expert projection weight is fp8 e4m3 "
            "with an fp32 `weight_scale_inv` per %dx%d block (scale = block amax / 448, "
            "ceil-padded grid, partial last blocks kept), the same checkpoint form "
            "`zai-org/GLM-5.3` ships; embeddings, head, norms, router and MTP glue stay "
            "bf16 and are listed in `quantization_config.modules_to_not_convert`.\n\n"
            "%d tensors quantized; %d modules native.\n"
            % (args.source_repo or "(local)", os.path.basename(out),
               args.source_repo or "(local)", args.source_revision or "(local)",
               args.block[0], args.block[1], len(quantized), len(native_modules)))
    print("quantized %d tensors; %d native modules; total_size %d"
          % (len(quantized), len(native_modules), total_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
