#!/usr/bin/env python3
"""Author a candidate scope file for a block-scaled FP8 checkpoint from its bytes.

    engines/tools/fp8_scope.py --index model.safetensors.index.json \
        --config config.json --repo owner/name --revision 40hex --out scope.json

The scope says, per registry tensor class, whether the class is quantized
(fp8_e4m3, 8 bits) or native (bf16, 16 bits). It is read from the
checkpoint INDEX -- a class is quantized when every one of its 2-D weights
has a `weight_scale_inv` sibling, native when none has -- never from the
repo name or the README. A class with both kinds of tensors is split by
what the bytes say (GLM-5.3's `moe.experts` is entirely FP8; its `attn.other`
holds the native `indexer.weights_proj` beside the FP8 `indexer.wk/wq_b`),
and the split is written out as two assignments with notes, so scope_digest
describes the artifact exactly.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

CLASS_RULES = (
    ("embed_tokens", re.compile(r"^model\.embed_tokens\.weight$")),
    ("lm_head", re.compile(r"^lm_head\.weight$")),
    ("mtp", re.compile(r"^model\.layers\.(\d+)\.(eh_proj|enorm|hnorm|shared_head)\.")),
    ("moe.router", re.compile(r"^model\.layers\.\d+\.mlp\.gate\.(weight|e_score_correction_bias)$")),
    ("moe.experts", re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.")),
    ("moe.shared_expert", re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.")),
    ("mlp.gate", re.compile(r"^model\.layers\.\d+\.mlp\.gate_proj\.weight$")),
    ("mlp.up", re.compile(r"^model\.layers\.\d+\.mlp\.up_proj\.weight$")),
    ("mlp.down", re.compile(r"^model\.layers\.\d+\.mlp\.down_proj\.weight$")),
    ("attn.qkv", re.compile(r"^model\.layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|q_proj|k_proj|v_proj)\.weight$")),
    ("attn.o", re.compile(r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$")),
    ("attn.other", re.compile(r"^model\.layers\.\d+\.self_attn\.")),
    ("norm", re.compile(r"(^model\.norm\.weight$|layernorm|\.norm\.)")),
)


def classify(key: str, num_hidden_layers: int):
    layer = re.match(r"^model\.layers\.(\d+)\.", key)
    if layer and int(layer.group(1)) >= num_hidden_layers:
        return "mtp"
    for name, pattern in CLASS_RULES:
        if pattern.search(key):
            return name
    return "other"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = json.load(open(args.config, encoding="utf-8"))
    qc = config["quantization_config"]
    block = qc["weight_block_size"]
    layers = int(config.get("num_hidden_layers") or (config.get("text_config") or {})["num_hidden_layers"])
    keys = list(json.load(open(args.index, encoding="utf-8"))["weight_map"])
    scaled = {k[:-len("_scale_inv")] for k in keys if k.endswith("_scale_inv")}
    census = defaultdict(lambda: {"fp8": [], "native": []})
    for key in keys:
        if key.endswith("_scale_inv"):
            continue
        cls = classify(key, layers)
        census[cls]["fp8" if key in scaled else "native"].append(key)
    note = ("read from %s@%s model.safetensors.index.json: a tensor is FP8 e4m3 with a "
            "%dx%d block scale when a weight_scale_inv sibling exists, native bf16 "
            "otherwise" % (args.repo, args.revision, block[0], block[1]))
    assignments = []
    for cls in sorted(census):
        fp8, native = census[cls]["fp8"], census[cls]["native"]
        if fp8:
            examples = sorted({re.sub(r"layers\.\d+", "layers.N", re.sub(r"experts\.\d+", "experts.E", k)) for k in fp8})
            assignments.append({
                "tensor_class": cls, "treatment": "quantized", "format": "fp8_e4m3",
                "bits_per_weight": 8, "layer_range": "all",
                "note": "%d tensors: %s. %s" % (len(fp8), ", ".join(examples[:6]), note)})
        if native:
            examples = sorted({re.sub(r"layers\.\d+", "layers.N", re.sub(r"experts\.\d+", "experts.E", k)) for k in native})
            assignments.append({
                "tensor_class": cls, "treatment": "native", "format": "bf16",
                "bits_per_weight": 16, "layer_range": "all",
                "note": "%d tensors: %s. %s" % (len(native), ", ".join(examples[:6]), note)})
    head_native = bool(census["lm_head"]["native"]) and not census["lm_head"]["fp8"]
    scope = {
        "policy": "mixed",
        "head_policy": "native" if head_native else "quantized",
        "kv_cache_dtype": "bf16",
        "mtp_included": bool(census.get("mtp")),
        "activation_quantization": None,
        "weight_block_size": block,
        "assignments": assignments,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(scope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in assignments:
        print("%-20s %-10s %-9s %s" % (row["tensor_class"], row["treatment"], row["format"],
                                       row["note"].split(":")[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
