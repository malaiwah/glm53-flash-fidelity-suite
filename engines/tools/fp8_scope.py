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

# A text-only release keys its stack `model.layers.N.`; a vision-language
# release (`Glm5NextForConditionalGeneration`: GLM-5.3-Flash and every
# quantization of it) nests the same stack at `model.language_model.layers.N.`
# beside `model.visual.*`, and its geometry under `config.text_config`. The
# vision tower is not a decoder class and lands in `other` by construction.
STACK = r"^model\.(?:language_model\.)?"
LAYER_RE = re.compile(STACK + r"layers\.(\d+)\.")

CLASS_RULES = (
    ("embed_tokens", re.compile(STACK + r"embed_tokens\.weight$")),
    ("lm_head", re.compile(r"^lm_head\.weight$")),
    ("mtp", re.compile(STACK + r"layers\.(\d+)\.(eh_proj|enorm|hnorm|shared_head)\.")),
    ("moe.router", re.compile(STACK + r"layers\.\d+\.mlp\.gate\.(weight|e_score_correction_bias)$")),
    ("moe.experts", re.compile(STACK + r"layers\.\d+\.mlp\.experts\.\d+\.")),
    ("moe.shared_expert", re.compile(STACK + r"layers\.\d+\.mlp\.shared_experts\.")),
    ("mlp.gate", re.compile(STACK + r"layers\.\d+\.mlp\.gate_proj\.weight$")),
    ("mlp.up", re.compile(STACK + r"layers\.\d+\.mlp\.up_proj\.weight$")),
    ("mlp.down", re.compile(STACK + r"layers\.\d+\.mlp\.down_proj\.weight$")),
    ("attn.qkv", re.compile(STACK + r"layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|q_proj|k_proj|v_proj)\.weight$")),
    ("attn.o", re.compile(STACK + r"layers\.\d+\.self_attn\.o_proj\.weight$")),
    ("attn.other", re.compile(STACK + r"layers\.\d+\.self_attn\.")),
    ("norm", re.compile("(" + STACK + r"norm\.weight$|layernorm|\.norm\.)")),
)


SCHEMA_ASSIGNMENT_KEYS = ("tensor_class", "treatment", "format", "bits_per_weight",
                          "layer_range", "note")


def layer_of(key: str):
    match = LAYER_RE.match(key)
    return int(match.group(1)) if match else None


def decoder_layers(config) -> int:
    """`num_hidden_layers` of the text stack: top level, or `text_config` for a VL release."""
    for block in (config, config.get("text_config") or {}):
        value = block.get("num_hidden_layers")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise SystemExit("REFUSED: config declares no num_hidden_layers at top level or "
                     "under text_config; the MTP boundary cannot be placed")


def range_string(layers) -> str:
    """'3', '4-77', '0-2,78' -- the registry's free-form but comparable layer_range."""
    layers = sorted(set(layers))
    if not layers:
        return "all"
    runs, start, prev = [], layers[0], layers[0]
    for value in layers[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in runs)


def assignments_from_census(census, source: str):
    """Registry-valid scope rows from a per-class census.

    `census` maps tensor_class -> list of (name, treatment, format, bits).
    The registry's SCOPE-004 refuses two rows with the same
    (tensor_class, layer_range), and the assignment schema is closed
    (tensor_class, treatment, format, bits_per_weight, layer_range, note).
    So a class whose tensors differ is written one of two ways, decided by
    the bytes: when the groups occupy DISJOINT layer sets (drowzeys: mcg on
    layer 3, mul1 on 4-77) each group gets its own layer_range; when they
    share layers (every FP8-derived GLM-5.3: attn.other carries the indexer's
    FP8 wk/wq_b beside native norms, the MTP block mixes both) the class is
    ONE row of `treatment: quantized, format: mixed, bits_per_weight: null`
    whose note carries the exact census. Never two rows on the same range.
    """
    rows = []
    for cls in sorted(census):
        groups = {}
        for name, treatment, fmt, bits in census[cls]:
            groups.setdefault((treatment, fmt, bits), []).append(name)
        layer_sets = {key: {layer_of(n) for n in names} - {None} for key, names in groups.items()}
        all_layers = set().union(*layer_sets.values()) if layer_sets else set()
        keys = sorted(groups, key=lambda k: (k[0], k[1], str(k[2])))
        disjoint = all(
            not (layer_sets[a] & layer_sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:])
        if len(keys) == 1 or (disjoint and all(layer_sets[k] for k in keys)):
            for key in keys:
                treatment, fmt, bits = key
                names = groups[key]
                rows.append({
                    "tensor_class": cls, "treatment": treatment, "format": fmt,
                    "bits_per_weight": bits,
                    "layer_range": "all" if len(keys) == 1 else range_string(layer_sets[key]),
                    "note": "%d tensors: %s. %s" % (len(names), ", ".join(_shapes(names)[:6]), source)})
            continue
        # overlapping groups on shared layers: one honest row. A class whose
        # groups are ALL native (bf16 weights beside fp32 router bias / SSM
        # scalars) is native at more than one width, not quantized: the
        # treatment says what was done to it, the format says how it is stored.
        census_note = "; ".join(
            "%d x %s:%s%s (%s)" % (len(groups[k]), k[0], k[1], ("@%s" % k[2]) if k[2] is not None else "",
                                   ", ".join(_shapes(groups[k])[:3]))
            for k in keys)
        rows.append({
            "tensor_class": cls,
            "treatment": "native" if all(k[0] == "native" for k in keys) else "quantized",
            "format": "mixed",
            "bits_per_weight": None,
            "layer_range": "all" if not all_layers else range_string(all_layers),
            "note": "class mixes formats on the same layers (SCOPE-004 admits one row per "
                    "class and layer_range): %s. %s" % (census_note, source)})
    for row in rows:
        assert set(row) <= set(SCHEMA_ASSIGNMENT_KEYS), row
    return rows


def _shapes(names):
    return sorted({re.sub(r"layers\.\d+", "layers.N", re.sub(r"experts\.\d+", "experts.E", n)) for n in names})


def classify(key: str, num_hidden_layers: int):
    layer = LAYER_RE.match(key)
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
    layers = decoder_layers(config)
    keys = list(json.load(open(args.index, encoding="utf-8"))["weight_map"])
    scaled = {k[:-len("_scale_inv")] for k in keys if k.endswith("_scale_inv")}
    census = defaultdict(list)
    for key in keys:
        if key.endswith("_scale_inv"):
            continue
        cls = classify(key, layers)
        if key in scaled:
            census[cls].append((key, "quantized", "fp8_e4m3", 8))
        else:
            census[cls].append((key, "native", "bf16", 16))
    note = ("read from %s@%s model.safetensors.index.json: a tensor is FP8 e4m3 with a "
            "%dx%d block scale when a weight_scale_inv sibling exists, native bf16 "
            "otherwise" % (args.repo, args.revision, block[0], block[1]))
    assignments = assignments_from_census(census, note)
    head_native = all(t == "native" for _, t, _, _ in census.get("lm_head", []))
    scope = {
        "policy": "mixed",
        "head_policy": "native" if head_native else "quantized",
        "kv_cache_dtype": "bf16",
        "mtp_included": bool(census.get("mtp")),
        "activation_quantization": None,
        # weight_block_size is not a scope-schema key; it is in every
        # assignment note and in the sealed weights_decode block.
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
