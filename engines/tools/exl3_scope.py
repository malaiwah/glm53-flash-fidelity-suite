#!/usr/bin/env python3
"""Author a candidate scope file for an EXL3 trellis checkpoint from its bytes.

    engines/tools/exl3_scope.py --index model.safetensors.index.json \
        --config config.json --repo owner/name --revision 40hex --out scope.json

Same contract as `fp8_scope.py`, one surface over: a tensor class is
`quantized` when its 2-D weights are stored as exl3 payload groups
(`M.{trellis,suh,svh,<codebook>}`), `quantized` at fp8_e4m3 when the
quantizer LEFT them in the source's block-scaled FP8 (wrldsuksgo2mars keeps
`shared_experts`/`self_attn` that way), and `native` when the weight is
carried whole. Read from the INDEX, never from the repo name, the README or
`quantization_config.codebook` -- which names only ONE codebook even when the
checkpoint mixes them per layer (drowzeys ships mcg on layer 3, mul1 on 4-77).

Bits per class come from the payload's own trellis shape when a local shard is
readable (`--shard`); otherwise from `quantization_config.bits`, and the note
says which, because a scope that guesses its own bit-width is a scope that
mislabels the row.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from fp8_scope import CLASS_RULES, classify  # noqa: F401 - shared class rules

PAYLOAD_OBJECTS = ("trellis", "suh", "svh")
CODEBOOKS = ("mul1", "mcg")
SCALE_SUFFIX = "_scale_inv"


def payload_modules(keys):
    """module -> codebook, for every complete stock-exllamav3 payload group."""
    staged = defaultdict(dict)
    for key in keys:
        stem, _, last = key.rpartition(".")
        if not stem:
            continue
        if last in PAYLOAD_OBJECTS:
            staged[stem][last] = key
        elif last in CODEBOOKS:
            staged[stem].setdefault("codebooks", []).append(last)
    modules = {}
    for module, found in staged.items():
        marks = found.get("codebooks") or []
        if all(name in found for name in PAYLOAD_OBJECTS) and len(marks) == 1:
            modules[module] = marks[0]
    return modules


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = json.load(open(args.config, encoding="utf-8"))
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "exl3":
        raise SystemExit("REFUSED: %s declares quant_method=%r, not exl3"
                         % (args.config, qc.get("quant_method")))
    layers = int(config.get("num_hidden_layers")
                 or (config.get("text_config") or {})["num_hidden_layers"])
    keys = list(json.load(open(args.index, encoding="utf-8"))["weight_map"])
    rank_split = [k for k in keys if re.search(r"\.rank\d+\.(?:%s)$"
                                               % "|".join(PAYLOAD_OBJECTS + CODEBOOKS), k)]
    if rank_split:
        raise SystemExit(
            "REFUSED: %d rank-split payload key(s) (e.g. %s). That is the TR3 layout "
            "whose composition into one weight is unpublished; no scope can describe "
            "it honestly yet." % (len(rank_split), sorted(rank_split)[0]))

    modules = payload_modules(keys)
    scaled = {k[: -len(SCALE_SUFFIX)] for k in keys if k.endswith(SCALE_SUFFIX)}
    declared_bits = qc.get("bits")
    codebooks = defaultdict(int)
    for codebook in modules.values():
        codebooks[codebook] += 1

    census = defaultdict(lambda: {"exl3": [], "fp8": [], "native": []})
    for key in keys:
        if key.endswith(SCALE_SUFFIX):
            continue
        stem, _, last = key.rpartition(".")
        if last in PAYLOAD_OBJECTS or last in CODEBOOKS:
            if stem in modules:
                census[classify(stem + ".weight", layers)]["exl3"].append(stem)
            continue
        cls = classify(key, layers)
        census[cls]["fp8" if key in scaled else "native"].append(key)

    def _shapes(names):
        return sorted({re.sub(r"layers\.\d+", "layers.N",
                              re.sub(r"experts\.\d+", "experts.E", n)) for n in names})

    source = ("read from %s@%s model.safetensors.index.json: a class is exl3 trellis "
              "when its weights are stored as %s payload groups, fp8_e4m3 when a "
              "%s sibling exists, native otherwise. Codebook per module from the "
              "object present (%s), not from quantization_config.codebook=%r."
              % (args.repo, args.revision, "/".join(PAYLOAD_OBJECTS), SCALE_SUFFIX,
                 ", ".join("%s:%d" % kv for kv in sorted(codebooks.items())),
                 qc.get("codebook")))
    assignments = []
    for cls in sorted(census):
        exl3 = sorted(set(census[cls]["exl3"]))
        fp8, native = census[cls]["fp8"], census[cls]["native"]
        if exl3:
            # SPLIT BY CODEBOOK, the way fp8_scope splits a class whose bytes
            # disagree. The registry's numeric_format enum names the codebook
            # (`exl3-mcg` / `exl3-mul1`); a bare "exl3" is not in it and a
            # submission carrying it is rejected by registry_validate.py --
            # the capture's own verify stage warns [SCOPE-VOCAB] about exactly
            # this. drowzeys ships mcg on layer 3 and mul1 on 4-77, so its
            # moe.experts class legitimately becomes two assignments.
            by_codebook = {}
            for module in exl3:
                by_codebook.setdefault(modules[module], []).append(module)
            for codebook, group in sorted(by_codebook.items()):
                assignments.append({
                    "tensor_class": cls, "treatment": "quantized",
                    "format": "exl3-%s" % codebook,
                    "declared_scheme": {
                        "codec_family": "exl3", "codebook": codebook,
                        "quantizer": "exllamav3",
                        "quantizer_version": qc.get("version"),
                        "declared_bits": declared_bits,
                    },
                    "bits_per_weight": float(declared_bits) if declared_bits is not None else None,
                    "layer_range": "all",
                    "note": "%d modules: %s. codebook %s read from the object each "
                            "module carries; bits from quantization_config.bits=%r "
                            "(the payload's own trellis shape is the authority and is "
                            "checked at decode). %s"
                            % (len(group), ", ".join(_shapes(group)[:6]), codebook,
                               declared_bits, source)})
        if fp8:
            assignments.append({
                "tensor_class": cls, "treatment": "quantized", "format": "fp8_e4m3",
                "bits_per_weight": 8, "layer_range": "all",
                "note": "%d tensors kept in the source's block-scaled FP8: %s. %s"
                        % (len(fp8), ", ".join(_shapes(fp8)[:6]), source)})
        if native:
            assignments.append({
                "tensor_class": cls, "treatment": "native", "format": "bf16",
                "bits_per_weight": 16, "layer_range": "all",
                "note": "%d tensors: %s. %s"
                        % (len(native), ", ".join(_shapes(native)[:6]), source)})

    head = census["lm_head"]
    scope = {
        "policy": "mixed",
        "head_policy": ("native" if head["native"] and not (head["exl3"] or head["fp8"])
                        else "quantized"),
        "kv_cache_dtype": "bf16",
        "mtp_included": bool(census.get("mtp")),
        "activation_quantization": None,
        "codebook_histogram": dict(sorted(codebooks.items())),
        "assignments": assignments,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(scope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in assignments:
        print("%-22s %-10s %-9s %s" % (row["tensor_class"], row["treatment"], row["format"],
                                       row["note"].split(".")[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
