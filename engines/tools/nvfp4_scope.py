#!/usr/bin/env python3
"""Author a candidate scope file for a modelopt NVFP4 checkpoint from its bytes.

    engines/tools/nvfp4_scope.py --index model.safetensors.index.json \
        --config config.json --repo owner/name --revision 40hex --out scope.json \
        [--headers-json cached-shard-headers.json]

Same contract as `fp8_scope.py` / `exl3_scope.py`, one surface over: a routed
expert projection is `quantized` at nvfp4 @ 4 when its module ships the
modelopt component set {weight U8 [out, in/2], weight_scale F8_E4M3
[out, in/16], weight_scale_2 F32 []} (+ the activation `input_scale`, read
for the record and never applied), `native` at its stored dtype when it ships
whole (the MTP layer's experts in every flagship export), and every other
tensor is `native` at the dtype its shard header states. Read from the INDEX
and the SHARD HEADERS (ranged HTTP, no shard download), never from the repo
name, the README or `quantization_config.ignore` -- the config declares
NVFP4 through two different spellings (config_groups vs a flat group_size)
and its ignore list is a producer's pattern list, not a tensor census.

The head policy is read off `lm_head.weight`'s own dtype. A routed module
whose component set or dtypes are anything else, a non-routed tensor stored
packed (U8/F8) or with a scale sibling, and an expert layer outside the
family geometry are each a REFUSAL here, at $0, not a silent under-count
that refuses on the pod after the fetch.

Rows are registry-valid by construction (`fp8_scope.assignments_from_census`):
one row per (tensor_class, layer_range), a single `format: mixed` row when a
class mixes stored widths on the same layers (the router's bf16 weight beside
its fp32 correction bias -- or, in an export that rounded the bias, a single
bf16 row: the digest then says so).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request
from collections import defaultdict

from fp8_scope import CLASS_RULES, assignments_from_census, classify, decoder_layers  # noqa: F401

GROUP_SIZE = 16
DECODE = ("weight", "weight_scale", "weight_scale_2")
ACTIVATION = ("input_scale",)
EXPERT_RE = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.([a-z0-9_]+)$")
DTYPE_FORMAT = {"BF16": ("bf16", 16), "F16": ("fp16", 16), "F32": ("fp32", 32)}
PACKED_DTYPES = ("U8", "I8", "F8_E4M3", "F8_E5M2")


def shard_headers(repo, revision, weight_map):
    """shard -> safetensors header (ranged HTTP reads: 8-byte length, then the JSON)."""
    headers = {}
    for shard in sorted(set(weight_map.values())):
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
        head = urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=0-7"})).read(8)
        n = struct.unpack("<Q", head)[0]
        headers[shard] = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=8-%d" % (8 + n - 1)})).read(n))
    return headers


def tensor_meta(headers, weight_map):
    """key -> {dtype, shape} from the shard headers, checked against the index."""
    meta = {}
    for shard, header in headers.items():
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if weight_map.get(key) != shard:
                raise SystemExit("REFUSED: %s is in shard %s's header but the index maps it to %r"
                                 % (key, shard, weight_map.get(key)))
            meta[key] = {"dtype": entry["dtype"], "shape": list(entry["shape"])}
    missing = sorted(set(weight_map) - set(meta))
    if missing:
        raise SystemExit("REFUSED: %d index keys absent from every shard header, e.g. %s"
                         % (len(missing), missing[:3]))
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--headers-json", default=None,
                        help="a JSON cache {shard: {header: {...}}} of the shard headers "
                             "fetched earlier by ranged HTTP; without it every header is "
                             "fetched now (one ranged request pair per shard)")
    parser.add_argument("--save-headers-json", default=None,
                        help="write the fetched headers to this JSON cache")
    args = parser.parse_args(argv)

    config = json.load(open(args.config, encoding="utf-8"))
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "modelopt" or qc.get("quant_algo") != "NVFP4":
        raise SystemExit("REFUSED: %s declares quant_method=%r quant_algo=%r: not a modelopt "
                         "NVFP4 artifact" % (args.config, qc.get("quant_method"),
                                             qc.get("quant_algo")))
    groups = qc.get("config_groups")
    if isinstance(groups, dict):
        weights = ((groups.get("group_0") or {}).get("weights") or {})
        declared = "config_groups.group_0.weights num_bits=%s group_size=%s" % (
            weights.get("num_bits"), weights.get("group_size"))
        group_size = weights.get("group_size")
    else:
        declared = "quantization_config.group_size=%s" % qc.get("group_size")
        group_size = qc.get("group_size")
    if group_size != GROUP_SIZE:
        raise SystemExit("REFUSED: declared group_size %r is not the NVFP4 group size %d"
                         % (group_size, GROUP_SIZE))
    producer = qc.get("producer") or {}
    layers = decoder_layers(config)
    weight_map = json.load(open(args.index, encoding="utf-8"))["weight_map"]

    if args.headers_json:
        cached = json.load(open(args.headers_json, encoding="utf-8"))
        headers = {shard: rec["header"] if "header" in rec else rec for shard, rec in cached.items()}
        missing = sorted(set(weight_map.values()) - set(headers))
        if missing:
            raise SystemExit("REFUSED: --headers-json lacks %d shard(s), e.g. %s"
                             % (len(missing), missing[:3]))
    else:
        headers = shard_headers(args.repo, args.revision, weight_map)
        if args.save_headers_json:
            with open(args.save_headers_json, "w", encoding="utf-8") as handle:
                json.dump({shard: {"header": header} for shard, header in headers.items()}, handle)
    meta = tensor_meta(headers, weight_map)

    modules = defaultdict(dict)
    census = defaultdict(list)
    activation_scales = 0
    for key in weight_map:
        match = EXPERT_RE.match(key)
        if match:
            layer = int(match.group(1))
            if layer > layers:
                raise SystemExit("REFUSED: %s names expert layer %d beyond the %d decoder "
                                 "layers + 1 MTP layer the config declares" % (key, layer, layers))
            modules[(layer, int(match.group(2)), match.group(3))][match.group(4)] = key
            continue
        entry = meta[key]
        if entry["dtype"] in PACKED_DTYPES:
            raise SystemExit("REFUSED: non-routed tensor %s is stored %s; this tool speaks the "
                             "routed-experts-only modelopt NVFP4 layout" % (key, entry["dtype"]))
        if key.endswith(("_scale", "_scale_inv", "_scale_2")):
            raise SystemExit("REFUSED: non-routed scale tensor %s; this tool speaks the "
                             "routed-experts-only modelopt NVFP4 layout" % key)
        fmt, bits = DTYPE_FORMAT.get(entry["dtype"], ("unknown", None))
        if fmt == "unknown":
            raise SystemExit("REFUSED: %s has dtype %s this tool does not label" % (key, entry["dtype"]))
        census[classify(key, layers)].append((key, "native", fmt, bits))

    quantized = 0
    for (layer, expert, projection), comps in sorted(modules.items()):
        name = "%s.weight" % key_stem(comps)
        present = set(comps)
        if present == {"weight"}:
            entry = meta[comps["weight"]]
            fmt, bits = DTYPE_FORMAT.get(entry["dtype"], ("unknown", None))
            if fmt == "unknown":
                raise SystemExit("REFUSED: %s ships as a lone %s weight with no scales: neither "
                                 "packed-with-scales nor native" % (name, entry["dtype"]))
            census[classify(name, layers)].append((name, "native", fmt, bits))
            continue
        if not set(DECODE) <= present or not present <= set(DECODE) | set(ACTIVATION):
            raise SystemExit("REFUSED: %s carries component set %s, not the modelopt NVFP4 "
                             "{%s} (+%s)" % (name, sorted(present), ", ".join(DECODE),
                                              ", ".join(ACTIVATION)))
        w, s, s2 = (meta[comps[c]] for c in DECODE)
        if (w["dtype"] != "U8" or s["dtype"] != "F8_E4M3" or s2["dtype"] != "F32"
                or len(w["shape"]) != 2 or s["shape"] != [w["shape"][0], w["shape"][1] * 2 // GROUP_SIZE]
                or s2["shape"] not in ([], [1])):
            raise SystemExit("REFUSED: %s components are not U8 [out, in/2] / F8_E4M3 [out, in/16] "
                             "/ F32 []: %s" % (name, {c: meta[comps[c]] for c in DECODE}))
        if "input_scale" in comps:
            activation_scales += 1
        quantized += 1
        census[classify(name, layers)].append((name, "quantized", "nvfp4", 4.0))

    source = ("read from %s@%s model.safetensors.index.json + every shard header (ranged HTTP): "
              "a routed expert projection is nvfp4 (e2m1, group %d along the input axis, "
              "weight U8 [out, in/2] + weight_scale F8_E4M3 [out, in/16] + weight_scale_2 F32 "
              "scalar; %d such modules) when it ships the modelopt component set, native at its "
              "stored dtype when it ships whole; every other tensor native at its stored dtype. "
              "Declared by quantization_config: quant_method modelopt, quant_algo NVFP4, %s, "
              "producer %s. %d modules also ship a per-tensor F32 input_scale (a static "
              "activation quantity: recorded, never applied by the weights-only decode)."
              % (args.repo, args.revision, GROUP_SIZE, quantized, declared,
                 ("%s %s" % (producer.get("name"), producer.get("version"))
                  if producer else "undeclared"),
                 activation_scales))
    assignments = assignments_from_census(census, source)
    head = census.get("lm_head", [])
    if not head:
        raise SystemExit("REFUSED: no lm_head.weight in the index; the head policy cannot be read")
    scope = {
        "policy": "mixed",
        "head_policy": ("native" if all(t == "native" for _, t, _, _ in head) else "quantized"),
        "kv_cache_dtype": "bf16",
        "mtp_included": bool(census.get("mtp")),
        "activation_quantization": None,
        "assignments": assignments,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(scope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in assignments:
        print("%-22s %-10s %-9s %s" % (row["tensor_class"], row["treatment"], row["format"],
                                       row["note"].split(".")[0]))
    print("quantized modules %d, input_scale tensors %d, head %s (%s), mtp_included %s"
          % (quantized, activation_scales, scope["head_policy"],
             meta["lm_head.weight"]["dtype"], scope["mtp_included"]))
    return 0


def key_stem(comps):
    """The module stem shared by a routed module's component keys."""
    return next(iter(comps.values())).rsplit(".", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
