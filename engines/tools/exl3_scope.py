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

Bits per class are the artifact's DECLARED bits (`quantization_config.bits`,
or `hybrid_tr3_tail.bits_avg` for a TR3 release with per-expert K3/K4 tiers);
the payload's own K is checked against that declaration at decode time by
`layer_outer.materialize_trellis_subset`, which refuses a mismatch, so the
row cannot mislabel its bit-width past the capture.

Rows are registry-valid by construction (`fp8_scope.assignments_from_census`):
one row per (tensor_class, layer_range), disjoint layer ranges when a class
splits by layer, and a single `format: mixed` row when it mixes formats on the
same layers. An incomplete payload group is a REFUSAL here, at $0, not a
silent under-count that refuses on the pod after the fetch.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from fp8_scope import CLASS_RULES, assignments_from_census, classify, decoder_layers  # noqa: F401

PAYLOAD_OBJECTS = ("trellis", "suh", "svh")
CODEBOOKS = ("mul1", "mcg")
SCALE_SUFFIX = "_scale_inv"


RANK_RE = re.compile(r"^(?P<module>.+)\.rank(?P<rank>\d+)$")


def payload_modules(keys, tp=None):
    """module -> codebook, for every complete stock-exllamav3 payload group.

    With `tp`, rank-sharded groups (`M.rank{r}.{...}`, r in 0..tp-1) count as
    ONE module and must be complete; without it a rank-sharded group refuses.
    """
    staged = defaultdict(dict)
    for key in keys:
        stem, _, last = key.rpartition(".")
        if not stem:
            continue
        if last in PAYLOAD_OBJECTS:
            staged[stem][last] = key
        elif last in CODEBOOKS:
            staged[stem].setdefault("codebooks", []).append(last)
    groups, incomplete = {}, []
    for stem, found in staged.items():
        marks = found.get("codebooks") or []
        if all(name in found for name in PAYLOAD_OBJECTS) and len(marks) == 1:
            groups[stem] = marks[0]
        else:
            incomplete.append("%s (has %s, markers %s)" % (
                stem, sorted(k for k in found if k in PAYLOAD_OBJECTS), sorted(marks)))
    if incomplete:
        raise SystemExit("REFUSED: %d incomplete payload group(s): %s%s"
                         % (len(incomplete), "; ".join(sorted(incomplete)[:3]),
                            " (+%d more)" % (len(incomplete) - 3) if len(incomplete) > 3 else ""))
    modules, ranked = {}, defaultdict(dict)
    for stem, codebook in groups.items():
        match = RANK_RE.match(stem)
        if match:
            ranked[match.group("module")][int(match.group("rank"))] = codebook
        else:
            modules[stem] = codebook
    if ranked and tp is None:
        raise SystemExit(
            "REFUSED: %d module(s) store rank-sharded payloads (e.g. %s) but the "
            "config declares no hybrid_tr3_tail.tp to compose them by."
            % (len(ranked), sorted(ranked)[0]))
    for module, by_rank in ranked.items():
        if sorted(by_rank) != list(range(tp)) or len(set(by_rank.values())) != 1:
            raise SystemExit("REFUSED: %s carries ranks %s / codebooks %s, not 0..%d of one codebook"
                             % (module, sorted(by_rank), sorted(set(by_rank.values())), tp - 1))
        modules[module] = by_rank[0]
    return modules


def shard_dtypes(repo, revision, weight_map):
    """key -> safetensors dtype string, from every shard header (ranged HTTP reads)."""
    import struct
    import urllib.request
    dtypes = {}
    for shard in sorted(set(weight_map.values())):
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
        head = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"})).read(8)
        n = struct.unpack("<Q", head)[0]
        header = json.loads(urllib.request.urlopen(
            urllib.request.Request(url, headers={"Range": "bytes=8-%d" % (8 + n - 1)})).read(n))
        for key, meta in header.items():
            if key != "__metadata__":
                dtypes[key] = meta.get("dtype")
    return dtypes


DTYPE_FORMAT = {"BF16": ("bf16", 16), "F16": ("fp16", 16), "F32": ("fp32", 32),
                "F8_E4M3": ("fp8_e4m3", 8)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtypes-from-hub", action="store_true",
                        help="read every shard header (ranged HTTP) so native classes are "
                             "labelled by their stored dtype (drowzeys carries F16, not "
                             "bf16); without it natives are labelled bf16 as fp8_scope does")
    args = parser.parse_args(argv)

    config = json.load(open(args.config, encoding="utf-8"))
    qc = config.get("quantization_config") or {}
    tail = config.get("hybrid_tr3_tail") or {}
    tail = tail if tail.get("format") == "exl3-trellis" else {}
    if qc.get("quant_method") != "exl3" and not tail:
        raise SystemExit("REFUSED: %s declares quant_method=%r and no hybrid_tr3_tail "
                         "(format exl3-trellis): not an exl3 artifact"
                         % (args.config, qc.get("quant_method")))
    layers = decoder_layers(config)
    weight_map = json.load(open(args.index, encoding="utf-8"))["weight_map"]
    keys = list(weight_map)
    tp = tail.get("tp") if isinstance(tail.get("tp"), int) and tail.get("tp") >= 2 else None

    modules = payload_modules(keys, tp=tp)
    scaled = {k[: -len(SCALE_SUFFIX)] for k in keys if k.endswith(SCALE_SUFFIX)}
    declared_bits = tail.get("bits_avg", tail.get("bits")) if tail else qc.get("bits")
    dtypes = shard_dtypes(args.repo, args.revision, weight_map) if args.dtypes_from_hub else {}
    codebooks = defaultdict(int)
    for codebook in modules.values():
        codebooks[codebook] += 1

    census = defaultdict(list)
    seen_modules = set()
    for key in keys:
        if key.endswith(SCALE_SUFFIX):
            continue
        stem, _, last = key.rpartition(".")
        if last in PAYLOAD_OBJECTS or last in CODEBOOKS:
            match = RANK_RE.match(stem)
            module = match.group("module") if match else stem
            if module in modules and module not in seen_modules:
                seen_modules.add(module)
                census[classify(module + ".weight", layers)].append(
                    (module + ".weight", "quantized", "exl3-%s" % modules[module],
                     float(declared_bits) if declared_bits is not None else None))
            continue
        cls = classify(key, layers)
        if key in scaled:
            census[cls].append((key, "quantized", "fp8_e4m3", 8))
        else:
            fmt, bits = DTYPE_FORMAT.get(dtypes.get(key, "BF16" if not dtypes else "unknown"),
                                         ("unknown", None))
            census[cls].append((key, "native", fmt, bits))
    source = ("read from %s@%s model.safetensors.index.json%s: a class is exl3 trellis "
              "when its weights are stored as %s payload groups (codebook from the object each "
              "module carries: %s; declared bits %r by %s%s), fp8_e4m3 when a %s sibling exists, "
              "native otherwise%s."
              % (args.repo, args.revision, " + shard headers" if dtypes else "",
                 "/".join(PAYLOAD_OBJECTS),
                 ", ".join("%s:%d" % kv for kv in sorted(codebooks.items())),
                 declared_bits, "hybrid_tr3_tail" if tail else "quantization_config",
                 ("; tp=%d rank shards per module, k_values %s, %s"
                  % (tp, tail.get("k_values"), tail.get("bits_scheme"))) if tp else "",
                 SCALE_SUFFIX,
                 " (stored dtype from the shard headers)" if dtypes else " (labelled bf16, headers not read)"))
    assignments = assignments_from_census(census, source)
    head = census.get("lm_head", [])
    # Only the registry scope schema's keys (additionalProperties: false); the
    # codebook census and declaration live in the assignment notes.
    scope = {
        "policy": "mixed",
        "head_policy": ("native" if head and all(t == "native" for _, t, _, _ in head)
                        else "quantized"),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
