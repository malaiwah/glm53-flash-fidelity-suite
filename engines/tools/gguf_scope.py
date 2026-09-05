#!/usr/bin/env python3
"""Author a candidate scope file for a llama.cpp GGUF build from its headers.

    engines/tools/gguf_scope.py --repo unsloth/GLM-5.3-GGUF --revision 40hex \\
        --path UD-Q4_K_XL --config official-config.json --out scope.json \\
        [--allowlist-out layer78-unexpected-keys.json]
    engines/tools/gguf_scope.py --file a.gguf --file b.gguf ... --config ... --out ...

Same contract as `fp8_scope.py` / `exl3_scope.py` / `nvfp4_scope.py`, one
surface over: every tensor's ggml type, dims and byte size are read from the
GGUF tensor tables (ranged HTTP against the repo, or local files; never a
weight read), mapped to the official HF name through `gguf_surface`'s proven
glm-dsa / glm5next name map, and classified with the registry's coarse
vocabulary.  Per (tensor_class, layer_range) ONE row:

  * `format` is the registry `numeric_format` the class's ggml types belong
    to: `gguf-i-quant` when any IQ type is present, `gguf-k-quant` when every
    quantized type is a K-quant or Q8_0 (the Flash precedent,
    gguf-evidence/udq4kxl-scope.json), the stored float format when the class
    is natively stored, `mixed` when native and quantized tensors share the
    same layers (the DSA indexer's Q8_0 projections beside its F32 norms);
  * `bits_per_weight` is MEASURED: 8 * bytes / elements over the class from
    ggml's own block traits (Q8_0 is 8.5, Q4_K 4.5, IQ3_XXS 3.0625 ...), never
    the number in the build's directory name; `null` for a `mixed` row, whose
    note carries the measured rate;
  * the note carries the exact ggml type census of the class.

`head_policy` is read from `output.weight`'s own ggml type (Q8_0 -> quantized,
so the comparison runs HEAD-1d own-heads); `mtp_included` is true when the
MTP block ships.  The official config is REQUIRED for glm-dsa: its
`indexer_types` says which layers own a DSA indexer, and the GGUF's indexer
tensors on the other layers are value-identical copies (proven in
gguf-evidence/glmdsa-layout-audit.json) that the model never loads -- they
are excluded from the census rather than counted twice.

`--allowlist-out` also writes the layer-outer unexpected-tensor allowlist for
the MTP block (the official names of every blk.<mtp> tensor, kv_b composed)
plus its `.provenance.json` sidecar, authored from the same header bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gguf_surface as gs  # noqa: E402
from fp8_scope import SCHEMA_ASSIGNMENT_KEYS, range_string  # noqa: E402

HF = "https://huggingface.co"
NATIVE = {"F32": ("fp32", 32.0), "F16": ("fp16", 16.0), "BF16": ("bf16", 16.0)}


def build_files(repo: str, revision: str, path: str):
    """Every .gguf under `path` of the repo at `revision`, as resolve URLs."""
    if not gs._REVISION.fullmatch(revision):
        raise SystemExit("REFUSED: --revision must be the immutable 40-hex commit")
    url = f"{HF}/api/models/{repo}/tree/{revision}/{path.strip('/')}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        rows = json.load(resp)
    names = sorted(r["path"] for r in rows if r.get("type") == "file" and r["path"].endswith(".gguf"))
    if not names:
        raise SystemExit(f"REFUSED: {repo}@{revision} has no .gguf under {path!r}")
    return [f"{HF}/{repo}/resolve/{revision}/{name}" for name in names]


def class_census(surface: gs.GgufSurface):
    """tensor_class -> {types, elements, bytes, layers, top_level, names}."""
    arch = surface.arch
    container, census = surface.container, surface.census
    per = defaultdict(lambda: {"types": defaultdict(int), "elements": 0, "bytes": 0,
                               "layers": set(), "top_level": False, "names": set()})

    def account(hf_name, layer, row):
        cls, _why = gs.scope_class_of(hf_name, layer, arch)
        entry = per[cls]
        per_block, block_bytes = gs.BLOCK_TRAITS[row["type"]]
        entry["types"][row["type"]] += 1
        entry["elements"] += int(row["elements"])
        entry["bytes"] += int(row["elements"]) // per_block * block_bytes
        if layer is None:
            entry["top_level"] = True
        else:
            entry["layers"].add(layer)
        entry["names"].add(hf_name)

    for gguf_name, hf_name in census.direct_map.items():
        match = gs._BLK.match(gguf_name)
        account(hf_name, int(match.group(1)) if match else None, container.tensors[gguf_name])
    for (layer, _half), gguf_name in census.mla.items():
        account(gs.kv_b_hf_name(layer, arch), layer, container.tensors[gguf_name])
    for (layer, projection), gguf_name in census.routed.items():
        account(gs.official_expert_name(layer, 0, projection, arch), layer,
                container.tensors[gguf_name])
    return per


def class_format(types):
    """(treatment, format, nominal-or-None) for a class's ggml type set."""
    quantized = [t for t in types if t not in NATIVE]
    if not quantized:
        formats = {NATIVE[t][0] for t in types}
        return "native", (formats.pop() if len(formats) == 1 else "mixed")
    if len(quantized) != len(types):
        return "quantized", "mixed"
    if any(t.startswith("IQ") for t in quantized):
        return "quantized", "gguf-i-quant"
    return "quantized", "gguf-k-quant"


def assignments(surface: gs.GgufSurface, source: str):
    rows = []
    per = class_census(surface)
    for cls in sorted(per):
        entry = per[cls]
        types = dict(sorted(entry["types"].items()))
        treatment, fmt = class_format(types)
        measured = 8.0 * entry["bytes"] / entry["elements"]
        layer_range = "all" if entry["top_level"] or not entry["layers"] else range_string(entry["layers"])
        shapes = sorted({_generic(n, surface.arch.layer_prefix) for n in entry["names"]})
        note = ("%d GGUF tensors: %s. ggml types: %s. MEASURED %.4f bits/weight over %d weights. %s"
                % (sum(types.values()), ", ".join(shapes[:8]),
                   ", ".join("%s x%d" % (t, n) for t, n in types.items()),
                   measured, entry["elements"], source))
        if fmt == "mixed":
            note = ("class mixes stored formats on the same layers (SCOPE-004 admits one row "
                    "per class and layer_range): " + note)
        if treatment == "native":
            note += (" Stored natively by the artifact; the lane loads it at the official "
                     "release's dtype (bf16 except e_score_correction_bias), a cast the layout "
                     "audit proved exact for every F32 tensor the converter widened.")
        rows.append({
            "tensor_class": cls,
            "treatment": treatment,
            "format": fmt,
            "bits_per_weight": None if fmt == "mixed" else round(measured, 4),
            "layer_range": layer_range,
            "note": note,
        })
    for row in rows:
        assert set(row) <= set(SCHEMA_ASSIGNMENT_KEYS), row
    return rows


def _generic(name: str, layer_prefix: str) -> str:
    name = re.sub(r"^" + re.escape(layer_prefix) + r"\.\d+\.", "layers.N.", name)
    return re.sub(r"experts\.\d+\.", "experts.E.", name)


def mtp_allowlist(surface: gs.GgufSurface):
    """Official names of every MTP-block tensor the artifact ships (kv_b composed)."""
    arch, census = surface.arch, surface.census
    names = [hf for g, hf in census.direct_map.items()
             if g.startswith(f"blk.{arch.mtp_layer}.")]
    if any(layer == arch.mtp_layer for layer, _ in census.mla):
        names.append(gs.kv_b_hf_name(arch.mtp_layer, arch))
    for (layer, projection), _g in census.routed.items():
        if layer == arch.mtp_layer:
            names.extend(gs.official_expert_name(layer, e, projection, arch)
                         for e in range(arch.num_experts))
    return sorted(names)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--file", action="append", dest="files",
                        help="every .gguf of the build (local path or https URL; repeat)")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path", help="build directory inside the repo (lists its .gguf files)")
    parser.add_argument("--config", required=True,
                        help="the official release's config.json (indexer_types)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allowlist-out", help="also write the MTP-layer unexpected-keys allowlist")
    args = parser.parse_args(argv)

    if bool(args.files) == bool(args.path):
        raise SystemExit("REFUSED: give exactly one of --file ... or --path")
    files = args.files or build_files(args.repo, args.revision, args.path)
    config = json.load(open(args.config, encoding="utf-8"))
    container = gs.GgufContainer([gs.GgufFile(f) for f in files])
    arch = gs.arch_for(container.architecture)
    full = gs.indexer_full_layers_from_config(config, arch)
    surface = gs.load_gguf_surface(files, repo=args.repo, revision=args.revision,
                                   require_file_hashes=False, indexer_full_layers=full)
    build = args.path or os.path.basename(os.path.dirname(files[0])) or "?"
    source = ("read from %s@%s %s: every ggml type, dim and byte count from the %d GGUF "
              "tensor tables (%d tensors, arch %s), names mapped through gguf_surface's proven "
              "%s map; bits are 8*bytes/elements from ggml block traits, not the build name"
              % (args.repo, args.revision, build, len(files), len(container.tensors),
                 container.architecture, arch.key))
    rows = assignments(surface, source)
    head = container.tensors.get("output.weight")
    if head is None:
        raise SystemExit("REFUSED: the build ships no output.weight (tied embeddings are not a "
                         "surface this lane measures)")
    quantized = {(r["format"], r["bits_per_weight"]) for r in rows if r["treatment"] == "quantized"}
    scope = {
        "policy": "none" if not quantized else ("uniform" if len(quantized) == 1 else "mixed"),
        "head_policy": "native" if head["type"] in NATIVE else "quantized",
        "kv_cache_dtype": "not_applicable",
        "mtp_included": any(r["tensor_class"] == "mtp" for r in rows),
        "activation_quantization": None,
        "assignments": rows,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(scope, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for row in rows:
        print("%-18s %-9s %-13s %-6s %s" % (row["tensor_class"], row["treatment"], row["format"],
                                           row["bits_per_weight"], row["layer_range"]))
    if args.allowlist_out:
        names = mtp_allowlist(surface)
        body = json.dumps(names, indent=1) + "\n"
        Path(args.allowlist_out).write_text(body, encoding="utf-8")
        canonical = hashlib.sha256(json.dumps(sorted(names), separators=(",", ":")).encode()).hexdigest()
        provenance = {
            "architecture": config.get("architectures", [None])[0],
            "artifact_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "canonical_sorted_names_sha256": canonical,
            "config_sha256": hashlib.sha256(open(args.config, "rb").read()).hexdigest(),
            "count": len(names),
            "decoder_layers": arch.mtp_layer,
            "derived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "derived_by": ("gguf_scope.py: every blk.%d.* tensor of the GGUF tensor tables mapped "
                           "to its official name (attn_k_b+attn_v_b -> kv_b_proj; fused "
                           "ffn_*_exps -> %d per-expert names each); the MTP block is the layer "
                           "transformers never builds" % (arch.mtp_layer, arch.num_experts)),
            "derived_from": "%s@%s %s GGUF headers" % (args.repo, args.revision, build),
            "per_layer_unexpected_counts": {str(arch.mtp_layer): len(names)},
        }
        Path(args.allowlist_out + ".provenance.json").write_text(
            json.dumps(provenance, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("allowlist: %d names -> %s" % (len(names), args.allowlist_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
