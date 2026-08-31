#!/usr/bin/env python3
"""Rebuild a stock-loadable checkpoint from a repo whose ROUTED EXPERTS are exl3 atoms.

Why this file exists
--------------------
`malaiwah/GLM-5.2-SIQ-Fruit` stores each routed expert projection as four
tensors -- ``.rank0.{trellis,suh,svh,mcg}`` -- instead of a weight matrix.
Stock `transformers` does not implement that storage, and it does **not** fail:
`from_pretrained` reports
``model.layers.{3..12}.mlp.experts.{gate_up,down}_proj`` as MISSING, randomly
initialises them, and hands back a model that runs.  Measured against the bf16
reference, such a model would produce a large, confident, entirely meaningless
KLD.  `engines/tools/hf_capture.py` now refuses that load outright.

The honest way to measure the artifact is therefore to **reconstruct** the
weights the quantizer encoded -- decode the trellis, write plain bf16 tensors --
and capture the reconstruction, declaring `treatment: reconstructed`.  That is
the same "dequantize and run" methodology the GGUF / MLX / EXL3 ecosystems use
for KLD, and it isolates weight error from kernel error rather than mixing them.

What this does NOT claim
------------------------
The reconstruction is not the vendor runtime.  A number measured on it is the
error of the **stored weights**, not of the serving stack that would execute
them (Fruit's production path is b12x/SparkInfer + vLLM with fp8/nvfp4 KV and
MTP, none of which is exercised here).  Every dataset written from this tree
must say so.

Decode provenance
-----------------
The trellis unpack, the tile permutation, the two Hadamard passes and the
suh/svh scaling are `engines/tools/exl3hf_surface.decode_payload_hf`, unchanged.
The `mcg` codebook table is `exl3hf_surface.mcg_lut`, transcribed from
exllamav3 v1.4.2 `codebook.cuh` and verified bitwise against the campaign's
frozen table over all 65,536 entries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3hf_surface as EXL3  # noqa: E402

ATOM_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
_EXPERT = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))"
                     r"\.rank0\.(?P<suffix>%s)$" % "|".join(ATOM_SUFFIXES))


def log(**fields: Any) -> None:
    print(json.dumps(fields, sort_keys=True), flush=True)


def die(message: str) -> "SystemExit":
    print("materialize_exl3_experts: ERROR: %s" % message, file=sys.stderr, flush=True)
    return SystemExit(3)


def group_atoms(keys) -> Dict[str, Dict[str, str]]:
    """module prefix -> {suffix: full tensor key}."""
    groups: Dict[str, Dict[str, str]] = {}
    for key in keys:
        match = _EXPERT.match(key)
        if match:
            groups.setdefault(match.group("prefix"), {})[match.group("suffix")] = key
    return groups


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="materialize_exl3_experts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the exl3-atom checkpoint directory")
    ap.add_argument("--out", required=True, help="the bf16 checkpoint to write")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference", default=None,
                    help="an unquantized checkpoint holding the SAME modules as plain "
                         "tensors. When given, every reconstructed matrix is scored "
                         "against it and the statistics are written to the receipt. "
                         "This is the evidence that the decode is the right decode.")
    ap.add_argument("--tier-bitmap", default=None,
                    help="the producer's own tier_bitmap.json. Its per-expert "
                         "`expert_rel_rt_mse` is an INDEPENDENT record of the encoder's "
                         "reconstruction error; agreeing with it is a decode proof that "
                         "does not depend on us.")
    ap.add_argument("--receipt", default=None, help="where to write the decode receipt")
    ap.add_argument("--limit", type=int, default=None, help="decode only N modules (smoke)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    if os.path.exists(args.out) and not args.force:
        raise die("%s exists (use --force)" % args.out)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    shards = sorted(n for n in os.listdir(args.src) if n.endswith(".safetensors"))
    if not shards:
        raise die("no .safetensors under %s" % args.src)

    reference: Dict[str, str] = {}
    if args.reference:
        for name in sorted(os.listdir(args.reference)):
            if name.endswith(".safetensors"):
                reference[name] = os.path.join(args.reference, name)

    tier: Dict[str, Any] = {}
    if args.tier_bitmap:
        tier = json.loads(open(args.tier_bitmap, "r", encoding="utf-8").read())

    # Non-tensor files travel verbatim; config.json is rewritten below.
    for name in sorted(os.listdir(args.src)):
        full = os.path.join(args.src, name)
        if os.path.isfile(full) and not name.endswith(".safetensors"):
            shutil.copy2(full, os.path.join(args.out, name))

    stats: List[Dict[str, Any]] = []
    bit_histogram: Dict[str, int] = {}
    decoded_total = 0
    started = time.monotonic()

    for shard in shards:
        src_path = os.path.join(args.src, shard)
        tensors = load_file(src_path)
        groups = group_atoms(tensors.keys())
        if not groups:
            shutil.copy2(src_path, os.path.join(args.out, shard))
            log(stage="shard_copied", shard=shard, tensors=len(tensors))
            continue

        ref_tensors: Dict[str, Any] = {}
        if reference.get(shard):
            ref_tensors = load_file(reference[shard])

        out: Dict[str, Any] = {}
        for key, value in tensors.items():
            if not _EXPERT.match(key):
                out[key] = value

        for prefix in sorted(groups):
            if args.limit is not None and decoded_total >= args.limit:
                break
            atoms = groups[prefix]
            missing = [s for s in ("trellis", "suh", "svh") if s not in atoms]
            if missing:
                raise die("%s: missing atom(s) %s" % (prefix, ", ".join(missing)))
            codebook = "mcg" if "mcg" in atoms else "mul1"
            marker = int(tensors[atoms[codebook]].reshape(-1)[0]) \
                if tensors[atoms[codebook]].numel() else None
            expected = EXL3.CODEBOOK_OBJECTS[codebook]
            if marker is not None and marker != expected:
                raise die("%s: %s marker %d != %d -- this is not the codebook it names"
                          % (prefix, codebook, marker, expected))

            trellis = tensors[atoms["trellis"]].to(args.device)
            bits = int(trellis.shape[-1]) // 16
            bit_histogram[str(bits)] = bit_histogram.get(str(bits), 0) + 1
            weight = EXL3.decode_payload_hf(
                trellis, tensors[atoms["suh"]].to(args.device),
                tensors[atoms["svh"]].to(args.device), codebook=codebook)
            out[prefix + ".weight"] = weight.to(torch.bfloat16).cpu().contiguous()
            decoded_total += 1

            ref = ref_tensors.get(prefix + ".weight")
            if ref is not None:
                a = out[prefix + ".weight"].float()
                b = ref.float()
                if a.shape != b.shape:
                    raise die("%s: reconstructed %s vs reference %s"
                              % (prefix, tuple(a.shape), tuple(b.shape)))
                num = float(torch.linalg.vector_norm(a - b).double())
                den = float(torch.linalg.vector_norm(b).double())
                cos = float(torch.nn.functional.cosine_similarity(
                    a.reshape(1, -1).double(), b.reshape(1, -1).double()).item())
                stats.append({"module": prefix, "bits": bits, "codebook": codebook,
                              "rel_l2": num / den if den else None,
                              "rel_mse": (num * num) / (den * den) if den else None,
                              "cosine": cos})
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        log(stage="shard_decoded", shard=shard, modules=len(groups), tensors=len(out))

    # The index must name the reconstructed tensors, not the atoms.
    index_path = os.path.join(args.out, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        index = json.loads(open(index_path, "r", encoding="utf-8").read())
        weight_map = {}
        total = 0
        for shard in sorted(os.listdir(args.out)):
            if not shard.endswith(".safetensors"):
                continue
            full = os.path.join(args.out, shard)
            total += os.path.getsize(full)
            for key in load_file(full).keys():
                weight_map[key] = shard
        index["weight_map"] = weight_map
        index.setdefault("metadata", {})["total_size"] = total
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

    # The reconstruction is bf16 everywhere; a quantization_config left in place
    # would make a reader think the tensors are still encoded.
    config_path = os.path.join(args.out, "config.json")
    removed = None
    if os.path.isfile(config_path):
        config = json.loads(open(config_path, "r", encoding="utf-8").read())
        removed = config.pop("quantization_config", None)
        config.pop("hybrid_tr3_tail", None)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

    summary: Dict[str, Any] = {
        "schema": "malaiwah.exl3-reconstruction-receipt.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "engines/tools/materialize_exl3_experts.py",
        "decoder": {
            "module": "engines/tools/exl3hf_surface.py",
            "function": "decode_payload_hf",
            "codebook_lut_sha256": EXL3.MCG_LUT_SHA256,
            "codebook_source": "exllamav3 v1.4.2 exllamav3/exllamav3_ext/quant/"
                               "codebook.cuh, decode_3inst<1>",
        },
        "modules_decoded": decoded_total,
        "bits_histogram": bit_histogram,
        "quantization_config_removed": removed is not None,
        "reconstruction_vs_reference": None,
        "tier_bitmap_crosscheck": None,
    }

    if stats:
        import statistics

        by_bits: Dict[str, Dict[str, Any]] = {}
        for row in stats:
            bucket = by_bits.setdefault(str(row["bits"]), {"rel_l2": [], "cosine": []})
            bucket["rel_l2"].append(row["rel_l2"])
            bucket["cosine"].append(row["cosine"])
        summary["reconstruction_vs_reference"] = {
            "modules_scored": len(stats),
            "by_bits": {k: {"modules": len(v["rel_l2"]),
                            "rel_l2_mean": statistics.fmean(v["rel_l2"]),
                            "rel_l2_max": max(v["rel_l2"]),
                            "cosine_mean": statistics.fmean(v["cosine"]),
                            "cosine_min": min(v["cosine"])}
                        for k, v in sorted(by_bits.items())},
            "note": "rel_l2 = ||reconstructed - reference|| / ||reference|| in fp64. "
                    "A WRONG codebook or unpack gives cosine near 0; these values are "
                    "the expected trellis reconstruction error at the stated bit rate.",
        }

        if tier:
            # tier_bitmap.json: {layer: {"k": [...], "expert_rel_rt_mse": [...]}}
            pairs = []
            for row in stats:
                m = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", row["module"])
                if not m:
                    continue
                layer, expert = m.group(1), int(m.group(2))
                entry = tier.get(layer)
                if not entry or expert >= len(entry.get("expert_rel_rt_mse", [])):
                    continue
                pairs.append({"module": row["module"],
                              "ours_rel_l2": row["rel_l2"],
                              "producer_expert_rel_rt_mse":
                                  entry["expert_rel_rt_mse"][expert],
                              "producer_k": entry["k"][expert],
                              "ours_bits": row["bits"]})
            if pairs:
                k_agree = sum(1 for p in pairs if p["producer_k"] == p["ours_bits"])
                # The decisive check. `expert_rel_rt_mse` is the ENCODER's own
                # record of ||W_hat - W||^2 / ||W||^2 for that expert, written at
                # quantization time by a program we did not write and cannot see.
                # Our rel_l2 squared is the same quantity measured from the
                # published bytes. If the codebook or the unpack were wrong the
                # two would not be in the same universe.
                for row in pairs:
                    ours = row["ours_rel_l2"] ** 2
                    theirs = row["producer_expert_rel_rt_mse"]
                    row["ours_rel_mse"] = ours
                    row["ratio_ours_over_producer"] = (ours / theirs) if theirs else None
                ratios = [p["ratio_ours_over_producer"] for p in pairs
                          if p["ratio_ours_over_producer"] is not None]
                summary["tier_bitmap_crosscheck"] = {
                    "modules": len(pairs),
                    "bit_rate_agreements": k_agree,
                    "bit_rate_disagreements": len(pairs) - k_agree,
                    "rel_mse_ratio_mean": statistics.fmean(ratios) if ratios else None,
                    "rel_mse_ratio_min": min(ratios) if ratios else None,
                    "rel_mse_ratio_max": max(ratios) if ratios else None,
                    "samples": pairs[:12],
                    "note": "producer_k and producer_expert_rel_rt_mse come from the "
                            "encoder's own tier_bitmap.json. ours_bits is "
                            "trellis.shape[-1] // 16 read off the payload; ours_rel_mse "
                            "is (||reconstructed - reference|| / ||reference||)^2 in "
                            "fp64. A ratio near 1.0 means our decode reproduces the "
                            "reconstruction error the quantizer itself recorded. The "
                            "residual gap is expected: the encoder scored against its "
                            "fp32 source, we score against the bf16 export.",
                }

    log(stage="done", modules=decoded_total, seconds=round(time.monotonic() - started, 1),
        out=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.receipt:
        with open(args.receipt, "w", encoding="utf-8") as handle:
            json.dump({**summary, "per_module": stats}, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
