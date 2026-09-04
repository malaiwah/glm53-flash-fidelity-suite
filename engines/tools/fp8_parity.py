#!/usr/bin/env python3
"""Real-tensor parity for the layer-outer FP8 block decoder.

    engines/tools/fp8_parity.py --shard <model-000NN-of-00141.safetensors> \
        --config <config.json> --repo zai-org/GLM-5.3 --revision <40-hex> \
        --out engines/tools/fp8-evidence/<name>.json

For every (weight, weight_scale_inv) pair in one REAL fetched shard, decode
with `layer_outer.dequantize_block_fp8` and with transformers' own
`Fp8Dequantize._dequantize_one`, and refuse unless the two bf16 tensors are
bitwise identical. The receipt records the shard's bytes and digest, the
config's quantization block, every tensor compared with its shape and the
digest of the decoded bytes, and the transformers/torch versions -- the
evidence the decode rule in AGENTS.md asks for before a surface ships.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bin"))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shard", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="compare at most N pairs (0 = every pair in the shard)")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from safetensors import safe_open
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    import layer_outer as LO
    from fidelity import common

    config = json.load(open(args.config, encoding="utf-8"))
    plan = LO.fp8_checkpoint_plan(type("Cfg", (), {"quantization_config": config["quantization_config"]})())
    reference = Fp8Dequantize(None)
    started = time.monotonic()
    rows = []
    mismatched = []
    with safe_open(args.shard, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        pairs = [k for k in keys if k + LO.FP8_SCALE_SUFFIX in keys]
        fp8_without_scale = [k for k in keys
                             if handle.get_slice(k).get_dtype() == "F8_E4M3"
                             and k + LO.FP8_SCALE_SUFFIX not in keys]
        if fp8_without_scale:
            print("REFUSED: fp8 tensors without a scale in this shard: %s"
                  % fp8_without_scale[:5])
            return 3
        if args.limit:
            pairs = pairs[:args.limit]
        block_m, block_n = plan["weight_block_size"]
        partial = 0
        for key in pairs:
            quantized = handle.get_tensor(key)
            scales = handle.get_tensor(key + LO.FP8_SCALE_SUFFIX)
            ours = LO.dequantize_block_fp8(quantized, scales, torch.bfloat16, (block_m, block_n))
            rows_n, cols_n = quantized.shape
            pad_rows = scales.shape[0] * block_m - rows_n
            pad_cols = scales.shape[1] * block_n - cols_n
            if pad_rows or pad_cols:
                # The reference refuses a partial block outright; the same
                # arithmetic on the zero-padded tensor, cropped, is the kernel
                # rule (an element's value never depends on its neighbours).
                partial += 1
                padded = torch.nn.functional.pad(
                    quantized.to(torch.float32), (0, pad_cols, 0, pad_rows)).to(torch.float8_e4m3fn)
                theirs = reference._dequantize_one(
                    padded, scales, output_dtype=torch.bfloat16)[:rows_n, :cols_n]
            else:
                theirs = reference._dequantize_one(quantized, scales, output_dtype=torch.bfloat16)
            equal = (ours.dtype == theirs.dtype == torch.bfloat16
                     and ours.shape == theirs.shape
                     and torch.equal(ours.contiguous().view(torch.int16),
                                     theirs.contiguous().view(torch.int16)))
            decoded_sha = hashlib.sha256(ours.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            rows.append({
                "key": key, "shape": list(quantized.shape), "scale_shape": list(scales.shape),
                "partial_block": bool(pad_rows or pad_cols),
                "fp8_dtype": str(quantized.dtype), "scale_dtype": str(scales.dtype),
                "decoded_bf16_sha256": decoded_sha, "bitwise_equal": bool(equal),
            })
            if not equal:
                mismatched.append(key)
    receipt = {
        "schema": "fidelity-suite/fp8-decode-parity.v1",
        "receipt_sha256": "",
        "checked_at": common.utcnow(),
        "artifact": {"repo_id": args.repo, "revision": args.revision,
                     "shard": os.path.basename(args.shard),
                     "shard_bytes": os.path.getsize(args.shard),
                     "shard_sha256": sha256_file(args.shard),
                     "config_sha256": sha256_file(args.config)},
        "decoder": {"file": "engines/tools/layer_outer.py",
                    "sha256": sha256_file(os.path.join(HERE, "layer_outer.py")),
                    "method": LO.FP8_DECODE_METHOD, "plan": plan},
        "reference": {"name": LO.FP8_DECODE_REFERENCE,
                      "transformers_version": transformers.__version__,
                      "torch_version": torch.__version__},
        "pairs_compared": len(rows), "pairs_in_shard": len(pairs),
        "pairs_with_partial_blocks": partial,
        "bitwise_equal_all": not mismatched, "mismatched": mismatched,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "tensors": rows,
    }
    receipt = common.seal(receipt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("%s: %d/%d pairs bitwise equal (%s); receipt %s"
          % ("PARITY" if not mismatched else "MISMATCH", len(rows) - len(mismatched),
             len(rows), receipt["artifact"]["shard"], args.out))
    return 0 if not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())
