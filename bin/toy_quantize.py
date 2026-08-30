#!/usr/bin/env python3
"""A deliberately crude quantizer, so the three-step architecture has a CANDIDATE.

This is NOT a quantization method.  It is round-to-nearest with a per-output-row
scale, applied to one named class of tensors and to nothing else, so that a
`fidelity-dataset compare` between the reference capture and the candidate
capture has a real, nonzero, explainable KLD.  It exists because proving the
comparison path needs two checkpoints, not because anyone should quantize this
way: there is no calibration, no error feedback, no group structure, no outlier
handling, and no attempt to preserve anything.

    w_hat[i, j] = round(w[i, j] / s[i]) * s[i],   s[i] = max_j |w[i, j]| / (2^(bits-1) - 1)

Everything the pattern does not match is copied through byte-for-byte, and the
emitted `--emit-scope` document says exactly that, so the candidate capture's
scope block is the truth rather than a label.

    bin/toy_quantize.py --src REF --dst CAND --bits 4 --match '.mlp.experts.' \\
        --emit-scope scope.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

SCHEME = ("round-to-nearest, per-output-row absmax scale, no calibration, no error "
          "feedback, no grouping, no outlier handling; dequantized back to the source "
          "dtype and stored as a normal safetensors checkpoint")


def quantize(src, dst, bits, patterns, dry_run=False):
    import torch
    from safetensors.torch import load_file, save_file

    shards = sorted(name for name in os.listdir(src) if name.endswith(".safetensors"))
    if not shards:
        raise SystemExit("toy_quantize: no *.safetensors in %s" % src)
    if not dry_run:
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            if not name.endswith(".safetensors"):
                full = os.path.join(src, name)
                if os.path.isfile(full):
                    shutil.copy2(full, os.path.join(dst, name))
    levels = 2 ** (bits - 1) - 1
    touched, skipped, total_elements = [], 0, 0
    for shard in shards:
        tensors = load_file(os.path.join(src, shard))
        for key in sorted(tensors):
            value = tensors[key]
            if not any(pattern in key for pattern in patterns) or value.ndim != 2:
                skipped += 1
                continue
            wide = value.float()
            scale = wide.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / levels
            tensors[key] = (torch.round(wide / scale) * scale).to(value.dtype)
            touched.append({"tensor": key, "shape": list(value.shape),
                            "elements": int(value.numel())})
            total_elements += int(value.numel())
        if not dry_run:
            save_file(tensors, os.path.join(dst, shard))
        del tensors
    return {"bits": bits, "patterns": list(patterns), "scheme": SCHEME,
            "tensors_quantized": len(touched), "tensors_untouched": skipped,
            "elements_quantized": total_elements, "quantized": touched}



def _registry_format(bits):
    """Map the toy scheme onto the registry's `numeric_format` vocabulary."""
    return {8: "int8", 4: "int4"}.get(bits, "unknown")


def scope_document(report, tensor_class, quantized_classes=None):
    """The scope block the candidate capture must declare.

    Every class NOT in `quantized_classes` is native, and says so.  A capture
    that quantizes eight tensors and declares a uniform policy is a lie the
    comparator cannot see.
    """
    quantized_classes = set(quantized_classes or [tensor_class])
    classes = ["embed_tokens", "attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.down",
               "moe.experts", "norm", "lm_head"]
    assignments = []
    for name in classes:
        if name in quantized_classes:
            # The FORMAT string must come from the registry's numeric_format
            # enum, or the submission this artifact eventually generates is
            # rejected on schema before anyone looks at the number. The exact
            # scheme goes in `_scheme` and in the capture's disclosures, where
            # free text belongs.
            assignments.append({"tensor_class": name, "treatment": "quantized",
                                "format": _registry_format(report["bits"]),
                                "bits_per_weight": report["bits"], "layer_range": None})
        else:
            assignments.append({"tensor_class": name, "treatment": "native",
                                "format": "bf16", "bits_per_weight": 16,
                                "layer_range": None})
    return {"policy": "mixed", "head_policy": "native", "kv_cache_dtype": "bf16",
            "assignments": assignments, "_scheme": report["scheme"],
            "_match_patterns": report["patterns"],
            "_tensors_quantized": report["tensors_quantized"],
            "_elements_quantized": report["elements_quantized"]}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toy_quantize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--match", action="append", required=True,
                        help="substring a tensor name must contain to be quantized "
                             "(repeatable)")
    parser.add_argument("--tensor-class", default="moe.experts",
                        help="the scope tensor_class these tensors belong to")
    parser.add_argument("--emit-scope", default=None)
    parser.add_argument("--emit-report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = quantize(args.src, args.dst, args.bits, args.match, dry_run=args.dry_run)
    if args.emit_scope:
        with open(args.emit_scope, "w", encoding="utf-8") as handle:
            json.dump(scope_document(report, args.tensor_class), handle, indent=2,
                      sort_keys=True)
    if args.emit_report:
        with open(args.emit_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    summary = dict(report)
    summary.pop("quantized", None)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not report["tensors_quantized"]:
        print("toy_quantize: REFUSED: no tensor matched %s -- a 'candidate' identical to "
              "its reference would compare to exactly 0.0 and read as a reproduction"
              % args.match, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
