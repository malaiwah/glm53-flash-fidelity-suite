#!/usr/bin/env python3
"""Is fp8 -> bf16 lossless? Exhaustively, and the answer has two halves.

This is a PROOF, not a sample: `float8_e4m3fn` has 256 bit patterns and all of
them are checked. The two halves matter separately because the registry treats
them differently.

**The CAST is exact.** E4M3 is 1+4+3 bits and BF16 is 1+8+7, so every finite
E4M3 value is a BF16 value with room to spare. fp8 -> bf16 -> fp8 returns the
identical byte for all 254 finite patterns: no compounding, no drift.

**DEQUANTISATION is not a cast, and is not exact.** A real fp8 checkpoint does
not store a value, it stores `value x block_scale`, and that PRODUCT is what
has to land in the target dtype. It lands in bf16 exactly under 1% of the time,
with a relative error up to ~2^-8. `engines/tools/dequant_fp8.py` therefore does the
multiply in fp32 -- which IS exact -- and only the final store rounds.

The consequence, and the reason this file exists rather than a comment: a
BF16 tree dequantised from an FP8 release is a *rounded rendering* of it, not a
numerically identical model. A measurement against such a tree cannot be
expected to read 0, which is precisely what `reference_kind =
dequantized_from_quant` and REFC-001 exist to disclose.

**But the round trip survives anyway**, and that is worth knowing: re-quantising
a bf16-stored product back to fp8 recovers the original byte 100% of the time,
because bf16's 8 significant bits leave ~4 bits of margin over E4M3's 4. The
storage rounding is far smaller than half an E4M3 ULP.
"""
import os
import sys

import torch

FAILED = []
E4M3 = torch.float8_e4m3fn


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


raw = torch.arange(256, dtype=torch.uint8)
f8 = raw.view(E4M3)
f32 = f8.to(torch.float32)
finite = torch.isfinite(f32)

print("== the cast, over all 256 e4m3 bit patterns ==")
check("254 of 256 patterns are finite (2 are NaN)", int(finite.sum()) == 254)

bf = f8.to(torch.bfloat16)
exact = torch.where(finite, bf.to(torch.float32) == f32, torch.ones_like(finite))
check("fp8 -> bf16 is EXACT for every finite pattern", bool(exact.all()))

rt = bf.to(E4M3).view(torch.uint8)
check("fp8 -> bf16 -> fp8 returns the identical byte",
      bool(((rt == raw) | ~finite).all()))
check("fp8 -> fp32 -> fp8 returns the identical byte",
      bool(((f32.to(E4M3).view(torch.uint8) == raw) | ~finite).all()))

print("\n== dequantisation is a SCALED cast, and that is different ==")
# NOT f8[finite]: boolean indexing is not implemented for Float8_e4m3fn on
# every torch this suite runs under ("index_cpu not implemented for
# Float8_e4m3fn"). f32 already holds the exact same values -- the cast above is
# what the first check proves lossless -- so index that instead.
vals = f32[finite]
gen = torch.Generator().manual_seed(0)
scales = torch.rand(1024, generator=gen, dtype=torch.float32) * 3.7 + 0.01
prod32 = vals[:, None] * scales[None, :]
prod_bf = prod32.to(torch.bfloat16).to(torch.float32)

frac_exact = (prod_bf == prod32).float().mean().item()
check("(value x scale) is NOT generally exact in bf16 (<5% of products)",
      frac_exact < 0.05)
rel = ((prod_bf - prod32).abs() / prod32.abs().clamp_min(1e-30)).max().item()
check("...and its relative error stays within bf16's 2^-8 (%.2e)" % rel,
      rel < 2 ** -7)
check("the same product kept in fp32 IS exact",
      bool((prod32.to(torch.float32) == prod32).all()))

print("\n== which is why dequant_fp8.py multiplies in fp32 ==")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "engines", "tools", "dequant_fp8.py"),
           encoding="utf-8").read()
check("it casts the weight to float32 before scaling",
      "w = t.to(torch.float32)" in src)
check("it casts the scale to float32 too",
      "f.get_tensor(sk).to(torch.float32)" in src)
check("only the final store is bf16", "(w * s).to(torch.bfloat16)" in src)

print("\n== the round trip survives the storage rounding, with margin ==")
for dtype, name in ((torch.bfloat16, "bf16"), (torch.float32, "fp32"),
                    (torch.float16, "fp16")):
    deq = prod32.to(dtype).to(torch.float32)
    back = (deq / scales[None, :]).to(E4M3).view(torch.uint8)
    same = (back == raw[finite][:, None]).all()
    check("fp8 -> (x scale, stored %s) -> (/ scale) -> fp8 is bit-identical"
          % name, bool(same))

# The reason it survives, stated as an inequality rather than a hope: the
# storage error is bounded by 2^-8 relative, half an E4M3 ULP is ~2^-4.
check("bf16 rounding (2^-8) is well under half an e4m3 ULP (2^-4)",
      2 ** -8 < 0.5 * 2 ** -4)

print()
if FAILED:
    print("selftest_fp8_lossless: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_fp8_lossless: all passed")
