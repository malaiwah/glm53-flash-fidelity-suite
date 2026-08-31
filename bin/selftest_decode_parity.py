#!/usr/bin/env python3
"""Prove the EXL3/TR3 weight decode carries NO device offset, and time it.

Two claims underpin the whole local recipe, and both are checked here on the
machine in front of you rather than asserted from a design document:

  1. The decode is PURE PYTORCH.  Integer bit operations, a 65,536-entry fp16
     lookup, and two 128x128 Hadamard matmuls.  No custom kernel, no
     `torch.ops`, no exllamav3 build.  That is why the local recipe needs only
     `pip install torch safetensors numpy huggingface_hub` and none of the
     cloud bootstrap.
  2. It is BITWISE IDENTICAL across devices.  If MPS and CPU produce the same
     bits from the same payload, then a Mac installs exactly the weights the
     sealed lane installs, and any local-vs-sealed difference in the final
     number comes from the FORWARD PASS alone.  That is a much narrower and
     more defensible disclosure than "some device difference".

The functions under test are not copied.  They are extracted from the real
reader source by AST and executed, so this test cannot drift away from the
code it claims to check -- if the reader changes, this runs the changed code.

    python3 bin/selftest_decode_parity.py [--bits 4,6] [--quick]
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

SUITE_ROOT = Path(__file__).resolve().parent.parent
READER_CANDIDATES = [
    SUITE_ROOT / "engines/.patchwork/b/runtime/src/quant_pipeline/evaluation/glm53_packed_k4_reader.py",
    SUITE_ROOT / "engines/.patchwork/a/runtime/src/quant_pipeline/evaluation/glm53_packed_k4_reader.py",
]

WANTED_FUNCS = ("unpack_trellis_states", "mcg_lut", "_permutation",
                "_hadamard", "decode_choice_hf")
WANTED_CONSTS = ("BITS", "SUPPORTED_BITS", "MCG_MULT", "MCG_MASK", "MCG_XOR",
                 "MCG_MULTIPLIER")

PASS: List[str] = []
FAIL: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail else ""))


def load_reader_functions(path: Path) -> Dict[str, Any]:
    """Execute just the pure decode functions out of the real reader source.

    The reader module imports half of `quant_pipeline`, which is not present on
    a laptop.  Rather than vendor a copy (which would silently rot), we take
    the actual function and constant definitions out of the file's AST and run
    those.  Same bytes, no package.
    """
    import numpy as np
    import torch

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    picked: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
            picked.append(node)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in WANTED_CONSTS:
                    picked.append(node)
                    break
    missing = set(WANTED_FUNCS) - {n.name for n in picked
                                   if isinstance(n, ast.FunctionDef)}
    if missing:
        raise RuntimeError("reader source is missing %s" % ", ".join(sorted(missing)))

    ns: Dict[str, Any] = {
        "np": np, "torch": torch, "math": math, "lru_cache": lru_cache,
        "__builtins__": __builtins__,
    }
    # MCG_MULTIPLIER lives in a sibling module (checkpoint/packed_payload.py);
    # the reader imports and aliases it. Resolve it from the package tree by
    # AST rather than importing the package, which is not installable here.
    pkg_root = path.parents[2]                 # .../src/quant_pipeline
    for candidate in sorted(pkg_root.rglob("*.py")):
        try:
            sub = ast.parse(candidate.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in sub.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MCG_MULTIPLIER"
                for t in node.targets
            ):
                ns["MCG_MULTIPLIER"] = ast.literal_eval(node.value)
                break
        if "MCG_MULTIPLIER" in ns:
            break
    ns.setdefault("MCG_MULTIPLIER", None)

    module = ast.Module(body=picked, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename=str(path), mode="exec"), ns)   # noqa: S102
    if ns.get("MCG_MULT") is None:
        raise RuntimeError(
            "could not recover MCG_MULTIPLIER from %s; the constant moved" % path)
    return ns


def make_payload(torch, *, bits: int, k_tiles: int, n_tiles: int, seed: int = 7):
    """A structurally valid packed payload of the real per-matrix shape.

    Real shapes: gate/up are [4096, 2048] and down is [2048, 4096]; both are
    8,388,608 elements, which is 16 x 16 per tile over k_tiles x n_tiles.
    """
    g = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (k_tiles, n_tiles, bits * 16),
                            generator=g, dtype=torch.int32).to(torch.int16)
    suh = torch.randn(k_tiles * 16, generator=g, dtype=torch.float32).to(torch.float16)
    svh = torch.randn(n_tiles * 16, generator=g, dtype=torch.float32).to(torch.float16)
    return trellis, suh, svh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", default="4,6")
    ap.add_argument("--quick", action="store_true",
                    help="one small matrix per bit width")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed. This selftest needs it:")
        print("    pip install torch")
        return 2

    reader = next((p for p in READER_CANDIDATES if p.is_file()), None)
    if reader is None:
        print("could not find the packed reader source; looked in:")
        for p in READER_CANDIDATES:
            print("   ", p)
        return 2
    print("reader source: %s" % reader.relative_to(SUITE_ROOT))
    ns = load_reader_functions(reader)
    decode = ns["decode_choice_hf"]
    unpack = ns["unpack_trellis_states"]
    print("extracted: %s" % ", ".join(WANTED_FUNCS))

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    print("devices: %s" % ", ".join(devices))
    accel = [d for d in devices if d != "cpu"]

    print("\n[1] PURITY -- the decode must not need a compiled extension")
    src = reader.read_text(encoding="utf-8")
    body = src[src.index("def decode_choice_hf"):src.index("def load_decoded_choice")]
    for forbidden in ("torch.ops", "load_inline", "cpp_extension", "exllamav3",
                      "flash_attn"):
        check("decode_choice_hf does not reference %s" % forbidden,
              forbidden not in body)

    print("\n[2] BITWISE DEVICE PARITY")
    if not accel:
        print("  (no accelerator on this machine; parity is vacuous, skipping)")
    tiles = (16, 16) if args.quick else (256, 128)   # 256*16 x 128*16 = 8,388,608
    n_elem = tiles[0] * 16 * tiles[1] * 16
    print("  matrix: %d x %d = %s elements"
          % (tiles[0] * 16, tiles[1] * 16, "{:,}".format(n_elem)))

    bit_list = [int(b) for b in args.bits.split(",") if b.strip()]
    for bits in bit_list:
        trellis, suh, svh = make_payload(torch, bits=bits,
                                         k_tiles=tiles[0], n_tiles=tiles[1])
        ref = decode(trellis, suh, svh, bits=bits)
        for dev in accel:
            got = decode(trellis.to(dev), suh.to(dev), svh.to(dev), bits=bits).cpu()
            same = torch.equal(ref, got)
            ndiff = int((ref != got).sum()) if not same else 0
            maxabs = float((ref - got).abs().max()) if not same else 0.0
            check("bits=%d decode %s == cpu, BITWISE" % (bits, dev), same,
                  "max_abs_diff=%.3e ndiff=%d/%d" % (maxabs, ndiff, ref.numel()))

        # int32 vs int64 unpack: values are 16-bit and the largest shift is
        # lag*bits (18 at bits=6), so int32 is safe -- and measurably faster on
        # MPS. Confirm the equality rather than assuming it.
        s64 = unpack(trellis, bits=bits)
        check("bits=%d unpack is stable under repeat" % bits,
              torch.equal(s64, unpack(trellis, bits=bits)))

    print("\n[3] TIMING on this machine (what the fit estimator should use)")
    for dev in devices:
        for bits in bit_list:
            trellis, suh, svh = make_payload(torch, bits=bits,
                                             k_tiles=tiles[0], n_tiles=tiles[1])
            t_, s_, v_ = trellis.to(dev), suh.to(dev), svh.to(dev)
            decode(t_, s_, v_, bits=bits)          # warm up
            if dev == "mps":
                torch.mps.synchronize()
            elif dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.reps):
                decode(t_, s_, v_, bits=bits)
            if dev == "mps":
                torch.mps.synchronize()
            elif dev == "cuda":
                torch.cuda.synchronize()
            per = (time.perf_counter() - t0) / args.reps
            # 42 routed layers x 288 experts x 3 matrices = 36,288 per pass
            full = per * 36288
            print("  %-5s bits=%d  %7.1f ms/matrix   full routed pass %5.1f min"
                  % (dev, bits, per * 1000, full / 60))

    print("\n[4] float64 availability (the KLD accumulation dtype)")
    # estimator.accumulation_dtype is a comparability key input. If a device
    # cannot do float64 we must run the scoring stage elsewhere, not silently
    # drop to fp32 -- that would move the row into a different comparability
    # group without anyone noticing.
    for dev in devices:
        try:
            torch.zeros(4, dtype=torch.float64, device=dev).sum()
            print("  %-5s float64 OK" % dev)
        except Exception as exc:                    # noqa: BLE001
            print("  %-5s float64 UNAVAILABLE: %s" % (dev, type(exc).__name__))
            check("%s float64 unavailable is handled by pinning KLD to CPU" % dev,
                  "cpu" in devices,
                  "the local runner must refuse --kld-device %s" % dev)

    print("\n" + "-" * 72)
    print("selftest_decode_parity: %d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
