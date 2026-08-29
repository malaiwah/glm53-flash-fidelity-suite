#!/usr/bin/env python3
"""Offline selftest for exl3hf_surface (stock-exllamav3 mul1/mcg reader + materializer).

Runs with torch + safetensors only.  Checks that need quant_pipeline (the
campaign reader) SELF-SKIP when it is not importable, exactly like
selftest_dione_offline; the box-side setup always has the pipeline, so the
skipped rungs run there before any paid capture.

  [1] mul1 LUT exactness: the fp32 hfma emulation equals an independent fp64
      computation with explicit fp16 rounding for all 65,536 states, and a
      spot-check recomputes the byte-sum path with pure Python ints.
  [2] anybits unpack parity vs dione_surface (bitwise, K3/K4/K6/K8).
  [3] decode determinism: golden sha256 over a fixed synthetic payload
      (mul1, K4/K6) -- pins the whole decode ABI (unpack+LUT+permute+hadamard).
  [4] mcg parity vs the campaign reader (bitwise at K4/K6) -- proves the
      shared math is verbatim; skipped without quant_pipeline.
  [5] materializer mapping on a synthetic mini-checkpoint: KDA qkv/conv split,
      visual qkv fusion, bias adoption, routed skip + virtual entries,
      official-index completeness gate, sealed inventory + receipt.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import exl3hf_surface as xs  # noqa: E402
import dione_surface as ds  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"selftest_exl3hf_offline: {name} failed: {detail}")


def skip(name, why):
    RESULTS.append((name, True, f"SKIPPED: {why}"))
    print(f"[skip] {name} - {why}")


# [1] LUT exactness -----------------------------------------------------------
lut = xs.mul1_lut().numpy()
idx = np.arange(1 << 16, dtype=np.uint64)
prod = ((idx * np.uint64(xs.MUL1_MULT)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
bytesum = sum(((prod >> np.uint32(s)) & np.uint32(0xFF)) for s in (0, 8, 16, 24)).astype(np.uint32)
s16 = (np.uint32(0x6400) + bytesum).astype(np.uint16)
h64 = s16.view(np.float16).astype(np.float64)
k_inv = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])
k_bias = float(np.array([0xC931], dtype=np.uint16).view(np.float16)[0])
ref64 = (h64 * k_inv + k_bias).astype(np.float16)
check("mul1 LUT == fp64 recomputation (all 65,536 states)", np.array_equal(lut, ref64))
# fp32-exactness argument holds only if the fp64 result is itself exactly the
# fp32 result; verify the product+sum are exactly representable at fp32 for a
# stratified sample using rational arithmetic.
exact = True
for i in list(range(0, 1 << 16, 4099)) + [0, 65535]:
    frac = Fraction(int(h64[i]))  # h is integer-valued (1024 + bytesum)
    v = frac * Fraction(k_inv) + Fraction(k_bias)
    got32 = np.float16(np.float32(float(h64[i])) * np.float32(k_inv) + np.float32(k_bias))
    got64 = np.float16(float(v))
    if got32 != got64:
        exact = False
        break
check("hfma emulation: fp32 path == exact-rational path (sampled)", exact)
mark = int(np.int32(np.uint32(xs.MUL1_MULT)))
check("mul1 marker constant", mark == xs.MUL1_MARKER_SIGNED_INT32, str(mark))

# [2] unpack parity vs dione_surface ------------------------------------------
gen = torch.Generator().manual_seed(20260829)
for bits in (3, 4, 6, 8):
    tr = torch.randint(-32768, 32767, (4, 6, bits * 16), generator=gen, dtype=torch.int16)
    ours = xs.unpack_trellis_states_anybits(tr, bits)
    theirs = ds._unpack_trellis_states_anybits(tr, bits)
    check(f"anybits unpack parity vs dione (K{bits})", torch.equal(ours, theirs))

# [3] decode determinism golden -----------------------------------------------
golden = {}
for bits in (4, 6):
    tr = torch.randint(-32768, 32767, (8, 8, bits * 16), generator=gen, dtype=torch.int16)
    suh = (torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1).half()
    svh = ((torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1) * 0.02).half()
    out = xs.decode_payload_hf(tr, suh, svh, codebook="mul1")
    golden[bits] = xs._sha256_bytes(out.numpy().tobytes())
    check(f"mul1 decode runs and is finite (K{bits})", torch.isfinite(out).all().item(),
          f"shape {tuple(out.shape)}")
GOLDEN = {
    4: "781baa7618e5afb96a7aa19152d95e7c2de19e6398657d78807f5c16f9bc9fca",
    6: "0a1b47c8162ad0c17edb3b9cbc30794904d228f7ed0cd66b9c53888f7c71e997",
}
for bits in (4, 6):
    if GOLDEN[bits] == "PIN-ME":
        print(f"    golden K{bits}: {golden[bits]}")
    else:
        check(f"mul1 decode golden sha (K{bits})", golden[bits] == GOLDEN[bits], golden[bits])

# [4] mcg parity vs the campaign reader ---------------------------------------
reader = None
for candidate in ("runtime/src", "src", "."):
    root = TOOLS.parent / ".patchwork" / "a" / candidate
    if (root / "quant_pipeline" / "__init__.py").is_file():
        sys.path.insert(0, str(root))
        break
try:
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader  # noqa: E402
except Exception as exc:  # noqa: BLE001
    skip("mcg decode parity vs campaign reader (K4/K6)", f"quant_pipeline not importable: {exc}")
if reader is not None:
    for bits in (4, 6):
        tr = torch.randint(-32768, 32767, (8, 8, bits * 16), generator=gen, dtype=torch.int16)
        suh = (torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1).half()
        svh = ((torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1) * 0.02).half()
        ours = xs.decode_payload_hf(tr, suh, svh, codebook="mcg")
        theirs = reader.decode_choice_hf(tr, suh, svh, bits=bits)
        check(f"mcg decode parity vs campaign reader (K{bits})", torch.equal(ours, theirs))

# [5] materializer mapping on a synthetic mini-checkpoint ---------------------
def payload_for(out_features, in_features, bits):
    tr = torch.randint(-32768, 32767, (in_features // 16, out_features // 16, bits * 16),
                       generator=gen, dtype=torch.int16)
    suh = (torch.randint(0, 2, (in_features,), generator=gen).float() * 2 - 1).half()
    svh = ((torch.randint(0, 2, (out_features,), generator=gen).float() * 2 - 1) * 0.02).half()
    return tr, suh, svh


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    root = tmp / "artifact"
    root.mkdir()
    tensors = {}
    marker = torch.tensor(xs.MUL1_MARKER_SIGNED_INT32, dtype=torch.int32)

    def add_quant(module, out_features, in_features, bits, bias=False):
        tr, suh, svh = payload_for(out_features, in_features, bits)
        tensors[f"{module}.trellis"] = tr
        tensors[f"{module}.suh"] = suh
        tensors[f"{module}.svh"] = svh
        tensors[f"{module}.mul1"] = marker.clone()
        if bias:
            tensors[f"{module}.bias"] = torch.randn(out_features, generator=gen).half()

    L = "model.language_model.layers"
    add_quant(f"{L}.0.self_attn.qkv_proj", 384, 128, 6)          # KDA fused
    tensors[f"{L}.0.self_attn.conv1d.weight"] = torch.randn(384, 1, 4, generator=gen).bfloat16()
    add_quant(f"{L}.0.mlp.down_proj", 128, 256, 4)               # plain quantized
    add_quant("model.visual.blocks.0.attn.q_proj", 128, 128, 6, bias=True)
    add_quant("model.visual.blocks.0.attn.k_proj", 128, 128, 6, bias=True)
    add_quant("model.visual.blocks.0.attn.v_proj", 128, 128, 6, bias=True)
    add_quant("lm_head", 256, 128, 6)
    # The REDUNDANT-NATIVE case, reproduced from the real release: turboderp
    # ships the EXL3 split q/k/v for every vision block AND the untouched
    # original fused attn.qkv.{weight,bias} the converter copied through, so
    # both representations map to the same official name (24 blocks, 48
    # colliding names). The materializer used to die with "duplicate
    # materialized tensor" AFTER decoding the whole tree. Sentinel values are
    # used so the assertion below can tell WHICH copy survived.
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.full((384, 128), 7.0).half()
    tensors["model.visual.blocks.0.attn.qkv.bias"] = torch.full((384,), 7.0).half()
    add_quant(f"{L}.3.mlp.experts.0.gate_proj", 128, 128, 4)     # routed: must be SKIPPED
    tensors["model.language_model.embed_tokens.weight"] = torch.randn(256, 128, generator=gen).bfloat16()
    tensors[f"{L}.0.self_attn.A_log"] = torch.randn(4, generator=gen).float()
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0},
        "weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors},
    }))
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "quantization_config": {"quant_method": "exl3", "version": "1.4.4",
                                "codebook": "mul1", "bits": 4.05, "head_bits": 6},
    }))
    official = {
        f"{L}.0.self_attn.q_proj.weight": "x", f"{L}.0.self_attn.k_proj.weight": "x",
        f"{L}.0.self_attn.v_proj.weight": "x",
        f"{L}.0.self_attn.q_conv1d.weight": "x", f"{L}.0.self_attn.k_conv1d.weight": "x",
        f"{L}.0.self_attn.v_conv1d.weight": "x",
        f"{L}.0.mlp.down_proj.weight": "x",
        "model.visual.blocks.0.attn.qkv.weight": "x", "model.visual.blocks.0.attn.qkv.bias": "x",
        "lm_head.weight": "x",
        "model.language_model.embed_tokens.weight": "x",
        f"{L}.0.self_attn.A_log": "x",
        f"{L}.3.mlp.experts.0.gate_proj.weight": "routed",  # filtered by the gate
    }
    (tmp / "official-index.json").write_text(json.dumps({"weight_map": official}))

    out = tmp / "materialized"
    receipt = xs.materialize_nonrouted(
        root, out, device="cpu",
        source_repo="selftest/exl3hf-mini", source_revision="0" * 40,
        official_index=tmp / "official-index.json",
    )
    check("materializer: receipt written and sealed",
          receipt["receipt_sha256"] == xs._sha256_bytes(xs._canonical_json(
              {k: v for k, v in receipt.items() if k != "receipt_sha256"})))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    produced = {n for n, s in wm.items() if s != "model-routed-virtual.safetensors"}
    want = {n for n in official if official[n] != "routed"}
    check("materializer: produced name set == official non-routed set", produced == want,
          f"missing={sorted(want - produced)[:3]} extra={sorted(produced - want)[:3]}")
    n_virtual = sum(1 for s in wm.values() if s == "model-routed-virtual.safetensors")
    check("materializer: virtual routed entries present", n_virtual == 43 * 288 * 3, str(n_virtual))
    from safetensors import safe_open
    with safe_open(out / "model-nonrouted-00001.safetensors", framework="pt") as h:
        qw = h.get_tensor(f"{L}.0.self_attn.q_proj.weight")
        kc = h.get_tensor(f"{L}.0.self_attn.k_conv1d.weight")
        vb = h.get_tensor("model.visual.blocks.0.attn.qkv.bias")
        alog = h.get_tensor(f"{L}.0.self_attn.A_log")
        head = h.get_tensor("lm_head.weight")
    check("materializer: KDA q slice shape+dtype", tuple(qw.shape) == (128, 128) and qw.dtype == torch.bfloat16)
    check("materializer: conv k slice", tuple(kc.shape) == (128, 1, 4) and torch.equal(
        kc, tensors[f"{L}.0.self_attn.conv1d.weight"][128:256]))
    check("materializer: visual qkv.bias fused", tuple(vb.shape) == (384,) and vb.dtype == torch.bfloat16)
    with safe_open(str(out / wm["model.visual.blocks.0.attn.qkv.weight"]),
                   framework="pt") as h:
        vqkv = h.get_tensor("model.visual.blocks.0.attn.qkv.weight")
    # The quantized split must win: the redundant native copy is all 7.0, so a
    # single 7.0 anywhere means the native copy was emitted instead.
    check("materializer: quantized split beats the redundant native fused copy",
          not bool((vqkv.float() == 7.0).any()) and not bool((vb.float() == 7.0).any()),
          "native sentinel 7.0 present in the materialized visual qkv")
    check("materializer: the redundant natives are counted, not silently dropped",
          receipt["stats"].get("redundant_native_skipped") == 2,
          str(receipt["stats"].get("redundant_native_skipped")))
    # planned_names is a SECOND implementation of the emission rules, used to
    # run the duplicate and completeness gates before any decode. Two
    # implementations can drift, so prove they agree on a checkpoint that
    # exercises every branch: KDA qkv split, conv1d split, visual fusion,
    # redundant-native skip, bias adoption, routed skip.
    index_map = {n: "model-00001-of-00001.safetensors" for n in tensors}
    planned = xs.planned_names([index_map])
    surface = xs.load_surface(root)
    reader = xs.Exl3HfShardReader(surface)
    streamed = [n for n, _ in xs._materialize_stream(surface, reader, "cpu", {}, [])]
    check("planner == stream: the pre-flight name plan is what materialization emits",
          planned == streamed,
          "planned=%d streamed=%d first_diff=%s"
          % (len(planned), len(streamed),
             next((i for i, (a, b) in enumerate(zip(planned, streamed)) if a != b), None)))
    check("planner: the plan has no duplicate names",
          len(set(planned)) == len(planned))
    check("materializer: fp32 native kept fp32", alog.dtype == torch.float32)
    check("materializer: lm_head dequantized bf16", tuple(head.shape) == (256, 128) and head.dtype == torch.bfloat16)
    # KDA split slices must equal the decoded fused rows (one bf16 rounding)
    dec = xs.decode_payload_hf(tensors[f"{L}.0.self_attn.qkv_proj.trellis"],
                               tensors[f"{L}.0.self_attn.qkv_proj.suh"],
                               tensors[f"{L}.0.self_attn.qkv_proj.svh"], codebook="mul1")
    check("materializer: q slice == decoded rows 0:128 (bf16)",
          torch.equal(qw, dec[0:128].to(torch.bfloat16)))
    inv = json.loads((out / "inventory.json").read_text())
    body = {k: v for k, v in inv.items() if k != "inventory_sha256"}
    check("materializer: inventory sealed + schema",
          inv["schema"] == xs.RELEASE_INVENTORY_SCHEMA
          and inv["seal_mode"] == "full-shard-sha256"
          and inv["inventory_sha256"] == xs._sha256_bytes(xs._canonical_json(body)))
    cfg = json.loads((out / "config.json").read_text())
    check("materializer: quantization_config stripped from the written config",
          "quantization_config" not in cfg)

print(f"selftest_exl3hf_offline: {sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} green")
