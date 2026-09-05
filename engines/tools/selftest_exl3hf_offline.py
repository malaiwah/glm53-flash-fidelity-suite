#!/usr/bin/env python3
"""Offline selftest for exl3hf_surface (stock-exllamav3 mul1/mcg reader + materializer).

Runs with torch + safetensors only.  Checks that need quant_pipeline (the
campaign reader) SELF-SKIP when it is not importable, exactly like
selftest_dione_offline; the box-side setup always has the pipeline, so the
skipped rungs run there before any paid capture.

  [1] mul1 LUT exactness: the fp32 hfma emulation equals an independent fp64
      computation with explicit fp16 rounding for all 65,536 states, and a
      spot-check recomputes the byte-sum path with pure Python ints.
  [1b] mcg LUT: pinned by a frozen digest and recomputed by an independent
      pure-integer route; needs NO private package.
  [2] anybits unpack parity vs dione_surface (bitwise, K3/K4/K6/K8).
  [3] decode parity against an independent fp64 reference (mul1, K4/K6),
      pinning unpack+LUT+permute+hadamard without requiring CPU BLAS guard
      bits to be identical across platforms.
  [4] mcg parity vs the campaign reader (bitwise at K4/K6) -- proves the
      shared math is verbatim; skipped without quant_pipeline.
  [6] K2 codec evidence (M4): the anybits unpack inverts the exllamav3
      pack.cu transliteration at K2, agrees with dione's copy, and the MCG K2
      decode agrees with the same independent fp64 reference.
  [5] materializer mapping on a synthetic mini-checkpoint: KDA qkv/conv split,
      visual qkv fusion, bias adoption, routed skip + virtual entries,
      official-index completeness gate, sealed inventory + receipt.
  [7] decoder parity vs exllamav3 ITSELF: every rung above compares our decoder
      with in-house code or a transliteration (review S2-4). The receipt
      layer-outer-evidence/exl3-decoder-parity-vs-exllamav3.json is produced by
      exl3_decoder_parity_vs_exllamav3.py on a CUDA host (exllamav3 1.4.2 has
      no CPU reconstruct). Here: the committed 8x8-tile windows re-decode to the
      committed digests and equal exllamav3's committed pre-Hadamard digests
      bitwise on ANY host; where `import exllamav3` succeeds with a CUDA device,
      exllamav3 reconstructs the windows again and the result is re-asserted.
      SKIPS loudly (never silently) when the receipt is absent or exllamav3
      cannot run.
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


def reference_decode_hf(trellis, suh, svh, codebook):
    """Independent fp64 transcription of EXL3's permute + two Hadamards."""
    states = xs.unpack_trellis_states_anybits(trellis, trellis.shape[-1] // 16)
    indices = states.numpy().astype(np.int64) & 0xFFFF
    values = xs.codebook_lut(codebook).numpy().astype(np.float64)[indices]

    permutation = [0] * 256
    for thread in range(32):
        rows = (
            (thread % 4) * 2,
            (thread % 4) * 2 + 1,
            (thread % 4) * 2 + 8,
            (thread % 4) * 2 + 9,
        )
        columns = (thread // 4, thread // 4 + 8)
        pairs = [(row, column) for column in columns for row in rows]
        for offset, (row, column) in enumerate(pairs):
            permutation[thread * 8 + offset] = row * 16 + column
    values = values[..., np.argsort(np.asarray(permutation))]
    k_tiles, n_tiles, _ = values.shape
    exl = (
        values.reshape(k_tiles, n_tiles, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(k_tiles * 16, n_tiles * 16)
    )

    hadamard = np.ones((1, 1), dtype=np.float64)
    while hadamard.shape[0] < 128:
        hadamard = np.block(
            [[hadamard, hadamard], [hadamard, -hadamard]])
    hadamard *= 1.0 / math.sqrt(128.0)
    left = np.matmul(
        hadamard, exl.reshape(-1, 128, exl.shape[1])).reshape(exl.shape)
    left *= suh.float().numpy().astype(np.float64).reshape(-1, 1)
    right = np.matmul(
        left.reshape(left.shape[0], -1, 128), hadamard).reshape(exl.shape)
    right *= svh.float().numpy().astype(np.float64).reshape(1, -1)
    return np.ascontiguousarray(right.T)


def check_decode_reference(name, got, trellis, suh, svh, codebook):
    reference = reference_decode_hf(trellis, suh, svh, codebook)
    actual = got.float().numpy().astype(np.float64)
    delta = np.abs(actual - reference)
    # Two 128-term fp32 reductions have gamma_128 ~= 7.6e-6 each. The
    # tolerance is >10x that forward-error scale, but far below errors from a
    # wrong LUT, permutation, orientation, or scale axis.
    close = np.allclose(actual, reference, rtol=2e-4, atol=2e-5)
    check(name, close,
          "max_abs=%.3e max_rel=%.3e" % (
              float(delta.max()),
              float(np.max(delta / np.maximum(np.abs(reference), 1e-12)))))


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

# [1b] mcg LUT: independent recomputation + frozen digest -----------------------
# The mcg table used to come from `quant_pipeline`, which is not published, so a
# fresh clone could decode `mul1` releases and nothing else. It is now built
# in-tree from exllamav3 v1.4.2 codebook.cuh `decode_3inst<1>`. This rung needs
# no private package: it recomputes the table by a DIFFERENT route (pure Python
# integers, one state at a time) and pins the whole table by digest.
mcg = xs.mcg_lut().numpy()
check("mcg LUT is 65,536 fp16 entries",
      mcg.shape == (1 << 16,) and mcg.dtype == np.float16)
check("mcg LUT matches its frozen digest",
      xs._sha256_bytes(np.ascontiguousarray(mcg).tobytes()) == xs.MCG_LUT_SHA256)
bad = []
for i in list(range(0, 1 << 16, 2591)) + [0, 1, 65535]:
    # x *= 0xCBAC1FED; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; hadd(lo_fp16, hi_fp16)
    x = (i * 0xCBAC1FED) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = np.array([x & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
    hi = np.array([(x >> 16) & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
    if np.float16(lo + hi) != mcg[i]:
        bad.append(i)
check("mcg LUT == pure-integer recomputation (stratified states)",
      not bad, f"disagreed at {bad[:5]}")
check("mcg LUT is finite and symmetric-ranged",
      bool(np.isfinite(mcg).all()) and abs(float(mcg.min()) + float(mcg.max())) < 1e-3,
      f"min={float(mcg.min())} max={float(mcg.max())}")
check("the mcg path no longer imports the unpublished campaign package",
      "quant_pipeline" not in (TOOLS / "exl3hf_surface.py").read_text()
      .split("def codebook_lut")[1])

# [2] unpack parity vs dione_surface ------------------------------------------
gen = torch.Generator().manual_seed(20260829)
for bits in (3, 4, 6, 8):
    tr = torch.randint(-32768, 32767, (4, 6, bits * 16), generator=gen, dtype=torch.int16)
    ours = xs.unpack_trellis_states_anybits(tr, bits)
    theirs = ds._unpack_trellis_states_anybits(tr, bits)
    check(f"anybits unpack parity vs dione (K{bits})", torch.equal(ours, theirs))

# [3] materialized decode against an independent reference --------------------
# PyTorch's fp32 SGEMM guard bits are not a cross-platform ABI: x86/Linux,
# arm64/Linux, and macOS can produce different raw and even once-rounded BF16
# digests from the same two Hadamard products. Test the algorithm instead:
# a separate fp64 transcription with a forward-error tolerance derived above.
for bits in (4, 6):
    tr = torch.randint(-32768, 32767, (8, 8, bits * 16), generator=gen, dtype=torch.int16)
    suh = (torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1).half()
    svh = ((torch.randint(0, 2, (8 * 16,), generator=gen).float() * 2 - 1) * 0.02).half()
    out = xs.decode_payload_hf(tr, suh, svh, codebook="mul1")
    check(f"mul1 decode runs and is finite (K{bits})", torch.isfinite(out).all().item(),
          f"shape {tuple(out.shape)}")
    check_decode_reference(
        f"mul1 decode agrees with independent fp64 reference (K{bits})",
        out, tr, suh, svh, "mul1")

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
# [5] below rebinds the bare name `reader` to an Exl3HfShardReader, so any later
# rung that asks "is the campaign reader available?" through `reader` gets an
# answer produced by something that is not the campaign reader.  That is the
# M1 probe lesson in one variable name; bind it under a name nothing reuses.
CAMPAIGN_READER = reader
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


# [6] K2 codec evidence ------------------------------------------------------
# M4 measures vcruz305/GLM-5.3-Flash-EXL3-K2, the first K2 artifact on this
# lane.  Every rung above pinned K3/K4/K6/K8; a rate that has never been
# exercised is a rate whose number nobody may publish.  The rungs use their own
# generator so nothing above them is re-rolled.
#
# The strongest available offline evidence is not self-consistency: it is that
# our unpack INVERTS the real exllamav3 packer.  selftest_dione_offline carries
# a numpy transliteration of quant/pack.cu, so import it and run K2 through it.
# It defines the packer at module scope and guards main(), so importing costs
# nothing and needs no quant_pipeline.
import selftest_dione_offline as sdo  # noqa: E402

k2gen = torch.Generator().manual_seed(20260830)
rng2 = np.random.default_rng(0x4B32)
for bits in (2,):
    states = sdo.tail_biting_states(rng2, tiles=48, bits=bits)
    packed = sdo.pack_trellis_reference(states, bits)
    packed_t = torch.from_numpy(packed.astype(np.int16)).reshape(6, 8, bits * 16)
    want = torch.from_numpy(states.astype(np.int16)).reshape(6, 8, 256)
    ours = xs.unpack_trellis_states_anybits(packed_t, bits)
    theirs = ds._unpack_trellis_states_anybits(packed_t, bits)
    check(f"K{bits}: anybits unpack inverts the exllamav3 pack.cu transliteration",
          torch.equal(ours, want))
    check(f"K{bits}: anybits unpack parity vs dione", torch.equal(ours, theirs))

# K2 decode reference: covers unpack + LUT + permutation + Hadamards at the
# rate M4 publishes without treating one CPU BLAS implementation as the ABI.
tr2 = torch.randint(-32768, 32767, (8, 8, 2 * 16), generator=k2gen, dtype=torch.int16)
suh2 = (torch.randint(0, 2, (8 * 16,), generator=k2gen).float() * 2 - 1).half()
svh2 = ((torch.randint(0, 2, (8 * 16,), generator=k2gen).float() * 2 - 1) * 0.02).half()
out2_mcg = xs.decode_payload_hf(tr2, suh2, svh2, codebook="mcg")
out2_mul1 = xs.decode_payload_hf(tr2, suh2, svh2, codebook="mul1")
check("K2 mcg decode is finite", torch.isfinite(out2_mcg).all().item(),
      f"shape {tuple(out2_mcg.shape)}")
check("K2 mcg and mul1 decodes differ (the codebook is load-bearing at K2 too)",
      not torch.equal(out2_mcg, out2_mul1))
check_decode_reference(
    "mcg decode agrees with independent fp64 reference (K2)",
    out2_mcg, tr2, suh2, svh2, "mcg")

if CAMPAIGN_READER is not None:
    if 2 in getattr(CAMPAIGN_READER, "SUPPORTED_BITS", ()):
        theirs2 = CAMPAIGN_READER.decode_choice_hf(tr2, suh2, svh2, bits=2)
        check("mcg decode parity vs campaign reader (K2)", torch.equal(out2_mcg, theirs2))
    else:
        skip("mcg decode parity vs campaign reader (K2)",
             "the campaign reader's SUPPORTED_BITS does not include 2; the "
             "pack.cu round-trip above is the K2 evidence")


# [7] decoder parity vs exllamav3's own reconstruction --------------------------
import exl3_decoder_parity_vs_exllamav3 as xp  # noqa: E402

PARITY_RUNG = "decoder parity vs exllamav3 reconstruct"
if not xp.DEFAULT_OUT.exists():
    skip(PARITY_RUNG,
         f"{xp.DEFAULT_OUT.relative_to(TOOLS.parent.parent)} absent: exllamav3 {xp.EXLLAMAV3_VERSION}'s "
         "reconstruct is a CUDA kernel; run engines/tools/exl3_decoder_parity_vs_exllamav3.py "
         "--install on a CUDA host (python 3.12, torch 2.11.0) to produce it")
else:
    parity = json.loads(xp.DEFAULT_OUT.read_text(encoding="utf-8"))
    check("parity receipt schema + pinned exllamav3 version",
          parity.get("schema") == xp.SCHEMA and parity.get("exllamav3_version") == xp.EXLLAMAV3_VERSION
          and parity.get("exllamav3_commit") == xp.EXLLAMAV3_COMMIT,
          f"{parity.get('schema')} exllamav3 {parity.get('exllamav3_version')}@{parity.get('exllamav3_commit')}")
    check("parity receipt covers both codebooks, K3 and K4, and >= 9 real modules",
          set(parity["codebooks"]) == {"mcg", "mul1"} and {3, 4} <= set(parity["k_values"])
          and parity["modules_compared"] >= 9 and len(parity["modules"]) == parity["modules_compared"])
    check("pre-Hadamard stage bitwise equal to exllamav3_ext.reconstruct on every module",
          parity["all_bitwise_pre_hadamard"] and all(m["pre_hadamard"]["equal"] for m in parity["modules"]),
          ", ".join(f"{m['name']}={m['pre_hadamard']['differing_elements']}" for m in parity["modules"]
                    if not m["pre_hadamard"]["equal"]))
    check("all_bitwise is the conjunction of the per-module fp16 verdicts",
          parity["all_bitwise"] == all(m["equal"] for m in parity["modules"]))
    windows = []
    for module in parity["modules"]:
        window = module["window"]
        trellis = xp.unb64(window["trellis_i16_b64"], "I16", window["trellis_shape"])
        suh = xp.unb64(window["suh_f16_b64"], "F16", [window["k_tiles"] * 16])
        svh = xp.unb64(window["svh_f16_b64"], "F16", [window["n_tiles"] * 16])
        check(f"committed window is one 128x128 Hadamard block at K{module['K']} ({module['label']} {module['name']})",
              tuple(trellis.shape) == (8, 8, module["K"] * 16) and suh.numel() == 128 and svh.numel() == 128)
        pre = xp.ours_pre_hadamard(trellis, module["codebook"])
        check(f"window pre-Hadamard re-decodes to the committed digest ({module['label']} {module['codebook']} K{module['K']})",
              xp.sha256_tensor(pre) == window["ours_pre_hadamard_sha256"])
        check(f"window pre-Hadamard digest equals exllamav3's ({module['label']} {module['codebook']} K{module['K']})",
              window["exllamav3_pre_hadamard_sha256"] == window["ours_pre_hadamard_sha256"]
              and window["pre_hadamard"]["equal"] and window["pre_hadamard"]["differing_elements"] == 0)
        windows.append((module, trellis, suh, svh, pre))
    try:
        _, exl3_live, imported = xp.import_exllamav3(expect_precompiled=False)
        if not torch.cuda.is_available():
            raise xp.ParityError("exllamav3 imported but no CUDA device is present")
    except Exception as exc:  # noqa: BLE001 - the skip names the failure
        skip("exllamav3 live re-reconstruction of the committed windows",
             f"{type(exc).__name__}: {str(exc)[:200]}")
    else:
        device = torch.device("cuda:0")
        for module, trellis, suh, svh, pre in windows:
            marker = torch.tensor([module["marker"]], dtype=torch.int32)
            t_pre, t_weight = xp.theirs_reconstruct(exl3_live, trellis, suh, svh, module["codebook"], marker, device)
            check(f"exllamav3 live pre-Hadamard equals committed digest and ours ({module['label']} K{module['K']})",
                  xp.sha256_tensor(t_pre) == module["window"]["exllamav3_pre_hadamard_sha256"]
                  and torch.equal(pre, t_pre))
            live = xp.compare(xp.ours_weight(trellis, suh, svh, module["codebook"]).to(torch.float16), t_weight)
            committed = module["window"]["weight_fp16"]
            check(f"exllamav3 live weight verdict reproduces the committed one ({module['label']} K{module['K']})",
                  live["equal"] == committed["equal"]
                  and live["differing_elements"] == committed["differing_elements"]
                  and abs(live["max_abs_diff"] - committed["max_abs_diff"]) <= 1e-12,
                  f"live {live} committed {committed}")

print(f"selftest_exl3hf_offline: {sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} green")
