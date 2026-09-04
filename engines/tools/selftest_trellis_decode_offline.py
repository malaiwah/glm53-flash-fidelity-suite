#!/usr/bin/env python3
"""Offline selftest for layer_outer's EXL3 trellis weight source.

The decode ARITHMETIC is exl3hf_surface's and is proven bitwise elsewhere
(`selftest_exl3hf_offline.py`: LUTs against an independent fp64 route, anybits
unpack against dione_surface at K2/K3/K4/K6/K8, mcg against the campaign
reader). What is new here, and what this file covers, is the WEIGHT SOURCE:
grouping a checkpoint's payload objects per module, choosing each module's
codebook from the object it actually carries, composing with the block-FP8
decoder for a mixed artifact, and refusing every shape of partial or
unrecognised payload rather than loading trellis bytes as weights.

  [1] payload grouping: three objects + exactly one codebook marker per module.
  [2] per-module codebook: mcg and mul1 in ONE checkpoint both decode, each
      through its own LUT (drowzeys ships mcg on layer 3, mul1 on 4-77).
  [3] decoded values equal exl3hf_surface.decode_payload_hf exactly, and the
      key the converter sees is `<module>.weight`.
  [4] a mismatched codebook marker is refused (payload not written by the
      codebook it names).
  [5] a partial payload group is refused, not skipped.
  [6] rank-split TR3 payloads (davidsyoung) are refused BY NAME.
  [7] an exl3 config with no payload group at all is refused.
  [8] mixed trellis + block-FP8 in one subset: both hooks run, FP8 tensors
      arrive dequantized, trellis modules arrive decoded (wrldsuksgo2mars).
  [9] non-payload tensors pass through untouched, by identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402

import exl3hf_surface as xs  # noqa: E402
import layer_outer as lo  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[%s] %s%s" % ("ok" if ok else "FAIL", name, (" - " + detail) if detail else ""))
    if not ok:
        raise SystemExit("selftest_trellis_decode_offline: %s failed: %s" % (name, detail))


def refuses(fn, fragment):
    try:
        fn()
    except lo.LayerOuterError as exc:
        return fragment in str(exc), str(exc)[:180]
    except Exception as exc:  # noqa: BLE001
        return False, "wrong exception %s: %s" % (type(exc).__name__, exc)
    return False, "no refusal"


class _Config:
    def __init__(self, quantization_config=None):
        self.quantization_config = quantization_config


def _payload(k_tiles=8, n_tiles=8, bits=3, seed=0):
    """One synthetic exl3 payload group of the stock object layout.

    8 tiles x 16 = 128 along each axis: the decode applies a 128x128 hadamard
    to each axis, so a tile count that is not a multiple of 8 is not a valid
    exl3 payload shape at all.
    """
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -(2 ** 15), 2 ** 15, (k_tiles, n_tiles, 16 * bits),
        generator=generator, dtype=torch.int16)
    suh = torch.randn(k_tiles * 16, generator=generator, dtype=torch.float32).to(torch.float16)
    svh = torch.randn(n_tiles * 16, generator=generator, dtype=torch.float32).to(torch.float16)
    return {"trellis": trellis, "suh": suh, "svh": svh}


def _subset(module, payload, codebook):
    marker = torch.tensor([xs.CODEBOOK_OBJECTS[codebook]], dtype=torch.int32)
    return {
        "%s.trellis" % module: payload["trellis"],
        "%s.suh" % module: payload["suh"],
        "%s.svh" % module: payload["svh"],
        "%s.%s" % (module, codebook): marker,
    }


def main() -> int:
    module_a = "model.layers.3.mlp.experts.0.gate_proj"
    module_b = "model.layers.4.mlp.experts.1.down_proj"
    pay_a, pay_b = _payload(seed=1), _payload(seed=2, bits=4)
    subset = {}
    subset.update(_subset(module_a, pay_a, "mcg"))
    subset.update(_subset(module_b, pay_b, "mul1"))

    groups = lo.trellis_payload_groups(subset)
    check("[1] two modules grouped from eight keys", set(groups) == {module_a, module_b},
          repr(sorted(groups)))
    check("[1] each group names its own codebook",
          groups[module_a]["codebook"] == "mcg" and groups[module_b]["codebook"] == "mul1")

    config = _Config({"quant_method": "exl3", "codebook": "mcg", "bits": 3.0})
    plan = lo.trellis_checkpoint_plan(config, list(subset))
    check("[2] plan counts both modules and both codebooks",
          plan["quantized_module_count"] == 2
          and plan["codebook_histogram"] == {"mcg": 1, "mul1": 1},
          repr(plan["codebook_histogram"]))

    stats = {"decoded_modules": 0, "trellis_bits": 0}
    out = lo.materialize_trellis_subset(subset, plan, torch.bfloat16, stats)
    check("[3] the converter sees <module>.weight for both",
          set(out) == {"%s.weight" % module_a, "%s.weight" % module_b}, repr(sorted(out)))
    for module, payload, codebook in ((module_a, pay_a, "mcg"), (module_b, pay_b, "mul1")):
        want = xs.decode_payload_hf(
            payload["trellis"], payload["suh"], payload["svh"], codebook=codebook)
        got = out["%s.weight" % module]
        check("[3] %s decodes exactly like decode_payload_hf (%s)" % (module.split(".")[-1], codebook),
              torch.equal(got, want.to(torch.bfloat16)),
              "max abs diff in bf16 %r"
              % (got.float() - want.to(torch.bfloat16).float()).abs().max().item())
    check("[2] stats counted both modules", stats["decoded_modules"] == 2)

    wrong = dict(subset)
    wrong["%s.mcg" % module_a] = torch.tensor([12345], dtype=torch.int32)
    ok, detail = refuses(
        lambda: lo.materialize_trellis_subset(wrong, plan, torch.bfloat16,
                                              {"decoded_modules": 0, "trellis_bits": 0}),
        "not written by the codebook it names")
    check("[4] a mismatched codebook marker is refused", ok, detail)

    partial = {key: value for key, value in subset.items() if not key.endswith(".svh")}
    ok, detail = refuses(lambda: lo.trellis_payload_groups(partial),
                         "incomplete trellis payload group")
    check("[5] a partial payload group is refused", ok, detail)

    two_markers = dict(subset)
    two_markers["%s.mul1" % module_a] = torch.tensor(
        [xs.CODEBOOK_OBJECTS["mul1"]], dtype=torch.int32)
    ok, detail = refuses(lambda: lo.trellis_payload_groups(two_markers),
                         "incomplete trellis payload group")
    check("[5] two codebook markers on one module is refused", ok, detail)

    rank_split = {
        "model.layers.3.mlp.experts.0.down_proj.rank0.trellis": pay_a["trellis"],
        "model.layers.3.mlp.experts.0.down_proj.rank0.suh": pay_a["suh"],
        "model.layers.3.mlp.experts.0.down_proj.rank0.svh": pay_a["svh"],
        "model.layers.3.mlp.experts.0.down_proj.rank0.mcg": torch.tensor([1], dtype=torch.int32),
    }
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(config, list(rank_split)),
                         "rank-split trellis payload")
    check("[6] rank-split TR3 payloads are refused by name", ok, detail)

    ok, detail = refuses(
        lambda: lo.trellis_checkpoint_plan(config, ["model.embed_tokens.weight"]),
        "carries no trellis/suh/svh payload group")
    check("[7] an exl3 config with no payload group is refused", ok, detail)

    check("[7] a non-exl3 config yields no trellis plan",
          lo.trellis_checkpoint_plan(_Config({"quant_method": "fp8"}), list(subset)) is None
          and lo.trellis_checkpoint_plan(_Config(None), list(subset)) is None)

    fp8_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scales = torch.tensor([[2.0]], dtype=torch.float32)
    mixed = dict(subset)
    mixed["model.layers.3.self_attn.o_proj.weight"] = fp8_weight
    mixed["model.layers.3.self_attn.o_proj.weight_scale_inv"] = scales
    mixed["model.layers.3.input_layernorm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    fp8_plan = lo.fp8_checkpoint_plan_for_mixed(config)
    stats2 = {"decoded_modules": 0, "trellis_bits": 0, "dequantized": 0,
              "scales_consumed": 0, "fp8_bytes": 0}
    out2 = lo.materialize_trellis_subset(mixed, plan, torch.bfloat16, stats2,
                                         fp8_plan=fp8_plan)
    want_fp8 = lo.dequantize_block_fp8(fp8_weight, scales, torch.bfloat16, (128, 128))
    check("[8] mixed artifact: FP8 tensors arrive dequantized",
          torch.equal(out2["model.layers.3.self_attn.o_proj.weight"], want_fp8),
          repr(out2["model.layers.3.self_attn.o_proj.weight"]))
    check("[8] mixed artifact: the scale key never reaches the converter",
          "model.layers.3.self_attn.o_proj.weight_scale_inv" not in out2)
    check("[8] mixed artifact: trellis modules still decode",
          stats2["decoded_modules"] == 2 and stats2["dequantized"] == 1)
    check("[9] a plain tensor passes through by identity",
          out2["model.layers.3.input_layernorm.weight"]
          is mixed["model.layers.3.input_layernorm.weight"])

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\nselftest_trellis_decode_offline: %d passed, %d failed"
          % (passed, len(RESULTS) - passed))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
