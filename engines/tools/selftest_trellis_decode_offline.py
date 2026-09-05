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
    # 0-dim, exactly as the real checkpoints write it (drowzeys layer 3
    # gate_proj.mcg: shape [], dtype I32). A 1-element 1-D fixture hides the
    # lazy-slice bug entirely.
    marker = torch.tensor(xs.CODEBOOK_OBJECTS[codebook], dtype=torch.int32)
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

    # bits is declared None here: the fixture mixes a K3 and a K4 module on
    # purpose, and a uniform declaration over that is exactly what rung [14]
    # refuses.
    config = _Config({"quant_method": "exl3", "codebook": "mcg", "bits": None})
    plan = lo.trellis_checkpoint_plan(config, list(subset))
    check("[2] plan counts both modules and both codebooks",
          plan["_observed"]["quantized_module_count"] == 2
          and plan["_observed"]["codebook_histogram"] == {"mcg": 1, "mul1": 1},
          repr(plan["_observed"]))
    # The CONTRACT half of the plan must mirror the controller's candidate
    # block exactly: qualify_root compares them for equality, and a mismatch
    # refuses only AFTER both cold runs and the self-compare have passed.
    import importlib.util
    spec = importlib.util.spec_from_file_location("mc", "bin/measure_cloud.py")
    contract_keys = {"quant_method", "codebook", "bits", "head_bits",
                     "modules_to_not_convert"}
    check("[2] the plan's contract keys mirror measure_cloud's candidate block",
          set(plan) - {"_observed"} == contract_keys,
          repr(sorted(set(plan) - {"_observed"})))

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
    wrong["%s.mcg" % module_a] = torch.tensor(12345, dtype=torch.int32)
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
        xs.CODEBOOK_OBJECTS["mul1"], dtype=torch.int32)
    ok, detail = refuses(lambda: lo.trellis_payload_groups(two_markers),
                         "incomplete trellis payload group")
    check("[5] two codebook markers on one module is refused", ok, detail)

    # [6] TP-sharded payloads. Without a declared tp the plan refuses; with
    # one, the ranks compose along the one axis the shapes admit, in
    # ascending order, and every inconsistency refuses by name.
    rank_keys = {}
    for r, pay in enumerate((pay_a, pay_b)):
        for name in ("trellis", "suh", "svh"):
            rank_keys["model.layers.3.mlp.experts.0.down_proj.rank%d.%s" % (r, name)] = pay[name]
        rank_keys["model.layers.3.mlp.experts.0.down_proj.rank%d.mcg" % r] = torch.tensor(
            xs.CODEBOOK_OBJECTS["mcg"], dtype=torch.int32)
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(config, list(rank_keys)),
                         "declares no hybrid_tr3_tail.tp")
    check("[6] rank-sharded payloads without a declared tp are refused", ok, detail)

    class _TailConfig(_Config):
        def __init__(self, qc, tail):
            super().__init__(qc)
            self.hybrid_tr3_tail = tail
            self.hidden_size = 128 * 1  # decoded part is [128, 128]; see below
            self.moe_intermediate_size = 128 * 2

    # parts decode to [128, 128] (8x8 tiles); two ranks tile a down_proj
    # [hidden=128, inter=256] along axis 1 only.
    tail = {"format": "exl3-trellis", "codebook": "mcg", "tp": 2, "bits_avg": 3.5,
            "k_values": [3, 4], "slicing": {"down_proj": "K-sliced: rank r = input cols"}}
    tcfg = _TailConfig({"quant_method": "modelopt"}, tail)
    tplan = lo.trellis_checkpoint_plan(tcfg, list(rank_keys))
    check("[6] a hybrid_tr3_tail declaration is accepted over a leftover quant_method",
          tplan["quant_method"] == "exl3" and tplan["codebook"] == "mcg" and tplan["bits"] == 3.5
          and tplan["_observed"]["quant_method_declared"] == "modelopt"
          and tplan["_observed"]["composition"]["tp"] == 2, repr(tplan))
    tail_shapes = {"model.layers.3.mlp.experts.0.down_proj.weight": (128, 256)}
    comp = tplan["_observed"]["composition"]
    tstats = {"decoded_modules": 0, "trellis_bits": 0}
    out6 = lo.materialize_trellis_subset(rank_keys, tplan, torch.bfloat16, tstats,
                                         composition=comp, expected_shape=tail_shapes.get)
    want6 = torch.cat([xs.decode_payload_hf(p["trellis"], p["suh"], p["svh"], codebook="mcg")
                       for p in (pay_a, pay_b)], dim=1).to(torch.bfloat16)
    check("[6] two ranks compose along the one admissible axis in ascending order",
          set(out6) == {"model.layers.3.mlp.experts.0.down_proj.weight"}
          and torch.equal(out6["model.layers.3.mlp.experts.0.down_proj.weight"], want6)
          and tstats["tp_composed_modules"] == 1 and tstats["tp_axes"] == {"down_proj": 1},
          repr({k: tuple(v.shape) for k, v in out6.items()}) + repr(tstats))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        rank_keys, tplan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0},
        composition=comp, expected_shape={"model.layers.3.mlp.experts.0.down_proj.weight": (256, 128)}.get),
        "the artifact declares")
    check("[6] a declared slicing that contradicts the admissible axis is refused", ok, detail)
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        rank_keys, tplan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0},
        composition=comp, expected_shape={"model.layers.3.mlp.experts.0.down_proj.weight": (256, 256)}.get),
        "along exactly one axis")
    check("[6] shapes that tile no axis are refused", ok, detail)
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        rank_keys, tplan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0},
        composition=None, expected_shape=tail_shapes.get), "carries no composition")
    check("[6] rank payloads without a composition are refused at decode", ok, detail)
    missing_rank = {k: v for k, v in rank_keys.items() if ".rank1." not in k}
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(tcfg, list(missing_rank)),
                         "do not carry exactly ranks")
    check("[6] a module missing a rank is refused at plan time", ok, detail)

    # [6b] verified zero-pad truncation
    plain = {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight":
             torch.cat([torch.randn(576, 64), torch.zeros(64, 64)]).to(torch.bfloat16)}
    zstats = {}
    out6b = lo.truncate_zero_padded_rows(
        plain, {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight": (576, 64)}.get, zstats)
    check("[6b] an all-zero tail is truncated to the expected shape and recorded",
          tuple(out6b["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"].shape) == (576, 64)
          and zstats["zero_padded_rows_truncated"]["count"] == 1
          and zstats["zero_padded_rows_truncated"]["rows"] == 64, repr(zstats))
    bad = {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight":
           torch.cat([torch.randn(576, 64), torch.zeros(64, 64)]).to(torch.bfloat16)}
    bad["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"][600, 3] = 1.0
    ok, detail = refuses(lambda: lo.truncate_zero_padded_rows(
        bad, {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight": (576, 64)}.get, {}),
        "not padding, a different tensor")
    check("[6b] one non-zero element in the tail refuses by name", ok, detail)
    exact = {"model.norm.weight": torch.ones(576)}
    out6c = lo.truncate_zero_padded_rows(exact, {"model.norm.weight": (576,)}.get, {})
    check("[6b] an exact-shape tensor passes through by identity",
          out6c["model.norm.weight"] is exact["model.norm.weight"])
    unknown = {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight": torch.zeros(640, 64)}
    out6d = lo.truncate_zero_padded_rows(unknown, lambda k: None, {})
    check("[6b] with no expected shape nothing is truncated",
          tuple(out6d["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"].shape) == (640, 64))

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

    # [10] THROUGH THE REAL CALLER: build_streamed_model keeps two separate
    # counter dicts and passes both. The mixed rung above seeded one combined
    # dict, which is more generous than any real caller -- and that gap let a
    # KeyError('dequantized') reach a live pod.
    fp8_counters = {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0}
    trellis_counters = {"decoded_modules": 0, "trellis_bits": 0}
    out3 = lo._materialized(mixed, None, plan, fp8_plan, torch.bfloat16,
                            fp8_counters, trellis_counters)
    check("[10] _materialized with the caller's two stats dicts decodes both surfaces",
          torch.equal(out3["model.layers.3.self_attn.o_proj.weight"], want_fp8)
          and trellis_counters["decoded_modules"] == 2
          and fp8_counters["dequantized"] == 1,
          "trellis %r fp8 %r" % (trellis_counters, fp8_counters))
    trellis_only = {"decoded_modules": 0, "trellis_bits": 0}
    out4 = lo._materialized(subset, None, plan, None, torch.bfloat16,
                            {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0},
                            trellis_only)
    check("[10] _materialized on a pure trellis subset needs no FP8 plan",
          set(out4) == {"%s.weight" % module_a, "%s.weight" % module_b}
          and trellis_only["decoded_modules"] == 2)

    # [11] THROUGH REAL LAZY SLICES, not eager tensors. safetensors hands the
    # streamer PySafeSlice objects; `slice[:]` raises on the 0-dim I32 codebook
    # marker, which every eager fixture above hides. This is the shape of the
    # subset build_streamed_model actually passes.
    import tempfile
    from safetensors.torch import save_file
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "shard.safetensors"
        save_file({k: v.contiguous() for k, v in subset.items()}, str(shard))
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            lazy = {key: handle.get_slice(key) for key in subset}
            check("[11] the marker really is a 0-dim lazy slice",
                  list(lazy["%s.mcg" % module_a].get_shape()) == [], "fixture is wrong")
            lazy_stats = {"decoded_modules": 0, "trellis_bits": 0}
            out5 = lo.materialize_trellis_subset(lazy, plan, torch.bfloat16, lazy_stats)
            want = xs.decode_payload_hf(pay_a["trellis"], pay_a["suh"], pay_a["svh"],
                                        codebook="mcg").to(torch.bfloat16)
            check("[11] lazy slices decode identically to eager tensors",
                  torch.equal(out5["%s.weight" % module_a], want)
                  and lazy_stats["decoded_modules"] == 2,
                  repr(lazy_stats))

    # [12] the decode device reaches the decoder. The trellis decode is
    # matmul-heavy and a host decode is ~11 h per cold run at GLM-5.3 scale,
    # so _materialized MUST forward the capture device; a default-to-cpu
    # signature silently reintroduces that.
    import inspect
    sig = inspect.signature(lo._materialized)
    check("[12] _materialized takes a device", "device" in sig.parameters)
    src = Path(lo.__file__).read_text()
    import re as _re
    calls = _re.findall(r"_materialized\((?:[^()]|\([^()]*\))*\)", src)
    calls = [c for c in calls if "trellis_stats" in c and "Dict[str, Any]" not in c]
    check("[12] both call sites pass device=device",
          len(calls) >= 2 and all("device=device" in c for c in calls),
          "call sites must forward the model device, not default to cpu: %r" % calls)
    dev_stats = {"decoded_modules": 0, "trellis_bits": 0}
    out6 = lo._materialized(subset, None, plan, None, torch.bfloat16,
                            {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0},
                            dev_stats, device="cpu")
    check("[12] an explicit device still decodes correctly",
          torch.equal(out6["%s.weight" % module_a],
                      xs.decode_payload_hf(pay_a["trellis"], pay_a["suh"], pay_a["svh"],
                                            codebook="mcg").to(torch.bfloat16)))

    # [13] DRIFT GUARD: the controller's candidate block and the pod's plan are
    # two implementations of one contract that qualify_root compares for exact
    # equality. Three real config shapes: inline exl3 with a codebook
    # (drowzeys), inline exl3 without one (wrldsuksgo2mars), and a
    # hybrid_tr3_tail declaration over a leftover ModelOpt block (davidsyoung).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "measure_cloud_under_test", str(Path(__file__).resolve().parents[2] / "bin" / "measure_cloud.py"))
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    shapes = {
        "drowzeys": ({"quant_method": "exl3", "codebook": "mul1", "bits": 3, "head_bits": 16,
                      "version": "1.4.5"}, None, list(subset), lo.TRELLIS_DECODE_METHOD),
        "wrld": ({"quant_method": "exl3", "bits": 4}, None, list(subset), lo.TRELLIS_DECODE_METHOD),
        "davidsyoung": ({"quant_method": "modelopt", "config_groups": {}},
                        {"format": "exl3-trellis", "codebook": "mcg", "tp": 2, "bits_avg": 3.25,
                         "slicing": {"down_proj": "K-sliced: rank r = input cols"}},
                        list(rank_keys), lo.TRELLIS_TP_COMPOSE_METHOD),
    }
    for label, (qc, tail, keys, method) in shapes.items():
        cfg = {"quantization_config": qc}
        if tail is not None:
            cfg["hybrid_tr3_tail"] = tail
        ctrl = mc._candidate_decode_plan(qc, cfg)
        pod_cfg = _TailConfig(qc, tail) if tail is not None else _Config(qc)
        pod = lo.trellis_checkpoint_plan(pod_cfg, keys)
        observed = pod.pop("_observed")
        pod_method = lo.TRELLIS_TP_COMPOSE_METHOD if observed["composition"] else lo.TRELLIS_DECODE_METHOD
        check("[13] %s: controller and pod agree on quantization_config" % label,
              ctrl["quantization_config"] == pod, "ctrl %r pod %r" % (ctrl["quantization_config"], pod))
        check("[13] %s: controller and pod agree on the method (%s)" % (label, method),
              ctrl["method"] == pod_method == method, "ctrl %r pod %r" % (ctrl["method"], pod_method))

    # [14] declared bits are bound to the payload bytes (review S2).
    uniform = lo.trellis_checkpoint_plan(_Config({"quant_method": "exl3", "codebook": "mcg",
                                                  "bits": 3.0}), list(subset))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        subset, uniform, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0}),
        "a K4 payload but the artifact declares bits=3.0")
    check("[14] a uniform bits declaration over a different K is refused", ok, detail)
    only_k3 = {k: v for k, v in subset.items() if module_b not in k}
    st14 = {"decoded_modules": 0, "trellis_bits": 0}
    lo.materialize_trellis_subset(only_k3, uniform, torch.bfloat16, st14)
    check("[14] a matching uniform declaration decodes and records the K histogram",
          st14["k_histogram"] == {"3": 1}, repr(st14))
    tail_k = {"format": "exl3-trellis", "codebook": "mcg", "tp": 2, "bits_avg": 3.5,
              "k_values": [3, 4], "slicing": {"down_proj": "K-sliced"}}
    tcfg_k = _TailConfig({"quant_method": "modelopt"}, tail_k)
    tplan_k = lo.trellis_checkpoint_plan(tcfg_k, list(rank_keys))
    comp_k = tplan_k["_observed"]["composition"]
    check("[14] a TR3 tail's k_values reach the composition", comp_k["k_values"] == [3, 4])
    st14b = {"decoded_modules": 0, "trellis_bits": 0}
    lo.materialize_trellis_subset(rank_keys, tplan_k, torch.bfloat16, st14b,
                                  composition=comp_k, expected_shape=tail_shapes.get)
    check("[14] mixed K3/K4 ranks are admitted under k_values [3, 4]",
          st14b["k_histogram"] == {"3": 1, "4": 1}, repr(st14b))
    tail_k3 = dict(tail_k, k_values=[3])
    tplan_k3 = lo.trellis_checkpoint_plan(_TailConfig({"quant_method": "modelopt"}, tail_k3),
                                          list(rank_keys))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        rank_keys, tplan_k3, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0},
        composition=tplan_k3["_observed"]["composition"], expected_shape=tail_shapes.get),
        "declares k_values [3]")
    check("[14] a K outside the declared k_values is refused", ok, detail)

    # [15] a bare fp8 tensor in a trellis-only tree is refused (review S4).
    bare = dict(subset)
    bare["model.layers.3.self_attn.o_proj.weight"] = torch.zeros(4, 4).to(torch.float8_e4m3fn)
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        bare, plan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0}),
        "tensor anywhere; loading it as bf16 would apply no block scale")
    check("[15] a bare fp8 tensor with no scale anywhere is refused", ok, detail)

    # [16] a plain weight beside a payload group is refused, not overwritten (review S5).
    both = dict(subset)
    both["%s.weight" % module_a] = torch.zeros(128, 128, dtype=torch.bfloat16)
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        both, plan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0}),
        "two versions of one tensor")
    check("[16] a plain weight beside its payload group is refused", ok, detail)

    # [17] the fp32 matmul policy is pinned and recorded (review S3).
    policy = lo._pin_fp32_matmul_policy()
    check("[17] TF32 is pinned off and the precision is highest",
          torch.backends.cuda.matmul.allow_tf32 is False
          and torch.get_float32_matmul_precision() == "highest"
          and policy["pinned"]["float32_matmul_precision"] == "highest"
          and "NVIDIA_TF32_OVERRIDE" in policy["before_pin"], repr(policy))

    # [18] The FP8 gate and the trellis gate consult ONE predicate. Three
    # davidsyoung pods died on 2026-09-05 after their fetch because
    # build_streamed_model asked `fp8_checkpoint_plan` about a
    # `quant_method: modelopt` leftover before the trellis gate one line
    # below could read the `hybrid_tr3_tail` declaration. The resolver runs
    # the exact pod decision on the exact config shape, at $0.
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors"
                                            for k in rank_keys}}))
        events = []
        dy_cfg = _TailConfig({"quant_method": "modelopt", "config_groups": {},
                              "producer": {"name": "modelopt"}},
                             {"format": "exl3-trellis", "codebook": "mcg", "tp": 2,
                              "bits_avg": 3.25, "k_values": [3, 4],
                              "slicing": {"down_proj": "K-sliced: rank r = input cols"}})
        check("[18] the predicate reads the tail over the ModelOpt leftover",
              lo.is_trellis_checkpoint(dy_cfg) and not lo.is_trellis_checkpoint(_Config(
                  {"quant_method": "modelopt", "config_groups": {}})))
        fp8_18, tr_18, trfp8_18, st_18 = lo.checkpoint_decode_plans(
            dy_cfg, td, lambda **kw: events.append(kw))
        check("[18] a hybrid_tr3_tail checkpoint passes the FP8 gate and plans a TP compose",
              fp8_18 is None and trfp8_18 is None and tr_18 is not None
              and tr_18["quant_method"] == "exl3" and st_18["declared_by"] == "hybrid_tr3_tail"
              and st_18["composition"]["tp"] == 2
              and [e["stage"] for e in events] == ["trellis_decode_plan"]
              and events[0]["method"] == lo.TRELLIS_TP_COMPOSE_METHOD,
              repr((fp8_18, tr_18, st_18, events)))
        ok, detail = refuses(lambda: lo.checkpoint_decode_plans(
            _Config({"quant_method": "modelopt", "config_groups": {}}), td, lambda **kw: None),
            "is not the block-scaled FP8 e4m3 weights-only form")
        check("[18] a ModelOpt block with NO tail declaration is still refused", ok, detail)
        native = lo.checkpoint_decode_plans(_Config(None), td, lambda **kw: events.append(kw))
        check("[18] a native tree plans nothing and never opens the index",
              native[:3] == (None, None, None) and len(events) == 1, repr(native))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\nselftest_trellis_decode_offline: %d passed, %d failed"
          % (passed, len(RESULTS) - passed))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
