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
  [19] the rotation-layout census is the SAME TEXT in hfmeta (controller)
       and layer_outer (pod), not merely agreeing on fixtures.
  [20] shared_h_v1 (willfalco / jpsequeira): the H-side vector resolved by
       name from `experts.shared_h.{proj}.rank{r}`, bitwise the stock decode;
       undeclared, missing, duplicated and orphaned vectors all refuse.
  [21] r7_shared (brandonmusic TR3v4): unsharded experts pass through with
       `r7_shared.gate_up_suh` / `down_svh`; K against r7 k_values.
  [22] non-routed exl3 modules (lm_head, o_proj, ...) decode by the same
       function, are named with K, checked against their declared bits; a
       decoded head is sealed as the candidate's own dequantized head.
  [23] a declared online mxfp8 overlay is the contract's activation_scheme.
  [24] the controller mirror reads the same layout contract from the index
       names and refuses without them.
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
                     "modules_to_not_convert", "rotation_layout", "shared_vectors",
                     "nonrouted_exl3", "activation_scheme"}
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

    # ---- EfficiencyFixes (review-efficiency S2-6 / S1-1) ------------------
    # [6c] the ranks are HELD in bf16 and composed the moment the last one
    # decodes, and the result is still bitwise cat-fp32-then-cast (`want6`):
    # a cast rounds each element alone and a cat only places them. The sink
    # sees the composed module before the loop ends, and a plain module after
    # it in the subset, in that order -- nothing waits for the loop to finish.
    offered = []

    def sink6c(key, tensor):
        offered.append((key, str(tensor.dtype)))
        return True
    eager_keys = dict(rank_keys)
    eager_keys.update(_subset(module_b, pay_b, "mul1"))
    tail_shapes_6c = dict(tail_shapes)
    st6c = {"decoded_modules": 0, "trellis_bits": 0}
    out6c = lo.materialize_trellis_subset(eager_keys, tplan, torch.bfloat16, st6c,
                                          composition=comp, expected_shape=tail_shapes_6c.get,
                                          sink=sink6c)
    check("[6c] rank shards are stored in the output dtype, not fp32",
          st6c.get("tp_rank_storage_dtype") == "bfloat16", repr(st6c))
    check("[6c] a module composes as soon as its last rank decodes and goes to the sink "
          "before later modules are decoded",
          [k for k, _ in offered] == ["model.layers.3.mlp.experts.0.down_proj.weight",
                                      "%s.weight" % module_b]
          and all(d == "torch.bfloat16" for _, d in offered) and out6c == {},
          repr(offered) + repr(sorted(out6c)))
    st6d = {"decoded_modules": 0, "trellis_bits": 0}
    out6d = lo.materialize_trellis_subset(rank_keys, tplan, torch.bfloat16, st6d,
                                          composition=comp, expected_shape=tail_shapes.get)
    check("[6c] cast-per-rank-then-cat is bitwise cat-then-cast",
          torch.equal(out6d["model.layers.3.mlp.experts.0.down_proj.weight"], want6))
    # ---- end EfficiencyFixes ---------------------------------------------

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
        "willfalco": ({"quant_method": "modelopt", "config_groups": {}},
                      {"format": "exl3-trellis", "codebook": "mcg", "tp": 2, "bits": "mixed",
                       "expert_bpw_mean": 3.25, "k_values": [3, 4], "k4_experts_total": 4800,
                       "slicing": {"down_proj": "K-sliced: rank r = input cols"}},
                      list(rank_keys), lo.TRELLIS_TP_COMPOSE_METHOD),
    }
    for label, (qc, tail, keys, method) in shapes.items():
        cfg = {"quantization_config": qc}
        if tail is not None:
            cfg["hybrid_tr3_tail"] = tail
        ctrl = mc._candidate_decode_plan(qc, cfg, index_keys=keys)
        pod_cfg = _TailConfig(qc, tail) if tail is not None else _Config(qc)
        pod = lo.trellis_checkpoint_plan(pod_cfg, keys)
        observed = pod.pop("_observed")
        pod_method = lo.TRELLIS_TP_COMPOSE_METHOD if observed["composition"] else lo.TRELLIS_DECODE_METHOD
        check("[13] %s: controller and pod agree on quantization_config" % label,
              ctrl["quantization_config"] == pod, "ctrl %r pod %r" % (ctrl["quantization_config"], pod))
        check("[13] %s: controller and pod agree on the method (%s)" % (label, method),
              ctrl["method"] == pod_method == method, "ctrl %r pod %r" % (ctrl["method"], pod_method))
        if label == "willfalco":
            # bits: "mixed" beside expert_bpw_mean: 3.25 -- the contract's bits
            # is the NUMERIC declaration on both sides, and the sniffer reads
            # the same number (it refused this shape as "no numeric bits_avg").
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
            from fidelity import hfmeta as hm
            check("[13] willfalco: bits: \"mixed\" + expert_bpw_mean 3.25 -> bits 3.25 on the "
                  "controller, the pod and hfmeta.tr3_tail_declared_bits",
                  ctrl["quantization_config"]["bits"] == 3.25 and pod["bits"] == 3.25
                  and hm.tr3_tail_declared_bits(tail) == 3.25
                  and hm.tr3_tail_declared_bits({"bits": "mixed"}) == "mixed",
                  repr((ctrl["quantization_config"]["bits"], pod["bits"])))

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
        fp8_18, tr_18, trfp8_18, st_18, nv_18, gg_18 = lo.checkpoint_decode_plans(
            dy_cfg, td, lambda **kw: events.append(kw))
        check("[18] a hybrid_tr3_tail checkpoint passes the FP8 gate and plans a TP compose",
              fp8_18 is None and trfp8_18 is None and nv_18 is None and gg_18 is None
              and tr_18 is not None
              and tr_18["quant_method"] == "exl3" and st_18["declared_by"] == "hybrid_tr3_tail"
              and st_18["composition"]["tp"] == 2
              and [e["stage"] for e in events] == ["trellis_decode_plan"]
              and events[0]["method"] == lo.TRELLIS_TP_COMPOSE_METHOD,
              repr((fp8_18, tr_18, st_18, events)))
        # A ModelOpt block with no tail is judged by the modelopt (NVFP4) gate
        # now, which refuses every modelopt form but NVFP4 by name -- before
        # the FP8 gate would have, and before the index is opened.
        ok, detail = refuses(lambda: lo.checkpoint_decode_plans(
            _Config({"quant_method": "modelopt", "config_groups": {}}), td, lambda **kw: None),
            "quant_algo=None is not the NVFP4 form this schedule decodes")
        check("[18] a ModelOpt block with NO tail declaration is still refused", ok, detail)
        native = lo.checkpoint_decode_plans(_Config(None), td, lambda **kw: events.append(kw))
        check("[18] a native tree plans nothing and never opens the index",
              native[:3] == (None, None, None) and len(events) == 1, repr(native))

    # [18c] SIDECAR-declared bits: jpsequeira's GLM-5.2 TR3 declares
    # `bits: "mixed"` with no numeric beside `bits_per_expert:
    # "expert_precision_map.json:bitrates"` -- a per-layer per-expert map
    # shipped in the repo root.  The pod reads `<model_dir>/<file>`, the
    # controller fetches it by name from the target repo/revision, and BOTH
    # must put the exact float mean of every entry into the contract's
    # `bits` with a byte-identical `declared_bits_source` block (same sha256).
    # The mean is exact (3.3947882401315788 for the real artifact) so two
    # independent readers agree to full float repr.
    import hashlib as _hl18c
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
    from fidelity import hfmeta as hm18c
    side_doc = {"3": {"bitrates": [3] * 200 + [4] * 56},
                "4": {"bitrates": [3] * 100 + [4] * 156}}
    side_raw = json.dumps(side_doc).encode("utf-8")
    side_sha = _hl18c.sha256(side_raw).hexdigest()

    def _pod_loader(sfile):
        with open(os.path.join(td, sfile), "rb") as _h:
            return json.loads(_h.read()), side_sha

    def _ctrl_loader(sfile):
        return side_doc, side_sha

    # hfmeta pure function: the mean and the source block
    mean18c, source18c = hm18c.tr3_tail_declared_bits(
        {"bits": "mixed", "bits_per_expert": "expert_precision_map.json:bitrates"},
        sidecar_loader=_ctrl_loader)
    mean_expected = (200 * 3 + 56 * 4 + 100 * 3 + 156 * 4) / 512
    check("[18c] hfmeta resolves the sidecar mean exactly and builds the source block",
          mean18c == mean_expected
          and source18c == {"sidecar": "expert_precision_map.json", "key": "bitrates",
                            "entries": 512, "histogram": {"3": 300, "4": 212},
                            "sha256": side_sha},
          repr((mean18c, source18c)))
    # Without a loader the legacy string is returned (so a refusal can name it)
    check("[18c] hfmeta without a loader returns the legacy string for a refusal",
          hm18c.tr3_tail_declared_bits({"bits": "mixed",
                                        "bits_per_expert": "x:y"}) == "mixed")

    # Pod side: write the sidecar into the checkpoint dir and plan
    with open(os.path.join(td, "expert_precision_map.json"), "wb") as _h:
        _h.write(side_raw)
    side_tail = {"format": "exl3-trellis", "codebook": "mcg", "tp": 2, "bits": "mixed",
                 "bits_per_expert": "expert_precision_map.json:bitrates",
                 "k_values": [3, 4], "experts_per_layer": 256, "moe_layers": [3, 4],
                 "rotation_layout": "shared_h_v1",
                 "shared_h_tensor_schema": lo.EXL3_SHARED_H_TENSOR_SCHEMA}

    class _SideConfig(_TailConfig):
        def __init__(self, qc, tail):
            super().__init__(qc, tail)
            self.n_routed_experts = 256

    side_cfg = _SideConfig({"quant_method": "modelopt"}, side_tail)
    side_keys = list(rank_keys)
    pod_plan = lo.trellis_checkpoint_plan(side_cfg, side_keys, model_dir=td)
    pod_plan.pop("_observed", None)
    check("[18c] the pod plan's bits is the exact sidecar mean and carries declared_bits_source",
          pod_plan["bits"] == mean_expected
          and pod_plan["declared_bits_source"] == source18c,
          repr((pod_plan["bits"], pod_plan.get("declared_bits_source"))))

    # Controller mirror: same sidecar through the injected loader, same block
    ctrl_decode = mc._candidate_decode_plan(
        side_cfg.quantization_config, {"quantization_config": side_cfg.quantization_config,
                                       "hybrid_tr3_tail": side_tail},
        index_keys=side_keys, sidecar_loader=_ctrl_loader)
    ctrl_qcfg = ctrl_decode["quantization_config"]
    check("[18c] the controller mirror's bits and declared_bits_source equal the pod's",
          ctrl_qcfg["bits"] == pod_plan["bits"]
          and ctrl_qcfg["declared_bits_source"] == pod_plan["declared_bits_source"],
          "ctrl %r pod %r" % (ctrl_qcfg.get("bits"), pod_plan.get("bits")))

    # [18d] SIDECAR refusals: the pod refuses by name when the sidecar is
    # absent, not strict JSON, or its expert count disagrees with
    # n_routed_experts.
    os.remove(os.path.join(td, "expert_precision_map.json"))
    ok, detail = refuses(
        lambda: lo.trellis_checkpoint_plan(side_cfg, side_keys, model_dir=td),
        "absent from the checkpoint directory")
    check("[18d] an absent sidecar is refused by name", ok, detail)
    with open(os.path.join(td, "expert_precision_map.json"), "wb") as _h:
        _h.write(b"not json {")
    ok, detail = refuses(
        lambda: lo.trellis_checkpoint_plan(side_cfg, side_keys, model_dir=td),
        "is not strict JSON")
    check("[18d] a sidecar that is not strict JSON is refused by name", ok, detail)
    with open(os.path.join(td, "expert_precision_map.json"), "wb") as _h:
        _h.write(json.dumps({"3": {"bitrates": [3] * 100},
                             "4": {"bitrates": [4] * 100}}).encode("utf-8"))
    ok, detail = refuses(
        lambda: lo.trellis_checkpoint_plan(side_cfg, side_keys, model_dir=td),
        "n_routed_experts")
    check("[18d] a sidecar whose expert count disagrees with n_routed_experts is refused", ok, detail)
    with open(os.path.join(td, "expert_precision_map.json"), "wb") as _h:
        _h.write(json.dumps({"3": {"bitrates": [3] * 256},
                             "5": {"bitrates": [4] * 256}}).encode("utf-8"))
    ok, detail = refuses(
        lambda: lo.trellis_checkpoint_plan(side_cfg, side_keys, model_dir=td),
        "moe_layers")
    check("[18d] a sidecar whose layer set disagrees with moe_layers is refused", ok, detail)

    # ---- rotation layouts (GLM-5.2 candidates) ----------------------------
    # [19] The layout census is ONE rule in two files: the pod's copy in
    # layer_outer must be the SAME TEXT as bin/fidelity/hfmeta's (the
    # controller's), not merely agree on today's fixtures.
    import inspect
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
    from fidelity import hfmeta as hm19
    for fn in ("exl3_rotation_groups", "exl3_declared_module_bits", "exl3_layout_contract"):
        check("[19] %s is byte-identical in hfmeta and layer_outer" % fn,
              inspect.getsource(getattr(hm19, fn)) == inspect.getsource(getattr(lo, fn)))
    check("[19] the layout constants agree",
          hm19.EXL3_ROTATION_LAYOUTS == lo.EXL3_ROTATION_LAYOUTS
          and hm19.EXL3_SHARED_H_TENSOR_SCHEMA == lo.EXL3_SHARED_H_TENSOR_SCHEMA
          and hm19._EXL3_EXPERT_RE.pattern == lo._EXL3_EXPERT_RE.pattern
          and hm19._EXL3_SHARED_H_RE.pattern == lo._EXL3_SHARED_H_RE.pattern
          and hm19._EXL3_R7_SHARED_RE.pattern == lo._EXL3_R7_SHARED_RE.pattern)

    # [20] shared_h_v1 (willfalco / jpsequeira): the rank groups carry only
    # the I-side vector; the H-side one (svh for down, suh for gate/up) is
    # the layer's `experts.shared_h.{proj}.rank{r}.{field}`. Resolved by
    # name, the decode is BITWISE the stock-layout decode of the same bytes.
    shared_tail = dict(tail, rotation_layout="shared_h_v1",
                       shared_h_tensor_schema=lo.EXL3_SHARED_H_TENSOR_SCHEMA)
    shared_keys = {k: v for k, v in rank_keys.items() if not k.endswith(".svh")}
    for r, pay in enumerate((pay_a, pay_b)):
        shared_keys["model.layers.3.mlp.experts.shared_h.down_proj.rank%d.svh" % r] = pay["svh"]
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(tcfg, list(shared_keys)),
                         "requires 'shared_h_v1'")
    check("[20] shared_h vectors without the tail's rotation_layout declaration are refused",
          ok, detail)
    scfg = _TailConfig({"quant_method": "modelopt"}, shared_tail)
    splan = lo.trellis_checkpoint_plan(scfg, list(shared_keys))
    sobs = splan.pop("_observed")
    check("[20] the plan names the layout, the shared vectors and their digest",
          splan["rotation_layout"] == "shared_h_v1"
          and splan["shared_vectors"]["count"] == 2
          and len(splan["shared_vectors"]["names_sha256"]) == 64
          and sobs["rotation_layout"] == "shared_h_v1"
          and sobs["modules_per_layout"] == {"shared_h_v1": 2}
          and sobs["composition"]["tp"] == 2, repr((splan, sobs)))
    sstats = {"decoded_modules": 0, "trellis_bits": 0,
              "module_bits_policy": sobs["module_bits_policy"]}
    offered20 = []
    out20 = lo.materialize_trellis_subset(
        shared_keys, splan, torch.bfloat16, sstats, composition=sobs["composition"],
        expected_shape=tail_shapes.get, sink=lambda k, t: offered20.append(k) or True)
    check("[20] shared_h_v1 decodes bitwise equal to the stock layout of the same bytes "
          "and hands the composed module to the sink",
          offered20 == ["model.layers.3.mlp.experts.0.down_proj.weight"] and out20 == {}
          and sstats["shared_vectors_applied"] == 2
          and sstats["modules_per_layout"] == {"shared_h_v1": 2}, repr((offered20, sstats)))
    held20 = lo.materialize_trellis_subset(
        shared_keys, splan, torch.bfloat16, dict(sstats), composition=sobs["composition"],
        expected_shape=tail_shapes.get)
    check("[20] ... bitwise", torch.equal(
        held20["model.layers.3.mlp.experts.0.down_proj.weight"], want6))
    check("[20] the shared vector never reaches the converter as a stray key",
          not any("shared_h" in k for k in held20), repr(sorted(held20)))
    missing20 = {k: v for k, v in shared_keys.items() if not k.endswith("rank1.svh")}
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(scfg, list(missing20)),
                         "incomplete trellis payload group")
    check("[20] a rank whose shared vector is missing is refused by name", ok, detail
          if "rank1 (missing ['svh']" in detail else "refusal does not name the module: " + detail)
    both20 = dict(shared_keys)
    both20["model.layers.3.mlp.experts.0.down_proj.rank0.svh"] = pay_a["svh"]
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(scfg, list(both20)),
                         "two candidates for one rotation vector")
    check("[20] a module carrying its own H-side vector beside the shared one is refused",
          ok, detail)
    orphan20 = dict(shared_keys)
    orphan20["model.layers.3.mlp.experts.shared_h.up_proj.rank0.suh"] = pay_a["suh"]
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(scfg, list(orphan20)),
                         "resolve no module")
    check("[20] a shared vector no module resolves is refused", ok, detail)
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(scfg, list(rank_keys)),
                         "carries no experts.shared_h vector")
    check("[20] a shared_h_v1 declaration over stock groups is refused", ok, detail)
    # [20b] the GLM-5.2 tails declare the slicing as "TP4 K-slice: rank r owns
    # input columns [512r,512r+512)" (davidsyoung: "K-sliced: ..."); the axis
    # TOKEN is what is read, and a declaration contradicting the shapes refuses.
    for slicing, expect in (("TP4 K-slice: rank r owns input columns [512r,512r+512)", True),
                            ("K-sliced: rank r = input cols", True),
                            ("TP4 N-slice: rank r owns output rows [512r,512r+512)", False),
                            ("rank r owns something", False)):
        gtail = dict(shared_tail, slicing={"down_proj": slicing})
        gplan = lo.trellis_checkpoint_plan(_TailConfig({"quant_method": "modelopt"}, gtail),
                                           list(shared_keys))
        gobs = gplan.pop("_observed")

        def compose20b(gplan=gplan, gobs=gobs):
            return lo.materialize_trellis_subset(
                shared_keys, gplan, torch.bfloat16,
                {"decoded_modules": 0, "trellis_bits": 0,
                 "module_bits_policy": gobs["module_bits_policy"]},
                composition=gobs["composition"], expected_shape=tail_shapes.get)
        if expect:
            out20b = compose20b()
            check("[20b] slicing %r composes along the admissible axis" % slicing,
                  torch.equal(out20b["model.layers.3.mlp.experts.0.down_proj.weight"], want6))
        else:
            ok, detail = refuses(compose20b, "the artifact declares")
            check("[20b] slicing %r contradicts the shapes and refuses" % slicing, ok, detail)

    # [21] r7_shared (brandonmusic 3.5 MTP78): UNSHARDED routed experts carry
    # only their I-side vector; `experts.r7_shared.gate_up_suh` serves gate
    # AND up, `experts.r7_shared.down_svh` serves down. Pass-through, no
    # composition; K checked against r7_routed_experts.k_values.
    gate_pay, up_pay, down_pay = _payload(seed=21), _payload(seed=22), _payload(seed=23, bits=4)
    r7_keys = {}
    for proj, pay in (("gate_proj", gate_pay), ("up_proj", up_pay), ("down_proj", down_pay)):
        own = "svh" if proj != "down_proj" else "suh"
        r7_keys["model.layers.10.mlp.experts.0.%s.trellis" % proj] = pay["trellis"]
        r7_keys["model.layers.10.mlp.experts.0.%s.%s" % (proj, own)] = pay[own]
        r7_keys["model.layers.10.mlp.experts.0.%s.mcg" % proj] = torch.tensor(
            xs.CODEBOOK_OBJECTS["mcg"], dtype=torch.int32)
    r7_keys["model.layers.10.mlp.experts.r7_shared.gate_up_suh"] = gate_pay["suh"]
    r7_keys["model.layers.10.mlp.experts.r7_shared.down_svh"] = down_pay["svh"]
    r7_qc = {"quant_method": "modelopt", "config_groups": {},
             "r7_routed_experts": {"schema": "r7-complete-v2-checkpoint-v1",
                                   "feature": "r7-asymmetric-two-stack",
                                   "moe_layers": [3, 77], "k_values": [3, 4, 5]}}
    r7_tail = {"format": "exl3-trellis", "codebook": "mcg", "tp": 4, "bits": 3.0}
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(
        _TailConfig({"quant_method": "modelopt"}, r7_tail), list(r7_keys)),
        "declares no r7_routed_experts block")
    check("[21] r7_shared vectors without the r7_routed_experts declaration are refused",
          ok, detail)
    r7cfg = _TailConfig(r7_qc, r7_tail)
    r7plan = lo.trellis_checkpoint_plan(r7cfg, list(r7_keys))
    r7obs = r7plan.pop("_observed")
    check("[21] the plan names r7_shared, 2 shared vectors, no composition, the r7 k_values",
          r7plan["rotation_layout"] == "r7_shared" and r7plan["shared_vectors"]["count"] == 2
          and r7obs["composition"] is None and r7obs["modules_per_layout"] == {"r7_shared": 3}
          and r7obs["module_bits_policy"]["r7_k_values"] == [3, 4, 5]
          and r7obs["r7_declaration"]["schema"] == "r7-complete-v2-checkpoint-v1",
          repr((r7plan, r7obs)))
    # brandonmusic's r7 encoder stores every expert with its intermediate
    # channels PERMUTED (gate/up rows, down columns, one permutation per
    # expert) and writes `permutations[E].new_to_old` into the layer manifest;
    # the decoder inverts it from that manifest, and refuses without it.
    import json as _json
    import tempfile as _tempfile
    r7_dir = _tempfile.mkdtemp(prefix="r7-manifests-")
    new_to_old = torch.randperm(128, generator=torch.Generator().manual_seed(21)).tolist()
    manifest21 = {
        "layer": 10, "schema_version": 2,
        "permutations": {"0": {"new_to_old": new_to_old, "policy": "energy_balanced"}},
        "vector_refs": {
            "model.layers.10.mlp.experts.0.gate_proj": {
                "suh": "model.layers.10.mlp.experts.r7_shared.gate_up_suh",
                "svh": "model.layers.10.mlp.experts.0.gate_proj.svh"},
            "model.layers.10.mlp.experts.0.up_proj": {
                "suh": "model.layers.10.mlp.experts.r7_shared.gate_up_suh",
                "svh": "model.layers.10.mlp.experts.0.up_proj.svh"},
            "model.layers.10.mlp.experts.0.down_proj": {
                "suh": "model.layers.10.mlp.experts.0.down_proj.suh",
                "svh": "model.layers.10.mlp.experts.r7_shared.down_svh"}},
    }
    Path(r7_dir, "r7-experts-layer-010.json").write_text(_json.dumps(manifest21))
    r7_qc["r7_routed_experts"]["bit_map_manifests"] = ["r7-experts-layer-010.json"]
    r7cfg = _TailConfig(r7_qc, r7_tail)
    r7cfg.moe_intermediate_size = 128
    r7plan = lo.trellis_checkpoint_plan(r7cfg, list(r7_keys))
    r7obs = r7plan.pop("_observed")
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        r7_keys, r7plan, torch.bfloat16,
        {"decoded_modules": 0, "trellis_bits": 0,
         "module_bits_policy": r7obs["module_bits_policy"]}),
        "carries no permutation source")
    check("[21] r7 experts without the layer-manifest permutation source are refused", ok, detail)
    ok, detail = refuses(lambda: lo.r7_permutation_source(
        r7cfg, _tempfile.mkdtemp(prefix="r7-empty-"), r7obs["r7_declaration"]),
        "absent from the checkpoint")
    check("[21] a listed manifest absent from the checkpoint is refused at plan time", ok, detail)
    r7source = lo.r7_permutation_source(r7cfg, r7_dir, r7obs["r7_declaration"])
    r7stats = {"decoded_modules": 0, "trellis_bits": 0,
               "module_bits_policy": r7obs["module_bits_policy"], "r7_permutations": r7source}
    out21 = lo.materialize_trellis_subset(r7_keys, r7plan, torch.bfloat16, r7stats)
    inverse21 = torch.argsort(torch.tensor(new_to_old))
    want21 = {}
    for proj, pay in (("gate_proj", gate_pay), ("up_proj", up_pay), ("down_proj", down_pay)):
        stored = xs.decode_payload_hf(
            pay["trellis"], gate_pay["suh"] if proj != "down_proj" else pay["suh"],
            pay["svh"] if proj != "down_proj" else down_pay["svh"],
            codebook="mcg").to(torch.bfloat16)
        want21["model.layers.10.mlp.experts.0.%s.weight" % proj] = stored.index_select(
            1 if proj == "down_proj" else 0, inverse21)
    check("[21] unsharded r7 experts pass through decoded bitwise, gate and up sharing one suh, "
          "the intermediate axis put back in source order by the manifest's inverse permutation",
          set(out21) == set(want21)
          and all(torch.equal(out21[k], want21[k]) for k in want21)
          and r7stats["shared_vectors_applied"] == 3 and r7stats["k_histogram"] == {"3": 2, "4": 1}
          and r7source.stats["experts_unpermuted"] == 3
          and r7source.stats["policies"] == {"energy_balanced": 3}
          and len(r7source.stats["manifest_sha256"]) == 1,
          repr((sorted(out21), r7stats, r7source.stats)))
    bad_refs = _json.loads(_json.dumps(manifest21))
    bad_refs["vector_refs"]["model.layers.10.mlp.experts.0.up_proj"]["suh"] = \
        "model.layers.10.mlp.experts.0.up_proj.suh"
    bad_dir = _tempfile.mkdtemp(prefix="r7-badrefs-")
    Path(bad_dir, "r7-experts-layer-010.json").write_text(_json.dumps(bad_refs))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        r7_keys, r7plan, torch.bfloat16,
        {"decoded_modules": 0, "trellis_bits": 0, "module_bits_policy": r7obs["module_bits_policy"],
         "r7_permutations": lo.r7_permutation_source(r7cfg, bad_dir, r7obs["r7_declaration"])}),
        "vector_refs name")
    check("[21] a manifest whose vector_refs disagree with the name resolution is refused",
          ok, detail)
    bad_perm = _json.loads(_json.dumps(manifest21))
    bad_perm["permutations"]["0"]["new_to_old"][0] = bad_perm["permutations"]["0"]["new_to_old"][1]
    bad_dir2 = _tempfile.mkdtemp(prefix="r7-badperm-")
    Path(bad_dir2, "r7-experts-layer-010.json").write_text(_json.dumps(bad_perm))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        r7_keys, r7plan, torch.bfloat16,
        {"decoded_modules": 0, "trellis_bits": 0, "module_bits_policy": r7obs["module_bits_policy"],
         "r7_permutations": lo.r7_permutation_source(r7cfg, bad_dir2, r7obs["r7_declaration"])}),
        "no valid 128-element permutation")
    check("[21] a manifest entry that is not a permutation is refused", ok, detail)
    r7_k2 = dict(r7_qc, r7_routed_experts=dict(r7_qc["r7_routed_experts"], k_values=[3]))
    p21 = lo.trellis_checkpoint_plan(_TailConfig(r7_k2, r7_tail), list(r7_keys))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        r7_keys, p21, torch.bfloat16,
        {"decoded_modules": 0, "trellis_bits": 0,
         "module_bits_policy": p21["_observed"]["module_bits_policy"],
         "r7_permutations": r7source}),
        "r7_routed_experts declares k_values [3]")
    check("[21] a K outside r7_routed_experts.k_values is refused", ok, detail)
    no_down = {k: v for k, v in r7_keys.items() if not k.endswith("r7_shared.down_svh")}
    ok, detail = refuses(lambda: lo.trellis_checkpoint_plan(r7cfg, list(no_down)),
                         "incomplete trellis payload group")
    check("[21] an unsharded expert whose shared vector is missing is refused by name",
          ok and "down_proj (missing ['svh']" in detail, detail)

    # [22] NON-ROUTED exl3 modules (o_proj / q_b_proj / indexer.wq_b / lm_head):
    # decoded by the same function wherever they sit, named with their K,
    # checked against the bits the artifact declares for them, and an exl3
    # head becomes the candidate's own dequantized head.
    head_pay, o_pay = _payload(seed=24, bits=8), _payload(seed=25, bits=6)
    nr_keys = dict(r7_keys)
    nr_keys.update(_subset("lm_head", head_pay, "mcg"))
    nr_keys.update(_subset("model.layers.10.self_attn.o_proj", o_pay, "mcg"))
    nr_tail = dict(r7_tail, protected_tensor_policy={
        "format": "exl3-protected-v1", "tensors": {"lm_head": {"bits": 8, "codebook": "mcg"}}})
    nr_qc = dict(r7_qc, tensor_storage={
        "model.layers.10.self_attn.o_proj": {"bits_per_weight": 6, "quant_format": "exl3"}})
    nrcfg = _TailConfig(nr_qc, nr_tail)
    nrplan = lo.trellis_checkpoint_plan(nrcfg, list(nr_keys))
    nrobs = nrplan.pop("_observed")
    check("[22] the contract counts and digests the non-routed modules with their declared bits",
          nrplan["nonrouted_exl3"] == {
              "count": 2, "names_sha256": lo._exl3_names_sha256(
                  ["lm_head", "model.layers.10.self_attn.o_proj"]),
              "declared_bits": {"6": 1, "8": 1}}
          and nrobs["module_bits_policy"]["nonrouted"] == {
              "lm_head": 8, "model.layers.10.self_attn.o_proj": 6}, repr(nrplan))
    nrstats = {"decoded_modules": 0, "trellis_bits": 0,
               "module_bits_policy": nrobs["module_bits_policy"], "r7_permutations": r7source}
    out22 = lo.materialize_trellis_subset(nr_keys, nrplan, torch.bfloat16, nrstats)
    check("[22] lm_head and o_proj decode exactly like decode_payload_hf and are named with K",
          torch.equal(out22["lm_head.weight"], xs.decode_payload_hf(
              head_pay["trellis"], head_pay["suh"], head_pay["svh"], codebook="mcg").to(torch.bfloat16))
          and torch.equal(out22["model.layers.10.self_attn.o_proj.weight"], xs.decode_payload_hf(
              o_pay["trellis"], o_pay["suh"], o_pay["svh"], codebook="mcg").to(torch.bfloat16))
          and nrstats["nonrouted_exl3_decoded"] == {"lm_head": 8,
                                                    "model.layers.10.self_attn.o_proj": 6}
          and nrstats["modules_per_layout"] == {"per_module": 2, "r7_shared": 3},
          repr(nrstats))

    class _Streamer:
        pass
    streamer22 = _Streamer()
    streamer22.trellis_plan = nrplan
    streamer22.trellis_stats = dict(nrstats, rotation_layout="r7_shared")
    identity = lo.head_decode_identity(streamer22)
    check("[22] a decoded exl3 head is the candidate's own dequantized head (HEAD-1d)",
          identity == {"quantized": True, "bits": 8, "source": "artifact_dequantized",
                       "method": lo.TRELLIS_DECODE_METHOD, "reference": lo.TRELLIS_DECODE_REFERENCE},
          repr(identity))
    streamer22.trellis_stats = dict(r7stats)
    check("[22] a head loaded as shipped stays native",
          lo.head_decode_identity(streamer22) is None)
    streamer22.trellis_stats = dict(nrstats, rotation_layout="r7_shared")
    evidence22 = lo.weights_decode_evidence(streamer22)
    check("[22] weights_decode evidence names the layout, its reader, the shared vectors and "
          "the non-routed modules with K",
          evidence22["quantization_config"]["rotation_layout"] == "r7_shared"
          and evidence22["rotation_layout"]["layout"] == "r7_shared"
          and "exl3_overlay.py" in evidence22["rotation_layout"]["reader"]
          and evidence22["rotation_layout"]["shared_vectors_applied"] == 3
          and evidence22["rotation_layout"]["nonrouted_exl3_modules"] == [
              {"name": "lm_head", "K": 8}, {"name": "model.layers.10.self_attn.o_proj", "K": 6}]
          and evidence22["rotation_layout"]["evidence"] == lo.TRELLIS_LAYOUT_EVIDENCE,
          repr(evidence22.get("rotation_layout")))
    wrong_bits = dict(nr_qc, tensor_storage={
        "model.layers.10.self_attn.o_proj": {"bits_per_weight": 4, "quant_format": "exl3"}})
    p22 = lo.trellis_checkpoint_plan(_TailConfig(wrong_bits, nr_tail), list(nr_keys))
    ok, detail = refuses(lambda: lo.materialize_trellis_subset(
        nr_keys, p22, torch.bfloat16,
        {"decoded_modules": 0, "trellis_bits": 0,
         "module_bits_policy": p22["_observed"]["module_bits_policy"],
         "r7_permutations": r7source}),
        "declares 4 bits for it")
    check("[22] a non-routed module whose K differs from its declared bits is refused", ok, detail)

    # [23] jpsequeira's online mxfp8 activation overlay lands as activation_scheme.
    ov_tail = dict(shared_tail, online_mxfp8_overlay={
        "activation": "dynamic_mxfp8", "format": "online-mxfp8-v1", "group_size": 32})
    ovplan = lo.trellis_checkpoint_plan(_TailConfig({"quant_method": "exl3", "bits": 3.0},
                                                    ov_tail), list(shared_keys))
    check("[23] a declared online mxfp8 overlay is the contract's activation_scheme",
          ovplan["activation_scheme"] == "dynamic_mxfp8" and splan["activation_scheme"] is None,
          repr(ovplan["activation_scheme"]))

    # [24] the controller mirror reads the SAME layout from the index names.
    for label, (qc, tail24, keys24) in {
        "willfalco/jpsequeira shared_h_v1": ({"quant_method": "modelopt"}, shared_tail,
                                             list(shared_keys)),
        "jpsequeira mxfp8 overlay": ({"quant_method": "exl3", "bits": 3.0}, ov_tail,
                                     list(shared_keys)),
        "brandonmusic r7_shared + dense6": (nr_qc, nr_tail, list(nr_keys)),
    }.items():
        cfg24 = {"quantization_config": qc, "hybrid_tr3_tail": tail24}
        ctrl = mc._candidate_decode_plan(qc, cfg24, index_keys=keys24)
        pod = lo.trellis_checkpoint_plan(_TailConfig(qc, tail24), keys24)
        pod.pop("_observed")
        check("[24] %s: controller and pod agree on the layout contract" % label,
              ctrl["quantization_config"] == pod,
              "ctrl %r pod %r" % (ctrl["quantization_config"], pod))
    try:
        mc._candidate_decode_plan({"quant_method": "modelopt"},
                                  {"quantization_config": {"quant_method": "modelopt"},
                                   "hybrid_tr3_tail": shared_tail})
        ok, detail = False, "no refusal"
    except mc.Refusal as exc:
        ok, detail = "index was not available" in str(exc), str(exc)[:160]
    check("[24] the controller refuses to bind an exl3 candidate without the index names",
          ok, detail)
    try:
        mc._candidate_decode_plan({"quant_method": "modelopt"},
                                  {"quantization_config": {"quant_method": "modelopt"},
                                   "hybrid_tr3_tail": tail}, index_keys=list(shared_keys))
        ok, detail = False, "no refusal"
    except mc.Refusal as exc:
        ok, detail = "requires 'shared_h_v1'" in str(exc), str(exc)[:160]
    check("[24] the controller refuses an undeclared shared_h layout at $0, as the pod would",
          ok, detail)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\nselftest_trellis_decode_offline: %d passed, %d failed"
          % (passed, len(RESULTS) - passed))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
