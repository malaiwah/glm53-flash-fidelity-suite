#!/usr/bin/env python3
"""scope_apply_provenance rewrites exactly what the evidence proves, and refuses the rest.

The drowzeys record read `attn.qkv=native:fp16@16` for a year of the tree's
life because the scope tool labels a class by its STORED dtype, while the
committed zero-pad evidence and the ZERO_PAD_METHOD comment already said the
same rows were `dequantize_block_fp8(zai-org/GLM-5.3)` cast to fp16. Two
committed records contradicted each other and both were published. The
evidence file settles it per class; this selftest proves the applier

  [1] rewrites a covered 16-bit native class to quantized:fp8_e4m3@8 and keeps
      the original census inside the note beside the evidence path;
  [2] leaves every uncovered class byte-identical;
  [3] refuses a class the evidence does not cover, an evidence verdict that is
      not the FP8 one, a sampled tensor whose own result contradicts the
      verdict, a covered class with no sampled tensor of its own, a covered
      assignment that is already quantized, and a scope authored from a
      different repository or revision;
  [4] moves scope_digest, so a corrected artifact record cannot keep its old
      digest by accident;
  [5] the committed drowzeys evidence + the committed drowzeys scope agree:
      every covered class in engines/scopes/scope--drowzeys-exl3.json is
      quantized:fp8_e4m3@8 with the evidence path in its note, and re-applying
      the evidence to the scope refuses (it is no longer a native row).

Offline: the evidence is a fixture in the shape nonrouted_provenance.py writes.
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "registry", "tools"))
import scope_apply_provenance as SAP  # noqa: E402
import registry_lib as L  # noqa: E402

FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


def refuses(label, fn, needle):
    try:
        fn()
    except SAP.ApplyError as exc:
        check("%s (refused: %s)" % (label, str(exc)[:90]), needle in str(exc))
        return
    check(label + " (did not refuse)", False)


CAND = ("drowzeys/keys-GLM-5.3-EXL3", "ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5")
FP8 = ("zai-org/GLM-5.3", "187fb9fff6319062325ff825627ef6db084d9bc6")
SOURCE = ("read from %s@%s model.safetensors.index.json + shard headers: a class is exl3 trellis "
          "when its weights are stored as trellis/suh/svh payload groups, fp8_e4m3 when a "
          "_scale_inv sibling exists, native otherwise (stored dtype from the shard headers)." % CAND)


def row(cls, treatment, fmt, bits, layer_range="all", note=None):
    return {"tensor_class": cls, "treatment": treatment, "format": fmt, "bits_per_weight": bits,
            "layer_range": layer_range, "note": note or ("4 tensors: model.layers.N.x.weight. " + SOURCE)}


def scope():
    return {
        "policy": "mixed", "head_policy": "native", "kv_cache_dtype": "bf16",
        "activation_quantization": None, "mtp_included": True,
        "assignments": [
            row("attn.qkv", "native", "fp16", 16),
            row("attn.o", "native", "fp16", 16),
            row("attn.other", "native", "mixed", None, "0-77"),
            row("embed_tokens", "native", "bf16", 16),
            row("lm_head", "native", "fp16", 16),
            row("mlp.down", "native", "fp16", 16),
            row("moe.experts", "quantized", "exl3-mul1", 3.0, "4-77"),
            row("moe.router", "native", "mixed", None, "3-77"),
            row("moe.shared_expert", "native", "fp16", 16),
            row("norm", "native", "bf16", 16),
        ],
    }


def tensor_result(rows=576):
    return {"rows_compared": rows, "elements_compared": rows * 64, "stored_dtype": "F16",
            SAP.EQ_ROOT: False, SAP.EQ_FP8: True,
            "n_diff_vs_bf16_root": 12345, "n_diff_vs_fp8_dequant": 0,
            "max_abs_diff_vs_bf16_root": 0.004, "max_abs_diff_vs_fp8_dequant_fp32": 3e-5}


def evidence(covers=("attn.qkv", "attn.o", "mlp.down", "moe.shared_expert")):
    return {
        "schema": SAP.EVIDENCE_SCHEMA, "verdict": SAP.FP8_VERDICT,
        "covers_classes": list(covers),
        "candidate": {"repo": CAND[0], "revision": CAND[1]},
        "bf16_root": {"repo": "zai-org/GLM-5.3-BF16", "revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639"},
        "fp8_release": {"repo": FP8[0], "revision": FP8[1]},
        "tensors": {
            "model.layers.0.self_attn.q_a_proj.weight": tensor_result(),
            "model.layers.0.self_attn.kv_b_proj.weight": tensor_result(),
            "model.layers.40.self_attn.o_proj.weight": tensor_result(),
            "model.layers.1.mlp.down_proj.weight": tensor_result(),
            "model.layers.3.mlp.shared_experts.gate_proj.weight": tensor_result(),
        },
    }


def main() -> int:
    ev_path = "engines/tools/layer-outer-evidence/fixture.json"

    print("[1] a covered 16-bit native class becomes quantized:fp8_e4m3@8 with the census kept")
    before = scope()
    after = SAP.apply(before, evidence(), ev_path)
    by = {a["tensor_class"]: a for a in after["assignments"]}
    for cls in ("attn.qkv", "attn.o", "mlp.down", "moe.shared_expert"):
        a = by[cls]
        check("%s treatment/format/bits" % cls,
              (a["treatment"], a["format"], a["bits_per_weight"]) == ("quantized", "fp8_e4m3", 8))
        check("%s note names the FP8 release, the evidence and the original census" % cls,
              "dequantize_block_fp8(%s@%s)" % (FP8[0], FP8[1][:8]) in a["note"]
              and ev_path in a["note"] and "Original census: 4 tensors" in a["note"]
              and "stored fp16" in a["note"])
    check("attn.qkv note lists its own sampled tensors, not another class's",
          "q_a_proj" in by["attn.qkv"]["note"] and "kv_b_proj" in by["attn.qkv"]["note"]
          and "o_proj" not in by["attn.qkv"]["note"].split("Original census")[0])

    print("[2] uncovered classes are untouched, the input is not mutated")
    for cls in ("attn.other", "embed_tokens", "lm_head", "moe.experts", "moe.router", "norm"):
        check("%s unchanged" % cls,
              by[cls] == next(a for a in scope()["assignments"] if a["tensor_class"] == cls))
    check("input scope not mutated", before == scope())
    check("top-level fields unchanged", all(after[k] == before[k] for k in
                                            ("policy", "head_policy", "kv_cache_dtype", "mtp_included")))

    print("[3] refusals")
    refuses("a class the evidence does not cover", lambda: SAP.apply(scope(), evidence(), ev_path, ["moe.router"]),
            "not covered")
    ev = evidence()
    ev["verdict"] = "stored_16bit_of_bf16_root"
    refuses("a non-FP8 verdict", lambda: SAP.apply(scope(), ev, ev_path), "verdict")
    ev = evidence()
    ev["tensors"]["model.layers.40.self_attn.o_proj.weight"]["n_diff_vs_fp8_dequant"] = 3
    refuses("a sampled tensor contradicting the verdict", lambda: SAP.apply(scope(), ev, ev_path), "does not carry")
    ev = evidence(covers=("attn.qkv", "mlp.gate"))
    refuses("a covered class with no sampled tensor", lambda: SAP.apply(scope(), ev, ev_path), "none of its sampled")
    sc = scope()
    sc["assignments"][0] = row("attn.qkv", "quantized", "fp8_e4m3", 8)
    refuses("a covered assignment that is already quantized", lambda: SAP.apply(sc, evidence(), ev_path),
            "not a 16-bit native row")
    ev = evidence()
    ev["candidate"]["revision"] = "0" * 40
    refuses("a scope authored from another revision", lambda: SAP.apply(scope(), ev, ev_path), "not authored from")
    ev = evidence()
    ev["schema"] = "something-else"
    refuses("a foreign evidence schema", lambda: SAP.apply(scope(), ev, ev_path), "schema")
    ev = evidence()
    ev["covers_classes"] = []
    refuses("evidence covering nothing", lambda: SAP.apply(scope(), ev, ev_path), "covers no classes")

    print("[4] scope_digest moves")
    old_d, new_d = L.scope_digest(scope()), L.scope_digest(after)
    check("digest changed", old_d != new_d)
    check("new digest carries attn.qkv=quantized:fp8_e4m3@8", "attn.qkv=quantized:fp8_e4m3@8" in new_d)
    check("new digest keeps moe.router=native:mixed", "moe.router=native:mixed" in new_d)

    print("[5] the committed drowzeys scope and evidence agree")
    ev_rel = "engines/tools/layer-outer-evidence/drowzeys-nonrouted-provenance.json"
    real_ev = json.load(open(os.path.join(HERE, "..", "..", ev_rel)))
    real_scope = json.load(open(os.path.join(HERE, "..", "scopes", "scope--drowzeys-exl3.json")))
    check("committed evidence carries the FP8 verdict over >= 10 tensors",
          real_ev["verdict"] == SAP.FP8_VERDICT and len(real_ev["tensors"]) >= 10)
    check("committed evidence covers the six non-routed classes",
          set(real_ev["covers_classes"]) == {"attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.down",
                                             "moe.shared_expert"})
    check("every sampled tensor is bitwise the FP8 dequant and not the BF16 root",
          all(t[SAP.EQ_FP8] and not t[SAP.EQ_ROOT] and t["n_diff_vs_fp8_dequant"] == 0
              for t in real_ev["tensors"].values()))
    for cls in real_ev["covers_classes"]:
        rows = [a for a in real_scope["assignments"] if a["tensor_class"] == cls]
        check("committed scope %s is quantized:fp8_e4m3@8 citing the evidence" % cls,
              rows and all((a["treatment"], a["format"], a["bits_per_weight"]) == ("quantized", "fp8_e4m3", 8)
                           and ev_rel in a["note"] for a in rows))
    for cls in ("moe.router", "attn.other", "mtp"):
        rows = [a for a in real_scope["assignments"] if a["tensor_class"] == cls]
        check("committed scope %s is treatment native (all-native census)" % cls,
              rows and all(a["treatment"] == "native" for a in rows))
    refuses("re-applying the evidence to the corrected scope",
            lambda: SAP.apply(real_scope, real_ev, ev_rel), "not a 16-bit native row")

    print()
    if FAILED:
        print("FAILED: %d" % len(FAILED))
        for f in FAILED:
            print("  - " + f)
        return 1
    print("selftest_scope_provenance: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
