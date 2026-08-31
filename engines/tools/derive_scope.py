#!/usr/bin/env python3
"""Derive a fidelity-dataset scope file from a checkpoint's OWN config and
weight index -- never from a guess.

The registry's `unknown_scope()` exists for third-party artifacts whose recipe
was never published.  For the artifacts measured here the recipe IS published:
`config.json:quantization_config` names the method, the bit width and the
exclusion list, and the weight index names every tensor that actually exists.
Writing "unknown" for something the producer published would be a fabricated
gap, so this tool reads both and reports what is there.

Method
------
1. Read `quantization_config`.  Two published shapes are handled:
   * ``quant_method: "fp8"`` -- ``modules_to_not_convert`` is an explicit list
     of module prefixes left in the original dtype.
   * ``quant_method: "compressed-tensors"`` -- ``config_groups[*].targets`` say
     what is quantized (typically ``["Linear"]``) and ``ignore`` says what is
     not.
2. Read the safetensors weight index (or the single-file header) for the real
   tensor names.
3. Assign each registry tensor_class by matching the model's actual parameter
   names against that class, then decide `quantized` vs `native` from step 1.
   A class with no matching tensor is reported as absent rather than invented.

Everything the tool concluded, and the evidence it concluded it from, is
written into the scope file's `derivation` block so a reader can re-check it.
"""

import argparse
import glob
import json
import os
import re
import sys

# registry tensor_class vocabulary (registry/tools/seed_registry.py)
CLASS_PATTERNS = [
    ("embed_tokens", [r"embed_tokens"]),
    ("lm_head", [r"(^|\.)lm_head"]),
    ("norm", [r"norm", r"A_log", r"dt_bias"]),
    ("attn.qkv", [r"self_attn\.(q|k|v)_proj", r"attn\.(q|k|v)_proj",
                  r"linear_attn\.in_proj"]),
    ("attn.o", [r"self_attn\.o_proj", r"attn\.o_proj", r"linear_attn\.out_proj",
                r"linear_attn\.conv1d"]),
    ("mlp.gate", [r"mlp\.gate(_proj)?(\.|$)", r"shared_expert_gate"]),
    ("mlp.up", [r"mlp\.up_proj", r"mlp\.linear_fc1"]),
    ("mlp.down", [r"mlp\.down_proj", r"mlp\.linear_fc2"]),
    ("moe.experts", [r"experts\.\d+\.", r"experts\."]),
]


def classify(name):
    """First matching class wins; order above is most-specific-first."""
    for cls, pats in CLASS_PATTERNS:
        for p in pats:
            if re.search(p, name):
                return cls
    return None


def load_tensor_names(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        doc = json.loads(open(idx).read())
        return sorted(doc["weight_map"].keys())
    names = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        import struct
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n).decode("utf-8"))
        names.extend(k for k in header if k != "__metadata__")
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-cache-dtype", default="bf16")
    args = ap.parse_args()

    config = json.loads(open(os.path.join(args.model_dir, "config.json")).read())
    qc = config.get("quantization_config") or {}
    method = qc.get("quant_method")
    names = load_tensor_names(args.model_dir)

    # ---- text-tower tensors only; the vision tower is never scored ----------
    text = [n for n in names if "visual" not in n and "vision" not in n]

    # ---- which tensors carry quantization state ----------------------------
    # A module's parameters may be stored either as a plain `.weight` (native)
    # or split into quantization state.  `pack-quantized` in particular emits
    # NO `.weight` at all -- the payload is `.weight_packed` -- so a probe that
    # only looks at `.weight` sees a quantized module as absent and the class
    # falsely reads native.  Work in terms of MODULE BASES instead: strip every
    # known parameter suffix, then ask whether that module carries quant state.
    QUANT_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".weight_packed",
                      ".weight_zero_point", ".weight_shape", ".weight_g_idx",
                      ".qweight", ".qzeros", ".scales", ".weight_scale_2",
                      "_scale_inv")
    PLAIN_SUFFIXES = (".weight", ".bias")

    def base_of(name):
        for suf in QUANT_SUFFIXES:
            if name.endswith(suf):
                return name[: -len(suf)].rstrip("."), True
        for suf in PLAIN_SUFFIXES:
            if name.endswith(suf):
                return name[: -len(suf)], False
        return name, False

    quantized_bases = set()
    module_bases = {}
    for n in text:
        b, is_q = base_of(n)
        module_bases.setdefault(b, set()).add(n)
        if is_q:
            quantized_bases.add(b)

    # `format` must come from the registry's numeric_format enum, or
    # registry_validate.py --submission REJECTS the scope (SCOPE-VOCAB). The
    # exact scheme goes in `declared_scheme` / a disclosure, not here.
    if method == "fp8":
        fmt = "fp8_e4m3" if (qc.get("fmt") or "e4m3") == "e4m3" else "fp8_e5m2"
        bits = 8
        scheme = "fp8 block-wise, fmt=%s, weight_block_size=%s, activation_scheme=%s" % (
            qc.get("fmt"), qc.get("weight_block_size"), qc.get("activation_scheme"))
        excluded = [m for m in (qc.get("modules_to_not_convert") or [])
                    if "visual" not in m]
        exclusion_rule = "config.quantization_config.modules_to_not_convert"
    elif method == "compressed-tensors":
        groups = qc.get("config_groups") or {}
        g0 = groups.get(sorted(groups)[0]) if groups else {}
        w = (g0 or {}).get("weights") or {}
        bits = int(w.get("num_bits") or 0) or None
        fmt = {4: "int4", 8: "int8"}.get(bits, "unknown")
        scheme = ("compressed-tensors %s, num_bits=%s, group_size=%s, "
                  "observer=%s, actorder=%s"
                  % (g0.get("format"), bits, w.get("group_size"),
                     w.get("observer"), w.get("actorder")))
        excluded = [m for m in (qc.get("ignore") or []) if "visual" not in m]
        exclusion_rule = "config.quantization_config.ignore"
    elif not qc:
        fmt, bits, excluded = "bf16", 16, []
        scheme = "unquantized"
        exclusion_rule = "no quantization_config: this is an unquantized checkpoint"
    else:
        sys.stderr.write("derive_scope: unhandled quant_method %r\n" % method)
        return 4

    # ---- per-class verdict from the real tensor names ----------------------
    assignments = []
    evidence = {}
    for cls, _ in CLASS_PATTERNS:
        members = [n for n in text if classify(n) == cls]
        if not members:
            evidence[cls] = {"tensors": 0, "verdict": "absent"}
            continue
        # A class is quantized iff its MODULES carry quantization state.
        probe = sorted({base_of(n)[0] for n in members})
        q = sum(1 for b in probe if b in quantized_bases)
        partial = False
        if qc and q == len(probe) and q > 0:
            treatment, f, b = "quantized", fmt, bits
        elif (not qc) or q == 0:
            treatment, f, b = "native", "bf16", 16
        else:
            # Genuinely part-quantized class.  The registry treatment enum is
            # {quantized, native, reconstructed, removed, not_present, unknown}
            # -- there is no "mixed", and inventing one would fail the schema
            # and corrupt scope_digest.  This model is HYBRID: full-attention
            # layers' projections are quantized while the linear-attention
            # (conv1d / in_proj_*) families are left in bf16, and both land in
            # the same coarse tensor_class.  We record the load-bearing verdict
            # -- every layer's main projection IS quantized -- and name the
            # native families exactly, in the derivation and in a disclosure,
            # rather than rounding the class to "unknown" and losing the fact
            # that the producer did publish this recipe.
            treatment, f, b, partial = "quantized", fmt, bits, True
        assignments.append({"tensor_class": cls, "treatment": treatment,
                            "format": f, "bits_per_weight": b,
                            "layer_range": None})
        row = {"tensors": len(members), "modules": len(probe),
               "modules_with_quant_state": q, "verdict": treatment,
               "example": members[0]}
        if partial:
            native_mods = [b for b in probe if b not in quantized_bases]
            fams = sorted({re.sub(r"\.\d+\.", ".*.", b).split("layers.*.")[-1]
                           for b in native_mods})
            row["partial"] = True
            row["native_modules"] = len(native_mods)
            row["native_families"] = fams[:12]
        evidence[cls] = row

    head = [a for a in assignments if a["tensor_class"] == "lm_head"]
    head_policy = "native" if (head and head[0]["treatment"] == "native") else (
        "quantized" if head else "native")
    policy = "none" if not qc else "mixed"

    layer_types = ((config.get("text_config") or {}).get("layer_types")
                   or config.get("layer_types") or [])
    n_linear = sum(1 for t in layer_types if t == "linear_attention")
    n_full = sum(1 for t in layer_types if t == "full_attention")

    partials = {c: e for c, e in evidence.items() if e.get("partial")}
    disclosures = []
    if partials:
        disclosures.append({
            "code": "partial_class_quantization",
            "severity": "info",
            "affects_comparability": False,
            "detail": (
                "This checkpoint is a HYBRID: of its %d text layers, %d use linear "
                "attention and %d full attention. The producer quantized every "
                "layer's main projection but left the linear-attention families "
                "(%s) in bf16. The registry tensor_class vocabulary is coarser than "
                "that split, so classes %s are recorded 'quantized' with this note "
                "rather than 'unknown': the recipe IS published, and scope.derivation "
                "carries the exact per-class tensor counts."
                % (len(layer_types), n_linear, n_full,
                   ", ".join(sorted({f for e in partials.values()
                                     for f in e.get("native_families", [])}))[:300],
                   ", ".join(sorted(partials)))),
        })

    doc = {
        "disclosures": disclosures,
        "policy": policy,
        "head_policy": head_policy,
        "kv_cache_dtype": args.kv_cache_dtype,
        "assignments": assignments,
        "derivation": {
            "tool": "derive_scope.py",
            "source": "the artifact's own config.json + safetensors weight index",
            "quant_method": method,
            "declared_format": fmt,
            "declared_scheme": scheme,
            "declared_bits": bits,
            "exclusion_rule": exclusion_rule,
            "excluded_text_modules": len(excluded),
            "excluded_sample": excluded[:12],
            "text_tensors_total": len(text),
            "tensors_with_quant_state": len(quantized_bases),
            "per_class_evidence": evidence,
            "note": ("vision-tower tensors are excluded from this scope: the panel "
                     "is text-only, no image or video token is ever fed, so the "
                     "vision tower does not participate in any scored forward pass."),
        },
    }
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"out": args.out, "quant_method": method,
                      "policy": policy, "head_policy": head_policy,
                      "assignments": {a["tensor_class"]: "%s:%s" % (a["treatment"], a["format"])
                                      for a in assignments}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
