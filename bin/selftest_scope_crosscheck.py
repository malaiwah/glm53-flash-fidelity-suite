#!/usr/bin/env python3
"""`--scope-json` must describe THE release being measured, not a sibling.

The flag copies its file verbatim into the sealed receipt and into the
artifact record, and its own help says the file "must be READ off the release,
never assumed".  Until this gate existed nothing enforced that, and the first
turboderp 2.05bpw receipt was built with the 4.05bpw branch's scope: same
repository, same class names, every quantized rate 1-2 bits too high.
`scope_digest` is computed over those assignments and the comparability key
over the digest, so the row would have been filed under a group describing a
recipe the artifact does not have.

Offline by construction: the release's published header is injected, so this
proves the COMPARISON, not the network.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_cloud as mc                                # noqa: E402

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


class Con:
    def ok(self, *a):
        pass

    def warn(self, *a):
        pass


def surface(head_bits, kind="exl3hf"):
    return type("S", (), {"surface": kind, "evidence": {"head_bits": head_bits},
                          "problems": []})()


def release(expert_bits=2, attn_bits=4, head_bits=5):
    """A minimal published quantization_config in the real emit's shape."""
    ts = {"lm_head": {"bits_per_weight": head_bits}}
    for layer in range(2):
        ts["model.language_model.layers.%d.self_attn.qkv_proj" % layer] = {
            "bits_per_weight": attn_bits}
        for e in range(3):
            ts["model.language_model.layers.%d.mlp.experts.%d.up_proj" % (layer, e)] = {
                "bits_per_weight": expert_bits}
    return {"quant_method": "exl3", "bits": 2.05, "head_bits": head_bits,
            "tensor_storage": ts}


def scope(expert_bits=2, attn_bits=4, head_bits=5):
    return {"policy": "mixed", "head_policy": "quantized",
            "assignments": [
                {"tensor_class": "moe.experts", "treatment": "quantized",
                 "format": "exl3-mul1", "bits_per_weight": expert_bits},
                {"tensor_class": "attn.qkv", "treatment": "quantized",
                 "format": "exl3-mul1", "bits_per_weight": attn_bits},
                {"tensor_class": "lm_head", "treatment": "quantized",
                 "format": "exl3-mul1", "bits_per_weight": head_bits},
                {"tensor_class": "norm", "treatment": "native",
                 "format": "bf16", "bits_per_weight": 16},
            ]}


def run(sc, rel, head_bits=5, kind="exl3hf"):
    """Returns None when accepted, the Refusal message when refused."""
    served = {"n": 0}

    def fake_fetch_json(repo_id, path, **kw):
        served["n"] += 1
        assert path == "quantization_config.json", path
        return rel

    real, mc.fetch_json = mc.fetch_json, fake_fetch_json
    try:
        mc._refuse_scope_contradicted_by_release(
            Con(), "turboderp/GLM-5.3-Flash-exl3", "51058cd5", surface(head_bits, kind),
            sc, {})
        return None
    except mc.Refusal as exc:
        # The class-by-class detail lives in the refusal's advice lines, which
        # are what a contributor actually reads.
        return "\n".join([str(exc)] + [str(a) for a in (getattr(exc, "advice", None) or [])])
    finally:
        mc.fetch_json = real


print("== --scope-json must match the release it claims to describe ==")

check("the release's own scope is ACCEPTED",
      run(scope(), release()) is None)

# The bug as it actually happened: the 4.05bpw branch's file, every quantized
# rate too high.  The head alone would catch this one -- the next case proves
# the gate does not merely check the head.
msg = run(scope(expert_bits=4, attn_bits=6, head_bits=6), release(), head_bits=5)
check("a sibling branch's scope is REFUSED", msg is not None)

# Head agrees, experts do not.  A head-only gate passes this and publishes a
# 2-bit artifact under a 4-bit recipe.
msg = run(scope(expert_bits=4), release(), head_bits=5)
check("REFUSES a wrong expert rate even when the head agrees", msg is not None)
check("...and names the class, the claim and the truth",
      bool(msg) and "moe.experts" in msg)

msg = run(scope(attn_bits=6), release(), head_bits=5)
check("REFUSES a wrong attention rate", msg is not None)

check("no --scope-json is a no-op (the honest 'unknown' default)",
      run(None, release()) is None)

# A class the header does not speak to is NOT decidable, and inventing a
# refusal there would block every release whose emit is older or sparser.
sc = scope()
sc["assignments"].append({"tensor_class": "mtp", "treatment": "quantized",
                          "format": "exl3-mul1", "bits_per_weight": 2})
check("a class the release does not publish is not a refusal",
      run(sc, release()) is None)

# Surfaces other than exl3hf publish no per-class rate; the head check is the
# whole gate there, and it must still fire.
check("a non-exl3 surface still gets the head check",
      run(scope(head_bits=6), release(), head_bits=5, kind="tr3-published") is not None)

print("\n== the class map reaches every module the real release ships ==")
for name, want in [
        ("model.language_model.layers.3.self_attn.kv_a_proj_with_mqa", "attn.qkv"),
        ("model.language_model.layers.0.self_attn.qkv_proj", "attn.qkv"),
        ("model.language_model.layers.0.self_attn.o_proj", "attn.o"),
        ("model.language_model.layers.3.mlp.experts.7.up_proj", "moe.experts"),
        ("model.language_model.layers.3.mlp.shared_experts.up_proj", "moe.shared_expert"),
        ("model.language_model.layers.3.mlp.gate", "moe.router"),
        ("model.language_model.layers.0.mlp.gate_proj", "mlp.gate"),
        ("lm_head", "lm_head"),
        ("model.language_model.embed_tokens", "embed_tokens"),
        ("model.visual.blocks.0.attn.qkv_proj", "other"),
]:
    check("%-24s -> %s" % (want, name.split(".")[-1]),
          mc._scope_class_of(name) == want)

print("\n== the run's own output is the last line of defence ==")
import seal_receipt as sr                                 # noqa: E402


class SealCon(Con):
    def err(self, *a):
        pass


def sealed(sc, head_bits, hist):
    summary = {"declared_head_bits": head_bits, "routed_bits_decode_histogram": hist}
    return sr._refuse_scope_contradicted_by_run(sc, summary, SealCon())


# The shape of the run that actually happened: a 2-bit artifact measured with
# the 4-bit sibling's scope.
conf = sealed(scope(expert_bits=4, attn_bits=6, head_bits=6), 5, {"K2": 907200})
check("a sibling's scope is REFUSED against the run's own decode census",
      bool(conf))
check("...naming the head it read", any("declared_head_bits 5" in c for c in conf))
check("...and the rate it decoded", any("K2" in c for c in conf))

check("the run's own scope seals cleanly",
      sealed(scope(), 5, {"K2": 907200}) == [])
check("no scope is a no-op at seal time too",
      sealed(None, 5, {"K2": 907200}) == [])
check("a run that recorded no census does not invent a conflict",
      sealed(scope(), None, {}) == [])
# A mixed-rate routed census is legitimate; membership, not equality.
check("a mixed routed census accepts any rate it contains",
      sealed(scope(expert_bits=3), 5, {"K2": 900, "K3": 100}) == [])

print("\n== the scope file is schema-checked before the rental, not after it ==")
import contextlib                                          # noqa: E402
import io                                                  # noqa: E402
import tempfile                                            # noqa: E402
from pathlib import Path as _P                             # noqa: E402

SUITE = _P(mc.SUITE_ROOT)
REAL = SUITE / "engines" / "exl3hf-evidence" / "scope-turbo-2.05bpw.json"
if not REAL.is_file():
    REAL = SUITE / "engines" / "tools" / "exl3hf-evidence" / "scope-turbo-2.05bpw.json"


def schema_check(doc):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(doc, fh)
        tmp = fh.name
    try:
        mc._validate_scope_json(Con(), tmp)
        return None
    except mc.Refusal as exc:
        return "\n".join([str(exc)] + [str(a) for a in (exc.advice or [])])
    finally:
        os.unlink(tmp)


base = json.loads(REAL.read_text())
check("the committed 2.05bpw scope satisfies the submission schema",
      schema_check(base) is None)

# The exact failure that cost the turbo-2.05bpw re-run its seal stage after
# 2 h 06 m of measuring: a property the submission schema does not allow.
withderiv = dict(base, derivation={"tool": "x"})
msg = schema_check(withderiv)
check("a scope carrying an extra property is REFUSED", bool(msg))
check("...naming the property, as the seal-time validator would",
      bool(msg) and "derivation" in msg)
check("...and saying what it would otherwise have cost",
      bool(msg) and "whole rental" in msg)

check("a scope missing head_policy is REFUSED",
      schema_check({k: v for k, v in base.items() if k != "head_policy"})
      is not None)

print("\n== the authoring tools read the nested VL stack (GLM-5.3-Flash) ==")
# GLM-5.3-Flash is `Glm5NextForConditionalGeneration`: its decoder stack is
# `model.language_model.layers.N.` beside `model.visual.*`, and its geometry is
# `config.text_config`. Before this rung the class rules matched
# `^model\.layers\.` only, so every one of the K3's 1207 named tensors was
# `other` and the scope file described nothing.
sys.path.insert(0, str(SUITE / "engines" / "tools"))
import fp8_scope                                            # noqa: E402
import exl3_scope                                           # noqa: E402

LM = "model.language_model."
for name, want in [
    (LM + "layers.3.mlp.experts.7.gate_proj.weight", "moe.experts"),
    (LM + "layers.3.mlp.gate.e_score_correction_bias", "moe.router"),
    (LM + "layers.0.mlp.down_proj.weight", "mlp.down"),
    (LM + "layers.4.self_attn.o_proj.weight", "attn.o"),
    (LM + "layers.4.self_attn.f_a_proj.weight", "attn.other"),
    (LM + "layers.4.input_layernorm.weight", "norm"),
    (LM + "embed_tokens.weight", "embed_tokens"),
    (LM + "norm.weight", "norm"),
    (LM + "layers.45.mlp.experts.7.gate_proj.weight", "mtp"),
    (LM + "layers.45.eh_proj.weight", "mtp"),
    ("model.visual.blocks.0.attn.qkv.weight", "other"),
    ("lm_head.weight", "lm_head"),
    ("model.layers.3.mlp.experts.7.gate_proj.weight", "moe.experts"),
    ("model.layers.78.eh_proj.weight", "mtp"),
]:
    check("%-52s -> %s" % (name, want), fp8_scope.classify(name, 45) == want)
check("layer_of reads the nested stack's index",
      fp8_scope.layer_of(LM + "layers.45.enorm.weight") == 45
      and fp8_scope.layer_of("model.layers.78.enorm.weight") == 78
      and fp8_scope.layer_of("model.visual.blocks.0.norm1.weight") is None)
check("decoder_layers reads text_config when the top level has none",
      fp8_scope.decoder_layers({"text_config": {"num_hidden_layers": 45}}) == 45
      and fp8_scope.decoder_layers({"num_hidden_layers": 78}) == 78)
try:
    fp8_scope.decoder_layers({"text_config": {}})
    check("decoder_layers refuses a config with no layer count", False)
except SystemExit as exc:
    check("decoder_layers refuses a config with no layer count", "num_hidden_layers" in str(exc))


def flash_mini_index():
    """The K3 layout in miniature: dense 0..2, routed 3..4, MTP block 5, vision tower."""
    keys = ["lm_head.weight", LM + "embed_tokens.weight", LM + "norm.weight",
            "model.visual.blocks.0.attn.qkv.weight", "model.visual.patch_embed.proj.weight"]
    for layer in range(6):
        keys += [LM + "layers.%d.input_layernorm.weight" % layer,
                 LM + "layers.%d.self_attn.o_proj.weight" % layer,
                 LM + "layers.%d.hc_attn_base" % layer]
        if layer < 3:
            keys += [LM + "layers.%d.mlp.%s.weight" % (layer, p)
                     for p in ("gate_proj", "up_proj", "down_proj")]
            continue
        keys += [LM + "layers.%d.mlp.gate.weight" % layer]
        for expert in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                stem = LM + "layers.%d.mlp.experts.%d.%s" % (layer, expert, proj)
                keys += [stem + "." + obj for obj in ("trellis", "suh", "svh", "mcg")]
    keys += [LM + "layers.5.eh_proj.weight", LM + "layers.5.enorm.weight"]
    return {"metadata": {"total_size": 1}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}


def author_flash_scope():
    with tempfile.TemporaryDirectory() as tmp:
        root = _P(tmp)
        (root / "index.json").write_text(json.dumps(flash_mini_index()))
        (root / "config.json").write_text(json.dumps({
            "architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next",
            "quantization_config": {"quant_method": "exl3", "bits": 3, "codebook": "mcg"},
            "text_config": {"num_hidden_layers": 5, "n_routed_experts": 2}}))
        with contextlib.redirect_stdout(io.StringIO()):
            rc = exl3_scope.main(["--index", str(root / "index.json"), "--config", str(root / "config.json"),
                                  "--repo", "example/flash-k3", "--revision", "0" * 40,
                                  "--out", str(root / "scope.json")])
        return rc, json.loads((root / "scope.json").read_text())


rc, flash = author_flash_scope()
rows = {(r["tensor_class"], r["layer_range"]): r for r in flash["assignments"]}
check("exl3_scope authors the nested stack without refusing", rc == 0)
check("routed experts are one quantized exl3-mcg row on layers 3-4",
      rows.get(("moe.experts", "all"), {}).get("format") == "exl3-mcg"
      and rows[("moe.experts", "all")]["treatment"] == "quantized"
      and rows[("moe.experts", "all")]["bits_per_weight"] == 3.0
      and rows[("moe.experts", "all")]["note"].startswith("12 tensors:"))
check("the MTP block (layer 5 of a 5-layer stack) is class mtp, mixed by the bytes",
      ("mtp", "5") in rows and rows[("mtp", "5")]["format"] == "mixed"
      and flash["mtp_included"] is True)
check("dense mlp / attention / norm / embed / head classes are reached",
      {c for c, _ in rows} >= {"mlp.gate", "mlp.up", "mlp.down", "attn.o", "norm",
                               "embed_tokens", "lm_head", "moe.router"})
other = rows.get(("other", "all"), {})
check("`other` holds only the vision tower and the hyper-connection scalars, not the stack",
      other.get("note", "").startswith("7 tensors:")
      and "model.visual" in other.get("note", "")
      and "experts" not in other.get("note", ""))
check("the head is native and the scope is schema-valid at the pre-spend gate",
      flash["head_policy"] == "native" and schema_check(flash) is None)
# A class stored at two NATIVE widths (bf16 gate weight beside an fp32 router
# bias) was written `treatment: quantized, format: mixed` -- a quantizer that
# touched nothing there, labelled as if it had. The treatment is what was done
# to the class; only the format is mixed.
native_two_widths = fp8_scope.assignments_from_census(
    {"moe.router": [(LM + "layers.3.mlp.gate.weight", "native", "bf16", 16),
                    (LM + "layers.3.mlp.gate.e_score_correction_bias", "native", "fp32", 32)],
     "moe.experts": [(LM + "layers.3.mlp.experts.0.up_proj.weight", "quantized", "exl3-mcg", 3.0),
                     (LM + "layers.3.mlp.experts.0.up_proj.weight_scale_inv", "native", "fp32", 32)]},
    "test")
by_class = {r["tensor_class"]: r for r in native_two_widths}
check("an all-native class at two widths is treatment native, format mixed",
      by_class["moe.router"]["treatment"] == "native"
      and by_class["moe.router"]["format"] == "mixed")
check("a class mixing quantized and native stays treatment quantized, format mixed",
      by_class["moe.experts"]["treatment"] == "quantized"
      and by_class["moe.experts"]["format"] == "mixed")

print()
if FAILED:
    print("selftest_scope_crosscheck: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_scope_crosscheck: all passed")
