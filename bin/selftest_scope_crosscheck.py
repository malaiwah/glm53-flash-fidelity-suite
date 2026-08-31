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

print()
if FAILED:
    print("selftest_scope_crosscheck: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_scope_crosscheck: all passed")
