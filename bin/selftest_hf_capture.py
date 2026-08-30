#!/usr/bin/env python3
"""T7 -- the portable capture engine, end to end, on real (tiny) weights.

Every other selftest in this repo builds its fixtures from hand-written JSON and
hand-packed safetensors bytes: they prove the FORMAT, never the CAPTURE.  This
one runs the whole three-step architecture against an actual torch model, which
is the gap that let `fidelity-dataset capture` ship for weeks while writing no
dataset at all.

    A1  capture twice, cold, from the same weights -> two sealed datasets
    A2  both verify (seal + checksums + every tensor content digest)
    A3  self-compare A vs A' == exactly 0.0 via the hash proof
    A4  self-compare A vs A' == exactly 0.0 with --force-compute (real matmul)
    A5  a toy quantizer produces B; B captures and verifies
    A6  compare A vs B is a MEASUREMENT with a nonzero KLD
    A7  the emitted submission is accepted by the registry validator
    A8  a tampered dataset is refused
    A9  the storage claim (hidden vs logit form) is arithmetic, not a slogan
    A10 capture REFUSES a candidate role with no scope description
    A11 `fidelity-dataset capture --engine hf-transformers` writes --out
    A12 the capture post-condition refuses an --out that holds no dataset

torch and transformers are optional: without them the file prints SKIP and
exits 0, so `bin/selftest_all.sh` on the numpy-only floor is unaffected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

BIN = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BIN)
sys.path.insert(0, BIN)

PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def run(argv, **kwargs):
    proc = subprocess.run([sys.executable] + argv, capture_output=True, text=True, **kwargs)
    return proc


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def tiny_model(path, vocab=64, hidden=16, layers=2, seed=0):
    """A real, tiny, randomly initialised causal LM saved as a checkpoint."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                         num_hidden_layers=layers, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)
    return path


def tiny_panel(path, windows=3, length=12, vocab=64, seed=1):
    """A panel tree in the upstream `quant-pipeline.glm53-token-panel.v1` layout."""
    import numpy as np

    sys.path.insert(0, BIN)
    from fidelity import dsformat as F

    arrays = os.path.join(path, "arrays")
    os.makedirs(arrays, exist_ok=True)
    rng = np.random.RandomState(seed)
    mask = np.ones(length, dtype=np.uint8)
    mask_path = os.path.join(arrays, "causal-mask-%d.npy" % length)
    np.save(mask_path, mask, allow_pickle=False)
    rows = []
    for index in range(windows):
        ids = rng.randint(0, vocab, size=length).astype(np.int32)
        token_path = os.path.join(arrays, "final-%04d.tokens.npy" % index)
        np.save(token_path, ids, allow_pickle=False)
        rows.append({"window_id": "final-%04d" % index, "role": "final",
                     "domain": "axis1_general", "document_id": "doc-%d" % index,
                     "prediction_positions": length - 1,
                     "token_ids_sha256": F.sha256_file(token_path),
                     "attention_mask_sha256": F.sha256_file(mask_path)})
    with open(os.path.join(path, "panel.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": "quant-pipeline.glm53-token-panel.v1",
                   "sealed_corpus_sha256": None, "windows": rows}, handle, indent=2)
    return path


def toy_quantize(src, dst, bits=4):
    """Round-to-nearest, per-output-row scale, on the MLP down_proj only.

    Deliberately crude and deliberately narrow: the point is a candidate that
    differs from the reference by a scheme that can be stated in one sentence.
    """
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if not name.endswith(".safetensors"):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    tensors = load_file(os.path.join(src, "model.safetensors"))
    levels = 2 ** (bits - 1) - 1
    touched = []
    for key, value in tensors.items():
        if "down_proj.weight" not in key:
            continue
        wide = value.float()
        scale = wide.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / levels
        tensors[key] = (torch.round(wide / scale) * scale).to(value.dtype)
        touched.append(key)
    save_file(tensors, os.path.join(dst, "model.safetensors"))
    return touched


SCOPE = {
    "policy": "mixed", "head_policy": "native", "kv_cache_dtype": "bf16",
    "assignments": [
        {"tensor_class": "mlp.down", "treatment": "quantized", "format": "int4",
         "bits_per_weight": 4, "layer_range": None},
        {"tensor_class": "embed_tokens", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "attn.qkv", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "attn.o", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "mlp.gate", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "mlp.up", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "moe.experts", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "norm", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
        {"tensor_class": "lm_head", "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None},
    ],
}


def capture(model, panel, out, *, role, dataset_id, name, scope_file=None, extra=(),
            via_wrapper=False):
    """Drive the engine directly, or through `fidelity-dataset capture --engine`."""
    tail = ["--model", model, "--panel", panel, "--dataset-id", dataset_id,
            "--dataset-name", name, "--device", "cpu",
            "--weights-repository", "selftest/tiny", "--model-revision", "0" * 40]
    if scope_file:
        tail += ["--scope-file", scope_file]
    tail += list(extra)
    if via_wrapper:
        return run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "capture",
                    "--out", out, "--form", "hidden", "--role", role,
                    "--lane", "local-cuda-budget", "--engine", "hf-transformers",
                    "--"] + tail)
    return run([os.path.join(REPO, "k6", "tools", "hf_capture.py"),
                "--out", out, "--role", role, "--lane", "local-cuda-budget"] + tail)


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        print("SKIP selftest_hf_capture: torch/transformers unavailable (%s)" % exc)
        return 0

    work = tempfile.mkdtemp(prefix="hfcap-")
    try:
        return _body(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _body(work):
    from fidelity import dsformat as F

    model = tiny_model(os.path.join(work, "reference"))
    panel = tiny_panel(os.path.join(work, "panel"))

    # -- A1 ------------------------------------------------------------------
    a = os.path.join(work, "ds-a")
    b = os.path.join(work, "ds-a2")
    first = capture(model, panel, a, role="root", dataset_id="fidelity--selftest.hf.root",
                    name="selftest root")
    second = capture(model, panel, b, role="root", dataset_id="fidelity--selftest.hf.root",
                     name="selftest root")
    check("A1 two cold captures exit 0", first.returncode == 0 and second.returncode == 0,
          (first.stderr or second.stderr)[-400:])
    if first.returncode != 0:
        print(first.stdout[-2000:], first.stderr[-2000:])
        return 1

    # -- A2 ------------------------------------------------------------------
    verify_a = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "verify", a])
    check("A2 dataset A verifies", verify_a.returncode == 0, verify_a.stdout[-500:])

    manifest_a = F.read_json(os.path.join(a, F.MANIFEST_NAME))
    manifest_b = F.read_json(os.path.join(b, F.MANIFEST_NAME))
    check("A2b two cold runs agree on capture_content_digest",
          manifest_a["capture"]["capture_content_digest"]
          == manifest_b["capture"]["capture_content_digest"])

    # -- A3 / A4 -------------------------------------------------------------
    for name, extra in (("A3 self-compare (hash proof)", []),
                        ("A4 self-compare (--force-compute)", ["--force-compute"])):
        out = os.path.join(work, name.split()[0])
        proc = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "compare",
                    "--reference", a, "--candidate", b, "--out", out, "--self-compare"] + extra)
        ok = proc.returncode in (0, 2)
        value = None
        if os.path.isfile(os.path.join(out, "comparison-receipt.json")):
            receipt = F.read_json(os.path.join(out, "comparison-receipt.json"))
            value = receipt["metric"]["value"]
            ok = (ok and value == 0.0 and str(value) != "-0.0"
                  and receipt["comparison_kind"] == "reproduction_confirmation")
        else:
            ok = False
        check(name + " == exactly 0.0", ok, "rc=%s value=%r %s"
              % (proc.returncode, value, proc.stdout[-400:]))

    # -- A5 ------------------------------------------------------------------
    quant_dir = os.path.join(work, "candidate")
    touched = toy_quantize(model, quant_dir)
    check("A5a the toy quantizer touched some tensors", bool(touched), str(touched))
    scope_file = os.path.join(work, "scope.json")
    with open(scope_file, "w", encoding="utf-8") as handle:
        json.dump(SCOPE, handle)
    c = os.path.join(work, "ds-b")
    third = capture(quant_dir, panel, c, role="quant",
                    dataset_id="fidelity--selftest.hf.quant", name="selftest quant",
                    scope_file=scope_file,
                    extra=["--codec", "rtn-int4-per-row", "--declared-bits", "4"])
    check("A5b candidate captures", third.returncode == 0, third.stderr[-400:])
    verify_c = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "verify", c])
    check("A5c candidate verifies", verify_c.returncode == 0, verify_c.stdout[-500:])

    # -- A6 / A7 -------------------------------------------------------------
    provenance = os.path.join(work, "provenance.json")
    prov = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "provenance-template"])
    if prov.returncode == 0:
        with open(provenance, "w", encoding="utf-8") as handle:
            handle.write(prov.stdout)
    out = os.path.join(work, "cmp-ab")
    proc = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "compare",
                "--reference", a, "--candidate", c, "--out", out])
    receipt_path = os.path.join(out, "comparison-receipt.json")
    ok = proc.returncode in (0, 2) and os.path.isfile(receipt_path)
    value = None
    kind = None
    if ok:
        receipt = F.read_json(receipt_path)
        value = receipt["metric"]["value"]
        kind = receipt.get("comparison_kind")
    check("A6 A vs B is a nonzero measurement",
          ok and kind == "measurement" and value is not None and value > 0.0,
          "rc=%s kind=%r value=%r %s" % (proc.returncode, kind, value, proc.stdout[-400:]))

    validate = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "validate",
                    "--receipt", receipt_path]) if os.path.isfile(receipt_path) else None
    check("A7 the receipt validates", validate is not None and validate.returncode in (0, 2),
          validate.stdout[-400:] if validate else "no receipt")

    # -- A8 ------------------------------------------------------------------
    tampered = os.path.join(work, "ds-tampered")
    shutil.copytree(a, tampered)
    victim = os.path.join(tampered, "capture", "hidden_0000.safetensors")
    with open(victim, "r+b") as handle:
        handle.seek(-2, os.SEEK_END)
        last = handle.read(2)
        handle.seek(-2, os.SEEK_END)
        handle.write(bytes([last[0] ^ 0xFF, last[1]]))
    broken = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "verify", tampered])
    check("A8 a flipped capture byte is refused", broken.returncode == 3,
          "rc=%s %s" % (broken.returncode, broken.stdout[-300:]))

    # -- A9 ------------------------------------------------------------------
    cap = manifest_a["capture"]
    rows = cap["scored_rows_total"]
    hidden_bytes = cap["total_size_bytes"]
    logit_bytes = rows * cap["vocab_size"] * 4
    check("A9 hidden form is smaller than logit form by hidden*2 : vocab*4",
          hidden_bytes < logit_bytes
          and abs(rows * cap["hidden_width"] * 2 - hidden_bytes) < 4096 * cap["records_count"],
          "rows=%d hidden=%d logit=%d" % (rows, hidden_bytes, logit_bytes))

    # -- A10 -----------------------------------------------------------------
    refused = capture(quant_dir, panel, os.path.join(work, "ds-noscope"), role="quant",
                      dataset_id="x", name="x")
    check("A10 a candidate with no --scope-file is refused", refused.returncode == 3,
          "rc=%s %s" % (refused.returncode, refused.stderr[-300:]))

    # -- A11 -----------------------------------------------------------------
    # `capture --out X` must leave a dataset at X. The wrapper used to exit 0
    # having written nothing there at all.
    wrapped = os.path.join(work, "ds-wrapped")
    proc = capture(model, panel, wrapped, role="root",
                   dataset_id="fidelity--selftest.hf.root", name="selftest root",
                   via_wrapper=True)
    check("A11 fidelity-dataset capture --engine hf-transformers writes --out",
          proc.returncode in (0, 2)
          and os.path.isfile(os.path.join(wrapped, F.MANIFEST_NAME))
          and "SEALED DATASET" in proc.stdout,
          "rc=%s %s" % (proc.returncode, proc.stdout[-400:]))

    # -- A12 -----------------------------------------------------------------
    # The post-condition must FIRE, not just pass: a capture that leaves no
    # dataset is a refusal, never a green exit.
    empty = os.path.join(work, "ds-empty")
    os.makedirs(empty)
    sys.path.insert(0, BIN)
    import fidelity_dataset

    code = fidelity_dataset._postcondition(empty)
    check("A12 the post-condition refuses an --out with no manifest", code == 3,
          "got %r" % code)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED %s: %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
