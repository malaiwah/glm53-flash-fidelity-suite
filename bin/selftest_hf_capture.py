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
    A13 the panel's build receipt ships verbatim inside the seal
    A14 a checkpoint transformers could not fully read is REFUSED, not captured
    A15 --allow-missing-weights stamps a BLOCKING disclosure instead
    A16 --base-capture repo@rev becomes the object the schema requires
    A17 a load report with `mismatched_keys` is REFUSED ("Reinit due to size mismatch")
    A18 a load report with `conversion_errors` is refused and is NOT overridable
    A19 a load that produced NO report is refused; unexamined != clean
    A20 `conversion_errors` is actually visible after a real load (the field
        `LoadStateDictInfo.to_dict()` deliberately drops)
    A21 checkpoint tensors the architecture does not use REFUSE the capture
    A22 --device-map dispatches instead of materialising, and skips the .to() that
        cannot work for a checkpoint bigger than one device
    A23 --allow-unexpected-tensors captures instead, and stamps a BLOCKING disclosure
    A24 the unexpected-tensor branch reads its own flag, not --allow-missing-weights
    A25 the FP8 quantizer's parallel-plan crash is identified by its own frame
    A26 neutralize_parallel_plan empties tp+ep plans, sub-configs included
    A27 load_model refuses that crash by name and points at --drop-parallel-plan
    A28 a non-MIT root copies exact source-license bytes and binds them
    A29 source-license identity drift is refused before a dataset is sealed

torch and transformers are optional: without them the file prints SKIP and
exits 0, so `bin/selftest_all.sh` on the numpy-only floor is unaffected.
"""

from __future__ import annotations
import hashlib

import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

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
    # A real panel carries a build receipt saying how its tokens were selected.
    # The capture must ship it, not merely hash it (A13).
    with open(os.path.join(path, "panel.receipt.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": "malaiwah.token-panel-build-receipt.v1",
                   "selection_rule": "selftest fixture: %d windows of %d random ids"
                                     % (windows, length)}, handle, indent=2)
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
            via_wrapper=False, env=None):
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
                    "--"] + tail, env=env)
    return run([os.path.join(REPO, "engines", "tools", "hf_capture.py"),
                "--out", out, "--role", role, "--lane", "local-cuda-budget"] + tail, env=env)


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
    from fidelity import panel as panel_contract

    model = tiny_model(os.path.join(work, "reference"))
    panel = tiny_panel(os.path.join(work, "panel"))

    # -- A0 ------------------------------------------------------------------
    # A legacy panel tree initially carries only the RAW receipt-file digest
    # and a repository-like tokenizer id.  Qualification compares the emitted
    # semantic receipt identity and exact tokenizer to the resolved job binding.
    sys.path.insert(0, os.path.join(REPO, "engines", "tools"))
    import hf_capture as HFC  # noqa: WPS433
    resolved_tokenizer = {
        "repository": "selftest/tokenizer",
        "revision": "a" * 40,
        "id": "selftest/tokenizer",
        "vocab_size": 64,
        "files": [{"path": "tokenizer.json", "bytes": 17,
                   "sha256": "b" * 64}],
        "files_verified": True,
        "identity_sha256": "c" * 64,
    }
    raw_receipt_sha256 = "e" * 64

    def bind_fixture(mode, declared, raw=raw_receipt_sha256):
        panel_fixture = HFC.Panel(
            root=panel, panel_id="panel--selftest", source="panel.json",
            receipt_sha256=raw, windows=[],
            tokenizer={"repository": "selftest/tokenizer", "revision": None,
                       "vocab_size": 64})
        HFC.bind_resolved_panel_tokenizer(
            panel_fixture,
            SimpleNamespace(panel_binding_evidence={"binding": {
                "tokenizer": resolved_tokenizer,
                "receipt": {
                    "declared_receipt_sha256": declared,
                    "receipt_file_sha256": raw_receipt_sha256,
                    "receipt_seal_mode": mode,
                },
            }}))
        return panel_fixture

    modern_panel = bind_fixture("self-blank", "d" * 64)
    legacy_panel = bind_fixture("legacy-field-absent", "f" * 64)
    check("A0a both admitted receipt conventions emit their declared identity",
          modern_panel.receipt_sha256 == "d" * 64
          and legacy_panel.receipt_sha256 == "f" * 64
          and modern_panel.receipt_sha256 != raw_receipt_sha256
          and legacy_panel.receipt_sha256 != raw_receipt_sha256)
    check("A0b binding preserves the independently verified raw receipt digest",
          modern_panel.receipt_file_sha256 == raw_receipt_sha256
          and legacy_panel.receipt_file_sha256 == raw_receipt_sha256)
    check("A0c verified binding replaces the legacy null tokenizer revision",
          modern_panel.tokenizer == resolved_tokenizer
          and modern_panel.tokenizer["revision"] == "a" * 40,
          modern_panel.tokenizer)

    def bind_refused(mode, declared, binding_raw):
        try:
            bind_fixture(mode, declared, raw=binding_raw)
        except SystemExit:
            return True
        return False

    check("A0d malformed declared receipt identity refuses",
          bind_refused("self-blank", "D" * 64, raw_receipt_sha256))
    check("A0e binding raw receipt mismatch refuses",
          bind_refused("self-blank", "d" * 64, "0" * 64))
    check("A0f unknown receipt seal convention refuses",
          bind_refused("invented", "d" * 64, raw_receipt_sha256))

    # -- A28/A29 -------------------------------------------------------------
    license_source = os.path.join(work, "source-license", "LICENSE")
    os.makedirs(os.path.dirname(license_source), exist_ok=True)
    license_payload = b"fixture native-weights license and notice\n"
    with open(license_source, "wb") as handle:
        handle.write(license_payload)
    license_sha256 = hashlib.sha256(license_payload).hexdigest()
    licensed = os.path.join(work, "ds-licensed")
    licensed_capture = capture(
        model, panel, licensed, role="root",
        dataset_id="fidelity--selftest.hf.licensed-root",
        name="selftest licensed root",
        extra=[
            "--dataset-license", "other",
            "--weights-license-file", license_source,
            "--weights-license-sha256", license_sha256,
            "--weights-license-bytes", str(len(license_payload)),
        ])
    licensed_manifest = None
    licensed_runtime = None
    licensed_manifest_path = os.path.join(licensed, F.MANIFEST_NAME)
    if os.path.isfile(licensed_manifest_path):
        licensed_manifest = F.read_json(licensed_manifest_path)
        licensed_runtime = F.read_json(os.path.join(
            licensed, licensed_manifest["runtime"]["file"]))
    licensed_verify = run([
        os.path.join(REPO, "bin", "fidelity_dataset.py"),
        "verify", licensed])
    observed_license = (
        ((licensed_runtime or {}).get("capture_tool") or {})
        .get("weights_license"))
    check("A28 a non-MIT root copies and binds exact source-license bytes",
          licensed_capture.returncode == 0
          and licensed_verify.returncode == 0
          and open(os.path.join(licensed, "LICENSE"), "rb").read()
              == license_payload
          and (licensed_manifest or {}).get("dataset", {}).get("license")
              == "other"
          and observed_license == {
              "source_file": "LICENSE",
              "dataset_path": "LICENSE",
              "bytes": len(license_payload),
              "sha256": license_sha256,
          },
          "capture_rc=%s verify_rc=%s binding=%r stderr=%s"
          % (licensed_capture.returncode, licensed_verify.returncode,
             observed_license, licensed_capture.stderr[-300:]))
    drifted = os.path.join(work, "ds-license-drifted")
    drifted_capture = capture(
        model, panel, drifted, role="root",
        dataset_id="fidelity--selftest.hf.licensed-root",
        name="selftest drifted license",
        extra=[
            "--dataset-license", "other",
            "--weights-license-file", license_source,
            "--weights-license-sha256", "0" * 64,
            "--weights-license-bytes", str(len(license_payload)),
        ])
    check("A29 source-license identity drift refuses before dataset sealing",
          drifted_capture.returncode != 0
          and not os.path.exists(os.path.join(drifted, F.MANIFEST_NAME))
          and "--weights-license-file SHA-256 mismatch" in (
              drifted_capture.stderr + drifted_capture.stdout),
          "rc=%s output=%s"
          % (drifted_capture.returncode,
             (drifted_capture.stderr + drifted_capture.stdout)[-300:]))

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

    # -- A2e-A2g -------------------------------------------------------------
    # The numeric policy is OBSERVED on every capture and lives outside the
    # fingerprint digest: a default-policy capture keeps the published policy
    # string byte-for-byte (so its stack_fingerprint_sha256 still matches the
    # published root), and the runtime receipt records what the fp32 GEMMs
    # actually ran under. NVIDIA_TF32_OVERRIDE=1 is refused by name: on a CPU
    # box the flag is inert, which is exactly why it cannot be detected by effect.
    runtime_a = F.read_json(os.path.join(a, manifest_a["runtime"]["file"]))
    observed = (runtime_a.get("runtime_environment") or {}).get("numeric_policy_observed") or {}
    check("A2e the runtime receipt observes the numeric policy on a plain capture",
          set(observed) >= {"NVIDIA_TF32_OVERRIDE", "allow_tf32_matmul", "allow_tf32_cudnn",
                            "float32_matmul_precision", "deviates_from_default"}
          and observed.get("deviates_from_default") is False
          and runtime_a["stack_fingerprint"]["numeric_policy"]
          == getattr(HFC, "DEFAULT_NUMERIC_POLICY", None),
          json.dumps(observed))
    canonical = json.dumps(runtime_a["stack_fingerprint"], sort_keys=True, separators=(",", ":"))
    check("A2f the observed block is outside the fingerprint digest",
          "numeric_policy_observed" not in canonical
          and hashlib.sha256(canonical.encode("utf-8")).hexdigest()
          == manifest_a["runtime"]["stack_fingerprint_sha256"])
    tf32 = capture(model, panel, os.path.join(work, "ds-tf32"), role="root",
                   dataset_id="fidelity--selftest.hf.root", name="selftest root",
                   env=dict(os.environ, NVIDIA_TF32_OVERRIDE="1"))
    check("A2g NVIDIA_TF32_OVERRIDE=1 refuses the capture before any forward",
          tf32.returncode != 0 and "NVIDIA_TF32_OVERRIDE" in (tf32.stderr + tf32.stdout),
          (tf32.stderr + tf32.stdout)[-300:])

    # -- A2c -----------------------------------------------------------------
    # Every sealed text member must pass the publisher's private-path scan,
    # because the seal covers it and the publisher refuses the whole dataset.
    # The validator's verdict once named the output directory as its subject
    # (the pod's /workspace/... path), which made every capture unpublishable
    # and was found by a paid run at its final step.
    from fidelity import dshub
    verdict = F.read_json(os.path.join(a, "validation", "structural-validation.json"))
    leaked = []
    for root_dir, _dirs, files in os.walk(a):
        for filename in files:
            relpath = os.path.relpath(os.path.join(root_dir, filename), a)
            if not dshub._textual_publish_member(relpath):
                continue
            body = open(os.path.join(root_dir, filename), "rb").read()
            if any(pattern in body for pattern in dshub._PRIVATE_ABSOLUTE_PATHS):
                leaked.append(relpath)
    check("A2c the sealed validation verdict names the dataset, not its directory",
          verdict.get("subject") == "dataset:fidelity--selftest.hf.root",
          "subject=%r" % verdict.get("subject"))
    check("A2d no sealed text member carries a private absolute path",
          leaked == [], "leaked=%r" % leaked)

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

    # -- A14 -----------------------------------------------------------------
    # A checkpoint this transformers build cannot fully read does not fail: it
    # RANDOMLY INITIALISES the missing parameters and returns a running model.
    # Capturing that produces a confident number for weights nobody measured.
    import shutil as _shutil
    from safetensors.torch import load_file as _load, save_file as _save

    holed = os.path.join(work, "reference-holed")
    _shutil.copytree(model, holed)
    shard = os.path.join(holed, "model.safetensors")
    tensors = _load(shard)
    victim = sorted(k for k in tensors if k.endswith("mlp.down_proj.weight"))[-1]
    del tensors[victim]
    _save(tensors, shard, metadata={"format": "pt"})

    refused = capture(holed, panel, os.path.join(work, "ds-holed"), role="root",
                      dataset_id="fidelity--selftest.hf.holed", name="holed")
    check("A14 a checkpoint with randomly initialised parameters is refused",
          refused.returncode == 1
          and "randomly initialised" in (refused.stderr + refused.stdout)
          and victim in (refused.stderr + refused.stdout),
          "rc=%s %s" % (refused.returncode, (refused.stderr or refused.stdout)[-400:]))

    # -- A15 -----------------------------------------------------------------
    # The override exists, and it is not quiet: it stamps a BLOCKING disclosure.
    forced = os.path.join(work, "ds-holed-forced")
    proc = capture(holed, panel, forced, role="root",
                   dataset_id="fidelity--selftest.hf.holed", name="holed",
                   extra=["--allow-missing-weights"])
    stamped = []
    if os.path.isfile(os.path.join(forced, F.MANIFEST_NAME)):
        stamped = [d for d in json.load(open(os.path.join(forced, F.MANIFEST_NAME)))
                   ["disclosures"] if d["code"] == "randomly_initialised_weights"]
    check("A15 --allow-missing-weights stamps a BLOCKING disclosure",
          proc.returncode in (0, 2) and len(stamped) == 1
          and stamped[0]["severity"] == "blocking"
          and stamped[0]["affects_comparability"] is True,
          "rc=%s stamped=%r" % (proc.returncode, stamped))

    # -- A16 -----------------------------------------------------------------
    # `--base-capture repo@rev` must produce the schema's OBJECT. Written
    # through as a bare string it made every capture that named its intended
    # root fail the validator the capture itself runs.
    based = os.path.join(work, "ds-based")
    proc = capture(quant_dir, panel, based, role="quant",
                   dataset_id="fidelity--selftest.hf.quant", name="based",
                   scope_file=scope_file,
                   extra=["--base-capture", "selftest/root-v1@" + "a" * 40])
    block = None
    if os.path.isfile(os.path.join(based, F.MANIFEST_NAME)):
        block = json.load(open(os.path.join(based, F.MANIFEST_NAME)))["dataset"]["base_capture"]
    check("A16 --base-capture repo@rev becomes the schema's object",
          proc.returncode in (0, 2) and isinstance(block, dict)
          and block.get("repository") == "selftest/root-v1"
          and block.get("revision") == "a" * 40
          and "dataset_sha256" in block,
          "rc=%s block=%r" % (proc.returncode, block))

    # The raw receipt must ship byte-verbatim, be named by the manifest, be
    # covered by checksums.txt and sit inside the seal. This generic fixture is
    # intentionally unbound, so its traceability identity remains the raw
    # digest; paid qualification separately requires the resolved semantic/raw
    # pair and verifies both.
    shipped = os.path.join(a, "panel", "panel-receipt.json")
    src_bytes = open(os.path.join(panel, "panel.receipt.json"), "rb").read()
    listed = [line.split("  ", 1)[1] for line in
              open(os.path.join(a, F.CHECKSUMS_NAME), "r", encoding="utf-8").read()
              .splitlines() if line.strip()]
    runtime_a = F.read_json(os.path.join(a, manifest_a["runtime"]["file"]))
    binding_evidence = runtime_a["capture_tool"]["resolved_panel_binding"]
    if isinstance(binding_evidence, dict):
        receipt_binding = binding_evidence["binding"]["receipt"]
        try:
            panel_contract.verify_bound_panel_receipt_bytes(
                receipt_binding, src_bytes, "A13 panel receipt")
            receipt_binding_valid = True
        except panel_contract.PanelError:
            receipt_binding_valid = False
        expected_declared = receipt_binding["declared_receipt_sha256"]
        expected_file_sha = receipt_binding["receipt_file_sha256"]
    else:
        receipt_binding_valid = True
        expected_declared = F.sha256_file(shipped)
        expected_file_sha = expected_declared
    check("A13 the panel build receipt ships verbatim, sealed and listed",
          os.path.isfile(shipped)
          and open(shipped, "rb").read() == src_bytes
          and manifest_a["panel"].get("panel_receipt_file")
              == "panel/panel-receipt.json"
          and manifest_a["panel"]["panel_receipt_sha256"]
              == expected_declared
          and F.sha256_file(shipped) == expected_file_sha
          and receipt_binding_valid
          and "panel/panel-receipt.json" in listed,
          "present=%s named=%r listed=%s"
          % (os.path.isfile(shipped),
             manifest_a["panel"].get("panel_receipt_file"),
             "panel/panel-receipt.json" in listed))

    # -- A17..A21 ------------------------------------------------------------
    # The load report has FOUR ways to say "these parameters are not the
    # artifact's", and CAPTURE-03 used to read exactly one of them.
    #
    # A17-A20 are asserted against `hf_capture`'s own report reader rather than
    # end to end, deliberately: transformers 5.16.1 happens to RAISE on
    # `mismatched_keys` and `conversion_errors` from inside `from_pretrained`,
    # so on this build the two paths are indistinguishable from outside. The
    # guard must not depend on the library continuing to make that choice --
    # `ignore_mismatched_sizes=True`, an older build, or a future refactor all
    # hand the report back instead of raising, and then this reader is the only
    # thing standing between a randomly initialised tensor and a published
    # number.
    sys.path.insert(0, os.path.join(REPO, "engines", "tools"))
    import hf_capture as HC

    def _refused(report, allow_missing):
        """True when the guard REFUSES; a missing guard is not a refusal."""
        guard = getattr(HC, "refuse_on_load_report", None)
        if guard is None:
            return "no refuse_on_load_report in hf_capture"
        try:
            guard(report, allow_missing)
        except SystemExit:
            return True
        return False

    def _report(**fields):
        reader = getattr(HC, "load_report", None)
        if reader is None:
            return None
        doc = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(),
               "error_msgs": [], "conversion_errors": {},
               getattr(HC, "REPORT_OBSERVED", "_o"): True,
               getattr(HC, "REPORT_AUGMENTED", "_a"): True}
        doc.update(fields)
        return reader(doc)

    # A17 -- `mismatched_keys`: present in the checkpoint at the WRONG SHAPE.
    # transformers' own loading report calls this "Reinit due to size mismatch",
    # i.e. a randomly initialised tensor under another heading, and
    # `missing_weight_keys` used to ignore the field entirely.
    mismatch = [("model.layers.0.mlp.down_proj.weight", (16, 32), (16, 16))]
    seen = HC.missing_weight_keys({"missing_keys": [], "mismatched_keys": mismatch})
    check("A17 a mismatched (wrong-shape, reinitialised) key counts as missing",
          seen == ["model.layers.0.mlp.down_proj.weight"]
          and _refused(_report(mismatched_keys=mismatch), False) is True,
          "missing_weight_keys -> %r; refused -> %r"
          % (seen, _refused(_report(mismatched_keys=mismatch), False)))

    # A18 -- `conversion_errors`: the field `LoadStateDictInfo.to_dict()`
    # deliberately drops. For a fused-expert MoE checkpoint the converter owns
    # 96.7% of the tensors, so this is the field that matters most and the one
    # the guard was never shown. Not overridable: an exception mid-conversion
    # leaves the parameter's contents unknown.
    conv = {"model.layers.3.mlp.experts.gate_up_proj":
            "MergeModulelist, Concatenate: expected 256 tensors, got 255"}
    check("A18 conversion_errors are refused, and --allow-missing-weights does not "
          "override them",
          _refused(_report(conversion_errors=conv), False) is True
          and _refused(_report(conversion_errors=conv), True) is True,
          "refused=%r forced=%r" % (_refused(_report(conversion_errors=conv), False),
                                    _refused(_report(conversion_errors=conv), True)))

    # A19 -- no report at all. `_from_pretrained` used to return a bare `{}` on
    # its fallback path, which the guard read as "no missing keys": an
    # UNEXAMINED load and a CLEAN load had the same value.
    empty = HC.load_report({}) if getattr(HC, "load_report", None) else None
    check("A19 a load with NO report is refused even with --allow-missing-weights "
          "(unexamined is not clean)",
          _refused(empty, True) is True,
          "refused=%r (report=%r)" % (_refused(empty, True), empty))

    # A20 -- the wrap actually takes effect against the INSTALLED transformers.
    _m, _c, live_info = HC.load_model(model, "cpu", "bfloat16")
    reader = getattr(HC, "load_report", None)
    live = reader(live_info) if reader else {}
    check("A20 conversion_errors are visible after a real load",
          bool(live.get("observed")) and bool(live.get("conversion_errors_visible"))
          and live.get("conversion_errors") == {},
          "observed=%r visible=%r info_keys=%r"
          % (live.get("observed"), live.get("conversion_errors_visible"),
             sorted(live_info)))
    del _m

    # A21/A23 -- `unexpected_keys` used to be a log line and a caveat, and M1
    # paid for that. `Qwen/Qwen3.8-27B-FP8` loads through stock transformers
    # with `unexpected: 64` because the producer's `modules_to_not_convert`
    # lists `...mlp.gate` and `should_convert_module` matches it with a
    # start-anchored `re.match`, so `...mlp.gate_proj` matched too: 65 of 65
    # gate_proj modules skipped FP8 conversion, their block scales fell out as
    # "unexpected", and the projection ran on fp8 bytes read as bf16 with the
    # scale never applied. Nothing raised. The benign case -- GLM-5.3-BF16's
    # 791-tensor MTP layer that `GlmMoeDsaForCausalLM` does not build -- is
    # INDISTINGUISHABLE from here, which is why the escape exists and why it is
    # blocking.
    extra_dir = os.path.join(work, "reference-extra")
    _shutil.copytree(model, extra_dir)
    extra_shard = os.path.join(extra_dir, "model.safetensors")
    extra_tensors = _load(extra_shard)
    import torch as _torch

    extra_tensors["model.layers.99.mlp.down_proj.weight"] = _torch.zeros(
        (4, 4), dtype=_torch.bfloat16)
    _save(extra_tensors, extra_shard, metadata={"format": "pt"})
    extra_out = os.path.join(work, "ds-extra")
    proc = capture(extra_dir, panel, extra_out, role="root",
                   dataset_id="fidelity--selftest.hf.root", name="extra")
    combined = (proc.stderr or "") + (proc.stdout or "")
    check("A21 an unexpected checkpoint tensor REFUSES without an exact allowlist",
          proc.returncode != 0
          and not os.path.isfile(os.path.join(extra_out, F.MANIFEST_NAME))
          and "model.layers.99.mlp.down_proj.weight" in combined
          and "--unexpected-tensors-allowlist" in combined
          and "Broad acceptance is obsolete" in combined,
          "rc=%s manifest=%s out=%s"
          % (proc.returncode, os.path.isfile(os.path.join(extra_out, F.MANIFEST_NAME)),
             combined[-400:]))

    # A23 -- the obsolete broad boolean must never authorize a capture.
    extra_out2 = os.path.join(work, "ds-extra-forced")
    proc = capture(extra_dir, panel, extra_out2, role="root",
                   dataset_id="fidelity--selftest.hf.root", name="extra-forced",
                   extra=["--allow-unexpected-tensors"])
    combined = (proc.stderr or "") + (proc.stdout or "")
    check("A23 --allow-unexpected-tensors is a refused obsolete route",
          proc.returncode != 0 and "obsolete" in combined
          and not os.path.isfile(os.path.join(extra_out2, F.MANIFEST_NAME)),
          "rc=%s out=%s" % (proc.returncode, combined[-300:]))

    # A24 -- the only accepted route is equality with a byte- and
    # semantic-digest-bound exact list.
    key = "model.layers.99.mlp.down_proj.weight"
    allowlist_path = os.path.join(work, "unexpected-keys.json")
    allowlist_raw = (json.dumps([key], indent=2) + "\n").encode()
    open(allowlist_path, "wb").write(allowlist_raw)
    raw_sha = hashlib.sha256(allowlist_raw).hexdigest()
    name_sha = hashlib.sha256(json.dumps(
        [key], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    binding = HC.load_unexpected_tensor_allowlist(allowlist_path, raw_sha, name_sha)
    unexpected_report = _report(unexpected_keys={key})
    passed = HC.refuse_on_load_report(unexpected_report, False, binding) == []
    evidence = unexpected_report.get("unexpected_tensor_allowlist") or {}
    check("A24 exact unexpected-key equality passes with full evidence",
          _refused(_report(unexpected_keys={key}), False) is True and passed
          and evidence.get("expected_keys") == [key]
          and evidence.get("observed_keys") == [key]
          and evidence.get("missing_keys") == []
          and evidence.get("extra_keys") == []
          and evidence.get("exact_match") is True,
          "evidence=%r" % evidence)

    # -- A22 -----------------------------------------------------------------
    # R2 in docs/GLM53-ROOT-FEASIBILITY.md: `load_model` materialised the whole
    # model on CPU and then called `.to(device)`. For zai-org/GLM-5.3-BF16 that
    # is 1,486.8 GB -- more than the largest rentable RAM (300 GB) and more than
    # the entire VRAM of an 8x H200 node (1,128 GB) -- so the default path
    # cannot load the root model on any machine we can rent. `device_map`
    # dispatches instead, and the post-load `.to()` MUST be skipped: calling it
    # on a dispatched model raises.
    try:
        import accelerate  # noqa: F401
        have_accelerate = True
    except Exception:
        have_accelerate = False
    if have_accelerate:
        try:
            dm_model, _dc, dm_info = HC.load_model(model, "cpu", "bfloat16",
                                                   device_map={"": "cpu"})
            dm_report = HC.load_report(dm_info)
            dm_logits = dm_model(input_ids=_torch.tensor([[1, 2, 3]])).logits
            check("A22 --device-map dispatches, skips .to(), and still reports the load",
                  tuple(dm_logits.shape)[:2] == (1, 3)
                  and dm_report["observed"] is True
                  and dm_report["conversion_errors_visible"] is True
                  and dm_report["missing_keys"] == [],
                  "logits=%r report=%r" % (tuple(dm_logits.shape), dm_report))
            del dm_model
        except SystemExit as exc:
            check("A22 --device-map dispatches, skips .to(), and still reports the load",
                  False, "refused: %s" % exc.code)
        except TypeError as exc:
            check("A22 --device-map dispatches, skips .to(), and still reports the load",
                  False, "load_model has no device_map parameter: %s" % exc)
    else:
        check("A22 --device-map dispatches, skips .to(), and still reports the load",
              True, "SKIPPED: accelerate not installed")

    # -- A25/A26/A27 ---------------------------------------------------------
    # The FP8 parallel-plan defect.  `transformers` 5.16.1's
    # `FineGrainedFP8HfQuantizer.update_tp_plan` does
    #
    #     layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
    #     updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
    #
    # `_impl_tp_layer_overrides` has ONE key (`deepgemm_megamoe`) and `impl` is
    # always None at that point, so any FP8 config with a non-empty parallel
    # plan raises `AttributeError: 'NoneType' object has no attribute 'get'`
    # BEFORE a single weight is read.  Every FP8 `deepseek_v4` repo is in that
    # set (`DeepseekV4Config.base_model_ep_plan` has 7 entries), i.e.
    # `deepseek-ai/DeepSeek-V4-Flash-0731` and its 100 quant children were
    # unloadable and the message the operator saw was a bare NoneType
    # AttributeError with no repo, no cause and no remedy.

    def _raise_like_transformers():
        """Reproduce the exception with a frame that looks like the real one."""
        import types

        source = ("def update_tp_plan(self, config):\n"
                  "    layer_overrides = None\n"
                  "    return {k: layer_overrides.get(v, v) for k, v in "
                  "config.items()}\n")
        module = types.ModuleType("quantizer_finegrained_fp8")
        code = compile(source, "/x/transformers/quantizers/quantizer_finegrained_fp8.py",
                       "exec")
        exec(code, module.__dict__)
        try:
            module.update_tp_plan(None, {"layers.*.mlp.experts": "grouped_gemm"})
        except AttributeError as exc:
            return exc
        return None

    # These four rungs are new in M2; on a tree that predates the fix the names
    # they exercise do not exist. Resolve them defensively so that tree reports
    # four honest FAILs instead of an AttributeError that stops the battery.
    _is_bug = getattr(HC, "_is_fp8_tp_plan_bug", None) or (lambda exc: None)
    _neutralize = getattr(HC, "neutralize_parallel_plan", None) or (lambda cfg: None)

    real = _raise_like_transformers()
    other = None
    try:
        None.get("x")               # same message, unrelated frame
    except AttributeError as exc:
        other = exc
    check("A25 the FP8 parallel-plan crash is recognised by frame, not by message "
          "(a lookalike AttributeError elsewhere is NOT it)",
          real is not None and _is_bug(real) is True
          and other is not None and _is_bug(other) is False,
          "real=%r other=%r" % (real, other))

    class _Cfg(object):
        pass

    cfg = _Cfg()
    cfg.base_model_ep_plan = {"layers.*.mlp.experts": "grouped_gemm"}
    cfg.base_model_tp_plan = {}
    cfg.text_config = _Cfg()
    cfg.text_config.base_model_tp_plan = {"layers.*.self_attn.q_proj": "colwise"}
    emptied = _neutralize(cfg)
    check("A26 neutralize_parallel_plan empties tp AND ep plans, recurses into "
          "text_config, and names exactly what it emptied",
          cfg.base_model_ep_plan == {} and cfg.text_config.base_model_tp_plan == {}
          and sorted(emptied or []) == ["base_model_ep_plan",
                                        "text_config.base_model_tp_plan"],
          "emptied=%r" % (emptied,))

    # A27: the refusal is reachable through `load_model` and names the remedy.
    # Driven by making `_from_pretrained` raise the real exception, so the test
    # needs neither an FP8 checkpoint nor a GPU.
    original = HC._from_pretrained
    seen = {}

    def _boom(cls, model_dir, torch_dtype, **extra):
        seen["config_passed"] = "config" in extra
        raise _raise_like_transformers()

    import contextlib
    import io

    def _refusal_text(**kwargs):
        """`fail()` prints to stderr and returns SystemExit(1); the MESSAGE is
        the thing under test, so read stderr rather than the exit code."""
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            try:
                HC.load_model(model, "cpu", "bfloat16", **kwargs)
            except SystemExit:
                pass
            except TypeError as exc:        # pre-M2 signature: no such keyword
                print("load_model: %s" % exc, file=buffer)
        return buffer.getvalue()

    HC._from_pretrained = _boom
    try:
        message = _refusal_text()
        check("A27 load_model REFUSES the FP8 parallel-plan crash by name, and points "
              "at --drop-parallel-plan instead of a bare NoneType AttributeError",
              "--drop-parallel-plan" in message and "update_tp_plan" in message
              and "deepgemm_megamoe" in message,
              message[:240])
        # ... and under the flag it stops diagnosing and hands the edited config
        # to from_pretrained, so a real load would get past the quantizer.
        seen.clear()
        second = _refusal_text(drop_parallel_plan=True)
        check("A27b under --drop-parallel-plan the edited config is handed to "
              "from_pretrained (otherwise config.json is re-read and the edit is lost)",
              seen.get("config_passed") is True and "--drop-parallel-plan" not in second,
              "config_passed=%r second=%r" % (seen.get("config_passed"), second[:160]))
    finally:
        HC._from_pretrained = original

    # -- A33 -----------------------------------------------------------------
    # A candidate whose tokenizer_config.json differs from the bound root's by
    # the one admitted loader key captures under --panel-binding: the binding
    # stays the root's, and the sealed dataset carries the
    # `tokenizer_config_loader_keys_ignored` disclosure with both digests.
    # Any other difference refuses the capture by name (the pod's gate).
    from fidelity import panel as PC
    bound_panel = os.path.join(work, "panel-bound")
    shutil.copytree(panel, bound_panel)
    root_tokens = os.path.join(work, "root-tokenizer")
    os.makedirs(root_tokens)
    root_config = b'{"model_max_length": 16, "tokenizer_class": "TokenizersBackend"}\n'
    for name, raw in (("tokenizer.json", b'{"version": "1"}\n'),
                      ("tokenizer_config.json", root_config)):
        open(os.path.join(root_tokens, name), "wb").write(raw)
    receipt_doc = json.load(open(os.path.join(bound_panel, "panel.receipt.json")))
    receipt_doc["tokenizer"] = {
        "repository": "selftest/root-weights", "revision": "a" * 40, "vocab_size": 64,
        "files_sha256": {name: F.sha256_file(os.path.join(root_tokens, name))
                         for name in ("tokenizer.json", "tokenizer_config.json")}}
    receipt_doc["receipt_sha256"] = ""
    receipt_doc["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt_doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    json.dump(receipt_doc, open(os.path.join(bound_panel, "panel.receipt.json"), "w"),
              indent=2, sort_keys=True)
    binding_doc = PC.resolve_panel(bound_panel, tokenizer_root=root_tokens).to_dict()
    binding_path = os.path.join(work, "panel.binding.json")
    binding_raw = json.dumps(binding_doc, sort_keys=True, separators=(",", ":")).encode()
    open(binding_path, "wb").write(binding_raw)
    binding_sha = hashlib.sha256(binding_raw).hexdigest()

    def candidate_tokens(label, config_bytes, with_reference=True):
        path = os.path.join(work, "cand-tokens-" + label)
        shutil.copytree(root_tokens, path)
        open(os.path.join(path, "tokenizer_config.json"), "wb").write(config_bytes)
        if with_reference:
            os.makedirs(os.path.join(path, PC.TOKENIZER_REFERENCE_SUBDIR))
            shutil.copy2(os.path.join(root_tokens, "tokenizer_config.json"),
                         os.path.join(path, PC.TOKENIZER_REFERENCE_SUBDIR, "tokenizer_config.json"))
        return path

    loader_only = b'{"local_files_only": false, "model_max_length": 16, ' \
                  b'"tokenizer_class": "TokenizersBackend"}\n'
    bound_out = os.path.join(work, "ds-bound-loader-key")
    proc = capture(quant_dir, bound_panel, bound_out, role="quant",
                   dataset_id="fidelity--selftest.hf.quant", name="bound-loader-key",
                   scope_file=scope_file,
                   extra=["--panel-binding", binding_path, "--panel-binding-sha256", binding_sha,
                          "--panel-tokenizer-root", candidate_tokens("loader", loader_only),
                          "--no-sanity-check"])
    manifest_path = os.path.join(bound_out, F.MANIFEST_NAME)
    disclosures = []
    sealed_binding = None
    if os.path.isfile(manifest_path):
        manifest_doc = json.load(open(manifest_path))
        disclosures = manifest_doc.get("disclosures") or []
        runtime_doc = F.read_json(os.path.join(bound_out, manifest_doc["runtime"]["file"]))
        sealed_binding = (runtime_doc["capture_tool"].get("resolved_panel_binding") or {}).get("binding")
    ignored = [d for d in disclosures if d.get("code") == "tokenizer_config_loader_keys_ignored"]
    check("A33 a candidate tokenizer_config.json differing by local_files_only ONLY captures "
          "under --panel-binding, the sealed binding is the root's, and the dataset carries "
          "the tokenizer_config_loader_keys_ignored disclosure (info, non-comparability) "
          "with both digests and the dropped key",
          proc.returncode in (0, 2) and sealed_binding == binding_doc
          and len(ignored) == 1 and ignored[0]["severity"] == "info"
          and ignored[0]["affects_comparability"] is False
          and hashlib.sha256(root_config).hexdigest() in ignored[0]["detail"]
          and hashlib.sha256(loader_only).hexdigest() in ignored[0]["detail"]
          and "['local_files_only']" in ignored[0]["detail"]
          and "never re-tokenized" in ignored[0]["detail"],
          "rc=%s stderr=%s disclosures=%r" % (proc.returncode, proc.stderr[-300:],
                                              [d.get("code") for d in disclosures]))
    other = capture(quant_dir, bound_panel, os.path.join(work, "ds-bound-other-key"), role="quant",
                    dataset_id="fidelity--selftest.hf.quant", name="bound-other-key",
                    scope_file=scope_file,
                    extra=["--panel-binding", binding_path, "--panel-binding-sha256", binding_sha,
                           "--panel-tokenizer-root", candidate_tokens(
                               "other", b'{"local_files_only": false, "model_max_length": 32, '
                                        b'"tokenizer_class": "TokenizersBackend"}\n'),
                           "--no-sanity-check"])
    check("A33b ... and a candidate that also changes model_max_length is REFUSED by key "
          "name before any capture arithmetic",
          other.returncode != 0 and "model_max_length" in (other.stderr or "")
          and not os.path.isfile(os.path.join(work, "ds-bound-other-key", F.MANIFEST_NAME)),
          "rc=%s stderr=%s" % (other.returncode, (other.stderr or "")[-300:]))

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED %s: %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
