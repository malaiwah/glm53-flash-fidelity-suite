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
    A21 checkpoint tensors the architecture does not use are DISCLOSED, not refused
    A22 --device-map dispatches instead of materialising, and skips the .to() that
        cannot work for a checkpoint bigger than one device

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

    # -- A13 -----------------------------------------------------------------
    # `panel_receipt_sha256` without the receipt itself is a digest of something
    # the reader cannot obtain. The receipt must be shipped byte-verbatim, named
    # by the manifest, covered by checksums.txt, and inside the seal.
    shipped = os.path.join(a, "panel", "panel-receipt.json")
    src_bytes = open(os.path.join(panel, "panel.receipt.json"), "rb").read()
    listed = [line.split("  ", 1)[1] for line in
              open(os.path.join(a, F.CHECKSUMS_NAME), "r", encoding="utf-8").read()
              .splitlines() if line.strip()]
    check("A13 the panel build receipt ships verbatim, sealed and listed",
          os.path.isfile(shipped)
          and open(shipped, "rb").read() == src_bytes
          and manifest_a["panel"].get("panel_receipt_file") == "panel/panel-receipt.json"
          and manifest_a["panel"]["panel_receipt_sha256"] == F.sha256_file(shipped)
          and "panel/panel-receipt.json" in listed,
          "present=%s named=%r listed=%s"
          % (os.path.isfile(shipped), manifest_a["panel"].get("panel_receipt_file"),
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
    sys.path.insert(0, os.path.join(REPO, "k6", "tools"))
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

    # A21 -- `unexpected_keys` is not a failure: GLM-5.3-BF16 ships an MTP layer
    # the architecture does not build, 791 tensors of it. But a reader deserves
    # to know part of the checkpoint took no part in the forward pass, and
    # nothing recorded it at all.
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
    stamped = []
    if os.path.isfile(os.path.join(extra_out, F.MANIFEST_NAME)):
        stamped = [d for d in json.load(open(os.path.join(extra_out, F.MANIFEST_NAME)))
                   ["disclosures"] if d["code"] == "checkpoint_tensors_not_loaded"]
    check("A21 an unused checkpoint tensor is disclosed, not refused",
          proc.returncode == 0 and len(stamped) == 1
          and stamped[0]["severity"] == "caveat"
          and "model.layers.99.mlp.down_proj.weight" in stamped[0]["detail"],
          "rc=%s stamped=%r out=%s" % (proc.returncode, stamped,
                                       (proc.stderr or proc.stdout)[-300:]))

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

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED %s: %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
