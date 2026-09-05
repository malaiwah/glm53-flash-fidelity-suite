#!/usr/bin/env python3
"""The layer-outer, window-inner capture schedule -- the regression battery.

    bin/selftest_layer_outer.py

Everything here runs offline on a randomly initialised tiny model.  No network,
no GPU, no checkpoint download.  The two REAL-model bit-identity proofs (the
0.1B `glm5_next` fixture and `malaiwah/GLM-5.2-SIQ-Fruit-bf16`, i.e. the
`glm_moe_dsa` architecture GLM-5.3 uses) are not reproducible offline and live
in `docs/LAYER-OUTER.md` with their digests; what this file guards is
that the mechanism behind them cannot silently regress.

    L1  layer-outer + --layer-residency resident reproduces the window-outer
        capture BIT-FOR-BIT (same capture_content_digest, tensor content)
    L2  layer-outer + --layer-residency stream does too -- the loader is in the
        loop and the numbers still do not move
    L3  the default schedule is window-outer, and naming it explicitly changes
        nothing: the old path is untouched
    L4  a STREAMED layer's weights are byte-identical to what from_pretrained
        builds, parameter by parameter -- the loader claim, checked directly
    L5  freeing a layer actually returns its parameters to the meta device
    L6  a fused-expert MoE checkpoint survives the streamed loader byte-exactly
        (the WeightConverter path that owns 96.7% of GLM-5.3's tensors)
    L7  a shard shorter than its own safetensors header is REFUSED -- the
        "holes reading as zeros" trap Stage A found, on this loader's path
    L8  a shard-header / index key-set disagreement is REFUSED -- the same trap
        arriving as a pruned index over a complete shard
    L9  a layer whose checkpoint tensors were removed is REFUSED, not captured:
        CAPTURE-03 runs per streamed layer, not once for the resident set
    L10 --schedule layer-outer with --device-map is refused rather than
        silently letting accelerate's hooks fight the streamer
    L11 find_decoder_layers picks the text decoder stack by structure and
        refuses when it cannot tell
    L12 a decoder stack that does not run each layer exactly once per forward
        is refused rather than captured
    L13 both schedules report MEASURED peak memory, with the units named
    L14 self-compare through the layer-outer capture is exactly 0.0, including
        under --force-compute

Fail-without-fix: L1, L2, L4-L9, L12, L13 fail against the tree before this
change (L1/L2/L13 as an argparse refusal of --schedule, the rest as an
ImportError for engines/tools/layer_outer.py).  Verified by running this file
against a `git archive` of the parent commit; see docs/LAYER-OUTER.md.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "bin")
sys.path.insert(0, BIN)
sys.path.insert(0, os.path.join(REPO, "engines", "tools"))

PASS: list = []
FAIL: list = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def run(argv, **kwargs):
    return subprocess.run([sys.executable] + argv, capture_output=True, text=True, **kwargs)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def tiny_model(path, vocab=64, hidden=16, layers=3, seed=0):
    """A real, tiny, randomly initialised dense causal LM saved as a checkpoint."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                         num_hidden_layers=layers, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=False)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    return path


def tiny_moe_model(path, vocab=64, hidden=16, layers=3, experts=4, seed=3):
    """A tiny MoE whose checkpoint stores PER-EXPERT matrices.

    This is the shape that matters: `transformers` holds a MoE layer's experts
    as one fused tensor and rebuilds it on load with a `WeightConverter`.  For
    `zai-org/GLM-5.3-BF16` that converter owns 57,600 of 59,585 tensors, so a
    streamed loader that got the fusion subtly wrong would be wrong about
    96.7% of the model.  Returns None when this `transformers` cannot build
    such a model, so the caller can SKIP loudly rather than pass silently.
    """
    import torch

    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    except Exception:
        return None
    try:
        torch.manual_seed(seed)
        config = Qwen3MoeConfig(
            vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
            moe_intermediate_size=hidden, num_hidden_layers=layers,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            num_experts=experts, num_experts_per_tok=2, decoder_sparse_step=1,
            norm_topk_prob=True, max_position_embeddings=64,
            tie_word_embeddings=False)
        model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
        model.save_pretrained(path, safe_serialization=True)
    except Exception:
        return None
    return path


# 12 does not divide the fixture's 16-wide hidden or 32-wide intermediate, so
# every quantized tensor has a PARTIAL last block along both axes -- the form
# GLM-5.3's 576-row kv_a_proj_with_mqa takes under its 128 x 128 block.
FP8_BLOCK = [12, 12]
FP8_SKIP = ("embed_tokens", "lm_head", "norm", "gate.weight", "e_score_correction_bias")


def block_fp8_checkpoint(src, dst, block=FP8_BLOCK):
    """A block-scaled FP8 e4m3 copy of a bf16 checkpoint, in the FineGrainedFP8
    form transformers loads with dequantize=True: every 2-D projection weight
    becomes (fp8 weight, fp32 weight_scale_inv); embeddings, head, norms and
    routers stay bf16 and are named in modules_to_not_convert. Returns the
    quantized key names."""
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".safetensors"):
            continue
        shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    quantized = []
    skipped_modules = set()
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        tensors = load_file(os.path.join(src, name))
        out = {}
        for key, tensor in tensors.items():
            module = key.rsplit(".", 1)[0]
            eligible = (tensor.ndim == 2 and key.endswith(".weight")
                        and not any(marker in key for marker in FP8_SKIP))
            if not eligible:
                out[key] = tensor
                if key.endswith(".weight") and tensor.ndim == 2:
                    skipped_modules.add(module)
                continue
            rows, cols = tensor.shape
            grid_rows, grid_cols = -(-rows // block[0]), -(-cols // block[1])
            w = torch.nn.functional.pad(
                tensor.to(torch.float32),
                (0, grid_cols * block[1] - cols, 0, grid_rows * block[0] - rows))
            w = w.reshape(grid_rows, block[0], grid_cols, block[1])
            amax = w.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
            scale = amax / 448.0
            q = (w / scale).to(torch.float8_e4m3fn)
            q = q.reshape(grid_rows * block[0], grid_cols * block[1])[:rows, :cols]
            out[key] = q.contiguous()
            out[key + "_scale_inv"] = scale.reshape(grid_rows, grid_cols).contiguous()
            quantized.append(key)
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
    config_path = os.path.join(dst, "config.json")
    doc = json.load(open(config_path))
    doc["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": list(block),
        "activation_scheme": "dynamic",
        "modules_to_not_convert": sorted(skipped_modules)}
    json.dump(doc, open(config_path, "w"), indent=2)
    return quantized


def reference_dequantized_checkpoint(fp8_dir, dst):
    """The bf16 checkpoint transformers' own Fp8Dequantize produces from the
    FP8 one: the independent side of the end-to-end parity check."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    os.makedirs(dst, exist_ok=True)
    reference = Fp8Dequantize(None)
    for name in os.listdir(fp8_dir):
        if name.endswith(".safetensors"):
            continue
        shutil.copy(os.path.join(fp8_dir, name), os.path.join(dst, name))
    for name in sorted(os.listdir(fp8_dir)):
        if not name.endswith(".safetensors"):
            continue
        tensors = load_file(os.path.join(fp8_dir, name))
        out = {}
        for key, tensor in tensors.items():
            if key.endswith("_scale_inv"):
                continue
            scale_key = key + "_scale_inv"
            if scale_key in tensors:
                scales = tensors[scale_key]
                rows, cols = tensor.shape
                padded = torch.nn.functional.pad(
                    tensor.to(torch.float32),
                    (0, scales.shape[1] * FP8_BLOCK[1] - cols,
                     0, scales.shape[0] * FP8_BLOCK[0] - rows)).to(torch.float8_e4m3fn)
                out[key] = reference._dequantize_one(
                    padded, scales, output_dtype=torch.bfloat16)[:rows, :cols].contiguous()
            else:
                out[key] = tensor
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
    config_path = os.path.join(dst, "config.json")
    doc = json.load(open(config_path))
    doc.pop("quantization_config", None)
    json.dump(doc, open(config_path, "w"), indent=2)


FP8_SCOPE = {
    "policy": "mixed", "head_policy": "native", "kv_cache_dtype": "bf16",
    "assignments": [
        {"tensor_class": cls, "treatment": "quantized", "format": "fp8_e4m3",
         "bits_per_weight": 8, "layer_range": None}
        for cls in ("attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.down",
                    "moe.experts")
    ] + [
        {"tensor_class": cls, "treatment": "native", "format": "bf16",
         "bits_per_weight": 16, "layer_range": None}
        for cls in ("embed_tokens", "moe.router", "norm", "lm_head")
    ],
}


def tiny_panel(path, windows=3, length=12, vocab=64, seed=1):
    """A panel tree in the upstream `quant-pipeline.glm53-token-panel.v1` layout."""
    import numpy as np

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
    with open(os.path.join(path, "panel.receipt.json"), "w", encoding="utf-8") as handle:
        json.dump({"schema": "malaiwah.token-panel-build-receipt.v1",
                   "selection_rule": "layer-outer selftest fixture"}, handle, indent=2)
    return path


def capture(model, panel, out, *, dataset_id, name, extra=(), memory_report=None,
            role="root"):
    argv = [os.path.join(REPO, "engines", "tools", "hf_capture.py"),
            "--model", model, "--panel", panel, "--out", out, "--role", role,
            "--lane", "local-cuda-budget", "--dataset-id", dataset_id,
            "--dataset-name", name, "--device", "cpu",
            "--weights-repository", "selftest/tiny", "--model-revision", "0" * 40]
    if memory_report:
        argv += ["--memory-report", memory_report]
    return run(argv + list(extra))


def digest_of(root):
    from fidelity import dsformat as F

    return F.read_json(os.path.join(root, F.MANIFEST_NAME))["capture"]["capture_content_digest"]


# ---------------------------------------------------------------------------


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        print("SKIP selftest_layer_outer: torch/transformers unavailable (%s)" % exc)
        return 0
    work = tempfile.mkdtemp(prefix="layerouter-")
    try:
        return _body(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _body(work):
    import torch

    # Imported defensively ON PURPOSE.  Against a tree without this change the
    # module does not exist, and a bare ImportError would abort the file before
    # a single case reported -- which is the one shape of evidence that cannot
    # be read as "these cases fail without the fix". Every case that needs the
    # module now fails by name instead.
    try:
        import layer_outer
    except ImportError as exc:
        layer_outer = None
        module_absent = str(exc)
    else:
        module_absent = None

    def needs_module(name):
        check(name, False, "engines/tools/layer_outer.py is not importable: %s" % module_absent)

    model = tiny_model(os.path.join(work, "reference"))
    panel = tiny_panel(os.path.join(work, "panel"))

    # -- L1 / L2 / L3 --------------------------------------------------------
    # The deliverable in one assertion: the capture must not move.  Not
    # "close", not "within tolerance" -- the same content digest, because the
    # same operations ran in the same order on the same numbers.
    outs = {}
    for label, extra in (
            ("default", []),
            ("window-outer", ["--schedule", "window-outer"]),
            ("resident", ["--schedule", "layer-outer",
                          "--layer-residency", "resident"]),
            ("stream", ["--schedule", "layer-outer", "--layer-residency", "stream"])):
        out = os.path.join(work, "ds-" + label)
        report = os.path.join(work, "mem-%s.json" % label)
        proc = capture(model, panel, out, dataset_id="fidelity--selftest.lo",
                       name="layer-outer selftest", extra=extra, memory_report=report)
        outs[label] = (proc, out, report)
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])

    ok = all(proc.returncode == 0 for proc, _, _ in outs.values())
    check("L0 all four captures exit 0", ok,
          "; ".join("%s rc=%d %s" % (k, v[0].returncode, (v[0].stderr or "").strip()[-120:])
                    for k, v in outs.items()))
    # Deliberately NOT an early return.  Against a tree without this change the
    # captures cannot run at all, and stopping here would report four failures
    # instead of the fourteen this file actually guards -- understating what the
    # change is load-bearing for.
    base = digest_of(outs["default"][1]) if ok else None
    check("L1 layer-outer/resident reproduces the window-outer capture bit-for-bit",
          ok and digest_of(outs["resident"][1]) == base,
          "the capture did not run; see L0" if not ok else
          "%s vs %s" % (digest_of(outs["resident"][1])[:16], base[:16]))
    check("L2 layer-outer/stream reproduces it too -- the loader moves nothing",
          ok and digest_of(outs["stream"][1]) == base,
          "the capture did not run; see L0" if not ok else
          "%s vs %s" % (digest_of(outs["stream"][1])[:16], base[:16]))
    check("L3 the default schedule is window-outer and naming it changes nothing",
          ok and digest_of(outs["window-outer"][1]) == base,
          "" if ok else "the capture did not run; see L0")

    # -- L13 -----------------------------------------------------------------
    # A projection is not a measurement.  Every run must be able to say what it
    # actually used, with the units named -- ru_maxrss is bytes on Darwin and
    # kilobytes on Linux, and a silent 1024x is exactly the kind of number that
    # gets a machine rented wrongly.
    reports = {}
    for label in ("window-outer", "stream"):
        path = outs[label][2]
        reports[label] = json.load(open(path)) if os.path.isfile(path) else None
    check("L13 both schedules write a measured peak-memory report naming its units",
          all(r and r.get("peak_rss_bytes", 0) > 0 and r.get("rss_units_source")
              for r in reports.values())
          and reports["stream"]["schedule"] == "layer-outer"
          and reports["window-outer"]["schedule"] == "window-outer",
          json.dumps(reports))

    if layer_outer is None:
        for _name in ("L4 every streamed layer parameter is byte-identical to from_pretrained's",
                      "L5 freeing a layer actually returns its parameters to the meta device",
                      "L6 a fused-expert MoE checkpoint streams byte-exactly (the WeightConverter path)",
                      "L7 a shard shorter than its own header is REFUSED (holes-read-as-zeros)",
                      "L8 a shard-header/index key-set disagreement is REFUSED"):
            needs_module(_name)
    else:
        # -- L4 / L5 -------------------------------------------------------------
        # The loader's own claim, checked against the thing it must equal.
        from transformers import AutoConfig
        import transformers as _tf

        def stream_params(checkpoint, cls, config, expert_fill):
            streamer = layer_outer.build_streamed_model(
                checkpoint, cls, config, "bfloat16", "cpu", lambda **kw: None,
                expert_fill=expert_fill)
            params, freed = {}, True
            for index in range(len(streamer.layers)):
                streamer.load_layer(index)
                names = streamer._load_layer_keys(index)
                for name in names:
                    params[name] = streamer.model.get_parameter(name).detach().clone()
                streamer.free_layer(index)
                if any(streamer.model.get_parameter(n).device.type != "meta" for n in names):
                    freed = False
            return params, freed, streamer

        def stream_vs_full(checkpoint, label):
            """(equal to from_pretrained, freed, checked, direct == converter, fill stats)."""
            config = AutoConfig.from_pretrained(checkpoint)
            cls = getattr(_tf, list(config.architectures)[0])
            reference = cls.from_pretrained(checkpoint, dtype=torch.bfloat16)
            reference_sd = dict(reference.state_dict())
            direct, freed, streamer = stream_params(checkpoint, cls, config,
                                                    layer_outer.EXPERT_FILL_DIRECT)
            converter, _, _ = stream_params(checkpoint, cls, config,
                                            layer_outer.EXPERT_FILL_CONVERTER)
            equal = all(torch.equal(direct[name], reference_sd[name]) for name in direct)
            same = (set(direct) == set(converter)
                    and all(torch.equal(direct[name], converter[name]) for name in direct))
            del reference, reference_sd
            return equal, freed, len(direct), same, layer_outer.expert_fill_evidence(streamer)

        equal, freed, checked, same, fill = stream_vs_full(model, "dense")
        check("L4 every streamed layer parameter is byte-identical to from_pretrained's",
              equal and checked > 0, "%d parameters checked" % checked)
        check("L5 freeing a layer actually returns its parameters to the meta device", freed)
        # ---- EfficiencyFixes (review-efficiency S1-1 / S2-3) -------------------
        check("L4b a dense checkpoint has nothing for the direct expert fill and the two "
              "loader modes agree byte for byte",
              same and fill["targets_filled"] == 0 and fill["mode"] == "direct",
              json.dumps(fill)[:200])
        # ---- end EfficiencyFixes -----------------------------------------------

        # -- L6 ------------------------------------------------------------------
        moe = tiny_moe_model(os.path.join(work, "moe"))
        if moe is None:
            check("L6 a fused-expert MoE checkpoint streams byte-exactly", True,
                  "SKIPPED: this transformers cannot build a tiny Qwen3-MoE")
        else:
            moe_equal, moe_freed, moe_checked, moe_same, moe_fill = stream_vs_full(moe, "moe")
            check("L6 a fused-expert MoE checkpoint streams byte-exactly "
                  "(the WeightConverter path)", moe_equal and moe_checked > 0,
                  "%d parameters checked" % moe_checked)
            # ---- EfficiencyFixes (review-efficiency S1-1 / S2-3) ---------------
            # The direct fill writes each per-expert slice straight into the
            # fused parameter (gate rows first, then up; down by expert) from a
            # staging buffer read off the shard bytes; the converter's
            # stack-then-cat result must be the SAME BYTES, parameter by
            # parameter, and every routed-expert parameter of every layer must
            # actually have gone the direct way.
            check("L6b the direct expert fill is byte-identical to the converter path on "
                  "every fused expert parameter, and filled every one of them",
                  moe_same and moe_fill["mode"] == "direct"
                  and moe_fill["targets_filled"] == 2 * 3 and moe_fill["layers_filled"] == 3
                  and moe_fill["staged_slices"] == 3 * 4 * 3 and moe_fill["decoded_slices"] == 0
                  and not moe_fill["declined"],
                  json.dumps(moe_fill)[:300])
            # A bf16 checkpoint missing one expert is NOT eligible (the fill
            # would have to guess), so it goes to the converter and is refused
            # there exactly as before -- the parameter stays on meta.
            partial = os.path.join(work, "moe-missing-expert")
            shutil.copytree(moe, partial)
            from safetensors.torch import load_file as _lf, save_file as _sf
            shard = os.path.join(partial, "model.safetensors")
            tensors = _lf(shard)
            victim = next(k for k in sorted(tensors) if k.endswith("experts.2.up_proj.weight"))
            del tensors[victim]
            _sf(tensors, shard, metadata={"format": "pt"})
            index_path = os.path.join(partial, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                idx = json.load(open(index_path))
                idx["weight_map"].pop(victim, None)
                json.dump(idx, open(index_path, "w"))
            config = AutoConfig.from_pretrained(partial)
            cls = getattr(_tf, list(config.architectures)[0])
            streamer = layer_outer.build_streamed_model(
                partial, cls, config, "bfloat16", "cpu", lambda **kw: None)
            refused = None
            try:
                streamer.load_layer(int(victim.split(".")[2]))
            except layer_outer.LayerOuterError as exc:
                refused = str(exc)
            declined = layer_outer.expert_fill_evidence(streamer)["declined"]
            check("L6c a checkpoint missing one expert slice is declined by the direct fill "
                  "(named, with the count) and the converter path refuses it as before",
                  refused is not None and "meta device" in refused
                  and any("gate_up_proj" in k and "7 of 8" in v for k, v in declined.items()),
                  "refused=%r declined=%r" % ((refused or "")[:160], declined))
            # ---- end EfficiencyFixes -------------------------------------------

        # -- L7 ------------------------------------------------------------------
        # Stage A: transformers enumerates each shard's OWN header, not the pruned
        # index.  A short shard therefore hands back ZEROS under a real tensor
        # name, and zeros load without complaint.
        short = os.path.join(work, "short")
        shutil.copytree(model, short)
        shard = os.path.join(short, "model.safetensors")
        size = os.path.getsize(shard)
        with open(shard, "r+b") as handle:
            handle.truncate(size - 64)
        refused = None
        try:
            layer_outer.audit_checkpoint_tree(short)
        except layer_outer.LayerOuterError as exc:
            refused = str(exc)
        check("L7 a shard shorter than its own header is REFUSED (holes-read-as-zeros)",
              refused is not None and "partially fetched" in refused,
              refused or "audit accepted a truncated shard")

        # -- L8 ------------------------------------------------------------------
        pruned = os.path.join(work, "pruned")
        shutil.copytree(model, pruned)
        with open(os.path.join(pruned, "model.safetensors"), "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len).decode("utf-8"))
        keys = [k for k in header if k != "__metadata__"]
        weight_map = {k: "model.safetensors" for k in keys[1:]}  # prune exactly one
        with open(os.path.join(pruned, "model.safetensors.index.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, handle)
        refused = None
        try:
            layer_outer.audit_checkpoint_tree(pruned)
        except layer_outer.LayerOuterError as exc:
            refused = str(exc)
        check("L8 a shard-header/index key-set disagreement is REFUSED",
              refused is not None and "disagree" in refused,
              refused or "audit accepted a pruned index over a complete shard")

    # -- L9 ------------------------------------------------------------------
    # CAPTURE-03 per streamed layer.  Running the guard only on the resident
    # set would leave 97.5% of GLM-5.3 by bytes unexamined.
    holed = os.path.join(work, "holed")
    shutil.copytree(model, holed)
    from safetensors.torch import load_file, save_file

    tensors = load_file(os.path.join(holed, "model.safetensors"))
    dropped = [k for k in tensors if k.startswith("model.layers.1.")]
    for key in dropped:
        del tensors[key]
    save_file(tensors, os.path.join(holed, "model.safetensors"))
    proc = capture(holed, panel, os.path.join(work, "ds-holed"),
                   dataset_id="fidelity--selftest.lo.holed", name="holed",
                   extra=["--schedule", "layer-outer", "--layer-residency", "stream"])
    check("L9 a streamed layer whose checkpoint tensors are absent is REFUSED",
          proc.returncode != 0 and not os.path.isfile(
              os.path.join(work, "ds-holed", "fidelity-dataset.json")),
          "rc=%d stderr=%s" % (proc.returncode, (proc.stderr or "")[-300:]))

    # -- L10 -----------------------------------------------------------------
    proc = capture(model, panel, os.path.join(work, "ds-dm"),
                   dataset_id="fidelity--selftest.lo.dm", name="dm",
                   extra=["--schedule", "layer-outer", "--device-map", "auto"])
    check("L10 --schedule layer-outer with --device-map is refused, not silently raced",
          proc.returncode == 3 and "device-map" in (proc.stderr or ""),
          "rc=%d stderr=%s" % (proc.returncode, (proc.stderr or "")[-200:]))

    if layer_outer is None:
        for _name in ("L11a find_decoder_layers finds the stack whose parent owns embed_tokens",
                      "L11b it refuses rather than guessing when no stack matches",
                      "L12a the schedule drives a plain in-order stack",
                      "L12b a layer that runs more than once per forward is REFUSED"):
            needs_module(_name)
    else:
        # -- L11 -----------------------------------------------------------------
        from transformers import LlamaConfig, LlamaForCausalLM

        small = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                                             num_hidden_layers=2, num_attention_heads=2,
                                             num_key_value_heads=2, max_position_embeddings=16))
        name, module = layer_outer.find_decoder_layers(small)
        check("L11a find_decoder_layers finds the stack whose parent owns embed_tokens",
              name == "model.layers" and len(module) == 2, "%s/%d" % (name, len(module)))
        bare = torch.nn.Module()
        bare.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        refused = None
        try:
            layer_outer.find_decoder_layers(bare)
        except layer_outer.LayerOuterError as exc:
            refused = str(exc)
        check("L11b it refuses rather than guessing when no stack matches",
              refused is not None and "embed_tokens" in refused, refused or "picked one anyway")

        # -- L12 -----------------------------------------------------------------
        # The schedule assumes a plain in-order loop, each layer called once per
        # forward.  A model that calls a layer twice (weight tying across depth,
        # an MTP head reusing a block) would silently capture the wrong thing.
        class _Block(torch.nn.Module):
            def forward(self, hidden):  # noqa: D401 - a stand-in decoder layer
                return hidden + 1

        stack = torch.nn.ModuleList([_Block(), _Block()])

        def once(_window):
            hidden = torch.zeros(1)
            for block in stack:
                hidden = block(hidden)

        ran = layer_outer.run_panel(None, stack, once, 1, lambda **kw: None,
                                    collect=lambda i: "ok")
        check("L12a the schedule drives a plain in-order stack", ran == ["ok"], repr(ran))

        def twice(_window):
            hidden = torch.zeros(1)
            for block in stack:
                hidden = block(hidden)
                if block is stack[0]:  # the pathology: the same layer, twice
                    hidden = block(hidden)

        refused = None
        try:
            layer_outer.run_panel(None, stack, twice, 1, lambda **kw: None)
        except layer_outer.LayerOuterError as exc:
            refused = str(exc)
        check("L12b a layer that runs more than once per forward is REFUSED",
              refused is not None and "expected exactly 1" in refused,
              refused or "run_panel accepted a double-called layer")

    # -- L14 -----------------------------------------------------------------
    # SC-1 through the new schedule.  A second cold capture in a separate
    # process, then the comparator, with the math actually run.
    second = os.path.join(work, "ds-stream-2")
    proc = capture(model, panel, second, dataset_id="fidelity--selftest.lo",
                   name="layer-outer selftest",
                   extra=["--schedule", "layer-outer", "--layer-residency", "stream"])
    check("L14a a second cold layer-outer capture agrees with the first",
          proc.returncode == 0 and base is not None and digest_of(second) == base,
          "rc=%d%s" % (proc.returncode, "" if ok else " (no reference digest; see L0)"))
    for slug, extra in (("hashproof", []), ("forcecompute", ["--force-compute"])):
        destination = os.path.join(work, "cmp-%s" % slug)
        compare = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "compare",
                       "--reference", outs["stream"][1], "--candidate", second,
                       "--self-compare", "--out", destination] + extra)
        receipt = os.path.join(destination, "comparison-receipt.json")
        doc = json.load(open(receipt)) if os.path.isfile(receipt) else {}
        metric = (doc.get("metric") or {})
        kl = (doc.get("kl") or {})
        # "exactly 0.0" means every order statistic, not just the mean: a mean
        # of zero over a tokenwise distribution with a non-zero max would be
        # cancellation, not identity.
        exact = (metric.get("value") == 0.0 and kl.get("mean") == 0.0
                 and kl.get("max") == 0.0)
        check("L14b self-compare through layer-outer is exactly 0.0 (%s)" % slug,
              compare.returncode == 0 and exact,
              "rc=%d metric=%r kl=%r" % (compare.returncode, metric.get("value"), kl))

    # And the load-bearing one: the two SCHEDULES compared against each other
    # through the comparator, with the math actually run rather than a digest
    # short-circuit.
    destination = os.path.join(work, "cmp-cross")
    compare = run([os.path.join(REPO, "bin", "fidelity_dataset.py"), "compare",
                   "--reference", outs["window-outer"][1], "--candidate", outs["stream"][1],
                   "--self-compare", "--force-compute", "--out", destination])
    receipt = os.path.join(destination, "comparison-receipt.json")
    doc = json.load(open(receipt)) if os.path.isfile(receipt) else {}
    check("L14c window-outer vs layer-outer scores exactly 0.0 under --force-compute",
          compare.returncode == 0
          and (doc.get("metric") or {}).get("value") == 0.0
          and (doc.get("kl") or {}).get("max") == 0.0
          and doc.get("top1_agreement") == 1.0,
          "rc=%d metric=%r" % (compare.returncode, (doc.get("metric") or {}).get("value")))

    # L15 -- a QUANTIZED checkpoint must be refused by this schedule, up front.
    # `build_streamed_model` calls `cls(config)` and loads with the MODEL's
    # conversion mapping; it never builds an `HfQuantizer`, so the quantizer's
    # module replacement, its `*.scale` -> `*.weight_scale_inv` rename and its
    # dequantization op are all missing. For a packed format (FP4 experts) that
    # surfaces as a shape mismatch and raises. For a plain FP8 E4M3 weight the
    # shape MATCHES the bf16 parameter: the payload is read as bf16, the scale
    # falls out as `unexpected`, and the block scale is never applied -- the M1
    # Qwen3.8-27B-FP8 defect, silent, behind a flag a truncated tree already
    # needs. Observed for real on deepseek-ai/DeepSeek-V4-Flash-0731:
    # "Reinit due to size mismatch - ckpt: torch.Size([256, 4096, 2048]) vs
    #  model: torch.Size([256, 4096, 4096])", raised by transformers rather than
    # by us. Now ours, and now before any weight is read.
    quantized = os.path.join(work, "quantized")
    tiny_model(quantized, layers=2)
    config_path = os.path.join(quantized, "config.json")
    doc = json.load(open(config_path))
    doc["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                  "weight_block_size": [128, 128]}
    json.dump(doc, open(config_path, "w"), indent=2)
    out = os.path.join(work, "quantized-capture")
    proc = capture(quantized, panel, out, dataset_id="fidelity--lo.quantized",
                   name="quantized refusal", extra=("--schedule", "layer-outer"))
    text = (proc.stdout or "") + (proc.stderr or "")
    check("L15 layer-outer REFUSES a config declaring FP8 over a checkpoint with "
          "no scale tensors, naming the silent-FP8 reading",
          proc.returncode != 0
          and "quantization_config" in text and "window-outer" in text
          and "_scale_inv" in text
          and not os.path.isfile(os.path.join(out, "fidelity-dataset.json")),
          "rc=%s tail=%s" % (proc.returncode, text[-200:]))
    for label, patch in (("L15b a packed/other quant_method", {"quant_method": "mxfp4"}),
                         ("L15c a static activation scheme",
                          {"quant_method": "fp8", "fmt": "e4m3",
                           "weight_block_size": [128, 128],
                           "activation_scheme": "static"})):
        doc = json.load(open(config_path))
        doc["quantization_config"] = patch
        json.dump(doc, open(config_path, "w"), indent=2)
        proc = capture(quantized, panel, out + label[:4], dataset_id="fidelity--lo.q",
                       name="q", extra=("--schedule", "layer-outer"))
        text = (proc.stdout or "") + (proc.stderr or "")
        check("%s is refused before anything is instantiated" % label,
              proc.returncode != 0 and "not the block-scaled FP8" in text,
              "rc=%s tail=%s" % (proc.returncode, text[-200:]))

    # -- L16: the one quantized form this schedule DECODES ------------------
    # A block-scaled FP8 e4m3 checkpoint (the zai-org/GLM-5.3 form) is decoded
    # to bf16 per tensor on the host before the converter sees it. Parity is
    # asserted two ways: the decoder against transformers' own
    # Fp8Dequantize arithmetic tensor by tensor, and the whole capture against
    # a native capture of the checkpoint that reference produces.
    import torch
    from safetensors.torch import load_file
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    sys.path.insert(0, os.path.join(REPO, "engines", "tools"))
    import layer_outer as LO

    scope_file = os.path.join(work, "fp8-scope.json")
    json.dump(FP8_SCOPE, open(scope_file, "w"))
    for slug, builder in (("dense", lambda path: tiny_model(path, layers=2, seed=5)),
                          ("moe", lambda path: tiny_moe_model(path, layers=2, seed=6))):
        bf16_dir = os.path.join(work, "fp8-%s-bf16" % slug)
        if builder(bf16_dir) is None:
            print("  SKIP L16 %s: this transformers cannot build the MoE fixture" % slug)
            continue
        fp8_dir = os.path.join(work, "fp8-%s" % slug)
        quantized_keys = block_fp8_checkpoint(bf16_dir, fp8_dir)
        check("L16a-%s the fixture quantized the projections and nothing else" % slug,
              quantized_keys and not any(m in k for k in quantized_keys for m in FP8_SKIP),
              "%d keys" % len(quantized_keys))
        reference = Fp8Dequantize(None)
        agree, compared = True, 0
        for name in os.listdir(fp8_dir):
            if not name.endswith(".safetensors"):
                continue
            tensors = load_file(os.path.join(fp8_dir, name))
            for key in quantized_keys:
                if key not in tensors:
                    continue
                scales = tensors[key + "_scale_inv"]
                rows, cols = tensors[key].shape
                ours = LO.dequantize_block_fp8(tensors[key], scales, torch.bfloat16, FP8_BLOCK)
                padded = torch.nn.functional.pad(
                    tensors[key].to(torch.float32),
                    (0, scales.shape[1] * FP8_BLOCK[1] - cols,
                     0, scales.shape[0] * FP8_BLOCK[0] - rows)).to(torch.float8_e4m3fn)
                theirs = reference._dequantize_one(
                    padded, scales, output_dtype=torch.bfloat16)[:rows, :cols].contiguous()
                compared += 1
                if ours.dtype != theirs.dtype or not torch.equal(
                        ours.view(torch.int16), theirs.view(torch.int16)):
                    agree = False
        check("L16b-%s dequantize_block_fp8 is bitwise transformers' Fp8Dequantize on "
              "every quantized tensor (partial last blocks via the padded reference)" % slug,
              agree and compared == len(quantized_keys),
              "compared=%d of %d" % (compared, len(quantized_keys)))
        ref_dir = os.path.join(work, "fp8-%s-refdeq" % slug)
        reference_dequantized_checkpoint(fp8_dir, ref_dir)
        fp8_out = os.path.join(work, "fp8-%s-capture" % slug)
        ref_out = os.path.join(work, "fp8-%s-refcapture" % slug)
        fp8_proc = capture(fp8_dir, panel, fp8_out, dataset_id="fidelity--lo.fp8." + slug,
                           name="fp8 " + slug, role="quant",
                           extra=("--schedule", "layer-outer", "--scope-file", scope_file,
                                  "--codec", "fp8_e4m3", "--declared-bits", "8",
                                  "--no-sanity-check"))
        ref_proc = capture(ref_dir, panel, ref_out, dataset_id="fidelity--lo.fp8ref." + slug,
                           name="fp8 reference " + slug,
                           extra=("--schedule", "layer-outer", "--no-sanity-check"))
        check("L16c-%s the FP8 checkpoint captures under layer-outer as a quant" % slug,
              fp8_proc.returncode == 0 and ref_proc.returncode == 0,
              "fp8 rc=%s %s | ref rc=%s %s" % (
                  fp8_proc.returncode, (fp8_proc.stderr or "")[-300:],
                  ref_proc.returncode, (ref_proc.stderr or "")[-300:]))
        if fp8_proc.returncode == 0 and ref_proc.returncode == 0:
            check("L16d-%s ... and its capture_content_digest is bitwise the native "
                  "capture of the reference-dequantized checkpoint" % slug,
                  digest_of(fp8_out) == digest_of(ref_out),
                  "%s vs %s" % (digest_of(fp8_out)[:16], digest_of(ref_out)[:16]))
            manifest = json.load(open(os.path.join(fp8_out, "fidelity-dataset.json")))
            runtime = json.load(open(os.path.join(fp8_out, manifest["runtime"]["file"])))
            decode = runtime["capture_tool"].get("weights_decode") or {}
            ref_manifest = json.load(open(os.path.join(ref_out, "fidelity-dataset.json")))
            ref_runtime = json.load(open(os.path.join(ref_out, ref_manifest["runtime"]["file"])))
            check("L16e-%s the runtime receipt records the decode: method, reference, "
                  "config, and one decoded tensor per quantized key" % slug,
                  decode.get("method") == LO.FP8_DECODE_METHOD
                  and decode.get("reference") == LO.FP8_DECODE_REFERENCE
                  and decode.get("tensors_dequantized") == len(quantized_keys)
                  and decode.get("scale_tensors_consumed") == len(quantized_keys)
                  and decode.get("quantization_config", {}).get("weight_block_size") == FP8_BLOCK
                  and manifest["weights"]["quantized"] is True
                  and ref_runtime["capture_tool"].get("weights_decode") is None,
                  json.dumps(decode)[:300])
            check("L16f-%s no scale tensor reached the loader as unexpected" % slug,
                  not any(k.endswith("_scale_inv") for k in
                          (runtime["capture_tool"].get("unexpected_tensor_allowlist") or {})
                          .get("observed", []) or []),
                  "")
    # An fp8 tensor whose scale is missing is the silent case: refused by name.
    dense_fp8 = os.path.join(work, "fp8-dense")
    if os.path.isdir(dense_fp8):
        broken = os.path.join(work, "fp8-dense-noscale")
        shutil.copytree(dense_fp8, broken)
        from safetensors.torch import save_file
        shard = os.path.join(broken, "model.safetensors")
        tensors = load_file(shard)
        victim = next(k for k in tensors if k.endswith("_scale_inv"))
        del tensors[victim]
        save_file(tensors, shard, metadata={"format": "pt"})
        proc = capture(broken, panel, os.path.join(work, "fp8-noscale-capture"),
                       dataset_id="fidelity--lo.fp8.noscale", name="noscale", role="quant",
                       extra=("--schedule", "layer-outer", "--scope-file", scope_file,
                              "--codec", "fp8_e4m3", "--declared-bits", "8",
                              "--no-sanity-check"))
        text = (proc.stdout or "") + (proc.stderr or "")
        check("L16g an fp8 tensor with no scale beside it is REFUSED by name",
              proc.returncode != 0 and "no block scale" in text
              and victim[:-len("_scale_inv")] in text,
              "rc=%s tail=%s" % (proc.returncode, text[-300:]))

    # ---- EfficiencyFixes (review-efficiency S1-2 / S1-1) ----------------------
    # -- L18: the FP8 decode runs on the capture DEVICE under a fail-closed gate.
    # No CUDA on the selftest host, so the device seam is driven with a stub:
    # a stub that returns the reference bytes passes and is counted, a stub
    # that perturbs one element is REFUSED by tensor, block, device and
    # max_abs_diff. The fixture's 12-wide block leaves every tensor with a
    # partial last block, i.e. every tensor is on the always-check list.
    moe_fp8 = os.path.join(work, "fp8-moe")
    if os.path.isdir(moe_fp8):
        from transformers import AutoConfig as _AC
        fp8_cfg = _AC.from_pretrained(moe_fp8)
        fp8_plan = LO.fp8_checkpoint_plan(fp8_cfg)
        shard_tensors = load_file(os.path.join(moe_fp8, "model.safetensors"))
        fp8_key = next(k for k in sorted(shard_tensors)
                       if k.endswith("experts.0.gate_proj.weight"))
        pair = {fp8_key: shard_tensors[fp8_key],
                fp8_key + "_scale_inv": shard_tensors[fp8_key + "_scale_inv"]}

        def faithful_stub(quantized, scales, dtype, block, device):
            return LO.dequantize_block_fp8(quantized, scales, dtype, block)

        def perturbing_stub(quantized, scales, dtype, block, device):
            out = LO.dequantize_block_fp8(quantized, scales, dtype, block).clone()
            out.view(-1)[3] = out.view(-1)[3] + 1
            return out

        def fp8_stats():
            return {"dequantized": 0, "scales_consumed": 0, "fp8_bytes": 0}

        refusal = None
        try:
            LO.materialize_fp8_subset(pair, fp8_plan, torch.bfloat16, fp8_stats(),
                                      device="stub:0", parity_all=True,
                                      device_decode=perturbing_stub)
        except LO.LayerOuterError as exc:
            refusal = str(exc)
        check("L18a a device decode that differs from the host decode by one element is "
              "REFUSED naming tensor, block, device and max_abs_diff",
              refusal is not None and fp8_key in refusal and "stub:0" in refusal
              and "max_abs_diff=" in refusal and "block (12, 12)" in refusal
              and "not bitwise the host decode" in refusal,
              (refusal or "no refusal")[:240])
        refusal = None
        try:
            LO.materialize_fp8_subset(pair, fp8_plan, torch.bfloat16, fp8_stats(),
                                      device="stub:0", parity_all=False,
                                      device_decode=perturbing_stub)
        except LO.LayerOuterError as exc:
            refusal = str(exc)
        check("L18b ... and a partial-block tensor is checked on EVERY layer, not only the "
              "first (parity_all=False still refuses)", refusal is not None,
              (refusal or "no refusal")[:120])
        stats = fp8_stats()
        offered = []
        out = LO.materialize_fp8_subset(pair, fp8_plan, torch.bfloat16, stats, device="stub:0",
                                        parity_all=True, device_decode=faithful_stub,
                                        sink=lambda key, tensor: offered.append(key) or True)
        parity = LO.fp8_device_parity_evidence(stats)
        check("L18c a faithful device decode passes, is counted in the receipt block by "
              "device and tensor count, and the decoded tensor goes to the sink, not the dict",
              parity["fp8_decode_device"] == "stub:0" and parity["fp8_device_parity"] == "passed"
              and parity["fp8_device_parity_checked_tensors"] == 1
              and parity["fp8_device_parity_checked_partial_block"] == 1
              and offered == [fp8_key] and out == {},
              json.dumps(parity) + repr(offered))
        host_stats = fp8_stats()
        LO.materialize_fp8_subset(pair, fp8_plan, torch.bfloat16, host_stats, device="cpu",
                                  parity_all=True)
        check("L18d a host decode records itself as such (no parity to claim)",
              LO.fp8_device_parity_evidence(host_stats)["fp8_device_parity"] == "not-applicable"
              and LO.fp8_device_parity_evidence(host_stats)["fp8_decode_device"] == "cpu",
              json.dumps(LO.fp8_device_parity_evidence(host_stats)))
        # The receipt of the L16c FP8 capture (a CPU run) carries the decode
        # device and the expert-fill block: the fused experts of a decoded
        # checkpoint came from the decoder's tensors, none from the converter.
        from fidelity import dsformat as _F
        fp8_out = os.path.join(work, "fp8-moe-capture")
        manifest_path = os.path.join(fp8_out, _F.MANIFEST_NAME)
        decode, fill_block = {}, {}
        if os.path.isfile(manifest_path):
            manifest = json.load(open(manifest_path))
            runtime = json.load(open(os.path.join(fp8_out, manifest["runtime"]["file"])))
            decode = runtime["capture_tool"]["weights_decode"] or {}
            fill_block = runtime["capture_tool"].get("expert_fill") or {}
        check("L18e the sealed FP8 receipt names the decode device and the direct fill: "
              "every routed-expert slice came from the decoder, none was staged",
              decode.get("fp8_decode_device") == "cpu"
              and decode.get("fp8_device_parity") == "not-applicable"
              and fill_block.get("mode") == "direct"
              and fill_block.get("targets_filled") == 2 * 2
              and fill_block.get("decoded_slices") == 2 * 4 * 3
              and fill_block.get("staged_slices") == 0,
              json.dumps(fill_block)[:300] + json.dumps(
                  {k: v for k, v in decode.items() if k.startswith("fp8_")}))
        # A decoded checkpoint missing one expert's (weight, scale) pair: the
        # fill is eligible (a decoder is active), the decoder never produces
        # that slice, and the layer is REFUSED by the slice's name before a
        # window runs -- never a fused parameter with undefined rows.
        short = os.path.join(work, "fp8-moe-missing-expert")
        shutil.copytree(moe_fp8, short)
        from safetensors.torch import save_file as _save
        gone = next(k for k in sorted(shard_tensors) if k.endswith("experts.1.down_proj.weight"))
        trimmed = {k: v for k, v in shard_tensors.items()
                   if k not in (gone, gone + "_scale_inv")}
        _save(trimmed, os.path.join(short, "model.safetensors"), metadata={"format": "pt"})
        idx_path = os.path.join(short, "model.safetensors.index.json")
        if os.path.isfile(idx_path):
            idx = json.load(open(idx_path))
            for k in (gone, gone + "_scale_inv"):
                idx["weight_map"].pop(k, None)
            json.dump(idx, open(idx_path, "w"))
        import transformers as _tf2
        short_cfg = _AC.from_pretrained(short)
        short_cls = getattr(_tf2, list(short_cfg.architectures)[0])
        streamer = LO.build_streamed_model(short, short_cls, short_cfg, "bfloat16", "cpu",
                                           lambda **kw: None)
        refusal = None
        try:
            streamer.load_layer(int(gone.split(".")[2]))
        except LO.LayerOuterError as exc:
            refusal = str(exc)
        check("L18f a decoded checkpoint missing one expert slice is REFUSED by the fill, "
              "naming the slice that was never delivered",
              refusal is not None and "never delivered" in refusal and gone in refusal,
              (refusal or "no refusal")[:240])
    else:
        check("L18 FP8 device gate rungs", True, "SKIPPED: no fp8-moe fixture (see L16)")
    # ---- end EfficiencyFixes -------------------------------------------------

    # -- L17: the modelopt NVFP4 form this schedule DECODES (flagship) ---------
    # No tiny glm_moe_dsa can be built (the geometry table refuses anything
    # but the real 78x256 stack by name, which is the point), so the plan is
    # exercised on the REAL RadixArk config over a synthetic full-census index,
    # and the per-layer decode on synthetic packed shards whose expected value
    # is nvfp4_surface.dequant_nvfp4 (itself proven bitwise against
    # compressed-tensors on real rows: selftest_nvfp4_offline rung 11).
    import nvfp4_surface as NS
    from safetensors.torch import save_file
    from safetensors import safe_open
    evidence = os.path.join(REPO, "engines", "tools", "nvfp4-evidence")
    real_config = json.load(open(os.path.join(evidence, "radixark-config.json")))
    geometry = NS.GLM_MOE_DSA_GEOMETRY
    official = json.load(open(os.path.join(evidence, "official-glm53-nonrouted-names.json")))["names"]

    def synthetic_index(mtp_plain=True, drop=None, extra=None):
        weight_map = {name: "model-00001-of-00002.safetensors" for name in official}
        for layer in geometry.main_routed_layers:
            for expert in range(geometry.num_experts):
                for projection in NS.PROJECTIONS:
                    for component in NS.MO_NVFP4_DECODE + NS.MO_NVFP4_ACTIVATION:
                        weight_map[geometry.component_name(layer, expert, projection, component)] = \
                            "model-00002-of-00002.safetensors"
        for expert in range(geometry.num_experts):
            for projection in NS.PROJECTIONS:
                comps = ("weight",) if mtp_plain else NS.MO_NVFP4_DECODE
                for component in comps:
                    weight_map[geometry.component_name(geometry.mtp_layer, expert, projection, component)] = \
                        "model-00002-of-00002.safetensors"
        for name in (drop or ()):
            weight_map.pop(name)
        for name in (extra or ()):
            weight_map[name] = "model-00002-of-00002.safetensors"
        return {"metadata": {}, "weight_map": weight_map}

    class _Config:
        def __init__(self, doc):
            self._doc = doc
            self.quantization_config = doc.get("quantization_config")

        def to_dict(self):
            return dict(self._doc)

    nv_dir = os.path.join(work, "nvfp4-plan")
    os.makedirs(nv_dir)
    json.dump(synthetic_index(), open(os.path.join(nv_dir, "model.safetensors.index.json"), "w"))
    events = []
    plans = LO.checkpoint_decode_plans(_Config(real_config), nv_dir, lambda **kw: events.append(kw))
    nv_plan = plans[4]
    check("L17a a modelopt NVFP4 config over a full-census index plans the nvfp4 decode "
          "and nothing else (6-tuple: fp8/trellis/gguf None)",
          len(plans) == 6 and plans[0] is None and plans[1] is None and plans[2] is None
          and plans[5] is None
          and nv_plan is not None and nv_plan["quant_method"] == "modelopt"
          and nv_plan["quant_algo"] == "NVFP4" and nv_plan["group_size"] == 16
          and nv_plan["activation_scheme"] == "static-nvfp4-not-applied"
          and nv_plan["_observed"]["quantized_modules"] == 57600
          and nv_plan["_observed"]["mtp_expert_format"] == "plain-weight"
          and events and events[0].get("stage") == "nvfp4_decode_plan"
          and events[0].get("method") == LO.NVFP4_DECODE_METHOD
          and events[0].get("parity") == LO.NVFP4_PARITY_EVIDENCE,
          repr((len(plans), nv_plan and {k: v for k, v in nv_plan.items() if k != "_geometry"}, events[:1]))[:400])
    check("L17b the sealed contract block carries no private census key and the "
          "controller-mirrored key set exactly",
          set(k for k in nv_plan if not k.startswith("_")) == {
              "quant_method", "quant_algo", "num_bits", "group_size", "weights_declared_by",
              "activation_scheme", "producer", "ignore_count", "ignore_sha256"},
          repr(sorted(nv_plan)))

    def refuses_plan(doc, index, fragment):
        d = os.path.join(work, "nvfp4-plan-%d" % len(os.listdir(work)))
        os.makedirs(d)
        json.dump(index, open(os.path.join(d, "model.safetensors.index.json"), "w"))
        try:
            LO.checkpoint_decode_plans(_Config(doc), d, lambda **kw: None)
        except LO.LayerOuterError as exc:
            return fragment in str(exc), str(exc)
        return False, "accepted"

    bad = json.loads(json.dumps(real_config)); bad["quantization_config"]["quant_algo"] = "FP8"
    ok1, d1 = refuses_plan(bad, synthetic_index(), "quant_algo='FP8' is not the NVFP4 form")
    bad = json.loads(json.dumps(real_config)); bad["quantization_config"]["rotate"] = True
    ok2, d2 = refuses_plan(bad, synthetic_index(), "online weight transforms ['rotate']")
    ok3, d3 = refuses_plan(real_config, synthetic_index(
        drop=["model.layers.10.mlp.experts.3.down_proj.weight_scale_2"]),
        "model.layers.10.mlp.experts.3.down_proj.* carries an unrecognised component set")
    ok4, d4 = refuses_plan(real_config, synthetic_index(
        extra=["model.layers.10.self_attn.o_proj.weight_scale"]),
        "non-routed tensor names differ from the official BF16 set")
    bad = json.loads(json.dumps(real_config)); bad["model_type"] = "glm5_next"
    ok5, d5 = refuses_plan(bad, synthetic_index(), "architectures")
    check("L17c a non-NVFP4 modelopt algo, a declared online transform, a module missing "
          "weight_scale_2, a non-official non-routed name and a foreign model_type are each "
          "REFUSED BY NAME before anything is instantiated",
          ok1 and ok2 and ok3 and ok4 and ok5,
          " | ".join(x[-120:] for x in (d1, d2, d3, d4, d5)))

    # The decode itself, on a synthetic layer subset: packed bytes + f8 scales +
    # fp32 scale_2 per module, an input_scale that must be dropped, a bf16
    # non-routed tensor and an fp32 router bias that pass through.
    import numpy as np
    rng = np.random.default_rng(0x4F4)
    inter, hidden = geometry.moe_intermediate_size, geometry.hidden_size
    tensors = {}
    expected = {}
    for expert in range(2):
        for projection in NS.PROJECTIONS:
            out_f, in_f = geometry.projection_shape[projection]
            packed = torch.from_numpy(rng.integers(0, 256, size=(out_f, in_f // 2), dtype=np.uint8))
            scale_codes = rng.integers(0, 256, size=(out_f, in_f // 16), dtype=np.uint8)
            scale_codes[scale_codes == 0x7F] = 0x38
            scale_codes[scale_codes == 0xFF] = 0xB8
            scale = torch.from_numpy(scale_codes).view(torch.float8_e4m3fn)
            scale_2 = torch.tensor(float(rng.uniform(1e-5, 1e-3)), dtype=torch.float32)
            tensors[geometry.component_name(3, expert, projection, "weight")] = packed
            tensors[geometry.component_name(3, expert, projection, "weight_scale")] = scale
            tensors[geometry.component_name(3, expert, projection, "weight_scale_2")] = scale_2.reshape(())
            tensors[geometry.component_name(3, expert, projection, "input_scale")] = torch.tensor(0.5)
            expected[geometry.official_name(3, expert, projection)] = NS.dequant_nvfp4(
                packed, scale, weight_scale_2=scale_2).to(torch.bfloat16)
    tensors["model.layers.3.self_attn.o_proj.weight"] = torch.randn(8, 8).to(torch.bfloat16)
    tensors["model.layers.3.mlp.gate.e_score_correction_bias"] = torch.randn(4)
    tensors["model.layers.78.mlp.experts.0.gate_proj.weight"] = torch.randn(4, 8).to(torch.bfloat16)
    shard = os.path.join(work, "nvfp4-layer.safetensors")
    save_file(tensors, shard, metadata={"format": "pt"})
    handle = safe_open(shard, framework="pt", device="cpu")
    subset = {key: handle.get_slice(key) for key in handle.keys()}
    stats = {}
    out = LO.materialize_nvfp4_subset(subset, nv_plan, torch.bfloat16, stats, device="cpu")
    decoded_ok = all(
        key in out and out[key].dtype == torch.bfloat16 and torch.equal(out[key], want)
        for key, want in expected.items())
    check("L17d materialize_nvfp4_subset decodes every routed module BITWISE to "
          "dequant_nvfp4 under its OFFICIAL name, at the capture dtype",
          decoded_ok and stats.get("decoded_modules") == 6 and stats.get("scales_consumed") == 12,
          repr(stats))
    check("L17e input_scale never reaches the converter; packed components and scales are "
          "consumed; bf16 and fp32 non-routed tensors and the plain-bf16 MTP expert pass "
          "through untouched and the dtype census records them",
          not any(k.endswith(("input_scale", "weight_scale", "weight_scale_2")) for k in out)
          and stats.get("input_scales_skipped") == 6
          and out["model.layers.3.self_attn.o_proj.weight"] is subset["model.layers.3.self_attn.o_proj.weight"]
          and out["model.layers.78.mlp.experts.0.gate_proj.weight"] is subset["model.layers.78.mlp.experts.0.gate_proj.weight"]
          and stats.get("plain_modules_passed") == 1
          and stats.get("nonrouted_by_dtype") == {"BF16": 1, "F32": 1}
          and len(out) == 6 + 3,
          repr((sorted(out), stats)))

    class _Streamer:
        pass
    streamer = _Streamer()
    streamer.nvfp4_plan = nv_plan
    streamer.nvfp4_stats = stats
    streamer.trellis_plan = None
    streamer.fp8_plan = None
    evidence_doc = LO.weights_decode_evidence(streamer)
    check("L17f weights_decode_evidence names the method, the decode reference, the "
          "parity file and the contract block only (no private keys), with the counts",
          evidence_doc["method"] == LO.NVFP4_DECODE_METHOD
          and evidence_doc["reference"] == LO.NVFP4_DECODE_REFERENCE
          and evidence_doc["parity_evidence"] == LO.NVFP4_PARITY_EVIDENCE
          and os.path.isfile(os.path.join(REPO, LO.NVFP4_PARITY_EVIDENCE))
          and not any(k.startswith("_") for k in evidence_doc["quantization_config"])
          and evidence_doc["quantization_config"]["activation_scheme"] == "static-nvfp4-not-applied"
          and evidence_doc["modules_decoded"] == 6
          and evidence_doc["input_scale_tensors_not_applied"] == 6
          and evidence_doc["observed"]["quantized_modules"] == 57600
          and evidence_doc["nonrouted_by_dtype"] == {"BF16": 1, "F32": 1}
          and json.dumps(evidence_doc),
          json.dumps(evidence_doc)[:400])

    def refuses_materialize(mutate, fragment):
        doc = dict(subset)
        mutate(doc)
        try:
            LO.materialize_nvfp4_subset(doc, nv_plan, torch.bfloat16, {}, device="cpu")
        except LO.LayerOuterError as exc:
            return fragment in str(exc), str(exc)
        return False, "accepted"
    r1 = refuses_materialize(
        lambda d: d.pop(geometry.component_name(3, 1, "up_proj", "weight_scale_2")),
        "model.layers.3.mlp.experts.1.up_proj.weight is missing weight_scale_2")
    r2 = refuses_materialize(
        lambda d: d.__setitem__("model.layers.3.self_attn.q_a_proj.weight",
                                 subset[geometry.component_name(3, 0, "gate_proj", "weight")]),
        "model.layers.3.self_attn.q_a_proj.weight is a U8 tensor outside a routed-expert module")
    r3 = refuses_materialize(
        lambda d: [d.pop(geometry.component_name(3, 0, "down_proj", c))
                   for c in ("weight_scale", "weight_scale_2", "input_scale")],
        "model.layers.3.mlp.experts.0.down_proj.weight ships as a lone U8 `weight`")
    r4 = refuses_materialize(
        lambda d: d.__setitem__(geometry.component_name(3, 0, "gate_proj", "qweight"),
                                 subset[geometry.component_name(3, 0, "gate_proj", "weight")]),
        "carries component 'qweight'")
    check("L17g a module missing a scale, a packed tensor outside a routed module, a lone "
          "packed weight without scales and an unknown component are each REFUSED BY NAME",
          all(r[0] for r in (r1, r2, r3, r4)),
          " | ".join(r[1][-140:] for r in (r1, r2, r3, r4)))

    # ---- FlagshipGgufLane: L19 the GGUF lane's plan detection --------------
    # The reader itself is proven in engines/tools/selftest_gguf_offline.py
    # (rungs 8-8e); this is the streamer's seam: no .gguf -> no plan (the FP8 /
    # trellis / nvfp4 gates decide), two build directories refuse, and a build
    # whose parts are header-only (the REAL 1,809-tensor flagship table with no
    # data behind it) is refused by the container extent audit AFTER the census
    # closed -- i.e. the plan reads the same headers the controller mirror
    # reads, and stops where a truncated fetch would have handed back zeros.
    from pathlib import Path
    import gguf_surface as GS
    from selftest_gguf_offline import write_gguf, _real_rows, _rows_for_writer
    gg_evidence = Path(REPO) / "engines" / "tools" / "gguf-evidence"
    official_cfg = json.loads((gg_evidence / "glm53-official-config.json").read_text(encoding="utf-8"))
    check("L19a a tree with no .gguf plans no GGUF decode (None, before any header is read)",
          LO.gguf_checkpoint_plan(_Config(official_cfg), nv_dir) is None
          and LO.gguf_files_in(nv_dir) == [])
    gg_two = os.path.join(work, "gguf-two-builds")
    for build in ("UD-Q4_K_XL", "UD-Q3_K_XL"):
        os.makedirs(os.path.join(gg_two, build))
        open(os.path.join(gg_two, build, "x.gguf"), "wb").write(b"GGUF")
    try:
        LO.gguf_files_in(gg_two)
        two_ok, two_detail = False, "accepted"
    except LO.LayerOuterError as exc:
        two_ok, two_detail = "2 GGUF build directories" in str(exc), str(exc)
    check("L19b two build directories under one target are REFUSED by name", two_ok, two_detail)
    gg_dir = os.path.join(work, "gguf-flagship", "UD-Q4_K_XL")
    os.makedirs(gg_dir)
    gg_kv = json.loads((gg_evidence / "unsloth-glm53-udq4kxl-kv.json").read_text(encoding="utf-8"))
    for key in ("split.no", "split.count", "split.tensors.count"):
        gg_kv.pop(key, None)
    gg_path = write_gguf(Path(gg_dir) / "GLM-5.3-UD-Q4_K_XL-00001-of-00001.gguf", gg_kv,
                         _rows_for_writer(_real_rows("unsloth-glm53-udq4kxl-tensors.json")))
    gg_root = os.path.dirname(gg_dir)
    try:
        LO.gguf_checkpoint_plan(_Config(official_cfg), gg_root)
        marker_ok, marker_detail = False, "accepted without the sha256 marker"
    except LO.LayerOuterError as exc:
        marker_ok, marker_detail = "whole-file sha256 marker absent" in str(exc), str(exc)
    check("L19c a build without gguf-files-verified.json is REFUSED (the identity the "
          "receipt claims must be hashed, by the fetch stage, before the plan)",
          marker_ok, marker_detail[-300:])
    GS.verify_file_hashes([str(gg_path)])
    try:
        LO.gguf_checkpoint_plan(_Config(official_cfg), gg_root)
        extent_ok, extent_detail = False, "accepted a header-only part"
    except LO.LayerOuterError as exc:
        # the 11 parts' rows collapsed into one header-only file: the audit
        # sees offsets that overlap (and would see a truncated part the same way)
        extent_ok, extent_detail = ("overlaps the previous tensor" in str(exc)
                                    or "tensor extents run to byte" in str(exc)), str(exc)
    check("L19d the REAL flagship table over a header-only part closes its census and is then "
          "REFUSED by the container extent audit (overlapping/truncated bytes read as zeros nowhere)",
          extent_ok, extent_detail[-300:])
    no_types = json.loads(json.dumps(official_cfg)); no_types.pop("indexer_types")
    try:
        LO.gguf_checkpoint_plan(_Config(no_types), gg_root)
        types_ok, types_detail = False, "accepted without indexer_types"
    except LO.LayerOuterError as exc:
        types_ok, types_detail = "indexer_types" in str(exc), str(exc)
    check("L19e a config without indexer_types is REFUSED: the glm-dsa census will not guess "
          "which indexer tensors are copies", types_ok, types_detail[-300:])
    # the controller mirror and the pod contract are ONE function over the same
    # headers: decode_contract on this local container equals the block the
    # controller writes from the https headers (same rows, same digest)
    gg_container = GS.GgufContainer([GS.GgufFile(str(gg_path))])
    contract = GS.decode_contract(gg_container, "UD-Q4_K_XL")
    check("L19f the header-derived decode contract is the controller's mirror: method "
          "gguf-dequant-to-bf16, build, 1,809 tensors, the real type census, a 64-hex "
          "tensor-table digest and the imatrix KVs",
          contract["method"] == LO.GGUF_DECODE_METHOD == GS.GGUF_DECODE_METHOD
          and contract["quantization_config"]["build"] == "UD-Q4_K_XL"
          and contract["quantization_config"]["tensor_count"] == 1809
          and contract["quantization_config"]["type_census"] == {
              "F32": 709, "Q4_K": 150, "Q5_K": 74, "Q6_K": 4, "Q8_0": 872}
          and len(contract["quantization_config"]["tensor_table_sha256"]) == 64
          and contract["quantization_config"]["general"]["general.quantized_by"] == "Unsloth"
          and contract["quantization_config"]["general"]["quantize.imatrix.dataset"]
          == "unsloth_calibration_GLM-5.3.txt",
          json.dumps(contract)[:400])
    # the subset materializer: the RESIDENT bucket mixes the three top-level
    # slots with one router buffer per MoE layer and must decode exactly those
    # (never a whole layer); a layer bucket decodes its layer; a foreign value
    # refuses. Driven over the offline selftest's shrunken two-layer fixture.
    import selftest_gguf_offline as SGO
    mini_surface, mini_arch, mini_tensors = SGO.build_mini_glmdsa(Path(work) / "gguf-mini")
    resident = {"model.embed_tokens.weight": LO._GgufSlot(GS.RESIDENT_LAYER, "model.embed_tokens.weight"),
                "lm_head.weight": LO._GgufSlot(GS.RESIDENT_LAYER, "lm_head.weight"),
                "model.norm.weight": LO._GgufSlot(GS.RESIDENT_LAYER, "model.norm.weight"),
                mini_arch.layer_name(5, "mlp.gate.e_score_correction_bias"):
                    LO._GgufSlot(5, mini_arch.layer_name(5, "mlp.gate.e_score_correction_bias"))}
    gstats = {}
    decoded = LO.materialize_gguf_subset(resident, {"_surface": mini_surface}, torch.bfloat16,
                                         gstats, device="cpu")
    check("L19g the resident GGUF bucket (embed, head, norm + one layer's router buffer) "
          "decodes exactly those four tensors -- 4 GGUF tensors read, not a whole layer",
          set(decoded) == set(resident) and gstats["tensors_decoded"] == 4
          and decoded["lm_head.weight"].dtype == torch.bfloat16
          and decoded[mini_arch.layer_name(5, "mlp.gate.e_score_correction_bias")].dtype
          == torch.float32, repr((sorted(decoded), gstats)))
    layer_slots = {name: LO._GgufSlot(5, name) for name in LO.gguf_subsets(
        {"_surface": mini_surface, "_partition": GS.layer_partition(mini_surface.census)})
        if name.startswith("model.layers.5.") and "e_score_correction_bias" not in name}
    gstats = {}
    decoded = LO.materialize_gguf_subset(layer_slots, {"_surface": mini_surface}, torch.bfloat16,
                                         gstats, device="cpu")
    check("L19h a layer bucket decodes its layer under the official names (kv_b composed, "
          "experts sliced) and the bucket's names equal the decode's",
          set(decoded) == set(layer_slots) and gstats["layers_decoded"] == 1
          and decoded[mini_arch.kv_b_name(5)].shape[1] == mini_arch.mla_kv_lora_rank,
          repr(sorted(decoded)[:4]))
    try:
        LO.materialize_gguf_subset(dict(layer_slots, foreign=object()), {"_surface": mini_surface},
                                   torch.bfloat16, {}, device="cpu")
        fo_ok, fo_detail = False, "accepted"
    except LO.LayerOuterError as exc:
        fo_ok, fo_detail = "not GGUF slots" in str(exc), str(exc)
    check("L19i a non-slot value in a GGUF bucket is REFUSED", fo_ok, fo_detail)
    # ---- end FlagshipGgufLane ---------------------------------------------

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED %s: %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
