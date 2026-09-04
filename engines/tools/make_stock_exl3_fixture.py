#!/usr/bin/env python3
"""Build a small STOCK-payload-group exl3 fixture from a rank0-atom Fruit tree.

    engines/tools/make_stock_exl3_fixture.py --src <Fruit pilot dir> --out <fixture dir>
    engines/tools/make_stock_exl3_fixture.py --src ... --out ... --verify

Why this exists
---------------
`layer_outer.materialize_trellis_subset` decodes the layout stock exllamav3
writes -- `M.{trellis,suh,svh,<codebook>}` -- which the full GLM-5.3 quants
(drowzeys, wrldsuksgo2mars) ship at 300-400 GB. There was no small tree in
that layout, so the first three attempts to measure one found their bugs on a
rented H200, one per pod: a stats-dict KeyError at layer 0, a 0-dim lazy-slice
IndexError at layer 3, and a host-side decode that extrapolated past the
runtime cap.

`malaiwah/GLM-5.2-SIQ-Fruit-pilot` is a 0.6 GB `GlmMoeDsaForCausalLM` whose
routed experts are exl3 atoms under a `.rank0.` prefix. Dropping that one
path element yields a complete, real-tensor tree in the stock layout: same
architecture, same codebook, same decode, 6 layers instead of 78. `--verify`
then drives the REAL streamed loader over it and checks that the fused expert
parameter's expert-0 slice is BITWISE equal to an independent decode of that
module's payload -- the whole path, for free, in about fifteen seconds.

This writes a FIXTURE, not an artifact: the tree is a renamed copy and no
number measured on it may be published.

What proves what
----------------
* `--verify` proves the PLUMBING: grouping, per-module codebook, placement and
  orientation, bitwise against `decode_payload_hf`. It cannot prove the decoder
  reproduces the quantizer's weights -- both sides would agree and both be
  wrong.
* That is proven separately, against the bf16 source of a real trellis quant:
  `materialize_exl3_experts.py --reference` on `malaiwah/GLM-5.2-SIQ-Fruit` vs
  `malaiwah/GLM-5.2-SIQ-Fruit-bf16` measures cosine 0.99773 and rel_l2 6.74% in
  fp64 at K4 -- the expected reconstruction error at that bit rate, where a
  wrong codebook or unpack gives cosine near 0. Receipt committed at
  `engines/tools/layer-outer-evidence/fruit-siq-trellis-reconstruction.json`.
* COHERENCE is the capture's own generation probe ("The capital of France is"
  -> " Paris"), enforced by default and refusing the capture on failure. It has
  never run on trellis-decoded weights, because no trellis capture has
  completed yet; Fruit cannot answer it either, being an undertrained proxy
  (which is exactly why `--sanity-expect ''` exists for Fruit and why a
  production capture must NOT pass it).
* Do NOT use `training/fruit_pilot.pt` as a reference: it is a different
  snapshot (`w_down` 128-wide against the artifact's declared 256).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RANK0 = re.compile(r"^(.*)\.rank0\.((?:trellis|suh|svh|mcg|mul1))$")
METADATA_FILES = ("config.json", "generation_config.json", "LICENSE",
                  "chat_template.jinja", "tokenizer.json", "tokenizer_config.json")


def build(src: Path, out: Path, codebook: str, force: bool) -> dict:
    from safetensors.torch import load_file, save_file

    if out.exists():
        if not force:
            raise SystemExit("REFUSED: %s exists; pass --force" % out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    weight_map, renamed, kept = {}, 0, 0
    for shard in sorted(src.glob("*.safetensors")):
        fixed = {}
        for key, value in load_file(str(shard)).items():
            match = RANK0.match(key)
            if match:
                fixed["%s.%s" % (match.group(1), match.group(2))] = value
                renamed += 1
            else:
                fixed[key] = value
                kept += 1
        save_file(fixed, str(out / shard.name), metadata={"format": "pt"})
        for key in fixed:
            weight_map[key] = shard.name
    if not renamed:
        raise SystemExit("REFUSED: %s holds no .rank0 payload atoms" % src)
    for name in METADATA_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    quant.update({"quant_method": "exl3", "codebook": codebook,
                  "bits": quant.get("bits") or 3.0})
    config["quantization_config"] = quant
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(f.stat().st_size for f in out.glob("*.safetensors"))},
         "weight_map": dict(sorted(weight_map.items()))}, indent=1) + "\n", encoding="utf-8")
    return {"renamed": renamed, "kept": kept, "shards": len(weight_map),
            "layers": config.get("num_hidden_layers"),
            "architecture": config["architectures"][0], "codebook": codebook}


def verify(model_dir: Path, layer: int) -> dict:
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    import exl3hf_surface as xs
    import layer_outer as lo

    config = AutoConfig.from_pretrained(str(model_dir))
    cls = AutoModelForCausalLM._model_mapping[type(config)]
    streamer = lo.build_streamed_model(
        str(model_dir), cls, config, "bfloat16", "cpu", lambda **_f: None)
    streamer.load_layer(layer)
    model = streamer.model
    fused = [name for name, _ in model.named_parameters()
             if ".layers.%d.mlp.experts" % layer in name]
    if not fused:
        raise SystemExit("REFUSED: layer %d has no fused expert parameter" % layer)
    for name in fused:
        param = model.get_parameter(name)
        if not bool(torch.isfinite(param).all()) or not bool(param.abs().sum() > 0):
            raise SystemExit("REFUSED: %s is non-finite or all zero" % name)
    module = "model.layers.%d.mlp.experts.0.down_proj" % layer
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    codebook = (json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
                ["quantization_config"]["codebook"])
    with safe_open(str(model_dir / weight_map[module + ".trellis"]),
                   framework="pt", device="cpu") as handle:
        want = xs.decode_payload_hf(
            handle.get_tensor(module + ".trellis"), handle.get_tensor(module + ".suh"),
            handle.get_tensor(module + ".svh"), codebook=codebook).to(torch.bfloat16)
    down = model.get_parameter([n for n in fused if "down" in n][0])
    slice0 = down[0]
    # ORIENTATION IS PINNED, not accepted either way. A `.T` fallback here
    # would pass a transposed load, which is the failure this check is for.
    if tuple(slice0.shape) != tuple(want.shape):
        raise SystemExit(
            "REFUSED: fused expert-0 slice is %s, the independent decode is %s -- "
            "the decoded tensor is not landing in the model's orientation"
            % (tuple(slice0.shape), tuple(want.shape)))
    if not bool(torch.equal(slice0, want)):
        raise SystemExit(
            "REFUSED: the fused expert-0 slice is NOT bitwise equal to an "
            "independent decode of %s" % module)
    return {
        "fused_parameters": fused, "layer": layer,
        "expert0_bitwise_equal_to_independent_decode": True,
        "orientation": "pinned (no transpose fallback)",
        # WHAT THIS DOES NOT PROVE, stated so nobody reads more into it:
        # equality against `decode_payload_hf` proves the PLUMBING -- grouping,
        # per-module codebook, placement, orientation -- not that the decoder
        # itself reproduces the quantizer's weights. That needs a comparison
        # against the bf16 source (`materialize_exl3_experts.py --reference`
        # computes rel_l2 in fp64), and coherence needs the capture's own
        # generation probe, which is enforced on a production capture
        # (--sanity-expect default "Paris") and has NEVER run on
        # trellis-decoded weights.
        "not_proven": ["decoder-vs-quantizer weight error (needs a bf16 reference)",
                       "output coherence (needs the capture's generation probe)"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True, help="local rank0-atom checkpoint directory")
    parser.add_argument("--out", required=True, help="fixture directory to write")
    parser.add_argument("--codebook", default="mcg", choices=("mcg", "mul1"))
    parser.add_argument("--layer", type=int, default=3,
                        help="layer to stream in --verify (first MoE layer)")
    parser.add_argument("--verify", action="store_true",
                        help="drive the real streamed loader over the fixture")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    built = build(Path(args.src), Path(args.out), args.codebook, args.force)
    print(json.dumps({"stage": "built", **built}, sort_keys=True))
    if args.verify:
        print(json.dumps({"stage": "verified",
                          **verify(Path(args.out), args.layer)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
