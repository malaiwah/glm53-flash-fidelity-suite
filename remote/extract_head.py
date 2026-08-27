#!/usr/bin/env python3
"""Extract the shared LM head (and the final-norm weight, for the record) from a
GLM-5.3-Flash checkpoint into standalone safetensors files.

    extract_head.py --model /home/glm53/models/bf16 --out /home/glm53/out

The head published with the dataset is what lets anyone score a candidate
without holding the 643 GB checkpoint. Tensor-name candidates cover the naming
observed in the zai-org index (language_model prefix) plus plain fallbacks; the
script fails loudly if it cannot find exactly one of each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HEAD_PATTERNS = (
    "lm_head.weight",
    "model.lm_head.weight",
    "language_model.lm_head.weight",
    "model.language_model.lm_head.weight",
)
NORM_PATTERNS = (
    "model.norm.weight",
    "model.language_model.norm.weight",
    "language_model.norm.weight",
    "model.language_model.final_layernorm.weight",
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def find_one(weight_map: dict, patterns: tuple, label: str) -> str:
    hits = [n for n in patterns if n in weight_map]
    if len(hits) != 1:
        near = sorted(n for n in weight_map if label.split("_")[0] in n and "layers" not in n)
        raise SystemExit(f"{label}: expected exactly one of {patterns}, found {hits}; "
                         f"nearby names: {near[:12]}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file

    model = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = model / "model.safetensors"
        if not single.is_file():
            raise SystemExit(f"neither index nor model.safetensors under {model}")
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}

    if not any(n in weight_map for n in HEAD_PATTERNS):
        config = json.loads((model / "config.json").read_text())
        tied = config.get("tie_word_embeddings", config.get("text_config", {}).get("tie_word_embeddings"))
        if tied:
            embeds = [n for n in weight_map if n.endswith("embed_tokens.weight")]
            if len(embeds) == 1:
                print("tied embeddings: using", embeds[0], "as the head")
                globals()["HEAD_PATTERNS"] = (embeds[0],)

    head_name = find_one(weight_map, HEAD_PATTERNS, "lm_head")
    norm_name = find_one(weight_map, NORM_PATTERNS, "final_norm")

    report = {"model": str(model), "revision": None, "tensors": {}}
    rev = model / "revision.txt"
    if rev.is_file():
        report["revision"] = rev.read_text().strip()

    for label, name, fname in (("head", head_name, "head.safetensors"),
                               ("final_norm", norm_name, "final_norm.safetensors")):
        shard = model / weight_map[name]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        dst = out / fname
        save_file({"weight": tensor.contiguous()}, str(dst))
        report["tensors"][label] = {
            "source_tensor": name, "source_shard": weight_map[name],
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "file": fname, "sha256": sha256_file(dst),
        }
        print(label, name, list(tensor.shape), str(tensor.dtype), "->", dst)

    (out / "head-extraction.json").write_text(json.dumps(report, indent=2))
    print("EXTRACT_HEAD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
