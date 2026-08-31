#!/usr/bin/env python3
"""F1-F9: `engines/tools/fetch_truncated_ckpt.py` against three real key layouts.

Why this file exists
--------------------
The truncation fetcher was written for `zai-org/GLM-5.3-BF16` and hard-coded
three of that checkpoint's accidents:

  * the decoder-layer key is ``^model\\.layers\\.(\\d+)\\.``;
  * the layer count is ``config["num_hidden_layers"]``;
  * a per-layer schedule is a TOP-LEVEL list of STRINGS.

None of the three architectures this suite was asked to qualify next satisfies
all three, and the failure mode of the first one is silent and expensive:

  * `deepseek-ai/DeepSeek-V4-Flash-0731` ships DeepSeek's native names
    (``layers.N.attn.wq_a.weight``) with no ``model.`` prefix.  Under the old
    regex NOT ONE key matched, every key therefore counted as a non-layer key
    that must be kept, and ``--layers 4`` would have planned a fetch of the
    ENTIRE 166.9 GB checkpoint while logging ``kept_tensors 72317`` as though
    that were a truncation.  F4 is that case; it fails on the pre-change tree
    by planning 100% of the bytes.
  * `MiniMaxAI/MiniMax-M3` keeps the layer count in ``text_config`` and its
    per-layer schedules as lists of INTS, one of them nested inside
    ``sparse_attention_config``.  F6/F7 are that case.

Everything here runs offline: the HTTP `Fetcher` is replaced by one that reads
a synthetic checkpoint out of a temp directory, so the tool's real planning,
range-coalescing, sparse-write, index-pruning and config-surgery code paths run
end to end with no network and no weights.

  bin/selftest_fetch_truncated.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "engines" / "tools" / "fetch_truncated_ckpt.py"

FAKE_REVISION = "0" * 40


def load_tool():
    spec = importlib.util.spec_from_file_location("fetch_truncated_ckpt", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# a synthetic published checkpoint on local disk
# ---------------------------------------------------------------------------


def write_shard(path: Path, tensors: Dict[str, Tuple[str, List[int], int]]) -> Dict[str, Any]:
    """Write a real safetensors file. `tensors` maps name -> (dtype, shape, nbytes)."""
    header: Dict[str, Any] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, nbytes = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for name in sorted(tensors):
            handle.write(bytes([(hash(name) + i) & 0xFF for i in range(tensors[name][2])]))
    return header


def build_repo(root: Path, config: Dict[str, Any],
               shards: Dict[str, Dict[str, Tuple[str, List[int], int]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    weight_map: Dict[str, str] = {}
    total = 0
    for shard, tensors in shards.items():
        write_shard(root / shard, tensors)
        for name, (_dtype, _shape, nbytes) in tensors.items():
            weight_map[name] = shard
            total += nbytes
    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))


def local_fetcher(module, repo_root: Path):
    """Replace the HTTP fetcher with one that reads `repo_root`."""

    class FileFetcher(object):
        def __init__(self, repo: str, revision: str, token: Any) -> None:
            self.root = repo_root

        def whole(self, name: str) -> bytes:
            return (self.root / name).read_bytes()

        def ranged(self, name: str, start: int, stop_inclusive: int) -> bytes:
            with open(self.root / name, "rb") as handle:
                handle.seek(start)
                data = handle.read(stop_inclusive - start + 1)
            if len(data) != stop_inclusive - start + 1:
                raise IOError("short read")
            return data

        def header(self, name: str):
            raw = self.ranged(name, 0, 7)
            length = struct.unpack("<Q", raw)[0]
            return length, json.loads(self.ranged(name, 8, 8 + length - 1))

    module.Fetcher = FileFetcher


# ---------------------------------------------------------------------------
# the three fixtures
# ---------------------------------------------------------------------------

BF16 = "BF16"
LAYERS = 6


def glm_repo(root: Path) -> None:
    """`model.layers.N.` + top-level string schedules -- the original shape."""
    config = {
        "model_type": "glm_moe_dsa", "num_hidden_layers": LAYERS,
        "hidden_size": 64, "first_k_dense_replace": 3,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 3,
        "indexer_types": ["full"] * 3 + ["shared"] * 3,
        "architectures": ["GlmMoeDsaForCausalLM"],
    }
    shards: Dict[str, Dict[str, Tuple[str, List[int], int]]] = {}
    for layer in range(LAYERS):
        shard = "model-%05d-of-00007.safetensors" % (layer + 1)
        shards[shard] = {
            "model.layers.%d.self_attn.q_proj.weight" % layer: (BF16, [8, 8], 128),
            "model.layers.%d.mlp.experts.0.gate_proj.weight" % layer: (BF16, [8, 8], 128),
            "model.layers.%d.mlp.experts.1.gate_proj.weight" % layer: (BF16, [8, 8], 128),
        }
    shards["model-00007-of-00007.safetensors"] = {
        "model.embed_tokens.weight": (BF16, [16, 8], 256),
        "model.norm.weight": (BF16, [8], 16),
        "lm_head.weight": (BF16, [16, 8], 256),
    }
    build_repo(root, config, shards)


def dsv4_repo(root: Path) -> None:
    """DeepSeek's native names: no `model.` prefix, plus an `mtp.` subtree."""
    config = {
        "model_type": "deepseek_v4", "num_hidden_layers": LAYERS,
        "hidden_size": 64, "num_hash_layers": 3,
        # 6 layers + 3 MTP modules: length 9 != 6, so it must be LEFT ALONE.
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 4, 4],
        "architectures": ["DeepseekV4ForCausalLM"],
    }
    shards: Dict[str, Dict[str, Tuple[str, List[int], int]]] = {}
    for layer in range(LAYERS):
        shards["model-%05d-of-00008.safetensors" % (layer + 1)] = {
            "layers.%d.attn.wq_a.weight" % layer: (BF16, [8, 8], 128),
            "layers.%d.ffn.experts.0.w1.weight" % layer: (BF16, [8, 8], 128),
            "layers.%d.ffn.experts.1.w1.weight" % layer: (BF16, [8, 8], 128),
        }
    shards["model-00007-of-00008.safetensors"] = {
        "mtp.0.attn.wq_a.weight": (BF16, [8, 8], 128),
        "mtp.0.ffn.experts.0.w1.weight": (BF16, [8, 8], 128),
    }
    shards["model-00008-of-00008.safetensors"] = {
        "embed.weight": (BF16, [16, 8], 256),
        "norm.weight": (BF16, [8], 16),
        "head.weight": (BF16, [16, 8], 256),
    }
    build_repo(root, config, shards)


def mm_repo(root: Path) -> None:
    """VL: layer count under `text_config`, int schedules, a vision tower with
    its OWN `...encoder.layers.N.` keys that must survive untouched."""
    config = {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": LAYERS, "hidden_size": 64,
            "moe_layer_freq": [0, 0, 0, 1, 1, 1],
            # A layer-INDEX list, one-indexed, in the shape Qwen3.8-Flash-Next
            # ships it. Length 2 != 6, so the schedule truncator never sees it.
            "ple_layer_ids": [2, 5],
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_attention_freq": [0, 0, 0, 1, 1, 1],
                "sparse_disable_index_value": [0, 0, 0, 1, 1, 1],
                "sparse_block_size": 128,
            },
        },
        "vision_config": {"num_hidden_layers": 4, "hidden_size": 32},
    }
    shards: Dict[str, Dict[str, Tuple[str, List[int], int]]] = {}
    for layer in range(LAYERS):
        shards["model-%05d-of-00008.safetensors" % (layer + 1)] = {
            "language_model.model.layers.%d.self_attn.q_proj.weight" % layer: (BF16, [8, 8], 128),
            "language_model.model.layers.%d.block_sparse_moe.experts.0.w1.weight" % layer:
                (BF16, [8, 8], 128),
        }
    shards["model-00007-of-00008.safetensors"] = {
        "vision_tower.vision_model.encoder.layers.%d.mlp.fc1.weight" % v: (BF16, [4, 4], 32)
        for v in range(4)
    }
    shards["model-00008-of-00008.safetensors"] = {
        "language_model.model.embed_tokens.weight": (BF16, [16, 8], 256),
        "language_model.model.norm.weight": (BF16, [8], 16),
        "language_model.lm_head.weight": (BF16, [16, 8], 256),
    }
    build_repo(root, config, shards)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class Result(object):
    def __init__(self, rc: int, dest: Path, receipt: Path) -> None:
        self.rc = rc
        self.dest = dest
        self.receipt = receipt

    @staticmethod
    def _read(path: Path) -> Dict[str, Any]:
        # A rung that fails must report a FAIL, not crash the battery: on a tree
        # without these flags several runs produce no output at all, and the
        # point of running this file there is to see which rungs go red.
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._read(self.dest / "config.json")

    @property
    def index(self) -> Dict[str, Any]:
        return self._read(self.dest / "model.safetensors.index.json") or {"weight_map": {}}

    @property
    def data(self) -> Dict[str, Any]:
        return self._read(self.receipt)


def run(module, repo_root: Path, work: Path, name: str, argv: List[str]) -> Result:
    local_fetcher(module, repo_root)
    dest = work / (name + ".ckpt")
    receipt = work / (name + ".receipt.json")
    argv = ["--repo", "fixture/" + name, "--revision", FAKE_REVISION,
            "--dest", str(dest), "--receipt", str(receipt), "--threads", "2"] + argv
    try:
        rc = module.main(argv)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    return Result(rc, dest, receipt)


PASS = [0]
FAIL = [0]


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASS[0] += 1
        print("  PASS  %s" % name)
    else:
        FAIL[0] += 1
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def main() -> int:
    module = load_tool()
    work = Path(tempfile.mkdtemp(prefix="selftest-fetch-trunc-"))
    devnull = open(os.devnull, "w")
    real_stdout = sys.stdout

    def quiet(fn, *a, **k):
        sys.stdout = devnull
        try:
            return fn(*a, **k)
        finally:
            sys.stdout = real_stdout

    print("== fetch_truncated_ckpt: three key layouts ==")

    # ---- GLM-5.3 shape: the DEFAULTS must not have moved -------------------
    glm = work / "glm-src"
    glm_repo(glm)
    res = quiet(run, module, glm, work, "glm", ["--layers", "2"])
    kept = set(res.index["weight_map"])
    expected = {
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.1.gate_proj.weight",
    }
    check("F1 default layer regex keeps exactly layers 0..N-1 plus non-layer keys",
          res.rc == 0 and kept == expected, "kept=%s" % sorted(kept - expected))
    cfg = res.config
    check("F2 default config surgery trims num_hidden_layers and both string schedules",
          cfg.get("num_hidden_layers") == 2
          and cfg.get("mlp_layer_types") == ["dense", "dense"]
          and cfg.get("indexer_types") == ["full", "full"],
          json.dumps({k: cfg.get(k) for k in
                      ("num_hidden_layers", "mlp_layer_types", "indexer_types")}))
    shard_files = sorted(p.name for p in res.dest.glob("*.safetensors"))
    check("F3 only the shards carrying a kept tensor are materialised",
          shard_files == ["model-00001-of-00007.safetensors",
                          "model-00002-of-00007.safetensors",
                          "model-00007-of-00007.safetensors"],
          str(shard_files))

    # ---- DeepSeek-V4 shape --------------------------------------------------
    ds = work / "ds-src"
    dsv4_repo(ds)
    res = quiet(run, module, ds, work, "ds-default", ["--layers", "2"])
    check("F4 a layer regex that matches NOTHING is REFUSED, not silently "
          "promoted to a whole-checkpoint fetch",
          res.rc != 0 and not (work / "ds-default.receipt.json").exists(),
          "rc=%s" % res.rc)

    res = quiet(run, module, ds, work, "ds", [
        "--layers", "2",
        "--layer-key-regex", r"^layers\.(\d+)\.",
        "--drop-key-regex", r"^mtp\."])
    kept = set(res.index["weight_map"])
    expected = {"embed.weight", "norm.weight", "head.weight",
                "layers.0.attn.wq_a.weight", "layers.0.ffn.experts.0.w1.weight",
                "layers.0.ffn.experts.1.w1.weight",
                "layers.1.attn.wq_a.weight", "layers.1.ffn.experts.0.w1.weight",
                "layers.1.ffn.experts.1.w1.weight"}
    check("F5 native DeepSeek names truncate, and --drop-key-regex removes the "
          "MTP subtree the architecture does not build",
          res.rc == 0 and kept == expected,
          "extra=%s missing=%s" % (sorted(kept - expected), sorted(expected - kept)))
    cfg = res.config
    check("F6 a list whose length is NOT the layer count is left alone "
          "(DeepSeek's compress_ratios covers layers + MTP modules)",
          cfg.get("num_hidden_layers") == 2
          and cfg.get("compress_ratios") == [0, 0, 4, 128, 4, 128, 4, 4, 4],
          json.dumps(cfg.get("compress_ratios")))

    # ---- MiniMax-M3 / VL shape ---------------------------------------------
    mm = work / "mm-src"
    mm_repo(mm)
    res = quiet(run, module, mm, work, "mm", [
        "--layers", "2", "--config-node", "text_config",
        "--layer-key-regex", r"^language_model\.model\.layers\.(\d+)\."])
    kept = set(res.index["weight_map"])
    vision = {k for k in kept if k.startswith("vision_tower.")}
    text_layers = {k for k in kept if k.startswith("language_model.model.layers.")}
    check("F7 the vision tower's own encoder.layers.N keys are NOT truncated "
          "by the text tower's layer regex",
          res.rc == 0 and len(vision) == 4 and len(text_layers) == 4,
          "vision=%d text=%d" % (len(vision), len(text_layers)))
    cfg = res.config
    text = cfg.get("text_config") or {"num_hidden_layers": None, "moe_layer_freq": None,
                                      "sparse_attention_config": {}}
    check("F8 --config-node trims the TEXT tower's count and leaves the vision "
          "tower's alone",
          text["num_hidden_layers"] == 2
          and (cfg.get("vision_config") or {}).get("num_hidden_layers") == 4,
          json.dumps({"text": text["num_hidden_layers"],
                      "vision": (cfg.get("vision_config") or {}).get("num_hidden_layers")}))
    sac = text.get("sparse_attention_config") or {}
    check("F9 int schedules are trimmed, including one nested a dict deeper",
          text.get("moe_layer_freq") == [0, 0]
          and sac.get("sparse_attention_freq") == [0, 0]
          and sac.get("sparse_disable_index_value") == [0, 0]
          and sac.get("sparse_block_size") == 128,
          json.dumps({"moe_layer_freq": text.get("moe_layer_freq"),
                      "sparse_attention_freq": sac.get("sparse_attention_freq"),
                      "sparse_disable_index_value": sac.get("sparse_disable_index_value")}))
    check("F9b the receipt names the selection it used, and every schedule it cut "
          "(paths relative to --config-node)",
          res.data.get("config_node") == "text_config"
          and res.data.get("layer_key_regex") == r"^language_model\.model\.layers\.(\d+)\."
          and res.data.get("truncated_schedule_lists") == [
              "moe_layer_freq",
              "sparse_attention_config.sparse_attention_freq",
              "sparse_attention_config.sparse_disable_index_value"],
          json.dumps(res.data.get("truncated_schedule_lists")))

    # F10: a list of layer INDICES, not a per-layer schedule. Qwen3.8-Flash-Next
    # ships `text_config.ple_layer_ids = [2]`, one-indexed, and `Qwen4ExpTextConfig`
    # REFUSES a config whose ids fall outside `[1, num_hidden_layers]`:
    #   ValueError: ple_layer_ids must contain one-indexed ids in [1, 1], got [2].
    # A truncation that leaves it alone does not load at all, and no
    # length-based rule can find it: its length is 1, never the layer count.
    res = quiet(run, module, mm, work, "mm-idx", [
        "--layers", "2", "--config-node", "text_config",
        "--layer-key-regex", r"^language_model\.model\.layers\.(\d+)\.",
        "--config-index-list", "ple_layer_ids:1"])
    text = (res.config.get("text_config") or {})
    check("F10 --config-index-list drops out-of-range LAYER INDICES (one-indexed), "
          "which no length-based rule can reach",
          res.rc == 0 and text.get("ple_layer_ids") == [2]
          and res.data.get("filtered_layer_index_lists") == {"ple_layer_ids": [2]},
          "ple_layer_ids=%r receipt=%r"
          % (text.get("ple_layer_ids"), res.data.get("filtered_layer_index_lists")))

    print()
    print("selftest_fetch_truncated: %d passed, %d failed" % (PASS[0], FAIL[0]))
    return 1 if FAIL[0] else 0


if __name__ == "__main__":
    sys.exit(main())
