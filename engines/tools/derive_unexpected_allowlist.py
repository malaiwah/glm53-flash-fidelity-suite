#!/usr/bin/env python3
"""Author an exact unexpected-tensor allowlist from the streamed loader itself.

`hf_capture.py --schedule layer-outer` refuses, per streamed layer, any
checkpoint tensor the model did not consume unless the complete set equals a
pinned allowlist. The only derivation that cannot disagree with that guard is
the guard's own loader: `transformers`' converter decides what is "unexpected",
and it decides differently from a reading of the checkpoint index. Found the
hard way on 2026-09-03: an allowlist authored from the pre-streaming aggregate
(the MTP block, 791 names) omitted the 50 DSA indexer tensors that Fruit's
checkpoint carries on layers whose `indexer_types` entry is `shared`, and the
first exact-allowlist capture refused at layer 3 -- after the pod had been
paid for, bootstrapped and had fetched the model.

This tool streams every layer once (no forward pass, no GPU needed; a CPU with
one layer of headroom is enough), unions each layer's report with the resident
load's, and writes the sorted names as the JSON array `hf_capture` binds by
SHA-256, plus a provenance sidecar naming the checkpoint identity, the stack,
and the per-layer counts it saw.

    engines/tools/derive_unexpected_allowlist.py \
        --model /path/to/checkpoint --out engines/tools/layer-outer-evidence/X.json

A load that is not clean -- missing, mismatched, conversion errors, error
messages -- is refused: an allowlist may only describe tensors a clean load
left unused, never paper over a broken one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_outer  # noqa: E402


def _fail(message: str) -> SystemExit:
    return SystemExit("derive_unexpected_allowlist: ERROR: %s" % message)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_names_sha256(names: List[str]) -> str:
    return hashlib.sha256(
        json.dumps(names, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def derive(model_dir: str, dtype: str, device: str) -> Dict[str, Any]:
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_dir)
    architectures = list(getattr(config, "architectures", None) or [])
    cls = None
    for name in architectures:
        cls = getattr(transformers, name, None)
        if cls is not None:
            break
    if cls is None:
        raise _fail("config.architectures is %r and transformers exposes none of them"
                    % (architectures,))

    per_layer: Dict[int, List[str]] = {}
    per_layer_problems: Dict[int, Dict[str, Any]] = {}

    def layer_guard(index: int, info: Dict[str, Any]) -> None:
        per_layer[index] = sorted(info.get("unexpected_keys") or [])
        problems = {
            key: info.get(key) for key in
            ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")
            if info.get(key)}
        if problems:
            per_layer_problems[index] = problems

    def log(**fields: Any) -> None:
        print(json.dumps(fields, sort_keys=True, default=str), flush=True)

    started = time.monotonic()
    streamer = layer_outer.build_streamed_model(
        model_dir, cls, config, dtype, device, log, layer_guard=layer_guard)
    try:
        layer_count = len(streamer.layers)
        for index in range(layer_count):
            streamer.load_layer(index)
            streamer.free_layer(index)
            log(stage="derive_layer", index=index, layers=layer_count,
                unexpected=len(per_layer.get(index, [])))
        report = layer_outer.streamed_loading_info(streamer)
    finally:
        streamer.close()

    if per_layer_problems:
        raise _fail("the load was not clean; an allowlist cannot be authored from it: %s"
                    % json.dumps({str(k): {kk: (len(v) if isinstance(v, (list, dict)) else v)
                                           for kk, v in val.items()}
                                  for k, val in per_layer_problems.items()},
                                 sort_keys=True)[:1200])
    for key in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if report.get(key):
            raise _fail("aggregate load report has %s: %s"
                        % (key, json.dumps(report[key], default=str)[:600]))

    names = sorted(set(report["unexpected_keys"]))
    if not names:
        raise _fail("the load left no checkpoint tensor unused; no allowlist is needed "
                    "and none may be authored")
    provenance = {
        "schema": "malaiwah.unexpected-tensor-allowlist-provenance.v1",
        "derived_by": os.path.basename(__file__),
        "derived_by_sha256": _sha256_file(os.path.abspath(__file__)),
        "layer_outer_sha256": _sha256_file(os.path.abspath(layer_outer.__file__)),
        "derived_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": os.path.abspath(model_dir),
        "config_sha256": _sha256_file(os.path.join(model_dir, "config.json")),
        "index_sha256": _sha256_file(
            os.path.join(model_dir, "model.safetensors.index.json")),
        "architecture": cls.__name__,
        "dtype": dtype, "device": device,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "decoder_layers": layer_count,
        "resident_load_unexpected": sorted(
            set(report["unexpected_keys"])
            - {name for rows in per_layer.values() for name in rows}),
        "per_layer_unexpected_counts": {
            str(index): len(rows) for index, rows in sorted(per_layer.items()) if rows},
        "count": len(names),
        "canonical_sorted_names_sha256": _canonical_names_sha256(names),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return {"names": names, "provenance": provenance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="local checkpoint directory (config, index, shards)")
    parser.add_argument("--out", required=True,
                        help="allowlist JSON array to write; refuses to overwrite")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    for name in ("config.json", "model.safetensors.index.json"):
        if not os.path.isfile(os.path.join(args.model, name)):
            raise _fail("--model lacks %s" % name)
    if os.path.lexists(args.out):
        raise _fail("--out exists; an allowlist is authored once per pin, delete it "
                    "deliberately: %s" % args.out)
    sidecar = args.out + ".provenance.json"
    if os.path.lexists(sidecar):
        raise _fail("provenance sidecar exists: %s" % sidecar)

    result = derive(args.model, args.dtype, args.device)
    body = json.dumps(result["names"], indent=2, ensure_ascii=False) + "\n"
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(body)
    result["provenance"]["artifact_sha256"] = hashlib.sha256(
        body.encode("utf-8")).hexdigest()
    with open(sidecar, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result["provenance"], indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "done", "out": args.out, "count": len(result["names"]),
                      "artifact_sha256": result["provenance"]["artifact_sha256"],
                      "canonical_sorted_names_sha256":
                          result["provenance"]["canonical_sorted_names_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
