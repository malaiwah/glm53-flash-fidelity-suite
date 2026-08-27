#!/usr/bin/env python3
"""Cross-validate our hidden-state replay against brandonmusic's independent
GLM-5.3-Flash-BF16-Teacher-Logits capture.

    cross_check.py suite    --logits DIR --out SUITE_DIR      # build mini-suite from his token rows
    cross_check.py compare  --logits DIR --capture CAP_DIR --head HEAD --out receipt.json

His dataset: window-NNNN.safetensors holding full-vocab float32 logits (2,047
scored positions per 2,048-token window), token rows as int32 .npy of shape
(2048,). Ours: capture the same token rows with fidelity.py, replay hidden@head,
and measure KLD(his ‖ ours) per position. Agreement at ~1e-5/-6 validates BOTH
pipelines end to end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def token_files(logits_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for f in sorted(logits_dir.rglob("*.npy")):
        if "mask" in f.name.lower():
            continue
        m = re.search(r"(\d+)", f.name)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def window_files(logits_dir: Path) -> dict[int, Path]:
    out = {}
    for f in sorted(logits_dir.rglob("window-*.safetensors")):
        m = re.search(r"window-(\d+)", f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def cmd_suite(args) -> int:
    import numpy as np

    logits_dir = Path(args.logits)
    rows = token_files(logits_dir)
    if not rows:
        raise SystemExit(f"no token .npy files under {logits_dir}")
    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    contexts = []
    for i, (num, f) in enumerate(rows):
        ids = [int(x) for x in np.load(f)]
        name = f"context-{i:04d}.json"
        (out / "tokens" / name).write_text(json.dumps(ids))
        contexts.append({"index": i, "stratum": "brandonmusic-teacher", "file": f"tokens/{name}",
                         "token_sha256": sha256_bytes(json.dumps(ids).encode()),
                         "tokens": len(ids), "source_cluster": f"bm-window-{num:04d}",
                         "source_npy": f.name})
    manifest = {
        "schema": "glm53flash-crosscheck-suite/1",
        "source_dataset": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
        "context_length": contexts[0]["tokens"],
        "scored_positions_per_context": contexts[0]["tokens"] - 1,
        "contexts": len(contexts),
        "context_index": contexts,
    }
    manifest["suite_token_sha256"] = sha256_bytes(
        "".join(c["token_sha256"] for c in contexts).encode())
    (out / "suite-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"contexts": len(contexts),
                      "suite_token_sha256": manifest["suite_token_sha256"]}))
    return 0


def cmd_compare(args) -> int:
    import torch
    from safetensors import safe_open

    logits_dir = Path(args.logits)
    cap = Path(args.capture)
    dev = torch.device(args.device)
    with safe_open(args.head, framework="pt", device="cpu") as f:
        key = "weight" if "weight" in f.keys() else f.keys()[0]
        head = f.get_tensor(key).to(dev, torch.bfloat16)
    windows = window_files(logits_dir)
    suite = json.loads(Path(args.suite, "suite-manifest.json").read_text())
    per = []
    for ctx in suite["context_index"]:
        num = int(ctx["source_cluster"].rsplit("-", 1)[1])
        wf = windows.get(num)
        hf = cap / f"hidden_{ctx['index']:04d}.safetensors"
        if wf is None or not hf.is_file():
            print(f"skip window {num}: missing {'logits' if wf is None else 'capture'}")
            continue
        with safe_open(str(hf), framework="pt", device="cpu") as f:
            hidden = f.get_tensor("hidden_states").to(dev, torch.bfloat16)
        with safe_open(str(wf), framework="pt", device="cpu") as f:
            k = f.keys()[0] if len(f.keys()) == 1 else next(
                (x for x in f.keys() if "logit" in x.lower()), f.keys()[0])
            his = f.get_tensor(k).to(dev, torch.float32)
        npos = min(hidden.shape[0], his.shape[0])
        ours = (hidden[:npos] @ head.T).float()
        ours = ours - ours.logsumexp(-1, keepdim=True)
        theirs = his[:npos] - his[:npos].logsumexp(-1, keepdim=True)
        kl = (theirs.exp() * (theirs - ours)).sum(-1).double()
        top1 = (theirs.argmax(-1) == ours.argmax(-1)).float().mean().item()
        per.append({"window": num, "positions": int(npos),
                    "mean_kld_his_vs_ours": float(kl.mean()),
                    "max_kld": float(kl.max()), "top1_agreement": top1})
        print("window", num, per[-1])
        del hidden, his, ours, theirs, kl
    if not per:
        raise SystemExit("no windows compared")
    receipt = {
        "schema": "glm53flash-crosscheck/1",
        "direction": "KLD(brandonmusic_teacher || our_replay), nats",
        "windows": len(per),
        "positions": sum(p["positions"] for p in per),
        "mean_kld": sum(p["mean_kld_his_vs_ours"] * p["positions"] for p in per)
                    / sum(p["positions"] for p in per),
        "max_kld": max(p["max_kld"] for p in per),
        "top1_agreement": sum(p["top1_agreement"] * p["positions"] for p in per)
                          / sum(p["positions"] for p in per),
        "per_window": per,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2))
    print("crosscheck_done " + json.dumps({k: receipt[k] for k in
                                           ("windows", "positions", "mean_kld", "top1_agreement")}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("suite")
    s.add_argument("--logits", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_suite)
    c = sub.add_parser("compare")
    c.add_argument("--logits", required=True)
    c.add_argument("--suite", required=True)
    c.add_argument("--capture", required=True)
    c.add_argument("--head", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cuda")
    c.set_defaults(func=cmd_compare)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
