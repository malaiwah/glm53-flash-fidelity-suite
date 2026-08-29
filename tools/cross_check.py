#!/usr/bin/env python3
"""Cross-validate our hidden-state replay against brandonmusic's independent
GLM-5.3-Flash-BF16-Teacher-Logits capture (v2: manifest-driven, hash-verified).

Dataset layout (from its dataset-manifest.json): 25 'final' windows, each
logits/window-NNNN.safetensors holding key 'logits' F32 [2047, 154880] with
metadata {window_id, token_ids_sha256, model_revision}; token rows live at
calibration/panel-v1/arrays/<window_id>.tokens.npy (int32/int64, shape (2048,)).
Pairing is verified against token_ids_sha256 before anything is captured.

    cross_check.py suite    --logits DIR --out SUITE_DIR
    cross_check.py compare  --logits DIR --suite SUITE_DIR --capture CAP_DIR \
                            --head HEAD --out receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_stackprint():
    """bin/fidelity/stackprint.py by path (repo checkout or VM bundle); a
    receipt without a stack fingerprint is refusable, so failure refuses."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "bin" / "fidelity" / "stackprint.py"
    try:
        spec = importlib.util.spec_from_file_location("glm53_stackprint", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise SystemExit(
            f"stack fingerprint module unavailable ({exc}) at {path}; "
            "re-run make_bundle.sh so bin/fidelity/stackprint.py ships next to tools/"
        )


def load_manifest(logits_dir: Path) -> list[dict]:
    man = json.loads((logits_dir / "dataset-manifest.json").read_text())
    return man["logit_files"]


def token_file_for(logits_dir: Path, window_id: str) -> Path:
    return logits_dir / "calibration" / "panel-v1" / "arrays" / f"{window_id}.tokens.npy"


def verified_ids(logits_dir: Path, entry: dict):
    """Token ids for one window, verified against the manifest hash. Returns
    (ids, verification_mode) or raises."""
    import numpy as np

    tf = token_file_for(logits_dir, entry["window_id"])
    if not tf.is_file():
        raise FileNotFoundError(f"token file missing: {tf}")
    arr = np.load(tf)
    want = entry["token_ids_sha256"]
    candidates = {
        "file_bytes": sha256_bytes(tf.read_bytes()),
        "array_bytes": sha256_bytes(arr.tobytes()),
        "array_int32": sha256_bytes(arr.astype(np.int32).tobytes()),
        "array_int64": sha256_bytes(arr.astype(np.int64).tobytes()),
        "json_ids": sha256_bytes(json.dumps([int(x) for x in arr]).encode()),
    }
    for mode, digest in candidates.items():
        if digest == want:
            return [int(x) for x in arr], mode
    raise SystemExit(
        f"token hash verification FAILED for {entry['window_id']}: none of "
        f"{list(candidates)} matches manifest token_ids_sha256")


def cmd_suite(args) -> int:
    logits_dir = Path(args.logits)
    entries = [e for e in load_manifest(logits_dir) if e.get("role") == "final"]
    if not entries:
        raise SystemExit("no role=final windows in dataset manifest")
    out = Path(args.out)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    contexts, modes = [], set()
    for i, entry in enumerate(sorted(entries, key=lambda e: e["window_id"])):
        ids, mode = verified_ids(logits_dir, entry)
        modes.add(mode)
        name = f"context-{i:04d}.json"
        (out / "tokens" / name).write_text(json.dumps(ids))
        contexts.append({"index": i, "stratum": entry.get("domain", "bm-teacher"),
                         "file": f"tokens/{name}",
                         "token_sha256": sha256_bytes(json.dumps(ids).encode()),
                         "tokens": len(ids),
                         "source_cluster": entry["window_id"],
                         "logits_path": entry["path"],
                         "manifest_token_ids_sha256": entry["token_ids_sha256"]})
    manifest = {
        "schema": "glm53flash-crosscheck-suite/2",
        "source_dataset": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
        "token_hash_verification": sorted(modes),
        "context_length": contexts[0]["tokens"],
        "scored_positions_per_context": contexts[0]["tokens"] - 1,
        "contexts": len(contexts),
        "context_index": contexts,
    }
    manifest["suite_token_sha256"] = sha256_bytes(
        "".join(c["token_sha256"] for c in contexts).encode())
    (out / "suite-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"contexts": len(contexts),
                      "verification": sorted(modes),
                      "suite_token_sha256": manifest["suite_token_sha256"][:16]}))
    return 0


def agreement_at(theirs, ours, offset: int):
    """Mean KLD + top1 comparing his positions p to our positions p+offset."""
    import torch

    n = theirs.shape[0]
    if offset >= 0:
        a, b = theirs[: n - offset], ours[offset:]
    else:
        a, b = theirs[-offset:], ours[: n + offset]
    kl = (a.exp() * (a - b)).sum(-1).double()
    top = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return float(kl.mean()), top


def cmd_compare(args) -> int:
    import torch
    from safetensors import safe_open

    logits_dir = Path(args.logits)
    cap = Path(args.capture)
    dev = torch.device(args.device)
    with safe_open(args.head, framework="pt", device="cpu") as f:
        key = "weight" if "weight" in f.keys() else f.keys()[0]
        head = f.get_tensor(key).to(dev, torch.bfloat16)
    suite = json.loads(Path(args.suite, "suite-manifest.json").read_text())
    per, offset_audit = [], {"-1": [], "0": [], "+1": []}
    for ctx in suite["context_index"]:
        hf = cap / f"hidden_{ctx['index']:04d}.safetensors"
        wf = logits_dir / ctx["logits_path"]
        if not (hf.is_file() and wf.is_file()):
            print(f"skip {ctx['source_cluster']}: missing file")
            continue
        with safe_open(str(hf), framework="pt", device="cpu") as f:
            hidden = f.get_tensor("hidden_states").to(dev, torch.bfloat16)
        with safe_open(str(wf), framework="pt", device="cpu") as f:
            his = f.get_tensor("logits").to(dev, torch.float32)
        npos = min(hidden.shape[0], his.shape[0])
        ours = (hidden[:npos] @ head.T).float()
        ours = ours - ours.logsumexp(-1, keepdim=True)
        theirs = his[:npos] - his[:npos].logsumexp(-1, keepdim=True)
        for off, bucket in ((-1, "-1"), (0, "0"), (1, "+1")):
            bucket_kl, bucket_top = agreement_at(theirs, ours, off)
            offset_audit[bucket].append((bucket_kl, bucket_top))
        kl0, top0 = agreement_at(theirs, ours, 0)
        per.append({"window": ctx["source_cluster"], "positions": int(npos),
                    "mean_kld_his_vs_ours": kl0, "top1_agreement": top0})
        print(ctx["source_cluster"], per[-1])
        del hidden, his, ours, theirs
    if not per:
        raise SystemExit("no windows compared")
    total = sum(p["positions"] for p in per)
    # The 2026-08-28 review found these receipts carried ZERO digests -- no
    # link from the number to the capture, the head, or the stack that made
    # it.  Operands are now named by digest, and the comparator host records
    # its own fingerprint (engine kind "none": this process serves nothing).
    manifest_path = cap / "capture-manifest.json"
    capture_manifest = (json.loads(manifest_path.read_text())
                        if manifest_path.is_file() else {})
    stackprint = load_stackprint()
    own_fp = stackprint.public_dict(stackprint.collect("none"))
    receipt = {
        "schema": "glm53flash-crosscheck/2",
        "direction": "KLD(brandonmusic_teacher || our_replay), nats",
        "capture_manifest_sha256": (sha256_file(manifest_path)
                                    if manifest_path.is_file() else None),
        "capture_stack_fingerprint_sha256":
            capture_manifest.get("stack_fingerprint_sha256"),
        "head_sha256": sha256_file(Path(args.head)),
        "stack_fingerprint": own_fp,
        "stack_fingerprint_sha256": stackprint.fingerprint_sha256(own_fp),
        "their_model_revision_note": "his metadata records an earlier repo revision; "
                                     "weights were never modified post-upload (config/template churn only)",
        "windows": len(per),
        "positions": total,
        "mean_kld": sum(p["mean_kld_his_vs_ours"] * p["positions"] for p in per) / total,
        "top1_agreement": sum(p["top1_agreement"] * p["positions"] for p in per) / total,
        "offset_audit_mean_top1": {k: (sum(t for _, t in v) / len(v) if v else None)
                                   for k, v in offset_audit.items()},
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
