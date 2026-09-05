#!/usr/bin/env python3
"""The bf16 logit-rounding term, measured on the real GLM-5.3 root.

Hidden-form rows are scored on fp32 logits recomputed from sealed bf16 hidden
states (`logits' = float32(h_bf16) @ float32(W_bf16)^T`, `dscompare._replay`).
A bf16 serving stack computes the same product and then ROUNDS EVERY LOGIT TO
BF16 before its softmax (ULP 0.125 at |logit| in [16, 32)).  This script
measures what that rounding does to the estimator, on the published root
capture and on real candidate captures, with the comparator's own replay and
fp64 estimator -- imported, not re-implemented -- so the number is the number
the rows would carry.

Three quantities per window, all in nats per token, all fp64:

  one_sided   KL(fp32 || bf16(fp32)) on the root alone: the distance between
              the replayed distribution and the served-stack distribution of the
              SAME hidden states.
  two_sided   for each candidate: KL(bf16(ref) || bf16(cand)) - KL(ref || cand)
              -- the change in the published quantity if both sides had been
              logit-form captures from a bf16 stack.  This is the term the
              hidden-form rows do not contain.

Inputs are sealed fidelity datasets (spec: docs/FIDELITY-DATASET-SPEC.md).
Given `--root DIR` the local sealed copy is used after its checksums.txt is
re-verified; otherwise the needed files are fetched from
malaiwah/glm53-fidelity-root-v1 with huggingface_hub (one window ~25 MB, the
head 1.9 GB).  Candidates are optional and local only.

    python3 reports/bf16-logit-rounding/measure.py \
        --root /path/to/glm53-fidelity-root-v1 --windows 0 \
        --candidate k4=/path/to/k4-dataset --candidate fp8=/path/to/fp8-dataset \
        --out reports/bf16-logit-rounding/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "bin"))

from fidelity import dscompare  # noqa: E402
from fidelity import dsformat as F  # noqa: E402
from fidelity import dsvalidate  # noqa: E402

ROOT_REPO = "malaiwah/glm53-fidelity-root-v1"
VOCAB_CHUNK = 8192


def round_to_bf16(x32: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16, returned widened to fp32.

    The torch `.to(torch.bfloat16)` rounding, as bit arithmetic: add
    0x7FFF plus the current LSB of the kept half to the uint32 pattern, then
    drop the low 16 bits.  Finite inputs only (the estimator refuses non-finite
    logits anyway).
    """
    bits = np.ascontiguousarray(x32, dtype="<f4").view("<u4")
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view("<f4")


def fetch_root(cache_dir: str, windows: list) -> str:
    from huggingface_hub import hf_hub_download  # noqa: WPS433

    files = ["fidelity-dataset.json", "checksums.txt", "capture/manifest.json",
             "panel/panel.json", "head/head.json", "head/weight.safetensors",
             "runtime/capture-runtime.json"]
    files += ["capture/hidden_%04d.safetensors" % w for w in windows]
    for rel in files:
        hf_hub_download(ROOT_REPO, rel, repo_type="dataset", local_dir=cache_dir)
    return cache_dir


def load_side(root: str, allow_partial: bool):
    report = dsvalidate.validate_dataset(root, verify_tensors=False, allow_partial=allow_partial)
    if report.errors:
        raise SystemExit("REFUSED: %s does not verify: %s" % (root, report.errors[0]["message"]))
    return dscompare.load_dataset(root, verify=False)


def head_t(dataset) -> np.ndarray:
    return np.ascontiguousarray(
        dscompare.load_tensor(dataset.head_path(), dataset.head["tensor_key"]).T)


def kld(a32: np.ndarray, b32: np.ndarray):
    values, matches, backend = dscompare.token_kld(a32, b32, "cpu")
    return values.astype(np.float64), int(matches), backend


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", help="local sealed root dataset (else fetched)")
    parser.add_argument("--cache", default=os.path.join(HERE, "cache"),
                        help="download directory when --root is not given")
    parser.add_argument("--windows", default="0",
                        help="comma-separated record indices, or 'all'")
    parser.add_argument("--candidate", action="append", default=[],
                        metavar="LABEL=DIR", help="local sealed candidate dataset(s)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    windows = None if args.windows == "all" else [int(w) for w in args.windows.split(",")]
    root_dir = args.root or fetch_root(args.cache, windows or [0])
    reference = load_side(root_dir, allow_partial=args.root is None)
    if windows is None:
        windows = [int(r["index"]) for r in reference.records]
    candidates = []
    for item in args.candidate:
        label, _, path = item.partition("=")
        candidates.append((label, load_side(path, allow_partial=False)))

    ref_records = {int(r["index"]): r for r in reference.records}
    ref_head = head_t(reference)
    cand_heads = {label: head_t(ds) for label, ds in candidates}
    cand_records = {label: {int(r["index"]): r for r in ds.records} for label, ds in candidates}

    per_window = []
    started = time.time()
    for index in windows:
        rec = ref_records[index]
        hidden = dscompare.load_tensor(reference.record_path(rec), rec["key"])
        ref32 = dscompare._replay(hidden, ref_head, VOCAB_CHUNK)
        ref_b = round_to_bf16(ref32)
        one, top1_same, backend = kld(ref32, ref_b)
        rev, _, _ = kld(ref_b, ref32)
        delta = np.abs(ref32.astype(np.float64) - ref_b.astype(np.float64))
        row = {
            "index": index,
            "window_id": rec.get("window_id"),
            "allocation_stratum": rec.get("allocation_stratum"),
            "positions": int(ref32.shape[0]),
            "estimator_backend": backend,
            "logit_abs_max": float(np.abs(ref32).max()),
            "logit_rounding_abs_max": float(delta.max()),
            "logit_rounding_abs_mean": float(delta.mean()),
            "one_sided": {
                "kl_fp32_vs_bf16_mean": float(one.mean()),
                "kl_fp32_vs_bf16_max": float(one.max()),
                "kl_bf16_vs_fp32_mean": float(rev.mean()),
                "top1_agreement": top1_same / ref32.shape[0],
            },
            "two_sided": {},
        }
        for label, ds in candidates:
            crec = cand_records[label][index]
            chid = dscompare.load_tensor(ds.record_path(crec), crec["key"])
            cand32 = dscompare._replay(chid, cand_heads[label], VOCAB_CHUNK)
            base, base_top1, _ = kld(ref32, cand32)
            both, both_top1, _ = kld(ref_b, round_to_bf16(cand32))
            row["two_sided"][label] = {
                "kl_fp32_mean": float(base.mean()),
                "kl_bf16_both_sides_mean": float(both.mean()),
                "delta_mean": float(both.mean() - base.mean()),
                "delta_relative": float((both.mean() - base.mean()) / base.mean()),
                "delta_abs_max_per_token": float(np.abs(both - base).max()),
                "top1_agreement_fp32": base_top1 / ref32.shape[0],
                "top1_agreement_bf16": both_top1 / ref32.shape[0],
            }
            del chid, cand32
        per_window.append(row)
        print("window %d (%s): one-sided %.3e nats, |logit| max %.2f, rounding max %.4f%s"
              % (index, row["window_id"], row["one_sided"]["kl_fp32_vs_bf16_mean"],
                 row["logit_abs_max"], row["logit_rounding_abs_max"],
                 "".join("; %s two-sided delta %+.3e (%+.2f%%)"
                         % (label, v["delta_mean"], 100 * v["delta_relative"])
                         for label, v in row["two_sided"].items())))
        del hidden, ref32, ref_b

    def mean_of(key_fn):
        return float(np.mean([key_fn(r) for r in per_window]))

    summary = {
        "windows": len(per_window),
        "one_sided_kl_fp32_vs_bf16_mean": mean_of(lambda r: r["one_sided"]["kl_fp32_vs_bf16_mean"]),
        "one_sided_kl_fp32_vs_bf16_window_min": float(min(
            r["one_sided"]["kl_fp32_vs_bf16_mean"] for r in per_window)),
        "one_sided_kl_fp32_vs_bf16_window_max": float(max(
            r["one_sided"]["kl_fp32_vs_bf16_mean"] for r in per_window)),
        "logit_abs_max": float(max(r["logit_abs_max"] for r in per_window)),
        "logit_rounding_abs_max": float(max(r["logit_rounding_abs_max"] for r in per_window)),
        "two_sided": {
            label: {
                "kl_fp32_mean": mean_of(lambda r: r["two_sided"][label]["kl_fp32_mean"]),
                "kl_bf16_both_sides_mean": mean_of(
                    lambda r: r["two_sided"][label]["kl_bf16_both_sides_mean"]),
                "delta_mean": mean_of(lambda r: r["two_sided"][label]["delta_mean"]),
            } for label, _ in candidates
        },
    }
    for label in summary["two_sided"]:
        block = summary["two_sided"][label]
        block["delta_relative"] = block["delta_mean"] / block["kl_fp32_mean"]
    result = {
        "schema": "malaiwah.bf16-logit-rounding-term.v1",
        "method": {
            "replay": "dscompare._replay: float32(h_bf16) @ float32(W_bf16)^T, numpy, vocab_chunk %d"
                      % VOCAB_CHUNK,
            "rounding": "round-to-nearest-even fp32 -> bf16 -> fp32 (reports/bf16-logit-rounding/"
                        "measure.py::round_to_bf16), applied to every logit",
            "estimator": "dscompare.token_kld (kld_report._token_kld when torch is importable, "
                         "else the identical numpy fp64 formula), full vocabulary, fp64",
            "replay_env": dscompare._numpy_replay_env(),
        },
        "reference": {
            "dataset_id": (reference.manifest.get("dataset") or {}).get("id"),
            "repository": (reference.manifest.get("dataset") or {}).get("repository"),
            "dataset_sha256": reference.manifest[F.SEAL_FIELD],
            "capture_content_digest": reference.content_digest,
            "head_tensor_content_sha256": reference.head.get("tensor_content_sha256"),
        },
        "candidates": {
            label: {
                "dataset_id": (ds.manifest.get("dataset") or {}).get("id"),
                "dataset_sha256": ds.manifest[F.SEAL_FIELD],
                "capture_content_digest": ds.content_digest,
                "head_tensor_content_sha256": ds.head.get("tensor_content_sha256"),
            } for label, ds in candidates
        },
        "summary": summary,
        "per_window": per_window,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    F.write_json(args.out, result)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
