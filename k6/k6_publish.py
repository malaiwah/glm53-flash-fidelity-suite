#!/usr/bin/env python3
"""
NAMING (operator + turboderp guidance 2026-08-28): repos use TR3 (the codec
family), NEVER 'EXL3' — these are not loadable by stock exllamav3 (no
glm5_next arch); cards must state codec vs runtime distinction explicitly.
HF publication for the K6/K6K8 campaign: weights, receipts, discussion draft.

Subcommands (interface pinned by stage_k6.sh):

  weights   --checkpoint --repo --recipe --receipts --card
            Stages quantization/recipe.json, provenance, the publication-gate
            receipt, README card and the MANIFEST.json + SHA256SUMS closed tree
            into the materialized checkpoint, then uploads with
            huggingface_hub.upload_large_folder (checkpointed, resumable).

  receipts  --receipts --repo --prefix --discussion-draft
            Uploads the receipt tree (json/md/npy) into the fidelity dataset
            under the given prefix and writes the discussion-comment draft
            (posted by the USER, never by automation).

Publication policy enforced here (release contract):
  * weights upload refuses to run unless the profile's packed-KLD receipt is
    green (QP_PUBLISH_UNQUALIFIED=1 is the explicit operator override for the
    disclosed failure-analysis path).
  * README cards must credit brandonmusic (pipeline, recipe, teacher logits,
    calibration captures, sealed-window protocol) and zai-org (source model),
    and carry the upstream model license.  A missing card file is an error
    unless --allow-default-card plus --license-id are given.
  * HF token comes from the environment (HF_TOKEN); it is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
SOURCE_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
UPSTREAM_CREDITS = {
    "pipeline": "github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw",
    "teacher_logits": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
    "reference_quant": "brandonmusic GLM-5.3-Flash EXL3/TR3-MCG 4bpw",
}


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"k6_publish: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".new-{os.getpid()}")
    staging.write_bytes(data)
    os.replace(staging, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(value: Dict[str, Any], field: str) -> Dict[str, Any]:
    body = dict(value)
    body[field] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _require_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise _fail(
            "HF_TOKEN is not set - the stage script exports it from "
            "/home/jl_fs/glm53-k6/.hf_token (never echoed)"
        )
    return token


def _profile_from_recipe(recipe: Dict[str, Any]) -> str:
    profile = str(recipe.get("profile", ""))
    if profile.startswith("k8"):
        return "k8"
    if profile.startswith("k6k8"):
        return "k6k8"
    if profile.startswith("k6"):
        return "k6"
    raise _fail(f"recipe declares no recognizable profile: {profile!r}")


def _default_card(
    *,
    repo: str,
    profile: str,
    recipe: Dict[str, Any],
    materialization: Dict[str, Any],
    packed_kld: Optional[Dict[str, Any]],
    license_id: str,
) -> str:
    bits_desc = {
        "k6": "uniform 6-bit routed experts (K6)",
        "k8": "uniform 8-bit routed experts (K8, malaiwah declared extension)",
        "k6k8": "mixed routed experts: gate/up 6-bit, down 8-bit (K6K8)",
    }[profile]
    mean = packed_kld.get("measured_mean_kld") if packed_kld else None
    mean_row = f"{mean:.6f}" if isinstance(mean, (int, float)) else "see receipts/"
    lines = [
        "---",
        f"license: {license_id}",
        f"base_model: {SOURCE_MODEL_ID}",
        "tags:",
        "- exl3",
        "- tr3-mcg",
        "- quantized",
        f"- glm-5.3-flash-{profile}",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        f"Quantized derivative of **{SOURCE_MODEL_ID} @ {SOURCE_REVISION[:8]}** "
        f"({bits_desc}, EXL3/TR3-MCG, topology-neutral canonical tensors).",
        "",
        "## Credits",
        "",
        "* **brandonmusic** - quantization pipeline, recipe, fp32 teacher logits, "
        "calibration captures and the sealed 25-window protocol "
        f"(`{UPSTREAM_CREDITS['teacher_logits']}`, "
        f"`{UPSTREAM_CREDITS['pipeline']}`).  This repo is an independent "
        "reproduction at a new rate using his published evidence chain.",
        "* **zai-org** - the GLM-5.3-Flash base model.  The license of this repo "
        "is inherited verbatim from the source model repository.",
        "* Our contribution: the K6/K6K8 rates, the 4x H200 port (see the "
        "deviations register in `receipts/`), and the receipt chain.",
        "",
        "## Fidelity (mean tokenwise KLD vs the fp32 BF16 teacher, fp64, "
        "25 sealed windows / 51,175 positions)",
        "",
        "| model | mean KLD |",
        "|---|---|",
        "| zai-org FP8 (as served) | 0.020615 |",
        "| brandonmusic K4 | 0.024555 |",
        f"| this repo ({profile.upper()}) | {mean_row} |",
        "",
        "## Disclosed deviations from the upstream sealed campaign",
        "",
        "1. Workers were 4x H200 SM90 (schema string kept verbatim; actual "
        "devices recorded truthfully in the launch plan's hardware attestation).",
        "2. ExLlamaV3 extension built with `TORCH_CUDA_ARCH_LIST=\"9.0;10.0\"` "
        "(genuinely carries SM100 code objects; executed on SM90).",
        "3. Student KLD capture ran EP8 (identical logits; install exact under "
        "any divisor of 288).",
        "4. Hessian artifacts pruned after each layer receipt sealed.",
        "5. Fresh transform seed, sealed in-repo (first-ever K6; determinism "
        "established by our own five-cold-run receipt).",
        "",
        "## Topology neutrality",
        "",
        "The checkpoint stores canonical unsharded tensors; no TP/EP topology is "
        "baked in (`qualified_tp_sizes []`, `serving_reader_qualified false` in "
        "the materialization receipt are the storage-honesty markers, not a "
        "defect).  TP packing happens at load time.",
        "",
        "## Receipts",
        "",
        "`receipts/checkpoint.json` seals the publication gate; the full chain "
        "(contract, per-layer receipts, KLD, five-run determinism, runtime "
        "qualification) lives in the fidelity dataset "
        "`malaiwah/GLM-5.3-Flash-fidelity-suite-v1` under `reports/exl3-k6/`.",
        "",
        f"Recipe: `quantization/recipe.json` (id `{recipe.get('recipe_id', 'see file')}`), "
        f"materialized bytes: {materialization.get('output_logical_bytes')}.",
    ]
    return "\n".join(lines) + "\n"


def _closed_tree(checkpoint: Path) -> "tuple[Dict[str, Any], str]":
    """MANIFEST.json + SHA256SUMS over every file except the two closures."""

    excluded = {"MANIFEST.json", "SHA256SUMS"}
    rows: List[Dict[str, Any]] = []
    sums: List[str] = []
    for path in sorted(checkpoint.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name.startswith("."):
            continue
        relative = path.relative_to(checkpoint).as_posix()
        digest = _sha256_file(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        sums.append(f"{digest}  {relative}")
        print(f"  manifest: {relative}", flush=True)
    manifest = _seal(
        {
            "schema": "malaiwah.glm53-k6-weights-manifest.v1",
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "files": rows,
        },
        "manifest_sha256",
    )
    return manifest, "\n".join(sums) + "\n"


def cmd_weights(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.resolve()
    receipts_dir = args.receipts.resolve()
    recipe = _read_json(args.recipe.resolve(), "recipe")
    profile = _profile_from_recipe(recipe)
    materialization = _read_json(
        checkpoint / "materialization-receipt.json", "materialization receipt"
    )
    if materialization.get("complete") is not True:
        raise _fail("materialization receipt is not complete - refusing to publish")

    packed_kld_path = receipts_dir / f"{profile}-packed-kld.json"
    packed_kld: Optional[Dict[str, Any]] = None
    if packed_kld_path.is_file():
        packed_kld = _read_json(packed_kld_path, "packed KLD receipt")
    unqualified_override = os.environ.get("QP_PUBLISH_UNQUALIFIED", "0") == "1"
    if not unqualified_override:
        if packed_kld is None:
            raise _fail(
                f"{packed_kld_path} absent - run qualify first, or set "
                "QP_PUBLISH_UNQUALIFIED=1 for the disclosed failure-analysis path",
                code=5,
            )
        if packed_kld.get("quality_gate_passed") is not True:
            raise _fail(
                "packed-KLD quality gate is RED - weights publish only as "
                "receipts + failure analysis (QP_PUBLISH_UNQUALIFIED=1)",
                code=5,
            )

    # --- stage publication files into the checkpoint tree -------------------
    _atomic_json(checkpoint / "quantization" / "recipe.json", recipe)
    _atomic_json(
        checkpoint / "provenance" / "source-model-revision.json",
        _seal(
            {
                "schema": "malaiwah.glm53-source-model-revision.v1",
                "source_model_id": SOURCE_MODEL_ID,
                "model_revision": SOURCE_REVISION,
                "weight_dtype": "bfloat16",
                "inventory_sha256": materialization.get("source_inventory_sha256"),
            },
            "receipt_sha256",
        ),
    )
    gate_evidence: Dict[str, Any] = {
        "materialization_receipt_sha256": materialization.get("receipt_sha256"),
        "materialization_receipt_file_sha256": _sha256_file(
            checkpoint / "materialization-receipt.json"
        ),
    }
    for name in (
        f"{profile}-packed-kld.json",
        f"{profile}-five-run-kld.json",
        f"{profile}-tp4-runtime-receipt.json",
    ):
        path = receipts_dir / name
        if path.is_file():
            gate_evidence[name.replace("-", "_").replace(".json", "_file_sha256")] = (
                _sha256_file(path)
            )
    gate = _seal(
        {
            # malaiwah namespace for BOTH profiles: upstream defines no
            # checkpoint-publication-gate schema, and his namespace is never
            # squatted for documents his verifiers did not define (DECISIONS #6)
            "schema": f"malaiwah.glm53-{profile}-checkpoint-publication-gate.v1",
            "profile": profile,
            "qualified": bool(packed_kld and packed_kld.get("quality_gate_passed")),
            "published_unqualified_failure_analysis": unqualified_override
            and not (packed_kld and packed_kld.get("quality_gate_passed")),
            "measured_mean_kld": (packed_kld or {}).get("measured_mean_kld"),
            "quality_gate": {"metric": "mean_tokenwise_kld", "threshold_lt": 0.06},
            "evidence": gate_evidence,
            "published_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "receipt_sha256",
    )
    _atomic_json(checkpoint / "receipts" / "checkpoint.json", gate)

    card_path = args.card.resolve() if args.card else None
    if card_path is not None and card_path.is_file():
        card_text = card_path.read_text(encoding="utf-8")
    elif args.allow_default_card:
        if not args.license_id:
            raise _fail(
                "--allow-default-card requires --license-id (read the license off "
                "the pinned source revision; do not guess)"
            )
        card_text = _default_card(
            repo=args.repo,
            profile=profile,
            recipe=recipe,
            materialization=materialization,
            packed_kld=packed_kld,
            license_id=args.license_id,
        )
    else:
        raise _fail(
            f"README card missing: {card_path} - author it (crediting brandonmusic "
            "and zai-org, license inherited verbatim from the source repo) or pass "
            "--allow-default-card --license-id <spdx>"
        )
    for needle in ("brandonmusic", "zai"):
        if needle.lower() not in card_text.lower():
            raise _fail(f"README card does not credit '{needle}' - release contract")
    _atomic_write(checkpoint / "README.md", card_text.encode("utf-8"))

    print("building MANIFEST.json + SHA256SUMS closed tree ...")
    manifest, sums = _closed_tree(checkpoint)
    _atomic_json(checkpoint / "MANIFEST.json", manifest)
    _atomic_write(checkpoint / "SHA256SUMS", sums.encode("utf-8"))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "repo": args.repo,
                          "file_count": manifest["file_count"],
                          "total_bytes": manifest["total_bytes"]}, sort_keys=True))
        return 0

    token = _require_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"uploading {checkpoint} -> {args.repo} (upload_large_folder, resumable) ...")
    api.upload_large_folder(
        repo_id=args.repo, folder_path=str(checkpoint), repo_type="model"
    )
    _atomic_json(
        receipts_dir / f"publish-weights-{profile}.json",
        _seal(
            {
                "schema": "malaiwah.glm53-k6-publish-receipt.v1",
                "repo": args.repo,
                "repo_url": f"https://huggingface.co/{args.repo}",
                "manifest_sha256": manifest["manifest_sha256"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "gate_receipt_sha256": gate["receipt_sha256"],
                "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "receipt_sha256",
        ),
    )
    print(json.dumps({"ok": True, "repo": args.repo}, sort_keys=True))
    return 0


def _discussion_draft(receipts_dir: Path, out_path: Path) -> None:
    if out_path.is_file():
        return
    packed = receipts_dir / "k6-packed-kld.json"
    mean = "<pending>"
    if packed.is_file():
        value = json.loads(packed.read_text(encoding="utf-8")).get("measured_mean_kld")
        if isinstance(value, (int, float)):
            mean = f"{value:.6f}"
    text = (
        "<!-- DRAFT: posted by the USER on brandonmusic's teacher-logits dataset; "
        "never posted by automation -->\n\n"
        "Using your published pipeline, calibration captures and sealed final "
        "windows, we produced the first K6 (and, gated, K6K8) EXL3/TR3-MCG quants "
        "of GLM-5.3-Flash on 4x H200 (deviations disclosed in-repo: H200 workers, "
        f"EP8 student capture, fresh transform seed).  K6 scored mean tokenwise "
        f"KLD {mean} vs your fp32 teacher on the same 25 windows "
        "(your K4: 0.024555, FP8: 0.020615).  Weights + full receipt chain: "
        "https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw and "
        "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1 "
        "(reports/exl3-k6/).  Thank you for publishing the entire evidence chain - "
        "it made an independent reproduction at a new rate possible.\n"
    )
    _atomic_write(out_path, text.encode("utf-8"))
    print(f"discussion draft written: {out_path}")


def cmd_receipts(args: argparse.Namespace) -> int:
    receipts_dir = args.receipts.resolve()
    if not receipts_dir.is_dir():
        raise _fail(f"receipts dir missing: {receipts_dir}")
    if args.discussion_draft:
        _discussion_draft(receipts_dir, args.discussion_draft.resolve())
    if args.dry_run:
        count = sum(
            1
            for path in receipts_dir.rglob("*")
            if path.is_file() and path.suffix in {".json", ".md", ".npy", ".txt"}
        )
        print(json.dumps({"dry_run": True, "repo": args.repo, "files": count},
                         sort_keys=True))
        return 0
    token = _require_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
    print(f"uploading receipts {receipts_dir} -> {args.repo}/{args.prefix} ...")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(receipts_dir),
        path_in_repo=args.prefix,
        allow_patterns=["**/*.json", "**/*.md", "**/*.npy", "**/*.txt",
                        "*.json", "*.md", "*.npy", "*.txt"],
        ignore_patterns=["**/.lock", "**/*.new-*"],
        commit_message="K6 campaign receipts (packed-kld, five-run, runtime, deviations)",
    )
    print(json.dumps({"ok": True, "repo": args.repo, "prefix": args.prefix},
                     sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k6_publish.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("weights", help="publish a materialized checkpoint to HF")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--card", type=Path, help="README card (authored by the operator)")
    p.add_argument("--allow-default-card", action="store_true")
    p.add_argument("--license-id", help="SPDX id read off the pinned source revision")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_weights)

    p = sub.add_parser("receipts", help="publish receipts into the fidelity dataset")
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--discussion-draft", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_receipts)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
