#!/usr/bin/env python3
"""GLM-5.3-Flash K6/K8/K6K8 EXL3-MCG encode-campaign driver (malaiwah).

Interface pinned by stage_k6.sh.  Subcommands:

    rehearse             P0 fixture/codec roundtrip + K8 probe + timing bench
    contract             sealed inventory -> preflight -> launch plan ->
                         profile selection -> layer preparations ->
                         campaign preparation receipt -> direct contract ->
                         work units + hash-chained work state
    prepare              (auxiliary) build layer preparations for a layer subset
                         so the GSS pass can be parallelized across GPUs
    encode-worker        claim -> encode_work_unit -> seal_layer -> prune hessians
    seal-main            author the driver-side sealed main receipt
    release-dead-claims  requeue claims of dead workers
    mtp                  MTP45 contract -> encode -> telemetry -> adapter seal
    materialize          reader-ABI receipt -> materialization plan -> shards
    shared-vector-ab     operator directive 2: down_suh shared-vs-private A/B

Design invariants honoured throughout:
  * receipts-over-exit-codes: every step writes a sealed JSON receipt and
    re-verifies it on resume; nothing is trusted from process memory.
  * resumability: per-expert receipts (native to encode_work_unit), idempotent
    per-layer preparation, layer receipts as the layer-done marker, per-shard
    materialization receipts, hash-chained work state guarded by an fcntl lock.
  * hessian prune (disclosed deviation 4): after a layer receipt seals, its
    Hessian artifacts are deleted; sealed layers are never re-entered.
  * lazy heavy imports: torch / the pinned pipeline / the r10 codec closure are
    imported only inside the subcommands that need them, so --help and offline
    validation work anywhere.  A missing r7_encoder/r10_codec.py produces an
    actionable error pointing at closure_status.json (upstream github issue #1).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

MODEL_REVISION_DEFAULT = "a6c167b62691b2bac901344b65cb651a70f53e43"
UPSTREAM_INVENTORY_SHA_PREFIX = "f56e9d6250e2d108"  # cross-check target (RUNBOOK P1.1)
SHAPLEY_REVISION = "9d83e7d0baea86604d604502f0d5456c2906486b"
RUN_QWEN_FAST_ENCODE_SHA = (
    "ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492"
)
PROFILE_BITS = {"k6": 6, "k8": 8}  # k8: malaiwah K8-uniform (DECISIONS.md 7)
CLOSURE_HELP = (
    "the ShapleyMCG r7_encoder numeric closure (r7_encoder/r10_codec.py and its "
    "package) is missing from --shapley-root.  It was requested upstream "
    "(github.com/brandonmmusic-max/... issue #1) and MUST be present before any "
    "encode.  See closure_status.json next to this campaign's receipts."
)
# Marker string embedded in the reconstructed fallback codec (fallback/
# r10_codec_reconstructed.py); its presence in r10_codec.py means the closure
# is OUR disclosed reconstruction, not Brandon's sealed implementation.
RECONSTRUCTION_MARKER = "k6-program.r10-fallback-reconstruction.v1"
RECONSTRUCTION_ACCEPTANCE_HELP = (
    "r7_encoder/r10_codec.py under --shapley-root is the RECONSTRUCTED fallback "
    "codec, but no operator acceptance was found.  The disclosed-reconstruction "
    "path requires an explicit operator decision (RUNBOOK G0 item 1): create "
    "RECONSTRUCTION-ACCEPTED.json next to the shapley root (i.e. in the campaign "
    "ROOT) with {\"accept_reconstructed_r10_codec\": true, \"operator\": ..., "
    "\"date\": ...}, or export QP_ACCEPT_RECONSTRUCTED_CLOSURE=1.  Never encode "
    "through the reconstruction silently."
)


# --------------------------------------------------------------------------- #
# generic helpers (no heavy imports)                                          #
# --------------------------------------------------------------------------- #

def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"k6_driver: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise _fail(f"no quant_pipeline package under {pipeline_root}")


def _import_pipeline(pipeline_root: Path) -> None:
    src = str(_pipeline_src(pipeline_root))
    if src not in sys.path:
        sys.path.insert(0, src)


def _closure_source(shapley_root: Path) -> Optional[str]:
    """None (absent), "upstream" (Brandon's files), or "reconstruction"."""

    codec = shapley_root / "r7_encoder" / "r10_codec.py"
    if not codec.is_file():
        return None
    try:
        text = codec.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return "reconstruction" if RECONSTRUCTION_MARKER in text else "upstream"


def _reconstruction_accepted(shapley_root: Path) -> bool:
    if os.environ.get("QP_ACCEPT_RECONSTRUCTED_CLOSURE") == "1":
        return True
    acceptance = shapley_root.parent / "RECONSTRUCTION-ACCEPTED.json"
    if not acceptance.is_file():
        return False
    try:
        value = json.loads(acceptance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("accept_reconstructed_r10_codec") is True


def _write_closure_status(
    root: Path, shapley_root: Path, source: Optional[str]
) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        status = {
            "schema": "malaiwah.glm53-k6-shapleymcg-closure-status.v1",
            "shapley_root": str(shapley_root),
            "required": [
                "r7_encoder/r10_codec.py",
                "r7_encoder/trellis.py (CodecConfig)",
                "encode_tr3_v31.py (referenced by the published receipts)",
            ],
            "r10_codec_present": source is not None,
            "closure_source": source,  # "upstream" | "reconstruction" | null
            "reconstruction_accepted": (
                _reconstruction_accepted(shapley_root)
                if source == "reconstruction"
                else None
            ),
            "upstream_request": "github issue #1 on the ShapleyMCG repository",
            "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (root / "closure_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def _require_codec_closure(shapley_root: Path, receipts_root: Path) -> None:
    source = _closure_source(shapley_root)
    _write_closure_status(receipts_root, shapley_root, source)
    if source is None:
        raise _fail(CLOSURE_HELP, code=6)
    if source == "reconstruction" and not _reconstruction_accepted(shapley_root):
        raise _fail(RECONSTRUCTION_ACCEPTANCE_HELP, code=6)


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".new-{os.getpid()}")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _find_extension(exllama_root: Path, override: Optional[str]) -> Path:
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise _fail(f"--extension does not exist: {path}")
        return path
    candidates = sorted(
        path
        for path in Path(exllama_root).rglob("*.so")
        if "ext" in path.name or "exllamav3" in path.name
    )
    if not candidates:
        candidates = sorted(Path(exllama_root).rglob("*.so"))
    if not candidates:
        # exllamav3 may have JIT-compiled into the torch_extensions cache
        # instead of leaving a .so in the source tree (EXLLAMA_NOCOMPILE /
        # editable-install layouts vary).
        cache = Path.home() / ".cache" / "torch_extensions"
        if cache.is_dir():
            candidates = sorted(cache.rglob("exllamav3_ext*.so"))
    if not candidates:
        raise _fail(
            f"no built extension (*.so) under {exllama_root} or the "
            "torch_extensions cache; run stage_k6.sh setup (in-place exllamav3 "
            "build) first, or pass --extension"
        )
    return candidates[0].resolve()


def _numeric_core(shapley_root: Path, override: Optional[str]) -> Path:
    """The numeric-core FILE handed to trellis.CodecConfig(numeric_core=...).

    This is NOT r10_codec.py: the codec loads this file as a module and
    fail-closes unless it exposes the v31 quantize surface (block_ldl, ldlq,
    pack_trellis, ...).  Upstream that file is encode_tr3_v31.py; the staged
    reconstruction ships the exllamav3-backed shim encode_tr3_fallback.py.
    """

    if override:
        return Path(override).resolve()
    package = shapley_root / "r7_encoder"
    for name in ("encode_tr3_v31.py", "encode_tr3_fallback.py"):
        candidate = package / name
        if candidate.is_file():
            return candidate.resolve()
    raise _fail(
        "no numeric core found under r7_encoder/ (need encode_tr3_v31.py "
        "[upstream] or encode_tr3_fallback.py [staged reconstruction shim]); "
        "pass --numeric-core explicitly.  " + CLOSURE_HELP,
        code=6,
    )


def _bits_for_profile(profile: str, output_root: Path) -> int:
    if profile in PROFILE_BITS:
        return PROFILE_BITS[profile]
    if profile == "k6k8":
        raise _fail(
            "profile k6k8 requires the malaiwah K6K8 support module "
            "(quant_pipeline.campaign.glm53_k6k8 per recipes/k6k8.json and RUNBOOK "
            "phase P2).  It is committed engineering, gated separately; this driver "
            "refuses to improvise per-projection rates through the sealed uniform "
            f"contracts.  Status file: {output_root / 'closure_status.json'}",
            code=7,
        )
    raise _fail(f"unknown profile: {profile}")


def _require_k8_seed(profile: str, output_root: Path) -> None:
    """K8 MUST reuse the K6 campaign transform seed (DECISIONS.md 7).

    The stage copies out-k6/transform-seed.json into the K8 output root before
    the contract step; a missing file here means a fresh seed would be minted,
    which would break assembly-compatibility with the K6 payload store - fail
    closed instead.
    """

    if profile == "k8" and not (output_root / "transform-seed.json").is_file():
        raise _fail(
            "profile k8 requires the K6 campaign transform seed at "
            f"{output_root / 'transform-seed.json'} (copy out-k6/"
            "transform-seed.json; NEVER mint a fresh seed for K8 - the K8 "
            "payload store must be assembly-compatible with K6)",
            code=9,
        )


# --------------------------------------------------------------------------- #
# sealed-document builders (driver-authored, verified by the pinned pipeline)  #
# --------------------------------------------------------------------------- #

def _seal(document: Dict[str, Any], field: str) -> Dict[str, Any]:
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes

    body = dict(document)
    body[field] = sha256_bytes(canonical_json(body))
    return body


def _build_inventory(
    bf16_root: Path, model_revision: str, out_path: Path, crosscheck_path: Path
) -> Dict[str, Any]:
    """Sealed quant-pipeline.glm-release-inventory.v1 over the full BF16 tree."""

    from safetensors import safe_open

    from quant_pipeline.campaign.glm53_uniform_k4 import (
        FIRST_MOE_LAYER,
        MAIN_LAYER_COUNT,
        MAIN_ROUTED_LAYERS,
        MTP_LAYERS,
        ROUTED_EXPERTS,
        _ROUTED,
        _inventory_surfaces,
    )
    from quant_pipeline.core.artifacts import sha256_file
    from quant_pipeline.normalization.artifact_v31 import tensor_sha256

    if out_path.is_file():
        inventory = _read_json(out_path, "inventory")
        _inventory_surfaces(inventory)  # re-verify the seal on resume
        print(f"inventory reused: {inventory['inventory_sha256']}")
        return inventory

    config_path = bf16_root / "config.json"
    index_path = bf16_root / "model.safetensors.index.json"
    for path in (config_path, index_path):
        if not path.is_file():
            raise _fail(f"BF16 checkpoint incomplete, missing {path}")
    index = _read_json(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise _fail("BF16 index has no weight_map")
    shards = sorted(set(weight_map.values()))
    print(f"hashing {len(shards)} BF16 shards (full-shard-sha256 seal mode) ...")
    shard_sha256: Dict[str, str] = {}
    for name in shards:
        shard_sha256[name] = sha256_file(bf16_root / name)
        print(f"  {name} {shard_sha256[name][:16]}", flush=True)

    tensors: List[Dict[str, Any]] = []
    dtype_names = {"torch.bfloat16": "BF16", "torch.float32": "F32", "torch.float16": "F16"}
    print("hashing per-tensor payloads (tensor_sha256 ABI) ...")
    for shard in shards:
        with safe_open(bf16_root / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                match = _ROUTED.fullmatch(name)
                if match is None:
                    scope = "native"
                else:
                    layer = int(match.group(1))
                    if layer in MAIN_ROUTED_LAYERS:
                        scope = "routed_expert"
                    elif layer in MTP_LAYERS:
                        scope = "mtp_routed_expert"
                    else:
                        scope = "native"
                tensors.append(
                    {
                        "tensor_name": name,
                        "scope": scope,
                        "dtype": dtype_names.get(str(value.dtype), str(value.dtype)),
                        "shape": [int(item) for item in value.shape],
                        "source_bytes": int(value.numel() * value.element_size()),
                        "source_payload_sha256": tensor_sha256(value),
                        "shard": shard,
                    }
                )
        print(f"  {shard}: cumulative {len(tensors)} tensors", flush=True)
    body = {
        "schema": "quant-pipeline.glm-release-inventory.v1",
        "seal_mode": "full-shard-sha256",
        "model_revision": model_revision,
        "checkpoint": str(bf16_root.resolve()),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shard_sha256": shard_sha256,
        "geometry": {
            "model_type": "glm5_next",
            "main_layers": MAIN_LAYER_COUNT,
            "mtp_layers": len(MTP_LAYERS),
            "first_moe_layer": FIRST_MOE_LAYER,
            "routed_experts": ROUTED_EXPERTS,
            "discovered_layers": list(range(MAIN_LAYER_COUNT + len(MTP_LAYERS))),
        },
        "tensors": tensors,
    }
    inventory = _seal(body, "inventory_sha256")
    _inventory_surfaces(inventory)  # fail-closed before persisting
    _atomic_json(out_path, inventory)
    # the shard hashes were just computed from disk: record the verification so
    # later Glm53BF16Source constructions skip the second 643 GB re-hash
    _atomic_json(
        out_path.with_name("inventory-shards-verified.json"),
        {
            "schema": "malaiwah.glm53-bf16-shards-verified.v1",
            "inventory_sha256": inventory["inventory_sha256"],
            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "full-shard sha256 computed during inventory build",
        },
    )
    matches = inventory["inventory_sha256"].startswith(UPSTREAM_INVENTORY_SHA_PREFIX)
    _atomic_json(
        crosscheck_path,
        {
            "schema": "malaiwah.glm53-inventory-crosscheck.v1",
            "our_inventory_sha256": inventory["inventory_sha256"],
            "upstream_inventory_sha256_prefix": UPSTREAM_INVENTORY_SHA_PREFIX,
            "binds_to_upstream_receipts": matches,
            "note": "equality binds the whole chain to brandonmusic's sealed receipts; "
            "recorded either way per RUNBOOK P1 step 1",
        },
    )
    print(
        f"inventory sealed: {inventory['inventory_sha256']} "
        f"(upstream cross-check match={matches})"
    )
    return inventory


def _adopt_inventory(
    doc_path: Path, bf16_root: Path, output_root: Path
) -> Dict[str, Any]:
    """Adopt an upstream sealed inventory verbatim; re-verify local shards."""

    from quant_pipeline.campaign.glm53_uniform_k4 import _inventory_surfaces
    from quant_pipeline.core.artifacts import sha256_file

    inventory = _read_json(doc_path, "upstream inventory")
    _inventory_surfaces(inventory)  # seal + geometry + census
    declared_root = Path(str(inventory.get("checkpoint", ""))).resolve()
    if declared_root != bf16_root.resolve():
        raise _fail(
            f"adopted inventory declares checkpoint {declared_root} but --bf16 is "
            f"{bf16_root} - Glm53BF16Source requires equality; place or symlink "
            "the BF16 tree at the declared path (and pass it as --bf16)"
        )
    marker = output_root / "inventory-shards-verified.json"
    if not marker.is_file():
        print("verifying local BF16 shards against the adopted inventory ...")
        for name, expected in inventory["shard_sha256"].items():
            observed = sha256_file(bf16_root / name)
            if observed != expected:
                raise _fail(
                    f"local shard {name} hash {observed[:16]}... differs from the "
                    f"adopted inventory ({str(expected)[:16]}...) - the BF16 tree "
                    "is not the sealed revision"
                )
            print(f"  {name} OK", flush=True)
        _atomic_json(
            marker,
            {
                "schema": "malaiwah.glm53-bf16-shards-verified.v1",
                "inventory_sha256": inventory["inventory_sha256"],
                "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method": "full-shard sha256 vs adopted upstream inventory",
            },
        )
    _atomic_json(output_root / "inventory.json", inventory)
    _atomic_json(
        output_root / "inventory-crosscheck.json",
        {
            "schema": "malaiwah.glm53-inventory-crosscheck.v1",
            "our_inventory_sha256": inventory["inventory_sha256"],
            "upstream_inventory_sha256_prefix": UPSTREAM_INVENTORY_SHA_PREFIX,
            "binds_to_upstream_receipts": inventory["inventory_sha256"].startswith(
                UPSTREAM_INVENTORY_SHA_PREFIX
            ),
            "adopted_verbatim_from": str(doc_path),
        },
    )
    print(f"adopted upstream inventory: {inventory['inventory_sha256']}")
    return inventory


def _build_preflight(inventory_sha256: str, out_path: Path) -> Dict[str, Any]:
    import torch

    from quant_pipeline.campaign.glm53_uniform_k4 import PREFLIGHT_SCHEMA, WORKERS

    if out_path.is_file():
        preflight = _read_json(out_path, "preflight")
        if preflight.get("checkpoint_inventory_sha256") == inventory_sha256:
            print("preflight reused")
            return preflight
    if not torch.cuda.is_available():
        raise _fail("preflight needs CUDA (the launch plan attests the actual devices)")
    count = torch.cuda.device_count()
    if count < WORKERS:
        raise _fail(f"preflight needs {WORKERS} visible GPUs, found {count}")
    gpus = []
    for index in range(WORKERS):
        major, minor = torch.cuda.get_device_capability(index)
        gpus.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": f"{major}.{minor}",
            }
        )
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "ready": True,
        "mode": "layer-streaming",
        "checkpoint_seal_mode": "full-shard-sha256",
        "checkpoint_inventory_sha256": inventory_sha256,
        "workers": WORKERS,
        "gpus": gpus,
    }
    preflight = _seal(body, "preflight_sha256")
    _atomic_json(out_path, preflight)
    return preflight


def _build_profile_selection(bits: int, shapley_root: Path, out_path: Path) -> Dict[str, Any]:
    from quant_pipeline.campaign.glm53_mcg_preparation import _verify_selection
    from quant_pipeline.core.artifacts import sha256_file

    if out_path.is_file():
        selection = _read_json(out_path, "profile selection")
        _verify_selection(selection, bits=bits)
        return selection
    driver_path = shapley_root / "scripts" / "run_qwen_fast_encode.py"
    if not driver_path.is_file():
        raise _fail(f"ShapleyMCG closure lacks {driver_path}")
    observed = sha256_file(driver_path)
    if observed != RUN_QWEN_FAST_ENCODE_SHA:
        raise _fail(
            "run_qwen_fast_encode.py sha mismatch (pin "
            f"{RUN_QWEN_FAST_ENCODE_SHA[:16]}..., observed {observed[:16]}...)"
        )
    body = {
        "schema": "quant-pipeline.glm53-shapleymcg-profile-selection.v1",
        "policy": "energy_balanced",
        "scale_family": "per128-grid",
        "bits": bits,
        "global_allocator_invoked": False,
        "candidate_rate_grid_invoked": False,
        "proposal_search_invoked": False,
        "profile_source": "public-run-qwen-fast-encode-defaults",
        "profile_fixed_before_encoding": True,
        "selection_rows_used": False,
        "selection_used_for_profile_choice": False,
        "selection_used_for_final_encoding": False,
        "confirmation_used_for_choice": False,
        "public_driver": "scripts/run_qwen_fast_encode.py",
        "public_shapleymcg_revision": SHAPLEY_REVISION,
        "run_qwen_fast_encode_sha256": RUN_QWEN_FAST_ENCODE_SHA,
    }
    selection = _seal(body, "selection_sha256")
    _verify_selection(selection, bits=bits)
    _atomic_json(out_path, selection)
    return selection


def _transform_seed(out_path: Path) -> str:
    """Freshly minted, then sealed forever (disclosed deviation 5)."""

    if out_path.is_file():
        return str(_read_json(out_path, "transform seed")["transform_seed_sha256"])
    seed = hashlib.sha256(
        b"malaiwah-glm53-k6-transform-seed\0" + os.urandom(32)
    ).hexdigest()
    _atomic_json(
        out_path,
        {
            "schema": "malaiwah.glm53-k6-transform-seed.v1",
            "transform_seed_sha256": seed,
            "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "first-ever K6; no upstream bitwise reference exists.  Our own "
            "five-cold-run receipt establishes determinism of OUR artifact.",
        },
    )
    return seed


def _reader_module_path() -> Path:
    import quant_pipeline.evaluation.glm53_packed_k4_reader as reader

    return Path(reader.__file__).resolve()


def _reader_abi_sha256() -> str:
    from quant_pipeline.core.artifacts import sha256_file

    return sha256_file(_reader_module_path())


def _prepared_source_doc(
    bits: int, shapley_root: Path, numeric_core: Path
) -> Dict[str, Any]:
    from quant_pipeline.campaign.glm53_direct_k4 import recipe_id_for_bits
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file

    files = sorted(
        path.relative_to(shapley_root).as_posix()
        for path in (shapley_root / "r7_encoder").rglob("*.py")
    )
    if not files:
        raise _fail(CLOSURE_HELP, code=6)
    tree = {name: sha256_file(shapley_root / name) for name in files}
    return {
        "recipe_id": recipe_id_for_bits(bits),
        "reviewed_glm53_entrypoint": True,
        "entrypoint": "quant_pipeline.campaign.glm53_prepared_backend.Glm53PreparedMCGBackend",
        "tree_sha256": sha256_bytes(canonical_json(tree)),
        "source_root": str(shapley_root),
        "numeric_core_path": str(numeric_core),
        "numeric_core_sha256": sha256_file(numeric_core),
        # recipe evidence (validated by _validate_recipe_evidence)
        "codec_family": "exl3-mcg",
        "mcg_multiplier_hex": "0xCBAC1FED",
        "bits": bits,
        "candidate_rate_grid": False,
        "global_allocator": False,
        "gate_up_hessian": "routed_p2_uncentered_full_hessian",
        "down_hessian": (
            f"decoded_k{bits}_candidate_conditioned_routed_p2_uncentered_full_hessian"
        ),
        "down_candidate_conditioned": True,
        "profile_source": "public-run-qwen-fast-encode-defaults",
        "profile_policy": "energy_balanced",
        "scale_family": "per128-grid",
        "profile_fixed_before_encoding": True,
        "selection_used_for_profile_choice": False,
        "selection_rows_used_for_encoding": False,
        "confirmation_rows_used_for_choice": False,
        "sqg_orchestration_imported": False,
    }


def _exllama_doc(extension: Path) -> Dict[str, Any]:
    from quant_pipeline.core.artifacts import sha256_file

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0;10.0")
    capabilities = [item.strip() for item in arch_list.replace(";", ",").split(",") if item.strip()]
    if "10.0" not in capabilities:
        raise _fail(
            "the sealed contract requires '10.0' in compute_capabilities; build the "
            'extension with TORCH_CUDA_ARCH_LIST="9.0;10.0" (disclosed deviation 2) '
            "and export the same TORCH_CUDA_ARCH_LIST when running this driver"
        )
    return {
        "fresh_build": True,
        "compute_capabilities": capabilities,
        "extension_path": str(extension),
        "extension_sha256": sha256_file(extension),
        "built_with_torch_cuda_arch_list": arch_list,
        "executes_on": "SM90 H200 (disclosed deviation: binary genuinely carries SM100 code objects)",
    }


def _source_closure_rows(shapley_root: Path) -> List[Dict[str, str]]:
    from quant_pipeline.core.artifacts import sha256_file

    rows = []
    for path in sorted((shapley_root / "r7_encoder").rglob("*.py")):
        rows.append(
            {
                "path": path.relative_to(shapley_root.parent).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if not any(row["path"].endswith("r7_encoder/r10_codec.py") for row in rows):
        raise _fail(CLOSURE_HELP, code=6)
    return rows


def _codec_probe(codec_adapter: Any, bits: int, device: str) -> Dict[str, Any]:
    """Encode/pack/decode one synthetic 256x256 matrix; independent-decode check."""

    import torch

    from quant_pipeline.evaluation.glm53_packed_k4_reader import decode_choice_hf
    from quant_pipeline.normalization.artifact_v31 import tensor_sha256

    torch.manual_seed(0x1FED)
    n, k = 256, 256
    weight = (torch.randn(n, k, dtype=torch.float32) * 0.02).to(device)
    covariance = torch.eye(k, dtype=torch.float32, device=device)
    suh = (torch.randint(0, 2, (k,), dtype=torch.int8).float() * 2 - 1).to(device)
    svh = (torch.randint(0, 2, (n,), dtype=torch.int8).float() * 2 - 1).to(device)
    candidates = codec_adapter.encode_candidates(
        unit_id="L0.E0.probe",
        weight_hf=weight,
        covariance=covariance,
        bits=(bits,),
        input_vector=suh,
        output_vector=svh,
    )
    candidate = candidates[bits]
    decoded = decode_choice_hf(
        candidate.packed.cpu(), suh.cpu(), svh.cpu(), bits=bits
    ).to(torch.float16)
    reference = candidate.reconstructed.detach().to("cpu", torch.float16)
    exact = tensor_sha256(decoded) == tensor_sha256(reference)
    return {
        "bits": bits,
        "shape": [n, k],
        "packed_sha256": candidate.packed_sha256,
        "reconstruction_sha256": candidate.reconstruction_sha256,
        "encode_pack_decode_exact": bool(exact),
    }


def _author_pure_mcg_receipts(
    *,
    bits: int,
    shapley_root: Path,
    codec_adapter: Any,
    device: str,
    backend_out: Path,
    preparation_out: Path,
) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    from quant_pipeline.campaign.glm53_mtp_k4 import (
        PURE_MCG_BACKEND_SCHEMA,
        PURE_MCG_PREPARATION_SCHEMA,
        verify_pure_mcg_backend_receipt,
        verify_pure_mcg_preparation_receipt,
    )
    from quant_pipeline.core.artifacts import sha256_file

    if backend_out.is_file() and preparation_out.is_file():
        backend = _read_json(backend_out, "pure MCG backend receipt")
        preparation = _read_json(preparation_out, "pure MCG preparation receipt")
        verify_pure_mcg_backend_receipt(backend)
        verify_pure_mcg_preparation_receipt(preparation)
        return backend, preparation

    closure = _source_closure_rows(shapley_root)
    probe = _codec_probe(codec_adapter, bits, device)
    if not probe["encode_pack_decode_exact"]:
        raise _fail(
            "actual MCG encode/pack/decode probe is NOT bit-exact - the numeric "
            "closure or extension differs; refusing to author qualification receipts"
        )
    reader_abi = _reader_abi_sha256()
    backend = _seal(
        {
            "schema": PURE_MCG_BACKEND_SCHEMA,
            "qualified": True,
            "bits": bits,
            "codec_family": "exl3-mcg",
            "mcg_multiplier_hex": "0xCBAC1FED",
            "sqg_orchestration_imported": False,
            "actual_mcg_encode_pack_decode_checked": True,
            "codec_class": "r7_encoder.r10_codec.R10TrellisCodec",
            "public_codec_adapter": "Exl3MCGCodec",
            "offline_reader_exact_decode_checked": True,
            "offline_reader_abi_sha256": reader_abi,
            "probe": probe,
            "source_closure": closure,
        },
        "receipt_sha256",
    )
    preparation = _seal(
        {
            "schema": PURE_MCG_PREPARATION_SCHEMA,
            "qualified": True,
            "bits": bits,
            "sqg_orchestration_imported": False,
            "public_shapleymcg_run_qwen_fast_encode_structure": True,
            "local_corrected_v1_numerical_order": True,
            "r7_encoder_r10_codec_closure": True,
            "codec_class": "r7_encoder.r10_codec.R10TrellisCodec",
            "local_corrected_v1_sha256": sha256_file(
                shapley_root / "r7_encoder" / "r10_codec.py"
            ),
            "source_closure": closure,
        },
        "receipt_sha256",
    )
    verify_pure_mcg_backend_receipt(backend)
    verify_pure_mcg_preparation_receipt(preparation)
    _atomic_json(backend_out, backend)
    _atomic_json(preparation_out, preparation)
    return backend, preparation


def _author_reader_abi_receipt(
    *, output_root: Path, contract: Dict[str, Any], bits: int, samples: int = 6
) -> Dict[str, Any]:
    """Exact-reconstruction spot check over sealed choices -> reader-ABI receipt."""

    import random

    import torch

    from quant_pipeline.campaign.glm53_direct_k4 import PROJECTIONS
    from quant_pipeline.checkpoint.packed_payload import (
        PackedMCGPayloadStore,
        checkpoint_payload_sha256,
    )
    from quant_pipeline.evaluation.glm53_packed_k4_reader import decode_choice_hf
    from quant_pipeline.campaign.glm53_mtp_k4 import READER_ABI_SCHEMA
    from quant_pipeline.normalization.artifact_v31 import tensor_sha256

    out_path = output_root / "reader-abi-receipt.json"
    if out_path.is_file():
        return _read_json(out_path, "reader ABI receipt")
    receipts = sorted(output_root.glob("experts/layer-*/expert-*.json"))
    if not receipts:
        raise _fail("no expert receipts yet - reader ABI receipt needs sealed choices")
    store = PackedMCGPayloadStore(output_root / "payload-store")
    rng = random.Random(0xCBAC1FED)
    checked: List[Dict[str, Any]] = []
    for path in rng.sample(receipts, min(samples, len(receipts))):
        receipt = _read_json(path, "expert receipt")
        projection = rng.choice(list(PROJECTIONS))
        choice = receipt["choices"][projection]
        store.verify_choice(choice)
        payload = {
            name: store.objects.load_tensor(choice["objects"][name])
            for name in ("trellis", "suh", "svh", "mcg")
        }
        if checkpoint_payload_sha256(payload) != choice["checkpoint_payload_sha256"]:
            raise _fail(f"checkpoint payload hash differs for {path.name}/{projection}")
        decoded = decode_choice_hf(
            payload["trellis"], payload["suh"], payload["svh"], bits=bits
        ).to(torch.float16)
        expected = choice["reconstruction_closure"]["payload_sha256"]
        observed = tensor_sha256(decoded)
        if observed != expected:
            raise _fail(
                f"independent decode differs from encoder reconstruction closure "
                f"({path.name}/{projection}) - reader ABI NOT qualified"
            )
        checked.append(
            {
                "expert_receipt": path.name,
                "layer": receipt["layer"],
                "expert": receipt["expert"],
                "projection": projection,
                "choice_sha256": choice["choice_sha256"],
            }
        )
    receipt = _seal(
        {
            "schema": READER_ABI_SCHEMA,
            "qualified": True,
            "bits": bits,
            "tp_sizes": [4] if bits in (6, 8) else [2, 4],
            "exact_reconstruction_checked": True,
            "reader_sha256": _reader_abi_sha256(),
            "contract_sha256": contract["contract_sha256"],
            "sampled_choices": checked,
        },
        "receipt_sha256",
    )
    _atomic_json(out_path, receipt)
    return receipt


# --------------------------------------------------------------------------- #
# work-state persistence                                                       #
# --------------------------------------------------------------------------- #

def _state_paths(output_root: Path) -> "tuple[Path, Path]":
    return output_root / "state" / "work-state.json", output_root / "state" / ".lock"


def _load_state(output_root: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    from quant_pipeline.campaign.glm53_direct_k4 import verify_work_state

    state_path, _ = _state_paths(output_root)
    state = _read_json(state_path, "work state")
    verify_work_state(contract, state)
    return state


def _persist_state(output_root: Path, state: Dict[str, Any]) -> None:
    state_path, _ = _state_paths(output_root)
    _atomic_json(state_path, state)
    history = output_root / "state" / "history"
    history.mkdir(parents=True, exist_ok=True)
    _atomic_json(history / f"state-{int(state['sequence']):06d}.json", state)


# --------------------------------------------------------------------------- #
# subcommand: contract                                                         #
# --------------------------------------------------------------------------- #

def _purge_sealed_codec_modules() -> None:
    """Drop cached r7_encoder modules so each build_layer_preparation performs a
    fresh SEALED import (exl3_mcg._codec refuses incumbents; per-layer codec
    construction re-verifies the closure hashes every time — the same cleanup
    pattern brandonmusic's own shapleymcg tests use between sealed imports)."""
    import sys as _sys
    for name in [n for n in _sys.modules if n == "r7_encoder" or n.startswith("r7_encoder.")]:
        _sys.modules.pop(name, None)


def cmd_contract(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    shapley_root = Path(args.shapley_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _require_codec_closure(shapley_root, output_root)
    bits = _bits_for_profile(args.profile, output_root)

    from quant_pipeline.calibration.glm53_capture import (
        CAPTURE_SCHEMA,
        verify_seal as verify_capture_seal,
    )
    from quant_pipeline.campaign import glm53_direct_k4 as direct

    if bits == 8:
        from quant_pipeline.campaign import glm53_uniform_k8 as uniform_plan
    else:
        from quant_pipeline.campaign import glm53_uniform_k6 as uniform_plan
    from quant_pipeline.campaign.glm53_mcg_preparation import (
        build_layer_preparation,
        seal_campaign_preparation,
    )
    from quant_pipeline.campaign.glm53_uniform_k4 import MAIN_ROUTED_LAYERS
    from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

    recipe = _read_json(Path(args.recipe), "recipe")
    model_revision = str(recipe.get("source_revision", MODEL_REVISION_DEFAULT))
    bf16_root = Path(args.bf16).resolve()
    cal_root = Path(args.calibration).resolve()
    capture_root = cal_root / "main-ep4-full"
    extension = _find_extension(Path(args.exllama_root).resolve(), args.extension)
    numeric_core = _numeric_core(shapley_root, args.numeric_core)

    # 1) sealed inventory (content-addressed reuse) + cross-check receipt.
    #    --inventory adopts an upstream-published sealed inventory VERBATIM
    #    (required for his captures: capture-manifest.json binds HIS
    #    inventory_sha256 and build_contract hard-requires equality); local
    #    shards are then re-verified against its shard_sha256 closure.
    if args.inventory:
        inventory = _adopt_inventory(
            Path(args.inventory).resolve(), bf16_root, output_root
        )
    else:
        inventory = _build_inventory(
            bf16_root,
            model_revision,
            output_root / "inventory.json",
            output_root / "inventory-crosscheck.json",
        )

    # 2) preflight + K6/K8 launch plan.  Upstream's shipped glm53_uniform_k6
    #    builder is K4-KL-GATED: build_launch_plan(inventory, preflight, *,
    #    k4_plan, k4_authorized_state).  The K4 plan is a pure planning
    #    document (launch_authorized False) built from the same inventory +
    #    preflight; the k6_authorized K4 STATE receipt is an existential input
    #    that this driver never fabricates - it must be provided (sourced from
    #    brandonmusic's published K4 receipts via the disclosed bridge doc, or
    #    from a locally executed K4 campaign).
    preflight = _build_preflight(
        inventory["inventory_sha256"], output_root / "preflight.json"
    )
    from quant_pipeline.campaign import glm53_uniform_k4 as uniform_k4

    k4_plan_path = (
        Path(args.k4_plan).resolve()
        if args.k4_plan
        else output_root / "k4-launch-plan.json"
    )
    if k4_plan_path.is_file():
        k4_plan = _read_json(k4_plan_path, "K4 launch plan")
        uniform_k4.verify_launch_plan(k4_plan)
    else:
        k4_plan = uniform_k4.build_launch_plan(inventory, preflight)
        _atomic_json(output_root / "k4-launch-plan.json", k4_plan)
    k4_state_path = (
        Path(args.k4_state).resolve()
        if args.k4_state
        else output_root / "k4-authorized-state.json"
    )
    if not k4_state_path.is_file():
        raise _fail(
            f"K4 gate state receipt missing: {k4_state_path}.  Upstream's K6 "
            "launch-plan builder requires a sealed glm53_uniform_k4 state "
            "receipt in phase k6_authorized that binds the K4 launch plan "
            f"({output_root / 'k4-launch-plan.json'}).  Author it as the "
            "disclosed bridge from brandonmusic's published K4 campaign "
            "receipts (his packed-KLD receipt is the KL-gate evidence) and "
            "pass it via --k4-state; this driver refuses to fabricate it.",
            code=8,
        )
    k4_authorized_state = _read_json(k4_state_path, "K4 authorized state")
    uniform_k4.verify_state(k4_plan, k4_authorized_state)
    plan_path = output_root / "launch-plan.json"
    if plan_path.is_file():
        launch_plan = _read_json(plan_path, "launch plan")
        uniform_plan.verify_launch_plan(launch_plan)
    else:
        launch_plan = uniform_plan.build_launch_plan(
            inventory,
            preflight,
            k4_plan=k4_plan,
            k4_authorized_state=k4_authorized_state,
        )
        _atomic_json(plan_path, launch_plan)
    print(f"launch plan: {launch_plan['launch_plan_sha256']}")

    # 3) profile selection + transform seed (K8 fail-fasts unless the K6 seed
    #    was copied in - same seed is an operator requirement)
    selection = _build_profile_selection(
        bits, shapley_root, output_root / "profile-selection.json"
    )
    _require_k8_seed(args.profile, output_root)
    seed = _transform_seed(output_root / "transform-seed.json")

    # 4) capture manifest (sealed) for the contract binding.  A verified copy
    #    is cached in output_root so the stage stays resumable after the disk
    #    ledger deletes calibration/main-ep4-full (post-encode, pre-materialize).
    manifest_cache = output_root / "capture-manifest.json"
    manifest_path = capture_root / "capture-manifest.json"
    if not manifest_path.is_file() and manifest_cache.is_file():
        manifest_path = manifest_cache
    capture_manifest = _read_json(manifest_path, "main capture manifest")
    verify_capture_seal(capture_manifest, schema=CAPTURE_SCHEMA, field="capture_sha256")
    if manifest_path != manifest_cache:
        _atomic_json(manifest_cache, capture_manifest)
    if capture_manifest.get("inventory_sha256") != inventory["inventory_sha256"]:
        raise _fail(
            "the calibration capture binds inventory "
            f"{str(capture_manifest.get('inventory_sha256'))[:16]}... but the local "
            f"inventory is {inventory['inventory_sha256'][:16]}... - build_contract "
            "hard-requires equality.  For brandonmusic's published captures, pass "
            "his sealed inventory document via --inventory (and keep the BF16 tree "
            "at the checkpoint path it declares); building a fresh inventory only "
            "works with self-captured calibration."
        )

    # 5) layer preparations (idempotent per layer; `prepare` can parallelize)
    verified_marker = output_root / "inventory-shards-verified.json"
    source = direct.Glm53BF16Source(
        inventory, bf16_root, verify_shards=not verified_marker.is_file()
    )
    if not verified_marker.is_file():
        _atomic_json(
            verified_marker,
            {
                "schema": "malaiwah.glm53-bf16-shards-verified.v1",
                "inventory_sha256": inventory["inventory_sha256"],
                "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
    manifests: List[Dict[str, Any]] = []
    for layer in MAIN_ROUTED_LAYERS:
        manifest_path = output_root / "preparation" / f"layer-{layer:03d}" / "preparation.json"
        if not manifest_path.is_file():
            print(f"preparing layer {layer} (K{bits} GSS) ...", flush=True)
            capture = direct.Glm53CaptureView(
                capture_root, layer, verify_hashes=args.verify_capture_hashes
            )
            _purge_sealed_codec_modules()
            build_layer_preparation(
                layer=layer,
                capture=capture,
                source=source,
                source_root=shapley_root,
                numeric_core=numeric_core,
                extension=extension,
                output_root=output_root / "preparation",
                transform_seed_sha256=seed,
                profile_selection=selection,
                device=args.device,
                bits=bits,
            )
        manifests.append(_read_json(manifest_path, f"layer {layer} preparation"))
    campaign_preparation = seal_campaign_preparation(manifests)
    _atomic_json(output_root / "campaign-preparation-receipt.json", campaign_preparation)

    # 6) pure-MCG qualification receipts (real probe) + direct contract
    codec_adapter = Exl3MCGCodec(
        source_root=shapley_root,
        numeric_core=numeric_core,
        extension=extension,
        device=args.device,
        sigma_reg=0.025,
    )
    pure_backend, pure_preparation = _author_pure_mcg_receipts(
        bits=bits,
        shapley_root=shapley_root,
        codec_adapter=codec_adapter,
        device=args.device,
        backend_out=output_root / "pure-mcg-backend-receipt.json",
        preparation_out=output_root / "pure-mcg-preparation-receipt.json",
    )
    prepared_source = _prepared_source_doc(bits, shapley_root, numeric_core)
    exllama = _exllama_doc(extension)
    contract_path = output_root / "contract.json"
    if contract_path.is_file():
        contract = _read_json(contract_path, "direct contract")
        direct.verify_contract(contract)
    else:
        preparation_doc = dict(campaign_preparation)
        preparation_doc.update(
            {
                "bits": bits,
                "codec_family": "exl3-mcg",
                "confirmation_report_only": True,
            }
        )
        contract = direct.build_contract(
            launch_plan=launch_plan,
            inventory=inventory,
            capture_manifest=capture_manifest,
            prepared_source=prepared_source,
            exllama=exllama,
            preparation=preparation_doc,
            reader_abi_sha256=_reader_abi_sha256(),
            pure_mcg_backend_receipt_sha256=pure_backend["receipt_sha256"],
            pure_mcg_preparation_receipt_sha256=pure_preparation["receipt_sha256"],
            bits=bits,
        )
        _atomic_json(contract_path, contract)
    print(f"direct contract: {contract['contract_sha256']}")

    # 7) work units + hash-chained work state
    units_path = output_root / "work-units.json"
    if units_path.is_file():
        units = _read_json(units_path, "work units")["units"]
    else:
        units = direct.build_work_units(contract)
        _atomic_json(units_path, {"schema": "malaiwah.k6-work-units.v1", "units": units})
    state_path, lock_path = _state_paths(output_root)
    with _locked(lock_path):
        if not state_path.is_file():
            state = direct.initial_work_state(contract, units)
            _persist_state(output_root, state)
        else:
            _load_state(output_root, contract)
    print(f"work units: {len(units)}; state chain ready at {state_path}")

    if args.reuse_gate_up_from:
        raise _fail(
            "--reuse-gate-up-from is a K6K8-only economy and needs the malaiwah "
            "K6K8 support module (RUNBOOK P2); it cannot apply to the uniform-K6 "
            "sealed contract",
            code=7,
        )
    return 0


# --------------------------------------------------------------------------- #
# subcommand: prepare (auxiliary parallelization helper)                       #
# --------------------------------------------------------------------------- #

def cmd_prepare(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    shapley_root = Path(args.shapley_root).resolve()
    _require_codec_closure(shapley_root, output_root)
    bits = _bits_for_profile(args.profile, output_root)

    from quant_pipeline.campaign import glm53_direct_k4 as direct
    from quant_pipeline.campaign.glm53_mcg_preparation import build_layer_preparation

    inventory = _read_json(output_root / "inventory.json", "inventory (run `contract` first)")
    selection = _read_json(output_root / "profile-selection.json", "profile selection")
    _require_k8_seed(args.profile, output_root)
    seed = _transform_seed(output_root / "transform-seed.json")
    extension = _find_extension(Path(args.exllama_root).resolve(), args.extension)
    numeric_core = _numeric_core(shapley_root, args.numeric_core)
    source = direct.Glm53BF16Source(inventory, Path(args.bf16).resolve(), verify_shards=False)
    for layer in _parse_layers(args.layers):
        manifest_path = output_root / "preparation" / f"layer-{layer:03d}" / "preparation.json"
        if manifest_path.is_file():
            print(f"layer {layer}: preparation already sealed")
            continue
        capture = direct.Glm53CaptureView(
            Path(args.calibration).resolve() / "main-ep4-full",
            layer,
            verify_hashes=args.verify_capture_hashes,
        )
        _purge_sealed_codec_modules()
        build_layer_preparation(
            layer=layer,
            capture=capture,
            source=source,
            source_root=shapley_root,
            numeric_core=numeric_core,
            extension=extension,
            output_root=output_root / "preparation",
            transform_seed_sha256=seed,
            profile_selection=selection,
            device=args.device,
            bits=bits,
        )
        print(f"layer {layer}: preparation sealed", flush=True)
    return 0


def _parse_layers(text: str) -> List[int]:
    layers: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            layers.extend(range(int(start), int(stop) + 1))
        else:
            layers.append(int(part))
    return layers


# --------------------------------------------------------------------------- #
# subcommand: encode-worker                                                    #
# --------------------------------------------------------------------------- #

def _build_backend(
    *,
    contract: Dict[str, Any],
    inventory: Dict[str, Any],
    shapley_root: Path,
    output_root: Path,
    device: str,
) -> Any:
    from quant_pipeline.campaign.glm53_prepared_backend import Glm53PreparedMCGBackend

    return Glm53PreparedMCGBackend(
        contract=contract,
        inventory=inventory,
        prepared_root=shapley_root,
        preparation_root=output_root / "preparation",
        hessian_root=output_root / "hessians",
        reader_abi_sha256=_reader_abi_sha256(),
        device=device,
    )


def _prune_layer_hessians(output_root: Path, layer: int) -> int:
    """Delete Hessian artifacts referenced by the layer's sealed expert receipts."""

    pruned = 0
    for path in sorted(output_root.glob(f"experts/layer-{layer:03d}/expert-*.json")):
        receipt = _read_json(path, "expert receipt")
        artifact = receipt.get("recipe_evidence", {}).get("hessian_artifact", {})
        hessian = Path(str(artifact.get("path", "")))
        if hessian.is_file():
            hessian.unlink()
            pruned += 1
    return pruned


def _maybe_seal_layer(
    output_root: Path, contract: Dict[str, Any], layer: int, prune: bool
) -> bool:
    from quant_pipeline.campaign.glm53_direct_k4 import NUM_EXPERTS, seal_layer

    receipt_path = output_root / "layers" / f"layer-{layer:03d}.json"
    if receipt_path.is_file():
        return True
    experts = len(list(output_root.glob(f"experts/layer-{layer:03d}/expert-*.json")))
    if experts != NUM_EXPERTS:
        return False
    seal_layer(output_root, contract, layer)
    print(f"layer {layer}: sealed ({experts} expert receipts)")
    if prune:
        count = _prune_layer_hessians(output_root, layer)
        print(f"layer {layer}: pruned {count} hessian artifacts (disclosed deviation 4)")
    return True


def _seal_encoded_experts(
    *,
    direct: Any,
    store: Any,
    batch: List[Any],
    encoded: List[Any],
    contract_sha: str,
    work_unit: Mapping[str, Any],
    capture: Any,
    backend_identity: Mapping[str, Any],
    source: Any,
    bits: int,
    output_root: Path,
    completed_by_expert: Dict[int, str],
) -> None:
    """Seal path of glm53_direct_k4.encode_work_unit, hoisted verbatim.

    --overlap-seal reschedules WHEN this runs (background thread while the GPU
    encodes the next batch), never WHAT it does: every pipeline call below is
    the same call, with the same inputs, in the same order, as the sealed
    encode_work_unit seal loop.  Receipt order within an expert (PROJECTIONS
    chaining from the work-unit sha) is preserved because this body is that
    loop.
    """

    for (expert, path, _request), raw_result in zip(batch, encoded, strict=True):
        result = dict(raw_result)
        evidence = result.get("recipe_evidence")
        payloads = result.get("projections")
        if not isinstance(evidence, Mapping) or not isinstance(payloads, Mapping):
            raise ValueError("prepared backend returned an incomplete expert triplet")
        if bits == 4:
            direct._validate_recipe_evidence(evidence)
        else:
            direct._validate_recipe_evidence(evidence, bits=bits)
        choices: Dict[str, Any] = {}
        predecessor = work_unit["work_unit_sha256"]
        for projection in direct.PROJECTIONS:
            payload = payloads.get(projection)
            if not isinstance(payload, Mapping):
                raise ValueError(f"prepared backend omitted {projection}")
            choices[projection] = store.put_choice(
                layer=capture.layer,
                expert=expert,
                projection=projection,
                choice_id=f"L{capture.layer:03d}.E{expert:03d}.{projection}.K{bits}",
                bits=bits,
                trellis=payload["trellis"],
                suh=payload["suh"],
                svh=payload["svh"],
                mcg=payload["mcg"],
                reconstruction=payload["reconstruction"],
                vector_topology=payload["vector_topology"],
                reader_abi_sha256=backend_identity["reader_abi_sha256"],
                provenance={
                    "contract_sha256": contract_sha,
                    "backend": backend_identity,
                    "source_payload_sha256": source.rows[direct.tensor_name(capture.layer, expert, projection)]["source_payload_sha256"],
                },
                predecessor_state_hash=predecessor,
            )
            predecessor = choices[projection]["choice_sha256"]
        body = {
            "schema": direct.EXPERT_RECEIPT_SCHEMA,
            "contract_sha256": contract_sha,
            "work_unit_sha256": work_unit["work_unit_sha256"],
            "layer": capture.layer,
            "expert": expert,
            "bits": bits,
            "projections": list(direct.PROJECTIONS),
            "candidate_rate_grid": False,
            "global_allocator": False,
            "down_candidate_conditioned": True,
            "capture_binding": capture.binding(),
            "backend": backend_identity,
            "choices": choices,
            "recipe_evidence": dict(evidence),
        }
        receipt = direct._seal(body, "receipt_sha256")
        direct.write_json(path, receipt)
        direct.verify_expert_receipt(
            output_root,
            path,
            contract_sha256=contract_sha,
            expected_bits=bits,
        )
        completed_by_expert[expert] = receipt["receipt_sha256"]


def _encode_work_unit_overlap(
    *,
    direct: Any,
    contract: Mapping[str, Any],
    work_unit: Mapping[str, Any],
    source: Any,
    capture: Any,
    backend: Any,
    output_root: Path,
    device: str,
    max_inflight_experts: int,
) -> Dict[str, Any]:
    """encode_work_unit with the CPU seal of batch N overlapped with the GPU
    encode of batch N+1 (opt-in via --overlap-seal).

    Identical to glm53_direct_k4.encode_work_unit except scheduling: the
    single encode_experts mega-call becomes per-batch calls (the backend
    already slices that mega-call into the same <= max_inflight_experts
    batches internally, so per-expert encode inputs are unchanged), and each
    batch's seal loop runs on ONE background thread while the next batch
    encodes.  Guarantees kept:
      * per-expert receipt content and intra-expert choice chaining: the seal
        body is the hoisted pipeline loop (_seal_encoded_experts), unchanged;
      * ordering: seals run in batch submission order on a single thread, so
        expert receipts land in the same ascending order as today;
      * unit receipt (and therefore complete_work_unit / seal_layer) only
        after ALL seals drained;
      * seal failure: no further encode batch starts after the failure is
        observed, the unit receipt is never written, the exception propagates
        exactly like today;
      * encode failure: pending seals drain (their receipts land, preserving
        resume), then the encode exception propagates.
    """

    import concurrent.futures

    contract_sha = direct.verify_contract(contract)
    bits = int(contract.get("rate", {}).get("bits", work_unit.get("bits", -1)))
    direct.recipe_id_for_bits(bits)
    direct._verify_seal(work_unit, direct.WORK_UNIT_SCHEMA, "work_unit_sha256")
    if (
        work_unit.get("contract_sha256") != contract_sha
        or work_unit.get("layer") != capture.layer
        or work_unit.get("bits") != bits
    ):
        raise ValueError("work unit/capture/contract binding differs")
    backend_identity = direct._verify_backend(contract, backend)
    output_root = Path(output_root)
    store = direct.PackedMCGPayloadStore(output_root / "payload-store")
    completed_by_expert: Dict[int, str] = {}
    pending: List[Any] = []
    start, stop = int(work_unit["expert_start"]), int(work_unit["expert_stop"])
    for expert in range(start, stop):
        path = direct._expert_path(output_root, capture.layer, expert)
        if path.exists():
            receipt = direct.verify_expert_receipt(
                output_root,
                path,
                contract_sha256=contract_sha,
                expected_bits=bits,
            )
            completed_by_expert[expert] = receipt["receipt_sha256"]
            continue
        weights = source.load_triplet(capture.layer, expert, device=device)
        request = direct.EncodeRequest(
            contract_sha256=contract_sha,
            layer=capture.layer,
            expert=expert,
            bits=bits,
            tensor_names={p: direct.tensor_name(capture.layer, expert, p) for p in direct.PROJECTIONS},
            source_weights=weights,
            capture=capture,
            preparation=contract["preparation"],
        )
        pending.append((expert, path, request))

    batch_size = max(1, int(max_inflight_experts))
    seal_futures: List[Any] = []

    def _raise_finished_seal_failure() -> None:
        for future in seal_futures:
            if future.done():
                future.result()  # re-raises the seal exception, failing the unit

    def _drain_all_seals() -> None:
        concurrent.futures.wait(seal_futures)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="seal-overlap"
    ) as sealer:
        try:
            for offset in range(0, len(pending), batch_size):
                # a seal failure aborts BEFORE any further encode starts.
                _raise_finished_seal_failure()
                batch = pending[offset : offset + batch_size]
                try:
                    encoded = list(
                        backend.encode_experts(
                            [request for _expert, _path, request in batch],
                            max_inflight_experts=max_inflight_experts,
                        )
                    )
                    if len(encoded) != len(batch):
                        raise ValueError("prepared backend batch result census differs")
                except BaseException:
                    # encode failure: drain pending seals (their receipts
                    # land, keeping resume intact), then abort with the
                    # encode error.
                    _drain_all_seals()
                    raise
                # a seal failure observed here discards this batch's encode
                # instead of sealing it (closest to the serial path, where a
                # seal failure precedes any further work).
                _raise_finished_seal_failure()
                seal_futures.append(
                    sealer.submit(
                        _seal_encoded_experts,
                        direct=direct,
                        store=store,
                        batch=batch,
                        encoded=encoded,
                        contract_sha=contract_sha,
                        work_unit=work_unit,
                        capture=capture,
                        backend_identity=backend_identity,
                        source=source,
                        bits=bits,
                        output_root=output_root,
                        completed_by_expert=completed_by_expert,
                    )
                )
            # drain EVERY seal before the unit receipt; first seal failure
            # propagates and the unit receipt is never written.
            _drain_all_seals()
            for future in seal_futures:
                future.result()
        except BaseException:
            _drain_all_seals()
            raise

    completed = [completed_by_expert[expert] for expert in range(start, stop)]
    unit_receipt = direct._seal(
        {
            "schema": "quant-pipeline.glm53-direct-mcg-work-unit-receipt.v1",
            "contract_sha256": contract_sha,
            "work_unit_sha256": work_unit["work_unit_sha256"],
            "layer": capture.layer,
            "expert_start": start,
            "expert_stop": stop,
            "expert_receipt_sha256": completed,
            "complete": len(completed) == stop - start,
        },
        "receipt_sha256",
    )
    path = output_root / "work-units" / f"{work_unit['work_unit_sha256']}.json"
    direct.write_json(path, unit_receipt)
    return unit_receipt


def cmd_encode_worker(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    shapley_root = Path(args.shapley_root).resolve()
    _require_codec_closure(shapley_root, output_root)
    _bits_for_profile(args.profile, output_root)

    from quant_pipeline.campaign import glm53_direct_k4 as direct

    contract = _read_json(output_root / "contract.json", "direct contract (run `contract` first)")
    direct.verify_contract(contract)
    inventory = _read_json(output_root / "inventory.json", "inventory")
    source = direct.Glm53BF16Source(inventory, Path(args.bf16).resolve(), verify_shards=False)
    backend = _build_backend(
        contract=contract,
        inventory=inventory,
        shapley_root=shapley_root,
        output_root=output_root,
        device=args.device,
    )
    capture_root = Path(args.calibration).resolve() / "main-ep4-full"
    _, lock_path = _state_paths(output_root)
    worker = args.worker
    overlap_seal = bool(getattr(args, "overlap_seal", False))
    max_units = getattr(args, "max_units", None)
    units_done = 0

    while True:
        if max_units is not None and units_done >= max_units:
            print(f"{worker}: reached --max-units {max_units} - stopping")
            return 0
        with _locked(lock_path):
            state = _load_state(output_root, contract)
            claim = state["active"].get(worker)
            if claim is None:
                if not state["pending"]:
                    print(f"{worker}: no pending work units remain")
                    return 0
                state, claim = direct.claim_next_work_unit(
                    contract, state, worker_id=worker
                )
                _persist_state(output_root, state)
        unit = state["units"][claim["work_unit_sha256"]]
        layer = int(unit["layer"])
        print(f"{worker}: claimed layer {layer} unit {claim['work_unit_sha256'][:16]}", flush=True)

        layer_receipt = output_root / "layers" / f"layer-{layer:03d}.json"
        unit_receipt_path = output_root / "work-units" / f"{unit['work_unit_sha256']}.json"
        if layer_receipt.is_file() and unit_receipt_path.is_file():
            # sealed layer (hessians possibly pruned): never re-enter encode.
            unit_receipt = _read_json(unit_receipt_path, "work-unit receipt")
        else:
            capture = direct.Glm53CaptureView(
                capture_root, layer, verify_hashes=args.verify_capture_hashes
            )
            started = time.monotonic()
            if overlap_seal:
                unit_receipt = _encode_work_unit_overlap(
                    direct=direct,
                    contract=contract,
                    work_unit=unit,
                    source=source,
                    capture=capture,
                    backend=backend,
                    output_root=output_root,
                    device=args.device,
                    max_inflight_experts=args.max_inflight_experts,
                )
            else:
                unit_receipt = direct.encode_work_unit(
                    contract=contract,
                    work_unit=unit,
                    source=source,
                    capture=capture,
                    backend=backend,
                    output_root=output_root,
                    device=args.device,
                    max_inflight_experts=args.max_inflight_experts,
                )
            mode = " (overlap-seal)" if overlap_seal else ""
            print(
                f"{worker}: layer {layer} encoded in "
                f"{time.monotonic() - started:.0f}s{mode}",
                flush=True,
            )
        with _locked(lock_path):
            state = _load_state(output_root, contract)
            state = direct.complete_work_unit(
                contract,
                state,
                worker_id=worker,
                work_unit_receipt_sha256=unit_receipt["receipt_sha256"],
            )
            _persist_state(output_root, state)
        # seal (and prune) outside the state lock: seal_layer re-verifies all
        # 288 expert receipts including Hessian artifact hashes (minutes).
        _maybe_seal_layer(
            output_root, contract, layer, args.prune_hessians_after_layer_seal
        )
        units_done += 1


# --------------------------------------------------------------------------- #
# subcommand: seal-main / release-dead-claims                                  #
# --------------------------------------------------------------------------- #

def cmd_seal_main(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    bits = _bits_for_profile(args.profile, output_root)

    from quant_pipeline.campaign import glm53_direct_k4 as direct
    from quant_pipeline.campaign.glm53_mtp_k4 import _main_receipt_schema
    from quant_pipeline.campaign.glm53_uniform_k4 import MAIN_ROUTED_LAYERS

    contract = _read_json(output_root / "contract.json", "direct contract")
    contract_sha = direct.verify_contract(contract)
    receipts = []
    for layer in MAIN_ROUTED_LAYERS:
        path = output_root / "layers" / f"layer-{layer:03d}.json"
        if not path.is_file():
            # opportunistic: seal any layer whose 288 expert receipts exist.
            if not _maybe_seal_layer(output_root, contract, layer, prune=False):
                print(f"layer {layer}: not complete - main cannot seal yet")
                return 3
        receipt = _read_json(path, f"layer {layer} receipt")
        direct._verify_seal(receipt, direct.LAYER_RECEIPT_SCHEMA, "receipt_sha256")
        if receipt.get("contract_sha256") != contract_sha or receipt.get("complete") is not True:
            raise _fail(f"layer {layer} receipt does not bind this contract")
        receipts.append(receipt["receipt_sha256"])
    main_receipt = _seal(
        {
            # K4/K6: upstream parametric family; K8: malaiwah.* (one helper,
            # minted here and verified by glm53_mtp_k4.build_contract)
            "schema": _main_receipt_schema(bits),
            "contract_sha256": contract_sha,
            "bits": bits,
            "layers": list(MAIN_ROUTED_LAYERS),
            "layer_receipt_sha256": receipts,
            "matrix_count": direct.MAIN_MATRIX_COUNT,
            "complete": True,
        },
        "receipt_sha256",
    )
    _atomic_json(output_root / "main-receipt.json", main_receipt)
    print(f"main receipt sealed: {main_receipt['receipt_sha256']}")
    return 0


def cmd_release_dead_claims(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    _bits_for_profile(args.profile, output_root)

    from quant_pipeline.campaign import glm53_direct_k4 as direct

    contract = _read_json(output_root / "contract.json", "direct contract")
    _, lock_path = _state_paths(output_root)
    with _locked(lock_path):
        state = _load_state(output_root, contract)
        active = dict(state["active"])
        if not active:
            print("no active claims to release")
            return 0
        for worker, claim in active.items():
            unit_sha = claim["work_unit_sha256"]
            unit_receipt = output_root / "work-units" / f"{unit_sha}.json"
            if unit_receipt.is_file():
                receipt = _read_json(unit_receipt, "work-unit receipt")
                state = direct.complete_work_unit(
                    contract,
                    state,
                    worker_id=worker,
                    work_unit_receipt_sha256=receipt["receipt_sha256"],
                )
                print(f"{worker}: claim finished on disk - completed {unit_sha[:16]}")
            else:
                pending = [unit_sha] + list(state["pending"])
                remaining = dict(state["active"])
                remaining.pop(worker)
                state = direct._work_successor(
                    contract, state, pending=pending, active=remaining
                )
                print(f"{worker}: dead claim requeued {unit_sha[:16]}")
        _persist_state(output_root, state)
    return 0


# --------------------------------------------------------------------------- #
# subcommand: mtp                                                              #
# --------------------------------------------------------------------------- #

def cmd_mtp(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    bits = _bits_for_profile(args.profile, output_root)
    shapley_root = Path(args.shapley_root).resolve() if args.shapley_root else None

    from quant_pipeline.campaign import glm53_direct_k4 as direct
    from quant_pipeline.campaign import glm53_mtp_k4 as mtp
    from quant_pipeline.campaign.glm53_mcg_preparation import build_layer_preparation

    contract = _read_json(output_root / "contract.json", "direct contract")
    direct.verify_contract(contract)
    inventory = _read_json(output_root / "inventory.json", "inventory")
    launch_plan = _read_json(output_root / "launch-plan.json", "launch plan")
    main_receipt = _read_json(
        output_root / "main-receipt.json",
        "main receipt (run seal-main first: main_must_complete_before_mtp)",
    )
    if shapley_root is None:
        shapley_root = Path(contract["recipe"]["source_root"]).resolve()
    _require_codec_closure(shapley_root, output_root)

    capture = mtp.Glm53MTP45CaptureView(
        Path(args.calibration).resolve() / "mtp45-ep4-full",
        verify_hashes=args.verify_capture_hashes,
    )
    source = direct.Glm53BF16Source(inventory, Path(args.bf16).resolve(), verify_shards=False)
    backend = _build_backend(
        contract=contract,
        inventory=inventory,
        shapley_root=shapley_root,
        output_root=output_root,
        device=args.device,
    )

    # MTP45 preparation (idempotent, layer-045)
    manifest_path = output_root / "preparation" / "layer-045" / "preparation.json"
    if not manifest_path.is_file():
        selection = _read_json(output_root / "profile-selection.json", "profile selection")
        _require_k8_seed(args.profile, output_root)
        seed = _transform_seed(output_root / "transform-seed.json")
        _purge_sealed_codec_modules()
        build_layer_preparation(
            layer=mtp.MTP_LAYER,
            capture=capture,
            source=source,
            source_root=shapley_root,
            numeric_core=Path(contract["recipe"]["numeric_core_path"]),
            extension=Path(contract["exllama"]["extension_path"]),
            output_root=output_root / "preparation",
            transform_seed_sha256=seed,
            profile_selection=selection,
            device=args.device,
            bits=bits,
        )
    preparation_manifest = _read_json(manifest_path, "MTP45 preparation")

    reader_abi = _author_reader_abi_receipt(
        output_root=output_root, contract=contract, bits=bits
    )
    pure_backend = _read_json(
        output_root / "pure-mcg-backend-receipt.json", "pure MCG backend receipt"
    )
    pure_preparation = _read_json(
        output_root / "pure-mcg-preparation-receipt.json", "pure MCG preparation receipt"
    )

    mtp_contract_path = output_root / "mtp-contract.json"
    if mtp_contract_path.is_file():
        mtp_contract = _read_json(mtp_contract_path, "MTP contract")
        mtp.verify_contract(mtp_contract)
    else:
        mtp_contract = mtp.build_contract(
            direct_contract=contract,
            launch_plan=launch_plan,
            main_receipt=main_receipt,
            capture=capture,
            preparation_manifest=preparation_manifest,
            reader_abi_receipt=reader_abi,
            pure_mcg_backend_receipt=pure_backend,
            pure_mcg_preparation_receipt=pure_preparation,
        )
        _atomic_json(mtp_contract_path, mtp_contract)

    units_path = output_root / "mtp-work-units.json"
    if units_path.is_file():
        units = _read_json(units_path, "MTP work units")["units"]
    else:
        units = mtp.build_work_units(mtp_contract, experts_per_unit=args.experts_per_unit)
        _atomic_json(units_path, {"schema": "malaiwah.k6-mtp-work-units.v1", "units": units})
    state_path = output_root / "mtp-state" / "work-state.json"
    lock_path = output_root / "mtp-state" / ".lock"
    with _locked(lock_path):
        if not state_path.is_file():
            _atomic_json(state_path, mtp.initial_state(mtp_contract, units))

    worker = args.worker
    while True:
        with _locked(lock_path):
            state = _read_json(state_path, "MTP state")
            mtp.verify_state(mtp_contract, state)
            claim = state["active"].get(worker)
            if claim is None:
                if not state["pending"]:
                    break
                state, claim = mtp.claim_next(mtp_contract, state, worker)
                _atomic_json(state_path, state)
        unit = state["units"][claim["work_unit_sha256"]]
        started = time.monotonic()
        unit_receipt = mtp.encode_work_unit(
            contract=mtp_contract,
            direct_contract=contract,
            unit=unit,
            source=source,
            capture=capture,
            backend=backend,
            output_root=output_root,
            device=args.device,
        )
        elapsed = time.monotonic() - started
        telemetry = _seal(
            {
                "schema": mtp.MTP_TELEMETRY_SCHEMA,
                "contract_sha256": mtp_contract["contract_sha256"],
                "work_unit_sha256": unit["work_unit_sha256"],
                "direct_work_unit_receipt_sha256": unit_receipt["receipt_sha256"],
                "layer": mtp.MTP_LAYER,
                "expert_start": unit["expert_start"],
                "expert_stop": unit["expert_stop"],
                "worker_id": worker,
                "elapsed_seconds": elapsed,
                "device": args.device,
            },
            "receipt_sha256",
        )
        _atomic_json(
            output_root / "mtp-telemetry" / f"{unit['work_unit_sha256']}.json", telemetry
        )
        with _locked(lock_path):
            state = _read_json(state_path, "MTP state")
            state = mtp.complete(
                mtp_contract, state, worker_id=worker, unit_receipt=unit_receipt
            )
            _atomic_json(state_path, state)
        print(
            f"MTP unit {unit['expert_start']}..{unit['expert_stop']} done in {elapsed:.0f}s",
            flush=True,
        )

    with _locked(lock_path):
        state = _read_json(state_path, "MTP state")
        packed, adapter = mtp.seal_mtp_layer(
            contract=mtp_contract,
            direct_contract=contract,
            launch_plan=launch_plan,
            state=state,
            output_root=output_root,
            backend_identity=backend.identity(),
        )
    print(f"MTP adapter receipt sealed: {adapter['receipt_sha256']}")
    return 0


# --------------------------------------------------------------------------- #
# subcommand: materialize                                                      #
# --------------------------------------------------------------------------- #

def cmd_materialize(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    output_root = Path(args.output_root).resolve()
    bits = _bits_for_profile(args.profile, output_root)

    from quant_pipeline.campaign import glm53_direct_k4 as direct
    from quant_pipeline.campaign.glm53_uniform_k4 import MAIN_ROUTED_LAYERS
    from quant_pipeline.checkpoint.glm53_mcg_materializer import materialize_checkpoint

    contract = _read_json(output_root / "contract.json", "direct contract")
    direct.verify_contract(contract)
    inventory = _read_json(output_root / "inventory.json", "inventory")
    mtp_adapter = _read_json(
        output_root / "mtp-adapter-receipt.json", "MTP adapter receipt (run mtp first)"
    )
    layer_receipts = [
        _read_json(output_root / "layers" / f"layer-{layer:03d}.json", f"layer {layer} receipt")
        for layer in MAIN_ROUTED_LAYERS
    ]
    reader_abi = _author_reader_abi_receipt(
        output_root=output_root, contract=contract, bits=bits
    )
    plan_path = output_root / "materialization-plan.json"
    if plan_path.is_file():
        plan = _read_json(plan_path, "materialization plan")
    else:
        plan = direct.build_materialization_plan(
            contract=contract,
            inventory=inventory,
            main_layer_receipts=layer_receipts,
            mtp_adapter_receipt=mtp_adapter,
            reader_abi_receipt=reader_abi,
        )
        _atomic_json(plan_path, plan)
    final = materialize_checkpoint(
        plan=plan,
        contract=contract,
        inventory=inventory,
        mtp_adapter_receipt=mtp_adapter,
        packed_root=output_root,
        source_root=Path(args.bf16).resolve(),
        output_root=Path(args.checkpoint).resolve(),
    )
    receipt = _read_json(
        Path(args.checkpoint).resolve() / "materialization-receipt.json",
        "materialization receipt",
    )
    print(
        json.dumps(
            {
                "materialization_receipt_sha256": receipt["receipt_sha256"],
                "output_logical_bytes": receipt["output_logical_bytes"],
                "bits": receipt["bits"],
                "complete": receipt["complete"],
                "verified": bool(final),
            },
            sort_keys=True,
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# subcommand: rehearse (P0)                                                    #
# --------------------------------------------------------------------------- #

def cmd_rehearse(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    shapley_root = Path(args.shapley_root).resolve()
    out_path = Path(args.output).resolve()
    _require_codec_closure(shapley_root, out_path.parent)

    import torch

    from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec
    from quant_pipeline.evaluation.glm53_packed_k4_reader import decode_choice_hf
    from quant_pipeline.normalization.artifact_v31 import tensor_sha256

    extension = _find_extension(Path(args.exllama_root).resolve(), args.extension)
    numeric_core = _numeric_core(shapley_root, args.numeric_core)
    adapter = Exl3MCGCodec(
        source_root=shapley_root,
        numeric_core=numeric_core,
        extension=extension,
        device=args.device,
        sigma_reg=0.025,
    )

    def _roundtrip(weight: "torch.Tensor", bits: int, unit_id: str) -> "tuple[bool, float]":
        n, k = weight.shape
        covariance = torch.eye(k, dtype=torch.float32, device=args.device)
        suh = (torch.randint(0, 2, (k,), dtype=torch.int8).float() * 2 - 1).to(args.device)
        svh = (torch.randint(0, 2, (n,), dtype=torch.int8).float() * 2 - 1).to(args.device)
        started = time.monotonic()
        candidate = adapter.encode_candidates(
            unit_id=unit_id,
            weight_hf=weight.to(args.device, torch.float32),
            covariance=covariance,
            bits=(bits,),
            input_vector=suh,
            output_vector=svh,
        )[bits]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        decoded = decode_choice_hf(
            candidate.packed.cpu(), suh.cpu(), svh.cpu(), bits=bits
        ).to(torch.float16)
        reference = candidate.reconstructed.detach().to("cpu", torch.float16)
        return tensor_sha256(decoded) == tensor_sha256(reference), elapsed

    torch.manual_seed(0xCBAC1FED & 0x7FFFFFFF)

    # 1) K6 roundtrip over fixture-derived matrices (cast BF16->F32, 128-aligned)
    fixture_matrices: List["torch.Tensor"] = []
    fixture_root = Path(args.fixture).resolve() if args.fixture else None
    if fixture_root is not None:
        from safetensors import safe_open

        shards = sorted(fixture_root.glob("*.safetensors"))
        if not shards:
            raise _fail(f"fixture has no safetensors shards: {fixture_root}")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    value = handle.get_tensor(name)
                    if value.ndim == 2 and value.shape[0] % 128 == 0 and value.shape[1] % 128 == 0:
                        fixture_matrices.append(value.to(torch.float32))
                    if len(fixture_matrices) >= args.fixture_matrices:
                        break
            if len(fixture_matrices) >= args.fixture_matrices:
                break
    if not fixture_matrices:
        # architecturally-complete tiny random fallback (fixture dims may not be
        # 128-aligned; the codec hard-requires K,N % 128 == 0)
        fixture_matrices = [torch.randn(256, 256) * 0.02 for _ in range(4)]
        fixture_source = "synthetic-256x256 (no 128-aligned 2-D fixture tensor found)"
    else:
        fixture_source = str(fixture_root)
    k6_exact = True
    for index, weight in enumerate(fixture_matrices):
        exact, _ = _roundtrip(weight, 6, f"L0.E{index}.probe")
        k6_exact = k6_exact and exact

    # 2) K8 codec probe (expected to be refused by the pinned adapter until the
    #    K6K8 codec extension lands; the receipt records either outcome)
    k8_probe: Dict[str, Any] = {"attempted": True}
    try:
        exact, _ = _roundtrip(torch.randn(256, 256) * 0.02, 8, "L0.E0.k8probe")
        k8_probe.update({"admitted": True, "encode_decode_exact": bool(exact)})
    except Exception as error:  # noqa: BLE001 - receipt captures the refusal
        k8_probe.update(
            {
                "admitted": False,
                "encode_decode_exact": False,
                "error": f"{type(error).__name__}: {error}",
                "note": "K8 requires the declared codec extension (RUNBOOK P2); "
                "red probe descopes K6K8, K6 continues",
            }
        )

    # 3) timing bench on full-size synthetic matrices (4096x2048 down-proj shape)
    #    --bench-bits 8 re-prices the K8 campaign (trellis edge count grows with
    #    the rate, so K8 seconds/matrix must be measured, not assumed == K6).
    bench_bits = int(args.bench_bits)
    bench_times: List[float] = []
    bench_exact = True
    for index in range(args.bench_full_size_matrices):
        weight = torch.randn(4096, 2048) * 0.02
        exact, elapsed = _roundtrip(weight, bench_bits, f"L1.E{index}.bench")
        bench_exact = bench_exact and exact
        bench_times.append(elapsed)
        print(f"bench K{bench_bits} {index + 1}/{args.bench_full_size_matrices}: {elapsed:.2f}s", flush=True)
    if bench_bits == 6:
        k6_exact = k6_exact and bench_exact
    spm = sum(bench_times) / max(1, len(bench_times))
    est_hours = 37152 * spm / 4 / 3600

    receipt = _seal(
        {
            "schema": "malaiwah.glm53-k6-rehearsal-receipt.v1",
            "k6_roundtrip_exact": bool(k6_exact),
            "k6_roundtrip_scope": (
                "Exl3MCGCodec encode -> packed trellis -> independent "
                "decode_choice_hf, fp16-payload bit-exact"
            ),
            "fixture_source": fixture_source,
            "fixture_matrix_count": len(fixture_matrices),
            "k8_probe": k8_probe,
            "bench_bits": bench_bits,
            "bench_roundtrip_exact": bool(bench_exact),
            f"seconds_per_full_size_matrix_k{bench_bits}": spm,
            "bench_matrix_count": len(bench_times),
            "bench_seconds": bench_times,
            "projected_main_plus_mtp_encode_hours_4gpu": est_hours,
            "device": args.device,
        },
        "receipt_sha256",
    )
    _atomic_json(out_path, receipt)
    print(json.dumps({k: receipt[k] for k in (
        "k6_roundtrip_exact", f"seconds_per_full_size_matrix_k{bench_bits}",
        "projected_main_plus_mtp_encode_hours_4gpu")}, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# subcommand: shared-vector-ab (operator directive 2)                          #
# --------------------------------------------------------------------------- #

def cmd_shared_vector_ab(args: argparse.Namespace) -> int:
    _import_pipeline(Path(args.pipeline_root))
    shapley_root = Path(args.shapley_root).resolve()
    out_path = Path(args.output).resolve()
    _require_codec_closure(shapley_root, out_path.parent)

    import torch

    from quant_pipeline.campaign import glm53_direct_k4 as direct
    from quant_pipeline.campaign.glm53_mcg_preparation import _sign
    from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

    inventory_path = Path(args.output_root or "").joinpath("inventory.json") if args.output_root else None
    extension = _find_extension(Path(args.exllama_root).resolve(), args.extension)
    numeric_core = _numeric_core(shapley_root, args.numeric_core)
    adapter = Exl3MCGCodec(
        source_root=shapley_root,
        numeric_core=numeric_core,
        extension=extension,
        device=args.device,
        sigma_reg=0.025,
    )
    bf16_root = Path(args.bf16).resolve()
    capture_root = Path(args.calibration).resolve() / "main-ep4-full"
    if inventory_path is not None and inventory_path.is_file():
        inventory = _read_json(inventory_path, "inventory")
        source = direct.Glm53BF16Source(inventory, bf16_root, verify_shards=False)
        load = lambda layer, expert: source.load_projection(layer, expert, "down_proj")  # noqa: E731
    else:
        # inventory-free load path so the A/B can run before the contract stage
        from safetensors import safe_open

        index = _read_json(bf16_root / "model.safetensors.index.json", "BF16 index")

        def load(layer: int, expert: int) -> "torch.Tensor":
            name = direct.tensor_name(layer, expert, "down_proj")
            shard = index["weight_map"][name]
            with safe_open(bf16_root / shard, framework="pt", device="cpu") as handle:
                return handle.get_tensor(name).contiguous()

    seed = _transform_seed(
        (Path(args.output_root) if args.output_root else out_path.parent)
        / "transform-seed.json"
    )
    from quant_pipeline.campaign.glm53_direct_k4 import INTERMEDIATE_SIZE

    per_layer: Dict[str, Dict[str, float]] = {}
    for layer in _parse_layers(args.layers):
        capture = direct.Glm53CaptureView(capture_root, layer, verify_hashes=False)
        deltas: List[float] = []
        for expert in args.experts:
            weight = load(layer, expert).to(args.device, torch.float32)
            n, k = weight.shape  # down: (HIDDEN, INTERMEDIATE)
            rows = capture.routed_rows(expert, "fit")
            if rows.rows == 0:
                continue
            take = rows.row_indices[: args.max_rows]
            import numpy as np

            hidden_np = np.ascontiguousarray(capture.hidden_u16[take])
            hidden = (
                torch.from_numpy(hidden_np.view(np.uint16))
                .view(torch.bfloat16)
                .to(args.device, torch.float32)
            )
            # replayed down-proj INPUT activations: silu(gate)*up is what the
            # down path actually sees, so replay through the real gate/up triplet
            triplet_gate = _load_named(bf16_root, direct.tensor_name(layer, expert, "gate_proj"))
            triplet_up = _load_named(bf16_root, direct.tensor_name(layer, expert, "up_proj"))
            gate_out = hidden @ triplet_gate.to(args.device, torch.float32).T
            up_out = hidden @ triplet_up.to(args.device, torch.float32).T
            x = torch.nn.functional.silu(gate_out) * up_out  # (rows, INTERMEDIATE)
            covariance = (x.T @ x) / max(1, x.shape[0])
            covariance += 0.025 * torch.eye(k, device=args.device)
            reference = x @ weight.T
            svh = _sign(n, seed, layer, expert, "down_proj", "svh").to(args.device)
            results = {}
            for label, domain in (
                ("expert_private", (layer, expert, "down_proj", "suh")),
                ("layer_shared", (layer, "down_proj", "suh", "shared")),
            ):
                suh = _sign(k, seed, *domain).to(args.device)
                candidate = adapter.encode_candidates(
                    unit_id=f"L{layer}.E{expert}.down_proj",
                    weight_hf=weight,
                    covariance=covariance,
                    bits=(6,),
                    input_vector=suh,
                    output_vector=svh,
                )[6]
                approx = x @ candidate.reconstructed.to(args.device, torch.float32).T
                results[label] = float(
                    torch.linalg.norm(approx - reference)
                    / torch.linalg.norm(reference).clamp_min(1e-12)
                )
            deltas.append(results["layer_shared"] - results["expert_private"])
            print(
                f"layer {layer} expert {expert}: private={results['expert_private']:.6f} "
                f"shared={results['layer_shared']:.6f}",
                flush=True,
            )
        per_layer[str(layer)] = {
            "mean_delta_relative_output_error": sum(deltas) / max(1, len(deltas)),
            "experts_measured": float(len(deltas)),
        }
    worst = max(
        (row["mean_delta_relative_output_error"] for row in per_layer.values()),
        default=float("inf"),
    )
    adopted = (
        "layer_shared_deviation" if worst <= args.threshold else "expert_private_upstream"
    )
    receipt = _seal(
        {
            "schema": "malaiwah.glm53-k6-down-suh-topology-ab.v1",
            "per_layer_delta": per_layer,
            "threshold": args.threshold,
            "worst_mean_delta": worst,
            "adopted_topology": adopted,
            "note": "operator directive 2 (DECISIONS.md); shared adoption is a "
            "disclosed recipe deviation and requires runtime hoist support "
            "(out of scope for this campaign's serving)",
        },
        "receipt_sha256",
    )
    _atomic_json(out_path, receipt)
    print(json.dumps({"adopted_topology": adopted, "worst_mean_delta": worst}))
    return 0


def _load_named(bf16_root: Path, name: str) -> Any:
    from safetensors import safe_open

    index = _read_json(bf16_root / "model.safetensors.index.json", "BF16 index")
    shard = index["weight_map"][name]
    with safe_open(bf16_root / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name).contiguous()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _common(parser: argparse.ArgumentParser, *, gpu: bool = True) -> None:
    parser.add_argument("--pipeline-root", required=True, help="patched pipeline tree")
    parser.add_argument("--shapley-root", help="public ShapleyMCG @ 9d83e7d0 (r7_encoder closure)")
    parser.add_argument("--exllama-root", help="exllamav3 @ c5d9c657 with built extension")
    parser.add_argument("--extension", help="explicit extension .so (else discovered)")
    parser.add_argument("--numeric-core", help="explicit numeric core file (default r7_encoder/r10_codec.py)")
    if gpu:
        parser.add_argument("--device", default="cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k6_driver.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rehearse", help="P0 fixture/codec roundtrip + K8 probe + bench")
    _common(p)
    p.add_argument("--fixture", help="GLM-5.3-Flash-0.1B-A0.1B root")
    p.add_argument("--fixture-matrices", type=int, default=4)
    p.add_argument("--bench-full-size-matrices", type=int, default=24)
    p.add_argument("--bench-bits", type=int, default=6, choices=(6, 8),
                   help="rate for the full-size timing bench (8 re-prices the K8 campaign)")
    p.add_argument("--probe-k8", action="store_true", help="kept for interface parity; the probe always runs and records its outcome")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_rehearse)

    p = sub.add_parser("contract", help="inventory -> plan -> preparation -> contract -> work state")
    _common(p)
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--bf16", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--inventory", help="adopt an upstream sealed inventory doc verbatim "
                   "(REQUIRED to bind brandonmusic's published captures)")
    p.add_argument("--k4-plan", help="sealed glm53_uniform_k4 launch plan "
                   "(default <output-root>/k4-launch-plan.json; built fresh if absent)")
    p.add_argument("--k4-state", help="sealed k6_authorized K4 state receipt "
                   "(default <output-root>/k4-authorized-state.json; NEVER fabricated)")
    p.add_argument("--reuse-gate-up-from", help="K6K8 gate/up verified-reuse source")
    p.add_argument("--verify-capture-hashes", action="store_true", default=False)
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser("prepare", help="(aux) build layer preparations for a subset")
    _common(p)
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--bf16", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--layers", required=True, help="e.g. 3-12 or 3,20,44")
    p.add_argument("--verify-capture-hashes", action="store_true", default=False)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("encode-worker", help="claim/encode/seal loop for one GPU")
    _common(p)
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--worker", required=True, help="h200-0..3 (or b200-0..3)")
    p.add_argument("--bf16", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--prune-hessians-after-layer-seal", action="store_true")
    p.add_argument("--max-inflight-experts", type=int, default=28)
    p.add_argument("--verify-capture-hashes", action="store_true", default=False)
    p.add_argument(
        "--overlap-seal",
        action="store_true",
        default=False,
        help="opt-in: seal batch N on one background thread while the GPU "
        "encodes batch N+1 (identical pipeline seal calls, rescheduled; "
        "default OFF preserves today's serial encode-then-seal behavior)",
    )
    p.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="stop after completing N work units (default: run until no "
        "pending work remains)",
    )
    p.set_defaults(func=cmd_encode_worker)

    p = sub.add_parser("seal-main", help="author the sealed main receipt")
    p.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT", ""))
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--output-root", required=True)
    p.set_defaults(func=cmd_seal_main)

    p = sub.add_parser("release-dead-claims", help="requeue dead worker claims")
    p.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT", ""))
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--output-root", required=True)
    p.set_defaults(func=cmd_release_dead_claims)

    p = sub.add_parser("mtp", help="MTP45 contract -> encode -> telemetry -> adapter seal")
    _common(p)
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--bf16", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--worker", default="h200-0")
    p.add_argument("--experts-per-unit", type=int, default=18)
    p.add_argument("--verify-capture-hashes", action="store_true", default=False)
    p.set_defaults(func=cmd_mtp)

    p = sub.add_parser("materialize", help="reader-ABI receipt -> plan -> shards -> receipt")
    p.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT", ""))
    p.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8"))
    p.add_argument("--output-root", required=True)
    p.add_argument("--bf16", required=True)
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=cmd_materialize)

    p = sub.add_parser("shared-vector-ab", help="down_suh shared-vs-private A/B")
    _common(p)
    p.add_argument("--bf16", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--layers", default="3,20,44")
    p.add_argument("--experts", type=int, nargs="+", default=[0, 96, 191, 287])
    p.add_argument("--max-rows", type=int, default=8192)
    p.add_argument("--threshold", type=float, default=1e-4)
    p.add_argument("--output-root", help="campaign root (reuses its transform seed)")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_shared_vector_ab)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "pipeline_root", ""):
        pass
    elif args.command in {"seal-main", "release-dead-claims", "materialize"}:
        raise _fail("--pipeline-root (or QP_PIPELINE_ROOT) is required")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
