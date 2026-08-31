#!/usr/bin/env python3
"""EP8 packed-K6 student logit capture over the 25 sealed final windows.

Adapted from brandonmusic's capture_glm53_packed_k4_student_logits_ep4.py with
three disclosed changes only:

  * EP size is env-driven (``QP_GLM53_EP_SIZE``, default 8 here) via patch
    0007; upstream EP4-on-B200-192GB peaks 184.8 GiB/rank and does not fit
    H200-141GB.  The reconstructed-expert install is exact under any divisor
    of 288, so the logits are unchanged.
  * The campaign evidence chain (packed root, contract, inventory, MTP adapter
    receipt) is derived from the materialized checkpoint's own
    materialization-receipt.json instead of five separate CLI paths.
  * ``--cold-run N`` labels the capture receipt so the five-cold-run receipt
    can prove five distinct captures; ``--emit-reference-panel`` (run 1) dumps
    the decoded reference parity panel for the TP4 runtime qualification with
    the upstream schema string VERBATIM and predeclared tolerances.

Launch:  QP_GLM53_EP_SIZE=8 torchrun --nproc-per-node=8 student_capture.py ...
Everything else (eager attention, tf32 off, use_cache off, fp32 stored logits,
non-routed parameters untouched, receipt schema
quant-pipeline.glm53-logit-capture.v1) is upstream-verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RELEASED_ARCHITECTURE = "Glm5NextForConditionalGeneration"
RELEASED_MODEL_TYPE = "glm5_next"
RELEASED_TEXT_MODEL_TYPE = "glm5_next_text"
REVISION = re.compile(r"[0-9a-f]{40}")
REFERENCE_PANEL_SCHEMA = "quant-pipeline.glm53-decoded-k4-tp2-reference-panel.v1"
REFERENCE_MAX_ABS_TOLERANCE = "0.5"
REFERENCE_MEAN_ABS_TOLERANCE = "0.005"


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"student_capture: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise _fail(f"no quant_pipeline package under {pipeline_root}")


def _import_pipeline(pipeline_root: Optional[str]) -> None:
    if pipeline_root:
        src = str(_pipeline_src(Path(pipeline_root)))
    else:
        env_root = os.environ.get("QP_PIPELINE_ROOT")
        if env_root:
            src = str(_pipeline_src(Path(env_root)))
        else:
            try:
                import quant_pipeline  # noqa: F401

                return
            except ImportError:
                raise _fail(
                    "quant_pipeline not importable: pass --pipeline-root, set "
                    "QP_PIPELINE_ROOT, or export PYTHONPATH to the patched tree"
                )
    if src not in sys.path:
        sys.path.insert(0, src)


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sealed_json(path: Path, schema: str, field: str) -> Dict[str, Any]:
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes

    value = _read_json(path, schema)
    body = dict(value)
    seal = body.pop(field, None)
    if (
        value.get("example_only") is True
        or value.get("schema") != schema
        or seal != sha256_bytes(canonical_json(body))
    ):
        raise _fail(f"invalid sealed {schema}: {path}")
    return value


def _find_token_panel_receipt(candidates: List[Path]) -> Path:
    """Locate the sealed token-panel receipt by schema, not by filename."""

    from quant_pipeline.evaluation.glm53_logits import PANEL_RECEIPT_SCHEMA

    seen: List[Path] = []
    for root in candidates:
        if root is None or not root.exists():
            continue
        pool = [root] if root.is_file() else sorted(root.glob("**/*.json"))
        for path in pool:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("schema") == PANEL_RECEIPT_SCHEMA:
                return path
            seen.append(path)
    raise _fail(
        "sealed token-panel receipt not found; searched "
        + ", ".join(str(item) for item in candidates)
        + " - pass --token-panel explicitly (calibration/panel-v1 receipt)"
    )


def _distributed_environment(expected_world: int) -> "tuple[int, int, int]":
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            f"launch with torchrun --nproc-per-node {expected_world} "
            f"(QP_GLM53_EP_SIZE={expected_world})"
        ) from error
    if world_size != expected_world or rank not in range(world_size):
        raise RuntimeError(
            f"world size {world_size} differs from QP_GLM53_EP_SIZE={expected_world}"
        )
    return rank, local_rank, world_size


def _tensor_device_type(value: Any) -> str:
    local = value.to_local() if hasattr(value, "to_local") else value
    return str(local.device.type)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="materialized ckpt (its receipt names the packed root); "
                             "required unless --surface dione")
    parser.add_argument("--bf16", type=Path, required=True,
                        help="official BF16 checkpoint (non-routed source)")
    parser.add_argument("--teacher", type=Path, required=True,
                        help="teacher final-window tree (panel receipt search root)")
    parser.add_argument("--profile", required=True, choices=("k6", "k8", "k6k8", "dione"))
    parser.add_argument("--cold-run", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emit-reference-panel", type=Path,
                        help="run 1 only: decoded reference parity panel output")
    parser.add_argument("--token-panel", type=Path,
                        help="explicit sealed token-panel receipt path")
    parser.add_argument("--pipeline-root", help="patched pipeline tree (else PYTHONPATH)")
    parser.add_argument("--attention-backend", default="eager")
    parser.add_argument("--roles", default="final")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sealed plan and exit without GPUs")
    # --surface dione: score a third-party 0xSero/Dione selective-EXL3 TP4
    # checkpoint (unsealed source, DISCLOSED) with the identical capture.
    parser.add_argument("--surface", choices=("packed", "dione"), default="packed",
                        help="packed = our sealed campaign checkpoints (default); "
                             "dione = 0xSero selective-EXL3 TP4 snapshot")
    parser.add_argument("--dione-root", type=Path,
                        help="dione mode: downloaded HF snapshot root")
    parser.add_argument("--dione-repo",
                        help="dione mode: HF repo id, e.g. 0xSero/GLM-5.3-Flash-EXL3-Q4")
    parser.add_argument("--dione-revision",
                        help="dione mode: immutable 40-hex repo commit of the snapshot")
    parser.add_argument("--verify-nonrouted", choices=("full", "sample", "names"),
                        default="sample",
                        help="dione mode: retained-tensor byte verification vs --bf16")
    parser.add_argument("--skip-shard-hashes", action="store_true",
                        help="dione mode: accept a snapshot without the "
                             "dione-shards-verified.json marker (disclosed)")
    args = parser.parse_args()
    _import_pipeline(args.pipeline_root)

    if args.surface == "dione":
        if args.profile != "dione":
            raise _fail("--surface dione requires --profile dione")
        return _dione_main(args)
    if args.profile == "dione":
        raise _fail("--profile dione requires --surface dione")
    if args.checkpoint is None:
        raise _fail("--checkpoint is required for the packed surfaces")

    if args.profile == "k6k8":
        expected_bits = None  # bits come from the contract; k6k8 uses malaiwah schemas
    else:
        expected_bits = {"k6": 6, "k8": 8}[args.profile]

    from quant_pipeline.core.artifacts import (
        canonical_json,
        prepare_empty_destination,
        sha256_bytes,
        sha256_file,
        write_json,
    )
    from quant_pipeline.campaign.glm53_direct_k4 import (
        contract_bits,
        contract_schema_for_bits,
    )
    from quant_pipeline.campaign.glm53_mtp_k4 import _mtp_adapter_schema
    from quant_pipeline.evaluation.glm53_logits import load_panel_windows
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader_module
    from quant_pipeline.evaluation.glm53_packed_k4_reader import (
        EP_SIZE,
        MAIN_ROUTED_LAYERS,
        install_local_main_experts,
        load_complete_surface,
        reader_identity,
        stored_encoder_closure,
    )

    checkpoint_root = args.checkpoint.resolve()
    materialization = _read_json(
        checkpoint_root / "materialization-receipt.json",
        "materialization receipt (materialize the checkpoint first)",
    )
    packed_root = Path(str(materialization.get("packed_root", ""))).resolve()
    if not packed_root.is_dir():
        raise _fail(
            f"packed root from the materialization receipt is absent: {packed_root}"
        )
    inventory_path = packed_root / "inventory.json"
    contract_path = packed_root / "contract.json"
    mtp_receipt_path = packed_root / "mtp-adapter-receipt.json"

    inventory = _sealed_json(
        inventory_path, "quant-pipeline.glm-release-inventory.v1", "inventory_sha256"
    )
    model_revision = str(inventory.get("model_revision", ""))
    if REVISION.fullmatch(model_revision) is None:
        raise _fail("inventory model revision is not an immutable 40-hex commit")
    if inventory.get("seal_mode") != "full-shard-sha256":
        raise _fail("student capture requires the exact full-hash BF16 inventory")
    if materialization.get("source_inventory_sha256") != inventory["inventory_sha256"]:
        raise _fail("materialized checkpoint binds a different inventory")

    raw_contract = _read_json(contract_path, "direct contract")
    bits = int(raw_contract.get("rate", {}).get("bits", -1))
    if expected_bits is not None and bits != expected_bits:
        raise _fail(f"profile {args.profile} expects K{expected_bits}, contract is K{bits}")
    contract = _sealed_json(
        contract_path, contract_schema_for_bits(bits), "contract_sha256"
    )
    if contract_bits(contract) != bits:
        raise _fail("packed student contract rate differs")
    if contract.get("inventory_sha256") != inventory["inventory_sha256"]:
        raise _fail("direct MCG contract targets another BF16 inventory")
    mtp_adapter = _sealed_json(
        mtp_receipt_path,
        # K4/K6: upstream family; K8: malaiwah.* - one helper, shared with the
        # sealer (glm53_mtp_k4) and the reader (load_complete_surface)
        _mtp_adapter_schema(bits),
        "receipt_sha256",
    )

    model_root = args.bf16.resolve()
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise _fail("official model requires config.json and model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    if (
        config.get("architectures") != [RELEASED_ARCHITECTURE]
        or config.get("model_type") != RELEASED_MODEL_TYPE
        or text_config.get("model_type") != RELEASED_TEXT_MODEL_TYPE
        or text_config.get("num_hidden_layers") != 45
        or text_config.get("num_nextn_predict_layers") != 1
        or text_config.get("n_routed_experts") != 288
        or text_config.get("hidden_size") != 4096
        or text_config.get("moe_intermediate_size") != 2048
    ):
        raise _fail("official GLM5Next main/MTP geometry differs")
    if inventory.get("config_sha256") != sha256_file(config_path) or inventory.get(
        "index_sha256"
    ) != sha256_file(index_path):
        raise _fail("BF16 inventory does not bind the local config/index")

    surface = load_complete_surface(
        root=packed_root, contract=contract, mtp_adapter_receipt=mtp_adapter
    )
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    panel_path = _find_token_panel_receipt(
        [
            args.token_panel,
            args.teacher.resolve(),
            args.teacher.resolve().parent / "panel-v1",
            args.out.resolve().parent.parent / "calibration" / "panel-v1",
        ]
    )
    panel_receipt, _, windows = load_panel_windows(
        panel_path, roles=roles, vocab_size=int(text_config["vocab_size"])
    )
    identity = reader_identity(
        Path(reader_module.__file__).resolve(), Path(__file__).resolve(), bits=bits
    )
    checkpoint_identity = sha256_bytes(
        canonical_json(
            {
                "schema": f"quant-pipeline.glm53-packed-k{bits}-student-identity.v1",
                "inventory_sha256": inventory["inventory_sha256"],
                "contract_sha256": surface.contract_sha256,
                "main_layer_receipt_sha256": list(surface.main_layer_receipt_sha256),
                "mtp_adapter_receipt_sha256": surface.mtp_adapter_receipt_sha256,
                "mtp_pack_receipt_sha256": surface.mtp_pack_receipt_sha256,
                "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
                "bits": bits,
                "codebook": "MCG",
                "nonrouted_policy": "official_source_native",
            }
        )
    )
    plan = {
        "schema": f"quant-pipeline.glm53-packed-k{bits}-student-logit-capture-plan.v1",
        "model": str(model_root),
        "model_revision": model_revision,
        "inventory_sha256": inventory["inventory_sha256"],
        "contract_sha256": surface.contract_sha256,
        "checkpoint_identity_sha256": checkpoint_identity,
        "materialization_receipt_sha256": materialization.get("receipt_sha256"),
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
        "roles": list(roles),
        "windows": len(windows),
        "prediction_positions": sum(window.prediction_positions for window in windows),
        "parallelism": (
            f"expert_parallel_world_size_{EP_SIZE}_contiguous_"
            f"{288 // EP_SIZE}_experts_per_rank"
        ),
        "ep_size_deviation": {
            "upstream": 4,
            "actual": EP_SIZE,
            "disclosed": True,
            "reason": "EP4 decoded-BF16 peaks 184.8 GiB/rank on B200-192GB; H200-141GB needs EP8",
        },
        "reader_mode": identity["mode"],
        "final_tp2_serving_kernel": False,
        "cold_run": args.cold_run,
        "main_routed_policy": (
            f"decode_hash_verified_packed_k{bits}_mcg_to_bf16_local_ep_parameters"
        ),
        "mtp_policy": "complete_and_receipt_required_but_not_executed_by_standard_logits",
        "nonrouted_policy": "untouched_official_checkpoint_parameters",
        "stored_logits_dtype": "float32",
        "output": str(args.out.resolve()),
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    rank, local_rank, world_size = _distributed_environment(EP_SIZE)
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, __version__ as transformers_version
    from transformers.distributed.configuration_utils import DistributedConfig

    if tuple(int(part) for part in transformers_version.split(".")[:2]) < (5, 16):
        raise RuntimeError("packed GLM5Next EP reader requires transformers>=5.16")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != EP_SIZE:
        raise RuntimeError(f"initialized process group is not world size {EP_SIZE}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    output_root = args.out.resolve()
    if rank == 0:
        prepare_empty_destination(output_root)
        (output_root / "logits").mkdir()
        write_json(output_root / "plan.json", plan | {"dry_run": False})
        write_json(output_root / "reader-identity.json", identity)
    dist.barrier()

    load_started = time.monotonic()
    distributed = DistributedConfig(
        tp_size=world_size, fsdp_size=1, pp_size=1, enable_expert_parallel=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_root,
        dtype=torch.bfloat16,
        distributed_config=distributed,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    non_cuda = sorted(
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if _tensor_device_type(value) != "cuda"
    )
    if non_cuda:
        raise RuntimeError(f"student is not fully GPU resident: {non_cuda[:8]}")
    versions_before = {name: int(value._version) for name, value in model.named_parameters()}
    closure = None
    if rank == 0:
        closure = stored_encoder_closure(
            surface,
            layer=MAIN_ROUTED_LAYERS[0],
            expert=0,
            projection="gate_proj",
            device=torch.device("cuda", local_rank),
        )
    install_started = time.monotonic()
    install = install_local_main_experts(
        model, surface, rank=rank, device=torch.device("cuda", local_rank)
    )
    torch.cuda.synchronize(local_rank)
    versions_after = {name: int(value._version) for name, value in model.named_parameters()}

    def _is_mutated_routed_parameter(name: str) -> bool:
        return any(
            f"language_model.layers.{layer}.mlp.experts.{suffix}" in name
            for layer in MAIN_ROUTED_LAYERS
            for suffix in ("gate_up_proj", "down_proj")
        )

    unexpected = sorted(
        name
        for name, version in versions_before.items()
        if versions_after[name] != version and not _is_mutated_routed_parameter(name)
    )
    if unexpected:
        raise RuntimeError(
            f"packed reader mutated non-routed official parameters: {unexpected[:8]}"
        )
    install.update(
        {
            "gpu": torch.cuda.get_device_name(local_rank),
            "load_seconds": install_started - load_started,
            "install_seconds": time.monotonic() - install_started,
            "allocated_bytes": int(torch.cuda.memory_allocated(local_rank)),
            "reserved_bytes": int(torch.cuda.memory_reserved(local_rank)),
            "parameter_version_nonrouted_unchanged": True,
        }
    )
    rank_installs: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(rank_installs, install)
    closures: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(closures, closure)
    if rank == 0:
        installed = sum(
            int(row["installed_matrix_count"]) for row in rank_installs if row
        )
        if installed != len(MAIN_ROUTED_LAYERS) * 288 * 3:
            raise RuntimeError(
                f"EP{EP_SIZE} rank installs do not close the complete main routed "
                f"matrix census ({installed})"
            )
        backend = {
            "schema": "quant-pipeline.glm53-packed-k4-offline-reader-backend.v1",
            "architecture": RELEASED_ARCHITECTURE,
            "model_revision": model_revision,
            "inventory_sha256": inventory["inventory_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity,
            "contract_sha256": surface.contract_sha256,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "packed_reader_abi_sha256": surface.packed_reader_abi_sha256,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "attention_backend": args.attention_backend,
            "parallelism": plan["parallelism"],
            "reader_mode": identity["mode"],
            "final_tp2_serving_kernel": False,
            "main_routed_runtime_dtype": f"bfloat16 decoded from packed K{bits} payload",
            "nonrouted_runtime_dtype": "official source dtype, untouched",
            "mtp_standard_logits_executed": False,
            "mtp_pack_receipt_sha256": surface.mtp_pack_receipt_sha256,
            "stored_encoder_closure": next(item for item in closures if item is not None),
            "rank_installs": rank_installs,
            "allow_tf32": False,
            "active_tp_plan": getattr(model, "_tp_plan", None),
            "active_ep_plan": getattr(model, "_ep_plan", None),
        }
        backend["backend_identity_sha256"] = sha256_bytes(canonical_json(backend))
        write_json(output_root / "backend.json", backend)
    else:
        backend = None
    backend_rows: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(backend_rows, backend)
    backend = next(item for item in backend_rows if item is not None)

    logit_records: List[Dict[str, Any]] = []
    capture_started = time.monotonic()
    input_device = torch.device("cuda", local_rank)
    for index, window in enumerate(windows):
        tokens = np.load(window.token_path, allow_pickle=False)
        mask = np.load(window.attention_mask_path, allow_pickle=False)
        causal_mask = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(input_device)
        attention_mask = (
            torch.from_numpy(np.asarray(mask, dtype=np.int64)).unsqueeze(0).to(input_device)
        )
        with torch.inference_mode():
            output_logits = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[:, :-1, :]
        selected = output_logits[0, torch.from_numpy(causal_mask).to(input_device)]
        if selected.shape != (window.prediction_positions, int(text_config["vocab_size"])):
            raise RuntimeError("student logits differ from sealed panel geometry")
        if rank == 0:
            stored = selected.float().cpu().contiguous()
            logit_path = (output_root / "logits" / f"window-{index:04d}.safetensors").resolve()
            save_file(
                {"logits": stored},
                logit_path,
                metadata={
                    "capture_role": "packed_student",
                    "student_label": f"uniform-k{bits}",
                    "cold_run": str(args.cold_run),
                    "window_id": window.window_id,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "checkpoint_identity_sha256": checkpoint_identity,
                    "runtime_reader_sha256": identity["runtime_reader_sha256"],
                    "backend_identity_sha256": backend["backend_identity_sha256"],
                },
            )
            logit_records.append(
                {
                    "window_id": window.window_id,
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "role": window.role,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "prediction_positions": window.prediction_positions,
                    "path": str(logit_path),
                    "bytes": logit_path.stat().st_size,
                    "sha256": sha256_file(logit_path),
                }
            )
            if index == 0 and args.emit_reference_panel is not None:
                panel_out = args.emit_reference_panel.resolve()
                panel_out.parent.mkdir(parents=True, exist_ok=True)
                indices = np.flatnonzero(causal_mask).astype(np.int64)
                save_file(
                    {
                        "input_ids": ids[0].cpu().contiguous(),
                        "attention_mask": attention_mask[0].cpu().contiguous(),
                        "prediction_indices": torch.from_numpy(indices).contiguous(),
                        "logits": stored,
                    },
                    panel_out,
                    metadata={
                        # schema string VERBATIM per upstream (identifies the
                        # ABI, not the rate) with predeclared tolerances
                        "schema": REFERENCE_PANEL_SCHEMA,
                        "max_abs_tolerance": REFERENCE_MAX_ABS_TOLERANCE,
                        "mean_abs_tolerance": REFERENCE_MEAN_ABS_TOLERANCE,
                        "attention_backend": args.attention_backend,
                        "bits": str(bits),
                        "window_id": window.window_id,
                        "checkpoint_identity_sha256": checkpoint_identity,
                        "runtime_reader_sha256": identity["runtime_reader_sha256"],
                        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
                    },
                )
                print(f"reference parity panel written: {panel_out}", flush=True)
        del ids, attention_mask, output_logits, selected
        dist.barrier()
    if rank == 0:
        receipt = {
            "schema": "quant-pipeline.glm53-logit-capture.v1",
            "capture_role": "packed_student",
            "cold_run": args.cold_run,
            "model_revision": model_revision,
            "checkpoint_identity_sha256": checkpoint_identity,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
            "backend_identity_sha256": backend["backend_identity_sha256"],
            "weight_dtype": f"EXL3/TR3 uniform-k{bits} offline-decoded to BF16",
            "logits_dtype": "float32",
            "kld_direction": "teacher_to_student",
            "prediction_positions": sum(window.prediction_positions for window in windows),
            "vocab_size": int(text_config["vocab_size"]),
            "logit_files": logit_records,
            "elapsed_seconds": time.monotonic() - capture_started,
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        write_json(output_root / "capture-receipt.json", receipt)
        print(json.dumps({"ok": True, "receipt_sha256": receipt["receipt_sha256"],
                          "cold_run": args.cold_run}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def _dione_main(args: argparse.Namespace) -> int:
    """--surface dione: identical EP8 teacher-forced capture over a 0xSero
    Dione selective-EXL3 TP4 snapshot.  DISCLOSED unsealed-source scoring:
    the snapshot ships no receipts/ABI, so the surface is decoded without
    seal verification (see dione_surface.SEAL_DISCLOSURE)."""

    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import dione_surface as ds

    from quant_pipeline.core.artifacts import (
        canonical_json,
        prepare_empty_destination,
        sha256_bytes,
        sha256_file,
        write_json,
    )
    from quant_pipeline.evaluation.glm53_logits import load_panel_windows
    from quant_pipeline.evaluation.glm53_packed_k4_reader import (
        EP_SIZE,
        MAIN_ROUTED_LAYERS,
    )

    if args.checkpoint is not None:
        raise _fail("--surface dione takes --dione-root, not --checkpoint")
    if args.dione_root is None or args.dione_revision is None:
        raise _fail("--surface dione requires --dione-root and --dione-revision")
    if REVISION.fullmatch(args.dione_revision) is None:
        raise _fail("--dione-revision must be the immutable 40-hex repo commit")
    surface = ds.load_dione_surface(
        args.dione_root,
        repo=args.dione_repo,
        revision=args.dione_revision,
        require_shard_hashes=not args.skip_shard_hashes,
    )
    bits = surface.bits

    model_root = args.bf16.resolve()
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise _fail("official model requires config.json and model.safetensors.index.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    if (
        config.get("architectures") != [RELEASED_ARCHITECTURE]
        or config.get("model_type") != RELEASED_MODEL_TYPE
        or text_config.get("model_type") != RELEASED_TEXT_MODEL_TYPE
        or text_config.get("num_hidden_layers") != 45
        or text_config.get("num_nextn_predict_layers") != 1
        or text_config.get("n_routed_experts") != 288
        or text_config.get("hidden_size") != 4096
        or text_config.get("moe_intermediate_size") != 2048
    ):
        raise _fail("official GLM5Next main/MTP geometry differs")
    if int(text_config["vocab_size"]) != surface.text_vocab_size:
        raise _fail("dione and official vocab sizes differ")

    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    panel_path = _find_token_panel_receipt(
        [
            args.token_panel,
            args.teacher.resolve(),
            args.teacher.resolve().parent / "panel-v1",
            args.out.resolve().parent.parent / "calibration" / "panel-v1",
        ]
    )
    panel_receipt, _, windows = load_panel_windows(
        panel_path, roles=roles, vocab_size=int(text_config["vocab_size"])
    )
    identity = ds.dione_reader_identity(Path(__file__).resolve(), bits=bits)
    checkpoint_identity = surface.checkpoint_identity_sha256()
    student_label = f"dione-exl3-k{bits}-tp4"
    plan = {
        "schema": ds.DIONE_PLAN_SCHEMA,
        "model": str(model_root),
        "dione_root": str(surface.root),
        "dione_repo": surface.repo,
        "dione_revision": surface.revision,
        "dione_format": surface.fmt,
        "source_repo": surface.source_repo,
        "source_revision": surface.source_revision,
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "exl3_manifest_sha256": surface.exl3_manifest_sha256,
        "shard_hash_verification": surface.shard_hash_verification,
        "bits": bits,
        "tp_slices_per_matrix": surface.tp_size,
        "checkpoint_identity_sha256": checkpoint_identity,
        "runtime_reader_sha256": identity["runtime_reader_sha256"],
        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
        "roles": list(roles),
        "windows": len(windows),
        "prediction_positions": sum(window.prediction_positions for window in windows),
        "parallelism": (
            f"expert_parallel_world_size_{EP_SIZE}_contiguous_"
            f"{288 // EP_SIZE}_experts_per_rank"
        ),
        "reader_mode": identity["mode"],
        "final_tp4_serving_kernel": False,
        "cold_run": args.cold_run,
        "main_routed_policy": (
            f"decode_dione_tp4_sliced_exl3_k{bits}_mcg_concat_to_bf16_local_ep_parameters"
        ),
        "mtp_policy": "native_bf16_in_dione_snapshot_but_not_executed_by_standard_logits",
        "nonrouted_policy": "untouched_official_checkpoint_parameters_verified_vs_dione_retained",
        "nonrouted_verification_mode": args.verify_nonrouted,
        "stored_logits_dtype": "float32",
        "seal_disclosure": ds.SEAL_DISCLOSURE,
        "output": str(args.out.resolve()),
        "dry_run": args.dry_run,
    }
    print(json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    rank, local_rank, world_size = _distributed_environment(EP_SIZE)
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, __version__ as transformers_version
    from transformers.distributed.configuration_utils import DistributedConfig

    if tuple(int(part) for part in transformers_version.split(".")[:2]) < (5, 16):
        raise RuntimeError("packed GLM5Next EP reader requires transformers>=5.16")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if dist.get_world_size() != EP_SIZE:
        raise RuntimeError(f"initialized process group is not world size {EP_SIZE}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()
    output_root = args.out.resolve()
    if rank == 0:
        prepare_empty_destination(output_root)
        (output_root / "logits").mkdir()
        write_json(output_root / "plan.json", plan | {"dry_run": False})
        write_json(output_root / "reader-identity.json", identity)
    dist.barrier()

    shards = ds.DioneShardReader(surface)
    nonrouted_record = None
    placement_audit = None
    if rank == 0:
        # both audits precede any install: fail before burning GPU-hours
        placement_audit = ds.audit_slice_placement(
            surface, shards, model_root, layer=MAIN_ROUTED_LAYERS[0], expert=0
        )
        nonrouted_record = ds.verify_nonrouted_tensors(
            surface, shards, model_root, mode=args.verify_nonrouted
        )
        write_json(output_root / "dione-placement-audit.json", placement_audit)
        write_json(output_root / "dione-nonrouted-verification.json", nonrouted_record)
    dist.barrier()

    load_started = time.monotonic()
    distributed = DistributedConfig(
        tp_size=world_size, fsdp_size=1, pp_size=1, enable_expert_parallel=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_root,
        dtype=torch.bfloat16,
        distributed_config=distributed,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_backend,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    non_cuda = sorted(
        name
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
        if _tensor_device_type(value) != "cuda"
    )
    if non_cuda:
        raise RuntimeError(f"student is not fully GPU resident: {non_cuda[:8]}")
    versions_before = {name: int(value._version) for name, value in model.named_parameters()}
    install_started = time.monotonic()
    install = ds.install_local_main_experts_dione(
        model, surface, shards, rank=rank, device=torch.device("cuda", local_rank)
    )
    torch.cuda.synchronize(local_rank)
    versions_after = {name: int(value._version) for name, value in model.named_parameters()}

    def _is_mutated_routed_parameter(name: str) -> bool:
        return any(
            f"language_model.layers.{layer}.mlp.experts.{suffix}" in name
            for layer in MAIN_ROUTED_LAYERS
            for suffix in ("gate_up_proj", "down_proj")
        )

    unexpected = sorted(
        name
        for name, version in versions_before.items()
        if versions_after[name] != version and not _is_mutated_routed_parameter(name)
    )
    if unexpected:
        raise RuntimeError(
            f"dione reader mutated non-routed official parameters: {unexpected[:8]}"
        )
    install.update(
        {
            "gpu": torch.cuda.get_device_name(local_rank),
            "load_seconds": install_started - load_started,
            "install_seconds": time.monotonic() - install_started,
            "allocated_bytes": int(torch.cuda.memory_allocated(local_rank)),
            "reserved_bytes": int(torch.cuda.memory_reserved(local_rank)),
            "parameter_version_nonrouted_unchanged": True,
        }
    )
    rank_installs: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(rank_installs, install)
    if rank == 0:
        installed = sum(
            int(row["installed_matrix_count"]) for row in rank_installs if row
        )
        if installed != len(MAIN_ROUTED_LAYERS) * 288 * 3:
            raise RuntimeError(
                f"EP{EP_SIZE} rank installs do not close the complete main routed "
                f"matrix census ({installed})"
            )
        backend = {
            "schema": "malaiwah.glm53-dione-offline-reader-backend.v1",
            "architecture": RELEASED_ARCHITECTURE,
            "dione_repo": surface.repo,
            "dione_revision": surface.revision,
            "source_repo": surface.source_repo,
            "source_revision": surface.source_revision,
            "checkpoint_identity_sha256": checkpoint_identity,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "attention_backend": args.attention_backend,
            "parallelism": plan["parallelism"],
            "reader_mode": identity["mode"],
            "final_tp2_serving_kernel": False,
            "main_routed_runtime_dtype": (
                f"bfloat16 decoded from dione TP4-sliced EXL3 K{bits} payload"
            ),
            "nonrouted_runtime_dtype": "official source dtype, untouched",
            "mtp_standard_logits_executed": False,
            "placement_audit": placement_audit,
            "nonrouted_verification": nonrouted_record,
            "rank_installs": rank_installs,
            "allow_tf32": False,
            "active_tp_plan": getattr(model, "_tp_plan", None),
            "active_ep_plan": getattr(model, "_ep_plan", None),
            "seal_disclosure": ds.SEAL_DISCLOSURE,
        }
        backend["backend_identity_sha256"] = sha256_bytes(canonical_json(backend))
        write_json(output_root / "backend.json", backend)
    else:
        backend = None
    backend_rows: List[Optional[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(backend_rows, backend)
    backend = next(item for item in backend_rows if item is not None)

    logit_records: List[Dict[str, Any]] = []
    capture_started = time.monotonic()
    input_device = torch.device("cuda", local_rank)
    for index, window in enumerate(windows):
        tokens = np.load(window.token_path, allow_pickle=False)
        mask = np.load(window.attention_mask_path, allow_pickle=False)
        causal_mask = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
        ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).unsqueeze(0).to(input_device)
        attention_mask = (
            torch.from_numpy(np.asarray(mask, dtype=np.int64)).unsqueeze(0).to(input_device)
        )
        with torch.inference_mode():
            output_logits = model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).logits[:, :-1, :]
        selected = output_logits[0, torch.from_numpy(causal_mask).to(input_device)]
        if selected.shape != (window.prediction_positions, int(text_config["vocab_size"])):
            raise RuntimeError("student logits differ from sealed panel geometry")
        if rank == 0:
            stored = selected.float().cpu().contiguous()
            logit_path = (output_root / "logits" / f"window-{index:04d}.safetensors").resolve()
            save_file(
                {"logits": stored},
                logit_path,
                metadata={
                    "capture_role": "packed_student",
                    "student_label": student_label,
                    "cold_run": str(args.cold_run),
                    "window_id": window.window_id,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "checkpoint_identity_sha256": checkpoint_identity,
                    "runtime_reader_sha256": identity["runtime_reader_sha256"],
                    "backend_identity_sha256": backend["backend_identity_sha256"],
                },
            )
            logit_records.append(
                {
                    "window_id": window.window_id,
                    "document_id": window.document_id,
                    "domain": window.domain,
                    "role": window.role,
                    "token_ids_sha256": window.token_sha256,
                    "attention_mask_sha256": window.attention_mask_sha256,
                    "prediction_positions": window.prediction_positions,
                    "path": str(logit_path),
                    "bytes": logit_path.stat().st_size,
                    "sha256": sha256_file(logit_path),
                }
            )
            if index == 0 and args.emit_reference_panel is not None:
                panel_out = args.emit_reference_panel.resolve()
                panel_out.parent.mkdir(parents=True, exist_ok=True)
                indices = np.flatnonzero(causal_mask).astype(np.int64)
                save_file(
                    {
                        "input_ids": ids[0].cpu().contiguous(),
                        "attention_mask": attention_mask[0].cpu().contiguous(),
                        "prediction_indices": torch.from_numpy(indices).contiguous(),
                        "logits": stored,
                    },
                    panel_out,
                    metadata={
                        "schema": REFERENCE_PANEL_SCHEMA,
                        "max_abs_tolerance": REFERENCE_MAX_ABS_TOLERANCE,
                        "mean_abs_tolerance": REFERENCE_MEAN_ABS_TOLERANCE,
                        "attention_backend": args.attention_backend,
                        "bits": str(bits),
                        "window_id": window.window_id,
                        "checkpoint_identity_sha256": checkpoint_identity,
                        "runtime_reader_sha256": identity["runtime_reader_sha256"],
                        "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
                    },
                )
                print(f"reference parity panel written: {panel_out}", flush=True)
        del ids, attention_mask, output_logits, selected
        dist.barrier()
    if rank == 0:
        receipt = {
            "schema": "quant-pipeline.glm53-logit-capture.v1",
            "capture_role": "packed_student",
            "cold_run": args.cold_run,
            "model_revision": surface.source_revision,
            "checkpoint_identity_sha256": checkpoint_identity,
            "runtime_reader_sha256": identity["runtime_reader_sha256"],
            "token_panel_receipt_sha256": panel_receipt["receipt_sha256"],
            "backend_identity_sha256": backend["backend_identity_sha256"],
            "weight_dtype": (
                f"EXL3 selective TP4 (Dione) uniform-K{bits} offline-decoded to BF16"
            ),
            "logits_dtype": "float32",
            "kld_direction": "teacher_to_student",
            "prediction_positions": sum(window.prediction_positions for window in windows),
            "vocab_size": int(text_config["vocab_size"]),
            "student_label": student_label,
            "dione_repo": surface.repo,
            "dione_revision": surface.revision,
            "dione_format": surface.fmt,
            "dione_bits": bits,
            "dione_config_sha256": surface.config_sha256,
            "dione_index_sha256": surface.index_sha256,
            "dione_exl3_manifest_sha256": surface.exl3_manifest_sha256,
            "dione_shard_hash_verification": surface.shard_hash_verification,
            "source_repo": surface.source_repo,
            "source_revision": surface.source_revision,
            "nonrouted_verification": nonrouted_record,
            "placement_audit_passed": bool(placement_audit and placement_audit.get("passed")),
            "seal_disclosure": ds.SEAL_DISCLOSURE,
            "logit_files": logit_records,
            "elapsed_seconds": time.monotonic() - capture_started,
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        write_json(output_root / "capture-receipt.json", receipt)
        print(json.dumps({"ok": True, "receipt_sha256": receipt["receipt_sha256"],
                          "cold_run": args.cold_run}, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
