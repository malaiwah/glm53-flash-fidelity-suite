"""Sealed GLM-5.3 uniform-K6 launch plan (malaiwah campaign).

Upstream ships only the VERIFIER contract for this module: the K6 branch of
``glm53_direct_k4.build_contract`` demands a sealed document with schema
``quant-pipeline.glm53-uniform-k6-four-b200-launch-plan.v1``, profile
``k6-tp4``, ``rate_contract.allowed_bits == [6]`` and
``rate_contract.global_allocator_invoked is False``, and
``glm53_mtp_k4.build_contract`` imports ``verify_launch_plan`` from this exact
module path.  The module itself is absent from the published runtime; this
file supplies it.

Like its K4 sibling this module is orchestration-only: it never starts a
process, imports CUDA, or invokes an allocator.

DISCLOSED DEVIATION (malaiwah): the upstream schema name pins "four-b200"
and the upstream K4 planner attests four B200/SM100 devices.  This campaign
executes on four H200 (SM90) single-GPU workers.  The schema string is kept
verbatim because the sealed upstream verifier requires it; the actual worker
hardware is recorded truthfully in ``scheduler.workers[*]`` and in
``hardware_attestation`` below, so every receipt names the real devices.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..core.artifacts import canonical_json, sha256_bytes
from .glm53_uniform_k4 import (
    ALL_ROUTED_MATRIX_COUNT,
    MAIN_ROUTED_LAYERS,
    MTP_LAYERS,
    MTP_ROUTED_MATRIX_COUNT,
    MAIN_ROUTED_MATRIX_COUNT,
    PREFLIGHT_SCHEMA,
    PROJECTIONS,
    ROUTED_EXPERTS,
    WORKERS,
    _ROUTED,
    _inventory_surfaces,
    _require_hash,
    _seal,
    _verify_seal,
)


K6_LAUNCH_PLAN_SCHEMA = "quant-pipeline.glm53-uniform-k6-four-b200-launch-plan.v1"
K6_PROFILE = "k6-tp4"
K6_BITS = 6
_ACCEPTED_DEVICE_CLASSES = (
    # (name substring, compute capability prefix, worker id prefix)
    ("B200", "10.", "b200"),
    ("H200", "9.0", "h200"),
)


def _workers(preflight: Mapping[str, Any], inventory_sha256: str) -> list[dict[str, Any]]:
    preflight_sha = _verify_seal(
        preflight,
        schema=PREFLIGHT_SCHEMA,
        field="preflight_sha256",
        label="four-worker K6 preflight",
    )
    if (
        preflight.get("ready") is not True
        or preflight.get("mode") != "layer-streaming"
        or preflight.get("checkpoint_seal_mode") != "full-shard-sha256"
        or preflight.get("checkpoint_inventory_sha256") != inventory_sha256
        or preflight.get("workers") != WORKERS
    ):
        raise ValueError("four-worker layer-streaming preflight is not execution-ready")
    gpus = preflight.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != WORKERS:
        raise ValueError("K6 launch plan requires exactly four preflight GPUs")
    workers: list[dict[str, Any]] = []
    indices: set[int] = set()
    classes: set[str] = set()
    for slot, raw in enumerate(gpus):
        if not isinstance(raw, Mapping):
            raise ValueError("preflight GPU row is malformed")
        index = raw.get("index")
        name = raw.get("name")
        capability = raw.get("compute_capability")
        matched = None
        for substring, prefix, worker_prefix in _ACCEPTED_DEVICE_CLASSES:
            if (
                isinstance(name, str)
                and substring in name
                and isinstance(capability, str)
                and capability.startswith(prefix)
            ):
                matched = worker_prefix
                break
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index in indices
            or matched is None
        ):
            raise ValueError(
                "uniform K6 workers must be four distinct attested B200 or H200 devices"
            )
        indices.add(index)
        classes.add(matched)
        workers.append(
            {
                "worker_id": f"{matched}-{slot}",
                "physical_gpu": index,
                "cuda_visible_devices": str(index),
                "codec_device": "cuda:0",
                "name": name,
                "compute_capability": capability,
                "preflight_sha256": preflight_sha,
            }
        )
    if len(classes) != 1:
        raise ValueError("K6 workers must all belong to one device class")
    return workers


def build_launch_plan(
    inventory: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the immutable, no-launch K6 work contract from sealed evidence."""

    main_rows, mtp_rows, native_rows = _inventory_surfaces(inventory)
    inventory_sha = str(inventory["inventory_sha256"])
    workers = _workers(preflight, inventory_sha)
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in MAIN_ROUTED_LAYERS}
    tensor_contract: list[dict[str, Any]] = []
    for row in main_rows:
        match = _ROUTED.fullmatch(row["tensor_name"])
        assert match is not None
        by_layer[int(match.group(1))].append(row)
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": K6_BITS,
                "disposition": "uniform_k6_direct_encode",
                "execution_track": "main_dynamic_layer_scheduler",
            }
        )
    for row in mtp_rows:
        tensor_contract.append(
            {
                "tensor_name": row["tensor_name"],
                "source_payload_sha256": row.get("source_payload_sha256"),
                "bits": K6_BITS,
                "disposition": "uniform_k6_separately_qualified_mtp_adapter",
                "execution_track": "mtp_adapter_after_main",
            }
        )

    import math

    units: list[dict[str, Any]] = []
    for layer, rows in by_layer.items():
        names = sorted(row["tensor_name"] for row in rows)
        if len(rows) != ROUTED_EXPERTS * len(PROJECTIONS):
            raise AssertionError("validated routed layer lost matrices")
        units.append(
            {
                "layer": layer,
                "expert_count": ROUTED_EXPERTS,
                "matrix_count": len(rows),
                "source_bytes": sum(int(row["source_bytes"]) for row in rows),
                "source_elements": sum(math.prod(row["shape"]) for row in rows),
                "tensor_names_sha256": sha256_bytes(canonical_json(names)),
                "bits": K6_BITS,
                "allowed_bits": [K6_BITS],
                "global_allocator": False,
                "candidate_rate_grid": False,
            }
        )
    queue = [
        row["layer"]
        for row in sorted(units, key=lambda row: (-int(row["source_bytes"]), int(row["layer"])))
    ]
    native_names = [row["tensor_name"] for row in native_rows]
    mtp_names = [row["tensor_name"] for row in mtp_rows]
    body = {
        "schema": K6_LAUNCH_PLAN_SCHEMA,
        "model_revision": inventory.get("model_revision"),
        "inventory_sha256": inventory_sha,
        "preflight_sha256": preflight["preflight_sha256"],
        "launch_authorized": False,
        "boundary": "sealed planning and receipt transitions only; this document starts no process",
        "profile": K6_PROFILE,
        "runtime_target": {"tensor_parallel_size": 4, "physical_workers": WORKERS},
        "hardware_attestation": {
            "upstream_schema_device": "B200/SM100",
            "actual_device_class": workers[0]["worker_id"].rsplit("-", 1)[0],
            "actual_devices": [
                {"name": row["name"], "compute_capability": row["compute_capability"]}
                for row in workers
            ],
            "deviation_disclosed": True,
        },
        "geometry": {
            "main_layers": list(MAIN_ROUTED_LAYERS),
            "mtp_layers": list(MTP_LAYERS),
            "routed_experts": ROUTED_EXPERTS,
            "projections": list(PROJECTIONS),
        },
        "rate_contract": {
            "allocation": "none_uniform_fixed_rate",
            "global_allocator_invoked": False,
            "candidate_rate_grid_invoked": False,
            "allowed_bits": [K6_BITS],
            "K6": ALL_ROUTED_MATRIX_COUNT,
            "main_routed_matrix_count": MAIN_ROUTED_MATRIX_COUNT,
            "mtp_routed_k6_matrix_count": MTP_ROUTED_MATRIX_COUNT,
            "all_main_plus_mtp_routed_matrix_count": ALL_ROUTED_MATRIX_COUNT,
            "main_must_complete_before_mtp": True,
            "mtp_may_not_remain_native_in_final_k6": True,
        },
        "native_copy_contract": {
            "policy": "byte_exact_source_copy",
            "includes_all_nonrouted": True,
            "includes_routed_mtp": False,
            "tensor_count": len(native_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in native_rows),
            "tensor_names_sha256": sha256_bytes(canonical_json(native_names)),
        },
        "routed_tensor_contract": tensor_contract,
        "work_units": units,
        "mtp_work_unit": {
            "layer": MTP_LAYERS[0],
            "expert_count": ROUTED_EXPERTS,
            "matrix_count": len(mtp_rows),
            "source_bytes": sum(int(row["source_bytes"]) for row in mtp_rows),
            "source_elements": sum(math.prod(row["shape"]) for row in mtp_rows),
            "tensor_names_sha256": sha256_bytes(canonical_json(mtp_names)),
            "bits": K6_BITS,
            "allowed_bits": [K6_BITS],
            "adapter_qualification_required": True,
            "scheduled_with_main_layers": False,
        },
        "scheduler": {
            "policy": "dynamic_next_unclaimed_whole_layer",
            "static_layer_partition_forbidden": True,
            "one_active_layer_per_worker": True,
            "workers": workers,
            "initial_queue": queue,
        },
    }
    if len(tensor_contract) != ALL_ROUTED_MATRIX_COUNT or len(mtp_rows) != MTP_ROUTED_MATRIX_COUNT:
        raise AssertionError("uniform K6 main/MTP census drift")
    return _seal(body, "launch_plan_sha256")


def verify_launch_plan(plan: Mapping[str, Any]) -> str:
    seal = _verify_seal(
        plan,
        schema=K6_LAUNCH_PLAN_SCHEMA,
        field="launch_plan_sha256",
        label="uniform K6 launch plan",
    )
    rate = plan.get("rate_contract")
    if (
        not isinstance(rate, Mapping)
        or rate.get("allowed_bits") != [K6_BITS]
        or rate.get("global_allocator_invoked") is not False
        or rate.get("candidate_rate_grid_invoked") is not False
        or rate.get("K6") != ALL_ROUTED_MATRIX_COUNT
        or rate.get("mtp_routed_k6_matrix_count") != MTP_ROUTED_MATRIX_COUNT
    ):
        raise ValueError("uniform K6 launch-plan rate census differs")
    if plan.get("profile") != K6_PROFILE:
        raise ValueError("uniform K6 launch plan must declare the k6-tp4 profile")
    if plan.get("launch_authorized") is not False:
        raise ValueError("a planning receipt may not authorize process launch")
    _require_hash(plan.get("inventory_sha256"), "K6 launch plan inventory")
    mtp = plan.get("mtp_work_unit")
    if not isinstance(mtp, Mapping):
        raise ValueError("uniform K6 launch plan lacks its MTP work unit")
    _require_hash(mtp.get("tensor_names_sha256"), "K6 MTP tensor-name census")
    return seal
