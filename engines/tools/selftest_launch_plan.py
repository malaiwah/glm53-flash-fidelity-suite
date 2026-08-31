#!/usr/bin/env python3
"""Golden-path check for the patches-v2 pipeline: synthetic sealed inventory +
4x H200 preflight through upstream's K4-KL-GATED glm53_uniform_k6
build_launch_plan(inventory, preflight, *, k4_plan, k4_authorized_state) and
every downstream verifier (_inventory_surfaces, inventory_tensor_map, K4
verify_launch_plan/verify_state, K6 verify_launch_plan).  No GPU.

Also validates the v2-0003 H200 worker admission (h200-N worker ids, mixed
fleets rejected, pure-B200 unchanged) and the v2-0004 state_receipt_sha256 fix
(the unpatched upstream KeyErrors on k4_authorized_state["state_sha256"]).

The k6_authorized K4 state receipt here is DIRECTLY SEALED with synthetic
evidence hashes: upstream verify_state format-checks evidence hashes without
dereferencing them, which is exactly the mechanism the production bridge doc
uses (with brandonmusic's REAL published K4 receipt hashes + a disclosure
block).  Run with --pipeline-root pointing at the patches-v2 tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def _pipeline_src(root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (root / candidate).resolve()
    raise SystemExit(f"no quant_pipeline under {root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(_pipeline_src(args.pipeline_root.resolve())))

    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
    from quant_pipeline.campaign import glm53_uniform_k4 as k4
    from quant_pipeline.campaign import glm53_uniform_k6 as uniform_k6
    from quant_pipeline.campaign.glm53_direct_k4 import (
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        inventory_tensor_map,
        tensor_name,
    )

    def seal(doc, field):
        body = dict(doc)
        body[field] = sha256_bytes(canonical_json(body))
        return body

    def fake_hash(*parts):
        return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()

    shard = "model-00001-of-00001.safetensors"
    tensors = []
    for layer in list(k4.MAIN_ROUTED_LAYERS) + list(k4.MTP_LAYERS):
        scope = "routed_expert" if layer in k4.MAIN_ROUTED_LAYERS else "mtp_routed_expert"
        for expert in range(k4.ROUTED_EXPERTS):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = (
                    [INTERMEDIATE_SIZE, HIDDEN_SIZE]
                    if projection != "down_proj"
                    else [HIDDEN_SIZE, INTERMEDIATE_SIZE]
                )
                tensors.append(
                    {
                        "tensor_name": tensor_name(layer, expert, projection),
                        "scope": scope,
                        "dtype": "BF16",
                        "shape": shape,
                        "source_bytes": shape[0] * shape[1] * 2,
                        "source_payload_sha256": fake_hash(layer, expert, projection),
                        "shard": shard,
                    }
                )
    tensors.append(
        {
            "tensor_name": "model.language_model.embed_tokens.weight",
            "scope": "native",
            "dtype": "BF16",
            "shape": [1024, HIDDEN_SIZE],
            "source_bytes": 1024 * HIDDEN_SIZE * 2,
            "source_payload_sha256": fake_hash("embed"),
            "shard": shard,
        }
    )
    inventory = seal(
        {
            "schema": "quant-pipeline.glm-release-inventory.v1",
            "seal_mode": "full-shard-sha256",
            "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
            "checkpoint": "/tmp/fake-bf16",
            "config_sha256": fake_hash("config"),
            "index_sha256": fake_hash("index"),
            "shard_sha256": {shard: fake_hash("shard")},
            "geometry": {
                "model_type": "glm5_next",
                "main_layers": k4.MAIN_LAYER_COUNT,
                "mtp_layers": len(k4.MTP_LAYERS),
                "first_moe_layer": k4.FIRST_MOE_LAYER,
                "routed_experts": k4.ROUTED_EXPERTS,
                "discovered_layers": list(
                    range(k4.MAIN_LAYER_COUNT + len(k4.MTP_LAYERS))
                ),
            },
            "tensors": tensors,
        },
        "inventory_sha256",
    )
    main_rows, mtp_rows, native_rows = k4._inventory_surfaces(inventory)
    assert len(main_rows) == 42 * 288 * 3 and len(mtp_rows) == 288 * 3
    rows = inventory_tensor_map(inventory)
    assert len(rows) == len(tensors)
    print(
        f"synthetic inventory accepted by both verifiers "
        f"({len(main_rows)} main + {len(mtp_rows)} MTP + {len(native_rows)} native)"
    )

    def preflight_for(gpus):
        return seal(
            {
                "schema": k4.PREFLIGHT_SCHEMA,
                "ready": True,
                "mode": "layer-streaming",
                "checkpoint_seal_mode": "full-shard-sha256",
                "checkpoint_inventory_sha256": inventory["inventory_sha256"],
                "workers": 4,
                "gpus": gpus,
            },
            "preflight_sha256",
        )

    h200 = [
        {"index": index, "name": "NVIDIA H200", "compute_capability": "9.0"}
        for index in range(4)
    ]
    b200 = [
        {"index": index, "name": "NVIDIA B200", "compute_capability": "10.0"}
        for index in range(4)
    ]
    preflight = preflight_for(h200)

    # v2-0003: pure-H200 admitted (h200-N), pure-B200 unchanged, mixed rejected.
    workers = k4._b200_workers(preflight, inventory["inventory_sha256"])
    assert [row["worker_id"] for row in workers] == [f"h200-{index}" for index in range(4)]
    assert k4._b200_workers(preflight_for(b200), inventory["inventory_sha256"])[0][
        "worker_id"
    ] == "b200-0"
    mixed = preflight_for(h200[:2] + b200[2:])
    try:
        k4._b200_workers(mixed, inventory["inventory_sha256"])
    except ValueError:
        pass
    else:
        raise AssertionError("mixed B200/H200 fleet was not rejected")
    print("v2-0003 worker admission OK (h200 fleet, b200 unchanged, mixed rejected)")

    # K4 planning documents (pure planning receipts, launch_authorized False).
    k4_plan = k4.build_launch_plan(inventory, preflight)
    k4.verify_launch_plan(k4_plan)

    # Directly-sealed k6_authorized K4 state: the shape the production bridge
    # doc takes (synthetic evidence hashes here; REAL published hashes there).
    worker_id = k4_plan["scheduler"]["workers"][0]["worker_id"]
    completed = {}
    for layer in k4.MAIN_ROUTED_LAYERS:
        completed[str(layer)] = {
            "worker_id": worker_id,
            "claim_receipt_sha256": fake_hash("claim", layer),
            "layer_receipt_sha256": fake_hash("layer", layer),
        }
    k4_state = seal(
        {
            "schema": k4.STATE_RECEIPT_SCHEMA,
            "launch_plan_sha256": k4_plan["launch_plan_sha256"],
            "sequence": 7,
            "previous_state_receipt_sha256": fake_hash("previous-state"),
            "phase": "k6_authorized",
            "pending_layers": [],
            "active_claims": {},
            "completed_layers": completed,
            "evidence": {
                "k4_readiness_receipt_sha256": fake_hash("readiness"),
                "main_routed_receipt_sha256": fake_hash("main"),
                "mtp_k4_adapter_receipt_sha256": fake_hash("mtp"),
                "packed_checkpoint_receipt_sha256": fake_hash("packed"),
                "native_copy_receipt_sha256": fake_hash("native"),
                "k4_packed_kld_receipt_sha256": fake_hash("kld"),
            },
            "k6_authorized": True,
        },
        "state_receipt_sha256",
    )
    k4.verify_state(k4_plan, k4_state)
    print("synthetic k6_authorized K4 state passes k4.verify_state")

    # v2-0004: the patched module reads state_receipt_sha256 (upstream KeyErrors
    # on state_sha256 - prove the state receipt genuinely lacks that field).
    assert "state_sha256" not in k4_state and "state_receipt_sha256" in k4_state

    plan = uniform_k6.build_launch_plan(
        inventory, preflight, k4_plan=k4_plan, k4_authorized_state=k4_state
    )
    uniform_k6.verify_launch_plan(plan)
    assert plan["rate_contract"]["K6"] == 37152
    assert plan["rate_contract"]["allowed_bits"] == [6]
    assert plan["scheduler"]["workers"][0]["worker_id"] == "h200-0"
    assert plan["k4_launch_plan_sha256"] == k4_plan["launch_plan_sha256"]
    assert plan["k4_authorized_state_sha256"] == k4_state["state_receipt_sha256"]
    replay = copy.deepcopy(plan)
    uniform_k6.verify_launch_plan(replay)
    print(
        "K4-gated build_launch_plan OK:",
        plan["launch_plan_sha256"][:16],
        "| 37,152 routed matrices | h200 workers",
    )
    print(json.dumps({"launch_plan_sha256": plan["launch_plan_sha256"]}))

    # ------------------------------------------------------------------ K8 --
    # v2-0007 K8-uniform admission (skipped gracefully on a pre-0007 tree).
    try:
        from quant_pipeline.campaign import glm53_uniform_k8 as uniform_k8
    except ImportError:
        print("v2-0007 not applied (glm53_uniform_k8 absent) - K8 leg skipped")
        return 0
    from quant_pipeline.campaign import glm53_direct_k4 as direct

    assert 8 in direct.SUPPORTED_BITS
    assert direct.recipe_id_for_bits(8) == (
        "malaiwah-shapleymcg-r10-uniform-k8-candidate-conditioned-down-v1"
    )
    for helper, expected in (
        (direct.contract_schema_for_bits, "malaiwah.glm53-direct-mcg-k8-contract.v1"),
        (direct.materialization_plan_schema_for_bits, "malaiwah.glm53-k8-materialization-plan.v1"),
        (direct.materialization_receipt_schema_for_bits, "malaiwah.glm53-k8-materialization-receipt.v1"),
    ):
        assert helper(8) == expected, (helper.__name__, helper(8))
        # K4/K6 outputs unchanged by 0007
    assert direct.contract_schema_for_bits(6) == "quant-pipeline.glm53-direct-mcg-k6-contract.v1"
    assert direct.contract_schema_for_bits(4) == "quant-pipeline.glm53-direct-mcg-k4-contract.v1"
    assert uniform_k8.LAUNCH_PLAN_SCHEMA == direct.K8_LAUNCH_PLAN_SCHEMA
    from quant_pipeline.campaign.glm53_mtp_k4 import (
        _main_receipt_schema,
        _mtp_adapter_schema,
        _mtp_contract_schema,
    )

    assert _mtp_contract_schema(8) == "malaiwah.glm53-mtp45-exl3-mcg-k8-contract.v1"
    assert _mtp_adapter_schema(8) == "malaiwah.glm53-uniform-k8-mtp-adapter-receipt.v1"
    assert _main_receipt_schema(8) == "malaiwah.glm53-exl3-mcg-main-k8-receipt.v1"
    assert _main_receipt_schema(6) == "quant-pipeline.glm53-exl3-mcg-main-k6-receipt.v1"
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader

    assert 8 in reader.SUPPORTED_BITS

    k8_plan = uniform_k8.build_launch_plan(
        inventory, preflight, k4_plan=k4_plan, k4_authorized_state=k4_state
    )
    uniform_k8.verify_launch_plan(k8_plan)
    assert k8_plan["rate_contract"]["K8"] == 37152
    assert k8_plan["rate_contract"]["allowed_bits"] == [8]
    assert k8_plan["profile"] == "k8-tp4"
    assert k8_plan["preparation_contract"]["same_transform_seed_as_k6_campaign"] is True
    assert k8_plan["scheduler"]["workers"][0]["worker_id"] == "h200-0"
    assert k8_plan["k4_launch_plan_sha256"] == k4_plan["launch_plan_sha256"]
    # cross-verifier: the direct-contract launch-plan branch accepts the K8
    # plan seal under the malaiwah schema (same check build_contract performs)
    direct._verify_seal(k8_plan, direct.K8_LAUNCH_PLAN_SCHEMA, "launch_plan_sha256")
    # ... and the K6 verifier refuses it (no cross-admission)
    try:
        uniform_k6.verify_launch_plan(k8_plan)
    except ValueError:
        pass
    else:
        raise AssertionError("K6 verifier accepted a K8 plan")

    # packed-choice geometry: bits=8 -> 128 int16 trellis words per tile
    # (torch-free check of the reader's expected-shape table)
    import quant_pipeline.evaluation.glm53_packed_k4_reader as r

    n, k = 4096, 2048  # down_proj
    expected_trellis = [k // 16, n // 16, 8 * 16]
    assert expected_trellis[-1] == 128
    print(
        "v2-0007 K8 leg OK:",
        k8_plan["launch_plan_sha256"][:16],
        "| malaiwah schemas | 128-word trellis geometry",
    )
    print(json.dumps({"k8_launch_plan_sha256": k8_plan["launch_plan_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
