#!/usr/bin/env python3
"""Local synthetic validation for patches-v2 0007 (K8-uniform admission).

CPU-only.  Exercises, at bits=8:
  1. PackedMCGPayloadStore.put_choice -> verify_choice -> reader
     verify_choice_descriptor (128-word trellis geometry accepted end-to-end);
  2. unpack_trellis_states: shape admission + exact roundtrip against a
     locally-computed inverse pack (states -> words -> states);
  3. a full synthetic K8 direct contract: build_contract(bits=8) ->
     verify_contract -> contract_bits == 8 -> build_work_units;
  4. negative controls: bits=5 refused by the store, a K6 contract still
     verifies (no behavior drift), K6 verifier refuses the K8 plan schema.
"""

import copy
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
assert ROOT, "usage: test_k8_synthetic.py <pipeline-root>"
for cand in ("runtime/src", "src", "."):
    if (ROOT / cand / "quant_pipeline" / "__init__.py").is_file():
        sys.path.insert(0, str(ROOT / cand))
        break

import torch  # noqa: E402

from quant_pipeline.core.artifacts import canonical_json, sha256_bytes  # noqa: E402
from quant_pipeline.campaign import glm53_direct_k4 as direct  # noqa: E402
from quant_pipeline.campaign import glm53_uniform_k4 as k4  # noqa: E402
from quant_pipeline.campaign import glm53_uniform_k8 as uniform_k8  # noqa: E402
from quant_pipeline.checkpoint.packed_payload import (  # noqa: E402
    MCG_MARKER_SIGNED_INT32,
    PackedMCGPayloadStore,
)
from quant_pipeline.evaluation.glm53_packed_k4_reader import (  # noqa: E402
    unpack_trellis_states,
    verify_choice_descriptor,
)
from quant_pipeline.calibration.glm53_capture import (  # noqa: E402
    CAPTURE_SCHEMA,
    MAIN_ROUTED_LAYERS,
)


def seal(doc, field):
    body = dict(doc)
    body[field] = sha256_bytes(canonical_json(body))
    return body


def fake_hash(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


# ---------------------------------------------------------------- 1) store --
torch.manual_seed(0x1FED)
n, k = 4096, 2048  # down_proj HF orientation (out, in)
bits = 8
trellis = torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16), dtype=torch.int16)
suh = (torch.randint(0, 2, (k,)).float() * 2 - 1).half()
svh = (torch.randint(0, 2, (n,)).float() * 2 - 1).half()
mcg = torch.tensor([MCG_MARKER_SIGNED_INT32], dtype=torch.int32)
reconstruction = torch.randn(n, k).half()

tmp = Path(tempfile.mkdtemp(prefix="k8store-"))
store = PackedMCGPayloadStore(tmp / "payload-store")
choice = store.put_choice(
    layer=3,
    expert=0,
    projection="down_proj",
    choice_id="L003.E000.down_proj.K8",
    bits=8,
    trellis=trellis,
    suh=suh,
    svh=svh,
    mcg=mcg,
    reconstruction=reconstruction,
    vector_topology={"suh": "expert_private", "svh": "expert_private"},
    reader_abi_sha256=fake_hash("reader-abi"),
    provenance={"synthetic": True},
    predecessor_state_hash=fake_hash("state"),
)
store.verify_choice(choice)
verified = verify_choice_descriptor(
    choice, layer=3, expert=0, projection="down_proj", bits=8
)
assert verified["objects"]["trellis"]["shape"] == [k // 16, n // 16, 128]
loaded = store.objects.load_tensor(choice["objects"]["trellis"])
assert torch.equal(loaded, trellis)
print("1) K8 packed-choice roundtrip OK (128-word trellis, store+reader verify)")

try:
    store.put_choice(
        layer=3, expert=0, projection="down_proj", choice_id="x", bits=5,
        trellis=trellis, suh=suh, svh=svh, mcg=mcg, reconstruction=reconstruction,
        vector_topology={"suh": "expert_private", "svh": "expert_private"},
        reader_abi_sha256=fake_hash("r"), provenance={},
        predecessor_state_hash=fake_hash("s"),
    )
except ValueError:
    print("   negative control: bits=5 refused by the store")
else:
    raise AssertionError("bits=5 was not refused")

# ------------------------------------------------------- 2) unpack inverse --
def pack_states(states, bits):
    """Inverse of unpack_trellis_states (per its own documented layout)."""

    states = states.to(torch.int64) & 0xFFFF
    edges = torch.zeros_like(states)
    # states = OR_lag roll(edges, lag) << lag*bits  with edges < 2^bits:
    # edge[i] = (state[(i+1) % 256... actually roll(+lag) shifts index forward;
    # recover edges as the low `bits` bits of the state at the position where
    # lag == 0 contributed them: edges[j] = states[j] & (2^bits - 1)
    edges = states & ((1 << bits) - 1)
    tiles16 = edges.reshape(-1, 16, 16)  # (tiles, 16 groups, 16 edges)
    word_bits = []
    for g in range(16):
        group = tiles16[:, g, :]  # (tiles, 16) edges of 2^bits
        # bitstream: 16 edges * bits bits, MSB-first per edge
        stream = torch.zeros(group.shape[0], 16 * bits, dtype=torch.int64)
        for e in range(16):
            for b in range(bits):
                stream[:, e * bits + b] = (group[:, e] >> (bits - 1 - b)) & 1
        word_bits.append(stream)
    stream = torch.stack(word_bits, dim=1)  # (tiles, 16, 16*bits)
    words = torch.zeros(stream.shape[0], 16, bits, dtype=torch.int64)
    for w in range(bits):
        chunk = stream[:, :, w * 16:(w + 1) * 16]
        for b in range(16):
            words[:, :, w] |= chunk[:, :, b] << (15 - b)
    words = words.reshape(stream.shape[0], bits * 16)
    words = words.reshape(words.shape[0], -1, 2).flip(-1).reshape(words.shape)
    return ((words + 32768) % 65536 - 32768).to(torch.int16)


states = torch.randint(0, 1 << bits, (4, 16, 256), dtype=torch.int64)
# construct full 16-bit states the way the trellis chains them (roll semantics)
full = torch.zeros(4 * 16, 256, dtype=torch.int64)
edges0 = states.reshape(-1, 256)
import math
for lag in range(math.ceil(16 / bits)):
    full |= torch.roll(edges0, shifts=lag, dims=-1) << (lag * bits)
full &= 0xFFFF
packed = pack_states(full.reshape(4, 16, 256), bits).reshape(4, 16, bits * 16)
unpacked = unpack_trellis_states(packed, bits=bits)
assert unpacked.shape == (4, 16, 256)
assert torch.equal(unpacked.to(torch.int64) & 0xFFFF, full.reshape(4, 16, 256)), (
    "K8 unpack does not invert the documented pack layout"
)
print("2) unpack_trellis_states K8 roundtrip exact (states->128 words->states)")

try:
    unpack_trellis_states(packed[..., :112], bits=8)
except ValueError:
    print("   negative control: wrong word count refused")
else:
    raise AssertionError("wrong word count accepted")

# ----------------------------------------------------------- 3) contract --
shard = "model-00001-of-00001.safetensors"
tensors = []
for layer in list(k4.MAIN_ROUTED_LAYERS) + list(k4.MTP_LAYERS):
    scope = "routed_expert" if layer in k4.MAIN_ROUTED_LAYERS else "mtp_routed_expert"
    for expert in range(k4.ROUTED_EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = [2048, 4096] if projection != "down_proj" else [4096, 2048]
            tensors.append({
                "tensor_name": direct.tensor_name(layer, expert, projection),
                "scope": scope, "dtype": "BF16", "shape": shape,
                "source_bytes": shape[0] * shape[1] * 2,
                "source_payload_sha256": fake_hash(layer, expert, projection),
                "shard": shard,
            })
tensors.append({
    "tensor_name": "model.language_model.embed_tokens.weight", "scope": "native",
    "dtype": "BF16", "shape": [1024, 4096], "source_bytes": 1024 * 4096 * 2,
    "source_payload_sha256": fake_hash("embed"), "shard": shard,
})
inventory = seal({
    "schema": "quant-pipeline.glm-release-inventory.v1",
    "seal_mode": "full-shard-sha256",
    "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
    "checkpoint": "/tmp/fake-bf16",
    "config_sha256": fake_hash("config"), "index_sha256": fake_hash("index"),
    "shard_sha256": {shard: fake_hash("shard")},
    "geometry": {
        "model_type": "glm5_next", "main_layers": k4.MAIN_LAYER_COUNT,
        "mtp_layers": len(k4.MTP_LAYERS), "first_moe_layer": k4.FIRST_MOE_LAYER,
        "routed_experts": k4.ROUTED_EXPERTS,
        "discovered_layers": list(range(k4.MAIN_LAYER_COUNT + len(k4.MTP_LAYERS))),
    },
    "tensors": tensors,
}, "inventory_sha256")
preflight = seal({
    "schema": k4.PREFLIGHT_SCHEMA, "ready": True, "mode": "layer-streaming",
    "checkpoint_seal_mode": "full-shard-sha256",
    "checkpoint_inventory_sha256": inventory["inventory_sha256"], "workers": 4,
    "gpus": [{"index": i, "name": "NVIDIA H200", "compute_capability": "9.0"} for i in range(4)],
}, "preflight_sha256")
k4_plan = k4.build_launch_plan(inventory, preflight)
worker_id = k4_plan["scheduler"]["workers"][0]["worker_id"]
completed = {
    str(layer): {
        "worker_id": worker_id,
        "claim_receipt_sha256": fake_hash("claim", layer),
        "layer_receipt_sha256": fake_hash("layer", layer),
    } for layer in k4.MAIN_ROUTED_LAYERS
}
k4_state = seal({
    "schema": k4.STATE_RECEIPT_SCHEMA,
    "launch_plan_sha256": k4_plan["launch_plan_sha256"], "sequence": 7,
    "previous_state_receipt_sha256": fake_hash("previous-state"),
    "phase": "k6_authorized", "pending_layers": [], "active_claims": {},
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
}, "state_receipt_sha256")
k8_plan = uniform_k8.build_launch_plan(
    inventory, preflight, k4_plan=k4_plan, k4_authorized_state=k4_state
)
uniform_k8.verify_launch_plan(k8_plan)

capture_manifest = seal({
    "schema": CAPTURE_SCHEMA,
    "inventory_sha256": inventory["inventory_sha256"],
    "roles": list(direct.REQUIRED_ROLES),
    "layers": list(MAIN_ROUTED_LAYERS),
}, "capture_sha256")
prepared_source = {
    "recipe_id": direct.recipe_id_for_bits(8),
    "reviewed_glm53_entrypoint": True,
    "entrypoint": "quant_pipeline.campaign.glm53_prepared_backend.Glm53PreparedMCGBackend",
    "tree_sha256": fake_hash("tree"),
    "codec_family": "exl3-mcg", "mcg_multiplier_hex": "0xCBAC1FED", "bits": 8,
    "candidate_rate_grid": False, "global_allocator": False,
    "gate_up_hessian": "routed_p2_uncentered_full_hessian",
    "down_hessian": "decoded_k8_candidate_conditioned_routed_p2_uncentered_full_hessian",
    "down_candidate_conditioned": True,
    "profile_source": "public-run-qwen-fast-encode-defaults",
    "profile_policy": "energy_balanced", "scale_family": "per128-grid",
    "profile_fixed_before_encoding": True,
    "selection_used_for_profile_choice": False,
    "selection_rows_used_for_encoding": False,
    "confirmation_rows_used_for_choice": False,
    "sqg_orchestration_imported": False,
}
exllama = {
    "fresh_build": True, "compute_capabilities": ["9.0", "10.0"],
    "extension_path": "/tmp/ext.so", "extension_sha256": fake_hash("ext"),
}
preparation = {
    "bits": 8, "codec_family": "exl3-mcg",
    "global_allocator_invoked": False, "candidate_rate_grid_invoked": False,
    "profile_source": "public-run-qwen-fast-encode-defaults",
    "profile_fixed_before_encoding": True,
    "selection_rows_used": False, "selection_used_for_profile_choice": False,
    "confirmation_report_only": True, "confirmation_used_for_choice": False,
    "selection_used_for_final_encoding": False,
    "transform_seed_sha256": fake_hash("seed"),
    "fixed_profile_receipt_sha256": fake_hash("profile"),
    "permutation_set_sha256": fake_hash("perm"),
    "gate_up_p2_set_sha256": fake_hash("p2"),
    "gss_receipts_sha256": fake_hash("gss"),
    "confirmation_report_receipt_sha256": fake_hash("confirm"),
}
contract = direct.build_contract(
    launch_plan=k8_plan,
    inventory=inventory,
    capture_manifest=capture_manifest,
    prepared_source=prepared_source,
    exllama=exllama,
    preparation=preparation,
    reader_abi_sha256=fake_hash("reader-abi"),
    pure_mcg_backend_receipt_sha256=fake_hash("backend"),
    pure_mcg_preparation_receipt_sha256=fake_hash("pure-prep"),
    bits=8,
)
assert contract["schema"] == "malaiwah.glm53-direct-mcg-k8-contract.v1"
direct.verify_contract(contract)
assert direct.contract_bits(contract) == 8
units = direct.build_work_units(contract)
assert len(units) == 42 and all(u["bits"] == 8 for u in units)
state = direct.initial_work_state(contract, units)
direct.verify_work_state(contract, state)
print("3) synthetic K8 contract build/verify OK "
      f"({contract['contract_sha256'][:16]}, 42 work units, state chain verifies)")

# --------------------------------------------------- 4) no-drift controls --
try:
    from quant_pipeline.campaign import glm53_uniform_k6 as uniform_k6
    uniform_k6.verify_launch_plan(k8_plan)
except ValueError:
    print("4) K6 verifier refuses the K8 plan (no cross-admission)")
else:
    raise AssertionError("K6 verifier accepted a K8 plan")

k6_plan = uniform_k6.build_launch_plan(
    inventory, preflight, k4_plan=k4_plan, k4_authorized_state=k4_state
)
prepared6 = dict(prepared_source, recipe_id=direct.recipe_id_for_bits(6), bits=6,
                 down_hessian="decoded_k6_candidate_conditioned_routed_p2_uncentered_full_hessian")
contract6 = direct.build_contract(
    launch_plan=k6_plan, inventory=inventory, capture_manifest=capture_manifest,
    prepared_source=prepared6, exllama=exllama,
    preparation=dict(preparation, bits=6),
    reader_abi_sha256=fake_hash("reader-abi"),
    pure_mcg_backend_receipt_sha256=fake_hash("backend"),
    pure_mcg_preparation_receipt_sha256=fake_hash("pure-prep"),
    bits=6,
)
assert contract6["schema"] == "quant-pipeline.glm53-direct-mcg-k6-contract.v1"
direct.verify_contract(contract6)
print("   K6 contract path unchanged (builds+verifies under the upstream schema)")
print("ALL K8 SYNTHETIC CHECKS GREEN")
