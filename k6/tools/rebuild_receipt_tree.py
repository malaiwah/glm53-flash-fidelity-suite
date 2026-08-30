#!/usr/bin/env python3
"""Materialize the packed reader's layers/ and experts/ receipt trees from the
PUBLISHED content-addressed payload store, and prove the result is the original.

WHY THIS IS NEEDED: malaiwah/GLM-5.3-Flash-TR3-partsbin-v1 publishes
payload-store/{objects,choices} plus 15 top-level receipts, but NOT the
`layers/layer-NNN.json` and `experts/layer-NNN/expert-MMM.json` receipt trees
that quant_pipeline's packed reader loads.  Those lived only on JarvisLabs
fs 3394, which no longer exists.

WHY THIS IS NOT INVENTING PROVENANCE: both receipt bodies are fully determined
by data that IS published --

  expert receipt body = {schema, contract_sha256, layer, expert, bits,
                         projections, candidate_rate_grid=False,
                         global_allocator=False, choices}
      where `choices` are the content-addressed choice descriptors verbatim
      (each file's name IS the sha256 of its content, so they cannot be faked)

  layer receipt body  = {schema, contract_sha256, layer, experts, matrix_count,
                         bits, expert_receipt_sha256[], choice_sha256[],
                         complete=True}
      -- exactly the body that the SHIPPED pipeline function
      build_mtp_packed_layer_receipt() constructs, i.e. the campaign's own
      canonical constructor.

and the result is then CHECKED against three independently published digests:

  1. main-receipt.json          -> layer_receipt_sha256
  2. materialization-plan.json  -> main_layer_receipt_sha256
  3. mtp-adapter-receipt.json   -> packed_payload_receipt_sha256

If the rebuilt seals equal those, the tree is byte-identical to the original and
the checkpoint identity the run computes will be the sealed one.  If any digest
differs, this script REFUSES and writes nothing -- we do not run a measurement
against a surface we cannot prove is the sealed K6 surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="packed root (…/packed/k6)")
    ap.add_argument("--pipeline-src", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="write the trees; without it, verify only")
    args = ap.parse_args()

    sys.path.insert(0, str(args.pipeline_src))
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes
    from quant_pipeline.evaluation import glm53_packed_k4_reader as R

    root = args.root.resolve()
    contract = json.loads((root / "contract.json").read_text())
    contract_sha256 = R.verify_contract(contract)
    bits = R.contract_bits(contract)
    projections = list(R.PROJECTIONS)
    num_experts = R.NUM_EXPERTS
    layers = list(R.MAIN_ROUTED_LAYERS)
    mtp_layer = R.MTP_LAYER

    print(f"contract_sha256={contract_sha256} bits={bits} experts={num_experts} "
          f"projections={projections} main_layers={len(layers)} mtp_layer={mtp_layer}")

    # ---- index every published choice by (layer, expert, projection) -------
    index: dict[tuple[int, int, str], dict] = {}
    dupes = 0
    choice_dir = root / "payload-store/choices"
    files = sorted(choice_dir.glob("*/*.json"))
    print(f"indexing {len(files)} published choice descriptors …")
    for path in files:
        row = json.loads(path.read_text())
        # the filename IS the content sha; the reader re-verifies the seal
        if path.stem != row.get("choice_sha256"):
            raise SystemExit(f"choice file name != choice_sha256: {path}")
        key = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
        if key in index:
            dupes += 1
        index[key] = row
    print(f"indexed {len(index)} unique (layer,expert,projection) keys, {dupes} duplicates")

    expected_keys = (len(layers) + 1) * num_experts * len(projections)
    if len(index) != expected_keys:
        raise SystemExit(f"choice census {len(index)} != expected {expected_keys}")

    # ---- rebuild expert receipts, then layer receipts ---------------------
    expert_bodies: dict[tuple[int, int], dict] = {}
    layer_bodies: dict[int, dict] = {}

    for layer in layers + [mtp_layer]:
        expert_hashes: list[str] = []
        choice_hashes: list[str] = []
        for expert in range(num_experts):
            choices = {}
            for projection in projections:
                row = index.get((layer, expert, projection))
                if row is None:
                    raise SystemExit(f"missing choice L{layer} E{expert} {projection}")
                # validate against the reader's own binding rules before use
                R.verify_choice_descriptor(row, layer=layer, expert=expert,
                                           projection=projection, bits=bits)
                choices[projection] = row
            body = {
                "schema": R.EXPERT_RECEIPT_SCHEMA,
                "contract_sha256": contract_sha256,
                "layer": layer,
                "expert": expert,
                "bits": bits,
                "projections": projections,
                "candidate_rate_grid": False,
                "global_allocator": False,
                "choices": choices,
            }
            body["receipt_sha256"] = sha256_bytes(canonical_json(body))
            expert_bodies[(layer, expert)] = body
            expert_hashes.append(body["receipt_sha256"])
            choice_hashes.extend(choices[p]["choice_sha256"] for p in projections)

        schema = (R.MTP_PACKED_LAYER_RECEIPT_SCHEMA if layer == mtp_layer
                  else R.LAYER_RECEIPT_SCHEMA)
        lbody = {
            "schema": schema,
            "contract_sha256": contract_sha256,
            "layer": layer,
            "experts": num_experts,
            "matrix_count": num_experts * len(projections),
            "bits": bits,
            "expert_receipt_sha256": expert_hashes,
            "choice_sha256": choice_hashes,
            "complete": True,
        }
        lbody["receipt_sha256"] = sha256_bytes(canonical_json(lbody))
        layer_bodies[layer] = lbody

    rebuilt_main = [layer_bodies[layer]["receipt_sha256"] for layer in layers]
    rebuilt_mtp = layer_bodies[mtp_layer]["receipt_sha256"]

    # ---- prove it against the PUBLISHED digests ---------------------------
    main_receipt = json.loads((root / "main-receipt.json").read_text())
    plan = json.loads((root / "materialization-plan.json").read_text())
    mtp_adapter = json.loads((root / "mtp-adapter-receipt.json").read_text())

    published_main_a = list(main_receipt.get("layer_receipt_sha256") or [])
    published_main_b = list(plan.get("main_layer_receipt_sha256") or [])
    published_mtp = mtp_adapter.get("packed_payload_receipt_sha256")

    checks = {
        "main_receipt.layer_receipt_sha256": rebuilt_main == published_main_a,
        "materialization_plan.main_layer_receipt_sha256": rebuilt_main == published_main_b,
        "mtp_adapter.packed_payload_receipt_sha256": rebuilt_mtp == published_mtp,
    }
    for name, ok in checks.items():
        print(f"  [{'ok' if ok else 'MISMATCH'}] {name}")
    _early = {
        "schema": "malaiwah.glm53-packed-receipt-tree-rematerialization.v1",
        "root": str(root), "contract_sha256": contract_sha256, "bits": bits,
        "num_experts": num_experts, "projections": projections,
        "main_routed_layers": len(layers), "mtp_layer": mtp_layer,
        "choices_indexed": len(index),
        "verified_against_published_digests": checks,
        "rebuilt_main_layer_receipt_sha256": rebuilt_main,
        "rebuilt_mtp_layer_receipt_sha256": rebuilt_mtp,
        "published_main_layer_receipt_sha256": published_main_a,
        "published_mtp_pack_receipt_sha256": published_mtp,
        "reconstruction_succeeded": all(checks.values()),
        "conclusion": (
            "the reader-validated field set reproduces neither published layer digest, "
            "so the original expert receipts carry fields that are NOT recoverable from "
            "the published payload store; the tree cannot be rematerialized without the "
            "originals (JarvisLabs fs 3394, destroyed)"
        ) if not all(checks.values()) else "rebuilt tree reproduces every published digest",
        "applied": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(_early, indent=2, sort_keys=True) + "\n")
    print(f"attempt receipt: {args.receipt}")
    if not all(checks.values()):
        print("\nfirst divergence:", file=sys.stderr)
        for i, (a, b) in enumerate(zip(rebuilt_main, published_main_a)):
            if a != b:
                print(f"  layer index {i}: rebuilt {a} != published {b}", file=sys.stderr)
                break
        raise SystemExit("REFUSING: the rebuilt tree does not reproduce the published "
                         "layer-receipt digests, so it is not the sealed K6 surface")

    receipt = {
        "schema": "malaiwah.glm53-packed-receipt-tree-rematerialization.v1",
        "why": ("partsbin-v1 publishes payload-store + top-level receipts but not the "
                "layers/ and experts/ receipt trees; the only copy lived on JarvisLabs "
                "fs 3394, which no longer exists"),
        "method": ("bodies rebuilt from the published content-addressed choice "
                   "descriptors using the shipped reader's own field contract and "
                   "canonical_json/sha256 seal; nothing is guessed"),
        "root": str(root),
        "contract_sha256": contract_sha256,
        "bits": bits,
        "num_experts": num_experts,
        "projections": projections,
        "main_routed_layers": len(layers),
        "mtp_layer": mtp_layer,
        "choices_indexed": len(index),
        "expert_receipts_built": len(expert_bodies),
        "layer_receipts_built": len(layer_bodies),
        "verified_against_published_digests": checks,
        "rebuilt_main_layer_receipt_sha256": rebuilt_main,
        "rebuilt_mtp_layer_receipt_sha256": rebuilt_mtp,
        "applied": bool(args.apply),
    }

    if args.apply:
        for (layer, expert), body in expert_bodies.items():
            out = root / "experts" / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        for layer, body in layer_bodies.items():
            out = root / "layers" / f"layer-{layer:03d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        print(f"wrote {len(expert_bodies)} expert receipts and {len(layer_bodies)} layer receipts")

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"receipt: {args.receipt}")
    print("ALL PUBLISHED DIGESTS REPRODUCED — the rebuilt tree is the sealed surface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
