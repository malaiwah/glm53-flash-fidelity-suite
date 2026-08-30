#!/usr/bin/env python3
"""Offline selftest for the STREAMING dione front-end (stream_score --source dione).

Runs with torch + safetensors only; the MCG decode rungs self-skip when
quant_pipeline (the campaign reader that owns the frozen MCG LUT) is not
importable, exactly like selftest_tr3_offline.  The box-side setup always has
the pipeline, so those rungs run there before any paid capture.

The fixture is not a toy.  It carries the REAL index shape -- 42 routed layers
x 288 experts x 3 projections x 4 TP ranks x 4 payload objects = 580,608 packed
names, the REAL 1,618 official non-routed names and the REAL 864 MTP-layer
expert names, 583,090 in total, the same number the live release's 65 MB index
holds -- and REAL K3/TP4 slice geometry for the modules it actually decodes.
Only the number of decoded modules, and the byte size of the natives, is small.

  [1] both published manifest spellings and BOTH manifest schemas load, and
      agree about source repo/revision, tp_size and bits
  [2] the name census is load-bearing: a missing slice, a stray name, a routed
      original shipped natively and an absent MTP expert are each refused
  [3] shard verification: a matching manifest passes, a tampered shard is
      refused, a weight file the manifest does not cover is refused, and a
      stale marker is refused
  [4] the published scope reports no `unknown`, validates against
      artifact.schema.json, and its digest is stable across loads
  [5] the materializer emits EXACTLY the official 1,618 names, excludes the 864
      MTP expert tensors, copies every dtype verbatim, and seals an inventory +
      receipt in the schemas stream_score binds against
  [6] the streaming non-routed VIEW over that tree filters exactly the routed
      entries and references only shards that exist
  [7] the payload reader returns tp_size geometry-checked slices, and refuses a
      wrong dtype/shape and a bad MCG marker
  [8] decode parity: decode_module_payload is bitwise load_decoded_module, and
      hash_payloads=False changes the census but never the tensor
      (needs quant_pipeline)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import dione_surface as ds  # noqa: E402
import tr3_surface as t3  # noqa: E402

RESULTS = []
REV = "8b099bf276507a17faea920deff3f62d5597fb52"
SRC_REV = "a6c167b62691b2bac901344b65cb651a70f53e43"
BITS = 3
TP = 4
LAYERS = ds.MAIN_ROUTED_LAYERS
EXPERTS = ds.NUM_EXPERTS
PROJ = ds.PROJECTIONS
FP32_SUFFIXES = (".A_log", ".dt_bias", ".e_score_correction_bias")
LAYER_SHARD = "layers/layer-03-part-0.safetensors"
RETAINED_SHARD = "retained/retained-00001-of-00120.safetensors"
DECODABLE = (3, 0)          # (layer, expert) whose 12 slice groups exist for real


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"selftest_dione_stream_offline: {name} failed: {detail}")


def skip(name, why):
    RESULTS.append((name, True, f"SKIPPED: {why}"))
    print(f"[skip] {name} - {why}")


def refuses(name, fn, needle):
    try:
        fn()
    except ValueError as exc:
        ok = needle.lower() in str(exc).lower()
        check(name, ok, str(exc)[:120])
        return
    check(name, False, "did NOT refuse")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
def build_weight_map(*, drop_slice=None, stray=None, native_routed=False,
                     drop_mtp=False):
    weight_map = {}
    for layer in LAYERS:
        for expert in range(EXPERTS):
            for proj in PROJ:
                for rank in range(TP):
                    for obj in ds.OBJECTS:
                        name = ds.slice_name(layer, expert, proj, rank, obj)
                        if drop_slice is not None and name == drop_slice:
                            continue
                        weight_map[name] = LAYER_SHARD
    for name in t3.official_nonrouted_names():
        weight_map[name] = RETAINED_SHARD
    if not drop_mtp:
        for expert in range(EXPERTS):
            for proj in PROJ:
                weight_map[ds.official_name(ds.MTP_LAYER, expert, proj)] = RETAINED_SHARD
    if native_routed:
        weight_map[ds.official_name(LAYERS[0], 0, "gate_proj")] = RETAINED_SHARD
    if stray:
        weight_map[stray] = LAYER_SHARD
    return weight_map


def real_slices():
    """Real K3/TP4 payload objects for one expert's three projections."""
    layer, expert = DECODABLE
    out = {}
    generator = torch.Generator().manual_seed(20260830)
    for proj in PROJ:
        geometry = ds.expected_slice_geometry(proj, bits=BITS, tp_size=TP)
        for rank in range(TP):
            trellis = torch.randint(-32768, 32767, geometry["trellis"][1],
                                    generator=generator, dtype=torch.int16)
            suh = torch.randn(geometry["suh"][1], generator=generator).to(torch.float16)
            svh = torch.randn(geometry["svh"][1], generator=generator).to(torch.float16)
            out[ds.slice_name(layer, expert, proj, rank, "trellis")] = trellis
            out[ds.slice_name(layer, expert, proj, rank, "suh")] = suh
            out[ds.slice_name(layer, expert, proj, rank, "svh")] = svh
            out[ds.slice_name(layer, expert, proj, rank, "mcg")] = torch.tensor(
                ds.MCG_MARKER_SIGNED_INT32, dtype=torch.int32)
    return out


def build_release(root: Path, *, manifest_name="EXL3_MANIFEST.json",
                  manifest_schema="v1", weight_map=None, quant_overrides=None,
                  bad_marker_tensor=None, extra_disk_shard=False):
    """A structurally real mini Dione release."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "layers").mkdir(exist_ok=True)
    (root / "retained").mkdir(exist_ok=True)
    weight_map = weight_map if weight_map is not None else build_weight_map()

    routed = real_slices()
    if bad_marker_tensor:
        routed[bad_marker_tensor] = torch.tensor(12345, dtype=torch.int32)
    save_file(routed, str(root / LAYER_SHARD))

    natives = {}
    for name in t3.official_nonrouted_names():
        dtype = torch.float32 if name.endswith(FP32_SUFFIXES) else torch.bfloat16
        natives[name] = torch.zeros(2, dtype=dtype)
    for expert in range(EXPERTS):
        for proj in PROJ:
            natives[ds.official_name(ds.MTP_LAYER, expert, proj)] = torch.zeros(
                2, dtype=torch.bfloat16)
    save_file(natives, str(root / RETAINED_SHARD))
    if extra_disk_shard:
        save_file({"stray": torch.zeros(1)}, str(root / "retained" / "extra.safetensors"))

    quant = {
        "bits_per_weight": float(BITS), "format": ds.DIONE_FORMAT,
        "mcg": True, "quant_method": ds.DIONE_QUANT_METHOD,
        "requires_custom_loader": True, "retained_dtype": "source_precision",
        "source_revision": SRC_REV, "target_expert_bpw": float(BITS),
        "tensor_parallel_size": TP, "trellis_k": BITS,
        "quantized_scope": "model.language_model.layers.3..44.mlp.experts.0..287."
                           "{gate_proj,up_proj,down_proj}.weight",
        "retained_scope": "attention, indexers, mHC, routers, shared experts, dense "
                          "layers 0-2, embeddings, lm_head, norms, vision, MTP",
    }
    quant.update(quant_overrides or {})
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "quantization_config": quant,
        "text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 45,
                        "first_k_dense_replace": 3, "n_routed_experts": EXPERTS,
                        "num_nextn_predict_layers": 1, "hidden_size": 4096,
                        "moe_intermediate_size": 2048, "vocab_size": 154880},
    }, indent=2), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}),
        encoding="utf-8")

    shards = [LAYER_SHARD, RETAINED_SHARD]
    if manifest_schema == "v1":
        manifest = {
            "schema_version": 1,
            "target_bpw": float(BITS),
            "runtime": {"tensor_parallel_size": TP, "status": "pending",
                        "cuda_graphs_required": True},
            "source": {"repo_id": "zai-org/GLM-5.3-Flash-BF16",
                       "sealed_revision": SRC_REV, "format": "bf16"},
            "files": [{"path": s, "sha256": sha_file(root / s),
                       "bytes": (root / s).stat().st_size} for s in shards],
        }
    else:  # the Q4 spelling
        manifest = {
            "schema": "glm53-selective-exl3-k4-tp4-v1",
            "bits_per_weight": float(BITS),
            "tensor_parallel_size": TP,
            "source_repo": "zai-org/GLM-5.3-Flash-BF16",
            "source_revision": SRC_REV,
            "quantized_shards": [{"name": Path(LAYER_SHARD).name,
                                  "sha256": sha_file(root / LAYER_SHARD),
                                  "bytes": (root / LAYER_SHARD).stat().st_size}],
            "retained_shards": [{"name": Path(RETAINED_SHARD).name,
                                 "sha256": sha_file(root / RETAINED_SHARD),
                                 "bytes": (root / RETAINED_SHARD).stat().st_size}],
        }
    (root / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="dione-selftest-"))
    try:
        return run(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(work: Path) -> int:
    # [1] both manifest spellings and both manifest schemas -------------------
    good = build_release(work / "good")
    surface = ds.load_dione_surface(good, repo="fixture/dione", revision=REV,
                                    require_shard_hashes=False)
    check("a well-formed release loads", surface.bits == BITS and surface.tp_size == TP,
          "bits=%s tp=%s" % (surface.bits, surface.tp_size))
    check("the uppercase manifest spelling is found",
          surface.exl3_manifest_name == "EXL3_MANIFEST.json"
          and surface.exl3_manifest_schema == "schema_version=1",
          "%s / %s" % (surface.exl3_manifest_name, surface.exl3_manifest_schema))
    check("the manifest binds the official source revision",
          surface.source_revision == SRC_REV
          and surface.source_repo == "zai-org/GLM-5.3-Flash-BF16")
    check("the census closes: 1,618 official natives + 864 MTP experts retained",
          len(surface.retained_names) == 2482
          and len([n for n in surface.retained_names
                   if ds._ROUTED.search(n) is None]) == 1618)

    q4 = build_release(work / "q4-spelling", manifest_name="exl3-manifest.json",
                       manifest_schema="q4")
    q4_surface = ds.load_dione_surface(q4, repo="fixture/dione", revision=REV,
                                       require_shard_hashes=False)
    check("the Q4 manifest spelling AND schema still load",
          q4_surface.exl3_manifest_name == "exl3-manifest.json"
          and q4_surface.exl3_manifest_schema == "glm53-selective-exl3-k4-tp4-v1"
          and q4_surface.source_revision == SRC_REV)

    # The case/underscore FOLD, exercised without relying on the filesystem's
    # own case sensitivity: on macOS "exl3_manifest.json" and
    # "EXL3_MANIFEST.json" are the SAME path, so the direct-name branch answers
    # first and the scan never runs.  Emptying the known-names list forces the
    # fold to be the thing under test on any filesystem.
    lower = build_release(work / "underscore", manifest_name="exl3_manifest.json")
    saved_names = ds.MANIFEST_NAMES
    ds.MANIFEST_NAMES = ()
    try:
        found = ds.find_manifest(lower)
        check("a case/underscore spelling is found by the fold, not the name list",
              found is not None
              and found.name.lower().replace("_", "-") == "exl3-manifest.json",
              str(found))
        check("a directory with no manifest at all folds to None",
              ds.find_manifest(work / "underscore" / "layers") is None)
    finally:
        ds.MANIFEST_NAMES = saved_names

    refuses("a manifest matching neither schema is refused, not guessed at",
            lambda: ds.parse_manifest({"shards": ["a"]}), "neither published schema")
    refuses("a manifest that disagrees with config on tp_size is refused",
            lambda: ds.load_dione_surface(
                build_release(work / "tp-mismatch",
                              quant_overrides={"tensor_parallel_size": 2}),
                require_shard_hashes=False),
            "only the published tp4 slicing")

    # [2] the census is load-bearing -----------------------------------------
    missing_name = ds.slice_name(7, 100, "up_proj", 2, "suh")
    refuses("a missing routed payload tensor is refused",
            lambda: ds.census_weight_map(build_weight_map(drop_slice=missing_name),
                                         tp_size=TP),
            "routed payload tensors absent")
    refuses("a routed name outside the declared scope is refused",
            lambda: ds.census_weight_map(
                build_weight_map(stray=ds.slice_name(45, 0, "gate_proj", 0, "trellis")),
                tp_size=TP),
            "outside the declared scope")
    refuses("a layer shipping BOTH packed and native routed tensors is refused",
            lambda: ds.census_weight_map(build_weight_map(native_routed=True),
                                         tp_size=TP),
            "both packed and native")
    refuses("an absent MTP native expert is refused",
            lambda: ds.census_weight_map(build_weight_map(drop_mtp=True), tp_size=TP),
            "mtp layer 45 native experts absent")

    # [3] shard verification --------------------------------------------------
    record = ds.verify_shard_hashes(good)
    check("verify-shards passes on an intact release",
          record["all_verified"] is True and record["shards"] == 2
          and record["shards_on_disk"] == 2)
    check("the marker gates a `full` load",
          ds.load_dione_surface(good, repo="fixture/dione", revision=REV,
                                require_shard_hashes=True).shard_hash_verification
          == "full")

    tampered = build_release(work / "tampered")
    ds.verify_shard_hashes(tampered)
    with open(tampered / RETAINED_SHARD, "r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 0xFF]))
    refuses("a tampered shard is refused by verify-shards",
            lambda: ds.verify_shard_hashes(tampered), "shard hash differs")

    uncovered = build_release(work / "uncovered", extra_disk_shard=True)
    refuses("a weight file the manifest does not cover is refused",
            lambda: ds.verify_shard_hashes(uncovered), "not covered by")

    stale = build_release(work / "stale")
    ds.verify_shard_hashes(stale)
    marker = json.loads((stale / "dione-shards-verified.json").read_text())
    marker["exl3_manifest_sha256"] = "0" * 64
    (stale / "dione-shards-verified.json").write_text(json.dumps(marker))
    refuses("a stale verification marker is refused",
            lambda: ds.load_dione_surface(stale, require_shard_hashes=True),
            "stale/foreign")

    nomanifest = build_release(work / "nomanifest")
    (nomanifest / "EXL3_MANIFEST.json").unlink()
    refuses("verify-shards with no manifest at all is refused",
            lambda: ds.verify_shard_hashes(nomanifest), "no release manifest")

    # [4] the published scope -------------------------------------------------
    scope = ds.published_scope(surface)
    digest = ds.scope_digest(surface)
    check("the published scope contains no `unknown`",
          "unknown" not in digest, digest)
    check("the routed experts are the only quantized class",
          sorted(a["tensor_class"] for a in scope["assignments"]
                 if a["treatment"] == "quantized") == ["moe.experts"])
    check("the head is declared NATIVE, not quantized",
          scope["head_policy"] == "native"
          and next(a for a in scope["assignments"]
                   if a["tensor_class"] == "lm_head")["treatment"] == "native")
    # artifact.schema.json's `scope` is additionalProperties:false, so a stray
    # key here is a REJECTED submission at seal time -- after both cold runs
    # are paid for.  Check it against the schema file, not a remembered list.
    schema_path = TOOLS.parent.parent / "registry" / "schema" / "artifact.schema.json"
    if schema_path.is_file():
        artifact_schema = json.loads(schema_path.read_text())
        allowed = set(artifact_schema["properties"]["scope"]["properties"])
        required = set(artifact_schema["properties"]["scope"]["required"])
        check("scope carries ONLY keys artifact.schema.json allows",
              set(scope) <= allowed and required <= set(scope),
              "extra=%s missing=%s" % (sorted(set(scope) - allowed),
                                       sorted(required - set(scope))))
        item_allowed = set(artifact_schema["properties"]["scope"]["properties"]
                           ["assignments"]["items"]["properties"])
        bad = [a["tensor_class"] for a in scope["assignments"]
               if not set(a) <= item_allowed]
        check("every assignment carries only allowed keys", not bad, str(bad))
        common = json.loads((schema_path.parent / "common.schema.json").read_text())
        classes = set(common["$defs"]["tensor_class"]["enum"])
        formats = set(common["$defs"]["numeric_format"]["enum"])
        unknown_cls = [a["tensor_class"] for a in scope["assignments"]
                       if a["tensor_class"] not in classes]
        unknown_fmt = [a["format"] for a in scope["assignments"]
                       if a["format"] not in formats]
        check("every tensor_class and format is in the registry vocabulary",
              not unknown_cls and not unknown_fmt,
              "classes=%s formats=%s" % (unknown_cls, unknown_fmt))
        head_enum = set(artifact_schema["properties"]["scope"]["properties"]
                        ["head_policy"]["enum"])
        kv_enum = set(artifact_schema["properties"]["scope"]["properties"]
                      ["kv_cache_dtype"]["enum"])
        policy_enum = set(artifact_schema["properties"]["scope"]["properties"]
                          ["policy"]["enum"])
        check("head_policy / kv_cache_dtype / policy are in their enums",
              scope["head_policy"] in head_enum
              and scope["kv_cache_dtype"] in kv_enum
              and scope["policy"] in policy_enum)
        report = ds.scope_report(surface)
        check("scope_report keeps provenance OUTSIDE the scope object",
              report["scope"] == scope and "source" in report
              and "schema" in report and "source" not in report["scope"])
    else:
        skip("scope validates against artifact.schema.json", "schema not present")
    check("scope digest is stable across loads",
          ds.scope_digest(ds.load_dione_surface(good, repo="fixture/dione",
                                                revision=REV)) == digest)

    # [5] the materializer ----------------------------------------------------
    out = work / "materialized"
    receipt = ds.materialize_nonrouted(surface, out)
    check("the materializer writes exactly the official 1,618 non-routed tensors",
          receipt["written_tensor_count"] == 1618
          and receipt["decoded_tensor_count"] == 0)
    check("the 864 MTP expert tensors are excluded from the measured tree",
          receipt["mtp_expert_tensors_excluded"] == 864)
    check("dtypes are copied VERBATIM (fp32 natives stay fp32)",
          receipt["dtype_census"].get("float32") == len(
              [n for n in t3.official_nonrouted_names() if n.endswith(FP32_SUFFIXES)])
          and receipt["dtype_census"].get("bfloat16") == 1618 - len(
              [n for n in t3.official_nonrouted_names() if n.endswith(FP32_SUFFIXES)]),
          json.dumps(receipt["dtype_census"], sort_keys=True))
    mat_index = json.loads((out / "model.safetensors.index.json").read_text())
    produced = {n for n in mat_index["weight_map"] if ds._ROUTED.search(n) is None}
    check("materialized name set == the official non-routed set",
          produced == set(t3.official_nonrouted_names()), "%d produced" % len(produced))
    check("virtual routed entries cover layers 3-44 AND the MTP layer",
          receipt["virtual_routed_entries"] == (len(LAYERS) + 1) * EXPERTS * len(PROJ))
    check("the materialized config carries no quantization_config",
          "quantization_config" not in json.loads((out / "config.json").read_text()))
    inventory = json.loads((out / "inventory.json").read_text())
    check("the inventory is sealed in the schema stream_score binds against",
          inventory["schema"] == ds.RELEASE_INVENTORY_SCHEMA
          and inventory["seal_mode"] == "full-shard-sha256"
          and inventory["model_revision"] == REV)
    check("the receipt binds THIS snapshot's config/index digests",
          receipt["schema"] == ds.DIONE_MATERIALIZATION_SCHEMA
          and receipt["source_config_sha256"] == surface.config_sha256
          and receipt["source_index_sha256"] == surface.index_sha256
          and receipt["inventory_sha256"] == inventory["inventory_sha256"])
    # the seal is a real digest over the body, not a decoration
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    check("the materialization receipt's own digest reproduces",
          hashlib.sha256(ds._canonical_json(body)).hexdigest()
          == receipt["receipt_sha256"])
    refuses("a non-routed set that differs from the official one is refused "
            "BEFORE anything is read",
            lambda: ds.materialize_nonrouted(surface, work / "mat-bad",
                                             official_names=("not.a.real.tensor",)),
            "differs from the official")

    # [6] the streaming non-routed view over that tree -------------------------
    import stream_score as ss

    view, view_record = ss.prepare_nonrouted_view(out, work / "viewdir")
    check("the view keeps 1,618 tensors and filters every routed entry",
          view_record["nonrouted_tensor_count"] == 1618
          and view_record["routed_tensor_count_filtered"]
          == (len(LAYERS) + 1) * EXPERTS * len(PROJ))
    view_index = json.loads((view / "model.safetensors.index.json").read_text())
    referenced = sorted(set(view_index["weight_map"].values()))
    check("every shard the view's index names exists on disk",
          all((view / shard).exists() for shard in referenced), str(referenced))

    # [7] the payload reader --------------------------------------------------
    reader = ds.DioneShardReader(surface)
    payloads = reader.payload(DECODABLE[0], DECODABLE[1], "gate_proj")
    check("payload() returns tp_size geometry-checked slices in rank order",
          len(payloads) == TP and [p["rank"] for p in payloads] == list(range(TP))
          and tuple(payloads[0]["trellis"].shape)
          == ds.expected_slice_geometry("gate_proj", bits=BITS, tp_size=TP)["trellis"][1])
    marker_name = ds.slice_name(DECODABLE[0], DECODABLE[1], "gate_proj", 1, "mcg")
    bad = build_release(work / "badmarker", bad_marker_tensor=marker_name)
    bad_surface = ds.load_dione_surface(bad, repo="fixture/dione", revision=REV,
                                        require_shard_hashes=False)
    refuses("a wrong MCG marker is refused at read time",
            lambda: ds.DioneShardReader(bad_surface).payload(
                DECODABLE[0], DECODABLE[1], "gate_proj"),
            "mcg marker differs")

    # threads: the reader must be usable from a pool, which is the whole point
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda proj: reader.payload(DECODABLE[0], DECODABLE[1], proj), PROJ))
    check("the reader serves a thread pool (thread-local handles)",
          all(len(r) == TP for r in results)
          and torch.equal(results[0][0]["trellis"], payloads[0]["trellis"]))

    # [8] decode parity -------------------------------------------------------
    try:
        ds._reader()
    except Exception as exc:  # noqa: BLE001 - the pipeline is box-side only
        skip("decode_module_payload is bitwise load_decoded_module",
             "quant_pipeline not importable (%s)" % type(exc).__name__)
        skip("hash_payloads changes the census, never the tensor", "same reason")
    else:
        split, split_census = ds.decode_module_payload(
            surface, payloads, layer=DECODABLE[0], expert=DECODABLE[1],
            projection="gate_proj", device="cpu", hash_payloads=True)
        whole, whole_census = ds.load_decoded_module(
            surface, reader, layer=DECODABLE[0], expert=DECODABLE[1],
            projection="gate_proj", device="cpu", hash_payloads=True)
        check("decode_module_payload is bitwise load_decoded_module",
              torch.equal(split, whole) and split_census == whole_census)
        nohash, nohash_census = ds.decode_module_payload(
            surface, payloads, layer=DECODABLE[0], expert=DECODABLE[1],
            projection="gate_proj", device="cpu", hash_payloads=False)
        check("hash_payloads changes the census, never the tensor",
              torch.equal(split, nohash) and nohash_census["slices"] == []
              and len(split_census["slices"]) == TP)

    print("\n%d rungs, all green" % len(RESULTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
