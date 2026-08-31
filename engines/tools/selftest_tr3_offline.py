#!/usr/bin/env python3
"""Offline selftest for tr3_surface (the SEALED TR3-published EXL3/MCG reader).

Runs with torch + safetensors only; the MCG decode rung self-skips when
quant_pipeline (the campaign reader that owns the frozen MCG LUT) is not
importable, exactly like selftest_exl3hf_offline.  The box-side setup always
has the pipeline, so that rung runs there before any paid capture.

The fixture is not a toy: it carries the REAL 1,618 official non-routed names
and the REAL 150,226-name index shape (43 routed layers x 288 experts x 3
projections x 4 payload objects + 1,618 natives), so every seal claim -- the
name-set digest, the count algebra, the official-name bijection -- is exercised
against the same arithmetic the live release satisfies.  Only the BYTES are
small.

  [1] a well-formed sealed release loads, and all 12 seal claims reproduce
  [2] every seal claim is LOAD-BEARING: tamper with one, get a refusal
  [3] scope refusals: wrong codebook / wrong scope / wrong non-routed policy /
      a quantized head are each refused BEFORE any decode
  [4] the shard binding: SHA256SUMS agreement is proven, disagreement refused,
      and `full` re-hashing agrees with both
  [5] the decode is exl3hf's, verbatim: tr3.expert_source's decode is bitwise
      equal to calling exl3hf_surface directly on the same payload
  [6] the published scope reports no `unknown`, and the digest is stable
  [7] the routed census closes the executed surface and names the MTP layer as
      present-but-not-executed
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import exl3hf_surface as xs  # noqa: E402
import tr3_surface as t3  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"selftest_tr3_offline: {name} failed: {detail}")


def skip(name, why):
    RESULTS.append((name, True, f"SKIPPED: {why}"))
    print(f"[skip] {name} - {why}")


def refuses(name, fn, needle):
    try:
        fn()
    except ValueError as exc:
        ok = needle.lower() in str(exc).lower()
        check(name, ok, str(exc)[:110])
        return
    check(name, False, "did NOT refuse")


ROUTED_LAYERS = tuple(range(3, 46))          # 3..44 executed + 45 MTP
EXPERTS = 288
PROJ = ("gate_proj", "up_proj", "down_proj")
FP32_SUFFIXES = (".A_log", ".dt_bias", ".e_score_correction_bias")
NONROUTED_SHARD = "model-00001-of-00002.safetensors"
ROUTED_SHARD = "model-00002-of-00002.safetensors"

# real payload geometry, from the live release's shard headers
GEOM = {"gate_proj": ((256, 128), 4096, 2048),
        "up_proj": ((256, 128), 4096, 2048),
        "down_proj": ((128, 256), 2048, 4096)}
DECODABLE = [(3, 0, "gate_proj"), (3, 0, "down_proj"), (44, 287, "up_proj")]


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_release(root: Path, *, bits=4, codebook="mcg", scope=t3.EXPECTED_SCOPE,
                  nonrouted_policy=t3.EXPECTED_NONROUTED_POLICY, head_bits=16,
                  mutate_index=None, mutate_abi=None, mutate_mat=None,
                  mutate_sums=None):
    """Write a structurally real, self-consistently sealed mini TR3 release."""
    root.mkdir(parents=True, exist_ok=True)
    official = list(t3.official_nonrouted_names())

    weight_map = {name: NONROUTED_SHARD for name in official}
    for layer in ROUTED_LAYERS:
        for expert in range(EXPERTS):
            for proj in PROJ:
                module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
                for obj in ("trellis", "suh", "svh", codebook):
                    weight_map[f"{module}.{obj}"] = ROUTED_SHARD

    # --- the two shard files ------------------------------------------------
    natives = {}
    for name in official:
        dtype = torch.float32 if name.endswith(FP32_SUFFIXES) else torch.bfloat16
        natives[name] = torch.zeros(2, dtype=dtype)
    save_file(natives, str(root / NONROUTED_SHARD))

    gen = torch.Generator().manual_seed(20260829)
    routed = {}
    marker = torch.tensor([xs.CODEBOOK_OBJECTS[codebook]], dtype=torch.int32)
    for layer, expert, proj in DECODABLE:
        module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        (kt, nt), in_f, out_f = GEOM[proj]
        routed[f"{module}.trellis"] = torch.randint(
            -32768, 32767, (kt, nt, bits * 16), generator=gen, dtype=torch.int16)
        routed[f"{module}.suh"] = (
            torch.randint(0, 2, (in_f,), generator=gen).float() * 2 - 1).half()
        routed[f"{module}.svh"] = (
            (torch.randint(0, 2, (out_f,), generator=gen).float() * 2 - 1) * 0.02).half()
        routed[f"{module}.{codebook}"] = marker.clone()
    save_file(routed, str(root / ROUTED_SHARD))

    # --- config -------------------------------------------------------------
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "quantization_config": {
            "bits": bits, "codebook": codebook, "head_bits": head_bits,
            "non_routed_dtype_policy": nonrouted_policy, "quant_method": "exl3",
            "scope": scope, "serving_reader_qualified": False, "version": "0.0.43",
        },
        "text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 45},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    index = {"metadata": {"total_size": 1}, "weight_map": weight_map}
    if mutate_index:
        mutate_index(index)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8")

    # --- the seal, computed over what was just written ----------------------
    names_digest = sha_bytes(canonical(sorted(index["weight_map"])))
    # Counts are DERIVED from the index that was actually written, so a tamper
    # that changes the name set leaves the count algebra self-consistent and
    # only the name-set check can fire.  A fixture whose tampers trip two
    # checks at once cannot prove either one is load-bearing.
    emitted = index["weight_map"]
    native_count = sum(1 for n in emitted if t3._ROUTED.search(n) is None)
    packed_count = len(emitted) - native_count
    routed_choices = packed_count // 4
    shard_sha = {name: sha_file(root / name)
                 for name in (NONROUTED_SHARD, ROUTED_SHARD)}
    mat_body = {
        "schema": "quant-pipeline.glm53-k4-materialization-receipt.v1",
        "bits": bits, "codec_family": "exl3-mcg", "complete": True,
        "main_and_mtp_complete": True,
        "config_sha256": sha_file(root / "config.json"),
        "index_sha256": sha_file(root / "model.safetensors.index.json"),
        "mcg_multiplier_hex": "0xCBAC1FED",
        "native_tensor_count": native_count,
        "nonrouted_native_exact": True,
        "output_logical_bytes": 1,
        "output_tensor_count": len(emitted),
        "output_tensor_names_sha256": names_digest,
        "packed_tensor_count": packed_count,
        "plan_sha256": "a" * 64,
        "quantization_config_sha256": "b" * 64,
        "routed_choice_count": routed_choices,
        "serving_reader_qualified": False,
        "shard_sha256": shard_sha,
    }
    if mutate_mat:
        mutate_mat(mat_body)
    mat = dict(mat_body, receipt_sha256=sha_bytes(canonical(mat_body)))
    (root / t3.MATERIALIZATION_FILE).write_text(
        json.dumps(mat, indent=2) + "\n", encoding="utf-8")

    abi = {
        "schema": t3.ABI_SCHEMA, "bits": bits, "codec_family": "exl3-mcg",
        "exllamav3": {"git_commit": "c" * 40, "version": "0.0.43",
                      "module_key_rule": "official_weight_name_without_.weight",
                      "written_suffixes": ["trellis", "suh", "svh", codebook]},
        "mcg_multiplier_hex": "0xCBAC1FED",
        "output_tensor_count": mat_body["output_tensor_count"],
        "output_tensor_names_sha256": mat_body["output_tensor_names_sha256"],
        "packed_reader_abi_sha256": "d" * 64,
        "plan_sha256": mat_body["plan_sha256"],
        "qualified_tp_sizes": [],
        "reason": "ExLlamaV3 v0.0.43 has no audited GLM-5.3 TP model load/inference receipt",
        "receipt_sha256": "e" * 64,
        "serving_reader_qualified": False,
        "storage_checkpoint_verified": True,
    }
    if mutate_abi:
        mutate_abi(abi)
    (root / t3.ABI_FILE).write_text(json.dumps(abi, indent=2) + "\n", encoding="utf-8")

    sums = {name: digest for name, digest in shard_sha.items()}
    if mutate_sums:
        mutate_sums(sums)
    (root / "SHA256SUMS").write_text(
        "".join("%s  %s\n" % (digest, name) for name, digest in sorted(sums.items())),
        encoding="utf-8")
    return root


REV = "0" * 40
work = Path(tempfile.mkdtemp(prefix="tr3-selftest-"))
try:
    # [1] the happy path ------------------------------------------------------
    good = build_release(work / "good")
    surface = t3.load_tr3_surface(good, repo="fixture/tr3", revision=REV)
    checks = surface.seal["checks"]
    check("sealed release loads", surface.declared_bits == 4.0 and surface.codebook == "mcg")
    check("every seal claim reproduced", all(c["passed"] for c in checks),
          "%d checks" % len(checks))
    check("seal covers all 12 published claims", len(checks) == 12,
          ", ".join(c["check"] for c in checks))
    check("index shape is the real one", len(surface.exl3.weight_map) == 150226,
          str(len(surface.exl3.weight_map)))
    check("non-routed census", surface.nonrouted_tensor_count == 1618,
          str(surface.nonrouted_tensor_count))
    check("routed module census", surface.routed_module_count == 43 * 288 * 3,
          str(surface.routed_module_count))
    check("dtype census read from shard headers, not assumed",
          surface.dtype_census.get("F32", 0) > 0 and surface.dtype_census.get("BF16", 0) > 0,
          json.dumps(surface.dtype_census, sort_keys=True))

    # [2] every claim is load-bearing ----------------------------------------
    def tamper(name, **kwargs):
        root = work / ("bad-" + name)
        shutil.rmtree(root, ignore_errors=True)
        build_release(root, **kwargs)
        return lambda: t3.load_tr3_surface(root, repo="x", revision=REV)

    refuses("tampered name digest is caught",
            tamper("names", mutate_abi=lambda a: a.update(
                output_tensor_names_sha256="f" * 64)),
            "output_tensor_names_sha256")
    refuses("plan disagreement is caught",
            tamper("plan", mutate_abi=lambda a: a.update(plan_sha256="9" * 64)),
            "plan_sha256_agreement")
    refuses("count algebra is caught",
            tamper("counts", mutate_mat=lambda m: m.update(routed_choice_count=1)),
            "payload_objects_per_choice")
    refuses("a non-exact non-routed claim is caught",
            tamper("nonexact", mutate_mat=lambda m: m.update(nonrouted_native_exact=False)),
            "nonrouted_native_exact")
    refuses("a missing official non-routed name is caught",
            tamper("names-missing",
                   mutate_index=lambda i: i["weight_map"].pop("lm_head.weight")),
            "nonrouted_name_set_equals_official")
    refuses("a foreign non-routed name is caught",
            tamper("names-extra",
                   mutate_index=lambda i: i["weight_map"].update(
                       {"model.language_model.not_official.weight": NONROUTED_SHARD})),
            "nonrouted_name_set_equals_official")
    refuses("a wrong MCG multiplier is caught",
            tamper("mult", mutate_abi=lambda a: a.update(mcg_multiplier_hex="0xDEADBEEF")),
            "mcg_multiplier")

    # the receipt's own self-seal: rewrite the file with a stale digest
    stale = work / "bad-selfseal"
    shutil.rmtree(stale, ignore_errors=True)
    build_release(stale)
    doc = json.loads((stale / t3.MATERIALIZATION_FILE).read_text())
    doc["output_logical_bytes"] = 2          # body changed, receipt_sha256 not
    (stale / t3.MATERIALIZATION_FILE).write_text(json.dumps(doc, indent=2) + "\n")
    refuses("materialization receipt self-seal is checked",
            lambda: t3.load_tr3_surface(stale, repo="x", revision=REV),
            "self_seal")

    # [3] scope refusals ------------------------------------------------------
    refuses("a mul1 release is not a TR3 release",
            tamper("mul1", codebook="mul1"), "codebook")
    refuses("a full-scope release is refused",
            tamper("scope", scope="everything"), "scope")
    refuses("a non-official non-routed policy is refused",
            tamper("policy", nonrouted_policy="dequantized"), "non_routed_dtype_policy")
    refuses("a quantized head is refused by THIS surface",
            tamper("head", head_bits=6), "head_bits")
    refuses("a non-immutable revision is refused",
            lambda: t3.load_tr3_surface(good, repo="x", revision="main"),
            "40-hex")

    # [4] the shard binding ---------------------------------------------------
    check("crosscheck proves the receipt map == published SHA256SUMS",
          surface.shard_verification["agreed"] == 2,
          surface.shard_verification["verification"])
    full = t3.verify_shard_digests(good, mode="full")
    check("full re-hash agrees with the receipt map", full["agreed"] == 2)
    refuses("SHA256SUMS disagreeing with the seal is refused",
            tamper("sums", mutate_sums=lambda s: s.update(
                {NONROUTED_SHARD: "0" * 64})),
            "disagree")
    noskip = t3.load_tr3_surface(good, repo="x", revision=REV, verify_shards="skip")
    check("skip is disclosed, not silent",
          noskip.shard_verification["mode"] == "skip"
          and "NOT bound" in noskip.shard_verification["verification"])

    # [5] the decode is exl3hf's, verbatim ------------------------------------
    exl3, reader = t3.expert_source(surface)
    check("expert_source hands back the exl3hf pair",
          isinstance(reader, xs.Exl3HfShardReader) and exl3 is surface.exl3)
    try:
        from quant_pipeline.evaluation.glm53_packed_k4_reader import mcg_lut  # noqa: F401
        have_pipeline = True
    except Exception:                                     # noqa: BLE001
        have_pipeline = False
    if not have_pipeline:
        skip("tr3 decode == exl3hf decode (bitwise)",
             "quant_pipeline absent (the MCG LUT lives there); runs on the box")
    else:
        layer, expert, proj = DECODABLE[0]
        via_tr3, census = xs.load_decoded_module(
            exl3, reader, layer=layer, expert=expert, projection=proj, device="cpu")
        module = xs.routed_module_name(layer, expert, proj)
        payload = xs.Exl3HfShardReader(surface.exl3).payload(module)
        direct, _ = xs.decode_module(surface.exl3, payload, module=module, device="cpu",
                                     expected_shape=xs.PROJECTION_SHAPE[proj])
        check("tr3 decode == exl3hf decode (bitwise)", torch.equal(via_tr3, direct),
              "shape %s, K%d" % (tuple(via_tr3.shape), census["bits"]))
        check("decoded bits read from the trellis shape", census["bits"] == 4)
        check("decoded matrix is finite", bool(torch.isfinite(via_tr3).all()))
        bad_marker = dict(payload)
        bad_marker["marker"] = torch.tensor([1234], dtype=torch.int32)
        refuses("a wrong per-module codebook marker is refused at decode",
                lambda: xs.decode_module(surface.exl3, bad_marker, module=module,
                                         device="cpu"),
                "marker differs")

    # [6] the published scope -------------------------------------------------
    scope = t3.published_scope(surface)
    digest = t3.scope_digest(surface)
    classes = {a["tensor_class"] for a in scope["assignments"]}
    check("scope names every class the registry knows",
          {"embed_tokens", "attn.qkv", "attn.o", "mlp.gate", "mlp.up", "mlp.down",
           "moe.experts", "moe.router", "moe.shared_expert", "norm", "lm_head",
           "mtp", "other"} <= classes, ", ".join(sorted(classes)))
    check("no class is recorded as unknown", "unknown" not in digest, digest[:120])
    check("the head is native in the digest", "lm_head=native:bf16@16" in digest
          and "head=native" in digest)
    # The digest must equal what the registry already holds for these weights.
    # A mirror's scope that disagrees with its upstream's is refused at
    # submission with exit 7 -- on the box, at the seal stage, after both cold
    # runs. Compare against the seeded record instead.
    seeded = TOOLS.parent.parent / "registry" / "data" / "artifacts.jsonl"
    if seeded.is_file():
        rows = {}
        for line in seeded.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r.get("scope_digest")
        upstream = rows.get("artifact--brandonmusic.glm-5.3-flash-tr3-4bpw")
        if upstream:
            check("scope digest equals the registry's record for the same weights",
                  digest == upstream,
                  "ours   %s\n          theirs %s" % (digest, upstream))
        else:
            skip("scope digest equals the registry's record", "no upstream row seeded")
    else:
        skip("scope digest equals the registry's record", "registry data not present")
    check("the routed experts are the only quantized class",
          sorted(a["tensor_class"] for a in scope["assignments"]
                 if a["treatment"] == "quantized") == ["moe.experts", "mtp"])
    # The scope is submitted verbatim into an artifact record, and
    # artifact.schema.json's `scope` is additionalProperties:false. A stray key
    # here is not extra documentation -- it is a REJECTED submission at the seal
    # stage, after both cold runs are paid for. Check it against the schema
    # itself, not against a remembered list.
    schema_path = (TOOLS.parent.parent / "registry" / "schema" / "artifact.schema.json")
    if schema_path.is_file():
        allowed = set(json.loads(schema_path.read_text())["properties"]["scope"]["properties"])
        required = set(json.loads(schema_path.read_text())["properties"]["scope"]["required"])
        check("scope carries ONLY keys artifact.schema.json allows",
              set(scope) <= allowed and required <= set(scope),
              "extra=%s missing=%s" % (sorted(set(scope) - allowed),
                                       sorted(required - set(scope))))
        item_allowed = set(json.loads(schema_path.read_text())["properties"]["scope"]
                           ["properties"]["assignments"]["items"]["properties"])
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
        report = t3.scope_report(surface)
        check("scope_report keeps provenance OUTSIDE the scope object",
              report["scope"] == scope and "source" in report
              and "schema" in report and "source" not in report["scope"])
    else:
        skip("scope validates against artifact.schema.json", "schema not present")
    check("scope digest is stable across loads",
          t3.scope_digest(t3.load_tr3_surface(good, repo="fixture/tr3", revision=REV))
          == digest)

    # [7] the materializer, on a TR3 tree ------------------------------------
    # The non-routed tensors cannot serve transformers from the artifact's own
    # shards: they are interleaved with 148,608 routed payload objects, and
    # transformers derives its checkpoint key set from the shard FILES. So the
    # SAME materializer exl3hf uses re-shards them -- and for a TR3 release it
    # must decode NOTHING. This rung proves that: the emitted set is the
    # official 1,618 names and every tensor is bitwise what the artifact holds.
    out = work / "materialized"
    official_index = work / "official-index.json"
    official_index.write_text(json.dumps({"weight_map": dict(
        {n: "x.safetensors" for n in t3.official_nonrouted_names()},
        **{"model.language_model.layers.3.mlp.experts.0.gate_proj.weight": "y.safetensors"})}))
    receipt = xs.materialize_nonrouted(
        good, out, device="cpu", source_repo="fixture/tr3", source_revision=REV,
        official_index=official_index)
    check("materializer runs on a TR3 tree",
          receipt.get("written_tensor_count") == 1618,
          "written_tensor_count=%r, official_index_check=%r"
          % (receipt.get("written_tensor_count"),
             (receipt.get("official_index_check") or {}).get("checked")))
    check("the materializer's official-index gate ran and passed",
          (receipt.get("official_index_check") or {}).get("checked") is True)
    mat_index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    produced = {n for n in mat_index if t3._ROUTED.search(n) is None}
    check("materialized name set == the official non-routed set",
          produced == set(t3.official_nonrouted_names()),
          "%d produced" % len(produced))
    from safetensors import safe_open as _open
    src_handle = _open(str(good / NONROUTED_SHARD), framework="pt", device="cpu")
    same, checked = True, 0
    for name in sorted(produced)[:400]:
        want = src_handle.get_tensor(name)
        shard = mat_index[name]
        with _open(str(out / shard), framework="pt", device="cpu") as h:
            got = h.get_tensor(name)
        checked += 1
        if got.dtype != want.dtype or not torch.equal(got, want):
            same = False
            check("materialized tensors are bitwise the artifact's", False,
                  "%s: %s vs %s" % (name, got.dtype, want.dtype))
            break
    check("materialized tensors are bitwise the artifact's (no decode)", same,
          "%d checked, dtypes preserved (bf16 stays bf16, fp32 stays fp32)" % checked)
    matcfg = json.loads((out / "config.json").read_text())
    check("the materialized config drops quantization_config",
          "quantization_config" not in matcfg)

    # [8] the routed census ---------------------------------------------------
    cens = t3.routed_census(surface)
    check("executed routed surface closes",
          cens["executed_modules"] == 42 * 288 * 3, str(cens["executed_modules"]))
    check("MTP is present and named as not executed",
          cens["mtp_layer_present"] and "NEVER" in cens["mtp_note"])
    check("identity binds repo, revision, seal and scope",
          len(surface.checkpoint_identity_sha256()) == 64
          and surface.checkpoint_identity_sha256()
          != t3.load_tr3_surface(good, repo="other/repo",
                                 revision=REV).checkpoint_identity_sha256())
finally:
    shutil.rmtree(work, ignore_errors=True)

passed = sum(1 for _, ok, _ in RESULTS if ok)
print("\nselftest_tr3_offline: %d/%d checks passed" % (passed, len(RESULTS)))
raise SystemExit(0 if passed == len(RESULTS) else 1)
