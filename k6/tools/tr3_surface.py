#!/usr/bin/env python3
"""TR3-published checkpoint surface adapter ("tr3") for the streaming scorer.

Scores a SEALED, TR3-published EXL3/MCG release -- brandonmusic's
``GLM-5.3-Flash-tr3-*`` layout and its byte-identical mirrors -- on OUR sealed
25-window panel with the SAME single-device EP8-emulated streaming capture used
for K6/K8/the BF16 floor, so the number ranks directly against the streaming
lane's own rows.

Why this is a front-end and not a codec
---------------------------------------
A TR3-published repo stores its routed payloads as ``M.{trellis,suh,svh,mcg}``
inside canonical HF shards -- the SAME object layout ``exl3hf_surface`` already
reads, with the SAME campaign MCG codebook the K6/K8 rows were measured
through.  Nothing about the decode is new.  What was missing was a reader
FRONT-END: something that recognises the surface, verifies the seal the
publisher actually shipped, and tells the streamer that the non-routed function
needs no materialization.  So this module composes ``exl3hf_surface`` for every
byte of decode math and owns only identity, seal and scope.

The three ways it differs from the stock-exllamav3 (``exl3hf``) surface
----------------------------------------------------------------------
1. SCOPE.  ``quantization_config.scope == "glm53_routed_experts_only"``: only
   ``*.mlp.experts.<n>.{gate,up,down}_proj`` are quantized.  All 1,618
   non-routed tensors -- embeddings, attention, dense MLPs, norms, the router,
   the vision tower AND ``lm_head`` -- ship as the OFFICIAL source tensors
   (``non_routed_dtype_policy == "official_source_native"``).  There is
   therefore nothing to materialize: ``--bf16`` is REFUSED and the streaming
   engine builds its non-routed view directly over the artifact snapshot, the
   way ``--source nvfp4`` does.
2. NAMES.  The storage ABI's ``module_key_rule`` is
   ``official_weight_name_without_.weight``, so every module carries its
   OFFICIAL name.  None of the exl3hf fusion remaps (KDA ``qkv_proj``, fused
   ``conv1d``, split visual ``q/k/v_proj``) apply, and the non-routed name set
   is asserted to be EXACTLY the official BF16 release's 1,618 non-routed
   names.
3. SEAL.  Unlike the stock-exllamav3 and Dione releases -- which ship no
   upstream receipts and are scored under an ``unsealed_source`` disclosure --
   a TR3 release publishes ``exl3-mcg-storage-abi.json`` and
   ``materialization-receipt.json``, and every claim in them is REPRODUCIBLE
   from the published bytes.  ``verify_seal`` recomputes all of it before a
   single payload is read:

     * the materialization receipt's own ``receipt_sha256`` self-seal;
     * ``config_sha256`` / ``index_sha256`` against the local files;
     * ``output_tensor_names_sha256`` = sha256(canonical_json(sorted(names)))
       over the index's 150,226 tensor names;
     * ``plan_sha256`` agreement between the ABI and the receipt;
     * the count algebra 4 x routed_choice_count == packed_tensor_count and
       packed + native == output == len(weight_map);
     * ``nonrouted_native_exact``;
     * the non-routed name set against the official release's own.

   Every one of those is checkable for a few hundred kilobytes, i.e. BEFORE
   renting anything.  ``tr3_surface.py verify`` is exactly that pre-flight.

DISCLOSED, verbatim from the artifact's own ABI: ``serving_reader_qualified``
is false and ``qualified_tp_sizes`` is empty, because "ExLlamaV3 v0.0.43 has no
audited GLM-5.3 TP model load/inference receipt".  That is a statement about
SERVING with a TP kernel.  This adapter neither serves nor uses a TP kernel: it
decodes offline, on one device, for a logit measurement, and the storage
checkpoint the ABI does qualify (``storage_checkpoint_verified``) is the only
thing it consumes.  The receipt records both facts rather than picking one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exl3hf_surface as xs3  # noqa: E402 - sibling tool module (the decode ABI)

TR3_SURFACE_SCHEMA = "malaiwah.glm53-tr3-surface.v1"
TR3_IDENTITY_SCHEMA = "malaiwah.glm53-tr3-student-identity.v1"
TR3_READER_IDENTITY_SCHEMA = "malaiwah.glm53-tr3-offline-reader-identity.v1"
TR3_SEAL_SCHEMA = "malaiwah.glm53-tr3-seal-verification.v1"
TR3_SCOPE_SCHEMA = "malaiwah.glm53-tr3-published-scope.v1"
TR3_STUDENT_LABEL = "tr3-exl3-mcg-4bpw"

ABI_FILE = "exl3-mcg-storage-abi.json"
MATERIALIZATION_FILE = "materialization-receipt.json"
ABI_SCHEMA = "quant-pipeline.glm53-exl3-mcg-storage-abi.v1"
EXPECTED_SCOPE = "glm53_routed_experts_only"
EXPECTED_NONROUTED_POLICY = "official_source_native"

MAIN_ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYER = 45
NUM_EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

_EVIDENCE = HERE / "nvfp4-evidence"
OFFICIAL_NONROUTED_NAMES = _EVIDENCE / "official-nonrouted-names.json"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ROUTED = re.compile(r"\.mlp\.experts\.(\d+)\.")

# The seal says the source is BYTE-EXACT official, so scoring it needs no
# unsealed_source caveat -- but the reader still says what it verified, because
# "sealed" with nothing recomputed is a word, not evidence.
SEAL_DISCLOSURE = (
    "sealed-source scoring: the release publishes exl3-mcg-storage-abi.json and "
    "materialization-receipt.json, and this adapter RECOMPUTED every claim in them "
    "from the published bytes before decoding (receipt self-seal, config/index "
    "digests, output_tensor_names_sha256 over all 150,226 names, plan_sha256 "
    "agreement, the packed/native/output count algebra, nonrouted_native_exact, and "
    "the non-routed name set against the official release). The ABI's own "
    "serving_reader_qualified=false concerns TP SERVING, which this offline "
    "single-device decode does not do."
)


def _fail(message: str) -> ValueError:
    return ValueError(f"tr3_surface: {message}")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False)
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise _fail(f"{label} is absent: {path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _fail(f"{label} is not valid JSON ({path}): {exc}") from None


def official_nonrouted_names() -> Tuple[str, ...]:
    doc = _read_json(OFFICIAL_NONROUTED_NAMES, "official non-routed name evidence")
    names = doc.get("names")
    if not isinstance(names, list) or len(names) != int(doc.get("count", -1)):
        raise _fail("official-nonrouted-names.json is malformed (names/count disagree)")
    return tuple(names)


# --------------------------------------------------------------------------
# seal verification -- every check is offline and costs kilobytes
# --------------------------------------------------------------------------
def verify_seal(root: Path, weight_map: Mapping[str, str], *,
                config_path: Path, index_path: Path) -> Dict[str, Any]:
    """Recompute every claim the release seals, and record what was checked.

    Returns the verification block a receipt embeds.  Raises on the first claim
    that does not reproduce: a seal that does not reproduce is worse than no
    seal, because it invites the reader to trust it.
    """
    root = Path(root)
    abi = _read_json(root / ABI_FILE, ABI_FILE)
    mat = _read_json(root / MATERIALIZATION_FILE, MATERIALIZATION_FILE)
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise _fail(f"published seal does not reproduce: {name} -- {detail}")

    check("abi_schema", abi.get("schema") == ABI_SCHEMA,
          "schema=%r (want %r)" % (abi.get("schema"), ABI_SCHEMA))

    # 1. the materialization receipt's own self-seal
    body = {k: v for k, v in mat.items() if k != "receipt_sha256"}
    recomputed = _sha256_bytes(_canonical_json(body))
    check("materialization_receipt_self_seal", recomputed == mat.get("receipt_sha256"),
          "recomputed %s vs declared %s" % (recomputed[:16], str(mat.get("receipt_sha256"))[:16]))

    # 2. the receipt binds THESE config/index bytes
    config_digest = _sha256_file(config_path)
    index_digest = _sha256_file(index_path)
    check("config_sha256", config_digest == mat.get("config_sha256"),
          "local %s vs receipt %s" % (config_digest[:16], str(mat.get("config_sha256"))[:16]))
    check("index_sha256", index_digest == mat.get("index_sha256"),
          "local %s vs receipt %s" % (index_digest[:16], str(mat.get("index_sha256"))[:16]))

    # 3. the ABI's digest over the whole emitted name set
    names_digest = _sha256_bytes(_canonical_json(sorted(weight_map)))
    check("output_tensor_names_sha256",
          names_digest == abi.get("output_tensor_names_sha256"),
          "recomputed %s over %d names vs ABI %s"
          % (names_digest[:16], len(weight_map),
             str(abi.get("output_tensor_names_sha256"))[:16]))

    # 4. the ABI and the receipt describe the same materialization plan
    check("plan_sha256_agreement", abi.get("plan_sha256") == mat.get("plan_sha256"),
          "abi %s vs receipt %s" % (str(abi.get("plan_sha256"))[:16],
                                    str(mat.get("plan_sha256"))[:16]))

    # 5. the count algebra
    routed_choices = int(mat.get("routed_choice_count", -1))
    packed = int(mat.get("packed_tensor_count", -1))
    native = int(mat.get("native_tensor_count", -1))
    output = int(mat.get("output_tensor_count", -1))
    check("payload_objects_per_choice", packed == 4 * routed_choices,
          "packed_tensor_count %d vs 4 x routed_choice_count %d" % (packed, routed_choices))
    check("tensor_count_algebra",
          packed + native == output == len(weight_map),
          "packed %d + native %d = %d, output %d, index %d"
          % (packed, native, packed + native, output, len(weight_map)))

    # 6. the non-routed function is claimed byte-exact official
    check("nonrouted_native_exact", mat.get("nonrouted_native_exact") is True,
          "receipt declares %r" % (mat.get("nonrouted_native_exact"),))

    # 7. ... and the name set proves the claim is about the RIGHT tensors
    want = set(official_nonrouted_names())
    have = {n for n in weight_map if _ROUTED.search(n) is None}
    missing = sorted(want - have)
    extra = sorted(have - want)
    check("nonrouted_name_set_equals_official", not missing and not extra,
          "missing %d (first %s), extra %d (first %s)"
          % (len(missing), missing[:3], len(extra), extra[:3]))
    check("native_tensor_count_matches_names", native == len(have),
          "receipt native_tensor_count %d vs %d non-routed names" % (native, len(have)))

    # 8. codebook constant: the ABI's multiplier is the campaign MCG constant
    #    the frozen LUT is built from.  Every module's own marker tensor is
    #    checked at decode time by exl3hf_surface.verify_marker.
    check("mcg_multiplier", str(abi.get("mcg_multiplier_hex", "")).lower()
          == hex(xs3.MCG_MULT), "abi %r vs campaign constant %s"
          % (abi.get("mcg_multiplier_hex"), hex(xs3.MCG_MULT)))

    return {
        "schema": TR3_SEAL_SCHEMA,
        "verified": True,
        "checks": checks,
        "abi": {
            "schema": abi.get("schema"),
            "bits": abi.get("bits"),
            "codec_family": abi.get("codec_family"),
            "exllamav3_git_commit": (abi.get("exllamav3") or {}).get("git_commit"),
            "exllamav3_version": (abi.get("exllamav3") or {}).get("version"),
            "module_key_rule": (abi.get("exllamav3") or {}).get("module_key_rule"),
            "written_suffixes": (abi.get("exllamav3") or {}).get("written_suffixes"),
            "mcg_multiplier_hex": abi.get("mcg_multiplier_hex"),
            "packed_reader_abi_sha256": abi.get("packed_reader_abi_sha256"),
            "plan_sha256": abi.get("plan_sha256"),
            "output_tensor_names_sha256": abi.get("output_tensor_names_sha256"),
            "receipt_sha256": abi.get("receipt_sha256"),
            "storage_checkpoint_verified": abi.get("storage_checkpoint_verified"),
            # disclosed verbatim; it is a SERVING qualification, not a storage one
            "serving_reader_qualified": abi.get("serving_reader_qualified"),
            "qualified_tp_sizes": abi.get("qualified_tp_sizes"),
            "serving_qualification_reason": abi.get("reason"),
        },
        "materialization": {
            "schema": mat.get("schema"),
            "receipt_sha256": mat.get("receipt_sha256"),
            "plan_sha256": mat.get("plan_sha256"),
            "config_sha256": mat.get("config_sha256"),
            "index_sha256": mat.get("index_sha256"),
            "quantization_config_sha256": mat.get("quantization_config_sha256"),
            "output_tensor_count": output,
            "packed_tensor_count": packed,
            "native_tensor_count": native,
            "routed_choice_count": routed_choices,
            "output_logical_bytes": mat.get("output_logical_bytes"),
            "nonrouted_native_exact": mat.get("nonrouted_native_exact"),
            "complete": mat.get("complete"),
            "main_and_mtp_complete": mat.get("main_and_mtp_complete"),
            "shard_count": len(mat.get("shard_sha256") or {}),
        },
        "official_nonrouted_names_sha256": _sha256_file(OFFICIAL_NONROUTED_NAMES),
    }


def verify_shard_digests(root: Path, *, mode: str = "crosscheck") -> Dict[str, Any]:
    """Bind the shard BYTES to the seal.

    ``crosscheck`` (default) is free: the release also publishes ``SHA256SUMS``,
    and the fetch stage runs ``sha256sum -c`` over it, so the only thing left to
    prove is that the two published lists AGREE -- a name-level pass over two
    small files.  ``full`` re-hashes every shard locally (~176 GB) and is what
    ``verify --full`` does when no SHA256SUMS is published.
    """
    root = Path(root)
    mat = _read_json(root / MATERIALIZATION_FILE, MATERIALIZATION_FILE)
    declared: Dict[str, str] = dict(mat.get("shard_sha256") or {})
    if not declared:
        raise _fail("materialization-receipt.json carries no shard_sha256 map")
    sums_path = root / "SHA256SUMS"
    result: Dict[str, Any] = {"mode": mode, "shards": len(declared)}
    if mode == "crosscheck":
        if not sums_path.is_file():
            raise _fail(
                "no SHA256SUMS published, so `crosscheck` has nothing to compare "
                "the receipt's shard map against; use mode='full'")
        published: Dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and len(parts[0]) == 64:
                published[parts[1].strip().lstrip("*")] = parts[0]
        disagree = sorted(
            name for name, digest in declared.items()
            if published.get(name) != digest
        )
        if disagree:
            raise _fail(
                "SHA256SUMS and materialization-receipt.json disagree on %d shard(s): %s"
                % (len(disagree), disagree[:3]))
        result.update({
            "verification": "receipt-shard-map == published SHA256SUMS",
            "sha256sums_sha256": _sha256_file(sums_path),
            "sha256sums_entries": len(published),
            "agreed": len(declared),
        })
        return result
    if mode != "full":
        raise _fail(f"unknown shard verification mode {mode!r} (crosscheck|full)")
    bad = []
    for name, digest in sorted(declared.items()):
        path = root / name
        if not path.is_file():
            raise _fail(f"shard absent: {path}")
        if _sha256_file(path) != digest:
            bad.append(name)
    if bad:
        raise _fail("%d shard(s) do not match the receipt: %s" % (len(bad), bad[:3]))
    result.update({"verification": "every shard re-hashed locally", "agreed": len(declared)})
    return result


# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------
@dataclass
class Tr3Surface:
    root: Path
    repo: Optional[str]
    revision: str
    # the DECODE-side surface: exl3hf_surface owns every byte of the codec path
    exl3: Any
    declared_bits: float
    declared_head_bits: Optional[int]
    quantizer_version: str
    exllamav3_pin: Optional[str]
    scope_policy: str
    nonrouted_policy: str
    config_sha256: str
    index_sha256: str
    seal: Dict[str, Any] = field(compare=False, default_factory=dict)
    shard_verification: Dict[str, Any] = field(compare=False, default_factory=dict)
    routed_module_count: int = 0
    nonrouted_tensor_count: int = 0
    routed_layers_present: Tuple[int, ...] = ()
    dtype_census: Dict[str, int] = field(compare=False, default_factory=dict)

    @property
    def codebook(self) -> str:
        return self.exl3.codebook

    def scope_census_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(published_scope(self)))

    def checkpoint_identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json({
            "schema": TR3_IDENTITY_SCHEMA,
            "tr3_repo": self.repo,
            "tr3_revision": self.revision,
            "codebook": self.codebook.upper(),
            "codec_family": "exl3-mcg",
            "declared_bits": self.declared_bits,
            "declared_head_bits": self.declared_head_bits,
            "quantizer_version": self.quantizer_version,
            "exllamav3_pin": self.exllamav3_pin,
            "scope_policy": self.scope_policy,
            "nonrouted_policy": self.nonrouted_policy,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "routed_module_count": self.routed_module_count,
            "nonrouted_tensor_count": self.nonrouted_tensor_count,
            "scope_census_sha256": self.scope_census_sha256(),
            "seal_verification": self.seal.get("checks"),
            "shard_verification": self.shard_verification.get("verification"),
            "seal_disclosure": SEAL_DISCLOSURE,
        }))


def _dtype_census(root: Path, weight_map: Mapping[str, str]) -> Dict[str, int]:
    """Per-dtype counts of the NON-ROUTED tensors, read from shard HEADERS only.

    Header-only: safetensors puts a JSON header at the front of every file, so
    this answers "what dtype did the publisher actually store" for a few
    megabytes of reads instead of 176 GB.  Recording it beats assuming BF16:
    A_log / dt_bias / e_score_correction_bias are fp32 in the official release
    and a scope report that called them bf16 would be wrong.
    """
    import struct

    census: Dict[str, int] = {}
    nonrouted = [n for n in weight_map if _ROUTED.search(n) is None]
    by_shard: Dict[str, List[str]] = {}
    for name in nonrouted:
        by_shard.setdefault(weight_map[name], []).append(name)
    for shard, names in by_shard.items():
        path = Path(root) / shard
        if not path.is_file():
            raise _fail(f"shard absent: {path}")
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length).decode("utf-8"))
        for name in names:
            entry = header.get(name)
            if entry is None:
                raise _fail(f"{shard} does not contain {name} (index disagrees with shard)")
            census[entry["dtype"]] = census.get(entry["dtype"], 0) + 1
    return census


def load_tr3_surface(root, *, repo: Optional[str], revision: str,
                     verify_shards: str = "crosscheck") -> Tr3Surface:
    """Open a TR3-published snapshot, fail-closed on the seal and the scope."""
    root = Path(root).resolve()
    if not _REVISION.match(revision or ""):
        raise _fail("--tr3-revision must be the immutable 40-hex commit")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = _read_json(config_path, "config.json")
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "exl3":
        raise _fail("config.quantization_config.quant_method is not 'exl3'")
    if str(quant.get("codebook")) != "mcg":
        raise _fail(
            "a TR3-published release is EXL3/MCG; this one declares codebook %r. "
            "A mul1 release is the exl3hf surface's job." % quant.get("codebook"))
    if quant.get("scope") != EXPECTED_SCOPE:
        raise _fail(
            "this adapter reads the routed-experts-only TR3 scope; the release "
            "declares scope=%r. A different scope means the non-routed function "
            "is NOT the official one and must not be read as if it were."
            % quant.get("scope"))
    if quant.get("non_routed_dtype_policy") != EXPECTED_NONROUTED_POLICY:
        raise _fail(
            "release declares non_routed_dtype_policy=%r, not %r: the streaming "
            "engine would build its non-routed view over tensors that are not the "
            "official ones" % (quant.get("non_routed_dtype_policy"),
                               EXPECTED_NONROUTED_POLICY))
    head_bits = quant.get("head_bits")
    if head_bits not in (16, None):
        raise _fail(
            "TR3 keeps lm_head native (head_bits 16); this release declares "
            "head_bits=%r, which changes the measured function and must be read "
            "by a surface that says so" % (head_bits,))

    # the decode-side surface, verbatim
    exl3 = xs3.load_surface(root)
    weight_map = exl3.weight_map

    seal = verify_seal(root, weight_map, config_path=config_path, index_path=index_path)
    shard_verification = (
        {"mode": "skip",
         "verification": "shard bytes NOT bound to the seal by this adapter"}
        if verify_shards == "skip"
        else verify_shard_digests(root, mode=verify_shards)
    )

    routed_layers = sorted({int(m.group(1)) for m in
                            (re.search(r"\.layers\.(\d+)\.mlp\.experts\.", n)
                             for n in weight_map) if m})
    routed_modules = len(exl3.quantized_modules)
    nonrouted = sum(1 for n in weight_map if _ROUTED.search(n) is None)
    expected_modules = len(routed_layers) * NUM_EXPERTS * len(PROJECTIONS)
    if routed_modules != expected_modules:
        raise _fail(
            "routed module census disagrees with the layer/expert geometry: "
            "%d quantized modules vs %d layers x %d experts x %d projections = %d"
            % (routed_modules, len(routed_layers), NUM_EXPERTS,
               len(PROJECTIONS), expected_modules))
    missing_main = [layer for layer in MAIN_ROUTED_LAYERS if layer not in routed_layers]
    if missing_main:
        raise _fail(
            "the executed routed layers 3..44 are not all present: missing %s"
            % missing_main[:5])

    return Tr3Surface(
        root=root,
        repo=repo,
        revision=revision,
        exl3=exl3,
        declared_bits=float(quant.get("bits", 0.0)),
        declared_head_bits=head_bits,
        quantizer_version=str(quant.get("version", "unknown")),
        exllamav3_pin=(seal["abi"] or {}).get("exllamav3_git_commit"),
        scope_policy=str(quant.get("scope")),
        nonrouted_policy=str(quant.get("non_routed_dtype_policy")),
        config_sha256=seal["materialization"]["config_sha256"],
        index_sha256=seal["materialization"]["index_sha256"],
        seal=seal,
        shard_verification=shard_verification,
        routed_module_count=routed_modules,
        nonrouted_tensor_count=nonrouted,
        routed_layers_present=tuple(routed_layers),
        dtype_census=_dtype_census(root, weight_map),
    )


def routed_census(surface: Tr3Surface) -> Dict[str, Any]:
    """The executed routed surface, per layer, from names alone."""
    present = 0
    absent: List[str] = []
    for layer in MAIN_ROUTED_LAYERS:
        for expert in range(NUM_EXPERTS):
            for projection in PROJECTIONS:
                module = xs3.routed_module_name(layer, expert, projection)
                if module in surface.exl3.quantized_modules:
                    present += 1
                elif len(absent) < 5:
                    absent.append(module)
    if absent:
        raise _fail("executed routed modules are missing from the index: %s" % absent)
    return {
        "executed_layers": list(MAIN_ROUTED_LAYERS),
        "experts_per_layer": NUM_EXPERTS,
        "projections": list(PROJECTIONS),
        "executed_modules": present,
        "modules_in_release": surface.routed_module_count,
        "mtp_layer_present": MTP_LAYER in surface.routed_layers_present,
        "mtp_note": (
            "layer %d's routed experts are present in the release and are NEVER "
            "decoded or executed by standard-logits scoring" % MTP_LAYER),
    }


# --------------------------------------------------------------------------
# scope: the release published it, so the registry does not have to guess
# --------------------------------------------------------------------------
def published_scope(surface: Tr3Surface) -> Dict[str, Any]:
    """The per-tensor-class recipe, READ from the artifact's own declarations.

    Every entry cites where it came from.  ``unknown`` is not used: this release
    states its scope (`glm53_routed_experts_only`), its non-routed policy
    (`official_source_native`), its head bits (16) and its bit rate (4), and the
    stored dtypes are readable from the shard headers.
    """
    cite = ("read from the release's OWN config.json quantization_config "
            "(scope=%s, non_routed_dtype_policy=%s, bits=%s, head_bits=%s, "
            "version=%s) plus exl3-mcg-storage-abi.json; the non-routed set is "
            "byte-exact official (nonrouted_native_exact, name set verified "
            "against the official BF16 release's 1,618 non-routed names)"
            % (surface.scope_policy, surface.nonrouted_policy,
               surface.declared_bits, surface.declared_head_bits,
               surface.quantizer_version))
    bits = surface.declared_bits
    native = lambda cls, fmt, bpw, note: {  # noqa: E731 - a table, not a function
        "tensor_class": cls, "treatment": "native", "format": fmt,
        "bits_per_weight": bpw, "layer_range": "all", "note": note + " " + cite}
    assignments = [
        native("embed_tokens", "bf16", 16, "stored BF16 in the artifact's own shards."),
        native("attn.qkv", "bf16", 16,
               "NOT quantized: routed-experts-only scope. KDA layers ship the "
               "official split q/k/v_proj; MLA layers the official q_a/q_b/"
               "kv_a_with_mqa/wq_b. No fusion remap applies."),
        native("attn.o", "bf16", 16, "NOT quantized: routed-experts-only scope."),
        native("attn.other", "mixed", None,
               "b_proj / f_a_proj / f_b_proj / g_a_proj / g_b_proj / conv1d / "
               "indexer and the attention norms ship as the official tensors; "
               "A_log, dt_bias and e_score_correction_bias are fp32 there and "
               "fp32 here (dtype census: %s)."
               % json.dumps(surface.dtype_census, sort_keys=True)),
        native("mlp.gate", "bf16", 16, "dense layers 0-2 only; NOT quantized."),
        native("mlp.up", "bf16", 16, "dense layers 0-2 only; NOT quantized."),
        native("mlp.down", "bf16", 16, "dense layers 0-2 only; NOT quantized."),
        native("moe.router", "fp32", 32,
               "the routing gate and e_score_correction_bias are native."),
        native("moe.shared_expert", "bf16", 16,
               "the shared expert is NOT routed and is NOT quantized."),
        native("norm", "bf16", 16, "all norms native."),
        native("lm_head", "bf16", 16,
               "head_bits 16: TR3 keeps the head native BF16, unlike stock "
               "exllamav3 which quantizes it."),
        native("other", "bf16", 16,
               "the vision tower ships the official fused attn.qkv and is never "
               "executed by text-only scoring."),
        {"tensor_class": "moe.experts", "treatment": "quantized",
         "format": "exl3-mcg", "bits_per_weight": bits, "layer_range": "3-44",
         "note": "%d modules = 42 layers x %d experts x 3 projections, all K%d. %s"
                 % (42 * NUM_EXPERTS * 3, NUM_EXPERTS, int(bits), cite)},
        {"tensor_class": "mtp", "treatment": "quantized",
         "format": "exl3-mcg", "bits_per_weight": bits, "layer_range": "45",
         "note": "layer 45's routed experts are quantized and present in the "
                 "release, and are NOT executed by standard-logits scoring: "
                 "present in the artifact, outside the measured function. %s" % cite},
    ]
    return {
        "schema": TR3_SCOPE_SCHEMA,
        "policy": "uniform",
        "head_policy": "native",
        "kv_cache_dtype": "not_applicable",
        "mtp_included": True,
        "activation_quantization": None,
        "assignments": assignments,
        "source": {
            "config_sha256": surface.config_sha256,
            "index_sha256": surface.index_sha256,
            "abi_plan_sha256": (surface.seal.get("abi") or {}).get("plan_sha256"),
            "materialization_receipt_sha256":
                (surface.seal.get("materialization") or {}).get("receipt_sha256"),
            "repo": surface.repo,
            "revision": surface.revision,
        },
    }


def scope_digest(surface: Tr3Surface) -> str:
    """The registry's one-line scope digest for this artifact."""
    scope = published_scope(surface)
    parts = []
    for entry in sorted(scope["assignments"], key=lambda e: e["tensor_class"]):
        bpw = entry["bits_per_weight"]
        fmt = entry["format"]
        parts.append("%s=%s:%s%s" % (
            entry["tensor_class"], entry["treatment"], fmt,
            "" if bpw is None else "@%g" % float(bpw)))
    parts.append("head=%s" % scope["head_policy"])
    parts.append("kv=%s" % scope["kv_cache_dtype"])
    return "|".join(parts)


# --------------------------------------------------------------------------
# identity + provenance blocks the receipts embed
# --------------------------------------------------------------------------
def tr3_reader_identity(runner_path, surface: Tr3Surface) -> Dict[str, Any]:
    body = {
        "schema": TR3_READER_IDENTITY_SCHEMA,
        "mode": "tr3_published_sealed_shard_offline_decode_for_logit_measurement",
        "serving_kernel": False,
        "final_tp2_kernel": False,
        "codebook": surface.codebook.upper(),
        "codec_family": "exl3-mcg",
        "bits": surface.declared_bits,
        "decode_executed": True,
        "decode_contract": (
            "exl3hf_surface.decode_module -> decode_payload_hf: the campaign's own "
            "anybits trellis unpack and fp32 Hadamard path with the FROZEN MCG LUT "
            "(quant_pipeline.evaluation.glm53_packed_k4_reader.mcg_lut), i.e. the "
            "identical codec the K6/K8 streaming rows were measured through"),
        "nonrouted_source": "artifact_own_official_native_tensors_no_materialization",
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "decode_module_sha256": _sha256_file(HERE / "exl3hf_surface.py"),
        "runner_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
    }
    body["runtime_reader_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


def surface_summary(surface: Tr3Surface) -> Dict[str, Any]:
    return {
        "schema": TR3_SURFACE_SCHEMA,
        "tr3_repo": surface.repo,
        "tr3_revision": surface.revision,
        "codebook": surface.codebook,
        "codec_family": "exl3-mcg",
        "declared_bits": surface.declared_bits,
        "declared_head_bits": surface.declared_head_bits,
        "quantizer": {"tool": "exllamav3", "version": surface.quantizer_version,
                      "git_commit": surface.exllamav3_pin},
        "scope_policy": surface.scope_policy,
        "nonrouted_policy": surface.nonrouted_policy,
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "routed_module_count": surface.routed_module_count,
        "nonrouted_tensor_count": surface.nonrouted_tensor_count,
        "nonrouted_dtype_census": surface.dtype_census,
        "routed_census": routed_census(surface),
        "scope_census_sha256": surface.scope_census_sha256(),
        "seal_verification": surface.seal,
        "shard_verification": surface.shard_verification,
        "seal_disclosure": SEAL_DISCLOSURE,
    }


def expert_source(surface: Tr3Surface):
    """The (surface, reader) pair the streamer's exl3hf fill loop consumes.

    Deliberately the SAME pair `--source exl3hf` passes: the routed payload
    layout and the decode are identical, so reimplementing the fill loop would
    create a second code path that could drift from the one M1 proved.
    """
    return surface.exl3, xs3.Exl3HfShardReader(surface.exl3)


# --------------------------------------------------------------------------
# CLI: pre-flight and probes
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ver = sub.add_parser("verify", help="recompute the published seal (no GPU, no decode)")
    ver.add_argument("--root", type=Path, required=True)
    ver.add_argument("--repo")
    ver.add_argument("--revision", required=True)
    ver.add_argument("--shards", choices=("crosscheck", "full", "skip"),
                     default="crosscheck")
    ver.add_argument("--out", type=Path)

    sc = sub.add_parser("scope", help="emit the registry-shaped scope JSON")
    sc.add_argument("--root", type=Path, required=True)
    sc.add_argument("--repo")
    sc.add_argument("--revision", required=True)
    sc.add_argument("--shards", choices=("crosscheck", "full", "skip"), default="skip")
    sc.add_argument("--out", type=Path)

    pr = sub.add_parser("probe", help="decode one routed module and print stats")
    pr.add_argument("--root", type=Path, required=True)
    pr.add_argument("--repo")
    pr.add_argument("--revision", required=True)
    pr.add_argument("--layer", type=int, default=3)
    pr.add_argument("--expert", type=int, default=0)
    pr.add_argument("--projection", default="gate_proj", choices=PROJECTIONS)
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--shards", choices=("crosscheck", "full", "skip"), default="skip")

    args = parser.parse_args()
    surface = load_tr3_surface(args.root, repo=args.repo, revision=args.revision,
                               verify_shards=args.shards)

    if args.command == "verify":
        payload = surface_summary(surface)
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        checks = surface.seal["checks"]
        print("\nseal: %d/%d checks reproduced; shards: %s"
              % (sum(1 for c in checks if c["passed"]), len(checks),
                 surface.shard_verification.get("verification")), file=sys.stderr)
        return 0

    if args.command == "scope":
        payload = {"scope": published_scope(surface),
                   "scope_digest": scope_digest(surface)}
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    exl3, reader = expert_source(surface)
    decoded, census = xs3.load_decoded_module(
        exl3, reader, layer=args.layer, expert=args.expert,
        projection=args.projection, device=args.device)
    print(json.dumps({
        "module": census["module"], "bits": census["bits"],
        "shape": list(decoded.shape), "dtype": str(decoded.dtype),
        "abs_max": float(decoded.abs().max()),
        "rms": float((decoded.double() ** 2).mean().sqrt()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
