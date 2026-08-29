#!/usr/bin/env python3
"""MLX affine-quantized checkpoint surface adapter for the streaming K-scorer.

Scores community MLX conversions of GLM-5.3-Flash (orcarouter/GLM-5.3-Flash-MLX
and layout-compatible repos) on OUR sealed 25-window panel with the SAME
streaming teacher-forced capture used for K6/K8/native-BF16, so the number is
directly comparable to the sealed campaign and to the Dione rows.

Format (verified against orcarouter/GLM-5.3-Flash-MLX @ c80f6810, header-level,
all 62 shards, 2026-08-29):

  * SCOPE: unlike our EXL3/TR3 family and Dione, this format quantizes BEYOND
    the routed experts.  Measured from the real index (113,446 tensors folding
    onto exactly the 38,770 official BF16 tensor names):
      - routed + MTP experts       (43 layers x 288 x 3)   -> affine-quantized
      - dense MLPs (layers 0-2)    (gate/up/down)          -> affine-quantized
      - shared experts (43 layers) (gate/up/down)          -> affine-quantized
      - DSA attention (12 layers)  (q_a/q_b/kv_a_mqa/o)    -> affine-quantized
      - EVERYTHING else (embed, lm_head, vision, KDA projections, kv_b_proj,
        indexer, norms, gates)                             -> source-dtype
        passthrough, byte-identical shapes/dtypes to the official tree
    The receipt therefore carries a SCOPE POLICY block; ``--bf16`` is NOT an
    input of this source (the quant snapshot supplies every tensor), it is only
    an optional cross-check target.
  * A quantized module X is the triplet ``X.weight`` (U32 packed), ``X.scales``
    and ``X.biases`` (F16 or BF16, one entry per ``group_size`` input columns).
    Packing is a plain little-endian bitstream per output row: element ``e``
    occupies bits ``[e*b, (e+1)*b)`` of the row's little-endian byte stream.
    Dequant: ``W[r, c] = q[r, c] * scales[r, c // G] + biases[r, c // G]`` with
    UNSIGNED ``q`` in ``[0, 2^b - 1]``.
  * Mixed bit-widths are real: per-tensor ``bits`` is DERIVED from shapes
    against the official BF16 shape census (``bits = 32 * packed_cols / in``,
    ``G = in / scales_cols``) and cross-checked against config.json's
    ``quantization`` override map.  The derivation is authoritative: orcarouter
    stores layer-45 (MTP) experts at 5-bit down_proj / 6-bit shared experts
    while its config override map does not mention layer 45 at all (measured;
    disclosed, not refused, because layer 45 is never executed).

DECODE CONTRACT - proven, not assumed: the plain-torch dequant below, rounded
once to float16, is BITWISE equal to ``mlx.core.dequantize`` (mlx 0.32.2) on
real ranged-fetched orcarouter tensors at 4, 5 and 6 bits (see
``mlx-evidence/real-dequant-fixtures.json``), and equal to the synthetic
reference packer on every bit-width 2..8.  The surface keeps fp32 and lets the
streaming installer do the suite's single fp32->bf16 rounding; the one
deviation from an MLX runtime (which would compute in f16) is exactly that
bf16 rounding, and it is disclosed in the capture receipt.

DISCLOSED DEVIATION - unsealed-source scoring: community MLX checkpoints ship
no per-choice receipts, no reconstruction closures and no sealed reader ABI.
This adapter decodes WITHOUT seal verification: it records the immutable repo
revision, the config/index sha256, the official-shape-census binding and
(optionally) whole-shard sha256 against an HF-derived manifest.  Every receipt
this adapter touches carries ``seal_disclosure`` saying exactly that.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

MLX_FORMAT = "glm53-mlx-affine-u32-v1"
MLX_SURFACE_SCHEMA = "malaiwah.glm53-mlx-affine-surface.v1"
MLX_IDENTITY_SCHEMA = "malaiwah.glm53-mlx-affine-student-identity.v1"
MLX_READER_IDENTITY_SCHEMA = "malaiwah.glm53-mlx-affine-offline-reader-identity.v1"
MLX_SHARDS_VERIFIED_SCHEMA = "malaiwah.glm53-mlx-shards-verified.v1"
MLX_VIEW_RECEIPT_SCHEMA = "malaiwah.glm53-mlx-nonrouted-decoded-view.v1"
SCOPE_POLICY_SCHEMA = "malaiwah.glm53-quant-scope-policy.v1"
OFFICIAL_CENSUS_SCHEMA = "malaiwah.glm53-official-bf16-shape-census.v1"
SEAL_DISCLOSURE = (
    "unsealed-source scoring: the MLX checkpoint ships no upstream receipts, "
    "reconstruction closures or sealed reader ABI; the packed surface was decoded "
    "WITHOUT seal verification (the immutable repo revision, config/index sha256 and "
    "the official-BF16 shape-census binding are recorded instead; whole-shard sha256 "
    "optionally verified against an HF-derived mlx-manifest.json)"
)
DTYPE_DISCLOSURE = (
    "decode dtype: fp32 dequant (q * scale + bias accumulated in float32, exact for "
    "b<=8 and f16/bf16 scales) with the suite's single fp32->bf16 rounding at expert "
    "install; an MLX runtime would compute in float16 - the fp32 decode rounded to "
    "f16 was proven BITWISE equal to mlx.core.dequantize on real tensors, so the "
    "only deviation is the final bf16 rounding, which is this lane's own install "
    "algebra for every measured surface"
)

MAIN_ROUTED_LAYERS = tuple(range(3, 45))
MTP_LAYER = 45
NUM_EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PROJECTION_SHAPE = {
    "gate_proj": (2048, 4096),
    "up_proj": (2048, 4096),
    "down_proj": (4096, 2048),
}
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
_REVISION = re.compile(r"[0-9a-f]{40}")
EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
# markers of the mlx-vlm ("pipenetwork") dialect this adapter refuses by name
_DIALECT_MARKERS = (
    (".switch_mlp.", "fused switch_mlp expert tensors (mlx-vlm dialect)"),
    ("language_model.model.", "language_model.model.* name prefix (mlx-vlm dialect)"),
    ("vision_model.", "vision_model.* rename (mlx-vlm dialect)"),
)

_TOOLS = Path(__file__).resolve().parent
OFFICIAL_CENSUS_DEFAULT = _TOOLS / "mlx-evidence" / "bf16-shape-census.json.gz"


def _fail(message: str) -> ValueError:
    return ValueError(f"mlx_surface: {message}")


def _canonical_json(value: Any) -> bytes:
    """Byte-identical to quant_pipeline.core.artifacts.canonical_json."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise _fail(f"{label} missing: {path}")
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_gz(path: Path) -> Optional[Path]:
    """Return path, path.gz, whichever exists (None otherwise)."""
    if path.is_file():
        return path
    gz_path = path.with_name(path.name + ".gz")
    if gz_path.is_file():
        return gz_path
    return None


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """8-byte little-endian length + JSON header; no torch, no full read."""
    with path.open("rb") as fh:
        head8 = fh.read(8)
        if len(head8) != 8:
            raise _fail(f"not a safetensors file (short header): {path}")
        (n,) = struct.unpack("<Q", head8)
        if n > (1 << 31):
            raise _fail(f"implausible safetensors header length {n}: {path}")
        raw = fh.read(n)
    header = json.loads(raw)
    header["__header_len__"] = n
    return header


def official_name(layer: int, expert: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


# ---------------------------------------------------------------------------
# official BF16 shape census (the architecture yardstick the derivation
# is checked against)
# ---------------------------------------------------------------------------
def load_official_census(path: Path) -> Tuple[Dict[str, Tuple[str, Tuple[int, ...]]], Dict[str, Any]]:
    census = _read_json(path, "official BF16 shape census")
    if (
        census.get("schema") != OFFICIAL_CENSUS_SCHEMA
        or not isinstance(census.get("tensors"), Mapping)
        or _REVISION.fullmatch(str(census.get("source_revision", ""))) is None
    ):
        raise _fail(f"official shape census is not a sealed-form {OFFICIAL_CENSUS_SCHEMA}: {path}")
    tensors = {
        name: (str(entry[0]), tuple(int(v) for v in entry[1]))
        for name, entry in census["tensors"].items()
    }
    if len(tensors) != int(census.get("tensor_count", -1)):
        raise _fail("official shape census tensor_count differs from its own table")
    meta = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "source_repo": census["source_repo"],
        "source_revision": census["source_revision"],
        "tensor_count": len(tensors),
    }
    return tensors, meta


# ---------------------------------------------------------------------------
# name folding + per-tensor (bits, group_size) derivation
# ---------------------------------------------------------------------------
def _detect_foreign_dialect(names) -> None:
    for name in names:
        for marker, label in _DIALECT_MARKERS:
            if marker in name:
                raise _fail(
                    f"unsupported MLX dialect: index tensor {name!r} carries {label}. "
                    "This adapter supports the HF-named per-expert layout "
                    "(orcarouter dialect) only; fused mlx-vlm checkpoints are a "
                    "named exclusion, not a silent skip."
                )


def census_index(
    weight_map: Mapping[str, str],
    tensor_meta: Mapping[str, Tuple[str, Tuple[int, ...]]],
    official: Mapping[str, Tuple[str, Tuple[int, ...]]],
) -> Dict[str, Any]:
    """Fail-closed census: fold weight/scales/biases triplets onto official
    names, derive per-tensor (bits, group_size) from shapes against the
    official census, and refuse anything unmapped BY NAME.

    ``tensor_meta``: name -> (dtype, shape) for every index tensor (from local
    shard headers or the fetched shard-headers evidence).
    """
    _detect_foreign_dialect(weight_map)
    names = set(weight_map)
    missing_meta = [name for name in names if name not in tensor_meta]
    if missing_meta:
        raise _fail(
            f"{len(missing_meta)} index tensors have no shard-header metadata "
            f"(first: {missing_meta[0]}) - shards absent and no shard-headers evidence"
        )
    scales = {n[: -len(".scales")] for n in names if n.endswith(".scales")}
    biases = {n[: -len(".biases")] for n in names if n.endswith(".biases")}
    weights = {n[: -len(".weight")] for n in names if n.endswith(".weight")}
    if scales != biases:
        odd = sorted(scales.symmetric_difference(biases))
        raise _fail(f"scales/biases sets differ - not mlx affine: {odd[:4]}")
    orphan = sorted(scales - weights)
    if orphan:
        raise _fail(f"scales/biases without a packed weight: {orphan[:4]}")
    quantized = sorted(scales & weights)
    logical = {n for n in names if not (n.endswith(".scales") or n.endswith(".biases"))}

    official_names = set(official)
    if logical != official_names:
        extra = sorted(logical - official_names)
        absent = sorted(official_names - logical)
        raise _fail(
            "logical tensor set does not biject the official BF16 census "
            f"({len(extra)} extra, first {extra[:3]}; {len(absent)} absent, first {absent[:3]})"
        )

    rows: Dict[str, Dict[str, Any]] = {}
    passthrough_mismatch: List[str] = []
    for module in quantized:
        w_dtype, w_shape = tensor_meta[module + ".weight"]
        s_dtype, s_shape = tensor_meta[module + ".scales"]
        b_dtype, b_shape = tensor_meta[module + ".biases"]
        o_dtype, o_shape = official[module + ".weight"]
        if w_dtype != "U32":
            raise _fail(f"packed weight is {w_dtype}, not U32 (mlx affine): {module}")
        if s_dtype not in ("F16", "BF16") or b_dtype != s_dtype:
            raise _fail(f"scales/biases dtype pair {s_dtype}/{b_dtype} unsupported: {module}")
        if s_shape != b_shape:
            raise _fail(f"scales/biases shapes differ: {module}")
        if len(o_shape) != 2 or len(w_shape) != 2 or len(s_shape) != 2:
            raise _fail(f"quantized module is not a 2-D matrix: {module} {o_shape}")
        out_f, in_f = o_shape
        if w_shape[0] != out_f or s_shape[0] != out_f:
            raise _fail(
                f"row count differs from official [{out_f},{in_f}]: {module} "
                f"weight {w_shape} scales {s_shape}"
            )
        if in_f % s_shape[1]:
            raise _fail(f"group size is not integral (in={in_f}, scales cols={s_shape[1]}): {module}")
        group_size = in_f // s_shape[1]
        packed_bits = w_shape[1] * 32
        if packed_bits % in_f:
            raise _fail(
                f"bits underivable (packed cols {w_shape[1]} x32 not divisible by in={in_f}): {module}"
            )
        bits = packed_bits // in_f
        if bits not in SUPPORTED_BITS:
            raise _fail(f"derived bits={bits} outside supported {SUPPORTED_BITS}: {module}")
        rows[module] = {
            "bits": bits,
            "group_size": group_size,
            "out_features": out_f,
            "in_features": in_f,
            "scales_dtype": s_dtype,
        }
    for name in sorted(logical):
        module = name[: -len(".weight")] if name.endswith(".weight") else None
        if module in rows:
            continue
        got = tensor_meta[name]
        want = official[name]
        if (got[0], tuple(got[1])) != (want[0], tuple(want[1])):
            passthrough_mismatch.append(f"{name}: {got} != official {want}")
    if passthrough_mismatch:
        raise _fail(
            f"{len(passthrough_mismatch)} passthrough tensors differ from the official "
            f"dtype/shape (first: {passthrough_mismatch[0]})"
        )

    routed, routed_mtp, nonrouted_q = [], [], []
    for module in quantized:
        match = EXPERT_RE.search(module + ".")
        if match is not None:
            layer = int(_LAYER_RE.search(module + ".").group(1))
            (routed_mtp if layer == MTP_LAYER else routed).append(module)
        else:
            nonrouted_q.append(module)
    expected_routed = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in MAIN_ROUTED_LAYERS
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    }
    if set(routed) != expected_routed:
        absent = sorted(expected_routed - set(routed))
        stray = sorted(set(routed) - expected_routed)
        raise _fail(
            f"routed expert census does not close (absent {len(absent)}, first {absent[:2]}; "
            f"stray {len(stray)}, first {stray[:2]})"
        )
    histogram: Dict[str, int] = {}
    for module, row in rows.items():
        key = f"b{row['bits']}-gs{row['group_size']}"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "quantized": rows,
        "routed_modules": routed,
        "routed_mtp_modules": sorted(routed_mtp),
        "nonrouted_quantized_modules": sorted(nonrouted_q),
        "passthrough_tensors": sorted(logical - {m + ".weight" for m in rows}),
        "bits_histogram": dict(sorted(histogram.items())),
        "logical_tensor_count": len(logical),
        "stored_tensor_count": len(names),
    }


def translate_override_key(key: str) -> List[str]:
    """config.json quantization override key -> logical module path(s).

    Quantizer namespace: ``model.layers.N....`` with fused ``switch_mlp``;
    checkpoint namespace: ``model.language_model.layers.N....`` per expert.
    """
    if not key.startswith("model.layers."):
        return []
    translated = "model.language_model.layers." + key[len("model.layers."):]
    if ".switch_mlp." in translated:
        head, tail = translated.split(".switch_mlp.", 1)
        return [f"{head}.experts.{expert}.{tail}" for expert in range(NUM_EXPERTS)]
    return [translated]


def crosscheck_config_declaration(
    config: Mapping[str, Any], census: Mapping[str, Any]
) -> Dict[str, Any]:
    """Derived (bits, gs) vs config.json's quantization declaration.

    Disagreement on layers 0..44 is a refusal.  Layer 45 (MTP, never executed
    by standard logits) is measured to be OUTSIDE the orcarouter override map
    while its stored shapes derive 5/6-bit modules; that is recorded as a
    disclosure, not invented into agreement and not refused.
    """
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise _fail("config.json has no quantization/quantization_config block")
    default_bits = int(quant.get("bits", -1))
    default_gs = int(quant.get("group_size", -1))
    if default_bits not in SUPPORTED_BITS or default_gs <= 0:
        raise _fail(f"config quantization defaults unsupported: bits={default_bits} gs={default_gs}")
    declared: Dict[str, Tuple[int, int]] = {}
    for key, value in quant.items():
        if not isinstance(value, Mapping):
            continue
        for module in translate_override_key(key):
            declared[module] = (int(value.get("bits", default_bits)),
                                int(value.get("group_size", default_gs)))
    disagreements: List[str] = []
    undeclared_l45: List[str] = []
    for module, row in census["quantized"].items():
        got = (row["bits"], row["group_size"])
        layer_match = _LAYER_RE.search(module + ".")
        layer = int(layer_match.group(1)) if layer_match else None
        want = declared.get(module, (default_bits, default_gs))
        if got == want:
            continue
        if layer == MTP_LAYER and module not in declared:
            undeclared_l45.append(f"{module}: derived b{got[0]}/gs{got[1]}")
            continue
        disagreements.append(f"{module}: derived b{got[0]}/gs{got[1]} != declared b{want[0]}/gs{want[1]}")
    if disagreements:
        raise _fail(
            f"config quantization declaration disagrees with stored shapes on "
            f"{len(disagreements)} modules (first: {disagreements[0]})"
        )
    return {
        "default_bits": default_bits,
        "default_group_size": default_gs,
        "override_keys": sum(1 for v in quant.values() if isinstance(v, Mapping)),
        "declared_modules_checked": len(census["quantized"]) - len(undeclared_l45),
        "mtp_layer45_modules_outside_config_overrides": len(undeclared_l45),
        "mtp_layer45_examples": undeclared_l45[:4],
    }


# ---------------------------------------------------------------------------
# the surface object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MlxSurface:
    root: Path
    repo: Optional[str]
    revision: str
    config_sha256: str
    index_sha256: str
    official_census_sha256: str
    official_source_repo: str
    official_source_revision: str
    default_bits: int
    default_group_size: int
    weight_map: Mapping[str, str]
    tensor_meta: Mapping[str, Tuple[str, Tuple[int, ...]]]
    census: Mapping[str, Any]
    config: Mapping[str, Any]
    config_agreement: Mapping[str, Any]
    shard_hash_verification: str  # "full" | "skipped"
    metadata_only: bool
    text_vocab_size: int
    container_overhead_bytes: int = 0
    declared_total_size: Optional[int] = None

    def quant_row(self, module: str) -> Mapping[str, Any]:
        row = self.census["quantized"].get(module)
        if row is None:
            raise _fail(f"not a quantized module: {module}")
        return row

    def student_label(self) -> str:
        histogram = self.census["bits_histogram"]
        base = f"mlx-affine-b{self.default_bits}-gs{self.default_group_size}"
        if len(histogram) > 1:
            digest = _sha256_bytes(_canonical_json(histogram))[:8]
            return f"{base}-mixed-{digest}"
        return base

    def scope_policy(self) -> Dict[str, Any]:
        census = self.census
        return {
            "schema": SCOPE_POLICY_SCHEMA,
            "policy": "quantization_extends_beyond_routed_experts",
            "measured_from": "index + shard-header census, not the README",
            "quantized_module_count": len(census["quantized"]),
            "routed_expert_modules": len(census["routed_modules"]),
            "mtp_expert_modules": len(census["routed_mtp_modules"]),
            "nonrouted_quantized_modules": len(census["nonrouted_quantized_modules"]),
            "nonrouted_quantized_kinds": _kind_histogram(census["nonrouted_quantized_modules"]),
            "passthrough_tensor_count": len(census["passthrough_tensors"]),
            "bits_histogram": dict(census["bits_histogram"]),
            "activations": "not_quantized_by_this_format_weights_only",
            "vision_policy": "source_dtype_passthrough_from_the_quant_snapshot",
        }

    def checkpoint_identity_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json(
                {
                    "schema": MLX_IDENTITY_SCHEMA,
                    "mlx_repo": self.repo,
                    "mlx_revision": self.revision,
                    "format": MLX_FORMAT,
                    "default_bits": self.default_bits,
                    "default_group_size": self.default_group_size,
                    "bits_histogram": dict(self.census["bits_histogram"]),
                    "config_sha256": self.config_sha256,
                    "index_sha256": self.index_sha256,
                    "official_census_sha256": self.official_census_sha256,
                    "official_source_repo": self.official_source_repo,
                    "official_source_revision": self.official_source_revision,
                    "scope_policy": self.scope_policy(),
                    "shard_hash_verification": self.shard_hash_verification,
                    "codebook": "affine-uint-grid",
                    "nonrouted_policy": "decoded_bf16_view_from_the_quant_snapshot",
                    "seal_disclosure": SEAL_DISCLOSURE,
                }
            )
        )

    def fetch_ledger(self) -> Dict[str, Any]:
        """Exact artifact bytes by class, from the shard-header data_offsets.

        The classes sum to ``total_artifact``; adding the safetensors container
        overhead (each shard's 8-byte length prefix plus its JSON header) gives
        ``on_disk_total_bytes``, which is reconciled against the index's own
        declared ``metadata.total_size`` - orcarouter's index declares the
        on-disk figure, and which convention a snapshot used is RECORDED rather
        than assumed (writers differ; transformers declares tensor bytes only).
        """
        sizes = {"routed_packed": 0, "mtp_packed": 0, "nonrouted_quantized_packed": 0,
                 "passthrough": 0}
        routed = set(self.census["routed_modules"])
        mtp = set(self.census["routed_mtp_modules"])
        nonrouted = set(self.census["nonrouted_quantized_modules"])
        for name, (dtype, shape) in self.tensor_meta.items():
            nbytes = _tensor_nbytes(dtype, shape)
            module = None
            for suffix in (".weight", ".scales", ".biases"):
                if name.endswith(suffix):
                    module = name[: -len(suffix)]
                    break
            if module in routed:
                sizes["routed_packed"] += nbytes
            elif module in mtp:
                sizes["mtp_packed"] += nbytes
            elif module in nonrouted:
                sizes["nonrouted_quantized_packed"] += nbytes
            else:
                sizes["passthrough"] += nbytes
        sizes["total_artifact"] = sum(sizes.values())
        sizes["container_overhead_bytes"] = self.container_overhead_bytes
        sizes["on_disk_total_bytes"] = sizes["total_artifact"] + self.container_overhead_bytes
        sizes["index_declared_total_size"] = self.declared_total_size
        if self.declared_total_size is None:
            sizes["declared_total_matches"] = "index_declares_no_total_size"
        elif self.declared_total_size == sizes["on_disk_total_bytes"]:
            sizes["declared_total_matches"] = "on_disk_with_container_headers"
        elif self.declared_total_size == sizes["total_artifact"]:
            sizes["declared_total_matches"] = "tensor_bytes_only"
        else:
            sizes["declared_total_matches"] = "neither_convention"
        sizes["decoded_nonrouted_view_estimate"] = sum(
            2 * row["out_features"] * row["in_features"]
            for module, row in self.census["quantized"].items()
            if module in nonrouted
        ) + sizes["passthrough"]
        return sizes


def _kind_histogram(modules) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for module in modules:
        kind = re.sub(r"\.\d+\.", ".N.", module)
        out[kind] = out.get(kind, 0) + 1
    return dict(sorted(out.items()))


_DTYPE_BYTES = {"U32": 4, "F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "U8": 1, "F64": 8}


def _tensor_nbytes(dtype: str, shape) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * _DTYPE_BYTES.get(dtype, 0)


def load_mlx_surface(
    root: "str | Path",
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
    official_census_path: "str | Path | None" = None,
    require_shard_hashes: bool = True,
) -> MlxSurface:
    root = Path(root).resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = _read_json(config_path, "mlx config.json")
    index = _read_json(index_path, "mlx model.safetensors.index.json")

    if not isinstance(config.get("quantization"), Mapping) and not isinstance(
        config.get("quantization_config"), Mapping
    ):
        raise _fail("config.json has no mlx quantization block - not an MLX quantized snapshot")
    quant = config.get("quantization") or config.get("quantization_config")
    mode = quant.get("mode")
    if mode not in (None, "affine"):
        raise _fail(f"mlx quantization mode {mode!r} is a named exclusion (affine only)")
    text = config.get("text_config", {})
    if (
        config.get("architectures") != ["Glm5NextForConditionalGeneration"]
        or config.get("model_type") != "glm5_next"
        or text.get("model_type") != "glm5_next_text"
        or text.get("num_hidden_layers") != 45
        or text.get("num_nextn_predict_layers") != 1
        or text.get("n_routed_experts") != NUM_EXPERTS
        or text.get("hidden_size") != 4096
        or text.get("moe_intermediate_size") != 2048
    ):
        raise _fail("mlx checkpoint does not carry official GLM5Next main/MTP geometry")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise _fail("index has no weight_map")
    if revision is not None and _REVISION.fullmatch(revision) is None:
        raise _fail("--mlx-revision must be the immutable 40-hex repo commit")

    official_path = Path(official_census_path) if official_census_path else OFFICIAL_CENSUS_DEFAULT
    resolved = _maybe_gz(official_path if official_path.suffix else official_path)
    if resolved is None:
        raise _fail(
            f"official BF16 shape census absent: {official_path} - it is the architecture "
            "yardstick every derived (bits, group_size) is checked against"
        )
    official, official_meta = load_official_census(resolved)

    tensor_meta, metadata_only, container_overhead = _collect_tensor_meta(root, weight_map)
    census = census_index(weight_map, tensor_meta, official)
    agreement = crosscheck_config_declaration(config, census)

    marker = root / "mlx-shards-verified.json"
    if marker.is_file():
        verified = _read_json(marker, "shard verification marker")
        if (
            verified.get("schema") != MLX_SHARDS_VERIFIED_SCHEMA
            or verified.get("index_sha256") != _sha256_file(index_path)
            or verified.get("all_verified") is not True
        ):
            raise _fail("stale/foreign mlx-shards-verified.json - re-run verify-shards")
        shard_hash_verification = "full"
    elif require_shard_hashes and not metadata_only:
        raise _fail(
            "whole-shard sha256 verification marker absent: run "
            f"`python mlx_surface.py verify-shards --mlx-root {root}` first, or pass "
            "--skip-shard-hashes for a disclosed unverified read"
        )
    else:
        shard_hash_verification = "skipped"

    return MlxSurface(
        root=root,
        repo=repo,
        revision=revision or "unpinned-local-snapshot",
        config_sha256=_sha256_file(config_path),
        index_sha256=_sha256_file(index_path),
        official_census_sha256=official_meta["sha256"],
        official_source_repo=official_meta["source_repo"],
        official_source_revision=official_meta["source_revision"],
        default_bits=int(quant.get("bits", -1)),
        default_group_size=int(quant.get("group_size", -1)),
        weight_map=dict(weight_map),
        tensor_meta=tensor_meta,
        census=census,
        config=config,
        config_agreement=agreement,
        shard_hash_verification=shard_hash_verification,
        metadata_only=metadata_only,
        text_vocab_size=int(text["vocab_size"]),
        container_overhead_bytes=container_overhead,
        declared_total_size=(
            int(index["metadata"]["total_size"])
            if isinstance(index.get("metadata"), Mapping)
            and index["metadata"].get("total_size") is not None
            else None
        ),
    )


def _collect_tensor_meta(
    root: Path, weight_map: Mapping[str, str]
) -> Tuple[Dict[str, Tuple[str, Tuple[int, ...]]], bool, int]:
    """((dtype, shape) per index tensor, metadata_only, container overhead bytes).

    Prefers real local shard headers; falls back to a ``shard-headers.json[.gz]``
    sidecar (the fetch-meta output) so a dry-run can census the REAL repo
    metadata with zero weight bytes on disk.  The container overhead is each
    shard's 8-byte length prefix plus its JSON header - the difference between
    "bytes of tensor" and "bytes on disk", which is what lets the fetch ledger
    be reconciled against the index's declared total_size.
    """
    shards = sorted(set(weight_map.values()))
    have_all_shards = all((root / shard).is_file() for shard in shards)
    meta: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    overhead = 0
    if have_all_shards:
        for shard in shards:
            header = read_safetensors_header(root / shard)
            overhead += int(header["__header_len__"]) + 8
            for name, value in header.items():
                if name in ("__metadata__", "__header_len__"):
                    continue
                meta[name] = (value["dtype"], tuple(value["shape"]))
        return meta, False, overhead
    sidecar = _maybe_gz(root / "shard-headers.json")
    if sidecar is None:
        missing = [shard for shard in shards if not (root / shard).is_file()]
        raise _fail(
            f"{len(missing)} shards absent (first: {missing[0]}) and no shard-headers.json "
            "sidecar - fetch shards, or run `mlx_surface.py fetch-meta` for a metadata-only census"
        )
    headers = _read_json(sidecar, "shard-headers sidecar")
    for shard in shards:
        if shard not in headers:
            raise _fail(f"shard-headers sidecar lacks {shard}")
        overhead += int(headers[shard].get("__header_len__", 0)) + 8
        for name, value in headers[shard].items():
            if name in ("__metadata__", "__header_len__"):
                continue
            meta[name] = (value["dtype"], tuple(value["shape"]))
    return meta, True, overhead


# ---------------------------------------------------------------------------
# dequant - plain torch, fp32 accumulate, byte-level unpack (no int64 math
# beyond gather indices, no float64 anywhere: MPS-safe by construction)
# ---------------------------------------------------------------------------
def unpack_affine_codes(weight, *, bits: int, in_features: int):
    """U32-packed little-endian bitstream -> int32 codes [rows, in_features]."""
    import torch

    if weight.dtype == torch.uint32 or weight.dtype == torch.int32:
        raw = weight.contiguous().view(torch.uint8)
    elif weight.dtype == torch.uint8:
        raw = weight.contiguous()
    else:
        raise _fail(f"packed weight dtype {weight.dtype} unsupported")
    rows, nbytes = raw.shape
    if in_features * bits != nbytes * 8:
        raise _fail(
            f"packed geometry differs: {in_features} x {bits} bits != {nbytes} bytes/row"
        )
    device = raw.device
    positions = torch.arange(in_features, device=device, dtype=torch.int64) * bits
    byte_idx = positions >> 3
    shift = (positions & 7).to(torch.int32)
    hi_idx = torch.clamp(byte_idx + 1, max=nbytes - 1)
    lo = raw.index_select(1, byte_idx).to(torch.int32)
    hi = raw.index_select(1, hi_idx).to(torch.int32)
    mask = (1 << bits) - 1
    return ((lo >> shift) | (hi << (8 - shift))) & mask


def dequant_affine(weight, scales, biases, *, bits: int, group_size: int):
    """fp32 dequant: q * scale + bias, groups along the input (last) axis.

    Exact in fp32: q is an integer <= 255, scales/biases are f16/bf16 (exact in
    fp32), and the fused multiply-add is two correctly-rounded fp32 ops whose
    single f16 rounding was proven bitwise-equal to mlx.core.dequantize.
    """
    import torch

    rows, groups = scales.shape
    in_features = groups * group_size
    codes = unpack_affine_codes(weight, bits=bits, in_features=in_features)
    if codes.shape[0] != rows:
        raise _fail(f"weight rows {codes.shape[0]} != scales rows {rows}")
    q = codes.to(torch.float32).reshape(rows, groups, group_size)
    scale = scales.to(torch.float32).unsqueeze(-1)
    bias = biases.to(torch.float32).unsqueeze(-1)
    return (q * scale + bias).reshape(rows, in_features)


def pack_affine_reference(codes, *, bits: int):
    """numpy reference packer (inverse of unpack_affine_codes) for selftests."""
    import numpy as np

    codes = np.asarray(codes, dtype=np.uint32)
    rows, cols = codes.shape
    total_bits = cols * bits
    if total_bits % 32:
        raise _fail("reference packer needs cols*bits divisible by 32")
    out = np.zeros((rows, total_bits // 32), dtype=np.uint32)
    for row in range(rows):
        for column in range(cols):
            value = int(codes[row, column]) & ((1 << bits) - 1)
            position = column * bits
            word, offset = position >> 5, position & 31
            out[row, word] |= (value << offset) & 0xFFFFFFFF
            spill = offset + bits - 32
            if spill > 0:
                out[row, word + 1] |= value >> (bits - spill)
    return out


# ---------------------------------------------------------------------------
# routed expert source for stream_score's ExpertStreamer
# ---------------------------------------------------------------------------
class MlxExpertSource:
    """Routed experts decoded from the MLX snapshot - shaped like
    ``NativeCheckpointSource``: ``load()`` returns (fp32 CPU tensor, census row)
    and the CALLER does the device move, ``fuse_gate_up``, the single bf16
    rounding, the slab ``copy_`` and the ``torch.equal`` close check.

    safetensors handles are cached PER THREAD (the streamer reads with a pool
    and handles are not documented thread-safe).
    """

    def __init__(self, surface: MlxSurface):
        if surface.metadata_only:
            raise _fail("metadata-only surface cannot stream weights (shards absent)")
        self.surface = surface
        self._local = threading.local()
        self._lock = threading.Lock()
        self.shards_read: set = set()
        self.bytes_read = 0

    # Per-thread handles, LRU-bounded: a 62-shard snapshot read by N pool
    # threads would otherwise hold 62*N mmaps open at once (macOS's default
    # 256-fd soft limit is reachable).  One layer's routed experts live in a
    # couple of adjacent shards, so a small window keeps the hit rate at ~1.
    MAX_OPEN_SHARDS_PER_THREAD = 8

    def _handle(self, shard: str):
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = self._local.handles = OrderedDict()
        handle = cache.get(shard)
        if handle is not None:
            cache.move_to_end(shard)
            return handle
        from safetensors import safe_open

        handle = safe_open(str(self.surface.root / shard), framework="pt", device="cpu")
        enter = getattr(handle, "__enter__", None)
        if enter is not None:
            handle = enter()
        cache[shard] = handle
        while len(cache) > self.MAX_OPEN_SHARDS_PER_THREAD:
            _, evicted = cache.popitem(last=False)
            exit_ = getattr(evicted, "__exit__", None)
            if exit_ is not None:
                exit_(None, None, None)
        return handle

    def _tensor(self, name: str):
        shard = self.surface.weight_map.get(name)
        if shard is None:
            raise _fail(f"tensor not in weight_map: {name}")
        return self._handle(shard).get_tensor(name), shard

    def load(self, *, layer: int, expert: int, projection: str):
        module = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
        row = self.surface.quant_row(module)
        weight, shard = self._tensor(module + ".weight")
        scales, _ = self._tensor(module + ".scales")
        biases, _ = self._tensor(module + ".biases")
        decoded = dequant_affine(
            weight, scales, biases, bits=row["bits"], group_size=row["group_size"]
        )
        if tuple(decoded.shape) != PROJECTION_SHAPE[projection]:
            raise _fail(
                f"decoded {module} has shape {tuple(decoded.shape)}, expected "
                f"{PROJECTION_SHAPE[projection]}"
            )
        nbytes = sum(
            int(t.numel() * t.element_size()) for t in (weight, scales, biases)
        )
        with self._lock:
            self.shards_read.add(shard)
            self.bytes_read += nbytes
        census_row = {
            "tensor": module + ".weight",
            "shard": shard,
            "bytes": nbytes,
            "dtype": "u32-affine",
            "quant": {"bits": row["bits"], "group_size": row["group_size"]},
        }
        return decoded, census_row


# ---------------------------------------------------------------------------
# non-routed DECODED view: everything except `.mlp.experts.N.` materialized as
# real BF16 safetensors shards, so the sealed from_pretrained constructor runs
# over it unchanged (the whole point: transformers' own key conversion, buffer
# construction and dtype handling, not a re-implementation)
# ---------------------------------------------------------------------------
def prepare_nonrouted_view_decoded(
    surface: MlxSurface,
    work_dir: Path,
    *,
    max_shard_bytes: int = 4 << 30,
    progress: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    view = Path(work_dir) / "mlx-nonrouted-decoded-view"
    receipt_path = view / "mlx-view-receipt.json"
    binding = {
        "schema": MLX_VIEW_RECEIPT_SCHEMA,
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "official_census_sha256": surface.official_census_sha256,
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
    }
    if receipt_path.is_file():
        previous = _read_json(receipt_path, "mlx view receipt")
        if all(previous.get(key) == value for key, value in binding.items()) and all(
            (view / shard).is_file() and (view / shard).stat().st_size == size
            for shard, size in previous.get("shard_sizes", {}).items()
        ):
            return view, {**previous, "reused": True}
        raise _fail(
            f"stale decoded view at {view} (different snapshot or adapter) - remove it "
            "or point --work-dir elsewhere"
        )
    view.mkdir(parents=True, exist_ok=True)

    config = {k: v for k, v in surface.config.items()
              if k not in ("quantization", "quantization_config")}
    (view / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    # Same aux-file classes stream_score's own prepare_nonrouted_view copies from
    # the BF16 tree, minus the two files this function writes itself and minus
    # this adapter's own sidecars (which are metadata about the artifact, not
    # part of the model directory).
    _OURS = {"config.json", "model.safetensors.index.json", "shard-headers.json",
             "shard-headers.json.gz", "mlx-manifest.json", "mlx-shards-verified.json",
             "mlx-view-receipt.json"}
    for entry in sorted(surface.root.iterdir()):
        if entry.is_dir() or entry.name in _OURS:
            continue
        if entry.suffix in (".json", ".jinja", ".txt", ".model"):
            (view / entry.name).write_bytes(entry.read_bytes())

    keep = [name for name in surface.weight_map
            if EXPERT_RE.search(name) is None
            and not (name.endswith(".scales") or name.endswith(".biases"))]
    keep.sort()
    quantized = surface.census["quantized"]
    handles: Dict[str, Any] = {}

    def _open(shard: str):
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(surface.root / shard), framework="pt", device="cpu")
            handles[shard] = handle
        return handle

    def _load(name: str):
        return _open(surface.weight_map[name]).get_tensor(name)

    new_map: Dict[str, str] = {}
    shard_sizes: Dict[str, int] = {}
    shard_hashes: Dict[str, str] = {}
    bucket: Dict[str, Any] = {}
    bucket_bytes = 0
    shard_index = 0
    decoded_count = 0
    passthrough_count = 0
    started = time.monotonic()

    def _flush():
        nonlocal bucket, bucket_bytes, shard_index
        if not bucket:
            return
        shard_index += 1
        shard_name = f"view-{shard_index:05d}.safetensors"
        path = view / shard_name
        save_file(bucket, str(path), metadata={"format": "pt"})
        for name in bucket:
            new_map[name] = shard_name
        shard_sizes[shard_name] = path.stat().st_size
        shard_hashes[shard_name] = _sha256_file(path)
        if progress:
            print(json.dumps({"view_shard": shard_name, "tensors": len(bucket),
                              "bytes": shard_sizes[shard_name]}), flush=True)
        bucket = {}
        bucket_bytes = 0

    with torch.inference_mode():
        for name in keep:
            module = name[: -len(".weight")] if name.endswith(".weight") else None
            if module in quantized:
                row = quantized[module]
                tensor = dequant_affine(
                    _load(name), _load(module + ".scales"), _load(module + ".biases"),
                    bits=row["bits"], group_size=row["group_size"],
                ).to(torch.bfloat16)
                decoded_count += 1
            else:
                tensor = _load(name)
                passthrough_count += 1
            bucket[name] = tensor.contiguous()
            bucket_bytes += tensor.numel() * tensor.element_size()
            if bucket_bytes >= max_shard_bytes:
                _flush()
    _flush()
    total = sum(shard_sizes.values())
    (view / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}),
        encoding="utf-8",
    )
    record = dict(binding)
    record.update(
        {
            "view_path": str(view),
            "nonrouted_tensor_count": len(keep),
            "decoded_module_count": decoded_count,
            "passthrough_tensor_count": passthrough_count,
            # what the view deliberately does NOT contain: the routed (and MTP)
            # expert modules, which the streamer decodes per layer instead
            "routed_modules_filtered": len(surface.census["routed_modules"])
            + len(surface.census["routed_mtp_modules"]),
            "routed_stored_tensors_filtered": 3 * (
                len(surface.census["routed_modules"])
                + len(surface.census["routed_mtp_modules"])
            ),
            "shards_written": len(shard_sizes),
            "shard_sizes": shard_sizes,
            "shard_sha256": shard_hashes,
            "total_bytes": total,
            "config_quantization_block_stripped": True,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
    )
    receipt_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return view, record


def verify_nonrouted_passthrough(
    surface: MlxSurface,
    bf16_root: "str | Path",
    *,
    sample_count: int = 48,
) -> Dict[str, Any]:
    """OPTIONAL cross-check: passthrough tensors vs the official BF16 tree.

    The quantizer kept these tensors untouched, so byte equality against the
    official checkpoint is expected; a deterministic sample is byte-compared.
    Never required (the official tree is NOT an input of this source).
    """
    import numpy as np
    import torch
    from safetensors import safe_open

    bf16_root = Path(bf16_root).resolve()
    bf16_index = _read_json(bf16_root / "model.safetensors.index.json", "official BF16 index")
    official_map = bf16_index["weight_map"]
    anchors = [name for name in ("model.language_model.embed_tokens.weight", "lm_head.weight",
                                 "model.language_model.norm.weight")
               if name in surface.tensor_meta]
    pool = sorted(set(surface.census["passthrough_tensors"]) - set(anchors))
    rng = np.random.default_rng(0x316C78)  # deterministic sample
    picks = rng.choice(len(pool), size=min(sample_count, len(pool)), replace=False)
    chosen = anchors + [pool[int(index)] for index in sorted(picks)]
    ours_handles: Dict[str, Any] = {}
    theirs_handles: Dict[str, Any] = {}

    def _get(handles, root, mapping, name):
        shard = mapping[name]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(Path(root) / shard), framework="pt", device="cpu")
            handles[shard] = handle
        return handle.get_tensor(name)

    for name in chosen:
        if name not in official_map:
            raise _fail(f"official BF16 index lacks {name}")
        ours = _get(ours_handles, surface.root, surface.weight_map, name)
        theirs = _get(theirs_handles, bf16_root, official_map, name)
        if ours.dtype != theirs.dtype or tuple(ours.shape) != tuple(theirs.shape):
            raise _fail(f"passthrough tensor geometry differs from official: {name}")
        if not torch.equal(ours, theirs):
            raise _fail(f"passthrough tensor bytes differ from official BF16: {name}")
    return {"mode": "sample", "byte_compared_tensors": len(chosen), "all_equal": True}


# ---------------------------------------------------------------------------
# identity + shard hashing + remote metadata
# ---------------------------------------------------------------------------
def mlx_reader_identity(runner_path: "str | Path", surface: MlxSurface) -> Dict[str, Any]:
    body = {
        "schema": MLX_READER_IDENTITY_SCHEMA,
        "mode": "offline_mlx_affine_u32_dequant_to_bf16_for_logit_measurement",
        "serving_kernel": False,
        "default_bits": surface.default_bits,
        "default_group_size": surface.default_group_size,
        "bits_histogram": dict(surface.census["bits_histogram"]),
        "codebook": "affine-uint-grid",
        "decode_dtype": "float32_accumulate_single_bf16_rounding",
        "adapter_sha256": _sha256_file(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(Path(runner_path).resolve()),
        "seal_disclosure": SEAL_DISCLOSURE,
        "dtype_disclosure": DTYPE_DISCLOSURE,
    }
    body["runtime_reader_sha256"] = _sha256_bytes(_canonical_json(body))
    return body


def verify_shard_hashes(root: "str | Path") -> Dict[str, Any]:
    """Hash every shard against mlx-manifest.json (fetch-meta output); marker file."""
    root = Path(root).resolve()
    manifest = _read_json(root / "mlx-manifest.json", "mlx-manifest.json (run fetch-meta first)")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise _fail("mlx-manifest.json carries no files table")
    rows: List[Dict[str, Any]] = []
    started = time.monotonic()
    for name, entry in sorted(files.items()):
        if not name.endswith(".safetensors"):
            continue
        path = root / name
        if not path.is_file():
            raise _fail(f"shard listed in manifest is absent: {path}")
        observed = _sha256_file(path)
        ok = observed == entry["sha256"] and path.stat().st_size == int(entry["bytes"])
        rows.append({"shard": name, "ok": ok})
        if not ok:
            raise _fail(f"shard hash differs from mlx-manifest.json: {path}")
    if not rows:
        raise _fail("mlx-manifest.json lists no safetensors shards")
    record = {
        "schema": MLX_SHARDS_VERIFIED_SCHEMA,
        "root": str(root),
        "index_sha256": _sha256_file(root / "model.safetensors.index.json"),
        "manifest_sha256": _sha256_file(root / "mlx-manifest.json"),
        "shards": len(rows),
        "all_verified": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (root / "mlx-shards-verified.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, tries: int = 4) -> bytes:
    import urllib.request

    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(request, timeout=120).read()
        except Exception:  # noqa: BLE001 - retried, re-raised on the last attempt
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise _fail("unreachable")


def _http_range(url: str, start: int, end: int) -> bytes:
    return _http_get(url, headers={"Range": f"bytes={start}-{end}"})


def fetch_meta(repo: str, revision: str, out: Path, *, subdir: str = "") -> Dict[str, Any]:
    """Ranged-fetch config + index + every shard HEADER + the HF file manifest.

    Produces a metadata-only snapshot dir a dry-run can census: config.json,
    model.safetensors.index.json, shard-headers.json.gz, mlx-manifest.json.
    NO weight bytes are fetched (headers are a few hundred KB per shard).
    """
    if _REVISION.fullmatch(revision) is None:
        raise _fail("fetch-meta needs the immutable 40-hex revision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/{prefix}"
    config_raw = _http_get(base + "config.json")
    (out / "config.json").write_bytes(config_raw)
    index_raw = _http_get(base + "model.safetensors.index.json")
    (out / "model.safetensors.index.json").write_bytes(index_raw)
    index = json.loads(index_raw)
    shards = sorted(set(index["weight_map"].values()))
    headers: Dict[str, Any] = {}
    fetched_bytes = len(config_raw) + len(index_raw)
    for shard in shards:
        url = base + shard
        head8 = _http_range(url, 0, 7)
        (length,) = struct.unpack("<Q", head8)
        raw = _http_range(url, 8, 8 + length - 1)
        header = json.loads(raw)
        header["__header_len__"] = length
        headers[shard] = header
        fetched_bytes += 8 + length
    with gzip.open(out / "shard-headers.json.gz", "wb", compresslevel=9) as fh:
        fh.write(json.dumps(headers, sort_keys=True, separators=(",", ":")).encode())
    tree = json.loads(
        _http_get(f"https://huggingface.co/api/models/{repo}/tree/{revision}"
                  f"{'/' + subdir.rstrip('/') if subdir else ''}?recursive=false")
    )
    files: Dict[str, Any] = {}
    for entry in tree:
        if entry.get("type") != "file":
            continue
        name = entry["path"]
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        lfs = entry.get("lfs") or {}
        files[name] = {"bytes": int(entry.get("size", -1)), "sha256": lfs.get("oid")}
    (out / "mlx-manifest.json").write_text(
        json.dumps({"schema": "malaiwah.glm53-mlx-hf-manifest.v1", "repo": repo,
                    "revision": revision, "subdir": subdir or None, "files": files},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"repo": repo, "revision": revision, "shards": len(shards),
            "metadata_bytes_fetched": fetched_bytes, "out": str(out)}


# ---------------------------------------------------------------------------
# standalone CLI
# ---------------------------------------------------------------------------
def _surface_summary(surface: MlxSurface) -> Dict[str, Any]:
    return {
        "schema": MLX_SURFACE_SCHEMA,
        "root": str(surface.root),
        "format": MLX_FORMAT,
        "mlx_repo": surface.repo,
        "mlx_revision": surface.revision,
        "default_bits": surface.default_bits,
        "default_group_size": surface.default_group_size,
        "bits_histogram": dict(surface.census["bits_histogram"]),
        "student_label": surface.student_label(),
        "config_sha256": surface.config_sha256,
        "index_sha256": surface.index_sha256,
        "official_census_sha256": surface.official_census_sha256,
        "official_source_repo": surface.official_source_repo,
        "official_source_revision": surface.official_source_revision,
        "config_agreement": dict(surface.config_agreement),
        "scope_policy": surface.scope_policy(),
        "fetch_ledger": surface.fetch_ledger(),
        "shard_hash_verification": surface.shard_hash_verification,
        "metadata_only": surface.metadata_only,
        "checkpoint_identity_sha256": surface.checkpoint_identity_sha256(),
        "seal_disclosure": SEAL_DISCLOSURE,
    }


_NP_DTYPE = {"U32": "<u4", "F16": "<f2", "F32": "<f4", "BF16": "<u2"}
_DTYPE_ITEMSIZE = {"U32": 4, "F32": 4, "F16": 2, "BF16": 2}


def fetch_tensor_ranged(
    repo: str, revision: str, shard: str, entry: Mapping[str, Any], header_len: int,
    *, subdir: str = "", rows: Optional[int] = None,
):
    """One tensor's exact bytes over HTTP range -> torch tensor (BF16 via int16 view).

    ``rows`` fetches only the first N output rows.  Affine packing is per-row
    (element e of row r lives at bits [e*b, (e+1)*b) of THAT row's byte stream),
    and rows are contiguous in a row-major safetensors buffer, so a row prefix
    is a self-contained, independently decodable tensor - which is what makes a
    multi-GB tensor (an 8-bit embedding, say) cross-checkable for 256 KB.
    """
    import numpy as np
    import torch

    prefix = f"{subdir.rstrip('/')}/" if subdir else ""
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{prefix}{shard}"
    start, end = entry["data_offsets"]
    base = 8 + header_len
    dtype, shape = entry["dtype"], list(entry["shape"])
    if rows is not None:
        if len(shape) != 2:
            raise _fail("row-prefix fetch needs a 2-D tensor")
        if rows > shape[0]:
            raise _fail(f"tensor has {shape[0]} rows, asked for {rows}")
        row_bytes = shape[1] * _DTYPE_ITEMSIZE[dtype]
        if (end - start) != shape[0] * row_bytes:
            raise _fail("header data_offsets disagree with dtype/shape - refusing a blind slice")
        shape = [rows, shape[1]]
        end = start + rows * row_bytes
    raw = _http_range(url, base + start, base + end - 1)
    array = np.frombuffer(raw, dtype=_NP_DTYPE[dtype]).copy().reshape(shape)
    if dtype == "BF16":
        return torch.from_numpy(array).view(torch.bfloat16)
    return torch.from_numpy(array)


def _module_tensors(surface: MlxSurface, module: str, sidecar_headers, args):
    """weight/scales/biases for one module - local shards, or ranged HTTP fetch."""
    if not surface.metadata_only:
        source = MlxExpertSource(surface)
        return tuple(source._tensor(module + suffix)[0]
                     for suffix in (".weight", ".scales", ".biases"))
    if not (args.repo and args.revision):
        raise _fail("metadata-only crosscheck needs --repo and --revision for ranged fetch")
    out = []
    for suffix in (".weight", ".scales", ".biases"):
        name = module + suffix
        shard = surface.weight_map[name]
        header = sidecar_headers[shard]
        out.append(fetch_tensor_ranged(args.repo, args.revision, shard, header[name],
                                       header["__header_len__"],
                                       subdir=getattr(args, "subdir", "") or ""))
    return tuple(out)


def compare_against_mlx(weight, scales, biases, *, bits: int, group_size: int):
    """Our fp32 dequant vs ``mlx.core.dequantize`` on the SAME stored tensors.

    mlx's kernel emits the dtype of the scales (F16 in most repos, BF16 in the
    mixed-4_8bit and Q9 ones), so the comparison is made in that dtype: our fp32
    result rounded ONCE must reproduce mlx's output bit for bit.  In fp32 the
    two differ by at most one ulp - mlx fuses the multiply-add and we do not -
    which is why the fp32 delta is reported alongside, never asserted to be 0.
    """
    import mlx.core as mx
    import numpy as np
    import torch

    def _to_mx(tensor):
        if tensor.dtype == torch.bfloat16:
            return mx.array(tensor.view(torch.int16).numpy()).view(mx.bfloat16)
        return mx.array(tensor.numpy())

    ours32 = dequant_affine(weight, scales, biases, bits=bits, group_size=group_size)
    theirs = mx.dequantize(_to_mx(weight), _to_mx(scales), _to_mx(biases),
                           group_size=group_size, bits=bits)
    if theirs.dtype == mx.bfloat16:
        out_dtype = "bfloat16"
        theirs_bits = np.array(theirs.view(mx.uint16))
        ours_bits = ours32.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        theirs_f32 = np.array(theirs.astype(mx.float32))
    else:
        out_dtype = "float16"
        theirs_np = np.array(theirs)
        theirs_bits = theirs_np.view(np.uint16)
        ours_bits = ours32.to(torch.float16).numpy().view(np.uint16)
        theirs_f32 = theirs_np.astype(np.float32)
    reference = (out_dtype, theirs_bits)
    return ours32, reference, {
        "mlx_output_dtype": out_dtype,
        "bitwise_equal_at_mlx_output_dtype": bool(np.array_equal(ours_bits, theirs_bits)),
        "max_abs_fp32_vs_mlx_output": float(np.abs(ours32.numpy() - theirs_f32).max()),
        "sample_w_first4": [float(v) for v in ours32.flatten()[:4]],
    }


def _cmd_crosscheck(args) -> int:
    """Reference cross-check vs mlx.core.dequantize on real tensors (macOS)."""
    surface = load_mlx_surface(
        args.mlx_root, repo=args.repo, revision=args.revision,
        official_census_path=args.official_census, require_shard_hashes=False,
    )
    try:
        import mlx.core as mx
    except ImportError:
        print(json.dumps({"skip": "mlx not importable on this machine "
                                  "(cross-check is a macOS-only rung)"}))
        return 0
    sidecar_headers = None
    if surface.metadata_only:
        sidecar = _maybe_gz(surface.root / "shard-headers.json")
        sidecar_headers = _read_json(sidecar, "shard-headers sidecar")
    modules = args.module or [
        "model.language_model.layers.3.mlp.experts.0.gate_proj",
        "model.language_model.layers.3.mlp.experts.0.down_proj",
        "model.language_model.layers.12.mlp.shared_experts.gate_proj",
    ]
    results = []
    for module in modules:
        row = surface.quant_row(module)
        weight, scales, biases = _module_tensors(surface, module, sidecar_headers, args)
        ours32, reference, verdict = compare_against_mlx(
            weight, scales, biases, bits=row["bits"], group_size=row["group_size"]
        )
        results.append(dict(
            {"module": module, "bits": row["bits"], "group_size": row["group_size"],
             "scales_dtype": row["scales_dtype"], "shape": list(ours32.shape)},
            **verdict
        ))
        if not verdict["bitwise_equal_at_mlx_output_dtype"]:
            print(json.dumps({"crosscheck": results}, indent=2))
            raise _fail(f"dequant differs from mlx.core.dequantize on {module}")
        if args.save_fixture_slice:
            _save_fixture_slice(Path(args.save_fixture_slice), module, row, weight, scales,
                                biases, reference, rows=int(args.fixture_rows),
                                repo=args.repo or "", revision=args.revision or "")
    print(json.dumps({"crosscheck": results, "mlx_version": getattr(mx, "__version__", None)},
                     indent=2, sort_keys=True))
    return 0


def _save_fixture_slice(out_dir: Path, module: str, row, weight, scales, biases, reference,
                        *, rows: int, repo: str = "", revision: str = "",
                        stem: Optional[str] = None) -> None:
    """First `rows` output rows of a real tensor + mlx's own output BIT PATTERN,
    saved as an npz fixture so the offline selftest replays the mlx equality on
    machines without mlx (packing is per-row: a row slice is decodable alone).

    ``reference`` is the (dtype_name, uint16 bit array) pair compare_against_mlx
    returns - the bits, not a float cast, so the replay is the same comparison.
    """
    import numpy as np
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = module.replace("model.language_model.", "").replace(".", "_")
    ref_dtype, ref_bits = reference

    def _raw(tensor):
        if tensor.dtype == torch.bfloat16:
            return tensor[:rows].view(torch.int16).numpy().view(np.uint16)
        return tensor[:rows].numpy()

    np.savez_compressed(
        out_dir / f"{stem}.npz",
        weight=weight[:rows].numpy(),
        scales=_raw(scales),
        biases=_raw(biases),
        scales_dtype=np.array(str(scales.dtype).replace("torch.", "")),
        ref_bits=ref_bits[:rows],
        ref_dtype=np.array(ref_dtype),
        bits=np.int64(row["bits"]),
        group_size=np.int64(row["group_size"]),
        module=np.array(module),
        repo=np.array(repo),
        revision=np.array(revision),
    )


def _cmd_crosscheck_raw(args) -> int:
    """Kernel cross-check on ONE named tensor of ANY mlx-affine repo, by row prefix.

    The surface refuses to SCORE a foreign dialect (fused switch_mlp, renamed
    modules), and that refusal stands.  The dequant KERNEL, though, is dialect
    independent, so this mode proves it at bit-widths the primary artifact does
    not contain - notably 8-bit, which only the mixed-4_8bit repos carry - by
    ranged-fetching a row prefix of one real tensor and comparing against
    mlx.core.dequantize.  ``group_size`` comes from the repo's own config
    declaration and ``bits`` is DERIVED from the stored shapes; when the config
    also declares bits for that exact module the two must agree.
    """
    try:
        import mlx.core as mx
    except ImportError:
        print(json.dumps({"skip": "mlx not importable on this machine "
                                  "(cross-check is a macOS-only rung)"}))
        return 0
    if _REVISION.fullmatch(args.revision or "") is None:
        raise _fail("crosscheck-raw needs the immutable 40-hex --revision")
    prefix = f"{args.subdir.rstrip('/')}/" if args.subdir else ""
    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{prefix}"
    config = json.loads(_http_get(base + "config.json"))
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise _fail("config.json has no quantization block")
    index = json.loads(_http_get(base + "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    results = []
    for module in args.module:
        names = {suffix: module + "." + suffix for suffix in ("weight", "scales", "biases")}
        for name in names.values():
            if name not in weight_map:
                raise _fail(f"{name} is not in {args.repo}'s index")
        shard = weight_map[names["weight"]]
        if any(weight_map[name] != shard for name in names.values()):
            raise _fail(f"{module}: weight/scales/biases are not in one shard")
        url = base + shard
        (header_len,) = struct.unpack("<Q", _http_range(url, 0, 7))
        header = json.loads(_http_range(url, 8, 8 + header_len - 1))
        declared = quant.get(module)
        group_size = int((declared or {}).get("group_size", quant.get("group_size", -1)))
        if group_size <= 0:
            raise _fail(f"{module}: group_size not declared by config")
        packed_cols = int(header[names["weight"]]["shape"][1])
        groups = int(header[names["scales"]]["shape"][1])
        in_features = groups * group_size
        if (32 * packed_cols) % in_features:
            raise _fail(f"{module}: bits underivable from shapes")
        bits = (32 * packed_cols) // in_features
        if bits not in SUPPORTED_BITS:
            raise _fail(f"{module}: derived bits={bits} outside {SUPPORTED_BITS}")
        declared_bits = int((declared or {}).get("bits", quant.get("bits", -1)))
        if declared is not None and declared_bits != bits:
            raise _fail(
                f"{module}: config declares b{declared_bits} but the stored shapes derive b{bits}"
            )
        rows = int(args.rows)
        tensors = {
            key: fetch_tensor_ranged(args.repo, args.revision, shard, header[name], header_len,
                                     subdir=args.subdir, rows=rows)
            for key, name in names.items()
        }
        row = {"bits": bits, "group_size": group_size}
        ours32, reference, verdict = compare_against_mlx(
            tensors["weight"], tensors["scales"], tensors["biases"],
            bits=bits, group_size=group_size,
        )
        results.append(dict(
            {"repo": args.repo, "revision": args.revision, "module": module,
             "bits": bits, "group_size": group_size,
             "scales_dtype": str(tensors["scales"].dtype).replace("torch.", ""),
             "declared_by_config": None if declared is None else dict(declared),
             "rows_fetched": rows, "shape": list(ours32.shape),
             "packed_bytes_fetched": int(tensors["weight"].numel() * 4)},
            **verdict
        ))
        if not verdict["bitwise_equal_at_mlx_output_dtype"]:
            print(json.dumps({"crosscheck_raw": results}, indent=2))
            raise _fail(f"dequant differs from mlx.core.dequantize on {args.repo}:{module}")
        if args.save_fixture_slice:
            _save_fixture_slice(
                Path(args.save_fixture_slice), module, row, tensors["weight"],
                tensors["scales"], tensors["biases"], reference, rows=rows,
                repo=args.repo, revision=args.revision,
                stem=(args.repo.split("/")[-1] + "-" + module).replace(".", "_"),
            )
    print(json.dumps({"crosscheck_raw": results,
                      "mlx_version": getattr(mx, "__version__", None)},
                     indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dry-run", help="census a snapshot (or fetched metadata) and print the plan")
    p.add_argument("--mlx-root", type=Path, required=True)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--official-census", type=Path)
    p.add_argument("--skip-shard-hashes", action="store_true")

    p = sub.add_parser("fetch-meta", help="ranged-fetch config/index/shard headers (no weights)")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--subdir", default="")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("verify-shards", help="hash every local shard against mlx-manifest.json")
    p.add_argument("--mlx-root", type=Path, required=True)

    p = sub.add_parser("crosscheck-raw",
                       help="prove the dequant KERNEL on one named tensor of ANY mlx-affine "
                            "repo (row prefix over HTTP range; covers bit-widths the scored "
                            "artifact does not contain)")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--subdir", default="")
    p.add_argument("--module", action="append", required=True,
                   help="logical module path as the INDEX spells it, without .weight")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--save-fixture-slice", help="write npz row-slice fixtures to this dir")

    p = sub.add_parser("crosscheck", help="prove the dequant vs mlx.core.dequantize on real tensors")
    p.add_argument("--mlx-root", type=Path, required=True)
    p.add_argument("--repo")
    p.add_argument("--revision")
    p.add_argument("--subdir", default="")
    p.add_argument("--official-census", type=Path)
    p.add_argument("--module", action="append")
    p.add_argument("--save-fixture-slice", help="write npz row-slice fixtures to this dir")
    p.add_argument("--fixture-rows", type=int, default=64)

    args = parser.parse_args()
    if args.command == "fetch-meta":
        print(json.dumps(fetch_meta(args.repo, args.revision, args.out, subdir=args.subdir),
                         indent=2, sort_keys=True))
        return 0
    if args.command == "verify-shards":
        print(json.dumps(verify_shard_hashes(args.mlx_root), sort_keys=True))
        return 0
    if args.command == "crosscheck":
        return _cmd_crosscheck(args)
    if args.command == "crosscheck-raw":
        return _cmd_crosscheck_raw(args)

    surface = load_mlx_surface(
        args.mlx_root, repo=args.repo, revision=args.revision,
        official_census_path=args.official_census,
        require_shard_hashes=not args.skip_shard_hashes,
    )
    print(json.dumps(_surface_summary(surface), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    # A refusal is an ANSWER, not a crash: these tools are run by hand before a
    # paid capture, so `_fail`'s named message prints on one line instead of
    # under a traceback -- the same handler gguf_surface uses, and the same exit
    # code (2), so a caller can tell "this artifact is refused" from "the tool
    # broke".
    import sys

    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
