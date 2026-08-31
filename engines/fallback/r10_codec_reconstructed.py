"""RECONSTRUCTED R10TrellisCodec — fallback for Brandon M. Music's sealed codec.

THIS IS NOT THE SEALED IMPLEMENTATION.  Brandon's `r7_encoder/r10_codec.py`
(class ``R10TrellisCodec``) and his numeric core ``encode_tr3_v31.py`` are in
no public tree (asked upstream in glm-5.3-flash-exl3-4bpw issue #1).  This
module is a clean-room reconstruction assembled from public receipts only:

  * the adapter contract  quant_pipeline/codecs/exl3_mcg.py  (both the
    shapleymcg copy and the glm-5.3-flash-exl3-4bpw copy, which admits K6),
  * the behavioral spec   shapleymcg/tests/test_exl3_mcg.py,
  * the sealed-lineage ancestor  bmmlaw_r7_encoder/trellis.py  (class
    Exl3TrellisCodec) plus constants/types/determinism/hessian from
    glm52-sqg-mcg-experiments @ bf37b066,
  * every call site in the two driver trees (glm53_prepared_backend.py,
    glm53_mcg_preparation.py, qwen_services.CorrectedPinnedGSSProducer,
    candidates/ledger.py, normalization/streaming_v31.py,
    scripts/run_qwen_fast_encode.py, scripts/encode_qwen_attention_k4.py),
  * the pinned exllamav3 @ c5d9c657 trellis kernels (quantize_tiles /
    pack_trellis / unpack_trellis / reconstruct, all instantiated for
    K=1..8 with the MCG codebook, multiplier 0xCBAC1FED).

The numeric heart is exllamav3's own quantize/encode ops — the same kernels
Brandon's v31 numeric core requires (its required-attribute list in the
ancestor loader is exactly exllamav3's exl3_lib/quantize.py public surface).
Every inference beyond those receipts is catalogued in RECONSTRUCTION.md next
to this file.  Receipts produced through this codec MUST disclose the
substitution: encode provenance carries a ``fallback_reconstruction`` block
and the codec identity hashes will NOT match Brandon's sealed closure.

DO NOT claim bit-identity with the sealed core.  The claim we make instead:
this codec is self-consistent (its own pack/unpack/reconstruct oracles hold,
repeat-encoding is byte-deterministic) and kernel-faithful (all trellis math
is executed by the pinned exllamav3 extension with mcg=True).

Staged layout (see stage_r7_encoder below): this file becomes
``r7_encoder/r10_codec.py``; a thin ``trellis.py`` re-exports CodecConfig so
the adapter's ``trellis.CodecConfig(...)`` and closure checks resolve;
``hessian.py`` is copied verbatim from the bmmlaw ancestor for the
prepared-backend's ``r7_encoder.hessian`` imports.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------
# Pinned geometry / codec constants (receipts: bmmlaw_r7_encoder/constants.py,
# exllamav3 exl3_lib/quantize.py, DECISIONS.md).
# --------------------------------------------------------------------------

HAD_K = 128
HAD_N = 128
TRELLIS_TILE = 16
MCG_MULT = 0xCBAC1FED
DEFAULT_SIGMA_REG = 0.025
LDL_FACTORIZATION_POLICY = "cuda-only-no-oom-fallback-v1"
# exllamav3 @ c5d9c657 exl3_lib/quantize.py line 15; shared by all codebooks.
EXPECTED_CODEBOOK_SCALE = 1.24371088
# The extension instantiates every kernel for K=1..8; the 4bpw adapters gate
# 3..5 (shapleymcg) / 3..6 (glm-5.3-flash-exl3-4bpw) at their own layer.  K8
# is the 128-int16-word-per-tile trellis needed by the K6K8 recipe.
KERNEL_BITS = tuple(range(1, 9))

RECONSTRUCTION_SCHEMA = "k6-program.r10-fallback-reconstruction.v1"
RECONSTRUCTION_SOURCES = (
    "bmmlaw_r7_encoder@bf37b06691c68525b74bddfa0a1a8216e695c95f",
    "exllamav3@c5d9c657",
    "quant_pipeline.codecs.exl3_mcg adapter contract",
    "shapleymcg tests/test_exl3_mcg.py",
    "glm53_prepared_backend/qwen_services/run_qwen_fast_encode call sites",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor) -> bytes:
    import torch

    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor) -> str:
    return sha256_bytes(_tensor_bytes(tensor))


# --------------------------------------------------------------------------
# CodecConfig.  Ancestor fields plus ``verify_files`` (receipt: the adapter
# constructs trellis.CodecConfig(..., verify_files=True)).  ``verify_files``
# semantics are an inference: True (production) demands sealed SHA-256s for
# the numeric core and extension and fails closed on drift; False permits
# hash-free loading for local probes only.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodecConfig:
    device: str = "cuda:0"
    sigma_reg: float = DEFAULT_SIGMA_REG
    numeric_core: Path | str | None = None
    numeric_core_sha256: str | None = None
    extension: Path | str | None = None
    extension_sha256: str | None = None
    verify_files: bool = True
    factorization_policy: str = LDL_FACTORIZATION_POLICY


# --------------------------------------------------------------------------
# Numeric core loading.  Ancestor discipline (sha-verified importlib file
# load) with the required-attribute set widened to what the driver trees
# actually consume:  block_rms (streaming_v31._required_core +
# StreamingLayerFitter) and g_scale_gss (CorrectedPinnedGSSProducer calls
# core.g_scale_gss(target, quant_args) -> (scale, objective)).
# --------------------------------------------------------------------------

_REQUIRED_CORE_ATTRIBUTES = (
    # ancestor bmmlaw_r7_encoder/trellis.py load_numeric_core required set
    "block_ldl",
    "ldlq",
    "pack_trellis",
    "blockwise_preapply_had_l_",
    "blockwise_preapply_had_r_",
    "preapply_had_l",
    "preapply_had_r",
    "_lazy_torch",
    "CODEBOOK_SCALE",
    # driver-tree receipts (see module docstring)
    "block_rms",
    "g_scale_gss",
    "quantize_tiles",
    "tensor_core_perm",
    "tensor_core_perm_i",
)


def load_numeric_core(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_files: bool = True,
) -> ModuleType:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"numeric core not found: {source}")
    if verify_files:
        if not expected_sha256:
            raise ValueError("numeric core requires a sealed expected SHA-256")
        if sha256_file(source) != expected_sha256:
            raise ValueError("numeric core bytes differ from the sealed environment")
    tag = (expected_sha256 or sha256_file(source))[:16]
    name = f"_r10_fallback_numeric_core_{tag}"
    incumbent = sys.modules.get(name)
    if incumbent is not None:
        return incumbent
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import numeric core {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    missing = [a for a in _REQUIRED_CORE_ATTRIBUTES if not hasattr(module, a)]
    if missing:
        raise ImportError(f"numeric core missing required functions: {missing}")
    return module


def verify_sealed_extension(
    core: ModuleType,
    *,
    expected_path: str | Path | None,
    expected_sha256: str | None,
    verify_files: bool = True,
) -> str:
    """Verify the extension the numeric core actually binds to.

    DISCLOSED DEVIATION from the ancestor: Brandon's loader imports the
    sealed ``exllamav3_ext`` binary by path.  The fallback numeric core is a
    shim over the installed exllamav3 package, whose own loader binds the
    extension (a prebuilt wheel module or a torch cpp_extension build).
    Loading a second copy of the same binary would double-register pybind
    ops, so instead we hash-verify the binary the core RESOLVED and fail
    closed on any mismatch with the sealed path/hash.  The verified file
    hash is returned for identity metadata either way.
    """

    _torch, extension = core._lazy_torch()
    loaded = Path(str(getattr(extension, "__file__", ""))).resolve()
    if not loaded.is_file():
        raise RuntimeError(
            f"cannot resolve the trellis extension binary from {extension!r}"
        )
    loaded_sha = sha256_file(loaded)
    if verify_files:
        if expected_path is None or not expected_sha256:
            raise ValueError("codec requires a sealed TRELLIS extension path/hash")
        if Path(expected_path).resolve() != loaded:
            raise RuntimeError(
                "ambient trellis extension path differs from the sealed binary: "
                f"{loaded} != {Path(expected_path).resolve()}"
            )
        if loaded_sha != expected_sha256:
            raise RuntimeError("ambient trellis extension differs from the sealed binary")
    return loaded_sha


def resolve_ambient_extension() -> Path:
    """Stage-time helper: path of the extension binary exllamav3 loads."""

    import torch  # noqa: F401  (extension requires torch first)
    from exllamav3.ext import exllamav3_ext as extension

    return Path(str(extension.__file__)).resolve()


# --------------------------------------------------------------------------
# CUDA-only LDL factorization — verbatim ancestor math
# (bmmlaw_r7_encoder/trellis.py cuda_only_block_ldl).
# --------------------------------------------------------------------------


def cuda_only_block_ldl(h, block: int, quant_args: Mapping[str, Any]):
    import torch

    if h.device.type != "cuda":
        raise ValueError("production LDL factorization is pinned to CUDA")
    size = int(h.shape[0])
    if size % block:
        raise ValueError("LDL dimension is not divisible by its block")
    blocks = size // block
    retries = 0
    while True:
        try:
            factor = torch.linalg.cholesky(h)
            proxy_h = h.cpu()
            h.copy_(factor)
            factor = h
            h = proxy_h
            break
        except torch._C._LinAlgError:
            retries += 1
            if retries > 10:
                raise
            h.diagonal().add_(
                2.0
                * float(quant_args.get("sigma_reg", DEFAULT_SIGMA_REG))
                * h.diagonal().mean()
            )
        except Exception as exc:
            if (
                exc.__class__.__name__ == "OutOfMemoryError"
                or "out of memory" in str(exc).lower()
            ):
                raise RuntimeError(
                    "CUDA LDL out of memory; CPU fallback is forbidden by the sealed recipe"
                ) from exc
            raise
    diagonal_blocks = torch.diagonal(
        factor.reshape(blocks, block, blocks, block), dim1=0, dim2=2
    ).permute(2, 0, 1)
    inverse = torch.linalg.inv(diagonal_blocks)
    factor = factor.view(size, blocks, block)
    for index in range(blocks):
        factor[:, index, :] = factor[:, index, :] @ inverse[index, :, :]
    factor = factor.reshape(size, size).contiguous()
    blocked = factor.view(blocks, block, blocks, block).permute(0, 2, 1, 3)
    indices = torch.arange(blocks)
    blocked[indices, indices] = torch.stack(
        [torch.eye(block, device=factor.device, dtype=h.dtype)] * blocks
    )
    return factor, h


# --------------------------------------------------------------------------
# Result record.  Field names are load-bearing receipts: the adapter reads
# .trellis/.reconstructed_kn/.suh/.svh/.packed_sha256/.reconstruction_sha256/
# .provenance; glm53_prepared_backend additionally reads .reconstructed_kn
# (moved to device) and .reconstruction_sha256.  Unlike the ancestor's
# EncodedTensor, bits are NOT restricted to (3,4,5): the kernels instantiate
# K=1..8 and the K6/K8 program needs 6 and 8 (inference, disclosed).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodedUnit:
    tensor_id: Any
    bits: int
    trellis: Any
    suh: Any
    svh: Any
    reconstructed_kn: Any
    proxy_loss: float
    packed_sha256: str
    reconstruction_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bits not in KERNEL_BITS:
            raise ValueError(f"bits={self.bits} outside kernel range {KERNEL_BITS}")


class R10TrellisCodec:
    """Reconstructed multi-rate EXL3/MCG trellis codec.

    Receipt-anchored surface (every method/attribute below has a named
    public caller):
      core, codebook_scale        glm53_mcg_preparation, StreamingLayerFitter,
                                  candidates/ledger.py:2647
      _quant_args                 qwen_services.CorrectedPinnedGSSProducer
      encode_bits                 quant_pipeline.codecs.exl3_mcg adapter
      encode_group                glm53_prepared_backend._encode_batch,
                                  run_qwen_fast_encode, encode_qwen_attention_k4
      cache_stats, clear_caches   glm53_prepared_backend._encode_batch
      decode_to_original          ancestor parity (kept for probes/replay)
    """

    def __init__(self, config: CodecConfig | Any = None) -> None:
        if config is None:
            config = CodecConfig()
        # Duck-typed on purpose: the adapter builds the config from the staged
        # r7_encoder.trellis module, which re-exports this class's CodecConfig,
        # but any object exposing the same attributes works.
        self.config = config
        self._core: ModuleType | None = None
        self._extension_sha256: str | None = None
        self._factor_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._factor_cache_limit = int(os.environ.get("R10_FALLBACK_FACTOR_CACHE", "4"))
        self.cache_stats: dict[str, int] = {
            "factor_hits": 0,
            "factor_misses": 0,
            "factor_evictions": 0,
            "cache_clears": 0,
        }

    # -- sealed loading ----------------------------------------------------

    @property
    def core(self) -> ModuleType:
        if self._core is None:
            verify = bool(getattr(self.config, "verify_files", True))
            core = load_numeric_core(
                self.config.numeric_core,
                expected_sha256=getattr(self.config, "numeric_core_sha256", None),
                verify_files=verify,
            )
            self._extension_sha256 = verify_sealed_extension(
                core,
                expected_path=getattr(self.config, "extension", None),
                expected_sha256=getattr(self.config, "extension_sha256", None),
                verify_files=verify,
            )
            self._core = core
        return self._core

    @property
    def codebook_scale(self) -> float:
        value = float(self.core.CODEBOOK_SCALE)
        if not math.isfinite(value) or value == 0:
            raise ValueError("numeric core exposes an invalid MCG codebook scale")
        return value

    # -- caches ------------------------------------------------------------

    def clear_caches(self) -> None:
        """Drop factor state so later covariances get a fresh factor domain."""

        self._factor_cache.clear()
        self.cache_stats["cache_clears"] += 1

    # -- primitive argument plumbing --------------------------------------

    def _quant_args(self, bits: int, sigma_reg: float) -> dict[str, Any]:
        """quant_args mapping consumed by the v31 primitives (ancestor
        _factor_covariance literal; also fed verbatim to core.g_scale_gss by
        CorrectedPinnedGSSProducer)."""

        bits = int(bits)
        if bits not in KERNEL_BITS:
            raise ValueError(f"K={bits} outside the instantiated kernel range 1..8")
        return {
            "K": bits,
            "devices": [int(str(self.config.device).split(":")[-1])],
            "mcg": True,
            "sigma_reg": float(sigma_reg),
            "buf_size_k": 128,
        }

    # -- ancestor numeric steps (verbatim math) ----------------------------

    def _validate_vectors(self, suh, svh, k: int, n: int):
        import torch

        su = torch.as_tensor(suh, dtype=torch.float32, device=self.config.device).flatten()
        sv = torch.as_tensor(svh, dtype=torch.float32, device=self.config.device).flatten()
        if tuple(su.shape) != (k,) or tuple(sv.shape) != (n,):
            raise ValueError(
                f"rotation shape mismatch: suh={tuple(su.shape)} vs {k}, "
                f"svh={tuple(sv.shape)} vs {n}"
            )
        if not torch.isfinite(su).all() or not torch.isfinite(sv).all():
            raise ValueError("rotation vectors must be finite")
        if (su == 0).any() or (sv == 0).any():
            raise ValueError("zero rotation/scale entry is not invertible")
        # FP16 storage boundary: validate after rounding and encode with the
        # rounded values so encode and serve cannot disagree.
        su_stored = su.half()
        sv_stored = sv.half()
        if not torch.isfinite(su_stored).all() or not torch.isfinite(sv_stored).all():
            raise ValueError("rotation vector overflows its stored FP16 representation")
        if (su_stored == 0).any() or (sv_stored == 0).any():
            raise ValueError("rotation vector underflows its stored FP16 representation")
        return su_stored.float(), sv_stored.float()

    def _regularize_weight(self, weight_kn, su, sv):
        weight = weight_kn.clone().to(dtype=weight_kn.dtype)
        weight.div_(sv.unsqueeze(0))
        self.core.blockwise_preapply_had_r_(weight, HAD_N)
        weight.div_(su.unsqueeze(1))
        self.core.blockwise_preapply_had_l_(weight, HAD_K)
        return weight

    def _prepare_factor(self, covariance, su, sigma_reg: float):
        """Symmetrize+damp+transform the metric, then block-LDL factor it.

        Content-addressed cache: gate and up projections of one expert share
        one covariance (glm53_prepared_backend builds both requests from a
        single gate_up_hessian), so the factor is reused whenever the
        (covariance bytes, stored-fp16 su bytes, sigma_reg) triple repeats.
        Returns (factor, covariance_sha256).
        """

        import torch

        covariance_sha = _tensor_sha256(torch.as_tensor(covariance, dtype=torch.float32))
        key = (covariance_sha, _tensor_sha256(su.half()), repr(float(sigma_reg)))
        cached = self._factor_cache.get(key)
        if cached is not None:
            self._factor_cache.move_to_end(key)
            self.cache_stats["factor_hits"] += 1
            return cached, covariance_sha
        self.cache_stats["factor_misses"] += 1

        h = torch.as_tensor(covariance, dtype=torch.float32, device=self.config.device).clone()
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] != su.numel():
            raise ValueError("full covariance shape does not match tensor K")
        h = (h + h.T) * 0.5
        diagonal = h.diagonal()
        mean = diagonal.mean()
        if not torch.isfinite(h).all() or not math.isfinite(float(mean.item())):
            raise ValueError("non-finite covariance")
        if float(mean.item()) <= 1e-20:
            raise ValueError("degenerate covariance; identity-H fallback is forbidden")
        diagonal.add_(float(sigma_reg) * mean)

        # Transform the metric by the exact stored input-side vector (this is
        # the covariance matching the inverse weight transform).
        h.mul_(su.unsqueeze(0))
        self.core.blockwise_preapply_had_r_(h, HAD_K)
        h.mul_(su.unsqueeze(1))
        self.core.blockwise_preapply_had_l_(h, HAD_K)

        policy = getattr(self.config, "factorization_policy", LDL_FACTORIZATION_POLICY)
        if policy != LDL_FACTORIZATION_POLICY:
            raise ValueError("LDL factorization policy differs from the sealed recipe")
        factor, _proxy_h = cuda_only_block_ldl(
            h, 16, {"sigma_reg": float(sigma_reg)}
        )
        indices = torch.arange(factor.shape[0], device=factor.device)
        factor[indices, indices] = 0

        self._factor_cache[key] = factor
        while len(self._factor_cache) > self._factor_cache_limit:
            self._factor_cache.popitem(last=False)
            self.cache_stats["factor_evictions"] += 1
        return factor, covariance_sha

    def _decode_regularized(self, packed, k: int, n: int, bits: int):
        torch, extension = self.core._lazy_torch()
        decoded = torch.empty((k, n), dtype=torch.float16, device=packed.device)
        extension.reconstruct(decoded, packed, bits, True, False)
        return decoded

    def decode_to_original(self, packed, suh, svh, bits: int):
        """Original-domain [K, N] reconstruction of a packed trellis.

        RECEIPT (re-review 2026-08-27): the pipeline's reader ABI demands the
        encoder's reconstruction closure be BIT-EXACT against the independent
        `glm53_packed_k4_reader.decode_choice_hf` CPU decode (the driver's
        rehearse/reader-ABI receipts hash-compare them; upstream's receipts
        claim `offline_reader_exact_decode_checked`).  The extension's fp16
        `reconstruct` kernel differs from that fp32 torch math at rounding
        scale, so whenever the pipeline is importable the closure is computed
        WITH the reader's own decode — exactness by construction.  The
        extension-based path remains only for standalone runs without the
        pipeline on sys.path and is recorded as such in provenance.
        """

        import torch

        try:
            from quant_pipeline.evaluation import glm53_packed_k4_reader as _reader

            decode_choice_hf = (
                _reader.decode_choice_hf
                if int(bits) in _reader.SUPPORTED_BITS
                else None  # rate outside the pinned reader (K3/K5 oracles; K8
                # until the K6K8 reader extension lands) -> extension path
            )
        except ImportError:
            decode_choice_hf = None
        if decode_choice_hf is not None:
            decoded_nk = decode_choice_hf(
                torch.as_tensor(packed).detach().cpu(),
                torch.as_tensor(suh).detach().cpu().half(),
                torch.as_tensor(svh).detach().cpu().half(),
                bits=bits,
            )
            self._original_domain_decode = "reader_exact_cpu_fp32"
            device = getattr(packed, "device", decoded_nk.device)
            return decoded_nk.T.contiguous().to(device)
        self._original_domain_decode = "extension_fp16_approx_reader_unavailable_or_rate_unsupported"
        k = int(torch.as_tensor(suh).numel())
        n = int(torch.as_tensor(svh).numel())
        regularized = self._decode_regularized(packed, k, n, bits).float()
        su = torch.as_tensor(suh, device=regularized.device, dtype=torch.float32).flatten()
        sv = torch.as_tensor(svh, device=regularized.device, dtype=torch.float32).flatten()
        decoded = self.core.preapply_had_l(regularized, HAD_K)
        decoded.mul_(su.unsqueeze(1))
        decoded = self.core.preapply_had_r(decoded, HAD_N)
        decoded.mul_(sv.unsqueeze(0))
        return decoded

    # -- shared finalization (oracles + hashes + record) -------------------

    def _finalize_unit(
        self,
        *,
        tensor_id,
        bits: int,
        weight_kn,
        covariance,
        covariance_sha256: str,
        su,
        sv,
        encoded,
        quant_args: Mapping[str, Any],
        quantized_regularized,
        sigma_reg: float,
        provenance: Mapping[str, Any] | None,
    ) -> EncodedUnit:
        import torch

        k, n = int(tensor_id.k), int(tensor_id.n)
        packed = self.core.pack_trellis(encoded, dict(quant_args))

        torch_module, extension = self.core._lazy_torch()
        unpacked = torch_module.zeros_like(encoded)
        extension.unpack_trellis(unpacked, packed, bits)
        if not torch_module.equal(unpacked, encoded):
            raise AssertionError("TRELLIS pack/unpack index mismatch")
        extension_decoded = self._decode_regularized(packed, k, n, bits)
        if not torch_module.equal(extension_decoded, quantized_regularized.half()):
            raise AssertionError("extension reconstruction differs from LDLQ values")
        repeated_decoded = self._decode_regularized(packed, k, n, bits)
        if not torch_module.equal(repeated_decoded, extension_decoded):
            raise AssertionError("TRELLIS extension reconstruction is not byte deterministic")
        expected_bytes = k * n * bits // 8
        if packed.numel() * packed.element_size() != expected_bytes:
            raise AssertionError("packed TRELLIS byte count disagrees with integer bits")

        reconstructed = self.decode_to_original(packed, su.half(), sv.half(), bits)
        error = reconstructed.double() - weight_kn.double()
        covariance_f64 = torch.as_tensor(covariance, device=error.device, dtype=torch.float64)
        numerator = torch.einsum("kn,kl,ln->", error, covariance_f64, error)
        denominator = torch.einsum(
            "kn,kl,ln->", weight_kn.double(), covariance_f64, weight_kn.double()
        ).clamp_min(1e-30)
        proxy_loss = float((numerator / denominator).item())

        metadata = dict(provenance or {})
        metadata.update(
            {
                "numeric_core": str(self.config.numeric_core),
                "full_k": k,
                "full_n": n,
                "mcg": f"0x{MCG_MULT:08X}",
                "codebook_scale": self.codebook_scale,
                "sigma_reg": float(sigma_reg),
                "factorization_policy": getattr(
                    self.config, "factorization_policy", LDL_FACTORIZATION_POLICY
                ),
                "extension_sha256": self._extension_sha256
                or getattr(self.config, "extension_sha256", None),
                "extension_repeat_oracle": True,
                "original_domain_decode": getattr(
                    self, "_original_domain_decode", "unknown"
                ),
                "covariance_sha256": covariance_sha256,
                # MANDATORY substitution disclosure — must survive into every
                # published receipt built from this candidate.
                "fallback_reconstruction": {
                    "schema": RECONSTRUCTION_SCHEMA,
                    "reconstructed_from": list(RECONSTRUCTION_SOURCES),
                    "sealed_core_files_absent": [
                        "r7_encoder/r10_codec.py",
                        "encode_tr3_v31.py",
                    ],
                    "bit_identity_with_sealed_core": "UNVERIFIED",
                },
            }
        )
        # Device residency receipt: run_qwen_fast_encode subtracts
        # candidate.reconstructed_kn (as bf16) from a GPU-resident source and
        # calls candidate.trellis.detach().cpu() — both stay on the encode
        # device here (deviation from the CPU-resident ancestor EncodedTensor).
        return EncodedUnit(
            tensor_id=tensor_id,
            bits=bits,
            trellis=packed.detach(),
            suh=su.half().detach(),
            svh=sv.half().detach(),
            reconstructed_kn=reconstructed.detach(),
            proxy_loss=proxy_loss,
            packed_sha256=_tensor_sha256(packed),
            reconstruction_sha256=_tensor_sha256(reconstructed.half()),
            provenance=metadata,
        )

    # -- public encode surface --------------------------------------------

    @staticmethod
    def _check_bits(bits) -> tuple[int, ...]:
        values = tuple(int(bit) for bit in bits)
        if not values:
            raise ValueError("bits tuple must be nonempty")
        if len(set(values)) != len(values):
            raise ValueError("bits tuple must not repeat rates")
        for bit in values:
            if bit not in KERNEL_BITS:
                raise ValueError(f"K={bit} outside the instantiated kernel range 1..8")
        return values

    def encode_bits(
        self,
        *,
        tensor_id,
        weight_hf,
        covariance,
        bits: Sequence[int],
        suh,
        svh,
        sigma_reg: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[int, EncodedUnit]:
        """Encode one tensor at each requested rate; factorize once.

        ``tensor_id`` is duck-typed (.key/.k/.n as in the adapter's
        GenericTensorId).  ``weight_hf`` is HF-layout [N, K].  Returns a dict
        preserving the input bit order (the adapter test asserts the key
        tuple equals the request tuple).
        """

        import torch

        values = self._check_bits(bits)
        effective_sigma = (
            float(self.config.sigma_reg) if sigma_reg is None else float(sigma_reg)
        )
        weight = torch.as_tensor(weight_hf, device=self.config.device, dtype=torch.float32)
        k, n = int(tensor_id.k), int(tensor_id.n)
        if tuple(weight.shape) != (n, k):
            raise ValueError(
                f"{tensor_id.key}: expected HF [N,K]={(n, k)}, got {tuple(weight.shape)}"
            )
        if k % HAD_K or n % HAD_N:
            raise ValueError(
                f"EXL3/MCG numeric core requires K and N divisible by 128; "
                f"{tensor_id.key} has [N,K]={(n, k)}"
            )
        weight_kn = weight.T.contiguous()
        su, sv = self._validate_vectors(suh, svh, k, n)
        regularized = self._regularize_weight(weight_kn, su, sv)
        factor, covariance_sha = self._prepare_factor(covariance, su, effective_sigma)

        result: dict[int, EncodedUnit] = {}
        for bit in values:
            quant_args = self._quant_args(bit, effective_sigma)
            quantized_regularized, encoded = self.core.ldlq(regularized, factor, quant_args)
            result[bit] = self._finalize_unit(
                tensor_id=tensor_id,
                bits=bit,
                weight_kn=weight_kn,
                covariance=covariance,
                covariance_sha256=covariance_sha,
                su=su,
                sv=sv,
                encoded=encoded,
                quant_args=quant_args,
                quantized_regularized=quantized_regularized,
                sigma_reg=effective_sigma,
                provenance=provenance,
            )
        return result

    # Ancestor-compatible single-rate entry point (kept for probes/parity).
    def encode(self, *, tensor_id, weight_hf, covariance, bits: int, suh, svh,
               sigma_reg: float | None = None, provenance=None) -> EncodedUnit:
        return self.encode_bits(
            tensor_id=tensor_id, weight_hf=weight_hf, covariance=covariance,
            bits=(int(bits),), suh=suh, svh=svh, sigma_reg=sigma_reg,
            provenance=provenance,
        )[int(bits)]

    # -- grouped encoding --------------------------------------------------

    def encode_group(
        self, requests: Sequence[Mapping[str, Any]], *, lockstep: bool | None = None
    ) -> list[dict[int, EncodedUnit]]:
        """Encode a batch of requests; order-preserving list of encode_bits
        results.  Receipts: glm53_prepared_backend._encode_batch and
        run_qwen_fast_encode ('prepares/factorizes each matrix once and
        locksteps all equal-bit LDLQ walks').

        The lockstep path batches only the quantize_tiles kernel launches
        across matrices — per-tile Viterbi is stateless across tiles, so the
        results are bit-identical to the serial path (asserted by the
        self-test, and re-checkable per group via
        R10_FALLBACK_VERIFY_LOCKSTEP=1).  Serial execution is the reference
        semantics; force it with R10_FALLBACK_LOCKSTEP=0.
        """

        requests = [dict(request) for request in requests]
        if not requests:
            return []
        if lockstep is None:
            lockstep = os.environ.get("R10_FALLBACK_LOCKSTEP", "1") != "0"
        first_bits = self._check_bits(requests[0]["bits"])
        homogeneous = (
            len(requests) > 1
            and len(first_bits) == 1
            and all(
                tuple(int(b) for b in request["bits"]) == first_bits
                and int(request["tensor_id"].k) == int(requests[0]["tensor_id"].k)
                and int(request["tensor_id"].n) == int(requests[0]["tensor_id"].n)
                for request in requests
            )
            and str(self.config.device).startswith("cuda")
        )
        if not (lockstep and homogeneous):
            return [self.encode_bits(**request) for request in requests]
        results = self._encode_group_lockstep(requests, first_bits[0])
        if os.environ.get("R10_FALLBACK_VERIFY_LOCKSTEP", "0") == "1":
            reference = self.encode_bits(**requests[0])
            bit = first_bits[0]
            if (
                reference[bit].packed_sha256 != results[0][bit].packed_sha256
                or reference[bit].reconstruction_sha256
                != results[0][bit].reconstruction_sha256
            ):
                raise AssertionError("lockstep group encode diverged from serial encode")
        return results

    def _encode_group_lockstep(
        self, requests: list[dict[str, Any]], bit: int
    ) -> list[dict[int, EncodedUnit]]:
        import torch

        core = self.core
        device = torch.device(self.config.device)

        prepared = []
        for request in requests:
            tensor_id = request["tensor_id"]
            k, n = int(tensor_id.k), int(tensor_id.n)
            weight = torch.as_tensor(
                request["weight_hf"], device=self.config.device, dtype=torch.float32
            )
            if tuple(weight.shape) != (n, k):
                raise ValueError(
                    f"{tensor_id.key}: expected HF [N,K]={(n, k)}, got {tuple(weight.shape)}"
                )
            if k % HAD_K or n % HAD_N:
                raise ValueError(
                    f"{tensor_id.key}: K and N must be divisible by 128"
                )
            sigma = request.get("sigma_reg")
            effective_sigma = (
                float(self.config.sigma_reg) if sigma is None else float(sigma)
            )
            weight_kn = weight.T.contiguous()
            su, sv = self._validate_vectors(request["suh"], request["svh"], k, n)
            regularized = self._regularize_weight(weight_kn, su, sv)
            factor, covariance_sha = self._prepare_factor(
                request["covariance"], su, effective_sigma
            )
            prepared.append(
                {
                    "tensor_id": tensor_id,
                    "weight_kn": weight_kn,
                    "covariance": request["covariance"],
                    "covariance_sha": covariance_sha,
                    "su": su,
                    "sv": sv,
                    "regularized": regularized,
                    "factor": factor,
                    "sigma": effective_sigma,
                    "provenance": request.get("provenance"),
                }
            )

        # Joint LDLQ walk — an exact transliteration of the pinned
        # exl3_lib.quantize.ldlq loop, with the quantize_tiles launch batched
        # over matrices.  All per-matrix linear algebra stays per-matrix.
        size_k = int(prepared[0]["tensor_id"].k)
        size_n = int(prepared[0]["tensor_id"].n)
        tiles_n = size_n // 16
        buf_size_k = 128
        perm = core.tensor_core_perm(device)
        perm_i = core.tensor_core_perm_i(device)
        quant_args = self._quant_args(bit, prepared[0]["sigma"])

        state = []
        for item in prepared:
            state.append(
                {
                    "prod_cache": torch.zeros((size_k, size_n), dtype=torch.float, device=device),
                    "weight_q": torch.zeros((size_k, size_n), dtype=torch.float, device=device),
                    "encoded": torch.zeros(
                        (size_k // 16, tiles_n, 256), dtype=torch.short, device=device
                    ),
                }
            )

        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16
                tile_batches = []
                for item, st in zip(prepared, state):
                    b_weight = item["regularized"][i:j]
                    b_weight_q = st["weight_q"][i:j]
                    b_L = item["factor"][i:j]
                    bb_err = b_weight[bj:] - b_weight_q[bj:]
                    bb_L = b_L[bj:, i + bi : i + bj]
                    compensation_term = st["prod_cache"][i:j][bi:bj]
                    compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
                    rows = b_weight[bi:bj] + compensation_term
                    tiles = (
                        rows.reshape(16, tiles_n, 16)
                        .permute(1, 0, 2)
                        .reshape(tiles_n, 256)
                    )
                    tile_batches.append(tiles[:, perm])
                stacked = torch.cat(tile_batches, dim=0)
                quant_w_all, quant_i_all = core.quantize_tiles(stacked, quant_args)
                for index, (item, st) in enumerate(zip(prepared, state)):
                    quant_w = quant_w_all[index * tiles_n : (index + 1) * tiles_n]
                    quant_i = quant_i_all[index * tiles_n : (index + 1) * tiles_n]
                    quant_w = quant_w[:, perm_i]
                    quant_w = (
                        quant_w.reshape(tiles_n, 16, 16)
                        .permute(1, 0, 2)
                        .reshape(16, size_n)
                    )
                    st["weight_q"][i:j][bi:bj] = quant_w
                    st["encoded"][i // 16 : j // 16][bi // 16 : bj // 16] = quant_i.unsqueeze(0)
            for item, st in zip(prepared, state):
                b_weight = item["regularized"][i:j]
                b_err = b_weight - st["weight_q"][i:j]
                st["prod_cache"].addmm_(item["factor"][i:j].T, b_err, alpha=1.0, beta=1.0)

        results: list[dict[int, EncodedUnit]] = []
        for item, st in zip(prepared, state):
            unit = self._finalize_unit(
                tensor_id=item["tensor_id"],
                bits=bit,
                weight_kn=item["weight_kn"],
                covariance=item["covariance"],
                covariance_sha256=item["covariance_sha"],
                su=item["su"],
                sv=item["sv"],
                encoded=st["encoded"],
                quant_args=quant_args,
                quantized_regularized=st["weight_q"],
                sigma_reg=item["sigma"],
                provenance=item["provenance"],
            )
            results.append({bit: unit})
        return results


# --------------------------------------------------------------------------
# Fallback numeric core shim.  Staged as the config.numeric_core file (the
# slot Brandon's encode_tr3_v31.py occupies).  It re-exports the pinned
# exllamav3 primitives, which carry the exact names the ancestor loader
# requires — receipt that v31 is an exllamav3-quantize derivative — and adds
# the two v31-only symbols (_lazy_torch, CODEBOOK_SCALE) plus the v31
# g_scale_gss argument order (core.g_scale_gss(target, quant_args)).
# --------------------------------------------------------------------------

NUMERIC_CORE_SHIM_SOURCE = '''\
"""encode_tr3_fallback.py — RECONSTRUCTED numeric core (NOT encode_tr3_v31.py).

Thin sealed shim over the pinned exllamav3 exl3_lib.quantize primitives.
Receipts published from this file must disclose the substitution.
"""

from exllamav3.modules.quant.exl3_lib.quantize import (  # exllamav3 @ c5d9c657
    block_ldl,
    block_rms,
    block_rms_n,
    blockwise_preapply_had_l_,
    blockwise_preapply_had_r_,
    codebook_scale as CODEBOOK_SCALE,
    ldlq,
    pack_signs,
    pack_trellis,
    preapply_had_l,
    preapply_had_r,
    quantize_tiles,
    quantize_tiles_multigpu,
    tensor_core_perm,
    tensor_core_perm_i,
)
from exllamav3.modules.quant.exl3_lib import quantize as _quantize

FALLBACK_RECONSTRUCTION = "k6-program.r10-fallback-reconstruction.v1"


def _lazy_torch():
    import torch
    from exllamav3.ext import exllamav3_ext

    return torch, exllamav3_ext


def g_scale_gss(target, quant_args, verbose=False, width=3, pb=None):
    """v31 argument order (receipt: CorrectedPinnedGSSProducer calls
    core.g_scale_gss(request.target, quant_args) and records 13 golden-
    section evaluations at width 3 — exactly what the pinned exllamav3
    implementation performs over [0.1, 1.9] at tol 0.01)."""

    return _quantize.g_scale_gss(target, verbose, quant_args, width=width, pb=pb)
'''


def write_numeric_core_shim(path: str | Path) -> str:
    """Write the shim and return its SHA-256 (for CodecConfig sealing)."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(NUMERIC_CORE_SHIM_SOURCE, encoding="utf-8")
    return sha256_file(destination)


# --------------------------------------------------------------------------
# Staging: materialize a reconstructed r7_encoder package the adapter can
# import (r7_encoder.r10_codec + r7_encoder.trellis + r7_encoder.hessian).
# --------------------------------------------------------------------------

_TRELLIS_REEXPORT = '''\
"""RECONSTRUCTED r7_encoder.trellis — re-exports the fallback codec surface.

The public adapter resolves CodecConfig from this module and hash-binds
every file in the package; these hashes will NOT match Brandon's sealed
closure and receipts must say so.
"""

from .r10_codec import (  # noqa: F401
    CodecConfig,
    EncodedUnit,
    R10TrellisCodec,
    cuda_only_block_ldl,
    load_numeric_core,
    verify_sealed_extension,
)

Exl3TrellisCodec = R10TrellisCodec  # ancestor-name compatibility
'''

_INIT_SOURCE = '''\
"""RECONSTRUCTED r7_encoder package (fallback; sealed closure unavailable).

Contents: r10_codec.py (reconstructed), trellis.py (re-export shim),
hessian.py (copied verbatim from bmmlaw_r7_encoder ancestor).  See
RECONSTRUCTION.md in the k6-program tree for the inference catalogue.
"""

RECONSTRUCTED_FALLBACK = "k6-program.r10-fallback-reconstruction.v1"
'''


def stage_r7_encoder(
    destination: str | Path,
    *,
    ancestors_dir: str | Path | None = None,
) -> dict[str, str]:
    """Create <destination>/r7_encoder/ and return {relpath: sha256}.

    ``ancestors_dir`` must point at the bmmlaw_r7_encoder sparse checkout so
    hessian.py can be copied verbatim (required by
    glm53_prepared_backend's ``r7_encoder.hessian`` imports).  Without it a
    package is still staged, but the prepared-backend path will fail closed.
    """

    root = Path(destination) / "r7_encoder"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text(_INIT_SOURCE, encoding="utf-8")
    (root / "r10_codec.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "trellis.py").write_text(_TRELLIS_REEXPORT, encoding="utf-8")
    # Numeric core: the file CodecConfig(numeric_core=...) loads and attribute-
    # checks.  Upstream this is encode_tr3_v31.py (absent); the reconstruction
    # ships the exllamav3-backed shim so drivers can default to
    # r7_encoder/encode_tr3_fallback.py.
    write_numeric_core_shim(root / "encode_tr3_fallback.py")
    if ancestors_dir is not None:
        hessian = Path(ancestors_dir) / "hessian.py"
        (root / "hessian.py").write_text(
            hessian.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*.py"))
    }


# --------------------------------------------------------------------------
# Self-test (see VALIDATION.md).  Requires CUDA + the pinned exllamav3.
#   python r10_codec_reconstructed.py --selftest [--device cuda:0]
#     [--pipeline-src /path/to/brandon-drivers/src] [--k 256] [--n 256]
# --------------------------------------------------------------------------


def _selftest(device: str, pipeline_src: str | None, k: int, n: int) -> int:
    import json
    import tempfile

    import torch

    report: dict[str, Any] = {"schema": "k6-program.r10-fallback-selftest.v1"}
    staging = Path(tempfile.mkdtemp(prefix="r10-fallback-selftest-"))
    core_path = staging / "encode_tr3_fallback.py"
    core_sha = write_numeric_core_shim(core_path)
    extension_path = resolve_ambient_extension()
    extension_sha = sha256_file(extension_path)
    config = CodecConfig(
        device=device,
        sigma_reg=DEFAULT_SIGMA_REG,
        numeric_core=core_path,
        numeric_core_sha256=core_sha,
        extension=extension_path,
        extension_sha256=extension_sha,
        verify_files=True,
    )
    codec = R10TrellisCodec(config)
    report["codebook_scale"] = codec.codebook_scale
    assert abs(codec.codebook_scale - EXPECTED_CODEBOOK_SCALE) < 1e-9

    @dataclass(frozen=True)
    class _Tid:
        key: str
        k: int
        n: int

    torch.manual_seed(20260827)
    weight_hf = torch.randn(n, k, dtype=torch.float32)  # HF [N, K]
    basis = torch.randn(k, 4 * k, dtype=torch.float32)
    covariance = (basis @ basis.T / (4 * k)) + 0.05 * torch.eye(k)
    suh = ((torch.randn(k).sign() + 1e-5).sign() * (0.5 + torch.rand(k))).float()
    svh = ((torch.randn(n).sign() + 1e-5).sign() * (0.5 + torch.rand(n))).float()
    tid = _Tid(key=f"selftest[{n}x{k}]", k=k, n=n)

    bits = (3, 4, 5, 6, 8)
    encoded = codec.encode_bits(
        tensor_id=tid, weight_hf=weight_hf, covariance=covariance,
        bits=bits, suh=suh, svh=svh,
    )
    assert tuple(encoded) == bits
    nmse = {}
    for bit, unit in encoded.items():
        words = 256 * bit // 16
        assert tuple(unit.trellis.shape) == (k // 16, n // 16, words), (
            bit, tuple(unit.trellis.shape))
        assert unit.trellis.numel() * unit.trellis.element_size() == k * n * bit // 8
        replay = codec.decode_to_original(unit.trellis, unit.suh, unit.svh, bit)
        assert torch.equal(replay.half(), unit.reconstructed_kn.half())
        weight_kn = weight_hf.T.to(replay.device)
        nmse[bit] = float(
            ((replay - weight_kn).square().sum() / weight_kn.square().sum()).item()
        )
    report["trellis_words_per_tile"] = {b: 256 * b // 16 for b in bits}
    report["k8_is_128_word_trellis"] = report["trellis_words_per_tile"][8] == 128
    report["nmse_by_bits"] = nmse
    assert nmse[8] < nmse[6] < nmse[4] < nmse[3], nmse
    report["proxy_loss_by_bits"] = {b: encoded[b].proxy_loss for b in bits}

    # Repeat determinism across a fresh codec instance.
    codec_b = R10TrellisCodec(config)
    encoded_b = codec_b.encode_bits(
        tensor_id=tid, weight_hf=weight_hf, covariance=covariance,
        bits=bits, suh=suh, svh=svh,
    )
    for bit in bits:
        assert encoded[bit].packed_sha256 == encoded_b[bit].packed_sha256, bit
        assert encoded[bit].reconstruction_sha256 == encoded_b[bit].reconstruction_sha256
    report["repeat_determinism"] = True

    # Grouped encode: serial vs lockstep bit-identity, for K6 and for K8.
    for bit in (6, 8):
        requests = []
        for index in range(4):
            w = torch.randn(n, k, dtype=torch.float32)
            s_u = ((torch.randn(k).sign() + 1e-5).sign() * (0.5 + torch.rand(k))).float()
            s_v = ((torch.randn(n).sign() + 1e-5).sign() * (0.5 + torch.rand(n))).float()
            requests.append({
                "tensor_id": _Tid(key=f"group{index}", k=k, n=n),
                "weight_hf": w,
                "covariance": covariance if index < 2 else covariance * 1.5,
                "bits": (bit,),
                "suh": s_u,
                "svh": s_v,
                "sigma_reg": DEFAULT_SIGMA_REG,
                "provenance": {"selftest_group_index": index},
            })
        serial = codec.encode_group(requests, lockstep=False)
        codec.clear_caches()
        locked = codec.encode_group(requests, lockstep=True)
        for s_result, l_result in zip(serial, locked):
            assert s_result[bit].packed_sha256 == l_result[bit].packed_sha256
            assert s_result[bit].reconstruction_sha256 == l_result[bit].reconstruction_sha256
        report[f"lockstep_equals_serial_k{bit}"] = True
    report["cache_stats"] = dict(codec.cache_stats)

    # Factor-cache reuse: two requests sharing (covariance, suh, sigma) must
    # produce one miss + one hit and identical results either way.
    codec.clear_caches()
    hits_before = codec.cache_stats["factor_hits"]
    shared_su = suh
    r_gate = dict(requests[0], suh=shared_su, covariance=covariance)
    r_up = dict(requests[1], suh=shared_su, covariance=covariance)
    codec.encode_group([r_gate, r_up], lockstep=False)
    assert codec.cache_stats["factor_hits"] == hits_before + 1
    report["factor_cache_hit_for_shared_hessian"] = True

    # Full adapter-contract integration (optional; needs the pipeline src).
    if pipeline_src:
        sys.path.insert(0, str(Path(pipeline_src).resolve()))
        from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

        package_root = staging / "closure"
        closure = stage_r7_encoder(package_root)
        report["staged_closure_sha256"] = closure
        adapter = Exl3MCGCodec(
            source_root=package_root,
            numeric_core=core_path,
            extension=extension_path,
            device=device,
            sigma_reg=DEFAULT_SIGMA_REG,
        )
        # The glm-5.3-flash-exl3-4bpw adapter admits K6; the shapleymcg copy
        # stops at K5.  Probe rather than parse.
        try:
            candidates = adapter.encode_candidates(
                unit_id="L3.E7.gate_proj",
                weight_hf=weight_hf,
                covariance=covariance,
                bits=(adapter_bits := (3, 4, 5, 6)),
                input_vector=suh,
                output_vector=svh,
                provenance={"selftest": True},
            )
        except ValueError:
            candidates = adapter.encode_candidates(
                unit_id="L3.E7.gate_proj",
                weight_hf=weight_hf,
                covariance=covariance,
                bits=(adapter_bits := (3, 4, 5)),
                input_vector=suh,
                output_vector=svh,
                provenance={"selftest": True},
            )
        assert tuple(candidates) == adapter_bits
        for bit in adapter_bits:
            candidate = candidates[bit]
            assert tuple(candidate.reconstructed.shape) == (n, k)
            expected = (
                candidate.packed.numel() * candidate.packed.element_size()
                + 2 * (k + n)  # fp16 suh + svh
            )
            assert candidate.stored_bytes == expected
            identity = candidate.metadata["codec_identity"]
            assert identity["backend_class"] == "r7_encoder.r10_codec.R10TrellisCodec"
            assert identity["sigma_reg"] == DEFAULT_SIGMA_REG
            assert set(identity["environment"]) == {
                "python", "machine", "torch", "torch_cuda", "compute_capability"
            }
            assert candidate.metadata["fallback_reconstruction"]["schema"] == RECONSTRUCTION_SCHEMA
        report["adapter_contract"] = {
            "bits": list(adapter_bits),
            "identity_schema": adapter.identity["identity_schema"],
            "closure_files": sorted(adapter.identity["python_closure_sha256"]),
        }

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pipeline-src", default=None)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    arguments = parser.parse_args()
    if not arguments.selftest:
        parser.error("nothing to do; pass --selftest")
    raise SystemExit(_selftest(arguments.device, arguments.pipeline_src, arguments.k, arguments.n))
