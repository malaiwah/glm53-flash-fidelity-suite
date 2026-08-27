"""Bit-exactness probe: fallback R10 decode vs exllamav3 NATIVE convert output.

Runs exllamav3's own quantize_exl3 (the function convert_model drives) with
mcg=True on one small matrix, then checks, per K in (6, 8):

  P1  format identity: unpack native packed trellis with the extension and
      re-pack through the fallback's pack path -> byte-identical trellis.
  P2  regularized-domain decode: extension reconstruct of the native trellis
      is byte-deterministic and equals itself across calls (canonical
      runtime dequant; the fallback decoder is this same op).
  P3  original-domain decode: fallback decode_to_original(native trellis,
      native fp16 suh/svh) vs the native returned weight_q. EXACT equality
      is NOT expected: native reconstructs with pre-rounding fp32 su/sv,
      while stored checkpoints (and the fallback, and Brandon's corrected
      R10) use the fp16-stored vectors. The probe records the max abs
      difference and asserts it stays at fp16-rounding scale. This gap is
      the documented encode/serve boundary the corrected codec closes.

This is a probe of OUR decode/pack fidelity against native convert
artifacts. It does NOT claim bit-identity with Brandon's sealed core.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r10_codec_reconstructed import (  # noqa: E402
    CodecConfig,
    R10TrellisCodec,
    resolve_ambient_extension,
    sha256_file,
    write_numeric_core_shim,
    _tensor_sha256,
)

from exllamav3.modules.quant.exl3_lib import quantize as q  # noqa: E402


def main() -> int:
    device = "cuda:0"
    k, n = 256, 256
    report = {"schema": "k6-program.r10-fallback-native-probe.v1", "k": k, "n": n}

    import tempfile

    staging = Path(tempfile.mkdtemp(prefix="r10-fallback-native-probe-"))
    core_path = staging / "encode_tr3_fallback.py"
    core_sha = write_numeric_core_shim(core_path)
    extension_path = resolve_ambient_extension()
    codec = R10TrellisCodec(
        CodecConfig(
            device=device,
            numeric_core=core_path,
            numeric_core_sha256=core_sha,
            extension=extension_path,
            extension_sha256=sha256_file(extension_path),
            verify_files=True,
        )
    )
    ext = codec.core._lazy_torch()[1]

    for bits in (6, 8):
        torch.manual_seed(777)
        weight = torch.randn(k, n, dtype=torch.float32, device=device)  # (in, out)
        rows = torch.randn(4 * k, k, dtype=torch.float32, device=device)
        h_sum = rows.T @ rows
        h_data = {"H": h_sum, "count": 4 * k, "finalized": False, "device": device}
        quant_args = {
            "K": bits,
            "devices": [device],
            "mcg": True,
            "sigma_reg": 0.025,
            "buf_size_k": 128,
            "seed": 1234,
            "apply_out_scales": None,
        }
        weight_q, proxy_err, out_tensors = q.quantize_exl3(
            weight.clone(), h_data, quant_args, return_weight_q=True
        )
        assert "mcg" in out_tensors, "native convert did not mark the MCG codebook"
        assert int(out_tensors["mcg"].view(torch.uint32).item()) == 0xCBAC1FED
        trellis = out_tensors["trellis"]
        suh = out_tensors["suh"]
        svh = out_tensors["svh"]
        words = 256 * bits // 16
        assert tuple(trellis.shape) == (k // 16, n // 16, words)

        # P1: unpack native trellis, re-pack through the fallback core path.
        unpacked = torch.zeros((k // 16, n // 16, 256), dtype=torch.short, device=trellis.device)
        ext.unpack_trellis(unpacked, trellis, bits)
        repacked = codec.core.pack_trellis(unpacked, {"K": bits})
        p1 = bool(torch.equal(repacked, trellis))
        assert p1, f"K{bits}: repack differs from native packed trellis"

        # P2: canonical regularized-domain decode, twice, byte-identical.
        d1 = codec._decode_regularized(trellis, k, n, bits)
        d2 = codec._decode_regularized(trellis, k, n, bits)
        p2 = bool(torch.equal(d1, d2))
        assert p2

        # P3: original-domain decode with the STORED fp16 vectors vs native
        # weight_q built from pre-rounding fp32 vectors.
        decoded = codec.decode_to_original(trellis, suh, svh, bits)
        native = weight_q  # (k, n) fp32, original domain
        diff = (decoded - native).abs()
        scale = native.abs().max().clamp_min(1e-12)
        max_abs = float(diff.max().item())
        max_rel = float((diff.max() / scale).item())
        # fp16 vector rounding is ~2^-11 relative; allow a small multiple.
        assert max_rel < 5e-3, (bits, max_abs, max_rel)

        report[f"k{bits}"] = {
            "native_proxy_err": float(proxy_err),
            "p1_repack_byte_identical": p1,
            "p2_decode_deterministic": p2,
            "p3_decode_vs_native_max_abs": max_abs,
            "p3_decode_vs_native_max_rel": max_rel,
            "native_trellis_sha256": _tensor_sha256(trellis),
            "words_per_tile": words,
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
