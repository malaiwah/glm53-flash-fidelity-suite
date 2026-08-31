#!/usr/bin/env python3
"""Offline (no GPU, no weights download) validation of the MLX surface adapter.

Proves, on this machine, in seconds:

  1. PACK-LAYOUT EQUIVALENCE + ACCELERATOR PARITY - a numpy reference packer
     (mlx's affine layout: a plain little-endian bitstream per output row)
     round-trips EXACTLY through ``unpack_affine_codes`` at every supported bit
     width (2,3,4,5,6,8) and group size (32/64/128), including the widths whose
     elements straddle 32-bit word boundaries; the dequant of those codes is
     exact in fp32; and the same kernel is BITWISE identical on whatever
     accelerator this machine has (MPS and/or CUDA) - which is only legal
     because the kernel has no float64 and no uint32 views in it.
  2. REAL-TENSOR mlx REPLAY - for every committed fixture in
     ``mlx-evidence/real-dequant-fixtures`` (row slices of REAL ranged-fetched
     tensors from orcarouter/GLM-5.3-Flash-MLX and
     pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit, together with the output
     mlx.core.dequantize itself produced), our fp32 dequant rounded ONCE to
     mlx's own output dtype is BITWISE equal to that stored output.  This is
     the mlx equality proof, replayable on machines where mlx cannot be
     installed (every CUDA box).
  3. LIVE mlx EQUALITY (macOS rung; SKIPped where mlx is absent) - round-trip
     through ``mlx.core.quantize``: our unpacked codes are EXACTLY mlx's codes
     and our dequant is bitwise equal to ``mlx.core.dequantize``, over the full
     bits x group-size grid, for f16 AND bf16 scales.
  4. REAL-METADATA CENSUS + REFUSALS - the census over the REAL orcarouter
     index and shard headers (revision c80f6810) closes at 37,338 quantized /
     1,432 passthrough / 38,770 logical tensors bijecting the REAL official
     BF16 census (revision a6c167b6), derives the mixed bit map (4/5/6-bit)
     that the config declaration independently agrees with, and REFUSES, by
     name: the mlx-vlm (pipenetwork) dialect, a census that does not close, an
     undeclared quantization mode, a bits-underivable tensor, a passthrough
     tensor that differs from the official dtype/shape, and an unpinned
     revision.
  5. STREAM PLUMBING - MlxExpertSource.load returns exactly the manual dequant
     at real routed geometry (2048x4096 / 4096x2048), with the right census
     row; prepare_nonrouted_view_decoded materializes a view whose quantized
     tensors are the manual dequant rounded once to bf16 and whose passthrough
     tensors are byte-identical, writes an index over exactly the non-routed
     names, strips the quantization block from config.json, reuses itself on a
     second call and REFUSES a stale view.
  6. DRY-RUN (surface) - ``mlx_surface.py dry-run`` over the REAL repo metadata
     prints the plan, and its fetch ledger reconciles EXACTLY with the index's
     own declared total_size (tensor bytes + the 62 shards' container headers).
     The mlx wiring inside stream_score.py is proven statically here too.
  7. DRY-RUN (runner) - ``stream_score.py --source mlx --dry-run`` prints its
     plan against that same metadata when a quant_pipeline tree is available
     (--pipeline-root / QP_PIPELINE_ROOT); SKIP otherwise, since the runner
     imports quant_pipeline unconditionally.
  8. REGISTRY ADAPTER - registry_add adapts a
     ``malaiwah.glm53-mlx-packed-kld-summary.v1`` receipt into a row: lane left
     to --lane, artifact revision pinned from the receipt, and the unsealed
     source / measured quantization scope / decoded non-routed weights /
     unverified shard hashes carried as coded disclosures.  A receipt with no
     scope census is REFUSED.

Run:  python3 selftest_mlx_offline.py [--pipeline-root DIR] [--keep]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
EVIDENCE = TOOLS / "mlx-evidence"
FIXTURES = EVIDENCE / "real-dequant-fixtures"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# What the REAL orcarouter snapshot must census to.  These are measured
# numbers, not targets: 36,288 routed (42 layers x 288 x 3) + 864 MTP
# (288 x 3) + 186 non-routed quantized = 37,338 quantized modules, and
# 37,338 + 1,432 passthrough = 38,770 = the official BF16 tensor count.
ORCAROUTER_EXPECTED = {
    "quantized": 37338,
    "routed_modules": 36288,
    "routed_mtp_modules": 864,
    "nonrouted_quantized_modules": 186,
    "passthrough_tensors": 1432,
    "logical_tensor_count": 38770,
    "stored_tensor_count": 113446,
    "bits_histogram": {"b4-gs64": 24822, "b5-gs64": 12387, "b6-gs64": 129},
}
ORCAROUTER_REVISION = "c80f6810b1a95b5be9042761becc6aa78d189782"
BF16_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"


def _pack_reference(codes: np.ndarray, bits: int) -> np.ndarray:
    """Independent transliteration of mlx's affine packing (see mlx-lm's
    quantized weight layout): element e of row r occupies bits
    [e*bits, (e+1)*bits) of that row's little-endian u32 bitstream.

    Deliberately written the OTHER way round from unpack_affine_codes (int
    accumulation per 32-bit word here, byte-level tensor ops there), so the
    round trip is a real cross-check and not one function inverting itself.
    """
    codes = np.asarray(codes, dtype=np.uint32)
    rows, cols = codes.shape
    assert (cols * bits) % 32 == 0, "reference packer needs cols*bits % 32 == 0"
    out = np.zeros((rows, cols * bits // 32), dtype=np.uint32)
    for r in range(rows):
        for c in range(cols):
            value = int(codes[r, c]) & ((1 << bits) - 1)
            position = c * bits
            word, offset = position >> 5, position & 31
            out[r, word] |= (value << offset) & 0xFFFFFFFF
            spill = offset + bits - 32
            if spill > 0:
                out[r, word + 1] |= value >> (bits - spill)
    return out


def _gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fh:
        dst.write_bytes(fh.read())


def _mock_metadata_root(scratch: Path) -> Path:
    """The REAL orcarouter config/index/shard-headers, laid out as a
    metadata-only snapshot root (exactly what `fetch-meta` produces)."""
    root = scratch / "orcarouter-meta"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(EVIDENCE / "orcarouter-config.json", root / "config.json")
    _gunzip(EVIDENCE / "orcarouter-index.json.gz", root / "model.safetensors.index.json")
    shutil.copy(EVIDENCE / "orcarouter-shard-headers.json.gz", root / "shard-headers.json.gz")
    return root


def _refuses(fn, needle: str) -> str:
    """Call fn; require a ValueError whose message NAMES the reason."""
    try:
        fn()
    except ValueError as error:
        message = str(error)
        assert needle in message, f"refusal did not mention {needle!r}: {message}"
        return message
    raise AssertionError(f"expected a refusal mentioning {needle!r}, got none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT"))
    parser.add_argument("--keep", action="store_true", help="keep the scratch roots")
    parser.add_argument("--json", type=Path, help="write the rung table here")
    args = parser.parse_args()

    import torch
    from safetensors.torch import save_file

    import mlx_surface as ms

    rng = np.random.default_rng(0x314159)
    passed = []
    scratch = Path(tempfile.mkdtemp(prefix="mlx-selftest-"))

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")

    try:
        # -------------------------------------------------------------- 1
        combos = 0
        for bits in ms.SUPPORTED_BITS:
            for group_size in (32, 64, 128):
                cols = group_size * 4
                if (cols * bits) % 32:
                    continue
                rows = 5
                codes = rng.integers(0, 1 << bits, size=(rows, cols), dtype=np.uint32)
                packed = _pack_reference(codes, bits)
                weight = torch.from_numpy(packed.view(np.int32).copy())
                got = ms.unpack_affine_codes(weight, bits=bits, in_features=cols)
                assert torch.equal(got, torch.from_numpy(codes.astype(np.int32))), (
                    f"unpack != reference pack (b{bits} gs{group_size})"
                )
                # dequant of those codes is exact in fp32 (integer q, f16 scales)
                scales = torch.from_numpy(
                    (rng.standard_normal((rows, cols // group_size)) * 0.01).astype(np.float16)
                )
                biases = torch.from_numpy(
                    (rng.standard_normal((rows, cols // group_size)) * 0.01).astype(np.float16)
                )
                want = (
                    torch.from_numpy(codes.astype(np.float32)).reshape(rows, -1, group_size)
                    * scales.float().unsqueeze(-1)
                    + biases.float().unsqueeze(-1)
                ).reshape(rows, cols)
                got_w = ms.dequant_affine(weight, scales, biases,
                                          bits=bits, group_size=group_size)
                assert torch.equal(got_w, want), f"dequant algebra differs (b{bits})"
                combos += 1
                # ACCELERATOR PARITY: the same kernel on the device this suite
                # scores on must produce the SAME bits.  No float64 and no
                # uint32 views are in it, which is what makes MPS legal here.
                for device in devices:
                    on_device = ms.dequant_affine(
                        weight.to(device), scales.to(device), biases.to(device),
                        bits=bits, group_size=group_size,
                    )
                    assert torch.equal(got_w, on_device.cpu()), (
                        f"dequant differs on {device} (b{bits} gs{group_size})"
                    )
        passed.append(
            "1 pack/unpack round trip + fp32 dequant algebra: %d bits x gs combos%s"
            % (combos,
               ", bitwise identical on " + "/".join(devices) if devices
               else " (no accelerator on this machine: CPU only)")
        )

        # -------------------------------------------------------------- 2
        fixtures = sorted(FIXTURES.glob("*.npz"))
        assert fixtures, f"no real-tensor fixtures under {FIXTURES}"
        replayed = []
        for path in fixtures:
            data = np.load(path, allow_pickle=False)
            bits = int(data["bits"])
            group_size = int(data["group_size"])
            scales_dtype = str(data["scales_dtype"])
            ref_dtype = str(data["ref_dtype"])
            weight = torch.from_numpy(data["weight"].view(np.int32).copy())

            def _tensor(name):
                raw = data[name]
                if scales_dtype == "bfloat16":
                    return torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16)
                return torch.from_numpy(raw.copy())

            ours = ms.dequant_affine(weight, _tensor("scales"), _tensor("biases"),
                                     bits=bits, group_size=group_size)
            if ref_dtype == "bfloat16":
                ours_bits = ours.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            else:
                ours_bits = ours.to(torch.float16).numpy().view(np.uint16)
            assert np.array_equal(ours_bits, data["ref_bits"]), (
                f"dequant != the mlx output stored in {path.name}"
            )
            module = str(data["module"])
            replayed.append(
                "%s b%d/gs%d %s"
                % (module.split("language_model.")[-1], bits, group_size, ref_dtype)
            )
        passed.append(
            "2 real-tensor mlx replay: %d fixtures bitwise equal (%s)"
            % (len(fixtures), "; ".join(sorted(replayed)))
        )

        # -------------------------------------------------------------- 3
        try:
            import mlx.core as mx
        except ImportError:
            passed.append("3 SKIPPED live mlx equality (mlx not importable: macOS-only rung)")
        else:
            grid = 0
            reference = (rng.standard_normal((8, 512)) * 0.02).astype(np.float32)
            for bits in ms.SUPPORTED_BITS:
                for group_size in (32, 64, 128):
                    packed, scale, bias = mx.quantize(
                        mx.array(reference), group_size=group_size, bits=bits
                    )
                    groups = 512 // group_size
                    codes_ref = np.array(
                        mx.dequantize(packed,
                                      mx.ones((8, groups), dtype=mx.float32),
                                      mx.zeros((8, groups), dtype=mx.float32),
                                      group_size=group_size, bits=bits)
                    ).astype(np.int64)
                    weight = torch.from_numpy(np.array(packed).view(np.int32).copy())
                    ours = ms.unpack_affine_codes(weight, bits=bits, in_features=512)
                    assert np.array_equal(ours.numpy().astype(np.int64), codes_ref), (
                        f"unpacked codes differ from mlx (b{bits} gs{group_size})"
                    )
                    for cast, torch_dtype in ((mx.float16, torch.float16),
                                              (mx.bfloat16, torch.bfloat16)):
                        s_c, b_c = scale.astype(cast), bias.astype(cast)
                        theirs = mx.dequantize(packed, s_c, b_c,
                                               group_size=group_size, bits=bits)
                        if torch_dtype is torch.bfloat16:
                            s_t = torch.from_numpy(
                                np.array(s_c.view(mx.uint16)).view(np.int16).copy()
                            ).view(torch.bfloat16)
                            b_t = torch.from_numpy(
                                np.array(b_c.view(mx.uint16)).view(np.int16).copy()
                            ).view(torch.bfloat16)
                            theirs_bits = np.array(theirs.view(mx.uint16))
                            ours_bits = (
                                ms.dequant_affine(weight, s_t, b_t, bits=bits,
                                                  group_size=group_size)
                                .to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
                            )
                        else:
                            s_t = torch.from_numpy(np.array(s_c).copy())
                            b_t = torch.from_numpy(np.array(b_c).copy())
                            theirs_bits = np.array(theirs).view(np.uint16)
                            ours_bits = (
                                ms.dequant_affine(weight, s_t, b_t, bits=bits,
                                                  group_size=group_size)
                                .to(torch.float16).numpy().view(np.uint16)
                            )
                        assert np.array_equal(ours_bits, theirs_bits), (
                            f"dequant differs from mlx.core.dequantize "
                            f"(b{bits} gs{group_size} {torch_dtype})"
                        )
                        grid += 1
            passed.append(
                "3 live mlx equality: %d (bits x gs x scale-dtype) cells, codes EXACT and "
                "output bitwise equal (mlx %s)" % (grid, getattr(mx, "__version__", "?"))
            )

        # -------------------------------------------------------------- 4
        census_path = EVIDENCE / "bf16-shape-census.json.gz"
        official, official_meta = ms.load_official_census(census_path)
        assert official_meta["source_revision"] == BF16_REVISION
        assert len(official) == ORCAROUTER_EXPECTED["logical_tensor_count"]

        index = json.loads(gzip.open(EVIDENCE / "orcarouter-index.json.gz", "rb").read())
        headers = json.loads(gzip.open(EVIDENCE / "orcarouter-shard-headers.json.gz", "rb").read())
        weight_map = index["weight_map"]
        meta = {}
        for shard, header in headers.items():
            for name, value in header.items():
                if name in ("__metadata__", "__header_len__"):
                    continue
                meta[name] = (value["dtype"], tuple(value["shape"]))
        census = ms.census_index(weight_map, meta, official)
        for key in ("routed_modules", "routed_mtp_modules", "nonrouted_quantized_modules",
                    "passthrough_tensors"):
            assert len(census[key]) == ORCAROUTER_EXPECTED[key], (key, len(census[key]))
        assert len(census["quantized"]) == ORCAROUTER_EXPECTED["quantized"]
        assert census["logical_tensor_count"] == ORCAROUTER_EXPECTED["logical_tensor_count"]
        assert census["stored_tensor_count"] == ORCAROUTER_EXPECTED["stored_tensor_count"]
        assert census["bits_histogram"] == ORCAROUTER_EXPECTED["bits_histogram"], \
            census["bits_histogram"]
        config = json.loads((EVIDENCE / "orcarouter-config.json").read_text())
        agreement = ms.crosscheck_config_declaration(config, census)
        assert agreement["default_bits"] == 4 and agreement["default_group_size"] == 64
        assert agreement["declared_modules_checked"] == 37047
        assert agreement["mtp_layer45_modules_outside_config_overrides"] == 291

        # every routed module the streamer will ask for is present and 2-D
        for layer in (3, 44):
            for projection in ms.PROJECTIONS:
                module = ("model.language_model.layers.%d.mlp.experts.0.%s" % (layer, projection))
                row = census["quantized"][module]
                assert (row["out_features"], row["in_features"]) == ms.PROJECTION_SHAPE[projection]

        # --- refusals, each one NAMED -----------------------------------
        refusals = {}
        pipe_index = json.loads(
            gzip.open(EVIDENCE / "pipenetwork-4bit-index.json.gz", "rb").read()
        )
        refusals["foreign_dialect"] = _refuses(
            lambda: ms.census_index(pipe_index["weight_map"], meta, official),
            "unsupported MLX dialect",
        )
        half_dropped = dict(weight_map)
        half_dropped.pop("model.language_model.layers.3.mlp.experts.0.gate_proj.weight")
        refusals["packed_weight_missing_from_a_triplet"] = _refuses(
            lambda: ms.census_index(half_dropped, meta, official),
            "scales/biases without a packed weight",
        )
        broken = dict(weight_map)
        for suffix in (".weight", ".scales", ".biases"):
            broken.pop("model.language_model.layers.3.mlp.experts.0.gate_proj" + suffix)
        refusals["census_does_not_close"] = _refuses(
            lambda: ms.census_index(broken, meta, official),
            "does not biject the official BF16 census",
        )
        bad_meta = dict(meta)
        bad_meta["model.language_model.layers.3.mlp.experts.0.gate_proj.weight"] = ("U32", (2048, 511))
        refusals["bits_underivable"] = _refuses(
            lambda: ms.census_index(weight_map, bad_meta, official),
            "bits underivable",
        )
        bad_pass = dict(meta)
        bad_pass["model.language_model.embed_tokens.weight"] = ("F32", (154880, 4096))
        refusals["passthrough_differs_from_official"] = _refuses(
            lambda: ms.census_index(weight_map, bad_pass, official),
            "passthrough tensors differ from the official",
        )
        mixed_up = json.loads(json.dumps(config))
        mixed_up["quantization"]["bits"] = 6
        refusals["config_disagrees_with_shapes"] = _refuses(
            lambda: ms.crosscheck_config_declaration(mixed_up, census),
            "disagrees with stored shapes",
        )
        moded = json.loads(json.dumps(config))
        moded["quantization"]["mode"] = "mxfp4"
        root = _mock_metadata_root(scratch)
        (root / "config-mxfp4.json").write_text(json.dumps(moded))
        mode_root = scratch / "mode-root"
        mode_root.mkdir(exist_ok=True)
        for name in ("model.safetensors.index.json", "shard-headers.json.gz"):
            os.symlink(root / name, mode_root / name)
        shutil.copy(root / "config-mxfp4.json", mode_root / "config.json")
        refusals["non_affine_mode"] = _refuses(
            lambda: ms.load_mlx_surface(mode_root, require_shard_hashes=False),
            "is a named exclusion (affine only)",
        )
        refusals["unpinned_revision"] = _refuses(
            lambda: ms.load_mlx_surface(root, revision="not-a-commit",
                                        require_shard_hashes=False),
            "immutable 40-hex repo commit",
        )
        refusals["shard_hashes_unverified"] = _refuses(
            lambda: ms.load_mlx_surface(scratch / "no-such-root"),
            "mlx config.json missing",
        )
        passed.append(
            "4 REAL orcarouter census closes (37,338 quantized / 1,432 passthrough / "
            "38,770 logical bijecting the REAL BF16 census a6c167b6), mixed bits "
            "b4:24822 b5:12387 b6:129 agree with config.json, and %d refusals fire by name"
            % len(refusals)
        )

        # -------------------------------------------------------------- 5
        surface_root = scratch / "mini-snapshot"
        surface_root.mkdir()
        bits, group_size = 4, 64
        tensors = {}
        expected = {}
        quantized_rows = {}
        for projection in ms.PROJECTIONS:
            module = "model.language_model.layers.3.mlp.experts.0." + projection
            out_f, in_f = ms.PROJECTION_SHAPE[projection]
            codes = rng.integers(0, 1 << bits, size=(out_f, in_f), dtype=np.uint32)
            packed = _pack_reference(codes, bits)
            scales = (rng.standard_normal((out_f, in_f // group_size)) * 0.01).astype(np.float16)
            biases = (rng.standard_normal((out_f, in_f // group_size)) * 0.01).astype(np.float16)
            tensors[module + ".weight"] = torch.from_numpy(packed.view(np.int32).copy())
            tensors[module + ".scales"] = torch.from_numpy(scales)
            tensors[module + ".biases"] = torch.from_numpy(biases)
            expected[projection] = ms.dequant_affine(
                tensors[module + ".weight"], tensors[module + ".scales"],
                tensors[module + ".biases"], bits=bits, group_size=group_size,
            )
            quantized_rows[module] = {
                "bits": bits, "group_size": group_size, "out_features": out_f,
                "in_features": in_f, "scales_dtype": "F16",
            }
        # two non-routed modules for the decoded view: one quantized, one passthrough
        nonrouted_module = "model.language_model.layers.3.self_attn.o_proj"
        codes = rng.integers(0, 1 << bits, size=(128, 256), dtype=np.uint32)
        tensors[nonrouted_module + ".weight"] = torch.from_numpy(
            _pack_reference(codes, bits).view(np.int32).copy()
        )
        tensors[nonrouted_module + ".scales"] = torch.from_numpy(
            (rng.standard_normal((128, 4)) * 0.01).astype(np.float16)
        )
        tensors[nonrouted_module + ".biases"] = torch.from_numpy(
            (rng.standard_normal((128, 4)) * 0.01).astype(np.float16)
        )
        quantized_rows[nonrouted_module] = {
            "bits": bits, "group_size": group_size, "out_features": 128,
            "in_features": 256, "scales_dtype": "F16",
        }
        passthrough_name = "model.language_model.norm.weight"
        tensors[passthrough_name] = torch.arange(64, dtype=torch.bfloat16)
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, str(surface_root / shard), metadata={"format": "pt"})
        (surface_root / "tokenizer_config.json").write_text('{"selftest": true}\n')

        mini = ms.MlxSurface(
            root=surface_root, repo="selftest/mini", revision="0" * 40,
            config_sha256="0" * 64, index_sha256="1" * 64, official_census_sha256="2" * 64,
            official_source_repo="zai-org/GLM-5.3-Flash-BF16", official_source_revision=BF16_REVISION,
            default_bits=bits, default_group_size=group_size,
            weight_map={name: shard for name in tensors},
            tensor_meta={name: ("U32", tuple(value.shape)) for name, value in tensors.items()},
            census={
                "quantized": quantized_rows,
                "routed_modules": sorted(
                    m for m in quantized_rows if ".mlp.experts." in m + "."
                ),
                "routed_mtp_modules": [],
                "nonrouted_quantized_modules": [nonrouted_module],
                "passthrough_tensors": [passthrough_name],
                "bits_histogram": {"b4-gs64": len(quantized_rows)},
                "logical_tensor_count": len(quantized_rows) + 1,
                "stored_tensor_count": len(tensors),
            },
            config={"architectures": ["Glm5NextForConditionalGeneration"],
                    "model_type": "glm5_next", "quantization": {"bits": 4, "group_size": 64},
                    "quantization_config": {"bits": 4, "group_size": 64},
                    "text_config": {"vocab_size": 154880}},
            config_agreement={}, shard_hash_verification="skipped", metadata_only=False,
            text_vocab_size=154880,
        )
        source = ms.MlxExpertSource(mini)
        for projection in ms.PROJECTIONS:
            decoded, row = source.load(layer=3, expert=0, projection=projection)
            assert torch.equal(decoded, expected[projection]), f"streamed {projection} differs"
            assert tuple(decoded.shape) == ms.PROJECTION_SHAPE[projection]
            assert row["shard"] == shard and row["quant"] == {"bits": 4, "group_size": 64}
            assert row["tensor"].endswith(projection + ".weight")
        assert source.shards_read == {shard} and source.bytes_read > 0

        work = scratch / "work"
        view, record = ms.prepare_nonrouted_view_decoded(mini, work, progress=False)
        assert record["decoded_module_count"] == 1 and record["passthrough_tensor_count"] == 1
        assert record["routed_modules_filtered"] == 3
        assert record["routed_stored_tensors_filtered"] == 9
        view_index = json.loads((view / "model.safetensors.index.json").read_text())
        assert set(view_index["weight_map"]) == {nonrouted_module + ".weight", passthrough_name}
        view_config = json.loads((view / "config.json").read_text())
        assert "quantization" not in view_config and "quantization_config" not in view_config
        assert (view / "tokenizer_config.json").is_file(), "aux files not carried into the view"
        from safetensors import safe_open

        with safe_open(str(view / view_index["weight_map"][passthrough_name]),
                       framework="pt", device="cpu") as handle:
            got_norm = handle.get_tensor(passthrough_name)
            got_o = safe_open(
                str(view / view_index["weight_map"][nonrouted_module + ".weight"]),
                framework="pt", device="cpu",
            ).get_tensor(nonrouted_module + ".weight")
        assert torch.equal(got_norm, tensors[passthrough_name]), "passthrough not byte-identical"
        want_o = ms.dequant_affine(
            tensors[nonrouted_module + ".weight"], tensors[nonrouted_module + ".scales"],
            tensors[nonrouted_module + ".biases"], bits=bits, group_size=group_size,
        ).to(torch.bfloat16)
        assert got_o.dtype == torch.bfloat16 and torch.equal(got_o, want_o), \
            "decoded non-routed tensor is not the fp32 dequant rounded once to bf16"
        again = ms.prepare_nonrouted_view_decoded(mini, work, progress=False)[1]
        assert again.get("reused") is True, "second call did not reuse the materialized view"
        stale = ms.MlxSurface(**dict(mini.__dict__, index_sha256="f" * 64))
        _refuses(lambda: ms.prepare_nonrouted_view_decoded(stale, work, progress=False),
                 "stale decoded view")
        passed.append(
            "5 stream plumbing: MlxExpertSource == manual dequant at real routed geometry; "
            "decoded view exact (bf16 quantized / byte-identical passthrough), reuses and "
            "refuses a stale view"
        )

        # -------------------------------------------------------------- 6
        env = dict(os.environ)
        env.pop("QP_PIPELINE_ROOT", None)
        run = subprocess.run(
            [sys.executable, str(TOOLS / "mlx_surface.py"), "dry-run",
             "--mlx-root", str(root), "--repo", "orcarouter/GLM-5.3-Flash-MLX",
             "--revision", ORCAROUTER_REVISION, "--skip-shard-hashes"],
            capture_output=True, text=True, env=env,
        )
        assert run.returncode == 0, run.stderr
        plan = json.loads(run.stdout)
        assert plan["schema"] == ms.MLX_SURFACE_SCHEMA
        assert plan["metadata_only"] is True
        assert plan["default_bits"] == 4 and plan["default_group_size"] == 64
        assert plan["student_label"].startswith("mlx-affine-b4-gs64-mixed-")
        assert plan["scope_policy"]["routed_expert_modules"] == 36288
        assert plan["scope_policy"]["nonrouted_quantized_modules"] == 186
        assert plan["official_source_revision"] == BF16_REVISION
        ledger = plan["fetch_ledger"]
        assert ledger["total_artifact"] == sum(
            ledger[key] for key in ("routed_packed", "mtp_packed",
                                    "nonrouted_quantized_packed", "passthrough")
        )
        # The index's own metadata.total_size is the independent check on the
        # ledger: for this snapshot it is the ON-DISK total, so tensor bytes +
        # the 62 shards' container headers must reproduce it EXACTLY.
        declared_total = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["metadata"]["total_size"]
        assert ledger["index_declared_total_size"] == declared_total
        assert ledger["on_disk_total_bytes"] == declared_total, (
            ledger["on_disk_total_bytes"], declared_total
        )
        assert ledger["declared_total_matches"] == "on_disk_with_container_headers"
        assert ledger["routed_packed"] == 183911841792
        assert ledger["passthrough"] == 13853375352
        identity = plan["checkpoint_identity_sha256"]
        assert len(identity) == 64

        source_text = (TOOLS / "stream_score.py").read_text()
        for needle in (
            '"mlx"',
            "mlx_surface_obj.student_label()",
            "prepare_nonrouted_view_decoded",
            "mlx_source=mlx_expert_source",
            "decoded_bf16_view_materialized_from_the_quant_snapshot",
            "streamed_decoded_mlx_affine_u32_to_bf16_one_layer_resident",
        ):
            assert needle in source_text, f"stream_score.py lost its mlx wiring: {needle}"

        passed.append(
            "6 dry-run: `mlx_surface.py dry-run` over the REAL repo metadata prints identity "
            "%s..., the scope census and a fetch ledger that reconciles EXACTLY with the "
            "index's declared total_size (%d B on disk)"
            % (identity[:12], ledger["on_disk_total_bytes"])
        )

        if args.pipeline_root:
            teacher = scratch / "teacher"
            _write_mock_panel(teacher, rng, args.pipeline_root)
            run = subprocess.run(
                [sys.executable, str(TOOLS / "stream_score.py"),
                 "--source", "mlx", "--profile", "mlx",
                 "--mlx-root", str(root), "--mlx-repo", "orcarouter/GLM-5.3-Flash-MLX",
                 "--mlx-revision", ORCAROUTER_REVISION, "--mlx-skip-shard-hashes",
                 "--teacher", str(teacher), "--cold-run", "1",
                 "--out", str(scratch / "out-dry"),
                 "--pipeline-root", args.pipeline_root, "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            assert run.returncode == 0, run.stderr
            stream_plan = json.loads(run.stdout.splitlines()[-1])
            assert stream_plan["dry_run"] is True
            assert stream_plan["weight_source"] == "mlx"
            assert stream_plan["nonrouted_policy"] == \
                "decoded_bf16_view_materialized_from_the_quant_snapshot"
            assert stream_plan["mlx"]["mlx_revision"] == ORCAROUTER_REVISION
            assert stream_plan["mlx"]["scope_policy"]["routed_expert_modules"] == 36288
            assert stream_plan["inventory_sha256"] is None
            passed.append(
                "7 stream_score --source mlx --dry-run: plan printed, nonrouted_policy "
                "decoded-view, scope census carried, no BF16 inventory required"
            )
        else:
            passed.append(
                "7 SKIPPED stream_score --source mlx --dry-run (no --pipeline-root / "
                "QP_PIPELINE_ROOT on this machine); its mlx wiring was proven statically above"
            )

        # -------------------------------------------------------------- 7
        registry_tools = TOOLS.parent.parent / "registry" / "tools"
        sys.path.insert(0, str(registry_tools))
        import registry_add as ra

        assert ra.MLX_SUMMARY in ra.STREAM_SUMMARIES and ra.MLX_SUMMARY in ra.OWN_SCHEMAS
        assert ra.LANE_STATED_BY_SCHEMA[ra.MLX_SUMMARY] is None, (
            "the mlx family's schema string does not name a lane; --lane must supply it"
        )
        summary = {
            "schema": ra.MLX_SUMMARY,
            "profile": "mlx-stream",
            "student_label": plan["student_label"],
            "cold_run_count": 2,
            "run_means": [0.0301234, 0.0301234],
            "distinct_tokenwise_kld_sha256": ["a" * 64],
            "bitwise_deterministic": True,
            "measured_mean_kld": 0.0301234,
            "kld_report_sha256": ["b" * 64, "c" * 64],
            "teacher_receipt_sha256": "d" * 64,
            "mlx_repo": "orcarouter/GLM-5.3-Flash-MLX",
            "mlx_revision": ORCAROUTER_REVISION,
            "mlx_shard_hash_verification": "skipped",
            "mlx_scope_policy": plan["scope_policy"],
            "nonrouted_policy": "decoded_bf16_view_materialized_from_the_quant_snapshot",
            "seal_disclosure": "unsealed-source scoring: [...]",
        }
        adapted = ra.adapt_stream_summary([(summary, "mlx-packed-kld.json", None)])
        assert adapted["receipt_schema"] == ra.MLX_SUMMARY
        assert adapted["lane"] is None and adapted["requires_lane"] is True
        assert adapted["artifact_revision"] == ORCAROUTER_REVISION
        codes = [entry["code"] for entry in adapted["verbatim_disclosure_coded"]]
        for code in ("unsealed_source", "quantization_scope", "nonrouted_weights_decoded",
                     "shard_hashes_unverified"):
            assert code in codes, (code, codes)
        scope_text = next(entry["detail"] for entry in adapted["verbatim_disclosure_coded"]
                          if entry["code"] == "quantization_scope")
        assert "36288 routed expert" in scope_text and "186 non-routed" in scope_text
        no_scope = dict(summary)
        no_scope.pop("mlx_scope_policy")
        try:
            ra.adapt_stream_summary([(no_scope, "mlx-packed-kld.json", None)])
        except ra.Refuse as error:
            assert "mlx_scope_policy" in str(error)
        else:
            raise AssertionError("a receipt without the scope census must be REFUSED")
        passed.append(
            "8 registry adapter: %s adapts to a row (lane via --lane), pins the artifact "
            "revision, carries 4 coded disclosures incl. the measured scope, and refuses a "
            "receipt with no scope census" % ra.MLX_SUMMARY
        )
    finally:
        if args.keep:
            print(f"kept scratch: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    skipped = sum(1 for line in passed if " SKIPPED" in line)
    for line in passed:
        print(("SKIP  " if " SKIPPED" in line else "PASS  ") + line)
    print("selftest_mlx_offline: %d/%d rungs green, %d skipped"
          % (len(passed) - skipped, len(passed), skipped))
    if args.json:
        args.json.write_text(json.dumps({"rungs": passed}, indent=2) + "\n")
    return 0


def _write_mock_panel(teacher: Path, rng, pipeline_root: str) -> None:
    """A synthetic sealed token panel (2 tiny windows) - enough for a plan print."""
    sys.path.insert(0, str(_pipeline_src(Path(pipeline_root))))
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file

    teacher.mkdir(parents=True, exist_ok=True)
    artifacts, rows = [], []
    for index in range(2):
        tokens = rng.integers(0, 154880, size=64, dtype=np.int64)
        mask = np.ones(64, dtype=np.int64)
        mask[:index] = 0
        token_path = teacher / f"tokens-{index}.npy"
        mask_path = teacher / f"mask-{index}.npy"
        np.save(token_path, tokens, allow_pickle=False)
        np.save(mask_path, mask, allow_pickle=False)
        digests = {}
        for path in (token_path, mask_path):
            digest = sha256_file(path)
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
            digests[path.name] = digest
        rows.append({
            "window_id": f"final-{index:04d}", "document_id": f"selftest-doc-{index}",
            "domain": "selftest", "role": "final",
            "token_ids_sha256": digests[token_path.name],
            "attention_mask_sha256": digests[mask_path.name],
            "prediction_positions": int(
                (np.asarray(mask[:-1], dtype=bool) & np.asarray(mask[1:], dtype=bool)).sum()
            ),
        })
    panel_path = teacher / "token-panel.json"
    panel_path.write_text(json.dumps({"schema": "quant-pipeline.glm53-token-panel.v1",
                                      "windows": rows}))
    panel_digest = sha256_file(panel_path)
    artifacts.append({"path": str(panel_path), "bytes": panel_path.stat().st_size,
                      "sha256": panel_digest})
    receipt = {"schema": "quant-pipeline.glm53-token-panel-receipt.v1",
               "token_panel_artifact_sha256": panel_digest, "artifacts": artifacts}
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    (teacher / "panel-receipt.json").write_text(json.dumps(receipt))


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise SystemExit(f"no quant_pipeline package under {pipeline_root}")


if __name__ == "__main__":
    raise SystemExit(main())
