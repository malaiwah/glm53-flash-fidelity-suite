#!/usr/bin/env python3
"""Offline (no GPU, no weights download) validation of the Dione surface adapter.

Proves, on this machine:
  1. PACK-LAYOUT EQUIVALENCE - a numpy reference packer transliterated from
     exllamav3's pack_trellis CUDA kernel (exllamav3-src @ c5d9c65 / 5f3c537,
     quant/pack.cu: 16 spans, big-endian 32-bit buffer, SWAP16 store) packs
     valid tail-biting trellis state streams that the campaign reader's
     unpack_trellis_states inverts EXACTLY (bits 4 and 6), and that the
     adapter's anybits copy inverts EXACTLY (bits 3, 4, 6, 2 -- K2 added
     for M4's vcruz305 EXL3-K2 pack).
  2. DECODE IDENTITY - the adapter's decode path is bitwise identical to the
     campaign reader's decode_choice_hf on identical payloads (bits 4, 6);
     the K3 anybits path is exercised for shape/orientation.
  3. SAFETENSORS E2E - a synthetic Dione-format shard (real GLM53 slice
     geometry) loaded through DioneShardReader + load_decoded_module equals
     the direct reader decode + rank-ordered concat, bitwise.
  4. REAL-METADATA CENSUS - census_weight_map over the REAL Q4 index
     (dione-evidence/index-q4.json, repo revision 99cccdf0) closes at
     580,608 packed / 2,482 retained, and the retained set + routed originals
     exactly biject the REAL official BF16 index (a6c167b6).
  5. DRY-RUN - `dione_surface.py dry-run` and `k6_student_capture.py
     --surface dione --dry-run` run to plan-print against a mock snapshot
     carrying the real config/index/manifest plus a synthetic sealed panel.

Run:  python3 selftest_dione_offline.py --pipeline-root <tree-with-quant_pipeline>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
EVIDENCE = TOOLS / "dione-evidence"


def _pipeline_src(pipeline_root: Path) -> Path:
    for candidate in ("runtime/src", "src", "."):
        if (pipeline_root / candidate / "quant_pipeline" / "__init__.py").is_file():
            return (pipeline_root / candidate).resolve()
    raise SystemExit(f"no quant_pipeline package under {pipeline_root}")


def pack_trellis_reference(states: np.ndarray, bits: int) -> np.ndarray:
    """Transliteration of pack_trellis_kernel<K> (exllamav3 quant/pack.cu)."""
    assert states.ndim == 2 and states.shape[1] == 256 and states.dtype == np.uint16
    tiles, mask = states.shape[0], (1 << bits) - 1
    packed = np.zeros((tiles, 256 * bits // 16), dtype=np.uint16)
    for tile in range(tiles):
        s_packed = np.zeros(256 * bits // 16, dtype=np.uint16)
        for span in range(16):
            i, j, k, buf = 16 * span, bits * span, 32, 0
            for _ in range(16):
                v = int(states[tile, i]) & mask
                k -= bits
                buf = (buf | (v << k)) & 0xFFFFFFFF
                if k <= 16:
                    s_packed[j] = (buf >> 16) & 0xFFFF
                    buf = (buf << 16) & 0xFFFFFFFF
                    k += 16
                    j += 1
                i += 1
        # final store does SWAP16 per uint32 == swap adjacent uint16 pairs
        packed[tile] = s_packed.reshape(-1, 2)[:, ::-1].reshape(-1)
    return packed


def tail_biting_states(rng: np.random.Generator, tiles: int, bits: int) -> np.ndarray:
    """Valid circular trellis streams: state_n = concat of trailing K-bit codes."""
    codes = rng.integers(0, 1 << bits, size=(tiles, 256), dtype=np.uint32)
    states = np.zeros((tiles, 256), dtype=np.uint32)
    for lag in range(math.ceil(16 / bits)):
        states |= np.roll(codes, lag, axis=1) << (lag * bits)
    return (states & 0xFFFF).astype(np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT"))
    parser.add_argument("--keep", action="store_true", help="keep the mock roots")
    args = parser.parse_args()
    if not args.pipeline_root:
        raise SystemExit("pass --pipeline-root (tree containing quant_pipeline)")
    src = str(_pipeline_src(Path(args.pipeline_root)))
    if src not in sys.path:
        sys.path.insert(0, src)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))

    import torch
    from safetensors.torch import save_file

    import dione_surface as ds
    from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file
    from quant_pipeline.evaluation import glm53_packed_k4_reader as reader

    rng = np.random.default_rng(0xD10E)
    passed = []

    # ------------------------------------------------------------------ 1
    # K2 is appended rather than prepended: the rng is consumed in order and
    # the rungs below draw from it, so a new rate at the FRONT would silently
    # re-roll every later fixture.
    for bits in (3, 4, 6, 2):
        states = tail_biting_states(rng, tiles=48, bits=bits)
        packed = pack_trellis_reference(states, bits)
        packed_t = torch.from_numpy(packed.astype(np.int16)).reshape(6, 8, bits * 16)
        got_any = ds._unpack_trellis_states_anybits(packed_t, bits)
        want = torch.from_numpy(states.astype(np.int16)).reshape(6, 8, 256)
        assert torch.equal(got_any, want), f"anybits unpack != packed states (K{bits})"
        if bits in reader.SUPPORTED_BITS:
            got_reader = reader.unpack_trellis_states(packed_t, bits=bits)
            assert torch.equal(got_reader, want), f"reader unpack != packed states (K{bits})"
            assert torch.equal(got_reader, got_any), f"reader != anybits unpack (K{bits})"
    passed.append("1 pack-layout equivalence (exllamav3 pack.cu transliteration): K3/K4/K6/K2")

    # ------------------------------------------------------------------ 2
    for bits in (4, 6):
        trellis = torch.from_numpy(
            rng.integers(-(2 ** 15), 2 ** 15, size=(32, 16, bits * 16), dtype=np.int64)
            .astype(np.int16)
        )
        suh = torch.from_numpy(rng.standard_normal(512).astype(np.float16))
        svh = torch.from_numpy(rng.standard_normal(256).astype(np.float16))
        ours = ds._decode_choice_hf_anybits(trellis, suh, svh, bits=bits)
        theirs = reader.decode_choice_hf(trellis, suh, svh, bits=bits)
        assert torch.equal(ours, theirs), f"anybits decode != reader decode (K{bits})"
        via_dispatch = ds.decode_slice(trellis, suh, svh, bits=bits)
        assert torch.equal(via_dispatch, theirs), f"decode_slice dispatch differs (K{bits})"
    trellis3 = torch.from_numpy(
        rng.integers(-(2 ** 15), 2 ** 15, size=(32, 16, 48), dtype=np.int64).astype(np.int16)
    )
    d3 = ds.decode_slice(trellis3, torch.randn(512).half(), torch.randn(256).half(), bits=3)
    assert tuple(d3.shape) == (256, 512), "K3 decode orientation differs"
    passed.append("2 decode identity vs campaign reader: bitwise equal (K4/K6), K3 shape ok")

    # ------------------------------------------------------------------ 3
    # The scratch dir used to be created INSIDE dione-evidence/, which is a
    # local evidence directory that does not travel in the measurement
    # bundle -- so this selftest could not run anywhere the source tree is a
    # subset, which is every measurement instance. Fall back to the system
    # temp dir when it is absent (the rungs that READ the evidence already
    # self-skip on the same condition).
    _scratch_parent = TOOLS / "dione-evidence"
    scratch = Path(tempfile.mkdtemp(
        prefix="dione-selftest-",
        dir=str(_scratch_parent) if _scratch_parent.is_dir() else None))
    mock = scratch / "mock-mini"
    (mock / "layers").mkdir(parents=True)
    bits, tp = 4, 4
    layer, expert = 3, 0
    tensors, manual = {}, {}
    for projection in ds.PROJECTIONS:
        geometry = ds.expected_slice_geometry(projection, bits=bits, tp_size=tp)
        slices = []
        for rank in range(tp):
            tr = torch.from_numpy(
                rng.integers(-(2 ** 15), 2 ** 15, size=geometry["trellis"][1], dtype=np.int64)
                .astype(np.int16)
            )
            suh = torch.from_numpy(rng.standard_normal(geometry["suh"][1][0]).astype(np.float16))
            svh = torch.from_numpy(rng.standard_normal(geometry["svh"][1][0]).astype(np.float16))
            mcg = torch.tensor(ds.MCG_MARKER_SIGNED_INT32, dtype=torch.int32)
            for obj, value in (("trellis", tr), ("suh", suh), ("svh", svh), ("mcg", mcg)):
                tensors[ds.slice_name(layer, expert, projection, rank, obj)] = value
            slices.append(reader.decode_choice_hf(tr, suh, svh, bits=bits))
        manual[projection] = torch.cat(slices, dim=ds.CONCAT_DIM[projection]).contiguous()
    shard_rel = "layers/layer-03-part-0.safetensors"
    save_file(tensors, str(mock / shard_rel))
    surface = ds.DioneSurface(
        root=mock, repo="selftest/mini", revision="0" * 40, bits=bits, tp_size=tp,
        fmt=ds.DIONE_FORMAT, source_repo="selftest/bf16", source_revision="1" * 40,
        config_sha256="0" * 64, index_sha256="0" * 64, exl3_manifest_sha256=None,
        weight_map={name: shard_rel for name in tensors},
        retained_names=(), shard_hash_verification="skipped", text_vocab_size=154880,
    )
    shreader = ds.DioneShardReader(surface)
    for projection in ds.PROJECTIONS:
        assembled, census = ds.load_decoded_module(
            surface, shreader, layer=layer, expert=expert, projection=projection, device="cpu"
        )
        assert torch.equal(assembled, manual[projection]), f"e2e assembly differs: {projection}"
        assert tuple(assembled.shape) == ds.PROJECTION_SHAPE[projection]
        assert len(census["slices"]) == tp and all(
            len(row["trellis_sha256"]) == 64 for row in census["slices"]
        )
    passed.append("3 safetensors e2e: DioneShardReader+load_decoded_module bitwise == manual")

    # ------------------------------------------------------------------ 4
    real_index = EVIDENCE / "index-q4.json"
    real_bf16_index = EVIDENCE / "bf16-index.json"
    if real_index.is_file() and real_bf16_index.is_file():
        weight_map = json.loads(real_index.read_text())["weight_map"]
        retained, counts = ds.census_weight_map(weight_map, tp_size=4)
        assert counts == {
            "packed_tensors": 580608,
            "packed_modules": 36288,
            "retained_tensors": 2482,
        }, counts
        official = set(json.loads(real_bf16_index.read_text())["weight_map"])
        routed = {
            ds.official_name(l, e, p)
            for l in ds.MAIN_ROUTED_LAYERS
            for e in range(ds.NUM_EXPERTS)
            for p in ds.PROJECTIONS
        }
        assert set(retained) | routed == official, "retained+routed != official BF16 tensor set"
        assert not (set(retained) & routed)
        passed.append(
            "4 REAL Q4 index census closes (580608 packed / 2482 retained) and "
            "bijects the REAL BF16 index"
        )
    else:
        passed.append("4 SKIPPED (dione-evidence real indexes absent)")

    # ------------------------------------------------------------------ 5
    mock_q4 = scratch / "mock-q4-root"
    mock_bf16 = scratch / "mock-bf16-root"
    teacher = scratch / "mock-teacher"
    # official BF16 config: evidence copy, a real checkpoint via QP_BF16_ROOT
    # (box: /home/jl_fs/models/bf16), or the Mac scratchpad mirror
    bf16_config = next(
        (
            path
            for path in (
                EVIDENCE / "bf16-config.json",
                Path(os.environ.get("QP_BF16_ROOT", "/nonexistent")) / "config.json",
                TOOLS.parent.parent / "bf16-config.json",
            )
            if path.is_file()
        ),
        None,
    )
    if (
        (EVIDENCE / "config-q4.json").is_file()
        and real_index.is_file()
        and bf16_config is not None
    ):
        mock_q4.mkdir()
        mock_bf16.mkdir()
        teacher.mkdir()
        shutil.copy(EVIDENCE / "config-q4.json", mock_q4 / "config.json")
        shutil.copy(real_index, mock_q4 / "model.safetensors.index.json")
        if (EVIDENCE / "exl3-manifest.json").is_file():
            shutil.copy(EVIDENCE / "exl3-manifest.json", mock_q4 / "exl3-manifest.json")
        shutil.copy(bf16_config, mock_bf16 / "config.json")
        shutil.copy(real_bf16_index, mock_bf16 / "model.safetensors.index.json")

        # synthetic sealed token panel (2 tiny final windows)
        artifacts = []
        rows = []
        for index in range(2):
            tokens = rng.integers(0, 154880, size=64, dtype=np.int64)
            mask = np.ones(64, dtype=np.int64)
            mask[:index] = 0  # distinct masks: content-addressed artifacts must be unique
            token_path = teacher / f"tokens-{index}.npy"
            mask_path = teacher / f"mask-{index}.npy"
            np.save(token_path, tokens, allow_pickle=False)
            np.save(mask_path, mask, allow_pickle=False)
            digests = {}
            for path in (token_path, mask_path):
                digest = sha256_file(path)
                artifacts.append(
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
                )
                digests[path.name] = digest
            rows.append(
                {
                    "window_id": f"final-{index:04d}",
                    "document_id": f"selftest-doc-{index}",
                    "domain": "selftest",
                    "role": "final",
                    "token_ids_sha256": digests[token_path.name],
                    "attention_mask_sha256": digests[mask_path.name],
                    "prediction_positions": int(
                        (np.asarray(mask[:-1], dtype=bool) & np.asarray(mask[1:], dtype=bool)).sum()
                    ),
                }
            )
        panel = {"schema": "quant-pipeline.glm53-token-panel.v1", "windows": rows}
        panel_path = teacher / "token-panel.json"
        panel_path.write_text(json.dumps(panel))
        panel_digest = sha256_file(panel_path)
        artifacts.append(
            {"path": str(panel_path), "bytes": panel_path.stat().st_size, "sha256": panel_digest}
        )
        receipt = {
            "schema": "quant-pipeline.glm53-token-panel-receipt.v1",
            "token_panel_artifact_sha256": panel_digest,
            "artifacts": artifacts,
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
        (teacher / "panel-receipt.json").write_text(json.dumps(receipt))

        env = dict(os.environ)
        env.pop("QP_PIPELINE_ROOT", None)
        run = subprocess.run(
            [
                sys.executable, str(TOOLS / "dione_surface.py"), "dry-run",
                "--root", str(mock_q4),
                "--repo", "0xSero/GLM-5.3-Flash-EXL3-Q4",
                "--revision", "99cccdf0e8741715662c383828a9ea601990c125",
                "--skip-shard-hashes",
            ],
            capture_output=True, text=True, env=env,
        )
        assert run.returncode == 0, run.stderr
        summary = json.loads(run.stdout)
        assert summary["bits"] == 4 and summary["packed_modules"] == 36288
        assert summary["source_revision"] == "a6c167b62691b2bac901344b65cb651a70f53e43"

        run = subprocess.run(
            [
                sys.executable, str(TOOLS / "k6_student_capture.py"),
                "--surface", "dione", "--profile", "dione",
                "--dione-root", str(mock_q4),
                "--dione-repo", "0xSero/GLM-5.3-Flash-EXL3-Q4",
                "--dione-revision", "99cccdf0e8741715662c383828a9ea601990c125",
                "--skip-shard-hashes",
                "--bf16", str(mock_bf16),
                "--teacher", str(teacher),
                "--cold-run", "1",
                "--out", str(scratch / "out-dry"),
                "--pipeline-root", args.pipeline_root,
                "--dry-run",
            ],
            capture_output=True, text=True, env=env,
        )
        assert run.returncode == 0, run.stderr
        plan = json.loads(run.stdout.splitlines()[-1])
        assert plan["schema"] == ds.DIONE_PLAN_SCHEMA and plan["dry_run"] is True
        assert plan["bits"] == 4 and plan["windows"] == 2
        assert plan["seal_disclosure"] == ds.SEAL_DISCLOSURE
        passed.append(
            "5 dry-run: dione_surface CLI + k6_student_capture --surface dione "
            "reach plan-print on the REAL config/index"
        )
    else:
        passed.append("5 SKIPPED (real config/index absent)")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    for line in passed:
        print("PASS", line)
    print(json.dumps({"ok": True, "checks": len(passed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
