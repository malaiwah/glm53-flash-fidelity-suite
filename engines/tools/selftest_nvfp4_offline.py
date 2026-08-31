#!/usr/bin/env python3
"""Offline (no GPU, no weights download) validation of the NVFP4 surface adapter.

Proves, on this machine, in seconds:

  1. F8E4M3 LUT EXACTNESS - the 256-entry float8_e4m3fn -> float32 table the
     adapter uses (so the decode needs no float8 kernel, which MPS lacks) is
     BIT-IDENTICAL to torch's own native cast on all 254 finite codes, -0.0
     included; the two NaN codes are NaN in both, payloads left unasserted
     because IEEE-754 does not specify them.
  2. E2M1 NIBBLE ORDER - the adapter's unpack over ALL 256 byte values equals
     compressed-tensors' unpack_fp4_from_uint8 math (transliterated here, and
     re-checked against the LIVE package when it is importable), signbits
     included: nibble 0b1000 is -0.0 in both.
  3. DEQUANT KNOWN-ANSWERS - hand-computed values for both scale conventions
     (compressed-tensors' divide, modelopt's multiply), a group-axis probe that
     a transposed-scale regression cannot pass, six refusals, and - when MPS is
     present - bitwise CPU==MPS for the whole kernel, which is also the proof
     it uses no float64 (MPS has none).
  4. REAL-TENSOR CROSS-CHECK - the four committed fixtures (RedHatAI and
     LibertAIDAI, gate_proj and down_proj of layer 3 expert 0, ranged-fetched
     from the pinned revisions) decode BITWISE to their expected fp32, which
     compressed-tensors 0.18.0 produced; with the package present the reference
     is re-derived live rather than trusted.
  5. REAL-METADATA CENSUS - census_weight_map over the REAL indexes of both
     repos (148,498 and 150,226 tensors) closes at 36,288 main NVFP4 modules +
     864 MTP modules + 1,618 non-routed names that biject the official BF16
     non-routed set - and nine doctored indexes are each REFUSED BY NAME.
  6. SURFACE LOAD + IDENTITY - both real configs load to the right layout,
     scope policy and activation disclosure; a synthetic genuine-W4A16 index
     takes the "fully captured" branch; the identity hash is stable across
     loads and moves when a pin moves; seven malformed snapshots are refused.
  7. STREAMING SOURCE E2E - Nvfp4ExpertSource reads a synthetic shard written
     under the REAL index's shard names and returns exactly what dequant_nvfp4
     returns, with a receipt-grade census row and correct byte/shard counters,
     for both dialects.
  8. CLI - `nvfp4_surface.py dry-run` reaches plan-print on the real config +
     index of both repos, and refuses an unpinned revision and an unverified
     snapshot.
  9. STREAM_SCORE DRY-RUN - `stream_score.py --source nvfp4 --dry-run` reaches
     plan-print against the real metadata (SKIPped without --pipeline-root),
     and refuses --bf16 and a profile/source mismatch.
 10. REGISTRY ADAPTER - registry_add turns an nvfp4 summary into a row that
     carries the repo/revision pin, the measured scope and the seal and
     activation caveats verbatim, gives a genuine W4A16 artifact no caveat it
     has not earned, and refuses eight summaries with those blocks stripped.

Two rungs degrade to SKIP rather than failing, and say so on the line they
print: 2/4's live reference needs `compressed-tensors` (absent on the CUDA
boxes), and 9 needs a --pipeline-root whose quant_pipeline imports (python
3.11+, for tomllib).

Run:  python3 selftest_nvfp4_offline.py [--pipeline-root <tree-with-quant_pipeline>]
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
EVIDENCE = TOOLS / "nvfp4-evidence"

# compressed-tensors' own table (compressors/nvfp4/helpers.py kE2M1ToFloat).
KE2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def refuses(fragment, call, *args, **kwargs):
    """Assert `call` raises a ValueError whose message NAMES the problem."""
    try:
        call(*args, **kwargs)
    except ValueError as exc:
        assert fragment in str(exc), (
            "refusal did not mention %r; it said: %s" % (fragment, exc)
        )
        return str(exc)
    raise AssertionError("expected a refusal mentioning %r, got none" % (fragment,))


def refuses_any(fragment, call, *args, **kwargs):
    """Like `refuses`, for paths where the refusal comes from a library below us."""
    try:
        call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the point is that SOMETHING refuses
        assert fragment in str(exc), (
            "refusal did not mention %r; it said: %s" % (fragment, exc)
        )
        return str(exc)
    raise AssertionError("expected a refusal mentioning %r, got none" % (fragment,))


def ct_reference_unpack(packed, dtype):
    """Transliteration of compressed_tensors unpack_fp4_from_uint8 (0.18.0)."""
    import torch

    flat = packed.flatten()
    combined = torch.stack((flat & 0x0F, (flat & 0xF0) >> 4), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    magnitudes = (combined & 0x07).to(torch.long)
    table = torch.tensor(KE2M1, dtype=torch.float32)
    values = table[magnitudes] * torch.where(signs, -1.0, 1.0)
    return values.reshape(packed.shape[0], packed.shape[1] * 2).to(dtype=dtype)


def random_scale_bytes(rng, shape):
    """Random e4m3fn scale codes avoiding the two NaN codes (S.1111.111).

    A NaN scale is REFUSED by the decode (rung 3 proves it), so a fixture that
    happened to draw one would be testing the refusal, not the arithmetic.
    """
    values = rng.integers(0, 256, size=shape, dtype=np.uint8)
    values[values == 0x7F] = 0x38
    values[values == 0xFF] = 0xB8
    return values


def real_index(tag):
    return json.loads(gzip.decompress((EVIDENCE / ("%s-index.json.gz" % tag)).read_bytes()))


def mock_root(scratch, name, evidence=None, layout_config=None, weight_map=None):
    """A snapshot root carrying the REAL config + index (optionally doctored).

    `name` is the directory; `evidence` picks which repo's committed metadata
    backs it (defaults to `name`, which is how the two undoctored roots are built).
    """
    evidence = evidence or name
    root = scratch / name
    root.mkdir(parents=True, exist_ok=True)
    config = layout_config
    if config is None:
        config = json.loads((EVIDENCE / ("%s-config.json" % evidence)).read_text(encoding="utf-8"))
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    index = real_index(evidence) if weight_map is None else {"weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True), encoding="utf-8"
    )
    return root


def synthetic_module_shard(root, weight_map, names_to_tensors):
    """Write tensors into files named by the REAL index's shard names."""
    from safetensors.torch import save_file

    by_shard = {}
    for name, tensor in names_to_tensors.items():
        by_shard.setdefault(weight_map[name], {})[name] = tensor
    for shard, tensors in by_shard.items():
        path = root / shard
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))
    return sorted(by_shard)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", default=os.environ.get("QP_PIPELINE_ROOT"),
                        help="tree containing quant_pipeline; rung 9 SKIPs without it")
    parser.add_argument("--keep", action="store_true", help="keep the mock roots")
    args = parser.parse_args()
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))

    import torch

    import nvfp4_surface as ns

    rng = np.random.default_rng(0x4FB4)
    passed = []
    scratch = Path(tempfile.mkdtemp(prefix="nvfp4-selftest-"))
    try:
        # -------------------------------------------------------------- 1
        codes = torch.arange(256, dtype=torch.uint8)
        ours = ns.f8e4m3_to_float32(codes)
        native = codes.view(torch.float8_e4m3fn).to(torch.float32)
        # BIT PATTERNS, not float equality: this is what catches a +0.0 where the
        # format says -0.0 (code 0x80), which `==` would happily call equal.
        finite = ~torch.isnan(native)
        assert torch.equal(ours[finite].view(torch.int32), native[finite].view(torch.int32)), \
            "f8e4m3 LUT bit patterns != native cast"
        assert int(finite.sum()) == 254, "e4m3fn has 254 finite codes"
        assert float(ours[0x80]) == 0.0 and bool(ours[0x80].signbit()), "code 0x80 is -0.0"
        # The two NaN codes (S.1111.111) are NaN in both, with DIFFERENT payloads:
        # torch's cast keeps the sign and a signalling-range payload, the LUT emits
        # Python's canonical quiet NaN. IEEE-754 does not specify NaN payloads and
        # they vary by device, so equality is asserted on NaN-ness, never on bits -
        # and no NaN ever reaches arithmetic: dequant_nvfp4 refuses a NaN scale
        # before it is used (rung 3).
        assert torch.equal(torch.isnan(ours), torch.isnan(native)), "f8e4m3 NaN set differs"
        assert int(torch.isnan(ours).sum()) == 2, "e4m3fn has exactly two NaN codes (S.1111.111)"
        # the float8 tensor path (what safetensors hands back) agrees too
        assert torch.equal(
            ns.f8e4m3_to_float32(codes.view(torch.float8_e4m3fn))[finite].view(torch.int32),
            native[finite].view(torch.int32),
        ), "float8-typed input takes a different path"
        refuses("expects float8_e4m3fn/uint8/float32",
                ns.f8e4m3_to_float32, torch.zeros(4, dtype=torch.int16))
        passed.append("1 f8e4m3 LUT bit-identical to the native cast for all 254 finite codes "
                      "(-0.0 included); both NaN codes NaN, payloads unspecified")

        # -------------------------------------------------------------- 2
        all_bytes = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        ours = ns.unpack_e2m1(all_bytes)
        theirs = ct_reference_unpack(all_bytes, torch.float32)
        assert torch.equal(ours, theirs), "e2m1 unpack != compressed-tensors reference"
        assert torch.equal(ours.signbit(), theirs.signbit()), "e2m1 unpack signbits differ"
        # low nibble FIRST, and 0b1000 is negative zero, not positive
        one_byte = ns.unpack_e2m1(torch.tensor([[0x38]], dtype=torch.uint8))
        assert float(one_byte[0, 0]) == -0.0 and bool(one_byte[0, 0].signbit()), "0x8 != -0.0"
        assert float(one_byte[0, 1]) == 1.5, "high nibble 0x3 decodes SECOND, as magnitude[3]=1.5"
        assert sorted({abs(float(v)) for v in ours.flatten()}) == KE2M1, "e2m1 magnitude set"
        refuses("must be 2-D uint8", ns.unpack_e2m1, torch.zeros(4, 4, dtype=torch.int8))
        try:
            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

            live = unpack_fp4_from_uint8(all_bytes, 16, 32, dtype=torch.float32)
            assert torch.equal(ours, live), "e2m1 unpack != LIVE compressed-tensors"
            assert torch.equal(ours.signbit(), live.signbit()), "LIVE signbits differ"
            note = "vs LIVE compressed-tensors"
        except ImportError:
            note = "vs transliterated reference (compressed-tensors absent: live rung SKIPPED)"
        passed.append("2 e2m1 nibble order/LUT, all 256 byte codes, %s" % note)

        # -------------------------------------------------------------- 3
        # one row, two groups of 16: values 1.0 and -6.0 alternating
        packed = torch.full((1, 16), 0xE1, dtype=torch.uint8)  # low 0x1=+0.5, high 0xE=-4.0
        scale_bytes = torch.tensor([[0x38, 0x40]], dtype=torch.uint8)  # 1.0, 2.0 in e4m3
        scale = scale_bytes.view(torch.float8_e4m3fn)
        got = ns.dequant_nvfp4(packed, scale, weight_global_scale=torch.tensor([4.0]))
        want = torch.empty(1, 32, dtype=torch.float32)
        want[0, 0::2] = 0.5
        want[0, 1::2] = -4.0
        want[0, :16] *= 1.0 / 4.0
        want[0, 16:] *= 2.0 / 4.0
        assert torch.equal(got, want), "compressed-tensors convention known-answer"
        got_mo = ns.dequant_nvfp4(packed, scale, weight_scale_2=torch.tensor(0.25))
        assert torch.equal(got_mo, want), "modelopt convention known-answer"
        # group axis: scales must apply along the LAST axis, per 16 columns. A
        # transposed or row-broadcast scale cannot produce this pattern.
        assert float(got[0, 15]) == -4.0 * 1.0 / 4.0 and float(got[0, 16]) == 0.5 * 2.0 / 4.0, (
            "group boundary is not at column 16 - scales are not on the input axis"
        )
        refuses("exactly one of weight_global_scale", ns.dequant_nvfp4, packed, scale)
        refuses("exactly one of weight_global_scale", ns.dequant_nvfp4, packed, scale,
                weight_global_scale=torch.tensor([1.0]), weight_scale_2=torch.tensor(1.0))
        refuses("does not match packed geometry", ns.dequant_nvfp4, packed,
                scale.reshape(2, 1), weight_global_scale=torch.tensor([1.0]))
        refuses("is NaN or zero", ns.dequant_nvfp4, packed, scale,
                weight_global_scale=torch.tensor([0.0]))
        refuses("contains NaN", ns.dequant_nvfp4, packed,
                torch.full((1, 2), 0xFF, dtype=torch.uint8).view(torch.float8_e4m3fn),
                weight_global_scale=torch.tensor([1.0]))
        refuses("not a multiple of the NVFP4 group size", ns.dequant_nvfp4,
                torch.zeros(1, 4, dtype=torch.uint8), torch.zeros(1, 1, dtype=torch.uint8),
                weight_global_scale=torch.tensor([1.0]))
        mps_note = "MPS absent: device rung SKIPPED"
        if torch.backends.mps.is_available():
            big = torch.from_numpy(rng.integers(0, 256, size=(64, 128), dtype=np.uint8))
            sc = torch.from_numpy(random_scale_bytes(rng, (64, 16))).view(torch.float8_e4m3fn)
            gs = torch.tensor([17280.0])
            cpu = ns.dequant_nvfp4(big, sc, weight_global_scale=gs)
            mps = ns.dequant_nvfp4(big.to("mps"), sc.to("mps"), weight_global_scale=gs.to("mps"))
            assert mps.dtype == torch.float32, "MPS decode must stay fp32"
            assert torch.equal(cpu, mps.to("cpu")), "MPS decode differs from CPU"
            assert torch.equal(cpu.signbit(), mps.to("cpu").signbit()), "MPS signbits differ"
            mps_note = "CPU==MPS bitwise (no float64 anywhere in the kernel)"
        passed.append("3 dequant known-answers (both conventions), group axis, 6 refusals; %s"
                      % mps_note)

        # -------------------------------------------------------------- 4
        fixtures = sorted(EVIDENCE.glob("*-l3e0-*.pt"))
        assert len(fixtures) == 4, "expected 4 real-tensor fixtures, found %d" % len(fixtures)
        live_checked = 0
        for path in fixtures:
            fixture = torch.load(path, map_location="cpu")
            if fixture["layout"] == ns.LAYOUT_COMPRESSED_TENSORS:
                decoded = ns.dequant_nvfp4(
                    fixture["packed"], fixture["weight_scale"],
                    weight_global_scale=fixture["weight_global_scale"],
                )
            else:
                decoded = ns.dequant_nvfp4(
                    fixture["packed"], fixture["weight_scale"],
                    weight_scale_2=fixture["weight_scale_2"],
                )
            want = fixture["expected_fp32"]
            assert torch.equal(decoded, want), "real-tensor decode differs: %s" % path.name
            assert torch.equal(decoded.signbit(), want.signbit()), "signbits: %s" % path.name
            rows, cols = want.shape
            assert (rows, cols) == (fixture["rows"], ns.PROJECTION_SHAPE[fixture["projection"]][1])
            try:
                from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
            except ImportError:
                continue
            values = unpack_fp4_from_uint8(fixture["packed"], rows, cols, dtype=torch.float32)
            scale32 = fixture["weight_scale"].to(torch.float32)
            if fixture["layout"] == ns.LAYOUT_COMPRESSED_TENSORS:
                effective = scale32 / fixture["weight_global_scale"].to(torch.float32).reshape(())
            else:
                effective = scale32 * fixture["weight_scale_2"].to(torch.float32).reshape(())
            groups = cols // ns.GROUP_SIZE
            reference = (
                values.reshape(rows, groups, ns.GROUP_SIZE) * effective.reshape(rows, groups, 1)
            ).reshape(rows, cols)
            assert torch.equal(decoded, reference), "LIVE ct reference differs: %s" % path.name
            live_checked += 1
        passed.append(
            "4 real fetched tensors (RedHatAI + LibertAIDAI, gate+down of L3/E0) decode "
            "BITWISE to the compressed-tensors reference%s"
            % ("" if live_checked == 4 else " (committed fixtures; live package absent)")
        )

        # -------------------------------------------------------------- 5
        official = set(json.loads(
            (EVIDENCE / "official-nonrouted-names.json").read_text(encoding="utf-8")
        )["names"])
        assert len(official) == 1618
        censuses = {}
        for tag, layout, tensors, mtp in (
            ("redhat", ns.LAYOUT_COMPRESSED_TENSORS, 148498, "fp8-scale-pair"),
            ("libertai", ns.LAYOUT_MODELOPT, 150226, "nvfp4"),
        ):
            weight_map = real_index(tag)["weight_map"]
            assert len(weight_map) == tensors, "%s index size drifted" % tag
            retained, counts = ns.census_weight_map(weight_map, layout=layout)
            assert counts["nvfp4_main_modules"] == 42 * 288 * 3 == 36288
            assert counts["mtp_modules"] == 288 * 3 == 864
            assert counts["mtp_expert_format"] == mtp, "%s MTP format drifted" % tag
            assert counts["retained_tensors"] == 1618
            assert set(retained) == official, "%s non-routed set != official" % tag
            assert ns._verify_nonrouted_names(retained) == "official-name-bijection"
            main_formats = {counts["per_layer_format"][str(layer)]
                            for layer in ns.MAIN_ROUTED_LAYERS}
            assert main_formats == {"nvfp4"}, "%s main layers not all nvfp4" % tag
            censuses[tag] = (weight_map, layout, counts)

        base, layout, _ = censuses["redhat"]
        doctored = dict(base)
        for component in ns.CT_NVFP4_DECODE + ns.CT_NVFP4_ACTIVATION:
            doctored.pop(ns.component_name(3, 0, "gate_proj", component), None)
        refuses("expert modules absent", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        del doctored[ns.component_name(3, 0, "gate_proj", "weight_packed")]
        refuses("unrecognised component set", ns.census_weight_map, doctored, layout=layout)
        # packed components AND an unquantized original in the same module: the
        # component-set signature is what catches this (`weight` is a legitimate
        # component name in both dialects, so a name-presence test could not)
        doctored = dict(base)
        doctored[ns.component_name(3, 0, "gate_proj", "weight")] = "s.safetensors"
        refuses("unrecognised component set", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        doctored[ns.component_name(3, 0, "gate_proj", "weight_zero_point")] = "s.safetensors"
        refuses("unknown component", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        doctored[ns.component_name(2, 0, "gate_proj", "weight_packed")] = "s.safetensors"
        refuses("unexpected expert layer 2", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        doctored[ns.component_name(3, 288, "gate_proj", "weight_packed")] = "s.safetensors"
        refuses("expert index 288 >= 288", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        for component in ns.CT_NVFP4_DECODE + ns.CT_NVFP4_ACTIVATION:
            doctored.pop(ns.component_name(3, 0, "gate_proj", component), None)
        doctored[ns.component_name(3, 0, "gate_proj", "weight")] = "s.safetensors"
        doctored[ns.component_name(3, 0, "gate_proj", "weight_scale")] = "s.safetensors"
        refuses("mixes expert formats", ns.census_weight_map, doctored, layout=layout)
        doctored = dict(base)
        for expert in range(288):
            for projection in ns.PROJECTIONS:
                for component in ns.CT_NVFP4_DECODE + ns.CT_NVFP4_ACTIVATION:
                    doctored.pop(ns.component_name(3, expert, projection, component), None)
                doctored[ns.component_name(3, expert, projection, "weight")] = "s.safetensors"
                doctored[ns.component_name(3, expert, projection, "weight_scale")] = "s.safetensors"
        refuses("not NVFP4-packed", ns.census_weight_map, doctored, layout=layout)
        refuses("unknown layout", ns.census_weight_map, base, layout="awq")
        drifted = [name for name in official if name != "lm_head.weight"]
        refuses("not the official one", ns._verify_nonrouted_names, drifted)
        passed.append(
            "5 REAL-index census closes for both repos (148,498 / 150,226 tensors -> 36,288 "
            "main + 864 MTP + 1,618 official non-routed names); 9 doctored indexes refused BY NAME"
        )

        # -------------------------------------------------------------- 6
        surfaces = {}
        for tag, layout, config_format, mtp in (
            ("redhat", ns.LAYOUT_COMPRESSED_TENSORS, "mixed-precision", "fp8-scale-pair"),
            ("libertai", ns.LAYOUT_MODELOPT, "NVFP4", "nvfp4"),
        ):
            root = mock_root(scratch, tag)
            surface = ns.load_nvfp4_surface(
                root, repo="mock/%s" % tag, revision="a" * 40, require_shard_hashes=False
            )
            assert surface.layout == layout and surface.config_format == config_format
            assert surface.shard_hash_verification == "skipped"
            assert surface.nonrouted_verification == "official-name-bijection"
            assert surface.scope["mtp_expert_format"] == mtp
            assert surface.quant_weights["num_bits"] == 4
            assert surface.quant_weights["group_size"] == ns.GROUP_SIZE
            # both flagship repos ship activation scales, so neither is fully captured
            assert surface.activations["activation_scale_tensors_present"] is True
            assert surface.activations["weights_only_decode_captures_artifact_fully"] is False
            assert "NOT captured" in surface.activations["disclosure"] or \
                   "not captured" in surface.activations["disclosure"]
            summary = ns.surface_summary(surface)
            assert summary["schema"] == ns.NVFP4_SURFACE_SCHEMA
            assert summary["seal_disclosure"] == ns.SEAL_DISCLOSURE
            assert summary["scope_policy"]["quantized_scope"].startswith("routed experts only")
            identity = surface.checkpoint_identity_sha256()
            again = ns.load_nvfp4_surface(
                root, repo="mock/%s" % tag, revision="a" * 40, require_shard_hashes=False
            )
            assert again.checkpoint_identity_sha256() == identity, "identity not stable"
            moved = ns.load_nvfp4_surface(
                root, repo="mock/%s" % tag, revision="b" * 40, require_shard_hashes=False
            )
            assert moved.checkpoint_identity_sha256() != identity, "identity ignores the revision"
            surfaces[tag] = surface

        # a GENUINE W4A16 artifact: no declared input_activations, no activation
        # scale tensors. The disclosure must say the weights-only decode captures
        # it fully rather than emitting a caveat it has not earned.
        w4a16_config = json.loads(
            (EVIDENCE / "libertai-config.json").read_text(encoding="utf-8")
        )
        w4a16_map = {name: "model-00001-of-00001.safetensors" for name in official}
        for layer in ns.MAIN_ROUTED_LAYERS + (ns.MTP_LAYER,):
            for expert in range(288):
                for projection in ns.PROJECTIONS:
                    for component in ns.MO_NVFP4_DECODE:
                        w4a16_map[ns.component_name(layer, expert, projection, component)] = \
                            "model-00001-of-00001.safetensors"
        w4a16_root = mock_root(scratch, "w4a16", evidence="libertai", layout_config=w4a16_config,
                              weight_map=w4a16_map)
        w4a16 = ns.load_nvfp4_surface(w4a16_root, repo="mock/w4a16", revision="c" * 40,
                                      require_shard_hashes=False)
        assert w4a16.activations["activation_scale_tensors_present"] is False
        assert w4a16.activations["weights_only_decode_captures_artifact_fully"] is True
        assert w4a16.activations["declared_input_activations"] is None

        refuses("whole-shard sha256 verification marker absent",
                ns.load_nvfp4_surface, surfaces["redhat"].root, revision="a" * 40)
        refuses("must be the immutable 40-hex repo commit",
                ns.load_nvfp4_surface, surfaces["redhat"].root, revision="main",
                require_shard_hashes=False)
        stale = mock_root(scratch, "stale", evidence="redhat")
        (stale / "nvfp4-shards-verified.json").write_text(
            json.dumps({"schema": ns.NVFP4_SHARDS_VERIFIED_SCHEMA, "all_verified": True,
                        "manifest_sha256": "0" * 64}), encoding="utf-8"
        )
        refuses("stale/foreign nvfp4-shards-verified.json",
                ns.load_nvfp4_surface, stale, revision="a" * 40)
        broken = json.loads((EVIDENCE / "redhat-config.json").read_text(encoding="utf-8"))
        del broken["quantization_config"]
        refuses("no quantization_config block", ns.load_nvfp4_surface,
                mock_root(scratch, "noquant", evidence="redhat", layout_config=broken), require_shard_hashes=False)
        broken = json.loads((EVIDENCE / "redhat-config.json").read_text(encoding="utf-8"))
        broken["quantization_config"]["quant_method"] = "awq"
        refuses("neither compressed-tensors nor modelopt", ns.load_nvfp4_surface,
                mock_root(scratch, "awq", evidence="redhat", layout_config=broken), require_shard_hashes=False)
        broken = json.loads((EVIDENCE / "redhat-config.json").read_text(encoding="utf-8"))
        broken["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 32
        refuses("static symmetric 4-bit group-16", ns.load_nvfp4_surface,
                mock_root(scratch, "gs32", evidence="redhat", layout_config=broken), require_shard_hashes=False)
        broken = json.loads((EVIDENCE / "redhat-config.json").read_text(encoding="utf-8"))
        broken["text_config"]["n_routed_experts"] = 128
        refuses("official GLM5Next main/MTP geometry", ns.load_nvfp4_surface,
                mock_root(scratch, "geom", evidence="redhat", layout_config=broken), require_shard_hashes=False)
        passed.append(
            "6 surface load: both dialects, W4A16-fully-captured branch, identity stable and "
            "revision-sensitive, 7 malformed snapshots refused"
        )

        # -------------------------------------------------------------- 7
        for tag in ("redhat", "libertai"):
            surface = surfaces[tag]
            components = surface.decode_components(3)
            built = {}
            expected = {}
            for projection in ns.PROJECTIONS:
                out_features, in_features = ns.PROJECTION_SHAPE[projection]
                packed = torch.from_numpy(
                    rng.integers(0, 256, size=(out_features, in_features // 2), dtype=np.uint8)
                )
                scale = torch.from_numpy(
                    random_scale_bytes(rng, (out_features, in_features // ns.GROUP_SIZE))
                ).view(torch.float8_e4m3fn)
                scalar = torch.tensor([1.0 / 128.0], dtype=torch.float32)
                built[ns.component_name(3, 0, projection, components[0])] = packed
                built[ns.component_name(3, 0, projection, components[1])] = scale
                built[ns.component_name(3, 0, projection, components[2])] = scalar
                if surface.layout == ns.LAYOUT_COMPRESSED_TENSORS:
                    expected[projection] = ns.dequant_nvfp4(
                        packed, scale, weight_global_scale=scalar)
                else:
                    expected[projection] = ns.dequant_nvfp4(packed, scale, weight_scale_2=scalar)
            shards = synthetic_module_shard(surface.root, surface.weight_map, built)
            source = ns.Nvfp4ExpertSource(surface)
            total_bytes = 0
            for projection in ns.PROJECTIONS:
                decoded, row = source.load(layer=3, expert=0, projection=projection)
                assert torch.equal(decoded, expected[projection]), \
                    "%s source decode != dequant_nvfp4 (%s)" % (tag, projection)
                assert decoded.dtype == torch.float32
                assert tuple(decoded.shape) == ns.PROJECTION_SHAPE[projection]
                assert row["tensor"] == ns.official_name(3, 0, projection)
                assert row["format"] == "nvfp4-e2m1-gs16-%s" % surface.layout
                assert set(row["components"]) == set(components), "census row components"
                for component in components:
                    assert len(row["components"][component]["sha256"]) == 64
                total_bytes += row["bytes"]
            assert source.decoded_modules == 3
            assert source.bytes_read == total_bytes
            assert source.shards_read == set(shards)
            refuses("is not a streamed main routed layer",
                    source.load, layer=ns.MTP_LAYER, expert=0, projection="gate_proj")
            # a shard that is present but lost a tensor fails LOUDLY, naming it -
            # nothing in this path can quietly substitute zeros for a missing expert
            missing = refuses_any(
                ns.component_name(3, 1, "gate_proj", components[0]),
                source.load, layer=3, expert=1, projection="gate_proj",
            )
            assert "does not contain tensor" in missing.lower() or "not in weight_map" in missing
            # a shard whose bytes do not match the declared geometry is refused,
            # not silently reshaped
            bad_root = scratch / ("bad-%s" % tag)
            bad_root.mkdir()
            shutil.copy(surface.root / "config.json", bad_root / "config.json")
            shutil.copy(surface.root / "model.safetensors.index.json",
                        bad_root / "model.safetensors.index.json")
            wrong = dict(built)
            name = ns.component_name(3, 0, "gate_proj", components[0])
            wrong[name] = torch.zeros(2048, 64, dtype=torch.uint8)
            synthetic_module_shard(bad_root, surface.weight_map, wrong)
            bad_surface = ns.load_nvfp4_surface(bad_root, revision="a" * 40,
                                                require_shard_hashes=False)
            refuses("expected (2048, 2048)", ns.Nvfp4ExpertSource(bad_surface).load,
                    layer=3, expert=0, projection="gate_proj")
        passed.append(
            "7 Nvfp4ExpertSource E2E on synthetic shards under the REAL shard names: decode "
            "identity, census row, byte/shard counters, 3 refusals - both dialects"
        )

        # -------------------------------------------------------------- 8
        env = dict(os.environ)
        env.pop("QP_PIPELINE_ROOT", None)
        for tag in ("redhat", "libertai"):
            run = subprocess.run(
                [sys.executable, str(TOOLS / "nvfp4_surface.py"), "dry-run",
                 "--root", str(surfaces[tag].root),
                 "--repo", "mock/%s" % tag, "--revision", "a" * 40, "--skip-shard-hashes"],
                capture_output=True, text=True, env=env,
            )
            assert run.returncode == 0, run.stderr
            summary = json.loads(run.stdout)
            assert summary["schema"] == ns.NVFP4_SURFACE_SCHEMA
            assert summary["layout"] == surfaces[tag].layout
            assert summary["group_size"] == 16
            assert summary["scope_policy"]["counts"]["nvfp4_main_modules"] == 36288
            assert summary["activations"]["disclosure"]
            assert summary["checkpoint_identity_sha256"] == \
                surfaces[tag].checkpoint_identity_sha256()
        run = subprocess.run(
            [sys.executable, str(TOOLS / "nvfp4_surface.py"), "dry-run",
             "--root", str(surfaces["redhat"].root), "--revision", "a" * 40],
            capture_output=True, text=True, env=env,
        )
        assert run.returncode != 0 and "verification marker absent" in run.stderr
        run = subprocess.run(
            [sys.executable, str(TOOLS / "nvfp4_surface.py"), "dry-run",
             "--root", str(surfaces["redhat"].root), "--revision", "main",
             "--skip-shard-hashes"],
            capture_output=True, text=True, env=env,
        )
        assert run.returncode != 0 and "40-hex repo commit" in run.stderr
        passed.append("8 nvfp4_surface CLI dry-run reaches plan-print on both REAL indexes; "
                      "unpinned revision and unverified snapshot refused")

        # -------------------------------------------------------------- 9
        source_root = None
        if args.pipeline_root:
            for candidate in ("runtime/src", "src", "."):
                probe = Path(args.pipeline_root) / candidate / "quant_pipeline" / "__init__.py"
                if probe.is_file():
                    source_root = str((Path(args.pipeline_root) / candidate).resolve())
                    break
        if source_root is None:
            passed.append("9 SKIPPED (no --pipeline-root/QP_PIPELINE_ROOT with quant_pipeline: "
                          "stream_score cannot import its sealed helpers)")
        else:
            sys.path.insert(0, source_root)
            from quant_pipeline.core.artifacts import canonical_json, sha256_bytes, sha256_file

            teacher = scratch / "mock-teacher"
            teacher.mkdir()
            artifacts = []
            rows = []
            for index in range(2):
                tokens = rng.integers(0, 154880, size=64, dtype=np.int64)
                mask = np.ones(64, dtype=np.int64)
                mask[:index] = 0
                token_path = teacher / ("tokens-%d.npy" % index)
                mask_path = teacher / ("mask-%d.npy" % index)
                np.save(token_path, tokens, allow_pickle=False)
                np.save(mask_path, mask, allow_pickle=False)
                digests = {}
                for path in (token_path, mask_path):
                    digest = sha256_file(path)
                    artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                                      "sha256": digest})
                    digests[path.name] = digest
                rows.append({
                    "window_id": "final-%04d" % index,
                    "document_id": "selftest-doc-%d" % index,
                    "domain": "selftest", "role": "final",
                    "token_ids_sha256": digests[token_path.name],
                    "attention_mask_sha256": digests[mask_path.name],
                    "prediction_positions": int(
                        (np.asarray(mask[:-1], dtype=bool)
                         & np.asarray(mask[1:], dtype=bool)).sum()
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

            def stream_score(*extra):
                return subprocess.run(
                    [sys.executable, str(TOOLS / "stream_score.py"),
                     "--source", "nvfp4", "--profile", "nvfp4",
                     "--nvfp4-root", str(surfaces["redhat"].root),
                     "--nvfp4-repo", "RedHatAI/GLM-5.3-Flash-NVFP4",
                     "--nvfp4-revision", "36c184c6cda000a481711306df5adde42f63321a",
                     "--nvfp4-skip-shard-hashes",
                     "--teacher", str(teacher), "--cold-run", "1",
                     "--out", str(scratch / "out-dry"),
                     "--pipeline-root", args.pipeline_root, "--dry-run", *extra],
                    capture_output=True, text=True, env=env,
                )

            run = stream_score()
            assert run.returncode == 0, run.stderr
            plan = json.loads(run.stdout.splitlines()[-1])
            assert plan["dry_run"] is True and plan["weight_source"] == "nvfp4"
            assert plan["window_count"] == 2 and plan["bits"] == 4
            assert plan["inventory_sha256"] is None, "an NVFP4 run has no sealed inventory"
            # the identity the RUNNER computed in its own process must equal the one
            # this process computes from the same root and the same pins
            pinned = ns.load_nvfp4_surface(
                surfaces["redhat"].root, repo="RedHatAI/GLM-5.3-Flash-NVFP4",
                revision="36c184c6cda000a481711306df5adde42f63321a",
                require_shard_hashes=False,
            )
            assert plan["checkpoint_identity_sha256"] == pinned.checkpoint_identity_sha256()
            assert plan["checkpoint_identity_sha256"] != \
                surfaces["redhat"].checkpoint_identity_sha256(), \
                "identity must move with the repo/revision pins"
            assert plan["nonrouted_policy"] == \
                "quant_snapshot_bf16_parameters_official_name_set_unquantized_in_artifact"
            assert plan["main_routed_policy"] == \
                "streamed_exact_fp32_nvfp4_e2m1_gs16_decode_to_bf16_one_layer_resident"
            assert "fp8_scale_pair" in plan["mtp_policy"], plan["mtp_policy"]
            block = plan["streaming_disclosure"]["nvfp4"]
            assert block["schema"] == ns.NVFP4_SURFACE_SCHEMA
            assert block["nvfp4_revision"] == "36c184c6cda000a481711306df5adde42f63321a"
            assert block["scope_policy"]["counts"]["nvfp4_main_modules"] == 36288
            assert block["activations"]["weights_only_decode_captures_artifact_fully"] is False
            assert block["seal_disclosure"] == ns.SEAL_DISCLOSURE
            assert any("activation quantization" in item
                       for item in plan["streaming_disclosure"]["sealed_path_differences"])
            run = stream_score("--bf16", str(surfaces["libertai"].root))
            assert run.returncode != 0 and "--bf16 plays no role" in run.stderr
            run = subprocess.run(
                [sys.executable, str(TOOLS / "stream_score.py"),
                 "--source", "nvfp4", "--profile", "k6",
                 "--nvfp4-root", str(surfaces["redhat"].root),
                 "--nvfp4-revision", "36c184c6cda000a481711306df5adde42f63321a",
                 "--teacher", str(teacher), "--cold-run", "1",
                 "--out", str(scratch / "out-dry2"),
                 "--pipeline-root", args.pipeline_root, "--dry-run"],
                capture_output=True, text=True, env=env,
            )
            assert run.returncode != 0 and "must be used together" in run.stderr
            passed.append(
                "9 stream_score --source nvfp4 --dry-run reaches plan-print on the REAL "
                "config/index (scope + activation caveat in the plan); --bf16 and a "
                "profile/source mismatch refused"
            )
        # -------------------------------------------------------------- 10
        sys.path.insert(0, str(TOOLS.parent.parent / "registry" / "tools"))
        import registry_add as ra

        assert ra.NVFP4_SUMMARY == "malaiwah.glm53-nvfp4-packed-kld-summary.v1"
        assert ra.NVFP4_SUMMARY in ra.STREAM_SUMMARIES and ra.NVFP4_SUMMARY in ra.OWN_SCHEMAS
        assert ra.LANE_STATED_BY_SCHEMA[ra.NVFP4_SUMMARY] is None, \
            "the nvfp4 family name carries no lane marker; --lane must supply it"

        def summary_receipt(**overrides):
            receipt = {
                "schema": ra.NVFP4_SUMMARY,
                "profile": "nvfp4-stream",
                "student_label": ns.NVFP4_STUDENT_LABEL,
                "measured_mean_kld": 0.0304,
                "run_means": [0.0304, 0.0304],
                "cold_run_count": 2,
                "distinct_tokenwise_kld_sha256": ["b" * 64],
                "bitwise_deterministic": True,
                "nvfp4_repo": "RedHatAI/GLM-5.3-Flash-NVFP4",
                "nvfp4_revision": "36c184c6cda000a481711306df5adde42f63321a",
                "seal_disclosure": ns.SEAL_DISCLOSURE,
                "scope_policy": dict(surfaces["redhat"].scope),
                "activations": dict(surfaces["redhat"].activations),
            }
            receipt.update(overrides)
            return [(receipt, str(scratch / "summary.json"), "c" * 64)]

        adapted = ra.adapt_stream_summary(summary_receipt())
        assert adapted["receipt_schema"] == ra.NVFP4_SUMMARY
        assert adapted["lane"] is None and adapted["requires_lane"] is True
        assert adapted["artifact_repo"] == "RedHatAI/GLM-5.3-Flash-NVFP4"
        assert adapted["artifact_revision"] == "36c184c6cda000a481711306df5adde42f63321a"
        codes = [d["code"] for d in adapted["verbatim_disclosure_coded"]]
        assert codes == ["unsealed_source", "quantization_scope",
                         "activation_quantization_not_captured"], codes
        assert ns.SEAL_DISCLOSURE in adapted["verbatim_disclosure_coded"][0]["detail"]
        assert "routed experts only" in adapted["verbatim_disclosure_coded"][1]["detail"]

        # a genuine W4A16 artifact earns NO activation caveat
        adapted = ra.adapt_stream_summary(summary_receipt(activations=dict(w4a16.activations)))
        codes = [d["code"] for d in adapted["verbatim_disclosure_coded"]]
        assert codes == ["unsealed_source", "quantization_scope"], codes

        for overrides, fragment in (
            ({"scope_policy": {}}, "no /scope_policy"),
            ({"seal_disclosure": None}, "no /seal_disclosure"),
            ({"activations": {}}, "no /activations"),
            ({"nvfp4_revision": "main"}, "immutable 40-hex repo commit"),
            ({"nvfp4_revision": "z" * 40}, "immutable 40-hex repo commit"),
            ({"activations": {"disclosure": "x"}}, "weights_only_decode_captures_artifact_"),
            ({"activations": {"weights_only_decode_captures_artifact_fully": False}},
             "activations.disclosure is missing"),
            ({"scope_policy": {"quantized_scope": "routed experts only"}},
             "no quantized_scope/nonrouted_policy"),
        ):
            try:
                ra.adapt_stream_summary(summary_receipt(**overrides))
            except ra.Refuse as exc:
                assert fragment in str(exc), "refusal did not mention %r: %s" % (fragment, exc)
            else:
                raise AssertionError("registry_add accepted %r" % (overrides,))
        passed.append(
            "10 registry_add adapts the nvfp4 summary family (repo/revision pinned, scope + "
            "seal + activation caveats verbatim, W4A16 earns none) and refuses 8 stripped ones"
        )
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            print("kept: %s" % scratch)

    for line in passed:
        print("  PASS  %s" % line)
    print("nvfp4 offline selftest: %d rungs" % len(passed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
