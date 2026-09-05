#!/usr/bin/env python3
"""Offline (no GPU, no weight download, no network) validation of the GGUF surface adapter.

Proves, on this machine, in seconds:

  1. REFERENCE EQUALITY - the adapter's plain-torch block kernels are BITWISE
     equal to gguf-py 0.19.0's reference ``dequantize`` on REAL ranged-fetched
     bytes of unsloth/GLM-5.3-Flash-GGUF UD-Q4_K_XL: Q4_K and Q5_K (the two
     types with the 12-byte 6-bit packed sub-block scales/mins), Q6_K (16-way
     int8 sub-scales) and Q8_0.  The reference outputs are COMMITTED, so the
     rung runs with gguf-py absent; when gguf-py IS importable the reference is
     recomputed live and the committed copy is proven to be its output.
  2. INDEPENDENT SCALE UNPACK - a scalar transliteration of llama.cpp's
     ``get_scale_min_k4`` (written from the C, not from the vectorized kernel)
     reproduces the adapter's Q4_K output element for element on those same
     real bytes.  This is what catches a sub-block scale bug that a
     same-code-twice comparison cannot.
  3. CONTAINER + REFUSALS - a synthesized llama.cpp v3 SPLIT is parsed back
     (per-part tensor tables, union table, split.tensors.count), and every
     refusal fires by NAME: an unmapped tensor, an unsupported ggml type, two
     alias spellings colliding on one official tensor, a wrong architecture, a
     wrong geometry, a missing split part, an unpinned revision.
  4. REAL-METADATA CENSUS - the REAL 1,412-tensor table closes (1,259 direct +
     129 fused expert + 24 MLA halves) and its 1,271 official names EXACTLY
     biject the real official BF16 index (38,770 tensors, a6c167b6) minus the
     37,152 routed and the 347 vision tensors.  The REAL ddh0 table refuses by
     type (IQ3_S/IQ4_XS) while every one of its names still maps - proving the
     second convert vintage's alias spellings are covered, not silently dropped.
  5. MLA RECONSTRUCTION - ``audit_mla_placement`` re-runs on REAL committed
     bytes (a leading head window of blk.3.attn_k_b/attn_v_b) against the REAL
     official BF16 kv_b_proj rows, and the shipped arrangement must win by the
     margin the adapter demands.
  5b. EXPERT SLOT ORDERING - the fused tensor's slot 0 is proven to BE official
     expert 0 (``audit_expert_placement`` against the committed BF16 payload in
     dione-evidence), which also settles the reversed-dims orientation and the
     ffn_gate_exps -> gate_proj mapping. A permuted expert order would decode
     cleanly, close every census, and measure the wrong model.
  6. EXPERT SLICE + MATERIALIZED VIEW - one routed expert sliced out of a fused
     tensor equals a direct decode of that byte range; the materialized
     non-routed view writes the official names/dtypes, round-trips bitwise, and
     is REUSED on a second call via its fingerprint stamp.
  7. DRY-RUN - ``gguf_surface.py dry-run`` reaches its plan print against a
     header-only GGUF carrying the REAL tensor table and the REAL metadata KVs,
     and ``stream_score.py --source gguf --dry-run`` does the same end to end
     (SKIPPED, with the reason printed, when quant_pipeline is not importable).

Run:  python3 engines/tools/selftest_gguf_offline.py
      python3 engines/tools/selftest_gguf_offline.py --pipeline-root <tree>   # adds rung 7b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
EVIDENCE = TOOLS / "gguf-evidence"
DIONE_EVIDENCE = TOOLS / "dione-evidence"
sys.path.insert(0, str(TOOLS))

import gguf_surface as gs  # noqa: E402


# ---------------------------------------------------------------------------
# a minimal GGUF WRITER (the adapter only reads; a writer is what lets the
# refusal rungs build the malformed artifacts they have to refuse)
# ---------------------------------------------------------------------------
_TYPE_ID = {name: tid for tid, name in gs.GGML_TYPE_IDS.items()}


def _kv_bytes(key: str, value) -> bytes:
    def _string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def _scalar(item) -> "tuple":
        if isinstance(item, bool):
            return 7, struct.pack("<B", int(item))
        if isinstance(item, int):
            if 0 <= item < (1 << 32):
                return 4, struct.pack("<I", item)
            return 11, struct.pack("<q", item)
        if isinstance(item, float):
            return 6, struct.pack("<f", item)
        if isinstance(item, str):
            return 8, _string(item)
        raise TypeError("unsupported KV value %r" % (item,))

    out = _string(key)
    if isinstance(value, list):
        if not value:
            return out + struct.pack("<II", 9, 4) + struct.pack("<Q", 0)
        element_type, _ = _scalar(value[0])
        body = b"".join(_scalar(item)[1] for item in value)
        return out + struct.pack("<I", 9) + struct.pack("<I", element_type) \
            + struct.pack("<Q", len(value)) + body
    type_id, body = _scalar(value)
    return out + struct.pack("<I", type_id) + body


def write_gguf(path: Path, kv: dict, rows: list, *, data: bytes = b"",
               version: int = 3, alignment: int = 32, magic: bytes = b"GGUF") -> Path:
    """Write a GGUF file. ``rows`` are {name, dims, type, offset} (offsets relative
    to data_start).  With ``data=b""`` the result is a HEADER-ONLY artifact: every
    census/dry-run path works on it, and no weight bytes exist to read."""
    head = magic + struct.pack("<I", version) + struct.pack("<QQ", len(rows), len(kv))
    head += b"".join(_kv_bytes(key, kv[key]) for key in kv)
    for row in rows:
        name = row["name"].encode("utf-8")
        head += struct.pack("<Q", len(name)) + name
        head += struct.pack("<I", len(row["dims"]))
        head += b"".join(struct.pack("<Q", int(d)) for d in row["dims"])
        head += struct.pack("<I", _TYPE_ID[row["type"]])
        head += struct.pack("<Q", int(row["offset"]))
    pad = (-len(head)) % alignment
    path.write_bytes(head + b"\0" * pad + data)
    return path


def _real_rows(name: str) -> dict:
    rows = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    for row in rows.values():
        row.pop("file", None)
    return rows


def _real_kv() -> dict:
    kv = json.loads((EVIDENCE / "unsloth-udq4kxl-kv.json").read_text(encoding="utf-8"))
    return {k: v for k, v in kv.items() if not k.startswith("tokenizer.")}


def _rows_for_writer(rows: dict) -> list:
    return [{"name": row["name"], "dims": row["dims"], "type": row["type"],
             "offset": row["offset"]} for row in sorted(rows.values(), key=lambda r: r["name"])]


def _refuses(fn, *needles) -> str:
    """Call fn(); require a gguf_surface refusal whose message names each needle."""
    try:
        fn()
    except ValueError as error:
        message = str(error)
        missing = [needle for needle in needles if needle not in message]
        if missing:
            raise AssertionError("refusal did not name %r: %s" % (missing, message))
        return message
    raise AssertionError("expected a refusal, got none")


# ---------------------------------------------------------------------------
# rung 2: llama.cpp's get_scale_min_k4 + Q4_K dequant, transliterated SCALAR
# from the C source -- deliberately not derived from the vectorized kernel.
# ---------------------------------------------------------------------------
def _f16(raw: bytes) -> float:
    return float(np.frombuffer(raw, dtype=np.float16)[0])


def q4k_dequant_scalar(block: bytes) -> "list":
    assert len(block) == 144
    d, dmin = _f16(block[0:2]), _f16(block[2:4])
    q = block[4:16]
    scales, mins = [], []
    for j in range(8):
        if j < 4:
            scales.append(q[j] & 63)
            mins.append(q[j + 4] & 63)
        else:
            scales.append((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4))
            mins.append((q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    qs = block[16:]
    out = []
    for group in range(4):
        chunk = qs[group * 32:(group + 1) * 32]
        for half in range(2):
            sub = group * 2 + half
            scale = np.float32(d) * np.float32(scales[sub])
            offset = np.float32(dmin) * np.float32(mins[sub])
            for byte in chunk:
                nibble = (byte & 0x0F) if half == 0 else (byte >> 4)
                out.append(np.float32(scale * np.float32(nibble) - offset))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-root",
                        help="tree carrying quant_pipeline; enables the stream_score dry-run rung")
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()
    import torch

    passed = []
    scratch = Path(tempfile.mkdtemp(prefix="gguf-selftest-"))

    # ---- 1. reference equality vs gguf-py ---------------------------------
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    try:
        import gguf as gguf_py
        from gguf.quants import dequantize as gguf_dequantize
        have_gguf = True
    except Exception:  # noqa: BLE001 - an optional cross-check reference
        have_gguf = False
    for qtype, spec in sorted(manifest["dequant_fixtures"].items()):
        raw = np.load(EVIDENCE / ("dequant_%s_blocks.npy" % qtype.lower())).tobytes()
        committed = np.load(EVIDENCE / ("dequant_%s_ggufpy_ref.npy" % qtype.lower()))
        n = spec["rows"] * spec["cols"]
        mine = gs.dequant_bytes(qtype, raw, n).reshape(spec["rows"], spec["cols"]).numpy()
        assert mine.dtype == np.float32, "%s decoded to %s, not float32" % (qtype, mine.dtype)
        assert np.array_equal(mine.view(np.uint32), committed.view(np.uint32)), (
            "%s is not bitwise equal to the committed gguf-py reference" % qtype)
        if have_gguf:
            live = gguf_dequantize(
                np.frombuffer(raw, dtype=np.uint8),
                getattr(gguf_py.GGMLQuantizationType, qtype),
            ).astype(np.float32).reshape(spec["rows"], spec["cols"])
            assert np.array_equal(live.view(np.uint32), committed.view(np.uint32)), (
                "the committed %s reference is NOT what this gguf-py produces" % qtype)
    passed.append(
        "1 reference equality: %s bitwise == committed gguf-py output on real "
        "unsloth bytes (Flash UD-Q4_K_XL; flagship UD-Q3_K_XL/UD-IQ4_XS)%s"
        % (", ".join(sorted(manifest["dequant_fixtures"])),
           " (and recomputed live from gguf-py)" if have_gguf
           else " (gguf-py absent: live recompute SKIPPED)"))

    # ---- 1b. accelerator decode parity -------------------------------------
    # The GGUF lane costs 23.7 min/window because the dequant runs on the CPU at
    # ~39 ms/matrix while the GPU sits at 2-4% (docs/GGUF-MEASUREMENT.md).
    # `dequant_bytes(..., device=)` moves the QUANTIZED bytes instead of the
    # 7.1x-larger fp32 result and runs the same kernels there.  That is only
    # allowed if it changes nothing, so this rung demands `torch.equal` -- not
    # allclose, not a tolerance -- against the CPU output rung 1 just proved
    # bitwise-equal to gguf-py, on the same REAL ranged-fetched UD-Q4_K_XL
    # bytes.
    #
    # It runs on whatever accelerator the host has: MPS on the laptop, CUDA on
    # the rented box.  bin/BUNDLE.txt ships this file precisely so the CUDA case
    # is checked ON THE INSTANCE, before the capture is paid for, rather than
    # inferred from the MPS case -- the same reason the exl3hf and mlx offline
    # selftests travel with their surfaces.
    accel = None
    if torch.cuda.is_available():
        accel = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accel = "mps"
    if accel is None:
        passed.append(
            "1b accelerator decode parity: SKIPPED (no CUDA and no MPS on this "
            "host; the check RUNS on the instance, which is where it counts)")
    else:
        checked = []
        for qtype, spec in sorted(manifest["dequant_fixtures"].items()):
            raw = np.load(EVIDENCE / ("dequant_%s_blocks.npy" % qtype.lower())).tobytes()
            n = spec["rows"] * spec["cols"]
            reference = gs.dequant_bytes(qtype, raw, n)
            fast = gs.dequant_bytes(qtype, raw, n, device=accel)
            assert str(fast.device).split(":")[0] == accel, (
                "%s: device= did not decode on %s (got %s)" % (qtype, accel, fast.device))
            assert fast.dtype == torch.float32
            assert torch.equal(fast.cpu(), reference), (
                "%s decoded on %s is NOT bitwise equal to the cpu reference "
                "(max|d| %g) -- the fast path may not ship"
                % (qtype, accel, float((fast.cpu() - reference).abs().max())))
            checked.append(qtype)
        passed.append(
            "1b accelerator decode parity: %s decoded on %s are torch.equal to "
            "the cpu reference on real UD-Q4_K_XL bytes (%d elements)"
            % (", ".join(checked), accel,
               sum(s["rows"] * s["cols"] for s in manifest["dequant_fixtures"].values())))

    # ---- 2. independent scalar scale unpack --------------------------------
    raw_q4k = np.load(EVIDENCE / "dequant_q4_k_blocks.npy").tobytes()
    mine = gs.dequant_bytes("Q4_K", raw_q4k, len(raw_q4k) // 144 * 256).numpy()
    scalar = []
    for index in range(len(raw_q4k) // 144):
        scalar.extend(q4k_dequant_scalar(raw_q4k[index * 144:(index + 1) * 144]))
    scalar_np = np.asarray(scalar, dtype=np.float32)
    assert np.array_equal(mine.view(np.uint32), scalar_np.view(np.uint32)), (
        "the vectorized Q4_K kernel disagrees with a scalar transliteration of "
        "llama.cpp get_scale_min_k4 -- max|d| %g" % float(np.abs(mine - scalar_np).max()))
    passed.append(
        "2 independent scale unpack: %d Q4_K blocks (%d elements, 8 sub-block "
        "scales+mins each) match a scalar get_scale_min_k4 transliteration bitwise"
        % (len(raw_q4k) // 144, mine.size))

    # ---- 3. container + refusals -------------------------------------------
    rows = _real_rows("unsloth-udq4kxl-tensors.json")
    kv = _real_kv()
    writer_rows = _rows_for_writer(rows)
    part_a = [r for r in writer_rows if not r["name"].startswith("blk.4")]
    part_b = [r for r in writer_rows if r["name"].startswith("blk.4")]
    kv_a = dict(kv, **{"split.no": 0, "split.count": 2, "split.tensors.count": len(writer_rows)})
    kv_b = {"general.architecture": kv["general.architecture"], "split.no": 1,
            "split.count": 2, "split.tensors.count": len(writer_rows)}
    split_dir = scratch / "split"
    split_dir.mkdir()
    a = write_gguf(split_dir / "part-00001-of-00002.gguf", kv_a, part_a)
    b = write_gguf(split_dir / "part-00002-of-00002.gguf", kv_b, part_b)
    surface = gs.load_gguf_surface([str(a), str(b)], repo="unsloth/GLM-5.3-Flash-GGUF",
                                   revision="2975ab414d30340466d8c51533c6e91f0cca64c1",
                                   require_file_hashes=False)
    assert len(surface.container.tensors) == len(rows)
    assert [f.name for f in surface.container.files] == [a.name, b.name]
    assert surface.container.architecture == "glm5next"

    _refuses(lambda: gs.load_gguf_surface([str(a)], require_file_hashes=False),
             "split artifact needs all 2 parts")
    _refuses(lambda: gs.load_gguf_surface([str(a), str(b)], require_file_hashes=False,
                                          revision="not-a-commit"),
             "immutable 40-hex repo commit")

    single_dir = scratch / "single"
    single_dir.mkdir()

    def _one(name, mutate_rows=None, mutate_kv=None):
        local_rows = [dict(r) for r in writer_rows]
        local_kv = dict(kv)
        local_kv.pop("split.no", None)
        local_kv.pop("split.count", None)
        local_kv.pop("split.tensors.count", None)
        if mutate_rows:
            mutate_rows(local_rows)
        if mutate_kv:
            mutate_kv(local_kv)
        return write_gguf(single_dir / name, local_kv, local_rows)

    good = _one("good.gguf")
    gs.load_gguf_surface([str(good)], require_file_hashes=False)

    bad = _one("unmapped.gguf",
               mutate_rows=lambda r: r.append({"name": "blk.3.mystery_proj.weight",
                                               "dims": [4096, 4096], "type": "F32",
                                               "offset": 0}))
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "no glm5next->HF mapping", "blk.3.mystery_proj.weight")

    def _retype(local_rows):
        for row in local_rows:
            if row["name"] == "blk.7.ffn_gate_exps.weight":
                row["type"] = "IQ2_XS"

    bad = _one("iq.gguf", mutate_rows=_retype)
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "v1 decode kernel", "IQ2_XS", "blk.7.ffn_gate_exps.weight")

    bad = _one("alias.gguf",
               mutate_rows=lambda r: r.append({"name": "blk.11.indexer.kpool_ape.weight",
                                               "dims": [4096], "type": "F32", "offset": 0}))
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "map to the same official tensor", "index_kpool_compress_ape")

    bad = _one("arch.gguf", mutate_kv=lambda k: k.update({"general.architecture": "llama"}))
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "general.architecture", "llama")

    bad = _one("geom.gguf", mutate_kv=lambda k: k.update({"glm5next.expert_count": 128}))
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "geometry gate", "expert_count")

    def _shrink(local_rows):
        # the TRANSPOSE of the true [in=4096, out=2048, experts=288]: a fused
        # tensor whose element count is right and whose axes are not
        for row in local_rows:
            if row["name"] == "blk.9.ffn_up_exps.weight":
                row["dims"] = [2048, 4096, 288]

    bad = _one("dims.gguf", mutate_rows=_shrink)
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False),
             "blk.9.ffn_up_exps.weight", "expected [in=4096, out=2048, experts=288]")

    _refuses(lambda: gs.load_gguf_surface([str(good)], require_file_hashes=True),
             "whole-file sha256 marker absent", "verify-files")
    marker = gs.verify_file_hashes([str(good)])
    assert marker["all_hashed"] and len(marker["files"]) == 1
    hashed = gs.load_gguf_surface([str(good)], require_file_hashes=True)
    assert hashed.file_hash_verification == "full"
    assert hashed.file_records[0]["sha256"] == marker["files"][0]["sha256"]
    passed.append(
        "3 container + refusals: a 2-part v3 split round-trips (%d tensors, union table "
        "checked against split.tensors.count) and 8 refusals fire BY NAME (unmapped tensor, "
        "unsupported type, alias collision, wrong arch, wrong geometry, wrong dims, missing "
        "split part, unpinned revision); the sha256 marker gate opens only after verify-files"
        % len(rows))

    # ---- 4. real-metadata census + bijection --------------------------------
    census = surface.census
    assert len(census.direct_map) == 1259, len(census.direct_map)
    assert len(census.routed) == 129, len(census.routed)
    assert len(census.mla) == 24 and len(census.mla_layers) == 12
    assert (len(census.direct_map) + len(census.routed) + len(census.mla)) == len(rows)
    official = json.loads((DIONE_EVIDENCE / "bf16-index.json").read_text(encoding="utf-8"))
    bijection = gs.verify_nonrouted_bijection(census, official["weight_map"].keys())
    assert bijection["bijection_ok"]
    assert bijection["official_tensors"] == 38770
    assert bijection["official_routed_tensors"] == 37152
    assert bijection["official_vision_tensors"] == 347
    assert bijection["nonrouted_mapped_tensors"] == 1271
    assert bijection["mla_reconstructed_tensors"] == 12
    scope = surface.scope_policy
    assert scope["embeddings_type"] == "Q8_0" and scope["lm_head_type"] == "Q8_0"
    assert scope["attention_kda_dsa_quantized"] is True
    assert scope["vision_in_artifact"] is False
    assert sorted(scope["routed_expert_types"]) == ["Q4_K", "Q5_K", "Q6_K"]

    # which of the repo's twelve builds v1 can actually score, from each
    # build's OWN type census (not from its directory name -- unsloth's
    # "Dynamic" recipe mixes IQ types into UD-Q2_K_XL and UD-Q3_K_XL, so a
    # name-based answer would be wrong in both directions)
    build_census = json.loads(
        (EVIDENCE / "unsloth-build-census.json").read_text(encoding="utf-8"))
    assert sorted(build_census["v1_supported_types"]) == sorted(gs.SUPPORTED_TYPES), (
        "the committed build census was taken against a different supported-type set")
    supported_builds, refused_builds = [], []
    for name, row in sorted(build_census["builds"].items()):
        assert row["census_complete"], "%s census did not close at 1412 tensors" % name
        recomputed = not [t for t in row["types"] if t not in gs.SUPPORTED_TYPES]
        assert recomputed == row["v1_supported"], (
            "%s: the census says v1_supported=%r but recomputing from its own type list "
            "gives %r" % (name, row["v1_supported"], recomputed))
        (supported_builds if recomputed else refused_builds).append(name)
    assert supported_builds == ["BF16", "Q8_0", "UD-IQ4_XS", "UD-Q3_K_XL", "UD-Q4_K_XL",
                                "UD-Q5_K_XL", "UD-Q6_K_XL"], supported_builds
    assert "UD-Q2_K_XL" in refused_builds and "UD-IQ3_XXS" in refused_builds, (
        "UD-Q2_K_XL (IQ2_XS/Q2_K) and UD-IQ3_XXS (IQ2_S/Q2_K) carry types without a kernel "
        "and must refuse; UD-Q3_K_XL is admitted only because IQ3_XXS/IQ4_XS/Q3_K landed")

    ddh0_rows = _real_rows("ddh0-tensors.json")
    ddh0_names = {name: gs.classify_tensor(name)[0] for name in ddh0_rows}
    unmapped = sorted(name for name, role in ddh0_names.items() if role == "unmapped")
    assert not unmapped, "ddh0 names left unmapped: %s" % unmapped[:5]
    ddh0 = _one("ddh0.gguf", mutate_rows=lambda r: r.__setitem__(
        slice(None), _rows_for_writer(ddh0_rows)),
        mutate_kv=lambda k: k.update({"general.architecture": "glm5-next"}))
    # the arch string differs, so the geometry keys must be renamed with it
    ddh0_kv = {("glm5-next." + key.split(".", 1)[1]) if key.startswith("glm5next.") else key: value
               for key, value in kv.items()}
    ddh0_kv.pop("split.no", None)
    ddh0_kv.pop("split.count", None)
    ddh0_kv.pop("split.tensors.count", None)
    ddh0_kv["general.architecture"] = "glm5-next"
    ddh0_dir = scratch / "ddh0"
    ddh0_dir.mkdir()
    ddh0 = write_gguf(ddh0_dir / "ddh0.gguf", ddh0_kv, _rows_for_writer(ddh0_rows))
    # ddh0 (IQ3_S/IQ4_XS experts, arch spelled glm5-next) was v1-REFUSED by
    # type; with the IQ3_S/IQ4_XS kernels landed it LOADS, and its census must
    # close on the same 1,412 names through the second convert vintage's aliases
    ddh0_surface = gs.load_gguf_surface([str(ddh0)], require_file_hashes=False)
    assert ddh0_surface.container.architecture == "glm5-next"
    assert sorted(ddh0_surface.scope_policy["routed_expert_types"]) == ["IQ3_S", "IQ4_XS"], \
        ddh0_surface.scope_policy["routed_expert_types"]
    assert len(ddh0_surface.census.direct_map) == 1259
    passed.append(
        "4 real-metadata census: 1,412 GGUF tensors close (1,259 direct + 129 fused + 24 MLA) "
        "and their 1,271 official names biject the real BF16 index (38,770 - 37,152 routed - "
        "347 vision); ddh0's second convert vintage maps 1,412/1,412 names (arch spelled "
        "glm5-next) and now loads on the IQ3_S/IQ4_XS kernels; of unsloth's 12 builds the "
        "adapter scores %s and refuses %s (types without a kernel)"
        % (", ".join(supported_builds), ", ".join(refused_builds)))

    # ---- 5. MLA placement audit on real bytes -------------------------------
    k_raw = np.load(EVIDENCE / "mla_k_b_head_window.npy").tobytes()
    v_raw = np.load(EVIDENCE / "mla_v_b_head_window.npy").tobytes()
    spec = manifest["mla_fixture"]
    heads = int(spec["heads_in_fixture"])
    official_rows = torch.from_numpy(
        np.load(EVIDENCE / "mla_official_row_window_bf16.npy")).view(torch.bfloat16)

    class _Window:
        tensors = {
            "blk.3.attn_k_b.weight": {"name": "blk.3.attn_k_b.weight", "type": "Q8_0",
                                      "dims": [256, 512, heads],
                                      "elements": heads * 512 * 256, "file": "w"},
            "blk.3.attn_v_b.weight": {"name": "blk.3.attn_v_b.weight", "type": "Q8_0",
                                      "dims": [512, 256, heads],
                                      "elements": heads * 512 * 256, "file": "w"},
        }
        _raw = {"blk.3.attn_k_b.weight": k_raw, "blk.3.attn_v_b.weight": v_raw}

        def read_tensor(self, name):
            return self._raw[name]

    window_census = gs.GgufCensus(
        direct_map={}, routed={},
        mla={(3, "k_b"): "blk.3.attn_k_b.weight", (3, "v_b"): "blk.3.attn_v_b.weight"},
        mla_layers=(3,))
    audit = gs.audit_mla_placement(_Window(), window_census, layer=3,
                                  official_bf16=official_rows)
    assert audit["passed"] and audit["winner"] == gs.MLA_KV_B_ARRANGEMENT
    shipped = audit["candidates"][gs.MLA_KV_B_ARRANGEMENT]
    runner_up = audit["best_other_rel_l2"]
    assert shipped["cosine"] > 0.999 and shipped["rel_l2"] < 0.01
    # scale-free: the gap that matters is rel-L2, and it must hold on a WINDOW
    # (a cosine margin would not -- see the note in audit_mla_placement)
    assert runner_up > 0.5 and runner_up > 100 * shipped["rel_l2"]
    # the committed record of the SAME audit over the complete 64-head tensor
    full_audit = json.loads((EVIDENCE / "mla-full-audit.json").read_text(encoding="utf-8"))
    assert full_audit["passed"] and full_audit["winner"] == gs.MLA_KV_B_ARRANGEMENT
    assert full_audit["candidates"][gs.MLA_KV_B_ARRANGEMENT]["cosine"] > 0.999
    passed.append(
        "5 MLA reconstruction: on real heads 0-%d of blk.3.attn_k_b/attn_v_b vs the real "
        "official kv_b_proj rows, %s scores rel-L2 %.4f (= the Q8_0 error) against %.4f for "
        "the closest wrong arrangement; the committed full 64-head audit agrees (%.4f vs %.4f)"
        % (heads - 1, gs.MLA_KV_B_ARRANGEMENT, shipped["rel_l2"], runner_up,
           full_audit["candidates"][gs.MLA_KV_B_ARRANGEMENT]["rel_l2"],
           min(score["rel_l2"] for name, score in full_audit["candidates"].items()
               if name != gs.MLA_KV_B_ARRANGEMENT)))

    # ---- 5b. expert SLOT ordering against the official BF16 expert ----------
    # The Q4_K fixture is the leading rows of blk.3.ffn_gate_exps.weight, i.e.
    # of whatever expert sits at slot 0; dione-evidence carries the official
    # BF16 layers.3.mlp.experts.0.gate_proj.weight. If slot 0 is HF expert 0,
    # they agree to the Q4_K error -- which also proves the reversed-dims
    # orientation and the ffn_gate_exps -> gate_proj mapping, since a
    # transposed read or the wrong projection fails the same test.
    official_expert_path = (DIONE_EVIDENCE / "payloads" /
                            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight.bin")
    if official_expert_path.is_file():
        rows_decoded = gs.dequant_bytes("Q4_K", raw_q4k, len(raw_q4k) // 144 * 256)
        rows_decoded = rows_decoded.reshape(-1, 4096)
        official_expert = torch.frombuffer(
            bytearray(official_expert_path.read_bytes()), dtype=torch.bfloat16
        ).reshape(2048, 4096)
        slot_audit = gs.audit_expert_placement(
            rows_decoded, official_expert,
            label="blk.3.ffn_gate_exps.weight slot 0 vs layers.3.mlp.experts.0.gate_proj")
        assert slot_audit["passed"]
        passed.append(
            "5b expert slot ordering: the fused tensor's slot 0 reproduces the official BF16 "
            "expert 0 at rel-L2 %.4f (the Q4_K error) while every row-shifted control sits at "
            ">= %.4f -- slot order, reversed-dims orientation and the projection mapping all "
            "hold" % (slot_audit["aligned"]["rel_l2"], slot_audit["best_control_rel_l2"]))
    else:
        passed.append("5b SKIPPED (dione-evidence official expert payload absent)")

    # ---- 6. expert slice + materialized non-routed view ---------------------
    from safetensors import safe_open
    from safetensors.torch import save_file

    rng = np.random.default_rng(0x6667)

    def _blocks(count: int, block_bytes: int, scale_fields: int) -> np.ndarray:
        """Random blocks whose leading f16 scale fields are VALID floats.

        Uniform random bytes would put NaN/Inf in the f16 scale of some blocks,
        and a NaN weight makes torch.equal false for reasons that have nothing
        to do with the slicing being tested.
        """
        out = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
        scales = (rng.standard_normal((count, scale_fields)) * 0.01).astype(np.float16)
        out[:, :2 * scale_fields] = scales.view(np.uint8)
        return out

    fused_raw = _blocks(288 * (2048 * 4096 // 256), 144, 2).tobytes()
    # llama.cpp stores norms F32 even though the official tree stores them BF16:
    # the view must DOWNCAST those, and pass through only the tensors the
    # official tree itself stores F32 (hc_attn_base here).
    norm = rng.standard_normal(4096).astype(np.float32)
    hc_base = rng.standard_normal(24).astype(np.float32)
    embed_blocks = _blocks(4 * (4096 // 32), 34, 1).tobytes()

    class _Mini:
        architecture = "glm5next"
        remote = False
        tensors = {
            "blk.3.ffn_gate_exps.weight": {"name": "blk.3.ffn_gate_exps.weight", "type": "Q4_K",
                                           "dims": [4096, 2048, 288],
                                           "elements": 288 * 2048 * 4096, "file": "mini"},
            "blk.11.hc_attn_base.weight": {"name": "blk.11.hc_attn_base.weight", "type": "F32",
                                           "dims": [24], "elements": 24, "file": "mini"},
            "blk.11.attn_norm.weight": {"name": "blk.11.attn_norm.weight", "type": "F32",
                                        "dims": [4096], "elements": 4096, "file": "mini"},
            "token_embd.weight": {"name": "token_embd.weight", "type": "Q8_0",
                                  "dims": [4096, 4], "elements": 4 * 4096, "file": "mini"},
        }
        _raw = {"blk.3.ffn_gate_exps.weight": fused_raw,
                "blk.11.hc_attn_base.weight": hc_base.tobytes(),
                "blk.11.attn_norm.weight": norm.tobytes(),
                "token_embd.weight": embed_blocks}

        def read_tensor(self, name):
            return self._raw[name]

        def read_tensor_range(self, name, offset, length):
            return self._raw[name][offset:offset + length]

    mini = _Mini()
    expert = 137
    offset, length = gs.expert_slice_range(mini.tensors["blk.3.ffn_gate_exps.weight"], expert)
    assert length == 2048 * 4096 // 256 * 144 and offset == expert * length
    mini_census = gs.GgufCensus(
        direct_map={"blk.11.hc_attn_base.weight":
                    "model.language_model.layers.11.hc_attn_base",
                    "blk.11.attn_norm.weight":
                    "model.language_model.layers.11.input_layernorm.weight",
                    "token_embd.weight": "model.language_model.embed_tokens.weight"},
        routed={(3, "gate_proj"): "blk.3.ffn_gate_exps.weight"}, mla={}, mla_layers=())
    sliced, row = gs.load_decoded_expert(mini, mini_census, layer=3, expert=expert,
                                         projection="gate_proj")
    direct = gs.dequant_bytes("Q4_K", fused_raw[offset:offset + length],
                              2048 * 4096).reshape(2048, 4096)
    assert torch.equal(sliced, direct), "the fused-tensor expert slice is not the expert"
    assert row["tensor"].endswith("layers.3.mlp.experts.137.gate_proj.weight")
    assert row["ggml_type"] == "Q4_K" and row["bytes"] == length

    mini_bf16 = scratch / "mini-bf16"
    mini_bf16.mkdir()
    vision = {
        "model.visual.blocks.0.mlp.fc1.weight":
            torch.arange(64, dtype=torch.float32).reshape(8, 8).to(torch.bfloat16)
    }
    save_file(vision, str(mini_bf16 / "vision.safetensors"))
    # a real official shard carrying the non-routed names at their OFFICIAL
    # dtypes, so verify_official_dtypes actually reads headers here
    save_file(
        {"model.language_model.layers.11.hc_attn_base": torch.zeros(24, dtype=torch.float32),
         "model.language_model.layers.11.input_layernorm.weight":
             torch.zeros(4096, dtype=torch.bfloat16),
         "model.language_model.embed_tokens.weight":
             torch.zeros(4, 4096, dtype=torch.bfloat16)},
        str(mini_bf16 / "nonrouted.safetensors"))
    weight_map = {name: "vision.safetensors" for name in vision}
    weight_map.update({name: "nonrouted.safetensors" for name in mini_census.direct_map.values()})
    for layer in range(3, 46):
        for index in range(288):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight_map[gs.official_expert_name(layer, index, projection)] = \
                    "absent.safetensors"
    (mini_bf16 / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (mini_bf16 / "config.json").write_text('{"model_type": "glm5_next"}', encoding="utf-8")

    mini_surface = gs.GgufSurface(
        container=mini, census=mini_census, repo="unsloth/GLM-5.3-Flash-GGUF",
        revision="2975ab414d30340466d8c51533c6e91f0cca64c1", architecture="glm5next",
        file_records=({"name": "mini", "bytes": 1, "sha256": "0" * 64},),
        file_hash_verification="full", type_census={"Q4_K": 1, "Q8_0": 1, "F32": 2},
        scope_policy={"policy": "test"}, quant_metadata={})
    view, record = gs.materialize_nonrouted_view(mini_surface, mini_bf16, scratch / "work",
                                                 progress=False)
    assert record["reused"] is False
    assert record["tensor_count"] == 4, record["tensor_count"]
    assert record["counts"] == {"decoded_bf16": 2, "f32_passthrough": 1,
                                "mla_reconstructed": 0, "vision_copied": 1}, record["counts"]
    audit_dtypes = record["official_dtype_audit"]
    assert audit_dtypes["tensors_checked_against_official_headers"] == 3
    assert audit_dtypes["policy_disagreements"] == 0
    assert (view / "config.json").is_file(), "config was not carried into the view"
    index = json.loads((view / "model.safetensors.index.json").read_text(encoding="utf-8"))
    with safe_open(str(view / index["weight_map"][
            "model.language_model.layers.11.hc_attn_base"]),
            framework="pt", device="cpu") as handle:
        stored = handle.get_tensor("model.language_model.layers.11.hc_attn_base")
    assert stored.dtype == torch.float32, "an official-float32 tensor was rounded to bf16"
    assert torch.equal(stored, torch.from_numpy(hc_base)), "F32 passthrough is not byte-exact"
    with safe_open(str(view / index["weight_map"][
            "model.language_model.layers.11.input_layernorm.weight"]),
            framework="pt", device="cpu") as handle:
        downcast = handle.get_tensor(
            "model.language_model.layers.11.input_layernorm.weight")
    assert downcast.dtype == torch.bfloat16, (
        "a GGUF-F32 tensor whose OFFICIAL dtype is bf16 was not downcast; the view would "
        "not be dtype-identical to a native build")
    assert torch.equal(downcast, torch.from_numpy(norm).to(torch.bfloat16))
    with safe_open(str(view / index["weight_map"]["model.language_model.embed_tokens.weight"]),
                   framework="pt", device="cpu") as handle:
        embed = handle.get_tensor("model.language_model.embed_tokens.weight")
    assert embed.dtype == torch.bfloat16 and tuple(embed.shape) == (4, 4096)
    assert gs.verify_view_nonrouted_values(mini_surface, view)["all_equal"]
    _, again = gs.materialize_nonrouted_view(mini_surface, mini_bf16, scratch / "work",
                                             progress=False)
    assert again["reused"] is True, "the fingerprint stamp did not reuse the view"
    passed.append(
        "6 slice + view: expert 137 sliced from a 288-expert Q4_K fused tensor equals a direct "
        "decode of its byte range; the materialized view writes official names (bf16 decoded, "
        "float32 passthrough byte-exact, vision copied), round-trips bitwise and is REUSED on "
        "the second call")

    # ---- 7. dry-run --------------------------------------------------------
    env = dict(os.environ)
    env.pop("QP_PIPELINE_ROOT", None)
    run = subprocess.run(
        [sys.executable, str(TOOLS / "gguf_surface.py"), "dry-run",
         "--file", str(a), "--file", str(b),
         "--repo", "unsloth/GLM-5.3-Flash-GGUF",
         "--revision", "2975ab414d30340466d8c51533c6e91f0cca64c1",
         "--bf16-index", str(DIONE_EVIDENCE / "bf16-index.json")],
        capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary["schema"] == gs.GGUF_SURFACE_SCHEMA
    assert summary["architecture"] == "glm5next"
    assert summary["tensor_count"] == 1412
    assert summary["streamed_routed_modules"] == 42 * 288 * 3
    assert summary["nonrouted_bijection"]["bijection_ok"] is True
    assert summary["nonrouted_tensors_from_artifact"] == 1271
    assert summary["scope_policy"]["embeddings_type"] == "Q8_0"
    assert summary["seal_disclosure"] == gs.SEAL_DISCLOSURE
    assert len(summary["checkpoint_identity_sha256"]) == 64
    passed.append(
        "7a dry-run: gguf_surface.py dry-run plans the REAL artifact from a header-only "
        "rebuild (1,412 tensors, %d streamed routed modules, bijection ok, identity %s...)"
        % (summary["streamed_routed_modules"], summary["checkpoint_identity_sha256"][:12]))

    # ---- 7c. the MEASURED per-tensor-class scope ---------------------------
    # Known answers over the REAL 1,412-tensor table, because this is the block
    # that decides what a published row CLAIMS the artifact quantized. Every
    # number below is arithmetic over ggml block traits, not a reading of the
    # build name -- which says "Q4_K" and is wrong about the whole non-routed
    # half of the file.
    run = subprocess.run(
        [sys.executable, str(TOOLS / "gguf_surface.py"), "scope",
         "--file", str(a), "--file", str(b),
         "--repo", "unsloth/GLM-5.3-Flash-GGUF",
         "--revision", "2975ab414d30340466d8c51533c6e91f0cca64c1"],
        capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    report = json.loads(run.stdout)
    scope = report["scope"]
    assert report["schema"] == gs.GGUF_SCOPE_SCHEMA
    by_class = {a_["tensor_class"]: a_ for a_ in scope["assignments"]}
    assert len(by_class) == len(scope["assignments"]), (
        "a tensor_class appears twice; scope_digest would double-count it")
    # the name says 4 bits; the artifact is 4.98, because everything outside the
    # routed experts is Q8_0. This is the number the codec block records as
    # bits_per_weight_effective.
    assert abs(report["measured_bits_per_weight"] - 4.98062529958249) < 1e-9, \
        report["measured_bits_per_weight"]
    # the three claims that separate a GGUF row from a routed-experts-only one
    assert by_class["lm_head"]["treatment"] == "quantized"
    assert by_class["embed_tokens"]["treatment"] == "quantized"
    assert by_class["attn.qkv"]["treatment"] == "quantized"
    assert scope["head_policy"] == "quantized"
    # ... and the one that separates it from a UNIFORM quant
    assert scope["policy"] == "mixed"
    assert by_class["moe.experts"]["bits_per_weight"] is None, (
        "a class holding Q4_K, Q5_K and Q6_K must not claim one nominal rate")
    assert "Q4_K x82" in by_class["moe.experts"]["note"]
    assert "Q6_K x3" in by_class["moe.experts"]["note"]
    # norms are the one thing a GGUF stores WIDER than the release
    assert by_class["norm"]["treatment"] == "native"
    assert by_class["norm"]["format"] == "fp32"
    assert "mmproj" in by_class["other"]["note"], (
        "the absent vision tower must be stated, not left to be assumed")
    assert report["source"]["quant_metadata"]["general.quantized_by"] == "Unsloth"
    # and the committed fixture the bin-side lane selftest seals against is THIS
    # scope, not a hand-edited copy of it: a fixture nothing re-derives is a
    # second source of truth waiting to drift from the first.
    fixture = EVIDENCE / "udq4kxl-scope.json"
    if fixture.is_file():
        assert json.loads(fixture.read_text(encoding="utf-8"))["scope"] == scope, (
            "%s no longer equals the scope recomputed from the real tensor table"
            % fixture)
    passed.append(
        "7c scope: the per-class recipe is MEASURED from the real 1,412-tensor table -- "
        "lm_head/embed_tokens/attn.qkv quantized, head_policy quantized, policy mixed, "
        "moe.experts claims no single rate (Q4_K/Q5_K/Q6_K), and the artifact measures "
        "%.4f bits/weight against a name that says 4"
        % report["measured_bits_per_weight"])

    if args.pipeline_root:
        passed.append(_stream_score_dry_run(scratch, args.pipeline_root, [a, b], env))
    else:
        passed.append("7b SKIPPED (no --pipeline-root: stream_score.py imports quant_pipeline)")

    passed.extend(_glmdsa_rungs(scratch, torch))

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    for line in passed:
        print("PASS", line)
    print(json.dumps({"ok": True, "checks": len(passed)}))
    return 0


def _canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# rungs 8-8e: the GLM-5.3 FLAGSHIP (general.architecture glm-dsa), the surface
# the layer-outer `gguf-dequant-to-bf16` lane decodes.
# ---------------------------------------------------------------------------
def build_mini_glmdsa(out_dir: Path):
    """A synthetic two-layer glm-dsa GGUF with REAL container bytes over a
    shrunken GgufArch (4 experts, 2 MLA heads, hidden 256): layers 5 and 78
    (the MTP block), every tensor family the flagship carries, Q8_0/Q4_K/
    IQ3_XXS/F32 payloads. Returns (surface, mini_arch, tensors) where
    `tensors` maps gguf name -> (type, dims, raw bytes). Shared by rung 8c and
    bin/selftest_layer_outer.py L19g-i."""
    arch = gs.GLM_DSA
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0x6D5A)
    HID, RANK, QR, INTER, HEADS, NOPE, VD, E = 256, 64, 64, 64, 2, 24, 32, 4
    mini_arch = gs.GgufArch(**dict(
        arch.__dict__, num_experts=E, mla_heads=HEADS, mla_k_nope=NOPE, mla_v_dim=VD,
        mla_kv_lora_rank=RANK,
        projection_shape={"gate_proj": (INTER, HID), "up_proj": (INTER, HID),
                          "down_proj": (HID, INTER)}))
    assert mini_arch.kv_b_rows == HEADS * (NOPE + VD) and mini_arch.mtp_layer == 78

    def blocks(count, block_bytes, scale_fields=1, scale_at=0):
        out = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
        scales = (rng.standard_normal((count, scale_fields)) * 0.01).astype(np.float16)
        out[:, scale_at:scale_at + 2 * scale_fields] = scales.view(np.uint8)
        return out.tobytes()

    def q8(n):
        return blocks(n // 32, 34)

    def q4k(n):
        return blocks(n // 256, 144, 2)

    def iq3(n):
        return blocks(n // 256, 98)

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32).tobytes()

    tensors = {  # name -> (type, dims fastest-first, bytes)
        "token_embd.weight": ("Q8_0", [HID, 8], q8(8 * HID)),
        "output.weight": ("Q8_0", [HID, 8], q8(8 * HID)),
        "output_norm.weight": ("F32", [HID], f32(HID)),
    }
    for layer in (5, 78):
        b = "blk.%d." % layer
        tensors.update({
            b + "attn_norm.weight": ("F32", [HID], f32(HID)),
            b + "ffn_norm.weight": ("F32", [HID], f32(HID)),
            b + "attn_q_a.weight": ("Q8_0", [HID, QR], q8(HID * QR)),
            b + "attn_q_a_norm.weight": ("F32", [QR], f32(QR)),
            b + "attn_q_b.weight": ("Q8_0", [QR, HEADS * 64], q8(QR * HEADS * 64)),
            b + "attn_kv_a_mqa.weight": ("Q8_0", [HID, RANK + 64], q8(HID * (RANK + 64))),
            b + "attn_kv_a_norm.weight": ("F32", [RANK], f32(RANK)),
            b + "attn_k_b.weight": ("Q8_0", [NOPE, RANK, HEADS], q8(NOPE * RANK * HEADS)),
            b + "attn_v_b.weight": ("Q8_0", [RANK, VD, HEADS], q8(RANK * VD * HEADS)),
            b + "attn_output.weight": ("Q8_0", [HEADS * VD, HID], q8(HEADS * VD * HID)),
            b + "ffn_gate_inp.weight": ("F32", [HID, E], f32(E, HID)),
            b + "exp_probs_b.bias": ("F32", [E], f32(E)),
            b + "ffn_gate_exps.weight": ("Q4_K", [HID, INTER, E], q4k(HID * INTER * E)),
            b + "ffn_up_exps.weight": ("IQ3_XXS", [HID, INTER, E], iq3(HID * INTER * E)),
            b + "ffn_down_exps.weight": ("Q4_K", [INTER, HID, E], q4k(HID * INTER * E)),
            b + "ffn_gate_shexp.weight": ("Q8_0", [HID, INTER], q8(HID * INTER)),
            b + "ffn_up_shexp.weight": ("Q8_0", [HID, INTER], q8(HID * INTER)),
            b + "ffn_down_shexp.weight": ("Q8_0", [INTER, HID], q8(HID * INTER)),
            b + "indexer.attn_k.weight": ("Q8_0", [HID, 32], q8(HID * 32)),
            b + "indexer.attn_q_b.weight": ("Q8_0", [QR, 64], q8(QR * 64)),
            b + "indexer.proj.weight": ("F32", [HID, 8], f32(8, HID)),
            b + "indexer.k_norm.weight": ("F32", [32], f32(32)),
            b + "indexer.k_norm.bias": ("F32", [32], f32(32)),
        })
    tensors["blk.78.nextn.eh_proj.weight"] = ("Q8_0", [2 * HID, HID], q8(2 * HID * HID))
    for suffix in ("nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"):
        tensors["blk.78." + suffix] = ("F32", [HID], f32(HID))
    rows_w, data, offset = [], b"", 0
    for name in sorted(tensors):
        ttype, dims, raw = tensors[name]
        rows_w.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data += raw
        pad = (-len(raw)) % 32
        data += b"\0" * pad
        offset += len(raw) + pad
    kv = json.loads((EVIDENCE / "unsloth-glm53-udq4kxl-kv.json").read_text(encoding="utf-8"))
    kv = json.loads((EVIDENCE / "unsloth-glm53-udq4kxl-kv.json").read_text(encoding="utf-8"))
    mini_kv = {k: v for k, v in kv.items() if k.startswith("glm-dsa.") or k == "general.architecture"}
    mini_path = write_gguf(Path(out_dir) / "mini-glmdsa.gguf", mini_kv, rows_w, data=data)
    mini_container = gs.GgufContainer([gs.GgufFile(str(mini_path))])
    # the census closure demands every routed layer; the synthetic file has two,
    # so build the census by hand from the same classifier, as layer_outer's
    # plan would on a truncated fixture tree
    direct_map, routed, mla = {}, {}, {}
    for name in mini_container.tensors:
        role = gs.classify_tensor(name, arch)
        assert role[0] != "unmapped", name
        if role[0] == "top":
            direct_map[name] = role[1]
        elif role[0] == "direct":
            direct_map[name] = role[2]
        elif role[0] == "routed":
            routed[(role[1], role[2])] = name
        else:
            mla[(role[1], role[2])] = name
    mini_census = gs.GgufCensus(direct_map=direct_map, routed=routed, mla=mla,
                                mla_layers=(5, 78), arch=mini_arch)
    mini_surface = gs.GgufSurface(
        container=mini_container, census=mini_census, repo="test", revision="0" * 40,
        architecture="glm-dsa", file_records=({"name": "mini-glmdsa.gguf", "bytes": 1, "sha256": None},),
        file_hash_verification="skipped", type_census={}, scope_policy={}, quant_metadata={})
    return mini_surface, mini_arch, tensors


def _glmdsa_rungs(scratch: Path, torch) -> "list":
    import gzip
    import hashlib

    passed = []
    arch = gs.GLM_DSA
    config = json.loads((EVIDENCE / "glm53-official-config.json").read_text(encoding="utf-8"))
    full = gs.indexer_full_layers_from_config(config, arch)
    assert full[:4] == (0, 1, 2, 6) and full[-1] == 78 and len(full) == 22, full

    # ---- 8. real-metadata census + bijection on the flagship ----------------
    rows = _real_rows("unsloth-glm53-udq4kxl-tensors.json")
    kv = json.loads((EVIDENCE / "unsloth-glm53-udq4kxl-kv.json").read_text(encoding="utf-8"))
    for key in ("split.no", "split.count", "split.tensors.count"):
        kv.pop(key, None)
    flag_dir = scratch / "glmdsa"
    flag_dir.mkdir()
    good = write_gguf(flag_dir / "flagship.gguf", kv, _rows_for_writer(rows))
    # the census REFUSES to guess which indexer tensors are copies
    _refuses(lambda: gs.load_gguf_surface([str(good)], require_file_hashes=False),
             "indexer_types", "glm-dsa")
    surface = gs.load_gguf_surface([str(good)], repo="unsloth/GLM-5.3-GGUF",
                                   revision="346b3591c7f28d1a23716f97a065ecf12ec14771",
                                   require_file_hashes=False, indexer_full_layers=full)
    census = surface.census
    assert surface.arch is arch and surface.container.architecture == "glm-dsa"
    assert len(surface.container.tensors) == 1809
    assert len(census.routed) == 76 * 3 and len(census.mla) == 79 * 2, (len(census.routed), len(census.mla))
    assert len(census.shared_indexer_copies) == (79 - 22) * 5 - 0, len(census.shared_indexer_copies)
    # 79 layers x 5 indexer tensors = 395 in the GGUF; 22 full layers keep theirs
    assert sum(1 for n in surface.container.tensors if ".indexer." in n) == 395
    assert len(census.direct_map) == 1809 - 228 - 158 - len(census.shared_indexer_copies)
    official = json.loads(gzip.open(EVIDENCE / "glm53-official-bf16-weight-map.json.gz", "rt").read())
    assert official["revision"] == "304b8051cfb2b260b61ce0cbe330e02a98e73639"
    bijection = gs.verify_nonrouted_bijection(census, official["weight_map"].keys())
    assert bijection["official_tensors"] == 59585
    assert bijection["official_routed_tensors"] == 76 * 256 * 3
    assert bijection["official_vision_tensors"] == 0
    assert bijection["mla_reconstructed_tensors"] == 79
    assert bijection["shared_indexer_copies_not_loaded"] == 285
    for gguf_name, parent in list(census.shared_indexer_copies.items())[:3]:
        assert parent in full and int(gguf_name.split(".")[1]) not in full
    # the layer partition covers every tensor exactly once and never a copy
    partition = gs.layer_partition(census)
    covered = [n for names in partition.values() for n in names]
    assert len(covered) == len(set(covered)) == 1809 - len(census.shared_indexer_copies)
    assert sorted(partition[gs.RESIDENT_LAYER]) == ["output.weight", "output_norm.weight",
                                                    "token_embd.weight"]
    assert sorted(l for l in partition if l >= 0) == list(range(79))
    # refusals specific to the flagship table
    bad_types = list(config["indexer_types"])
    bad_types[0] = "shared"
    _refuses(lambda: gs.indexer_full_layers_from_config(dict(config, indexer_types=bad_types), arch),
             "layer 0 must be 'full'")
    _refuses(lambda: gs.indexer_full_layers_from_config(dict(config, indexer_types=bad_types[:-1]), arch),
             "77 entries", "78 decoder layers")
    bad_kv = dict(kv)
    bad_kv["glm-dsa.attention.key_length_mla"] = 192
    bad = write_gguf(flag_dir / "geom.gguf", bad_kv, _rows_for_writer(rows))
    _refuses(lambda: gs.load_gguf_surface([str(bad)], require_file_hashes=False,
                                          indexer_full_layers=full),
             "geometry gate", "attention.key_length_mla")
    passed.append(
        "8 glm-dsa census: the real 1,809-tensor UD-Q4_K_XL table closes (1,138 direct + 228 "
        "fused + 158 MLA halves + 285 shared-indexer copies never loaded) and bijects the real "
        "official BF16 index (59,585 - 58,368 routed); the census refuses without the "
        "config's indexer_types and on a wrong MLA geometry")

    # ---- 8a. flagship gguf-py parity fixtures were taken from the RIGHT builds -
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    build_types = json.loads((EVIDENCE / "unsloth-glm53-build-types.json").read_text(encoding="utf-8"))
    for qtype in ("IQ3_XXS", "IQ4_XS", "Q3_K", "IQ3_S"):
        spec = manifest["dequant_fixtures"][qtype]
        src = spec["source"]
        assert src["repo"] == "unsloth/GLM-5.3-GGUF" and src["architecture"] == "glm-dsa"
        assert build_types["builds"][src["build"]]["types_by_tensor"][spec["gguf_tensor"]] == qtype, (
            qtype, src)
        # and their type traits are the ones the census sizes bytes with
        per_block, block_bytes = gs.BLOCK_TRAITS[qtype]
        assert spec["raw_bytes"] == spec["rows"] * spec["cols"] // per_block * block_bytes
    # the two flagship builds the lane measures are FULLY decodable, by type
    for build, want in (("UD-Q3_K_XL", {"F32", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K",
                                        "IQ4_XS", "IQ3_XXS"}),
                        ("UD-IQ4_XS", {"F32", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K",
                                       "IQ4_XS", "IQ3_S"})):
        types = set(build_types["builds"][build]["types_by_tensor"].values())
        assert types == want, (build, types)
        assert types <= set(gs.SUPPORTED_TYPES)
    passed.append(
        "8a flagship parity provenance: the IQ3_XXS/IQ4_XS/Q3_K/IQ3_S fixtures come from the "
        "named tensors of UD-Q3_K_XL / UD-IQ4_XS (type-checked against the committed build "
        "tables), and both builds' full type sets are decodable")

    # ---- 8b. MLA composition is EXACT on real BF16 windows -------------------
    spec = manifest["flagship_glmdsa"]["mla_fixture"]
    k = torch.from_numpy(np.load(EVIDENCE / "glmdsa_mla_k_b_head_window.npy").astype(np.int32)).to(torch.int32)
    v = torch.from_numpy(np.load(EVIDENCE / "glmdsa_mla_v_b_head_window.npy").astype(np.int32)).to(torch.int32)
    off = torch.from_numpy(np.load(EVIDENCE / "glmdsa_mla_official_row_window_bf16.npy").astype(np.int32))
    heads = int(spec["heads_in_fixture"])
    assert k.shape == (heads, 512, 192) and v.shape == (heads, 256, 512)
    window_arch = gs.GgufArch(**dict(arch.__dict__, mla_heads=heads))
    composed = gs.compose_kv_b(k.to(torch.float32), v.to(torch.float32), window_arch)
    assert composed.shape == (heads * 448, 512)
    assert torch.equal(composed.to(torch.int32), off), "kv_b composition is not bit-exact"
    wrong = torch.cat([v.to(torch.float32), k.to(torch.float32).transpose(1, 2)], dim=1).reshape(-1, 512)
    assert not torch.equal(wrong.to(torch.int32), off)
    audit = json.loads((EVIDENCE / "glmdsa-layout-audit.json").read_text(encoding="utf-8"))
    assert audit["all_equal"] is True and audit["official"]["revision"] == "304b8051cfb2b260b61ce0cbe330e02a98e73639"
    comparisons = [c for c in audit["comparisons"] if "equal" in c]
    assert len(comparisons) >= 45 and all(c["equal"] for c in comparisons)
    labels = " ".join(c["label"] for c in comparisons)
    for needle in ("kv_b_proj composition", "q_b_proj", "kv_a_proj_with_mqa", "dense mlp.gate_proj",
                   "routed expert 255", "indexer.wk", "embed_tokens", "lm_head", "model.norm",
                   "MTP layer 78 eh_proj"):
        assert needle in labels, needle
    shared = next(c for c in audit["comparisons"] if c["label"].startswith("shared-indexer"))
    assert shared["official_has_indexer_on_layers"] == list(full)
    assert all(all(t["equals_prev_full_layer_%d" % row["shares_with_full_layer"]]
                   for t in row["tensors"].values()) for row in shared["per_layer"].values())
    tok = json.loads((EVIDENCE / "glmdsa-tokenizer-order-audit.json").read_text(encoding="utf-8"))
    for build in ("BF16", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-IQ4_XS"):
        row = tok["variants"][build]
        assert row["tokens"] == 154880 and row["order_mismatches_vs_official"] == 0
        assert row["absent_count"] == 24 and row["merges_equal_official"] is True
    passed.append(
        "8b glm-dsa layout: kv_b_proj = per-head [k_b^T (192 rows); v_b (256 rows)] is BIT-EXACT "
        "against the official BF16 rows on the committed 2-head window (and the wrong order is not); "
        "the committed audit holds %d exact comparisons over the BF16 build incl. every MLA/dense/"
        "routed/indexer/embed/head/MTP class, 285 shared-indexer copies value-identical to their "
        "parents, and the tokenizer order equals the official vocab on all four builds (24 trailing "
        "[PAD] ids past the official 154856)" % len(comparisons))

    # ---- 8c. materialize_layer on a synthetic two-layer glm-dsa GGUF -----------
    # The real geometry is 3.2e9 elements per fused expert tensor; the reader's
    # LOGIC (name map, slot slicing, kv_b composition, dtype policy) does not
    # depend on the numbers, so the fixture runs the same code over a shrunken
    # GgufArch (4 experts, 2 MLA heads, hidden 256) whose dims the file carries.
    mini_surface, mini_arch, tensors = build_mini_glmdsa(flag_dir / "mini")
    HID, RANK, QR, INTER, HEADS, NOPE, VD, E = 256, 64, 64, 64, 2, 24, 32, 4
    mini_container = mini_surface.container
    direct_map, routed, mla = (mini_surface.census.direct_map, mini_surface.census.routed,
                               mini_surface.census.mla)
    stats: dict = {}
    out = gs.materialize_layer(mini_surface, 5, stats=stats)
    want_names = {arch.layer_name(5, s) for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias", "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight", "mlp.shared_experts.down_proj.weight",
        "self_attn.indexer.wk.weight", "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.weights_proj.weight", "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias")}
    want_names |= {arch.expert_name(5, e, p) for e in range(E) for p in gs.PROJECTIONS}
    assert set(out) == want_names, (sorted(set(out) ^ want_names)[:5])
    assert out[arch.kv_b_name(5)].shape == (HEADS * (NOPE + VD), RANK) and out[arch.kv_b_name(5)].dtype == torch.bfloat16
    assert out[arch.layer_name(5, "mlp.gate.e_score_correction_bias")].dtype == torch.float32
    assert out[arch.layer_name(5, "mlp.gate.weight")].dtype == torch.bfloat16
    assert out[arch.layer_name(5, "input_layernorm.weight")].dtype == torch.bfloat16
    assert out[arch.expert_name(5, 0, "gate_proj")].shape == (INTER, HID)
    assert out[arch.expert_name(5, E - 1, "down_proj")].shape == (HID, INTER)
    assert out[arch.layer_name(5, "self_attn.q_b_proj.weight")].shape == (HEADS * 64, QR)
    assert stats["tensors_decoded"] == 23 and stats["official_tensors_produced"] == 19 + 3 * E
    assert stats["ggml_types"] == {"F32": 9, "Q8_0": 11, "Q4_K": 2, "IQ3_XXS": 1}, stats["ggml_types"]
    # values: the expert slice equals a direct decode of its byte range, once bf16-rounded
    row = mini_container.tensors["blk.5.ffn_up_exps.weight"]
    rel, nbytes = gs.expert_slice_range(row, 2, mini_arch)
    direct = gs.dequant_bytes("IQ3_XXS", mini_container.read_tensor_range(row["name"], rel, nbytes),
                              INTER * HID).reshape(INTER, HID).to(torch.bfloat16)
    assert torch.equal(out[arch.expert_name(5, 2, "up_proj")], direct)
    # kv_b: composed from the two decoded halves, then ONE rounding
    k_full = gs.dequant_bytes("Q8_0", mini_container.read_tensor("blk.5.attn_k_b.weight"), NOPE * RANK * HEADS)
    v_full = gs.dequant_bytes("Q8_0", mini_container.read_tensor("blk.5.attn_v_b.weight"), RANK * VD * HEADS)
    assert torch.equal(out[arch.kv_b_name(5)], gs.compose_kv_b(k_full, v_full, mini_arch).to(torch.bfloat16))
    # e_score_correction_bias passes through byte-exact (official float32)
    bias = np.frombuffer(tensors["blk.5.exp_probs_b.bias"][2], dtype=np.float32)
    assert np.array_equal(out[arch.layer_name(5, "mlp.gate.e_score_correction_bias")].numpy(), bias)
    resident = gs.materialize_layer(mini_surface, gs.RESIDENT_LAYER)
    assert set(resident) == {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    assert resident["lm_head.weight"].shape == (8, HID) and resident["lm_head.weight"].dtype == torch.bfloat16
    mtp = gs.materialize_layer(mini_surface, 78)
    assert arch.layer_name(78, "eh_proj.weight") in mtp and mtp[arch.layer_name(78, "eh_proj.weight")].shape == (HID, 2 * HID)
    assert len(mtp) == 19 + 4 + 3 * E
    _refuses(lambda: gs.materialize_layer(mini_surface, 6), "not a layer of this artifact")
    # `only`: the streamer's resident load asks a layer for ONE buffer
    bias_name = arch.layer_name(5, "mlp.gate.e_score_correction_bias")
    part_stats: dict = {}
    part = gs.materialize_layer(mini_surface, 5, only=[bias_name], stats=part_stats)
    assert set(part) == {bias_name} and part_stats["tensors_decoded"] == 1
    assert torch.equal(part[bias_name], out[bias_name])
    one_expert = arch.expert_name(5, 1, "gate_proj")
    part = gs.materialize_layer(mini_surface, 5, only=[one_expert, arch.kv_b_name(5)])
    assert set(part) == {one_expert, arch.kv_b_name(5)}
    assert torch.equal(part[one_expert], out[one_expert])
    _refuses(lambda: gs.materialize_layer(mini_surface, 5, only=["model.layers.5.nope"]),
             "does not carry the requested tensors")
    plan = gs.materialize_plan(mini_surface)
    assert plan["kv_b_shape"] == [HEADS * (NOPE + VD), RANK] and plan["mtp_layer"] == 78
    assert gs.materialize_plan(surface)["kv_b_shape"] == [28672, 512]
    passed.append(
        "8c materialize_layer: a synthetic two-layer glm-dsa GGUF (Q8_0/Q4_K/IQ3_XXS/F32, real "
        "container bytes, shrunken geometry) decodes layer 5 into %d official names (E x 3 experts sliced "
        "from the fused tensors, kv_b_proj composed per head, e_score_correction_bias kept fp32, norms "
        "cast bf16), the resident set and the MTP block; expert and kv_b values equal a direct "
        "decode rounded once" % len(out))

    # ---- 8d. shared-indexer copies: the proof routine and its refusal -------------
    # a container where layer 5 declares itself a copy of layer 78 (synthetic
    # parent): identical bytes pass, one flipped byte refuses by name
    copy_census = gs.GgufCensus(direct_map=direct_map, routed=routed, mla=mla, mla_layers=(5, 78),
                                arch=mini_arch, shared_indexer_copies={
                                    "blk.5.indexer.k_norm.bias": 78})
    twin_dir = flag_dir / "twin"
    twin_dir.mkdir()
    twin_tensors = dict(tensors)
    twin_tensors["blk.5.indexer.k_norm.bias"] = twin_tensors["blk.78.indexer.k_norm.bias"]
    rows_t, data_t, offset = [], b"", 0
    for name in sorted(twin_tensors):
        ttype, dims, raw = twin_tensors[name]
        rows_t.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data_t += raw + b"\0" * ((-len(raw)) % 32)
        offset += len(raw) + (-len(raw)) % 32
    mini_kv = {k: v for k, v in kv.items() if k.startswith("glm-dsa.") or k == "general.architecture"}
    twin = gs.GgufContainer([gs.GgufFile(str(write_gguf(twin_dir / "twin.gguf", mini_kv, rows_t, data=data_t)))])
    proof = gs.verify_shared_indexer_copies(twin, copy_census)
    assert proof == {"copies_compared": 1, "copies_by_parent_layer": {78: 1}, "all_value_identical": True}
    _refuses(lambda: gs.verify_shared_indexer_copies(mini_container, copy_census),
             "blk.5.indexer.k_norm.bias", "not a value-identical copy")
    passed.append("8d shared-indexer proof: value-identical copies pass, a differing copy refuses by name")

    # ---- 8e. the scope tool over the real header table ---------------------------
    import gguf_scope
    source = "test source"
    rows_scope = gguf_scope.assignments(surface, source)
    by_class = {r["tensor_class"]: r for r in rows_scope}
    assert by_class["lm_head"] == dict(by_class["lm_head"], treatment="quantized", format="gguf-k-quant",
                                       bits_per_weight=8.5, layer_range="all")
    assert by_class["moe.experts"]["format"] == "gguf-k-quant" and by_class["moe.experts"]["layer_range"] == "3-77"
    assert abs(by_class["moe.experts"]["bits_per_weight"] - 4.8611) < 1e-4
    assert by_class["mtp"]["layer_range"] == "78" and by_class["mtp"]["format"] == "mixed"
    assert by_class["attn.other"]["format"] == "mixed" and by_class["attn.other"]["bits_per_weight"] is None
    assert by_class["moe.router"]["treatment"] == "native" and by_class["moe.router"]["format"] == "fp32"
    assert by_class["mlp.gate"]["layer_range"] == "0-2"
    pairs = [(r["tensor_class"], r["layer_range"]) for r in rows_scope]
    assert len(pairs) == len(set(pairs)), "SCOPE-004: duplicate (class, range)"
    committed = json.loads((TOOLS.parent / "scopes" / "scope--glm53-gguf-unsloth-udq4kxl.json").read_text(encoding="utf-8"))
    strip = lambda rows_: [{k: v for k, v in r.items() if k != "note"} for r in rows_]  # noqa: E731
    assert strip(committed["assignments"]) == strip(rows_scope), "the committed scope drifted from the header table"
    assert committed["head_policy"] == "quantized" and committed["mtp_included"] is True
    allow = gguf_scope.mtp_allowlist(surface)
    committed_allow = json.loads((TOOLS / "layer-outer-evidence" /
                                  "gguf-unsloth-glm53-layer78-unexpected-keys.json").read_text(encoding="utf-8"))
    assert allow == committed_allow and len(allow) == 791
    assert hashlib.sha256(json.dumps(sorted(allow), separators=(",", ":")).encode()).hexdigest() == \
        "61e5f26aed8bca408c5de5347d8e1668b0c5716237dad1fc98c47bc108f4ae57"
    passed.append(
        "8e gguf_scope: the committed UD-Q4_K_XL scope equals the rows re-derived from the header "
        "table (13 classes, one row per (class, range), measured bits: experts 4.8611, head 8.5) "
        "and the committed 791-name MTP allowlist equals the header census (digest 61e5f26a...)")
    return passed


def _mock_official_side(scratch: Path) -> "tuple":
    """The official side of a run, from the REAL committed config + index.

    stream_score's BF16 identity gate hashes both files and compares them to
    the sealed inventory, so this exercises that gate for real rather than
    around it -- only the SHARDS are absent, and a --dry-run reads none.
    """
    import hashlib

    bf16 = scratch / "bf16"
    bf16.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DIONE_EVIDENCE / "bf16-config.json", bf16 / "config.json")
    shutil.copyfile(DIONE_EVIDENCE / "bf16-index.json", bf16 / "model.safetensors.index.json")
    inventory = {
        "schema": "quant-pipeline.glm-release-inventory.v1",
        "model_repo": "zai-org/GLM-5.3-Flash-BF16",
        "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
        "seal_mode": "full-shard-sha256",
        "config_sha256": _sha256_file(bf16 / "config.json"),
        "index_sha256": _sha256_file(bf16 / "model.safetensors.index.json"),
    }
    inventory["inventory_sha256"] = hashlib.sha256(_canonical_json(inventory)).hexdigest()
    inventory_path = scratch / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    teacher = scratch / "teacher"
    teacher.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    rows, artifacts = [], []
    for index in range(2):
        wid = "final-%04d" % index
        tokens = rng.integers(0, 154880, size=64, dtype=np.int64)
        mask = np.ones(64, dtype=np.int64)
        mask[:index] = 0
        token_path = teacher / ("tokens-%s.npy" % wid)
        mask_path = teacher / ("mask-%s.npy" % wid)
        np.save(token_path, tokens, allow_pickle=False)
        np.save(mask_path, mask, allow_pickle=False)
        digests = {}
        for path in (token_path, mask_path):
            digest = _sha256_file(path)
            artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": digest})
            digests[path.name] = digest
        rows.append({
            "window_id": wid, "document_id": "doc-%d" % index, "domain": "selftest",
            "role": "final", "token_ids_sha256": digests[token_path.name],
            "attention_mask_sha256": digests[mask_path.name],
            "prediction_positions": int((mask[:-1].astype(bool) & mask[1:].astype(bool)).sum())})
    (teacher / "token-panel.json").write_text(
        json.dumps({"schema": "quant-pipeline.glm53-token-panel.v1", "windows": rows}),
        encoding="utf-8")
    panel_digest = _sha256_file(teacher / "token-panel.json")
    artifacts.append({"path": str(teacher / "token-panel.json"),
                      "bytes": (teacher / "token-panel.json").stat().st_size,
                      "sha256": panel_digest})
    receipt = {"schema": "quant-pipeline.glm53-token-panel-receipt.v1",
               "token_panel_artifact_sha256": panel_digest, "artifacts": artifacts}
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    (teacher / "panel-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return bf16, inventory_path, teacher


def _stream_score_dry_run(scratch: Path, pipeline_root: str, files, env) -> str:
    """Rung 7b: the full CLI reaches plan-print with --source gguf."""
    bf16, inventory_path, teacher = _mock_official_side(scratch)
    argv = [sys.executable, str(TOOLS / "stream_score.py"),
            "--source", "gguf", "--profile", "gguf"]
    for path in files:
        argv += ["--gguf-file", str(path)]
    argv += ["--gguf-repo", "unsloth/GLM-5.3-Flash-GGUF",
             "--gguf-revision", "2975ab414d30340466d8c51533c6e91f0cca64c1",
             "--skip-gguf-hashes",
             "--bf16", str(bf16), "--inventory", str(inventory_path),
             "--teacher", str(teacher), "--cold-run", "1",
             "--out", str(scratch / "out-dry"), "--pipeline-root", pipeline_root, "--dry-run"]
    run = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-3000:]
    plan = json.loads(run.stdout.splitlines()[-1])
    assert plan["weight_source"] == "gguf" and plan["dry_run"] is True
    assert plan["nonrouted_policy"].startswith("decoded_from_the_same_gguf_artifact")
    assert plan["bits"] is None
    assert plan["gguf_routed_layout"]["routed_tensor_count"] == 42 * 288 * 3
    assert plan["gguf_routed_layout"]["nonrouted_bijection"]["bijection_ok"] is True
    assert plan["gguf_surface"]["tensor_count"] == 1412
    assert plan["streaming_disclosure"]["scope_policy"]["embeddings_type"] == "Q8_0"
    assert len(plan["checkpoint_identity_sha256"]) == 64

    # and the profile pairing is enforced end to end
    bad = subprocess.run([a if a != "gguf" or i != argv.index("--profile") + 1 else "k6"
                          for i, a in enumerate(argv)],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 1, "a --source gguf / --profile k6 pair was accepted"
    return ("7b dry-run: stream_score.py --source gguf reaches plan-print over the REAL "
            "config+index (identity gate hashed them), and refuses --profile k6")


if __name__ == "__main__":
    raise SystemExit(main())
