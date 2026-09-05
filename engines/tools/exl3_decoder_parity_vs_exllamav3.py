#!/usr/bin/env python3
"""Real-tensor parity of `exl3hf_surface.decode_payload_hf` against exllamav3's OWN reconstruction.

    engines/tools/exl3_decoder_parity_vs_exllamav3.py --install \\
        --out engines/tools/layer-outer-evidence/exl3-decoder-parity-vs-exllamav3.json

One command on a CUDA host (python 3.12, torch 2.11.0 cu12x/cu13x): it installs
the pinned exllamav3 v1.4.2 release wheel (URL + sha256 recorded below and in
the receipt), range-fetches ONLY the payload tensors of the modules in
MODULE_PLAN (a few MB per module, never a shard), decodes each with our
decoder and with exllamav3's `LinearEXL3` (the class its loader builds from
stored `trellis/suh/svh/<codebook>` tensors, `modules/linear.py::load_exl3`),
and writes the receipt `bin/fidelity/dscompare.py` reads to word the
`weights_reconstructed` caveat.

Two stages are compared, because exllamav3 rounds differently from us:

* pre_hadamard: `LinearEXL3.get_inner_weight_tensor()` = `exllamav3_ext.reconstruct`
  (`exllamav3_ext/quant/reconstruct.cu`), the trellis unpack + codebook + tile
  layout as fp16 [in, out]. Pure integer/LUT work on both sides, so this stage
  is asserted BITWISE. It is the stage the served GEMM kernel consumes.
* weight: `LinearEXL3.get_weight_tensor()` = the above, then exllamav3's
  `preapply_had_l` (fp32 matmul, cast to fp16), `*= suh` (fp16), `preapply_had_r`
  (fp32, cast to fp16), `*= svh` (fp16): four fp16 roundings. Ours keeps fp32
  through both Hadamards and rounds once, so the fp16-cast weights may differ
  at the fp16 ULP; the receipt reports torch.equal, max_abs_diff and the count
  of differing elements in fp16 and against our fp32 output.

Per module the receipt also carries a COMMITTED WINDOW: the first 8x8 trellis
tiles plus the matching 128 `suh`/`svh` values (base64) and the digests of
exllamav3's two outputs on that window, so `selftest_exl3hf_offline.py` can
re-assert the pre_hadamard stage bitwise on any host, and the whole result on
a host where `import exllamav3` succeeds.

Why this needs a GPU: exllamav3 1.4.2 has no CPU reconstruction. The PyPI
wheel is pure python and `import exllamav3` (`exllamav3/__init__.py` ->
`model.config` -> ... -> `exllamav3/ext.py:147`) JIT-builds the CUDA extension
through `torch.utils.cpp_extension.load` unless a precompiled `exllamav3_ext`
is installed; without a toolkit that dies in `_join_cuda_home` ("CUDA_HOME
environment variable is not set"). With one, `reconstruct` is still a
`__global__` kernel launched on the current CUDA stream
(`reconstruct.cu:11-84,108-109`); `exllamav3_ext/cpu/` holds an int8 MoE GEMM,
not a reconstruct. `--fetch-only` runs the fetch and our half anywhere.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import exl3hf_surface as xs  # noqa: E402

SCHEMA = "malaiwah.exl3-decoder-parity-vs-exllamav3.v1"
DEFAULT_OUT = TOOLS / "layer-outer-evidence" / "exl3-decoder-parity-vs-exllamav3.json"

EXLLAMAV3_VERSION = "1.4.2"
EXLLAMAV3_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"  # git tag v1.4.2
EXLLAMAV3_RELEASE = "https://github.com/turboderp-org/exllamav3/releases/download/v1.4.2/"
# Release wheels carrying a precompiled `exllamav3_ext` (no JIT build), keyed by the
# CUDA major of the torch that will import them. Digests taken 2026-09-05.
EXLLAMAV3_WHEELS = {
    12: ("exllamav3-1.4.2+cu128.torch2.11.0-cp312-cp312-linux_x86_64.whl",
         "1cca2df47f671938a3ee508cad8eaae9e170a7202f541021b13c9803a0a0550a"),
    13: ("exllamav3-1.4.2+cu132.torch2.11.0-cp312-cp312-linux_x86_64.whl",
         "fb131e9c97ec270f5d72e28e4331197b0360fa55f10c67d87b6418a0a029fc7d"),
}
WHEEL_TORCH = "2.11.0"
WHEEL_PYTHON = (3, 12)
RECONSTRUCT_ENTRYPOINT = (
    "exllamav3.modules.quant.exl3.LinearEXL3.get_weight_tensor "
    "(get_inner_weight_tensor -> exllamav3_ext.reconstruct, exllamav3_ext/quant/reconstruct.cu; "
    "then exl3_lib.quantize.preapply_had_l/_r and the suh/svh fp16 multiplies)"
)

WINDOW_TILES = 8  # 8 x 8 tiles = one 128 x 128 Hadamard block, the smallest self-contained window

# (label, repo, revision, shard, module, codebook[, objects]). One module per (release, K)
# for davidsyoung, the two Fruit expert-0 modules the reconstruction receipt
# already names, and one drowzeys module per codebook. K is read from the
# header and recorded; it is not an input. The optional 7th element names
# where an object lives when it is NOT `<module>.<object>` in the same shard:
# {"suh"|"svh": (shard, tensor name)} -- the layer-shared rotation vectors of
# the GLM-5.2 layouts (`layer_outer.exl3_rotation_groups`; evidence in
# layer-outer-evidence/glm52-exl3-layouts-parity.json), which exllamav3's
# LinearEXL3 takes as plain suh/svh tensors once resolved by name.
MODULE_PLAN: Tuple[Tuple[Any, ...], ...] = (
    ("fruit", "malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("fruit", "malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.gate_proj.rank0", "mcg"),
    ("dy30", "davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw", "eeab94eb6e95b4e4d13d94af55ab3c420d6f52d3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("dy325", "davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "6d6bd738c0c1635513e0bd0fdf0302049bd820a9",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("dy325", "davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "6d6bd738c0c1635513e0bd0fdf0302049bd820a9",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.3.down_proj.rank0", "mcg"),
    # 3.42's layer-3 expert-0/3 down_proj rank0 are byte-identical to 3.25's (same tier,
    # same atoms), so 3.42 contributes an expert that is K4 here and K3 in 3.25, on rank 1.
    ("dy342", "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw", "99c6f951333d2b38f1efefa533c7afadf0d376e3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.20.gate_proj.rank1", "mcg"),
    ("dy342", "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw", "99c6f951333d2b38f1efefa533c7afadf0d376e3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.3.up_proj.rank1", "mcg"),
    ("drowzeys", "drowzeys/keys-GLM-5.3-EXL3", "ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5",
     "model-00001-of-00041.safetensors", "model.layers.3.mlp.experts.0.gate_proj", "mcg"),
    ("drowzeys", "drowzeys/keys-GLM-5.3-EXL3", "ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5",
     "model-00002-of-00041.safetensors", "model.layers.4.mlp.experts.0.gate_proj", "mul1"),
    # GLM-5.2 shared_h_v1: the down_proj rank's svh is the layer's shared vector.
    ("willfalco", "willfalco/GLM-5.2-EXL3-TR3-3.42bpw", "700c99dfa75d61cba4dda1ce9a36478bc217728d",
     "model-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj.rank0", "mcg",
     {"svh": ("model-layer-010.safetensors",
              "model.layers.10.mlp.experts.shared_h.down_proj.rank0.svh")}),
    # jpsequeira keeps the shared vectors in their own shard; plus its exl3 wq_b (K6).
    ("jpsequeira", "jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2",
     "b92479840ef92fbeb7d774187f91cf5a2a659ade",
     "projection-mixed-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj.rank0", "mcg",
     {"svh": ("shared-h-layer-010.safetensors",
              "model.layers.10.mlp.experts.shared_h.down_proj.rank0.svh")}),
    ("jpsequeira", "jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2",
     "b92479840ef92fbeb7d774187f91cf5a2a659ade",
     "exl3-exemption-layer-010.safetensors", "model.layers.10.self_attn.indexer.wq_b", "mcg"),
    # brandonmusic r7_shared: unsharded experts; gate_up_suh serves gate AND up, down_svh down.
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj", "mcg",
     {"svh": ("r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.r7_shared.down_svh")}),
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.0.up_proj", "mcg",
     {"suh": ("r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.r7_shared.gate_up_suh")}),
    # brandonmusic's dense-6 non-routed module (K6), stock layout.
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "model-layer-010.safetensors", "model.layers.10.self_attn.q_b_proj", "mcg"),
)

_NP_DTYPE = {"I16": "<i2", "I32": "<i4", "F16": "<f2"}
_EXPECTED_DTYPE = {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32", "mul1": "I32"}


class ParityError(RuntimeError):
    pass


def _fail(message: str) -> ParityError:
    return ParityError(f"exl3_decoder_parity_vs_exllamav3: {message}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(value) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(value.detach().cpu().contiguous().numpy()).tobytes())


# --------------------------------------------------------------------------
# ranged fetch (header + exact tensor byte spans; never a shard)
# --------------------------------------------------------------------------
def _http(url: str, start: Optional[int] = None, end: Optional[int] = None, tries: int = 4) -> bytes:
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
                data = response.read()
            if start is not None and len(data) != end - start + 1:
                raise _fail(f"range {start}-{end} of {url} returned {len(data)} bytes")
            return data
        except ParityError:
            raise
        except Exception:  # noqa: BLE001 - retried, re-raised on the last attempt
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise _fail("unreachable")


def shard_header(cache: Path, repo: str, revision: str, shard: str) -> Dict[str, Any]:
    """The safetensors header of one shard by two range requests, cached on disk."""
    if xs._REVISION.fullmatch(revision) is None:
        raise _fail(f"{repo}: revision must be the immutable 40-hex commit, got {revision!r}")
    path = cache / repo.replace("/", "__") / revision / f"{shard}.header.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
    (length,) = struct.unpack("<Q", _http(url, 0, 7))
    header = json.loads(_http(url, 8, 8 + length - 1))
    header["__header_len__"] = length
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
    return header


def fetch_tensor(cache: Path, repo: str, revision: str, shard: str, header: Dict[str, Any], name: str,
                 obj: Optional[str] = None):
    """One tensor's exact bytes by range -> (torch tensor, sha256, byte count).

    `obj` names the payload object the tensor stands for when the tensor's own
    suffix does not (a layer-shared `down_svh` / `gate_up_suh` / `rank0.svh`)."""
    import numpy as np
    import torch

    entry = header.get(name)
    if entry is None:
        raise _fail(f"{repo}@{revision[:8]} {shard} has no tensor {name}")
    expected = _EXPECTED_DTYPE[obj or name.rsplit(".", 1)[1]]
    if entry["dtype"] != expected:
        raise _fail(f"{name} is {entry['dtype']}, expected {expected}")
    start, end = entry["data_offsets"]
    path = cache / repo.replace("/", "__") / revision / f"{name}.bin"
    if path.exists():
        raw = path.read_bytes()
        if len(raw) != end - start:
            raise _fail(f"cached {path} is {len(raw)} bytes, header says {end - start}; delete it")
    else:
        base = 8 + int(header["__header_len__"])
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
        raw = _http(url, base + start, base + end - 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    array = np.frombuffer(raw, dtype=_NP_DTYPE[entry["dtype"]]).copy().reshape(entry["shape"])
    return torch.from_numpy(array), sha256_bytes(raw), len(raw)


# --------------------------------------------------------------------------
# our decoder, and the pre-Hadamard stage it passes through
# --------------------------------------------------------------------------
def ours_pre_hadamard(trellis, codebook: str):
    """fp16 [in, out]: unpack + codebook LUT + tile layout, exactly as decode_payload_hf
    composes them (exl3hf_surface.py, decode_payload_hf, before the first Hadamard)."""
    import torch

    bits = trellis.shape[-1] // 16
    states = xs.unpack_trellis_states_anybits(trellis, bits)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = xs.codebook_lut(codebook, states.device).index_select(0, indices.flatten()).reshape_as(states)
    values = values.index_select(-1, torch.argsort(xs._permutation(states.device)))
    k_tiles, n_tiles, _ = values.shape
    return values.reshape(k_tiles, n_tiles, 16, 16).permute(0, 2, 1, 3).reshape(k_tiles * 16, n_tiles * 16).contiguous()


def ours_weight(trellis, suh, svh, codebook: str):
    """fp32 [in, out] (exllamav3 orientation) from decode_payload_hf's [out, in]."""
    return xs.decode_payload_hf(trellis, suh, svh, codebook=codebook).T.contiguous()


def compare(ours, theirs) -> Dict[str, Any]:
    """ours and theirs same shape; equality in ours' dtype, differences in fp32."""
    import torch

    if tuple(ours.shape) != tuple(theirs.shape):
        raise _fail(f"shape mismatch ours {tuple(ours.shape)} theirs {tuple(theirs.shape)}")
    theirs = theirs.to(ours.device)
    diff = (ours.float() - theirs.float()).abs()
    return {
        "equal": bool(torch.equal(ours, theirs.to(ours.dtype))),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "differing_elements": int((ours != theirs.to(ours.dtype)).sum().item()),
        "elements": int(ours.numel()),
    }


def window_of(trellis, suh, svh):
    k = min(WINDOW_TILES, trellis.shape[0])
    n = min(WINDOW_TILES, trellis.shape[1])
    if k != WINDOW_TILES or n != WINDOW_TILES:
        raise _fail(f"module smaller than one {WINDOW_TILES}x{WINDOW_TILES}-tile window: {tuple(trellis.shape)}")
    return (trellis[:k, :n, :].contiguous(), suh[: k * 16].contiguous(), svh[: n * 16].contiguous())


def b64(tensor) -> str:
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(tensor.cpu().numpy()).tobytes()).decode("ascii")

def unb64(text: str, dtype: str, shape) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.frombuffer(base64.b64decode(text), dtype=_NP_DTYPE[dtype]).copy().reshape(shape))


# --------------------------------------------------------------------------
# exllamav3: install the pinned wheel, import, reconstruct
# --------------------------------------------------------------------------
def wheel_for_this_stack():
    import torch

    if sys.version_info[:2] != WHEEL_PYTHON:
        raise _fail(f"release wheels are cp{WHEEL_PYTHON[0]}{WHEEL_PYTHON[1]}; this is python {platform.python_version()}")
    if not torch.__version__.startswith(WHEEL_TORCH):
        raise _fail(f"release wheels are built for torch {WHEEL_TORCH}; this is {torch.__version__}")
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise _fail("no CUDA device: exllamav3's reconstruct is a CUDA kernel (see module docstring)")
    major = int(torch.version.cuda.split(".")[0])
    if major not in EXLLAMAV3_WHEELS:
        raise _fail(f"no pinned wheel for CUDA {torch.version.cuda}")
    name, digest = EXLLAMAV3_WHEELS[major]
    return {"name": name, "url": EXLLAMAV3_RELEASE + name.replace("+", "%2B"), "sha256": digest,
            "cuda_tag": name.split("+")[1].split(".")[0]}


def install_exllamav3(log) -> Dict[str, Any]:
    wheel = wheel_for_this_stack()
    with tempfile.TemporaryDirectory(prefix="exl3wheel-") as tmp:
        path = Path(tmp) / wheel["name"]
        log(f"fetching {wheel['url']}")
        with urllib.request.urlopen(urllib.request.Request(wheel["url"]), timeout=600) as response, path.open("wb") as out:
            digest = hashlib.sha256()
            for chunk in iter(lambda: response.read(1 << 22), b""):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != wheel["sha256"]:
            raise _fail(f"wheel digest {digest.hexdigest()} != pinned {wheel['sha256']}")
        log(f"wheel sha256 verified; pip install {wheel['name']}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", str(path)], check=True)
    wheel["installed_by"] = f"{sys.executable} -m pip install <verified wheel>"
    return wheel


def import_exllamav3(expect_precompiled: bool = True):
    """Import exllamav3 and prove it is the pinned version with a precompiled extension."""
    try:
        exllamav3 = importlib.import_module("exllamav3")
    except Exception as exc:  # noqa: BLE001 - the receipt names the failure
        raise _fail(f"import exllamav3 failed: {type(exc).__name__}: {exc}") from exc
    version = getattr(exllamav3, "__version__", None) or importlib.import_module("exllamav3.version").__version__
    if version != EXLLAMAV3_VERSION:
        raise _fail(f"exllamav3 {version} imported; this parity pins {EXLLAMAV3_VERSION}")
    ext = importlib.import_module("exllamav3.ext").exllamav3_ext
    ext_file = getattr(ext, "__file__", "") or ""
    precompiled = bool(ext_file) and Path(ext_file).resolve().parent == Path(exllamav3.__file__).resolve().parent.parent
    if expect_precompiled and not precompiled:
        raise _fail(f"exllamav3_ext at {ext_file!r} is not the release wheel's precompiled extension")
    exl3 = importlib.import_module("exllamav3.modules.quant.exl3")
    return exllamav3, exl3, {"version": version, "package_file": exllamav3.__file__,
                             "extension_file": ext_file, "extension_precompiled": precompiled}


def theirs_reconstruct(exl3, trellis, suh, svh, codebook: str, marker, device):
    """(pre_hadamard fp16 [in,out], weight fp16 [in,out]) through exllamav3's LinearEXL3."""
    trellis = trellis.to(device)
    suh = suh.to(device)
    svh = svh.to(device)
    linear = exl3.LinearEXL3(
        None, trellis.shape[0] * 16, trellis.shape[1] * 16,
        suh=suh, svh=svh, trellis=trellis,
        mcg=marker if codebook == "mcg" else None,
        mul1=marker if codebook == "mul1" else None,
    )
    if int(linear.K) != trellis.shape[-1] // 16 or bool(linear.mcg) != (codebook == "mcg") or bool(linear.mul1) != (codebook == "mul1"):
        raise _fail("LinearEXL3 did not adopt the payload's K/codebook")
    pre = linear.get_inner_weight_tensor().contiguous()
    weight = linear.get_weight_tensor().contiguous()
    return pre.cpu(), weight.cpu()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def module_record(cache: Path, plan_row, log) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch one module, decode it with our decoder; return (record, tensors)."""
    import torch

    label, repo, revision, shard, module, codebook = plan_row[:6]
    objects = dict(plan_row[6]) if len(plan_row) > 6 else {}
    header = shard_header(cache, repo, revision, shard)
    tensors, digests, fetched = {}, {}, 0
    for obj in ("trellis", "suh", "svh", codebook):
        obj_shard, obj_name = objects.get(obj, (shard, f"{module}.{obj}"))
        obj_header = header if obj_shard == shard else shard_header(cache, repo, revision, obj_shard)
        tensors[obj], digests[obj], n = fetch_tensor(cache, repo, revision, obj_shard, obj_header,
                                                     obj_name, obj=obj)
        fetched += n
    marker = int(tensors[codebook].reshape(-1)[0])
    if marker != xs.CODEBOOK_OBJECTS[codebook]:
        raise _fail(f"{module}: {codebook} marker {marker} != {xs.CODEBOOK_OBJECTS[codebook]}")
    trellis, suh, svh = tensors["trellis"], tensors["suh"], tensors["svh"]
    bits = trellis.shape[-1] // 16
    in_features, out_features = trellis.shape[0] * 16, trellis.shape[1] * 16
    if suh.numel() != in_features or svh.numel() != out_features:
        raise _fail(f"{module}: suh/svh lengths {suh.numel()}/{svh.numel()} != {in_features}/{out_features}")
    pre = ours_pre_hadamard(trellis, codebook)
    weight = ours_weight(trellis, suh, svh, codebook)
    w_trellis, w_suh, w_svh = window_of(trellis, suh, svh)
    log(f"{label} {module} K{bits} {codebook} [{in_features},{out_features}] fetched {fetched} B")
    record = {
        "label": label, "repo": repo, "revision": revision, "shard": shard, "name": module,
        "codebook": codebook, "K": bits, "marker": marker,
        "shape_in_out": [in_features, out_features], "elements": int(in_features * out_features),
        "input_sha256": digests, "input_bytes": fetched,
        "objects": {obj: {"shard": s, "name": n} for obj, (s, n) in objects.items()},
        "ours": {"pre_hadamard_sha256": sha256_tensor(pre),
                 "weight_fp16_sha256": sha256_tensor(weight.to(torch.float16))},
        "window": {
            "k_tiles": WINDOW_TILES, "n_tiles": WINDOW_TILES,
            "trellis_shape": list(w_trellis.shape), "trellis_i16_b64": b64(w_trellis),
            "suh_f16_b64": b64(w_suh), "svh_f16_b64": b64(w_svh),
            "ours_pre_hadamard_sha256": sha256_tensor(ours_pre_hadamard(w_trellis, codebook)),
        },
    }
    return record, {"trellis": trellis, "suh": suh, "svh": svh, "marker": tensors[codebook],
                    "pre": pre, "weight": weight, "window": (w_trellis, w_suh, w_svh)}


def run_parity(exl3, record: Dict[str, Any], tensors: Dict[str, Any], device) -> None:
    import torch

    codebook = record["codebook"]
    t_pre, t_weight = theirs_reconstruct(exl3, tensors["trellis"], tensors["suh"], tensors["svh"],
                                         codebook, tensors["marker"], device)
    record["pre_hadamard"] = compare(tensors["pre"], t_pre)
    fp16 = compare(tensors["weight"].to(torch.float16), t_weight)
    record["fp32"] = compare(tensors["weight"], t_weight.float())
    record["equal"] = fp16.pop("equal")
    record.update(fp16)
    record["exllamav3"] = {"pre_hadamard_sha256": sha256_tensor(t_pre), "weight_fp16_sha256": sha256_tensor(t_weight)}
    w_trellis, w_suh, w_svh = tensors["window"]
    w_pre, w_weight = theirs_reconstruct(exl3, w_trellis, w_suh, w_svh, codebook, tensors["marker"], device)
    window = record["window"]
    window["exllamav3_pre_hadamard_sha256"] = sha256_tensor(w_pre)
    window["exllamav3_weight_fp16_sha256"] = sha256_tensor(w_weight)
    window["pre_hadamard"] = compare(ours_pre_hadamard(w_trellis, codebook), w_pre)
    window["weight_fp16"] = compare(ours_weight(w_trellis, w_suh, w_svh, codebook).to(torch.float16), w_weight)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("FIDELITY_SCRATCH", tempfile.gettempdir())) / "exl3parity-cache",
                        help="fetched payload bytes and shard headers (re-runs are offline)")
    parser.add_argument("--install", action="store_true",
                        help="pip-install the pinned v1.4.2 release wheel for this torch/CUDA first")
    parser.add_argument("--fetch-only", action="store_true",
                        help="fetch + our half only; prints digests, writes nothing (no exllamav3 needed)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(f"[exl3-parity] {message}", flush=True)

    import torch

    started = time.monotonic()
    records: List[Dict[str, Any]] = []
    tensors: List[Dict[str, Any]] = []
    for row in MODULE_PLAN:
        record, held = module_record(args.cache_dir, row, log)
        records.append(record)
        tensors.append(held)
    if args.fetch_only:
        for record in records:
            print(json.dumps({k: record[k] for k in ("label", "name", "K", "codebook", "input_sha256", "ours")}, sort_keys=True))
        log(f"fetch-only: {len(records)} modules, {sum(r['input_bytes'] for r in records)} bytes, "
            f"{time.monotonic() - started:.1f}s; nothing written")
        return 0

    wheel = install_exllamav3(log) if args.install else None
    _, exl3, imported = import_exllamav3(expect_precompiled=args.install)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise _fail("--device must be a CUDA device (see module docstring)")
    for record, held in zip(records, tensors):
        run_parity(exl3, record, held, device)
        log(f"{record['name']}: pre_hadamard equal={record['pre_hadamard']['equal']} "
            f"fp16 equal={record['equal']} max_abs_diff={record['max_abs_diff']:.3e} "
            f"differing={record['differing_elements']}/{record['elements']}")
    for held in tensors:
        held.clear()

    receipt = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "engines/tools/exl3_decoder_parity_vs_exllamav3.py",
        "exllamav3_version": EXLLAMAV3_VERSION,
        "exllamav3_commit": EXLLAMAV3_COMMIT,
        "exllamav3": {**imported, "wheel": wheel, "reconstruct_entrypoint": RECONSTRUCT_ENTRYPOINT},
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "python_version": platform.python_version(),
        "ours": {"module": "engines/tools/exl3hf_surface.py", "function": "decode_payload_hf",
                 "code_sha256": xs._sha256_file(TOOLS / "exl3hf_surface.py"),
                 "mcg_lut_sha256": xs.MCG_LUT_SHA256},
        "modules_compared": len(records),
        "codebooks": sorted({r["codebook"] for r in records}),
        "k_values": sorted({r["K"] for r in records}),
        "all_bitwise_pre_hadamard": all(r["pre_hadamard"]["equal"] for r in records),
        "all_bitwise": all(r["equal"] for r in records),
        "modules": records,
        "note": (
            "pre_hadamard compares exllamav3_ext.reconstruct (fp16 [in,out]) with our unpack+LUT+tile "
            "layout bitwise. weight compares LinearEXL3.get_weight_tensor (four fp16 roundings) with "
            "decode_payload_hf cast to fp16 once; fp32 compares against our unrounded output. "
            "window fields are the committed 8x8-tile inputs selftest_exl3hf_offline.py re-asserts."
        ),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    log(f"wrote {args.out}: modules={len(records)} all_bitwise_pre_hadamard={receipt['all_bitwise_pre_hadamard']} "
        f"all_bitwise={receipt['all_bitwise']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
