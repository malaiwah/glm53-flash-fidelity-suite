#!/usr/bin/env python3
"""Real-tensor parity of the exl3 ROTATION LAYOUTS against the BF16 source.

    engines/tools/exl3_layout_parity.py \\
        --out engines/tools/layer-outer-evidence/glm52-exl3-layouts-parity.json

For each layout the streaming decoder speaks (`layer_outer.exl3_rotation_groups`:
stock `per_module`, willfalco/jpsequeira `shared_h_v1`, brandonmusic `r7_shared`)
this range-fetches ONE real module's payload objects -- and, under a shared
layout, the layer-shared vector its group resolves to BY NAME -- runs them
through the REAL pod path (`trellis_checkpoint_plan` on the artifact's own
config.json + `materialize_trellis_subset`), and compares the decoded bf16
weight against the same rows of the official BF16 release the artifact was
encoded from (every tail names `source_index_sha256` 5fd47a92..., the
zai-org/GLM-5.2 index). A correct trellis reconstruction sits at the K's own
error -- rel_l2 ~0.15 at K3, ~0.07-0.09 at K4, ~0.02 at K6, ~0.006 at K8,
halving per bit (fruit K4 0.067, glm53 tp K4 0.067) -- while a wrong vector, a
wrong layout, an undone permutation or a wrong unpack sits near cosine 0.

Negative control per shared layout: the same module decoded with the SAME
projection's shared vector from the NEXT layer (a real vector of the right
shape and dtype, from the wrong layer) must fall to cosine ~0 -- the proof that
the name resolution, not the vector's shape, is what makes the decode right.

Large modules are decoded on a WINDOW of whole 128-row output blocks: the
decode applies a 128x128 Hadamard per axis block, so the first n_tiles (a
multiple of 8) of the trellis with the matching svh prefix decode to exactly
the first 16*n_tiles output rows (selftest_exl3hf_offline asserts the window
identity); the official rows are fetched by the same range.

Never a shard: header + exact tensor byte spans only (~400 MB total).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
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
import layer_outer as lo  # noqa: E402

SCHEMA = "malaiwah.exl3-rotation-layout-parity.v1"
DEFAULT_OUT = TOOLS / "layer-outer-evidence" / "glm52-exl3-layouts-parity.json"
OFFICIAL = ("zai-org/GLM-5.2", "cf457fa734ab149ffef225f80893eb38c6ff5cdc")
OFFICIAL_INDEX_SHA256 = "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"

#: (label, repo, revision). The three GLM-5.2 candidates whose layouts the
#: decoder learned to speak, at the revisions the plan was proven on.
ARTIFACTS = {
    "willfalco": ("willfalco/GLM-5.2-EXL3-TR3-3.42bpw",
                  "700c99dfa75d61cba4dda1ce9a36478bc217728d"),
    "jpsequeira": ("jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2",
                   "b92479840ef92fbeb7d774187f91cf5a2a659ade"),
    "brandonmusic": ("brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
                     "7c73450f05a151439d0f184f216b1eefcc394a31"),
}

#: One case = one decoded weight compared against the official rows.
#: module: the weight the pod path emits; layer: for the wrong-layer control;
#: window_rows: decode/compare only the first N output rows (None = whole).
CASES: Tuple[Dict[str, Any], ...] = (
    {"artifact": "willfalco", "layout": "shared_h_v1", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.down_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "willfalco", "layout": "shared_h_v1", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.gate_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "jpsequeira", "layout": "shared_h_v1", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.down_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "jpsequeira", "layout": "per_module", "layer": 10,
     "module": "model.layers.10.self_attn.indexer.wq_b", "control": None},
    {"artifact": "jpsequeira", "layout": "per_module", "layer": None,
     "module": "lm_head", "window_rows": 2048, "control": None},
    {"artifact": "brandonmusic", "layout": "r7_shared", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.down_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "brandonmusic", "layout": "r7_shared", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.gate_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "brandonmusic", "layout": "r7_shared", "layer": 10,
     "module": "model.layers.10.mlp.experts.0.up_proj", "control": "wrong-layer-shared-vector"},
    {"artifact": "brandonmusic", "layout": "per_module", "layer": 10,
     "module": "model.layers.10.self_attn.o_proj", "window_rows": 1024, "control": None},
    {"artifact": "brandonmusic", "layout": "per_module", "layer": 78,
     "module": "model.layers.78.mlp.experts.0.down_proj", "control": None},
)

_NP_DTYPE = {"I16": "<i2", "I32": "<i4", "F16": "<f2", "BF16": "<u2"}
#: rel_l2 ceiling per K for a decode to count as IN BAND: about 1.3x the error
#: real GLM-5.x modules show at that K (fruit/glm53 K4 0.067; this receipt's
#: ladder K3 0.14-0.17, K4 0.088, K6 0.021-0.023, K8 0.0057). A wrong vector,
#: layout or permutation gives rel_l2 ~1.41 (cosine ~0), two decades away.
IN_BAND_REL_L2 = {2: 0.45, 3: 0.22, 4: 0.12, 5: 0.07, 6: 0.035, 8: 0.010}


class ParityError(RuntimeError):
    pass


def _fail(message: str) -> ParityError:
    return ParityError("exl3_layout_parity: %s" % message)


# --------------------------------------------------------------------------
# ranged fetch (header + exact tensor byte spans; never a shard)
# --------------------------------------------------------------------------
def _http(url: str, start: Optional[int] = None, end: Optional[int] = None, tries: int = 4) -> bytes:
    headers = {}
    token = os.environ.get("HF_TOKEN_FILE")
    if token and Path(token).is_file():
        headers["Authorization"] = "Bearer " + Path(token).read_text().strip()
    if start is not None:
        headers["Range"] = "bytes=%d-%d" % (start, end)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
                data = response.read()
            if start is not None and len(data) != end - start + 1:
                raise _fail("range %d-%d of %s returned %d bytes" % (start, end, url, len(data)))
            return data
        except ParityError:
            raise
        except Exception:  # noqa: BLE001 - retried, re-raised on the last attempt
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise _fail("unreachable")


class Fetcher:
    def __init__(self, cache: Path):
        self.cache = cache
        self.indexes: Dict[Tuple[str, str], Dict[str, str]] = {}
        self.headers: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.bytes_fetched = 0

    def _path(self, repo: str, revision: str, name: str) -> Path:
        return self.cache / repo.replace("/", "__") / revision / name

    def json_file(self, repo: str, revision: str, name: str) -> Any:
        path = self._path(repo, revision, name)
        if not path.exists():
            data = _http("https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, name))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return json.loads(path.read_text(encoding="utf-8"))

    def weight_map(self, repo: str, revision: str) -> Dict[str, str]:
        key = (repo, revision)
        if key not in self.indexes:
            self.indexes[key] = self.json_file(repo, revision, "model.safetensors.index.json")["weight_map"]
        return self.indexes[key]

    def header(self, repo: str, revision: str, shard: str) -> Dict[str, Any]:
        key = (repo, revision, shard)
        if key in self.headers:
            return self.headers[key]
        path = self._path(repo, revision, shard + ".header.json")
        if path.exists():
            header = json.loads(path.read_text(encoding="utf-8"))
        else:
            url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
            (length,) = struct.unpack("<Q", _http(url, 0, 7))
            header = json.loads(_http(url, 8, 8 + length - 1))
            header["__header_len__"] = length
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
        self.headers[key] = header
        return header

    def tensor(self, repo: str, revision: str, name: str, rows: Optional[int] = None,
               n_tiles: Optional[int] = None):
        """One tensor's exact bytes -> (torch tensor, sha256, bytes, shard, dtype, shape).

        `rows` keeps the first rows of a 2-D tensor (one contiguous range);
        `n_tiles` keeps the first n-tiles of a [k_tiles, n_tiles, 16K] trellis
        (one range per k_tile over a single kept-alive connection, so a 1 GB
        head costs the 12 MB its window needs).
        """
        import numpy as np
        import torch

        shard = self.weight_map(repo, revision).get(name)
        if shard is None:
            raise _fail("%s@%s has no tensor %s" % (repo, revision[:8], name))
        header = self.header(repo, revision, shard)
        entry = header[name]
        shape = list(entry["shape"])
        start, end = entry["data_offsets"]
        base = 8 + int(header["__header_len__"])
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, revision, shard)
        suffix = ""
        spans: List[Tuple[int, int]]
        if rows is not None:
            if rows > shape[0]:
                raise _fail("%s: %d rows requested of %s" % (name, rows, shape))
            per_row = (end - start) // shape[0]
            spans = [(start, start + rows * per_row)]
            shape[0] = rows
            suffix = ".rows%d" % rows
        elif n_tiles is not None:
            if len(shape) != 3 or n_tiles > shape[1]:
                raise _fail("%s: %d n-tiles requested of %s" % (name, n_tiles, shape))
            per_tile = shape[2] * 2
            per_k = shape[1] * per_tile
            spans = [(start + k * per_k, start + k * per_k + n_tiles * per_tile)
                     for k in range(shape[0])]
            shape[1] = n_tiles
            suffix = ".ntiles%d" % n_tiles
        else:
            spans = [(start, end)]
        path = self._path(repo, revision, name + suffix + ".bin")
        want = sum(b - a for a, b in spans)
        if path.exists() and path.stat().st_size == want:
            raw = path.read_bytes()
        else:
            raw = self._ranges(url, [(base + a, base + b - 1) for a, b in spans])
            self.bytes_fetched += len(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        array = np.frombuffer(raw, dtype=_NP_DTYPE[entry["dtype"]]).copy().reshape(shape)
        tensor = torch.from_numpy(array)
        if entry["dtype"] == "BF16":
            tensor = tensor.view(torch.bfloat16)
        return tensor, hashlib.sha256(raw).hexdigest(), len(raw), shard, entry["dtype"], list(entry["shape"])

    @staticmethod
    def _ranges(url: str, spans: List[Tuple[int, int]]) -> bytes:
        """Concatenated byte ranges; many spans ride one kept-alive connection
        to the resolved (CDN) location."""
        if len(spans) == 1:
            return _http(url, spans[0][0], spans[0][1])
        import http.client
        import urllib.parse

        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as head:
            final = head.geturl()
        parsed = urllib.parse.urlsplit(final)
        target = parsed.path + ("?" + parsed.query if parsed.query else "")
        out = bytearray()
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=300)
        try:
            for index, (a, b) in enumerate(spans):
                for attempt in range(4):
                    try:
                        conn.request("GET", target, headers={"Range": "bytes=%d-%d" % (a, b)})
                        response = conn.getresponse()
                        data = response.read()
                        if response.status not in (200, 206) or len(data) != b - a + 1:
                            raise _fail("span %d/%d (%d-%d): status %d, %d bytes"
                                        % (index + 1, len(spans), a, b, response.status, len(data)))
                        out += data
                        break
                    except (http.client.HTTPException, OSError):
                        if attempt == 3:
                            raise
                        conn.close()
                        conn = http.client.HTTPSConnection(parsed.netloc, timeout=300)
        finally:
            conn.close()
        return bytes(out)

    def groups(self, repo: str, revision: str):
        """The pod's census of the whole index, once per artifact."""
        key = (repo, revision)
        if not hasattr(self, "_groups"):
            self._groups: Dict[Tuple[str, str], Any] = {}
        if key not in self._groups:
            self._groups[key] = lo.exl3_rotation_groups(list(self.weight_map(repo, revision)))
        return self._groups[key]

    def plan(self, repo: str, revision: str, config):
        """`trellis_checkpoint_plan` on the whole index, once per artifact -> (contract, observed)."""
        key = (repo, revision)
        if not hasattr(self, "_plans"):
            self._plans: Dict[Tuple[str, str], Any] = {}
        if key not in self._plans:
            plan = lo.trellis_checkpoint_plan(config, list(self.weight_map(repo, revision)))
            observed = plan.pop("_observed")
            self._plans[key] = (plan, observed)
        contract, observed = self._plans[key]
        return dict(contract), observed


# --------------------------------------------------------------------------
# the pod path on the fetched objects
# --------------------------------------------------------------------------
class _Config:
    def __init__(self, doc: Dict[str, Any]):
        self.quantization_config = doc.get("quantization_config")
        self.hybrid_tr3_tail = doc.get("hybrid_tr3_tail")
        text = doc.get("text_config") or doc
        self.hidden_size = text.get("hidden_size")
        self.moe_intermediate_size = text.get("moe_intermediate_size")


def group_keys(fetch: "Fetcher", repo: str, revision: str, module: str) -> Tuple[List[str], bool]:
    """Every index key of `module`'s payload group(s) -- the stock objects, its
    rank shards and, under a shared layout, the layer-shared vector each group
    resolves to -- found by the SAME census the pod runs on the index; and
    whether any of them resolved a shared vector."""
    groups, _ = fetch.groups(repo, revision)
    keys = set()
    shared = False
    for stem, objects in groups.items():
        if stem == module or stem.startswith(module + ".rank"):
            keys.update(objects[name] for name in ("trellis", "suh", "svh", "marker"))
            shared = shared or objects.get("shared") is not None
    return sorted(keys), shared


def compare(decoded, reference) -> Dict[str, Any]:
    import torch

    if tuple(decoded.shape) != tuple(reference.shape):
        raise _fail("shape mismatch decoded %s reference %s" % (tuple(decoded.shape), tuple(reference.shape)))
    a = decoded.to(torch.float64).reshape(-1)
    b = reference.to(torch.float64).reshape(-1)
    diff = a - b
    return {
        "cosine": float(torch.dot(a, b) / (a.norm() * b.norm())),
        "rel_l2": float(diff.norm() / b.norm()),
        "rel_mse": float((diff * diff).mean() / (b * b).mean()),
        "elements": int(a.numel()),
    }


def run_case(fetch: Fetcher, case: Dict[str, Any], log) -> Dict[str, Any]:
    import torch

    repo, revision = ARTIFACTS[case["artifact"]]
    config_doc = fetch.json_file(repo, revision, "config.json")
    config = _Config(config_doc)
    weight_map = fetch.weight_map(repo, revision)
    module = case["module"]
    keys, module_shared = group_keys(fetch, repo, revision, module)
    if not keys:
        raise _fail("%s: no payload keys for %s" % (case["artifact"], module))
    window = case.get("window_rows")
    subset: Dict[str, Any] = {}
    inputs: Dict[str, Any] = {}
    for key in keys:
        if window is not None and key.endswith(".trellis"):
            tensor, digest, size, shard, dtype, shape = fetch.tensor(
                repo, revision, key, n_tiles=window // 16)
        else:
            tensor, digest, size, shard, dtype, shape = fetch.tensor(repo, revision, key)
            if window is not None and key.endswith(".svh"):
                tensor = tensor[:window].contiguous()
        subset[key] = tensor
        inputs[key] = {"shard": shard, "dtype": dtype, "shape": shape, "sha256": digest,
                       "bytes": size, "window": (None if window is None else
                                                 "first %d n-tiles" % (window // 16)
                                                 if key.endswith(".trellis") else
                                                 "first %d values" % window
                                                 if key.endswith(".svh") else None)}
    # The plan is the pod's: once, on the artifact's own config and the WHOLE
    # index (the declaration cross-checks read every name); the decode of the
    # fetched subset then runs under that plan exactly as a layer's would.
    plan, observed = fetch.plan(repo, revision, config)
    stats = {"decoded_modules": 0, "trellis_bits": 0,
             "module_bits_policy": observed["module_bits_policy"]}
    if observed["rotation_layout"] == "r7_shared" and module_shared:
        # The r7 experts are stored with their intermediate channels permuted
        # (r7_encoder/permutation.py); the pod undoes it from the layer's own
        # manifest, fetched here into the checkpoint directory the source reads.
        manifest = "r7-experts-layer-%03d.json" % case["layer"]
        fetch.json_file(repo, revision, manifest)
        stats["r7_permutations"] = lo.r7_permutation_source(
            config, str(fetch._path(repo, revision, "")), {"bit_map_manifests": [manifest]})
    hidden, inter = config.hidden_size, config.moe_intermediate_size

    def expected_shape(key: str):
        if key.endswith(".down_proj.weight"):
            return (hidden, inter)
        if key.endswith(".gate_proj.weight") or key.endswith(".up_proj.weight"):
            return (inter, hidden)
        return None

    out = lo.materialize_trellis_subset(subset, plan, torch.bfloat16, stats,
                                        composition=observed["composition"],
                                        expected_shape=expected_shape)
    weight_key = module + ".weight"
    if set(out) != {weight_key}:
        raise _fail("%s: the pod path emitted %s, not %s" % (case["artifact"], sorted(out), weight_key))
    decoded = out[weight_key]
    reference, ref_digest, ref_size, ref_shard, _, ref_shape = fetch.tensor(
        *OFFICIAL, weight_key, rows=window)
    metrics = compare(decoded, reference)
    layout = observed["rotation_layout"]
    module_layout = layout if module_shared else "per_module"
    if module_layout != case["layout"]:
        raise _fail("%s %s resolved as %s, the case expects %s"
                    % (case["artifact"], module, module_layout, case["layout"]))
    bits = sorted(int(k) for k in stats["k_histogram"])
    record = {
        "artifact": case["artifact"], "repo": repo, "revision": revision,
        "module": module, "layout_of_checkpoint": layout, "layout_of_module": module_layout,
        "K": bits, "codebook_histogram": observed["codebook_histogram"],
        "composition": ({"tp": observed["composition"]["tp"], "axes": stats.get("tp_axes")}
                        if observed["composition"] else None),
        "shared_vectors_applied": int(stats.get("shared_vectors_applied", 0)),
        "window_rows": window, "decoded_shape": list(decoded.shape),
        "inputs": inputs,
        "reference": {"repo": OFFICIAL[0], "revision": OFFICIAL[1], "tensor": weight_key,
                      "shard": ref_shard, "shape": ref_shape, "rows_compared": list(reference.shape),
                      "sha256_of_rows": ref_digest, "bytes": ref_size},
        "decoded_bf16_sha256": hashlib.sha256(
            decoded.contiguous().view(torch.int16).numpy().tobytes()).hexdigest(),
        "vs_reference": metrics,
    }
    log("%s %s [%s] K%s %s -> cosine %.6f rel_l2 %.4f"
        % (case["artifact"], module, layout, bits, list(decoded.shape), metrics["cosine"], metrics["rel_l2"]))
    if case.get("control") == "wrong-layer-shared-vector":
        record["control"] = wrong_layer_control(fetch, case, repo, revision, config, weight_map,
                                                subset, keys, plan, observed, expected_shape,
                                                reference, log, stats)
    source = stats.get("r7_permutations")
    if source is not None:
        # The SAME decode without the manifest's inverse permutation -- what the
        # bytes look like as stored, i.e. what a reader that resolves the shared
        # vectors correctly but ignores permutations[E] would load.
        expert = lo._EXL3_EXPERT_RE.match(module)
        entry = source._layer(case["layer"])["permutations"][expert.group("expert")]
        axis = 1 if expert.group("proj") == "down_proj" else 0
        as_stored = decoded.index_select(axis, torch.tensor(entry["new_to_old"], dtype=torch.long))
        record["r7_intermediate_permutation"] = {
            "policy": entry.get("policy"), "manifest": "r7-experts-layer-%03d.json" % case["layer"],
            "manifest_sha256": next(iter(source.stats["manifest_sha256"].values())),
            "reference": lo.R7_UNPERMUTE_REFERENCE,
            "as_stored_without_unpermute_vs_reference": compare(as_stored, reference),
        }
        log("   as stored (permutations[%s] not undone): cosine %.6f"
            % (expert.group("expert"),
               record["r7_intermediate_permutation"]["as_stored_without_unpermute_vs_reference"]["cosine"]))
    return record


def wrong_layer_control(fetch, case, repo, revision, config, weight_map, subset, keys, plan,
                        observed, expected_shape, reference, log, stats) -> Dict[str, Any]:
    """The same payload decoded with the NEXT layer's shared vector of the same
    projection: right shape, right dtype, wrong layer -> cosine ~0."""
    import torch

    layer = case["layer"]
    shared_keys = [k for k in keys if ".shared_h." in k or ".r7_shared." in k]
    if not shared_keys:
        raise _fail("%s: no shared vector in the group of %s" % (case["artifact"], case["module"]))
    swapped = dict(subset)
    replaced = {}
    for key in shared_keys:
        other = key.replace(".layers.%d." % layer, ".layers.%d." % (layer + 1), 1)
        tensor, digest, _, _, _, _ = fetch.tensor(repo, revision, other)
        swapped[key] = tensor
        replaced[key] = {"replaced_by": other, "sha256": digest}
    stats = {"decoded_modules": 0, "trellis_bits": 0,
             "module_bits_policy": observed["module_bits_policy"],
             "r7_permutations": stats.get("r7_permutations")}
    out = lo.materialize_trellis_subset(swapped, dict(plan), torch.bfloat16, stats,
                                        composition=observed["composition"],
                                        expected_shape=expected_shape)
    metrics = compare(out[case["module"] + ".weight"], reference)
    log("   control (layer %d's shared vector): cosine %.6f rel_l2 %.4f"
        % (layer + 1, metrics["cosine"], metrics["rel_l2"]))
    return {"kind": case["control"], "shared_vectors_replaced": replaced, "vs_reference": metrics}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("FIDELITY_SCRATCH", tempfile.gettempdir())) / "exl3layout-cache")
    parser.add_argument("--only", action="append", default=None,
                        help="artifact label(s) to run (default: all)")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print("[exl3-layout-parity] %s" % message, flush=True)

    started = time.monotonic()
    fetch = Fetcher(args.cache_dir)
    official_index = fetch._path(*OFFICIAL, "model.safetensors.index.json")
    fetch.weight_map(*OFFICIAL)
    index_digest = hashlib.sha256(official_index.read_bytes()).hexdigest()
    if index_digest != OFFICIAL_INDEX_SHA256:
        raise _fail("official index sha256 %s != %s" % (index_digest, OFFICIAL_INDEX_SHA256))
    records = []
    for case in CASES:
        if args.only and case["artifact"] not in args.only:
            continue
        records.append(run_case(fetch, case, log))
    receipt = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "engines/tools/exl3_layout_parity.py",
        "decoder": {"module": "engines/tools/exl3hf_surface.py", "function": "decode_payload_hf",
                    "code_sha256": xs._sha256_file(TOOLS / "exl3hf_surface.py"),
                    "mcg_lut_sha256": xs.MCG_LUT_SHA256,
                    "pod_path": "layer_outer.trellis_checkpoint_plan + materialize_trellis_subset "
                                "on the artifact's own config.json and index names"},
        "layout_readers": dict(lo.TRELLIS_LAYOUT_READERS),
        "reference": {"repo": OFFICIAL[0], "revision": OFFICIAL[1],
                      "index_sha256": index_digest,
                      "note": "every artifact's hybrid_tr3_tail names this index as "
                              "source_index_sha256 / bf16_source_index_sha256"},
        "artifacts": {label: {"repo": repo, "revision": revision}
                      for label, (repo, revision) in ARTIFACTS.items()},
        "reading": (
            "A CORRECT decode sits at its K's trellis reconstruction error, halving per bit: "
            "K3 rel_l2 ~0.15, K4 ~0.07-0.09 (fruit-siq-trellis-reconstruction.json 0.0674, "
            "glm53-exl3-tp-rank-and-zero-pad-parity.json 0.067), K6 ~0.02, K8 ~0.006; "
            "in_band_rel_l2 is that ceiling per K. Each shared-layout control decodes the same "
            "bytes with the NEXT layer's shared vector of the same projection and falls to "
            "cosine ~0: the vector's NAME resolution is what the decode rests on, not its "
            "shape. The r7 modules additionally show the bytes AS STORED (the manifest's "
            "permutations[E] not undone) at cosine ~0: the layer manifest's new_to_old is "
            "part of the layout, and the decoder inverts it."),
        "in_band_rel_l2": {str(k): v for k, v in sorted(IN_BAND_REL_L2.items())},
        "modules_compared": len(records),
        "all_in_band": all(r["vs_reference"]["rel_l2"] <= IN_BAND_REL_L2[max(r["K"])]
                           and r["vs_reference"]["cosine"] > 0.98 for r in records),
        "all_controls_near_zero": all(
            abs(r["control"]["vs_reference"]["cosine"]) < 0.05 for r in records if r.get("control"))
        and all(abs(r["r7_intermediate_permutation"]["as_stored_without_unpermute_vs_reference"]["cosine"]) < 0.05
                for r in records if r.get("r7_intermediate_permutation")),
        "bytes_fetched": fetch.bytes_fetched,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "modules": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.out)
    log("wrote %s: modules=%d all_in_band=%s all_controls_near_zero=%s"
        % (args.out, len(records), receipt["all_in_band"], receipt["all_controls_near_zero"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParityError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
