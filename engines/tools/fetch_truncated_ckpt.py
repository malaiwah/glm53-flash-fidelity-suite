#!/usr/bin/env python3
"""Materialise a LAYER-TRUNCATED slice of a sharded HF BF16 checkpoint by ranged fetch.

Why this exists
---------------
`engines/tools/fetch_nonrouted_sparse.py` fetches the NON-routed byte ranges of a
tree, because the streaming lane reads routed experts from a quantized
artifact.  That is the wrong shape for the question this tool answers.

`docs/GLM53-ROOT-FEASIBILITY.md` Stage A needs the opposite: a checkpoint that
is *architecturally complete but short* -- every tensor of layers `0..N-1`
including all 256 routed experts of the first sparse layer, plus
`embed_tokens`, `lm_head` and `model.norm` -- so that `engines/tools/hf_capture.py`
can be driven down its PRODUCTION code path against real published weights on
a desk-sized machine.  For `zai-org/GLM-5.3-BF16` at `--layers 4` that is
25.9 GB instead of 1,506.7 GB, and it exercises the one mechanism that 96.7%
of the real checkpoint depends on: the `WeightConverter` that fuses 256
per-expert matrices into one 3-D parameter.

How
---
  * `config.json` and `model.safetensors.index.json` are fetched verbatim at
    the pinned revision and their sha256 recorded.
  * every shard that carries a wanted tensor becomes a SPARSE local file of
    its full apparent size: the published safetensors header lands verbatim at
    offset 0, each wanted tensor's byte range lands at its exact published
    offset, and every other region stays a hole that nothing reads.  Offsets
    are therefore the published offsets and the bytes are the published bytes,
    which is what makes a byte-for-byte comparison against a loaded model
    meaningful.
  * a PRUNED `model.safetensors.index.json` naming only the wanted tensors is
    written, so `transformers` never asks for a hole.
  * `config.json` is rewritten with `num_hidden_layers = N`.  Nothing else is
    touched -- same `model_type`, same `first_k_dense_replace`, same
    `indexer_types`, same expert layout -- because the point is to change the
    SIZE and nothing else.

  Full-shard sha256 verification is impossible by construction (most of each
  shard was never fetched).  Binding comes from the pinned revision, the
  recorded index/config digests, and the per-tensor digests in the receipt.

Beyond `model.layers.N.`
-----------------------
The first version of this tool hard-coded `^model\\.layers\\.(\\d+)\\.` as the
decoder-layer key, `num_hidden_layers` as the config knob and "top-level list of
strings" as the schedule shape.  All three are GLM-5.3's, and none of them is
universal:

  * `deepseek-ai/DeepSeek-V4-Flash-0731` ships DeepSeek's native names --
    `layers.N.attn.wq_a.weight`, `embed.weight`, `head.weight` -- with no
    `model.` prefix, plus a separate `mtp.N.` subtree the architecture does not
    build.  Under the old regex NOTHING matched, so every layer counted as
    non-layer and the "truncation" would have fetched all 166.9 GB.
  * `Qwen/Qwen3.8-Flash-Next` (`qwen4_exp`) and `MiniMaxAI/MiniMax-M3`
    (`minimax_m3_vl`) are VL models: the decoder is
    `model.language_model.layers.N.` / `language_model.model.layers.N.`, the
    layer count lives in `config.text_config.num_hidden_layers`, and there is a
    vision tower whose OWN `...encoder.layers.N.` keys must NOT be truncated.
  * MiniMax's per-layer schedules (`moe_layer_freq`,
    `sparse_attention_config.sparse_attention_freq`, ...) are lists of INTs,
    nested one level down.  A truncator that only trims top-level string lists
    leaves them at their full length and `transformers` refuses the config.

So the three assumptions are now flags -- `--layer-key-regex`,
`--drop-key-regex`, `--config-node` -- whose DEFAULTS reproduce the GLM-5.3
behaviour exactly (`bin/selftest_fetch_truncated.py` pins that).

Usage:
  fetch_truncated_ckpt.py --repo zai-org/GLM-5.3-BF16 --revision <40-hex> \
      --layers 4 --dest /path/to/ckpt --receipt /path/to/receipt.json \
      [--threads 12] [--token-file ~/.hf_token] [--dry-run]

  fetch_truncated_ckpt.py --repo deepseek-ai/DeepSeek-V4-Flash-0731 \
      --revision <40-hex> --layers 4 \
      --layer-key-regex '^layers\\.(\\d+)\\.' --drop-key-regex '^mtp\\.' ...

  fetch_truncated_ckpt.py --repo MiniMaxAI/MiniMax-M3 --revision <40-hex> \
      --layers 4 --config-node text_config \
      --layer-key-regex '^language_model\\.model\\.layers\\.(\\d+)\\.' ...

The token is read from `--token-file` or `HF_TOKEN`; it is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# GLM-5.3's decoder-layer key.  The DEFAULT of --layer-key-regex, kept as a
# module constant so the old behaviour has a name and a regression test.
DEFAULT_LAYER_KEY = r"^model\.layers\.(\d+)\."


def _fail(message: str, code: int = 1) -> "SystemExit":
    print("fetch_truncated_ckpt: ERROR: %s" % message, file=sys.stderr, flush=True)
    return SystemExit(code)


def log(**fields: Any) -> None:
    print(json.dumps(fields, sort_keys=True), flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

# Standalone copy of bin/fidelity/common.make_no_cross_origin_auth_handler():
# this file is scp'd to remote boxes as a single script, so it cannot import
# the suite package.  Keep the semantics identical: urllib's default redirect
# handler forwards `Authorization` across hosts, and HF `/resolve/` URLs 302
# to pre-signed CDN hosts, so the default handler hands the Hub token to
# whatever host the endpoint names (2026-08-31 peer review, High).


def _origin(url: str):
    """(scheme, host, port) with default ports normalised."""
    import urllib.parse
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return ("", "", None)
    scheme = (parts.scheme or "").lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return (scheme, host, port)


class _NoCrossOriginAuth(urllib.request.HTTPRedirectHandler):
    """Strip `Authorization` when a redirect leaves the original origin
    (host change, https->http downgrade, or port change).  Redirect loops
    stay bounded by urllib's own max_redirections."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl)
        if new is not None and _origin(newurl) != _origin(req.full_url):
            new.headers = {k: v for k, v in new.headers.items()
                           if k.lower() != "authorization"}
            new.unredirected_hdrs = {
                k: v for k, v in getattr(new, "unredirected_hdrs", {}).items()
                if k.lower() != "authorization"}
        return new


_OPENER = urllib.request.build_opener(_NoCrossOriginAuth())


class Fetcher:
    def __init__(self, repo: str, revision: str, token: Optional[str]) -> None:
        self.base = "https://huggingface.co/%s/resolve/%s/" % (repo, revision)
        self.token = token

    def _request(self, name: str, byte_range: Optional[Tuple[int, int]],
                 attempts: int = 6) -> bytes:
        url = self.base + name
        last: Optional[Exception] = None
        for attempt in range(attempts):
            req = urllib.request.Request(url)
            if self.token:
                req.add_header("Authorization", "Bearer " + self.token)
            if byte_range is not None:
                req.add_header("Range", "bytes=%d-%d" % byte_range)
            try:
                with _OPENER.open(req, timeout=300) as handle:
                    data = handle.read()
                if byte_range is not None:
                    want = byte_range[1] - byte_range[0] + 1
                    if len(data) != want:
                        raise IOError("short read: %d != %d" % (len(data), want))
                return data
            except Exception as exc:  # network flake; retry with backoff
                last = exc
                if attempt + 1 == attempts:
                    break
                time.sleep(min(30.0, 1.5 * (2 ** attempt)))
        raise _fail("GET %s %s failed after %d attempts: %s"
                    % (name, byte_range, attempts, last))

    def whole(self, name: str) -> bytes:
        return self._request(name, None)

    def ranged(self, name: str, start: int, stop_inclusive: int) -> bytes:
        return self._request(name, (start, stop_inclusive))

    def header(self, name: str) -> Tuple[int, Dict[str, Any]]:
        raw = self.ranged(name, 0, 7)
        length = struct.unpack("<Q", raw)[0]
        body = self.ranged(name, 8, 8 + length - 1)
        return length, json.loads(body)


# ---------------------------------------------------------------------------
# range planning
# ---------------------------------------------------------------------------


def coalesce(ranges: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Merge half-open [start, stop) ranges separated by less than `gap` bytes.

    Merging across a small hole costs a few wasted bytes and saves a whole
    round trip; the wasted bytes land in regions nothing ever reads.
    """
    if not ranges:
        return []
    ordered = sorted(ranges)
    out = [list(ordered[0])]
    for start, stop in ordered[1:]:
        if start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], stop)
        else:
            out.append([start, stop])
    return [(a, b) for a, b in out]


def split(ranges: List[Tuple[int, int]], chunk: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for start, stop in ranges:
        cursor = start
        while cursor < stop:
            end = min(stop, cursor + chunk)
            out.append((cursor, end))
            cursor = end
    return out


# ---------------------------------------------------------------------------
# config surgery
# ---------------------------------------------------------------------------


def config_node(config: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Resolve a dotted path to the sub-dict that owns `num_hidden_layers`.

    "" -> the config itself (GLM-5.3, DeepSeek-V4). "text_config" -> the text
    tower of a VL config (Qwen3.8-Flash-Next, MiniMax-M3), whose vision tower
    has its OWN, different, layer count that must be left alone.
    """
    node = config
    if not path:
        return node
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise _fail("--config-node %r: no %r in the config" % (path, part))
        node = node[part]
    if not isinstance(node, dict):
        raise _fail("--config-node %r does not name an object" % path)
    return node


def filter_index_lists(node: Dict[str, Any], specs: Sequence[str],
                       layers: int) -> Dict[str, List[Any]]:
    """Drop out-of-range entries from lists that name LAYER INDICES.

    A per-layer *schedule* has one entry per layer and is handled by
    `truncate_schedules`. A layer-INDEX list is a different animal: it is short,
    and its entries point AT layers. `Qwen/Qwen3.8-Flash-Next` carries
    `text_config.ple_layer_ids = [2]` -- one-indexed -- and `Qwen4ExpTextConfig`
    validates it:

        ValueError: ple_layer_ids must contain one-indexed ids in [1, 1], got [2].

    so a truncation that leaves it alone does not load at all. Each spec is
    `name[:base]` relative to `--config-node`, `base` being 0 or 1 (default 0).
    """
    trimmed: Dict[str, List[Any]] = {}
    for spec in specs:
        name, _, base_text = spec.partition(":")
        base = int(base_text) if base_text else 0
        if base not in (0, 1):
            raise _fail("--config-index-list %r: base must be 0 or 1" % spec)
        target = node
        parts = name.split(".")
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise _fail("--config-index-list %r: no %r under the config node"
                            % (spec, part))
            target = target[part]
        leaf = parts[-1]
        if not isinstance(target, dict) or leaf not in target:
            raise _fail("--config-index-list %r: no %r under the config node" % (spec, leaf))
        value = target[leaf]
        if not isinstance(value, list):
            raise _fail("--config-index-list %r does not name a list" % spec)
        kept = [v for v in value if isinstance(v, int) and base <= v < layers + base]
        target[leaf] = kept
        trimmed[name] = kept
    return trimmed


def truncate_schedules(node: Any, source_layers: int, layers: int,
                       prefix: str = "") -> Dict[str, List[Any]]:
    """Trim every per-layer schedule list under `node` to `layers` entries.

    A "schedule" is any list whose length equals the SOURCE layer count and
    whose entries are all strings or all ints -- `mlp_layer_types`,
    `indexer_types`, `layer_types`, `moe_layer_freq`,
    `sparse_attention_config.sparse_attention_freq`, ... The recursion is what
    reaches MiniMax-M3's, which sit one dict deeper than the layer count.

    The length test is the whole discriminator, so a config that happened to
    carry an unrelated list of exactly `num_hidden_layers` entries would be
    trimmed too.  That is why every trimmed key is RETURNED and logged: the
    operator sees the list of what was cut, rather than trusting a heuristic
    silently.  Lists of a different length (DeepSeek-V4's 46-entry
    `compress_ratios` against 43 layers -- 43 layers plus 3 MTP modules) are
    left alone, which is correct: `DeepseekV4Config` slices them itself.
    """
    trimmed: Dict[str, List[Any]] = {}
    if not isinstance(node, dict):
        return trimmed
    for key, value in list(node.items()):
        where = prefix + key
        if isinstance(value, dict):
            trimmed.update(truncate_schedules(value, source_layers, layers, where + "."))
        elif isinstance(value, list) and len(value) == source_layers and value and (
                all(isinstance(v, str) for v in value)
                or all(isinstance(v, bool) is False and isinstance(v, int) for v in value)):
            node[key] = value[:layers]
            trimmed[where] = node[key]
    return trimmed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_truncated_ckpt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--layers", type=int, required=True,
                    help="keep model.layers.0 .. model.layers.N-1")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--coalesce-gap-bytes", type=int, default=8 << 20)
    ap.add_argument("--range-chunk-bytes", type=int, default=256 << 20)
    ap.add_argument("--token-file", type=Path, default=None)
    ap.add_argument("--layer-key-regex", default=DEFAULT_LAYER_KEY,
                    help="regex whose group(1) is the DECODER layer index. Keys that do "
                         "not match are kept unconditionally (embeddings, head, final "
                         "norm, vision tower). Default is GLM-5.3's %(default)r.")
    ap.add_argument("--drop-key-regex", action="append", default=[],
                    help="repeatable: checkpoint keys matching this are excluded "
                         "entirely, whatever the layer regex says. Use it for subtrees "
                         "the architecture does not build (e.g. '^mtp\\.').")
    ap.add_argument("--config-index-list", action="append", default=[],
                    metavar="NAME[:BASE]",
                    help="repeatable: a config list (relative to --config-node) whose "
                         "entries are LAYER INDICES rather than per-layer values; entries "
                         "outside the kept range are dropped. BASE is 0 or 1 (default 0). "
                         "e.g. 'ple_layer_ids:1' for Qwen3.8-Flash-Next.")
    ap.add_argument("--config-node", default="",
                    help="dotted path to the config object that owns num_hidden_layers "
                         "and the per-layer schedules ('' = the config itself; "
                         "'text_config' for a VL config).")
    ap.add_argument("--extra-file", action="append", default=[],
                    help="additional repo file to fetch verbatim (tokenizer.json, ...)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and print the byte budget; fetch nothing")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise _fail("--revision must be a 40-hex commit sha, not a branch name: "
                    "a truncated checkpoint that cannot name its source is not evidence")
    if args.layers < 1:
        raise _fail("--layers must be >= 1")

    token = None
    if args.token_file is not None:
        token = args.token_file.expanduser().read_text().strip()
    elif os.environ.get("HF_TOKEN"):
        token = os.environ["HF_TOKEN"].strip()

    fetcher = Fetcher(args.repo, args.revision, token)
    args.dest.mkdir(parents=True, exist_ok=True)

    config_raw = fetcher.whole("config.json")
    index_raw = fetcher.whole("model.safetensors.index.json")
    config = json.loads(config_raw)
    index = json.loads(index_raw)
    weight_map: Dict[str, str] = index["weight_map"]
    layer_node = config_node(config, args.config_node)
    if "num_hidden_layers" not in layer_node:
        raise _fail("no num_hidden_layers under --config-node %r" % args.config_node)
    source_layers = int(layer_node["num_hidden_layers"])
    if args.layers > source_layers:
        raise _fail("--layers %d exceeds the checkpoint's %d" % (args.layers, source_layers))
    layer_key = re.compile(args.layer_key_regex)
    drops = [re.compile(pattern) for pattern in args.drop_key_regex]
    log(stage="source", repo=args.repo, revision=args.revision,
        tensors=len(weight_map), total_size=index.get("metadata", {}).get("total_size"),
        num_hidden_layers=source_layers, config_node=args.config_node,
        layer_key_regex=args.layer_key_regex, drop_key_regex=list(args.drop_key_regex),
        config_sha256=sha256_bytes(config_raw), index_sha256=sha256_bytes(index_raw))

    wanted: Dict[str, str] = {}
    matched_layer_keys = 0
    dropped = 0
    for key, shard in weight_map.items():
        if any(pattern.search(key) for pattern in drops):
            dropped += 1
            continue
        match = layer_key.match(key)
        if match is not None:
            matched_layer_keys += 1
            if int(match.group(1)) < args.layers:
                wanted[key] = shard
        else:
            wanted[key] = shard          # embed_tokens / norm / lm_head / vision tower
    if matched_layer_keys == 0:
        # Silence here is the dangerous answer: with no match every key looks
        # like a non-layer key, so the "truncation" quietly becomes the whole
        # checkpoint. That is a 166.9 GB accident on DeepSeek-V4-Flash.
        raise _fail(
            "--layer-key-regex %r matched NONE of this checkpoint's %d tensor names, so "
            "nothing would be truncated and the fetch would be the WHOLE model. Sample "
            "names: %s" % (args.layer_key_regex, len(weight_map),
                           ", ".join(sorted(weight_map)[:4])))
    shards = sorted(set(wanted.values()))
    log(stage="plan", kept_tensors=len(wanted), shards=len(shards), layers=args.layers,
        layer_keys_seen=matched_layer_keys, dropped_by_regex=dropped)

    # Published headers fix every offset.  Fetch them first; they are tiny.
    headers: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(args.threads, len(shards))) as pool:
        futures = {pool.submit(fetcher.header, s): s for s in shards}
        for future in as_completed(futures):
            headers[futures[future]] = future.result()

    plan: Dict[str, Dict[str, Any]] = {}
    fetch_bytes = 0
    apparent_bytes = 0
    for shard in shards:
        length, header = headers[shard]
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        data_start = 8 + length
        apparent = data_start + max(v["data_offsets"][1] for v in entries.values())
        apparent_bytes += apparent
        raw_ranges = []
        for key, entry in entries.items():
            if key not in wanted:
                continue
            begin, end = entry["data_offsets"]
            raw_ranges.append((data_start + begin, data_start + end))
        merged = coalesce(raw_ranges, args.coalesce_gap_bytes)
        chunks = split(merged, args.range_chunk_bytes)
        fetch_bytes += sum(b - a for a, b in chunks)
        plan[shard] = {"header_len": length, "header": header, "apparent": apparent,
                       "data_start": data_start, "chunks": chunks,
                       "tensors": sorted(k for k in entries if k in wanted)}
        log(stage="shard_plan", shard=shard, apparent_bytes=apparent,
            kept_tensors=len(plan[shard]["tensors"]),
            fetch_bytes=sum(b - a for a, b in chunks), requests=len(chunks))

    extras_bytes = 0
    log(stage="budget", fetch_bytes=fetch_bytes,
        fetch_gb=round(fetch_bytes / 1e9, 3),
        whole_shard_bytes=apparent_bytes, whole_shard_gb=round(apparent_bytes / 1e9, 3),
        saved_gb=round((apparent_bytes - fetch_bytes) / 1e9, 3))
    if args.dry_run:
        return 0

    started = time.monotonic()
    done_bytes = [0]
    for shard in shards:
        entry = plan[shard]
        path = args.dest / shard
        with open(path, "wb") as handle:
            handle.truncate(entry["apparent"])          # sparse: holes, not zeros on disk

    # The header must be the PUBLISHED bytes, not a re-serialisation: a
    # re-serialised header would change offsets and silently invalidate every
    # range we are about to write.  Fetch it verbatim.
    def write_header(shard: str) -> int:
        entry = plan[shard]
        raw = fetcher.ranged(shard, 0, entry["data_start"] - 1)
        with open(args.dest / shard, "r+b") as handle:
            handle.seek(0)
            handle.write(raw)
        return len(raw)

    def write_chunk(shard: str, start: int, stop: int) -> int:
        raw = fetcher.ranged(shard, start, stop - 1)
        with open(args.dest / shard, "r+b") as handle:
            handle.seek(start)
            handle.write(raw)
        done_bytes[0] += len(raw)
        return len(raw)

    jobs: List[Tuple[str, int, int]] = []
    for shard in shards:
        for start, stop in plan[shard]["chunks"]:
            jobs.append((shard, start, stop))

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        header_futures = [pool.submit(write_header, s) for s in shards]
        for future in as_completed(header_futures):
            future.result()
        futures = {pool.submit(write_chunk, *job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                elapsed = max(1e-6, time.monotonic() - started)
                log(stage="progress", chunks_done=completed, chunks=len(futures),
                    bytes=done_bytes[0], mb_per_s=round(done_bytes[0] / elapsed / 1e6, 1))

    for name in args.extra_file:
        raw = fetcher.whole(name)
        (args.dest / name).write_bytes(raw)
        extras_bytes += len(raw)
        log(stage="extra", file=name, bytes=len(raw), sha256=sha256_bytes(raw))

    pruned_index = {
        "metadata": {"total_size": sum(
            v["data_offsets"][1] - v["data_offsets"][0]
            for shard in shards
            for k, v in plan[shard]["header"].items()
            if k != "__metadata__" and k in wanted)},
        "weight_map": {k: v for k, v in sorted(wanted.items())},
    }
    (args.dest / "model.safetensors.index.json").write_text(
        json.dumps(pruned_index, indent=2, sort_keys=True))

    # Per-layer schedule lists must be truncated with the layer count.
    # `transformers` 5.16.1 validates this and REFUSES the config otherwise:
    #   ValueError: `num_hidden_layers` (4) must be equal to the number of
    #               `mlp_layer_types` (78)
    # so a "change only num_hidden_layers" truncation does not load at all.
    # Truncation preserves the schedule of the layers we keep, which is the
    # property that matters: for GLM-5.3 that is dense/dense/dense/sparse and
    # indexer full/full/full/shared -- so layer 3 still consumes layer 2's
    # `prev_topk_indices` exactly as it does in the 78-layer model.
    truncated_config = json.loads(json.dumps(config))     # deep copy, not dict()
    target = config_node(truncated_config, args.config_node)
    target["num_hidden_layers"] = args.layers
    trimmed = truncate_schedules(target, source_layers, args.layers)
    index_lists = filter_index_lists(target, args.config_index_list, args.layers)
    (args.dest / "config.json").write_text(json.dumps(truncated_config, indent=2))
    log(stage="config", num_hidden_layers=args.layers, config_node=args.config_node,
        truncated_lists=trimmed, filtered_index_lists=index_lists)

    # Per-tensor digests, read back from the file we just wrote.  These are the
    # preimage of the byte-for-byte comparison Stage A performs against the
    # loaded model, so they are recorded rather than recomputed later.
    digests: Dict[str, Dict[str, Any]] = {}
    for shard in shards:
        entry = plan[shard]
        with open(args.dest / shard, "rb") as handle:
            for key in entry["tensors"]:
                meta = entry["header"][key]
                begin, end = meta["data_offsets"]
                handle.seek(entry["data_start"] + begin)
                raw = handle.read(end - begin)
                digests[key] = {"shard": shard, "dtype": meta["dtype"],
                                "shape": meta["shape"], "bytes": len(raw),
                                "offset": entry["data_start"] + begin,
                                "sha256": sha256_bytes(raw)}

    receipt = {
        "schema": "malaiwah.truncated-checkpoint-receipt.v1",
        "tool": "engines/tools/fetch_truncated_ckpt.py",
        "repo": args.repo, "revision": args.revision,
        "source_num_hidden_layers": source_layers,
        "kept_num_hidden_layers": args.layers,
        "config_node": args.config_node,
        "layer_key_regex": args.layer_key_regex,
        "drop_key_regex": list(args.drop_key_regex),
        "dropped_by_regex": dropped,
        "truncated_schedule_lists": sorted(trimmed),
        "filtered_layer_index_lists": index_lists,
        "source_tensors": len(weight_map),
        "source_total_size": index.get("metadata", {}).get("total_size"),
        "kept_tensors": len(wanted),
        "kept_total_size": pruned_index["metadata"]["total_size"],
        "shards": shards,
        "source_config_sha256": sha256_bytes(config_raw),
        "source_index_sha256": sha256_bytes(index_raw),
        "fetched_bytes": fetch_bytes + extras_bytes,
        "whole_shard_bytes_avoided": apparent_bytes - fetch_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "full_shard_sha256_verifiable": False,
        "full_shard_sha256_note":
            "each local shard is SPARSE -- only the byte ranges of the kept tensors were "
            "fetched, at their published offsets, under the published header. A full-file "
            "sha256 therefore cannot equal the published shard's. Binding is the pinned "
            "revision plus source_index_sha256 plus the per-tensor digests below.",
        "tensor_digests": digests,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    log(stage="done", dest=str(args.dest), receipt=str(args.receipt),
        kept_tensors=len(wanted), fetched_bytes=fetch_bytes + extras_bytes,
        elapsed_seconds=round(time.monotonic() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
