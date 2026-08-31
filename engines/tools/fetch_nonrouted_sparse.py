#!/usr/bin/env python3
"""Ranged fetch of the NON-ROUTED tensors of a sharded BF16 tree into sparse shards.

The GLM-5.3-Flash BF16 release interleaves 1,618 non-routed tensors (~19.34 GB)
with 623 GB of routed experts across 47 of its 120 shards.  The streaming
scorer only ever READS the non-routed byte ranges (prepare_nonrouted_view
filters the index; safetensors loads by offset), so this tool materializes
exactly those ranges:

  * config.json + model.safetensors.index.json are fetched VERBATIM at the
    pinned revision and verified against the sealed release inventory's
    config_sha256 / index_sha256 (the same binding stream_score enforces).
  * each shard referenced by a non-routed tensor becomes a SPARSE local file
    of its full apparent size: the safetensors header bytes land at offset 0,
    every non-routed tensor's byte range lands at its exact recorded offset,
    and the routed regions stay holes that nothing ever reads.
  * full-shard sha256 verification is IMPOSSIBLE by construction (the routed
    bytes are not fetched) -- this is a disclosed property of the tree, and
    the receipt says so.  Binding comes from the inventory-verified index
    (which fixes every offset) plus downstream gates: the loader's
    missing/mismatched/error report and the sealed panel-number reproduction.
  * the lm_head.weight tensor is additionally extracted to its own
    safetensors file and, when a published head extraction is given,
    verified against it by FILE sha256 and by tensor CONTENT equality.

Usage:
  fetch_nonrouted_sparse.py --repo zai-org/GLM-5.3-Flash-BF16 \
      --revision <40-hex> --inventory <sealed inventory.json> \
      --dest /home/models/bf16 --head-out /home/models/head/lm_head.safetensors \
      [--published-head-url <hf resolve url>] [--published-head-sha <64-hex>] \
      [--threads 12] --receipt <path>

HF_TOKEN is read from the environment when present (the zai repo is public;
the token is never printed).
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

EXPERT_CHECKPOINT_KEY = re.compile(r"\.mlp\.experts\.\d+\.")  # == stream_score.EXPERT_CHECKPOINT_KEY


def _fail(message: str, code: int = 1) -> "SystemExit":
    print(f"fetch_nonrouted_sparse: ERROR: {message}", file=sys.stderr, flush=True)
    return SystemExit(code)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _request(url: str, range_header: str = None, attempts: int = 6):
    last = None
    for attempt in range(attempts):
        try:
            headers = {"User-Agent": "glm53-fidelity-suite/fetch_nonrouted_sparse"}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            if range_header:
                headers["Range"] = range_header
            request = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(request, timeout=180)
        except Exception as error:  # noqa: BLE001 - retried, then surfaced
            last = error
            time.sleep(min(2 ** attempt, 30))
    raise _fail(f"GET {url} [{range_header}] failed after {attempts} attempts: {last}")


def fetch_bytes(url: str, start: int = None, stop: int = None) -> bytes:
    range_header = None
    if start is not None:
        range_header = f"bytes={start}-{stop - 1}"
    with _request(url, range_header) as response:
        data = response.read()
    if start is not None and len(data) != stop - start:
        raise _fail(f"ranged GET returned {len(data)} bytes, wanted {stop - start} ({url})")
    return data


def coalesce(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge only STRICTLY adjacent ranges -- a gap is routed bytes we must not fetch."""
    merged: List[Tuple[int, int]] = []
    for begin, end in sorted(ranges):
        if merged and merged[-1][1] == begin:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--head-out", type=Path)
    parser.add_argument("--head-tensor", default="lm_head.weight")
    parser.add_argument("--published-head-url")
    parser.add_argument("--published-head-sha")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--range-chunk-bytes", type=int, default=256 << 20,
                        help="split giant coalesced ranges for parallelism/retry granularity")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise _fail("--revision must be an immutable 40-hex commit")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("model_revision") != args.revision:
        raise _fail(f"inventory pins model_revision {inventory.get('model_revision')}, "
                    f"but --revision is {args.revision}")

    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    args.dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # ---- config + index, verified against the sealed inventory -----------
    aux_records = {}
    for name, expected_key in (("config.json", "config_sha256"),
                               ("model.safetensors.index.json", "index_sha256")):
        dest = args.dest / name
        if not dest.is_file():
            dest.write_bytes(fetch_bytes(base + name))
        digest = sha256_file(dest)
        expected = inventory.get(expected_key)
        if digest != expected:
            raise _fail(f"{name} sha256 {digest} != inventory {expected_key} {expected}")
        aux_records[name] = {"sha256": digest, "verified_against_inventory": True}
        print(f"{name}: sha256 verified against the sealed inventory", flush=True)

    index = json.loads((args.dest / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index["weight_map"]
    keep = {name: shard for name, shard in weight_map.items()
            if EXPERT_CHECKPOINT_KEY.search(name) is None}
    shards = sorted(set(keep.values()))
    print(f"non-routed tensors: {len(keep)} across {len(shards)} of "
          f"{len(set(weight_map.values()))} shards", flush=True)

    # ---- per-shard sparse fetch ------------------------------------------
    shard_records: List[Dict[str, Any]] = []
    total_fetched = 0
    tasks = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for shard in shards:
            url = base + shard
            head8 = fetch_bytes(url, 0, 8)
            header_len = struct.unpack("<Q", head8)[0]
            header_raw = fetch_bytes(url, 8, 8 + header_len)
            header = json.loads(header_raw)
            data_base = 8 + header_len
            wanted = [name for name in header if name != "__metadata__" and name in keep]
            all_end = max(entry["data_offsets"][1] for key, entry in header.items()
                          if key != "__metadata__")
            shard_size = data_base + all_end
            ranges = coalesce([(data_base + header[name]["data_offsets"][0],
                                data_base + header[name]["data_offsets"][1]) for name in wanted])
            # split giant runs so a mid-transfer failure retries a chunk, not 4 GB
            split: List[Tuple[int, int]] = []
            for begin, end in ranges:
                while end - begin > args.range_chunk_bytes:
                    split.append((begin, begin + args.range_chunk_bytes))
                    begin += args.range_chunk_bytes
                split.append((begin, end))
            path = args.dest / shard
            if not path.is_file() or path.stat().st_size != shard_size:
                with open(path, "wb") as handle:
                    handle.truncate(shard_size)
            with open(path, "r+b") as handle:
                handle.seek(0)
                handle.write(head8 + header_raw)
            fetched = sum(end - begin for begin, end in split)
            total_fetched += fetched + data_base
            shard_records.append({
                "shard": shard,
                "apparent_bytes": shard_size,
                "nonrouted_tensors": len(wanted),
                "ranges": len(split),
                "fetched_bytes": fetched + data_base,
                "sparse": True,
                "full_shard_sha256_verifiable": False,
            })
            done_marker = args.dest / (shard + ".ranges-done")
            if done_marker.is_file():
                continue
            for begin, end in split:
                tasks.append((pool.submit(fetch_bytes, url, begin, end), shard, begin, end))
        print(f"fetching {len(tasks)} ranges across {len(shards)} shards "
              f"({total_fetched / 1e9:.2f} GB incl. headers)", flush=True)
        by_shard_fds: Dict[str, Any] = {}
        completed = 0
        try:
            for future, shard, begin, end in [(t[0], t[1], t[2], t[3]) for t in tasks]:
                data = future.result()
                fd = by_shard_fds.get(shard)
                if fd is None:
                    fd = os.open(args.dest / shard, os.O_WRONLY)
                    by_shard_fds[shard] = fd
                os.pwrite(fd, data, begin)
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"  ranges {completed}/{len(tasks)}", flush=True)
        finally:
            for fd in by_shard_fds.values():
                os.close(fd)
    for shard in shards:
        (args.dest / (shard + ".ranges-done")).write_text("done\n")

    elapsed = time.monotonic() - started

    # ---- head extraction + published-extraction verification -------------
    head_record: Dict[str, Any] = {}
    if args.head_out:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        head_shard = keep.get(args.head_tensor)
        if head_shard is None:
            raise _fail(f"{args.head_tensor} is not a non-routed tensor of this index")
        with safe_open(args.dest / head_shard, framework="pt", device="cpu") as handle:
            head = handle.get_tensor(args.head_tensor)
        if head.dtype != torch.bfloat16:
            raise _fail(f"{args.head_tensor} dtype {head.dtype} != bfloat16")
        args.head_out.parent.mkdir(parents=True, exist_ok=True)
        save_file({args.head_tensor: head.contiguous()}, args.head_out,
                  metadata={"source_repo": args.repo, "source_revision": args.revision,
                            "source_shard": head_shard})
        content_sha = sha256_bytes(head.contiguous().view(torch.uint16).numpy().tobytes())
        head_record = {
            "tensor": args.head_tensor,
            "source_shard": head_shard,
            "shape": list(head.shape),
            "dtype": "bfloat16",
            "out": str(args.head_out.resolve()),
            "out_file_sha256": sha256_file(args.head_out),
            "tensor_content_sha256": content_sha,
        }
        if args.published_head_url:
            published_path = args.head_out.with_name("published-head.safetensors")
            if not published_path.is_file():
                published_path.write_bytes(fetch_bytes(args.published_head_url))
            published_sha = sha256_file(published_path)
            head_record["published_head_url"] = args.published_head_url
            head_record["published_head_file_sha256"] = published_sha
            if args.published_head_sha:
                if published_sha != args.published_head_sha:
                    raise _fail(f"published head file sha {published_sha} != expected "
                                f"{args.published_head_sha}")
                head_record["published_head_sha_matches_receipt"] = True
            with safe_open(published_path, framework="pt", device="cpu") as handle:
                published_keys = list(handle.keys())
                published = handle.get_tensor(published_keys[0])
            equal = bool(torch.equal(published, head))
            head_record["published_head_tensor_name"] = published_keys[0]
            head_record["published_head_content_equals_live_fetch"] = equal
            if not equal:
                raise _fail("published head extraction and live-fetched lm_head.weight DIFFER "
                            "in content -- refusing to continue with an unpinned head")
            print("lm_head content equality vs published extraction: VERIFIED", flush=True)

    receipt = {
        "schema": "malaiwah.glm53-nonrouted-sparse-fetch.v1",
        "repo": args.repo,
        "revision": args.revision,
        "inventory_sha256": inventory.get("inventory_sha256"),
        "aux": aux_records,
        "nonrouted_tensor_count": len(keep),
        "shards": shard_records,
        "shards_referenced": len(shards),
        "total_fetched_bytes": total_fetched,
        "sparse_disclosure": (
            "shards are sparse: only the safetensors header and the non-routed tensor "
            "byte ranges are materialized; full-shard sha256 cannot be verified on this "
            "tree by construction. Binding: config/index verified against the sealed "
            "full-shard-sha256 inventory; the loader's exact-load report and the sealed "
            "panel-number reproduction gate the content downstream."
        ),
        "head": head_record,
        "elapsed_seconds": round(elapsed, 1),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(args.receipt), "fetched_gb": round(total_fetched / 1e9, 2),
                      "elapsed_s": round(elapsed, 1)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
