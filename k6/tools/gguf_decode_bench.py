#!/usr/bin/env python3
"""Measure the GGUF lane's fill rate, and prove the fast path changes nothing.

Why this exists rather than a full capture.  `docs/GGUF-MEASUREMENT.md` prices a
GGUF row at 23.7 min/window -> 19.7 h -> ~$32, and the bottleneck was isolated
to the DEQUANT (39.12 ms/matrix, GPU at 2-4%, 1,300% CPU on a 128-core host that
was 74% idle).  Fixing that needs a before/after number, and a full capture to
get one costs the $32 the fix exists to avoid.  So this reproduces the fill
loop -- the part that takes the time -- against the artifact's REAL bytes, and
nothing else.

What it reproduces, from `stream_score.ExpertStreamer._fill_range`, verbatim:

    ThreadPoolExecutor(decode_threads)
      -> GgufExpertSource.load       (os.pread of one expert slice + dequant)
      -> payload.to(device)          (a no-op when the decode already landed)
      -> fuse_gate_up = cat((gate, up), 0)
      -> ONE fp32 -> bf16 rounding
      -> copy_ into the resident slab
      -> torch.equal close check

Two things it deliberately does NOT do: run the forward (the forward is
unchanged and is not what costs 23.7 minutes), and load the non-routed view.
The quantity reported is milliseconds per expert MATRIX and seconds per layer
FILL, which is exactly the quantity `k6/tools/gguf-evidence/
udq4kxl-decode-timings-a100.jsonl` recorded and the quantity
`engines.json:minutes_per_window_by_surface.gguf` is derived from
(36,288 matrices per window).

THE ACCEPTANCE TEST is `--verify`.  It runs the whole fill twice -- once with
the dequant on CPU, once on the accelerator -- and demands the two resulting
BF16 SLABS be `torch.equal`.  Not the decoded fp32; the slab, i.e. the exact
bytes the model's expert forward reads.  If every installed weight is
bit-identical then the forward is the same function on the same lane and the
tokenwise KLD tensor it produces is the same tensor: the fast path cannot move
a published number.  That is the property the whole GGUF surface rests on, and
it is checked here on real UD-Q4_K_XL expert tensors rather than a fixture.

A single part of a split GGUF is enough and is what this takes: 33 GB rather
than 200 GB, holding whole fused expert tensors for several layers.
`load_gguf_surface` correctly refuses an incomplete split, so this drives
`GgufFile` directly -- it is a benchmark, not a measurement, and it seals
nothing.

Usage:
    python3 gguf_decode_bench.py --gguf <part.gguf> [--experts 288]
                                 [--decode-threads 16] [--device cuda]
                                 [--modes cpu,device] [--verify] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gguf_surface as gs  # noqa: E402
import progress as progress_meter  # noqa: E402

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def fused_rows(gguf: gs.GgufFile, experts: int) -> Dict[int, Dict[str, Any]]:
    """{layer: {projection: row}} for layers whose first ``experts`` slices are HERE.

    Present in the TABLE is not present on disk.  A GGUF header lists every
    tensor in the whole split, and this benchmark is deliberately run against a
    truncated download -- the header plus the first few GB, rather than the
    33 GB part or the 200 GB build -- so a row can name a tensor whose data
    lands past EOF.  Reading it would fail as a short read three minutes into a
    rented hour.

    The bound checked is the exact one the fill will read: the END of expert
    ``experts - 1``, not the end of the fused tensor.  `--experts 288` therefore
    demands the whole 1.4 GB tensor and a smaller run demands only its prefix,
    which is what makes a partial download usable at all.
    """
    found: Dict[int, Dict[str, Any]] = {}
    data_start = int(gguf.info["data_start"])
    for name, row in gguf.tensors.items():
        kind = gs.classify_tensor(name)
        if kind[0] != "routed":
            continue
        try:
            rel, nbytes = gs.expert_slice_range(row, experts - 1)
        except Exception:  # noqa: BLE001 - not a fused 288-expert tensor
            continue
        if data_start + int(row["offset"]) + rel + nbytes > gguf.size:
            continue
        layer, projection = kind[1], kind[2]
        found.setdefault(layer, {})[projection] = row
    return {layer: projections for layer, projections in sorted(found.items())
            if len(projections) == len(PROJECTIONS)}


def fill_once(gguf: gs.GgufFile, rows: Dict[str, Dict[str, Any]], *, experts: int,
              decode_threads: int, device, decode_device, torch,
              meter: Optional[progress_meter.Progress] = None):
    """One layer fill, shaped exactly like ExpertStreamer._fill_range.

    Returns (gate_up slab, down slab, seconds, bytes read).  The slabs are
    returned so --verify can compare them; a real fill keeps them resident.
    """
    out_gate, in_gate = gs.PROJECTION_SHAPE["gate_proj"]
    out_down, in_down = gs.PROJECTION_SHAPE["down_proj"]
    gate_up = torch.empty(experts, out_gate * 2, in_gate, dtype=torch.bfloat16, device=device)
    down = torch.empty(experts, out_down, in_down, dtype=torch.bfloat16, device=device)
    read_bytes = 0
    lock = threading.Lock()

    def load(expert: int, projection: str):
        nonlocal read_bytes
        row = rows[projection]
        rel, nbytes = gs.expert_slice_range(row, expert)
        out_features, in_features = gs.PROJECTION_SHAPE[projection]
        raw = gguf.read_tensor_range(row["name"], rel, nbytes)
        with lock:
            read_bytes += nbytes
        flat = gs.dequant_bytes(row["type"], raw, out_features * in_features,
                                device=decode_device)
        return flat.reshape(out_features, in_features)

    jobs: "queue.Queue[Any]" = queue.Queue(maxsize=max(2, decode_threads * 2))
    error: List[BaseException] = []

    def producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=decode_threads) as pool:
                pending: List[Any] = []
                for expert in range(experts):
                    pending.append((expert, [pool.submit(load, expert, p)
                                             for p in PROJECTIONS]))
                    while len(pending) > decode_threads:
                        jobs.put(pending.pop(0))
                for item in pending:
                    jobs.put(item)
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            jobs.put(None)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    with torch.inference_mode():
        while True:
            item = jobs.get()
            if item is None:
                break
            expert, futures = item
            decoded = [f.result().to(device) for f in futures]
            gate_up_bf16 = torch.cat((decoded[0], decoded[1]), 0).to(dtype=torch.bfloat16)
            down_bf16 = decoded[2].to(dtype=torch.bfloat16)
            gate_up[expert].copy_(gate_up_bf16)
            down[expert].copy_(down_bf16)
            if not torch.equal(gate_up[expert], gate_up_bf16) or not torch.equal(
                    down[expert], down_bf16):
                raise RuntimeError("BF16 streamed expert installation did not close exactly")
            if meter is not None:
                meter.update(3)
            del decoded, gate_up_bf16, down_bf16
    thread.join()
    if error:
        raise error[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return gate_up, down, time.monotonic() - started, read_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="ONE part of the build (33 GB is enough)")
    ap.add_argument("--layer", type=int, action="append",
                    help="layer to fill (repeatable; default: the first present)")
    ap.add_argument("--experts", type=int, default=gs.NUM_EXPERTS,
                    help="experts per fill (default 288 = a real fill)")
    ap.add_argument("--decode-threads", type=int, default=min(16, (os.cpu_count() or 8)),
                    help="matches stream_score's own default")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch.set_num_threads for the CPU path (0 = leave torch's "
                         "default). On a 128-core host torch's default intra-op pool "
                         "is 128 wide, so `--decode-threads 16` asks for 2,048 OMP "
                         "threads on 128 cores -- which is a candidate explanation for "
                         "1,300%% CPU that decodes 25 matrices a second, and is worth "
                         "measuring rather than assuming")
    ap.add_argument("--modes", default="cpu,device",
                    help="cpu = the reference dequant; device = dequant on --device")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--verify", action="store_true",
                    help="assert the two modes produce torch.equal BF16 slabs")
    ap.add_argument("--json", help="write the measured rows here")
    args = ap.parse_args()

    import torch

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    gguf = gs.GgufFile(args.gguf)
    present = fused_rows(gguf, args.experts)
    if not present:
        print("gguf_decode_bench: no layer in %s has all three fused expert tensors "
              "with %d experts' worth of bytes present (%d bytes on disk) -- fetch "
              "more of the part, or lower --experts"
              % (args.gguf, args.experts, gguf.size), file=sys.stderr)
        return 3
    layers = args.layer or [sorted(present)[0]]
    for layer in layers:
        if layer not in present:
            print("gguf_decode_bench: layer %d is not complete in this part (have %s)"
                  % (layer, ", ".join(str(x) for x in sorted(present))), file=sys.stderr)
            return 3
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(json.dumps({
        "gguf": os.path.basename(args.gguf),
        "bytes": gguf.size,
        "complete_routed_layers": sorted(present),
        "layers": layers,
        "experts": args.experts,
        "decode_threads": args.decode_threads,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else str(device)),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "types": sorted({present[layer][p]["type"] for layer in layers for p in PROJECTIONS}),
    }, sort_keys=True), flush=True)

    rows_out: List[Dict[str, Any]] = []
    slabs: Dict[str, Any] = {}
    for layer in layers:
        for mode in modes:
            decode_device = None if mode == "cpu" else device
            for rep in range(args.repeat):
                meter = progress_meter.Progress(
                    args.experts * 3, label="bench L%03d %s" % (layer, mode))
                gate_up, down, seconds, read_bytes = fill_once(
                    gguf, present[layer], experts=args.experts,
                    decode_threads=args.decode_threads, device=device,
                    decode_device=decode_device, torch=torch, meter=meter)
                meter.close()
                matrices = args.experts * 3
                row = {
                    "layer": layer,
                    "mode": mode,
                    "rep": rep,
                    "seconds": round(seconds, 3),
                    "matrices": matrices,
                    "ms_per_matrix": round(seconds / matrices * 1000, 3),
                    "artifact_bytes_read": read_bytes,
                    # 36,288 matrices is one window of the 42 main routed
                    # layers x 288 experts x 3 projections.
                    "projected_min_per_window": round(
                        seconds / matrices * 36288 / 60.0, 2),
                    "ggml_types": sorted({present[layer][p]["type"] for p in PROJECTIONS}),
                    "decode_threads": args.decode_threads,
                    "torch_num_threads": torch.get_num_threads(),
                }
                if device.type == "cuda":
                    row["peak_device_bytes"] = int(torch.cuda.max_memory_allocated(device))
                print(json.dumps(row, sort_keys=True), flush=True)
                rows_out.append(row)
                if args.verify and rep == 0:
                    slabs.setdefault(layer, {})[mode] = (gate_up, down)
                else:
                    del gate_up, down
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if args.verify:
            have = slabs.get(layer, {})
            if len(have) < 2:
                print("gguf_decode_bench: --verify needs two modes", file=sys.stderr)
                return 3
            (a_mode, (a_gate, a_down)), (b_mode, (b_gate, b_down)) = sorted(have.items())
            same = bool(torch.equal(a_gate, b_gate) and torch.equal(a_down, b_down))
            verdict = {
                "verify": "bf16_slab_bitwise",
                "layer": layer,
                "modes": [a_mode, b_mode],
                "experts": args.experts,
                "bf16_elements_compared": int(a_gate.numel() + a_down.numel()),
                "torch_equal": same,
            }
            if not same:
                delta = (a_gate.float() - b_gate.float()).abs().max()
                verdict["max_abs_diff_gate_up"] = float(delta)
            print(json.dumps(verdict, sort_keys=True), flush=True)
            rows_out.append(verdict)
            del slabs[layer]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not same:
                print("gguf_decode_bench: REFUSED -- the accelerator decode does NOT "
                      "reproduce the cpu slab bitwise; the fast path may not ship",
                      file=sys.stderr)
                return 1

    if args.json:
        Path(args.json).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows_out) + "\n",
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
