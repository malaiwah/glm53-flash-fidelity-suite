#!/usr/bin/env python3
"""Rent a GPU, measure what it is worth for SCORING, give it back.

Two jobs in one command, and the second is the reason it exists.

**It answers "how long will this take here?"** A spec sheet does not, because a
fidelity measurement is neither training nor generation: per window it walks
every routed expert matrix, dequantises it, does ONE skinny GEMM against a
2047-token block, and throws the weights away. So the numbers that decide the
bill are streaming bandwidth and a skinny GEMM at the model's own shapes --
and on the streaming lane the dominant term turns out to be neither the card's
compute nor its VRAM, but the HOST's PCIe link.

That is not a hypothesis. Measured on three rentals:

    card                      read GB/s   h2d GB/s   gemm TF   per-matrix ms
    RTX 4000 Ada Generation       329.4        1.6      83.0          10.593
    RTX A4500                     566.2       24.4      80.4           1.148
    A100 80GB PCIe               1378.5       26.6     167.0           0.899

The 4000 Ada and the A4500 have the SAME compute to within 3%. The 4000 Ada
took 9.2x longer per matrix, because the host it landed on gave 1.6 GB/s
host-to-device instead of 24. Rent the same card from a different host and that
number changes; it is a property of the machine, not of the GPU.

**And it proves the whole rental loop works** -- rent, wait for SSH, upload,
execute, collect a result, tear down -- for a few cents, before a real
measurement commits hours and dollars to a provider or a host nobody has tried.
Every failure this suite hit while adding three providers (ids that are not
integers, a running state spelled differently, storage that dies with the
instance, sshd not up when the API says "running", a host whose PCIe is
oversubscribed 15x) shows up here in about four minutes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

PAYLOAD = Path(__file__).with_name("cardbench_payload.py")

# Measured on the reference card, so an estimate has something real to scale.
REFERENCE = {"gpu": "NVIDIA A100 80GB PCIe", "stream_matrix_ms": 0.899}


def run_bench(provider, *, gpu: Optional[str] = None, ask_id: Optional[Any] = None,
              storage: int = 30, name: str = "fidbench",
              con=None, keep: bool = False) -> Dict[str, Any]:
    """Rent one instance, benchmark it, destroy it. Returns the payload's JSON.

    The instance is destroyed in `finally`, including when the benchmark
    raises: a benchmark that leaks an instance has cost more than it measured.
    """
    def say(msg):
        if con is not None:
            con.step(msg)
        else:
            print("  " + msg, flush=True)

    mid = None
    started = time.time()
    try:
        kw = {"storage": storage, "name": name}
        if ask_id is not None:
            kw["ask_id"] = ask_id
        if gpu:
            kw["gpu_type"] = gpu
        created = provider.create(**kw)
        mid = created.get("machine_id") or created.get("pod_id")
        if mid is None:
            raise RuntimeError("provider returned no machine id: %r" % (created,))
        say("rented %s" % mid)
        provider._endpoint(mid, wait=900)
        say("ssh up after %.0fs" % (time.time() - started))
        provider.upload(mid, str(PAYLOAD), "/tmp/cardbench.py")
        out = provider.exec_stdout(mid, "python3 /tmp/cardbench.py 2>&1 | tail -30",
                                   timeout=900)
        try:
            doc = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except Exception:                                 # noqa: BLE001
            raise RuntimeError("benchmark produced no JSON:\n%s" % out[-600:])
        doc["provider"] = getattr(provider, "provider", "?")
        doc["wall_seconds"] = round(time.time() - started, 1)
        return doc
    finally:
        if mid is not None and not keep:
            try:
                provider.destroy(mid)
                say("destroyed %s" % mid)
            except Exception as exc:                      # noqa: BLE001
                say("DESTROY FAILED for %s: %s -- check the provider console"
                    % (mid, exc))


def bench_existing(provider, machine_id, *, con=None) -> Dict[str, Any]:
    """Benchmark a box that is ALREADY rented, without touching its lifecycle.

    This is the preflight form: the instance exists, setup has installed torch,
    and the question is whether the next three hours are worth starting here.
    It creates nothing and destroys nothing.
    """
    provider.upload(machine_id, str(PAYLOAD), "/tmp/cardbench.py")
    out = provider.exec_stdout(machine_id, "python3 /tmp/cardbench.py 2>&1 | tail -30",
                               timeout=900)
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1])
    except Exception:                                     # noqa: BLE001
        raise RuntimeError("preflight benchmark produced no JSON:\n%s" % out[-600:])


def gate(doc: Dict[str, Any], *, min_h2d_gbps: Optional[float] = None,
         min_gemm_tflops: Optional[float] = None) -> Optional[str]:
    """Is this machine fast enough to be worth the run? None means yes.

    The case this exists for is real and was measured twice on one Vast offer:
    an RTX 4000 Ada whose host wires the card at **Gen4 x1 of Gen4 x16**. Same
    GPU, same compute to within 3% of a sibling host -- and 1.6 GB/s instead of
    11.0, which turns a 3-hour measurement into a 20-hour one. Nothing in any
    catalogue exposes link width, and the failure is invisible until the bill
    arrives.

    It is checked AFTER setup and BEFORE the fetch, because setup is minutes
    and the fetch is the first expensive thing.
    """
    bad = []
    h2d = doc.get("h2d_GBps")
    if min_h2d_gbps and h2d is not None and h2d < min_h2d_gbps:
        link = (doc.get("pcie_load") or {}).get("text", "unknown")
        bad.append("host->device is %.1f GB/s, below the required %.1f "
                   "(PCIe link under load: %s)" % (h2d, min_h2d_gbps, link))
    gemm = doc.get("expert_gemm_TFLOPs")
    if min_gemm_tflops and gemm is not None and gemm < min_gemm_tflops:
        bad.append("expert GEMM is %.1f TFLOP/s, below the required %.1f"
                   % (gemm, min_gemm_tflops))
    return "; ".join(bad) if bad else None


def estimate(doc: Dict[str, Any], *, matrices_per_window: int) -> Dict[str, Any]:
    """Scale a measured per-matrix time into a per-window one.

    Deliberately linear and deliberately labelled an estimate: the per-matrix
    step is the inner loop of the streaming lane, so windows scale with it, but
    a real run also pays fetch, materialize and the panel, which this does not
    model and which `measure-cloud --dry-run` does.
    """
    per_matrix_ms = doc.get("stream_matrix_ms")
    if not per_matrix_ms:
        return {}
    win_min = per_matrix_ms * matrices_per_window / 1000.0 / 60.0
    return {
        "matrices_per_window": matrices_per_window,
        "minutes_per_window": round(win_min, 2),
        "relative_to_reference": round(
            per_matrix_ms / REFERENCE["stream_matrix_ms"], 2),
        "reference": REFERENCE["gpu"],
    }


def render(doc: Dict[str, Any], est: Optional[Dict[str, Any]] = None) -> str:
    lines = [
        "  gpu                    %s  (%.1f GB, sm %s)"
        % (doc.get("gpu"), doc.get("vram_gb", 0), doc.get("sm")),
        "  torch / cuda           %s / %s" % (doc.get("torch"), doc.get("cuda")),
        "",
        "  device read            %8.1f GB/s" % doc.get("read_GBps", 0),
        "  host -> device  cold   %8.1f GB/s" % doc.get("h2d_cold_GBps", 0),
        "  host -> device  warm   %8.1f GB/s   <- the streaming lane's real limit"
        % doc.get("h2d_GBps", 0),
        "  PCIe link  idle        %s" % (doc.get("pcie_idle") or {}).get("text", "?"),
        "  PCIe link  under load  %s" % (doc.get("pcie_load") or {}).get("text", "?"),
        "  expert GEMM            %8.1f TFLOP/s bf16 (2047x4096x2048)"
        % doc.get("expert_gemm_TFLOPs", 0),
        "  dense 4k GEMM          %8.1f TFLOP/s bf16" % doc.get("dense_4k_TFLOPs", 0),
        "  per-matrix step        %8.3f ms    (upload + cast + matmul)"
        % doc.get("stream_matrix_ms", 0),
    ]
    if est:
        lines += [
            "",
            "  ESTIMATE (streaming lane, %d matrices/window)"
            % est["matrices_per_window"],
            "    %.2f min/window, %.2fx the %s reference"
            % (est["minutes_per_window"], est["relative_to_reference"],
               est["reference"]),
            "    This scales the inner loop only. It does NOT include fetch,",
            "    materialize or the panel -- `measure-cloud --dry-run` does.",
        ]
    return "\n".join(lines)
