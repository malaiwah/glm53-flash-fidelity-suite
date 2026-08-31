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

# Same set, and for the same reason, as measure_cloud._RUNNING_STATES: every
# provider spells "this box will accept work now" differently.
_READY_STATES = frozenset({"running", "run", "active", "ready"})


def wait_ready(provider, machine_id: Any, *, wait: float = 900,
               poll: float = 10.0) -> None:
    """Block until the instance will accept an exec -- on ANY backend.

    The three SSH backends expose `_endpoint`, which waits for a reachable
    host:port. That is the only honest readiness signal when the next call
    opens a socket, so it is used whenever it exists.

    JarvisLabs has no endpoint at all: it execs through its own CLI, so there
    is no host and no port to wait for. Calling `_endpoint` on it raised
    AttributeError -- and it raised *after* `create` had already started the
    meter, which is the expensive half. `--provider jarvislabs` is one of the
    four choices this tool advertises in `--help`, and on that one choice it
    rented a box, failed on the next line, and tore it down again every single
    time. Nothing offline caught it because nothing offline asks a provider
    object for a method it does not have.

    So: endpoint where there is one, instance state where there is not.
    """
    ep = getattr(provider, "_endpoint", None)
    if callable(ep):
        ep(machine_id, wait=wait)
        return
    deadline = time.time() + wait
    seen = None
    while time.time() < deadline:
        inst = provider.get(machine_id)
        seen = getattr(inst, "status", None) if inst is not None else None
        if str(seen or "").strip().lower() in _READY_STATES:
            return
        time.sleep(poll)
    raise RuntimeError(
        "instance %s never became ready within %ds (last state: %r)"
        % (machine_id, int(wait), seen))


def _catalogue_rate(provider, inst) -> Optional[float]:
    """The listed rate for the offer this instance corresponds to, or None.

    Second best, and labelled as such by living in its own function: it is the
    price the catalogue advertises rather than the price the contract carries.
    Used only where the provider genuinely publishes no per-instance rate.
    """
    try:
        offers = provider.gpus()
    except Exception:                                     # noqa: BLE001
        return None
    want = (getattr(inst, "gpu_type", "") or "").strip().lower()
    spot = bool(getattr(inst, "is_spot", False))
    region = (getattr(inst, "region", "") or "").strip().lower()
    cands = [o for o in offers
             if (o.gpu_type or "").strip().lower() == want
             and bool(getattr(o, "spot", False)) == spot]
    if region:
        exact = [o for o in cands if (o.region or "").strip().lower() == region]
        cands = exact or cands
    prices = [float(o.price) for o in cands if o.price]
    return round(min(prices), 4) if prices else None


def rented_rate(provider, machine_id: Any) -> Dict[str, Any]:
    """What the box that was actually rented costs per hour, from its record.

    The point of this benchmark is dollars per MEASUREMENT, and that number is
    `minutes_per_window x $/hour`. Reading the rate off a catalogue afterwards
    is not the same thing: on a marketplace the ask you searched and the
    contract you got are different objects, and a rate typed into a table by
    hand cannot be re-derived from the artifact. So the rate is read back from
    the instance that is billing, and stored beside the timing it explains.

    Providers spell it four ways and one does not report it at all; a missing
    rate is recorded as null rather than guessed.
    """
    try:
        inst = provider.get(machine_id)
    except Exception:                                     # noqa: BLE001
        return {}
    if inst is None:
        return {}
    raw = getattr(inst, "raw", None) or {}
    rate, source = None, "none"
    for key, scale in (("cost_per_hr", 1.0), ("dph_total", 1.0),
                       ("price_cents_per_hour", 0.01)):
        if raw.get(key) is not None:
            rate, source = round(float(raw[key]) * scale, 4), "contract:" + key
            break
    if rate is None:
        # JarvisLabs reports a running TOTAL and no rate. Its catalogue does
        # carry one, so the rate is still derived from the provider rather
        # than typed into a table by a human -- matched on the GPU, the region
        # and the billing mode of the instance that exists. It is labelled
        # `catalogue` because it is the advertised price, not the contract.
        rate = _catalogue_rate(provider, inst)
        source = "catalogue" if rate is not None else "none"
    return {"usd_per_hour": rate,
            "rate_source": source,
            "gpu_type_billed": getattr(inst, "gpu_type", None),
            "region": getattr(inst, "region", None),
            "is_spot": bool(getattr(inst, "is_spot", False))}


def _measure(provider, machine_id: Any, say, *, attempts: int = 4,
             settle: float = 45.0) -> Dict[str, Any]:
    """Run the payload and INSIST on a real result, retrying a boot race.

    A receipt full of zeros is worse than a failure: it is publishable, it
    tabulates, and nothing about it says the card was missing. This raises
    instead, naming what the payload said.

    The one error worth retrying is `no cuda`. Lambda reports an instance
    `active` with an IP, and sshd accepts a connection, several minutes before
    the driver stack is usable -- a gpu_1x_h100_sxm5 rented on 2026-08-31
    answered `torch.cuda.is_available() == False` 238 s in and wrote a receipt
    of zeros for a card that was fine. That is the same lesson as
    "watch run STATE, not output counts": the API's readiness and the machine's
    are different claims. Everything else fails at once, because a box that
    cannot import torch will not learn to.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        out = provider.exec_stdout(
            machine_id, "python3 /tmp/cardbench.py 2>&1 | tail -30", timeout=900)
        try:
            doc = json.loads(out[out.index("{"):out.rindex("}") + 1])
        except Exception:                                 # noqa: BLE001
            raise RuntimeError("benchmark produced no JSON:\n%s" % out[-600:])
        err = doc.get("error")
        if not err and doc.get("stream_matrix_ms"):
            return doc
        last = err or "no stream_matrix_ms in %r" % (sorted(doc),)
        if err != "no cuda" or attempt == attempts:
            break
        say("payload says 'no cuda' (%d/%d) -- the driver stack is not up yet, "
            "waiting %ds" % (attempt, attempts, int(settle)))
        time.sleep(settle)
    raise RuntimeError(
        "the benchmark did not measure anything on %s: %s. A receipt of zeros "
        "would tabulate as if it were a slow machine, so none was written."
        % (machine_id, last))


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
        wait_ready(provider, mid, wait=900)
        say("ready after %.0fs" % (time.time() - started))
        rate = rented_rate(provider, mid)
        provider.upload(mid, str(PAYLOAD), "/tmp/cardbench.py")
        doc = _measure(provider, mid, say)
        doc["provider"] = getattr(provider, "provider", "?")
        doc["wall_seconds"] = round(time.time() - started, 1)
        doc["rental"] = dict(rate, machine_id=str(mid),
                             requested_gpu=gpu, requested_ask_id=(
                                 str(ask_id) if ask_id is not None else None),
                             at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime()))
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

    Note what is NOT gated: the link width. A Lambda GH200 reports Gen4 x1 at
    idle and under load -- the oversubscription signature exactly -- and moves
    379 GB/s host-to-device, because its host memory reaches the die over
    NVLink-C2C rather than over PCIe. Gating on width would refuse the fastest
    machine this suite has measured. Bandwidth is the fact; the link is the
    explanation printed beside it.
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
    out = {
        "matrices_per_window": matrices_per_window,
        "minutes_per_window": round(win_min, 2),
        "relative_to_reference": round(
            per_matrix_ms / REFERENCE["stream_matrix_ms"], 2),
        "reference": REFERENCE["gpu"],
    }
    # Dollars per hour is the wrong axis and dollars per WINDOW is the right
    # one: a card at three times the rate that finishes in a third of the time
    # is a wash. The two halves are only comparable when they come from the
    # same rental, so the rate is the one read back off the billing instance.
    rate = (doc.get("rental") or {}).get("usd_per_hour")
    if rate:
        out["usd_per_hour"] = rate
        out["usd_per_window"] = round(win_min / 60.0 * float(rate), 5)
    return out


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
    rent = doc.get("rental") or {}
    # The rate of the CONTRACT, not of the ask -- and the GPU name comes from
    # nvidia-smi above, never from the provider's own instance record, which
    # on one provider is an opaque machine id and on another was simply wrong.
    if rent:
        lines += [
            "  billed rate            %s"
            % ("not reported by this provider"
               if rent.get("usd_per_hour") is None
               else "$%.4f/h" % rent["usd_per_hour"]),
        ]
    if est:
        lines += [
            "",
            "  ESTIMATE (streaming lane, %d matrices/window)"
            % est["matrices_per_window"],
            "    %.2f min/window, %.2fx the %s reference"
            % (est["minutes_per_window"], est["relative_to_reference"],
               est["reference"]),
        ]
        if est.get("usd_per_window") is not None:
            lines.append("    $%.5f per window at $%.4f/h  <- the axis that "
                         "actually ranks providers"
                         % (est["usd_per_window"], est["usd_per_hour"]))
        lines += [
            "    This scales the inner loop only. It does NOT include fetch,",
            "    materialize or the panel -- `measure-cloud --dry-run` does.",
        ]
    return "\n".join(lines)
