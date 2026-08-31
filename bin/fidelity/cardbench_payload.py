#!/usr/bin/env python3
"""What a card is worth for FIDELITY SCORING, which is not what a spec sheet says.

The measure stage is not training and it is not generation. Per window it
walks every routed expert matrix, dequantises it, does ONE matmul against a
2047-token activation block, and throws the weights away. So the two numbers
that decide how long a measurement takes are:

  1. how fast the card can stream weights it will use exactly once  -> BANDWIDTH
  2. how fast it does a skinny GEMM at the model's own shapes       -> GEMM

Everything is timed with cuda events after a warmup, and the GEMM shapes are
GLM-5.3-Flash's real ones (hidden 4096, expert intermediate 2048, 2047-token
window), so the ratio between cards transfers to a real run.
"""
import json, os, sys, time
import torch

def sync(): torch.cuda.synchronize()

def _pcie_state():
    """Link generation and width, straight from the driver.

    This is what makes "the card was asleep" and "the host is oversubscribed"
    distinguishable instead of a matter of opinion: a Gen1 x1 link at idle that
    becomes Gen4 x16 under load was parked, and a link that stays narrow under
    sustained traffic is *usually* what the machine offers.

    Usually, not always -- and the exception is the fastest machine in the
    survey. A Lambda GH200 reports **Gen4 x1 of Gen4 x1** at idle and under
    load, the exact signature of the oversubscribed host this field exists to
    expose, while measuring **379 GB/s** host-to-device: fourteen times a
    Gen4 x16 A100. Its PCIe link is vestigial because the host memory does not
    travel over PCIe at all, it travels over NVLink-C2C. So the LINK is
    context for the bandwidth number and never a substitute for it. Anything
    that gates on width rather than on measured GB/s will refuse the best card
    it can rent; `bench.gate` deliberately gates on `h2d_GBps` alone.
    """
    import subprocess
    q = ("pcie.link.gen.current,pcie.link.width.current,"
         "pcie.link.gen.max,pcie.link.width.max")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        gen, width, gen_max, width_max = [x.strip() for x in out.split(",")[:4]]
        return {"gen": gen, "width": width, "gen_max": gen_max,
                "width_max": width_max, "text": "Gen%s x%s of Gen%s x%s"
                % (gen, width, gen_max, width_max)}
    except Exception as exc:                              # noqa: BLE001
        return {"error": str(exc)[:80]}


def timeit(fn, iters, warmup=3):
    for _ in range(warmup): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync()
    return (time.perf_counter() - t0) / iters

def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no cuda"})); return 1
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    out = {"gpu": name, "vram_gb": round(props.total_memory / 1024**3, 1),
           "sm": "%d.%d" % (props.major, props.minor),
           "torch": torch.__version__, "cuda": torch.version.cuda}

    # -- 1. streaming bandwidth: read a big bf16 buffer once, like a weight ---
    n = 256 * 1024 * 1024 // 2                      # 256 MB of bf16
    buf = torch.randn(n, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: buf.sum(), 20)
    out["read_GBps"] = round(buf.numel() * 2 / t / 1e9, 1)

    # host->device, which is what a streaming lane pays when the model does not
    # fit. MEASURED IN TWO PHASES ON PURPOSE.
    #
    # An NVIDIA card parks its PCIe link when idle -- dropping link generation
    # and/or width, commonly to Gen1 x1 -- and only ramps back under sustained
    # traffic. A short warmup therefore measures the RAMP, not the link, and
    # reports a number several times too low for a card that is merely asleep.
    # The first version of this benchmark did exactly that and concluded a host
    # was oversubscribed when the card had simply not woken up.
    #
    # So: record the link state from nvidia-smi cold, push traffic for a few
    # seconds, record it again, and report cold and warm bandwidth separately.
    host = torch.empty(n, dtype=torch.bfloat16, device="cpu").pin_memory()
    dst = torch.empty_like(buf)
    nbytes = host.numel() * 2

    out["pcie_idle"] = _pcie_state()
    # cold: the very first transfers, before the link has any reason to ramp
    sync(); t0 = time.perf_counter()
    for _ in range(3):
        dst.copy_(host, non_blocking=True)
    sync()
    out["h2d_cold_GBps"] = round(3 * nbytes / (time.perf_counter() - t0) / 1e9, 1)

    # sustained: keep the link busy for a fixed wall-clock stretch, then time it
    warm_until = time.perf_counter() + 4.0
    while time.perf_counter() < warm_until:
        dst.copy_(host, non_blocking=True)
    sync()
    out["pcie_load"] = _pcie_state()
    t = timeit(lambda: dst.copy_(host, non_blocking=True), 20, warmup=5)
    out["h2d_GBps"] = round(nbytes / t / 1e9, 1)
    out["h2d_ramp_x"] = (round(out["h2d_GBps"] / out["h2d_cold_GBps"], 2)
                         if out["h2d_cold_GBps"] else None)
    del host, dst

    # -- 2. the shapes a GLM-5.3-Flash window actually multiplies -------------
    T, H, I = 2047, 4096, 2048
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    w = torch.randn(H, I, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: x @ w, 50)
    out["expert_gemm_ms"] = round(t * 1e3, 3)
    out["expert_gemm_TFLOPs"] = round(2 * T * H * I / t / 1e12, 1)

    # a dense bf16 GEMM, for a spec-sheet-comparable number
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: a @ b, 50)
    out["dense_4k_TFLOPs"] = round(2 * 4096**3 / t / 1e12, 1)
    del a, b

    # -- 3. the whole per-matrix step: upload -> cast -> matmul --------------
    # This is the shape of the streaming inner loop when weights do not fit.
    wq = torch.empty(H, I, dtype=torch.bfloat16, device="cpu").pin_memory()
    def step():
        g = wq.to(dev, non_blocking=True)
        return x @ g
    t = timeit(step, 30)
    out["stream_matrix_ms"] = round(t * 1e3, 3)

    print(json.dumps(out, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
