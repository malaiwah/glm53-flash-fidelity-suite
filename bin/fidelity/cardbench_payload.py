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

    # host->device, which is what a streaming lane pays when the model does not fit
    host = torch.empty(n, dtype=torch.bfloat16, device="cpu").pin_memory()
    dst = torch.empty_like(buf)
    t = timeit(lambda: dst.copy_(host, non_blocking=True), 10)
    out["h2d_GBps"] = round(host.numel() * 2 / t / 1e9, 1)
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
