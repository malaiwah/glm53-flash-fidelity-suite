# M1.5 — the replay backend, measured

Every number in `docs/CAPTURE-SCALING-PLAN.md` §3.2–3.4 and in
`docs/M1-QWEN38-ROOT-LEARNINGS.md` learnings 2/4/22 comes from the files in this
directory. Nothing here was captured: all three runs read
[`malaiwah/qwen38-27b-fidelity-root-v1`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-root-v1)
exactly as published (`dataset_sha256 8a658364…`, re-verified on the box), and
nothing was uploaded.

## The box

One on-demand **NVIDIA RTX PRO 6000 Blackwell Server Edition** (96 GB, sm_12.0),
28 vCPU, JarvisLabs IN1 machine 488056, destroyed after the run. Same GPU class
and same core count as the M1 rental, which is what makes the wall-clocks below
comparable to M1's own. torch 2.11.0+cu130, numpy 2.2.6 on **scipy-openblas**.

## What was run

| file | what it is |
|---|---|
| `bench-gpu.json` | the 512-window self-compare with `--replay-device cuda` |
| `bench-cpu.json` | the identical comparison with `--replay-device numpy` (the default, the published path) |
| `bench-delta.json` | `KLD(numpy replay ‖ cuda replay)` over 16 real root windows |
| `comparison-receipt-cuda.json` | the sealed receipt from the cuda run |
| `comparison-receipt-numpy.json` | the sealed receipt from the numpy run |

Both comparisons are the **same command** but for the two replay flags:

```
fidelity-dataset compare --reference root-a --candidate root-b --out … \
    --self-compare --force-compute --no-verify-tensors \
    --device cuda --replay-device {cuda|numpy}
```

`--device cuda` on **both** sides is deliberate: M1's 60m19s comparison ran the
fp64 estimator on the GPU and only the head matmul in numpy. Timing the numpy
replay against a CPU estimator would have inflated the speedup with a term the
fix does not touch.

`root-b` is a hardlink copy of `root-a` (`cp -al`), so the two trees have equal
`capture_content_digest` and the comparison is an SC-1 reproduction
confirmation — the same class as M1's published floor, and the one quantity a
backend change must not be able to move.

## The three results

**1. The floor is backend-independent.** Both runs: `comparison_kind`
`reproduction_confirmation`, metric **exactly 0.0**, top-1 **exactly 1.000000**,
every percentile 0.0, `self_compare.force_compute_agreed: true` (the computed
array is bitwise identical to the hash proof), and `tokenwise-kld.npy` sha256
**`8be5dccaf885d7dadca697c4d54cff60d1c8c8333b57761c31d882c9f9ec9e5d`** — the
digest M1 published, byte for byte, through a matmul that ran on a different
processor.

**2. The speedup.** See `bench-*.json` `seconds`, and §3.2 of the plan.
Peak device memory on the cuda path: **7,129,923,584 bytes (6.64 GiB)**, of
which the fp32 head is 5.09 GB.

**3. The replay-backend floor.** `bench-delta.json`: mean **5.237e-12 nats**
over 32,752 positions, max 1.791e-10, **top-1 agreement 1.000000** (not one
argmax flipped), max absolute logit delta 3.624e-05. Small enough that a
published row agrees to ~9 significant figures; not zero, which is why the numpy
path stays the default and `comparator.replay_backend` is now on every receipt.

## Reproducing

`bin/selftest_replay_device.py` (T12) holds the same properties offline on
fixtures, with no GPU and no network, and is wired into `bin/selftest_all.sh`.
The driver used on the box is not committed — it is 200 lines of `subprocess`
around the real CLI, and the CLI invocations above are the whole of it.
