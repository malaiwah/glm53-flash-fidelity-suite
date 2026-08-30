# Plan of record — how we scale captures, and what it costs

**Status:** plan of record, 2026-08-30.

**Measured 2026-08-30** (`766a7e8`). The layer-outer engine is built and
bit-identical to the window-outer schedule on two architectures and two devices;
`docs/GLM53-LAYER-OUTER.md` carries the digests. Measured on Fruit on an L4:
peak CUDA allocated **10.409 -> 2.167 GB (4.80x)**, resident weights **9.144 ->
1.471 GB**. The GLM-5.3 projection below is revised accordingly and is now an
extrapolation from measurement rather than from arithmetic alone.

Revised GLM-5.3 root capture: peak VRAM **~47-51 GB** (not 81.7 -- the engine
streams whole layers, so only embed+head+norm, 3.81 GB, stays resident),
**0.4-1.6 min/window** (not 13-26), **Stage C $1.08-3.52** (not $38-96). One
H200 with ~90 GB spare.

Two figures in that projection to distrust until the first `layer_load` line of
a real Stage B run, per the engine's own report: the expert-fusion transient
(bounded from a CPU RSS reading, and Fruit's routed set is 24x smaller at the
same vocabulary), and a ~0.13 ms/tensor size-independent load overhead that
extrapolates to ~12.5 min/run over GLM-5.3's 76,800 source expert tensors per
layer -- the same order as the IO itself.

---

## 1. The parallelism decision, and why two of three options are wrong

| Form | Bit-identical? | I/O cost | Verdict |
|---|---|---|---|
| **Tensor parallel** (split a layer across GPUs) | **NO** | — | **Rejected** |
| **Window parallel** (each GPU takes some windows) | yes | **N x weights** | Rejected for large models |
| **Pipeline parallel** (each GPU owns a layer slice) | yes | 1x weights | **Adopted** |

**Tensor parallel is rejected on correctness, not performance.** Splitting a
layer across devices changes reduction and expert-combine order, which changes
the arithmetic. Not hypothetical: it is the measured mechanism behind our own
lane gap. `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` is a **byte-identical mirror**
of `brandonmusic/GLM-5.3-Flash-tr3-4bpw`; the same bytes read
`0.025503427634363769` on our lane and `0.024554564249958208` on his — ~9.5e-4
nats apart with nothing differing but the runtime. A TP capture would not be a
faster measurement of the same thing; it would be a different lane, and under
our own rules could not be ranked against any existing row.

**Window parallel is bit-safe but pays N times for the weights.** Each device
needs every layer, so a 1.5 TB model streamed layer-by-layer becomes 12 TB of
reads across 8 devices. Fine for models that fit one device; wrong for exactly
the ones that forced layer-outer to exist.

**Pipeline parallel is what layer-outer was already reaching for.** Device *k*
owns layers `[k*L/N, (k+1)*L/N)`; windows flow through. Each layer is
materialised **exactly once in total**, so disk I/O equals the single-device
run. Every window's arithmetic on a given layer happens entirely on one device
with no cross-device reduction, so it must reproduce the single-device digests
**bitwise** — and that is the acceptance test, not a benchmark.

### 1.1 Fetch/compute overlap comes free with layer-outer

Layer-outer needs layer *k* only when it reaches it, so layer *k+1* downloads
while layer *k* computes. Wall-clock becomes `max(fetch, compute)` rather than
`fetch + compute`. This matters because fetch is network-bound and does not
shrink with device count: 1.5 TB at the 430-600 MB/s we have measured is ~45-60
minutes regardless of how many GPUs are attached. Without overlap, a multi-GPU
run pays that hour at the multi-GPU rate.

## 2. Pricing (JarvisLabs, observed 2026-08-30, spot, IN2)

Per-GPU-hour and linear; 8 free of each at time of writing:

| GPU | VRAM | $/GPU-h | 8x |
|---|---:|---:|---:|
| H200 | 141 GB | 1.99 | 15.92 |
| H100 | 80 GB | 1.19 | 9.52 |
| RTX-PRO6000 (IN1) | 96 GB | 0.99 | 7.92 |
| L4 | 24 GB | 0.29 | 2.32 |

**Under layer-outer, per-device memory is one layer plus activations** — ~19.3
GB/layer for the 78L/6144 models. That fits an 80 GB H100 with room to spare, so
**8x H100 is expected to beat 1x H200 on both cost and wall-clock.** The H200's
141 GB buys nothing once the schedule stops needing the model resident.
RTX-PRO6000 is IN1-only, which matters if a shared filesystem is in play
(filesystems are region-bound).

## 3. Cost model

```
wall  ~ max(bytes / fetch_rate, windows * runs * min_per_window / devices) + fill/drain
cost  ~ wall * devices * $per_gpu_hour
```

`min_per_window` scales roughly as `layers * hidden^2` against the measured
GLM-5.3-Flash anchor of **2.37-3.12 min/window** (45 layers, hidden 4096, single
device) — a ~3.9x factor for the 78L/6144 models.

### 3.1 That anchor is a STREAMING anchor, and the model above is missing a term

Measured on Qwen3.8-27B (M1, 2026-08-30, `docs/M1-QWEN38-ROOT-LEARNINGS.md`), one
RTX PRO 6000, 64L/hidden 5120, **model resident on the device**: a 512-window root
capture took **335.1 s = 0.0109 min/window**, ~250x below the 2.37-3.12 band.
The band describes streaming a model that does not fit; when the model fits, the
forward pass is the only cost and capture is nearly free. Use the streaming
anchor only for the streaming regime.

**The formula above prices the capture and omits the comparison, which is the
larger term.** On the same box and the same data, one 512-window comparison took
**60 min 19 s (0.118 min/window)** against that 335 s capture — the comparison
costs **~10.8x the capture it consumes**. `dscompare._replay` applies the head
with a numpy matmul on the CPU and only the fp64 KLD reduction runs on the GPU,
so a comparison pins ~20 cores while the GPU sits at 0%. Concurrent comparisons
contend: two together ran ~1h40m each rather than ~1h.

    wall_compare ~ windows * candidates * min_per_window_compare / cpu_parallelism

with a measured `min_per_window_compare` of **0.118** at vocab 248,320 / hidden
5,120. It scales with `positions * vocab * hidden`. **Budget rungs by comparison
count, not by model gigabytes**, and treat moving `_replay` onto the GPU — the
head is already resident there for the KLD step — as the highest-value
optimization available to the toolchain.

**Anchors already paid for:** the Flash 4-rung ladder at $8.02 / ~$11.60 / $6.65
/ $5.41 per artifact; Fruit root+candidate at $0.25 total; the 0.1B fixture
end-to-end at $0.00; **M1 Qwen3.8-27B root (3 cold runs) + 2 candidates + 3
comparisons + publish at $5.12 total.**

**Fetch rate, measured:** 52 GiB in 85 s = **627 MB/s** with
`HF_HUB_ENABLE_HF_TRANSFER=1` on a 28-vCPU IN1 box — at or above the top of the
430-600 MB/s band assumed in §1.1.

## 4. Families, ranked by value per dollar

| Family | Reference | Size | Geometry | Quant children | Note |
|---|---|---:|---|---:|---|
| **GLM-5.2** | `zai-org/GLM-5.2` | 1,506.7 GB | 78L/6144/256e | **100** | natively unquantized; no `-BF16` sibling |
| **GLM-5.3** | `zai-org/GLM-5.3-BF16` | 1,506.7 GB | 78L/6144/256e | 37 | same geometry as 5.2 — one engine serves both |
| **GLM-5.3-Flash** | `-Flash-BF16` | 642.7 GB | 45L/4096/288e | 67 | **re-capture**: drives our floor 0.011506 -> 0.0 |
| **Qwen3.8-27B** | `Qwen/Qwen3.8-27B` | 55.6 GB | 64L/5120 dense, **hybrid attn + vision + MTP** | — | **DONE (M1, $5.12)**: new same-lane group, 37 old rows NOT upgraded — see below |
| Qwen3.5-397B | `Qwen3.5-397B-A17B` | 806.8 GB | — | GGUF/MLX/REAP | backfill; 1,553 likes on the root |

**GLM-5.2 is the best deal on the board.** Identical size and geometry to
GLM-5.3 — one engine, one panel design, one capture cost — but ~3x the quantized
children (unsloth GGUF 642 likes, nvidia NVFP4 319, lukealonso, 0xSero REAP).
Per dollar of root capture it unlocks the most downstream measurement.

**Re-capturing GLM-5.3-Flash is not redundant.** Our eight Flash rows sit above a
**0.011506** floor because they were measured against a teacher captured on
another stack. A same-lane root capture drives that floor to **exactly 0.0** —
demonstrated on Fruit — converting every Flash attributable number from inferred
by subtraction into directly measured.

**Qwen3.8-27B was a rounding error** at 55.6 GB — done for **$5.12** — but the
claim that it "retroactively upgrades 37 existing rows" was WRONG and is
withdrawn. A same-lane root does not upgrade a row measured against a different
teacher: the comparability key binds the reference, so the new rows form a NEW
group (`cmp--05e16411a5932713`) beside the old one (`cmp--4a93702ded23e01a`), and
the 37 older rows keep their inferred floors untouched. What a same-lane capture
buys is a *new* group whose floor is a measured 0.0, plus — because the panel was
deliberately reused — a legible lane delta: AWQ-INT4 reads 0.022449 same-lane
against 0.022818 cross-lane, **3.685e-4 nats lower with higher top-1**, the
direction a cross-stack term predicts. Expect the same of the Flash re-capture:
it creates a clean group, it does not retroactively fix the eight existing Flash
rows.

Note also that the geometry line above understated the model: Qwen3.8-27B is
`Qwen3_5ForConditionalGeneration` — multimodal, with 48 linear-attention layers
to 16 full-attention, and an MTP block. Dense (zero expert tensors). Check
architecture before sizing a rung from a one-line description.

## 5. Sequencing

1. **Single-device layer-outer** proves bit-identity against the current schedule
   on two small models. The correctness gate. *(in flight)*
2. **Pipeline-parallel** as a pure throughput change whose acceptance test is
   reproducing the single-device digests exactly. Fetch/compute overlap lands here.
3. **Cheapest root first** — Qwen3.8-27B *(**DONE** 2026-08-30, $5.12, floor
   measured at exactly 0.0; `docs/M1-QWEN38-ROOT-LEARNINGS.md`)*, then the
   GLM-5.3-Flash re-capture — to validate the cost model against small bills
   before a 1.5 TB run. M1's verdict on the model: the capture term was ~250x
   cheaper than projected and the comparison term, which the model omitted
   entirely, is the one that costs.
4. **GLM-5.2 and GLM-5.3** on the proven engine.
5. Qwen3.5-397B if the backfill still looks worth it.

## 6. What would change this plan

- Layer-outer failing bit-identity on `glm_moe_dsa` — then the schedule is wrong
  and nothing downstream is safe.
- Pipeline fill/drain costing more than the parallelism returns at 78 layers.
- Fetch rate materially below 430-600 MB/s, which would make every large capture
  fetch-bound and shift the answer back toward one device.
- Spot preemption on multi-hour runs; measured behaviour, not assumed.
