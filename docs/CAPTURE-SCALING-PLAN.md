# Plan of record — how we scale captures, and what it costs

**Status:** plan of record, 2026-08-30.

**Amended 2026-08-30 (M1.5).** §3 is rewritten around measurement rather than
extrapolation: the comparison term, which the old model omitted entirely, was
the dominant one, and `compare --replay-device cuda` cuts it **10.13x**
(1,754.71 s -> 173.27 s on the real Qwen3.8 panel, same box, peak 7.13 GB). It
is opt-in and named on every receipt because it moves the last digits by
5.24e-12 nats. Receipts in `reports/m15-replay-backend/`.

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

## 3. Cost model — budget COMPARISONS, not gigabytes

M1 (Qwen3.8-27B, 2026-08-30) and M1.5 (the replay fix, same day, same GPU class)
between them replaced every term in the old model. The old formula priced the
capture, anchored it to a streaming figure, and omitted the comparison entirely
— which was the dominant term. This section is what the measurements say.

```
wall_capture   ~ max(bytes / fetch_rate, windows * runs * min_per_window / devices)
wall_compare   ~ candidates * windows * positions * vocab * hidden * C_replay
cost           ~ (wall_capture + wall_compare) * devices * $per_gpu_hour
```

### 3.1 The three measured constants

| term | value | measured on |
|---|---|---|
| `fetch_rate` | **627 MB/s** (52 GiB in 85 s, `HF_HUB_ENABLE_HF_TRANSFER=1`) | 28-vCPU IN1 box, M1 |
| `min_per_window`, model **resident** | **0.0109** (335.1 s / 512 windows, 27B) | 1x RTX PRO 6000, M1 |
| `min_per_window`, model **streamed** | 2.37-3.12 (GLM-5.3-Flash) | the old anchor — streaming lane ONLY |
| `C_replay`, numpy on CPU | **1.317e-12 s** per (position x vocab x hidden), both sides | 1x RTX PRO 6000 + 28 vCPU, M1.5 |
| `C_replay`, cuda fp32 | **1.30e-13 s** per (position x vocab x hidden), both sides | same box, same comparison, M1.5 |

**The capture anchor is regime-dependent and the two regimes are ~250x apart.**
2.37-3.12 min/window describes STREAMING a model that does not fit the device.
When the model fits, the forward pass is the only cost and the capture is nearly
free: 512 windows of a 27B model took 335.1 s. Use the streaming anchor only for
the streaming regime, which for us means GLM-5.3-Flash (642.7 GB) and larger.

### 3.2 The comparison was the whole bill, and it no longer is

M1 measured one 512-window comparison at **60 min 19 s** against a **335 s**
capture of the same panel on the same box — **10.8x the capture it consumes.**
The cause was one line: `dscompare._replay` reconstructed `logits' = hidden @
head.T` in **numpy on the CPU**, while the GPU — already holding that same head
for the fp64 KLD estimator — sat at 0%. (Watched live during the M1.5 baseline
run: `nvidia-smi` reports 0% utilisation for the entire numpy comparison.)

M1.5 moved it. `compare --replay-device cuda` runs the head matmul on the
device the estimator already uses, one position block at a time, so the full
`[positions x vocab]` fp32 logit array is never materialised on the host.

**Measured, same box, same 512-window comparison, end to end through the CLI:**

| path | wall | GPU | peak device memory |
|---|---:|---:|---:|
| `--replay-device numpy` (default, the published path) | **1,754.71 s (29 min 15 s)** | 0% | — |
| `--replay-device cuda` | **173.27 s** | 88% | **7.13 GB** (6.64 GiB) |

**10.13x** on the real Qwen3.8 panel: 512 contexts, 1,048,064 scored
positions, vocab 248,320, hidden 5,120, both sides replayed. Against M1's 335.1 s
capture of the same panel, the comparison drops from **5.24x the capture to
0.52x** of it — 26.4 minutes of wall clock returned per comparison.

**One caveat on the baseline, stated rather than buried.** M1's own numpy
comparison took **3,619 s** on a different rental of the same GPU class; the
numpy path here took 1,754.71 s, 2.1x faster, for the same work. Nothing in the
code differs — the plausible causes are BLAS build and thread count (this box:
numpy 2.2.6 on scipy-openblas, 28 CPUs by affinity), page-cache state (the
13 GB dataset was warm from the preceding cuda run), and M1's note that two
concurrent comparisons contended. That spread is exactly why the A/B above was
run **in one process, on one box, back to back, with only the replay flag
different**. Quote 10.13x, not 20.9x.

Peak device memory is bounded and small: the fp32 head is 5.09 GB
(248,320 x 5,120 x 4 B) and everything else is one position block —
`--chunk-positions 128` gives 127 MB of fp32 logits and 254 MB of fp64 per side.
The 7.13 GB peak is ~7% of a 96 GB RTX PRO 6000, so a comparison fits alongside a
resident model in MEMORY. It no longer fits alongside one in COMPUTE — see §3.5.

### 3.3 It is opt-in, because it changes the last digits — by 5.24e-12 nats

An fp32 GEMM is not one function. BLAS accumulates in fp32 in an order the
implementation's blocking chooses, so `hidden @ head.T` has different last bits
on OpenBLAS, on Accelerate and on cuBLAS. The published Qwen3.8 rows are
16-significant-digit values of a quantity whose inputs carry that noise, so they
are already BLAS-bound and a backend change moves them.

**How much, measured** — `KLD(numpy-fp32-CPU replay || cuda-fp32 replay)` over
16 real root windows, 32,752 positions, the same estimator on both sides:

| quantity | value |
|---|---:|
| mean tokenwise KLD | **5.237e-12 nats** |
| p99 | 3.029e-11 nats |
| max | 1.791e-10 nats |
| **top-1 agreement** | **1.000000** — not one argmax flipped |
| max absolute logit delta | 3.624e-05 |
| max relative logit delta | 1.360e-06 |

Read that as a **replay-backend floor**, in the same units as every other floor
this campaign quotes. It is **2.2e9 times smaller** than the streaming lane
floor of 0.011506 nats, and **1.75e-9 relative** to the smallest published
Qwen3.8 row (FP8, 0.002989850396847924) — so a row would agree to roughly nine
significant figures and differ somewhere past the tenth. Not zero. Therefore:

- **the numpy path stays the default**, and the published rows stay reproducible
  on the machine and library that produced them;
- every receipt now names `comparator.replay_backend`
  (`numpy:cpu:float32` / `torch:cuda:float32` / `none` for a hash-proof
  short-circuit), because a silent backend swap is precisely the undeclared
  difference this format exists to stop;
- **pick ONE replay backend per comparability group and keep it.** Two rows
  measured under different backends differ by a term neither of them measured.

**THE FLOOR IS BACKEND-INDEPENDENT, and that was verified rather than argued.**
A self-compare replays bitwise-equal hidden states through one head on one
backend, so both sides get bitwise-equal logits and the KLD is exactly 0.0
whatever the backend rounds to. Re-run on the published root through the new
path: metric **exactly 0.0**, top-1 **exactly 1.000000**, every percentile 0.0,
and `tokenwise-kld.npy` sha256
**`8be5dccaf885d7dadca697c4d54cff60d1c8c8333b57761c31d882c9f9ec9e5d`** — the
published M1 floor digest, byte for byte, through a matmul that ran on a
different processor. `bin/selftest_replay_device.py` (T12) holds that line
offline as a gate.

`--replay-dtype float64` accumulates the replay in fp64 instead. It is more
accurate AND more reproducible across backends (fp64 reduction-order differences
are ~1e-16 relative rather than ~1e-6), but it is a DIFFERENT measurement from
either fp32 path and is offered as one, not as a better spelling of the same
number.

### 3.4 What this does to M2 and M3

Per-window comparison cost scales as `positions x vocab x hidden`. Against the
Qwen3.8 measurement (2,047 positions, vocab 248,320, hidden 5,120 = 1.0):

| rung | vocab | hidden | per-window vs Qwen3.8 | 512-window compare, cuda | 512-window compare, numpy |
|---|---:|---:|---:|---:|---:|
| M1 Qwen3.8-27B | 248,320 | 5,120 | 1.00 | **173 s** (measured) | **1,755 s** (measured) |
| M2 GLM-5.3-Flash | 154,880 | 4,096 | 0.499 | ~86 s | ~876 s |
| M3 GLM-5.3 / GLM-5.2 | 154,880* | 6,144 | 0.748 | ~130 s | ~1,313 s |

\* GLM-5.3-Flash's vocabulary is 154,880 (`hidden_replay.py::EXPECTED_VOCAB`,
and the `--vocab-chunk 9680` divisor everything in this repo uses). The 78L/6144
GLM-5.3/GLM-5.2 row ASSUMES the same vocabulary and has not been checked against
those configs — do that before quoting the M3 line, per M1 learning 16.

**The consequence for the ladder is that the comparison term stops mattering.**
At four candidates on a 512-window Flash panel the comparison budget goes from
**~58 min** of wall clock (4 x 14.6 min) to **~6 min** of GPU time — from the
dominant line item to less than the fetch. What is left to budget is the
**capture**, which for a 642.7 GB Flash root is streaming-regime and
fetch-overlapped (~17 min of fetch at 627 MB/s), and the **number of
candidates**, which is now bounded by what the engine can load (M1 learning 8)
rather than by what the comparisons cost.

**A caution that survives the speedup.** A rung that adopts `--replay-device
cuda` has chosen a replay backend for its whole comparability group (§3.3). If a
later candidate has to be compared on a machine with no GPU, it must either use
the same backend or start a new group. Decide once, at the top of the rung, and
record it — `comparator.replay_backend` is in every receipt precisely so this is
checkable after the fact.

**Anchors already paid for:** the Flash 4-rung ladder at $8.02 / ~$11.60 / $6.65
/ $5.41 per artifact; Fruit root+candidate at $0.25 total; the 0.1B fixture
end-to-end at $0.00; **M1 Qwen3.8-27B root (3 cold runs) + 2 candidates + 3
comparisons + publish at $5.12 total**; **M1.5 the replay fix, measured against
the published root on one on-demand RTX PRO 6000, at $1.27.**

### 3.5 Two operational notes the M1.5 run paid for

**A published fidelity dataset does not survive `snapshot_download` unedited.**
Fetching `malaiwah/qwen38-27b-fidelity-root-v1` into a `local_dir` adds
`.cache/huggingface/**` (1,550 files) and a `.gitattributes` the Hub inserts;
neither is in `checksums.txt`, and the SEAL-1(c) unlisted-file gate refuses the
whole comparison. Delete both after fetching, or pass `--allow-partial` and
accept `covers_full_panel: false`. Cost this run two wasted job launches.

**Captures and comparisons no longer trivially pack.** M1 learning 4 said
capture (GPU-bound) and comparison (CPU-bound) could run concurrently for free.
With `--replay-device cuda` the comparison is GPU-bound too, so they now
contend. Pack a comparison against a *fetch*, not against a capture — or leave
the comparison on the numpy path when a capture is running, which is also the
right choice for reproducing an existing group.

## 4. Families, ranked by value per dollar

| Family | Reference | Size | Geometry | Quant children | Note |
|---|---|---:|---|---:|---|
| **GLM-5.2** | `zai-org/GLM-5.2` | 1,506.7 GB | 78L/6144/256e | **100** | natively unquantized; no `-BF16` sibling |
| **GLM-5.3** | `zai-org/GLM-5.3-BF16` | 1,506.7 GB | 78L/6144/256e | 37 | same geometry as 5.2 — one engine serves both |
| **GLM-5.3-Flash** | `-Flash-BF16` | 642.7 GB | 45L/4096/288e | 67 | **re-capture**: a NEW group with a 0.0 floor beside the existing eight rows, which it does not upgrade |
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
rows. **Say that on the M2 plan, not just here**: the eight Flash rows keep their
0.011506 inferred floor forever unless each of them is re-measured against the
new root, and that re-measurement is a cost line (eight comparisons, ~86 s each
on the cuda replay path — which is exactly the kind of line the old cost model
would have made prohibitive and no longer does).

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
   entirely, is the one that cost. *(**M1.5 DONE** 2026-08-30, $1.27: the
   comparison term is now 10.13x smaller and §3 is rewritten around what was
   measured rather than what was projected.)*

   Two things M2 must decide before it starts, both new since M1.5: **which
   replay backend the whole group uses** (§3.3 — it is part of the number), and
   whether the GLM-5.3 MTP layer's `--allow-unexpected-tensors` blocking
   disclosure is acceptable on the sealed root (it is required now; the capture
   refuses otherwise).
4. **GLM-5.2 and GLM-5.3** on the proven engine.
5. Qwen3.5-397B if the backfill still looks worth it.

## 6. What would change this plan

- Layer-outer failing bit-identity on `glm_moe_dsa` — then the schedule is wrong
  and nothing downstream is safe.
- Pipeline fill/drain costing more than the parallelism returns at 78 layers.
- Fetch rate materially below 430-600 MB/s, which would make every large capture
  fetch-bound and shift the answer back toward one device.
- Spot preemption on multi-hour runs; measured behaviour, not assumed.
