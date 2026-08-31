# The MLX surface — scoring community Apple-silicon quants on the sealed yardstick

`mlx_surface.py` + `stream_score.py --source mlx` score an **MLX
affine-quantized** community conversion of GLM-5.3-Flash on OUR sealed 25-window
panel, with the SAME streaming teacher-forced capture, the SAME fp64 estimator
and the SAME EP8/fp32 lane as K6/K8/Dione/native-BF16. The measured function and
the receipt shape do not change; what changes is where the weights come from and
what the receipt must disclose about them.

Primary target: **orcarouter/GLM-5.3-Flash-MLX** (HF tensor names, per-expert
`weight`/`scales`/`biases` triplets). The repo root is the 4-bit build; its
`2-bit/`, `3-bit/`, `4-bit/`, `6-bit/` subdirectories are separate artifacts with
their own config/index — each is measured as its own row.

## 1. The scope finding — this format quantizes past the routed experts

Every other surface in this suite quantizes routed experts only, so the
non-routed model comes from the official BF16 tree. Measured from orcarouter's
own index and shard headers (revision `c80f6810`, 113,446 stored tensors), the
MLX artifact is **not** shaped that way:

| class | modules | bits |
|---|---|---|
| routed experts, layers 3–44 | 36,288 | 4-bit gate/up, 5-bit down |
| MTP layer-45 experts (never executed) | 864 | same mix |
| shared experts, 43 layers | 129 | 6-bit |
| dense MLPs, layers 0–2 | 9 | 4-bit gate/up, 5-bit down |
| DSA attention (`q_a`,`q_b`,`kv_a_mqa`,`o`), 12 layers | 48 | 4-bit |
| **everything else** (embed_tokens, lm_head, vision, KDA projections, `kv_b_proj`, indexer, norms, gates) | **1,432 tensors, source dtype passthrough** | — |

37,338 quantized modules + 1,432 passthrough tensors = **38,770**, which bijects
the official BF16 tensor set exactly. Two consequences, both mechanical:

* **`--bf16` is not an input of this source.** The non-routed model is built from
  a MATERIALIZED DECODED VIEW of the quant snapshot itself: passthrough tensors
  copied verbatim, quantized non-routed tensors dequantized in fp32 and rounded
  once to bf16, written as real safetensors shards (~19 GB, hash-stamped and
  reused) that the sealed `from_pretrained` constructor loads unchanged — same
  zero-missing / zero-stray load assertions as every other lane. Passing
  `--bf16` anyway only enables an optional byte-identity cross-check of the
  passthrough tensors against the official tree.
* **The receipt carries a measured `scope_policy` block**, censused from the
  index and shard headers rather than asserted from the format family, and
  `registry_add` copies it onto the row as a coded disclosure. A reader must
  never be able to mistake this row for "experts quantized, everything else
  official".

## 2. The decode contract — proven against mlx, not assumed

Packing is a plain little-endian bitstream per output row: element `e` occupies
bits `[e*b, (e+1)*b)` of that row's byte stream, `q` unsigned in `[0, 2^b-1]`.
Dequant is `W[r,c] = q[r,c] * scales[r, c//G] + biases[r, c//G]`, groups along
the input axis. The kernel is plain torch, byte-level unpack, fp32 accumulate —
no `float64`, no `torch.uint32` views, no int64 beyond gather indices, so it
runs on CUDA, MPS and CPU under this suite's device policy, and it is bitwise
identical across them (selftest rung 1; measured MPS==CPU at every bit width on
this Mac). In the streaming path the decode itself runs CPU-side in the reader
pool and only the fp32 result is moved to the device, exactly like the packed
lane's IO/decode split.

Per-tensor `(bits, group_size)` are **DERIVED from shapes** against the official
BF16 shape census (`bits = 32 * packed_cols / in_features`,
`G = in_features / scales_cols`) and cross-checked against `config.json`'s
`quantization` override map. The derivation is authoritative because it is what
the bytes actually say; a disagreement on layers 0–44 is a refusal. (Measured
disclosure: orcarouter stores 291 layer-45 modules at 5/6-bit that its config
override map does not mention at all. Layer 45 is never executed, so this is
recorded, not refused.)

**Cross-check numbers, real ranged-fetched tensors, mlx 0.32.2** — our fp32
dequant rounded ONCE to mlx's own output dtype is BITWISE equal to
`mlx.core.dequantize` on the full tensor:

| repo | module | bits | shape | mlx out | bitwise |
|---|---|---|---|---|---|
| orcarouter `c80f6810` | `layers.3.mlp.experts.0.gate_proj` | 4 | 2048×4096 | f16 | yes |
| orcarouter | `layers.3.mlp.experts.0.down_proj` | 5 | 4096×2048 | f16 | yes |
| orcarouter | `layers.12.mlp.shared_experts.gate_proj` | 6 | 2048×4096 | f16 | yes |
| orcarouter | `layers.0.mlp.down_proj` | 5 | 4096×12288 | f16 | yes |
| orcarouter | `layers.45.mlp.experts.0.down_proj` (MTP) | 5 | 4096×2048 | f16 | yes |
| orcarouter | `layers.3.self_attn.o_proj` | 4 | 4096×16384 | f16 | yes |
| pipenetwork mixed-4_8bit `d43ea8b4` | `embed_tokens` (row slice) | 8 | 64×4096 | **bf16** | yes |

The mixed-4_8bit row is a KERNEL check on a repo whose *dialect* this adapter
refuses to score (see §5): it covers the 8-bit width and the bf16-scale dtype
that orcarouter does not contain. Row slices are legitimate because packing is
per-row.

In fp32 our result and mlx's differ by at most one ulp (mlx fuses the
multiply-add; we do not), which is why the receipts claim bitwise equality **at
mlx's output dtype** and report the fp32 delta rather than asserting it is zero.
Our lane keeps fp32 through the decode and applies the suite's single
fp32→bf16 rounding at expert install — the same install algebra every other
surface gets, and the only deviation from an MLX runtime.

## 3. Provenance — an unsealed source, said out loud

Community checkpoints ship no receipts, no reconstruction closures and no sealed
reader ABI. Identity is therefore: pinned 40-hex repo revision + `config.json`
and index sha256 + the official-BF16 shape-census binding + the adapter's own
sha256, optionally + whole-shard sha256 against an HF-derived manifest
(`verify-shards`, recorded as `full` or `skipped`). Every receipt this adapter
touches carries `seal_disclosure` saying exactly that — same policy as the Dione
lane.

## 4. Running one

```bash
cd k6/tools
PY=/path/to/python                     # torch + safetensors; mlx optional (macOS)

# 0. offline validation (8 rungs, ~8 s, no network, no GPU)
$PY selftest_mlx_offline.py

# 1. metadata only: config + index + every shard HEADER + the HF file manifest
#    (a few hundred KB; NO weight bytes)
$PY mlx_surface.py fetch-meta --repo orcarouter/GLM-5.3-Flash-MLX \
    --revision c80f6810b1a95b5be9042761becc6aa78d189782 --out /tmp/mlx-meta

# 2. census + plan against that metadata (this is the dry-run gate)
$PY mlx_surface.py dry-run --mlx-root /tmp/mlx-meta \
    --repo orcarouter/GLM-5.3-Flash-MLX --revision c80f6810... --skip-shard-hashes

# 3. prove the dequant against mlx on real tensors (macOS; SKIPs elsewhere)
$PY mlx_surface.py crosscheck --mlx-root /tmp/mlx-meta \
    --repo orcarouter/GLM-5.3-Flash-MLX --revision c80f6810...

# 4. with the real snapshot on disk: hash every shard against the manifest
$PY mlx_surface.py verify-shards --mlx-root /data/GLM-5.3-Flash-MLX

# 5. capture (per cold run) — note there is NO --bf16
$PY stream_score.py --source mlx --profile mlx \
    --mlx-root /data/GLM-5.3-Flash-MLX --mlx-repo orcarouter/GLM-5.3-Flash-MLX \
    --mlx-revision c80f6810... \
    --teacher $TEACH --cold-run 1 --out $ROOT/runs/mlx-1 --pipeline-root $PIPE

# 6. aggregate -> malaiwah.glm53-mlx-packed-kld-summary.v1
$PY kld_report.py --profile mlx --teacher $TEACH \
    --runs $ROOT/runs/mlx-1 $ROOT/runs/mlx-2 --out $RCPT/mlx-packed-kld.json

# 7. registry row (the family does not name a lane, so --lane must)
python3 registry/tools/registry_add.py ... --lane streaming --third-party-artifact
```

The first capture materializes the decoded non-routed view under `--work-dir`
(~19 GB, printed as it is written) and reuses it on later runs; a view built
from a different snapshot or a different adapter build is REFUSED, not silently
reused.

`--profile mlx` and `--source mlx` must be used together. The student label is
derived from the artifact (`mlx-affine-b4-gs64-mixed-<hash of the bit
histogram>`), so `kld_report` gates the family prefix on the first run and
then requires that exact label of every other run.

## 5. Named exclusions (refused by name, never skipped silently)

* **The mlx-vlm ("pipenetwork") dialect** — fused `switch_mlp` expert tensors,
  `language_model.model.*` prefixes, `vision_model.*` renames, and no MTP layer
  at all. Refused with the marker that identified it. Its tensors can still be
  used for kernel cross-checks (`crosscheck-raw`).
* **Non-affine MLX modes** (`mxfp4`, `mxfp8`, `nvfp4` quantization modes in
  newer mlx-lm) — refused by declared mode.
* **A census that does not close** — any index tensor that does not fold onto
  exactly one official BF16 name, or any official name with no counterpart, is
  named in the error.
* **A passthrough tensor whose dtype/shape differs from the official tree**, a
  tensor whose bits are not derivable from its shapes, a scales/biases pair with
  no packed weight, a config declaration that disagrees with the stored shapes,
  and a capture without a pinned 40-hex revision.
* **inferencerlabs Q9** (group size 32, BF16 scales, NO `model.safetensors.index.json`)
  is decodable by this kernel but needs a header-glob census; second wave.

**Not wired yet (named, not hidden):** `bin/measure <mlx repo>` still refuses
with "no lane has a reader". `bin/fidelity/hfmeta.py` already normalizes the
codec to `mlx-affine`, but `sniff_surface` has no `mlx` surface and
`bin/engines.json` has no lane entry pointing at `--source mlx`, so the
planner cannot route to this adapter yet. The capture path above is driven
directly, exactly as the Dione lane was before its planner entry existed.

## 6. Validation status (all offline, no GPU, no rental)

`selftest_mlx_offline.py` — 8 rungs, ~8 s:

1. pack/unpack round trip vs an independent reference packer, 18 bits×group-size
   combos, plus the fp32 dequant algebra — and the same kernel run on this
   machine's accelerator is BITWISE identical to CPU (measured on MPS at every
   bit width; the CUDA leg runs the same rung on a box);
2. real-tensor mlx replay from 7 committed fixtures (the table in §2), bitwise —
   this rung runs where mlx cannot be installed;
3. live mlx equality via `mlx.core.quantize`, 36 cells (bits × group size ×
   f16/bf16 scales): unpacked codes EXACTLY mlx's codes, output bitwise equal;
4. the REAL orcarouter census closes and 9 refusals fire by name;
5. stream plumbing: `MlxExpertSource` == manual dequant at real routed geometry;
   the decoded view is exact, reuses itself, and refuses a stale view;
6. `mlx_surface.py dry-run` over the real metadata, with a fetch ledger that
   reconciles EXACTLY with the index's own declared `total_size`
   (203,976,457,080 tensor bytes + 15,619,216 container-header bytes =
   203,992,076,296);
7. `stream_score.py --source mlx --dry-run` (needs a `quant_pipeline` tree;
   SKIPs without one — the wiring is proven statically in rung 6);
8. `registry_add` adapts the summary schema into a row, pins the artifact
   revision, emits the four coded disclosures, and REFUSES a receipt with no
   scope census.
