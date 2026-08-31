# Fallback R10 codec — validation plan and executed results

Environment for all executed steps: JarvisLabs machine 484453 (L4, SM89),
venv `/home/ubuntu/k6prep/venv-b` (Python 3.12.13, torch 2.11.0+cu130,
exllamav3 @ c5d9c657 with the JIT extension cached at
`~/.cache/torch_extensions/py312_cu130/exllamav3_ext/exllamav3_ext.so`).
Remote copy of this directory: `/home/ubuntu/k6prep/fallback-val/fallback-val/`.
Receipts: `receipts/selftest-l4.json`, `receipts/native-probe-l4.json`
(verbatim stdout of the two commands below, 2026-08-27).

## Plan → status

| # | Probe | Command | Status (L4, 2026-08-27) |
|---|---|---|---|
| V1 | Module import, embedded-source parse, staging, bit gating (no GPU) | local `python3` import probe + `py_compile` | PASS (macOS py3.9 compile+import; runtime target py3.12) |
| V2 | Ext op availability + binding mode | inline `python -c` on L4 | PASS — `quantize_tiles/pack_trellis/unpack_trellis/reconstruct` present; JIT ext NOT in `sys.modules["exllamav3_ext"]` (motivates verify-only extension seal) |
| V3 | K3/4/5/6/8 encode/decode roundtrip, trellis word counts, stored bytes, built-in oracles, NMSE monotonicity | `python r10_codec_reconstructed.py --selftest --device cuda:0 --pipeline-src .../pipeline-src` | PASS (EXIT=0) |
| V4 | **K8 probe (K6K8 gate)** — 128-word trellis | same selftest | PASS — `trellis_words_per_tile[8] == 128`, oracles held, NMSE(K8) < NMSE(K6) |
| V5 | Repeat determinism across fresh codec instances | same selftest | PASS — identical `packed_sha256` + `reconstruction_sha256` for all rates |
| V6 | encode_group lockstep == serial, K6 and K8, 4-matrix groups | same selftest | PASS — byte-identical hashes |
| V7 | Factor-cache reuse for shared (hessian, suh, σ) | same selftest | PASS — exactly one hit for the gate/up pair |
| V8 | Full adapter contract (glm-5.3-flash-exl3-4bpw `Exl3MCGCodec`): closure hashing, `backend_class`, environment keys, stored_bytes, K6 admission, disclosure block present | same selftest with `--pipeline-src` | PASS — bits (3,4,5,6) admitted and encoded |
| V9 | Bit-exactness vs exllamav3 NATIVE convert (`quantize_exl3`, mcg=True), one 256×256 matrix, K6 and K8 | `python probe_native_convert.py` | PASS (EXIT=0), see numbers below |
| V10 | H200/SM90 rerun of V3–V9 + full-size (k=6144) memory behavior + prepared-backend end-to-end with `r7_encoder.hessian` | Phase P0 rehearsal | NOT RUN (needs the paid H200; do during fixture rehearsal if fallback is activated) |

## Key numbers (L4)

Selftest (`receipts/selftest-l4.json`):

- `codebook_scale` = 1.24371088 (loaded through the sealed shim)
- NMSE by K on the synthetic 256×256 tensor:
  K3 3.59e-2 > K4 1.04e-2 > K5 3.48e-3 > K6 1.68e-3 > K8 1.08e-3
- proxy_loss by K: K3 2.91e-2 … K6 1.35e-3 … K8 8.64e-4
- staged closure (adapter `python_closure_sha256`): `__init__.py`
  fed78f2a…, `r10_codec.py` c096b179…, `trellis.py` 2216a370… — these hashes
  are what a published `codec_identity` would carry in fallback mode (they
  intentionally do NOT match any sealed closure).

Native-convert probe (`receipts/native-probe-l4.json`):

- P1 repack byte-identity vs native packed trellis: TRUE for K6 and K8
- P2 decode determinism: TRUE
- P3 original-domain decode vs native `weight_q`: max_rel 7.7e-4 (K6),
  6.6e-4 (K8) — at fp16-vector-rounding scale, the documented encode/serve
  boundary (native reconstructs with pre-rounding fp32 su/sv; stored
  checkpoints and this codec use the fp16-stored vectors). This gap is a
  property of native convert, not of the fallback.
- native convert marked the tensors with mcg = 0xCBAC1FED (asserted).

## Explicit non-claims

- No bit-identity with Brandon's sealed `encode_tr3_v31.py` /
  `r10_codec.py` is claimed or measurable until he publishes them
  (RECONSTRUCTION.md §4).
- V10 items remain open; in particular the 6144×6144 fp64 proxy-loss einsum
  and the 151 MiB factor-cache entries were not exercised at GLM scale on
  the L4.

## Re-run instructions

```sh
# from k6-program/fallback/ with the pipeline subset staged as in
# /home/ubuntu/k6prep/fallback-val/fallback-val/
export PATH=/home/ubuntu/k6prep/venv-b/bin:$PATH
python r10_codec_reconstructed.py --selftest --device cuda:0 \
  --pipeline-src $PWD/pipeline-src --k 256 --n 256
python probe_native_convert.py
```
