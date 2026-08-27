# R10TrellisCodec fallback reconstruction — inference catalogue & disclosure rules

Status: **DRAFT / DORMANT**. Activate ONLY if Brandon does not publish
`r7_encoder/r10_codec.py` and `encode_tr3_v31.py` (asked in
glm-5.3-flash-exl3-4bpw issue #1; RUNBOOK Phase G0 item 1 is abort-level on
their absence). If the sealed files land, DELETE this fallback from the run
tree and use his files — never mix the two closures in one campaign.

Deliverables in this directory:

- `r10_codec_reconstructed.py` — the codec (becomes `r7_encoder/r10_codec.py`
  when staged), plus the numeric-core shim source, the package stager, and a
  `--selftest` entry point.
- `probe_native_convert.py` — bit-exactness probe against exllamav3's native
  convert path (see VALIDATION.md).
- `VALIDATION.md` — validation plan + executed L4 results.
- `receipts/` — machine outputs of the executed validation.

## 1. What is missing, what replaces it

| Sealed (absent) | Fallback replacement |
|---|---|
| `r7_encoder/r10_codec.py` (class `R10TrellisCodec`) | `r10_codec_reconstructed.py` staged as `r7_encoder/r10_codec.py` |
| `encode_tr3_v31.py` (numeric core) | `encode_tr3_fallback.py` — generated shim re-exporting the pinned exllamav3 `exl3_lib.quantize` primitives (`write_numeric_core_shim`) |
| `r7_encoder/trellis.py` (sealed descendant of `bmmlaw_r7_encoder/trellis.py`) | generated re-export shim (`_TRELLIS_REEXPORT`) exposing `CodecConfig` |
| `r7_encoder/hessian.py` | copied VERBATIM from `bmmlaw_r7_encoder/hessian.py` @ `bf37b066` (same symbols the prepared backend imports: `FullCovarianceAccumulator`, `down_inputs_from_roundtrip`) |
| `r7_encoder/__init__.py`, `constants.py`, `types.py`, `determinism.py` | `__init__.py` generated (provenance marker). constants/types/determinism are NOT staged: nothing in the driver trees imports them from `r7_encoder`, and the ancestor `types.EncodedTensor` would reject K6/K8 (`ALLOWED_BITS=(3,4,5)`). |

Numeric heart: all trellis math (Viterbi tile quantization, pack/unpack,
reconstruct) is executed by the pinned exllamav3 @ `c5d9c657` CUDA extension
with `mcg=True` (multiplier `0xCBAC1FED`), the same kernels the sealed core
requires. The kernels are instantiated for K=1..8, so K6 and K8 (the
128-word-per-tile trellis) need no kernel work.

## 2. Receipt inventory (surface elements with a named public source)

| Element | Receipt |
|---|---|
| `CodecConfig(device, sigma_reg, numeric_core, numeric_core_sha256, extension, extension_sha256, verify_files)` | adapter `Exl3MCGCodec._codec()` constructs exactly these kwargs |
| `R10TrellisCodec(config)` single-arg ctor | adapter `_codec()` |
| `encode_bits(tensor_id=, weight_hf=, covariance=, bits=tuple, suh=, svh=, sigma_reg=, provenance=)` → `{bit: unit}` | adapter `encode_candidates` |
| unit fields `.trellis .suh .svh .reconstructed_kn .packed_sha256 .reconstruction_sha256 .provenance` | adapter + `glm53_prepared_backend._projection` + `run_qwen_fast_encode` |
| result keys preserve request bit order | `tests/test_exl3_mcg.py`: `assert tuple(result) == (3, 4, 5)` |
| `reconstructed_kn` is [K,N]; adapter transposes to HF [N,K] | adapter `.T.contiguous()`; ancestor `EncodedTensor.reconstructed_kn` |
| `encode_group(list_of_request_mappings)` → order-preserving list of `{bit: unit}` | `glm53_prepared_backend._encode_batch` (K6 production hot path), `run_qwen_fast_encode`, `encode_qwen_attention_k4` |
| group encoder "prepares/factorizes each matrix once and locksteps all equal-bit LDLQ walks" | `run_qwen_fast_encode.py` comment (L232) |
| `.core` module attr with v31 primitives | `glm53_mcg_preparation` (`backend.core` → `StreamingLayerFitter`), `streaming_v31._required_core` |
| core surface: `block_ldl ldlq pack_trellis blockwise_preapply_had_{l,r}_ preapply_had_{l,r} _lazy_torch CODEBOOK_SCALE` | ancestor `load_numeric_core` required list — exactly exllamav3 `exl3_lib/quantize.py`'s public names plus two v31 extras |
| core `block_rms` | `streaming_v31._required_core` + 3 call sites |
| core `g_scale_gss(target, quant_args) -> (scale, objective)`, 13 evaluations, width 3 | `qwen_services.CorrectedPinnedGSSProducer`; exllamav3's implementation performs exactly 13 evals over [0.1,1.9] @ tol 0.01, width 3 |
| `.codebook_scale` float property | `candidates/ledger.py:2647`, `glm53_mcg_preparation` |
| `._quant_args(bits, sigma_reg)` | `CorrectedPinnedGSSProducer` |
| quant_args literal `{"K", "devices", "mcg": True, "sigma_reg", "buf_size_k": 128}` | ancestor `_factor_covariance` |
| `.cache_stats` (dict) and `.clear_caches()` | `glm53_prepared_backend._encode_batch` ("fresh factor domain" between gate/up and down phases) |
| fp16 vector boundary (validate & encode with fp16-rounded suh/svh) | ancestor `_validate_vectors`; prepared-backend receipt string `fp16_vector_boundary` |
| pack/unpack + reconstruct-vs-LDLQ + repeat-determinism + byte-count oracles | ancestor `encode` |
| `reconstruction_sha256` hashes the FP16 cast of the fp32 reconstruction | ancestor; `run_qwen_fast_encode` "historical FP16 reconstruction identity" |
| `stored_bytes` = trellis + fp16 suh + fp16 svh bytes | adapter formula + `run_qwen_fast_encode` |
| K,N divisible by 128 | adapter `_parse_unit`; ancestor constants (`HAD_K=HAD_N=128`) |
| CUDA-only LDL, sigma-damped Cholesky retries, policy string | ancestor `cuda_only_block_ldl` + `LDL_FACTORIZATION_POLICY` |
| σ_reg default 0.025; MCG 0xCBAC1FED | ancestor constants; adapter default; contract JSON |
| returned tensors live on the encode device | `run_qwen_fast_encode` subtracts `candidate.reconstructed_kn` (bf16) from a GPU-resident source and calls `candidate.trellis.detach().cpu()`; deviation from the CPU-resident ancestor `EncodedTensor` |
| generic tensor ids (no GLM constants) | adapter `GenericTensorId` docstring ("accepted by the pinned codec without GLM constants") |

## 3. Inference catalogue (everything beyond receipts)

1. **encode_bits factorizes once, then runs one LDLQ walk per rate.** Backed
   by the fast-encode comment; the exact sealed loop structure is unknown.
   Consequence-free for numerics: `block_ldl` does not depend on K.
2. **`verify_files` semantics.** True ⇒ SHA-256s are mandatory and enforced
   at load; False ⇒ hash-free loading for local probes. Pure inference from
   the name; the adapter always passes True.
3. **`EncodedUnit` admits K=1..8** (kernel instantiation range). The
   ancestor's `EncodedTensor` restricts to (3,4,5); the sealed R10 record
   type must already be wider for the K6-supporting adapter to work. K-range
   admission POLICY remains at the adapter layer (3–5 shapleymcg / 3–6
   glm-5.3-flash-exl3-4bpw; K8 needs a disclosed adapter-admission patch —
   DECISIONS item 6).
4. **Extension binding is verify-only, not load-by-path.** The ancestor
   imports the sealed `.so` under the module name `exllamav3_ext` after
   checking sys.modules for an incumbent. The fallback numeric core is a shim
   over the installed exllamav3 package, whose JIT extension is NOT
   registered in `sys.modules["exllamav3_ext"]` (verified on the L4: it loads
   as `exllamav3.ext.exllamav3_ext` from `~/.cache/torch_extensions/...`).
   Loading the same binary a second time would double-register pybind ops, so
   the fallback hash-verifies the binary the core actually resolved
   (`core._lazy_torch()[1].__file__`) against the sealed path+hash and fails
   closed on mismatch. Functionally equivalent seal; different mechanism.
5. **Numeric core = exllamav3 shim.** The v31 required-name list is exactly
   exllamav3's `exl3_lib/quantize.py` surface + `_lazy_torch` +
   `CODEBOOK_SCALE`; the GSS receipts (13 evaluations, width 3, algorithm
   name `encode_tr3_v31.g_scale_gss`) match exllamav3's implementation with
   reordered arguments. `CODEBOOK_SCALE = 1.24371088` is exllamav3's
   constant; that his v31 exposes the same VALUE is highly likely
   (checkpoints interoperate with exllamav3 serving) but UNVERIFIED.
6. **Factor cache internals.** Content-addressed: key = (covariance-fp32
   bytes SHA, stored-fp16 suh bytes SHA, sigma_reg). LRU, default capacity 4
   (`R10_FALLBACK_FACTOR_CACHE`). Rationale: gate and up requests of one
   expert share one `gate_up_hessian` object, and the K6 preparation's
   gate/up `suh` is layer-shared, so the 6144² Cholesky is reused whenever
   the triple repeats — presumably why the sealed codec has caches at all.
   `cache_stats` key NAMES (`factor_hits/factor_misses/factor_evictions/
   cache_clears`) are invented; the prepared backend only snapshots the dict
   into a receipt, so mismatched names change receipt cosmetics, not
   behavior. Correctness does not depend on hits: a miss recomputes the same
   deterministic factor.
7. **Lockstep implementation.** Only the `quantize_tiles` kernel launches are
   batched across matrices; all per-matrix linear algebra is per-matrix and
   the loop is an exact transliteration of the pinned
   `exl3_lib.quantize.ldlq`. Per-tile Viterbi is stateless across tiles, so
   lockstep == serial bit-exactly (asserted in the self-test for K6 and K8;
   re-checkable per group with `R10_FALLBACK_VERIFY_LOCKSTEP=1`; force serial
   with `R10_FALLBACK_LOCKSTEP=0`). The sealed lockstep's throughput profile
   is unknown; ours only claims numeric equality with the serial reference.
8. **Provenance metadata keys** mirror the ancestor
   (`numeric_core full_k full_n mcg codebook_scale sigma_reg
   factorization_policy extension_sha256 extension_repeat_oracle
   covariance_sha256`) plus the mandatory `fallback_reconstruction` block.
   The sealed R10's exact metadata keys are unknown.
9. **`proxy_loss`** keeps the ancestor's fp64 H-weighted relative error.
   Nothing in the driver trees reads it; retained for parity and receipts.
10. **CPU devices are out of scope**: `_quant_args` requires a CUDA ordinal
    and the LDL is CUDA-pinned (matches the sealed recipe; the adapter's
    "cpu" tests use a fake codec tree, not this class).
11. **`hessian.py` verbatim-ancestor assumption.** The prepared backend's
    calls type-check against the bmmlaw ancestor exactly
    (`FullCovarianceAccumulator(INTERMEDIATE_SIZE, device=, guided=True)`,
    `finalize(sigma_reg, add_damping=False)` → `.matrix/.rows/.weight_sum`,
    `down_inputs_from_roundtrip(hidden, gate_rt, up_rt)`). Any sealed drift
    in that file is invisible to us.

## 4. What is NOT claimed

- **No bit-identity with Brandon's sealed core.** His `encode_tr3_v31.py`
  may contain corrections beyond exllamav3's quantize.py (the "corrected
  R10" language implies at least orchestration-level fixes; which of them
  live in the numeric core vs in r10_codec.py is unknowable from public
  receipts). Any KLD numbers produced through this fallback are OUR numbers,
  comparable to his only as a disclosed reproduction, not as his pipeline's
  output.
- The `codec_identity` hashes (`python_closure_sha256`, `numeric_core_sha256`,
  `extension_sha256`) will not match any sealed value he may later publish.
- The sealed panel scoring remains valid (it consumes checkpoints, not the
  encoder), but the "same sealed encoder" equivalence claim is unavailable.

## 5. Mandatory receipt disclosure when activated

1. Every encode candidate carries
   `provenance.fallback_reconstruction = {schema, reconstructed_from,
   sealed_core_files_absent, bit_identity_with_sealed_core: "UNVERIFIED"}` —
   emitted automatically by `_finalize_unit`; do not strip it.
2. `stage_k6.sh` must, in fallback mode: (a) record the staged closure
   `{relpath: sha256}` map (from `stage_r7_encoder`) plus the shim and
   extension hashes in the campaign receipts; (b) copy this RECONSTRUCTION.md
   and VALIDATION.md into the published receipts tree; (c) put a
   "reconstructed fallback codec — not the sealed R10 closure" paragraph in
   the model card / README next to the H200-vs-B200 deviation disclosure.
3. The publication gate must refuse to publish if
   `fallback_reconstruction` is present in candidate metadata but the README
   disclosure paragraph is absent.
4. K6K8 additionally requires the disclosed adapter-admission patch for K8
   (adapter L175); record it in the same patch series as the H200 worker
   deviations.
5. If Brandon publishes the sealed files mid-campaign: stop, do NOT merge
   trees; restart encoding from the sealed closure or finish and publish
   with the fallback disclosure — never a mixture.

## Amendment (adversarial re-review, 2026-08-27): inference 12 — reader-exact closure

The first end-to-end `k6_driver.py rehearse` run on the L4 (through the real
adapter + staged fallback + pinned reader) produced `k6_roundtrip_exact:
false`: the original-domain reconstruction computed through the extension's
fp16 `reconstruct` kernel differs from the reader's independent
`decode_choice_hf` fp32 torch decode at fp16-rounding scale.  Upstream's
receipts (`offline_reader_exact_decode_checked`, the reader-ABI
`exact_reconstruction_checked`) prove Brandon's sealed codec produces
reader-BIT-EXACT closures, so the fallback was wrong in kind, not just in
bits.  **Inference 12**: the sealed codec's original-domain reconstruction
closure is (or is equivalent to) the reader's own decode math.  Fix:
`decode_to_original` now computes the closure WITH
`glm53_packed_k4_reader.decode_choice_hf` (CPU fp32) whenever the pipeline is
importable AND the reader admits the rate (`SUPPORTED_BITS`, checked
dynamically — K8 flips to reader-exact automatically once the K6K8 reader
extension lands) — exactness by construction (re-validated on the L4:
`k6_roundtrip_exact: true`); the extension-based path survives only for
pipeline-less standalone runs and reader-unsupported rates (K3/K5 oracles,
K8 today), recorded per-candidate in provenance as
`original_domain_decode`.  The staged
package also now ships the numeric-core shim as
`r7_encoder/encode_tr3_fallback.py` (the file `CodecConfig(numeric_core=...)`
loads; upstream's is `encode_tr3_v31.py` — drivers must never pass
`r10_codec.py` as the numeric core, its module surface fails the
required-attribute check).
