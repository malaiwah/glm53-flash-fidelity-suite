# R10 closure delta check — our reconstruction vs the sealed core

**Verdict: bit-identical. 120/120 encodes, 624 MiB of packed trellis compared, 0 differing bytes.**

This is the comparison we promised in
[glm-5.3-flash-exl3-4bpw issue #1](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/issues/1),
run against the sealed R10 closure published at commit
`57db68f1db0d1fefe1dcd2b9350d3f6968c786f0`, path `reproducibility/r10/`.

Machine-readable receipt: [`closure-comparison.json`](closure-comparison.json).

## 1. The upstream bundle verifies

`reproducibility/r10/verify_bundle.py` run unmodified:

```json
{"file_count": 41, "numeric_core_sha256": "e9a85a47…c75032", "ok": true,
 "r10_codec_sha256": "8b31fb8d…c2a1eded"}
```

Both key identities were also reproduced independently with `sha256sum`:

| file | sha256 |
| --- | --- |
| `r7_encoder/r10_codec.py` | `8b31fb8d1214df63fa1557175a926f6d2d680d69d2cb3689d1df4b5c62a1eded` |
| `lineage/encode_tr3_v31.py` | `e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032` |

No mismatch, nothing to report upstream.

## 2. What was compared

Both codecs were driven through the *same* adapter
(`quant_pipeline.codecs.exl3_mcg.Exl3MCGCodec`), on the same H200, through the
same compiled `exllamav3_ext`, at `sigma_reg=0.025` — in separate processes,
since the two `r7_encoder` packages cannot coexist in `sys.modules`.

Inputs are real campaign artifacts, not synthetic:

- **weights** — GLM-5.3-Flash BF16 source, unmodified;
- **suh/svh** — the real per-expert normalization vectors read out of the K6
  campaign's own layer preparations, i.e. exactly what our encoder consumed;
- **covariance** — real production Hessians.

**Sample: 24 expert-projection matrices × 5 rates = 120 encodes.**
Layers 3 / 14 / 25 / 36 / 45, experts 0 / 37 / 113 / 201, all three projections
(gate / up / down), both shapes ([2048, 4096] and [4096, 2048]).

- *Tier A — production-exact* (12 matrices): layer 45, real per-expert
  production Hessians.
- *Tier B — layer coverage* (12 matrices): layers 3/14/25/36 with real weights
  and real suh/svh; the streaming campaign only retained per-expert Hessians for
  layer 45, so a shape-matched production Hessian stands in. Both codecs got
  bit-identical input, so the head-to-head is exactly fair; only the "this is
  literally a production encode" claim is weaker for those 12.

Input identity was asserted per case (weight, covariance, suh, svh SHA-256 all
equal across the two runs) before any comparison was scored.

## 3. Result on the admitted domain (K3 / K4 / K5)

| rate | encodes | packed byte-identical | differing bytes | decoded rel. Frobenius err (mean) |
| --- | --- | --- | --- | --- |
| K3 | 24 | 24 / 24 | 0 | 0.1709 |
| K4 | 24 | 24 / 24 | 0 | 0.0866 |
| K5 | 24 | 24 / 24 | 0 | 0.0440 |

**72/72 byte-identical — 0 differing bytes out of 301,989,888.** The decoded
reconstruction SHA-256, the stored `suh`, and the stored `svh` are identical in
all 72 cases. `mcg` = `0xCBAC1FED`, `codebook_scale` = `1.24371088`,
`sigma_reg` = `0.025` on both sides.

## 4. The K6/K8 finding — his sealed core refuses those rates

This is the substantive discovery, and it changes our disclosure language.

The sealed closure gates bit width in **three** places:

- `r7_encoder/r10_codec.py:392` — `raise ValueError("Round 10 accepts only 3, 4, or 5 bits")`
- `r7_encoder/trellis.py:337` — `"Round 7 production codec accepts only 3, 4, or 5 bits"`
- `r7_encoder/constants.py:31` — `ALLOWED_BITS = (3, 4, 5)`, enforced again in `EncodedTensor.__post_init__`

All 48 K6/K8 encode attempts through his public API raised
`ValueError: Round 10 accepts only 3, 4, or 5 bits`.

So the question "would his pipeline have produced our K6/K8?" has a clean
answer: **as published, his pipeline does not emit K6 or K8 at all.** Our K6/K8
were never a substitution for something his code would have done — they are a
declared *rate extension*, which is what our adapter's own admission note says.

But the gates are policy, not numerics. Nothing in either implementation
branches on K: his `_quant_args` and ours both pass `K` straight through to the
exllamav3 kernels, which are instantiated for K=1..8. So we tested it.

Driving **his sealed bound methods and his sealed numeric core** at K6/K8 —
re-executing his `encode_group` body verbatim, skipping only the two
non-numeric admission checks, modifying **no sealed byte** (hashes re-verified
after the run, `sealed_unmodified: true`):

| rate | encodes | packed byte-identical | differing bytes | decoded rel. Frobenius err (mean) |
| --- | --- | --- | --- | --- |
| K6 | 24 | 24 / 24 | 0 | 0.02263 |
| K8 | 24 | 24 / 24 | 0 | 0.00722 |

**48/48 byte-identical — 0 differing bytes out of 352,321,536.**

His numeric machinery, run at K6/K8, produces exactly the bytes our
reconstruction produces. The only thing standing between his sealed core and
our K6/K8 artifacts is an admission constant.

## 5. Fidelity impact

Zero — and that is meant literally, not as a rounding.

- packed-byte difference: **0** of 654,311,424 bytes compared
- decoded-weight error delta between codecs: **max |Δ| = 0.0** across all 120 encodes
- reconstruction SHA-256 identical: 120/120

Because the substitution changes no encoded bit, it cannot change any logit, so
its KLD impact is identically zero and no re-scoring is warranted. For scale:
the quantization error the two codecs jointly make is ~2.3e-02 relative
Frobenius at K6 and ~7.2e-03 at K8 — the inter-codec difference is 0 against
that.

## 6. Provenance — the code compared is the code that shipped

The staged reconstruction's closure hashes reproduce, byte for byte, the
`python_closure_sha256` block recorded in the K6 campaign's own
`out-k6/preparation/layer-NNN/preparation.json`:

| file | sha256 |
| --- | --- |
| `__init__.py` | `fed78f2aac5938b9a5c3bbb0d8ae6febca7e5c0202d9c2a891213c2d1038e415` |
| `encode_tr3_fallback.py` | `beb14a7a7ebc26bb8ceb78585cfb0628f40c8182f602633debeff50247acfe6d` |
| `r10_codec.py` | `a3ea18b25011210b7a8bcaac1ecd99086bb1d0567c875c09948273c20f93c8d8` |
| `trellis.py` | `2216a3709b2b361d50e78a730d82f0deb0f2fb3a35d9e98e40e1c1900fb391cd` |

So this is not a comparison of some cleaned-up variant — it is the same code
that encoded K6.

## 7. Caveats, stated plainly

1. **Only difference found, anywhere: a diagnostic field.** His fast R10 path
   hardcodes `proxy_loss = 0.0` and records
   `covariance_proxy_loss_evaluated: False`; ours computes a real value
   (e.g. `9.5e-04`). It is metadata — not part of the packed payload, the
   suh/svh vectors, or the decoded weights, all of which are identical.
2. **Extension build hash.** The `exllamav3_ext` binary present today
   (`7d5fec66…`) differs from those recorded during the K6 (`5a968c10…`) and K8
   (`e0201c10…`) encodes. All three are builds of the same pinned source,
   exllamav3 @ `c5d9c657` (v0.0.43), clean tree — the `.so` hashes differ
   through build non-determinism, not source drift. Both codecs here shared one
   binary, so the comparison is fair. This experiment compares the Python codec
   layer; it did not re-encode the shipped artifacts.
3. **Tier B covariance** is a shape-matched stand-in (see §2). Fairness is
   unaffected; production-exactness applies to the 12 tier-A matrices.
4. **The K6/K8 comparison bypasses an admission gate.** It demonstrates what his
   numeric core computes at those rates. It is not a claim that his shipped
   pipeline emits K6/K8 — as published, it does not.

## 8. What this means for the K6/K8 disclosure

The old language — "unverified substitution, identity hashes will not match" —
was appropriately cautious and is now superseded on the numeric question. Our
model cards can say:

> The R10 codec used to encode this model is a disclosed reconstruction, not
> Brandon M. Music's sealed implementation, and its file hashes differ from his.
> Against the sealed R10 closure published at
> `brandonmmusic-max/glm-5.3-flash-exl3-4bpw@57db68f1`, the reconstruction was
> verified to produce **byte-identical packed trellis output** on 120 encodes
> spanning 24 expert-projection matrices, five layers, all three projections and
> rates K3/K4/K5/K6/K8 — 624 MiB of packed bytes, zero differing. On K3/K4/K5
> this was through his public API; on K6/K8, which his published core refuses by
> an admission constant (`ALLOWED_BITS = (3, 4, 5)`), it was through his sealed
> numeric primitives directly. The substitution changes no encoded bit, so its
> fidelity impact is zero.

The honest framing of K6/K8 is **declared rate extension of a verified-equivalent
codec** — not "what his pipeline would have produced", because his published
pipeline produces nothing at those rates.

---

*Method, per-case numbers and all hashes: [`closure-comparison.json`](closure-comparison.json)
(schema `malaiwah.r10-closure-comparison.v2`).*
