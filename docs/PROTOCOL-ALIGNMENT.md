# Protocol alignment: brandonmusic's proposed community standard vs ours

**Date:** 2026-08-29
**His standard:** `eval/kld/` in `huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw`,
governed by `kld quantization fidelity report.md`
(sha256 `692ff9e50bc70e716f1a94f1d9a4f3fb2c6d797f639dc8da84b17b069a20b9fc`)
**Our side:** `github.com/malaiwah/glm53-flash-fidelity-suite`, `registry/`, `bin/jointstd/`
**Our frozen protocol:** `registry/protocol/glm53-joint-kld-protocol.v1.json`
file `80df521eb46fba68538dd90aa3f2baf22b1e440b8b560555646ff9bbeb35961b`,
scoring `20ea68c0c730a9d2444148b234a610a5821a50dfa9980e2446c02723317b5e98`

Short version: **his standard is better than what we had on seven of eleven
elements, and we adopted all seven.** Two elements are ours and he has no
equivalent. Two are genuine divergences, both now measured rather than argued
about. One of his own rules broke inside his own campaign and we propose a fix.

Everything below is reproducible offline from this repository:
`bin/selftest_joint_standard.py` (112 cases) and `registry/ make check`
(62 cases + 433 joint-invariant checks).

---

## 1. Element-by-element

Legend: **ADOPTED** — his is better, we took it. **EQUIVALENT** — we already had
it under another name. **DIVERGENT** — we differ, with a measured reason.
**OURS** — we have it, he does not.

| # | Element | His | Ours before | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | Direction and units: KL(teacher‖student), nats | yes | yes | EQUIVALENT | `metric.direction = reference_to_candidate` on all 66 registry rows |
| 2 | Full-vocabulary FP32 teacher logits | yes | yes | EQUIVALENT | same teacher artifact, same `teacher_receipt_sha256 2ae08117…` |
| 3 | FP32+ log-softmax, FP64 accumulation | fp64 both | fp64 both | EQUIVALENT | `estimator.accumulation_dtype = float64` |
| 4 | **Padded lm_head columns masked on both sides** | yes | **no** | **DIVERGENT — measured, §3** | new field `estimator.vocab_masking_policy`; effect ≈ 1e-10 nats |
| 5 | Frozen token ids, published hashes, teacher-forced, bs=1, eager, no MTP/spec/prefix-cache | yes | yes | EQUIVALENT | our panel record pins the same `token_ids_sha256` values |
| 6 | **R0 canary: self-KLD exactly 0.0 AND a one-position shift explodes** | half — the shift half is a synthetic unit test, not a session gate | self-KLD only | **ADOPTED, and extended** — §5.1 | `bin/joint-standard canary`; 5 FIRE cases in the selftest |
| 7 | **≥3 cold runs, report `sigma_run` beside the statistical SE, combine in quadrature** | yes | run means recorded, sigma never reported | **ADOPTED** — §5.4 | `uncertainty.sigma_run`, `.se_total`, invariant JOINT-002/003 |
| 8 | **Window-clustered block bootstrap, BCa, B=5000** | yes | `uncertainty.method = "none"` on all 60 rows | **ADOPTED** — §5.3 | every row now carries a BCa interval; 16/16 of his published endpoints reproduced |
| 9 | **Percentile only with ≥100 exceedances; never compare max across different N** | yes | not enforced | **ADOPTED** — §5.5 | `percentile_guard`, plus a refusal our data needs (§5.5) |
| 10 | **Per-domain and per-position tables** | yes | computed in every kld-report, never published | **ADOPTED** — §5.2 | `by_domain` on 12 rows; his non-uniformity finding reproduces on our data |
| 11 | **Calibration-overlap scan: document hash AND 13-gram** | yes | document/shingle scan on OUR panels, none on his | **ADOPTED** — §4 | our scan reproduces his 25/25 exactly |
| 12 | **One frozen protocol file, hash in every output** | yes — but it broke, §6 | no protocol file at all | **ADOPTED, with a fix** — §6 | two hashes: file + scoring-subset |
| 13 | **Rank by paired differences + McNemar, never by overlapping CIs** | yes | paired per-window t-interval | **ADOPTED** — §5.6 | his 5 published McNemar p-values reproduced; ours upgraded to BCa + sign test |
| 14 | Measured BF16 **floor** and attributable error | no equivalent; §5.3 of his report argues against subtraction | yes | **OURS — divergent, §7** | the floor framing is scope-stable where its inputs are not (+1.4% vs −9% / −16%) |
| 15 | Schema-enforced registry with mechanical refusals | no | yes | **OURS** | 90 invariants, 8 new; CMP-003 caught a real error in this very work (§4.3) |
| 16 | Lane separation (`same_stack` / `cross_stack`) | no field, but his data proves the need | yes | **OURS — and his data confirms it, §8** | his 0.0305 vs 0.0246 for one artifact |
| 17 | Multi-format decode surfaces (TR3/EXL3, dione, MLX, GGUF, NVFP4) | EXL3 + NVFP4 | 5 surfaces | OURS | `k6/tools/stream_score.py --source` |
| 18 | Threshold for the overlap scan justified | bare `0.05` literal, no sensitivity analysis | n/a | **OURS — new, §4.2** | the published means swing 15% across plausible thresholds |

**Where he is plainly ahead and we simply took his design:** rows 6–13. That is
most of the standard. Our contribution to those rows is implementation and
validation, not design, and the protocol file says so.

---

## 2. What we built

New files, all offline, stdlib-only except where a logits tensor forces numpy:

```
bin/jointstd/protocol.py    frozen protocol, two hashes, stamp + refusal
bin/jointstd/canary.py      R0-a self-KLD, R0-b shift explosion, R0-c alignment band
bin/jointstd/ngram.py       n-gram calibration-overlap scanner + threshold sensitivity
bin/jointstd/stats.py       clustered SE, window block bootstrap (percentile+BCa),
                            sigma_run, quadrature, McNemar, percentile guards, per-domain
bin/jointstd/chi2.py        chi-square and normal tails without scipy
bin/jointstd/oracle.py      calls HIS kld_eval when importable (§9)
bin/joint_standard.py       the CLI: protocol | overlap-scan | canary | analyze |
                            paired | mcnemar | stamp
bin/selftest_joint_standard.py    112 cases: known answers, oracle checks, FIRE cases
registry/protocol/…         the frozen protocol, the window selection, per-window inputs
registry/tools/joint_enrich.py       writes the new fields into the seeded rows
registry/tools/registry_joint_check.py   JOINT-001..008, the arithmetic JSON Schema can't
```

---

## 3. Divergence 1 — the padded lm_head columns

GLM-5.3-Flash stores 154,880 lm_head columns for a 154,856-token vocabulary.
His protocol masks the 24 padded columns out of both sides before the
log-softmax. Ours never has: `k6/tools/k6_kld_report.py::_token_kld` takes a
log-softmax over the full last dimension.

We did not argue about the size of this. We downloaded his teacher window
`final-0000` (1,268,157,840 B, sha256 `9f49af1b…`, verified), reconstructed the
4096-dim hidden states from it by least squares against a real `lm_head.weight`
(relative rms residual 1.6e-3), and ran the full masked-vs-unmasked comparison on
real logits for ten student configurations spanning mean KLD 4.8e-5 to 1.0 nats.

**The padded columns are not dead.** Their rows have norm ≈ 0.4795 against a
typical real-row norm of 1.21, and they are mutually cosine-0.999998 — one
untrained direction, repeated 24 times. Their logits run −4.19 to −0.46 against
a mean top-1 logit of 19.72, so they hold about 1.6e-8 of the probability mass.

**The measured effect of masking:**

| student configuration | unmasked mean KLD | masked − unmasked | relative |
|---|---|---|---|
| BF16 floor (shared native head) | 0.011512861121 | +8.56e-11 | +7.44e-09 |
| K6 tr3-6bpw (shared native head) | 0.013709446677 | +1.01e-10 | +7.40e-09 |
| official FP8 (shared native head) | 0.020555552137 | +1.50e-10 | +7.31e-09 |
| Dione Q4 (shared native head) | 0.027261644072 | +1.97e-10 | +7.24e-09 |
| RTN per-row int8 head | 0.000048287080 | +4.45e-13 | +9.21e-09 |
| RTN per-row int6 head (stock EXL3) | 0.000997485627 | +4.60e-12 | +4.61e-09 |
| RTN per-row int4 head | 0.016038835491 | +9.90e-11 | +6.17e-09 |
| group-128 affine 6b (GGUF/MLX-like) | 0.000348939975 | +4.69e-12 | +1.34e-08 |
| group-128 affine 4b (MLX 4-bit-like) | 0.005698363496 | +6.83e-11 | +1.20e-08 |
| **adversarial global-scale int4 head** | 0.183019918220 | **−5.12e-08** | −2.80e-07 |

The last row is the stress case: a deliberately bad head quantization that
shifts the padded logits by +2.1 nats on average and 4.2 nats at worst, and
blows the KLD to 0.183. Even there the masking changes the answer by 5e-8 nats.

**Verdict for every published malaiwah number: neither a correction nor a bias
disclosure. A protocol-policy disclosure only.** Each of our eight published
values changes at the 8th or 9th significant figure and nowhere earlier:

| published row | as published (unmasked) | masked equivalent | first sig. figure that moves |
|---|---|---|---|
| BF16 floor (streaming) | 0.011505922619330299 | 0.011505922704933474 | 9th |
| K8 tr3-8bpw | 0.012384191023436866 | 0.012384191115368088 | 9th |
| BF16 floor (cross-stack) | 0.012711599817250709 | 0.012711599911537296 | 9th |
| K6 streaming | 0.013714888822596553 | 0.013714888924089065 | 9th |
| K6 sealed | 0.013723384665701147 | 0.013723384767254605 | 9th |
| official FP8 | 0.020615254540417995 | 0.020615254691072615 | 9th |

For scale, in nats: padded-masking delta **1.0e-10**; our own sealed-vs-streaming
bridge **8.5e-6**; his window-clustered SE on this panel **3.19e-3**. The
divergence is 83,000× smaller than the tightest real uncertainty we publish and
31,000,000× smaller than his interval.

**We are adopting masking anyway**, as a zero-cost convergence on his standard,
not as a fix. Going forward `estimator.vocab_masking_policy` is a required-to-be-
stated field (invariant JOINT-007) and the protocol file sets
`padded_column_policy: mask_both_sides`.

**It is deliberately NOT a comparability key input.** A key input is something
that can move a comparison. This cannot: it is four orders of magnitude below
the smallest difference any of our tables resolve. Adding it to the key would
re-key all 66 rows and break every published cross-reference for nothing.

**Where it would have mattered and did not:** we worried about stock-EXL3 quants
with `head_bits=6..8` and about the Dione Q4, where teacher and student have
different head weights so the padded columns can genuinely differ. Case B above
covers exactly that — quantized heads, including group-128 affine — and the
answer is the same to within a factor of two. The Dione Q4 turns out to keep
`lm_head` native BF16 anyway, so every malaiwah number on his panel has a shared
unquantized head.

---

## 4. Divergence 2 — the calibration-clean scope

### 4.1 We reproduced his scan exactly

We fetched the 665 published token arrays (5.53 MB) and re-ran his 13-gram
overlap scan with an independent implementation:

```
$ bin/joint-standard overlap-scan --panel panel.json --arrays arrays/ \
    --expect his/window_selection.json
cross-check against window_selection.json: 25 windows, 0 mismatches
```

All 25 windows match his published `shared_13gram_count` and
`shared_13gram_fraction` exactly. The committed result is
`registry/protocol/window-selection.brandonmusic-final25.json`.

His finding stands and it is the single most important thing in his standard:
**`document_id_in_calibration` is `false` for all 25 windows** — document-level
separation is clean — **and six of them still share 37–39 % of their 13-grams
with calibration-role windows.** Document-hash dedup alone does not catch it.

One detail worth spelling out, because it is easy to get wrong: the denominator
is the **deduplicated** gram set. An axis4 window has only ~710 distinct 13-grams
out of 2036 slices, because that corpus repeats itself. Using 2036 would report
13 % instead of 38 %.

**The excluded set is not one domain.** Six axis4 windows, plus `final-0021`
(axis2_legal, 7.1 %) and `final-0022` (axis3_code_agentic, 5.8 %). A 19-window
whole-axis4 drop is a *different* scope and gives materially different answers.

### 4.2 The 0.05 threshold is unjustified, and it matters

His threshold is a bare `frac > 0.05` literal in `cli.py` with no published
sensitivity analysis. Here is the analysis. Window counts first:

| threshold | 0.02 | 0.03 | 0.04 | **0.05** | 0.06 | 0.075 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|---|---|
| windows kept | 10 | 13 | 16 | **17** | 18 | 19 | 19 | 19 |

The *window count* has a plateau at 19 for any threshold in [0.075, 0.20]. The
*numbers* do not. Relative move of each published mean from the full 25-window panel:

| threshold | n | K6 sealed | K8 stream | FP8 x-stack | BF16 floor x-stack | his 4bpw |
|---|---|---|---|---|---|---|
| 0.20 / 0.10 / 0.075 | 19 | −5.04 % | −5.23 % | +2.62 % | −6.06 % | +6.65 % |
| 0.06 | 18 | −12.20 % | −12.88 % | −2.61 % | −12.43 % | +1.69 % |
| **0.05 (his)** | **17** | **−14.91 %** | **−12.55 %** | **−9.46 %** | **−16.24 %** | **+1.61 %** |
| 0.04 | 16 | −10.85 % | −8.39 % | −5.23 % | −12.24 % | +6.39 % |
| 0.03 | 13 | −11.02 % | −8.04 % | −11.40 % | −12.95 % | +8.71 % |
| 0.02 | 10 | −5.91 % | −4.29 % | −7.00 % | −10.25 % | +18.19 % |

Moving the threshold from 0.075 to 0.05 triples the size of the correction to
K6. The highest overlap among the retained windows is 4.75 % (`final-0014`), so
0.05 separates cleanly — but only just, and nothing about 0.05 is derived.
**This is the first thing worth a joint decision.**

One result *is* robust across every threshold: **every malaiwah row moves down
and his 4bpw row moves up, at all eight thresholds.** The sign flip between
contributors is not an artifact of 0.05.

### 4.3 Both scopes, published side by side

Recomputed from our own published per-window arrays. No GPU, no re-measurement:
every value is the equal-weight mean of exactly the windows its scope names, and
`registry/tools/registry_joint_check.py` re-derives all twelve to prove it.

| row | scope | mean | BCa 95 % | SE (window-clustered) | deff | sigma_run |
|---|---|---|---|---|---|---|
| K6 tr3-6bpw sealed 5-run | panel25 | 0.013723384666 | [0.011155, 0.016627] | 1.440e-03 | — | 0.0 (5 runs) |
| K6 tr3-6bpw sealed 5-run | clean17 | 0.011677286369 | [0.008856, 0.014785] | 1.563e-03 | — | 0.0 (5 runs) |
| K6 tr3-6bpw streaming 2-run | panel25 | 0.013714888823 | [0.011157, 0.016630] | 1.437e-03 | 23.3 | 0.0 (2 runs) |
| K6 tr3-6bpw streaming 2-run | clean17 | 0.011675992694 | [0.008863, 0.014779] | 1.559e-03 | 28.5 | 0.0 (2 runs) |
| K8 tr3-8bpw streaming 2-run | panel25 | 0.012384191023 | [0.009991, 0.015219] | 1.387e-03 | 21.1 | 0.0 (2 runs) |
| K8 tr3-8bpw streaming 2-run | clean17 | 0.010829419870 | [0.008063, 0.013838] | 1.500e-03 | 24.9 | 0.0 (2 runs) |
| official FP8 cross-stack | panel25 | 0.020615254540 | [0.016469, 0.025624] | 2.348e-03 | — | — |
| official FP8 cross-stack | clean17 | 0.018665326569 | [0.014192, 0.024794] | 2.739e-03 | — | — |
| BF16 floor cross-stack | panel25 | 0.012711599817 | [0.010329, 0.015392] | 1.342e-03 | — | — |
| BF16 floor cross-stack | clean17 | 0.010647639361 | [0.008124, 0.013491] | 1.403e-03 | — | — |
| his 4bpw (his k4 scorer) | panel25 | 0.024554564250 | [0.019433, 0.035881] | 3.810e-03 | 73.9 | 0.0 (5 runs) |
| his 4bpw (his k4 scorer) | clean17 | 0.024948837056 | [0.018141, 0.041662] | 5.348e-03 | 105.0 | 0.0 (5 runs) |

Scope deltas: K6 sealed **−14.91 %**, K6 streaming −14.87 %, K8 −12.55 %,
FP8 −9.46 %, BF16 cross-stack floor −16.24 %, **his 4bpw +1.61 %**.

**Two rows cannot be recomputed at all.** The BF16 *streaming* floor
(0.011505922619) and the Dione Q4 (0.027262784815) have scalar-only receipts —
run means and a tokenwise digest, no per-window array. Their registry rows now
say so in `notes` rather than quietly having no clean sibling.

**The registry caught a real error here.** The first version of these rows sat
under the parent panel's comparability key, and invariant CMP-003 refused them:
*"shares comparability key … with rows scoring a different number of positions."*
That is correct. A different window set is a different panel, so `clean17` now
has its own derived panel record (`panel--glm53.brandonmusic.final25-clean17`)
and its own derived reference, which makes a clean17-vs-panel25 table
structurally impossible rather than merely discouraged.

### 4.4 Do the conclusions survive?

Paired per-window comparisons, BCa on the differences, on both scopes:

| comparison | scope | mean A | mean B | ratio | 95 % CI of A−B (BCa) | A better in | sign p |
|---|---|---|---|---|---|---|---|
| K6 − K8 | panel25 | 0.013723385 | 0.012384191 | 1.108 | [+0.000695, +0.002330] | 5/25 | 4.1e-03 |
| K6 − K8 | clean17 | 0.011677286 | 0.010829420 | 1.078 | [+0.000153, +0.001573] | 4/17 | 4.9e-02 |
| K6 − FP8 | panel25 | 0.013723385 | 0.020615255 | 0.666 | [−0.009982, −0.004872] | 25/25 | 6.0e-08 |
| K6 − FP8 | clean17 | 0.011677286 | 0.018665327 | 0.626 | [−0.010506, −0.004798] | 17/17 | 1.5e-05 |
| K8 − FP8 | clean17 | 0.010829420 | 0.018665327 | 0.580 | [−0.011966, −0.005493] | 17/17 | 1.5e-05 |
| his 4bpw − K6 | clean17 | 0.024948837 | 0.011677286 | 2.137 | [+0.008430, +0.028684] | 0/17 | 1.5e-05 |

- **K8 better than K6 SURVIVES, weakened.** The paired BCa still excludes zero on
  the clean scope, but the interval lower bound falls from +0.000695 to +0.000153
  and the sign test goes from p = 0.0041 to p = 0.049 — right at the edge. The
  headline gap shrinks by about a third. We will not restate "K8 is better than
  K6" without the scope attached.
- **K6 better than FP8 STRENGTHENS.** 17/17 windows on the clean scope, ratio
  0.666 → 0.626.
- Note the two marginal CIs for K6 and K8 overlap almost completely on both
  scopes. Anyone eyeballing them would call it a tie. The paired interval is
  ~10× tighter and says otherwise. That is exactly his point about ranking, and
  our data demonstrates it.

### 4.5 Attributable error on the clean scope, and a result that argues for it

Same-lane only. The cross-stack pair (official FP8 minus the cross-stack BF16
replay floor, both `cross_stack`) is the one we can recompute on both scopes:

| scope | FP8 | same-lane floor | attributable | BCa 95 % | ratio |
|---|---|---|---|---|---|
| panel25 | 0.020615255 | 0.012711600 | **0.007903655** | [+0.005823, +0.011253] | 1.622 |
| clean17 | 0.018665327 | 0.010647639 | **0.008017687** | [+0.005663, +0.012022] | 1.753 |

**The attributable error moves +1.44 % between scopes while its two inputs move
−9.46 % and −16.24 %.** The subtraction is the stable quantity here; the raw
numbers are the unstable ones. That is a direct, measured answer to §5.3 of his
report ("do not publish subtracted numbers"), and §7 below takes it up properly.

The **same-lane K6/K8 attributable table cannot be recomputed** on the clean
scope: their same-lane floor is the streaming BF16 floor, whose receipt is
scalar-only. Borrowing the cross-stack floor instead would be exactly the
cross-lane subtraction our own BIAS-006 refuses, so we do not. The published
panel25 attributable ratio (K6 0.002209 / K8 0.000878 = 2.52×) therefore stands
as a panel25 number only, and re-deriving it on the clean scope needs one
re-measurement of the streaming BF16 floor with per-window output — cheap, and
on the list.

### 4.6 His non-uniformity finding reproduces on our data

He measured NVFP4-over-EXL3 ratios of 1.50× general / 1.97× legal / 1.65×
code-agentic and concluded a single-corpus mean hides where a codec hurts. Same
test, our artifacts, clean scope:

| domain | K6 sealed | K8 stream | FP8 x-stack | FP8/K6 | FP8/floor (same lane) |
|---|---|---|---|---|---|
| axis1_general (7 w) | 0.011739694 | 0.011367036 | 0.019568765 | 1.667× | 1.705× |
| axis2_legal (5 w) | 0.011448683 | 0.010324227 | 0.020672762 | **1.806×** | 2.050× |
| axis3_code_agentic (5 w) | 0.011818519 | 0.010581951 | 0.015393078 | **1.303×** | 1.532× |

A 1.39× spread across domains, and legal is the worst domain on our data as it
is on his. On the panel25 scope the contaminated axis4 domain shows the smallest
ratio of all (1.19×), which is what contamination should look like: it compresses
the differences it touches.

---

## 5. What we adopted, and how each piece is validated

`bin/selftest_joint_standard.py`, run with his `kld_eval` on `PYTHONPATH` and the
real panel/teacher available: **112 passed, 0 failed, 0 skipped**. On a stock
`python3` with none of that: **106 passed, 0 failed, 6 skipped** (the skips are
the oracle and real-data cases, and they announce themselves).

### 5.1 R0 canary — both halves, and it fires

His `cmd_canary_loader` implements R0-a (self-KLD exactly 0.0) plus a
teacher-top1-vs-realized band. R0-b, the one-position shift, exists only as
`tests/test_kld.py::test_one_position_shift_is_entropy_scale` on V=512 random
logits — it never runs against the real teacher inside a session. His own
proposed standard names it as part of R0. That gap is the concrete thing we can
hand back.

Ours runs both, against whatever logits it is given, and refuses on failure:

```
  PASS  R0 passes on a well-formed teacher    self-KLD 0.0 both scopes; shift 12.418 nats = 4.7x entropy
  PASS  FIRE R0-a: one nudged logit out of 524288        R0-a: 1 of 256 positions have non-zero KLD…
  PASS  FIRE R0-a: deliberately misaligned pair          R0-a: 255 of 255 positions have non-zero KLD…
  PASS  FIRE R0-b: near-constant rows pass R0-a at 0.0 and still fail   R0-b: one-position shift gave mean 3.54289e-15 nats…
  PASS  FIRE R0-c: teacher-top1 agreement outside the band
  PASS  self-KLD is exactly 0.0 in BOTH scopes with padded columns present
  PASS  ORACLE: kld_eval.score_window == our fp64 kernel   max|d|=2.67e-16
  PASS  REAL teacher window passes R0        shift 7.903 nats vs entropy 1.770 (4.5x)
```

The R0-b FIRE case is the interesting one. It hands the canary a teacher whose
rows barely differ — what a left-on prefix cache looks like. That teacher passes
R0-a at **exactly 0.0** and is still broken, and only R0-b catches it. A canary
nobody has watched fire is decoration.

On his real teacher window: self-KLD 0.0 in both the masked and unmasked scopes,
one-position shift 7.903 nats against a mean teacher entropy of 1.770 — 4.5×.

### 5.2 Per-domain stratification

Our `kld-report.json` has carried a `per_domain` block all along; we simply never
published it. Now every enriched registry row carries `by_domain` with a
window-clustered SE and a BCa interval, and invariant JOINT-006 requires the
per-domain positions to sum to `measurement_scope.scored_positions`.

### 5.3 Window block bootstrap with BCa

Textbook BCa (Efron & Tibshirani): percentile endpoints, `z0` from the bootstrap
mass below the observed statistic, acceleration from a leave-one-window-out
jackknife. Three backends, and the receipt names which one ran — `kld_eval`
(his), `numpy` (ours, driving PCG64 with his seed and draw pattern), `stdlib`
(ours, no dependencies).

Known-answer, against his four published analysis receipts, from his 25
per-window means alone:

```
  PASS  published percentile CI exl3/selected   [0.022341151367, 0.036971250314] max|d|=6.9e-18
  PASS  published BCa CI exl3/selected          [0.022653106208, 0.037476289386] max|d|=3.5e-18
  PASS  published percentile CI exl3/panel      [0.024646838372, 0.037044329848] max|d|=6.9e-18
  PASS  published BCa CI exl3/panel             [0.024964506716, 0.037418759476] max|d|=6.9e-18
  PASS  published percentile CI nvfp4/selected  [0.036804664652, 0.064325111657] max|d|=2.8e-17
  PASS  published BCa CI nvfp4/selected         [0.038010260038, 0.066435042806] max|d|=2.1e-17
  PASS  published percentile CI nvfp4/panel     [0.038471793958, 0.061625142006] max|d|=6.9e-18
  PASS  published BCa CI nvfp4/panel            [0.039814482413, 0.063889491867] max|d|=1.4e-17
  PASS  ORACLE: kld_eval.block_bootstrap == ours   max|d|=6.94e-18
```

Sixteen endpoints to within one ULP, on a different OS with much newer numpy —
and then the same input through his own code, agreeing to 7e-18. His four
`se_clustered_window` values reproduce exactly too.

Why per-window means suffice: every window is exactly 2047 scored positions, so
the token-weighted mean of a window resample equals the plain mean of the
resampled window means. Our receipts carry per-window means but not per-token
arrays; that equivalence is what makes all 60 existing rows analysable with no
GPU. It is asserted in the checker, not assumed.

The **design effect** this exposes is the practical payoff:
`deff_window` runs 21–29 on our rows and 74–105 on his 4bpw. The naive SE
understates by 4.6× to 10×. Anyone who has ever quoted `std/sqrt(N)` on this
panel — us included — was quoting an interval five to ten times too narrow.

### 5.4 sigma_run and quadrature

```
  PASS  sigma_run exl3_3cold_25w (3 runs)        got 0.000000000000e+00
  PASS  sigma_run nvfp4_2cold_25w (2 runs)       got 3.332041111285e-04
  PASS  two-run sigma carries its 1-dof flag
  PASS  2-run sigma == |delta|/sqrt(2)
  PASS  his published SE_total 6.00e-3           got 6.000156395e-03
  PASS  sigma_run/SE ratio inside his 0.20 gate  ratio=0.05562
  PASS  FIRE: sigma_run/SE = 0.5 trips the 0.20 gate
```

One honest note back to him: his headline NVFP4 `sigma_run = 3.33e-4` is an
**n = 2** estimate. `std(ddof=1)` over two values is exactly `|Δ|/√2` — one
degree of freedom, and we verified it is exactly that. He *did* run three cold
NVFP4 runs, but only on a 3-window subset, where sigma is 1.40e-3, four times
larger. Our implementation flags any two-run sigma in the receipt note.

On clean-scope rows we drop `determinism.run_means` (they are panel-scope means
and would be wrong) and quote `sigma_run = 0.0` **only** where the runs are
bitwise identical — because bitwise identity implies identical per-run means on
*every* subset of windows. Anything else is not recoverable from panel-scope run
means, so it is omitted rather than guessed. Invariant JOINT-003 enforces that a
zero sigma requires bitwise evidence.

### 5.5 Percentile-exceedance guard, plus one refusal our data needs

The guard reproduces his behaviour: at N = 34,799 it admits p90/p95/p99 and
suppresses p99.9 (34.8 exceedances).

**A micro-finding to flag:** his `n * (1.0 - q) >= MIN_EXCEEDANCES` is wrong by
one ULP at an exact boundary. `1 - 0.9` is `0.09999999999999998`, so n = 1000,
q = 0.90 evaluates to 99.99999999999998 and the guard suppresses a quantile with
exactly 100 exceedances. Ours uses a tolerance. This changes **no** decision in
his campaign — his panel sizes are 34,799 and 51,175 — but it is a real edge.

**And a refusal he does not need but we do.** Our published receipts carry
per-window percentiles, not per-token arrays. A panel p95 is *not* a function of
per-window p95s, so `guard_pooled_percentiles` returns a refusal rather than a
number:

> pooled token percentiles are not derivable from per-window summaries; a panel
> p95 is not a function of per-window p95s

The remedy is the same thing we would ask of him (§10): publish per-window
sufficient statistics.

### 5.6 McNemar

Continuity-corrected chi-square plus the exact binomial, with no scipy. Hand
check first, arithmetic written into the assertion:

> A right / B wrong = 12, B right / A wrong = 3.
> χ² = (|12−3| − 1)² / 15 = 64/15 = 4.2666666666666666
> exact two-sided binomial = 2·(1+15+105+455)/2¹⁵ = 1152/32768 = **0.03515625**

Then his five published p-values, reproduced from the published cell counts:

```
  PASS  published p reproduced: nvfp4-vs-exl3 clean   963/1629  p=5.440149e-39  rel=1.3e-15
  PASS  published p reproduced: nvfp4-vs-exl3 panel  1273/2120  p=8.570506e-48  rel=1.4e-14
  PASS  published p reproduced: paired_blockm         459/485   p=4.158279e-01  rel=5.3e-16
  PASS  published p reproduced: paired_chunk          477/443   p=2.766049e-01  rel=1.1e-14
  PASS  published p reproduced: paired_kv             525/467   p=7.033428e-02  rel=2.4e-15
```

Our own rows cannot carry McNemar yet: it needs per-position top-1 agreement for
both runs, and per-window means cannot supply a contingency table. The `paired`
verb says exactly that instead of inventing one.

---

## 6. His "one frozen protocol file" rule broke — and the fix

His rule is right. It did not survive his own campaign. Three distinct
`protocol_sha256` values appear across his published receipts:

| hash | stamped into |
|---|---|
| `53e165dd…` | `teacher_manifest`, `window_selection`, `run-run1/2/3/3b`, `run-r0a/b/c`, `r5_sweep_r0*` |
| `8e80e8e1…` | `analysis-run1-{selected,panel}`, `kld_card.{md,json}`, `r5_sweep_cold3`, `paired_kvfp8` |
| `4d1d91ad…` | all NVFP4 receipts — **and the file currently published** |

He discloses two of the three transitions himself. The mechanism is visible in
`scripts/make_protocol.py`: the generator writes `governing_document` as a plain
string reading "NOT FOUND", while the published file has it as a dict carrying
the report's sha256 and also carries a `student_nvfp4:` block the generator never
writes. The file was hand-edited twice after generation, to add identity
metadata. **Neither edit changed a scoring rule**, and yet the EXL3 headline and
the NVFP4 headline it is compared against carry different protocol hashes, and no
published receipt except the NVFP4 ones carries the hash of the file now on the
Hub. (One receipt, `r0_student_canary.json`, carries no `protocol_sha256` at all —
it is written outside `_write_json()`.)

That is precisely the failure the rule exists to prevent, and it is not
carelessness — it is what happens when the hash covers bytes that include
identity metadata.

**Our fix: hash the scoring-relevant subset, and publish both hashes.**

- `protocol_file_sha256` — sha256 of the raw bytes. His rule, unchanged,
  byte-level provenance.
- `protocol_scoring_sha256` — sha256 of a canonical JSON serialisation
  (`sort_keys`, `separators=(',',':')`, ASCII, UTF-8) of exactly
  `{scoring, selection, uncertainty, determinism, canary_r0, lane, reporting}`.

Two receipts are comparable when the **scoring** hashes match. A file hash that
moved while the scoring hash held is a provenance note, not an incomparability.
Proven in the selftest, both directions:

```
  PASS  identity-only edit: file hash MOVES, scoring hash HOLDS  80df521eb46f -> 1c64ad3aada8
  PASS  a scoring edit MOVES the scoring hash                    20ea68c0c730 -> cb8f78c4dde8
```

And the stamp is enforced, not merely written: `require_stamp` refuses an
unstamped receipt, a foreign schema, and a stale scoring hash, and every verb of
`bin/joint-standard` runs it before writing anything.

---

## 7. Divergence 3 — floors and subtraction

§5.3 of his report ranks "Subtraction (not recommended)" third and says *"Do not
publish subtracted numbers."* Our headline attributable-error framing **is** a
subtraction. This is a real disagreement and we are not going to paper over it.

Our position, and what we changed:

1. **The subtraction is strictly same-lane.** Registry rule BIAS-006 refuses a
   floor whose lane differs from the row's, and now also refuses one whose
   *scope* differs (§4.5). We never publish a subtracted number without the raw
   row and the floor beside it.
2. **His §5.3 item 1 is our BIAS-006, independently arrived at.** "Isolate and
   gate … measure the engine's floor … report the decomposition" is what the
   floor rows are for. The disagreement is narrower than it looks: it is about
   whether the *difference* may be quoted as a headline, not about whether the
   floor should be measured.
3. **New evidence, from §4.5.** Across the panel25→clean17 scope change, the
   cross-stack FP8 attributable error moves **+1.44 %** while its two inputs move
   **−9.46 %** and **−16.24 %**. The subtraction is the quantity that survives a
   contamination correction; the raw numbers are the ones that do not. That is an
   argument *for* publishing the decomposition, and it is measured rather than
   asserted.
4. **What we concede.** A subtracted number is only meaningful when the floor is
   same-lane, same-scope, same-teacher and same-panel. Three of those four
   constraints are now mechanical refusals. The fourth (same-scope) was added
   this week because his scan showed why it was needed.

Proposal: publish the raw row, the floor, and the difference-with-interval, and
never the difference alone. His §5.3 item 2 (difference-in-differences) is
compatible with that.

---

## 8. Where our data independently confirms his, and vice versa

Worth stating plainly because collaboration is the point.

**His data confirms our lane rule.** His same 4bpw artifact reads **0.0305** on
his serving stack and **0.024555** on the packed reference stack. That ~24 % gap
between two readings of one checkpoint is exactly what our
`estimator.stack_relation` field exists to keep apart, and our measured BF16
floors (0.011506 same-stack, 0.012712 cross-stack, a 0.001206 nats difference)
are that gap's formalisation. He arrived at the observation independently and
without needing our field.

**Our registry predicted his contamination finding.** Our panel record for his
panel has carried the disclosure `weak_contamination_guard` since we ingested it:

> This panel's only contamination guard is ROLE SEPARATION … No lexical or
> n-gram scan is published … materially weaker than the malaiwah v5 suites,
> which run a 12-word shingle whole-document pre-exclusion and report 0 hits.

He then ran the scan and found 37–39 % overlap in axis4. That is a mutual
confirmation running in our direction.

**His determinism result matches ours on a different lane.** EXL3 through the
b12x path: 25/25 windows bitwise identical across three cold boots including a
shuffled window order, `sigma_run` exactly 0.0. Our sealed lane: 5 cold runs, one
distinct `tokenwise_kld_sha256`, `sigma_run` exactly 0.0. Two independent stacks,
same conclusion — **determinism is a property of the kernel path, not of the
format**. His NVFP4 counter-example (0/25 bitwise, 93.65 % of tokens changed,
2,652 top-1 flips, a single token swinging 4.69 nats, and yet the mean barely
moving) is the sharpest demonstration of that anyone has published, and it is why
`sigma_run` belongs beside the mean rather than in a footnote.

---

## 9. Calling his harness instead of reimplementing — the explicit evaluation

We were asked to prefer calling his code where it is good and installable. Verb
by verb, from reading the source and running it on this Mac:

**`kld-eval inspect | select | run | sweep-analyze | paired | analyze | card`:
NOT CALLABLE BY US.** Every verb begins with `kld_eval.protocol.load_verified()`,
which refuses to proceed unless five files hash to values recorded in *his*
protocol.yaml, at absolute paths inside his machine:

```
$ python -m kld_eval.cli inspect
kld_eval.protocol.ProtocolMismatch: student config.json: missing at
  /home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw/config.json
```

Reusing the CLI means forking his protocol.yaml with our own identity block,
which is authoring a Derivative, not calling his tool.

**`kld_eval.analysis.stats`, `kld_eval.kld.core`, `kld_eval.teacher.token_ngrams`:
CALLABLE, AND WE CALL THEM.** Pure numpy/scipy/pandas/torch, no GPU, no engine,
no HF token, no access to his checkpoint. His 16 unit tests pass unmodified on a
fresh macOS-arm64 venv with libraries far newer than his pins (numpy 2.5.2 vs
1.26.4, scipy 1.18.1 vs 1.17.1, pandas 3.0.5 vs 2.3.3, torch 2.13 vs 2.12):

```
$ python -m pytest tests/ -q
................                          [100%]
16 passed in 1.26s
```

`bin/jointstd/oracle.py` imports that layer when it is importable and uses it as
the oracle our own implementation is pinned against — bootstrap, clustered SE,
n-gram digests and the per-token kernel, all four cross-checked in the selftest.
Nothing of his is copied into this repository.

**Why we still keep our own implementation:**

1. **Dependency.** `registry/ make check` must run on a stock interpreter with no
   `pip install` — that is the registry's contract with contributors. scipy and
   pandas are not available there. The stdlib fallback exists for that
   environment and agrees with the numpy path to 1.8 % of a CI width.
2. **Licence** (§10).

**Recommendation:** import `kld_eval` as a library where it is installed; do not
fork the CLI; do not vendor the source.

---

## 10. Licence — an operator decision, not a default

His model repo carries `license: other`, `license_name: shapleymcg-1.0`, and a
29,918-byte root `LICENSE`: the **SHAPLEYMCG LICENSE 1.0**, an
"Attribution-Required, Source-Available License with Named Exclusion",
© 2026 Brandon M. Music. §10.5 states it is not OSI-approved and should not be
called open source.

- §1.5(a) names an **Excluded Party** (the person behind the "0xSero" persona,
  `github.com/0xsero` and `huggingface.co/0xSero`); §1.6 makes those Excluded
  Channels; §4.1 grants them no rights; §4.3 says public availability is not
  permission.
- **We are not the Excluded Party** — the opposite: his `THIRD_PARTY_NOTICES.md`
  credits "malaiwah for the GLM-5.2 MTP-78 overlay and calibration capture", and
  his Schedule A audit clears 99 of 112 repositories and states the excluded
  party's REAP pipeline is independent and predates his corpus.
- **But we publish a fidelity number for an 0xSero artifact**
  (`measurement--glm53.dione-q4.brandonmusic-final25` = 0.027262784815 for
  `artifact--0xsero.glm-5.3-flash-exl3-q4`). Measuring a third party's model is
  not "use of the Work" and is not done for their benefit, so §4.2 does not reach
  it. The live questions are §5.1 ("must not … knowingly permit the Excluded
  Party to obtain it through You") against a public GitHub repository, and §1.2's
  broad Derivative definition, which sweeps in "re-implementations made with
  reference to the Work".
- §3 attribution is a **condition on the grant**, not a covenant (§2.4): use
  outside it is unlicensed. §3.1 requires the full LICENSE text plus the Schedule
  B notice in every copy and Derivative; §3.2 requires prominent attribution in
  any published benchmark, model card, README or post relying on the Work.

**What we did:** clean-room implementation of the statistics (BCa,
cluster-robust SE, McNemar, Wilson, n-gram shingling are all textbook and are
specified in Appendix A of the report he asked us to read); adoption of the
protocol design with loud citation; **no `kld_eval` source vendored into this
repository**. The one thing we do reproduce is a fixture of 25 per-window means
and his published CI endpoints — measurement *results*, cited and attributed, not
code — so that his numbers are the known-answer test our implementation must pass.

**Open question for him, and it costs nothing to ask:** may we depend on
`kld_eval` as a library, and what attribution do you want where? A one-line
answer removes the ambiguity entirely.

---

## 11. Corrections both sides should make

**Ours:**

1. `uncertainty.method: "none"` on all 60 rows — **fixed**, 12 rows now carry BCa
   intervals and the rest carry the new fields where their receipts allow.
2. No per-domain publication despite computing it — **fixed** (`by_domain`).
3. No protocol file — **fixed** (`registry/protocol/glm53-joint-kld-protocol.v1.json`).
4. No masking policy recorded — **fixed** (`estimator.vocab_masking_policy`,
   invariant JOINT-007); the numbers themselves need no correction (§3).
5. Attributable-error headline (2.52×) is a **panel25** number; the clean-scope
   version needs a re-measured streaming BF16 floor with per-window output.
6. Our published rows quoted no interval at all. Anyone who inferred one from
   `std/sqrt(N)` was off by 4.6–10× (the design effect, §5.3).

**His, offered in the same spirit:**

1. **Scope labelling.** The clean-17-window means are 0.029258 (EXL3) and
   0.049640 (NVFP4); the full-25 means are 0.030480 and 0.049218. The "51,175
   positions" figure belongs to the panel scope, not the clean one. A summary
   that puts 0.0305 in a row labelled "clean scope" is mixing them.
2. `kld_card.md` says **"25 windows / 34,799 scored positions"** — 34,799 is 17
   windows. Looks like `analysis/cards.py` takes the window count from the
   manifest and the token count from the selected scope.
3. The headline table pairs the **clean-scope** NVFP4 mean/CI with a `sigma_run`
   computed on the **panel** scope, and the quadrature check uses the **panel**
   SE.
4. `sigma_run = 3.33e-4` is n = 2, one degree of freedom (§5.4).
5. The KV/backend paired control ran on the 12-window subset `final-0000…0011`,
   which **includes three contaminated windows** (0003, 0007, 0011). Within-engine
   and paired, so low impact — but it is not a clean-scope result.
6. The EXL3-vs-NVFP4 comparison changes three things at once (weight format,
   attention backend, KV dtype) plus the container image. His
   `paired_kvfp8_vs_kvnvfp4` control bounds two of the three at ±0.006 nats
   against a 0.0204-nats effect — **that is a strong result and it deserves to be
   stated explicitly**; "KV dtype is a non-lever" undersells what the experiment
   actually controlled.
7. **The 0.0246 anchor is labelled three different ways** and this one matters
   most — see §12.
8. `_percentile_ok`'s float boundary (§5.5).
9. `r0_student_canary.json` carries no `protocol_sha256`.
10. The per-token Parquet is gitignored, so **no third party can recompute his
    percentiles, top-1 rates, per-domain CIs, `rms_delta_p`, `ln_ppl_ratio`, or
    any McNemar cell count** — only the means and mean-CIs, via the per-window
    arrays. Publishing per-window sufficient statistics (n, Σd, Σd², top-1 counts,
    a few quantiles) would close that at negligible cost. It is the single
    highest-value thing to ask for.

---

## 12. The 0.0246 anchor must be resolved before either side publishes a lane gap

The lane-gap argument both sides want as a headline rests on one number, and it
is described three different ways:

1. **His `RUN_SUMMARY.md`:** "the community/cloud pipeline (transformers eager,
   EP4, B200) reported ≈0.0246 for this checkpoint" — i.e. *his artifact measured
   by us*.
2. **His `kld_card.md`:** "official FP8 ≈ 0.0206 (other stack) / 0.0246 (this
   stack)" — i.e. *the FP8 release, not his 4bpw*.
3. **Our registry:** `measurement--glm53.brandonmusic-4bpw.brandonmusic-final25`
   = 0.024554564250, on `pipeline--brandonmusic.glm53-packed-kld`, whose record
   reads "brandonmusic packed-surface KLD scorer (k4-tp2)", disclosure
   `author_reported_only`. Source: his own `results/five-cold-run-kld.json`
   (sha256 `d955bfae…`, hash-verified). That file names no engine and no stack.

**We have never measured his 4bpw ourselves.** Our published `reports/` tree
contains no 4bpw report, and both 4bpw rows in our registry sit on *his*
pipelines. So (1) is very likely him attributing his own k4-tp2 number to us, and
(3) is a TP2 packed-surface replay, not "transformers eager, EP4, B200". And
there is a plausible collision behind (2): our registry also holds the official
FP8 on the single-window sub-panel at 0.024629 / 0.024582 / 0.024611 — **two
unrelated quantities that both round to 0.0246 at three significant figures.**

The arithmetic he built on it checks out: 0.030480 − 0.024555 = **0.005926 nats**
(ratio 1.2413), and his panel BCa lower bound 0.024965 excludes 0.024555 by
0.000410 — exactly his "narrowly excludes by ~0.0004". **The number is right; the
label on it is wrong in at least two of three places.** Our registry already
marks that row `class: advisory` because it is a different reader; his Discord
post cites our 0.0137 beside his numbers as a same-panel reference point with no
such caveat. Both should be fixed before either lane-gap claim goes out.

---

## 13. Open questions for a joint decision

1. **The overlap threshold.** 0.05 is undefended and the correction it produces
   varies by 3× across plausible values (§4.2). Options: keep 0.05 and justify
   it; move to 0.075 where the window count plateaus; or publish a
   threshold-sensitivity band with every scope-dependent number. We lean towards
   the third.
2. **Report both scopes, always?** Our answer is yes — they move different
   contributors' rows in *opposite* directions, so neither can stand in for the
   other. That is now mechanical on our side: `clean17` is a separate panel with
   a separate comparability key.
3. **Protocol hashing: bytes or scoring subset?** We propose both (§6).
4. **Does the padded-column policy belong in a comparability key?** We say no,
   with 1e-10 nats of evidence (§3), and record it as a disclosed estimator
   property instead.
5. **May a same-lane, same-scope floor difference be published as a headline?**
   His §5.3 says no; our §4.5 evidence says the difference is the *stable* half.
   Worth ten minutes.
6. **Per-window sufficient statistics as a publication requirement?** Neither of
   us publishes per-token arrays. Both of us could publish per-window
   (n, Σd, Σd², top-1 count, quantile sketch) cheaply, and that makes every
   statistic in this standard independently recomputable by a third party.
7. **Licence terms for depending on `kld_eval`** (§10).
8. **Minimum cold runs on a non-deterministic path.** His own standard says ≥3;
   his headline NVFP4 sigma is n = 2. We would rather the standard held.

---

## 14. Reproducing everything here

```bash
# the frozen protocol and its two hashes
bin/joint-standard protocol

# reproduce his 13-gram scan (needs his panel.json + the 665 token arrays, 5.5 MB)
bin/joint-standard overlap-scan --panel panel.json --arrays arrays/ \
    --expect his/window_selection.json

# R0 on a real teacher window
bin/joint-standard canary --teacher window-0000.safetensors

# any of our rows, either scope
bin/joint-standard analyze \
    --report registry/protocol/per-window/k6-sealed.json \
    --scope-file registry/protocol/window-selection.brandonmusic-final25.json \
    --scope selected --oracle

# rank two quants properly
bin/joint-standard paired --a …/k6-sealed.json --b …/k8-streaming.json \
    --label-a K6 --label-b K8 --scope selected \
    --scope-file registry/protocol/window-selection.brandonmusic-final25.json

# validation
python3 bin/selftest_joint_standard.py          # 106 cases (112 with the oracle + real data)
cd registry && make check                       # 62 cases + 433 joint-invariant checks
bin/selftest_all.sh                             # 38 cases
```

Emitted analyses are in `docs/joint-standard/analysis/`; the per-window inputs
they read are in `registry/protocol/per-window/`.
