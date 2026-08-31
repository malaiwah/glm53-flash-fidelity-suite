# Corrections published to already-public artifacts

Every entry here is a change made to something that was already on the Hub. Each one is
**additive**: no sealed file was edited, no digest that a third party may have pinned was
invalidated, and no measured value moved. What changed is metadata that was missing or
misleading.

Published 2026-08-29 by the peer review described in `docs/REVIEW-DEFERRED.md`.

---

## 1. `malaiwah/GLM-5.3-Flash-fidelity-suite-v1` (dataset)

Commit `a98e2bfd6544326337f85c0886d569baa67acc82`.

**Files added** (5):

| Path | Why |
|---|---|
| `reference-bf16-shard0/capture-manifest-shard.json` | shard-scoped coverage |
| `reference-bf16-shard0/capture-cut-point.json` | the semantic point |
| `as-served-fp8-shard0/capture-manifest-shard.json` | shard-scoped coverage |
| `as-served-fp8-shard0/capture-cut-point.json` | the semantic point |
| `SHA256SUMS` | 6162 → 6166 lines, so it still covers every file |

**Files edited: none.** `capture-manifest-full.json` in both shards is byte-identical;
its sha256 is unchanged and still matches the pre-existing `SHA256SUMS` line.

### What was wrong

**Coverage (CC-06 / SH-11).** Both shard directories hold 512 `hidden_*.safetensors`.
Both ship a `capture-manifest-full.json` that says `complete: true`, `contexts: 5120`,
`expected_contexts: 5120`, `filter: "all"` and lists **5,120** `captures[]` records, with
no `shard_of`. A consumer trusting those fields believes it holds ten times the captures
it holds, and any coverage or statistical-power figure computed from `captures[]` is wrong
by 10x. Verified live before the fix: 513 tree entries / 512 safetensors per shard.

`capture-manifest-shard.json` states the shard-scoped truth — `complete: false`,
`contexts: 512`, `declared_contexts_full_run: 5120`, `shard_of {index: 0, total: 10,
stride: 1}`, `index_range [0, 511]` — and its `captures[]` records are asserted
byte-identical to the corresponding records in the full manifest, so the two cannot drift.
The full manifest is left in place and named in `full_manifest`, because it is the honest
record of the full 5,120-context run and its 5,120 sha256 rows let a third party verify
captures they generate themselves.

**Cut point (CC-18).** Neither manifest nor the safetensors `__metadata__` declared where
in the graph the tensors were taken. The only statement was prose in the dataset card, and
shipping `head/final_norm.safetensors` beside `head/head.safetensors` actively suggests a
norm+head replay. `capture-cut-point.json` declares
`semantic_point: after_final_rmsnorm_before_lm_head`, `tensor_key: hidden_states` and
`final_norm.applied_at_replay: false`.

The value was determined from the published bytes, not assumed: per-token RMS of
`hidden_0000.safetensors` is min 0.784 / median 1.380 / max 1.442 against
`rms(final_norm.weight) = 1.4315` — the signature of a post-RMSNorm-with-weight tensor.

Why it matters, quantified: a reader who normalises their own capture but replays this
reference as-is gets a mean KLD of ~0.014 nats from protocol mismatch alone — larger than
the published BF16 floor (0.011506) and larger than the K6 headline (0.013723) — with no
crash and a plausible top-1 agreement. The symmetric mistake (normalising both sides)
mostly cancels, at ~0.15%; the asymmetric one does not.

---

## 2. `malaiwah/GLM-5.3-Flash-TR3-6bpw` and `-TR3-8bpw` (models)

Commits `50c443d0b1003539e1c417b0a9fbb37ee6d830d5` (6bpw) and
`b5ca3b37c1053f1a3bcd9b5ca9ffa9dbc5e7fbb9` (8bpw).

**File added:** `MATERIALIZATION-PATHS.md` (one per repo).
**Files edited: none.**

### What was wrong (CC-17 / SEC-05, "known defect 4")

`materialization-receipt.json` records the producer's absolute paths on a rented GPU
filesystem that no longer exists:

| repo | `packed_root` | `output_root` |
|---|---|---|
| TR3-6bpw | `/home/jl_fs/glm53-k6/out-k6` | `/home/jl_fs/glm53-k6/ckpt-k6` |
| TR3-8bpw | `/home/jl_fs/glm53-k6/out-k8` | `/home/jl_fs/glm53-k6/ckpt-k8` |

A reader following them gets a hard failure pointing at someone else's machine, with
nothing in the repository saying why.

### Why a sidecar and not a correction

The receipt is self-sealed, and that seal is verified against the **published bytes** on
every measurement run (`engines/tools/tr3_surface.py::verify_seal`, reached from
`bin/measure_cloud.py`, which raises `this release's PUBLISHED seal does not reproduce`).
Editing the paths would permanently break every future measurement against these releases
and would falsify a record of where the encode actually ran.

Verified after publishing, with the project's own canonicalisation: both receipts'
`receipt_sha256` still reproduce (`3cb08d4d...`, `b12e257e...`).

The sidecar states that the fields are sealed provenance rather than resolvable locations,
names the reading path that works (`--source tr3` / `--source exl3hf`), records that
`--source checkpoint` and `--source payload-store` are producer-side paths unreachable
from a published repo, and notes the two releases' differing schema namespaces.

---

## 3. `malaiwah/quant-fidelity-registry` — the 42 per-domain intervals (STAT-01 + STAT-17)

**Published 2026-08-30. This one changes NUMBERS, not only metadata**, which is why it
was written up for an operator decision in `docs/REVIEW-DEFERRED.md` first and is
disclosed here before the mirror rather than after.

### What was wrong

Forty-two per-domain confidence intervals in `registry/data/measurements.jsonl` were
emitted as `ci95_low`/`ci95_high` with `interval_kind: "bca"`, and every consumer
read them as 95% intervals. They are not. Simulated against a lognormal fitted to each
cell's own window means, **4000 replications per cell**, the procedure that produced
them measures:

| procedure | mean coverage | min | max | cells that ever return a negative lower endpoint |
|---|---|---|---|---|
| **BCa, B=1000 — what was published** | **81.3%** | 77.2% | 84.5% | 0 of 42 |
| BCa, B=20000 | **81.5%** | 77.8% | 84.7% | 0 of 42 |
| bootstrap-t on log(mean), B=20000 | **92.2%** | 89.0% | 95.0% | 0 of 42 |
| bootstrap-t on the raw mean, B=20000 | **92.2%** | 88.8% | 94.9% | 42 of 42 |
| **Student-t on log(mean), delta SE — what ships now** | **92.0%** | 88.3% | 94.7% | 0 of 42 |

Three things follow from that table, and only the first was in the original finding.

1. **The deficit is small-`g`, not Monte Carlo.** A domain has 5 to 7 windows. Raising
   B twentyfold moves coverage from 81.3% to 81.5% — nothing. The intervals were not
   noisy, they were the wrong shape.
2. **It fails in the harmful direction.** Truth lands *above* the interval far more
   often than below on every one of the 42 cells, so the endpoints systematically
   understate divergence. The practical consequence is false **separations**: a reader
   using two per-domain intervals to say "legal differs from code" was wrong about one
   time in five, not one in twenty.
3. **Bootstrap-t on log — the fix the review recommended — is not publishable here.**
   Its coverage is right, but on the real `k8-8bpw-stream / clean17 / axis2_legal` cell
   (five ordinary windows, cv 0.47) it returns an upper endpoint of **0.187 nats around
   a mean of 0.0103**, eighteen times the estimate, because resamples that draw four
   copies of one window collapse the studentizing denominator. Three of the 42 cells
   exceed 10x. Replacing an interval that is too narrow with one that is absurd is not
   a correction.

Separately, every domain was bootstrapped from the **same** seed (STAT-17), so at equal
window counts the strata drew byte-identical resample index streams and shared their
Monte-Carlo error — arbitrarily, since it pairs domain A's k-th window with domain B's
k-th, which are unrelated windows.

### What changed

`interval_method: "delta_t_log"` — a Student-t interval on `log(mean)` with the
delta-method SE, exponentiated back. It is:

* **calibrated**: 92.0% measured against 81.3% for what it replaces;
* **non-negative by construction**, unlike bootstrap-t on the raw mean, which puts a
  negative lower bound on a KL divergence on all 42 cells at least once in simulation;
* **bounded**: the widest published upper endpoint is 3.0x its own estimate;
* **free of any resample stream**, which retires STAT-17 rather than mitigating it. There
  is no seed to share and no seed noise to argue about. `bootstrap_seed` and
  `bootstrap_b` are published as `null`, and `df` and `t_critical` are published instead,
  so any reader can re-derive both endpoints by hand from `mean` and the window means.

And the part that is the actual fix: **every cell now carries `coverage_measured`**,
stating what it delivers (92.0% mean, 88.3%-94.7% across cells) against the nominal
95%. STAT-01 was never that the endpoints were computed wrongly — they reproduce
`scipy.stats.bootstrap(method='BCa')` to within Monte-Carlo error. It was that the row
claimed a level nobody had ever measured for it. 92% is not 95%, and at five windows
nothing is; the row says so now instead of implying otherwise.

The **panel-level** block is deliberately unchanged: 25 (or 17) windows is not the
small-`g` regime, its BCa endpoints are the joint standard's interop surface, and
`bin/jointstd/fixtures/brandonmusic-known-answer.json` pins four external panels'
endpoints as a known-answer test. It now states its own measured coverage, **90.2%**
(88.2%-91.6%), as a caveat rather than a method change.

### What did NOT change — recomputed, not asserted

`registry/tools/reseed_delta.py` diffs the old and new data field by field:

```
headline metric.value changed                       : 0
top-level uncertainty numbers changed               : 0
by_domain mean / se_clustered / positions changed   : 0
by_domain CI endpoints changed                      : 84
worst relative move                                 : 40.2441%
```

"top-level uncertainty numbers" is every numeric field of `uncertainty`:
`ci95_low`, `ci95_high`, `se_clustered`, `se_naive`, `deff`, `sigma_run`,
`sigma_run_runs`, `se_total`, `clusters`, `samples`, `bootstrap_b`, `bootstrap_seed`.
No headline KLD moved, no top-level interval moved, no domain **mean** moved, no
`se_clustered` moved. Only the per-domain interval endpoints did, which is the change.

The pre-reseed endpoints are recorded verbatim in
`registry/protocol/coverage/pre-reseed-by-domain-endpoints.json`, and
`registry/tools/selftest_stat01_reseed.py` T4 **regenerates all 42 of them bit-for-bit**
from the current tree via `domain_table(interval="bca", seed=20260829, b=1000)`. A
change to published numbers that cannot reproduce the numbers it replaced is a
replacement nobody can audit.

### Every endpoint that moved

84 endpoints across 42 cells in 12 rows. `g` is the window count; `coverage` is the
measured coverage of the NEW interval on that cell.

| row | domain | g | endpoint | old | new | move | coverage |
|---|---|---|---:|---|---|---:|---:|
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | high | 0.01470304709955 | 0.0206201603723462 | 40.24% | 91.5% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | high | 0.0146455765393629 | 0.0204438851774009 | 39.59% | 91.6% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | high | 0.0142031683941756 | 0.0194981396999935 | 37.28% | 91.8% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | high | 0.0127606142875556 | 0.0174256033531627 | 36.56% | 92.0% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | high | 0.0274139026615211 | 0.037301408074051 | 36.07% | 92.3% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis3_code_agentic | 5 | high | 0.018829245106684 | 0.0255960024198146 | 35.94% | 93.0% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis1_general | 7 | high | 0.0632458995868885 | 0.0857221564544668 | 35.54% | 88.5% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis1_general | 7 | high | 0.0632458995868885 | 0.0857221564544668 | 35.54% | 88.3% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis2_legal | 5 | high | 0.0140421676005723 | 0.018553471404305 | 32.13% | 93.2% |
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis2_legal | 5 | high | 0.0160690631627848 | 0.020908489318015 | 30.12% | 94.0% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis2_legal | 5 | high | 0.0160678085083828 | 0.0208746570858632 | 29.92% | 93.8% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis1_general | 7 | high | 0.0171856256903807 | 0.0221783671088039 | 29.05% | 89.4% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis1_general | 7 | high | 0.0171856256903807 | 0.0221783671088039 | 29.05% | 88.9% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis2_legal | 5 | high | 0.0134894053090631 | 0.0172718130296169 | 28.04% | 94.7% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis1_general | 7 | high | 0.0308080190222817 | 0.0394312730584889 | 27.99% | 90.8% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis1_general | 7 | high | 0.0308080190222817 | 0.0394312730584889 | 27.99% | 89.6% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis2_legal | 5 | high | 0.0312770895245161 | 0.0400159742932373 | 27.94% | 93.1% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis1_general | 7 | high | 0.0169481966706246 | 0.0216298677947107 | 27.62% | 90.5% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis1_general | 7 | high | 0.0169481966706246 | 0.0216298677947107 | 27.62% | 89.8% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis1_general | 7 | high | 0.0179698106065556 | 0.0228683581346213 | 27.26% | 89.6% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis1_general | 7 | high | 0.0179698106065556 | 0.0228683581346213 | 27.26% | 90.4% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis1_general | 7 | high | 0.0180539141657523 | 0.0228515253188578 | 26.57% | 90.7% |
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis1_general | 7 | high | 0.0180539141657523 | 0.0228515253188578 | 26.57% | 89.2% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis3_code_agentic | 6 | high | 0.0158980178094747 | 0.0200966151421894 | 26.41% | 92.0% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis3_code_agentic | 6 | high | 0.0157773248722071 | 0.0199358083469868 | 26.36% | 92.5% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis2_legal | 6 | high | 0.0333109277622976 | 0.0419119587833581 | 25.82% | 92.8% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis1_general | 7 | low | 0.0130863484603428 | 0.00973269656644206 | 25.63% | 88.5% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis1_general | 7 | low | 0.0130863484603428 | 0.00973269656644206 | 25.63% | 88.3% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis3_code_agentic | 6 | high | 0.0266172611807926 | 0.0333251869282183 | 25.20% | 92.9% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis3_code_agentic | 6 | high | 0.013400582816622 | 0.0167357696427708 | 24.89% | 92.4% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis3_code_agentic | 6 | high | 0.015295580879197 | 0.0190970400492564 | 24.85% | 91.5% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis2_legal | 6 | high | 0.0229641193662481 | 0.0285225875864488 | 24.21% | 93.5% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis2_legal | 6 | high | 0.0231152816895481 | 0.0284980850697114 | 23.29% | 92.7% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis2_legal | 6 | high | 0.0215529348993126 | 0.0264933054464872 | 22.92% | 92.4% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis2_legal | 6 | low | 0.00868958577767956 | 0.00679478053774629 | 21.81% | 93.1% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis3_code_agentic | 6 | low | 0.01292955316766 | 0.0102603967319488 | 20.64% | 91.7% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis2_legal | 6 | low | 0.00948764630182324 | 0.00755708411059913 | 20.35% | 92.7% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis2_legal | 6 | low | 0.00945059955210543 | 0.0075326177293666 | 20.29% | 93.5% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis2_legal | 5 | low | 0.0133782494588542 | 0.0106798117322309 | 20.17% | 93.1% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis2_legal | 6 | high | 0.0201555253222197 | 0.0241898758275795 | 20.02% | 93.1% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis4_reasoning_termination | 6 | high | 0.0194574823056433 | 0.0233491876734253 | 20.00% | 94.0% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis4_reasoning_termination | 6 | high | 0.0194687742946016 | 0.0233437418760391 | 19.90% | 93.9% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis2_legal | 6 | low | 0.00846119653533074 | 0.00677727400218751 | 19.90% | 92.4% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis2_legal | 5 | high | 0.03054002779953 | 0.0363207634599982 | 18.93% | 94.5% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis4_reasoning_termination | 6 | high | 0.0231959596843492 | 0.0275411013659124 | 18.73% | 93.9% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis4_reasoning_termination | 6 | high | 0.0193313666457206 | 0.022914734711037 | 18.54% | 93.8% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis4_reasoning_termination | 6 | high | 0.0234153924479643 | 0.0277223326623028 | 18.39% | 93.8% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis4_reasoning_termination | 6 | high | 0.0199229260934004 | 0.0235430703943109 | 18.17% | 93.8% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis3_code_agentic | 6 | high | 0.0336722869646146 | 0.0396634983133733 | 17.79% | 91.7% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis2_legal | 6 | low | 0.0206921714436851 | 0.0171957936505267 | 16.90% | 94.2% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis2_legal | 6 | high | 0.0381768041762928 | 0.044325858860981 | 16.11% | 94.2% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis2_legal | 5 | low | 0.00748432508780676 | 0.00630204457649469 | 15.80% | 93.8% |
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis2_legal | 5 | low | 0.00741764093831472 | 0.00626885769138277 | 15.49% | 94.0% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis2_legal | 5 | low | 0.00676907665294915 | 0.00574499795652428 | 15.13% | 93.2% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis2_legal | 6 | low | 0.016162669006654 | 0.0137297280156348 | 15.05% | 92.8% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis2_legal | 5 | low | 0.0176522002832597 | 0.0152153936958678 | 13.80% | 94.5% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis2_legal | 5 | low | 0.00680953316828439 | 0.00588916435248655 | 13.52% | 94.7% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis1_general | 7 | low | 0.0109185877361317 | 0.00971149355609811 | 11.06% | 90.8% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis1_general | 7 | low | 0.0109185877361317 | 0.00971149355609811 | 11.06% | 89.6% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | low | 0.0131380905015073 | 0.0116875692580655 | 11.04% | 92.3% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis4_reasoning_termination | 6 | low | 0.0101183551441137 | 0.00909529600277872 | 10.11% | 93.8% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis3_code_agentic | 6 | low | 0.00752306464090629 | 0.00827594725033383 | 10.01% | 92.5% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis3_code_agentic | 6 | low | 0.0075841382045903 | 0.0082958504323992 | 9.38% | 92.0% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | low | 0.00633166765827416 | 0.00574299349246938 | 9.30% | 91.8% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis4_reasoning_termination | 6 | low | 0.0106904621661296 | 0.00975244476698162 | 8.77% | 93.8% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis1_general | 7 | low | 0.00652693605339505 | 0.00603112547641908 | 7.60% | 90.7% |
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis1_general | 7 | low | 0.00652693605339505 | 0.00603112547641908 | 7.60% | 89.2% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis1_general | 7 | low | 0.00652904155193087 | 0.00604844918926325 | 7.36% | 89.6% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis1_general | 7 | low | 0.00652904155193087 | 0.00604844918926325 | 7.36% | 90.4% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis3_code_agentic | 6 | low | 0.00630852752896403 | 0.0065913357843431 | 4.48% | 92.4% |
| `glm53.official-fp8.brandonmusic-final25.crossstack` | axis4_reasoning_termination | 6 | low | 0.0134197293325685 | 0.0128921149086502 | 3.93% | 93.8% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis4_reasoning_termination | 6 | low | 0.0131347396422671 | 0.0136421759638968 | 3.86% | 93.9% |
| `glm53.k8-8bpw-stream.brandonmusic-final25` | axis1_general | 7 | low | 0.00597953663624455 | 0.00582592475828369 | 2.57% | 89.4% |
| `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` | axis1_general | 7 | low | 0.00597953663624455 | 0.00582592475828369 | 2.57% | 88.9% |
| `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | low | 0.00661136111139256 | 0.00676876692567102 | 2.38% | 91.6% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis3_code_agentic | 6 | low | 0.00720143663925755 | 0.00704898719616047 | 2.12% | 91.5% |
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | axis3_code_agentic | 6 | low | 0.0137852150441904 | 0.0140267052829716 | 1.75% | 92.9% |
| `glm53.bf16-replay-floor.brandonmusic-final25` | axis1_general | 7 | low | 0.00618281829293444 | 0.00608950456838462 | 1.51% | 90.5% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis1_general | 7 | low | 0.00618281829293444 | 0.00608950456838462 | 1.51% | 89.8% |
| `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` | axis3_code_agentic | 5 | low | 0.00937378879574877 | 0.00925718200021231 | 1.24% | 93.0% |
| `glm53.k6-6bpw.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | low | 0.00669197221005582 | 0.0067738265150795 | 1.22% | 91.5% |
| `glm53.k6-6bpw.brandonmusic-final25` | axis4_reasoning_termination | 6 | low | 0.0109347444662552 | 0.0108499905272563 | 0.78% | 93.9% |
| `glm53.k6-6bpw-stream.brandonmusic-final25` | axis4_reasoning_termination | 6 | low | 0.0108991642999101 | 0.0108333067519636 | 0.60% | 94.0% |
| `glm53.bf16-replay-floor.brandonmusic-final25.clean17` | axis3_code_agentic | 5 | low | 0.00582858230646766 | 0.00579517332734868 | 0.57% | 92.0% |

### The two published analysis surfaces that carried the old endpoints

The registry is not the only place these 42 intervals appear, and a correction that
leaves a second published copy disagreeing with the first is not a correction.

* **`docs/joint-standard/analysis/*.json`** (12 analysis receipts, GitHub) — regenerated
  by `bin/joint-standard analyze`, which reads `domain_table` and therefore picks the new
  interval up automatically. Verified field by field: nothing moved outside `by_domain`'s
  interval fields, plus two keys that had drifted behind earlier work
  (`scope.window_sizes_declared`, `se_quadrature.ci95_total`). Every mean, SE, design
  effect, panel bootstrap and sigma_run is byte-identical.
* **`reports/clean-scope-recompute.json`** (GitHub **and** the
  `GLM-5.3-Flash-fidelity-suite-v1` dataset) — regenerated. Diffed leaf by leaf against
  the published copy: exactly 126 leaves changed, all of them interval fields (42 `ci95`,
  42 `ci95_bca` now explicitly `null`, 42 new `interval_method`). **Zero** changes outside
  them — every scope mean, every scope delta, every attributable subtraction is unchanged.
  The emitter now names the interval it emits rather than writing a Student-t interval
  into a key called `ci95_bca`, which would be the same class of mislabel this whole
  section exists to remove.

### How to verify it yourself

```bash
cd registry
make check                     # 0 errors, 84 selftests, 440 joint checks
make reseed-check              # the rows are still a function of their receipts
make stat-selftest             # the 22 assertions behind this section
make coverage                  # regenerate the coverage record (minutes)
python3 tools/reseed_delta.py OLD.jsonl data/measurements.jsonl
```


---

## 4. `malaiwah/quant-fidelity-registry` — harness identity on every row

**Published 2026-08-30. Additive: no measured value changed.**

### What was wrong

Every number in this registry is a function of some code, and no row said *which*. That
sounds like a documentation gap until the day a defect is found in the estimator — and
then it is the difference between "these 12 rows predate the fix" and "all 72 rows are
now under suspicion and none of them can be cleared". The peer review that produced §3
also left roughly 130 lower-severity findings open with the honest note that none is
known to move a published number and none has been individually cleared either. Without
a code stamp that liability floats over every row forever, and every future one.

### What was added

A `harness` block on every measurement row (`schema/common.schema.json#/$defs/harness`,
implementation and reasoning in `registry/tools/harness_id.py`):

| field | what it is |
|---|---|
| `code_digests[]` | `{role, path, sha256}` for every file on the path from published inputs to the published number — read from the **bytes**, never transcribed |
| `tool_versions` | python / numpy / torch, as they were when the number was produced |
| `repository` | url, commit, `commit_role`, `dirty` — a human pointer |
| `harness_id` | `harness--` + sha256 over `{boundary, code_digests, tool_versions}`, first 16 hex |
| `covers[]` | which parts of *this row* the stamp attests |
| `recorded` | `false` is a legal, honest answer |

**Where the boundary is drawn, and why.** A digest over the whole repository is useless:
it changes on a docs edit, so the field churns for reasons that cannot affect a number
and stops carrying information. A digest over the estimator alone is unsafe: every BCa
endpoint calls `chi2.norm_ppf`, so a one-ULP change in `chi2.py` moves published numbers
while `stats.py` is byte-identical, and the stamp would say "same code" about two
different numbers. The boundary is the **computational closure** — the estimator, its
numerical support, the protocol stamper, the enrichment layer, and the coverage simulator,
because `coverage_measured` is a published number too. It is *not* `seed_registry.py`,
which assembles rows and changes whenever an unrelated row is added.

The boundary errs deliberately toward **over-sensitivity**, and the guarantee is stated
one-way everywhere it appears: equal `harness_id` means identical code; a differing id
means read `code_digests`, whose roles name exactly what moved. A false alarm costs a
reader one diff; a missed change costs them a wrong comparison.

**What is not in the id.** `repository.commit` is recorded and excluded — a commit sha
changes on a docs edit, and a commit cannot be recorded by the change that introduces it.
`tool_versions` *is* in the id: CPython 3.12 switched builtin `sum()` to Neumaier
summation and moved this project's reductions in the last ULP, so an interpreter is part
of the estimator. `tool_versions` and `repository.commit` are pinned **literals**, not
readings of whoever runs `make check` today, because a harness block is a historical
record of the run that produced the number — and because `make reseed-check` has to give
the same answer on 3.9 and 3.12.

### What was grandfathered, and how it is marked

All 72 pre-existing rows are listed in `schema/harness-grandfather.json`, which states
that it is **frozen and never appended to** — an allowlist that grows means nothing.
Invariant HARN-001 requires a recorded harness covering `metric.value` on every row *not*
in it, so a new row that cannot say which code produced its number is refused.

Of those 72:

* **6 rows are now fully attributed.** The `.clean17` rows' headline *is* computed here —
  `joint_enrich._clean_row` re-reduces the published per-window means over the 17-window
  scope — so their harness covers `metric.value` and they carry no `harness_unrecorded`
  disclosure. Their inputs are receipts, cited in `provenance.sources`; the harness is
  the code identity, the sources are the data identity.
* **6 rows are partially attributed.** The `panel25` siblings' `uncertainty`, `by_domain`
  and `protocol` blocks are derived here; their `metric.value` came off a GPU and is only
  *checked* here. `covers` says exactly that. Claiming `metric.value` because the check
  passes would be the precise failure the block exists to prevent.
* **66 rows carry `harness_unrecorded`**, an `info` disclosure on the row itself — a
  consumer pulling one JSONL line does not read the schema directory (HARN-004).

**Digests were not reconstructed for historical rows.** Today's files are not the files
that produced them, and a plausible-looking digest set would be a fabricated provenance
record. `recorded: false` with a null id is the honest shape, and HARN-002 refuses an
unrecorded harness that carries digests anyway.

Nothing was retroactively invalidated. Those receipts are still hashed and those values
still reproduce; what is missing is attribution, and it is now recorded as missing.

`registry_add.py` builds the block from a submission's `produced_by` (entrypoint,
`entrypoint_sha256`, revision, dependencies) — which was already required and was already
a harness in all but name — or from `--harness-manifest`. It stamps only what it can
attest: `registry_add` did not compute `metric.value`, the measuring run did.

---

## 5. `malaiwah/quant-fidelity-registry` — provenance assertions need sources

**Published 2026-08-30. Additive: no measured value changed.**

### What was wrong

A metric row in this registry has always required a hashed receipt. An **assertion** —
a claim about how an artifact was produced, or where it came from — required nothing, and
the validator had nothing to object to. So two mechanism claims about the SIQ-Fruit
artifacts reached two published dataset cards and two registry rows with no structured
source at all, and passed validation cleanly:

* *"Every tensor is bf16 and comes from the trained checkpoint by a direct cast … no
  dequantization step exists anywhere in the exporter"* — which decides the artifact's
  `reference_kind`, which decides whether a KL number measured against it means what it
  says;
* *"The exporter copies the [NVFP4/modelopt] block from the parent GLM-5.2 config rather
  than authoring it"* — the difference between "the producer mislabelled this" and "a
  field was copied forward".

Both claims are true, and both were re-read against the source before being written. That
is exactly the problem: the process that produced them was diligence, not a rule, and a
rule is what survives the next author.

### What was added

`disclosure` gains `asserts_provenance` and its own `sources[]` (with an optional `lines`
anchor on `source`). Three invariants:

* **PROV-014** — `asserts_provenance: true` requires a non-empty `sources`.
* **PROV-015** — every source cited by a provenance assertion must be **pinned**: a 40-hex
  commit in the path, a revision, or a sha256. `/blob/main/`, `/resolve/main/` and
  `/tree/main/` are refused outright. This is a lesson already paid for here: cite by
  COMMIT, never by branch, because line numbers move and a citation that quietly stops
  pointing at what it claimed still reads as evidence.
* **PROV-016** — on an artifact or model record, a disclosure whose text reasons from a
  source-code file must set `asserts_provenance`. Without this the marker is opt-in and
  PROV-014 is decorative: the failure is precisely an author writing a mechanism claim and
  not thinking of it as one.

Both Fruit disclosures now carry line-anchored citations pinned to
`75b0840fe2ff42181945fab94bd4a81286114422` in `proxy-fruit`: `export_fruit.py` 262-266
and 317-333 for the direct cast, 373-378 for the config inheritance, plus the
revision-pinned `tier_bitmap.json` and the producer's own review at 210-230 as independent
corroboration.

---

## Not published, deliberately

Nothing from `docs/REVIEW-DEFERRED.md` is now held back for an operator decision on
published numbers. The remaining deferrals in that file are blocked by **file ownership**
(a live measurement campaign holds `bin/measure_cloud.py`), not by publication risk.
