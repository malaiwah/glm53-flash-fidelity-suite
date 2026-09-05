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

## 6. `malaiwah/quant-fidelity-registry` — 37 Qwen rows relabeled `float32_reduce_legacy` (P1-06)

**Published 2026-08-31, from the independent peer review's P1-06.** This changes a
LABEL and therefore a comparability key on 37 rows. No measured value moved, no
receipt was edited, no id changed.

### What was wrong

The producer behind every Qwen3.8-27B row on the `qwen38-kld-ladder` and
`qwen38-gguf-cross-engine` pipelines — `tools/fidelity.py`'s replay comparator —
computed logits, normalizers, probabilities and the **vocabulary sum in float32**
and cast the already-reduced result to float64. Its receipts declared
`accumulation: float64`, and the 37 registry rows seeded from them inherited
`estimator.accumulation_dtype: "float64"` — a false precision contract.

Demonstrated, not hypothesized: on synthetic near-equal distributions over a
50,000-entry vocabulary, the float32 reduction returns **negative** per-token
"KL" around `-8e-7` where the true float64 value is `~+2e-8`
(`bin/selftest_fidelity_reducer.py` reproduces it deterministically; KL is
non-negative, so the entire negative excursion is estimator error). No published
row was observed with a negative headline value, and the ladder's published means
sit at `1e-3`–`1e-1` — three to five orders above that error scale — but "the
error is probably small" is not the same claim as "accumulated in float64".

### What changed

* `tools/fidelity.py` now casts to float64 **before** log-sum-exp, probability,
  product and the vocabulary reduction, in both `context_metrics` and
  `qualification_metrics`, and refuses non-finite or materially negative
  per-token KL instead of reporting it. Known-answer tests pin the fixed reducer
  to a dense float64 reference within 1e-12 on exactly the construction where
  the float32 path goes negative.
* The 37 rows are relabeled `accumulation_dtype: "float32_reduce_legacy"` (new
  schema enum value), each with a `fp32_vocab_reduction` disclosure
  (`affects_comparability: true`). Because `accumulation_dtype` is one of the
  seven comparability-key fields, the relabel moves every one of these rows to a
  new `comparability.key` **by construction** — that is the system working: they
  remain rankable against each other (same reducer, same panels) and are no
  longer in any group a true-float64 row could join.
* The two pipeline records correct `numerics.accumulation_dtype` from `fp64` to
  `fp32` and carry the same disclosure.
* The 3 `qwen38-hf.*` rows on the `fidelity-dataset-hf.rtxpro6000` pipeline are
  untouched: their producer (`bin/fidelity/dscompare.py`) normalizes and
  accumulates in float64 and keeps its `float64` label.

### Sensitivity, measured at the operating point

Simulated at the ladder's real geometry — vocabulary 248,320, chunk 24,832,
float32 logits, per-token KL calibrated to the published levels (5e-4 GGUF
floor, 5e-3 fp8, 2e-2, 1e-1), two base-distribution shapes, 2,048 tokens per
cell, seed 20260831:

| quantity | measured |
|---|---|
| per-token fp32-reduction error, worst | ~2.2e-6 nats |
| effect on a MEAN over tokens (the published statistic) | ~1e-8 nats (signs mixed; cancels) |
| relative error on the mean at the floor level (5e-4) | ≤ 0.13% |
| relative error on the mean at ladder levels (≥ 5e-3) | ≤ 0.014% |
| smallest adjacent-row gap in any relabeled group | 5.07e-5 nats (turboderp-6bpw vs k6-parity, 1m panel) |

The mean-level bias sits three orders of magnitude below the tightest published
gap, so **no published Qwen mean moves at its quoted precision and no ordering
changes** — consistent with the review's own reading ("a false precision
contract, not that all published Qwen rankings are numerically reversed"). The
negative-KL failure needs per-token KL near 1e-7, which no published row
approaches.

### What was NOT done

The 37 values were **not** re-run on the real captures. The sensitivity table
above is synthetic (calibrated to the published levels, not replayed from the
hidden states); a true rerun needs the private Qwen3.8-27B captures replayed on
a rented GPU. Until then the honest state is the relabel plus this disclosure,
not silently "corrected" numbers. When any row is re-measured with the fixed
reducer it will publish under a `float64` key as a new row, never by
overwriting these.

---

## 7. K6/K8 paired inference on the Brandon panel — the independent unit is the source document (P1-15 + P1-16)

**Published 2026-08-31.** This changes the INTERPRETATION of published
statistics, not any measured value. Independent peer review, verified by
recomputation from the committed per-window series before anything was changed.

### What was wrong

Two defects, one confounded pair of receipts:

1. **Pseudoreplication (P1-15).** The sealed 25-window panel derives from
   **four source documents** — one per axis, 7/6/6/6 windows
   (`registry/protocol/window-selection.brandonmusic-final25.json`,
   `per_window[].document_id`); clean17 holds three (7/5/5). The published
   paired K6−K8 analysis resampled and sign-tested **windows** as exchangeable
   units: mean diff 0.001339, BCa [+0.000695, +0.002330], sign test
   **p = 0.004077** (clean17: 0.000848, [+0.000153, +0.001573], **p = 0.0490**).
   Windows cut from one document share its topic, style and register; splitting
   the same four documents into more windows shrinks that interval without
   adding independent textual evidence. At the document level the exact
   two-sided sign test is **p = 0.125** (4 of 4 positive) and **0.25**
   (3 of 3); an equal-document t interval is [+0.000049, +0.002710] full and
   [−0.000256, +0.002078] clean17.
2. **Mixed-lane pairing (P1-16).** The published K6−K8 receipts paired the
   **sealed** K6 series against the **streaming** K8 series, and the loader
   (`bin/joint_standard.py cmd_paired`) reduced each input to `{window: mean}`,
   discarding lane and every other contract field. Of the five published
   pairings only FP8-vs-BF16floor was same-lane.

### What changed

* Every `docs/joint-standard/analysis/paired.*.json` regenerated (same seeds,
  same B; every previously published number reproduces bit-for-bit) with:
  `document_level` — per-document means, exact document sign test, equal-document
  t interval, and the statement that it is the only inferential block;
  `window_stats_are` — the window-level statistics relabelled descriptive of
  this fixed panel; `contract_a`/`contract_b`/`cross_lane` — each side's lane,
  recorded, with the K6−K8 receipt carrying the measured streaming↔sealed
  bridge verbatim.
* `bin/joint_standard.py paired` now **refuses a mixed-lane contrast** without
  an explicit `--bridge` statement, and refuses recorded reference/estimator
  mismatches outright. `--document-map` carries window→document provenance;
  without it a receipt labels itself descriptive-only.
* New same-lane receipt `paired.K6stream-vs-K8.*`: mean diff **0.001331** full
  / **0.000847** clean17 — the ordering is lane-robust.
* `docs/PROTOCOL-ALIGNMENT.md` §4.4 carries the correction banner; the K6/K8
  model cards are regenerated with the same relabelling.

### What survives, and what is withdrawn

**Survives:** the K6/K8 panel means themselves (bitwise-evidenced, untouched);
the K8-better-than-K6 ordering *on this panel* — all four document means
positive, same-lane recompute preserves it; every per-window array and BCa
endpoint as a **description of these exact windows**.

**Withdrawn:** the reading of window-level sign-test p-values (0.0041 / 0.049)
and BCa intervals as population inference; any implication that the 25-window
panel supplies 25 independent observations. A population claim about these two
quantizers awaits a panel with many independent source documents per domain.

### What did NOT change

No `metric.value`, no registry row, no receipt digest of any measurement. The
pre-correction paired receipts are superseded in place by regeneration; their
statistical fields are bit-identical, so any third party who pinned a number
still finds it — under a label that now says what it is.

---

## 8. "quantization-attributable" renamed `excess_over_control`, and the 2.52x ratio withdrawn (P1-05)

**Published 2026-08-31.** A terminology-and-claim correction: no measured value
moves, and `reseed_delta` over the full collection confirms it (0 metric
values, 0 uncertainty numbers, 0 domain endpoints changed; the only moved
fields are `comparability.bias.detail`, two `notes`, and two disclosure
strings, on 11 rows).

### What was wrong

The registry, the model cards, `WHAT-WE-MEASURE.md`, `llms.txt` and
`engines/BF16-FLOOR.md` called the floor-subtracted number
"quantization-attributable error". Algebraically the quantity is
`D(P‖Q_quant) − D(P‖Q_control) = E_P[log Q_control − log Q_quant]`: not itself
a divergence, capable of being negative, and equal to the quantization effect
only if the two paths differ by nothing else — an assumption this project's
own pipeline (~24%) and hardware (2.97e-4 nats) studies show is non-trivial.
"Attributable" asserted causality the design does not isolate.

Worse, two of these residuals were published as a ratio — **"K8's quantization
error is 2.52x smaller than K6's"** — with no uncertainty. A ratio of two
small residuals magnifies control error; the same data read 1.11x in raw
means.

### What changed

* Name, everywhere user-facing: `excess_over_control`. Rendered registry
  column "Excess over control (nats)"; card metric type
  `kl_divergence_excess_over_control` and `x_fidelity` field
  `excess_over_control` (spec, schema, generator, validator, examples);
  `WHAT-WE-MEASURE.md` §4, `llms.txt` Rules 3–4, `engines/BF16-FLOOR.md`,
  `registry/README.head.md`.
* The **2.52x ratio is withdrawn** wherever it appeared without uncertainty.
  The two residuals themselves (K6-stream 0.002209, K8-stream 0.000878 nats,
  panel25, streaming lane) still stand, printed beside their raw values with
  the floor named.
* The 11 registry rows whose `bias.detail` / `notes` carried the old term now
  carry the corrected sentence, which names the old term and the rename date
  inline, so a reader landing on the row sees both. Old wording, verbatim, for
  the record: *"…netting it out gives an estimated quantization-attributable
  error of R nats here — an estimate, not an identity, because KL is not
  additive, and it is only meaningful because both terms are small and share
  the same reference and lane."* New emissions from `registry_add.py` use the
  new sentence.
* No registry id, receipt, digest, or `metric.value` changed. The
  comparability keys are untouched.

---

## 9. One platform-dependent clustered-SE last digit (STAT-20)

**Corrected in source 2026-08-31.** This changes one derived uncertainty value,
not a measured KLD, interval endpoint, rank, comparability key, or receipt.

### What was wrong

`se_from_window_summaries` evaluated the last step as
`math.sqrt(scale * ssq) / n`. That rounds at `sqrt` and again at division.
For the persisted binary64 window means, macOS and Linux landed on opposite
sides of a 15-significant-digit publication boundary. An untouched checkout
therefore reported `RESEED DRIFT` even though its receipt inputs were identical.

### What changed

The historical binary64 residual and `math.fsum` evaluation order is retained.
Only the final square-root/division expression is evaluated at high precision
and converted to binary64 once. `STAT-20` pins both real boundary cases.

Exactly one published field moves:

| row | domain | field | old | new | delta (new − old) |
|---|---|---|---:|---:|---:|
| `glm53.brandonmusic-4bpw.brandonmusic-final25` | `axis2_legal` | `by_domain[].se_clustered` | 0.00508491797330785 | 0.00508491797330784 | −9.54e−18 |

The relative change is $1.88\times10^{-15}$ ($1.88\times10^{-13}\%$). No
headline `metric.value`, confidence interval, domain mean, ordering, or
scientific conclusion changes. Harness code digests and harness ids on the
derived GLM rows also move, as required: they now identify the corrected
statistics implementation instead of claiming the old bytes produced them.

---

## 10. Registry receipt links no longer point at one workstation

**Corrected in source 2026-08-31.** This changes provenance links only. No
measured value, uncertainty, interval, rank, comparability key, receipt digest,
or registry id changes.

### What was wrong

Sixty-eight registry records contained 76 `uri` fields that named absolute
paths on the maintainer's former workstation. Thirty-seven Qwen measurement
rows also carried a public mirror beside the inaccessible local source, so the
registry both published a dead path and duplicated the usable evidence.
`seed_registry.py` depended on the same uncommitted directory, which made
`make reseed-check` fail on a clean clone.

### What changed

* The 37 Qwen receipts that supply measurement values are committed under
  `registry/protocol/qwen38-receipts-public-8558b8c/`. Their manifest binds
  every filename, byte count and SHA-256 to public repository commit
  `8558b8ca3bba028f852f4b53167b79b4cd552f93`; the seeder hashes the exact
  bytes it parses and refuses a missing or changed file.
* All 74 Qwen source-URI occurrences now point directly at that immutable
  public commit. The cross-engine comparator is labelled contextual rather
  than represented as the byte source for `metric.value`.
* The two GLM suite-manifest occurrences now cite the committed
  `suite/suite-manifest.json`; its existing SHA-256 is unchanged.
* The 37 redundant local-plus-mirror source pairs are each one direct public
  source. `PROV-017` makes the validator refuse POSIX, `file:`, UNC,
  home-relative and Windows drive-absolute host paths in every published
  `uri`; the selftest mutates a clean record to prove the gate fires.

Collection hashes and the two repository copies of the generated model-card
registry-snapshot hashes move because their provenance bytes changed. They are
metadata digests, not scientific measurements. Publishing the regenerated
cards to the Hub remains a separate, permissioned action.

---

## 11. Five GLM-5.3 trellis scopes, drowzeys' non-routed provenance, and intervals on the six GLM-5.3 rows (science review S1-1, S1-3, S2-3, S3)

**Published 2026-09-05.** Suite commit `2ffb1cb9fa367de2b5cd854636df8a4a9acc7c1c`; registry mirror
`malaiwah/quant-fidelity-registry` @ `5304f3e8f6358a6b7a99ccc58a1871e9a6e0e3f0` (previous `b58bca8293bb`);
one additive comment on each of the six Hub discussions (URLs below). **No measured
value moved**: every `metric.value`, top-1, percentile, strata mean and receipt digest is
byte-identical before and after. What changed is five `scope` blocks (and therefore
five `scope_digest` strings), six `uncertainty` blocks that were `none`, one
artifact's provenance prose, `estimator.logits_dtype` provenance and four caveat wordings.

### What was wrong

**Scope strings (S1-1).** `engines/tools/exl3_scope.py` / `fp8_scope.py` wrote
`treatment: quantized` for any tensor class that mixes storage formats on shared layers.
For the router that census is `75 x native:bf16@16 (mlp.gate.weight); 75 x native:fp32@32
(e_score_correction_bias)` -- every group native -- so all five trellis artifacts were
published as `moe.router=quantized:mixed`, and drowzeys additionally as
`attn.other=quantized:mixed` and `mtp=quantized:mixed` (its MTP experts are stored fp16
and are not trellis-quantized). The tool was fixed in `56ff020` (an all-native census is
`treatment native`); the rows were seeded and the discussion posts rendered before that.

**drowzeys provenance (S1-3).** Two committed records contradicted each other. The
artifact record said the fp16 attention/MLP/shared-expert tensors were "native,
unquantized, the BF16 release's values after an fp16 round trip"; the zero-pad evidence
(`glm53-exl3-tp-rank-and-zero-pad-parity.json`) and the `ZERO_PAD_METHOD` comment said the
same rows were `dequantize_block_fp8(zai-org/GLM-5.3@187fb9ff)` cast to fp16 -- the FP8
release. Settled from bytes, spend-free (`engines/tools/nonrouted_provenance.py`, evidence
`engines/tools/layer-outer-evidence/drowzeys-nonrouted-provenance.json`): the leading 576
rows of ten tensors -- `layers.0.self_attn.{q_a_proj,kv_a_proj_with_mqa,q_b_proj}`,
`layers.40.self_attn.{kv_b_proj,o_proj}`, `layers.{0,2,1}.mlp.{gate,up,down}_proj`,
`layers.{3,77}.mlp.shared_experts.{gate,down}_proj` -- range-read from
`drowzeys/keys-GLM-5.3-EXL3@ebf3c8bb`, `zai-org/GLM-5.3-BF16@304b8051` and
`zai-org/GLM-5.3@187fb9ff` (+ `_scale_inv`), are **bitwise equal to
`fp16(dequantize_block_fp8(fp8, scale, float32))` on every tensor (0 differing elements)
and not equal to `fp16(bf16_root)` (98-99 % of elements differ)**. The evidence and the
code comment were right; the artifact prose and the scope were wrong. The whole non-routed
path carries the FP8 release's 8-bit block quantization stored at 16 bits.

**Intervals (S2-3).** The six GLM-5.3 rows published 16-digit point values with
`uncertainty.method: none`, while the per-window means in their own receipts spread 17-26x
between the easiest and hardest window; `joint_enrich.py` could only enrich the Flash
panel25 series.

**Wording (S3).** "WEIGHT-ONLY, THEREFORE A LOWER BOUND" (FP8 and K4 rows) states a
bound a mean KL does not have; the pod receipt was cited as "same value" while differing by
up to 3.8e-9; `decode_payload_hf` was attributed to exllamav3 (it is
`engines/tools/exl3hf_surface.py`); the macro-mean note claimed exact equality where the
last bit differs by 2.8e-17; `estimator.logits_dtype` was a hard-set constant rather than a
value read from the receipt.

### What changed

| artifact | old `scope_digest` | new `scope_digest` |
|---|---|---|
| `wrldsuksgo2mars.glm-5.3-exl3-k4-v1` | `attn.o=quantized:fp8_e4m3@8|attn.other=quantized:mixed|attn.qkv=quantized:fp8_e4m3@8|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=quantized:fp8_e4m3@8|mlp.gate=quantized:fp8_e4m3@8|mlp.up=quantized:fp8_e4m3@8|moe.experts=quantized:exl3-mcg@4|moe.router=quantized:mixed|moe.shared_expert=quantized:fp8_e4m3@8|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` | `attn.o=quantized:fp8_e4m3@8|attn.other=quantized:mixed|attn.qkv=quantized:fp8_e4m3@8|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=quantized:fp8_e4m3@8|mlp.gate=quantized:fp8_e4m3@8|mlp.up=quantized:fp8_e4m3@8|moe.experts=quantized:exl3-mcg@4|moe.router=native:mixed|moe.shared_expert=quantized:fp8_e4m3@8|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` |
| `davidsyoung.glm-5.3-exl3-tr3-3.0bpw` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3|moe.router=quantized:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3|moe.router=native:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` |
| `davidsyoung.glm-5.3-exl3-tr3-3.25bpw` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3.25|moe.router=quantized:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3.25|moe.router=native:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` |
| `davidsyoung.glm-5.3-exl3-tr3-3.42bpw` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3.421875|moe.router=quantized:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` | `attn.o=native:bf16@16|attn.other=native:bf16@16|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@3.421875|moe.router=native:mixed|moe.shared_expert=native:bf16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` |
| `drowzeys.keys-glm-5.3-exl3` | `attn.o=native:fp16@16|attn.other=quantized:mixed|attn.qkv=native:fp16@16|embed_tokens=native:bf16@16|lm_head=native:fp16@16|mlp.down=native:fp16@16|mlp.gate=native:fp16@16|mlp.up=native:fp16@16|moe.experts=quantized:exl3-mcg@3|moe.experts=quantized:exl3-mul1@3|moe.router=quantized:mixed|moe.shared_expert=native:fp16@16|mtp=quantized:mixed|norm=native:bf16@16|head=native|kv=bf16` | `attn.o=quantized:fp8_e4m3@8|attn.other=native:mixed|attn.qkv=quantized:fp8_e4m3@8|embed_tokens=native:bf16@16|lm_head=native:fp16@16|mlp.down=quantized:fp8_e4m3@8|mlp.gate=quantized:fp8_e4m3@8|mlp.up=quantized:fp8_e4m3@8|moe.experts=quantized:exl3-mcg@3|moe.experts=quantized:exl3-mul1@3|moe.router=native:mixed|moe.shared_expert=quantized:fp8_e4m3@8|mtp=native:mixed|norm=native:bf16@16|head=native|kv=bf16` |

* The five scope files were re-authored from bytes by `exl3_scope.py --dtypes-from-hub`
  at HEAD; drowzeys' six non-routed classes were then rewritten by the new
  `engines/tools/scope_apply_provenance.py` from the evidence file (it refuses any class
  the evidence does not cover, and any non-16-bit-native row). Each corrected artifact
  carries a `scope_record_corrected` disclosure (asserts provenance; cites the scope file
  and the evidence by sha256 and the tool fix by commit) naming both digests.
* New invariant **SCOPE-011** (`registry/tools/registry_validate.py`,
  `registry/schema/invariants.json`): a `treatment=quantized` assignment whose note census
  is all `native:*` groups is an error. Its selftest fixtures are the OLD published
  strings: 7 findings on the pre-fix data, 0 after.
* `registry/tools/joint_enrich.py`: `SERIES` entries now declare a source
  (`per-window` file or a comparison receipt's `per_context`) and whether the Flash
  panel25 enrichment applies. The six GLM-5.3 rows get exactly the `uncertainty` block --
  window-block bootstrap, percentile + BCa, B=5000, seed 20260829, cluster-robust SE,
  25 clusters, 51,175 samples, sigma_run 0.0 from two bitwise-identical cold runs -- with
  coverage stated as nominal (no coverage simulation exists for corpus5x5-v1), plus the
  24-df t-interval and the paired adjacent-row ordering in the note. `make reseed-check`
  proves the rows are a function of the receipts.

| row | value | 95 % BCa (window-block) | 95 % t (24 df) | SE (clustered) | half-width / mean |
|---|---:|---|---|---:|---:|
| `fp8-dequantized.corpus5x5-v1` | 0.022305 | [0.01890, 0.02892] | [0.01739, 0.02722] | 0.00238 | 22 % |
| `exl3-k4-wrldsuksgo2mars.corpus5x5-v1` | 0.044804 | [0.03709, 0.05970] | [0.03376, 0.05585] | 0.00535 | 25 % |
| `exl3-tr3-3.42bpw-davidsyoung.corpus5x5-v1` | 0.062842 | [0.05101, 0.08207] | [0.04711, 0.07857] | 0.00762 | 25 % |
| `exl3-tr3-3.25bpw-davidsyoung.corpus5x5-v1` | 0.073059 | [0.05976, 0.09392] | [0.05558, 0.09054] | 0.00847 | 24 % |
| `exl3-tr3-3.0bpw-davidsyoung.corpus5x5-v1` | 0.083833 | [0.06846, 0.10931] | [0.06316, 0.10450] | 0.01002 | 25 % |
| `exl3-keys-drowzeys.corpus5x5-v1` | 0.102333 | [0.08395, 0.13351] | [0.07725, 0.12742] | 0.01215 | 25 % |

The t-intervals reproduce the review's §5 table to every printed digit; the BCa endpoints
sit above them by 0.0015-0.0067 on both ends (right-skewed window means).

| adjacent pair (higher − lower) | mean Δ | sd Δ | higher on | sign-test p |
|---|---:|---:|---:|---:|
| exl3-k4-wrldsuksgo2mars − fp8-dequantized | +0.022499 | 0.016845 | 24/25 | 1.55e-06 |
| exl3-tr3-3.42bpw-davidsyoung − exl3-k4-wrldsuksgo2mars | +0.018038 | 0.012639 | 25/25 | 5.96e-08 |
| exl3-tr3-3.25bpw-davidsyoung − exl3-tr3-3.42bpw-davidsyoung | +0.010218 | 0.006363 | 25/25 | 5.96e-08 |
| exl3-tr3-3.0bpw-davidsyoung − exl3-tr3-3.25bpw-davidsyoung | +0.010774 | 0.011687 | 22/25 | 0.000157 |
| exl3-keys-drowzeys − exl3-tr3-3.0bpw-davidsyoung | +0.018499 | 0.013137 | 25/25 | 5.96e-08 |

Every published adjacent ordering is supported by the paired windows; the one to hold
loosely is 3.25 vs 3.0 bpw (22/25). The drowzeys-vs-davidsyoung-3.0 gap (0.0185 nats,
25/25) now carries a caveat on the drowzeys row: it includes the FP8 release's non-expert
error (0.0223 nats on the FP8 row itself) and is not a codec-vs-codec comparison.

* `estimator.logits_dtype` on the seven GLM-5.3 rows is derived from the receipt's
  `comparator.replay_backend` (`numpy:cpu:float32` → `fp32`); the receipts' own
  `logits_dtype: bf16` (the capture dtype) is corrected forward by the comparator fix
  (`553d0c1`), sealed receipts untouched.
* Wordings: "expected to understate a served W8A8 deployment; the activation term is not
  measured"; "same value to 8 significant digits; see local_device_reduction_order";
  `engines/tools/exl3hf_surface.py:decode_payload_hf`; "equal to 1e-16; the token mean is
  the published value".

### Posted

Additive comments, each with the corrected Scope line, the interval, the paired ordering,
and the comparator's class caveat (the post header's `strict` is filed as `advisory`):

* https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1/discussions/1#6a9c022a39b4b7083c4d4475
* https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw/discussions/1#6a9c022b64a4fd14bc1fd7da
* https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw/discussions/1#6a9c022b58ecc7a2bb5c4e43
* https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/discussions/3#6a9c022bddfa6e75bafced4e
* https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3/discussions/1#6a9c022b80dd4dc0564bf708
* https://huggingface.co/zai-org/GLM-5.3/discussions/18#6a9c022b64a4fd14bc1fd7f7 (scope unchanged; interval and wording only)

Verification: `cd registry && make check` 0 errors (77 pre-existing warnings), 93
selftests, joint 501/0; `make reseed-check` clean; `bin/check_doc_numbers.py` 211/0;
`engines/tools/selftest_scope_provenance.py` 43 checks (10 fail on the parent tree);
`registry_selftest.py` section S fails on the parent tree; `selftest_stat01_reseed.py` T8
fails on the parent tree. Mirror verified byte-for-byte on every changed file.

---

## Not published, deliberately

Nothing from `docs/REVIEW-DEFERRED.md` is now held back for an operator decision on
published numbers. The remaining deferrals in that file are blocked by **file ownership**
(a live measurement campaign holds `bin/measure_cloud.py`), not by publication risk.
