# Contributing a measurement

You measured a quant. Here is how to get it into the registry.

**You submit exactly one file: the submission receipt your runner printed.**
You never edit `data/*.jsonl` and you never write a registry row by hand — a
measurement row carries derived fields (`comparability.key`, `scope_digest`) and
five cross-references, and hand-writing those is how wrong numbers get in. Our
tools generate the rows from your receipt.

---

## 0. Pick something that can actually be measured

Two minutes here saves an afternoon. A measurement needs a panel you can
download, a reader for the artifact's storage surface **on the lane you can
run**, and a profile for that surface at that bit rate — and the intersection
is narrower than the repo's feature list suggests. The full picture is
[README → *Before you rent*](../README.md#before-you-rent-what-is-measurable-today);
the short version:

* **The panel decides the model.** A measurement may not introduce a panel
  (§6), and only `panel--glm53.brandonmusic.final25` has a built-in fetch
  descriptor. The five Qwen3.8-27B panels are `availability.status: private`,
  so **Qwen3.8-27B quants cannot be measured from outside today** — the runner
  refuses at plan time with "has no local fetch descriptor in this checkout".
  In practice that means: a GLM-5.3-Flash quant, on brandonmusic's 25-window
  panel.
* **The lane decides the surface.** The cloud lane reads `packed`,
  `native-bf16`, `exl3hf`, `tr3-published` and `dione`; the local lanes read
  only `packed` and `native-bf16`. So the local recipe cannot execute a
  third-party quant measurement today, and MLX / GGUF / NVFP4 / AWQ / GPTQ
  repos are refused on every lane (the decoders exist under `k6/tools/` but no
  lane lists them).

**Is it already measured?** The front gate answers this for you, and it is the
first thing both runners do:

```bash
bin/measure <hf-repo>                     # prints the rows and exits 0 if so
bin/registry-view check <hf-repo>         # same question, tiers EXACT/STALE/...
bin/registry-view rows --model glm        # everything measured for a model
```

Two traps worth knowing before you conclude "already measured":

* **A multi-artifact repo is measured per branch or subpath.** The registry
  records `huggingface.path` (e.g. `branch 4.05bpw`), but the gate keys on the
  repo id, so asking about a *different* branch of a measured repo reports the
  other branch's row and calls the difference `revision drift`. That is not
  drift — it is an unmeasured artifact, and `--force` is the correct flag
  (it records a new artifact rather than restating an old one). The refusal
  now says which scope the existing rows cover.
* **Not every unmeasured repo is a gap worth filling.** An abliterated or
  fine-tuned derivative has different reference weights, so scoring it against
  the base model's teacher measures the fine-tune, not the quantization.

**Then price it before you commit.** `--dry-run` does every check, creates
nothing and spends $0.00, and it will tell you things you cannot see from the
model card — one branch of a well-known EXL3 release is missing 22 of the
model's 1,618 non-routed tensors, and several "release" repos are two-file
placeholders. Both are refused for free.

---

## 1. Produce the receipt

Either runner seals one at the end of a run, under `<out>/receipts/`:

```bash
# cloud -- rents a GPU, measures, tears the instance down, prints the real cost
# (if you have already run `jl setup`, the runner finds that credential and you
#  do not need to export anything; the same is true of a cached HF login)
export JL_API_KEY=...        # optional -- see above
./bin/measure-cloud \
    --model  <hf-repo> \
    --panel  brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane   streaming --spot --max-runtime 8h

# local -- Apple Silicon, or a consumer CUDA card under a hard VRAM budget
./bin/measure-local \
    --artifact <hf-repo> \
    --panel    brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --vram-budget 30
```

Run it with `--dry-run` (cloud) or `--estimate-only` (local) first. Both
validate everything, download nothing, create nothing and spend nothing, and
both print what the run will cost you in dollars, disk, memory and hours.

**Keep the default of at least two cold runs.** A single run produces a real
number that the registry will nonetheless reject: `run_count >= 2` is required,
because one run cannot demonstrate determinism.

Check it sealed correctly before you send it — four lines, no dependencies:

```python
import json, hashlib
d = json.load(open("submission.json")); claimed = d["receipt_sha256"]; d["receipt_sha256"] = ""
canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
print(hashlib.sha256(canon.encode()).hexdigest() == claimed)   # must print True
```

If it prints `False`, the file was edited after the run. Re-run rather than
patching it; we will bounce a broken seal.

Optionally, run our exact checks yourself before you submit — no `pip install`,
no network, stock Python 3.9. (This is the only step in the whole process that
touches git, and you can skip it.)

```bash
git clone https://huggingface.co/datasets/malaiwah/quant-fidelity-registry
cd quant-fidelity-registry
python tools/registry_validate.py --submission ~/submission.json
```

It prints the row it would generate, its comparability key, and its class — or
exactly what is wrong. Doing this first is the difference between a same-day
merge and a round trip.

The Hugging Face dataset repo above is the one that exists today and carries
`schema/`, `tools/` and `data/`. The GitHub mirror named in §3 is **not live
yet**; until it is, clone the HF repo for this step and submit by discussion.

**What runs in that clone, and what does not.** The published dataset repo is a
*subset* of the suite repo's `registry/` directory — it ships `tools/`,
`schema/` and `data/` at its root, without the `registry/` level above them and
without a few files that live only in the suite. So:

| in a clone of the HF dataset | |
|---|---|
| `python tools/registry_validate.py --submission <receipt>` | **works** — this is the command you need |
| `python tools/registry_validate.py --strict` (`make validate`) | works |
| `python tools/registry_validate.py --offline-selftest` | works |
| `make render-check` | **fails** — `README.head.md` is not published |
| `make joint` | **fails** — `registry_joint_check.py` is not published |
| `make selftest` | **partially fails** — some cases reach for suite-only files |
| `make check` (which runs all four) | **cannot pass in this repo today** |

Run the submission check; do not be alarmed when `make check` does not go green
there. `make check` is green in the suite repo (`quant-fidelity-suite`), where
all of its inputs exist. If you have that clone, `bin/registry-submit
<receipt.json>` is the same validation with a friendlier wrapper.

### What it costs, before you start

Both runners print a dollar estimate before they spend anything, but here is
the shape of the bill so you can decide whether to bother. All figures are
**measured**, on JarvisLabs spot GPUs in `IN2`, on the GLM-5.3-Flash sealed
25-window panel (51,175 positions).

The streaming lane fits on ONE GPU and its bottleneck is reading weights, not
matmul. What you pay for is therefore (a) pulling the artifact onto the box and
(b) reading it once per window.

| what | size | why you need it |
|---|---:|---|
| the quant you are measuring | 176-331 GB typical | the student |
| teacher logit panel | 30 GB | fp32 teacher logits for 25 windows |
| sealed token panel | 13 MB | the exact token ids and masks |
| BF16 source tree, partial | ~235 GB | **only if your artifact has no non-routed tensors of its own.** The scorer takes the non-routed 1,618 tensors (19.0 GB) from the official tree, and they live in 47 of its 120 shards, so you download 47 shards to use 19 GB of them. An artifact that carries its own non-routed weights skips this entirely. |
| fp32 student logits you will write | 31.7 GB **per cold run** | kept, because the determinism check compares them |

Disk: budget the artifact + 2 x 31.7 GB and do not size the instance for the
encode-era numbers. Two of our runs died on "Disk quota exceeded" because a
ledger written for encoding never accounted for measurement output.

**Keep two cold runs.** `run_count >= 2` is required, and the second run is not
a formality: it is the determinism evidence.

#### The bill, measured

Two reference points from our own runs, both 25-window panels with two cold
runs, both on **1x H200 spot in IN2 at $1.99/GPU-h**, weights already on a
local filesystem:

| what was measured | capture wall clock, 2 cold runs | GPU-hours | at $1.99/h |
|---|---:|---:|---:|
| K6 (231 GB payload store) | 11,018.9 s + 10,488.7 s | 5.97 | **$11.89** |
| K8 (304 GB payload store) | 14,254.5 s + 13,755.8 s | 7.78 | **$15.48** |
| BF16 floor (599 GB source tree, no decode) | 12,514.5 s + 12,4xx s | ~6.95 | **~$13.83** |

Add the transfers if the bytes are not already on the box. Measured from IN2
with plain `curl` against Hugging Face: **50.9 MB/s** on one stream, **180
MB/s** on four (`hf_transfer` uses more and should beat this):

| transfer | 180 MB/s |
|---|---:|
| a 176 GB quant | 0.27 h |
| a 331 GB quant | 0.51 h |
| teacher panel, 30 GB | 0.05 h |
| BF16 non-routed shards, ~235 GB, only if your artifact lacks its own | 0.36 h |

Bootstrap (python3.12 + `torch==2.11.0+cu130` + `transformers==5.16.1` +
safetensors/numpy/accelerate) is another ~5-10 min.

#### One end-to-end run, start to finish, by someone who had not done it before

The numbers above are the K6/K8 **payload-store** path. Measuring a
**third-party release from its own HF shards** — which is what an outside
contributor actually does — is cheaper, because the routed payloads are read
from the release rather than a content-addressed store. A full run of
`turboderp/GLM-5.3-Flash-exl3` at 2.05bpw (85.2 GB artifact), 25-window panel,
2 cold runs, 1x H200 spot in IN2, nothing cached:

| stage | wall clock |
|---|---:|
| bundle upload + watchdog | 5 min |
| bootstrap (`setup`) | 6 min 21 s |
| fetch artifact (85.2 GB) | 4 min 15 s |
| materialize non-routed tree | 2 min 08 s |
| fetch panel (31.7 GB) | 4 min 15 s |
| **measure, 2 cold runs x 25 windows** | **2 h 09 m** |
| score + seal + pull + teardown | 8 min |
| **total** | **2 h 37 m** |

Cost, all four ways the runner reports it: estimated **$6.80**, computed
**$5.23**, billed **$5.28**, balance delta **$5.18**. Zero preemptions.

**So budget ~3 h and ~$6-9 for a third-party EXL3-family release**, and use the
older 5.5-9.5 h / $11-19 figure only for a payload-store measurement. Either
way, take the number `--dry-run` prints for your target: it was 30% high here,
in the safe direction.

Add ~0.4 h and ~$0.8 if you also have to pull the BF16 shards.

Two hardware traps worth knowing before you rent:

* **The driver, not the GPU generation, is the gate.** An A100-80GB at
  $0.89/GPU-h fits the 47.1 GB working set and `transformers`'
  `_can_use_grouped_mm` has no compute-capability check — but the instance
  image we drew shipped NVIDIA driver 12080, and a `torch 2.11.0+cu130` venv
  cannot initialise CUDA on it at all. Check
  `nvidia-smi --query-gpu=driver_version` and one `torch.cuda` init in the
  first five minutes.
* **Disk is sized by the measurement, not the artifact.** Each cold run writes
  `positions x vocab x 4` bytes of fp32 student logits — 31.7 GB here — and
  both runs are kept, because comparing them is the determinism evidence.



### Fields that must be filled

The runner fills all of these. If you are hand-assembling a receipt from an
older run, these are the ones without which we cannot build a row:

| Field | Why |
|---|---|
| `artifact.repository` + `artifact.revision` | Identity. `revision` must be the immutable 40-hex commit, not `main`. |
| `artifact.scope` + `scope_digest` | What was actually quantized. A number without this is not attributable to a recipe. |
| `panel.panel_ref` + `panel.panel_token_sha256` | A fidelity number means nothing without the panel it was scored on. |
| `reference.reference_ref` + `teacher_receipt_sha256` | Which teacher capture you scored against. |
| `metric.{name,value,units,direction}` | `value` at full float64 precision — never rounded. |
| `estimator.{accumulation_dtype,stack_relation,head_policy}` | These three decide which rows yours is comparable to. |
| `determinism.{run_count,evidence_kind,evidence_hashes}` | See §5. |
| `measurement_scope.{scored_positions,covers_full_panel}` | A subset is fine; a subset presented as the whole panel is not. |
| `measurer.{name,handle}` | How you are credited. |
| `disclosures` | Non-empty. Nothing to disclose is written as one entry with `code: "no_known_deviations"`. |
| `lane` | `sealed-ep8`, `streaming`, `local-mps`, `local-cuda-budget`. Lanes are not interchangeable. |
| `produced_by.{entrypoint,entrypoint_sha256,revision}` | **Which code produced the number.** Since 2026-08-30 this is not optional: `registry_add` turns it into the row's `harness` block and invariant HARN-001 refuses any row that has none. |

Strongly recommended: `auxiliary_metrics.top1_agreement`. A KL number without
top-1 agreement hides which kind of divergence it is.

### Why `produced_by` became mandatory

Every measured value is a function of some code, and until 2026-08-30 no row said
*which*. When a peer review found a defect in the interval estimator, the honest
statement was that it put every published row equally under suspicion and cleared
none of them — 70+ numbers, one shared liability, no way to say which predated the
fix. The `harness` block is the field test that answers it: equal `harness_id`
means byte-identical code across the digest set, and a differing id points you at
the `code_digests` entry whose role changed.

The 72 rows that predate the mechanism are listed in
`schema/harness-grandfather.json` and each carries a `harness_unrecorded`
disclosure. That list is **frozen and never appended to** — an allowlist that
grows means nothing. New rows state their harness or are refused.

---

## 2. Submit it — primary path: Hugging Face discussion

**This is the path we recommend.** No git, no fork, no CI to argue with.

1. Open <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
2. **New discussion**, title: `submission: <repo> on <panel>`
   e.g. `submission: 0xSero/GLM-5.3-Flash-EXL3-Q4 on glm53-final25`
3. Paste the template below, with your receipt inside the fence.

````markdown
### Submission

- **Artifact:** 0xSero/GLM-5.3-Flash-EXL3-Q4 @ 99cccdf0e8741715662c383828a9ea601990c125
- **Panel:** panel--glm53.brandonmusic.final25
- **Reference:** reference--brandonmusic.glm53-bf16-fp32-logits.final25
- **Metric:** mean_of_run_means_tokenwise_kld = 0.027262784814670614 nats
- **Lane:** sealed-ep8
- **I am:** the measurer / also the quant's author? → measurer only
- **Anything odd about this run:** none

<details><summary>submission.json</summary>

```json
{ ...paste the whole file... }
```

</details>
````

That is the whole submission. Attach `submission.json` as a file too if the
discussion editor lets you — pasted JSON is fine, we hash it either way.

**Why this is the primary path.** The registry lives on Hugging Face, your
artifact lives on Hugging Face, and your HF username is already the attribution
we record. The thing you are submitting is a machine-generated receipt, so git
literacy buys you nothing here: a pull request would carry all of a PR's
friction and none of its benefit, because the registry rows still have to be
generated by our tools rather than written in the diff. A discussion is also
permanent, timestamped and publicly quotable, which is exactly what a
`source.kind: "discussion"` on your row needs to point at.

---

## 3. Submit it — fallback path: GitHub pull request

Use this if you prefer review-in-diff, you are submitting several measurements
at once, or you want CI to check the receipt before a human sees it.

Mirror: <https://github.com/malaiwah/quant-fidelity-registry>

> **NOT LIVE YET.** That URL currently 404s: the mirror and its CI workflow are
> written but not published. Until it is up, use the discussion path in §2 —
> it is the recommended path anyway. Everything below describes what the mirror
> will do when it exists, not what you can do today.

```
quant-fidelity-registry/
├─ schema/            *.schema.json, invariants.json    ← contract, don't edit
├─ data/              *.jsonl                           ← GENERATED, don't edit
├─ index.json                                           ← GENERATED, don't edit
├─ receipts/<handle>/<slug>.json                        ← YOU ADD EXACTLY ONE FILE HERE
├─ tools/             registry_lib / registry_add / registry_validate
└─ .github/workflows/validate.yml
```

```bash
git clone https://github.com/malaiwah/quant-fidelity-registry && cd quant-fidelity-registry
mkdir -p receipts/<your-hf-handle>
cp ~/submission.json receipts/<your-hf-handle>/glm-5.3-flash-exl3-q4.json
git checkout -b submit/glm-5.3-flash-exl3-q4
git add receipts/ && git commit -m "measurement: 0xSero/GLM-5.3-Flash-EXL3-Q4 on glm53-final25"
git push origin submit/glm-5.3-flash-exl3-q4   # then open the PR
```

The directory must be your HF handle and the file must be your sealed receipt.
Do not touch `data/`, `index.json` or `schema/` — CI fails the PR if you do.

### What CI runs

`.github/workflows/validate.yml`, on every PR:

1. `registry_validate.py --offline-selftest` — asserts no tool imports a
   networking library. Validation never fetches anything, and needs no
   dependencies: schema checking runs on `tools/_minischema.py`, so you can run
   every check below yourself on a stock interpreter with no `pip install`.
2. **Diff gate** — the PR touches only `receipts/**`. A hand-edited row in
   `data/` is the failure mode this check exists to stop.
3. **Receipt gate**, per changed receipt — parses; validates against
   `schema/submission.schema.json`; recomputes `receipt_sha256` and rejects a
   broken seal; recomputes `scope_digest` from `artifact.scope`; checks the
   directory name equals `measurer.handle`.
4. `registry_add.py --receipt <each> --write` — regenerates `data/*.jsonl` and
   `index.json`, then asserts the regeneration touched **only** rows derived
   from the submitted receipts.
5. `registry_validate.py --strict` — every invariant in `schema/invariants.json`.
6. Comments on the PR with the generated `measurement--…` id, its
   `comparability.key`, its class, and the rows it can be compared against.

Green CI is not automatic merge; a maintainer still reads it.

---

## 4. What we do with it, and how you are credited

Identical for both paths:

1. We save your receipt to `receipts/<your-handle>/<slug>.json` and record its
   sha256.
2. We run the same steps CI runs (§3) locally.
3. We reply — in your thread or on your PR — with the row id, its comparability
   key, its class, and which existing rows it sits next to. If we refuse it, the
   reply says exactly which check failed and what to change. Either way you get
   a real answer, not silence.

**Attribution.** Credit is not transferable and the validator enforces it
(invariant `PROV-006`):

- `provenance.measurer` = your name, your HF handle, `https://huggingface.co/<handle>` — **you** get credit for the number.
- `artifact.producer` = whoever made the quant. If that is not you, it is not you.
- `panel.author` / `reference.author` = whoever built the panel and captured the teacher.
- Your discussion or PR URL is attached to the row as a `source` of kind
  `discussion` / `github_file`, so anyone reading the row can find your
  submission and argue with it.
- `CONTRIBUTORS.md` is regenerated from the distinct measurer handles in
  `data/measurements.jsonl`. You do not have to add yourself.

**How your row is classified.** Two tiers, and the difference is not about
trust:

- `strict` — self-measured by the registry maintainer, same-stack, sealed panel,
  no comparability-affecting disclosure.
- `advisory` — everything else, including every measurement contributed from
  outside. Advisory rows appear in the same table as strict rows **when the
  comparability key matches**, always visually marked.

Separately, `provenance.independently_verified` becomes `true` the moment a
different party reproduces your number on the same panel and reference. That is
the flag worth chasing, and it is the reason submitting a receipt with published
evidence beats submitting a bare number.

---

## 5. Things that will get your submission bounced

- `artifact.revision` is `main`, a tag, or a short sha. It must be the 40-hex commit.
- The seal does not verify — the file was edited after the run.
- `estimator.stack_relation` or the `determinism` block was hand-edited. The
  runner emits these, and the receipt seal covers them, so an edit after the run
  breaks `receipt_sha256`. Be clear about what that does and does not stop:
  re-sealing an edited receipt produces a file we cannot distinguish from an
  honest one, so `cross_stack` -> `same_stack` — the single most damaging edit
  possible here — is deterred, not detected. What *is* mechanically enforced is
  the generator path: `registry_add` reads `stack_relation` out of the receipt
  family and **refuses (exit 6)** any `--stack-relation` flag that contradicts
  it, unless you pass `--disclosure` saying on what evidence, in which case the
  override is stamped onto the row and into `field_provenance` for the reader.
  The registry's defence against a re-sealed lie is that your receipt is public
  and your number sits next to other people's on the same panel.
- `determinism.identical_across_runs: true` backed by a **receipt-file** or
  **archive** hash. Report files embed run indices, paths and timestamps and
  differ across bit-identical runs. Only a tensor-content hash
  (`tokenwise_kld_sha256` and friends) can support that claim — the schema
  refuses the others outright.
- A subset run with `covers_full_panel: true`; a subset run (`covers_full_panel:
  false`) with no `subset_of_panel` disclosure saying *which* subset; or any run
  claiming more scored positions than the panel it names actually holds.
- `panel.panel_ref` names a panel the registry does not have. See §6.
- `produced_by` carries no `entrypoint_sha256`, so the code that produced the
  number cannot be identified (HARN-001).
- A disclosure that claims *how* the artifact was made — a mechanism, a lineage,
  an inherited config, a code path — with no `sources` (PROV-014), or with a
  source pinned to a **branch** rather than a commit (PROV-015). A metric has
  always needed a hashed receipt here; an assertion needs one too. Cite
  `/blob/<40-hex-commit>/file.py` with a `lines` anchor, never `/blob/main/`:
  line numbers move, and a citation that quietly stops pointing at what it
  claimed still reads as evidence.
- `panel.panel_token_sha256` is not the digest the named panel carries. This
  catches the most common honest mistake there is: measuring on your own corpus
  and then picking the closest-looking `panel_ref` out of the README. Your tokens
  are not that panel's tokens and the two numbers do not belong in one table.
  Open a `panel:` discussion instead (§6).
- `reference.teacher_receipt_sha256` is not the capture digest the named
  reference carries. A number measured against a different teacher — another
  BF16 capture, or an FP8 release dequantized to BF16 — is a different quantity.
  It is welcome here under its own reference record; it is not welcome inside
  somebody else's. Where the named reference has no capture digest on file we
  cannot check yours, and the row is forced to carry a
  `teacher_capture_unverified` disclosure, which makes it advisory.
- `measurer.handle: malaiwah` from a `produced_by.repository` that is not ours.
  `self-measured` is this registry's highest trust level and `class: strict`
  follows from it. A submission cannot assert either on the maintainer's behalf.
  Put your own handle there and be credited for your own work.
- A rounded `metric.value`. Full float64 or nothing. Honesty note on this one:
  the registry cannot *detect* a rounded value — `0.0053` round-trips through
  `repr()` exactly like a computed number does — so this rule is enforced by the
  seal over your receipt and by the fact that anyone can recompute your number
  from the panel you named. Send the value your tool printed, all of it.

---

## 6. New panels, new models

A measurement can only reference a panel and a teacher capture that already
exist as records. If yours does not:

Open a discussion titled `panel: <name>` **before** you measure, with the panel's
token digest (`panel_token_sha256`), how many contexts and scored positions it
has, its scoring window (`score_from`), the tokenizer it is bound to, and where
it can be downloaded. Same for a teacher capture: the capture stack, dtypes, and
the receipt sha256. We add the records, then your measurement submits normally.

A panel whose token ids are not pinned by a content hash cannot be sealed, and
measurements on it are permanently advisory. That is not a judgement — it is
just that nobody else can reproduce a panel they cannot reconstruct.

---

## 7. Worked example

[`docs/examples/dione-q4.submission.json`](docs/examples/dione-q4.submission.json)
is a real, sealed, schema-valid submission: 0xSero's EXL3 Q4 quant of
GLM-5.3-Flash, measured on brandonmusic's 25-window sealed panel against his
BF16 fp32 teacher logits.

```
artifact     0xSero/GLM-5.3-Flash-EXL3-Q4 @ 99cccdf0e874...
panel        panel--glm53.brandonmusic.final25   (25 windows, 51,175 positions)
reference    reference--brandonmusic.glm53-bf16-fp32-logits.final25
metric       mean_of_run_means_tokenwise_kld = 0.027262784814670614 nats,
             direction reference_to_candidate
estimator    float64 accumulation, same_stack, native_head
determinism  5 cold runs, 5 DIFFERENT report hashes, ONE tokenwise-KL tensor hash
receipt_sha256   47a07bfb54fdd59522ffb4f1babd26a87fc6419928032b867d30999ad917307b
```

Ingesting it reproduces the registry's existing row exactly — same
`scope_digest`, same comparability key `cmp--202b717f3219c414` as
`measurement--glm53.dione-q4.brandonmusic-final25`. That round-trip is the whole
contract: your receipt determines your row, and nothing else does.

Three things in it are worth copying:

**The determinism block is the honest shape.** Five runs produced five different
`per_run_report_sha256` values and exactly one `evidence_hashes` entry. The
report hashes are recorded for traceability and are explicitly *not* the
evidence; the tensor hash is. This is what a real bitwise-determinism claim
looks like.

**The scope says "unknown" where it does not know.** 0xSero never published a
per-tensor-class recipe, so `embed_tokens`, `attn.*` and `lm_head` are recorded
as `treatment: "unknown"` rather than guessed, and the row carries
`artifact_identity_incomplete`. You can see the gap in the digest itself:
`attn.o=unknown:unknown|…|moe.experts=quantized:exl3-mcg@4|head=unknown|kv=unknown`.
An honest gap costs the row its `strict` class; a confident guess would have
corrupted every comparison downstream. If you can parse the release's manifest,
do — a complete scope is worth real accuracy.

**The disclosures are specific.** `unsealed_source` says exactly what could not
be verified (no upstream receipts or sealed reader ABI) and exactly what was
verified instead (every whole shard against the release's own manifest).
`third_party_artifact_self_measured` separates credit for the quant from credit
for the number. Vague disclosures are worse than none — they cost the row its
comparability without telling anyone why.

> Note: this example was reconstructed from the published five-run rollup for
> documentation, so a few optional fields (`top1_agreement`, `cost`,
> `peak_vram_gb`) are `null`. A receipt straight out of a runner fills them.

---

## Questions

Open a discussion. "Is this comparable to X?" is a good question and the answer
is usually in the comparability key.

## Put the number on your model card too

A registry row is queryable; a card annotation is what someone sees when they
land on your model. Both, ideally.

```bash
bin/fidelity-card annotate --card README.md --role quant \
    --measurement-id <your-row-id> --out README.md --diff
bin/fidelity-card validate --card README.md
```

It emits a conformant HF [`model-index`](https://huggingface.co/docs/hub/model-cards)
result — so leaderboards and crawlers read your KLD with no bespoke code — plus a
small additive `x_fidelity` block carrying what `model-index` cannot express
(lane, scope, head identity, the receipt link). Unknown top-level keys pass the
Hub validator, so it will not break your card.

Copy-paste template and the three fields people get wrong:
[`docs/CARD-ANNOTATION-SPEC.md` §0](../docs/CARD-ANNOTATION-SPEC.md).
