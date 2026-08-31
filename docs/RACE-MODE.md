# Race mode — capturing a root while the checkpoint is still downloading

> `bin/measure-cloud --role root --race`
>
> Two things happen: the fetch stops being a barrier and becomes a priority queue
> the capture blocks on layer by layer; and the result of the first cold run is
> published under **its own identity** as a preview, never as a first version of
> the final root.
>
> The first is an engineering win with a measured number attached. The second is
> the only reason the first is allowed to exist.

## The problem this solves

A model lands on Hugging Face. Within hours, a handful of quanters with
pre-release access publish quants of it. Nobody — including them — knows how good
any of those quants are, because **there is no root fidelity dataset to measure
against**. Being first to publish that root is a genuine service to everyone
downstream, and it is a race measured in hours.

The default pipeline serializes two long operations that do not have to be
serialized:

```
stage fetch_target   [=========== download the whole checkpoint ===========]
stage capture                                                              [==== capture ====]
```

## Why the overlap is sound

`k6/tools/layer_outer.py` runs the capture as

```
for each layer:  load it once;  for each window: push that window through it;  free it
```

Layer N's weights are not read until the capture reaches layer N. So:

```
fetch    [ index ][ resident ][ L0 ][ L1 ][ L2 ] ... [ L45 ][ tail ]
capture           ........... [ build ][ L0 ][ L1 ][ L2 ] ...  [ L45 ][ seal ]
```

`k6/tools/race_fetch.py` reads `model.safetensors.index.json` — whose
`weight_map` is the only statement of which shard holds which tensor — and
buckets every shard by the **first** layer that needs it. The fetch becomes a
priority queue: index and tokenizer, then the resident set, then layer 0, layer
1, and so on, with every remaining file continuing to download at full speed in
the background.

`layer_outer.build_streamed_model` takes an optional `gate`. With one it opens
and audits only the shards the resident load will actually read — computed from
the model's own stack prefix and the conversion mapping's renames, not taken from
the fetch plan's bucket — and inside `load_layer(N)` it **blocks** until layer N's
shards have landed, audits them, then opens them. Worst case that is no slower
than fetch-then-capture. Typical case it hides most of the fetch.

One consequence is not obvious and cost a run to find: **a buffer that belongs to
one layer rides with that layer**, not with the resident set. See
"[It found a real defect](#it-found-a-real-defect-the-first-time-it-touched-a-real-model)"
below.

### The head is needed FIRST, not last

The natural guess — "the final norm and `lm_head` are needed last, order them
last" — is **wrong for this capture path**, and acting on it would deadlock the
run at layer 0.

* `build_streamed_model` performs one resident load of everything outside the
  decoder stack *before* any layer is streamed, and refuses if a non-layer
  parameter is still on the meta device.
* `hf_capture` reads `head.weight.shape` for the vocabulary and hidden sizes, and
  registers its capture tap as a forward **pre-hook on the head**.
* A hidden-form dataset publishes the head's own tensor bytes, because `compare`
  refuses a hidden-form capture whose head content digest is null (HEAD-4, no
  override).

So on the form we actually publish, the head is a priority-0 file. Priority comes
from the checkpoint's own `weight_map`, never from a guess about which tensors
"come last".

### Every way the plan can be wrong is conservative

A shard's priority is the **minimum** layer index over the decoder tensors it
carries, and any key the layer regex does not match falls into the resident
bucket. The two ways an unfamiliar architecture can defeat the regex — a vision
tower's `...encoder.layers.N.` keys, or a decoder spelling the pattern misses —
both pull a shard **earlier** than strictly needed. Neither can release a layer
whose bytes have not landed.

And if the plan were wrong in that direction anyway,
`layer_outer.audit_checkpoint_tree(shards=…)` runs on each layer's shards
immediately before the first load that reads them, and refuses a shard that is
absent or shorter than its own safetensors header. That is the guard that matters,
because **a short safetensors shard does not raise — it reads as zeros.** The gate
is an optimisation; the audit is the guard. A gate timeout is a refusal, never a
proceed.

## What it measured

### On a real checkpoint, over a real link

`malaiwah/GLM-5.2-SIQ-Fruit-bf16` at `ef68013a` — 10.10 GB, 16 shards
(embeddings, head, and one shard per layer), 13 built decoder layers plus an MTP
block. Both arms on the same laptop CPU, `--schedule layer-outer
--layer-residency stream`, an 8-window synthetic panel at 256 tokens, 8 parallel
downloads each. Receipt: [`reports/race-mode/fruit-ab.json`](../reports/race-mode/fruit-ab.json).

```
control   hf download the whole repo   84.66 s   (119 MB/s over 8 workers)
          then capture                 18.86 s
          total                       103.52 s

race      capture, fetch overlapped     91.01 s
          saved                         12.51 s   (12.1% of the total)

both arms  capture_content_digest = 38821f7c593db0b793cfd1593873e27afe07481d33cabbb87e892e3913e73361
```

**Read the 12% honestly.** This run is in the *fetch-dominated* regime — 85 s of
fetch against 19 s of capture — so the ceiling on the saving is the **capture**,
not the fetch: at most 18.86 s could be hidden, and 12.51 s was (66% of it). The
run's own `blocked_seconds` is 72.4 s, which is the fetch the capture could not
hide because there was not enough capture to hide it behind. The per-layer trace
shows the mechanism working exactly as designed: layers 0, 1 and 2 were ready
with **zero wait**, and layers 3 onward blocked 4–9 s each while the fetch stayed
just ahead.

A production root is the **opposite** regime — hours of capture against tens of
minutes of fetch — where the same mechanism hides the whole fetch. The
generalisable statement is not the percentage, it is the bound:

> the saving is at most `min(fetch_wall, capture_wall)`, less the un-overlappable
> prefix (the resident set plus layer 0).

Every run writes its own `race-fetch-report.json` with `blocked_seconds` measured
per block, so the claim for any particular run is a receipt rather than this
paragraph.

### The digest is the part that matters

Both arms produced the same `capture_content_digest`. Race mode changes when
bytes arrive, never which bytes, never the order of any arithmetic.
`bin/selftest_race_mode.py` R6 asserts that offline as a regression, on a
simulated link with an injected per-file delay, because a schedule cannot be
A/B-tested repeatedly on a real fetch — you get one arm, once, at whatever the
link happened to be doing:

```
control: fetch 0.46s + capture 1.69s = 2.15s total
race:    1.68s total  ->  0.47s saved (102% of the fetch hidden)
```

### The 0.1B CI fixture cannot demonstrate this, and says so

`inference-optimization/GLM-5.3-Flash-0.1B-A0.1B` ships a **single**
`model.safetensors` and no `model.safetensors.index.json`. There is no map from
tensor to shard, therefore no fetch order, therefore nothing to overlap — and
race mode refuses it by name at the `race_bootstrap` stage rather than
pretending. That is the correct answer, not a gap.

### It found a real defect the first time it touched a real model

The first live run refused, and it was right to:

```
REFUSED: 10 parameter(s) were NOT usable from the checkpoint and were randomly
initialised by transformers ... model.layers.{3..12}.mlp.gate.e_score_correction_bias
```

`e_score_correction_bias` is a router-correction **buffer**, and
`StreamedModel`'s rule is that buffers are never streamed — they load with the
resident set. On this architecture that is four kilobytes living inside each
layer's 845 MB shard, so a resident load that waits for them waits for ten of the
fourteen shards: the overlap would have been deleted, and (because the gated
resident load had not opened those shards) the buffers were reported MISSING and
randomly initialised instead.

The fix is the loop order's own logic: under a gate, **a buffer belonging to one
layer rides with that layer**, and a per-layer guard refuses if the checkpoint
does not deliver it. The ungated path is untouched — its bit-identity against
`from_pretrained` is already proven — and the digest equality above is what
demonstrates the two agree.

This is worth stating plainly because it is the argument for running the thing
rather than reasoning about it: the plan looked correct on paper and on a
synthetic fixture, and was wrong on the first real MoE it met.

## The identity problem, which is the important half

Michel's original sketch was: publish a preliminary dataset after run 1, then
update it after run 2.

**Updating a published root in place would silently corrupt every measurement
made against it**, and this project's own machinery says why.
`registry/tools/registry_lib.py` binds `reference_id` into
`COMPARABILITY_KEY_FIELDS`, and `bin/fidelity/dscompare.py` feeds the reference
dataset's own `dataset.id` into that field. Two rows share a comparability group
— i.e. get ranked against each other, in one table — exactly when those seven
fields match. If a root's **content** changed while its **identity** stayed the
same, rows measured against the old bytes and rows measured against the new bytes
would land in the same group and be quietly incomparable. That is precisely the
class of error REFC-001/REFC-006 and [`DESIGNATED-REFERENCE.md`](DESIGNATED-REFERENCE.md)
exist to prevent.

So the rule is:

> **The preview and the final are different identities, not two versions of one.**

### How that is enforced

| | preview | final |
|---|---|---|
| `dataset.id` | `…-root-v1.preview` | `…-root-v1` |
| cold runs | 1 | 2, agreeing digest-for-digest |
| `determinism.identical_across_runs` | `null` | established by the SC-1 self-compare |
| `not_submittable` | `true` | absent |
| `preview` block | names its successor | absent |
| disclosure | `preview_capture`, **blocking**, `affects_comparability: true` | — |
| sealed & immutable | yes | yes |
| updated in place, ever | **no** | no |

Nothing about this is a new mechanism. It is three existing ones, pointed at a
new case:

1. **Identity.** A different `dataset.id` is a different `reference_id` is a
   different comparability key is a different table. Mechanical, not careful.
   `--preview-of` **refuses** when the preview id equals the final id, naming
   `reference_id` as the reason — `bin/measure-cloud` refuses it before any spend,
   and `hf_capture` refuses it again at seal time.
2. **Publishability.** `not_submittable: true` is the marker
   `bin/fidelity/receipt.py::_scan_for_unsubmittable` already refuses at any depth
   of a submission's input blocks.
3. **The comparator.** `dscompare.run_gates` raises a **blocking** disclosure when
   either side is a preview, and `emit_submission`'s SC-5 check already refuses
   any comparison carrying one — with the same message it uses for a
   head-substituted number.

The preview is a real, sealed, verifiable `malaiwah.fidelity-dataset.v1`. It
`verify`s, it `describe`s, and you can compare a quant against it and get a real
receipt with a real number. **What you cannot do is turn that number into a
registry row.** A row measured against the preview is a true statement about the
preview; it does not become a statement about the final, and it does not enter the
public record until it is re-run against the final.

### What the preview's card must say

`--preview-of` writes it into the dataset's own manifest and README, so it cannot
be lost by a copy-paste:

> THIS IS A PRELIMINARY CAPTURE. It is backed by ONE cold run. Cross-run
> determinism is NOT demonstrated: a second cold capture agreeing digest-for-digest,
> plus the exactly-0.0 self-compare, is what would establish it, and neither has
> happened here. It is sealed and immutable and will NEVER be updated in place —
> the complete-evidence capture is a SEPARATE dataset with a separate id, named
> below.

Note the registry's own precedent: a published measurement row requires
`run_count >= 2` (DET-001, and the refusal in `bin/invoke_scorer.py`) precisely
because one run cannot demonstrate determinism. A single-run *root* is subject to
the same logic, and gets the same answer.

## The generation sanity check

**This one is not about racing, and it is now on for every capture.**

`refuse_on_load_report` and `audit_checkpoint_tree` are guards over names, shapes
and counts. There is a class of catastrophe that passes all of them: a shard whose
bytes are the right *length* and the wrong *content*.

* a sparse-file fetch that left a hole the right size — the tensor reads as zeros
  (`audit_checkpoint_tree`'s own docstring says this is what it does **not**
  catch);
* `--allow-missing-weights` over a real absence, where `transformers` randomly
  initialises the parameters and hands back a model that runs — observed on
  `malaiwah/GLM-5.2-SIQ-Fruit`, whose routed experts came back with mean ~0 and
  std 0.0199, with correct names, correct shapes and a correct tensor count;
* a plain FP8 payload cast into a bf16 parameter with its block scale never
  applied — identical shapes, and only `unexpected_keys` to show for it.

None of those survives asking the model a question a language model can answer:

```
"The capital of France is"  ->  " Paris"
```

### Why it is cheap enough to be unconditional

Under the layer-outer schedule the probe is **one more window** pushed through the
layers the schedule is already loading. It costs one extra forward per layer —
1/N of an N-window panel, about 4% at N=25 — and **zero extra weight loading**,
which is the part that costs money. It is never a second pass over the checkpoint;
for a 1.5 TB model that would be the whole capture again. Under the window-outer
schedule it is one extra forward.

Its hidden state is discarded. It is never written to the dataset and never enters
`capture_content_digest`; `bin/selftest_race_mode.py` R11 asserts the digests with
and without it are equal, rather than reasoning that they should be.

### What it enforces, and what it merely records

Always **recorded**, in `manifest.generation_sanity_probe`: the top-1 token id, its
detokenisation, its probability, the distribution's entropy, the uniform entropy to
compare against, and the top 5. A probe that could not run records *why* — a skip
is a verdict, never a silent pass, and it also raises a `caveat` disclosure.

**Fail-closed unconditionally** on a degenerate distribution — every logit exactly
equal. That is what an all-zeros tensor produces, it is model-agnostic, and no
trained model at any scale emits it. (R10 zeroes a head and asserts the refusal.)

**Fail-closed on the content** when an expectation is declared. `bin/measure-cloud
--role root` declares `--sanity-expect Paris` by default, because every real root
is a real pretrained language model; `--sanity-expect ''` records without
enforcing, for a model genuinely expected to answer otherwise. The expectation is
checked against the tokenizer's own encoding of the continuation rather than by
string compare, so it behaves the same on SentencePiece and BPE. If enforcement is
requested and the tokenizer cannot be loaded, that is a **refusal** — a fail-closed
check that could not run has not passed.

## The sequence

```
1.  race_bootstrap   config.json + tokenizer + model.safetensors.index.json.
                     Kilobytes. Refuses a checkpoint with no shard index: without
                     the weight_map there is no fetch ORDER, only a download.

2.  race_capture     ONE process. The priority fetch runs in background threads;
                     the layer-outer loader blocks per layer. The generation
                     probe rides along as an extra window. At the end the fetch
                     is joined, the checkpoint identity digest is taken over the
                     COMPLETE tree, the release's published SHA256SUMS are
                     verified, and the dataset is sealed.

3.  verify           seal + digest chain + tensor content, before the box dies.
                     The last moment at which a bad capture is free to discard.

4.  publish the PREVIEW      <id>.preview, labelled as above.

--- a second rental, or a second run on the same box ---

5.  a second cold capture    same weights, same panel.
6.  compare A B --self-compare  -> must be exactly 0.0 (SC-1). One distinct
                                   capture digest across both runs is what
                                   `determinism.identical_across_runs` needs.
7.  publish the FINAL root   <id>, full evidence, its own identity.
```

Steps 5–7 are the maintainer's, and this branch builds the path without walking
it: nothing here publishes to Hugging Face.

## The honest part

**This project's entire product is trustworthy numbers, and a race is in tension
with that.** Speed pressure is how a measurement gets published before its
evidence is complete, and "we will fix it in the next version" is how a published
number that was wrong stays cited.

The mitigation is **labelling and identity separation, not speed**. Concretely:

* Race mode does not skip a single guard. The checkpoint audit, the per-layer
  load report, the meta-device check, the seal, the digest chain and the tensor
  content verification all run exactly as they do on the ordinary path. What race
  mode removes is *waiting*, not *checking*.
* The preview is not a lower-quality root. It is a root with **less evidence**,
  under a different name, and the difference is stated in its manifest, in its
  card, in a blocking disclosure, and in the refusal any attempt to publish a row
  against it produces.
* The **preview** is structurally unsubmittable, on the same two axes preview
  *receipts* already were. Racing itself is not: a race capture without
  `--preview-of` — run 2 of a two-run root, say — is an ordinary capture with an
  ordinary receipt, because the only thing race mode changed was the download
  order and the digest proves it.
* **One real difference between the gated and ungated loaders, stated rather than
  buried.** Under a gate, a buffer belonging to one layer loads with that layer
  instead of with the resident set. The bytes and the arithmetic are identical;
  the *moment* differs. The defence is not the argument, it is the measurement:
  the two paths produce the same `capture_content_digest` on the real Fruit
  checkpoint (above) and on the CI fixture (R6), and a per-layer guard refuses if
  the checkpoint fails to deliver a deferred buffer.
* A `--race-simulate-source` run — the offline harness that makes the overlap
  measurable — stamps its own blocking disclosure, so a dataset produced by the
  test rig can never be mistaken for a measurement of anything.

What remains genuinely at risk, and is worth saying out loud: a preview root
published in the first hours of a model's life will be *used* — someone will rank
their quants against it and quote the number. The blocking disclosure stops it
entering our registry; it does not stop a person quoting it in a forum post. The
only real defence there is that the preview's own card says what it is, in the
first paragraph, and names the dataset that supersedes it.

If the maintainer decides that is not defence enough, the correct response is to
not publish previews at all — the machinery is built so that omitting `--race` and
`--preview-of` leaves the ordinary two-run path untouched.

## Flags

```bash
bin/measure-cloud --role root --race \
    --model <hf-repo> --panel-dir k6/panels/<panel> \
    --dataset-id my-fidelity-root-v1.preview \
    --preview-of my-fidelity-root-v1 \
    --max-cost 15 --dry-run
```

| flag | what it does |
|---|---|
| `--race` | overlap the fetch with the capture; needs `--schedule layer-outer` and refuses otherwise, for $0.00 |
| `--race-workers N` | parallel downloads (default 8). Ordering is by priority, so this widens the queue, it does not reorder it |
| `--preview-of ID` | seal as a preview superseded by `ID`. Refuses when `ID == --dataset-id` |
| `--sanity-expect S` | the continuation the probe requires (default `Paris`); `''` records without enforcing |

Engine-level (`k6/tools/hf_capture.py`, reachable through
`bin/fidelity-dataset capture -- …`): `--race-repo`, `--race-revision`,
`--race-workers`, `--race-timeout-seconds`, `--race-layer-key-regex`,
`--race-report`, `--sanity-prompt`, `--sanity-expect`, `--no-sanity-check`,
`--preview-of`, and the test-harness pair `--race-simulate-source` /
`--race-simulate-seconds`.

## Files

| path | what it is |
|---|---|
| `k6/tools/race_fetch.py` | the plan, the gate, the fetcher, the measured report |
| `k6/tools/generation_probe.py` | the sanity probe and its verdict |
| `k6/tools/layer_outer.py` | `audit_checkpoint_tree(shards=…)` and `build_streamed_model(gate=…)` |
| `k6/tools/hf_capture.py` | starts the fetch, runs the probe, seals the preview identity |
| `bin/fidelity/dscompare.py` | the provenance gate that blocks a preview from the registry |
| `bin/stage_measure.sh` | `race_bootstrap`, `race_capture` |
| `bin/measure_cloud.py` | `--race`, `--race-workers`, `--preview-of`, `--sanity-expect` |
| `bin/selftest_race_mode.py` | R1–R16 (17 cases), every one of which fails without the change — verified by running the file against a `git archive` of the parent commit: 0 passed, 17 failed |
| `reports/race-mode/fruit-ab.json` | the real-network A/B receipt: both arms' timings, both digests, the fetch plan and every per-file download |
