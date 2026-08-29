---
pretty_name: Quantization Fidelity Registry
license: cc-by-4.0
tags:
  - quantization
  - fidelity
  - kl-divergence
  - registry
  - evaluation
configs:
  - config_name: measurements
    data_files: data/measurements.jsonl
  - config_name: artifacts
    data_files: data/artifacts.jsonl
  - config_name: panels
    data_files: data/panels.jsonl
  - config_name: references
    data_files: data/references.jsonl
  - config_name: pipelines
    data_files: data/pipelines.jsonl
  - config_name: models
    data_files: data/models.jsonl
---

# Quantization Fidelity Registry

A public, schema'd, receipt-backed, cross-model index of **quantization quality measurements**.

It exists to answer one question that nothing else answers today: *show me every measured quant of
model X, with its fidelity number and enough provenance to know whether that number means anything.*

It is the sibling of [`0xSero/local-ai-registry`](https://huggingface.co/datasets/0xSero/local-ai-registry),
which answers *how fast, how much VRAM, how much money*. This one answers *how faithful*. Ids and the
`huggingface` identity block are deliberately shaped so records can be cross-linked; every artifact,
model, panel and pipeline carries a `cross_refs.local_ai_registry` slot, and a link is never presented
as verified unless it has been.

---

## The rule this registry exists to enforce

> **Two fidelity numbers may be compared if and only if their `comparability.key` values are equal.**

A bare `kld: 0.027` is worse than nothing. A KL divergence is only meaningful relative to a specific
set of tokens, measured against a specific teacher capture, in a specific direction, at a specific
accumulator precision, through a specific stack relation, with a specific head policy. Change any one
of those and you have a different quantity that happens to be printed with the same units.

So the key is a **hash over exactly those seven things**:

```
comparability.key = "cmp--" + sha256("|".join([
    panel_id,            # WHICH TOKENS, including the scored-position policy
    reference_id,        # WHICH TEACHER CAPTURE (a capture, not a model: artifact + panel + stack + precision)
    metric_name,         # mean_tokenwise_kld, mean_of_run_means_tokenwise_kld, ...
    direction,           # reference_to_candidate (KL(P_teacher || Q_student)) or the reverse
    accumulation_dtype,  # float64 vs float32 over 10M positions is a different estimator
    stack_relation,      # same_stack, or cross_stack (which carries a known upward bias)
    head_policy,         # the candidate's own lm_head, or one shared head applied to both sides
]))[:16]
```

`tools/registry_validate.py` recomputes this key for every row from the row's own fields and rejects a
mismatch (`CMP-001`). A hand-written key cannot move a number into a table where it does not belong.
`tools/registry_render.py` groups tables by that key and by nothing else, and `--check` fails if the
committed README differs from what the data renders. **The tables below are a pure function of
`data/*.jsonl`.** They were never typed by hand and cannot drift.

### A worked example: one valid comparison and one invalid one

Five numbers, all for GLM-5.3-Flash, all on brandonmusic's sealed 25-window / 51,175-position panel,
all against the same stored fp32 teacher logits, all KL(teacher || student) in nats, all accumulated
in float64. They are printed as **two tables, not one**, because they are two quantities. A reader who
skims tables rather than paragraphs should be stopped by the layout, not only by the prose underneath
it:

**Group `cmp--202b717f3219c414`** -- sealed-lane same-stack capture, five cold runs each. These three may
be ranked against one another. (This group holds five rows today; the other two came off a different
measurement *lane* and are the subject of the section after next.)

| | value | metric | stack_relation |
|---|---:|---|---|
| malaiwah TR3 6bpw (K6), 253.5 GB | 0.013723384665701147 | `mean_of_run_means_tokenwise_kld` | `same_stack` |
| brandonmusic tr3 4bpw, 175.6 GB | 0.024554564249958208 | `mean_of_run_means_tokenwise_kld` | `same_stack` |
| 0xSero EXL3 Q4 (Dione), 187.6 GB | 0.027262784814670614 | `mean_of_run_means_tokenwise_kld` | `same_stack` |

**Group `cmp--4a8630bdcadab97f`** -- **a different quantity, not a continuation of the table above.**
Single-pass cross-stack replay against that same stored teacher. These two may be ranked against each
other and against nothing above them.

| | value | metric | stack_relation |
|---|---:|---|---|
| BF16 replay (the floor) | 0.012711599817250710 | `mean_tokenwise_kld` | `cross_stack` |
| official FP8 (our replay) | 0.020615254540417995 | `mean_tokenwise_kld` | `cross_stack` |

Note the sizes in the first table. K6 leads it, and K6 is also the largest artifact in it by 66 GB.
Rank within a comparability group is a fidelity ordering, not a value judgement: fidelity is bought
with bits, and a table sorted by fidelity alone will usually put the biggest quant on top. The
question worth asking of these three is not which is first, it is what the 4bpw pair cost relative to
each other -- 0.024555 against 0.027263 at 175.6 GB against 187.6 GB.

**VALID:** *"On brandonmusic's 25-window panel, our K6 (0.013723) is closer to the BF16 teacher than
his 4bpw (0.024555), which is in turn closer than the Dione Q4 (0.027263)."*
Same key. Same tokens, same teacher, same estimator, same surface. The comparison is exactly what the
numbers are for. (One of the three is his own measurement on his own stack, so the row is marked
`advisory` and the table says so -- but the panel and the teacher are provably identical, because his
receipt's `token_panel_receipt_sha256` and `teacher_receipt_sha256` are byte-identical to ours.)

**INVALID:** *"The official FP8 release (0.020615) beats his 4bpw (0.024555) and loses to our K6."*
Different key -- and it differs on two axes at once. The FP8 number came from replaying the model through **our** vLLM stack and scoring it
against a teacher captured on **his** transformers/eager stack. That is a `cross_stack` measurement and
it carries a stack-difference term on top of the quantization error. We know how big that term is,
because we measured it on the same panel: replaying the reference's own **unquantized BF16 weights**
through our stack scores **0.012712** against those same teacher logits. So 0.020615 is an upper bound,
not a result. The naive difference is 0.007904 -- an *estimate*, not an identity, because KL is not
additive. **This registry does not subtract floors and publish the remainder.** It puts the floor in
the table, in bold, labelled, immediately above the biased row.

### And one comparison the key alone does not stop

A comparability key has seven inputs and none of them is the measurement lane. Two rows can therefore
share a key -- same panel, same teacher, same metric, same direction, same float64, same
`same_stack`, same `native_head` -- and still have been produced on different machines by different
code paths. Group `cmp--202b717f3219c414` now contains exactly that: our K6 measured on the sealed
8x H200 lane at **0.013723384665701147**, and the *same weights* measured on a one-GPU streaming lane
at **0.013714888822596553**. Sorted into one list, the streaming row lands above the sealed one and
reads as a better quant. It is not a quant at all: it is one artifact, measured twice.

So the renderer tables a non-sealed lane's rows apart from the rest of its group, and the lane's
pipeline record carries the *measured* bridge to the sealed lane rather than an adjective: signed
delta **-8.4958e-06** nats on the panel mean, worst single window **2.8735e-04**,
`tokenwise_kld_sha256_matches_sealed: false`, `publishable_as_reproduction: false`. The last two are
the load-bearing ones. A mean that agrees to five decimal places is not a reproduction when the
per-token array underneath it differs, and the lane says so about itself.

That bridge is one artifact's, on one panel, and it is not subtractable. The 8bpw row in the same
lane has no sealed-lane sibling to bridge against, so its bias block records the offset as
`direction: unknown` and its magnitude as null -- which is what "we do not know" looks like when it
has to survive a schema.

The streaming lane also carries its own floor: the reference's own **unquantized BF16 weights**,
scored through this SAME streaming harness rather than the cross-stack replay pipeline. It reads
**0.011506** nats -- the cost of comparing across capture stacks plus bf16 non-associativity, with
zero quantization involved -- and it is emphatically NOT the cross-stack floor above (0.012712,
a different pipeline, a different lane, a different comparability key). Unlike the cross-stack case,
this registry DOES publish the netted-out number here, as an *Attributable (nats)* column in the
lane's own sub-table: K6-stream nets to 0.002209, K8-stream to 0.000878, a 2.52x spread against a
raw ratio of only 1.11x. It is still an estimate, not an identity -- KL is not additive -- but both
terms are small, share the same reference, and now also share the same lane, which the cross-stack
pair does not. `BIAS-006` is what keeps the two floors from ever crossing: a floor's
`floor_measurement_ref` must have been measured on the SAME lane as the row naming it, so the
cross-stack floor can never be subtracted from a streaming-lane row, nor this one from a
cross-stack row, even on the rare occasion the two share a comparability key. See
`k6/BF16-FLOOR.md` for the full analysis.

The second differing axis is the metric itself: the K6 / 4bpw / Dione rows are
`mean_of_run_means_tokenwise_kld` over five cold runs, while the cross-stack rows are a single
`mean_tokenwise_kld` pass. When a measurement is bitwise reproducible those two coincide numerically,
but they are not the same estimator in general -- brandonmusic's own v44 FP8 runs span 0.024016 to
0.024883 -- so the registry keeps them apart rather than assuming determinism it has not evidenced.

Ask the tool rather than reasoning it out yourself:

```
$ python3 tools/registry_validate.py \
    --explain measurement--glm53.k6-6bpw.brandonmusic-final25 \
    --against measurement--glm53.official-fp8.brandonmusic-final25.crossstack

NOT COMPARABLE. Differing comparability-key fields:
  metric_name         mean_of_run_means_tokenwise_kld
                      mean_tokenwise_kld
  stack_relation      same_stack
                      cross_stack
Everything else matches (panel_id, reference_id, direction, accumulation_dtype, head_policy).
measurement--glm53.official-fp8.brandonmusic-final25.crossstack declares bias.direction=upward with floor
measurement--glm53.bf16-replay-floor.brandonmusic-final25 (value 0.01271159981725071). Subtracting floors
is NOT sanctioned by this registry: the floor is context, not a correction.
```

A third case worth stating outright, because it is the one most likely to mislead: the MLX builds are
measured against the official FP8 release **dequantized to BF16**, not against a BF16 teacher. Their
6-bit reads `0.0063`, which is numerically smaller than our K6's `0.013723`. It is not better. It is a
different quantity -- the reference itself is quantized, so the FP8 error sits in the teacher instead of
in the student. Those rows carry `reference_kind: dequantized_from_quant`, a mandatory
`different_reference_kind` disclosure, and a panel marked `undisclosed`. They will never appear in a
table with a `native_bf16` row.

---

## What a row must carry

Every measurement names, and cannot validate without: a **model** and a pinned **artifact**; a **panel**
(its own first-class record: corpus lineage, context and position counts, tokenizer, contamination
guard, scored-position policy, token digest, availability); a **reference** -- modelled as a *capture*
`(artifact, panel, stack, logits precision, head source)`, so naming a teacher has already named a panel;
a **pipeline**; the **KL direction**; the **estimator** precision; the **run count with typed determinism
evidence**; the **measurement scope**; the **provenance**; the derived **comparability key**; and a
non-empty **disclosures** array.

Three of those deserve emphasis, because they are where fidelity registries usually go wrong.

**Determinism needs evidence, not a boolean.** A receipt file's own sha256 proves nothing about
numerics -- report files embed timestamps, paths and run indices, and differ across bit-identical runs.
Only *tensor content* digests can back a determinism claim, and the schema blocks the rest
(`DET-001`). The registry's own data is what taught it: in the K6 five-run receipt the five runs carry
five **different** `student_backend_identity_sha256` values (five genuinely distinct cold executions)
and one **identical** `tokenwise_kld_sha256`. Container hashes would have said "nondeterministic";
tensor content says "bitwise identical". Conversely brandonmusic's v44 FP8 rows report five *distinct*
tokenwise digests and a non-zero spread, and are recorded as not reproducible -- while his v44/v71/v75
NVFP4 rows report a single digest across five runs and are recorded as bitwise identical. Same author,
same panel, opposite verdicts, both evidenced.

**Who measured it is four separate facts, not one.** `provenance.measured_by` is
`self-measured | author-reported | third-party-reported`. `independently_verified` is a *separate*
boolean that is never implied by it, and setting it true requires a verifier who is a different party
than the measurer (`PROV-003`). Whether the *artifact* is ours is a third axis, carried by the
`third_party_artifact_self_measured` disclosure -- the Dione Q4 row is 0xSero's weights and our number,
and the table says exactly that. Whether the *panel* is ours is the fourth. Third-party numbers are
welcome here and are never silently merged with ours.

**Panels are identified by their tokens, and the scoring window is part of that identity.** Our GLM
suite scored from position 0 gives 0.028104; the *same tokens*, the *same artifact*, the *same teacher*,
scored from position 1024, gives 0.018794. A 33% move with nothing changed but which positions were
averaged. So the second one is a separate panel record with `derivation.kind: scoring_window_change`,
therefore a separate comparability key, therefore structurally unable to share a table with the first.

---

## Provenance notes on this seeding

Two things in this data are worth stating plainly rather than burying in a disclosure.

**brandonmusic's 25-window panel is genuinely sealed, and we verified it ourselves.** Its identity is
`panel.json` from his public teacher-logits dataset at revision `95f4fdd9`, sha256
`6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff` -- a manifest of 665 windows, each
with its own `token_ids_sha256`, of which 25 carry `role: final`. That digest was recomputed by
downloading the file during seeding and it matches the `token_panel_artifact_sha256` his own panel
receipt declares. The receipt's self-declared digest `0beec577...` is recorded separately in
`identity.panel_receipt_sha256` and is explicitly barred from being used as a token identity or as
determinism evidence (`PANEL-002`).

**That panel's contamination guard is weaker than ours, and the tables say so.** Its only guard is role
separation: the 25 `final` windows come from the same packed corpus as the 384 fit / 128 conditional-fit
/ 64 selection / 64 confirmation windows, and are declared qualification-only. No lexical or n-gram scan
is published. Our v5 suites run a 12-word shingle whole-document pre-exclusion against the calibration
corpus and report 0 hits out of 941 documents scanned, 44 excluded. Those are not the same guard, and
the validator warns whenever a `strict` row rests on a panel whose `contamination.checked` is false
(`PANEL-006`). It applies equally to every row on that panel, so it does not disturb comparisons
*within* it.

---

<!-- BEGIN GENERATED: tables -->

<!-- GENERATED BY tools/registry_render.py FROM data/*.jsonl -- DO NOT EDIT BY HAND.
     Every number below is rendered from a measurement row. Edit the row, then re-render. -->

## How to read the tables below

15 tables follow, one per comparability group, across 2 models. Three things are true of all of them, and each is a mistake somebody has already made with numbers like these:

1. **A number means nothing outside its own table.** Every table states the seven-part key its rows share. Two numbers under different keys are different quantities that happen to print in the same units.
2. **The smallest number on this page is not the best quant.** Today it is unsloth Qwen3.8-27B-GGUF BF16 at 0.000507355 nats -- and it is not a quant at all -- those are unquantized weights, read by a second engine, measuring what two engines disagree by. Sorting this file by value and reading off the top is the single easiest way to be wrong with it.
3. **Nothing here compares two models.** A KL divergence is measured over one model's own vocabulary against that model's own teacher. GLM-5.3-Flash's numbers and Qwen3.8-27B's numbers are not on a shared scale and never will be.

Attribution is a column, not a footnote: *measured by us*, *measured by us (their artifact)* and *reported by <name>* are three different epistemic states and the tables never merge them.

## Qwen3.8-27B

`model--qwen.qwen3.8-27b` -- published by Qwen (Alibaba). Tokenizer `qwen3.8`, vocabulary 248320.

### Panel: malaiwah Qwen3.8-27B distribution-fidelity suite v5 -- 5,120 contexts

> **Panel disclosure -- `unsealed_source`:** The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest c79dfad3..., but the token files themselves are not published, so a third party cannot reproduce the digest today.

#### Group `cmp--5f556b50b25762a2` -- 6 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-10m` -- malaiwah Qwen3.8-27B distribution-fidelity suite v5 -- 5,120 contexts
  5120 contexts x 2047 scored positions = **10,480,640 scored positions**, score_from 0
  sealed: **yes** (token digest `510541f6861b589d...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--5f556b50b25762a2`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a93702ded23e01a` (12 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-10m -> panel--qwen38.malaiwah.suite-v5-shard0-1m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m
> - `cmp--12bfc6ec82b47678` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-10m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024
> - `cmp--1ef6a9b5901f8e2a` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-10m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256
> - `cmp--1669ccf7958fb75c` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-10m -> panel--qwen38.malaiwah.suite-v5-shards01-2m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah Qwen3.8-27B EXL3 K5K6 hydrated | `exl3-mcg @5` | 21.6 GB | **0.00275963** | [0.00254024, 0.00302032] | 97.70 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 | `exl3-mcg @5` | 30.6 GB | **0.00320988** | [0.00298238, 0.00348017] | 97.52 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 context | `exl3-mcg @5` | 20.7 GB | **0.00350936** | [0.00321967, 0.00385239] | 97.44 % | 1 run, unevidenced | measured by us | local receipt |
| Qwen3.8-27B FP8 (official) | `fp8_e4m3 @8` | 30.9 GB | **0.00529394** | [0.00492736, 0.00572785] | 96.79 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B K4 | `exl3-mcg @4` | 28.3 GB | **0.0106039** | [0.00963981, 0.0117463] | 95.76 % | 1 run, unevidenced | measured by us | local receipt |
| unsloth Qwen3.8-27B NVFP4 | `nvfp4 @4` | -- | **0.0310586** | [0.0279161, 0.0347947] | 92.90 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |

> **The same artifact, measured elsewhere in this file.** 6 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 8%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 6 artifacts and their ranges</summary>
>
> - **Qwen3.8-27B FP8 (official)** -- 5 values here, from **0.00495487** to **0.00529563** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6** -- 5 values here, from **0.0030196** to **0.00320988** nats (6% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 context** -- 5 values here, from **0.00324322** to **0.00350936** nats (8% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 hydrated** -- 5 values here, from **0.00257964** to **0.00275963** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`.
> - **malaiwah Qwen3.8-27B K4** -- 5 values here, from **0.00987561** to **0.0106039** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`.
> - **unsloth Qwen3.8-27B NVFP4** -- 2 values here, from **0.0301154** to **0.0310586** nats (3% apart). Other tables: `cmp--4a93702ded23e01a`.
>
> </details>

<details><summary>Disclosures for the rows above (13)</summary>

- `qwen38.k5k6-hydrated.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-hydrated.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-context.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-context.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.official-fp8.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.official-fp8.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k4.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k4.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.unsloth-nvfp4.suite-v5-10m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.unsloth-nvfp4.suite-v5-10m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.unsloth-nvfp4.suite-v5-10m` **single_run**: One pass. Repeatability was not established for this row.

</details>

### Panel: malaiwah Qwen3.8-27B suite v5, shards 0-1 -- 1,024 contexts

Derived from `panel--qwen38.malaiwah.suite-v5-10m` by **shard_subset**: shards 0 and 1 of 10 (1,024 contexts, 495 source clusters).

> **Panel disclosure -- `unsealed_source`:** No combined token digest was published for the two-shard union; the two per-shard digests are recorded instead, which pin the content but are not a single panel identity.

> **Panel disclosure -- `unsealed_source`:** The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest c79dfad3..., but the token files themselves are not published, so a third party cannot reproduce the digest today.

#### Group `cmp--1669ccf7958fb75c` -- 5 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-shards01-2m` -- malaiwah Qwen3.8-27B suite v5, shards 0-1 -- 1,024 contexts
  1024 contexts x 2047 scored positions = **2,096,128 scored positions**, score_from 0
  sealed: **no** -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--1669ccf7958fb75c`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a93702ded23e01a` (12 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shards01-2m -> panel--qwen38.malaiwah.suite-v5-shard0-1m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m
> - `cmp--5f556b50b25762a2` (6 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shards01-2m -> panel--qwen38.malaiwah.suite-v5-10m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m
> - `cmp--12bfc6ec82b47678` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shards01-2m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024
> - `cmp--1ef6a9b5901f8e2a` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shards01-2m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah Qwen3.8-27B EXL3 K5K6 hydrated | `exl3-mcg @5` | 21.6 GB | **0.00275854** | [0.0025346, 0.00302484] | 97.75 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 | `exl3-mcg @5` | 30.6 GB | **0.00320026** | [0.00296909, 0.00347268] | 97.56 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 context | `exl3-mcg @5` | 20.7 GB | **0.00350243** | [0.00321244, 0.00384432] | 97.49 % | 1 run, unevidenced | measured by us | local receipt |
| Qwen3.8-27B FP8 (official) | `fp8_e4m3 @8` | 30.9 GB | **0.00529563** | [0.00492561, 0.00572892] | 96.85 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B K4 | `exl3-mcg @4` | 28.3 GB | **0.0105726** | [0.0096214, 0.0117039] | 95.83 % | 1 run, unevidenced | measured by us | local receipt |

> **The same artifact, measured elsewhere in this file.** 5 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 8%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 5 artifacts and their ranges</summary>
>
> - **Qwen3.8-27B FP8 (official)** -- 5 values here, from **0.00495487** to **0.00529563** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6** -- 5 values here, from **0.0030196** to **0.00320988** nats (6% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 context** -- 5 values here, from **0.00324322** to **0.00350936** nats (8% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 hydrated** -- 5 values here, from **0.00257964** to **0.00275963** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B K4** -- 5 values here, from **0.00987561** to **0.0106039** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
>
> </details>

<details><summary>Disclosures for the rows above (10)</summary>

- `qwen38.k5k6-hydrated.suite-v5-shards01-2m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-hydrated.suite-v5-shards01-2m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6.suite-v5-shards01-2m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6.suite-v5-shards01-2m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-context.suite-v5-shards01-2m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-context.suite-v5-shards01-2m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.official-fp8.suite-v5-shards01-2m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.official-fp8.suite-v5-shards01-2m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k4.suite-v5-shards01-2m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k4.suite-v5-shards01-2m` **single_run**: One pass. Repeatability was not established for this row.

</details>

### Panel: malaiwah Qwen3.8-27B suite v5, shard 0 -- 512 contexts

Derived from `panel--qwen38.malaiwah.suite-v5-10m` by **shard_subset**: shard 0 of 10 (512 of 5,120 contexts, 330 of 842 source clusters). Different tokens, therefore a different digest and a different comparability key. K6-parity 0.001634 lives here; the FP8 baseline on this panel is 0.005197, NOT the 10M panel's 0.005294.

> **Panel disclosure -- `unsealed_source`:** The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest c79dfad3..., but the token files themselves are not published, so a third party cannot reproduce the digest today.

This panel carries **2 separate comparability groups**. They are different measurements of different things and are never merged.

#### Group `cmp--4a93702ded23e01a` -- 12 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-shard0-1m` -- malaiwah Qwen3.8-27B suite v5, shard 0 -- 512 contexts
  512 contexts x 2047 scored positions = **1,048,064 scored positions**, score_from 0
  sealed: **yes** (token digest `caef8a4628d6c07c...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--4a93702ded23e01a`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--46a8a19f0fa33bed` (4 rows): `stack_relation` same_stack -> cross_stack
> - `cmp--5f556b50b25762a2` (6 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m -> panel--qwen38.malaiwah.suite-v5-10m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m
> - `cmp--12bfc6ec82b47678` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024
> - `cmp--1ef6a9b5901f8e2a` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| turboderp Qwen3.8-27B exl3 6.00bpw | `exl3-mcg @6` | 23.0 GB | **0.00158316** | [0.00149496, 0.00168324] | 98.28 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B EXL3 K6-parity | `exl3-mcg @6` | 23.1 GB | **0.00163382** | [0.00154118, 0.00174151] | 98.25 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 hydrated | `exl3-mcg @5` | 21.6 GB | **0.00269988** | [0.00251653, 0.00291183] | 97.80 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 | `exl3-mcg @5` | 30.6 GB | **0.00314136** | [0.00294675, 0.00336868] | 97.61 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 context | `exl3-mcg @5` | 20.7 GB | **0.00340941** | [0.00317041, 0.00368653] | 97.55 % | 1 run, unevidenced | measured by us | local receipt |
| turboderp Qwen3.8-27B exl3 5.00bpw | `exl3-mcg @5` | 19.9 GB | **0.00400463** | [0.00371442, 0.00433631] | 97.37 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| Qwen3.8-27B FP8 (official) | `fp8_e4m3 @8` | 30.9 GB | **0.00519706** | [0.00487991, 0.00555746] | 96.92 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B K4 | `exl3-mcg @4` | 28.3 GB | **0.0103453** | [0.00956259, 0.0112458] | 95.91 % | 1 run, unevidenced | measured by us | local receipt |
| Qwen3.8-27B AWQ-INT4 (upstream unattributed) | `awq @4` | -- | **0.0228179** | [0.0212457, 0.024624] | 93.94 % | 1 run, unevidenced | measured by us | local receipt |
| unsloth Qwen3.8-27B NVFP4 | `nvfp4 @4` | -- | **0.0301154** | [0.0276372, 0.0329647] | 93.16 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| gittensor-model-hub Qwen3.8-27B NVFP4 (RTX5090) | `nvfp4 @4` | 20.6 GB | **0.0621631** | [0.0584911, 0.0663596] | 89.85 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| Qwen3.8-27B MTP-NVFP4 (upstream unattributed) | `nvfp4 @4` | -- | **0.15128** | [0.141538, 0.163025] | 84.74 % | 1 run, unevidenced | measured by us | local receipt |

> **The same artifact, measured elsewhere in this file.** 6 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 8%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 6 artifacts and their ranges</summary>
>
> - **Qwen3.8-27B FP8 (official)** -- 5 values here, from **0.00495487** to **0.00529563** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6** -- 5 values here, from **0.0030196** to **0.00320988** nats (6% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 context** -- 5 values here, from **0.00324322** to **0.00350936** nats (8% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 hydrated** -- 5 values here, from **0.00257964** to **0.00275963** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B K4** -- 5 values here, from **0.00987561** to **0.0106039** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--5f556b50b25762a2`.
> - **unsloth Qwen3.8-27B NVFP4** -- 2 values here, from **0.0301154** to **0.0310586** nats (3% apart). Other tables: `cmp--5f556b50b25762a2`.
>
> </details>

<details><summary>Disclosures for the rows above (30)</summary>

- `qwen38.turboderp-6bpw.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.turboderp-6bpw.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.turboderp-6bpw.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k6-parity.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k6-parity.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-hydrated.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-hydrated.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-context.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-context.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.turboderp-5bpw.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.turboderp-5bpw.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.turboderp-5bpw.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.official-fp8.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.official-fp8.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k4.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k4.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.awq-int4.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.awq-int4.suite-v5-shard0-1m` **artifact_identity_incomplete**: The upstream repository for this artifact is not recorded by the receipt; only a local path. The measurement is ours and real, the artifact identity is not established.
- `qwen38.awq-int4.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.unsloth-nvfp4.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.unsloth-nvfp4.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.unsloth-nvfp4.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.gittensor-nvfp4.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.gittensor-nvfp4.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.gittensor-nvfp4.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.mtp-nvfp4.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.mtp-nvfp4.suite-v5-shard0-1m` **artifact_identity_incomplete**: The upstream repository for this artifact is not recorded by the receipt; only a local path. The measurement is ours and real, the artifact identity is not established.
- `qwen38.mtp-nvfp4.suite-v5-shard0-1m` **single_run**: One pass. Repeatability was not established for this row.

</details>

#### Group `cmp--46a8a19f0fa33bed` -- 4 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-shard0-1m` -- malaiwah Qwen3.8-27B suite v5, shard 0 -- 512 contexts
  512 contexts x 2047 scored positions = **1,048,064 scored positions**, score_from 0
  sealed: **yes** (token digest `caef8a4628d6c07c...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `cross_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--46a8a19f0fa33bed`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a93702ded23e01a` (12 rows): `stack_relation` cross_stack -> same_stack
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| **unsloth Qwen3.8-27B-GGUF BF16** _(measurement floor)_ | `bf16` | 54.7 GB | **0.000507355** | [0.000492078, 0.00052326] | 99.07 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| unsloth Qwen3.8-27B-GGUF Q8_0 | `gguf-k-quant @8` | 29.0 GB | **0.00108681** | [0.00105026, 0.00112685] | 98.53 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| unsloth Qwen3.8-27B-GGUF Q6_K | `gguf-k-quant @6` | 22.9 GB | **0.00203522** | [0.00193876, 0.00214482] | 97.98 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| unsloth Qwen3.8-27B-GGUF UD-Q5_K_XL | `gguf-k-quant @5` | 20.2 GB | **0.00444353** | [0.00415816, 0.00476989] | 97.20 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |

> **Bias on unsloth Qwen3.8-27B-GGUF BF16** -- cross_stack_capture_replay, direction upward. THIS ROW IS THE FLOOR. Unquantized BF16 weights read by llama.cpp and scored against the vLLM BF16 reference: what two engines disagree by on identical weights. 0.000507 nats, 99.07% top-1. Every GGUF row on this panel contains this term; no EXL3 or FP8 row does.

> **Bias on unsloth Qwen3.8-27B-GGUF Q8_0** -- cross_stack_capture_replay, direction upward. llama.cpp candidate capture vs vLLM reference capture. The cross-engine floor on this exact panel is 0.000507 nats, so this is an UPPER BOUND. Naive net of floor: 0.0005794503201991574 -- an estimate, not an identity, because KL is not additive.

> **Bias on unsloth Qwen3.8-27B-GGUF Q6_K** -- cross_stack_capture_replay, direction upward. llama.cpp candidate capture vs vLLM reference capture. The cross-engine floor on this exact panel is 0.000507 nats, so this is an UPPER BOUND. Naive net of floor: 0.0015278671188742878 -- an estimate, not an identity, because KL is not additive.

> **Bias on unsloth Qwen3.8-27B-GGUF UD-Q5_K_XL** -- cross_stack_capture_replay, direction upward. llama.cpp candidate capture vs vLLM reference capture. The cross-engine floor on this exact panel is 0.000507 nats, so this is an UPPER BOUND. Naive net of floor: 0.003936170795822309 -- an estimate, not an identity, because KL is not additive.

<details><summary>Disclosures for the rows above (16)</summary>

- `qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m` **cross_engine_capture**: The candidate was captured with llama.cpp; the reference and every EXL3/FP8 row on this panel were captured under vLLM. This number therefore contains a llama.cpp-vs-vLLM term on top of quantization error, which can only inflate it. That term is measured: 0.000507 nats.
- `qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m` **single_run**: One pass.
- `qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.gguf-bf16-engine-floor.suite-v5-shard0-1m` note: CONTROL ROW / CROSS-ENGINE FLOOR.
- `qwen38.unsloth-gguf-q8-0.suite-v5-shard0-1m` **cross_engine_capture**: The candidate was captured with llama.cpp; the reference and every EXL3/FP8 row on this panel were captured under vLLM. This number therefore contains a llama.cpp-vs-vLLM term on top of quantization error, which can only inflate it. That term is measured: 0.000507 nats.
- `qwen38.unsloth-gguf-q8-0.suite-v5-shard0-1m` **single_run**: One pass.
- `qwen38.unsloth-gguf-q8-0.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.unsloth-gguf-q8-0.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.unsloth-gguf-q6-k.suite-v5-shard0-1m` **cross_engine_capture**: The candidate was captured with llama.cpp; the reference and every EXL3/FP8 row on this panel were captured under vLLM. This number therefore contains a llama.cpp-vs-vLLM term on top of quantization error, which can only inflate it. That term is measured: 0.000507 nats.
- `qwen38.unsloth-gguf-q6-k.suite-v5-shard0-1m` **single_run**: One pass.
- `qwen38.unsloth-gguf-q6-k.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.unsloth-gguf-q6-k.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.
- `qwen38.unsloth-gguf-ud-q5-k-xl.suite-v5-shard0-1m` **cross_engine_capture**: The candidate was captured with llama.cpp; the reference and every EXL3/FP8 row on this panel were captured under vLLM. This number therefore contains a llama.cpp-vs-vLLM term on top of quantization error, which can only inflate it. That term is measured: 0.000507 nats.
- `qwen38.unsloth-gguf-ud-q5-k-xl.suite-v5-shard0-1m` **single_run**: One pass.
- `qwen38.unsloth-gguf-ud-q5-k-xl.suite-v5-shard0-1m` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.unsloth-gguf-ud-q5-k-xl.suite-v5-shard0-1m` **artifact_identity_incomplete**: The per-tensor-class quantization recipe for this artifact was never published, so scope.assignments records 'unknown' rather than a guessed allocation. Its scope_digest shows the gap.

</details>

### Panel: malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 256

Derived from `panel--qwen38.malaiwah.suite-v5-shard0-1m` by **scoring_window_change**: score_from 0 -> 256 on shard 0.

> **Panel disclosure -- `unsealed_source`:** The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest c79dfad3..., but the token files themselves are not published, so a third party cannot reproduce the digest today.

#### Group `cmp--1ef6a9b5901f8e2a` -- 5 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256` -- malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 256
  512 contexts x 1791 scored positions = **916,992 scored positions**, score_from 256, windowed
  sealed: **yes** (token digest `caef8a4628d6c07c...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--1ef6a9b5901f8e2a`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a93702ded23e01a` (12 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256 -> panel--qwen38.malaiwah.suite-v5-shard0-1m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m
> - `cmp--5f556b50b25762a2` (6 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256 -> panel--qwen38.malaiwah.suite-v5-10m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m
> - `cmp--12bfc6ec82b47678` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256 -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024
> - `cmp--1669ccf7958fb75c` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256 -> panel--qwen38.malaiwah.suite-v5-shards01-2m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah Qwen3.8-27B EXL3 K5K6 hydrated | `exl3-mcg @5` | 21.6 GB | **0.00265978** | [0.0024736, 0.00287669] | 97.84 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 | `exl3-mcg @5` | 30.6 GB | **0.00310033** | [0.00290065, 0.00333055] | 97.64 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 context | `exl3-mcg @5` | 20.7 GB | **0.00334231** | [0.00310111, 0.00362553] | 97.59 % | 1 run, unevidenced | measured by us | local receipt |
| Qwen3.8-27B FP8 (official) | `fp8_e4m3 @8` | 30.9 GB | **0.00509007** | [0.00477169, 0.00544966] | 96.97 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B K4 | `exl3-mcg @4` | 28.3 GB | **0.0101538** | [0.00936883, 0.0110793] | 95.98 % | 1 run, unevidenced | measured by us | local receipt |

> **The same artifact, measured elsewhere in this file.** 5 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 8%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 5 artifacts and their ranges</summary>
>
> - **Qwen3.8-27B FP8 (official)** -- 5 values here, from **0.00495487** to **0.00529563** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6** -- 5 values here, from **0.0030196** to **0.00320988** nats (6% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 context** -- 5 values here, from **0.00324322** to **0.00350936** nats (8% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 hydrated** -- 5 values here, from **0.00257964** to **0.00275963** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B K4** -- 5 values here, from **0.00987561** to **0.0106039** nats (7% apart). Other tables: `cmp--12bfc6ec82b47678`, `cmp--1669ccf7958fb75c`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
>
> </details>

<details><summary>Disclosures for the rows above (10)</summary>

- `qwen38.k5k6-hydrated.suite-v5-shard0-1m.scorefrom256` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-hydrated.suite-v5-shard0-1m.scorefrom256` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6.suite-v5-shard0-1m.scorefrom256` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6.suite-v5-shard0-1m.scorefrom256` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-context.suite-v5-shard0-1m.scorefrom256` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-context.suite-v5-shard0-1m.scorefrom256` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.official-fp8.suite-v5-shard0-1m.scorefrom256` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.official-fp8.suite-v5-shard0-1m.scorefrom256` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k4.suite-v5-shard0-1m.scorefrom256` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k4.suite-v5-shard0-1m.scorefrom256` **single_run**: One pass. Repeatability was not established for this row.

</details>

### Panel: malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 1024

Derived from `panel--qwen38.malaiwah.suite-v5-shard0-1m` by **scoring_window_change**: score_from 0 -> 1024 on shard 0.

> **Panel disclosure -- `unsealed_source`:** The qwen38 v5 token suite is pinned by suite_token_sha256 and by its manifest digest c79dfad3..., but the token files themselves are not published, so a third party cannot reproduce the digest today.

#### Group `cmp--12bfc6ec82b47678` -- 5 rows

**Panel** `panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024` -- malaiwah Qwen3.8-27B suite v5 shard 0, scored from position 1024
  512 contexts x 1023 scored positions = **523,776 scored positions**, score_from 1024, windowed
  sealed: **yes** (token digest `caef8a4628d6c07c...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024` -- native_bf16, artifact `artifact--qwen.qwen3.8-27b-bf16` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--12bfc6ec82b47678`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a93702ded23e01a` (12 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024 -> panel--qwen38.malaiwah.suite-v5-shard0-1m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m
> - `cmp--5f556b50b25762a2` (6 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024 -> panel--qwen38.malaiwah.suite-v5-10m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-10m
> - `cmp--1ef6a9b5901f8e2a` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024 -> panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom256; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom256
> - `cmp--1669ccf7958fb75c` (5 rows): `panel_id` panel--qwen38.malaiwah.suite-v5-shard0-1m.scorefrom1024 -> panel--qwen38.malaiwah.suite-v5-shards01-2m; `reference_id` reference--malaiwah.qwen38-bf16-vllm.suite-v5-shard0-1m.scorefrom1024 -> reference--malaiwah.qwen38-bf16-vllm.suite-v5-shards01-2m
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah Qwen3.8-27B EXL3 K5K6 hydrated | `exl3-mcg @5` | 21.6 GB | **0.00257964** | [0.00239759, 0.00278828] | 97.86 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 | `exl3-mcg @5` | 30.6 GB | **0.0030196** | [0.0028234, 0.00324563] | 97.68 % | 1 run, unevidenced | measured by us | local receipt |
| malaiwah Qwen3.8-27B EXL3 K5K6 context | `exl3-mcg @5` | 20.7 GB | **0.00324322** | [0.00300571, 0.00352013] | 97.62 % | 1 run, unevidenced | measured by us | local receipt |
| Qwen3.8-27B FP8 (official) | `fp8_e4m3 @8` | 30.9 GB | **0.00495487** | [0.00463566, 0.005316] | 97.02 % | 1 run, unevidenced | measured by us (their artifact) | local receipt |
| malaiwah Qwen3.8-27B K4 | `exl3-mcg @4` | 28.3 GB | **0.00987561** | [0.00910329, 0.0107555] | 96.04 % | 1 run, unevidenced | measured by us | local receipt |

> **The same artifact, measured elsewhere in this file.** 5 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 8%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 5 artifacts and their ranges</summary>
>
> - **Qwen3.8-27B FP8 (official)** -- 5 values here, from **0.00495487** to **0.00529563** nats (7% apart). Other tables: `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6** -- 5 values here, from **0.0030196** to **0.00320988** nats (6% apart). Other tables: `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 context** -- 5 values here, from **0.00324322** to **0.00350936** nats (8% apart). Other tables: `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B EXL3 K5K6 hydrated** -- 5 values here, from **0.00257964** to **0.00275963** nats (7% apart). Other tables: `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
> - **malaiwah Qwen3.8-27B K4** -- 5 values here, from **0.00987561** to **0.0106039** nats (7% apart). Other tables: `cmp--1669ccf7958fb75c`, `cmp--1ef6a9b5901f8e2a`, `cmp--4a93702ded23e01a`, `cmp--5f556b50b25762a2`.
>
> </details>

<details><summary>Disclosures for the rows above (10)</summary>

- `qwen38.k5k6-hydrated.suite-v5-shard0-1m.scorefrom1024` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-hydrated.suite-v5-shard0-1m.scorefrom1024` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6.suite-v5-shard0-1m.scorefrom1024` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6.suite-v5-shard0-1m.scorefrom1024` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k5k6-context.suite-v5-shard0-1m.scorefrom1024` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k5k6-context.suite-v5-shard0-1m.scorefrom1024` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.official-fp8.suite-v5-shard0-1m.scorefrom1024` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.official-fp8.suite-v5-shard0-1m.scorefrom1024` **single_run**: One pass. Repeatability was not established for this row.
- `qwen38.k4.suite-v5-shard0-1m.scorefrom1024` **revision_unpinned**: No measurement receipt for this artifact records a Hub revision. Every kld5 receipt records model_revision=null / model_revision_source='none'. Identity rests on index_sha256 and the per-shard sha256 map the receipt carries.
- `qwen38.k4.suite-v5-shard0-1m.scorefrom1024` **single_run**: One pass. Repeatability was not established for this row.

</details>


## GLM-5.3-Flash

`model--zai-org.glm-5.3-flash` -- published by Z.ai. Tokenizer `glm-5.3-flash`, vocabulary 154880.

### Panel: malaiwah GLM-5.3-Flash distribution-fidelity suite v5 -- 5,120 contexts

> **Panel disclosure -- `no_known_deviations`:** No deviation from this registry's default protocol is known for this record.

#### Group `cmp--9b009314102d9e8b` -- 1 row

**Panel** `panel--glm53.malaiwah.suite-v5-10m` -- malaiwah GLM-5.3-Flash distribution-fidelity suite v5 -- 5,120 contexts
  5120 contexts x 2047 scored positions = **10,480,640 scored positions**, score_from 0
  sealed: **yes** (token digest `2e0ea09683564554...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.glm53-bf16-vllm.suite-v5-10m` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.b1967181` @b1967181a3917ae70a437f4884748f6b8e3a1f4d
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--9b009314102d9e8b`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--e6cdd07242bdde05` (1 row): `panel_id` panel--glm53.malaiwah.suite-v5-10m -> panel--glm53.malaiwah.suite-v5-10m.scorefrom1024; `reference_id` reference--malaiwah.glm53-bf16-vllm.suite-v5-10m -> reference--malaiwah.glm53-bf16-vllm.suite-v5-10m.scorefrom1024
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.
>
> **Single-row group.** This number has nothing in the registry to be ranked against. It is a stated fact, not a placing.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| GLM-5.3-Flash official FP8 | `fp8_e4m3 @8` | 328.4 GB | **0.0281039** | [0.0272053, 0.0289822] | 94.27 % | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16.json) |

> **The same artifact, measured elsewhere in this file.** One of the artifacts below also carries a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 51%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **GLM-5.3-Flash official FP8** -- 4 values here, from **0.0186653** to **0.0281039** nats (51% apart). Other tables: `cmp--4a8630bdcadab97f`, `cmp--e6cdd07242bdde05`, `cmp--eee09298c558ab21`.

<details><summary>Disclosures for the rows above (1)</summary>

- `glm53.official-fp8.malaiwah-suite-v5-10m` **single_run**: One pass; determinism not established for this row.

</details>

### Panel: malaiwah GLM-5.3-Flash suite v5, scored from position 1024

Derived from `panel--glm53.malaiwah.suite-v5-10m` by **scoring_window_change**: score_from 0 -> 1024. Identical tokens, half the scored positions, and a materially different number: 0.028104 becomes 0.018794 on the same artifact and the same teacher. This is the clearest demonstration in the registry that the scored-position policy is part of panel identity.

> **Panel disclosure -- `no_known_deviations`:** No deviation from this registry's default protocol is known for this record.

#### Group `cmp--e6cdd07242bdde05` -- 1 row

**Panel** `panel--glm53.malaiwah.suite-v5-10m.scorefrom1024` -- malaiwah GLM-5.3-Flash suite v5, scored from position 1024
  5120 contexts x 1023 scored positions = **5,237,760 scored positions**, score_from 1024, windowed
  sealed: **yes** (token digest `2e0ea09683564554...`) -- contamination scan: **yes, 0 hits**
**Reference (teacher)** `reference--malaiwah.glm53-bf16-vllm.suite-v5-10m.scorefrom1024` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.b1967181` @b1967181a3917ae70a437f4884748f6b8e3a1f4d
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `shared_reference_head`
**Comparability key** `cmp--e6cdd07242bdde05`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--9b009314102d9e8b` (1 row): `panel_id` panel--glm53.malaiwah.suite-v5-10m.scorefrom1024 -> panel--glm53.malaiwah.suite-v5-10m; `reference_id` reference--malaiwah.glm53-bf16-vllm.suite-v5-10m.scorefrom1024 -> reference--malaiwah.glm53-bf16-vllm.suite-v5-10m
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.
>
> **Single-row group.** This number has nothing in the registry to be ranked against. It is a stated fact, not a placing.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| GLM-5.3-Flash official FP8 | `fp8_e4m3 @8` | 328.4 GB | **0.0187943** | [0.0180739, 0.0194941] | 95.12 % | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/report-fp8-vs-bf16-scorefrom1024.json) |

> **The same artifact, measured elsewhere in this file.** One of the artifacts below also carries a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 51%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **GLM-5.3-Flash official FP8** -- 4 values here, from **0.0186653** to **0.0281039** nats (51% apart). Other tables: `cmp--4a8630bdcadab97f`, `cmp--9b009314102d9e8b`, `cmp--eee09298c558ab21`.

<details><summary>Disclosures for the rows above (2)</summary>

- `glm53.official-fp8.malaiwah-suite-v5-10m.scorefrom1024` **single_run**: One pass; determinism not established.
- `glm53.official-fp8.malaiwah-suite-v5-10m.scorefrom1024` note: Same tokens, same artifact, same teacher as the 0.028104 row. Dropping the first 1024 scored positions of every context moves the number by 33%. That is why the scored-position policy is part of panel identity.

</details>

### Panel: brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows

> **Panel disclosure -- `weak_contamination_guard`:** This panel's only contamination guard is ROLE SEPARATION: the 25 'final' windows are drawn from the same packed corpus as the 384 fit / 128 conditional-fit / 64 selection / 64 confirmation windows and are declared qualification-only. No lexical or n-gram scan is published, and the underlying document provenance is published only as a digest. This is materially weaker than the malaiwah v5 suites, which run a 12-word shingle whole-document pre-exclusion and report 0 hits. Do not describe the two guards as equivalent. It applies equally to every row on this panel, so it does not disturb comparisons WITHIN the panel.

This panel carries **2 separate comparability groups**. They are different measurements of different things and are never merged.

#### Group `cmp--202b717f3219c414` -- 7 rows

**Panel** `panel--glm53.brandonmusic.final25` -- brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows
  25 contexts x 2047 scored positions = **51,175 scored positions**, score_from 0
  sealed: **yes** (token digest `6bafe3283c54bc93...`) -- contamination scan: **NOT RUN**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final25` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_of_run_means_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `native_head`
**Comparability key** `cmp--202b717f3219c414`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a8630bdcadab97f` (2 rows): `metric_name` mean_of_run_means_tokenwise_kld -> mean_tokenwise_kld; `stack_relation` same_stack -> cross_stack
> - `cmp--2b9c401d13806d7e` (4 rows): `panel_id` panel--glm53.brandonmusic.final25 -> panel--glm53.brandonmusic.final25-clean17; `reference_id` reference--brandonmusic.glm53-bf16-fp32-logits.final25 -> reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

> **4 of this group's 7 rows came off a different measurement lane** (`streaming`) and are tabled on their own below, not mixed into the ordering here. The key does not carry the lane; this file does.

| Artifact | Codec | Size | mean_of_run_means_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah GLM-5.3-Flash TR3 6bpw (K6) | `exl3-mcg @6` | 253.5 GB | **0.0137234** | [0.0111548, 0.0166267] | -- | 5 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-five-run-kld.json) |
| brandonmusic GLM-5.3-Flash tr3 4bpw | `exl3-mcg @4` | 175.6 GB | **0.0245546** | [0.019433, 0.035881] | -- | 5 runs, bitwise identical | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json) |
| 0xSero GLM-5.3-Flash EXL3 Q4 (Dione, TP4-sliced) | `exl3-mcg @4` | 187.6 GB | **0.0272628** | -- | -- | 5 runs, bitwise identical | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/dione-q4-packed-kld.json) |

##### Lane `streaming` -- 4 of this group's 7 rows

> **A different lane. Same key, and that is exactly the problem this table solves.** The comparability key is a function of the panel, the teacher, the metric, the direction, the estimator precision, the stack relation and the head policy -- and these rows match the table above on all seven. What they do not share is the machine and the code path that produced the candidate logits, and lanes are not interchangeable. Sorting them into one list would read as a ranking; where the same artifact appears in both, it is one set of weights measured twice, not two quants.
>
> **What the lane is** (from `pipeline--malaiwah.glm53-stream-packed-kld`): **1 device**, expert-parallel width **8 emulated in one process**, routed-expert combine order `fp32`.
>
> **Bridge to the `sealed-ep8` lane, measured on this panel:** signed delta **-8.4958e-06** nats on the mean against `measurement--glm53.k6-6bpw.brandonmusic-final25`, worst single window **0.00028735**, over 25 windows. Tokenwise KL array matches the sealed run: **no**. The runner's own verdict on whether this may be published as a reproduction of the sealed number: **no** (verdict `LARGER_DELTA_SEE_DISCLOSURE`).
>
> That bridge is one artifact's, on one panel. It is not a constant and it is not subtractable: a row in this table whose artifact has no sealed-lane row has no measured offset at all, and says so in its own bias line.

| Artifact | Codec | Size | mean_of_run_means_tokenwise_kld (nats) | Attributable (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---:|---|---:|---|---|---|
| **GLM-5.3-Flash BF16 @a6c167b6** _(measurement floor)_ | `bf16` | -- | **0.0115059** | -- | -- | -- | 2 runs, bitwise identical | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-bf16-kld.json) |
| malaiwah GLM-5.3-Flash TR3 8bpw (K8) | `exl3-mcg @8` | 331.4 GB | **0.0123842** | 0.000878268 | [0.00999057, 0.0152193] | -- | 2 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-k8-kld.json) |
| malaiwah GLM-5.3-Flash TR3 6bpw (K6) | `exl3-mcg @6` | 253.5 GB | **0.0137149** | 0.00220897 | [0.0111567, 0.01663] | 96.56 % | 2 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-k6-kld.json) |
| turboderp GLM-5.3-Flash EXL3 4.05bpw (stock exllamav3, mul1, quantized head) | `exl3-mul1 @4.05` | 165.2 GB | **0.0255264** | 0.0140205 | -- | 95.10 % | 2 runs, bitwise identical | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-turbo-4.05bpw-kld.json) |

> **Attributable (nats)** = this row's value minus its named floor's value (`comparability.bias.floor_measurement_ref`) -- the raw number with this lane's own measurement floor netted out. It is an estimate, not an identity: KL is not additive, and the subtraction is only meaningful because both terms are small and share the same reference and the same lane. A row with no floor named shows `--`, not zero: absence of a floor is not evidence the floor is zero. BIAS-002/004/006 guarantee any floor named here shares this row's comparability key, measures unquantized weights, and was measured on this row's own lane -- so this column can never mix a floor from a different panel, a different kind of thing, or a different lane into the subtraction.

> **Bias on GLM-5.3-Flash BF16 @a6c167b6** -- other, direction unknown. THIS ROW IS THE FLOOR for the 'streaming' lane: it replays the reference's own unquantized weights through the SAME streaming harness that scored every other row on this pipeline, so its divergence against the stored teacher logits is the lane's zero-point, not a quantization result. It is NOT the cross-stack floor recorded elsewhere in this registry (a different pipeline, a different lane, a different comparability key) and is never interchangeable with it: subtracting one lane's floor from another lane's row is exactly the mistake BIAS-006 exists to catch. The lane's offset against the sealed-ep8 lane is NOT measured for this artifact: no sealed-lane counterpart to this profile exists to bridge against.

> **Bias on malaiwah GLM-5.3-Flash TR3 8bpw (K8)** -- other, direction unknown. Measured on the 'streaming' lane, whose offset against the sealed-ep8 lane is known to be non-zero but was NOT measured for this artifact: no sealed-lane row for it exists to bridge against. The lane offset measured for a sibling artifact on this panel is not transferable -- it is a property of the routing, not a constant. This lane's own measurement floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) is 0.011505922619330299 nats; netting it out gives an estimated quantization-attributable error of 0.0008782684041065674 nats here -- an estimate, not an identity, because KL is not additive, and it is only meaningful because both terms are small and share the same reference and lane.

> **Bias on malaiwah GLM-5.3-Flash TR3 6bpw (K6)** -- other, direction downward. Lane offset, MEASURED not estimated: this 'streaming'-lane run scores 0.013714888822596553 against the sealed-ep8 lane's 0.013723384665701147 on the same panel, a signed delta of -8.495843104593809e-06 nats (|max| 0.00028735280093581186 on any one of 25 windows). The tokenwise KL array does NOT match the sealed one, and the runner's own verdict is publishable_as_reproduction=False, so this number stands beside the sealed one rather than replacing it. This lane's own measurement floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) is 0.011505922619330299 nats; netting it out gives an estimated quantization-attributable error of 0.0022089662032662542 nats here -- an estimate, not an identity, because KL is not additive, and it is only meaningful because both terms are small and share the same reference and lane.

> **Bias on turboderp GLM-5.3-Flash EXL3 4.05bpw (stock exllamav3, mul1, quantized head)** -- other, direction unknown. Measured on the 'streaming' lane, whose offset against the sealed-ep8 lane is known to be non-zero but was NOT measured for this artifact: no sealed-lane row for it exists to bridge against. This lane's own measurement floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) is 0.011505922619330299 nats; netting it out gives an estimated quantization-attributable error of 0.014020504296142185 nats here -- an estimate, not an identity, because KL is not additive, and it is only meaningful because both terms share the same reference and lane.

> **The same artifact, measured elsewhere in this file.** 4 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 19%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 4 artifacts and their ranges</summary>
>
> - **GLM-5.3-Flash BF16 @a6c167b6** -- 3 values here, from **0.0106476** to **0.0127116** nats (19% apart). Other tables: `cmp--4a8630bdcadab97f`, `cmp--eee09298c558ab21`.
> - **brandonmusic GLM-5.3-Flash tr3 4bpw** -- 3 values here, from **0.0227508** to **0.0249488** nats (10% apart). Other tables: `cmp--18990ab191ea7a67`, `cmp--2b9c401d13806d7e`.
> - **malaiwah GLM-5.3-Flash TR3 6bpw (K6)** -- 6 values here, from **0.011676** to **0.0137234** nats (18% apart). Other tables: `cmp--2b9c401d13806d7e`.
> - **malaiwah GLM-5.3-Flash TR3 8bpw (K8)** -- 2 values here, from **0.0108294** to **0.0123842** nats (14% apart). Other tables: `cmp--2b9c401d13806d7e`.
>
> </details>

<details><summary>Disclosures for the rows above (19)</summary>

- `glm53.bf16-stream-floor.brandonmusic-final25` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.bf16-stream-floor.brandonmusic-final25` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset against the sealed lane is NOT measured for this artifact: no sealed-lane row for it exists to bridge against. This row is itself the streaming lane's measurement floor -- the zero-point the K6-stream and K8-stream rows in this same table subtract to obtain their own quantization-attributable error (see their bias blocks).
- `glm53.bf16-stream-floor.brandonmusic-final25` note: CONTROL ROW / STREAMING-LANE MEASUREMENT FLOOR. Not the cross-stack floor (measurement--glm53.bf16-replay-floor.brandonmusic-final25, 0.012712 nats, pipeline--malaiwah.glm53-crosscheck): a different pipeline, a different lane, a different comparability key -- BIAS-002 already keeps the two apart by key, and BIAS-006 additionally forbids naming one as the other's floor even inside a shared key. Provenance of the fields the summary receipt does not carry: metric.direction and estimator.accumulation_dtype are SUPPLIED as reference_to_candidate / float64, matching every other row on this pipeline, because the scorer is the same unmodified tools/k6_kld_report.py. measurement_scope.scored_positions and contexts are SUPPLIED as the panel's own 51,175 positions over 25 contexts (25 x 2047) -- like the K8-stream row, no verdict receipt exists for this profile to read the window count from. determinism.identical_across_runs is RECOMPUTED from run_means and distinct_tokenwise_kld_sha256; the receipt's own bitwise_deterministic flag was checked against that, not copied. cold_run_count (2) was checked against len(run_means) and len(kld_report_sha256), both 2. No clean17 sibling: receipt registry/receipts/malaiwah/stream-bf16-kld.json is scalar-only (run_means + a tokenwise digest, no per_window block), so the calibration-clean scope cannot be recomputed without re-running the measurement.
- `glm53.k8-8bpw-stream.brandonmusic-final25` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.k8-8bpw-stream.brandonmusic-final25` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset against the sealed lane is NOT measured for this artifact: no sealed-lane row for it exists to bridge against.
- `glm53.k8-8bpw-stream.brandonmusic-final25` note: This receipt does not name its lane. Its schema string is malaiwah.glm53-k8-packed-kld-summary.v1 and its profile reads 'k8-tp4' -- neither carries the '-stream-' marker the K6 summary's family name does -- so 'streaming' here is OPERATOR-ASSERTED (operator inventory, 2026-08-28) and not read off the file. It is recorded as the more caveated of the two possibilities on purpose: if the assertion is wrong the row is under-claimed, never over-claimed. Also supplied rather than read: metric.direction, estimator.accumulation_dtype, measurement_scope.scored_positions and contexts -- this family is a scalar summary and states none of them, and unlike the K6 row there is no verdict receipt here to read the window count from. No top-1 agreement was produced for this run. determinism.identical_across_runs is RECOMPUTED from run_means and distinct_tokenwise_kld_sha256. comparability.bias.floor_measurement_ref: SUPPLIED by --floor-measurement once the streaming-lane floor row below existed; build_row checked it was measured on this SAME lane before writing the reference (exit 7 otherwise).
- `glm53.k6-6bpw-stream.brandonmusic-final25` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.k6-6bpw-stream.brandonmusic-final25` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. On this panel the lane's offset against the sealed lane IS measured: -8.495843104593809e-06 nats on the mean (max 0.00028735280093581186 on any one window over 25 windows), and the tokenwise KL array is NOT the sealed one, so the run is not a reproduction of the sealed number.
- `glm53.k6-6bpw-stream.brandonmusic-final25` note: Provenance of the fields the summary receipt does not carry. metric.direction and estimator.accumulation_dtype: SUPPLIED -- the k6-stream summary states neither, and both are recorded as the sealed lane's because the scorer is the same unmodified tools/k6_kld_report.py, invoked as --profile k6-stream. measurement_scope.contexts: READ from the verdict receipt's 25-entry per_window array, whose streaming means average to exactly the summary's measured_mean_kld. scored_positions: SUPPLIED as the panel's own 51,175 (25 x 2047), which the equal-weighted window average is consistent with. determinism.identical_across_runs: RECOMPUTED from run_means and distinct_tokenwise_kld_sha256; the receipt's bitwise_deterministic flag was checked against that, not copied. The verdict's sealed_mean_kld is bit-identical to the sealed K6 row in this file, which is what makes the delta a comparison of these two rows and not of two unrelated numbers. comparability.bias.floor_measurement_ref: SUPPLIED by --floor-measurement once the streaming-lane floor row below existed; build_row checked it was measured on this SAME lane before writing the reference (exit 7 otherwise).
- `glm53.brandonmusic-4bpw.brandonmusic-final25` **author_reported_only**: Measured and published by brandonmusic on his own stack. We have not re-run it. It is nonetheless unusually well anchored: his receipt's token_panel_receipt_sha256 (0beec577...) and teacher_receipt_sha256 (2ae08117...) are byte-identical to ours, so the panel and the teacher are provably the same. Only the reader differs (1fb3be87... vs our 1ccce446...).
- `glm53.brandonmusic-4bpw.brandonmusic-final25` note: On the single-window sub-panel the same artifact reads 0.022751 -- a 7% swing from 0.024555 over the full 25 windows.
- `glm53.turbo-4.05bpw-stream.brandonmusic-final25` **unsealed_source**: seal_disclosure (verbatim from the receipt): unsealed-source scoring: stock exllamav3 releases ship no upstream receipts, reconstruction closures or sealed reader ABI; the packed surface was decoded WITHOUT seal verification (consumed payload sha256s and the immutable repo revision are recorded instead)
- `glm53.turbo-4.05bpw-stream.brandonmusic-final25` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.turbo-4.05bpw-stream.brandonmusic-final25` **quantized_head**: declared_head_bits 6 (verbatim from the receipt): this artifact's lm_head is itself quantized by the producer, unlike the TR3 artifacts on this panel which keep it native BF16. It is APPLIED natively from the artifact's own weights -- no shared or replayed head -- so estimator.head_policy is native_head; the quantization is artifact identity.
- `glm53.turbo-4.05bpw-stream.brandonmusic-final25` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset against the sealed lane is NOT measured for this artifact: no sealed-lane row for it exists to bridge against.
- `glm53.turbo-4.05bpw-stream.brandonmusic-final25` note: First artifact measured end to end by bin/measure-cloud. The receipt's family name carries no lane marker, so 'streaming' is SUPPLIED by --lane; direction, accumulation dtype, scored positions and context count are supplied too (this family is a scalar summary). determinism.identical_across_runs is RECOMPUTED from run_means and distinct_tokenwise_kld_sha256. The non-routed weights are the ARTIFACT's own, dequantized from its shards -- including its 6-bit head -- so no official-release weight is in the measured function; the materialization receipt is 3653c55f0dc729c3fccc6bbe5d8949b55e27517ade5d8c546fec79de03dd1c81. 907,200 K4 expert matrices were decoded per cold run. Top-1 agreement 0.9509916951636541, identical across both cold runs, read from the per-run kld-report.json (the scalar summary family did not carry it at the time this row was written; k6_kld_report now emits it).
- `glm53.dione-q4.brandonmusic-final25` **unsealed_source**: The Dione checkpoint ships no upstream receipts or sealed reader ABI. The packed surface was decoded without seal verification; the immutable revision 99cccdf0... and the consumed payload sha256s were recorded instead (dione_shard_hash_verification: full).
- `glm53.dione-q4.brandonmusic-final25` **artifact_identity_incomplete**: The release's own scope manifest was not parsed into this registry, so the artifact's per-class recipe is recorded as unknown.
- `glm53.dione-q4.brandonmusic-final25` note: The receipt's cold_run_deviation field reads verbatim '5 cold runs, not 5 (budget; disclosed)' -- a self-contradictory template string. cold_run_count is 5 and run_means has 5 entries, so five runs is what happened; the string is a receipt-generator defect and is recorded here rather than copied into a disclosure. No clean17 sibling: receipt reports/dione-q4-packed-kld.json is scalar-only (no per_window block), so the calibration-clean scope cannot be recomputed without re-running the measurement.

</details>

#### Group `cmp--4a8630bdcadab97f` -- 2 rows

**Panel** `panel--glm53.brandonmusic.final25` -- brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows
  25 contexts x 2047 scored positions = **51,175 scored positions**, score_from 0
  sealed: **yes** (token digest `6bafe3283c54bc93...`) -- contamination scan: **NOT RUN**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final25` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `cross_stack`, head_policy `native_head`
**Comparability key** `cmp--4a8630bdcadab97f`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--eee09298c558ab21` (2 rows): `panel_id` panel--glm53.brandonmusic.final25 -> panel--glm53.brandonmusic.final25-clean17; `reference_id` reference--brandonmusic.glm53-bf16-fp32-logits.final25 -> reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
> - `cmp--202b717f3219c414` (7 rows): `metric_name` mean_tokenwise_kld -> mean_of_run_means_tokenwise_kld; `stack_relation` cross_stack -> same_stack
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| **GLM-5.3-Flash BF16 @a6c167b6** _(measurement floor)_ | `bf16` | -- | **0.0127116** | [0.0103287, 0.0153918] | 96.65 % | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/crosscheck-brandonmusic.json) |
| GLM-5.3-Flash official FP8 | `fp8_e4m3 @8` | 328.4 GB | **0.0206153** | [0.0164689, 0.0256239] | 95.63 % | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/fp8-on-brandon-panel.json) |

> **Bias on GLM-5.3-Flash BF16 @a6c167b6** -- cross_stack_capture_replay, direction upward. THIS ROW IS THE FLOOR. It replays the reference's own BF16 weights through our vLLM stack and scores them against brandonmusic's stored fp32 teacher logits. 0.012712 nats is therefore what two stacks disagree by on identical unquantized weights -- not a quantization result. No floor is named because none exists below it.

> **Bias on GLM-5.3-Flash official FP8** -- cross_stack_capture_replay, direction upward. Teacher captured on brandonmusic's transformers/eager stack, candidate replayed on our vLLM stack. The same-stack BF16 replay floor on this exact panel is 0.012712, so this number is an UPPER BOUND on the FP8 release's own divergence. The naive difference is 0.007904 -- an estimate, not an identity, because KL is not additive. Do not subtract and publish.

> **The same artifact, measured elsewhere in this file.** 2 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 51%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **GLM-5.3-Flash BF16 @a6c167b6** -- 3 values here, from **0.0106476** to **0.0127116** nats (19% apart). Other tables: `cmp--202b717f3219c414`, `cmp--eee09298c558ab21`.
> - **GLM-5.3-Flash official FP8** -- 4 values here, from **0.0186653** to **0.0281039** nats (51% apart). Other tables: `cmp--9b009314102d9e8b`, `cmp--e6cdd07242bdde05`, `cmp--eee09298c558ab21`.

<details><summary>Disclosures for the rows above (5)</summary>

- `glm53.bf16-replay-floor.brandonmusic-final25` **cross_stack_capture**: Teacher captured on transformers/eager (B200 x4); candidate replayed on our vLLM stack. The offset audit confirms position alignment: top-1 agreement is 0.9665 at offset 0 and 0.0159 / 0.0162 at offsets -1 / +1.
- `glm53.bf16-replay-floor.brandonmusic-final25` **single_run**: One pass; determinism not established.
- `glm53.bf16-replay-floor.brandonmusic-final25` note: CONTROL ROW / MEASUREMENT FLOOR. Every cross-stack row on this panel contains this term.
- `glm53.official-fp8.brandonmusic-final25.crossstack` **cross_stack_capture**: This row cannot be ranked against the K6 / Dione / 4bpw rows on the same panel: those are same-stack sealed-capture numbers and this is a cross-stack replay. Their comparability keys differ, and the registry's tables are grouped by that key.
- `glm53.official-fp8.brandonmusic-final25.crossstack` **single_run**: One pass; determinism not established.

</details>

### Panel: brandonmusic panel v1, calibration-clean subset -- 17 of 25 final windows

Derived from `panel--glm53.brandonmusic.final25` by **shard_subset**: The 17 of 25 sealed windows whose 13-gram overlap with the calibration-role windows is at or below 0.05. Dropped: final-0003, final-0007, final-0011, final-0015, final-0019, final-0021, final-0022, final-0023. Six of the eight are axis4_reasoning_termination at 37-39% overlap, which removes that domain entirely; the other two (final-0021 at 7.1%, final-0022 at 5.8%) are legal and code-agentic, so this is NOT a whole-domain drop and a 19-window axis4-only exclusion is a DIFFERENT scope. The highest overlap among the retained windows is 4.75% (final-0014), so the 5% threshold separates cleanly but not by much -- it is inherited from brandonmusic and is an open joint decision, not a derived constant.

> **Panel disclosure -- `subset_of_panel`:** 17 of the parent panel's 25 windows, 34799 of 51,175 scored positions. Rows on this panel must never be tabled beside rows on the parent panel: excluding the contaminated windows moves different contributors' numbers in OPPOSITE directions (every malaiwah row falls 12.6-16.2%, brandonmusic's own 4bpw row rises 1.6%).

> **Panel disclosure -- `calibration_panel_overlap`:** This panel exists because the parent's contamination guard was role separation only. The n-gram scan that produced it found one whole domain sharing 37-39% of its 13-grams with calibration-role windows despite clean document-level separation -- the finding is brandonmusic's and the reproduction is ours.

This panel carries **2 separate comparability groups**. They are different measurements of different things and are never merged.

#### Group `cmp--2b9c401d13806d7e` -- 4 rows

**Panel** `panel--glm53.brandonmusic.final25-clean17` -- brandonmusic panel v1, calibration-clean subset -- 17 of 25 final windows
  17 contexts x 2047 scored positions = **34,799 scored positions**, score_from 0
  sealed: **yes** (token digest `ecfce5997ab9106c...`) -- contamination scan: **yes, 8 hits**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_of_run_means_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `native_head`
**Comparability key** `cmp--2b9c401d13806d7e`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--eee09298c558ab21` (2 rows): `metric_name` mean_of_run_means_tokenwise_kld -> mean_tokenwise_kld; `stack_relation` same_stack -> cross_stack
> - `cmp--202b717f3219c414` (7 rows): `panel_id` panel--glm53.brandonmusic.final25-clean17 -> panel--glm53.brandonmusic.final25; `reference_id` reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17 -> reference--brandonmusic.glm53-bf16-fp32-logits.final25
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

> **2 of this group's 4 rows came off a different measurement lane** (`streaming`) and are tabled on their own below, not mixed into the ordering here. The key does not carry the lane; this file does.

| Artifact | Codec | Size | mean_of_run_means_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah GLM-5.3-Flash TR3 6bpw (K6) | `exl3-mcg @6` | 253.5 GB | **0.0116773** | [0.00885591, 0.0147851] | -- | 5 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/k6-five-run-kld.json) |
| brandonmusic GLM-5.3-Flash tr3 4bpw | `exl3-mcg @4` | 175.6 GB | **0.0249488** | [0.0181407, 0.0416622] | -- | 5 runs, bitwise identical | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/five-cold-run-kld.json) |

##### Lane `streaming` -- 2 of this group's 4 rows

> **A different lane. Same key, and that is exactly the problem this table solves.** The comparability key is a function of the panel, the teacher, the metric, the direction, the estimator precision, the stack relation and the head policy -- and these rows match the table above on all seven. What they do not share is the machine and the code path that produced the candidate logits, and lanes are not interchangeable. Sorting them into one list would read as a ranking; where the same artifact appears in both, it is one set of weights measured twice, not two quants.
>
> **What the lane is** (from `pipeline--malaiwah.glm53-stream-packed-kld`): **1 device**, expert-parallel width **8 emulated in one process**, routed-expert combine order `fp32`.
>
> **Bridge to the `sealed-ep8` lane, measured on this panel:** signed delta **-8.4958e-06** nats on the mean against `measurement--glm53.k6-6bpw.brandonmusic-final25`, worst single window **0.00028735**, over 25 windows. Tokenwise KL array matches the sealed run: **no**. The runner's own verdict on whether this may be published as a reproduction of the sealed number: **no** (verdict `LARGER_DELTA_SEE_DISCLOSURE`).
>
> That bridge is one artifact's, on one panel. It is not a constant and it is not subtractable: a row in this table whose artifact has no sealed-lane row has no measured offset at all, and says so in its own bias line.

| Artifact | Codec | Size | mean_of_run_means_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| malaiwah GLM-5.3-Flash TR3 8bpw (K8) | `exl3-mcg @8` | 331.4 GB | **0.0108294** | [0.0080632, 0.0138378] | -- | 2 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-k8-kld.json) |
| malaiwah GLM-5.3-Flash TR3 6bpw (K6) | `exl3-mcg @6` | 253.5 GB | **0.011676** | [0.00886271, 0.0147792] | -- | 2 runs, bitwise identical | measured by us | [receipt](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/resolve/main/receipts/malaiwah/stream-k6-kld.json) |

> **Bias on malaiwah GLM-5.3-Flash TR3 8bpw (K8)** -- other, direction unknown. Measured on the 'streaming' lane, whose offset against the sealed-ep8 lane is known to be non-zero but was NOT measured for this artifact: no sealed-lane row for it exists to bridge against. The lane offset measured for a sibling artifact on this panel is not transferable -- it is a property of the routing, not a constant. This lane's own measurement floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) is 0.011505922619330299 nats; netting it out gives an estimated quantization-attributable error of 0.0008782684041065674 nats here -- an estimate, not an identity, because KL is not additive, and it is only meaningful because both terms are small and share the same reference and lane. NO FLOOR ON THIS SCOPE: the same-lane floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) has a scalar-only receipt with no per-window array, so it cannot be recomputed on the calibration-clean window set. Rather than borrow the panel25 floor -- a cross-scope subtraction -- this row carries no floor reference at all.

> **Bias on malaiwah GLM-5.3-Flash TR3 6bpw (K6)** -- other, direction downward. Lane offset, MEASURED not estimated: this 'streaming'-lane run scores 0.013714888822596553 against the sealed-ep8 lane's 0.013723384665701147 on the same panel, a signed delta of -8.495843104593809e-06 nats (|max| 0.00028735280093581186 on any one of 25 windows). The tokenwise KL array does NOT match the sealed one, and the runner's own verdict is publishable_as_reproduction=False, so this number stands beside the sealed one rather than replacing it. This lane's own measurement floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) is 0.011505922619330299 nats; netting it out gives an estimated quantization-attributable error of 0.0022089662032662542 nats here -- an estimate, not an identity, because KL is not additive, and it is only meaningful because both terms are small and share the same reference and lane. NO FLOOR ON THIS SCOPE: the same-lane floor (measurement--glm53.bf16-stream-floor.brandonmusic-final25) has a scalar-only receipt with no per-window array, so it cannot be recomputed on the calibration-clean window set. Rather than borrow the panel25 floor -- a cross-scope subtraction -- this row carries no floor reference at all.

> **The same artifact, measured elsewhere in this file.** 3 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 18%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> <details><summary>the 3 artifacts and their ranges</summary>
>
> - **brandonmusic GLM-5.3-Flash tr3 4bpw** -- 3 values here, from **0.0227508** to **0.0249488** nats (10% apart). Other tables: `cmp--18990ab191ea7a67`, `cmp--202b717f3219c414`.
> - **malaiwah GLM-5.3-Flash TR3 6bpw (K6)** -- 6 values here, from **0.011676** to **0.0137234** nats (18% apart). Other tables: `cmp--202b717f3219c414`.
> - **malaiwah GLM-5.3-Flash TR3 8bpw (K8)** -- 2 values here, from **0.0108294** to **0.0123842** nats (14% apart). Other tables: `cmp--202b717f3219c414`.
>
> </details>

<details><summary>Disclosures for the rows above (13)</summary>

- `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. The lane's offset against the sealed lane is NOT measured for this artifact: no sealed-lane row for it exists to bridge against.
- `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.k8-8bpw-stream.brandonmusic-final25.clean17` note: Calibration-clean scope recompute of measurement--glm53.k8-8bpw-stream.brandonmusic-final25. panel25 0.012384191023437 -> clean17 0.010829419869883 (-0.001554771153554, -12.55%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.
- `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` **reduced_run_count**: cold_run_deviation (verbatim from the receipt): 2 cold runs, not 5 (budget; disclosed)
- `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` **non_sealed_lane**: Produced by the 'streaming' lane, not the sealed-ep8 lane. On this panel the lane's offset against the sealed lane IS measured: -8.495843104593809e-06 nats on the mean (max 0.00028735280093581186 on any one window over 25 windows), and the tokenwise KL array is NOT the sealed one, so the run is not a reproduction of the sealed number.
- `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.k6-6bpw-stream.brandonmusic-final25.clean17` note: Calibration-clean scope recompute of measurement--glm53.k6-6bpw-stream.brandonmusic-final25. panel25 0.013714888822597 -> clean17 0.011675992693735 (-0.002038896128862, -14.87%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.
- `glm53.k6-6bpw.brandonmusic-final25.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.k6-6bpw.brandonmusic-final25.clean17` note: Calibration-clean scope recompute of measurement--glm53.k6-6bpw.brandonmusic-final25. panel25 0.013723384665701 -> clean17 0.011677286368695 (-0.002046098297006, -14.91%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.
- `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` **author_reported_only**: Measured and published by brandonmusic on his own stack. We have not re-run it. It is nonetheless unusually well anchored: his receipt's token_panel_receipt_sha256 (0beec577...) and teacher_receipt_sha256 (2ae08117...) are byte-identical to ours, so the panel and the teacher are provably the same. Only the reader differs (1fb3be87... vs our 1ccce446...).
- `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.brandonmusic-4bpw.brandonmusic-final25.clean17` note: Calibration-clean scope recompute of measurement--glm53.brandonmusic-4bpw.brandonmusic-final25. panel25 0.024554564249958 -> clean17 0.024948837055615 (+0.000394272805657, +1.61%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.

</details>

#### Group `cmp--eee09298c558ab21` -- 2 rows

**Panel** `panel--glm53.brandonmusic.final25-clean17` -- brandonmusic panel v1, calibration-clean subset -- 17 of 25 final windows
  17 contexts x 2047 scored positions = **34,799 scored positions**, score_from 0
  sealed: **yes** (token digest `ecfce5997ab9106c...`) -- contamination scan: **yes, 8 hits**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `cross_stack`, head_policy `native_head`
**Comparability key** `cmp--eee09298c558ab21`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--4a8630bdcadab97f` (2 rows): `panel_id` panel--glm53.brandonmusic.final25-clean17 -> panel--glm53.brandonmusic.final25; `reference_id` reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17 -> reference--brandonmusic.glm53-bf16-fp32-logits.final25
> - `cmp--2b9c401d13806d7e` (4 rows): `metric_name` mean_tokenwise_kld -> mean_of_run_means_tokenwise_kld; `stack_relation` cross_stack -> same_stack
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| **GLM-5.3-Flash BF16 @a6c167b6** _(measurement floor)_ | `bf16` | -- | **0.0106476** | [0.00812433, 0.0134906] | -- | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/crosscheck-brandonmusic.json) |
| GLM-5.3-Flash official FP8 | `fp8_e4m3 @8` | 328.4 GB | **0.0186653** | [0.0141924, 0.0247936] | -- | 1 run, unevidenced | measured by us (their artifact) | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/fp8-on-brandon-panel.json) |

> **Bias on GLM-5.3-Flash BF16 @a6c167b6** -- cross_stack_capture_replay, direction upward. THIS ROW IS THE FLOOR. It replays the reference's own BF16 weights through our vLLM stack and scores them against brandonmusic's stored fp32 teacher logits. 0.012712 nats is therefore what two stacks disagree by on identical unquantized weights -- not a quantization result. No floor is named because none exists below it.

> **Bias on GLM-5.3-Flash official FP8** -- cross_stack_capture_replay, direction upward. Teacher captured on brandonmusic's transformers/eager stack, candidate replayed on our vLLM stack. The same-stack BF16 replay floor on this exact panel is 0.012712, so this number is an UPPER BOUND on the FP8 release's own divergence. The naive difference is 0.007904 -- an estimate, not an identity, because KL is not additive. Do not subtract and publish. Scope-matched: this row's floor reference is the clean17 floor, not the panel25 one. Subtracting a floor measured on a different WINDOW SET is the same class of error as subtracting one measured on a different LANE, and this registry refuses both.

> **The same artifact, measured elsewhere in this file.** 2 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 51%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **GLM-5.3-Flash BF16 @a6c167b6** -- 3 values here, from **0.0106476** to **0.0127116** nats (19% apart). Other tables: `cmp--202b717f3219c414`, `cmp--4a8630bdcadab97f`.
> - **GLM-5.3-Flash official FP8** -- 4 values here, from **0.0186653** to **0.0281039** nats (51% apart). Other tables: `cmp--4a8630bdcadab97f`, `cmp--9b009314102d9e8b`, `cmp--e6cdd07242bdde05`.

<details><summary>Disclosures for the rows above (8)</summary>

- `glm53.bf16-replay-floor.brandonmusic-final25.clean17` **cross_stack_capture**: Teacher captured on transformers/eager (B200 x4); candidate replayed on our vLLM stack. The offset audit confirms position alignment: top-1 agreement is 0.9665 at offset 0 and 0.0159 / 0.0162 at offsets -1 / +1.
- `glm53.bf16-replay-floor.brandonmusic-final25.clean17` **single_run**: One pass; determinism not established.
- `glm53.bf16-replay-floor.brandonmusic-final25.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.bf16-replay-floor.brandonmusic-final25.clean17` note: Calibration-clean scope recompute of measurement--glm53.bf16-replay-floor.brandonmusic-final25. panel25 0.012711599817251 -> clean17 0.010647639361035 (-0.002063960456216, -16.24%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.
- `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` **cross_stack_capture**: This row cannot be ranked against the K6 / Dione / 4bpw rows on the same panel: those are same-stack sealed-capture numbers and this is a cross-stack replay. Their comparability keys differ, and the registry's tables are grouped by that key.
- `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` **single_run**: One pass; determinism not established.
- `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` **subset_of_panel**: 17 of the panel's 25 sealed windows (34,799 of 51,175 scored positions). The excluded 8 are the windows the calibration-overlap scan flags; see measurement_scope.calibration_overlap_scan.
- `glm53.official-fp8.brandonmusic-final25.crossstack.clean17` note: Calibration-clean scope recompute of measurement--glm53.official-fp8.brandonmusic-final25.crossstack. panel25 0.020615254540418 -> clean17 0.018665326569455 (-0.001949927970963, -9.46%). Not a correction and not a supersession: the two scopes answer different questions and move different contributors' rows in opposite directions. Never compare a clean17 value against a panel25 value.

</details>

### Panel: brandonmusic panel v1, single window final-0000

Derived from `panel--glm53.brandonmusic.final25` by **shard_subset**: window final-0000 alone, 1/25 of the parent panel. 2,047 scored positions instead of 51,175. brandonmusic's runtime receipts score this window only. The same artifact reads 0.022751 here and 0.024555 over the full 25 windows, a 7% swing -- which is why this is a separate panel record.

> **Panel disclosure -- `weak_contamination_guard`:** This panel's only contamination guard is ROLE SEPARATION: the 25 'final' windows are drawn from the same packed corpus as the 384 fit / 128 conditional-fit / 64 selection / 64 confirmation windows and are declared qualification-only. No lexical or n-gram scan is published, and the underlying document provenance is published only as a digest. This is materially weaker than the malaiwah v5 suites, which run a 12-word shingle whole-document pre-exclusion and report 0 hits. Do not describe the two guards as equivalent. It applies equally to every row on this panel, so it does not disturb comparisons WITHIN the panel.

> **Panel disclosure -- `subset_of_panel`:** A single 2,047-position window. Numbers on this panel have far wider sampling error than the 25-window panel and must never be tabled beside it.

This panel carries **2 separate comparability groups**. They are different measurements of different things and are never merged.

#### Group `cmp--b55c2d693d127f20` -- 6 rows

**Panel** `panel--glm53.brandonmusic.final-0000` -- brandonmusic panel v1, single window final-0000
  1 contexts x 2047 scored positions = **2,047 scored positions**, score_from 0
  sealed: **yes** (token digest `338027e62f41540f...`) -- contamination scan: **NOT RUN**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final-0000` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_of_run_means_tokenwise_kld, direction reference_to_candidate, accumulation unknown
**Estimation surface** stack_relation `same_stack`, head_policy `native_head`
**Comparability key** `cmp--b55c2d693d127f20`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--18990ab191ea7a67` (2 rows): `metric_name` mean_of_run_means_tokenwise_kld -> mean_tokenwise_kld; `accumulation_dtype` unknown -> float64
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_of_run_means_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| GLM-5.3-Flash official FP8 weights served with FP8 MLA KV | `fp8_e4m3 @8` | -- | **0.0245817** | -- | 93.63 % | 5 runs, sd 0.00016 | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v71/kld/fp8-dcp2-route128-five-run-kld.json) |
| GLM-5.3-Flash official FP8 weights served with FP8 MLA KV | `fp8_e4m3 @8` | -- | **0.0246106** | -- | 93.73 % | 5 runs, sd 0.000257 | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v75/kld/fp8-five-run-kld.json) |
| GLM-5.3-Flash official FP8 weights served with FP8 MLA KV | `fp8_e4m3 @8` | -- | **0.0246286** | -- | 93.80 % | 5 runs, sd 0.000326 | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/fp8-five-run-kld-receipt.json) |
| brandonmusic GLM-5.3-Flash NVFP4 runtime build | `nvfp4 @4` | -- | **0.0547574** | -- | 91.50 % | 5 runs, bitwise identical | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v71/kld/nvfp4-dcp2-route128-power2-five-run-kld.json) |
| brandonmusic GLM-5.3-Flash NVFP4 runtime build | `nvfp4 @4` | -- | **0.0547574** | -- | 91.50 % | 5 runs, bitwise identical | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v75/kld/nvfp4-five-run-kld.json) |
| brandonmusic GLM-5.3-Flash NVFP4 runtime build | `nvfp4 @4` | -- | **0.0605349** | -- | 91.55 % | 5 runs, bitwise identical | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/nvfp4-five-run-kld-receipt.json) |

> **The same artifact, measured elsewhere in this file.** One of the artifacts below also carries a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 25%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **brandonmusic GLM-5.3-Flash NVFP4 runtime build** -- 6 values here, from **0.0547574** to **0.0682296** nats (25% apart). Other tables: `cmp--18990ab191ea7a67`.

<details><summary>Disclosures for the rows above (13)</summary>

- `glm53.official-fp8.v71.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: FP8 MLA NoPE, route128 SMEM, TP2/EP2, DCP2 B12X A2A eager no-MTP. We have not re-run it.
- `glm53.official-fp8.v71.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.official-fp8.v75.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: v75 release image, FP8 MLA NoPE, route128 SMEM/register, TP2/EP2, DCP2 direct symmetric-memory A2A. We have not re-run it.
- `glm53.official-fp8.v75.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.official-fp8.v44.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: v43 TP2 DCP1 eager no-MTP FP8 MLA KV, GPUs 2,3. We have not re-run it.
- `glm53.official-fp8.v44.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.nvfp4.v71.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: NVFP4 MLA NoPE, power-of-two ceil amax scale v2, route128 SMEM, TP2/EP2, DCP2 B12X A2A eager no-MTP. We have not re-run it.
- `glm53.nvfp4.v71.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.nvfp4.v75.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: v75 release image, NVFP4 MLA NoPE calibrated power-of-two 46-layer scales. We have not re-run it.
- `glm53.nvfp4.v75.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.nvfp4.v44.brandonmusic-final-0000` **author_reported_only**: Measured and published by brandonmusic on his own runtime image. Regime as published: v44 TP2 DCP1 eager no-MTP NVFP4 MLA KV, GPUs 2,3. We have not re-run it.
- `glm53.nvfp4.v44.brandonmusic-final-0000` **estimator_unknown**: This receipt family (glm53-r19-runtime-kld-repeated.v1) publishes no compute_dtype, so the accumulation precision of brandonmusic's scorer is not established for these rows and is recorded as unknown. All six rows in this group share that condition, so they remain mutually comparable; a row whose receipt attests float64 would not join them. His other two GLM-5.3-Flash receipts do declare float64, which makes it likely but not evidenced here.
- `glm53.nvfp4.v44.brandonmusic-final-0000` **quality_gate_failed**: The author's own gate (mean tokenwise KLD < 0.06) did NOT pass. Recorded because a failing gate is a fact about the artifact, not a reason to hide the row.

</details>

#### Group `cmp--18990ab191ea7a67` -- 2 rows

**Panel** `panel--glm53.brandonmusic.final-0000` -- brandonmusic panel v1, single window final-0000
  1 contexts x 2047 scored positions = **2,047 scored positions**, score_from 0
  sealed: **yes** (token digest `338027e62f41540f...`) -- contamination scan: **NOT RUN**
**Reference (teacher)** `reference--brandonmusic.glm53-bf16-fp32-logits.final-0000` -- native_bf16, artifact `artifact--zai-org.glm-5.3-flash-bf16.a6c167b6` @a6c167b62691b2bac901344b65cb651a70f53e43
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation float64
**Estimation surface** stack_relation `same_stack`, head_policy `native_head`
**Comparability key** `cmp--18990ab191ea7a67`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** The nearest neighbouring groups differ in:
> - `cmp--b55c2d693d127f20` (6 rows): `metric_name` mean_tokenwise_kld -> mean_of_run_means_tokenwise_kld; `accumulation_dtype` float64 -> unknown
> 
> Those numbers are in this file, under their own headings. Quoting one under the other heading is the mistake this layout exists to prevent: the key is a function of the panel, the teacher, the metric, the direction and the estimator, and the validator recomputes it from those fields rather than trusting the stamped value. What that catches is a row filed under a key its own fields do not produce. It does not catch a number attributed to the wrong panel in the first place -- no offline checker can. That is what the receipt digests on every row are for.
>
> Also, and always: **every table for a different model.** A KL number is a divergence over one model's own vocabulary against that model's own teacher. It is not a quality score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| brandonmusic GLM-5.3-Flash tr3 4bpw | `exl3-mcg @4` | 175.6 GB | **0.0227508** | -- | 93.84 % | 1 run, unevidenced | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/results/tp2-runtime-window-kld.json) |
| brandonmusic GLM-5.3-Flash NVFP4 runtime build | `nvfp4 @4` | -- | **0.0682296** | -- | 91.99 % | 1 run, unevidenced | reported by brandonmusic | [receipt](https://raw.githubusercontent.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/main/runtime-results/v44/kld/nvfp4-dynamic-scale-control-kld-report.json) |

> **The same artifact, measured elsewhere in this file.** 2 of the artifacts below also carry a number in another table -- on a different panel, teacher or estimator -- and the widest of those spans 25%. None of the readings is wrong and none is interchangeable with another. Quoting one of them as *the* number for the artifact, without its table, is the misuse this registry exists to make obvious.
>
> - **brandonmusic GLM-5.3-Flash NVFP4 runtime build** -- 4 values here, from **0.0547574** to **0.0682296** nats (25% apart). Other tables: `cmp--b55c2d693d127f20`.
> - **brandonmusic GLM-5.3-Flash tr3 4bpw** -- 3 values here, from **0.0227508** to **0.0249488** nats (10% apart). Other tables: `cmp--202b717f3219c414`, `cmp--2b9c401d13806d7e`.

<details><summary>Disclosures for the rows above (6)</summary>

- `glm53.brandonmusic-4bpw.tp2-runtime.brandonmusic-final-0000` **author_reported_only**: brandonmusic's custom TP2 runtime on the single qualification window. The receipt notes runtime_raw_decoded_parity_passed false with runtime_rank_output_identical true.
- `glm53.brandonmusic-4bpw.tp2-runtime.brandonmusic-final-0000` **single_run**: One run.
- `glm53.brandonmusic-4bpw.tp2-runtime.brandonmusic-final-0000` note: THE PANEL-SCOPE OBJECT LESSON: the same artifact reads 0.022751 here and 0.024555 over the full 25 windows, against the same teacher. A 7% swing from window selection alone.
- `glm53.nvfp4-dynamic-scale-control.brandonmusic-final-0000` **author_reported_only**: brandonmusic's dynamic-scale CONTROL for the v44 NVFP4 row: same window, same teacher, dynamic instead of calibrated power-of-two scales.
- `glm53.nvfp4-dynamic-scale-control.brandonmusic-final-0000` **single_run**: One run.
- `glm53.nvfp4-dynamic-scale-control.brandonmusic-final-0000` **quality_gate_failed**: mean_kld_gate_passed false at threshold 0.06.

</details>

### Panel: orcarouter MLX evaluation set (undisclosed)

> **Panel disclosure -- `undisclosed_panel`:** Neither the token set, the window count nor the scored-position total is published. Numbers on this panel can be reported but cannot be compared with anything measured on a known panel -- including other rows for the same model.

#### Group `cmp--492e9b16e8bd6fbd` -- 5 rows

**Panel** `panel--orcarouter.undisclosed` -- orcarouter MLX evaluation set (undisclosed)
  -- contexts x -- scored positions = **undisclosed scored positions**, score_from None
  sealed: **no** -- contamination scan: **NOT RUN**
**Reference (teacher)** `reference--orcarouter.glm53-fp8-dequantized.undisclosed` -- dequantized_from_quant, artifact `artifact--orcarouter.glm-5.3-flash-fp8-dequantized` @unpinned revision
**Metric** mean_tokenwise_kld, direction reference_to_candidate, accumulation unknown
**Estimation surface** stack_relation `same_stack`, head_policy `unknown`
**Comparability key** `cmp--492e9b16e8bd6fbd`

> **What this table is.** Every row here shares the comparability key above: the same tokens, the same teacher capture, the same metric and direction, the same estimator precision, the same stack relation and the same head policy. Ranking them against each other is the one thing this registry says you may do.
>
> **Rank is not a verdict.** The table is sorted by fidelity alone, and fidelity buys bits: a larger, higher-bitrate quant will usually sit above a smaller one, which is not news. Read the Size and Codec columns before reading the order, and compare like against like.
>
> **What it is NOT comparable to.** Every other table in this file: no other group shares this key. That includes every table for a different model -- a KL number is a divergence over one model's own vocabulary against that model's own teacher, never a score that can be carried between models.

| Artifact | Codec | Size | mean_tokenwise_kld (nats) | CI95 | Top-1 | Runs | Attribution | Receipt |
|---|---|---:|---:|---|---:|---|---|---|
| orcarouter GLM-5.3-Flash-MLX 6-bit | `mlx-affine @6` | 295.6 GB | **0.0063** | -- | 97.76 % | 1 run, unevidenced | reported by orcarouter | model_card |
| orcarouter GLM-5.3-Flash-MLX 4-bit | `mlx-affine @4` | 204.0 GB | **0.0131** | -- | 96.13 % | 1 run, unevidenced | reported by orcarouter | model_card |
| orcarouter GLM-5.3-Flash-MLX 3-bit | `mlx-affine @3` | 184.3 GB | **0.0421** | -- | 92.06 % | 1 run, unevidenced | reported by orcarouter | model_card |
| orcarouter GLM-5.3-Flash-MLX 2-bit | `mlx-affine @2` | 145.0 GB | **0.1647** | -- | 86.56 % | 1 run, unevidenced | reported by orcarouter | model_card |
| orcarouter GLM-5.3-Flash-MLX 2bit-lite | `mlx-affine @2` | 102.5 GB | **0.3456** | -- | 77.19 % | 1 run, unevidenced | reported by orcarouter | model_card |

<details><summary>Disclosures for the rows above (30)</summary>

- `glm53.orcarouter-mlx-6bit.undisclosed` **author_reported_only**: Reported by orcarouter on their model card. No receipt, no estimator precision, no run count.
- `glm53.orcarouter-mlx-6bit.undisclosed` **different_reference_kind**: Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 teacher. Numbers against a quantized reference are systematically smaller. This row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's panel -- they are not the same quantity.
- `glm53.orcarouter-mlx-6bit.undisclosed` **undisclosed_panel**: Evaluation set not disclosed: no token digest, window count or position total.
- `glm53.orcarouter-mlx-6bit.undisclosed` **subset_of_panel**: Panel coverage unknown, so covers_full_panel is false by default.
- `glm53.orcarouter-mlx-6bit.undisclosed` **estimator_unknown**: Accumulation precision and head policy are not published.
- `glm53.orcarouter-mlx-6bit.undisclosed` note: Perplexity reported alongside on the same card: 2.7864 (FP8 reference 2.7797).
- `glm53.orcarouter-mlx-4bit.undisclosed` **author_reported_only**: Reported by orcarouter on their model card. No receipt, no estimator precision, no run count.
- `glm53.orcarouter-mlx-4bit.undisclosed` **different_reference_kind**: Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 teacher. Numbers against a quantized reference are systematically smaller. This row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's panel -- they are not the same quantity.
- `glm53.orcarouter-mlx-4bit.undisclosed` **undisclosed_panel**: Evaluation set not disclosed: no token digest, window count or position total.
- `glm53.orcarouter-mlx-4bit.undisclosed` **subset_of_panel**: Panel coverage unknown, so covers_full_panel is false by default.
- `glm53.orcarouter-mlx-4bit.undisclosed` **estimator_unknown**: Accumulation precision and head policy are not published.
- `glm53.orcarouter-mlx-4bit.undisclosed` note: Perplexity reported alongside on the same card: 2.862 (FP8 reference 2.7797).
- `glm53.orcarouter-mlx-3bit.undisclosed` **author_reported_only**: Reported by orcarouter on their model card. No receipt, no estimator precision, no run count.
- `glm53.orcarouter-mlx-3bit.undisclosed` **different_reference_kind**: Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 teacher. Numbers against a quantized reference are systematically smaller. This row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's panel -- they are not the same quantity.
- `glm53.orcarouter-mlx-3bit.undisclosed` **undisclosed_panel**: Evaluation set not disclosed: no token digest, window count or position total.
- `glm53.orcarouter-mlx-3bit.undisclosed` **subset_of_panel**: Panel coverage unknown, so covers_full_panel is false by default.
- `glm53.orcarouter-mlx-3bit.undisclosed` **estimator_unknown**: Accumulation precision and head policy are not published.
- `glm53.orcarouter-mlx-3bit.undisclosed` note: Perplexity reported alongside on the same card: 3.0566 (FP8 reference 2.7797).
- `glm53.orcarouter-mlx-2bit.undisclosed` **author_reported_only**: Reported by orcarouter on their model card. No receipt, no estimator precision, no run count.
- `glm53.orcarouter-mlx-2bit.undisclosed` **different_reference_kind**: Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 teacher. Numbers against a quantized reference are systematically smaller. This row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's panel -- they are not the same quantity.
- `glm53.orcarouter-mlx-2bit.undisclosed` **undisclosed_panel**: Evaluation set not disclosed: no token digest, window count or position total.
- `glm53.orcarouter-mlx-2bit.undisclosed` **subset_of_panel**: Panel coverage unknown, so covers_full_panel is false by default.
- `glm53.orcarouter-mlx-2bit.undisclosed` **estimator_unknown**: Accumulation precision and head policy are not published.
- `glm53.orcarouter-mlx-2bit.undisclosed` note: Perplexity reported alongside on the same card: 4.3622 (FP8 reference 2.7797).
- `glm53.orcarouter-mlx-2bitlite.undisclosed` **author_reported_only**: Reported by orcarouter on their model card. No receipt, no estimator precision, no run count.
- `glm53.orcarouter-mlx-2bitlite.undisclosed` **different_reference_kind**: Measured against the official FP8 release DEQUANTIZED TO BF16, not against a BF16 teacher. Numbers against a quantized reference are systematically smaller. This row's 6-bit 0.0063 is NOT better than the K6 6bpw 0.013723 on brandonmusic's panel -- they are not the same quantity.
- `glm53.orcarouter-mlx-2bitlite.undisclosed` **undisclosed_panel**: Evaluation set not disclosed: no token digest, window count or position total.
- `glm53.orcarouter-mlx-2bitlite.undisclosed` **subset_of_panel**: Panel coverage unknown, so covers_full_panel is false by default.
- `glm53.orcarouter-mlx-2bitlite.undisclosed` **estimator_unknown**: Accumulation precision and head policy are not published.
- `glm53.orcarouter-mlx-2bitlite.undisclosed` note: Perplexity reported alongside on the same card: 6.7018 (FP8 reference 2.7797).

</details>


<!-- END GENERATED: tables -->

---

## Using the data

```
data/models.jsonl        the upstream models
data/artifacts.jsonl     one concrete weight set at one pinned revision + a STRUCTURED quantization scope
data/panels.jsonl        the token sets, including scored-position policy, sealing and contamination guard
data/references.jsonl    teacher captures: (artifact, panel, stack, precision, head source)
data/pipelines.jsonl     the measuring and producing stacks
data/measurements.jsonl  the rows
index.json               counts, collection digests, and the comparability-key groups as DATA
schema/*.schema.json     JSON Schema draft 2020-12
schema/invariants.json   the machine-readable rules the validator enforces, with severities
```

Resolver rule: every `*_ref` is the `id` of a record in the collection named by the ref's id prefix
(`model--`, `artifact--`, `panel--`, `reference--`, `pipeline--`, `measurement--`). That is the only
thing a consumer needs to know to join the files.

Query it with one line of `jq` -- the mission's original complaint, answered:

```bash
# every measured quant of GLM-5.3-Flash with its number, panel and who measured it
jq -r 'select(.model_ref=="model--zai-org.glm-5.3-flash")
       | [.metric.value, .artifact_ref, .panel_ref, .provenance.measured_by, .comparability.key]
       | @tsv' data/measurements.jsonl | sort -n

# only rows you may legitimately rank against our K6
jq -r --arg k cmp--202b717f3219c414 'select(.comparability.key==$k)
       | [.metric.value, .artifact_ref] | @tsv' data/measurements.jsonl | sort -n
```

## Tools

```bash
python3 tools/registry_validate.py                  # schema + every invariant, offline, no installs
python3 tools/registry_validate.py --strict --json  # CI mode
python3 tools/registry_validate.py --explain <id> [--against <id>]
python3 tools/registry_render.py [--check]          # regenerate / verify README tables + index.json
python3 tools/registry_add.py from-receipt --receipt R --artifact A --panel P ...
python3 tools/seed_registry.py --check              # the seeded rows are regenerable (see the note below)
make check                                          # validate + render --check + fixtures
```

**What `seed_registry.py --check` does and does not prove.** The 37 Qwen3.8-27B rows are read
live out of their receipt files on every run — the seeder refuses to build if a receipt is
missing — so `--check` genuinely re-derives those values from receipts and byte-compares them.
The 20 GLM-5.3-Flash rows are transcribed literals: their receipts live on the Hub and in
third-party repositories, and this tooling is offline by contract, so for those rows `--check`
proves that `data/` matches `seed_registry.py`, not that `seed_registry.py` matches the receipt.
Each of those rows records its receipt's `sha256`, so the binding is checkable by hand: fetch the
`uri`, hash it, and compare the value at the `field_provenance` pointer. All 20 were checked that
way on 2026-08-28 and all 20 matched at full float64. Nothing in CI rechecks it, because nothing
in CI is allowed to reach the network.

Both tools run on a stock interpreter with **no network and no pip**: `tools/_minischema.py` is a
vendored draft-2020-12 validator covering exactly the keyword subset these schemas use, and it raises
on any keyword it does not implement rather than silently ignoring it. When the real `jsonschema`
library is importable, `--jsonschema-lib both` runs both and the CI job fails if their verdicts differ,
so the vendored one cannot quietly drift.

## Credit

The artifacts and the numbers in this registry mostly belong to other people. Specifically:

- **brandonmusic** built the sealed GLM-5.3-Flash token panel, captured and published the fp32 BF16
  teacher logits that four of our own numbers are measured against, produced the tr3-4bpw checkpoint,
  and measured and published the 4bpw and runtime-image rows on his own stack. The panel and the
  teacher are his work; we are guests on them.
- **0xSero** produced the GLM-5.3-Flash EXL3 Q4 (Dione) release. The Q4 number here is ours, the
  artifact is theirs. `local-ai-registry` is also theirs, and this registry is shaped to interoperate
  with it.
- **orcarouter (Continuum AI Corp)** produced the GLM-5.3-Flash MLX builds and reported their own
  fidelity numbers, which are included here as their measurements against their reference, quarantined
  from ours rather than merged into them.
- **turboderp** wrote exllamav3, without which most of the EXL3 artifacts in this registry would not
  exist, and published the Qwen3.8-27B exl3 branches measured here.
- **Z.ai (zai-org)** published GLM-5.3-Flash and its official FP8 release. **Qwen (Alibaba)** published
  Qwen3.8-27B and its FP8 release. **unsloth**, **gittensor-model-hub** and the authors of the
  AWQ-INT4 and MTP-NVFP4 builds produced artifacts we measured.

Where an upstream author's identity could not be established from a receipt, this registry records
`repository: null` and says so, rather than asserting a repo id it cannot back up.
