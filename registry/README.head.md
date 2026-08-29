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
