# HF card fidelity-provenance annotation — `fidelity-provenance/v1`

**Status:** v1, frozen for implementation.
**Companion:** [`FIDELITY-DATASET-SPEC.md`](FIDELITY-DATASET-SPEC.md).
**Schema:** [`schema/fidelity-card-annotation.schema.json`](schema/fidelity-card-annotation.schema.json).
**Worked cards:** [`examples/card-k6.yaml`](examples/card-k6.yaml),
[`examples/card-k8.yaml`](examples/card-k8.yaml),
[`examples/card-root-bf16.yaml`](examples/card-root-bf16.yaml),
[`examples/card-dataset-suite-v1.yaml`](examples/card-dataset-suite-v1.yaml).

---

## 1. The problem, and what HF already gives us

A model card should be able to say, machine-readably, three things:

1. **this model's fidelity was measured** — against which panel, which reference, on which lane, with
   what estimator, and where the registry row is;
2. **this model's fidelity dataset lives here** — REQUIRED for a root, OPTIONAL for a quant;
3. **this model's head identity** — without which nobody may replay its hidden states through
   anyone else's head (the head trap, `FIDELITY-DATASET-SPEC.md` §8).

Nobody in this space uses HF's structured card fields at all today. Verified:

| card | frontmatter it actually carries |
|---|---|
| `festr2/kimi-k3-distribution-fidelity-1024x2048-v1` | `pretty_name`, `license`, `task_categories`, `language` |
| `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` | `pretty_name`, `license`, `task_categories`, `tags` |
| `malaiwah/GLM-5.3-Flash-fidelity-suite-v1` | `license: mit` + 5 tags |
| `malaiwah/GLM-5.3-Flash-TR3-6bpw` (K6) | `license`, `base_model`, `base_model_relation`, `pipeline_tag`, `library_name`, 14 tags |

**No `model-index`, no `datasets:`, no eval results anywhere.** Every fidelity number in our own
cards lives in prose tables. `model-index` and `base_model_relation` are standardized, parsed and
completely unused here. Extending them is unopposed.

### 1.1 What the Hub validates (measured, not assumed)

The push-time gate is `POST https://huggingface.co/api/validate-yaml` with
`{"content": "<full README including frontmatter>", "repoType": "model"|"dataset"}` — the same
endpoint `huggingface_hub.repocard.metadata_validate` calls.

| field | behaviour | evidence |
|---|---|---|
| `license` | hard enum, HTTP 400 | rejected with the full enum list |
| `language` | hard enum (ISO), HTTP 400 | `not_a_lang_code` → 400 |
| `base_model_relation` | **hard enum, exactly 4 values**: `adapter`, `merge`, `quantized`, `finetune` | 400 on anything else |
| `size_categories[]` | type-checked, 400 on int | |
| `pipeline_tag`, `task_categories[]` | **warning only** | unknown → warning listing the 57 official tags |
| `library_name` | not validated | `exllamav3` accepted silently |
| `metrics[]` (top level) | **not validated**, free strings | `kl_divergence` accepted |
| `model-index[0].name` | **required**, 400 | |
| `model-index[].results[].metrics[].value` | **required**, 400 | |
| `model-index[].results[].source.url` | **required if `source:` present**, 400 | |
| `model-index[].results[].task.type` | **not validated**, free string | `distribution-fidelity` passes |
| unknown top-level keys | accepted, 0 errors, **and preserved by the Hub API** | `TheBloke/Llama-2-7B-GGUF` re-serves `model_creator`, `prompt_template`, `quantized_by` in `cardData` |

**YAML gotcha:** an all-digit 40-char revision parses as an integer and is rejected with
`"…dataset.revision" must be a string`. **Quote every digest and revision.**

### 1.2 Two land mines in `huggingface_hub`'s parser (both reproduced in-process)

1. **Only ONE `model-index` entry is safe.** `model_index_to_eval_results` flattens all entries into
   one list and keeps the **last** entry's `name`. A 2-entry index named A and B round-trips to a
   single entry named **B** carrying both results — silent data corruption.
   **Rule: exactly one `model-index` entry.**
2. **Results merge on a 5-tuple that excludes `args`.** The merge key is
   `(task_type, dataset_type, dataset_config, dataset_split, dataset_revision)`. Two results
   identical in that tuple but differing in `dataset.args` merge, and the **second result's `args`
   are silently discarded** while both metric values are re-attributed to the first dataset block.
   Reproduced:
   ```
   in : result(config=c1, args={lane: checkpoint}) ; result(config=c1, args={lane: serving})
   out: result(config=c1, args={lane: checkpoint}, metrics=[both])   # lane:serving GONE
   ```
   This is exactly the lane mixing the registry forbids (**BIAS-006**).
   **Rule: any discriminator that must survive — above all LANE — lives in `dataset.config`,
   `dataset.split` or `dataset.revision`, never only in `args`.**

Unknown keys nested *inside* a `model-index` result (`x_fidelity:` beside `task:`) or inside
`dataset:` pass the Hub validator but are **silently dropped** by `EvalResult`, which is a fixed
dataclass. Do not use them. `dataset.args` and `metrics[].args` are spec'd free-form maps, survive
the Hub **and** survive a library round-trip (`HuggingFaceH4/zephyr-7b-beta` ships
`dataset.args: {num_few_shot: 25}` in production), and are the correct extension point.

---

## 2. The annotation, in two layers

**Layer 1** expresses the measurement as a fully conformant `model-index` result, so HF leaderboards
and every existing card reader see it.
**Layer 2** is one small, namespaced, additive top-level block, `x_fidelity:`, for what `model-index`
structurally cannot express.

Layer 1 alone loses head identity, determinism evidence, lane bridges and the dataset pointer.
Layer 2 alone is invisible to every HF tool. Both, cross-checked against each other, is the design.

### 2.1 Layer 1 — the `model-index` mapping

| registry concept | `model-index` slot | why there |
|---|---|---|
| scoring workload | `task.type: text-generation` | a real pipeline tag keeps the widget coherent; the panel is next-token prediction over text |
| — | `task.name` | `"Distribution fidelity (KL divergence vs BF16 reference)"` |
| panel repo | `dataset.type` | the Hub dataset id holding the panel/reference capture |
| panel pretty name | `dataset.name` | required |
| panel id | `dataset.config` | part of the merge key |
| **lane** | **`dataset.split`** | **must be in the merge key or §1.2.2 silently merges lanes** |
| panel revision | `dataset.revision` (quoted) | pins the panel bytes |
| panel identity detail | `dataset.args` | `panel_id`, `panel_token_sha256`, `contexts`, `scored_positions`, `context_length`, `tokenizer@rev`, `vocab_size`, `reference_id` |
| metric | `metrics[].type: kl_divergence`, `.value` | full float64 precision, never rounded |
| estimator / determinism / registry ids | `metrics[].args` | `units`, `higher_is_better`, `direction`, `estimator`, `accumulation_dtype`, `logits_dtype`, `head_policy`, `stack_relation`, `lane`, `run_count`, `population_stddev_of_run_means`, `determinism`, `measurement_id`, `comparability_key` |
| registry row | `source: {name, url}` | `source.url` required when `source:` present; the dataset-viewer search URL is a real deep link |
| floor-subtracted number | a **second** metric `kl_divergence_quantization_attributable`, **same-lane result only** | carries `floor_measurement_id`, `floor_lane` and the non-additivity caveat; enforces **BIAS-006** structurally |
| top-1 | `top1_agreement` metric, `higher_is_better: true` | registry **STAT-005** wants it on every published KL row |

Plus, top level: `base_model` + `base_model_relation: quantized`; `datasets:` listing the panel repo
and the fidelity dataset; `metrics: [kl_divergence, top1_agreement]`; tag `fidelity-provenance`.

**Known trade-off, disclosed:** `datasets:` is the **only** field that produces `dataset:<id>` Hub
tags and the bidirectional model↔dataset link — proven by counter-example, since
`jonatasgrosman/wav2vec2-large-xlsr-53-english` names a dataset in `model-index[].dataset.type` that
does **not** appear in its Hub tags. But the Hub renders `datasets:` under the label *"Datasets used
to train:"*, which is semantically wrong for an evaluation panel. There is no eval-dataset
equivalent field. Discoverability wins; the spec discloses it.

`verified` / `verifyToken` are HF-controlled (their own eval service). Our self-measured rows render
unverified. **Never fake them.**

### 2.2 Layer 2 — `x_fidelity:`

One top-level key. `x_` is the classic additive-extension convention (HTTP `X-`, OpenAPI `x-`):
collision-safe and obviously non-HF. If HF ever standardizes a `fidelity:` block — the
`co2_eq_emissions` path, which is precisely a nested domain-specific top-level block HF standardized
after community usage — the validator accepts both and the generator emits the standard name.

```yaml
x_fidelity:
  spec: https://github.com/malaiwah/glm53-flash-fidelity-suite/blob/main/docs/FIDELITY-DATASET-SPEC.md
  spec_version: fidelity-provenance/v1
  role: quant                       # root | quant | fidelity-dataset
  reference_model: zai-org/GLM-5.3-Flash-BF16
  reference_revision: 'a6c167b62691b2bac901344b65cb651a70f53e43'
  fidelity_dataset: null            # or {repository, revision, dataset_sha256, capture_content_digest, form, role}
  registry:
    dataset: malaiwah/quant-fidelity-registry
    schema_version: quant-fidelity-registry/v1
    artifact_id: artifact--malaiwah.glm-5.3-flash-tr3-6bpw
    measurement_ids: [ ... ]
  scope_digest: 'attn.o=quantized:exl3-mcg@6|…|head=native|kv=bf16'
  head:
    policy: native                  # native | quantized | shared_reference | unknown
    quantized: false
    bits: 16
    lm_head_tensor_content_sha256: 'aa21c427…'   # NORMATIVE identity
    lm_head_file_sha256: '47eaf729…'             # container digest, never the identity
    final_norm_tensor_content_sha256: null
    final_norm_file_sha256: 'c228a123…'
    equality_receipt: 'https://…/reports/head-equality-fp8.json'
    replay_permitted: true          # false when the content digest is null
  measurements:
    - id: measurement--glm53.k6-6bpw.brandonmusic-final25
      lane: sealed-ep8
      status: published
      comparability_key: cmp--202b717f3219c414
      panel_id: panel--glm53.brandonmusic.final25
      reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25
      pipeline_id: pipeline--malaiwah.glm53-packed-kld
      metric_name: mean_of_run_means_tokenwise_kld
      value: 0.013723384665701147
      units: nats
      direction: reference_to_candidate
      run_count: 5
      determinism:
        evidence_kind: tokenwise_kld_sha256
        distinct_evidence_hash_count: 1
        identical_across_runs: true
        evidence_hashes: ['52e35723…']
      floor_measurement_id: null
      quantization_attributable: null
      measured_by: self-measured
      disclosures: [no_known_deviations]
```

### 2.3 The cross-check invariant (what stops the two layers drifting)

**XC-1** Every `x_fidelity.measurements[].value` equals the `model-index` metric value whose
`args.measurement_id` matches.
**XC-2** That measurement's `lane` equals that result's `dataset.split`.
**XC-3** Every `measurement_id` in Layer 2 appears in exactly one Layer-1 result, and vice versa —
one `model-index` result per registry measurement, no more. (K8 has **no** `sealed-ep8` row in the
registry, so its card MUST NOT carry a `split: sealed-ep8` result.)
**XC-4** `floor_lane == lane` on any `quantization_attributable` value — BIAS-006 at card level.
**XC-5** `head.replay_permitted: true` requires `lm_head_tensor_content_sha256` non-null. A null
content digest **forbids** cross-artifact hidden replay (spec HEAD-4).

---

## 3. Required / optional matrix

| field | root model | quant model | fidelity dataset repo |
|---|---|---|---|
| `x_fidelity.spec` / `spec_version` / `role` | **R** | **R** | **R** |
| `x_fidelity.fidelity_dataset` | **R** (non-null) | O (`null` is the common case) | n/a — it *is* the dataset |
| `x_fidelity.registry.measurement_ids` | O | **R**, non-empty | O |
| `x_fidelity.head.*` content digest | **R** | **R** | **R** |
| `x_fidelity.scope_digest` | **R** (all-native) | **R** | **R** |
| `x_fidelity.measurements[]` | O (self-compare only) | **R**, ≥ 1 | O |
| `x_fidelity.captured_model` | n/a | n/a | **R** (`{repository, revision, role}`) |
| `x_fidelity.form` / `panel` / `lane` / `seal` | n/a | n/a | **R** |
| `model-index` | O — if present it is the self-compare reproduction (`value: 0.0`, `args.comparison_kind: self_compare`, `args.exact_zero_asserted: true`) | **R**, ≥ 1 result | **n/a** (dataset cards have no `model-index`) |
| `base_model` + `base_model_relation: quantized` | n/a | **R** | n/a |
| top-level `datasets:` includes the fidelity dataset | **R** | O | n/a |

Notes:

* A **dataset card** has no `model-index` and no `base_model` rendering, so `x_fidelity` carries
  everything. (`base_model:` on a dataset card validates clean but is undocumented and does not
  render — do not rely on it.)
* `base_model_relation` has exactly four legal values and **none of them means "fidelity
  reference"**. A root's relationship to its captures is expressed only through `x_fidelity` and
  `datasets:`.
* arXiv / paper linking is **not** a YAML field: the Hub scrapes arXiv and HF-Paper links out of the
  card **body** and synthesizes an `arxiv:<ID>` tag. Nothing to design; put the spec link in the body.

---

## 4. Worked K6 card (validated)

Full file: [`examples/card-k6.yaml`](examples/card-k6.yaml). Every value is real, read from
`registry/data/*.jsonl`. Abridged here to the load-bearing structure:

```yaml
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
datasets:
  - brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
  - malaiwah/GLM-5.3-Flash-fidelity-suite-v1
metrics: [kl_divergence, top1_agreement]
tags: [glm, glm-5, tr3, trellis, mcg, quantized, 6-bit, moe, exllamav3,
       fidelity, kl-divergence, fidelity-provenance]
model-index:
  - name: GLM-5.3-Flash-TR3-6bpw
    results:
      - task: {type: text-generation, name: Distribution fidelity (KL divergence vs BF16 reference)}
        dataset:
          type: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
          name: GLM-5.3-Flash sealed qualification panel v1 (25 final windows)
          config: final25
          split: sealed-ep8                       # <- LANE lives in the merge key
          revision: '95f4fdd94bf29989db2e0d1054e4931f55edb6aa'
          args: {panel_id: panel--glm53.brandonmusic.final25,
                 panel_token_sha256: '6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff',
                 contexts: 25, scored_positions: 51175, context_length: 2048,
                 tokenizer: 'zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43',
                 vocab_size: 154880,
                 reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25}
        metrics:
          - type: kl_divergence
            name: Mean tokenwise KLD (reference || candidate), nats
            value: 0.013723384665701147
            args: {units: nats, higher_is_better: false, direction: reference_to_candidate,
                   estimator: full_vocabulary_fp64, accumulation_dtype: float64,
                   logits_dtype: fp32, head_policy: native_head, stack_relation: same_stack,
                   lane: sealed-ep8, run_count: 5, population_stddev_of_run_means: 0.0,
                   determinism: bitwise_identical_across_runs,
                   measurement_id: measurement--glm53.k6-6bpw.brandonmusic-final25,
                   comparability_key: cmp--202b717f3219c414}
        source:
          name: quant-fidelity-registry
          url: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/measurements?q=measurement--glm53.k6-6bpw.brandonmusic-final25
      - task: {type: text-generation, name: Distribution fidelity (KL divergence vs BF16 reference)}
        dataset: {…same panel…, split: streaming}
        metrics:
          - {type: kl_divergence, value: 0.013714888822596553, args: {lane: streaming, run_count: 2, …}}
          - type: kl_divergence_quantization_attributable
            value: 0.0022089662032662542
            args: {derived: true, derivation: candidate_minus_same_lane_floor,
                   floor_value: 0.011505922619330299,
                   floor_measurement_id: measurement--glm53.bf16-stream-floor.brandonmusic-final25,
                   floor_lane: streaming,
                   caveat: 'KL is not additive; this is an estimate, valid only against a floor
                            measured on the same lane, same panel and same reference.'}
          - {type: top1_agreement, value: 0.9656277479237909, args: {higher_is_better: true, lane: streaming}}
        source: {name: quant-fidelity-registry, url: '…k6-6bpw-stream…'}
x_fidelity:
  spec_version: fidelity-provenance/v1
  role: quant
  …
```

**K8 is the same template** with one result at `split: streaming`, `value: 0.012384191023436866`,
`measurement_id: measurement--glm53.k8-8bpw-stream.brandonmusic-final25`, `run_count: 2`, plus the
attributable metric against the same streaming floor. K8 has **no** `sealed-ep8` registry row, so its
card carries no `sealed-ep8` result (XC-3).

### 4.1 Verification results

| check | result |
|---|---|
| Hub `validate-yaml`, K6 card | `{"errors": [], "warnings": []}` |
| Hub `validate-yaml`, root card | clean |
| Hub `validate-yaml`, dataset card (`repoType: dataset`) | clean |
| Hub `validate-yaml`, K6 card **after** a library round-trip | clean |
| `huggingface_hub` 1.29.0 `ModelCard.load` parse, K6 / K8 | OK — **6** / **3** eval results, `model_name` resolved, `x_fidelity` preserved in `to_dict()` |
| `ModelCard.save()` → `load()` round-trip, K6 | `x_fidelity` **deep-equal to the source**, 6 eval results and every `metrics[].args` key retained, `base_model_relation` retained |
| YAML → `ModelCardData` → YAML deep structural equality | **True** (zero lost / added / changed keys) for K6 and root |
| Hub `validate-yaml` accepts an unknown top-level `x_fidelity` | HTTP 200 (isolated probe, minimal card) |
| Hub `validate-yaml` accepts unknown `metrics[].args` keys | HTTP 200 (isolated probe) |
| Hub `validate-yaml` rejects a malformed `model-index` | HTTP 400 `"model-index[0].results[0].metrics" must be an array` — so the 200s above are meaningful |
| generic `model-index` walk (PyYAML only, no spec knowledge) | extracts all 6 `(model, dataset, config, metric, value)` rows from K6 |

`ModelCard.validate()` is a real network POST to `https://huggingface.co/api/validate-yaml`
(`huggingface_hub.repocard.RepoCard.validate`), not a local check — so a PASS on that axis is the
Hub's own verdict, not ours.

**Real-world shape comparison.** Three published cards that actually carry `model-index` were fetched
and compared field-for-field: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (17 rows,
`verified: true`), `facebook/wav2vec2-base-960h` (2 rows), and `BAAI/bge-small-en-v1.5` (**990** rows,
MTEB, every row pinning `dataset.revision`). Our shape is the MTEB shape: many results on one card,
each scoped by `dataset.type` + `config` + `split` and pinned by `revision`. The one field we use
that **none** of them uses is `metrics[].args` — it parses, round-trips and validates, but we are
alone in it, which is the whole substance of the §6 friction analysis. None of the three sets
`metric_args`; `verified` is HF-controlled and we correctly never set it (GEN-7).

Negative controls (proving the validator is not a no-op): bad `license`, bad `language`,
non-string `size_categories`, missing `model-index[0].name`, missing metric `value`, missing
`source.url`, and an all-digit revision all return HTTP 400 with specific error paths.

**Not verified:** live Hub *rendering* of the eval widget for our card. That needs one push to a
scratch/private model repo — a permissioned action. The shape is byte-identical to production cards
(`HuggingFaceH4/zephyr-7b-beta` uses the same `dataset.config` / `dataset.split` / `dataset.args` /
`metrics[].args` / `source` structure) and the push-time validator passes it, so confidence is high,
but the operator should authorize one private scratch push before annotating K6/K8 for real.

---

## 5. Generator and validator

### 5.1 `bin/fidelity-card annotate`

Input: registry `measurement_ids` (or an `artifact_id`). Output: the two layers merged into an
existing card.

**GEN-1** Quote every digest and revision string (§1.1 gotcha).
**GEN-2** Emit exactly **one** `model-index` entry (§1.2.1).
**GEN-3** Put lane in `dataset.split` (§1.2.2).
**GEN-4** Refuse to emit two results sharing
`(task.type, dataset.type, dataset.config, dataset.split, dataset.revision)`.
**GEN-5** Preserve unknown existing card keys: parse with `ModelCard(text)`, merge into `card.data`,
never rewrite the body.
**GEN-6** Write with `to_yaml(original_order=[...])` to minimise diff churn.
**GEN-7** Never set `verified` or `verifyToken`.
**GEN-8** Never invent a head digest. A measurement whose artifact has no published head content
digest emits `head.lm_head_tensor_content_sha256: null` **and**
`head.replay_permitted: false` **and** an explanatory `note`.

### 5.2 `bin/fidelity-card validate`

Three independent axes, all must pass:

1. **Hub axis** — POST the assembled README to `https://huggingface.co/api/validate-yaml`; fail on
   any error. (`--offline` skips this axis and says so in the report; it is the only networked
   check in the tool.)
2. **Round-trip axis** — parse with `huggingface_hub.ModelCard`, re-emit, and assert the
   YAML → `ModelCardData` → YAML round-trip is **structurally identical**. This catches the
   multi-entry collapse and the args-drop merge automatically, without hard-coding either.
3. **Our axis** — role-conditional required fields (§3), XC-1..XC-5, `floor_lane == lane`,
   head-digest presence before hidden replay is claimed, and that every `measurement_id` resolves in
   `registry/data/measurements.jsonl` with a matching `value`, `lane` and `comparability_key`.

### 5.3 Registry adapter

The comparison receipt carries a `card_annotation` block verbatim so `registry_add` can round-trip a
submitted card's claim back to the row it cites. Adding it needs a new adapter keyed on
`malaiwah.fidelity-comparison-receipt.v1` in `registry/tools/registry_add.py` — see the build plan.

---

## 6. HF eval-results v2 — the right long-term home, not shippable today

`hub-docs/docs/hub/eval-results.md` describes a **decentralized** eval system that maps almost 1:1
onto the operator architecture:

* a dataset repo becomes a **Benchmark** by adding `eval.yaml` (`name`, `description`,
  `evaluation_framework`, `tasks[]`), and then hosts a leaderboard that auto-aggregates results from
  model repos across the Hub;
* a model repo publishes scores as `.eval_results/*.yaml`
  (`- dataset: {id, task_id, revision}` / `value` / `date` / `source` / `notes` / `verifyToken`);
* **anyone can submit results for anyone's model by PR**; they render as community-provided while the
  PR is open, and the model author can close the PR to remove a disputed score.

That last bullet is *exactly* "a quant author publishes their own capture and lets others compare
without re-running anything". It is live in production on our own base-model family: `zai-org/GLM-5.3`
ships `.eval_results/{hle,deep-swe,terminal-bench-2.1,terminal-bench-3.0}.yaml` and
`GET /api/datasets/cais/hle/leaderboard` returns it ranked #1 at 62.5.

**Three hard blockers, all outside this repo:**

1. `evaluation_framework` is a closed enum in `huggingface.js/packages/tasks/src/eval.ts` (~30
   entries: inspect-ai, mteb, math-arena, harbor, swe-bench, nemo-evaluator, …). None fits. The docs
   explicitly invite additions — it is an upstream PR.
2. Benchmarks are **allow-listed in beta** ("get in touch so we can add it to the allow-list").
3. **The format cannot express direction.** `value` is a bare number with no metric name, no units
   and no `higher_is_better`. A leaderboard would rank KLD as if higher were better. It also cannot
   express lane, floor, run count, determinism, head identity or reference identity — only a free
   `notes:` string.

**Decision:** ship `model-index` + `x_fidelity` now; emit `.eval_results/fidelity.yaml` behind an
**off-by-default** `--eval-results-v2` flag so the day a fidelity benchmark is allow-listed is a
one-switch day. The `evaluation_framework` PR and the allow-list request are operator outreach items,
alongside the Festr conversation.

---

## 7. Known risks

* **The Hub's web metadata-editor widget may not preserve unknown top-level keys.** The API preserves
  them (proven on production repos), but a human editing metadata through the web UI is an untested
  path. Cards carrying `x_fidelity` should be edited as **README text**, never through the metadata
  widget. State this in the card body.
* **`datasets:` renders as "Datasets used to train:"** — semantically wrong for an eval panel, and
  unavoidable (§2.1).
* **A naive leaderboard cannot tell our headline number from our derived number.** Measured, not
  supposed: a generic `model-index` walk over the K6 card (PyYAML, no knowledge of this spec) yields
  six rows, of which **four** are `kl_divergence` — two lanes x two scopes — plus one
  `kl_divergence_quantization_attributable` at **0.00221** sitting in the same `metrics[]` list as
  the raw **0.01371**. `model-index` has no concept of a *primary* metric, and everything that
  disambiguates these rows (`lane`, `derived: true`, `floor_measurement_id`, `higher_is_better:
  false`) lives in `metrics[].args`, which — per §4.1 — **no published card in the wild uses and no
  existing consumer reads**. So the honest statement of Layer 1's reach is:

  > a leaderboard-style consumer can extract our KLD **value** with zero bespoke code, and cannot
  > extract its **comparability** with any amount of it short of implementing this spec.

  Three consequences we accept deliberately. (1) The value is still in the standard slot, because a
  wrong-but-findable number that we can correct beats a right number nobody can parse. (2) The metric
  `type` strings are self-describing (`kl_divergence_quantization_attributable` is not going to be
  mistaken for `kl_divergence` by a *human*), which is why the derived metric got a distinct type
  rather than an `args` flag on the same one. (3) **This is the strongest argument for the
  `.eval_results/` v2 path in §6**, whose per-benchmark scoping would let the headline row be named
  once and the rest be omitted. Until then, GEN-* must never emit a card whose *only*
  `kl_divergence`-family row is the derived one, and a consumer we care about should be pointed at
  `x_fidelity.measurements[]`, which is ordered and unambiguous.
* **`dataset.split` carries the lane, which is not what `split` means anywhere else.** HF's `split`
  is train/validation/test; we put `streaming` / `sealed-ep8` there. This is deliberate and is
  documented at the top of `bin/fidelity/cardmeta.py`: `huggingface_hub` merges eval results on
  `(task.type, dataset.type, dataset.config, dataset.split, dataset.revision)`, so a lane carried
  only in `args` would cause two lanes' results to **silently collapse into one row** — precisely
  the lane mixing BIAS-006 forbids. Overloading `split` is the least-bad available slot, but an
  outsider will not guess it; XC-2 enforces `args.lane == dataset.split` so the two can never drift.
* **Our K6/K8 cards cannot carry a non-null head content digest until the capture tool publishes
  one.** `head-extraction.json` / `head-equality-fp8.json` publish the *file* digest `47eaf729…`;
  `k6/hidden-replay-evidence/nonrouted-sparse-fetch.json` publishes the *content* digest
  `aa21c427…` — which is the correct value, but it is a working-tree artifact and not yet published
  as part of a sealed dataset. Until it is, the generator emits `replay_permitted: false` (GEN-8) and
  the comparator refuses cross-artifact hidden replay against those cards (HEAD-4). This is the
  intended behaviour, not a workaround.

---

## 8. Implementation addenda (v1, 2026-08-29)

**GEN-9 — never emit a `null` inside `args`; omit the key.** Measured, not
assumed: `huggingface_hub`'s `EvalResult` **drops** null-valued keys inside
`dataset.args` and `metrics[].args` on a round-trip. A card emitting
`population_stddev_of_run_means: null` therefore passes the Hub validator but is
not the card the Hub re-serves, and the round-trip axis flags it as `changed`. A
missing key and a null key mean the same thing here, and only the missing form
survives. The generator strips nulls; the validator refuses them.

**GEN-10 — measurement SCOPE lives in `dataset.config`.** §2.1 puts lane in
`dataset.split` because the merge key would otherwise discard it. The same
argument applies one axis over, and live registry data proves it: the registry
carries `panel25` (25 windows, 51,175 positions) and `clean17` (the 17 that
survive a 13-gram calibration-overlap scan, 34,799 positions) rows for the same
artifact, panel and lane. They share
`(task.type, dataset.type, dataset.config, dataset.split, dataset.revision)`
exactly, so `huggingface_hub` would merge them and silently drop one result's
args. `config` becomes `<panel-config>[-<scope_name>]`, the scope detail goes in
`dataset.args`, and `dataset.name` says which subset it is. The generator
refuses any two results that still collide (GEN-4), which is how this was found.

**XC-6 — a floor must match on scope as well as lane.** XC-4 checks
`floor_lane == lane`. The generator additionally withholds the
`kl_divergence_quantization_attributable` metric — and records the reason in
`x_fidelity.measurements[].quantization_attributable_withheld` — when the floor
row does not resolve, was measured on another lane, or was measured over another
**scope**. Withholding an unverifiable number is the correct behaviour; printing
it is not. Case K8b.

**A registry snapshot travels with the annotation.**
`x_fidelity.registry.snapshot.data_sha256` records the digest of each
`registry/data/*.jsonl` file the card was generated from. A registry clone is a
moving target; without this, a card and the rows it cites can drift with nothing
on either side to say so.
