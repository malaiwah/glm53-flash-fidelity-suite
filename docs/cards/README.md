# `docs/cards/` — the annotated K6 and K8 model cards

These are the **reference implementation** of
[`../CARD-ANNOTATION-SPEC.md`](../CARD-ANNOTATION-SPEC.md), applied to our own
two published models. They are generated files: the frontmatter is produced by
`bin/fidelity-card annotate` from live registry rows, and the **body is byte-identical
to the published card** — `annotate` never rewrites prose.

```
GLM-5.3-Flash-TR3-6bpw.README.md    malaiwah/GLM-5.3-Flash-TR3-6bpw
GLM-5.3-Flash-TR3-8bpw.README.md    malaiwah/GLM-5.3-Flash-TR3-8bpw
```

## They are not published

Pushing a card to a model repository is a permissioned act and is the Ship
phase's job. What is done here is everything that can be verified without
pushing:

| axis | result |
|---|---|
| live Hub `POST /api/validate-yaml` (the same gate a `git push` runs) | clean, both cards |
| `huggingface_hub` YAML → `ModelCardData` → YAML round-trip | structurally identical, both cards |
| our XC-1..XC-5 cross-checks against `registry/data/measurements.jsonl` | clean, both cards |

What is **not** verified is how the eval widget *renders*, which needs one push
to a private scratch model repo. The shape is byte-for-byte the structure
`HuggingFaceH4/zephyr-7b-beta` uses in production, so confidence is high, but
the operator should authorize that one scratch push before annotating the real
repositories.

## Regenerating

The registry is a moving target — rows get added, scopes get split. Each card
records which registry state produced it, in
`x_fidelity.registry.snapshot.data_sha256`. Regeneration is one command:

```bash
bin/fidelity-card annotate \
  --card <the current published README.md> \
  --role quant --model-name GLM-5.3-Flash-TR3-6bpw \
  --artifact-id artifact--malaiwah.glm-5.3-flash-tr3-6bpw \
  --base-model zai-org/GLM-5.3-Flash-BF16 \
  --reference-model zai-org/GLM-5.3-Flash-BF16 \
  --reference-revision a6c167b62691b2bac901344b65cb651a70f53e43 \
  --head-file-sha256 47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0 \
  --final-norm-file-sha256 c228a123dee3062c3ad0129094e9d98a264e33087ee88d79c8d6c5a6e60f2fed \
  --equality-receipt "https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/head-equality-fp8.json" \
  --dataset brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
  --dataset malaiwah/GLM-5.3-Flash-fidelity-suite-v1 \
  --out docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md --validate
```

`--artifact-id` resolves every **published** measurement for that artifact, so a
new row appears in the card automatically and XC-3 keeps the two layers in step.

## What the annotation actually says

Layer 1, `model-index` — one entry, one result per (panel-scope, lane) pair:

* **lane lives in `dataset.split`.** `huggingface_hub` merges results on
  `(task.type, dataset.type, dataset.config, dataset.split, dataset.revision)`,
  so a lane carried only in `dataset.args` is silently discarded and two lanes
  collapse into one — exactly the mixing **BIAS-006** forbids.
* **measurement scope lives in `dataset.config`.** The registry carries
  `panel25` (all 25 windows) and `clean17` (the 17 that survive a 13-gram
  calibration-overlap scan) rows for the same artifact and lane. They score
  different position counts and must not merge.
* the floor-subtracted number is a **second metric in the same result**, never a
  headline, carrying `floor_measurement_id`, `floor_lane` and the
  non-additivity caveat.

Layer 2, `x_fidelity` — what `model-index` structurally cannot hold: the
fidelity-dataset pointer, the registry ids, the scope digest, determinism
evidence, and head identity.

## The head digest is deliberately null

Both cards ship `x_fidelity.head.lm_head_tensor_content_sha256: null` and
`replay_permitted: false`.

That is not an omission. Our published receipts
(`head-extraction.json`, `head-equality-fp8.json`) record the head's **file**
digest `47eaf729…`, which is a container digest and never an identity. The
**tensor content** digest is `aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4`
— recomputed independently from the published `head/head.safetensors` by
`bin/fidelity/dsformat.py::tensor_content_sha256` — but it is not yet part of a
sealed, published dataset. Until it is, the generator refuses to write it
(GEN-8) and the comparator refuses cross-artifact hidden replay against these
cards (HEAD-4). Filling it in is one line once a capture publishes it.
