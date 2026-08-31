# Build plan — fidelity dataset tooling

Implementation plan for [`FIDELITY-DATASET-SPEC.md`](FIDELITY-DATASET-SPEC.md) and
[`CARD-ANNOTATION-SPEC.md`](CARD-ANNOTATION-SPEC.md). Every design decision is already made in those
two documents; this file says **what to write, where, with what signature, and what it must refuse**.

---

## 0. The hard constraints, restated as rules

A concurrent measurement workflow (M1–M4) owns these five files. **Do not edit them:**

```
bin/measure_cloud.py   bin/stage_measure.sh   bin/fidelity/hfmeta.py
bin/engines.json       bin/invoke_engine.py
```

Also **do not edit** `k6/tools/stream_score.py` (the format adapters just merged there).

Everything below is **new files only**. Existing code is **imported, wrapped or shelled out to** —
never modified. The precedent is `k6/tools/hidden_replay.py`, which attaches a capture hook by
monkeypatching `stream_score.build_streaming_model` at run time and changes nothing in
`stream_score`'s own path.

`registry/` may gain a `registry_add` adapter and new invariants, but **`make check` must stay at 0
errors**.

Python: `bin/` tools run on **py3.9** (system `python3`) with **stdlib only**; torch paths run under
`FIDELITY_PYTHON` / `.venv/bin/python`. Same split every existing `bin/` tool uses.

---

## 1. New files

### 1.1 Library — `bin/fidelity/`

| file | py | deps | purpose |
|---|---|---|---|
| `bin/fidelity/dsformat.py` | 3.9 | stdlib | Format constants, path rules, the five digest preimages, `capture_content_digest`, seal/verify, `checksums.txt` read/write, manifest read/write. **No torch at import.** |
| `bin/fidelity/dsmanifest.py` | 3.9 | stdlib | Manifest builders: `from_stream_capture()`, `from_hidden_capture()`, `from_serving_manifest()`, `panel_binding()`, `head_block()`, `runtime_block()`, `coverage_block()`. |
| `bin/fidelity/dsvalidate.py` | 3.9 | stdlib | Structural + seal validator: JSON Schema via the registry's vendored `_minischema`, plus the ~40 rules the schema cannot express. |
| `bin/fidelity/dscompare.py` | torch | torch, safetensors | The gate ladder and the fp64 estimator. Imports from `kld_report` and `hidden_replay`; reimplements nothing. |
| `bin/fidelity/dsadapt.py` | 3.9 | stdlib | Foreign-format adapters: `k3v1`, `k3v0_window`, `llamacpp_kld`, `malaiwah_serving_v2`. |
| `bin/fidelity/dshub.py` | 3.9 | stdlib (+ `huggingface_hub` when importable) | Fetch / publish / list. Token read from `HF_TOKEN` or a file path; registered with `common.register_secret()` **before** anything can print. |
| `bin/fidelity/cardmeta.py` | 3.9 | `PyYAML` when present, else a refusal | Card generator + validator (Layers 1 and 2), the three validation axes, `.eval_results/` v2 emitter behind a flag. |

### 1.2 CLIs — `bin/`

| shim | script | subcommands |
|---|---|---|
| `bin/fidelity-dataset` | `bin/fidelity_dataset.py` | `capture`, `verify`, `compare`, `validate`, `adapt`, `describe`, `publish` |
| `bin/fidelity-card` | `bin/fidelity_card.py` | `annotate`, `validate` |

Shims follow the existing one-line pattern:

```bash
#!/usr/bin/env bash
# <one-line description>. See fidelity_dataset.py --help.
exec "${FIDELITY_PYTHON:-python3}" "$(cd "$(dirname "$0")" && pwd)/fidelity_dataset.py" "$@"
```

### 1.3 Selftests — `bin/`

| file | py | what |
|---|---|---|
| `bin/selftest_fidelity_dataset.py` | 3.9 + numpy | The synthetic matrix (§5). No torch, no network, no GPU. |
| `bin/selftest_fidelity_card.py` | 3.9 + PyYAML | The card matrix (§6). `--offline` skips the one networked axis. |
| `bin/selftest_fidelity_compare.py` | torch | Known-answer KLD and the self-compare exactness assertions that need torch. Skips cleanly when torch is absent. |

Registered in `bin/selftest_all.sh` via its existing `t` / `s` helpers, in the offline block:

```
t "fidelity dataset format + seals (T6)"   0 python3 bin/selftest_fidelity_dataset.py
t "fidelity card annotation (T7)"          0 python3 bin/selftest_fidelity_card.py --offline
t "fidelity comparator known answers (T8)" 0 "$PY" bin/selftest_fidelity_compare.py
```

`bin/selftest_all.sh` **is** editable — it is not one of the five reserved files. Confirm at
implementation time that the measurement workflow has not claimed it; if it has, ship the three
lines as a patch note in the JOURNAL instead.

### 1.4 Schemas and docs (already written)

```
docs/FIDELITY-DATASET-SPEC.md
docs/CARD-ANNOTATION-SPEC.md
docs/FIDELITY-DATASET-BUILD-PLAN.md            (this file)
docs/schema/fidelity-dataset.schema.json
docs/schema/fidelity-comparison-receipt.schema.json
docs/schema/fidelity-card-annotation.schema.json
docs/examples/fidelity-dataset.root-glm53-bf16.json
docs/examples/fidelity-dataset.quant-glm53-k6.json
docs/examples/fidelity-comparison-receipt.k6-vs-bf16.json
docs/examples/fidelity-comparison-receipt.self-compare.json
docs/examples/card-{k6,k8,root-bf16,dataset-suite-v1}.yaml
docs/examples/SYNTHETIC-DIGESTS.md
```

Still to write: a `WHAT-WE-MEASURE.md` section 8 ("Capture and comparison are two steps"), a
`bin/README.md` section, and a JOURNAL entry.

---

## 2. CLI signatures

### 2.1 `bin/fidelity-dataset capture`

```
bin/fidelity-dataset capture
    --out DIR                       dataset root to create (must not exist, or --force)
    --form {hidden,logit}           default: hidden
    --role {root,quant,derived}     REQUIRED, no default
    --panel PATH|REPO@REV           sealed panel receipt or an HF panel repo
    --dataset-id ID                 e.g. fidelity--malaiwah.glm53-bf16.final25.hidden
    --name TEXT
    --lane {sealed-ep8,streaming,local-mps,local-cuda-budget,other}   REQUIRED
    [--head PATH]                   head payload to ship; default: extract from the weights
    [--emit-k3-compat]              also write compat/ (spec §12.3)
    [--repeats N]                   capture N additional runs into determinism/repeat-NN/
    [--upstream-receipt PATH ...]   copy verbatim into upstream/, stripping host paths
    [--shard-of I/N]                declare this a shard
    [--dry-run]                     validate every input and the plan; exit 0 without a GPU
    [--publish REPO]                push to HF after `verify` passes locally
    [--private]
    -- <stream_score argv ...>      everything after `--` is passed through verbatim
```

**Behaviour**

1. Pre-flight refusals on the pass-through argv, inherited from `hidden_replay.run_capture`:
   refuse `--sweep` (extra forwards interleave hiddens), refuse any `--store-positions` other than
   `all`, and **require an explicit `--token-panel`** (the wrapper needs the mask `.npy` paths, which
   the capture receipt does not carry).
2. Hidden form: exec `k6/tools/hidden_replay.py capture -- <argv>`. Logit form: exec
   `k6/tools/stream_score.py <argv>` directly. Either way `stream_score`'s own path is byte-identical
   to a plain run.
3. After the run, read `capture-receipt.json`, `backend.json`, and (hidden form)
   `hidden-capture.json`; assert `len(hiddens) == len(logit_files)`.
4. Build the dataset tree, compute all digests, write `checksums.txt`, seal the manifest.
5. Run `validate --verify-tensors` on the result and **refuse to finish** if it fails.
6. `--publish` runs `verify` again on the fetched copy after upload.

**`--dry-run` is the CI hook.** `stream_score --dry-run` validates every input, seal and layout and
exits 0 **without touching weights or a GPU**; the capture command forwards it and then validates the
*planned* manifest shape. That gives a conformance test that runs on this Mac.

**Known work item (suite scale).** `hidden_replay`'s tap accumulates every window in CPU RAM before
writing: 25 windows × 2048 × 4096 × 2 B ≈ 419 MiB is fine, but a 400-context shard is ~6.6 GiB and
the full 10.48M-position suite is ~86 GB. For anything beyond a panel-sized capture the tap must
**flush per window and drop**, which needs a post-forward callback in the tap rather than the current
post-`main()` loop. Implement the streaming tap in `bin/fidelity/dsmanifest.py` as an optional
`flush_fn`, still without editing `stream_score.py`.

**Model-agnosticism.** `hidden_replay` hardcodes `EXPECTED_VOCAB = 154880` and `HIDDEN_WIDTH = 4096`.
The capture command must read both from the model config and pass them down, so v1 is not
GLM-only.

### 2.2 `bin/fidelity-dataset verify`

```
bin/fidelity-dataset verify
    DATASET                         local dir, or hf://<repo>[@<rev>]
    [--verify-tensors]              recompute every tensor_content_sha256 (slow, exact)
    [--manifest-only]               seal + schema only; no file reads beyond the manifest
    [--allow-partial]               permit coverage.complete == false
    [--json OUT]                    write the verification report
    [--cache DIR]
```

Checks, in order, each with its own exit reason:

| # | check | failure |
|---|---|---|
| 1 | manifest parses; `schema` and `format_version` exact | `bad_schema` |
| 2 | JSON Schema (vendored `_minischema`) | `schema_invalid` |
| 3 | `dataset_sha256` self-seal recomputes | `seal_failed` |
| 4 | `sha256(checksums.txt) == seal.checksums_sha256` | `seal_failed` |
| 5 | every line of `checksums.txt` verifies; no unlisted file; no symlink | `unlisted_file` / `missing_file` / `symlink` |
| 6 | every path-valued field is relative and resolves inside the root (PATH-1/3) | `path_escape` |
| 7 | sub-manifests self-seal (`capture/manifest.json`, `panel/panel.json`, `head/head.json`, `runtime/capture-runtime.json`, `panel/panel-remap.json`) | `seal_failed` |
| 8 | `capture_content_digest` recomputes from the record list | `digest_mismatch` |
| 9 | `suite_token_hash_sha256` recomputes from `panel/tokens/` | `panel_digest_mismatch` |
| 10 | BIND-1..6 | `panel_binding` |
| 11 | coverage COV-1..3 | `incomplete` (unless `--allow-partial`) |
| 12 | `--verify-tensors`: every `tensor_content_sha256`, `payload_sha256`, `file_sha256`, and the head's | `tensor_mismatch` |
| 13 | remap REMAP-1..3 | `remap_invalid` |

**There is no `--force`.** A tampered or partial dataset is refused; `--allow-partial` narrows the
refusal to a stamped disclosure and nothing else.

Exit codes: `0` ok, `2` warnings only, `3` refused, `4` bad usage.

### 2.3 `bin/fidelity-dataset compare`

```
bin/fidelity-dataset compare
    --reference A                   local dir or hf://<repo>[@<rev>]
    --candidate B
    --out DIR
    [--device auto|cpu|cuda:N|mps]
    [--vocab-chunk N]               must divide vocab_size exactly; 9680 for GLM-5.3-Flash
    [--chunk-positions N]
    [--head PATH]                   only with --disclose-head-substitution
    [--self-compare]                assert A and B are the same capture
    [--force-compute]               run the math even when the hash proof answers
    [--allow-cross-lane]
    [--allow-partial]
    [--disclose-head-substitution]
    [--emit-submission]             also write a registry submission receipt
    [--json OUT]
```

Runs the gate ladder (spec §10.1) and stops at the first refusal with the named reason and exit 3.
Then computes (spec §10.2), then writes `<out>/comparison-receipt.json` and
`<out>/tokenwise-kld.npy`.

**What it wraps, never reimplements:**

| from | used for |
|---|---|
| `k6/tools/k6_kld_report.py::_token_kld` | the fp64 full-vocabulary per-token KL |
| `k6/tools/k6_kld_report.py::_load_slice`, `_record_map`, `_resolve_teacher_paths`, `_find_teacher_receipt` | slice streaming, record matching, digest-verified path fallback, schema-not-filename discovery |
| `k6/tools/hidden_replay.py::_replay_logits`, `summarize_tokenwise`, `payload_sha256`, `tensor_content_sha256` | hidden→logit replay, the `{mean, median, p95, p99, p99_9, max}` block, the two content digests |
| `k6/tools/stream_score.py::resolve_device`, `apply_numeric_policy` | device selection and the numeric policy, so the comparator's numerics match the capture lane's |
| `bin/fidelity/stackprint.py::from_backend_json`, `fingerprint_sha256` | the embedded stack fingerprint |
| `bin/fidelity/previewstats.py` | context / source-cluster / stratified-cluster bootstraps |
| `bin/fidelity/receipt.py::build_submission`, `assert_submittable`, `produced_by_block` | the registry submission and its refusals |
| `registry/tools/registry_lib.py::scope_digest`, comparability key | derived hashes, computed by the registry's own code |

Two behaviours worth naming because they are easy to get wrong:

* **`kld_report` resume semantics.** If `<run>/kld-report.json` already exists it is returned
  as-is after a `student_label` check. The comparator must either respect that (and say so in the
  receipt) or write into a fresh directory. Silently reusing a stale report is the failure mode.
* **`glm53_logits.load_capture_receipt` enforces an exact 10-key set** on every `logit_files[]` row
  (`set(row) != required` raises). **Never add keys to that row.** Extra per-record fields go in a
  parallel array in our own manifest.

### 2.4 `bin/fidelity-dataset validate`

```
bin/fidelity-dataset validate DATASET [--verify-tensors] [--json OUT] [--strict]
```

Same engine as `verify` steps 1–13, but writes `validation/structural-validation.json` into the
dataset and reports **every** failure rather than stopping at the first. `--strict` turns warnings
into errors. This is the command a third party runs before trusting a downloaded dataset.

Also validates a **comparison receipt** when handed one
(`validate --receipt <path>` → `fidelity-comparison-receipt.schema.json` + the SC-1/SC-2/BIAS-001
conditionals).

### 2.5 `bin/fidelity-dataset adapt`

```
bin/fidelity-dataset adapt
    --source {k3v1,k3v0-window,llamacpp-kld,malaiwah-serving-v2}
    --in PATH|hf://<repo>[@<rev>]
    --out DIR
    [--panel PATH]                  when the source does not carry one
    [--allow-partial]
    [--recompute-content-digests]   read tensors to upgrade container digests to content digests
```

Emits a conformant dataset whose `interop.adapted_from.inferred_fields[]` names every field the
adapter had to synthesize. **Every entry in that array forces `comparability.class = advisory`** at
compare time. Per-source detail is in spec §12.4 (k3v1) and §12.6 (llama.cpp).

The adapter **refuses to fabricate**: a k3 candidate directory is refused with the explanation that
his artifact publishes no candidate captures at all, only compare receipts.

### 2.6 `bin/fidelity-dataset describe` / `publish`

```
bin/fidelity-dataset describe DATASET [--format {text,json,markdown}]
bin/fidelity-dataset publish DATASET --repo REPO [--private] [--revision-message TEXT]
```

`describe` prints the identity card: role, form, lane, panel, head digest, coverage, determinism,
seal, divergences. `publish` refuses unless `verify --verify-tensors` passes first, refuses to push
a dataset whose `structural_status` is `draft`, and re-verifies the fetched copy afterwards.

### 2.7 `bin/fidelity-card annotate` / `validate`

```
bin/fidelity-card annotate
    --card PATH|hf://<repo>          the existing README.md to merge into
    --role {root,quant,fidelity-dataset}
    (--measurement-id ID ...  |  --artifact-id ID)
    [--registry DIR]                 default: ./registry
    [--fidelity-dataset REPO@REV]    fills x_fidelity.fidelity_dataset
    [--dataset-sha256 HEX]
    [--head-content-sha256 HEX]      never invented; omitted -> replay_permitted false
    [--eval-results-v2]              ALSO emit .eval_results/fidelity.yaml (off by default)
    [--out PATH] [--in-place] [--diff]

bin/fidelity-card validate
    --card PATH|hf://<repo>
    [--registry DIR] [--offline] [--json OUT]
```

`annotate` obeys GEN-1..GEN-8 (card spec §5.1). `validate` runs the three axes (card spec §5.2);
`--offline` skips the Hub axis and **says so in the report** rather than silently passing.

---

## 3. Integration points — documented, not built

Integration with the measurement workflow is **out of scope**. These are the seams, so whoever owns
those files later has a one-line change rather than an investigation:

| seam | what a future integration would do |
|---|---|
| `bin/stage_measure.sh` stage order is `measure → score → seal` | a `dataset` stage slots between `measure` and `score`: capture once, score many times |
| `bin/engines.json` `scorer` block + `bin/invoke_scorer.py` compose the scorer argv from `flag_map` | a new capture lane is a JSON edit there, not new code |
| `bin/fidelity/stackprint.py::from_backend_json(backend)` is documented as *"call this right after `backend` is assembled and store as `backend['stack_fingerprint']`"* | `stream_score.py` is off-limits, so the **dataset manifest** calls it instead, on the published `backend.json`. When `stream_score` is unfrozen, move the call and the manifest reads the field it already expects. |
| `tools/fidelity.py cmd_replay --candidate-head` defaults to `None` | mirror HEAD-1b there; until then the comparator refuses the same condition |
| `bin/measure_cloud.py` produces the run trees the capture command reads | nothing to change: the capture command reads the tree, it is not called by it |

---

## 4. Registry changes

Minimal, additive, and `make check` must stay at 0 errors.

1. **New adapter** in `registry/tools/registry_add.py`, keyed on the exact string
   `malaiwah.fidelity-comparison-receipt.v1`, added to `OWN_SCHEMAS`. It maps the receipt onto a
   measurement row and **refuses** any receipt whose `comparison_kind != "measurement"` — the
   registry-side twin of `_scan_for_unsubmittable`.
2. **Two new disclosure codes** in `registry/schema/invariants.json → known_disclosure_codes`
   (required by **DISC-004**):
   * `lossy_capture_codec` — a capture whose stored values are not the model's values;
   * `head_substituted` — a comparison that applied one artifact's head to another's hidden states.
3. **Four new invariants**, all mechanically checkable from data already in the rows:
   * **DS-001** a row derived from a fidelity dataset carries the dataset's `dataset_sha256` as a
     hashed source;
   * **DS-002** `head_substituted` at severity `blocking` forces `status ∈ {pending, retracted}`
     (a specialization of DISC-003, stated so the code is not silently downgraded);
   * **DS-003** a `reproduction_confirmation` receipt never becomes a measurement row;
   * **DS-004** a row citing a fidelity dataset on a lane different from the row's own lane carries a
     `cross_engine_capture` or `non_sealed_lane` disclosure.
4. **No schema-enum changes.** `lane` keeps its five values; a k3-adapted dataset maps to `other`
   with `lane_inferred: true`. Adding `serving` would reclassify existing rows and is an operator
   decision (spec §14).

Run `cd registry && make check` after each step. Non-negotiable.

---

## 5. Synthetic test matrix — `bin/selftest_fidelity_dataset.py`

All fixtures are built in a temp dir by the test itself: pure JSON plus tiny numpy tensors. No
network, no GPU, no torch. Each case names the spec rule it exercises.

### 5.1 Format and seal

| # | case | expect |
|---|---|---|
| F1 | round-trip: build → seal → verify | pass |
| F2 | flip one byte in a capture tensor | `tensor_mismatch`, exit 3 |
| F3 | flip one character in `checksums.txt` | `seal_failed`, exit 3 |
| F4 | re-serialize the manifest with different key order | seal still verifies (canonical JSON) |
| F5 | add an unknown top-level key | verify passes (additive rule §1.3) |
| F6 | add an extra file not in `checksums.txt` | `unlisted_file` |
| F7 | delete a listed file | `missing_file` |
| F8 | absolute path in `capture.records[].file` | `path_escape` (PATH-1) |
| F9 | `..` escaping the root | `path_escape` |
| F10 | a symlink in the tree | refused (PATH-4) |
| F11 | `compat/reference-hidden/manifest.json` with `../../capture/...` | permitted (PATH-3) |
| F12 | manifest with `dataset_sha256` recomputed after an edit | old seal fails, new seal passes |
| F13 | `capture_content_digest` vs a reordered record list | digest unchanged (sorted by index) |
| F14 | `capture_content_digest` vs a changed tensor | digest changes |
| F15 | rewrite only the safetensors `__metadata__` | `file_sha256` changes, `payload_sha256` and `tensor_content_sha256` do **not** (DET-D2) |

### 5.2 Panel binding

| # | case | expect |
|---|---|---|
| P1 | matching panels | pass |
| P2 | different `suite_token_hash_sha256` | `panel_mismatch`, exit 3 |
| P3 | same aggregate, one record's `token_ids_json_sha256` differs | `panel_mismatch` (BIND-2) |
| P4 | `attention_mask_sha256` differs | `panel_mismatch` (BIND-3) |
| P5 | `scoring_window.score_from` 0 vs 1024 | `panel_mismatch` (PANEL-D3, D-3) |
| P6 | `panel_receipt_sha256` reused as `panel_token_sha256` | `schema_invalid` (PANEL-D2) |
| P7 | tokens edited, aggregate not recomputed | `panel_digest_mismatch` (BIND-6) |
| P8 | compact vs default `json.dumps` separators | the two preimages differ; both recorded, only the compact one is normative (§5.1) |
| P9 | remap entry whose target does not hash to its key | `remap_invalid` (REMAP-2) |

### 5.3 Head identity

| # | case | expect |
|---|---|---|
| H1 | hidden↔hidden, equal head content digests | pass; `head_policy = shared_reference_head`; disclosure severity `info` (HEAD-1a) |
| H2 | hidden↔hidden, **different** head content digests | **REFUSE**, exit 3 (HEAD-1b) |
| H3 | H2 + `--disclose-head-substitution` | emits with `class = advisory`, bias `downward`, disclosure `head_substituted` severity `blocking` |
| H4 | logit↔logit, different head digests | pass, never refuse; `head_policy = native_head` (HEAD-2) |
| H5 | hidden↔logit, head digests equal | pass (HEAD-3) |
| H6 | hidden↔logit, head digests differ | REFUSE (HEAD-3) |
| H7 | hidden form, `head.tensor_content_sha256 == null` | REFUSE, **no override** (HEAD-4) |
| H8 | hidden form, `head.applied_in_capture == true` | `schema_invalid` (HEAD-5) |
| H9 | `role: root`, `head.present == false` | `schema_invalid` (HEAD-6) |
| H10 | `semantic_point` post-norm + `final_norm.applied_at_replay == true` | `schema_invalid` (HEAD-7) |
| H11 | head `file_sha256` equal but `tensor_content_sha256` different | REFUSE — content is normative (O-6) |

### 5.4 Lane, stack, coverage, lossy

| # | case | expect |
|---|---|---|
| L1 | same lane | `same_lane: true`, `usable_as_floor: true` |
| L2 | different lanes, no flag | `lane_mismatch`, exit 3 |
| L3 | L2 + `--allow-cross-lane` | `class advisory`, bias block, **`usable_as_floor: false`** (BIAS-006) |
| L4 | equal `lane_identity_sha256` and `stack_fingerprint_sha256` | `stack_relation = same_stack`, no bias needed |
| L5 | differing stack fingerprints | `cross_stack` ⇒ bias block REQUIRED (BIAS-001); receipt without one fails schema |
| C1 | declared 5120, present 512, `complete: true` | `schema_invalid` / `incomplete` (COV-1 — our own O-3 defect) |
| C2 | C1 corrected with `shard_of: {0, 10}` | passes with `covers_full_panel: false` |
| C3 | index sets differ between A and B, no flag | `coverage_mismatch` |
| C4 | C3 + `--allow-partial` | intersect + `subset_of_panel` disclosure (SCOPE-010) |
| X1 | `lossy_codec` non-null on one side | `class advisory` + `lossy_capture_codec` disclosure (D-8) |
| X2 | `dtype_lossless: false` | `class advisory` (FORM-1) |

### 5.5 Numerics — `bin/selftest_fidelity_compare.py`

| # | case | expect |
|---|---|---|
| N1 | **known-answer KLD**: two hand-built captures, 2 positions × 8-entry vocab, analytic fp64 KL | computed == analytic to 1e-15 |
| N2 | `KL(x‖x)` on a random capture | all-zero, exactly |
| N3 | **self-compare, A == B by digest** | mean exactly `0.0`, top-1 exactly `1.0`, every per-window max `+0.0` (not `-0.0`), `comparison_kind = reproduction_confirmation`, `short_circuited: true` |
| N4 | N3 with `--force-compute` | the computed array is bitwise identical to the short-circuit answer |
| N5 | **the T1 constant**: `tokenwise-kld.npy` for a 51,175-position panel | **409,528 bytes**, sha256 `3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17` |
| N6 | same weights identity, different capture content | `comparison_kind = run_to_run_floor`, never `reproduction_confirmation` (SC-2) |
| N7 | vocab-chunk invariance: two chunk sizes | mean difference `< 1e-12` |
| N8 | `--vocab-chunk` that does not divide `vocab_size` | refused with the divisor hint (154880 → 9680) |
| N9 | a NaN injected into one capture | hard refusal, never a clamp |
| N10 | a permuted head applied at replay | KLD is large (order 10, cf. `hidden_replay_selftest` rung c: 11.33 vs 1.013e-4) — proves the estimator has teeth |
| N11 | reproduction-confirmation receipt fed to `build_submission` | `NotSubmittable` (SC-3) |

### 5.6 Real-artifact fixtures (cheap, metadata only)

| # | fixture | what it proves |
|---|---|---|
| R1 | our published `reports/*.json` (27 small files) | the receipt adapters read real receipts |
| R2 | kimi-k3 `manifest.json` + `suite-manifest.json` + `capture-runtime.json` (~2.5 MB, already fetched) | the `k3v1` adapter runs end-to-end on real metadata |
| R3 | `festr2/kimi-k3-full-mxfp4-kld-reference-32x2048` `suite-manifest.json` + `ref/manifest.json` (49 KB) | a second, structurally different fixture: window-form **and** logit-form |
| R4 | `docs/examples/*.json` | schema + seal round-trip on the shipped worked examples |
| R5 | `docs/examples/card-*.yaml` vs `registry/data/measurements.jsonl` | XC-1..XC-5 on live registry data |

**No bulk tensors are downloaded by any test.**

### 5.7 Already-green tests this work must not break

`bin/selftest_zero_floor.py` (PASS 8/0/0) and `k6/tools/hidden_replay_selftest.py` (6/0, including
`e-hook-mechanism: captured == post-norm (bitwise)` and `f-payload-sha`) are the existing proofs the
comparator's assumptions rest on. Both must still pass after every change.

---

## 6. Card test matrix — `bin/selftest_fidelity_card.py`

| # | case | expect |
|---|---|---|
| K1 | the four shipped example cards | schema 0 errors; round-trip structurally identical |
| K2 | K6 card XC-1..XC-5 against `registry/data/measurements.jsonl` | 0 failures |
| K3 | two `model-index` entries | refused (GEN-2 / §1.2.1) |
| K4 | two results sharing the 5-tuple merge key | refused (GEN-4 / §1.2.2) |
| K5 | lane only in `dataset.args`, not in `split` | refused |
| K6 | all-digit unquoted revision | refused (the YAML integer trap) |
| K7 | `replay_permitted: true` with a null head content digest | refused (XC-5) |
| K8 | `quantization_attributable` whose `floor_lane != lane` | refused (XC-4 / BIAS-006) |
| K9 | K8 card carrying a `split: sealed-ep8` result | refused — no such registry row (XC-3) |
| K10 | `base_model_relation: fidelity-reference` | refused: the enum has exactly four values |
| K11 | a card with pre-existing unknown top-level keys | preserved verbatim through annotate (GEN-5) |
| K12 | `verified: true` set by hand | refused (GEN-7) |
| K13 | live Hub `validate-yaml` on all four cards | `{"errors": [], "warnings": []}` — **skipped under `--offline`, and the skip is reported** |

---

## 7. Implementation order

1. `bin/fidelity/dsformat.py` + `bin/selftest_fidelity_dataset.py` §5.1 — digests and seals first;
   everything else depends on them being right.
2. `bin/fidelity/dsvalidate.py` + `validate` + §5.2–5.4 — the refusals, before anything can produce
   data that would need them.
3. `bin/fidelity/dsmanifest.py` + `capture --dry-run` — buildable and CI-testable without a GPU.
4. `bin/fidelity/dscompare.py` + `compare` + `bin/selftest_fidelity_compare.py` §5.5.
5. `bin/fidelity/dshub.py` + `verify hf://…` + `publish`.
6. `bin/fidelity/dsadapt.py` + `adapt` + §5.6 R2/R3.
7. `bin/fidelity/cardmeta.py` + `bin/fidelity-card` + §6.
8. Registry adapter + invariants; `make check` at 0 errors.
9. Docs: `WHAT-WE-MEASURE.md` §8, `bin/README.md`, JOURNAL.
10. Apply the annotation to the real K6 and K8 cards (**after** an operator authorizes one private
    scratch push to confirm live rendering).

Steps 1–4 need nothing but this Mac. Steps 5–7 need read-only HF. Step 10 needs a decision.

---

## 8. Out of scope, with reasons

Beyond spec §14, which lists the format-level exclusions:

| item | why |
|---|---|
| **Any edit to the five reserved files or `k6/tools/stream_score.py`** | the M1–M4 workflow owns them; every capture path is wrapped instead |
| **Wiring the dataset stage into `bin/measure-cloud`** | explicitly out of scope in the brief; seams documented in §3 |
| **A `serving` lane enum value** | changes `submission.schema.json` and reclassifies existing rows; operator decision |
| **Running a real capture** | no GPU, no rentals. Everything here is validated with synthetic fixtures and real *metadata* |
| **Downloading kimi-k3 tensors** (30 GB hidden / 120 GiB live-logit) | metadata only. The consequence is that imported k3 determinism evidence downgrades to `run_mean_equality_only` unless someone later pays for one pass over the 64 sentinel files (~1.8 GB) to compute content digests |
| **Publishing our K6/K8 fidelity datasets** | needs a capture run, which needs a GPU |
| **Signature / attestation (GPG, sigstore)** | the seal is tamper-evident, not tamper-proof; binding it to an identity is a separate decision |
| **A web viewer for datasets** | `registry-view` already renders measurements; a dataset browser is a nice-to-have with no current consumer |
| **Automatic HF PR submission of annotated cards** | posting to someone's repo is a permissioned action; `annotate --diff` produces the patch, a human sends it |

---

## 9. Open items the implementer must decide (each with a default)

| # | item | default |
|---|---|---|
| 1 | `quant_pipeline` is not installed on this Mac (only scratchpad checkouts). Tools importing `glm53_logits` need a path. | Accept `--pipeline-root` / `QP_PIPELINE_ROOT` exactly as `stream_score.py` and `kld_report.py` do. Do **not** vendor `glm53_logits.py`. |
| 2 | No sealed token-panel receipt exists locally; the only real one is a scratchpad copy whose `artifacts[]` are `/workspace/...` absolute paths. | Synthetic panels for all fixtures; the remap path (§7.4) is exercised with a synthetic sealed receipt. |
| 3 | Two head-digest conventions live in our published receipts (`47eaf729…` file vs `aa21c427…` content). | Content is normative; the generator carries both; comparing across conventions is a hard error (H11). |
| 4 | Our two lanes disagree on the hidden tensor key (`hidden` vs `hidden_states`). | `hidden_states` is normative; readers accept `hidden` from pre-v1 artifacts and rewrite on ingest with a disclosure. |
| 5 | kimi-k3's per-file `sha256` is a **container** hash, so imported sentinel determinism evidence cannot reach our required strength without reading tensors. | Downgrade to `run_mean_equality_only` and say so; `--recompute-content-digests` is the paid upgrade path. |
| 6 | `bin/selftest_all.sh` may be claimed by the measurement workflow. | If it is, ship the three registration lines as a JOURNAL patch note instead of editing the file. |
| 7 | Live Hub *rendering* of the annotated card is unverified. | Ask the operator for one private scratch-repo push before annotating K6/K8 for real. |

---

## 10. As built (2026-08-29) — deviations from this plan

Everything in §1 was built. Five deviations, each because reality contradicted
the plan:

| # | planned | built | why |
|---|---|---|---|
| 1 | `registry/` gains a `registry_add` adapter, two disclosure codes and four invariants | **nothing under `registry/` was touched**; the changes are specified in [`REGISTRY-INTEGRATION.md`](REGISTRY-INTEGRATION.md) | the sequential measurement workflow holds `registry/schema/invariants.json`, `registry/data/*.jsonl`, `registry/index.json`, `registry/Makefile` and `registry/tools/seed_registry.py` open in the working tree, and `make check` passed through an intermediate 11-error state during this work. Editing a 90-invariant file another workflow is editing is how you get a conflict in the one place correctness is enforced. `make check` ends at 62 passed / 0 failed with none of this work in it. |
| 2 | `bin/fidelity/dscompare.py` is a torch module | the **gate ladder is stdlib + numpy** and torch is imported lazily inside the estimator | every refusal (head, panel, lane, coverage, lossy) is then testable on the system python3 with no GPU, which is where the whole synthetic matrix runs. `kld_report._token_kld` is still imported and called whenever torch is present, and `comparator.estimator_backend` records which path ran (addendum A-3). |
| 3 | `adapt` always emits a conformant dataset | `adapt --source k3v1 / k3v0-window / llamacpp-kld` emits a **translation report** when the tensors are not local | a sealed dataset requires the bytes: `capture_content_digest` and `checksums.txt` cannot be computed from metadata. Saying so is better than inventing a digest. `adapt --source malaiwah-serving-v2` emits a full sealed dataset, because those tensors ARE local. |
| 4 | `capture` builds the dataset from a live capture tree | `capture` runs the pre-flight refusals, execs the wrapped scorer, and `--dry-run` works end to end; **assembling from a live tree is the one path this machine cannot exercise** | no GPU. The builders (`dsmanifest.DatasetWriter`) and the tree reader (`dsadapt.adapt_serving_v2`) are both exercised against real published tensors, so the untested seam is narrow and named. |
| 5 | `validate` writes `validation/structural-validation.json` into the dataset | it is written by `capture`/`adapt` **before** the seal; `validate` writes to `--json OUT` | a post-seal write is an `unlisted_file` refusal, and resealing would change `dataset_sha256` and break the external anchor (addendum A-2). |

### Test counts, as run

```
bin/selftest_fidelity_dataset.py   66 passed, 0 failed   F1-F15 P1-P9 H1-H11 L1-L5 C1-C4 X1-X2 I1-I15 R1 R4
bin/selftest_fidelity_compare.py   16 passed, 0 failed   N1-N11 (+N3b, N6b, N6c, N8b, N11b)
bin/selftest_fidelity_card.py      14 passed, 0 failed   K1-K13 (+K8b), live Hub axis included
bin/selftest_all.sh                38 passed, 0 failed, 3 skipped
cd registry && make check          62 passed, 0 failed
```

### The open items in §9, resolved

| # | resolution |
|---|---|
| 1 | `quant_pipeline` is still not installed; nothing built here imports `glm53_logits`. The comparator reads safetensors directly with a 30-line stdlib reader, so `--pipeline-root` was not needed. |
| 2 | Synthetic panels for every fixture, as planned. The remap path (REMAP-1..3) is exercised with a synthetic sealed receipt whose `artifacts[]` carry `/workspace/...` absolute paths (case P9). |
| 3 | Content is normative; both digests are carried; H11 makes cross-convention comparison a refusal. The adapter recomputes `aa21c427…` from the published head and cross-checks `47eaf729…` against `head-extraction.json` (case I12). |
| 4 | `hidden_states` is normative; the serving-v2 adapter accepts `hidden` and rewrites it, recording the rewrite in `inferred_fields`. |
| 5 | Downgraded to `run_mean_equality_only` and said so, as planned (case I5). `--recompute-content-digests` is the documented paid upgrade. |
| 6 | `bin/selftest_all.sh` was **not** claimed exclusively — the measurement workflow added a `joint standard` block to it during this work and the three new lines were applied on top without conflict. |
| 7 | Still open, and the only thing blocking the Ship phase: live Hub *rendering* needs one push to a private scratch model repo. Push-time validation and the library round-trip are both proven clean. |
