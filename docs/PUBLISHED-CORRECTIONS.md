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
every measurement run (`k6/tools/tr3_surface.py::verify_seal`, reached from
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

<!--STAT01-SECTION-->

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
