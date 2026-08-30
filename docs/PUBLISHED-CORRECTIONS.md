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

## Not published, deliberately

The per-domain confidence intervals in `registry/data/measurements.jsonl` are
**undercovering** — they claim 95% and measure 78–83% — and correcting them moves 79
published endpoints by up to 24%. That is a change to numbers, not to metadata, so it is
written up for an operator decision in `docs/REVIEW-DEFERRED.md` rather than pushed.
