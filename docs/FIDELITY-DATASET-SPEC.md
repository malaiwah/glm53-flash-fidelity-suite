# Fidelity dataset format — `malaiwah.fidelity-dataset.v1`

**Status:** v1, frozen for implementation. **Format id:** `malaiwah.fidelity-dataset.v1`.
**Schema:** [`schema/fidelity-dataset.schema.json`](schema/fidelity-dataset.schema.json).
**Comparison receipt:** [`schema/fidelity-comparison-receipt.schema.json`](schema/fidelity-comparison-receipt.schema.json).
**Card annotation:** [`CARD-ANNOTATION-SPEC.md`](CARD-ANNOTATION-SPEC.md).
**Build plan:** [`FIDELITY-DATASET-BUILD-PLAN.md`](FIDELITY-DATASET-BUILD-PLAN.md).

---

## 0. Why this exists

Today capture and comparison are **fused**. `k6/tools/stream_score.py` runs a model over a panel and
`k6/tools/k6_kld_report.py` scores it against a teacher, and the only durable output is a *number*
plus receipts pointing at filesystem paths. Three consequences, all of which have already bitten us:

1. **Every measurement re-pays for capture.** Scoring quant N against the BF16 reference re-runs the
   reference, or depends on a teacher tree somebody is still holding.
2. **Teachers are non-portable.** `capture-receipt.json`'s `logit_files[].path` are absolute paths on
   the capture box. `materialization-receipt.json`'s `packed_root` is
   `/home/jl_fs/glm53-k6/out-k6`.
3. **A lost capture kills reproducibility.** That JarvisLabs filesystem was destroyed after being
   wrongly declared redundant. The sealed `layers/*.json` and `experts/*.json` receipt trees existed
   nowhere else. The published K6/K8 checkpoints are still self-contained *for serving*
   (`exl3-mcg-storage-abi.json` present, payloads inline, readable via `stream_score --source
   exl3hf`), but the `--source checkpoint` and `--source payload-store` reading paths are now
   **unreachable from public artifacts**, and the published materialization receipt still names the
   dead path. The registry already has a field for exactly this condition:
   `reference.logits_available`, documented as *"false means a number against this reference can
   never be re-derived, only re-run."*

Splitting capture from comparison fixes all three:

| | fused (today) | separated (this spec) |
|---|---|---|
| root reference | re-run per measurement | captured **once**, published, downloaded |
| quant capture | discarded | **publishable standalone** — a quant author contributes a capture with no access to our infrastructure |
| lost machine | reference dies | dataset survives; `logits_available` stays true |
| same-lane floor | a separate cross-stack measurement with a 1e-2-class residual | when A and B are captured **on the same lane**, the floor collapses toward 0 and what remains is quantization error |

The last row is the important one. Our published cross-stack floor
(`measurement--glm53.bf16-replay-floor.brandonmusic-final25`) is **0.012712 nats** — comparable in
magnitude to K6's entire 0.013723. That floor is comparison overhead, not quantization. Two captures
made on one lane and compared offline in fp64 remove it structurally rather than by subtraction,
which the registry forbids across lanes anyway (invariant **BIAS-006**).

### Three steps, one tool, three modes

```
step 1   capture   reference (root) weights + panel  ->  fidelity dataset A     [publish: REQUIRED for a root]
step 2   capture   quantized weights  + panel        ->  fidelity dataset B     [publish: OPTIONAL]
step 3   compare   A, B                              ->  KLD + determinism + registry-submittable receipt
                   A, A                              ->  reproduction confirmation, exactly 0.0
```

Step 2 must be publishable **without** step 3 having been run, and step 3 must run with **neither**
set of weights present. Both are hard requirements on the format, and both are things the kimi-k3
artifact (the only serious prior art) cannot do.

---

## 1. Scope, conformance, stability

### 1.1 What a conformant dataset is

A **fidelity dataset** is a directory (or an HF dataset repository at a pinned revision) that holds:

* a **panel binding** — which token sequences were scored, by digest;
* a **capture** — one tensor file per panel record, in **hidden form** or **logit form**;
* a **head identity** — the digest of the `lm_head` weight that this capture's own forward held;
* a **capture runtime** — lane, stack fingerprint, container, capture-code digests;
* **determinism evidence** over tensor content;
* a **seal** — a self-covering digest chain rooted in one publishable value.

It says nothing about *another* model. It is a measurement of one artifact against a panel, not a
comparison. Comparison is step 3 and produces a different object (§10).

### 1.2 Conformance levels

| level | meaning | who claims it |
|---|---|---|
| **structural** | passes `bin/fidelity-dataset validate`: schema, seal chain, path rules, digest consistency | any dataset |
| **sealed** | structural **and** every tensor's `tensor_content_sha256` recomputed and matched | `validate --verify-tensors` |
| **qualified** | sealed **and** carries a `validation/replay-qualification.json` (hidden form) or a determinism receipt with `run_count >= 2` and one distinct content digest | opt-in, separate receipt |

These are deliberately **three fields, not one**. Festr's validator hard-refuses any artifact whose
`status != "qualified"`, which conflates *structurally valid* with *scientifically accepted* and
makes a partial or in-progress capture unrepresentable. Here `structural_status` is the validator's
verdict and `qualification` is a separate, optional receipt.

### 1.3 Stability guarantees for v1

* **Additive only.** v1 readers MUST ignore unknown keys. v1.x may add optional keys; it may never
  add a required key, remove a key, change a key's type, or change a digest preimage.
* **Digest preimages are frozen.** The five preimages in §5.1 are part of the format's identity. A
  change to any of them is v2.
* **`format_version: 1`** is an integer and never changes within v1. The `schema` string
  `malaiwah.fidelity-dataset.v1` is the dispatch key; tools MUST dispatch on the exact string and
  refuse unknown ones rather than guess (the `registry_add.py` rule).
* **Deprecation:** a v1 key may be marked deprecated in a later minor release and MUST keep working
  for the life of v1.

---

## 2. On-disk layout

Paths are relative to the dataset root. `NNNN` is the zero-padded **panel record index**, not a
sequence number within the capture: a shard holding records 512–1023 names its files
`hidden_0512.safetensors` … `hidden_1023.safetensors`.

```
<root>/
  fidelity-dataset.json                 REQUIRED  the manifest, self-sealed (§5.3)
  checksums.txt                         REQUIRED  sha256␠␠relpath over every file except itself
                                                  and fidelity-dataset.json (§5.4)
  README.md                             REQUIRED  dataset card carrying x_fidelity (CARD-ANNOTATION-SPEC §4)

  panel/
    panel.json                          REQUIRED  the panel binding (§7)
    tokens/context-NNNN.json            REQUIRED  token id array per record, compact JSON
    masks/context-NNNN.npy              REQUIRED when any record is padded / variable length
    panel-receipt.json                  OPTIONAL  the upstream sealed panel receipt, byte-verbatim
    panel-remap.json                    REQUIRED when panel-receipt.json carries absolute paths (§7.4)

  capture/
    manifest.json                       REQUIRED  per-record tensor manifest (§6)
    hidden_NNNN.safetensors             hidden form  key "hidden_states", bf16, [scored_rows, hidden_width]
    logits_NNNN.safetensors             logit  form  key "logits",        fp32/bf16, [scored_rows, vocab_size]

  head/
    head.json                           REQUIRED (hidden form); RECOMMENDED (logit form)  (§8)
    weight.safetensors                  OPTIONAL  the head payload, tensor key "weight"
    final_norm.safetensors              OPTIONAL  informational; NEVER applied at replay (§8.5)

  runtime/
    capture-runtime.json                REQUIRED  lane + stack fingerprint + container + code digests (§9)
    pip-freeze.txt                      OPTIONAL  preimage of stack_fingerprint.pip_freeze_sha256
    engine-log.txt                      OPTIONAL

  determinism/
    determinism.json                    REQUIRED when manifest.determinism.run_count > 1
    repeat-01/manifest.json             OPTIONAL  additional captures of the SAME weights on the SAME lane
    repeat-01/hidden_NNNN.safetensors
    repeat-02/...

  validation/
    structural-validation.json          written by `validate`; MUST be present in a published dataset
    replay-qualification.json           OPTIONAL  required to claim hidden-replay fidelity (§8.6)
    contamination.json                  OPTIONAL  panel/benchmark overlap receipt

  compat/                               OPTIONAL  emitted by `capture --emit-k3-compat` (§12.3)
    suite-manifest.json
    reference-hidden/manifest.json
    lm-head/manifest.json
    lm-head/weight.safetensors          OPTIONAL  alias file whose single tensor key is "weight"

  upstream/                             OPTIONAL  verbatim copies of producing receipts (§9.4)
    capture-receipt.json
    backend.json
    plan.json
    reader-identity.json
    hidden-capture.json
```

### 2.1 Path rules (mechanical, validator-enforced)

**PATH-1** Every path-valued string anywhere in any manifest is **relative** and, after
`os.path.normpath`, resolves **inside** the dataset root. Absolute paths are a hard error.

**PATH-2** No manifest field may name a host-local directory, even informationally. Fields known to
carry them upstream (`packed_root`, `output_root`, `capture_chunk_dir`, `logit_files[].path`) MUST be
stripped when a receipt is copied into `upstream/`, and the stripping MUST be recorded as
`upstream_receipts[].stripped_fields[]`. This is the rule our data loss taught, made mechanical.
Festr encodes the same rule as `raw_chunks_retained: false`; we adopt that field name (§6.2) and add
the general form.

**PATH-3** `..` is permitted only inside `compat/`, and only where it still resolves inside the root
(`compat/reference-hidden/manifest.json` names `../../capture/hidden_0000.safetensors`).

**PATH-4** No symlinks. `checksums.txt` covers regular files only; a symlink is a hard error.

---

## 3. Root vs quant: the required/optional matrix

`dataset.role` is one of `root`, `quant`, `derived`.

* **root** — a capture of *reference* (unquantized, or vendor-released) weights. It is the shared
  yardstick. Its dataset is a public good.
* **quant** — a capture of quantized weights.
* **derived** — a capture produced by replaying or transforming another capture rather than by
  running weights (e.g. hidden→logit materialization). Always `class: advisory`.

| block / file | root | quant | derived | note |
|---|---|---|---|---|
| `fidelity-dataset.json` | **R** | **R** | **R** | |
| `checksums.txt` | **R** | **R** | **R** | |
| `README.md` with `x_fidelity` | **R** | **R** | **R** | role = `fidelity-dataset` |
| `panel/panel.json` | **R** | **R** | **R** | |
| `panel/tokens/` | **R** | **R** | **R** | a panel referenced but not shipped is not a binding |
| `panel/masks/` | R *if padded* | R *if padded* | R *if padded* | |
| `capture/manifest.json` + tensors | **R** | **R** | **R** | |
| `head/head.json` | **R** | **R** | **R** | digests non-null: see HEAD-4 |
| `head/weight.safetensors` | **R** | O | O | a root that ships no head cannot be replayed against |
| `runtime/capture-runtime.json` | **R** | **R** | **R** | |
| `determinism` block | **R** | **R** | **R** | `run_count >= 1`; `>= 2` REQUIRED for `qualified` |
| `weights` block | **R** | **R** | **R** | repo + revision + config digest |
| `scope` block | **R** (all-native) | **R** | **R** | registry `scope_digest` recipe |
| `base_capture` block | **must be null** | O, RECOMMENDED | **R** | which root this is meant to be compared to |
| `validation/structural-validation.json` | **R** | **R** | **R** | |
| `validation/replay-qualification.json` | O | O | O | required to *claim* replay fidelity |
| `lossy_codec` | must be `null` | may be non-null | may be non-null | non-null ⇒ advisory |
| registry `reference` record | **SHOULD** exist, `logits_available: true` | not applicable | not applicable | |

**R** = required, **O** = optional.

Two rules that are not obvious:

* **ROOT-1** A `root` dataset MUST declare `scope.policy = "native"` with every tensor class
  `native`, and `head.quantized = false`. A dataset whose weights are quantized is a `quant`
  dataset even if its author calls it a reference. (`reference_kind = dequantized_from_quant`
  exists in the registry precisely so this stays honest — REFC-001.)
* **QUANT-1** A `quant` dataset's `base_capture` is **optional**, because a capture must be
  publishable before its comparison partner exists. When present it names the intended root by
  `dataset_sha256` and/or `repository@revision`, and the comparator warns (never refuses) if the
  actual A differs.

---

## 4. Form: hidden vs logit

### 4.1 Storage arithmetic (why hidden is the default)

Measured against our real published files, for GLM-5.3-Flash (`vocab 154,880`, `hidden 4,096`):

| | per scored position | 25-window panel (51,175 pos) | 10.48M-position suite |
|---|---|---|---|
| fp32 logits | 619,520 B (605 KiB) | **31.70 GB** | **6.49 TB** |
| bf16 hidden | 8,192 B (8 KiB) | **419 MB** | **85.9 GB** |

Ratio **75.6×**. This is not theoretical: `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` paid
**811,621,019,136 bytes** for fp32 logits on a 640-window panel, and our
`malaiwah/GLM-5.3-Flash-fidelity-suite-v1` publishes hiddens for the same class of panel in a
fraction of that.

**Hidden form is the DEFAULT and RECOMMENDED form.** Logit form remains fully expressible because
some stacks do not have a separable head.

### 4.2 The two forms

| field | hidden form | logit form |
|---|---|---|
| `capture.form` | `"hidden"` | `"logit"` |
| `capture.semantic_point` | `"after_final_rmsnorm_before_lm_head"` | `"lm_head_output_before_sampling"` or `"live_lm_head_output_before_sampling"` |
| `capture.tensor_key` | `"hidden_states"` | `"logits"` |
| `capture.dtype` | model residual dtype, normally `"BF16"` | `"F32"` or `"BF16"` |
| tensor shape | `[scored_rows, hidden_width]` | `[scored_rows, vocab_size]` |
| `capture.hidden_width` | REQUIRED | must be null |
| `capture.vocab_size` | REQUIRED (for the head) | REQUIRED |
| `capture.head_separable` | must be `true` | `true` or `false` + `head_not_separable_reason` |
| `head.applied_in_capture` | must be `false` | must be `true` |

**FORM-1** `capture.dtype_lossless` is REQUIRED and is `true` only when the stored dtype is not
narrower than the dtype the value had in the forward pass. GLM-5.3-Flash's post-norm residual is
natively `torch.bfloat16`, so a bf16 hidden capture is lossless and the field is `true`. An fp32
logit capture of a bf16 forward is also `true` (widening). A bf16 capture of an fp32 logit is
`false`, and a `false` here forces `class: advisory` at compare time.

**FORM-2** `semantic_point` is REQUIRED and has no default. Our currently published capture manifest
(`glm53flash-fidelity-capture/2`) declares no cut point at all and ships `final_norm.safetensors`
next to the head, which actively implies a norm+head replay it does not want. That ambiguity is a
defect this field removes; see §8.5.

**FORM-3** The value `"after_final_rmsnorm_before_lm_head"` is **adopted verbatim from kimi-k3**. Our
capture is at exactly that cut — verified in code, not documentation: `tools/fidelity.py`
`_rpc_install_hook` registers a *post*-hook on `…language_model.norm` (module output), and the replay
path computes `hidden @ head.T` and never applies `final_norm`; `k6/tools/hidden_replay.py` takes the
lm_head module's *input* via a forward **pre**-hook, which is the same tensor.

---

## 5. Digests and the seal

### 5.1 The five preimages (frozen for v1)

| name | preimage | notes |
|---|---|---|
| `file_sha256` | sha256 of the whole file bytes | the container digest. What `checksums.txt` carries. **Never determinism evidence** (§11). |
| `payload_sha256` | sha256 of the safetensors data region: read `<Q` header length at offset 0, skip `8 + header_len`, hash the rest | survives `__metadata__` churn; implemented today in `k6/tools/hidden_replay.py::payload_sha256` and `k6/stage_k6.sh` L4 |
| `tensor_content_sha256` | sha256 of the raw little-endian bytes of the **named tensor only** | container-independent. bf16 is hashed via its `uint16` view. Implemented today in `k6/tools/hidden_replay.py::tensor_content_sha256` |
| `token_ids_json_sha256` | `sha256(json.dumps(ids, separators=(",",":")).encode("utf-8"))` | **compact separators — kimi-k3's preimage, ADOPTED.** Our historical preimage used default separators (`", "`); it is preserved as `token_ids_sha256_legacy`. |
| `suite_token_hash_sha256` | `sha256("\n".join(per_record_token_ids_json_sha256_hex).encode("ascii"))`, records in ascending index order | **newline join — kimi-k3's preimage, ADOPTED.** Our historical aggregate joined with `""`; preserved as `panel_token_sha256_legacy`. |

Same token ids produced *different* hashes under our old preimages at both levels. That is a
preimage divergence, not a naming one, and it is the one thing an adapter cannot paper over without
re-reading `tokens/`. v1 ends it by adopting Festr's.

### 5.2 `capture_content_digest` — the manifest-independent identity

```
capture_content_digest = sha256("\n".join(
    "%d:%s" % (record["index"], record["tensor_content_sha256"])
    for record in sorted(records, key=lambda r: r["index"])
).encode("ascii"))
```

This is the identity of *what was captured*, independent of every container, every manifest
serialization, and every piece of metadata. It is:

* the value compared for the **A == B self-compare short-circuit** (§10.4);
* the value that appears in `determinism.evidence_hashes` (§11);
* the value a card and a registry row cite.

**Why not the manifest's own digest.** Festr binds a comparison to a reference by
`reference.manifest_sha256` — a hash of a JSON *file*. In his own published artifact that binding has
**already drifted**: `manifest.json` and the sentinel receipts say the reference-hidden manifest is
`f0ea6a85…`, while `results/qsrt-k2-…/distribution-fidelity.json` says `66f1fe71…`. The candidate
comparison ran against a re-emitted manifest and his validator does not check `results/`, so it
passes. A digest over a re-serializable container is not a stable identity. The registry knows this
already: `determinism.evidence_kind` lists `receipt_file_sha256` and `container_or_archive_sha256`
only so contributors can record them honestly, and **blocks both** from supporting a determinism
claim (**DET-001**).

### 5.3 Manifest self-seal (the shipped method, reused)

`fidelity-dataset.json` carries `dataset_sha256`, computed exactly as
`bin/fidelity/common.py::seal` and `registry/tools/registry_lib.py` do it — the same four-line recipe
the registry documents to contributors, and the same one that seals
`reports/stack-provenance-retro.json`:

```python
body = dict(manifest); body["dataset_sha256"] = ""
dataset_sha256 = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
```

Verification is the same computation. Anyone can do it with `python3 -c` and no imports from us.

The stack fingerprint embedded at `runtime/capture-runtime.json → stack_fingerprint` keeps its own
`stack_fingerprint_sha256`, computed by `bin/fidelity/stackprint.py::fingerprint_sha256`, which
hashes the block **minus** `VOLATILE_KEYS = ("collected_utc", "paths", "stack_fingerprint_sha256")`
so two collections on an identical stack hash identically. That property is preserved verbatim; the
dataset seal does **not** re-hash it, it carries the value.

### 5.4 The seal chain (self-covering, tamper-evident, acyclic)

```
dataset_sha256                          self-blanked seal over the whole manifest
  ├── seal.checksums_sha256  ──────►  checksums.txt
  │                                     └── sha256 of EVERY published file except
  │                                         checksums.txt and fidelity-dataset.json
  ├── capture.capture_content_digest ─► every capture tensor, by CONTENT
  ├── panel.suite_token_hash_sha256  ─► every tokens/context-NNNN.json, by CONTENT
  ├── head.tensor_content_sha256     ─► head/weight.safetensors, by CONTENT
  └── runtime.capture_runtime_sha256 ─► runtime/capture-runtime.json (itself self-sealed)
```

There is no cycle: the manifest covers `checksums.txt` by digest, `checksums.txt` covers every other
file, and the manifest covers itself by self-blanking. **Every published byte is covered.** The
content digests are a second, container-independent path to the same bytes, so a re-serialized
container is detected as *changed container, unchanged content* rather than as corruption.

`checksums.txt` format is **adopted verbatim from kimi-k3**: one line per file,
`<64-hex><space><space><relpath>`, sorted by path, LF endings, `sha256sum --check`-compatible,
excluding itself. A reviewer with no tooling of ours verifies the payload with one coreutils command.

**The external anchor.** `dataset_sha256` is a single 64-hex value that MUST be published outside the
dataset: in the model card's `x_fidelity.fidelity_dataset.dataset_sha256`, and in any registry row
citing the dataset. Festr's `manifest.json` is unsigned and nothing outside his artifact commits to
it, so his only immutable root is the HF revision. Ours has both.

**SEAL-1** `verify` recomputes: (a) `dataset_sha256`; (b) `sha256(checksums.txt)` vs
`seal.checksums_sha256`; (c) every line of `checksums.txt`; (d) `capture_content_digest`,
`suite_token_hash_sha256`, `head.tensor_content_sha256` when `--verify-tensors`.
Any mismatch is exit 3 and the dataset is refused. There is no `--force`.

**SEAL-2** A dataset whose file set is a **superset** of `checksums.txt` (an extra file nobody
hashed) is refused with `unlisted_file`. A dataset whose file set is a **subset** is refused with
`missing_file` unless `--allow-partial`, which is only legal for capture tensors and their manifest
rows, never for a manifest, seal, panel, head or runtime file.

---

## 6. `capture/manifest.json`

### 6.1 Header

```json
{
  "schema": "malaiwah.fidelity-capture-manifest.v1",
  "format_version": 1,
  "receipt_sha256": "<self-blanked seal>",
  "created_utc": "2026-08-29T00:00:00Z",
  "run_name": "reference-bf16",
  "form": "hidden",
  "semantic_point": "after_final_rmsnorm_before_lm_head",
  "tensor_key": "hidden_states",
  "dtype": "BF16",
  "dtype_lossless": true,
  "hidden_width": 4096,
  "vocab_size": 154880,
  "context_length": 2048,
  "scored_rows_per_context": 2047,
  "total_scored_rows": 51175,
  "total_size_bytes": 419228000,
  "suite_token_hash_sha256": "<panel binding>",
  "capture_content_digest": "<§5.2>",
  "runtime_manifest": "../runtime/capture-runtime.json",
  "runtime_manifest_sha256": "<file digest of that file>",
  "records": [ ... ]
}
```

`runtime_manifest` + `runtime_manifest_sha256` as a **relative path plus digest** is adopted from
kimi-k3, including his validator's rule that an absolute path is refused.

### 6.2 Record

Exactly these keys. Unknown keys are permitted (additive rule) but MUST NOT be load-bearing.

```json
{
  "index": 0,
  "context_index": 0,
  "window_index": 0,
  "window_id": "final-0000",
  "file": "hidden_0000.safetensors",
  "key": "hidden_states",
  "dtype": "BF16",
  "shape": [2047, 4096],
  "size_bytes": 16769120,
  "sha256": "<file_sha256>",
  "payload_sha256": "<§5.1>",
  "tensor_content_sha256": "<§5.1>",
  "token_ids_json_sha256": "<compact preimage>",
  "token_ids_sha256_legacy": "<our historical preimage, or null>",
  "attention_mask_sha256": "<or null>",
  "prediction_positions": 2047,
  "scored_rows": 2047,
  "role": "final",
  "domain": "axis1_general",
  "document_id": "…",
  "allocation_stratum": null,
  "semantic_class": null,
  "source_cluster_id": null,
  "elapsed_seconds": 41.2,
  "request_id": null,
  "raw_chunks_retained": false
}
```

**Three index aliases on purpose.** `index`, `context_index` and `window_index` all carry the same
integer. kimi-k3's comparator resolves a record index by trying `context_index`, then
`window_index`, then `index`, in that order — a fallback he wrote for his own 32×2048 predecessor
which happens to accept our window-based panel. Emitting all three costs a few bytes per record and
makes our capture readable by his unmodified comparator.

**REC-1** `index` values are unique and `>= 0`. Duplicates are a hard error (his validator's rule).
**REC-2** `key` equals the header `tensor_key` exactly. Festr's comparator hard-codes
`"hidden_states"` / `"logits"` and refuses anything else; our own
`k6/tools/hidden_replay.py` currently writes `"hidden"`, which would be rejected on a one-word
difference. **v1 normative key is `"hidden_states"`**; a reader MUST accept `"hidden"` from a
pre-v1 artifact and MUST rewrite it on ingest, recording the rewrite as a disclosure.
**REC-3** `shape[0] == scored_rows`, and for hidden form `shape[1] == hidden_width`, for logit form
`shape[1] == vocab_size`.
**REC-4** `raw_chunks_retained` is `false` in every published dataset, and its being `false` means
the host-local chunk keys were **stripped**, not merely absent (PATH-2).
**REC-5** `attention_mask_sha256` is REQUIRED (non-null) whenever the panel ships masks. Our packed
and streaming lanes vary mask construction; Festr's one-request-per-context path makes it invariant
and he has no equivalent field. The name is adopted from the `quant-pipeline` lineage
(`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` `logit_files[]`), which is our own pipeline's
output published by a third party.

### 6.3 Coverage

```json
"coverage": {
  "declared_records": 5120,
  "present_records": 512,
  "complete": false,
  "index_range": [0, 511],
  "shard_of": {"index": 0, "total": 10, "stride": 1},
  "missing_indices_sample": [512, 513, 514]
}
```

**COV-1** `complete` is `true` iff `present_records == declared_records` and the present index set is
exactly `range(declared_records)`.
**COV-2** `complete: false` with `shard_of: null` requires a `subset_detail` string.
**COV-3** `verify` refuses an incomplete dataset unless `--allow-partial`; `compare` on an incomplete
dataset stamps `measurement_scope.covers_full_panel = false` and a `subset_of_panel` disclosure
(registry **SCOPE-010**).

This block exists because of **our own** published defect: `reference-bf16-shard0/`'s
`capture-manifest-full.json` declares `expected_contexts: 5120`, `contexts: 5120`,
`captures: [5120 entries]`, `complete: true` — in a repository that holds indices 0–511 and nothing
else. Nothing machine-readable says "shard 0 of 10". Festr's validator hard-codes `!= 1024 → raise`,
which catches his case and generalizes to nobody's.

---

## 7. Panel binding

### 7.1 The panel is a separate object

**PANEL-D1** The panel is referenced by id and bound by digest; it is not *owned* by the capture.
A capture carries a copy of the tokens so it is self-verifying, and names the panel by
`panel_id` + `suite_token_hash_sha256` + `repository@revision` so two captures can be proven to be on
the same panel without either owning it.

This is the hinge on which "dataset B is publishable standalone" turns. Festr embeds the panel inside
the *reference* artifact, which is exactly why his candidate captures cannot be published: a
candidate has no artifact to live in, so only the compare receipt survives. (His
`results/qsrt-k2-…/distribution-fidelity.json` cites `candidate.manifest_sha256 = e491330f…`, a
manifest that exists nowhere in his 2,418-file repository.) The registry already models panels as
first-class records with their own ids and `derived_from` lineage
(`registry/schema/panel.schema.json`), so this costs us nothing.

### 7.2 `panel/panel.json`

```json
{
  "schema": "malaiwah.fidelity-panel-binding.v1",
  "format_version": 1,
  "receipt_sha256": "<self-blanked seal>",
  "panel_id": "panel--glm53.brandonmusic.final25",
  "name": "brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows",
  "suite_token_hash_sha256": "<§5.1 aggregate, newline join>",
  "panel_token_sha256_legacy": "6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff",
  "token_digest_algorithm": {
    "per_record": "sha256(json.dumps(ids, separators=(',',':')).encode('utf-8'))",
    "aggregate": "sha256('\\n'.join(per_record_hex).encode('ascii'))",
    "legacy_per_record": "sha256(json.dumps(ids).encode('utf-8'))",
    "legacy_aggregate": "sha256(''.join(per_record_hex).encode('utf-8'))"
  },
  "repository": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
  "revision": "95f4fdd94bf29989db2e0d1054e4931f55edb6aa",
  "panel_receipt_sha256": "0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5",
  "contexts": 25,
  "context_length": 2048,
  "positions_per_context": 2047,
  "scored_positions_total": 51175,
  "scoring_window": {
    "score_from": 0,
    "windowed": false,
    "min_left_context_tokens": 1,
    "dropped_positions_total": 0,
    "policy": "every prediction position of every window is scored; nothing is dropped"
  },
  "tokenizer": {
    "id": "glm-5.3-flash",
    "repository": "zai-org/GLM-5.3-Flash-BF16",
    "revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
    "vocab_size": 154880,
    "add_special_tokens": false,
    "chat_template_applied": false
  },
  "contamination": {
    "checked": false,
    "method": "role separation only; no lexical or n-gram scan published",
    "hits": null,
    "receipt": null
  },
  "strata": {"axis1_general": {"contexts": 7}, "axis2_legal": {"contexts": 6},
             "axis3_code_agentic": {"contexts": 6}, "axis4_reasoning_termination": {"contexts": 6}},
  "records": [
    {
      "index": 0, "context_index": 0, "window_index": 0,
      "window_id": "final-0000",
      "token_file": "tokens/context-0000.json",
      "token_ids_json_sha256": "…",
      "token_ids_sha256_legacy": "338027e62f41540f73e38c6f9b4b9a06a50196cbd38cd9c69f11886af9d3cf9f",
      "token_ids_first16": [1, 2, 3, "…"],
      "token_ids_last16": ["…"],
      "num_tokens": 2048,
      "prediction_positions": 2047,
      "attention_mask_file": "masks/context-0000.npy",
      "attention_mask_sha256": "…",
      "role": "final", "domain": "axis1_general", "document_id": "…",
      "allocation_stratum": null, "semantic_class": null, "source_cluster_id": null,
      "partition": null, "sentinel": false
    }
  ]
}
```

**PANEL-D2** `panel_receipt_sha256` is **traceability only**. It is never reused as a token identity
and never appears in `determinism.evidence_hashes` — registry invariant **PANEL-002**, restated here
so the dataset validator enforces it too.

**PANEL-D3** `scoring_window` is part of **panel identity**, not a comparator flag. Changing
`score_from` from 0 to 1024 produces a *different panel with `derived_from`*, never a variant field
(registry **PANEL-005**, **PANEL-009**). This is what makes a llama.cpp-geometry number
(`score_from = n_ctx/2`) structurally incomparable to a Festr-geometry number (`score_from = 0`)
instead of silently comparable. Both are expressible; neither is comparable to the other.

**PANEL-D4** `token_ids_first16` / `token_ids_last16` are adopted from Festr's 32×2048 suite: a
costless eyeball check that does not require downloading `tokens/`.

**PANEL-D5 — the `contexts` name collision. KNOWN DEFECT, v1-frozen.**
`panel.json.contexts` is an **integer record count**. In the one format v1 claims compatibility with,
`contexts` is the **record list**. Same key, same nesting level, incompatible types. Consequences,
all measured rather than predicted:

* Handing `panel/panel.json` to Festr's comparator as `--suite-manifest` fails with
  `TypeError: 'int' object is not iterable`, from `suite.get("contexts", suite.get("windows"))`.
* Any reader that sniffs a manifest by probing for `contexts` — a reasonable thing to write, given
  his artifact is the only prior art — mis-types our panel.
* The mitigation is structural, not cosmetic: `--emit-k3-compat` (§12.3) must write a **separate**
  `compat/suite-manifest.json` in which `contexts` is the list. The two spellings can never coexist
  in one file.

This should have been `context_count`, which is unambiguous and collides with nothing. It is **not**
being renamed: §1.3 freezes key types for the life of v1, and no key has ever been more likely to be
read by a tool written against the other format. It is called out here, in the JSON Schema
description, and in §12.3 so that nobody has to discover it the way we did. **v2 renames it to
`context_count` and keeps `contexts` as a deprecated integer alias.** Anyone building on v1 today
should treat "is `contexts` an int or a list?" as the format-discrimination test between the two
lineages — it is, unfortunately, a reliable one.

### 7.3 Panel record ↔ capture record binding

**BIND-1** Every capture record's `index` has a panel record with the same `index`.
**BIND-2** `capture.records[i].token_ids_json_sha256 == panel.records[i].token_ids_json_sha256`.
**BIND-3** `capture.records[i].attention_mask_sha256 == panel.records[i].attention_mask_sha256` when
both are non-null.
**BIND-4** `capture.records[i].scored_rows == panel.records[i].prediction_positions`.
**BIND-5** `sum(prediction_positions) == scored_positions_total` when `coverage.complete`.
**BIND-6** `panel.suite_token_hash_sha256` recomputes from `panel.records[].token_ids_json_sha256`,
and each of those recomputes from `panel/tokens/context-NNNN.json`.

### 7.4 `panel/panel-remap.json` — the seal-preserving path fix

Upstream sealed receipts carry absolute paths (`quant-pipeline.glm53-token-panel-receipt.v1`'s
`artifacts[]` rows are `{path, bytes, sha256}` with paths like
`/workspace/artifacts/dataset/calibration/panel-v1/...`). Rewriting them breaks their seal;
shipping them unmodified breaks portability. The only correct answer is a **sidecar remap keyed by
digest**, which is exactly what `k6_kld_report._resolve_teacher_paths` already does for logits:
sealed absolute path first, `<root>/logits/<basename>` fallback **with sha256 verified before use**.

```json
{
  "schema": "malaiwah.fidelity-path-remap.v1",
  "receipt_sha256": "…",
  "for_receipt_sha256": "0beec577…",
  "for_receipt_file": "panel-receipt.json",
  "resolution_rule": "resolve by sha256, never by path; a fallback path is used only after its content digest matches the sealed row",
  "entries": {
    "<sha256 from the sealed receipt>": "tokens/context-0000.json"
  }
}
```

**REMAP-1** Every `artifacts[].sha256` in the copied sealed receipt appears exactly once as a key.
**REMAP-2** The remapped file's recomputed sha256 equals the key. A remap entry whose target does not
hash to its key is a hard error.
**REMAP-3** The sealed receipt in `upstream/` or `panel/panel-receipt.json` is byte-verbatim; its
`receipt_sha256` still verifies.

---

## 8. Head identity — the head trap, made structural

### 8.1 The trap

"Shared head" means shared **application**, not shared **weights**. If a quant quantizes its own
`lm_head`, replaying its hidden states through the **reference** head erases its head-quantization
error and flatters it.

This is not hypothetical. Verified per artifact:

| artifact | `lm_head` in the weights | what its forward actually held |
|---|---|---|
| our TR3 K6 / K8 | native BF16, `quantization_config.head_bits: 16`, plain tensor in the index (`{"dtype":"BF16","shape":[154880,4096]}`, verified by range-reading the published shard header) | the reference head |
| zai FP8 | not converted (`modules_to_not_convert`) | bitwise-equal to the BF16 head |
| stock EXL3 (turboderp) | **quantized**, `head_bits` 6–8 | its **own dequantized** head |
| GGUF | `output.weight` quantized | its own head |
| MLX | `lm_head.weight` quantized | its own head |
| NVFP4 | explicitly not quantized, BF16 in-repo | its own (native-equivalent) head |

kimi-k3's format **cannot express this**. His comparator takes one `--lm-head` path, loads one
`head_weight`, and applies it to *both* `DistributionSource`s. Neither manifest records which head
that capture's own weights imply; nothing refuses, warns, or records a mismatch. His own case is
legitimate (QSRT K2 quantizes routed experts only), but a stock-EXL3 candidate with `head_bits 6`
scored that way would have its head error erased silently.

Our own code has the same default live: `tools/fidelity.py cmd_replay` takes a **required** `--head`
and an **optional** `--candidate-head` that defaults to `None`, silently pushing both sides through
the reference head.

### 8.2 `head/head.json` and the manifest `head` block

Every capture — reference and candidate alike — declares its own head.

```json
{
  "schema": "malaiwah.fidelity-head-identity.v1",
  "receipt_sha256": "…",
  "present": true,
  "file": "weight.safetensors",
  "tensor_key": "lm_head.weight",
  "compat_tensor_key": "weight",
  "shape": [154880, 4096],
  "dtype": "BF16",
  "bias": null,
  "file_sha256": "47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0",
  "raw_tensor_sha256": "aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4",
  "tensor_content_sha256": "aa21c427970f64edd82669db3a8fb46613084e8bc271a3728784a52eb3f25ab4",
  "quantized": false,
  "bits": 16,
  "source": "native",
  "applied_in_capture": false,
  "final_norm": {
    "file": "final_norm.safetensors",
    "tensor_key": "model.language_model.norm.weight",
    "shape": [4096], "dtype": "BF16",
    "file_sha256": "c228a123dee3062c3ad0129094e9d98a264e33087ee88d79c8d6c5a6e60f2fed",
    "tensor_content_sha256": "…",
    "applied_in_capture": true,
    "applied_at_replay": false
  },
  "equality_receipt": {
    "schema": "glm53flash-head-equality/1",
    "file": "../validation/head-equality.json",
    "sha256": "…",
    "claim": "head_equal and final_norm_equal against zai-org/GLM-5.3-Flash-BF16"
  }
}
```

**HEAD-IDENT** The **normative** head identity is `tensor_content_sha256`.
`raw_tensor_sha256` is a REQUIRED alias carrying the identical value — it is Festr's field name for a
tensor-content digest (his `manifest.json → lm_head.raw_tensor_sha256`), and emitting it makes our
head readable by his tooling. `file_sha256` is a container digest and is **never** the identity.

This resolves a live inconsistency in our own published receipts: `head-extraction.json` and
`head-equality-fp8.json` record the **file** digest `47eaf729…`, while
`k6/hidden-replay-evidence/nonrouted-sparse-fetch.json` records the **tensor content** digest
`aa21c427…` for the same weight. v1 requires both, names content as normative, and forbids comparing
one to the other.

`source` is one of:

| value | meaning |
|---|---|
| `native` | the artifact ships the head unquantized; the forward used it as-is |
| `artifact_dequantized` | the artifact's head is quantized and was dequantized into the materialized view the forward loaded (stock EXL3, GGUF, MLX) |
| `shared_reference_head` | this capture did not have a head of its own; a named external head was applied |
| `unknown` | not established — forces advisory |

### 8.3 Comparator refusal rules

Numbered so the implementation and the test matrix can name them.

| id | condition | action |
|---|---|---|
| **HEAD-1a** | either side hidden-form, `A.head.tensor_content_sha256 == B.head.tensor_content_sha256`, and that head is the one applied | **ALLOW**. `estimator.head_policy = "shared_reference_head"`. Disclosure `shared_reference_head`, severity `info`. `class` may remain `strict`. This is the zai BF16/FP8 case our head-equality receipt licenses, and the precondition **REFC-003** checks. |
| **HEAD-1b** | either side hidden-form, head content digests **differ** | **REFUSE**, exit 3. Override `--disclose-head-substitution --head <path>` emits with `head_policy = "shared_reference_head"`, `class = "advisory"`, bias `{kind: "other", direction: "downward"}`, and disclosure `head_substituted` severity **`blocking`** — which under registry **DISC-003** forces `status ∈ {pending, retracted}`. A head-substituted number is not publishable as a measurement. |
| **HEAD-2** | both sides logit-form | **ALLOW**, never refuse: each capture already applied its own head, so head-quantization error is *inside* the measurement, which is correct. `head_policy = "native_head"`. Record both digests. A null digest ⇒ disclosure `estimator_unknown`, `class advisory`. |
| **HEAD-3** | mixed hidden ↔ logit | **REFUSE** unless the head replayed onto the hidden side has `tensor_content_sha256` equal to the head that produced the logit side. Then ALLOW with `head_policy = "native_head"` and a disclosure naming the replay. |
| **HEAD-4** | a hidden-form dataset with `head.tensor_content_sha256 == null` | **INVALID for cross-artifact comparison.** Validator: error. Comparator: exit 3, no override. A capture that cannot name its own head cannot be scored through anyone's. |
| **HEAD-5** | `form == "hidden"` and `head.applied_in_capture == true` | structural error: the declared cut is *before* the head. |
| **HEAD-6** | `head.present == false` on a `root` dataset | structural error (§3): a root nobody can replay against is not a yardstick. |

### 8.4 Why `shared_reference_head` and not a new enum value

`registry/schema/measurement.schema.json` already defines
`estimator.head_policy ∈ {native_head, shared_reference_head, dequantized_head, unknown}`, and
invariant **REFC-003** binds `reference.capture.head_source == shared_head_artifact` ⟺
`estimator.head_policy == shared_reference_head` **in both directions**. Any comparison that
substitutes a head therefore already fails registry validation if it does not say so. Refusing at
compare time is strictly better than failing months later at submission.

### 8.5 `final_norm` is shipped, never applied

Our published `head/final_norm.safetensors` sits next to the head and implies a norm+head replay it
does not want: the capture is **already after** the final norm. v1 makes this explicit with
`final_norm.applied_in_capture: true` / `applied_at_replay: false`, and:

**HEAD-7** A hidden-form dataset with `semantic_point == "after_final_rmsnorm_before_lm_head"` and
`final_norm.applied_at_replay == true` is a structural error. Replay applies the head **only**.

### 8.6 `validation/replay-qualification.json`

Optional, but REQUIRED to claim that a hidden-form capture reproduces the live logits.
Adopted from Festr's `hidden-replay-qualification.json`, with one field promoted to REQUIRED:

```json
{
  "schema": "malaiwah.fidelity-replay-qualification.v1",
  "receipt_sha256": "…",
  "canonical_comparator": {
    "device": "cuda:0",
    "tf32": false, "deterministic_algorithms": true,
    "bf16_reduced_precision_reduction": false,
    "cublas_workspace_config": ":4096:8",
    "two_pass_full_vocabulary": true,
    "vocab_chunk": 9680, "position_block": 128,
    "accumulation_dtype": "float64",
    "lm_head_tensor_content_sha256": "aa21c427…",
    "source_file_hashes_verified": true
  },
  "alternative_comparator": { "…same shape, different chunk sizes…" },
  "canonical_result": {
    "direction": "KL(live || replayed)",
    "kl": {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0, "max": 0.0},
    "top1_agreement": 1.0
  },
  "alternative_result": { "…" },
  "chunk_invariance_mean_kld_difference": 0.0,
  "bitwise_equal_to_live": true,
  "status": "qualified"
}
```

**QUAL-1** `comparator.device` is **REQUIRED**. Festr's headline replay figure of `1.229325e-6` comes
from a receipt whose comparator block says `"device": "cpu"`; his *GPU* replay is bitwise exact —
provable from his own artifact, because his validator forces the hidden-replay and live-logit
sentinel receipts to be equal on every key except `reference`/`candidate`/`comparator`, and the
published pair 00-vs-01 reports `kl mean = 0.0032166685936858316` on **both** paths, identical to
the last digit. Replay error is a property of the **replay device and kernel**, not of the artifact.
A replay-qualification receipt without its comparator device is meaningless, so v1 refuses one.

**QUAL-2** `chunk_invariance_mean_kld_difference` is REQUIRED: two vocab-chunk settings must agree.

---

## 9. Capture runtime and stack fingerprint

### 9.1 `runtime/capture-runtime.json`

```json
{
  "schema": "malaiwah.fidelity-capture-runtime.v1",
  "format_version": 1,
  "receipt_sha256": "<self-blanked seal>",
  "lane": "streaming",
  "lane_identity_sha256": "…",
  "lane_identity_inputs": ["torch_version","cuda_runtime_version","device_name","grouped_mm_kernel",
                           "numeric_policy","attention_backend","experts_implementation",
                           "parallelism","ep_emulate","reduce_order"],
  "stack_fingerprint": { "…malaiwah.stack-fingerprint.v1 verbatim…" },
  "stack_fingerprint_sha256": "…",
  "container": {
    "image_digest": "sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4",
    "image_reference": null,
    "image_repository_digest": null
  },
  "runtime_environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"},
  "source_files": {"k6/tools/stream_score.py": "022a167e…", "k6/tools/hidden_replay.py": "87940124…"},
  "capture_tool": {
    "file": "bin/fidelity_dataset.py", "sha256": "…",
    "wraps": ["k6/tools/hidden_replay.py", "k6/tools/stream_score.py"],
    "mechanism": "monkeypatch stream_score.build_streaming_model; forward pre-hook on model.get_output_embeddings()"
  },
  "weights": {
    "repository": "malaiwah/GLM-5.3-Flash-TR3-6bpw",
    "revision": "…",
    "model_revision": "a6c167b62691b2bac901344b65cb651a70f53e43",
    "config_sha256": "…", "index_sha256": "…",
    "checkpoint_identity_sha256": "a8668be3…",
    "runtime_reader_sha256": "1ccce446…",
    "backend_identity_sha256": "d19c049f…"
  },
  "upstream_receipts": [
    {"file": "../upstream/capture-receipt.json", "schema": "quant-pipeline.glm53-logit-capture.v1",
     "sha256": "…", "stripped_fields": ["logit_files[].path"]}
  ]
}
```

### 9.2 Three blocks adopted from kimi-k3

* **`container`** — his `capture-runtime.json` records `image_id`, `image_reference` **and**
  `image_repository_digest`. We already learned (the hard way, in `stackprint.py`) that
  `docker save`/`load` strips tags and the pin file is the only trustworthy source. Same field names.
* **`runtime_environment`** — a flat dict of literal env vars. Ours is `stack_fingerprint.env_pins`
  over a fixed `ENV_PIN_KEYS` tuple; we emit **both**, because a reader who knows only his format
  still gets the env.
* **`source_files: {repo-relative path → sha256}`** — pins the **capture code by content**, not by
  commit. This is the single best idea in his format and we have no equivalent. `stream_score.py`
  and `hidden_replay.py` and the capture front-end are all pinned here.

### 9.3 `lane`

`lane ∈ {sealed-ep8, streaming, local-mps, local-cuda-budget, other}` — the registry's own
`submission.schema.json` enum, unchanged. Adding a lane requires a registry schema change, which is
the point: lanes are not interchangeable, and a number from a non-sealed lane carries a **measured**
offset (see `pipeline--malaiwah.glm53-stream-packed-kld`'s `lane.bridge` block: `delta_mean_kld
-8.4958e-6`, `max_abs_per_window_delta 2.8735e-4`, `tokenwise_kld_sha256_matches_sealed: false`,
`publishable_as_reproduction: false`).

kimi-k3 has exactly one lane (vLLM serving) and therefore no concept of one. We have eight capture
surfaces (`checkpoint`, `payload-store`, `dione`, `native`, `exl3hf`, `mlx`, `gguf`, `nvfp4`) across
several lanes, so lane must be a first-class field the comparator gates on.

### 9.4 `upstream/`

Verbatim copies of the receipts that produced the capture. They keep their own seals and MUST still
verify. Any field stripped for PATH-2 is named in `stripped_fields[]`. Fields not derivable from
captures at all (K6's `native_copy_receipt_sha256`, `reader_audit_receipt_sha256`,
`checkpoint_receipt_sha256`) travel here as an opaque passthrough rather than being invented into the
manifest.

---

## 10. Comparison (step 3)

### 10.1 The gate ladder — ordered, each gate named

`compare A B` runs these in order and stops at the first refusal.

| # | gate | refusal id | override |
|---|---|---|---|
| 1 | seal: both datasets verify (§5) | `seal_failed` | none |
| 2 | form: hidden↔hidden, logit↔logit, or HEAD-3 | `form_mismatch` | none |
| 3 | panel: `suite_token_hash_sha256` equal; index sets equal; per-record `token_ids_json_sha256` equal; `attention_mask_sha256` equal when both present; `scored_rows` equal; `scoring_window` equal | `panel_mismatch` | none |
| 4 | head: HEAD-1..7 | `head_mismatch` | `--disclose-head-substitution` (HEAD-1b only) |
| 5 | lane: `lane` equal ⇒ `same_lane: true` | `lane_mismatch` | `--allow-cross-lane` ⇒ bias block + advisory |
| 6 | stack: `same_stack` iff `lane_identity_sha256` **and** `stack_fingerprint_sha256` both equal; else `cross_stack` ⇒ bias block REQUIRED (**BIAS-001**) | — | — |
| 7 | vocab/width: `vocab_size` equal; `hidden_width` equal (hidden form) | `geometry_mismatch` | none |
| 8 | coverage: index sets equal, else intersect | `coverage_mismatch` | `--allow-partial` ⇒ `covers_full_panel: false` + `subset_of_panel` |
| 9 | lossy: either `lossy_codec` non-null ⇒ advisory + `lossy_capture_codec` disclosure | — | — |
| 10 | self-compare short-circuit (§10.4) | — | — |
| 11 | compute (§10.2) | — | — |

**Cross-lane is not a floor.** A cross-lane comparison may never be cited as
`comparability.bias.floor_measurement_ref`: **BIAS-006** requires the floor row to come from a
pipeline declaring the **same** lane. The comparator stamps `usable_as_floor: false` on any
cross-lane receipt so a downstream tool cannot launder it.

### 10.2 Numerics

Fixed, not configurable except where noted:

* Full **vocabulary**, no truncation, no top-k.
* `torch.log_softmax` in **float64** (`estimator.accumulation_dtype = "float64"`), matching
  `k6/tools/k6_kld_report.py::_token_kld`. `float32` is expressible but lands the receipt in a
  *different comparability class* — `accumulation_dtype` is one of the seven comparability-key
  inputs, so an fp32 receipt and an fp64 receipt on identical data are correctly not comparable.
  (Festr's log-probs are fp32 with an fp64 reduction and a `clamp_min_(0)`; his receipts therefore
  belong to a different class than ours *on the same data*, and v1 makes that visible rather than
  silent.)
* Direction `KL(reference || candidate)`. The receipt carries **both** spellings:
  `direction: "reference_to_candidate"` (registry vocabulary) and
  `direction_label: "KL(reference || candidate)"` (Festr's literal string, which his receipt
  comparator refuses to read anything else in place of).
* Non-finite intermediate ⇒ hard refusal, never a clamp.
* Streamed in `--chunk-positions` slices; `--vocab-chunk` must divide `vocab_size` exactly.
  **For GLM-5.3-Flash, `vocab_size = 154,880` is NOT divisible by kimi-k3's default 10,240.**
  `154880 = 2^7 × 5 × 11 × 22`; working values include **9,680** (154880/16), 15,488, 7,744.
  Ship 9,680 in the README of any k3-compat dataset.
* Statistics block shape `{mean, median, p95, p99, p99_9, max}` — identical to what
  `k6_kld_report.py` already emits and to Festr's, so the numbers line up field-for-field.
* Aggregation: `kl_micro_token_mean` (the headline), plus `kl_macro_*_mean` per declared stratum, per
  domain, and per context-depth bucket.

### 10.3 `context_depth_buckets` (adopted)

`0000-0255 / 0256-0511 / 0512-1023 / 1024-1535 / 1536-2046`, Festr's exact bucket edges. This
directly answers our own `scorefrom1024` question and lets a llama.cpp-geometry number
(`score_from = n_ctx/2`) be read off a full-panel capture **without recapturing**.

### 10.4 Self-compare: A == B

**SC-1 — reproduction confirmation.** `A.capture_content_digest == B.capture_content_digest`.

Because our checkpoint lane is bitwise deterministic, this is a real reproduction proof, not a smoke
test. The comparator asserts, **without running the matmul** (the T1 hash proof):

* every tokenwise KL value is exactly `+0.0`; fp64 `log_softmax` of bitwise-equal inputs is
  deterministic and `t_logp - s_logp` is a subtraction of equal doubles;
* panel mean exactly `0.0` — not epsilon;
* `top1_agreement` exactly `1.0`;
* every per-window max is `+0.0` (not `-0.0`);
* the tokenwise array is `np.save` of `scored_positions` float64 zeros. For our 51,175-position
  panel that is **409,528 bytes**, sha256
  **`3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17`** — a fixed constant,
  independently reproduced;
* for hidden form, the head content digests must also be equal, because identical hiddens say
  nothing about identical logits if the heads differ.

`--force-compute` runs the full math anyway and asserts the computed result is bitwise identical to
the short-circuit. CI runs both.

Receipt: `comparison_kind = "reproduction_confirmation"`. **Not a measurement row.**

**SC-2 — run-to-run floor.** Digests differ but `weights_identity` is equal
(`model_revision` + `checkpoint_identity_sha256` + `scope_digest` all equal).

Never assume small. A different grouped-mm kernel, GPU class or torch build changes the bf16 forward
itself; that class of difference is exactly what produced our 0.011506 cross-topology floor, which is
**1e-2-class**. The comparator computes the residual and labels it
`comparison_kind = "run_to_run_floor"`, never a reproduction and never a quantization result.

* `same_lane: true` ⇒ this row is a legal `floor_measurement_ref` target under **BIAS-006**.
* `same_lane: false` ⇒ `usable_as_floor: false`, `class: advisory`, bias block required.

**SC-3 — not submittable as a measurement.** `bin/fidelity/receipt.py::_scan_for_unsubmittable`
already refuses, at any depth, anything carrying `capture_role: bf16_teacher`,
`not_submittable: true`, or a `-preview.` schema. A reproduction-confirmation or floor receipt
therefore declares `comparison_kind` explicitly, and only `comparison_kind == "measurement"` may be
handed to `registry_add`.

### 10.5 The comparison receipt

Schema `malaiwah.fidelity-comparison-receipt.v1`, self-sealed by the §5.3 method, defined in
[`schema/fidelity-comparison-receipt.schema.json`](schema/fidelity-comparison-receipt.schema.json).
Top-level shape:

```
schema, format_version, receipt_sha256, created_utc,
comparison_kind          measurement | reproduction_confirmation | run_to_run_floor
reference: {dataset_sha256, capture_content_digest, role, form, lane, label, repository, revision,
            head: {tensor_content_sha256, quantized, source}, stack_fingerprint_sha256,
            lane_identity_sha256, weights: {...}, scope_digest}
candidate: {…same shape…}
panel:     {panel_id, suite_token_hash_sha256, contexts, scored_positions, scoring_window, tokenizer}
gates:     {seal, form, panel, head, lane, stack, geometry, coverage, lossy}   each {passed, detail}
comparator:{device, accumulation_dtype, logits_dtype, two_pass, vocab_chunk, position_block,
            tf32, deterministic_algorithms, bf16_reduced_precision_reduction,
            cublas_workspace_config, head_applied_tensor_content_sha256, source_file_hashes_verified}
metric:    {name, value, units: "nats", direction: "reference_to_candidate",
            direction_label: "KL(reference || candidate)", higher_is_better: false}
kl:        {mean, median, p95, p99, p99_9, max}
js:        {mean, median, p95, p99, p99_9, max}          optional
top1_agreement
kl_micro_token_mean, kl_macro_stratum_mean, per_stratum{}, per_domain{},
context_depth_buckets{}, per_context[], high_kld_contexts[], top1_discordant_contexts[]
uncertainty: {method, ci95_low, ci95_high, clusters, samples}
estimator:   {accumulation_dtype, logits_dtype, two_pass, vocab_chunk, stack_relation, head_policy}
determinism: {run_count, identical_across_runs, evidence_kind, evidence_hashes,
              distinct_evidence_hash_count, run_means, population_stddev_of_run_means}
measurement_scope: {scored_positions, contexts, covers_full_panel, subset_detail, position_filter}
comparability: {class, usable_as_floor, bias}
tokenwise: {path, bytes, sha256}
disclosures[]
```

**A registry submission is a separate object.** `compare --emit-submission` calls
`bin/fidelity/receipt.py::build_submission`, which self-seals and computes `scope_digest` **and** the
comparability key with `registry/tools/registry_lib.py` — the registry's own code, imported, never
reimplemented. Two implementations of a hash function is two chances to disagree.

---

## 11. Determinism evidence

```json
"determinism": {
  "run_count": 5,
  "cold_start_per_run": true,
  "evidence_kind": "hidden_state_tensor_sha256",
  "evidence_hashes": ["<capture_content_digest run 1>", "…"],
  "distinct_evidence_hash_count": 1,
  "identical_across_runs": true,
  "repeats": [{"name": "repeat-01", "dir": "determinism/repeat-01",
               "capture_content_digest": "…"}],
  "repeat_noise": {
    "kl_canonical_to_repeat_mean": null,
    "kl_repeat_to_canonical_mean": null,
    "js_mean": null,
    "interpretation": "Treat changes at or below this magnitude as runtime noise unless confirmed with repeated captures."
  },
  "note": "…"
}
```

**DET-D1** `identical_across_runs: true` requires `evidence_kind ∈ {hidden_state_tensor_sha256,
logits_tensor_sha256, tokenwise_kld_sha256, sealed_tokenwise_digest}`, `run_count >= 2`, and
`distinct_evidence_hash_count == 1`. (Mirrors registry **DET-001** / **DET-004**.)

**DET-D2 — file digests are not evidence.** `stream_score.py` writes a safetensors `__metadata__`
block containing `cold_run`, `checkpoint_identity_sha256` and `runtime_reader_sha256`, so **the whole
file digest of a logits file differs between bitwise-identical cold runs**. Confirmed empirically
(`k6/tools/hidden_replay_selftest.py` rung `f-payload-sha`: *"payload hashes equal across metadata
variants; file hashes differ"*) and in the published data (`reports/k6-five-run-kld.json`: five
different `student_capture_receipt_sha256`, five different `student_backend_identity_sha256`, **one**
`tokenwise_kld_sha256` `52e35723…`, `population_stddev_of_run_means: 0.0`).

The validator therefore **refuses** any `evidence_hashes` entry that equals a `sha256` (container)
field anywhere in the manifest while differing from the corresponding `tensor_content_sha256`.
Evidence hashes are `capture_content_digest` values or tokenwise-array digests. Nothing else.

**DET-D3** `panel_receipt_sha256` never appears in `evidence_hashes` (registry **PANEL-002**).

**DET-D4** `run_count < 5` on a published dataset carries a `reduced_run_count` disclosure
(**DET-006**, warn).

**DET-D5 — `repeat_noise` is a self-declared noise floor.** Adopted from Festr's 32×2048
`ref/manifest.json`, including the human-readable `interpretation` string. His artifact declares
`kl_canonical_to_repeat_mean: 0.003501…` with *"Treat changes around 0.004 or below as runtime noise
unless confirmed with repeated captures."* — machine-readable, self-declared, and honest. His 1024×2048
sentinel repeats show `~3.2e-3` KLD **between identical vLLM runs**, 400× his replay error. Our
serving lane's equivalent is `reports/determinism-bf16.json`: 32 sentinels, **20 byte-identical**, 12
mismatched. Our checkpoint lane's is 25/25. The field exists so that difference is a number in a
manifest and not a footnote.

---

## 12. Interop with kimi-k3

Reference artifact: `festr2/kimi-k3-distribution-fidelity-1024x2048-v1` (2,418 files, 1,175
downloads) and its window-form predecessor `festr2/kimi-k3-full-mxfp4-kld-reference-32x2048`.
Comparator: `comparators/compare-kimi-k3-hidden-replay.py`. Everything below was read from the
published files and source, not from prose.

**Position: v1 is a strict superset with byte-level payload compatibility and additive metadata.**
Adopt his file names, directory names, tensor keys, dtypes, digest preimages and `checksums.txt`
verbatim; add our blocks as new keys he ignores; never rename a field he reads.

The reason is empirical, not diplomatic: our published
`malaiwah/GLM-5.3-Flash-fidelity-suite-v1` hiddens are captured at the **same semantic point, dtype,
container, tensor key and filename pattern** as his. Only the metadata around them differs. Refusing
to converge would be inventing a difference that does not exist.

### 12.1 ADOPTED VERBATIM — identical name, identical semantics

| item | value |
|---|---|
| `semantic_point` | `"after_final_rmsnorm_before_lm_head"`, `"live_lm_head_output_before_sampling"` |
| `tensor_key` | `"hidden_states"` / `"logits"` |
| tensor filenames | `hidden_NNNN.safetensors` / `logits_NNNN.safetensors` |
| tensor form | one tensor per file, safetensors, BF16 hidden `[scored_rows, hidden_width]` |
| `token_ids_json_sha256` | field name **and** the compact `separators=(",",":")` preimage |
| `suite_token_hash_sha256` | field name **and** the `"\n".join(...)` ASCII preimage |
| `runtime_manifest` + `runtime_manifest_sha256` | relative path + digest; absolute refused |
| `raw_chunks_retained: false` | ⇒ host-local keys were stripped |
| `checksums.txt` | flat `sha256␠␠relpath`, `sha256sum --check`-compatible, excludes itself |
| `partition`, `sentinel`, `partition_salt`, `sentinel_salt` | derivable partition assignment |
| `raw_tensor_sha256` | his name for a tensor-content digest, used for the head |
| `token_ids_first16` / `token_ids_last16` | cheap eyeball check |
| `repeat_noise` + `interpretation` | self-declared machine-readable noise floor |
| `source_files: {path → sha256}` | capture code pinned by content |
| statistic block | `{mean, median, p95, p99, p99_9, max}` |
| `direction` label | `"KL(reference || candidate)"` |
| `context_depth_buckets` | `0000-0255 / 0256-0511 / 0512-1023 / 1024-1535 / 1536-2046` |
| per-result mini-seal | `results/<label>/manifest.json` with `files: {name → sha256}` |
| `container.{image_id, image_reference, image_repository_digest}` | all three |
| `runtime_environment` | flat dict of literal env vars |

### 12.2 ADAPTED — his idea, our name or our stricter form

| item | his | ours |
|---|---|---|
| record list key | `contexts[]` \| `windows[]` | emit `records[]`, **plus** `contexts` and `windows` aliases in `compat/`; every record carries `index` **and** `context_index` **and** `window_index` so both comparators bind |
| stratum | `allocation_stratum` + `semantic_class` | emit `allocation_stratum`; keep our `stratum`/`domain` as aliases |
| cluster | `source_cluster_id` | emit `source_cluster_id`; keep `source_cluster` alias |
| tensor binding | per-file `sha256` (container) only | `sha256` **+ `payload_sha256` + `tensor_content_sha256`**; manifest identity is the derived `capture_content_digest`, never the manifest file's hash |
| capture runtime | vLLM-specific `runtime` dict | `malaiwah.stack-fingerprint.v1` under `stack_fingerprint`, **plus** his flat `container` / `source_files` / `runtime_environment` blocks |
| replay qualification | `hidden-replay-qualification.json` | same file name and shape; **`comparator.device` promoted to REQUIRED** (QUAL-1) |
| bootstrap | context / source-cluster / stratified-source-cluster, 10k resamples | keep all three and the `cluster_unit` strings verbatim |
| status | `status: "qualified"`, validator refuses anything else | split into `structural_status` (validator verdict) and a separate `qualification` receipt, so a partial capture is representable |

### 12.3 `--emit-k3-compat`

> **STATUS: SPECIFIED, NOT YET IMPLEMENTED.** `bin/fidelity-dataset capture` has no
> `--emit-k3-compat` flag today and emits no `compat/` tree; `interop.k3_compat_emitted` is
> therefore `false` in every dataset this repo has produced. The design below has been **proven by
> a reference shim** against Festr's unmodified comparator (§12.3.2) — but until the flag lands,
> `interop.compatible_with: ["kimi-k3-distribution-fidelity/1"]` claims only that this format is a
> deliberate superset of his, **not** that his tools can read a dataset as emitted. Do not cite it
> as the latter.

Writes `compat/` with the alias keys his comparator reads (`contexts`, `context_index`,
compact `token_ids_json_sha256`, `allocation_stratum`, `source_cluster_id`) and a
`compat/lm-head/weight.safetensors` whose single tensor key is `"weight"`. Cost: a few hundred bytes
of duplicated JSON per dataset plus one optional alias file. Benefit: his comparator runs on our
data unmodified.

#### 12.3.1 What his reader requires

Checked line-by-line against his `validate_source_manifest`, then **executed**, the complete list of
what our dataset must satisfy for his tool to read it:

1. tensor dir `manifest.json` whose `suite_token_hash_sha256` matches the `--suite-manifest` we ship.
   **We do not emit this key in `capture/manifest.json` at all** — it lives only in `panel/panel.json`.
   `compat/` must copy it up. This is the first hard stop; his reader raises
   `Suite token hash mismatch` before looking at anything else.
2. that manifest has `contexts` **or** `windows` as a **list**. Ours names the list `records`, so his
   `manifest.get("contexts", manifest.get("windows"))` returns `None` and he raises
   `Source manifest has no contexts or windows`. Second hard stop.
3. each record has `context_index` \| `window_index` \| `index`; `file`; `key == "hidden_states"`;
   `sha256`; `token_ids_json_sha256` matching the suite record. **Already satisfied natively** —
   v1 records carry all three index aliases and the compact token digest.
4. the suite manifest has `context_length` and a record list with `token_file` +
   `token_ids_json_sha256`, and `tokens/` files hash under **his compact preimage**.
   Two traps here, and neither was in the original analysis:
   * **`token_file` resolves against the suite manifest's OWN directory**, because he calls
     `validate_suite_tokens(args.suite_manifest.parent, contexts)`. Our
     `panel.records[].token_file` is **dataset-root-relative** (`panel/tokens/context-0000.json`),
     so pointing him at `panel/panel.json` makes him look in `panel/panel/tokens/`. `compat/` must
     rewrite `token_file` relative to itself.
   * **`panel.json` cannot be handed to him directly under any aliasing**, because of the
     `contexts` collision in §7.2: ours is an integer count, his is the record list. Feeding him
     `panel/panel.json` yields `TypeError: 'int' object is not iterable`. `compat/suite-manifest.json`
     must be a **separate file**, not `panel.json` with keys bolted on.
5. tensors BF16, `shape[0] >= context_length - 1`, `shape[1] == --hidden-width` (pass `4096`).
   **Already satisfied natively.**
6. `--lm-head` is a BF16 `[vocab, hidden]` safetensors whose single tensor key is **`weight`**, and
   if `manifest.json` sits beside it its `file_sha256` must match. **Already satisfied natively** —
   v1 writes `head/weight.safetensors` with tensor key `weight`, and names our sidecar `head.json`,
   not `manifest.json`, so his optional cross-check correctly no-ops.
7. `--vocab-size 154880` must be divisible by `--vocab-chunk`; **9,680 works, his default 10,240 does
   not** and his parser errors out.

So the real work is items **1, 2 and 4** — all metadata, all in `compat/`. Items 3, 5, 6 were already
true, and item 7 is a documented invocation argument.

#### 12.3.2 Executed proof, and the number it produced

A reference shim emitting `compat/` per the above was run over two real v1 datasets adapted from our
own published capture (BF16 root and as-served FP8, 2 contexts x 2047 rows, vocab 154,880, our real
`[154880, 4096]` BF16 head). Festr's **unmodified** `compare-kimi-k3-hidden-replay.py` then ran
end to end:

```
validate_suite_tokens            PASS   his compact preimage over our tokens/
validate_source_manifest ref     PASS   2 tensors, container hashes verified
validate_source_manifest cand    PASS   2 tensors, container hashes verified
full two-pass comparison         OK     --vocab-chunk 9680 --position-block 128 --device cpu
```

| | mean KL(ref‖cand), nats | top-1 agreement |
|---|---|---|
| **his** comparator, fp32 log-probs, `clamp_min_(0)` | 0.03564599129280951 | 0.925256472887152 |
| **ours** (`fidelity-dataset compare`), fp64 log-probs | 0.03526219355348638 | 0.9257449926722032 |
| difference | **3.84e-4 (+1.09 % relative)** | 4.9e-4 |

**This is the empirical case for D-5.** Two careful implementations, the same bytes, the same panel,
the same head, the same direction — and a 1.1 % disagreement that comes entirely from estimator
precision (his log-probs are fp32 with a `clamp_min_(0)` that can only bias the mean upward; ours are
fp64 throughout, `dscompare.token_kld` casting to `float64` before the log-softmax). It is small in
absolute terms and **larger than the entire quantization-attributable signal we report for K6**
(0.00221 nats). A KLD number without `estimator.accumulation_dtype` is not comparable to another
KLD number, and this table is the receipt for that claim rather than an argument for it.

Note also what his receipt records about the head: `comparator.lm_head_file_sha256` names the one
head he was given and says nothing about whether it was the candidate's own. In this run that was
harmless (both sides are our TR3 lineage with a native BF16 head), but it is exactly the head trap
D-1 exists to refuse, visible in his own output format.

### 12.4 The k3 adapter — precisely what it needs

`bin/fidelity-dataset adapt --source k3v1` is pure metadata translation; **no tensor rewriting** is
needed, because BF16 `[n, hidden]` safetensors with key `hidden_states` is already our native form.

1. **Panel shim.** `suite-manifest.json` `contexts[]` → our `records[]`; `context_index` → `index`;
   `allocation_stratum` kept; `source_cluster_id` kept; `token_file` → `token_file`. Carry
   `semantic_class`, `language`, `representation_type`, `dataset*` as opaque extras.
2. **Digest re-derivation (unavoidable, cheap).** Read `tokens/*.json`, emit both preimages
   (§5.1). 1,024 small JSON reads; seconds.
3. **Head identity synthesis.** His artifact has no per-capture head field. Set `source: "artifact"`,
   `file_sha256` and `raw_tensor_sha256` from his `manifest.lm_head`, `quantized: null`
   (**unknown**), and stamp disclosure `head_identity_inferred_from_reference_artifact`. Any compare
   of a k3 reference against a non-k3 candidate is therefore **advisory, never strict** — correct,
   because his format genuinely does not record it.
4. **Lane** = `serving`, derived from `capture-runtime.json.runtime` (vLLM TP16); stamp
   `lane_inferred: true`. (Maps to the registry's `other` lane unless a `serving` lane is added.)
5. **Scoring window**: `score_from: 0`, `windowed: false`, `min_left_context_tokens: 1`, derived from
   `scored_positions_per_context == context_length - 1`.
6. **Stack fingerprint**: wrap his `capture-runtime.json` with `origin: "k3v1-capture-runtime"`; his
   `container.image_repository_digest`, `runtime.*_commit`, `runtime_environment` and `source_files`
   all map onto existing stackprint concepts.
7. **Determinism**: import `validation/artifact-validation.json.sentinel_repeat_mean_kld`
   (`{"00-vs-01": 0.0032166685936858316, "00-vs-02": 0.0031814546488495347,
   "01-vs-02": 0.0031337794740479803}`). His per-file `sha256` is a **container** hash, so the
   imported `evidence_kind` downgrades to `run_mean_equality_only` unless we recompute content
   digests locally (one pass over 64 sentinel files, ~1.8 GB — a decision, not a blocker).
8. **Coverage**: `declared = 1024`, `present` = what was downloaded. His README explicitly blesses
   partial downloads (skip the 120 GiB `sentinel-live-logits/`), so `--allow-partial` is the normal
   path.
9. **Refusal to fabricate.** His artifact publishes **no candidate captures**, only compare receipts.
   The realistic k3 adapter target is therefore **reference-side only**, and comparing our GLM
   candidate to his Kimi reference is meaningless anyway (different model). The adapter's value is
   validation-by-construction of the format, plus the ability to ingest a future kimi-k3-style
   artifact *for any model, including GLM*.

### 12.5 MUST DIVERGE — ten items, each a refusal we can name

| id | divergence | the failure it prevents | whose gap |
|---|---|---|---|
| **D-1** | every capture declares its **own** head identity; comparator refuses a mismatch (§8.3) | replaying an EXL3 `head_bits 6` candidate's hiddens through the reference head erases its head-quantization error | his |
| **D-2** | `lane` is a required top-level field; cross-lane compare needs a flag and a disclosure | BIAS-006 is checked at compare time instead of months later at submission | his (one lane) |
| **D-3** | `scoring_window` is part of **panel identity**, not a comparator flag | a `score_from=1024` number silently compared to a `score_from=0` number | both |
| **D-4** | `self_compare` is a first-class mode asserting **exactly** 0.0 | the same-lane floor problem; his `sentinel_repeat_mean_kld` is this done by hand with a naming convention | nobody has it |
| **D-5** | `estimator.accumulation_dtype` required; fp64 default | his fp32 log-probs land in a different comparability class than our fp64 **on the same data** — that must be visible | his |
| **D-6** | `coverage` block + partial refusal | **our own** manifest claiming 5,120 captures in a 512-file repo; his validator hard-codes `!= 1024 → raise` | ours |
| **D-7** | panel is a separate, independently referenceable object | his candidate captures cannot be published standalone at all | his |
| **D-8** | `lossy_codec` block (nullable) | ingesting llama.cpp `.kld` (uint16 log-probs, `max_logit − 16` clamp) without laundering a lossy capture as exact | field-wide |
| **D-9** | `attention_mask_sha256` per record | our packed/streaming lanes vary mask construction; his single-request path makes it invariant | ours |
| **D-10** | hidden form is the default; logit form must declare `head_separable: false` + reason | 31.7 GB vs 419 MB on our panel; 811 GB is what fp32 logits actually cost in the wild | his (same call, undeclared) |

The machine-readable form of this table is the manifest's `interop` block:

```json
"interop": {
  "compatible_with": ["kimi-k3-distribution-fidelity/1"],
  "k3_compat_emitted": true,
  "divergences": [{"id": "D-1", "field": "head", "reason": "…"}, "…"]
}
```

That array **is** the outreach document: it says *we read your format, here is exactly where we
differ and why*, without requiring anyone to read prose.

### 12.6 Other prior art, and where it lands

| artifact | what it is | provenance | verdict |
|---|---|---|---|
| **llama.cpp `.kld`** (`--kl-divergence-base`) | the de-facto community format. Binary: magic `"_logits_"`, `uint32 n_ctx`, `int32 n_vocab`, `int32 n_chunk`, tokens, then per row `2*((n_vocab+1)/2)+4` uint16 (first 4 = `scale`, `min_log_prob` as 2 floats) | **none** — no hashes, no model id, no tokenizer id, no runtime; `n_vocab` mismatch is a *warning* | **Lossy**: 16-bit quantized log-probs, hard `max_logit − 16` floor, scores the **second half only** (`first = n_ctx/2`). Ingestible as logit-form with `lossy_codec` + `scoring_window.score_from = n_ctx/2` + `head_separable: false` + `stack_fingerprint: null`. Lands **advisory**. Payoff: every published llama.cpp KLD number becomes *placeable* for the first time. The panel travels **by value** inside the file — genuinely good, and worth stealing for tiny panels. |
| **exllamav3 `eval/model_diff.py`** | bare safetensors, key `"logits"` (or `tensor.{i}`); panel regenerated on the fly from wiki2 | none | a debugging dump, not a dataset. Container convention matches everyone's. |
| **`brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits`** | fp32 full-vocab logits on **our exact panel**; 811,621,019,136 B for the 640-window panel | `dataset-manifest.json` + `capture-receipt.json` (= our `CAPTURE_SCHEMA`) + `backend.json` + independent Hub verification at an immutable revision | **our own pipeline's lineage, published by a third party.** Two fields adopted from it: `attention_mask_sha256` (D-9) and `role` as a panel-record field. Empirical proof of the storage argument. |
| **`festr2/kld-reference-qwen35-35b-a3b-fp8`**, **`AesSedai/reference-logits`** | 100 bare `N.safetensors` / a `.bin` + `.pt` with a README saying it is a scratchpad | none | the floor of the field, and live demand for this spec. |
| **lm-eval-harness** | `--log_samples` JSONL + a results JSON with `git_hash`, `model_args` | git hash + model args | adjacent, not competing: it standardizes *eval results*, not *captures*. Its results schema is what HF `model-index` reads — the hook for the card annotation. |

**There is no competing capture-dataset standard.** One lossy binary format with zero provenance, one
well-engineered single-author artifact family, our own pipeline's output published twice under two
schema names, and a long tail of unlabelled tensor dumps.

---

## 13. Defects in our own published artifacts that v1 fixes

Stated plainly, because a spec that only names other people's gaps is marketing.

| id | defect | fixed by |
|---|---|---|
| **O-1** | `semantic_point` is nowhere declared. `glm53flash-fidelity-capture/2` has no cut field; the README says only "capture final-norm hidden states"; shipping `final_norm.safetensors` next to the head implies a norm+head replay. A third party will guess wrong. | FORM-2, §8.5, HEAD-7 |
| **O-2** | tensor key inconsistency **inside our own tooling**: `tools/fidelity.py` writes `hidden_states`, `k6/tools/hidden_replay.py` writes `hidden`. Festr's comparator hard-codes `hidden_states` and would reject the streaming-lane captures on a one-word difference. | REC-2 (`hidden_states` normative, `hidden` accepted on ingest and rewritten) |
| **O-3** | the published manifest **overclaims**: `expected_contexts: 5120`, `complete: true`, in a repo holding indices 0–511. Nothing says "shard 0 of 10". | §6.3 coverage, COV-1..3 |
| **O-4** | per-capture records are thin: `{index, sha256, shape}` — no `file`, `dtype`, `key`, token digest or content digest. Panel binding is one top-level hash, so a single swapped or reordered hidden file passes. | §6.2 record, BIND-1..6 |
| **O-5** | K6's published `materialization-receipt.json` still names `packed_root: /home/jl_fs/glm53-k6/out-k6`. `stream_score --source checkpoint` does `packed_root = Path(materialization["packed_root"]).resolve()` then fails if it is not a directory — **there is no override flag** — and `--source payload-store` needs `contract.json`, `inventory.json`, `mtp-adapter-receipt.json` and the `payload-store/` trees, none published. Both packed reading paths are unreachable from public artifacts. (`--source exl3hf` **is** reachable: payloads are inline and `lm_head.weight` is a plain BF16 tensor.) | PATH-2, §9.4 `stripped_fields`, and the existence of the dataset itself — a published capture makes the reading path irrelevant |
| **O-6** | two head-digest conventions live in our published receipts: `head-extraction.json` / `head-equality-fp8.json` record the **file** digest `47eaf729…`; `nonrouted-sparse-fetch.json` records the **tensor content** digest `aa21c427…`. | HEAD-IDENT: both required, content normative, cross-convention comparison forbidden |
| **O-7** | `tools/fidelity.py cmd_replay --candidate-head` defaults to `None` and silently applies the reference head to both sides — the head trap, live in our own code. | HEAD-1b refusal; a mirrored fix in `tools/fidelity.py` is listed as out of scope (§14) |

---

## 14. Out of scope for v1 (with reasons)

| item | why |
|---|---|
| **Integration with `bin/measure-cloud`** | explicitly out of scope in the brief; a sequential measurement workflow owns `bin/measure_cloud.py`, `bin/stage_measure.sh`, `bin/fidelity/hfmeta.py`, `bin/engines.json`, `bin/invoke_engine.py`. Integration points are documented in the build plan instead. |
| **Editing `k6/tools/stream_score.py`** | same reason, plus the format adapters just merged there. All capture paths are **wrapped**, never edited — the precedent `k6/tools/hidden_replay.py` already sets. |
| **Fixing `tools/fidelity.py cmd_replay`'s `--candidate-head` default** (O-7) | a real bug, in a file this work does not own. Filed as a follow-up; the comparator refuses the same condition in the meantime. |
| **Re-publishing corrected K6/K8 materialization receipts** (O-5) | changing a published artifact's receipts is an operator decision with downstream consequences for anyone who pinned them; the dataset makes the defect harmless without touching them. |
| **A new registry `lane` value (`serving`)** | would change `submission.schema.json`'s enum and reclassify existing rows. k3-adapted datasets map to `other` with `lane_inferred: true` until an operator decides. |
| **HF eval-results v2 (`.eval_results/*.yaml`, benchmark `eval.yaml`)** | architecturally the right long-term home and confirmed live in production on `zai-org/GLM-5.3`, but blocked on three upstream items: an `evaluation_framework` enum PR to `huggingface.js`, an HF benchmark allow-list request (explicitly beta), and the fact that its bare `value:` cannot express units or direction — a KLD leaderboard would sort backwards. Emitted behind an off-by-default flag; see CARD-ANNOTATION-SPEC §6. |
| **Publishing large captures** | the tooling can `--publish` a dataset, but this work publishes only the spec and the updated cards. No GPU, no rentals, no bulk upload. |
| **Suite-scale capture (10.48M positions, ~86 GB)** | the format supports it and the sharding rules exist; producing one needs a GPU. The RAM fix the capture command needs for it (per-window flush instead of accumulate) is specified in the build plan. |
| **Bootstrap CIs beyond context / source-cluster / stratified-source-cluster** | `bin/fidelity/previewstats.py` already computes cluster bootstraps; nothing new is designed here. |
| **A signature scheme (GPG / sigstore)** | the seal is tamper-**evident**, not tamper-**proof**. Binding `dataset_sha256` to an identity is a separate decision with key-management consequences. The external anchor (card + registry row + HF revision) is v1's answer. |

---

## 15. Worked examples

* Root: [`examples/fidelity-dataset.root-glm53-bf16.json`](examples/fidelity-dataset.root-glm53-bf16.json)
* Quant: [`examples/fidelity-dataset.quant-glm53-k6.json`](examples/fidelity-dataset.quant-glm53-k6.json)
* Comparison receipt: [`examples/fidelity-comparison-receipt.k6-vs-bf16.json`](examples/fidelity-comparison-receipt.k6-vs-bf16.json)
* Self-compare receipt: [`examples/fidelity-comparison-receipt.self-compare.json`](examples/fidelity-comparison-receipt.self-compare.json)

Every digest in the examples that is marked `<synthetic>` is a placeholder. Every digest **not** so
marked is a real value read from a published artifact or a registry row, so the examples double as
fixtures.

---

## 16. Implementation addenda (v1, 2026-08-29)

Four clarifications the implementation forced. All are compatible with v1's
additive-only rule: none adds a required key, changes a type, or changes a
digest preimage.

**A-1 — `runtime_manifest` may carry `..`.** PATH-3 says `..` is permitted only
inside `compat/`, but §6.1's own example gives
`"runtime_manifest": "../runtime/capture-runtime.json"` inside
`capture/manifest.json` — because that field is adopted verbatim from kimi-k3,
whose `reference-hidden/manifest.json` names `../capture-runtime.json` and whose
validator refuses only an ABSOLUTE path. The rule as implemented: a path is
resolved relative to the directory of the file that carries it and must resolve
**inside the dataset root**; `..` is accepted under `compat/` and on the
`runtime_manifest` field, and refused everywhere else. Containment — what PATH-3
actually protects — is enforced in every case. Cases F8, F9, F11.

**A-2 — a `validation/structural-validation.json` is written BEFORE the seal.**
§3 requires the file and §5.4 requires `checksums.txt` to cover every published
byte, so a `validate` run that wrote its report *into* a sealed dataset would
make it `unlisted_file`. `capture` and `adapt` therefore write the report before
`checksums.txt`; a third party validating a downloaded dataset writes theirs
outside it (`validate --json OUT`). Resealing after validation would change
`dataset_sha256` and break the external anchor, which is worse.

**A-3 — `estimator_backend` is recorded.** §10.2 fixes the estimator as
`torch.log_softmax` in float64. The implementation calls
`k6_kld_report._token_kld` — imported, not copied — whenever torch is
importable, and falls back to the identical fp64 formula in numpy when it is
not. Which path ran is recorded in `comparator.estimator_backend`, because a
silent backend swap is exactly the class of undeclared difference this format
exists to surface. Both paths are asserted against the same analytic oracle
(case N1).

**A-4 — a floor must match on SCOPE as well as LANE.** §10.1 and registry
**BIAS-006** require a floor to come from the same lane. Live registry data
showed the same failure one axis over: a 17-window `clean17` row citing a
25-window `panel25` floor. A floor over a different set of scored positions is
not this row's zero-point, for exactly the reason a floor from another lane is
not. Implemented as a refusal in the card generator
(`cardmeta.attributable_refusal`, case K8b) and specified as registry invariant
**DS-005** in [`REGISTRY-INTEGRATION.md`](REGISTRY-INTEGRATION.md).
