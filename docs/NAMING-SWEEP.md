# NAMING-SWEEP.md — retiring GLM-specific names from a model-agnostic scorer

This repository began as one campaign: measure the fidelity of a K6 encode of
GLM-5.3-Flash. It is now a general yardstick. It has measured GLM-5.3-Flash,
Qwen3.8-27B and Fruit, and carries working engines for MiniMax-M3, DeepSeek V4
and Qwen3.8-Flash-Next. The GitHub repository was renamed
`glm53-fidelity-suite` → `quant-fidelity-suite`; the code did not follow, and a
name that says "glm53" on a MiniMax run is no longer quaint, it is wrong.

This document is the decision list for that sweep, written **before** the
changes, so every verdict is reviewable line by line.

## The rule the sweep runs on

There are two kinds of "GLM" and "K6" in this tree, and they get treated
oppositely.

**RENAME** — the name is *incidental*: it records where the code started, not
what it is. Directory names, module filenames, environment variables, local
variables, filesystem roots, and prose that says "GLM" where it means "the
model".

**KEEP** — the name is *identity*: something hashed, sealed, published, or
pointed at by someone else.

* **Registry ids** (`measurement--glm53.*`, `artifact--zai-org.*`,
  `panel--glm53.brandonmusic.final25`, `reference--*`, `pipeline--*`,
  `model--*`). `COMPARABILITY_KEY_FIELDS` hashes `panel_id` and
  `reference_id`; renaming one silently regroups every measurement that
  referenced it. **76 of the registry's 159 ids contain "glm". All 76 stay.**
* **Receipt schema strings** (`malaiwah.glm53-*-kld-summary.v1`,
  `quant-pipeline.glm53-token-panel.v1`, `glm53flash-fidelity-capture/2`, …).
  They appear in sealed receipts whose `receipt_sha256` covers their bytes.
  **131 distinct schema literals in the code contain "glm". All 131 stay.**
* **Anything under `registry/receipts/`** (10 sealed receipts),
  **`registry/data/*.jsonl`**, **`reports/stack-provenance-retro.json`**, and
  every `*.receipt.json` / `*-evidence/*.json` under the engine tree.
* **Real model names, HF repo ids, revisions, architecture ids**
  (`zai-org/GLM-5.3-Flash-BF16`, `glm_moe_dsa`, `glm5_next`,
  `Glm5NextForConditionalGeneration`, `GlmMoeDsaForCausalLM`), and upstream
  module names from brandonmusic's pipeline
  (`quant_pipeline.evaluation.glm53_packed_k4_reader`).
* **Profile labels naming a published receipt family** (`turbo-4.05bpw`,
  `k6-stream-tp4`, `--profile k6|k8|k6k8`).

When in doubt the string stays, and the reason is written down. A rename that
breaks a seal is far worse than a name that reads oddly.

---

## RENAME — the decision list

| # | from | to | scope | why |
|---|---|---|---|---|
| R1 | `k6/` (directory) | `engines/` | 343 tracked files moved with `git mv`; ~120 text files reference the path | The directory name means "the K6 quant campaign". It is now the home of *every* capture engine — `stream_score.py`, `hf_capture.py`, `layer_outer.py`, the MLX/GGUF/NVFP4/EXL3 surfaces, and the committed MiniMax panel. `bin/README.md` and `bin/BUNDLE.txt` already call its contents "the measurement engines". |
| R2 | `FIDELITY_K6_ROOT` | `FIDELITY_ENGINE_ROOT` | 15 occurrences, 8 files | An exported on-instance root. The old name says which campaign paid for the box. The old spelling is still **read as a fallback** so a mixed bundle cannot silently point at nothing. |
| R3 | `/home/jl_fs/glm53-k6`, `/workspace/glm53-k6` | `/home/jl_fs/fidelity-engine`, `/workspace/fidelity-engine` | 12 literals in `bin/` | A model name baked into a filesystem path on rented hardware. This is the same defect class as the three `/home/jl_fs` roots that each cost a paid run, one provider deep instead of one model deep. Consequence: the first run against a provider filesystem that still holds the old tree re-bootstraps into the new root. The bootstrap is idempotent and guarded, so this costs time, not correctness. |
| R4 | `Teardown.k6_root` | `Teardown.engine_root` | `bin/measure_cloud.py` | Follows R2. |
| R5 | `k6/tools/k6_kld_report.py` | `engines/tools/kld_report.py` | bundle entry, `engines.json` entrypoint, 2 shell guards, docs | It scores any capture against any teacher. Nothing in it is K6. |
| R6 | `k6/tools/k6_student_capture.py` | `engines/tools/student_capture.py` | bundle entry, `engines.json` | Same. |
| R7 | `k6/tools/k6_driver.py` | `engines/tools/campaign_driver.py` | `stage_campaign.sh`, `engines/RUNBOOK.md` | It drives *an* encode campaign; the profile is a `--profile` flag (`k6`, `k8`, `k6k8`), not part of the tool. |
| R8 | `k6/tools/k6_publish.py` | `engines/tools/publish_release.py` | `stage_campaign.sh`, `engines/RUNBOOK.md` | Same. |
| R9 | `k6/k6_publish.py` | *deleted* | — | **Byte-identical duplicate** of `k6/tools/k6_publish.py` (sha256 match). Nothing invokes the top-level copy: `stage_k6.sh` runs `$TOOLS/k6_publish.py`. This is the drift class `selftest_canonical_json.py` exists for, caught before it drifted. |
| R10 | `k6/stage_k6.sh` | `engines/stage_campaign.sh` | `bin/bootstrap_measure.sh` prose, `bin/stage_measure.sh` prose, `bin/_check_kld_profiles.py`, docs | It is the encode-campaign staging script; the measurement lane has owned its own bootstrap since `bootstrap_measure.sh` landed. |
| R11 | `~/.cache/glm53-fidelity` | `~/.cache/quant-fidelity` | `bin/fidelity/registry_client.py`, `bin/fixture_fetch.py` | `FIDELITY_CACHE_DIR`'s default. Renaming orphans an existing cache; the only thing in it is the 0.1B fixture and registry snapshots, both re-fetchable. |
| R12 | `docs/GLM53-LAYER-OUTER.md` | `docs/LAYER-OUTER.md` | 8 references | The document's own first gate table is `glm5_next`, `glm_moe_dsa` **and** `minimax_m3_vl`, and its second line says "No GLM-5.3 capture was run". It is the design document for a general capture schedule. |
| R13 | `selftest_progress.py` rung P11's hardcoded `k6/tools/` prefix | derived from `BUNDLE.txt` itself | `bin/selftest_progress.py` | The bundle-completeness rung was itself hardcoded to the campaign directory, so the directory rename would have silently reduced it to checking nothing. Deriving the engine directory from the bundle makes the rung rename-proof. |
| R14 | prose: "`k6/tools/` … assumes GLM-5.3-Flash and the K6 encode" | prose naming the engine tree | `bin/README.md`, `bin/BUNDLE.txt`, `registry/CONTRIBUTING.md` | False as written: those surfaces read MLX, GGUF, NVFP4, EXL3 and stream four architectures. |
| R15 | `/Users/someone/Projects/glm53-fidelity-suite/registry` | `…/quant-fidelity-suite/registry` | `bin/selftest_fidelity_card.py` fixture | A test fixture quoting the pre-rename repository name. |

---

## KEEP — and the reason

| what | count | why it stays |
|---|---|---|
| registry ids containing `glm` in `registry/data/*.jsonl` | 76 | Published identities. `panel_id` and `reference_id` are hashed into `comparability.key`; changing one regroups every measurement that used it. |
| receipt schema literals containing `glm` in `bin/`, `engines/tools/`, `registry/tools/` | 131 | They are inside sealed receipts. `receipt_sha256` covers their bytes and registry invariant `RECEIPT-001` refuses a row whose receipt was edited. |
| `registry/receipts/**` (incl. `stream-k6-kld.json`, `stream-k6-verdict.json`) | 10 files | Sealed and published. The absolute paths inside them (`/home/jl_fs/glm53-k6/...`) record where the run actually happened; that is provenance, not configuration. |
| `registry/data/*.jsonl` prose citing `k6/tools/dione_surface.py`, `k6/tools/hf_capture.py`, `k6/tools/k6_kld_report.py`, `k6/tools/derive_scope.py`, `k6/K8-ANOMALY.json` | 14 rows | These name the code **at the commit that produced the number**. Rewriting them would change `scope_digest` and would break `make reseed-check` against `seed_registry.py`. Historical paths in a provenance field are correct even after the tree moves. |
| `registry/tools/seed_registry.py` strings that generate the rows above | — | Same reason: it is the generator of `registry/data`. |
| `reports/stack-provenance-retro.json` | 1 | Sealed receipt; it already pins its claims to commit `6a04873`, where those paths existed. |
| `docs/examples/fidelity-dataset.{root-glm53-bf16,quant-glm53-k6}.json`, `docs/examples/fidelity-comparison-receipt.k6-vs-bf16.json` | 3 | Sealed (`receipt_sha256`) and re-verified by `bin/selftest_fidelity_dataset.py`. They are the published *examples* of our GLM-5.3 root and K6 quant captures — the names describe the data. |
| `docs/joint-standard/analysis/k6-*.json`, `paired.K6-vs-*.json`, `registry/protocol/per-window/k6-*.json` | 12 | Analysis outputs for the K6 artifact. The name is the subject. |
| `registry/protocol/glm53-joint-kld-protocol.v1.json` | 1 | The filename mirrors its own `schema` id `malaiwah.glm53-joint-kld-protocol.v1`, which is sealed. Renaming the file and not the schema (or vice versa) is exactly the two-sided-agreement bug this sweep is supposed to avoid. |
| `engines/recipes/{k6,k8,k6k8}.json`, `--profile k6|k8|k6k8`, `QP_STREAM_PROFILE` values | — | They name real encode profiles that published rows were measured at. |
| `engines/BF16-FLOOR.md`, `engines/K8-ANOMALY.md`, `engines/DECISIONS.md`, `engines/RUNBOOK.md` | — | Their subject *is* the K6/K8 campaign. Only their directory moves. |
| `engines/tools/glm53-stagea-evidence/`, `engines/tools/exl3hf-evidence/scope-turbo-*.json`, `engines/tools/mlx-evidence/real-dequant-fixtures/GLM-5_3-Flash-*.npz` | — | Evidence describing actual GLM-5.3-Flash artifacts. |
| `engines/patches*/`, `engines/.patchwork/` | 20 patches + 2 trees | Verbatim patch series against brandonmusic's pipeline, applied on the instance by `bootstrap_measure.sh` and hashed in `hidden-replay-evidence/patches-applied.txt`. Their filenames are content. |
| `remote/` (`/home/ubuntu/glm53`, `/glm53` container mount, `vllm/vllm-openai:glm53-flash-*`) | 10 files | The historical GLM-5.3 vLLM serving-lane campaign. Those paths are on a machine that produced published receipts, and the published model cards ship `MODEL_PROFILE=glm53-k6` as a deployment instruction. |
| `bin/fidelity/stackprint.py` `IMAGE_PIN_CONVENTION_PATH = "/glm53/out/image-pin.txt"` | 1 | It must equal the path `remote/vm_setup.sh` actually writes. Changing one side silently loses the container digest. A new test now asserts the two agree (see below). |
| `port/glm5_next.py.draft`, `port/tests/glm5_layer_parity.py` | 2 | `glm5_next` is the architecture's real `model_type`. |
| `docs/GLM53-ROOT-FEASIBILITY.md` | 1 | Its subject is literally "Can we capture a root fidelity dataset for GLM-5.3". |
| `docs/cards/GLM-5.3-Flash-TR3-*.README.md` | 2 | Published HF model cards for real GLM-5.3-Flash artifacts. |
| `JOURNAL.md` | 1 | A dated log of what happened. Paths in it are correct as of the entry that mentions them; rewriting them would be rewriting history. |

---

## Known consequence of R1 that cannot be fixed from this repository

Two **already-published** HF model cards deep-link into the old directory:

* `docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md` → `.../blob/main/k6/BF16-FLOOR.md`,
  `.../blob/main/k6/fallback/closure-comparison.json`
* `docs/cards/GLM-5.3-Flash-TR3-8bpw.README.md` → `.../blob/main/k6/K8-ANOMALY.md`,
  `.../blob/main/k6/BF16-FLOOR.md`, `.../blob/main/k6/fallback/closure-comparison.json`

GitHub does not redirect file paths across a directory rename, so the copies
**on Hugging Face** will 404 until the cards are re-pushed. The sources in
`docs/cards/` are updated by this sweep, so a re-push fixes them; publishing is
outside this change's remit and is left as an explicit follow-up.

`llms.txt` links (`k6/BF16-FLOOR.md`, `k6/K8-ANOMALY.md`, `k6/HANDOFF.md`) are
served from this repository's `main` and are corrected here, so they heal on
merge.

---

## Tests added or strengthened by this sweep

Each is verified to fail without its fix; the commit that adds it says so.

1. **`bin/selftest_naming_sweep.py` — published identity is frozen.**
   A snapshot of every registry id and every receipt-schema literal in the tree
   is checked in. The test fails if any of them *disappears or changes*
   (additions are fine). This is the regression test for the very operation
   this document describes.
2. **`bin/selftest_naming_sweep.py` — no model-specific hardcoding on the
   instance.** Extends the `/home/jl_fs` rule in
   `selftest_provider_portability.py` from *provider* names to *model* names:
   no uploaded, on-instance file may contain a model or campaign token
   (`glm`, `k6`, `k8`, `qwen`, `minimax`, `deepseek`, `fruit`) inside a
   filesystem path literal.
3. **`bin/selftest_naming_sweep.py` — two-file agreements.** The engine-root
   environment variable emitted by `measure_cloud._stage_env()` must be the one
   every on-instance script reads; `stackprint.IMAGE_PIN_CONVENTION_PATH` must
   be the path `remote/vm_setup.sh` writes; `bin/engines.json` entrypoints must
   exist and be bundled.
4. **`bin/selftest_naming_sweep.py` — no byte-identical duplicated module.**
   The `k6_publish.py` pair is the instance this sweep found; the check is
   general.
5. **`bin/selftest_progress.py` P11** now derives the engine directory from
   `BUNDLE.txt` instead of hardcoding `k6/tools/`, and follows imports for
   every bundled Python file, not only the engines.
