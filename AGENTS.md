# AGENTS.md — working on this repository

You are editing code that produces published scientific claims. Numbers from this
tree are on Hugging Face model cards, in a public registry other people query, and
in a community standards discussion. A silently wrong number here is worse than a
crash, because a crash gets fixed and a wrong number gets cited. Fail closed
rather than infer a revision, surface, scope, profile, lane, dependency or metric.

If you are here to *use* the yardstick rather than change it, read
[`llms.txt`](llms.txt) instead — it carries the rules that decide whether two
numbers may be compared. [`WHAT-WE-MEASURE.md`](WHAT-WE-MEASURE.md) says what a
number actually is.

## What this is, and how a number gets made

There is no `src/` tree and no application server. The product is a set of Python
and Bash CLIs over explicit filesystem/JSON state, in six stages:

1. **Resolve and gate.** `bin/measure` (→ `bin/measure_one.py`) parses an HF
   target, asks the registry whether it is already measured *before* any work or
   spend, pins a 40-hex revision, walks `base_model` lineage to a root the
   registry knows, inherits panel/reference precedent, sniffs the artifact's
   storage surface, and picks a lane.
2. **Plan.** `bin/measure_local.py` solves identity, device, memory, disk and
   invocation for hardware you already own; `bin/measure_cloud.py` adds provider
   capacity, a cost estimate, runtime/cost limits, leases and teardown backstops.
   Shared policy lives in `bin/fidelity/` — HF identity, lineage, the registry
   front gate, fit arithmetic, engines, stages, receipts, provider adapters.
3. **Execute.** `bin/engines.json` is the authored lane → entrypoint/profile/flag
   contract; `bin/fidelity/engines.py` maps lane-neutral values onto the engine
   CLI that was actually probed. Remote work persists `job.json` and drives
   `bin/stage_measure.sh` through `bin/fidelity/stages.py`. Two transports exist:
   SSH plus an uploaded bundle, and the measurement image
   (`container/Dockerfile`, entrypoint `bin/container_entry.py`, built by
   `container/build.sh`, CI in `.github/workflows/container-image.yml`).
4. **Measure and score.** `engines/tools/stream_score.py` (single device,
   streaming) and `engines/tools/student_capture.py` (distributed sealed capture)
   produce logits through per-format adapters (`engines/tools/*_surface.py`);
   `engines/tools/kld_report.py` verifies teacher/candidate identity, computes
   tokenwise KL(reference ‖ candidate) in fp64, and aggregates cold runs.
5. **Seal.** `bin/fidelity/receipt.py` and `bin/seal_receipt.py` reject preview and
   teacher material, bind code and artifact digests, and emit a self-sealed
   submission receipt.
6. **Ingest and publish.** `bin/registry-submit <receipt.json>` validates offline
   and publishes nothing. Maintainer ingestion files the receipt under
   `registry/receipts/<handle>/`, derives rows with `registry/tools/registry_add.py`,
   validates them, and regenerates `registry/data/*.jsonl`, `index.json` and
   `README.md` with the registry's render tool.

The portable dataset route is parallel, not a shortcut around any of that:
`bin/fidelity-dataset capture | verify | compare | publish` makes a root capture a
public good that later measurements read instead of re-paying for. `measure-cloud
--role root` is how a root is captured; `--race` overlaps that capture with the
still-running download ([`docs/RACE-MODE.md`](docs/RACE-MODE.md)). Model-card
publication is a separate, permissioned step *after* registry identity exists.

State is files — plans, `job.json`, leases, stage logs, `.done` markers, captures,
reports, receipts — and a `.done` marker appears only after success. CLIs are
synchronous; cloud work uses detached remote stages plus polling, a heartbeat
thread and locked teardown. Do not introduce an async framework without a
demonstrated need.

## Where things live

| path | what it owns, and its boundary |
|---|---|
| `bin/` | user CLIs, local/cloud controllers, receipt/dataset/card tooling, selftests, the on-box bundle manifest. Orchestrates; implements no model math. |
| `bin/fidelity/` | shared controller policy: HF identity, lineage, the registry front gate, fit arithmetic, engines, stages, provider adapters, receipts, dataset formats. |
| `engines/tools/` | capture and scoring engines, storage-surface adapters, real-tensor parity evidence, offline selftests. Tensor-heavy code lives only here. |
| `engines/` | campaign recipes, panels, upstream patch series, the stage driver, operational evidence. Deliberately model- and campaign-specific in places. |
| `registry/` | schemas, invariants, sealed receipts, generated records/index/README, frozen protocols, add/validate/render tools. |
| `docs/` | frozen wire-format contracts, operational plans, analyses, additive corrections. Status prose ages; schemas and receipts are the stronger evidence. |
| `container/` | the reproducible measurement image and its build wrapper. |
| `reports/`, `registry/protocol/` | receipt-backed experimental evidence and frozen protocol inputs. Cite these; never copy their numbers into fresh prose. |
| `suite/`, `calsuite/` | committed suite and calibration manifests; token payloads are generated and gitignored. |
| `tools/`, `remote/` | the original GLM serving-lane harness and VM campaign pipeline. Historical, not the current generic runner. |
| `port/` | draft native exllamav3 architecture port and manual parity harnesses. Not production code. |

The engine tree's root is `FIDELITY_ENGINE_ROOT` (the pre-2026-08-31 spelling
`FIDELITY_K6_ROOT` is still *read* as a fallback, never written).

## Verify before you claim

The single most expensive failure mode in this project's history is an agent
believing a document, a docstring, or another agent instead of running the thing.

- **Probe CLIs, never read their docs.** A runner was once written against a
  scorer's *documented* flags; five of them did not exist, and the lane could not
  run at all. `--help` is cheap; so is `bin/measure-local --probe-engines`, which
  reports what each lane's entrypoint really accepts. Never document a flag you
  did not confirm.
- **Hash tensor CONTENT, never containers.** Receipts embed `elapsed_seconds`;
  safetensors embed `__metadata__` including `cold_run`. Two bitwise-identical
  computations produce different file digests. We raised two false
  "nondeterminism" alarms in one hour before comparing tensor bytes and finding
  `max_abs_diff` exactly 0.0.
- **A guard must name every dependency it guards.** An install block gated on
  `import torch, transformers, safetensors, huggingface_hub` silently skipped
  `hf_transfer` on any host that pre-shipped the first four.
- **Watch run STATE, not output counts.** A failed remote run leaves its box idle
  but *running*; a stalled file counter looks exactly like slow progress.
- **Read identity from bytes.** Scope, rate, profile and tensor inventory come
  from artifact metadata, never from a repo name or a filename.

## Commands that define "done"

```bash
bash bin/selftest_all.sh          # the full local battery; must be 0 failed (82 passed today)
cd registry && make check         # schema + invariants + render drift + selftests; must be 0 errors
python3 bin/selftest_<tool>.py    # per-tool suites; run the ones you touched
```

Run the closest test first, then the battery. It is spend-free and GPU-free but
not hermetic: some sections use network metadata, read-only account queries, or a
cached fixture. Read the internal `SKIP` lines — an outer PASS can still hide an
optional rung. Green means green: no new skips, no gate weakened to pass. If you
fix a defect, add a regression test that **fails without your fix** — and verify
that by reverting it in a scratch copy, not by assuming.

Contract-specific gates on top of the battery: `bin/check_doc_numbers.py` for a
numeric doc or card change; `bin/selftest_naming_sweep.py` before any rename;
`selftest_container.py` + `selftest_bundle_complete.py` for container, bootstrap
or bundle changes; the matching `engines/tools/selftest_*_offline.py` plus
committed real-tensor parity for an engine or surface; `--probe-engines` for any
edit to `bin/engines.json`; and in `registry/`, `make reseed-check` for
receipt-derived rows and `make stat-selftest` for interval statistics.

Read-only and planning commands, safe to run freely:

```bash
bin/measure <hf-repo-or-url> --plan-only
bin/measure-local --artifact <repo> --panel <dataset> --estimate-only
bin/measure-cloud --model <repo> --panel <dataset> --lane streaming \
    --max-cost <usd> --max-runtime <duration> --dry-run
bin/registry-view rows --model <name> --lane <lane> --registry local
bin/registry-submit <receipt.json>                     # validates; publishes nothing
container/build.sh --tag quant-fidelity-measure:dev    # docker or podman; refuses a dirty tree
python3 bin/changelog.py --all --out CHANGELOG.md      # CHANGELOG.md is generated
```

There is no root build, package manifest, lockfile, lint, format or type-check
target, and no pytest discovery — tests are executable selftests. Do not invent a
second toolchain; match the file you are editing.

## Dependency discipline

- `bin/` controller paths and all of `registry/` must run on **stock python3.9
  with no installs**. The registry vendors `_minischema.py` precisely so a
  contributor needs nothing. Optional dataset/card/engine modes may lazily import
  PyYAML, NumPy, torch or safetensors; do not pull those into a startup path.
- Torch-dependent local engines run under `FIDELITY_PYTHON` (default: homebrew
  `/opt/homebrew/bin/python3.14` when present, else `python3`). Use a venv or an
  explicit interpreter rather than changing system packages.
- The paid CUDA environment is **python3.12 only** and `bin/bootstrap_measure.sh`
  — not a requirements file — is its install contract; `container/Dockerfile`
  bakes the same recipe. Pins live in the script and the build metadata.
- MPS cannot do float64 at all — it raises. KLD accumulation pins to CPU.
- The stdlib rule is about `bin/` and `registry/`. It is **not** a licence to
  argue "no dependencies" anywhere else: `bootstrap_measure.sh` installs torch,
  transformers, accelerate and `rich` on the instance, so `engines/tools/` engines
  already run inside a stack. Before rejecting a library, check whether it is
  *already transitively installed* — that argument has been made here and been
  wrong. [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md) is the audit of every
  hand-rolled component, with the verdict and the reason for each.

## Numerical rules that are not negotiable

- Full vocabulary, fp64 accumulation, direction KLD(reference ‖ candidate).
  No top-k, ever. Never clamp a non-finite value into plausibility — refuse.
- **Never compare a single window** to rank two artifacts: per-window scatter
  (sd ≈ 1.7e-3) exceeds the effect between adjacent bit-widths (≈ 1.2e-3).
  Previews prove liveness, not quality.
- **Never subtract a floor from a different lane.** Invariant `BIAS-006` refuses
  it; do not route around the validator.
- Rank only within a group whose recomputed `comparability.key` values match, and
  check `index.json`'s per-group predicate — an equal key alone is not a licence.
- A decode surface must be proven **bitwise** against the ecosystem reference
  implementation (mlx.core, gguf-py, compressed-tensors, exllamav3) on real
  fetched tensors before it ships.

## Money and rented machines

`bin/measure-cloud` spends real money on someone's account, on any of four
providers (`--provider {jarvislabs,runpod,vast,lambda}`).

- Run it only when asked, and start with `--dry-run`, an explicit `--max-cost`
  and a realistic `--max-runtime`. `--dry-run` creates nothing at all.
- Teardown must be guaranteed on success, failure, exception and interrupt, with
  the on-instance watchdog as backstop. Never weaken that path.
- A leaked instance is a blocker-level defect. Verify with `jl list` afterwards.
- Budget for the **measurement** phase, not just compute: each cold run writes
  ~32 GB of fp32 logits, and runs are kept for the determinism check.
- Never create, pause, or destroy a machine you did not create.

Recipes and per-provider detail: [`docs/CLOUD-RECIPES.md`](docs/CLOUD-RECIPES.md).
Running a model whose modeling code ships in its repo:
[`docs/REMOTE-CODE-POLICY.md`](docs/REMOTE-CODE-POLICY.md).

## Secrets

Read the HF token from a file; never echo it, never put it in argv, a log, a
receipt, a bundle, or git. `measure-cloud` transports it as a 0600 file and
shreds it at teardown — match that standard. Never `set -x` in a shell path that
could see a credential. Before publishing any artifact, grep it for credentials
**and for private absolute paths**: a published receipt once pointed at
`/home/jl_fs/...` on a filesystem that no longer exists.

## Concurrency: this repo has multiple agents in it

Several workflows may be editing simultaneously.

- `git pull --rebase origin main` before every commit.
- **Stage only the files you changed. Never `git add -A`.**
- If another workflow owns a file (a live measurement campaign owns the runner
  files), review it read-only and write your patch into `docs/REVIEW-DEFERRED.md`
  instead of editing it.
- Box and repo copies of a script have drifted before, and a downstream agent
  then "verified" a CLI that did not exist. After any on-box fix, pull it back
  into git the same day.

## Publishing

Publishing to HF, GitHub or GHCR is outward-facing and requires explicit
approval. Model cards, datasets and registry mirrors are the user's public record.

- Never publish a number you cannot trace to a receipt. If an experiment did not
  run, publish nothing — an honest "blocked, here is why" beats a receipt of
  invented metrics.
- Changing a published number requires quantifying the delta and disclosing it
  additively, not editing history.
- Third-party numbers stay visibly third-party: `measured_by` is enumerated and
  the validator refuses conflation.
- `registry/data/*.jsonl`, `registry/index.json` and generated README tables are
  derived. Change the receipt, schema or tool and regenerate; never hand-edit a
  published number.
- The contributor path is [`registry/CONTRIBUTING.md`](registry/CONTRIBUTING.md),
  and the unaided walkthrough is
  [`docs/THIRD-PARTY-QUICKSTART.md`](docs/THIRD-PARTY-QUICKSTART.md). Point people
  at those rather than restating them — a duplicated contract is a contract that
  drifts.

## How code in here is written

- Python: four spaces, `snake_case`, `pathlib.Path`, small dataclasses for durable
  concepts, `argparse`, `main(...) -> int`, `raise SystemExit(main())`. Shell uses
  `set -euo pipefail`, explicit traps and quoted paths. Match the surrounding
  file; do not mass-restyle older registry code.
- User commands are hyphenated wrappers (`measure-cloud`); implementations are
  underscore modules (`measure_cloud.py`); tests are `selftest_<feature>.py`.
  Provenance-bearing fields carry explicit suffixes (`_ref`, `_revision`,
  `_sha256`, `_schema`, `_bytes`, `_gb`), and schema strings are versioned.
- **An expected invalid state is a refusal, not a guess**: `Refusal(reason,
  advice)` in controllers, tool-prefixed `_fail()` in engines, coded
  `Refuse(code, message, remedy)` in registry ingestion. Preserve the exit codes
  and give an actionable remedy. Registry validation accumulates findings in a
  `Report` — do not stop at the first and hide the rest.
- Write structured artifacts atomically when they can be interrupted, and keep
  plans, jobs and receipts self-describing: missing provenance is never inferred
  later. Provider objects, `Console`, config paths, environment roots and
  simulated devices are the injection seams; tests use stubs and scratch dirs.
  There is no DI container and no global state store.

## Where the hard-won detail lives

- [`JOURNAL.md`](JOURNAL.md) — the append-only campaign ledger, 63 dated entries,
  each written at the milestone and never edited after the fact.
- [`docs/NAMING-SWEEP.md`](docs/NAMING-SWEEP.md) — which names in this tree
  are incidental and which are IDENTITY. Registry ids are hashed into
  `comparability.key`, receipt schema strings sit inside sealed receipts, and
  a published row's `harness.code_digests[].path` is inside `harness_id`.
  Read it before renaming anything; `bin/selftest_naming_sweep.py` enforces it.
- [`engines/HANDOFF.md`](engines/HANDOFF.md) — 24 numbered operational lessons for
  running a campaign.
- [`docs/FIDELITY-DATASET-SPEC.md`](docs/FIDELITY-DATASET-SPEC.md) and
  [`docs/CARD-ANNOTATION-SPEC.md`](docs/CARD-ANNOTATION-SPEC.md) — frozen public
  wire formats; evolve them additively.
  [`docs/PUBLISHED-CORRECTIONS.md`](docs/PUBLISHED-CORRECTIONS.md) is how a
  published number gets corrected without rewriting history.
- [`docs/CAPTURE-SCALING-PLAN.md`](docs/CAPTURE-SCALING-PLAN.md) — plan of record
  for scaling a capture: the parallelism decision (tensor-parallel changes the
  numbers and is rejected), the cost model, and per-family budgets.
  [`docs/CONTAINER.md`](docs/CONTAINER.md) is why the image exists.
- [`docs/`](docs/) and [`bin/README.md`](bin/README.md) — everything else. Both
  carry historical status; verify against `--help` and the code.
