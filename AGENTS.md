# Repository Guidelines

## Project Overview

`quant-fidelity-suite` is a receipt-backed scientific measurement system for quantization fidelity. It measures full-vocabulary KL divergence between an artifact and a pinned reference, records the exact panel/runtime/scope, and publishes only schema-validated rows whose provenance can be traced to sealed receipts.

A plausible wrong number is worse than a crash. Fail closed rather than infer a revision, surface, scope, profile, lane, dependency, or metric. For comparison/ranking work, read `llms.txt` and `WHAT-WE-MEASURE.md` first; rows are comparable only when their recomputed `comparability.key` values match.

## Architecture & Data Flow

There is no conventional `src/` tree or application server. The product is a set of Python/Bash CLIs with filesystem/JSON state:

1. **Resolve and gate:** `bin/measure` wraps `bin/measure_one.py`. `run()` parses an HF target, checks the registry before work or spend, resolves a 40-hex revision, walks lineage, picks panel/reference precedent, sniffs the storage surface, and selects a lane/profile.
2. **Plan:** `bin/measure_local.py` combines `bin/fidelity/{hfmeta,lineage,registry_client,census,engines}.py` to solve identity, device, memory, disk, and invocation constraints. `bin/measure_cloud.py::plan` adds provider capacity, cost/runtime limits, leases, and teardown backstops.
3. **Execute:** `bin/engines.json` is the authored lane-to-entrypoint/profile/flag contract. `bin/fidelity/engines.py::build_invocation` maps lane-neutral values to the probed engine CLI. Cloud jobs persist `job.json`, then `bin/fidelity/stages.py` drives `bin/stage_measure.sh` → `invoke_engine.py` / `invoke_scorer.py`.
4. **Measure and score:** `engines/tools/student_capture.py` is the distributed sealed capture; `engines/tools/stream_score.py` is the single-device streaming capture. Surface adapters decode storage formats. `engines/tools/kld_report.py` verifies teacher/candidate identity, computes tokenwise `KL(reference || candidate)` in fp64, and aggregates cold runs.
5. **Seal:** `bin/fidelity/receipt.py` and `bin/seal_receipt.py` reject preview/teacher material, bind code and artifact digests, and emit a self-sealed submission receipt.
6. **Ingest and publish:** `bin/registry-submit RECEIPT` validates offline; it does not publish. Maintainer ingestion saves the receipt under `registry/receipts/<handle>/`, derives records with `registry/tools/registry_add.py`, checks them with `registry_validate.py`, and renders `registry/data/*.jsonl`, `index.json`, and `README.md` with `registry_render.py`.

The portable dataset route is parallel: `bin/fidelity-dataset capture` creates sealed root/candidate datasets, `verify` checks content and seals, `compare` creates evidence plus an optional submission claim, and `publish` uploads then fetches back and verifies. Model-card publication is a separate, permissioned step after registry identity exists.

**Architectural patterns:**

- Orchestration belongs in `bin/`; reusable controller policy in `bin/fidelity/`; tensor-heavy engines in `engines/tools/`; publication rules in `registry/tools/` and `registry/schema/`.
- State is explicit JSON/files: plans, jobs, leases, stage logs/`.done` markers, captures, reports, and receipts. A `.done` marker appears only after success. Watch provider/run state, not output-file counts.
- CLIs are synchronous. Cloud work uses detached remote stages plus polling, a daemon heartbeat thread, and locked teardown; streaming uses bounded worker threads for I/O; distributed capture uses process-level expert parallelism. Do not introduce an async framework without a demonstrated need.
- Provider objects, `Console`, config paths, environment roots, panel descriptors, and simulated devices are explicit injection seams. Tests use stubs and scratch files; there is no dependency-injection container or global state store.

**Known sharp edges:** documentation contains historical status and stale flag examples. Probe `--help` and `bin/measure-local --probe-engines`; trust executable behavior and receipts over prose. `measure-local` is plan-only unless `--execute` is supplied, and its documented schedule/surface routing has source-level contradictions—an accepted plan is not proof that the engine can execute that surface. Always record an explicit lane; validator and renderer handling of an undeclared lane is not uniform.

## Key Directories

| Path | Purpose and boundary |
|---|---|
| `bin/` | User CLIs, local/cloud controllers, receipt/dataset/card tooling, regressions, and the on-box upload manifest. Orchestrates; does not implement model math. |
| `bin/fidelity/` | Shared stdlib-first policy: HF identity, lineage, registry front gate, fit arithmetic, engines, stages, provider adapters, receipts, and dataset formats. |
| `engines/tools/` | Capture/scoring engines and storage-surface adapters (`*_surface.py`) plus real-tensor parity evidence and offline selftests. |
| `engines/` | Quantization campaign recipes, panels, upstream patch series, stage driver, and operational evidence. Some material is deliberately model/campaign-specific. |
| `registry/` | Offline schemas/invariants, sealed receipts, generated records/index/README, frozen protocols, and add/validate/render tools. |
| `docs/` | Frozen dataset/card contracts, operational plans, scientific analyses, additive corrections, and generated examples. Status prose can age; schemas and receipts are stronger evidence. |
| `container/` | Reproducible measurement image and Docker/Podman build wrapper. |
| `suite/`, `calsuite/` | Committed suite/calibration manifests; token payloads are generated/ignored. |
| `reports/`, `registry/protocol/` | Receipt-backed experimental evidence and frozen/derived protocol inputs. Cite these rather than copying prose numbers. |
| `tools/`, `remote/` | Original GLM serving-lane harness and VM campaign pipeline. Historical/campaign-specific, not the current generic runner. |
| `port/` | Draft native exllamav3 architecture port and manual parity harnesses; not production code. |

## Development Commands

Probe real interfaces before changing or documenting them:

```bash
bin/measure --help
bin/measure-local --help
bin/measure-cloud --help
bin/registry-view --help
bin/measure-local --probe-engines
```

Safe planning and read-only use:

```bash
bin/measure <hf-repo-or-url> --plan-only
bin/measure-local --artifact <repo> --panel <dataset> --estimate-only
bin/measure-cloud --model <repo> --panel <dataset> --lane streaming \
  --max-cost <usd> --max-runtime <duration> --dry-run
bin/registry-view rows --model <name> --lane <lane> --registry local
bin/registry-submit <receipt.json>       # validation only; publishes nothing
```

Build and generated outputs:

```bash
container/build.sh --tag quant-fidelity-measure:dev   # Docker or Podman; clean tree expected
python3 bin/changelog.py --all --out CHANGELOG.md     # CHANGELOG.md is generated
(cd registry && make render)                          # generated README/index from registry data
```

There is no root build, package-install, lint, format, or static-type-check target. Do not invent a second style/toolchain. Match the file being edited and use the behavioral gates below.

Required verification:

```bash
python3 bin/selftest_<tool>.py    # targeted regression for the changed tool
bash bin/selftest_all.sh          # full local battery; require 0 failed
(cd registry && make check)       # schema, invariants, render drift, selftests, joint checks
```

Additional contract-specific gates include `python3 bin/check_doc_numbers.py`, `python3 bin/selftest_naming_sweep.py`, `python3 bin/selftest_container.py`, the matching `engines/tools/selftest_*_offline.py`, and NumPy-dependent registry targets such as `make reseed-check` or `make stat-selftest`.

## Code Conventions & Common Patterns

### Python and shell

- Python files use four spaces, `snake_case`, uppercase constants, `pathlib.Path`, small dataclasses for durable concepts, `argparse`, `main(...)->int`, and `raise SystemExit(main())`. Match older registry style where present; do not mass-restyle.
- User commands are hyphenated executable wrappers (`measure-cloud`); implementations are underscore modules (`measure_cloud.py`). Tests are `selftest_<feature>.py` or `<tool>_selftest.py`.
- Provenance-bearing fields use explicit suffixes such as `_ref`, `_revision`, `_sha256`, `_schema`, `_bytes`, and `_gb`. Schema strings are namespaced and versioned.
- Shell orchestration normally uses `set -euo pipefail`, explicit traps, quoted paths, and uppercase environment variables. Never enable `set -x` where credentials may exist.

### Errors, state, and dependencies

- Expected invalid states are **refusals**, not guesses: `Refusal(reason, advice)` in controllers, tool-prefixed `_fail()` errors in engines, and coded `Refuse(code, message, remedy)` in registry ingestion. Preserve stable exit codes and actionable remedies; fail closed on unknown formats, schemas, profiles, identities, dependencies, or arithmetic.
- Registry validation accumulates independent findings in `Report`; do not stop after the first issue and hide the rest.
- Write structured artifacts atomically when they can be interrupted. Keep plans/jobs/receipts self-describing; never infer missing provenance later.
- Dependency guards must name every dependency they guard. Do not gate several imports with a check that can silently skip installing the one newly required package.
- Reuse existing helpers and canonical serializers. Deliberate duplication across independently shipped trees must remain byte/behavior compatible and is guarded by selftests.

### Scientific and identity invariants

- Full vocabulary only; fp64 log-softmax/accumulation; direction `KL(reference || candidate)`. Never top-k, clamp non-finite values into plausibility, or move fp64 KLD to MPS.
- Compare/rank only equal recomputed `comparability.key` groups. Never rank a single window; previews prove liveness, not quality. Subtract only a same-lane floor (`BIAS-006`).
- Hash tensor **content**, not receipt/safetensors/container bytes: timestamps and metadata may differ for identical computation. Determinism claims require content evidence and repeated cold runs.
- Read scope, rate, profile, and tensor inventory from artifact bytes/metadata; do not infer them from a repo or filename. New decode surfaces require bitwise parity with the ecosystem reference on real fetched tensors before shipping.
- Canonical JSON, registry IDs, schema strings, sealed receipt bytes, profile labels, protocol filenames, and provenance paths are published identity. Read `docs/NAMING-SWEEP.md` before renaming; `harness.code_digests[].path` participates in `harness_id`.
- `registry/data/*.jsonl`, `registry/index.json`, and generated README/table blocks are derived. Change receipts/schema/tooling and regenerate; never hand-edit a published number.

### Security, cloud, publishing, and collaboration

- `bin/measure-cloud` spends real money. Run it only when requested; start with `--dry-run`, an explicit `--max-cost`, and a realistic `--max-runtime`. Preserve teardown on success, failure, exceptions, and signals; the watchdog is only a backstop. After real work, verify account state with `jl list`. Never alter a machine the workflow did not create.
- Tokens belong in mode-0600 files, never argv, logs, receipts, bundles, or git. Shred transported secrets during teardown. Before publication, reject credentials and private absolute paths.
- Publish no metric without its receipt. Published-number changes require a measured delta and additive disclosure; keep third-party `measured_by` attribution distinct. Publishing to HF, GitHub, or GHCR requires explicit approval.
- Multiple agents may work concurrently. Before committing, `git pull --rebase origin main`; stage only files you changed, never `git add -A`. If a live campaign owns a runner file, review it read-only and record the patch in `docs/REVIEW-DEFERRED.md`. Sync on-box fixes back into git promptly.

## Important Files

| File | Why it matters |
|---|---|
| `llms.txt` | Short assistant-facing comparison rules and machine-readable registry links. |
| `WHAT-WE-MEASURE.md` | Normative scientific meaning of the metric, scope, lanes, floors, and comparability tuple. |
| `bin/README.md` | Detailed CLI intent and architecture history; verify current status against `--help` and code. |
| `bin/engines.json` | Authoritative lane/entrypoint/flag/profile/surface/receipt-class contract. |
| `bin/measure_one.py`, `bin/measure_local.py`, `bin/measure_cloud.py` | One-link front end and local/cloud planning/execution entry points. |
| `bin/fidelity/stages.py`, `bin/stage_measure.sh` | Shared stage graph and on-instance state machine. |
| `engines/tools/stream_score.py`, `engines/tools/student_capture.py`, `engines/tools/kld_report.py` | Current capture and numerical scoring boundary. |
| `bin/BUNDLE.txt`, `bin/bootstrap_measure.sh` | Exact on-box shipment and Python 3.12 measurement environment recipe. |
| `registry/schema/invariants.json`, `registry/Makefile` | Machine-enforced publication rules and authoritative local registry gates. |
| `docs/FIDELITY-DATASET-SPEC.md`, `docs/CARD-ANNOTATION-SPEC.md` | Frozen public wire formats; evolve additively. |
| `docs/DEPENDENCIES.md`, `docs/NAMING-SWEEP.md` | Dependency decisions and incidental-name versus published-identity rules. |
| `JOURNAL.md`, `docs/PUBLISHED-CORRECTIONS.md` | Append-only failure/decision history and additive public corrections; do not rewrite history. |

## Runtime/Tooling Preferences

- Primary languages: CPython and Bash. No Node/Bun package, Rust/Go build, project package manifest, lockfile, or repository-wide formatter/linter/type checker exists.
- Keep `bin/` controller/validation paths and all `registry/` tooling compatible with stock Python 3.9 and the standard library. Optional dataset/card/engine modes may lazily require PyYAML, NumPy, Torch, safetensors, or provider CLIs; do not pull those imports into stdlib-only startup paths.
- The paid CUDA measurement environment is Python **3.12 only**, built by `bin/bootstrap_measure.sh`; `container/Dockerfile` uses Ubuntu 24.04. The script—not a requirements file—is the install contract and pins Torch/Transformers plus the external pipeline/patch series. Build metadata records the actual resolution.
- Torch-dependent local engines run under `FIDELITY_PYTHON`. The current fallback prefers `/opt/homebrew/bin/python3.14` when present, then `python3`. Use a venv or explicit interpreter rather than changing system packages blindly.
- MPS cannot perform float64; KLD stays on CPU. Hardware/architecture and replay backend affect identity and must be recorded, not normalized away.
- Container builds use `container/build.sh` with Docker or Podman. JarvisLabs access uses the external `jl` CLI (normally installed with `uv tool install jarvislabs` or pipx); other providers use their adapters.
- Root GitHub Actions covers the container path only. The nested registry workflow describes a not-yet-live standalone mirror and does not replace local `make check`; ordinary source changes must be verified locally.

## Testing & QA

- Tests are executable Python/Bash selftests, not pytest/unittest discovery. They use known answers, exact exit codes, mutation/FIRE cases, subprocess stubs, temporary directories, deterministic seeds, and bitwise tensor checks. No source-code coverage threshold exists; `(cd registry && make coverage)` measures scientific interval coverage, not code coverage.
- Every defect fix needs a regression that fails without the fix. Prove this by reverting the fix in a scratch copy, then restore it. Test observable behavior; source-text checks are not a substitute for executing a stage or CLI.
- Run the closest test first, then `bash bin/selftest_all.sh`. The aggregate is spend-free/GPU-free but not fully hermetic: some sections use network metadata, read-only account queries, or a cached/fetched fixture. Inspect internal `SKIP` lines—an outer PASS or zero failures can still hide optional rungs. Introduce no new skips and never weaken a gate.
- Registry changes require `(cd registry && make check)`. Add `make reseed-check` for receipt-derived-row changes and `make stat-selftest` for interval/statistics changes. Use `make validate-both` only when external `jsonschema` is installed.
- Engine or surface changes require the matching offline selftest plus upstream-reference parity on committed real-tensor evidence; pipeline-dependent parity must run on the actual measurement box before capture. Changes to `bin/engines.json` additionally require `bin/measure-local --probe-engines`.
- Numeric documentation/card changes require `python3 bin/check_doc_numbers.py` and the relevant card/joint-standard selftest. Container/bootstrap/bundle changes require `selftest_container.py`, `selftest_bundle_complete.py`, and/or `selftest_shell_guards.sh` as applicable.
- Green means zero failures with expected evidence present. Do not treat an optional skipped parity rung, a dry plan, or generated-file counts as proof of end-to-end behavior.
