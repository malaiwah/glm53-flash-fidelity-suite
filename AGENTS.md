# AGENTS.md — working on this repository

You are editing code that produces published scientific claims. Numbers from this
tree are on Hugging Face model cards, in a public registry other people query, and
in a community standards discussion. A silently wrong number here is worse than a
crash, because a crash gets fixed and a wrong number gets cited.

If you are here to *use* the yardstick rather than change it, read
[`llms.txt`](llms.txt) instead — it carries the rules that decide whether two
numbers may be compared.

## Verify before you claim

The single most expensive failure mode in this project's history is an agent
believing a document, a docstring, or another agent instead of running the thing.

- **Probe CLIs, never read their docs.** A runner was once written against a
  scorer's *documented* flags; five of them did not exist, and the lane could not
  run at all. `--help` is cheap.
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

## Commands that define "done"

```bash
bash bin/selftest_all.sh          # the full local battery; must be 0 failed
cd registry && make check         # schema + invariants + generated tables; must be 0 errors
python3 bin/selftest_<tool>.py    # per-tool suites; run the ones you touched
```

Green means green: no new skips introduced, no test weakened to pass. If you fix
a defect, add a regression test that **fails without your fix** — and verify that
by reverting it in a scratch copy, not by assuming.

## Dependency discipline

- `bin/` tools and `registry/` must run on **stock python3.9 with no installs**.
  The registry vendors `_minischema.py` precisely so a contributor needs nothing.
  Do not add a third-party import to those paths.
- Torch-dependent code (the scorer, decode surfaces) may use the homebrew
  python3.14 environment. Keep the split clean.
- MPS cannot do float64 at all — it raises. KLD accumulation pins to CPU.

## Numerical rules that are not negotiable

- Full vocabulary, fp64 accumulation, direction KLD(reference ‖ candidate).
  No top-k, ever.
- **Never compare a single window** to rank two artifacts: per-window scatter
  (sd ≈ 1.7e-3) exceeds the effect between adjacent bit-widths (≈ 1.2e-3).
- **Never subtract a floor from a different lane.** Invariant `BIAS-006` refuses
  it; do not route around the validator.
- A decode surface must be proven **bitwise** against the ecosystem reference
  implementation (mlx.core, gguf-py, compressed-tensors, exllamav3) on real
  fetched tensors before it ships.

## Money and rented machines

`bin/measure-cloud` spends real money on someone's account.

- Teardown must be guaranteed on success, failure, exception and interrupt, with
  the on-instance watchdog as backstop. Never weaken that path.
- A leaked instance is a blocker-level defect. Verify with `jl list` afterwards.
- Budget for the **measurement** phase, not just compute: each cold run writes
  ~32 GB of fp32 logits, and runs are kept for the determinism check.
- Never create, pause, or destroy a machine you did not create.

## Secrets

Read the HF token from a file; never echo it, never put it in argv, a log, a
receipt, or git. `measure-cloud` transports it as a 0600 file and shreds it at
teardown — match that standard. Before publishing any artifact, grep it for
credentials **and for private absolute paths**: a published receipt once pointed
at `/home/jl_fs/...` on a filesystem that no longer exists.

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

Publishing to HF or GitHub is outward-facing. Model cards, datasets and registry
mirrors are the user's public record.

- Never publish a number you cannot trace to a receipt. If an experiment did not
  run, publish nothing — an honest "blocked, here is why" beats a receipt of
  invented metrics.
- Changing a published number requires quantifying the delta and disclosing it,
  not editing history.
- Third-party numbers stay visibly third-party: `measured_by` is enumerated and
  the validator refuses conflation.

## Where the hard-won detail lives

- [`JOURNAL.md`](JOURNAL.md) — 38 numbered lessons, each from a real failure.
- [`k6/HANDOFF.md`](k6/HANDOFF.md) — twenty operational lessons for running a campaign.
- [`WHAT-WE-MEASURE.md`](WHAT-WE-MEASURE.md) — what a number actually is.
- [`docs/`](docs/) — the dataset format spec, card annotation spec, protocol alignment.
- [`docs/CAPTURE-SCALING-PLAN.md`](docs/CAPTURE-SCALING-PLAN.md) — plan of record for scaling a capture: the parallelism decision (tensor-parallel changes the numbers and is rejected), the cost model, and per-family budgets.
