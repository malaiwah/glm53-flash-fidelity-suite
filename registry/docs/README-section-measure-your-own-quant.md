<!--
SPLICE TARGET: registry/README.md
Insert as a top-level section, after the "What is in here" / collections section
and before the schema reference. Verbatim; do not summarize it further.
-->

## Measure your own quant

Every number in this registry was produced by a recipe you can run yourself, on
someone else's weights, and submit. That is the point of it.

**Cloud — one paste, ~$10–15 at spot, tears the instance down for you:**

```bash
export JL_API_KEY=...        # your key; never logged, never written to a receipt
./bin/measure-cloud \
    --model brandonmusic/GLM-5.3-Flash-tr3-4bpw \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane  streaming --spot --max-runtime 8h
```

It prints a cost estimate and waits for your confirmation, creates the instance,
fetches weights and panel, runs the measurement, seals the receipt, pulls it
back to your machine, destroys the instance — including on failure or Ctrl-C —
and prints what it actually cost, four different ways.

Add `--dry-run` to see all of that and create nothing. It resolves the repo to
an immutable commit, sizes the instance, prices the run and refuses anything
that will not fit — for the cost of a few hundred kilobytes of metadata.

**Local — same measurement, your hardware.** 128 GB Apple Silicon via MPS, or a
consumer CUDA card under a hard VRAM budget:

```bash
./bin/measure-local \
    --artifact <hf-repo> \
    --panel    brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --vram-budget 30
```

It tells you up front what it needs — disk, RAM, VRAM schedule and hours, with
each number's provenance — and refuses with advice rather than thrashing if it
will not fit. The hours come from a five-second benchmark of the real decode on
*your* machine, not from a table of GPUs somebody once owned. `--estimate-only`
stops after the plan. `--simulate-device "RTX 5090:32"` plans for hardware you
do not have yet.

**Then submit it.** One file, one discussion, no git required:

> <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
> → **New discussion** → title `submission: <repo> on <panel>` → paste your
> `submission.json`.

A GitHub pull request against the mirror works too. Both paths, the exact
template, what gets bounced and how you are credited:
[CONTRIBUTING.md](CONTRIBUTING.md). A real, sealed, schema-valid example:
[`docs/examples/dione-q4.submission.json`](docs/examples/dione-q4.submission.json).

**Before you start:** the panel and the teacher capture you score against have
to exist in the registry already. Every panel in `data/panels.jsonl` is fair
game. If you need a new one, open a `panel: <name>` discussion first —
CONTRIBUTING.md §6 says what to put in it.
