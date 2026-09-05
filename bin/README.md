# `bin/` — the recipes

`measure`, `measure-cloud` and `measure-local` are the user-facing product: a
stranger with a quant and a GPU should be able to paste one line and get a
sealed, submittable number — or the honest answer that the number already
exists, or a refusal that names its arithmetic. `registry-view` browses the
public registry from the CLI; `registry-submit` checks a sealed receipt the
way the registry will, before it is sent anywhere.

## Why here and not in `engines/tools/`

`engines/tools/` holds the ENGINES — the things that read a checkpoint and emit
logits. Every one of them is model-agnostic now (they have run GLM-5.3-Flash,
Qwen3.8-27B, Fruit, MiniMax-M3 and DeepSeek V4), which is why the directory is
no longer called `k6/`. But an engine is still a thing you point at a
checkpoint, and the runners here are the thing a stranger types. Putting them
at the top level alongside `registry/` is what makes the pair read as a product
rather than as internal tooling.

The measurement **engines** stay where they are. `bin/` orchestrates; it does
not measure. That split is enforced by `engines.json`.

## The one-command flow (`bin/measure`)

```bash
bin/measure <hf-url-or-repo>[@rev][/subpath] [--plan-only] [--force] [--path SUB]
```

Nine steps, one status line each; every refusal states its remedy, exit codes
`0` (already-measured report OR completed measurement/preview), `3` (refusal),
`4` (no data source), never a stack trace:

1. **parse** the target (URL, `org/name`, `@rev`, `/tree/rev`, trailing
   subpath → `--path` hint for multi-artifact repos);
2. **load the registry** (public HF dataset first, local clone fallback,
   snapshot printed);
3. **resolve the revision** (default: the live head — what a download today
   fetches; HF's 401-for-nonexistent is reported as the three-way it is:
   gone/private/gated are indistinguishable unauthenticated);
4. **already measured?** → print the rows + receipt links and exit 0
   (`--force` to measure anyway; revision drift needs `--force` or
   `--accept-measured-revision`);
5. **lineage**: walk `base_model` tags to a root the registry knows (both zai
   roots — FP8 and BF16 — land on the same model);
6. **panel + teacher**: the pair prior measurements of that model used, with
   the alternatives and their `--panel/--teacher` overrides printed;
7. **surface**: a repo no lane can read (e.g. `tr3-published`, MLX) is
   refused HERE, for $0.00, naming the missing reader;
8. **lane**: `local-mps` on Apple Silicon, `local-cuda-budget` otherwise;
   `--lane streaming` is redirected to `measure-cloud` (bin/measure never
   rents);
9. **hand off** to `measure-local --execute` (or `--estimate-only` with
   `--plan-only`), whose preflight verifies FIDELITY_PYTHON, torch,
   transformers>=5.16, quant_pipeline, the teacher tree and disk FIRST and
   refuses with *all* missing prerequisites and their remedies at once.

## Registry lookup & viewer (`bin/registry-view`)

```bash
bin/registry-view check  zai-org/GLM-5.3-Flash          # STALE by default: live head vs pinned rev
bin/registry-view rows   --model glm --lane streaming   # filtered tables
bin/registry-view lineage 0xSero/GLM-5.3-Flash-EXL3-Q4  # walk + panel/teacher pick
```

Rendering rules (reproduced from `registry/tools/registry_render.py`, the
normative reference): rows are grouped by the **recomputed** comparability key
— never the stored block — one table per key; named-lane rows are tabled
apart from no-declared-lane rows (None means "no declared lane", NOT
"sealed"); sorting only within a lane sub-table; a filter may HIDE groups but
never MERGE them; single-row groups say "nothing to rank against"; subset
panels always carry their caveat; the footer names the snapshot that answered
(dataset commit sha or local git HEAD). `check` tiers artifacts EXACT /
UNPINNED / STALE / PINNED-UNVERIFIED and quotes the `revision_unpinned`
disclosure verbatim. Data sources: `--registry auto|hf|local[:PATH]` — `check`
prefers the published mirror, `rows`/`lineage` prefer the offline clone.

## Preview scoring (`bin/kld-preview`)

Scores capture trees locally, labeled honestly. CENSUS mode (a sealed full
capture): every position scored exactly with the sealed fp64 math — 0.15
ms/position on CPU, ~8 s/panel, so **scoring never motivates sampling**; the
receipt is a preview only because the lane differs from the teacher's.
SAMPLED mode (a `--store-positions per-window:<m>` capture): teacher rows
sliced at the student's stored indices, stratified estimator with FPC, quoted
CI the WIDER of z and bootstrap, tail disclosure (max sampled value + top-3
share — with a printed warning when the estimate is tail-dominated, because
on heavy tails both intervals are anti-conservative and the remedy is more
positions).

The 25-window rule is structural: no panel estimate unless ALL windows
contributed, because per-window KLD scatter (sd 7.2e-3) — and even the paired
per-window delta (sd 2.0e-3) — exceeds the K6-vs-K8 effect (1.33e-3), so a
single window has no power to compare quants (lessons
28/29). Window subsets get per-window diagnostics only.

Preview receipts are **structurally unsubmittable on two independent axes**:
(1) bin-side — schema contains `-preview.`, headline field is
`preview_panel_mean_estimate` (never `measured_mean_kld`),
`not_submittable: true`, and `fidelity/receipt.build_submission` refuses all
three markers anywhere in its inputs; (2) registry-side — no
`submission_schema` key (the validator's const gate refuses) and no
`registry_add` adapter accepts a `-preview.` schema string (demonstrated
live: exit 3 naming the string). Position sampling is a storage/teacher-
bandwidth knob, not a compute knob: the causal trunk runs every position
regardless.

## Floor-aware stats (`bin/fidelity-stats`)

`attributable`: quant panel mean minus the SAME-lane floor, gated on
`teacher_receipt_sha256` identity. The canonical cross-lane refusal does the
arithmetic for you: subtracting the cross-stack floor 0.012712 from the
streaming K8 mean gives 0.012384 − 0.012712 = **−0.000328** — a negative
attributable for an 8-bit quant, arithmetic proof the floors are not
interchangeable; the same-lane floor gives 0.012384 − 0.011506 = +0.000878.
`engines/BF16-FLOOR.json` (the analysis) is refused as a floor input by name, and
"floor = 0" is accepted only with T1 hash evidence
(`engines/SAME-LANE-TEACHER.md`). `--from-registry MEASUREMENT_ID` fetches the
public receipt and gates on its real teacher sha.

`paired-delta`: the honest CI for a two-run difference — paired per-window
deltas, exact t (incomplete-beta), BCa bootstrap over windows, exact sign
test, Wilcoxon; refuses cross-teacher/cross-panel pairs. The printed estimand
statement is mandatory: the census difference itself is exact (deterministic
lane); the CI answers generalization to new windows and must never be
presented as measurement noise. Design constants at n=25: paired SE 3.47e-4,
MDE ~1.01e-3 — effects below ~1e-3 need more text, not more runs.

## Engine pinning

A lane whose engine is not `pinned: true` in `engines.json` **refuses to
plan**. It does not guess flags.

**All five lanes are now pinned**: `sealed-ep8` (against
`engines/tools/k6_student_capture.py`), and `bf16-floor`, `streaming`,
`local-mps`, `local-cuda-budget` (all against `engines/tools/stream_score.py`,
every required flag verified by `bin/measure-local --probe-engines` in the
real file). The 2026-08-29 reconciliation was done by PROBING the CLI (AST
scrape of the argparse declarations), never by reading docs. Planner-only
knobs (`--window-batch`, `--nonrouted-residency`, `--decode-batch-matrices`,
`--prefetch-depth`) are never forwarded to an engine; `--vram-budget` maps to
the engine's `--vram-budget-gb`; `--reduce-order native` is refused at
invocation build (a sealed-lane concept — engine orders are
fp32|sequential|reverse|pairwise|rotate:N). Local lanes emit
`receipt_class: preview`; only the streaming lane's chain is submittable.
Timing stays honest: the local lanes' `minutes_per_window` is **null** —
decode is measured (16–20 ms/matrix MPS) but the KDA trunk forward is not,
and no number is invented (run `bin/measure-local --fixture fetch` for the
fixture-scale datum).

### History: the 2026-08 flag reconciliation

Preserved verbatim, because the guess-nothing rule earned its keep — the
contract that was written for these lanes was wrong in every guessed
spelling. The engine takes `--teacher` not `--panel`, `--source` not
`--surface`, `--vram-budget-gb` not `--vram-budget`, `--slab-experts` not
`--expert-chunk`; `--window-batch`, `--kld-device` and
`--nonrouted-residency` do not exist at all. Fixing the spellings would not
have been enough:

* `--profile` accepts only `k6|k8|k6k8|native-bf16`, and the controller sent
  `k4` for these lanes (now: `profile_map` in `engines.json`, with a refusal
  for unmapped bit-widths).
* **Every source path resolves to a packed root** and requires
  `contract.json`, `inventory.json`, `mtp-adapter-receipt.json` and
  `payload-store/{objects,choices}` — this campaign's own encode output —
  plus a `--bf16` tree, with ONE exception: `--source native --profile
  native-bf16` needs no packed root at all — only the `--bf16` tree, a sealed
  `inventory.json` (`--inventory`) and the panel. That is the BF16-floor lane
  (see `engines/BF16-FLOOR.md`), and it is also the shape a `tr3-published` reader
  would need. `--source dione` raises *"not enabled in this build"*.

*(That last paragraph is history: `tr3-published`, `dione`, `exl3hf` and `gguf`
readers have all landed since. The GGUF one is the odd member of the set -- it
reads a llama.cpp container, whose repo is a shelf of a dozen builds and whose
quantization covers the whole forward rather than the routed experts alone, so
`--path` is required and its rows are not rankable against the others. See
[`docs/GGUF-MEASUREMENT.md`](../docs/GGUF-MEASUREMENT.md).)*

Which lane reads which surface today is deliberately **not** restated here:
the authoritative table is the generated support matrix in
[README → *Before you rent*](../README.md#before-you-rent-what-is-measurable-today),
rendered from `engines.json` by `bin/render_support_matrix.py` and
drift-checked by `bin/selftest_support_matrix.py`. A surface no lane lists is
refused at plan time by the surface check (`engines.json` → `surfaces`), for
$0.00, instead of after the rental — and `bin/measure` says exactly that at
step 7.

## Adding a new engine or surface

1. Add the entrypoint and `required_flags` to `engines.json`.
2. Declare `surfaces` — the artifact kinds it can actually open. This is what
   stops a rental for bytes nothing can read; leaving it empty disables the
   check.
3. `bin/measure-local --probe-engines` until every required flag is found.
4. Only then `pinned: true` with a filled `flag_map`.

## Performance notes (local)

The engine's only schedule is window-major (`--stream-mode window-major`,
deliberately: it replays the sealed per-window `model()` call verbatim). The
planner prices it honestly (`window_major_cost` in `local-plan.json`): decode
16–20 ms/matrix on MPS → ~11 min/pass, ×25 windows with `--decode-cache none`
(~4.5 h), vs `--decode-cache disk` = one decode + 25 re-reads of the 609 GB
decoded surface (~42–51 min at the 5–6 GB/s of Apple internal NVMe — IF 609
GB is free; measure your disk before assuming). `ram` caches
`floor(0.8·budget/14.5 GB)` layers (7 of 42 on 128 GB). `--unpack-device cpu`
is a fixed flag of the local-mps lane (the MPS int64 escape; decode stays
bitwise). The scorer takes `--chunk-positions 512` (selftest-proven) vs the
sealed default 16. A layer-major preview schedule (decode once + 25 forwards)
is future work, gated on fixture-proven bitwise equivalence plus ≥1 real
window — no engine implements it today, and the planner's legacy layer-outer
block says so in its own `note`.

## Layout

| Path | What it is |
|---|---|
| `measure`, `measure-cloud`, `measure-local`, `registry-view`, `registry-submit`, `fidelity-stats`, `kld-preview`, `fixture` | one-line wrappers, so the headline paste has no `python3` in it |
| `measure_one.py` | the one-command front-end: resolve, gate, lineage, pick, sniff, hand off |
| `measure_cloud.py` | the paid controller: rents one RunPod pod, measures on it, retrieves the sealed result, destroys the pod. Always enforced: `--max-cost` cap, `--max-runtime` deadline, teardown on every exit path, the installed reaper as backstop. Strict campaign mode (`--campaign-*`, `--runpod-safety-proof`) is opt-in; [`docs/CLOUD-RECIPES.md`](../docs/CLOUD-RECIPES.md) |
| `measure_local.py` | the local runner: registry gate, device discovery, memory solver, micro-benchmark, window-major cost, `--execute` with preflight |
| `registry_view.py` | check / rows / lineage against the local clone or the public dataset |
| `fidelity_stats.py` | floor-aware attributable + paired-window deltas (stdlib statistics) |
| `kld_preview.py` | census/sampled preview scorer (torch; fp64 pinned to CPU) |
| `fixture_fetch.py` | the 0.1B CI fixture, cached by commit |
| `fidelity/census.py` | **the shared, testable core** — model census, VRAM/disk/RAM arithmetic, the memory solver, `window_major_cost`. Pure stdlib. |
| `fidelity/hfmeta.py` | revision pinning, blob sizes, surface sniffing, lineage metadata, panel descriptors |
| `fidelity/registry_client.py` | load local/HF registry, tier matcher, renderer, the front gate |
| `fidelity/lineage.py` | base_model walk → registry model → panel/teacher pick |
| `fidelity/previewstats.py` | stratified estimator + FPC + position bootstrap (pure stdlib, unit-tested) |
| `fidelity/runpodapi.py`, `fidelity/runpodsafety.py`, `fidelity/cloudlease.py`, `fidelity/campaign.py` | RunPod control plane, target identity and drill-proof gates, v2 leases plus the systemd reaper, and the spend ledger (per-run by default, one locked campaign ledger in strict mode) |
| `fidelity/engines.py`, `engines.json` | which scorer each lane invokes, how, and `preflight` |
| `fidelity/receipt.py`, `seal_receipt.py` | build and seal a `submission-receipt.v1` — written as `measurement-receipt.json`, which IS the submission receipt a contributor sends; the preview/teacher denylist |
| `fidelity-doctor` | offline local prerequisite check; paid authorization remains the exact `measure-cloud --dry-run` pre-POST gate |
| `render_support_matrix.py` | renders the README support matrix from `engines.json` (`--write`/`--check`); the end of hand-written support claims |
| `stage_measure.sh`, `watchdog.sh`, `invoke_engine.py` | the on-instance side |
| `BUNDLE.txt` | exactly what gets uploaded to rented hardware |

## Fidelity datasets — capture, verify, compare (`bin/fidelity-dataset`)

Scoring is **three separable steps**, one tool, three modes
([`docs/FIDELITY-DATASET-SPEC.md`](../docs/FIDELITY-DATASET-SPEC.md)):

```
step 1  capture   reference (root) weights + panel  ->  fidelity dataset A
step 2  capture   quantized weights + panel         ->  fidelity dataset B
step 3  compare   A, B  ->  KLD + determinism + a registry-submittable receipt
                  A, A  ->  reproduction confirmation, exactly 0.0
```

A root capture is a public good: produced once, sealed, published, and
thereafter downloaded rather than re-run. Step 2 is publishable **standalone**,
before any comparison exists. Step 3 runs with **neither** set of weights
present.

### Race mode — `--role root --race` (engine-level experiment; refused on the paid path)

**Status 2026-09-05: not runnable on the paid path.** `--race`, `--preview-of`
and `--race-workers` are hidden from `measure-cloud --help` and refused at
three layers before any spend: `measure_cloud.py` `_runpod_forbidden`
(*"--race (not wired on the RunPod path yet; see docs/RACE-MODE.md)"*),
`bin/fidelity/stages.py` `stage_sequence(role="root", race=True)` (*"race/preview
root capture is unsupported by the first safe paid path"*), and
`bin/stage_measure.sh` `race_bootstrap`/`race_capture`. What exists is the
identity separation below (`engines/tools/race_fetch.py`, tested by
`bin/selftest_race_mode.py`) and the design in
[`docs/RACE-MODE.md`](../docs/RACE-MODE.md). The sub-hour path that DOES run
today is `--resume-capture <out-A>/result/dataset --resume-origin-job
<out-A>/result/job.json`: cold run 1 on pod A, cold run 2 plus qualification
on pod B (the published GLM-5.3 root was made this way; a one-run capture is
never publishable as the root). On the container-disk layout the saving race
mode would buy is bounded by `min(fetch, capture)` ≈ 10 min (JOURNAL 2026-09-04:
1.5 TB in 12 min, cold run ~10 min).

When a model lands on the Hub, the quants appear within hours and nobody can
say how good any of them are, because there is no root to measure against.
[`docs/RACE-MODE.md`](../docs/RACE-MODE.md) is the whole story; the short form
of what is built:

* **The fetch stops being a barrier.** `engines/tools/race_fetch.py` reads
  `model.safetensors.index.json`, buckets every shard by the first layer that
  needs it, and downloads in that order while the capture runs. The layer-outer
  loader blocks on layer N's shards only when it is about to load layer N. Worst
  case no slower than fetch-then-capture; every run writes a
  `race-fetch-report.json` with `blocked_seconds` measured, so the saving is a
  receipt rather than a claim. The digest is unchanged: race mode changes when
  bytes arrive, never which.
  The head is a **priority-0** file, not a last one — the resident load, the
  vocab/hidden sizes and the capture tap all need it before layer 0.
* **A preview is a different DATASET, not an earlier version of one.**
  `--preview-of FINAL_ID` seals the first cold run under its own `dataset.id`
  with `not_submittable: true` and a blocking `preview_capture` disclosure.
  `reference_id` is a `COMPARABILITY_KEY_FIELDS` member, so updating a published
  root in place would put rows measured against different bytes into ONE
  comparability group; passing the same id for both is refused by name.
* **The generation sanity check runs on EVERY capture**, race or not
  (`engines/tools/generation_probe.py`). `"The capital of France is"` → `" Paris"`,
  as one extra window through the schedule already loading every layer: ~1/N of
  an N-window panel and zero extra weight loading. It is the only guard here
  that sees a shard which loaded as ZEROS — names, shapes and tensor counts are
  all correct in that case. Recorded always; fail-closed unconditionally on a
  degenerate distribution and on a declared `--sanity-expect`.

### Before you start — what exists today, and what does not

The format and the tooling are complete and tested; the **published artifacts
they consume are not yet in place**. Read this before planning GPU time.

| you will want | state today | what to do |
|---|---|---|
| a **root fidelity dataset** to fetch (step 1) | **one is published: [`malaiwah/fruit-fidelity-root-v1`](https://huggingface.co/datasets/malaiwah/fruit-fidelity-root-v1)** — a sealed `malaiwah.fidelity-dataset.v1` hidden-form root for the 5B GLM-5.2-SIQ-Fruit CI fixture, 385 MB, 16 contexts, `describe`/`verify` both resolve it. What does **not** exist is a root for a *production* model: publishing suite-scale captures is out of scope for v1 (spec §14), so there is nothing to fetch for GLM-5.3-Flash or Qwen3.8-27B. `hf://malaiwah/some-fidelity-dataset` in the examples below is still a **placeholder**, not a resolvable id. | To see the whole three-step path working, point at the Fruit root: `bin/fidelity-dataset describe hf://malaiwah/fruit-fidelity-root-v1`. For a production model, capture your own root from the reference weights (same `capture` command, `--role root`), or translate our published serving-lane capture with `adapt --source malaiwah-serving-v2` — that repo (`malaiwah/GLM-5.3-Flash-fidelity-suite-v1`) is **not** itself a conformant dataset, so `verify hf://…` on it will fail. |
| a **token panel** for `--token-panel` (step 1/2) | a **sealed token-panel receipt** produced by the quant pipeline. `capture` hard-requires it — the wrapper needs the mask `.npy` paths, which `capture-receipt.json` does not carry. **One published panel receipt exists**, inside the Fruit root above (`panel/panel-receipt.json` plus `panel/tokens/` and `panel/masks/`), and `fidelity-dataset` fetches it with the dataset. **For every production model the panel is still yours to obtain**: no command here fetches or builds one for GLM-5.3-Flash or Qwen3.8-27B. (Separately, and confusingly: the *runners* — `measure-cloud`/`measure-local` — do not use `--token-panel` at all. They take `--panel <hf-dataset>` and carry one built-in fetch descriptor, brandonmusic's 25-window GLM-5.3-Flash panel. Anything else needs `--panel-descriptor`, and the five Qwen3.8-27B panels are `private` so no descriptor can be written for them.) | Obtain or build a panel receipt before booking a GPU. Both sides of a comparison must be on the *same* panel — `compare` refuses `panel_mismatch` with no override, by design (PANEL-D3). Once you have one, note that its `verified_artifacts` are pinned to the **producer's absolute paths** (`/workspace/artifacts/…`); `bin/stage_panel_paths.py --panel <dir>` copies your fetched files into those paths and verifies each by digest. On a cold box, skipping it fails with `artifact identity mismatch` *after* the fetch, the materialize and the model load. |
| a **cost/time estimate** for a capture | `capture --dry-run` validates inputs, seal and layout only. Unlike `measure-local --estimate-only` and `measure-cloud --dry-run`, **it prints no hours, no VRAM and no dollars.** | Size the run with `measure-local --estimate-only` / `measure-cloud --dry-run` first. As a reference point, a 25-window / 2-cold-run panel is ~2.4-2.7 GPU-hours of scoring on the streaming lane (`engines.json` carries the measured minutes-per-window per surface: 2.82 tr3-published, 3.12 exl3hf, 3.19 dione), ~3.5-4 h end to end with bootstrap, fetch and materialize. The ~8.35 h that used to be quoted here was the K6 **payload-store** path at 7.35 min/window, superseded by M2. |
| to **submit a comparison to the registry** (step 3) | **works, and needs one input file.** `compare --emit-submission` requires `--submission-provenance FILE`: the artifact (HF repo at a 40-hex revision, codec, quantization scope), `panel_ref` and `reference_ref` are registry identities a fidelity dataset cannot know, and `panel_ref`/`reference_ref` must already exist because a measurement may not introduce a panel. Without the file the command **refuses** rather than writing empty blocks. | `fidelity-dataset provenance-template --out prov.json`, fill it in, then `compare … --emit-submission --submission-provenance prov.json`. The command then runs `registry_validate.py --submission` **on its own output** and prints ACCEPTED/REJECTED, so you find out now rather than in review. Copy the accepted file into `registry/receipts/<your-handle>/` and open a PR. |
| to **annotate your card** with the result | `fidelity-card annotate --role quant` resolves its numbers from **published registry measurements**, not from your comparison receipt. With no row yet it refuses, and the refusal names the ordering. | The order is: capture → compare → **get the row into the registry** → then annotate. There is no receipt-to-card path, by design: a card cites registry ids, not local receipts. `--role fidelity-dataset` **is** usable — pass `--fidelity-dataset-root DIR` and every value is read out of that dataset's own manifest. `annotate` always self-validates and exits non-zero rather than writing an invalid card. |

```bash
# step 1/2 -- capture (wraps hidden_replay.py / stream_score.py; never edits them)
bin/fidelity-dataset capture --out ds-bf16 --form hidden --role root \
    --lane sealed-ep8 -- --source native --token-panel <panel> --store-positions all ...
bin/fidelity-dataset capture --dry-run --out /tmp/x --role root --lane sealed-ep8 -- ...
        # --dry-run validates every input, seal and layout and exits 0 WITHOUT a GPU.
        # This is the CI conformance hook.

# verify -- seal + digest chain; stops at the first refusal; there is no --force
# Tensor content digests are recomputed BY DEFAULT: the seal covers the manifest
# and checksums.txt, so a byte flipped inside a tensor whose checksums were then
# refreshed is only caught here. --no-verify-tensors opts out for huge suites,
# and the receipt records which of the two ran.
bin/fidelity-dataset verify ds-bf16
bin/fidelity-dataset verify hf://malaiwah/some-fidelity-dataset@<rev> --no-verify-tensors

# validate -- reports EVERY failure, with the spec rule each one enforces
bin/fidelity-dataset validate ds-bf16 --verify-tensors --json report.json
bin/fidelity-dataset validate --receipt out/comparison-receipt.json

# describe -- the identity card
bin/fidelity-dataset describe ds-bf16

# step 3 -- compare
bin/fidelity-dataset compare --reference ds-bf16 --candidate ds-k6 --out cmp \
    --vocab-chunk 8192            # fixed safe profile; final block may be partial
bin/fidelity-dataset compare --reference ds-bf16 --candidate ds-bf16 --out repro \
    --self-compare --force-compute
        # A == B is a REPRODUCTION CONFIRMATION: exactly 0.0, top-1 exactly 1.0,
        # answered by hash proof; --force-compute runs the math and asserts
        # bitwise agreement.

# step 3 -- the same comparison, with the head matmul on the GPU (opt-in)
bin/fidelity-dataset compare --reference ds-bf16 --candidate ds-k6 --out cmp \
    --device cuda --replay-device cuda
        # For a HIDDEN-form capture, `compare` reconstructs logits as
        # hidden @ head.T. By default that runs in numpy on the CPU while the
        # GPU holds the head for the fp64 estimator and does nothing else --
        # `nvidia-smi` reads 0% for the whole comparison. --replay-device cuda
        # moves it. Measured on the published Qwen3.8-27B root (512 windows,
        # 1,048,064 positions, vocab 248,320, hidden 5,120), one RTX PRO 6000,
        # same process, same data, only this flag different:
        #     numpy  1,754.71 s   GPU  0%
        #     cuda     173.27 s   GPU 88%   peak 7.13 GB device memory
        # 10.13x, and it reproduced the published floor's tokenwise-kld.npy
        # digest 8be5dcca... byte for byte.
        #
        # IT IS NOT THE DEFAULT, AND THAT IS DELIBERATE. An fp32 GEMM
        # accumulates in an order the BLAS chooses, so numpy-on-OpenBLAS,
        # numpy-on-Accelerate and cuBLAS give different last bits from the same
        # head and the same hidden states. The floor is immune (both sides get
        # identical logits, so the KLD is exactly 0.0 either way) but a nonzero
        # row is not. Every receipt now names the backend in
        # `comparator.replay_backend`; rows measured under different values are
        # not rankable against each other. Pick one per comparability group and
        # keep it.
        #
        # --replay-dtype float64 accumulates the replay in fp64 instead: more
        # accurate, and much more reproducible across backends, but a DIFFERENT
        # measurement from either fp32 path.

# step 3 -- a registry submission (needs identities a dataset cannot know)
bin/fidelity-dataset provenance-template --out prov.json     # skeleton; fill it in
bin/fidelity-dataset compare --reference ds-bf16 --candidate ds-k6 --out cmp \
    --vocab-chunk 8192 --emit-submission --submission-provenance prov.json
        # refuses rather than writing empty blocks, then runs the registry's own
        # `registry_validate.py --submission` on the file it just wrote.

# adapt -- foreign artifacts
bin/fidelity-dataset adapt --source k3v1 --in <kimi-k3 artifact> --out k3-ds \
    --emit-dataset --emit-k3-compat        # --emit-dataset needs the tensors present
bin/fidelity-dataset adapt --source malaiwah-serving-v2 --in <capture dir> \
    --suite <suite dir> --head-dir <head dir> --out ds --limit 8 --emit-k3-compat
bin/fidelity-dataset adapt --source llamacpp-kld --in base.kld --out kld-translation

# interop -- make the kimi-k3 comparator read our dataset, unmodified
bin/fidelity-dataset verify-k3-compat ds-bf16
        # compat/ is three JSON files of RELATIVE ALIASES: no tensor is copied,
        # and the tree is written before the seal so checksums.txt covers it.
```

**Exit codes:** `0` ok, `2` warnings only, `3` refused, `4` bad usage.

**The refusals worth knowing** (each names its spec rule):

| you will hit | because |
|---|---|
| `head_mismatch` (HEAD-1b) | the two hidden-form captures declare different `lm_head` tensor-content digests. Replaying one artifact's hiddens through the other's head erases its head-quantization error and flatters it. `--disclose-head-substitution` proceeds but forces `advisory`, a downward bias block and a **blocking** disclosure — i.e. not publishable. |
| `head_mismatch` (HEAD-4) | a hidden-form dataset with a null head content digest. **No override.** |
| `panel_mismatch` (PANEL-D3) | `scoring_window.score_from` differs. That is a different *panel*, not a comparator flag — which is what makes a llama.cpp-geometry number structurally incomparable rather than silently comparable. There is deliberately no override, so the refusal prints a `remedy:` line saying so. |
| `panel_mismatch` (PANEL-D6) | the two captures declare different tokenizers. `suite_token_hash_sha256` hashes token **ids** — integers — so it cannot see this; two tokenizers can emit the same ids from different text. A field null on either side is *unknown*, not different. |
| `head_substitution_vacuous` (HEAD-1c) | the capture content digests are **equal** and the head digests **differ** — a head-only quant (stock EXL3 `head_bits` 6–8). Hidden replay through one head erases the only difference there is and would report an exact reproduction. **No override**: publish logit-form captures, where each side runs its own head. |
| `lane_mismatch` | different lanes. `--allow-cross-lane` proceeds and stamps `usable_as_floor: false`, so **BIAS-006** cannot be laundered downstream. |
| `unlisted_file` / `missing_file` | the tree is not exactly what `checksums.txt` covers. `--allow-partial` narrows this to capture tensors and stamps `covers_full_panel: false`. |
| `bad_vocab_chunk` | `--vocab-chunk` must be a positive integer. A final partial vocabulary block is processed exactly; the safe root qualification profile binds **8192**. |
| `replay_device_mismatch` | `--replay-device` names a device the estimator does not use. The replayed logits would cross the bus twice per position block, which is slower than the numpy path it replaces. Set `--device` to the same value. |
| `replay_backend_unavailable` | `--replay-device` other than `numpy` needs torch. The default needs nothing. |
| `bad_replay_dtype` | `--replay-dtype` is `float32` or `float64`. |

`checksums.txt` is `sha256sum --check`-compatible, so a reviewer with none of
our tooling verifies the payload with one coreutils command.

## Card annotation (`bin/fidelity-card`)

Machine-readable fidelity provenance on an HF model or dataset card
([`docs/CARD-ANNOTATION-SPEC.md`](../docs/CARD-ANNOTATION-SPEC.md)): one
conformant `model-index` entry plus one additive `x_fidelity:` block.

```bash
bin/fidelity-card annotate --card README.md --role quant \
    --artifact-id artifact--malaiwah.glm-5.3-flash-tr3-6bpw \
    --base-model zai-org/GLM-5.3-Flash-BF16 --out README.annotated.md --diff --validate

# a capture publisher's own card (step 2 is publishable standalone)
bin/fidelity-card annotate --card README.md --role fidelity-dataset \
    --fidelity-dataset-root ds-bf16 --fidelity-dataset malaiwah/my-root@main \
    --out README.annotated.md
        # every value is read from that dataset's manifest; nothing is retyped.

bin/fidelity-card validate --card README.md            # three axes
bin/fidelity-card validate --card README.md --offline  # skips the Hub axis, and SAYS so
```

Three validation axes, all must pass: the live Hub `validate-yaml` endpoint
(the same push-time gate), a `huggingface_hub` round-trip that must be
structurally identical, and our own XC-1..XC-5 cross-checks against
`registry/data/measurements.jsonl`.

`annotate` **always validates its own output** against our XC checks and exits
non-zero rather than writing an invalid card; `--validate` adds the Hub and
round-trip axes. It derives `reference_model` / `reference_revision` from the
registry (measurement → reference → artifact → `huggingface.repository`) instead
of asking you to retype them, and **warns by name** for any field it had to
leave null and the flag that would supply it.

`annotate` never rewrites the card **body**, never sets `verified` /
`verifyToken` (HF-controlled), and never invents a head digest — an artifact
with no published head *content* digest gets `replay_permitted: false` and an
explanatory note. Generated cards for K6 and K8 live in
[`docs/cards/`](../docs/cards/); publishing them is a separate, permissioned act.

## Selftests

```bash
bin/selftest_all.sh                        # everything below; PASS/FAIL/SKIP ledger
python3 bin/selftest_fit.py                # 41 known-answer checks (census, solver, window-major cost)
python3 bin/selftest_decode_parity.py      # needs torch; decode bitwise MPS==CPU
python3 bin/selftest_registry_view.py      # T1: loader, tiers, never-merge renderer
python3 bin/selftest_stats.py              # T2: K8-ANOMALY known answers, refusal arithmetic
python3 bin/selftest_preview_stats.py      # T3: unbiasedness, coverage, FPC, panel gate
python3 bin/selftest_zero_floor.py         # T4: the exact-0.0 identity (+ fixed npy sha)
python3 bin/selftest_submission_refusal.py # T5: previews/teachers cannot become rows
python3 bin/selftest_fidelity_dataset.py   # T6: format, seals, panel/head/lane/coverage refusals
python3 bin/selftest_fidelity_compare.py   # T8: known-answer KLD, exact self-compare, SC-3
python3 bin/selftest_fidelity_card.py      # T7: card annotation, 3 axes (--offline skips the Hub axis)
python3 bin/selftest_support_matrix.py     # CX1: README support matrix == engines.json (render drift fails)
python3 bin/selftest_readme_recipes.py     # CX2: every fenced README recipe command parses against the real CLI
python3 bin/fidelity-doctor                # CX3: is this machine ready? read-only, prints no secret
python3 engines/tools/stream_score_selftest.py --only g,h,i,j,k # engine-edit rungs
bin/registry-view --selftest-live          # live dataset, keys, value tripwire
```

The paid reaper is RunPod-only and user-systemd-backed. Selftests use stubs or
`reaper --sweep --dry-run`, which destroys nothing. `measure-cloud reaper
--provider runpod --install` needs the owner-only RunPod key and a user
manager with linger; it seals a snapshot of the reaper sources and the timer
runs that snapshot. A checkout that has moved on since is advisory drift, not
an unhealthy reaper: the paid controller warns and proceeds, and re-running
`--install` picks up the newer checkout. Before every create POST the
controller checks that the timer is active, the user manager persists, the
snapshot is intact, the account id and lease path match and the health stamp
is fresh.
