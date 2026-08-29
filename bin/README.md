# `bin/` — the recipes

`measure`, `measure-cloud` and `measure-local` are the user-facing product: a
stranger with a quant and a GPU should be able to paste one line and get a
sealed, submittable number — or the honest answer that the number already
exists, or a refusal that names its arithmetic. `registry-view` browses the
public registry from the CLI; `registry-submit` checks a sealed receipt the
way the registry will, before it is sent anywhere.

## Why here and not in `k6/tools/`

`k6/tools/` is campaign-scoped — its name is a campaign, its contents assume
GLM-5.3-Flash and the K6 encode. The runners are meant for people who have
never heard of K6 and are measuring some other model entirely. Putting them at
the top level alongside `registry/` is what makes the pair read as a product
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
contributed, because per-window scatter (sd 1.73e-3) exceeds the K6-vs-K8
effect (1.22e-3) — a single window has no power to compare quants (lessons
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
`k6/BF16-FLOOR.json` (the analysis) is refused as a floor input by name, and
"floor = 0" is accepted only with T1 hash evidence
(`k6/SAME-LANE-TEACHER.md`). `--from-registry MEASUREMENT_ID` fetches the
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
`k6/tools/k6_student_capture.py`), and `bf16-floor`, `streaming`,
`local-mps`, `local-cuda-budget` (all against `k6/tools/stream_score.py`,
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
  (see `k6/BF16-FLOOR.md`), and it is also the shape a `tr3-published` reader
  would need. `--source dione` raises *"not enabled in this build"*.

So no lane can read a third-party `tr3-published` artifact, which is what a
stranger's quant almost always is. Until a `tr3-published` reader exists,
measuring someone else's repo is refused at plan time by the surface check
(`engines.json` → `surfaces`), for $0.00, instead of after the rental — and
`bin/measure` says exactly that at step 7.

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
| `measure_cloud.py` | the cloud controller: registry gate, preflight, fit, instance selection, cost, four-layer teardown, reaper |
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
| `fidelity/jlapi.py` | the single chokepoint for every `jl` call |
| `fidelity/engines.py`, `engines.json` | which scorer each lane invokes, how, and `preflight` |
| `fidelity/receipt.py`, `seal_receipt.py` | build and seal a `submission-receipt.v1`; the preview/teacher denylist |
| `stage_measure.sh`, `watchdog.sh`, `invoke_engine.py` | the on-instance side |
| `BUNDLE.txt` | exactly what gets uploaded to rented hardware |

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
python3 k6/tools/stream_score_selftest.py --only g,h,i,j,k # T6: engine-edit rungs
bin/registry-view --selftest-live          # T8: live dataset, keys, value tripwire
```

The reaper section is safe by default: the selftest runs `reaper --sweep
--dry-run`, which reports what WOULD be destroyed and destroys nothing (a
real sweep is `bin/measure-cloud reaper --sweep`, run deliberately). On a
machine without the `jl` CLI the sweep test SKIPs with the install remedy
instead of failing. `SELFTEST_SKIP_ACCOUNT=1` still skips the whole section
(e.g. to avoid even read-only account API calls while another session rents).
