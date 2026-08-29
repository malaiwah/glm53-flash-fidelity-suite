# GLM-5.3-Flash fidelity suite & quantization program

Tools, receipts, and campaign log for measuring — and then beating — the
quality cost of quantizing [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(321B-total / A18B MoE, `glm5_next` hybrid KDA/DSA architecture with mHC
hyper-connections). Everything here was produced within ~48h of the model's
release and is receipt-driven: every published number links to a JSON receipt
with pinned revisions and sha256s.

## Measure a quant from an HF link — one command

```bash
bin/measure malaiwah/GLM-5.3-Flash-TR3-6bpw
bin/measure https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX/tree/main/4-bit
```

(Both are already measured, so both answer from the registry for $0.00 —
the honest common case. A repo whose live head has moved since it was
measured — `zai-org/GLM-5.3-Flash-BF16` today — refuses with the drift
remedies instead of silently answering about different bytes.)

It resolves the revision (live head by default), **asks the public registry
first** — an already-measured artifact gets its rows and receipt links printed
and exit 0, nothing spent — then walks `base_model` lineage to the registry's
model, picks the panel/teacher prior measurements used (alternatives printed
with override flags), sniffs the repo's packing surface, picks the lane for
your machine, and hands off to `measure-local --execute`. Refusals name their
arithmetic or remedy: revision drift needs `--force` or
`--accept-measured-revision`; a surface no lane can read (most third-party
repos today) is refused for $0.00 with the missing reader named; missing
torch/transformers/quant_pipeline/teacher/disk are all listed at once with
their install commands. `--plan-only` stops at the plan.

## Browse the registry

```bash
bin/registry-view check malaiwah/GLM-5.3-Flash-TR3-6bpw   # already measured? (tiers + rows + receipts)
bin/registry-view rows --model glm --lane streaming        # filtered, never-merged tables
bin/registry-view lineage 0xSero/GLM-5.3-Flash-EXL3-Q4     # base-model walk + panel/teacher pick
```

Works offline against the local clone and online against the public dataset
(`--registry auto|hf|local[:PATH]`; the footer names the snapshot that
answered). Rows are grouped by recomputed comparability key and split by lane
— filters can hide groups but never merge them, so cross-reference ranking is
structurally impossible. Floor-aware analysis lives in `bin/fidelity-stats`
(the streaming lane's floor and the cross-lane refusal arithmetic), local
preview scoring in `bin/kld-preview`; the same-lane-teacher plan that drives
the floor to zero is [`k6/SAME-LANE-TEACHER.md`](k6/SAME-LANE-TEACHER.md).

## Measure a quant yourself — two copy-paste recipes

Every number in this repo was produced by a recipe you can run on someone
else's weights and submit to the [fidelity registry](registry/). Both recipes
produce the **same sealed receipt**, so a number measured on a rented H200 and
a number measured on a laptop are the same kind of object and can be ranked
against each other.

Both refuse *before* spending anything when the run will not fit, and both say
what they need — dollars, disk, memory, hours — with each figure's provenance.
Both now run the same registry front gate as `bin/measure` before planning.

### Recipe 1 — cloud: rent, measure, tear down

```bash
export JL_API_KEY=...      # never logged, never written to a receipt

bin/measure-cloud reaper --install     # required for any run over 2h

bin/measure-cloud \
    --model <hf-repo> \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane  streaming --spot --max-runtime 12h
```

`--max-runtime` must exceed the estimated work, which `--dry-run` prints —
8h does not cover the 8.35h a 25-window / 2-cold-run panel needs, and the
runner refuses rather than paying for a run its own watchdog would kill.

Resolves the repo to an immutable commit, sizes and prices the instance, asks
for confirmation, creates it, fetches weights and panel, measures, seals the
receipt, pulls it back, **destroys the instance**, and prints what it actually
cost — estimated, computed, billed, and as an account-balance delta, because
any one of those alone can lie.

Start with `--dry-run`: it does every check, creates nothing, and spends
nothing. Teardown is guaranteed on success, failure, exception and Ctrl-C, and
three further layers sit under that trap for the case where the controller
itself dies — see `bin/measure_cloud.py`, class `Teardown`.

```
bin/measure-cloud reaper --install    # the backstop; the runner asks for it
bin/measure-cloud reaper --sweep      # clean up from any machine, after a laptop dies
```

### Recipe 2 — local: your own hardware

```bash
bin/measure-local \
    --artifact brandonmusic/GLM-5.3-Flash-tr3-4bpw \
    --panel    brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --vram-budget 30
```

Works on a 128 GB Apple-Silicon Mac via MPS, and on a 32 GB consumer CUDA card
under a `--vram-budget` that is a **hard bound, not a hint**. A 600 GB model
fits in 30 GB because the schedule is inverted: instead of streaming the whole
checkpoint once per panel window, it goes layer-outer and pushes all 25 windows
through each layer, so every expert is decoded exactly once for the whole panel.
`--expert-chunk` and `--window-batch` then shrink the peak without moving the
number — experts are visited in ascending order and accumulated sequentially in
fp32, so the result is bit-identical at any setting.

`--estimate-only` prints the plan and stops. `--simulate-device "RTX 5090:32"`
plans for hardware you do not own yet.

```
$ bin/measure-local --artifact ... --panel ... --vram-budget 30 --estimate-only
  expert_chunk    156 of 288 experts  (numerics-invariant)
  window_batch    25 of 25 windows -> 1 pass(es) over the checkpoint
  peak VRAM       25.45 GB of 30.00 GB budget (85%)
```

Ask for too little and it refuses with the arithmetic, not a stack trace:

```
REFUSE: no schedule fits a 3.60 GB budget
        minimum viable budget for this model at 4 bpw is 4.58 GB
        that floor is set by the lm_head step -- the lm_head weight (1.27 GB)
        and one window of fp32 logits (1.27 GB) must be resident together, and
        neither shrinks with --expert-chunk or --window-batch
        run the cloud recipe instead:  bin/measure-cloud --lane streaming
```

Verify the machine before trusting it — both selftests are offline and take
under a minute:

```bash
bin/measure-local --selftest      # fit estimator vs known cases + decode parity
```

### Recipe 3 — submit it

```bash
bin/registry-submit <out>/receipts/measurement-receipt.json
```

Prints the row your receipt would generate, its comparability key, and the rows
it can be ranked against — or exactly which check it failed. Then open a
discussion on the registry dataset and attach the file; a GitHub PR against the
mirror works too. Both paths, the paste template, and how you are credited:
[`registry/CONTRIBUTING.md`](registry/CONTRIBUTING.md).

> **Requirements.** The cloud recipe needs the `jl` CLI
> (`uv tool install jarvislabs`). The local recipe needs only
> `pip install torch safetensors numpy huggingface_hub` — the EXL3/TR3 decode
> is pure PyTorch, so none of the CUDA-13/flash-attn/exllamav3 bootstrap the
> cloud lane uses is required on your own machine. On a Homebrew or distro
> Python that install is blocked by PEP 668 ("externally-managed-environment");
> use a venv (`python3 -m venv ~/.venvs/fidelity` and point `FIDELITY_PYTHON`
> at its `python3`) or add `--break-system-packages` knowingly.

## Headline results

| Measurement | Value | Where |
|---|---|---|
| Official FP8 vs BF16, mean KLD (10.48M positions) | **0.028104 nats** (CI95 [0.0272, 0.0290], top-1 94.3%) | [fidelity dataset](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1) |
| Official FP8 on brandonmusic's sealed 25-window panel | **0.020615 nats** / top-1 95.6% | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/fp8-on-brandon-panel.json) |
| Cross-stack BF16 floor (our replay vs his teacher) | 0.012712 nats | [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/crosscheck-brandonmusic.json) |
| glm5_next launch nondeterminism (first report + interventions) | pins → ~10× flip-rate reduction | [vLLM PR #53906 comments](https://github.com/vllm-project/vllm/pull/53906), `reports/determinism-*.json` |

## What's in this repo

| Path | What it is |
|---|---|
| [`bin/`](bin/) | **The two copy-paste recipes above**: `measure-cloud`, `measure-local`, `registry-submit`, the shared fit estimator (`fidelity/census.py`), the engine pin file (`engines.json`), and two offline selftests |
| [`registry/`](registry/) | **The fidelity registry**: schemas, seeded rows, submission receipt format, validator, and [CONTRIBUTING.md](registry/CONTRIBUTING.md) |
| [`tools/`](tools/) | The fidelity harness (vLLM hidden-state capture → shared-head replay → exact full-vocab KL), activation capture, cross-stack checker, publishers |
| [`remote/`](remote/) | The self-driving on-VM pipeline + stage scripts used for the overnight 8×H200 capture campaign |
| [`k6/`](k6/) | **The K6/K6K8 EXL3 quantization program** (in progress): runbook, stage driver, patch series onto [brandonmusic's pipeline](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw), driver tools, recipes, and the disclosed [r10 codec reconstruction](k6/fallback/) |
| [`port/`](port/) | Design bundle for a native exllamav3 `glm5_next` architecture port (blueprint, draft, parity harness, adversarial review) |
| [`suite/`](suite/), [`calsuite/`](calsuite/) | The held-out evaluation suite (5,120×2,048 ctx) and calibration token sets |
| [`JOURNAL.md`](JOURNAL.md) | The captain's log: every decision, failure, cost, and 24 lessons learned |

## Published datasets

- [malaiwah/GLM-5.3-Flash-fidelity-suite-v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1) — the quality reference: BF16 + FP8-as-served hidden states over 10.48M positions, shared lm_head, all receipts.
- [malaiwah/GLM-5.3-Flash-calibration-activations-v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-calibration-activations-v1) — 147 GB of MoE block-input activations + router logits (natural routing), for calibration-aware quantization work.

## Credits & lineage

Methodology descends from the author's Qwen3.8-27B fidelity/quant work
([malaiwah/qwen38-27b-exl3](https://github.com/malaiwah/qwen38-27b-exl3)).
The K6 program builds directly on
[brandonmusic](https://huggingface.co/brandonmusic)'s GLM-5.3-Flash EXL3
pipeline and BF16 teacher-logits dataset — see the
[co-credited corroboration thread](https://huggingface.co/brandonmusic/GLM-5.3-Flash-EXL3-4bpw/discussions/1).
Base model by [Z.ai](https://huggingface.co/zai-org); quant format by
[turboderp's exllamav3](https://github.com/turboderp-org/exllamav3).
