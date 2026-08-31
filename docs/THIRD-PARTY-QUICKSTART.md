# Third-party quickstart — from a fresh clone to a submitted measurement

This is the unaided path for someone who has never used this repo: exact
prerequisites, one verification command, one named target that is genuinely
measurable today, the fail-closed dry-run, the exact prompt you will see
before any money moves, the file you get, and the one live place to submit it.
Everything here costs $0.00 until the single step that says otherwise.

What is measurable at all is decided by the generated
[support matrix](../README.md#before-you-rent-what-is-measurable-today) —
rendered from `bin/engines.json`, never hand-written. The short version for a
third-party quant: **you will be renting** (the local lanes read only `packed`
and `native-bf16`, so `bin/measure-local` cannot execute a third-party
measurement), and today that means a GLM-5.3-Flash quant on brandonmusic's
25-window panel, on the `streaming` lane, via `bin/measure-cloud`.

---

## 1. Prerequisites, with install commands

| what | install / setup | needed for |
|---|---|---|
| python ≥ 3.9 | stock; no `pip install` for any `bin/` tool | everything |
| the suite | `git clone https://github.com/malaiwah/quant-fidelity-suite && cd quant-fidelity-suite` | everything |
| ONE provider credential | JarvisLabs: `uv tool install jarvislabs && jl setup --token <token> --yes` (or `export JL_API_KEY=...`) · RunPod: `export RUNPOD_KEY_FILE=~/.runpod_key` · Vast: `export VAST_KEY_FILE=~/.vast_key` · Lambda: `export LAMBDA_KEY_FILE=~/.lambda_key` — key files are 0600, one line, never on argv ([`docs/CLOUD-RECIPES.md` §1](CLOUD-RECIPES.md)) | the paid run |
| SSH key | `ssh-keygen -t ed25519` — RunPod and Vast inject `~/.ssh/id_ed25519.pub` at create; **Lambda attaches keys by name from your account**, so register one in its console first; JarvisLabs needs none (the `jl` CLI is the transport) | RunPod / Vast / Lambda |
| HF auth | `hf auth login`, or `export HF_TOKEN=$(cat ~/.hf_token)` — read from a file, never echoed, shredded at teardown | gated repos, faster transfer; public repos work without it |
| lease reaper | `bin/measure-cloud reaper --install` — the teardown backstop for when your laptop dies mid-run; **the runner refuses any run over 2 h without it** | the paid run |
| local torch env | NOT needed for the cloud recipe. Only for `measure-local --execute` / preview scoring: `python3 -m venv ~/.venvs/fidelity && ~/.venvs/fidelity/bin/pip install torch safetensors numpy huggingface_hub && export FIDELITY_PYTHON=~/.venvs/fidelity/bin/python3` | local lanes only |

## 2. Verify, with one command

```bash
bin/fidelity-doctor
```

Read-only, offline, prints no secret. `OK` on a row means that path works;
every `WARN`/`FAIL` names its remedy. Exit 0 means a $0.00 dry-run can run
from this machine. (Deeper checks when you want them:
`bin/measure-local --selftest`, `bin/measure-local --probe-engines`,
`bash bin/selftest_all.sh`; inside the container image, `docker run <image>
doctor`.)

## 3. A named target that is measurable today

`unsloth/GLM-5.3-Flash-GGUF` at revision
`2975ab414d30340466d8c51533c6e91f0cca64c1` — the model's largest quant
audience, wired end-to-end on the streaming lane (see
[`docs/GGUF-MEASUREMENT.md`](GGUF-MEASUREMENT.md) for what a GGUF row means —
it quantizes the whole forward, so it is **not** rankable against
routed-experts-only rows). This repo is a *shelf* of twelve builds at one
revision, so **`--path` is required** — it names the build, and the planner
lists the choices if you omit it. Of the twelve, v1 decodes five: `BF16`,
`Q8_0`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`.

Check it is still unmeasured first (an already-measured artifact answers from
the registry for $0.00 and exits 0 — the honest common case):

```bash
bin/registry-view check unsloth/GLM-5.3-Flash-GGUF
```

## 4. The fail-closed dry-run — $0.00, creates nothing

```bash
bin/measure-cloud \
    --model    unsloth/GLM-5.3-Flash-GGUF \
    --revision 2975ab414d30340466d8c51533c6e91f0cca64c1 \
    --path     UD-Q4_K_XL \
    --panel    brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane     streaming --spot \
    --provider jarvislabs --gpu H200 \
    --max-cost 25 --max-runtime 12h \
    --measurer <your-hf-handle> \
    --dry-run
```

Flag notes: `--max-cost` refuses any plan whose estimated band high exceeds
it;
`--max-runtime` must exceed the estimated work (the dry-run prints it) or the
runner refuses rather than paying for a run its own watchdog would kill;
`--provider`/`--gpu` pick the box (prices and measured $/window:
[`docs/CLOUD-COMPARISON.md`](CLOUD-COMPARISON.md)); **`--measurer` is how you
are credited** — it defaults to the maintainer's handle, and a submission
claiming the maintainer's handle from someone else's run is bounced
(CONTRIBUTING §5), so set it to your HF handle now, not at submission time.

**The dry-run is fail-closed, and its verdict is tri-state.** Every mandatory
gate ends `verified`, `failed`, or `not_checked` — a gate that could not run
(network blip, missing import) is *not* treated as passed. You will see one
of exactly three endings:

* `all checks passed; a real run would proceed to the confirmation prompt.`
* `N check(s) would REFUSE a real run:` followed by each named check — exit 3.
* `INCOMPLETE -- this dry run CANNOT AUTHORIZE a run.` naming the gates that
  were `not_checked`. The numbers printed above it are fallback estimates,
  not verdicts about your artifact; a **real** run refuses until every
  mandatory gate verifies.

Take the hours/dollars the dry-run prints for *your* target over any constant
in prose. Ballpark for this target: the planner prices `gguf` at 3.19
min/window (a conservative placeholder — the first finished GGUF capture
replaces it), so 2 cold runs × 25 windows ≈ 2.7 GPU-h of scoring, plus
fetching a ~200 GB build, bootstrap and materialize — budget roughly $8–15
and half a day of `--max-runtime` headroom on an 80 GB-class card.

## 5. The paid step, and the exact prompt you will see

Re-run the same command **without `--dry-run`**. After the plan, the runner
stops and asks — this is the only moment money is about to move:

```
Create 1 x H200 (IN2, spot) and spend up to ~$9.52?  [y/N]
```

(The template is `Create <count> x <gpu> (<region>, <spot|on-demand>) and
spend up to ~$<band-high>?  [y/N]` — the dollar figure is the plan's band
high, not the point estimate. `--yes` skips the prompt; leave it off the
first time.) Answering anything but `y`/`yes` aborts with $0.00 spent.

Then it rents, fetches, measures 2 cold runs, seals, pulls the receipts back,
**destroys the instance** (guaranteed on success, failure, exception and
Ctrl-C, with the reaper as backstop), and prints the cost four ways —
estimated, computed, billed, balance delta. Verify nothing is still billing:
`jl list` (or your provider's console) should show nothing of yours running.

## 6. What you now have

```
<out>/receipts/measurement-receipt.json      # <out> defaults to ./fidelity-runs/<job-id>
```

**This file is your submission receipt** — schema
`quant-fidelity-registry/submission-receipt.v1`, sealed by `receipt_sha256`
over its canonical JSON. It is the one and only thing you submit. (Older docs
call the same object `submission.json`; same file, one noun: the measurement
receipt IS the submission receipt.) Do not edit it — any edit breaks the seal
and the registry bounces it; re-run instead.

## 7. Validate it the way the registry will — offline, $0.00

```bash
bin/registry-submit <out>/receipts/measurement-receipt.json
```

Prints the row your receipt generates, its comparability key and class, and
the rows it may be ranked against — or exactly which check failed. Doing this
before submitting is, in the maintainer's own words, the difference between a
same-day merge and a round trip.

## 8. Submit it — one live destination

**Hugging Face discussion** on the registry dataset — this is the live path:

1. Open <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
2. New discussion titled `submission: <repo> on <panel>`
3. Paste the template from [CONTRIBUTING §2](../registry/CONTRIBUTING.md) with
   your receipt inside the fence (attach the file too if the editor allows).

The GitHub pull-request mirror described in CONTRIBUTING §3 is **not live** —
the URL 404s today; CONTRIBUTING says so and this document will keep saying so
until it exists. Do not wait for it.

## 9. What happens next — what is promised, and what is not

From [CONTRIBUTING §4](../registry/CONTRIBUTING.md), which is the commitment
this project actually makes:

* The maintainer saves your receipt under `receipts/<your-handle>/`, runs the
  same checks CI would, and **replies in your thread** with the generated row
  id, its comparability key, its class, and which rows it sits next to — or,
  if refused, exactly which check failed and what to change. "Either way you
  get a real answer, not silence."
* **No response-time SLA is promised anywhere**, and this document will not
  invent one. The stated fast path is a receipt that already passed
  `bin/registry-submit`: validated submissions are same-day-mergeable; broken
  seals and missing fields cost a round trip.
* Acceptance criteria are mechanical and public: the seal verifies, the
  40-hex revision, `run_count >= 2` with tensor-content determinism evidence,
  a scope census (for a GGUF the registry refuses the row without one), a
  `produced_by` block with `entrypoint_sha256` (HARN-001), your own handle in
  `measurer` — the complete bounce list is
  [CONTRIBUTING §5](../registry/CONTRIBUTING.md).
* Outside measurements enter as class `advisory` (that is about provenance,
  not trust), shown in the same tables when the comparability key matches.
  The flag worth chasing afterwards is `independently_verified: true` — a
  different party reproducing your number on the same panel and reference.

## If your target is not measurable

The refusal will name why for $0.00: no panel you can fetch (Qwen3.8-27B is
closed to outside measurement today — panels are private), no reader for the
surface on any lane (MLX / NVFP4 / AWQ / GPTQ), or no profile at that rate.
Those are the real boundaries of the system today; the
[support matrix](../README.md#before-you-rent-what-is-measurable-today) is
the authoritative statement of them, and
[CONTRIBUTING §6](../registry/CONTRIBUTING.md) is the path for proposing a
new panel or model.
