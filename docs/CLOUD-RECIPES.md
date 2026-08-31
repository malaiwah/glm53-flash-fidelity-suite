# Measuring on a rented GPU: the working recipes

Four providers work today. The measurement is identical on all of them — full
vocabulary KLD in fp64 is not vendor-specific — so the only real questions are
what it costs, whether the GPU fits, and whether the box is guaranteed to die
when you are done.

Everything below has been run. Where a number is quoted it was measured, and
where something is unproven it says so.

## 0. Sixty seconds to your first refusal

Do this before you rent anything. `--dry-run` costs **$0.00**, creates nothing,
runs every check the real run runs, and prints the cost band.

```bash
export HF_TOKEN=$(cat ~/.hf_token)
export RUNPOD_KEY_FILE=~/.runpod_key        # 0600, key on one line

./bin/measure-cloud --provider runpod --on-demand \
    --model  turboderp/GLM-5.3-Flash-exl3 \
    --panel  brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane   streaming --max-cost 20 --dry-run
```

A refusal here is the tool working. The common ones and what they mean:

| refusal | it is telling you |
|---|---|
| `no available instance fits this lane` | no offer has the VRAM the fit check computed. Also fires when you asked for `--spot` on a provider whose offers are all on-demand. |
| `cheapest-that-fits picked X, but lane streaming was validated on H200` | you must choose: `--gpu H200` for the comparable number, or name another GPU and accept that the timing estimate does not transfer. |
| `this release is missing N of the official model's tensors` | the release is incomplete and would be measured with randomly initialised weights. This one saved a rental. |
| `the supplied --scope-json contradicts this release's own published rates` | your scope file belongs to a different artifact — usually a sibling branch of the same repo. |
| `--scope-json does not satisfy the submission schema` | the file has a property the receipt schema forbids. Caught here it costs nothing; it used to be caught at seal time, after the whole run. |

## 1. Credentials

Every backend reads its key **from a file**, never from argv, so it cannot show
up in `ps` on a shared machine.

```bash
export RUNPOD_KEY_FILE=~/.runpod_key      # or RUNPOD_API_KEY
export VAST_KEY_FILE=~/.vast_key          # or VAST_API_KEY
export LAMBDA_KEY_FILE=~/.lambda_key      # or LAMBDA_API_KEY
export JL_API_KEY=...                     # JarvisLabs uses its own CLI login
chmod 600 ~/.*_key
```

SSH: RunPod and Vast inject `~/.ssh/id_ed25519.pub` at create time. **Lambda
does not** — it attaches keys *by name* from the account, so register one in
the Lambda console first or `create` refuses.

## 2. The four providers, measured

Prices are what these accounts were actually quoted, and they move.

| | JarvisLabs | RunPod | Vast.ai | Lambda |
|---|---|---|---|---|
| flag | `--provider jarvislabs` | `--provider runpod` | `--provider vast` | `--provider lambda` |
| spot | yes | on-demand here | on-demand here | **no** |
| 80 GB rate seen | H200 spot **$1.99** | A100-SXM4 **$1.59** | A100 PCIe **$0.575** | H100 SXM5 **$4.29** |
| storage | **separable** — outlives the box | dies with the pod | dies with the instance | **fixed per type** |
| pick a disk size | yes | yes | yes, at rent time | **no** |
| balance API | yes | yes | yes | **no** (pay-as-you-go) |
| transport | `jl` CLI | SSH | SSH | SSH (`ubuntu@`) |
| first SSH | — | ~12 s | **~99 s** (image pull) | — |

**Which to use.** Vast is the cheapest by a wide margin and the least
predictable — it is a marketplace, so you rent one specific person's machine.
RunPod is the best-behaved API and had high stock on A100-SXM4. JarvisLabs is
the only one with a filesystem that survives its instance, which is what makes
a preempted spot box cheap to resume. Lambda is the most expensive and the most
predictable, which makes it the right place for something that must not be
interrupted and the wrong place for anything cheap.

## 3. Recipes

### RunPod

```bash
export RUNPOD_KEY_FILE=~/.runpod_key HF_TOKEN=$(cat ~/.hf_token)
./bin/measure-cloud --provider runpod --on-demand \
    --gpu "NVIDIA A100-SXM4-80GB" \
    --model  <hf-repo> --revision <40-hex> \
    --panel  brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane   streaming --max-cost 20 --max-runtime 8h --yes
```

Use the **full** GPU id (`NVIDIA A100-SXM4-80GB`), not the display name.
Community cloud is cheaper and thinner; secure had stock High.

### Vast.ai

```bash
export VAST_KEY_FILE=~/.vast_key HF_TOKEN=$(cat ~/.hf_token)
./bin/measure-cloud --provider vast --on-demand --gpu "A100 PCIE" \
    --model <hf-repo> --revision <40-hex> \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane  streaming --max-cost 12 --max-runtime 8h --yes
```

**Always pass `--gpu` on Vast.** Without it, "cheapest that fits ≥63 GB" on a
real account was a **CMP 170HX** — a 64 GB *mining* card that satisfies the VRAM
filter and is useless for this work.

### Lambda

```bash
export LAMBDA_KEY_FILE=~/.lambda_key HF_TOKEN=$(cat ~/.hf_token)
./bin/measure-cloud --provider lambda --on-demand --gpu gpu_1x_h100_sxm5 \
    --model <hf-repo> --revision <40-hex> \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane  streaming --max-cost 25 --max-runtime 6h --yes
```

`--gpu` is a Lambda **instance type**, not a GPU model. `gpu_1x_a100_sxm4` is
40 GB and will not hold a GLM-5.3-Flash streaming measurement (needs 63 GB);
the smallest type that does is `gpu_1x_h100_sxm5`.

### JarvisLabs

```bash
export JL_API_KEY=... HF_TOKEN=$(cat ~/.hf_token)
./bin/measure-cloud --model <hf-repo> \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane streaming --spot --gpu H200 --max-cost 20 --max-runtime 8h --yes
```

## 4. Money, and not leaking a box

`--max-cost` has **no default**. Pass it. It refuses when the estimate's *band
high* exceeds the number, before creating anything.

Cost is reported four ways — estimated / computed / billed / balance-delta —
because any one of them can lie.

Teardown is four independent layers, because any one can fail:

* **L0** controller trap on EXIT/INT/TERM/HUP plus `atexit`
* **L1** on-instance watchdog with an absolute deadline and a heartbeat
* **L2** a lease file under `~/.fidelity-cloud/leases/`, swept by a reaper on
  *your* machine with *your* credentials — no secret leaves it
* **L3** the deadline encoded in the instance NAME (`fidcloud-<job>-exp<epoch>`),
  so `measure-cloud reaper --sweep` can clean up from any machine with the account

```bash
./bin/measure-cloud reaper --install     # once
./bin/measure-cloud reaper --list
./bin/measure-cloud reaper --sweep --dry-run
```

**One thing to know if you drive this from a script:** launching the run and
then waiting for it *in the same shell invocation* means a timeout on that
invocation kills the whole process group, run included. It tears down cleanly —
that is L0 working — but you paid for the setup. Launch, return, poll
separately.

## 5. What does NOT travel between providers

The arithmetic does. The hardware does not.

Bitwise determinism is a **per-device** property. Our determinism evidence is
"N cold runs produced one distinct tokenwise-KLD tensor hash" *on one device*.
Two H200s from two vendors ought to agree; an H200 and an A100 should not be
assumed to. Reproducing a number on different silicon is a **result worth
publishing**, not an assumption worth making — which is why `measure-cloud`
refuses a non-validated GPU until you name it explicitly, and records
`on_validated_hardware=false` when you do.

This bites hardest on Vast, whose cheapness comes from renting whatever a host
happens to own: different drivers, different host CPUs, sometimes different
silicon under one GPU name. Use it for work whose output is content-digested
and verified — `verify` recomputes the whole digest chain before teardown — and
not for establishing a determinism claim.

## 6. Adding a fifth provider

The contract is eighteen methods; `docs/CLOUD-PROVIDERS.md` §1 lists them and
says which are load-bearing for safety. The SSH half is already written
(`fidelity/sshbase.py`), so a new backend is lifecycle plus catalogue.

Declare `separable_storage = False` unless the provider's disk genuinely
outlives its instance. Getting that wrong is not a create error — it is
`No space left on device` three stages into a paid run.

The honest test that a backend works is **not** that it ran. It is
re-measuring an artifact that already has a sealed receipt and getting the same
number: for `turboderp/GLM-5.3-Flash-exl3` @ 2.05bpw that is
`0.12163767673339457` and its tokenwise-KLD tensor hash.
