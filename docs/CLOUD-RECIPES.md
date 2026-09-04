# Measuring on rented GPUs

`bin/measure-cloud` rents one RunPod GPU pod, runs a fidelity measurement on
it, pulls the sealed result back and destroys the pod. Paid measurement is
RunPod-only and SSH-only; JarvisLabs, Vast, Lambda, spot instances, recovery,
adoption and race mode are refused before provider mutation.

`bin/measure-cloud --help` is the ground truth for every flag below. The
executable walkthrough is [`THIRD-PARTY-QUICKSTART.md`](THIRD-PARTY-QUICKSTART.md);
this document explains the boundary.

## What is always enforced

These four hold on every paid run. Nothing else is required to start one.

**Cost cap — `--max-cost`.** Before anything is created the controller
computes the all-in maximum liability: the live GPU rate for the whole
deadline, storage for the deadline at the tariff defaults, and the retrieval
and delete reserve (`--retrieval-delete-reserve`, default 21600 s). If that
exceeds `--max-cost` the run is refused. There is no default cap; a cap the
tool picked would turn a legitimate run into a refusal you cannot attribute.

**Absolute deadline — `--max-runtime`.** The workload deadline is written into
the durable lease (the reaper destroys the pod at it), the on-pod watchdog and
the provider's own timer. The provider timer is a hint, never evidence of
cleanup; the lease is what the reaper enforces.

**Teardown on every exit path.** Success, failure, exception and Ctrl-C all
request deletion of the exact pod id the run created. A pod is "gone" only
when provider inventory proves exact absence; `EXITED` is not absence.
Retrieval exhaustion still deletes the pod.

**Autonomous reaper.** `measure-cloud reaper --provider runpod --install` puts
a user-systemd timer on this machine that reads the leases and destroys any
pod past its deadline, even if the controller process is dead. Every paid run
refuses unless that timer is installed, active and its user manager survives
logout (`loginctl enable-linger`). The timer executes a sealed snapshot; a
checkout that has since moved on is a `source_drift` warning in the dry-run
plan, not a refusal — the installed reaper still guards the run. Reinstall to
pick up the newer checkout. An install sealed under the older v2 control
schema is refused with the same reinstall command.

## The recipe

Once per machine and RunPod account:

```bash
bin/measure-cloud reaper --provider runpod --install
```

Then the minimal root capture, exactly as the `--help` epilog shows it:

```bash
bin/measure-cloud --provider runpod --role root \
    --model zai-org/GLM-5.3-BF16 --revision <40-hex> \
    --panel-dir engines/panels/<panel> \
    --dataset-id fidelity--<id> --publish-root-to <owner>/<repo> \
    --hf-token-file ~/.hf_token --measurer <hub-handle> \
    --max-cost 40 --max-runtime 3h30m --retrieval-delete-reserve 14400 \
    --out ~/fidelity-runs/<name> --dry-run
```

`--dry-run` runs every check, prints the plan and spends $0.00. Re-run the
same command without `--dry-run` to spend; the interactive prompt quotes the
calculated maximum and the hard cap, and only `y`/`yes` permits the single
create POST. `--yes` skips the prompt.

Required: `--provider`, `--role`, `--model`, `--revision` (for a paid run;
`--dry-run` resolves and prints `main`'s commit when it is omitted),
`--panel-dir`, `--dataset-id`, `--measurer`, `--max-cost`, `--max-runtime`,
`--out`. `--publish-root-to` and `--hf-token-file` are only needed when the
dataset is to be published from this machine after teardown; without them the
sealed dataset stays under `--out`.

Derived unless you override them: GPU from the target's authored timing
evidence (`--gpu` when it has none); pod storage from the checkpoint plus both
cold captures (`--storage`); host vCPU and memory minima from the model bytes
(`--min-vcpu`, `--min-memory-gb`); `--dataset-repository` from
`--publish-root-to`; `--dataset-name` from `--dataset-id`; the
unexpected-tensor allowlist from the authored evidence for the target; the
download token from `--hf-token-file` (`--hf-download-token-file` to ship a
separate read-only token to the pod); the RunPod key from
`~/.config/runpod/api_key` (`--runpod-key-file`); on-demand, secure cloud and
fail-on-preempt. Every derived value is printed in the dry-run plan.

Each run also gets its own ledger under the reaper state directory with
ceiling = `--max-cost`. Pods in the account that this tool did not create are
tolerated. An earlier lease that may still hold a pod refuses the run and
names the lease; `--allow-unresolved-leases` proceeds anyway, and the reaper
destroys that pod at its own deadline regardless.

## Strict campaign mode (opt-in)

Use it when the RunPod account is dedicated to this suite, when several
attempts must share one ceiling, or when you want a sealed proof that the
installed reaper really destroyed a pod after the controller died. All four
flags go together:

```text
--campaign-ledger FILE --campaign-ceiling USD --campaign-reserve USD --campaign-reaper-margin USD
```

The ledger is a locked file beside `--lease-dir` that accounts for every
attempt against one ceiling, refuses admission beside pods it does not own,
and holds liability until billing settles. `--campaign-width 2` is admitted
only with a verified published root archive for the exact root identity.

`measure-cloud drill` is the paid controller-loss drill: it creates one small
pod, kills its own controller, and seals `proof.json` only after the
user-systemd reaper issued the exact-id destroy at the lease deadline,
inventory proved absence and billing settled. Pass that file as
`--runpod-safety-proof` (requires `--campaign-ledger`) and it is validated
exactly as before: it binds to this exact checkout, this ledger and this
account, and a stale or foreign proof is refused. The `--help` epilog shows
both commands.

| mechanism | default mode | strict campaign mode |
|---|---|---|
| safety proof | not required | `--runpod-safety-proof` validated against this checkout, ledger and account |
| campaign ledger | auto-created per run, ceiling = `--max-cost`, foreign pods tolerated | one explicit locked ledger; admission refused beside pods it does not own |
| billing settlement | advisory; the reaper settles it after teardown | liability held in the ledger until billing settles |
| reaper health | snapshot integrity; checkout drift is a warning | the same |

## RunPod: pin the datacenter, watch the dashboard

Hub fetch throughput on RunPod secure H200 hosts differed **10x** on
2026-09-04 with the same repository, command and container-disk layout:

| pod host | datacenter (ipinfo) | fetch rate | 750 GB fetch |
|---|---|---:|---:|
| `103.196.86.20`, `.112`, `.136` | Raleigh NC (`US-NC-1`) | 1.3-2.9 s per 5 GB shard, ~1.7-2.4 GB/s | ~12 min |
| `152.236.142.242` | Denver CO | 15-28 s per shard, ~240 MB/s | ~52 min |

Three attempts landed on the slow host in a row (RunPod re-offers the same
box). At $4.59/h the slow fetch alone is ~$4 per attempt. The receipts of
those runs did not record where the pod ran; they do now
(`machine.data_center_id` / `location` in the live attestation), and
`--runpod-datacenter US-NC-1` pins the create. A pin **refuses** when the
datacenter has no stock; it never falls back elsewhere. Stock per datacenter:

```
query { dataCenters { id gpuAvailability(input: {secureCloud: true}) { available stockStatus gpuTypeId } } }
```

The stage driver mirrors its `stage_measure/<stage>:` lines to the
container's PID 1 stdout, so the RunPod dashboard **Logs** tab shows stage
progress for a detached run without SSH. That stream is advisory; the
per-stage log files retrieved through `--result-sink` are the evidence.

## RunPod: the measurement image on the safe SSH path

`--runpod-image ghcr.io/malaiwah/quant-fidelity-measure@sha256:<digest>` boots the
`:ssh` target of the measurement image (sshd + the locked stack baked at
`/opt/fidelity`). The bootstrap seeds the per-attempt venv and pipeline from the
image when the wheel lock matches, so `setup` is seconds instead of ~7 minutes of
pip; the live attestation probes CUDA through the image venv and records
`cuda.interpreter`. Proven 2026-09-04 on an L40S (US-MO-1): Fruit root, two
cold runs bitwise (`d75e830c…`), qualified, torn down, $0.53. The digest differs
from the published L4 root because determinism is per device, not because of
the image. The `:ssh` tag is amd64 only; pin the digest, never the tag.

## Credentials and identity

- RunPod API bytes come from an owner-only mode-0600 regular file
  (`--runpod-key-file`). They never appear in argv, logs, receipts or
  bundles.
- Target identity is resolved anonymously from `https://huggingface.co`. The
  target download on the pod uses the read token from
  `--hf-download-token-file` (default: `--hf-token-file`), transported as a
  0600 file in a 0700 directory and shredded right after `fetch_target`.
  Panels remain anonymous.
- An ED25519 public key must exist locally before create. The controller
  reads the fresh pod's ED25519 fingerprint from RunPod's authenticated
  container-log stream, compares it to the network keyscan, and connects
  with `StrictHostKeyChecking=yes`; there is no fingerprint prompt or TOFU.
- The write token in `--hf-token-file` stays on the controller and is used
  only for optional publication.

## Publication

Publication is optional and controller-local. With `--publish-root-to`, the
qualified dataset is pushed from this machine after verified retrieval and
provider-confirmed absence of the pod; the token never reaches the pod.
Without it the sealed dataset stays under `--out`. Billing is advisory: if
RunPod has not published the bucket yet, the lease closes on proven absence
and the reaper settles billing later.

## Exit codes

| code | meaning |
|---|---|
| 0 | ok |
| 1 | the run failed and the pod is proven gone |
| 3 | refused before anything was created ($0.00) |
| 90 | a pod may remain — run `bin/measure-cloud reaper --provider runpod --list` |

## Emergency inspection

```bash
bin/measure-cloud reaper --provider runpod --list
bin/measure-cloud reaper --provider runpod --sweep --dry-run
```

`--list` shows every lease with its state and whether the timer is healthy.
A real `--sweep` destroys only exact ids authorized by leases this tool wrote.
Never delete, pause or adopt a resource you did not create.
