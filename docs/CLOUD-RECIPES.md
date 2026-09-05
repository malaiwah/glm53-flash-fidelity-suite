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
and delete reserve (`--retrieval-delete-reserve`; by default the retrieval
contract's own minimum, derived from the result archive bound and printed in
the plan — 13818 s for a 5 GB archive). If that exceeds `--max-cost` the run
is refused, together with every other arithmetic finding of the same plan
(timing bound, publication destination, unresolved leases) in one report. There is no default cap; a cap the
tool picked would turn a legitimate run into a refusal you cannot attribute.

**Absolute deadline — `--max-runtime`.** The workload deadline is written into
the durable lease (the reaper destroys the pod at it), the on-pod watchdog and
the provider's own timer. For a root whose bound is authored in
`bin/engines.json` `root_timing_profiles` it defaults to that bound (the plan
says `defaulted to the authored bound`); for any other target it is required. The provider timer is a hint, never evidence of
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
    --max-cost 65 --max-runtime 7h30m --retrieval-delete-reserve 14400 \
    --out ~/fidelity-runs/<name> --dry-run
```

The two numbers are the tool's own for this target: `--max-runtime` must be
at least the bound authored in `bin/engines.json`
`root_timing_profiles` for `zai-org/GLM-5.3-BF16@304b8051` on an H200
(26925 s, so `7h30m`), and `--max-cost` must cover the all-in maximum the
controller computes from that deadline plus the reserve ($62.925 at $4.59/h
on 2026-09-05). When the bound moves, the dry-run refuses with the new
number; `bin/selftest_readme_recipes.py` fails when a documented recipe
falls below the authored bound.

`--dry-run` runs every check, prints the plan (target, authored bound and its
derivation, GPU and rate, datacenter, the hard cap labelled as the bound,
the expected spend when the row carries measured components, storage, the
reserve and its derivation, and every gate) and spends $0.00; the full plan
JSON goes to `--plan-json FILE` or, with `--json`, to stdout. Re-run the
same command without `--dry-run` to spend; the interactive prompt quotes the
calculated maximum and the hard cap, and only `y`/`yes` permits the single
create POST. `--yes` skips the prompt.

Required: `--provider`, `--model`, `--revision` (for a paid run;
`--dry-run` resolves and prints `main`'s commit when it is omitted),
`--panel-dir`, `--dataset-id`, `--measurer`, `--max-cost`, `--out`, and
`--max-runtime` unless the target has an authored timing row. `--role` defaults to `quant`; a root capture and the candidate route
below both pass `--role root`. `--publish-root-to` and `--hf-token-file`
are only needed when the dataset is to be published from this machine after
teardown; without them the sealed dataset stays under `--out`
([QUICKSTART §4](THIRD-PARTY-QUICKSTART.md) shows `fidelity-dataset publish`
for publishing it later).

### The candidate route: a quant against a published root

Every GLM-5.3 quant row was measured this way — the root protocol on a
quantized target, scored on the pod against the published root dataset. It
is `--role root` plus four flags that go together: `--candidate-scope`
(authored from the checkpoint index with `engines/tools/exl3_scope.py` or
`fp8_scope.py`), `--candidate-codec`, `--candidate-bits` and
`--reference-dataset OWNER/REPO@40HEX`. A quant has no authored timing row,
so `--gpu H200` (the root's GPU) and your own `--max-runtime` are the bound:

```bash
bin/measure-cloud --provider runpod --role root \
    --model <owner>/<quant> --revision <40-hex> \
    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
    --dataset-id fidelity--glm53.<hub-handle>.quant.<slug> \
    --candidate-scope engines/scopes/scope--<slug>.json --candidate-codec exl3-mcg --candidate-bits 3.25 \
    --reference-dataset malaiwah/glm53-fidelity-root-v1@9c4a29ee10f393ed2fdbdb9262c1192ddb1507b4 \
    --gpu H200 --runpod-datacenter US-NC-1 \
    --hf-token-file ~/.hf_token --measurer <hub-handle> \
    --max-cost 45 --max-runtime 3h30m --retrieval-delete-reserve 14400 \
    --out ~/fidelity-runs/<name> --dry-run
```

Observed on H200 / US-NC-1 (JOURNAL 2026-09-05): a 394 GB EXL3 candidate
takes ~33–45 min of pod time, ≈ $3–4; the `$45` is the hard cap the plan
prints, not the estimate. The pre-spend gates read the trellis/FP8 identity
from bytes, verify the panel is exact for the reference root, bind the
reference dataset's seal and content digest into the job, and resolve the
unexpected-tensor allowlist from the authored table — or, for a pin without
a row, derive it from the checkpoint's index at plan time
(`engines/tools/index_census_allowlist.py`), bind it by both digests as
`inputs/allowlist.json` and record the gate's provenance as
`derived_from_index`; a row in `bin/fidelity/runpodsafety.py` is the
attestation that upgrades it to `authored`. The walkthrough with the observed
dry-run output is [QUICKSTART §3b](THIRD-PARTY-QUICKSTART.md).

Derived unless you override them: GPU from the target's authored timing
evidence (`--gpu` when it has none); pod storage from the checkpoint plus both
cold captures (`--storage`); host vCPU and memory minima from the model bytes
(`--min-vcpu`, `--min-memory-gb`); `--dataset-repository` from
`--publish-root-to`; `--dataset-name` from `--dataset-id`; the
unexpected-tensor allowlist from the authored evidence for the target, else
from its index census; `--max-runtime` from the authored bound and
`--retrieval-delete-reserve` from the retrieval contract; the
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

`--list` prints one block per unresolved lease — state, the pod ids it
authorizes and whether they are in the account inventory now, created/
workload/reap times with ages, the last event — and a count of settled
leases (`--all` lists them) plus the timer's `health`. A lease in state
`AMBIGUOUS` with no pod id (create was attempted, no pod of the exact name
was ever observed) blocks new runs with *an earlier lease may still hold a
pod*; the reaper cannot settle it by design (`cloudlease.py` yields
`ambiguous-needs-operator`), so the block names the wrong-name pods that
appeared in its create window, whether any still exists, and the operator
act: verify in the RunPod console that no pod of that name exists, then pass
`--allow-unresolved-leases` to proceed beside it — the reaper still destroys
anything past its deadline. `--sweep --dry-run` prints every action row the
sweep would take (`would-…`, `ambiguous-needs-operator`, `billing-pending`);
a real `--sweep` destroys only exact ids authorized by leases this tool
wrote. Never delete, pause or adopt a resource you did not create.

Billing settles after the pod is gone: RunPod publishes an hourly bucket only
when the hour closes, so a run whose pod vanished at :44 leaves the lease
`ABSENCE_CONFIRMED` with `billing: settled: false` in `terminal-receipt.json`
and the reaper closes it (`reconciled: true`) on a sweep after the hour plus
300 s; a closure is never sealed over an unpublished bucket.
