# Cloud providers: the contract, and a wish list

This suite rents GPUs to capture roots and measure quants. Today it can only
rent them from **one** provider, JarvisLabs, because `bin/fidelity/jlapi.py`
is the only backend and `measure_cloud.py` calls it directly. Nothing about
the measurement science requires that, and depending on a single vendor is a
liability for a project whose whole claim is that its numbers are
reproducible by other people.

This document says exactly what a provider has to be able to do before the
driver can use it, scores the plausible candidates against that, and names the
one thing that is *not* a portability problem but looks like one.

## 1. The contract

Extracted from what the driver actually calls, not from what providers
advertise. A backend must supply all of **A**, or the safety guarantees this
suite makes are not true on it.

### A. Required — a run cannot be safe without these

| capability | why the driver needs it | JL method |
|---|---|---|
| enumerate offers: GPU model, VRAM, $/hr, spot flag, region, free count | the fit check refuses a GPU that cannot hold the model, and the cost band is computed before anything is created | `gpus()` |
| create with an explicit disk size | roots are 100 GB–1.5 TB; a fixed-disk provider cannot host a capture | `create()` |
| **destroy**, idempotent and confirmable | L0/L2/L3 teardown; "confirmable" means a later query can prove it is gone | `destroy()` |
| list instances **from any machine on the account** | L3 name-deadline sweep runs from a laptop that did not create the instance | `list_instances()` |
| set an instance **name** at creation | the deadline is encoded in it (`fidcloud-<job>-exp<epoch>`), which is the only teardown layer that survives losing the lease file | `create(name=)` |
| exec a shell command, returning stdout **and exit code** | every stage is one exec; a provider that returns only logs cannot tell a finished stage from a dead one | `exec()` |
| upload / download files | the 74-file bundle in, the receipts tree out | `upload()` / `download()` |

### B. Strongly wanted — buys money or safety, not correctness

| capability | what it buys |
|---|---|
| spot / interruptible instances | ~2-3x cheaper; the stage design is already preemption-tolerant (every stage is receipt-resumable) |
| queryable **billed cost** per instance | the run reconciles estimated / computed / billed / balance-delta, because any one of them can lie |
| account balance | refuse to start a run the account cannot pay for |
| persistent filesystem separable from the instance | a preempted spot instance keeps its 300 GB of fetched weights |
| region selection | data-residency, and fetch bandwidth to the Hub |

### C. Nice to have

Startup scripts; VPC; an SDK that is not a CLI subprocess; per-second billing.

## 2. The wish list

Scored on the contract above. **Nothing here has been tested** — this is a
survey to decide what to try, and every row needs its A-column claims verified
against the live API before it is trusted.

| provider | spot | disk control | exec + exit code | notes |
|---|---|---|---|---|
| **JarvisLabs** (current) | yes, containers | yes, and separable filesystems | yes, via `jl exec` | the reference implementation; the only one actually exercised |
| **RunPod** | yes (community + secure) | yes, network volumes | yes, REST + SSH | closest match to the contract; probably the cheapest port |
| **Vast.ai** | yes, marketplace | yes | yes, SSH | cheapest per FLOP, **but see §3** — heterogeneous hosts are a methodological problem, not just an ops one |
| **Lambda** | limited | fixed per instance type | SSH | reliable, simple, fewer spot options; good for a *known-hardware* lane |
| **CoreWeave** | yes | yes | k8s-native | enterprise-shaped; heavier to drive than the others |
| **Modal** | serverless | ephemeral + volumes | function-shaped, not shell-shaped | would need the stage model rewritten; strong for the *comparison* step |
| **Paperspace / DigitalOcean** | limited | yes | SSH | |
| **Nebius / DataCrunch / Prime Intellect / SF Compute** | varies | varies | SSH | worth pricing for the 1.5 TB GLM-5.3 root specifically |

**Which to do first.** RunPod, because it satisfies column A without
qualification and its spot pricing is comparable to JarvisLabs, so the port can
be validated by re-running a measurement we already have a sealed receipt for
and checking the number is bitwise identical.

> **Open question for Michel:** which of these do you already hold credits on?
> That should decide the order, ahead of anything in this table.

## 3. The thing that is not a portability problem, and the one that is

**Not a problem: the arithmetic.** Full-vocabulary KLD in fp64 is
deterministic given the same weights, the same panel and the same reduction
order. Nothing about it is vendor-specific.

**Actually a problem: the hardware, and it is already half-solved.**
`measure-cloud` refuses to run on a GPU the lane was not validated on, because
both constants the plan depends on — minutes/window and the observed VRAM peak
— were *measured* on an H200, and it records `on_validated_hardware` in the
plan. That guard is provider-agnostic and it is what makes a multi-provider
world safe: a new provider does not weaken any claim, it just needs its own
validated-hardware entry.

Two things follow that must not be glossed over:

1. **Bitwise determinism is a per-device property, not a global one.** Our
   determinism evidence is "N cold runs produced one distinct tokenwise-KLD
   tensor hash" — on one device. Two H200s from two vendors should agree, and
   an H200 and an A100 should *not* be assumed to. Cross-device reproduction is
   a result worth publishing, not an assumption worth making.
2. **Vast.ai's heterogeneity is the sharp edge.** Its cheapness comes from
   renting whatever a host happens to own — different driver versions, different
   host CPUs, sometimes different silicon under one GPU name. That is fine for
   capture *throughput* and hostile to a claim of bitwise reproduction. If we
   use Vast, it should be for work whose output is content-digested and verified
   (`verify` recomputes the whole digest chain before teardown), never for
   establishing a determinism claim.

## 4. What porting actually costs

The driver is already close to portable and was not designed to be, which is
luck rather than foresight:

* `Teardown` (all four layers), the lease file, the deadline-encoded name, the
  cost reconciliation and every stage in `stage_measure.sh` are provider-neutral
  — they speak the contract in §1, not JarvisLabs.
* `jlapi.py` is 489 lines and is the entire vendor surface.

So the port is: define `Provider` as the §1-A protocol, rename the current
class to `JarvisLabsProvider`, add `--provider`, and write the second backend.
The honest test that it worked is not "it ran" — it is **re-measuring an
artifact we already have a sealed receipt for and getting the same number**,
which for `turboderp/GLM-5.3-Flash-exl3` @ 2.05bpw means reproducing
`0.12163767673339457` and its tokenwise-KLD tensor hash.
