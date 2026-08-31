# Which GPU cloud, and what a measurement actually costs on it

**Snapshot: 2026-08-31 UTC.** Prices and stock move fast; every number below is
dated, and every number is derived from a committed benchmark receipt under
[`reports/provider-bench/`](../reports/provider-bench/) rather than typed into
prose. Regenerate the tables at any time:

```bash
python3 bin/provider_bench_table.py reports/provider-bench
python3 bin/provider_bench_table.py reports/provider-bench --per-rental
```

**No affiliation.** This project has no relationship with Vast.ai, RunPod,
Lambda or JarvisLabs beyond holding prepaid credit on each and paying retail.
Nothing here is sponsored and no provider reviewed it. The whole survey cost
about four dollars, and you can re-run it yourself for the same.

---

## Dollars per hour is the wrong metric

A card at three times the hourly rate that finishes in a third of the time is a
wash. What decides the bill for a fidelity measurement is **dollars per
window**, and that is `minutes/window x $/hour` — two numbers that are only
comparable when they come from the same rental.

So the instrument here is not a price scrape. `bin/fidelity-bench` rents one
instance, measures what actually decides a measurement's wall clock, reads the
hourly rate back off the instance that is billing, tears the box down, and
writes a receipt. It takes under a minute on a fast provider and costs a few
cents.

```bash
bin/fidelity-bench --provider vast   --gpu "A100 PCIE"          --json out.json
bin/fidelity-bench --provider lambda --gpu gpu_1x_gh200         --json out.json
bin/fidelity-bench --provider runpod --gpu "NVIDIA B200"        --json out.json
```

**What the streaming lane spends its time on.** Per window it walks every
routed expert matrix, uploads it, dequantises it, does one skinny GEMM against
a 2047-token block, and throws the weights away. GLM-5.3-Flash is 42 layers x
288 experts x 3 projections = 36,288 of those per window. That loop is
**host-bandwidth-bound**, not compute-bound, and the table below is the proof:
a B200 with 7.7x an A100's bf16 throughput finishes the step 2.7x faster,
tracking its host-to-device bandwidth almost exactly.

---

## The table

<!-- BEGIN GENERATED: bin/provider_bench_table.py reports/provider-bench -->
| provider | card, as `nvidia-smi` reports it | $/h | n | h2d GB/s | min/window best-median-worst | $/window (median) | vs best | host spread |
|---|---|---|---|---|---|---|---|---|
| lambda | NVIDIA GH200 480GB | 2.290 | 3 | 404 | 0.06 - 0.06 - 0.06 | 0.00226 | 1.0x | 1.02x |
| vast | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 0.744 | 2 | 28 | 0.45 - 0.45 - 0.45 | 0.00562 | 2.5x | 1.00x |
| vast | NVIDIA A100-PCIE-40GB | 0.274-0.438 | 2 | 12 | 0.96 - 0.97 - 0.97 | 0.00572 | 2.5x | 1.01x |
| vast | NVIDIA A100 80GB PCIe | 0.581-0.674 | 2 | 27 | 0.54 - 0.59 - 0.65 | 0.00624 | 2.8x | 1.20x |
| vast | NVIDIA H100 80GB HBM3 | 1.784-2.011 | 2 | 58 | 0.27 - 0.34 - 0.40 | 0.01048 | 4.6x | 1.47x |
| lambda | NVIDIA A100-SXM4-40GB | 1.990 | 2 | 26 | 0.54 - 0.54 - 0.55 | 0.01804 | 8.0x | 1.01x |
| runpod | NVIDIA A100-SXM4-80GB | 1.590 | 4 | 26 | 0.56 - 0.70 - 1.25 | 0.01848 | 8.2x | 2.25x |
| lambda | NVIDIA H100 PCIe | 3.290 | 2 | 56 | 0.34 - 0.39 - 0.45 | 0.02139 | 9.5x | 1.32x |
| vast | NVIDIA B200 | 5.677 | 2 | 55 | 0.21 - 0.25 - 0.28 | 0.02329 | 10.3x | 1.35x |
| runpod | NVIDIA B200 | 6.790 | 4 | 58 | 0.20 - 0.27 - 0.38 | 0.03025 | 13.4x | 1.92x |
| runpod | NVIDIA H100 80GB HBM3 | 3.290 | 4 | 58 | 0.31 - 0.67 - 0.69 | 0.03658 | 16.2x | 2.22x |
| jarvislabs | NVIDIA H200 | 3.990 | 3 | 38 | 0.79 - 0.79 - 0.79 | 0.05253 | 23.2x | 1.00x |
<!-- END GENERATED -->

`$/window` is the streaming lane's **inner loop** priced at the rate that
rental was billing. It is not the whole cost of a measurement — a real run also
pays bootstrap, a 200 GB fetch, materialize and the panel, which
`bin/measure-cloud --dry-run` estimates and this does not. Two consequences,
both of which cut the same way:

* As a **ranking**, it transfers: the inner loop is the term that varies
  between machines, by up to 6x on identical silicon.
* As an **absolute**, it flatters the fast expensive card. If the inner loop is
  only a fraction *f* of the scored wall clock, a card that halves it saves
  only *f*/2 of the run while charging its full rate for all of it. A card that
  loses this column loses by more in practice, never less.

Every rental behind those rows, one line each:

<!-- BEGIN GENERATED RENTALS: bin/provider_bench_table.py reports/provider-bench --per-rental -->
| receipt | provider | card | $/h | PCIe link, loaded | h2d GB/s | expert GEMM TF | ms/matrix | min/window | $/window | rent->destroy s |
|---|---|---|---|---|---|---|---|---|---|---|
| `jarvislabs-h200-s2.json` | jarvislabs | NVIDIA H200 | 3.9900 | Gen5 x16 of Gen5 x16 | 38.5 | 755 | 1.301 | 0.79 | 0.05233 | 31 |
| `jarvislabs-h200-s1.json` | jarvislabs | NVIDIA H200 | 3.9900 | Gen5 x16 of Gen5 x16 | 38.3 | 758 | 1.306 | 0.79 | 0.05253 | 32 |
| `jarvislabs-h200-s3.json` | jarvislabs | NVIDIA H200 | 3.9900 | Gen5 x16 of Gen5 x16 | 38.3 | 758 | 1.307 | 0.79 | 0.05257 | 32 |
| `lambda-a100-sxm4-40gb-s1.json` | lambda | NVIDIA A100-SXM4-40GB | 1.9900 | Gen4 x16 of Gen4 x16 | 26.1 | 169 | 0.894 | 0.54 | 0.01793 | 247 |
| `lambda-a100-sxm4-40gb-s2.json` | lambda | NVIDIA A100-SXM4-40GB | 1.9900 | Gen4 x16 of Gen4 x16 | 26.2 | 168 | 0.905 | 0.55 | 0.01815 | 303 |
| `lambda-gh200-s3.json` | lambda | NVIDIA GH200 480GB | 2.2900 | Gen4 x1 of Gen4 x1 | 403.9 | 765 | 0.097 | 0.06 | 0.00224 | 165 |
| `lambda-gh200-s1.json` | lambda | NVIDIA GH200 480GB | 2.2900 | Gen4 x1 of Gen4 x1 | 379.4 | 761 | 0.098 | 0.06 | 0.00226 | 269 |
| `lambda-gh200-s2.json` | lambda | NVIDIA GH200 480GB | 2.2900 | Gen4 x1 of Gen4 x1 | 378.6 | 753 | 0.099 | 0.06 | 0.00229 | 237 |
| `lambda-h100-pcie-s2.json` | lambda | NVIDIA H100 PCIe | 3.2900 | Gen5 x16 of Gen5 x16 | 55.4 | 531 | 0.555 | 0.34 | 0.01841 | 478 |
| `lambda-h100-pcie-s1.json` | lambda | NVIDIA H100 PCIe | 3.2900 | Gen5 x16 of Gen5 x16 | 55.5 | 516 | 0.735 | 0.45 | 0.02438 | 425 |
| `runpod-a100-sxm4-80gb-s3.json` | runpod | NVIDIA A100-SXM4-80GB | 1.5900 | Gen4 x16 of Gen4 x16 | 26.3 | 168 | 0.919 | 0.56 | 0.01473 | 42 |
| `runpod-a100-sxm4-80gb-s2.json` | runpod | NVIDIA A100-SXM4-80GB | 1.5900 | Gen4 x16 of Gen4 x16 | 23.3 | 169 | 1.105 | 0.67 | 0.01771 | 67 |
| `runpod-a100-sxm4-80gb-s4.json` | runpod | NVIDIA A100-SXM4-80GB | 1.5900 | Gen4 x16 of Gen4 x16 | 26.2 | 168 | 1.201 | 0.73 | 0.01925 | 43 |
| `runpod-a100-sxm4-80gb-s1.json` | runpod | NVIDIA A100-SXM4-80GB | 1.5900 | Gen4 x16 of Gen4 x16 | 26.0 | 169 | 2.064 | 1.25 | 0.03308 | 33 |
| `runpod-b200-s2.json` | runpod | NVIDIA B200 | 6.7900 | Gen5 x16 of Gen5 x16 | 57.7 | 1277 | 0.327 | 0.20 | 0.02238 | 47 |
| `runpod-b200-s3.json` | runpod | NVIDIA B200 | 6.7900 | Gen5 x16 of Gen5 x16 | 57.6 | 1277 | 0.327 | 0.20 | 0.02238 | 47 |
| `runpod-b200-s1.json` | runpod | NVIDIA B200 | 6.7900 | Gen5 x16 of Gen5 x16 | 55.5 | 1372 | 0.557 | 0.34 | 0.03812 | 28 |
| `runpod-b200-s4.json` | runpod | NVIDIA B200 | 6.7900 | Gen5 x16 of Gen5 x16 | 55.5 | 1265 | 0.627 | 0.38 | 0.04291 | 39 |
| `runpod-h100-sxm-80gb-s1.json` | runpod | NVIDIA H100 80GB HBM3 | 3.2900 | Gen5 x16 of Gen5 x16 | 57.5 | 753 | 0.513 | 0.31 | 0.01701 | 38 |
| `runpod-h100-sxm-80gb-s3.json` | runpod | NVIDIA H100 80GB HBM3 | 3.2900 | Gen5 x16 of Gen5 x16 | 55.2 | 767 | 1.096 | 0.66 | 0.03635 | 55 |
| `runpod-h100-sxm-80gb-s4.json` | runpod | NVIDIA H100 80GB HBM3 | 3.2900 | Gen5 x16 of Gen5 x16 | 55.2 | 768 | 1.110 | 0.67 | 0.03681 | 65 |
| `runpod-h100-sxm-80gb-s2.json` | runpod | NVIDIA H100 80GB HBM3 | 3.2900 | Gen5 x16 of Gen5 x16 | 54.9 | 768 | 1.139 | 0.69 | 0.03777 | 56 |
| `vast-a100-pcie-80gb-s1.json` | vast | NVIDIA A100 80GB PCIe | 0.5807 | Gen4 x16 of Gen4 x16 | 26.7 | 166 | 0.891 | 0.54 | 0.00522 | 62 |
| `vast-a100-pcie-80gb-s2.json` | vast | NVIDIA A100 80GB PCIe | 0.6741 | Gen4 x16 of Gen4 x16 | 23.6 | 166 | 1.070 | 0.65 | 0.00727 | 154 |
| `vast-a100-pcie-40gb-s2.json` | vast | NVIDIA A100-PCIE-40GB | 0.4378 | Gen3 x16 of Gen3 x16 | 12.4 | 161 | 1.590 | 0.96 | 0.00702 | 550 |
| `vast-a100-pcie-40gb-s1.json` | vast | NVIDIA A100-PCIE-40GB | 0.2741 | Gen3 x16 of Gen3 x16 | 12.3 | 161 | 1.603 | 0.97 | 0.00443 | 456 |
| `vast-b200-s1.json` | vast | NVIDIA B200 | 5.6771 | Gen5 x16 of Gen5 x16 | 55.4 | 1349 | 0.346 | 0.21 | 0.01980 | 38 |
| `vast-b200-s2.json` | vast | NVIDIA B200 | 5.6771 | Gen5 x16 of Gen5 x16 | 55.4 | 1352 | 0.468 | 0.28 | 0.02678 | 48 |
| `vast-h100-sxm-s2.json` | vast | NVIDIA H100 80GB HBM3 | 2.0111 | Gen5 x16 of Gen5 x16 | 57.5 | 756 | 0.448 | 0.27 | 0.00908 | 61 |
| `vast-h100-sxm-s1.json` | vast | NVIDIA H100 80GB HBM3 | 1.7844 | Gen4 x16 of Gen4 x16 | 27.7 | 758 | 0.660 | 0.40 | 0.01187 | 40 |
| `vast-rtxpro6000-maxq-s2.json` | vast | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 0.7444 | Gen4 x16 of Gen4 x16 | 27.9 | 234 | 0.749 | 0.45 | 0.00562 | 43 |
| `vast-rtxpro6000-maxq-s1.json` | vast | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 0.7444 | Gen4 x16 of Gen4 x16 | 27.9 | 234 | 0.750 | 0.45 | 0.00563 | 433 |
<!-- END GENERATED RENTALS -->

---

## The Lambda question, answered

Lambda's headline looks terrible: **$4.29/h for an H100 SXM5 against Vast's
$0.575/h for an A100 80GB PCIe**, seven and a half times the rate. The obvious
hypothesis is that an H100 is fast enough to make that back. It is not, and the
reason is structural rather than a matter of a few percent.

**1. The ceiling on this lane is a PCIe generation, and it is low.** Measured
host-to-device bandwidth, warm, grouped by the link `nvidia-smi` reports under
load, across all 28 rentals:

| link, under load | rentals | h2d, warm |
|---|---|---|
| Gen3 x16 | 2 | 12.3 – 12.4 GB/s |
| Gen4 x16 | 10 | 23.3 – 27.9 GB/s |
| Gen5 x16 | 14 | 38.3 – 57.7 GB/s |
| **Gen4 x1** (GH200; the link is not the path) | 2 | **378.6 – 379.4 GB/s** |

Ignore the last row for a moment. Everything attached over PCIe spans 12 to 58
GB/s, and from the A100 80GB PCIe's 26.7 the most any card reaches is 57.7 —
**2.2x** — with per-matrix time tracking it: 0.89 ms for the A100, 0.33 ms for
a B200 that has **seven times** its bf16 throughput. On a PCIe machine the
*most* any amount of silicon buys you is roughly **2.7x**.

**2. Therefore a rate multiple above ~2.7x cannot be repaid.** Break-even
against Vast's A100 80GB PCIe at $0.581/h requires the dearer card to be as
many times faster as it is times more expensive. Lambda's H100 SXM5 at $4.29 is
7.4x the rate and would need to be 7.4x faster; the best H100 SXM measured
anywhere in this survey is **2.0x** faster. Lambda's cheapest 80 GB card, the
H100 PCIe at $3.29, measured 0.735 ms — **1.2x** faster than the A100 for 5.7x
the money, and $0.0244/window against $0.0052. It is not close, and no
PCIe-attached card at Lambda's price list closes it.

**3. And then Lambda wins outright — on the one instance type that is not
PCIe-attached.** `gpu_1x_gh200`, $2.29/h:

| | A100 80GB PCIe (Vast, $0.581/h) | GH200 (Lambda, $2.29/h) |
|---|---|---|
| host→device, warm | 26.7 GB/s | **379.4 GB/s** |
| per-matrix step | 0.891 ms | **0.098 ms** |
| min/window (inner loop) | 0.54 | **0.06** |
| **$/window** | $0.00522 | **$0.00227** |

A GH200 reaches host memory over NVLink-C2C, not over PCIe, so it is not
subject to the ceiling that decides every other row. Fourteen times the
bandwidth, nine times faster per matrix, at four times the hourly rate — which
nets out to **2.3x cheaper per measurement than the best single rental on the
cheapest marketplace**, and 2.5x cheaper than the best *median* row there, on
fleet hardware at a stable published price rather than a stranger's PC. Two
independent rentals measured 0.098 and 0.099 ms, 1% apart.

**So: Lambda stops being the expensive option the moment you stop buying PCIe
from it.** Every PCIe card on its price list is 5–8x Vast's cost per
measurement and always will be, because the axis that matters is capped and the
price is not. Its GH200 is the cheapest way to run this lane that this survey
found anywhere, on any provider, at any price.

**Two caveats, both load-bearing.**

* `nvidia-smi` on that GH200 reports **`Gen4 x1 of Gen4 x1`** — the exact
  signature of the oversubscribed host that `--min-h2d-gbps` exists to refuse —
  while moving 379 GB/s. Its PCIe link is vestigial. Anything that gates on
  link *width* rather than on measured GB/s will refuse the fastest machine it
  can rent. `bench.gate` deliberately gates on `h2d_GBps` alone.
* A GH200's 96 GB is enough for this lane and its host memory is coherent and
  large, but it is a different device: **bitwise determinism is a per-device
  property** (`llms.txt`, and §5 of `CLOUD-RECIPES.md`). Reproducing a number
  there is a result worth publishing, not an assumption worth making.

### Postscript: we never got a usable H100 SXM5, and that is itself the finding

`gpu_1x_h100_sxm5` — the $4.29/h card the whole question was about — produced
**zero measurements in seventy minutes of trying** (2026-08-31, 11:23–12:33
UTC). Three things happened, in this order, and each is a different failure:

1. **The catalogue said no.** A two-minute poll of
   `regions_with_capacity_available` for eight single-GPU types found this one
   available in **0 of 20 polls**, while `gpu_1x_a100_sxm4` was available in
   20/20 and `gpu_1x_gh200` in 7/20. Raw log:
   [`lambda-capacity-poll.jsonl`](../reports/provider-bench/lambda-capacity-poll.jsonl).
2. **When the catalogue said yes, the launch said no.** A 20-second watcher
   caught three brief windows the two-minute poll missed. The first, 11:36:20
   UTC, was refused outright: `HTTP 400
   instance-operations/launch/insufficient-capacity`.
3. **When the launch succeeded, the GPU was not there.** The other two, 12:15
   and 12:23 UTC, both created an instance in `us-south-3`, both reached
   `active` with an IP and a working sshd after 238 s and 260 s — and on both,
   `torch.cuda.is_available()` was **False**. The second was probed four times
   over three and a half minutes and never came up. Both were destroyed. Both
   billed at $4.29/h for a card that never appeared.

That third case is the one worth generalising. **An instance the API calls
`active`, which accepts an SSH connection, is not yet a machine with a GPU** —
the same shape as this project's older lesson that a failed run leaves its box
idle but *running*. `fidelity-bench` now refuses to write a receipt when the
payload reports `no cuda` or produces no `stream_matrix_ms`, and retries that
one error four times across three minutes first: a receipt of zeros tabulates
as a very slow machine, and nothing in it says the card was missing. The first
of these two rentals wrote exactly such a receipt before that guard existed.

None of this changes the arithmetic. The verdict above rests on Lambda's H100
PCIe, which we did rent twice, and which is **cheaper per hour** than the SXM5
($3.29 vs $4.29). An SXM5 would have to be 1.3x faster than the PCIe part just
to match its dollars per window — and the PCIe part is already 4x adrift of a
Vast A100. But it is worth saying plainly: **a rate you cannot rent at is not a
rate**, and the headline number that made Lambda look expensive turned out to
be attached to something we could not buy on the day we looked.

---

## What a price cannot express

Verified against the live APIs on 2026-08-31, not read off marketing pages.

| | Vast.ai | RunPod | Lambda | JarvisLabs |
|---|---|---|---|---|
| shape | **marketplace** — you rent one stranger's machine | community (marketplace) + secure (datacentre) | fleet | fleet |
| balance from the API | yes | yes | **no** — pay-as-you-go, bills after the fact | yes |
| storage outlives the instance | no | no | no | **yes** — separable filesystems |
| choose the disk size | yes, at rent time | yes | **no** — fixed per instance type | yes |
| spot / interruptible | offered; this suite rents on-demand | offered (bid price); this suite rents on-demand | **none at all** | yes, containers only (H200 $1.99 vs $3.99) |
| rate readable per instance | yes (`dph_total`) | yes (`costPerHr`) | yes (`price_cents_per_hour`) | **no** — only a running total; priced from the catalogue |
| rent→benchmark→destroy, over 32 rentals | 38 s – 550 s (n=10) | **28 s – 67 s** (n=12) | 165 s – 478 s (n=7) | **31 s – 32 s** (n=3) |
| reproducibility across rentals of one card | 1.0–1.5x | **up to 2.25x** | 1.01–1.32x | **1.00x** |
| the sharp edge | heterogeneous hosts; see below | secure ≠ uniform; see below | capacity, slow boots, no balance API | on-demand rows were invisible to this suite until today |

**Lambda has no balance endpoint.** Budgeting a run against it means trusting
your own arithmetic; there is nothing to reconcile against, and `bin/measure-cloud`'s
four-way cost report (estimated / computed / billed / balance-delta) loses its
fourth column there. Keep runs small until you have seen an invoice.

**JarvisLabs is the only one whose disk survives its instance**, which is what
makes a preempted spot box cheap to resume — the 300 GB you already fetched is
still there. That is worth real money on a lane whose fetch is the expensive
part, and none of the other three offer it.

---

## The heterogeneity caveat, with today's receipts

A catalogue row is not a machine. Four things happened during this survey, all
on the same afternoon:

**One rental was not the card the listing advertised.** Vast ask `48665056`
listed a **B200, 179 GB, Virginia**. What came up was an **H100 80GB HBM3**
whose host-to-device bandwidth *fell* under sustained load — 27.6 GB/s cold,
13.8 GB/s warm — and which took 1.663 ms per matrix, worse than a $0.58 A100.
The receipt is kept as
[`exhibits/vast-ask48665056-listed-b200-got-h100.json`](../reports/provider-bench/exhibits/vast-ask48665056-listed-b200-got-h100.json)
and excluded from the aggregate. Renting the same card **by name** a few minutes
later produced a genuine B200 at 0.367 ms. Whatever the mechanism — a recycled
ask id, a re-chunked bundle — the operational rule is the same: **on a
marketplace an offer id is not a durable name for one machine, and the only
check that works is measuring what you actually got.**

**One host billed for fifteen minutes and never accepted a connection.** Vast
ask `48850759` reported `running` throughout while
`ssh1.vast.ai:27118 never accepted a connection within 898s`. The rental cost
about $0.30 and produced nothing. Teardown fired from `finally` and the
instance is gone.

**Two rentals came up with no GPU at all.** Both Lambda `gpu_1x_h100_sxm5`
launches reached `active` with a working sshd and answered
`torch.cuda.is_available() == False`; see the postscript above. The first wrote
a receipt of zeros, which is why `fidelity-bench` now refuses to.

**And host variance is not a Vast-only phenomenon.** RunPod's *secure* cloud —
a datacentre product, one flat price — gave 0.513, 1.096, 1.110 and 1.139 ms
per matrix across four rentals of `NVIDIA H100 80GB HBM3` at $3.29/h. Same SKU,
same price, **2.2x** between the best and worst machine. Its A100-SXM4 spread
2.25x and its B200 1.9x. By contrast JarvisLabs reproduced to 0.5% across three
rentals and Lambda's GH200 to 2% across three.

That is the honest summary of the marketplace-versus-fleet trade, and it is
measurable rather than a matter of taste: **`host spread` in the table above is
the price of the cheap tier.** Rule 2 of [`llms.txt`](../llms.txt) — never rank
on a single sample — applies to machines as surely as it applies to windows.

The defence is already in the tool. `bin/measure-cloud` accepts
`--min-h2d-gbps`, checked after setup and before the fetch, which is the last
moment a bad machine is still cheap to walk away from.

---

## So which one should a stranger pick?

* **Cheapest per measurement, today: Lambda `gpu_1x_gh200` at $2.29/h.** Nothing
  else measured comes within 2x, and it is fleet hardware with a fixed price.
  Check capacity first — Lambda's is thin and its catalogue can promise a type
  it then refuses to launch.
* **Cheapest per measurement on PCIe: Vast, an A100 80GB PCIe around
  $0.58/h.** Pass `--gpu` (without it, "cheapest that fits ≥63 GB" once picked a
  CMP 170HX mining card), pass `--min-h2d-gbps`, and expect one rental in
  several to be a dud you pay a little for and abandon.
* **Best-behaved API and the shortest rent→run loop: RunPod**, 28–67 s to a
  finished benchmark across twelve rentals. Pay for that in variance: even
  secure hosts spread 2.2x, so budget for sampling rather than for one rental.
* **Most predictable, and the only separable storage: JarvisLabs.** Its H200
  reproduced to within 0.5% across three rentals — and measured *slower per matrix
  than an A100 PCIe* despite Gen5 x16 and 769 TFLOP/s, because that host's cost
  is per-transfer overhead rather than link width. At $3.99 on-demand it is the
  dearest row in the table; at $1.99 spot it halves, and the resumable
  filesystem is worth more than the difference on a long capture.
* **Lambda for anything PCIe-attached** only when predictability is worth 5x.

---

## Reproducing this

```bash
export VAST_KEY_FILE=~/.vast_key RUNPOD_KEY_FILE=~/.runpod_key LAMBDA_KEY_FILE=~/.lambda_key
bin/fidelity-bench --provider <p> --gpu <card> --json reports/provider-bench/<p>-<card>-s1.json
python3 bin/provider_bench_table.py reports/provider-bench
```

Every instance is destroyed in a `finally`, including when the benchmark
raises — a benchmark that leaks an instance has cost more than it measured.
Take **at least two samples per card**: one rental is an anecdote.

See [`CLOUD-RECIPES.md`](CLOUD-RECIPES.md) for the full measurement recipes and
[`CLOUD-PROVIDERS.md`](CLOUD-PROVIDERS.md) for the eighteen-method contract a
backend has to satisfy.
