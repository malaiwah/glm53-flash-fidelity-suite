# `bin/` — the two recipes

`measure-cloud` and `measure-local` are the user-facing product: a stranger
with a quant and a GPU should be able to paste one line and get a sealed,
submittable number. `registry-submit` checks that number the way the registry
will, before it is sent anywhere.

## Why here and not in `k6/tools/`

`k6/tools/` is campaign-scoped — its name is a campaign, its contents assume
GLM-5.3-Flash and the K6 encode. The runners are meant for people who have
never heard of K6 and are measuring some other model entirely. Putting them at
the top level alongside `registry/` is what makes the pair read as a product
rather than as internal tooling.

The measurement **engines** stay where they are. `bin/` orchestrates; it does
not measure. That split is enforced by `engines.json`.

## Layout

| Path | What it is |
|---|---|
| `measure-cloud`, `measure-local`, `registry-submit` | one-line wrappers, so the headline paste has no `python3` in it |
| `measure_cloud.py` | the cloud controller: preflight, fit, instance selection, cost, four-layer teardown, reaper |
| `measure_local.py` | the local runner: device discovery, memory solver, micro-benchmark, refusals |
| `fidelity/census.py` | **the shared, testable core** — model census, VRAM/disk/RAM arithmetic, the memory solver. Pure stdlib, no torch, no network. |
| `fidelity/hfmeta.py` | revision pinning, blob sizes, surface sniffing, panel descriptors. A few hundred KB answers "will this work?" |
| `fidelity/jlapi.py` | the single chokepoint for every `jl` call |
| `fidelity/engines.py`, `engines.json` | which scorer each lane invokes, and how |
| `fidelity/receipt.py`, `seal_receipt.py` | build and seal a `submission-receipt.v1` |
| `stage_measure.sh`, `watchdog.sh`, `invoke_engine.py` | the on-instance side |
| `BUNDLE.txt` | exactly what gets uploaded to rented hardware |
| `selftest_fit.py`, `selftest_decode_parity.py` | offline; run them before trusting a plan |

## Engine pinning

A lane whose engine is not `pinned: true` in `engines.json` **refuses to plan**.
It does not guess flags. A plausible-looking wrong flag is how you spend an hour
of H200 time discovering that `--reduce-order` was spelled `--reduce_order`.

Today `sealed-ep8` is pinned and flag-verified against
`k6/tools/k6_student_capture.py`. The `streaming`, `local-mps` and
`local-cuda-budget` lanes point at `k6/tools/stream_score.py`, which **is now
in this checkout** (recovered 2026-08-28 from the validation box's shared
filesystem, sha256 `b7411804…60acae`) and stays `pinned: false` on purpose.

`bin/measure-local --probe-engines` reports why, and the guess-nothing rule
earned its keep: the contract that was written for these lanes was wrong in
every guessed spelling. The engine takes `--teacher` not `--panel`, `--source`
not `--surface`, `--vram-budget-gb` not `--vram-budget`, `--slab-experts` not
`--expert-chunk`; `--window-batch`, `--kld-device` and `--nonrouted-residency`
do not exist at all.

Fixing the spellings would not be enough, and this is the thing to fix first:

* `--profile` accepts only `k6|k8|k6k8`, and the controller sends `k4` for
  these lanes.
* **Every source path resolves to a packed root** and requires
  `contract.json`, `inventory.json`, `mtp-adapter-receipt.json` and
  `payload-store/{objects,choices}` — this campaign's own encode output — plus
  a `--bf16` tree. `--source dione` raises *"not enabled in this build"*.

So no lane can currently read a third-party `tr3-published` artifact, which is
what a stranger's quant almost always is. Until a `tr3-published` reader
exists, `--lane streaming` on someone else's repo is refused at plan time by
the surface check (`engines.json` → `surfaces`), for $0.00, instead of after
the rental.

## Adding a new engine or surface

1. Add the entrypoint and `required_flags` to `engines.json`.
2. Declare `surfaces` — the artifact kinds it can actually open. This is what
   stops a rental for bytes nothing can read; leaving it empty disables the
   check.
3. `bin/measure-local --probe-engines` until every required flag is found.
4. Only then `pinned: true` with a filled `flag_map`.

## Selftests

```bash
bin/selftest_all.sh                    # everything below, ~2 min, spends nothing
python3 bin/selftest_fit.py            # 33 known-answer checks, no GPU, no network
python3 bin/selftest_decode_parity.py  # needs torch; ~1 min
```

`selftest_fit.py` checks the census against independently measured figures and
the solver against four known devices, including two that must be **refused**.
`selftest_decode_parity.py` extracts the real decode functions from the reader
source by AST — no vendored copy, so it cannot drift — and proves on your
machine that the decode is pure PyTorch and bitwise identical across devices.
That is what lets a local receipt claim its only device offset is in the
forward pass.
