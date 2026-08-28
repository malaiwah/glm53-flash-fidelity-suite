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
`local-cuda-budget` lanes all point at `k6/tools/stream_score.py`, which is
**not in this checkout** — it was built and validated on the streaming box and
has not been synced back. Each carries the contract it needs.

When that file lands:

```bash
bin/measure-local --probe-engines     # scrapes --help, reports found/missing flags
```

then set `pinned: true` and fill `flag_map` for those three lanes. Nothing else
changes.

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
