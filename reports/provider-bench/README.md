# provider-bench — the raw evidence behind `docs/CLOUD-COMPARISON.md`

One JSON per rental, written by `bin/fidelity-bench --json`. Nothing here was
edited by hand; the tables in the comparison document are generated from these
files and can be regenerated at any time:

```bash
python3 bin/provider_bench_table.py reports/provider-bench              # per card
python3 bin/provider_bench_table.py reports/provider-bench --per-rental # per receipt
python3 bin/provider_bench_table.py reports/provider-bench --inject docs/CLOUD-COMPARISON.md
```

**Naming.** `<provider>-<card>-s<N>.json`, where `N` indexes independent
rentals of the same card. Two rentals of one SKU at one price are not the same
machine — four RunPod *secure* H100 rentals at a flat $3.29/h spread 2.2x — so
a single sample is an anecdote and the table reports best/median/worst rather
than a mean.

**What is in a receipt.** The payload's measurements (device read, host→device
cold and warm, PCIe link at idle and under load, an expert-shape bf16 GEMM, a
dense GEMM, the per-matrix step) plus a `rental` block recording the hourly
rate **read back off the instance that was billing**, its `rate_source`
(`contract:<field>` or `catalogue`), the region, and the UTC timestamp. That is
what makes `$/window` derivable from the artifact instead of from a note.

**`exhibits/`** holds receipts that are deliberately excluded from the
aggregate because they document a failure mode rather than a card:

* `vast-ask48665056-listed-b200-got-h100.json` — a Vast offer listed as a
  B200 (179 GB, Virginia) that came up an H100 80GB HBM3 whose host→device
  bandwidth *fell* under sustained load, 27.6 GB/s cold to 13.8 GB/s warm.

**`lambda-capacity-poll.jsonl`** is a two-minute poll of Lambda's
`regions_with_capacity_available` for eight single-GPU instance types, taken
across the survey. It is the evidence for the capacity claims in the comparison
document — in particular that `gpu_1x_h100_sxm5` was never available in any
poll.

Snapshot: 2026-08-31 UTC. Prices move; re-run before trusting the numbers.
