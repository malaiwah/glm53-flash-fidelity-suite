# Contributed receipts

One file per submission, at `receipts/<your-hf-handle>/<slug>.json`, in the sealed
`quant-fidelity-registry/submission-receipt.v1` format. See `../CONTRIBUTING.md`.

The maintainer's own receipts live here too, under `receipts/malaiwah/`, in whatever sealed
family the runner that produced them writes -- they are not submission receipts and do not
self-seal. They are here for the same reason yours are: a row that cites
`receipts/<handle>/<file>.json` cites bytes in this repository that anyone can fetch and hash,
rather than a path on the machine that ran it. `data/measurements.jsonl` carries the sha256 of
each one, and `tools/registry_selftest.py` rebuilds the rows from them on every run.

This is the **only** directory a contributor adds to. `data/`, `index.json` and `README.md`
are generated from what lands here; `schema/` is the contract. CI fails a PR that edits them.

Check yours before you send it — offline, no installs:

```bash
python3 tools/registry_validate.py --submission receipts/<handle>/<slug>.json
```

It prints the row it would generate, its comparability key, its class, and the rows it
could be ranked against — or exactly which check it failed.
