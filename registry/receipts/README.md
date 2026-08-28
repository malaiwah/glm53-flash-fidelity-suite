# Contributed receipts

One file per submission, at `receipts/<your-hf-handle>/<slug>.json`, in the sealed
`quant-fidelity-registry/submission-receipt.v1` format. See `../CONTRIBUTING.md`.

This is the **only** directory a contributor adds to. `data/`, `index.json` and `README.md`
are generated from what lands here; `schema/` is the contract. CI fails a PR that edits them.

Check yours before you send it — offline, no installs:

```bash
python3 tools/registry_validate.py --submission receipts/<handle>/<slug>.json
```

It prints the row it would generate, its comparability key, its class, and the rows it
could be ranked against — or exactly which check it failed.
