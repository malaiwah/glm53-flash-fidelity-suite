# `docs/` — the fidelity dataset format

Scoring is three separable steps, one tool, three modes:

```
step 1   capture   reference (root) weights + panel  ->  fidelity dataset A     publish: REQUIRED for a root
step 2   capture   quantized weights  + panel        ->  fidelity dataset B     publish: OPTIONAL
step 3   compare   A, B                              ->  KLD + determinism + a registry-submittable receipt
                   A, A                              ->  reproduction confirmation, exactly 0.0
```

Capture and comparison are fused today, so every measurement re-pays for capture, teachers are
non-portable, and a lost capture kills reproducibility — which already happened. Separating them
makes a root capture a public good, lets quant authors contribute captures without our
infrastructure, and largely dissolves the same-lane floor problem.

| document | what it is |
|---|---|
| [`FIDELITY-DATASET-SPEC.md`](FIDELITY-DATASET-SPEC.md) | the format: layout, manifest schema, seal, root-vs-quant matrix, hidden-vs-logit form, head identity and the comparator refusal rules, panel binding, stack fingerprint, determinism evidence, kimi-k3 interop, our own defects it fixes, and what is out of scope |
| [`CARD-ANNOTATION-SPEC.md`](CARD-ANNOTATION-SPEC.md) | machine-readable fidelity provenance on a HuggingFace card: a conformant `model-index` result plus a small additive `x_fidelity` block, with the Hub's real validation behaviour measured rather than assumed |
| [`FIDELITY-DATASET-BUILD-PLAN.md`](FIDELITY-DATASET-BUILD-PLAN.md) | exact new file names, CLI signatures, what each command validates and refuses, which existing code each wraps (never edits), the synthetic test matrix, registry changes, and open items |
| [`REGISTRY-INTEGRATION.md`](REGISTRY-INTEGRATION.md) | the three additive `registry/` changes a step-3 receipt needs — specified, and deliberately not applied while a concurrent workflow holds those files open |
| [`RACE-MODE.md`](RACE-MODE.md) | capturing a root while the checkpoint is still downloading, for the hours-long window after a model lands: the priority-ordered overlapped fetch and what it measured, why the preview and the final are DIFFERENT identities rather than two versions of one, the generation sanity check every capture now runs, and an honest section on the tension between racing and trustworthy numbers |
| [`cards/`](cards/) | the annotation applied to our own K6 and K8 cards: the reference implementation, generated and validated, not published |

## The tooling

Built to this spec, all new files, nothing existing edited:

```
bin/fidelity-dataset      capture | verify | validate | compare | adapt | describe | publish
bin/fidelity-card         annotate | validate
bin/fidelity/dsformat.py     the five digest preimages, the seal, checksums.txt, path rules
bin/fidelity/dsmanifest.py   builders + the DatasetWriter that lays out the tree in seal order
bin/fidelity/dsvalidate.py   JSON Schema (via the registry's vendored _minischema) + ~40 rules
bin/fidelity/dscompare.py    the gate ladder + the fp64 estimator (k6_kld_report._token_kld)
bin/fidelity/dsadapt.py      k3v1 | k3v0-window | malaiwah-serving-v2 | llamacpp-kld
bin/fidelity/dshub.py        digest-driven fetch, refusing publish
bin/fidelity/cardmeta.py     the card generator and its three validation axes
```

Usage is in [`../bin/README.md`](../bin/README.md). The 110 selftest cases are
`bin/selftest_fidelity_{dataset,compare,card}.py`, registered in
`bin/selftest_all.sh`; every case names the spec rule it exercises.

## Schemas

Written in the keyword subset `registry/tools/_minischema.py` implements, so they validate offline on
a stock interpreter with no `pip install`:

```
schema/fidelity-dataset.schema.json              the capture manifest
schema/fidelity-comparison-receipt.schema.json   the step-3 output
schema/fidelity-card-annotation.schema.json      the x_fidelity card block
```

## Worked examples

Every digest is either a real value from a published artifact or a registry row, or a deterministic
placeholder listed in [`examples/SYNTHETIC-DIGESTS.md`](examples/SYNTHETIC-DIGESTS.md). **The seals
are always real** — the manifests and receipts self-verify as shipped.

```
examples/fidelity-dataset.root-glm53-bf16.json          step 1, a hypothetical GLM-5.3-Flash BF16 root
examples/fidelity-dataset.quant-glm53-k6.json           step 2, our K6 quant
examples/fidelity-comparison-receipt.k6-vs-bf16.json    step 3, a measurement
examples/fidelity-comparison-receipt.self-compare.json  step 3, A == B -> exactly 0.0
examples/card-k6.yaml  card-k8.yaml  card-root-bf16.yaml  card-dataset-suite-v1.yaml
```

Verify any of them without importing anything of ours:

```bash
python3 - <<'EOF'
import json, hashlib
d = json.load(open("docs/examples/fidelity-dataset.quant-glm53-k6.json"))
body = dict(d); body["dataset_sha256"] = ""
canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
print(hashlib.sha256(canon.encode()).hexdigest() == d["dataset_sha256"])
EOF
```

That four-line recipe is the same one the registry documents to contributors and the same one that
seals `reports/stack-provenance-retro.json`.
- [CAPTURE-SCALING-PLAN.md](CAPTURE-SCALING-PLAN.md): plan of record for scaling captures — why tensor-parallel is rejected on correctness, why pipeline-parallel is adopted, and what each model family costs to measure.
