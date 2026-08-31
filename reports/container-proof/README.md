# The container acceptance test, 2026-08-31

One Lambda `gpu_1x_a10` box (NVIDIA A10, driver 570.148.08), one GPU, suite
revision `f45a8be`. The same checkpoint
(`inference-optimization/GLM-5.3-Flash-0.1B-A0.1B` @ `7c3a6d3d`) captured twice
over the same 2x256 panel — 510 scored positions, built on the box from this
repository's own documents — and driven by the same `bin/container_entry.py`,
so the only variable between the arms was the environment.

* **arm A** — the current path: `bootstrap_measure.sh` on the host (python3.12
  from deadsnakes, the pinned wheel set, the pipeline at its pin + patches),
  then `stage_measure.sh setup / fetch_target / capture / verify`.
* **arm B** — the image: `docker run --gpus all -v …:/workspace …
  quant-fidelity-measure:proof capture …`, the same stages inside the
  container.

## Result

| | arm A | arm B |
|---|---|---|
| `capture_content_digest` | `b42ffe8f1d1dfcfdd78452339cdcd913c8be9ceae13f88f6348f19b43a960549` | **identical** |
| window 0 tensor | `552af179f75d99fd9c1c32e74fddc12919c521f40b66372d05da25c242bc3a0d` | **identical** |
| window 1 tensor | `abd137acbc04ab2536e9f8b1d606cafc84aeb9c18b169f1b7cc3cf913b87987f` | **identical** |
| head tensor | `58b4b967abf4b663cc8827413ff2feddeb524347f426f3bf3dad3ecb105bbab6` | **identical** |
| `stack_fingerprint_sha256` | `18735425224674010b93195cf2260ee9ebe1e813e6e902957c6b649ce4778746` | **identical** |
| `lane_identity_sha256` | `2d0992fc5dbbf694f458d97486dfadab9f7ad2c0cb8ce089f4373c078a85eff8` | **identical** |
| `dataset_sha256` | `4d8eaae9…` | `587cbd6f…` |
| `runtime.container.image_digest` | `null` | `sha256:372d542ccf7e4deb8908ad3949b515200dfb66fdff3980690bf7ad8e5f457d5a` |
| python | 3.12.13 (deadsnakes) | 3.12.3 (Ubuntu 24.04) |
| `setup` stage | 134 s cold | 53–70 s |

`dataset_sha256` differs for one reason and it is the intended one: arm B's
runtime receipt records the image it ran in. `stack_fingerprint_sha256` is
equal, which is what `dscompare` reads to decide `stack_relation` — so the two
captures are same-stack and either can serve as the other's floor.

The two arms did not share an interpreter patch version and the tensors are
still bit-identical. That is evidence about how much of the stack the digest
pins; it is not a claim that patch versions never matter.

## What is here

| file | what it is |
|---|---|
| `armA/…/fidelity-dataset.json`, `armB/…` | the two sealed manifests |
| `armA/…/capture/manifest.json`, `armB/…` | the per-window tensor digests the table above compares |
| `armA/…/runtime/capture-runtime.json`, `armB/…` | the runtime receipts; the `container` block is the only structural difference |
| `armB/fidelity/job.json` | the contract the container wrote, including `produced_by` and the environment block |
| `panel/` | the panel and its receipt, so the selection rule can be re-run |
| `armA2.txt`, `armB2.txt` | the two runs' stage logs |
| `build2.txt` | the image build (amd64) |

## What it does not prove

A 0.1B fixture over two windows exercises the plumbing, not the arithmetic of a
real measurement: it certifies that the image computes the same bytes as the
host bootstrap on one GPU, and nothing about any published number. Determinism
is a per-device property (`docs/ARCHITECTURE-DETERMINISM.md`), so this
comparison is only meaningful because both arms ran on one box.

`fidelity-dataset compare --self-compare` over the two trees is **refused**, and
correctly: `PANEL-D6` compares tokenizer identity, which a capture records as
the local path of the model tree — `/home/ubuntu/…` vs `/workspace/…`. See
`docs/REVIEW-DEFERRED.md`.
