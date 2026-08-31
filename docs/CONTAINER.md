# Running the measurement as a container

The SSH path rents a *machine*: create it, parse the id it answers with, poll
its state, size its disk, find where its filesystem is mounted, upload a
bundle, bootstrap it, and remember to destroy it. Porting that to three clouds
produced **five defects, and not one of them was about the measurement**:

| defect | what it cost |
|---|---|
| machine ids that are not integers | crashed *after* the pod existed — twice — leaving a billing instance unadopted |
| the running state spelled `"Running"` vs `"RUNNING"` | two healthy polls read as a preemption; a box torn down mid-bootstrap |
| those ids compared as ints inside a set | the "is it really gone?" check would report a **live** instance destroyed |
| storage sized 100 GB | correct only where the disk is separable; `No space left on device`, 45 min into a paid run |
| three hardcoded `/home/jl_fs` roots | the worst stalled an A100 at **0% GPU for two hours** at $1.59/h — $3.07 before anyone noticed |

Every one is an artefact of orchestrating a machine instead of running an
image. A container has no id to parse, no state to poll, no disk to size, and
its filesystem root is a mount the caller chose. This is that image.

**The SSH path is not removed.** JarvisLabs is driven through its own CLI and
has no custom-image path at all, and a new transport proves itself before it
replaces a working one. `bin/measure-cloud` is unchanged in behaviour.

---

## What is in the image

`bin/bootstrap_measure.sh` is the specification, and the Dockerfile **runs that
script** rather than paraphrasing it — so there is exactly one definition of
the environment and the image cannot drift from it. `bin/selftest_container.py`
rung C9 fails the build if the Dockerfile ever grows its own `pip install
torch`.

| baked | why |
|---|---|
| `python3.12` on plain `ubuntu:24.04` | the proven env recipe is py3.12-only, and the bootstrap asserts it. Deliberately **not** a vendor pytorch template: the RunPod one ships torch *without* numpy and its system python is PEP 668 externally-managed, which killed two runs |
| the pinned wheel set (`torch==2.11.0+cu130`, `transformers==5.16.1`, …) | reproducible numbers. torch's cu130 wheels carry their own CUDA userspace; only `libcuda.so` comes from the host |
| the quant pipeline at its pin + patches `0001-0006`, `0008` | the reader bytes every sealed streaming row binds |
| `bin/BUNDLE.txt`'s audited file set | the same list the SSH uploader uses — one list, two transports |
| `/opt/fidelity/BUILD.json` + `image-pin.txt` | what the build actually resolved, and a content digest over it |

What is **not** baked: the run root. `/workspace` is a `VOLUME`, always.
Defaulting it into the image would put a 200 GB fetch on the container's own
layer — the shape of the bug that had a restarted pod skip `setup` as done
while the tree it wrote had evaporated.

Steps 5 and 6 of the bootstrap (the four offline surface batteries) run at
**container start**, not at build: the GGUF battery's rung 1b re-decodes the
committed real bytes on *this box's* CUDA device, and a builder has no GPU.
Running it there would either fail a good build or pass vacuously.

### The image contains no credential

The HF token arrives at runtime — `--token-file`, or `HF_TOKEN` — is written to
`$FS/.secrets/hf_token` 0600, which is the file `stage_measure.sh load_token`
already reads, and is then **removed from the environment the stages see**. It
never reaches argv, which is the property `bin/measure_cloud.py` has and must
keep. Rung C4 asserts all four halves of that.

---

## Building

```bash
container/build.sh --tag <registry>/<name>:<tag>
```

It refuses a dirty checkout. `produced_by.revision` is required by the
submission schema and has no "unknown" value — an image built from uncommitted
edits would stamp every receipt with a commit that does not describe its own
bytes. (`--allow-dirty` exists for a throwaway build and says so.)

The build context is ~10 MB: `.dockerignore` excludes everything and lets back
in only the four trees the image needs, then re-excludes the heavy evidence
directories — while re-including the two files inside them that `BUNDLE.txt`
actually lists, because a setup-time selftest opens them fail-closed. Rung C5
checks every bundle entry against those patterns, since an over-eager exclusion
does not fail the build: it produces an image that dies in `setup` on a rented
box.

---

## Driving it

The entrypoint mirrors the CLI and writes the **same `job.json` contract**
`stage_measure.sh` already reads. It does not reimplement a stage:
`bin/stage_measure.sh` owns what a stage does and `bin/fidelity/stages.py` owns
which stages run, and the SSH controller now asks that same module — so the two
transports cannot drift into different sequences.

```bash
# a reference (root) capture
docker run --gpus all --rm \
    -v /data/run:/workspace \
    -v $PWD/engines/panels/panel--x:/panel:ro \
    -e HF_TOKEN \
    <image> capture \
        --model <repo> --revision <40-hex> \
        --panel-dir /panel --dataset-id fidelity--x.malaiwah.root.bf16 \
        --lane streaming --gpu "NVIDIA A10"

# measuring a quantized artifact
docker run --gpus all --rm -v /data/run:/workspace -e HF_TOKEN \
    <image> measure \
        --model <repo> --revision <40-hex> --surface exl3hf --bits 4.0 \
        --profile k6 --panel-descriptor /panel.json --lane streaming

# one stage, against a job document that already exists
docker run --gpus all --rm -v /data/run:/workspace <image> stage measure

# what is this image, and what can it see?
docker run --gpus all --rm <image> doctor
```

Useful everywhere: `--dry-run` prints the job document and the stage list and
creates nothing; `--job FILE` uses a planner-written document verbatim;
`--only` / `--stop-after` narrow the sequence; `--image-pin <digest>` tells the
run which image the launcher pulled.

### What it refuses rather than guesses

`measure` without `--profile` is refused, naming
`bin/engines.json`'s `profile_map_by_surface` as the remedy. The cloud planner
resolves that from the *sniffed* surface and bit rate; in a container nothing
sniffs, and guessing is not the smaller failure — `k6` is a real profile naming
a real receipt family, so a wrong guess publishes a wrong label instead of
crashing. Use `--job` to carry a planner-resolved document.

### Resuming

Unchanged: every stage writes a `.done` marker under `$FS/receipts/done`, and a
capture whose dataset already exists is skipped. Re-running the same command
against the same mount resumes; the suite is re-synced into the mount by digest,
so a second start copies nothing.

---

## Which image ran, recorded

Two receipt fields that have been `null` on **every** capture this repository
has ever sealed now have an answer.

* `runtime.container.image_digest` — filled by `engines/tools/hf_capture.py` from
  `STACKPRINT_IMAGE_PIN` or the baked pin file, the convention
  `fidelity/stackprint.py` already reads. `docker load` strips the registry
  digest, so the file is the only identity that survives every transport; the
  build also writes an `image_content_sha256` over the resolved wheel versions,
  the pipeline commit, every applied patch and every bundled file.
* `produced_by.container_image` / `container_digest` — an SSH-driven instance
  has no git checkout, so the controller must compute `produced_by` on the
  caller's laptop and ship it. An image carries the revision, so a
  containerised run names its own code.

**Both are outside `stack_fingerprint_sha256`, on purpose.** That digest is
what `dscompare` reads to decide `stack_relation`, and a cross-stack verdict
stamps `usable_as_floor: false`. The container is *where* the stack ran, not
*what the stack is* — so recording it must not make a containerised capture
incomparable with the identical computation run outside one. Rung C8 asserts
that, and asserts that with no pin present the runtime receipt is byte-identical
to what it was before the field learned how to be filled: a published dataset
does not get to shift because we added a container.

---

## The acceptance test

**Bit-identical output.** A containerised capture must produce the same
`capture_content_digest` as the current path on the same GPU. If it does not,
the image changed the arithmetic — that is a failure, not a variance.

```bash
# same box, same GPU, same panel, same checkpoint
#   arm A: the current path      -- bin/stage_measure.sh setup && ... capture
#   arm B: the container         -- docker run ... capture
jq -r '.capture.capture_content_digest' A/fidelity-dataset.json
jq -r '.capture.capture_content_digest' B/fidelity-dataset.json
```

`dataset_sha256` is *expected* to differ between the two: arm B's runtime
receipt records the image it ran in and arm A's does not. That is the honest
answer, not a discrepancy — the tensors are the claim, and
`fidelity-dataset compare --self-compare` over the two trees is the exact-0.0
reproduction confirmation.

Determinism is a **per-device** property (`docs/ARCHITECTURE-DETERMINISM.md`:
two A100s in two clouds agree bitwise; an H200 is 2.973e-04 nats away, 13× the
gap this registry publishes between two 4-bit quantizers). The comparison above
is only meaningful on **one** GPU.

---

## Running it on a provider

| provider | custom image | how |
|---|---|---|
| RunPod | yes | `imageName` on `podFindAndDeployOnDemand` — `fidelity.runpodapi.RunPod.create` already takes `image=` |
| Vast.ai | yes | image chosen at rent time |
| Lambda | instances are **VMs**, not containers | `docker build` and `docker run` directly on the box; no registry involved |
| JarvisLabs | no | stays on the SSH path |

A RunPod or Vast pod must **pull** the image, so it has to exist in a registry
they can reach. A Lambda instance does not: it is a real VM with Docker, so the
image can be built and run on the box itself, which is also the cheapest way to
get the two arms of the acceptance test onto one GPU.

**Nested containers do not work on RunPod.** Probed on a live pod
(`NVIDIA RTX A4000`, community cloud, driver 580.126.20): running as uid 0 but
with `CapEff 0x00000000a80425fb` — the Docker default set, no `CAP_SYS_ADMIN` —
`unshare -U -r true` fails and `/dev/fuse` does not exist. So rootful podman
(needs `CAP_SYS_ADMIN` for mount namespaces) and rootless podman (needs
`CLONE_NEWUSER`, which the default seccomp profile blocks without
`CAP_SYS_ADMIN`) are both out. Building the image *on* a RunPod pod is not a
route; the image has to arrive from a registry.
