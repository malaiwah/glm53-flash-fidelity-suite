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

**`capture` and `measure` are pod-contract verbs, not a human's command line.**
Both require a planner-written `--target-descriptor` (the exact shard census
the controller computes: config/index digests, shard bytes, download manifest —
`bin/selftest_container.py` and `bin/selftest_root_publish.py` show its
shape; no user-facing tool writes one), the three resource minima
(`--workspace-available-bytes-minimum`, `--container-available-bytes-minimum`,
`--expected-vram-bytes`) and a `--gpu` string with an **authored timing row**
in `bin/engines.json`: today `NVIDIA L4` for the Fruit root and `NVIDIA H200`
for GLM-5.3-Flash-BF16 and GLM-5.3-BF16 (`root_timing_profiles`); `measure`
additionally admits only the `tr3-6bpw` / `native-bf16` profiles. Any other
card refuses with `root_timing_evidence_absent` before anything is fetched —
the row exists to cap paid spend, and the entrypoint does not know it is not
on a meter. There is no `--surface`/`--bits` on `measure` (the target
descriptor carries them). Run as written, with `--dry-run`:

```
$ python3 bin/container_entry.py capture --dry-run --model zai-org/GLM-5.3-BF16 --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
      --panel-dir /panel --dataset-id fidelity--x.malaiwah.root.bf16 --lane streaming --gpu "NVIDIA A10"
fidelity capture: error: the following arguments are required: --target-descriptor, --workspace-available-bytes-minimum, --container-available-bytes-minimum, --expected-vram-bytes
$ python3 bin/container_entry.py measure --dry-run --model x/y --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
      --profile k6 --panel-descriptor /panel.json --lane streaming
fidelity measure: error: the following arguments are required: --target-descriptor, --gpu, --workspace-available-bytes-minimum, --container-available-bytes-minimum, --expected-vram-bytes
```

What a human at a workstation can drive:

```bash
# what is this image, and what can it see? (needs a writable /workspace)
docker run --gpus all --rm -v /data/run:/workspace <image> doctor
docker run --rm <image> version

# one stage, against a job document a controller already wrote
docker run --gpus all --rm -v /data/run:/workspace <image> stage --job /workspace/fidelity/job.json capture

# the zero-venv LOCAL route: the image's pinned interpreter running the dataset
# tools it ships (fidelity_dataset.py, hf_capture.py, the committed panels).
# Same argv as README Recipe 2 -> Local GPU quickstart; the checkpoint and the
# output live on your mount. Nothing here is a pod contract.
docker run --gpus all --rm -v /nvme:/nvme -e HF_HOME=/nvme/hf \
    --entrypoint /opt/fidelity/venv/bin/python <image> \
    /opt/fidelity/suite/bin/fidelity_dataset.py capture --engine hf-transformers \
        --out /nvme/ds/root-1 --form hidden --role root --lane streaming -- \
        --model /nvme/models/m --model-revision <40-hex> --weights-repository <owner>/<repo> --repository <handle>/<dataset-repo> \
        --panel /opt/fidelity/suite/engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
        --panel-id panel--glm53.malaiwah.corpus5x5-v1 --schedule layer-outer --device cuda --dtype bfloat16 \
        --dataset-id fidelity--<family>.<handle>.root.bf16 --dataset-name "<name>" \
        --run-name root-cold-1 --cold-run root-cold-1 --author <handle> --role root --sanity-expect Paris
```

Size it first with `bin/measure-local --artifact <repo> --panel <dataset>
--estimate-only` on the host: GLM-5.3-class captures measured 37.53 GB
allocated / 57.08 GB reserved (bf16, FP8) and 56.86 GB (K4 trellis) on H200 and
are refused below 64 GB; the container adds nothing to that arithmetic. The
container runs as root, so `chown` the mount afterwards (below).

Useful everywhere: `--dry-run` prints the job document and the stage list and
creates nothing; `--job FILE` uses a planner-written document verbatim;
`--only` / `--stop-after` narrow the sequence; `--image-pin <digest>` tells the
run which image the launcher pulled.

## Getting the answer back

A container has no controller holding an SSH connection open, so nothing pulls
`receipts.tar.gz` back the way `measure_cloud.py` does. For a while that was
only half-solved: `--publish-root-to` gave a multi-GB **root capture** a way
home, and the verb this project exists to serve had none. `measure` seals
`receipts/measurement-receipt.json` — 4–40 KB, and *the* object the registry
ingests — and a container-native run ended by naming the path it was written
to, inside a pod-scoped volume on a provider whose REST API
(`/v1/pods`, `/v1/pods/{id}`, `/billing/pods`) serves no logs and no files, in
an image that runs no sshd. The same was true of `stage`, and of a **failed**
run, whose receipts and logs are the evidence you most want and can least often
reach.

So the product is not "a dataset". It is *whatever this run sealed*, and the
caller picks the channel, because only the caller knows what they can read.

```bash
# stdout: always on, needs no flag. The only channel every platform has.
docker run --gpus all --rm -v /data/run:/workspace <image> measure ...

# a second copy onto a mount you control
... <image> measure ... --result-sink file:/workspace/out

# PUT the bundle somewhere you can read: presigned S3/R2/GCS, a collector,
# or an ntfy topic (which you can poll back with ?poll=1)
... <image> measure ... --result-sink https://ntfy.sh/<your-topic>
```

| sink | carries | use it when |
|---|---|---|
| `stdout` (always) | the summary + the receipt inline under 256 KB | any provider whose console or `docker logs` you can read |
| `file:PATH` | receipts + `job.json` + summary | a bind mount or a VM you own (Lambda, your own box) |
| `https://URL` | the same, as `tar.gz`, by **PUT** | automation; the pod cannot read anything back |
| `--publish-root-to` | the sealed **dataset** | a root/preview capture — multi-GB does not belong in a log frame |

Four properties worth stating, because each one is a defect that happened:

* **stdout is unconditional and delivered first.** If a later sink is
  misconfigured or its collector is down, the answer has already been printed.
  A failing sink is reported and **never changes the run's exit code** — the
  measurement either happened or it did not, and a collector being down is not
  a measurement result.
* **Delivery is in the `finally`**, so a run that fails at stage three still
  reports what it has. On RunPod the run root dies with the pod.
* **The HF token is shredded *before* any sink runs**, and `.secrets/` and the
  multi-GB `.stream-work/` scratch tree are excluded from every bundle.
* **A sink URL is often itself the credential** (a presigned PUT), so it is
  registered for redaction, `FIDELITY_RESULT_SINK` is the preferred channel —
  providers echo `dockerArgs` back in their consoles and API listings, and
  environment variables they do not — and a failure names the host and path
  but never the query string.

The frame is greppable on purpose:

```
===== FIDELITY-RESULT BEGIN =====
{ "schema": "malaiwah.fidelity-result-summary.v1", "verb": "measure",
  "status": "ok", "files": [ {"path": "...", "sha256": "..."} ] }
----- measurement-receipt.json -----
{ ... }
===== FIDELITY-RESULT END =====
```

Over the 256 KB cap the receipt is **withheld rather than dumped** — the frame
still carries its sha256, so the artifact stays identifiable and the summary
does not get pushed out of a provider's log buffer by bytes nobody can use in
that form.

---

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

## The container reproduced a published root, bitwise

On 2026-08-31 the image captured `malaiwah/GLM-5.2-SIQ-Fruit-bf16` on a RunPod
L4 — container-native, no SSH, no bundle upload, the panel read from inside the
image — and published it to
[`malaiwah/fruit-fidelity-root-container-v1`](https://huggingface.co/datasets/malaiwah/fruit-fidelity-root-container-v1).

```
capture_content_digest  b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c   container
                        b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c   published root
```

Identical, and so is every field of the stack fingerprint — torch 2.11.0+cu130,
transformers 5.16.1, CUDA 13.0, `NVIDIA L4`, `transformers-eager`, default
matmul precision. The L4 was chosen deliberately: it is the card that produced
the published root, and [`ARCHITECTURE-DETERMINISM.md`](ARCHITECTURE-DETERMINISM.md)
established that the GPU MODEL, not the provider or the host, is what moves
these bits. So this is a same-architecture reproduction through an entirely
different transport, and it is the first row in this project whose
`container.image_digest` is not null:
`sha256:65425cfd9d31fb8f0e8d58d1548ad6b46704aabebfcc60d42b5e59d1f5f6f5f0`.

`total_size_bytes` differs by 112 (67080528 vs 67080416): safetensors header
padding, not content. The content digest covers the tensors, which is the point
of having one.

**The new capture is the more honest artifact.** It carries two disclosures the
published root does not: a `checkpoint_tensors_not_loaded` caveat and a
**blocking** `unexpected_tensors_overridden`, both about the 791 tensors of
Fruit's layer-13 MTP draft head, which `transformers` does not build. The
published root predates that guard and says nothing about them. The numbers
agree exactly; only the disclosure does not.

### It took four rentals, and every refusal was correct

Total spend **$0.30**. Not one attempt produced a wrong number; each stopped:

| # | stopped at | cause | now caught by |
|---|---|---|---|
| 1 | `--panel-dir has no panel.json` | the committed panels reached `engines/panels/` two commits before `bin/BUNDLE.txt`, so `container_prune` correctly stripped them | `selftest_container` **C5l**, statically |
| 2 | 791 homeless tensors | Fruit's MTP draft head; indistinguishable from a quantizer that silently did not engage | `--allow-unexpected-tensors`, blocking disclosure |
| 3 | generation sanity probe | Fruit answered `' the'`, not `Paris` — it is a proxy, not an assistant | `--sanity-expect ''`, which needs **argv as a list** |
| 4 | driver 12040 vs a cu130 image | the host's CUDA was 12.4 | `require_accelerator()` **C12** + `allowedCudaVersions` |

Attempt 3 is why `create(docker_cmd=[...])` exists: the documented remedy is an
EMPTY argument, and no flat `dockerArgs` string can carry one.

Attempt 4 is the one worth remembering. The bootstrap saw the dead accelerator
and reported `PASS 1b accelerator decode parity: SKIPPED (no CUDA and no MPS on
this host; the check RUNS on the instance, which is where it counts)` — on the
instance. `setup` passed, 10 GB was fetched, and the capture died on the first
`.to(cuda)`. Five cents here; a 117 GB fetch on a GLM-5.3 root.

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

## The acceptance test, and its result

**Bit-identical output.** A containerised capture must produce the same
`capture_content_digest` as the current path on the same GPU. If it does not,
the image changed the arithmetic — that is a failure, not a variance.

**Run, 2026-08-31, Lambda `gpu_1x_a10` (NVIDIA A10, one box, one GPU):**

| | arm A — host bootstrap | arm B — the image |
|---|---|---|
| `capture_content_digest` | `b42ffe8f1d1dfcfdd784…a960549` | **identical** |
| per-window tensor digests | `552af179…`, `abd137ac…` | **identical** |
| `head.tensor_content_sha256` | `58b4b967…` | **identical** |
| `stack_fingerprint_sha256` | `18735425…` | **identical** |
| `lane_identity_sha256` | `2d0992fc…` | **identical** |
| `dataset_sha256` | `4d8eaae9…` | `587cbd6f…` — **differs, by design** |
| `runtime.container.image_digest` | `null` | `sha256:372d542c…` |
| `setup` stage | 134 s cold | 53–70 s (bootstrap all-no-op) |

Both arms captured `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B` @
`7c3a6d3d` over a 2×256 panel built on the box from this repository's own docs
(510 scored positions), driven by the same `bin/container_entry.py` so that the
only variable between them was the environment. Evidence, including both
manifests and both runtime receipts:
[`reports/container-proof/`](../reports/container-proof/).

`dataset_sha256` differs for exactly one reason, and it is the right one: arm B's
runtime receipt records the image it ran in and arm A's does not. Every other
field above is equal, including `stack_fingerprint_sha256` — which is what
`dscompare` reads to decide `stack_relation`, so the two captures are
same-stack and one can serve as the other's floor.

One detail worth keeping: the two arms did **not** have identical interpreters.
The host bootstrap installed python **3.12.13** from deadsnakes; the image has
Ubuntu 24.04's **3.12.3**. The tensors are still bit-identical, and the
fingerprint does not include the patch version — so this is evidence about how
much of the stack the digest actually pins, not a claim that patch versions
never matter.

`fidelity-dataset compare --self-compare` was **refused** on this pair, and
correctly: `PANEL-D6` compares the two captures' tokenizer identity, which is
recorded as the local path of the model tree (`/home/ubuntu/…/models/target`
vs `/workspace/…/models/target`). On the SSH path both arms always share a
root, so this never surfaced; a container has a different mount root by
construction. Written up in
[`REVIEW-DEFERRED.md`](REVIEW-DEFERRED.md) rather than fixed here, because it
changes a published manifest field.

Determinism is a **per-device** property (`docs/ARCHITECTURE-DETERMINISM.md`:
two A100s in two clouds agree bitwise; an H200 is 2.973e-04 nats away, 13× the
gap this registry publishes between two 4-bit quantizers). The comparison above
is only meaningful on **one** GPU, which is why both arms ran on one box.

### One operational wart

The container runs as root, so everything it writes into the mount is
root-owned and a later `rm -rf` from the login user fails file by file. Either
run it with `--user "$(id -u):$(id -g)"` or clean up with `sudo`. It is not a
correctness problem and it is exactly the kind of thing that only shows up the
second time you use the mount.

---

## Historical provider-container evidence

Direct provider-native container launch is **not an admitted measurement path**.
The current paid controller accepts only a fresh RunPod secure on-demand pod
over its authenticated SSH lifecycle, with a durable lease, installed reaper,
controller-loss proof, bounded result retrieval, exact absence, and billing
reconciliation. It deliberately exposes no `--image` path and refuses direct
`RunPod.create(...)`, Vast, Lambda, and JarvisLabs execution before mutation.

The container experiments below remain implementation evidence for the image;
they are not a runnable rental recipe. Use
[`THIRD-PARTY-QUICKSTART.md`](THIRD-PARTY-QUICKSTART.md) for the only current
paid boundary.

Three things that cost real money to learn:

* **The panel can come from the image.** `engines/panels/` ships two committed
  panels, so a container-native capture fetches no panel at all — but only if
  they are in `bin/BUNDLE.txt`. They landed in `engines/panels/` two commits
  before they landed in that list, and the image built in between refused its
  own committed panel on a rented L4: *"--panel-dir ... has no panel.json"*.
  `--require-all` proves every listed entry arrived; it cannot prove that what
  a capture needs was listed. `selftest_container.py` rung **C5l** now does.
* **`docker_args` is one flat string** *as this backend sends it*, because
  `podFindAndDeployOnDemand` is GraphQL and takes `dockerArgs` as a single
  string; an argument containing a space then depends on how the provider
  splits it. Prefer flags without spaces for now — `--gpu` is optional anyway,
  since the receipt's device name is read from torch by
  `fidelity/stackprint.py`, not from that flag. The real fix is the REST API:
  `POST /v1/pods` takes `dockerEntrypoint` and `dockerStartCmd` as **arrays**,
  so argv is a list and the quoting question disappears. Verified live.
* **A failed run's log is on a volume you cannot reach.** RunPod's REST API
  serves no logs and no files, and this image runs no sshd, so when a capture
  died at its `capture` stage the log had to be recovered by rewriting the
  running pod's entrypoint to `tar /workspace | curl` it out. That is why
  `--result-sink` now carries `logs/` (tail-capped) and why stdout prints the
  failing stage's log inline. Two better answers exist and are not yet wired:
  a `networkVolumeId` that outlives the pod, and the array entrypoint above.
* **Configuration and secrets travel in `env`, never in `docker_args`** — the
  args string is argv and RunPod returns it verbatim from
  `query { pod { dockerArgs } }`.

**Nested containers do not work on RunPod.** Probed on a live pod
(`NVIDIA RTX A4000`, community cloud, driver 580.126.20): running as uid 0 but
with `CapEff 0x00000000a80425fb` — the Docker default set, no `CAP_SYS_ADMIN` —
`unshare -U -r true` fails and `/dev/fuse` does not exist. So rootful podman
(needs `CAP_SYS_ADMIN` for mount namespaces) and rootless podman (needs
`CLONE_NEWUSER`, which the default seccomp profile blocks without
`CAP_SYS_ADMIN`) are both out. Building the image *on* a RunPod pod is not a
route; the image has to arrive from a registry.

### Vast.ai (2026-09-05, transport rehearsal)

`bin/fidelity/vastapi.py` `create(docker_cmd=[...])` launches the image on Vast
via `PUT /api/v0/asks/{id}/` with `runtype: "ssh"` and the full command in
`onstart` (prep + `exec python3.12 .../container_entry.py capture ...`).
`env` is a **string in Docker flag format** (`-e KEY=VAL`), not a dict — the
REST API confirms this in its OpenAPI schema. `onstart` is limited to 4048
chars; gzip+base64 (Vast's own documented workaround) handles longer scripts.

| observed | detail |
|---|---|
| Tesla T4, Nevada, cuda_max 13.0, driver 580.126.09 | image loaded, accelerator detected, panel staged, job.json written |
| `--result-sink https://ntfy.sh/<topic>` | tar.gz delivered, HTTP 200, retrieved via `?poll=1` |
| **blocked at `setup`** | the Nevada host's network has a broken SSL proxy to huggingface.co (cert hostname mismatch); `stage_measure.sh` uses `urllib` with strict SSL and fails before any model fetch |
| spend | ~$0.08 across 6 attempts; balance $19.56 → $19.48; all instances destroyed |

The transport pipeline works end-to-end (launch → image run → result frame →
ntfy delivery → retrieval). The blocking issue is host-specific SSL, not the
image or the transport. A host with working SSL to huggingface.co (or an
`HF_ENDPOINT` mirror) would complete the capture.

---

## Two architectures, and why arm64 is not decoration

The image targets `linux/amd64` **and** `linux/arm64`. The reason is a
measurement, not a preference: benchmarking eleven cards across four providers
found Lambda's `gpu_1x_gh200` — Grace, so **aarch64** — the cheapest per
measurement of anything measured anywhere, 0.098 ms/matrix against an A100
PCIe's 0.891, because this lane is host-bandwidth-bound and NVLink-C2C is not
PCIe ([`CLOUD-COMPARISON.md`](CLOUD-COMPARISON.md)). An arm64 image is what
makes that turnkey instead of a hand-built ARM stack.

**The pins hold on both.** Checked against the real indexes rather than
assumed:

| | aarch64 |
|---|---|
| `torch==2.11.0+cu130` | `manylinux_2_28_aarch64` published alongside `_x86_64` — the *same version string* |
| `kbnf`, `hf_transfer`, `tokenizers`, `safetensors` | aarch64 wheels published |
| `pydantic==2.5.3`, `formatron==0.5.0` | pure python |

Nothing in the wheel set falls back to a source build, so the arm64 image pins
what the amd64 one pins. **The one x86-only artefact in the recipe** is the
flash-attn wheel URL, and `bootstrap_measure.sh` fetches it only inside the
exllamav3 branch — which is not taken, because the measurement path imports the
pipeline without it. An arm64 run that ever *does* need exllamav3 will have to
build it. That is stated here rather than papered over.

**Both images have been built.** On one Lambda A10 box, from suite revision
`b76fb79`:

| | amd64 (native) | arm64 (QEMU on the same x86 box) |
|---|---|---|
| bootstrap layer | 73 s | **933 s** (12.8x) |
| whole build | ~1.5 min | ~19 min |
| image size | 6.45 GB | 6.12 GB |
| `torch` | `2.11.0+cu130` | `2.11.0+cu130` — same string |
| `transformers` | `5.16.1` | `5.16.1` |
| pipeline commit | `ce1bf970` | `ce1bf970` |
| `pins.arch` | `x86_64` | `aarch64` |
| `image_content_sha256` | `8045406b…` | `ec55d704…` |

`docker run --platform linux/arm64 <image> doctor` under emulation reports
`torch 2.11.0+cu130 cuda 13.0 | transformers 5.16.1` and
`cuda_available False` — correct, since QEMU has no GPU. What that run proves
is that the arm64 stack **installs and imports**; whether a GH200 produces
usable numbers with it is a question for a GH200, and the qualification of that
card is a separate piece of work.

12.8x on the bootstrap layer is the whole case for native arm64 runners
(`runs-on: ubuntu-24.04-arm`) once they are worth the switch. It is not a
blocker: nineteen minutes is a CI build, and the two architectures are separate
matrix jobs so the amd64 leg does not wait for it.

**The architecture is a pin, not a label.** `container_manifest.py` records
`arch` and `platform` among the pins, so the two images behind one multi-arch
tag carry **different** `image_content_sha256` values. That is correct: they
are different stacks that happen to share a tag. This repository has already
measured that the GPU *model* alone moves a number by 13× the gap it publishes
between two 4-bit quantizers; a receipt that cannot say which architecture
produced it is missing a fact of the same class.

---

## Releasing it

[`.github/workflows/container-image.yml`](../.github/workflows/container-image.yml)
builds both architectures and, once enabled, publishes to GHCR.

**Nothing is published until a maintainer says so.** The registry push is gated
on the repository variable `PUBLISH_CONTAINER`; unset, the workflow builds both
architectures, prints the plan and the digests into the run summary, and pushes
nothing. Landing the file is not a decision to publish — enabling it is one
switch (Settings → Secrets and variables → Actions → Variables →
`PUBLISH_CONTAINER=true`).

**The rules live in a script, not in `${{ }}`.** `bin/release_plan.py` decides
the tags, the platforms and the publish gate; the workflow calls it and builds
what it said. A workflow expression is untestable anywhere except GitHub — you
find out it was wrong by pushing a tag and reading a red run — so
`bin/selftest_container.py` rung C11 drives that script with known inputs and
known answers, offline:

```bash
python3 bin/release_plan.py --event release --ref refs/tags/v1.2.3 \
    --sha "$(git rev-parse HEAD)" --publish true
```

| ref | tags |
|---|---|
| `refs/tags/v1.2.3` | `sha-<12>`, `1.2.3`, `1.2`, `1`, `latest` |
| `refs/tags/v1.2.3-rc1` | `sha-<12>`, `1.2.3` — a prerelease never moves the series tags or `latest` |
| `refs/heads/main` | `sha-<12>`, `main` |
| `workflow_dispatch` | `sha-<12>`, `dev` |

The `sha-<12>` tag is always first and is what the image records as its own
`IMAGE_REFERENCE`, because a receipt needs a reference that still means these
bytes tomorrow and `latest` never does.

**QEMU now, native runners later.** The arm64 leg cross-builds under QEMU. That
is emulated I/O and unpacking rather than emulated compilation — every pinned
wheel publishes an aarch64 build — so it is slow but not pathological, and each
architecture is a separate matrix job so the fast one does not wait for the slow
one. If it becomes the bottleneck the upgrade is `runs-on: ubuntu-24.04-arm`,
which needs no other change.

**The digest is the point.** `produced_by.container_digest` has been `null` on
every row this repository has published. The `manifest` job prints the digest of
the multi-arch tag it just created, together with the exact command that cites
it:

```
docker run --gpus all -v /data:/workspace \
  ghcr.io/malaiwah/quant-fidelity-measure:sha-<12> capture ... --image-pin sha256:<digest>
```

That value lands in the capture receipt as `runtime.container.image_digest` and
in `produced_by.container_digest`.

---

## The changelog is generated

```bash
bin/changelog.py                 # since the last tag
bin/changelog.py --all --out CHANGELOG.md
bin/changelog.py --check         # what CI (and rung C11q) runs
```

Not git-cliff, not release-drafter — both assume Conventional Commits: a short
subject, a machine-readable type, and a body nobody reads. This repository's
commits are the opposite, long-form prose explaining what failed and why, and
by a convention that has held across the whole history the **first line is
already a changelog entry**. So the generator takes that line, splits the topic
off at the first colon, and groups by topic; adding a tool would add a
dependency, a config file and a second convention to produce the same list.

The topic rule is deliberately strict about the leading lowercase letter, since
a subject can also open with a file (`AGENTS.md:`), an identifier (`REFC-006:`)
or a flag (`--pipeline-root:`) — grouping by those gives one section per commit,
which is a list with extra headings rather than a changelog.
