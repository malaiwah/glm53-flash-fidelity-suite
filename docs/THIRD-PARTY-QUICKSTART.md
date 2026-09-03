# Third-party quickstart — from a fresh clone to a submitted measurement

This is the unaided path for the paid route: one fresh secure on-demand
RunPod pod reached by authenticated SSH, rented for exactly one measurement
and destroyed afterwards. Everything here is $0.00 except the single step
marked **PAID**. No command here publishes externally unless you add
`--publish-root-to`.

What is always enforced, and what strict campaign mode adds on top, is in
[`CLOUD-RECIPES.md`](CLOUD-RECIPES.md). `bin/measure-cloud --help` is the
ground truth for every flag.

## 1. Prerequisites

- Stock Python 3.9 or newer; `bin/` needs no local install.
- A clean clone: tracked files unmodified. Untracked files are ignored.
- A RunPod API key in an owner-only mode-0600 file, default
  `~/.config/runpod/api_key` (`--runpod-key-file` otherwise). Never argv or
  environment.
- A Hugging Face token in an owner-only mode-0600 file; the recipes below
  pass it as `--hf-token-file ~/.hf_token` (default
  `~/.cache/huggingface/token`). It publishes from this machine and, unless
  you pass a separate read-only `--hf-download-token-file`, authenticates the
  target download on the pod.
- `~/.ssh/id_ed25519.pub`, accepted by RunPod at create.
- A `systemd --user` session that survives logout:
  `loginctl enable-linger $USER`.

```bash
chmod 600 ~/.config/runpod/api_key ~/.hf_token
test -f ~/.ssh/id_ed25519.pub
bin/fidelity-doctor
```

`bin/fidelity-doctor` is offline and prints no secret. Do not proceed on a
failure.

## 2. Install the autonomous reaper (once per machine and account)

```bash
bin/measure-cloud reaper --provider runpod --install
```

This changes only local user-systemd state and performs read-only RunPod
account queries; it creates no provider resource. Every paid run refuses
unless this timer is installed and healthy. A checkout newer than the
installed snapshot is a warning in the plan, not a refusal; re-run the
install to pick it up.

## 3. Dry-run the recipe — $0.00

The full GLM-5.3 root capture, with every derived value left to the
controller (GPU, storage, host minima, dataset repository and name, tensor
allowlist, download token; the dry-run plan prints each one):

```bash
bin/measure-cloud --provider runpod --role root \
    --model zai-org/GLM-5.3-BF16 --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
    --dataset-id fidelity--glm53.malaiwah.root.bf16 --publish-root-to malaiwah/glm53-fidelity-root-v1 \
    --hf-token-file ~/.hf_token --measurer malaiwah \
    --max-cost 40 --max-runtime 3h30m --out ~/fidelity-runs/glm53-root --dry-run
```

Replace `--dataset-id`, `--publish-root-to`, `--measurer`, `--hf-token-file`
and `--out` with your own identities. Because the hidden-form dataset
redistributes the checkpoint's native output-head weights, the dry-run
validates the exact pinned `LICENSE` bytes anonymously and records
`license: other`.

`--dry-run` runs every check — target identity, clean checkout, reaper
health, account inventory and balance, cost quote against `--max-cost` — and
prints the plan without creating anything. A failure is a refusal that names
its reason, exit code 3.

The same shape on the suite's 5B CI fixture. Its authored timing row in
`bin/engines.json` names an L4 and a 1-hour conservative bound, so the GPU is
derived and `--max-runtime 1h` is that bound:

```bash
bin/measure-cloud --provider runpod --role root \
    --model malaiwah/GLM-5.2-SIQ-Fruit-bf16 --revision ef68013aa6e16453cf52b5b77647f72fbe258c3c \
    --panel-dir engines/panels/panel--fruit.malaiwah.heldout-v1 \
    --dataset-id fidelity--fruit.<your-hf-handle>.root.bf16 \
    --measurer <your-hf-handle> \
    --max-cost 5 --max-runtime 1h --out ~/fidelity-runs/fruit-root --dry-run
```

Without `--publish-root-to` the sealed dataset stays under `--out`.

A quant measurement uses `--role quant` (the default) with `--panel` instead
of `--panel-dir` and `--dataset-id`. The one public quant the paid route
admits today is already in the registry, so its command exits `no_spend` at
the front gate before any account access:

```bash
bin/measure-cloud --provider runpod \
    --model malaiwah/GLM-5.3-Flash-TR3-6bpw --revision 9ab94105a71708a19c6d960d24b4aa6d459f5623 \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --measurer <your-hf-handle> \
    --max-cost 20 --max-runtime 12h --out ~/fidelity-runs/k6 --dry-run
```

## 4. The paid run — the one paid step

**PAID:** repeat the exact dry-run command without `--dry-run`. Leave off
`--yes` for the interactive confirmation:

```text
Create one secure on-demand RunPod (calculated maximum $…; hard cap $…;
--max-cost is the whole budget)? [y/N]
```

Only `y` or `yes` permits the single create POST. From there the controller
authenticates the pod's ED25519 fingerprint through RunPod's container-log
API, runs the capture, retrieves and verifies the archive, deletes the pod
and proves its absence. Retrieval failure still deletes the pod; the lease
deadline and the reaper remain the backstops. Exit codes: 0 ok; 1 the run
failed and the pod is proven gone; 3 refused before anything was created;
90 a pod may remain — run `bin/measure-cloud reaper --provider runpod --list`.

The verified root outputs are:

```text
<out>/result.tar.gz
<out>/result/receipts/root-qualification.json
```

For a quant the corresponding output is
`<out>/result/receipts/measurement-receipt.json`.

### Optional: strict campaign mode

If the account is dedicated to this suite, several attempts must share one
ceiling, or you want a sealed proof that the reaper destroyed a real pod
after the controller died, add the four `--campaign-*` flags and run the paid
`measure-cloud drill` first. The flags, the drill and the tradeoffs are in
[`CLOUD-RECIPES.md`](CLOUD-RECIPES.md#strict-campaign-mode-opt-in); the
`--help` epilog shows both commands.

## 5. Validate a quant receipt the way the registry will — offline, $0.00

```bash
bin/registry-submit <out>/result/receipts/measurement-receipt.json
```

Prints the row your receipt generates, its comparability key and class, and
the rows it may be ranked against — or exactly which check failed. Doing this
before submitting is, in the maintainer's own words, the difference between a
same-day merge and a round trip.

## 6. Submit it — one live destination

**Hugging Face discussion** on the registry dataset — this is the live path:

1. Open <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
2. New discussion titled `submission: <repo> on <panel>`
3. Paste the template from [CONTRIBUTING §2](../registry/CONTRIBUTING.md) with
   your receipt inside the fence (attach the file too if the editor allows).

The GitHub pull-request mirror described in CONTRIBUTING §3 is **not live** —
the URL 404s today; CONTRIBUTING says so and this document will keep saying so
until it exists. Do not wait for it.

## 7. What happens next — what is promised, and what is not

From [CONTRIBUTING §4](../registry/CONTRIBUTING.md), which is the commitment
this project actually makes:

* The maintainer saves your receipt under `receipts/<your-handle>/`, runs the
  same checks CI would, and **replies in your thread** with the generated row
  id, its comparability key, its class, and which rows it sits next to — or,
  if refused, exactly which check failed and what to change. "Either way you
  get a real answer, not silence."
* **No response-time SLA is promised anywhere**, and this document will not
  invent one. The stated fast path is a receipt that already passed
  `bin/registry-submit`: validated submissions are same-day-mergeable; broken
  seals and missing fields cost a round trip.
* Acceptance criteria are mechanical and public: the seal verifies, the
  40-hex revision, `run_count >= 2` with tensor-content determinism evidence,
  a scope census (for a GGUF the registry refuses the row without one), a
  `produced_by` block with `entrypoint_sha256` (HARN-001), your own handle in
  `measurer` — the complete bounce list is
  [CONTRIBUTING §5](../registry/CONTRIBUTING.md).
* Outside measurements enter as class `advisory` (that is about provenance,
  not trust), shown in the same tables when the comparability key matches.
  The flag worth chasing afterwards is `independently_verified: true` — a
  different party reproducing your number on the same panel and reference.

## If your target is not measurable

The refusal will name why for $0.00: no panel you can fetch (Qwen3.8-27B is
closed to outside measurement today — panels are private), no reader for the
surface on any lane (MLX / NVFP4 / AWQ / GPTQ), or no profile at that rate.
Those are the real boundaries of the system today; the
[support matrix](../README.md#before-you-rent-what-is-measurable-today) is
the authoritative statement of them, and
[CONTRIBUTING §6](../registry/CONTRIBUTING.md) is the path for proposing a
new panel or model.
