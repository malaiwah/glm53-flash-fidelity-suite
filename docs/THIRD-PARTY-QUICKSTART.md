# Third-party quickstart — from a fresh clone to a submitted measurement

This is the unaided path for the **current paid contract**, not the broader
engine-capability matrix. Paid execution is temporarily limited to exact
authored K6 quant and BF16 root pins over one fresh secure on-demand RunPod
reached by authenticated SSH. GGUF, K8, other revisions and other providers
refuse before mutation.

Everything is $0.00 until the controller-loss drill and the measurement steps
explicitly marked **PAID**. No command here publishes externally.

## 1. Local and account prerequisites

- Stock Python 3.9 or newer; `bin/` needs no local install.
- A committed, fully clean clone. The drill and measurement compare HEAD,
  index, worktree and untracked state again immediately before provider POST.
- A RunPod API key in an absolute owner-only regular file; never in argv or an
  environment value.
- A Hugging Face read token in a separate owner-only mode-0600 regular file.
  It authenticates the high-bandwidth target download on the pod.
- `~/.ssh/id_ed25519.pub`, accepted by RunPod at create.
- A `systemd --user` session on the controller machine.
- Explicit campaign limits chosen below the account's available balance.

```bash
export RUNPOD_KEY_FILE="$HOME/.config/runpod/api_key"
export HF_DOWNLOAD_TOKEN_FILE="$HOME/.config/huggingface/runpod_read_token"
export FIDELITY_STATE="$HOME/.fidelity-cloud"
export CAMPAIGN_LEDGER="$FIDELITY_STATE/campaign.json"
export CAMPAIGN_CEILING_USD="REPLACE"
export CAMPAIGN_RESERVE_USD="REPLACE"
export CAMPAIGN_REAPER_MARGIN_USD="REPLACE"
export DRILL_CAP_USD="REPLACE"
export ATTEMPT_CAP_USD="REPLACE"

chmod 600 "$RUNPOD_KEY_FILE"
chmod 600 "$HF_DOWNLOAD_TOKEN_FILE"
test -f "$HOME/.ssh/id_ed25519.pub"
```

Use a read-scoped token for `HF_DOWNLOAD_TOKEN_FILE`. The controller verifies
the exact target anonymously, transports this token separately as a 0600 file,
uses it only during `fetch_target`, and confirms its removal immediately after
that stage. It never enters argv, logs, the bundle, `job.json`, or a receipt.
The owner/write token used by optional root publication is a separate
`--hf-token-file` and remains on the controller.

`REPLACE` is deliberate: the suite must not invent the user's financial
limits. Each value is validated as a finite decimal. The campaign ceiling is a
cumulative cap; reserve and cleanup margin remain unavailable to new work.

## 2. Verify the checkout

```bash
bin/fidelity-doctor
bin/measure-local --probe-engines
bash bin/selftest_all.sh
```

The full battery is spend-free and GPU-free. Do not proceed on a failure or a
new skip in an applicable rung.

## 3. Install and verify the autonomous reaper

This changes only local user-systemd state and performs read-only RunPod account
queries; it creates no provider resource:

```bash
bin/measure-cloud reaper \
    --provider runpod --runpod-key-file "$RUNPOD_KEY_FILE" \
    --reaper-state-dir "$FIDELITY_STATE" \
    --lease-dir "$FIDELITY_STATE/leases-v2" \
    --install
```

The timer is account-bound. A missing/stale health stamp, changed account id,
wrong lease directory or failed initial sweep blocks both the drill and every
measurement.

## 4. Produce the controller-loss proof

First validate the drill plan. This makes read-only provider queries and no
paid call:

```bash
bin/measure-cloud drill \
    --provider runpod --runpod-key-file "$RUNPOD_KEY_FILE" \
    --reaper-state-dir "$FIDELITY_STATE" \
    --lease-dir "$FIDELITY_STATE/leases-v2" \
    --campaign-ledger "$CAMPAIGN_LEDGER" \
    --campaign-ceiling "$CAMPAIGN_CEILING_USD" \
    --campaign-reserve "$CAMPAIGN_RESERVE_USD" \
    --campaign-reaper-margin "$CAMPAIGN_REAPER_MARGIN_USD" \
    --campaign-width 1 --max-cost "$DRILL_CAP_USD" \
    --out "$FIDELITY_STATE/drill" --dry-run
```

**PAID:** repeat that command without `--dry-run` and add `--yes`. It performs
one small L4 create POST, deliberately kills its controller, and accepts a
proof only after the independent user-systemd reaper issues the exact-id
destroy at the absolute lease deadline, complete inventory proves absence and
billing stabilizes. RunPod `terminateAfter` is recorded as an untrusted hint,
not evidence of cleanup. Before the first SSH byte, the controller
automatically reads the exact ED25519 fingerprint from the fresh pod's bounded
authenticated RunPod v2 container-log stream and compares it to the untrusted
network keyscan. No operator fingerprint prompt or first-hop TOFU is used.

The accepted artifact is:

```text
$FIDELITY_STATE/drill/proof.json
```

The validator rejects a stale proof, a different checkout/control manifest, a
different RunPod account, missing artifacts, or an invalid lifecycle/billing
chain. Never copy another campaign's proof.

## 5. Exact supported K6 dry-run

```bash
bin/registry-view check malaiwah/GLM-5.3-Flash-TR3-6bpw

bin/measure-cloud \
    --provider runpod --on-demand --region secure --on-preempt fail \
    --model malaiwah/GLM-5.3-Flash-TR3-6bpw \
    --revision 9ab94105a71708a19c6d960d24b4aa6d459f5623 \
    --panel brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --lane streaming --schedule window-major --cold-runs 2 --gpu H200 \
    --runpod-key-file "$RUNPOD_KEY_FILE" \
    --hf-download-token-file "$HF_DOWNLOAD_TOKEN_FILE" \
    --reaper-state-dir "$FIDELITY_STATE" \
    --lease-dir "$FIDELITY_STATE/leases-v2" \
    --runpod-safety-proof "$FIDELITY_STATE/drill/proof.json" \
    --campaign-ledger "$CAMPAIGN_LEDGER" \
    --campaign-ceiling "$CAMPAIGN_CEILING_USD" \
    --campaign-reserve "$CAMPAIGN_RESERVE_USD" \
    --campaign-reaper-margin "$CAMPAIGN_REAPER_MARGIN_USD" \
    --campaign-width 1 \
    --max-cost "$ATTEMPT_CAP_USD" --max-runtime 12h \
    --measurer YOUR_HF_HANDLE \
    --out "$HOME/fidelity-runs/k6" --dry-run
```

Replace every `YOUR_*` and `REPLACE` sentinel with the intended identity or
limit before running any paid command below. The safe controller refuses
literal placeholders and has no default measurer; refusal happens before
RunPod account access.

`--dry-run` creates no RunPod resource. K6 currently exits `no_spend` at the
registry front gate, before account access. Any target that proceeds must use a
previously absent `--out` path and pass current official anonymous HF metadata,
clean source, proof/reaper state, fresh complete pods-plus-network-volumes
inventory, current balance and campaign admission. “Not checked” cannot
authorize a paid run.

### Exact Fruit root dry-run

The root route uses the checked-in Fruit panel and exact tensor allowlist. Set
new intended dataset identity values; publication remains optional and is not
part of this command:

```bash
export ROOT_DATASET_ID="REPLACE"
export ROOT_DATASET_REPOSITORY="YOUR_HANDLE/REPLACE"
export ROOT_DATASET_NAME="REPLACE"
export ROOT_ATTEMPT_CAP_USD="REPLACE"
export ROOT_MAX_RUNTIME="REPLACE"

bin/measure-cloud \
    --provider runpod --on-demand --region secure --on-preempt fail \
    --role root \
    --model malaiwah/GLM-5.2-SIQ-Fruit-bf16 \
    --revision ef68013aa6e16453cf52b5b77647f72fbe258c3c \
    --panel-dir engines/panels/panel--fruit.malaiwah.heldout-v1 \
    --dataset-id "$ROOT_DATASET_ID" \
    --dataset-repository "$ROOT_DATASET_REPOSITORY" \
    --dataset-name "$ROOT_DATASET_NAME" \
    --unexpected-tensor-allowlist \
        engines/tools/layer-outer-evidence/fruit-layer13-unexpected-keys.json \
    --lane streaming --form hidden --schedule layer-outer \
    --capture-device cuda --cold-runs 2 --gpu L4 \
    --replay-device numpy --replay-dtype float32 \
    --replay-vocab-chunk 8192 \
    --runpod-key-file "$RUNPOD_KEY_FILE" \
    --hf-download-token-file "$HF_DOWNLOAD_TOKEN_FILE" \
    --reaper-state-dir "$FIDELITY_STATE" \
    --lease-dir "$FIDELITY_STATE/leases-v2" \
    --runpod-safety-proof "$FIDELITY_STATE/drill/proof.json" \
    --campaign-ledger "$CAMPAIGN_LEDGER" \
    --campaign-ceiling "$CAMPAIGN_CEILING_USD" \
    --campaign-reserve "$CAMPAIGN_RESERVE_USD" \
    --campaign-reaper-margin "$CAMPAIGN_REAPER_MARGIN_USD" \
    --campaign-width 1 \
    --max-cost "$ROOT_ATTEMPT_CAP_USD" \
    --max-runtime "$ROOT_MAX_RUNTIME" \
    --measurer YOUR_HF_HANDLE \
    --out "$HOME/fidelity-runs/fruit-root" --dry-run
```

Two fresh processes each emit `run_count=1`; qualification binds both
independent verifications and the forced exact self-comparison. An unpublished
qualified archive remains valid evidence. `--publish-root-to`, when used, must
equal `--dataset-repository` and runs only on the controller after teardown.


### Exact full GLM-5.3 root dry-run

The full GLM route is separately bound to its pinned checkpoint identity,
25-window panel, layer-78 unexpected-tensor allowlist and H200 timing. Because
the hidden-form dataset redistributes the checkpoint's native output-head
weights, the controller copies the exact pinned model `LICENSE` bytes into the
dataset and records `license: other`; it never relabels those weights as MIT.

```bash
export ROOT_DATASET_ID="fidelity--glm53.malaiwah.root.bf16"
export ROOT_DATASET_REPOSITORY="malaiwah/glm53-fidelity-root-v1"
export ROOT_DATASET_NAME="GLM-5.3 BF16 root fidelity dataset (hidden form)"
export ROOT_ATTEMPT_CAP_USD="40"
export ROOT_MAX_RUNTIME="6h"

bin/measure-cloud \
    --provider runpod --on-demand --region secure --on-preempt fail \
    --role root \
    --model zai-org/GLM-5.3-BF16 \
    --revision 304b8051cfb2b260b61ce0cbe330e02a98e73639 \
    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \
    --dataset-id "$ROOT_DATASET_ID" \
    --dataset-repository "$ROOT_DATASET_REPOSITORY" \
    --dataset-name "$ROOT_DATASET_NAME" \
    --unexpected-tensor-allowlist \
        engines/tools/layer-outer-evidence/glm53-layer78-unexpected-keys.json \
    --lane streaming --form hidden --schedule layer-outer \
    --capture-device cuda --cold-runs 2 --gpu H200 \
    --replay-device numpy --replay-dtype float32 \
    --replay-vocab-chunk 8192 \
    --runpod-key-file "$RUNPOD_KEY_FILE" \
    --hf-download-token-file "$HF_DOWNLOAD_TOKEN_FILE" \
    --reaper-state-dir "$FIDELITY_STATE" \
    --lease-dir "$FIDELITY_STATE/leases-v2" \
    --runpod-safety-proof "$FIDELITY_STATE/drill/proof.json" \
    --campaign-ledger "$CAMPAIGN_LEDGER" \
    --campaign-ceiling "$CAMPAIGN_CEILING_USD" \
    --campaign-reserve "$CAMPAIGN_RESERVE_USD" \
    --campaign-reaper-margin "$CAMPAIGN_REAPER_MARGIN_USD" \
    --campaign-width 1 \
    --max-cost "$ROOT_ATTEMPT_CAP_USD" \
    --max-runtime "$ROOT_MAX_RUNTIME" \
    --measurer malaiwah \
    --out "$HOME/fidelity-runs/glm53-root" --dry-run
```

The dry-run validates the exact source-license bytes anonymously before spend.
For controller-local publication after qualification and teardown, add
`--publish-root-to "$ROOT_DATASET_REPOSITORY"` and an owner-only
`--hf-token-file`; the target repository must be absent.

## 6. The paid execution boundary

The published registry already contains K6, so the exact K6 command currently
returns `no_spend` before provider access. Do not route around that result:
safe RunPod refuses `--force`.

An unmeasured exact authored root can reach paid admission after every local,
account and scientific prerequisite passes. **PAID:** repeat its exact command
without `--dry-run`. Leave off `--yes` for the interactive confirmation:

```text
Create one secure on-demand RunPod (calculated maximum $…; hard cap $…;
campaign ceiling $… with reserve $…)? [y/N]
```

Only `y` or `yes` permits the single create POST. The controller never retries
an ambiguous create as science work. After creation, it authenticates the
pod's ED25519 fingerprint through the bounded RunPod v2 container-log API and
compares it to the untrusted network keyscan. Success means bounded archive
retrieval to a fresh local temporary directory, exact SHA-256/size/member
verification, pod deletion, exact absence, billing reconciliation and campaign
release. Retrieval failure still deletes the pod; the durable lease and reaper
remain backstops.

The verified root outputs are:

```text
<out>/result.tar.gz
<out>/result/receipts/root-qualification.json
```

For a future exact quant explicitly added to the authored admission set, the
corresponding output is
`<out>/result/receipts/measurement-receipt.json`. Engine capability alone does
not add a paid target.

## 7. Validate a quant receipt the way the registry will — offline, $0.00

```bash
bin/registry-submit <out>/result/receipts/measurement-receipt.json
```

Prints the row your receipt generates, its comparability key and class, and
the rows it may be ranked against — or exactly which check failed. Doing this
before submitting is, in the maintainer's own words, the difference between a
same-day merge and a round trip.

## 8. Submit it — one live destination

**Hugging Face discussion** on the registry dataset — this is the live path:

1. Open <https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/discussions>
2. New discussion titled `submission: <repo> on <panel>`
3. Paste the template from [CONTRIBUTING §2](../registry/CONTRIBUTING.md) with
   your receipt inside the fence (attach the file too if the editor allows).

The GitHub pull-request mirror described in CONTRIBUTING §3 is **not live** —
the URL 404s today; CONTRIBUTING says so and this document will keep saying so
until it exists. Do not wait for it.

## 9. What happens next — what is promised, and what is not

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
