# Measuring on rented GPUs: current safe contract

Paid measurement is temporarily **RunPod-only, SSH-only and exact-target-only**.
JarvisLabs, Vast, Lambda, provider-native containers, spot instances, recovery,
adoption, persistent volumes, preview/race and remote publication are refused
before provider mutation.

The canonical executable walkthrough is
[`THIRD-PARTY-QUICKSTART.md`](THIRD-PARTY-QUICKSTART.md). Keep commands there
rather than copying them into a second drifting recipe. This document explains
the operational boundary.

## Credentials and identity

- RunPod API bytes come from an absolute owner-only mode-0600 regular file.
  They never appear in argv, logs, receipts or bundles.
- Official target identity is still resolved anonymously from the literal
  `https://huggingface.co` endpoint. The high-bandwidth target download then
  uses the explicit read-scoped `--hf-download-token-file`: the controller
  transports it as `.secrets/hf_token` with mode 0600 inside a 0700 directory,
  never puts its bytes in argv or logs, and confirms erasure immediately after
  `fetch_target`. Panels remain anonymous.
- An ED25519 public key must exist locally before create.
- Before the first SSH byte, the controller waits at most 15 minutes for one
  exact ED25519 fingerprint line from the fresh pod's bounded authenticated
  RunPod v2 container-log stream. Each network read and the whole stream have
  independent deadlines. It compares that value to the untrusted network
  keyscan, then writes a fresh per-attempt `known_hosts` file and uses
  `StrictHostKeyChecking=yes`. No operator fingerprint prompt or first-hop TOFU
  is permitted.
- A separate owner/write token used for optional root publication stays on the
  controller. Local publication is possible only after qualification, verified
  retrieval, provider-confirmed absence and billing reconciliation.

## Lifecycle prerequisites

The paid route requires all of these before one create POST:

1. exact official HF repository and 40-hex revision identity;
2. exact target metadata, shard census, allowlist and scientific profile;
3. a committed fully clean checkout, including no untracked files;
4. an account-bound healthy user-systemd reaper;
5. a current accepted controller-loss/autonomous-reaper drill proof from the
   same checkout and control-plane closure;
6. a fresh complete RunPod pods-plus-network-volumes inventory;
7. a fresh RunPod balance and tariff-validity observation;
8. one locked campaign ledger admitting cumulative settled, unresolved and
   proposed exposure under the configured ceiling, reserve and cleanup margin;
9. exact runtime, retrieval/delete reserve and per-attempt cost caps;
10. explicit confirmation unless the operator deliberately passes `--yes`.

The checkout, manifests, provider account, reaper health, balance, inventory
and campaign generation are checked again immediately before mutation. A
failure is a refusal, never a fallback estimate.

## One-POST execution

The controller writes a durable lease in `PREPARED`, records campaign
`CREATING`, fsyncs `POST_INTENT`, then performs exactly one create POST with a
full job hash plus random 96-bit attempt id in the resource name. It never
retries an ambiguous response as a new science attempt. Any exact ids found
during response-loss reconciliation are bound only for cleanup.

The pod must converge as one exact secure on-demand resource with the quoted
GPU, image, storage and CPU/RAM minima. The create request carries
`terminateAfter`, but admission treats it only as an untrusted hint; the
account-bound reaper independently enforces the same absolute lease deadline.
Scientific stages then execute only from sealed `job.json`; ambient
configuration cannot change their paths, profile, threading, cache, panel or
target.

## Retrieval and deletion

Results are archived on the pod under a role-specific member with stored
DEFLATE and exact uncompressed/transfer-size contracts. Retrieval gets at most
three fresh-temporary-directory attempts. Deadline planning funds every
remaining attempt plus final deletion. Before extracting a large payload, the
controller authenticates the manifest and `job.json`, then applies the
job-specific archive bounds. Accepted output is SHA-256/size/member verified
and extracted without symlinks, hardlinks, traversal, duplicate paths or
special files.

Success or failure then requests deletion of every exact campaign-owned id.
The pod remains chargeable until full inventory proves exact absence; `EXITED`
is not absence. Billing reconciliation follows provider absence. The campaign
reservation releases only when both proof sets bind the exact same ids.
Retrieval exhaustion still deletes the pod; the durable lease, provider
deadline and independent reaper remain backstops.

## Campaign limits

`--max-cost` caps one attempt. It is not a campaign budget.
`--campaign-ceiling`, `--campaign-reserve` and
`--campaign-reaper-margin` govern the shared ledger atomically. Requested width
is admitted under the ledger lock. Width two additionally requires a verified
published root archive for the exact root identity; otherwise effective width
is one.

Tariff defaults are evidence with an expiry, not timeless constants. Once
`--tariff-valid-until` has passed, planning refuses until the operator supplies
current authored rates and validity.

## Scientific admission

The first production quant route admits only:

```text
malaiwah/GLM-5.3-Flash-TR3-6bpw
@ 9ab94105a71708a19c6d960d24b4aa6d459f5623
```

It binds the exact K6 public seal, materialization evidence, runtime profile,
official BF16 metadata and historical checkpoint-verdict bridge. The pinned K8
release is refused because no equivalent evidence binds its measured student
checkpoint identity to the sealed K8 surface; K6 evidence is not transferable.

The root route admits only exact authored BF16 pins with their matching
checked-in panel, unexpected-tensor allowlist, target identity, timing and
license contract. Each creates two distinct fresh-process `run_count=1`
hidden-state captures, independently verifies both, forces an exact
self-comparison using NumPy/CPU float32 replay with vocabulary chunks, and
emits root qualification. Remote work ends there. Publication is optional and
controller-local after teardown proof.

## Spend-free planning versus the paid drill

`measure-cloud ... --dry-run` creates no provider resource. It still performs
read-only account queries and refuses without current lifecycle/campaign
evidence.

`measure-cloud drill ... --dry-run` validates the drill plan without mutation.
The same command with `--yes` is deliberately paid: it creates one small pod,
kills its controller, and accepts proof only after the independent
user-systemd reaper issues the exact-id destroy at the absolute lease deadline,
fresh complete inventory proves absence, and billing stabilizes. The requested
provider `terminateAfter` value is recorded but explicitly untrusted. Run it
only with explicit spend authorization.

## Local image and other providers

The container image remains a tested local/developer transport on hardware the
operator already controls. It is not a shortcut around paid admission.
Provider adapters for JarvisLabs, Vast and Lambda remain historical/read-only
implementation evidence; the current `measure-cloud` command refuses them
before mutation.

## Emergency inspection

Use the same key, account-bound state and lease directories as installation:

```bash
bin/measure-cloud reaper --provider runpod --list
bin/measure-cloud reaper --provider runpod --sweep --dry-run
```

A real sweep is deliberate and destroys only exact ids authorized by leases
this tool wrote. Never delete, pause or adopt a resource you did not create.
