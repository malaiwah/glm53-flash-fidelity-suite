# Cloud providers: paid execution boundary

The current paid controller admits exactly one backend: **RunPod secure
on-demand pods over SSH**. `bin/measure-cloud` refuses JarvisLabs, Vast, Lambda,
provider-native containers, spot instances, recovery/adoption, volumes and
pause/hold modes before provider mutation.

This is narrower than the adapters present in `bin/fidelity/providers/`.
Adapter existence means code can describe or test a control plane; it does not
authorize spending or prove the four guarantees a paid run relies on.
The executable procedure is
[`THIRD-PARTY-QUICKSTART.md`](THIRD-PARTY-QUICKSTART.md); the safety explanation
is [`CLOUD-RECIPES.md`](CLOUD-RECIPES.md).

## Required provider properties

A paid backend cannot be admitted unless live evidence proves all of these:

| property | safety reason |
|---|---|
| complete account-wide inventory of every chargeable resource class | absence cannot be inferred from one pod lookup or a local lease |
| unique operator-authored resource names | response-loss reconciliation and independent cleanup need exact ownership |
| idempotent deletion plus exact-absence confirmation | `EXITED`, a stopped process or a successful DELETE response may still bill |
| provider-enforced termination deadline | cleanup must survive controller and reaper loss |
| current balance and billing history | campaign exposure and settlement cannot be inferred from estimates |
| stable offer identity, hardware, price and region | the pre-create quote must bind the resource actually created |
| authenticated SSH host identity | API-supplied IP/port plus network keyscan alone permits machine-in-the-middle execution |

File transfer and command execution do not have to be provider APIs. The current
route deliberately uses audited SSH transport after out-of-band host-key
authentication.

## RunPod closure

The RunPod backend uses official HTTPS control-plane endpoints: REST/v2 for
read-only inventory, balance, offers and billing, and GraphQL for one pod
create plus deletion. A durable lease and
campaign record are written before the create request. Each attempt has one
full job hash plus random 96-bit id; no ambiguous response is retried as a
fresh science attempt.

The account inventory includes pods and network volumes. The controller binds
only exact ids carrying its unique attempt identity, but never adopts them for
science after an ambiguous create response. Such resources are cleanup-only.
Deletion is complete only when a fresh full inventory proves every exact id
absent and billing reconciliation binds those same ids.

The independent user-systemd reaper uses the same owner-only API-key file,
account-bound state directory and v2 lease directory as the controller. It
enforces the lease's absolute reap deadline itself. RunPod `terminateAfter` is
still sent at the same timestamp, but is an untrusted provider hint: the real
control plane has been observed leaving a pod live after that value.

## Portability does not imply comparability

Full-vocabulary KLD in fp64 is provider-independent arithmetic. Measurement
identity is not. GPU model, engine profile, artifact surface, panel, reference,
schedule and code closure remain bound in the receipt and comparability key.
Moving the same target to another hardware or engine profile does not make its
number comparable by assertion.

The first admitted RunPod quant is the exact authored K6 target/profile. Root
captures are separately authored per exact checkpoint, panel, allowlist,
hardware and license identity. Evidence for one target cannot authorize K8,
another quant, another root revision or a filename-near checkpoint.

## Admitting another backend

Adding an adapter is insufficient. A new paid backend needs:

1. full inventory and exact-absence semantics for every chargeable resource;
2. one-create response-loss and ownership tests;
3. an independent account-wide reaper that enforces the absolute lease deadline;
4. fresh price, balance, campaign-ledger and billing reconciliation;
5. authenticated host identity and credential-file transport;
6. controller-loss and autonomous-reaper drills on the real control plane;
7. a bitwise/content-digest comparison against an already sealed target under
   the exact same scientific profile.

Until that closure is implemented, tested and drilled, the command must refuse
the provider before any create request.
