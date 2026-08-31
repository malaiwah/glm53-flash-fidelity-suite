# Response to the independent peer review of 2026-08-31

**Review:** independent source/data/statistical-method/reproducibility/security/operations
review of this repository at commit `4522a99b`, 958 lines, received 2026-08-31.
**Response date:** 2026-08-31. **Posture:** the review was treated as a colleague
who is right until proven wrong. Every finding below was re-verified against
this repository's own receipts before any code moved; where we could recompute
the reviewer's numbers, we did, independently, and they reproduced.

The short version: **the review is substantially correct.** Its four scientific-
governance findings (P1-01, P1-02, P1-05, P1-15/16) go to the heart of this
project's promise and all four are CONFIRMED. Its statistical recomputations
reproduce exactly. One dated observation (no visible Actions runs) was already
amended by the reviewer's own chronology addendum and needs no rebuttal from us.

## How the science findings were verified

- **P1-15:** recomputed from the committed per-window series
  (`registry/protocol/per-window/`) and the window-selection receipt before
  changing anything: 4 source documents (7/6/6/6 windows), clean17 = 3
  (7/5/5); all document means positive; exact two-sided sign test p = 0.125
  (full) / 0.25 (clean17); equal-document t interval [+0.0000487, +0.0027102]
  full, [−0.000256, +0.002078] clean17. **The reviewer's numbers, to the
  digit.** Same-lane recompute (K6-streaming vs K8-streaming): mean diff
  0.001331 full / 0.000847 clean17 — ordering survives, also as stated.
- **P1-02:** the committed cards' `scope_digest` read
  `attn.o=quantized:...|attn.qkv=quantized:...|mlp.*=quantized:...` against a
  registry artifact record correctly stating those tensor classes are native.
  The live Hub cards served the same stale scope. Confirmed, and confirmed
  that `fidelity-card validate` passed them.
- **P1-01:** the seven-field key demonstrably omits lane (group
  `cmp--202b717f3219c414` holds one artifact measured on two lanes), pipeline
  (~24% measured effect), hardware (2.97e-4 nats A100-vs-H200), and scope.
  The "if and only if" phrasing was ours and it was wrong on the *if* half.
- **P1-05:** the algebra is as the review states:
  `D(P‖Q_quant) − D(P‖Q_control) = E_P[log Q_control − log Q_quant]` — not a
  divergence, sign-indefinite, causal only under assumptions our own
  pipeline/hardware studies undermine. The 2.52× ratio carried no uncertainty.

## Per-finding disposition

Verdicts: **CONFIRMED** (finding correct, fixed as recommended or equivalent),
**CONFIRMED-PARTIAL** (correct in part, or fix goes beyond/differs from the
recommendation), **PENDING** (owned, not yet landed). Batch owners: *science*
(this response's author), *numerics* (`tools/fidelity.py`, registry ingestion),
*operations* (`measure_cloud`, remote stages, security).

| Finding | Verdict | Action | Commit(s) |
|---|---|---|---|
| **P1-01** comparability key not sufficient | **CONFIRMED** | Review's option 2, which matches a decision already recorded in `docs/ARCHITECTURE-DETERMINISM.md` (hardware was deliberately kept out of the key): the key is redefined as a **necessary partition key**, unversioned and unchanged (rehashing would regroup all 76 published rows); a machine-readable per-group predicate (`comparable: true/false/unknown` + reasons over lane, pipeline, scope coverage, hardware) is rendered into `index.json`, recomputed and enforced by new validator rule **CMP-007** (a hand-promoted mixed-lane group is rejected like a forged key); the rendered README states each group's verdict above its table; `registry-submit`/`--explain` peer listings apply the predicate pairwise instead of printing same-key rows flatly as "comparable"; the iff claim is withdrawn by name and date in `registry/README.head.md`, `WHAT-WE-MEASURE.md` §5, `llms.txt` Rule 1. | `87a2214`, `fc6215e` |
| **P1-02** false scope on public K6/K8 cards | **CONFIRMED** | Both cards regenerated from the current registry (`fidelity-card annotate`; frontmatter is generated, never hand-edited) and **pushed to the Hub** — live commits [`9ab94105`](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-6bpw/commit/9ab94105a71708a19c6d960d24b4aa6d459f5623) (6bpw) and [`7199f6f1`](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-8bpw/commit/7199f6f1a211084c240614806f046f11a52dad64) (8bpw), which also carry the CC-01 power-arithmetic fix to the live K8 card. New validator rule **XC-7**: a quant card's `artifact_id` must resolve, its `scope_digest` must equal the registry artifact's, and a stale registry snapshot is an **error** unless the card marks itself archival. The pre-XC-7 validator passed the false-scope cards; selftests K8c/K8d/K8e fail without it. 75→76 counts fixed in `README.md`/`llms.txt` and now derived-checked by `check_doc_numbers`. | `097364c` |
| P1-03 reaper false-success / wrong-target | CONFIRMED (operations) | Leases authorize, names only discover, destroys are provider-confirmed; deadline parsing bounded. | `686ef6d`, `a50d0f4` |
| P1-04 aggregate gate not a portable release signal | CONFIRMED-PARTIAL (operations, in progress) | Reviewer's addendum already notes 69/0/0 in the maintainer environment and same-day Actions runs; environment-coupled tests being classified; `make check-release` now fails on warnings; dry-run INCOMPLETE verdict landed. Root CI matrix beyond the container workflow remains open. | `6e81dda`, `bdf6eb6`, `6249521`, `249ca8a`; remainder PENDING (operations) |
| **P1-05** "quantization-attributable" is not causal attribution | **CONFIRMED** | Renamed **`excess_over_control`** everywhere user-facing: rendered registry column + note, card metric type `kl_divergence_excess_over_control` and `x_fidelity` fields (spec, schema, generator, validator, examples), `WHAT-WE-MEASURE.md` (now stating the algebra), `llms.txt`, `engines/BF16-FLOOR.md`, `registry/README.head.md`. The **2.52× ratio is withdrawn** wherever it appeared without uncertainty; the residuals (0.002209 / 0.000878 nats) stand beside their raw values with the floor named. Registry bias-detail strings inside 11 published rows corrected per the registry's correction rules — additive, disclosed, old wording quoted verbatim, `reseed_delta` receipt showing 0 numbers moved — as `PUBLISHED-CORRECTIONS.md` §8; new `registry_add` emissions use the new sentence and the seeded rows still rebuild byte-identically through it. | `c459090` |
| P1-06 37 Qwen rows claim float64, reduce in float32 | CONFIRMED (numerics) | Reduction now float64 before the vocabulary sum; the 37 rows relabeled `float32_reduce_legacy` with comparability keys split; correction published (`PUBLISHED-CORRECTIONS.md` §6). | `e7e6464`, `bdbe6f3` |
| P1-07 determinism ingest from one digest + four missing | CONFIRMED (numerics) | One valid digest per claimed run, equality across all, else `identical` is not asserted. | `c2a9ee5` |
| P1-08 NaN/Infinity pass schema and seal | CONFIRMED (numerics) | Refused at ingest, seal and render; `allow_nan=False` canonical serialization. | `1f94ce3` |
| P1-09 zero-KL receipt for disjoint datasets | CONFIRMED (numerics) | Disjoint pair refused before compute; outputs publish atomically after validation. | `5410508` |
| P1-10 same-stack fingerprint underbound | PENDING (numerics/operations) | Owner to unify the canonical stack fingerprint and separate exact identity from bridged equivalence; row to be filled when it lands. | — |
| P1-11 mandatory TR3 seal check fails open | CONFIRMED (operations) | Tri-state `verified/failed/not_checked`; `not_checked` blocks a real run. | `a96571f` |
| P1-12 resume can relabel old outputs | CONFIRMED (operations) | Job identity resolved-first and widened; stage markers bind inputs/outputs. | `21c92f7` |
| P1-13 container reuse returns model A's receipts for B | CONFIRMED (operations) | Markers bind the job contract; reused roots refuse on mismatch. | `21c92f7` |
| P1-14 liveness-probe failure launches duplicate writers | CONFIRMED (operations) | Tri-state liveness; unknown does not authorize a second writer. | `21c92f7` |
| **P1-15** pseudoreplication: 25 windows from 4 documents | **CONFIRMED** | Reviewer's numbers reproduced independently before any change (see above). `document_id` preserved through `bin/jointstd/stats.py` / `bin/joint_standard.py`; every paired receipt regenerated (same seeds/B — every previously published statistic reproduces bit-for-bit) with a `document_level` block as the **only inferential statement** and `window_stats_are` relabeling window-level statistics as descriptive of this fixed panel; `PROTOCOL-ALIGNMENT.md` §4.4 correction banner; model cards carry the same correction; logged as `PUBLISHED-CORRECTIONS.md` §7. **Withdrawn:** window-level sign-test p = 0.0041/0.049 and BCa intervals *as population inference*; any implication of 25 independent observations. **Survives:** the panel means themselves; the K8-over-K6 ordering *on this panel* (all 4 document means positive; same-lane recompute 0.001331/0.000847 preserves it); every per-window number as a description of these exact windows. | `50930b1`, `6baf9bd` |
| **P1-16** paired loader drops provenance, permits mixed-lane inference | **CONFIRMED** | The loader now carries each side's measurement contract (lane/reference/estimator/scope, as recorded); a **mixed-lane contrast refuses** without an explicit `--bridge` statement (carried verbatim into the receipt; a bridge is context, not a correction); recorded reference/estimator mismatches refuse with no override; the historical mixed-lane pairings are regenerated with their mix declared in `cross_lane`, and the new same-lane `paired.K6stream-vs-K8` receipt is published beside the historical one. Selftests: mixed-lane refusal, bridged emission, same-lane-no-bridge — each fails on the pre-fix tool. | `50930b1` |
| P1-17 container workflow drops its publish plan | CONFIRMED (operations) | Publish plan survives the pipeline; armed run publishes. | `df7e7bd` |

**Related, same batch:** RC-001 (remote-code policy) was CONFIRMED-PARTIAL on
follow-up — a row could carry a `remote_code` disclosure while its recorded
harness digested only the suite's own closure. Tightened same day: the harness
must carry a `code_digests[]` entry with `role=remote_model_code`, or the row
is refused; selftest added (`fc6215e`).

## The claim-to-evidence audit table, claim by claim

| Review claim audit | Our verdict | Disposition |
|---|---|---|
| Equal key **iff** comparable | Review right; claim was false | Withdrawn; necessary-only key + CMP-007 predicate (`87a2214`) |
| Registry encodes panel/teacher/direction/precision/scope/lane | Review right; key has seven fields, no scope/lane | Prose aligned in `WHAT-WE-MEASURE.md` §5, `llms.txt`, `registry/README.head.md`; the two-layer encoding (key + predicate) now actually exists (`87a2214`) |
| 25/17-window CIs and sign tests reflect independent evidence | Review right | Document-level reanalysis is the inferential statement; window level relabeled descriptive (`50930b1`) |
| Paired inference enforces compatibility | Review right; loader dropped everything | Contract carried; mixed designs refuse (`50930b1`) |
| K6/K8 card scope describes the artifact | Review right; false on committed and live cards | Regenerated + pushed; XC-7 makes recurrence a validator error (`097364c`, Hub `9ab94105`/`7199f6f1`) |
| Qwen accumulation is float64 | Review right (numerics) | Fixed + 37 rows relabeled/keys split (`e7e6464`, `bdbe6f3`) |
| Multi-run evidence bitwise identical | Review right (numerics) | Per-run digests required (`c2a9ee5`) |
| Canonical sealed JSON can contain NaN | Review right (numerics) | Refused (`1f94ce3`) |
| Partial comparison compares a common subset | Review right (numerics) | Disjoint refused pre-compute (`5410508`) |
| Dry-run runs all pre-rental gates | Review right (operations) | INCOMPLETE verdict blocks (`bdf6eb6`, `a96571f`) |
| TR3 seal mandatory before rent | Review right (operations) | Tri-state gate (`a96571f`) |
| Same image-content hash = same compute bytes | PENDING (operations) | Container manifest scope to be honest about what it covers |
| Token absent during remote-code capture | PENDING (operations) | Fetch/capture separation |
| Every number links to a public receipt | Review right; 37 rows pointed at one workstation | Public pinned mirrors cited on all 37, hash-verified; 78 validator warnings → 41 (`2712088`) |
| No third-party source code | PENDING (operations/licensing) | Inventory + scoped licensing |
| "Nothing floats" in the container | PENDING (operations) | Digest/hash locks |

## Claim language

The review's safe-claim template and unsafe-claims list are adopted nearly
verbatim in `WHAT-WE-MEASURE.md` §6a and `llms.txt`. The unsafe list now names
the withdrawn 2.52× ratio explicitly, so the retraction travels with the rule
that would have prevented it.

## Where the review was wrong

We looked. In the science/governance batch: **nowhere that survives checking.**
The one candidate — "no recorded Actions runs at the audited snapshot" —
is dated (the container workflow ran the same day CI landed), and the
reviewer's own chronology addendum already says so; the current review text
("At the audit instant there were no Actions runs; later same-day runs are
documented in the chronology addendum") is accurate. Every number the review
computed about our statistics (document counts, sign-test p-values, document-
level t intervals, the same-lane recompute) reproduced exactly from our own
committed data, which is the strongest kind of review a project can receive.
The correct response to being measured with your own yardstick and found short
is to fix the yardstick's claims, not the reviewer's report.

## What this changes about the project's promise, in one paragraph

The registry's numbers did not move — not one `metric.value`, receipt, or
digest. What moved is what we *claim* they prove. An equal comparability key
now certifies partition membership, not likeness; the K6/K8 contrast is a
described property of one four-document panel, not a population inference; the
floor-subtracted residual estimates excess over a control, not a caused
quantity; and the public cards say what was actually quantized. The K6→K8
ordering, the FP8 comparisons, the floors, and the determinism evidence all
survive — at their honest scope, which is the only scope worth publishing at.
