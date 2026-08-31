# Handoff — the measurement session

You run the captures and publish the root **datasets**. A second session
(the "registry session") turns what you publish into registry **rows**. The
split is deliberate: measuring and admitting evidence are different jobs, and
the registry has rules a measurer should not have to hold in their head.

> **Your half:** rent, capture, verify, publish the dataset to the Hub, tear
> down, report.
> **Not your half:** writing `registry/data/*.jsonl`, minting measurement ids,
> comparability keys, `make check`, seeding, HF registry-dataset pushes.
> Do not edit anything under `registry/`.

Repo: `https://github.com/malaiwah/quant-fidelity-suite`. Read `AGENTS.md`
first — it is short and every line in it was paid for.

---

## 0. The container, and the four things that will bite you

```bash
docker run --gpus all --rm -v /data/run:/workspace \
    ghcr.io/malaiwah/quant-fidelity-measure:main doctor
```

Public, no login, `linux/amd64` + `linux/arm64`. Current digest
`sha256:9434d971ec8de52b73316f162461374b818057f9e6cd866bcef2282dafa1e0d5`
— **pin it** with `--image-pin`, because that is what lands in the receipt's
`produced_by.container_digest`. The entrypoint mirrors the CLI (`measure`,
`capture`, `stage`, `doctor`, `version`) and `--dry-run` works on all of them
and spends nothing. Full contract: [`docs/CONTAINER.md`](CONTAINER.md).

This image was validated on 2026-08-31 by re-capturing a published root and
getting **the same content digest, bitwise** (§5). It took four rentals and
$0.30 to get there, and all four lessons are now either fixed in code or listed
here. They will still bite you if you ignore them.

**(a) Results come back through a sink, never off the filesystem.** A rented
pod's volume is not readable by you: RunPod's REST API serves no logs and no
files, and this image runs no sshd. `stdout` is always delivered (framed
between `===== FIDELITY-RESULT BEGIN =====` and `===== FIDELITY-RESULT END =====`),
and it now carries the failing stage's log inline. Add `--result-sink` for a
channel you can automate:

```
--result-sink file:/workspace/out          # a mount you own
--result-sink https://ntfy.sh/<topic>      # PUT of a tar.gz; poll it back with ?poll=1
```

A URL that carries its own credential goes in the `FIDELITY_RESULT_SINK`
environment variable, **never** in the command — providers echo the command
back in their consoles and API listings.

**(b) Use `docker_cmd=[...]`, not `docker_args="..."`.** The list goes through
REST `POST /v1/pods` as real argv. The string goes through GraphQL as one flat
field and cannot carry an argument with a space, nor an **empty** one. That is
not hypothetical: `--sanity-expect ''` is an empty argument and is required for
proxy models (§0d).

**(c) Pass `cuda_versions=["13.0"]`.** The image pins torch cu130. A host with
a 12.4 driver will pass `setup`, fetch the whole checkpoint, and die on the
first `.to(cuda)`. The container now refuses that up front — but refusing still
costs a rental, and not landing there costs nothing.

**(d) Two capture guards you must answer deliberately, not reflexively.**
  * `--allow-unexpected-tensors` — needed whenever the checkpoint carries an
    MTP/draft block `transformers` does not build (Fruit's layer 13, and
    **GLM-5.3 has one too**). It forces a *blocking* disclosure, which is
    correct: the same signature is what a silently-disengaged quantizer looks
    like. Never pass it without first checking the converted/excluded module
    split against the checkpoint's real tensor names.
  * `--sanity-expect ''` — the generation probe asks "The capital of France
    is" and expects `Paris`. That is a real check (it catches a shard loaded as
    zeros). Undertrained proxy models fail it legitimately; `''` records the
    probe without enforcing it. **Do not** pass `''` for a production model.

A worked launcher lives in [`docs/CONTAINER.md`](CONTAINER.md) under
"Running it on a provider".

---

## 1. Priority one — the re-measurements

`cd registry && make check` prints **41 warnings**. Three rules are 35 of them:

| rule | n | meaning | yours? |
|---|---|---|---|
| `HARN-005` | 15 | a row's recorded code digest ≠ the current checkout | **No. Leave it.** It is the drift detector working and it is supposed to grow. |
| `STAT-005` | 11 | a published KL with no `top1_agreement` | 9 of them |
| `FLOOR-003` | 9 | a ranked comparability group with no measurement floor | 3 of them |

Do not try to drive the count to zero. Several of these are honest statements
about evidence nobody has.

### 1a. First, a code change with no GPU

Five of the eleven `STAT-005` rows are `clean17` **recomputes** — the same
per-window means re-averaged over the 17 of 25 windows that survive the
calibration-overlap scan. Top-1 over a subset cannot be derived from a
whole-panel scalar, and the receipts retain per-window **mean KLD only**.

**Make the scorer emit per-window top-1 counts** alongside the per-window means.
Without this, §1b clears seven rows and the next subset scope reintroduces the
warning.

### 1b. Re-measure seven GLM-5.3-Flash artifacts

Panel `panel--glm53.brandonmusic.final25`, streaming lane, same settings as the
existing rows. All seven need re-running: the original student logits were the
transient ~63 GB tree deleted at teardown.

| row (`measurement--glm53.` prefix dropped) | repo | revision |
|---|---|---|
| `bf16-stream-floor.brandonmusic-final25` | `zai-org/GLM-5.3-Flash-BF16` | `a6c167b62691b2bac901344b65cb651a70f53e43` |
| `bf16-replay-floor.brandonmusic-final25` | `zai-org/GLM-5.3-Flash-BF16` | same |
| `dione-q4.brandonmusic-final25` | `0xSero/GLM-5.3-Flash-EXL3-Q4` | `99cccdf0e8741715662c383828a9ea601990c125` |
| `k6-6bpw.brandonmusic-final25` | `malaiwah/GLM-5.3-Flash-TR3-6bpw` | **unpinned — resolve and pin** |
| `k6-6bpw-stream.brandonmusic-final25` | same | same |
| `k8-8bpw-stream.brandonmusic-final25` | `malaiwah/GLM-5.3-Flash-TR3-8bpw` | **unpinned — resolve and pin** |
| `official-fp8.brandonmusic-final25.crossstack` | `zai-org/GLM-5.3-Flash` | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |

Three of these already publish a panel-wide top-1; they are on the list because
their `clean17` children need the **per-window** counts from §1a.

**Two rows are not yours to fix.** `brandonmusic-4bpw.brandonmusic-final25` and
its `.clean17` are `measured_by: author-reported`. We do not invent a top-1 for
somebody else's number. They keep the warning.

### 1c. Floors — read before measuring

Nine groups, four different problems. Only three are measurements.

**(a) One is nearly free, and it is the interesting one.**
`cmp--2b9c401d13806d7e` (4 rows on `panel--glm53.brandonmusic.final25-clean17`)
has a floor one field away: `bf16-stream-floor.brandonmusic-final25` is
`float64` / `same_stack` / `native_head` — the group's exact key, on the
**panel25** scope rather than **clean17**. Recomputing it over the 17 retained
windows is the operation already applied to four rows in that group, and §1b
produces the per-window means it needs. No extra GPU.

> ⚠️ **Expect a result, not a formality.** The panel25 floor is **0.011506**;
> `k8-8bpw-stream.clean17` reports **0.010829**. If the recompute lands above
> that value, `FLOOR-002` fires as an **error**, and the honest reading is that
> the 8bpw row sits at or under the lane's noise floor. Report it. Do not
> suppress it, and do not "fix" it by choosing a different scope.

**(b) Two need a floor that does not exist** — `cmp--18990ab191ea7a67` and
`cmp--b55c2d693d127f20`, both on `panel--glm53.brandonmusic.final-0000`, which
has no floor under any key. A BF16 self-compare clears the first (`float64`).
It does **not** clear the second, whose `accumulation_dtype` is `unknown`: that
is a records problem — identify what those six rows actually accumulated in
before measuring anything for them.

**(c) Five are a decision, not a measurement.** The Qwen groups
(`cmp--75b64be1f101ed22` ×12 rows, `c8c4df32774bdb63`, `726ac1b18b8129fa`,
`0bb49e8411b6dc75`, `47c0bc74ebec3fa7`) carry `float32_reduce_legacy` from the
P1-06 correction. Any floor measured today runs the *fixed* float64 comparator,
lands under a different key, and does not bound them. Two honest options —
**ask Michel, do not pick one**: re-run the old comparator (in git at
`e7e6464^`) to floor its own apparatus, or migrate all 33 rows onto float64
where `qwen38-hf.bf16-selfcompare-floor` already sits.

**(d) One is not fixable by us.** `cmp--492e9b16e8bd6fbd` (5 orcarouter MLX
rows) sits on `panel--orcarouter.undisclosed`. We do not have the panel.

---

## 2. The ladder, in order

Plan of record: [`docs/CAPTURE-SCALING-PLAN.md`](CAPTURE-SCALING-PLAN.md).
Everything M1 learned that M2 and M3 inherit:
[`docs/M1-QWEN38-ROOT-LEARNINGS.md`](M1-QWEN38-ROOT-LEARNINGS.md). Read both
before rung M2.

| rung | target | size | state |
|---|---|---:|---|
| M1 | `Qwen/Qwen3.8-27B` | 55.6 GB | **done** 2026-08-30, $5.12, floor exactly 0.0 |
| M1.5 | comparison-term rework | — | **done**, $1.27 |
| M2 | **GLM-5.3-Flash re-capture** | 642.7 GB | **next** |
| M3a | MiniMax-M3 root | — | capture proven, dataset lost to teardown |
| M3b | **GLM-5.2** | 1,506.7 GB | best value on the board |
| M3c | **GLM-5.3** | 1,506.7 GB | the lineage question |
| M4 | Qwen3.5-397B | 806.8 GB | backfill |
| M5 | `Tencent Hy4-preview`, `DeepSeek-V4-Flash-Vision-Exp` | — | trending; feasibility unassessed |
| M6 | Kimi-K3 / Qwen 2.4T | — | GH200-economics question, not yet a plan |

### M2 — GLM-5.3-Flash, same-lane root

`zai-org/GLM-5.3-Flash-BF16` @ `a6c167b62691b2bac901344b65cb651a70f53e43`.

**Reuse the panel; do not mint one.** M1's learning 14: a fresh panel would
make the new number differ from the existing eight Flash rows by panel *and*
lane, with no way to separate them. The panel directory is public and already
in the layout `capture` wants:

```
brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits @ 95f4fdd94bf29989db2e0d1054e4931f55edb6aa
  calibration/panel-v1/panel.json
  calibration/panel-v1/arrays/*.npy
```

Fetch that subtree and point `--panel-dir` at it. **Transport the tokens, never
re-tokenize**, and seal-check before use.

Two decisions M2 must make before it starts, both flagged in the plan:
1. **Which replay backend the whole group uses** — it is part of the number
   (`CAPTURE-SCALING-PLAN.md` §3.3).
2. **Whether the GLM-5.3 MTP layer's blocking `--allow-unexpected-tensors`
   disclosure is acceptable on a sealed root.** It is required — the capture
   refuses otherwise. Michel decides; do not decide it silently.

**State plainly on every row you hand over:** a same-lane root does **not**
retroactively upgrade the eight existing Flash rows. The comparability key binds
the reference, so this creates a *new* group beside the old one. The eight keep
their inferred 0.011506 floor forever unless each is re-measured against the new
root — which is a separate cost line, ~86 s each on the cuda replay path.

### M3a — MiniMax-M3

`MiniMaxAI/MiniMax-M3` @ `f0e1c1e04d40177e4673a22097036854f536e9c0` (unchanged
upstream). Panel `panel--minimaxm3.malaiwah.corpus5x5` is **baked into the
image** at `/opt/fidelity/suite/engines/panels/` — no panel fetch needed.

This root was captured twice, bitwise-identically, and then destroyed at
teardown before it was published:

```
f84237e4b3c4a50a9c3aee9f271573fb11c279f68b16b88e508d398aae822276
```

That is your acceptance target. Teardown now refuses to destroy a
verified-but-unpublished root, and `--publish-root-to` uploads before teardown
can run, so the failure mode is closed — but capture it expecting that digest,
and say so loudly if it differs. Race mode (`--race --preview-of`) is available
and untested in anger; a preview publishes under a **different** dataset id and
is never updated in place.

### M3b/M3c — GLM-5.2 and GLM-5.3, and the lineage question

Same geometry (78L / 6144 / 256e), so one engine and one panel design serve
both. GLM-5.2 is the better first move: identical cost, ~3× the quantized
children.

Capturing both roots is what makes the open scientific question answerable:
**can a full-vocabulary KLD distinguish a post-training descendant from an
unrelated model?** GLM-5.3 is understood to be post-training only on top of
GLM-5.2. Architecture and tokenizer identity are already verified. Capture both
against the same panel and hand both over; the analysis is the registry
session's.

---

## 3. Rules — not style preferences

* **Never `git add -A`.** Stage your own files by name; other sessions write in
  this tree. `git pull --rebase origin main` before every commit.
* **Do not touch `registry/`.** That is the other session's half.
* **Every regression test must be verified failing against the pre-fix code**
  (a `git worktree` at the prior commit). A test that has never failed has never
  been shown to test anything. Watch out for a crash counting as a "fail" —
  check exit codes, not grep hits.
* **`--dry-run` before anything is created**, every time. Destroy every instance
  you create, on every exit path.
* **Never edit a sealed receipt**, never rename a registry id, never touch
  `reports/stack-provenance-retro.json`.
* **The HF token is read from a file, never echoed, never committed, never in
  argv.** Same for provider keys.
* **Machines `485913` and `483634` must never be touched.**
* `bash bin/selftest_all.sh` (**83 passed / 0 failed / 0 skipped**) green before
  every push.
* **An outer PASS can hide a SKIP.** This cost a rental on 2026-08-31: the
  bootstrap reported `PASS 1b accelerator decode parity: SKIPPED (no CUDA … the
  check RUNS on the instance, which is where it counts)` — while running on the
  instance. Read what a PASS actually asserted.

---

## 4. What to hand back, per capture

Post these and stop; the registry session takes it from there.

1. **Dataset repo + revision** — e.g. `malaiwah/glm53-flash-fidelity-root-v1`
   at `2a3133f1…`. `receipts/publish-root.json` has both, plus
   `verified_after_publish`.
2. **`capture_content_digest`** from `fidelity-dataset.json`, and whether it
   matched an expected value if there was one.
3. **The result bundle** from your sink (receipts + logs + `job.json` +
   `result-summary.json`).
4. **The image digest you pinned** and the GPU model. Both are part of the
   number: the GPU *model* — not the provider, not the host — is what moves
   these bits ([`ARCHITECTURE-DETERMINISM.md`](ARCHITECTURE-DETERMINISM.md)).
5. **Every disclosure the capture emitted**, especially blocking ones, and the
   reasoning behind any override you passed.
6. **What it cost**, and what the plan said it would.

Do not pre-judge admissibility. A capture with a blocking disclosure is still
worth handing over — it is evidence with a caveat attached, and the caveat is
the registry's job to record.

## 5. The container's own acceptance test, for reference

On 2026-08-31 this image re-captured `malaiwah/GLM-5.2-SIQ-Fruit-bf16` on a
RunPod L4 and reproduced the published root **bitwise**:

```
capture_content_digest  b417acc22b8aa7f3294b8e62c4b619bc5051aef9fd8a073602572a30af6b3e1c
```

Same digest as `malaiwah/fruit-fidelity-root-v1`, and all nine stack-fingerprint
fields identical. Published to `malaiwah/fruit-fidelity-root-container-v1`. If
you want to convince yourself the transport is sound before spending real money,
that run costs about five cents and the expected answer is written above.
