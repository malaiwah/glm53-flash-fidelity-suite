# What we measure (read this before comparing any two numbers)

The recipes in this repo tell you **how** to produce a KLD number. This
document says **what** that number actually is — because two "KLD = 0.02"
figures produced with different answers to the questions below are not
measurements of the same thing, and ranking them against each other is
meaningless.

## 1. The quantity

**Mean KL divergence D(teacher ‖ student)** over the **full output
vocabulary** (154,880 entries for GLM-5.3-Flash), accumulated in **fp64**,
averaged over a sealed panel of **51,175 positions** (25 windows × 2,047
scored positions, 2,048-token contexts).

- **Direction matters.** `KLD(teacher ‖ student)` weights errors by the
  *teacher's* probability mass: the student is penalized for assigning low
  probability where the teacher assigns high. The reverse direction
  (`student ‖ teacher`) is a different number. Every receipt states the
  direction; rows with different directions are never comparable.
- Top-1 agreement (argmax match rate) is carried as a secondary,
  coarser metric. It saturates: quants that differ 2× in KLD can sit
  within 0.3 points of top-1.

## 2. The measured function — yes, the lm_head is inside it

The function being measured is the **entire model**:
`tokens → embeddings → all 45 layers (KDA linear attention, DSA, MLA,
hyper-connections, routers, routed experts, shared experts, norms) →
lm_head → logits`.

KLD is an *output-distribution* metric, so the head's matmul is always part
of the computation. But distinguish the **computation** from the
**weights**:

- **The lm_head weights are never quantized in OUR artifacts.** Our TR3
  quants keep it bit-exact BF16 (along with the entire KDA/attention path,
  hyper-connections, routers, norms and embeddings — only routed experts
  and the MTP layer carry quantized weights). Z.ai's official FP8 release
  draws the *same* boundary: its `modules_to_not_convert` list keeps the
  head and the attention path native. Two teams drew the sensitivity split
  independently in the same place.
- **Third-party artifacts may draw it elsewhere, and some do.** The
  stock-exllamav3 releases quantize the head at 6 bits; a llama.cpp GGUF
  quantizes `output` and `token_embd` too. For those rows a number is *not*
  "body-quantization error observed through a native head" — the head is
  part of what was quantized, and the row's measured `scope_policy` block
  (§5a) says so. Never read a row's number without it.
- So a number from a routed-experts-only row is "body-quantization error,
  observed through the full model including the (native) head" — never
  "head quantization error".
- In the cross-stack lane (below), the head is additionally applied
  **outside the serving engine**, identically to both teacher and student
  hidden states in fp32 ("shared-head replay"), so head arithmetic can
  never contribute a *differential* error between the two sides.

### 2c. The dataset route's estimand — bf16 hiddens, fp32 replay, own heads

Every GLM-5.3 row (2026-09-04/05) was produced by the capture/compare split of
§8, not by the fused Flash pipeline above, and its number is a slightly
different object:

- **What is captured.** The post-final-norm hidden state of every scored
  position, exactly as the model's own bf16 forward produced it (`hf_capture`
  refuses anything that is not the bf16 bytes), plus the artifact's own
  `lm_head` by tensor-content digest. No logits are stored.
- **What is scored.** `fidelity-dataset compare` recomputes
  `logits = float32(h_bf16) @ float32(W_bf16)^T` for each side through **its
  own sealed head** (`--own-heads`, HEAD-1d: `head_policy: native_head`, the
  head error is inside the number as under HEAD-2) on the numpy fp32 path, and
  applies the fp64 estimator. The receipt now says so:
  `estimator.logits_dtype: float32`, `estimator.hidden_dtype: bf16`,
  `comparator.replay_backend: numpy:cpu:float32`, and `comparator.replay_env`
  names the BLAS, its thread count and the CPU, because the last digits of an
  fp32 GEMM are the BLAS's accumulation order (the workstation-vs-pod term on
  the six rows is 1.8e-10 … 3.8e-9 nats).
- **What a serving stack adds that these rows do not contain.** A bf16 stack
  computes the same product and then **rounds every logit to bf16** before its
  softmax — up to ±0.0625 at |logit| in [16, 32) and ±0.125 in [32, 64)
  (GLM-5.3's logits reach |46|). Measured on the real root
  ([`reports/bf16-logit-rounding/`](reports/bf16-logit-rounding/README.md),
  window `final-0000`, 2,047 positions, the comparator's own replay and
  estimator): KL(fp32 ‖ bf16-rounded) of the root alone is **1.7e-5 nats**, and
  rounding **both** sides of a real comparison moves the published quantity by
  **−1.3e-4 nats (−0.42 %) on the K4 row and −2.7e-5 nats (−0.22 %) on the FP8
  row**. So: hidden-form rows are scored on fp32 logits recomputed from sealed
  bf16 hidden states; logit-form rows from a bf16 stack additionally carry a
  term of the 1e-5–1e-4 nats class (well under 1 % of any GLM-5.3 row). The two
  are the same `head_policy` but not the same estimand to the last percent —
  compare them as such.
- **What "weights-only" means here.** A trellis or FP8 candidate is captured
  from a bf16 reconstruction of its stored weights under the same `transformers`
  forward as the root (`runtime.capture_tool.weights_decode` on the sealed
  runtime receipt). The served kernel's own numerics — exllamav3's fp16
  activations and on-the-fly dequant, an FP8 stack's per-token activation
  quantization — are **not in the number**. The comparator files every such
  receipt as `advisory` with `weights_reconstructed` or
  `activation_quantization_not_captured` (gate 9b, 2026-09-05); the six receipts
  sealed before that gate existed say `strict` and are corrected additively, not
  re-sealed.
- **Head-only artifacts.** Under `--own-heads`, two captures with bitwise-equal
  hidden states and different heads (stock EXL3 `head_bits` 6–8) are a
  measurement of exactly the head-quantization KL (`head_only_difference`);
  through one shared head the same pair is 0.0 by construction and is still
  refused (HEAD-1c).

## 3. Weights, or weights + serving stack? Both exist — as two disclosed lanes

This is the single most important disclosure on any row.

**Checkpoint lane** (called `sealed-ep8` and `streaming` in receipts) —
*measures the artifact, not the engine.* The quantized payload is decoded
exactly (the decode is bitwise-verified against an independent
implementation) and run through a **reference forward** built on
`transformers` — deliberately *not* vLLM or any production serving kernel.
This is what the K6 (0.013723), K8 (0.012384), Dione-Q4 (0.027263) and
BF16-floor (0.011506) numbers are. Because no serving nondeterminism is
present, these runs are **bitwise deterministic** — five (or two) cold runs
produce identical means to the last digit, and the receipts prove it with
content hashes.

**Serving lane** (`cross_stack` in the registry) — *measures the artifact
and the stack together.* The body runs through the actual serving engine
(vLLM), hidden states are captured, and the shared head is applied outside.
The official FP8 figure (0.020615) is this lane **by necessity**: the FP8
release is W8A8 (`fmt: e4m3`, `activation_scheme: dynamic` in its
`quantization_config`) — its activation quantization *only exists at serve
time*, so there is no serving-free way to measure what users of that
release actually get.

Neither lane is "wrong" — they answer different questions. "How good is
this checkpoint?" is the checkpoint lane. "What do I actually get from this
release, served?" is the serving lane. The registry keys them apart so they
cannot be silently ranked against each other.

**A checkpoint-lane row on a W4A4-style artifact is weights-only, and says
so.** Some community quants (the NVFP4 snapshots, `--source nvfp4`) quantize
weights *and* declare quantized activations. The checkpoint lane can decode
and score the weights exactly; the activation half only exists at serve time,
so it is **not in the number** — the same limitation the official FP8 release
has, disclosed the same way. The surface reads the artifact's own config and
index to decide which case it is, rather than assuming one per format family:
a genuine W4A16 artifact is captured *fully* by a weights-only decode and gets
no caveat, while an artifact that declares quantized activations or ships
activation scale tensors carries `activation_quantization_not_captured` on its
registry row.

## 4. The floor — why zero quantization still scores above zero

Running the **unquantized BF16 weights** as the student still scores
**0.011506** on the streaming lane and **0.012712** on the cross-stack
lane. That residual is the *price of the comparison itself*: the teacher
logits were captured on a different runtime than the student replay, and
bf16 addition is not associative — expert-combine order alone moves logits
materially. Consequences:

- **Excess over control** = row − *same-lane* floor (called "attributable
  error" before 2026-08-31; renamed per peer-review P1-05). K6 = 0.002209,
  K8 = 0.000878 on the streaming lane. The difference is an *estimate of the
  excess divergence over the unquantized control*, not a causal attribution:
  algebraically it is `E_P[log Q_control − log Q_quant]`, which is not itself
  a divergence, can be negative, and isolates quantization only if the two
  paths differ by nothing else. Do not quote a ratio of two of these residuals
  without uncertainty — the withdrawn "2.52×" headline was exactly that
  mistake: a ratio of small residuals that magnifies control error.
- **Cross-lane floor subtraction is invalid** and the registry's validator
  refuses it mechanically (invariant BIAS-006).
- A quant scoring *at* the floor is not "perfect" — the panel has simply
  run out of resolving power for it.

## 5. What a row must pin for two rows to be comparable

Same **panel** (corpus, windows, positions, tokenizer), same **teacher**
(weights *and* the runtime that captured its logits, by hash), same
**direction**, same **estimator precision**, same **scope policy** (what is
quantized vs native), same **lane** — or a disclosed, *measured* bridge
between lanes (ours: streaming vs sealed = −8.5e-6, receipt-backed, one
artifact on one panel, not a constant). If a number you meet in the wild
does not pin these, it is an anecdote, not a measurement.

The [registry](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry)
encodes this in **two layers, because one hash cannot carry it all**. Its
seven-field `comparability.key` (panel, teacher capture, metric, direction,
estimator precision, stack relation, head policy) is a **necessary partition
key**: rows under different keys are never comparable and render in separate
tables. It is *not* a sufficient certificate — the key deliberately omits the
measurement lane, the candidate pipeline, hardware, and the artifact's scope,
each of which this project has **measured** moving results by more than the
gaps it publishes between quantizers. The rest of the contract is a
machine-readable per-group predicate in the registry's `index.json`
(`comparable: true/false/unknown`, with reasons), recomputed by the validator
(`CMP-007`) so it cannot be hand-promoted. Before 2026-08-31 this document and
the registry stated the key rule as an "if and only if"; an independent peer
review correctly rejected the *if* half, and the claim is withdrawn — see
`docs/PEER-REVIEW-RESPONSE.md`.

### 5a. Scope policy is not a footnote

Our own EXL3/TR3 artifacts and the third-party Dione trees quantize the
**routed experts only**: they run the reference's untouched embeddings,
attention/KDA/DSA path, shared experts, routers, norms and `lm_head`. The
routed experts are ~97% of the parameters, so that is where nearly all the
loss lives — but it is a *choice*, and it is the choice those rows share.

The community conversions do not all make it, and the differences are not
predictable from the format name:

| family | what the artifact quantizes | measured how |
|---|---|---|
| K6 / K8 / Dione / BF16 floor | routed experts only | contract / index |
| stock-exllamav3 (turbo) | full scope, **6-bit `lm_head` included** | artifact config |
| **MLX** (orcarouter) | routed + dense MLPs + shared experts + 4 DSA projections — 186 non-routed modules on top of 36,288 routed, 4/5/6-bit mix; 1,432 tensors keep their source dtype (embeddings, `lm_head` and the whole KDA path among them) | index + all 62 shard headers |
| **GGUF** (unsloth UD-Q4_K_XL) | everything: `token_embd`, `output` (the lm_head), every attention projection and the shared experts, at Q8_0 | the GGUF tensor table |
| **NVFP4** (RedHatAI, LibertAI) | routed experts only — same scope as K6, different format | the compressed-tensors index |

Three consequences the tooling enforces rather than documents:

- The receipt's scope block is **measured from the artifact's own index /
  tensor table**, never inferred from the format family. A summary that
  arrives without it is refused, not recorded.
- `stream_score.py --source mlx` and `--source gguf` supply **all** tensors
  from the artifact: the non-routed ones are decoded into a materialized
  view and the sealed `from_pretrained` runs over that, so the forward
  really is the artifact's. The official BF16 tree contributes only
  config/tokenizer files and (for GGUF) the vision tower, which the main
  GGUF does not carry and the text-only panel never executes.
- `registry_add.py` renders the block as a row disclosure, so reading
  "0.0x nats, 4-bit" without it cannot silently compare a
  quantized-experts-only artifact against one that also quantized its
  attention path and its head.

Read the block, not the file extension: MLX and GGUF are both "community
4-bit", and they quantize different things; NVFP4 and K6 are different
formats with the *same* scope.

## 6. What these numbers are NOT

- Not a task benchmark: KLD measures distributional fidelity to the
  teacher, not reasoning, coding, or long-context skill.
- Not a long-context result: panel contexts are 2,048 tokens.
- Not a serving-quality statement for checkpoint-lane rows: a checkpoint
  can measure superbly and still be served badly by a buggy kernel — the
  checkpoint lane deliberately cannot see that.
- Not transferable across panels: on the sealed 25-window panel the
  per-window KLD scatter is sd 7.2e-3, and the paired per-window K6-vs-K8
  delta has sd 2.0e-3, against an effect of 1.33e-3 — which is also why
  **a single window can never compare two quants** (campaign lessons 28/29).

## 7. The stack fingerprint — answering the kernel question

"Which vLLM build? Which kernels? Was enforce-eager on?" A number whose
receipt cannot answer those is a number you are trusting, not checking. As
of 2026-08-29 every capture receipt embeds a **stack fingerprint**
(`malaiwah.stack-fingerprint.v1`, hashed canonically with timestamps
excluded, so identical stacks hash identically) and the answer is a field,
not folklore.

- **Serving lane** — the fingerprint is queried from the *live engine*:
  exact vLLM build (version + git sha), torch/CUDA, `enforce_eager` and the
  compilation/cudagraph modes read out of `vllm_config`, the attention
  backend (requested *and* selected, each with its source), the kernel-config
  knobs, the determinism-relevant env pins (Triton autotune, NCCL shape,
  symm-mem, cuBLAS workspace), the container image digest, GPU inventory,
  and the sha256 of the full pip freeze (freeze written alongside). A fact
  the engine will not expose is recorded as **unknown with the reason** —
  never defaulted. Capture manifests embed it verbatim; qualify/replay/
  cross-check receipts additionally name their capture operands **by
  manifest digest**, so the chain from summary number to serving stack is
  hashes end to end.
- **Checkpoint lane** — a reference `transformers` forward has no vLLM, no
  `torch.compile`, no CUDA graphs; its fingerprint says
  *not-applicable-with-reason* instead of a hollow "false". The lane already
  records more than the serving lane ever did: `backend.json` **probes** the
  dispatched grouped-MM kernel with real dtypes, and `lane_identity_sha256`
  hashes exactly the lane-naming fields. The teacher's half of that chain is
  public in brandonmusic's dataset down to `backend.json`
  (`attention_backend: eager`, torch 2.11.0+cu130, EP4). Wiring the
  fingerprint into `stream_score.py` itself waits for an in-flight merge of
  that file; the adapter (`stackprint.from_backend_json`) ships now.
- **The sealed rows** predate the fingerprint. Their evidence is assembled
  retroactively in `reports/stack-provenance-retro.json` (in the suite
  dataset): each sealed summary receipt, *by its digest*, mapped to its
  environment evidence files *by theirs*, plus the established
  `enforce_eager=True` / CUDA-graphs-off / `FLASH_ATTN_MLA_SPARSE` facts —
  each fact labeled with **how** it was established (receipt field, code
  default at the pinned commit — the capture harness *hard-refuses* to run
  without eager mode — or per-boot engine log, by log digest). What could
  not be established is listed as **unknown**, plainly: the sealed launches'
  Triton autotune winner configs (bounded instead by the measured 8.7e-4
  launch-noise floor), the DeepGEMM mHC JIT identity, the full 40-char vLLM
  commit behind `g487ecf187`.
- **The rule going forward:** a receipt without a stack fingerprint — or a
  summary that does not cite its operands' fingerprints by digest — is
  **refusable**, exactly like an unpinned panel. If a tool in this repo
  produced it after 2026-08-29, that absence is a bug report.

## 8. Capture and comparison are two steps, not one

Everything above describes what a *number* must pin. This section is about
*when the work is done*, and it is the one structural change of 2026-08-29.

Until now, capture and comparison were **fused**: `engines/tools/stream_score.py`
ran a model over the panel and `engines/tools/k6_kld_report.py` scored it against a
teacher, and the only durable output was a number plus receipts pointing at
filesystem paths. Three consequences, all of which bit us:

1. **Every measurement re-paid for capture.** Scoring quant *N* against the
   BF16 reference re-ran the reference, or depended on a teacher tree somebody
   was still holding.
2. **Teachers were not portable.** `capture-receipt.json`'s
   `logit_files[].path` are absolute paths on the capture box.
3. **A lost capture killed reproducibility.** The JarvisLabs filesystem holding
   our sealed `layers/*.json` and `experts/*.json` receipt trees was destroyed
   after being wrongly declared redundant. The published K6/K8 checkpoints are
   still self-contained *for serving* — payloads inline, readable through
   `stream_score --source exl3hf` — but the `--source checkpoint` and
   `--source payload-store` reading paths are now **unreachable from public
   artifacts**, and the published materialization receipt still names the dead
   path `/home/jl_fs/glm53-k6/out-k6`. The registry already had a field for
   exactly this condition: `reference.logits_available`, documented as *"false
   means a number against this reference can never be re-derived, only
   re-run."*

### The three steps

```
step 1  capture   reference (root) weights + panel  ->  fidelity dataset A
step 2  capture   quantized weights + panel         ->  fidelity dataset B
step 3  compare   A, B  ->  KLD + determinism + a registry-submittable receipt
                  A, A  ->  reproduction confirmation, exactly 0.0
```

One tool, three modes: `bin/fidelity-dataset capture | verify | compare`.

* **A root capture is a public good.** It is produced once when weights drop —
  or after the fact — sealed, published, and thereafter downloaded rather than
  re-run. Publishing it is what flips `logits_available` to true.
* **Step 2 is publishable standalone.** A quant author can publish their own
  capture with no access to our infrastructure and before any comparison
  exists. Most users discard it; the format does not care.
* **Step 3 runs with neither set of weights present.** It needs two datasets
  and fp64 arithmetic.

### Why this shrinks the floor

Section 4 explains the floor: zero quantization still scores above zero,
because the two sides of a comparison were produced by different stacks. Our
published cross-stack floor is **0.012712 nats** — comparable in magnitude to
K6's entire 0.013723. That number is comparison overhead, not quantization.

When A and B are captured **on the same lane** and compared offline in fp64,
that overhead is removed *structurally* rather than by subtraction — which the
registry forbids across lanes anyway (**BIAS-006**). What remains is
quantization error. The same-lane floor problem does not get corrected; it
largely stops existing.

### The three things the format makes checkable that prose could not

* **Head identity.** "Shared head" means shared *application*, not shared
  *weights*. Replaying a candidate's hidden states through the **reference**
  head erases its head-quantization error and flatters it. Every capture now
  declares its own `lm_head` by **tensor content** digest, and the comparator
  **refuses** a hidden-form comparison across differing heads (HEAD-1b) unless
  you pass `--disclose-head-substitution`, which forces `class: advisory`, a
  downward bias block, and a **blocking** disclosure — i.e. not publishable.
  There is one case that override must not reach: a quant that changes **only**
  the head (stock EXL3 `head_bits` 6–8 does exactly this) produces post-norm
  hiddens bitwise identical to the reference's, so its capture digest matches
  and replaying both sides through one head subtracts a quantity from itself —
  0.0 nats, top-1 1.0, labelled a reproduction. The comparator refuses that
  outright on the shared-head path (HEAD-1c, no override); `--own-heads`
  (HEAD-1d) replays each side through its own sealed head, so the same pair is
  exactly the head-quantization KL and is filed as a measurement with a
  `head_only_difference` disclosure (§2c).
* **Self-compare.** Comparing a capture against itself is a *reproduction
  confirmation* and must yield exactly `0.0`, top-1 exactly `1.0`, and a
  tokenwise array of literal zeros. For our 51,175-position panel that array is
  a fixed constant: 409,528 bytes, sha256 `3ffddc61…be17`. Anything else means
  the estimator or the reader is broken.
* **Storage.** Post-final-norm BF16 hiddens are **75.6x** smaller than fp32
  logits: 419 MB vs 31.70 GB for the 25-window panel, 85.9 GB vs 6.49 TB for
  the 10.48M-position suite. Hidden form is therefore the default; logit form
  stays expressible for stacks whose head is not separable.

### What is runnable today, and what is not

The split is a format plus tooling, not yet a published corpus, and it is worth
saying plainly which is which:

**Runnable now, on a laptop, with no GPU and no weights.** Step 3 in full:
`verify`, `validate`, `describe`, `compare` (including the A == B reproduction
confirmation and every gate refusal), `provenance-template`, `--emit-submission`
with the registry's own gate run over the output, `adapt` from our published
serving-lane capture or from a kimi-k3 artifact, `verify-k3-compat`, and the
whole card generator and validator. The real BF16-vs-FP8 comparison in this
document — 0.0353 nats over 4,094 positions through the real `[154880, 4096]`
head — runs in about ten seconds on this Mac.

**Not runnable yet, and both ends are the reason.** No conformant **root
fidelity dataset** is published, so "download it rather than re-run it" is the
architecture's promise and not yet its state; and no **token panel** is
published, so `capture` cannot be started from a clean checkout. Publishing
suite-scale captures is out of scope for v1 (spec §14) — the 85.9 GB is the easy
part; deciding what a canonical root *is* is the operator decision. Until then
the honest description of steps 1 and 2 is: the format is fixed and the wrapper
is written, and the artifacts they consume have to be produced.
[`bin/README.md`](bin/README.md#before-you-start--what-exists-today-and-what-does-not)
states each gap at the point of use.

Format: [`docs/FIDELITY-DATASET-SPEC.md`](docs/FIDELITY-DATASET-SPEC.md).
Card annotation: [`docs/CARD-ANNOTATION-SPEC.md`](docs/CARD-ANNOTATION-SPEC.md).

## 6a. The claim template — what a number here may honestly say

Adopted 2026-08-31 from the independent peer review, because the difference
between the safe form and the unsafe forms is where every misuse of this data
begins.

**The safe claim:**

> "On panel X, under pipeline/lane/hardware configuration Y, this exact
> candidate's teacher-forced next-token distribution had lower mean
> teacher-to-candidate KL than candidate Z, with the reported sampling and run
> uncertainty."

Every clause is load-bearing: the panel (Rule 2: never one window), the full
configuration (the comparability key is necessary, not sufficient — §5), the
exact candidate (an artifact revision, not a format family), teacher-forced
next-token distributions (not free-running generation), the direction, and the
uncertainty (at its honest unit — for the Brandon panel that unit is the
source document, of which there are four).

**Unsafe claims, without additional evidence none of this data supplies:**

- "This is the best quant."
- "K6 preserves quality 2.52× better than K8." *(withdrawn — a residual ratio
  with no uncertainty; see §4)*
- "The residual is the amount caused by quantization." *(it is the excess over
  a same-lane control, not a causal attribution)*
- "A lower KL means higher task accuracy." *(KL is a distributional-drift
  diagnostic; the flips literature shows accuracy and drift dissociate)*
- "The result applies to long context or served generation." *(2,048-token
  teacher-forced panels; the checkpoint lane deliberately cannot see serving)*
- "The two numbers share a comparability key, so the better one is the better
  quant." *(equal keys make candidates, not a certificate — check the
  registry's per-group predicate)*

## The measurers' checklist

1. State the direction, estimator precision, and scored-position count.
2. Pin the panel, the teacher (weights + capturing runtime), and the
   artifact revision by hash.
3. Say which lane you measured — artifact, or artifact + stack — and why.
4. Publish run count and determinism evidence (content hashes, never
   container hashes — receipts embed timestamps and metadata that differ
   between bit-identical runs).
5. Measure and publish **your lane's floor**; report the excess over that
   control beside the raw value, never a residual ratio without uncertainty.
6. Disclose every deviation before anyone asks.
7. Never quote a single window as a rate comparison.
8. Fingerprint the stack — engine build, eager/graph state, attention
   backend, kernels, env pins, image digest — and cite that fingerprint by
   digest from every receipt that used it.
9. **Publish the capture, not just the number** (section 8). A sealed fidelity
   dataset is what lets anyone re-derive your result instead of re-running it,
   and it is the only thing that survives losing the machine you measured on.
