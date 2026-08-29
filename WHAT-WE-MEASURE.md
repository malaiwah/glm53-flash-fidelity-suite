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

- **The lm_head weights are never quantized** in any artifact measured
  here. Our TR3 quants keep it bit-exact BF16 (along with the entire
  KDA/attention path, hyper-connections, routers, norms and embeddings —
  only routed experts and the MTP layer carry quantized weights). Z.ai's
  official FP8 release draws the *same* boundary: its
  `modules_to_not_convert` list keeps the head and the attention path
  native. Two teams drew the sensitivity split independently in the same
  place.
- So a number here is "body-quantization error, observed through the full
  model including the (native) head" — never "head quantization error".
- In the cross-stack lane (below), the head is additionally applied
  **outside the serving engine**, identically to both teacher and student
  hidden states in fp32 ("shared-head replay"), so head arithmetic can
  never contribute a *differential* error between the two sides.

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

## 4. The floor — why zero quantization still scores above zero

Running the **unquantized BF16 weights** as the student still scores
**0.011506** on the streaming lane and **0.012712** on the cross-stack
lane. That residual is the *price of the comparison itself*: the teacher
logits were captured on a different runtime than the student replay, and
bf16 addition is not associative — expert-combine order alone moves logits
materially. Consequences:

- **Attributable error** = row − *same-lane* floor. That is the number that
  ranks quants fairly: K6 = 0.002209, K8 = 0.000878 (2.52× apart, where the
  raw means are only 1.11× apart).
- **Cross-lane floor subtraction is invalid** and the registry's validator
  refuses it mechanically (invariant BIAS-006).
- A quant scoring *at* the floor is not "perfect" — the panel has simply
  run out of resolving power for it.

## 5. What a row must pin for two rows to be comparable

Same **panel** (corpus, windows, positions, tokenizer), same **teacher**
(weights *and* the runtime that captured its logits, by hash), same
**direction**, same **estimator precision**, same **scope policy** (what is
quantized vs native), same **lane** — or a disclosed, *measured* bridge
between lanes (ours: streaming vs sealed = −8.5e-6, receipt-backed). The
[registry](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry)
encodes all of this as a comparability key and renders rows from different
keys in separate tables. If a number you meet in the wild does not pin
these, it is an anecdote, not a measurement.

## 6. What these numbers are NOT

- Not a task benchmark: KLD measures distributional fidelity to the
  teacher, not reasoning, coding, or long-context skill.
- Not a long-context result: panel contexts are 2,048 tokens.
- Not a serving-quality statement for checkpoint-lane rows: a checkpoint
  can measure superbly and still be served badly by a buggy kernel — the
  checkpoint lane deliberately cannot see that.
- Not transferable across panels: per-window scatter (sd ≈ 1.7e-3) exceeds
  the K6-vs-K8 effect (1.2e-3), which is also why **a single window can
  never compare two quants** (campaign lessons 28/29).

## The measurers' checklist

1. State the direction, estimator precision, and scored-position count.
2. Pin the panel, the teacher (weights + capturing runtime), and the
   artifact revision by hash.
3. Say which lane you measured — artifact, or artifact + stack — and why.
4. Publish run count and determinism evidence (content hashes, never
   container hashes — receipts embed timestamps and metadata that differ
   between bit-identical runs).
5. Measure and publish **your lane's floor**; report attributable error.
6. Disclose every deviation before anyone asks.
7. Never quote a single window as a rate comparison.
