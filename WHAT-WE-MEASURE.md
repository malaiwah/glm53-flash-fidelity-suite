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
- Not transferable across panels: per-window scatter (sd ≈ 1.7e-3) exceeds
  the K6-vs-K8 effect (1.2e-3), which is also why **a single window can
  never compare two quants** (campaign lessons 28/29).

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
8. Fingerprint the stack — engine build, eager/graph state, attention
   backend, kernels, env pins, image digest — and cite that fingerprint by
   digest from every receipt that used it.
