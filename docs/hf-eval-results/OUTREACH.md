# Two asks for the Hugging Face Hub team

Drafts. Nothing here has been sent. Both concern
[Evaluation Results](https://huggingface.co/docs/hub/eval-results).

---

## Ask 1 — add `fidelity-kld` to the evaluation-framework enum

PR against
[`huggingface.js/packages/tasks/src/eval.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/eval.ts):

```ts
"fidelity-kld": {
    name: "fidelity-kld",
    description:
        "Distribution-fidelity measurement: full-vocabulary KL divergence between a candidate model and a reference model on frozen token IDs, teacher-forced, fp64. Used to measure what quantization costs. Lower is better.",
    url: "https://github.com/malaiwah/glm53-flash-fidelity-suite",
},
```

Every framework currently in that enum scores a model against human-authored
answers. This one does not: it measures how far a model's output distribution
has moved from another model's. It is the axis on which quantizations are
compared, and it has no home in the current list.

## Ask 2 — allow-list a distribution-fidelity panel as a Benchmark

`eval.yaml` is drafted in this directory. The dataset would host the frozen
token IDs and the reference identity; model repos would carry scores in
`.eval_results/`.

---

## The message

> Hi — we maintain a receipt-backed registry of quantization fidelity
> measurements (KL divergence against an unquantized reference), currently
> covering ~70 measurements across three model families and six formats
> (EXL3/TR3, GGUF, MLX, NVFP4, AWQ, FP8). We would like these to live in the
> Hub's eval-results system rather than in our own dataset and in discussion
> threads, and we have two asks — but the second one may need a design
> conversation first, so here is the shape of the problem.
>
> **What we measure.** Not accuracy against human answers. We take a frozen set
> of token IDs, run a reference model (e.g. `zai-org/GLM-5.3-Flash-BF16`) and a
> candidate (a quantization of it) over the same positions teacher-forced, and
> compute mean `KL(reference || candidate)` over the full 154,880-entry
> vocabulary in fp64. Lower is better; exactly 0.0 means the candidate
> reproduces the reference distribution bit-for-bit. It is how you find out what
> a 4-bit quantization actually costs, and it is cheap: about $6 of rented GPU.
>
> **Ask 1 is small:** add `fidelity-kld` to the `EVALUATION_FRAMEWORKS` enum. PR
> ready. Every existing entry is a QA/agentic framework; nothing there fits a
> reference-relative distributional metric.
>
> **Ask 2 is the allow-list**, and it comes with three questions where we would
> rather follow your design than invent one:
>
> 1. **Lower-is-better, unbounded.** The examples are accuracies in [0,1]. Ours
>    is a divergence in nats: 0.0125 is excellent, 0.155 is bad, and there is no
>    upper bound. Does the leaderboard have a direction/format convention, or
>    should we encode it in the task id?
>
> 2. **The score is meaningless without its tuple.** A KLD number is comparable
>    to another only if panel, reference model+revision, KL direction, estimator
>    precision, quantization scope and measurement lane all match. Two examples
>    of what goes wrong: the *same bytes* measured through a serving engine and
>    through a reference forward differ by ~0.00095 nats (we have a byte-identical
>    mirror pair that shows this); and replaying one model's hidden states through
>    a *different* model's LM head erases that model's head-quantization error and
>    flatters it. We currently encode lane in `task_id` and put scope and
>    `head_bits` in `notes`, but `notes` is free text and a leaderboard cannot
>    enforce it. Is there an intended place for structured, comparability-defining
>    metadata — or would you accept a benchmark defining several tasks purely to
>    keep incomparable results apart?
>
> 3. **Third-party results are the norm here, not the exception.** Most
>    quantizations are published by people who do not run this measurement. We
>    measure them and open a PR. Your community-provided badge and
>    author-can-close-the-PR mechanism fit that exactly — better than the
>    discussion threads we have been using. We want to confirm that submitting
>    results *to other people's model repos*, at volume, with clear attribution
>    (`source.user`) and a linked receipt, is the intended use and not an abuse
>    of it. We would only ever submit numbers we produced ourselves, with the
>    receipt attached.
>
> There is also a `verifyToken` path we cannot currently use — our measurement
> is not an inspect-ai task and does not run in HF Jobs. If there is a route to
> auditable verification for a non-inspect framework we would be glad to hear it;
> our own substitute is that every run is bitwise deterministic (repeated cold
> runs produce identical means) and every number ships with a digest-pinned
> receipt.
>
> Repo: https://github.com/malaiwah/glm53-flash-fidelity-suite (MIT)
> Registry: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry
> Method: https://github.com/malaiwah/glm53-flash-fidelity-suite/blob/main/WHAT-WE-MEASURE.md

---

## If ask 2 stalls

`model-index` already works today, needs no gatekeeper, and is what our cards
carry now (see `docs/CARD-ANNOTATION-SPEC.md`). The eval-results path is
strictly better — it puts a number on the model page as structured data with a
dispute mechanism, instead of prose in a thread — but it is not a blocker for
anything.
