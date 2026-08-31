---
license: mit
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- glm
- glm-5
- glm5_next
- tr3
- trellis
- mcg
- quantized
- 6-bit
- moe
- reasoning
- text-generation
- fidelity
- kl-divergence
- exllamav3
- fidelity-provenance
datasets:
- brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
- malaiwah/GLM-5.3-Flash-fidelity-suite-v1
metrics:
- kl_divergence
- top1_agreement
model-index:
- name: GLM-5.3-Flash-TR3-6bpw
  results:
  - task:
      type: text-generation
      name: Distribution fidelity (KL divergence vs BF16 reference)
    dataset:
      type: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
      name: brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows -- panel25 subset
      config: final25-panel25
      split: streaming
      revision: 95f4fdd94bf29989db2e0d1054e4931f55edb6aa
      args:
        panel_id: panel--glm53.brandonmusic.final25
        panel_token_sha256: 6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff
        contexts: 25
        scored_positions: 51175
        context_length: 2048
        tokenizer: zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43
        vocab_size: 154880
        reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25
        scope_name: panel25
        covers_full_panel: true
        scope_selection_sha256: 180678c464c8c325a4675a9fd5c54247e99624f0fc911580805e98fd960f79fe
    metrics:
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), nats
      value: 0.013714888822596553
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
        accumulation_dtype: float64
        logits_dtype: fp32
        head_policy: native_head
        stack_relation: same_stack
        lane: streaming
        run_count: 2
        population_stddev_of_run_means: 0.0
        determinism: bitwise_identical_across_runs
        measurement_id: measurement--glm53.k6-6bpw-stream.brandonmusic-final25
        comparability_key: cmp--202b717f3219c414
    - type: kl_divergence_excess_over_control
      name: KLD excess over same-lane unquantized control, nats
      value: 0.0022089662032662542
      args:
        units: nats
        higher_is_better: false
        derived: true
        derivation: candidate_minus_same_lane_floor
        floor_value: 0.011505922619330299
        floor_measurement_id: measurement--glm53.bf16-stream-floor.brandonmusic-final25
        floor_lane: streaming
        caveat: KL is not additive; this is an estimate, valid only against a floor measured on the same
          lane, same panel and same reference.
    - type: top1_agreement
      name: Top-1 agreement with reference
      value: 0.9656277479237909
      args:
        units: fraction
        higher_is_better: true
        lane: streaming
    source:
      name: quant-fidelity-registry
      url: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/measurements?q=measurement--glm53.k6-6bpw-stream.brandonmusic-final25
  - task:
      type: text-generation
      name: Distribution fidelity (KL divergence vs BF16 reference)
    dataset:
      type: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
      name: brandonmusic panel v1, calibration-clean subset -- 17 of 25 final windows -- clean17 subset
      config: final25-clean17
      split: streaming
      revision: 95f4fdd94bf29989db2e0d1054e4931f55edb6aa
      args:
        panel_id: panel--glm53.brandonmusic.final25-clean17
        panel_token_sha256: ecfce5997ab9106cb544d41f26c3c78730d246f3d3bf3ed8fe6f20bbd0847d83
        contexts: 17
        scored_positions: 34799
        context_length: 2048
        tokenizer: zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43
        vocab_size: 154880
        reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
        scope_name: clean17
        covers_full_panel: false
        scope_selection_sha256: 180678c464c8c325a4675a9fd5c54247e99624f0fc911580805e98fd960f79fe
    metrics:
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), nats
      value: 0.0116759926937349
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
        accumulation_dtype: float64
        logits_dtype: fp32
        head_policy: native_head
        stack_relation: same_stack
        lane: streaming
        run_count: 2
        determinism: bitwise_identical_across_runs
        measurement_id: measurement--glm53.k6-6bpw-stream.brandonmusic-final25.clean17
        comparability_key: cmp--2b9c401d13806d7e
    source:
      name: quant-fidelity-registry
      url: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/measurements?q=measurement--glm53.k6-6bpw-stream.brandonmusic-final25.clean17
  - task:
      type: text-generation
      name: Distribution fidelity (KL divergence vs BF16 reference)
    dataset:
      type: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
      name: brandonmusic GLM-5.3-Flash sealed qualification panel v1 -- 25 final windows -- panel25 subset
      config: final25-panel25
      split: sealed-ep8
      revision: 95f4fdd94bf29989db2e0d1054e4931f55edb6aa
      args:
        panel_id: panel--glm53.brandonmusic.final25
        panel_token_sha256: 6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff
        contexts: 25
        scored_positions: 51175
        context_length: 2048
        tokenizer: zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43
        vocab_size: 154880
        reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25
        scope_name: panel25
        covers_full_panel: true
        scope_selection_sha256: 180678c464c8c325a4675a9fd5c54247e99624f0fc911580805e98fd960f79fe
    metrics:
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), nats
      value: 0.013723384665701147
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
        accumulation_dtype: float64
        logits_dtype: fp32
        head_policy: native_head
        stack_relation: same_stack
        lane: sealed-ep8
        run_count: 5
        population_stddev_of_run_means: 0.0
        determinism: bitwise_identical_across_runs
        measurement_id: measurement--glm53.k6-6bpw.brandonmusic-final25
        comparability_key: cmp--202b717f3219c414
    source:
      name: quant-fidelity-registry
      url: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/measurements?q=measurement--glm53.k6-6bpw.brandonmusic-final25
  - task:
      type: text-generation
      name: Distribution fidelity (KL divergence vs BF16 reference)
    dataset:
      type: brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
      name: brandonmusic panel v1, calibration-clean subset -- 17 of 25 final windows -- clean17 subset
      config: final25-clean17
      split: sealed-ep8
      revision: 95f4fdd94bf29989db2e0d1054e4931f55edb6aa
      args:
        panel_id: panel--glm53.brandonmusic.final25-clean17
        panel_token_sha256: ecfce5997ab9106cb544d41f26c3c78730d246f3d3bf3ed8fe6f20bbd0847d83
        contexts: 17
        scored_positions: 34799
        context_length: 2048
        tokenizer: zai-org/GLM-5.3-Flash-BF16@a6c167b62691b2bac901344b65cb651a70f53e43
        vocab_size: 154880
        reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
        scope_name: clean17
        covers_full_panel: false
        scope_selection_sha256: 180678c464c8c325a4675a9fd5c54247e99624f0fc911580805e98fd960f79fe
    metrics:
    - type: kl_divergence
      name: Mean tokenwise KLD (reference || candidate), nats
      value: 0.0116772863686953
      args:
        units: nats
        higher_is_better: false
        direction: reference_to_candidate
        estimator: full_vocabulary_fp64
        accumulation_dtype: float64
        logits_dtype: fp32
        head_policy: native_head
        stack_relation: same_stack
        lane: sealed-ep8
        run_count: 5
        determinism: bitwise_identical_across_runs
        measurement_id: measurement--glm53.k6-6bpw.brandonmusic-final25.clean17
        comparability_key: cmp--2b9c401d13806d7e
    source:
      name: quant-fidelity-registry
      url: https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/measurements?q=measurement--glm53.k6-6bpw.brandonmusic-final25.clean17
x_fidelity:
  spec: https://github.com/malaiwah/quant-fidelity-suite/blob/main/docs/FIDELITY-DATASET-SPEC.md
  spec_version: fidelity-provenance/v1
  role: quant
  reference_model: zai-org/GLM-5.3-Flash-BF16
  reference_revision: a6c167b62691b2bac901344b65cb651a70f53e43
  fidelity_dataset: null
  registry:
    dataset: malaiwah/quant-fidelity-registry
    schema_version: quant-fidelity-registry/v1
    artifact_id: artifact--malaiwah.glm-5.3-flash-tr3-6bpw
    measurement_ids:
    - measurement--glm53.k6-6bpw-stream.brandonmusic-final25
    - measurement--glm53.k6-6bpw-stream.brandonmusic-final25.clean17
    - measurement--glm53.k6-6bpw.brandonmusic-final25
    - measurement--glm53.k6-6bpw.brandonmusic-final25.clean17
    snapshot:
      data_sha256:
        models: c2a33da659332578fcd803473a13fae811abee8990c5d05807ca48d884154ea8
        artifacts: 7c8e848f8dd41727327f81337057eef6ce656d23564a09e39b7fe64405d2e551
        panels: 38a33529d34b400ec92fd0b725de562324e486b9a6c59aaa79e251235e09b603
        references: ca593279ddaab6c3338bfb214db7c95ec57d7607cb231b61c14020d4b18bebe8
        pipelines: 4c143473c80b75d1cd9cec6d0e1a5985dc68a2c1e5df20b52dbb6a6183e5e694
        measurements: 259d490cebd39571e4d1ecfefe7aa9f490a390f8cdf236f16ee901c51fb6e6f2
  scope_digest: attn.o=native:bf16@16|attn.other=native:mixed|attn.qkv=native:bf16@16|embed_tokens=native:bf16@16|lm_head=native:bf16@16|mlp.down=native:bf16@16|mlp.gate=native:bf16@16|mlp.up=native:bf16@16|moe.experts=quantized:exl3-mcg@6|moe.router=native:fp32@32|moe.shared_expert=native:bf16@16|mtp=quantized:exl3-mcg@6|norm=native:bf16@16|other=native:bf16@16|head=native|kv=bf16
  head:
    policy: native
    quantized: false
    bits: 16
    lm_head_tensor_content_sha256: null
    lm_head_file_sha256: 47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0
    final_norm_tensor_content_sha256: null
    final_norm_file_sha256: c228a123dee3062c3ad0129094e9d98a264e33087ee88d79c8d6c5a6e60f2fed
    equality_receipt: https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/reports/head-equality-fp8.json
    replay_permitted: false
    note: lm_head_tensor_content_sha256 is null because no head-identity receipt has been published for
      this artifact yet. A comparator MUST refuse to replay this artifact's hidden states through any
      other artifact's head until it is filled in (FIDELITY-DATASET-SPEC HEAD-4). The published receipts
      record only the FILE digest, which is a container digest and never an identity (O-6).
  measurements:
  - id: measurement--glm53.k6-6bpw-stream.brandonmusic-final25
    lane: streaming
    status: published
    comparability_key: cmp--202b717f3219c414
    panel_id: panel--glm53.brandonmusic.final25
    reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25
    pipeline_id: pipeline--malaiwah.glm53-stream-packed-kld
    metric_name: mean_of_run_means_tokenwise_kld
    value: 0.013714888822596553
    units: nats
    direction: reference_to_candidate
    run_count: 2
    determinism:
      evidence_kind: tokenwise_kld_sha256
      distinct_evidence_hash_count: 1
      identical_across_runs: true
      evidence_hashes:
      - 9657ede36b9f4b09a2c74916239c6d9a3baebce5f3fa64af7af388b0686aa284
    floor_measurement_id: measurement--glm53.bf16-stream-floor.brandonmusic-final25
    excess_over_control: 0.0022089662032662542
    excess_over_control_withheld: null
    measured_by: self-measured
    disclosures:
    - reduced_run_count
    - non_sealed_lane
    - harness_unrecorded
  - id: measurement--glm53.k6-6bpw-stream.brandonmusic-final25.clean17
    lane: streaming
    status: published
    comparability_key: cmp--2b9c401d13806d7e
    panel_id: panel--glm53.brandonmusic.final25-clean17
    reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
    pipeline_id: pipeline--malaiwah.glm53-stream-packed-kld
    metric_name: mean_of_run_means_tokenwise_kld
    value: 0.0116759926937349
    units: nats
    direction: reference_to_candidate
    run_count: 2
    determinism:
      evidence_kind: tokenwise_kld_sha256
      distinct_evidence_hash_count: 1
      identical_across_runs: true
      evidence_hashes:
      - 9657ede36b9f4b09a2c74916239c6d9a3baebce5f3fa64af7af388b0686aa284
    floor_measurement_id: null
    excess_over_control: null
    excess_over_control_withheld: null
    measured_by: self-measured
    disclosures:
    - reduced_run_count
    - non_sealed_lane
    - subset_of_panel
    - calibration_panel_overlap
  - id: measurement--glm53.k6-6bpw.brandonmusic-final25
    lane: sealed-ep8
    status: published
    comparability_key: cmp--202b717f3219c414
    panel_id: panel--glm53.brandonmusic.final25
    reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25
    pipeline_id: pipeline--malaiwah.glm53-packed-kld
    metric_name: mean_of_run_means_tokenwise_kld
    value: 0.013723384665701147
    units: nats
    direction: reference_to_candidate
    run_count: 5
    determinism:
      evidence_kind: tokenwise_kld_sha256
      distinct_evidence_hash_count: 1
      identical_across_runs: true
      evidence_hashes:
      - 52e35723dacd0314acb85bcee86d2faefd5c12ff9d82c6e026e05d35ee15db4b
    floor_measurement_id: null
    excess_over_control: null
    excess_over_control_withheld: null
    measured_by: self-measured
    disclosures:
    - harness_unrecorded
  - id: measurement--glm53.k6-6bpw.brandonmusic-final25.clean17
    lane: sealed-ep8
    status: published
    comparability_key: cmp--2b9c401d13806d7e
    panel_id: panel--glm53.brandonmusic.final25-clean17
    reference_id: reference--brandonmusic.glm53-bf16-fp32-logits.final25-clean17
    pipeline_id: pipeline--malaiwah.glm53-packed-kld
    metric_name: mean_of_run_means_tokenwise_kld
    value: 0.0116772863686953
    units: nats
    direction: reference_to_candidate
    run_count: 5
    determinism:
      evidence_kind: tokenwise_kld_sha256
      distinct_evidence_hash_count: 1
      identical_across_runs: true
      evidence_hashes:
      - 52e35723dacd0314acb85bcee86d2faefd5c12ff9d82c6e026e05d35ee15db4b
    floor_measurement_id: null
    excess_over_control: null
    excess_over_control_withheld: null
    measured_by: self-measured
    disclosures:
    - subset_of_panel
    - calibration_panel_overlap
---

# GLM-5.3-Flash-TR3-6bpw (K6)

**The first 6-bit (K6) TR3/MCG trellis quantization of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** —
321B-total / A18B MoE, `glm5_next` hybrid architecture. Routed experts and the
MTP layer quantized at K6 (96-word trellis, MCG `0xCBAC1FED`); everything else
(KDA linear-attention layers, DSA indexer, hyper-connections, routers, norms,
embeddings, lm_head) **bit-exact native BF16**. 253.5 GB — 77% of the official
FP8's footprint.

## Quality — SEALED five-cold-run qualification

> ### ⚠ Scope disclosure — this number is a **panel25** number
>
> Added 2026-08-29. **Nothing here is a correction: 0.013723 is and remains the
> correct mean over the full 25-window panel.** What changed is that the panel is
> now known to contain calibration-adjacent windows, so the scope has to travel
> with the number.
>
> [brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
> ran a **13-gram overlap scan** of his sealed panel against its own
> calibration-role windows and found that the whole `axis4_reasoning` domain
> shares **37–39 %** of its 13-grams with calibration material — despite the
> panel being clean at the *document-hash* level. Document-hash dedup is not
> enough. He excluded that domain and scored his primary numbers on the **17
> windows that survive**. The finding, the scan and the 0.05 threshold are his.
>
> Every malaiwah number on this panel used all 25 windows, so every one of them
> carries the same contamination. Recomputed on his clean scope, from our own
> published per-window arrays (no GPU, no re-measurement — this is arithmetic on
> data already published):
>
> | | panel25 (published) | clean17 (his scope) | move |
> |---|---:|---:|---:|
> | **K6 sealed** | 0.013723 | **0.011677** | −14.91 % |
> | K6 streaming | 0.013715 | 0.011676 | −14.87 % |
> | K8 | 0.012384 | 0.010829 | −12.55 % |
> | official FP8 | 0.020615 | 0.018665 | −9.46 % |
> | BF16 floor (cross-stack) | 0.012712 | 0.010648 | −16.24 % |
> | brandonmusic 4bpw | 0.024555 | 0.024949 | **+1.61 %** |
>
> **The comparisons hold as descriptions of this panel, and one of them
> strengthens.** K6 beats the official FP8 on **17 of 17** clean windows, and
> the margin *widens*: 1.50× on panel25 becomes **1.60×** on clean17. The
> K8-over-K6 result survives but weakens — the paired BCa interval still
> excludes zero on clean17, with its lower bound falling from +0.000695 to
> +0.000153.
>
> **Statistical correction (2026-08-31, peer review P1-15).** This panel's 25
> windows derive from only **four source documents** (clean17: three), so
> window-level sign tests and intervals describe these exact windows rather
> than independent evidence. At the document level the K8-over-K6 contrast
> reads: all four (three) document means favour K8, exact sign test
> **p = 0.125** (panel25) / **0.25** (clean17) — the previously printed
> window-level p = 0.0041 / 0.049 are withdrawn as inferential statements. The
> ordering survives on this panel; a population claim awaits a panel with many
> independent documents per domain. **We will not restate "K8 is better than
> K6" without naming the scope — or beyond this panel.**
>
> **Do not difference a panel25 number against a clean17 one.** They are answers
> to different questions. Our registry enforces this structurally: `clean17` is
> its own derived panel with its own comparability key.
>
> The **excess-over-control** table below (formerly "quantization-attributable";
> renamed 2026-08-31, P1-05) cannot be recomputed on the clean scope — its floor
> is the *streaming* BF16 floor, whose receipt is scalar-only (run means and a
> tokenwise digest, no per-window array), and substituting the cross-stack floor
> would be the cross-lane subtraction our registry refuses. It stands as a
> panel25 number.
>
> Full recompute, with per-domain tables, paired intervals and provenance:
> [`reports/clean-scope-recompute.json`](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/clean-scope-recompute.json).
> Working: [PROTOCOL-ALIGNMENT.md](https://github.com/malaiwah/quant-fidelity-suite/blob/main/docs/PROTOCOL-ALIGNMENT.md) §4.
>
> **One protocol note, not a correction.** His protocol masks the 24 padded
> `lm_head` columns before the log-softmax; ours never has. Measured on his real
> teacher window, the padded columns hold ~1.6e-8 of the probability mass, and
> because this quant shares the teacher's native BF16 head the effect collapses
> to `KLD × mass` — **1.0e-10 nats**, moving the value above at its *9th*
> significant figure. For scale, our own sealed-vs-streaming bridge is 8.5e-6 and
> the window-clustered SE on this panel is 3.19e-3. No correction and no bias
> disclosure is warranted; we are adopting masking anyway. Script and receipts:
> [`bin/padded_column_study.py`](https://github.com/malaiwah/quant-fidelity-suite/blob/main/bin/padded_column_study.py).


**Mean KLD(teacher ‖ K6) = 0.013723 nats over the full sealed panel (25
windows, 51,175 positions per run) — five cold runs, bitwise identical
(population stddev exactly 0.0)**, the same determinism property as
brandonmusic's protocol. Quality gate (< 0.06): **passed**. Receipts:
[`receipts/k6-five-run-kld.json`](receipts/k6-five-run-kld.json),
[`receipts/k6-packed-kld.json`](receipts/k6-packed-kld.json) (evidence-artifact
hashes included).

| Model | Mean KLD (nats) | Size | Scope |
|---|---:|---:|---|
| **This K6 (sealed)** | **0.013723** | 254 GB | full panel × 5 bitwise-identical runs |
| This K6, streaming lane | 0.013715 | 254 GB | full panel × 2 bitwise-identical runs; −8.5e-6 vs sealed ([receipt](receipts/stream-k6-kld.json)) |
| [**K8 sibling**](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-8bpw) | **0.012384** | 331 GB | full panel × 2 bitwise-identical runs, streaming lane |
| Official FP8 (full panel) | 0.020615 | 328 GB | cross-stack, [receipt](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/blob/main/reports/fp8-on-brandon-panel.json) |
| brandonmusic 4bpw | 0.024555 | 176 GB | full panel, his stack |
| Official FP8 (his stack, v44) | 0.024629 | 328 GB | 1 window × 5 runs |
| NVFP4 (his stack, v44) | 0.060535 | ~180 GB | 1 window × 5 runs |

**K6 delivers 1.5× lower divergence than the official FP8 release at 77 % of
its bytes** (1.8× vs the 4bpw, 4.4× vs NVFP4). Panel-wide top-1: **96.56 %**
(full 25-window panel, streaming lane). Serving is now independently live-qualified
on 4× RTX PRO 6000 Blackwell (SM120) with the digest-pinned turnkey image and
profile below; the earlier SM90 qualification limitation no longer applies to the
serving claim.

### Excess over control (the floor removed)

(Called "quantization-attributable error" before 2026-08-31; renamed per
peer-review P1-05 — the difference estimates excess divergence over the
same-lane unquantized control and is **not** a causal attribution.)

Scoring the **unquantized BF16 weights** against this teacher on this panel
already costs **0.011506 nats** — the price of the comparison itself (teacher
captured on a different runtime; bf16 addition is not associative across
differing expert-combine orders). Two cold runs, identical means. Removing it:

| | panel KLD | excess over control |
|---|---:|---:|
| BF16 (floor) | 0.011506 | — |
| K8 (331 GB) | 0.012384 | **0.000878** |
| K6 (254 GB) | 0.013715 | **0.002209** |

K8's residual is smaller than K6's — 0.000878 vs 0.002209 nats — where the raw
means sit only 1.11x apart, because the floor is common to both rows and
dominates both. **The previously published ratio of the two residuals
("2.52x") is withdrawn**: a ratio of small residuals magnifies control error
and carried no uncertainty. Read the residuals beside the raw values, with the
floor named. Method, receipts and the ways this subtraction can be misused:
[BF16-FLOOR.md](https://github.com/malaiwah/quant-fidelity-suite/blob/main/engines/BF16-FLOOR.md).

## What this is (and is not)

- **Codec:** EXL3-format TR3/MCG trellis (turboderp's
  [exllamav3](https://github.com/turboderp-org/exllamav3) kernels @ `c5d9c657`
  did the encoding math), through
  [brandonmusic's GLM-5.3 quantization pipeline](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw)
  with a small disclosed patch series.
- **Serving runtime:** use
  [`malaiwah/glm52-exl3-vast`](https://github.com/malaiwah/glm52-exl3-vast)
  with `MODEL_PROFILE=glm53-k6`. The image pins the qualified Glm5Next vLLM,
  B12X, EXL3, CUDA, and 21-file fail-closed runtime overlay as one contract.
- **Not stock exllamav3/TabbyAPI or stock upstream vLLM:** those stacks do not
  carry this complete `glm5_next` + TR3/MCG K6 serving path.
- **Topology-neutral checkpoint:** canonical unsharded tensors; TP layout is a
  load-time decision. The qualified deployment is exactly TP4/DCP4 on four
  96 GiB RTX PRO 6000 Blackwell GPUs.
- **Measured memory:** the packaged image loads 58.96 GiB of model tensors per
  rank. At GMU 0.93, final profiling reported 63.74–63.78 GiB weights +
  non-torch, 3.02 GiB peak activations, 0.45–0.46 GiB CUDA graphs, and
  21.52–21.56 GiB KV per GPU.
- **Shared down-`suh` topology:** all 288 experts per layer share the
  down-projection input sign vector (measured fidelity-free: worst-layer
  −2×10⁻⁶ relative output error). A one-transform-per-layer grouped-GEMM hoist
  remains an optimization opportunity; it is not claimed by this release.

## Provenance & disclosed deviations

Full receipts ship in this repo and in the
[fidelity suite](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1).
Key pins: BF16 source `zai-org/GLM-5.3-Flash-BF16` (weights == `a6c167b6`),
calibration = brandonmusic's published EP4 captures (sealed inventory
`f56e9d62…` adopted verbatim), same transform-seed discipline as the K8
sibling (parts-bin assembly compatible). Disclosed deviations from his sealed
K4 campaign: encoded on 4×H200 SM90 (his: 4×B200 SM100; fat `9.0;10.0`
extension build), **verified-equivalent** R10 codec (we
encoded with a reconstruction while his numeric core was unpublished; he has
since published the sealed closure, and a head-to-head on identical real
inputs came back **120/120 encodes byte-identical — 624 MiB of packed trellis,
0 differing bytes, decoded-weight delta exactly 0.0**. His published core
admits only K3/K4/K5, so K6/K8 are a *declared rate extension*, not a
substitution; driving his sealed primitives past that admission constant
reproduces our bytes exactly. Fidelity impact is identically zero. Evidence:
[closure-comparison.json](https://github.com/malaiwah/quant-fidelity-suite/blob/main/engines/fallback/closure-comparison.json),
[issue #1](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/issues/1)), K4-KL gate satisfied via a disclosed bridge
document carrying his real published K4 receipt hashes, qualification at EP8
(his reader default EP4). The five-run qualification receipts land here when
sealed.

## Family

| | K4 | **K6 (this)** | K8 |
|---|---|---|---|
| Repo | [brandonmusic's 4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | here | [malaiwah/GLM-5.3-Flash-TR3-8bpw](https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-8bpw) |
| Size | 176 GB | 254 GB | 331 GB |
| Mean KLD (same panel) | 0.024555 | **0.013723** | **0.012384** |

Same pipeline, calibration, and panel across the family. A payload parts-bin
dataset (K6 + K8 per-choice payloads, same seed) is published:
[GLM-5.3-Flash-TR3-partsbin-v1](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-TR3-partsbin-v1) — multi-precision K6K8 mixes become offline assembly, no GPU re-encode.

## Lineage on the Hub

Z.ai published two sibling roots for this model and neither declares the other:
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (the
**FP8** release, where most traffic lands) and
[`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
(the **BF16** weights). This quant declares BF16 as its `base_model` because
that is what it was actually quantized from — the FP8 release is a *sibling*
quantization of the same model, not our source, and it is the baseline we
measure against rather than build on. Quants that list FP8 as their base were
genuinely made from the FP8 weights; the trees differ for real reasons.

Related work on the same model, all measured on one panel in the
[quant-fidelity registry](https://huggingface.co/datasets/malaiwah/quant-fidelity-registry):
[brandonmusic 4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw),
[0xSero Dione Q4](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Q4),
[orcarouter MLX](https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX).
Collection: [GLM-5.3-Flash — measured quants & fidelity](https://huggingface.co/collections/malaiwah/glm-53-flash-measured-quants-and-fidelity-6a91f253e7107818359f37c8).

## Credits

Base model by [Z.ai](https://huggingface.co/zai-org). Quantization pipeline,
calibration captures, and teacher panel by
[brandonmusic](https://huggingface.co/brandonmusic) (co-credited — see the
[collaboration thread](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1)).
Trellis codec and kernels by [turboderp](https://github.com/turboderp-org/exllamav3).
Campaign log, tools, and every patch:
[malaiwah/quant-fidelity-suite](https://github.com/malaiwah/quant-fidelity-suite).

## Serving — live-qualified turnkey profile

### Qualification result

The shipped profile is **`glm53-k6`**. It was booted from the packaged appliance
on 4× RTX PRO 6000 Blackwell 96 GiB and passed the appliance's arithmetic,
factual, instruction-following, strict structured-output, and tokenizer-exact
32K retrieval gates. Runtime fingerprint:
`vllm-0.1.dev20111+g7f1e92bec.d20260827-tp4-95ae22a9`.

- Appliance source commit: [`a0d05f76994cf44f3667c0d2910d3b0e4d305d23`](https://github.com/malaiwah/glm52-exl3-vast/commit/a0d05f76994cf44f3667c0d2910d3b0e4d305d23)
- Qualified parent: `verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
- Published appliance: `ghcr.io/malaiwah/glm52-exl3-vast@sha256:5a0d4b370e9f6a2ef85fa8b8c213492122554b34ba18d630a3a78130758914cf`
- Shape: TP4 / DCP4 A2A, B12X sparse MLA, Triton MoE, EXL3 K6,
  calibrated NVFP4-DS MLA KV, MTP off, batch 3,072, C8, GMU 0.93
- Request limit: **458,752 tokens**; text-only qualification scope

The cap is a correctness boundary, not a memory-capacity guess. Two independent
448K trials produced tokenizer-exact 449,461- and 449,462-token documents and
retrieved 3/3 facts at 15%, 55%, and 90% depth. A 480K trial exhausted both
2,048- and 4,096-token answer budgets. A concurrent 505K stress trial caused
persistent degenerate follow-on output until restart. Therefore this release
**does not claim a 500K usable request**. The 458,752-token envelope leaves about
9K tokens beyond the longest passing document for template, query, and output.

| document tokens | independent seed | retrieval | elapsed |
|---:|---:|---:|---:|
| 384,612 | 20260831 | 3/3 | 81.058 s |
| 449,462 | 20260901 | 3/3 | 95.839 s |
| 449,461 | 20260903 | 3/3 | 94.578 s |

The final appliance boot auto-profiled **20,043,933 logical KV tokens**
(43.69× one maximum request) and 21.52–21.56 GiB KV per GPU. That large pool is
concurrency capacity; it does not override the single-request correctness gate.

### Measured throughput

Unique-prefix prefill, one request, no prefix reuse:

| prompt | client-observed tok/s | server-accounted tok/s |
|---:|---:|---:|
| 8K | 2,983 | — |
| 32K | 4,322 | 5,238 |
| 64K | 4,637 | 5,326 |
| 128K | 4,907 | 5,326 |

Aggregate target-only decode (`MTP_TOKENS=0`):

| input context | C1 tok/s | C4 tok/s | C8 tok/s |
|---:|---:|---:|---:|
| 0 | 75.15 | 241.31 | 397.19 |
| 32K | 69.51 | 234.64 | 349.38 |
| 128K | 64.72 | 223.01 | 323.64 |

No preemption was observed in the qualification matrix. These are measurements
from one 4× RTX PRO 6000 Blackwell PCIe host, not guarantees for other topology,
clock, thermal, driver, storage, or request mixes.

### Why K6 is the production default

K8 improves raw panel KLD from 0.013723 to 0.012384 nats (absolute 0.001339)
but grows from 254 GB to 331 GB: about 77 GB / 30% more checkpoint bytes. Its
live-qualified eager profile uses 78.94–78.97 GiB/GPU for weights plus non-torch
allocations and leaves 7.10–7.14 GiB/GPU for KV. K6 leaves 21.52–21.56 GiB/GPU
for KV and is 5.3–7.3× faster at short context, or 8.3–24.4× faster when the
measured 32K/128K prefill cost is included. K8 is the qualified fidelity-first
alternative; K6 remains the production default for its quality/bytes/throughput
balance on 4×96 GiB.

### Docker Compose

Prerequisites: Linux x86-64, four visible RTX PRO 6000 Blackwell GPUs, NVIDIA
driver ≥ 590.48.01 / CUDA 13.2 compatibility, the NVIDIA Container Toolkit,
and roughly 300 GiB free persistent storage for the checkpoint plus caches.
PCIe P2P on this card family requires NVIDIA's open kernel modules; see the
[RTX 6000 Pro multi-GPU notes](https://github.com/brandonmmusic-max/rtx6kpro).

```yaml
name: glm53-k6
services:
  api:
    image: ghcr.io/malaiwah/glm52-exl3-vast@sha256:5a0d4b370e9f6a2ef85fa8b8c213492122554b34ba18d630a3a78130758914cf
    pull_policy: always
    restart: unless-stopped
    network_mode: host
    ipc: host
    shm_size: 32gb
    stop_grace_period: 2m
    ulimits:
      memlock:
        soft: -1
        hard: -1
    environment:
      MODEL_PROFILE: glm53-k6
      AUTH: key
      VLLM_API_KEY: ${VLLM_API_KEY:?set VLLM_API_KEY to a long random secret}
      HF_TOKEN: ${HF_TOKEN:-}
      SSH_ENABLED: "0"
      SOUL_ENABLED: "0"
      VERIFY_HEALTH_TIMEOUT_S: "3600"
    volumes:
      - /srv/glm53-turnkey:/workspace
      - /srv/glm53-cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
```

```bash
sudo mkdir -p /srv/glm53-turnkey /srv/glm53-cache
export VLLM_API_KEY="$(openssl rand -hex 32)"
docker compose up -d
docker compose logs -f
```

First boot downloads about 237 GiB and can take substantial time. The container
is ready only after the log reports `>>> Verified: serving; long-context
retrieval verified`. API: `http://HOST:8000/v1`; dashboard:
`http://HOST:1111`. The served model name is `GLM-5.3-Flash-K6`.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5.3-Flash-K6","messages":[{"role":"user","content":"Reply with exactly READY"}],"max_tokens":256}'
```

Do not replace only the checkpoint path in another vLLM command. The profile,
parent digest, runtime overlays, quantization, attention backend, DCP topology,
KV calibration, scheduler, and graph widths are one qualified contract.
