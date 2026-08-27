## Operator design inputs (2026-08-27)
1. K6K8 mixed precision: IMPLEMENT support in the pinned pipeline (operator
   directive) — prior art: operator's Qwen3.8-27B multi-K recipe + GLM-5.2
   mixed lineage (NVFP4-TR3-Hybrid, willfalco, madeby561). K6-uniform ships
   first and never blocks on the K6K8 patch set.
2. Shared suh/svh across routed experts (per layer): evaluate for performance
   (one activation Hadamard transform per layer vs per expert on the
   grouped_gemm path) IF the KLD cost is ~nil. Decide via layerwise A/B on the
   captured activations dataset (quantize a few representative expert layers
   shared vs per-expert, replay block inputs, compare output divergence) BEFORE
   committing full conversions. Operator research link PENDING — fold in when
   provided. Check pinned exllamav3 + materializer/storage-ABI support for
   shared sign vectors before assuming feasibility.
3. TOPOLOGY-NEUTRAL QUANT (operator directive 2026-08-27): the published
   checkpoint must not bake in a TP/EP topology. Publish canonical unsharded
   tensors (his format's output_root/native side, not a TP-packed layout);
   any packing for a specific TP happens at load/deploy time. Design must map
   which materializer steps depend on target_tensor_parallel /
   qualified_tp_sizes and keep those OUT of the published artifact (or publish
   canonical + validate load on TP2 AND TP4). Shared-suh (item 2) is
   compatible: the per-layer input transform is replicated across ranks.
4. Shared-suh finding (from operator's qwen38 P3.1 + his 4bpw index audit):
   Brandon's 4bpw is fully per-expert/per-projection suh+svh (288x3 per MoE
   layer, no sharing/merging). P3.1-style shared input-suh + per-component svh
   would be a format IMPROVEMENT but requires runtime changes to exploit →
   K6-uniform ships in HIS exact format first; shared-suh is a gated
   experiment (layerwise A/B on activations dataset) after K6K8.
5. PROVEN ENV RECIPE (L4 smoke rounds 1-5, 2026-08-27): python3.12 (deadsnakes
   on 22.04 VM) + torch==2.11.0 --index-url .../whl/cu130 (matches VM CUDA 13.0
   toolkit AND Brandon's recorded stack) + transformers==5.16.1 + flash-attn
   2.8.3 via cu13torch2.10 cxx11abiTRUE cp312 wheel (imports fine on torch
   2.11; NO torch2.11 wheel exists; source build fails) + kbnf +
   formatron==0.5.0 + pydantic==2.5.3 (0.5.0 breaks on newer pydantic:
   pydantic.typing.Type; 0.4.11 breaks differently) + accelerate safetensors
   ninja packaging rich tokenizers pillow + xformers optional-warning-only +
   pip install --no-deps git+exllamav3@c5d9c657. Verified: `import exllamav3;
   from exllamav3 import ext` GREEN on SM89/CUDA13. Fixture forward WORKS
   stock on torch 2.11 (scatter int32 bug is torch<=2.6-era only).
   TORCH_CUDA_ARCH_LIST: 8.9 L4 / 9.0 H100+H200.
6. K6 IS A BUILT-IN PROFILE (miner receipt): glm53_direct_k4.py L53
   SUPPORTED_BITS=(4,6); k6 recipe ids + contract/materialization schemas +
   publication profile (bits=6, tp=4) all pre-exist. K6-uniform requires NO
   code edits. K8 needs codec adapter admission (L175) + 128-word trellis
   verification; runtime derives per-module K from trellis shape (L501) —
   serving structurally mixed-ready behind ~6 surgical validators.
