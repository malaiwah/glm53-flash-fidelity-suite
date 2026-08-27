# exllamav3 glm5_next port blueprint (GLM-5.3-Flash K6)

> Produced 2026-08-27 by a 7-agent design workflow (blueprint -> draft -> parity harness -> adversarial review) against exllamav3 v1.4.4.

## Summary

PORT BLUEPRINT complete. Core finding: exllamav3 v1.4.4 already contains ~80% of what GLM-5.3-Flash needs — dsv4's mHC HyperConnection torch path is numerically identical to vLLM's mhc_pre/post_torch (verified line-by-line, incl. hardcoded post_mult 2.0 and Sinkhorn eps loop), the checkpoint's hc_attn_fn/_base/_scale naming matches dsv4's "{key}_fn" convention exactly, glm_moe_dsa provides the MLA+indexer+sigmoid-noaux_tc MoE skeleton, and GDN provides the conv-state/recurrent-cache machinery. New code: one KimiDeltaAttention module (KDA ≠ GDN: 6 split in-projections, per-KEY-CHANNEL safe gate g=-5·sigmoid(exp(A_log)·(f_b(f_a(x))+dt_bias)) vs GDN's scalar-per-head decay, sigmoid output gate vs silu, calls fla.ops.kda.chunk_kda/fused_recurrent_kda which exist upstream with initial_state support), a kpool-compressed indexer mode (cache ki+gate_score in a 256-wide idx plane, softmax-pool by 4, relu scoring, +tail), NoPE guards for rope_dim=0 through the MLA triton kernels, a 20-line mean ContractStreams (GLM has no learned hc_head), and a sigmoid option in GatedRMSNorm (fla upstream supports it; exl3's is silu-only). Keep BF16/fp16-unquantized: all mHC params, A_log/dt_bias/convs, b/f_a/f_b/g_a/g_b (~234 MB total), kv_b_proj, all indexer tensors except wq_b, norms, embed, routers. Validation: per-layer torch-oracle parity + decode-equals-prefill + selection-Jaccard on one RTX PRO 6000, chained-layer end-to-end (KL ≤ 5e-3, top-1 ≥ 99.5%). ~72 expert-hours + 12–20 h GPU conversion. 3 hardest: KDA state lifecycle, NoPE zero-width paths, kpool selection parity.

## Detail

PORT BLUEPRINT — GLM-5.3-Flash (glm5_next) on exllamav3 v1.4.4
================================================================

Verified inputs: arch_string is "Glm5NextForConditionalGeneration" (top-level model_type glm5_next, text_config model_type glm5_next_text); all params under text_config; checkpoint prefix model.language_model.layers.N; lm_head.weight top-level. Conversion input is the pre-dequantized BF16 checkpoint (~642 GB); the port needs no fp8 load path. indexer_types is ["full"]×45 (no "shared" layers in this model). MTP layer 45 (DSA + MoE + eh_proj/enorm/hnorm/shared_head.norm, NO hc tensors) and model.visual.* (347 tensors) are skipped in v1 by simply not registering "mtp"/"vision" components — unreferenced tensors fall out of the compiled output.

Local evidence used (paths for the coding agent):
- exl3 framework cache: /private/tmp/claude-501/-Users-mbelleau-Projects-GLM/c1546622-1c41-4561-ba68-92b6b9cb9811/scratchpad/exl3/ (and full clone with csrc at .../scratchpad/exllamav3-src)
- reference math: .../scratchpad/refs/ (kda.py, fla_kda.py, mhc_torch.py, mlx_language.py, hf_config.json, st_index.json, ...)

KEY DISCOVERY (changes the plan vs. what the studies assumed): exllamav3 already ships full mHC support from DeepSeek-V4 — modules/hyperconnections.py (ExpandStreams / HyperConnection / HyperHead + fused ext.hc_mix/hc_apply CUDA kernels) and TransformerBlock already accepts attn_hc/mlp_hc and implements exactly GLM's site order (hc.mix → collapsed → weighted attn_norm → attn → hc.apply_; same for mlp). The torch fallback in HyperConnection.mix() is line-for-line the vLLM mhc_pre_torch math: sigmoid(pre·s0+b)+hc_eps; 2.0·sigmoid(post·s1+b) (post_mult 2.0 hardcoded — equals GLM's default mhc_post_mult_value); softmax+eps, col-normalize, then (iters−1)× row/col Sinkhorn with eps 1e-6; weightless RMS norm folded before the fn GEMM (equivalent to vLLM's rsqrt-after-matmul since the GEMM is linear). apply_() is post⊗y + combᵀx, identical to mhc_post_torch's einsum. And the GLM checkpoint's tensor names (layers.N.hc_attn_fn/_base/_scale, hc_ffn_*) match dsv4's "{key}_fn" convention verbatim when key = f"{prefix}.layers.{i}.hc_attn". The only mHC deltas: (1) GLM's hc tensors are stored BF16 (dsv4's are fp32) → add allow_bf16=True to the three get_tensor calls in HyperConnection.load() and HyperHead.load() (they .float() immediately, so this is safe for both models); (2) GLM has NO learned hc_head — final contract is a plain mean over the 4 streams → new ~20-line ContractStreams module; (3) assert mhc_post_mult_value == 2.0 (config default; the ext kernel and torch path hardcode it).

------------------------------------------------------------
1. FILE-BY-FILE PLAN
------------------------------------------------------------

A. exllamav3/architecture/glm5_next.py — NEW (~350 lines). Model skeleton = glm_moe_dsa.py + qwen3_next.py tails + deepseek_v4.py HC wiring.
   - class Glm5NextConfig(Config): arch_string = "Glm5NextForConditionalGeneration"; __init__ calls super().__init__(directory, {"text": Glm5NextModel}, **kwargs). All reads under "text_config->..." (read_cfg supports nesting; vocab/eos already handle it). Read/assert:
     hidden_size 4096, num_hidden_layers 45, rms_norm_eps 1e-5, vocab_size 154880, tie_word_embeddings False;
     layer_types (list, 45 entries: "linear_attention"|"deepseek_sparse_attention"), mlp_layer_types + first_k_dense_replace 3;
     linear_attn_config dict: num_heads 64, head_dim 128, short_conv_kernel_size 4, gate_lower_bound −5.0 (assert present; the SAFE gate is the only variant this port implements);
     q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 256, assert qk_rope_head_dim == 0, v_head_dim 256, num_attention_heads 64, assert mla_use_nope == True; NO rope settings (rope_settings=None; do NOT call read_rope_settings_default — override_head_dim=0 would break it); sm_scale = 256**−0.5;
     index_n_heads 32, index_head_dim 128, index_topk 2048, assert index_kpool == 4, assert index_kpool_compress == True, assert index_kpool_always_select_tail == True, indexer_types (all "full" here — keep pass-through support);
     MoE (same asserts as glm_moe_dsa): n_shared_experts 1, n_routed_experts 288, num_experts_per_tok 8, assert scoring_func "sigmoid", assert topk_method "noaux_tc", assert norm_topk_prob True, n_group/topk_group ∈ (None,1), routed_scaling_factor 2.5, moe_intermediate_size 2048, intermediate_size 12288, swiglu_limit 10.0;
     assert mhc == True, hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6, assert_cfg(float,"mhc_post_mult_value",2.0,optional=True), assert_cfg(bool,"mhc_no_norm_weight",False,optional=True).
   - class Glm5NextModel(Model): config_class = Glm5NextConfig; key_prefix = "model.language_model" (glm4v_moe pattern). Module list:
     Embedding(key=f"{p}.embed_tokens") → ExpandStreams(hc_mult=4) → for idx in 0..44: TransformerBlock(key=f"{p}.layers.{idx}",
       attn_norm = RMSNorm(f"...{idx}.input_layernorm", eps 1e-5),
       attn = KimiDeltaAttention(key=f"...{idx}.self_attn", qmap="block.attn", out_dtype=torch.float) if layer_types[idx]=="linear_attention" else MLAttention(key=f"...{idx}.self_attn", qk_rope_head_dim=0, rope_settings=None, indexer_mode="kpool", index_kpool=4, ..., qmap="block.attn", out_dtype=torch.float, select_hq_bits=2),
       attn_hc = HyperConnection(key=f"...{idx}.hc_attn", hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5),
       mlp_norm = RMSNorm(f"...{idx}.post_attention_layernorm", eps 1e-5),
       mlp = GatedMLP(intermediate 12288, act_limit=10.0, select_hq_bits=1) for idx<3 else BlockSparseMLP(288 experts, top-8, key_e_score_bias="gate.e_score_correction_bias", router_type="dots", routed_scaling_factor=2.5, act_limit=10.0, shared_experts=GatedMLP(f"...mlp.shared_experts", intermediate 2048, act_limit=10.0, select_hq_bits=2), interm_dtype=torch.half, out_dtype=torch.float),
       mlp_hc = HyperConnection(f"...{idx}.hc_ffn", ...))
     → ContractStreams() → RMSNorm(f"{p}.norm", out_dtype=torch.half) → Linear("lm_head", qbits_key="head_bits", caps={"logits_output": True}).
     Tail: self.calibration_all_experts = True; caps: {"recurrent_states": True, "default_recurrent_checkpoint_interval": 2048, "linear_attn": True, "supports_tp": False}; self.recurrent_state_cls = GDNState (position bookkeeping is shape-agnostic).
     prepare_inputs: input_ids = prepare_for_attn(input_ids, params); prepare_for_recurrence(input_ids, params, self); return input_ids (qwen3_next pattern — glm_moe_dsa only did the first; this model needs BOTH because the cache is hybrid MLA-paged + recurrent).
     check_compat: import fla.ops.kda (chunk_kda, fused_recurrent_kda) and fla.modules.fused_norm_gate.rms_norm_gated — hard-require, qwen3_next pattern. VERIFIED: upstream fla-org/flash-linear-attention main exports exactly chunk_kda and fused_recurrent_kda from fla/ops/kda/__init__.py; fused_recurrent_kda supports initial_state, output_final_state, use_qk_l2norm_in_kernel, and even in-kernel safe gate (A_log/dt_bias/lower_bound via use_gate_in_kernel); chunk_kda takes precomputed g log-decay [B,T,H,K] with initial_state + output_final_state. FusedRMSNormGated/rms_norm_gated accepts activation="sigmoid" ("if self.activation not in ['swish','silu','sigmoid']: raise"). Pin the fla version in check_compat with a clear error message.

B. exllamav3/modules/kimi_delta_net.py — NEW (~700 lines, cloned from gated_delta_net.py). Class KimiDeltaAttention. See §2 for the precise GDN delta. caps {"recurrent_cache": True}; layer_state_cls = GDNLayerState reused verbatim with num_v_heads=64, k_head_dim=128, v_head_dim=128, conv fdim=24576 (GDN state shapes are already (B, hist+1, H, Dk, Dv) fp32 + conv (B, fdim, kernel+hist) bf16 — the same family). Keep a torch_recurrent_kda reference function in-file ("For reference, not used"), mirroring GDN's practice; it doubles as the parity oracle.

C. exllamav3/modules/mla_attn.py — MODIFY. Two orthogonal changes:
   (1) NoPE support: allow qk_rope_head_dim == 0 with rope_settings None. Guard every rope branch (`if self.rope is not None` already exists at line ~567; audit lines 380–410 _indexer_keys partial-rope, 555–630 q_pe/k_pe slicing, 795 kpe_cache alloc) so zero-width slices never call .contiguous()/rope.apply; kpe plane allocated at width 0 or None.
   (2) New indexer_mode "kpool" alongside "full"/"shared": adds raw tensors idx_kpool_gate = Linear(hidden→128, key ".indexer.index_kpool_compress_gate" — NOTE this is a bare weight tensor, not weight-suffixed: it is stored as `...indexer.index_kpool_compress_gate` [128,4096]; load raw like A_log rather than as a Linear, or teach Linear an exact-key mode) and idx_kpool_ape (raw [4,128] fp32, key ".indexer.index_kpool_compress_ape"). k_norm is LayerNorm WITH bias, eps 1e-6 (existing idx_k_norm is LayerNorm — parameterize eps). Scoring path per §"indexer math" below. idx_plane_dim = 2*index_head_dim = 256 in kpool mode (per-token cache of ki[128] ++ gate_score[128]).

D. exllamav3/modules/attention_fn/mla_triton.py + dsa_triton.py — MODIFY: constexpr HAS_ROPE (D_r == 0) guard so the rope-plane tl.loads and the q_pe accumulation are compiled out. The kernels already take D_r as a parameter; this is a mechanical guard, but it MUST be done in the kernels (a 0-width tl.load is undefined, not merely wasteful).

E. exllamav3/cache/mla.py — MODIFY: CacheLayer_MLA_fp16 to accept idx_plane_dim=256 (already parameterized via idx_plane_dim per Study 1) and kpe plane width 0. cache/cache.py needs NO structural change: it instantiates recurrent state per module with caps["recurrent_cache"] (KDA layers) and MLA cache layers for MLAttention modules; the MLA+recurrent combination is new but composes per-layer — add an integration test rather than new code.

F. exllamav3/modules/hyperconnections.py — MODIFY: allow_bf16=True on the six get_tensor calls (HyperConnection.load, HyperHead.load); add class ContractStreams(Module): forward = x.mean(dim=2), out fp16 (mirror ExpandStreams; ~20 lines; no tensors; optimizer_targets []). Consistent with TransformerBlock's existing export_state stream-mean.

G. exllamav3/modules/gated_rmsnorm.py — MODIFY: add activation: str = "silu" parameter; thread through forward_fla (rms_norm_gated(activation=...)) and forward_torch (F.silu → torch.sigmoid); when activation != "silu", route forward() to forward_fla/forward_torch instead of ext.gated_rms_norm (the CUDA ext is silu-only — extending it is a later optimization, not v1).

H. exllamav3/architecture/architectures.py — +1 import (from .glm5_next import Glm5NextModel), +1 list entry.
I. exllamav3/modules/__init__.py — export KimiDeltaAttention, ContractStreams.
J. NEW test harness (not shipped): tests/glm5_parity/reference_glm5.py (pure-torch oracle assembled from refs/mhc_torch.py verbatim + Study-2 KDA recurrence + naive materialized NoPE MLA + kpool indexer + sigmoid-noaux_tc MoE with the asymmetric clamp) + runner per §4.

COMPLETE TENSOR MAP (p = model.language_model; i = layer idx; all shapes checkpoint-verified from st_index + shard headers):
- p.embed_tokens.weight [154880,4096] → Embedding, never quantized, fp16 copy.
- p.layers.i.hc_attn_fn [24,16384] / _base [24] / _scale [3], hc_ffn_* — HyperConnection raw, loaded .float(), stored as-is (bf16 in, carried through by get_tensors).
- p.layers.i.input_layernorm.weight [4096], post_attention_layernorm.weight [4096] → RMSNorm raw.
- KDA layers (34× at i ∉ {3,7,...,43}), under p.layers.i.self_attn.:
  q_proj.weight [8192,4096] → Linear, K6 (qmap block.attn.input); k_proj.weight [8192,4096] → K6 (same Hessian label); v_proj.weight [8192,4096] → K6 (same); o_proj.weight [4096,8192] → K6 (qmap block.attn.output);
  b_proj.weight [64,4096], f_a_proj.weight [128,4096], f_b_proj.weight [8192,128], g_a_proj.weight [128,4096], g_b_proj.weight [8192,128] → Linear qmap=None, pad_to=1 → fp16 UNQUANTIZED (see §3);
  q_conv1d.weight / k_conv1d.weight / v_conv1d.weight [8192,1,4] → raw (allow_bf16), merged at load into one [24576,1,4] depthwise conv (bit-identical: depthwise conv is per-channel), re-emitted split by get_tensors;
  A_log [64] fp32 raw; dt_bias [8192] fp32 raw; o_norm.weight [128] → GatedRMSNorm(activation="sigmoid") raw. NO biases anywhere; conv has no bias.
- DSA layers (11× at i ∈ {3,7,...,43}), under p.layers.i.self_attn.:
  q_a_proj.weight [1536,4096] → K6 (qmap .input); q_a_layernorm.weight [1536] raw; q_b_proj.weight [16384,1536] → K6 (qmap .q_a); kv_a_proj_with_mqa.weight [512,4096] → K6 (qmap .input; exactly kv_lora_rank rows — no rope rows); kv_a_layernorm.weight [512] raw; kv_b_proj.weight [32768,512] → RAW fp16 (existing MLA design: split into w_uk_flat/w_uv_flat at load, absorb is a bmm, reconstructed by get_tensors); o_proj.weight [4096,16384] → K6 (qmap .o);
  indexer.wq_b.weight [4096,1536] → K6 with qmap .q_a (inherits q_b's Hessian — existing precedent); indexer.wk.weight [128,4096] fp16 unquantized; indexer.k_norm.weight/.bias [128] raw (LayerNorm eps 1e-6); indexer.weights_proj.weight [32,4096] fp16 unquantized; indexer.index_kpool_compress_gate [128,4096] fp16 unquantized raw; indexer.index_kpool_compress_ape [4,128] fp32 raw.
- MoE (i ≥ 3): mlp.gate.weight [288,4096] unquantized (router; fp32 GEMM at run time), mlp.gate.e_score_correction_bias [288] fp32 raw; mlp.experts.{0..287}.gate_proj/up_proj [2048,4096] and down_proj [4096,2048] → K6 (gate/up share one 4096² Hessian "block.mlp.input", each down has its own — existing BlockSparseMLP layout); mlp.shared_experts.gate/up/down (2048 interm) → K6, select_hq_bits=2.
- Dense (i < 3): mlp.gate_proj/up_proj [12288,4096], down_proj [4096,12288] → K6, select_hq_bits=1.
- p.norm.weight [4096] raw; lm_head.weight [154880,4096] → qbits_key head_bits (default 6; 16 = keep fp16).
- SKIPPED (no module references them → dropped at compile): p.layers.45.* (MTP), model.visual.*.

------------------------------------------------------------
2. VERBATIM REUSE vs NEW CODE — and the exact KDA-vs-GDN delta
------------------------------------------------------------

Reused VERBATIM (zero changes): TransformerBlock incl. HC site semantics; ExpandStreams; HyperConnection math + ext.hc_mix/hc_apply kernels (post_mult 2.0 matches); BlockSparseMLP with router_type="dots" (sigmoid → +e_score_bias for top-8 selection → weights from UNBIASED scores → normalize → ×2.5; identical to GLM-5.2) and act_limit; GatedMLP; Embedding/RMSNorm/Linear; GDNLayerState + GDNState + conv causal_conv1d_update slotted Triton kernel + batched_conv_rewind/batched_state_rewind + recurrent_util (prepare_for_recurrence/advance); the whole conversion pipeline (calibration_all_experts capture, grouped LDLQ, resume machinery); glm_moe_dsa's MLA absorbed path, latent cache, dsa_topk ext, indexer bypass rule (sparse only when max(host_seqlens)+seqlen > index_topk — below that GLM-5.3's indexer selects everything, per the mlx reference, so dense is exact).
  Note on act_limit: exl3's silu clamp is post-activation (min(silu(g),10)·clamp(u,±10)) vs vLLM's pre-activation clamp (clamp(g,max=10) then g·sigmoid(g)·clamp(u,±10)). They differ only for g∈(10,∞) by ≤ 4.5e-4 absolute at |out|≈10 — one bf16 ulp at 10 is 0.0625, so this is far below checkpoint precision. Accept; document in the arch file header.

Reused WITH SMALL MODS: mla_attn.py (NoPE guards + kpool mode), mla/dsa triton kernels (HAS_ROPE), CacheLayer_MLA (plane widths), hyperconnections.py (allow_bf16 + ContractStreams), gated_rmsnorm.py (sigmoid).

NEW CODE — KimiDeltaAttention, precise delta from GatedDeltaNet:
  a) In-projections. GDN: fused qkvz (2·nk·dk + 2·nv·dv) + tiny ba (2·nv), or Qwen3.5 split qkv/z/b/a. KDA: SIX separate checkpoint projections, and the gates are LOW-RANK two-stage: q,k,v [8192] each; b [64] (beta logits, one scalar per head); f_a [128]→f_b→[8192]=[64,128] (decay logits per KEY channel); g_a [128]→g_b→[8192]=[64,128] (output-gate logits per V channel). GDN's ext.gated_delta_net_fused_op unpack kernels do NOT apply — replace with plain Linear forwards + torch elementwise (the two extra 128-rank GEMMs are noise at this scale). Optionally fuse q|k|v|b|f_a|g_a into one GEMM later (vLLM does); v1 keeps them separate = checkpoint layout, zero remap.
  b) Decay semantics — THE core math difference. GDN: scalar decay per V-HEAD: g[t,h] = −exp(A_log[h])·softplus(a[t,h]+dt_bias[h]); state update S ← exp(g)·S (whole [Dk,Dv] scaled by one scalar). KDA (safe gate, checkpoint-verified gate_lower_bound −5.0): PER-KEY-CHANNEL vector decay: g_log[t,h,:] = −5.0 · sigmoid( exp(A_log[h]) · (f_b_out[t,h,:] + dt_bias[h,:]) ) ∈ (−5,0)^128; state update S ← diag(exp(g_log_t))·S — decay indexed on the Dk axis. dt_bias is [8192]→[64,128] per key channel, NOT per head. Sigmoid-bounded, NOT softplus — the unbounded variant is not used by this checkpoint.
  c) Recurrence (per head, S ∈ R^{128×128} fp32): S ← diag(exp(g_t))·S; kv_mem = Sᵀk_t; delta = (v_t − kv_mem)·beta_t (beta = sigmoid(b), scalar per head); S ← S + k_t⊗delta; o_t = Sᵀq_t. q/k are per-head-vector L2-normalized (x·rsqrt(sum(x²)+1e-6) — SUM not mean), q additionally scaled 128^−0.5 (fla applies both in-kernel with use_qk_l2norm_in_kernel=True).
  d) Kernel dispatch — SIMPLER than GDN. GDN needs its own CUDA recurrent kernel because fla's chunk_gated_delta_rule can't take history in its flow. fla's chunk_kda accepts initial_state + output_final_state, so: prefill/chunk (per active slot, GDN-style loop) → compute g_log in torch fp32 (one elementwise expression over [T,64,128]) → chunk_kda(q,k,v,g_log,beta,scale=128^−0.5, initial_state=S_slot, output_final_state=True, use_qk_l2norm_in_kernel=True); decode → fused_recurrent_kda(same); NO ext.cuda_recurrent_* port, NO BC_* CUDA-graph block for v1 (decode runs the fla Triton kernel; warm it up before graph capture or exempt KDA decode from graphing — see risk R4).
  e) Conv: identical machinery, channels 24576 (GDN slotted causal_conv1d_update reused as-is), SiLU after conv, no bias.
  f) Output gate: GDN = GatedRMSNorm silu(z); KDA = RMSNorm(o; o_norm.weight, eps 1e-5) · SIGMOID(g2) — use patched GatedRMSNorm(activation="sigmoid") via fla rms_norm_gated (verified supported) with torch fallback.
  g) Cache/state: GDNLayerState constructor args change only in dims; clear/rewind/stash/unstash/checkpointing logic untouched.

Indexer math for "kpool" mode (new, ~150 lines in mla_attn.py; oracle = refs/indexer_kpool.py + mlx _pooled_states):
  per token append to idx plane: ki = LayerNorm_{1e-6}(wk(x)) [128] and gs = x@gateᵀ [128] (256 fp16 values/token). At query time t (only when context > 2048): qi = wq_b(q_a_latent) → [32,128]; w = weights_proj(x) fp32 [32]; for each complete pool P of 4 consecutive tokens: probs[.,d] = softmax over 4 slots of (gs[slot,d]+ape[slot,d]); pool_key[d] = Σ probs·ki; score(t,P) = Σ_h w_h·relu(qi_h·pool_key·128^−0.5)·32^−0.5, pool visible iff its last token ≤ t; top-512 pools by score → 2048 token indices; always append the trailing partial pool's ≤3 tokens. Reuse ext.dsa_topk on pool scores then expand ×4. NO rope, NO Hadamard/fp8 (vLLM storage detail — dropped in bf16 port). Pool recomputation each forward is O(T·128) — memory-bound like reading the plane; acceptable v1, kernel-fuse later.

------------------------------------------------------------
3. WHAT STAYS BF16/UNQUANTIZED, AND WHY
------------------------------------------------------------
- mHC hc_attn_*/hc_ffn_* (45×2×(24×16384+24+3) ≈ 35M): raw fp32-at-load. The Sinkhorn fixed-point loop and the sigmoid pre/post gates modulate the ONLY pathway every activation takes (4-stream residual); error here is global and multiplicative across 45 layers. Also structurally unquantizable: fn reads the raw stream stack, so no Hessian capture site exists. HF fp8 config itself lists hyper_connection in modules_to_not_convert.
- KDA A_log [64], dt_bias [8192], conv weights [3×8192×4]: raw (GDN precedent, get_tensor allow_bf16). They sit inside exp/sigmoid of a recurrence — per-token decay error compounds over the full context.
- KDA b/f_a/f_b/g_a/g_b: fp16 unquantized (qmap None, pad_to 1). Rationale: (i) GDN precedent keeps ba_proj and Qwen3.5's split b/a fp16 ("router-like: tiny, and noise is coherent"); (ii) f/g feed the decay and output gates — same compounding argument as A_log; (iii) the 128-rank bottleneck makes LDLQ Hessians degenerate; (iv) cost: ~3.4M params/layer × 34 layers ≈ 117M ≈ 234 MB fp16 ≈ +0.6% of the ~245 GB K6 output. Cheap insurance.
- All norms (input/post_attention/q_a/kv_a layernorms, o_norm, final norm, indexer k_norm): raw — exl3 never quantizes norms.
- kv_b_proj [32768,512] per DSA layer: raw fp16 (existing MLA design — absorb is a bmm; ~184 MB total across 11 layers).
- Indexer wk/weights_proj/kpool gate/ape: fp16/fp32 unquantized (existing "router-like" precedent — selection noise is coherent across every token). wq_b stays QUANTIZED with the .q_a Hessian label (existing glm_moe_dsa decision; it's 6.3M params/layer and its noise only perturbs relative scores within a head).
- embed_tokens: fp16 copy (framework-enforced). MoE router gate + e_score_correction_bias: unquantized fp32 GEMM (moe_router_dtype float32 honored; expert SELECTION must be stable — a flipped expert id is a discrete error).
- Everything else at K6 (-b 6.0), lm_head at head_bits (6 default), -hq on (select_hq_bits already set per module above, glm_moe_dsa precedent).
Conversion command: python convert.py -i <bf16_ckpt> -o <out> -w <work_on_persistent_vol> -b 6.0 -hq [-hb 6] ; resume with -w <work> -r. No -mb/-vb needed (components not registered).

------------------------------------------------------------
4. LAYER-PARITY VALIDATION PLAN (one RTX PRO 6000, BF16 checkpoint on disk)
------------------------------------------------------------
Harness pattern = the converter's own loop: Config.from_directory → Model.from_config(component="text") → for each module: module.load(device) → forward → module.unload(). Unquantized checkpoints load Linears as fp16 passthrough, so exl3 modules run pre-quantization. Peak VRAM = one sparse layer ≈ 15 GB fp16 — trivial on 96 GB; whole-model residency is impossible (642 GB) and NOT needed.

Stage 0 — Oracle (tests/glm5_parity/reference_glm5.py, pure torch fp32, CPU-or-GPU): mhc_pre/post_torch copied verbatim from refs/mhc_torch.py; KDA per-token recurrence exactly as §2c (also a chunked cross-check via refs/fla_kda.py math); naive materialized NoPE MLA (q_b/kv_b expanded, softmax over full or selected set); kpool indexer per §2 oracle; sigmoid-noaux_tc MoE with asymmetric clamp (gate clamp max=10 one-sided, up ±10). Load tensors per layer directly from the safetensors shards by name.

Stage 1 — Per-module parity (each test: random activations bsz1 × seqlen {1, 17, 512, 4096}, N(0,σ) σ∈{0.5, 5}, PLUS real activations: run the oracle end-to-end over two 2048-token fidelity-suite prompts on CPU overnight once, dumping per-layer stream stacks; feed those as inputs):
  T1 mHC: HyperConnection.mix/apply_ (torch path) vs mhc_pre/post_torch → max abs ≤ 1e-5 (both fp32, same ops — near-bit-exact expected); fused ext.hc_mix path vs torch path → rel ≤ 1e-3 (half collapse). ContractStreams + final norm + head vs oracle → rel ≤ 1e-3.
  T2 KDA (the big one): (a) module forward vs per-token oracle: out max abs ≤ 5e-3, rel Frobenius ≤ 1e-3; final recurrent state rel ≤ 2e-3 after 4096 tokens. (b) three-way consistency chunk_kda vs fused_recurrent_kda vs torch oracle on identical fp32 inputs ≤ 1e-3 rel (isolates fla from module plumbing). (c) DECODE-EQUALS-PREFILL: 512 tokens in one prefill vs 512 single-token decode steps through the exl3 module with a live Cache → rel ≤ 2e-3 (catches conv-state slotting, state advance, gate/beta decode paths). (d) rewind test: advance 300, rewind 44 (checkpoint interval crossing), re-advance → matches straight run ≤ 2e-3.
  T3 MLA/NoPE dense: context ≤ 2048 (bypass ⇒ dense exact): vs oracle softmax attention rel ≤ 2e-3; ALSO cross page boundary (seqlen 300 appended after 200 cached) to exercise paged append with 0-width kpe plane.
  T4 Sparse selection: contexts {2049, 8192, 32768}: (i) selected token-index sets vs oracle — Jaccard ≥ 0.999 excluding score-tie rows (compare pool scores with tolerance 1e-4; ties may permute); tail tokens always present; (ii) end output rel ≤ 5e-3 (selection differences at the margin contribute ~0 mass); (iii) bit-equality of dense-vs-sparse at context exactly index_topk (bypass boundary).
  T5 MoE: expert-ID EXACT match on 4096 real rows (fp32 router — any mismatch is a bug, not noise); output rel ≤ 2e-3; dense layers and shared expert same threshold; clamp band test: synthetic rows driving gate_pre through (9.5, 10.5) — accept documented ≤ 5e-4 deviation vs vLLM clamp.
Threshold rationale: fp16 weights vs fp32 oracle gives ~1e-3 rel for a single GEMM chain; 5e-3 max-abs headroom covers the 8192-wide reductions. Anything worse indicates a real defect, not precision.

Stage 2 — Chained end-to-end (streamed): sequentially load layer i, feed the ORACLE's layer-(i−1) real-activation stream stack, record exl3 output, also carry exl3's own chained output in parallel; both over 2×2048 real tokens. Acceptance: per-layer (oracle-input) rel ≤ 5e-3 every layer; chained drift ≤ 1% rel at layer 45; final logits: mean KL(oracle‖exl3) ≤ 5e-3, top-1 agreement ≥ 99.5%, top-8 set overlap ≥ 99%. Runtime ≈ 45 sequential layer loads from disk ≈ 30–60 min.

Stage 3 — Conversion smoke: truncated 5-layer model (hack num_hidden_layers=5 in a copied config + subset shards) through convert.py end-to-end at -b 6.0 including a kill-and-resume mid-layer; then full K6 run; then the published fidelity suite (malaiwah/GLM-5.3-Flash-fidelity-suite-v1) against the BF16 reference — acceptance per suite spec (out of this blueprint's scope, but Stage 2's KL numbers predict it).

------------------------------------------------------------
5. RISK REGISTER — the 3 hardest parts (+2 watch items)
------------------------------------------------------------
R1 (hardest) KDA state lifecycle across execution modes. Chunked-fla vs recurrent-fla vs decode, initial_state handoff per slot, conv-state slotting, rewind/stash for speculative decoding and the 2048-token CPU checkpoints. Per-KEY-channel decay is easy to transpose silently (decay on Dk axis; GDN's habits point at Dv), and l2norm eps (sum+1e-6) differs from naive mean-based RMS. Failure mode: correct short outputs, drift after ~1K tokens. Mitigation: T2c/d decode-equals-prefill + rewind tests are non-negotiable gates; keep the torch oracle in-file.
R2 NoPE (D_r = 0) through the MLA triton kernels and cache planes. Zero-width tl.load/stores are UB — may pass on one GPU and corrupt on another; every rope touchpoint (main attn, cache append, kpe plane alloc, dsa_attn q_pe arg) must be constexpr-guarded and exercised across page boundaries (T3). This touches upstream-shared kernels, so guard behind HAS_ROPE to keep GLM-5.2 bit-identical.
R3 kpool indexer parity. Selection errors are silent quality degradation, visible only >2048 context and only statistically. Softmax pooling numerics (fp16-cached gs/ki vs vLLM's fp8+Hadamard pipeline) can reorder near-tie pools — hence Jaccard-with-tie-tolerance testing at three context lengths, plus the bypass-boundary bit-equality check. Also the 256-wide idx plane doubles indexer cache traffic vs GLM-5.2 (still ≈ 512 B/token/DSA-layer fp16 — fine).
R4 (watch) Decode performance: no BC_* CUDA-graph block for KDA; fla Triton kernels autotune on first call — must warm up every (H,D,slot) shape before exl3's graph capture, or exempt KDA layers from graphs initially. Ship v1 correct-but-slower; port a BC block later.
R5 (watch) Conversion-scale plumbing: hybrid MLA+recurrent Cache is a first (integration test it before the big run); ~34 GB fp32 stream state per checkpoint write (×2 transient during ckpt swap — needs ~256 GB RAM box or accept slower cadence via -cpi); dsv4 precedent says stream-shaped reference states already survive resume (quant_preserves are re-initialized empty on resume — port must not depend on cross-module preserves, and per the plan it does not).

------------------------------------------------------------
6. HOUR ESTIMATES (expert already familiar with exllamav3)
------------------------------------------------------------
1. Arch file + config asserts + registration + tensor-name dry-load (stc coverage check vs st_index): 4 h
2. hyperconnections allow_bf16 + ContractStreams + T1 mHC parity: 3 h
3. GatedRMSNorm sigmoid option: 1 h
4. KimiDeltaAttention module (clone GDN; six projections; safe gate; fla calls; GDNLayerState wiring): 16 h
5. KDA parity harness incl. oracle + T2a–d green: 6 h
6. MLA NoPE audit + HAS_ROPE kernel guards + T3: 8 h
7. kpool indexer + idx-plane changes + T4: 14 h
8. MoE/dense wiring (act_limit, router dtype) + T5: 3 h
9. Hybrid cache integration + prepare_inputs + Stage-2 chained validation: 8 h
10. 5-layer conversion smoke incl. kill/resume; fix conversion-only fallout (get_tensors round-trip, capture-mode HC no-inplace already handled upstream): 5 h
11. Full K6 conversion babysit + fidelity suite + writeup: 4 h active (+12–20 h single-GPU wall, ~7–12 h on 2 GPUs; trust the converter's ETA after the first hour; work_dir on persistent volume, resume with -w -r)
TOTAL ≈ 72 active expert-hours; ~2 calendar weeks including GPU runs. Critical path: items 4–5 (KDA) can proceed in parallel with 6–7 (DSA) by two agents; item 9 requires both.
