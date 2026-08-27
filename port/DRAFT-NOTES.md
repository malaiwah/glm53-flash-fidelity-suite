# Draft notes — exllamav3/architecture/glm5_next.py

> Produced 2026-08-27 by a 7-agent design workflow (blueprint -> draft -> parity harness -> adversarial review) against exllamav3 v1.4.4.

## Summary

First draft of exllamav3/architecture/glm5_next.py written and syntax-checked (py_compile OK). Follows glm_moe_dsa (MLA/DSA/MoE tails), qwen3_next (recurrent prepare_inputs + fla check_compat), deepseek_v4 (mHC wiring), glm4v_moe (text_config-> reads, model.language_model prefix). All tensor keys and config values verified against refs/st_index.json and refs/hf_config.json. PORT-CHECK tags mark: KimiDeltaAttention constructor signature (module not yet written), MLAttention kpool-mode kwargs (index_kpool, shared+kpool untested), ContractStreams (new module), fla minimum version pin, chat template. Working copy at /private/tmp/claude-501/-Users-mbelleau-Projects-GLM/c1546622-1c41-4561-ba68-92b6b9cb9811/scratchpad/glm5_next_draft.py.

## Detail

Draft of exllamav3/architecture/glm5_next.py (syntax-checked working copy at /private/tmp/claude-501/-Users-mbelleau-Projects-GLM/c1546622-1c41-4561-ba68-92b6b9cb9811/scratchpad/glm5_next_draft.py). Depends on plan items B/C/F/I landing: KimiDeltaAttention (modules/kimi_delta_net.py), MLAttention "kpool" indexer mode + index_kpool kwarg, ContractStreams in modules/hyperconnections.py, and both exported from modules/__init__.py. Also requires architectures.py registration (import Glm5NextModel, add to list).

```python
from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    MLAttention,
    GatedMLP,
    Linear,
    BlockSparseMLP,
    HyperConnection,
    ExpandStreams,
    ContractStreams,
    KimiDeltaAttention,
    GDNState,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence

# GLM-5.3-Flash (glm5_next): hybrid 45-layer text stack.
#
#   - 34 KDA layers (Kimi Delta Attention): linear attention with per-key-channel low-rank
#     decay gates (f_a/f_b), per-head beta (b_proj), depthwise short conv (kernel 4) on q/k/v,
#     SAFE lower-bounded gate (gate_lower_bound -5.0) and a sigmoid-gated output RMSNorm.
#     Recurrent state, no paged KV.
#   - 11 DSA layers at idx 3, 7, ..., 43: MLA in NoPE form (qk_rope_head_dim == 0, no rope
#     anywhere; positional information comes entirely from the KDA layers) with a lightning
#     indexer in "kpool" mode: per-token index keys plus a learned 4:1 pooled compression
#     (index_kpool_compress_gate/_ape) that scores key pools, always selecting the tail pool.
#     Latent (kv_lora_rank 512) paged cache, absorbed attention, index_topk 2048.
#   - mHC (manifold-constrained hyper-connections): the residual runs as hc_mult = 4 fp32
#     streams; every sublayer site mixes them through a HyperConnection (Sinkhorn-normalized
#     combine, sigmoid pre/post, post_mult 2.0). Same machinery as DeepSeek-V4; GLM has no
#     learned head so the final contract is a plain mean over the streams (ContractStreams).
#   - MoE: layers 0-2 dense (intermediate 12288), layers 3-44 sparse with 288 routed experts,
#     top-8, sigmoid scoring + noaux_tc selection bias, weights normalized then scaled by 2.5,
#     plus one shared expert (moe_intermediate 2048). SwiGLU clamp (swiglu_limit 10.0).
#     NOTE on the clamp: exllamav3 applies min(silu(g), limit) * clamp(u, +-limit)
#     (post-activation) where the reference clamps g before silu. The two differ only for
#     g > limit, by <= 4.5e-4 absolute at |out| ~= 10 -- far below one bf16 ulp at 10
#     (0.0625), i.e. below checkpoint precision. Accepted as-is.
#
# Checkpoint namespace: model.language_model.layers.N.*, lm_head.weight top-level. The MTP
# block (model.language_model.layers.45.*, num_nextn_predict_layers == 1) and the vision
# tower (model.visual.*) are not ported in v1: no "mtp"/"vision" components are registered,
# so those tensors are never referenced and fall out of the compiled output.
#
# Reference implementations: transformers models/glm5_next, vLLM glm5_next, mlx-lm glm5_next.


class Glm5NextConfig(Config):
    arch_string = "Glm5NextForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Glm5NextModel},
            **kwargs
        )

        # All text parameters live under text_config (top-level model_type glm5_next wraps
        # glm5_next_text plus the vision tower we skip)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.tie_word_embeddings = self.read_cfg(
            bool, ["tie_word_embeddings", "text_config->tie_word_embeddings"], False
        )

        # Layer schedule: KDA vs. DSA per layer
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) >= self.num_hidden_layers and all(
            t in ("linear_attention", "deepseek_sparse_attention")
            for t in self.layer_types[:self.num_hidden_layers]
        ), f"Unexpected layer_types: {self.layer_types}"

        # KDA (linear attention) params. Only the SAFE lower-bounded gate variant is
        # implemented, so gate_lower_bound must be present
        la_cfg = self.read_cfg(dict, "text_config->linear_attn_config", no_default)
        self.kda_num_heads = int(la_cfg["num_heads"])
        self.kda_head_dim = int(la_cfg["head_dim"])
        self.kda_conv_kernel_size = int(la_cfg["short_conv_kernel_size"])
        assert "gate_lower_bound" in la_cfg, \
            "glm5_next: linear_attn_config.gate_lower_bound missing (only the SAFE " \
            "lower-bounded KDA gate is supported)"
        self.kda_gate_lower_bound = float(la_cfg["gate_lower_bound"])

        # MLA (DSA layers) params. NoPE: qk_rope_head_dim == 0 and no rope settings at all --
        # do not call read_rope_settings_default here (override_head_dim = 0 would break it)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "text_config->q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, "text_config->kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "text_config->qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "text_config->qk_rope_head_dim", 0)
        assert self.qk_rope_head_dim == 0, \
            f"glm5_next: expected NoPE MLA (qk_rope_head_dim == 0), got {self.qk_rope_head_dim}"
        self.assert_cfg(bool, "text_config->mla_use_nope", True, True)
        self.v_head_dim = self.read_cfg(int, "text_config->v_head_dim", no_default)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # Reported head dim, for allocation and logging; the cache stores the latent instead
        self.head_dim = self.qk_head_dim
        self.rope_settings = None
        self.sm_scale = self.qk_head_dim ** -0.5

        # DSA lightning indexer, kpool flavor: per-token keys plus learned 4:1 pooled
        # compression with always-selected tail pool. The kpool scoring path is the only
        # variant implemented, so the flags must be present with these values
        self.index_n_heads = self.read_cfg(int, "text_config->index_n_heads", no_default)
        self.index_head_dim = self.read_cfg(int, "text_config->index_head_dim", no_default)
        self.index_topk = self.read_cfg(int, "text_config->index_topk", no_default)
        self.index_kpool = self.read_cfg(int, "text_config->index_kpool", no_default)
        assert self.index_kpool == 4, \
            f"glm5_next: only index_kpool == 4 is supported, got {self.index_kpool}"
        self.assert_cfg(bool, "text_config->index_kpool_compress", True, True)
        self.assert_cfg(bool, "text_config->index_kpool_always_select_tail", True, True)
        self.indexer_types = self.read_cfg(list, "text_config->indexer_types", no_default)
        assert len(self.indexer_types) >= self.num_hidden_layers and all(
            t in ("full", "shared") for t in self.indexer_types
        ) and self.indexer_types[0] == "full", \
            f"Unexpected indexer_types: {self.indexer_types}"

        # MLP params
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "text_config->n_shared_experts", 1)
        self.num_experts = self.read_cfg(int, "text_config->n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->num_experts_per_tok", 8)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->routed_scaling_factor", 1.0)
        self.swiglu_limit = self.read_cfg(float, "text_config->swiglu_limit", 10.0)
        first_k_dense = self.read_cfg(int, "text_config->first_k_dense_replace", 3)
        self.mlp_layer_types = self.read_cfg(
            list, "text_config->mlp_layer_types",
            ["dense" if idx < first_k_dense else "sparse" for idx in range(self.num_hidden_layers)]
        )
        assert all(t in ("dense", "sparse") for t in self.mlp_layer_types)
        self.n_group = self.read_cfg(int, "text_config->n_group", 1)
        self.topk_group = self.read_cfg(int, "text_config->topk_group", 1)
        assert self.n_group in (None, 1) and self.topk_group in (None, 1), \
            f"Group-limited expert routing (n_group = {self.n_group}, topk_group = " \
            f"{self.topk_group}) is not supported"
        self.assert_cfg(str, "text_config->scoring_func", "sigmoid", True)
        self.assert_cfg(str, "text_config->topk_method", "noaux_tc", True)
        self.assert_cfg(bool, "text_config->norm_topk_prob", True, True)

        # mHC. post_mult 2.0 is hardcoded in both the ext kernels and the torch fallback of
        # HyperConnection, and 2.0 is the glm5_next config default when the key is absent
        self.assert_cfg(bool, "text_config->mhc", True)
        self.hc_mult = self.read_cfg(int, "text_config->hc_mult", 4)
        self.hc_sinkhorn_iters = self.read_cfg(int, "text_config->hc_sinkhorn_iters", 20)
        self.hc_eps = self.read_cfg(float, "text_config->hc_eps", 1e-6)
        self.assert_cfg(float, "text_config->mhc_post_mult_value", 2.0, True)
        self.assert_cfg(bool, "text_config->mhc_no_norm_weight", False, True)

        # MTP block exists in the checkpoint (layers.45) but is not ported in v1
        self.num_mtp_layers = self.read_cfg(int, "text_config->num_nextn_predict_layers", 0)


class Glm5NextModel(Model):
    config_class = Glm5NextConfig

    def __init__(
        self,
        config: Glm5NextConfig,
        key_prefix: str = "model.language_model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            ExpandStreams(
                config = config,
                key = "hc_expand",
                hc_mult = config.hc_mult,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"

            if config.layer_types[idx] == "linear_attention":
                # PORT-CHECK: KimiDeltaAttention is the new module (modules/kimi_delta_net.py,
                # cloned from GatedDeltaNet). Keep this call in sync with its final signature.
                # KDA is head-symmetric (64 k-heads == 64 v-heads, head_dim 128 both sides);
                # q/k/v/o are quantized, the tiny gate projections (b, f_a/f_b, g_a/g_b) stay
                # fp16 inside the module (qmap None, pad_to 1), conv/A_log/dt_bias/o_norm raw.
                attn = KimiDeltaAttention(
                    config = config,
                    key = f"{key}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_heads = config.kda_num_heads,
                    head_dim = config.kda_head_dim,
                    rms_norm_eps = config.rms_norm_eps,
                    conv_kernel_size = config.kda_conv_kernel_size,
                    gate_lower_bound = config.kda_gate_lower_bound,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_b = "b_proj",
                    key_f_a = "f_a_proj",
                    key_f_b = "f_b_proj",
                    key_g_a = "g_a_proj",
                    key_g_b = "g_b_proj",
                    key_conv1d_q = "q_conv1d",
                    key_conv1d_k = "k_conv1d",
                    key_conv1d_v = "v_conv1d",
                    key_a_log = "A_log",
                    key_dt_bias = "dt_bias",
                    key_norm = "o_norm",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
            else:
                # PORT-CHECK: indexer_mode "kpool" is the new MLAttention mode (plan item C):
                # NoPE scoring over per-token index keys + gate scores cached in a 256-wide
                # idx plane, learned 4:1 pool compression (index_kpool_compress_gate/_ape),
                # tail pool always selected. GLM-5.3-Flash ships indexer_types == ["full"]*45;
                # the "shared" mapping below is pass-through support only and is untested in
                # combination with kpool scoring.
                attn = MLAttention(
                    config = config,
                    key = f"{key}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = 0,
                    v_head_dim = config.v_head_dim,
                    rope_settings = None,
                    q_lora_rank = config.q_lora_rank,
                    sm_scale = config.sm_scale,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                    indexer_mode = "kpool" if config.indexer_types[idx] == "full" else "shared",
                    index_n_heads = config.index_n_heads,
                    index_head_dim = config.index_head_dim,
                    index_topk = config.index_topk,
                    # Indexer k_norm is LayerNorm (with bias) at eps 1e-6, independent of the
                    # model-wide rms_norm_eps 1e-5
                    index_norm_eps = 1e-6,
                    # PORT-CHECK: new kwarg on MLAttention (plan item C); the compress gate
                    # ("{key}.indexer.index_kpool_compress_gate", bare [128, 4096] tensor, not
                    # weight-suffixed) and ape ("...compress_ape", [4, 128] fp32) are loaded
                    # raw inside the module relative to key_indexer
                    index_kpool = config.index_kpool,
                )

            if config.mlp_layer_types[idx] == "dense":
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                    select_hq_bits = 1,
                )
            else:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.e_score_correction_bias",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                    router_type = "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = config.n_group,
                    topk_group = config.topk_group,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key}.mlp.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        activation_fn = "silu",
                        act_limit = config.swiglu_limit,
                        select_hq_bits = 2,
                    ) if config.num_shared_experts else None,
                )

            def _hc(tag: str):
                # Checkpoint stores {key}.hc_attn_fn / _base / _scale (bf16; the loader is
                # patched with allow_bf16 = True and immediately promotes to fp32) matching
                # dsv4's "{key}_fn" convention verbatim
                return HyperConnection(
                    config = config,
                    key = f"{key}.hc_{tag}",
                    hc_mult = config.hc_mult,
                    hidden_size = config.hidden_size,
                    sinkhorn_iters = config.hc_sinkhorn_iters,
                    hc_eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    attn_hc = _hc("attn"),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
                    mlp_hc = _hc("ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            # PORT-CHECK: ContractStreams is the new weightless stream collapse in
            # modules/hyperconnections.py (plan item F): plain mean over the hc_mult streams,
            # fp16 out. GLM-5.3 has no learned hc head (no hc_head_* tensors in the
            # checkpoint), unlike DeepSeek-V4's HyperHead
            ContractStreams(
                config = config,
                key = "hc_contract",
                hc_mult = config.hc_mult,
            ),
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Hybrid cache: MLA latent pages on the 11 DSA layers, recurrent GDN-family state on
        # the 34 KDA layers. Cache assembly composes per-module (caps["recurrent_cache"] on
        # KimiDeltaAttention, MLA cache layers on MLAttention); GDNState's position
        # bookkeeping is shape-agnostic and reused as the state front-end
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState

        # No TP: the MLA latent cache cannot be split by head, and linear-attn TP is
        # unimplemented (same as qwen3_next / glm_moe_dsa)
        self.caps.update({"supports_tp": False})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Both preparations are required: paged-attention block tables for the DSA layers
        # AND recurrent-state bookkeeping for the KDA layers (glm_moe_dsa only needs the
        # former, qwen3_next needs both -- this model follows qwen3_next)
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # PORT-CHECK: GLM-4.5/5.x family template, same as glm_moe_dsa; verify against the
        # shipped chat_template before release
        p = f"[gMASK]<sop>"
        if system_prompt:
            p += f"<|system|>\n{system_prompt}"
        p += f"<|user|>\n{prompt}"
        p += f"<|assistant|>\n"
        return p


    @override
    def check_compat(self):
        try:
            import fla
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
            from fla.modules.fused_norm_gate import rms_norm_gated
        except (ModuleNotFoundError, ImportError) as e:
            print(" ## GLM-5.3 (glm5_next) requires flash-linear-attention with KDA support "
                  "(https://github.com/fla-org/flash-linear-attention): pip install "
                  "flash-linear-attention")
            raise e
        # PORT-CHECK: pin the true minimum. Required features, all verified present on fla
        # main: fla.ops.kda.{chunk_kda, fused_recurrent_kda} (fused_recurrent_kda with
        # initial_state / output_final_state / use_qk_l2norm_in_kernel and the in-kernel SAFE
        # gate via use_gate_in_kernel; chunk_kda taking precomputed [B, T, H, K] log-decay),
        # and rms_norm_gated(activation = "sigmoid")
        min_fla = (0, 4, 0)
        try:
            ver = tuple(int(x) for x in fla.__version__.split(".")[:3])
        except Exception:
            ver = None
        if ver is not None and ver < min_fla:
            raise RuntimeError(
                f" ## GLM-5.3 (glm5_next) requires flash-linear-attention >= "
                f"{'.'.join(str(v) for v in min_fla)} (found {fla.__version__}): KDA kernels "
                f"(fla.ops.kda) and sigmoid-gated RMSNorm are missing in older releases"
            )
```

Design notes for the reviewer (grounded in verified sources):

1. Structure follows glm_moe_dsa.py (MLA/DSA + sigmoid noaux_tc MoE tails, dense-first-3 split, head/tail modules), qwen3_next.py (recurrent caps, GDNState, prepare_inputs doing BOTH prepare_for_attn and prepare_for_recurrence, fla check_compat), deepseek_v4.py (ExpandStreams + per-site HyperConnection wiring through TransformerBlock attn_hc/mlp_hc), and glm4v_moe.py ("text_config->" nested reads, key_prefix "model.language_model").

2. All checkpoint tensor keys verified against refs/st_index.json: layers.N.hc_attn_fn/_base/_scale and hc_ffn_* match HyperConnection's "{key}_fn" convention with key = f"{prefix}.layers.{idx}.hc_attn"; KDA keys q/k/v/b/f_a/f_b/g_a/g_b _proj, q/k/v_conv1d, A_log, dt_bias, o_norm, o_proj; DSA keys q_a/q_b/kv_a/kv_b/o_proj + indexer.wq_b/wk/k_norm/weights_proj/index_kpool_compress_gate/_ape; embed_tokens, norm, lm_head. No hc_head_* tensors exist, confirming ContractStreams (mean) over HyperHead.

3. All config values verified against refs/hf_config.json text_config: qk_rope_head_dim 0, mla_use_nope true, index_kpool 4 with compress/always_select_tail true, swiglu_limit 10.0, mhc true with hc_mult 4 / sinkhorn 20 / eps 1e-6; mhc_post_mult_value and mhc_no_norm_weight are absent (hence optional=True asserts, which fail loudly if a future checkpoint sets non-default values). rms_norm_eps is 1e-5; the indexer k_norm LayerNorm eps 1e-6 is passed separately via the existing index_norm_eps kwarg (present in v1.4.4 MLAttention with default 1e-6).

4. Constructor signatures verified against the v1.4.4 source for RMSNorm, Embedding, Linear, GatedMLP (activation_fn/act_limit exist), BlockSparseMLP (router_type "dots", key_e_score_bias, n_group/topk_group, act_limit), HyperConnection (config, key, hc_mult, hidden_size, sinkhorn_iters, hc_eps, rms_norm_eps), ExpandStreams, MLAttention (all kwargs used exist today EXCEPT index_kpool and the "kpool" indexer_mode value, both plan item C — PORT-CHECK tagged). KimiDeltaAttention and ContractStreams do not exist yet; their calls are the proposed contract, PORT-CHECK tagged.

5. Uncertain spots, all tagged # PORT-CHECK in the file: (a) KimiDeltaAttention signature (single num_heads/head_dim since KDA is head-symmetric, unlike GDN's num_k_heads/num_v_heads split; split conv keys mirror GDN's key_conv1d_q/k/v pattern); (b) "kpool" vs "shared" indexer_mode mapping (this checkpoint is all-"full" → all-"kpool"; "shared" retained as pass-through but untested with kpool scoring); (c) ContractStreams as a no-tensor module with a synthetic key "hc_contract" (mirrors dsv4's "hc_expand" synthetic key); (d) fla minimum version pinned provisionally at 0.4.0 — the real floor is whichever release first ships fla.ops.kda.{chunk_kda, fused_recurrent_kda} and rms_norm_gated(activation="sigmoid"), to be confirmed against fla's changelog; (e) default_chat_prompt copied from glm_moe_dsa ([gMASK]<sop> family) pending verification against the shipped chat_template.

6. Deliberate choices consistent with the blueprint: rope_settings = None and no read_rope_settings_default call (override_head_dim = 0 would break it); sm_scale = 256**-0.5 from the unabsorbed qk head dim; select_hq_bits 2 on both attention types and shared experts, 1 on the dense MLPs; head via qbits_key "head_bits"; num_nextn_predict_layers read but no "mtp" component registered and no "vision" component registered, so layers.45.* and model.visual.* fall out of the compiled output; calibration_all_experts = True; supports_tp False.
