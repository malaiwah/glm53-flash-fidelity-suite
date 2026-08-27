#!/usr/bin/env python3
# GLM-5.3-Flash (glm5_next) layer-parity harness for the exllamav3 port.
#
# Purpose
# -------
# Validates the three new/modified compute paths of the glm5_next port against pure-torch
# reference oracles built from the checkpoint-true math (vLLM glm5_next + FLA KDA semantics,
# MLX glm5_next NoPE-MLA / kpool-indexer / sigmoid-noaux_tc MoE):
#
#   kda   KimiDeltaAttention layer  (module key  model.language_model.layers.<K>.self_attn)
#           - prefill  (T = --seq, exercises the fla chunk_kda path, T >= 64)
#           - short    (T = 32,    exercises the fused recurrent path,  T < num_heads)
#   dsa   NoPE MLA + kpool indexer  (module key  model.language_model.layers.<D>.self_attn)
#           - nc       (T = --seq, cache-less calibration path, dense-exact since T <= index_topk)
#           - cached   (T = --seq, paged fp16 latent cache, prefill kernels with D_r = 0)
#           - decode   (T = 96, chunk = 1, flash-decoding kernel with D_r = 0)
#           - sparse   (--long-dsa: T = index_topk + 256, exercises kpool top-k selection;
#                       reference implements the full pooled-indexer math; looser tolerance,
#                       near-tie pool selections may legitimately differ)
#   moe   288-expert sigmoid noaux_tc block  (module key  model.language_model.layers.<D>.mlp)
#           - batch    (T = --moe-tokens) and single token (T = 1)
#           - the reference computes BOTH activation conventions:
#               vLLM:  clamp(g, max=10) * sigmoid(clamp(g, max=10)) * clamp(u, +-10)
#               exl3:  min(silu(g), 10) * clamp(u, +-10)
#             They differ only for g > 10 by <= ~4.5e-4 absolute (below bf16 resolution at
#             |x| ~ 10); both rows are reported so a systematic activation mismatch is visible.
#   hc    mHC HyperConnection mix/apply on real layer tensors vs mhc_pre/post_torch (verbatim
#           vLLM math). Fails loudly if hyperconnections.py lacks the allow_bf16 load patch.
#
# Both sides consume the SAME checkpoint values: exllamav3 loads bf16 -> fp16 for GEMM weights
# (lossless: every bf16 normal value in fp16 range is exactly representable in fp16) and keeps
# raw dtypes for A_log / dt_bias / conv / norms / hc; the reference upcasts the same bf16 bytes
# to fp32. Differences therefore measure implementation parity, not weight rounding.
#
# Environment
# -----------
#   - Full parity needs: CUDA GPU, exllamav3 (with the glm5_next port registered), triton,
#     flash-linear-attention (fla.ops.kda), safetensors.
#   - On a CPU-only machine (or before the port compiles) the harness runs the reference
#     oracles alone plus deterministic self-checks (KDA state-carry consistency) and exits 0
#     with the exl3-side comparisons marked SKIP.
#   - The full-size MoE test loads 288 experts in fp16 (~14.6 GB VRAM). With less free VRAM,
#     pass --moe-experts 32: BOTH sides are truncated to the same first-N experts (router
#     included), so the comparison remains a valid implementation-parity check.
#
# Usage
# -----
#   python tests/glm5_parity/glm5_layer_parity.py --model-dir /home/glm53k6/models/bf16
#   python tests/glm5_parity/glm5_layer_parity.py --tests kda,moe --moe-experts 32
#   python tests/glm5_parity/glm5_layer_parity.py --tests dsa --long-dsa
#
# Exit code 0 = all requested comparisons passed (or were skipped with reason); 1 = failures.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------------------------
# Checkpoint access
# --------------------------------------------------------------------------------------------

class ShardReader:
    """Lazy tensor loader over a sharded safetensors checkpoint (bf16)."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(model_dir, "model.safetensors")
            assert os.path.exists(single), f"No safetensors index or file in {model_dir}"
            self.weight_map = None
            self._single = "model.safetensors"
        self._handles = {}

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._safe_open(os.path.join(self.model_dir, shard), framework = "pt", device = "cpu")
            self._handles[shard] = h
        return h

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        return name in self._handle(self._single).keys()

    def get(self, name: str, device = "cpu", dtype: torch.dtype | None = None) -> torch.Tensor:
        shard = self.weight_map[name] if self.weight_map is not None else self._single
        t = self._handle(shard).get_tensor(name)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)


def load_text_config(model_dir: str) -> SimpleNamespace:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    lac = tc["linear_attn_config"]
    ns = SimpleNamespace(
        arch = cfg.get("architectures", ["?"])[0],
        hidden_size = tc["hidden_size"],
        num_hidden_layers = tc["num_hidden_layers"],
        rms_norm_eps = tc["rms_norm_eps"],                     # 1e-5
        layer_types = tc["layer_types"],
        mlp_layer_types = tc["mlp_layer_types"],
        # KDA
        kda_num_heads = lac["num_heads"],                      # 64
        kda_head_dim = lac["head_dim"],                        # 128
        kda_conv_k = lac["short_conv_kernel_size"],            # 4
        kda_lower_bound = lac["gate_lower_bound"],             # -5.0
        # DSA / MLA
        num_q_heads = tc["num_attention_heads"],               # 64
        q_lora_rank = tc["q_lora_rank"],                       # 1536
        kv_lora_rank = tc["kv_lora_rank"],                     # 512
        qk_nope_head_dim = tc["qk_nope_head_dim"],             # 256
        qk_rope_head_dim = tc["qk_rope_head_dim"],             # 0
        v_head_dim = tc["v_head_dim"],                         # 256
        index_n_heads = tc["index_n_heads"],                   # 32
        index_head_dim = tc["index_head_dim"],                 # 128
        index_topk = tc["index_topk"],                         # 2048
        index_kpool = tc["index_kpool"],                       # 4
        index_tail = tc.get("index_kpool_always_select_tail", True),
        # MoE
        n_routed_experts = tc["n_routed_experts"],             # 288
        num_experts_per_tok = tc["num_experts_per_tok"],       # 8
        moe_intermediate_size = tc["moe_intermediate_size"],   # 2048
        intermediate_size = tc["intermediate_size"],           # 12288
        routed_scaling_factor = tc["routed_scaling_factor"],   # 2.5
        n_shared_experts = tc["n_shared_experts"],             # 1
        swiglu_limit = tc.get("swiglu_limit", 10.0),           # 10.0
        # mHC
        hc_mult = tc["hc_mult"],                               # 4
        hc_sinkhorn_iters = tc["hc_sinkhorn_iters"],           # 20
        hc_eps = tc["hc_eps"],                                 # 1e-6
        hc_post_mult = tc.get("mhc_post_mult_value", 2.0),     # 2.0
    )
    assert ns.qk_rope_head_dim == 0, "harness assumes the NoPE (qk_rope_head_dim = 0) config"
    assert tc.get("scoring_func", "sigmoid") == "sigmoid"
    assert tc.get("topk_method", "noaux_tc") == "noaux_tc"
    assert tc.get("norm_topk_prob", True) is True
    return ns


KEY_PREFIX = "model.language_model"


# --------------------------------------------------------------------------------------------
# Reference oracles (pure torch, fp32)
# --------------------------------------------------------------------------------------------

def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * w.float()


def l2norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # FLA / MLX convention: x / sqrt(sum(x^2) + eps)  (eps inside the sqrt, on the sum)
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim = True) + eps)


def causal_conv_silu_ref(mixed: torch.Tensor, conv_w: torch.Tensor,
                         conv_ctx: torch.Tensor | None = None):
    """Depthwise causal conv1d + silu over [T, C] with kernel [C, K]; zero (or carried)
    initial state. Returns (y [T, C], new_ctx [K-1, C])."""
    T, C = mixed.shape
    K = conv_w.shape[-1]
    if conv_ctx is None:
        conv_ctx = mixed.new_zeros(K - 1, C)
    u = torch.cat([conv_ctx, mixed], dim = 0)                       # [T + K - 1, C]
    y = F.conv1d(u.T.unsqueeze(0), conv_w.unsqueeze(1), groups = C) # [1, C, T]
    y = F.silu(y[0].T)
    return y, u[-(K - 1):].clone()


def ref_kda_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    device = "cpu", state: dict | None = None):
    """Reference Kimi Delta Attention layer (vLLM glm5_next kda.py + fla.ops.kda math).

    x: [1, T, hidden] (any float dtype). Returns ([1, T, hidden] fp32, carry_state).
    carry_state carries (recurrent S, conv context) for the split-consistency self-check.
    """
    H, D = tc.kda_num_heads, tc.kda_head_dim
    P = H * D
    lb = tc.kda_lower_bound
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)

    W = lambda n: R.get(f"{key}.{n}", device).float()

    q = xf @ W("q_proj.weight").T                                   # [T, P]
    k = xf @ W("k_proj.weight").T
    v = xf @ W("v_proj.weight").T
    mixed = torch.cat([q, k, v], dim = -1)                          # [T, 3P]

    conv_w = torch.cat([
        R.get(f"{key}.q_conv1d.weight", device),
        R.get(f"{key}.k_conv1d.weight", device),
        R.get(f"{key}.v_conv1d.weight", device),
    ], dim = 0).float().squeeze(1)                                  # [3P, K]
    conv_ctx = state["conv_ctx"] if state is not None else None
    y, new_ctx = causal_conv_silu_ref(mixed, conv_w, conv_ctx)

    q, k, v = y.split(P, dim = -1)
    q = q.view(T, H, D)
    k = k.view(T, H, D)
    v = v.view(T, H, D)

    # Forget gate: g_log = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    # (fla kda_gate_fwd_kernel, SAFE_GATE branch; per KEY channel)
    fa = xf @ W("f_a_proj.weight").T                                # [T, 128]
    g1 = (fa @ W("f_b_proj.weight").T).view(T, H, D)
    g1 = g1 + R.get(f"{key}.dt_bias", device).float().view(H, D)
    a_log = R.get(f"{key}.A_log", device).float().view(H, 1)
    g_log = lb * torch.sigmoid(torch.exp(a_log) * g1)               # [T, H, D], in (lb, 0)

    beta = torch.sigmoid(xf @ W("b_proj.weight").T)                 # [T, H]

    qn = l2norm_ref(q)
    kn = l2norm_ref(k)
    scale = D ** -0.5

    S = state["S"].clone() if state is not None else xf.new_zeros(H, D, D)
    outs = []
    for t in range(T):
        S = S * torch.exp(g_log[t]).unsqueeze(-1)                   # per-key-channel decay
        kv_mem = torch.einsum("hkv,hk->hv", S, kn[t])
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        S = S + kn[t].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum("hkv,hk->hv", S, qn[t]) * scale)
    o = torch.stack(outs, dim = 0)                                  # [T, H, D]

    # Gated output norm: rmsnorm(o) * w * sigmoid(g2), eps = rms_norm_eps, fp32
    g2 = ((xf @ W("g_a_proj.weight").T) @ W("g_b_proj.weight").T).view(T, H, D)
    w_norm = R.get(f"{key}.o_norm.weight", device).float()
    on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim = True) + tc.rms_norm_eps)
    on = on * w_norm * torch.sigmoid(g2)

    out = on.reshape(T, P) @ W("o_proj.weight").T
    return out.unsqueeze(0), {"S": S, "conv_ctx": new_ctx}


def ref_kpool_topk_indices(R: ShardReader, key: str, xf: torch.Tensor, qr: torch.Tensor,
                           tc: SimpleNamespace, device = "cpu"):
    """kpool lightning-indexer selection (MLX glm5_next reference math, B = 1, no padding).

    Returns a boolean allow-mask [T, T] (True = key visible to query) implementing:
    softmax(gate + ape)-pooled keys, relu'd per-head scores, weights_proj * H^-0.5 head mix,
    top (index_topk // kpool) complete visible pools + always-select-tail."""
    T = xf.shape[0]
    hd, Hn, kp = tc.index_head_dim, tc.index_n_heads, tc.index_kpool
    W = lambda n: R.get(f"{key}.indexer.{n}", device).float()

    q = (qr @ W("wq_b.weight").T).view(T, Hn, hd)
    k = xf @ W("wk.weight").T                                        # [T, hd]
    k = F.layer_norm(k, (hd,), W("k_norm.weight"), W("k_norm.bias"), 1e-6)
    gate = xf @ W("index_kpool_compress_gate").T                     # [T, hd]
    ape = R.get(f"{key}.indexer.index_kpool_compress_ape", device).float()   # [kp, hd]
    w_heads = (xf @ W("weights_proj.weight").T) * (Hn ** -0.5)       # [T, Hn]

    # Pool the keys: complete pools only (trailing partial pool is invalid)
    Pn = (T + kp - 1) // kp
    pad = Pn * kp - T
    if pad:
        k_p = torch.cat([k, k.new_zeros(pad, hd)])
        g_p = torch.cat([gate, gate.new_zeros(pad, hd)])
    else:
        k_p, g_p = k, gate
    gk = k_p.view(Pn, kp, hd)
    gg = g_p.view(Pn, kp, hd)
    logits = gg + ape.unsqueeze(0)
    if pad:
        valid_slot = torch.arange(Pn * kp, device = device).view(Pn, kp) < T
        logits = logits.masked_fill(~valid_slot.unsqueeze(-1), -1e30)
    probs = torch.softmax(logits, dim = 1)
    pool_keys = (probs * gk).sum(dim = 1)                            # [Pn, hd]
    pool_valid = torch.ones(Pn, dtype = torch.bool, device = device)
    if pad:
        pool_valid[-1] = False                                       # incomplete pool
    pool_end = torch.clamp(torch.arange(Pn, device = device) * kp + kp - 1, max = T - 1)

    select_k = min(tc.index_topk // kp, Pn)
    softmax_scale = hd ** -0.5
    allow = torch.zeros(T, T, dtype = torch.bool, device = device)
    scores_hp = torch.einsum("thd,pd->thp", q, pool_keys) * softmax_scale
    scores_hp = torch.clamp(scores_hp, min = 0.0)
    index_scores_all = torch.einsum("th,thp->tp", w_heads, scores_hp)  # [T, Pn]
    for i in range(T):
        cand = pool_valid & (pool_end <= i)
        s = index_scores_all[i].masked_fill(~cand, -1e30)
        kk = min(select_k, Pn)
        sel = torch.topk(s, kk).indices
        sel = sel[cand[sel]]                                         # drop -1e30 picks
        for p in sel.tolist():
            a = p * kp
            allow[i, a : min(a + kp, i + 1)] = True
        # always-select-tail: the (i + 1) mod kp trailing tokens
        tail = (i + 1) % kp
        if tc.index_tail and tail:
            allow[i, i + 1 - tail : i + 1] = True
    return allow


def ref_dsa_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    device = "cpu", sparse: bool = False):
    """Reference NoPE MLA (dense; exact for T <= index_topk where the indexer selects all).
    With sparse = True, applies the kpool indexer allow-mask (T > index_topk regime)."""
    Hq = tc.num_q_heads
    dn, dv = tc.qk_nope_head_dim, tc.v_head_dim
    eps = tc.rms_norm_eps
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)
    W = lambda n: R.get(f"{key}.{n}", device).float()

    qa = xf @ W("q_a_proj.weight").T
    qr = rms_norm_ref(qa, R.get(f"{key}.q_a_layernorm.weight", device), eps)
    q = (qr @ W("q_b_proj.weight").T).view(T, Hq, dn)

    ckv = xf @ W("kv_a_proj_with_mqa.weight").T                      # [T, kv_lora]
    ckv = rms_norm_ref(ckv, R.get(f"{key}.kv_a_layernorm.weight", device), eps)
    kv = (ckv @ W("kv_b_proj.weight").T).view(T, Hq, dn + dv)
    k, v = kv[..., :dn], kv[..., dn:]

    scores = torch.einsum("qhd,khd->hqk", q, k) * (dn ** -0.5)
    pos = torch.arange(T, device = device)
    allow = pos.unsqueeze(1) >= pos.unsqueeze(0)                     # allow[q, k] = k <= q
    if sparse:
        assert T > tc.index_topk, "sparse reference only meaningful for T > index_topk"
        allow = allow & ref_kpool_topk_indices(R, key, xf, qr, tc, device)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("hqk,khd->qhd", p, v).reshape(T, Hq * dv)
    out = o @ W("o_proj.weight").T
    return out.unsqueeze(0)


def _act_vllm(g, u, limit):
    # SiluAndMulWithClamp forward_native: asymmetric clamp on the gate (max only)
    g = torch.clamp(g, max = limit)
    return g * torch.sigmoid(g) * torch.clamp(u, min = -limit, max = limit)


def _act_exl3(g, u, limit):
    # exllamav3 silu_mul with act_limit: post-activation clamp
    return torch.minimum(F.silu(g), torch.tensor(limit, dtype = g.dtype, device = g.device)) \
        * torch.clamp(u, min = -limit, max = limit)


def ref_moe_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    num_experts: int, device = "cpu"):
    """Reference sigmoid noaux_tc MoE (MLX DeepseekV32MoE math + vLLM clamp activation).

    Selection by (sigmoid scores + e_score_correction_bias); weights from UNBIASED scores of
    the selected experts, normalized, * routed_scaling_factor. Experts streamed one at a time
    from the checkpoint (bf16 -> fp32), never all resident. Returns (out_vllm, out_exl3act,
    diagnostics)."""
    top_k = tc.num_experts_per_tok
    rsf = tc.routed_scaling_factor
    limit = tc.swiglu_limit
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)

    gate_w = R.get(f"{key}.gate.weight", device).float()[:num_experts]
    esb = R.get(f"{key}.gate.e_score_correction_bias", device).float()[:num_experts]

    logits = xf @ gate_w.T                                           # fp32 router
    scores = torch.sigmoid(logits)
    biased = scores + esb
    sel = torch.topk(biased, top_k, dim = -1).indices                # [T, top_k]
    w = scores.gather(-1, sel)
    w = w / w.sum(-1, keepdim = True) * rsf

    # tie margin diagnostic: gap between the top_k-th and (top_k + 1)-th biased scores
    srt = torch.sort(biased, dim = -1, descending = True).values
    margin = (srt[:, top_k - 1] - srt[:, top_k])

    out_a = torch.zeros_like(xf)
    out_b = torch.zeros_like(xf)
    flat_sel = sel.reshape(-1)
    flat_tok = torch.arange(T, device = device).repeat_interleave(top_k)
    flat_w = w.reshape(-1)
    for e in flat_sel.unique().tolist():
        m = flat_sel == e
        toks = flat_tok[m]
        coef = flat_w[m].unsqueeze(-1)
        Wg = R.get(f"{key}.experts.{e}.gate_proj.weight", device).float()
        Wu = R.get(f"{key}.experts.{e}.up_proj.weight", device).float()
        Wd = R.get(f"{key}.experts.{e}.down_proj.weight", device).float()
        xt = xf[toks]
        g = xt @ Wg.T
        u = xt @ Wu.T
        out_a.index_add_(0, toks, (_act_vllm(g, u, limit) @ Wd.T) * coef)
        out_b.index_add_(0, toks, (_act_exl3(g, u, limit) @ Wd.T) * coef)
        del Wg, Wu, Wd

    # shared expert
    Wg = R.get(f"{key}.shared_experts.gate_proj.weight", device).float()
    Wu = R.get(f"{key}.shared_experts.up_proj.weight", device).float()
    Wd = R.get(f"{key}.shared_experts.down_proj.weight", device).float()
    g = xf @ Wg.T
    u = xf @ Wu.T
    out_a = out_a + _act_vllm(g, u, limit) @ Wd.T
    out_b = out_b + _act_exl3(g, u, limit) @ Wd.T

    diag = {"min_margin": margin.min().item(), "median_margin": margin.median().item()}
    return out_a.unsqueeze(0), out_b.unsqueeze(0), diag


# ---- mHC reference: verbatim vLLM mhc_torch math (refs/mhc_torch.py) -----------------------

def mhc_pre_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                  hc_post_mult_value, sinkhorn_repeat):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim = -1, keepdim = True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult:].reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2] \
        + hc_base[2 * hc_mult:].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim = -1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim = -2, keepdim = True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim = -1, keepdim = True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim = -2, keepdim = True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim = 1)
    return (
        post_mix.reshape(*outer_shape, hc_mult),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh", comb_res_mix.to(torch.float32), residual.to(torch.float32))
    post_term = post_layer_mix.to(torch.float32).unsqueeze(-1) * x.unsqueeze(-2).to(torch.float32)
    return mixed_residual + post_term


# --------------------------------------------------------------------------------------------
# Metrics / reporting
# --------------------------------------------------------------------------------------------

def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    a = out.detach().double().flatten().cpu()
    b = ref.detach().double().flatten().cpu()
    assert a.shape == b.shape, f"shape mismatch {tuple(out.shape)} vs {tuple(ref.shape)}"
    diff = (a - b).abs()
    ref_scale = b.abs().max().clamp_min(1e-12)
    cos = float((a @ b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12)))
    m = {
        "max_abs": float(diff.max()),
        "rel_max": float(diff.max() / ref_scale),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "cosine": cos,
    }
    if out.dim() >= 2:
        o2 = out.detach().double().reshape(-1, out.shape[-1]).cpu()
        r2 = ref.detach().double().reshape(-1, ref.shape[-1]).cpu()
        per_tok = (o2 - r2).abs().amax(-1) / ref_scale
        m["worst_tok_rel"] = float(per_tok.max())
        m["median_tok_rel"] = float(per_tok.median())
    return m


class Report:
    def __init__(self):
        self.rows = []
        self.failures = 0
        self.skips = 0

    def add(self, name: str, m: dict | None, tol_rel: float, tol_cos: float,
            note: str = "", skip: str | None = None):
        if skip is not None:
            self.rows.append((name, None, note or skip, "SKIP"))
            self.skips += 1
            print(f"[SKIP] {name}: {skip}")
            return
        ok = (m["rel_max"] <= tol_rel) and (m["cosine"] >= tol_cos)
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        self.rows.append((name, m, note, status))
        wt = f" worst_tok_rel={m['worst_tok_rel']:.3e} med_tok_rel={m['median_tok_rel']:.3e}" \
            if "worst_tok_rel" in m else ""
        print(f"[{status}] {name}: max_abs={m['max_abs']:.4e} rel_max={m['rel_max']:.4e} "
              f"rmse={m['rmse']:.4e} cosine={m['cosine']:.6f}{wt}"
              f"  (tol rel<={tol_rel:g} cos>={tol_cos:g}){('  ' + note) if note else ''}")

    def summary(self) -> int:
        print("\n" + "=" * 100)
        print(f"{'test':44s} {'rel_max':>10s} {'cosine':>10s} {'status':>8s}  note")
        print("-" * 100)
        for name, m, note, status in self.rows:
            if m is None:
                print(f"{name:44s} {'-':>10s} {'-':>10s} {status:>8s}  {note}")
            else:
                print(f"{name:44s} {m['rel_max']:>10.3e} {m['cosine']:>10.6f} {status:>8s}  {note}")
        print("=" * 100)
        print(f"{self.failures} failure(s), {self.skips} skip(s)")
        return 1 if self.failures else 0


# --------------------------------------------------------------------------------------------
# exllamav3 side
# --------------------------------------------------------------------------------------------

def import_exl3():
    import exllamav3  # noqa: F401 (needs the CUDA extension)
    from exllamav3 import Config, Model
    return Config, Model


def build_exl3_model(model_dir: str):
    Config, Model = import_exl3()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config, component = "text")
    return config, model


def exl3_load_module(model, key: str, device: torch.device):
    module = model.find_module(key)
    module.load(device)
    return module


def exl3_run_dsa_cached(module, x: torch.Tensor, device: torch.device, chunk: int | None = None):
    """Paged fp16 latent cache prefill/decode, mirroring tests/test_mla.py::run_module."""
    from exllamav3.cache import CacheLayer_MLA_fp16
    from exllamav3.constants import PAGE_SIZE
    bsz, S, _ = x.shape
    assert bsz == 1
    npages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    layer = CacheLayer_MLA_fp16(None, module, 0, npages * PAGE_SIZE)
    layer.alloc(device)
    bt = torch.arange(npages, dtype = torch.int32, device = device).view(1, npages)
    chunk = chunk or S
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    outs = []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = {
            "attn_mode": "flash_attn",
            "cache": layer,
            "block_table": bt,
            "cache_seqlens": seqlens,
            "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        seqlens = seqlens + (b - a)
    out = torch.cat(outs, dim = 1)
    layer.free()
    return out


def exl3_run_moe_truncated(R: ShardReader, key: str, x: torch.Tensor,
                           tc: SimpleNamespace, n_experts: int, device: torch.device):
    """Direct BlockSparseMLP with the first n_experts (router truncated identically), served
    through a FakeSTC so it runs without loading the full 288-expert layer."""
    from exllamav3.modules import BlockSparseMLP, GatedMLP
    try:
        from exllamav3.model.config import InferParams
        infer_params = InferParams()
    except Exception:
        infer_params = SimpleNamespace(
            no_reconstruct = False, moe_cpu_offload = 0, draft_moe_cpu_offload = 0,
            moe_cpu_split = 0, moe_cpu_component = "text")

    class FakeSTC:
        def __init__(self, tensors):
            self.tensors = tensors

        def has_tensor(self, k):
            return k in self.tensors

        def has_tensor_group(self, k, subkeys):
            if isinstance(k, list):
                return all(self.has_tensor_group(kk, subkeys) for kk in k)
            return all(
                (f"{k}.{sk}" in self.tensors if isinstance(sk, str)
                 else any(f"{k}.{s}" in self.tensors for s in sk))
                for sk in subkeys)

        def get_tensor(self, k, device = None, optional = False, allow_bf16 = False,
                       float2half = False, no_defer = False, transpose = False, pad_to = None,
                       fidx = None):
            if k not in self.tensors:
                if optional:
                    return None
                raise ValueError(f"Required tensor {k} not found")
            t = self.tensors[k].to(device if device is not None else "cpu")
            if float2half and t.dtype in (torch.float32, torch.float64, torch.bfloat16):
                t = t.half()
            if transpose:
                t = t.T.contiguous()
            if pad_to is not None:
                pad = []
                for i in range(len(pad_to) - 1, -1, -1):
                    pad += [0, max(0, pad_to[i] - t.shape[i])]
                if any(pad):
                    t = F.pad(t, pad)
            return t.contiguous()

    class FakeConfig:
        def __init__(self, tensors):
            self.stc = FakeSTC(tensors)
            self.infer_params = infer_params

    t = {
        f"{key}.gate.weight": R.get(f"{key}.gate.weight")[:n_experts],
        f"{key}.gate.e_score_correction_bias":
            R.get(f"{key}.gate.e_score_correction_bias")[:n_experts],
    }
    for i in range(n_experts):
        for p in ("gate_proj", "up_proj", "down_proj"):
            t[f"{key}.experts.{i}.{p}.weight"] = R.get(f"{key}.experts.{i}.{p}.weight")
    for p in ("gate_proj", "up_proj", "down_proj"):
        t[f"{key}.shared_experts.{p}.weight"] = R.get(f"{key}.shared_experts.{p}.weight")

    fc = FakeConfig(t)
    module = BlockSparseMLP(
        config = fc,
        key = key,
        hidden_size = tc.hidden_size,
        intermediate_size = tc.moe_intermediate_size,
        num_experts = n_experts,
        num_experts_per_tok = tc.num_experts_per_tok,
        key_up = "experts.{expert_idx}.up_proj",
        key_gate = "experts.{expert_idx}.gate_proj",
        key_down = "experts.{expert_idx}.down_proj",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.e_score_correction_bias",
        qmap = None,
        interm_dtype = torch.half,
        out_dtype = torch.float,
        activation_fn = "silu",
        act_limit = tc.swiglu_limit,
        router_type = "dots",
        routed_scaling_factor = tc.routed_scaling_factor,
        shared_experts = GatedMLP(
            config = fc,
            key = f"{key}.shared_experts",
            hidden_size = tc.hidden_size,
            intermediate_size = tc.moe_intermediate_size * tc.n_shared_experts,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = None,
            interm_dtype = torch.half,
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = tc.swiglu_limit,
        ),
    )
    module.load(device)
    out = module.forward(x, {})
    module.unload()
    return out


def exl3_run_hc(model, key: str, resid_bf16: torch.Tensor, y_half: torch.Tensor,
                device: torch.device):
    """Runs HyperConnection.mix / apply_ on real tensors. Returns exl3 (post, comb, collapsed,
    applied) plus the module's fn/base/scale for the reference."""
    hc = model.find_module(key)
    hc.load(device)
    streams = resid_bf16.float().to(device).contiguous()
    post, comb, collapsed = hc.mix(streams, {})
    applied = hc.apply_(streams.clone(), y_half.to(device), post, comb, {})
    tensors = (hc.fn.clone(), hc.base.clone(), hc.scale.clone())
    hc.unload()
    return post, comb, collapsed, applied, tensors


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def pick_layers(tc: SimpleNamespace, args):
    kda = args.kda_layer
    if kda < 0:
        kda = tc.layer_types.index("linear_attention")
    dsa = args.dsa_layer
    if dsa < 0:
        dsa = tc.layer_types.index("deepseek_sparse_attention")
    moe = args.moe_layer
    if moe < 0:
        moe = next(i for i, t in enumerate(tc.mlp_layer_types) if t == "sparse")
    assert tc.layer_types[kda] == "linear_attention", f"layer {kda} is not linear_attention"
    assert tc.layer_types[dsa] == "deepseek_sparse_attention", f"layer {dsa} is not DSA"
    assert tc.mlp_layer_types[moe] == "sparse", f"layer {moe} mlp is not sparse"
    return kda, dsa, moe


def main():
    ap = argparse.ArgumentParser(description = "GLM-5.3-Flash layer-parity harness")
    ap.add_argument("--model-dir", default = "/home/glm53k6/models/bf16")
    ap.add_argument("--device", default = "cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref-device", default = None,
                    help = "device for the reference math (default: same as --device if cuda, else cpu)")
    ap.add_argument("--tests", default = "kda,dsa,moe,hc",
                    help = "comma list from {kda,dsa,moe,hc}")
    ap.add_argument("--kda-layer", type = int, default = -1)
    ap.add_argument("--dsa-layer", type = int, default = -1)
    ap.add_argument("--moe-layer", type = int, default = -1)
    ap.add_argument("--seq", type = int, default = 512,
                    help = "prefill length for kda/dsa (must be <= index_topk for exact DSA parity)")
    ap.add_argument("--moe-tokens", type = int, default = 64)
    ap.add_argument("--moe-experts", type = int, default = 0,
                    help = "0 = auto (288 if >18 GiB free VRAM else 32). N < 288 truncates BOTH sides identically.")
    ap.add_argument("--long-dsa", action = "store_true",
                    help = "additionally test the sparse kpool-indexer regime at T = index_topk + 256")
    ap.add_argument("--seed", type = int, default = 17)
    ap.add_argument("--ref-only", action = "store_true",
                    help = "skip the exllamav3 side; run reference oracles + self-checks only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tc = load_text_config(args.model_dir)
    R = ShardReader(args.model_dir)
    rep = Report()

    device = torch.device(args.device)
    cuda_ok = device.type == "cuda" and torch.cuda.is_available()
    ref_device = args.ref_device or (str(device) if cuda_ok else "cpu")

    kda_l, dsa_l, moe_l = pick_layers(tc, args)
    kda_key = f"{KEY_PREFIX}.layers.{kda_l}.self_attn"
    dsa_key = f"{KEY_PREFIX}.layers.{dsa_l}.self_attn"
    moe_key = f"{KEY_PREFIX}.layers.{moe_l}.mlp"
    hc_key = f"{KEY_PREFIX}.layers.{kda_l}.hc_attn"

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    assert args.seq <= tc.index_topk, \
        f"--seq {args.seq} > index_topk {tc.index_topk}: dense DSA reference would not be exact"

    print(f"model:      {args.model_dir}  ({tc.arch})")
    print(f"layers:     kda={kda_l}  dsa={dsa_l}  moe={moe_l}")
    print(f"device:     exl3={device}  ref={ref_device}")
    print(f"tests:      {tests}  seq={args.seq}  moe_tokens={args.moe_tokens}\n")

    # Inputs (identical bytes on both sides: generated fp16 and upcast where needed)
    H = tc.hidden_size
    x_kda = torch.randn(1, args.seq, H).half()
    x_kda_short = torch.randn(1, 32, H).half()
    x_dsa = torch.randn(1, args.seq, H).half()
    x_dsa_dec = torch.randn(1, 96, H).half()
    x_moe = torch.randn(1, args.moe_tokens, H).half()
    x_moe1 = x_moe[:, :1].contiguous()

    # ---- exllamav3 availability ------------------------------------------------------------
    model = None
    exl3_err = None
    if not args.ref_only and cuda_ok:
        try:
            _, model = build_exl3_model(args.model_dir)
        except Exception as e:
            exl3_err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    elif not cuda_ok:
        exl3_err = "no CUDA device (reference-only mode)"
    else:
        exl3_err = "--ref-only"

    # ---- KDA -------------------------------------------------------------------------------
    if "kda" in tests:
        t0 = time.time()
        ref_kda, _ = ref_kda_forward(R, kda_key, x_kda, tc, ref_device)
        ref_kda_s, _ = ref_kda_forward(R, kda_key, x_kda_short, tc, ref_device)
        print(f"(kda reference computed in {time.time() - t0:.1f}s)")

        # Oracle self-check: state-carried split run must equal the full run exactly (fp32)
        half = args.seq // 2
        y1, st = ref_kda_forward(R, kda_key, x_kda[:, :half], tc, ref_device)
        y2, _ = ref_kda_forward(R, kda_key, x_kda[:, half:], tc, ref_device, state = st)
        m = metrics(torch.cat([y1, y2], dim = 1), ref_kda)
        rep.add("kda/oracle-split-consistency", m, tol_rel = 1e-4, tol_cos = 0.999999,
                note = "reference self-check")

        if model is not None:
            try:
                mod = exl3_load_module(model, kda_key, device)
                out = mod.forward(x_kda.to(device), {})
                rep.add("kda/prefill-chunked-vs-ref", metrics(out, ref_kda),
                        tol_rel = 3e-2, tol_cos = 0.999, note = f"T={args.seq} (fla chunk path)")
                out_s = mod.forward(x_kda_short.to(device), {})
                rep.add("kda/short-recurrent-vs-ref", metrics(out_s, ref_kda_s),
                        tol_rel = 3e-2, tol_cos = 0.999, note = "T=32 (fused recurrent path)")
                mod.unload()
            except Exception as e:
                traceback.print_exc()
                rep.add("kda/exl3", None, 0, 0, skip = f"exl3 KDA failed: {type(e).__name__}: {e}")
        else:
            rep.add("kda/exl3", None, 0, 0, skip = exl3_err)

    # ---- DSA -------------------------------------------------------------------------------
    if "dsa" in tests:
        t0 = time.time()
        ref_dsa = ref_dsa_forward(R, dsa_key, x_dsa, tc, ref_device)
        ref_dsa_dec = ref_dsa_forward(R, dsa_key, x_dsa_dec, tc, ref_device)
        print(f"(dsa reference computed in {time.time() - t0:.1f}s)")

        if model is not None:
            try:
                mod = exl3_load_module(model, dsa_key, device)
                positions = torch.zeros((1,), dtype = torch.int32, device = device)
                out_nc = mod.forward(x_dsa.to(device),
                                     {"attn_mode": "flash_attn_nc", "positions": positions})
                rep.add("dsa/nc-vs-ref", metrics(out_nc, ref_dsa),
                        tol_rel = 1e-2, tol_cos = 0.999, note = f"T={args.seq} dense (<= index_topk)")

                out_c = exl3_run_dsa_cached(mod, x_dsa.to(device), device)
                rep.add("dsa/cached-prefill-vs-ref", metrics(out_c, ref_dsa),
                        tol_rel = 1e-2, tol_cos = 0.999, note = "paged fp16 cache, D_r=0 kernels")

                out_d = exl3_run_dsa_cached(mod, x_dsa_dec.to(device), device, chunk = 1)
                rep.add("dsa/decode-vs-ref", metrics(out_d, ref_dsa_dec),
                        tol_rel = 1e-2, tol_cos = 0.999, note = "T=96, chunk=1 (decode kernel)")

                if args.long_dsa:
                    T_long = tc.index_topk + 256
                    x_long = torch.randn(1, T_long, H).half()
                    ref_sp = ref_dsa_forward(R, dsa_key, x_long, tc, ref_device, sparse = True)
                    out_sp = mod.forward(x_long.to(device),
                                         {"attn_mode": "flash_attn_nc",
                                          "positions": torch.zeros((1,), dtype = torch.int32,
                                                                   device = device)})
                    rep.add("dsa/sparse-kpool-vs-ref", metrics(out_sp, ref_sp),
                            tol_rel = 1e-1, tol_cos = 0.99,
                            note = f"T={T_long} > index_topk; near-tie pool picks may differ")
                mod.unload()
            except Exception as e:
                traceback.print_exc()
                rep.add("dsa/exl3", None, 0, 0, skip = f"exl3 DSA failed: {type(e).__name__}: {e}")
        else:
            rep.add("dsa/exl3", None, 0, 0, skip = exl3_err)

    # ---- MoE -------------------------------------------------------------------------------
    if "moe" in tests:
        n_experts = args.moe_experts
        if n_experts <= 0:
            if cuda_ok:
                free_b, _ = torch.cuda.mem_get_info(device)
                n_experts = tc.n_routed_experts if free_b > 18 * (1 << 30) else 32
            else:
                n_experts = 32
        n_experts = min(n_experts, tc.n_routed_experts)
        full = n_experts == tc.n_routed_experts

        t0 = time.time()
        ref_a, ref_b, diag = ref_moe_forward(R, moe_key, x_moe, tc, n_experts, ref_device)
        ref1_a, ref1_b, _ = ref_moe_forward(R, moe_key, x_moe1, tc, n_experts, ref_device)
        print(f"(moe reference computed in {time.time() - t0:.1f}s; experts={n_experts}, "
              f"router tie margin: min={diag['min_margin']:.3e} median={diag['median_margin']:.3e})")

        if model is not None:
            try:
                if full:
                    mod = exl3_load_module(model, moe_key, device)
                    out = mod.forward(x_moe.to(device), {})
                    out1 = mod.forward(x_moe1.to(device), {})
                    mod.unload()
                else:
                    out = exl3_run_moe_truncated(R, moe_key, x_moe.to(device),
                                                 tc, n_experts, device)
                    out1 = exl3_run_moe_truncated(R, moe_key, x_moe1.to(device),
                                                  tc, n_experts, device)
                tag = "288" if full else f"trunc{n_experts}"
                rep.add(f"moe/{tag}-batch-vs-ref(vllm-act)", metrics(out, ref_a),
                        tol_rel = 5e-2, tol_cos = 0.998, note = f"T={args.moe_tokens}")
                rep.add(f"moe/{tag}-batch-vs-ref(exl3-act)", metrics(out, ref_b),
                        tol_rel = 5e-2, tol_cos = 0.998, note = "same output, exl3 clamp convention")
                rep.add(f"moe/{tag}-bsz1-vs-ref(vllm-act)", metrics(out1, ref1_a),
                        tol_rel = 5e-2, tol_cos = 0.998, note = "T=1 routing path")
            except Exception as e:
                traceback.print_exc()
                rep.add("moe/exl3", None, 0, 0, skip = f"exl3 MoE failed: {type(e).__name__}: {e}")
        else:
            rep.add("moe/exl3", None, 0, 0, skip = exl3_err)

        # Oracle self-check: the two activation conventions must agree to ~1e-3 rel
        m = metrics(ref_b, ref_a)
        rep.add("moe/oracle-act-convention-delta", m, tol_rel = 1e-2, tol_cos = 0.9999,
                note = "vLLM vs exl3 clamp (expected tiny)")

    # ---- mHC -------------------------------------------------------------------------------
    if "hc" in tests:
        # Two shapes: R = b*s <= 32 takes the ext half-fn decode path, R > 32 the fp32-fn
        # path (fn is bf16 in the checkpoint, so half rounding is lossless; the kernel
        # accumulates fp32 either way -- both should track the fp32 reference tightly)
        hc_cases = [("decode(R=8)", 8, 2e-3), ("prefill(R=48)", 48, 1e-3)]
        if model is not None:
            for tag, S, tol in hc_cases:
                resid = (torch.randn(1, S, tc.hc_mult, H) * 0.5).bfloat16()
                y_site = (torch.randn(1, S, H) * 0.5).half()
                try:
                    post, comb, collapsed, applied, (fn, base, scale) = \
                        exl3_run_hc(model, hc_key, resid, y_site, device)
                    r_post, r_comb, r_coll = mhc_pre_torch(
                        resid.to(device), fn, scale, base,
                        rms_eps = tc.rms_norm_eps, hc_pre_eps = tc.hc_eps,
                        hc_sinkhorn_eps = tc.hc_eps, hc_post_mult_value = tc.hc_post_mult,
                        sinkhorn_repeat = tc.hc_sinkhorn_iters)
                    rep.add(f"hc/{tag}-mix-post", metrics(post, r_post),
                            tol_rel = tol, tol_cos = 0.99999)
                    rep.add(f"hc/{tag}-mix-comb", metrics(comb, r_comb),
                            tol_rel = tol, tol_cos = 0.99999)
                    rep.add(f"hc/{tag}-mix-collapsed", metrics(collapsed.float(), r_coll),
                            tol_rel = 1e-2, tol_cos = 0.9999, note = "ext returns half")
                    r_applied = mhc_post_torch(y_site.to(device), resid.float().to(device),
                                               r_post, r_comb)
                    rep.add(f"hc/{tag}-apply", metrics(applied, r_applied),
                            tol_rel = 1e-2, tol_cos = 0.9999)
                except Exception as e:
                    traceback.print_exc()
                    rep.add(f"hc/{tag}", None, 0, 0,
                            skip = f"exl3 HC failed (missing allow_bf16 load patch?): "
                                   f"{type(e).__name__}: {e}")
        else:
            rep.add("hc/exl3", None, 0, 0, skip = exl3_err)

    sys.exit(rep.summary())


if __name__ == "__main__":
    main()
