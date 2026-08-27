#!/usr/bin/env python3
"""Build a synthetic mini GLM-5.3-Flash checkpoint (same tensor names / config structure,
tiny dims) and smoke-test the parity harness reference oracles."""
import json, os, sys, torch
from safetensors.torch import save_file

root = os.path.dirname(os.path.abspath(__file__))
mini = os.path.join(root, "mini_glm5")
os.makedirs(mini, exist_ok = True)

H = 64            # hidden
KH, KD, CK = 4, 8, 4          # kda heads, head_dim, conv kernel
QLR, KVLR, NOPE, VH, NQ = 32, 16, 16, 16, 4
INH, IHD, ITOPK, KP = 4, 8, 16, 4
NE, TOPK_E, MI, II = 8, 4, 32, 128

cfg = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": {
        "model_type": "glm5_next_text",
        "hidden_size": H,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "linear_attn_config": {"num_heads": KH, "head_dim": KD,
                               "short_conv_kernel_size": CK, "gate_lower_bound": -5.0},
        "num_attention_heads": NQ,
        "q_lora_rank": QLR, "kv_lora_rank": KVLR,
        "qk_nope_head_dim": NOPE, "qk_rope_head_dim": 0, "v_head_dim": VH,
        "index_n_heads": INH, "index_head_dim": IHD, "index_topk": ITOPK,
        "index_kpool": KP, "index_kpool_always_select_tail": True,
        "n_routed_experts": NE, "num_experts_per_tok": TOPK_E,
        "moe_intermediate_size": MI, "intermediate_size": II,
        "routed_scaling_factor": 2.5, "n_shared_experts": 1, "swiglu_limit": 10.0,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    },
}
with open(os.path.join(mini, "config.json"), "w") as f:
    json.dump(cfg, f)

torch.manual_seed(3)
def r(*s, scale = 0.08, dtype = torch.bfloat16):
    return (torch.randn(*s) * scale).to(dtype)

p = "model.language_model.layers"
t = {}
# layer 0: KDA
k0 = f"{p}.0.self_attn"
P = KH * KD
t[f"{k0}.q_proj.weight"] = r(P, H)
t[f"{k0}.k_proj.weight"] = r(P, H)
t[f"{k0}.v_proj.weight"] = r(P, H)
t[f"{k0}.o_proj.weight"] = r(H, P)
t[f"{k0}.b_proj.weight"] = r(KH, H)
t[f"{k0}.f_a_proj.weight"] = r(KD, H)
t[f"{k0}.f_b_proj.weight"] = r(P, KD)
t[f"{k0}.g_a_proj.weight"] = r(KD, H)
t[f"{k0}.g_b_proj.weight"] = r(P, KD)
t[f"{k0}.q_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
t[f"{k0}.k_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
t[f"{k0}.v_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
t[f"{k0}.A_log"] = torch.randn(KH).float() * 0.5
t[f"{k0}.dt_bias"] = torch.randn(P).float() * 0.5
t[f"{k0}.o_norm.weight"] = (torch.randn(KD) * 0.1 + 1).bfloat16()
# layer 1: DSA
k1 = f"{p}.1.self_attn"
t[f"{k1}.q_a_proj.weight"] = r(QLR, H)
t[f"{k1}.q_a_layernorm.weight"] = (torch.randn(QLR) * 0.1 + 1).bfloat16()
t[f"{k1}.q_b_proj.weight"] = r(NQ * NOPE, QLR)
t[f"{k1}.kv_a_proj_with_mqa.weight"] = r(KVLR, H)
t[f"{k1}.kv_a_layernorm.weight"] = (torch.randn(KVLR) * 0.1 + 1).bfloat16()
t[f"{k1}.kv_b_proj.weight"] = r(NQ * (NOPE + VH), KVLR)
t[f"{k1}.o_proj.weight"] = r(H, NQ * VH)
t[f"{k1}.indexer.wq_b.weight"] = r(INH * IHD, QLR)
t[f"{k1}.indexer.wk.weight"] = r(IHD, H)
t[f"{k1}.indexer.k_norm.weight"] = (torch.randn(IHD) * 0.1 + 1).float()
t[f"{k1}.indexer.k_norm.bias"] = (torch.randn(IHD) * 0.05).float()
t[f"{k1}.indexer.weights_proj.weight"] = r(INH, H)
t[f"{k1}.indexer.index_kpool_compress_gate"] = r(IHD, H)
t[f"{k1}.indexer.index_kpool_compress_ape"] = torch.randn(KP, IHD).float() * 0.1
# layer 1 MoE
m1 = f"{p}.1.mlp"
t[f"{m1}.gate.weight"] = r(NE, H)
t[f"{m1}.gate.e_score_correction_bias"] = torch.randn(NE).float() * 0.1
for i in range(NE):
    t[f"{m1}.experts.{i}.gate_proj.weight"] = r(MI, H)
    t[f"{m1}.experts.{i}.up_proj.weight"] = r(MI, H)
    t[f"{m1}.experts.{i}.down_proj.weight"] = r(H, MI)
t[f"{m1}.shared_experts.gate_proj.weight"] = r(MI, H)
t[f"{m1}.shared_experts.up_proj.weight"] = r(MI, H)
t[f"{m1}.shared_experts.down_proj.weight"] = r(H, MI)
# hc tensors (layer 0)
hc3 = 2 * 4 + 16
t[f"{p}.0.hc_attn_fn"] = r(hc3, 4 * H, scale = 0.05)
t[f"{p}.0.hc_attn_base"] = torch.randn(hc3).bfloat16() * 0.1
t[f"{p}.0.hc_attn_scale"] = (torch.randn(3) * 0.1 + 1).bfloat16()

save_file(t, os.path.join(mini, "model.safetensors"))
wm = {k: "model.safetensors" for k in t}
with open(os.path.join(mini, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": wm}, f)
print("mini checkpoint written:", mini, f"({len(t)} tensors)")

# ---- run the harness reference-only -------------------------------------------------------
sys.argv = ["glm5_layer_parity.py", "--model-dir", mini, "--ref-only",
            "--seq", "16", "--moe-tokens", "8", "--moe-experts", "8",
            "--kda-layer", "0", "--dsa-layer", "1", "--moe-layer", "1", "--device", "cpu"]
sys.path.insert(0, os.path.join(root, "exllamav3-src", "tests", "glm5_parity"))
import glm5_layer_parity as hp

# direct oracle exercises beyond main(): sparse indexer ref + hc refs
tc = hp.load_text_config(mini)
R = hp.ShardReader(mini)
x_long = torch.randn(1, ITOPK + KP * 3, H).half()
out_sparse = hp.ref_dsa_forward(R, k1, x_long, tc, "cpu", sparse = True)
out_dense = hp.ref_dsa_forward(R, k1, x_long, tc, "cpu", sparse = False)
assert torch.isfinite(out_sparse).all()
d = (out_sparse - out_dense).abs().max().item()
print(f"sparse-vs-dense ref (T > topk, should differ a little): max abs diff {d:.3e}")
assert d > 0, "sparse mask had no effect?"
# sparse ref must equal dense ref when T <= index_topk is emulated by select_k covering all
x_short = torch.randn(1, ITOPK, H).half()
out_dense_s = hp.ref_dsa_forward(R, k1, x_short, tc, "cpu", sparse = False)
assert torch.isfinite(out_dense_s).all()

resid = torch.randn(1, 4, 4, H).bfloat16()
fn = R.get(f"{p}.0.hc_attn_fn").float()
base = R.get(f"{p}.0.hc_attn_base").float()
scale = R.get(f"{p}.0.hc_attn_scale").float()
post, comb, coll = hp.mhc_pre_torch(resid, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
# comb must be near doubly-stochastic after 20 sinkhorn iters
rows = comb.sum(-1); cols = comb.sum(-2)
print("hc comb row-sum range:", rows.min().item(), rows.max().item(),
      "col-sum range:", cols.min().item(), cols.max().item())
y = torch.randn(1, 4, H).half()
applied = hp.mhc_post_torch(y, resid.float(), post, comb)
assert applied.shape == (1, 4, 4, H)
print("direct oracle checks OK\n")

try:
    hp.main()
except SystemExit as e:
    print("harness exit code:", e.code)
    sys.exit(e.code)
