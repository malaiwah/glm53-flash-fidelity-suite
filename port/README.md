# exllamav3 glm5_next port — design bundle (pre-implementation)

Design + first-draft for running GLM-5.3-Flash natively on exllamav3 (target:
first K6 trellis quant, scored on the fidelity suite in this repo). Produced
2026-08-27 by a 7-agent workflow against exllamav3 v1.4.4, before any
implementation session. Status: **not yet implemented or GPU-tested** — this is
the blueprint the K6 session starts from.

Core finding: exllamav3 v1.4.4 already ships ~80% of what glm5_next needs
(DeepSeek-V4 mHC hyper-connections verified numerically identical to vLLM's,
`glm_moe_dsa` MLA/indexer/noaux_tc-MoE skeleton, GDN recurrent-cache
machinery). New code required: a KimiDeltaAttention module (safe-gate KDA, not
GDN), a kpool-compressed indexer mode, NoPE (rope_dim=0) guards, a ~20-line
mean ContractStreams, and a sigmoid option in GatedRMSNorm. Estimate ~72
expert-hours + 12–20 h GPU conversion.

| File | What it is |
|---|---|
| [BLUEPRINT.md](BLUEPRINT.md) | Full port blueprint: file-by-file plan, config asserts, tensor-key mapping, what to keep unquantized, hardest-3 risks |
| [glm5_next.py.draft](glm5_next.py.draft) | Syntax-checked first draft of `exllamav3/architecture/glm5_next.py` (PORT-CHECK tags mark unresolved deps) |
| [DRAFT-NOTES.md](DRAFT-NOTES.md) | What the draft assumes and which plan items must land first |
| [tests/glm5_layer_parity.py](tests/glm5_layer_parity.py) | Per-layer torch-oracle parity harness (KDA / NoPE-MLA / noaux_tc MoE / mHC), smoke-tested on a synthetic mini-checkpoint |
| [tests/mini_ckpt_test.py](tests/mini_ckpt_test.py) | Builds the synthetic mini-checkpoint the harness self-checks against |
| [PARITY.md](PARITY.md) | Harness design + measured self-check numbers |
| [REVIEW.md](REVIEW.md) | Adversarial review — **read first**: one blocker (fla version floor can silently accept a fla without the SAFE gate), missing-module dependency list, harness defects |

Context: brandonmusic's 4bpw EXL3 (custom Transformers TP2 adapter +
exllamav3 kernels) proves the quant path works today without this native port;
this bundle is the road to a self-contained exllamav3 architecture and the
K6/K5 variants. See JOURNAL.md at repo root for the campaign log.
