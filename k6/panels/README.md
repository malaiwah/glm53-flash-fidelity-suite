# Panels built by this suite

A panel is a *yardstick*, and a yardstick belongs to the model family it was
built for. `k6/tools/hf_capture.py` consumes the upstream
`quant-pipeline.glm53-token-panel.v1` layout (`panel.json` + `arrays/`), and
every panel we had in that layout was built for GLM-5.3-Flash by somebody
else's pipeline. Reusing one against a different model is the cross-model
comparison our own rules forbid: the token ids may be numerically valid, but
the panel was selected for another model's corpus and another model's
calibration separation, and a number measured on it invites being ranked
against numbers it has no business being ranked against.

So a model family that needs its own yardstick gets its own panel, with its
own `panel_id`, built by `k6/tools/build_token_panel.py` — a rule anyone can
re-run, with no RNG anywhere in it.

| panel_id | model family | shape | corpus |
|---|---|---|---|
| `panel--minimaxm3.malaiwah.corpus5x5` | `minimax_m3_vl` (vocab 200,064) | 5 strata x 5 windows, 2048 ctx, 2047 scored -> **51,175 positions** | `malaiwah/qwen38-27b-fidelity-suite-v5` @ `7797fcce`, `corpus/text/` |

The MiniMax panel is deliberately the **same shape** as
`panel--glm53.brandonmusic.final25` (25 x 2047 = 51,175 scored positions), so
statistical power is comparable across families even though the panels
themselves are not interchangeable and rows measured on them never share a
comparability group.

## Reproducing

Every panel here is rebuildable from its own `panel.receipt.json`, which
records the corpus repository AND revision, the per-document sha256, the
tokenizer file digests, and the selection rule in full. Rebuild and compare
`suite_token_hash_sha256`.

## What the builder does NOT check

`separation.checked` is `false` and says so in the receipt: this tool runs no
lexical or n-gram scan against a quantizer's calibration corpus. Panel /
calibration separation is a DECLARATION, not a verified property, and any
measurement whose artifact was calibrated on overlapping text must carry that
as a disclosure rather than relying on the panel to have excluded it.
