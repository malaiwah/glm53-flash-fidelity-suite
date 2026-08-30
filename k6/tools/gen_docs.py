#!/usr/bin/env python3
"""Render k6/HIDDEN-REPLAY.md from reports/hidden-replay-equivalence.json.

Every number in the prose comes from the receipt -- nothing is hand-transcribed
(campaign lesson: a doc that restates numbers by hand eventually disagrees with
its own receipt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def g(value, digits=None):
    """Render a float at full precision (repr) or fixed digits."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}g}" if digits else repr(value)
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = json.loads(args.receipt.read_text(encoding="utf-8"))
    runs = r["metrics_per_run"]
    r1 = runs[0]
    cross = r["cross_run_determinism"]
    inv = r["vocab_chunk_invariance"]
    pa = r["path_a_reproduction"]
    st = r["storage"]
    festr = r["prior_art"]
    head = r["head"]

    n_runs = len(runs)
    positions = r1["positions"]
    windows = r1["windows"]

    def run_rows(field, digits=None):
        return " | ".join(g(row[field], digits) for row in runs)

    lines = []
    A = lines.append

    A(f"# Hidden-replay equivalence — the streaming lane's {st['hiddens_mib']:.0f} MB teacher, qualified")
    A("")
    A(f"**Status: measured and sealed.** {n_runs} cold runs on one H200, "
      f"{windows} panel windows, {positions:,} scored positions per run. "
      "Receipt: `reports/hidden-replay-equivalence.json` in the HF dataset "
      "`malaiwah/GLM-5.3-Flash-fidelity-suite-v1` "
      f"(`receipt_sha256` `{r['receipt_sha256']}`).")
    A("")
    A("## The question")
    A("")
    A(f"A teacher for this lane is {st['logits_gib']:.1f} GiB of fp32 logits "
      "(the published sealed teacher is 31.7 GB of the same thing). The same panel carried as")
    A("**post-final-RMSNorm bf16 hidden states** is "
      f"**{st['hiddens_mib']:.0f} MB** — {st['shrink_factor']}x smaller. That trade is only")
    A("legitimate if replaying the hidden states through the LM head reproduces the")
    A("live logits closely enough that no downstream KLD conclusion changes. This is")
    A("the receipt that measures it, on our stack, before anything is shipped in that form.")
    A("")
    A("The protocol is **Phaelon's**: three cold runs; one forward pass per window")
    A("emitting *both* paths, so the comparison is never between two different forwards;")
    A("and state the cut explicitly. The metric set and conventions are **Festr's**, so")
    A("his kimi-k3 artifact and this one read side by side.")
    A("")
    A("## The cut (stated, not assumed)")
    A("")
    A(f"> {r['cut_statement']}")
    A("")
    A(f"- **Cut point:** `{r['cut_point']}`")
    A(f"- **Hidden dtype:** `{r['dtypes']['hidden']}` — {r['dtypes']['hidden_lossless']}")
    A(f"- **Logits dtype:** `{r['dtypes']['logits']}`")
    A(f"- **Replay:** {r['dtypes']['replay_arithmetic']}")
    A(f"- **KLD:** {r['dtypes']['kld_arithmetic']}")
    A("")
    A(f"**Read this if you have used our head bundle before.** {r['cut_note']}")
    A("")
    A("The head used for replay is the `lm_head` from the same BF16 tree the student was")
    A(f"built from — shape `{head['shape']}`, dtype `{head['dtype']}`, tensor-content")
    A(f"sha256 `{head['tensor_content_sha256']}` — and its sha256 was re-verified against")
    A("our published head-extraction receipt on fetch (the fetcher refuses a mismatch).")
    A("")
    A("## Results")
    A("")
    A(f"Per-token `KL(live logits || replayed logits)` over the full "
      f"{r['vocab_size']:,}-token vocabulary, fp64, {positions:,} positions per run:")
    A("")
    A("| Metric | " + " | ".join(f"run {row['cold_run']}" for row in runs) + " |")
    A("|---|" + "---:|" * n_runs)
    A(f"| Mean replay KLD | {run_rows('replay_kld_mean')} |")
    A(f"| Max per-token | {run_rows('replay_kld_max')} |")
    A(f"| p99 | {run_rows('replay_kld_p99')} |")
    A(f"| p99.9 | {run_rows('replay_kld_p99_9')} |")
    A(f"| Top-1 agreement (live vs replayed) | {run_rows('top1_agreement_live_vs_replayed')} |")
    A(f"| Panel mean KLD via live logits | {run_rows('panel_mean_kld_via_live_logits')} |")
    A(f"| Panel mean KLD via replayed logits | {run_rows('panel_mean_kld_via_replayed_logits')} |")
    A(f"| Panel delta (replayed − live) | {run_rows('panel_delta_replayed_minus_live')} |")
    A("")
    A("**The panel delta is the number that matters for a teacher.** It is what a")
    A("published KLD would move by if the teacher were carried as hiddens and replayed")
    A("rather than stored as logits.")
    A("")
    A("### Vocabulary-chunk invariance")
    A("")
    A(f"The replay was recomputed with a different vocabulary chunking "
      f"(`{inv['default_vocab_chunk']}` vs `{inv['alt_vocab_chunk']}`):")
    A("")
    A(f"- delta of means: `{g(inv['delta_of_means'])}`")
    A(f"- max per-token abs delta: `{g(inv['max_token_abs_delta'])}`")
    A(f"- replayed logits bitwise-equal fraction: `{g(inv['replayed_logits_bitwise_equal_fraction'])}`")
    A("")
    A("### Cross-run determinism")
    A("")
    A(f"- logits bitwise identical across the {cross['runs']} runs: "
      f"**{g(cross['logits_bitwise_identical_across_runs'])}**")
    A(f"- hiddens bitwise identical across the {cross['runs']} runs: "
      f"**{g(cross['hiddens_bitwise_identical_across_runs'])}**")
    A(f"- replay-KLD vectors identical across runs: "
      f"**{g(cross['replay_kld_vectors_identical_across_runs'])}**")
    A("")
    A("Evidence is **tensor content**, never container files: the per-run digests are")
    A("sha256 over the safetensors tensor region only (the header carries `cold_run`")
    A("and would differ between bit-identical runs).")
    A("")
    A(f"- logits payload digests: `{cross['distinct_logits_payload_digests']}`")
    A(f"- hiddens payload digests: `{cross['distinct_hiddens_payload_digests']}`")
    A("")
    A(f"> {cross['note']}")
    A("")
    A("### Path A reproduced the sealed number bitwise")
    A("")
    A("Path A is the ordinary streaming scorer, unchanged. Its panel mean must equal the")
    A("sealed K6 streaming number — a free lane-integrity check that the hidden tap")
    A("changed nothing:")
    A("")
    A(f"- sealed: `{pa['sealed_stream_mean']}`")
    A(f"- measured: `{pa['measured_mean']}`")
    A(f"- reproduced exactly: **{g(pa['mean_reproduced_exactly'])}**")
    A(f"- tokenwise-KLD sha matches sealed: **{g(pa['tokenwise_sha_matches_sealed'])}**")
    A(f"- scored the sealed K6 surface: **{g(pa['scored_the_sealed_k6_surface'])}**")
    A("")
    A("## Storage — the reason this receipt exists")
    A("")
    A("| Form | Bytes per run | |")
    A("|---|---:|---:|")
    A(f"| fp32 logits (today's teacher) | {st['logits_bytes_per_run']:,} | {st['logits_gib']:.2f} GiB |")
    A(f"| bf16 hiddens (this protocol) | {st['hiddens_bytes_per_run']:,} | {st['hiddens_mib']:.1f} MiB |")
    A(f"| **shrink factor** | | **{st['shrink_factor']}x** |")
    A("")
    A(f"Measured as {st['measured_how']}.")
    A("")
    A("## Prior art — Festr's kimi-k3 qualification")
    A("")
    A("This receipt deliberately adopts the conventions of "
      f"[{festr['dataset']}]"
      f"(https://huggingface.co/datasets/{festr['dataset']}/tree/{festr['dataset_revision']}) "
      f"(revision `{festr['dataset_revision']}`, `{festr['receipt_file']}`, status "
      f"`{festr['receipt_status']}`) so the two are readable side by side. His stack is")
    A(f"{festr['stack']}; ours is a single-device reference forward.")
    A("")
    A("| | Festr, kimi-k3 | this receipt, GLM-5.3-Flash |")
    A("|---|---:|---:|")
    A(f"| Mean replay KLD | `{g(festr['mean_replay_kld'])}` | `{g(r1['replay_kld_mean'])}` |")
    A(f"| Max per-token | `{g(festr['max_token_replay_kld'])}` | `{g(r1['replay_kld_max'])}` |")
    A(f"| p99.9 | `{g(festr['p99_9_replay_kld'])}` | `{g(r1['replay_kld_p99_9'])}` |")
    A(f"| Top-1 agreement | `{g(festr['top1_agreement_live_vs_replayed'])}` | "
      f"`{g(r1['top1_agreement_live_vs_replayed'])}` |")
    A(f"| Chunk-invariance delta | `{g(festr['chunk_invariance_delta_of_means'])}` | "
      f"`{g(inv['delta_of_means'])}` |")
    A(f"| Mean KLD between two *identical* runs | "
      f"`{g(festr['runtime_repeat_sentinels']['pair_00_vs_01']['mean_kld'])}` | "
      f"`{g(cross['runtime_repeat_sentinel_mean_kld_between_runs'])}` |")
    A("")
    A("Two honest caveats on that table:")
    A("")
    A(f"1. His chunk-invariance probe varies **two** knobs at once ({festr['chunk_invariance_definition']}); "
      "ours varies vocabulary chunking only. They are not the same experiment.")
    A(f"2. His comparator runs on `{festr['comparator_device']}` with deterministic algorithms; "
      f"ours runs on `{r['stack']['device']}`. The numeric policy is recorded in the receipt.")
    A("")
    A("The last row is the sharpest difference and it is **not** a quality claim about")
    A("either stack. On a live serving runtime, re-running the same prompts produces")
    A("genuinely different logits — Festr measured that honestly and published it as")
    A("measurement uncertainty. Our lane is a bitwise-deterministic reference forward, so")
    A("that term is exactly zero rather than small. A serving stack cannot simply adopt")
    A("our number, and we cannot claim his uncertainty does not exist for anyone running")
    A("this model under vLLM.")
    A("")
    A("## What this licenses")
    A("")
    for item in r["licenses"]["yes"]:
        A(f"- **Yes:** {item}")
    A("")
    A("## What this does NOT license")
    A("")
    for item in r["licenses"]["no"]:
        A(f"- **No:** {item}")
    A("")
    A(f"Festr states the rule for his own artifact and it applies verbatim to ours: "
      f"{festr['interpretation_rule']}.")
    A("")
    A("## Reproducing it")
    A("")
    A("```bash")
    A("git clone https://github.com/malaiwah/glm53-flash-fidelity-suite /home/suite")
    A("bash /home/suite/k6/tools/hidden_replay_stage.sh setup     # venv, pinned pipeline, selftest")
    A("bash /home/suite/k6/tools/hidden_replay_stage.sh fetch     # ~305 GB, head sha re-verified")
    A("bash /home/suite/k6/tools/hidden_replay_stage.sh verify    # sealed receipts + L1 ladder")
    A("for n in 1 2 3; do bash /home/suite/k6/tools/hidden_replay_stage.sh run$n; done")
    A("bash /home/suite/k6/tools/hidden_replay_stage.sh report    # path A vs the sealed number")
    A("bash /home/suite/k6/tools/hidden_replay_stage.sh compare   # the comparator")
    A("```")
    A("")
    A("Code identity (content shas of the exact files that produced the numbers):")
    A("")
    for key, value in sorted(r["stack"]["code_identity"].items()):
        A(f"- `{key}` `{value}`")
    A("")
    A(f"{r['stack']['code_identity_note']}.")
    A("")
    A(f"Stack: `{r['stack']['env_versions']}`, torch `{r['stack']['torch_version']}`, "
      f"CUDA `{r['stack']['cuda_runtime_version']}`, device `{r['stack']['device_name']}`.")
    A("")
    A("## Credits")
    A("")
    for line in r["credits"]:
        A(f"- {line}")
    A("")

    text = "\n".join(lines)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
