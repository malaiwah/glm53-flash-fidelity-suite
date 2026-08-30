# Discord reply — paste-ready

Ten messages, each under Discord's 2000-character limit, in order. Message 10 is
optional (it is the licence question) and can be sent as a DM instead.

Full working: `docs/PROTOCOL-ALIGNMENT.md` in
`github.com/malaiwah/glm53-flash-fidelity-suite`.

---

## Message 1 — what we took

Read the whole harness, ran your tests, reproduced your numbers. Your protocol is
better than ours on most of it, so we adopted it rather than argue. What we took,
and where each one honestly stands:

**Live in our tooling and data:** window-clustered block bootstrap with BCa (12
rows carry intervals now — all 16 rows we'd published on your two panels had
`uncertainty: none`); per-domain tables, on 12 rows, which we'd been computing
in every report and publishing in none; the 13-gram overlap scan; `sigma_run`
beside the SE and in quadrature, on 8 rows; paired-difference ranking with BCa and
a sign test; one frozen protocol file, hashed into every receipt the new tooling
emits.

**Adopted in code, not yet in our results — worth being clear about:** McNemar
runs and matches your five published p-values, but we can't apply it to our own
rows at all; a contingency table needs per-position top-1 agreement and per-window
means can't supply one. Same for the percentile guard: our receipts have no
pooled percentiles for it to guard, so the verb returns a refusal. And R0 fires on
every case in our selftest and gates the new CLI, but it is not yet wired into our
production capture path.

One correction in your favour, since you can check it: an earlier draft said
*every* row we publish had no interval. Not true — elsewhere in the registry 39
of 60 rows already carried a context-cluster bootstrap. Your panel was the hole,
not the whole registry.

Validation, from your 25 per-window means alone: all 16 of your published
percentile and BCa endpoints to within one ULP (max |diff| 2.8e-17); three of
your four `se_clustered_window` bit-identical and the fourth to one ULP
(1.7e-18); your paired diff and ratio CIs exactly; all five McNemar p-values to
1.5e-14 relative. Different OS, numpy 2.5.2 vs your 1.26.4. The same input
through your own `kld_eval.analysis.stats` agrees to 7e-18, and your 16 unit
tests pass unmodified on a fresh macOS venv.

## Message 2 — the padded columns, bounded on your teacher

You mask the 24 padded lm_head columns. We never have. Rather than guess, we
measured it.

Downloaded your teacher window final-0000 (1.27 GB, sha256 verified). Straight
about what's real: the teacher numbers come out of your tensor; the students
don't. A real delta for K6 or FP8 needs their student logits on your panel, i.e.
a GPU run we didn't do, so we reconstructed the hidden states by least squares
against a real lm_head (rel. rms residual 1.6e-3) and built thirteen *synthetic*
students on top, mean KLD 4.8e-5 to 1.0 nats, to stress it.

The teacher side carries the answer. The padded rows aren't dead: norm ~0.4795 vs
~1.21 for a typical real row, all 24 mutually cosine-0.999998 — one untrained
direction repeated. They hold ~1.6e-8 of the probability mass. What that caps
depends on the case, and the two cases are four orders apart: in general the
delta is that mass times however many nats the student's padded logits are
displaced, so ~1e-8, never past ~1e-7. Share the head and the displacement is
zero and it collapses to KLD x mass — that's the 1e-10, and every row of ours on
your panel is shared-head.

The synthetic students agree: +7.2e-9 to +7.4e-9 relative across shared BF16, RTN
per-row int8/int6/int4 and group-128 affine 6b/4b — including a deliberately
awful global-scale int4 head displacing the padded logits +2.1 nats and blowing
KLD to 0.183. Even there it moves 5e-8 nats.

Every published number of ours moves at the 8th or 9th significant figure,
nowhere earlier: K6 sealed 0.013723384665701147 -> 0.013723384767254605. No
correction, no bias disclosure, just a field recording the convention. For scale:
delta 1e-10, our sealed-vs-streaming bridge 8.5e-6, your SE 3.19e-3. We're
adopting masking anyway — costs nothing, one less difference.

That ran out of tree at first, results only, which by your standard isn't good
enough. It's committed now — `bin/padded_column_study.py` + receipts. The
load-bearing half needs only your window, no lm_head, 7 s.

## Message 3 — your contamination finding, reproduced, and what it does to our numbers

We fetched the 665 published token arrays and re-ran your 13-gram scan with our
own implementation. All 25 windows match your published counts and fractions
exactly, 0 mismatches.

Your finding holds and it's the most important thing in the standard:
`document_id_in_calibration` is false for all 25 windows, document separation is
clean, and six of them still share 37-39% of their 13-grams with calibration
windows. Document-hash dedup does not catch it.

One thing worth flagging for anyone else reproducing this: the denominator is the
deduplicated gram set. An axis4 window has only ~710 distinct 13-grams out of
2036 slices because that corpus repeats itself. Using 2036 gives 13% instead of
38%.

We then recomputed our own published means on your 17-window clean scope, from
our own per-window arrays. No GPU, no re-measurement.

    K6 sealed   0.013723 -> 0.011677  (-14.9%)
    K6 stream   0.013715 -> 0.011676  (-14.9%)
    K8 stream   0.012384 -> 0.010829  (-12.6%)
    FP8 x-stack 0.020615 -> 0.018665  (-9.5%)
    BF16 floor (cross-stack) 0.012712 -> 0.010648  (-16.2%)
    your 4bpw   0.024555 -> 0.024949  (+1.6%)

Four of ours fall, yours rises. We checked eight thresholds from 0.02 to 0.20:
yours rises at all eight, and K6, K8 and the BF16 floor fall at all eight, so the
contrast isn't an artifact of 0.05. One exception, in our own numbers: the
cross-stack FP8 row falls at the five tightest thresholds and rises +2.6% at 0.075
and looser, because the two windows that separate the 17-window scope from the
19-window one (final-0021, final-0022) score more than twice FP8's clean-scope
mean. Either way neither scope stands in for the other and both have to be
published.

Two of our rows can't be recomputed at all: the BF16 streaming floor and the
Dione Q4 have scalar-only receipts with no per-window array. Our fault, noted.

## Message 4 — did the conclusions survive, and one thing about your threshold

Paired per-window, BCa on the differences, both scopes:

K8 better than K6 survives but weakens a lot. Panel scope the interval is
[+0.000695, +0.002330] with sign test p=0.004. Clean scope it's
[+0.000153, +0.001573], p=0.049. Still excludes zero, but it's sitting on the
line and the gap shrinks about a third. We won't restate that claim without the
scope attached.

K6 better than FP8 gets stronger: 17/17 windows on the clean scope, ratio
0.666 -> 0.626.

Also worth noting because it's your point about ranking: the marginal CIs for K6
and K8 overlap almost entirely on both scopes. Anyone eyeballing them calls it a
tie. The paired interval is 4.2x tighter on the clean scope (1.42e-3 wide against
5.93e-3 and 5.77e-3 for the two marginals) and 3.4x on the panel scope, and it
doesn't call it a tie. Our data makes your argument better than our old paired
t-interval did.

Your per-domain non-uniformity reproduces on our artifacts too. Clean scope,
FP8-over-K6 ratio: general 1.67x, legal 1.81x, code-agentic 1.30x. A 1.39x spread,
and legal is the worst domain for us as it is for you.

Now the one thing we'd push back on. Your 0.05 threshold is a bare literal in
cli.py with no sensitivity analysis published, and it matters:

    threshold  0.075  ->  19 windows, K6 moves -5.0%
    threshold  0.06   ->  18 windows, K6 moves -12.2%
    threshold  0.05   ->  17 windows, K6 moves -14.9%
    threshold  0.04   ->  16 windows, K6 moves -10.9%

Window count plateaus at 19 for anything in [0.075, 0.20]. The numbers don't
plateau anywhere. Moving from 0.075 to 0.05 triples the size of the correction.
Highest overlap among your retained windows is 4.75%, so 0.05 does separate — but
only just, and nothing about 0.05 is derived. Worth a joint decision.

## Message 5 — one real mutual confirmation, and a claim we cut

We had three of these written. On checking, only one is actually our data
confirming yours, so here's the honest version.

**The real one: determinism is a kernel-path property.** You: 25/25 bitwise
across three cold boots with shuffled window order, sigma_run exactly 0.0. Us on
a completely different lane: 5 cold runs, one distinct tokenwise KLD hash,
sigma_run exactly 0.0. Two independent stacks, same conclusion. And your NVFP4
counter-example (0/25 bitwise, 93.65% of tokens changed, 2652 top-1 flips, one
token swinging 4.69 nats, mean barely moving) is the best demonstration of why
sigma_run belongs next to the mean that anyone's published.

**The one we cut.** We were going to say your 0.030480-vs-0.024555 gap confirms
our registry's lane rule. It doesn't, twice over. Both numbers are yours — we
measured neither, we just ingested one. And `stack_relation` in our schema means
"did reference and candidate logits come from the same runtime *within one
measurement*", which both your readings satisfy; we actually tag the 0.024555 row
`same_stack`. What separates your two numbers is the pipeline, and we keep
`pipeline_ref` out of the comparability key — six rows from five pipelines share
one key in our data right now.

So your gap is a finding neither the standard nor our registry encodes yet, and
that's the gap we'd most like to close jointly: the key needs an engine-path
component, or every number needs an engine identity beside it. (And see Message 9
before either of us builds a headline on 0.024555 anyway.)

**Smaller, but true:** our panel record for your panel has carried a
`weak_contamination_guard` disclosure since 28 Aug, a day before your scan
published, saying role separation alone is materially weaker than an n-gram scan.
Flagging a missing check isn't predicting its result — the finding is yours. It
does suggest the disclosure field was pointed at the right thing.

## Message 6 — what we can put on the table

- Measured BF16 floors and attributable error. You're on record against
  subtraction in section 5.3, and this is our one real disagreement, so here's a
  fact rather than an opinion: across the panel->clean scope change, the
  cross-stack FP8 attributable error moves +1.44% while its two inputs move -9.5%
  and -16.2%. The subtraction is the stable half. We'd still never publish it
  without the raw row and the floor beside it, and our tooling refuses a floor
  from a different lane or a different scope.
- A schema-enforced registry with mechanical refusals rather than conventions.
  It caught a real mistake in this very work: our first clean-scope rows sat
  under the parent panel's comparability key and the validator refused them,
  because a different window set is a different panel. They now have their own
  panel record. 90 invariants, runs on a stock interpreter, no pip install.
- Multi-format decode surfaces. `k6/tools/stream_score.py --source` takes
  checkpoint, payload-store, dione, native, exl3hf, mlx, gguf, nvfp4 — so one
  KLD yardstick spans EXL3, Dione, MLX, GGUF and NVFP4 against the same teacher.
  Working today; the NVFP4 surface is the one most likely to be useful to you.
- A protocol-hash fix, below.
- An R0-b implementation that runs against the real teacher in-session. Yours has
  the shift check as a synthetic unit test on V=512 random logits; the session
  gate does the exactly-0.0 half. Ours does both, and the failure it catches is a
  teacher whose rows barely differ — a left-on prefix cache — which passes the 0.0
  check perfectly and is still broken. ~200 lines, stdlib + numpy, no dependency
  on the rest of our stack. Yours if you want it.

All of it is MIT as of today (we had no LICENSE file at all until this reply
forced the question — you clearly thought harder about your licence than we did
about ours). Our THIRD_PARTY_NOTICES states plainly that we vendor no one's
code, yours included.

## Message 7 — small corrections both ways

Ours first:

- Every row we published on your panels — 16 of them, 8 on final25 and 8 on the
  final-0000 sub-panel — carried no interval at all.
  Anyone who inferred one from std/sqrt(N) would have been off by 4.6x to 10x;
  the window design effect on this panel runs 21-29 for our rows and 74-105 for
  your 4bpw. (We never published a naive SE ourselves. We published nothing,
  which on your panel is worse.)
- Our 2.52x attributable-error ratio is a panel-scope number. The clean-scope
  version needs one re-measured streaming BF16 floor with per-window output.

Yours, same spirit:

- Scope labels. Clean 17-window means are 0.029258 and 0.049640; full-25 are
  0.030480 and 0.049218. The 51,175-positions figure belongs to the panel scope.
  A table with 0.0305 in a row labelled "clean" is mixing the two.
- kld_card.md says "25 windows / 34,799 scored positions". 34,799 is 17 windows.
- The headline table pairs the clean-scope NVFP4 mean with a panel-scope
  sigma_run, and the quadrature uses the panel SE.
- sigma_run 3.33e-4 is n=2, so it's |delta|/sqrt(2) with one degree of freedom.
  You did run three NVFP4 cold runs but only on 3 windows, where sigma is 1.40e-3.
- The KV/backend paired control ran on final-0000..0011, which includes three
  contaminated windows.
- `_percentile_ok` uses `n * (1.0 - q) >= 100`, and 1-0.9 is 0.09999999999999998,
  so n=1000 q=0.90 suppresses a quantile with exactly 100 exceedances. Doesn't
  change any decision in your campaign, just an edge.
- r0_student_canary.json carries no protocol_sha256 at all — looks like it's
  written outside _write_json().

## Message 8 — the protocol hash, and a suggested fix

The "one frozen protocol file" rule is right and it didn't survive your own
campaign. Three different protocol_sha256 values appear across your published
receipts:

    53e165dd  teacher_manifest, window_selection, run-run1/2/3/3b, run-r0a/b/c
    8e80e8e1  analysis-run1-*, kld_card.*, r5_sweep_cold3, paired_kvfp8
    4d1d91ad  all the NVFP4 receipts — and the file currently on the Hub

You disclose two of the three transitions yourself. The cause is visible in
make_protocol.py: the generator writes governing_document as a plain string
reading "NOT FOUND", while the published file has it as a dict with the report's
sha256 and also carries a student_nvfp4 block the generator never writes. So it
was hand-edited twice after generation, both times to add identity metadata.
Neither edit changed a scoring rule — and yet your EXL3 headline and the NVFP4
headline it's compared against carry different protocol hashes, and no receipt
except the NVFP4 ones carries the hash of the file that's published now.

That's not carelessness, it's what happens when the hash covers bytes that
include identity metadata. What we've done on our side, and would suggest:
publish two hashes.

    protocol_file_sha256     sha256 of the raw bytes. Your rule, unchanged.
    protocol_scoring_sha256  sha256 of a canonical JSON serialisation of only
                             the scoring-relevant blocks.

Two receipts are comparable when the scoring hashes match. A file hash that moved
while the scoring hash held is a provenance note, not an incomparability. Our
selftest checks it both ways: an identity-only edit moves the file hash and holds
the scoring hash; a scoring edit moves both.

## Message 9 — two asks

1. **The 0.0246 anchor needs a label.** It's described three ways: your
RUN_SUMMARY says the community pipeline (transformers eager, EP4, B200) reported
it for your checkpoint; your kld_card says it's the official FP8 on this stack;
our registry has it as your own k4-tp2 packed scorer, sourced from your
five-cold-run-kld.json, which names no engine. We have never measured your 4bpw
ourselves — there's no 4bpw report in our published tree, and both 4bpw rows in
our registry sit on your pipelines. There's also a coincidence that may explain
the third version: the official FP8 on the single-window sub-panel reads 0.024629,
which also rounds to 0.0246. The arithmetic you built on it is fine
(0.030480 - 0.024555 = 0.005926, and your BCa lower bound excludes it by 0.00041),
it's the label that's wrong in at least two places. Worth fixing before either of
us publishes a lane-gap claim.

2. **Per-window sufficient statistics.** Your per-token Parquet is gitignored, so
nobody outside can recompute your percentiles, top-1 rates, per-domain CIs,
rms_delta_p, ln_ppl_ratio or any McNemar cell — only the means and mean-CIs, via
the per-window arrays in your run-*.json. Publishing n, sum d, sum d^2, top-1
count and a few quantiles per window would close that for almost no bytes. Same
applies to us, and we'll do it too. It's the single highest-value change either
of us could make for third-party reproducibility.

## Message 10 (optional, or DM) — licence

One housekeeping question. Your repo carries the ShapleyMCG 1.0 licence with the
named exclusion. We're not the excluded party — your THIRD_PARTY_NOTICES credits
us — but we do publish a fidelity number for an 0xSero artifact, and section 1.2's
Derivative definition covers re-implementations made with reference to the Work.

So we've been careful: we clean-room implemented the statistics (BCa,
cluster-robust SE, McNemar, Wilson, n-gram shingling, all textbook and all spelled
out in Appendix A of your report), adopted the protocol design with citation, and
vendored none of your source. The one thing we reproduce is a fixture of your 25
per-window means and your published CI endpoints, as the known-answer test our
implementation has to pass — results, cited, not code.

Two questions, and either answer is fine:

1. Can we depend on `kld_eval` as an installed library, or would you rather we
   didn't?
2. What attribution do you want, and where?
