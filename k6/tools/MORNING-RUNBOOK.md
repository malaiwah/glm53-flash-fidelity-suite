# MORNING RUNBOOK - qualify_k8, then Dione Q4 (and optionally 3.0bpw) scoring

One session, 8x H200 box (K6-qualify VM 485017; fleet 484853 stays on K8 work).
No HF tokens needed anywhere below - both 0xSero repos are public and all
fetches are anonymous read-only.

Everything below was validated offline overnight (2026-08-28, this Mac):
`tools/selftest_dione_offline.py` is 5/5 green - pack-layout equivalence vs
exllamav3 pack.cu (K3/K4/K6), bitwise decode identity vs the campaign reader
(K4/K6), safetensors end-to-end assembly, REAL Q4 index census
(580,608 packed / 2,482 retained, bijects the real BF16 index), and both
dry-runs against the real config/index.  A REAL-payload placement audit
(L3 E0 gate+down, ranged-fetched from the live repo) proved the TP4 slice
layout: gate/up = out-dim rank-ordered blocks (concat dim 0), down = in-dim
rank-ordered blocks (concat dim 1), identity-placement cosine 0.996, assembled
rel-L2 vs official BF16 0.089/0.083 (= expected 4bpw reconstruction error).
Evidence: `tools/dione-evidence/real-payload-placement-audit.json`.

## 0. Preflight (no GPU, ~5 min)

```bash
# sync tools atomically (upload as .new, then mv - same convention as stage_k6.sh)
for f in k6_student_capture.py k6_kld_report.py dione_surface.py selftest_dione_offline.py; do
  scp tools/$f box:$ROOT/tools/$f.new && ssh box "mv $ROOT/tools/$f.new $ROOT/tools/$f"
done
# also sync tools/dione-evidence/{index-q4,bf16-index,config-q4,exl3-manifest}.json
# (needed for the box-side selftest check 4/5; skip = those checks self-skip)
ssh box "cd $ROOT/tools && $VENV/bin/python selftest_dione_offline.py --pipeline-root $PIPE"
```

$ROOT=/home/jl_fs/glm53-k6, $PIPE=$ROOT/pipeline (patched, patches-v2),
$BF16=/home/jl_fs/models/bf16 (pinned a6c167b6 == the exact source revision the
Dione conversion quantized from - the retained-tensor byte verification below
should therefore pass identically), $TEACH=$ROOT/teacher-final,
$RCPT=$ROOT/receipts, $VENV=$ROOT/venv.

## 1. qualify_k8 (3 cold runs, existing stage, unchanged)

```bash
./stage_k6.sh qualify_k8
```

Three EP8 cold captures + fp64 KLD + K8 TP4 runtime receipt.  Uses
pipeline-k8 tree of record (the stage script handles that).  Gate: mean < 0.06.

## 2. Dione Q4 scoring

Disk first: Q4 snapshot = **187.4 GB** (174.5 GiB; 168 layer shards + 49
retained shards + metadata).  Each capture run stores 25 x 2047 x 154880 fp32
logits = **31.7 GB**.  Check `df -h /home/jl_fs` for >= ~230 GB free
(+150 GB more if 3.0bpw happens in the same session).  Decoded-expert
RAM/VRAM is IDENTICAL to our EP8 K6/K8 runs (same BF16 install targets,
[36,4096,4096]+[36,4096,2048] per layer per rank) - no new memory math.

```bash
# a) download (public, anonymous, pinned to the immutable commit)
REV_Q4=99cccdf0e8741715662c383828a9ea601990c125
hf download 0xSero/GLM-5.3-Flash-EXL3-Q4 --revision $REV_Q4 --local-dir $ROOT/dione-q4

# b) whole-shard sha256 vs the release's own exl3-manifest.json (~15-25 min NVMe)
$VENV/bin/python $ROOT/tools/dione_surface.py verify-shards --root $ROOT/dione-q4

# c) CPU probe: layout census + slice-placement audit vs $BF16 (~2 min)
#    expect: identity cosine ~0.996 per projection, passed:true
$VENV/bin/python $ROOT/tools/dione_surface.py probe \
  --root $ROOT/dione-q4 --bf16 $BF16 --pipeline-root $PIPE

# d) dry-run, then the capture (1 cold run; add runs 2/3 only if budget allows -
#    the stack already proved bitwise determinism on K4/K6)
$VENV/bin/python $ROOT/tools/k6_student_capture.py \
  --surface dione --profile dione \
  --dione-root $ROOT/dione-q4 --dione-repo 0xSero/GLM-5.3-Flash-EXL3-Q4 \
  --dione-revision $REV_Q4 \
  --bf16 $BF16 --teacher $TEACH --cold-run 1 \
  --out $RCPT/dione-q4-student-run1 --pipeline-root $PIPE --dry-run

QP_GLM53_EP_SIZE=8 $VENV/bin/torchrun --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 \
  $ROOT/tools/k6_student_capture.py \
  --surface dione --profile dione \
  --dione-root $ROOT/dione-q4 --dione-repo 0xSero/GLM-5.3-Flash-EXL3-Q4 \
  --dione-revision $REV_Q4 \
  --bf16 $BF16 --teacher $TEACH --cold-run 1 \
  --out $RCPT/dione-q4-student-run1 --pipeline-root $PIPE

# e) fp64 KLD (teacher_to_student, sealed 25-window final panel, 51,175 pos)
$VENV/bin/python $ROOT/tools/k6_kld_report.py --profile dione-q4 \
  --teacher $TEACH --runs $RCPT/dione-q4-student-run1 \
  --out $RCPT/dione-q4-packed-kld.json \
  --comparison-out $RCPT/comparison-table.md
```

Wall time per capture run = same as a K6 EP8 run (model load + install +
25 windows); install decodes 4 TP slices per matrix instead of 1 payload but
skips the per-choice hash-verify walk, so expect comparable or slightly less.

What the capture does differently in `--surface dione` (all DISCLOSED in the
receipts): no materialization/contract/inventory/MTP receipts exist, so the
surface is decoded WITHOUT seal verification (`seal_disclosure` field);
whole-shard hashes vs exl3-manifest.json stand in (step b writes the marker
the capture requires); rank 0 verifies the retained non-routed tensors are
byte-identical to $BF16 (`--verify-nonrouted sample` default; use `full` if
paranoid, ~68 GB of reads) and re-runs the slice-placement audit before any
install; MTP layer 45 ships natively in their snapshot and is not executed
(same as our captures).  Decode math is the campaign reader's own
`decode_choice_hf` - bitwise the same function our sealed K4/K6/K8 numbers
used.

## 3. Optionally: 3.0bpw

As of 2026-08-28 ~03:00 EDT the repo `0xSero/GLM-5.3-Flash-EXL3-3.0bpw`
(commit 9909e1f1) contains ONLY a "Campaign status: pending" README - no
weights.  Check before planning:

```bash
curl -s https://huggingface.co/api/models/0xSero/GLM-5.3-Flash-EXL3-3.0bpw | jq '.siblings | length'
```

If the weights are up: snapshot ~**149 GB** (~139 GiB: trellis scales 3/4 vs
Q4's 152.2 GB, vectors ~1.3 GB constant, retained 33.8 GB), planned as EXL3
**K3** in the same TP4 layout (their README).  The adapter is
bits-parameterized: trellis [.,.,48] geometry, decode via the anybits path
(same math, proven bitwise-equal to the reader at K4/K6 and roundtrip-proven
at K3 in the selftest).  Same commands with the new revision pin,
`--dione-root $ROOT/dione-3bpw`, out dirs `dione-3.0bpw-student-run1`, report
`--profile dione-3.0bpw`.  The probe (step c) gates the layout before any GPU
time; K3 identity cosine will be lower (~0.98) but must still dominate off-
diagonals by the audit's margin.

Cleanup when receipts are sealed: the snapshots ($ROOT/dione-q4,
$ROOT/dione-3bpw) can be deleted; the receipts pin repo+revision+shard hashes.

## 4. Discussion post for 0xSero's page (open a Community discussion on the Q4 repo)

Fill the numbers from `$RCPT/comparison-table.md` /
`$RCPT/dione-q4-packed-kld.json` (and K6/K8 receipts).  Template:

---

**Title: Scored GLM-5.3-Flash-EXL3-Q4 on our sealed 25-window KLD panel (same
harness as brandonmusic's 4bpw and our K6/K8) - results + receipts**

Hi - nice release, and thanks for shipping the exl3-manifest + validation
reports with it. We maintain a teacher-forced fidelity harness for
GLM-5.3-Flash quants (the one brandonmusic's EXL3/TR3-MCG 4bpw was scored
with): 25 sealed 2048-token final windows (51,175 prediction positions),
fp32 BF16-teacher logits, fp64 tokenwise KL(teacher||student), whole model
executed with the routed experts reconstructed offline to BF16 - so every
model in the table is measured by the exact same math on the exact same
tokens.

We wrote a small adapter for your TP4-sliced selective-EXL3 layout (each
routed projection = 4 independent EXL3-K4/MCG quantizations, gate/up sliced on
out-features, down on in-features; rank-ordered concat reassembles the HF
tensor - we verified the placement against the BF16 originals before scoring,
identity-block cosine 0.996). Decode is the same independently-implemented
EXL3 trellis/MCG decoder our other numbers used, applied per slice.

| model | routed bpw | mean tokenwise KLD vs BF16 teacher (25 sealed windows, 51,175 pos, fp64) |
|---|---|---|
| zai-org FP8 (as served) | 8 | 0.020615 |
| brandonmusic K4 (EXL3/TR3-MCG) | 4.01 | 0.024555 (five-run mean, stddev 0) |
| malaiwah K6 (EXL3/TR3-MCG) | 6.01 | <from $RCPT/k6-packed-kld.json> |
| malaiwah K8 (EXL3/TR3-MCG) | 8.01 | <from $RCPT/k8-packed-kld.json> |
| **0xSero Dione Q4 (EXL3 K4, TP4-sliced)** | 4.0 | **<from dione-q4-packed-kld.json>** |

(If 3.0bpw was scored, add its row.)

Methodology notes, for apples-to-apples honesty: (1) your wikitext
KL(bf16->q4) = 0.0658 is a different panel/metric, so it is not comparable to
this column - that is exactly why we re-scored on ours; (2) your checkpoint
ships no per-tensor receipts, so unlike the other rows this one is
"unsealed-source": we verified every shard against your exl3-manifest.json
sha256s, byte-verified the retained BF16 tensors against zai-org @ a6c167b6,
and sealed a canonical digest of the per-slice payload-hash census into
each rank's install record, but there is no encoder-side
closure to close against - the capture receipt discloses this; (3) the
TP4-sliced quantization is a slightly different problem than whole-tensor
(four independent trellis fits + Hadamard rotations per matrix), which is an
interesting design point in its own right - the number above measures what
you actually shipped.

Full receipts (capture plan, shard-hash marker, placement audit, non-routed
byte verification, per-window KLDs): <link to published receipts tree>.
Happy to re-run if you cut a new revision, and the adapter is available if
you want it - credit to the Dione workflow + ExLlamaV3/turboderp for the
format, zai for the base. Nice work fitting it in 174.5 GiB.

---

Tone check before posting: we are guests on their release page - lead with
the result, disclose our deviations as OURS (unsealed-source scoring), no
gatekeeping about their missing receipts, offer the tooling.
