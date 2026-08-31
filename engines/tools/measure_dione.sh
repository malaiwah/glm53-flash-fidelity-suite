#!/bin/bash
# Q4 (0xSero) measurement: probe -> 2 cold runs -> escalate to 5 if hashes differ -> report.
#
# SH-06 / CC-09. This ran under `set -u` alone and its escalation trigger could not
# fail: `tensor_digest` over a logits/ directory with no .safetensors returns the
# sha256 of nothing (e3b0c442...) for BOTH runs, and any failure of the embedded
# python left BOTH variables empty -- either way `[ "$H1" != "$H2" ]` is false and
# the script reports the two runs as identical without ever escalating to five.
#
# What that did NOT do is weaken the published Q4 number. The registry row
# measurement--glm53.dione-q4.brandonmusic-final25 records run_count 5 with
# evidence_kind tokenwise_kld_sha256, so on that run the gate DID fire and the
# published determinism claim rests on the fp64 report's tokenwise digest rather
# than on this comparison. The defect is prospective: the next artifact measured
# through this script could be published as deterministic on two runs whose
# contents were never actually compared.
set -euo pipefail
R=/home/jl_fs/glm53-k6
V=$R/venv/bin
PIPE=$R/pipeline
BF16=/home/jl_fs/models/bf16
TEACH=$R/teacher-final
RCPT=$R/receipts
Q4=/home/ubuntu/q4
REV=99cccdf0e8741715662c383828a9ea601990c125
NTFY="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"

$V/python $R/tools/dione_surface.py probe --root $Q4 --bf16 $BF16 --pipeline-root $PIPE || { echo PROBE_FAILED; exit 1; }
echo PROBE_OK

run_one() {
  local n=$1
  # An `A && { ...; }` guard is the wrong shape under `set -e`: when A is false the
  # whole list is the failing command and the shell exits. Say it with `if`.
  if [ -f "$RCPT/dione-q4-student-run$n/capture-receipt.json" ]; then
    echo "run$n exists"
    return 0
  fi
  QP_GLM53_EP_SIZE=8 QP_PIPELINE_ROOT=$PIPE PYTHONPATH=$PIPE/src:$R/shapleymcg:$R/sqg-mcg NVIDIA_TF32_OVERRIDE=0 \
  $V/torchrun --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 \
    $R/tools/student_capture.py --surface dione --profile dione \
    --dione-root $Q4 --dione-repo 0xSero/GLM-5.3-Flash-EXL3-Q4 --dione-revision $REV \
    --bf16 $BF16 --teacher $TEACH --cold-run $n \
    --out $RCPT/dione-q4-student-run$n --pipeline-root $PIPE
}

for n in 1 2; do run_one $n || { echo "RUN_FAILED_$n"; exit 1; }; done
# Determinism check compares TENSOR CONTENT, never file bytes: capture receipts
# embed elapsed_seconds and safetensors embed __metadata__ (cold_run, backend
# identity), so container hashes ALWAYS differ between runs.
tensor_digest() { $V/python - "$1" <<'PY'
import glob, hashlib, sys
from safetensors import safe_open
files = sorted(glob.glob(sys.argv[1] + "/logits/*.safetensors"))
if not files:
    # The sha256 of nothing is a CONSTANT, and it is equal to itself. A digest over
    # zero windows must never be compared as evidence of determinism.
    sys.stderr.write("tensor_digest: no logits/*.safetensors under %s\n" % sys.argv[1])
    raise SystemExit(3)
h = hashlib.sha256()
n = 0
for p in files:
    with safe_open(p, framework="np") as f:
        for k in sorted(f.keys()):
            h.update(f.get_tensor(k).tobytes())
            n += 1
if n == 0:
    sys.stderr.write("tensor_digest: %d file(s) but no tensors\n" % len(files))
    raise SystemExit(3)
# The tensor COUNT rides along so the comparison below can refuse a digest over
# nothing even if this guard is ever loosened.
print("%s %d" % (h.hexdigest(), n))
PY
}
H1=$(tensor_digest "$RCPT/dione-q4-student-run1")
H2=$(tensor_digest "$RCPT/dione-q4-student-run2")
RUNS="$RCPT/dione-q4-student-run1 $RCPT/dione-q4-student-run2"
echo "run hashes: $H1 $H2"
# Two empty strings compare equal, and so do two digests over zero tensors. Refuse
# both rather than reading either as determinism. (`set -e` already stops a failing
# tensor_digest; this stays as the statement of the rule.)
if [ -z "$H1" ] || [ -z "$H2" ]; then
  echo "DETERMINISM_GATE_UNUSABLE: a run digest is empty" >&2
  exit 4
fi
if [ "${H1##* }" -eq 0 ] || [ "${H2##* }" -eq 0 ]; then
  echo "DETERMINISM_GATE_UNUSABLE: a run digest covered zero tensors" >&2
  exit 4
fi
if [ "$H1" != "$H2" ]; then
  echo "ESCALATING to 5 runs (nondeterminism detected)"
  curl -s -d "Q4: runs differ -> escalating to 5 cold runs" "$NTFY" -H "Title: q4 escalation" >/dev/null
  for n in 3 4 5; do run_one $n || exit 1; RUNS="$RUNS $RCPT/dione-q4-student-run$n"; done
fi

QP_PIPELINE_ROOT=$PIPE PYTHONPATH=$PIPE/src:$R/shapleymcg:$R/sqg-mcg \
  # `dione` is not a profile kld_report accepts -- its choices are dione-q4 /
  # dione-3.0bpw -- so this line exited 2 as committed, after both captures. And
  # the report was written without a .json suffix while the find below looks for
  # `*dione-q4*kld*.json`, so the search could never have matched it.
  $V/python $R/tools/kld_report.py --profile dione-q4 --teacher $TEACH \
    --runs $RUNS --out $RCPT/dione-q4-kld.json || { echo REPORT_FAILED; exit 1; }
echo Q4_REPORT_DONE
find $RCPT -maxdepth 2 -name "*dione-q4*kld*.json" | head -3
curl -s -d "Q4 MEASUREMENT COMPLETE — report sealed" "$NTFY" -H "Title: Q4 number in" -H "Priority: high" >/dev/null
