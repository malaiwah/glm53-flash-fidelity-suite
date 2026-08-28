#!/bin/bash
# Q4 (0xSero) measurement: probe -> 2 cold runs -> escalate to 5 if hashes differ -> report.
set -u
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
  [ -f "$RCPT/dione-q4-student-run$n/capture-receipt.json" ] && { echo "run$n exists"; return 0; }
  QP_GLM53_EP_SIZE=8 QP_PIPELINE_ROOT=$PIPE PYTHONPATH=$PIPE/src:$R/shapleymcg:$R/sqg-mcg NVIDIA_TF32_OVERRIDE=0 \
  $V/torchrun --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 \
    $R/tools/k6_student_capture.py --surface dione --profile dione \
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
h = hashlib.sha256()
for p in sorted(glob.glob(sys.argv[1] + "/logits/*.safetensors")):
    with safe_open(p, framework="np") as f:
        for k in sorted(f.keys()):
            h.update(f.get_tensor(k).tobytes())
print(h.hexdigest())
PY
}
H1=$(tensor_digest "$RCPT/dione-q4-student-run1")
H2=$(tensor_digest "$RCPT/dione-q4-student-run2")
RUNS="$RCPT/dione-q4-student-run1 $RCPT/dione-q4-student-run2"
echo "run hashes: $H1 $H2"
if [ "$H1" != "$H2" ]; then
  echo "ESCALATING to 5 runs (nondeterminism detected)"
  curl -s -d "Q4: runs differ -> escalating to 5 cold runs" "$NTFY" -H "Title: q4 escalation" >/dev/null
  for n in 3 4 5; do run_one $n || exit 1; RUNS="$RUNS $RCPT/dione-q4-student-run$n"; done
fi

QP_PIPELINE_ROOT=$PIPE PYTHONPATH=$PIPE/src:$R/shapleymcg:$R/sqg-mcg \
  $V/python $R/tools/k6_kld_report.py --profile dione --teacher $TEACH --runs $RUNS --out $RCPT/dione-q4-kld || { echo REPORT_FAILED; exit 1; }
echo Q4_REPORT_DONE
find $RCPT -maxdepth 2 -name "*dione-q4*kld*.json" | head -3
curl -s -d "Q4 MEASUREMENT COMPLETE — report sealed" "$NTFY" -H "Title: Q4 number in" -H "Priority: high" >/dev/null
