#!/bin/bash
# verify -> run1 -> run2 -> run3, chained so no GPU-hour is spent waiting for an
# operator to notice a stage finished.  Each stage is individually guarded by
# the published stage script, so this is resumable after a spot preemption.
#
# Timing is written to /home/hr/receipts/stage-timing.txt after every stage, so
# the run-1 duration can be used to project cost before runs 2 and 3 commit.
set -uo pipefail
S=/home/suite/engines/tools/hidden_replay_stage.sh
T=/home/hr/receipts/stage-timing.txt
NTFY="${NTFY_URL:-${QP_NTFY_URL:-}}"

note() { [ -n "$NTFY" ] || return 0; curl -s -m 10 -H "Title: $2" -H "Tags: $3" -d "$1" "$NTFY" >/dev/null 2>&1 || true; }

for stage in verify run1 run2 run3; do
  t0=$(date +%s)
  echo "=== $stage starting $(date -u +%H:%M:%S) ==="
  bash "$S" "$stage"
  rc=$?
  t1=$(date +%s)
  echo "$stage rc=$rc elapsed_s=$((t1-t0)) at=$(date -u +%FT%TZ)" | tee -a "$T"
  if [ $rc -ne 0 ]; then
    note "stage $stage FAILED rc=$rc after $((t1-t0))s" "GLM53 hidden-replay FAILED" "rotating_light" urgent
    exit $rc
  fi
  note "$stage done in $(( (t1-t0)/60 )) min" "GLM53 hidden-replay $stage" "white_check_mark"
done
echo "=== all captures complete ==="
cat "$T"
