#!/bin/bash
# Self-driving 8x H200 session pipeline. Runs ON the rental; advances itself,
# enforces every gate, posts ntfy at each transition, and halts (never guesses)
# on red gates. The Mac/cloud supervisor is optional for progress, required only
# for exceptions, publishing, and pausing the instance at the end.
#
# Usage: BUDGET_USD=<balance at launch> bash pipeline.sh
# Halt/consult protocol: on a judgment gate the pipeline writes
#   /home/ubuntu/glm53/HOLD.<reason>, posts high-priority ntfy, and waits.
#   Supervisor resumes by:  rm HOLD.* && touch /home/ubuntu/glm53/DECISION.<go|trim|skip_fp8>
set -uo pipefail
ROOT=/home/ubuntu/glm53
FS=/home/jl_fs/glm53
S="$ROOT/bundle/remote/stage.sh"
DL="$ROOT/bundle/remote/download_model_vm.sh"
NTFY_URL="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"
RATE=31.92
BUDGET_USD="${BUDGET_USD:-200}"
START_TS=$(date +%s)
ntfy() { curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true; }
spent() { python3 -c "import time; print(round((time.time() - $START_TS) / 3600 * $RATE, 2))"; }
remaining() { python3 -c "print(round($BUDGET_USD - $(spent), 2))"; }

hold() {  # hold <reason> <message>
  touch "$ROOT/HOLD.$1"
  ntfy "$2 — pipeline holding. Resume: remove HOLD file + touch DECISION.<go|trim|skip_fp8>" "GLM53 HOLD: $1" "vertical_traffic_light" "high"
  while [ -e "$ROOT/HOLD.$1" ]; do sleep 60; done
  ntfy "hold $1 released, continuing" "GLM53 resume" "arrow_forward"
}

run_stage() {  # run_stage <name> — stage.sh already ntfys start/ok/fail
  if ! bash "$S" "$1"; then
    ntfy "stage $1 failed; spent ~\$$(spent). Pipeline HALTED for diagnosis (instance still billing)." "GLM53 PIPELINE HALTED at $1" "rotating_light" "urgent"
    exit 1
  fi
}

# ---- restore from filesystem (models were pre-downloaded by the prep phase)
mkdir -p "$ROOT"/{models,captures,out,logs}
ntfy "8x pipeline starting. Budget \$$BUDGET_USD, rate \$$RATE/h. Restoring from filesystem." "GLM53 8x session START" "rocket"
if [ ! -d "$ROOT/models/bf16" ]; then
  time cp -a "$FS/models/bf16" "$ROOT/models/bf16"
fi
cp -a "$FS/out/." "$ROOT/out/" 2>/dev/null || true   # heads + equality receipt from prep

[ -f "$ROOT/out/gen-check.json" ] || run_stage gen_check

# ---- BF16 leg with in-flight pace probe
bf16_done() { python3 -c "import json;m=json.load(open('$ROOT/captures/bf16/capture-manifest.json'));exit(0 if m['complete'] else 1)" 2>/dev/null; }
if bf16_done; then PACE_OK=1; CAP_PID=""; else
bash "$S" capture_bf16 &
CAP_PID=$!
PACE_OK=""
for i in $(seq 1 120); do
  sleep 30
  N=$(python3 -c "import json;print(json.load(open('$ROOT/captures/bf16/capture-manifest.json')).get('contexts',0))" 2>/dev/null || echo 0)
  if [ "${N:-0}" -ge 64 ]; then
    PACE=$(python3 -c "import json;m=json.load(open('$ROOT/captures/bf16/capture-manifest.json'));print(round(m['elapsed_sec']/max(m['contexts'],1),2))")
    PROJ=$(python3 -c "print(round(5120*$PACE/3600*$RATE*2,0))")  # both legs, rough
    ntfy "pace probe: ${PACE}s/ctx at $N contexts; projected both-legs capture cost ~\$$PROJ; spent \$$(spent)" "GLM53 G2 pace probe" "stopwatch"
    PACE_OK=$(python3 -c "print(1 if $PACE <= 2.5 else 0)")
    break
  fi
  kill -0 $CAP_PID 2>/dev/null || break
done
if [ "$PACE_OK" = "0" ]; then
  kill $CAP_PID 2>/dev/null; wait $CAP_PID 2>/dev/null
  hold "pace" "G2: pace exceeds 2.5s/ctx"
  if [ -e "$ROOT/DECISION.trim" ]; then
    ntfy "trim decision received but trimming requires a rebuilt suite from the supervisor — holding for new bundle" "GLM53 trim path" "scissors" "high"
    hold "trim-bundle" "waiting for trimmed suite upload"
  fi
  bash "$S" capture_bf16 &
  CAP_PID=$!
fi
wait $CAP_PID || { ntfy "capture_bf16 failed; spent \$$(spent)" "GLM53 PIPELINE HALTED at capture_bf16" "rotating_light" "urgent"; exit 1; }
fi

[ -f "$ROOT/out/determinism-bf16.json" ] || run_stage sentinel_bf16
[ -f "$ROOT/out/qualify-bf16.json" ] || run_stage qualify_bf16
acts_done() { python3 -c "import json;m=json.load(open('$ROOT/activations/bf16-cal/activation-manifest.json'));exit(0 if m['complete'] else 1)" 2>/dev/null; }
acts_done || run_stage activations
if [ -d "$FS/crosscheck/bm-teacher-logits" ] && [ ! -f "$ROOT/out/crosscheck-brandonmusic.json" ]; then
  [ -d "$ROOT/crosscheck" ] || cp -a "$FS/crosscheck" "$ROOT/crosscheck"
  run_stage cross_check
fi

# FP8 weights were pre-downloaded to the FS by prep; copy while gates verify
if [ ! -d "$ROOT/models/fp8" ]; then time cp -a "$FS/models/fp8" "$ROOT/models/fp8"; fi

if ! bash "$S" free_bf16; then
  hold "qualify-gate" "G3 numeric gates failed (qualify KLD / determinism) - supervisor review before BF16 delete"
  bash "$S" free_bf16 || { ntfy "free_bf16 failed again after hold release" "GLM53 PIPELINE HALTED" "rotating_light" "urgent"; exit 1; }
fi

# ---- FP8 leg (budget-gated)
if [ "$(python3 -c "print(1 if $(remaining) >= 70 else 0)")" = "1" ] && [ ! -e "$ROOT/DECISION.skip_fp8" ]; then
  run_stage head_check_fp8
  run_stage capture_fp8
  run_stage sentinel_fp8
  run_stage qualify_fp8
  run_stage replay
else
  ntfy "FP8 leg skipped (budget remaining \$$(remaining) or explicit skip). BF16-only dataset ships." "GLM53 G4: FP8 leg skipped" "fast_forward" "high"
fi

run_stage env_receipt
run_stage package
run_stage publish
cp -a "$ROOT/activations" "$FS/activations" 2>/dev/null || true
cp -a "$ROOT/deliverables" "$FS/deliverables"    # second copy on the filesystem
touch "$ROOT/PIPELINE_COMPLETE" "$FS/PIPELINE_COMPLETE"
ntfy "Pipeline COMPLETE. Spent ~\$$(spent). Deliverables on VM + filesystem. INSTANCE STILL BILLING at \$$RATE/h — supervisor must download + pause." "GLM53 SESSION COMPLETE" "tada" "urgent"
