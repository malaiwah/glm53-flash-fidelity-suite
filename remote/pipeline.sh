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
# Notification endpoint: opt-in via NTFY_URL (or QP_NTFY_URL). Unset = no
# notifications. A public repo must not pin its operator's channel.
NTFY_URL="${NTFY_URL:-${QP_NTFY_URL:-}}"
RATE=31.92
BUDGET_USD="${BUDGET_USD:-200}"
# SH-21. BUDGET_USD was interpolated straight into `python3 -c`, so a malformed value
# (`$200`, `200 USD`, `200,00`, a zero-padded `0450`) made remaining() print NOTHING, the
# budget test at G5 evaluated a SyntaxError to empty, and the FP8 leg was SILENTLY SKIPPED
# and announced as a deliberate budget decision. Half the measurement dropped by a typo,
# reported as a choice. Validate on entry, and pass values by argv below rather than by
# string interpolation.
case "$BUDGET_USD" in ''|*[!0-9.]*)
  echo "BUDGET_USD must be numeric (got: $BUDGET_USD)" >&2; exit 2;; esac
python3 -c 'import sys; float(sys.argv[1])' "$BUDGET_USD" 2>/dev/null || {
  echo "BUDGET_USD must be a number (got: $BUDGET_USD)" >&2; exit 2; }
START_TS=$(date +%s)
ntfy() { [ -n "$NTFY_URL" ] || return 0; curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true; }
spent() { python3 -c "import time,sys; print(round((time.time()-float(sys.argv[1]))/3600*float(sys.argv[2]), 2))" "$START_TS" "$RATE"; }
remaining() { python3 -c "import sys; print(round(float(sys.argv[1])-float(sys.argv[2]), 2))" "$BUDGET_USD" "$(spent)"; }

hold() {  # hold <reason> <message>
  # SH-14. This blocks forever with no deadline on an instance that keeps billing at
  # $RATE/h. The box cannot stop its own bill -- by design, an on-instance script never
  # carries a JarvisLabs credential -- so the only thing that shortens an unattended hold
  # is escalating re-notification. Silence for eight hours is what it used to do.
  touch "$ROOT/HOLD.$1"
  ntfy "$2 — pipeline holding. Resume: remove HOLD file + touch DECISION.<go|trim|skip_fp8>" "GLM53 HOLD: $1" "vertical_traffic_light" "high"
  _held=0
  while [ -e "$ROOT/HOLD.$1" ]; do
    sleep 60
    _held=$(( _held + 1 ))
    if [ $(( _held % 30 )) -eq 0 ]; then
      ntfy "STILL HOLDING on '$1' after $(( _held / 60 ))h$(( _held % 60 ))m — spent \$$(spent), remaining \$$(remaining). The instance is billing at \$$RATE/h and cannot stop itself." \
           "GLM53 HOLD $1 still open" "rotating_light" "urgent"
    fi
  done
  ntfy "hold $1 released after $(( _held / 60 ))h$(( _held % 60 ))m, continuing" "GLM53 resume" "arrow_forward"
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
# SH-02. The gate only evaluated once the capture reached 64 contexts INSIDE the 60-minute
# probe window, so PACE_OK stayed empty for a catastrophically slow capture -- the exact
# runaway the gate exists to stop -- and `[ "$PACE_OK" = "0" ]` was then false. The window
# is 120 x 30s = ~3606s, so the gate could only ever fire at <= ~56 s/ctx even with zero
# engine load time, and VLLM_ENGINE_READY_TIMEOUT_S is 3600, so a slow load disarmed it
# entirely at ANY pace. The one budget circuit-breaker on a $31.92/h 8xH200 was inverted:
# it caught slow-but-survivable and missed the runaway.
#
# Three outcomes now, not two, and an unparseable or absent pace is a HOLD, never a pass.
PACE_OK=""
PROBE_EXIT="window_expired"
PACE_N0_TS=""
for i in $(seq 1 120); do
  sleep 30
  N=$(python3 -c "import json;print(json.load(open('$ROOT/captures/bf16/capture-manifest.json')).get('contexts',0))" 2>/dev/null || echo 0)
  # Start the clock at the first tick that has produced a context, so the vLLM engine
  # load (up to VLLM_ENGINE_READY_TIMEOUT_S) does not consume the measurement window.
  if [ "${N:-0}" -gt 0 ] && [ -z "$PACE_N0_TS" ]; then PACE_N0_TS=$(date +%s); fi
  if [ "${N:-0}" -ge 64 ]; then
    PROBE_EXIT="measured"
    PACE=$(python3 -c "import json;m=json.load(open('$ROOT/captures/bf16/capture-manifest.json'));print(round(m['elapsed_sec']/max(m['contexts'],1),2))")
    PROJ=$(python3 -c "print(round(5120*$PACE/3600*$RATE*2,0))")  # both legs, rough
    ntfy "pace probe: ${PACE}s/ctx at $N contexts; projected both-legs capture cost ~\$$PROJ; spent \$$(spent)" "GLM53 G2 pace probe" "stopwatch"
    PACE_OK=$(python3 -c "print(1 if $PACE <= 2.5 else 0)")
    break
  fi
  # The capture died: that is NOT a pace verdict. Leave it to the `wait` below, which
  # diagnoses it and halts loudly -- turning this into a 60s-poll hold would be an
  # observability regression on the most common real failure.
  kill -0 $CAP_PID 2>/dev/null || { PROBE_EXIT="proc_gone"; break; }
done

# SH-02: the window expired without ever reaching 64 contexts. Project from whatever was
# reached rather than treating silence as a pass.
if [ "$PROBE_EXIT" = "window_expired" ]; then
  if [ "${N:-0}" -le 0 ]; then
    ntfy "pace probe saw ZERO contexts in 60 min (engine wedged, or still loading); spent \$$(spent)" \
         "GLM53 G2 pace probe: no progress" "rotating_light" "urgent"
    hold "pace-no-progress" "G2: no context produced within the probe window"
  else
    _EL=$(( $(date +%s) - ${PACE_N0_TS:-$(date +%s)} ))
    PACE=$(python3 -c "print(round($_EL/max($N,1),2))" 2>/dev/null || echo "")
    ntfy "pace probe: only $N contexts in 60 min (~${PACE}s/ctx); spent \$$(spent)" \
         "GLM53 G2 pace probe: window expired" "rotating_light" "urgent"
    hold "pace" "G2: fewer than 64 contexts in the probe window (~${PACE}s/ctx)"
  fi
fi
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
