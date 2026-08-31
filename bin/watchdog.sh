#!/usr/bin/env bash
# On-instance watchdog -- teardown layer L1.
#
#   watchdog.sh <deadline_epoch> <heartbeat_timeout_seconds> <fs_root>
#
# Two independent triggers, either of which stops the work:
#   * DEADLINE  -- the absolute --max-runtime the controller was started with.
#   * HEARTBEAT -- the controller touches $FS/heartbeat every 60s. If that file
#                  goes stale, the controller is presumed dead and this run is
#                  no longer being watched by anyone who can pay attention.
#
# WHAT IT DELIBERATELY DOES NOT DO: destroy the instance. Destroying requires a
# JarvisLabs account credential, and putting a full account key (create /
# destroy / billing) on rented third-party hardware for the length of a run is
# strictly worse than the leak it would prevent. So this stops the workload,
# seals whatever receipts exist, and writes ABANDONED.json; destruction is the
# job of the controller trap (L0), the laptop reaper (L2) or the name-deadline
# sweep (L3), all of which run where the credentials already live.
set -uo pipefail

DEADLINE="${1:?usage: watchdog.sh <deadline_epoch> <heartbeat_timeout> <fs_root>}"
HB_TIMEOUT="${2:?}"
FS="${3:?}"

mkdir -p "$FS/receipts" "$FS/logs"
echo "watchdog armed pid=$$ deadline=$DEADLINE heartbeat_timeout=${HB_TIMEOUT}s" >&2

stop_work() {  # stop_work <reason>
  local reason="$1"
  echo "watchdog: $reason -- stopping workload" >&2
  # Kill the measurement, not the whole box: the seal step below still needs a
  # working shell, and a half-written receipt is worse than none.
  pkill -f 'stage_measure.sh' 2>/dev/null || true
  pkill -f 'student_capture.py' 2>/dev/null || true
  # The pre-2026-08-31 name. A box bootstrapped from an older bundle is still
  # running a process called k6_student_capture.py, and a watchdog that cannot
  # stop the workload it was hired to stop is a billing leak.
  pkill -f 'k6_student_capture.py' 2>/dev/null || true
  pkill -f 'stream_score.py' 2>/dev/null || true
  pkill -f 'torchrun' 2>/dev/null || true
  sleep 5
  cat > "$FS/ABANDONED.json" <<EOF
{
  "schema": "fidelity-suite/abandoned.v1",
  "reason": "$reason",
  "stopped_at": "$(date -u +%FT%TZ)",
  "deadline_epoch": $DEADLINE,
  "heartbeat_timeout_seconds": $HB_TIMEOUT,
  "note": "The watchdog stopped the workload. It cannot destroy this instance -- see the header of bin/watchdog.sh. If you are reading this on a running box, it is still billing: destroy it."
}
EOF
  # Best-effort seal so a partial run still yields something checkable.
  if [ -x "$FS/bin/stage_measure.sh" ]; then
    bash "$FS/bin/stage_measure.sh" seal >>"$FS/logs/watchdog-seal.log" 2>&1 || true
  fi
  echo "watchdog: workload stopped, ABANDONED.json written" >&2
  exit 0
}

touch "$FS/heartbeat"
while true; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    stop_work "max-runtime deadline reached"
  fi
  if [ -f "$FS/heartbeat" ]; then
    hb=$(stat -c %Y "$FS/heartbeat" 2>/dev/null || stat -f %m "$FS/heartbeat" 2>/dev/null || echo "$now")
    age=$(( now - hb ))
    if [ "$age" -ge "$HB_TIMEOUT" ]; then
      stop_work "controller heartbeat stale (${age}s >= ${HB_TIMEOUT}s)"
    fi
  fi
  sleep 30
done
