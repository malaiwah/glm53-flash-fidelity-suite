#!/usr/bin/env bash
# On-instance watchdog -- teardown layer L1.
#
# Normal mode:
#   watchdog.sh <deadline_epoch> <heartbeat_timeout_seconds> <fs_root> [pgid_record]
#
# Stage arming primitive (call after launching the stage with setsid):
#   watchdog.sh --record-stage-pgid <fs_root> <stage_leader_pid> [pgid_record]
#
# The watchdog never searches process command lines.  It signals only the
# recorded process group after proving either the setsid leader identity or
# every surviving group member's recorded session.  Missing, stale, reused,
# or unprovable identity fails closed rather than risking another process.
set -uo pipefail
umask 077

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

proc_identity() { # proc_identity <pid>; prints: pgrp session start_ticks
  local pid="$1" line rest state ppid pgrp session tty_nr tpgid flags
  local minflt cminflt majflt cmajflt utime stime cutime cstime priority nice
  local threads itreal start_ticks remainder
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [ -r "/proc/$pid/stat" ] || return 1
  IFS= read -r line < "/proc/$pid/stat" || return 1
  rest="${line##*) }"
  read -r state ppid pgrp session tty_nr tpgid flags minflt cminflt \
    majflt cmajflt utime stime cutime cstime priority nice threads itreal \
    start_ticks remainder <<<"$rest"
  [[ "$pgrp" =~ ^[1-9][0-9]*$ && "$session" =~ ^[1-9][0-9]*$ \
     && "$start_ticks" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s %s %s\n' "$pgrp" "$session" "$start_ticks"
}

atomic_file() { # atomic_file <target>; content on stdin
  local target="$1"
  python3 -c '
import os
import secrets
import sys

target = os.path.abspath(sys.argv[1])
directory = os.path.dirname(target)
basename = os.path.basename(target)
dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
dir_fd = os.open(directory, dir_flags)
tmp_name = None
try:
    create_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0))
    for unused in range(100):
        candidate = "." + basename + "." + secrets.token_hex(12)
        try:
            fd = os.open(candidate, create_flags, 0o600, dir_fd=dir_fd)
            tmp_name = candidate
            break
        except FileExistsError:
            continue
    if tmp_name is None:
        raise OSError("could not allocate atomic temporary file")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(sys.stdin.buffer.read())
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, basename, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        tmp_name = None
        os.fsync(dir_fd)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
finally:
    os.close(dir_fd)
' "$target"
}

record_stage_pgid() {
  local fs="${1:?missing fs_root}" leader="${2:?missing stage leader pid}"
  local record="${3:-$fs/runtime/stage.pgid}" identity pgrp session start_ticks now
  [[ "$leader" =~ ^[1-9][0-9]*$ ]] || {
    echo "watchdog: stage leader pid is not numeric" >&2; return 2; }
  mkdir -p "$fs/runtime" "$fs/receipts"
  identity="$(proc_identity "$leader")" || {
    echo "watchdog: stage leader $leader is not a live Linux process" >&2; return 2; }
  read -r pgrp session start_ticks <<<"$identity"
  # setsid makes the leader its own process-group and session leader.  Without
  # both equalities, group signalling could include the caller's SSH shell.
  if [ "$pgrp" != "$leader" ] || [ "$session" != "$leader" ]; then
    echo "watchdog: stage $leader is not an isolated setsid process group" >&2
    return 2
  fi
  now="$(date +%s)"
  {
    printf 'version=1\n'
    printf 'leader_pid=%s\n' "$leader"
    printf 'pgid=%s\n' "$pgrp"
    printf 'session_id=%s\n' "$session"
    printf 'start_ticks=%s\n' "$start_ticks"
    printf 'recorded_at_epoch=%s\n' "$now"
  } | atomic_file "$record" || return 2
  {
    printf '{\n'
    printf '  "schema": "fidelity-suite/watchdog-stage-pgid.v1",\n'
    printf '  "leader_pid": %s,\n' "$leader"
    printf '  "pgid": %s,\n' "$pgrp"
    printf '  "session_id": %s,\n' "$session"
    printf '  "proc_start_ticks": %s,\n' "$start_ticks"
    printf '  "recorded_at_epoch": %s,\n' "$now"
    printf '  "record_path": "%s"\n' "$(json_escape "$record")"
    printf '}\n'
  } | atomic_file "$fs/receipts/watchdog-stage-pgid.json"
}

if [ "${1:-}" = "--record-stage-pgid" ]; then
  shift
  record_stage_pgid "$@"
  exit $?
fi

DEADLINE="${1:?usage: watchdog.sh <deadline_epoch> <heartbeat_timeout> <fs_root> [pgid_record]}"
HB_TIMEOUT="${2:?missing heartbeat timeout}"
FS="${3:?missing fs_root}"
PGID_RECORD="${4:-$FS/runtime/stage.pgid}"
[[ "$DEADLINE" =~ ^[1-9][0-9]*$ ]] || {
  echo "watchdog: deadline must be a positive epoch" >&2; exit 2; }
[[ "$HB_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "watchdog: heartbeat timeout must be positive" >&2; exit 2; }

mkdir -p "$FS/receipts" "$FS/logs" "$FS/runtime"
WATCHDOG_IDENTITY="$(proc_identity "$$")" || {
  echo "watchdog: cannot establish own Linux process identity" >&2; exit 2; }
read -r WATCHDOG_PGID WATCHDOG_SESSION WATCHDOG_START_TICKS <<<"$WATCHDOG_IDENTITY"
ARMED_AT="$(date +%s)"
{
  printf '{\n'
  printf '  "schema": "fidelity-suite/watchdog-armed.v2",\n'
  printf '  "watchdog_pid": %s,\n' "$$"
  printf '  "watchdog_pgid": %s,\n' "$WATCHDOG_PGID"
  printf '  "proc_start_ticks": %s,\n' "$WATCHDOG_START_TICKS"
  printf '  "armed_at_epoch": %s,\n' "$ARMED_AT"
  printf '  "deadline_epoch": %s,\n' "$DEADLINE"
  printf '  "heartbeat_timeout_seconds": %s,\n' "$HB_TIMEOUT"
  printf '  "stage_pgid_record": "%s"\n' "$(json_escape "$PGID_RECORD")"
  printf '}\n'
} | atomic_file "$FS/receipts/watchdog-armed.json" || {
  echo "watchdog: could not write arming proof" >&2; exit 2; }
echo "watchdog armed pid=$$ deadline=$DEADLINE heartbeat_timeout=${HB_TIMEOUT}s pgid_record=$PGID_RECORD" >&2

write_abandoned() { # write_abandoned <reason> <stopped> <detail>
  local reason="$1" stopped="$2" detail="$3" now
  now="$(date -u +%FT%TZ)"
  {
    printf '{\n'
    printf '  "schema": "fidelity-suite/abandoned.v2",\n'
    printf '  "reason": "%s",\n' "$(json_escape "$reason")"
    printf '  "stopped_at": "%s",\n' "$now"
    printf '  "deadline_epoch": %s,\n' "$DEADLINE"
    printf '  "heartbeat_timeout_seconds": %s,\n' "$HB_TIMEOUT"
    printf '  "stage_process_group_stopped": %s,\n' "$stopped"
    printf '  "stage_pgid_record": "%s",\n' "$(json_escape "$PGID_RECORD")"
    printf '  "detail": "%s",\n' "$(json_escape "$detail")"
    printf '  "note": "This watchdog cannot destroy the instance. A retained controller lease must confirm provider absence and billing reconciliation."\n'
    printf '}\n'
  } | atomic_file "$FS/ABANDONED.json" || {
    echo "watchdog: could not write ABANDONED.json" >&2
    exit 92
  }
}

load_stage_record() {
  local key value version="" leader="" pgid="" session="" start="" recorded=""
  [ -f "$PGID_RECORD" ] && [ ! -L "$PGID_RECORD" ] || return 1
  while IFS='=' read -r key value; do
    case "$key" in
      version) version="$value" ;;
      leader_pid) leader="$value" ;;
      pgid) pgid="$value" ;;
      session_id) session="$value" ;;
      start_ticks) start="$value" ;;
      recorded_at_epoch) recorded="$value" ;;
      *) return 1 ;;
    esac
  done < "$PGID_RECORD"
  [[ "$version" = 1 && "$leader" =~ ^[1-9][0-9]*$ \
     && "$pgid" =~ ^[1-9][0-9]*$ && "$session" =~ ^[1-9][0-9]*$ \
     && "$start" =~ ^[1-9][0-9]*$ && "$recorded" =~ ^[1-9][0-9]*$ ]] || return 1
  STAGE_LEADER="$leader"
  STAGE_PGID="$pgid"
  STAGE_SESSION="$session"
  STAGE_START_TICKS="$start"
}

group_session_is_exact() {
  local path pid identity member_pgrp member_session member_start found=1
  for path in /proc/[0-9]*; do
    [ -e "$path" ] || continue
    pid="${path#/proc/}"
    identity="$(proc_identity "$pid")" || continue
    read -r member_pgrp member_session member_start <<<"$identity"
    if [ "$member_pgrp" = "$STAGE_PGID" ]; then
      found=0
      [ "$member_session" = "$STAGE_SESSION" ] || return 2
    fi
  done
  return "$found"
}

stop_work() { # stop_work <reason>
  local reason="$1" identity current_pgrp current_session current_start
  local still_live=0 group_live=0
  echo "watchdog: $reason -- stopping recorded stage process group" >&2
  if ! load_stage_record; then
    write_abandoned "$reason" false "missing or malformed stage PGID record; no process was signalled"
    echo "watchdog: refusing unproven process-group signal" >&2
    exit 91
  fi
  if [ "$STAGE_LEADER" != "$STAGE_PGID" ] || [ "$STAGE_LEADER" != "$STAGE_SESSION" ] \
     || [ "$STAGE_PGID" = "$WATCHDOG_PGID" ]; then
    write_abandoned "$reason" false "stage PGID record is not an isolated group; no process was signalled"
    echo "watchdog: refusing unsafe stage PGID record" >&2
    exit 91
  fi
  if identity="$(proc_identity "$STAGE_LEADER")"; then
    read -r current_pgrp current_session current_start <<<"$identity"
    if [ "$current_pgrp" != "$STAGE_PGID" ] \
       || [ "$current_session" != "$STAGE_SESSION" ] \
       || [ "$current_start" != "$STAGE_START_TICKS" ]; then
      write_abandoned "$reason" false "stage PID was reused or changed identity; no process was signalled"
      echo "watchdog: refusing signal after stage identity mismatch" >&2
      exit 91
    fi
  elif [ -e "/proc/$STAGE_LEADER" ]; then
    write_abandoned "$reason" false "stage leader identity became unreadable; no process was signalled"
    exit 91
  fi

  if kill -0 -- "-$STAGE_PGID" 2>/dev/null; then
    group_live=1
    if ! group_session_is_exact; then
      write_abandoned "$reason" false "recorded PGID exists without provable membership in its recorded session; no process was signalled"
      echo "watchdog: refusing signal to a reused or unprovable process group" >&2
      exit 91
    fi
  fi
  if [ "$group_live" -eq 1 ]; then
    # A surviving child retains both the recorded PGID and recorded session ID.
    kill -TERM -- "-$STAGE_PGID" 2>/dev/null || true
    for _unused in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 -- "-$STAGE_PGID" 2>/dev/null || { still_live=0; break; }
      still_live=1
      sleep 0.5
    done
    if [ "$still_live" -eq 1 ]; then
      kill -KILL -- "-$STAGE_PGID" 2>/dev/null || true
      sleep 1
    fi
  fi
  if kill -0 -- "-$STAGE_PGID" 2>/dev/null; then
    write_abandoned "$reason" false "recorded stage process group survived TERM and KILL"
    exit 91
  fi
  if [ "$group_live" -eq 1 ]; then
    write_abandoned "$reason" true "exact recorded stage process group received TERM then, if needed, KILL"
  else
    write_abandoned "$reason" true "exact recorded stage process group was already absent"
  fi
  echo "watchdog: recorded workload stopped, ABANDONED.json written" >&2
  exit 0
}

while true; do
  now="$(date +%s)"
  if [ "$now" -ge "$DEADLINE" ]; then
    stop_work "max-runtime deadline reached"
  fi
  if [ ! -f "$FS/heartbeat" ] || [ -L "$FS/heartbeat" ]; then
    stop_work "controller heartbeat missing or not a regular file"
  fi
  if ! hb="$(stat -c %Y "$FS/heartbeat" 2>/dev/null)"; then
    if ! hb="$(stat -f %m "$FS/heartbeat" 2>/dev/null)"; then
      stop_work "controller heartbeat metadata unreadable"
    fi
  fi
  if ! [[ "$hb" =~ ^[0-9]+$ ]]; then
    stop_work "controller heartbeat metadata is not an integer epoch"
  fi
  if [ "$hb" -gt "$now" ]; then
    stop_work "controller heartbeat mtime is in the future"
  fi
  age=$(( now - hb ))
  if [ "$age" -ge "$HB_TIMEOUT" ]; then
    stop_work "controller heartbeat stale (${age}s >= ${HB_TIMEOUT}s)"
  fi
  sleep 30
done
