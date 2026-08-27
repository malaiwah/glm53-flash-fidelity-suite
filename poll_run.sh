#!/bin/bash
# Poll a jl run until it finishes; print final state + tail. Exit 0 on success.
#   poll_run.sh <run_id> [interval_s] [max_minutes]
RID="$1"; INT="${2:-90}"; MAXMIN="${3:-360}"
END=$(( $(date +%s) + MAXMIN*60 ))
while true; do
  ST=$(jl run status "$RID" --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('state') or d.get('status') or 'unknown')" 2>/dev/null)
  case "$ST" in
    succeeded) echo "RUN_OK $RID"; jl run logs "$RID" --tail 25 2>/dev/null; exit 0 ;;
    failed)    echo "RUN_FAILED $RID"; jl run logs "$RID" --tail 60 2>/dev/null; exit 1 ;;
  esac
  [ "$(date +%s)" -gt "$END" ] && { echo "RUN_TIMEOUT $RID state=$ST"; exit 2; }
  sleep "$INT"
done
