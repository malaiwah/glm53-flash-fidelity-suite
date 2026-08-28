#!/usr/bin/env bash
# On-instance stage driver for the cloud recipe.
#
#   stage_measure.sh <setup|fetch_target|fetch_panel|measure|seal>
#
# The bootstrap is NOT reimplemented here.  `k6/stage_k6.sh setup` already owns
# the proven container bootstrap -- deadsnakes python3.12, CUDA 13.0, torch
# 2.11.0+cu130, transformers 5.16.1, the flash-attn wheel, pydantic/formatron/
# kbnf, and exllamav3 @ c5d9c657 with a rebuild guard -- and it is idempotent
# and sudo-aware, which matters because containers lose apt state across a
# pause.  This script arranges the layout that stage_k6.sh expects and calls it.
#
# Every stage writes a marker into $DONE, so a stage that already finished is a
# no-op.  That is what makes a spot preemption cost one stage instead of the
# whole run: resume, re-run setup (idempotent), and the driver skips forward.
#
# NEVER `set -x` in a stage where HF_TOKEN is in scope.
set -euo pipefail

STAGE="${1:?usage: stage_measure.sh <stage>}"
FS="${FIDELITY_FS_ROOT:-/home/jl_fs/fidelity}"
ROOT="${FIDELITY_K6_ROOT:-/home/jl_fs/glm53-k6}"
RCPT="$FS/receipts"
DONE="$RCPT/done"
LOGS="$FS/logs"
MODELS="$FS/models"
PANEL="$FS/panel"
VENV="$ROOT/venv"
PY="$VENV/bin/python"

mkdir -p "$RCPT" "$DONE" "$LOGS" "$MODELS" "$PANEL" "$FS/.secrets"
chmod 700 "$FS/.secrets" 2>/dev/null || true

# Config written by the controller before any stage runs.
CONF="$FS/job.json"
# Read a dotted path out of job.json. Uses the system python3 rather than the
# venv's, because this must work in `setup` -- before the venv exists.
jqget() {  # jqget <dotted.path> [default]
  python3 -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    print(sys.argv[3]); raise SystemExit(0)
cur = doc
for part in sys.argv[2].split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = sys.argv[3]
        break
print(cur if not isinstance(cur, (dict, list)) else json.dumps(cur))
' "$CONF" "$1" "${2-}"
}

log() { echo "[$(date -u +%FT%TZ)] stage_measure/$STAGE: $*"; }

marker="$DONE/$STAGE.done"
if [ "$STAGE" != "setup" ] && [ -f "$marker" ]; then
  log "already done (marker $marker) -- skipping"
  exit 0
fi

# Load the HF token from its 0600 file, never from argv or the environment of a
# command line that could be observed in the process list.
load_token() {
  if [ -f "$FS/.secrets/hf_token" ]; then
    HF_TOKEN="$(cat "$FS/.secrets/hf_token")"
    export HF_TOKEN
  fi
}

case "$STAGE" in

setup)
  log "delegating bootstrap to k6/stage_k6.sh setup"
  # stage_k6.sh guards on the BF16 directory existing. For the measurement
  # recipe we do not need the 643 GB of weights -- both published surfaces ship
  # non-routed tensors natively -- but the capture DOES bind
  # inventory.config_sha256/index_sha256 to the local files, so the skeleton
  # must carry the ORIGINAL config.json and model.safetensors.index.json bytes
  # verbatim. Rewriting either would break the seal.
  BF16_DIR="${BF16:-/home/jl_fs/models/bf16}"
  mkdir -p "$BF16_DIR"
  if [ ! -f "$BF16_DIR/config.json" ]; then
    log "fetching BF16 metadata skeleton (config + index only, ~4 MB)"
    python3 - "$BF16_DIR" <<'PY'
import sys, urllib.request, pathlib
root = pathlib.Path(sys.argv[1])
base = "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/main/"
for name in ("config.json", "model.safetensors.index.json"):
    dest = root / name
    if dest.exists():
        continue
    with urllib.request.urlopen(base + name, timeout=120) as r:
        dest.write_bytes(r.read())
    print("fetched", name, dest.stat().st_size, "bytes")
PY
  fi
  bash "$ROOT/stage_k6.sh" setup
  touch "$marker"
  log "done"
  ;;

fetch_target)
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  DEST="$MODELS/target"
  [ -n "$REPO" ] || { echo "job.json has no target.repo_id" >&2; exit 2; }
  log "fetching $REPO @ $REV -> $DEST"
  mkdir -p "$DEST"
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --revision "$REV" \
      --local-dir "$DEST" --max-workers 8 >>"$LOGS/fetch_target.log" 2>&1
  # Verify what the release seals, not what we hope: SHA256SUMS if published.
  if [ -f "$DEST/SHA256SUMS" ]; then
    log "verifying published SHA256SUMS"
    ( cd "$DEST" && sha256sum -c SHA256SUMS --quiet ) \
      | tee -a "$RCPT/shard-verification.txt"
  else
    log "no SHA256SUMS published; recording that fact in the receipt"
    echo "no SHA256SUMS in release" > "$RCPT/shard-verification.txt"
  fi
  df -h "$FS" | tee -a "$LOGS/fetch_target.log"
  touch "$marker"
  log "done"
  ;;

fetch_panel)
  load_token
  REPO="$(jqget panel.repo_id)"
  REV="$(jqget panel.revision)"
  log "fetching panel $REPO @ $REV (include-scoped)"
  # Include-scoping is not an optimisation, it is the difference between 32 GB
  # and 1.3 TB. The globs come from the panel descriptor, never from a constant.
  INCLUDES=$(python3 - "$CONF" <<'PY'
import json, sys, shlex
doc = json.load(open(sys.argv[1]))
for pattern in doc.get("panel", {}).get("include", ["*"]):
    print("--include", shlex.quote(pattern), end=" ")
PY
)
  eval HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --repo-type dataset --revision "$REV" \
      --local-dir "$PANEL" $INCLUDES >>"$LOGS/fetch_panel.log" 2>&1
  du -sh "$PANEL" | tee -a "$LOGS/fetch_panel.log"
  touch "$marker"
  log "done"
  ;;

measure)
  LANE="$(jqget lane streaming)"
  RUNS="$(jqget cold_runs 1)"
  log "lane=$LANE cold_runs=$RUNS"
  for run in $(seq 1 "$RUNS"); do
    # Receipt-resumable: a run whose capture receipt already exists is skipped,
    # so a preemption costs at most one in-flight run.
    if [ -f "$RCPT/run-$run/capture-receipt.json" ]; then
      log "run $run already captured -- skipping"
      continue
    fi
    mkdir -p "$RCPT/run-$run"
    log "run $run starting"
    python3 "$FS/bin/invoke_engine.py" --job "$CONF" --lane "$LANE" \
      --cold-run "$run" --out "$RCPT/run-$run" \
      2>&1 | tee -a "$LOGS/measure-run-$run.log"
  done
  touch "$marker"
  log "done"
  ;;

seal)
  log "sealing submission receipt"
  python3 "$FS/bin/seal_receipt.py" --job "$CONF" --receipts "$RCPT" \
      --out "$RCPT/measurement-receipt.json" 2>&1 | tee -a "$LOGS/seal.log"
  ( cd "$RCPT" && sha256sum measurement-receipt.json > RECEIPT.sha256 ) || true
  touch "$marker"
  log "done"
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  echo "stages: setup fetch_target fetch_panel measure seal" >&2
  exit 2
  ;;
esac
