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
export VENV

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

# Every stage after setup runs under the venv setup builds.  Without this guard
# a stage launched before setup finished died as a bare `exit 127` -- "not
# found" -- which says nothing about the actual dependency.
if [ "$STAGE" != "setup" ] && [ ! -x "$PY" ]; then
  echo "stage_measure: error: $STAGE needs the venv interpreter $PY, which does not exist yet." >&2
  echo "  The setup stage builds it. Run (or finish) 'stage_measure.sh setup' first." >&2
  exit 3
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
  # The measurement lane owns its bootstrap (bin/bootstrap_measure.sh).  It
  # used to call k6/stage_k6.sh, which (a) was never in the upload bundle and
  # (b) hard-stops a decode-only run on an ENCODER closure gate.  See that
  # script's header for the full reasoning.
  #
  # The official BF16 config + index are still fetched: the capture binds
  # inventory.config_sha256/index_sha256 to local files, and the exl3hf
  # materializer checks its produced non-routed name set against the official
  # index.  Both need the ORIGINAL bytes -- at the PINNED revision, not main,
  # which can move under us between two measurements of the same artifact.
  BF16_DIR="${BF16:-/home/jl_fs/models/bf16}"
  BF16_REV="$(jqget official_bf16_revision a6c167b62691b2bac901344b65cb651a70f53e43)"
  mkdir -p "$BF16_DIR" "$ROOT"
  if [ ! -f "$BF16_DIR/config.json" ] || [ ! -f "$BF16_DIR/model.safetensors.index.json" ]; then
    log "fetching BF16 metadata skeleton @ $BF16_REV (config + index only, ~16 MB)"
    python3 - "$BF16_DIR" "$BF16_REV" <<'PYSKEL'
import sys, urllib.request, pathlib
root, rev = pathlib.Path(sys.argv[1]), sys.argv[2]
base = "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/%s/" % rev
for name in ("config.json", "model.safetensors.index.json"):
    dest = root / name
    if dest.exists():
        continue
    with urllib.request.urlopen(base + name, timeout=300) as r:
        dest.write_bytes(r.read())
    print("fetched", name, dest.stat().st_size, "bytes")
PYSKEL
  fi
  # patches-v2 ships in the upload tree; the pipeline clone expects it at $ROOT.
  if [ -d "$FS/k6/patches-v2" ]; then
    mkdir -p "$ROOT/patches-v2"
    cp -f "$FS"/k6/patches-v2/* "$ROOT/patches-v2/"
  fi
  log "bootstrapping (measurement-only recipe)"
  bash "$FS/bin/bootstrap_measure.sh" 2>&1 | tee -a "$LOGS/setup.log"
  df -h "$FS" | tee -a "$LOGS/setup.log"
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
  #
  # `sha256sum -c` over the whole list is the wrong instrument for a MIRROR.
  # Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw republishes brandonmusic's weights
  # byte-for-byte but trims his 120 .materialization/shards/*.json sidecars and
  # ships its own README/LICENSE -- while copying his SHA256SUMS verbatim. So
  # `-c` reports 122 failures, all of them files that are absent or deliberately
  # different, and NONE of them a weight. Under `set -o pipefail` that non-zero
  # exit killed the stage after a 175 GB download and a full checksum pass.
  #
  # What the verification has to answer is narrower and stronger: does every
  # WEIGHT file present on disk match the digest the release published for it,
  # and is every weight file covered by the list at all? Entries for files this
  # repo does not publish are REPORTED, never silently dropped and never
  # treated as a weight failure.
  if [ -f "$DEST/SHA256SUMS" ]; then
    log "verifying published SHA256SUMS (weights fail-closed; absent sidecars reported)"
    python3 "$FS/bin/verify_published_sums.py" --root "$DEST" \
        --out "$RCPT/shard-verification.json" \
        2>&1 | tee "$RCPT/shard-verification.txt"
  else
    log "no SHA256SUMS published; recording that fact in the receipt"
    echo "no SHA256SUMS in release" > "$RCPT/shard-verification.txt"
  fi
  # A surface that can verify its release's PUBLISHED seal does it here --
  # right after the bytes land, ~10 minutes in -- not at capture time four
  # stages and three GPU-hours later. The same pass writes the artifact's own
  # scope, which seal_receipt prefers over the registry's record and over its
  # pessimistic default (M1 lesson: recording `unknown` when the producer
  # published the answer is the same failure as guessing).
  SURFACE="$(jqget target.surface)"
  if [ "$SURFACE" = "tr3-published" ]; then
    log "verifying the release's published seal (tr3)"
    "$VENV/bin/python" "$FS/k6/tools/tr3_surface.py" verify \
        --root "$DEST" --repo "$REPO" --revision "$REV" \
        --shards crosscheck --out "$RCPT/artifact-seal-verification.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    "$VENV/bin/python" "$FS/k6/tools/tr3_surface.py" scope \
        --root "$DEST" --repo "$REPO" --revision "$REV" \
        --out "$RCPT/artifact-scope.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    log "seal verified; scope written to $RCPT/artifact-scope.json"
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
  # The sealed token-panel receipt names its 667 artifacts by ABSOLUTE producer
  # path and verifies each by digest. Stage them there now, where a miss is one
  # named file, rather than at load_panel_windows four stages later.
  python3 "$FS/bin/stage_panel_paths.py" --panel "$PANEL" \
      2>&1 | tee -a "$LOGS/fetch_panel.log"
  touch "$marker"
  log "done"
  ;;

materialize)
  # exl3hf only: dequantize the artifact's non-routed function into the BF16
  # tree the streaming engine loads as --bf16.  A no-op for other surfaces.
  SURFACE="$(jqget target.surface)"
  if [ "$SURFACE" != "exl3hf" ]; then
    log "surface=$SURFACE needs no materialization -- skipping"
    touch "$marker"; exit 0
  fi
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  BF16_DIR="${BF16:-/home/jl_fs/models/bf16}"
  log "materializing non-routed BF16 tree from $MODELS/target"
  "$VENV/bin/python" "$FS/k6/tools/exl3hf_surface.py" materialize \
      --root "$MODELS/target" --out "$MODELS/target-bf16-materialized" \
      --device cuda --source-repo "$REPO" --source-revision "$REV" \
      --official-index "$BF16_DIR/model.safetensors.index.json" \
      2>&1 | tee -a "$LOGS/materialize.log"
  df -h "$FS" | tee -a "$LOGS/materialize.log"
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
    "$PY" "$FS/bin/invoke_engine.py" --job "$CONF" --lane "$LANE" \
      --cold-run "$run" --out "$RCPT/run-$run" \
      2>&1 | tee -a "$LOGS/measure-run-$run.log"
  done
  touch "$marker"
  log "done"
  ;;

score)
  # stream_score CAPTURES logits; the divergence is computed here, across the
  # cold runs, by the lane's pinned scorer.  Without this stage `seal` finds no
  # kld-report.json and exits 2 -- after the whole rental is spent.
  LANE="$(jqget lane streaming)"
  log "scoring cold runs (lane=$LANE)"
  "$PY" "$FS/bin/invoke_scorer.py" --job "$CONF" --lane "$LANE" \
      --receipts "$RCPT" --device "${KLD_DEVICE:-cuda}" \
      2>&1 | tee -a "$LOGS/score.log"
  # The fp32 student logits are transient by design: ~31.7 GB per cold run,
  # and the divergence they were captured for is now computed and sealed. They
  # also sit inside the receipts tree the controller downloads at teardown, so
  # leaving them turns a receipts pull into a 63 GB transfer that times out
  # (observed: `jl download ... timed out after 300.0 seconds`).
  KEEP="$(jqget keep_student_logits false)"
  if [ "$KEEP" != "True" ] && [ "$KEEP" != "true" ]; then
    for d in "$RCPT"/run-*/logits; do
      [ -d "$d" ] || continue
      log "removing transient student logits: $d ($(du -sh "$d" | cut -f1))"
      rm -rf "$d"
    done
  else
    log "keep_student_logits is set -- the per-run logit trees are retained"
  fi
  df -h "$FS" | tee -a "$LOGS/score.log"
  touch "$marker"
  log "done"
  ;;

seal)
  log "sealing submission receipt"
  "$PY" "$FS/bin/seal_receipt.py" --job "$CONF" --receipts "$RCPT" \
      --out "$RCPT/measurement-receipt.json" 2>&1 | tee -a "$LOGS/seal.log"
  ( cd "$RCPT" && sha256sum measurement-receipt.json > RECEIPT.sha256 ) || true
  touch "$marker"
  log "done"
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  echo "stages: setup fetch_target fetch_panel materialize measure score seal" >&2
  exit 2
  ;;
esac
