#!/bin/bash
# Self-driving remainder of the PREP phase on the L4 VM.
# Waits for the already-running BF16 download to finish, then chains:
#   download_fp8 -> heads -> smoke -> fsbench
# Survives the Mac supervisor sleeping; every stage still posts ntfy via prep_stage.sh.
set -uo pipefail
ROOT=/home/ubuntu/glm53
FS=/home/jl_fs/glm53
NTFY_URL="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"
ntfy() { curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true; }

echo "waiting for BF16 download to complete..."
for i in $(seq 1 480); do
  [ -f "$FS/models/bf16/revision.txt" ] && break
  sleep 60
done
if [ ! -f "$FS/models/bf16/revision.txt" ]; then
  ntfy "BF16 download did not finish within 8h — prep chain aborting" "GLM53 prep chain STALLED" "rotating_light" "high"
  exit 1
fi

if [ -d "$ROOT/bundle.new" ]; then
  rm -rf "$ROOT/bundle.old"; mv "$ROOT/bundle" "$ROOT/bundle.old"; mv "$ROOT/bundle.new" "$ROOT/bundle"
  ntfy "bundle swapped to r3 (activation tools included)" "GLM53 prep bundle swap" "arrows_counterclockwise"
fi

set -e
bash "$ROOT/bundle/remote/prep_stage.sh" download_fp8
bash "$ROOT/bundle/remote/prep_stage.sh" heads
bash "$ROOT/bundle/remote/prep_stage.sh" smoke
bash "$ROOT/bundle/remote/prep_stage.sh" act_smoke
bash "$ROOT/bundle/remote/prep_stage.sh" fsbench
"$ROOT/hfenv/bin/hf" download brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits \
    --repo-type dataset --local-dir "$FS/crosscheck/bm-teacher-logits" --max-workers 4 \
    && ntfy "brandonmusic teacher-logits downloaded for cross-validation" "GLM53 prep: crosscheck data" "handshake" \
    || ntfy "brandonmusic teacher-logits download failed - cross-check will be skipped" "GLM53 prep: crosscheck skipped" "warning"
touch "$FS/PREP_COMPLETE"
ntfy "All prep green: both checkpoints on the filesystem, image tarball saved, heads extracted + FP8 head-equality receipt, harness smoke passed inside the glm53-flash image. 8x launch waits only on the top-off. Prep VM can be paused." "GLM53 PREP COMPLETE" "tada"
echo PREP_CHAIN_DONE
