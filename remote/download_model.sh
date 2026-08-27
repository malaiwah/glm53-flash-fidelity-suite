#!/bin/bash
# Download a pinned GLM-5.3-Flash checkpoint.
#   download_model.sh <repo_id> <revision_sha> <dst_dir>
# Writes revision.txt into the snapshot so fidelity.py's model_identity pins it.
set -euo pipefail
REPO="$1"; REV="$2"; DST="$3"
ROOT=/home/glm53
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/home/glm53/hf-cache

"$ROOT/venv/bin/hf" download "$REPO" --revision "$REV" --local-dir "$DST" \
  --exclude "*.pth" "original/*"
echo "$REV" > "$DST/revision.txt"
du -sh "$DST"
ls "$DST" | head -20
echo DOWNLOAD_DONE "$REPO" "$REV"
