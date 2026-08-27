#!/bin/bash
# Download a pinned GLM-5.3-Flash checkpoint on the VM host.
#   download_model_vm.sh <repo_id> <revision_sha> <dst_dir>
set -euo pipefail
REPO="$1"; REV="$2"; DST="$3"
ROOT=/home/ubuntu/glm53
export HF_HOME="$ROOT/hf-cache"
export HF_XET_HIGH_PERFORMANCE=1

"$ROOT/hfenv/bin/hf" download "$REPO" --revision "$REV" --local-dir "$DST" --max-workers 8
echo "$REV" > "$DST/revision.txt"
du -sh "$DST"
echo DOWNLOAD_DONE "$REPO" "$REV"
