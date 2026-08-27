#!/bin/bash
# Package tools + suite + remote scripts for upload to the VM.
set -euo pipefail
cd "$(dirname "$0")"
test -f suite/suite-manifest.json || { echo "suite not built yet" >&2; exit 1; }
IMAGE_REF="${IMAGE_REF:-vllm/vllm-openai@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4}"
rm -rf bundle_stage bundle.tar.gz
mkdir -p bundle_stage/bundle
cp -r tools suite calsuite remote cal_data bundle_stage/bundle/
sed -i '' "s|__IMAGE_REF__|${IMAGE_REF}|" bundle_stage/bundle/remote/vm_setup.sh bundle_stage/bundle/remote/stage.sh
rm -f bundle_stage/bundle/remote/setup_env.sh bundle_stage/bundle/remote/download_model.sh
find bundle_stage \( -name '__pycache__' -o -name '._*' -o -name '.DS_Store' \) -exec rm -rf {} + 2>/dev/null || true
COPYFILE_DISABLE=1 tar -czf bundle.tar.gz -C bundle_stage bundle
rm -rf bundle_stage
du -sh bundle.tar.gz
echo BUNDLE_DONE "$IMAGE_REF"
