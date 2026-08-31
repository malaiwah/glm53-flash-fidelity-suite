#!/bin/bash
# On-box stage driver for the hidden-replay equivalence experiment
# (k6/HIDDEN-REPLAY.md; Phaelon's sign-off protocol).
#
#   hidden_replay_stage.sh <setup|fetch|verify|run1|run2|run3|report|compare>
#
# Runs on a FRESH single-H200 JarvisLabs container with NO shared filesystem:
# every input is fetched from public HF repos, every path lives under $HR_ROOT
# (container-local disk).  Stages are guarded by done-markers / output files,
# so a spot preemption costs one stage.
#
# Layout (all overridable by env):
#   HR_ROOT   /home/hr          work root
#   SUITE     /home/suite       git clone of malaiwah/quant-fidelity-suite
#   PIPE      $HR_ROOT/pipeline brandonmmusic-max/glm-5.3-flash-exl3-4bpw @ pin + patches-v2 0001-0011
#   BF16      $HR_ROOT/models/bf16       sparse non-routed tree (fetch_nonrouted_sparse.py)
#   PACKED    $HR_ROOT/packed/k6         published K6 payload store (TR3-partsbin-v1, k6/)
#   TEACH     $HR_ROOT/teacher           sealed 25-window teacher + token panel (scoped fetch)
#
# HF token: $HR_ROOT/.secrets/hf_token (0600).  Read into env, never echoed,
# never on an argv.  NEVER set -x in this file.
set -euo pipefail

STAGE="${1:?usage: hidden_replay_stage.sh <stage>}"
HR_ROOT="${HR_ROOT:-/home/hr}"
SUITE="${SUITE:-/home/suite}"
PIPE="${PIPE:-$HR_ROOT/pipeline}"
VENV="$HR_ROOT/venv"
PY="$VENV/bin/python"
BF16="$HR_ROOT/models/bf16"
HEADD="$HR_ROOT/models/head"
PACKED="$HR_ROOT/packed/k6"
TEACH="$HR_ROOT/teacher"
CAL="$TEACH/calibration"
RUNS="$HR_ROOT/runs"
WORK="$HR_ROOT/work"
RCPT="$HR_ROOT/receipts"
LOGS="$HR_ROOT/logs"
TOOLS="$SUITE/k6/tools"
DONE="$RCPT/done"

PIPE_REPO=https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw
PIPE_PIN=ce1bf9706b6aa18435e2baccab63bdd72299257c
BF16_REPO=zai-org/GLM-5.3-Flash-BF16
BF16_REVISION=a6c167b62691b2bac901344b65cb651a70f53e43
PARTSBIN_REPO=malaiwah/GLM-5.3-Flash-TR3-partsbin-v1
TEACHER_REPO=brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits
PUBLISHED_HEAD_URL="https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/main/head/head.safetensors"
PUBLISHED_HEAD_SHA=47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0
SEALED_TEACHER_RECEIPT_SHA=2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5
SEALED_PANEL_RECEIPT_SHA=0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5
SEALED_INVENTORY_SHA=f56e9d6250e2d108f8307322f033e53c0ff26d5b2688ebf12b891c26439fea44
SEALED_STREAM_MEAN=0.013714888822596553
SEALED_STREAM_TOKENWISE_SHA=9657ede36b9f4b09a2c74916239c6d9a3baebce5f3fa64af7af388b0686aa284
SEALED_STREAM_CHECKPOINT_IDENTITY=a8668be3592493035e98a52994e0e3c43548a9757eadb79f7ae939f2f32de1c1

NTFY_URL="${QP_NTFY_URL:-https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33}"
ntfy() {  # ntfy <body> <title> <tags> [priority]
  curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} \
       -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true
}

mkdir -p "$HR_ROOT" "$RUNS" "$WORK" "$RCPT" "$DONE" "$LOGS" "$HR_ROOT/.secrets"
chmod 700 "$HR_ROOT/.secrets" 2>/dev/null || true

load_token() {
  if [ -f "$HR_ROOT/.secrets/hf_token" ]; then
    HF_TOKEN="$(cat "$HR_ROOT/.secrets/hf_token")"
    export HF_TOKEN
  fi
}

# The sealed artifact rows record the producer's /workspace paths; the identity
# check refuses symlinked FILES, so link DIRECTORIES (same rule as stage_k6.sh).
farm() {
  SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"
  $SUDO mkdir -p /workspace/artifacts/dataset /workspace/artifacts/evaluation /workspace/models/zai-org 2>/dev/null || true
  $SUDO ln -sfn "$CAL" /workspace/artifacts/dataset/calibration 2>/dev/null || true
  $SUDO ln -sfn "$TEACH" /workspace/artifacts/evaluation/glm53-teacher-final-ep4 2>/dev/null || true
  $SUDO ln -sfn "$BF16" /workspace/models/zai-org/GLM-5.3-Flash-BF16 2>/dev/null || true
}

export PYTHONPATH="$PIPE/src"
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$HR_ROOT/hf"

trap 'rc=$?; if [ $rc -eq 0 ]; then
        ntfy "hidden-replay stage $STAGE completed" "GLM53 hidden-replay OK: $STAGE" "white_check_mark"
      else
        ntfy "hidden-replay stage $STAGE FAILED rc=$rc" "GLM53 hidden-replay FAILED: $STAGE" "rotating_light" "high"
      fi' EXIT

case "$STAGE" in

setup)
  # measure_stream's own minimal bootstrap: python3.12 venv + the pinned
  # streaming runtime.  No exllamav3, no nvcc, no encoder toolchain.
  if ! command -v python3.12 >/dev/null; then
    ASROOT=""; [ "$(id -u)" = 0 ] || ASROOT="sudo"
    $ASROOT apt-get update -qq >/dev/null 2>&1 || true
    $ASROOT apt-get install -y -qq software-properties-common >/dev/null 2>&1 || true
    $ASROOT add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 || true
    $ASROOT apt-get update -qq >/dev/null 2>&1 || true
    for p in python3.12 python3.12-venv python3.12-dev; do
      $ASROOT apt-get install -y -qq "$p" >/dev/null 2>&1 || true
    done
  fi
  command -v python3.12 >/dev/null || { echo "python3.12 unavailable" >&2; exit 1; }
  [ -x "$PY" ] || python3.12 -m venv "$VENV"
  "$PY" -c "import sys; assert sys.version_info[:2]==(3,12)"
  "$VENV/bin/pip" -q install --upgrade pip
  "$PY" -c "import torch" 2>/dev/null || \
    "$VENV/bin/pip" -q install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
  "$PY" -c "import transformers, safetensors, numpy, accelerate" 2>/dev/null || \
    "$VENV/bin/pip" -q install "transformers==5.16.1" safetensors numpy accelerate
  "$PY" -c "import huggingface_hub, hf_transfer" 2>/dev/null || \
    "$VENV/bin/pip" -q install huggingface_hub hf_transfer
  "$PY" -c "
import torch, transformers
assert torch.__version__.startswith('2.11.0'), torch.__version__
assert transformers.__version__ == '5.16.1', transformers.__version__
print('stream env OK torch', torch.__version__, 'transformers', transformers.__version__)
" | tee "$RCPT/env-versions.txt"
  # pipeline @ pin + the FULL patches-v2 series 0001-0011 (the state of the
  # campaign tree when the sealed streaming K6 number was measured).
  if [ ! -d "$PIPE/.git" ]; then
    git clone -q "$PIPE_REPO" "$PIPE"
  fi
  if ! command -v patch >/dev/null; then
    ASROOT=""; [ "$(id -u)" = 0 ] || ASROOT="sudo"
    $ASROOT apt-get install -y -qq patch >/dev/null 2>&1 || true
  fi
  command -v patch >/dev/null || { echo "GNU patch unavailable (series 0005 creates a file the "\
"epoch-timestamp way, which git-apply refuses)" >&2; exit 1; }
  if ! grep -q "_STORED_BITS" "$PIPE/src/quant_pipeline/checkpoint/packed_payload.py"; then
    git -C "$PIPE" checkout -q "$PIPE_PIN"
    git -C "$PIPE" diff --quiet && git -C "$PIPE" diff --cached --quiet
    for p in "$SUITE"/k6/patches-v2/00{01,02,03,04,05,06,07,08,09,10,11}-*.patch; do
      ( cd "$PIPE" && patch -p1 -s < "$p" )
    done
    ( cd "$SUITE/k6/patches-v2" && sha256sum 00*.patch SERIES ) | tee "$RCPT/patches-applied.txt"
  else
    [ "$(git -C "$PIPE" rev-parse HEAD)" = "$PIPE_PIN" ] || { echo "$PIPE HEAD is not $PIPE_PIN" >&2; exit 1; }
    grep -q "K8_RECIPE_ID" "$PIPE/src/quant_pipeline/campaign/glm53_direct_k4.py" \
      || { echo "pipeline is part-patched (0001-0006 without 0007+); delete $PIPE and re-run setup" >&2; exit 1; }
  fi
  "$PY" -c "
import sys; sys.path.insert(0, '$PIPE/src')
import quant_pipeline.evaluation.glm53_logits, quant_pipeline.evaluation.glm53_packed_k4_reader
import quant_pipeline.campaign.glm53_direct_k4, quant_pipeline.core.artifacts
from quant_pipeline.normalization.absolute_v31 import ALLOWED_BITS
assert {6, 8} <= set(ALLOWED_BITS), ALLOWED_BITS
print('patched pipeline import OK')"
  "$PY" "$TOOLS/hidden_replay_selftest.py" --json "$RCPT/hidden-replay-selftest.json"
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv | tee "$RCPT/nvidia-smi.txt"
  touch "$DONE/setup.done"
  ;;

fetch)
  load_token
  HF="$VENV/bin/hf"
  if [ ! -f "$DONE/fetch-packed.done" ]; then
    "$HF" download "$PARTSBIN_REPO" --repo-type dataset \
      --include "k6/*" --local-dir "$HR_ROOT/packed" --max-workers 16 \
      >> "$LOGS/fetch-packed.log" 2>&1
    for need in contract.json inventory.json mtp-adapter-receipt.json; do
      test -f "$PACKED/$need" || { echo "packed fetch incomplete: $need missing" >&2; exit 1; }
    done
    test -d "$PACKED/payload-store/objects" || { echo "payload-store/objects missing" >&2; exit 1; }
    touch "$DONE/fetch-packed.done"
  fi
  if [ ! -f "$DONE/fetch-teacher.done" ]; then
    "$HF" download "$TEACHER_REPO" --repo-type dataset \
      --include "logits/window-*.safetensors" "capture-receipt.json" "backend.json" \
                "plan.json" "token-panel-receipt.json" "source-inventory.json" \
                "dataset-manifest.json" "calibration/panel-v1/*" \
      --local-dir "$TEACH" --max-workers 16 \
      >> "$LOGS/fetch-teacher.log" 2>&1
    test -f "$TEACH/capture-receipt.json" || { echo "teacher receipt missing" >&2; exit 1; }
    N=$(ls "$TEACH/logits"/window-*.safetensors | wc -l)
    [ "$N" = 25 ] || { echo "teacher window count $N != 25" >&2; exit 1; }
    test -f "$CAL/panel-v1/panel.receipt.json" || { echo "token panel receipt missing" >&2; exit 1; }
    touch "$DONE/fetch-teacher.done"
  fi
  if [ ! -f "$DONE/fetch-bf16.done" ]; then
    "$PY" "$TOOLS/fetch_nonrouted_sparse.py" \
      --repo "$BF16_REPO" --revision "$BF16_REVISION" \
      --inventory "$PACKED/inventory.json" \
      --dest "$BF16" --head-out "$HEADD/lm_head.safetensors" \
      --published-head-url "$PUBLISHED_HEAD_URL" \
      --published-head-sha "$PUBLISHED_HEAD_SHA" \
      --threads 12 --receipt "$RCPT/nonrouted-sparse-fetch.json" \
      2>&1 | tail -20
    touch "$DONE/fetch-bf16.done"
  fi
  farm
  df -h "$HR_ROOT" | tee -a "$LOGS/fetch.log"
  du -sh "$PACKED" "$TEACH" "$BF16" 2>/dev/null | tee -a "$LOGS/fetch.log"
  touch "$DONE/fetch.done"
  ;;

verify)
  farm
  "$PY" - <<PYEOF
import json, sys
teacher = json.load(open("$TEACH/capture-receipt.json"))
assert teacher.get("receipt_sha256") == "$SEALED_TEACHER_RECEIPT_SHA", teacher.get("receipt_sha256")
panel = json.load(open("$CAL/panel-v1/panel.receipt.json"))
assert panel.get("receipt_sha256") == "$SEALED_PANEL_RECEIPT_SHA", panel.get("receipt_sha256")
inventory = json.load(open("$PACKED/inventory.json"))
assert inventory.get("inventory_sha256") == "$SEALED_INVENTORY_SHA", inventory.get("inventory_sha256")
missing = [row["path"] for row in panel.get("artifacts", [])
           if not __import__("pathlib").Path(row["path"]).is_file()][:4]
assert not missing, f"panel artifact paths unresolved (symlink farm?): {missing}"
print("verify OK: teacher receipt, panel receipt, inventory, artifact paths")
PYEOF
  "$PY" "$TOOLS/stream_score_selftest.py" \
      --packed-root "$PACKED" --require a,b,d,e \
      --pipeline-root "$PIPE" --json "$RCPT/stream-selftest.json" \
      > "$LOGS/stream-selftest.log" 2>&1 \
    || { tail -30 "$LOGS/stream-selftest.log" >&2; exit 1; }
  echo "L1 offline ladder PASSED"
  touch "$DONE/verify.done"
  ;;

run1|run2|run3)
  N="${STAGE#run}"
  farm
  OUT="$RUNS/hidden-run$N"
  if [ -f "$OUT/hidden-capture.json" ]; then
    echo "cold run $N already captured (hidden-capture.json present)"
    exit 0
  fi
  rm -rf "$OUT"
  QP_GLM53_EP_SIZE=8 "$PY" "$TOOLS/hidden_replay.py" capture -- \
      --source payload-store --packed-root "$PACKED" \
      --bf16 "$BF16" --teacher "$TEACH" \
      --token-panel "$CAL/panel-v1/panel.receipt.json" \
      --out "$OUT" --cold-run "$N" --profile k6 \
      --device cuda:0 --ep-emulate 8 --reduce-order fp32 \
      --decode-cache none --decode-threads "$(nproc)" \
      --work-dir "$WORK" \
      > "$LOGS/hidden-run$N.log" 2>&1 \
    || { tail -40 "$LOGS/hidden-run$N.log" >&2; exit 1; }
  tail -3 "$LOGS/hidden-run$N.log"
  ntfy "cold run $N captured (logits + hiddens)" "GLM53 hidden-replay run$N done" "checkered_flag"
  ;;

report)
  farm
  "$PY" "$TOOLS/kld_report.py" --profile k6-stream \
      --teacher "$TEACH" --runs "$RUNS/hidden-run1" "$RUNS/hidden-run2" "$RUNS/hidden-run3" \
      --fp8-baseline 0.020615 --k4-baseline 0.024555 \
      --device cuda:0 --chunk-positions 16 \
      --out "$RCPT/stream-k6-kld-3run.json" \
      > "$LOGS/kld-report.log" 2>&1 \
    || { tail -40 "$LOGS/kld-report.log" >&2; exit 1; }
  "$PY" - <<PYEOF
import json
summary = json.load(open("$RCPT/stream-k6-kld-3run.json"))
report = json.load(open("$RUNS/hidden-run1/kld-report.json"))
check = {
    "schema": "malaiwah.glm53-hidden-replay-reproduction-check.v1",
    "sealed_stream_mean": $SEALED_STREAM_MEAN,
    "measured_mean": summary["measured_mean_kld"],
    "mean_reproduced_exactly": summary["measured_mean_kld"] == $SEALED_STREAM_MEAN,
    "sealed_tokenwise_sha256": "$SEALED_STREAM_TOKENWISE_SHA",
    "distinct_tokenwise_kld_sha256": summary["distinct_tokenwise_kld_sha256"],
    "tokenwise_sha_matches_sealed": summary["distinct_tokenwise_kld_sha256"] == ["$SEALED_STREAM_TOKENWISE_SHA"],
    "sealed_checkpoint_identity": "$SEALED_STREAM_CHECKPOINT_IDENTITY",
    "student_checkpoint_identity_sha256": report.get("student_checkpoint_identity_sha256"),
    "scored_the_sealed_k6_surface": report.get("student_checkpoint_identity_sha256") == "$SEALED_STREAM_CHECKPOINT_IDENTITY",
    "bitwise_deterministic_across_runs": summary["bitwise_deterministic"],
    "run_means": summary["run_means"],
}
open("$RCPT/reproduction-check.json", "w").write(json.dumps(check, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: check[k] for k in ("measured_mean", "mean_reproduced_exactly",
      "tokenwise_sha_matches_sealed", "scored_the_sealed_k6_surface",
      "bitwise_deterministic_across_runs")}, sort_keys=True))
PYEOF
  ntfy "path-A report done; see reproduction-check.json" "GLM53 hidden-replay path A" "bar_chart"
  touch "$DONE/report.done"
  ;;

compare)
  farm
  "$PY" "$TOOLS/hidden_replay.py" compare \
      --runs "$RUNS/hidden-run1" "$RUNS/hidden-run2" "$RUNS/hidden-run3" \
      --head "$HEADD/lm_head.safetensors" \
      --teacher "$TEACH" --device cuda:0 --chunk-positions 16 \
      --alt-vocab-chunk 8192 --invariance-run 1 \
      --out "$RCPT/hidden-replay-comparator.json" \
      > "$LOGS/compare.log" 2>&1 \
    || { tail -40 "$LOGS/compare.log" >&2; exit 1; }
  tail -2 "$LOGS/compare.log"
  ntfy "hidden-replay comparator done" "GLM53 hidden-replay comparator" "microscope"
  touch "$DONE/compare.done"
  ;;

*)
  echo "unknown stage: $STAGE (setup fetch verify run1 run2 run3 report compare)" >&2
  exit 2
  ;;
esac
