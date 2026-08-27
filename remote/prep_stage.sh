#!/bin/bash
# Prep-phase driver for the CHEAP VM (L4, IN2) writing to the shared filesystem.
# Usage: prep_stage.sh <stage>
# Filesystem layout produced: /home/jl_fs/glm53/{models/{bf16,fp8,smoke},image,out,smoke}
set -euo pipefail
ROOT=/home/ubuntu/glm53            # bundle + hfenv live on the VM disk
FS=/home/jl_fs/glm53               # heavy artifacts live on the shared filesystem
IMAGE_REF="$(awk '{print $1}' "$ROOT/out/image-pin.txt" 2>/dev/null || true)"
IMAGE_REF="${IMAGE_REF:-__IMAGE_REF__}"
NTFY_URL="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"
STAGE="$1"
ntfy() { curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true; }
mkdir -p "$FS"/{models,image,out} "$ROOT/logs"
echo "running:prep-$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
trap 'rc=$?; if [ $rc -eq 0 ]; then ntfy "prep stage $STAGE completed" "GLM53 prep OK: $STAGE" "white_check_mark";
      else ntfy "prep stage $STAGE FAILED rc=$rc" "GLM53 prep FAILED: $STAGE" "rotating_light" "high"; fi' EXIT

drun() {  # docker run against the FILESYSTEM mount
  local ep="$1"; shift
  sudo docker run --rm -i --gpus all --ipc=host --shm-size=16g \
    -v "$FS:/glm53" -v "$ROOT/bundle:/glm53/bundle:ro" \
    -e NVIDIA_TF32_OVERRIDE=0 -e HF_HUB_OFFLINE=1 \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    --entrypoint "$ep" "$IMAGE_REF" "$@"
}
HF="$ROOT/hfenv/bin/hf"
export HF_HOME=/home/jl_fs/glm53/hf-cache HF_XET_HIGH_PERFORMANCE=1

case "$STAGE" in
download_bf16)
  "$HF" download zai-org/GLM-5.3-Flash-BF16 --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
      --local-dir "$FS/models/bf16" --max-workers 8
  echo b1967181a3917ae70a437f4884748f6b8e3a1f4d > "$FS/models/bf16/revision.txt"
  du -sh "$FS/models/bf16"
  ;;
download_fp8)
  "$HF" download zai-org/GLM-5.3-Flash --revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a \
      --local-dir "$FS/models/fp8" --max-workers 8
  echo 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a > "$FS/models/fp8/revision.txt"
  du -sh "$FS/models/fp8"
  ;;
download_smoke)
  "$HF" download Qwen/Qwen3-0.6B --local-dir "$FS/models/smoke" --max-workers 4
  python3 -c "
import json, urllib.request, pathlib
sha = json.load(urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-0.6B'))['sha']
pathlib.Path('$FS/models/smoke/revision.txt').write_text(sha + '\n')
print('smoke model ready, pinned', sha)"
  ;;
image_save)
  sudo docker save -o /tmp/vllm-glm53.tar "$IMAGE_REF"
  sudo mv /tmp/vllm-glm53.tar "$FS/image/vllm-glm53.tar"
  sudo chown ubuntu:ubuntu "$FS/image/vllm-glm53.tar"
  cp "$ROOT/out/image-pin.txt" "$FS/image/"
  ls -la "$FS/image/"
  ;;
heads)
  drun python3 /glm53/bundle/remote/extract_head.py --model /glm53/models/bf16 --out /glm53/out
  drun python3 /glm53/bundle/remote/extract_head.py --model /glm53/models/fp8 --out /glm53/out/fp8-head-check
  python3 - <<'HEADCHK'
import json, pathlib
fs = pathlib.Path("/home/jl_fs/glm53")
a = json.load(open(fs / "out/head-extraction.json"))["tensors"]
b = json.load(open(fs / "out/fp8-head-check/head-extraction.json"))["tensors"]
receipt = {"schema": "glm53flash-head-equality/1",
           "head_equal": a["head"]["sha256"] == b["head"]["sha256"],
           "final_norm_equal": a["final_norm"]["sha256"] == b["final_norm"]["sha256"],
           "bf16": {k: v["sha256"] for k, v in a.items()},
           "fp8": {k: v["sha256"] for k, v in b.items()}}
(fs / "out/head-equality-fp8.json").write_text(json.dumps(receipt, indent=2))
print("HEAD_EQUALITY", json.dumps(receipt))
assert receipt["head_equal"] and receipt["final_norm_equal"], "heads differ across repos"
HEADCHK
  ;;
smoke)
  # End-to-end fidelity pipeline on a tiny model INSIDE the glm53-flash image:
  # suite -> capture x2 -> byte-compare -> qualify -> replay(cap1 vs cap2).
  rm -rf "$FS/smoke"; mkdir -p "$FS/smoke"
  drun bash -c '
    set -euo pipefail
    mkdir -p /work/exllamav3/exllamav3/conversion
    ln -sfn /glm53/bundle/cal_data /work/exllamav3/exllamav3/conversion/standard_cal_data
    F=/glm53/bundle/tools/fidelity.py
    python3 $F suite   --model /glm53/models/smoke --out /glm53/smoke/suite --contexts 6 --context-length 512
    python3 $F capture --model /glm53/models/smoke --suite /glm53/smoke/suite --out /glm53/smoke/cap1 \
                       --tensor-parallel 1 --gpu-memory-utilization 0.7 --no-hash-shards --chunk-accumulate \
                       --engine-kwargs "{\"enable_flashinfer_autotune\": false}"
    python3 $F capture --model /glm53/models/smoke --suite /glm53/smoke/suite --out /glm53/smoke/cap2 \
                       --tensor-parallel 1 --gpu-memory-utilization 0.7 --no-hash-shards --chunk-accumulate \
                       --engine-kwargs "{\"enable_flashinfer_autotune\": false}"
    python3 /glm53/bundle/remote/extract_head.py --model /glm53/models/smoke --out /glm53/smoke/head
    python3 $F qualify --model /glm53/models/smoke --suite /glm53/smoke/suite --hidden /glm53/smoke/cap1 \
                       --head /glm53/smoke/head/head.safetensors --out /glm53/smoke/qualify.json \
                       --contexts 2 --tensor-parallel 1 --gpu-memory-utilization 0.7 \
                       --engine-kwargs "{\"enable_flashinfer_autotune\": false}"
    python3 $F replay  --reference /glm53/smoke/cap1 --candidate /glm53/smoke/cap2 \
                       --head /glm53/smoke/head/head.safetensors --suite /glm53/smoke/suite \
                       --out /glm53/smoke/replay.json --no-hash-shards --device cuda
    python3 - <<PY
import json, hashlib
c1 = json.load(open("/glm53/smoke/cap1/capture-manifest.json"))
c2 = json.load(open("/glm53/smoke/cap2/capture-manifest.json"))
same = all(a["sha256"] == b["sha256"] for a, b in zip(c1["captures"], c2["captures"]))
q = json.load(open("/glm53/smoke/qualify.json"))
r = json.load(open("/glm53/smoke/replay.json"))
summary = {"capture_byte_identical_across_engine_loads": same,
           "qualify_mean_kld_live_vs_replayed": q["mean_kld_live_vs_replayed"],
           "qualify_top1": q["top1_agreement"],
           "replay_cap1_vs_cap2_mean_kld": r["token_mean_kld"]}
print("SMOKE_SUMMARY", json.dumps(summary))
assert same, "captures not byte-identical across engine loads"
assert q["mean_kld_live_vs_replayed"] <= 1e-4, q["mean_kld_live_vs_replayed"]
assert r["token_mean_kld"] <= 1e-9, r["token_mean_kld"]
print("SMOKE_GREEN")
PY
  '
  ;;
act_smoke)
  drun python3 /glm53/bundle/tools/activation_capture.py \
    --model /glm53/models/smoke --suite /glm53/smoke/suite --out /glm53/smoke/acts \
    --tensor-parallel 1 --gpu-memory-utilization 0.7 --contexts 2 \
    --engine-kwargs "{\"enable_flashinfer_autotune\": false}"
  python3 -c "
import json
m = json.load(open('/home/jl_fs/glm53/smoke/acts/activation-manifest.json'))
assert m['complete'] and m['contexts'] == 2, m
print('ACT_SMOKE_GREEN', len(m['hooked_modules']), 'modules hooked')"
  ;;
fsbench)
  sync; echo "sequential read of one BF16 shard from the filesystem:"
  dd if="$FS/models/bf16/model-00001-of-00120.safetensors" of=/dev/null bs=64M count=40 2>&1 | tail -1
  ;;
*) echo "unknown prep stage: $STAGE" >&2; exit 2 ;;
esac
