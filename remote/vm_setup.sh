#!/bin/bash
# Stage 0 (H200 VM): docker + NVIDIA container toolkit + pinned vLLM image + hf CLI.
# IMAGE_REF is substituted by make_bundle.sh.
set -euo pipefail
IMAGE_REF="${IMAGE_REF:-__IMAGE_REF__}"
ROOT=/home/ubuntu/glm53
mkdir -p "$ROOT"/{models,captures,out,logs}

echo "=== driver / GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

if ! command -v docker >/dev/null; then
  echo "=== installing docker ==="
  sudo apt-get update -qq && sudo apt-get install -y -qq docker.io
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  if ! command -v nvidia-ctk >/dev/null; then
    echo "=== installing nvidia-container-toolkit ==="
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  fi
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
sudo usermod -aG docker ubuntu || true

DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
  echo "driver $DRIVER_MAJOR < 580: falling back to cu129 image variant"
  IMAGE_REF="vllm/vllm-openai:glm53-flash-x86_64-cu129"
fi
echo "=== pulling pinned image $IMAGE_REF ==="
sudo docker pull "$IMAGE_REF"
sudo docker inspect --format '{{.Id}} {{.RepoDigests}}' "$IMAGE_REF" | tee "$ROOT/out/image-pin.txt"

echo "=== hf download venv (host) ==="
sudo apt-get update -qq >/dev/null 2>&1 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3.10-venv python3.12-venv python3-pip 2>&1 | tail -1 || true
if [ ! -x "$ROOT/hfenv/bin/hf" ]; then
  python3 -m venv "$ROOT/hfenv"
  "$ROOT/hfenv/bin/pip" -q install --upgrade pip
  "$ROOT/hfenv/bin/pip" -q install "huggingface_hub[cli]" hf_xet
fi
"$ROOT/hfenv/bin/hf" version || "$ROOT/hfenv/bin/hf" --help | head -2

echo "=== smoke: torch sees 8 GPUs inside the image ==="
sudo docker run --rm -i --gpus all --entrypoint python3 "$IMAGE_REF" - <<'EOF'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("vllm", vllm.__version__)
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("glm5_next in registry:", any("Glm5Next" in a for a in archs))
EOF
df -h /home
echo VM_SETUP_DONE
