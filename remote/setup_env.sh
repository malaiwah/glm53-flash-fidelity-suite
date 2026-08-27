#!/bin/bash
# Stage 0: build the inference venv on the JarvisLabs instance.
# VLLM_PIN is substituted before upload (see RUNBOOK.md).
set -euo pipefail

VLLM_PIN="${VLLM_PIN:-__VLLM_PIN__}"
ROOT=/home/glm53
mkdir -p "$ROOT"/{models,captures,out,logs}

python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install --upgrade pip
# vLLM brings its own torch pin; do not inherit the template's torch.
"$ROOT/venv/bin/pip" install "$VLLM_PIN"
"$ROOT/venv/bin/pip" install "huggingface_hub[hf_transfer]" safetensors "transformers>=4.55"

"$ROOT/venv/bin/python" - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} sm{p.major}{p.minor} {p.total_memory/2**30:.0f}GiB")
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /home
echo SETUP_ENV_DONE
