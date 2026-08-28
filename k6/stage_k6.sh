#!/bin/bash
# Stage driver for the GLM-5.3-Flash K6 / K6K8 EXL3-MCG quantization campaign.
# Usage: stage_k6.sh <stage-name>
#
# Runs on JarvisLabs GPU containers (IN2) with filesystem 3394 mounted at
# /home/jl_fs.  All state lives on the fs, so spot preemption + re-create
# resumes cleanly: every stage is guarded by a done-file, and the conversion
# stages resume at per-expert granularity through the pipeline's own receipt
# files (encode_work_unit skips experts whose sealed receipt already exists).
#
# Atomic self-update convention (control session): upload as stage_k6.sh.new,
# then `mv stage_k6.sh.new stage_k6.sh`.  This script refuses to run while a
# half-synced .new file exists.
#
# Stages:
#   setup fixture_rehearsal shared_vector_ab convert_k6 convert_k8
#   materialize_k8 convert_k6k8 qualify_k6 qualify_k8 qualify_k6k8
#   upload_weights publish_receipts
set -euo pipefail

ROOT=/home/jl_fs/glm53-k6
BF16=/home/jl_fs/models/bf16              # pinned zai-org/GLM-5.3-Flash-BF16 (weights == a6c167b6)
NTFY_URL="https://ntfy.sh/omp-396220bc418fb23ea7a57901a54c7b33"
STAGE="${1:?usage: stage_k6.sh <stage>}"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
[ -e "$SELF.new" ] && { echo "half-synced $SELF.new exists - finish the mv first" >&2; exit 3; }

# Source pins (all three code trees come from GitHub, cloned+pinned in setup):
PIPE_REPO=https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw
PIPE_PIN=ce1bf9706b6aa18435e2baccab63bdd72299257c
SHAPLEY_REPO=https://github.com/brandonmmusic-max/shapleymcg
SHAPLEY_PIN=c9b752b9ea2c1d5bd2b0d63317c4cb8e04a9027c   # HEAD; 9d83e7d0 closure rev is in its history
SQGEXP_REPO=https://github.com/brandonmmusic-max/glm52-sqg-mcg-experiments
SQGEXP_PIN=bf37b06691c68525b74bddfa0a1a8216e695c95f    # sparse: bmmlaw_r7_encoder/ only
PIPE=$ROOT/pipeline                        # glm-5.3-flash-exl3-4bpw @ pin + patches-v2 applied (src/ layout)
SHAPLEY=$ROOT/shapleymcg                   # public ShapleyMCG @ pin (scripts/run_qwen_fast_encode.py closure)
SQGEXP=$ROOT/sqg-mcg                       # glm52-sqg-mcg-experiments sparse @ pin (bmmlaw_r7_encoder pkg)
EXL3=$ROOT/exllamav3                       # exllamav3 @ c5d9c657 (v0.0.43), built in-place
TOOLS=$ROOT/tools                          # our driver tools (k6_driver.py etc.)
VENV=$ROOT/venv
CAL=$ROOT/calibration                      # main-ep4-full/ mtp45-ep4-full/ panel-v1/
TEACH=$ROOT/teacher-final                  # 25 sealed final-window fp32 logits
OUT_K6=$ROOT/out-k6                        # payload-store/ experts/ layers/ mtp-*/ hessians/
OUT_K8=$ROOT/out-k8                        # K8-uniform parts bin (DECISIONS.md 7)
OUT_K6K8=$ROOT/out-k6k8
CKPT_K6=$ROOT/ckpt-k6                      # materialized checkpoints
CKPT_K8=$ROOT/ckpt-k8
CKPT_K6K8=$ROOT/ckpt-k6k8
FP8_DIR=/home/jl_fs/glm53/models/fp8       # 328 GB reference FP8 (re-downloadable; K8 ledger eviction target)
RCPT=$ROOT/receipts
DONE=$RCPT/done
mkdir -p "$ROOT/logs" "$RCPT" "$DONE"

ntfy() {  # ntfy <body> <title> <tags> [priority]
  curl -s -m 10 -H "Title: $2" -H "Tags: $3" ${4:+-H "Priority: $4"} \
       -d "$1" "$NTFY_URL" >/dev/null 2>&1 || true
}

echo "running:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
trap 'rc=$?; if [ $rc -eq 0 ]; then
        echo "done:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
        ntfy "stage $STAGE completed" "GLM53-K6 OK: $STAGE" "white_check_mark"
      else
        echo "failed:$STAGE $(date -u +%FT%TZ)" > "$ROOT/logs/stage.state"
        ntfy "stage $STAGE FAILED rc=$rc - control session will diagnose via jl run logs" \
             "GLM53-K6 FAILED: $STAGE" "rotating_light" "high"
      fi' EXIT
ntfy "stage $STAGE started" "GLM53-K6 start: $STAGE" "arrow_forward"

# Resume guard: a stage that already sealed its done-file is a no-op.
# EXCEPTION: setup always re-runs.  P0/P1/P3/P4 are DIFFERENT instances sharing
# this fs; setup is idempotent by design and must re-verify the environment
# (venv health, exllamav3 pin, ShapleyMCG pins, nvidia-smi receipt) on every
# fresh container instead of being skipped by a done-file another instance wrote.
if [ "$STAGE" != setup ] && [ -f "$DONE/$STAGE.done" ]; then
  echo "stage $STAGE already done: $(cat "$DONE/$STAGE.done")"
  exit 0
fi

# HF token: read from file, exported, never echoed. set +x is never enabled.
hf_env() {
  test -f "$ROOT/.hf_token" || { echo "no $ROOT/.hf_token on fs - upload stages need it" >&2; return 1; }
  export HF_TOKEN="$(cat "$ROOT/.hf_token")"
  export HF_HUB_ENABLE_HF_TRANSFER=1
}

PY=$VENV/bin/python
# venv bin on PATH: torch JIT extension loads shell out to `ninja` via PATH
# (pip-installed into the venv), and subprocesses inherit the right python.
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PIPE/src:$SHAPLEY:$SQGEXP"
# BUG FIX (found during K8 enablement review): seal-main / release-dead-claims
# / materialize are invoked in convert_k6 WITHOUT --pipeline-root, and the
# driver hard-requires it (or QP_PIPELINE_ROOT) for exactly those subcommands -
# convert_k6 would have failed right after its encode completed.  The env var
# is the minimal fix that covers every call site without touching the k6
# command lines mid-campaign.
export QP_PIPELINE_ROOT="$PIPE"
# Fat-build attestation (disclosed deviation 2): the driver requires this env
# to match the extension evidence at EVERY invocation, not just at build time.
export TORCH_CUDA_ARCH_LIST="9.0;10.0"
export NVIDIA_TF32_OVERRIDE=0
# Single-node collectives: pin bootstrap sockets to a real local iface —
# multi-NIC VMs hung the first collective when gloo/NCCL picked the wrong one.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo,enp3s0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

mark_done() { echo "$(date -u +%FT%TZ)" > "$DONE/$STAGE.done"; }

# v2-0008 top-up (adversarial K8 review finding): normalization ALLOWED_BITS is
# (3,4,5) at the pin and streaming_v31.FitSampleSpec.from_input imports it, so
# build_layer_preparation REFUSES bits=6 AND bits=8 - K6 GSS would crash at
# "preparing layer 3" without this.  0008 touches no reader/closure-hashed file
# and may land at ANY time (unlike 0007).  The P1 fleet fs already carries the
# widening as a hot-edit (verified byte-identical to 0008's output); this guard
# codifies it and back-fills the receipt line.
ensure_0008() {
  local target="$PIPE/src/quant_pipeline/normalization/absolute_v31.py"
  local p="$ROOT/patches-v2/0008-v31-allowed-bits-k6-k8.patch"
  [ -f "$target" ] || return 0
  if grep -qF "ALLOWED_BITS = frozenset((3, 4, 5))" "$target"; then
    test -f "$p" || { echo "patches-v2/0008-v31-allowed-bits-k6-k8.patch missing on fs - upload it (GSS preparation refuses bits 6/8 without it)" >&2; exit 1; }
    ( cd "$PIPE" && patch -p1 -s < "$p" )
  fi
  if [ -f "$p" ] && ! grep -q "0008-v31-allowed-bits" "$RCPT/patches-v2-applied.txt" 2>/dev/null; then
    ( cd "$ROOT/patches-v2" && sha256sum 0008-*.patch ) | tee -a "$RCPT/patches-v2-applied.txt"
  fi
}

# Fail fast if the fs cannot absorb the stage's writes (RUNBOOK abort criterion:
# never start a fleet encode that will run the fs out mid-layer).
require_free_gb() {  # require_free_gb <gb> <why>
  local free_gb
  free_gb=$(df -BG --output=avail /home/jl_fs 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$free_gb" ] || free_gb=$(df -g /home/jl_fs | awk 'NR==2{print $4}')
  if [ "$free_gb" -lt "$1" ]; then
    echo "only ${free_gb} GB free on fs, need $1 GB: $2" >&2
    exit 4
  fi
  echo "fs free ${free_gb} GB (>= $1 GB required: $2)"
}

# Run one single-GPU encode worker (used by convert stages).  The driver
# claims dynamic work units and skips experts whose sealed receipt exists,
# so re-running after a preemption loses at most one in-flight expert batch.
run_workers() {  # run_workers <profile: k6|k8|k6k8> <output_root>
  local profile="$1" out="$2" pids=() i rcs=0
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i "$PY" "$TOOLS/k6_driver.py" encode-worker \
      --profile "$profile" --worker "h200-$i" \
      $( [ "$profile" = k8 ] && echo --overlap-seal ) \
      --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
      --bf16 "$BF16" --calibration "$CAL" --output-root "$out" \
      --prune-hessians-after-layer-seal \
      > "$ROOT/logs/${profile}-worker-$i.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || rcs=$((rcs+1)); done
  return $rcs
}

case "$STAGE" in

setup)
  # Idempotent: containers lose system packages on pause; everything below
  # re-runs safely and the venv persists on the fs.
  test -d "$BF16" || { echo "BF16 checkpoint missing at $BF16" >&2; exit 1; }
  # Containers run as root; VMs run as ubuntu — prefix privileged ops with
  # sudo when not root (VM bootstrap failed silently without this).
  ASROOT=""
  [ "$(id -u)" = 0 ] || ASROOT="sudo"
  # Container self-bootstrap (idempotent; containers lose apt state on pause):
  # deadsnakes python3.12 and CUDA toolkit 13.0 (nvcc must emit sm_100 for the
  # disclosed fat build).  Root inside jl containers, no sudo.
  if ! command -v python3.12 >/dev/null; then
    $ASROOT apt-get update -qq >/dev/null 2>&1 || true
    $ASROOT apt-get install -y -qq software-properties-common >/dev/null 2>&1 || true
    $ASROOT add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 || true
    $ASROOT apt-get update -qq >/dev/null 2>&1 || true
    for p in python3.12 python3.12-venv python3.12-dev; do
      $ASROOT apt-get install -y -qq "$p" >/dev/null 2>&1 || echo "apt $p failed (tolerated; guard below decides)"
    done
  fi
  if ! { command -v nvcc >/dev/null && nvcc --list-gpu-arch 2>/dev/null | grep -q compute_100; }; then
    ( cd /tmp \
      && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      && $ASROOT dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null 2>&1 \
      && $ASROOT apt-get update -qq >/dev/null 2>&1 \
      && $ASROOT apt-get install -y -qq cuda-toolkit-13-0 >/dev/null 2>&1 ) || echo "cuda-toolkit-13-0 install failed (tolerated; guard below decides)"
    $ASROOT ln -sfn /usr/local/cuda-13.0 /usr/local/cuda 2>/dev/null || true
    export PATH="/usr/local/cuda-13.0/bin:$PATH"
  fi
  # PROVEN ENV RECIPE (DECISIONS.md item 5) was validated on python 3.12.
  # Fail fast on anything else instead of building an unvalidated venv.
  PYBIN="$(command -v python3.12 || true)"
  [ -n "$PYBIN" ] || PYBIN="$(command -v python3)"
  "$PYBIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "container python is not 3.12 ($("$PYBIN" -V 2>&1)) - the proven env recipe (DECISIONS.md 5) is py3.12-only; pick a template with python3.12 or install it (deadsnakes) before setup" >&2; exit 1; }
  "$PYBIN" -V | tee "$RCPT/python-version.txt"
  "$PYBIN" -m venv "$VENV" 2>/dev/null || true
  "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
    || { echo "existing venv at $VENV is not py3.12 - delete it and re-run setup" >&2; exit 1; }
  "$VENV/bin/pip" -q install --upgrade pip
  # setuptools/wheel/ninja/packaging: required for the --no-build-isolation
  # exllamav3 build below (venvs on Python >= 3.12 ship without setuptools).
  "$VENV/bin/pip" -q install setuptools wheel ninja packaging
  "$VENV/bin/pip" -q install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
  "$VENV/bin/pip" -q install "transformers==5.16.1" safetensors numpy huggingface_hub hf_transfer \
    accelerate rich tokenizers pillow \
    "pydantic==2.5.3" "formatron==0.5.0" kbnf
  # flash-attn: exllamav3 @ the pin hard-imports it; no torch2.11 wheel exists,
  # the cu13torch2.10 wheel is ABI-compatible (proven on the L4, DECISIONS 5)
  "$VENV/bin/python" -c "import flash_attn" 2>/dev/null || "$VENV/bin/pip" -q install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  # exllamav3 pinned checkout, tracked tree must stay clean (verify_exllamav3_source),
  # so build in-place; the extension binary is untracked and therefore allowed.
  if [ ! -d "$EXL3/.git" ]; then
    git clone https://github.com/turboderp-org/exllamav3 "$EXL3"
  fi
  git -C "$EXL3" checkout c5d9c657966ffeeaa9353f0cc899f18629da4a13
  git -C "$EXL3" diff --quiet && git -C "$EXL3" diff --cached --quiet
  # DISCLOSED DEVIATION: arch list includes 10.0 (SM100) alongside 9.0 (SM90)
  # so the extension genuinely carries SM100 code objects; the sealed contract
  # requires "10.0" in compute_capabilities evidence.  We RUN on SM90 H200.
  # sm_100 needs CUDA >= 12.8 in the container's toolkit: fail here, not 20
  # minutes into a silent nvcc error inside the extension build.
  command -v nvcc >/dev/null || { echo "no nvcc in container image - pick a template with a CUDA toolkit >= 12.8" >&2; exit 1; }
  nvcc --version | tee "$RCPT/nvcc-version.txt"
  nvcc --list-gpu-arch 2>/dev/null | grep -q "compute_100" \
    || { echo "container nvcc cannot target sm_100 (need CUDA >= 12.8) - SM100 evidence build impossible" >&2; exit 1; }
  # --no-deps per the proven recipe (DECISIONS.md 5): exllamav3's declared deps
  # (formatron/kbnf/...) are import-guarded and their resolver pulls broke the
  # smoke rounds; every dep the campaign actually needs is pinned above.
  # Rebuild guard: a rebuilt .so breaks sealed prep identities (two boxes
  # sharing this tree clobbered a campaign once). Build ONLY if import fails.
  if ! TORCH_CUDA_ARCH_LIST="9.0;10.0" "$PY" -c "import exllamav3; from exllamav3 import ext" 2>/dev/null; then
    ( cd "$EXL3" && TORCH_CUDA_ARCH_LIST="9.0;10.0" "$VENV/bin/pip" -q install --no-build-isolation --no-deps -e . )
  fi
  # Import smoke + extension-binary receipt.  If the editable install did not
  # precompile, this import JIT-builds into ~/.cache/torch_extensions (container
  # -local: re-runs after preemption, which is why setup always re-runs).  The
  # driver's _find_extension searches both locations.
  TORCH_CUDA_ARCH_LIST="9.0;10.0" "$PY" -c "import exllamav3; from exllamav3 import ext; print('exllamav3 ext import OK')"
  { find "$EXL3" -name '*.so' 2>/dev/null || true; find "$HOME/.cache/torch_extensions" -name 'exllamav3_ext*.so' 2>/dev/null || true; } | tee "$RCPT/exllamav3-ext-path.txt"
  test -s "$RCPT/exllamav3-ext-path.txt" || { echo "no exllamav3 extension binary found after build+import" >&2; exit 1; }
  # Pipeline: his GitHub tree at the pinned commit + our patches-v2 series.
  # Idempotent: a fresh clone is patched once; an already-patched tree is
  # detected by the marker symbol and re-verified, never re-patched.
  if [ ! -d "$PIPE/.git" ]; then
    git clone "$PIPE_REPO" "$PIPE"
  fi
  git -C "$PIPE" fetch -q origin "$PIPE_PIN" 2>/dev/null || true
  if ! grep -q "_STORED_BITS" "$PIPE/src/quant_pipeline/checkpoint/packed_payload.py"; then
    git -C "$PIPE" checkout -q "$PIPE_PIN"
    git -C "$PIPE" diff --quiet && git -C "$PIPE" diff --cached --quiet
    test -d "$ROOT/patches-v2" || { echo "patches-v2 missing at $ROOT/patches-v2 - upload the series first" >&2; exit 1; }
    ( cd "$PIPE" && for p in "$ROOT"/patches-v2/000[1-6]-*.patch; do patch -p1 -s < "$p"; done )
    ( cd "$ROOT/patches-v2" && sha256sum 000[1-6]-*.patch SERIES ) | tee "$RCPT/patches-v2-applied.txt"
  else
    # already patched: the base commit must still be the pin underneath
    [ "$(git -C "$PIPE" rev-parse HEAD)" = "$PIPE_PIN" ] || { echo "$PIPE HEAD is not $PIPE_PIN" >&2; exit 1; }
  fi
  # v2-0007 K8-uniform admission top-up (DECISIONS.md 7).  0007 edits the
  # READER file whose byte-hash every sealed K6 choice binds
  # (decoder.reader_abi_sha256), so it may land ONLY when the K6 campaign is
  # complete - or has not sealed a single choice yet.  Mid-K6 instances keep
  # the exact 0001-0006 reader bytes.
  if ! grep -q "K8_RECIPE_ID" "$PIPE/src/quant_pipeline/campaign/glm53_direct_k4.py" \
     && [ -f "$ROOT/patches-v2/0007-k8-uniform-admission.patch" ] \
     && { [ -f "$DONE/convert_k6.done" ] || [ ! -d "$OUT_K6/payload-store" ]; }; then
    ( cd "$PIPE" && patch -p1 -s < "$ROOT/patches-v2/0007-k8-uniform-admission.patch" )
    ( cd "$ROOT/patches-v2" && sha256sum 0007-*.patch ) | tee -a "$RCPT/patches-v2-applied.txt"
  fi
  ensure_0008
  "$PY" -c "import sys; sys.path.insert(0, '$PIPE/src'); import quant_pipeline.campaign.glm53_uniform_k6, quant_pipeline.campaign.glm53_uniform_k4, quant_pipeline.publication.glm53_k6_postmtp, quant_pipeline.evaluation.glm53_packed_k4_reader; from quant_pipeline.normalization.absolute_v31 import ALLOWED_BITS; assert {6, 8} <= set(ALLOWED_BITS); print('patched pipeline import OK')"
  # ShapleyMCG @ pin (encoder closure) + sparse bmmlaw_r7_encoder @ pin.
  if [ ! -d "$SHAPLEY/.git" ]; then
    git clone "$SHAPLEY_REPO" "$SHAPLEY"
  fi
  git -C "$SHAPLEY" checkout -q "$SHAPLEY_PIN"
  if [ ! -d "$SQGEXP/.git" ]; then
    git clone --filter=blob:none --sparse "$SQGEXP_REPO" "$SQGEXP"
    git -C "$SQGEXP" sparse-checkout set bmmlaw_r7_encoder
  fi
  git -C "$SQGEXP" checkout -q "$SQGEXP_PIN"
  # Closure pin checks + CLOSURE GATE (existential inputs for the encoder).
  # Encode may proceed ONLY IF one of:
  #   (a) UPSTREAM: Brandon's r10_codec.py AND encode_tr3_v31.py landed in the
  #       pinned trees (issue #1 answered), or
  #   (b) RECONSTRUCTION, OPERATOR-ACCEPTED: $ROOT/RECONSTRUCTION-ACCEPTED.json
  #       exists ({"accept_reconstructed_r10_codec": true, ...}, authored by the
  #       operator, never by automation) AND $ROOT/fallback/
  #       r10_codec_reconstructed.py is on the fs - setup then stages the
  #       disclosed fallback package into $SHAPLEY/r7_encoder/ (untracked files
  #       on top of the pinned checkout; every receipt hash-discloses them).
  # Anything else hard-stops HERE, not mid-rental.  receipts/closure_status.json
  # records which world we are in; the driver re-checks per-invocation.
  "$PY" - <<PYEOF
import hashlib, json, pathlib, sys, time
p = pathlib.Path("$SHAPLEY/scripts/run_qwen_fast_encode.py")
assert p.is_file(), "ShapleyMCG closure missing run_qwen_fast_encode.py"
digest = hashlib.sha256(p.read_bytes()).hexdigest()
expected = "ceea8c64d63ffb60cdf95adee3ba7b488c54303d3a85502798b2c3fd0fcbb492"
assert digest == expected, f"run_qwen_fast_encode.py sha mismatch: {digest}"
pkg = pathlib.Path("$SQGEXP/bmmlaw_r7_encoder")
assert any(pkg.rglob("*.py")), "bmmlaw_r7_encoder numeric core absent (sparse clone failed?)"
marker = "k6-program.r10-fallback-reconstruction.v1"
missing = [name for name in ("r10_codec.py", "encode_tr3_v31.py")
           if not list(pkg.rglob(name)) and not list(pathlib.Path("$SHAPLEY").rglob(name))]
codec = pathlib.Path("$SHAPLEY/r7_encoder/r10_codec.py")
staged = {}
if not missing and marker not in (codec.read_text(encoding="utf-8", errors="replace") if codec.is_file() else ""):
    source = "upstream"
else:
    acceptance_path = pathlib.Path("$ROOT/RECONSTRUCTION-ACCEPTED.json")
    fallback_path = pathlib.Path("$ROOT/fallback/r10_codec_reconstructed.py")
    accepted = False
    if acceptance_path.is_file():
        try:
            accepted = json.loads(acceptance_path.read_text())["accept_reconstructed_r10_codec"] is True
        except Exception:
            accepted = False
    if not (accepted and fallback_path.is_file()):
        sys.exit("CLOSURE GATE: encoder closure files missing upstream (issue #1 pending): "
                 f"{missing or ['(reconstruction present but unaccepted)']}\n"
                 "Either Brandon's files must land in the pinned trees, or the operator must "
                 "explicitly accept the disclosed reconstruction by authoring "
                 "$ROOT/RECONSTRUCTION-ACCEPTED.json with accept_reconstructed_r10_codec: true "
                 "(and uploading fallback/r10_codec_reconstructed.py to $ROOT/fallback/).  "
                 "Refusing to continue.")
    sys.path.insert(0, str(fallback_path.parent))
    import r10_codec_reconstructed as fallback
    staged = fallback.stage_r7_encoder("$SHAPLEY", ancestors_dir=pkg)
    source = "reconstruction"
status = {
    "schema": "malaiwah.glm53-k6-shapleymcg-closure-status.v1",
    "shapley_root": "$SHAPLEY",
    "closure_source": source,
    "r10_codec_present": True,
    "reconstruction_accepted": source == "reconstruction" or None,
    "staged_files_sha256": staged or None,
    "upstream_request": "github issue #1 on the ShapleyMCG repository",
    "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
pathlib.Path("$RCPT/closure_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(f"closure gate: source={source} OK" + (f" ({len(staged)} staged files)" if staged else ""))
PYEOF
  # Fail fast on the rest of the control-session uploads the stages depend on.
  for need in "$TOOLS/k6_driver.py" "$TOOLS/k6_student_capture.py" \
              "$TOOLS/k6_kld_report.py" "$TOOLS/k6_publish.py" \
              "$ROOT/recipes/k6.json" "$ROOT/recipes/k8.json" "$ROOT/recipes/k6k8.json"; do
    test -f "$need" || { echo "missing on fs: $need - upload the k6-program tree first" >&2; exit 1; }
  done
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv | tee "$RCPT/nvidia-smi.txt"
  mark_done
  ;;

fixture_rehearsal)
  # P0 on the tiny architecturally-complete random model: full contract ->
  # preparation -> encode -> seal -> materialize -> reader-decode roundtrip at
  # K6, plus the K8 codec probe and the full-size per-matrix timing benchmark
  # that prices P1.  Writes receipts/rehearsal.json; convert stages refuse to
  # run without it.
  test -d "$ROOT/fixture/GLM-5.3-Flash-0.1B-A0.1B" \
    || { echo "fixture missing: download inference-optimization/GLM-5.3-Flash-0.1B-A0.1B to $ROOT/fixture/ first" >&2; exit 1; }
  "$PY" "$TOOLS/k6_driver.py" rehearse \
    --fixture "$ROOT/fixture/GLM-5.3-Flash-0.1B-A0.1B" \
    --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
    --bench-full-size-matrices 24 --probe-k8 \
    --output "$RCPT/rehearsal.json"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/receipts/rehearsal.json"))
assert r["k6_roundtrip_exact"], "K6 encode->decode roundtrip not exact on fixture"
spm = r["seconds_per_full_size_matrix_k6"]
est_h = 37152 * spm / 4 / 3600
print(f"K6 per-matrix {spm:.2f}s -> est main+MTP encode wall {est_h:.1f} h on 4 GPUs")
assert est_h < 24, f"projected encode {est_h:.1f}h busts the budget - STOP and re-plan"
print("K8 probe:", r.get("k8_probe", "not-run"))
PYEOF
  mark_done
  ;;

shared_vector_ab)
  # Operator directive (DECISIONS.md #2): decide down_suh topology
  # (shared-per-layer vs upstream expert-private) BEFORE any fleet encode.
  # Quantize 2-3 representative layers both ways, replay captured block
  # inputs, compare output divergence.  Adopt shared only if delta ~nil.
  test -f "$DONE/setup.done" && test -f "$DONE/fixture_rehearsal.done"
  test -d "$CAL/main-ep4-full" || { echo "calibration/main-ep4-full missing at $CAL - download it before shared_vector_ab" >&2; exit 1; }
  # --output-root "$OUT_K6": the A/B mints/reuses the CAMPAIGN transform seed
  # (out-k6/transform-seed.json) so its verdict is measured under the exact
  # sign vectors the fleet encode will use.
  mkdir -p "$OUT_K6"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$TOOLS/k6_driver.py" shared-vector-ab \
    --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
    --bf16 "$BF16" --calibration "$CAL" --layers 3,20,44 \
    --output-root "$OUT_K6" \
    --output "$RCPT/shared-vector-ab.json"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/receipts/shared-vector-ab.json"))
print("down_suh shared-vs-private divergence per layer:", r["per_layer_delta"])
print("DECISION:", r["adopted_topology"], "(threshold ~nil:", r["threshold"], ")")
assert r["adopted_topology"] in ("expert_private_upstream", "layer_shared_deviation")
PYEOF
  mark_done
  ;;

convert_k6)
  test -f "$DONE/setup.done" && test -f "$DONE/fixture_rehearsal.done"
  test -f "$DONE/shared_vector_ab.done"
  ensure_0008   # GSS preparation refuses bits=6 without the ALLOWED_BITS widening
  # Inputs the control session downloads out-of-band (no automated fetch here):
  test -d "$CAL/main-ep4-full" || { echo "calibration/main-ep4-full missing at $CAL - download from brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits first" >&2; exit 1; }
  test -d "$CAL/mtp45-ep4-full" || { echo "calibration/mtp45-ep4-full missing at $CAL" >&2; exit 1; }
  test -d "$CAL/panel-v1" || { echo "calibration/panel-v1 missing at $CAL" >&2; exit 1; }
  # Disk ledger: payload-store ~254 GB + <=60 GB transient hessians + receipts.
  require_free_gb 350 "K6 payload store + per-layer hessian transient"
  mkdir -p "$OUT_K6"
  # 1) Sealed inventory + launch plan + contract (idempotent; sealed docs are
  #    content-addressed and reused if already present).
  #    UPSTREAM INVENTORY (G0 item 2): brandonmusic's captures bind HIS sealed
  #    inventory_sha256 (f56e9d62...) and Glm53BF16Source requires the BF16
  #    tree at the checkpoint path his inventory declares
  #    (/workspace/models/zai-org/GLM-5.3-Flash-BF16 per his capture plan).
  #    Download his sealed inventory doc to $CAL/upstream-inventory.json and
  #    symlink the declared path to $BF16; a FRESH inventory can NEVER match
  #    his sha (it seals our local path) and the contract fails closed on the
  #    capture-manifest binding - so require the doc here, not mid-hash.
  test -f "$CAL/upstream-inventory.json" \
    || { echo "upstream sealed inventory missing at $CAL/upstream-inventory.json - his captures cannot bind a fresh inventory (G0 item 2); fetch it from the teacher-logits dataset first" >&2; exit 1; }
  UPSTREAM_BF16_PATH=$("$PY" -c "import json;print(json.load(open('$CAL/upstream-inventory.json'))['checkpoint'])")
  if [ "$UPSTREAM_BF16_PATH" != "$BF16" ] && [ ! -e "$UPSTREAM_BF16_PATH" ]; then
    mkdir -p "$(dirname "$UPSTREAM_BF16_PATH")"
    ln -sfn "$BF16" "$UPSTREAM_BF16_PATH"
  fi
  # K4 GATE (upstream glm53_uniform_k6 is K4-KL-gated): the k6_authorized K4
  # state receipt is authored by the CONTROL SESSION as the disclosed bridge
  # from brandonmusic's published K4 receipts; the driver refuses to fabricate.
  test -f "$OUT_K6/k4-authorized-state.json" \
    || { echo "K4 gate state receipt missing at $OUT_K6/k4-authorized-state.json - author the disclosed bridge doc (RUNBOOK P1 step 0) before convert_k6" >&2; exit 1; }
  "$PY" "$TOOLS/k6_driver.py" contract --profile k6 \
    --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
    --bf16 "$UPSTREAM_BF16_PATH" --calibration "$CAL" --output-root "$OUT_K6" \
    --recipe "$ROOT/recipes/k6.json" \
    --inventory "$CAL/upstream-inventory.json"
  # 2) Main layers 3..44: four single-GPU workers, dynamic whole-layer units,
  #    per-expert receipt resume.  Retry loop tolerates transient worker exits;
  #    the completion check is the sealed main receipt, not exit codes.
  for attempt in 1 2 3; do
    run_workers k6 "$OUT_K6" || true
    [ -f "$OUT_K6/main-receipt.json" ] && break
    "$PY" "$TOOLS/k6_driver.py" seal-main --profile k6 --output-root "$OUT_K6" || true
    [ -f "$OUT_K6/main-receipt.json" ] && break
    echo "main not sealed after attempt $attempt; requeueing dead claims"
    "$PY" "$TOOLS/k6_driver.py" release-dead-claims --profile k6 --output-root "$OUT_K6" || true
  done
  test -f "$OUT_K6/main-receipt.json"
  # 3) MTP layer 45 (separate contract, only after main is complete+sealed).
  "$PY" "$TOOLS/k6_driver.py" mtp --profile k6 \
    --pipeline-root "$PIPE" --bf16 "$BF16" --calibration "$CAL" --output-root "$OUT_K6"
  test -f "$OUT_K6/mtp-adapter-receipt.json"
  # 4) Free the calibration captures before materializing (disk ledger:
  #    payload-store + materialized checkpoint cannot coexist with 475 GB of
  #    captures on the 2 TB fs).  They are re-downloadable from the Hub.
  du -sh "$CAL" && rm -rf "$CAL/main-ep4-full"
  # 5) Materialize (per-shard receipts -> resumable).
  "$PY" "$TOOLS/k6_driver.py" materialize --profile k6 \
    --output-root "$OUT_K6" --bf16 "$BF16" --checkpoint "$CKPT_K6"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/ckpt-k6/materialization-receipt.json"))
assert r["schema"] == "quant-pipeline.glm53-k6-materialization-receipt.v1", r["schema"]
assert r["bits"] == 6 and r["complete"] and r["nonrouted_native_exact"] and r["main_and_mtp_complete"]
assert r["qualified_tp_sizes"] == [] and r["serving_reader_qualified"] is False
# Receipt-exact K6 size: native 19,339,524,984 + 37,152 x 6,303,748 payload bytes.
assert r["output_logical_bytes"] == 253_536_370_680, r["output_logical_bytes"]
print("K6 materialization receipt GREEN:", r["output_logical_bytes"], "bytes")
PYEOF
  df -h /home/jl_fs
  mark_done
  ;;

convert_k8)
  # K8 TREE OF RECORD: the K8 contract/preparations were sealed from the
  # isolated pipeline-k8 tree (0007 applied pre-K6-completion, legal because
  # isolated).  The prepared-backend closure seal binds those exact bytes -
  # every K8 driver call must use the same tree, not $ROOT/pipeline.
  PIPE="$ROOT/pipeline-k8"
  export QP_PIPELINE_ROOT="$PIPE"
  export PYTHONPATH="$PIPE/src:$SHAPLEY:$SQGEXP"
  # P1b - K8-UNIFORM parts-bin campaign (operator directive, DECISIONS.md 7):
  # runs on the SAME 4x H200 fleet immediately after convert_k6.  Same
  # calibration, same transform seed, same profile parameters - only the rate
  # differs (routed experts at K8, 128-word trellis).  Purpose: (a) shippable
  # ~309 GiB near-BF16 flagship; (b) with the sealed K6 payload store, future
  # multi-precision K6K8 is OFFLINE ASSEMBLY of per-choice payloads, no
  # re-encode.  NO materialize here - that is materialize_k8 (disk ledger).
  test -f "$DONE/convert_k6.done"
  # patch 0007 (K8 admission) top-up: safe ONLY now - it edits the reader file
  # whose byte-hash the sealed K6 choices bind; convert_k6 is complete.
  if ! grep -q "K8_RECIPE_ID" "$PIPE/src/quant_pipeline/campaign/glm53_direct_k4.py"; then
    test -f "$ROOT/patches-v2/0007-k8-uniform-admission.patch" \
      || { echo "patches-v2/0007-k8-uniform-admission.patch missing on fs - upload it first" >&2; exit 1; }
    ( cd "$PIPE" && patch -p1 -s < "$ROOT/patches-v2/0007-k8-uniform-admission.patch" )
    ( cd "$ROOT/patches-v2" && sha256sum 0007-*.patch ) | tee -a "$RCPT/patches-v2-applied.txt"
  fi
  ensure_0008   # GSS preparation refuses bits=8 without the ALLOWED_BITS widening
  "$PY" -c "import sys; sys.path.insert(0, '$PIPE/src'); import quant_pipeline.campaign.glm53_uniform_k8; import quant_pipeline.campaign.glm53_direct_k4 as d; from quant_pipeline.normalization.absolute_v31 import ALLOWED_BITS; assert 8 in d.SUPPORTED_BITS and 8 in ALLOWED_BITS; print('K8 admission import OK')"
  # K8 codec gate: receipts/rehearsal.json recorded k8_probe admitted=false
  # (pre-0007 by design).  Re-probe now on the idle fleet + K8 timing bench
  # (K8 trellis edges > K6: seconds/matrix must be measured, not assumed).
  if [ ! -f "$RCPT/rehearsal-k8.json" ]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" "$TOOLS/k6_driver.py" rehearse \
      --fixture "$ROOT/fixture/GLM-5.3-Flash-0.1B-A0.1B" \
      --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
      --bench-full-size-matrices 6 --bench-bits 8 \
      --output "$RCPT/rehearsal-k8.json"
  fi
  # non-tautological bit-verify on THIS SM90 fleet: reconstructed-codec K8
  # pack vs exllamav3 NATIVE convert (VALIDATION.md V9 covered L4/SM89 only).
  if [ ! -f "$RCPT/k8-native-probe.txt" ]; then
    ( cd "$ROOT/fallback" && PYTHONPATH="$PIPE/src:$SHAPLEY:$SQGEXP" CUDA_VISIBLE_DEVICES=0 \
        "$PY" probe_native_convert.py ) | tee "$RCPT/k8-native-probe.txt"
  fi
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/receipts/rehearsal-k8.json"))
probe = r["k8_probe"]
assert probe.get("admitted") is True, f"K8 still refused post-0007: {probe}"
assert probe.get("encode_decode_exact") is True, f"K8 roundtrip not exact: {probe}"
assert r.get("bench_roundtrip_exact") is True, "K8 full-size bench roundtrip not exact"
spm = r["seconds_per_full_size_matrix_k8"]
est_h = 37152 * spm / 4 / 3600
print(f"K8 per-matrix {spm:.2f}s -> est main+MTP encode wall {est_h:.1f} h on 4 GPUs")
assert est_h < 24, f"projected K8 encode {est_h:.1f}h busts the budget - STOP and re-plan"
PYEOF
  # Disk ledger, EVICTION FIRST (adversarial-review reorder): the P1b ledger
  # can NEVER close with the FP8 reference tree (328 GB, re-downloadable)
  # resident - post-K6 free is ~469 GB and the calibration re-download alone
  # needs 464 GB, a ~5 GB knife-edge.  Evict FP8 BEFORE the operator re-
  # downloads the captures so the download lands into ~797 GB instead.
  # Receipted - never silent.
  free_now=$(df -BG --output=avail /home/jl_fs 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$free_now" ] && [ "$free_now" -lt 800 ] && [ -d "$FP8_DIR" ]; then
    fp8_gb=$(du -s -BG "$FP8_DIR" 2>/dev/null | cut -f1 | tr -dc '0-9')
    cat > "$RCPT/fp8-evicted.json" <<JSON
{
  "schema": "malaiwah.glm53-k8-fp8-eviction.v1",
  "path": "$FP8_DIR",
  "approx_gb": ${fp8_gb:-328},
  "free_gb_before": $free_now,
  "reason": "K8 ledger cannot close with FP8 resident: cal re-download (464 GB) + K8 payload store (312 GB) + hessian transient need the headroom; FP8 is re-downloadable",
  "re_download": "zai-org/GLM-5.3-Flash (FP8, as served) - needed again only for fidelity-suite baseline reruns",
  "evicted_utc": "$(date -u +%FT%TZ)"
}
JSON
    rm -rf "$FP8_DIR"
    ntfy "evicted FP8 tree (${fp8_gb:-328} GB) for K8 cal-redownload + payload headroom" "GLM53-K8: fp8 evicted" "wastebasket"
  fi
  # Inputs (calibration/main-ep4-full was deleted in convert_k6 step 4; the
  # control session re-downloads it AFTER the FP8 eviction above).
  test -d "$CAL/main-ep4-full" || { echo "calibration/main-ep4-full missing - re-download it now (FP8 eviction above freed the room) and re-run convert_k8" >&2; exit 1; }
  test -d "$CAL/mtp45-ep4-full" || { echo "calibration/mtp45-ep4-full missing at $CAL" >&2; exit 1; }
  test -f "$CAL/upstream-inventory.json" \
    || { echo "upstream sealed inventory missing at $CAL/upstream-inventory.json (G0 item 2)" >&2; exit 1; }
  # SAME TRANSFORM SEED (operator requirement - assembly compatibility with
  # the K6 payload store).  The driver also fail-fasts (profile k8 never mints).
  test -f "$OUT_K6/transform-seed.json" \
    || { echo "out-k6/transform-seed.json missing - K8 must reuse the K6 campaign seed" >&2; exit 1; }
  mkdir -p "$OUT_K8"
  if [ -f "$OUT_K8/transform-seed.json" ]; then
    cmp -s "$OUT_K6/transform-seed.json" "$OUT_K8/transform-seed.json" \
      || { echo "out-k8/transform-seed.json DIFFERS from out-k6 - refusing to encode an incompatible parts bin" >&2; exit 1; }
  else
    cp "$OUT_K6/transform-seed.json" "$OUT_K8/transform-seed.json"
  fi
  # K4 gate bridge: the K8 launch plan is K4-KL-gated exactly like K6; reuse
  # the SAME sealed planning + bridge docs the K6 campaign used.
  for doc in k4-launch-plan.json k4-authorized-state.json; do
    test -f "$OUT_K6/$doc" || { echo "out-k6/$doc missing - convert_k6 leaves it; cannot gate the K8 plan" >&2; exit 1; }
    [ -f "$OUT_K8/$doc" ] || cp "$OUT_K6/$doc" "$OUT_K8/$doc"
  done
  # Disk ledger (RUNBOOK P1b): K8 payload store 312 GB + <=60 GB transient
  # hessians land while BF16 + calibration + out-k6 + ckpt-k6 stay resident.
  # (FP8 was evicted above, before the calibration re-download.)
  require_free_gb 100 "K8 encode floor (operator minimum; RUNBOOK P1b targets >=333 GB at encode start)"
  free_now=$(df -BG --output=avail /home/jl_fs 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$free_now" ] && [ "$free_now" -lt 300 ] \
    && ntfy "K8 encode starting with only ${free_now} GB free (<300; ledger plans ~333) - watch the ledger" "GLM53-K8 disk warning" "warning" "high"
  # BF16 path binding: same upstream-inventory symlink dance as convert_k6.
  UPSTREAM_BF16_PATH=$("$PY" -c "import json;print(json.load(open('$CAL/upstream-inventory.json'))['checkpoint'])")
  if [ "$UPSTREAM_BF16_PATH" != "$BF16" ] && [ ! -e "$UPSTREAM_BF16_PATH" ]; then
    mkdir -p "$(dirname "$UPSTREAM_BF16_PATH")"
    ln -sfn "$BF16" "$UPSTREAM_BF16_PATH"
  fi
  # contract (K8-specific GSS preparations run inside) -> encode -> seal -> MTP
  "$PY" "$TOOLS/k6_driver.py" contract --profile k8 \
    --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
    --bf16 "$UPSTREAM_BF16_PATH" --calibration "$CAL" --output-root "$OUT_K8" \
    --recipe "$ROOT/recipes/k8.json" \
    --inventory "$CAL/upstream-inventory.json"
  for attempt in 1 2 3; do
    run_workers k8 "$OUT_K8" || true
    [ -f "$OUT_K8/main-receipt.json" ] && break
    "$PY" "$TOOLS/k6_driver.py" seal-main --profile k8 --output-root "$OUT_K8" --pipeline-root "$PIPE" || true
    [ -f "$OUT_K8/main-receipt.json" ] && break
    echo "K8 main not sealed after attempt $attempt; requeueing dead claims"
    "$PY" "$TOOLS/k6_driver.py" release-dead-claims --profile k8 --output-root "$OUT_K8" --pipeline-root "$PIPE" || true
  done
  test -f "$OUT_K8/main-receipt.json"
  "$PY" "$TOOLS/k6_driver.py" mtp --profile k8 \
    --pipeline-root "$PIPE" --bf16 "$UPSTREAM_BF16_PATH" --calibration "$CAL" --output-root "$OUT_K8"
  test -f "$OUT_K8/mtp-adapter-receipt.json"
  df -h /home/jl_fs
  mark_done
  ;;

materialize_k8)
  # K8 TREE OF RECORD: the K8 contract/preparations were sealed from the
  # isolated pipeline-k8 tree (0007 applied pre-K6-completion, legal because
  # isolated).  The prepared-backend closure seal binds those exact bytes -
  # every K8 driver call must use the same tree, not $ROOT/pipeline.
  PIPE="$ROOT/pipeline-k8"
  export QP_PIPELINE_ROOT="$PIPE"
  export PYTHONPATH="$PIPE/src:$SHAPLEY:$SQGEXP"
  # Ledger-gated K8 checkpoint materialization (operator: materialize AFTER
  # uploading/deleting what the ledger allows).  Deletes the re-downloadable
  # calibration captures first (mirrors convert_k6 step 4), then requires
  # room for ckpt-k8 (331 GB); if still short, upload+delete ckpt-k6 (254 GB)
  # per the RUNBOOK P1b ledger and re-run this stage.
  test -f "$DONE/convert_k8.done"
  du -sh "$CAL" 2>/dev/null || true
  rm -rf "$CAL/main-ep4-full"
  require_free_gb 340 "K8 materialized checkpoint (331 GB) - if short: upload ckpt-k6, delete it, re-run"
  "$PY" "$TOOLS/k6_driver.py" materialize --profile k8 \
    --pipeline-root "$PIPE" --output-root "$OUT_K8" --bf16 "$BF16" --checkpoint "$CKPT_K8"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/ckpt-k8/materialization-receipt.json"))
assert r["schema"] == "malaiwah.glm53-k8-materialization-receipt.v1", r["schema"]
assert r["bits"] == 8 and r["complete"] and r["nonrouted_native_exact"] and r["main_and_mtp_complete"]
assert r["qualified_tp_sizes"] == [] and r["serving_reader_qualified"] is False
# Receipt-exact K8 size: native 19,339,524,984 + 37,152 x 8,400,900 payload bytes.
assert r["output_logical_bytes"] == 331_449_761_784, r["output_logical_bytes"]
print("K8 materialization receipt GREEN:", r["output_logical_bytes"], "bytes")
PYEOF
  df -h /home/jl_fs
  mark_done
  ;;

convert_k6k8)
  # CONDITIONAL STAGE - runs only after K6 has published AND the K8 codec
  # probe passed AND remaining budget clears the gate in the RUNBOOK.
  test -f "$DONE/convert_k6.done"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/receipts/rehearsal.json"))
probe = r.get("k8_probe", {})
assert probe.get("encode_decode_exact") is True, "K8 codec probe not green - K6K8 is NO-GO"
PYEOF
  # calibration/main-ep4-full was deleted before the K6 materialize (disk
  # ledger) - the control session must re-download it before this stage.
  test -d "$CAL/main-ep4-full" || { echo "calibration/main-ep4-full missing - re-download it (deleted in convert_k6) before convert_k6k8" >&2; exit 1; }
  # Disk ledger P2 peak: with calibration re-resident, the encode adds the
  # K6K8 payload store (~280 GB) + <=60 GB hessian transient; materialize then
  # swaps cal (-464) for ckpt-k6k8 (+280).  The 2 TB fs closes ONLY if ckpt-k6
  # (254 GB) was deleted after its publication spot-check - this guard is how
  # that requirement is enforced.
  require_free_gb 400 "K6K8 payload + hessian transient (delete ckpt-k6 after verified publication to free 254 GB)"
  mkdir -p "$OUT_K6K8"
  "$PY" "$TOOLS/k6_driver.py" contract --profile k6k8 \
    --pipeline-root "$PIPE" --shapley-root "$SHAPLEY" --exllama-root "$EXL3" \
    --bf16 "$BF16" --calibration "$CAL" --output-root "$OUT_K6K8" \
    --recipe "$ROOT/recipes/k6k8.json" \
    --reuse-gate-up-from "$OUT_K6/payload-store"
  for attempt in 1 2 3; do
    run_workers k6k8 "$OUT_K6K8" || true
    [ -f "$OUT_K6K8/main-receipt.json" ] && break
    "$PY" "$TOOLS/k6_driver.py" seal-main --profile k6k8 --output-root "$OUT_K6K8" || true
    [ -f "$OUT_K6K8/main-receipt.json" ] && break
    "$PY" "$TOOLS/k6_driver.py" release-dead-claims --profile k6k8 --output-root "$OUT_K6K8" || true
  done
  test -f "$OUT_K6K8/main-receipt.json"
  "$PY" "$TOOLS/k6_driver.py" mtp --profile k6k8 \
    --pipeline-root "$PIPE" --bf16 "$BF16" --calibration "$CAL" --output-root "$OUT_K6K8"
  "$PY" "$TOOLS/k6_driver.py" materialize --profile k6k8 \
    --output-root "$OUT_K6K8" --bf16 "$BF16" --checkpoint "$CKPT_K6K8"
  test -f "$CKPT_K6K8/materialization-receipt.json"
  df -h /home/jl_fs
  mark_done
  ;;

qualify_k6)
  # Runs on the 8x H200 instance (EP8 offline reader; QP_GLM53_EP_SIZE=8 is
  # the disclosed deviation from upstream EP4-on-B200).  Five cold student
  # captures + fp64 tokenwise KLD vs the 25 sealed teacher windows + the
  # decoded reference parity panel for the TP4 runtime qualification.
  test -f "$CKPT_K6/materialization-receipt.json"
  test -d "$TEACH" || { echo "teacher final-window logits missing at $TEACH" >&2; exit 1; }
  for run in 1 2 3 4 5; do
    [ -f "$RCPT/k6-student-run$run/capture-receipt.json" ] && continue
    QP_GLM53_EP_SIZE=8 "$VENV/bin/torchrun" --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 "$TOOLS/k6_student_capture.py" \
      --checkpoint "$CKPT_K6" --bf16 "$BF16" --teacher "$TEACH" \
      --profile k6 --cold-run "$run" --out "$RCPT/k6-student-run$run" \
      $( [ "$run" = 1 ] && echo --emit-reference-panel "$RCPT/k6-reference-panel.safetensors" )
  done
  "$PY" "$TOOLS/k6_kld_report.py" --profile k6 \
    --teacher "$TEACH" --runs "$RCPT"/k6-student-run{1,2,3,4,5} \
    --fp8-baseline 0.020615 --k4-baseline 0.024555 \
    --out "$RCPT/k6-packed-kld.json" --five-run-out "$RCPT/k6-five-run-kld.json" \
    --comparison-out "$RCPT/comparison-table.md"
  "$PY" - <<'PYEOF'
import json
r = json.load(open("/home/jl_fs/glm53-k6/receipts/k6-packed-kld.json"))
assert r["quality_gate_passed"] and r["measured_mean_kld"] < 0.06, r["measured_mean_kld"]
print("K6 mean tokenwise KLD:", r["measured_mean_kld"], "(gate < 0.06 GREEN)")
PYEOF
  # TP4 packed-runtime qualification (4 ranks; run from the 8-GPU box).
  ( cd "$PIPE" && PYTHONPATH=src "$VENV/bin/torchrun" --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=4 \
      scripts/qualify_glm53_custom_tp2_runtime.py \
      --model "$CKPT_K6" --bits 6 --exllamav3-source "$EXL3" \
      --reference-panel "$RCPT/k6-reference-panel.safetensors" \
      --max-abs-tolerance 0.5 --mean-abs-tolerance 0.005 \
      --max-new-tokens 4 \
      --observed-logits-output "$RCPT/k6-tp4-observed.safetensors" \
      --output "$RCPT/k6-tp4-runtime-receipt.json" )
  "$PY" -c "import json; r=json.load(open('$RCPT/k6-tp4-runtime-receipt.json')); assert r['qualified'], r; print('TP4 runtime receipt GREEN')"
  mark_done
  ;;

qualify_k8)
  # K8 TREE OF RECORD: the K8 contract/preparations were sealed from the
  # isolated pipeline-k8 tree (0007 applied pre-K6-completion, legal because
  # isolated).  The prepared-backend closure seal binds those exact bytes -
  # every K8 driver call must use the same tree, not $ROOT/pipeline.
  PIPE="$ROOT/pipeline-k8"
  export QP_PIPELINE_ROOT="$PIPE"
  export PYTHONPATH="$PIPE/src:$SHAPLEY:$SQGEXP"
  # K8 qualification: THREE cold EP8 student captures (budget; disclosed) +
  # fp64 tokenwise KLD + TP4 packed-runtime qualification (K8 is a shippable
  # flagship, so the runtime receipt is required like K6's).
  test -f "$CKPT_K8/materialization-receipt.json"
  # v2-0009 top-up: the upstream qualify script pins --bits choices=(4,6) and
  # would refuse --bits 8 at argparse before any GPU work.
  if grep -q 'choices=(4, 6))' "$PIPE/scripts/qualify_glm53_custom_tp2_runtime.py"; then
    test -f "$ROOT/patches-v2/0009-qualify-script-k8-admission.patch" \
      || { echo "patches-v2/0009-qualify-script-k8-admission.patch missing on fs - upload it first" >&2; exit 1; }
    ( cd "$PIPE" && patch -p1 -s < "$ROOT/patches-v2/0009-qualify-script-k8-admission.patch" )
    ( cd "$ROOT/patches-v2" && sha256sum 0009-*.patch ) | tee -a "$RCPT/patches-v2-applied.txt"
  fi
  test -d "$TEACH" || { echo "teacher final-window logits missing at $TEACH" >&2; exit 1; }
  for run in 1 2 3; do
    [ -f "$RCPT/k8-student-run$run/capture-receipt.json" ] && continue
    QP_GLM53_EP_SIZE=8 "$VENV/bin/torchrun" --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 "$TOOLS/k6_student_capture.py" \
      --checkpoint "$CKPT_K8" --bf16 "$BF16" --teacher "$TEACH" \
      --profile k8 --cold-run "$run" --out "$RCPT/k8-student-run$run" \
      $( [ "$run" = 1 ] && echo --emit-reference-panel "$RCPT/k8-reference-panel.safetensors" )
  done
  "$PY" "$TOOLS/k6_kld_report.py" --profile k8 \
    --teacher "$TEACH" --runs "$RCPT"/k8-student-run{1,2,3} \
    --fp8-baseline 0.020615 --k4-baseline 0.024555 \
    --out "$RCPT/k8-packed-kld.json" \
    --comparison-out "$RCPT/comparison-table.md"
  "$PY" -c "import json; r=json.load(open('$RCPT/k8-packed-kld.json')); assert r['measured_mean_kld'] < 0.06, r['measured_mean_kld']; print('K8 mean KLD:', r['measured_mean_kld'])"
  ( cd "$PIPE" && PYTHONPATH=src "$VENV/bin/torchrun" --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=4 \
      scripts/qualify_glm53_custom_tp2_runtime.py \
      --model "$CKPT_K8" --bits 8 --exllamav3-source "$EXL3" \
      --reference-panel "$RCPT/k8-reference-panel.safetensors" \
      --max-abs-tolerance 0.5 --mean-abs-tolerance 0.005 \
      --max-new-tokens 4 \
      --observed-logits-output "$RCPT/k8-tp4-observed.safetensors" \
      --output "$RCPT/k8-tp4-runtime-receipt.json" )
  "$PY" -c "import json; r=json.load(open('$RCPT/k8-tp4-runtime-receipt.json')); assert r['qualified'], r; print('K8 TP4 runtime receipt GREEN')"
  mark_done
  ;;

qualify_k6k8)
  test -f "$CKPT_K6K8/materialization-receipt.json"
  test -d "$TEACH" || { echo "teacher final-window logits missing at $TEACH" >&2; exit 1; }
  # Descoped to THREE cold runs (budget); disclosed in the receipt tree.
  for run in 1 2 3; do
    [ -f "$RCPT/k6k8-student-run$run/capture-receipt.json" ] && continue
    QP_GLM53_EP_SIZE=8 "$VENV/bin/torchrun" --master-port $((29500 + RANDOM % 2000)) --nproc-per-node=8 "$TOOLS/k6_student_capture.py" \
      --checkpoint "$CKPT_K6K8" --bf16 "$BF16" --teacher "$TEACH" \
      --profile k6k8 --cold-run "$run" --out "$RCPT/k6k8-student-run$run"
  done
  "$PY" "$TOOLS/k6_kld_report.py" --profile k6k8 \
    --teacher "$TEACH" --runs "$RCPT"/k6k8-student-run{1,2,3} \
    --fp8-baseline 0.020615 --k4-baseline 0.024555 \
    --out "$RCPT/k6k8-packed-kld.json" \
    --comparison-out "$RCPT/comparison-table.md"
  "$PY" -c "import json; r=json.load(open('$RCPT/k6k8-packed-kld.json')); assert r['measured_mean_kld'] < 0.06, r['measured_mean_kld']; print('K6K8 mean KLD:', r['measured_mean_kld'])"
  mark_done
  ;;

upload_weights)
  hf_env
  # Release contract: weights publish as qualified only after the KLD gate is
  # green.  QP_PUBLISH_UNQUALIFIED=1 is the explicit, operator-only override
  # for the "publish receipts + failure analysis" path in the RUNBOOK.
  if [ ! -f "$DONE/qualify_k6.done" ] && [ "${QP_PUBLISH_UNQUALIFIED:-0}" != 1 ]; then
    echo "qualify_k6 not done - refusing to publish weights (set QP_PUBLISH_UNQUALIFIED=1 only for the disclosed failure-analysis path)" >&2
    exit 5
  fi
  # Fail fast: the README card is authored by the operator during P3/P4 prep
  # (license read off the pinned source revision - k6_publish enforces credits).
  test -f "$ROOT/cards/K6-README.md" \
    || { echo "README card missing at $ROOT/cards/K6-README.md - author it before upload_weights (RUNBOOK P4)" >&2; exit 1; }
  "$PY" "$TOOLS/k6_publish.py" weights \
    --checkpoint "$CKPT_K6" --repo malaiwah/GLM-5.3-Flash-TR3-6bpw \
    --recipe "$ROOT/recipes/k6.json" --receipts "$RCPT" \
    --card "$ROOT/cards/K6-README.md"
  if [ -f "$CKPT_K8/materialization-receipt.json" ] && [ -f "$DONE/qualify_k8.done" ]; then
    test -f "$ROOT/cards/K8-README.md" \
      || { echo "README card missing at $ROOT/cards/K8-README.md - author it before the K8 upload" >&2; exit 1; }
    "$PY" "$TOOLS/k6_publish.py" weights \
      --checkpoint "$CKPT_K8" --repo malaiwah/GLM-5.3-Flash-TR3-8bpw \
      --recipe "$ROOT/recipes/k8.json" --receipts "$RCPT" \
      --card "$ROOT/cards/K8-README.md"
  fi
  if [ -f "$CKPT_K6K8/materialization-receipt.json" ] && [ -f "$DONE/qualify_k6k8.done" ]; then
    "$PY" "$TOOLS/k6_publish.py" weights \
      --checkpoint "$CKPT_K6K8" --repo malaiwah/GLM-5.3-Flash-TR3-6bpwK8-mixed \
      --recipe "$ROOT/recipes/k6k8.json" --receipts "$RCPT" \
      --card "$ROOT/cards/K6K8-README.md"
  fi
  mark_done
  ;;

publish_receipts)
  hf_env
  # Receipts + tokenwise-KLD vectors + comparison table into the existing
  # fidelity dataset under reports/exl3-k6/.
  "$PY" "$TOOLS/k6_publish.py" receipts \
    --receipts "$RCPT" --repo malaiwah/GLM-5.3-Flash-fidelity-suite-v1 \
    --prefix reports/exl3-k6 \
    --discussion-draft "$RCPT/discussion-comment.md"
  mark_done
  ;;

*)
  echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
