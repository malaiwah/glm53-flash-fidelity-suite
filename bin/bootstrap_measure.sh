#!/usr/bin/env bash
# Measurement-only bootstrap for a COLD instance.
#
#   bootstrap_measure.sh            (called by stage_measure.sh setup)
#
# WHY THIS EXISTS.  The cloud recipe used to delegate its bootstrap to
# `engines/stage_campaign.sh setup` (called stage_k6.sh until 2026-08-31), on the
# reasoning that the campaign script's container recipe is the proven one.  Two facts made that unrunnable, and both only show
# up on a cold box:
#
#   1. stage_campaign.sh was never in bin/BUNDLE.txt, so it -- and the patches-v2
#      series it applies -- never reached the instance at all.  Its first line
#      of work is `bash $ROOT/stage_campaign.sh setup` against a file that is not
#      there.
#   2. stage_campaign.sh setup is an ENCODING campaign bootstrap.  It clones
#      ShapleyMCG and the sparse sqg-mcg encoder, then hard-stops on a CLOSURE
#      GATE demanding the r10 codec closure or an operator-signed
#      RECONSTRUCTION-ACCEPTED.json.  A measurement decodes; it never encodes.
#      Gating a measurement on an encoder's closure is a dependency on work
#      that has nothing to do with the number.
#
# So the measurement lane owns its own bootstrap.  Everything it DOES keep is
# byte-for-byte the proven recipe (DECISIONS.md item 5) that produced the K6,
# K8 and BF16-floor streaming rows -- same python, same torch/cu130 wheel, same
# transformers, same pipeline pin, same patch series (0001-0006 + 0008), so the
# reader bytes the capture receipts bind are the same bytes.  What it drops is
# only what encoding needs: ShapleyMCG, sqg-mcg, the closure gate, and the
# calibration trees.
#
# exllamav3 is built ONLY IF the pipeline cannot import without it.  Neither
# stream_score.py nor kld_report.py imports the package; the CUDA toolkit +
# extension build is ~20 minutes of rental that a decode-only run should not
# pay for on faith.  The probe decides, not an assumption.
#
# Idempotent: every step is guarded, so a spot preemption re-runs it for free.
# NEVER `set -x` here: HF_TOKEN may be exported by the caller.
set -euo pipefail

# FIDELITY_K6_ROOT is the pre-2026-08-31 spelling, still read as a fallback.
ROOT="${FIDELITY_ENGINE_ROOT:-${FIDELITY_K6_ROOT:-/home/jl_fs/fidelity-engine}}"
FS="${FIDELITY_FS_ROOT:-/home/jl_fs/fidelity}"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
PIPE="$ROOT/pipeline"
EXL3="$ROOT/exllamav3"
RCPT="$FS/receipts"
PATCHES="$ROOT/patches-v2"

# Pins, verbatim from engines/stage_campaign.sh (the tree that produced the sealed rows).
PIPE_REPO=https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw
PIPE_PIN=ce1bf9706b6aa18435e2baccab63bdd72299257c
EXL3_REPO=https://github.com/turboderp-org/exllamav3
EXL3_PIN=c5d9c657966ffeeaa9353f0cc899f18629da4a13
TORCH_SPEC="torch==2.11.0"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

mkdir -p "$ROOT" "$RCPT"
log() { echo "[$(date -u +%FT%TZ)] bootstrap_measure: $*"; }

ASROOT=""
[ "$(id -u)" = 0 ] || ASROOT="sudo"

# ---- 1. python 3.12 -------------------------------------------------------
if ! command -v python3.12 >/dev/null; then
  log "installing python3.12 (deadsnakes)"
  $ASROOT apt-get update -qq >/dev/null 2>&1 || true
  $ASROOT apt-get install -y -qq software-properties-common >/dev/null 2>&1 || true
  $ASROOT add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 || true
  $ASROOT apt-get update -qq >/dev/null 2>&1 || true
  for p in python3.12 python3.12-venv python3.12-dev; do
    $ASROOT apt-get install -y -qq "$p" >/dev/null 2>&1 \
      || log "apt $p failed (tolerated; the guard below decides)"
  done
fi
PYBIN="$(command -v python3.12 || true)"
[ -n "$PYBIN" ] || PYBIN="$(command -v python3)"
"$PYBIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || {
  echo "python is not 3.12 ($("$PYBIN" -V 2>&1)); the proven env recipe is py3.12-only" >&2
  exit 1
}
"$PYBIN" -V | tee "$RCPT/python-version.txt"

# ---- 2. venv + the pinned wheel set --------------------------------------
if [ ! -x "$PY" ]; then
  log "creating venv at $VENV"
  "$PYBIN" -m venv "$VENV"
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' || {
  echo "existing venv at $VENV is not py3.12 - delete it and re-run setup" >&2; exit 1; }
if ! "$PY" -c "import torch, transformers, safetensors, huggingface_hub, hf_transfer" 2>/dev/null; then
  log "installing the pinned wheel set (torch 2.11.0+cu130, transformers 5.16.1)"
  "$VENV/bin/pip" -q install --upgrade pip
  "$VENV/bin/pip" -q install setuptools wheel ninja packaging
  "$VENV/bin/pip" -q install $TORCH_SPEC --index-url "$TORCH_INDEX"
  "$VENV/bin/pip" -q install "transformers==5.16.1" safetensors numpy \
      huggingface_hub hf_transfer accelerate rich tokenizers pillow \
      "pydantic==2.5.3" "formatron==0.5.0" kbnf
fi
# hf_transfer is REQUIRED, not optional: every fetch stage exports
# HF_HUB_ENABLE_HF_TRANSFER=1, and huggingface_hub errors out when that flag is
# set without the package.  The guard above skips the whole wheel block when the
# template already ships torch/transformers/huggingface_hub -- which is the
# normal case on the pytorch template -- so hf_transfer alone can be missing on
# an otherwise complete box.  That cost a real measurement a slow fetch before
# it was caught (JOURNAL, lesson 33).  Ensure it separately and idempotently.
if ! "$PY" -c "import hf_transfer" 2>/dev/null; then
  log "hf_transfer absent (template supplied the rest) - installing it alone"
  "$VENV/bin/pip" -q install hf_transfer
fi
"$PY" -c "import hf_transfer" 2>/dev/null || {
  echo "hf_transfer still not importable; fetch stages set HF_HUB_ENABLE_HF_TRANSFER=1 and would fail" >&2
  exit 1; }

"$PY" - <<'PY' | tee "$RCPT/wheel-versions.txt"
import torch, transformers, safetensors, numpy, hf_transfer
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "| transformers", transformers.__version__,
      "| safetensors", safetensors.__version__, "| numpy", numpy.__version__,
      "| hf_transfer", getattr(hf_transfer, "__version__", "present"))
PY

# ---- 3. the pipeline at its pin, with the measurement patch series --------
if [ ! -d "$PIPE/.git" ]; then
  log "cloning the quant pipeline @ $PIPE_PIN"
  git clone -q "$PIPE_REPO" "$PIPE"
fi
git -C "$PIPE" fetch -q origin "$PIPE_PIN" 2>/dev/null || true
if ! grep -q "_STORED_BITS" "$PIPE/src/quant_pipeline/checkpoint/packed_payload.py"; then
  git -C "$PIPE" checkout -q "$PIPE_PIN"
  # SH-03. `A && B` is an AND-OR list, which bash EXEMPTS from set -e, so this asserted
  # nothing: a dirty pipeline tree was silently accepted and the series applied on top of
  # it, producing a different reader with no error. Reproduced end to end -- a dirty file
  # plus a patch touching a different region of it gives `patch` exit 0, no .rej, and a
  # decoder that hashes differently from the clean pin+series build. 0006 patches the
  # reader the measurement decodes with, and reader_abi_sha256 is only RECORDED, never
  # compared against an expected value, so that divergence is invisible downstream.
  git -C "$PIPE" diff --quiet && git -C "$PIPE" diff --cached --quiet || {
    echo "$PIPE working tree is dirty - refusing to patch onto it" >&2
    git -C "$PIPE" status --porcelain | head -20 >&2
    exit 1; }
  test -d "$PATCHES" || { echo "patches-v2 missing at $PATCHES - the bundle did not upload it" >&2; exit 1; }
  log "applying patches 0001-0006"
  _want=$(grep -cE '^000[1-6]-.*\.patch$' "$PATCHES/SERIES" || echo 0)
  _have=$(ls -1 "$PATCHES"/000[1-6]-*.patch 2>/dev/null | wc -l | tr -d ' ')
  [ "$_want" -gt 0 ] && [ "$_have" = "$_want" ] || {
    echo "patch series incomplete: SERIES names $_want of 0001-0006, $PATCHES holds $_have" >&2
    ls -1 "$PATCHES" >&2
    exit 1; }
  # SH-23. The glob was applied with no cardinality check, so a bundle that dropped a
  # patch produced a differently-patched tree, exit 0, and a receipt quietly listing one
  # fewer line. BUNDLE.txt's own policy is "missing files are skipped (the controller
  # logs which)", and measure_cloud implements that as a WARNING -- so the upstream half
  # is fail-open by design and this is the consumer that has to notice.
  # Asserted against SERIES rather than a magic 6: SERIES is uploaded, is already hashed
  # into the receipt, and this also catches an unexpected EXTRA file matching the glob.
  ( cd "$PIPE" && for p in "$PATCHES"/000[1-6]-*.patch; do patch -p1 -s < "$p"; done )
  ( cd "$PATCHES" && sha256sum 000[1-6]-*.patch SERIES ) | tee "$RCPT/patches-v2-applied.txt"
else
  [ "$(git -C "$PIPE" rev-parse HEAD)" = "$PIPE_PIN" ] \
    || { echo "$PIPE HEAD is not $PIPE_PIN" >&2; exit 1; }
fi
# 0008 widens normalization ALLOWED_BITS to include 6/8; the streaming reader
# import path pulls that constant in.  Touches no reader/closure-hashed file.
V31="$PIPE/src/quant_pipeline/normalization/absolute_v31.py"
if [ -f "$V31" ] && grep -qF "ALLOWED_BITS = frozenset((3, 4, 5))" "$V31"; then
  test -f "$PATCHES/0008-v31-allowed-bits-k6-k8.patch" \
    || { echo "patches-v2/0008 missing on fs" >&2; exit 1; }
  log "applying patch 0008"
  ( cd "$PIPE" && patch -p1 -s < "$PATCHES/0008-v31-allowed-bits-k6-k8.patch" )
  ( cd "$PATCHES" && sha256sum 0008-*.patch ) | tee -a "$RCPT/patches-v2-applied.txt"
fi

# ---- 4. exllamav3 ONLY if the pipeline import demands it ------------------
probe() {
  "$PY" -c "
import sys; sys.path.insert(0, '$PIPE/src')
import quant_pipeline.evaluation.glm53_packed_k4_reader
import quant_pipeline.evaluation.glm53_logits
import quant_pipeline.core.artifacts
import quant_pipeline.campaign.glm53_direct_k4
print('pipeline import OK')
" 2>&1
}
if ! probe >"$RCPT/pipeline-import.txt" 2>&1; then
  log "pipeline import failed without exllamav3; building it (see receipt)"
  cat "$RCPT/pipeline-import.txt" || true
  if ! { command -v nvcc >/dev/null && nvcc --list-gpu-arch 2>/dev/null | grep -q compute_100; }; then
    ( cd /tmp \
      && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      && $ASROOT dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null 2>&1 \
      && $ASROOT apt-get update -qq >/dev/null 2>&1 \
      && $ASROOT apt-get install -y -qq cuda-toolkit-13-0 >/dev/null 2>&1 ) \
      || log "cuda-toolkit-13-0 install failed (the build below decides)"
    $ASROOT ln -sfn /usr/local/cuda-13.0 /usr/local/cuda 2>/dev/null || true
    export PATH="/usr/local/cuda-13.0/bin:$PATH"
  fi
  "$PY" -c "import flash_attn" 2>/dev/null || "$VENV/bin/pip" -q install "$FLASH_ATTN_WHL"
  if [ ! -d "$EXL3/.git" ]; then git clone -q "$EXL3_REPO" "$EXL3"; fi
  git -C "$EXL3" checkout -q "$EXL3_PIN"
  if ! TORCH_CUDA_ARCH_LIST="9.0;10.0" "$PY" -c "import exllamav3" 2>/dev/null; then
    ( cd "$EXL3" && TORCH_CUDA_ARCH_LIST="9.0;10.0" \
        "$VENV/bin/pip" -q install --no-build-isolation --no-deps -e . )
  fi
  probe | tee "$RCPT/pipeline-import.txt"
else
  cat "$RCPT/pipeline-import.txt"
  log "exllamav3 NOT built: the measurement path imports the pipeline without it"
  echo "not-built: pipeline imports cleanly without exllamav3" > "$RCPT/exllamav3-build.txt"
fi

# ---- INSTALL/CHECK SPLIT (container build) --------------------------------
# Steps 1-4 INSTALL; steps 5-6 CHECK.  `container/Dockerfile` runs this script
# at image build time with FIDELITY_BOOTSTRAP_INSTALL_ONLY=1 so the four
# minutes of apt/pip/git every rental used to pay are baked into a layer -- and
# then runs the SAME script unset at container start, where every install step
# is guarded and no-ops, and the checks run.
#
# The split is not tidiness.  The gguf battery's rung 1b re-decodes the
# committed real bytes on THIS BOX'S CUDA device and demands torch.equal
# against the reference; a docker builder has no GPU, so running it there would
# either fail a good build or, worse, pass vacuously and retire a check that
# only means something on the measuring machine.  Same reasoning for the other
# three adapter batteries: they are pre-flight, and pre-flight belongs to the
# flight.
#
# Unset (every existing caller) the behaviour below is byte-for-byte unchanged.
if [ -n "${FIDELITY_BOOTSTRAP_INSTALL_ONLY:-}" ]; then
  log "install-only: steps 5-6 (adapter import + offline selftests) deferred to run time"
  log "bootstrap complete (install-only)"
  exit 0
fi

# ---- 5. the surface adapters must import too ------------------------------
# Cheap here, expensive after a 165 GB fetch: a missing bundle entry or a typo
# in an adapter is a syntax error we can see for free, before any download.
QP_PIPELINE_ROOT="$PIPE" "$PY" - <<PY | tee "$RCPT/adapter-import.txt"
import sys
sys.path.insert(0, "$FS/engines/tools")
sys.path.insert(0, "$PIPE/src")
import exl3hf_surface, dione_surface   # noqa: F401
print("surface adapters import OK:", exl3hf_surface.EXL3HF_SURFACE_SCHEMA)
PY

# ---- 6. the exl3hf offline selftest, INCLUDING the rungs that need the
#         pipeline (they self-skip on the laptop; this is the only place the
#         mcg-parity rung can run before a paid capture) --------------------
if [ -f "$FS/engines/tools/selftest_tr3_offline.py" ]; then
  log "running the tr3 offline selftest (seal recompute + mcg decode parity)"
  ( cd "$FS/engines/tools" && PYTHONPATH="$PIPE/src" "$PY" selftest_tr3_offline.py ) \
    | tee "$RCPT/selftest-tr3.txt"
fi
if [ -f "$FS/engines/tools/selftest_dione_offline.py" ]; then
  log "running the dione offline selftest (pack layout + decode identity + real-index census)"
  ( cd "$FS/engines/tools" && PYTHONPATH="$PIPE/src" "$PY" selftest_dione_offline.py \
      --pipeline-root "$PIPE" ) | tee "$RCPT/selftest-dione.txt"
fi
if [ -f "$FS/engines/tools/selftest_dione_stream_offline.py" ]; then
  log "running the dione STREAMING front-end selftest (manifests + shards + scope + materializer)"
  ( cd "$FS/engines/tools" && PYTHONPATH="$PIPE/src" "$PY" selftest_dione_stream_offline.py ) \
    | tee "$RCPT/selftest-dione-stream.txt"
fi
if [ -f "$FS/engines/tools/selftest_exl3hf_offline.py" ]; then
  log "running the exl3hf offline selftest (mcg-parity rung included)"
  ( cd "$FS/engines/tools" && PYTHONPATH="$PIPE/src" "$PY" selftest_exl3hf_offline.py ) \
    | tee "$RCPT/selftest-exl3hf.txt"
fi
# The GGUF surface's offline battery has been in bin/BUNDLE.txt since the lane
# landed -- "so the refusals and the two layout audits can be re-run on the
# instance before a paid capture" -- and nothing here ever ran it. A capability
# nothing invokes is indistinguishable from a missing one, which is the lesson
# the GGUF lane itself was written from.
#
# Rung 1b is why it matters now: it re-decodes the committed REAL UD-Q4_K_XL
# bytes on THIS BOX'S CUDA device and demands torch.equal against the CPU output
# that rung 1 proved bitwise-equal to gguf-py. That is the acceptance test for
# the accelerator dequant the capture is about to use by default, and a laptop's
# MPS pass is evidence for CUDA, not proof of it. It runs here, before the fetch
# and before any GPU-hour is spent on a number.
#
# Fail-CLOSED, and by the file's own `set -euo pipefail` rather than by anything
# written here: `cmd | tee` propagates cmd's status under pipefail, so a decode
# that does not reproduce the reference stops the bootstrap instead of being
# printed into a log nobody reads until the receipt is already sealed.
if [ -f "$FS/engines/tools/selftest_gguf_offline.py" ]; then
  log "running the gguf offline selftest (gguf-py parity + CUDA decode parity)"
  ( cd "$FS/engines/tools" && PYTHONPATH="$PIPE/src" "$PY" selftest_gguf_offline.py \
      --pipeline-root "$PIPE" ) | tee "$RCPT/selftest-gguf.txt"
fi

log "bootstrap complete"
