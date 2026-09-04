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
# exllamav3 is built ONLY IF the pipeline imports it.  Neither stream_score.py
# nor kld_report.py needs the package; the CUDA toolkit + extension build is
# ~20 minutes of rental that a decode-only run should not pay for on faith.
# When an import does load it, its checkout and import path are verified rather
# than trusting whichever editable package the template happens to expose.
#
# Deterministic: source trees are reconstructed from their pins on every run.
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

# Source commits are accepted only when both their Git tree and deterministic
# git-archive SHA-256 match.  Every Python wheel, including transitive
# dependencies, is an exact URL with an exact SHA-256 in WHEEL_LOCK.
PIPE_REPO=https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw
PIPE_PIN=ce1bf9706b6aa18435e2baccab63bdd72299257c
PIPE_TREE=9aede904994714b0aad824646f80046bb7b6c874
PIPE_ARCHIVE_SHA256=cfb7bf9f2c11e71683ce3a7fe1e1d8a3cdd089ebdd1b2eec42bf604c18a05fbd
EXL3_REPO=https://github.com/turboderp-org/exllamav3
EXL3_PIN=c5d9c657966ffeeaa9353f0cc899f18629da4a13
EXL3_TREE=8b00c03978d850d2b53224acbd92018e107707d1
EXL3_ARCHIVE_SHA256=3c13cdd74d5fc3c75f426c7b6ae8d8543207483831522280d1d641b974cf452c
# No torch2.11-tagged flash-attn 2.8.3 wheel exists.  This authored
# torch2.10-tagged artifact is the proven compatibility choice; on a measuring
# GPU validate_flash_attn verifies it with an actual kernel call.
FLASH_ATTN_VERSION="2.8.3+cu13torch2.10cxx11abitrue"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FLASH_ATTN_SHA256=910d8db9def162de5b7c15474b933e7e2371e93733b980e9d3c07cd3bf2f568e
CUDA_KEYRING_SHA256=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_LOCK="$SCRIPT_DIR/requirements-cu130-py312.lock"

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

# ---- 2. venv + the exact hashed wheel closure ---------------------------
# Inside the measurement image the exact closure is already baked at
# FIDELITY_IMAGE_ROOT (venv, pipeline and exllamav3 at their pins).  A
# per-attempt engine root elsewhere is SEEDED from it by symlink when the
# image's wheel lock is byte-identical to this bundle's; otherwise the fresh
# path below runs untouched.  The receipts record which happened.
IMAGE_ROOT="${FIDELITY_IMAGE_ROOT:-}"
seed_from_image() {
  [ -n "$IMAGE_ROOT" ] && [ -d "$IMAGE_ROOT/venv" ] || return 0
  [ "$(realpath -m "$ROOT")" != "$(realpath -m "$IMAGE_ROOT")" ] || return 0
  local image_lock="$IMAGE_ROOT/suite/bin/requirements-cu130-py312.lock"
  if [ ! -f "$image_lock" ] || [ "$(sha256sum < "$image_lock")" != "$(sha256sum < "$WHEEL_LOCK")" ]; then
    log "image wheel lock differs from this bundle's; building a fresh venv"
    echo "image-seed: refused (wheel lock differs)" | tee "$RCPT/image-seed.txt"
    return 0
  fi
  if [ ! -e "$VENV" ]; then
    ln -s "$IMAGE_ROOT/venv" "$VENV"
    log "venv seeded from image $IMAGE_ROOT/venv (wheel lock matches)"
  fi
  local name dest
  for name in pipeline exllamav3; do
    case "$name" in pipeline) dest="$PIPE" ;; *) dest="$EXL3" ;; esac
    if [ -d "$IMAGE_ROOT/$name/.git" ] && [ ! -e "$dest" ]; then
      ln -s "$IMAGE_ROOT/$name" "$dest"
      log "$name seeded from image $IMAGE_ROOT/$name (pins verified below)"
    fi
  done
  echo "image-seed: venv=$(readlink -f "$VENV") pipeline=$(readlink -f "$PIPE" 2>/dev/null || echo absent) exllamav3=$(readlink -f "$EXL3" 2>/dev/null || echo absent)" \
    | tee "$RCPT/image-seed.txt"
}
seed_from_image
if [ ! -x "$PY" ]; then
  log "creating venv at $VENV"
  "$PYBIN" -m venv "$VENV"
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' || {
  echo "existing venv at $VENV is not py3.12 - delete it and re-run setup" >&2; exit 1; }

validate_direct_wheels() {
  "$PY" - <<'PY'
# DIRECT_WHEEL_VALIDATOR_BEGIN
import importlib.metadata as metadata
import sys

expected = {
    "pip": "25.0.1",
    "setuptools": "75.8.0",
    "wheel": "0.45.1",
    "ninja": "1.11.1.3",
    "packaging": "24.2",
    "torch": "2.11.0+cu130",
    "transformers": "5.16.1",
    "safetensors": "0.8.0",
    "numpy": "2.5.2",
    "huggingface-hub": "1.28.0",
    "hf-transfer": "0.1.9",
    "accelerate": "1.14.0",
    "rich": "13.9.4",
    "tokenizers": "0.23.1",
    "Pillow": "11.1.0",
    "pydantic": "2.5.3",
    "formatron": "0.5.0",
    "kbnf": "0.4.2",
}
actual = {}
problems = []
for distribution, wanted in expected.items():
    try:
        actual[distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        problems.append(f"{distribution}: missing (expected {wanted})")
        continue
    if actual[distribution] != wanted:
        problems.append(
            f"{distribution}: {actual[distribution]} installed, expected {wanted}"
        )

try:
    import torch
except Exception as exc:
    problems.append(f"torch: distribution is present but import failed: {exc}")
else:
    if torch.version.cuda != "13.0":
        problems.append(
            f"torch CUDA: {torch.version.cuda!r} reported, expected '13.0'"
        )

if problems:
    print("direct-wheel validation failed:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    raise SystemExit(1)

for distribution in expected:
    print(f"{distribution}=={actual[distribution]}")
print("torch.version.cuda==13.0")
# DIRECT_WHEEL_VALIDATOR_END
PY
}

_direct_wheels_reinstalled=0
if ! validate_direct_wheels >"$RCPT/wheel-validation-before.txt" 2>&1; then
  cat "$RCPT/wheel-validation-before.txt" >&2
  _direct_wheels_reinstalled=1
fi
[ -f "$WHEEL_LOCK" ] && [ ! -L "$WHEEL_LOCK" ] || {
  echo "exact wheel lock is absent or unsafe: $WHEEL_LOCK" >&2
  exit 1
}
sha256sum "$WHEEL_LOCK" | tee "$RCPT/wheel-lock-sha256.txt"
log "enforcing the exact hashed Python 3.12/cu130 wheel closure"
"$PY" -m pip -q install --no-deps --require-hashes --only-binary=:all: \
  -r "$WHEEL_LOCK"
validate_direct_wheels | tee "$RCPT/wheel-versions.txt"
"$PY" -m pip check | tee "$RCPT/pip-check.txt"

# ---- 3. the pipeline at its pin, with the measurement patch series --------
reconstruct_checkout() {
  local repo="$1" pin="$2" expected_tree="$3" expected_archive_sha256="$4"
  local destination="$5" label="$6"
  local destination_real="" top="" top_real="" reuse=0
  case "$destination" in
    "$PIPE"|"$EXL3") ;;
    *)
      echo "refusing to reconstruct non-dedicated path: $destination" >&2
      exit 1
      ;;
  esac
  # A symlink is never reused -- except one the image seed placed, pointing
  # at the image's own checkout under FIDELITY_IMAGE_ROOT, which is reset to
  # the exact pin below like any reused checkout.
  local seeded=0
  if [ -L "$destination" ] && [ -n "$IMAGE_ROOT" ]; then
    case "$(realpath -m "$destination")" in
      "$(realpath -m "$IMAGE_ROOT")"/*) seeded=1 ;;
    esac
  fi
  if { [ ! -L "$destination" ] || [ "$seeded" -eq 1 ]; } && [ -d "$destination" ]; then
    destination_real="$(realpath "$destination" 2>/dev/null || true)"
    top="$(git -C "$destination" rev-parse --show-toplevel 2>/dev/null || true)"
    [ -z "$top" ] || top_real="$(realpath "$top" 2>/dev/null || true)"
    if [ -n "$destination_real" ] && [ "$top_real" = "$destination_real" ]; then
      reuse=1
    fi
  fi
  if [ "$reuse" -eq 0 ]; then
    # destination is one of the two dedicated children checked above.  Removing
    # a symlink removes only the link; a nested/outer worktree is never cleaned.
    rm -rf -- "$destination"
    log "cloning $label @ $pin"
    git clone -q "$repo" "$destination"
  fi
  if git -C "$destination" remote get-url origin >/dev/null 2>&1; then
    git -C "$destination" remote set-url origin "$repo"
  else
    git -C "$destination" remote add origin "$repo"
  fi
  if ! git -C "$destination" cat-file -e "$pin^{commit}" 2>/dev/null; then
    git -C "$destination" fetch -q origin "$pin"
  fi
  git -C "$destination" checkout -q --detach -f "$pin"
  git -C "$destination" reset -q --hard "$pin"
  git -C "$destination" clean -q -ffdx
  [ "$(git -C "$destination" rev-parse HEAD)" = "$pin" ] || {
    echo "$label checkout did not resolve to $pin" >&2
    exit 1
  }
  [ "$(git -C "$destination" rev-parse 'HEAD^{tree}')" = "$expected_tree" ] || {
    echo "$label tree does not match the authored pin" >&2
    exit 1
  }
  local archive_sha256
  archive_sha256="$(
    git -C "$destination" archive --format=tar HEAD | sha256sum | cut -d' ' -f1
  )"
  [ "$archive_sha256" = "$expected_archive_sha256" ] || {
    echo "$label deterministic archive SHA-256 mismatch" >&2
    exit 1
  }
}

# Always start from the exact clean commit.  A marker in a previously patched
# file, the right HEAD with local edits, and ignored build residue are all
# discarded before the authored series is applied.
reconstruct_checkout \
  "$PIPE_REPO" "$PIPE_PIN" "$PIPE_TREE" "$PIPE_ARCHIVE_SHA256" \
  "$PIPE" "quant pipeline"
test -f "$PATCHES/SERIES" || {
  echo "patches-v2/SERIES missing at $PATCHES - the bundle did not upload it" >&2
  exit 1
}
mapfile -t _series < <(
  grep -E '^(000[1-6]|0008)-.*\.patch$' "$PATCHES/SERIES"
)
shopt -s nullglob
_files=( "$PATCHES"/000[1-6]-*.patch "$PATCHES"/0008-*.patch )
shopt -u nullglob
[ "${#_series[@]}" -eq 7 ] && [ "${#_files[@]}" -eq "${#_series[@]}" ] || {
  echo "measurement patch series incomplete: SERIES names ${#_series[@]}, filesystem holds ${#_files[@]} (expected 7)" >&2
  exit 1
}
for _index in "${!_series[@]}"; do
  [ "$(basename "${_files[$_index]}")" = "${_series[$_index]}" ] || {
    echo "measurement patch mismatch at entry $_index: SERIES=${_series[$_index]} file=$(basename "${_files[$_index]}")" >&2
    exit 1
  }
done
log "applying cardinality-checked measurement patches 0001-0006 + 0008"
for _patch in "${_series[@]}"; do
  ( cd "$PIPE" && patch -p1 -s --fuzz=0 < "$PATCHES/$_patch" )
done
( cd "$PATCHES" && sha256sum "${_series[@]}" SERIES ) \
  | tee "$RCPT/patches-v2-applied.txt"

# ---- 4. exllamav3 ONLY if a successful pipeline import loads it -----------
probe() {
  QP_PIPELINE_ROOT="$PIPE" "$PY" - <<'PY'
import os
import sys

sys.path.insert(0, os.path.join(os.environ["QP_PIPELINE_ROOT"], "src"))
import quant_pipeline.evaluation.glm53_packed_k4_reader
import quant_pipeline.evaluation.glm53_logits
import quant_pipeline.core.artifacts
import quant_pipeline.campaign.glm53_direct_k4

loaded = any(
    name == "exllamav3" or name.startswith("exllamav3.")
    for name in sys.modules
)
print("pipeline import OK")
print("exllamav3-loaded:", "yes" if loaded else "no")
PY
}

validate_flash_attn() {
  FLASH_ATTN_EXPECTED="$FLASH_ATTN_VERSION" \
  FLASH_ATTN_INSTALL_ONLY="${FIDELITY_BOOTSTRAP_INSTALL_ONLY:-}" \
    "$PY" - <<'PY'
import importlib.metadata as metadata
import os

wanted = os.environ["FLASH_ATTN_EXPECTED"]
actual = metadata.version("flash-attn")
if actual.lower() != wanted:
    raise SystemExit(f"flash-attn {actual} installed, expected {wanted}")
import flash_attn
import torch

print(f"flash-attn=={actual}")
if os.environ["FLASH_ATTN_INSTALL_ONLY"]:
    print("flash-attn CUDA runtime smoke=deferred (install-only)")
else:
    if not torch.cuda.is_available():
        raise SystemExit("flash-attn runtime validation requires the measuring CUDA device")
    from flash_attn import flash_attn_func

    q = torch.randn((1, 4, 1, 16), device="cuda", dtype=torch.float16)
    output = flash_attn_func(q, q, q, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    if output.shape != q.shape or not torch.isfinite(output).all().item():
        raise SystemExit("flash-attn CUDA runtime smoke returned invalid output")
    print("flash-attn CUDA runtime smoke=passed")
PY
}

validate_exl3_import() {
  EXL3_EXPECTED="$EXL3" "$PY" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

expected = Path(os.environ["EXL3_EXPECTED"]).resolve()
dist = metadata.distribution("exllamav3")
direct = json.loads(dist.read_text("direct_url.json") or "{}")
parsed = urlparse(direct.get("url", ""))
source = Path(unquote(parsed.path)).resolve() if parsed.scheme == "file" else None
if source != expected or direct.get("dir_info", {}).get("editable") is not True:
    location = str(source) if source is not None else "<non-file source>"
    raise SystemExit(
        f"exllamav3 is not the editable checkout at {expected}; source={location}"
    )

import exllamav3

module_files = []
for name, module in tuple(sys.modules.items()):
    if name != "exllamav3" and not name.startswith("exllamav3."):
        continue
    filename = getattr(module, "__file__", None)
    if filename:
        resolved = Path(filename).resolve()
        module_files.append(resolved)
        try:
            resolved.relative_to(expected)
        except ValueError:
            raise SystemExit(
                f"{name} imported from {resolved}, outside exact checkout {expected}"
            )
if not module_files:
    raise SystemExit("exllamav3 imported without a verifiable module path")
print(f"exllamav3 source={expected}")
PY
}

_needs_exl3=1
if probe >"$RCPT/pipeline-import.txt" 2>&1 \
    && grep -q '^exllamav3-loaded: no$' "$RCPT/pipeline-import.txt"; then
  _needs_exl3=0
fi
if [ "$_needs_exl3" -eq 1 ]; then
  log "pipeline requires exllamav3; reconstructing its exact source checkout"
  cat "$RCPT/pipeline-import.txt" || true
  if ! { command -v nvcc >/dev/null && nvcc --list-gpu-arch 2>/dev/null | grep -q compute_100; }; then
    ( cd /tmp \
      && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      && printf '%s  %s\n' "$CUDA_KEYRING_SHA256" \
          cuda-keyring_1.1-1_all.deb | sha256sum -c - \
      && $ASROOT dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null 2>&1 \
      && $ASROOT apt-get update -qq >/dev/null 2>&1 \
      && $ASROOT apt-get install -y -qq cuda-toolkit-13-0 >/dev/null 2>&1 ) \
      || log "cuda-toolkit-13-0 install failed (the build below decides)"
    $ASROOT ln -sfn /usr/local/cuda-13.0 /usr/local/cuda 2>/dev/null || true
    export PATH="/usr/local/cuda-13.0/bin:$PATH"
  fi
  if ! FIDELITY_BOOTSTRAP_INSTALL_ONLY=1 \
      validate_flash_attn >"$RCPT/flash-attn-version.txt" 2>&1; then
    "$PY" -m pip -q install --force-reinstall --no-deps \
      "$FLASH_ATTN_WHL#sha256=$FLASH_ATTN_SHA256"
  fi
  validate_flash_attn | tee "$RCPT/flash-attn-version.txt"

  reconstruct_checkout \
    "$EXL3_REPO" "$EXL3_PIN" "$EXL3_TREE" "$EXL3_ARCHIVE_SHA256" \
    "$EXL3" "exllamav3"
  if [ "$_direct_wheels_reinstalled" -eq 1 ] \
      || ! validate_exl3_import >"$RCPT/exllamav3-build.txt" 2>&1; then
    # A newly installed torch invalidates any previously compiled editable
    # extension even when its direct_url and Python package path are unchanged.
    ( cd "$EXL3" && TORCH_CUDA_ARCH_LIST="9.0;10.0" \
        "$PY" -m pip -q install --force-reinstall --no-build-isolation --no-deps -e . )
  fi
  [ "$(git -C "$EXL3" rev-parse HEAD)" = "$EXL3_PIN" ] \
    && git -C "$EXL3" diff --quiet \
    && git -C "$EXL3" diff --cached --quiet || {
      echo "exllamav3 tracked source changed during its build" >&2
      exit 1
    }
  validate_exl3_import | tee "$RCPT/exllamav3-build.txt"
  probe | tee "$RCPT/pipeline-import.txt"
else
  cat "$RCPT/pipeline-import.txt"
  log "exllamav3 NOT built: the measurement path does not import it"
  echo "not-built: pipeline imports without loading exllamav3" > "$RCPT/exllamav3-build.txt"
fi
"$PY" -m pip check | tee "$RCPT/pip-check.txt"
"$PY" -m pip list --format=freeze \
  | LC_ALL=C sort \
  | tee "$RCPT/resolver-selected-wheel-versions.txt"

# ---- INSTALL/CHECK SPLIT (container build) --------------------------------
# Steps 1-4 INSTALL; steps 5-6 CHECK.  `container/Dockerfile` runs this script
# at image build time with FIDELITY_BOOTSTRAP_INSTALL_ONLY=1 so apt/pip/git work
# is baked into a layer, then runs the SAME script unset at container start.
# Direct wheels no-op only after exact distribution/CUDA validation; source
# checkouts are deliberately reconstructed so dirty or ignored residue cannot
# cross setup runs.
#
# The split is not tidiness.  The gguf battery's rung 1b re-decodes the
# committed real bytes on THIS BOX'S CUDA device and demands torch.equal
# against the reference; a docker builder has no GPU, so running it there would
# either fail a good build or, worse, pass vacuously and retire a check that
# only means something on the measuring machine.  Same reasoning for the other
# three adapter batteries: they are pre-flight, and pre-flight belongs to the
# flight.
#
# Unset (every existing caller) continues into the runtime-only checks below.
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
