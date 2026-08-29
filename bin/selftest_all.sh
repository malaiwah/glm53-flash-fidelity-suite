#!/usr/bin/env bash
# Everything that can be verified without spending a cent or touching a GPU.
#
#   bin/selftest_all.sh
#
# Cases marked NETWORK need Hugging Face reachable (a few hundred KB of
# metadata); cases marked ACCOUNT need the `jl` CLI authenticated but only
# ever issue read-only queries. Nothing here creates an instance, downloads a
# checkpoint, or publishes anything.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${FIDELITY_PYTHON:-python3}"
VPY="$ROOT/.venv/bin/python"
[ -x "$VPY" ] || VPY="$PY"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0; skip=0
t() {  # t <name> <expected_rc> <cmd...>
  local name="$1" exp="$2"; shift 2
  "$@" >"$TMP/out.log" 2>&1; local rc=$?
  if [ "$rc" = "$exp" ]; then
    printf '  PASS  %s\n' "$name"; pass=$((pass+1))
  else
    printf '  FAIL  %s (rc=%s, expected %s)\n' "$name" "$rc" "$exp"
    sed 's/^/         /' "$TMP/out.log" | tail -6
    fail=$((fail+1))
  fi
}
s() {  # s <name> <reason> -- a prerequisite is absent; SKIP is a verdict,
       # printed with the missing thing, never silently dropped
  printf '  SKIP  %s (%s)\n' "$1" "$2"; skip=$((skip+1))
}
have_module() {  # have_module <python> <module>
  "$1" -c "import $2" >/dev/null 2>&1
}

MODEL=brandonmusic/GLM-5.3-Flash-tr3-4bpw
PANEL=brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits

echo "== selftests (offline) =="
t "fit estimator, 41 known-answer checks"  0 "$PY" bin/selftest_fit.py
t "decode parity + timing (needs torch)"   0 "$PY" bin/selftest_decode_parity.py
t "registry client/viewer/matcher (T1)"    0 python3 bin/selftest_registry_view.py
t "floor-aware stats known answers (T2)"   0 python3 bin/selftest_stats.py
t "preview estimator coverage (T3)"        0 python3 bin/selftest_preview_stats.py
t "submission refusability (T5)"           0 python3 bin/selftest_submission_refusal.py
t "stack fingerprint (T9: deterministic, engine-absent, MPS/CUDA-absent)" \
                                           0 python3 bin/selftest_stackprint.py
t "zero-floor identity (T4; SKIPs inside when numpy/torch absent)" \
                                           0 "$PY" bin/selftest_zero_floor.py
t "stream_score ladder rungs g,h,i,j (teacher role / preview refusal / \
sampling / receipt stability)"             0 python3 k6/tools/stream_score_selftest.py --only g,h,i,j
# The three community-quant weight-decode surfaces.  Each proves its dequant
# against that ecosystem's own reference implementation on REAL ranged-fetched
# tensors (replayed here from committed fixtures so the proof runs with no
# network and on boxes where the reference library cannot be installed),
# censuses the real artifact's tensor names against the official BF16 set, and
# exercises every refusal.  No GPU, no rental, no weights.
#
# mlx needs torch+safetensors, so it runs under FIDELITY_PYTHON like the
# fixture ladder; gguf and nvfp4 run under $PY.
MLXPY="${FIDELITY_PYTHON:-/opt/homebrew/bin/python3.14}"
[ -x "$MLXPY" ] || MLXPY="$PY"
if have_module "$MLXPY" torch && have_module "$MLXPY" safetensors; then
  t "mlx surface offline (8 rungs: mlx equality, census, refusals, plumbing, \
dry-runs, registry adapter)"               0 "$MLXPY" k6/tools/selftest_mlx_offline.py
else
  s "mlx surface offline" "torch/safetensors not importable under $MLXPY -- export FIDELITY_PYTHON"
fi
if have_module "$PY" torch; then
  t "gguf surface offline (dequant vs gguf-py, census, MLA audit, refusals)" \
                                           0 "$PY" k6/tools/selftest_gguf_offline.py
else
  s "gguf surface offline" "torch not importable by $PY"
fi

echo "== cloud planner (NETWORK, ACCOUNT; --dry-run creates nothing) =="
# This target is 'tr3-published' and NO lane has a reader for it: both engines
# resolve a packed_root out of the materialization receipt and require the
# payload store to be present, which a third-party repo never publishes. The
# check that was supposed to catch this (hfmeta.sniff_surface's packed_root
# trap) is guarded by `if info.surface == "packed"`, and this repo carries
# exl3-mcg-storage-abi.json, so it classifies as tr3-published and routes
# around the trap. Asserting rc=0 here asserted that a rental which cannot
# possibly succeed would be approved.
# --skip-registry-check everywhere below: these cases test the PLANNER's own
# refusals; the registry front gate (tested separately) would otherwise answer
# "already measured" first, because this target has published rows.
t "sealed-ep8 refuses: no reader for tr3-published" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --i-accept-leak-risk \
    --skip-registry-check --dry-run --out "$TMP/c1"
t "streaming refuses: no reader for tr3-published (lane now PINNED)" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane streaming --spot --max-runtime 12h --i-accept-leak-risk \
    --skip-registry-check --dry-run --out "$TMP/c2"
t "refuses without a teardown backstop" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --skip-registry-check \
    --dry-run --out "$TMP/c3"
t "refuses a max-runtime shorter than the work" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 2h --i-accept-leak-risk \
    --skip-registry-check --dry-run --out "$TMP/c4"
t "cloud front gate: already-measured artifact answers for \$0.00" 0 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --i-accept-leak-risk \
    --dry-run --out "$TMP/c5"

echo "== local planner (NETWORK) =="
# rc expectations CHANGED 2026-08-29: the streaming/local lanes are now PINNED,
# so a clean --estimate-only plan exits 0 (it used to exit 3 on "engine
# unpinned"). --skip-registry-check isolates the planner from the front gate.
t "this machine, auto device (clean plan now exits 0)" 0 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --estimate-only --skip-registry-check --out "$TMP/l1"
t "RTX 5090 32GB honours a 30GB budget" 0 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "RTX 5090:32" --vram-budget 30 --estimate-only --skip-registry-check --out "$TMP/l2"
t "128GB Mac fits"                   0 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "Mac Studio:128::unified" --estimate-only --skip-registry-check --out "$TMP/l3"
t "4GB card is REFUSED"              3 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "GTX 1650:4" --skip-registry-check --out "$TMP/l4"
t "--kld-device mps is refused (no fp64 on MPS)" 3 "$PY" bin/measure_local.py --artifact x/y --panel z --kld-device mps
t "engine probe (all five lanes pinned, flags found)" 0 "$PY" bin/measure_local.py --probe-engines
t "--execute preflight-refuses with remedies (no traceback)" 3 \
  "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" \
    --skip-registry-check --execute --work "$TMP/lw" --out "$TMP/l7"

echo "== registry front gate + one-command (NETWORK) =="
t "measure-local gate: already-measured exits 0" 0 \
  "$PY" bin/measure_local.py --artifact malaiwah/GLM-5.3-Flash-TR3-6bpw \
    --panel "$PANEL" --estimate-only --out "$TMP/g1"
t "registry-view check (live TR3-6bpw: sealed + streaming rows)" 0 \
  bin/registry-view check malaiwah/GLM-5.3-Flash-TR3-6bpw
t "registry-view rows (local clone, streaming lane)" 0 \
  bin/registry-view rows --model glm --lane streaming --registry local
t "bin/measure: already-measured report, exit 0" 0 \
  bin/measure malaiwah/GLM-5.3-Flash-TR3-6bpw
t "registry live selftest (T8: snapshot, keys, tripwire)" 0 \
  bin/registry-view --selftest-live

echo "== teardown backstop (ACCOUNT, read-only) =="
# `reaper --list` is lease-file-driven and safe anywhere. The sweep runs
# with --dry-run here: it reports what WOULD be destroyed and destroys
# nothing -- destruction must never be a side effect of "run the selftests"
# (JOURNAL lesson 22; usability review 2026-08-28). A real sweep is
# `bin/measure-cloud reaper --sweep`, run deliberately. Machines without
# the jl CLI (any Mac that has never rented) SKIP the sweep instead of
# failing. SELFTEST_SKIP_ACCOUNT=1 still skips the whole section.
if [ -n "${SELFTEST_SKIP_ACCOUNT:-}" ]; then
  s "reaper --list" "SELFTEST_SKIP_ACCOUNT set (concurrent rental on this account)"
  s "reaper --sweep (dry-run)" "SELFTEST_SKIP_ACCOUNT set"
else
  t "reaper --list"  0 "$PY" bin/measure_cloud.py reaper --list
  if command -v jl >/dev/null 2>&1; then
    t "reaper --sweep (dry-run: reports, destroys nothing)" 0 \
      "$PY" bin/measure_cloud.py reaper --sweep --dry-run
  else
    s "reaper --sweep (dry-run)" "the jl CLI is not on PATH (this machine has never rented) -- uv tool install jarvislabs, or ignore: cloud teardown is irrelevant locally"
  fi
fi

echo "== registry =="
t "offline selftest"        0 "$VPY" registry/tools/registry_validate.py --root registry --offline-selftest
t "strict (2 = warnings only)" 2 "$VPY" registry/tools/registry_validate.py --root registry --strict
t "registry's own selftest" 0 "$VPY" registry/tools/registry_selftest.py
( cd registry && "$VPY" tools/registry_validate.py --submission docs/examples/dione-q4.submission.json ) >"$TMP/we.log" 2>&1
if grep -q '^ACCEPTED' "$TMP/we.log"; then
  echo "  PASS  worked example validates"; pass=$((pass+1))
else
  echo "  FAIL  worked example"; tail -5 "$TMP/we.log" | sed 's/^/         /'; fail=$((fail+1))
fi

echo "== receipt round trip: bundle-only filesystem, no git, no network =="
# Proves the on-instance seal path: stage exactly what BUNDLE.txt uploads into
# a bare directory, then seal and validate from inside it. This is the check
# that caught a missing bundle dependency and two null provenance fields.
"$PY" - "$TMP/fs" <<'STAGE'
import pathlib, shutil, sys
root = pathlib.Path(__file__).resolve().parent if False else pathlib.Path(".").resolve()
fs = pathlib.Path(sys.argv[1]); n = 0
for line in (root / "bin/BUNDLE.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    src = root / line
    if not src.is_file():
        continue
    dst = fs / line
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst); n += 1
print("staged %d bundle files" % n)
STAGE
"$PY" - "$TMP/fs" <<'FIXTURE'
import json, pathlib, sys
sys.path.insert(0, "bin")
from pathlib import Path
from fidelity.hfmeta import DEFAULT_PANEL
from fidelity.receipt import produced_by_block
fs = pathlib.Path(sys.argv[1])
panel = dict(DEFAULT_PANEL.to_dict(), revision="0" * 40)
job = {
    "recipe": "cloud", "lane": "streaming", "reduce_order": "fp32",
    "cold_runs": 2, "profile": "k4",
    "target": {"repo_id": "brandonmusic/GLM-5.3-Flash-tr3-4bpw",
               "revision": "61e26e1484e16d7a603f77040cda9b43cc4a31d6",
               "size_bytes": 175789306501, "codec": "exl3-mcg", "bits": 4.0,
               "container": "exl3", "precision_label": "4bpw",
               "shard_hash_verification": "full",
               "exllamav3_pin": "c5d9c657"},
    "panel": panel,
    "reference": {"reference_ref": panel["reference_ref"],
                  "teacher_receipt_sha256": panel["teacher_receipt_sha256"],
                  "teacher_backend_identity_sha256":
                      panel["teacher_backend_identity_sha256"]},
    "measurer": {"name": "selftest", "handle": "selftest", "url": None,
                 "is_artifact_author": False},
    "producer": {"name": "brandonmusic", "handle": "brandonmusic",
                 "url": "https://huggingface.co/brandonmusic"},
    "environment": {"gpu": "NVIDIA H200", "gpu_count": 1, "tensor_parallel": 1,
                    "host": "selftest"},
    "produced_by": produced_by_block(Path("."), "bin/measure_cloud.py",
                                     {"lane": "streaming"}),
}
(fs / "job.json").write_text(json.dumps(job, indent=2))
(fs / "metrics.json").write_text(json.dumps({
    "metric_name": "mean_of_run_means_tokenwise_kld",
    "value": 0.0245691, "run_means": [0.0245691, 0.0245691],
    "evidence_hashes": ["4b2f0c19aa7e5d1188f3c0a94e6b7d2215ac9f83e0d47b6c1a9e2f5083c17e4d"],
    "per_run_report_sha256": [
        "c19a4b2f0c19aa7e5d1188f3c0a94e6b7d2215ac9f83e0d47b6c1a9e2f5083c1",
        "7d2215ac9f83e0d47b6c1a9e2f5083c1c19a4b2f0c19aa7e5d1188f3c0a94e6b"],
    "determinism_note": "selftest fixture; not a real measurement.",
}, indent=2))
(fs / "receipts").mkdir(exist_ok=True)
print("fixture written")
FIXTURE
"$PY" "$TMP/fs/bin/seal_receipt.py" --job "$TMP/fs/job.json" \
    --receipts "$TMP/fs/receipts" --metrics-json "$TMP/fs/metrics.json" \
    --out "$TMP/fs/receipts/measurement-receipt.json" >"$TMP/seal.log" 2>&1
if grep -q '^ACCEPTED' "$TMP/seal.log"; then
  echo "  PASS  bundle-only seal -> registry ACCEPTED"; pass=$((pass+1))
else
  echo "  FAIL  bundle-only seal"; tail -8 "$TMP/seal.log" | sed 's/^/         /'; fail=$((fail+1))
fi

echo "== fixture (NETWORK first time; torch+transformers) =="
FIXPY="${FIDELITY_PYTHON:-/opt/homebrew/bin/python3.14}"
[ -x "$FIXPY" ] || FIXPY=python3
if ! have_module "$FIXPY" torch; then
  s "fixture ladder" "torch not importable under $FIXPY -- export FIDELITY_PYTHON"
elif ! have_module "$FIXPY" transformers; then
  s "fixture ladder" "transformers not importable under $FIXPY -- \"$FIXPY\" -m pip install 'transformers>=5.16' (on Homebrew/distro Python add --break-system-packages, or use a venv and export FIDELITY_PYTHON=/path/to/venv/bin/python)"
else
  if FIXTURE_PATH="$(python3 bin/fixture_fetch.py --print 2>/dev/null)" || \
     FIXTURE_PATH="$(python3 bin/fixture_fetch.py 2>/dev/null | tail -1)"; then
    t "fixture ladder b,c,f,g,h,i,j (0.1B, whole chain)" 0 \
      "$FIXPY" k6/tools/stream_score_selftest.py --fixture "$FIXTURE_PATH" \
        --only b,c,f,g,h,i,j
  else
    s "fixture ladder" "fixture fetch failed (network?) -- run bin/fixture manually"
  fi
fi

echo
echo "selftest_all: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
