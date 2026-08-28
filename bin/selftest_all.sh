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

pass=0; fail=0
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

MODEL=brandonmusic/GLM-5.3-Flash-tr3-4bpw
PANEL=brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits

echo "== selftests (offline) =="
t "fit estimator, 33 known-answer checks"  0 "$PY" bin/selftest_fit.py
t "decode parity + timing (needs torch)"   0 "$PY" bin/selftest_decode_parity.py

echo "== cloud planner (NETWORK, ACCOUNT; --dry-run creates nothing) =="
t "sealed-ep8 plan passes all checks" 0 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --i-accept-leak-risk \
    --dry-run --out "$TMP/c1"
t "streaming refuses: engine unpinned" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane streaming --spot --max-runtime 12h --i-accept-leak-risk \
    --dry-run --out "$TMP/c2"
t "refuses without a teardown backstop" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --dry-run --out "$TMP/c3"
t "refuses a max-runtime shorter than the work" 3 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 2h --i-accept-leak-risk \
    --dry-run --out "$TMP/c4"

echo "== local planner (NETWORK) =="
t "this machine, auto device"        3 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --estimate-only --out "$TMP/l1"
t "RTX 5090 32GB honours a 30GB budget" 3 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "RTX 5090:32" --vram-budget 30 --estimate-only --out "$TMP/l2"
t "128GB Mac fits"                   3 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "Mac Studio:128::unified" --estimate-only --out "$TMP/l3"
t "4GB card is REFUSED"              3 "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" --simulate-device "GTX 1650:4" --out "$TMP/l4"
t "--kld-device mps is refused (no fp64 on MPS)" 3 "$PY" bin/measure_local.py --artifact x/y --panel z --kld-device mps
t "engine probe"                     0 "$PY" bin/measure_local.py --probe-engines

echo "== teardown backstop (ACCOUNT, read-only) =="
t "reaper --list"  0 "$PY" bin/measure_cloud.py reaper --list
t "reaper --sweep" 0 "$PY" bin/measure_cloud.py reaper --sweep

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

echo
echo "selftest_all: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
