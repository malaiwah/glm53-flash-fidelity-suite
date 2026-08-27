#!/bin/sh
# overlap_smoke.sh - single-unit control-vs-overlap measurement for
# k6_driver.py encode-worker --overlap-seal, on the K8 campaign (out-k8).
#
# Runs ON the box (jl machine 484853), e.g.:
#   jl upload 484853 tools/overlap_smoke.sh /home/jl_fs/glm53-k6/tools/overlap_smoke.sh
#   jl run --on 484853 --json --yes -- sh /home/jl_fs/glm53-k6/tools/overlap_smoke.sh
#
# Protocol (operator-approved):
#   1. GATE: refuses to run unless out-k8 has contract.json, work state, at
#      least 2 sealed layer preparations, and at least 2 pending work units.
#      (r_c0cdabb9 is building contract/prep - this script never interferes,
#      it only refuses until that work exists.)
#   2. CONTROL FIRST: one nice-19 encode-worker, --max-units 1, NO overlap.
#   3. OVERLAP NEXT: same worker, --max-units 1, --overlap-seal.
#      Both units are real K8 campaign work banked in out-k8 - nothing wasted.
#   4. VERIFY: every expert receipt of both smoke layers re-verified with the
#      pipeline's own verify_expert_receipt (this includes re-hashing payload
#      choices and the routed Hessian artifacts).
#   5. Reports the two "encoded in Xs" walls and the relative gain.
#
# Politeness: nice -n 19 everywhere, ONE worker, one GPU (default: physical
# GPU 3; override with SMOKE_GPU). Does NOT pass
# --prune-hessians-after-layer-seal so step 4 can re-verify artifacts; the
# two smoke layers' Hessians stay on disk for the orchestrator to prune after
# its own inspection (same one-liner the campaign uses).

set -eu

ROOT=${ROOT:-/home/jl_fs/glm53-k6}
DRIVER=${DRIVER:-$ROOT/tools/k6_driver.py}
PY=${PY:-$ROOT/venv/bin/python}
OUT=$ROOT/out-k8
SMOKE_GPU=${SMOKE_GPU:-3}
WORKER=${WORKER:-k8-smoke-0}
LOGDIR=$ROOT/logs/overlap-smoke
mkdir -p "$LOGDIR"

fail() { echo "GATE/FAIL: $*" >&2; exit 3; }

# --- gate 0: driver must carry the flag (orchestrator sync happened) --------
"$PY" "$DRIVER" encode-worker --help 2>/dev/null | grep -q -- "--overlap-seal" \
  || fail "$DRIVER does not know --overlap-seal (sync the updated driver first)"

# --- gate 1: out-k8 contract + prep + state readiness -----------------------
[ -f "$OUT/contract.json" ] || fail "out-k8/contract.json absent (r_c0cdabb9 not done)"
[ -f "$OUT/state/work-state.json" ] || fail "out-k8/state/work-state.json absent (contract step not finished)"
PREPARED=$(ls "$OUT"/preparation/layer-*/preparation.json 2>/dev/null | wc -l)
[ "$PREPARED" -ge 2 ] || fail "only $PREPARED sealed layer preparations (need >= 2)"

# the first TWO pending units (the ones control+overlap will claim, in order)
# must belong to layers whose preparation is sealed.
nice -n 19 "$PY" - "$OUT" <<'EOF' || fail "first two pending units not encodable yet"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
state = json.loads((out / "state" / "work-state.json").read_text())
pending = state.get("pending", [])
assert len(pending) >= 2, f"only {len(pending)} pending units (need >= 2)"
units = state["units"]
for sha in pending[:2]:
    layer = int(units[sha]["layer"])
    prep = out / "preparation" / f"layer-{layer:03d}" / "preparation.json"
    assert prep.is_file(), f"unit {sha[:16]} needs layer {layer} preparation (not sealed yet)"
    print(f"gate: pending unit {sha[:16]} -> layer {layer} (preparation sealed)")
EOF
echo "gate ok: $PREPARED prepared layers, GPU $SMOKE_GPU"

run_worker() {
    _tag=$1; shift
    _log=$LOGDIR/$_tag.log
    echo "=== $_tag: start $(date -u +%FT%TZ) ==="
    CUDA_VISIBLE_DEVICES=$SMOKE_GPU nice -n 19 "$PY" "$DRIVER" encode-worker \
        --profile k8 \
        --worker "$WORKER" \
        --pipeline-root "$ROOT/pipeline-k8" \
        --shapley-root "$ROOT/shapleymcg" \
        --exllama-root "$ROOT/exllamav3" \
        --bf16 /home/jl_fs/models/bf16 \
        --calibration "$ROOT/calibration" \
        --output-root "$OUT" \
        --max-units 1 \
        "$@" >"$_log" 2>&1 || { cat "$_log"; fail "$_tag worker failed (log: $_log)"; }
    cat "$_log"
    echo "=== $_tag: end $(date -u +%FT%TZ) ==="
}

# --- 2: control first (real work banked) ------------------------------------
run_worker control
CONTROL_LINE=$(grep "encoded in" "$LOGDIR/control.log" | tail -1)
CONTROL_LAYER=$(echo "$CONTROL_LINE" | sed -n 's/.*layer \([0-9]*\) encoded.*/\1/p')
[ -n "$CONTROL_LAYER" ] || fail "control run encoded nothing (claimed unit already encoded? see $LOGDIR/control.log)"

# --- 3: overlap on the NEXT unit (real work banked) -------------------------
run_worker overlap --overlap-seal
OVERLAP_LINE=$(grep "encoded in" "$LOGDIR/overlap.log" | tail -1)
OVERLAP_LAYER=$(echo "$OVERLAP_LINE" | sed -n 's/.*layer \([0-9]*\) encoded.*/\1/p')
[ -n "$OVERLAP_LAYER" ] || fail "overlap run encoded nothing (claimed unit already encoded? see $LOGDIR/overlap.log)"

# --- 4: pipeline-native receipt verification over BOTH smoke layers ---------
nice -n 19 "$PY" - "$ROOT" "$OUT" "$CONTROL_LAYER" "$OVERLAP_LAYER" <<'EOF'
import json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
layers = [int(x) for x in sys.argv[3:5]]
sys.path.insert(0, str(root / "pipeline-k8" / "src"))
from quant_pipeline.campaign import glm53_direct_k4 as direct
contract = json.loads((out / "contract.json").read_text())
contract_sha = direct.verify_contract(contract)
bits = int(contract["rate"]["bits"])
for layer in layers:
    receipts = sorted(out.glob(f"experts/layer-{layer:03d}/expert-*.json"))
    assert len(receipts) == direct.NUM_EXPERTS, (layer, len(receipts))
    for path in receipts:
        direct.verify_expert_receipt(out, path, contract_sha256=contract_sha, expected_bits=bits)
    print(f"layer {layer}: {len(receipts)} expert receipts verify (pipeline verify_expert_receipt)")
EOF

# --- 5: honest report --------------------------------------------------------
CONTROL_S=$(echo "$CONTROL_LINE" | sed -n 's/.*encoded in \([0-9]*\)s.*/\1/p')
OVERLAP_S=$(echo "$OVERLAP_LINE" | sed -n 's/.*encoded in \([0-9]*\)s.*/\1/p')
echo "RESULT control : layer $CONTROL_LAYER  ${CONTROL_S}s   ($CONTROL_LINE)"
echo "RESULT overlap : layer $OVERLAP_LAYER  ${OVERLAP_S}s   ($OVERLAP_LINE)"
awk -v c="$CONTROL_S" -v o="$OVERLAP_S" 'BEGIN {
    if (c > 0 && o > 0)
        printf "RESULT gain    : %.1f%% wall reduction -> %.1f%% throughput gain (target >= 20%%)\n",
               100*(c-o)/c, 100*(c/o - 1);
    else print "RESULT gain    : could not parse timings - inspect logs in '"$LOGDIR"'";
}'
echo "note: smoke layers $CONTROL_LAYER and $OVERLAP_LAYER were NOT hessian-pruned (verification needs the artifacts);"
echo "note: prune after inspection with the campaign prune helper if desired."
echo "note: identical-cost basis: every routed layer is 288 experts x identical projection shapes."
