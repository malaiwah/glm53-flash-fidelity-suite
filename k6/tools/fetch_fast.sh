#!/bin/bash
# Fast packed-store fetch, then hand off to the published stage script for the
# teacher + BF16 legs.
#
# WHY THIS EXISTS (disclose in the receipt): the K6 payload store is 98,878
# files (61,711 content-addressed .bin objects + 37,152 choice .json).  `hf
# download` plans every file before writing one byte and stalled >20 min at
# ~18 KB/s of listing traffic.  This fetches the pinned path list directly from
# the resolve endpoint with N parallel curls -- same bytes, same revision pin.
set -uo pipefail

HR_ROOT=/home/hr
PACKED="$HR_ROOT/packed/k6"
DONE="$HR_ROOT/receipts/done"
REPO=malaiwah/GLM-5.3-Flash-TR3-partsbin-v1
REV=9a4ec0b6f4a21d6a76ab05f98360857a32d16927
PAR_OBJ="${PAR_OBJ:-64}"
PAR_SMALL="${PAR_SMALL:-12}"

export HF_TOKEN="$(cat $HR_ROOT/.secrets/hf_token)"
mkdir -p "$DONE"

# Two passes with different parallelism: the 61,711 .bin objects are ~4 MB and
# bandwidth-bound (high P is fine); the 37,167 small .json files are
# request-rate-bound and drew HTTP 429s at P=96, so they get a gentle P.
echo "=== packed fetch: objects (P=$PAR_OBJ) ==="
t0=$(date +%s)
bash "$HR_ROOT/par_fetch.sh" "$HR_ROOT/list_objects.txt" "$REPO" "$REV" \
     "$HR_ROOT/packed" "$PAR_OBJ" 2>"$HR_ROOT/logs/par_objects.err"
echo "objects rc=$? elapsed=$(( $(date +%s) - t0 ))s fails=$(grep -c '^FAIL' "$HR_ROOT/logs/par_objects.err" 2>/dev/null || echo 0)"

echo "=== packed fetch: small files (P=$PAR_SMALL) ==="
t1=$(date +%s)
bash "$HR_ROOT/par_fetch.sh" "$HR_ROOT/list_small.txt" "$REPO" "$REV" \
     "$HR_ROOT/packed" "$PAR_SMALL" 2>"$HR_ROOT/logs/par_small.err"
echo "small rc=$? elapsed=$(( $(date +%s) - t1 ))s fails=$(grep -c '^FAIL' "$HR_ROOT/logs/par_small.err" 2>/dev/null || echo 0)"

# Retry whatever still failed, gently, from both passes.
cat "$HR_ROOT/logs/par_objects.err" "$HR_ROOT/logs/par_small.err" 2>/dev/null \
  | awk '/^FAIL/{print $3}' | sort -u > "$HR_ROOT/retry.txt"
NRETRY=$(wc -l < "$HR_ROOT/retry.txt")
if [ "$NRETRY" -gt 0 ]; then
  echo "=== retrying $NRETRY paths (P=8) ==="
  bash "$HR_ROOT/par_fetch.sh" "$HR_ROOT/retry.txt" "$REPO" "$REV" \
       "$HR_ROOT/packed" 8 2>"$HR_ROOT/logs/par_retry.err"
  echo "retry rc=$? remaining=$(grep -c '^FAIL' "$HR_ROOT/logs/par_retry.err" 2>/dev/null || echo 0)"
fi
echo "total packed elapsed=$(( $(date +%s) - t0 ))s"

# ---- completeness: every pinned path present and non-empty -----------------
echo "=== completeness check ==="
python3 - <<'PY'
import os
root = "/home/hr/packed"
missing = []
empty = []
with open("/home/hr/partsbin_files.txt") as fh:
    paths = [line.strip() for line in fh if line.strip()]
for rel in paths:
    p = os.path.join(root, rel)
    try:
        if os.path.getsize(p) == 0:
            empty.append(rel)
    except OSError:
        missing.append(rel)
print(f"pinned={len(paths)} missing={len(missing)} empty={len(empty)}")
for rel in (missing + empty)[:10]:
    print("  MISSING/EMPTY", rel)
open("/home/hr/receipts/packed-completeness.txt", "w").write(
    f"pinned={len(paths)} missing={len(missing)} empty={len(empty)}\n")
raise SystemExit(1 if (missing or empty) else 0)
PY
[ $? -eq 0 ] || { echo "packed fetch INCOMPLETE -- refusing to continue" >&2; exit 1; }

# ---- integrity: the object filename IS the sha256 of its bytes -------------
# Full verification of all 61,711 objects (254 GB of hashing) is not free, so
# verify a deterministic pseudo-random sample plus every non-object file's
# presence.  Content is additionally gated downstream by the loader's exact-load
# report and by path A reproducing the sealed panel number bitwise.
echo "=== content-address verification (sample) ==="
python3 - <<'PY'
import hashlib, os, random
root = "/home/hr/packed"
objs = []
with open("/home/hr/partsbin_files.txt") as fh:
    for line in fh:
        rel = line.strip()
        if "/payload-store/objects/" in rel and rel.endswith(".bin"):
            objs.append(rel)
random.Random(20260829).shuffle(objs)
sample = objs[:400]
bad = []
total = 0
for rel in sample:
    want = os.path.basename(rel)[:-4]
    h = hashlib.sha256()
    with open(os.path.join(root, rel), "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
            total += len(blk)
    if h.hexdigest() != want:
        bad.append(rel)
print(f"objects_total={len(objs)} sampled={len(sample)} bytes_hashed={total} mismatches={len(bad)}")
for rel in bad[:5]:
    print("  SHA MISMATCH", rel)
open("/home/hr/receipts/packed-content-verify.txt", "w").write(
    f"objects_total={len(objs)} sampled={len(sample)} bytes_hashed={total} mismatches={len(bad)}\n")
raise SystemExit(1 if bad else 0)
PY
[ $? -eq 0 ] || { echo "content-address verification FAILED" >&2; exit 1; }

# ---- required top-level files ---------------------------------------------
for need in contract.json inventory.json mtp-adapter-receipt.json; do
  test -f "$PACKED/$need" || { echo "packed fetch incomplete: $need missing" >&2; exit 1; }
done
test -d "$PACKED/payload-store/objects" || { echo "payload-store/objects missing" >&2; exit 1; }
test -d "$PACKED/payload-store/choices" || { echo "payload-store/choices missing" >&2; exit 1; }

du -sh "$PACKED"
touch "$DONE/fetch-packed.done"
echo "=== packed done; handing off to stage script for teacher + bf16 ==="

exec bash /home/suite/k6/tools/hidden_replay_stage.sh fetch
