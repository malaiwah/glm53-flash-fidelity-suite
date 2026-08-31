#!/bin/bash
# Pull the small receipt artifacts off the box into a local dir for receipt
# assembly.  The bulk artifacts (logits, hiddens) STAY on the box for the
# verifier -- only JSON/text receipts come down.
set -euo pipefail
BOX="${1:-486679}"
DEST="${2:?dest dir}"
mkdir -p "$DEST"

# Bundle on the box first: one download beats 15 round trips.
jl exec "$BOX" -- bash -lc '
set -e
D=/home/hr/pull; rm -rf $D; mkdir -p $D
cp -f /home/hr/receipts/hidden-replay-comparator.json  $D/ 2>/dev/null || true
cp -f /home/hr/receipts/reproduction-check.json        $D/ 2>/dev/null || true
cp -f /home/hr/receipts/stream-k6-kld-3run.json        $D/ 2>/dev/null || true
cp -f /home/hr/receipts/nonrouted-sparse-fetch.json    $D/ 2>/dev/null || true
cp -f /home/hr/receipts/hidden-replay-selftest.json    $D/ 2>/dev/null || true
cp -f /home/hr/receipts/env-versions.txt               $D/ 2>/dev/null || true
cp -f /home/hr/receipts/nvidia-smi.txt                 $D/ 2>/dev/null || true
cp -f /home/hr/receipts/packed-completeness.txt        $D/ 2>/dev/null || true
cp -f /home/hr/receipts/packed-content-verify.txt      $D/ 2>/dev/null || true
cp -f /home/hr/receipts/patches-applied.txt            $D/ 2>/dev/null || true
cp -f /home/hr/receipts/stream-selftest.json           $D/ 2>/dev/null || true
for n in 1 2 3; do
  R=/home/hr/runs/hidden-run$n
  [ -f $R/hidden-capture.json ]   && cp -f $R/hidden-capture.json   $D/hidden-capture-run$n.json
  [ -f $R/capture-receipt.json ]  && cp -f $R/capture-receipt.json  $D/capture-receipt-run$n.json
  [ -f $R/backend.json ]          && cp -f $R/backend.json          $D/backend-run$n.json
  [ -f $R/kld-report.json ]       && cp -f $R/kld-report.json       $D/kld-report-run$n.json
  [ -f $R/reader-identity.json ]  && cp -f $R/reader-identity.json  $D/reader-identity-run$n.json
  [ -f $R/plan.json ]             && cp -f $R/plan.json             $D/plan-run$n.json
done
tar czf /home/hr/pull.tar.gz -C /home/hr pull
ls -la /home/hr/pull.tar.gz
'
jl download "$BOX" /home/hr/pull.tar.gz "$DEST/pull.tar.gz"
tar xzf "$DEST/pull.tar.gz" -C "$DEST" --strip-components=1
echo "--- pulled into $DEST ---"
ls -la "$DEST"
