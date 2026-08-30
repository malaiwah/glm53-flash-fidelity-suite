#!/bin/bash
# Parallel fetch of a pinned HF file list into a local-dir layout.
#
#   par_fetch.sh <list-file> <repo> <revision> <dest-root> <parallelism>
#
# The k6 payload store is content-addressed: 61,711 object filenames ARE the
# sha256 of their bytes, so verification is free and is done here.  `hf
# download` plans all 98,878 files before writing one byte, which stalls; this
# fetches each path directly from the resolve endpoint with N parallel curls.
#
# Token comes from $HF_TOKEN in the environment (never an argv, never echoed).
set -uo pipefail

LIST="${1:?list}"; REPO="${2:?repo}"; REV="${3:?revision}"; DEST="${4:?dest}"; PAR="${5:-48}"

: "${HF_TOKEN:?HF_TOKEN must be exported}"
mkdir -p "$DEST"

# One curl per path; skip files already present with non-zero size (resumable).
fetch_one() {
  rel="$1"
  out="$DEST/$rel"
  if [ -s "$out" ]; then return 0; fi
  mkdir -p "$(dirname "$out")"
  attempt=0
  while [ $attempt -lt 6 ]; do
    code=$(curl -sL --fail-with-body -m 900 --retry 4 --retry-delay 2 --retry-all-errors \
          -H "Authorization: Bearer $HF_TOKEN" \
          -o "$out.part" -w "%{http_code}" \
          "https://huggingface.co/datasets/$REPO/resolve/$REV/$rel" 2>/dev/null)
    [ "$code" = "200" ] && break
    rm -f "$out.part"
    attempt=$((attempt+1))
    # 429 = rate limit: back off with jitter rather than hammering
    sleep $(( attempt * 3 + (RANDOM % 4) ))
  done
  if [ "$code" != "200" ]; then
    rm -f "$out.part"
    echo "FAIL $code $rel" >&2
    return 1
  fi
  mv -f "$out.part" "$out"
  return 0
}
export -f fetch_one
export DEST REPO REV HF_TOKEN

xargs -a "$LIST" -P "$PAR" -I{} bash -c 'fetch_one "$@"' _ {}
rc=$?
echo "xargs rc=$rc"
exit $rc
