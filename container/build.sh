#!/usr/bin/env bash
# Build the measurement image, with the pins the receipt will have to name.
#
#   container/build.sh [--tag NAME:TAG] [--engine docker|podman] [-- <extra args>]
#
# The one thing this adds over a raw `docker build` is that it refuses to build
# an image whose receipts could not name their own code.  `produced_by` needs a
# git revision and the schema has no "unknown" value for it -- deliberately,
# because a receipt that cannot say which code produced it is not reproducible.
# On an SSH-driven instance that block has to be computed on the caller's
# laptop and shipped in job.json; an image can carry it, but only if the build
# was given the revision.  So the revision is a build argument and a dirty tree
# is a REFUSAL, not a warning: an image built from uncommitted edits would
# stamp every receipt with a commit that does not describe its own bytes.
#
# NEVER `set -x`: nothing here handles a token, and it must stay that way.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="quant-fidelity-measure:dev"
ENGINE=""
ALLOW_DIRTY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$ENGINE" ]; then
  for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1; then ENGINE="$candidate"; break; fi
  done
fi
[ -n "$ENGINE" ] || { echo "no docker or podman on PATH" >&2; exit 3; }

REV="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
[ -n "$REV" ] || { echo "no git checkout at $ROOT: the image could not record
  the revision its receipts must name. Build from a checkout." >&2; exit 3; }

if [ "$ALLOW_DIRTY" = 0 ]; then
  if ! git -C "$ROOT" diff --quiet || ! git -C "$ROOT" diff --cached --quiet; then
    echo "REFUSED: the checkout is dirty, so revision $REV does not describe" >&2
    echo "  the bytes this image would contain -- and every receipt it seals" >&2
    echo "  would cite it anyway." >&2
    git -C "$ROOT" status --porcelain | head -20 >&2
    echo "  Commit, or pass --allow-dirty for a throwaway build." >&2
    exit 3
  fi
fi

echo "engine   $ENGINE"
echo "tag      $TAG"
echo "revision $REV"
exec "$ENGINE" build -f "$ROOT/container/Dockerfile" \
  --build-arg "SUITE_REVISION=$REV" \
  --build-arg "IMAGE_REFERENCE=$TAG" \
  -t "$TAG" "${EXTRA[@]}" "$ROOT"
