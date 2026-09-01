#!/usr/bin/env bash
# On-instance stage driver for the cloud recipe.
#
#   stage_measure.sh <setup|fetch_target|fetch_panel|measure|seal>
#
# The bootstrap is NOT reimplemented here: `setup` arranges the layout and then
# runs bin/bootstrap_measure.sh, which owns the proven container recipe --
# deadsnakes python3.12, CUDA 13.0, torch 2.11.0+cu130, transformers 5.16.1,
# the flash-attn wheel, pydantic/formatron/kbnf, and exllamav3 @ c5d9c657 with
# a rebuild guard -- and is idempotent and sudo-aware, which matters because
# containers lose apt state across a pause.
#
# It used to delegate to `engines/stage_campaign.sh setup` (then called stage_k6.sh)
# instead; see bootstrap_measure.sh's header for the two reasons that could
# never work.  These lines said otherwise for weeks while the code below
# already called bootstrap_measure.sh.
#
# Every stage writes a marker into $DONE, so a stage that already finished is a
# no-op.  That is what makes a spot preemption cost one stage instead of the
# whole run: resume, re-run setup (idempotent), and the driver skips forward.
#
# Token bytes are file-only: never put them in argv, logs, or process environments.
set -euo pipefail
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_TOKEN_PATH

STAGE="${1:?usage: stage_measure.sh <stage>}"
SCRIPT_PATH="$(readlink -f -- "$0")"
FS="$(readlink -f -- "$(dirname "$SCRIPT_PATH")/..")"
# The engine checkout is controller-provisioned outside the staged suite.  Resolve
# it once, then overwrite every ambient compatibility spelling passed to children.
ROOT="$(readlink -f -- "${FIDELITY_ENGINE_ROOT:-${FIDELITY_K6_ROOT:-/home/jl_fs/fidelity-engine}}")"
RCPT="$FS/receipts"
DONE="$RCPT/done"
LOGS="$FS/logs"
MODELS="$FS/models"
PANEL="$FS/panel"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
export VENV
export FIDELITY_FS_ROOT="$FS"
export FIDELITY_SUITE_ROOT="$FS"
export FIDELITY_ENGINE_ROOT="$ROOT"
export FIDELITY_ENGINE_PYTHON="$PY"
export QP_PIPELINE_ROOT="$ROOT/pipeline"
export BF16="$FS/models/bf16"
export TR3_BF16="$FS/models/target-bf16-materialized"
unset FIDELITY_K6_ROOT

# Config written by the controller before any stage runs.  Verify its shared
# job.v2 self-identity before creating a directory, downloading a byte, or
# starting compute: uploaded job.json is untrusted transport input.
CONF="$FS/job.json"
JOB_PREFLIGHT="$(python3 - "$CONF" "$FS/bin" "$FS" "$ROOT" <<'PYJOB'
import hashlib, os, sys

job_path, bin_root, fs_root, engine_root = sys.argv[1:]
sys.path.insert(0, bin_root)
try:
    with open(job_path, "rb") as handle:
        raw = handle.read()
    from fidelity import jobcontract
    job = jobcontract.parse_job_bytes(raw)
    jobcontract.validate_execution_job(job)
    execution = job.get("execution_attempt") or {}
    if execution.get("kind") == "runpod-ssh":
        if os.path.realpath(execution.get("remote_root", "")) \
                != os.path.realpath(fs_root):
            raise jobcontract.JobContractError(
                "execution.remote_root differs from staged suite root")
        if os.path.realpath(execution.get("engine_root", "")) \
                != os.path.realpath(engine_root):
            raise jobcontract.JobContractError(
                "execution.engine_root differs from staged engine root")
    identity = job["job_id_full"]
    print("%s:%s" % (identity, hashlib.sha256(raw).hexdigest()))
except Exception as exc:
    raise SystemExit("stage_measure: job.json self-identity REFUSED: %s" % exc)
PYJOB
)"
JOB_BINDING="${JOB_PREFLIGHT%%:*}"
JOB_SHA="${JOB_PREFLIGHT#*:}"

mkdir -p "$RCPT" "$DONE" "$LOGS" "$MODELS" "$PANEL" "$FS/.secrets"
chmod 700 "$FS/.secrets" 2>/dev/null || true
# Read a dotted path out of strict job.json using stock Python before the venv.
jqget() {  # jqget <dotted.path> [default]
  python3 -c '
import json, sys
sys.path.insert(0, sys.argv[4])
from fidelity import jobcontract
try:
    with open(sys.argv[1], "rb") as handle:
        doc = jobcontract.parse_job_bytes(handle.read())
except Exception as exc:
    raise SystemExit("stage_measure: job.json strict parse REFUSED: %s" % exc)
cur = doc
for part in sys.argv[2].split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = sys.argv[3]
        break
if cur is None:
    cur = sys.argv[3]
print(cur if not isinstance(cur, (dict, list)) else json.dumps(cur))
' "$CONF" "$1" "${2-}" "$FS/bin"
}

log() { echo "[$(date -u +%FT%TZ)] stage_measure/$STAGE: $*"; }

# --------------------------------------------------------------------------
# Atomic job+attempt-bound stage markers.
# --------------------------------------------------------------------------
validate_marker() {  # validate_marker PATH EXPECTED_STAGE
  python3 - "$1" "$JOB_BINDING" "$JOB_SHA" "$2" <<'PYMARK'
import datetime, pathlib, re, sys

path = pathlib.Path(sys.argv[1])
try:
    mode = path.lstat().st_mode
except OSError as exc:
    raise SystemExit("cannot stat marker %s: %s" % (path, exc))
import stat
if not stat.S_ISREG(mode):
    raise SystemExit("marker %s is not a regular file" % path)
expected = {
    "job_id_full": sys.argv[2],
    "job_sha256": sys.argv[3],
    "stage": sys.argv[4],
}
try:
    lines = path.read_text(encoding="utf-8").splitlines()
except OSError as exc:
    raise SystemExit("cannot read marker %s: %s" % (path, exc))
if len(lines) != 4 or any("=" not in line for line in lines):
    raise SystemExit("marker %s is torn/legacy (expected exactly four fields)" % path)
pairs = [line.split("=", 1) for line in lines]
if len({key for key, _value in pairs}) != 4:
    raise SystemExit("marker %s contains duplicate fields" % path)
doc = dict(pairs)
if set(doc) != {"job_id_full", "job_sha256", "stage", "completed_at"}:
    raise SystemExit("marker %s has missing/unexpected fields" % path)
if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                doc["completed_at"]) is None:
    raise SystemExit("marker %s has invalid completed_at" % path)
for key, value in expected.items():
    if doc.get(key) != value:
        raise SystemExit("marker %s %s mismatch" % (path, key))
try:
    datetime.datetime.strptime(doc["completed_at"], "%Y-%m-%dT%H:%M:%SZ")
except (TypeError, ValueError):
    raise SystemExit("marker %s has invalid completed_at" % path)
PYMARK
}

write_marker() {
  python3 - "$marker" "$JOB_BINDING" "$JOB_SHA" "$STAGE" <<'PYMARK'
import datetime, os, pathlib, tempfile, sys

path = pathlib.Path(sys.argv[1])
text = (
    "job_id_full=%s\njob_sha256=%s\nstage=%s\ncompleted_at=%s\n"
    % (sys.argv[2], sys.argv[3], sys.argv[4],
       datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
).encode("utf-8")
fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".stage-marker-", suffix=".tmp")
try:
    with os.fdopen(fd, "wb") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, str(path))
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except BaseException:
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    raise
PYMARK
}

marker="$DONE/$STAGE.done"
if [ -e "$marker" ]; then
  if ! validate_marker "$marker" "$STAGE"; then
    echo "stage_measure: REFUSING invalid/stale marker $marker; use a fresh run root." >&2
    exit 7
  fi
  if [ "$STAGE" != "setup" ] && [ "$STAGE" != "fetch_target" ]; then
    log "already done (marker $marker) -- skipping"
    exit 0
  fi
fi
# Every stage after setup runs under the venv setup builds.  Without this guard
# a stage launched before setup finished died as a bare `exit 127` -- "not
# found" -- which says nothing about the actual dependency.
if [ "$STAGE" != "setup" ] && [ ! -x "$PY" ]; then
  echo "stage_measure: error: $STAGE needs the venv interpreter $PY, which does not exist yet." >&2
  echo "  The setup stage builds it. Run (or finish) 'stage_measure.sh setup' first." >&2
  exit 3
fi

# --------------------------------------------------------------------------
# Atomic per-stage lock (P1-14)
#
# The controller's liveness probe can answer "unknown" (ssh flake, API blip),
# and an unknown must never authorize a second writer -- two capture
# processes interleaving receipts/run-N/ is not a crash, it is a corrupted
# measurement that looks finished.  mkdir is the atomic primitive every
# POSIX filesystem has; the owner file records who holds it.  A lock whose
# recorded pid is dead is stale (OOM, preemption) and is taken over.
# --------------------------------------------------------------------------
LOCK="$RCPT/locks/$STAGE.lock"
mkdir -p "$RCPT/locks"
write_lock_owner() {
  {
    echo "job_id_full=$JOB_BINDING"
    echo "pid=$$"
    echo "host=$(hostname 2>/dev/null || echo '?')"
    echo "started=$(date -u +%FT%TZ)"
  } > "$LOCK/owner"
}
if mkdir "$LOCK" 2>/dev/null; then
  write_lock_owner
else
  opid="$(sed -n 's/^pid=//p' "$LOCK/owner" 2>/dev/null | head -1)"
  if [ -n "$opid" ] && kill -0 "$opid" 2>/dev/null; then
    echo "stage_measure: stage $STAGE is ALREADY RUNNING here (lock $LOCK, pid $opid) -- refusing to start a second writer." >&2
    exit 8
  fi
  log "stale lock $LOCK (owner pid ${opid:-unknown} is gone) -- taking over"
  rm -rf "$LOCK"
  mkdir "$LOCK" 2>/dev/null || { echo "stage_measure: lost the lock race for $STAGE" >&2; exit 8; }
  write_lock_owner
fi
trap 'rm -rf "$LOCK"' EXIT

# Point Hugging Face clients at the controller-written token file without ever
# materializing its bytes in this shell or a process environment.
load_token() {
  local token_path="$FS/.secrets/hf_token"
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_TOKEN_PATH
  if [ ! -e "$token_path" ]; then
    [ ! -L "$token_path" ] || {
      echo "HF token file REFUSED: dangling symlink" >&2
      return 2
    }
    export HF_TOKEN_PATH="$token_path"
    return
  fi
  python3 - "$token_path" <<'PYTOKEN'
import os, stat, sys
path = sys.argv[1]
if not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("HF token file REFUSED: O_NOFOLLOW is unavailable")
flags = os.O_RDONLY | os.O_NOFOLLOW
try:
    fd = os.open(path, flags)
except OSError as exc:
    raise SystemExit("HF token file REFUSED: %s" % exc)
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("HF token file REFUSED: not a regular file")
    if info.st_uid != os.getuid():
        raise SystemExit("HF token file REFUSED: owner differs from stage uid")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("HF token file REFUSED: mode must be exactly 0600")
finally:
    os.close(fd)
PYTOKEN
  export HF_TOKEN_PATH="$token_path"
}

require_stage_marker() {  # prerequisite stage, bound to this exact job attempt
  local required="$1"
  local path="$DONE/$required.done"
  if [ ! -e "$path" ]; then
    echo "stage_measure/$STAGE REFUSES: prerequisite $required has not completed ($path absent)." >&2
    exit 3
  fi
  if ! validate_marker "$path" "$required"; then
    echo "stage_measure/$STAGE REFUSES: prerequisite $required marker is stale, torn, or unbound." >&2
    exit 7
  fi
}

require_target_census() {
  require_stage_marker fetch_target
  python3 - "$CONF" "$FS/bin" "$RCPT/fetch-target-census.json" \
      "$JOB_BINDING" "$JOB_SHA" <<'PYCENSUS'
import sys

job_path, bin_root, receipt_path, job_id, job_file_sha = sys.argv[1:]
sys.path.insert(0, bin_root)
from fidelity import common, jobcontract

with open(job_path, "rb") as handle:
    job = jobcontract.parse_job_bytes(handle.read())
try:
    with open(receipt_path, "rb") as handle:
        receipt = jobcontract.parse_job_bytes(handle.read())
except (OSError, ValueError, TypeError) as exc:
    raise SystemExit("target census REFUSED: receipt is absent/invalid: %s" % exc)
keys = {
    "schema", "receipt_sha256", "verified_at", "job_id_full",
    "job_file_sha256", "repository", "revision", "config_sha256",
    "index_sha256", "shard_manifest_sha256", "model_bytes", "shards",
    "index_shards",
}
target = job["target"]
if (not isinstance(receipt, dict)
        or set(receipt) != keys
        or receipt.get("schema") != "fidelity.fetch-target-census.v1"
        or not common.verify_seal(receipt)
        or receipt.get("job_id_full") != job_id
        or receipt.get("job_file_sha256") != job_file_sha
        or receipt.get("repository") != target["repo_id"]
        or receipt.get("revision") != target["revision"]
        or receipt.get("config_sha256") != target["config_sha256"]
        or receipt.get("index_sha256") != target["index_sha256"]
        or receipt.get("shard_manifest_sha256")
        != target["shard_manifest_sha256"]
        or receipt.get("model_bytes") != target["model_bytes"]
        or receipt.get("shards") != target["shards"]
        or receipt.get("index_shards")
        != [row["path"] for row in target["shards"]]):
    raise SystemExit("target census REFUSED: receipt/job identities differ")
PYCENSUS
}

case "$STAGE" in

setup)
  # The measurement lane owns its bootstrap (bin/bootstrap_measure.sh).  It
  # used to call engines/stage_campaign.sh, which (a) was never in the upload bundle and
  # (b) hard-stops a decode-only run on an ENCODER closure gate.  See that
  # script's header for the full reasoning.
  #
  # The official BF16 config + index are still fetched: the capture binds
  # inventory.config_sha256/index_sha256 to local files, and the exl3hf
  # materializer checks its produced non-routed name set against the official
  # index.  Both need the ORIGINAL bytes -- at the PINNED revision, not main,
  # which can move under us between two measurements of the same artifact.
  BF16_DIR="${BF16:-$FS/models/bf16}"
  # For every other surface this tree is 16 MB of config + index, and living on
  # the container's own layer is harmless. A GGUF run also stores the ~4.2 GB
  # vision-carrying shard here, and THAT is not harmless on a provider whose
  # persistent disk is a mounted volume: the stage markers live on the volume,
  # so a restarted pod would skip `setup` as done while the tree it wrote had
  # evaporated with the container. Put it beside the markers instead. Scoped to
  # gguf on purpose -- moving it for every surface would strand a run that is
  # in flight right now under the old path.
  if [ "$(jqget target.surface)" = "gguf" ]; then
    BF16_DIR="${BF16:-$FS/models/bf16}"
  fi
  BF16_REV="$(jqget official_bf16_revision a6c167b62691b2bac901344b65cb651a70f53e43)"
  mkdir -p "$BF16_DIR" "$ROOT"
  if [ ! -f "$BF16_DIR/config.json" ] || [ ! -f "$BF16_DIR/model.safetensors.index.json" ]; then
    log "fetching BF16 metadata skeleton @ $BF16_REV (config + index only, ~16 MB)"
    python3 - "$BF16_DIR" "$BF16_REV" <<'PYSKEL'
import sys, urllib.request, pathlib
root, rev = pathlib.Path(sys.argv[1]), sys.argv[2]
base = "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/%s/" % rev
for name in ("config.json", "model.safetensors.index.json"):
    dest = root / name
    if dest.exists():
        continue
    with urllib.request.urlopen(base + name, timeout=300) as r:
        dest.write_bytes(r.read())
    print("fetched", name, dest.stat().st_size, "bytes")
PYSKEL
  fi
  python3 - "$CONF" "$BF16_DIR" "$BF16_REV" <<'PYBF16'
import hashlib
import json
import pathlib
import sys

job_path, root, revision = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
job = json.loads(job_path.read_text(encoding="utf-8"))
if (job.get("profile") or {}).get("profile_id") != "tr3-6bpw":
    raise SystemExit(0)
target = job.get("target") or {}
identity = target.get("official_bf16_identity")
if (job.get("official_bf16_revision") != revision
        or not isinstance(identity, dict)):
    raise SystemExit(
        "setup REFUSED: K6 job lacks exact official BF16 metadata identity")
for filename, prefix in (
        ("config.json", "config"),
        ("model.safetensors.index.json", "index")):
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise SystemExit(
            "setup REFUSED: official BF16 %s is not a regular file" % filename)
    raw = path.read_bytes()
    expected_bytes = identity.get(prefix + "_bytes")
    expected_sha = identity.get(prefix + "_sha256")
    if (isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or len(raw) != expected_bytes
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise SystemExit(
            "setup REFUSED: official BF16 %s differs from job identity"
            % filename)
PYBF16
  # patches-v2 ships in the upload tree; the pipeline clone expects it at $ROOT.
  if [ -d "$FS/engines/patches-v2" ]; then
    mkdir -p "$ROOT/patches-v2"
    cp -f "$FS"/engines/patches-v2/* "$ROOT/patches-v2/"
  fi
  log "bootstrapping (measurement-only recipe)"
  bash "$FS/bin/bootstrap_measure.sh" 2>&1 | tee -a "$LOGS/setup.log"
  # --source gguf needs MORE of the official tree than the skeleton above and
  # far LESS than a full clone, and it needs a sealed inventory that no
  # publisher ships.
  #
  # MORE: a GGUF container carries no tokenizer and no vision tower (llama.cpp
  # ships the projector as a separate mmproj file), so gguf_surface's
  # materialized view copies the official config/tokenizer sidecars and reads
  # model.visual.* out of the official shards. On GLM-5.3-Flash all 347 visual
  # tensors live in ONE shard of 120, so this is ~4.2 GB rather than 1.4 TB --
  # computed from the index, never assumed, because a release that spreads the
  # tower over three shards must fetch three.
  #
  # LESS: every measured weight comes from the artifact. No routed expert and
  # no attention projection is read from this tree, which is exactly the scope
  # difference that makes a GGUF row not comparable to a routed-experts-only
  # one.
  #
  # And the inventory: stream_score's identity gate hashes the config.json and
  # index it actually loads against a sealed quant-pipeline.glm-release-
  # inventory.v1. zai publishes no such file, and the surfaces that have one
  # got it from their own materializer. Here it is written over the two
  # OFFICIAL files at the pinned revision, so the gate binds the bytes on this
  # disk to that commit rather than to nothing.
  if [ "$(jqget target.surface)" = "gguf" ]; then
    load_token
    mapfile -d '' -t OFFICIAL < <(python3 - "$BF16_DIR" <<'PYVIS'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
wanted = ["config.json", "generation_config.json", "processor_config.json",
          "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
index = json.loads((root / "model.safetensors.index.json").read_text())
wanted += sorted({shard for name, shard in index["weight_map"].items()
                  if name.startswith("model.visual.")})
for name in wanted:
    sys.stdout.write("--include\0" + name + "\0")
PYVIS
    )
    log "fetching the official config/tokenizer + vision-carrying shards ($((${#OFFICIAL[@]} / 2)) patterns)"
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
      "$VENV/bin/hf" download zai-org/GLM-5.3-Flash-BF16 --revision "$BF16_REV" \
        --local-dir "$BF16_DIR" --max-workers 8 "${OFFICIAL[@]}" \
        >>"$LOGS/setup.log" 2>&1
    python3 - "$BF16_DIR" "$BF16_REV" "$FS/models/bf16-inventory.json" <<'PYINV'
import hashlib, json, pathlib, sys

root, revision, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


record = {
    "schema": "quant-pipeline.glm-release-inventory.v1",
    "model_repo": "zai-org/GLM-5.3-Flash-BF16",
    "model_revision": revision,
    "seal_mode": "full-shard-sha256",
    "config_sha256": sha256(root / "config.json"),
    "index_sha256": sha256(root / "model.safetensors.index.json"),
    "shards": {p.name: sha256(p) for p in sorted(root.glob("*.safetensors"))},
    "provenance": (
        "written on the measuring instance over the OFFICIAL release files at the "
        "pinned revision, for a --source gguf run. It binds ONLY what that run reads "
        "from the official tree: config/tokenizer and the vision-carrying shards. "
        "Every measured weight is decoded from the GGUF artifact instead, so this is "
        "NOT a claim that the official weights were scored."),
}
record["inventory_sha256"] = hashlib.sha256(
    (json.dumps(record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out, "binding", record["config_sha256"][:12], record["index_sha256"][:12])
PYINV
  fi
  df -h "$FS" | tee -a "$LOGS/setup.log"
  write_marker
  log "done"
  ;;

fetch_target)
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  DEST="$MODELS/target"
  [ -n "$REPO" ] || { echo "job.json has no target.repo_id" >&2; exit 2; }
  mkdir -p "$DEST"
  # The plan binds every repository file needed by this exact execution.
  # Download only those literal paths. This prevents a mutable repository
  # listing, an added sidecar, or a multi-build GGUF shelf from changing disk
  # demand after admission. Built as a NUL-delimited bash array because these
  # names are data from job.json; the shell must never parse or eval them.
  mapfile -d '' -t TARGET_INCLUDES < <(
    python3 - "$CONF" "$FS/bin" <<'PY'
import sys

job_path, bin_root = sys.argv[1:]
sys.path.insert(0, bin_root)
from fidelity import jobcontract

with open(job_path, "rb") as handle:
    job = jobcontract.parse_job_bytes(handle.read())
jobcontract.verify_job(job)
for row in job["target"]["download_manifest"]:
    sys.stdout.write("--include\0" + row["path"] + "\0")
PY
  )
  [ "${#TARGET_INCLUDES[@]}" -gt 0 ] || {
    echo "job.json has no exact target download manifest" >&2
    exit 2
  }
  log "fetching $REPO @ $REV -> $DEST  ($((${#TARGET_INCLUDES[@]} / 2)) exact files)"
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --revision "$REV" \
      --local-dir "$DEST" --max-workers 8 "${TARGET_INCLUDES[@]}" \
      >>"$LOGS/fetch_target.log" 2>&1
  log "censusing downloaded target against the exact job contract"
  python3 - "$CONF" "$FS/bin" "$DEST" "$JOB_BINDING" "$JOB_SHA" \
      "$RCPT/fetch-target-census.json" <<'PYCENSUS'
import hashlib, json, os, pathlib, stat, sys

job_path, bin_root, root_text, expected_job, expected_job_file, out = sys.argv[1:]
sys.path.insert(0, bin_root)
from fidelity import common, jobcontract

with open(job_path, "rb") as handle:
    job_raw = handle.read()
job = jobcontract.parse_job_bytes(job_raw)
job_id = jobcontract.verify_job(job)
target = job["target"]
job_file_sha = hashlib.sha256(job_raw).hexdigest()
if job_id != expected_job or job_file_sha != expected_job_file:
    raise SystemExit("fetch_target census REFUSED: job identity drift")
root = pathlib.Path(root_text)

def safe_read(rel_text, read_content=True):
    rel = pathlib.PurePosixPath(rel_text)
    if (rel.is_absolute() or not rel.parts
            or any(part in ("", ".", "..") for part in rel.parts)):
        raise SystemExit("fetch_target census REFUSED: unsafe target path %r" % rel_text)
    flags_dir = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
    fd = os.open(str(root), flags_dir)
    try:
        for part in rel.parts[:-1]:
            child = os.open(part, flags_dir, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(
            rel.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=fd)
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise SystemExit(
                    "fetch_target census REFUSED: target is not a regular file: %s"
                    % rel_text)
            chunks = []
            if read_content:
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            return b"".join(chunks), info.st_size
        finally:
            os.close(file_fd)
    finally:
        os.close(fd)

config_raw, _config_size = safe_read("config.json")
index_raw, _index_size = safe_read("model.safetensors.index.json")
config_sha = hashlib.sha256(config_raw).hexdigest()
index_sha = hashlib.sha256(index_raw).hexdigest()
if config_sha != target["config_sha256"]:
    raise SystemExit("fetch_target census REFUSED: config SHA-256 differs from job")
if index_sha != target["index_sha256"]:
    raise SystemExit("fetch_target census REFUSED: index SHA-256 differs from job")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(
                "fetch_target census REFUSED: index contains duplicate key %r" % key)
        result[key] = value
    return result

try:
    index = json.loads(index_raw.decode("utf-8"), object_pairs_hook=unique_object)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise SystemExit("fetch_target census REFUSED: index is not strict UTF-8 JSON: %s"
                     % exc)
weight_map = index.get("weight_map") if isinstance(index, dict) else None
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("fetch_target census REFUSED: index has no non-empty weight_map")
if any(not isinstance(name, str) for name in weight_map.values()):
    raise SystemExit("fetch_target census REFUSED: index shard names are not strings")
index_shards = sorted(set(weight_map.values()))

expected_shards = target["shards"]
expected_names = [row["path"] for row in expected_shards]
if index_shards != expected_names:
    raise SystemExit(
        "fetch_target census REFUSED: missing/extra indexed shards "
        "(job=%r index=%r)" % (expected_names, index_shards))
observed = []
for row in expected_shards:
    _raw, size = safe_read(row["path"], read_content=False)
    if size != row["bytes"]:
        raise SystemExit(
            "fetch_target census REFUSED: shard size differs for %s" % row["path"])
    observed.append({"path": row["path"], "bytes": size})
if sum(row["bytes"] for row in observed) != target["model_bytes"]:
    raise SystemExit("fetch_target census REFUSED: model_bytes differs from job")
shard_sha = hashlib.sha256(json.dumps(
    observed, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
if shard_sha != target["shard_manifest_sha256"]:
    raise SystemExit("fetch_target census REFUSED: shard manifest differs from job")

discovered = []
for base, dirs, files in os.walk(str(root), topdown=True, followlinks=False):
    for name in list(dirs):
        full = os.path.join(base, name)
        if os.path.islink(full):
            raise SystemExit(
                "fetch_target census REFUSED: symlink directory in target: %s" % full)
    for name in files:
        if not name.endswith(".safetensors"):
            continue
        full = os.path.join(base, name)
        if os.path.islink(full) or not os.path.isfile(full):
            raise SystemExit(
                "fetch_target census REFUSED: non-regular shard in target: %s" % full)
        discovered.append(pathlib.Path(full).relative_to(root).as_posix())
if sorted(discovered) != index_shards:
    raise SystemExit(
        "fetch_target census REFUSED: downloaded safetensors differ from index")

receipt = common.seal({
    "schema": "fidelity.fetch-target-census.v1",
    "verified_at": common.utcnow(),
    "job_id_full": job_id,
    "job_file_sha256": job_file_sha,
    "repository": target["repo_id"],
    "revision": target["revision"],
    "config_sha256": config_sha,
    "index_sha256": index_sha,
    "shard_manifest_sha256": shard_sha,
    "model_bytes": sum(row["bytes"] for row in observed),
    "shards": observed,
    "index_shards": index_shards,
})
common.write_json(out, receipt)
PYCENSUS
  # Verify what the release seals, not what we hope: SHA256SUMS if published.
  #
  # `sha256sum -c` over the whole list is the wrong instrument for a MIRROR.
  # Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw republishes brandonmusic's weights
  # byte-for-byte but trims his 120 .materialization/shards/*.json sidecars and
  # ships its own README/LICENSE -- while copying his SHA256SUMS verbatim. So
  # `-c` reports 122 failures, all of them files that are absent or deliberately
  # different, and NONE of them a weight. Under `set -o pipefail` that non-zero
  # exit killed the stage after a 175 GB download and a full checksum pass.
  #
  # What the verification has to answer is narrower and stronger: does every
  # WEIGHT file present on disk match the digest the release published for it,
  # and is every weight file covered by the list at all? Entries for files this
  # repo does not publish are REPORTED, never silently dropped and never
  # treated as a weight failure.
  if [ -f "$DEST/SHA256SUMS" ]; then
    log "verifying published SHA256SUMS (weights fail-closed; absent sidecars reported)"
    python3 "$FS/bin/verify_published_sums.py" --root "$DEST" \
        --out "$RCPT/shard-verification.json" \
        2>&1 | tee "$RCPT/shard-verification.txt"
  else
    log "no SHA256SUMS published; recording that fact in the receipt"
    echo "no SHA256SUMS in release" > "$RCPT/shard-verification.txt"
  fi
  # A surface that can verify its release's PUBLISHED seal does it here --
  # right after the bytes land, ~10 minutes in -- not at capture time four
  # stages and three GPU-hours later. The same pass writes the artifact's own
  # scope, which seal_receipt prefers over the registry's record and over its
  # pessimistic default (M1 lesson: recording `unknown` when the producer
  # published the answer is the same failure as guessing).
  SURFACE="$(jqget target.surface)"
  if [ "$SURFACE" = "tr3-published" ]; then
    log "verifying the release's published seal (tr3)"
    "$VENV/bin/python" "$FS/engines/tools/tr3_surface.py" verify \
        --root "$DEST" --repo "$REPO" --revision "$REV" \
        --shards crosscheck --out "$RCPT/artifact-seal-verification.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    "$VENV/bin/python" "$FS/engines/tools/tr3_surface.py" scope \
        --root "$DEST" --repo "$REPO" --revision "$REV" \
        --out "$RCPT/artifact-scope.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    log "seal verified; scope written to $RCPT/artifact-scope.json"
  fi
  if [ "$SURFACE" = "dione" ]; then
    # A Dione release publishes no seal at all, so there is nothing to
    # recompute -- but it DOES publish a per-shard sha256 manifest, and the
    # only cheap moment to hash 149 GB against it is right after it lands.
    # The marker this writes is what `--dione-verify-shards full` requires at
    # capture time, four stages and three GPU-hours later.
    log "hashing every shard against the release manifest (dione)"
    "$VENV/bin/python" "$FS/engines/tools/dione_surface.py" verify-shards \
        --root "$DEST" > "$RCPT/artifact-shard-verification.json" \
        2>>"$LOGS/fetch_target.log"
    "$VENV/bin/python" "$FS/engines/tools/dione_surface.py" scope \
        --root "$DEST" --repo "$REPO" --revision "$REV" \
        --out "$RCPT/artifact-scope.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    log "shards verified; scope written to $RCPT/artifact-scope.json"
  fi
  if [ "$SURFACE" = "gguf" ]; then
    # A community GGUF publishes no seal, no encoder receipt and no per-file
    # digest list -- so the ONLY identity a receipt can claim beyond the repo
    # commit is the whole-file sha256 of the parts this run actually read.
    # gguf_surface REQUIRES that marker at capture time (the alternative is
    # --skip-gguf-hashes, which is a disclosed unverified read), and the only
    # cheap moment to hash 200 GB is right after it lands -- not four stages
    # and three GPU-hours later.
    log "hashing every part of the build (gguf verify-files)"
    mapfile -d '' -t GGUF_PARTS < <(python3 - "$CONF" "$DEST" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
root = sys.argv[2].rstrip("/")
for row in (doc.get("target") or {}).get("artifact_files") or []:
    name = row.get("name") if isinstance(row, dict) else row
    if name:
        sys.stdout.write("--file\0" + root + "/" + name + "\0")
PY
    )
    "$VENV/bin/python" "$FS/engines/tools/gguf_surface.py" verify-files \
        "${GGUF_PARTS[@]}" > "$RCPT/artifact-file-verification.json" \
        2>>"$LOGS/fetch_target.log"
    log "parts hashed; marker written beside the build"
    # The same pass writes the artifact's per-tensor-class recipe, MEASURED
    # from the container's own tensor table. seal_receipt prefers this over its
    # unknown-everything default, and for a GGUF that default would be wrong
    # twice: it records `unknown` for embeddings/attention/lm_head, which this
    # artifact quantizes and DECLARES it quantizes, and it would record the
    # dense MLPs at the build's nominal rate when they are Q8_0.
    "$VENV/bin/python" "$FS/engines/tools/gguf_surface.py" scope \
        "${GGUF_PARTS[@]}" --repo "$REPO" --revision "$REV" \
        --out "$RCPT/artifact-scope.json" \
        >/dev/null 2>>"$LOGS/fetch_target.log"
    log "scope written to $RCPT/artifact-scope.json"
  fi
  df -h "$FS" | tee -a "$LOGS/fetch_target.log"
  write_marker
  log "done"
  ;;

fetch_panel)
  REPO="$(jqget panel.repo_id)"
  REV="$(jqget panel.revision)"
  log "fetching panel $REPO @ $REV (include-scoped)"
  # Include-scoping is not an optimisation, it is the difference between 32 GB
  # and 1.3 TB. The globs come from the panel descriptor, never from a constant.
  # This is a PUBLIC dataset fetch.  It must not call load_token or inherit any
  # Hugging Face credential: safe RunPod deliberately transports no token.
  # Pin the official endpoint, disable implicit credential discovery, and use
  # a cache/token namespace isolated from authenticated target downloads.
  #
  # SEC-01.  This used to be an `eval`, which existed only to word-split
  # $INCLUDES -- and gave $REPO and $REV a SECOND round of shell parsing.
  # `panel.repo_id` reaches here verbatim from an operator-supplied
  # --panel-descriptor, so a repo id containing $(...) ran as root and the
  # logged argv showed only the substituted result.  hfmeta validates both
  # fields too, but the shell must still pass each value as one argv element.
  #
  # A NUL-delimited bash array also fixes a second, quieter bug: a newline
  # inside an include pattern was silently split into two argv entries by word
  # splitting.  Array elements must NOT be pre-quoted -- shlex.quote here would
  # make the literal quotes part of the glob.  Needs bash 4.4+ for `mapfile -d`;
  # the instance is Ubuntu bash 5.  Do not port this idiom to a macOS-local
  # script, where bash 3.2 has no mapfile at all.
  mapfile -d '' -t INCLUDES < <(python3 - "$CONF" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for pattern in doc.get("panel", {}).get("include", ["*"]):
    sys.stdout.write("--include\0" + pattern + "\0")
PY
  )
  PUBLIC_HF_HOME="$FS/.hf-public-panel"
  mkdir -p "$PUBLIC_HF_HOME/hub"
  chmod 0700 "$PUBLIC_HF_HOME" "$PUBLIC_HF_HOME/hub"
  [ ! -e "$PUBLIC_HF_HOME/no-token" ] && [ ! -L "$PUBLIC_HF_HOME/no-token" ] || {
    echo "anonymous panel token path unexpectedly exists" >&2
    exit 2
  }
  env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
      -u HUGGINGFACE_HUB_TOKEN \
      HF_ENDPOINT=https://huggingface.co \
      HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
      HF_HUB_ENABLE_HF_TRANSFER=1 \
      HF_HOME="$PUBLIC_HF_HOME" \
      HF_HUB_CACHE="$PUBLIC_HF_HOME/hub" \
      HF_TOKEN_PATH="$PUBLIC_HF_HOME/no-token" \
      "$VENV/bin/hf" download "$REPO" --repo-type dataset --revision "$REV" \
        --local-dir "$PANEL" "${INCLUDES[@]}" >>"$LOGS/fetch_panel.log" 2>&1
  du -sh "$PANEL" | tee -a "$LOGS/fetch_panel.log"
  # The sealed token-panel receipt names its 667 artifacts by ABSOLUTE producer
  # path and verifies each by digest. Stage them there now, where a miss is one
  # named file, rather than at load_panel_windows four stages later.
  python3 "$FS/bin/stage_panel_paths.py" --panel "$PANEL" \
      2>&1 | tee -a "$LOGS/fetch_panel.log"
  write_marker
  log "done"
  ;;

materialize)
  require_target_census
  # Write the artifact's NON-ROUTED function into a tree of its own, which the
  # streaming engine loads as --bf16.  Two surfaces need it, for two different
  # reasons, and the same code serves both:
  #   exl3hf  -- the non-routed tensors are QUANTIZED, so they must be decoded.
  #   tr3     -- they are already the official tensors, but they share shards
  #              with the routed payloads, and transformers derives its
  #              checkpoint key set from the shard FILES rather than the index.
  #              A symlink view therefore reports 54,272 routed payload tensors
  #              as unloaded and the load gate refuses. Here the materializer
  #              decodes NOTHING: it re-shards the natives verbatim.
  SURFACE="$(jqget target.surface)"
  case "$SURFACE" in
    exl3hf|tr3-published|dione) ;;
    *) log "surface=$SURFACE needs no materialization -- skipping"
       write_marker; exit 0 ;;
  esac
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  BF16_DIR="${BF16:-$FS/models/bf16}"
  log "materializing non-routed BF16 tree from $MODELS/target"
  if [ "$SURFACE" = "dione" ]; then
    #   dione -- the retained tensors are already the official ones at source
    #            precision, so this decodes NOTHING; it exists because those
    #            shards also carry the 864 MTP expert tensors the streaming
    #            view filters out of the index.
    "$VENV/bin/python" "$FS/engines/tools/dione_surface.py" materialize \
        --root "$MODELS/target" --out "$MODELS/target-bf16-materialized" \
        --repo "$REPO" --revision "$REV" \
        --official-index "$BF16_DIR/model.safetensors.index.json" \
        2>&1 | tee -a "$LOGS/materialize.log"
  else
  "$VENV/bin/python" "$FS/engines/tools/exl3hf_surface.py" materialize \
      --root "$MODELS/target" --out "$MODELS/target-bf16-materialized" \
      --device cuda --source-repo "$REPO" --source-revision "$REV" \
      --official-index "$BF16_DIR/model.safetensors.index.json" \
      2>&1 | tee -a "$LOGS/materialize.log"
  fi
  df -h "$FS" | tee -a "$LOGS/materialize.log"
  write_marker
  log "done"
  ;;

measure)
  require_target_census
  LANE="$(jqget lane streaming)"
  RUNS="$(jqget cold_runs 1)"
  log "lane=$LANE cold_runs=$RUNS"
  for run in $(seq 1 "$RUNS"); do
    # Receipt-resumable: a run whose capture receipt already exists is skipped,
    # so a preemption costs at most one in-flight run.
    if [ -f "$RCPT/run-$run/capture-receipt.json" ]; then
      log "run $run already captured -- skipping"
      continue
    fi
    mkdir -p "$RCPT/run-$run"
    log "run $run starting"
    "$PY" "$FS/bin/invoke_engine.py" --job "$CONF" --lane "$LANE" \
      --cold-run "$run" --out "$RCPT/run-$run" \
      2>&1 | tee -a "$LOGS/measure-run-$run.log"
  done
  write_marker
  log "done"
  ;;

score)
  # stream_score CAPTURES logits; the divergence is computed here, across the
  # cold runs, by the lane's pinned scorer.  Without this stage `seal` finds no
  # kld-report.json and exits 2 -- after the whole rental is spent.
  LANE="$(jqget lane streaming)"
  log "scoring cold runs (lane=$LANE)"
  "$PY" "$FS/bin/invoke_scorer.py" --job "$CONF" --lane "$LANE" \
      --receipts "$RCPT" \
      2>&1 | tee -a "$LOGS/score.log"
  # The fp32 student logits are transient by design: ~31.7 GB per cold run,
  # and the divergence they were captured for is now computed and sealed. They
  # also sit inside the receipts tree the controller downloads at teardown, so
  # leaving them turns a receipts pull into a 63 GB transfer that times out
  # (observed: `jl download ... timed out after 300.0 seconds`).
  KEEP="$(jqget keep_student_logits false)"
  if [ "$KEEP" != "True" ] && [ "$KEEP" != "true" ]; then
    for d in "$RCPT"/run-*/logits; do
      [ -d "$d" ] || continue
      log "removing transient student logits: $d ($(du -sh "$d" | cut -f1))"
      rm -rf "$d"
    done
  else
    log "keep_student_logits is set -- the per-run logit trees are retained"
  fi
  df -h "$FS" | tee -a "$LOGS/score.log"
  write_marker
  log "done"
  ;;

seal)
  log "sealing submission receipt"
  "$PY" "$FS/bin/seal_receipt.py" --job "$CONF" --receipts "$RCPT" \
      --out "$RCPT/measurement-receipt.json" 2>&1 | tee -a "$LOGS/seal.log"
  ( cd "$RCPT" && sha256sum measurement-receipt.json > RECEIPT.sha256 ) || true
  write_marker
  log "done"
  ;;

capture|capture_repeat)
  ROLE="$(jqget role quant)"
  if [ "$ROLE" != "root" ]; then
    echo "the $STAGE stage is --role root only (job.json says role=$ROLE)" >&2
    exit 2
  fi
  require_target_census
  PREVIEW_OF="$(jqget capture.preview_of)"
  RACE="$(jqget capture.race __absent__)"
  if [ -n "$PREVIEW_OF" ] \
      || { [ "$RACE" != "false" ] && [ "$RACE" != "False" ]; }; then
    echo "$STAGE REFUSES: preview/race paid roots are unsupported by the first safe SSH path." >&2
    exit 3
  fi
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  DATASET_REPO="$(jqget capture.dataset_repository)"
  DEST_REPO="$(jqget capture.publish_root_to)"
  LANE="$(jqget lane)"
  FORM="$(jqget capture.form)"
  SCHED="$(jqget capture.schedule)"
  ENGINE="$(jqget capture.engine)"
  DTYPE="$(jqget capture.dtype)"
  PANEL_REL="$(jqget capture.panel_dir)"
  PANEL_ID="$(jqget capture.panel_id)"
  DSID="$(jqget capture.dataset_id)"
  DSNAME="$(jqget capture.dataset_name)"
  AUTHOR="$(jqget capture.author)"
  EXPECT="$(jqget capture.sanity_expect Paris)"
  DEVICE="$(jqget capture.device)"
  PANEL_BINDING_REL="$(jqget panel.binding_path)"
  PANEL_BINDING_SHA="$(jqget panel.binding_file_sha256)"
  ALLOWLIST_REL="$(jqget capture.unexpected_tensor_allowlist.path)"
  ALLOWLIST_ARTIFACT_SHA="$(jqget capture.unexpected_tensor_allowlist.artifact_sha256)"
  ALLOWLIST_NAMES_SHA="$(jqget capture.unexpected_tensor_allowlist.canonical_sorted_names_sha256)"
  [ -n "$REPO" ] || { echo "job.json has no target.repo_id" >&2; exit 2; }
  [ -n "$DATASET_REPO" ] || {
    echo "job.json has no capture.dataset_repository intended identity" >&2
    exit 2
  }
  [ "$DATASET_REPO" != "$REPO" ] || {
    echo "$STAGE REFUSES: target weights repository and dataset repository are the same ($REPO)." >&2
    exit 3
  }
  if [ -n "$DEST_REPO" ] && [ "$DEST_REPO" != "$DATASET_REPO" ]; then
    echo "$STAGE REFUSES: publish_root_to must be absent or exactly dataset_repository." >&2
    exit 3
  fi
  [ "$ENGINE" = "hf-transformers" ] || {
    echo "$STAGE REFUSES: capture.engine must be hf-transformers." >&2; exit 3;
  }
  [ "$DTYPE" = "bfloat16" ] || {
    echo "$STAGE REFUSES: capture.dtype must be bfloat16." >&2; exit 3;
  }
  [ -n "$DSID" ] || { echo "job.json has no capture.dataset_id" >&2; exit 2; }
  [ -n "$DSNAME" ] || { echo "job.json has no capture.dataset_name" >&2; exit 2; }
  [ -n "$AUTHOR" ] || { echo "job.json has no capture.author" >&2; exit 2; }
  [ -n "$PANEL_ID" ] || { echo "job.json has no capture.panel_id" >&2; exit 2; }
  [ -n "$LANE" ] || { echo "job.json has no lane" >&2; exit 2; }
  [ -n "$FORM" ] || { echo "job.json has no capture.form" >&2; exit 2; }
  [ -n "$SCHED" ] || { echo "job.json has no capture.schedule" >&2; exit 2; }
  [ -n "$DEVICE" ] || { echo "job.json has no capture.device" >&2; exit 2; }
  [ -n "$PANEL_REL" ] || { echo "job.json has no capture.panel_dir" >&2; exit 2; }
  PANEL_PATH="$(python3 - "$FS" "$PANEL_REL" "$FS/bin" <<'PYPATH'
import pathlib, stat, sys
sys.path.insert(0, sys.argv[3])
from fidelity import jobcontract

root = pathlib.Path(sys.argv[1]).resolve()
rel = jobcontract.canonical_relative_path(
    sys.argv[2], "capture.panel_dir")
target = root
for part in rel.parts:
    target = target / part
    try:
        mode = target.lstat().st_mode
    except OSError as exc:
        raise SystemExit("panel directory is absent: %s" % exc)
    if stat.S_ISLNK(mode):
        raise SystemExit("capture.panel_dir may not traverse a symlink")
if not stat.S_ISDIR(target.lstat().st_mode):
    raise SystemExit("capture.panel_dir is not a directory: %s" % target)
print(target)
PYPATH
)"

  if [ "$STAGE" = "capture" ]; then
    OUT="$FS/dataset"
    PROCESS_LABEL="root-cold-1"
  else
    require_stage_marker verify
    OUT="$FS/dataset-repeat"
    PROCESS_LABEL="root-cold-2"
  fi
  PANEL_BINDING="$(python3 - "$FS" "$PANEL_BINDING_REL" "$PANEL_BINDING_SHA" \
      "$CONF" "$FS/bin" <<'PYPANEL'
import hashlib, pathlib, stat, sys
sys.path.insert(0, sys.argv[5])
from fidelity import jobcontract

root = pathlib.Path(sys.argv[1]).resolve()
rel_text, expected, job_path = sys.argv[2:5]
with open(job_path, "rb") as handle:
    job = jobcontract.parse_job_bytes(handle.read())
if "allow_unexpected_tensors" in (job.get("capture") or {}):
    raise SystemExit(
        "capture.allow_unexpected_tensors is obsolete; broad acceptance always refuses")
if not expected:
    raise SystemExit("job.json has no panel.binding_file_sha256")
rel = jobcontract.canonical_relative_path(
    rel_text, "panel.binding_path")
target = root
for part in rel.parts:
    target = target / part
    try:
        mode = target.lstat().st_mode
    except OSError as exc:
        raise SystemExit("panel binding file is absent: %s" % exc)
    if stat.S_ISLNK(mode):
        raise SystemExit("panel.binding_path may not traverse a symlink")
if not stat.S_ISREG(target.lstat().st_mode):
    raise SystemExit("panel binding path is not a regular file: %s" % target)
raw = target.read_bytes()
jobcontract.parse_job_bytes(raw)
observed = hashlib.sha256(raw).hexdigest()
if observed != expected:
    raise SystemExit("panel.binding_file_sha256 mismatch: expected %s, observed %s"
                     % (expected, observed))
print(target)
PYPANEL
)"
  EXTRA=(--sanity-expect "$EXPECT"
         --panel-binding "$PANEL_BINDING"
         --panel-binding-sha256 "$PANEL_BINDING_SHA"
         --panel-tokenizer-root "$MODELS/target")
  if [ -n "$ALLOWLIST_REL$ALLOWLIST_ARTIFACT_SHA$ALLOWLIST_NAMES_SHA" ]; then
    if [ -z "$ALLOWLIST_REL" ] || [ -z "$ALLOWLIST_ARTIFACT_SHA" ] || [ -z "$ALLOWLIST_NAMES_SHA" ]; then
      echo "$STAGE REFUSES: unexpected_tensor_allowlist path and both SHA-256 identities are all-or-none." >&2
      exit 3
    fi
    ALLOWLIST_PATH="$(python3 - "$FS" "$ALLOWLIST_REL" "$ALLOWLIST_ARTIFACT_SHA" \
        "$ALLOWLIST_NAMES_SHA" "$FS/bin" <<'PYALLOW'
import hashlib, json, pathlib, stat, sys
sys.path.insert(0, sys.argv[5])
from fidelity import jobcontract

root = pathlib.Path(sys.argv[1]).resolve()
rel = jobcontract.canonical_relative_path(
    sys.argv[2], "capture.unexpected_tensor_allowlist.path")
target = root
for part in rel.parts:
    target = target / part
    try:
        mode = target.lstat().st_mode
    except OSError as exc:
        raise SystemExit("unexpected tensor allowlist is absent: %s" % exc)
    if stat.S_ISLNK(mode):
        raise SystemExit("unexpected tensor allowlist may not traverse a symlink")
if not stat.S_ISREG(target.lstat().st_mode):
    raise SystemExit("unexpected tensor allowlist is not a regular file")
raw = target.read_bytes()
observed_raw = hashlib.sha256(raw).hexdigest()
if observed_raw != sys.argv[3]:
    raise SystemExit("unexpected tensor allowlist raw SHA-256 mismatch")
try:
    names = json.loads(raw.decode("utf-8"),
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           ValueError("non-finite JSON token %s" % value)))
except (UnicodeDecodeError, ValueError, TypeError) as exc:
    raise SystemExit("unexpected tensor allowlist is not strict JSON: %s" % exc)
if (not isinstance(names, list) or not names
        or any(not isinstance(name, str) or not name for name in names)):
    raise SystemExit("unexpected tensor allowlist must be a non-empty JSON string array")
if len(names) != len(set(names)):
    raise SystemExit("unexpected tensor allowlist contains duplicate names")
canonical = json.dumps(sorted(names), separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode("utf-8")
observed_names = hashlib.sha256(canonical).hexdigest()
if observed_names != sys.argv[4]:
    raise SystemExit("unexpected tensor allowlist canonical-name SHA-256 mismatch")
print(target)
PYALLOW
)"
    EXTRA+=(--unexpected-tensors-allowlist "$ALLOWLIST_PATH"
            --unexpected-tensors-allowlist-sha256 "$ALLOWLIST_ARTIFACT_SHA"
            --unexpected-tensors-name-sha256 "$ALLOWLIST_NAMES_SHA")
  fi
  if [ -e "$OUT" ]; then
    echo "$STAGE REFUSES: $OUT already exists without this stage's bound done marker." >&2
    echo "  Use a fresh run root; a partial/stale capture is never adopted as a fresh process." >&2
    exit 3
  fi
  log "capturing fresh process $PROCESS_LABEL: $REPO @ $REV -> $OUT"
    # The two repository arguments are intentionally different identities:
    # weights_repository is what was executed; repository is the intended
    # dataset identity whether or not a later mutation is authorized.
    HF_HOME="$FS/hf" "$PY" "$FS/bin/fidelity_dataset.py" capture \
        --out "$OUT" --form "$FORM" --role root --lane "$LANE" \
        --engine "$ENGINE" -- \
        --model "$MODELS/target" --weights-repository "$REPO" \
        --repository "$DATASET_REPO" --model-revision "$REV" \
        --panel "$PANEL_PATH" --panel-id "$PANEL_ID" \
        --schedule "$SCHED" --device "$DEVICE" --dtype "$DTYPE" \
        --dataset-id "$DSID" --dataset-name "$DSNAME" \
        --run-name "$PROCESS_LABEL" --cold-run "$PROCESS_LABEL" \
        --author "$AUTHOR" --role root \
        "${EXTRA[@]}" \
        2>&1 | tee -a "$LOGS/$STAGE.log"
  du -sh "$OUT" | tee -a "$LOGS/$STAGE.log"
  write_marker
  log "done"
  ;;

race_bootstrap)
  echo "race_bootstrap REFUSES: preview/race paid roots are unsupported by the first safe SSH path." >&2
  exit 3
  ;;

race_capture)
  echo "race_capture REFUSES: preview/race paid roots are unsupported by the first safe SSH path." >&2
  exit 3
  ;;

publish_root)
  echo "publish_root REFUSES: root publication is controller-local only after verified retrieval, provider-confirmed pod absence, and billing reconciliation." >&2
  exit 3
  ;;

verify|verify_repeat)
  ROLE="$(jqget role quant)"
  if [ "$ROLE" != "root" ]; then
    echo "the $STAGE stage is --role root only (job.json says role=$ROLE)" >&2
    exit 2
  fi
  if [ "$STAGE" = "verify" ]; then
    require_stage_marker capture
    OUT="$FS/dataset"
    VERIFY_RECEIPT="$RCPT/dataset-verify.json"
  else
    require_stage_marker capture_repeat
    OUT="$FS/dataset-repeat"
    VERIFY_RECEIPT="$RCPT/dataset-repeat-verify.json"
  fi
  [ -d "$OUT" ] || { echo "$STAGE REFUSES: dataset path absent ($OUT)" >&2; exit 3; }
  log "independently verifying $OUT (seal + digest chain + tensor content)"
  "$PY" "$FS/bin/fidelity_dataset.py" verify "$OUT" --json "$VERIFY_RECEIPT" \
      2>&1 | tee -a "$LOGS/$STAGE.log"
  "$PY" "$FS/bin/fidelity_dataset.py" describe "$OUT" \
      2>&1 | tee -a "$LOGS/$STAGE.log"
  write_marker
  log "done"
  ;;

compare_root)
  require_stage_marker verify
  require_stage_marker verify_repeat
  [ "$(realpath "$FS/dataset")" != "$(realpath "$FS/dataset-repeat")" ] || {
    echo "compare_root REFUSES: canonical and repeat resolve to one path." >&2
    exit 3
  }
  REPLAY_DEVICE="$(jqget capture.replay_device)"
  REPLAY_DTYPE="$(jqget capture.replay_dtype)"
  VOCAB_CHUNK="$(jqget capture.vocab_chunk)"
  [ -n "$REPLAY_DEVICE" ] || {
    echo "compare_root REFUSES: job.json must explicitly name capture.replay_device." >&2
    exit 2
  }
  [ -n "$REPLAY_DTYPE" ] || {
    echo "compare_root REFUSES: job.json must explicitly name capture.replay_dtype." >&2
    exit 2
  }
  [ -n "$VOCAB_CHUNK" ] || {
    echo "compare_root REFUSES: job.json must explicitly name capture.vocab_chunk." >&2
    exit 2
  }
  if [ "$REPLAY_DEVICE" = "numpy" ]; then
    COMPARE_DEVICE="cpu"
  else
    COMPARE_DEVICE="$REPLAY_DEVICE"
  fi
  log "running forced SC-1 between distinct cold captures (replay=$REPLAY_DEVICE/$REPLAY_DTYPE)"
  "$PY" "$FS/bin/fidelity_dataset.py" compare \
      --reference "$FS/dataset" --candidate "$FS/dataset-repeat" \
      --reference-label root-cold-1 --candidate-label root-cold-2 \
      --self-compare --force-compute --device "$COMPARE_DEVICE" \
      --replay-device "$REPLAY_DEVICE" --replay-dtype "$REPLAY_DTYPE" \
      --vocab-chunk "$VOCAB_CHUNK" --out "$RCPT/root-comparison" \
      2>&1 | tee -a "$LOGS/compare_root.log"
  write_marker
  log "done"
  ;;

qualify_root)
  require_stage_marker verify
  require_stage_marker verify_repeat
  require_stage_marker compare_root
  "$PY" "$FS/bin/fidelity_dataset.py" qualify-root \
      --job "$CONF" \
      --first "$FS/dataset" --repeat "$FS/dataset-repeat" \
      --first-label root-cold-1 --repeat-label root-cold-2 \
      --first-verify "$RCPT/dataset-verify.json" \
      --repeat-verify "$RCPT/dataset-repeat-verify.json" \
      --comparison "$RCPT/root-comparison/comparison-receipt.json" \
      --out "$RCPT/root-qualification.json" \
      2>&1 | tee -a "$LOGS/qualify_root.log"
  write_marker
  log "done"
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  echo "stages: setup fetch_target fetch_panel materialize measure score seal" >&2
  echo "        capture verify capture_repeat verify_repeat compare_root qualify_root publish_root" >&2
  echo "        race_bootstrap/race_capture explicitly refuse paid roots" >&2
  exit 2
  ;;
esac
