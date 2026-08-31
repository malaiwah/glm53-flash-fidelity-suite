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
# NEVER `set -x` in a stage where HF_TOKEN is in scope.
set -euo pipefail

STAGE="${1:?usage: stage_measure.sh <stage>}"
FS="${FIDELITY_FS_ROOT:-/home/jl_fs/fidelity}"
# The engine tree. FIDELITY_K6_ROOT is the pre-2026-08-31 spelling, kept
# as a fallback so a controller and an instance from different checkouts
# cannot silently disagree about where the venv is.
ROOT="${FIDELITY_ENGINE_ROOT:-${FIDELITY_K6_ROOT:-/home/jl_fs/fidelity-engine}}"
RCPT="$FS/receipts"
DONE="$RCPT/done"
LOGS="$FS/logs"
MODELS="$FS/models"
PANEL="$FS/panel"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
export VENV

mkdir -p "$RCPT" "$DONE" "$LOGS" "$MODELS" "$PANEL" "$FS/.secrets"
chmod 700 "$FS/.secrets" 2>/dev/null || true

# Config written by the controller before any stage runs.
CONF="$FS/job.json"
# Read a dotted path out of job.json. Uses the system python3 rather than the
# venv's, because this must work in `setup` -- before the venv exists.
jqget() {  # jqget <dotted.path> [default]
  python3 -c '
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    print(sys.argv[3]); raise SystemExit(0)
cur = doc
for part in sys.argv[2].split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = sys.argv[3]
        break
# A JSON null is ABSENT, not the four-letter string "None". Without this line
# every `[ -n "$X" ]` guard in this file reads a null as PRESENT: a job that set
# no preview handed the capture `--preview-of None`, a dataset id spelled None,
# and a job with no capture.panel_dir failed with "panel not uploaded: .../None"
# instead of naming the missing key.
if cur is None:
    cur = sys.argv[3]
print(cur if not isinstance(cur, (dict, list)) else json.dumps(cur))
' "$CONF" "$1" "${2-}"
}

log() { echo "[$(date -u +%FT%TZ)] stage_measure/$STAGE: $*"; }

marker="$DONE/$STAGE.done"
if [ "$STAGE" != "setup" ] && [ -f "$marker" ]; then
  log "already done (marker $marker) -- skipping"
  exit 0
fi

# Every stage after setup runs under the venv setup builds.  Without this guard
# a stage launched before setup finished died as a bare `exit 127` -- "not
# found" -- which says nothing about the actual dependency.
if [ "$STAGE" != "setup" ] && [ ! -x "$PY" ]; then
  echo "stage_measure: error: $STAGE needs the venv interpreter $PY, which does not exist yet." >&2
  echo "  The setup stage builds it. Run (or finish) 'stage_measure.sh setup' first." >&2
  exit 3
fi

# Load the HF token from its 0600 file, never from argv or the environment of a
# command line that could be observed in the process list.
load_token() {
  if [ -f "$FS/.secrets/hf_token" ]; then
    HF_TOKEN="$(cat "$FS/.secrets/hf_token")"
    export HF_TOKEN
  fi
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
  touch "$marker"
  log "done"
  ;;

fetch_target)
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  DEST="$MODELS/target"
  [ -n "$REPO" ] || { echo "job.json has no target.repo_id" >&2; exit 2; }
  mkdir -p "$DEST"
  # A GGUF repo is a SHELF, not an artifact: unsloth/GLM-5.3-Flash-GGUF
  # publishes twelve builds at one revision and a whole-repo download is 2.55
  # TB for the ~200 GB one build actually needs. The plan already chose the
  # build and listed its parts by name, so fetch exactly those -- and only
  # those, because a `--include <build>/*` glob would also pull a sidecar the
  # publisher adds tomorrow into a run whose receipt names today's file list.
  #
  # Built as a NUL-delimited bash array for the same reason fetch_panel is:
  # these names come from job.json, and the shell must not be parsing data.
  # There is no `eval` here and there must never be one.
  mapfile -d '' -t TARGET_INCLUDES < <(python3 - "$CONF" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
target = doc.get("target") or {}
if (target.get("surface") or "") != "gguf":
    raise SystemExit(0)
for row in target.get("artifact_files") or []:
    name = row.get("name") if isinstance(row, dict) else row
    if name:
        sys.stdout.write("--include\0" + name + "\0")
PY
  )
  if [ "${#TARGET_INCLUDES[@]}" -gt 0 ]; then
    log "fetching $REPO @ $REV -> $DEST  ($((${#TARGET_INCLUDES[@]} / 2)) named files of a multi-build repo)"
  else
    log "fetching $REPO @ $REV -> $DEST"
  fi
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --revision "$REV" \
      --local-dir "$DEST" --max-workers 8 "${TARGET_INCLUDES[@]}" \
      >>"$LOGS/fetch_target.log" 2>&1
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
  touch "$marker"
  log "done"
  ;;

fetch_panel)
  load_token
  REPO="$(jqget panel.repo_id)"
  REV="$(jqget panel.revision)"
  log "fetching panel $REPO @ $REV (include-scoped)"
  # Include-scoping is not an optimisation, it is the difference between 32 GB
  # and 1.3 TB. The globs come from the panel descriptor, never from a constant.
  # SEC-01.  This used to be an `eval`, which existed only to word-split
  # $INCLUDES -- and gave $REPO and $REV a SECOND round of shell parsing on a
  # box that holds a live HF token.  `panel.repo_id` reaches here verbatim from
  # an operator-supplied --panel-descriptor, so a repo id containing $(...) ran
  # as root, and the logged argv showed only the substituted result: an
  # injection invisible in fetch_panel.log.  hfmeta.load_panel_descriptor now
  # validates both fields at ingestion, but that is a backstop in another file
  # and another language; the shell must not be parsing data at all.
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
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --repo-type dataset --revision "$REV" \
      --local-dir "$PANEL" "${INCLUDES[@]}" >>"$LOGS/fetch_panel.log" 2>&1
  du -sh "$PANEL" | tee -a "$LOGS/fetch_panel.log"
  # The sealed token-panel receipt names its 667 artifacts by ABSOLUTE producer
  # path and verifies each by digest. Stage them there now, where a miss is one
  # named file, rather than at load_panel_windows four stages later.
  python3 "$FS/bin/stage_panel_paths.py" --panel "$PANEL" \
      2>&1 | tee -a "$LOGS/fetch_panel.log"
  touch "$marker"
  log "done"
  ;;

materialize)
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
       touch "$marker"; exit 0 ;;
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
  touch "$marker"
  log "done"
  ;;

measure)
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
  touch "$marker"
  log "done"
  ;;

score)
  # stream_score CAPTURES logits; the divergence is computed here, across the
  # cold runs, by the lane's pinned scorer.  Without this stage `seal` finds no
  # kld-report.json and exits 2 -- after the whole rental is spent.
  LANE="$(jqget lane streaming)"
  log "scoring cold runs (lane=$LANE)"
  "$PY" "$FS/bin/invoke_scorer.py" --job "$CONF" --lane "$LANE" \
      --receipts "$RCPT" --device "${KLD_DEVICE:-cuda}" \
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
  touch "$marker"
  log "done"
  ;;

seal)
  log "sealing submission receipt"
  "$PY" "$FS/bin/seal_receipt.py" --job "$CONF" --receipts "$RCPT" \
      --out "$RCPT/measurement-receipt.json" 2>&1 | tee -a "$LOGS/seal.log"
  ( cd "$RCPT" && sha256sum measurement-receipt.json > RECEIPT.sha256 ) || true
  touch "$marker"
  log "done"
  ;;

capture)
  # --role root. There is no candidate and no reference: this stage runs the
  # reference model itself over the panel and WRITES the sealed dataset that
  # later measurements are a distance from. hf_capture.py writes the dataset at
  # --out itself, so there is no separate assembly step.
  ROLE="$(jqget role quant)"
  if [ "$ROLE" != "root" ]; then
    echo "the capture stage is --role root only (job.json says role=$ROLE)" >&2
    exit 2
  fi
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  LANE="$(jqget lane streaming)"
  FORM="$(jqget capture.form hidden)"
  SCHED="$(jqget capture.schedule layer-outer)"
  PANEL_REL="$(jqget capture.panel_dir)"
  PANEL_ID="$(jqget capture.panel_id)"
  DSID="$(jqget capture.dataset_id)"
  DSNAME="$(jqget capture.dataset_name)"
  AUTHOR="$(jqget capture.author malaiwah)"
  EXPECT="$(jqget capture.sanity_expect Paris)"
  ALLOW_UNEXPECTED="$(jqget capture.allow_unexpected_tensors false)"
  # THE GPU THIS RUN IS PAYING FOR. hf_capture's --device defaults to "cpu" and
  # nothing here ever set it, so every --role root capture ran the forward on
  # the CPU with the rented card at 0% utilisation -- observed on a Lambda
  # GH200 ($2.29/h) and a RunPod A100 ($1.39/h), and visible in the sealed
  # dataset as runtime.stack_fingerprint.device "cpu". The `materialize` and
  # `measure` stages have always passed `--device cuda`; only the root path was
  # missed, and its GPU is the entire reason the box is rented.
  DEVICE="$(jqget capture.device cuda)"
  OUT="$FS/dataset"
  [ -n "$DSID" ] || { echo "job.json has no capture.dataset_id" >&2; exit 2; }
  [ -n "$PANEL_REL" ] || { echo "job.json has no capture.panel_dir" >&2; exit 2; }
  [ -d "$FS/$PANEL_REL" ] || { echo "panel not uploaded: $FS/$PANEL_REL" >&2; exit 2; }
  # An ARRAY, never an eval: these values come from job.json and the shell must
  # not be parsing data (SEC-01). Same construction as race_capture below.
  #
  # --sanity-expect was written into job.json by the controller and read by
  # race_capture only, so on THIS path -- the default one -- the probe ran
  # unenforced: a capture that generated nonsense sealed clean, and the operator
  # who passed `--sanity-expect Paris` was told a check had happened that had
  # not. The two capture stages now forward the same two flags.
  EXTRA=()
  EXTRA+=(--sanity-expect "$EXPECT")
  if [ "$ALLOW_UNEXPECTED" = "true" ] || [ "$ALLOW_UNEXPECTED" = "True" ]; then
    EXTRA+=(--allow-unexpected-tensors)
  fi
  if [ -d "$OUT" ]; then
    log "dataset already written at $OUT -- skipping (receipt-resumable)"
  else
    log "capturing $REPO @ $REV  form=$FORM schedule=$SCHED panel=$PANEL_ID"
    # --model is the LOCAL tree fetch_target already downloaded; --repository
    # and --model-revision keep the dataset's recorded identity pointing at the
    # published repo rather than at a path on a machine that will not exist.
    HF_HOME="$FS/hf" "$PY" "$FS/bin/fidelity_dataset.py" capture \
        --out "$OUT" --form "$FORM" --role root --lane "$LANE" \
        --engine hf-transformers -- \
        --model "$MODELS/target" --weights-repository "$REPO" \
        --repository "$REPO" --model-revision "$REV" \
        --panel "$FS/$PANEL_REL" --panel-id "$PANEL_ID" \
        --schedule "$SCHED" --device "$DEVICE" --dtype bfloat16 \
        --dataset-id "$DSID" --dataset-name "$DSNAME" \
        --author "$AUTHOR" --role root \
        "${EXTRA[@]}" \
        2>&1 | tee -a "$LOGS/capture.log"
  fi
  du -sh "$OUT" | tee -a "$LOGS/capture.log"
  touch "$marker"
  log "done"
  ;;

race_bootstrap)
  # RACE MODE, step 1. Fetch ONLY the kilobytes that make everything else
  # plannable: config.json (the architecture), the tokenizer (the sanity probe
  # needs it), and above all model.safetensors.index.json -- whose weight_map is
  # the map from tensor name to shard, i.e. the only statement of which shard
  # holds which layer. Without it there is no fetch ORDER, only a download.
  #
  # This is deliberately NOT `fetch_target`: no shard is fetched here at all.
  # The shards are fetched by the capture, in the order it needs them.
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  DEST="$MODELS/target"
  [ -n "$REPO" ] || { echo "job.json has no target.repo_id" >&2; exit 2; }
  mkdir -p "$DEST"
  log "fetching the bootstrap files of $REPO @ $REV (no shards)"
  # `--include` per name rather than a whole-repo pull: the shards are the
  # capture's business, and a sidecar the publisher adds tomorrow must not
  # silently join a run whose receipt names today's file list.
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --revision "$REV" --local-dir "$DEST" \
      --include config.json \
      --include model.safetensors.index.json \
      --include generation_config.json \
      --include "tokenizer*" \
      --include special_tokens_map.json \
      --include chat_template.jinja \
      >>"$LOGS/race_bootstrap.log" 2>&1 || true
  if [ ! -f "$DEST/model.safetensors.index.json" ]; then
    echo "race mode needs $DEST/model.safetensors.index.json: it is the map from" >&2
    echo "  tensor to shard, and without it the fetch cannot be ordered by layer." >&2
    echo "  A single-shard checkpoint has nothing to overlap; run without --race." >&2
    exit 3
  fi
  python3 -c 'import json,sys; wm=json.load(open(sys.argv[1])).get("weight_map") or {}; print("index: %d tensors over %d shards" % (len(wm), len(set(wm.values()))))' \
      "$DEST/model.safetensors.index.json" | tee -a "$LOGS/race_bootstrap.log"
  df -h "$FS" | tee -a "$LOGS/race_bootstrap.log"
  touch "$marker"
  log "done"
  ;;

race_capture)
  # RACE MODE, step 2. The fetch and the capture are ONE process: hf_capture
  # starts a priority-ordered background fetch (resident set, then layer 0,
  # layer 1, ...) and the layer-outer loader blocks on the shards for layer N
  # only when it is about to load layer N. Worst case this is no slower than
  # fetch-then-capture; the report written at the end says what it actually
  # was, measured, per block -- it is not asserted here.
  ROLE="$(jqget role quant)"
  if [ "$ROLE" != "root" ]; then
    echo "the race_capture stage is --role root only (job.json says role=$ROLE)" >&2
    exit 2
  fi
  load_token
  REPO="$(jqget target.repo_id)"
  REV="$(jqget target.revision)"
  LANE="$(jqget lane streaming)"
  FORM="$(jqget capture.form hidden)"
  SCHED="$(jqget capture.schedule layer-outer)"
  PANEL_REL="$(jqget capture.panel_dir)"
  PANEL_ID="$(jqget capture.panel_id)"
  DSID="$(jqget capture.dataset_id)"
  DSNAME="$(jqget capture.dataset_name)"
  AUTHOR="$(jqget capture.author malaiwah)"
  WORKERS="$(jqget capture.race_workers 8)"
  PREVIEW_OF="$(jqget capture.preview_of)"
  EXPECT="$(jqget capture.sanity_expect Paris)"
  ALLOW_UNEXPECTED="$(jqget capture.allow_unexpected_tensors false)"
  DEVICE="$(jqget capture.device cuda)"   # see the capture stage above
  OUT="$FS/dataset"
  [ -n "$DSID" ] || { echo "job.json has no capture.dataset_id" >&2; exit 2; }
  [ -n "$PANEL_REL" ] || { echo "job.json has no capture.panel_dir" >&2; exit 2; }
  [ -d "$FS/$PANEL_REL" ] || { echo "panel not uploaded: $FS/$PANEL_REL" >&2; exit 2; }
  [ "$SCHED" = "layer-outer" ] || {
    echo "race mode needs schedule=layer-outer (job.json says $SCHED)" >&2; exit 2; }
  # Built as an ARRAY, never an eval: these values come from job.json, and the
  # shell must not be parsing data (SEC-01).
  EXTRA=()
  if [ -n "$PREVIEW_OF" ]; then EXTRA+=(--preview-of "$PREVIEW_OF"); fi
  EXTRA+=(--sanity-expect "$EXPECT")
  if [ "$ALLOW_UNEXPECTED" = "true" ] || [ "$ALLOW_UNEXPECTED" = "True" ]; then
    EXTRA+=(--allow-unexpected-tensors)
  fi
  if [ -d "$OUT" ]; then
    log "dataset already written at $OUT -- skipping (receipt-resumable)"
  else
    log "race capture $REPO @ $REV workers=$WORKERS panel=$PANEL_ID preview_of=${PREVIEW_OF:-none}"
    HF_HOME="$FS/hf" HF_HUB_ENABLE_HF_TRANSFER=1 "$PY" "$FS/bin/fidelity_dataset.py" capture \
        --out "$OUT" --form "$FORM" --role root --lane "$LANE" \
        --engine hf-transformers -- \
        --model "$MODELS/target" --weights-repository "$REPO" \
        --repository "$REPO" --model-revision "$REV" \
        --panel "$FS/$PANEL_REL" --panel-id "$PANEL_ID" \
        --schedule "$SCHED" --device "$DEVICE" --layer-residency stream --dtype bfloat16 \
        --dataset-id "$DSID" --dataset-name "$DSNAME" \
        --author "$AUTHOR" --role root \
        --race-repo "$REPO" --race-revision "$REV" \
        --race-workers "$WORKERS" \
        --race-report "$RCPT/race-fetch-report.json" \
        "${EXTRA[@]}" \
        2>&1 | tee -a "$LOGS/race_capture.log"
  fi
  # The release's published seal, verified once the tree is COMPLETE -- which
  # under race mode is only true after the capture returns, not before it
  # starts. Same instrument as fetch_target, moved to the moment it can run.
  if [ -f "$MODELS/target/SHA256SUMS" ]; then
    log "verifying published SHA256SUMS (weights fail-closed; absent sidecars reported)"
    python3 "$FS/bin/verify_published_sums.py" --root "$MODELS/target" \
        --out "$RCPT/shard-verification.json" \
        2>&1 | tee "$RCPT/shard-verification.txt"
  else
    log "no SHA256SUMS published; recording that fact in the receipt"
    echo "no SHA256SUMS in release" > "$RCPT/shard-verification.txt"
  fi
  du -sh "$OUT" | tee -a "$LOGS/race_capture.log"
  df -h "$FS" | tee -a "$LOGS/race_capture.log"
  touch "$marker"
  log "done"
  ;;

verify)
  # Recompute the dataset's own seal and digest chain BEFORE the box is
  # destroyed. This is the last moment at which a bad capture is free to throw
  # away: after teardown the only way to find out is to re-rent and re-capture.
  # Tensor content is recomputed by default -- the seal covers the manifest and
  # checksums.txt, so a byte flipped inside a tensor whose checksums were then
  # refreshed is only ever caught here.
  OUT="$FS/dataset"
  log "verifying $OUT (seal + digest chain + tensor content)"
  "$PY" "$FS/bin/fidelity_dataset.py" verify "$OUT" \
      2>&1 | tee -a "$LOGS/verify.log"
  "$PY" "$FS/bin/fidelity_dataset.py" describe "$OUT" \
      2>&1 | tee -a "$LOGS/verify.log"
  touch "$marker"
  log "done"
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  echo "stages: setup fetch_target fetch_panel materialize measure score seal" >&2
  echo "        capture verify   (--role root)" >&2
  echo "        race_bootstrap race_capture verify   (--role root --race)" >&2
  exit 2
  ;;
esac
