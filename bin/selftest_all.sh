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
# Many component selftests use `assert`; an inherited optimization flag must
# not turn their checks into no-ops while this battery reports them as passed.
unset PYTHONOPTIMIZE
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
# P1-06. The replay comparator's KL reduction must run in float64 BEFORE the
# vocabulary sum: the float32 reduction returned negative "KL" (~-8e-7) on
# near-equal 50k-vocab distributions while its receipts declared fp64.
t "fidelity reducer: fp64 known answers (P1-06)" 0 "$VPY" bin/selftest_fidelity_reducer.py
t "registry client/viewer/matcher (T1)"    0 python3 bin/selftest_registry_view.py
# P1-07. identical_across_runs=true needs one valid digest PER claimed run, all
# equal. The old ingest collapsed digests to a set first, so one digest plus
# four missing digests manufactured "five runs, bitwise identical".
t "determinism ingest: per-run digests (P1-07)" 0 python3 bin/selftest_registry_determinism.py
t "floor-aware stats known answers (T2)"   0 python3 bin/selftest_stats.py
t "preview estimator coverage (T3)"        0 python3 bin/selftest_preview_stats.py
t "submission refusability (T5)"           0 python3 bin/selftest_submission_refusal.py
t "scope must match the release (T5b)"      0 python3 bin/selftest_scope_crosscheck.py
t "root capture: --role root (T5c)"        0 python3 bin/selftest_root_capture.py
# T15. Race mode: the overlapped fetch, the preview/final identity separation and
# the generation sanity probe. The overlap is measured against a CONTROL arm on a
# simulated link, because a schedule cannot be A/B-tested on a real 1.5 TB fetch;
# the identity cases are the ones that matter most, because updating a published
# root in place would put rows measured against different bytes in ONE
# comparability group. Needs torch+transformers; SKIPs loudly without them.
t "race mode: overlap, preview identity, generation probe (T15)" \
                                           0 "$VPY" bin/selftest_race_mode.py
t "provider portability (T5d)"             0 python3 bin/selftest_provider_portability.py
# T17. The container transport. Porting to three clouds produced five defects and
# not one was about the measurement; an image deletes that category, but only if
# it drives the SAME stages from the SAME contract. These rungs are the anti-drift
# ones: one stage sequence with one owner, one job.json contract with two writers,
# the token still a 0600 file, and -- the acceptance test in code -- recording
# WHICH container ran must not move stack_fingerprint_sha256, because that digest
# is what dscompare reads to decide stack_relation.
t "container transport (T17)"                0 "$VPY" bin/selftest_container.py
t "fp8 -> bf16 losslessness (T5e)"         0 python3 bin/selftest_fp8_lossless.py
t "canonical_json: bin == registry (T5f)"  0 python3 bin/selftest_canonical_json.py
# P1-08. NaN/Infinity are not JSON: refused at the parse (parse_constant), by
# both canonical serializers (allow_nan=False), and by the minischema's
# recursive finiteness walk -- bound checks fail open on NaN otherwise.
t "non-finite rejection: ingest/seal/render (P1-08)" 0 python3 bin/selftest_nonfinite_rejection.py
t "bundle completeness (T5g)"              0 python3 bin/selftest_bundle_complete.py
# T18. The guards for what a NAMING SWEEP can destroy. This tree is being swept
# of GLM/K6 names now that it measures four families; half those strings are
# identity, not history. 157 registry ids and 239 receipt schema literals are
# frozen in bin/published-identity.json -- COMPARABILITY_KEY_FIELDS hashes
# panel_id and reference_id, so renaming one regroups every measurement that
# referenced it, silently. Also: both sides of each two-file agreement, and no
# helper existing byte-identically in two places (k6_publish.py did).
t "naming sweep: published identity + two-file agreements (T18)" \
                                           0 python3 bin/selftest_naming_sweep.py
# T13. The GGUF lane end to end: shelf -> plan -> argv -> fetch -> receipt. The
# adapter, the scorer, the aggregator and the registry adapter all existed and
# NONE of it was reachable, because no lane classified a llama.cpp container as
# anything -- so the runner refused the model's largest quant audience with a
# refusal that was simply untrue. Rung 5 is the one that stops the next surface
# landing half-wired.
t "gguf lane: shelf, profile, argv, fetch scope, receipt (T13)" \
                                           0 python3 bin/selftest_gguf_lane.py
# T19. The bearer token must not follow a redirect off the original origin.
# HF /resolve/ URLs 302 to CDN hosts; urllib's default handler forwards
# Authorization across that hop. Driven against local stub servers -- no
# network, no real token. (Peer review 2026-08-31, security chapter.)
t "no cross-origin bearer forwarding (T19: R1-R8)" \
                                           0 python3 bin/selftest_hf_redirect.py
# T20. Secret files: 0600 from the first instant at both ends, chmod 700 on
# the remote directory BEFORE upload, upload to a temp name + atomic rename,
# exclusive/no-follow creation (a planted symlink on the container bind mount
# used to be written THROUGH), and token cleanup in a finally. Stub provider,
# no network, no real token. (Peer review 2026-08-31, security chapter.)
t "secret file creation + transport + cleanup (T20: S1-S9)" \
                                           0 python3 bin/selftest_secret_files.py
# T21. SSH host authentication. StrictHostKeyChecking=no + /dev/null removed
# server authentication from the channel carrying the token and the receipts;
# now: per-run trust-on-first-use, a known_hosts file under the run dir, and
# the accepted fingerprint recorded for the cost receipt.
t "ssh per-run TOFU host keys (T21: K1-K6)" \
                                           0 python3 bin/selftest_sshbase.py
# T22. The reaper (P1-03). Destruction requires a provider id from a lease
# THIS tool wrote; names only discover; every destroy is confirmed terminal
# with retry/backoff or the sweep exits EXIT_LEAK; dry-run enumerates exactly
# the real run's mutations. Mocked provider, $0.00.
t "reaper: lease-authorized, confirmed, faithful dry-run (T22: P1-P8)" \
                                           0 python3 bin/selftest_reaper.py
# T23. Mandatory gates are tri-state (P1-11): verified / failed / not_checked.
# An import failure or network blip used to warn-and-continue, and the dry
# run then ended in "all checks passed" about a seal nobody recomputed. Now a
# real run refuses on not_checked; a dry run visibly downgrades to
# ESTIMATE ONLY. G5 forces offline via a closed local port -- no network.
t "seal gates fail closed, offline dry-run is estimate-only (T23: G1-G5)" \
                                           0 python3 bin/selftest_seal_gate.py
# T24. Job identity and duplicate writers (P1-12/P1-14). Identity is hashed
# AFTER revision resolution, at 256 bits, including the suite HEAD; adoption
# compares the lease's full id, never the 8-char name prefix; liveness is
# tri-state and unknown never authorizes a launch. Stub provider, $0.00.
t "job identity resolved-first + tri-state liveness (T24: J1-J7)" \
                                           0 python3 bin/selftest_job_identity.py
# T25. ROOT-1: a sealed root gets PUBLISHED (publish_root stage after verify,
# receipt pins the uploaded revision) and teardown refuses to destroy a
# verified-but-unpublished root without an explicit override -- a sealed,
# twice-validated root was destroyed at teardown for $6.59. Also the
# container-native RunPod launch (our image + dockerArgs; token via env only).
t "root publish + teardown guard + container-native launch (T25: RP1-RP9)" \
                                           0 python3 bin/selftest_root_publish.py
# T26. Result sinks. ROOT-1 gave a multi-GB root capture a way home and gave
# `measure` -- whose 4-40 KB receipt IS the submission object -- none at all.
# A container-native run ended by naming a path inside a pod-scoped volume, on
# a provider whose REST API serves no logs and no files and whose image runs no
# sshd. stdout is unconditional because it is the only channel every platform
# has; file: and https: are for the caller who can read one.
t "result sinks: the answer gets off the box (T26: R1-R27)" \
                                           0 python3 bin/selftest_result_sink.py
# T14. The progress meter. Both capture engines print one line at the start and
# one at the end, which on the streaming lane is a 2-3 hour silence in a stage
# log that looks exactly like a hang. The rungs that matter are the ones about
# the OUTPUT SHAPE: every stage runs `nohup ... > stage-<name>.log`, so a
# carriage-return spinner writes one megabyte-long line, and the meter must fall
# back to throttled newline-terminated lines when stdout is not a TTY. P11 is
# the drift guard: a engines/tools module an engine imports but BUNDLE.txt does not
# list is an ImportError at the start of the measure stage, i.e. after the
# bootstrap, the 200 GB fetch and the panel have all been paid for.
t "progress meter: file vs TTY, throttle, wiring, bundle, stall counter (T14)" \
                                           0 python3 bin/selftest_progress.py
t "stack fingerprint (T9: deterministic, engine-absent, MPS/CUDA-absent)" \
                                           0 python3 bin/selftest_stackprint.py
# The money chokepoint (T11). `jl list` is the only thing that answers "is this
# instance alive?", and four call sites spend or leak money on an empty answer.
# Driven against a stub jl: no network, no account, no rental.
t "jl list envelope (T11: an unreadable answer is not an empty account)" \
                                           0 python3 bin/selftest_jlapi.py
# The shell guards (T10). Every prerequisite/cleanliness/budget/pace guard in the
# rental scripts, driven against real fixtures -- a scratch git repo, a truncated
# patch series, malformed BUDGET_USD values. These guards used to be `A && B`
# lists, which `set -e` exempts, so they asserted nothing.
t "shell guards (T10: SH-02/03/14/19/21/23 + SEC-01 fixtures)" \
                                           0 bash bin/selftest_shell_guards.sh
# T16. The stage driver, EXECUTED. Two of its eleven stages were ever run by a
# test; the rest were "covered" by grepping the file for a substring -- which is
# the shape of test all four of the expensive stage bugs walked straight through
# (QP_PIPELINE_ROOT hardcoded in `measure`, then again in `score`, the roots
# never exported, and jqget printing a JSON null as the string "None"). This
# drives every stage under a real bash with argv-logging stubs and asserts, per
# stage: roots come from the environment, a missing input fails CLOSED, the
# .done marker appears only on success, and no argument names a path the
# environment did not supply. Verified by reintroducing all four bugs.
t "stage driver: every stage executed (T16)" \
                                           0 python3 bin/selftest_stage_measure.py
# The class that GUARANTEES a rented GPU is destroyed had no test at all, which
# is how CLI-01 (an API outage read as "destroyed") and CLI-02(b) (a teardown
# that marks itself done before doing anything) survived a full review cycle.
# Fake `jl`, no rental, no network.
t "teardown guarantees (CLI-01/02b/11, CC-07, L52)" \
                                           0 "$PY" bin/selftest_teardown.py
t "zero-floor identity (T4; SKIPs inside when numpy/torch absent)" \
                                           0 "$PY" bin/selftest_zero_floor.py
# The three-step fidelity dataset tool: format+seals+refusals, the comparator's
# numerics, and the HF card annotation. All three run offline on the system
# python3; the card case's live-Hub axis is the one networked check and it
# SKIPs (loudly) under --offline.
t "fidelity dataset format, seals and refusals (T6)" \
                                           0 python3 bin/selftest_fidelity_dataset.py
t "fidelity comparator known answers + exact self-compare (T8)" \
                                           0 "$PY" bin/selftest_fidelity_compare.py
t "fidelity card annotation, 3 axes (T7)" \
                                           0 python3 bin/selftest_fidelity_card.py
# The replay backend (T12). M1 measured the comparison at 10.8x the capture it
# consumes because the head matmul ran in numpy on the CPU with the GPU at 0%.
# --replay-device moves it; these rungs hold the line that moving it must not
# move the floor, and that the backend is named on every receipt rather than
# swapped silently. Runs on any interpreter with numpy; the torch rungs SKIP
# loudly without torch.
t "replay backend: floor is backend-independent, backend is named (T12: R1-R9)" \
                                           0 "$PY" bin/selftest_replay_device.py
# The portable capture engine's own battery (CAPTURE-03's four ways a load can
# hand back a model that is not the artifact's, the --device-map path, the
# panel receipt).  It existed and was never wired in here, so nothing ran it.
t "hf-transformers capture engine: load-report guards, refusals (A1-A22)" \
                                           0 "$PY" bin/selftest_hf_capture.py
# The truncation fetcher, against three real checkpoint key layouts (GLM-5.3's
# `model.layers.N.`, DeepSeek-V4's bare `layers.N.` + `mtp.` subtree, MiniMax-M3's
# VL `language_model.model.layers.N.` with the layer count under `text_config`).
# F4 is the one that matters: a layer regex matching NOTHING used to make the
# tool plan a fetch of the WHOLE checkpoint and log it as a truncation.
# Offline -- the HTTP fetcher is replaced by one reading a synthetic repo.
t "truncation fetcher: three key layouts, config surgery, refusals (F1-F10)" \
                                           0 "$PY" bin/selftest_fetch_truncated.py
# The layer-outer/window-inner schedule and its streaming residency.  L1-L3 are
# the deliverable in three assertions: the new loop order must produce the SAME
# capture_content_digest as the old one, and the default must stay the old one.
t "layer-outer schedule: bit-identity, streamed loader, holes guard (L1-L14)" \
                                           0 "$PY" bin/selftest_layer_outer.py
# The joint fidelity standard (protocol hash, R0 canary, 13-gram overlap scan,
# clustered SE + BCa block bootstrap, sigma_run in quadrature, McNemar,
# percentile guards, and the registry's JOINT-* invariants). Its known-answer
# cases reproduce brandonmusic's published endpoints from his per-window means;
# its ORACLE cases call his own kld_eval when PYTHONPATH points at it and SKIP
# loudly otherwise; its FIRE cases prove every gate rejects bad input.
t "joint standard (known answers, canary FIRE cases, registry invariants)" \
                                           0 "$PY" bin/selftest_joint_standard.py
# Every anchored number in docs/PROTOCOL-ALIGNMENT.md and docs/ALIGNMENT-REPLY.md
# re-derived from the committed receipts, plus the two model cards' scope
# disclosure. Nothing tied those documents to the data before this, and five
# wrong numbers reached them through the gap; this is the gate that closes it.
t "doc-vs-receipt: every alignment/card number re-derived" \
                                           0 python3 bin/check_doc_numbers.py
# CX1/CX2 (contributor-experience review 2026-08-31). The README's support
# matrix is GENERATED from bin/engines.json (render-drift fails, same pattern
# as registry/README) and every fenced command in the README's recipe sections
# parses against the real CLI it names -- hand-written support claims and
# unexecutable recipes are the two defects these pin down.
t "support matrix: generated, current, single-source (CX1)" \
                                           0 python3 bin/selftest_support_matrix.py
t "README recipes: every fenced command parses; local limits stated (CX2)" \
                                           0 python3 bin/selftest_readme_recipes.py
# CX3: the one-command readiness probe the quickstart points at. Read-only and
# offline; exit 0 means "a $0.00 dry-run can run from here", which is true of
# any intact checkout -- missing credentials are warnings, not failures.
t "fidelity-doctor: readiness probe runs clean (CX3)" \
                                           0 python3 bin/fidelity-doctor
t "stream_score ladder rungs g,h,i,j,k,l (teacher role / preview refusal / \
sampling / receipt stability / source dispatch / decode-cache identity)" \
                                           0 python3 engines/tools/stream_score_selftest.py --only g,h,i,j,k,l
# The three helpers every claim in docs/ARCHITECTURE-DETERMINISM.md rests on:
# the partition explanation test (which must reject BOTH failure directions --
# same attribute different result, and same result different attribute), the
# fp64 KLD estimator (exactly 0.0 on identical inputs, finite on extreme
# logits), and the float32 ULP encoder. Not hypothetical coverage: cases [13]
# and [14] caught a sign error in the ULP encoder that scored +0.0 against -0.0
# as 2^32 ULPs and made every quoted ULP figure wrong. Needs numpy only.
if have_module "$VPY" numpy; then
t "arch-determinism analysis helpers (partition/KLD/ULP, 14 cases)" \
                                           0 "$VPY" reports/arch-determinism/selftest_arch_determinism.py
else
  s "arch-determinism analysis helpers" "numpy not importable"
fi
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
dry-runs, registry adapter)"               0 "$MLXPY" engines/tools/selftest_mlx_offline.py
else
  s "mlx surface offline" "torch/safetensors not importable under $MLXPY -- export FIDELITY_PYTHON"
fi
if have_module "$PY" torch; then
  t "gguf surface offline (dequant vs gguf-py, census, MLA audit, refusals)" \
                                           0 "$PY" engines/tools/selftest_gguf_offline.py
else
  s "gguf surface offline" "torch not importable by $PY"
fi
# The NVFP4 weight-decode surface: dequant proven against compressed-tensors on
# real ranged-fetched tensors, the name census closed against both repos' real
# indexes, and the registry adapter exercised. Needs torch (float8 + MPS); its
# two conditional rungs (live compressed-tensors, stream_score --dry-run) print
# their own SKIP line rather than failing, so this stays one case either way.
# TPY: the first interpreter that actually has torch.  Selecting $VPY and then
# SKIPPING when the venv has no torch meant these rungs had never run on this
# machine at all -- a skip is a verdict, and "not run" is not one of the good
# ones.
TPY="$VPY"
have_module "$TPY" torch || TPY="$PY"
have_module "$TPY" torch || TPY=""
if [ -n "$TPY" ]; then
  t "nvfp4 surface offline (decode vs compressed-tensors, census, registry adapter)" \
                                           0 "$TPY" engines/tools/selftest_nvfp4_offline.py \
                                             ${QP_PIPELINE_ROOT:+--pipeline-root "$QP_PIPELINE_ROOT"}
else
  s "nvfp4 surface offline" "no torch in $VPY or $PY"
fi

# tr3 surface: the SEALED TR3-published reader. The fixture carries the real
# 1,618 official non-routed names and the real 150,226-name index shape, so the
# seal arithmetic is exercised against the same numbers the live release
# satisfies. The MCG decode-parity rung self-skips without quant_pipeline.
if [ -n "$TPY" ]; then
  t "tr3 surface offline (seal recompute, tampers, scope, decode == exl3hf)" \
                                           0 "$TPY" engines/tools/selftest_tr3_offline.py
else
  s "tr3 surface offline" "no torch in $VPY or $PY"
fi

echo "== cloud planner (NETWORK, ACCOUNT; --dry-run creates nothing) =="
# This target is 'tr3-published'. The SEALED-EP8 lane still has no reader for
# it: that engine resolves a packed_root out of the materialization receipt and
# requires the payload store to be present, which a third-party repo never
# publishes. (The check that was supposed to catch this -- hfmeta.sniff_surface's
# packed_root trap -- is guarded by `if info.surface == "packed"`, and this repo
# carries exl3-mcg-storage-abi.json, so it classifies as tr3-published and
# routes around the trap. Asserting rc=0 there asserted that a rental which
# cannot possibly succeed would be approved.)
#
# The STREAMING lane gained a reader in M2 (stream_score --source tr3 via
# tr3_surface), so its case flipped from "refuses" to "plans, having recomputed
# the release's published seal for free". The refusal it now produces is the
# max-runtime one below, which is a different property.
# --skip-registry-check everywhere below: these cases test the PLANNER's own
# refusals; the registry front gate (tested separately) would otherwise answer
# "already measured" first, because this target has published rows.
cloud_isolated() {
  local py="$PY"
  case "$py" in
    */*) ;;
    *) py="$(command -v "$py")" || return 1 ;;
  esac
  mkdir -p "$TMP/no-reaper-home"
  HOME="$TMP/no-reaper-home" PATH="/usr/bin:/bin" \
    "$py" bin/measure_cloud.py "$@"
}
cloud_refusal_check() {
  local log="$1" needle="$2"; shift 2
  cloud_isolated "$@" >"$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ] && grep -Fq -- "$needle" "$log"; then
    return 0
  fi
  tail -20 "$log"
  return 1
}
t "sealed-ep8 refuses specifically because tr3-published has no reader" 0 \
  cloud_refusal_check "$TMP/c1.log" "has no --profile" \
    --model "$MODEL" --panel "$PANEL" --lane sealed-ep8 --spot \
    --max-runtime 30h --i-accept-leak-risk --skip-registry-check \
    --dry-run --out "$TMP/c1"
t "streaming PLANS a tr3-published target (reader landed in M2)" 0 \
  cloud_isolated --model "$MODEL" --panel "$PANEL" \
    --lane streaming --gpu H200 --spot --max-runtime 12h --i-accept-leak-risk \
    --skip-registry-check --dry-run --out "$TMP/c2"
grep -q '"seal_verification"' "$TMP/c2/plan.json" \
  && grep -q '"checks_passed": 12' "$TMP/c2/plan.json" \
  && t "the tr3 plan carries a 12/12 seal recompute" 0 true \
  || t "the tr3 plan carries a 12/12 seal recompute" 0 false
t "a 12h run without a reaper specifically refuses its teardown risk" 0 \
  cloud_refusal_check "$TMP/c3.log" "no teardown backstop for a 12h run" \
    --model "$MODEL" --panel "$PANEL" --lane streaming --gpu H200 --spot \
    --max-runtime 12h --skip-registry-check --dry-run --out "$TMP/c3"
t "a 2h streaming plan specifically refuses its 3h+ workload" 0 \
  cloud_refusal_check "$TMP/c4.log" \
    "--max-runtime 2h is shorter than the estimated work" \
    --model "$MODEL" --panel "$PANEL" --lane streaming --gpu H200 --spot \
    --max-runtime 2h --i-accept-leak-risk --skip-registry-check \
    --dry-run --out "$TMP/c4"
# A dry-run refusal must carry its REMEDY, not just its complaint. Five of the
# six would-refuse sites used to print `refusal.reason` alone, so the advice --
# which flag to raise, which files a new engine profile needs -- reached nobody,
# in the one mode the docs tell a newcomer to start with.
#
# $MODEL is tr3-published, which the sealed-ep8 lane has no profile for, so the
# profile would-refuse fires deterministically here. The teardown refusal has
# its own empty-HOME test above; this invocation accepts that risk explicitly,
# so a host reaper cannot add or remove the condition being inspected.
# The exit code is not asserted -- a dry run may go on to hit a HARD refusal
# (no instance capacity today) and exit 1 rather than 3. What is asserted is
# that the advice reached stdout, and that it names a table that exists.
cloud_isolated --model "$MODEL" --panel "$PANEL" \
  --lane sealed-ep8 --spot --max-runtime 30h --i-accept-leak-risk \
  --skip-registry-check --dry-run --out "$TMP/c3b" >"$TMP/c3b.log" 2>&1 || true
if grep -q 'WOULD REFUSE (real run): .*has no --profile' "$TMP/c3b.log" \
   && grep -q 'FOUR files must agree' "$TMP/c3b.log" \
   && grep -q 'TR3_PROFILES' "$TMP/c3b.log" \
   && ! grep -q 'TR3_PUBLISHED_PROFILES' "$TMP/c3b.log"; then
  t "a dry-run WOULD-REFUSE prints its remedy, naming tables that exist" 0 true
else
  t "a dry-run WOULD-REFUSE prints its remedy, naming tables that exist" 0 false
  sed 's/^/         /' "$TMP/c3b.log" | grep -i "would refuse" | head -4
fi
# and the named tables must actually be there
"$PY" - <<'PYEOF' && t "every PROFILE_TABLE_NAMES entry exists in stream_score.py" 0 true \
  || t "every PROFILE_TABLE_NAMES entry exists in stream_score.py" 0 false
import ast, sys
sys.path.insert(0, "bin")
import measure_cloud as MC
src = open("engines/tools/stream_score.py", encoding="utf-8").read()
names = {t.id for n in ast.parse(src).body if isinstance(n, ast.Assign)
         for t in n.targets if hasattr(t, "id")}
missing = sorted(set(MC.PROFILE_TABLE_NAMES.values()) - names)
sys.exit(1 if missing else 0)
PYEOF
t "cloud front gate: already-measured artifact answers for \$0.00" 0 \
  "$PY" bin/measure_cloud.py --model "$MODEL" --panel "$PANEL" \
    --lane sealed-ep8 --spot --max-runtime 30h --i-accept-leak-risk \
    --dry-run --out "$TMP/c5"

echo "== local planner (NETWORK) =="
# Capacity checks are facts about the host running this battery. A valid planner
# therefore exits 0 on a large volume and 3 with ONLY a disk-capacity refusal on
# a small one. Treating a developer's 400 GB mount as a test prerequisite made
# these hardware-planning checks platform-dependent.
local_plan_check() {
  local mode="$1" out="$2"; shift 2
  "$PY" bin/measure_local.py "$@" --out "$out"
  local rc=$?
  PLAN_RC="$rc" PLAN_MODE="$mode" "$PY" - "$out/local-plan.json" <<'PYEOF'
import json, os, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
rc = int(os.environ["PLAN_RC"])
blockers = doc.get("would_refuse") or []
if rc != (3 if blockers else 0):
    raise SystemExit("return code %d disagrees with blockers %r" % (rc, blockers))
for key in ("device", "lane", "storage_need", "disk_free_bytes"):
    if key not in doc:
        raise SystemExit("plan omitted %s" % key)
if os.environ["PLAN_MODE"] == "known-fit":
    if "memory_plan" not in doc:
        raise SystemExit("known-fit simulated device produced no memory plan")
    other = [b for b in blockers if not b.startswith("not enough disk:")]
    if other:
        raise SystemExit("known-fit device had non-disk blockers: %r" % other)
PYEOF
}
t "this machine auto device reports an internally consistent plan" 0 \
  local_plan_check host "$TMP/l1" --artifact "$MODEL" --panel "$PANEL" \
    --estimate-only --skip-registry-check
t "RTX 5090 32GB honours a 30GB budget; host disk gate remains real" 0 \
  local_plan_check known-fit "$TMP/l2" --artifact "$MODEL" --panel "$PANEL" \
    --simulate-device "RTX 5090:32" --vram-budget 30 \
    --estimate-only --skip-registry-check
t "128GB Mac fits; host disk gate remains real" 0 \
  local_plan_check known-fit "$TMP/l3" --artifact "$MODEL" --panel "$PANEL" \
    --simulate-device "Mac Studio:128::unified" \
    --estimate-only --skip-registry-check
local_small_device_check() {
  "$PY" bin/measure_local.py --artifact "$MODEL" --panel "$PANEL" \
    --simulate-device "GTX 1650:4" --skip-registry-check --out "$TMP/l4" \
    >"$TMP/l4.log" 2>&1
  local rc=$?
  [ "$rc" = 3 ] \
    && grep -q "no schedule fits a .* GB budget" "$TMP/l4.log" \
    && grep -q "minimum viable budget" "$TMP/l4.log"
}
t "4GB card has a capacity refusal independent of host disk" 0 \
  local_small_device_check
t "--simulate-device rejects non-finite capacity at argparse" 2 \
  "$PY" bin/measure_local.py --artifact x/y --panel z \
    --simulate-device "invalid:nan" --estimate-only
t "--simulate-device rejects overflowing capacity at argparse" 2 \
  "$PY" bin/measure_local.py --artifact x/y --panel z \
    --simulate-device "invalid:1e308" --estimate-only
t "--simulate-device cannot authorize execution on different hardware" 3 \
  "$PY" bin/measure_local.py --artifact x/y --panel z \
    --simulate-device "RTX 5090:32" --execute
t "--kld-device mps is refused (no fp64 on MPS)" 3 "$PY" bin/measure_local.py --artifact x/y --panel z --kld-device mps
t "engine probe (all five lanes pinned, flags found)" 0 "$PY" bin/measure_local.py --probe-engines
local_preflight_check() {
  "$PY" - "$TMP" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "bin")
from fidelity.engines import load_engines, preflight
root = Path(sys.argv[1])
problems = preflight(
    load_engines()["local-cuda-budget"], suite_root=Path("."),
    pipeline_root=str(root / "missing-pipeline"),
    teacher_dir=root / "missing-teacher")
missing = [row.get("missing", "") for row in problems]
if not any("quant_pipeline package under --pipeline-root" in text for text in missing):
    raise SystemExit("preflight did not name the missing pipeline: %r" % missing)
if not any("teacher tree with a sealed capture receipt" in text for text in missing):
    raise SystemExit("preflight did not name the missing teacher: %r" % missing)
if not all(row.get("remedy") for row in problems):
    raise SystemExit("preflight returned a problem without a remedy: %r" % problems)
PYEOF
}
t "--execute preflight accumulates missing inputs with remedies" 0 local_preflight_check

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
# The STAT-01/STAT-17 arithmetic, on the real 42 published cells. Kept OUT of
# `registry/ make check` on purpose: it needs numpy, and `make check` is the one
# command a contributor must be able to run on a stock interpreter with no
# installs. Skipped rather than failed where numpy is absent, for the same reason.
if "$VPY" -c "import numpy" >/dev/null 2>&1; then
  t "per-domain interval: coverage, seeds, regenerable old endpoints" 0 \
    "$VPY" registry/tools/selftest_stat01_reseed.py
else
  s "per-domain interval selftest" "numpy is not importable by $VPY"
fi
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
# SH-19. These two staging steps were BARE: selftest_all.sh runs under `set -u` with
# no `set -e`, so a failure here (BUNDLE.txt renamed, a bad import, a full disk)
# incremented neither pass nor fail. The run still printed "N passed, 0 failed" and
# exited 0 while the round-trip it exists to prove had never been staged. A step whose
# failure is invisible is worse than no step: it launders absence into evidence.
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
stage_rc=$?
staged_n=0
[ -d "$TMP/fs/bin" ] && staged_n=$(find "$TMP/fs" -type f | wc -l | tr -d ' ')
if [ "$stage_rc" = 0 ] && [ "${staged_n:-0}" -gt 0 ]; then
  printf '  PASS  %s\n' "bundle staging ($staged_n files from bin/BUNDLE.txt)"; pass=$((pass+1))
else
  printf '  FAIL  %s (rc=%s, %s files staged)\n' "bundle staging" "$stage_rc" "${staged_n:-0}"
  fail=$((fail+1))
fi
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
fixture_rc=$?
if [ "$fixture_rc" = 0 ] && [ -s "$TMP/fs/job.json" ] && [ -s "$TMP/fs/metrics.json" ]; then
  printf '  PASS  %s\n' "round-trip fixture written (job.json + metrics.json)"; pass=$((pass+1))
else
  printf '  FAIL  %s (rc=%s)\n' "round-trip fixture" "$fixture_rc"; fail=$((fail+1))
fi
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
      "$FIXPY" engines/tools/stream_score_selftest.py --fixture "$FIXTURE_PATH" \
        --only b,c,f,g,h,i,j
  else
    s "fixture ladder" "fixture fetch failed (network?) -- run bin/fixture manually"
  fi
fi

echo
echo "selftest_all: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
