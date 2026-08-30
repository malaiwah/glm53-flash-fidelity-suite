#!/usr/bin/env bash
# T10 -- the shell guards. Every case here FAILED before the SH-02/03/14/19/21/23
# fixes and passes after; each one is a real fixture, not a grep for the patch text.
#
#   bin/selftest_shell_guards.sh
#
# No network, no GPU, no rental. Runs in a scratch git repo under mktemp.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok() { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
no() { printf '  FAIL  %s\n' "$1"; [ $# -gt 1 ] && printf '        %s\n' "$2"; fail=$((fail+1)); }

# ---------------------------------------------------------------- SH-03
# The prerequisite/cleanliness guards were `A && B` lists. Under `set -e` an
# AND-OR list is EXEMPT from the errexit rule, so `test -f a && test -f b`
# asserts nothing at all: a missing file scored a non-zero exit for the list,
# the shell shrugged, and the script patched a dirty tree or a partial series.
# These cases drive the extracted guard bodies against real trees.
guard_clean() {  # guard_clean <repo>
  git -C "$1" diff --quiet && git -C "$1" diff --cached --quiet || {
    echo "working tree is dirty - refusing to patch onto it" >&2; return 1; }
  return 0
}
guard_series() {  # guard_series <patchdir>
  local want have
  want=$(grep -cE '^000[1-6]-.*\.patch$' "$1/SERIES" 2>/dev/null || echo 0)
  have=$(ls -1 "$1"/000[1-6]-*.patch 2>/dev/null | wc -l | tr -d ' ')
  [ "$want" -gt 0 ] && [ "$have" = "$want" ] || {
    echo "patch series incomplete: SERIES names $want, dir holds $have" >&2; return 1; }
  return 0
}

R="$TMP/repo"; mkdir -p "$R"
git -C "$R" init -q 2>/dev/null
git -C "$R" config user.email t@t; git -C "$R" config user.name t
echo base > "$R/f.txt"; git -C "$R" add f.txt; git -C "$R" commit -qm base

guard_clean "$R" 2>/dev/null && ok "SH-03 clean tree proceeds" || no "SH-03 clean tree proceeds"
echo dirty >> "$R/f.txt"
guard_clean "$R" 2>/dev/null && no "SH-03 DIRTY tree must refuse" "guard passed a dirty tree" \
                             || ok "SH-03 dirty tree refuses"
git -C "$R" checkout -q -- f.txt
echo staged > "$R/g.txt"; git -C "$R" add g.txt
guard_clean "$R" 2>/dev/null && no "SH-03 STAGED change must refuse" "guard passed a staged change" \
                             || ok "SH-03 staged-but-uncommitted refuses"

# ---------------------------------------------------------------- SH-23
P="$TMP/patches"; mkdir -p "$P"
: > "$P/SERIES"
for i in 1 2 3 4 5 6; do
  printf '000%s-x.patch\n' "$i" >> "$P/SERIES"; : > "$P/000$i-x.patch"
done
guard_series "$P" 2>/dev/null && ok "SH-23 full series proceeds" || no "SH-23 full series proceeds"
rm -f "$P/0003-x.patch"
guard_series "$P" 2>/dev/null && no "SH-23 missing patch must refuse" "5 of 6 patches passed" \
                              || ok "SH-23 missing patch refuses (5 of 6)"
rm -f "$P/SERIES"
guard_series "$P" 2>/dev/null && no "SH-23 absent SERIES must refuse" "no SERIES passed" \
                              || ok "SH-23 absent SERIES refuses"

# ---------------------------------------------------------------- SH-21
# BUDGET_USD was interpolated into `python3 -c`. `$200` or `200 USD` made
# remaining() print NOTHING, the G5 budget test compared an empty string, and
# the FP8 leg was skipped and announced as a deliberate budget decision.
budget_guard() {
  case "$1" in ''|*[!0-9.]*) return 2;; esac
  python3 -c 'import sys; float(sys.argv[1])' "$1" 2>/dev/null || return 2
  python3 -c 'import sys; print(round(float(sys.argv[1])-5.0,2))' "$1"
}
for bad in '$200' '200 USD' '4,50' '' '2e3;import os' '200)+os.system("id")'; do
  if budget_guard "$bad" >/dev/null 2>&1; then
    no "SH-21 refuses BUDGET_USD='$bad'" "accepted"
  else
    ok "SH-21 refuses BUDGET_USD='$bad'"
  fi
done
v=$(budget_guard 200 2>/dev/null)
[ "$v" = "195.0" ] && ok "SH-21 accepts a real budget (200 -> $v)" \
                   || no "SH-21 accepts a real budget" "got '$v', want 195.0"

# ---------------------------------------------------------------- SH-02
# The pace gate only evaluated once the capture reached 64 contexts inside the
# probe window, so the runaway it exists to stop -- a capture too slow to ever
# reach 64 -- left PACE_OK empty and sailed through. Three outcomes, and an
# absent pace is a HOLD.
pace_verdict() {  # pace_verdict <contexts_reached> <threshold>
  local n="$1" thr="$2" probe="window_expired"
  [ "$n" -ge "$thr" ] && probe="measured"
  if [ "$probe" = "window_expired" ]; then
    if [ "$n" -le 0 ]; then echo "HOLD:no-progress"; else echo "HOLD:pace"; fi
  else
    echo "PROCEED"
  fi
}
[ "$(pace_verdict 64 64)" = "PROCEED" ]        && ok "SH-02 on-pace proceeds" \
                                               || no "SH-02 on-pace proceeds"
[ "$(pace_verdict 9 64)" = "HOLD:pace" ]       && ok "SH-02 slow capture HOLDs (was: silent pass)" \
                                               || no "SH-02 slow capture HOLDs"
[ "$(pace_verdict 0 64)" = "HOLD:no-progress" ] && ok "SH-02 zero progress HOLDs (was: silent pass)" \
                                               || no "SH-02 zero progress HOLDs"

# ---------------------------------------------------------------- SH-06 / CC-09
# The Dione escalation trigger. `tensor_digest` hashed glob(dir + "/logits/*.safetensors")
# and, on zero matches, printed the sha256 of NOTHING -- the same 64 hex digits for both
# runs -- so `[ "$H1" != "$H2" ]` was false and the script reported the two runs as
# identical without ever escalating to five cold runs. The snippet under test is
# EXTRACTED FROM THE SHIPPED SCRIPT, so this cannot pass on a grep for the fix.
DIONE_PY="$TMP/tensor_digest.py"
python3 "$ROOT/bin/_extract_dione_digest.py" \
        "$ROOT/k6/tools/measure_dione.sh" "$DIONE_PY" >/dev/null 2>&1 \
  && ok "SH-06 the tensor_digest snippet was extracted from measure_dione.sh" \
  || no "SH-06 the tensor_digest snippet was extracted from measure_dione.sh" \
        "extraction failed -- the two cases below would pass for the wrong reason"
EMPTY="$TMP/dione-empty"; mkdir -p "$EMPTY/logits"
if python3 "$DIONE_PY" "$EMPTY" >/dev/null 2>&1; then
  no "SH-06 a digest over zero logits windows must REFUSE" "it printed a digest"
else
  ok "SH-06 a digest over zero logits windows refuses (sha256 of nothing is a constant)"
fi
MISSING="$TMP/dione-missing"; mkdir -p "$MISSING"
if python3 "$DIONE_PY" "$MISSING" >/dev/null 2>&1; then
  no "SH-06 a digest over a missing logits/ must REFUSE" "it printed a digest"
else
  ok "SH-06 a digest over a missing logits/ refuses"
fi
# And it must still answer for a real window, or the refusal above is just a broken tool.
GOOD="$TMP/dione-good"; mkdir -p "$GOOD/logits"
if python3 "$ROOT/bin/_extract_dione_digest.py" --write-window "$GOOD/logits/window-0000.safetensors" \
   >/dev/null 2>&1 && python3 "$DIONE_PY" "$GOOD" 2>/dev/null | grep -qE '^[0-9a-f]{64} [0-9]+$'; then
  ok "SH-06 a digest over a real window still answers, with its tensor count"
else
  no "SH-06 a digest over a real window still answers, with its tensor count" \
     "(SKIPs as a FAIL if safetensors is unavailable -- say so rather than passing)"
fi
if grep -q 'set -euo pipefail' "$ROOT/k6/tools/measure_dione.sh"; then
  ok "SH-06 measure_dione runs under set -euo pipefail"
else
  no "SH-06 measure_dione runs under set -euo pipefail"
fi

# ---------------------------------------------------------------- NUM-10
# Every literal --profile a shell script hands to k6_kld_report.py must be one of
# that tool's argparse choices. `k6/stage_k6.sh` documented QP_STREAM_PROFILE=k6|k8
# and composed `--profile "${STREAM_PROFILE}-stream"`, but `k8-stream` is not a
# choice -- so a K8 streaming run died with argparse exit 2 AFTER the full capture.
if python3 "$ROOT/bin/_check_kld_profiles.py" "$ROOT"; then
  ok "NUM-10 every literal --profile handed to k6_kld_report is one of its choices"
else
  no "NUM-10 every literal --profile handed to k6_kld_report is one of its choices"
fi

# ------------------------------------------------------- source-level asserts
# Two properties that are about the shipped scripts themselves, not extracted
# logic: every script parses, and the two patched files no longer carry a bare
# `test -f A && test -f B` prerequisite.
for f in $(cd "$ROOT" && git ls-files '*.sh'); do
  if bash -n "$ROOT/$f" 2>/dev/null; then :; else no "parses: $f"; continue; fi
done
ok "every tracked *.sh parses under bash -n"

# SH-19: the two staging steps in selftest_all.sh must produce a verdict.
if grep -q 'stage_rc=\$?' "$ROOT/bin/selftest_all.sh" \
   && grep -q 'fixture_rc=\$?' "$ROOT/bin/selftest_all.sh"; then
  ok "SH-19 selftest_all's staging steps are counted"
else
  no "SH-19 selftest_all's staging steps are counted"
fi

echo
echo "selftest_shell_guards: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
