#!/bin/bash
# Autonomous VM-side progress heartbeat -> ntfy.sh, independent of the control session.
#   notify_heartbeat.sh [interval_seconds]
# Stops itself when the package stage completes or when logs/heartbeat.stop exists.
ROOT=/home/ubuntu/glm53
# Notification endpoint: opt-in via NTFY_URL (or QP_NTFY_URL). Unset = no
# notifications. A public repo must not pin its operator's channel.
NTFY_URL="${NTFY_URL:-${QP_NTFY_URL:-}}"
INTERVAL="${1:-900}"
mkdir -p "$ROOT/logs"

while true; do
  BODY=$(python3 - <<'EOF'
import json, pathlib, shutil, subprocess, time
root = pathlib.Path("/home/ubuntu/glm53")
lines = []
state = "(no stage yet)"
p = root / "logs/stage.state"
if p.is_file():
    state = p.read_text().strip()
lines.append(f"stage: {state}")
for variant in ("bf16", "fp8"):
    m = root / f"captures/{variant}/capture-manifest.json"
    if m.is_file():
        try:
            man = json.loads(m.read_text())
            done = man.get("contexts", 0)
            want = man.get("expected_contexts", 0)
            el = man.get("elapsed_sec", 0)
            if done and el:
                pace = el / done
                eta_min = (want - done) * pace / 60 if want else 0
                lines.append(f"{variant}: {done}/{want} ctx, {pace:.2f}s/ctx, ETA {eta_min:.0f} min")
            else:
                lines.append(f"{variant}: {done}/{want} ctx")
        except Exception as e:
            lines.append(f"{variant}: manifest unreadable ({e})")
du = shutil.disk_usage("/home")
lines.append(f"disk: {du.used/1e9:.0f}/{du.total/1e9:.0f} GB used")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20)
    utils = [u.strip() for u in smi.stdout.strip().split("\n") if u.strip()]
    lines.append("gpu util %: " + " ".join(utils))
except Exception:
    pass
for q in ("qualify-bf16", "qualify-fp8"):
    f = root / f"out/{q}.json"
    if f.is_file():
        try:
            r = json.loads(f.read_text())
            lines.append(f"{q}: KLD(live||replayed)={r['mean_kld_live_vs_replayed']:.2e} top1={r['top1_agreement']:.4f}")
        except Exception:
            pass
r = root / "out/report-fp8-vs-bf16.json"
if r.is_file():
    try:
        rep = json.loads(r.read_text())
        lines.append(f"FP8-vs-BF16: mean KLD {rep['token_mean_kld']:.6f}, top1 {rep['top1_agreement']:.4f}")
    except Exception:
        pass
print("\n".join(lines))
EOF
)
  [ -z "$NTFY_URL" ] || \
  curl -s -m 15 -H "Title: GLM53 heartbeat (VM)" -H "Tags: hourglass_flowing_sand" \
       -d "$BODY" "$NTFY_URL" >/dev/null 2>&1 || true

  if [ -f "$ROOT/logs/heartbeat.stop" ] || grep -q '^done:package' "$ROOT/logs/stage.state" 2>/dev/null; then
    [ -z "$NTFY_URL" ] || \
    curl -s -m 15 -H "Title: GLM53 heartbeat stopping" -H "Tags: checkered_flag" \
         -d "package stage complete or stop requested — VM heartbeat ends" "$NTFY_URL" >/dev/null 2>&1 || true
    exit 0
  fi
  sleep "$INTERVAL"
done
