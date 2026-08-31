#!/usr/bin/env python3
"""Mandatory gates are tri-state: verified / failed / not_checked (P1-11).

WHY THIS EXISTS
---------------
Peer review 2026-08-31: the pre-rental TR3 seal gate caught an import failure
or a Hugging Face error, WARNED, and continued planning -- so a network blip
wore a passing gate's clothes, the dry run ended in "all checks passed", and
a real run would have rented hardware for an artifact whose seal nobody
recomputed.  The offline dry-run fallback likewise skipped every
surface/seal/lane/profile gate and still returned a confident success.

  G1  gate_not_checked on a real run raises a Refusal (blocks the rental).
  G2  gate_not_checked on --dry-run records the tri-state and downgrades the
      plan to estimate-only instead of raising.
  G3  _verify_tr3_seal with the verifier unimportable: not_checked, refusal
      on a real run.
  G4  _verify_tr3_seal with the Hub unreachable: not_checked, refusal on a
      real run; estimate-only on a dry run.
  G5  end-to-end fully-offline --dry-run (Hub unreachable AND provider CLI
      absent -- the field tester's exact scenario): exits 0, the verdict is
      "INCOMPLETE -- this dry run CANNOT AUTHORIZE a run", it never prints
      "all checks passed", it refuses to price (UNPRICEABLE, no $0.00/h, no
      dollar total), and plan.json carries estimate_only=true, the unchecked
      gate list (target-resolution AND instance-pricing), and a null
      point_usd.

G5 forces offline by pointing HF_ENDPOINT at a closed local port and
stripping PATH of the jl CLI; nothing here touches the network or an account.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:10]:
            print("        %s" % line)


def load_mc():
    spec = importlib.util.spec_from_file_location(
        "measure_cloud", str(ROOT / "bin" / "measure_cloud.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Con:
    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        def log(*a):
            self.lines.append("%s %s" % (name, " ".join(str(x) for x in a)))
        return log

    def text(self):
        return "\n".join(self.lines)


def main():
    MC = load_mc()

    # G1: real run -> Refusal.
    plan = {}
    try:
        MC.gate_not_checked(Con(), plan, argparse.Namespace(dry_run=False),
                            "tr3-seal", "network down")
        check("G1 not_checked blocks a real run", False, "no Refusal raised")
    except MC.Refusal as exc:
        check("G1 not_checked blocks a real run",
              "could not be checked" in str(exc)
              and plan["gates"]["tr3-seal"]["status"] == "not_checked",
              str(exc))

    # G2: dry run -> recorded, estimate-only, no exception.
    plan, con = {}, Con()
    MC.gate_not_checked(con, plan, argparse.Namespace(dry_run=True),
                        "tr3-seal", "network down")
    check("G2 dry run records tri-state and downgrades to estimate-only",
          plan["gates"]["tr3-seal"]["status"] == "not_checked"
          and plan.get("estimate_only") is True
          and plan.get("gates_not_checked") == ["tr3-seal"]
          and "ESTIMATE-ONLY" in con.text(), "%s\n%s" % (plan, con.text()))

    # G3: verifier unimportable -> not_checked, refusal on a real run.
    with tempfile.TemporaryDirectory() as td:
        MC.SUITE_ROOT = Path(td)          # no engines/tools here
        sys.modules.pop("tr3_surface", None)
        plan = {}
        try:
            MC._verify_tr3_seal(Con(), "org/repo", "r" * 40, plan,
                                args=argparse.Namespace(dry_run=False))
            check("G3 unimportable verifier refuses a real run", False,
                  "no Refusal")
        except MC.Refusal:
            check("G3 unimportable verifier refuses a real run",
                  plan["gates"]["tr3-seal"]["status"] == "not_checked", plan)
        plan, con = {}, Con()
        MC._verify_tr3_seal(con, "org/repo", "r" * 40, plan,
                            args=argparse.Namespace(dry_run=True))
        check("G3b ...and is a visible estimate-only downgrade on a dry run",
              plan.get("estimate_only") is True
              and plan["gates"]["tr3-seal"]["status"] == "not_checked",
              plan)

    # G4: verifier importable, Hub unreachable -> not_checked.
    MC.SUITE_ROOT = ROOT
    sys.modules.pop("tr3_surface", None)

    def unreachable(*a, **kw):
        raise MC.HFError("network error for x: unreachable (stub)")

    MC.fetch_json = unreachable
    plan = {}
    try:
        MC._verify_tr3_seal(Con(), "org/repo", "r" * 40, plan,
                            args=argparse.Namespace(dry_run=False))
        check("G4 unreachable Hub refuses a real run", False, "no Refusal")
    except MC.Refusal:
        check("G4 unreachable Hub refuses a real run",
              plan["gates"]["tr3-seal"]["status"] == "not_checked", plan)
    plan = {}
    MC._verify_tr3_seal(Con(), "org/repo", "r" * 40, plan,
                        args=argparse.Namespace(dry_run=True))
    check("G4b ...and estimate-only on a dry run",
          plan.get("estimate_only") is True, plan)

    # G5: the whole planner, fully offline (Hub AND provider), dry.
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env["HF_ENDPOINT"] = "http://127.0.0.1:9"     # closed port -> URLError
        env["PATH"] = "/usr/bin:/bin"                 # no jl CLI -> no offers
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "bin" / "measure_cloud.py"),
             "--model", "brandonmusic/GLM-5.3-Flash-tr3-4bpw",
             "--panel", "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
             "--lane", "streaming", "--gpu", "H200", "--spot",
             "--max-runtime", "12h", "--i-accept-leak-risk",
             "--skip-registry-check", "--dry-run", "--out", td],
            capture_output=True, text=True, timeout=600, env=env)
        out = proc.stdout + proc.stderr
        plan_file = Path(td) / "plan.json"
        plan = json.loads(plan_file.read_text()) if plan_file.is_file() else {}
        check("G5 offline dry-run exits 0 (estimate mode may continue)",
              proc.returncode == 0, out[-1500:])
        check("G5b the verdict is INCOMPLETE/cannot-authorize, never "
              "'all checks passed'",
              "INCOMPLETE" in out and "CANNOT AUTHORIZE" in out
              and "all checks passed" not in out, out[-1500:])
        check("G5c plan.json records estimate_only and BOTH unchecked gates",
              plan.get("estimate_only") is True
              and "target-resolution" in (plan.get("gates_not_checked") or [])
              and "instance-pricing" in (plan.get("gates_not_checked") or []),
              json.dumps({k: plan.get(k) for k in
                          ("estimate_only", "gates_not_checked", "gates")},
                         indent=1))
        cost = plan.get("cost_estimate") or {}
        check("G5d no rate exists -> the plan refuses to price "
              "(UNPRICEABLE, no $0.00/h, null point_usd)",
              "UNPRICEABLE" in out and "$0.00/h" not in out
              and cost.get("unpriceable") is True
              and cost.get("point_usd") is None
              and cost.get("rate_per_hour") is None,
              json.dumps(cost, indent=1)[:600])

    print()
    if FAILED:
        print("selftest_seal_gate: %d FAILED" % len(FAILED))
        return 1
    print("selftest_seal_gate: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
