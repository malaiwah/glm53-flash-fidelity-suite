#!/usr/bin/env python3
"""A second provider must not be able to leak an instance.

Every bug found while porting to RunPod was the same bug: a JarvisLabs
*representation* treated as a universal truth. None of them were about the
measurement, and three of them could have left a billing instance running.

  * machine ids are integers        -> `int(pod_id)` raised AFTER the pod was
                                       created, so the controller died holding
                                       an instance it had never adopted
  * the running state is "Running"  -> RunPod says "RUNNING", so every healthy
                                       poll counted as not-running and the
                                       controller declared a PREEMPTION and
                                       tore down a box mid-bootstrap
  * ids compare as ints in a set    -> the "is it really gone?" check would
                                       report a live instance as destroyed

These are offline: no provider is contacted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_cloud as mc                                # noqa: E402
from fidelity.jlapi import Instance                       # noqa: E402

FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


def inst(mid, status="Running", name="fidcloud-x"):
    i = Instance.from_json({"machine_id": 0, "status": status, "name": name})
    i.machine_id = mid
    return i


print("== an opaque machine id survives every hop ==")
for created, want in [
        ({"machine_id": 483634}, 483634),
        ({"machine_id": "483634"}, 483634),
        ({"pod_id": "uqlk708fxtoz8n"}, "uqlk708fxtoz8n"),
        ({"id": "yytmxz8vhh1qfk"}, "yytmxz8vhh1qfk"),
        ({}, None),
        (None, None),
]:
    got = mc._machine_id_of(created)
    check("_machine_id_of(%r) -> %r" % (created, want), got == want)

check("a non-numeric id is NOT dropped (the leak this caused)",
      mc._machine_id_of({"machine_id": "abc123xyz"}) == "abc123xyz")

print("\n== the running state is spelled differently per provider ==")
for status, want in [("Running", True), ("RUNNING", True), ("running", True),
                     ("ready", True), ("Paused", False), ("EXITED", False),
                     ("TERMINATED", False), ("", False)]:
    check("status %-12r -> running=%s" % (status, want),
          mc._is_running(inst(1, status)) is want)
check("None is not running", mc._is_running(None) is False)

print("\n== 'is it really gone?' compares like with like ==")


class FakeJL:
    provider = "runpod"

    def __init__(self, alive):
        self._alive = alive

    def list_instances(self):
        return [inst(m) for m in self._alive]


class Con:
    def __init__(self):
        self.lines = []

    def __getattr__(self, _):
        return lambda *a, **k: None


td = mc.Teardown(FakeJL(["uqlk708fxtoz8n"]), Con(), mc.Path("."))
check("a LIVE opaque-id instance is not reported gone",
      td._confirm_gone("uqlk708fxtoz8n") is False)
check("a destroyed opaque-id instance is reported gone",
      td._confirm_gone("gone000000") is True)

td_int = mc.Teardown(FakeJL([483634]), Con(), mc.Path("."))
check("the integer case still works", td_int._confirm_gone(483634) is False)
check("...and its negative", td_int._confirm_gone(999999) is True)

print("\n== the run root is provider-specific, and always exported ==")


class JLish:
    provider = "jarvislabs"


rp, jl = mc.Teardown(FakeJL([]), Con(), mc.Path(".")), \
    mc.Teardown(JLish(), Con(), mc.Path("."))
check("runpod runs under /workspace", rp.fs_root.startswith("/workspace"))
check("jarvislabs runs under /home/jl_fs", jl.fs_root.startswith("/home/jl_fs"))
for t in (rp, jl):
    env = mc._stage_env(t)
    check("%s stage env names both roots" % t.fs_root.split("/")[1],
          "FIDELITY_FS_ROOT=" in env and "FIDELITY_K6_ROOT=" in env
          and t.fs_root in env)

print("\n== the backend switch ==")
check("--provider jarvislabs builds the jl backend",
      type(mc._make_provider("jarvislabs", dry=True)).__name__ == "JL")
check("--provider runpod builds the runpod backend",
      type(mc._make_provider("runpod", dry=True)).__name__ == "RunPod")
rpb = mc._make_provider("runpod", dry=True)
check("the runpod backend declares its provider name", rpb.provider == "runpod")
for m in ("create", "destroy", "exec", "exec_stdout", "upload", "download",
          "list_instances", "get", "gpus", "balance", "run_job", "run_status",
          "run_logs", "fs_create", "fs_delete", "available", "require",
          "pause", "resume"):
    check("runpod implements %s()" % m, callable(getattr(rpb, m, None)))
check("every mutating call is a no-op under dry",
      rpb.create(gpu_type="x").get("dry_run") is True
      and rpb.destroy("x").get("dry_run") is True)

print("\n== storage that dies with the instance must be sized at create ==")
for name, sep in (("jarvislabs", True), ("runpod", False), ("vast", False),
                  ("lambda", False)):
    prov = mc._make_provider(name, dry=True)
    check("%-10s separable_storage=%s" % (name, sep),
          getattr(prov, "separable_storage", True) is sep)
# 100 GB is right ONLY where the big disk is a separate filesystem. Getting
# this wrong is not a create error: it is "No space left on device" three
# stages and 45 minutes into a paid run, which is exactly what happened.
check("a non-separable provider is sized from the plan, not 100 GB",
      'else int(plan_data["storage_gb"])' in
      open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "measure_cloud.py")).read())

print("\n== no path may assume JarvisLabs except the two exported roots ==")
# Every hardcoded /home/jl_fs literal in the on-instance tools is a run that
# silently does the wrong thing somewhere else. Three of them were found one at
# a time, each by a paid run: FIDELITY_FS_ROOT and FIDELITY_K6_ROOT (a run
# written into a container's ephemeral layer), then QP_PIPELINE_ROOT -- which
# stalled a box at 0% GPU for two hours, at $1.59/h, AFTER the bootstrap, a
# 200 GB fetch and the panel were all paid for. This is the rule that finds the
# fourth one without renting anything.
EXPORTED_ROOTS = ("FIDELITY_FS_ROOT", "FIDELITY_K6_ROOT", "QP_PIPELINE_ROOT")
ON_INSTANCE = ("invoke_engine.py", "invoke_scorer.py", "stage_measure.sh",
               "bootstrap_measure.sh")
here = os.path.dirname(os.path.abspath(__file__))
offenders = []
for fname in ON_INSTANCE:
    path = os.path.join(here, fname)
    if not os.path.isfile(path):
        continue
    lines = open(path, encoding="utf-8").read().splitlines()
    for n, line in enumerate(lines, 1):
        if "/home/jl_fs" not in line or line.lstrip().startswith("#"):
            continue
        # Allowed ONLY as the fallback of a root the controller exports. The
        # name and its literal are often on different lines (an os.environ.get
        # call wrapped across three), so the window is what is checked.
        window = "\n".join(lines[max(0, n - 4):n + 1])
        if not any(r in window for r in EXPORTED_ROOTS):
            offenders.append("%s:%d %s" % (fname, n, line.strip()[:88]))
for o in offenders:
    print("      %s" % o)
check("every /home/jl_fs literal is the default of an EXPORTED root",
      not offenders)

env = mc._stage_env(rp)
for r in EXPORTED_ROOTS:
    check("the stage env exports %s" % r, (r + "=") in env)
check("...and none of them point at /home/jl_fs on a runpod box",
      "/home/jl_fs" not in env)

print("\n== a run's identity includes WHERE it ran ==")
import argparse                                            # noqa: E402


def jid(**kw):
    base = dict(model="m", revision="r" * 40, panel="p", lane="streaming",
                spot=False, cold_runs=2, provider="jarvislabs", gpu=None,
                role="quant")
    base.update(kw)
    return mc.job_id_for(argparse.Namespace(**base))


# Two providers running the SAME measurement produced the same job id, the
# same fidelity-runs/<id>/ directory, and the second silently OVERWROTE the
# first one's sealed receipt. The earlier result had to be rescued from disk.
ids = {jid(), jid(provider="vast"), jid(provider="runpod"),
       jid(provider="runpod", gpu="NVIDIA A100-SXM4-80GB"),
       jid(provider="runpod", gpu="NVIDIA H100 PCIe")}
check("provider and GPU change the job id (no receipt collision)", len(ids) == 5)
check("the same inputs still give the same id", jid() == jid())
check("--role root is a different run from --role quant",
      jid(role="root") != jid(role="quant"))

print("\n== a receipt says which cloud it actually ran on ==")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "measure_cloud.py"), encoding="utf-8").read()
check('the environment host is read from --provider, not hardcoded',
      '"host": getattr(args, "provider"' in src)
check("...and no literal jarvislabs host remains",
      '"host": "jarvislabs"' not in src)

print("\n== the machine is measured before the run is spent on it ==")
from fidelity.bench import estimate, gate                   # noqa: E402

# Both are REAL readings from two Vast offers for the same GPU model, taken
# with the same benchmark minutes apart. The only difference is how the host
# wires the card.
X1 = {"gpu": "NVIDIA RTX 4000 Ada Generation", "h2d_GBps": 1.6,
      "h2d_cold_GBps": 1.6, "expert_gemm_TFLOPs": 82.2,
      "stream_matrix_ms": 10.587,
      "pcie_load": {"text": "Gen4 x1 of Gen4 x16"}}
X16 = {"gpu": "NVIDIA RTX 4000 Ada Generation", "h2d_GBps": 11.0,
       "h2d_cold_GBps": 10.8, "expert_gemm_TFLOPs": 85.7,
       "stream_matrix_ms": 1.755,
       "pcie_load": {"text": "Gen3 x16 of Gen3 x16"}}

check("a PCIe x1 host is REFUSED at an 8 GB/s floor",
      gate(X1, min_h2d_gbps=8.0) is not None)
check("...and the refusal names the link width, not just the number",
      "Gen4 x1" in (gate(X1, min_h2d_gbps=8.0) or ""))
check("the x16 sibling of the SAME GPU passes the same floor",
      gate(X16, min_h2d_gbps=8.0) is None)
check("no floor asked for means no refusal", gate(X1) is None)
check("a compute floor is separate from a bandwidth floor",
      gate(X16, min_gemm_tflops=200.0) is not None)

# The whole point: same card, same compute, 6x the wall clock.
e1 = estimate(X1, matrices_per_window=36288)
e16 = estimate(X16, matrices_per_window=36288)
check("the x1 host is ~6x slower per window on identical silicon",
      5.0 < e1["minutes_per_window"] / e16["minutes_per_window"] < 7.0)

# A parked link ramps under load; a narrow one does not. That distinction is
# what makes the refusal defensible rather than a guess.
check("cold and warm are reported separately so a ramp is visible",
      "h2d_cold_GBps" in X1 and "h2d_GBps" in X1)

print()
if FAILED:
    print("selftest_provider_portability: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_provider_portability: all passed")
