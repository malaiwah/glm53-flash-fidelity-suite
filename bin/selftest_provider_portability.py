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

print()
if FAILED:
    print("selftest_provider_portability: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_provider_portability: all passed")
