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

# EVERY backend's run root must be WRITABLE BY THE USER IT LOGS IN AS. Lambda
# is the one that is not root: `ubuntu`, on an image whose `/home` is root-owned
# 0755 -- so the `/home/jl_fs` default is EACCES and `mkdir -p
# /home/jl_fs/fidelity/logs` kills the run during the bundle upload, two
# minutes in, on every rental. Observed on a live gpu_1x_gh200 before this
# check existed; the box had already been paid for.
for _p in ("jarvislabs", "runpod", "vast", "lambda"):
    _t = mc.Teardown(mc._make_provider(_p, dry=True), Con(), mc.Path("."))
    _user = getattr(mc._make_provider(_p, dry=True), "ssh_user", "root")
    # A non-root user may only be given a root under ITS OWN home (or a mount
    # the image gives it, which a backend states via `run_base`).
    _ok = _user == "root" or _t.fs_root.startswith("/home/%s/" % _user) \
        or _t.fs_root.startswith("/workspace/")
    check("%s (%s@) gets a run root that user can create: %s"
          % (_p, _user, _t.fs_root), _ok)
    check("...and its engine root sits beside it", _ok and
          _t.engine_root.rsplit("/", 1)[0] == _t.fs_root.rsplit("/", 1)[0])
    _env = mc._stage_env(_t)
    check("...and the stage env exports that root, not the default",
          _t.fs_root in _env and (_user == "root" or "/home/jl_fs" not in _env))
# The root the controller CHOOSES, not just the ones it exports: it used to be
# `/home/jl_fs/glm53-k6` -- one provider and one model baked into the same
# path, on a box that may be measuring MiniMax.
check("neither engine root names a model or a campaign (%s / %s)"
      % (rp.engine_root, jl.engine_root),
      not any(t in (rp.engine_root + jl.engine_root).lower()
              for t in ("glm", "qwen", "minimax", "deepseek", "fruit",
                        "k6", "k8", "tr3")))
for t in (rp, jl):
    env = mc._stage_env(t)
    check("%s stage env names both roots" % t.fs_root.split("/")[1],
          "FIDELITY_FS_ROOT=" in env and "FIDELITY_ENGINE_ROOT=" in env
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
# a time, each by a paid run: FIDELITY_FS_ROOT and FIDELITY_ENGINE_ROOT (a run
# written into a container's ephemeral layer), then QP_PIPELINE_ROOT -- which
# stalled a box at 0% GPU for two hours, at $1.59/h, AFTER the bootstrap, a
# 200 GB fetch and the panel were all paid for. This is the rule that finds the
# fourth one without renting anything.
EXPORTED_ROOTS = ("FIDELITY_FS_ROOT", "FIDELITY_ENGINE_ROOT", "QP_PIPELINE_ROOT")
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

print("\n== the bench waits for readiness on backends that have no socket ==")
from fidelity.bench import wait_ready                       # noqa: E402

# `fidelity-bench --provider jarvislabs` advertises a fourth backend and could
# never use it: run_bench called provider._endpoint(), which only the three SSH
# backends have. The AttributeError landed on the line AFTER create(), so every
# invocation rented a box, failed, and tore it down -- paying for a benchmark
# that produced nothing. Readiness has to be asked for in a way a CLI-transport
# provider can answer.


class SSHish:
    """Has an endpoint, like RunPod / Vast / Lambda."""

    def __init__(self):
        self.asked = 0

    def _endpoint(self, mid, *, wait=900):
        self.asked += 1
        return ("1.2.3.4", 22)

    def get(self, mid):                       # must NOT be consulted
        raise AssertionError("endpoint provider must not fall back to get()")


class CLIish:
    """No endpoint at all, like JarvisLabs: state is the only signal.

    The last state REPEATS rather than falling through to "Running": a fixture
    that quietly becomes ready once its script runs out cannot fail the
    never-comes-up case, and this one did not until it was made to repeat.
    """

    def __init__(self, states):
        self._states = list(states)

    def get(self, mid):
        s = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return inst(mid, s)


s = SSHish()
wait_ready(s, "pod-x", wait=5)
check("an SSH backend is still waited on via its endpoint", s.asked == 1)

check("a CLI backend with no endpoint has no _endpoint to call",
      getattr(mc._make_provider("jarvislabs", dry=True), "_endpoint", None) is None)

wait_ready(CLIish(["Running"]), 483634, wait=5, poll=0)
check("a CLI backend that is already Running returns at once", True)

wait_ready(CLIish(["Launching", "Launching", "Running"]), 483634, wait=5, poll=0)
check("...and one that is still launching is waited for", True)

try:
    wait_ready(CLIish(["Launching"]), 483634, wait=0.01, poll=0)
    timed_out = False
except RuntimeError as exc:
    timed_out = "never became ready" in str(exc)
check("a CLI backend that never comes up raises rather than hanging", timed_out)

print("\n== a detached job's liveness cannot be read from pgrep ==")
ssh_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fidelity", "sshbase.py"), encoding="utf-8").read()
# "pgrep appears nowhere" is the wrong assertion: the file DOCUMENTS at length
# why pgrep cannot work here, and that prose is the point. What must not exist
# is pgrep inside a command this code actually sends to a machine.
sent = [ln for ln in ssh_src.splitlines()
        if "pgrep" in ln and ("exec" in ln or "echo RUNNING" in ln
                              or "-f %s" in ln or "-f {" in ln)]
check("no command sent to a machine shells out to pgrep", not sent)
check("the wrapper records its OWN pid ($$), not the launcher's $!",
      "echo $$ > {d}/pid" in ssh_src and "echo $! > {d}/pid" not in ssh_src)
check("liveness is kill -0 on that pid",
      "kill -0 $(cat {d}/pid)" in ssh_src)
# Why not pgrep, stated so it is not "simplified" back: the probe names the run
# DIRECTORY, which contains the plain run id, so the id is in the probe's own
# cmdline whatever the pattern -- confirmed on procps-ng 4.0.4, where a dead
# target answers RUNNING for both the plain and the bracketed form.
check("...and the file explains why the bracket trick cannot work here",
      "procps-ng" in ssh_src and "bracket" in ssh_src)

print("\n== a benchmark that measured nothing must not write a receipt ==")
from fidelity.bench import _measure                          # noqa: E402

# A Lambda gpu_1x_h100_sxm5 answered `{"error": "no cuda"}` 238 s after the API
# called it active, and the run wrote a receipt of zeros, exit 0. That receipt
# tabulates as a very slow machine and nothing in it says the card was absent.


class Payload:
    """Replays a scripted sequence of payload stdouts, one per exec."""

    def __init__(self, outs):
        self.outs = list(outs)
        self.calls = 0

    def exec_stdout(self, mid, cmd, timeout=0):
        self.calls += 1
        return self.outs[min(self.calls - 1, len(self.outs) - 1)]


GOOD = '{"gpu": "X", "stream_matrix_ms": 0.9}'
NOCUDA = '{"error": "no cuda"}'

p_ok = Payload([GOOD])
check("a real result is returned on the first try",
      _measure(p_ok, "m", lambda *_: None)["stream_matrix_ms"] == 0.9
      and p_ok.calls == 1)

p_race = Payload([NOCUDA, NOCUDA, GOOD])
check("'no cuda' is RETRIED -- the API calls a box ready before the driver is",
      _measure(p_race, "m", lambda *_: None, settle=0)["stream_matrix_ms"] == 0.9
      and p_race.calls == 3)

for outs, why in ((["{\"error\": \"no cuda\"}"], "no cuda forever"),
                  (['{"gpu": "X"}'], "no stream_matrix_ms")):
    try:
        _measure(Payload(outs), "m", lambda *_: None, attempts=2, settle=0)
        raised = False
    except RuntimeError as exc:
        raised = "did not measure anything" in str(exc)
    check("a receipt of zeros is REFUSED (%s)" % why, raised)

print("\n== every launch is a retry loop, with backoff that is polite ==")
import time as _time

_sleeps = []
_real_sleep = mc.time.sleep
mc.time.sleep = lambda t: _sleeps.append(t)


class Flaky:
    """Capacity-fails n times, then succeeds."""
    def __init__(self, n, msg="SUPPLY_CONSTRAINT: no longer any instances available"):
        self.n, self.msg, self.calls = n, msg, 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.n:
            raise RuntimeError(self.msg)
        return {"machine_id": "ok-after-%d" % self.calls}


try:
    _sleeps.clear()
    got = mc._create_with_retry(Flaky(3), Con(), gpu_type="x")
    check("capacity failures are retried to success", got["machine_id"] == "ok-after-4")
    # THIS RUNG USED TO BE FLAKY, AND FLAKY IS WORSE THAN ABSENT. It asserted
    # 1.5 < s[i+1]/s[i] < 2.7 on two JITTERED samples. The implementation is
    # wait = base * 2**(n-1) * uniform(0.8, 1.2), so a ratio of two draws
    # spans 2*0.8/1.2 = 1.33 to 2*1.2/0.8 = 3.0 -- outside that band on
    # 13.7% of runs (200k trials). A battery that goes red one time in seven
    # trains people to re-run it until it is green, which costs more than the
    # rung was ever worth. Assert the CONTRACT rather than a sampled ratio:
    # every wait lies inside its OWN analytic band. Deterministic, and it
    # still fails if the exponent or the jitter is dropped.
    _bands = [(30.0 * (2 ** i) * 0.8, 30.0 * (2 ** i) * 1.2) for i in range(3)]
    check("...with EXPONENTIAL backoff (each wait inside its own band)",
          len(_sleeps) == 3
          and all(lo <= s <= hi for s, (lo, hi) in zip(_sleeps, _bands)))
    check("...and jitter, so fleets do not synchronise",
          all(s % 30 != 0 for s in _sleeps))

    _sleeps.clear()
    try:
        mc._create_with_retry(Flaky(99), Con(), attempts=3, gpu_type="x")
        check("exhausted retries become a Refusal", False)
    except mc.Refusal as r:
        check("exhausted retries become a Refusal", "backoff" in str(r))
        check("...that says $0.00 was spent",
              any("$0.00" in a for a in (r.advice or [])))

    class Broken:
        def create(self, **kw):
            raise RuntimeError("401 unauthorized: bad api key")
    _sleeps.clear()
    try:
        mc._create_with_retry(Broken(), Con(), gpu_type="x")
        check("a NON-capacity error is not retried", False)
    except RuntimeError:
        check("a NON-capacity error is not retried", _sleeps == [])
finally:
    mc.time.sleep = _real_sleep

# A box with no working CUDA is a refusal at preflight, not an advisory: 7 of 8
# H100-SXM5 launches in the survey billed with torch.cuda.is_available()==False.
src_mc = open(os.path.join(here, "measure_cloud.py"), encoding="utf-8").read()
check("preflight REFUSES a no-cuda machine",
      '"no cuda"' in src_mc and "no working CUDA device" in src_mc)

print()
if FAILED:
    print("selftest_provider_portability: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_provider_portability: all passed")
