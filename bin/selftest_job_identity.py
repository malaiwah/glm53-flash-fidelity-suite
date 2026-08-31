#!/usr/bin/env python3
"""Job identity is resolved-first and wide; liveness is tri-state.

WHY THIS EXISTS
---------------
P1-12: the job id was sha1(requested args)[:8] -- it hashed `--revision main`
BEFORE resolution, ignored the suite's own code state, and 32 bits of it
named the instance.  A rerun after upstream `main` (or this repo) moved would
adopt the old machine and relabel outputs older bytes produced.
P1-14: a liveness probe that could not run answered False -- the same value
as CONFIRMED DEAD -- and the controller then launched a second writer into a
live capture.

  J1  identity is derived from the RESOLVED revision: two resolutions of the
      same command are two identities; the full id is 256-bit hex and the
      display id is its 8-char prefix.
  J2  the suite HEAD is part of the identity.
  J3  the panel's resolved revision is part of the identity.
  J4  adoption: an instance wearing this job's display prefix whose lease
      does NOT carry this job's full identity is REFUSED, and nothing is
      created; a matching lease adopts.
  J5  _stage_liveness answers alive / dead / unknown, distinctly.
  J6  _stage_is_alive: only a CONFIRMED dead authorizes a launch; unknown
      retries and then refuses the launch.
  J7  sshbase.run_status: a probe that cannot run reports state=unknown,
      never "failed".

Stub provider, no network, $0.00.
"""
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity import sshbase  # noqa: E402
from fidelity.jlapi import JLError  # noqa: E402

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


MC = load_mc()


def mk_args(**kw):
    base = dict(model="org/model", revision="main", panel="org/panel",
                lane="streaming", spot=True, cold_runs=2,
                provider="jarvislabs", gpu="H200", role="quant",
                out=None, keep_fs=False)
    base.update(kw)
    return argparse.Namespace(**base)


class Inst:
    def __init__(self, machine_id, name, status="running"):
        self.machine_id, self.name, self.status = machine_id, name, status
        self.fs_id = 42


class StubJL:
    dry = False

    def __init__(self, instances=()):
        self.instances = list(instances)
        self.created = []

    def list_instances(self):
        return list(self.instances)

    def fs_create(self, **kw):
        self.created.append(("fs", kw))
        return {"fs_id": 1}

    def create(self, **kw):
        self.created.append(("inst", kw))
        return {"machine_id": 999}

    def exec(self, *a, **kw):
        return {"exit_code": 0, "stdout": "", "stderr": ""}


class Con:
    def __getattr__(self, name):
        return lambda *a, **kw: None


def main():
    rev_a, rev_b = "a" * 40, "b" * 40

    # J1
    i1 = MC.job_identity(mk_args(), resolved_revision=rev_a)
    i2 = MC.job_identity(mk_args(), resolved_revision=rev_b)
    i1_again = MC.job_identity(mk_args(), resolved_revision=rev_a)
    check("J1 two resolved revisions are two identities; same inputs are one",
          i1["job_id_full"] != i2["job_id_full"]
          and i1 == i1_again, (i1, i2))
    check("J1b full id is 256-bit hex; display is its 8-char prefix",
          len(i1["job_id_full"]) == 64
          and all(c in "0123456789abcdef" for c in i1["job_id_full"])
          and i1["job_id"] == i1["job_id_full"][:8], i1)

    # J2: the suite head moves -> the identity moves.
    orig_head = MC._suite_head
    try:
        MC._suite_head = lambda: "c" * 40
        moved = MC.job_identity(mk_args(), resolved_revision=rev_a)
    finally:
        MC._suite_head = orig_head
    check("J2 the suite HEAD is part of the identity",
          moved["job_id_full"] != i1["job_id_full"])

    # J3
    p1 = MC.job_identity(mk_args(), resolved_revision=rev_a,
                         panel_revision=rev_a)
    p2 = MC.job_identity(mk_args(), resolved_revision=rev_a,
                         panel_revision=rev_b)
    check("J3 the panel's resolved revision is part of the identity",
          p1["job_id_full"] != p2["job_id_full"])

    # J4: the adoption gate.
    with tempfile.TemporaryDirectory() as td:
        MC.LEASE_DIR = Path(td)
        args = mk_args(out=str(Path(td) / "out"))
        ident = MC.job_identity(args, resolved_revision=rev_a)
        plan = {"job_id": ident["job_id"], "job_id_full": ident["job_id_full"],
                "instance_name": "fidcloud-%s-x0" % ident["job_id"],
                "deadline_epoch": 4102444800.0,
                "chosen": {"gpu_type": "H200", "gpus": 1, "region": "us"},
                "storage_gb": 100, "requirement": {"ep_size": 1}}
        running = Inst("m-1", "fidcloud-%s-xzz" % ident["job_id"])

        # A lease from a DIFFERENT identity (the resume-relabel case).
        MC.write_lease(ident["job_id"], name=plan["instance_name"],
                       deadline=4102444800.0, machine_id="m-1", fs_id=None,
                       job_id_full="f" * 64)
        jl = StubJL([running])
        td_obj = MC.Teardown(jl, Con(), Path(td) / "out")
        try:
            MC.execute(args, Con(), jl, dict(plan), td_obj)
            check("J4 a prefix-matching instance with a foreign lease refuses",
                  False, "no Refusal")
        except MC.Refusal as exc:
            check("J4 a prefix-matching instance with a foreign lease refuses",
                  "full identity" in str(exc) and not jl.created,
                  "%s created=%s" % (exc, jl.created))

        # A lease carrying THIS identity: adoption proceeds (we stop the run
        # right after by making the bootstrap raise a sentinel).
        MC.write_lease(ident["job_id"], name=plan["instance_name"],
                       deadline=4102444800.0, machine_id="m-1", fs_id=None,
                       job_id_full=ident["job_id_full"])

        class Sentinel(RuntimeError):
            pass

        orig_boot = MC._bootstrap_and_run
        orig_hb = MC._start_heartbeat
        MC._bootstrap_and_run = lambda *a, **kw: (_ for _ in ()).throw(Sentinel())
        MC._start_heartbeat = lambda *a, **kw: None
        try:
            jl = StubJL([running])
            td_obj = MC.Teardown(jl, Con(), Path(td) / "out")
            try:
                MC.execute(args, Con(), jl, dict(plan), td_obj)
                check("J4b a matching lease adopts", False, "no Sentinel")
            except Sentinel:
                check("J4b a matching lease adopts (nothing new created)",
                      td_obj.machine_id == "m-1" and not jl.created,
                      (td_obj.machine_id, jl.created))
        finally:
            MC._bootstrap_and_run = orig_boot
            MC._start_heartbeat = orig_hb

    # J5 / J6: tri-state liveness.
    class LiveJL:
        dry = False

        def __init__(self, answer):
            self.answer = answer
            self.calls = 0

        def exec_stdout(self, mid, cmd, **kw):
            self.calls += 1
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    class TD:
        machine_id = 7

    check("J5 alive / dead / unknown are distinct verdicts",
          MC._stage_liveness(LiveJL("alive\n"), TD(), "measure") == "alive"
          and MC._stage_liveness(LiveJL("gone\n"), TD(), "measure") == "dead"
          and MC._stage_liveness(LiveJL(JLError("boom")), TD(), "measure")
          == "unknown"
          and MC._stage_liveness(LiveJL("garbled"), TD(), "measure")
          == "unknown")

    jl = LiveJL(JLError("api down"))
    verdict = MC._stage_is_alive(jl, TD(), "measure", retries=3,
                                 sleep=lambda *_: None)
    check("J6 unknown-after-retries REFUSES the launch (returns True) and "
          "actually retried",
          verdict is True and jl.calls == 4, jl.calls)
    check("J6b only a confirmed dead authorizes a launch",
          MC._stage_is_alive(LiveJL("gone\n"), TD(), "measure",
                             sleep=lambda *_: None) is False
          and MC._stage_is_alive(LiveJL("alive\n"), TD(), "measure",
                                 sleep=lambda *_: None) is True)

    # J7: sshbase.run_status probe failure is unknown, not failed.
    class T(sshbase.SSHTransport):
        def _endpoint(self, machine_id, *, wait=900):
            return ("h", 22)

        def exec_stdout(self, *a, **kw):
            raise JLError("ssh flaked")

    st = T().run_status("r_1", machine_id=5)
    check("J7 a failed probe reports state=unknown, never failed",
          st.get("state") == "unknown", st)

    print()
    if FAILED:
        print("selftest_job_identity: %d FAILED" % len(FAILED))
        return 1
    print("selftest_job_identity: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
