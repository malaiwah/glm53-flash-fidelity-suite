#!/usr/bin/env python3
"""The reaper: leases authorize, names only discover, destroys are confirmed.

WHY THIS EXISTS
---------------
P1-03 (peer review 2026-08-31): the sweep destroyed on a NAME-parsed deadline
alone (a name is guessable and its base36 suffix parses to nonsense), swallowed
destroy errors and still exited 0, never confirmed the instance actually
reached a terminal state, and its --dry-run omitted the phantom-lease
retirements the real run performs.  The two failure modes are opposite and
both severe: billing that continues behind a false-success cleanup, and
destroying a machine this tool did not create.

Driven entirely against a mocked provider -- no network, no account, $0.00.

  P1  stale lease: destroyed, CONFIRMED against provider state, lease
      retired, exit 0.
  P2  destroy raises: lease kept, exit EXIT_LEAK (90), not 0.
  P3  destroy "succeeds" but the instance stays listed as running:
      unconfirmed -> lease kept, exit EXIT_LEAK.
  P4  name-only candidate (expired-looking fidcloud-* name, no lease):
      NEVER destroyed; reported for the operator instead.
  P5  implausible deadline (name parsing to a nonsense epoch, and a lease
      with deadline 0): neither authorizes anything.
  P6  dry-run enumerates exactly the real run's mutations -- the destroy AND
      the phantom-lease retirement -- and performs none of them.
  P7  destroy confirmation rides out an eventually-consistent listing
      (still listed once, gone on the retry) without failing the sweep.
"""
import importlib.util
import json
import sys
import tempfile
import time
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


MC = load_mc()
from fidelity.jlapi import JLError  # noqa: E402


class Inst:
    def __init__(self, machine_id, name="", status="running"):
        self.machine_id, self.name, self.status = machine_id, name, status
        self.fs_id = None


class FakeJL:
    """A provider that behaves exactly as each case instructs."""

    def __init__(self, instances=(), *, destroy_raises=False,
                 destroy_is_noop=False, lag_listings=0):
        self.instances = list(instances)
        self.destroy_raises = destroy_raises
        self.destroy_is_noop = destroy_is_noop
        self.lag_listings = lag_listings   # listings that still show a
        self.destroyed = []                # destroyed instance (P7)
        self._lagging = {}

    def list_instances(self):
        out = list(self.instances)
        for mid, left in list(self._lagging.items()):
            if left > 0:
                out.append(Inst(mid, status="running"))
                self._lagging[mid] = left - 1
        return out

    def destroy(self, mid):
        if self.destroy_raises:
            raise JLError("api said no")
        self.destroyed.append(mid)
        if self.destroy_is_noop:
            return {}
        self.instances = [i for i in self.instances
                          if str(i.machine_id) != str(mid)]
        if self.lag_listings:
            self._lagging[mid] = self.lag_listings
        return {}


class Con:
    def __init__(self):
        self.lines = []

    def _log(self, kind, *a):
        self.lines.append("%s %s" % (kind, " ".join(str(x) for x in a)))

    def say(self, *a):
        self._log("say", *a)

    def warn(self, *a):
        self._log("warn", *a)

    def err(self, *a):
        self._log("err", *a)

    def ok(self, *a):
        self._log("ok", *a)

    def text(self):
        return "\n".join(self.lines)


def lease(dirpath, job_id, machine_id, deadline):
    path = Path(dirpath) / ("%s.json" % job_id)
    path.write_text(json.dumps({
        "job_id": job_id, "name": "fidcloud-%s-x%s" % (job_id, "0"),
        "machine_id": machine_id, "fs_id": None,
        "deadline_epoch": deadline, "created_at": "t", "pid": 1}))
    return path


def sweep(jl, **kw):
    con = Con()
    rc = MC.reaper_sweep(con, jl=jl, sleep=lambda *_: None,
                         confirm_attempts=3, **kw)
    return rc, con


def main():
    now = time.time()

    with tempfile.TemporaryDirectory() as td:
        MC.LEASE_DIR = Path(td)

        # P1: stale lease -> destroy, confirm, retire, exit 0.
        path = lease(td, "job1", 111, now - 60)
        jl = FakeJL([Inst(111, "fidcloud-job1-x0")])
        rc, con = sweep(jl)
        check("P1 stale lease is destroyed, confirmed, and retired (rc 0)",
              rc == MC.EXIT_OK and jl.destroyed == [111]
              and not path.exists() and "confirmed gone" in con.text(),
              con.text())

        # P2: destroy raises -> lease kept, EXIT_LEAK.
        path = lease(td, "job2", 222, now - 60)
        jl = FakeJL([Inst(222)], destroy_raises=True)
        rc, con = sweep(jl)
        check("P2 a failed destroy keeps the lease and exits EXIT_LEAK",
              rc == MC.EXIT_LEAK and path.exists()
              and "non-zero" in con.text(), "rc=%s\n%s" % (rc, con.text()))
        path.unlink()

        # P3: destroy call succeeds, instance stays listed running.
        path = lease(td, "job3", 333, now - 60)
        jl = FakeJL([Inst(333)], destroy_is_noop=True)
        rc, con = sweep(jl)
        check("P3 an unconfirmed destroy keeps the lease and exits EXIT_LEAK",
              rc == MC.EXIT_LEAK and path.exists()
              and "NOT confirmed" in con.text(),
              "rc=%s\n%s" % (rc, con.text()))
        path.unlink()

        # P4: name-only candidate must never be destroyed.
        expired_name = MC.deadline_name("cafe0123", now - 3600)
        jl = FakeJL([Inst(444, expired_name)])
        rc, con = sweep(jl)
        check("P4 an expired-looking NAME with no lease is reported, "
              "never destroyed (rc 0)",
              rc == MC.EXIT_OK and jl.destroyed == []
              and "no lease of this tool authorizes" in con.text(),
              "rc=%s destroyed=%s\n%s" % (rc, jl.destroyed, con.text()))

        # P5a: implausible name deadline (epoch 1) is ignored outright.
        jl = FakeJL([Inst(555, "fidcloud-beef0000-x1")])
        rc, con = sweep(jl)
        check("P5a an implausible name deadline authorizes nothing",
              rc == MC.EXIT_OK and jl.destroyed == []
              and "implausible" in con.text(),
              "rc=%s\n%s" % (rc, con.text()))

        # P5b: a lease with deadline 0 must not read as "expired forever".
        path = lease(td, "job5", 666, 0)
        jl = FakeJL([Inst(666)])
        rc, con = sweep(jl)
        check("P5b a lease with a nonsense deadline is skipped with a warning",
              jl.destroyed == [] and path.exists()
              and "implausible" in con.text(),
              "destroyed=%s\n%s" % (jl.destroyed, con.text()))
        path.unlink()

        # P6: dry-run enumerates the destroy AND the phantom retirement,
        # mutates nothing.
        stale = lease(td, "job6", 777, now - 60)
        phantom = lease(td, "job7", 888, now + 3600)   # machine not listed
        jl = FakeJL([Inst(777)])
        rc, con = sweep(jl, dry=True)
        check("P6 dry-run lists the destroy and the lease retirement and "
              "touches nothing",
              rc == MC.EXIT_OK and jl.destroyed == []
              and stale.exists() and phantom.exists()
              and "WOULD destroy 777" in con.text()
              and "WOULD retire lease job7.json" in con.text(),
              "rc=%s\n%s" % (rc, con.text()))
        # ... and the real run performs exactly those two mutations.
        rc, con = sweep(jl)
        check("P6b the real run performs exactly what dry-run announced",
              rc == MC.EXIT_OK and jl.destroyed == [777]
              and not stale.exists() and not phantom.exists(),
              "rc=%s destroyed=%s\n%s" % (rc, jl.destroyed, con.text()))

        # P8: a lease from ANOTHER provider is invisible to the jl backend:
        # never destroyed (jl would aim at a same-numbered JarvisLabs box),
        # never retired as a phantom (it is alive on a cloud jl cannot list).
        p8 = Path(td) / "job9.json"
        p8.write_text(json.dumps({
            "job_id": "job9", "name": "fidcloud-job9-x0",
            "provider": "runpod", "machine_id": "k2j9xq1abc", "fs_id": None,
            "deadline_epoch": now - 60, "created_at": "t", "pid": 1}))
        jl = FakeJL([])
        rc, con = sweep(jl)
        check("P8 an expired lease from another provider is left alone "
              "(no destroy, no retirement)",
              rc == MC.EXIT_OK and jl.destroyed == [] and p8.exists()
              and "leaving it alone" in con.text(),
              "rc=%s\n%s" % (rc, con.text()))
        # ... and a legacy provider-less lease with a NON-NUMERIC id (the
        # live RunPod controller writes these) is equally untouchable.
        p8.write_text(json.dumps({
            "job_id": "job9", "name": "fidcloud-job9-x0",
            "machine_id": "k2j9xq1abc", "fs_id": None,
            "deadline_epoch": now - 60, "created_at": "t", "pid": 1}))
        jl = FakeJL([])
        rc, con = sweep(jl)
        check("P8b a provider-less lease with a non-numeric id is left alone",
              rc == MC.EXIT_OK and jl.destroyed == [] and p8.exists(),
              "rc=%s\n%s" % (rc, con.text()))
        p8.unlink()

        # P7: eventually-consistent listing -- still shown once, gone after.
        path = lease(td, "job8", 999, now - 60)
        jl = FakeJL([Inst(999)], lag_listings=1)
        rc, con = sweep(jl)
        check("P7 confirmation rides out one stale listing (rc 0)",
              rc == MC.EXIT_OK and not path.exists()
              and "confirmed gone" in con.text(),
              "rc=%s\n%s" % (rc, con.text()))

    print()
    if FAILED:
        print("selftest_reaper: %d FAILED" % len(FAILED))
        return 1
    print("selftest_reaper: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
