#!/usr/bin/env python3
"""Offline regression tests for measure_cloud.Teardown.  No rental, no network.

There was no test for `Teardown` at all, which is how CLI-01 and CLI-02(b)
survived: the class that guarantees a rented GPU is destroyed was the one class
nobody exercised.

  CLI-01a  a total API outage must NOT be read as "destroyed"
  CLI-01b  a healthy destroy IS confirmed, from `jl list`, and drops the lease
  CLI-01c  an outage on the CONFIRM only (destroy succeeded) is still unconfirmed
  CLI-01d  an exception escaping a destroy step leaves leaked=True and KEEPS the lease
  CLI-02b  a raise before the steps must not mark the teardown done
  CLI-02c  a second run() after a failed one RETRIES the destroy
  CLI-11   a receipts archive cannot write outside outdir (absolute path,
           `..`, and symlink-plus-write-through)
  CC-07    the packed_root trap is not disarmed by a shard receipt or by a
           file merely named "payload..."
  L52      a resume ATTACHES to a live stage instead of launching a second copy,
           and the liveness probe cannot be answered by its own command text

Stock python3, stdlib only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import measure_cloud as mc            # noqa: E402
from fidelity.jlapi import JLError, Instance   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name,
                          ("  -- " + detail) if (detail and not cond) else ""))


class FakeJL:
    """A `jl` that can be healthy, or down, or down only on the read side."""

    def __init__(self, *, destroy_ok=True, list_ok=True, alive=(487502,),
                 destroy_raises=None):
        self.dry = False
        self.destroy_ok = destroy_ok
        self.list_ok = list_ok
        self.alive = set(alive)
        self.destroy_raises = destroy_raises
        self.destroy_calls = 0
        self.fs_deleted = []

    def destroy(self, mid):
        self.destroy_calls += 1
        if self.destroy_raises is not None:
            raise self.destroy_raises
        if not self.destroy_ok:
            raise JLError("jl destroy %s exited 1: Max retries exceeded" % mid)
        self.alive.discard(mid)
        return {}

    def list_instances(self):
        if not self.list_ok:
            raise JLError("jl list exited 1: Max retries exceeded")
        return [Instance(machine_id=m, status="Running", gpu_type="H200", num_gpus=1,
                         region="IN2", is_spot=True, cost=0.0, runtime=None,
                         fs_id=None, storage_gb=None, name=None, raw={})
                for m in sorted(self.alive)]

    def get(self, mid):
        # deliberately the BROKEN oracle: swallows everything, returns None
        return None

    def exec(self, *a, **k):
        return None

    def exec_stdout(self, *a, **k):
        return ""

    def download(self, *a, **k):
        return None

    def fs_delete(self, fs_id):
        if not self.destroy_ok:
            raise JLError("jl filesystem remove %s exited 1" % fs_id)
        self.fs_deleted.append(fs_id)
        return {}


class QuietConsole:
    def __init__(self, raise_on_say=None):
        self.raise_on_say = raise_on_say
        self.lines = []

    def say(self, text=""):
        if self.raise_on_say is not None:
            raise self.raise_on_say
        self.lines.append(text)

    def step(self, text=""):
        self.lines.append(text)

    def ok(self, text="", detail=""):
        self.lines.append("OK " + text)

    def warn(self, text=""):
        self.lines.append("WARN " + text)

    def err(self, text=""):
        self.lines.append("ERR " + text)


def make_teardown(tmp, jl, con=None, *, fs_id=None, machine_id=487502):
    td = mc.Teardown.__new__(mc.Teardown)
    import threading
    td.jl = jl
    td.con = con or QuietConsole()
    td.outdir = Path(tmp)
    td.fs_root = "/home/jl_fs/fidelity"
    td.machine_id = machine_id
    td.fs_id = fs_id
    td.keep_fs = False
    td.lease_path = Path(tmp) / "lease.json"
    td.lease_path.write_text(json.dumps({"job_id": "test", "machine_id": machine_id}))
    td.done = False
    td._running = False
    td.leaked = False
    td.held = False
    td.hold_on_failure = False
    td.completed = True
    td.pull_timeout = 1
    td._lock = threading.Lock()
    return td


def main():
    tmp = tempfile.mkdtemp(prefix="teardown-selftest-")

    # CLI-01a: total outage. destroy raises AND list raises.
    jl = FakeJL(destroy_ok=False, list_ok=False)
    td = make_teardown(tmp, jl)
    td._destroy_instance()
    check("CLI-01a  a total API outage is NOT read as 'destroyed'",
          td.leaked is True and td.machine_id == 487502,
          "leaked=%r machine_id=%r" % (td.leaked, td.machine_id))
    check("CLI-01a  and all five attempts were made",
          jl.destroy_calls == 5, str(jl.destroy_calls))

    # CLI-01b: healthy destroy.
    jl = FakeJL()
    td = make_teardown(tmp, jl)
    td._destroy_instance()
    check("CLI-01b  a healthy destroy IS confirmed from `jl list`",
          td.leaked is False and td.machine_id is None and jl.destroy_calls == 1,
          "leaked=%r mid=%r calls=%d" % (td.leaked, td.machine_id, jl.destroy_calls))

    # CLI-01c: destroy SUCCEEDS but the account cannot be read back.
    jl = FakeJL(destroy_ok=True, list_ok=False)
    td = make_teardown(tmp, jl)
    td._destroy_instance()
    check("CLI-01c  an unreadable account leaves destruction UNCONFIRMED",
          td.leaked is True and td.machine_id == 487502,
          "leaked=%r mid=%r" % (td.leaked, td.machine_id))

    # the broken oracle, stated as a fact so the test documents WHY:
    check("CLI-01   `jl get` returns None during an outage AND after a destroy",
          FakeJL(list_ok=False).get(487502) is None and FakeJL().get(1) is None)

    # CLI-01d: an unexpected exception in a destroy step keeps the lease.
    jl = FakeJL(destroy_raises=RuntimeError("socket exploded"))
    td = make_teardown(tmp, jl, fs_id=None)
    td.run("test")
    check("CLI-01d  a raising destroy leaves leaked=True",
          td.leaked is True, "leaked=%r" % td.leaked)
    check("CLI-01d  ...and the lease is KEPT for the reaper",
          td.lease_path.is_file())

    # CLI-02b: a console write that raises before the steps.
    jl = FakeJL()
    con = QuietConsole(raise_on_say=OSError(5, "Input/output error"))
    td = make_teardown(tmp, jl, con)
    raised = None
    try:
        td.run("test")
    except Exception as exc:                            # noqa: BLE001
        raised = exc
    check("CLI-02b  a raise before the steps does NOT mark teardown done",
          td.done is False and td._running is False,
          "done=%r running=%r raised=%r" % (td.done, td._running, raised))

    # CLI-02c: the retry is real -- a second run() destroys.
    td.con = QuietConsole()
    td.run("retry")
    check("CLI-02c  a second run() RETRIES and destroys",
          td.machine_id is None and jl.destroy_calls >= 1,
          "mid=%r calls=%d" % (td.machine_id, jl.destroy_calls))

    # ---- CLI-11: an archive built on a rented box cannot escape outdir -----
    import tarfile
    ex = Path(tempfile.mkdtemp(prefix="teardown-tar-"))
    outdir = ex / "out"; outdir.mkdir()
    victim = ex / "VICTIM.txt"
    victim.write_text("original\n")
    payload = ex / "payload.txt"
    payload.write_text("evil\n")
    arch = ex / "receipts.tar.gz"
    with tarfile.open(arch, "w:gz") as tf:
        tf.add(payload, arcname="receipts/ok.txt")
        tf.add(payload, arcname="/" + str(victim).lstrip("/"))       # absolute
        tf.add(payload, arcname="receipts/../../VICTIM.txt")         # traversal
        link = tarfile.TarInfo("receipts/link")
        link.type = tarfile.SYMTYPE
        link.linkname = str(ex)
        tf.addfile(link)
        tf.add(payload, arcname="receipts/link/VICTIM.txt")          # write-through

    class ArchiveJL(FakeJL):
        def __init__(self, arch):
            super().__init__()
            self.arch = arch

        def download(self, mid, remote, local, **k):
            import shutil as _sh
            _sh.copy(self.arch, local)

    td = make_teardown(str(outdir), ArchiveJL(arch))
    td.outdir = outdir
    td._pull_receipts()
    check("CLI-11  the victim outside outdir is UNTOUCHED",
          victim.read_text() == "original\n", repr(victim.read_text()))
    check("CLI-11  the legitimate member IS extracted",
          (outdir / "receipts" / "ok.txt").is_file())
    check("CLI-11  no symlink was created under outdir",
          not (outdir / "receipts" / "link").is_symlink())

    # ---- CC-07: what actually counts as a published payload store ----------
    from fidelity import hfmeta as HM
    def trap_fires(names):
        meta = HM.RepoMeta(repo_id="x/y", repo_type="model", revision="0" * 40,
                           requested_revision="0" * 40, last_modified=None,
                           files=[(n, 1) for n in names])
        real = HM.fetch_json
        HM.fetch_json = lambda *a, **k: {"packed_root": "/home/producer/store"}
        try:
            info = HM.sniff_surface(meta)
        finally:
            HM.fetch_json = real
        return info.surface, bool(info.problems)

    bare = ["config.json", "materialization-receipt.json", "model-00001.safetensors"]
    shards = bare + [".materialization/shards/%03d.json" % i for i in range(3)]
    noise = bare + ["payload_notes.txt"]
    half = bare + ["payload-store/objects/aa", "contract.json", "inventory.json",
                   "mtp-adapter-receipt.json"]
    full = half + ["payload-store/choices/bb"]
    for label, names, want in (("a bare packed repo", bare, True),
                               ("+ .materialization shard receipts", shards, True),
                               ("+ a file named payload_notes.txt", noise, True),
                               ("+ half a store (objects, no choices)", half, True),
                               ("+ a COMPLETE payload store", full, False)):
        surface, fired = trap_fires(names)
        check("CC-07  %-38s trap fires=%s" % (label, want),
              surface == "packed" and fired == want,
              "surface=%s fired=%s" % (surface, fired))

    # ---- LESSON 52: attach before launch ----------------------------------
    class ProbeJL(FakeJL):
        """Records what the controller asked, and what it launched."""

        def __init__(self, alive_stages=()):
            super().__init__()
            self.alive_stages = set(alive_stages)
            self.launched = []
            self.probes = []

        def exec_stdout(self, mid, cmd, **k):
            self.probes.append(cmd)
            # the REAL semantics of `pgrep -f '[s]tage_measure.sh <stage>'`:
            # it matches the running stage, and it must NOT match this probe's
            # own command line, which is what the bracket class buys.
            for st in self.alive_stages:
                if ("stage_measure.sh " + st) in cmd.replace("[s]tage", "stage"):
                    if cmd.count("[s]tage_measure.sh") == 1:
                        return "alive\n"
            return "gone\n"

        def run_job(self, mid, cmd):
            self.launched.append(cmd)
            return {"run_id": "r_fake"}

    jl = ProbeJL(alive_stages={"measure"})
    td = make_teardown(tmp, jl)
    probe = getattr(mc, "_stage_is_alive", None)
    if probe is None:
        for name in ("L52  a live stage is detected",
                     "L52  a stage that was never started is NOT reported alive",
                     "L52  the probe uses the bracket class, so it cannot match itself",
                     "L52  the controller ATTACHES rather than launching a duplicate"):
            check(name, False, "measure_cloud has no _stage_is_alive: the resume "
                               "launches unconditionally")
    else:
        check("L52  a live stage is detected", probe(jl, td, "measure"))
        check("L52  a stage that was never started is NOT reported alive",
              not probe(jl, td, "nosuchstage"),
              "probe: %s" % (jl.probes[-1] if jl.probes else "none"))
        check("L52  the probe uses the bracket class, so it cannot match itself",
              all("[s]tage_measure.sh" in c for c in jl.probes), str(jl.probes[:1]))
        # and the behaviour that matters: _run_stage must not launch into a
        # stage that is already running.
        jl2 = ProbeJL(alive_stages={"measure"})
        td2 = make_teardown(tmp, jl2)
        # already past the deadline, so _await_stage returns on its FIRST
        # check and the suite does not sit through a 120s poll.
        plan_data = {"deadline_epoch": time.time() - 1}
        try:
            mc._run_stage(None, QuietConsole(), jl2, td2, plan_data, "measure")
        except Exception:                               # noqa: BLE001
            pass                                        # deadline/abort is fine
        check("L52  the controller ATTACHES rather than launching a duplicate",
              not jl2.launched, "launched: %s" % jl2.launched[:1])

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
