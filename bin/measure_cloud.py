#!/usr/bin/env python3
"""measure-cloud -- rent a GPU, measure a quant's fidelity, seal a receipt, tear down.

    export JL_API_KEY=...
    bin/measure-cloud --model <hf-repo> --panel <hf-dataset> --lane streaming --spot

It spends other people's money, so the safety properties are not optional and
are not configurable away:

  * TEARDOWN IS GUARANTEED on every exit path -- success, failure, exception,
    SIGINT, SIGTERM, SIGHUP -- by a trap that runs before anything else, and by
    three further layers underneath it (see `Teardown` below), because a trap
    does not run when the laptop's battery dies.
  * A COST ESTIMATE is printed and confirmed BEFORE anything is created.
    `--dry-run` stops there and creates nothing at all.
  * A HARD MAX-RUNTIME kill switch is enforced by the controller AND by an
    on-instance watchdog that does not need the controller to be alive.
  * NO TOKEN is ever echoed, put on a command line, or written to a receipt or
    log. Every captured stream passes through a redaction filter first.
  * The runner REFUSES, with advice, when the target obviously will not fit --
    before an instance exists, not after a 200 GB download.

Run `bin/measure-cloud --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The official BF16 release the whole campaign is pinned to.  Every lane binds
# its config/index sha256 into the capture receipt, so this is an identity, not
# a convenience default: resolving `main` instead would let the reference move
# between two measurements of the same artifact.
OFFICIAL_BF16_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"

from fidelity import census as C                       # noqa: E402
from fidelity.common import (                          # noqa: E402
    Console, human_bytes, human_duration, parse_duration, read_json,
    redact, register_secret, sha256_file, utcnow, write_json,
)
from fidelity.dsformat import resolve_inside                 # noqa: E402
from fidelity.engines import EngineUnpinned, build_invocation, load_engines  # noqa: E402
from fidelity.hfmeta import (                          # noqa: E402
    HF_ENDPOINT, HFError, RepoMeta, fetch_file, fetch_json, hf_token,
    load_panel_descriptor, repo_meta, safetensors_header, sniff_surface,
)
from fidelity.jlapi import JL, JLError, JLNotInstalled, select_offer  # noqa: E402
from fidelity.receipt import produced_by_block                      # noqa: E402
from fidelity.stages import stage_sequence                          # noqa: E402

VERSION = "0.1.0"
LEASE_DIR = Path.home() / ".fidelity-cloud" / "leases"
GB = C.GB

EXIT_OK = 0
EXIT_REFUSED = 3
EXIT_INTERRUPTED = 130
EXIT_LEAK = 90          # teardown could not be confirmed -- the loudest failure


# ==========================================================================
# Teardown
# ==========================================================================


class Teardown:
    """Four independent layers, because any one of them can fail.

    L0  controller trap        -- try/finally + SIGINT/SIGTERM/SIGHUP + atexit.
                                  Covers everything except the controller dying
                                  without running code (kill -9, battery, sleep).
    L1  on-instance watchdog   -- absolute deadline plus a controller heartbeat.
                                  Stops the WORK and seals partial receipts. It
                                  deliberately cannot destroy the instance,
                                  because that would require putting a full
                                  account credential on rented hardware (see
                                  --self-destruct-token, off by default).
    L2  laptop lease reaper    -- launchd/cron on the CALLER's machine, reading
                                  ~/.fidelity-cloud/leases/*.json and destroying
                                  anything past its deadline, with the caller's
                                  own credentials. No secret ever leaves.
    L3  name-encoded deadline  -- instances are named fidcloud-<job>-exp<epoch>,
                                  so `measure-cloud reaper --sweep` can clean up
                                  from ANY machine with the account, using only
                                  `jl list`. This is the path a human uses after
                                  a laptop dies, and the only one that covers
                                  the sub-second window between `jl resume`
                                  returning a NEW machine id and the lease file
                                  being rewritten.

    The teeth: the runner refuses to start unless the reaper is installed, or
    --max-runtime <= 2h, or the caller explicitly accepts the leak risk.
    """

    def __init__(self, jl: JL, con: Console, outdir: Path, *,
                 pull_timeout: float = 300.0) -> None:
        self.jl, self.con, self.outdir = jl, con, outdir
        self.pull_timeout = pull_timeout
        self.machine_id: Optional[int] = None
        self.fs_id: Optional[int] = None
        self.keep_fs = False
        # Where the run lives on the instance. JarvisLabs mounts its
        # persistent filesystem at /home/jl_fs; a RunPod volume mounts at
        # /workspace. Both stage scripts already honour FIDELITY_FS_ROOT and
        # FIDELITY_ENGINE_ROOT -- what was missing is that nothing ever SET them,
        # so a non-JarvisLabs box would have written the whole run into the
        # container's ephemeral layer and lost it on the first restart.
        #
        # The BASE is the backend's to declare, because it is a fact about that
        # provider's image and not something this file can infer. Lambda is why:
        # its instances log in as `ubuntu`, not root, and `/home` there is
        # root-owned 0755 -- so `mkdir -p /home/jl_fs/fidelity` is EACCES and
        # the run dies two minutes in, during the bundle upload, on every
        # Lambda rental. Measured on a live gpu_1x_gh200:
        #     drwxr-xr-x 3 root root /home
        #     mkdir: cannot create directory '/home/jl_fs': Permission denied
        # The provider classes already state where they may write (`RUNS`), so
        # `run_base` states it once beside it rather than being re-derived from
        # a provider name here. Unset -> the previous behaviour exactly, which
        # is what keeps jarvislabs, runpod and vast byte-identical.
        base = getattr(jl, "run_base", None) or (
            "/workspace" if getattr(jl, "provider", "jarvislabs") == "runpod"
            else "/home/jl_fs")
        self.fs_root = base + "/fidelity"
        # Named for what it holds -- the venv, the pipeline clone and the
        # patch series -- not for the K6 campaign that first paid for it.
        self.engine_root = base + "/fidelity-engine"
        self.lease_path: Optional[Path] = None
        self.done = False
        # CLI-02(b): re-entrancy is a SEPARATE flag from completion, so a
        # teardown that raises mid-way does not mark itself finished.
        self._running = False
        self.leaked = False
        # --hold-on-failure: on a FAILED exit, pull the receipts and shred the
        # secrets as always, but leave the instance alive so the half-finished
        # work (a 165 GB fetch, a materialized tree, one cold run) can be
        # inspected and resumed instead of re-bought.  L1/L2/L3 still expire
        # it: the lease is deliberately KEPT, and the name still carries the
        # deadline, so a held box is bounded, not leaked.
        self.hold_on_failure = False
        self.held = False
        # Set to True only when the measurement actually completed. The hold
        # used to key off the teardown REASON string ("failed: ..."), which
        # never matched: a stage failure raises a bare RuntimeError, main()
        # catches only (JLError, HFError, Refusal), and the `finally` then
        # tears down with reason "normal exit". The box was destroyed with the
        # fetch on it, which is the exact outcome the flag exists to prevent.
        self.completed = False
        self._lock = threading.Lock()

    def adopt(self, machine_id: Optional[int], fs_id: Optional[int] = None) -> None:
        """Adopt a (possibly renumbered) machine id and persist it immediately.

        `jl resume` can return a NEW machine_id.  Anything that does not adopt
        it unconditionally will destroy the wrong box, or nothing at all.

        The filesystem id goes into the lease too.  A lease naming only the
        instance leaves the reaper able to stop the compute bill and unable to
        remove the 400 GB volume behind it, which keeps billing on its own --
        and that is exactly the case when an EXISTING instance is adopted,
        because the lease is written before the adoption that learns its fs.
        """
        self.machine_id = machine_id
        if fs_id is not None:
            self.fs_id = fs_id
        if self.lease_path and self.lease_path.is_file():
            try:
                lease = read_json(str(self.lease_path))
                lease["machine_id"] = machine_id
                if self.fs_id is not None:
                    lease["fs_id"] = self.fs_id
                lease["updated_at"] = utcnow()
                write_json(str(self.lease_path), lease)
            except OSError:
                pass

    def run(self, reason: str = "") -> None:
        # CLI-02(b).  `done` used to be set HERE, before the announcement and
        # before the steps.  Anything that raised in between -- a console write
        # to a closed pty raises OSError(EIO), not only BrokenPipeError --
        # skipped every destroy with `done` already True, so the atexit hook and
        # the outer `finally` both no-op'd and the instance was never destroyed.
        # The re-entrancy guard is a SEPARATE flag, cleared in a finally, and
        # `done` is set only after the steps loop has been attempted.  A second
        # run() therefore RETRIES rather than no-ops, which is safe:
        # _destroy_instance clears machine_id on confirmed destruction and
        # _destroy_fs early-returns once fs_id is None.
        with self._lock:
            if self.done or self._running:
                return
            self._running = True
        if self.machine_id is None and self.fs_id is None:
            # Nothing to destroy, but a lease may already be on disk: it is
            # written BEFORE `jl create` on purpose. Leaving it behind makes
            # `reaper --list` report a phantom job forever.
            try:
                self._drop_lease()
            finally:
                with self._lock:
                    self._running = False
                    self.done = True
            return
        # Printing "do NOT interrupt" is not a defence.  A second ^C re-enters
        # the signal handler, finds done=True, no-ops, and sys.exit()s straight
        # through the destroy that has not happened yet -- which leaks the
        # instance at the exact moment the user was trying to stop the bill.
        # Take the choice away for the duration instead of asking for it.
        prev = {}
        steps_attempted = False
        # CLI-02(b), second half.  The SIG_IGN restore used to live in the
        # `finally` of the steps try, so anything that raised between installing
        # the handlers and reaching that try left SIGINT/SIGTERM/SIGHUP ignored
        # for the life of the process -- the process that just leaked a GPU,
        # immune to ^C and to `kill`.  The try now opens BEFORE the handlers are
        # installed, so the restore and the flag reset run on every path out.
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                try:
                    prev[sig] = signal.signal(sig, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
            hold = bool(self.hold_on_failure and not self.completed)
            self.con.say("")
            self.con.step("teardown%s (^C is ignored until this finishes)"
                          % ((" -- " + reason) if reason else ""))
            steps = [self._pull_receipts, self._collect_env, self._shred_secrets]
            if hold:
                self.held = True
            else:
                steps += [self._destroy_instance, self._destroy_fs, self._drop_lease]
            steps_attempted = True
            for step in steps:
                try:
                    step()
                except Exception as exc:                # noqa: BLE001
                    # A failure inside teardown must never skip the destroy that
                    # comes after it.  That is the whole reason each step is
                    # individually wrapped instead of the block as a whole.
                    self.con.warn("teardown step %s: %s"
                                  % (step.__name__, redact(str(exc))))
                    # CLI-01, second half: an exception escaping a DESTROY step
                    # used to leave `leaked` False, and _drop_lease then deleted
                    # the lease, so the backstop never looked at the box again.
                    if step in (self._destroy_instance, self._destroy_fs):
                        self.leaked = True
        finally:
            for sig, handler in prev.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
            with self._lock:
                self._running = False
                # `done` marks a teardown that RAN its steps. A teardown that
                # raised before them is not done, and a second run() must retry.
                self.done = steps_attempted
        if self.held:
            self.con.say("")
            self.con.say("*" * 78)
            self.con.say("**  HELD (--hold-on-failure): instance %s is STILL RUNNING and"
                         % self.machine_id)
            self.con.say("**  STILL BILLING, so the finished stages survive for a resume.")
            self.con.say("**    inspect:  jl exec %s 'tail -50 %s/logs/*.log'"
                         % (self.machine_id, self.fs_root))
            # There is no `adopt` subcommand; re-running the SAME command is
            # the resume, because plan() adopts any Running instance whose name
            # starts with this job's prefix and every finished stage is skipped
            # by its done-marker. Printing a command that does not exist sent
            # the reader looking for a feature instead of at the answer.
            self.con.say("**    resume :  re-run the same measure-cloud command -- it ADOPTS this")
            self.con.say("**              instance by job id and skips every finished stage")
            self.con.say("**    DESTROY:  jl destroy %s --yes" % self.machine_id)
            self.con.say("**  Its lease is kept, so the reaper still destroys it at the"
                         " deadline.")
            self.con.say("*" * 78)

    # -- steps -------------------------------------------------------------

    def _pull_receipts(self) -> None:
        """Bring the receipts home as ONE archive, not as a directory walk.

        `jl download -r` moves a tree file by file, and each file is an API
        round trip of ten seconds or so. A 34-file, 21 MB receipts directory
        therefore blew the 300-second timeout and the whole measurement came
        home with nothing -- twice, observed. Tarring on the instance turns it
        into one transfer, and the tar is kept next to the extracted tree as
        the thing whose digest can be quoted.
        """
        if self.machine_id is None or self.jl.dry:
            return
        dest = self.outdir / "receipts"
        dest.mkdir(parents=True, exist_ok=True)
        self.con.step("pulling receipts (timeout %ds)" % int(self.pull_timeout))
        archive = "%s/receipts.tar.gz" % self.fs_root
        try:
            self.jl.exec(self.machine_id,
                         "cd %s && tar czf %s receipts" % (self.fs_root, archive),
                         timeout=self.pull_timeout)
            local = self.outdir / "receipts.tar.gz"
            self.jl.download(self.machine_id, archive, str(local),
                             recursive=False, timeout=self.pull_timeout)
            if local.is_file():
                import tarfile

                # CLI-11 / SEC-08.  This was `tf.extractall(self.outdir)`,
                # annotated "our own archive".  It is not: it is built by a
                # `tar czf` on a RENTED instance and arrives through the vendor
                # control plane.  On python3.9 -- this tree's stated stock
                # target -- extractall applies no filter and warns about
                # nothing, and `outdir` defaults to ./fidelity-runs/<job> under
                # the CWD the README tells you to run from, so two `..` reach
                # the suite's own source.  The explicit pass below is the
                # load-bearing one; `filter="data"` is added only where it
                # exists (PEP 706 landed in 3.9.17, and passing it on an older
                # 3.9 raises TypeError).
                #
                # ORDER MATTERS: links are rejected BEFORE the realpath check.
                # realpath runs before extraction, when the symlink does not
                # exist yet, so both `receipts/link` and `receipts/link/x` test
                # as inside the root -- and then the extraction follows the link
                # and overwrites the victim.
                #
                # Skip-with-a-warning, never raise: a raise lands in the
                # `except` below and falls back to `jl download -r`, which this
                # docstring records as having lost a whole measurement twice.
                _plain = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
                out_root = str(self.outdir.resolve())
                with tarfile.open(local) as tf:
                    safe = []
                    for m in tf.getmembers():
                        if m.issym() or m.islnk() or m.type not in _plain:
                            self.con.warn(
                                "receipts.tar.gz: refusing %s member %s"
                                % ("link" if (m.issym() or m.islnk()) else "special",
                                   redact(m.name)))
                            continue
                        if os.path.isabs(m.name) or ".." in Path(m.name).parts:
                            self.con.warn("receipts.tar.gz: refusing escaping "
                                          "member %s" % redact(m.name))
                            continue
                        try:
                            resolve_inside(out_root, m.name, "receipts.tar.gz")
                        except Exception as exc:        # noqa: BLE001
                            self.con.warn("receipts.tar.gz: refusing member %s (%s)"
                                          % (redact(m.name), redact(str(exc))))
                            continue
                        safe.append(m)
                    kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                    tf.extractall(self.outdir, members=safe, **kw)  # noqa: S202
                n = len(list(dest.rglob("*")))
                self.con.ok("receipts pulled", "%d entries via %s"
                            % (n, local.name))
                return
        except Exception as exc:                        # noqa: BLE001
            self.con.warn("archive pull failed (%s); falling back to a tree walk"
                          % redact(str(exc)))
        self.jl.download(self.machine_id, self.fs_root + "/receipts", str(dest),
                         recursive=True, timeout=self.pull_timeout)
        n = len(list(dest.rglob("*")))
        self.con.ok("receipts pulled", "%d entries" % n)

    def _collect_env(self) -> None:
        if self.machine_id is None or self.jl.dry:
            return
        try:
            out = self.jl.exec_stdout(
                self.machine_id,
                "nvidia-smi || true; df -h || true; "
                "%s/venv/bin/pip freeze 2>/dev/null || true" % self.fs_root,
                timeout=120, check=False)
            (self.outdir / "environment.txt").write_text(redact(out), encoding="utf-8")
        except Exception:                               # noqa: BLE001
            pass

    def _shred_secrets(self) -> None:
        if self.machine_id is None or self.jl.dry:
            return
        self.jl.exec(self.machine_id,
                     "shred -u %s/.secrets/* 2>/dev/null || rm -f %s/.secrets/* 2>/dev/null; true"
                     % (self.fs_root, self.fs_root), timeout=120)
        self.con.ok("secrets shredded")

    def _confirm_gone(self, mid: int) -> Optional[bool]:
        """True gone, False alive, None unknown.

        CLI-01.  `jl get` CANNOT answer this.  On a healthy API a destroyed
        instance is a 404, which `JLApi.get()` turns into None -- and so is
        every transient outage, because `get()` swallows JLError.  Reading
        `None` as "destroyed" declared success on attempt 1 of 5 with zero
        successful API interaction, cleared machine_id, left `leaked` False and
        deleted the lease, so the reaper never looked at the box again.

        `list_instances()` propagates JLError by contract (see its docstring:
        it must not answer "none" when it does not know), which is exactly the
        third state this needs.  Making `get()` strict instead would be worse:
        `get() -> None` IS the normal, load-bearing signal for a successful
        destroy, so a strict variant would fire the leak banner on every run.
        """
        try:
            # str(): ids are ints on one provider and opaque strings on
            # another, and a mixed-type set silently never matches.
            alive = {str(i.machine_id) for i in self.jl.list_instances()}
        except JLError:
            return None
        except Exception:                               # noqa: BLE001
            return None
        # str() on BOTH sides. This is the "is it really gone?" check that
        # decides whether the leak banner fires, so a silent type mismatch
        # here would report a live, billing instance as destroyed -- the worst
        # possible direction for this particular comparison to fail in.
        return str(mid) not in alive

    def _destroy_instance(self) -> None:
        if self.machine_id is None:
            return
        if self.jl.dry:
            self.con.ok("would destroy instance", str(self.machine_id))
            return
        mid = self.machine_id
        for attempt in range(5):
            destroy_raised = None
            try:
                self.jl.destroy(mid)
            except JLError as exc:
                destroy_raised = exc
                self.con.warn("destroy attempt %d: %s" % (attempt + 1, redact(str(exc))))
            except Exception as exc:                    # noqa: BLE001
                # An unexpected exception must fall through to the next attempt,
                # never escape with leaked=False.
                destroy_raised = exc
                self.con.warn("destroy attempt %d raised %s: %s"
                              % (attempt + 1, type(exc).__name__, redact(str(exc))))
            time.sleep(min(2 ** attempt, 20))
            gone = self._confirm_gone(mid)
            if gone is True:
                self.con.ok("instance destroyed", str(mid))
                self.machine_id = None
                return
            if gone is None:
                self.con.warn("destroy attempt %d: could not READ the account, so "
                              "destruction is unconfirmed (not assumed)" % (attempt + 1))
            elif destroy_raised is None:
                self.con.warn("destroy attempt %d: instance %s is still listed"
                              % (attempt + 1, mid))
        self.leaked = True
        self.con.say("")
        self.con.say("!" * 78)
        self.con.say("!!  COULD NOT CONFIRM DESTRUCTION OF INSTANCE %s" % mid)
        self.con.say("!!  IT MAY STILL BE BILLING. Run this now:")
        self.con.say("!!      jl destroy %s --yes" % mid)
        self.con.say("!" * 78)

    def _destroy_fs(self) -> None:
        if self.fs_id is None or self.jl.dry:
            return
        if self.keep_fs:
            # A kept filesystem keeps accruing after the instance is gone. Say
            # so plainly: the caller now owns a standing charge.
            self.con.warn(
                "filesystem %s KEPT (--keep-fs). It continues to accrue storage "
                "charges until you run: jl filesystem delete %s --yes"
                % (self.fs_id, self.fs_id))
            return
        fsid = self.fs_id
        try:
            self.jl.fs_delete(fsid)
        except JLError as exc:
            # A filesystem that outlives its instance keeps billing storage
            # forever, and nothing else in the four layers looks for one. Treat
            # it exactly like an undestroyed instance: set `leaked`, keep the
            # lease so the reaper retries, and exit EXIT_LEAK.
            self.leaked = True
            self.con.say("")
            self.con.say("!" * 78)
            self.con.say("!!  COULD NOT DELETE FILESYSTEM %s  (%s)"
                         % (fsid, redact(str(exc))))
            self.con.say("!!  IT IS STILL BILLING STORAGE. Run this now:")
            self.con.say("!!      jl filesystem remove %s --yes" % fsid)
            self.con.say("!" * 78)
            return
        self.con.ok("filesystem deleted", str(fsid))
        self.fs_id = None

    def _drop_lease(self) -> None:
        if self.lease_path and self.lease_path.is_file() and not self.leaked:
            self.lease_path.unlink()


# ==========================================================================
# Leases and the reaper
# ==========================================================================


def write_lease(job_id: str, *, name: str, deadline: float,
                machine_id: Optional[int], fs_id: Optional[int]) -> Path:
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    path = LEASE_DIR / ("%s.json" % job_id)
    write_json(str(path), {
        "job_id": job_id,
        "name": name,
        "machine_id": machine_id,
        "fs_id": fs_id,
        "deadline_epoch": deadline,
        "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
        "created_at": utcnow(),
        "pid": os.getpid(),
    })
    return path


def reaper_installed() -> bool:
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "LaunchAgents"
                / "com.malaiwah.fidelity-reaper.plist").is_file()
    marker = Path.home() / ".fidelity-cloud" / "reaper-installed"
    return marker.is_file()


def reaper_install(con: Console) -> int:
    self_path = Path(__file__).resolve()
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_dir.mkdir(parents=True, exist_ok=True)
        plist = plist_dir / "com.malaiwah.fidelity-reaper.plist"
        plist.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0"><dict>\n'
            '  <key>Label</key><string>com.malaiwah.fidelity-reaper</string>\n'
            '  <key>ProgramArguments</key><array>\n'
            '    <string>%s</string><string>%s</string>\n'
            '    <string>reaper</string><string>--sweep</string>\n'
            '  </array>\n'
            '  <key>StartInterval</key><integer>300</integer>\n'
            '  <key>RunAtLoad</key><true/>\n'
            '  <key>StandardErrorPath</key><string>%s/reaper.log</string>\n'
            '</dict></plist>\n'
            % (sys.executable, self_path, str(LEASE_DIR.parent)),
            encoding="utf-8")
        con.ok("reaper installed", str(plist))
        con.say("    load it now:  launchctl load -w %s" % plist)
    else:
        marker = Path.home() / ".fidelity-cloud" / "reaper-installed"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(utcnow(), encoding="utf-8")
        con.ok("reaper marker written", str(marker))
        con.say("    add to crontab:")
        con.say("      */5 * * * * %s %s reaper --sweep >> ~/.fidelity-cloud/reaper.log 2>&1"
                % (sys.executable, self_path))
    return EXIT_OK


def deadline_name(job_id: str, deadline: float) -> str:
    """`fidcloud-<job>-x<base36 epoch>` -- 25 chars, and that matters.

    The same string names the instance AND the filesystem, and
    `jl filesystem create --name` rejects anything over 30 characters.  The
    original `fidcloud-<8hex>-exp<10-digit epoch>` is always 31, so every real
    run died on its first mutating call.  Base36 buys four characters of
    headroom without giving up the self-describing deadline that L3 needs.
    """
    return "fidcloud-%s-x%s" % (job_id, _b36(int(deadline)))


def _b36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


def parse_deadline_name(name: str) -> Optional[int]:
    """Deadline out of an instance name, in either encoding.

    `-exp<decimal>` is still accepted so a sweep run from a newer checkout can
    still reap an instance created by an older one.
    """
    for sep, base in (("-x", 36), ("-exp", 10)):
        head, found, tail = name.rpartition(sep)
        if found and tail:
            try:
                return int(tail, base)
            except ValueError:
                continue
    return None


def reaper_sweep(con: Console, *, dry: bool = False) -> int:
    """Destroy anything past its deadline, from leases AND from instance names.

    The name-encoded path matters more than it looks: it needs no local state
    at all, so it still works from a machine that has never seen this job.
    """
    jl = JL(dry=dry)
    try:
        jl.require()
    except (JLNotInstalled, JLError) as exc:
        con.err(str(exc))
        return 1
    now = time.time()
    targets: Dict[int, str] = {}

    for path in sorted(LEASE_DIR.glob("*.json")) if LEASE_DIR.is_dir() else []:
        try:
            lease = read_json(str(path))
        except (OSError, ValueError):
            continue
        if lease.get("machine_id") and float(lease.get("deadline_epoch", 0)) < now:
            targets[int(lease["machine_id"])] = "lease %s expired" % path.name

    try:
        for inst in jl.list_instances():
            name = inst.name or ""
            if not name.startswith("fidcloud-"):
                continue
            deadline = parse_deadline_name(name)
            if deadline is not None and deadline < now:
                targets.setdefault(inst.machine_id,
                                   "name deadline %d passed" % deadline)
    except JLError as exc:
        con.warn("could not list instances: %s" % redact(str(exc)))

    # A lease whose instance no longer exists is a phantom: `reaper --list`
    # keeps reporting a job that is already destroyed, which is exactly the
    # noise that makes an operator stop reading the list. It happens whenever a
    # box is torn down by hand -- the case --hold-on-failure creates on purpose,
    # since holding KEEPS the lease. Retire those here, where we already have
    # the live instance list, and say which.
    if not dry:
        try:
            alive = {str(i.machine_id) for i in jl.list_instances()}
        except JLError:
            alive = None
        if alive is not None:
            for path in sorted(LEASE_DIR.glob("*.json")) if LEASE_DIR.is_dir() else []:
                try:
                    lease = read_json(str(path))
                except (OSError, ValueError):
                    continue
                mid = lease.get("machine_id")
                if mid and str(mid) not in alive and str(mid) not in targets:
                    con.say("reaper: retiring lease %s (machine %s is gone)"
                            % (path.name, mid))
                    path.unlink(missing_ok=True)

    if not targets:
        con.say("reaper: nothing expired")
        return EXIT_OK
    for mid, why in sorted(targets.items()):
        con.say("reaper: destroying %s (%s)" % (mid, why))
        if not dry:
            try:
                jl.destroy(mid)
            except JLError as exc:
                con.err("reaper could not destroy %s: %s" % (mid, redact(str(exc))))
    return EXIT_OK


def reaper_list(con: Console) -> int:
    if not LEASE_DIR.is_dir() or not any(LEASE_DIR.glob("*.json")):
        con.say("no active leases in %s" % LEASE_DIR)
        return EXIT_OK
    now = time.time()
    for path in sorted(LEASE_DIR.glob("*.json")):
        lease = read_json(str(path))
        left = float(lease.get("deadline_epoch", 0)) - now
        con.say("  %-14s machine %-10s fs %-8s %s"
                % (lease.get("job_id"), lease.get("machine_id"), lease.get("fs_id"),
                   ("expires in " + human_duration(left)) if left > 0
                   else "EXPIRED %s ago" % human_duration(-left)))
    return EXIT_OK


# ==========================================================================
# Planning
# ==========================================================================


class Refusal(RuntimeError):
    def __init__(self, reason: str, advice: List[str]) -> None:
        self.reason, self.advice = reason, advice
        super().__init__(reason)


def would_refuse(con, plan: Dict[str, Any], refusal: "Refusal") -> None:
    """Record a dry-run refusal AND print the remedy that goes with it.

    A refusal carries two things: what is wrong, and what to do about it. The
    six dry-run sites here each decided for themselves how much of the advice
    to show -- three showed none of it, two truncated it, one showed all of it
    -- so the site that mattered most in practice (no engine --profile for this
    surface at this bit rate, whose advice names the files to edit) printed
    nothing but the complaint. That is exactly backwards: the docs send you to
    `--dry-run` FIRST, so dry-run is the mode where the remedy matters most,
    and a reader who never triggers a real refusal never sees it.
    `measure_local.plan.problem` already did this correctly; this is the same
    contract on the cloud side, in ONE place so the next site cannot drift.
    """
    con.warn("WOULD REFUSE (real run): %s" % refusal.reason)
    for line in refusal.advice:
        if line and not line.startswith("Nothing was created"):
            con.say("           %s" % line)
    plan.setdefault("would_refuse", []).append(refusal.reason)





# Providers spell the same state differently: JarvisLabs "Running", RunPod
# "RUNNING". A hardcoded exact match read every healthy RunPod poll as
# not-running, and after two of those the controller declared a PREEMPTION and
# tore down a box that was busy building exllamav3. Compared case-folded, in
# one place, so a third provider cannot reintroduce it.
_RUNNING_STATES = frozenset({"running", "run", "active", "ready"})


def _is_running(inst) -> bool:
    return inst is not None and str(getattr(inst, "status", "")).strip().lower() \
        in _RUNNING_STATES


def _stage_env(td: "Teardown") -> str:
    """The two roots every stage script reads, as an inline env prefix.

    Both scripts default to the JarvisLabs paths, which is correct there and
    silently wrong anywhere else. Exporting them explicitly costs nothing on
    JarvisLabs (the values are identical to the defaults) and is the whole
    difference between a working and a lost run on any other provider.
    """
    engine = getattr(td, "engine_root", "/home/jl_fs/fidelity-engine")
    # FIDELITY_K6_ROOT is exported alongside the new name for one release: a
    # container image or an instance script from an older checkout still reads
    # only the old spelling, and a root that resolves to nothing is a run
    # written into the container's ephemeral layer.
    return ("FIDELITY_FS_ROOT=%s FIDELITY_ENGINE_ROOT=%s FIDELITY_K6_ROOT=%s "
            "QP_PIPELINE_ROOT=%s"
            % (shlex.quote(td.fs_root), shlex.quote(engine), shlex.quote(engine),
               shlex.quote("%s/pipeline" % engine)))


def _make_provider(name: str, *, dry: bool = False):
    """The provider is one object with eighteen methods; everything else in
    this file -- fit, cost band, lease, all four teardown layers, every stage
    -- is written against that surface rather than against a vendor."""
    if name == "runpod":
        from fidelity.runpodapi import RunPod
        return RunPod(dry=dry)
    if name == "vast":
        from fidelity.vastapi import Vast
        return Vast(dry=dry)
    if name == "lambda":
        from fidelity.lambdaapi import LambdaCloud
        return LambdaCloud(dry=dry)
    return JL(dry=dry)


def _machine_id_of(created: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Pull a machine id out of whatever shape the provider answered with.

    A vendor that renames `machine_id` to `id` must not turn into a leaked
    instance, so this accepts either and returns None rather than raising.

    It must also accept a NON-NUMERIC id. JarvisLabs machine ids are integers;
    RunPod pod ids are opaque strings like `k2j9xq1abc`. The old
    `str(value).isdigit()` test rejected those, returned None, and the
    controller would have read that as "creation failed" -- while a pod was
    running and billing. That is precisely the leak this function exists to
    prevent, so the id is returned as-is when it is not an integer.
    """
    if not isinstance(created, dict):
        return None
    for key in ("machine_id", "pod_id", "id", "instance_id"):
        value = created.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            return int(value) if value.strip().isdigit() else value.strip()
    return None


def _find_by_name(jl: JL, name: str) -> Optional[int]:
    """Last-resort id recovery: the instance name is unique to this job."""
    try:
        for inst in jl.list_instances():
            if inst.name == name and inst.status.lower() not in (
                    "destroyed", "terminated"):
                return inst.machine_id
    except JLError:
        pass
    return None


def job_id_for(args: argparse.Namespace) -> str:
    """A run's identity, and therefore where its receipts land.

    PROVIDER and GPU are part of it. They were not, and the same measurement
    run on two providers produced the same job id, the same
    `fidelity-runs/<id>/` directory, and the second run silently OVERWROTE the
    first one's sealed receipt. That was found the only way it can be: two
    runs of one artifact on two clouds finished hours apart and the earlier
    result had to be rescued from disk before the later one landed.

    It matters beyond housekeeping. Cross-device reproduction is a RESULT this
    project publishes -- H200 and A100 agree to three significant figures and
    are not bitwise identical -- so the hardware a number came from is part of
    that number's identity, not an incidental detail of where it ran.
    """
    key = json.dumps({
        "model": args.model, "revision": args.revision, "panel": args.panel,
        "lane": args.lane, "spot": args.spot, "cold_runs": args.cold_runs,
        "provider": getattr(args, "provider", "jarvislabs"),
        "gpu": getattr(args, "gpu", None),
        "role": getattr(args, "role", "quant"),
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:8]




def _validate_scope_json(con: Console, path: str) -> None:
    """Schema-check --scope-json NOW, not after the rental.

    The scope file is copied verbatim into the sealed receipt, and the receipt
    is schema-validated at SEAL time -- the last stage of the run. A scope file
    carrying one property the submission schema does not allow therefore costs
    the entire rental before anything says so: the turbo-2.05bpw re-run measured
    for 2 h 06 m, scored, sealed a receipt with the right number in it, and only
    then failed with

        /artifact/scope: additional property 'derivation' is not allowed

    That receipt was recoverable by re-sealing offline from the pulled receipts,
    so no money was lost -- but nothing about that was designed, and a run whose
    receipts had not been pulled would have had to be paid for twice.

    Validated against the REAL submission schema rather than a hand-copied
    subset: the scope is embedded in a skeleton document and only the errors
    under /artifact/scope are reported, so this cannot drift from what the
    validator will actually enforce.
    """
    tools = SUITE_ROOT / "registry" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        import _minischema                                # noqa: WPS433
        reg = _minischema.Registry(str(SUITE_ROOT / "registry" / "schema"))
    except Exception as exc:                              # noqa: BLE001
        con.warn("scope schema check skipped: %s" % redact(str(exc)))
        return
    try:
        doc = read_json(path)
    except Exception as exc:                              # noqa: BLE001
        raise Refusal("--scope-json %s is not readable JSON (%s)"
                      % (path, redact(str(exc))),
                      ["Nothing was created. $0.00 spent."])
    try:
        errs = reg.validate({"artifact": {"scope": doc}},
                            "submission.schema.json")
    except Exception as exc:                              # noqa: BLE001
        con.warn("scope schema check skipped: %s" % redact(str(exc)))
        return
    scoped = [e for e in errs
              if str(getattr(e, "path", "")).startswith("/artifact/scope")]
    if scoped:
        raise Refusal(
            "--scope-json %s does not satisfy the submission schema" % path,
            ["%s: %s" % (getattr(e, "path", "?"), getattr(e, "message", e))
             for e in scoped[:6]]
            + ["",
               "This file is copied VERBATIM into the sealed receipt, and the "
               "receipt is schema-checked at seal time -- the last stage of the "
               "run. Caught here it costs nothing; caught there it costs the "
               "whole rental.",
               "Nothing was created. $0.00 spent."])
    con.ok("scope file schema", "valid against submission.schema.json")


def _refuse_quantized_root(con: Console, target, surface, plan: Dict[str, Any]) -> None:
    """A reference must be the unquantized thing, or it is not a reference.

    Every measurement in this registry is a distance FROM a root, so a root
    that is itself quantized silently redefines what every downstream number
    means: rows would report divergence from somebody's quantization rather
    than from the model, while still being labelled a floor.

    This is not a theoretical risk. `layer_outer.py` builds no `HfQuantizer`,
    so for a plain FP8 weight the shape MATCHES the bf16 parameter, the payload
    is read as bf16 and the block scale is never applied -- a wrong capture
    that raises nothing (the M1 Qwen3.8-27B-FP8 defect). And it is not rare:
    every `deepseek_v4` repo on the Hub ships a `quantization_config`, which is
    exactly why that family has no root.

    Decided from the release's own config, before anything is rented.
    """
    # Decide from the checkpoint's OWN config, not from surface classification.
    # `sniff_surface` returns "unknown" for plenty of perfectly unquantized
    # roots -- zai-org/GLM-5.3-BF16 and zai-org/GLM-5.2 both do -- and refusing
    # on "unknown" would block the exact captures this mode exists for. The
    # authoritative, cheap and unambiguous evidence is whether the release
    # publishes a `quantization_config` at all.
    try:
        cfg = fetch_json(target.repo_id, "config.json", revision=target.revision)
    except HFError as exc:
        raise Refusal(
            "--role root, but this checkpoint's config.json could not be read "
            "(%s)" % redact(str(exc)),
            ["A root must be shown to be unquantized before it is captured, and "
             "that is decided from config.json:quantization_config.",
             "Nothing was created. $0.00 spent."])
    qc = cfg.get("quantization_config") or (
        cfg.get("text_config") or {}).get("quantization_config")
    if not qc:
        con.ok("root is unquantized",
               "surface %s: config.json declares no quantization_config"
               % surface.surface)
        plan.setdefault("target", {})["root_unquantized"] = True
        return
    method = qc.get("quant_method") or qc.get("fmt") or "declared"
    raise Refusal(
        "--role root, but this checkpoint publishes a quantization_config "
        "(quant_method %s)" % method,
        ["A root is the thing every later measurement is a distance FROM. "
         "Capturing a quantized checkpoint as a root would publish "
         "divergence-from-somebody's-quantization under the name of a floor.",
         "Worse, it would not fail loudly: the layer-outer schedule builds no "
         "HfQuantizer, so an FP8 weight has the same SHAPE as the bf16 "
         "parameter -- the payload is read as bf16 and the block scale is "
         "never applied.",
         "Point --model at the unquantized release. If the family publishes "
         "none -- as no deepseek_v4 repo on the Hub does -- then it has no "
         "root and no floor can be measured for it.",
         "Nothing was created. $0.00 spent."])


def _refuse_incomplete_exl3hf(con: Console, repo_id: str, revision: str,
                              plan: Dict[str, Any]) -> None:
    """Does this release actually contain the whole model?

    A stock-exllamav3 conversion is only measurable if its non-routed tensors
    cover the official non-routed set: the streaming lane loads them as the
    model. This is decidable from two index files -- the artifact's and the
    official release's -- plus the MTP sidecar's safetensors header, so it
    costs metadata, not a rental.

    It is not hypothetical. turboderp's 3.05bpw branch is missing 22 tensors
    the 4.05bpw branch and the official release both carry
    (`self_attn.indexer.index_kpool_compress_{ape,gate}` on all 11 MLA layers).
    Loading it would leave the sparse-attention indexer's k-pool compression
    randomly initialised, and the resulting number would describe a model
    nobody has.
    """
    import sys as _sys
    tools = SUITE_ROOT / "engines" / "tools"
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    try:
        import exl3hf_surface as xs3
    except Exception as exc:                             # noqa: BLE001
        con.warn("completeness gate skipped: %s" % redact(str(exc)))
        return
    try:
        artifact_wm = fetch_json(repo_id, "model.safetensors.index.json",
                                    revision=revision)["weight_map"]
        official_wm = fetch_json("zai-org/GLM-5.3-Flash-BF16",
                                    "model.safetensors.index.json",
                                    revision=OFFICIAL_BF16_REVISION)["weight_map"]
        maps = [artifact_wm]
        mtp = safetensors_header(repo_id, "mtp.safetensors", revision=revision)
        if mtp:
            maps.append({name: "mtp.safetensors" for name in mtp})
    except HFError as exc:
        con.warn("completeness gate skipped: %s" % redact(str(exc)))
        return

    planned = xs3.planned_names(maps)
    want = {n for n in official_wm if not xs3._ROUTED.search(n)}
    missing = sorted(want - set(planned))
    duplicated = sorted({n for n in planned if planned.count(n) > 1})         if len(set(planned)) != len(planned) else []
    plan.setdefault("target", {})["nonrouted_completeness"] = {
        "official_nonrouted": len(want), "planned": len(set(planned)),
        "missing": len(missing), "duplicated": len(duplicated),
    }
    if missing:
        raise Refusal(
            "this release is missing %d of the official model's %d non-routed "
            "tensors, so it cannot be loaded complete" % (len(missing), len(want)),
            ["first missing: %s" % m for m in missing[:4]]
            + ["... and %d more" % (len(missing) - 4) if len(missing) > 4 else "",
               "",
               "The streaming lane loads the non-routed tensors AS the model. A "
               "tensor the release does not ship would be randomly initialised by "
               "transformers, and the measured number would describe a model "
               "nobody has.",
               "Read from the release's own index at the pinned revision.",
               "Nothing was created. $0.00 spent."])
    con.ok("non-routed completeness", "%d/%d official tensors, no duplicates"
           % (len(set(planned)), len(want)))


# Registry tensor_class <- real module-name mapping, most-specific first.
# Deliberately duplicated from engines/tools/derive_scope.py's vocabulary rather
# than imported: this gate must keep working if that tool is refactored, and
# a silent vocabulary drift here would turn a REFUSAL into a pass.
_SCOPE_CLASS_PATTERNS = [
    ("other", [r"visual\.", r"vision"]),
    ("lm_head", [r"(^|\.)lm_head"]),
    ("embed_tokens", [r"embed_tokens"]),
    ("mtp", [r"(^|\.)mtp\.", r"\.mtp$"]),
    ("moe.experts", [r"experts\.\d+\."]),
    ("moe.shared_expert", [r"shared_expert"]),
    ("moe.router", [r"mlp\.gate$"]),
    ("attn.qkv", [r"qkv_proj", r"self_attn\.(q|k|v)_proj", r"\.wq_b", r"q_[ab]_proj",
                  r"kv_a_proj_with_mqa"]),
    ("attn.o", [r"o_proj"]),
    ("mlp.gate", [r"mlp\.gate_proj"]),
    ("mlp.up", [r"mlp\.up_proj"]),
    ("mlp.down", [r"mlp\.down_proj"]),
]


def _scope_class_of(name: str) -> Optional[str]:
    for cls, pats in _SCOPE_CLASS_PATTERNS:
        for p in pats:
            if re.search(p, name):
                return cls
    return None


def _refuse_scope_contradicted_by_release(con: Console, repo_id: str,
                                          revision: str, surface,
                                          scope: Optional[Dict[str, Any]],
                                          plan: Dict[str, Any]) -> None:
    """Is the supplied --scope-json actually THIS release's recipe?

    `--scope-json` copies its file verbatim into the sealed receipt and into
    the artifact record, and its own help says the file "must be READ off the
    release, never assumed".  Nothing enforced that.  A producer who publishes
    several rates on several BRANCHES of one repo -- turboderp ships 4.05,
    3.05 and 2.05bpw that way -- makes the failure trivially easy: the scope
    file for a sibling branch is a valid file, describes the same repo, names
    the same classes, and is wrong in every rate.

    That is not a cosmetic error.  `scope_digest` is computed over these
    assignments and the comparability key is computed over the digest, so a
    wrong scope files the row under a group describing a recipe the artifact
    does not have -- the exact confusion this registry exists to prevent.

    Decidable for free from the release's own published header, so it runs at
    plan time, before anything is rented.
    """
    if not scope:
        return
    claimed = {a.get("tensor_class"): a.get("bits_per_weight")
               for a in (scope.get("assignments") or [])
               if a.get("treatment") == "quantized"}
    if not claimed:
        return

    # (1) The head, from evidence the surface sniffer already read.
    declared_head = (getattr(surface, "evidence", None) or {}).get("head_bits")
    if declared_head is not None and claimed.get("lm_head") not in (None, declared_head):
        raise Refusal(
            "the supplied --scope-json says lm_head is %s bits; this release's own "
            "quantization_config declares head_bits %s"
            % (claimed.get("lm_head"), declared_head),
            ["The scope file describes a DIFFERENT artifact than the one being "
             "measured -- most often a sibling branch of the same repo.",
             "scope_digest, and therefore the comparability key, is computed over "
             "these assignments: publishing this would file the row under a group "
             "describing a recipe this artifact does not have.",
             "Derive the scope from THIS revision (%s) or omit --scope-json and "
             "take the honest 'unknown' default." % (revision or "")[:12],
             "Nothing was created. $0.00 spent."])

    # (2) Every quantized class, from the per-module rates the release itself
    #     publishes.  exl3 states `bits_per_weight` per module in
    #     quantization_config.json; no other surface publishes a per-class rate
    #     we can check, so for those the head check above is the whole gate.
    if surface.surface != "exl3hf":
        con.ok("scope vs release", "head_bits %s agrees" % declared_head)
        return
    try:
        qc = fetch_json(repo_id, "quantization_config.json", revision=revision)
    except HFError as exc:
        con.warn("scope gate: per-class check skipped: %s" % redact(str(exc)))
        return
    storage = qc.get("tensor_storage") or {}
    if not storage:
        con.warn("scope gate: release publishes no tensor_storage; head check only")
        return

    observed: Dict[str, set] = {}
    for name, entry in storage.items():
        bits = entry.get("bits_per_weight")
        if bits is None:
            continue
        cls = _scope_class_of(name)
        if cls:
            observed.setdefault(cls, set()).add(bits)

    # The header's own scalars cover classes tensor_storage spells per-module
    # under names the class map cannot reach (the MTP sidecar is a separate
    # file; the vision tower has its own rate).
    for key, cls in (("mtp_bits", "mtp"), ("vision_bits", "other")):
        if qc.get(key) is not None:
            observed.setdefault(cls, set()).add(qc[key])

    mismatch = []
    for cls, bits in sorted(claimed.items()):
        seen = observed.get(cls)
        if not seen or bits is None:
            continue                      # class absent from the header: not decidable
        if bits not in seen:
            mismatch.append((cls, bits, sorted(seen)))
    plan.setdefault("target", {})["scope_crosscheck"] = {
        "classes_checked": len([c for c in claimed if c in observed]),
        "mismatched": len(mismatch),
    }
    if mismatch:
        raise Refusal(
            "the supplied --scope-json contradicts this release's own published "
            "per-module rates in %d tensor class(es)" % len(mismatch),
            ["%s: scope says %s bits, the release publishes %s"
             % (c, b, "/".join(str(x) for x in seen)) for c, b, seen in mismatch]
            + ["",
               "Read from %s/quantization_config.json at revision %s."
               % (repo_id, (revision or "")[:12]),
               "scope_digest, and therefore the comparability key, is computed over "
               "these assignments -- a wrong scope does not merely mislabel the row, "
               "it files it under the wrong comparability group.",
               "Nothing was created. $0.00 spent."])
    con.ok("scope vs release", "%d quantized classes agree with the release's own "
           "per-module rates" % len([c for c in claimed if c in observed]))


def _verify_tr3_seal(con: Console, repo_id: str, revision: str,
                     plan: Dict[str, Any]) -> None:
    """Recompute the release's OWN seal from its metadata, before renting.

    A TR3-published release is the one third-party surface in this suite that
    seals itself, and every claim of that seal is checkable from three small
    files -- config.json, model.safetensors.index.json and the two receipts --
    which is a few hundred kilobytes and no rental at all. Doing it here rather
    than on the instance means a release whose seal does NOT reproduce costs
    $0.00 to reject, and one whose seal DOES reproduce arrives at the box with
    its verification already recorded in the plan.

    It also subsumes the exl3hf completeness gate: check 7 is name-set equality
    against the official release's own non-routed set.
    """
    import sys as _sys
    tools = SUITE_ROOT / "engines" / "tools"
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    try:
        import tr3_surface as tr3s
    except Exception as exc:                             # noqa: BLE001
        con.warn("seal gate skipped: %s" % redact(str(exc)))
        return
    import tempfile

    try:
        weight_map = fetch_json(repo_id, "model.safetensors.index.json",
                                revision=revision)["weight_map"]
        blobs = {name: fetch_file(repo_id, name, revision=revision)
                 for name in ("config.json", tr3s.ABI_FILE, tr3s.MATERIALIZATION_FILE)}
    except HFError as exc:
        con.warn("seal gate skipped: %s" % redact(str(exc)))
        return
    with tempfile.TemporaryDirectory(prefix="tr3-seal-") as tmp:
        root = Path(tmp)
        for name, blob in blobs.items():
            (root / name).write_bytes(blob)
        # the index is re-serialised byte-exactly by re-fetching it raw: the
        # seal digests the FILE, not our parse of it
        index_bytes = fetch_file(repo_id, "model.safetensors.index.json",
                                 revision=revision)
        (root / "model.safetensors.index.json").write_bytes(index_bytes)
        try:
            seal = tr3s.verify_seal(
                root, weight_map,
                config_path=root / "config.json",
                index_path=root / "model.safetensors.index.json")
        except ValueError as exc:
            raise Refusal(
                "this release's PUBLISHED seal does not reproduce",
                [redact(str(exc)),
                 "",
                 "A seal that does not reproduce is worse than no seal: it "
                 "invites the reader to trust a claim nobody checked.",
                 "Recomputed from the release's own bytes at the pinned "
                 "revision. Nothing was created. $0.00 spent."])
    passed = sum(1 for c in seal["checks"] if c["passed"])
    plan.setdefault("target", {})["seal_verification"] = {
        "verified": True, "checks_passed": passed,
        "checks": [c["check"] for c in seal["checks"]],
        "materialization_receipt_sha256": seal["materialization"]["receipt_sha256"],
        "plan_sha256": seal["abi"]["plan_sha256"],
        "exllamav3_git_commit": seal["abi"]["exllamav3_git_commit"],
        "nonrouted_native_exact": seal["materialization"]["nonrouted_native_exact"],
        "serving_reader_qualified": seal["abi"]["serving_reader_qualified"],
    }
    plan["target"]["nonrouted_completeness"] = {
        "official_nonrouted": seal["materialization"]["native_tensor_count"],
        "planned": seal["materialization"]["native_tensor_count"],
        "missing": 0, "duplicated": 0,
    }
    con.ok("published seal", "%d/%d claims recomputed from the release's own bytes"
           % (passed, len(seal["checks"])))


def _verify_gguf_readable(con: Console, repo_id: str, revision: str,
                          surface, plan: Dict[str, Any]) -> None:
    """Prove the chosen build's ggml types are decodable, from its own headers.

    This gate exists because the obvious shortcut is wrong. A build's NAME
    looks like it names its quantization -- and for the unsloth "UD" (Unsloth
    Dynamic) recipes it does not: `UD-Q2_K_XL` carries IQ2_XS/IQ3_XXS/IQ4_XS
    tensors and `UD-Q3_K_XL` carries IQ3_XXS/IQ4_XS, neither of which
    gguf_surface v1 has a kernel for. A planner that trusted the directory name
    would rent a box, download 100 GB and be refused at census time by a type
    it could have read for free.

    A GGUF header is at the front of the file and `gguf_surface` reads https
    locations by range request, so the whole check costs a few hundred
    kilobytes and no rental. It answers three questions the plan should not
    guess: every type is supported, the architecture and geometry are this
    model's, and the container's non-routed tensors are a bijection with the
    official non-routed name set (nothing missing, nothing stray).

    Skipped -- loudly -- where numpy is not importable, because `bin/` is
    stdlib-only by policy (AGENTS.md) and the surfaces under `engines/tools/` are
    not. A skipped gate is a warning, never a silent pass.
    """
    import sys as _sys
    tools = SUITE_ROOT / "engines" / "tools"
    if str(tools) not in _sys.path:
        _sys.path.insert(0, str(tools))
    try:
        import gguf_surface as ggs
    except Exception as exc:                             # noqa: BLE001
        con.warn("gguf readability gate skipped: %s" % redact(str(exc)))
        con.warn("  the build's ggml types will not be checked until the "
                 "instance runs census -- after the fetch is paid for")
        return
    urls = ["%s/%s/resolve/%s/%s" % (HF_ENDPOINT, repo_id, revision, name)
            for name, _ in surface.artifact_files]
    try:
        loaded = ggs.load_gguf_surface(urls, repo=repo_id, revision=revision,
                                       require_file_hashes=False)
    except ValueError as exc:
        raise Refusal(
            "this GGUF build cannot be decoded by gguf_surface v1",
            [redact(str(exc)),
             "",
             "Read from the build's OWN headers at the pinned revision, not "
             "from its name: unsloth's UD recipes mix IQ types into builds "
             "whose names say Q2_K/Q3_K.",
             "Another build of the same repo may well be measurable -- "
             "`--path <build>` picks one, and the planner lists them.",
             "Adding a type means adding a kernel WITH a bitwise-vs-gguf-py "
             "proof (engines/tools/selftest_gguf_offline.py), not skipping tensors.",
             "Nothing was created. $0.00 spent."])
    except Exception as exc:                             # noqa: BLE001
        con.warn("gguf readability gate skipped: %s" % redact(str(exc)))
        return
    summary = ggs.surface_summary(loaded)
    try:
        official = fetch_json("zai-org/GLM-5.3-Flash-BF16",
                              "model.safetensors.index.json",
                              revision=OFFICIAL_BF16_REVISION)["weight_map"]
        bijection = ggs.verify_nonrouted_bijection(loaded.census, official.keys())
    except (HFError, ValueError) as exc:
        con.warn("gguf non-routed bijection skipped: %s" % redact(str(exc)))
        bijection = None
    if bijection is not None and not bijection.get("bijection_ok"):
        raise Refusal(
            "this GGUF build's non-routed tensors are not the official "
            "non-routed set",
            ["missing %d, stray %d" % (len(bijection.get("missing") or []),
                                       len(bijection.get("stray") or [])),
             "The streaming lane loads the decoded non-routed tensors AS the "
             "model, so a name the container does not carry would be randomly "
             "initialised and the number would describe a model nobody has.",
             "Nothing was created. $0.00 spent."])
    plan.setdefault("target", {})["gguf_verification"] = {
        "verified": True,
        "architecture": summary.get("architecture"),
        "tensor_count": summary.get("tensor_count"),
        "streamed_routed_modules": summary.get("streamed_routed_modules"),
        "nonrouted_tensors_from_artifact":
            summary.get("nonrouted_tensors_from_artifact"),
        "type_census": summary.get("type_census"),
        "scope_policy": summary.get("scope_policy"),
        "checkpoint_identity_sha256": summary.get("checkpoint_identity_sha256"),
        "seal_disclosure": summary.get("seal_disclosure"),
        "nonrouted_bijection_ok": (bijection or {}).get("bijection_ok"),
        "read_from": "https range requests over the build's own headers",
    }
    plan["target"]["nonrouted_completeness"] = {
        "official_nonrouted": len([n for n in (official or {})
                                   if ".mlp.experts." not in n]) if bijection else None,
        "planned": summary.get("nonrouted_tensors_from_artifact"),
        "missing": len((bijection or {}).get("missing") or []),
        "duplicated": 0,
    }
    meta = summary.get("quant_metadata") or {}
    version = meta.get("general.quantization_version")
    quantized_by = meta.get("general.quantized_by")
    plan["target"].update({
        # The receipt's artifact block is built from these. Without them the
        # seal would call a llama.cpp container an EXL3 one quantized by
        # exllamav3, because those are the defaults four of the five surfaces
        # share.
        "container": "gguf",
        "quantizer_tool": ("llama.cpp (quantized_by: %s)" % quantized_by
                           if quantized_by else "llama.cpp"),
        "quantizer_version": (("gguf quantization_version %s" % version)
                              if version is not None else None),
        # NOMINAL bits come from the build name and are already on the target.
        # This is the artifact's own rate, computed from ggml block traits over
        # every tensor it stores -- the number that shows a "Q4_K_XL" build is
        # really ~5 bits/weight because everything outside the routed experts
        # is Q8_0.
        "bits_per_weight_effective": ggs.measured_bits_per_weight(loaded),
    })
    imatrix = {k: v for k, v in meta.items() if k.startswith("quantize.imatrix.")}
    if imatrix:
        # Calibration is a comparability fact, not trivia: an imatrix-calibrated
        # build and an uncalibrated one at the same nominal rate are different
        # artifacts. The submission schema's codec block has no calibration
        # field (additionalProperties:false), so it travels as a disclosure --
        # and because it is a claim about HOW the artifact was made, it carries
        # its own pinned source (PROV-014/015).
        plan.setdefault("disclosures", []).append({
            "code": "imatrix_calibrated",
            "severity": "info",
            "affects_comparability": False,
            "asserts_provenance": True,
            "detail": ("the build declares importance-matrix calibration in its "
                       "own GGUF metadata: %s. A calibrated quant and an "
                       "uncalibrated one at the same nominal rate are different "
                       "artifacts."
                       % ", ".join("%s=%s" % (k.split(".", 2)[2], imatrix[k])
                                   for k in sorted(imatrix))),
            "sources": [{
                "kind": "hf_file",
                "uri": "%s/%s/resolve/%s/%s"
                       % (HF_ENDPOINT, repo_id, revision,
                          surface.artifact_files[0][0]),
                "note": "general/quantize KV of the container's own header, "
                        "read at the pinned revision",
            }],
        })
    effective = plan["target"].get("bits_per_weight_effective")
    if effective:
        con.kv("measured rate", "%.4f bits/weight over every tensor the "
                                "container stores (the name says %g)"
               % (effective, surface.bits or 0.0))
    census = summary.get("type_census") or {}
    con.ok("gguf readable", "%s, %d tensors, types %s"
           % (summary.get("architecture"), summary.get("tensor_count") or 0,
              ", ".join("%s x%d" % (t, census[t]) for t in sorted(census))))


#: Which table in `engines/tools/stream_score.py` holds a surface's profiles. Named
#: rather than derived: the constants are not a mechanical transform of the
#: surface string (`tr3-published` lives in TR3_PROFILES), and a refusal that
#: sends the reader to a constant that does not exist is worse than one that
#: sends them nowhere.
PROFILE_TABLE_NAMES = {
    "exl3hf": "EXL3HF_PROFILES",
    "tr3-published": "TR3_PROFILES",
    "dione": "DIONE_PROFILES",
    # GGUF has no profile TABLE, because it has no per-rate profile: one
    # reader, one receipt family and one format-wide student label serve every
    # build. The constant to read is the label itself.
    "gguf": "GGUF_STUDENT_LABEL",
}


def resolve_profile(lane_spec, surface: Optional[str], bits: Optional[float]) -> Optional[str]:
    """The engine --profile for this (surface, bits), or None.

    The surface-scoped map wins over the bits-only one.  Two surfaces publish
    at the same nominal rate -- a 4.0-bpw TR3 release and a 4.0-bpw Dione
    release are the same number, a different codec, a different scope and a
    different receipt family -- so a bits-only key cannot name a profile once
    both exist.

    A surface map may also be RATE-INDEPENDENT, spelled with the key "*": the
    profile names the receipt family and the student label, and for a format
    whose label is format-wide those do not vary with the bit rate. GGUF is the
    case: one reader, one receipt family, one student label
    ("gguf-llamacpp"), and a dozen builds at a dozen rates behind them. "*" is
    checked BEFORE the bits keys so such a surface resolves even when the rate
    is unknown -- which it legitimately is for a mixed-type build whose name
    claims one number and whose headers hold four.
    """
    if lane_spec is None:
        return None
    by_surface = getattr(lane_spec, "profile_map_by_surface", None) or {}
    generic = getattr(lane_spec, "profile_map", None) or {}
    if surface == "native-bf16":
        return (by_surface.get(surface) or {}).get("native") or generic.get("native")
    wildcard = (by_surface.get(surface or "") or {}).get("*")
    if wildcard:
        return wildcard
    if bits is None:
        return None
    keys = []
    for candidate in (bits, round(float(bits), 4)):
        text = ("%g" % float(candidate))
        if text not in keys:
            keys.append(text)
        text2 = str(candidate)
        if text2 not in keys:
            keys.append(text2)
    scoped = by_surface.get(surface or "") or {}
    for key in keys:
        if key in scoped:
            return scoped[key]
    # A surface with its own map is AUTHORITATIVE for that surface: falling
    # through to the bits-only map would hand a Dione release a TR3 profile.
    if scoped:
        return None
    for key in keys:
        if key in generic:
            return generic[key]
    return None


def plan(args: argparse.Namespace, con: Console, jl: JL) -> Dict[str, Any]:
    """Everything that must be true BEFORE money is spent."""
    plan: Dict[str, Any] = {"job_id": job_id_for(args), "created": False}
    engines = load_engines()
    if args.lane not in engines:
        raise Refusal("no engine configured for lane %r" % args.lane,
                      ["known lanes: " + ", ".join(sorted(engines))])

    # -- registry front gate: is this artifact already measured? -----------
    # BEFORE any preflight, because "the answer already exists" is the
    # cheapest possible outcome of a cloud run.
    if getattr(args, "role", "quant") == "root":
        # The front gate answers "does a published measurement of this artifact
        # already exist?".  A root capture produces no measurement row -- it
        # produces the dataset later rows are measured AGAINST -- so the gate
        # has nothing to say here and asking it would print rows about a
        # different question.
        plan["registry_check"] = "not-applicable-for-root"
    elif getattr(args, "skip_registry_check", False):
        con.warn("--skip-registry-check: not asking the registry first")
        plan["registry_check"] = "skipped"
    else:
        from fidelity.registry_client import front_gate

        gate = front_gate(
            repo=args.model, revision=args.revision,
            path_hint=getattr(args, "path", None),
            source=getattr(args, "registry", "auto"),
            force=getattr(args, "force", False),
            accept_measured_revision=getattr(args, "accept_measured_revision",
                                             False),
            con=con)
        plan["registry_check"] = gate["status"]
        if gate["status"] == "already-measured":
            plan["status"] = "already-measured"
            return plan
        if gate["status"] == "stale-refused":
            raise Refusal(
                "this repo was measured at a pinned revision that is not the "
                "one you asked about (rows printed above)",
                ["pass --accept-measured-revision to target the measured commit",
                 "pass --force to measure the new commit as a NEW artifact "
                 "record"])
        if gate.get("status") == "proceed-stale-accepted" and \
                gate.get("measured_revision"):
            args.revision = gate["measured_revision"]
        con.say("")

    con.say("PREFLIGHT%s" % (" " * 54 + "(no spend yet)"))

    # -- tooling -----------------------------------------------------------
    if jl.available():
        jl.require()
        con.ok("jl %s" % jl.version)
    elif args.dry_run:
        con.warn("jl not installed -- dry run continues without it")
    else:
        raise JLNotInstalled(
            "the `jl` CLI is not on PATH.\n"
            "  install:  uv tool install jarvislabs\n"
            "  auth:     jl setup --token <your-token> --yes   (or export JL_API_KEY)")

    balance = jl.balance() if jl.available() else None
    if balance is not None:
        con.ok("account balance", "$%.2f" % balance)
        plan["balance_before"] = balance

    token = hf_token()
    con.ok("HF_TOKEN", "present (redacted)" if token else "absent (public repos only)")
    plan["hf_token_used"] = bool(token)

    # -- teardown teeth ----------------------------------------------------
    max_runtime = parse_duration(args.max_runtime)
    plan["max_runtime_seconds"] = max_runtime
    if reaper_installed():
        con.ok("lease reaper", "installed")
    elif max_runtime <= 2 * 3600:
        con.ok("lease reaper", "not installed; --max-runtime %s is within the 2h "
                              "self-limiting window" % args.max_runtime)
    elif args.i_accept_leak_risk:
        con.warn("no reaper installed and --max-runtime %s > 2h. You accepted the "
                 "leak risk: if this controller dies without running its trap, the "
                 "instance keeps billing until a human runs `bin/measure-cloud "
                 "reaper --sweep`." % args.max_runtime)
    else:
        refusal = Refusal(
            "no teardown backstop for a %s run" % args.max_runtime,
            ["install the reaper:  bin/measure-cloud reaper --install",
             "or cap the run:      --max-runtime 2h",
             "or accept the risk:  --i-accept-leak-risk",
             "Nothing was created. $0.00 spent."])
        if not args.dry_run:
            raise refusal
        # A dry run's job is to surface EVERY problem in one pass, not to stop
        # at the first one. Record it as a would-refuse and keep validating.
        would_refuse(con, plan, refusal)

    # -- target ------------------------------------------------------------
    con.say("")
    offline = False
    try:
        target = repo_meta(args.model, "model", args.revision or "main")
    except HFError as exc:
        if not args.dry_run:
            raise
        # A 404/401/403 is a verdict about THIS repo, not a network problem.
        # Continuing on the pinned census would print a complete, confident,
        # entirely fictional plan for a repo that does not exist -- which is
        # the exact failure a typo produces, and --dry-run's whole job is to
        # catch it.  Only a transport error earns the offline fallback.
        if re.search(r"HTTP (?:401|403|404)\b", str(exc)):
            con.err(str(exc))
            plan.setdefault("would_refuse", []).append(
                "target %s could not be resolved on Hugging Face" % args.model)
            con.say("           the repo id, the revision, or your HF_TOKEN is wrong;")
            con.say("           nothing below this line describes YOUR model.")
            raise Refusal(
                "target %s could not be resolved on Hugging Face" % args.model,
                ["check the repo id and --revision",
                 "for a private or gated repo:  export HF_TOKEN=...",
                 "Nothing was created. $0.00 spent."])
        con.warn("cannot reach Hugging Face (%s); dry run continues with the "
                 "pinned GLM-5.3-Flash census" % exc)
        target, offline = None, True

    if target is not None:
        con.kv("target", target.repo_id)
        con.kv("revision", "%s  (resolved from %s)"
               % (target.revision, target.requested_revision))
        if target.last_modified:
            con.kv("last modified", target.last_modified)
        con.kv("files / size", "%d / %s" % (len(target.files),
                                            human_bytes(target.total_bytes)))
        surface = sniff_surface(target, getattr(args, "path", None))
        if surface.evidence.get("gguf_builds"):
            con.kv("builds", "%d published at this revision%s"
                   % (len(surface.evidence["gguf_builds"]),
                      ("; measuring %s" % surface.path) if surface.path else ""))
        con.kv("surface", "%s%s" % (surface.surface,
                                    ("  codec %s@%s" % (surface.codec_family, surface.bits))
                                    if surface.codec_family else ""))
        if surface.exllamav3_pin:
            con.kv("exllamav3 pin", surface.exllamav3_pin)
        if surface.nonrouted_native is not None:
            con.kv("non-routed native", surface.nonrouted_native)
        if surface.tp_sliced:
            con.kv("tensor-parallel", "pre-sliced, world size %s" % surface.tp_world_size)
        plan["target"] = {
            "repo_id": target.repo_id, "revision": target.revision,
            "size_bytes": target.total_bytes, "files": len(target.files),
            "last_modified": target.last_modified,
            "surface": surface.surface, "codec": surface.codec_family,
            "bits": surface.bits, "exllamav3_pin": surface.exllamav3_pin,
            "nonrouted_native": surface.nonrouted_native,
            "tp_sliced": surface.tp_sliced,
            # Sniffed evidence the receipt would otherwise drop on the floor.
            # A stock-exllamav3 release has no storage-ABI file, so
            # exllamav3_pin is null -- but its config states the quantizer
            # VERSION, and the artifact record has a field for exactly that.
            "quantizer_version": surface.evidence.get("quantizer_version"),
            "head_bits": surface.evidence.get("head_bits"),
            "quantized_from": surface.evidence.get("original_quantization_config_fmt"),
            # A repo that publishes several artifacts at one revision: the
            # build, and the exact files it is made of. The fetch stage
            # downloads these by name and the capture argv names every one,
            # because for a GGUF the repo commit alone does not identify what
            # was measured.
            "path": surface.path,
            "artifact_files": [{"name": name, "bytes": size}
                               for name, size in surface.artifact_files],
        }
        if getattr(args, "role", "quant") == "root":
            _refuse_quantized_root(con, target, surface, plan)
        if surface.surface == "exl3hf" and not surface.problems:
            _refuse_incomplete_exl3hf(con, target.repo_id, target.revision, plan)
        if not surface.problems:
            _refuse_scope_contradicted_by_release(
                con, target.repo_id, target.revision, surface,
                read_json(args.scope_json) if getattr(args, "scope_json", None) else None,
                plan)
        if surface.surface == "tr3-published" and not surface.problems:
            _verify_tr3_seal(con, target.repo_id, target.revision, plan)
        if surface.surface == "gguf" and not surface.problems:
            _verify_gguf_readable(con, target.repo_id, target.revision,
                                  surface, plan)
        if surface.problems:
            # "no adapter" and "you must pick one" are different verdicts and
            # the second one used to wear the first one's headline. A GGUF
            # shelf IS readable; what it lacks is a choice only the operator
            # can make, and telling them the format is unsupported sends them
            # to write an adapter that already exists.
            raise Refusal(
                ("this repo publishes several artifacts and the measurement "
                 "must name one"
                 if surface.surface != "unknown" else
                 "this artifact cannot be read by any available surface adapter"),
                surface.problems + [
                    "",
                    "This is detected from the repo's own metadata, at a cost of a "
                    "few hundred kilobytes, so it costs nothing to find out.",
                    "Nothing was created. $0.00 spent.",
                ])
        # Knowing WHICH surface this is, is not the same as having a lane that
        # can read it.  Without this check the runner happily prices, rents and
        # downloads 176 GB for an artifact whose bytes no engine in the suite
        # can open -- the failure lands after the money, which is the one place
        # it must never land.
        lane_surfaces = engines[args.lane].surfaces
        if lane_surfaces and surface.surface not in lane_surfaces:
            refusal = Refusal(
                "lane '%s' has no reader for a '%s' artifact"
                % (args.lane, surface.surface),
                ["%s reads: %s" % (args.lane, ", ".join(lane_surfaces)),
                 engines[args.lane].surfaces_note or "",
                 "",
                 "This is the repo's own metadata, so it costs nothing to find "
                 "out here and a full rental to find out on the instance.",
                 "Nothing was created. $0.00 spent."])
            if not args.dry_run:
                raise refusal
            would_refuse(con, plan, refusal)
        # The engine's --profile is part of the plan, not an execute-time
        # afterthought: it names the receipt family, the student label and the
        # bit rate the engine cross-checks against the release's own
        # declaration.  Resolving it here means an artifact this suite has no
        # profile for is refused for $0.00 instead of dying on argparse after
        # the fetch -- or worse, running under the old `or "k6"` fallback and
        # sealing a receipt that calls a third-party quant a K6 payload-store
        # run.
        resolved_profile = resolve_profile(engines[args.lane], surface.surface,
                                           surface.bits)
        plan["profile"] = resolved_profile
        if resolved_profile:
            con.kv("engine profile", "%s  (surface %s, bits %s)"
                   % (resolved_profile, surface.surface, surface.bits))
        else:
            refusal = Refusal(
                "lane '%s' has no --profile for a '%s' artifact at %s bpw"
                % (args.lane, surface.surface, surface.bits),
                ["The profile names the receipt family "
                 "(malaiwah.glm53-<profile>-packed-kld-summary.v1), the student "
                 "label the KLD report expects, and the bit rate the engine "
                 "checks against the release's own declaration.",
                 "FOUR files must agree, and a profile added to only some of "
                 "them fails later and more expensively than this:",
                 "  1. engines/tools/stream_score.py    %s: profile -> (declared "
                 "bpw, student label), and the --profile argparse choices"
                 % PROFILE_TABLE_NAMES.get(surface.surface,
                                           "the <SURFACE>_PROFILES table"),
                 "  2. engines/tools/kld_report.py   the run-label map, "
                 "PROFILE_SURFACE_FAMILY, the student-label map and its "
                 "--profile choices (the display strings are PER PROFILE -- "
                 "read head_bits off the release rather than copying a "
                 "neighbouring rate's line)",
                 "  3. bin/engines.json            lanes.%s."
                 "profile_map_by_surface['%s'] -- a surface with its own map is "
                 "AUTHORITATIVE, so the bits-only profile_map is NOT consulted "
                 "for it and editing that one has no effect"
                 % (args.lane, surface.surface),
                 "  4. registry/tools/registry_add.py  the accepted summary "
                 "schemas, or the row cannot be ingested once you have paid "
                 "for the number",
                 "engines/tools/selftest_kld_report_offline.py derives its coverage "
                 "from stream_score's tables, so run it: a half-added profile "
                 "fails NUM-15 offline, before any rental.",
                 "",
                 "Nothing was created. $0.00 spent."])
            if not args.dry_run:
                raise refusal
            would_refuse(con, plan, refusal)
        # For every other surface the repo IS the artifact. For a GGUF shelf
        # the difference is 2.55 TB vs 200 GB, and pricing the shelf would
        # refuse a run that fits comfortably.
        artifact_bytes = float(surface.artifact_bytes or target.total_bytes)
        bits = float(surface.bits or 4.0)
    else:
        plan["target"] = {"repo_id": args.model, "revision": args.revision,
                          "offline": True}
        artifact_bytes = 176.0 * GB
        bits = 4.0

    # -- panel -------------------------------------------------------------
    con.say("")
    descriptor = load_panel_descriptor(args.panel_descriptor or args.panel)
    panel_bytes = 31.71 * GB
    panel_rev = descriptor.revision
    if not offline:
        try:
            pmeta = repo_meta(descriptor.repo_id, "dataset",
                              args.panel_revision or descriptor.revision)
            panel_rev = pmeta.revision
            panel_bytes = float(pmeta.bytes_matching(descriptor.include))
            con.kv("panel", pmeta.repo_id)
            con.kv("revision", panel_rev)
            con.kv("include", ", ".join(descriptor.include))
            con.kv("fetches", "%s of %s (%.1f%% of the repo)"
                   % (human_bytes(panel_bytes), human_bytes(pmeta.total_bytes),
                      100.0 * panel_bytes / max(1.0, pmeta.total_bytes)))
        except HFError as exc:
            if not args.dry_run:
                raise
            con.warn("panel metadata unavailable (%s); using pinned sizes" % exc)
    con.kv("panel shape", "%d contexts x %d positions = %d scored"
           % (descriptor.contexts, descriptor.positions_per_context,
              descriptor.scored_positions))
    plan["panel"] = dict(descriptor.to_dict(), revision=panel_rev,
                         fetch_bytes=panel_bytes)

    # -- fit ---------------------------------------------------------------
    con.say("")
    cen = C.glm53_flash_census()
    con.say("  fit")
    con.kv("base decoded BF16", "%s  (non-routed %.2f GB + routed %.2f GB)"
           % (human_bytes(cen.total_bf16_bytes), C.gb(cen.nonrouted_bytes),
              C.gb(cen.routed_main_bytes + cen.routed_mtp_bytes)), indent=4)
    con.kv("census source", cen.census_source, indent=4)
    req = C.lane_requirement(cen, args.lane)
    con.kv("lane", "%s -> %d GPU(s), EP%d" % (args.lane, req.gpus, req.ep_size), indent=4)
    con.kv("required VRAM", "%.0f GB/GPU" % C.gb(req.per_gpu_bytes), indent=4)
    for k, v in req.components.items():
        con.kv("  %s" % k, "%.2f GB" % C.gb(v), indent=4)
    plan["census"] = cen.to_dict()
    plan["requirement"] = req.to_dict()

    # exl3hf artifacts additionally materialize their non-routed function as a
    # local BF16 tree (~the model's non-routed footprint) before any capture.
    # A GGUF materializes one too, and for the strongest reason of the four:
    # its non-routed tensors are QUANTIZED, so the view is the artifact's own
    # embeddings / lm_head / attention path decoded, not a re-shard of the
    # official tensors.
    materialized_bytes = (
        cen.nonrouted_bytes
        if plan["target"].get("surface") in ("exl3hf", "tr3-published", "dione",
                                             "gguf") else 0.0
    )
    need = C.storage_need(artifact_bytes=artifact_bytes, panel_bytes=panel_bytes,
                          keep_student_logits=args.keep_student_logits,
                          cold_runs=args.cold_runs,
                          extra_bytes=materialized_bytes)
    storage_gb = args.storage or C.round_up_storage_gb(need.total_bytes)
    con.kv("disk", "%s artifact%s + %s panel + %s transient student logits "
                   "(%d runs) + %s toolchain + 15%% -> %d GB fs"
           % (human_bytes(artifact_bytes),
              (" + %s materialized non-routed" % human_bytes(materialized_bytes))
              if materialized_bytes else "",
              human_bytes(panel_bytes),
              human_bytes(need.transient_student_logits_bytes), args.cold_runs,
              human_bytes(need.toolchain_bytes), storage_gb), indent=4)
    # Providers that rent from a marketplace filter offers themselves, so the
    # requirement has to be in the plan and not only in the selector's locals.
    try:
        plan["required_vram_gb"] = int(req.per_gpu_bytes / (1024 ** 3))
    except Exception:                                     # noqa: BLE001
        pass
    plan["storage_gb"] = storage_gb
    plan["storage_need"] = need.to_dict()

    # -- instance selection -------------------------------------------------
    con.say("")
    con.say("  instance selection")
    offer, table = None, []
    if jl.available():
        try:
            offer, table = select_offer(
                jl.gpus(), required_vram_bytes=req.per_gpu_bytes, gpus=req.gpus,
                spot=args.spot, gpu_type=args.gpu, region=args.region)
        except JLError as exc:
            con.warn("could not query GPU availability: %s" % redact(str(exc)))
    for row in sorted(table, key=lambda r: (r["verdict"] != "ok", r["price"]))[:8]:
        mark = "*" if offer and row["verdict"] == "ok" and \
            row["gpu_type"] == offer.gpu_type and row["region"] == offer.region \
            and abs(row["price"] - offer.price) < 1e-9 else " "
        con.say("    %s %-14s %-5s %4.0f GB  $%-6.2f free=%-3d %s"
                % (mark, row["gpu_type"], row["region"] or "-", row["vram_gb"],
                   row["price"], row["free"], row["verdict"]))
    plan["candidates"] = table

    if offer is None:
        if not table:
            if args.dry_run:
                con.warn("no availability data (jl unreachable); dry run continues")
                offer = None
            else:
                raise Refusal("could not enumerate GPU offers", [
                    "check `jl gpus --json` by hand", "Nothing was created."])
        else:
            closest = sorted(table, key=lambda r: -r["vram_gb"])[:3]
            advice = ["lane %s needs >=%.0f GB/GPU x%d"
                      % (args.lane, C.gb(req.per_gpu_bytes), req.gpus)]
            advice += ["  %-14s %s %4.0f GB $%.2f free=%d -- %s"
                       % (r["gpu_type"], r["region"] or "-", r["vram_gb"],
                          r["price"], r["free"], r["verdict"]) for r in closest]
            if args.lane == "sealed-ep8":
                sreq = C.lane_requirement(cen, "streaming")
                advice.append("--lane streaming needs only >=%.0f GB on ONE GPU"
                              % C.gb(sreq.per_gpu_bytes))
            advice.append("or wait for capacity and retry")
            advice.append("Nothing was created. $0.00 spent.")
            raise Refusal("no available instance fits this lane", advice)
    else:
        plan["chosen"] = {
            "gpu_type": offer.gpu_type, "region": offer.region,
            "vram_gb": round(offer.vram_bytes / 1e9), "price_per_gpu_hour": offer.price,
            "spot": offer.spot, "gpus": req.gpus,
        }
        # Cheapest-that-fits is the right default, but "fits the arithmetic" and
        # "is the hardware this lane's numbers were established on" are not the
        # same claim, and the receipt has to be able to tell them apart.
        validated_on = {"streaming": "H200", "sealed-ep8": "H200"}.get(args.lane)
        if validated_on and offer.gpu_type.upper() != validated_on:
            plan["chosen"]["validated_hardware"] = validated_on
            plan["chosen"]["on_validated_hardware"] = False
            # A WARNING was not enough.  Cheapest-that-fits silently swapped an
            # A100-80GB in for the H200 the streaming lane's rows were all
            # measured on, and TWO measured numbers travel with the GPU:
            #   * minutes/window, which prices the run AND sets --max-runtime.
            #     H200's 7.35 min/window on slower silicon means the deadline
            #     lands mid-run-2 and the rental buys nothing.
            #   * observed_peak VRAM, from which the headroom is computed.
            # And the row itself is no longer same-lane in the sense the
            # registry's comparability key means: bf16 kernels differ across
            # architectures, so the student logits differ. Refuse by default,
            # and make asking for it explicit and disclosed.
            if not args.gpu:
                raise Refusal(
                    "cheapest-that-fits picked %s ($%.2f/h), but lane %s was "
                    "validated on %s, and both constants this plan runs on "
                    "-- minutes/window and the observed VRAM peak -- were "
                    "MEASURED there"
                    % (offer.gpu_type, offer.price, args.lane, validated_on),
                    ["--gpu %s              measure on the validated hardware "
                     "(the comparable choice)" % validated_on,
                     "--gpu %s   deliberately measure on this one; the plan "
                     "records on_validated_hardware=false and the timing "
                     "estimate is NOT transferable" % offer.gpu_type,
                     "Nothing was created. $0.00 spent."])
            con.warn(
                "measuring on %s ($%.2f/h) although lane %s was validated on %s "
                "-- explicitly requested with --gpu. The VRAM arithmetic holds; "
                "the observed-peak and minutes/window figures do NOT transfer "
                "across architectures, so the cost estimate and the deadline "
                "are unbacked here."
                % (offer.gpu_type, offer.price, args.lane, validated_on))
        elif validated_on:
            plan["chosen"]["validated_hardware"] = validated_on
            plan["chosen"]["on_validated_hardware"] = True

    # -- engine ------------------------------------------------------------
    con.say("")
    engine = engines[args.lane]
    probe = engine.probe(SUITE_ROOT)
    plan["engine"] = {"lane": args.lane, "entrypoint": engine.entrypoint,
                      "pinned": engine.pinned, "probe": probe}
    if engine.pinned and probe["present"] and not probe["missing_flags"]:
        con.ok("engine %s" % engine.entrypoint,
               "pinned, %d/%d required flags verified"
               % (len(probe["found_flags"]), len(engine.required_flags)))
    elif engine.pinned and probe["missing_flags"]:
        raise Refusal(
            "engine %s is pinned but its CLI has drifted" % engine.entrypoint,
            ["missing required flags: " + ", ".join(probe["missing_flags"]),
             "re-pin bin/engines.json for lane %s" % args.lane,
             "Nothing was created. $0.00 spent."])
    elif args.dry_run:
        would_refuse(con, plan, Refusal(
            "lane %r has no pinned engine" % args.lane,
            ["engine %s: %s" % (engine.entrypoint, engine.unpinned_reason),
             "pin it in bin/engines.json lanes.%s.engine" % args.lane]))
    else:
        raise EngineUnpinned(engine)

    # -- cost --------------------------------------------------------------
    con.say("")
    rate = (offer.price * req.gpus) if offer else 0.0
    fetch_gb = C.gb(artifact_bytes + panel_bytes)
    timing = getattr(engine, "timing", None) or {}
    if not timing:
        timing = json.loads((SUITE_ROOT / "bin" / "engines.json").read_text(
            encoding="utf-8"))["lanes"][args.lane].get("timing", {})
    # A lane-wide minutes/window stopped being expressible once the surfaces on
    # one lane diverged: a Dione matrix is FOUR TP-rank slices decoded separately
    # and concatenated, so it runs slower per window than a tr3 matrix at the same
    # rate on the same lane. Prefer the measured per-surface figure; fall back to
    # the lane's for a surface nobody has timed yet, and say which was used.
    by_surface = timing.get("minutes_per_window_by_surface") or {}
    surface_key = (plan["target"] or {}).get("surface")
    per_window = float(timing.get("minutes_per_window", 9.05))
    measured = bool(timing.get("measured"))
    timing_basis = "lane"
    if surface_key in by_surface:
        per_window = float(by_surface[surface_key])
        measured = True
        timing_basis = "surface %s" % surface_key
    phases = [
        ("bootstrap", 0.42, "apt + cuda13 + torch + exllamav3 build"),
        ("fetch", max(0.05, fetch_gb / 190.0 / 3600.0 * 1000.0), "%.0f GB @ ~190 MB/s" % fetch_gb),
    ]
    if plan["target"].get("surface") in ("exl3hf", "tr3-published", "dione"):
        phases.append(
            ("materialize", 0.06,
             "%s non-routed -> %s tree (MEASURED 2m06s on exl3hf; tr3 and "
             "dione releases copy rather than decode)"
             % ("dequantize" if plan["target"].get("surface") == "exl3hf"
                else "re-shard", human_bytes(materialized_bytes))))
    phases += [
        ("measure", args.cold_runs * descriptor.contexts * per_window / 60.0,
         "%d run(s) x %d windows @ ~%.2f min (%s%s)"
         % (args.cold_runs, descriptor.contexts, per_window, timing_basis,
            "" if measured else ", ESTIMATED")),
        ("seal + pull", 0.08, ""),
    ]
    total_h = sum(h for _, h, _ in phases)
    plan["timing"] = dict(timing, minutes_per_window=per_window,
                          minutes_per_window_basis=timing_basis)
    storage_rate = storage_gb * 0.00017      # inferred; see the caveat printed below
    point = rate * total_h + storage_rate * total_h
    con.say("  COST ESTIMATE")
    con.kv("rate", "%d x %s %s  $%.2f/h"
           % (req.gpus, offer.gpu_type if offer else "?",
              "spot" if args.spot else "on-demand", rate), indent=4)
    for name, hours, why in phases:
        con.say("    %-14s %-34s %5.2f h  $%6.2f" % (name, why, hours, rate * hours))
    con.say("    %-14s %-34s %5.2f h  $%6.2f"
            % ("storage", "%d GB fs (rate INFERRED, +/-100%%)" % storage_gb,
               total_h, storage_rate * total_h))
    con.say("    %s" % ("-" * 66))
    con.say("    %-50s POINT   $%6.2f" % ("", point))
    band_hi = point * 1.40
    con.say("    %-50s BAND    $%6.2f - $%6.2f" % ("", point, band_hi))
    ceiling = (rate + storage_rate) * (max_runtime / 3600.0)
    con.say("    %-50s CEILING $%6.2f   (--max-runtime %s)"
            % ("", ceiling, args.max_runtime))
    plan["cost_estimate"] = {
        "rate_per_hour": rate, "phases": [{"name": n, "hours": h, "note": w}
                                          for n, h, w in phases],
        "point_usd": point, "band_high_usd": band_hi, "ceiling_usd": ceiling,
        "storage_rate_per_hour": storage_rate,
        "storage_rate_provenance":
            "INFERRED from reconciling one live instance against its list rate; "
            "JarvisLabs publishes no storage line. Treat as +/-100% and rely on "
            "the balance delta for ground truth.",
    }
    if args.cold_runs < 2:
        con.warn(
            "--cold-runs %d produces a receipt the registry will REJECT: a "
            "published row needs run_count >= 2, because one run cannot show "
            "determinism. The measurement still runs and the number is still "
            "real; it just cannot be submitted. Use --cold-runs 2 to submit."
            % args.cold_runs)
        plan["submittable"] = False
    else:
        plan["submittable"] = True

    if not measured:
        con.warn("the measure phase uses an UNMEASURED per-window time (%.1f min). "
                 "Provenance: %s" % (per_window, timing.get("provenance", "unknown")))

    # A kill switch shorter than the work is not a safety feature, it is a way
    # to pay for a run that can never finish.  Catch it here, for free, rather
    # than at hour six with a half-finished panel.
    if total_h > max_runtime / 3600.0:
        refusal = Refusal(
            "--max-runtime %s is shorter than the estimated work (%.2f h)"
            % (args.max_runtime, total_h),
            ["the watchdog would kill this run before it finished, and you would "
             "pay for every hour up to that point",
             "raise it:            --max-runtime %dh" % int(total_h * 1.5 + 1),
             "or shorten the run:  --cold-runs 1"
             + ("" if args.cold_runs == 1 else "  (currently %d)" % args.cold_runs),
             "or pick the cheaper lane: --lane streaming"
             if args.lane == "sealed-ep8" else "",
             "Nothing was created. $0.00 spent."])
        if not args.dry_run:
            raise refusal
        would_refuse(con, plan, refusal)

    if args.max_cost and band_hi > args.max_cost:
        refusal = Refusal(
            "estimated band high $%.2f exceeds --max-cost $%.2f" % (band_hi, args.max_cost),
            ["raise --max-cost, or pick a cheaper lane/GPU",
             "Nothing was created. $0.00 spent."])
        if not args.dry_run:
            raise refusal
        would_refuse(con, plan, refusal)

    # -- teardown plan ------------------------------------------------------
    deadline = time.time() + max_runtime
    name = deadline_name(plan["job_id"], deadline)
    plan["instance_name"] = name
    plan["deadline_epoch"] = deadline
    con.say("")
    con.say("  TEARDOWN PLAN")
    con.say("    L0 controller trap on EXIT/INT/TERM/HUP")
    con.say("    L1 on-instance watchdog: deadline %s, heartbeat %ds"
            % (args.max_runtime, args.heartbeat_timeout))
    con.say("    L2 laptop lease reaper: %s/%s.json" % (LEASE_DIR, plan["job_id"]))
    con.say("    L3 name deadline: %s" % name)
    con.say("    filesystem %s at end"
            % ("KEPT (--keep-fs; it keeps billing)" if args.keep_fs else "destroyed"))
    return plan


# ==========================================================================
# Execution
# ==========================================================================


def execute(args: argparse.Namespace, con: Console, jl: JL,
            plan_data: Dict[str, Any], td: Teardown) -> Dict[str, Any]:
    outdir = Path(args.out or ("./fidelity-runs/%s" % plan_data["job_id"])).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    td.outdir = outdir

    chosen = plan_data["chosen"]
    region = chosen["region"]
    started = time.time()

    # L4: write the lease BEFORE `jl create`, so the window between create
    # returning and the id being known is still covered -- the sweep matches on
    # the name, which is decided here.
    td.lease_path = write_lease(plan_data["job_id"], name=plan_data["instance_name"],
                                deadline=plan_data["deadline_epoch"],
                                machine_id=None, fs_id=None)
    con.step("lease written  %s" % td.lease_path)

    # Adopt an existing instance for this exact job rather than creating a
    # duplicate (a controller that died and was restarted must not double-spend).
    # Match on the JOB PREFIX, not the whole name.  The name is
    # `fidcloud-<job>-x<base36 deadline>` and the deadline is computed from
    # "now", so an exact-name compare never matched on a restart: the very
    # mechanism meant to stop a double-spend created a SECOND instance and a
    # SECOND filesystem while the first was still billing.  The job id is the
    # identity; the suffix is only the L3 expiry stamp.
    job_prefix = "fidcloud-%s-" % plan_data["job_id"]
    for inst in jl.list_instances():
        if (inst.name or "").startswith(job_prefix) and _is_running(inst):
            con.step("adopting existing instance %d for this job (name %s)"
                     % (inst.machine_id, inst.name))
            td.adopt(inst.machine_id, fs_id=inst.fs_id)
            # The lease and the plan must now speak about the instance that
            # EXISTS, or teardown's L3 sweep looks for a name nobody wears.
            plan_data["instance_name"] = inst.name
            break

    if td.machine_id is None:
        fs = jl.fs_create(storage=plan_data["storage_gb"], region=region,
                          name=plan_data["instance_name"])
        td.fs_id = fs.get("fs_id") or fs.get("id")
        con.step("filesystem created  fs %s (%d GB, %s)"
                 % (td.fs_id, plan_data["storage_gb"], region))
        created = None
        try:
            created = jl.create(
                gpu_type=chosen["gpu_type"], num_gpus=chosen["gpus"],
                spot=args.spot, region=region, fs_id=td.fs_id,
                # 100 GB is right ONLY where the big disk is a separate
                # filesystem attached to a small box. Where storage dies with
                # the instance, the whole plan has to fit on it -- and getting
                # this wrong is not a create error, it is a `No space left on
                # device` three stages and 45 minutes into a paid run.
                storage=(100 if getattr(jl, "separable_storage", True)
                         else int(plan_data["storage_gb"])),
                min_vram_gb=int(plan_data.get("required_vram_gb") or 0),
                name=plan_data["instance_name"], template="pytorch")
        finally:
            # The id must be adopted even when `create` raised or answered in a
            # shape we did not expect.  An unadopted id is not a failed create:
            # the box may well be up and billing, and teardown skips every
            # machine step when machine_id is None.  So: take whatever key the
            # response used, and if there is none, ASK THE ACCOUNT -- the name
            # was decided before the call precisely so this lookup is possible.
            mid = _machine_id_of(created)
            if mid is None:
                mid = _find_by_name(jl, plan_data["instance_name"])
                if mid is not None:
                    con.warn("`jl create` did not return a usable machine id; "
                             "recovered %s by name from `jl list`" % mid)
            if mid is not None:
                # NOT int(): RunPod pod ids are opaque strings, and the
                # cast raised AFTER the pod was created -- the controller died
                # holding a running, billing instance it had not adopted, which
                # is the one state teardown cannot cover.
                td.adopt(mid)
        if td.machine_id is None:
            raise RuntimeError(
                "`jl create` returned no machine id and no instance named %s "
                "appeared in `jl list`. If a box was created anyway it is NOT "
                "tracked by this controller -- check `jl list` yourself."
                % plan_data["instance_name"])
        con.step("instance created  machine %s" % td.machine_id)

    heartbeat_stop = threading.Event()
    _start_heartbeat(jl, td, heartbeat_stop, con)
    try:
        _bootstrap_and_run(args, con, jl, td, plan_data, outdir)
    finally:
        heartbeat_stop.set()

    elapsed = time.time() - started
    return _reconcile_cost(jl, td, plan_data, elapsed, outdir, con)


def _start_heartbeat(jl: JL, td: Teardown, stop: threading.Event,
                     con: Console) -> None:
    """Touch a file on the instance so the watchdog knows we are alive."""
    def beat() -> None:
        while not stop.wait(60):
            if td.machine_id is None:
                continue
            try:
                jl.exec(td.machine_id, "touch %s/heartbeat" % td.fs_root, timeout=60)
            except Exception:                           # noqa: BLE001
                pass
    threading.Thread(target=beat, daemon=True).start()


def _job_document(args, plan_data) -> Dict[str, Any]:
    """The on-instance contract: everything the stages need, and nothing secret.

    Written to `$FS/job.json`, which `stage_measure.sh` reads for every stage.
    It carries no token -- the HF token travels separately as a 0600 file.
    """
    panel = dict(plan_data["panel"])
    scope = None
    if getattr(args, "scope_json", None):
        scope = read_json(args.scope_json)
    # The engine profile follows the lane's own profile_map, keyed by the
    # sniffed surface/bits -- never a constant.  (The former hard-coded "k4"
    # was not even a stream_score --profile choice; a streaming measure stage
    # would have died on argparse at hour ~1 of the rental.)
    target = plan_data.get("target") or {}
    profile = plan_data.get("profile")
    if not profile:
        # Offline/adopted plans predate the plan-time resolution; recompute
        # from the same function rather than from a second expression, and
        # REFUSE rather than defaulting -- "k6" is a real profile that names a
        # real receipt family, so falling back to it does not fail loudly, it
        # publishes a wrong label.
        profile = resolve_profile(load_engines().get(args.lane),
                                  target.get("surface"), target.get("bits"))
    role = getattr(args, "role", "quant")
    if not profile and role != "root":
        raise Refusal(
            "no engine --profile for surface %r at %r bpw on lane %r"
            % (target.get("surface"), target.get("bits"), args.lane),
            ["Add it to bin/engines.json lanes.%s.profile_map_by_surface." % args.lane])
    capture: Dict[str, Any] = {}
    if role == "root":
        # A root capture reads no engine profile: there is no quantized surface
        # to decode and no reference to diverge from. What it needs instead is
        # the identity of the dataset it will WRITE, because a capture with no
        # identity cannot be published or cited.
        pdir = getattr(args, "panel_dir", None)
        capture = {
            "role": "root",
            "form": getattr(args, "form", "hidden"),
            "schedule": getattr(args, "schedule", "layer-outer"),
            "panel_dir": (str(Path(pdir).resolve().relative_to(SUITE_ROOT))
                          if pdir else None),
            "panel_id": (json.loads((Path(pdir) / "panel.json").read_text())
                         .get("panel_id") if pdir else None),
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name or args.dataset_id,
            "author": args.measurer,
            # Race mode: the fetch runs INSIDE the capture, ordered by the
            # layer-outer schedule's own needs. `preview_of` is what makes the
            # result a separate identity rather than a first draft of one.
            "race": bool(getattr(args, "race", False)),
            "race_workers": int(getattr(args, "race_workers", 8) or 8),
            "preview_of": getattr(args, "preview_of", None) or None,
            # '' means "run the probe, record it, do not enforce it"; the stage
            # script passes the value through verbatim.
            "sanity_expect": getattr(args, "sanity_expect", "Paris"),
            # hf_capture REFUSES a checkpoint carrying tensors the architecture
            # has no home for, because that is what a quantization path silently
            # failing to engage looks like. A speculative-decoding block is the
            # other thing it looks like, and the two are indistinguishable from
            # the load report alone -- so the override exists and forces a
            # BLOCKING disclosure. Nothing reached it from here: this runner had
            # no flag, so a ROOT capture of any checkpoint that ships an MTP or
            # draft block died at the capture stage with the box already paid
            # for. Found on `malaiwah/GLM-5.2-SIQ-Fruit-bf16` -- this suite's own
            # CI fixture, whose 791 unhoused tensors are its MTP layer 13 -- and
            # the same shape applies to GLM-5.3-Flash and GLM-5.3, which is
            # where the roots that matter are going.
            "allow_unexpected_tensors": bool(
                getattr(args, "allow_unexpected_tensors", False)),
        }
    return {
        "role": role,
        "capture": capture,
        "recipe": "cloud",
        "job_id": plan_data["job_id"],
        "lane": args.lane,
        # Who made the measurement.  Without it seal_receipt defaults to
        # "unknown", so every cloud receipt UNDER-CLAIMED its own provenance
        # even when the registry row was authored correctly by hand (M1 blocker).
        # --measurer overrides; the default is the identity this suite publishes
        # its registry and its receipts under.
        "measurer": {
            "name": args.measurer, "handle": args.measurer,
            "url": "https://huggingface.co/%s" % args.measurer,
            "is_artifact_author": False,
        },
        "reduce_order": args.reduce_order,
        "cold_runs": args.cold_runs,
        "profile": profile,
        "target": plan_data["target"],
        "panel": panel,
        "reference": {
            "reference_ref": panel.get("reference_ref"),
            "teacher_receipt_sha256": panel.get("teacher_receipt_sha256"),
            "teacher_backend_identity_sha256":
                panel.get("teacher_backend_identity_sha256"),
        },
        "environment": {
            "gpu": plan_data["chosen"]["gpu_type"],
            "gpu_count": plan_data["chosen"]["gpus"],
            "tensor_parallel": plan_data["requirement"]["ep_size"],
            # The provider actually rented from, not a constant. A receipt
            # that says "jarvislabs" about a Vast A100 is a false provenance
            # claim in the one block whose job is provenance -- and it was
            # emitted on every non-JarvisLabs run.
            "host": getattr(args, "provider", "jarvislabs"),
        },
        "keep_student_logits": bool(args.keep_student_logits),
        # Disclosures the PLANNER established from the artifact's own metadata
        # (e.g. a GGUF's declared imatrix calibration). seal_receipt appends
        # them to the lane's own; a plan with nothing to add sends [].
        "disclosures": plan_data.get("disclosures") or [],
        # seal_receipt prefers job["scope"] over the registry's existing record
        # and over its own unknown-everything default.
        "scope": scope,
        # The official BF16 release whose config/index the capture binds and
        # whose non-routed name set the exl3hf materializer checks against.
        # PINNED: `main` moving between two measurements of one artifact would
        # silently change what "complete" means.
        "official_bf16_revision": OFFICIAL_BF16_REVISION,
        "produced_by": produced_by_block(SUITE_ROOT, "bin/measure_cloud.py",
                                         dependencies={
                                             "lane": args.lane,
                                             "reduce_order": args.reduce_order,
                                         }),
    }


def _bootstrap_and_run(args, con, jl, td, plan_data, outdir) -> None:
    bundle = SUITE_ROOT / "bin" / "BUNDLE.txt"
    files = [ln.strip() for ln in bundle.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    # A root capture's panel is chosen per run, so it cannot be a static
    # BUNDLE.txt entry -- but it is small (tens of tokens files plus one mask)
    # and it must arrive by the same digest-diff path as everything else, or a
    # resumed box would silently keep an older panel.
    panel_dir = getattr(args, "panel_dir", None)
    if panel_dir:
        pd = Path(panel_dir).resolve()
        if not (pd / "panel.json").is_file():
            raise Refusal("--panel-dir %s has no panel.json" % pd,
                          ["A panel directory is panel.json + arrays/.",
                           "Build one with engines/tools/build_token_panel.py."])
        try:
            pd.relative_to(SUITE_ROOT)
        except ValueError:
            raise Refusal(
                "--panel-dir must live inside the suite checkout (%s)" % SUITE_ROOT,
                ["The bundle uploader addresses files by their path RELATIVE to "
                 "the suite root, so a panel outside it has no remote path.",
                 "Commit the panel under engines/panels/ and pass that path.",
                 "Nothing was created. $0.00 spent."])
        files += sorted(str(f.relative_to(SUITE_ROOT))
                        for f in pd.rglob("*") if f.is_file())
    present = [rel for rel in files if (SUITE_ROOT / rel).is_file()]
    for rel in files:
        if rel not in present:
            con.warn("bundle entry not present locally, skipped: %s" % rel)

    # Upload only what actually differs.  Each `jl upload` is one API round
    # trip of ~10-15 s, so re-sending 49 unchanged files costs ~10 minutes of a
    # billing instance -- paid again on every adoption of a box that already
    # has them.  One `sha256sum` over the remote paths answers the question in
    # a single call; a box with no bundle yet simply returns nothing and
    # everything uploads, which is the same behaviour as before.
    remote_digests: Dict[str, str] = {}
    if present:
        try:
            listing = jl.exec_stdout(
                td.machine_id,
                "sha256sum %s 2>/dev/null || true"
                % " ".join("%s/%s" % (td.fs_root, rel) for rel in present),
                timeout=300, check=False)
            for line in listing.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2 and len(parts[0]) == 64:
                    remote_digests[parts[1].strip()] = parts[0]
        except JLError:
            remote_digests = {}

    stale = [rel for rel in present
             if remote_digests.get("%s/%s" % (td.fs_root, rel))
             != sha256_file(str(SUITE_ROOT / rel))]
    con.step("uploading bundle (%d of %d files; %d already current)"
             % (len(stale), len(present), len(present) - len(stale)))
    made: set = set()
    for rel in stale:
        src = SUITE_ROOT / rel
        remote_dir = "%s/%s" % (td.fs_root, os.path.dirname(rel))
        if remote_dir not in made:
            jl.exec(td.machine_id, "mkdir -p %s" % remote_dir, timeout=120)
            made.add(remote_dir)
        jl.upload(td.machine_id, str(src), "%s/%s" % (td.fs_root, rel))

    # job.json is the contract every stage reads. Without it the stages have no
    # repo id, no revision and no panel include globs, and `fetch_target` exits
    # 2 on an empty repo_id -- after the instance is already billing.
    job_path = outdir / "job.json"
    write_json(str(job_path), _job_document(args, plan_data))
    jl.upload(td.machine_id, str(job_path), "%s/job.json" % td.fs_root)
    con.ok("job.json uploaded", "%d bytes" % job_path.stat().st_size)

    token = hf_token()
    if token:
        # Never on a command line: it would land in the remote process list and
        # in jl's local run records. Written to a 0600 file, shredded at teardown.
        tmp = outdir / ".hf_token"
        tmp.write_text(token, encoding="utf-8")
        os.chmod(tmp, 0o600)
        try:
            jl.exec(td.machine_id, "mkdir -p %s/.secrets" % td.fs_root)
            jl.upload(td.machine_id, str(tmp), "%s/.secrets/hf_token" % td.fs_root)
            jl.exec(td.machine_id, "chmod 600 %s/.secrets/hf_token" % td.fs_root)
        finally:
            tmp.unlink(missing_ok=True)
        con.ok("HF token transported", "0600 file, never argv, shredded at teardown")

    jl.exec(td.machine_id, "mkdir -p %s/logs %s/receipts/done" % (td.fs_root, td.fs_root))
    con.step("arming on-instance watchdog")
    jl.exec(td.machine_id,
            "nohup bash %s/bin/watchdog.sh %d %d %s >%s/logs/watchdog.log 2>&1 &"
            % (td.fs_root, int(plan_data["deadline_epoch"]),
               int(args.heartbeat_timeout), td.fs_root, td.fs_root))

    # The sequence itself lives in fidelity/stages.py, because the container
    # entrypoint drives the SAME stages and two copies of this list is two
    # chances to drift -- a drift that does not crash, it just skips
    # `materialize` and measures a tree nothing decoded.
    stages = stage_sequence(getattr(args, "role", "quant"),
                            race=bool(getattr(args, "race", False)),
                            surface=(plan_data.get("target") or {}).get("surface"))
    for stage in stages:
        _run_stage(args, con, jl, td, plan_data, stage)
        if stage == "setup":
            _preflight_bench(args, con, jl, td, plan_data)



def _preflight_bench(args, con, jl, td, plan_data) -> None:
    """Measure the machine we just rented, before spending the run on it.

    Setup takes minutes; the fetch is the first expensive thing. In between is
    the only moment where the box exists, torch works, and almost nothing has
    been paid for -- so it is where "is this machine worth it?" belongs.

    The case is not hypothetical. One Vast offer for an RTX 4000 Ada wires the
    card at **Gen4 x1 of a Gen4 x16 slot**: same GPU, compute within 3% of a
    sibling host, and 1.6 GB/s host-to-device instead of 11.0. That is a 3-hour
    measurement becoming a 20-hour one, and no catalogue anywhere exposes link
    width. Verified as a wiring fact rather than a sleeping link by reading
    pcie.link.width.current idle AND after four seconds of sustained traffic --
    a parked link ramps, this one does not.

    Reports always; ABORTS only when a threshold was asked for.
    """
    if getattr(args, "no_preflight_bench", False):
        return
    from fidelity.bench import bench_existing, gate

    try:
        doc = bench_existing(jl, td.machine_id, con=con)
    except Exception as exc:                              # noqa: BLE001
        # A failed benchmark must not fail the run: it is an advisory unless a
        # threshold was set, and the run itself is the thing being protected.
        con.warn("preflight benchmark skipped: %s" % redact(str(exc))[:200])
        return
    plan_data["preflight_bench"] = doc
    link = (doc.get("pcie_load") or {}).get("text", "?")
    con.ok("machine measured",
           "h2d %.1f GB/s (cold %.1f), expert GEMM %.0f TF, PCIe %s"
           % (doc.get("h2d_GBps") or 0, doc.get("h2d_cold_GBps") or 0,
              doc.get("expert_gemm_TFLOPs") or 0, link))

    why = gate(doc, min_h2d_gbps=getattr(args, "min_h2d_gbps", None),
               min_gemm_tflops=getattr(args, "min_gemm_tflops", None))
    if why:
        raise Refusal(
            "this machine is below the floor you set: %s" % why,
            ["Setup is paid for; the fetch and the measure stage are not.",
             "Teardown runs next, so nothing is left billing.",
             "Retry to draw a different host, or lower the floor."])


def _run_stage(args, con, jl, td, plan_data, stage: str) -> None:
    """Launch one stage and WAIT for it, surviving spot preemption.

    Every stage is receipt-resumable (`$DONE/<stage>.done`, and per-run capture
    receipts inside `measure`), so a preemption costs at most one stage --
    usually one in-flight cold run. The rule that makes it work: after any
    resume or recreate, ADOPT whatever machine id came back, unconditionally,
    and rewrite the lease before doing anything else. `jl resume` renumbers.
    """
    deadline = plan_data["deadline_epoch"]
    preemptions = 0
    while True:
        con.step("stage %s" % stage)
        started = time.time()
        # DETACHED, in its own session, orphaned to init. The controller's job is
        # to SUPERVISE the stage, not to own it: a controller that dies -- a
        # signal, a closed laptop, a harness that reaps long-lived background
        # tasks -- used to take the remote process group with it. Observed on
        # M2: a two-hour capture was killed at window 22 of 25 by the death of
        # the local process watching it, and the whole run had to be redone.
        # With setsid the stage keeps going and the controller re-attaches to it
        # by done-marker on the next poll, or on a later resume.
        # ATTACH BEFORE LAUNCH.  The stage's own guard is its done-marker,
        # which by definition does not exist while the stage is RUNNING, so a
        # controller that resumes into a live stage used to start a SECOND copy
        # of it: two capture processes writing receipts/run-N/logits/ at once,
        # which is not a crash but a corrupted measurement that looks finished.
        # The probe that answers this already existed for the launcher's
        # "succeeded" case (lesson 44, and note the `[s]tage_measure.sh` bracket
        # class -- `pgrep -f` matches full command lines and the naive pattern
        # finds the probe's own shell).  It just ran too late.  Observed on M4:
        # the harness reaped the controller at 00:59 with 19 of 25 windows
        # captured, and the only safe resume was to wait for the marker by hand.
        run_id = None
        if _stage_is_alive(jl, td, stage):
            con.warn("stage %s is ALREADY RUNNING on %s -- attaching to it "
                     "instead of launching a second copy"
                     % (stage, td.machine_id))
        else:
            run = jl.run_job(
                td.machine_id,
                "%s nohup setsid bash %s/bin/stage_measure.sh %s "
                ">>%s/logs/stage-%s.log 2>&1 </dev/null & echo launched %s"
                % (_stage_env(td), td.fs_root, stage, td.fs_root, stage, stage))
            run_id = (run or {}).get("run_id") or (run or {}).get("id")
        outcome = _await_stage(con, jl, td, run_id, stage, deadline)
        if outcome == "done":
            con.ok("stage %s" % stage, human_duration(time.time() - started))
            return
        if outcome == "failed":
            # Do not assert the teardown's decision here: --hold-on-failure may
            # keep the instance, and this line used to say "will now be
            # destroyed" beside a HELD banner saying the opposite. The teardown
            # prints what it actually did.
            raise RuntimeError(
                "stage %s failed on the instance; its log was pulled to the "
                "output directory (the teardown block below says what happened "
                "to the instance)" % stage)
        if outcome == "deadline":
            raise RuntimeError(
                "--max-runtime reached during stage %s; stopping and tearing "
                "down. Partial receipts have been pulled." % stage)

        # outcome == "preempted"
        preemptions += 1
        lost = time.time() - started
        rate = plan_data["cost_estimate"]["rate_per_hour"]
        plan_data.setdefault("preemption_log", []).append({
            "stage": stage, "at": utcnow(), "old_machine_id": td.machine_id,
            "minutes_lost": round(lost / 60, 1),
            "usd_lost": round(rate * lost / 3600.0, 2),
        })
        if args.on_preempt == "fail" or preemptions > args.max_preemptions:
            raise RuntimeError(
                "preempted %d time(s) during stage %s (limit %d)"
                % (preemptions, stage, args.max_preemptions))
        con.warn("preemption %d during stage %s -- lost %s ($%.2f)"
                 % (preemptions, stage, human_duration(lost), rate * lost / 3600.0))
        _recover(args, con, jl, td, plan_data)
        # setup is idempotent and containers lose apt state across a pause, so
        # it must be re-run before the stage that was interrupted.
        if stage != "setup":
            con.step("re-running setup after recovery (idempotent)")
            r = jl.run_job(td.machine_id,
                           "%s bash %s/bin/stage_measure.sh setup"
                           % (_stage_env(td), td.fs_root))
            _await_stage(con, jl, td, (r or {}).get("run_id"), "setup", deadline)


def _stage_is_alive(jl, td, stage: str) -> bool:
    """Is `stage_measure.sh <stage>` running on the instance right now?

    `[s]tage_measure` and NOT `stage_measure`: `pgrep -f` matches full command
    lines, and this probe's own shell carries the pattern in ITS command line,
    so the naive form finds itself and answers "alive" for a stage that never
    existed (verified against a live instance: `pgrep -f 'stage_measure.sh
    nosuchstage'` -> alive).  The bracket class matches the real process, whose
    cmdline holds "stage_measure.sh", and not the probe, whose cmdline holds
    "[s]tage_measure.sh".  JOURNAL lesson 36 / 44.

    An unreadable instance answers False: the caller then LAUNCHES, and a
    launch into an already-running stage is caught by the stage's own marker
    on the next poll.  Answering True on no evidence would hang the controller
    on a stage that is not there.
    """
    if td.machine_id is None or jl.dry:
        return False
    try:
        out = jl.exec_stdout(
            td.machine_id,
            "pgrep -f '[s]tage_measure.sh %s' >/dev/null 2>&1 "
            "&& echo alive || echo gone" % stage,
            timeout=120, check=False)
    except JLError:
        return False
    return (out or "").strip().splitlines()[-1:] == ["alive"]


#: How many consecutive 120 s polls may read the SAME progress counter before
#: the controller says so out loud.  Five polls is ten minutes; the slowest unit
#: any engine counts is a streaming window at ~24 min, but the meter emits
#: mid-window lines every 30 s, so ten minutes of a frozen counter is not slow
#: work -- it is a stalled one.  (The first GGUF run sat at 0% GPU for two hours
#: and nothing said so.)
_PROGRESS_STALL_POLLS = 5


def _progress_counter(text: str) -> "Optional[int]":
    """The item count from the newest ``progress:`` line in a log tail.

    engines/tools/progress.py renders ``progress: <label> <n>/<total> ...`` (or
    ``progress: <label> <n> ...`` when the total is unknown).  ``n`` counts
    within one unit and RESETS when the next unit starts (each layer fill gets
    its own meter), so this is not a monotonic stage-wide clock and must not be
    read as one.  What it supports is the weaker, sufficient test: the newest
    meter line reading the SAME number, poll after poll, means nothing has
    advanced -- which is the one thing `_stage_is_alive`'s pgrep cannot see,
    because a hung process is still a process.  A reset makes the numbers
    differ, so a reset can only ever CLEAR the suspicion, never raise it.

    Returns None when the tail carries no meter line at all -- an engine
    predating the meter, or a stage that has not reached its loop yet.  None is
    not evidence of a stall and never counts as one.
    """
    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(progress_meter_PREFIX):
            continue
        for token in line[len(progress_meter_PREFIX):].split():
            head = token.split("/")[0]
            if head.isdigit():
                found = int(head)
                break
    return found


#: Duplicated from engines/tools/progress.py rather than imported: bin/ must run on
#: stock python3.9 with no torch and no k6 tools on sys.path, and this is one
#: seven-character string.  bin/selftest_progress.py asserts the two agree.
progress_meter_PREFIX = "progress:"


def _run_call(jl, method: str, td, run_id: str, **kw):
    """Ask a backend about a detached run, whichever way it takes the question.

    THIS IS A MONEY BUG FIX, not a tidy-up. `jlapi.JL.run_status(run_id)` needs
    no instance -- the `jl` CLI knows which box a run belongs to. Every SSH
    backend has to be told, and `sshbase` says so by RAISING:

        def run_status(self, run_id, machine_id=None):
            if machine_id is None:
                raise JLError("run_status needs machine_id on this backend")

    `_await_stage` called `jl.run_status(run_id)` with one argument. So on
    runpod, vast and lambda -- three of the four providers -- that call raised
    on EVERY poll, `state` fell back to "", both the `failed` and the
    `succeeded` branches below became unreachable, and the loop degenerated to
    "wait for the done marker, or for `--max-runtime`". A stage that SUCCEEDS
    still ends the poll (its marker is authoritative). A stage that FAILS never
    does. `run_logs` had the identical signature mismatch, so the text
    fallback -- the one remaining way out -- could not read the log either, and
    its `except JLError: continue` swallowed the evidence.

    Measured on a live Lambda GH200: the capture stage exited non-zero at
    15:03:2x and at 15:12 the controller had not noticed, across four 120 s
    polls, with the instance ACTIVE and the GPU at 0%. On a Fruit-scale run
    that is thirty cents; on an M2/M3 root it is bounded only by
    `--max-runtime`, which is the "$3.07 sitting at 0% GPU" shape
    `docs/CLOUD-RECIPES.md` section 6 already warns about, arriving through a
    different door.

    Passing `machine_id` positionally would break JL, whose signature has no
    such parameter; passing it as a keyword and falling back on TypeError works
    for both and needs no edit to either backend.
    """
    fn = getattr(jl, method)
    try:
        return fn(run_id, machine_id=td.machine_id, **kw)
    except TypeError:
        return fn(run_id, **kw)


def _await_stage(con, jl, td, run_id, stage: str, deadline: float) -> str:
    """Poll until the stage ends. Returns done | failed | preempted | deadline.

    The verdict comes from the run's own STATE and exit code, not from a grep
    over the last 40 log lines.  Text-matching "Traceback" cannot see a stage
    that exits non-zero quietly -- a refusal printed in a shape we did not
    anticipate, a `set -e` abort, an OOM kill -- and such a stage looked exactly
    like one still working, so the controller waited on it until --max-runtime
    and paid for the whole window.  The done MARKER is still consulted first,
    because a stage that finished and wrote its marker is done no matter what
    the run wrapper reports.
    """
    quiet = 0
    last_progress: Optional[int] = None
    stalled_polls = 0
    while True:
        if time.time() > deadline:
            return "deadline"
        time.sleep(120)
        inst = jl.get(td.machine_id) if td.machine_id else None
        if not _is_running(inst):
            # Not running is not automatically preemption -- confirm before
            # acting, because a transient API blip should not trigger a rebuild.
            quiet += 1
            if quiet >= 2:
                con.warn("instance %s status=%s -- treating as preemption"
                         % (td.machine_id, inst.status if inst else "gone"))
                return "preempted"
            continue
        quiet = 0

        # 1. the stage's own marker: authoritative for success.  Compare the
        #    remote STDOUT exactly -- `jl exec --json` echoes the command back
        #    in its payload, so a substring test over the whole response finds
        #    the probe's own words and answers "done" every time.
        try:
            marker = jl.exec_stdout(
                td.machine_id,
                "test -f %s/receipts/done/%s.done && echo yes || echo no"
                % (td.fs_root, stage), timeout=120)
            if marker.strip().splitlines()[-1:] == ["yes"]:
                return "done"
        except (JLError, IndexError):
            pass

        # 2. the managed run's state.
        state, code = "", None
        if run_id is None:
            # ATTACHED, not launched: there is no managed run to ask about, so
            # the marker checked above and the liveness probe are the only
            # signals.  Do not fall through to the `state == ""` paths, which
            # read an unknown run state.
            if _stage_is_alive(jl, td, stage):
                continue
            con.warn("stage %s: attached to a live stage that has now exited "
                     "without writing its done marker" % stage)
            return "failed"
        if run_id:
            try:
                st = _run_call(jl, "run_status", td, run_id)
                if isinstance(st, dict):
                    state = str(st.get("state") or "").lower()
                    code = st.get("exit_code")
            except JLError:
                state = ""
        if state in ("failed", "error", "cancelled", "canceled", "stopped"):
            con.warn("stage %s: run %s state=%s exit_code=%s"
                     % (stage, run_id, state, code))
            return "failed"
        if state == "succeeded":
            # The managed run is the LAUNCHER, not the stage: it starts a
            # detached, own-session process and returns immediately. "succeeded"
            # therefore means "launched", and the only honest signals left are
            # the marker (checked above) and whether the stage process is still
            # alive. Ask the instance, and compare the answer exactly -- a probe
            # whose output can be confused with its own command text answers
            # yes forever (that was M1's lesson 36).
            # ONE implementation of the probe (see _stage_is_alive): this
            # used to be a second copy of the same pgrep, and two copies of a
            # probe are two places for the bracket class to be dropped.
            if _stage_is_alive(jl, td, stage):
                continue
            con.warn("stage %s: no done marker and no live stage process "
                     "(launcher exit_code=%s)" % (stage, code))
            return "failed"

        # 3. still running: surface an early diagnosis from the log if there is
        #    one, but never conclude success from text.
        try:
            logs = _run_call(jl, "run_logs", td, run_id, tail=40) if run_id else ""
        except JLError:
            continue
        text = logs if isinstance(logs, str) else json.dumps(logs)
        if "Traceback" in text or "REFUSED" in text or "stage_measure: error" in text:
            return "failed"

        # 4. liveness is not progress.  `_stage_is_alive` answers "is the
        #    process there", which a hung process answers yes to forever.  The
        #    engine's meter publishes a monotonic item counter; a counter that
        #    has not moved in _PROGRESS_STALL_POLLS polls is REPORTED, and only
        #    reported.  This deliberately does NOT return a verdict: the run may
        #    legitimately be inside a long non-counting phase (a 200 GB fetch, a
        #    materialize), and killing a paid run on a heuristic is a worse
        #    failure than watching a stalled one until --max-runtime.  Teardown
        #    semantics are untouched; the operator gets told.
        seen = _progress_counter(text)
        if seen is None:
            continue
        if last_progress is not None and seen == last_progress:
            stalled_polls += 1
            if stalled_polls == _PROGRESS_STALL_POLLS:
                con.warn(
                    "stage %s: progress counter stuck at %d for %d polls (~%d min) "
                    "-- the process is alive but not advancing; check GPU "
                    "utilization before it burns the whole --max-runtime"
                    % (stage, seen, stalled_polls, stalled_polls * 2))
                stalled_polls = 0
        else:
            stalled_polls = 0
        last_progress = seen


def _recover(args, con, jl, td, plan_data) -> None:
    """Resume or recreate, then ADOPT the returned id before anything else."""
    inst = jl.get(td.machine_id) if td.machine_id else None
    new_id = None
    if inst is not None and inst.status.lower() in ("paused", "pausing", "stopped"):
        con.step("resuming %s" % td.machine_id)
        res = jl.resume(td.machine_id, spot=args.spot)
        new_id = (res or {}).get("machine_id")
    if new_id is None and args.on_preempt in ("resume", "recreate"):
        con.step("recreating instance for this job")
        chosen = plan_data["chosen"]
        res = jl.create(gpu_type=chosen["gpu_type"], num_gpus=chosen["gpus"],
                        spot=args.spot, region=chosen["region"], fs_id=td.fs_id,
                        storage=100, name=plan_data["instance_name"],
                        template="pytorch")
        new_id = (res or {}).get("machine_id")
    if new_id is None:
        raise RuntimeError("could not recover the instance after preemption")
    td.adopt(new_id)   # opaque string on some providers; never int()
    con.ok("adopted machine", str(td.machine_id))


def _reconcile_cost(jl, td, plan_data, elapsed, outdir, con) -> Dict[str, Any]:
    """Four numbers, all printed, none of them trusted alone."""
    rate = plan_data["cost_estimate"]["rate_per_hour"]
    computed = rate * (elapsed / 3600.0)
    inst = jl.get(td.machine_id) if td.machine_id else None
    billed = inst.billed_usd if inst else None
    after = jl.balance()
    before = plan_data.get("balance_before")
    delta = (before - after) if (before is not None and after is not None) else None
    cost = {
        "estimated_usd": plan_data["cost_estimate"]["point_usd"],
        "computed_usd": computed,
        "computed_basis": "controller wall clock from create to destroy, "
                          "bootstrap INCLUDED",
        "billed_usd": billed,
        "billed_basis": "jl get .cost, a running USD total (not a rate)",
        "balance_delta_usd": delta,
        "balance_before": before, "balance_after": after,
        "wall_clock_seconds": elapsed,
        "reconciliation":
            (billed - computed) if (billed is not None) else None,
        # A spot number that hides four restarts is not a truthful cost.
        "preemptions": len(plan_data.get("preemption_log") or []),
        "preemption_log": plan_data.get("preemption_log") or [],
        "usd_lost_to_preemption": round(sum(
            e.get("usd_lost", 0.0) for e in plan_data.get("preemption_log") or []), 2),
        "storage": {
            "filesystem_gb": plan_data.get("storage_gb"),
            "rate_provenance":
                plan_data["cost_estimate"]["storage_rate_provenance"],
        },
    }
    write_json(str(outdir / "cost-receipt.json"), cost)
    return cost


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure-cloud",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subcommand", nargs="?", default=None,
                   choices=(None, "reaper", "adopt"),
                   help="reaper: manage the teardown backstop; adopt: resume a job")
    p.add_argument("--min-h2d-gbps", type=float,
                   help="ABORT after setup if the rented machine's "
                        "host-to-device bandwidth is below this. The streaming "
                        "lane is bandwidth-bound, and a host that wires the "
                        "card at PCIe x1 turns a 3-hour run into a 20-hour one "
                        "while looking identical in the catalogue. 8 is a "
                        "reasonable floor for an x16 slot.")
    p.add_argument("--min-gemm-tflops", type=float,
                   help="ABORT after setup if the measured expert-shape bf16 "
                        "GEMM is below this")
    p.add_argument("--no-preflight-bench", action="store_true",
                   help="skip the post-setup machine measurement entirely")
    p.add_argument("--provider", default="jarvislabs",
                   choices=("jarvislabs", "runpod", "vast", "lambda"),
                   help="which cloud to rent from. The measurement is the same "
                        "on any of them -- fp64 KLD is not vendor-specific -- "
                        "but BITWISE determinism is a per-device property, so a "
                        "row measured on one provider's silicon reproducing on "
                        "another's is a RESULT, not an assumption.")

    t = p.add_argument_group("target")
    t.add_argument("--role", default="quant", choices=("quant", "root"),
                   help="quant (default): measure an artifact's divergence from a "
                        "reference and seal a submission receipt. root: CAPTURE a "
                        "reference model's own activations on a panel and seal a "
                        "publishable fidelity dataset -- no divergence, no "
                        "reference, nothing to compare against yet. A root is paid "
                        "for once and read by every later measurement.")
    t.add_argument("--model", help="HF repo id of the artifact to measure")
    t.add_argument("--revision", help="40-hex pin (default: resolve main and show it)")
    t.add_argument("--path",
                   help="which artifact inside a repo that publishes several at "
                        "one revision (a GGUF shelf: --path UD-Q4_K_XL). The "
                        "planner lists the builds when the choice is yours to make.")
    t.add_argument("--registry", default="auto",
                   help="auto | hf | local[:PATH] -- where the already-measured "
                        "front gate reads from")
    t.add_argument("--skip-registry-check", action="store_true")
    t.add_argument("--force", action="store_true",
                   help="measure even though published rows exist")
    t.add_argument("--accept-measured-revision", action="store_true",
                   help="on revision drift, target the registry's measured commit")

    rt = p.add_argument_group("root capture (--role root)")
    rt.add_argument("--panel-dir",
                    help="LOCAL panel directory (panel.json + arrays/) shipped to "
                         "the instance with the bundle. A root capture for a model "
                         "family with no published panel needs one; build it with "
                         "engines/tools/build_token_panel.py against that family's OWN "
                         "tokenizer. Mutually exclusive with --panel.")
    rt.add_argument("--dataset-id",
                    help="identity of the fidelity dataset to write, e.g. "
                         "minimaxm3-fidelity-root-v1")
    rt.add_argument("--dataset-name", help="human-readable name for that dataset")
    rt.add_argument("--form", default="hidden", choices=("hidden", "logit"),
                    help="hidden (default) stores hidden states and applies the "
                         "head at compare time; logit stores full-vocabulary "
                         "logits, which for a 200k vocab is ~32x larger and is "
                         "why published roots are hidden-form")
    rt.add_argument("--schedule", default="layer-outer",
                    choices=("layer-outer", "window-outer"))
    rt.add_argument("--race", action="store_true",
                    help="RACE MODE (docs/RACE-MODE.md): fetch the checkpoint WHILE "
                         "capturing it, in the order the layer-outer schedule needs "
                         "it, instead of waiting for the whole tree. Merges the "
                         "fetch_target and capture stages. It changes when bytes "
                         "arrive, never which -- the capture_content_digest is the "
                         "one a fetch-then-capture run produces.")
    rt.add_argument("--race-workers", type=int, default=8,
                    help="parallel downloads for --race (default 8)")
    rt.add_argument("--preview-of", metavar="FINAL_DATASET_ID",
                    help="seal this capture as a PRELIMINARY dataset superseded by "
                         "FINAL_DATASET_ID -- one cold run, determinism NOT "
                         "demonstrated, its own dataset id, not_submittable, and a "
                         "blocking disclosure. A preview and a final are two "
                         "datasets, never two versions of one: `reference_id` is a "
                         "comparability-key field, so updating a published root in "
                         "place would silently make old and new rows share a table.")
    rt.add_argument("--sanity-expect", default="Paris",
                    help="the continuation the generation sanity probe requires for "
                         "\"The capital of France is\" (default Paris). Pass '' to "
                         "record the probe without enforcing it -- for a model that "
                         "is genuinely expected to answer otherwise. The probe runs "
                         "either way: it is one extra window through a schedule that "
                         "is already loading every layer.")
    rt.add_argument("--allow-unexpected-tensors", action="store_true",
                    help="proceed when the checkpoint carries tensors this "
                         "architecture has no home for. hf_capture refuses them by "
                         "default because that is what a quantization path failing "
                         "to engage looks like -- and it is ALSO what a "
                         "speculative-decoding block looks like on a model whose MTP "
                         "layer stock transformers does not build. Pass this only "
                         "after checking the unhoused names ARE that block; it "
                         "forces a BLOCKING disclosure into the sealed dataset "
                         "either way.")

    pl = p.add_argument_group("panel (a parameter, never a constant)")
    pl.add_argument("--panel", help="HF dataset id of the panel/teacher")
    pl.add_argument("--panel-revision")
    pl.add_argument("--panel-descriptor",
                    help="JSON file with include globs, contexts, positions")

    ln = p.add_argument_group("lane")
    ln.add_argument("--lane", default="streaming", choices=("streaming", "sealed-ep8"))
    ln.add_argument("--reduce-order", default="fp32", choices=("fp32", "native"))
    ln.add_argument("--cold-runs", type=int, default=None)

    i = p.add_argument_group("instance")
    i.add_argument("--spot", dest="spot", action="store_true", default=True)
    i.add_argument("--on-demand", dest="spot", action="store_false")
    i.add_argument("--gpu", help="override the selector (still fit-checked)")
    i.add_argument("--region")
    i.add_argument("--storage", type=int, help="filesystem GB (default: computed)")
    i.add_argument("--fs-id", type=int)
    i.add_argument("--keep-fs", action="store_true",
                   help="do NOT destroy the filesystem (it keeps billing)")
    i.add_argument("--keep-student-logits", action="store_true")

    s = p.add_argument_group(
        "safety (default-on except --max-cost, which has no default)")
    s.add_argument("--max-runtime", default="6h")
    s.add_argument("--heartbeat-timeout", type=int, default=900)
    s.add_argument("--max-preemptions", type=int, default=3)
    # NOT default-on, unlike its neighbours in this group, and the --help
    # heading says "safety (all default-on)". Left off by default deliberately:
    # a cap the runner picked for you turns a legitimate expensive run into a
    # refusal the user cannot attribute. But the help text has to say so, or a
    # reader budgeting a rental believes a cap is in force when none is.
    s.add_argument("--max-cost", type=float,
                   help="refuse if the estimated cost BAND HIGH exceeds this "
                        "many dollars. NO DEFAULT -- without it there is no "
                        "cost cap, only --max-runtime's ceiling")
    s.add_argument("--on-preempt", default="resume",
                   choices=("resume", "recreate", "fail"))
    s.add_argument("--i-accept-leak-risk", action="store_true")
    s.add_argument("--measurer", default="malaiwah",
                   help="handle credited as the MEASURER on the sealed receipt "
                        "(the artifact's producer is read from the repo id and "
                        "is a separate field)")
    s.add_argument("--scope-json",
                   help="JSON file carrying the artifact's quantization SCOPE "
                        "(policy/head_policy/kv_cache_dtype/assignments), for a "
                        "release that publishes its own per-tensor-class recipe. "
                        "Without it the sealed receipt records the honest default: "
                        "routed experts quantized, everything else 'unknown'. The "
                        "file's content is copied verbatim into job.json and into "
                        "the artifact record -- so it must be READ off the release, "
                        "never assumed.")
    s.add_argument("--hold-on-failure", action="store_true",
                        help="on a FAILED stage, keep the instance alive (receipts "
                             "pulled, secrets shredded, lease kept so the reaper "
                             "still expires it) instead of destroying it. For "
                             "proving an unexercised path, where re-buying a "
                             "finished fetch costs more than the hold.")
    s.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    s.add_argument("--dry-run", action="store_true",
                   help="validate everything, create nothing, spend $0.00")

    o = p.add_argument_group("output")
    o.add_argument("--out", help="default ./fidelity-runs/<jobid>")
    o.add_argument("--install", action="store_true", help="reaper --install")
    o.add_argument("--sweep", action="store_true", help="reaper --sweep")
    o.add_argument("--list", action="store_true", help="reaper --list")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()

    if args.subcommand == "reaper":
        if args.install:
            return reaper_install(con)
        if args.sweep:
            # --dry-run: report what WOULD be destroyed, destroy nothing.
            # Selftests use this so that destruction is never a side effect
            # of "run the selftests" (usability review, 2026-08-28).
            return reaper_sweep(con, dry=args.dry_run)
        return reaper_list(con)

    if getattr(args, "role", "quant") == "root":
        if not args.model or not (args.panel or args.panel_dir):
            con.err("--role root requires --model and one of --panel / --panel-dir")
            return EXIT_REFUSED
        if args.panel and args.panel_dir:
            con.err("--panel and --panel-dir are mutually exclusive")
            return EXIT_REFUSED
        if not args.dataset_id:
            con.err("--role root requires --dataset-id (the identity of the "
                    "dataset being written; a capture with no identity cannot "
                    "be published or cited)")
            return EXIT_REFUSED
        if args.race and args.schedule != "layer-outer":
            con.err("--race needs --schedule layer-outer: every other schedule "
                    "materialises the whole model before the first window, so "
                    "there is no not-yet-arrived layer to overlap the fetch "
                    "with. Nothing was created. $0.00 spent.")
            return EXIT_REFUSED
        if args.preview_of and args.preview_of == args.dataset_id:
            con.err("--preview-of is the same id as --dataset-id. A preview and "
                    "a final are two DATASETS, not two versions of one: "
                    "reference_id is a comparability-key field, so rows measured "
                    "against the one-run bytes and rows measured against the "
                    "full-evidence bytes would land in the SAME comparability "
                    "group and be silently incomparable. Convention: give the "
                    "preview the final id with a `.preview` suffix.")
            return EXIT_REFUSED
        if args.race and not args.preview_of:
            # Not a refusal: racing run 2 of a two-run root under the final id is
            # a perfectly good use of the flag. But sealing run 1 under the FINAL
            # id and publishing it is the exact failure --preview-of exists to
            # prevent, and the moment to say so is before the rental.
            con.warn("--race without --preview-of",
                     "this capture will be sealed under the FINAL dataset id from "
                     "ONE cold run. Fine if a second cold capture and the "
                     "self-compare come before you publish it; if you intend to "
                     "publish now, pass --preview-of <final id> so the preliminary "
                     "result gets its own identity")
        if args.panel_dir:
            # Checked HERE, before the plan and before any spend. The uploader
            # addresses bundle files by their path RELATIVE to the suite root,
            # so a panel outside it has no remote path -- and discovering that
            # inside _bootstrap_and_run means finding out after the instance is
            # already running.
            pd = Path(args.panel_dir).resolve()
            if not (pd / "panel.json").is_file():
                con.err("--panel-dir %s has no panel.json (a panel directory "
                        "is panel.json + arrays/; build one with "
                        "engines/tools/build_token_panel.py)" % pd)
                return EXIT_REFUSED
            try:
                pd.relative_to(SUITE_ROOT)
            except ValueError:
                con.err("--panel-dir must live inside the suite checkout (%s): "
                        "the bundle uploader addresses files by their path "
                        "relative to the suite root, so a panel outside it has "
                        "no remote path. Commit it under engines/panels/ and pass "
                        "that path." % SUITE_ROOT)
                return EXIT_REFUSED
    elif not args.model or not args.panel:
        con.err("--model and --panel are required")
        return EXIT_REFUSED
    if args.cold_runs is None:
        # 2, not 1. The registry's measurement schema requires run_count >= 2
        # for a published row, so a single-run receipt is a number you cannot
        # submit -- and discovering that after paying for the run is the worst
        # possible time. 3 for the sealed lane, matching how K6 was measured.
        args.cold_runs = 2 if args.lane == "streaming" else 3

    if getattr(args, "scope_json", None):
        try:
            _validate_scope_json(con, args.scope_json)
        except Refusal as exc:
            con.say("")
            con.say("REFUSE: %s" % exc.reason)
            for line in exc.advice:
                con.say("        %s" % line)
            return EXIT_REFUSED

    register_secret(os.environ.get("JL_API_KEY"))
    jl = _make_provider(getattr(args, "provider", "jarvislabs"), dry=args.dry_run)
    td = Teardown(jl, con, Path(args.out or ".").resolve())
    td.keep_fs = args.keep_fs
    td.hold_on_failure = bool(getattr(args, "hold_on_failure", False))

    def _signal(signum, _frame):
        con.say("")
        con.step("signal %d -- entering guaranteed teardown (do NOT press ^C again)"
                 % signum)
        td.run("signal %d" % signum)
        sys.exit(EXIT_INTERRUPTED if signum == signal.SIGINT else 1)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal)
        except (ValueError, OSError):
            pass
    atexit.register(lambda: td.run("atexit"))

    con.say("fidelity-cloud %s   job %s" % (VERSION, job_id_for(args)))
    con.rule()
    try:
        plan_data = plan(args, con, jl)
    except Refusal as exc:
        con.say("")
        con.say("REFUSE: %s" % exc.reason)
        for line in exc.advice:
            con.say("        %s" % line)
        return EXIT_REFUSED
    except (EngineUnpinned, JLNotInstalled) as exc:
        con.say("")
        con.say("REFUSE: %s" % exc)
        return EXIT_REFUSED
    except (HFError, JLError) as exc:
        con.err(str(exc))
        return 1

    if plan_data.get("status") == "already-measured":
        con.say("")
        con.rule()
        con.say("ALREADY MEASURED -- the registry rows above answer this "
                "request; nothing was rented, $0.00 spent. Pass --force to "
                "measure anyway (e.g. to reproduce).")
        return EXIT_OK

    if args.dry_run:
        con.say("")
        con.rule()
        outdir = Path(args.out or ("./fidelity-runs/%s" % plan_data["job_id"])).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        write_json(str(outdir / "plan.json"), plan_data)
        blockers = plan_data.get("would_refuse") or []
        con.say("DRY RUN -- nothing created, $0.00 spent.")
        con.say("plan written to %s" % (outdir / "plan.json"))
        if blockers:
            con.say("")
            con.say("%d check(s) would REFUSE a real run:" % len(blockers))
            for b in blockers:
                con.say("  - %s" % b)
            return EXIT_REFUSED
        con.say("all checks passed; a real run would proceed to the confirmation prompt.")
        return EXIT_OK

    if not args.yes:
        con.say("")
        answer = input("Create %d x %s (%s, %s) and spend up to ~$%.2f?  [y/N] "
                       % (plan_data["chosen"]["gpus"], plan_data["chosen"]["gpu_type"],
                          plan_data["chosen"]["region"],
                          "spot" if args.spot else "on-demand",
                          plan_data["cost_estimate"]["band_high_usd"])).strip().lower()
        if answer not in ("y", "yes"):
            con.say("aborted before creating anything. $0.00 spent.")
            return EXIT_REFUSED

    con.rule()
    try:
        cost = execute(args, con, jl, plan_data, td)
        td.completed = True
    except (JLError, HFError, Refusal, RuntimeError) as exc:
        # A stranger should get the sentence that says what broke and whether
        # anything is still billing, not a stack trace ending in our internals.
        # td.run() in the finally has already destroyed whatever existed.
        td.run("failed: %s" % type(exc).__name__)
        con.say("")
        con.err(redact(str(exc)))
        con.say("        the run stopped here; teardown above says what, if "
                "anything, is still billing")
        return EXIT_LEAK if td.leaked else 1
    finally:
        td.run("normal exit")

    con.say("")
    con.say("COST")
    for key in ("estimated_usd", "computed_usd", "billed_usd", "balance_delta_usd"):
        if cost.get(key) is not None:
            con.kv(key.replace("_usd", ""), "$%.2f" % cost[key], indent=2)
    return EXIT_LEAK if td.leaked else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
