#!/usr/bin/env python3
"""measure-cloud -- rent one GPU pod, measure, retrieve, destroy.

    bin/measure-cloud --provider runpod --role root --model <repo> \
        --revision <40-hex> --panel-dir <panel> --dataset-id <id> \
        --publish-root-to <owner/repo> --hf-token-file <file> \
        --measurer <handle> --max-cost <usd> --max-runtime <duration> \
        --out <dir> --dry-run

Paid measurement is RunPod-only and SSH-only: one fresh secure-cloud
on-demand pod, one durable POST intent, one provider POST, verified result
retrieval, unconditional delete.  Four things are always enforced:
`--max-cost` caps the all-in liability; `--max-runtime` is an absolute
deadline in the lease, the on-pod watchdog and the provider; the pod is
destroyed on success, failure, exception and interrupt; and an installed
user-systemd reaper destroys it at the deadline if this process dies.

Strict campaign mode -- a cross-run ledger with explicit limits, foreign-pod
refusal, and an optional sealed controller-loss drill proof -- is opt-in via
`--campaign-ledger`.  `--dry-run` runs every check and creates nothing.

Run `bin/measure-cloud --help` for the full flag list and examples.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
import re
import shlex
import secrets
import signal
import sys
import subprocess
import stat
import tempfile
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The official BF16 release the whole campaign is pinned to.  Every lane binds
# its config/index sha256 into the capture receipt, so this is an identity, not
# a convenience default: resolving `main` instead would let the reference move
# between two measurements of the same artifact.
OFFICIAL_BF16_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
POST_CREATE_CONVERGENCE_SECONDS = 180
POST_CREATE_CONVERGENCE_POLL_SECONDS = 10

from fidelity import census as C                       # noqa: E402
from fidelity.common import (                          # noqa: E402
    Console, human_bytes, human_duration, parse_duration,
    private_directory_chain_error, read_json, redact, register_secret,
    sha256_file, shred_secret_file, utcnow, write_json, write_secret_file,
)
from fidelity.dsformat import resolve_inside                 # noqa: E402
from fidelity.engines import (EngineProfileRefused, EngineUnpinned,
                              build_invocation, load_engines,
                              require_supported_profile, resolve_profile_timing,
                              resolve_root_timing)                    # noqa: E402
from fidelity.hfmeta import (                          # noqa: E402
    HF_ENDPOINT, HFError, RepoMeta, exl3_layout_contract, fetch_file, fetch_json,
    hf_token, load_panel_descriptor, repo_meta, safetensors_header, sniff_surface,
    tr3_tail_declared_bits,
)
from fidelity.jlapi import JL, JLError, JLNotInstalled, select_offer  # noqa: E402
from fidelity.jobcontract import (JobContractError, build_imported_capture_receipt,
                                  finalize_bundle_manifest, finalize_job,
                                  parse_job_bytes, seal_execution_job,
                                  validate_execution_job, verify_job)  # noqa: E402
from fidelity.receipt import produced_by_block                              # noqa: E402
from fidelity.stages import stage_sequence                                  # noqa: E402

VERSION = "0.1.0"
LEASE_DIR = Path.home() / ".fidelity-cloud" / "leases"
GB = C.GB

EXIT_OK = 0
EXIT_FAILED = 1          # the run failed; every pod it created is proven gone
EXIT_REFUSED = 3         # refused before anything was created; $0.00 spent
EXIT_INTERRUPTED = 130
EXIT_LEAK = 90          # teardown could not be confirmed -- the loudest failure
# How long consecutive liveness-probe failures (ssh timeouts, proxy errors)
# are tolerated during a remote stage before the pod is treated as
# unreachable.  Well under the 900-second heartbeat window the on-pod
# watchdog uses, so the controller decides first.
STAGE_PROBE_OUTAGE_SECONDS = 300
# One progress line per stage this often while it runs.
STAGE_PROGRESS_SECONDS = 300


class RunFailed(RuntimeError):
    """A paid run that did not succeed.

    ``liability_may_remain`` decides the exit code: 90 is reserved for the
    case an operator must act on -- a pod this controller created is not
    proven absent.  A run that captured nothing useful but tore down cleanly
    is a failure, not a leak, and saying "leak" for it trained operators to
    ignore the code that matters.
    """

    def __init__(self, message: str, *, liability_may_remain: bool) -> None:
        super().__init__(message)
        self.liability_may_remain = liability_may_remain


# ==========================================================================
# Legacy provider teardown helpers
# ==========================================================================


class Teardown:
    """Historical non-RunPod controller closure, unreachable from paid main.

    Kept for cleanup of pre-v2 leases and offline portability evidence. The
    current RunPod path uses `cloudlease.py` plus provider `terminateAfter`.
    These are the legacy four independent layers, because any one can fail.

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

    The legacy runner required its reaper, a short runtime, or an explicit risk
    override. The current paid route accepts none of those alternatives.
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
        # ROOT-1: a sealed, twice-validated root dataset was destroyed at
        # teardown because nothing preserved it. When this run is a root
        # capture, a VERIFIED-but-unpublished dataset on the box makes
        # teardown HOLD the instance (same bounded shape as
        # --hold-on-failure: lease kept, reaper still expires it) unless
        # --allow-unpublished-root explicitly says to destroy the only copy.
        self.allow_unpublished_root = False
        self.root_publish_expected = False   # this run is --role root
        self.root_verified = False           # the verify stage completed
        self.root_published = False          # the publish_root stage completed
        self.held_for_unpublished_root = False
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
            # getattr, not attribute access: selftest_teardown (and any
            # embedder) builds Teardown via __new__ without __init__, and the
            # guard must degrade to "off" there, never to an AttributeError
            # inside the one code path that must not raise.
            hold_root = bool(getattr(self, "root_publish_expected", False)
                             and getattr(self, "root_verified", False)
                             and not getattr(self, "root_published", False)
                             and not getattr(self, "allow_unpublished_root",
                                             False))
            self.con.say("")
            self.con.step("teardown%s (^C is ignored until this finishes)"
                          % ((" -- " + reason) if reason else ""))
            steps = [self._pull_receipts, self._collect_env, self._shred_secrets]
            self.held_for_unpublished_root = hold_root
            if hold or hold_root:
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
        if self.held and getattr(self, "held_for_unpublished_root", False):
            self.con.say("")
            self.con.say("*" * 78)
            self.con.say("**  HELD: instance %s holds a sealed+VERIFIED root dataset that"
                         % self.machine_id)
            self.con.say("**  was NEVER PUBLISHED. Destroying it would destroy the only")
            self.con.say("**  copy of the evidence (that already happened once: $6.59).")
            self.con.say("**    publish it:  re-run with --publish-root-to <hf-dataset-repo>")
            self.con.say("**                 (the run adopts this box and only the publish")
            self.con.say("**                 stage remains), or pull %s/dataset by hand"
                         % self.fs_root)
            self.con.say("**    destroy   :  re-run with --allow-unpublished-root, or")
            self.con.say("**                 jl destroy %s --yes" % self.machine_id)
            self.con.say("**  The instance is STILL BILLING. Its lease is kept, so the")
            self.con.say("**  reaper still destroys it at the deadline -- publish before then.")
            self.con.say("*" * 78)
        elif self.held:
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
                machine_id: Optional[int], fs_id: Optional[int],
                provider: str = "jarvislabs",
                job_id_full: Optional[str] = None) -> Path:
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    path = LEASE_DIR / ("%s.json" % job_id)
    write_json(str(path), {
        "job_id": job_id,
        # The 256-bit identity adoption compares against (P1-12); the 8-char
        # display id above only names the file and the instance.
        "job_id_full": job_id_full or job_id,
        "name": name,
        # The sweep can only drive ONE backend (the jl CLI); a lease must say
        # whose instance it names, or an expired RunPod lease could aim
        # `jl destroy <id>` at whatever JarvisLabs machine wears that number.
        "provider": provider,
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


#: How far a deadline may sit from "now" and still be believed.  A base36
#: suffix in a HUMAN-chosen or future-schema name can parse to any integer at
#: all -- decades expired, or centuries out -- and an implausible number must
#: never steer a destroy (P1-03).
REAPER_PLAUSIBLE_WINDOW = 90 * 86400

_TERMINAL_STATES = ("destroyed", "terminated", "deleted", "exited")


def _confirm_destroyed(jl, mid, con: Console, *, attempts: int = 5,
                       sleep=time.sleep, base: float = 2.0) -> bool:
    """After a destroy: absent from the listing, or in a terminal state.

    A destroy call returning is NOT the instance being gone -- providers are
    eventually consistent and the call can silently no-op.  Retry with
    backoff; an unreadable listing is UNKNOWN, not confirmation either way.
    """
    for attempt in range(attempts):
        if attempt:
            sleep(min(60.0, base * (2 ** (attempt - 1))))
        try:
            listing = jl.list_instances()
        except JLError as exc:
            con.warn("reaper: cannot list to confirm %s is gone (attempt "
                     "%d/%d): %s" % (mid, attempt + 1, attempts,
                                     redact(str(exc))[:120]))
            continue
        inst = next((i for i in listing
                     if str(i.machine_id) == str(mid)), None)
        if inst is None:
            return True
        status = str(getattr(inst, "status", "")).strip().lower()
        if status in _TERMINAL_STATES:
            return True
        con.warn("reaper: %s is still '%s' after destroy (attempt %d/%d)"
                 % (mid, status or "?", attempt + 1, attempts))
    return False


def reaper_sweep(con: Console, *, dry: bool = False, jl=None,
                 sleep=time.sleep, confirm_attempts: int = 5) -> int:
    """Destroy what an EXPIRED LEASE authorizes; only report everything else.

    Authorization model (P1-03): a destroy target must come from a provider
    instance ID recorded in a lease THIS TOOL wrote.  Instance names and the
    deadlines parsed out of them are discovery only -- they surface a
    candidate for the operator, they never authorize destruction, because a
    name is guessable, reusable, and parseable into nonsense.  The two
    failure modes this shape prevents are opposite and both severe: billing
    that continues behind a false-success cleanup, and destroying a machine
    this tool did not create (AGENTS.md: never destroy a machine you did not
    create).

    Every destroy is confirmed against provider state with retry/backoff; an
    unconfirmed destroy keeps the lease and makes the sweep exit EXIT_LEAK,
    because "I told the API to delete it" is a claim, not a receipt.

    --dry-run enumerates EXACTLY what the real run would mutate -- destroys
    AND lease retirements -- and mutates nothing.
    """
    if jl is None:
        jl = JL(dry=dry)
        try:
            jl.require()
        except (JLNotInstalled, JLError) as exc:
            con.err(str(exc))
            return 1
    now = time.time()

    leases = []
    for path in sorted(LEASE_DIR.glob("*.json")) if LEASE_DIR.is_dir() else []:
        try:
            leases.append((path, read_json(str(path))))
        except (OSError, ValueError):
            continue

    _drive_memo: Dict[str, bool] = {}

    def can_drive(path, lease) -> bool:
        """This sweep drives the jl CLI only.  A lease naming another
        provider's instance is INVISIBLE to it and must be left entirely
        alone: not destroyed (`jl destroy` would aim at whatever JarvisLabs
        machine wears that number), and not retired as a phantom (the
        instance is alive on a cloud jl cannot list)."""
        if path.name in _drive_memo:
            return _drive_memo[path.name]
        verdict = True
        provider = str(lease.get("provider") or "").strip().lower()
        mid = lease.get("machine_id")
        if provider and provider != "jarvislabs":
            con.warn("reaper: lease %s names a %s instance; this sweep "
                     "drives only the jl CLI -- leaving it alone"
                     % (path.name, provider))
            verdict = False
        elif mid is not None and not str(mid).isdigit():
            con.warn("reaper: lease %s has a non-numeric machine id %r, "
                     "which cannot be a JarvisLabs machine -- leaving it "
                     "alone" % (path.name, mid))
            verdict = False
        _drive_memo[path.name] = verdict
        return verdict

    # -- authorized targets: expired leases with a provider instance id ----
    targets: Dict[str, Any] = {}   # str(mid) -> (mid, reason)
    for path, lease in leases:
        mid = lease.get("machine_id")
        if not mid:
            continue
        if not can_drive(path, lease):
            continue
        deadline = float(lease.get("deadline_epoch", 0) or 0)
        if deadline <= 0 or deadline > now + REAPER_PLAUSIBLE_WINDOW:
            con.warn("reaper: lease %s carries an implausible deadline %r -- "
                     "skipping it (a nonsense deadline is not authorization)"
                     % (path.name, lease.get("deadline_epoch")))
            continue
        if deadline < now:
            targets[str(mid)] = (mid, "lease %s expired" % path.name)

    # -- one listing serves discovery, phantom retirement and reporting ----
    try:
        instances = jl.list_instances()
    except JLError as exc:
        con.warn("could not list instances: %s" % redact(str(exc)))
        instances = None

    if instances is not None:
        lease_mids = {str(lease.get("machine_id"))
                      for _, lease in leases if lease.get("machine_id")}
        for inst in instances:
            name = inst.name or ""
            if not name.startswith("fidcloud-") or str(inst.machine_id) in lease_mids:
                continue
            deadline = parse_deadline_name(name)
            if deadline is None:
                continue
            if abs(deadline - now) > REAPER_PLAUSIBLE_WINDOW:
                con.warn("reaper: instance %s (%s) has an implausible name "
                         "deadline %d -- ignored" % (inst.machine_id, name,
                                                     deadline))
                continue
            if deadline < now:
                # DISCOVERED, never destroyed: no lease of this tool names it.
                con.warn(
                    "reaper: instance %s (%s) LOOKS expired by its name but "
                    "no lease of this tool authorizes destroying it. Verify "
                    "and destroy it yourself: jl destroy %s"
                    % (inst.machine_id, name, inst.machine_id))

    # -- phantom leases: instance gone, lease still on disk ----------------
    retire = []
    if instances is not None:
        alive = {str(i.machine_id) for i in instances}
        for path, lease in leases:
            mid = lease.get("machine_id")
            if (mid and str(mid) not in alive and str(mid) not in targets
                    and can_drive(path, lease)):
                retire.append((path, mid, "machine %s is gone" % mid))

    if not targets and not retire:
        con.say("reaper: nothing expired")
        return EXIT_OK

    failures = []
    for key in sorted(targets):
        mid, why = targets[key]
        lease_paths = [path for path, lease in leases
                       if str(lease.get("machine_id")) == str(mid)]
        if dry:
            con.say("reaper: WOULD destroy %s (%s), confirm it terminal, and "
                    "then retire %s" % (mid, why,
                                        ", ".join(p.name for p in lease_paths)))
            continue
        con.say("reaper: destroying %s (%s)" % (mid, why))
        try:
            jl.destroy(mid)
        except JLError as exc:
            con.err("reaper could not destroy %s: %s -- the lease is KEPT and "
                    "this sweep exits non-zero" % (mid, redact(str(exc))))
            failures.append(mid)
            continue
        if not _confirm_destroyed(jl, mid, con, attempts=confirm_attempts,
                                  sleep=sleep):
            con.err("reaper: destroy of %s was NOT confirmed terminal -- it "
                    "may still be billing. The lease is KEPT so the next "
                    "sweep retries, and this sweep exits non-zero." % mid)
            failures.append(mid)
            continue
        con.say("reaper: %s confirmed gone" % mid)
        for path in lease_paths:
            path.unlink(missing_ok=True)

    for path, mid, why in retire:
        if dry:
            con.say("reaper: WOULD retire lease %s (%s)" % (path.name, why))
            continue
        con.say("reaper: retiring lease %s (%s)" % (path.name, why))
        path.unlink(missing_ok=True)

    return EXIT_LEAK if failures else EXIT_OK


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


def gate_verified(plan: Dict[str, Any], gate: str, **detail) -> None:
    """Record a mandatory gate's positive verdict, distinctly (P1-11)."""
    plan.setdefault("gates", {})[gate] = dict({"status": "verified"}, **detail)


def gate_failed(plan: Dict[str, Any], gate: str, reason: str) -> None:
    plan.setdefault("gates", {})[gate] = {"status": "failed", "reason": reason}


def gate_not_checked(con, plan: Dict[str, Any], args, gate: str,
                     reason: str) -> None:
    """A MANDATORY gate that could not run is a verdict of its own (P1-11).

    The old shape -- warn and continue -- let an import failure or a network
    blip wear a passing gate's clothes: the planner then said "all checks
    passed" about an artifact whose seal nobody recomputed, and the user
    rented hardware the gate existed to protect.  `not_checked` is neither
    `verified` nor `failed`:

      * a REAL run refuses -- renting on an unchecked mandatory gate is the
        exact spend the gate exists to prevent;
      * a --dry-run continues, but the plan is downgraded to ESTIMATE-ONLY,
        the summary names every unchecked gate, and the "all checks passed"
        line is never printed.
    """
    plan.setdefault("gates", {})[gate] = {"status": "not_checked",
                                          "reason": reason}
    plan.setdefault("gates_not_checked", []).append(gate)
    plan["estimate_only"] = True
    if getattr(args, "dry_run", False):
        con.warn("gate '%s' NOT CHECKED: %s" % (gate, reason))
        con.warn("  the plan is now ESTIMATE-ONLY: this is a verdict about "
                 "the check, not about the artifact")
    else:
        raise Refusal(
            "mandatory gate '%s' could not be checked: %s" % (gate, reason),
            ["A gate that cannot run is not a gate that passed (P1-11): "
             "renting on an unchecked gate is the spend it exists to prevent.",
             "Fix the cause (network, or the engines/tools import) and re-run;",
             "--dry-run produces a visibly estimate-only plan meanwhile.",
             "Nothing was created. $0.00 spent."])


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



# Capacity failures, as each provider actually spells them. Matched against the
# exception TEXT because that is all a CLI-or-HTTP backend surfaces; a pattern
# here must be specific enough not to swallow a real error (an auth failure or
# a malformed request retried eight times is eight times as confusing).
_CAPACITY_PATTERNS = (
    "SUPPLY_CONSTRAINT",                  # runpod graphql
    "no longer any instances available",  # runpod prose
    "insufficient-capacity",              # lambda
    "insufficient capacity",
    "no capacity",
    "out of stock",
    "no rentable",                        # vast search came back empty
)


def _create_with_retry(jl, con: Console, *, attempts: int = 6,
                       base_wait: float = 30.0, max_wait: float = 600.0, **kw):
    """Create an instance, riding out capacity gaps with exponential backoff.

    Capacity is a property of the MINUTE, not of the plan: the provider survey
    measured GH200 present in 12% of polls and healthy in 3 of 8 launches, two
    H100 types under 25%, and RunPod answering SUPPLY_CONSTRAINT for pairings
    its own catalogue advertised. Michel's rule, and it generalises: every
    launch is a retry loop, on every provider -- with exponential backoff plus
    jitter, because a fleet of controllers hammering a capacity-starved API at
    a fixed cadence is how a shortage becomes an outage. Backoff is manners,
    not just resilience.

    Only capacity-shaped failures retry. Anything else -- auth, a bad GPU id,
    a malformed request -- raises immediately, because retrying a mistake six
    times just delays reading the error.
    """
    import random

    last = None
    for attempt in range(1, attempts + 1):
        try:
            return jl.create(**kw)
        except Exception as exc:                          # noqa: BLE001
            text = str(exc)
            if not any(pat.lower() in text.lower() for pat in _CAPACITY_PATTERNS):
                raise
            last = exc
            if attempt == attempts:
                break
            wait = min(max_wait, base_wait * (2 ** (attempt - 1)))
            wait *= random.uniform(0.8, 1.2)
            con.warn("no capacity (attempt %d/%d): %s -- retrying in %.0fs"
                     % (attempt, attempts, redact(text)[:120], wait))
            time.sleep(wait)
    raise Refusal(
        "no capacity after %d attempts with exponential backoff" % attempts,
        ["last answer: %s" % redact(str(last))[:200],
         "The provider is out of this GPU right now; the plan is unchanged.",
         "Re-run to keep trying, pick another GPU, or another provider.",
         "Nothing was created. $0.00 spent."])


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
    return job_identity(args)["job_id"]


@contextlib.contextmanager
def _anonymous_hf_environment():
    """Isolate every paid-plan model/panel read from controller credentials."""
    official = "https://huggingface.co"
    ambient_endpoint = os.environ.get("HF_ENDPOINT")
    if ambient_endpoint not in (None, "", official):
        raise Refusal(
            "safe RunPod metadata requires exact https://huggingface.co",
            ["ambient HF_ENDPOINT: %s" % ambient_endpoint])
    names = (
        "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
        "HF_TOKEN_PATH", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE", "HUGGINGFACE_CO_STAGING",
        "HUGGINGFACE_CO_URL_TEMPLATE", "HF_INFERENCE_ENDPOINT",
        "HF_ENDPOINT", "HF_HOME", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE",
        "HUGGINGFACE_ASSETS_CACHE", "HF_XET_CACHE", "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE", "XDG_CACHE_HOME",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN")
    saved = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="fidelity-anonymous-hf-") as home:
        try:
            for name in names:
                os.environ.pop(name, None)
            os.environ["HF_ENDPOINT"] = official
            os.environ["HF_HOME"] = home
            os.environ["HF_TOKEN_PATH"] = str(Path(home) / "no-token")
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            yield {
                "schema": "fidelity-suite/anonymous-hf-access.v1",
                "endpoint": official,
                "isolated_hf_home": True,
                "implicit_token_disabled": True,
                "cleared_token_sources": sorted(names),
            }
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _source_checkout_proof(*, include_untracked: bool) -> Dict[str, Any]:
    """Return a no-write proof of the exact clean suite checkout."""
    head_result = subprocess.run(
        ["git", "-C", str(SUITE_ROOT), "rev-parse", "--verify", "HEAD"],
        capture_output=True, text=True, timeout=30, check=False)
    head = (head_result.stdout or "").strip()
    if (head_result.returncode != 0
            or re.fullmatch(r"[0-9a-f]{40}", head) is None):
        raise Refusal("suite source is not an exact 40-hex Git HEAD", [])
    mode = "all" if include_untracked else "no"
    status_result = subprocess.run(
        ["git", "-C", str(SUITE_ROOT), "status", "--porcelain=v1",
         "--untracked-files=%s" % mode],
        capture_output=True, text=True, timeout=30, check=False)
    if status_result.returncode != 0:
        raise Refusal(
            "suite Git cleanliness check failed",
            [redact((status_result.stderr or "")[-500:])])
    status = (status_result.stdout or "").encode("utf-8")
    if status:
        raise Refusal(
            "safe RunPod execution requires a clean source checkout",
            ["git status mode: %s" % mode])
    return {
        "schema": "fidelity-suite/source-checkout-proof.v1",
        "head": head,
        "status_mode": mode,
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _probe_root_stage_clis() -> Dict[str, Any]:
    """Execute every root-stage CLI help and bind all flags we will emit."""
    probes = (
        ("capture-wrapper", ("bin/fidelity_dataset.py", "capture", "--help"),
         ("--out", "--form", "--role", "--lane", "--engine")),
        ("capture-engine", ("engines/tools/hf_capture.py", "--help"), (
            "--model", "--model-revision", "--panel", "--panel-id",
            "--panel-binding", "--panel-binding-sha256",
            "--panel-tokenizer-root", "--weights-repository", "--repository",
            "--schedule", "--device", "--dtype", "--dataset-id",
            "--dataset-name", "--run-name", "--cold-run", "--author",
            "--sanity-expect", "--unexpected-tensors-allowlist",
            "--unexpected-tensors-allowlist-sha256",
            "--unexpected-tensors-name-sha256")),
        ("verify", ("bin/fidelity_dataset.py", "verify", "--help"),
         ("--json",)),
        ("describe", ("bin/fidelity_dataset.py", "describe", "--help"), ()),
        ("compare", ("bin/fidelity_dataset.py", "compare", "--help"), (
            "--reference", "--candidate", "--reference-label",
            "--candidate-label", "--self-compare", "--force-compute",
            "--device", "--replay-device", "--replay-dtype",
            "--vocab-chunk", "--out")),
        ("qualify-root",
         ("bin/fidelity_dataset.py", "qualify-root", "--help"), (
             "--job", "--first", "--repeat", "--first-label",
             "--repeat-label", "--first-verify", "--repeat-verify",
             "--comparison", "--out")),
        ("publish", ("bin/fidelity_dataset.py", "publish", "--help"), (
            "--repo", "--qualification", "--job", "--result-archive",
            "--expected-archive-sha256", "--expected-archive-bytes",
            "--expected-head", "--token-file", "--receipt",
            "--revision-message")),
    )
    rows = []
    for name, argv, required_flags in probes:
        result = subprocess.run(
            [sys.executable] + [str(SUITE_ROOT / argv[0])] + list(argv[1:]),
            cwd=str(SUITE_ROOT), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30, check=False)
        output = (result.stdout or "") + (result.stderr or "")
        help_tokens = set(re.findall(
            r"(?<![\w-])(--[a-z0-9][a-z0-9-]+)", output.lower()))
        missing = [
            flag for flag in required_flags if flag.lower() not in help_tokens]
        if result.returncode != 0 or missing:
            raise Refusal(
                "root stage CLI help probe failed for %s" % name,
                ["missing flags: %s" % ",".join(missing),
                 "return code: %d" % result.returncode])
        rows.append({
            "name": name, "help_ok": True,
            "help_sha256": hashlib.sha256(
                output.encode("utf-8")).hexdigest(),
            "required_flags": list(required_flags),
        })
    proof = {
        "schema": "fidelity-suite/root-cli-probe.v1",
        "interpreter": "%d.%d" % (
            sys.version_info.major, sys.version_info.minor),
        "probes": rows,
        "probe_sha256": "",
    }
    proof["probe_sha256"] = hashlib.sha256(
        _canonical_bytes(proof)).hexdigest()
    return proof


def _suite_head() -> str:
    """The code state that will produce the receipts, as one string.

    A git checkout answers with HEAD; a tarball answers with a digest of the
    two files that drive a cloud run.  Part of the job identity (P1-12): the
    same command re-run after the suite moved is a DIFFERENT producer, and
    letting it adopt the old machine relabels old outputs with new code.
    """
    try:
        proc = subprocess.run(["git", "-C", str(SUITE_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        head = (proc.stdout or "").strip()
        if len(head) == 40:
            return head
    except Exception:                                    # noqa: BLE001
        pass
    digest = hashlib.sha256()
    for rel in ("bin/measure_cloud.py", "bin/stage_measure.sh"):
        try:
            digest.update((SUITE_ROOT / rel).read_bytes())
        except OSError:
            digest.update(("missing:%s" % rel).encode())
    return "files-%s" % digest.hexdigest()


def job_identity(args: argparse.Namespace, *,
                 resolved_revision: "Optional[str]" = None,
                 panel_revision: "Optional[str]" = None) -> Dict[str, str]:
    """The wide job identity: resolve first, then hash at 256 bits (P1-12).

    The old identity was sha1(requested args)[:8]: it hashed `--revision main`
    BEFORE resolution, ignored the suite's own code state, and 32 bits of it
    named the instance -- so a rerun after upstream `main` (or this repo)
    moved would adopt the old machine, overwrite job.json with the newly
    resolved target, and skip stages whose bytes the OLD identity produced.
    A sealed-looking receipt could then name a producer state that did not
    create its evidence.

    So: `resolved_revision` (the 40-hex the planner pinned) and the panel's
    resolved revision go in once known, the suite HEAD always goes in, and
    the FULL 64-hex id is what leases, job.json and stage markers store and
    compare.  The 8-char prefix is DISPLAY (and the instance-name length
    budget); it is never the basis of a comparison on its own.
    """
    key = json.dumps({
        "model": args.model,
        "revision": resolved_revision or args.revision,
        "revision_resolved": bool(resolved_revision),
        "panel": args.panel,
        "panel_revision": panel_revision,
        "lane": args.lane, "spot": args.spot, "cold_runs": args.cold_runs,
        "provider": getattr(args, "provider", "jarvislabs"),
        "gpu": getattr(args, "gpu", None),
        "role": getattr(args, "role", "quant"),
        "suite_head": _suite_head(),
    }, sort_keys=True)
    full = hashlib.sha256(key.encode()).hexdigest()
    return {"job_id_full": full, "job_id": full[:8]}




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


CANDIDATE_DECODE_METHOD = "fp8-block-dequant-to-bf16"
#: `engines/tools/layer_outer.TRELLIS_DECODE_METHOD`, kept in step with it.
CANDIDATE_DECODE_METHOD_TRELLIS = "exl3-trellis-decode-to-bf16"
#: `engines/tools/layer_outer.TRELLIS_TP_COMPOSE_METHOD`: the same decode with
#: the artifact's tensor-parallel rank shards composed into whole weights.
CANDIDATE_DECODE_METHOD_TRELLIS_TP = "exl3-trellis-tp-compose-to-bf16"
#: `engines/tools/layer_outer.NVFP4_DECODE_METHOD`, kept in step with it.
CANDIDATE_DECODE_METHOD_NVFP4 = "nvfp4-modelopt-dequant-to-bf16"
#: `engines/tools/nvfp4_surface.MODELOPT_ACTIVATION_SCHEME`: the static
#: per-tensor `input_scale` a W4A4 kernel applies to activations is never
#: applied to weights by a weights-only decode, and the contract says so.
CANDIDATE_NVFP4_ACTIVATION_SCHEME = "static-nvfp4-not-applied"
CANDIDATE_NVFP4_GROUP_SIZE = 16
#: `nvfp4_surface.MO_ONLINE_TRANSFORM_KEYS`: a true value declares an online
#: rotation the decode-and-run measurement would not apply; refused by name.
CANDIDATE_NVFP4_ONLINE_TRANSFORMS = ("rotate", "learned_rotation", "quarot_r1_fold",
                                     "expert_block_reorder")


def _nvfp4_ignore_sha256(ignore) -> str:
    """Byte-identical to `nvfp4_surface.modelopt_ignore_sha256` (stdlib only)."""
    names = sorted(str(item) for item in (ignore or []))
    canonical = (json.dumps(names, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def _nvfp4_candidate_decode_plan(qc: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of `nvfp4_surface.modelopt_nvfp4_plan`'s CONTRACT block, field for field.

    The pod additionally censuses the index (57,600 routed modules in the
    modelopt {weight, weight_scale, weight_scale_2} layout, the official
    non-routed name set) and refuses there; the controller cannot see the
    index from the config alone, so that refusal stays where the index is
    readable. Everything in the contract derives from config.json text.
    """
    algo = qc.get("quant_algo")
    if algo != "NVFP4":
        raise Refusal(
            "candidate quantization_config quant_method='modelopt' quant_algo=%r is not "
            "the NVFP4 form the layer-outer loader decodes" % (algo,),
            ["Decodable today: modelopt NVFP4 (e2m1 group-16 routed experts, "
             "engines/tools/nvfp4_surface.py). Another modelopt form needs its decoder "
             "authored and proven bitwise first."])
    transforms = [key for key in CANDIDATE_NVFP4_ONLINE_TRANSFORMS if qc.get(key) is True]
    if transforms:
        raise Refusal(
            "candidate quantization_config declares online weight transforms %s"
            % (transforms,),
            ["A rotation folded into the activations at serving time is not applied by "
             "a decode-and-run measurement; the number would describe a model nobody "
             "serves."])
    groups = qc.get("config_groups")
    if isinstance(groups, dict):
        if sorted(groups) != ["group_0"]:
            raise Refusal("unexpected modelopt config groups %s" % sorted(groups), [])
        weights = (groups.get("group_0") or {}).get("weights") or {}
        if (weights.get("num_bits") != 4 or weights.get("group_size") != CANDIDATE_NVFP4_GROUP_SIZE
                or weights.get("dynamic") not in (False, None) or weights.get("type") != "float"):
            raise Refusal(
                "candidate config_groups.group_0.weights is not static float 4-bit group-16 "
                "NVFP4: %r" % (dict(weights),), [])
        declared_by = "config_groups.group_0.weights"
    elif groups is None:
        group_size = qc.get("group_size")
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise Refusal(
                "candidate modelopt quantization_config declares neither config_groups nor "
                "an integer top-level group_size; the weight format is undeclared", [])
        if group_size != CANDIDATE_NVFP4_GROUP_SIZE:
            raise Refusal("candidate modelopt group_size %d is not the NVFP4 group size 16"
                          % group_size, [])
        num_bits = qc.get("num_bits", 4)
        if isinstance(num_bits, bool) or num_bits != 4:
            raise Refusal("candidate modelopt num_bits %r is not 4 (NVFP4)" % (num_bits,), [])
        declared_by = "quantization_config.group_size"
    else:
        raise Refusal("candidate quantization_config.config_groups is not a mapping", [])
    producer = qc.get("producer")
    producer = ({"name": str(producer.get("name")), "version": str(producer.get("version"))}
                if isinstance(producer, dict) else None)
    ignore = qc.get("ignore") or []
    return {
        "method": CANDIDATE_DECODE_METHOD_NVFP4,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "num_bits": 4,
            "group_size": CANDIDATE_NVFP4_GROUP_SIZE,
            "weights_declared_by": declared_by,
            "activation_scheme": CANDIDATE_NVFP4_ACTIVATION_SCHEME,
            "producer": producer,
            "ignore_count": len(ignore),
            "ignore_sha256": _nvfp4_ignore_sha256(ignore),
        },
        "_declaration": {"declared_by": declared_by, "transforms_declared": transforms,
                         "ignore_count": len(ignore)},
    }

#: `engines/tools/gguf_surface.GGUF_DECODE_METHOD`, kept in step with it.
CANDIDATE_DECODE_METHOD_GGUF = "gguf-dequant-to-bf16"


def _gguf_surface_module():
    """`engines/tools/gguf_surface` is stdlib at import (numpy/torch lazy), so the
    controller can read a build's headers with the SAME parser the pod uses."""
    tools = SUITE_ROOT / "engines" / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import gguf_surface  # noqa: E402

    return gguf_surface


def _gguf_candidate_plan(con: Console, target: RepoMeta, surface, plan: Dict[str, Any],
                         official_config: Dict[str, Any]) -> Dict[str, Any]:
    """The GGUF candidate's decode contract + identity, from header bytes at $0.

    Mirrors `engines/tools/layer_outer.gguf_checkpoint_plan` field for field:
    both call `gguf_surface.decode_contract` over the same tensor tables (the
    pod on local files, this on https range requests), so `qualify_root`'s
    equality check is a check of the same function on the same bytes. The
    census that decides decodability -- architecture in the arch table,
    geometry gate, every tensor nameable and of a kernel-backed type, the
    indexer copies recognised from the OFFICIAL config's indexer_types -- runs
    here too, so an undecodable build is refused before a rental, not after
    a 467 GB fetch.
    """
    ggs = _gguf_surface_module()
    if not surface.path or not surface.artifact_files:
        raise Refusal("a GGUF candidate needs --path <build>; the repo is a shelf of builds",
                      ["builds: %s" % ", ".join(sorted(surface.evidence.get("gguf_builds") or {}))])
    urls = ["%s/%s/resolve/%s/%s" % (HF_ENDPOINT, target.repo_id, target.revision, name)
            for name, _ in surface.artifact_files]
    try:
        container = ggs.GgufContainer([ggs.GgufFile(u) for u in urls])
        arch = ggs.arch_for(container.architecture)
        full = ggs.indexer_full_layers_from_config(official_config, arch)
        loaded = ggs.load_gguf_surface(urls, repo=target.repo_id, revision=target.revision,
                                       require_file_hashes=False, indexer_full_layers=full)
    except ValueError as exc:
        raise Refusal(
            "this GGUF build cannot be decoded by the layer-outer gguf lane: %s" % redact(str(exc)),
            ["Read from the build's OWN headers at the pinned revision, not from its name.",
             "Adding a type or an architecture means adding it to engines/tools/gguf_surface.py "
             "WITH the bitwise proof (selftest_gguf_offline.py), not skipping tensors.",
             "Nothing was created. $0.00 spent."])
    if official_config.get("num_hidden_layers") != arch.mtp_layer:
        raise Refusal(
            "the reference release's config declares %r decoder layers but the %s GGUF "
            "carries %d before its MTP block" % (official_config.get("num_hidden_layers"),
                                                  arch.key, arch.mtp_layer), [])
    build = surface.path.rstrip("/").rpartition("/")[2]
    decode = ggs.decode_contract(loaded.container, build)
    census = decode["quantization_config"]["type_census"]
    codec = "gguf-i-quant" if any(t.startswith("IQ") for t in census) else "gguf-k-quant"
    plan.setdefault("target", {})["gguf_verification"] = {
        "verified": True,
        "architecture": arch.key,
        "family": arch.family,
        "build": build,
        "tensor_count": len(loaded.container.tensors),
        "type_census": census,
        "tensor_table_sha256": decode["quantization_config"]["tensor_table_sha256"],
        "shared_indexer_copies_not_loaded": len(loaded.census.shared_indexer_copies),
        "mla_layers": len(loaded.census.mla_layers),
        "checkpoint_identity_sha256": loaded.checkpoint_identity_sha256(),
        "codec_from_census": codec,
        "read_from": "https range requests over the build's own headers",
    }
    plan["_gguf_loaded"] = loaded
    con.ok("candidate is a llama.cpp GGUF build",
           "%s/%s: arch %s, %d tensors, types %s; every tensor decoded to bf16 on the "
           "capture device (%s), kv_b composed, experts sliced; codec by census %s"
           % (target.repo_id, build, arch.key, len(loaded.container.tensors),
              ", ".join("%s x%d" % (t, census[t]) for t in sorted(census)),
              decode["method"], codec))
    return decode


def _gguf_model_file_identity(target: RepoMeta, surface, loaded, official_config_raw: bytes,
                              official_ref: Tuple[str, str]) -> Dict[str, Any]:
    """`_model_file_identity` for a GGUF build: no config.json, no index.

    The config is the OFFICIAL release's (the reference root's weights repo,
    fetched anonymously; the candidate stage copies it beside the build so the
    HF model class can be built), and the tensor table digest stands where the
    safetensors index digest stands. The shards are the build's own .gguf
    files; the download manifest is exactly those plus the repo's LICENSE.
    """
    ggs = _gguf_surface_module()
    if re.fullmatch(r"[0-9a-f]{40}", target.revision) is None:
        raise Refusal("RunPod target revision is not an exact 40-hex pin", [])
    sizes = dict(target.files)
    shards = []
    for name, size in sorted(surface.artifact_files):
        pure = PurePosixPath(name)
        if (("\\" in name) or pure.is_absolute() or pure.as_posix() != name
                or any(part in ("", ".", "..") for part in pure.parts)):
            raise Refusal("GGUF build contains an unsafe file path", [])
        if sizes.get(name) != size or not isinstance(size, int) or size <= 0:
            raise Refusal("GGUF build file %s is missing or size-unknown in the repo listing" % name, [])
        shards.append({"path": name, "bytes": size})
    table_json = ggs._canonical_json([
        {"name": n, "dims": [int(d) for d in r["dims"]], "type": r["type"],
         "offset": int(r["offset"]), "file": r["file"]}
        for n, r in sorted(loaded.container.tensors.items())])
    manifest = list(shards)
    if "LICENSE" in sizes:
        manifest.append({"path": "LICENSE", "bytes": sizes["LICENSE"]})
    manifest.sort(key=lambda row: row["path"])
    config = json.loads(official_config_raw.decode("utf-8"))
    vocab_size = loaded.container.geometry_value("vocab_size")
    hidden_size = loaded.container.geometry_value("embedding_length")
    if (int(vocab_size) != config.get("vocab_size")
            or int(hidden_size) != config.get("hidden_size")):
        raise Refusal(
            "the GGUF's geometry (vocab %s, hidden %s) differs from the reference release's "
            "config (%s, %s)" % (vocab_size, hidden_size, config.get("vocab_size"),
                                 config.get("hidden_size")), [])
    return {
        "config_sha256": hashlib.sha256(official_config_raw).hexdigest(),
        "config_bytes": len(official_config_raw),
        "config_source": "%s@%s config.json (the reference root's release; the GGUF ships "
                         "none and the candidate stage copies this one beside the build)"
                         % official_ref,
        "index_sha256": hashlib.sha256(table_json).hexdigest(),
        "index_bytes": len(table_json),
        "index_source": "sha256 of the canonical JSON GGUF tensor table (name, dims, type, "
                        "offset, file); a GGUF ships no safetensors index",
        "model_bytes": sum(row["bytes"] for row in shards),
        "shards": shards,
        "download_bytes_total": sum(row["bytes"] for row in manifest),
        "download_manifest": manifest,
        "download_manifest_sha256": hashlib.sha256(_canonical_bytes(manifest)).hexdigest(),
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "shard_manifest_sha256": hashlib.sha256(_canonical_bytes(shards)).hexdigest(),
    }


def _exl3_layout_block(index_keys, qc, tail) -> Dict[str, Any]:
    """The rotation-layout keys of an exl3 candidate's contract, from the index.

    `hfmeta.exl3_layout_contract` is byte-identical to the pod's copy in
    layer_outer: the layout is READ FROM THE INDEX NAMES (stock per-module
    vectors, willfalco/jpsequeira's `experts.shared_h`, brandonmusic's
    `experts.r7_shared`), cross-checked against the declaration, and refused
    -- at $0, before a rental -- when a module's vectors cannot be resolved
    by name, exactly as the pod would refuse it after the fetch.
    """
    if index_keys is None:
        raise Refusal(
            "an exl3 candidate's rotation layout is read from model.safetensors.index.json, "
            "and the index was not available", [])
    try:
        layout, detail = exl3_layout_contract(list(index_keys), qc, tail)
    except ValueError as exc:
        raise Refusal(
            "exl3 candidate: %s" % exc,
            ["The pod's trellis decoder (layer_outer.trellis_checkpoint_plan) applies "
             "the same rule and would refuse after the fetch; nothing was created."])
    layout["_layout_detail"] = {
        "modules_per_layout": detail["census"]["per_layout"],
        "nonrouted_bits": detail["nonrouted_bits"],
        "r7_declaration": detail["r7_declaration"],
    }
    return layout


def _candidate_decode_plan(qc, cfg=None, index_keys=None,
                          sidecar_loader=None) -> Dict[str, Any]:
    """The decode the streaming loader will apply, from the config and the index names.

    Mirrors `engines/tools/layer_outer.fp8_checkpoint_plan` /
    `trellis_checkpoint_plan` field for field (the pod compares its runtime
    receipt against this block), without importing torch on the controller.
    `index_keys` (the index's weight_map names) is REQUIRED for an exl3
    candidate: its rotation layout, shared-vector and non-routed-module
    digests are contract, read from the names on both sides.
    """
    tail = (cfg or {}).get("hybrid_tr3_tail") if isinstance(cfg, dict) else None
    tail = tail if isinstance(tail, dict) and tail.get("format") == "exl3-trellis" else None
    if (not isinstance(qc, dict) or not qc) and tail is None:
        raise Refusal(
            "--candidate-scope, but this checkpoint publishes no quantization_config: "
            "it is an unquantized release; capture it as a root", [])
    qc = qc if isinstance(qc, dict) else {}
    method, fmt, block = qc.get("quant_method"), qc.get("fmt"), qc.get("weight_block_size")
    activation = qc.get("activation_scheme")
    if tail is not None:
        # davidsyoung's TR3 releases: the exl3 declaration is the top-level
        # hybrid_tr3_tail block; the quantization_config is a leftover
        # ModelOpt/NVFP4 block that describes nothing in the checkpoint. The
        # pod refuses unless the payload keys agree (rank shards, tp ranks
        # each). Mirrors layer_outer.trellis_checkpoint_plan's tail branch.
        codebook = tail.get("codebook")
        if codebook is not None and str(codebook) not in ("mul1", "mcg"):
            raise Refusal(
                "hybrid_tr3_tail codebook=%r is not one this decoder speaks (mul1/mcg)"
                % (codebook,), [])
        tp = tail.get("tp")
        composed = isinstance(tp, int) and not isinstance(tp, bool) and tp >= 2
        layout = _exl3_layout_block(index_keys, qc, tail)
        # The weights_decode block has an EXACT key set (jobcontract);
        # what the artifact declared by is console/plan evidence, not
        # contract, so it rides beside the block under a private key the
        # caller pops before binding.
        # jpsequeira's GLM-5.2 TR3 declares `bits: "mixed"` with no numeric
        # and a `bits_per_expert: "<file>:<key>"` sidecar.  The controller
        # resolves it here through the caller's sidecar_loader (the same
        # anonymous fetch_file the pod uses against the checkpoint dir), so
        # the contract's `bits` is the float mean and `declared_bits_source`
        # is byte-identical to the pod's block (same sha256).
        declared = tr3_tail_declared_bits(tail, sidecar_loader=sidecar_loader)
        if isinstance(declared, tuple):
            bits_value, declared_bits_source = declared
        else:
            bits_value, declared_bits_source = declared, None
        contract = {
            "quant_method": "exl3",
            "codebook": str(codebook) if codebook is not None else None,
            "bits": bits_value,
            "head_bits": None,
            "modules_to_not_convert": [],
        }
        if declared_bits_source is not None:
            contract["declared_bits_source"] = declared_bits_source
        contract.update({k: v for k, v in layout.items() if not k.startswith("_")})
        return {
            "method": (CANDIDATE_DECODE_METHOD_TRELLIS_TP if composed
                       else CANDIDATE_DECODE_METHOD_TRELLIS),
            "quantization_config": contract,
            "_declaration": {"declared_by": "hybrid_tr3_tail",
                             "quant_method_declared": method,
                             "tp": tp if composed else None,
                             "layout": layout["_layout_detail"]},
        }
    if method == "exl3":
        # The trellis surface `engines/tools/layer_outer.materialize_trellis_subset`
        # decodes: stock-exllamav3 payload groups, per-module codebook, bits
        # from the payload's own trellis shape. The rank-split TR3 layout is
        # refused on the pod by name (its composition is unpublished); the
        # controller cannot see payload SHAPES from the index alone, so that
        # refusal stays where the bytes are readable.
        codebook = qc.get("codebook")
        if codebook is not None and str(codebook) not in ("mul1", "mcg"):
            raise Refusal(
                "candidate quantization_config codebook=%r is not one this decoder "
                "speaks (mul1/mcg)" % (codebook,),
                ["exl3hf_surface transcribes exllamav3's mul1 and mcg codebooks; "
                 "another codebook needs its LUT authored and proven first."])
        layout = _exl3_layout_block(index_keys, qc, None)
        contract = {
            "quant_method": method,
            "codebook": str(codebook) if codebook is not None else None,
            "bits": qc.get("bits"),
            "head_bits": qc.get("head_bits"),
            "modules_to_not_convert": sorted(
                str(m) for m in (qc.get("modules_to_not_convert") or [])),
        }
        contract.update({k: v for k, v in layout.items() if not k.startswith("_")})
        return {
            "method": CANDIDATE_DECODE_METHOD_TRELLIS,
            "quantization_config": contract,
            "_declaration": {"declared_by": "quantization_config",
                             "quant_method_declared": method, "tp": None,
                             "layout": layout["_layout_detail"]},
        }
    if method == "modelopt":
        # The modelopt NVFP4 surface `engines/tools/layer_outer.materialize_nvfp4_subset`
        # decodes: routed experts packed e2m1 group-16 with an f8 per-group
        # scale and an fp32 per-tensor scale_2, decoded on the capture device
        # (engines/tools/nvfp4-evidence/glm53-nvfp4-parity.json). Note a
        # davidsyoung-style leftover modelopt block under a hybrid_tr3_tail
        # was taken by the tail branch above and never reaches here.
        return _nvfp4_candidate_decode_plan(qc)
    if not (method == "fp8" and fmt == "e4m3"
            and isinstance(block, list) and len(block) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in block)
            and activation in (None, "dynamic")):
        raise Refusal(
            "candidate quantization_config quant_method=%r fmt=%r weight_block_size=%r "
            "activation_scheme=%r is not a surface the layer-outer loader decodes "
            "(block-scaled FP8 e4m3 weights-only, stock-exllamav3 exl3 trellis, or "
            "modelopt NVFP4)"
            % (method, fmt, block, activation),
            ["Decodable today: zai-org/GLM-5.3-style FineGrainedFP8, exl3 "
             "trellis with mul1/mcg payload groups, and modelopt NVFP4 (e2m1 group-16 "
             "routed experts). Another surface needs its decoder authored and proven "
             "bitwise first."])
    return {
        "method": CANDIDATE_DECODE_METHOD,
        "quantization_config": {
            "quant_method": method, "fmt": fmt,
            "weight_block_size": [int(block[0]), int(block[1])],
            "activation_scheme": activation,
            "modules_to_not_convert": sorted(
                str(m) for m in (qc.get("modules_to_not_convert") or [])),
        },
    }


CANDIDATE_SCOPE_REMOTE = "candidate/scope.json"


def _candidate_reference_config(args, plan_data: Dict[str, Any]) -> Tuple[bytes, Tuple[str, str]]:
    """The reference root's WEIGHTS release config.json, anonymously, as bytes.

    The root dataset's manifest names the release it was captured from
    (`weights.repository` @ `weights.model_revision`); that release's
    config.json is what the pod copies beside a GGUF build so the HF model
    class can be constructed (stage_measure.sh fetch_reference/capture).
    """
    cached = plan_data.get("_candidate_reference_config")
    if cached is not None:
        return cached
    manifest = _candidate_reference_manifest(args, plan_data)
    weights = manifest.get("weights") or {}
    repo = weights.get("repository")
    rev = weights.get("model_revision") or weights.get("revision")
    if not isinstance(repo, str) or not repo or not isinstance(rev, str) \
            or re.fullmatch(r"[0-9a-f]{40}", rev) is None:
        raise Refusal("the reference dataset names no pinned weights release (weights."
                      "repository/model_revision)", [])
    try:
        raw = fetch_file(repo, "config.json", revision=rev)
    except HFError as exc:
        raise Refusal("the reference release's config.json could not be read from %s@%s: %s"
                      % (repo, rev[:12], redact(str(exc))), [])
    plan_data["_candidate_reference_config"] = (raw, (repo, rev))
    return raw, (repo, rev)


def _candidate_reference_manifest(args, plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """The reference root's top manifest, fetched ANONYMOUSLY once per plan."""
    cached = plan_data.get("_candidate_reference_manifest")
    if cached is not None:
        return cached
    from fidelity import dshub
    from fidelity import dsformat as F

    match = re.fullmatch(r"([^\s/@]+/[^\s/@]+)@([0-9a-f]{40})",
                         args.reference_dataset or "")
    if match is None:
        raise Refusal("--reference-dataset must be OWNER/REPO@<40-hex revision>", [])
    ref_repo, ref_rev = match.group(1), match.group(2)
    dshub.validate_repo_id(ref_repo)
    with tempfile.TemporaryDirectory(prefix="fidelity-reference-") as cache:
        try:
            root = dshub.fetch_dataset(
                "hf://%s@%s" % (ref_repo, ref_rev), os.path.join(cache, "reference"),
                token=None, manifest_only=True)
            manifest = F.load_manifest(root)
        except Exception as exc:
            raise Refusal("--reference-dataset could not be read anonymously: %s"
                          % redact(str(exc)), [])
    dataset = manifest.get("dataset") or {}
    if dataset.get("role") != "root" or dataset.get("structural_status") != "sealed":
        raise Refusal("--reference-dataset is not a sealed root dataset", [])
    plan_data["_candidate_reference_manifest"] = manifest
    plan_data["_candidate_reference_ref"] = (ref_repo, ref_rev)
    return manifest


def _candidate_block(args, plan_data: Dict[str, Any], con: Console,
                     binding_panel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """job.capture.candidate: the authored scope (by digest), codec, bits,
    the decode the loader must apply, and the exact reference dataset."""
    flags = (args.candidate_scope, args.candidate_codec, args.candidate_bits,
             args.reference_dataset)
    if not any(flag is not None for flag in flags):
        return None
    if not all(flag is not None for flag in flags):
        raise Refusal(
            "--candidate-scope, --candidate-codec, --candidate-bits and "
            "--reference-dataset are all-or-none", [])
    from fidelity import dshub, dsmanifest
    from fidelity import dsformat as F

    scope_path = Path(args.candidate_scope).expanduser().resolve()
    if scope_path.is_symlink() or not scope_path.is_file():
        raise Refusal("--candidate-scope must be a regular JSON file", [])
    raw = scope_path.read_bytes()
    try:
        doc = parse_job_bytes(raw)
    except JobContractError as exc:
        raise Refusal("--candidate-scope is not strict JSON: %s" % exc, [])
    for key in ("policy", "head_policy", "assignments"):
        if key not in doc:
            raise Refusal("--candidate-scope lacks %r" % key, [])
    try:
        scope_block = dsmanifest.scope_block(
            doc["assignments"], doc["head_policy"],
            doc.get("kv_cache_dtype", "bf16"), doc["policy"])
    except Exception as exc:
        raise Refusal("--candidate-scope does not form a scope block: %s" % exc, [])
    # The registry's scope rules, applied at $0: SCOPE-004 (one row per
    # class and layer_range), the closed assignment schema, class coverage,
    # and the numeric_format enum. scope_digest is sealed into the dataset, so
    # a scope the registry refuses can only be fixed by a re-capture -- both
    # rows published on 2026-09-04 carried duplicate rows this would have
    # refused before any spend.
    from fidelity import dsvalidate
    scope_report = dsvalidate.Report(str(scope_path))
    dsvalidate._validate_scope_vocabulary(scope_block, scope_report, strict=True)
    if scope_report.errors:
        raise Refusal(
            "--candidate-scope would be refused by the registry (%d finding(s))"
            % len(scope_report.errors),
            ["%s: %s" % (e.get("rule"), e.get("message")) for e in scope_report.errors])
    for w in scope_report.warnings:
        con.warn("scope %s: %s" % (w.get("rule"), w.get("message")))
    if not (isinstance(args.candidate_bits, float) and 0 < args.candidate_bits <= 16):
        raise Refusal("--candidate-bits must be in (0, 16]", [])
    # The registry's closed codec vocabulary, READ from its schema (S2-5: the
    # free-text flag had drifted three ways across real jobs -- exl3,
    # exl3-mcg, exl3-trellis -- and one of them is not a registry value).
    formats = dsvalidate.registry_numeric_formats()
    if not formats:
        raise Refusal("registry/schema/common.schema.json numeric_format enum is unreadable", [])
    if args.candidate_codec not in formats:
        raise Refusal(
            "--candidate-codec %r is not in the registry's numeric_format vocabulary"
            % args.candidate_codec,
            ["choose one of: " + ", ".join(sorted(formats)),
             "the codec is a row identity; a value outside the enum cannot be filed"])
    decode = plan_data.get("_candidate_decode")
    if decode is None:
        raise Refusal("candidate decode plan is absent; the target gate did not run", [])
    # --candidate-bits re-declares what the target gate already read from the
    # release: quantization_config.bits / hybrid_tr3_tail.bits_avg (exl3),
    # num_bits (nvfp4), or the format's own width (fp8 e4m3 = 8). Refuse on
    # disagreement with both numbers; the pod re-checks the payload's K.
    qcfg = decode.get("quantization_config") or {}
    declared = qcfg.get("bits")
    if declared is None:
        if qcfg.get("quant_method") == "fp8" and qcfg.get("fmt") == "e4m3":
            declared = 8
        elif qcfg.get("quant_method") == "modelopt":
            declared = 4
        elif qcfg.get("container") == "gguf":
            # a GGUF declares no bit width; the NOMINAL rate is the build
            # name's (the same number `sniff_surface` puts on target.bits, so
            # the root qualification contract closes) and the MEASURED rate
            # rides on bits_per_weight_effective and in every scope row
            from fidelity.hfmeta import gguf_nominal_rate
            declared = gguf_nominal_rate(str(qcfg.get("build")))[0]
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        if abs(float(declared) - args.candidate_bits) > 1e-9:
            raise Refusal(
                "--candidate-bits %g disagrees with the checkpoint's own declaration %g"
                % (args.candidate_bits, float(declared)),
                ["the release declares %g (%s); pass --candidate-bits %g, or fix the "
                 "release" % (float(declared),
                              "hybrid_tr3_tail.bits_avg"
                              if decode.get("method") == CANDIDATE_DECODE_METHOD_TRELLIS_TP
                              else "quantization_config", float(declared))])
    else:
        con.warn("candidate bits: the release declares no bit width the controller "
                 "reads; --candidate-bits %g is checked against the payload on the pod"
                 % args.candidate_bits)

    manifest = _candidate_reference_manifest(args, plan_data)
    ref_repo, ref_rev = plan_data["_candidate_reference_ref"]
    if ref_repo == args.dataset_repository:
        raise Refusal("--reference-dataset must differ from --dataset-repository", [])
    dataset = manifest.get("dataset") or {}
    ref_panel = manifest.get("panel") or {}
    if ref_panel.get("panel_id") != binding_panel.get("id") or (
            ref_panel.get("suite_token_hash_sha256")
            != binding_panel.get("suite_token_hash_sha256")):
        raise Refusal(
            "--reference-dataset was captured on panel %r (%s), not this job's %r (%s)"
            % (ref_panel.get("panel_id"), str(ref_panel.get("suite_token_hash_sha256"))[:16],
               binding_panel.get("id"), str(binding_panel.get("suite_token_hash_sha256"))[:16]),
            [])
    reference = {
        "repository": ref_repo, "revision": ref_rev,
        "dataset_sha256": manifest.get(F.SEAL_FIELD),
        "capture_content_digest": (manifest.get("capture") or {}).get("capture_content_digest"),
        "dataset_id": dataset.get("id"),
        "panel_id": ref_panel.get("panel_id"),
        "suite_token_hash_sha256": ref_panel.get("suite_token_hash_sha256"),
    }
    plan_data["_candidate_scope_local"] = str(scope_path)
    block = {
        "scope": {"path": CANDIDATE_SCOPE_REMOTE,
                  "sha256": hashlib.sha256(raw).hexdigest(),
                  "scope_digest": scope_block["scope_digest"]},
        "codec": args.candidate_codec,
        "declared_bits": args.candidate_bits,
        "weights_decode": decode,
        "reference": reference,
    }
    from fidelity.jobcontract import valid_candidate
    if not valid_candidate(block):
        raise Refusal("candidate block is incomplete: %s" % json.dumps(block)[:300], [])
    con.ok("candidate reference",
           "%s@%s dataset_sha256 %s, capture %s, panel %s"
           % (ref_repo, ref_rev[:12], reference["dataset_sha256"][:16],
              reference["capture_content_digest"][:16], reference["panel_id"]))
    return block


def _report_candidate_result(con: Console, extracted: Path, job: Dict[str, Any]) -> None:
    """The number, from the verified archive, on the console: KLD(root || candidate)
    with its top-1 agreement, the reference and candidate seals, and the
    receipt path. The registry session ingests the receipt; this line is for
    the operator watching the run."""
    receipt_path = extracted / "receipts" / "reference-comparison" / "comparison-receipt.json"
    if not receipt_path.is_file():
        con.warn("candidate comparison receipt is absent from the verified archive")
        return
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    metric = doc.get("metric") or {}
    top1 = doc.get("top1_agreement")
    if isinstance(top1, dict):
        top1 = top1.get("value", top1.get("mean"))
    candidate = job["capture"]["candidate"]
    con.ok("candidate scored",
           "KLD(root || candidate) = %r nats (%s); top-1 agreement %r; "
           "reference %s@%s (%s); candidate %s %s bits (%s); receipt %s"
           % (metric.get("value"), metric.get("name"), top1,
              candidate["reference"]["repository"],
              candidate["reference"]["revision"][:12],
              candidate["reference"]["dataset_sha256"][:16],
              candidate["codec"], candidate["declared_bits"],
              str((doc.get("candidate") or {}).get("dataset_sha256"))[:16],
              receipt_path.relative_to(extracted)))


def _refuse_quantized_root(con: Console, target, surface, plan: Dict[str, Any],
                           args=None) -> None:
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
    if getattr(args, "candidate_scope", None) and surface.surface == "gguf":
        # A GGUF repo ships no config.json at all: what it declares about its
        # quantization is in its own headers, and the model-class config is
        # the reference root's release (read anonymously through the reference
        # dataset's manifest, exactly as the pod's fetch_reference stage does).
        official_raw, official_ref = _candidate_reference_config(args, plan)
        official = json.loads(official_raw.decode("utf-8"))
        decode = _gguf_candidate_plan(con, target, surface, plan, official)
        plan["_gguf_official_config"] = (official_raw, official_ref)
        plan.setdefault("target", {})["root_unquantized"] = False
        plan["_candidate_decode"] = decode
        return
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
    designated = bool(getattr(args, "designated_reference", False))
    if getattr(args, "candidate_scope", None):
        # A candidate is the root protocol on a quantized target: the config
        # MUST declare the one form the streaming loader decodes, and what it
        # declares is bound into the job so the pod's runtime receipt has to
        # record exactly that decode.
        index_keys = None
        tail = cfg.get("hybrid_tr3_tail")
        if ((isinstance(qc, dict) and qc.get("quant_method") == "exl3")
                or (isinstance(tail, dict) and tail.get("format") == "exl3-trellis")):
            # An exl3 candidate's rotation layout is contract and is read from
            # the index NAMES (stock, experts.shared_h, experts.r7_shared) --
            # the same rule the pod applies after the fetch, applied here at $0.
            try:
                index_keys = list(fetch_json(
                    target.repo_id, "model.safetensors.index.json",
                    revision=target.revision)["weight_map"])
            except (HFError, KeyError, TypeError, ValueError) as exc:
                raise Refusal(
                    "exl3 candidate: model.safetensors.index.json could not be read (%s); "
                    "the rotation layout is decided from its names" % redact(str(exc)),
                    ["Nothing was created. $0.00 spent."])
        # jpsequeira's GLM-5.2 TR3 declares bits: "mixed" beside
        # bits_per_expert: "expert_precision_map.json:bitrates"; the sidecar
        # is fetched by name from the target repo/revision through the same
        # anonymous fetch_file the pod uses against the checkpoint dir, so
        # the controller mirror's declared_bits_source is byte-identical to
        # the pod's (same sha256).
        def _controller_sidecar_loader(sfile):
            sraw = fetch_file(target.repo_id, sfile, revision=target.revision)
            return json.loads(sraw), hashlib.sha256(sraw).hexdigest()
        decode = _candidate_decode_plan(
            qc, cfg, index_keys=index_keys, sidecar_loader=_controller_sidecar_loader)
        declaration = decode.pop("_declaration", None)
        qcfg = decode["quantization_config"]
        if qcfg["quant_method"] == "exl3":
            layout = (declaration or {}).get("layout") or {}
            con.ok("candidate is exl3 trellis",
                   "quant_method %s codebook %s declared bits %s; decoded to bf16 per "
                   "module (%s). Per-module codebook and the payload's own bit width are "
                   "read from the checkpoint on the pod and checked against the "
                   "declaration%s"
                   % (qcfg["quant_method"], qcfg["codebook"], qcfg["bits"],
                      decode["method"],
                      ("; declared by hybrid_tr3_tail (quantization_config says %r), "
                       "tp=%s rank shards composed per module"
                       % (declaration.get("quant_method_declared"), declaration.get("tp")))
                      if declaration and declaration.get("declared_by") == "hybrid_tr3_tail"
                      else ""))
            con.ok("exl3 rotation layout %s" % qcfg["rotation_layout"],
                   "read from the index names: %s; %d layer-shared rotation vector(s) "
                   "(sha256 %s); %d non-routed exl3 module(s) decoded by the same function "
                   "(declared bits %s); activation overlay %s"
                   % (", ".join("%s x%d" % kv for kv in sorted(
                          (layout.get("modules_per_layout") or {}).items())) or "none",
                      qcfg["shared_vectors"]["count"],
                      (qcfg["shared_vectors"]["names_sha256"] or "none")[:12],
                      qcfg["nonrouted_exl3"]["count"],
                      qcfg["nonrouted_exl3"]["declared_bits"] or "{}",
                      qcfg["activation_scheme"] or "none declared"))
            if qcfg["nonrouted_exl3"].get("declared_bits", {}).get("undeclared"):
                con.warn("%d non-routed exl3 module(s) carry no declared bits; the pod "
                         "records each payload's own K and the row's label rests on "
                         "the K histogram"
                         % qcfg["nonrouted_exl3"]["declared_bits"]["undeclared"])
            if "lm_head" in (layout.get("nonrouted_bits") or {}):
                con.ok("exl3 lm_head",
                       "the head is an exl3 payload: the capture ships the candidate's OWN "
                       "dequantized head (quantized, artifact_dequantized) and the "
                       "comparison runs under HEAD-1d, own heads")
        elif qcfg["quant_method"] == "modelopt":
            con.ok("candidate is modelopt NVFP4",
                   "quant_algo %s, %d-bit group-%d weights declared by %s, producer %s; "
                   "routed experts decoded to bf16 per module on the capture device (%s); "
                   "the static input_scale is an activation quantity and is NOT applied "
                   "(%s); %d ignore entries hashed into the contract. The index census "
                   "(57,600 routed modules, official non-routed names) runs on the pod."
                   % (qcfg["quant_algo"], qcfg["num_bits"], qcfg["group_size"],
                      qcfg["weights_declared_by"],
                      ("%s %s" % (qcfg["producer"]["name"], qcfg["producer"]["version"])
                       if qcfg["producer"] else "undeclared"),
                      decode["method"], qcfg["activation_scheme"], qcfg["ignore_count"]))
        else:
            con.ok("candidate is block-scaled FP8",
                   "quant_method %s fmt %s block %s; decoded to bf16 per tensor "
                   "(%s), %d modules kept native"
                   % (qcfg["quant_method"], qcfg["fmt"],
                      qcfg["weight_block_size"], decode["method"],
                      len(qcfg["modules_to_not_convert"])))
        plan.setdefault("target", {})["root_unquantized"] = False
        plan["_candidate_decode"] = decode
        return
    if not qc:
        if designated:
            # The flag on an UNQUANTIZED checkpoint is a contradiction, and
            # accepting it would let a proxy reference be minted for a family
            # that has a real root -- turning an advisory-by-necessity
            # mechanism into an advisory-by-convenience one.
            raise Refusal(
                "--designated-reference, but this checkpoint publishes no "
                "quantization_config: it IS an unquantized root",
                ["A designated reference exists for families that publish no "
                 "unquantized weights at all. This family does. Capture it as "
                 "a plain root and drop the flag.",
                 "Nothing was created. $0.00 spent."])
        con.ok("root is unquantized",
               "surface %s: config.json declares no quantization_config"
               % surface.surface)
        plan.setdefault("target", {})["root_unquantized"] = True
        return
    method = qc.get("quant_method") or qc.get("fmt") or "declared"
    if designated:
        # REFC-006's case, entered explicitly and recorded everywhere the
        # output travels: the plan, the job document (below, via this field),
        # and therefore the sealed dataset. Its 0.0 self-compare is an origin
        # we chose, not a floor we measured.
        con.warn("DESIGNATED REFERENCE: this checkpoint is quantized "
                 "(quant_method %s) and is being captured as the family's "
                 "designated reference -- an origin by designation, not a "
                 "measured floor. Rows against it will be advisory "
                 "(REFC-006)." % method)
        plan.setdefault("target", {})["root_unquantized"] = False
        plan["target"]["designated_reference"] = {
            "quant_method": method,
            "note": "quantized_proxy reference; no unquantized release "
                    "exists for this family"}
        return
    # A quantized checkpoint is what the CANDIDATE route measures. Say so with
    # the command filled in from what this call already knows; the human who
    # passed --reference-dataset without the other three was one flag away.
    reference = getattr(args, "reference_dataset", None) if args is not None else None
    codec_hint = {"fp8": "fp8_e4m3", "exl3": "exl3-mcg", "modelopt": "nvfp4"}.get(
        str(method), "<registry numeric_format>")
    bits_hint = qc.get("bits") if isinstance(qc.get("bits"), (int, float)) else (
        8 if method == "fp8" else "<declared bpw>")
    tool = "fp8_scope.py" if method == "fp8" else "exl3_scope.py"
    command = [
        "  measure-cloud --provider runpod --role candidate \\",
        "    --model %s --revision %s \\" % (target.repo_id, target.revision),
        "    --panel-dir %s --dataset-id %s \\"
        % (getattr(args, "panel_dir", None) or "engines/panels/<panel-of-the-root>",
           getattr(args, "dataset_id", None) or "fidelity--<family>.<hub-handle>.quant.<slug>"),
        "    --reference-dataset %s \\"
        % (reference or "<OWNER/REPO@40HEX of the family's published root dataset>"),
        "    --candidate-scope <engines/tools/%s output> --candidate-codec %s "
        "--candidate-bits %s \\" % (tool, codec_hint, bits_hint),
        "    --gpu H200 --measurer %s --max-cost <usd> --max-runtime <duration> "
        "--out <dir> --dry-run" % (getattr(args, "measurer", None) or "<hub-handle>"),
    ]
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
         "To measure THIS quantized checkpoint against its family's published "
         "root, use the candidate route (docs/THIRD-PARTY-QUICKSTART.md 3b):"]
        + command +
        ["To capture a root, point --model at the unquantized release. If the "
         "family publishes none -- as no deepseek_v4 repo on the Hub does -- "
         "then it has no root and no floor can be measured for it.",
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
                     plan: Dict[str, Any], *, args, profile: str) -> None:
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
        gate_not_checked(con, plan, args, "tr3-seal",
                         "engines/tools/tr3_surface not importable: %s"
                         % redact(str(exc)))
        return
    import tempfile

    try:
        policy = tr3s.preflight_public_profile(
            profile, repo=repo_id, revision=revision)
    except ValueError as exc:
        gate_failed(plan, "tr3-public-profile", redact(str(exc)))
        raise Refusal(
            "TR3 public profile is not admitted before evidence fetch",
            [redact(str(exc)), "Nothing was created. $0.00 spent."])
    required_files = {
        "config.json", "model.safetensors.index.json",
        tr3s.ABI_FILE, tr3s.MATERIALIZATION_FILE,
    }
    if policy is not None:
        required_files.update(policy["raw_sha256"])
    try:
        blobs = {
            name: fetch_file(repo_id, name, revision=revision)
            for name in sorted(required_files)
        }
        index_doc = parse_job_bytes(
            blobs["model.safetensors.index.json"])
        weight_map = index_doc.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise JobContractError(
                "TR3 index has no non-empty weight_map")
    except (HFError, JobContractError, UnicodeError, ValueError) as exc:
        gate_not_checked(con, plan, args, "tr3-seal",
                         "could not fetch the release's seal metadata: %s"
                         % redact(str(exc)))
        return
    with tempfile.TemporaryDirectory(prefix="tr3-seal-") as tmp:
        root = Path(tmp)
        for name, blob in blobs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        try:
            seal = tr3s.verify_seal(
                root, weight_map,
                config_path=root / "config.json",
                index_path=root / "model.safetensors.index.json")
            profile_evidence = tr3s.verify_public_profile_evidence(
                root, profile, repo=repo_id, revision=revision,
                shard_verification={
                    "mode": "crosscheck",
                    "verification": {
                        "status": "deferred-to-remote-crosscheck",
                        "verified": False,
                        "required_mode": "crosscheck",
                    },
                })
        except ValueError as exc:
            gate_failed(plan, "tr3-seal", redact(str(exc)))
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
    if profile_evidence is not None:
        plan["target"]["public_profile_evidence"] = profile_evidence
        gate_verified(
            plan, "tr3-public-profile",
            profile=profile,
            verdict_raw_sha256=profile_evidence["verdict_raw_sha256"],
            checkpoint_identity_sha256=
            profile_evidence["checkpoint_identity_sha256"],
            shard_verification="deferred-to-remote-crosscheck")
    plan["target"]["nonrouted_completeness"] = {
        "official_nonrouted": seal["materialization"]["native_tensor_count"],
        "planned": seal["materialization"]["native_tensor_count"],
        "missing": 0, "duplicated": 0,
    }
    gate_verified(plan, "tr3-seal", checks_passed=passed)
    con.ok("published seal", "%d/%d claims recomputed from the release's own bytes"
           % (passed, len(seal["checks"])))


def _verify_gguf_readable(con: Console, repo_id: str, revision: str,
                          surface, plan: Dict[str, Any], *, args) -> None:
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
        gate_not_checked(con, plan, args, "gguf-readability",
                         "engines/tools/gguf_surface not importable: %s"
                         % redact(str(exc)))
        return
    urls = ["%s/%s/resolve/%s/%s" % (HF_ENDPOINT, repo_id, revision, name)
            for name, _ in surface.artifact_files]
    try:
        loaded = ggs.load_gguf_surface(urls, repo=repo_id, revision=revision,
                                       require_file_hashes=False)
    except ValueError as exc:
        gate_failed(plan, "gguf-readability", redact(str(exc)))
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
        gate_not_checked(con, plan, args, "gguf-readability",
                         "gguf_surface could not read the build: %s"
                         % redact(str(exc)))
        return
    summary = ggs.surface_summary(loaded)
    try:
        official = fetch_json("zai-org/GLM-5.3-Flash-BF16",
                              "model.safetensors.index.json",
                              revision=OFFICIAL_BF16_REVISION)["weight_map"]
        bijection = ggs.verify_nonrouted_bijection(loaded.census, official.keys())
    except (HFError, ValueError) as exc:
        gate_not_checked(con, plan, args, "gguf-bijection",
                         "official non-routed index unavailable: %s"
                         % redact(str(exc)))
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
    gate_verified(plan, "gguf-readability")
    if bijection is not None and bijection.get("bijection_ok"):
        gate_verified(plan, "gguf-bijection")
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
        # Every surface/seal/lane/profile gate below keys off the resolved
        # target, so NONE of them will run: this plan is an estimate and must
        # say so instead of ending in "all checks passed" (P1-11).
        gate_not_checked(con, plan, args, "target-resolution",
                         "Hugging Face unreachable; the pinned census stands "
                         "in for the real repo")
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
            _refuse_quantized_root(con, target, surface, plan, args=args)
        if surface.surface == "exl3hf" and not surface.problems:
            _refuse_incomplete_exl3hf(con, target.repo_id, target.revision, plan)
        if not surface.problems:
            _refuse_scope_contradicted_by_release(
                con, target.repo_id, target.revision, surface,
                read_json(args.scope_json) if getattr(args, "scope_json", None) else None,
                plan)
        if surface.surface == "tr3-published" and not surface.problems:
            if surface.bits is None:
                raise Refusal(
                    "TR3 surface has no exact public bit profile", [])
            _verify_tr3_seal(
                con, target.repo_id, target.revision, plan, args=args,
                profile="tr3-%gbpw" % float(surface.bits))
        if surface.surface == "gguf" and not surface.problems:
            _verify_gguf_readable(con, target.repo_id, target.revision,
                                  surface, plan, args=args)
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

    # -- job identity, now that every mutable reference is resolved --------
    # (P1-12: identity is derived AFTER resolution, at 256 bits; the 8-char
    # prefix is display + instance-name budget, never compared on its own.)
    identity = job_identity(
        args,
        resolved_revision=(target.revision if target is not None else None),
        panel_revision=panel_rev)
    if identity["job_id"] != plan["job_id"]:
        con.kv("job id", "%s  (final; provisional was %s)"
               % (identity["job_id"], plan["job_id"]))
    plan["job_id"] = identity["job_id"]
    plan["job_id_full"] = identity["job_id_full"]
    plan["suite_head"] = _suite_head()

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
                # No offers means NO RATE EXISTS: pricing below refuses to
                # invent one, and the verdict is INCOMPLETE (a field tester
                # got "an UNKNOWN GPU at $0.00/h" and a $0.21 "plan").
                gate_not_checked(con, plan, args, "instance-pricing",
                                 "provider offers unreachable; no rate "
                                 "exists for this plan")
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
    timing_basis = "lane"
    authored_per_window = timing.get("minutes_per_window")
    if surface_key in by_surface:
        authored_per_window = by_surface[surface_key]
        timing_basis = "surface %s" % surface_key
    try:
        per_window = float(authored_per_window)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Refusal(
            "lane %s has no numeric timing evidence for surface %s"
            % (args.lane, surface_key),
            ["author an exact target/profile/hardware timing before paid use",
             "Nothing was created. $0.00 spent."]) from exc
    if not math.isfinite(per_window) or per_window <= 0:
        raise Refusal(
            "lane %s timing evidence for surface %s is not positive finite"
            % (args.lane, surface_key),
            ["correct bin/engines.json before paid use",
             "Nothing was created. $0.00 spent."])
    measured = bool(timing.get("measured") or surface_key in by_surface)
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
    if offer is None:
        # NO RATE EXISTS.  Printing "$0.00/h" and a storage-only dollar total
        # is not an estimate, it is an invented number wearing one's clothes;
        # a third party read exactly that as a $0.21 plan.  Refuse to price:
        # hours are still shown (they come from the timing model), dollars
        # are not.
        con.say("  COST ESTIMATE: UNPRICEABLE -- provider offers were not "
                "readable, so no rate exists")
        for name, hours, why in phases:
            con.say("    %-14s %-34s %5.2f h  $   ?" % (name, why, hours))
        con.say("    %-14s %-34s %5.2f h  $   ?"
                % ("storage", "%d GB fs (rate INFERRED, +/-100%%)" % storage_gb,
                   total_h))
        point = band_hi = ceiling = None
        plan["cost_estimate"] = {
            "rate_per_hour": None, "unpriceable": True,
            "phases": [{"name": n, "hours": h, "note": w}
                       for n, h, w in phases],
            "point_usd": None, "band_high_usd": None, "ceiling_usd": None,
            "storage_rate_per_hour": storage_rate,
            "storage_rate_provenance":
                "INFERRED from reconciling one live instance against its list "
                "rate; JarvisLabs publishes no storage line. Treat as +/-100% "
                "and rely on the balance delta for ground truth.",
        }
    else:
        point = rate * total_h + storage_rate * total_h
        con.say("  COST ESTIMATE")
        con.kv("rate", "%d x %s %s  $%.2f/h"
               % (req.gpus, offer.gpu_type,
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

    if (args.max_cost and band_hi is not None
            and band_hi > float(args.max_cost)):
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
# Safe RunPod controller (one fresh SSH pod, no adoption/recovery)
# ==========================================================================

CONTROL_PLANE_PATHS = (
    "bin/measure_cloud.py", "bin/result_archive.py",
    "bin/fidelity/campaign.py", "bin/fidelity/cloudlease.py",
    "bin/fidelity/common.py", "bin/fidelity/engines.py",
    "bin/fidelity/hfmeta.py", "bin/fidelity/jlapi.py",
    "bin/fidelity/jobcontract.py", "bin/fidelity/panel.py",
    "bin/fidelity/resultsink.py", "bin/fidelity/runpodapi.py",
    "bin/fidelity/runpoddrill.py", "bin/fidelity/runpodsafety.py",
    "bin/fidelity/sshbase.py", "bin/fidelity/stages.py",
)

# Where a RunPod pod's run root lives. Measured 2026-09-03 on a secure H200
# (us-co-1): the pod VOLUME at /workspace is MooseFS over FUSE, shared with
# other tenants, mode-forcing, 508 MB/s direct read idle and ~215 MB/s under
# contention -- the layer-outer capture streams the whole checkpoint through
# it once per cold run (162 min for GLM-5.3). The CONTAINER disk is the host's
# local NVMe overlay: 3.5 GB/s write, 5.9 GB/s read. Nothing here survives
# the pod, and the safe path never restarts or adopts one, so the run root
# belongs on the container disk. A nominal volume is still created so the
# live attestation's /workspace probe keeps its meaning.
RUNPOD_STORAGE_LAYOUTS = {
    "container-disk": {"run_base": "/root", "nominal_volume_gb": 10},
    "pod-volume": {"run_base": "/workspace", "nominal_volume_gb": None},
}


def _runpod_run_roots(layout: str, job_id_full: str, attempt: str):
    base = RUNPOD_STORAGE_LAYOUTS[layout]["run_base"]
    return ("%s/fidelity/%s/%s" % (base, job_id_full, attempt),
            "%s/fidelity-engine/%s/%s" % (base, job_id_full, attempt))


def _canonical_bytes(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

_RUNPOD_TEMP_HOLDS = []



def _strict_file_manifest(paths, *, registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """Hash an authored path closure; missing/symlink/escape/duplicate all refuse."""
    root = SUITE_ROOT.resolve()
    seen, rows = set(), []
    for raw in paths:
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise Refusal("manifest contains an invalid path %r" % raw, [])
        pure = PurePosixPath(raw)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise Refusal("manifest path escapes the suite: %r" % raw, [])
        rel = str(pure)
        if rel in seen:
            raise Refusal("manifest contains duplicate path %s" % rel, [])
        seen.add(rel)
        path = root.joinpath(*pure.parts)
        cursor = root
        for part in pure.parts:
            cursor = cursor / part
            try:
                mode = cursor.lstat().st_mode
            except OSError:
                raise Refusal("manifest file is missing: %s" % rel, [])
            if stat.S_ISLNK(mode):
                raise Refusal("manifest path contains a symlink: %s" % rel, [])
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise Refusal("manifest path escapes the suite: %s" % rel, [])
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise Refusal("manifest entry is not a regular file: %s" % rel, [])
        rows.append({"path": rel, "bytes": path.stat().st_size,
                     "sha256": sha256_file(str(path))})
    rows.sort(key=lambda row: row["path"])
    if registry_path is not None:
        result = finalize_bundle_manifest(rows, "BUNDLE.txt")
    else:
        result = {
            "schema": "fidelity-suite/control-plane-manifest.v1",
            "source": "authored-control-plane-closure",
            "files": rows,
            "manifest_sha256": hashlib.sha256(
                _canonical_bytes(rows)).hexdigest(),
        }
    return result


def _bundle_manifest() -> Dict[str, Any]:
    registry = SUITE_ROOT / "bin" / "BUNDLE.txt"
    if not registry.is_file() or registry.is_symlink():
        raise Refusal("BUNDLE.txt is missing or symlinked", [])
    entries = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            entries.append(value)
    return _strict_file_manifest(entries, registry_path=registry)


def _bundle_registry_identity() -> Dict[str, Any]:
    registry = SUITE_ROOT / "bin" / "BUNDLE.txt"
    if (not registry.is_file() or registry.is_symlink()
            or registry.resolve().parent != (SUITE_ROOT / "bin").resolve()):
        raise Refusal("BUNDLE.txt must be an exact regular suite file", [])
    return {
        "path": "bin/BUNDLE.txt", "bytes": registry.stat().st_size,
        "sha256": sha256_file(str(registry)),
    }


def _control_manifest() -> Dict[str, Any]:
    return _strict_file_manifest(CONTROL_PLANE_PATHS)


def _model_file_identity(target: RepoMeta) -> Dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", target.revision) is None:
        raise Refusal("RunPod target revision is not an exact 40-hex pin", [])
    try:
        config_raw = fetch_file(target.repo_id, "config.json",
                                revision=target.revision)
        index_raw = fetch_file(target.repo_id, "model.safetensors.index.json",
                               revision=target.revision)
        index = parse_job_bytes(index_raw)
        config = parse_job_bytes(config_raw)
    except (HFError, JobContractError, UnicodeError, ValueError) as exc:
        raise Refusal("target config/index identity is unavailable: %s" % exc, [])
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise Refusal("target index has no non-empty weight_map", [])
    mapped_shards = list(weight_map.values())
    if any(not isinstance(name, str) for name in mapped_shards):
        raise Refusal("target index contains a non-string shard path", [])
    shard_names = sorted(set(mapped_shards))
    for name in shard_names:
        pure = PurePosixPath(name)
        if (("\\" in name) or pure.is_absolute()
                or pure.as_posix() != name
                or any(part in ("", ".", "..") for part in pure.parts)):
            raise Refusal("target index contains an unsafe shard path", [])
    if len({path for path, _size in target.files}) != len(target.files):
        raise Refusal("target repository contains duplicate file paths", [])
    sizes = dict(target.files)
    if any(name not in sizes or sizes[name] <= 0 for name in shard_names):
        raise Refusal("target index names a missing or size-unknown shard", [])
    repository_shards = sorted(
        path for path, _size in target.files
        if path.endswith(".safetensors"))
    if repository_shards != shard_names:
        raise Refusal(
            "repository safetensors census differs from indexed shards", [])
    shards = [{"path": name, "bytes": sizes[name]} for name in shard_names]
    repository_files = []
    for path, size in sorted(target.files):
        pure = PurePosixPath(path)
        if (("\\" in path) or pure.is_absolute()
                or pure.as_posix() != path
                or any(part in ("", ".", "..") for part in pure.parts)
                or isinstance(size, bool) or not isinstance(size, int)
                or size < 0):
            raise Refusal(
                "target repository contains an unsafe or size-unknown file",
                [])
        repository_files.append({"path": path, "bytes": size})
    # A vision-language config (GLM-5.3-Flash-BF16, MiniMax-M3) nests the
    # language model's geometry under text_config; the engines already read
    # it there. The identity digests stay over the raw bytes.
    text_config = config.get("text_config")
    geometry = text_config if isinstance(text_config, dict) else config
    vocab_size = geometry.get("vocab_size")
    if (isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int) or vocab_size <= 0):
        raise Refusal("target config lacks positive exact vocab_size", [])
    hidden_size = geometry.get("hidden_size")
    if (isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int) or hidden_size <= 0):
        raise Refusal("target config lacks positive exact hidden_size", [])
    return {
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "config_bytes": len(config_raw),
        "index_bytes": len(index_raw),
        "model_bytes": sum(row["bytes"] for row in shards),
        "shards": shards,
        "download_bytes_total": sum(
            row["bytes"] for row in repository_files),
        "download_manifest": repository_files,
        "download_manifest_sha256": hashlib.sha256(
            _canonical_bytes(repository_files)).hexdigest(),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "shard_manifest_sha256": hashlib.sha256(
            _canonical_bytes(shards)).hexdigest(),
    }


_DOCUMENTED_IDENTITY_SENTINELS = frozenset({
    "change-me", "changeme", "example", "example root", "example-handle",
    "example-root", "example-org/example-root", "none", "null", "owner/name",
    "placeholder", "replace", "tbd", "todo", "unknown", "unset",
    "your-hf-handle", "your hf handle", "your_hf_handle",
    "your_handle/replace",
})


def _identity_is_placeholder(value: Any) -> bool:
    """Reject recipe substitutions and conventional unknown-value sentinels."""
    if not isinstance(value, str) or not value or value != value.strip():
        return True
    folded = value.casefold()
    return (
        folded in _DOCUMENTED_IDENTITY_SENTINELS
        or folded.startswith("$")
        or (folded.startswith("<") and folded.endswith(">"))
        or "placeholder" in folded
        or folded.startswith("example-")
        or folded.startswith("example/")
        or folded.startswith("example-org/")
    )


def _apply_runpod_defaults(args) -> None:
    """Fill every flag whose only admissible value is derivable.

    The first safe path required the operator to type twelve flags that
    admit exactly one value.  A recipe of 37 tokens is not a safety
    property; it is a transcription test.  Each default here is the value
    the profile would otherwise refuse anything but, or is read from
    another flag the operator already had to give.
    """
    if getattr(args, "region", None) is None:
        args.region = "secure"
    if getattr(args, "on_preempt", None) is None:
        args.on_preempt = "fail"
    if getattr(args, "role", None) == "root":
        if not getattr(args, "dataset_name", None):
            args.dataset_name = getattr(args, "dataset_id", None)
        if not getattr(args, "dataset_repository", None):
            if getattr(args, "publish_root_to", None):
                args.dataset_repository = args.publish_root_to
            elif (getattr(args, "measurer", None)
                    and getattr(args, "dataset_id", None)):
                # Unpublished: the canonical identity is the measurer's own
                # namespace under the dataset id.  Publishing later with a
                # different repo needs --dataset-repository at capture time.
                args.dataset_repository = "%s/%s" % (
                    args.measurer, args.dataset_id)
    if not getattr(args, "hf_download_token_file", None):
        candidate = getattr(args, "hf_token_file", None)
        if candidate and Path(candidate).expanduser().is_file():
            args.hf_download_token_file = candidate


def _campaign_ledger_requested(args) -> bool:
    return bool(getattr(args, "campaign_ledger", None))


def _runpod_forbidden(args) -> List[str]:
    _apply_runpod_defaults(args)
    forbidden = []
    checks = (
        ("spot/interruptible offer", bool(args.spot)),
        ("--fs-id", getattr(args, "fs_id", None) is not None),
        ("--keep-fs", bool(getattr(args, "keep_fs", False))),
        ("--hold-on-failure", bool(getattr(args, "hold_on_failure", False))),
        ("--i-accept-leak-risk", bool(getattr(args, "i_accept_leak_risk", False))),
        ("--allow-unpublished-root", bool(getattr(args, "allow_unpublished_root", False))),
        ("--race (not wired on the RunPod path yet; see docs/RACE-MODE.md)",
         bool(getattr(args, "race", False))),
        ("--preview-of (not wired on the RunPod path yet)",
         bool(getattr(args, "preview_of", None))),
        ("--skip-registry-check", bool(getattr(args, "skip_registry_check", False))),
        ("--force", bool(getattr(args, "force", False))),
        ("--accept-measured-revision", bool(getattr(args, "accept_measured_revision", False))),
        ("--no-preflight-bench", bool(getattr(args, "no_preflight_bench", False))),
        ("--keep-student-logits", bool(getattr(args, "keep_student_logits", False))),
        ("--designated-reference", bool(getattr(args, "designated_reference", False))),
    )
    forbidden.extend(label for label, present in checks if present)
    measurer = getattr(args, "measurer", None)
    if (_identity_is_placeholder(measurer)
            or re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,94}[A-Za-z0-9])?",
                measurer or "") is None):
        forbidden.append(
            "--measurer must be an explicit non-placeholder Hub handle")
    if getattr(args, "on_preempt", None) != "fail":
        forbidden.append("--on-preempt must be fail")
    if getattr(args, "cold_runs", None) != 2:
        forbidden.append("--cold-runs must be 2")
    if getattr(args, "max_cost", None) is None:
        forbidden.append("--max-cost is required")
    decimal_rules = {
        "max_cost": True,
        "campaign_ceiling": True,
        "campaign_reserve": True,
        "campaign_reaper_margin": False,
        "runpod_container_running_tariff": False,
        "runpod_container_stopped_tariff": False,
        "runpod_pod_running_tariff": False,
        "runpod_pod_stopped_tariff": False,
        "runpod_network_tariff": False,
    }
    for name, positive in decimal_rules.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        try:
            parsed = Decimal(str(value))
        except Exception:
            forbidden.append(
                "--%s must be finite decimal" % name.replace("_", "-"))
            continue
        if (not parsed.is_finite() or parsed < 0
                or (positive and parsed <= 0)):
            forbidden.append(
                "--%s must be %s finite decimal"
                % (name.replace("_", "-"),
                   "positive" if positive else "nonnegative"))
    integer_bounds = {
        "heartbeat_timeout": 86400,
        "retrieval_delete_reserve": 604800,
        "timer_api_lag": 86400,
        "runpod_billing_wait": 86400,
    }
    for name, maximum in integer_bounds.items():
        value = getattr(args, name, None)
        if name == "retrieval_delete_reserve" and value is None:
            continue    # derived from the retrieval contract at plan time
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 or value > maximum):
            forbidden.append(
                "--%s must be in 1..%d"
                % (name.replace("_", "-"), maximum))
    # Two admissible values, both deliberate: "Paris" (the default) enforces
    # the on-pod generation probe; "" records the probe without enforcing it,
    # which is the operator's declaration that the target is an undertrained
    # proxy (the 5B Fruit fixture answers " the"). Anything else is a typo.
    if (getattr(args, "role", None) == "root"
            and getattr(args, "sanity_expect", None) not in ("Paris", "")):
        forbidden.append(
            "--sanity-expect must be Paris (enforced) or '' (recorded, not "
            "enforced; undertrained proxies only) for root")
    if getattr(args, "campaign_name", None) != "fidcloud-":
        forbidden.append(
            "--campaign-name is fixed to fidcloud- in safe RunPod mode")
    if not getattr(args, "max_runtime", None) and getattr(args, "role", None) != "root":
        forbidden.append("--max-runtime is required")
    if getattr(args, "hf_download_token_file", None) in (None, ""):
        forbidden.append(
            "--hf-download-token-file is required (or give --hf-token-file "
            "pointing at an existing owner-only read token file)")
    if _campaign_ledger_requested(args):
        # Strict campaign mode: cross-run accounting with explicit limits.
        for name in ("campaign_ceiling", "campaign_reserve",
                     "campaign_reaper_margin"):
            if getattr(args, name, None) in (None, ""):
                forbidden.append(
                    "--%s is required with --campaign-ledger"
                    % name.replace("_", "-"))
    else:
        for name in ("campaign_ceiling", "campaign_reserve",
                     "campaign_reaper_margin"):
            if getattr(args, name, None) not in (None, ""):
                forbidden.append(
                    "--%s only applies with --campaign-ledger; without a "
                    "campaign, --max-cost is the whole budget"
                    % name.replace("_", "-"))
        if getattr(args, "runpod_safety_proof", None):
            forbidden.append(
                "--runpod-safety-proof binds to a campaign ledger; pass "
                "--campaign-ledger (and its limits) with it")
        if getattr(args, "campaign_width", 1) != 1:
            forbidden.append(
                "--campaign-width 2 needs --campaign-ledger")
    if (getattr(args, "campaign_width", 1) == 2
            and not getattr(args, "width_two_root_archive", None)):
        forbidden.append("--width-two-root-archive is required for width 2")
    if getattr(args, "campaign_width", 1) not in (1, 2):
        forbidden.append("--campaign-width must be 1 or 2")
    expected_schedule = (
        "layer-outer" if getattr(args, "role", None) == "root"
        else "window-major")
    if getattr(args, "schedule", None) != expected_schedule:
        forbidden.append("--schedule must be %s" % expected_schedule)
    if getattr(args, "lane", None) != "streaming":
        forbidden.append("--lane must be streaming")
    if getattr(args, "capture_device", None) != "cuda":
        forbidden.append("--capture-device must be cuda")
    if getattr(args, "reduce_order", None) != "fp32":
        forbidden.append("--reduce-order must be fp32")
    if getattr(args, "role", None) == "root":
        for name in ("dataset_id", "dataset_name", "dataset_repository"):
            if _identity_is_placeholder(getattr(args, name, None)):
                forbidden.append(
                    "--%s must be explicit and non-placeholder"
                    % name.replace("_", "-"))
        dataset_repository = getattr(args, "dataset_repository", None)
        publish_root_to = getattr(args, "publish_root_to", None)
        if (not dataset_repository
                or re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9._-]*/"
                    r"[A-Za-z0-9][A-Za-z0-9._-]*",
                    dataset_repository) is None):
            forbidden.append(
                "--dataset-repository must be an exact owner/name repo id")
        if publish_root_to and publish_root_to != dataset_repository:
            forbidden.append(
                "--publish-root-to must equal --dataset-repository")
        if getattr(args, "replay_device", None) != "numpy":
            forbidden.append("--replay-device must be numpy")
        if getattr(args, "replay_dtype", None) != "float32":
            forbidden.append("--replay-dtype must be float32")
        if getattr(args, "replay_vocab_chunk", None) != 8192:
            forbidden.append("--replay-vocab-chunk must be 8192")
        if getattr(args, "form", None) != "hidden":
            forbidden.append("--form must be hidden")
    if getattr(args, "region", None) != "secure":
        forbidden.append("--region must be secure (the default)")
    return forbidden


def _exact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# RunPod's published per-GB-month storage rates on the date below.  They are
# the DEFAULTS of the five --runpod-*-tariff flags and change rarely; the
# GPU rate itself is always the live offer.  A previous version refused every
# paid run unless the flags equalled these literals AND the wall clock fell
# inside a seven-day window ending 2026-09-07 -- a time bomb that would have
# refused all spend four days after the last edit.  Age is now advisory.
RUNPOD_STORAGE_TARIFF_PINNED_AT = "2026-08-31T00:00:00Z"
RUNPOD_STORAGE_TARIFF_STALE_AFTER_DAYS = 30


def _runpod_quote(args, chosen, target, profile, timing, storage_gb,
                  container_disk_gb, workload_seconds, result_archive_contract,
                  warnings: Optional[List[str]] = None,
                  deferred: Optional[List["Refusal"]] = None):
    from fidelity.campaign import CostQuote, RUNPOD_TARIFF_SOURCE

    tariffs = {}
    for name in ("runpod_container_running_tariff",
                 "runpod_container_stopped_tariff",
                 "runpod_pod_running_tariff", "runpod_pod_stopped_tariff",
                 "runpod_network_tariff"):
        try:
            tariffs[name] = Decimal(str(getattr(args, name)))
        except (InvalidOperation, ValueError):
            raise Refusal("--%s is not a decimal" % name.replace("_", "-"), [])
        if not tariffs[name].is_finite() or tariffs[name] < 0:
            raise Refusal(
                "--%s must be a non-negative decimal"
                % name.replace("_", "-"), [])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        tariff_effective = datetime.strptime(
            args.tariff_effective_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise Refusal(
            "--tariff-effective-at must be an exact UTC timestamp "
            "like 2026-08-31T00:00:00Z", [])
    tariff_age_days = (now - tariff_effective).days
    if (warnings is not None
            and tariff_age_days > RUNPOD_STORAGE_TARIFF_STALE_AFTER_DAYS):
        warnings.append(
            "storage tariffs were pinned %d days ago (%s); verify them "
            "against %s and pass --runpod-*-tariff / --tariff-effective-at "
            "if RunPod changed its per-GB-month prices. The GPU rate is live "
            "and --max-cost still caps the whole run."
            % (tariff_age_days, args.tariff_effective_at,
               RUNPOD_TARIFF_SOURCE))
    quote_until = now + timedelta(minutes=5)
    reserve = Decimal(str(args.retrieval_delete_reserve))
    lag = Decimal(str(args.timer_api_lag))
    workload = Decimal(str(workload_seconds))
    provider_deadline = workload + reserve
    rate = Decimal(str(chosen["price_per_gpu_hour"])) * Decimal(chosen["gpus"])
    quote_fields = dict(
        reserved_compute_usd_per_hour=rate,
        live_compute_usd_per_hour=rate,
        container_disk_size_gb=Decimal(int(container_disk_gb)),
        container_disk_running_usd_per_gb_month=tariffs[
            "runpod_container_running_tariff"],
        container_disk_stopped_usd_per_gb_month=tariffs[
            "runpod_container_stopped_tariff"],
        pod_disk_size_gb=Decimal(int(storage_gb)),
        pod_disk_running_usd_per_gb_month=tariffs[
            "runpod_pod_running_tariff"],
        pod_disk_stopped_usd_per_gb_month=tariffs[
            "runpod_pod_stopped_tariff"],
        network_volume_size_gb=Decimal(0),
        network_volume_usd_per_gb_month=tariffs["runpod_network_tariff"],
        storage_month_hours=Decimal(672),
        network_billing_increment_seconds=Decimal(3600),
        tariff_source=RUNPOD_TARIFF_SOURCE,
        tariff_effective_at=args.tariff_effective_at,
        quoted_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        valid_until=quote_until.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target="%s@%s" % (target.repo_id, target.revision),
        profile=profile,
        timing_kind=("named-conservative-bound" if args.role == "root"
                     else "exact-target-profile"),
        timing_evidence=hashlib.sha256(_canonical_bytes(timing)).hexdigest(),
        workload_deadline_seconds=workload,
        provider_termination_deadline_seconds=provider_deadline,
        retrieval_delete_reserve_seconds=reserve,
        timer_api_lag_seconds=lag)
    # Price the run with a cap that cannot bind, so a refusal can say what
    # the run actually costs and which flag moves it, instead of CostQuote's
    # bare "hard_cap_usd is below the all-in calculated maximum".
    unbounded = Decimal(10) ** 9
    priced = CostQuote(hard_cap_usd=unbounded, **quote_fields)
    calculated = priced.calculated_maximum_usd()
    cap = Decimal(str(args.max_cost))
    if calculated > cap:
        hours = (workload + reserve) / Decimal(3600)
        refusal = Refusal(
            "all-in maximum $%s exceeds --max-cost %s"
            % (calculated.quantize(Decimal("0.01")), args.max_cost),
            ["GPU $%s/h for %s h (workload %s s + retrieval/delete reserve "
             "%s s) plus storage for that window"
             % (rate, hours.quantize(Decimal("0.01")), workload, reserve),
             "raise --max-cost to at least %s; the reserve is already the "
             "retrieval contract's minimum unless you raised it"
             % calculated.quantize(Decimal("0.01"), rounding=ROUND_CEILING)])
        if deferred is None:
            raise refusal
        # Planning accumulates the arithmetic findings (S1-2): report this
        # one beside the others and keep going with the unbounded price so
        # the rest of the plan can still be computed. Nothing is created
        # while a deferred refusal is pending.
        deferred.append(refusal)
        return priced
    return CostQuote(hard_cap_usd=cap, **quote_fields)


def _defer_refusal(plan_data: Dict[str, Any], refusal: "Refusal") -> None:
    """Record an arithmetic pre-spend refusal instead of raising it.

    The timing bound, the cost cap, the publication destination and the
    lease scope are all computed from the same plan state and none has a
    side effect, so a human should see every one of them in one dry-run
    rather than one per 20-second round trip (AGENTS.md: accumulate
    findings, do not stop at the first and hide the rest)."""
    plan_data.setdefault("_deferred_refusals", []).append(refusal)


def _raise_deferred_refusals(plan_data: Dict[str, Any]) -> None:
    deferred = plan_data.get("_deferred_refusals") or []
    if not deferred:
        return
    if len(deferred) == 1:
        raise deferred[0]
    lines: List[str] = []
    for index, refusal in enumerate(deferred, 1):
        lines.append("[%d] %s" % (index, refusal.reason))
        lines.extend("    " + line for line in refusal.advice)
    raise Refusal(
        "%d pre-spend findings; every one must be settled before a pod is "
        "created" % len(deferred), lines)


def _write_verified_panel_archive(panel_root: Path, destination: Path,
                                  tokenizer_root: Optional[str]):
    from fidelity.panel import resolve_panel, write_panel_archive
    temporary = None
    selected_root = Path(tokenizer_root).resolve() if tokenizer_root else None
    if selected_root is None:
        preliminary = resolve_panel(panel_root, role="final").to_dict()
        tokenizer = preliminary.get("tokenizer") or {}
        if tokenizer.get("files_verified") is not True:
            repository, revision = tokenizer.get("repository"), tokenizer.get("revision")
            files = tokenizer.get("files")
            if (not repository or re.fullmatch(r"[0-9a-f]{40}",
                                               str(revision or "")) is None
                    or not isinstance(files, list) or not files):
                raise Refusal("panel tokenizer receipt cannot be prefetched exactly", [])
            temporary = tempfile.TemporaryDirectory(
                prefix="fidelity-tokenizer-receipt-")
            selected_root = Path(temporary.name)
            for row in files:
                pure = PurePosixPath(row.get("name", ""))
                if (pure.is_absolute()


                        or any(part in ("", ".", "..") for part in pure.parts)):
                    raise Refusal("tokenizer receipt contains unsafe file path", [])
                body = fetch_file(repository, str(pure), revision=revision)
                output = selected_root.joinpath(*pure.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(body)
    archive = write_panel_archive(
        panel_root, destination, role="final", tokenizer_root=selected_root)
    if (archive["binding"].get("tokenizer") or {}).get("files_verified") is not True:
        raise Refusal("panel tokenizer receipt files are not verified before spend", [])
    return archive, temporary
def _prefetch_quant_panel(meta: RepoMeta) -> Dict[str, Any]:
    """Fetch and validate only the pinned final25 token/mask panel bytes."""
    import stage_panel_paths

    receipt_candidates = [
        path for path, _size in meta.files
        if path.endswith("token-panel-receipt.json")]
    if len(receipt_candidates) != 1:
        raise Refusal(
            "pinned panel tree lacks one exact token-panel receipt", [])
    temporary = tempfile.TemporaryDirectory(
        prefix="fidelity-quant-panel-preflight-")
    _RUNPOD_TEMP_HOLDS.append(temporary)
    root = Path(temporary.name)
    receipt_rel = receipt_candidates[0]
    receipt_path = root / receipt_rel
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    receipt_path.write_bytes(fetch_file(
        meta.repo_id, receipt_rel, repo_type="dataset",
        revision=meta.revision, timeout=120))
    artifacts = stage_panel_paths.validate_receipt(receipt_path)
    rows = []
    for artifact in artifacts:
        relative = "calibration/panel-v1/%s" % "/".join(
            artifact.relative_parts)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = fetch_file(
            meta.repo_id, relative, repo_type="dataset",
            revision=meta.revision, timeout=120)
        if (len(data) != artifact.bytes
                or hashlib.sha256(data).hexdigest() != artifact.sha256):
            raise Refusal("pinned panel byte drift: %s" % relative, [])
        destination.write_bytes(data)
        rows.append({
            "path": relative, "bytes": len(data),
            "sha256": artifact.sha256})
    check = stage_panel_paths.stage_panel(
        root, receipt_path, check_only=True,
        destination_prefix=root / "destination/calibration/panel-v1",
        destination_anchor=root)
    if check.get("artifacts") != 667 or check.get("unresolved") != 0:
        raise Refusal("pinned final25 panel validation is incomplete", [])
    rows.append({
        "path": receipt_rel, "bytes": receipt_path.stat().st_size,
        "sha256": sha256_file(str(receipt_path))})
    rows.sort(key=lambda row: row["path"])
    return {
        "local_root": str(root), "files": rows,
        "manifest_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
        "receipt_sha256": stage_panel_paths.PINNED_RECEIPT_SHA256,
        "token_panel_sha256": stage_panel_paths.PINNED_PANEL_SHA256,
        "contexts": stage_panel_paths.PINNED_FINAL_WINDOWS,
        "prediction_positions":
            stage_panel_paths.PINNED_FINAL_PREDICTION_POSITIONS,
    }


def _validate_quant_panel_descriptor(
        descriptor, meta: RepoMeta, panel_validation: Dict[str, Any],
        reference_manifest: Dict[str, Any]) -> None:
    """Bind every descriptor field to the fetched, sealed final25 evidence."""
    expected = {
        "panel_ref": "panel--glm53.brandonmusic.final25",
        "repo_id": reference_manifest["repo_id"],
        "revision": reference_manifest["revision"],
        "contexts": reference_manifest["contexts"],
        "positions_per_context":
            reference_manifest["positions_per_context"],
        "scored_positions": reference_manifest["prediction_positions"],
        "roles": "final",
        "panel_token_sha256": panel_validation["token_panel_sha256"],
        "panel_receipt_sha256":
            reference_manifest["token_panel_receipt_sha256"],
        "reference_ref": reference_manifest["reference_ref"],
        "teacher_receipt_sha256":
            reference_manifest["capture_receipt_sha256"],
        "teacher_backend_identity_sha256":
            reference_manifest["backend_identity_sha256"],
    }
    actual = {
        name: getattr(descriptor, name)
        for name in expected
    }
    drift = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected if actual[name] != expected[name]
    }
    includes = list(descriptor.include)
    expected_includes = list(reference_manifest["include"])
    if (len(includes) != len(set(includes))
            or sorted(includes) != sorted(expected_includes)):
        drift["include"] = {
            "expected": expected_includes,
            "actual": includes,
        }
    fetched_bytes = meta.bytes_matching(includes)
    if fetched_bytes != reference_manifest["included_repo_bytes"]:
        drift["fetch_bytes"] = {
            "expected": reference_manifest["included_repo_bytes"],
            "actual": fetched_bytes,
        }
    if panel_validation["receipt_sha256"] != (
            reference_manifest["token_panel_receipt_sha256"]):
        drift["validated_panel_receipt"] = {
            "expected": reference_manifest["token_panel_receipt_sha256"],
            "actual": panel_validation["receipt_sha256"],
        }
    if drift:
        raise Refusal(
            "panel descriptor differs from the fetched sealed final25 evidence",
            [json.dumps(drift, sort_keys=True)])


def _resolve_authored_quant_scope(args, target, con):
    """Return exact registry-authored scope and its immutable source binding."""
    from fidelity.registry_client import load as load_registry
    expected = {
        ("malaiwah/GLM-5.3-Flash-TR3-6bpw",
         "9ab94105a71708a19c6d960d24b4aa6d459f5623"):
            "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
    }
    artifact_id = expected.get((target.repo_id, target.revision))
    if artifact_id is None:
        raise Refusal("target has no authored exact scope source", [])
    try:
        registry = load_registry(args.registry, purpose="check", con=con)
        artifact = registry.collections["artifacts"][artifact_id]
    except Exception as exc:
        raise Refusal("authored exact scope is unavailable: %s" % exc, [])
    hf = artifact.get("huggingface") or {}
    if ((hf.get("repository") or "").lower() != target.repo_id.lower()
            or artifact.get("id") != artifact_id):
        raise Refusal("authored scope source targets different bytes", [])
    scope = artifact.get("scope")
    digest = artifact.get("scope_digest")
    if not isinstance(scope, dict) or not scope or not isinstance(digest, str):
        raise Refusal("authored artifact scope is incomplete", [])
    supplied = read_json(args.scope_json) if args.scope_json else None
    if supplied is not None and _canonical_bytes(supplied) != _canonical_bytes(scope):
        raise Refusal("--scope-json differs from exact authored artifact scope",
                      [])
    source_sha = hashlib.sha256(_canonical_bytes(artifact)).hexdigest()
    return json.loads(_canonical_bytes(scope).decode("utf-8")), {
        "artifact_id": artifact_id, "scope_digest": digest,
        "source_sha256": source_sha,
        "registry_snapshot": registry.snapshot_id,
    }


_FULL_GLM53_ROOT = (
    "root", "zai-org/GLM-5.3-BF16",
    "304b8051cfb2b260b61ce0cbe330e02a98e73639")
_FULL_GLM53_LICENSE = {
    "source_path": "LICENSE",
    "dataset_path": "LICENSE",
    "bytes": 4263,
    "sha256": "96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c",
}




def _root_dataset_license_contract(target: RepoMeta) -> Dict[str, Any]:
    """Bind copied native weights to the exact source-license bytes."""
    if ("root", target.repo_id, target.revision) != _FULL_GLM53_ROOT:
        return {"dataset_license": "mit", "weights_license": None}
    try:
        raw = fetch_file(
            target.repo_id, _FULL_GLM53_LICENSE["source_path"],
            revision=target.revision)
    except HFError as exc:
        raise Refusal(
            "full GLM-5.3 source license is unavailable: %s" % exc, [])
    observed = {
        "source_path": _FULL_GLM53_LICENSE["source_path"],
        "dataset_path": _FULL_GLM53_LICENSE["dataset_path"],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    sizes = dict(target.files)
    if (observed != _FULL_GLM53_LICENSE
            or sizes.get(observed["source_path"]) != observed["bytes"]):
        raise Refusal(
            "full GLM-5.3 source-license identity differs from the authored pin",
            [])
    return {"dataset_license": "other", "weights_license": observed}


def _open_existing_runpod_campaign(args, provider_account_id: str):
    """Open the explicit campaign ledger; strict mode never creates one."""
    from fidelity.campaign import CampaignLedger, CampaignLedgerError
    campaign_path = str(Path(args.campaign_ledger).resolve())
    if not Path(campaign_path).is_file():
        raise Refusal(
            "--campaign-ledger names a file that does not exist; strict "
            "campaign mode opens an existing ledger. Create one with "
            "`measure-cloud drill --provider runpod --campaign-ledger ...` "
            "or omit --campaign-ledger for a per-run ledger", [])
    try:
        ledger = CampaignLedger(
            campaign_path, "runpod", provider_account_id)
        snapshot = ledger.snapshot()
    except (CampaignLedgerError, OSError, ValueError) as exc:
        raise Refusal(
            "campaign ledger is unavailable: %s"
            % redact(str(exc)), [])
    expected = (
        Decimal(str(args.campaign_ceiling)),
        Decimal(str(args.campaign_reserve)),
        Decimal(str(args.campaign_reaper_margin)),
        2, "USD", "runpod", provider_account_id)
    actual = (
        Decimal(snapshot["hard_ceiling_usd"]),
        Decimal(snapshot["reserve_floor_usd"]),
        Decimal(snapshot["cleanup_reaper_margin_usd"]),
        snapshot["max_concurrent_attempts"], snapshot["currency"],
        snapshot["provider"], snapshot["provider_account_id"])
    if actual != expected:
        raise Refusal(
            "existing campaign ledger identity differs from this measurement",
            [])
    return campaign_path, ledger


def _auto_campaign_ledger_path(args, job_id_full: str,
                               attempt: Optional[str] = None) -> str:
    """Per-attempt ledger beside the lease directory.

    The lease records only the ledger's leaf name and `campaign_coordinates`
    joins it to the lease root's PARENT, so the file must live there, not in
    an independently configurable state directory.  The attempt id is part
    of the name: one ledger per paid attempt, so a retry of the same job
    starts clean instead of inheriting the previous attempt's liability.
    """
    from fidelity.cloudlease import DEFAULT_LEASE_DIR
    lease_dir = Path(getattr(args, "lease_dir", None) or DEFAULT_LEASE_DIR)
    leaf = "auto-%s%s.json" % (
        job_id_full[:16], ("-" + attempt) if attempt else "")
    return str((lease_dir.resolve().parent / leaf))


def _auto_campaign_limits(args) -> Dict[str, Any]:
    """Ceiling = --max-cost, nothing reserved, nothing held for the reaper.

    A per-run ledger exists so the reaper can release liability after a
    controller crash and so the receipt carries a durable accounting record;
    it does not span runs.  Cross-run accounting is what --campaign-ledger
    opts into.  Foreign resources are tolerated: a pod someone else runs on
    the same account is not this measurement's liability and must not
    refuse it.
    """
    return {
        "hard_ceiling_usd": Decimal(str(args.max_cost)),
        "reserve_floor_usd": Decimal(0),
        "cleanup_reaper_margin_usd": Decimal(0),
        "max_concurrent_attempts": 1,
        "foreign_resources_policy": "tolerate",
    }


def _create_auto_campaign_ledger(args, provider_account_id: str,
                                 job_id_full: str, attempt: str):
    """Create the per-attempt ledger for execution."""
    from fidelity.campaign import CampaignLedger, CampaignLedgerError
    limits = _auto_campaign_limits(args)
    campaign_path = _auto_campaign_ledger_path(args, job_id_full, attempt)
    try:
        ledger = CampaignLedger.create(
            campaign_path, limits["hard_ceiling_usd"],
            limits["reserve_floor_usd"], limits["cleanup_reaper_margin_usd"],
            max_concurrent_attempts=limits["max_concurrent_attempts"],
            provider="runpod", provider_account_id=provider_account_id,
            foreign_resources_policy=limits["foreign_resources_policy"])
    except (CampaignLedgerError, OSError, ValueError) as exc:
        raise Refusal(
            "per-attempt campaign ledger could not be created at %s: %s"
            % (campaign_path, redact(str(exc))),
            ["the parent of --lease-dir must be an owner-only directory "
             "(mode 0700) that you own"])
    return campaign_path, ledger


# RunPod's gpuTypeId for the normalized --gpu spellings this controller has
# rented or benchmarked (reports/provider-bench).  Extend here, not in a
# per-target branch.
_RUNPOD_GPU_IDS = {
    "L4": "NVIDIA L4",
    "A40": "NVIDIA A40",
    "L40S": "NVIDIA L40S",
    "A100": "NVIDIA A100-SXM4-80GB",
    "A100PCIE": "NVIDIA A100 80GB PCIe",
    "H100": "NVIDIA H100 80GB HBM3",
    "H100NVL": "NVIDIA H100 NVL",
    "H200": "NVIDIA H200",
    "B200": "NVIDIA B200",
}

# Host capacity for a root capture.  The layer-outer engine
# (docs/LAYER-OUTER.md) keeps ONE decoder layer plus the embeddings, head and
# norms resident; everything else is an mmap the OS may evict.  So host
# memory scales with the largest layer, not the checkpoint: assume a large
# checkpoint has at least 40 layers, hold three layers' worth for torch,
# hidden states and page-cache headroom, plus 8 GB for the resident set.
# vCPU feeds hf_transfer and the numpy replay; one per 8 GB, 8..24.
#
# The previous literals for both GLM-5.3 roots were (28 vCPU, 300 GB),
# copied from the K6 quant lane that keeps a whole model resident on a
# JarvisLabs host.  On 2026-09-03 no secure H200 host offered that pair and
# twelve consecutive creates were refused SUPPLY_CONSTRAINT while H200
# stock itself read Medium; (16, 128) with the same 1.9 TB disk read Medium.
# Fruit's pair is measured on an L4 and kept.  All of it is overridable.
_ROOT_HOST_CAPACITY_TABLE = {
    "malaiwah/GLM-5.2-SIQ-Fruit-bf16": (4, 32),
}
_ROOT_HOST_ASSUMED_MIN_LAYERS = 40
_ROOT_HOST_LAYERS_HELD = 3
_ROOT_HOST_RESIDENT_SET_GB = 8
_ROOT_HOST_MIN_MEMORY_GB = 64
_ROOT_HOST_MIN_VCPU = 8
_ROOT_HOST_MAX_VCPU = 24


def _root_host_capacity(args, target, identity) -> Tuple[int, int, str]:
    override_vcpu = getattr(args, "min_vcpu", None)
    override_memory = getattr(args, "min_memory_gb", None)
    if override_vcpu is not None and override_memory is not None:
        return int(override_vcpu), int(override_memory), "operator-override"
    authored = _ROOT_HOST_CAPACITY_TABLE.get(target.repo_id)
    if authored is not None:
        vcpu, memory = authored
        basis = "measured-host-capacity"
    else:
        layer_gb = -(-int(identity["model_bytes"])
                     // (_ROOT_HOST_ASSUMED_MIN_LAYERS * 10 ** 9))
        memory = _ROOT_HOST_RESIDENT_SET_GB + _ROOT_HOST_LAYERS_HELD * layer_gb
        memory = max(_ROOT_HOST_MIN_MEMORY_GB, -(-memory // 32) * 32)
        vcpu = max(_ROOT_HOST_MIN_VCPU,
                   min(_ROOT_HOST_MAX_VCPU, memory // 8))
        basis = "derived-from-layer-residency"
    if override_vcpu is not None:
        vcpu, basis = int(override_vcpu), "operator-override"
    if override_memory is not None:
        memory, basis = int(override_memory), "operator-override"
    return vcpu, memory, basis


def _root_gpu_choice(args, target, *, form: str) -> str:
    """The GPU the timing table names for this target, else --gpu."""
    from fidelity.engines import _load_engine_config
    evidenced = sorted({
        str(row.get("gpu"))
        for row in (_load_engine_config(None).get("root_timing_profiles") or [])
        if row.get("target_repo") == target.repo_id
        and row.get("target_revision") == target.revision
        and row.get("form") == form
        and isinstance(row.get("gpu"), str)})
    requested = (args.gpu or "").upper().replace("-", "").replace(" ", "")
    if requested:
        return next(
            (gpu for gpu in evidenced
             if gpu.upper().replace("-", "") == requested),
            args.gpu)
    if len(evidenced) == 1:
        return evidenced[0]
    if evidenced:
        raise Refusal(
            "timing evidence exists for more than one GPU (%s); pass --gpu"
            % ", ".join(evidenced), [])
    raise Refusal(
        "no authored timing evidence for %s@%s; pass --gpu (for example "
        "--gpu H200) and --max-runtime becomes the workload bound"
        % (target.repo_id, target.revision[:12]), [])


def _operator_bound_root_timing(args, target, gpu: str,
                                identity: Dict[str, Any]) -> Dict[str, Any]:
    """A timing record whose only evidence is the operator's --max-runtime."""
    hours = Decimal(parse_duration(args.max_runtime)) / Decimal(3600)
    return {
        "target_repo": target.repo_id,
        "target_revision": target.revision,
        "gpu": gpu, "form": args.form,
        "schedule": "two-fresh-process-qualification",
        # Kept as a string so the planner's Decimal(str(...)) * 3600 is
        # exactly the operator's seconds; a float overshot 3h20m by 6e-13
        # and refused its own bound.
        "conservative_upper_hours": str(hours),
        "resource_admission": {
            "required": True,
            "mode": "controller_explicit_safe_resources"},
        "evidence": {
            "source": "operator --max-runtime",
            "max_runtime": args.max_runtime,
            "derivation": (
                "No authored timing row exists for this target and GPU. "
                "The operator's --max-runtime is the workload bound; the "
                "reap deadline, on-pod watchdog and cost quote derive from "
                "it and the run is torn down at that deadline."),
        },
        "model_identity": {
            "model_bytes": identity["model_bytes"],
            "config_sha256": identity["config_sha256"],
            "index_sha256": identity["index_sha256"],
        },
    }


def _reaper_health_remedy(health: Dict[str, Any], args) -> List[str]:
    """Name what is wrong with the reaper and the exact command that fixes it.

    The reaper is the only thing that destroys a pod after this controller
    dies, so it is a hard gate -- but a hard gate whose refusal says only
    "not healthy" costs an operator a read of three thousand lines.
    """
    install = (
        "measure-cloud reaper --provider runpod --install"
        + (" --reaper-state-dir %s" % args.reaper_state_dir
           if args.reaper_state_dir else "")
        + (" --lease-dir %s" % args.lease_dir if args.lease_dir else ""))
    advice = []
    timer = health.get("timer") or {}
    if not timer.get("is-enabled") or not timer.get("is-active"):
        advice.append(
            "the user systemd timer is not enabled/active; run: %s" % install)
    linger = health.get("user_manager_persistence") or {}
    if not linger.get("ok"):
        advice.append(
            "your user manager does not survive logout; run: "
            "loginctl enable-linger $USER   (then reinstall: %s)" % install)
    service = health.get("service_last_result") or {}
    if not service.get("ok"):
        advice.append(
            "the last reaper sweep failed (%s); inspect: journalctl --user "
            "-u fidelity-cloud-reaper@runpod.service -n 50"
            % (service.get("error") or service.get("result") or "unknown"))
    if not health.get("control_ok"):
        advice.append(
            "(or it predates control v4); reinstall: %s" % install)
    age = health.get("stamp_age_seconds")
    if health.get("stamp") is None:
        advice.append(
            "no health stamp yet; the timer writes one on its first "
            "successful sweep -- run: measure-cloud reaper --provider runpod "
            "--sweep")
    elif isinstance(age, (int, float)) and age > 900:
        advice.append(
            "the last healthy sweep is %d s old (limit 900); run: "
            "measure-cloud reaper --provider runpod --sweep" % int(age))
    if not advice:
        advice.append("reinstall: %s" % install)
    return advice


def _root_workload_bound(timing: Dict[str, Any], *, storage_layout: str,
                         captures: int):
    """Seconds the workload may take, from the row's measured components.

    A row that carries evidence.components_seconds (fetch, setup, one cold
    run per storage layout, verify+compare+qualify, margin_factor) yields a
    bound shaped like the run that is about to happen: a resumed root
    captures once, a fresh root twice, and a container-disk pod streams
    weights ~10x faster than the pod volume. A row without components keeps
    its single conservative_upper_hours. The derivation is recorded in the
    plan; the 3.5 h bound that expired mid-run on 2026-09-03 was a number
    with no shape.
    """
    components = (timing.get("evidence") or {}).get("components_seconds")
    hours = Decimal(str(timing["conservative_upper_hours"]))
    if not isinstance(components, dict):
        return hours * 3600, {
            "basis": "conservative_upper_hours", "hours": str(hours)}
    try:
        cold_run = Decimal(str(components["cold_run"][storage_layout]))
        fetch = Decimal(str(components["fetch"]))
        setup = Decimal(str(components["setup"]))
        fixed = Decimal(str(components["verify_compare_qualify"]))
        margin = Decimal(str(components["margin_factor"]))
    except (KeyError, TypeError, InvalidOperation) as exc:
        raise Refusal(
            "timing row components_seconds lack an entry for storage layout "
            "%r (%s); author it from a measurement or use a layout the row "
            "measured" % (storage_layout, exc), [])
    if any(value <= 0 for value in (cold_run, fetch, setup, fixed)) or margin < 1:
        raise Refusal("timing row components_seconds are not positive", [])
    seconds = ((fetch + setup + cold_run * captures + fixed) * margin
               ).to_integral_value(rounding=ROUND_CEILING)
    return seconds, {
        "basis": "components_seconds", "storage_layout": storage_layout,
        "captures": captures, "fetch": str(fetch), "setup": str(setup),
        "cold_run": str(cold_run), "verify_compare_qualify": str(fixed),
        "margin_factor": str(margin), "seconds": str(seconds),
        "conservative_upper_hours_row": str(hours),
    }

DERIVED_ALLOWLIST_REMOTE = "inputs/allowlist.json"


def _derive_index_census_allowlist(target: RepoMeta, identity: Dict[str, Any],
                                   plan_data: Dict[str, Any],
                                   con: Console) -> Optional[Dict[str, Any]]:
    """Bind the index census of the never-built block when nothing is authored.

    The method that made every GLM-5.3 allowlist, run by its committed tool
    (`engines/tools/index_census_allowlist.py`) on the two files the planner
    already hashed into the identity: every index key at or past the declared
    decoder count. The file rides to the pod as `inputs/allowlist.json` and is
    bound into job.json by both digests exactly as an authored one; the pod's
    guard is unchanged (the loader's unexpected set must equal the list). The
    `_ALLOWLISTS` table in runpodsafety stays the attestation that upgrades
    the gate's provenance from `derived_from_index` to `authored`. None when
    the index has no key past the boundary (nothing to allowlist).
    """
    scratch = tempfile.TemporaryDirectory(prefix="fidelity-allowlist-plan-")
    _RUNPOD_TEMP_HOLDS.append(scratch)
    root = Path(scratch.name)
    try:
        config_raw = fetch_file(target.repo_id, "config.json", revision=target.revision)
        index_raw = fetch_file(target.repo_id, "model.safetensors.index.json",
                               revision=target.revision)
    except HFError as exc:
        raise Refusal("allowlist derivation cannot read config/index: %s"
                      % redact(str(exc)), [])
    if (hashlib.sha256(config_raw).hexdigest() != identity["config_sha256"]
            or hashlib.sha256(index_raw).hexdigest() != identity["index_sha256"]):
        raise Refusal("config/index bytes differ from the identity hashed moments ago", [])
    (root / "config.json").write_bytes(config_raw)
    (root / "model.safetensors.index.json").write_bytes(index_raw)
    out = root / "allowlist.json"
    run = subprocess.run(
        [sys.executable, str(SUITE_ROOT / "engines/tools/index_census_allowlist.py"),
         "--repo", target.repo_id, "--revision", target.revision,
         "--index", str(root / "model.safetensors.index.json"),
         "--config", str(root / "config.json"), "--out", str(out)],
        cwd=str(SUITE_ROOT / "engines/tools"), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if run.returncode != 0:
        message = (run.stderr or run.stdout).strip()
        if "nothing to allowlist" in message:
            return None
        raise Refusal("allowlist derivation failed: %s" % redact(message[-400:]), [])
    raw = out.read_bytes()
    sidecar = json.loads((root / "allowlist.json.provenance.json").read_text("utf-8"))
    from fidelity.runpodsafety import _strict_json, canonical_bytes
    names = _strict_json(raw, "derived allowlist")
    if (not isinstance(names, list) or names != sorted(set(names))
            or hashlib.sha256(raw).hexdigest() != sidecar["artifact_sha256"]
            or sidecar["index_sha256"] != identity["index_sha256"]
            or sidecar["config_sha256"] != identity["config_sha256"]):
        raise Refusal("derived allowlist does not describe the identity it was made from", [])
    names_sha = hashlib.sha256(canonical_bytes(names)).hexdigest()
    if names_sha != sidecar["canonical_sorted_names_sha256"]:
        raise Refusal("derived allowlist canonical digest differs from its sidecar", [])
    plan_data["_derived_allowlist_local"] = str(out)
    plan_data["derived_allowlist_provenance"] = sidecar
    con.ok("allowlist derived from index",
           "%d names past layer %d (%s); artifact %s..., names %s...; register "
           "these in runpodsafety._ALLOWLISTS to attest it"
           % (sidecar["count"], sidecar["decoder_layers"], sidecar["architecture"],
              sidecar["artifact_sha256"][:16], names_sha[:16]))
    return {"path": DERIVED_ALLOWLIST_REMOTE,
            "artifact_sha256": sidecar["artifact_sha256"],
            "canonical_sorted_names_sha256": names_sha,
            "count": sidecar["count"], "decoder_layers": sidecar["decoder_layers"]}



#: `bin/stage_measure.sh`'s candidate rule: these binding files are symlinked
#: from the REFERENCE root's release on the pod (publisher metadata, no
#: tokenization effect); every other bound file (tokenizer.json,
#: tokenizer_config.json, ...) is taken from the CANDIDATE and byte-checked by
#: the capture against the root's digest.
CANDIDATE_TOKENIZER_FILES_FROM_REFERENCE = frozenset(
    {"config.json", "generation_config.json", "LICENSE", "chat_template.jinja"})


def candidate_tokenizer_files_mismatch(binding_files, fetch) -> List[Dict[str, Any]]:
    """The bound tokenization files the candidate does NOT carry byte-identically.

    `binding_files` is the panel binding's `tokenizer.files` (name, bytes,
    sha256 of the ROOT's file); `fetch(name) -> bytes | None` reads the
    candidate's file at its pinned revision (None when absent). Returns one
    row per differing or absent file, empty when the pod's byte gate will
    pass. Pure so the selftest can drive it without the Hub.

    This is the gate the 2026-09-05 RadixArk attempt lacked: the controller
    validated the binding's LABELS and rented a pod that refused on the first
    tokenizer_config.json digest (one added loader key) after fetching 465 GB.
    """
    rows: List[Dict[str, Any]] = []
    for entry in binding_files or []:
        name = str(entry.get("name"))
        if name in CANDIDATE_TOKENIZER_FILES_FROM_REFERENCE:
            continue
        want = str(entry.get("sha256"))
        raw = fetch(name)
        if raw is None:
            rows.append({"name": name, "expected_sha256": want, "observed_sha256": None,
                         "expected_bytes": entry.get("bytes"), "observed_bytes": None})
            continue
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            rows.append({"name": name, "expected_sha256": want, "observed_sha256": got,
                         "expected_bytes": entry.get("bytes"), "observed_bytes": len(raw)})
    return rows


def _refuse_candidate_tokenizer_mismatch(con: Console, target, binding: Dict[str, Any],
                                         reference_repo: str, reference_revision: str,
                                         plan_data: Dict[str, Any]) -> None:
    """Read the candidate's tokenization files NOW and refuse before any rental.

    The pod's gate is `fidelity.panel`'s: byte identity, or -- for
    tokenizer_config.json only -- loader-key equivalence against the ROOT's
    bytes (`panel.tokenizer_config_equivalent`, allowlist
    `panel.TOKENIZER_CONFIG_LOADER_KEYS`). The same rule runs here, on the
    root's file read from the reference release, so what the pod would admit
    is admitted here and what it would refuse is refused for $0.
    """
    from fidelity import panel as panel_contract

    files = (binding.get("tokenizer") or {}).get("files") or []

    def fetch(name: str):
        try:
            return fetch_file(target.repo_id, name, revision=target.revision)
        except HFError as exc:
            if re.search(r"HTTP 404\b", str(exc)):
                return None
            raise Refusal(
                "candidate tokenizer file %s could not be read from %s@%s: %s"
                % (name, target.repo_id, target.revision[:12], redact(str(exc))),
                ["Nothing was created. $0.00 spent."])

    mismatch = candidate_tokenizer_files_mismatch(files, fetch)
    checked = [str(e.get("name")) for e in files
               if str(e.get("name")) not in CANDIDATE_TOKENIZER_FILES_FROM_REFERENCE]
    equivalences: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    for row in mismatch:
        if (row["name"] != panel_contract.TOKENIZER_CONFIG_EQUIVALENCE_FILE
                or row["observed_sha256"] is None):
            refused.append(dict(row, why="byte identity required"))
            continue
        try:
            root_raw = fetch_file(reference_repo, row["name"], revision=reference_revision)
        except HFError as exc:
            raise Refusal(
                "reference root's %s could not be read from %s@%s for the loader-key "
                "equivalence test: %s" % (row["name"], reference_repo,
                                          str(reference_revision)[:12], redact(str(exc))),
                ["Nothing was created. $0.00 spent."])
        if hashlib.sha256(root_raw).hexdigest() != row["expected_sha256"]:
            raise Refusal(
                "reference root's %s at %s@%s does not carry the panel binding's digest"
                % (row["name"], reference_repo, str(reference_revision)[:12]),
                ["Nothing was created. $0.00 spent."])
        try:
            equivalences.append(panel_contract.tokenizer_config_equivalent(
                root_raw, fetch(row["name"])))
        except panel_contract.PanelError as exc:
            refused.append(dict(row, why=str(exc)))
    if refused:
        raise Refusal(
            "candidate tokenizer file(s) differ from the reference root's panel binding: %s"
            % ", ".join(r["name"] for r in refused),
            ["%s: root sha256 %s (%s bytes) vs candidate %s (%s bytes) -- %s"
             % (r["name"], r["expected_sha256"][:16], r["expected_bytes"],
                (r["observed_sha256"] or "absent")[:16], r["observed_bytes"], r["why"])
             for r in refused]
            + ["The candidate route takes tokenizer.json and tokenizer_config.json from "
               "the CANDIDATE and the capture byte-checks them against the root's "
               "digests (bin/stage_measure.sh; tokenizer_config.json may differ by the "
               "loader-only keys %s and nothing else); the pod would refuse on the first "
               "mismatch after fetching the whole checkpoint."
               % list(panel_contract.TOKENIZER_CONFIG_LOADER_KEYS),
               "Publisher-metadata files (%s) come from the reference root and are "
               "not checked here." % ", ".join(sorted(CANDIDATE_TOKENIZER_FILES_FROM_REFERENCE)),
               "Nothing was created. $0.00 spent."])
    gate_verified(plan_data, "candidate-tokenizer-files",
                  candidate=target.repo_id, candidate_revision=target.revision,
                  reference=reference_repo, reference_revision=reference_revision,
                  files_checked=checked,
                  loader_key_equivalences=[
                      {k: v for k, v in e.items() if k != "reason"} for e in equivalences])
    if equivalences:
        plan_data["warnings"].append(
            "candidate %s differs from the reference root's by loader-only keys "
            "(root %s, candidate %s; dropped from candidate: %s); admitted by "
            "fidelity.panel's exact rule and recorded on the dataset as the "
            "tokenizer_config_loader_keys_ignored disclosure"
            % (equivalences[0]["name"], equivalences[0]["root_sha256"][:16],
               equivalences[0]["candidate_sha256"][:16],
               equivalences[0]["keys_dropped_from_candidate"]))
    con.ok("candidate tokenizer files",
           "%s read from %s@%s and %s the reference root's binding"
           % (", ".join(checked), target.repo_id, target.revision[:12],
              ("byte-identical to" if not equivalences else
               "loader-key equivalent to (%s)" % ", ".join(e["name"] for e in equivalences))))


def _refuse_gguf_tokenizer_mismatch(con: Console, target, binding: Dict[str, Any],
                                    reference_repo: str, reference_revision: str,
                                    plan_data: Dict[str, Any]) -> None:
    """The GGUF candidate's EMBEDDED vocabulary against the reference root's files.

    A GGUF ships no tokenizer.json: its vocabulary is the `tokenizer.ggml.tokens`
    array (index = id) and its BPE merges. The pod runs the reference root's
    tokenizer files (the candidate stage links them; the panel is already
    tokenized), so the gate here is that the artifact's own vocabulary IS that
    vocabulary: every id the root's tokenizer.json defines carries the same
    string, the merges are identical, and the only extra ids are llama.cpp's
    [PAD<id>] fillers up to the declared vocab_size. The root's files are read
    at the revision the binding names and their digests must be the binding's.
    Read from bytes at $0 -- a build converted from another vocabulary refuses
    here, not after a 467 GB fetch.
    """
    ggs = _gguf_surface_module()
    loaded = plan_data["_gguf_loaded"]
    files = {str(e.get("name")): e for e in ((binding.get("tokenizer") or {}).get("files") or [])}
    need = [n for n in ("tokenizer.json", "tokenizer_config.json") if n in files]
    if "tokenizer.json" not in files:
        raise Refusal("the reference root's panel binding names no tokenizer.json to check the "
                      "GGUF vocabulary against", [])
    digests = {}
    for name in need:
        try:
            raw = fetch_file(reference_repo, name, revision=reference_revision)
        except HFError as exc:
            raise Refusal("reference root's %s could not be read from %s@%s: %s"
                          % (name, reference_repo, str(reference_revision)[:12], redact(str(exc))),
                          ["Nothing was created. $0.00 spent."])
        got = hashlib.sha256(raw).hexdigest()
        if got != str(files[name].get("sha256")):
            raise Refusal("reference root's %s at %s@%s does not carry the panel binding's digest"
                          % (name, reference_repo, str(reference_revision)[:12]), [])
        digests[name] = got
        if name == "tokenizer.json":
            try:
                proof = ggs.tokenizer_matches(loaded.container.kv, raw)
            except ValueError as exc:
                raise Refusal(
                    "the GGUF build's embedded vocabulary is not the reference root's: %s"
                    % redact(str(exc)),
                    ["The lane runs the root's tokenizer files over an already-tokenized panel; "
                     "that is only honest when the artifact's own token table is the same "
                     "vocabulary, id for id, with the same merges.",
                     "Nothing was created. $0.00 spent."])
    gate_verified(plan_data, "candidate-tokenizer-files",
                  candidate=target.repo_id, candidate_revision=target.revision,
                  reference=reference_repo, reference_revision=reference_revision,
                  files_checked=need, gguf_vocabulary=proof, reference_digests=digests,
                  rule="gguf embedded vocabulary equals the root's tokenizer.json by id; "
                       "the pod links the root's tokenizer files beside the build")
    con.ok("candidate tokenizer (gguf)",
           "tokenizer.ggml.tokens (%d ids, %d [PAD] fillers) and %d merges equal the reference "
           "root's tokenizer.json (%s) by id; the pod runs the root's tokenizer files"
           % (proof["tokens"], proof["pad_fillers"], proof["merges"],
              digests["tokenizer.json"][:16]))


def _plan_runpod_anonymous(
        args, con: Console, provider,
        anonymous_access: Dict[str, Any]) -> Dict[str, Any]:
    """Build the complete finalized job and every paid gate before admission."""
    from fidelity.cloudlease import (
        DEFAULT_STATE_DIR, LeaseStore, systemd_reaper_health,
        validate_lease_liability_scope, validate_unresolved_lease_scope)
    from fidelity.panel import (
        PanelError, validate_reference_manifest, validate_root_panel_binding)
    from fidelity.runpodsafety import (
        validate_current_public_root, validate_safety_proof,
        validate_unexpected_tensor_allowlist, validate_width_two_root_archive,
    )

    plan_data: Dict[str, Any] = {
        "provider": "runpod", "created": False, "gates": {},
        "would_refuse": [], "safe_runpod": True, "warnings": [],
        "_deferred_refusals": [],
    }
    plan_data["anonymous_access"] = anonymous_access
    target = repo_meta(args.model, "model", args.revision or "main")
    if target.private is not False:
        raise Refusal(
            "target repository is not anonymously public", [])
    known_k8_target = (
        "quant", "malaiwah/GLM-5.3-Flash-TR3-8bpw",
        "7199f6f1a211084c240614806f046f11a52dad64")
    if (args.role, target.repo_id, target.revision) == known_k8_target:
        refusal_engine = load_engines().get(args.lane)
        if refusal_engine is None:
            raise Refusal("K8 refusal evidence has no configured lane", [])
        try:
            require_supported_profile(
                refusal_engine, surface="tr3-published", bits=8.0)
        except EngineProfileRefused as exc:
            raise Refusal(str(exc), [])
        raise Refusal(
            "K8 is refused before spend because its pinned target has no "
            "sealed surface-to-measurement verdict bridge", [])
    # Any public model with an exact revision may be planned.  Identity is
    # still proven from bytes below (surface sniff, unquantized root,
    # pinned license where one is authored); what is gone is the hardcoded
    # four-tuple allowlist that made every new model a source edit.
    from fidelity.registry_client import front_gate
    registry_gate = None
    if args.role == "quant":
        registry_gate = front_gate(
            repo=args.model, revision=target.revision,
            path_hint=getattr(args, "path", None), source=args.registry,
            force=False, accept_measured_revision=False, con=con,
            already_measured_advice=(
                "Safe RunPod refuses --force and performs no duplicate "
                "paid quant measurement."))
        if registry_gate.get("status") == "already-measured":
            return {
                "provider": "runpod", "created": False, "no_spend": True,
                "status": "already-measured",
                "target": {
                    "repo_id": target.repo_id,
                    "revision": target.revision,
                    "requested_revision": target.requested_revision,
                },
            }
        if registry_gate.get("status") != "proceed":
            raise Refusal(
                "registry identity gate refuses an unavailable output",
                ["status: %s" % registry_gate.get("status")])
    # Tracked files must be clean so the receipt's code digest means what
    # it says.  Untracked scratch files are not part of any bundle or
    # digest and no longer refuse a run.
    initial_source_proof = _source_checkout_proof(include_untracked=False)
    plan_data["source_checkout"] = {
        "initial": initial_source_proof,
        "pre_post": dict(initial_source_proof),
    }
    gate_verified(
        plan_data, "clean-source-checkout",
        head=initial_source_proof["head"])
    if args.role == "root":
        plan_data["cli_probe"] = _probe_root_stage_clis()
        gate_verified(
            plan_data, "root-stage-cli-probe",
            probe_sha256=plan_data["cli_probe"]["probe_sha256"])
    forbidden = _runpod_forbidden(args)
    if forbidden:
        raise Refusal("safe RunPod profile refuses: %s" % ", ".join(forbidden), [])
    download_token = _load_required_hf_download_token(
        args.hf_download_token_file)
    del download_token
    gate_verified(
        plan_data, "explicit-download-credential",
        storage="owned-mode-0600-file",
        remote_lifecycle="fetch-target-only")
    outdir = Path(args.out).resolve() if args.out else None
    if outdir is None:
        raise Refusal("real-safe RunPod planning requires explicit --out", [])
    if outdir.exists():
        raise Refusal("output path already exists", [])
    gate_verified(plan_data, "safe-profile", offer="on-demand", preemption="fail",
                  cold_runs=2, controller="one-ssh-fresh-pod")

    bundle = _bundle_manifest()
    bundle_registry = _bundle_registry_identity()
    bundle_contract_sha256 = hashlib.sha256(_canonical_bytes({
        "bundle": bundle, "registry": bundle_registry})).hexdigest()
    control = _control_manifest()
    plan_data["bundle"] = bundle
    plan_data["bundle_registry"] = bundle_registry
    plan_data["bundle_contract_sha256"] = bundle_contract_sha256
    plan_data["control_plane"] = control

    if not provider.available():
        raise Refusal("RunPod key file is missing, unreadable, unstable or invalid", [])
    provider.require()
    gate_verified(plan_data, "stable-runpod-key")
    initial_status = provider.status()
    initial_account_id = str(initial_status.get("id") or "").strip()
    if not initial_account_id:
        raise Refusal("RunPod status lacks exact myself.id", [])
    health = systemd_reaper_health(
        state_dir=Path(args.reaper_state_dir),
        lease_dir=Path(args.lease_dir), provider="runpod",
        provider_account_id=initial_account_id)
    if not health.get("ok"):
        raise Refusal(
            "the installed RunPod reaper is not healthy",
            _reaper_health_remedy(health, args))
    drift = health.get("source_drift") or {}
    if drift.get("drift"):
        plan_data["warnings"].append(
            "installed reaper is older than this checkout (%s changed); it "
            "still guards this run. Update it when convenient: "
            "measure-cloud reaper --provider runpod --install"
            % (", ".join(drift.get("changed") or [])
               or drift.get("reason") or "unknown"))
    gate_verified(
        plan_data, "reaper-health",
        **{key: value for key, value in health.items()
           if key != "source_drift"})

    surface = sniff_surface(target, getattr(args, "path", None))
    gguf_candidate = (surface.surface == "gguf" and args.role == "root"
                      and bool(getattr(args, "candidate_scope", None)))
    if surface.problems:
        raise Refusal("target surface metadata is not usable", list(surface.problems))
    if gguf_candidate:
        # A GGUF build has no config.json and no safetensors index: its
        # identity is its own tensor table plus the reference release's
        # config, both read by `_refuse_quantized_root`'s gguf branch first.
        _refuse_quantized_root(con, target, surface, plan_data, args=args)
        official_raw, official_ref = plan_data["_gguf_official_config"]
        identity = _gguf_model_file_identity(target, surface, plan_data["_gguf_loaded"],
                                             official_raw, official_ref)
    else:
        identity = _model_file_identity(target)
    license_contract = (
        _root_dataset_license_contract(target)
        if args.role == "root" else None)
    if args.role == "root" and not gguf_candidate:
        _refuse_quantized_root(con, target, surface, plan_data, args=args)
    _refuse_scope_contradicted_by_release(
        con, target.repo_id, target.revision, surface,
        read_json(args.scope_json) if args.scope_json else None, plan_data)
    if surface.surface == "tr3-published":
        if surface.bits is None:
            raise Refusal("TR3 surface has no exact public bit profile", [])
        tr3_profile = "tr3-%gbpw" % float(surface.bits)
        _verify_tr3_seal(
            con, target.repo_id, target.revision, plan_data,
            args=args, profile=tr3_profile)
    if args.role == "root":
        registry_gate = front_gate(
            repo=args.model, revision=target.revision,
            path_hint=getattr(args, "path", None), source=args.registry,
            force=False, accept_measured_revision=False, con=con,
            already_measured_advice=(
                "Safe RunPod refuses --force; a separately identified candidate "
                "measurement (new --dataset-id) continues only through its own gates."
                if getattr(args, "candidate_scope", None) else
                "Safe RunPod refuses --force; a separately identified root "
                "qualification continues only through its own gates."))
        if registry_gate.get("status") not in ("proceed", "already-measured"):
            raise Refusal(
                "registry identity gate refuses an unavailable output",
                ["status: %s" % registry_gate.get("status")])
    gate_verified(plan_data, "scientific-precedent-and-surface",
                  registry=registry_gate.get("status"),
                  surface=surface.surface)
    target_doc = {
        "repo_id": target.repo_id, "revision": target.revision,
        "requested_revision": target.requested_revision,
        "surface": surface.surface, "codec": surface.codec_family,
        "bits": surface.bits, "path": surface.path,
        "size_bytes": (
            surface.artifact_bytes
            if surface.artifact_bytes is not None else identity["model_bytes"]),
        "precision_label": (
            "bfloat16" if surface.surface == "native-bf16"
            else ("%g bpw" % surface.bits
                  if surface.bits is not None else None)),
        "codebook": surface.codebook,
        "container": {
            "native-bf16": "safetensors",
            "fp8-block": "safetensors",
            "tr3-published": "tr3",
            "exl3hf": "exl3",
            "dione": "exl3",
            "gguf": "gguf",
        }.get(surface.surface),
        "shard_hash_verification": "full",
        "bits_per_weight_effective": None,
        "group_size": None,
        "exllamav3_pin": surface.exllamav3_pin,
        "quantizer_tool": surface.codec_family,
        "quantizer_version": surface.exllamav3_pin,
        "config_sha256": identity["config_sha256"],
        "index_sha256": identity["index_sha256"],
        "config_bytes": identity["config_bytes"],
        "index_bytes": identity["index_bytes"],
        "model_bytes": identity["model_bytes"],
        "vocab_size": identity["vocab_size"],
        "hidden_size": identity["hidden_size"],
        "shards": identity["shards"],
        "shard_manifest_sha256": identity["shard_manifest_sha256"],
        "download_bytes_total": identity["download_bytes_total"],
        "download_manifest": identity["download_manifest"],
        "download_manifest_sha256": identity["download_manifest_sha256"],
    }
    if license_contract is not None:
        target_doc["weights_license"] = license_contract["weights_license"]
    if gguf_candidate:
        ggs = _gguf_surface_module()
        loaded = plan_data["_gguf_loaded"]
        verification = (plan_data.get("target") or {}).get("gguf_verification") or {}
        meta = loaded.quant_metadata
        # codec by CENSUS, not by the build name: UD-Q3_K_XL carries IQ3_XXS /
        # IQ4_XS experts and is an i-quant artifact whatever its name says.
        target_doc["codec"] = verification.get("codec_from_census", surface.codec_family)
        target_doc["quantizer_tool"] = ("llama.cpp (quantized_by: %s)" % meta["general.quantized_by"]
                                        if meta.get("general.quantized_by") else "llama.cpp")
        target_doc["quantizer_version"] = (
            "gguf quantization_version %s" % meta["general.quantization_version"]
            if meta.get("general.quantization_version") is not None else None)
        target_doc["bits_per_weight_effective"] = ggs.measured_bits_per_weight(loaded)
        target_doc["config_source"] = identity["config_source"]
        target_doc["index_source"] = identity["index_source"]
        target_doc["gguf_verification"] = verification
        if args.candidate_codec != target_doc["codec"]:
            raise Refusal(
                "--candidate-codec %s disagrees with the build's own type census (%s)"
                % (args.candidate_codec, target_doc["codec"]),
                ["ggml types present: %s" % ", ".join(sorted(verification.get("type_census") or {})),
                 "an IQ-bearing build is gguf-i-quant; a K-quant/Q8_0-only build is gguf-k-quant",
                 "Nothing was created. $0.00 spent."])
    for evidence_key in (
            "seal_verification", "nonrouted_completeness",
            "public_profile_evidence"):
        evidence = (plan_data.get("target") or {}).get(evidence_key)
        if evidence is not None:
            target_doc[evidence_key] = evidence
    plan_data["target"] = target_doc
    gate_verified(plan_data, "target-byte-identity",
                  revision=target.revision,
                  shard_manifest_sha256=identity["shard_manifest_sha256"])
    if args.role == "quant":
        job_scope, scope_binding = _resolve_authored_quant_scope(
            args, target, con)
        gate_verified(
            plan_data, "authored-exact-scope",
            artifact_id=scope_binding["artifact_id"],
            source_sha256=scope_binding["source_sha256"])
    else:
        job_scope = {
            "kind": "root-capture", "engine": "hf-transformers",
            "dtype": "bfloat16", "form": args.form,
        }
        scope_binding = {
            "source": "root-capture-contract",
            "source_sha256": hashlib.sha256(
                _canonical_bytes(job_scope)).hexdigest(),
        }

    engines = load_engines()
    engine = engines.get(args.lane)
    if engine is None:
        raise Refusal("no engine configured for lane %r" % args.lane, [])
    if args.role == "root":
        if not (SUITE_ROOT / "engines/tools/hf_capture.py").is_file():
            raise Refusal("root capture engine is absent", [])
        gpu = _root_gpu_choice(args, target, form=args.form)
    else:
        probe = engine.probe(
            SUITE_ROOT, paid=True, python="python3", env=dict(os.environ))
        if not engine.pinned or not probe["help_ok"]:
            raise Refusal(
                "quant engine/scorer live CLI help probe failed",
                [json.dumps(probe, sort_keys=True)])
        gpu = "H200"
        plan_data["cli_probe"] = probe
        normalized_arg_gpu = (
            (args.gpu or gpu).upper().replace("-", "").replace(" ", ""))
        if normalized_arg_gpu != gpu.upper().replace("-", ""):
            raise Refusal("exact target timing requires GPU %s" % gpu, [])
    descriptor = None
    pmeta = None
    reference_manifest = None
    if args.role == "quant":
        descriptor = load_panel_descriptor(args.panel_descriptor or args.panel)
        pmeta = repo_meta(descriptor.repo_id, "dataset",
                          args.panel_revision or descriptor.revision)
        if (pmeta.repo_id
                != "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
                or pmeta.revision
                != "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"):
            raise Refusal(
                "initial safe quant path requires the exact authored panel pin",
                [])
        quant_panel_validation = _prefetch_quant_panel(pmeta)
        if quant_panel_validation["contexts"] != descriptor.contexts:
            raise Refusal(
                "validated token panel count differs from descriptor", [])
        reference_manifest = validate_reference_manifest(
            pmeta.repo_id, pmeta.revision, repo_meta=pmeta)
        _validate_quant_panel_descriptor(
            descriptor, pmeta, quant_panel_validation, reference_manifest)
    if args.role == "root":
        from fidelity.engines import RootTimingUnavailable
        try:
            timing = resolve_root_timing(
                target_repo=target.repo_id, target_revision=target.revision,
                gpu=gpu, form=args.form,
                schedule="two-fresh-process-qualification")
        except RootTimingUnavailable:
            # No authored timing evidence for this exact target/GPU.  The
            # operator's --max-runtime IS the bound: the reap deadline, the
            # on-pod watchdog and the cost quote all derive from it.  A
            # named-conservative table entry is better evidence when one
            # exists, and the registry receipt records which was used.
            if args.max_runtime is None:
                raise Refusal(
                    "no authored timing evidence for %s@%s on %s, and no --max-runtime"
                    % (target.repo_id, target.revision[:12], gpu),
                    ["pass --max-runtime: with no bin/engines.json root_timing_profiles "
                     "row for this target the operator's deadline is the workload "
                     "bound (GLM-5.3 candidates on an H200 ran with 3h30m; observed "
                     "33-45 min)"])
            timing = _operator_bound_root_timing(args, target, gpu, identity)
            plan_data["warnings"].append(
                "no authored timing evidence for %s@%s on %s; using your "
                "--max-runtime %s as the workload bound. If the capture "
                "needs longer it is stopped and torn down at that deadline."
                % (target.repo_id, target.revision[:12], gpu,
                   args.max_runtime))
        timing_identity = timing["model_identity"]
        for key, actual in (("model_bytes", identity["model_bytes"]),
                            ("config_sha256", identity["config_sha256"]),
                            ("index_sha256", identity["index_sha256"])):
            if timing_identity.get(key) != actual:
                raise Refusal("root timing evidence differs from target %s" % key, [])
        profile = "root-hf-transformers-bf16"
        if (timing.get("evidence") or {}).get("source") == "operator --max-runtime":
            estimated_seconds = Decimal(parse_duration(args.max_runtime))
        else:
            estimated_seconds, derivation = _root_workload_bound(
                timing, storage_layout=args.storage_layout,
                captures=1 if getattr(args, "resume_capture", None) else 2)
            plan_data["workload_bound_derivation"] = derivation
    else:
        profile = resolve_profile(engine, surface.surface, surface.bits)
        if not profile:
            raise Refusal("target has no exact engine profile", [])
        try:
            require_supported_profile(
                engine, surface=surface.surface, bits=surface.bits)
        except EngineProfileRefused as exc:
            raise Refusal(str(exc), [])
        if profile.lower().startswith("k8") or surface.bits == 8:
            raise Refusal("K8 is refused before spend", [])
        if profile != "tr3-6bpw":
            raise Refusal(
                "initial safe RunPod quant path permits only the K6 bridge", [])
    if args.role == "root":
        profile_doc = {
            "profile_id": "root-hf-transformers-bf16",
            "lane": "root", "source": "native",
            "surface": surface.surface, "form": args.form,
            "engine": "hf-transformers", "compute_dtype": "bfloat16",
            "device": "cuda",
            "schedule": "two-fresh-process-qualification",
        }
    else:
        profile_source = {"tr3-6bpw": "tr3"}.get(profile)
        if profile_source is None:
            raise Refusal("quant profile has no authored source identity", [])
        profile_doc = {
            "profile_id": profile, "lane": args.lane,
            "source": profile_source, "surface": surface.surface,
            "bits": surface.bits,
        }
        timing = resolve_profile_timing(
            engine, profile=profile, surface=surface.surface, bits=surface.bits,
            target_repo=target.repo_id, target_revision=target.revision, gpu=gpu)
        runtime_windows = (timing.get("runtime_profile") or {}).get("window_count")
        if pmeta.private is not False:
            raise Refusal(
                "panel repository is not anonymously public", [])
        if runtime_windows != descriptor.contexts:
            raise Refusal("timing window_count differs from resolved panel contexts", [])
        estimated_seconds = (
            Decimal(str(timing["minutes_per_window"])) * 60
            * Decimal(descriptor.contexts) * Decimal(args.cold_runs))
    plan_data["profile"], plan_data["timing"] = profile, timing
    if args.role == "quant":
        runtime = dict(timing.get("runtime_profile") or {})
        expected_cache = "none"
        required_runtime = {
            "gpu": "H200", "gpu_count": 1,
            "window_count": descriptor.contexts,
            "decode_cache": expected_cache, "decode_threads": 28,
            "reader_threads": 28, "min_vcpu_count": 28,
            "min_memory_gb": 300, "controller_processes_per_pod": 1,
        }
        drift = {
            key: {"expected": value, "actual": runtime.get(key)}
            for key, value in required_runtime.items()
            if runtime.get(key) != value}
        if drift:
            raise Refusal(
                "timing evidence lacks the exact invocation/resource contract",
                [json.dumps(drift, sort_keys=True)])
        runtime_contract = dict(
            required_runtime,
            device="cuda",
            expert_parallel={"mode": "single_device", "world_size": 1},
            reduce_order="fp32",
            capacity_basis="authored-profile-measured-host")
    else:
        min_vcpu_count, min_memory_gb, capacity_basis = _root_host_capacity(
            args, target, identity)
        runtime_contract = {
            "min_vcpu_count": min_vcpu_count,
            "min_memory_gb": min_memory_gb,
            "gpu_count": 1, "device": "cuda",
            "expert_parallel": {"mode": "single_device", "world_size": 1},
            "reduce_order": "fp32",
            "capacity_basis": capacity_basis,
        }
    plan_data["runtime_contract"] = runtime_contract
    if args.max_runtime is None:
        # S1-2: the bound is the tool's own number when a timing row exists;
        # requiring the human to retype it (and refusing when they were off)
        # cost the documented recipe two round trips.
        args.max_runtime = "%ds" % int(estimated_seconds)
        plan_data["max_runtime_source"] = (
            "defaulted to the authored bound; --max-runtime to override upward")
        plan_data["warnings"].append(
            "--max-runtime defaulted to the authored bound %d s (bin/engines.json "
            "root_timing_profiles[%s@%s] on %s); pass --max-runtime to override "
            "upward" % (int(estimated_seconds), target.repo_id,
                        target.revision[:12], gpu))
    max_runtime = parse_duration(args.max_runtime)
    if estimated_seconds > Decimal(max_runtime):
        _defer_refusal(plan_data, Refusal(
            "target-specific timing exceeds --max-runtime: the bound is %s s "
            "(%s), --max-runtime is %s s"
            % (estimated_seconds,
               plan_data.get("workload_bound_derivation", {}).get(
                   "basis", "conservative_upper_hours"), max_runtime),
            ["raise --max-runtime to at least the bound (or omit it: the "
             "authored bound is the default); it is the deadline the watchdog "
             "enforces, not an estimate",
             "the cost below is priced at the bound, so one edit settles both"]))
        # Price the run at the bound so the cost finding beside this one is
        # the number the human will see after the one edit that fixes this.
        max_runtime = float(estimated_seconds)
    gate_verified(plan_data, "target-profile-timing", profile=profile,
                  evidence_sha256=hashlib.sha256(_canonical_bytes(timing)).hexdigest(),
                  workload_bound_seconds=str(estimated_seconds),
                  derivation=plan_data.get("workload_bound_derivation"))

    panel_doc: Dict[str, Any]
    panel_local = None
    binding_local = None
    if args.role == "root":
        if not args.panel_dir:
            raise Refusal("safe RunPod root requires local --panel-dir", [])
        panel_root = Path(args.panel_dir).resolve()
        panel_temp = tempfile.TemporaryDirectory(prefix="fidelity-panel-plan-")
        _RUNPOD_TEMP_HOLDS.append(panel_temp)
        panel_local = Path(panel_temp.name) / "panel.tar"
        archive, tokenizer_temp = _write_verified_panel_archive(
            panel_root, panel_local, args.panel_tokenizer_root)
        if getattr(args, "candidate_scope", None):
            # A candidate is scored against a published root, so the panel
            # contract that applies is the REFERENCE root's, and the
            # candidate's tokenizer files must be byte-identical to that
            # root's (the binding names the candidate repo; only the
            # repository/revision labels differ, every file digest and the
            # identity must match, or the validator refuses).
            reference = _candidate_reference_manifest(args, plan_data)
            reference_weights = reference.get("weights") or {}
            reference_repo = reference_weights.get("repository")
            reference_revision = (reference_weights.get("model_revision")
                                  or reference_weights.get("revision"))
            # The binding's tokenizer block is the PANEL's identity pin (which
            # repo's tokenizer files the panel was sealed with); the reference
            # root is a different fact (which model produced the reference
            # logits).  Overwriting the pin with the reference repo conflated
            # them: exact for a GLM-5.3-BF16 root (same repo), wrong for the
            # GLM-5.2 root, which legitimately pins 5.3-BF16's byte-identical
            # tokenizer and was refused with "tokenizer pin ... differ"
            # (2026-09-05, every 5.2 candidate at $0).  The validator
            # dispatches on (reference repo, revision) and checks the pin
            # against the panel's own constants; nothing is relabelled.
            relabelled = json.loads(_canonical_bytes(archive["binding"]).decode("utf-8"))
            validated_root_binding = validate_root_panel_binding(
                relabelled, reference_repo, reference_revision)
            if validated_root_binding != relabelled:
                raise PanelError(
                    "root panel validator changed the resolved binding")
            if plan_data.get("_gguf_loaded") is not None:
                _refuse_gguf_tokenizer_mismatch(
                    con, target, relabelled, reference_repo, reference_revision, plan_data)
            else:
                _refuse_candidate_tokenizer_mismatch(
                    con, target, relabelled, reference_repo, reference_revision, plan_data)
            con.ok("candidate panel",
                   "exact for reference root %s@%s"
                   % (reference_repo, str(reference_revision)[:12]))
        else:
            validated_root_binding = validate_root_panel_binding(
                archive["binding"], target.repo_id, target.revision)
            if validated_root_binding != archive["binding"]:
                raise PanelError(
                    "root panel validator changed the resolved binding")
        if tokenizer_temp is not None:
            _RUNPOD_TEMP_HOLDS.append(tokenizer_temp)
        binding_bytes = _canonical_bytes(archive["binding"])
        binding_local = panel_local.with_suffix(".binding.json")
        binding_local.write_bytes(binding_bytes)
        panel_doc = {
            "resolved_binding": archive["binding"],
            "binding_path": "inputs/panel.binding.json",
            "binding_file_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "archive_path": "inputs/panel.tar",
            "archive_bytes": archive["bytes"],
            "archive_sha256": archive["sha256"],
            "content_path": "inputs/panel",
        }
        plan_data["_panel_archive_local"] = str(panel_local)
        plan_data["_panel_binding_local"] = str(binding_local)
    else:
        panel_doc = dict(
            descriptor.to_dict(), revision=pmeta.revision,
            fetch_bytes=pmeta.bytes_matching(descriptor.include),
            validated_reference_manifest=reference_manifest,
            validated_token_panel={
                key: value for key, value in quant_panel_validation.items()
                if key != "local_root"})
        plan_data["_quant_panel_root"] = quant_panel_validation["local_root"]
    plan_data["panel"] = panel_doc
    gate_verified(plan_data, "panel-binding",
                  binding_sha256=panel_doc.get("binding_file_sha256",
                                               hashlib.sha256(_canonical_bytes(panel_doc)).hexdigest()))
    plan_data["anonymous_access"]["target"] = {
        "repo_id": target.repo_id, "revision": target.revision,
        "private": target.private,
        "config_sha256": identity["config_sha256"],
        "index_sha256": identity["index_sha256"],
    }
    plan_data["anonymous_access"]["panel"] = (
        {
            "repo_id": pmeta.repo_id, "revision": pmeta.revision,
            "private": pmeta.private,
            "reference_manifest_sha256":
                reference_manifest["manifest_sha256"],
            "token_panel_validated": True,
        } if args.role == "quant" else {
            "mode": "local-job-bound-archive",
            "binding_sha256": panel_doc["binding_file_sha256"],
        })
    gate_verified(
        plan_data, "anonymous-public-metadata",
        target_config_sha256=identity["config_sha256"],
        target_index_sha256=identity["index_sha256"],
        panel_private=(pmeta.private if pmeta is not None else False))

    allowlist = None
    if args.role == "root":
        from fidelity.runpodsafety import authored_allowlist_path
        if not args.unexpected_tensor_allowlist:
            args.unexpected_tensor_allowlist = authored_allowlist_path(
                target.repo_id, target.revision, suite_root=SUITE_ROOT)
        if args.unexpected_tensor_allowlist:
            allowlist = validate_unexpected_tensor_allowlist(
                args.unexpected_tensor_allowlist, target_repo=target.repo_id,
                target_revision=target.revision, suite_root=SUITE_ROOT)
            gate_verified(
                plan_data, "exact-unexpected-tensor-allowlist",
                provenance="authored",
                artifact_sha256=allowlist["artifact_sha256"],
                names_sha256=allowlist[
                    "canonical_sorted_names_sha256"])
            allowlist_bundle_rows = [
                row for row in bundle["files"]
                if row["path"] == allowlist["path"]]
            if (len(allowlist_bundle_rows) != 1
                    or allowlist_bundle_rows[0]["sha256"]
                    != allowlist["artifact_sha256"]):
                raise Refusal(
                    "authored allowlist differs from frozen bundle manifest", [])
        else:
            allowlist = _derive_index_census_allowlist(
                target, identity, plan_data, con)
            if allowlist is not None:
                gate_verified(
                    plan_data, "exact-unexpected-tensor-allowlist",
                    provenance="derived_from_index",
                    artifact_sha256=allowlist["artifact_sha256"],
                    names_sha256=allowlist["canonical_sorted_names_sha256"],
                    count=allowlist["count"])
                plan_data["warnings"].append(
                    "no authored unexpected-tensor allowlist for %s@%s: bound "
                    "the index census instead (%d names past layer %s, sha256 "
                    "%s...). The pod refuses unless the loader's unexpected set "
                    "equals it exactly. To attest it, register the printed "
                    "digests in bin/fidelity/runpodsafety.py _ALLOWLISTS and "
                    "commit the file under engines/tools/layer-outer-evidence/"
                    % (target.repo_id, target.revision[:12], allowlist["count"],
                       allowlist["decoder_layers"], allowlist["artifact_sha256"][:16]))
            else:
                plan_data["warnings"].append(
                    "no unexpected-tensor allowlist for %s@%s and its index "
                    "carries no key past the declared decoder layers: the "
                    "capture refuses on the pod if the loader reports any "
                    "tensor the architecture does not declare"
                    % (target.repo_id, target.revision[:12]))
        if args.sanity_expect == "":
            plan_data["warnings"].append(
                "--sanity-expect '': the on-pod generation probe (\"The "
                "capital of France is\") is RECORDED in the dataset but not "
                "enforced. That is the declaration of an undertrained proxy; "
                "a production root must run with the default (Paris).")

    offers = provider.gpus()
    provider_gpu_id = _RUNPOD_GPU_IDS.get(
        gpu.upper().replace("-", "").replace(" ", ""))
    if provider_gpu_id is None:
        raise Refusal(
            "GPU %r has no RunPod offer name here; known: %s"
            % (gpu, ", ".join(sorted(_RUNPOD_GPU_IDS))), [])
    candidates = [offer for offer in offers
                  if offer.gpu_type == provider_gpu_id
                  and offer.spot is False
                  and offer.region == "secure"
                  and offer.free_devices >= 1]
    if not candidates:
        raise Refusal("no secure on-demand RunPod %s offer is available" % gpu, [])
    def _offer_exact_rate(candidate):
        raw_rate = (candidate.raw or {}).get(
            "uninterruptablePriceDecimal")
        try:
            parsed = Decimal(str(raw_rate))
        except Exception as exc:
            raise Refusal(
                "RunPod offer lacks exact decimal on-demand rate", []) from exc
        if not parsed.is_finite() or parsed <= 0:
            raise Refusal(
                "RunPod offer exact decimal rate is not positive finite", [])
        return parsed
    offer = sorted(
        candidates,
        key=lambda item: (_offer_exact_rate(item), item.gpu_type))[0]
    exact_offer_rate = _offer_exact_rate(offer)
    chosen = {
        "provider": "runpod", "provider_gpu_id": provider_gpu_id,
        "provider_gpu_display": provider_gpu_id,
        # resultsink._validate_runpod_attestation binds the attestation's
        # gpu_model to environment.gpu; without this key the archive build
        # refused a complete, qualified Fruit capture (2026-09-03).
        "gpu": provider_gpu_id,
        "gpu_type": gpu, "gpus": 1, "region": offer.region,
        "hard_cap_usd": str(Decimal(str(args.max_cost))),
        "vram_bytes": int(offer.vram_bytes),
        "price_per_gpu_hour": format(exact_offer_rate, "f"),
        "price_per_gpu_hour_display": offer.price,
        "ssh_host_key_policy":
            "strict-ed25519-out-of-band-runpod-web-terminal",
        "ssh_endpoint_binding": "provider-api-exact-pod-id",
    }
    plan_data["chosen"] = chosen
    gate_verified(plan_data, "exact-offer", **chosen)

    if args.role == "root":
        vocab_size = identity["vocab_size"]
        hidden_size = identity["hidden_size"]
        scored_positions = (
            panel_doc["resolved_binding"]["panel"]
            .get("scored_positions_total"))
        if (isinstance(scored_positions, bool)
                or not isinstance(scored_positions, int)
                or scored_positions <= 0):
            raise Refusal(
                "root panel lacks exact selected prediction positions", [])
        hidden_bytes = scored_positions * hidden_size * 2
        shared_head_bytes = vocab_size * hidden_size * 2
        capture_bytes_per_process = hidden_bytes + shared_head_bytes
        panel_bytes = int(panel_doc["archive_bytes"])
        artifact_bytes = identity["download_bytes_total"]
        # Archive creation can coexist with both raw fresh-process captures.
        extra_bytes = capture_bytes_per_process * 2
        archive_uncompressed = extra_bytes + 67108864
        archive_transfer = (
            archive_uncompressed
            + ((archive_uncompressed + 16382) // 16383) * 5 + 64)
        target_doc["root_capture_storage"] = {
            "form": "hidden", "storage_dtype": "bfloat16",
            "selected_prediction_positions": scored_positions,
            "vocab_size": vocab_size, "hidden_size": hidden_size,
            "bytes_per_element": 2, "fresh_processes": 2,
            "hidden_bytes_per_process": hidden_bytes,
            "shared_head_bytes_per_process": shared_head_bytes,
            "bytes_per_process": capture_bytes_per_process,
            "capture_bytes_total": capture_bytes_per_process * 2,
            "capture_archive_duplicate_upper_bound_bytes":
                capture_bytes_per_process * 2,
            "required_dataset_trees": 2,
            "result_archive_max_members": scored_positions * 2 + 128,
            "result_archive_max_uncompressed_bytes": archive_uncompressed,
            "result_archive_max_transfer_bytes": archive_transfer,
        }
    else:
        panel_bytes = panel_doc["fetch_bytes"]
        artifact_bytes = identity["download_bytes_total"]
        extra_bytes = 0
    if args.role == "root":
        result_archive_contract = {
            name: target_doc["root_capture_storage"][name]
            for name in (
                "required_dataset_trees", "result_archive_max_members",
                "result_archive_max_uncompressed_bytes",
                "result_archive_max_transfer_bytes")
        }
    else:
        quant_archive_uncompressed = 2 * 1024 ** 3
        result_archive_contract = {
            "retained_content": [
                "receipts", "reports", "bounded-log-tails", "control"],
            "result_archive_max_members": 2048,
            "result_archive_max_uncompressed_bytes":
                quant_archive_uncompressed,
            "result_archive_max_transfer_bytes":
                quant_archive_uncompressed
                + ((quant_archive_uncompressed + 16382) // 16383) * 5 + 64,
        }
    target_doc["result_archive_contract"] = result_archive_contract
    if args.role == "quant" and profile == "tr3-6bpw":
        official_target = repo_meta(
            "zai-org/GLM-5.3-Flash-BF16", "model", OFFICIAL_BF16_REVISION)
        official_identity = _model_file_identity(official_target)
        artifact_bytes += (
            official_identity["config_bytes"] + official_identity["index_bytes"])
        target_doc["official_bf16_identity"] = official_identity
    if (args.role == "quant"
            and surface.surface in ("exl3hf", "tr3-published", "dione", "gguf")):
        extra_bytes += C.glm53_flash_census().nonrouted_bytes
    if args.role == "root" and getattr(args, "candidate_scope", None):
        # The verified reference root dataset lands beside the candidate's
        # own two captures: one more hidden-form dataset plus its download
        # cache (the fetch hashes into place, so ~2x the tree at peak).
        extra_bytes += 2 * 4 * 1024 ** 3
    storage_need = C.storage_need(
        artifact_bytes=float(artifact_bytes), panel_bytes=float(panel_bytes),
        keep_student_logits=False, cold_runs=2, extra_bytes=float(extra_bytes))
    computed_storage = C.round_up_storage_gb(storage_need.total_bytes)
    if args.storage is not None and args.storage < computed_storage:
        raise Refusal("--storage is smaller than exact storage arithmetic", [])
    storage_gb = int(args.storage or computed_storage)
    plan_data["storage_need"] = storage_need.to_dict()
    archive_container_bytes = result_archive_contract[
        "result_archive_max_transfer_bytes"]
    layout = args.storage_layout
    plan_data["storage_layout"] = layout
    run_bytes = int(
        Decimal(str(storage_need.total_bytes)).to_integral_value(
            rounding=ROUND_CEILING))
    if layout == "container-disk":
        # Weights, captures and the archive staging all live on the
        # container disk; the volume is nominal (see RUNPOD_STORAGE_LAYOUTS).
        plan_data["storage_gb"] = RUNPOD_STORAGE_LAYOUTS[layout][
            "nominal_volume_gb"]
        plan_data["container_disk_gb"] = max(
            20, C.round_up_storage_gb(
                run_bytes + archive_container_bytes + 67108864))
        workspace_available_bytes_minimum = 1024 ** 3
        container_available_bytes_minimum = int(
            run_bytes + archive_container_bytes + 67108864)
    else:
        plan_data["storage_gb"] = storage_gb
        plan_data["container_disk_gb"] = max(
            20, C.round_up_storage_gb(archive_container_bytes + 67108864))
        workspace_available_bytes_minimum = run_bytes
        container_available_bytes_minimum = int(
            archive_container_bytes + 67108864)
    plan_data["resource_requirements"] = {
        "workspace_available_bytes_minimum":
            workspace_available_bytes_minimum,
        "container_available_bytes_minimum":
            container_available_bytes_minimum,
        "min_vcpu_count": runtime_contract["min_vcpu_count"],
        "min_memory_gb": runtime_contract["min_memory_gb"],
        "expected_vram_bytes": chosen["vram_bytes"],
    }
    result_archive_bound = archive_container_bytes
    local_verify_bound_seconds = max(
        60, (result_archive_bound + 16777215) // 16777216)
    retrieval_attempts = 3
    retrieval_delete_minimum = (
        1800
        + retrieval_attempts * (3600 + local_verify_bound_seconds)
        + 300)
    if args.retrieval_delete_reserve is None:
        # S2-4: the reserve funds exactly this contract -- archive build
        # (1800 s), three bounded 3600 s download attempts each followed by
        # local verification sized by the archive, and the delete (300 s).
        # Its minimum IS the derived value; every GLM-5.3 candidate passed
        # it by hand (13818 s for a 5.13 GB archive). The old flat default
        # (21600 s, 6 h) tipped a $4 candidate over a $45 cap.
        args.retrieval_delete_reserve = retrieval_delete_minimum
        plan_data["retrieval_delete_reserve_source"] = (
            "derived: 1800 build + 3 x (3600 download + %d verify) + 300 delete; "
            "--retrieval-delete-reserve to override upward"
            % local_verify_bound_seconds)
    elif int(args.retrieval_delete_reserve) < retrieval_delete_minimum:
        raise Refusal(
            "--retrieval-delete-reserve is below three bounded downloads, "
            "their local verification work, archive build, and deletion reserve",
            ["minimum seconds: %d (omit the flag to use exactly that)"
             % retrieval_delete_minimum])
    plan_data["retrieval_delete_contract"] = {
        "remote_archive_build_timeout_seconds": 1800,
        "download_attempts": retrieval_attempts,
        "download_timeout_seconds_per_attempt": 3600,
        "local_verify_extract_bound_seconds_per_attempt":
            local_verify_bound_seconds,
        "final_delete_reserve_seconds": 300,
        "minimum_reserve_seconds": retrieval_delete_minimum,
        "bound_archive_bytes": result_archive_bound,
    }
    plan_data["post_create_convergence"] = {
        "schema": "fidelity-suite/runpod-post-create-convergence.v1",
        "timeout_seconds": POST_CREATE_CONVERGENCE_SECONDS,
        "poll_seconds": POST_CREATE_CONVERGENCE_POLL_SECONDS,
    }
    quote = _runpod_quote(
        args, chosen, target, profile, timing, storage_gb,
        plan_data["container_disk_gb"], Decimal(max_runtime),
        result_archive_contract, warnings=plan_data["warnings"],
        deferred=plan_data["_deferred_refusals"])
    plan_data["cost_quote"] = quote.to_dict()
    from fidelity.runpodapi import DEFAULT_IMAGE as RUNPOD_IMAGE
    # The pod's image: runpod/pytorch by digest (the locked stack is rebuilt
    # on the pod), or the measurement image's ssh target pinned by digest
    # (the stack is baked and the bootstrap seeds from it). Never a tag.
    runpod_image = getattr(args, "runpod_image", None) or RUNPOD_IMAGE
    if re.fullmatch(r"[a-z0-9.\-/]+@sha256:[0-9a-f]{64}", runpod_image) is None:
        raise Refusal("--runpod-image must be an immutable name@sha256:<64-hex> reference",
                      ["a tag can be repointed after the receipt is sealed"])
    chosen["image"] = runpod_image
    # Pin the datacenter when asked: Hub fetch throughput differed ~10x
    # between RunPod secure hosts on 2026-09-04 (1.7 GB/s vs 240 MB/s on
    # the same repo), and the location is the one lever the request has.
    # Recorded on the plan; the attestation records what was served.
    chosen["data_center_id"] = getattr(args, "runpod_datacenter", None)
    chosen["image_reference_mutable"] = False
    plan_data["max_runtime_seconds"] = max_runtime
    plan_data["provider_termination_seconds"] = int(
        quote.provider_termination_deadline_seconds)
    gate_verified(plan_data, "all-in-decimal-quote",
                  calculated_maximum_usd=str(quote.calculated_maximum_usd()),
                  hard_cap_usd=str(quote.hard_cap_usd))

    plan_status = provider.status()
    provider_account_id = str(plan_status.get("id") or "").strip()
    if not provider_account_id:
        raise Refusal("RunPod status lacks exact myself.id", [])
    chosen["provider_account_id"] = provider_account_id
    plan_data["provider_account_id"] = provider_account_id
    inventory = provider.chargeable_inventory()
    if not inventory.get("complete"):
        raise Refusal("RunPod pod/network-volume inventory is incomplete", [])
    if provider_account_id != initial_account_id:
        raise Refusal("RunPod account identity changed during planning", [])
    if _campaign_ledger_requested(args) and (
            Path(args.campaign_ledger).resolve().parent
            != Path(args.lease_dir).resolve().parent):
        raise Refusal(
            "campaign ledger must be a sibling of the lease directory", [])
    publication_preflight = None
    if args.role == "root" and args.publish_root_to is not None:
        from fidelity import dshub
        try:
            publication_preflight = dshub.preflight_create(
                args.publish_root_to, args.hf_token_file)
        except Exception as exc:
            _defer_refusal(plan_data, Refusal(
                "local root publication preflight failed: %s" % redact(str(exc)),
                ["--publish-root-to %s must be a dataset repo the token can create "
                 "and that does not exist yet; pick a fresh name (the published "
                 "GLM-5.3 root is malaiwah/glm53-fidelity-root-v1), or drop the "
                 "flag and publish later with fidelity-dataset publish "
                 "(docs/THIRD-PARTY-QUICKSTART.md 4)" % args.publish_root_to]))
    plan_data["inventory_plan"] = inventory
    gate_verified(plan_data, "complete-chargeable-inventory",
                  pods=len(inventory["families"]["pods"]["resources"]),
                  network_volumes=len(
                      inventory["families"]["network_volumes"]["resources"]))

    if args.runpod_safety_proof:
        # Opt-in: a sealed controller-loss drill proof for this exact
        # checkout, ledger and account.  The reaper is what tears down after
        # a controller dies; the proof is empirical evidence that it did so
        # once, here.  Validated exactly as before when given.
        proof = validate_safety_proof(
            args.runpod_safety_proof, bundle_contract_sha256,
            control["manifest_sha256"], provider_account_id,
            str(Path(args.campaign_ledger).resolve()))
        proof_account_id = str(
            (proof.get("proof") or {}).get("provider_account_id") or "")
        if proof_account_id != provider_account_id:
            raise Refusal(
                "RunPod safety proof belongs to a different provider account", [])
        plan_data["safety_proof_sha256"] = proof["proof"]["proof_sha256"]
        gate_verified(plan_data, "current-paid-fault-drill",
                      proof_sha256=proof["proof"]["proof_sha256"])
    else:
        plan_data["safety_proof_sha256"] = None
        gate_verified(
            plan_data, "teardown-backstops",
            controller="delete-on-exit-exception-interrupt",
            on_pod="watchdog at deadline",
            autonomous="installed systemd reaper at reap deadline",
            drill_proof="not requested (--runpod-safety-proof)")

    capture: Dict[str, Any] = {}
    if args.role == "root":
        allowlist_job = (
            {key: allowlist[key] for key in (
                "path", "artifact_sha256", "canonical_sorted_names_sha256")}
            if allowlist is not None else None)
        binding_panel = panel_doc["resolved_binding"]["panel"]
        capture = {
            "engine": "hf-transformers",
            "dtype": "bfloat16",
            "dataset_repository": args.dataset_repository,
            "role": "root", "form": args.form,
            "schedule": "layer-outer",
            "panel_dir": panel_doc["content_path"],
            "panel_id": binding_panel["id"],
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name or args.dataset_id,
            "author": args.measurer,
            "race": False, "preview_of": None,
            "sanity_expect": args.sanity_expect,
            "device": args.capture_device,
            "publish_root_to": args.publish_root_to,
            "unexpected_tensor_allowlist": allowlist_job,
            "resume_capture": _resume_capture_identity(args),
            "candidate": _candidate_block(args, plan_data, con, binding_panel),
            "dataset_license": license_contract["dataset_license"],
            "weights_license": license_contract["weights_license"],
            "replay_device": args.replay_device,
            "replay_dtype": args.replay_dtype,
            "vocab_chunk": args.replay_vocab_chunk,
            "replay": {"device": args.replay_device, "dtype": args.replay_dtype,
                       "vocab_chunk": args.replay_vocab_chunk},
            # HEAD-1d: compare_reference replays each side through the head its
            # own dataset sealed. Bitwise-identical to the shared-head replay
            # when the candidate's head IS the source tensor (official FP8
            # releases), and the only strict path when it is not (every
            # exllamav3 head_bits=16 head is the source head after an fp16
            # round trip). Recorded on the job so the pod's stage and this
            # plan agree on what head policy the receipt will carry.
            "own_heads": True,
            "root_protocol": {
                "schedule": "two-fresh-process-qualification",
                "fresh_processes": 2, "run_count_per_process": 1,
                "exact_self_comparison": True,
                "qualification_required": True,
                "canonical_publication_required":
                    args.publish_root_to is not None,
                "publication_mode": (
                    "canonical-public" if args.publish_root_to
                    else "qualified-unpublished"),
            },
        }
    job = finalize_job({
        "schema": "fidelity-suite/job.v2",
        "role": args.role, "recipe": "cloud", "lane": args.lane,
        "cold_runs": 2, "profile": profile_doc, "timing": timing,
        "target": target_doc, "panel": panel_doc,
        "reference": {
            "reference_ref": panel_doc.get("reference_ref"),
            "teacher_receipt_sha256": panel_doc.get("teacher_receipt_sha256"),
            "teacher_backend_identity_sha256":
                panel_doc.get("teacher_backend_identity_sha256"),
        },
        "resource_requirements": plan_data["resource_requirements"],
        "post_create_convergence": plan_data["post_create_convergence"],
        "source_checkout": plan_data["source_checkout"],
        "cli_probe": plan_data["cli_probe"],
        "anonymous_access": plan_data["anonymous_access"],
        "capture": capture,
        **({
            "scoring": {
                "schema": "fidelity-suite/kld-scoring.v1",
                "device": "cuda", "chunk_positions": 512,
                "compute_dtype": "float64",
                "direction": "reference_to_candidate",
                "vocabulary": "full",
                "reduction": "mean_of_run_means_tokenwise_kld",
            },
        } if args.role == "quant" else {}),
        "environment": chosen,
        "bundle": bundle, "bundle_registry": bundle_registry,
        "bundle_contract_sha256": bundle_contract_sha256,
        "scope": job_scope,
        "scope_binding": scope_binding,
        "control_plane": control,
        "publication_preflight": publication_preflight,
        "measurer": {
            "name": args.measurer, "handle": args.measurer,
            "url": "https://huggingface.co/%s" % args.measurer,
            "is_artifact_author":
                args.measurer == target.repo_id.split("/", 1)[0],
        },
        "attribution_binding": {
            "target_owner": target.repo_id.split("/", 1)[0],
            "measurer_handle": args.measurer,
            "rule": "exact-handle-equality",
        },
        "runtime": runtime_contract,
        "reduce_order": args.reduce_order,
        "disclosures": plan_data.get("disclosures") or [],
        "keep_student_logits": False,
        "official_bf16_revision": OFFICIAL_BF16_REVISION,
        "produced_by": produced_by_block(
            SUITE_ROOT, "bin/measure_cloud.py",
            dependencies={
                "lane": args.lane, "provider": "runpod",
                "profile": profile_doc["profile_id"]}),
        "execution_attempt": {
            "kind": "runpod-ssh",
            "attempt_id": None, "cost_quote": None, "engine_root": None,
            "execution_contract_sha256": None,
            "lease_path": None, "pre_create_safety": None,
            "prepared_create": None,
            "remote_root": None, "storage_layout": None,
            "workload_deadline_utc": None, "provider_terminate_after": None,
            "planned_at": None,
        },
    })
    verify_job(job)
    # The root-qualification contract the pod will enforce at qualify_root,
    # evaluated here at $0. Two candidate runs on 2026-09-04 reached
    # qualify_root after both cold runs and the self-compare before this
    # contract refused them (target surface hardcoded fp8-block; an
    # unrecorded decode) -- everything the contract reads from the JOB is
    # knowable now, so refuse now.
    if args.role == "root":
        from fidelity.jobcontract import root_qualification_contract
        try:
            root_qualification_contract(job)
        except JobContractError as exc:
            raise Refusal(
                "the job cannot form its root-qualification contract: %s" % exc,
                ["qualify_root would refuse this job on the pod after both cold runs; "
                 "the target surface/codec/bits must agree with the candidate block's "
                 "declared decode"])
    if (job["produced_by"]["revision"]
            != plan_data["source_checkout"]["initial"]["head"]):
        raise Refusal(
            "produced_by revision differs from clean source HEAD", [])
    plan_data["job"] = job
    plan_data["job_id"], plan_data["job_id_full"] = job["job_id"], job["job_id_full"]
    if args.role == "root":
        _validate_resume_capture(args, plan_data, con)
    preview_now = _exact_utc_now()
    preview_valid = (
        datetime.now(timezone.utc) + timedelta(minutes=5)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    provider_resources = []
    for family_name, family in (inventory.get("families") or {}).items():
        for row in family.get("resources") or []:
            resource = {
                "family": family_name,
                "id": str(row.get("id") or "").strip(),
                "name": str(row.get("name") or "").strip(),
                "status": str(row.get("status") or "").strip(),
            }
            if not all(resource.values()):
                raise Refusal(
                    "provider inventory lacks exact id/name/status", [])
            provider_resources.append(resource)
    preview_common = {
        "provider": "runpod",
        "provider_account_id": provider_account_id,
        "balance_available_usd": plan_status.get("clientBalance"),
        "balance_observed_at": preview_now,
        "balance_valid_until": preview_valid,
        "balance_source": "RunPod myself.clientBalance",
        "inventory_observed_at": inventory["observed_at_utc"],
        "inventory_valid_until": preview_valid,
        "inventory_complete": True,
        "provider_resources": provider_resources,
        "inventory_source": inventory["schema"],
    }
    preview_attempt = "0" * 24
    lease_store_plan = LeaseStore(Path(args.lease_dir))
    if _campaign_ledger_requested(args):
        # Strict campaign: the ledger must pre-exist with matching limits,
        # and every unresolved lease must belong to it and to the reaper's
        # last sealed count.
        campaign_path, preview_ledger = _open_existing_runpod_campaign(
            args, provider_account_id)
        try:
            unresolved_scope = validate_unresolved_lease_scope(
                lease_store_plan, health,
                provider="runpod", provider_account_id=provider_account_id,
                campaign_ledger_path=Path(campaign_path))
        except Exception as exc:
            raise Refusal(
                "unresolved leases are outside current reaper/campaign scope: %s"
                % exc, [])
        plan_data["unresolved_lease_scope"] = unresolved_scope
        gate_verified(
            plan_data, "canonical-unresolved-lease-scope", **unresolved_scope)
        _raise_deferred_refusals(plan_data)
        preview_snapshot = preview_ledger.snapshot()
        decision = preview_ledger.preview_reserve_with_provider_snapshot(
            preview_snapshot["generation"], job["job_id_full"],
            preview_attempt, quote, preview_now,
            effective_width=args.campaign_width, **preview_common)
        plan_data["campaign_mode"] = "explicit"
    else:
        # Per-run campaign: refuse only while an earlier lease may still
        # hold a pod; preview admission in memory against --max-cost.
        from fidelity.campaign import CampaignLedger
        from fidelity.cloudlease import LeaseError
        campaign_path = _auto_campaign_ledger_path(args, job["job_id_full"])
        try:
            liability_scope = validate_lease_liability_scope(
                lease_store_plan, provider="runpod",
                provider_account_id=provider_account_id,
                allow_live=bool(getattr(args, "allow_unresolved_leases", False)))
        except LeaseError as exc:
            # An AMBIGUOUS lease with no pod id is not settled by waiting or
            # sweeping (cloudlease yields ambiguous-needs-operator); say what
            # settles it instead of the two remedies that cannot.
            _defer_refusal(plan_data, Refusal(
                str(exc).split(". Wait for the reaper", 1)[0],
                ["inspect: measure-cloud reaper --provider runpod --list",
                 "an ACTIVE lease is a pod of yours still running: wait for its "
                 "deadline or finish that run first",
                 "an AMBIGUOUS lease with no pod id needs you: verify in the RunPod "
                 "console that no pod named fidcloud-* from that attempt exists, "
                 "then pass --allow-unresolved-leases to proceed beside it (the "
                 "reaper still destroys anything past its deadline)"]))
            liability_scope = None
        if liability_scope is not None:
            plan_data["unresolved_lease_scope"] = liability_scope
            gate_verified(
                plan_data, "no-live-liability-leases", **liability_scope)
        # Every arithmetic finding is in by now; the campaign preview below
        # would only restate the cost one as CEILING_EXCEEDED.
        _raise_deferred_refusals(plan_data)
        limits = _auto_campaign_limits(args)
        decision = CampaignLedger.preview_new_campaign(
            job_hash=job["job_id_full"], attempt=preview_attempt,
            quote=quote, now=preview_now,
            effective_width=1, **limits, **preview_common)
        plan_data["campaign_mode"] = "per-run"
    plan_data["campaign_ledger_path"] = campaign_path
    plan_data["campaign_admission_preview"] = decision.to_dict()
    if not decision.admitted:
        raise Refusal(
            "campaign admission preview refused [%s]: %s"
            % (decision.code, decision.message), [])
    if args.campaign_width == 2:
        width_two = validate_width_two_root_archive(
            args.width_two_root_archive, job)
        current_public = validate_current_public_root(
            width_two["publication"])
        gate_verified(
            plan_data, "width-two-fruit-root",
            publication_sha256=hashlib.sha256(
                _canonical_bytes(current_public)).hexdigest())
        plan_data["_width_two_authorization"] = {
            "fruit_public_archive_sha256":
                sha256_file(args.width_two_root_archive),
            "fruit_proof_sha256": hashlib.sha256(
                _canonical_bytes(current_public)).hexdigest(),
        }
    else:
        gate_verified(plan_data, "campaign-width", width=1)
    _raise_deferred_refusals(plan_data)
    con.say("RUNPOD PLAN")
    con.kv("target", "%s@%s" % (target.repo_id, target.revision))
    derivation = plan_data.get("workload_bound_derivation") or {}
    if derivation.get("basis") == "components_seconds":
        con.kv("profile timing",
               "%s / bound %s s = (%s fetch + %s setup + %d x %s cold run + %s "
               "verify/compare/qualify) x %s margin; authored bin/engines.json "
               "root_timing_profiles[%s@%s] on %s"
               % (profile, derivation["seconds"], derivation["fetch"],
                  derivation["setup"], derivation["captures"], derivation["cold_run"],
                  derivation["verify_compare_qualify"], derivation["margin_factor"],
                  target.repo_id, target.revision[:8], chosen["gpu_type"]))
    elif derivation.get("basis") == "conservative_upper_hours":
        con.kv("profile timing", "%s / bound %s h authored in bin/engines.json "
               "root_timing_profiles[%s@%s] (conservative_upper_hours, no components)"
               % (profile, derivation["hours"], target.repo_id, target.revision[:8]))
    elif (timing.get("evidence") or {}).get("source") == "operator --max-runtime":
        con.kv("profile timing", "%s / operator --max-runtime %s (no authored timing "
               "row for this target on %s)"
               % (profile, args.max_runtime, chosen["gpu_type"]))
    else:
        con.kv("profile timing", "%s / %s" % (profile, timing.get("evidence")))
    con.kv("job hash", job["job_id_full"])
    rate = Decimal(str(chosen["price_per_gpu_hour"])) * Decimal(chosen["gpus"])
    con.kv("gpu", "%s x%d (secure cloud, on-demand) $%s/h"
           % (chosen["gpu"], chosen["gpus"], chosen["price_per_gpu_hour_display"]))
    if chosen.get("data_center_id"):
        con.kv("datacenter", "%s (pinned; the create refuses elsewhere)"
               % chosen["data_center_id"])
    else:
        con.kv("datacenter", "UNPINNED -- RunPod re-offers slow hosts; "
               "--runpod-datacenter US-NC-1 measured 1.7-2.4 GB/s vs 240 MB/s "
               "(docs/CLOUD-RECIPES.md)")
    con.kv("all-in hard cap", "$%s (calculated $%s) -- the BOUND: GPU rate x "
           "(workload deadline + retrieval/delete reserve) + storage; not the estimate"
           % (quote.hard_cap_usd, quote.calculated_maximum_usd()))
    if derivation.get("basis") == "components_seconds":
        # The authored components without the margin are the run the row
        # measured; that is the only spend estimate this tool will state.
        measured = (Decimal(derivation["fetch"]) + Decimal(derivation["setup"])
                    + Decimal(derivation["cold_run"]) * derivation["captures"]
                    + Decimal(derivation["verify_compare_qualify"]))
        con.kv("expected spend",
               "~$%s for ~%d min at $%s/h -- the authored components without the "
               "margin (the row's measured run); the cap above is the bound"
               % ((measured / 3600 * rate).quantize(Decimal("0.01")),
                  int(measured / 60), rate))
    else:
        con.kv("expected spend", "not stated: no authored components for this "
               "target; the cap above is the bound, and the receipts of prior "
               "runs of this route are the only estimate (docs/THIRD-PARTY-QUICKSTART.md 3b)")
    con.kv("storage", "%s: container disk %d GB, pod volume %d GB, "
           "run root under %s" % (
               plan_data["storage_layout"], plan_data["container_disk_gb"],
               plan_data["storage_gb"],
               RUNPOD_STORAGE_LAYOUTS[plan_data["storage_layout"]]["run_base"]))
    con.kv("workload bound", "%s s (%s); retrieval/delete reserve %s s (%s)"
           % (quote.workload_deadline_seconds,
              plan_data.get("max_runtime_source", "--max-runtime"),
              quote.retrieval_delete_reserve_seconds,
              plan_data.get("retrieval_delete_reserve_source",
                            "--retrieval-delete-reserve")))
    con.kv("campaign", (
        "per-run ledger %s (ceiling = --max-cost; foreign pods tolerated)"
        % Path(campaign_path).name
        if plan_data["campaign_mode"] == "per-run" else
        "explicit ledger %s (ceiling $%s, reserve $%s, reaper margin $%s)"
        % (Path(campaign_path).name, args.campaign_ceiling,
           args.campaign_reserve, args.campaign_reaper_margin)))
    for name in sorted(plan_data["gates"]):
        con.ok(name)
    for line in plan_data["warnings"]:
        con.warn("note: " + line)
    return plan_data


def plan_runpod(args, con: Console, provider) -> Dict[str, Any]:
    """Plan using only the same anonymous official-Hub access the pod has."""
    with _anonymous_hf_environment() as anonymous_access:
        return _plan_runpod_anonymous(
            args, con, provider, anonymous_access)


def _ledger_transition(ledger, method: str, *args, **kwargs):
    for _unused in range(8):
        result = getattr(ledger, method)(
            ledger.snapshot()["generation"], *args, **kwargs)
        if result.code != "GENERATION_CONFLICT":
            return result
    raise Refusal("campaign ledger remained generation-conflicted", [])


def _authenticate_runpod_ssh_host(
        con: Console, provider, pod_id: str, outdir: Path) -> Dict[str, Any]:
    """Authenticate the first SSH key against the provider's HTTPS log API."""
    provider.set_known_hosts(outdir / "ssh_known_hosts")
    con.say("Authenticating RunPod ED25519 host key from provider logs...")
    log_evidence = provider.ssh_host_ed25519_fingerprint(pod_id)
    verified = provider.verify_host_key(
        pod_id, log_evidence["fingerprint"])
    proof = {
        "schema": "fidelity-suite/runpod-ssh-host-key-proof.v2",
        "provider": "runpod",
        "provider_id": str(pod_id),
        "verified_at_utc": _exact_utc_now(),
        "verification_source": "runpod-authenticated-v2-container-log",
        "provider_log_endpoint_origin": log_evidence["endpoint_origin"],
        "provider_log_source": log_evidence["source"],
        "provider_log_tail": log_evidence["tail"],
        "provider_log_observed_at_utc": log_evidence["observed_at_utc"],
        "provider_log_line_sha256": log_evidence["line_sha256"],
        "provider_log_line": log_evidence["line"],
        "provider_log_fingerprint": log_evidence["fingerprint"],
        "algorithm": verified.get("algorithm"),
        "fingerprint": verified.get("fingerprint"),
        "host": verified.get("host"),
        "port": verified.get("port"),
        "known_hosts_sha256": sha256_file(str(outdir / "ssh_known_hosts")),
        "proof_sha256": "",
    }
    proof["proof_sha256"] = hashlib.sha256(
        _canonical_bytes(proof)).hexdigest()
    path = outdir / "runpod-ssh-host-key-proof.json"
    with path.open("xb") as stream:
        stream.write(
            json.dumps(
                proof, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": path, "proof": proof}


def _bind_lease_cleanup_to_campaign(
        ledger_path: str, lease: Dict[str, Any]) -> None:
    """Bind the lease's complete exact ID set without projecting terminal proof."""
    from fidelity.campaign import CampaignLedger
    from fidelity.cloudlease import campaign_cleanup_binding_evidence
    request = (lease.get("create") or {}).get("request") or {}
    key = request.get("campaign_attempt_key")
    ids = sorted(set(str(value)
                     for value in lease.get("provider_resource_ids") or []))
    if not isinstance(key, str) or not ids:
        return
    ledger = CampaignLedger(
        ledger_path, request.get("provider"),
        request.get("provider_account_id"))
    item = (ledger.snapshot().get("attempts") or {}).get(key)
    if item is None:
        return
    if not item.get("provider_ids"):
        evidence = campaign_cleanup_binding_evidence(lease, ids)
        bound = _ledger_transition(
            ledger, "bind_provider_for_cleanup", key, ids, evidence)
        if bound.code not in (
                "PROVIDER_BOUND_FOR_CLEANUP",
                "PROVIDER_CLEANUP_BINDING_UNCHANGED"):
            raise Refusal("campaign cleanup binding failed: %s"
                          % bound.message, [])
        item = (ledger.snapshot().get("attempts") or {}).get(key) or {}
    if sorted(item.get("provider_ids") or []) != ids:
        raise Refusal("campaign and lease exact provider ID sets differ", [])


def _runpod_secrets_dir(fs_root: str) -> str:
    """Where the RunPod path keeps the HF token on the pod.

    NOT under fs_root: /workspace is the pod volume, and it accepted
    `chmod 600` while reporting 0666 (Fruit smoke, 2026-09-03). The
    container's own disk honours modes, so the 0700/0600 contract lives
    there, keyed by the attempt so two attempts can never share a path.
    """
    return "/root/.fidelity-secrets/%s" % hashlib.sha256(
        fs_root.encode("utf-8")).hexdigest()[:24]


def _cleanup_remote_secret(provider, pod_id, fs_root, *, secrets_dir=None):
    secret_dir = secrets_dir or "%s/.secrets" % fs_root
    secret = "%s/hf_token" % secret_dir
    command = (
        "set -eu; "
        "[ ! -L {directory} ]; [ ! -L {secret} ]; "
        "if [ -f {secret} ]; then "
        "(command -v shred >/dev/null && shred -u -- {secret}) "
        "|| rm -f -- {secret}; fi; "
        "rm -rf -- {directory}; "
        "[ ! -e {directory} ] && [ ! -L {directory} ]"
    ).format(directory=shlex.quote(secret_dir), secret=shlex.quote(secret))
    try:
        provider.exec(pod_id, command, timeout=60)
        return {"confirmed": True, "path": secret}
    except Exception as exc:
        return {"confirmed": False, "path": secret,
                "error": redact(str(exc))}


def _freeze_verified_bundle(bundle, outdir: Path) -> Dict[str, Any]:
    """Seal every bundle byte before any campaign reservation/provider POST."""
    local = outdir / ".frozen-bundle"
    local.mkdir(mode=0o700)
    archive = local / "bundle.tar.gz"
    manifest_path = local / "manifest.json"
    manifest_bytes = _canonical_bytes(bundle)
    manifest_path.write_bytes(manifest_bytes)
    helper_paths = {}
    helper_names = ("__init__.py", "jobcontract.py", "runpodsafety.py")
    with archive.open("xb") as output:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(
                    fileobj=compressed, mode="w",
                    format=tarfile.USTAR_FORMAT) as tar:
                for row in bundle["files"]:
                    data = (SUITE_ROOT / row["path"]).read_bytes()
                    if (len(data) != row["bytes"]
                            or hashlib.sha256(data).hexdigest()
                            != row["sha256"]):
                        raise RuntimeError(
                            "bundle source changed before paid admission: %s"
                            % row["path"])
                    info = tarfile.TarInfo(row["path"])
                    info.size = len(data); info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""; info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                    frozen_file = local / "suite" / row["path"]
                    frozen_file.parent.mkdir(parents=True, exist_ok=True)
                    frozen_file.write_bytes(data)
                    name = PurePosixPath(row["path"]).name
                    if row["path"] == "bin/fidelity/%s" % name and name in helper_names:
                        helper = local / name
                        helper.write_bytes(data)
                        helper_paths[name] = {
                            "path": str(helper), "sha256": row["sha256"]}
        output.flush()
        os.fsync(output.fileno())
    if set(helper_paths) != set(helper_names):
        raise RuntimeError("frozen bundle lacks remote verification helpers")
    return {
        "archive_path": str(archive),
        "archive_sha256": sha256_file(str(archive)),
        "archive_bytes": archive.stat().st_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "helpers": helper_paths,
        "suite_root": str(local / "suite"),
    }


def _resume_capture_identity(args) -> Optional[Dict[str, Any]]:
    """The job-side identity of the sealed dataset --resume-capture names.

    Cheap and exact: the dataset manifest's own seal and content digest, the
    manifest file's digest, and the origin job if one is named. The full
    identity gate (tensors recomputed, every recipe field held to this job's
    contract) runs once the job is finalized, in _validate_resume_capture.
    """
    if not getattr(args, "resume_capture", None):
        return None
    root = Path(args.resume_capture).expanduser().resolve()
    manifest_path = root / "fidelity-dataset.json"
    if (not root.is_dir() or root.is_symlink() or manifest_path.is_symlink()
            or not manifest_path.is_file()):
        raise Refusal("--resume-capture must be a sealed dataset directory", [
            "it needs fidelity-dataset.json at its top level"])
    raw = manifest_path.read_bytes()
    try:
        doc = parse_job_bytes(raw)
    except JobContractError as exc:
        raise Refusal("--resume-capture manifest is not strict JSON: %s" % exc, [])
    dataset_sha = str(doc.get("dataset_sha256") or "")
    digest = str(((doc.get("capture") or {}) if isinstance(doc, dict)
                  else {}).get("capture_content_digest") or "")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in (dataset_sha, digest)):
        raise Refusal("--resume-capture dataset is not sealed", [
            "fidelity-dataset.json must carry a 64-hex dataset_sha256 and "
            "capture.capture_content_digest"])
    origin = None
    if getattr(args, "resume_origin_job", None):
        origin_path = Path(args.resume_origin_job).expanduser().resolve()
        if origin_path.is_symlink() or not origin_path.is_file():
            raise Refusal("--resume-origin-job must be a regular job.json", [])
        origin_raw = origin_path.read_bytes()
        try:
            origin_job = parse_job_bytes(origin_raw)
        except JobContractError as exc:
            raise Refusal("--resume-origin-job is not strict JSON: %s" % exc, [])
        attempt = (origin_job.get("execution_attempt") or {})
        origin = {
            "job_id_full": str(origin_job.get("job_id_full") or ""),
            "attempt_id": str(attempt.get("attempt_id") or ""),
            "job_file_sha256": hashlib.sha256(origin_raw).hexdigest(),
        }
        if (re.fullmatch(r"[0-9a-f]{64}", origin["job_id_full"]) is None
                or re.fullmatch(r"[0-9a-f]{24}", origin["attempt_id"]) is None):
            raise Refusal("--resume-origin-job lacks an executed job identity", [])
    resealed = (doc.get("dataset") or {}).get("resealed")
    resealed_from = None
    if resealed is not None:
        # A re-sealed import (fidelity-dataset reseal) names the seal it came
        # from; the job carries that chain so every later receipt can too.
        if (not isinstance(resealed, dict)
                or resealed.get("schema") != "fidelity-dataset.reseal.v1"
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(resealed.get("from_dataset_sha256", ""))) is None
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(resealed.get("receipt_sha256", ""))) is None
                or resealed.get("receipt") != "validation/reseal-receipt.json"
                or not isinstance(resealed.get("reason"), str)):
            raise Refusal("--resume-capture dataset.resealed block is not the "
                          "v1 reseal identity", [])
        receipt_path = root / "validation" / "reseal-receipt.json"
        if (receipt_path.is_symlink() or not receipt_path.is_file()
                or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                != resealed["receipt_sha256"]):
            raise Refusal("--resume-capture reseal receipt is missing or differs "
                          "from the manifest's receipt_sha256", [])
        resealed_from = {
            "dataset_sha256": resealed["from_dataset_sha256"],
            "reason": resealed["reason"],
            "receipt": resealed["receipt"],
            "receipt_sha256": resealed["receipt_sha256"],
        }
    return {
        "dataset_sha256": dataset_sha,
        "capture_content_digest": digest,
        "dataset_manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "origin": origin,
        "resealed_from": resealed_from,
    }


def _validate_resume_capture(args, plan_data: Dict[str, Any], con: Console) -> None:
    """Hold the imported cold run 1 to exactly this job's recipe, before spend.

    Reuses the qualifier's own capture-vs-job check (fidelity_dataset
    _check_capture_job_contract) so the plan cannot admit a dataset the pod's
    qualify_root would later refuse, and adds the two facts a bitwise
    reproduction depends on that the recipe does not name: the same
    container image digest and the same GPU model.
    """
    resume = plan_data["job"]["capture"].get("resume_capture")
    if resume is None:
        return
    import fidelity_dataset as FD
    root = Path(args.resume_capture).expanduser().resolve()
    con.say("Verifying the resumed cold run 1 (tensors recomputed)...")
    try:
        identity = FD._capture_identity(str(root), "root-cold-1", "canonical")
        FD._check_capture_job_contract(
            plan_data["job"], identity, "resumed canonical")
    except FD.RootQualificationError as exc:
        raise Refusal("--resume-capture does not match this job's recipe: %s"
                      % exc, ["capture cold run 1 fresh instead, or point "
                              "--resume-capture at a dataset of this recipe"])
    target = plan_data["job"]["target"]
    if (identity["weights_repository"] != target["repo_id"]
            or identity["weights_revision"] != target["revision"]):
        raise Refusal("--resume-capture was captured from different weights", [])
    if (identity["dataset_sha256"] != resume["dataset_sha256"]
            or identity["capture_content_digest"]
            != resume["capture_content_digest"]):
        raise Refusal("--resume-capture changed between planning steps", [])
    image = plan_data["job"]["environment"]["image"]
    container = identity.get("runtime_container") or {}
    if (container.get("image_reference") != image
            or container.get("image_digest") != image.rsplit("@", 1)[1]):
        raise Refusal(
            "--resume-capture ran in a different container image (%s); a "
            "bitwise reproduction is only expected on the same image"
            % container.get("image_reference"), [])
    manifest = json.loads((root / "fidelity-dataset.json").read_text("utf-8"))
    runtime = json.loads((root / manifest["runtime"]["file"]).read_text("utf-8"))
    device_name = (runtime.get("stack_fingerprint") or {}).get("device_name")
    expected_gpu = plan_data["chosen"]["provider_gpu_display"]
    if device_name != expected_gpu:
        raise Refusal(
            "--resume-capture ran on %r; this plan rents %r, and a bitwise "
            "reproduction is only expected on the same GPU model"
            % (device_name, expected_gpu), ["pass --gpu to match"])
    plan_data["resume_capture"] = dict(resume, path=str(root),
                                       process_label=identity["process_label"],
                                       device_name=device_name)
    gate_verified(plan_data, "resume-capture-identity",
                  dataset_sha256=resume["dataset_sha256"],
                  capture_content_digest=resume["capture_content_digest"],
                  device_name=device_name, origin=resume.get("origin"))


def _freeze_resume_capture(plan_data: Dict[str, Any], outdir: Path) -> Optional[Dict[str, Any]]:
    """Archive the imported cold run 1 exactly, with a verified-file manifest."""
    resume = plan_data.get("resume_capture")
    if resume is None:
        return None
    root = Path(resume["path"])
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Refusal("--resume-capture contains a symlink: %s" % path, [])
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            data = path.read_bytes()
            rows.append({"path": rel, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
    manifest = finalize_bundle_manifest(rows, "resume-capture")
    if not any(row["path"] == "fidelity-dataset.json"
               and row["sha256"] == resume["dataset_manifest_file_sha256"]
               for row in rows):
        raise Refusal("--resume-capture changed before paid admission", [])
    local = outdir / ".frozen-inputs"
    local.mkdir(mode=0o700, exist_ok=True)
    archive = local / "resume-capture.tar.gz"
    manifest_path = local / "resume-capture.manifest.json"
    manifest_bytes = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    with archive.open("xb") as output:
        # Level 1: bf16 tensors barely compress and the pod verifies bytes,
        # not the container; the point is one exact stream, not size.
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0,
                           compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w",
                              format=tarfile.USTAR_FORMAT) as tar:
                for row in rows:
                    data = (root / row["path"]).read_bytes()
                    if (len(data) != row["bytes"]
                            or hashlib.sha256(data).hexdigest() != row["sha256"]):
                        raise Refusal("--resume-capture changed while freezing", [])
                    info = tarfile.TarInfo(row["path"])
                    info.size = len(data); info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""; info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
        output.flush()
        os.fsync(output.fileno())
    return {
        "archive_path": str(archive),
        "archive_sha256": sha256_file(str(archive)),
        "archive_bytes": archive.stat().st_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": len(rows),
    }


def _import_resume_capture(provider, pod_id, fs_root, plan_data, job,
                           frozen, outdir: Path, con: Console) -> Dict[str, Any]:
    """Land the frozen cold run 1 in {fs}/dataset exactly, then seal a receipt."""
    resume = plan_data["resume_capture"]
    run_base = fs_root.rsplit("/fidelity/", 1)[0]
    suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", str(pod_id))
    remote_archive = "%s/.fidelity-resume-%s.tar.gz" % (run_base, suffix)
    remote_manifest = "%s/.fidelity-resume-%s.json" % (run_base, suffix)
    con.say("Importing the resumed cold run 1 (%.2f GB)..."
            % (frozen["archive_bytes"] / 1e9))
    # The archive's bound is its own frozen size: a root dataset is 2.5 GB
    # for GLM-5.3 and the transport's default 512 MiB cap is for bundles.
    provider.upload(pod_id, frozen["archive_path"], remote_archive,
                    max_bytes=int(frozen["archive_bytes"]))
    provider.upload(pod_id, frozen["manifest_path"], remote_manifest)
    provider.exec(
        pod_id,
        "PYTHONPATH={fs}/bin python3 -m fidelity.runpodsafety extract-bundle "
        "--archive {archive} --manifest {manifest} "
        "--destination {fs}/dataset --sha256 {sha} --bytes {size} "
        "&& rm -f -- {archive} {manifest}".format(
            fs=shlex.quote(fs_root), archive=shlex.quote(remote_archive),
            manifest=shlex.quote(remote_manifest),
            sha=frozen["archive_sha256"], size=frozen["archive_bytes"]),
        timeout=1800)
    receipt = build_imported_capture_receipt(
        job_id_full=job["job_id_full"],
        attempt_id=job["execution_attempt"]["attempt_id"],
        resume={key: resume[key] for key in (
            "dataset_sha256", "capture_content_digest",
            "dataset_manifest_file_sha256", "origin", "resealed_from")},
        archive_sha256=frozen["archive_sha256"],
        archive_bytes=frozen["archive_bytes"],
        manifest_sha256=frozen["manifest_sha256"],
        file_count=frozen["file_count"], source_path=resume["path"],
        imported_at=_exact_utc_now())
    local = outdir / "imported-capture.json"
    local.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    provider.upload(pod_id, str(local), "%s/receipts/imported-capture.json" % fs_root)
    con.ok("cold run 1 imported",
           "dataset_sha256 %s; cold run 2 is captured fresh and must "
           "reproduce it bitwise" % resume["dataset_sha256"][:16])
    return receipt

def _freeze_root_inputs(plan_data: Dict[str, Any], outdir: Path) -> None:
    """Copy the two job-bound panel files before any campaign reservation."""
    destination = outdir / ".frozen-inputs"
    destination.mkdir(mode=0o700)
    rows = [
        ("_panel_archive_local", "panel.tar",
         plan_data["job"]["panel"]["archive_sha256"],
         plan_data["job"]["panel"]["archive_bytes"]),
        ("_panel_binding_local", "panel.binding.json",
         plan_data["job"]["panel"]["binding_file_sha256"], None),
    ]
    candidate = (plan_data["job"]["capture"] or {}).get("candidate")
    if candidate is not None:
        rows.append(("_candidate_scope_local", "candidate-scope.json",
                     candidate["scope"]["sha256"], None))
    if plan_data.get("_derived_allowlist_local"):
        rows.append(("_derived_allowlist_local", "allowlist.json",
                     plan_data["job"]["capture"]["unexpected_tensor_allowlist"]["artifact_sha256"],
                     None))
    for key, name, expected_sha, expected_bytes in rows:
        source = Path(plan_data[key])
        metadata = source.lstat()
        if (source.is_symlink() or not source.is_file()
                or metadata.st_uid != os.getuid()):
            raise RuntimeError(
                "root panel input is not an owned regular file: %s" % source)
        data = source.read_bytes()
        if (hashlib.sha256(data).hexdigest() != expected_sha
                or (expected_bytes is not None
                    and len(data) != int(expected_bytes))):
            raise RuntimeError(
                "root panel input changed before paid admission: %s" % source)
        frozen = destination / name
        with frozen.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        plan_data[key] = str(frozen)
    directory_fd = os.open(str(destination), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)



def _verify_frozen_suite(frozen: Dict[str, Any],
                         bundle: Dict[str, Any]) -> None:
    """Revalidate every local executable byte against its pre-spend seal."""
    manifest = Path(frozen["manifest_path"])
    if (sha256_file(str(manifest)) != frozen["manifest_sha256"]
            or json.loads(manifest.read_text(encoding="utf-8")) != bundle):
        raise RuntimeError("frozen bundle manifest changed after paid admission")
    suite = Path(frozen["suite_root"])
    for row in bundle["files"]:
        path = suite / row["path"]
        metadata = path.lstat()
        if (path.is_symlink() or not path.is_file()
                or metadata.st_uid != os.getuid()
                or metadata.st_size != row["bytes"]
                or sha256_file(str(path)) != row["sha256"]):
            raise RuntimeError(
                "frozen suite byte changed after paid admission: %s"
                % row["path"])

def _upload_verified_bundle(provider, pod_id, fs_root, bundle, frozen):
    archive = Path(frozen["archive_path"])
    manifest_path = Path(frozen["manifest_path"])
    archive_sha = frozen["archive_sha256"]
    archive_bytes = frozen["archive_bytes"]
    if (sha256_file(str(archive)) != archive_sha
            or archive.stat().st_size != archive_bytes
            or sha256_file(str(manifest_path)) != frozen["manifest_sha256"]):
        raise RuntimeError("frozen bundle changed before transfer")
    suffix = re.sub(r"[^A-Za-z0-9_-]", "_", str(pod_id))
    # Beside the run root, on the same disk, so the extractor's atomic
    # rename never crosses filesystems.
    run_base = fs_root.rsplit("/fidelity/", 1)[0]
    remote_archive = "%s/.fidelity-bundle-%s.tar.gz" % (run_base, suffix)
    remote_manifest = "%s/.fidelity-manifest-%s.json" % (run_base, suffix)
    remote_helper_dir = "%s/.fidelity-helper-%s" % (run_base, suffix)
    provider.exec(
        pod_id,
        "set -eu; test ! -e {root}; test ! -L {root}; "
        "test ! -e {archive}; test ! -e {manifest}; "
        "test ! -e {helper}; mkdir -m 700 -- {helper}; "
        "mkdir -m 700 -- {helper}/fidelity".format(
            root=shlex.quote(fs_root), archive=shlex.quote(remote_archive),
            manifest=shlex.quote(remote_manifest),
            helper=shlex.quote(remote_helper_dir)))
    provider.upload(pod_id, str(archive), remote_archive)
    provider.upload(pod_id, str(manifest_path), remote_manifest)
    for name, helper in frozen["helpers"].items():
        provider.upload(
            pod_id, helper["path"],
            "%s/fidelity/%s" % (remote_helper_dir, name))
    transfer_checks = [
        (remote_archive, archive_sha),
        (remote_manifest, frozen["manifest_sha256"]),
    ] + [
        ("%s/fidelity/%s" % (remote_helper_dir, name), helper["sha256"])
        for name, helper in sorted(frozen["helpers"].items())
    ]
    provider.exec(
        pod_id, "set -eu; " + "; ".join(
            "test \"$(sha256sum {path} | cut -d' ' -f1)\" = {digest}".format(
                path=shlex.quote(path), digest=digest)
            for path, digest in transfer_checks))
    provider.exec(
        pod_id,
        "PYTHONPATH={helper} python3 -m fidelity.runpodsafety "
        "extract-bundle --archive {archive} --manifest {manifest} "
        "--destination {root} --sha256 {sha} --bytes {size}".format(
            helper=shlex.quote(remote_helper_dir),
            archive=shlex.quote(remote_archive),
            manifest=shlex.quote(remote_manifest),
            root=shlex.quote(fs_root), sha=archive_sha, size=archive_bytes),
        timeout=1800)
    provider.exec(
        pod_id, "rm -rf -- %s; rm -f -- %s %s" % (
            shlex.quote(remote_helper_dir), shlex.quote(remote_archive),
            shlex.quote(remote_manifest)))
def _cleanup_ambiguous_runpod_create(
        provider, lease_store, lease_ref, ledger, campaign_key):
    """Bind, terminate and reconcile every exact ambiguous create candidate."""
    from fidelity.cloudlease import (
        ABSENCE_CONFIRMED, campaign_cleanup_binding_evidence,
        finalize_campaign_after_absence, runpod_authoritative_listing)
    lease = lease_store.read(lease_ref)
    ids = sorted(set(str(value)
                     for value in lease.get("provider_resource_ids") or []))
    if not ids:
        raise RuntimeError(
            "ambiguous RunPod create has no exact cleanup candidate; "
            "lease liability retained")
    evidence = campaign_cleanup_binding_evidence(lease, ids)
    item = (ledger.snapshot().get("attempts") or {}).get(campaign_key) or {}
    if not item.get("provider_ids"):
        bound = _ledger_transition(
            ledger, "bind_provider_for_cleanup",
            campaign_key, ids, evidence)
        if bound.code not in (
                "PROVIDER_BOUND_FOR_CLEANUP",
                "PROVIDER_CLEANUP_BINDING_UNCHANGED"):
            raise RuntimeError(
                "cannot bind ambiguous candidates for cleanup: %s"
                % bound.message)
    current = lease_store.read(lease_ref)
    if current["state"] not in ("DESTROYING", "ABSENCE_CONFIRMED", "TERMINAL"):
        lease_ref = lease_store.request_destroy(
            lease_ref, {"reason": "ambiguous create cleanup",
                        "provider_ids": ids})
    destroy_errors = []
    for provider_id in ids:
        try:
            provider.destroy(provider_id)
        except Exception as exc:
            destroy_errors.append("%s: %s" % (provider_id, redact(str(exc))))
    expected_account_id = str(
        (lease.get("create") or {}).get("request", {}).get(
            "provider_account_id") or "")
    for _unused in range(20):
        status = provider.status()
        observed_account_id = str(status.get("id") or "").strip()
        if observed_account_id != expected_account_id:
            raise RuntimeError(
                "ambiguous cleanup cannot verify the lease provider account")
        graphql_pods = provider.list_lifecycle_resources()
        inventory = provider.chargeable_inventory()
        listing, absence_proof = runpod_authoritative_listing(
            provider, graphql_pods, observed_account_id,
            inventory=inventory)
        lease_ref = lease_store.confirm_exact_absence(
            lease_ref, listing, authoritative_inventory=absence_proof)
        if lease_ref.state == ABSENCE_CONFIRMED:
            break
        time.sleep(3)
    if lease_ref.state != ABSENCE_CONFIRMED:
        raise RuntimeError(
            "ambiguous RunPod candidates remain chargeable; liability retained"
            + ("; destroy errors: " + "; ".join(destroy_errors)
               if destroy_errors else ""))
    billing = provider.reconcile_billing(lease_store.read(lease_ref))
    lease_ref = lease_store.stage_billing_reconciliation(
        lease_ref, billing)
    finalize_campaign_after_absence(
        provider, lease_store.read(lease_ref), lease_store.root)
    lease_ref = lease_store.record_billing_reconciled(
        lease_ref, billing)
    released = (
        (ledger.snapshot().get("attempts") or {})
        .get(campaign_key, {}).get("released"))
    if released is not True:
        raise RuntimeError(
            "ambiguous-create campaign liability was not durably released")
    return lease_ref


def _runpod_stage_command(fs_root, engine_root, stage, image_digest,
                          image_reference, secrets_dir):
    """The on-pod shell command that launches a setsid stage leader and waits
    for its self-recorded process-group id (or the leader's exit), bounded.

    The leader (stage_measure.sh) self-records its process group as its first
    act under setsid; this wrapper waits for that record so a stage that
    finishes faster than the SSH round-trip is a success, not a spurious
    exit 70.  A live leader with no record after the bound is a genuine
    record failure (TERM the group, exit 70).  A dead leader with no record
    propagates its own exit code (it failed before it could record)."""
    return (
        "set -eu; "
        # The record is per-STAGE but the path is per-run: without clearing
        # it, from the second stage on the wait below is satisfied at once by
        # the PREVIOUS leader's record (a stale pgid until this leader
        # overwrites it moments later).  Clear it so the wait is for this
        # stage's own self-record.
        "rm -f {fs}/runtime/stage.pgid; "
        "setsid env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN "
        "-u HUGGINGFACE_HUB_TOKEN -u HF_HUB_OFFLINE "
        "-u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE "
        "-u HUGGINGFACE_CO_STAGING -u HUGGINGFACE_CO_URL_TEMPLATE "
        "-u HF_INFERENCE_ENDPOINT -u HF_HUB_CACHE "
        "-u HUGGINGFACE_HUB_CACHE -u HF_ASSETS_CACHE "
        "-u HUGGINGFACE_ASSETS_CACHE -u HF_XET_CACHE "
        "-u TRANSFORMERS_CACHE -u HF_DATASETS_CACHE -u XDG_CACHE_HOME "
        "HF_ENDPOINT=https://huggingface.co "
        "HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH={fs}/.no-token "
        "HF_HOME={fs}/hf-anonymous "
        "FIDELITY_FS_ROOT={fs} FIDELITY_SUITE_ROOT={fs} "
        "FIDELITY_ENGINE_ROOT={engine} "
        "FIDELITY_SECRETS_DIR={secrets} "
        "STACKPRINT_IMAGE_PIN={image_digest} "
        "FIDELITY_IMAGE_REFERENCE={image_reference} "
        "bash {fs}/bin/stage_measure.sh {stage} "
        ">>{fs}/logs/stage-{stage}.log 2>&1 </dev/null & "
        "leader=$!; "
        "record={fs}/runtime/stage.pgid; "
        # The leader self-records its process group as its first act (see
        # stage_measure.sh).  Wait for that record to appear -- or for the
        # leader to exit -- so a stage that finishes faster than the SSH
        # round-trip is a success, not a spurious exit 70.  A live leader
        # with no record after the bound is a genuine record failure: TERM
        # the group and refuse.  A dead leader with no record propagates its
        # own exit code (it failed before it could record).
        "_wait_secs=${{STAGE_PGID_WAIT_SECS:-30}}; "
        "_i=0; "
        "while [ \"$_i\" -lt \"$((_wait_secs * 5))\" ]; do "
        "if [ -f \"$record\" ] || ! kill -0 \"$leader\" 2>/dev/null; then "
        "break; fi; sleep 0.2; _i=$((_i + 1)); done; "
        "if [ -f \"$record\" ]; then "
        "_code=0; wait \"$leader\" || _code=$?; exit \"$_code\"; "
        "elif kill -0 \"$leader\" 2>/dev/null; then "
        "kill -TERM -- \"-$leader\" 2>/dev/null || true; "
        "_code=0; wait \"$leader\" 2>/dev/null || _code=$?; exit 70; "
        "else "
        "_code=0; wait \"$leader\" 2>/dev/null || _code=$?; "
        "exit \"$_code\"; fi"
    ).format(
        fs=shlex.quote(fs_root), engine=shlex.quote(engine_root),
        secrets=shlex.quote(secrets_dir),
        stage=shlex.quote(stage),
        image_digest=shlex.quote(image_digest),
        image_reference=shlex.quote(str(image_reference)))


def _runpod_stage(
        provider, pod_id, fs_root, engine_root, stage, deadline,
        image_reference, progress=None):
    image_match = re.fullmatch(
        r".+@(sha256:[0-9a-f]{64})", str(image_reference))
    if image_match is None:
        raise RuntimeError("stage image reference is not immutable")
    image_digest = image_match.group(1)
    command = _runpod_stage_command(
        fs_root, engine_root, stage, image_digest, str(image_reference),
        _runpod_secrets_dir(fs_root))
    run = provider.run_job(pod_id, command)
    run_id = (run or {}).get("run_id") or (run or {}).get("id")
    if not run_id:
        raise RuntimeError("stage launch returned no run id")
    # The transport returns "unknown" when the PROBE could not run -- an ssh
    # timeout, a proxy hiccup -- and documents that it is never evidence of
    # death.  Treating it as terminal aborted a multi-hour capture on one
    # thirty-second blip, the failure class that lost drill #5.  Only an
    # evidence-based verdict ends the stage immediately; probe failures are
    # tolerated for STAGE_PROBE_OUTAGE_SECONDS, after which the host is
    # treated as unreachable and the run is torn down.
    probe_outage_started = None
    started = time.time()
    last_progress = started
    last_bytes = None
    while time.time() < deadline:
        status = provider.run_status(run_id, machine_id=pod_id)
        state = str(status.get("state") or "").lower()
        if state == "succeeded":
            return
        if state in ("failed", "error"):
            raise RuntimeError(
                "stage %s ended in %s (%s); recovery is refused; evidence:\n%s"
                % (stage, state,
                   ", ".join("%s=%s" % (key, status[key])
                             for key in ("exit_code", "note")
                             if status.get(key) is not None) or "no detail",
                   _runpod_stage_failure_evidence(
                       provider, pod_id, fs_root, stage, run_id)))
        if state == "unknown":
            if probe_outage_started is None:
                probe_outage_started = time.time()
            elif (time.time() - probe_outage_started
                  > STAGE_PROBE_OUTAGE_SECONDS):
                raise RuntimeError(
                    "stage %s liveness probe has failed for %d s (%s); the "
                    "pod is treated as unreachable"
                    % (stage, STAGE_PROBE_OUTAGE_SECONDS,
                       status.get("note") or "no detail"))
        else:
            probe_outage_started = None
        # A three-hour stage that prints nothing is a run nobody can judge
        # (2026-09-03: the deadline decision was made from a manual ssh
        # probe). One line every STAGE_PROGRESS_SECONDS: bytes landed and
        # rate for a fetch, the engine's last progress line otherwise, and
        # the time left before the workload deadline.
        if progress is not None and time.time() - last_progress >= (
                STAGE_PROGRESS_SECONDS):
            now = time.time()
            line, landed = _runpod_stage_progress(
                provider, pod_id, fs_root, stage)
            detail = []
            if landed is not None:
                rate = ((landed - last_bytes) / (now - last_progress)
                        if last_bytes is not None else None)
                detail.append("%.1f GB on disk%s" % (
                    landed / 1e9,
                    " (%.0f MB/s)" % (rate / 1e6) if rate else ""))
                last_bytes = landed
            if line:
                detail.append(line)
            progress("stage %s +%dm, %dm before the workload deadline: %s" % (
                stage, (now - started) // 60, max(0, deadline - now) // 60,
                " | ".join(detail) or "no output yet"))
            last_progress = now
        time.sleep(15)
    raise RuntimeError("workload deadline reached during stage %s" % stage)


def _runpod_stage_progress(provider, pod_id, fs_root, stage):
    """Best-effort (last engine log line, bytes landed) for a running stage."""
    fs = shlex.quote(fs_root)
    command = (
        "tail -c 4000 %s/logs/%s.log 2>/dev/null | tr '\\r' '\\n' | grep . "
        "| tail -1 | cut -c1-220; echo; "
        "du -sb %s/models/target 2>/dev/null | cut -f1"
        % (fs, shlex.quote(stage), fs))
    try:
        result = provider.exec(pod_id, command, timeout=60, check=False)
    except Exception:  # noqa: BLE001 -- a progress line is never a verdict
        return None, None
    lines = str((result or {}).get("stdout") or "").split("\n")
    line = redact(lines[0].strip()) if lines else ""
    landed = None
    if stage == "fetch_target" and len(lines) > 2 and lines[2].strip().isdigit():
        landed = int(lines[2].strip())
    return (line or None), landed


def _runpod_stage_failure_evidence(provider, pod_id, fs_root, stage, run_id,
                                   max_bytes: int = 6000) -> str:
    """Best-effort evidence of a failed stage, read BEFORE the pod is destroyed.

    The pod is the only place any of this exists; teardown follows the
    failure within seconds, and a bare "stage setup ended in failed" cost a
    rental to diagnose (Fruit smoke, 2026-09-03). Each item is one shell
    command whose failure is itself evidence, never an exception.
    """
    fs = shlex.quote(fs_root)
    run_dir = "%s/%s" % (provider.RUNS, run_id)
    items = (
        ("stage log tail",
         "tail -c %d %s/logs/stage-%s.log" % (
             int(max_bytes), fs, shlex.quote(stage))),
        ("launch wrapper output", "tail -c 2000 %s/output.log" % run_dir),
        ("launch wrapper state",
         "ls -la %s; echo exit_code=$(cat %s/exit_code 2>/dev/null); "
         "echo pid=$(cat %s/pid 2>/dev/null)" % (run_dir, run_dir, run_dir)),
        ("watchdog log tail", "tail -c 2000 %s/logs/watchdog.log" % fs),
        ("ABANDONED.json", "cat %s/receipts/ABANDONED.json" % fs),
        ("run root", "ls -la %s %s/logs %s/runtime %s/receipts" % (
            fs, fs, fs, fs)),
    )
    parts = []
    for label, command in items:
        try:
            result = provider.exec(pod_id, command + " 2>&1", timeout=90,
                                   check=False)
            text = str((result or {}).get("stdout") or "").strip()
        except Exception as exc:  # noqa: BLE001
            text = "(unavailable: %s)" % str(exc)[:300]
        parts.append("--- %s ---\n%s" % (label, text or "(empty)"))
    return redact("\n".join(parts))

def execute_runpod(
        args, con: Console, provider, plan_data,
        download_token: str) -> Dict[str, Any]:
    """One reserved attempt, one POST, one SSH pod and one archive download."""
    from fidelity.campaign import (
        CostQuote, attempt_key as campaign_attempt_key)
    from fidelity.cloudlease import (
        ABSENCE_CONFIRMED, CreateResponsePersistenceError, LeaseStore,
        campaign_cleanup_binding_evidence, exact_resource_name,
        finalize_campaign_after_absence, runpod_authoritative_listing,
        systemd_reaper_health, utc_iso, validate_lease_liability_scope,
        validate_unresolved_lease_scope)
    from fidelity.runpodsafety import (
        validate_current_public_root, validate_safety_proof,
        validate_width_two_root_archive)
    from fidelity.runpodapi import DEFAULT_IMAGE, RunPodCreateResponseError
    from fidelity.resultsink import extract_verified_archive, verify_archive
    if (not isinstance(download_token, str) or not download_token
            or any(character.isspace() for character in download_token)):
        raise Refusal("exact Hugging Face download token is unavailable", [])
    register_secret(download_token)


    outdir = Path(args.out).resolve()
    output_parent = outdir.parent
    parent_metadata = output_parent.lstat()
    if (not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()):
        raise Refusal(
            "output parent must be an existing owned non-symlink directory", [])
    chain_error = private_directory_chain_error(
        output_parent, owner_uid=os.getuid())
    if chain_error:
        raise Refusal(
            "output parent is not protected from cross-uid replacement",
            [chain_error, "choose an owned path with no group/world-writable "
             "non-sticky ancestor"])
    archive_bound = int(
        plan_data["retrieval_delete_contract"]["bound_archive_bytes"])
    uncompressed_bound = int(
        (plan_data["target"].get("root_capture_storage") or {}).get(
            "result_archive_max_uncompressed_bytes", archive_bound))
    if args.role == "root" and args.publish_root_to:
        # Preserve the qualified archive A and extracted tree U while building
        # the final archive A. HfApi streams the source tree; no second U exists.
        local_capacity_required = (
            2 * archive_bound + uncompressed_bound + 67108864)
    else:
        local_capacity_required = archive_bound + uncompressed_bound + 67108864
    filesystem = os.statvfs(str(output_parent))
    local_capacity_free = filesystem.f_bavail * filesystem.f_frsize
    if local_capacity_free < local_capacity_required:
        raise Refusal(
            "local output filesystem lacks archive and extraction capacity",
            ["required bytes: %d" % local_capacity_required,
             "available bytes: %d" % local_capacity_free])
    plan_data["local_output_capacity"] = {
        "required_free_bytes": local_capacity_required,
        "observed_free_bytes": local_capacity_free,
        "filesystem_device": parent_metadata.st_dev,
    }
    if (args.role == "root" and args.publish_root_to
            and args.publish_root_to != args.dataset_repository):
        raise Refusal(
            "root publication authorization differs from destination", [])
    campaign_mode = plan_data.get("campaign_mode") or "explicit"
    if campaign_mode == "explicit":
        ledger_file = Path(args.campaign_ledger).resolve()
        if ledger_file.parent != Path(args.lease_dir).resolve().parent:
            raise Refusal(
                "campaign ledger must be a sibling of the lease directory", [])
        ledger_path, ledger = _open_existing_runpod_campaign(
            args, plan_data["provider_account_id"])
    else:
        # One ledger per attempt, beside the lease directory where the
        # reaper resolves campaign coordinates (`campaign_coordinates` joins
        # the leaf to the lease root's parent).  A per-job name reopened a
        # previous attempt's history on retry and refused it for width or
        # ceiling that --dry-run had never predicted.
        attempt = secrets.token_hex(12)
        ledger_path, ledger = _create_auto_campaign_ledger(
            args, plan_data["provider_account_id"], plan_data["job_id_full"],
            attempt)
    foreign_tolerated = (
        ledger.foreign_resources_policy(ledger.snapshot()) == "tolerate")

    # Refresh all admission facts immediately under durable ledger locking.
    # Freeze every run input BEFORE the fresh status/inventory/balance
    # snapshots and the 30-second server-time window: archiving the 2.5 GB
    # resumed GLM-5.3 dataset (2026-09-04) outlived both, and the POST was
    # refused as expired.
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    outdir.chmod(0o700)
    frozen_bundle = _freeze_verified_bundle(plan_data["bundle"], outdir)
    if args.role == "root":
        _freeze_root_inputs(plan_data, outdir)
    frozen_resume = _freeze_resume_capture(plan_data, outdir)
    status = provider.status()
    current_account_id = str(status.get("id") or "").strip()
    if current_account_id != plan_data["provider_account_id"]:
        raise Refusal(
            "RunPod provider account changed after planning; freeze", [])
    fresh_health = systemd_reaper_health(
        state_dir=Path(args.reaper_state_dir),
        lease_dir=Path(args.lease_dir), provider="runpod",
        provider_account_id=current_account_id)
    if not fresh_health.get("ok"):
        raise Refusal(
            "the installed RunPod reaper stopped being healthy after planning",
            _reaper_health_remedy(fresh_health, args))
    if campaign_mode == "explicit":
        try:
            validate_unresolved_lease_scope(
                LeaseStore(Path(args.lease_dir)), fresh_health,
                provider="runpod", provider_account_id=current_account_id,
                campaign_ledger_path=Path(ledger_path))
        except Exception as exc:
            raise Refusal(
                "RunPod unresolved lease scope changed before paid admission: %s"
                % exc, [])
    else:
        from fidelity.cloudlease import LeaseError
        try:
            validate_lease_liability_scope(
                LeaseStore(Path(args.lease_dir)), provider="runpod",
                provider_account_id=current_account_id,
                allow_live=bool(getattr(args, "allow_unresolved_leases", False)))
        except LeaseError as exc:
            raise Refusal(str(exc), [])
    fresh_safety = {
        "checked_at": _exact_utc_now(),
        "reaper_health_sha256": hashlib.sha256(
            _canonical_bytes(fresh_health)).hexdigest(),
        "safety_proof_file_sha256": None,
        "safety_proof_sha256": None,
        "provider_account_id": current_account_id,
        "provider_gpu_id": plan_data["chosen"]["provider_gpu_id"],
        "image": plan_data["chosen"]["image"],
        "bundle_contract_sha256": plan_data["bundle_contract_sha256"],
        "control_manifest_sha256":
            plan_data["control_plane"]["manifest_sha256"],
    }
    if args.runpod_safety_proof:
        fresh_proof = validate_safety_proof(
            args.runpod_safety_proof,
            plan_data["bundle_contract_sha256"],
            plan_data["control_plane"]["manifest_sha256"],
            current_account_id, ledger_path)
        fresh_safety["safety_proof_file_sha256"] = sha256_file(
            args.runpod_safety_proof)
        fresh_safety["safety_proof_sha256"] = (
            fresh_proof["proof"]["proof_sha256"])
    balance = status.get("clientBalance")
    inventory = provider.chargeable_inventory()
    if balance is None or not inventory.get("complete"):
        raise Refusal("immediate RunPod balance/inventory is unknown", [])
    fresh_offers = provider.gpus()
    provider_gpu_id = plan_data["chosen"]["provider_gpu_id"]
    fresh_candidates = [
        offer for offer in fresh_offers
        if offer.gpu_type == provider_gpu_id
        and offer.spot is False and offer.region == "secure"
        and offer.free_devices >= 1]
    if not fresh_candidates:
        raise Refusal(
            "exact secure on-demand RunPod offer disappeared before reserve",
            [])
    try:
        exact_fresh = [
            (Decimal(str(
                (candidate.raw or {}).get("uninterruptablePriceDecimal"))),
             candidate)
            for candidate in fresh_candidates
        ]
        if any(not rate.is_finite() or rate <= 0 for rate, _ in exact_fresh):
            raise ValueError("non-positive or non-finite rate")
        fresh_rate, fresh_offer = sorted(
            exact_fresh, key=lambda item: (item[0], item[1].gpu_type))[0]
    except Exception as exc:
        raise Refusal(
            "fresh RunPod offer lacks exact decimal rate", []) from exc
    if (not fresh_rate.is_finite() or fresh_rate <= 0
            or fresh_rate
            != Decimal(plan_data["chosen"]["price_per_gpu_hour"])
            or int(fresh_offer.vram_bytes)
            != int(plan_data["chosen"]["vram_bytes"])):
        raise Refusal(
            "RunPod offer changed after plan; re-plan before paying", [])
    quote_target = type("_QuoteTarget", (), {
        "repo_id": plan_data["target"]["repo_id"],
        "revision": plan_data["target"]["revision"]})()
    quote = _runpod_quote(
        args, plan_data["chosen"], quote_target, plan_data["profile"],
        plan_data["timing"], plan_data["storage_gb"],
        plan_data["container_disk_gb"],
        Decimal(plan_data["max_runtime_seconds"]),
        plan_data["target"]["result_archive_contract"])
    current_provider_resources = []
    for family_name, family in (inventory.get("families") or {}).items():
        for row in family.get("resources") or []:
            resource = {
                "family": family_name,
                "id": str(row.get("id") or "").strip(),
                "name": str(row.get("name") or "").strip(),
                "status": (
                    "PRESENT" if family_name == "network_volumes"
                    else str(row.get("status") or "").strip()),
            }
            if not all(resource.values()):
                raise Refusal(
                    "provider inventory lacks exact id/name/status", [])
            current_provider_resources.append(resource)
    now_dt = datetime.now(timezone.utc)
    observed = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    valid = (now_dt + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recorded = _ledger_transition(
        ledger, "record_provider_snapshot",
        balance_available_usd=balance, balance_observed_at=observed,
        balance_valid_until=valid,
        balance_source="RunPod myself.clientBalance",
        inventory_observed_at=inventory["observed_at_utc"],
        inventory_valid_until=valid, inventory_complete=True,
        provider_resources=current_provider_resources,
        inventory_source=inventory["schema"],
        provider="runpod", provider_account_id=current_account_id)
    if not recorded.applied:
        raise Refusal("campaign snapshot was not recorded", [])
    canonical_inventory = ledger.snapshot()["inventory"]
    unknown_resources = canonical_inventory["unknown_resources"]
    if unknown_resources and not foreign_tolerated:
        raise Refusal(
            "canonical campaign classifier found unknown chargeable resources",
            ["unknown: %s" % value for value in unknown_resources])
    outstanding = sum(
        1 for item in ledger.snapshot()["attempts"].values()
        if not item["released"])
    if outstanding >= args.campaign_width:
        raise Refusal("campaign outstanding count reached width", [])

    fresh_safety["server_time"] = provider.server_time_evidence(
        max_clock_delta_seconds=30, max_evidence_age_seconds=30)
    if campaign_mode == "explicit":
        attempt = secrets.token_hex(12)
    storage_layout = plan_data["storage_layout"]
    fs_root, engine_root = _runpod_run_roots(
        storage_layout, plan_data["job_id_full"], attempt)
    container_disk_gb = plan_data["container_disk_gb"]
    quote_epoch = datetime.strptime(
        quote.quoted_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc).timestamp()
    workload_epoch = quote_epoch + float(
        quote.workload_deadline_seconds)
    terminate_epoch = quote_epoch + float(
        quote.provider_termination_deadline_seconds)
    if time.time() >= terminate_epoch:
        raise Refusal("fresh quote provider deadline already elapsed", [])
    terminate_after = utc_iso(terminate_epoch)
    lease_store = LeaseStore(Path(args.lease_dir))
    expected_lease_name = "%s.%s.json" % (
        plan_data["job_id_full"], attempt)
    exact_pod_name = exact_resource_name(plan_data["job_id_full"], attempt)
    prepared_create = provider.prepare_safe_create(
        gpu_type=plan_data["chosen"]["provider_gpu_id"],
        num_gpus=1, spot=False, region="secure", offer="on-demand",
        data_center_id=plan_data["chosen"].get("data_center_id"),
        storage_gb=plan_data["storage_gb"],
        container_disk_gb=container_disk_gb,
        min_vcpu=plan_data["runtime_contract"]["min_vcpu_count"],
        min_ram_gb=plan_data["runtime_contract"]["min_memory_gb"],
        name=exact_pod_name,
        image=plan_data["job"]["environment"]["image"],
        terminate_after=terminate_after)
    prepared_create_doc = prepared_create.to_dict()
    job = json.loads(_canonical_bytes(plan_data["job"]).decode("utf-8"))
    job["execution_attempt"] = {
        "kind": "runpod-ssh", "attempt_id": attempt,
        "cost_quote": quote.to_dict(),
        "execution_contract_sha256": None,
        "pre_create_safety": fresh_safety,
        "prepared_create": prepared_create_doc,
        "engine_root": engine_root,
        "storage_layout": storage_layout,
        "lease_path": expected_lease_name, "remote_root": fs_root,
        "workload_deadline_utc": utc_iso(workload_epoch),
        "provider_terminate_after": terminate_after,
        "planned_at": quote.quoted_at,
    }
    job = seal_execution_job(job)
    validate_execution_job(job)
    job_bytes = (
        json.dumps(job, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n").encode("utf-8")
    job_path = outdir / "job.json"
    with job_path.open("xb") as stream:
        stream.write(job_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    output_directory_fd = os.open(str(outdir), os.O_RDONLY)
    try:
        os.fsync(output_directory_fd)
    finally:
        os.close(output_directory_fd)
    campaign_key = campaign_attempt_key(plan_data["job_id_full"], attempt)
    lease_ref = None
    request = {
        "attempt_key": attempt,
        "campaign_attempt_key": campaign_key,
        "campaign_ledger": Path(ledger_path).name,
        "provider": "runpod",
        "provider_account_id": current_account_id,
        "gpu_type": plan_data["chosen"]["provider_gpu_id"],
        "normalized_gpu": plan_data["chosen"]["gpu_type"],
        "num_gpus": 1, "secure_cloud": True,
        "storage_gb": plan_data["storage_gb"],
        "remote_root": fs_root,
        "execution_contract_sha256":
            job["execution_attempt"]["execution_contract_sha256"],
        "grounding_bundle": {
            "schema": "fidelity-suite/grounding-bundle.v1",
            "archive_sha256": frozen_bundle["archive_sha256"],
            "archive_bytes": frozen_bundle["archive_bytes"],
            "manifest_sha256": frozen_bundle["manifest_sha256"],
        },
        "engine_root": engine_root,
        "container_disk_gb": container_disk_gb,
        "image": plan_data["chosen"]["image"],
        "min_vcpu_count": plan_data["runtime_contract"]["min_vcpu_count"],
        "min_memory_gb": plan_data["runtime_contract"]["min_memory_gb"],
        "workload_contract": plan_data["runtime_contract"],
        "offer": "on-demand", "network_volume": None,
        "terminate_after": terminate_after, "quote": quote.to_dict(),
        "pre_create_safety": fresh_safety["server_time"],
        "prepared_create": prepared_create_doc,
    }
    if (request["quote"] != job["execution_attempt"]["cost_quote"]
            or request["pre_create_safety"]
            != job["execution_attempt"]["pre_create_safety"]["server_time"]
            or request["prepared_create"]
            != job["execution_attempt"]["prepared_create"]
            or request["execution_contract_sha256"]
            != job["execution_attempt"]["execution_contract_sha256"]
            or request["engine_root"]
            != job["execution_attempt"]["engine_root"]
            or request["remote_root"]
            != job["execution_attempt"]["remote_root"]
            or request["provider_account_id"]
            != job["environment"]["provider_account_id"]
            or request["workload_contract"] != job["runtime"]
            or request["min_vcpu_count"]
            != job["runtime"]["min_vcpu_count"]
            or request["min_memory_gb"]
            != job["runtime"]["min_memory_gb"]):
        raise Refusal(
            "lease workload/quote contract differs from finalized job", [])
    pre_resources = inventory["families"]["pods"]["resources"]
    pre_network_volumes = (
        inventory["families"]["network_volumes"]["resources"])
    try:
        with lease_store.paid_admission_lock():
            try:
                if campaign_mode == "explicit":
                    validate_unresolved_lease_scope(
                        lease_store, fresh_health,
                        provider="runpod",
                        provider_account_id=current_account_id,
                        campaign_ledger_path=Path(ledger_path))
                else:
                    validate_lease_liability_scope(
                        lease_store, provider="runpod",
                        provider_account_id=current_account_id,
                        allow_live=bool(getattr(
                            args, "allow_unresolved_leases", False)))
            except Exception as exc:
                # Nothing has been created yet: a scope change at the lock
                # is a refusal, not a leak.
                raise Refusal(
                    "lease scope changed at paid admission: %s" % exc, [])
            # PREPARED and the campaign reservation become visible together
            # while every paid controller sharing this lease root is excluded.
            lease_ref = lease_store.begin_create(
                job_hash=plan_data["job_id_full"], provider="runpod",
                request=request, pre_create_resources=pre_resources,
                pre_create_network_volumes=pre_network_volumes,
                create_deadline_epoch=time.time() + 300,
                workload_deadline_epoch=workload_epoch, attempt_id=attempt)
            if lease_ref.path.name != expected_lease_name:
                raise RuntimeError(
                    "prepared lease path differs from finalized execution job")
            if args.campaign_width == 2:
                width_two = validate_width_two_root_archive(
                    args.width_two_root_archive, job)
                current_public = validate_current_public_root(
                    width_two["publication"])
                current_authorization = {
                    "fruit_public_archive_sha256":
                        sha256_file(args.width_two_root_archive),
                    "fruit_proof_sha256": hashlib.sha256(
                        _canonical_bytes(current_public)).hexdigest(),
                }
                if current_authorization != plan_data[
                        "_width_two_authorization"]:
                    raise Refusal(
                        "width-two Fruit publication proof changed before "
                        "reserve", [])
                width_result = ledger.authorize_concurrent_width_two(
                    ledger.snapshot()["generation"], _exact_utc_now(),
                    current_authorization["fruit_public_archive_sha256"],
                    current_authorization["fruit_proof_sha256"])
                if not width_result.applied:
                    raise Refusal(
                        "campaign width-two authorization refused [%s]: %s"
                        % (width_result.code, width_result.message), [])
            admitted = ledger.reserve(
                ledger.snapshot()["generation"],
                plan_data["job_id_full"], attempt,
                quote, observed, effective_width=args.campaign_width)
            if not admitted.admitted:
                raise Refusal(
                    "campaign admission refused [%s]: %s"
                    % (admitted.code, admitted.message), [])
            if admitted.attempt_key != campaign_key:
                raise RuntimeError(
                    "campaign reservation key differs from prepared lease")
    except BaseException as primary:
        from fidelity.cloudlease import cancel_prepared_lease
        # Nothing was POSTed: close the PREPARED lease through the same
        # protocol the reaper uses (durable intent, ledger release, no-POST
        # proof) so their evidence agrees, and never let that bookkeeping
        # replace the error that stopped the run.
        lease_path = Path(args.lease_dir) / expected_lease_name
        try:
            if lease_path.exists():
                document = lease_store.read(lease_path)
                if document.get("state") == "PREPARED":
                    cancel_prepared_lease(
                        lease_store, lease_store.ref(lease_path, document),
                        "controller failure before provider POST: %s"
                        % redact(str(primary))[:400])
        except BaseException as cleanup:
            raise RuntimeError(
                "%s (and closing the PREPARED lease failed: %s)"
                % (primary, cleanup)) from primary
        raise

    pod_id = None
    token_cleanup_required = False
    watchdog_armed = False
    secret_cleanup = {"confirmed": True, "not_applicable": True}
    run_error = None
    primary_error = None
    operational_errors = []

    def record_operational_error(exc):
        operational_errors.append(exc)
        return run_error if run_error is not None else exc

    def response_loss_resources(pods, volumes):
        resources = []
        for family, rows in (("pods", pods), ("network_volumes", volumes)):
            for row in rows:
                resource = {
                    "family": family,
                    "id": str(_machine_id_of(row) or "").strip(),
                    "name": str(row.get("name") or "").strip(),
                    "status": (
                        "PRESENT" if family == "network_volumes"
                        else str(row.get("status") or "").strip()),
                }
                if not all(resource.values()):
                    raise RuntimeError(
                        "response-loss inventory lacks exact id/name/status")
                resources.append(resource)
        return resources

    def authoritative_account_inventory():
        fresh_status = provider.status()
        fresh_account_id = str(fresh_status.get("id") or "").strip()
        if (fresh_account_id != current_account_id
                or fresh_status.get("clientBalance") is None):
            raise RuntimeError(
                "authoritative inventory cannot verify the RunPod account")
        graphql_pods = provider.list_lifecycle_resources()
        strict_inventory = provider.chargeable_inventory()
        union_pods, absence_proof = runpod_authoritative_listing(
            provider, graphql_pods, fresh_account_id,
            inventory=strict_inventory)
        strict_volumes = list(
            strict_inventory["families"]["network_volumes"]["resources"])
        response_loss_resources(union_pods, strict_volumes)
        return (
            fresh_status, graphql_pods, union_pods, strict_volumes,
            strict_inventory, absence_proof)

    def authorized_response_loss_siblings(
            pods, volumes, intended_provider_id=None):
        resources = response_loss_resources(pods, volumes)
        classified = ledger.classify_provider_resources(resources)
        pre_ids = set(
            lease_store.read(lease_ref)["create"]["pre_create_provider_ids"])
        new_pods = {
            row["id"] for row in resources
            if row["family"] == "pods" and row["id"] not in pre_ids}
        intended = (
            set() if intended_provider_id is None
            else {str(intended_provider_id)})
        return sorted(
            (set(classified["known_pod_ids"]) & new_pods) - intended)

    def record_response_loss_inventory(
            pods, volumes, fresh_status, strict_inventory):
        now = datetime.now(timezone.utc)
        inventory_valid_until = (now + timedelta(minutes=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        recorded = _ledger_transition(
            ledger, "record_provider_snapshot",
            provider="runpod", provider_account_id=current_account_id,
            balance_available_usd=fresh_status["clientBalance"],
            balance_observed_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            balance_valid_until=inventory_valid_until,
            balance_source="RunPod myself.clientBalance",
            inventory_observed_at=strict_inventory["observed_at_utc"],
            inventory_valid_until=inventory_valid_until,
            inventory_complete=True,
            provider_resources=response_loss_resources(pods, volumes),
            inventory_source=strict_inventory["schema"])
        if not recorded.applied:
            raise RuntimeError(
                "response-loss campaign inventory could not be recorded")
        lease = lease_store.read(lease_ref)
        ids = lease.get("provider_resource_ids") or []
        if ids:
            bound = _ledger_transition(
                ledger, "bind_provider_for_cleanup", campaign_key, ids,
                campaign_cleanup_binding_evidence(lease, ids))
            if bound.code not in (
                    "PROVIDER_BOUND_FOR_CLEANUP",
                    "PROVIDER_CLEANUP_BINDING_UNCHANGED"):
                raise RuntimeError(
                    "response-loss exact IDs could not be bound for cleanup")
    failed_stage = None
    stages_done = []
    host_key_evidence = None
    archive_verified = None
    post_create_convergence_evidence = None
    local_archive = outdir / "result.tar.gz"
    heartbeat_stop = threading.Event()
    try:
        response = None
        try:
            pre_post_source = _source_checkout_proof(
                include_untracked=False)
            if pre_post_source != job["source_checkout"]["pre_post"]:
                raise RuntimeError(
                    "suite HEAD/index/worktree changed before provider POST")
            _verify_frozen_suite(frozen_bundle, plan_data["bundle"])
            precreate_now = time.time()
            sealed_server_time = fresh_safety["server_time"]
            server_evidence_age = (
                precreate_now
                - float(sealed_server_time["local_received_epoch"]))
            if (not math.isfinite(server_evidence_age)
                    or server_evidence_age < -1
                    or server_evidence_age > 30
                    or abs(float(
                        sealed_server_time[
                            "local_minus_server_seconds"])) > 30):
                raise RuntimeError(
                    "sealed RunPod server-time evidence expired before "
                    "provider POST")
            snapshot_valid_epoch = datetime.strptime(
                valid, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc).timestamp()
            if precreate_now >= snapshot_valid_epoch:
                raise RuntimeError(
                    "sealed RunPod balance/inventory snapshot expired "
                    "before POST")
            if (precreate_now >= workload_epoch
                    or precreate_now >= terminate_epoch - 300):
                raise RuntimeError(
                    "sealed RunPod quote no longer leaves a safe "
                    "execution window")
            creating = _ledger_transition(
                ledger, "mark_creating", campaign_key)
            if not creating.applied:
                raise RuntimeError(
                    "campaign CREATING was not recorded: %s"
                    % creating.message)
            lease_ref = lease_store.record_post_intent(lease_ref)
            # From here a POST may reach the provider; an exception is no
            # longer "nothing created" and must surface as a possible leak.
            plan_data["_post_intent_recorded"] = True
        except BaseException as primary:
            from fidelity.cloudlease import cancel_prepared_lease
            try:
                current = lease_store.read(lease_ref)
                if current.get("state") == "PREPARED":
                    lease_ref = cancel_prepared_lease(
                        lease_store, lease_ref,
                        "controller failure before provider POST: %s"
                        % redact(str(primary))[:400])
            except BaseException as cleanup:
                raise RuntimeError(
                    "%s (and closing the PREPARED lease failed: %s)"
                    % (primary, cleanup)) from primary
            raise
        try:
            # Sole create POST. The lease-side submission lock excludes a
            # concurrent response-loss reconciliation until its exact response
            # ID is durably recorded.
            lease_ref, response = lease_store.submit_create_and_record(
                lease_ref,
                lambda: provider.submit_prepared_create(prepared_create))
        except RunPodCreateResponseError as create_exc:
            returned_provider_id = str(create_exc.provider_id)
            pod_id = returned_provider_id
            failed_stage = "provision"
            lease_ref = getattr(create_exc, "durable_lease_ref", None)
            if (lease_ref is None
                    or lease_store.read(lease_ref).get("provider_resource_ids")
                    != [returned_provider_id]):
                raise RuntimeError(
                    "structured create response lacks its durable exact-ID "
                    "lease binding") from create_exc
            _bind_lease_cleanup_to_campaign(
                ledger_path, lease_store.read(lease_ref))
            try:
                (response_status, unused_graphql, post_pods, post_volumes,
                 response_inventory, unused_proof) = (
                    authoritative_account_inventory())
            except BaseException as inventory_exc:
                raise RuntimeError(
                    "RunPod create response was unqualified; its exact "
                    "provider ID is durably bound, but complete account-bound "
                    "recovery inventory is unavailable: %s"
                    % redact(str(inventory_exc))) from create_exc
            response_siblings = authorized_response_loss_siblings(
                post_pods, post_volumes, pod_id)
            lease_ref = lease_store.bind_post_create_inventory(
                lease_ref, post_pods, network_volumes=post_volumes,
                authorized_sibling_pod_ids=response_siblings)
            record_response_loss_inventory(
                post_pods, post_volumes, response_status, response_inventory)
            if lease_ref.state == "AMBIGUOUS":
                _cleanup_ambiguous_runpod_create(
                    provider, lease_store, lease_ref, ledger, campaign_key)
            raise RuntimeError(
                "RunPod returned an unqualified create response for durable "
                "pod id %s; cleanup only" % pod_id)
        except CreateResponsePersistenceError as persistence_exc:
            returned_id = str(persistence_exc.provider_id or "")
            if not returned_id:
                raise RuntimeError(
                    "committed create response lacks a recoverable exact ID"
                ) from persistence_exc
            pod_id = returned_id
            failed_stage = "provision"
            lease = lease_store.read(lease_ref)
            lease_ids = sorted(lease.get("provider_resource_ids") or [])
            if lease_ids:
                if lease_ids != [returned_id]:
                    raise RuntimeError(
                        "persisted lease IDs differ from committed response ID")
                lease_ref = lease_store.ref(lease_ref.path, lease)
            cleanup_binding = _ledger_transition(
                ledger, "bind_provider_for_cleanup", campaign_key,
                [returned_id],
                campaign_cleanup_binding_evidence(lease, [returned_id]))
            if cleanup_binding.code not in (
                    "PROVIDER_BOUND_FOR_CLEANUP",
                    "PROVIDER_CLEANUP_BINDING_UNCHANGED"):
                raise RuntimeError(
                    "returned RunPod ID could not be durably authorized "
                    "for cleanup: %s" % cleanup_binding.message
                ) from persistence_exc
            raise RuntimeError(
                "RunPod committed pod %s, but its lease response binding "
                "failed; campaign cleanup liability is retained"
                % returned_id) from persistence_exc
        except Exception as create_exc:
            try:
                (response_status, unused_graphql, complete, complete_volumes,
                 response_inventory, unused_proof) = (
                    authoritative_account_inventory())
            except BaseException as inventory_exc:
                raise RuntimeError(
                    "RunPod create response was lost and complete "
                    "account-bound recovery inventory is unavailable; "
                    "the CREATING liability is retained: %s"
                    % redact(str(inventory_exc))) from create_exc
            response_siblings = authorized_response_loss_siblings(
                complete, complete_volumes)
            rejection_codes = tuple(
                getattr(create_exc, "rejection_codes", ()) or ())
            lease_ref = lease_store.reconcile_response_lost(
                lease_ref, complete, network_volumes=complete_volumes,
                response_provider_id=None,
                create_window_closed=False,
                authorized_sibling_pod_ids=response_siblings,
                response_error=redact(str(create_exc)),
                provider_rejection_codes=rejection_codes)
            record_response_loss_inventory(
                complete, complete_volumes, response_status,
                response_inventory)
            ids = lease_store.read(lease_ref).get(
                "provider_resource_ids") or []
            if lease_ref.state == "TERMINAL" and not ids:
                # The provider refused the create by name and nothing exists.
                # Release the reservation so one capacity refusal cannot close
                # paid admission for the entire campaign.
                release = _ledger_transition(
                    ledger, "cancel_before_create", campaign_key,
                    utcnow(), "PROVIDER_REJECTED_CREATE",
                    "provider refused create: %s"
                    % ", ".join(rejection_codes))
                if release.code not in (
                        "CANCELLED_BEFORE_CREATE",
                        "CANCELLATION_ALREADY_RECORDED"):
                    raise RuntimeError(
                        "provider refused the create but its reservation "
                        "could not be released: %s" % release.message)
                raise Refusal(
                    "RunPod refused the create outright (%s); nothing was "
                    "created and $0.00 was spent"
                    % ", ".join(rejection_codes),
                    ["the lease is terminal and the campaign reservation is "
                     "released", "retry when the requested capacity returns"])
            if not ids:
                raise RuntimeError(
                    "RunPod create response was lost with no exact cleanup "
                    "candidate; the %s liability is retained for reaping: %s"
                    % (lease_ref.state, redact(str(create_exc))))
            if len(ids) > 1:
                _cleanup_ambiguous_runpod_create(
                    provider, lease_store, lease_ref, ledger, campaign_key)
                raise RuntimeError(
                    "ambiguous RunPod create candidates were terminated; "
                    "scientific execution is refused: %s"
                    % redact(str(create_exc)))
            pod_id = str(ids[0])
            failed_stage = "provision"
            raise RuntimeError(
                "RunPod create response was lost; sole reconciled candidate "
                "%s is cleanup-only and scientific execution is refused: %s"
                % (pod_id, redact(str(create_exc))))
        if response is not None:
            returned_id = str(_machine_id_of(response) or "")
            if not returned_id:
                raise RuntimeError("RunPod create returned no exact pod id")
            pod_id = returned_id
            failed_stage = "provision"
            durable_ids = lease_store.read(
                lease_ref).get("provider_resource_ids") or []
            if durable_ids != [returned_id]:
                raise RuntimeError(
                    "RunPod response ID differs from its durable lease binding")
            if str(response.get("image_name") or "") != job["environment"]["image"]:
                raise RuntimeError(
                    "RunPod create response image differs from finalized job")
            try:
                acknowledged_rate = Decimal(str(response.get("cost_per_hr")))
            except (ValueError, TypeError, InvalidOperation) as exc:
                raise RuntimeError(
                    "RunPod create response lacks an exact acknowledged "
                    "hourly rate") from exc
            if (not acknowledged_rate.is_finite()
                    or acknowledged_rate <= 0):
                raise RuntimeError(
                    "RunPod create response acknowledged hourly rate is invalid")
        convergence_contract = job["post_create_convergence"]
        convergence_started = time.time()
        convergence_deadline = min(
            convergence_started + convergence_contract["timeout_seconds"],
            terminate_epoch)
        convergence_observations = []
        convergence_ready = False
        convergence_failure = None
        expected_pod_name = lease_store.read(
            lease_ref)["create"]["exact_name"]
        pre_pod_ids = {
            str(row.get("id") or "")
            for row in inventory["families"]["pods"]["resources"]}
        pre_volume_ids = {
            str(row.get("id") or "")
            for row in
            inventory["families"]["network_volumes"]["resources"]}
        post_create_status = {}
        post_graphql_pods = []
        post_lifecycle_pods = []
        post_identity_volumes = []
        post_create_resources = []
        post_create_inventory = {"complete": False}
        authorized_sibling_ids = []
        post_authoritative_observed = False
        while True:
            observed_at = _exact_utc_now()
            try:
                (post_create_status, post_graphql_pods,
                 post_lifecycle_pods, post_identity_volumes,
                 post_create_inventory, unused_absence_proof) = (
                    authoritative_account_inventory())
                post_authoritative_observed = True
                strict_families = post_create_inventory["families"]
                strict_pods = strict_families["pods"]["resources"]
                strict_volumes = strict_families[
                    "network_volumes"]["resources"]
                lifecycle_resources = response_loss_resources(
                    post_lifecycle_pods, post_identity_volumes)
                graphql_resources = response_loss_resources(
                    post_graphql_pods, post_identity_volumes)
                strict_resources = response_loss_resources(
                    strict_pods, strict_volumes)
                post_create_resources = lifecycle_resources
                malformed_nonown_pod_ids = []
                classified_post = ledger.classify_provider_resources(
                    lifecycle_resources)
                classified_strict = ledger.classify_provider_resources(
                    strict_resources)
                new_post_pod_ids = {
                    row["id"] for row in lifecycle_resources
                    if row["family"] == "pods"
                    and row["id"] not in pre_pod_ids}
                authorized_sibling_ids = sorted(
                    (set(classified_post["known_pod_ids"])
                     & new_post_pod_ids) - {str(pod_id)})
                exact_name_ids = {
                    row["id"] for row in lifecycle_resources
                    if row["family"] == "pods"
                    and row["id"] not in pre_pod_ids
                    and row["name"] == expected_pod_name}
                wrong_name_ids = (
                    new_post_pod_ids - exact_name_ids - {str(pod_id)})
                blockers = wrong_name_ids - set(authorized_sibling_ids)
                new_volume_ids = {
                    row["id"] for row in lifecycle_resources
                    if row["family"] == "network_volumes"
                    and row["id"] not in pre_volume_ids}
                intended_rows = [
                    row for row in post_graphql_pods
                    if str(_machine_id_of(row) or "") == str(pod_id)]
                intended_name = (
                    str(intended_rows[0].get("name") or "").strip()
                    if len(intended_rows) == 1 else "")
                intended_status = (
                    str(intended_rows[0].get("status") or "").strip().upper()
                    if len(intended_rows) == 1 else "")
                intended_wrong_name = bool(
                    intended_name and intended_name != expected_pod_name)
                intended_terminal = intended_status in {
                    "EXITED", "FAILED", "STOPPED", "TERMINATED", "DELETED"}
                intended_exact = (
                    len(intended_rows) == 1
                    and intended_name == expected_pod_name
                    and intended_status == "RUNNING")
                try:
                    lifecycle_rate = Decimal(str(
                        intended_rows[0].get("cost_per_hr")))
                    lifecycle_economics = (
                        lifecycle_rate.is_finite() and lifecycle_rate > 0)
                except (IndexError, ValueError, TypeError, InvalidOperation):
                    lifecycle_rate = None
                    lifecycle_economics = False
                extra_exact = exact_name_ids - {str(pod_id)}
                account_id = str(post_create_status.get("id") or "").strip()
                account_changed = bool(
                    account_id and account_id != current_account_id)
                strict_rows = [
                    row for row in strict_pods
                    if str(row.get("id") or "") == str(pod_id)]
                strict_name = (
                    str(strict_rows[0].get("name") or "").strip()
                    if len(strict_rows) == 1 else "")
                strict_status = (
                    str(strict_rows[0].get("status") or "").strip().upper()
                    if len(strict_rows) == 1 else "")
                strict_wrong_name = bool(
                    strict_name and strict_name != expected_pod_name)
                strict_terminal = strict_status in {
                    "EXITED", "FAILED", "STOPPED", "TERMINATED", "DELETED"}
                strict_exact = (
                    len(strict_rows) == 1
                    and strict_name == expected_pod_name
                    and strict_status == "RUNNING")
                try:
                    strict_rate = Decimal(str(
                        strict_rows[0].get("cost_per_hr")))
                    strict_economics = (
                        strict_rate.is_finite() and strict_rate > 0)
                except (IndexError, ValueError, TypeError, InvalidOperation):
                    strict_rate = None
                    strict_economics = False
                rate_views_exact = bool(
                    lifecycle_economics and strict_economics
                    and acknowledged_rate == lifecycle_rate == strict_rate)
                rate_views_mismatch = bool(
                    (lifecycle_economics
                     and lifecycle_rate != acknowledged_rate)
                    or (strict_economics
                        and strict_rate != acknowledged_rate))
                graphql_keys = {
                    (row["family"], row["id"])
                    for row in graphql_resources}
                strict_keys = {
                    (row["family"], row["id"])
                    for row in strict_resources}
                family_closure_exact = graphql_keys == strict_keys
                # Pre-existing foreign pods are "unknown" to the classifier;
                # the delta against the pre-create inventory (blockers,
                # new_volume_ids, extra_exact) still catches anything that
                # appeared during this create window.
                strict_unknown_beyond_intended = [
                    row for row in classified_strict["unknown_resources"]
                    if row != {"family": "pods", "id": str(pod_id)}
                    and not foreign_tolerated]
                fatal_delta = bool(
                    blockers or new_volume_ids or extra_exact
                    or malformed_nonown_pod_ids
                    or len(intended_rows) > 1
                    or intended_wrong_name or intended_terminal
                    or strict_wrong_name or strict_terminal
                    or strict_unknown_beyond_intended
                    or account_changed or rate_views_mismatch)
                convergence_ready = bool(
                    intended_exact and strict_exact and rate_views_exact
                    and post_create_inventory.get("complete")
                    and family_closure_exact
                    and account_id == current_account_id
                    and not fatal_delta)
                convergence_observations.append({
                    "observed_at": observed_at,
                    "identity_resources": lifecycle_resources,
                    "strict_chargeable_resources": strict_resources,
                    "combined_resources": post_create_resources,
                    "chargeable_inventory_complete":
                        bool(post_create_inventory.get("complete")),
                    "chargeable_family_completeness": {
                        family_name: bool(family.get("complete"))
                        for family_name, family in sorted(
                            strict_families.items())
                    },
                    "strict_pod": (
                        {
                            "id": str(strict_rows[0].get("id") or ""),
                            "name": strict_name,
                            "status": strict_status,
                            "cost_per_hr":
                                strict_rows[0].get("cost_per_hr"),
                        } if len(strict_rows) == 1 else None),
                    "account_id": account_id or None,
                    "intended_identity_exact_running": intended_exact,
                    "strict_identity_exact_running": strict_exact,
                    "strict_economics_positive": strict_economics,
                    "hourly_rate_views": {
                        "create_acknowledged": format(
                            acknowledged_rate, "f"),
                        "graphql_lifecycle": (
                            format(lifecycle_rate, "f")
                            if lifecycle_economics else None),
                        "rest_chargeable_inventory": (
                            format(strict_rate, "f")
                            if strict_economics else None),
                        "exactly_equal": rate_views_exact,
                    },
                    "full_family_id_closure": family_closure_exact,
                    "authorized_sibling_pod_ids": authorized_sibling_ids,
                    "unattributed_sibling_pod_ids": sorted(blockers),
                    "malformed_nonown_pod_ids":
                        sorted(malformed_nonown_pod_ids),
                    "strict_unknown_beyond_intended":
                        strict_unknown_beyond_intended,
                    "extra_exact_name_pod_ids": sorted(extra_exact),
                    "new_network_volume_ids": sorted(new_volume_ids),
                })
                if fatal_delta:
                    convergence_failure = (
                        "post-create hourly rate views disagree"
                        if rate_views_mismatch else
                        "post-create identity family delta is unsafe")
                    break
                if convergence_ready:
                    break
            except Exception as exc:
                convergence_observations.append({
                    "observed_at": observed_at,
                    "error": redact(str(exc)),
                })
            if time.time() >= convergence_deadline:
                convergence_failure = (
                    "post-create identity/economics convergence timed out")
                break
            time.sleep(convergence_contract["poll_seconds"])
        if post_authoritative_observed:
            lease_ref = lease_store.bind_post_create_inventory(
                lease_ref, post_lifecycle_pods,
                network_volumes=post_identity_volumes,
                authorized_sibling_pod_ids=authorized_sibling_ids)
        post_create_convergence_evidence = {
            "schema": "fidelity-suite/runpod-post-create-convergence-evidence.v1",
            "contract": convergence_contract,
            "started_at": datetime.fromtimestamp(
                convergence_started, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            "deadline_at": datetime.fromtimestamp(
                convergence_deadline, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            "finished_at": _exact_utc_now(),
            "converged": convergence_ready,
            "failure": convergence_failure,
            "observations": convergence_observations,
            "evidence_sha256": None,
        }
        post_create_convergence_evidence["evidence_sha256"] = hashlib.sha256(
            _canonical_bytes(post_create_convergence_evidence)).hexdigest()
        convergence_path = outdir / "runpod-post-create-convergence.json"
        with convergence_path.open("xb") as stream:
            stream.write(
                json.dumps(
                    post_create_convergence_evidence, indent=2,
                    sort_keys=True, ensure_ascii=False,
                    allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        convergence_directory_fd = os.open(str(outdir), os.O_RDONLY)
        try:
            os.fsync(convergence_directory_fd)
        finally:
            os.close(convergence_directory_fd)
        snapshot_now = datetime.now(timezone.utc)
        snapshot_observed = snapshot_now.strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshot_valid = (
            snapshot_now + timedelta(minutes=5)).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        post_balance = post_create_status.get("clientBalance")
        post_balance_observed = snapshot_observed
        post_balance_valid = snapshot_valid
        post_balance_source = "RunPod myself.clientBalance"
        if post_balance is None:
            post_balance = balance
            post_balance_observed = observed
            post_balance_valid = valid
            post_balance_source = (
                "RunPod myself.clientBalance sealed pre-create fallback")
        post_snapshot = _ledger_transition(
            ledger, "record_provider_snapshot",
            provider="runpod", provider_account_id=current_account_id,
            balance_available_usd=post_balance,
            balance_observed_at=post_balance_observed,
            balance_valid_until=post_balance_valid,
            balance_source=post_balance_source,
            inventory_observed_at=snapshot_observed,
            inventory_valid_until=snapshot_valid,
            inventory_complete=bool(
                convergence_ready and post_create_inventory.get("complete")),
            provider_resources=post_create_resources,
            inventory_source=
                "fidelity-suite/runpod-post-create-identity+chargeable.v1")
        if not post_snapshot.applied:
            raise RuntimeError(
                "post-create campaign inventory snapshot was not recorded")
        if not convergence_ready or lease_ref.state != "ACTIVE":
            cleanup_lease = lease_store.read(lease_ref)
            cleanup_ids = (
                cleanup_lease.get("provider_resource_ids")
                or [str(pod_id)])
            cleanup_binding = _ledger_transition(
                ledger, "bind_provider_for_cleanup", campaign_key,
                cleanup_ids,
                campaign_cleanup_binding_evidence(
                    cleanup_lease, cleanup_ids))
            if cleanup_binding.code not in (
                    "PROVIDER_BOUND_FOR_CLEANUP",
                    "PROVIDER_CLEANUP_BINDING_UNCHANGED"):
                raise RuntimeError(
                    "campaign exact provider cleanup binding failed: %s"
                    % cleanup_binding.message)
            if lease_ref.state != "ACTIVE":
                _cleanup_ambiguous_runpod_create(
                    provider, lease_store, lease_ref, ledger, campaign_key)
            raise RuntimeError(
                "%s; exact acknowledged pod is cleanup-only"
                % (convergence_failure or "post-create family delta"))
        canonical_post_inventory = ledger.snapshot()["inventory"]
        unknown_after_create = [
            row for row in canonical_post_inventory["unknown_resources"]
            if not (foreign_tolerated
                    and row != {"family": "pods", "id": str(pod_id)})]
        if unknown_after_create != [{"family": "pods", "id": str(pod_id)}]:
            raise RuntimeError(
                "post-create campaign classifier found resources beyond the "
                "single acknowledged pod; cleanup is restricted to attributed IDs")
        lease = lease_store.read(lease_ref)
        binding = provider.validate_safe_resource_binding(
            pod_id, expected_name=lease["create"]["exact_name"],
            gpu_type_id=plan_data["chosen"]["provider_gpu_id"],
            secure_cloud=True, gpu_count=1,
            volume_gb=plan_data["storage_gb"],
            container_disk_gb=container_disk_gb,
            image_name=request["image"], terminate_after=terminate_after)
        host_key_evidence = _authenticate_runpod_ssh_host(
            con, provider, str(pod_id), outdir)
        live_attestation = provider.attest_live_resource(
            pod_id,
            expected_gpu_model=plan_data["chosen"]["provider_gpu_display"],
            expected_vram_bytes=plan_data["chosen"]["vram_bytes"],
            min_vcpu=plan_data["runtime_contract"]["min_vcpu_count"],
            min_ram_gb=plan_data["runtime_contract"]["min_memory_gb"],
            volume_gb=plan_data["storage_gb"],
            container_disk_gb=container_disk_gb,
            workspace_available_bytes_minimum=job["resource_requirements"][
                "workspace_available_bytes_minimum"],
            container_available_bytes_minimum=job["resource_requirements"][
                "container_available_bytes_minimum"])
        lease_ref = lease_store.record_identity_attestation(
            lease_ref, live_attestation)
        filesystems = (
            (live_attestation.get("observed") or {}).get("filesystems") or {})
        workspace_available = (
            filesystems.get("workspace", {}).get("available_bytes"))
        container_available = (
            filesystems.get("container", {}).get("available_bytes"))
        required_remote_peak = int(
            job["resource_requirements"][
                "workspace_available_bytes_minimum"])
        # Written BEFORE the floor check: a refused attestation is the only
        # evidence of WHY a paid pod was refused (an ssh-image rehearsal on
        # 2026-09-04 lost it and the reason with it).
        attestation_path = outdir / "runpod-live-attestation.json"
        with attestation_path.open("xb") as stream:
            stream.write(
                json.dumps(
                    live_attestation, indent=2, sort_keys=True,
                    ensure_ascii=False, allow_nan=False).encode("utf-8")
                + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        pinned_datacenter = plan_data["chosen"].get("data_center_id")
        served = live_attestation.get("provider_record") or {}
        if pinned_datacenter is not None and (
                served.get("error") is not None
                or served.get("data_center_id") != pinned_datacenter):
            raise RuntimeError(
                "RunPod served datacenter %r differs from the --runpod-datacenter "
                "pin %r (provider_record error: %s); refusing to run there"
                % (served.get("data_center_id"), pinned_datacenter,
                   served.get("error")))
        if (live_attestation.get("ok") is not True
                or isinstance(workspace_available, bool)
                or not isinstance(workspace_available, int)
                or workspace_available < required_remote_peak
                or isinstance(container_available, bool)
                or not isinstance(container_available, int)
                or container_available < job["resource_requirements"][
                    "container_available_bytes_minimum"]):
            raise RuntimeError(
                "live RunPod resource attestation failed resource floors: "
                "failures=%s transport_error=%s workspace_available=%s "
                "container_available=%s (evidence: %s)"
                % (live_attestation.get("failures"),
                   live_attestation.get("transport_error"),
                   workspace_available, container_available,
                   attestation_path))
        live_rate = Decimal(str(binding["observed"]["cost_per_hr"]))
        quoted_rate = Decimal(str(quote.live_compute_usd_per_hour))
        if (not live_rate.is_finite() or live_rate <= 0
                or not strict_rate.is_finite() or strict_rate <= 0
                or live_rate != acknowledged_rate
                or lifecycle_rate != acknowledged_rate
                or strict_rate != acknowledged_rate
                or live_rate != quoted_rate):
            raise RuntimeError(
                "create, GraphQL lifecycle, REST inventory, refreshed binding "
                "and quoted hourly rates are not exactly equal")
        actual_quote = quote
        bound = _ledger_transition(
            ledger, "bind_actual_quote", campaign_key, pod_id, actual_quote)
        if (not bound.applied or bound.code != "ACTUAL_QUOTE_BOUND"
                or bound.action == "TERMINATE_IMMEDIATELY"):
            raise RuntimeError(
                "campaign provider/rate binding failed: %s" % bound.message)
        if (ledger.snapshot()["inventory"]["unknown_resources"]
                and not foreign_tolerated):
            raise RuntimeError(
                "campaign inventory remains unknown after exact quote binding")
        running = _ledger_transition(
            ledger, "mark_phase", campaign_key, "RUNNING")
        if (not running.applied
                or running.code not in ("PHASE_RECORDED", "PHASE_UNCHANGED")):
            raise RuntimeError(
                "campaign RUNNING transition failed: %s" % running.message)

        _upload_verified_bundle(
            provider, pod_id, fs_root, plan_data["bundle"], frozen_bundle)
        # The engine root's parent does not exist on a fresh pod, and
        # stage_measure.sh resolves it with `readlink -f` before anything
        # else; that exits 1 silently under `set -e` (Fruit smoke, 2026-09-03).
        provider.exec(pod_id, "mkdir -p {0}/inputs {0}/logs "
                      "{0}/receipts/done {0}/runtime {1}".format(
                          shlex.quote(fs_root), shlex.quote(engine_root)))
        provider.upload(pod_id, str(job_path), "%s/job.json" % fs_root)
        provider.upload(
            pod_id, str(attestation_path),
            "%s/receipts/runpod-live-attestation.json" % fs_root)
        provider.upload(
            pod_id, str(host_key_evidence["path"]),
            "%s/receipts/runpod-ssh-host-key-proof.json" % fs_root)
        if args.role == "root":
            provider.upload(pod_id, plan_data["_panel_archive_local"],
                            "%s/inputs/panel.tar" % fs_root)
            provider.upload(pod_id, plan_data["_panel_binding_local"],
                            "%s/inputs/panel.binding.json" % fs_root)
            if plan_data.get("_candidate_scope_local"):
                provider.exec(pod_id, "mkdir -p -m 0700 %s/candidate"
                              % shlex.quote(fs_root), timeout=60)
                provider.upload(pod_id, plan_data["_candidate_scope_local"],
                                "%s/%s" % (fs_root, CANDIDATE_SCOPE_REMOTE))
            if plan_data.get("_derived_allowlist_local"):
                # The pod resolves capture.unexpected_tensor_allowlist.path
                # under the run root and re-checks both digests before the
                # capture (bin/stage_measure.sh), exactly as for an authored
                # file inside the bundle.
                provider.upload(pod_id, plan_data["_derived_allowlist_local"],
                                "%s/%s" % (fs_root, DERIVED_ALLOWLIST_REMOTE))
            provider.exec(
                pod_id,
                "python3 {fs}/bin/fidelity/runpodsafety.py extract-panel "
                "--archive {fs}/inputs/panel.tar "
                "--binding {fs}/inputs/panel.binding.json "
                "--destination {fs}/inputs/panel".format(
                    fs=shlex.quote(fs_root)), timeout=900)
        provider.upload(
            pod_id, str(convergence_path),
            "%s/receipts/runpod-post-create-convergence.json" % fs_root)
        if frozen_resume is not None:
            _import_resume_capture(
                provider, pod_id, fs_root, plan_data, job, frozen_resume,
                outdir, con)
        provider.exec(
            pod_id,
            "python3 -c {code} {job} {digest}".format(
                code=shlex.quote(
                    "import hashlib,sys;"
                    "sys.path.insert(0,sys.argv[1].rsplit('/',1)[0]+'/bin');"
                    "p=sys.argv[1];b=open(p,'rb').read();"
                    "assert hashlib.sha256(b).hexdigest()==sys.argv[2];"
                    "from fidelity.jobcontract import "
                    "parse_job_bytes,validate_execution_job;"
                    "validate_execution_job(parse_job_bytes(b))"),
                job=shlex.quote("%s/job.json" % fs_root),
                digest=hashlib.sha256(job_bytes).hexdigest()))
        provider.exec(
            pod_id,
            "umask 077; tmp={fs}/.heartbeat.$$; : > \"$tmp\"; "
            "mv \"$tmp\" {fs}/heartbeat; chmod 600 {fs}/heartbeat".format(
                fs=shlex.quote(fs_root)))
        heartbeat_td = type("_TD", (), {
            "machine_id": pod_id, "fs_root": fs_root})()
        _start_heartbeat(provider, heartbeat_td, heartbeat_stop, con)
        provider.exec(
            pod_id,
            "nohup setsid bash {fs}/bin/watchdog.sh {deadline} {heartbeat} "
            "{fs} >{fs}/logs/watchdog.log 2>&1 </dev/null &".format(
                fs=shlex.quote(fs_root), deadline=int(workload_epoch),
                heartbeat=int(args.heartbeat_timeout)))
        provider.exec(
            pod_id,
            "i=0; while [ ! -f {fs}/receipts/watchdog-armed.json ] "
            "&& [ \"$i\" -lt 50 ]; do sleep 0.1; i=$((i+1)); done; "
            "test -f {fs}/receipts/watchdog-armed.json; "
            "python3 {fs}/bin/fidelity/runpodsafety.py verify-watchdog "
            "--fs-root {fs} --deadline {deadline} "
            "--heartbeat-timeout {heartbeat}".format(
                fs=shlex.quote(fs_root), deadline=int(workload_epoch),
                heartbeat=int(args.heartbeat_timeout)))
        watchdog_armed = True
        token_cleanup_required = True
        try:
            _transport_hf_token(
                provider, pod_id, fs_root, outdir, download_token,
                secrets_dir=_runpod_secrets_dir(fs_root))
        finally:
            download_token = ""
        con.ok(
            "HF download token installed",
            "0600 file, never argv; target-fetch scope only")
        for stage in stage_sequence(
                args.role, race=False,
                surface=plan_data["target"]["surface"],
                publish_root=False,
                candidate=(job.get("capture") or {}).get("candidate") is not None):
            failed_stage = stage
            if stage == "fetch_target":
                secret_cleanup = _runpod_fetch_target_and_remove_token(
                    provider, pod_id, fs_root, engine_root,
                    workload_epoch, job["environment"]["image"],
                    progress=con.say)
                token_cleanup_required = False
                con.ok(
                    "HF download token removed",
                    "authenticated target fetch complete; remote erasure confirmed")
            else:
                con.say("stage %s started" % stage)
                _runpod_stage(
                    provider, pod_id, fs_root, engine_root, stage,
                    workload_epoch, job["environment"]["image"],
                    progress=con.say)
            stages_done.append(stage)
            if stage == "setup":
                bench_doc = _preflight_bench(
                    args, con, provider, heartbeat_td, plan_data,
                    fail_closed=True,
                    python_executable="%s/venv/bin/python" % engine_root,
                    remote_payload=(
                        "%s/bin/fidelity/cardbench_payload.py" % fs_root))
                bench_bytes = (
                    json.dumps(
                        bench_doc, indent=2, sort_keys=True,
                        ensure_ascii=False, allow_nan=False) + "\n").encode(
                            "utf-8")
                bench_path = outdir / "preflight-bench.json"
                with bench_path.open("xb") as stream:
                    stream.write(bench_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                remote_bench = "%s/receipts/preflight-bench.json" % fs_root
                provider.upload(pod_id, str(bench_path), remote_bench)
                provider.exec(
                    pod_id,
                    "test \"$(sha256sum {path} | cut -d' ' -f1)\" = {digest}"
                    .format(
                        path=shlex.quote(remote_bench),
                        digest=hashlib.sha256(bench_bytes).hexdigest()))
        failed_stage = None
    except BaseException as exc:
        primary_error = exc
        run_error = exc
    finally:
        if pod_id is not None:
            exit_evidence = {
                "failed_stage": failed_stage,
                "run_error": redact(str(run_error)) if run_error else None,
                "stages_done": list(stages_done),
            }
            try:
                # An attempt that never attested (host key never surfaced,
                # create bound but no post-create binding) is still CREATING
                # in the ledger; EXITED is only reachable from LIVE/RUNNING.
                # The cleanup binding below handles that attempt's release.
                attempt_phase = (
                    (ledger.snapshot().get("attempts") or {})
                    .get(campaign_key) or {}).get("phase")
                if attempt_phase in ("LIVE", "RUNNING", "EXITED"):
                    exited = _ledger_transition(
                        ledger, "mark_phase", campaign_key, "EXITED")
                    if (not exited.applied
                            or exited.code not in (
                                "PHASE_RECORDED", "PHASE_UNCHANGED")):
                        raise RuntimeError(
                            "campaign EXITED transition failed: %s"
                            % exited.message)
                current_lease = lease_store.read(lease_ref)
                if current_lease["state"] == "ACTIVE":
                    lease_ref = lease_store.transition(
                        lease_ref, to_state="ACTIVE",
                        event="WORKLOAD_EXITED", evidence=exit_evidence)
            except BaseException as exc:
                run_error = record_operational_error(exc)
            if token_cleanup_required:
                secret_cleanup = _cleanup_remote_secret(
                    provider, pod_id, fs_root,
                    secrets_dir=_runpod_secrets_dir(fs_root))
                if not secret_cleanup.get("confirmed"):
                    run_error = record_operational_error(RuntimeError(
                        "remote token erasure is unconfirmed; revoke the "
                        "RunPod-scoped Hugging Face read token immediately"))
            try:
                if host_key_evidence is None:
                    # The pod never authenticated (host key never surfaced
                    # in provider logs); nothing was uploaded, no stage ran,
                    # and there is no channel to fetch anything through.
                    raise RuntimeError(
                        "no result archive: the pod's SSH host key was never "
                        "authenticated, so no workload reached it")
                remote_archive = "/tmp/fidelity-result-%s-%s.tar.gz" % (
                    plan_data["job_id"], attempt)
                archive_failed_stage = failed_stage
                if primary_error is not None and archive_failed_stage is None:
                    archive_failed_stage = "post-run"
                argv = [
                    "python3", "%s/bin/result_archive.py" % fs_root,
                    "--fs-root", fs_root,
                    "--verb", "capture" if args.role == "root" else "measure",
                    "--status", (
                        "failed" if primary_error is not None
                        else ("qualified-unpublished"
                              if args.role == "root" else "completed")),
                    "--stages", ",".join(stages_done),
                    "--out", remote_archive,
                ]
                if archive_failed_stage:
                    argv += ["--failed-stage", archive_failed_stage]
                transfer = json.loads(provider.exec_stdout(
                    pod_id, " ".join(shlex.quote(value) for value in argv),
                    timeout=1800).strip())
                if (not isinstance(transfer, dict)
                        or set(transfer) != {"path", "bytes", "sha256"}
                        or transfer.get("path") != remote_archive
                        or isinstance(transfer.get("bytes"), bool)
                        or not isinstance(transfer.get("bytes"), int)
                        or transfer["bytes"] <= 0
                        or transfer["bytes"] > job["target"][
                            "result_archive_contract"][
                                "result_archive_max_transfer_bytes"]
                        or re.fullmatch(
                            r"[0-9a-f]{64}", str(transfer.get("sha256", "")))
                        is None):
                    raise RuntimeError(
                        "remote result archive transfer record is noncanonical "
                        "or exceeds the finalized transfer cap")
                observed_lines = provider.exec_stdout(
                    pod_id,
                    "set -eu; sha256sum {path} | cut -d' ' -f1; "
                    "stat -c %s {path}".format(
                        path=shlex.quote(remote_archive))).strip().splitlines()
                if (len(observed_lines) != 2
                        or observed_lines[0] != transfer["sha256"]
                        or int(observed_lines[1]) != int(transfer["bytes"])):
                    raise RuntimeError(
                        "independent on-pod archive digest/size differs "
                        "from writer transfer record")
                extracted = outdir / "result"
                download_error = None
                retrieval_contract = plan_data["retrieval_delete_contract"]
                total_attempts = retrieval_contract["download_attempts"]
                verify_bound = retrieval_contract[
                    "local_verify_extract_bound_seconds_per_attempt"]
                delete_reserve = retrieval_contract[
                    "final_delete_reserve_seconds"]
                for download_attempt in range(1, total_attempts + 1):
                    remaining = terminate_epoch - time.time()
                    attempts_left = total_attempts - download_attempt + 1
                    required_remaining = (
                        attempts_left * (
                            retrieval_contract[
                                "download_timeout_seconds_per_attempt"]
                            + verify_bound)
                        + delete_reserve)
                    if remaining < required_remaining:
                        download_error = RuntimeError(
                            "provider deadline cannot fund every remaining "
                            "bounded retrieval attempt and final DELETE")
                        break
                    commit_started = False
                    try:
                        with tempfile.TemporaryDirectory(
                                prefix=".result-download-%d-" % download_attempt,
                                dir=str(outdir)) as hold:
                            os.chmod(hold, 0o700)
                            downloaded = Path(hold) / "result.tar.gz"
                            attempt_extracted = Path(hold) / "result"
                            provider.download_bounded(
                                pod_id, remote_archive, str(downloaded),
                                expected_bytes=transfer["bytes"],
                                max_bytes=job["target"][
                                    "result_archive_contract"][
                                        "result_archive_max_transfer_bytes"],
                                timeout=retrieval_contract[
                                    "download_timeout_seconds_per_attempt"])
                            metadata = downloaded.lstat()
                            if (downloaded.is_symlink()
                                    or not downloaded.is_file()
                                    or metadata.st_uid != os.getuid()):
                                raise RuntimeError(
                                    "downloaded result is not an owned regular file")
                            verified_attempt = extract_verified_archive(
                                downloaded, attempt_extracted,
                                expected_sha256=transfer["sha256"],
                                expected_bytes=transfer["bytes"])
                            with downloaded.open("rb") as stream:
                                os.fsync(stream.fileno())
                            commit_started = True
                            os.link(str(downloaded), str(local_archive))
                            os.rename(str(attempt_extracted), str(extracted))
                            directory_fd = os.open(str(outdir), os.O_RDONLY)
                            try:
                                os.fsync(directory_fd)
                            finally:
                                os.close(directory_fd)
                            archive_verified = verified_attempt
                        download_error = None
                        break
                    except BaseException as exc:
                        download_error = exc
                        if commit_started:
                            break
                if archive_verified is None:
                    raise RuntimeError(
                        "verified result retrieval exhausted: %s"
                        % redact(str(download_error)))
                archived_job = (extracted / "job.json").read_bytes()
                if (archived_job != job_bytes
                        or json.loads(archived_job.decode("utf-8")) != job):
                    raise RuntimeError(
                        "verified archive carries a different finalized job.json")
                archived_attestation_bytes = (
                    extracted / "receipts"
                    / "runpod-live-attestation.json").read_bytes()
                local_attestation_bytes = attestation_path.read_bytes()
                if archived_attestation_bytes != local_attestation_bytes:
                    raise RuntimeError(
                        "archived live attestation differs from exact local bytes")
                archived_host_key_bytes = (
                    extracted / "receipts"
                    / "runpod-ssh-host-key-proof.json").read_bytes()
                local_host_key_bytes = host_key_evidence["path"].read_bytes()
                if archived_host_key_bytes != local_host_key_bytes:
                    raise RuntimeError(
                        "archived SSH host-key proof differs from exact local "
                        "operator-authenticated bytes")
                archived_attestation = parse_job_bytes(
                    archived_attestation_bytes)
                durable_provider_ids = lease_store.read(
                    lease_ref).get("provider_resource_ids") or []
                if (archived_attestation.get("provider_id") != str(pod_id)
                        or durable_provider_ids != [str(pod_id)]):
                    raise RuntimeError(
                        "live attestation provider id differs from durable lease")
                if (job.get("capture") or {}).get("candidate") is not None:
                    _report_candidate_result(con, extracted, job)
            except BaseException as exc:
                if run_error is None:
                    primary_error = exc
                    run_error = exc
            heartbeat_stop.set()
            destroy_intent_recorded = False
            try:
                current = lease_store.read(lease_ref)
                if current["state"] not in (
                        "DESTROYING", "ABSENCE_CONFIRMED", "TERMINAL"):
                    lease_ref = lease_store.request_destroy(
                        lease_ref, {"reason": "controller exit",
                                    "provider_id": pod_id})
                _bind_lease_cleanup_to_campaign(
                    ledger_path, lease_store.read(lease_ref))
                phase = _ledger_transition(
                    ledger, "mark_phase", campaign_key,
                    "TERMINATE_REQUESTED")
                if not phase.applied:
                    raise RuntimeError(
                        "campaign destroy intent was not recorded: %s"
                        % phase.message)
                destroy_intent_recorded = True
            except BaseException as exc:
                run_error = record_operational_error(exc)
            # A pod that failed before its watchdog was armed has nothing to
            # disarm; the refusal that step would raise is not evidence.
            if destroy_intent_recorded and watchdog_armed:
                try:
                    provider.exec(
                        pod_id,
                        "python3 {fs}/bin/fidelity/runpodsafety.py "
                        "disarm-watchdog --fs-root {fs} --deadline {deadline} "
                        "--heartbeat-timeout {heartbeat}".format(
                            fs=shlex.quote(fs_root),
                            deadline=int(workload_epoch),
                            heartbeat=int(args.heartbeat_timeout)))
                except BaseException as exc:
                    run_error = record_operational_error(exc)
            if not destroy_intent_recorded:
                run_error = record_operational_error(RuntimeError(
                    "durable destroy intent failed; controller issued no DELETE; "
                    "watchdog/reaper/provider deadline retain liability"))
            else:
                delete_error = None
                try:
                    provider.destroy(pod_id)
                except BaseException as exc:
                    delete_error = redact(str(exc))
                try:
                    absence_error = None
                    for _unused in range(20):
                        try:
                            (absence_status, unused_graphql, absence_pods,
                             absence_volumes, absence_inventory,
                             absence_proof) = (
                                authoritative_account_inventory())
                            lease_ref = lease_store.confirm_exact_absence(
                                lease_ref, absence_pods,
                                authoritative_inventory=absence_proof)
                            if lease_ref.state != ABSENCE_CONFIRMED:
                                absence_resources = response_loss_resources(
                                    absence_pods, absence_volumes)
                                balance_observed = absence_status[
                                    "observed_at_utc"]
                                inventory_observed = absence_inventory[
                                    "observed_at_utc"]
                                balance_valid = (
                                    datetime.strptime(
                                        balance_observed,
                                        "%Y-%m-%dT%H:%M:%SZ").replace(
                                            tzinfo=timezone.utc)
                                    + timedelta(minutes=5)).strftime(
                                        "%Y-%m-%dT%H:%M:%SZ")
                                inventory_valid = (
                                    datetime.strptime(
                                        inventory_observed,
                                        "%Y-%m-%dT%H:%M:%SZ").replace(
                                            tzinfo=timezone.utc)
                                    + timedelta(minutes=5)).strftime(
                                        "%Y-%m-%dT%H:%M:%SZ")
                                classified = _ledger_transition(
                                    ledger, "record_provider_snapshot",
                                    provider="runpod",
                                    provider_account_id=current_account_id,
                                    balance_available_usd=absence_status[
                                        "clientBalance"],
                                    balance_observed_at=balance_observed,
                                    balance_valid_until=balance_valid,
                                    balance_source=(
                                        "RunPod myself.clientBalance"),
                                    inventory_observed_at=inventory_observed,
                                    inventory_valid_until=inventory_valid,
                                    inventory_complete=True,
                                    provider_resources=absence_resources,
                                    inventory_source=absence_inventory["schema"])
                                if not classified.applied:
                                    raise RuntimeError(
                                        "post-DELETE campaign inventory "
                                        "was not recorded")
                            absence_error = None
                        except BaseException as exc:
                            absence_error = exc
                        if lease_ref.state == ABSENCE_CONFIRMED:
                            break
                        time.sleep(3)
                    if (lease_ref.state != ABSENCE_CONFIRMED
                            and absence_error is not None):
                        raise absence_error
                    if lease_ref.state == ABSENCE_CONFIRMED:
                        # The pod is proven absent from a complete listing:
                        # nothing is billing.  Settlement is advisory from
                        # here.  RunPod publishes its hour bucket up to an
                        # hour and some minutes later; one attempt is made
                        # now, and the installed reaper settles it on a
                        # later sweep otherwise.  Treating "not yet
                        # published" as an operational failure refused the
                        # publication of a qualified dataset and exited 90
                        # with the pod already gone.
                        billing_pending_reason = None
                        try:
                            billing_lease = lease_store.read(lease_ref)
                            absence_rows = [
                                row for row in billing_lease.get("history") or []
                                if row.get("to") == "ABSENCE_CONFIRMED"]
                            if not absence_rows:
                                raise RuntimeError(
                                    "billing wait requires exact absence evidence")
                            absence_epoch = datetime.strptime(
                                absence_rows[-1]["at"],
                                "%Y-%m-%dT%H:%M:%SZ").replace(
                                    tzinfo=timezone.utc).timestamp()
                            billing_wait = max(
                                0.0, 300.0 - (time.time() - absence_epoch))
                            if billing_wait > args.runpod_billing_wait:
                                raise RuntimeError(
                                    "configured RunPod billing wait cannot "
                                    "reach the 300-second stabilization window")
                            if billing_wait:
                                time.sleep(billing_wait)
                            billing = provider.reconcile_billing(
                                lease_store.read(lease_ref))
                            lease_ref = lease_store.stage_billing_reconciliation(
                                lease_ref, billing)
                            finalize_campaign_after_absence(
                                provider, lease_store.read(lease_ref),
                                lease_store.root)
                            lease_ref = lease_store.record_billing_reconciled(
                                lease_ref, billing)
                        except Exception as exc:  # noqa: BLE001
                            # Provider or publication lag, never control
                            # flow: an interrupt here must still fail the run.
                            billing_pending_reason = redact(str(exc))[:300]
                        if lease_store.read(lease_ref).get("state") != "TERMINAL":
                            if foreign_tolerated:
                                settle_note = (
                                    "The installed reaper settles it on a "
                                    "later sweep; check with: measure-cloud "
                                    "reaper --provider runpod --list")
                            else:
                                settle_note = (
                                    "This is a strict campaign ledger: the "
                                    "reaper settles it only once the account "
                                    "holds no pod the campaign does not own; "
                                    "check with: measure-cloud reaper "
                                    "--provider runpod --list")
                            con.warn(
                                "billing pending: pod %s is proven absent; "
                                "billing did not settle yet (%s). %s"
                                % (pod_id, billing_pending_reason
                                   or "stabilization", settle_note))
                            exit_evidence["billing"] = {
                                "settled": False,
                                "pending_reason": billing_pending_reason,
                                "lease": lease_ref.path.name,
                            }
                        else:
                            exit_evidence["billing"] = {
                                "settled": True,
                                "lease": lease_ref.path.name,
                            }
                    else:
                        remedy = ""
                        if not secret_cleanup.get("confirmed"):
                            remedy = (
                                "; remote token erasure is unconfirmed: revoke "
                                "the RunPod-scoped Hugging Face token immediately")
                        detail = (
                            "; DELETE response error: " + delete_error
                            if delete_error else "")
                        run_error = record_operational_error(RuntimeError(
                            "exact RunPod id remains listed; liability retained"
                            + detail + remedy))
                except BaseException as exc:
                    run_error = record_operational_error(exc)
            heartbeat_stop.set()
    if args.role == "root" and args.publish_root_to:
        current_lease = lease_store.read(lease_ref)
        if run_error is not None:
            # The run already failed and says why; a publication that could
            # not happen is a consequence, not a second operational error.
            con.say("publication skipped: the run did not qualify a root")
        elif (archive_verified is None
                or "qualify_root" not in stages_done
                or current_lease.get("state") not in (
                    ABSENCE_CONFIRMED, "TERMINAL")):
            run_error = record_operational_error(RuntimeError(
                "local publication requires a qualified verified archive and "
                "exact pod absence (lease is %s)" % current_lease.get("state")))
        else:
            frozen_suite = Path(frozen_bundle["suite_root"])
            publisher = frozen_suite / "bin/fidelity_dataset.py"
            final_archive = outdir / "result-published.tar.gz"
            clean_environment = dict(os.environ)
            for secret_name in (
                    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                    "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH",
                    "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
                    "TRANSFORMERS_OFFLINE", "HUGGINGFACE_CO_STAGING",
                    "HUGGINGFACE_CO_URL_TEMPLATE", "HF_INFERENCE_ENDPOINT",
                    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                    "HF_ASSETS_CACHE", "HUGGINGFACE_ASSETS_CACHE",
                    "HF_XET_CACHE", "TRANSFORMERS_CACHE",
                    "HF_DATASETS_CACHE", "XDG_CACHE_HOME"):
                clean_environment.pop(secret_name, None)
            publication_hf_home = tempfile.TemporaryDirectory(
                prefix="fidelity-publish-hf-")
            clean_environment["HF_ENDPOINT"] = "https://huggingface.co"
            clean_environment["HF_HOME"] = publication_hf_home.name
            clean_environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            clean_environment["HF_TOKEN_PATH"] = str(
                Path(publication_hf_home.name) / ".no-token")
            try:
                _verify_frozen_suite(
                    frozen_bundle, plan_data["bundle"])
                publication = subprocess.run(
                    [
                        sys.executable, str(publisher), "publish",
                        str(extracted / "dataset"),
                        "--repo", args.publish_root_to,
                        "--qualification",
                        str(extracted / "receipts/root-qualification.json"),
                        "--job", str(extracted / "job.json"),
                        "--result-archive", str(local_archive),
                        "--expected-archive-sha256",
                        archive_verified["archive_sha256"],
                        "--expected-archive-bytes",
                        str(archive_verified["archive_bytes"]),
                        "--expected-head", "absent",
                        "--token-file", args.hf_token_file,
                        "--receipt",
                        str(extracted / "receipts/publish-root.json"),
                        "--revision-message",
                        "fidelity root %s" % job["job_id"],
                    ],
                    cwd=str(frozen_suite), env=clean_environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, check=False)
                if publication.returncode != 0:
                    # The publisher frames its refusal on stdout; a traceback
                    # lands on stderr. Report both or the receipt says nothing.
                    raise RuntimeError(
                        "local atomic root publication failed (exit %d): %s"
                        % (publication.returncode,
                           redact((publication.stdout + publication.stderr)[-1500:])))
                _verify_frozen_suite(
                    frozen_bundle, plan_data["bundle"])
                rebuild = subprocess.run(
                    [
                        sys.executable,
                        str(frozen_suite / "bin/result_archive.py"),
                        "--fs-root", str(extracted), "--verb", "capture",
                        "--status", "completed",
                        "--stages", ",".join(stages_done + ["publish_root"]),
                        "--out", str(final_archive),
                    ],
                    cwd=str(frozen_suite), env=clean_environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, check=False)
                if rebuild.returncode != 0:
                    raise RuntimeError(
                        "local published archive rebuild failed: %s"
                        % redact(rebuild.stderr[-1000:]))
                rebuilt = json.loads(rebuild.stdout.strip())
                archive_verified = verify_archive(
                    final_archive,
                    expected_sha256=rebuilt["sha256"],
                    expected_bytes=rebuilt["bytes"])
                archive_verified["published_archive_path"] = str(final_archive)
            except BaseException as exc:
                run_error = record_operational_error(exc)
            publication_hf_home.cleanup()
    receipt_lease = lease_store.read(lease_ref)
    receipt_campaign = ledger.snapshot()
    receipt_attempt = (
        (receipt_campaign.get("attempts") or {}).get(campaign_key) or {})
    verified_manifest = (
        archive_verified.get("manifest")
        if isinstance(archive_verified, dict) else None)
    scientific_status = (
        str(verified_manifest.get("status"))
        if isinstance(verified_manifest, dict) else "unverified")
    # Operational success means: no operational errors and the pod is proven
    # absent.  Billing settlement and campaign release follow the provider's
    # publication lag and are the installed reaper's job from here; they are
    # reported in the receipt, not required for success.
    lease_state = receipt_lease.get("state")
    pod_gone = lease_state in (ABSENCE_CONFIRMED, "TERMINAL")
    operational_success = not operational_errors and pod_gone
    operational_status = "completed" if operational_success else "failed"
    combined_status = scientific_status
    if (not operational_success
            and scientific_status in (
                "completed", "qualified-unpublished")):
        combined_status = "completed-operational-failure"
    elif not operational_success:
        combined_status = "operational-failure"
    terminal_receipt = {
        "schema": "fidelity-suite/runpod-terminal-receipt.v1",
        "job_id_full": job["job_id_full"],
        "execution_contract_sha256":
            job["execution_attempt"]["execution_contract_sha256"],
        "attempt_id": attempt,
        "provider": "runpod",
        "provider_account_id": current_account_id,
        "provider_resource_id": pod_id,
        "provider_trust": {
            "image_reference_mutable":
                job["environment"]["image_reference_mutable"],
            "ssh_host_key_policy":
                job["environment"]["ssh_host_key_policy"],
            "ssh_endpoint_binding":
                job["environment"]["ssh_endpoint_binding"],
        },
        "lease": {
            "record": lease_ref.path.name,
            "state": receipt_lease.get("state"),
            "generation": receipt_lease.get("generation"),
            "sha256": hashlib.sha256(
                _canonical_bytes(receipt_lease)).hexdigest(),
            "billing": receipt_lease.get("billing_reconciliation"),
        },
        "campaign": {
            "record": Path(ledger_path).name,
            "generation": receipt_campaign.get("generation"),
            "sha256": hashlib.sha256(
                _canonical_bytes(receipt_campaign)).hexdigest(),
            "released": receipt_attempt.get("released") is True,
        },
        "post_create_convergence": post_create_convergence_evidence,
        "result_archive_sha256": (
            archive_verified.get("archive_sha256")
            if isinstance(archive_verified, dict) else None),
        "scientific_status": scientific_status,
        "operational_status": operational_status,
        "combined_status": combined_status,
        "operational_success": operational_success,
        "scientific_error": (
            redact(str(primary_error)) if primary_error is not None else None),
        "operational_errors": [
            redact(str(error)) for error in operational_errors],
        "error": redact(str(run_error)) if run_error is not None else None,
        "receipt_sha256": None,
    }
    terminal_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_bytes(terminal_receipt)).hexdigest()
    terminal_path = outdir / "terminal-receipt.json"
    terminal_tmp = outdir / ".terminal-receipt.tmp"
    with terminal_tmp.open("xb") as stream:
        stream.write(
            json.dumps(
                terminal_receipt, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(str(terminal_tmp), str(terminal_path))
    terminal_tmp.unlink()
    receipt_directory_fd = os.open(str(outdir), os.O_RDONLY)
    try:
        os.fsync(receipt_directory_fd)
    finally:
        os.close(receipt_directory_fd)
    # A lease that never left a no-resource state (nothing acknowledged, and
    # TERMINAL, e.g. the provider refused the create) means nothing was
    # spent and nothing can be leaking: that is a refusal.  A lease still
    # CREATING/AMBIGUOUS/DESTROYING with no acknowledged id may hold a pod
    # the reaper has yet to find; that is a leak-class failure even though
    # `pod_id` is None.
    liability_may_remain = not pod_gone
    nothing_created = pod_id is None and lease_state == "TERMINAL"
    if operational_errors:
        operational_detail = "; ".join(
            redact(str(error)) for error in operational_errors)
        primary_detail = (
            "; primary scientific/execution error: %s"
            % redact(str(primary_error)) if primary_error is not None else "")
        if nothing_created:
            raise Refusal(
                redact(str(primary_error)) if primary_error is not None
                else operational_detail,
                [line for line in operational_detail.split("; ")
                 if primary_error is None
                 or line != redact(str(primary_error))])
        raise RunFailed(
            ("unreconciled operational liability/failure: "
             if liability_may_remain else "run failed after clean teardown: ")
            + operational_detail + primary_detail,
            liability_may_remain=liability_may_remain)
    if run_error is not None:
        if nothing_created:
            raise Refusal(redact(str(run_error)), [])
        raise RunFailed(
            redact(str(run_error)), liability_may_remain=liability_may_remain)
    if not operational_success:
        raise RunFailed(
            "success requires the pod to be proven absent (lease is %s)"
            % lease_state, liability_may_remain=liability_may_remain)
    if archive_verified is None:
        raise RunFailed(
            "success requires a verified off-pod result archive",
            liability_may_remain=False)
    return {"estimated_usd": float(quote.calculated_maximum_usd()),
            "result_archive": archive_verified,
            "secret_cleanup": secret_cleanup}

# ==========================================================================
# Execution
# ==========================================================================




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
            "designated_reference": (plan_data.get("target") or {}).get(
                "designated_reference"),
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
            # ROOT-1: when set, stage_sequence appends `publish_root` and the
            # stage uploads the sealed dataset from the instance.
            "publish_root_to": getattr(args, "publish_root_to", None) or None,
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
            # The device the FORWARD runs on. hf_capture defaults to "cpu" and
            # the stage script never overrode it, so every root capture ran on
            # the CPU of a box rented for its GPU. Default "cuda" here matches
            # what the materialize and measure stages have always passed.
            "device": getattr(args, "capture_device", None) or "cuda",
        }
    return {
        "role": role,
        "capture": capture,
        "recipe": "cloud",
        "job_id": plan_data["job_id"],
        # Stage markers bind to THIS (P1-12/P1-13): the stage runner refuses
        # a marker whose job_id_full differs, which kills both the
        # resume-relabel and the container-reuse silent skips.
        "job_id_full": plan_data.get("job_id_full") or plan_data["job_id"],
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


def _load_secure_hf_token(path_value: str) -> Optional[str]:
    """Read an optional owned 0600 token file without following links."""
    path = Path(path_value).expanduser()
    try:
        descriptor = os.open(
            str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600):
            raise Refusal(
                "Hugging Face token file must be an owned 0600 regular file",
                [])
        raw = os.read(descriptor, 65537)
        if len(raw) > 65536:
            raise Refusal("Hugging Face token file is unexpectedly large", [])
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise Refusal("Hugging Face token file is not UTF-8", []) from exc
    if not token or any(character.isspace() for character in token):
        raise Refusal("Hugging Face token file contains no exact token", [])
    register_secret(token)
    return token


def _load_required_hf_download_token(path_value: Optional[str]) -> str:
    """Load the explicit read credential used only for the remote target fetch."""
    if not path_value:
        raise Refusal("--hf-download-token-file is required", [])
    token = _load_secure_hf_token(path_value)
    if token is None:
        raise Refusal("Hugging Face download token file does not exist", [])
    return token


def _transport_hf_token(
        provider, machine_id, fs_root: str, outdir: Path, token: str,
        *, secrets_dir: Optional[str] = None) -> None:
    """Move the token to the instance without one loose instant at either end.

    Never on a command line: it would land in the remote process list and in
    provider request records. Locally the file is born 0600 inside a 0700
    directory. Remotely the secrets directory is chmod 700 BEFORE anything
    lands in it, the upload goes to a unique temporary name, is tightened to
    0600, and only then atomically renamed to the path the stages read. No
    reader can observe a partial or loose-mode token at `<secrets>/hf_token`.

    Both modes are read back and compared, not assumed: RunPod's /workspace
    volume accepted `chmod 600` and reported 0666 (Fruit smoke, 2026-09-03),
    which is why the RunPod path keeps its secrets on the container disk.
    A filesystem that will not hold the mode is a refusal, never a warning.
    """
    local = outdir / ".secrets-local" / "hf_token"
    write_secret_file(str(local), token)
    remote_dir = secrets_dir or "%s/.secrets" % fs_root
    remote_tmp = "%s/.hf_token.up.%d" % (remote_dir, os.getpid())
    try:
        provider.exec(
            machine_id,
            "set -eu; test ! -e {0}; test ! -L {0}; "
            "mkdir -p -m 700 -- {0}; "
            "test \"$(stat -c %a -- {0})\" = 700; "
            "test ! -e {1}; test ! -L {1}"
            .format(shlex.quote(remote_dir), shlex.quote(remote_tmp)))
        provider.upload(machine_id, str(local), remote_tmp)
        provider.exec(
            machine_id,
            "set -eu; test ! -L {tmp}; test -f {tmp}; "
            "chmod 600 -- {tmp}; "
            "test \"$(stat -c %a -- {tmp})\" = 600; "
            "test ! -e {final}; test ! -L {final}; "
            "mv -- {tmp} {final}"
            .format(tmp=shlex.quote(remote_tmp),
                    final=shlex.quote("%s/hf_token" % remote_dir)))
    except Exception as exc:
        raise RuntimeError(
            "HF token transport refused: the remote secrets directory %s "
            "must be an owner-only 0700 directory holding a 0600 file; the "
            "pod volume did not hold those modes (%s)"
            % (remote_dir, redact(str(exc))[:300]))
    finally:
        shred_secret_file(str(local))


def _runpod_fetch_target_and_remove_token(
        provider, pod_id, fs_root: str, engine_root: str, deadline: float,
        image_reference: str, progress=None) -> Dict[str, Any]:
    """Run the authenticated target fetch and always remove its remote token."""
    stage_error = None
    try:
        _runpod_stage(
            provider, pod_id, fs_root, engine_root, "fetch_target",
            deadline, image_reference, progress=progress)
    except BaseException as exc:
        stage_error = exc
    cleanup = _cleanup_remote_secret(
        provider, pod_id, fs_root, secrets_dir=_runpod_secrets_dir(fs_root))
    if not cleanup.get("confirmed"):
        failure = RuntimeError(
            "remote token erasure after target fetch is unconfirmed; revoke "
            "the RunPod-scoped Hugging Face read token immediately")
        if stage_error is not None:
            raise failure from stage_error
        raise failure
    if stage_error is not None:
        raise stage_error.with_traceback(stage_error.__traceback__)
    return cleanup


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
        _transport_hf_token(
            jl, td.machine_id, td.fs_root, outdir, token)
        con.ok("HF token transported", "0600 in a 0700 dir at both ends, "
               "renamed into place, never argv, shredded at teardown")

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
                            surface=(plan_data.get("target") or {}).get("surface"),
                            publish_root=bool(getattr(args, "publish_root_to",
                                                      None)),
                            candidate=bool(getattr(args, "candidate_scope", None)))
    for stage in stages:
        _run_stage(args, con, jl, td, plan_data, stage)
        if stage == "setup":
            _preflight_bench(args, con, jl, td, plan_data)



def _preflight_bench(
        args, con, jl, td, plan_data, *, fail_closed: bool = False,
        python_executable: Optional[str] = None,
        remote_payload: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        if fail_closed:
            raise Refusal("safe RunPod requires the host benchmark", [])
        return None
    from fidelity.bench import bench_existing, gate

    try:
        doc = bench_existing(
            jl, td.machine_id, con=con,
            python_executable=python_executable,
            remote_payload=remote_payload)
    except Exception as exc:                              # noqa: BLE001
        if fail_closed:
            raise Refusal(
                "safe RunPod host benchmark failed closed: %s"
                % redact(str(exc))[:200], []) from exc
        con.warn("preflight benchmark skipped: %s" % redact(str(exc))[:200])
        return
    if doc.get("error") == "no cuda":
        # Not an advisory: seven of eight Lambda H100-SXM5 launches in the
        # provider survey came up with a healthy nvidia-smi and
        # torch.cuda.is_available() == False (CUDA error 802). A box that
        # cannot see its GPU cannot measure anything; failing here costs the
        # setup stage, staying costs the whole deadline.
        raise Refusal(
            "the rented machine has no working CUDA device "
            "(torch.cuda.is_available() is False while the instance bills)",
            ["This is a per-host fault, not a plan fault -- teardown runs "
             "next; re-run to draw a different host."])
    if fail_closed:
        if doc.get("error") is not None:
            raise Refusal(
                "safe RunPod host benchmark reported an error: %s"
                % redact(str(doc.get("error")))[:200], [])
        malformed = []
        for name in (
                "h2d_GBps", "h2d_cold_GBps",
                "expert_gemm_TFLOPs", "stream_matrix_ms"):
            value = doc.get(name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value <= 0):
                malformed.append(name)
        for name in ("gpu", "torch", "cuda"):
            value = doc.get(name)
            if not isinstance(value, str) or not value.strip():
                malformed.append(name)
        if malformed:
            raise Refusal(
                "safe RunPod host benchmark omitted valid measurements: %s"
                % ", ".join(malformed), [])
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
    return doc


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
            # "alive" OR "unknown-after-retries": both refuse the launch --
            # a network blip must never authorize a second writer (P1-14).
            con.warn("stage %s is (or may be) ALREADY RUNNING on %s -- "
                     "attaching to it instead of launching a second copy"
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
            # ROOT-1 bookkeeping: teardown's unpublished-root guard keys off
            # which of these two stages completed on this box.
            if stage == "verify":
                td.root_verified = True
            elif stage == "publish_root":
                td.root_published = True
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


def _stage_liveness(jl, td, stage: str) -> str:
    """Is `stage_measure.sh <stage>` running? -> "alive" | "dead" | "unknown".

    `[s]tage_measure` and NOT `stage_measure`: `pgrep -f` matches full command
    lines, and this probe's own shell carries the pattern in ITS command line,
    so the naive form finds itself and answers "alive" for a stage that never
    existed (verified against a live instance: `pgrep -f 'stage_measure.sh
    nosuchstage'` -> alive).  The bracket class matches the real process, whose
    cmdline holds "stage_measure.sh", and not the probe, whose cmdline holds
    "[s]tage_measure.sh".  JOURNAL lesson 36 / 44.

    Tri-state on purpose (P1-14): a probe that cannot run -- ssh flake, API
    blip -- used to answer False, the same value as CONFIRMED DEAD, and the
    caller then launched a SECOND writer into a live capture.  "unknown" is
    its own verdict and never authorizes a launch.
    """
    if td.machine_id is None or jl.dry:
        return "dead"
    try:
        out = jl.exec_stdout(
            td.machine_id,
            "pgrep -f '[s]tage_measure.sh %s' >/dev/null 2>&1 "
            "&& echo alive || echo gone" % stage,
            timeout=120, check=False)
    except JLError:
        return "unknown"
    tail = (out or "").strip().splitlines()[-1:]
    if tail == ["alive"]:
        return "alive"
    if tail == ["gone"]:
        return "dead"
    return "unknown"


def _stage_is_alive(jl, td, stage: str, *, retries: int = 3,
                    wait: float = 20.0, sleep=time.sleep) -> bool:
    """True unless the stage is CONFIRMED dead.

    Callers use this to decide "may I launch?" -- and the only answer that
    may authorize a launch is a CONFIRMED "dead".  On "unknown" the probe is
    retried; still unknown after `retries` means the caller must NOT launch a
    duplicate writer (P1-14): the poll loop keeps watching, the marker or a
    later confirmed answer resolves it, and --max-runtime bounds the wait.
    """
    verdict = _stage_liveness(jl, td, stage)
    attempt = 0
    while verdict == "unknown" and attempt < retries:
        attempt += 1
        sleep(wait)
        verdict = _stage_liveness(jl, td, stage)
    return verdict != "dead"


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
    # The host key(s) this run accepted on first use: an on-path substitution
    # after the first connection would have failed loudly, and the receipt
    # says which key the whole run trusted.
    if hasattr(jl, "host_key_fingerprints"):
        fingerprints = jl.host_key_fingerprints()
        if fingerprints:
            cost["ssh_host_key_fingerprints"] = fingerprints
    write_json(str(outdir / "cost-receipt.json"), cost)
    return cost


# ==========================================================================
# CLI
# ==========================================================================


def _runpod_reaper_command(args, con: Console, provider) -> int:
    from fidelity.cloudlease import (
        LeaseStore, install_systemd_user_timer, reap_once,
        systemd_reaper_health,
    )
    store = LeaseStore(Path(args.lease_dir))
    state_dir = Path(args.reaper_state_dir)
    provider_account_id = str(provider.status().get("id") or "").strip()
    if not provider_account_id:
        con.err("RunPod status lacks exact myself.id")
        return EXIT_LEAK
    if args.install and args.dry_run:
        preview = reap_once(store, {"runpod": provider}, dry_run=True)
        con.kv("install", "dry-run; no health/unit/systemctl mutation")
        for failure in preview.failures:
            con.err(json.dumps(failure, sort_keys=True))
        return EXIT_OK if preview.ok else EXIT_LEAK
    if args.install:
        from fidelity.runpodapi import DEFAULT_KEY_FILE
        selected_key_path = (
            args.runpod_key_file or os.environ.get("RUNPOD_KEY_FILE")
            or DEFAULT_KEY_FILE)
        key_path = str(Path(selected_key_path).expanduser().resolve())
        install_systemd_user_timer(
            [sys.executable,
             str(Path(__file__).with_name("reap_cloud_leases.py").resolve()),
             "--provider", "runpod", "--sweep",
             "--lease-dir", str(store.root),
             "--reaper-state-dir", str(state_dir),
             "--runpod-key-file", key_path],
            lease_dir=store.root, provider="runpod",
            provider_account_id=provider_account_id,
            state_dir=state_dir)
        health = systemd_reaper_health(
            state_dir=state_dir, lease_dir=store.root, provider="runpod",
            provider_account_id=provider_account_id)
        if not health["ok"]:
            con.err("installed RunPod reaper timer is not healthy")
            return EXIT_LEAK
        return EXIT_OK
    if args.list:
        # One block per lease that still matters (S2-2): state, the pod ids
        # it authorizes, ages and deadlines, the last event, and for an
        # AMBIGUOUS lease the create-window evidence and what settles it.
        # Terminal leases are a count unless --all; the old output was 96
        # filenames and a state word.
        now = datetime.now(timezone.utc)
        inventory_ids = None
        try:
            inventory = provider.chargeable_inventory()
            if inventory.get("complete"):
                inventory_ids = {
                    row["id"] for row in inventory["families"]["pods"]["resources"]}
        except Exception as exc:                              # noqa: BLE001
            con.warn("provider inventory unavailable (%s); pod presence not shown"
                     % redact(str(exc)))
        terminal = 0
        for ref, document in store.list(include_terminal=True):
            state = document["state"]
            if state == "TERMINAL" and not getattr(args, "all", False):
                terminal += 1
                continue
            create = document.get("create") or {}
            history = document.get("history") or []
            first_at = history[0].get("at") if history else None
            last = history[-1] if history else {}
            con.say("%s  %s" % (ref.path.name[:24], state))
            con.kv("  pod ids", ", ".join(document.get("provider_resource_ids") or [])
                   or "none (no pod was ever attributed to this lease)")
            if inventory_ids is not None and document.get("provider_resource_ids"):
                present = [pod for pod in document["provider_resource_ids"]
                           if pod in inventory_ids]
                con.kv("  in inventory now", ", ".join(present) or "none")
            con.kv("  created", "%s (%s)" % (first_at or "?", _age_text(first_at, now)))
            con.kv("  workload deadline", "%s (%s)" % (
                create.get("workload_deadline_utc") or "?",
                _age_text(create.get("workload_deadline_utc"), now)))
            con.kv("  reap deadline", "%s (%s)" % (
                create.get("reap_deadline_utc") or "?",
                _age_text(create.get("reap_deadline_utc"), now)))
            con.kv("  last event", "%s %s" % (last.get("at") or "?",
                                             last.get("event") or last.get("reason") or "?"))
            if state == "AMBIGUOUS" and not document.get("provider_resource_ids"):
                blockers = ((document.get("terminal_proof") or {})
                            .get("ambiguous_create") or {})
                wrong = blockers.get("wrong_name_new_pod_ids") or []
                con.kv("  blockers", "create window closed with no pod of the exact "
                       "name; %d wrong-name pod(s) appeared in the window: %s"
                       % (len(wrong), ", ".join(wrong) or "none"))
                if inventory_ids is not None:
                    still = [pod for pod in wrong if pod in inventory_ids]
                    con.kv("  those pods now", (
                        "still in inventory: %s" % ", ".join(still) if still
                        else "none exists in the account inventory"))
                con.kv("  needs operator", "the reaper cannot settle a lease with no pod "
                       "id (cloudlease: ambiguous-needs-operator). Verify in the "
                       "RunPod console that no pod named %s exists; then pass "
                       "--allow-unresolved-leases on the next run to proceed beside "
                       "this lease" % (create.get("exact_name") or "fidcloud-*"))
        if terminal:
            con.kv("terminal", "%d lease(s) settled (--all to list them)" % terminal)
        health = systemd_reaper_health(
            state_dir=state_dir, lease_dir=store.root, provider="runpod",
            provider_account_id=provider_account_id)
        con.kv("health", "ok" if health["ok"] else "not healthy")
        return EXIT_OK if health["ok"] else EXIT_LEAK
    result = reap_once(
        store, {"runpod": provider}, dry_run=bool(args.dry_run))
    # Every action row, dry-run or not: what was (or would be) destroyed,
    # what was settled, and which lease needs an operator. Silence used to
    # mean "nothing failed", which read as "nothing to do".
    for action in result.actions:
        detail = {key: value for key, value in action.items()
                  if key not in ("action", "lease") and value not in (None, [], {})}
        con.kv(str(action.get("action")),
               "%s%s" % (str(action.get("lease"))[:24],
                         ("  " + json.dumps(detail, sort_keys=True)[:160]) if detail else ""))
    if not result.actions:
        con.kv("actions", "none: no lease is past its deadline or awaiting settlement")
    for failure in result.failures:
        con.err(json.dumps(failure, sort_keys=True))
    return EXIT_OK if result.ok else EXIT_LEAK


def _age_text(stamp: Optional[str], now: datetime) -> str:
    if not stamp:
        return "unknown"
    try:
        when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return "unparsed"
    delta = int((now - when).total_seconds())
    hours, minutes = divmod(abs(delta) // 60, 60)
    text = "%dh%02dm" % (hours, minutes)
    return (text + " ago") if delta >= 0 else ("in " + text)


def _runpod_drill_contract(args):
    """Build the synthetic, fully bound job used only by the proof producer."""
    args.runpod_drill_manifest_refresh = lambda: {
        "bundle_contract_sha256": hashlib.sha256(_canonical_bytes({
            "bundle": _bundle_manifest(),
            "registry": _bundle_registry_identity(),
        })).hexdigest(),
        "control_manifest_sha256": _control_manifest()["manifest_sha256"],
    }
    bundle = _bundle_manifest()
    registry = _bundle_registry_identity()
    bundle_digest = hashlib.sha256(_canonical_bytes({
        "bundle": bundle, "registry": registry})).hexdigest()
    control = _control_manifest()
    shard_rows = [{"path": "controller-loss-drill.bin", "bytes": 1}]
    shard_digest = hashlib.sha256(_canonical_bytes(shard_rows)).hexdigest()
    job = finalize_job({
        "schema": "fidelity-suite/job.v2", "role": "quant",
        "recipe": "runpod-controller-loss-drill",
        "lane": "fault-drill", "cold_runs": 2,
        "target": {
            "repo_id": "fidelity-suite/runpod-controller-loss-drill",
            "revision": control["manifest_sha256"][:40],
            "config_sha256": hashlib.sha256(b"{}").hexdigest(),
            "index_sha256": hashlib.sha256(b"drill-index").hexdigest(),
            "model_bytes": 1, "shards": shard_rows,
            "shard_manifest_sha256": shard_digest,
            "download_bytes_total": 1,
            "download_manifest": shard_rows,
            "download_manifest_sha256": shard_digest,
        },
        "bundle": bundle, "bundle_registry": registry,
        "bundle_contract_sha256": bundle_digest,
        "control_plane": control,
        "panel": {"kind": "controller-loss-drill", "contexts": 1},
        "reference": {"kind": "controller-loss-drill"},
        "profile": {
            "profile_id": "runpod-drill-secure-l4-on-demand",
            "lane": "fault-drill"},
        "timing": {"kind": "autonomous-reaper-deadline", "seconds": 300},
        "capture": {},
        "scope": {"kind": "controller-loss-drill"},
        "environment": {
            "provider": "runpod", "gpu_type": "L4",
            "provider_gpu_id": "NVIDIA L4", "offer": "on-demand",
            "secure_cloud": True,
        },
        "runtime": {
            "min_vcpu": 4, "min_ram_gb": 16,
            "device": "cpu", "reduce_order": "fp32",
        },
        "produced_by": produced_by_block(
            SUITE_ROOT, "bin/measure_cloud.py",
            dependencies={"mode": "runpod-controller-loss-drill"}),
        "execution_attempt": {
            "kind": "runpod-ssh", "attempt_id": None,
            "cost_quote": None, "engine_root": None,
            "execution_contract_sha256": None,
            "lease_path": None, "pre_create_safety": None,
            "prepared_create": None,
            "remote_root": None,
            "workload_deadline_utc": None,
            "provider_terminate_after": None,
            "planned_at": _exact_utc_now(),
        },
    })
    args.runpod_drill_job_document = job
    args.runpod_drill_job_json = None
    args.runpod_drill_bundle_manifest_sha256 = bundle_digest
    args.runpod_drill_control_manifest_sha256 = control["manifest_sha256"]
def _emit_plan_json(args, con: Console, public_plan: Dict[str, Any]) -> None:
    body = json.dumps(public_plan, sort_keys=True)
    target = getattr(args, "plan_json", None)
    if target:
        path = Path(target).expanduser()
        if path.exists():
            raise Refusal("--plan-json %s exists; name a fresh file" % path, [])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
        con.kv("plan json", "%s (%d bytes)" % (path, len(body) + 1))
    if getattr(args, "json", False):
        con.say(body)
    elif not target:
        con.kv("plan json", "%d bytes; --plan-json FILE writes it, --json prints it"
               % len(body))


def _main_runpod(args, con: Console, provider) -> int:
    if args.subcommand == "adopt":
        con.err("safe RunPod mode refuses adoption/recovery")
        return EXIT_REFUSED
    phase = "plan"
    plan_data = None
    try:
        plan_data = plan_runpod(args, con, provider)
        public_plan = {key: value for key, value in plan_data.items()
                       if not key.startswith("_")}
        if plan_data.get("no_spend") or args.dry_run:
            # The plan JSON is for agents and files, not the terminal: the
            # human block above is the plan a person reads, and the 150-200 KB
            # single line used to be the last thing on their screen (S2-7).
            _emit_plan_json(args, con, public_plan)
            return EXIT_OK
        if not args.yes:
            from fidelity.campaign import CostQuote
            prompt_quote = CostQuote.from_dict(plan_data["cost_quote"])
            budget = (
                "campaign ceiling $%s with reserve $%s"
                % (args.campaign_ceiling, args.campaign_reserve)
                if plan_data.get("campaign_mode") == "explicit"
                else "--max-cost is the whole budget")
            answer = input(
                "Create one secure on-demand RunPod (calculated maximum "
                "$%s; hard cap $%s; %s)? [y/N] " % (
                    prompt_quote.calculated_maximum_usd(),
                    prompt_quote.hard_cap_usd, budget))
            if answer.strip().lower() not in ("y", "yes"):
                return EXIT_REFUSED
        download_token = _load_required_hf_download_token(
            args.hf_download_token_file)
        previous = {}

        def _interrupt(signum, _frame):
            raise KeyboardInterrupt("signal %s" % signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, _interrupt)
        phase = "execute"
        try:
            result = execute_runpod(
                args, con, provider, plan_data, download_token)
        finally:
            download_token = ""
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        con.ok("RunPod result archive verified",
               json.dumps(result, sort_keys=True))
        return EXIT_OK
    except Refusal as exc:
        con.err("REFUSE: %s" % exc.reason)
        for line in exc.advice:
            con.err("        %s" % line)
        return EXIT_REFUSED
    except RunFailed as exc:
        if exc.liability_may_remain:
            con.err("RunPod execution failed and a pod may remain: %s"
                    % redact(str(exc)))
            con.err("        the installed reaper destroys it at its reap "
                    "deadline; check now with: measure-cloud reaper "
                    "--provider runpod --list")
            return EXIT_LEAK
        con.err("RunPod run failed (pod proven gone): %s" % redact(str(exc)))
        return EXIT_FAILED
    except BaseException as exc:
        post_intent = (
            phase == "execute"
            and isinstance(plan_data, dict)
            and plan_data.get("_post_intent_recorded"))
        if not post_intent:
            # No POST intent was ever recorded: nothing was created and
            # nothing can be leaking.  Whatever raised is a refusal with
            # its reason, not a leak.
            con.err("REFUSE: %s" % redact(str(exc)))
            return EXIT_REFUSED
        con.err("RunPod execution failed: %s" % redact(str(exc)))
        con.err("        teardown could not be confirmed; check with: "
                "measure-cloud reaper --provider runpod --list")
        return EXIT_LEAK


def _nonnegative_decimal_arg(value: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError("must be a finite non-negative decimal")
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative decimal")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure-cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Rent one GPU pod, run a fidelity measurement on it, pull the\n"
            "sealed result back, and destroy the pod.\n"
            "\n"
            "What is always enforced: --max-cost caps the whole run's\n"
            "liability (GPU + storage for the full deadline); --max-runtime\n"
            "is an absolute deadline written into the lease, the on-pod\n"
            "watchdog and the provider; the pod is destroyed on success,\n"
            "failure, exception and Ctrl-C; and an installed user-systemd\n"
            "reaper destroys it at the deadline even if this process dies.\n"
            "Nothing is created until every check passes; --dry-run runs\n"
            "them all for $0.00."),
        epilog=(
            "one-time setup (per machine and RunPod account):\n"
            "  measure-cloud reaper --provider runpod --install\n"
            "\n"
            "capture a root fidelity dataset (the common case; --max-runtime is\n"
            "the bound authored in bin/engines.json for this target and\n"
            "--max-cost the all-in maximum it yields -- the dry-run names the\n"
            "new numbers when they move):\n"
            "  measure-cloud --provider runpod --role root \\\n"
            "    --model zai-org/GLM-5.3-BF16 --revision <40-hex> \\\n"
            "    --panel-dir engines/panels/<panel> \\\n"
            "    --dataset-id fidelity--<id> "
            "--publish-root-to <owner>/<repo> \\\n"
            "    --hf-token-file ~/.hf_token --measurer <hub-handle> \\\n"
            "    --max-cost 65 --max-runtime 7h30m "
            "--retrieval-delete-reserve 14400 \\\n"
            "    --out ~/fidelity-runs/<name> --dry-run\n"
            "  # re-run without --dry-run to spend\n"
            "\n"
            "measure a quant against a published root (the candidate route;\n"
            "how every GLM-5.3 quant row was made -- docs/THIRD-PARTY-QUICKSTART.md 3b):\n"
            "  measure-cloud --provider runpod --role candidate \\\n"
            "    --model <owner>/<quant> --revision <40-hex> \\\n"
            "    --panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1 \\\n"
            "    --dataset-id fidelity--glm53.<hub-handle>.quant.<slug> \\\n"
            "    --candidate-scope <scope.json> --candidate-codec exl3-mcg "
            "--candidate-bits 3.25 \\\n"
            "    --reference-dataset malaiwah/glm53-fidelity-root-v1@"
            "9c4a29ee10f393ed2fdbdb9262c1192ddb1507b4 \\\n"
            "    --gpu H200 --runpod-datacenter US-NC-1 --measurer <hub-handle> \\\n"
            "    --max-cost 45 --max-runtime 3h30m "
            "--retrieval-delete-reserve 14400 \\\n"
            "    --out ~/fidelity-runs/<name> --dry-run\n"
            "  # the scope: engines/tools/exl3_scope.py or fp8_scope.py on the\n"
            "  # quant's config.json + model.safetensors.index.json ($0)\n"
            "\n"
            "strict campaign mode (opt-in; cross-run accounting and a\n"
            "sealed controller-loss drill proof bound to this checkout):\n"
            "  measure-cloud drill --provider runpod --campaign-ledger "
            "~/.fidelity-cloud/c.json \\\n"
            "    --campaign-ceiling 48 --campaign-reserve 5 "
            "--campaign-reaper-margin 1 --max-cost 2 --out <dir>\n"
            "  measure-cloud ... --campaign-ledger ~/.fidelity-cloud/c.json "
            "--campaign-ceiling 48 \\\n"
            "    --campaign-reserve 5 --campaign-reaper-margin 1 "
            "--runpod-safety-proof <dir>/proof.json\n"
            "\n"
            "exit codes: 0 ok; 1 the run failed and the pod is proven gone;\n"
            "3 refused before anything was created ($0.00); 90 a pod may\n"
            "remain -- run `measure-cloud reaper --provider runpod --list`."))
    p.add_argument(
        "subcommand", nargs="?", default=None,
        choices=("reaper", "adopt", "drill"),
        help="reaper: install/inspect/sweep the autonomous teardown timer "
             "(--install, --list, --sweep). drill: run a paid controller-loss "
             "drill that seals a safety proof for strict campaign mode. "
             "adopt: not supported on RunPod.")
    p.add_argument(
        "--provider", default="runpod",
        choices=("jarvislabs", "runpod", "vast", "lambda"),
        help="default runpod, the only provider paid measurement runs on; "
             "jarvislabs is accepted solely for `reaper` cleanup of "
             "historical leases, vast and lambda are refused before any "
             "provider mutation.")

    t = p.add_argument_group("what to measure")
    t.add_argument(
        "--role", default="quant", choices=("quant", "root", "candidate"),
        help="quant (default): measure a quantized artifact's divergence "
             "from its reference on the legacy teacher-logits lane and seal "
             "a submittable receipt. root: capture an unquantized reference "
             "model's own activations on a panel and seal a publishable "
             "fidelity dataset; paid for once, read by every later "
             "measurement. candidate: the root protocol on a QUANTIZED "
             "target, scored on the pod against --reference-dataset (the "
             "route behind every GLM-5.3 quant row); requires the four "
             "candidate flags below. `--role root` with those flags is the "
             "same path.")
    t.add_argument("--model", help="Hugging Face repo id to measure, e.g. "
                                   "zai-org/GLM-5.3-BF16")
    t.add_argument(
        "--revision",
        help="exact 40-hex commit of that repo. Required for a paid run so "
             "the receipt names bytes, not a moving branch. Omit to have "
             "--dry-run resolve and print main's current commit.")
    t.add_argument(
        "--path",
        help="which artifact inside a repo that ships several at one "
             "revision (a GGUF shelf: --path UD-Q4_K_XL). The planner lists "
             "the choices when there are several.")
    t.add_argument(
        "--measurer",
        help="your Hugging Face handle, credited as the measurer on the "
             "sealed receipt. Required for anything that may be published.")
    t.add_argument(
        "--registry", default="auto",
        help="where the already-measured front gate reads from: auto "
             "(default), hf, or local[:PATH]")

    rt = p.add_argument_group("root capture (--role root)")
    rt.add_argument(
        "--panel-dir",
        help="local panel directory (panel.json + arrays/) shipped to the "
             "pod. Build one for a new model family with "
             "engines/tools/build_token_panel.py against that family's own "
             "tokenizer.")
    rt.add_argument(
        "--dataset-id",
        help="identity of the fidelity dataset this capture writes, e.g. "
             "fidelity--glm53.malaiwah.root.bf16. Required for root.")
    rt.add_argument(
        "--publish-root-to", metavar="OWNER/REPO",
        help="after qualification and verified retrieval, publish the "
             "dataset to this public Hub dataset repo from this machine "
             "(the token never reaches the pod). Optional; without it the "
             "sealed dataset stays under --out.")
    rt.add_argument(
        "--dataset-repository", metavar="OWNER/REPO",
        help="canonical repo identity recorded in the dataset. Defaults to "
             "--publish-root-to.")
    rt.add_argument(
        "--dataset-name",
        help="human-readable dataset name. Defaults to --dataset-id.")
    rt.add_argument(
        "--unexpected-tensor-allowlist",
        help="checked-in JSON list of tensor names this exact revision "
             "carries beyond what its architecture declares. Resolved "
             "automatically when one is authored for the target; only needed "
             "for a new model that turns out to need one.")
    rt.add_argument(
        "--resume-capture", default=None,
        help="root only: a sealed dataset directory captured earlier with "
             "exactly this recipe (same weights pin, panel, image and GPU "
             "model). It becomes cold run 1 on the pod; cold run 2 is "
             "captured fresh and must reproduce it bitwise for "
             "qualification. Verified locally (tensors recomputed) before "
             "any spend, and again by the pod. The imported capture is the "
             "one published, so it must be publishable: captures sealed "
             "before 2026-09-04 carry their pod path in validation/ and "
             "the publisher refuses them.")
    rt.add_argument(
        "--resume-origin-job", default=None,
        help="the executed job.json that produced --resume-capture; its "
             "identity is recorded as the imported cold run's origin")
    rt.add_argument(
        "--runpod-datacenter", default=None, metavar="ID",
        help="pin the RunPod secure datacenter (e.g. US-MO-1); the create "
             "refuses rather than falls back elsewhere. Recorded on the "
             "plan; the live attestation records the datacenter served.")
    rt.add_argument(
        "--runpod-image", default=None, metavar="NAME@sha256:HEX",
        help="the pod image, pinned by digest (default: runpod/pytorch, on "
             "which the locked stack is rebuilt every pod). The measurement "
             "image's ssh target, ghcr.io/malaiwah/quant-fidelity-measure:ssh, "
             "boots with the stack baked and the bootstrap seeds from it.")
    rt.add_argument(
        "--storage-layout", choices=sorted(RUNPOD_STORAGE_LAYOUTS),
        default="container-disk",
        help="where the run root lives on the pod: container-disk (the "
             "host's local NVMe; default, ~10x faster weight streaming) or "
             "pod-volume (/workspace, MooseFS on RunPod secure cloud)")
    rt.add_argument(
        "--sanity-expect", default="Paris",
        help="the continuation the on-pod generation probe requires for "
             "\"The capital of France is\" (default Paris, enforced). Pass '' "
             "to record the probe without enforcing it -- only for an "
             "undertrained proxy such as the 5B Fruit fixture, never a "
             "production root; the plan warns and the dataset records the "
             "probe's verdict either way.")
    rt.add_argument(
        "--panel-tokenizer-root",
        help="local pinned tokenizer receipt files; otherwise prefetched")
    rt.add_argument("--form", default="hidden", choices=("hidden", "logit"),
                    help=argparse.SUPPRESS)
    rt.add_argument("--schedule", default="layer-outer",
                    choices=("layer-outer", "window-outer", "window-major"),
                    help=argparse.SUPPRESS)
    rt.add_argument("--capture-device", default="cuda", help=argparse.SUPPRESS)
    rt.add_argument("--replay-device", default="numpy", help=argparse.SUPPRESS)
    rt.add_argument("--replay-dtype", default="float32", help=argparse.SUPPRESS)
    rt.add_argument("--replay-vocab-chunk", type=int, default=8192,
                    help=argparse.SUPPRESS)

    cand = p.add_argument_group(
        "candidate measurement (--role candidate; all four flags go together)")
    cand.add_argument(
        "--candidate-scope", default=None, metavar="SCOPE.json",
        help="the authored quantization scope of the target (tensor classes, "
             "formats, bits), read off its index by engines/tools/exl3_scope.py "
             "or fp8_scope.py and validated here against the registry's scope "
             "rules at $0. The dataset is captured under this scope; the "
             "loader decodes the checkpoint's exl3 trellis payloads or "
             "block-scaled FP8 to bf16 per module (weights-only, "
             "dequantize-and-run); the qualified capture is scored against "
             "--reference-dataset on the pod, and the comparison receipt "
             "travels in the result archive.")
    cand.add_argument(
        "--candidate-codec", default=None, metavar="CODEC",
        help="the candidate's codec in the registry's closed numeric_format "
             "vocabulary (registry/schema/common.schema.json), e.g. exl3-mcg, "
             "exl3-trellis, fp8_e4m3; anything else is refused with the list")
    cand.add_argument("--candidate-bits", type=float, default=None, metavar="BITS",
                      help="declared bits per weight of the candidate, e.g. 3.25 or 8; "
                           "checked against the checkpoint's own declaration "
                           "(quantization_config.bits / hybrid_tr3_tail.bits_avg) "
                           "before spend and against the payload on the pod")
    cand.add_argument(
        "--reference-dataset", default=None, metavar="OWNER/REPO@40HEX",
        help="the published root dataset the candidate is scored against; its "
             "seal, content digest and panel are bound into the job and the "
             "pod refuses any other dataset under that name")

    pl = p.add_argument_group("quant measurement (--role quant)")
    pl.add_argument("--panel", help="Hub dataset id of the panel/teacher")
    pl.add_argument("--panel-revision", help="exact revision of that panel")
    pl.add_argument("--panel-descriptor",
                    help="JSON file with include globs, contexts, positions")
    pl.add_argument(
        "--scope-json",
        help="JSON file carrying the artifact's quantization scope "
             "(policy/head_policy/kv_cache_dtype/assignments) read off the "
             "release; copied verbatim into the receipt. Without it the "
             "receipt records the honest default.")
    pl.add_argument("--lane", default="streaming",
                    choices=("streaming", "sealed-ep8"), help=argparse.SUPPRESS)
    pl.add_argument("--reduce-order", default="fp32",
                    choices=("fp32", "native"), help=argparse.SUPPRESS)
    pl.add_argument("--cold-runs", type=int, default=None,
                    help=argparse.SUPPRESS)

    i = p.add_argument_group("where to run, and how much it may cost")
    i.add_argument(
        "--max-cost", type=_nonnegative_decimal_arg, metavar="USD",
        help="REQUIRED. Refuse if the all-in maximum (GPU rate x deadline "
             "+ storage for the deadline + retrieval reserve) exceeds this. "
             "There is no default: a cap the tool picked for you would turn "
             "a legitimate run into a refusal you cannot attribute.")
    i.add_argument(
        "--max-runtime", metavar="DURATION",
        help="absolute workload deadline like 3h30m, written into the lease "
             "(the reaper destroys the pod at it), the on-pod watchdog and "
             "the provider's own timer. REQUIRED for a target with no "
             "authored timing row; for a root with one (bin/engines.json "
             "root_timing_profiles) it defaults to that bound and the plan "
             "says so -- pass it only to override upward.")
    i.add_argument(
        "--gpu",
        help="GPU class, e.g. H200, L4, A100. Defaults to the one named by "
             "the target's authored timing evidence; required for a target "
             "that has none.")
    i.add_argument(
        "--storage", type=int, metavar="GB",
        help="pod volume size (default: computed from the checkpoint plus "
             "both cold captures)")
    i.add_argument("--min-vcpu", type=int,
                   help="override the derived minimum host vCPU count")
    i.add_argument("--min-memory-gb", type=int,
                   help="override the derived minimum host memory")
    i.add_argument(
        "--retrieval-delete-reserve", type=int, default=None, metavar="SEC",
        help="seconds funded after the workload deadline for archive build, "
             "up to three bounded download attempts with their local "
             "verification, and the final delete. Default: exactly that "
             "contract's minimum, derived from the result archive bound "
             "(1800 + 3 x (3600 + verify) + 300; 13818 s for a 5 GB "
             "archive) and printed in the plan; pass a larger value to "
             "override upward. Part of the cost cap.")
    i.add_argument("--region", default="secure", help=argparse.SUPPRESS)
    i.add_argument("--spot", dest="spot", action="store_true", default=False,
                   help=argparse.SUPPRESS)
    i.add_argument("--on-demand", dest="spot", action="store_false",
                   help=argparse.SUPPRESS)
    i.add_argument("--on-preempt", default="fail",
                   choices=("resume", "recreate", "fail"),
                   help=argparse.SUPPRESS)
    i.add_argument("--heartbeat-timeout", type=int, default=900,
                   help=argparse.SUPPRESS)
    i.add_argument("--max-preemptions", type=int, default=3,
                   help=argparse.SUPPRESS)
    i.add_argument("--timer-api-lag", type=int, default=600,
                   help=argparse.SUPPRESS)
    i.add_argument("--min-h2d-gbps", type=float,
                   help="abort after setup if the pod's measured "
                        "host-to-device bandwidth is below this (GB/s). A "
                        "card wired at PCIe x1 turns a 3-hour run into 20; "
                        "8 is a reasonable floor for x16.")
    i.add_argument("--min-gemm-tflops", type=float,
                   help="abort after setup if the measured bf16 GEMM is "
                        "below this")

    c = p.add_argument_group("credentials (files only; never argv or env)")
    c.add_argument(
        "--runpod-key-file", metavar="FILE",
        help="RunPod API key file (default ~/.config/runpod/api_key)")
    c.add_argument(
        "--hf-token-file", metavar="FILE",
        default=os.path.expanduser("~/.cache/huggingface/token"),
        help="owner-only 0600 Hugging Face token file used on THIS machine "
             "for publication (default ~/.cache/huggingface/token)")
    c.add_argument(
        "--hf-download-token-file", metavar="FILE",
        help="owner-only 0600 Hugging Face READ token file transported to "
             "the pod for the target download and shredded right after. "
             "Defaults to --hf-token-file when that file exists; give a "
             "separate read-only token if you prefer not to ship a write "
             "token to a rented machine.")

    o = p.add_argument_group("output and control")
    o.add_argument("--out", metavar="DIR",
                   help="local output directory (must not exist yet). "
                        "Receives job.json, the verified result archive, the "
                        "extracted dataset and terminal-receipt.json.")
    o.add_argument("--dry-run", action="store_true",
                   help="run every check, print the plan, create nothing, "
                        "spend $0.00")
    o.add_argument("--yes", action="store_true",
                   help="skip the spend confirmation prompt")
    o.add_argument(
        "--allow-unresolved-leases", action="store_true",
        help="proceed even though an earlier lease may still hold a pod "
             "(the reaper destroys it at its deadline regardless). Without "
             "this the run refuses and names the lease.")
    o.add_argument("--install", action="store_true",
                   help="with `reaper`: install the user-systemd template "
                        "timer fidelity-cloud-reaper@<provider>.timer from "
                        "this checkout (one instance per provider account)")
    o.add_argument("--sweep", action="store_true",
                   help="with `reaper`: run one sweep now")
    o.add_argument("--list", action="store_true",
                   help="with `reaper`: one block per unresolved lease (state, "
                        "pod ids, deadlines, last event; for an AMBIGUOUS lease "
                        "its blockers and what settles it) plus the timer's "
                        "health; terminal leases are a count")
    o.add_argument("--all", action="store_true",
                   help="with `reaper --list`: list terminal leases too")
    o.add_argument("--plan-json", metavar="FILE",
                   help="with --dry-run: write the full plan JSON to this new "
                        "file (--out stays untouched)")
    o.add_argument("--json", action="store_true",
                   help="with --dry-run: also print the full plan JSON to stdout "
                        "as the last line (for agents; ~150 KB)")
    o.add_argument("--lease-dir", metavar="DIR",
                   default=str(Path.home() / ".fidelity-cloud" / "leases-v2"),
                   help="lease directory (default ~/.fidelity-cloud/leases-v2)")
    o.add_argument("--reaper-state-dir", metavar="DIR",
                   default=str(Path.home() / ".fidelity-cloud"),
                   help="reaper state directory (default ~/.fidelity-cloud)")
    o.add_argument(
        "--runpod-billing-wait", type=int, default=1800, metavar="SEC",
        help="how long to wait after teardown for the first billing read "
             "(default 1800). Billing is advisory: the reaper settles it "
             "later if RunPod has not published the bucket yet.")

    s = p.add_argument_group(
        "strict campaign mode (opt-in; all four flags go together)")
    s.add_argument(
        "--campaign-ledger", metavar="FILE",
        help="a locked ledger beside --lease-dir that accounts for EVERY "
             "attempt against one ceiling, refuses admission beside pods it "
             "does not own, and holds liability until billing settles. "
             "Without it, each run gets its own ledger with ceiling = "
             "--max-cost and foreign pods are tolerated.")
    s.add_argument("--campaign-ceiling", metavar="USD",
                   help="total campaign liability ceiling")
    s.add_argument("--campaign-reserve", metavar="USD",
                   help="balance floor the campaign never spends below")
    s.add_argument("--campaign-reaper-margin", metavar="USD",
                   help="liability held back for the reaper's own cleanup")
    s.add_argument(
        "--runpod-safety-proof", metavar="FILE",
        help="proof.json sealed by `measure-cloud drill` for this exact "
             "checkout, ledger and account; validated as evidence that the "
             "installed reaper destroyed a real pod after this controller "
             "was killed. Requires --campaign-ledger.")
    s.add_argument("--campaign-width", type=int, default=1,
                   help="concurrent attempts admitted (1 or 2; 2 needs "
                        "--width-two-root-archive)")
    s.add_argument("--width-two-root-archive", help=argparse.SUPPRESS)
    s.add_argument("--campaign-name", default="fidcloud-",
                   help=argparse.SUPPRESS)

    d = p.add_argument_group("drill (with the `drill` subcommand)")
    from fidelity.runpoddrill import (
        DEFAULT_TERMINATE_SECONDS, DEFAULT_WORKLOAD_SECONDS)
    d.add_argument("--runpod-drill-workload-seconds", type=int,
                   default=DEFAULT_WORKLOAD_SECONDS, metavar="SEC",
                   help="seconds the drill pod must reach ready within "
                        "(default %d)" % DEFAULT_WORKLOAD_SECONDS)
    d.add_argument("--runpod-drill-terminate-seconds", type=int,
                   default=DEFAULT_TERMINATE_SECONDS, metavar="SEC",
                   help="seconds until the reaper must have destroyed the "
                        "drill pod (default %d)" % DEFAULT_TERMINATE_SECONDS)
    d.add_argument("--runpod-drill-poll-seconds", type=int, default=15,
                   metavar="SEC", help="provider poll interval (default 15)")
    d.add_argument(
        "--runpod-drill-billing-wait-seconds", type=int, default=7200,
        metavar="SEC",
        help="how long the drill waits for RunPod to publish the pod's "
             "billing bucket (default 7200; RunPod publishes hourly and "
             "settles a few minutes after the hour closes)")

    a = p.add_argument_group("storage tariffs (defaults are RunPod's "
                             "published per-GB-month rates)")
    a.add_argument("--runpod-container-running-tariff", default="0.10",
                   metavar="USD")
    a.add_argument("--runpod-container-stopped-tariff", default="0.00",
                   metavar="USD")
    a.add_argument("--runpod-pod-running-tariff", default="0.10",
                   metavar="USD")
    a.add_argument("--runpod-pod-stopped-tariff", default="0.20",
                   metavar="USD")
    a.add_argument("--runpod-network-tariff", default="0.07", metavar="USD")
    a.add_argument("--tariff-effective-at",
                   default=RUNPOD_STORAGE_TARIFF_PINNED_AT,
                   help="when the tariff defaults were last verified; older "
                        "than %d days prints a reminder"
                        % RUNPOD_STORAGE_TARIFF_STALE_AFTER_DAYS)

    # Legacy flags from the JarvisLabs controller.  Parsed so old scripts
    # fail with a clear refusal instead of an argparse error; every one is
    # refused before spend on RunPod.
    legacy = p.add_argument_group("legacy (refused on RunPod)")
    for flag in ("--race", "--preview-of", "--designated-reference",
                 "--skip-registry-check", "--force",
                 "--accept-measured-revision", "--no-preflight-bench",
                 "--i-accept-leak-risk", "--allow-unpublished-root",
                 "--hold-on-failure", "--keep-fs", "--keep-student-logits"):
        if flag == "--preview-of":
            legacy.add_argument(flag, metavar="FINAL_DATASET_ID",
                                help=argparse.SUPPRESS)
        else:
            legacy.add_argument(flag, action="store_true",
                                help=argparse.SUPPRESS)
    legacy.add_argument("--race-workers", type=int, default=8,
                        help=argparse.SUPPRESS)
    legacy.add_argument("--fs-id", type=int, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()

    if args.subcommand == "reaper":
        if args.provider is None:
            con.err("reaper requires explicit --provider runpod; "
                    "historical JarvisLabs lease cleanup requires explicit "
                    "--provider jarvislabs")
            return EXIT_REFUSED
        if args.provider == "runpod":
            from fidelity.runpodapi import RunPod
            provider = RunPod(
                dry=bool(args.dry_run), key_file=args.runpod_key_file)
            return _runpod_reaper_command(args, con, provider)
        if args.provider != "jarvislabs":
            con.err("reaper supports RunPod and historical JarvisLabs cleanup")
            return EXIT_REFUSED
        if args.install:
            return reaper_install(con)
        if args.sweep:
            return reaper_sweep(con, dry=args.dry_run)
        return reaper_list(con)
    if args.subcommand == "drill":
        if args.provider != "runpod":
            con.err("drill subcommand is RunPod-only")
            return EXIT_REFUSED
        from fidelity.runpodapi import RunPod
        from fidelity.runpoddrill import run_drill
        try:
            _runpod_drill_contract(args)
            provider = RunPod(
                dry=bool(args.dry_run), key_file=args.runpod_key_file)
            return run_drill(args, con, provider)
        except (Refusal, ValueError, OSError) as exc:
            con.err("RunPod drill refused: %s" % redact(str(exc)))
            return EXIT_REFUSED

    if getattr(args, "role", "quant") == "candidate":
        # The spelled-out name of "the root protocol on a quantized target".
        # One code path: everything downstream reads role == "root" and
        # branches on candidate_scope, exactly as the m-* jobs were run.
        missing = [flag for flag, value in (
            ("--candidate-scope", args.candidate_scope),
            ("--candidate-codec", args.candidate_codec),
            ("--candidate-bits", args.candidate_bits),
            ("--reference-dataset", args.reference_dataset)) if value is None]
        if missing:
            con.err("--role candidate requires %s (docs/THIRD-PARTY-QUICKSTART.md 3b)"
                    % ", ".join(missing))
            return EXIT_REFUSED
        args.role = "root"
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
        if args.publish_root_to:
            from fidelity import dshub
            try:
                dshub.validate_repo_id(args.publish_root_to)
            except dshub.HubError as exc:
                con.err("--publish-root-to: %s" % exc)
                return EXIT_REFUSED
        if args.publish_root_to and getattr(args, "preview_of", None) and \
                args.publish_root_to.rsplit("/", 1)[-1] == args.preview_of:
            con.err("--publish-root-to %r wears the FINAL dataset's name while "
                    "--preview-of says this capture is its PREVIEW. A preview "
                    "and a final are two datasets with two identities "
                    "(docs/RACE-MODE.md); publish the preview under its own "
                    "repo (convention: %s-preview) and the final under %s "
                    "after run 2 + verify + self-compare."
                    % (args.publish_root_to, args.publish_root_to,
                       args.preview_of))
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
            con.warn("--race without --preview-of: "
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
            if args.provider != "runpod":
                try:
                    pd.relative_to(SUITE_ROOT)
                except ValueError:
                    con.err(
                        "--panel-dir must live inside the suite checkout (%s): "
                        "the legacy uploader addresses suite-relative files"
                        % SUITE_ROOT)
                    return EXIT_REFUSED
    elif not args.model or not args.panel:
        con.err("--model and --panel are required")
        return EXIT_REFUSED
    if getattr(args, "publish_root_to", None) and             getattr(args, "role", "quant") != "root":
        con.err("--publish-root-to is --role root only: a quant measurement "
                "publishes through the registry, not as a dataset repo")
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

    if args.provider == "runpod":
        from fidelity.runpodapi import RunPod
        provider = RunPod(
            dry=bool(args.dry_run), key_file=args.runpod_key_file)
        return _main_runpod(args, con, provider)

    con.err(
        "paid measurement execution requires explicit --provider runpod; "
        "provider %s is refused before any provider mutation"
        % (args.provider if args.provider is not None else "<missing>"))
    return EXIT_REFUSED



if __name__ == "__main__":
    raise SystemExit(main())
