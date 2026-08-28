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
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import census as C                       # noqa: E402
from fidelity.common import (                          # noqa: E402
    Console, human_bytes, human_duration, parse_duration, read_json,
    redact, register_secret, utcnow, write_json,
)
from fidelity.engines import EngineUnpinned, build_invocation, load_engines  # noqa: E402
from fidelity.hfmeta import (                          # noqa: E402
    HFError, RepoMeta, hf_token, load_panel_descriptor, repo_meta, sniff_surface,
)
from fidelity.jlapi import JL, JLError, JLNotInstalled, select_offer  # noqa: E402
from fidelity.receipt import produced_by_block                      # noqa: E402

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
        self.fs_root = "/home/jl_fs/fidelity"
        self.lease_path: Optional[Path] = None
        self.done = False
        self.leaked = False
        self._lock = threading.Lock()

    def adopt(self, machine_id: Optional[int]) -> None:
        """Adopt a (possibly renumbered) machine id and persist it immediately.

        `jl resume` can return a NEW machine_id.  Anything that does not adopt
        it unconditionally will destroy the wrong box, or nothing at all.
        """
        self.machine_id = machine_id
        if self.lease_path and self.lease_path.is_file():
            try:
                lease = read_json(str(self.lease_path))
                lease["machine_id"] = machine_id
                lease["updated_at"] = utcnow()
                write_json(str(self.lease_path), lease)
            except OSError:
                pass

    def run(self, reason: str = "") -> None:
        with self._lock:
            if self.done:
                return
            self.done = True
        if self.machine_id is None and self.fs_id is None:
            # Nothing to destroy, but a lease may already be on disk: it is
            # written BEFORE `jl create` on purpose. Leaving it behind makes
            # `reaper --list` report a phantom job forever.
            self._drop_lease()
            return
        # Printing "do NOT interrupt" is not a defence.  A second ^C re-enters
        # the signal handler, finds done=True, no-ops, and sys.exit()s straight
        # through the destroy that has not happened yet -- which leaks the
        # instance at the exact moment the user was trying to stop the bill.
        # Take the choice away for the duration instead of asking for it.
        prev = {}
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                prev[sig] = signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass
        self.con.say("")
        self.con.step("teardown%s (^C is ignored until this finishes)"
                      % ((" -- " + reason) if reason else ""))
        try:
            for step in (self._pull_receipts, self._collect_env, self._shred_secrets,
                         self._destroy_instance, self._destroy_fs, self._drop_lease):
                try:
                    step()
                except Exception as exc:                # noqa: BLE001
                    # A failure inside teardown must never skip the destroy that
                    # comes after it.  That is the whole reason each step is
                    # individually wrapped instead of the block as a whole.
                    self.con.warn("teardown step %s: %s"
                                  % (step.__name__, redact(str(exc))))
        finally:
            for sig, handler in prev.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    # -- steps -------------------------------------------------------------

    def _pull_receipts(self) -> None:
        if self.machine_id is None or self.jl.dry:
            return
        dest = self.outdir / "receipts"
        dest.mkdir(parents=True, exist_ok=True)
        self.con.step("pulling receipts (timeout %ds)" % int(self.pull_timeout))
        self.jl.download(self.machine_id, self.fs_root + "/receipts", str(dest),
                         recursive=True, timeout=self.pull_timeout)
        n = len(list(dest.rglob("*")))
        self.con.ok("receipts pulled", "%d entries" % n)

    def _collect_env(self) -> None:
        if self.machine_id is None or self.jl.dry:
            return
        try:
            out = self.jl.exec(
                self.machine_id,
                "nvidia-smi || true; df -h || true; "
                "%s/venv/bin/pip freeze 2>/dev/null || true" % self.fs_root,
                timeout=120)
            (self.outdir / "environment.txt").write_text(
                redact(out if isinstance(out, str) else json.dumps(out, indent=2)),
                encoding="utf-8")
        except Exception:                               # noqa: BLE001
            pass

    def _shred_secrets(self) -> None:
        if self.machine_id is None or self.jl.dry:
            return
        self.jl.exec(self.machine_id,
                     "shred -u %s/.secrets/* 2>/dev/null || rm -f %s/.secrets/* 2>/dev/null; true"
                     % (self.fs_root, self.fs_root), timeout=120)
        self.con.ok("secrets shredded")

    def _destroy_instance(self) -> None:
        if self.machine_id is None:
            return
        if self.jl.dry:
            self.con.ok("would destroy instance", str(self.machine_id))
            return
        mid = self.machine_id
        for attempt in range(5):
            try:
                self.jl.destroy(mid)
            except JLError as exc:
                self.con.warn("destroy attempt %d: %s" % (attempt + 1, redact(str(exc))))
            time.sleep(min(2 ** attempt, 20))
            inst = self.jl.get(mid)
            if inst is None or inst.status.lower() in ("destroyed", "terminated", ""):
                self.con.ok("instance destroyed", str(mid))
                self.machine_id = None
                return
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


def _machine_id_of(created: Optional[Dict[str, Any]]) -> Optional[int]:
    """Pull a machine id out of whatever shape `jl create` answered with.

    A vendor that renames `machine_id` to `id` must not turn into a leaked
    instance, so this accepts either and returns None rather than raising.
    """
    if not isinstance(created, dict):
        return None
    for key in ("machine_id", "id", "instance_id"):
        value = created.get(key)
        if isinstance(value, (int, str)) and str(value).isdigit():
            return int(value)
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
    key = json.dumps({
        "model": args.model, "revision": args.revision, "panel": args.panel,
        "lane": args.lane, "spot": args.spot, "cold_runs": args.cold_runs,
    }, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def plan(args: argparse.Namespace, con: Console, jl: JL) -> Dict[str, Any]:
    """Everything that must be true BEFORE money is spent."""
    plan: Dict[str, Any] = {"job_id": job_id_for(args), "created": False}

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
        con.warn("WOULD REFUSE (real run): %s" % refusal.reason)
        for line in refusal.advice[:3]:
            con.say("           %s" % line)
        plan.setdefault("would_refuse", []).append(refusal.reason)

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
        surface = sniff_surface(target)
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
        }
        if surface.problems:
            raise Refusal(
                "this artifact cannot be read by any available surface adapter",
                surface.problems + [
                    "",
                    "This is detected from the repo's own metadata, at a cost of a "
                    "few hundred kilobytes, so it costs nothing to find out.",
                    "Nothing was created. $0.00 spent.",
                ])
        artifact_bytes = float(target.total_bytes)
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

    need = C.storage_need(artifact_bytes=artifact_bytes, panel_bytes=panel_bytes,
                          keep_student_logits=args.keep_student_logits)
    storage_gb = args.storage or C.round_up_storage_gb(need.total_bytes)
    con.kv("disk", "%s artifact + %s panel + %s toolchain + 15%% -> %d GB fs"
           % (human_bytes(artifact_bytes), human_bytes(panel_bytes),
              human_bytes(need.toolchain_bytes), storage_gb), indent=4)
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
            con.warn(
                "chose %s ($%.2f/h) because it is the cheapest row that fits, but "
                "lane %s was validated on %s. The VRAM arithmetic holds; the "
                "observed-peak figure it is sized against does not transfer "
                "automatically to another architecture. Pass --gpu %s to match "
                "the validation hardware."
                % (offer.gpu_type, offer.price, args.lane, validated_on, validated_on))
            plan["chosen"]["validated_hardware"] = validated_on
            plan["chosen"]["on_validated_hardware"] = False
        elif validated_on:
            plan["chosen"]["on_validated_hardware"] = True

    # -- engine ------------------------------------------------------------
    con.say("")
    engines = load_engines()
    engine = engines.get(args.lane)
    if engine is None:
        raise Refusal("no engine configured for lane %r" % args.lane,
                      ["known lanes: " + ", ".join(sorted(engines))])
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
        con.warn("WOULD REFUSE (real run): lane %r has no pinned engine" % args.lane)
        con.say("           engine %s: %s"
                % (engine.entrypoint, engine.unpinned_reason))
        plan.setdefault("would_refuse", []).append(
            "lane %s engine unpinned (%s)" % (args.lane, engine.entrypoint))
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
    per_window = float(timing.get("minutes_per_window", 9.05))
    measured = bool(timing.get("measured"))
    phases = [
        ("bootstrap", 0.42, "apt + cuda13 + torch + exllamav3 build"),
        ("fetch", max(0.05, fetch_gb / 190.0 / 3600.0 * 1000.0), "%.0f GB @ ~190 MB/s" % fetch_gb),
        ("measure", args.cold_runs * descriptor.contexts * per_window / 60.0,
         "%d run(s) x %d windows @ ~%.1f min%s"
         % (args.cold_runs, descriptor.contexts, per_window,
            "" if measured else " ESTIMATED")),
        ("seal + pull", 0.08, ""),
    ]
    total_h = sum(h for _, h, _ in phases)
    plan["timing"] = dict(timing, minutes_per_window=per_window)
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
        con.warn("WOULD REFUSE (real run): %s" % refusal.reason)
        plan.setdefault("would_refuse", []).append(refusal.reason)

    if args.max_cost and band_hi > args.max_cost:
        refusal = Refusal(
            "estimated band high $%.2f exceeds --max-cost $%.2f" % (band_hi, args.max_cost),
            ["raise --max-cost, or pick a cheaper lane/GPU",
             "Nothing was created. $0.00 spent."])
        if not args.dry_run:
            raise refusal
        con.warn("WOULD REFUSE (real run): %s" % refusal.reason)
        plan.setdefault("would_refuse", []).append(refusal.reason)

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
    for inst in jl.list_instances():
        if inst.name == plan_data["instance_name"] and inst.status == "Running":
            con.step("adopting existing instance %d for this job" % inst.machine_id)
            td.adopt(inst.machine_id)
            td.fs_id = inst.fs_id
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
                spot=args.spot, region=region, fs_id=td.fs_id, storage=100,
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
                td.adopt(int(mid))
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
    return {
        "recipe": "cloud",
        "job_id": plan_data["job_id"],
        "lane": args.lane,
        "reduce_order": args.reduce_order,
        "cold_runs": args.cold_runs,
        "profile": "k6" if args.lane == "sealed-ep8" else "k4",
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
            "host": "jarvislabs",
        },
        "keep_student_logits": bool(args.keep_student_logits),
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
    con.step("uploading bundle (%d files)" % len(files))
    made: set = set()
    for rel in files:
        src = SUITE_ROOT / rel
        if not src.is_file():
            con.warn("bundle entry not present locally, skipped: %s" % rel)
            continue
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

    for stage in ("setup", "fetch_target", "fetch_panel", "measure", "seal"):
        _run_stage(args, con, jl, td, plan_data, stage)


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
        run = jl.run_job(td.machine_id,
                         "bash %s/bin/stage_measure.sh %s" % (td.fs_root, stage))
        run_id = (run or {}).get("run_id") or (run or {}).get("id")
        outcome = _await_stage(con, jl, td, run_id, stage, deadline)
        if outcome == "done":
            con.ok("stage %s" % stage, human_duration(time.time() - started))
            return
        if outcome == "failed":
            raise RuntimeError(
                "stage %s failed on the instance; the log was pulled to the "
                "output directory and the instance will now be destroyed" % stage)
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
                           "bash %s/bin/stage_measure.sh setup" % td.fs_root)
            _await_stage(con, jl, td, (r or {}).get("run_id"), "setup", deadline)


def _await_stage(con, jl, td, run_id, stage: str, deadline: float) -> str:
    """Poll until the stage ends. Returns done | failed | preempted | deadline."""
    quiet = 0
    while True:
        if time.time() > deadline:
            return "deadline"
        time.sleep(120)
        inst = jl.get(td.machine_id) if td.machine_id else None
        if inst is None or inst.status not in ("Running",):
            # Not running is not automatically preemption -- confirm before
            # acting, because a transient API blip should not trigger a rebuild.
            quiet += 1
            if quiet >= 2:
                con.warn("instance %s status=%s -- treating as preemption"
                         % (td.machine_id, inst.status if inst else "gone"))
                return "preempted"
            continue
        quiet = 0
        try:
            logs = jl.run_logs(run_id, tail=40) if run_id else ""
        except JLError:
            continue
        text = logs if isinstance(logs, str) else json.dumps(logs)
        if "stage_measure/%s: done" % stage in text or "$DONE/%s.done" % stage in text:
            return "done"
        marker = jl.exec(td.machine_id,
                         "test -f %s/receipts/done/%s.done && echo DONE || echo PENDING"
                         % (td.fs_root, stage), timeout=120)
        if "DONE" in str(marker):
            return "done"
        if "Traceback" in text or "REFUSED" in text or "stage_measure: error" in text:
            return "failed"


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
    td.adopt(int(new_id))
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

    t = p.add_argument_group("target")
    t.add_argument("--model", help="HF repo id of the artifact to measure")
    t.add_argument("--revision", help="40-hex pin (default: resolve main and show it)")

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

    s = p.add_argument_group("safety (all default-on)")
    s.add_argument("--max-runtime", default="6h")
    s.add_argument("--heartbeat-timeout", type=int, default=900)
    s.add_argument("--max-preemptions", type=int, default=3)
    s.add_argument("--max-cost", type=float)
    s.add_argument("--on-preempt", default="resume",
                   choices=("resume", "recreate", "fail"))
    s.add_argument("--i-accept-leak-risk", action="store_true")
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
            return reaper_sweep(con)
        return reaper_list(con)

    if not args.model or not args.panel:
        con.err("--model and --panel are required")
        return EXIT_REFUSED
    if args.cold_runs is None:
        # 2, not 1. The registry's measurement schema requires run_count >= 2
        # for a published row, so a single-run receipt is a number you cannot
        # submit -- and discovering that after paying for the run is the worst
        # possible time. 3 for the sealed lane, matching how K6 was measured.
        args.cold_runs = 2 if args.lane == "streaming" else 3

    register_secret(os.environ.get("JL_API_KEY"))
    jl = JL(dry=args.dry_run)
    td = Teardown(jl, con, Path(args.out or ".").resolve())
    td.keep_fs = args.keep_fs

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
    except (JLError, HFError, Refusal) as exc:
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
