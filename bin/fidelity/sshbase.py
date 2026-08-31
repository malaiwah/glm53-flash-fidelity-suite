#!/usr/bin/env python3
"""The SSH half of a cloud backend, written once.

JarvisLabs ships a CLI that does exec/upload/download for us. Every other
provider hands out an SSH endpoint and expects the client to do the rest, so
RunPod, Vast.ai and Lambda would otherwise carry three copies of the same
transport -- and three copies of the two non-obvious bugs in it, both found the
hard way on RunPod:

* **A detached job must record its own exit code.** The obvious spelling,
  ``nohup cmd & ( wait $!; echo $? > exit_code )``, never writes the file:
  ``wait`` only knows children of the shell that spawned them, and the subshell
  is not that shell. `run_status` then saw no exit code and reported a healthy
  job as FAILED.
* **Liveness must come from a pid the WRAPPER wrote about itself.** ``echo $!``
  captures the backgrounded shell, which forks and exits almost immediately, so
  the first poll after launch -- and the controller polls immediately -- called
  a running stage dead. ``pgrep -f`` is not the answer either: this probe names
  the run directory, which is built from the plain run id, so the id is in the
  probe's own command line no matter how the pattern is written, and the
  bracket-class trick that works in ``measure_cloud._stage_is_alive`` cannot
  work here. Verified on Linux with procps-ng 4.0.4: against a dead target,
  both the plain and the bracketed pattern answer RUNNING. The wrapper writes
  ``$$`` and ``kill -0`` reads it.



A subclass supplies `_endpoint()` (host, port), `ssh_user` and `ssh_key`.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from typing import Any, Dict, List

from .jlapi import JLError, redact


class SSHTransport:
    """exec / upload / download / detached jobs over plain ssh + scp."""

    ssh_user = "root"
    ssh_key = ""
    dry = False
    RUNS = "/workspace/.fidruns"

    # -- subclass contract -------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        raise NotImplementedError

    # -- readiness ---------------------------------------------------------
    @staticmethod
    def _tcp_ready(host: str, port: int, *, timeout: float = 5) -> bool:
        import socket
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    def _await_ssh(self, host: str, port: int, *, wait: float = 900) -> None:
        """A provider says "running" well before sshd accepts.

        Vast reports `running` as soon as the contract exists, while the
        container is still pulling its image -- measured at 99 s on a smoke
        instance. Returning the endpoint at that moment made the very first
        remote command die with `Connection refused`, which the controller
        correctly treated as a failed run and tore the box down. The endpoint
        is not ready until something answers on it.
        """
        deadline = time.time() + wait
        while time.time() < deadline:
            if self._tcp_ready(host, port):
                return
            time.sleep(10)
        raise JLError("ssh on %s:%s never accepted a connection within %ds"
                      % (host, port, int(wait)))

    # -- ssh ---------------------------------------------------------------
    def _ssh_opts(self) -> List[str]:
        return ["-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-o", "ServerAliveInterval=30"]

    def exec(self, machine_id: Any, command: str, *,
             timeout: float = 600, check: bool = True) -> Any:
        """Returns {exit_code, stdout, stderr} -- the shape the controller reads.

        The controller checks `exit_code` INSIDE the payload rather than
        trusting the transport's own exit status, because on the CLI-driven
        backend those are different things. Keeping the same shape here means
        no caller needs a branch.
        """
        if self.dry:
            return {"exit_code": 0, "stdout": "", "stderr": "", "dry_run": True}
        host, port = self._endpoint(machine_id)
        argv = (["ssh", "-i", self.ssh_key, "-p", str(port)] + self._ssh_opts()
                + ["%s@%s" % (self.ssh_user, host), "sh -lc " + shlex.quote(command)])
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise JLError("remote command timed out after %ss" % timeout)
        res = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        if check and p.returncode != 0:
            raise JLError("remote command exited %s: %s"
                          % (p.returncode, redact((p.stderr or p.stdout)[:400])))
        return res

    def exec_stdout(self, machine_id: Any, command: str, *,
                    timeout: float = 600, check: bool = True) -> str:
        return str(self.exec(machine_id, command, timeout=timeout,
                             check=check).get("stdout") or "")

    def _scp(self, machine_id: Any, src: str, dst: str, *,
             recursive: bool, timeout: float) -> Any:
        host, port = self._endpoint(machine_id)
        argv = ["scp", "-i", self.ssh_key, "-P", str(port)] + self._ssh_opts()
        if recursive:
            argv.append("-r")
        argv += [src, dst]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise JLError("scp failed: %s" % redact(p.stderr[:300]))
        return {"ok": True}

    def upload(self, machine_id: Any, local: str, remote: str) -> Any:
        if self.dry:
            return {"dry_run": True}
        host, port = self._endpoint(machine_id)
        return self._scp(machine_id, local,
                         "%s@%s:%s" % (self.ssh_user, host, remote),
                         recursive=True, timeout=1800)

    def download(self, machine_id: Any, remote: str, local: str,
                 *, recursive: bool = True, timeout: float = 900) -> Any:
        if self.dry:
            return {"dry_run": True}
        host, port = self._endpoint(machine_id)
        return self._scp(machine_id,
                         "%s@%s:%s" % (self.ssh_user, host, remote), local,
                         recursive=recursive, timeout=timeout)

    # -- detached jobs -----------------------------------------------------
    def run_job(self, machine_id: Any, command: str) -> Any:
        run_id = "r_%d" % int(time.time() * 1000)
        d = "%s/%s" % (self.RUNS, run_id)
        # The WRAPPER writes its own pid ($$), not the launcher's $!.
        # `$!` is the backgrounded shell, which forks and exits almost at once,
        # so a pid recorded that way is dead within a second of a healthy start.
        launcher = (
            "mkdir -p {d} && printf '%s' {cmd} > {d}/run.sh && "
            "setsid sh -c 'echo $$ > {d}/pid; sh {d}/run.sh > {d}/output.log 2>&1; "
            "echo $? > {d}/exit_code' </dev/null >/dev/null 2>&1 & "
            "sleep 1; echo launched {rid}"
        ).format(d=d, cmd=shlex.quote(command), rid=run_id)
        self.exec(machine_id, launcher, timeout=180)
        return {"run_id": run_id, "machine_id": str(machine_id)}

    def run_status(self, run_id: str, machine_id: Any = None) -> Dict[str, Any]:
        if machine_id is None:
            raise JLError("run_status needs machine_id on this backend")
        d = "%s/%s" % (self.RUNS, run_id)
        # Liveness comes from the WRAPPER'S OWN pid, not from pgrep.
        #
        # pgrep was tried and does not work here, which is worth recording
        # because the obvious fix does not work either. `pgrep -f` matches full
        # command lines, and this probe's own shell carries the pattern in ITS
        # command line. measure_cloud._stage_is_alive solves that with a
        # bracket class -- `[s]tage_measure.sh` matches the real process and not
        # the probe, whose cmdline holds the literal brackets (JOURNAL 36/44).
        #
        # That trick CANNOT work here, because this command also names the run
        # DIRECTORY, which is built from the plain run id -- so the unbracketed
        # id is in the probe's own cmdline no matter how the pattern is
        # written. Confirmed on Linux (procps-ng 4.0.4): with the target dead,
        # `pgrep -f r_1788...` AND `pgrep -f '[r]_1788...'` both answer
        # RUNNING. On macOS BSD pgrep neither does, which is why this needed a
        # Linux box to see at all.
        #
        # `kill -0` on a pid the wrapper wrote about itself has no such
        # ambiguity. It matters only when a job dies WITHOUT writing exit_code
        # -- OOM, preemption, a reaped container -- which is exactly the branch
        # that decides whether the controller fails in one poll or waits out
        # --max-runtime on a billing instance.
        out = self.exec_stdout(
            machine_id,
            "if [ -f {d}/exit_code ]; then echo DONE $(cat {d}/exit_code); "
            "elif [ -f {d}/pid ] && kill -0 $(cat {d}/pid) 2>/dev/null; "
            "then echo RUNNING; else echo GONE; fi".format(d=d),
            timeout=120).strip().split()
        if not out:
            return {"state": "unknown", "run_id": run_id}
        if out[0] == "DONE":
            code = int(out[1]) if len(out) > 1 and out[1].lstrip("-").isdigit() else 1
            return {"state": "succeeded" if code == 0 else "failed",
                    "exit_code": code, "run_id": run_id}
        if out[0] == "RUNNING":
            return {"state": "running", "run_id": run_id}
        return {"state": "failed", "run_id": run_id,
                "note": "no exit_code file and no process matching the run id"}

    def run_logs(self, run_id: str, *, tail: int = 50,
                 machine_id: Any = None) -> Any:
        if machine_id is None:
            raise JLError("run_logs needs machine_id on this backend")
        return self.exec_stdout(
            machine_id, "tail -n %d %s/%s/output.log 2>/dev/null || true"
            % (int(tail), self.RUNS, run_id), timeout=120)
