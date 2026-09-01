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

import hashlib
import math
import os
import re
import shlex
import selectors
import stat
import subprocess
import time
import tempfile
from typing import Any, Dict, List, Optional

from .jlapi import JLError, redact

_EXEC_STREAM_MAX_BYTES = 1024 * 1024
_SCP_STREAM_MAX_BYTES = 128 * 1024
_HOST_KEY_STREAM_MAX_BYTES = 128 * 1024
_UPLOAD_MAX_BYTES = 512 * 1024 * 1024


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _bounded_process(
        argv: List[str], *, timeout: float, stdout_max_bytes: int,
        stderr_max_bytes: int, label: str,
        input_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Run one process with fixed time and captured-output byte ceilings."""
    if (isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0):
        raise JLError("%s timeout must be finite and positive" % label)
    for name, value in (
            ("stdout", stdout_max_bytes), ("stderr", stderr_max_bytes)):
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            raise JLError(
                "%s %s limit must be a nonnegative integer" % (label, name))
    if input_bytes is not None and not isinstance(input_bytes, bytes):
        raise JLError("%s input must be bytes" % label)

    process = None
    selector = None
    streams: Dict[int, tuple] = {}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {
        "stdout": stdout_max_bytes,
        "stderr": stderr_max_bytes,
    }
    deadline = time.monotonic() + float(timeout)
    try:
        process = subprocess.Popen(
            argv,
            stdin=(subprocess.PIPE if input_bytes is not None
                   else subprocess.DEVNULL),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            start_new_session=True)
        if process.stdout is None or process.stderr is None:
            raise JLError("%s did not expose bounded output streams" % label)
        if input_bytes is not None:
            if process.stdin is None:
                raise JLError("%s did not expose its bounded input" % label)
            pending = memoryview(input_bytes)
            while pending:
                written = os.write(process.stdin.fileno(), pending)
                if written <= 0:
                    raise OSError("short write to %s input" % label)
                pending = pending[written:]
            process.stdin.close()

        selector = selectors.DefaultSelector()
        for name, stream in (
                ("stdout", process.stdout), ("stderr", process.stderr)):
            selector.register(stream, selectors.EVENT_READ, name)
            streams[stream.fileno()] = (name, stream)
        while streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise JLError("%s timed out after %ss" % (label, timeout))
            try:
                events = selector.select(remaining)
            except InterruptedError:
                continue
            if not events:
                raise JLError("%s timed out after %ss" % (label, timeout))
            for key, _mask in events:
                name = key.data
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    selector.unregister(stream)
                    streams.pop(stream.fileno(), None)
                    continue
                if len(buffers[name]) + len(chunk) > limits[name]:
                    raise JLError(
                        "%s %s exceeded %d bytes"
                        % (label, name, limits[name]))
                buffers[name].extend(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise JLError("%s timed out after %ss" % (label, timeout))
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise JLError("%s timed out after %ss" % (label, timeout))
        return {
            "returncode": returncode,
            "stdout": bytes(buffers["stdout"]).decode("utf-8", "replace"),
            "stderr": bytes(buffers["stderr"]).decode("utf-8", "replace"),
        }
    except BaseException as exc:
        if process is not None:
            _stop_process(process)
        if isinstance(exc, JLError):
            raise
        if isinstance(exc, OSError):
            raise JLError("%s failed: %s" % (label, exc))
        raise
    finally:
        if selector is not None:
            try:
                selector.close()
            except OSError:
                pass
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _snapshot_upload(path: str, snapshot_dir: str, max_bytes: int) -> tuple:
    """Copy one bounded link-free tree, then return its stable snapshot."""
    if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            or max_bytes < 0):
        raise JLError("upload maximum must be a nonnegative integer")
    source = os.path.abspath(os.fspath(path))
    basename = os.path.basename(os.path.normpath(source))
    if basename in ("", ".", ".."):
        raise JLError("upload source must have a stable basename")
    destination = os.path.join(snapshot_dir, basename)
    read_flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                  | getattr(os, "O_NOFOLLOW", 0))
    entries = 0
    total = 0

    def unchanged(before: os.stat_result, after: os.stat_result) -> bool:
        return (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) == (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )

    def copy_regular(source_fd: int, selected: str) -> None:
        nonlocal total
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise JLError("upload source contains a non-regular file")
        if total + int(before.st_size) > max_bytes:
            raise JLError("upload source exceeds %d bytes" % max_bytes)
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        destination_fd = os.open(selected, flags, 0o600)
        copied = 0
        try:
            os.fchmod(destination_fd, 0o600)
            while True:
                chunk = os.read(source_fd, 65536)
                if not chunk:
                    break
                copied += len(chunk)
                if total + copied > max_bytes:
                    raise JLError(
                        "upload source exceeds %d bytes" % max_bytes)
                pending = memoryview(chunk)
                while pending:
                    written = os.write(destination_fd, pending)
                    if written <= 0:
                        raise OSError("short write to upload snapshot")
                    pending = pending[written:]
            after = os.fstat(source_fd)
            if copied != int(before.st_size) or not unchanged(before, after):
                raise JLError("upload source changed while it was snapshotted")
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        total += copied

    def copy_directory(source_fd: int, selected: str) -> None:
        nonlocal entries
        before = os.fstat(source_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise JLError("upload source contains a non-directory entry")
        os.mkdir(selected, 0o700)
        try:
            with os.scandir(source_fd) as children:
                names = sorted(entry.name for entry in children)
        except OSError as exc:
            raise JLError("upload directory cannot be read: %s" % exc)
        for name in names:
            entries += 1
            if entries > 10000:
                raise JLError(
                    "upload contains more than 10000 filesystem entries")
            try:
                info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            except OSError as exc:
                raise JLError("upload entry is unavailable: %s" % exc)
            child_destination = os.path.join(selected, name)
            if stat.S_ISLNK(info.st_mode):
                raise JLError("upload source contains a symbolic link")
            if stat.S_ISREG(info.st_mode):
                try:
                    child_fd = os.open(name, read_flags, dir_fd=source_fd)
                except OSError as exc:
                    raise JLError("upload file cannot be opened: %s" % exc)
                try:
                    if not unchanged(info, os.fstat(child_fd)):
                        raise JLError(
                            "upload entry changed before it was snapshotted")
                    copy_regular(child_fd, child_destination)
                finally:
                    os.close(child_fd)
                continue
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(
                        name, read_flags | getattr(os, "O_DIRECTORY", 0),
                        dir_fd=source_fd)
                except OSError as exc:
                    raise JLError("upload directory cannot be opened: %s" % exc)
                try:
                    if not unchanged(info, os.fstat(child_fd)):
                        raise JLError(
                            "upload entry changed before it was snapshotted")
                    copy_directory(child_fd, child_destination)
                finally:
                    os.close(child_fd)
                continue
            raise JLError("upload source contains a non-regular file")
        after = os.fstat(source_fd)
        if not unchanged(before, after):
            raise JLError("upload directory changed while it was snapshotted")

    try:
        root_info = os.lstat(source)
    except OSError as exc:
        raise JLError("upload source is unavailable: %s" % exc)
    if stat.S_ISLNK(root_info.st_mode):
        raise JLError("upload source contains a symbolic link")
    if stat.S_ISREG(root_info.st_mode):
        try:
            root_fd = os.open(source, read_flags)
        except OSError as exc:
            raise JLError("upload file cannot be opened: %s" % exc)
        try:
            if not unchanged(root_info, os.fstat(root_fd)):
                raise JLError(
                    "upload root changed before it was snapshotted")
            copy_regular(root_fd, destination)
        finally:
            os.close(root_fd)
        return destination, total, False
    if not stat.S_ISDIR(root_info.st_mode):
        raise JLError("upload source contains a non-regular file")
    try:
        root_fd = os.open(
            source, read_flags | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise JLError("upload directory cannot be opened: %s" % exc)
    try:
        if not unchanged(root_info, os.fstat(root_fd)):
            raise JLError(
                "upload root changed before it was snapshotted")
        copy_directory(root_fd, destination)
    finally:
        os.close(root_fd)
    return destination, total, True


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

    # -- host keys ---------------------------------------------------------
    # Ephemeral hosts do not have a reusable global key.  The first supported
    # paid path therefore obtains the ED25519 fingerprint through an
    # authenticated provider-console side channel, compares it to an untrusted
    # keyscan, and only then writes the per-attempt known_hosts file.
    _known_hosts: Optional[str] = None

    def set_known_hosts(self, path) -> None:
        """Select the owner-controlled per-attempt host-key evidence path."""
        self._known_hosts = str(path)

    def _known_hosts_file(self) -> str:
        if not self._known_hosts:
            raise JLError(
                "SSH host key has not been authenticated for this attempt")
        try:
            info = os.lstat(self._known_hosts)
        except OSError as exc:
            raise JLError(
                "authenticated known_hosts file is unavailable: %s" % exc)
        if (not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or (hasattr(os, "getuid") and info.st_uid != os.getuid())
                or info.st_size <= 0):
            raise JLError(
                "authenticated known_hosts file is not an owner mode-0600 "
                "regular file")
        return self._known_hosts

    def scan_host_key(self, machine_id: Any) -> Dict[str, Any]:
        """Read one ED25519 key without trusting it or opening an SSH session."""
        if self.dry:
            raise JLError("dry-run cannot scan a live SSH host key")
        host, port = self._endpoint(machine_id)
        try:
            scanned = _bounded_process(
                ["ssh-keyscan", "-T", "30", "-p", str(port),
                 "-t", "ed25519", host],
                timeout=45,
                stdout_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                stderr_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                label="ssh-keyscan")
        except JLError as exc:
            raise JLError("ssh-keyscan failed: %s" % exc)
        rows = []
        for line in scanned["stdout"].splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 3 or fields[1] != "ssh-ed25519":
                raise JLError("ssh-keyscan returned a malformed host key")
            rows.append((fields[1], fields[2]))
        unique = sorted(set(rows))
        if scanned["returncode"] != 0 or len(unique) != 1:
            raise JLError(
                "ssh-keyscan did not return exactly one ED25519 host key")
        key_type, key_body = unique[0]
        public_key = "%s %s\n" % (key_type, key_body)
        try:
            fingerprint_result = _bounded_process(
                ["ssh-keygen", "-E", "sha256", "-lf", "-"],
                input_bytes=public_key.encode("utf-8"), timeout=30,
                stdout_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                stderr_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                label="ssh-keygen fingerprint")
        except JLError as exc:
            raise JLError("ssh-keygen fingerprint failed: %s" % exc)
        fingerprint_fields = fingerprint_result["stdout"].strip().split()
        fingerprint = (
            fingerprint_fields[1]
            if fingerprint_result["returncode"] == 0
            and len(fingerprint_fields) >= 2 else "")
        if re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint) is None:
            raise JLError("scanned ED25519 fingerprint is noncanonical")
        host_field = (
            host if int(port) == 22
            else "[%s]:%d" % (host, int(port)))
        return {
            "host": host,
            "port": int(port),
            "algorithm": key_type,
            "fingerprint": fingerprint,
            "known_hosts_entry": "%s %s %s\n" % (
                host_field, key_type, key_body),
        }

    def verify_host_key(
            self, machine_id: Any, expected_fingerprint: str) -> Dict[str, Any]:
        """Authenticate a scan against an independently obtained fingerprint."""
        expected = str(expected_fingerprint or "").strip()
        if re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", expected) is None:
            raise JLError(
                "expected SSH ED25519 fingerprint must be canonical SHA256")
        scanned = self.scan_host_key(machine_id)
        if scanned["fingerprint"] != expected:
            raise JLError(
                "RunPod console fingerprint differs from the network keyscan")
        if not self._known_hosts:
            raise JLError("per-attempt known_hosts path was not selected")
        path = os.path.abspath(self._known_hosts)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(scanned["known_hosts_entry"])
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            raise JLError(
                "per-attempt known_hosts already exists; refusing replacement")
        self._known_hosts_file()
        return {
            "algorithm": scanned["algorithm"],
            "fingerprint": scanned["fingerprint"],
            "host": scanned["host"],
            "port": scanned["port"],
            "known_hosts_sha256": hashlib.sha256(
                scanned["known_hosts_entry"].encode("utf-8")).hexdigest(),
        }

    def host_key_fingerprints(self) -> List[str]:
        """SHA256 fingerprints of every host key this run accepted."""
        path = self._known_hosts
        if not path or not os.path.isfile(path):
            return []
        try:
            result = _bounded_process(
                ["ssh-keygen", "-l", "-f", path], timeout=30,
                stdout_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                stderr_max_bytes=_HOST_KEY_STREAM_MAX_BYTES,
                label="ssh-keygen listing")
        except JLError:
            return []
        if result["returncode"] != 0:
            return []
        return [line.strip() for line in result["stdout"].splitlines()
                if line.strip()]

    # -- ssh ---------------------------------------------------------------
    def _ssh_opts(self) -> List[str]:
        return ["-o", "StrictHostKeyChecking=yes",
                "-o", "UserKnownHostsFile=%s" % self._known_hosts_file(),
                "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", "HostKeyAlgorithms=ssh-ed25519",
                "-o", "UpdateHostKeys=no",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                "-o", "IdentitiesOnly=yes",
                "-o", "IdentityAgent=none",
                "-o", "PasswordAuthentication=no",
                "-o", "KbdInteractiveAuthentication=no",
                "-o", "ForwardAgent=no",
                "-o", "ClearAllForwardings=yes",
                "-o", "RequestTTY=no",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3"]

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
        result = _bounded_process(
            argv, timeout=timeout,
            stdout_max_bytes=_EXEC_STREAM_MAX_BYTES,
            stderr_max_bytes=_EXEC_STREAM_MAX_BYTES,
            label="remote command")
        res = {"exit_code": result["returncode"],
               "stdout": result["stdout"], "stderr": result["stderr"]}
        if check and result["returncode"] != 0:
            raise JLError("remote command exited %s: %s"
                          % (result["returncode"], redact(
                              (result["stderr"] or result["stdout"])[:400])))
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
        result = _bounded_process(
            argv, timeout=timeout,
            stdout_max_bytes=_SCP_STREAM_MAX_BYTES,
            stderr_max_bytes=_SCP_STREAM_MAX_BYTES,
            label="scp")
        if result["returncode"] != 0:
            raise JLError("scp failed: %s" % redact(result["stderr"][:300]))
        return {"ok": True}

    def upload(self, machine_id: Any, local: str, remote: str, *,
               max_bytes: int = _UPLOAD_MAX_BYTES) -> Any:
        if self.dry:
            return {"dry_run": True}
        with tempfile.TemporaryDirectory(
                prefix=".fidelity-upload-") as snapshot_dir:
            snapshot, uploaded_bytes, recursive = _snapshot_upload(
                local, snapshot_dir, max_bytes)
            host, port = self._endpoint(machine_id)
            result = self._scp(
                machine_id, snapshot,
                "%s@%s:%s" % (self.ssh_user, host, remote),
                recursive=recursive, timeout=1800)
        result["bytes"] = uploaded_bytes
        return result

    def download(self, machine_id: Any, remote: str, local: str,
                 *, recursive: bool = True, timeout: float = 900) -> Any:
        if self.dry:
            return {"dry_run": True}
        host, port = self._endpoint(machine_id)
        return self._scp(machine_id,
                         "%s@%s:%s" % (self.ssh_user, host, remote), local,
                         recursive=recursive, timeout=timeout)

    def download_bounded(
            self, machine_id: Any, remote: str, local: str, *,
            expected_bytes: int, max_bytes: int,
            timeout: float = 900) -> Any:
        """Download one exact-size regular file without buffering SSH output."""
        if self.dry:
            return {"dry_run": True}
        if (isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0):
            raise JLError("expected download size must be a nonnegative integer")
        if (isinstance(max_bytes, bool)
                or not isinstance(max_bytes, int)
                or max_bytes < 0):
            raise JLError("maximum download size must be a nonnegative integer")
        if expected_bytes > max_bytes:
            raise JLError("expected download size exceeds its maximum")
        if (isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or timeout <= 0):
            raise JLError("download timeout must be finite and positive")

        host, port = self._endpoint(machine_id)
        argv = (["ssh", "-i", self.ssh_key, "-p", str(port)]
                + self._ssh_opts()
                + ["%s@%s" % (self.ssh_user, host),
                   "cat -- " + shlex.quote(os.fspath(remote))])
        destination = os.fspath(local)
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0))
        fd = -1
        created = False
        process = None
        selector = None
        deadline = time.monotonic() + float(timeout)
        received = 0
        try:
            fd = os.open(destination, flags, 0o600)
            created = True
            os.fchmod(fd, 0o600)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
                raise JLError(
                    "download destination is not an owner mode-0600 "
                    "regular file")

            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0)
            if process.stdout is None:
                raise JLError("SSH download did not expose a byte stream")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    raise JLError(
                        "SSH download timed out after %ss" % timeout)
                if not selector.select(remaining_time):
                    raise JLError(
                        "SSH download timed out after %ss" % timeout)
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    selector.unregister(process.stdout)
                    break
                new_total = received + len(chunk)
                if new_total > expected_bytes or new_total > max_bytes:
                    raise JLError(
                        "SSH download exceeded its declared byte count")
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short write to download destination")
                    view = view[written:]
                received = new_total

            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise JLError(
                    "SSH download timed out after %ss" % timeout)
            try:
                returncode = process.wait(timeout=remaining_time)
            except subprocess.TimeoutExpired:
                raise JLError(
                    "SSH download timed out after %ss" % timeout)
            if returncode != 0:
                raise JLError("SSH download exited %s" % returncode)
            if received != expected_bytes:
                raise JLError(
                    "SSH download returned %d bytes; expected %d"
                    % (received, expected_bytes))
            os.fsync(fd)
            os.close(fd)
            fd = -1
            selector.close()
            selector = None
            process.stdout.close()
            return {"ok": True, "bytes": received}
        except BaseException as exc:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            cleanup_error = None
            if created:
                try:
                    os.unlink(destination)
                except FileNotFoundError:
                    pass
                except OSError as unlink_exc:
                    cleanup_error = unlink_exc
            if cleanup_error is not None:
                raise JLError(
                    "SSH download failed and its partial file could not be "
                    "removed: %s" % cleanup_error) from exc
            if isinstance(exc, JLError):
                raise
            if isinstance(exc, OSError):
                raise JLError("SSH download failed: %s" % exc)
            raise
        finally:
            if selector is not None:
                try:
                    selector.close()
                except OSError:
                    pass
            if process is not None and process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass

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
        # Tri-state (P1-14): a probe that cannot run is UNKNOWN, never a
        # verdict.  GONE below is evidence-based dead (the box answered and
        # found neither an exit_code nor a live pid); an ssh failure or an
        # empty answer must not wear that verdict, because the caller treats
        # "failed" as permission to act on a job that may be alive.
        try:
            out = self.exec_stdout(
                machine_id,
                "if [ -f {d}/exit_code ]; then echo DONE $(cat {d}/exit_code); "
                "elif [ -f {d}/pid ] && kill -0 $(cat {d}/pid) 2>/dev/null; "
                "then echo RUNNING; else echo GONE; fi".format(d=d),
                timeout=120).strip().split()
        except JLError as exc:
            return {"state": "unknown", "run_id": run_id,
                    "note": "liveness probe failed (%s); not evidence of "
                            "death" % redact(str(exc))[:200]}
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
