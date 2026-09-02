#!/usr/bin/env python3
"""Authenticated ephemeral SSH host keys; first-hop TOFU is forbidden.

The measurement channel carries model evidence and control artifacts. A network
keyscan is therefore untrusted until the controller compares its ED25519
fingerprint to the fresh pod's authenticated provider-log event. The transport
must refuse every SSH/SCP call before that comparison, write one owner-only
per-attempt known_hosts file, and then use that strict ED25519 pin without
ambient host-key files or update behavior. Command and SCP diagnostics carry
fixed byte ceilings; uploads are pre-counted without following links; exact-size
downloads are streamed through local pipes. The tests use no network/provider.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity import sshbase  # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:8]:
            print("        %s" % line)


class T(sshbase.SSHTransport):
    ssh_user = "root"
    ssh_key = "/dev/null"

    def _endpoint(self, machine_id, *, wait=900):
        return ("198.51.100.7", 22)


def opts_dict(opts):
    return {opts[i + 1].split("=", 1)[0]: opts[i + 1].split("=", 1)[1]
            for i in range(0, len(opts), 2)
            if opts[i] == "-o" and "=" in opts[i + 1]}


def command_opts(argv):
    found = {}
    for index, item in enumerate(argv[:-1]):
        if item != "-o" or "=" not in argv[index + 1]:
            continue
        name, value = argv[index + 1].split("=", 1)
        found.setdefault(name, []).append(value)
    return found

def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        t = T()
        try:
            t._ssh_opts()
        except sshbase.JLError:
            unauthenticated_refused = True
        else:
            unauthenticated_refused = False
        check("K1 SSH refuses before an out-of-band authenticated key exists",
              unauthenticated_refused)

        run_kh = td / "run" / "ssh_known_hosts"
        t.set_known_hosts(run_kh)
        try:
            t._ssh_opts()
        except sshbase.JLError:
            path_only_refused = True
        else:
            path_only_refused = False
        check("K2 selecting a path does not establish trust",
              path_only_refused and not run_kh.exists())

        key_body = "A" * 68
        known_hosts_entry = (
            "198.51.100.7 ssh-ed25519 %s\n" % key_body)
        expected_fingerprint = "SHA256:" + "B" * 43
        t.scan_host_key = lambda _machine_id: {
            "host": "198.51.100.7",
            "port": 22,
            "algorithm": "ssh-ed25519",
            "fingerprint": expected_fingerprint,
            "known_hosts_entry": known_hosts_entry,
        }
        evidence = t.verify_host_key(9, expected_fingerprint)
        opts = opts_dict(t._ssh_opts())
        check("K3 authenticated scan writes one exact owner-mode-0600 key",
              run_kh.read_text() == known_hosts_entry
              and (run_kh.stat().st_mode & 0o777) == 0o600
              and evidence["known_hosts_sha256"]
                  == hashlib.sha256(
                      known_hosts_entry.encode("utf-8")).hexdigest()
              and opts.get("StrictHostKeyChecking") == "yes"
              and opts.get("UserKnownHostsFile") == str(run_kh),
              (evidence, opts))

        mismatch_path = td / "mismatch" / "ssh_known_hosts"
        mismatch = T()
        mismatch.set_known_hosts(mismatch_path)
        mismatch.scan_host_key = t.scan_host_key
        try:
            mismatch.verify_host_key(9, "SHA256:" + "C" * 43)
        except sshbase.JLError:
            mismatch_refused = True
        else:
            mismatch_refused = False
        check("K4 network key differing from provider-log identity is refused",
              mismatch_refused and not mismatch_path.exists())

        try:
            t.verify_host_key(9, expected_fingerprint)
        except sshbase.JLError:
            replacement_refused = True
        else:
            replacement_refused = False
        check("K5 an authenticated key file is immutable within the attempt",
              replacement_refused and run_kh.read_text() == known_hosts_entry)

        noninteractive = {
            "BatchMode": "yes",
            "IdentitiesOnly": "yes",
            "IdentityAgent": "none",
            "PasswordAuthentication": "no",
            "KbdInteractiveAuthentication": "no",
            "ForwardAgent": "no",
            "ClearAllForwardings": "yes",
            "RequestTTY": "no",
            "ServerAliveCountMax": "3",
        }
        isolated_trust = {
            "StrictHostKeyChecking": "yes",
            "UserKnownHostsFile": str(run_kh),
            "GlobalKnownHostsFile": "/dev/null",
            "HostKeyAlgorithms": "ssh-ed25519",
            "UpdateHostKeys": "no",
        }
        recorded = []
        recorded_limits = []

        def fake_bounded_process(argv, **kw):
            recorded.append(list(argv))
            recorded_limits.append(dict(kw))
            return {"returncode": 0, "stdout": "", "stderr": ""}

        original_bounded_process = sshbase._bounded_process
        sshbase._bounded_process = fake_bounded_process
        try:
            t.exec(9, "true")
            t._scp(9, "/tmp/a", "root@198.51.100.7:/tmp/b",
                   recursive=False, timeout=5)
        finally:
            sshbase._bounded_process = original_bounded_process
        parsed_commands = [command_opts(argv) for argv in recorded]
        check("K6 exec and scp carry exact isolated trust and key-only auth",
              len(recorded) == 2 and all(
                  all(options.get(name) == [value]
                      for name, value in isolated_trust.items())
                  and all(options.get(name) == [value]
                          for name, value in noninteractive.items())
                  for options in parsed_commands)
              and recorded_limits[0]["stdout_max_bytes"]
                  == sshbase._EXEC_STREAM_MAX_BYTES
              and recorded_limits[0]["stderr_max_bytes"]
                  == sshbase._EXEC_STREAM_MAX_BYTES
              and recorded_limits[1]["stdout_max_bytes"]
                  == sshbase._SCP_STREAM_MAX_BYTES
              and recorded_limits[1]["stderr_max_bytes"]
                  == sshbase._SCP_STREAM_MAX_BYTES,
              (recorded, recorded_limits))
        bounded_stdout = False
        bounded_stderr = False
        try:
            sshbase._bounded_process(
                [sys.executable, "-c",
                 "import os; os.write(1, b'x' * 5)"],
                timeout=1, stdout_max_bytes=4, stderr_max_bytes=4,
                label="stdout-overflow")
        except sshbase.JLError:
            bounded_stdout = True
        try:
            sshbase._bounded_process(
                [sys.executable, "-c",
                 "import os; os.write(2, b'x' * 5)"],
                timeout=1, stdout_max_bytes=4, stderr_max_bytes=4,
                label="stderr-overflow")
        except sshbase.JLError:
            bounded_stderr = True
        check("K6b command capture refuses stdout and stderr above fixed bytes",
              bounded_stdout and bounded_stderr)

        upload_source = td / "upload.bin"
        upload_source.write_bytes(b"abcde")
        upload_calls = []
        replacement_source = td / "replacement.bin"
        replacement_source.write_bytes(b"x" * 10)
        original_scp = t._scp

        def snapshot_scp(machine_id, src, dst, *, recursive, timeout):
            upload_source.unlink()
            upload_source.symlink_to(replacement_source)
            snapshot_bytes = Path(src).read_bytes()
            upload_calls.append(
                (src, dst, recursive, timeout, snapshot_bytes,
                 Path(src) != upload_source))
            return {"ok": True}

        t._scp = snapshot_scp
        try:
            try:
                t.upload(9, str(upload_source), "/workspace/upload.bin",
                         max_bytes=4)
            except sshbase.JLError:
                upload_oversize_refused = True
            else:
                upload_oversize_refused = False
            upload_result = t.upload(
                9, str(upload_source), "/workspace/upload.bin", max_bytes=5)
            upload_link = td / "upload-link"
            upload_link.symlink_to(upload_source)
            try:
                t.upload(9, str(upload_link), "/workspace/upload-link",
                         max_bytes=5)
            except sshbase.JLError:
                upload_link_refused = True
            else:
                upload_link_refused = False
        finally:
            t._scp = original_scp
        check("K6c upload uses stable counted snapshot bytes for SCP",
              upload_oversize_refused and len(upload_calls) == 1
              and upload_result == {"ok": True, "bytes": 5}
              and upload_calls[0][2] is False
              and upload_calls[0][4] == b"abcde"
              and upload_calls[0][5] is True,
              (upload_calls, upload_result))
        check("K6d uploads never follow a symbolic link",
              upload_link_refused and len(upload_calls) == 1)
        race_source = td / "race-upload.bin"
        race_source.write_bytes(b"first")
        race_replacement = td / "race-replacement.bin"
        race_replacement.write_bytes(b"other")
        real_open = sshbase.os.open
        swapped = [False]

        def swap_before_root_open(path, *args, **kwargs):
            if (not swapped[0]
                    and os.fspath(path) == str(race_source)):
                swapped[0] = True
                race_replacement.replace(race_source)
            return real_open(path, *args, **kwargs)

        root_replacement_refused = False
        sshbase.os.open = swap_before_root_open
        try:
            with tempfile.TemporaryDirectory() as snapshot_dir:
                try:
                    sshbase._snapshot_upload(
                        str(race_source), snapshot_dir, 5)
                except sshbase.JLError:
                    root_replacement_refused = True
        finally:
            sshbase.os.open = real_open
        check("K6e upload refuses root replacement between lstat and open",
              swapped[0] and root_replacement_refused)
        bounded_argv = []
        real_popen = sshbase.subprocess.Popen

        def bounded(script, destination, expected, maximum, timeout=1,
                    remote="/tmp/a file; echo untrusted"):
            def fake_popen(argv, **kw):
                bounded_argv.append(list(argv))
                return real_popen([sys.executable, "-c", script], **kw)

            sshbase.subprocess.Popen = fake_popen
            try:
                return t.download_bounded(
                    9, remote, destination, expected_bytes=expected,
                    max_bytes=maximum, timeout=timeout)
            finally:
                sshbase.subprocess.Popen = real_popen

        exact_path = td / "exact.bin"
        fsync_calls = []
        real_fsync = sshbase.os.fsync

        def recording_fsync(fd):
            fsync_calls.append(fd)
            return real_fsync(fd)

        sshbase.os.fsync = recording_fsync
        try:
            exact_result = bounded(
                "import os; os.write(1, b'abcd')",
                exact_path, 4, 8)
        finally:
            sshbase.os.fsync = real_fsync
        expected_remote_command = (
            "cat -- " + sshbase.shlex.quote(
                "/tmp/a file; echo untrusted"))
        check("K7 bounded download streams exact bytes to a synced mode-0600 file",
              exact_result == {"ok": True, "bytes": 4}
              and exact_path.read_bytes() == b"abcd"
              and (exact_path.stat().st_mode & 0o777) == 0o600
              and len(fsync_calls) == 1,
              (exact_result, fsync_calls))
        bounded_options = command_opts(bounded_argv[-1])
        check("K8 bounded download quotes its path and uses isolated SSH trust",
              bounded_argv[-1][-1] == expected_remote_command
              and all(bounded_options.get(name) == [value]
                      for name, value in isolated_trust.items()),
              bounded_argv[-1])

        failure_cases = [
            ("K9 short bounded stream is refused and removed",
             "import os; os.write(1, b'abc')",
             td / "short.bin", 4, 8, 1),
            ("K10 oversized bounded stream is refused and removed",
             "import os; os.write(1, b'abcde')",
             td / "oversize.bin", 4, 8, 1),
            ("K11 timed-out partial bounded stream is killed and removed",
             "import os,time; os.write(1, b'ab'); time.sleep(2)",
             td / "timeout.bin", 4, 8, 0.05),
            ("K12 nonzero bounded SSH exit is refused and removed",
             "import os,sys; os.write(1, b'abcd'); sys.exit(7)",
             td / "nonzero.bin", 4, 8, 1),
        ]
        for label, script, destination, expected, maximum, limit in failure_cases:
            try:
                bounded(script, destination, expected, maximum, limit)
            except sshbase.JLError:
                refused = True
            else:
                refused = False
            check(label, refused and not destination.exists())

        dry = T()
        dry.dry = True
        dry_path = td / "dry.bin"
        dry_result = dry.download_bounded(
            9, "/remote", dry_path, expected_bytes=4, max_bytes=8)
        check("K13 dry bounded download performs no local or network mutation",
              dry_result == {"dry_run": True}
              and not dry_path.exists())

        empty = T()
        check("K14 no authenticated connection means no fingerprint record",
              empty.host_key_fingerprints() == [])

        if shutil.which("ssh-keygen"):
            keyfile = td / "hostkey"
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                            "-f", str(keyfile)], check=True)
            pub = keyfile.with_suffix(".pub").read_text().strip()
            run_kh.write_text("198.51.100.7 %s\n" % pub)
            prints = t.host_key_fingerprints()
            check("K15 authenticated host key yields a SHA256 fingerprint",
                  prints and any("SHA256:" in line for line in prints), prints)
        else:
            print("  SKIP  K15 (ssh-keygen not on PATH)")

    print()
    if FAILED:
        print("selftest_sshbase: %d FAILED" % len(FAILED))
        return 1
    print("selftest_sshbase: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
