#!/usr/bin/env python3
"""SSH host authentication: per-run trust-on-first-use, never disabled.

WHY THIS EXISTS
---------------
Peer review 2026-08-31 (security chapter, High): the SSH transport used
`StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null`, which removes
server authentication entirely from the channel that carries the HF token
and every measurement artifact.  The fix is per-run TOFU: `accept-new`
records the first-seen key into a per-run known_hosts file, every later
connection in the run refuses a changed key, and the fingerprint is
recorded so the receipt can carry it.

  K1  the option set says accept-new, never `no`, and points at a real
      per-run file, never /dev/null.
  K2  the same transport instance keeps ONE known_hosts file across calls
      (that persistence is what turns TOFU into a per-run pin).
  K3  set_known_hosts pins the file under the run dir, creating parents.
  K4  the real ssh argv (exec and scp paths) carries those options.
  K5  host_key_fingerprints reads SHA256 fingerprints out of the recorded
      file (skipped when ssh-keygen is unavailable).
  K6  an empty run (no connection yet) reports no fingerprints and does not
      crash.

No network, no provider: subprocess.run is stubbed for K4.
"""
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


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        t = T()
        opts = opts_dict(t._ssh_opts())
        check("K1 StrictHostKeyChecking=accept-new (never 'no')",
              opts.get("StrictHostKeyChecking") == "accept-new", opts)
        check("K1b UserKnownHostsFile is a real file path, not /dev/null",
              opts.get("UserKnownHostsFile") not in (None, "/dev/null")
              and os.path.isabs(opts["UserKnownHostsFile"]), opts)

        opts2 = opts_dict(t._ssh_opts())
        check("K2 one known_hosts file per transport instance, stable "
              "across calls",
              opts2["UserKnownHostsFile"] == opts["UserKnownHostsFile"])
        try:
            os.unlink(opts["UserKnownHostsFile"])
        except OSError:
            pass

        t2 = T()
        run_kh = td / "run" / "ssh_known_hosts"
        t2.set_known_hosts(run_kh)
        opts3 = opts_dict(t2._ssh_opts())
        check("K3 set_known_hosts pins the file under the run dir",
              opts3["UserKnownHostsFile"] == str(run_kh)
              and (td / "run").is_dir(), opts3)

        # K4: the argv actually handed to ssh/scp.
        recorded = []

        def fake_run(argv, **kw):
            recorded.append(list(argv))

            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        orig = sshbase.subprocess.run
        sshbase.subprocess.run = fake_run
        try:
            t2.exec(9, "true")
            t2._scp(9, "/tmp/a", "root@198.51.100.7:/tmp/b",
                    recursive=False, timeout=5)
        finally:
            sshbase.subprocess.run = orig
        joined = ["\x00".join(argv) for argv in recorded]
        check("K4 exec and scp argv carry accept-new + the per-run file",
              len(recorded) == 2 and all(
                  "StrictHostKeyChecking=accept-new" in j
                  and ("UserKnownHostsFile=%s" % run_kh) in j
                  and "StrictHostKeyChecking=no" not in j
                  and "UserKnownHostsFile=/dev/null" not in j
                  for j in joined),
              recorded)

        # K6 before K5: nothing recorded yet.
        t3 = T()
        check("K6 no connection yet -> no fingerprints, no crash",
              t3.host_key_fingerprints() == [])

        # K5: a real key in the file yields a SHA256 fingerprint.
        if shutil.which("ssh-keygen"):
            keyfile = td / "hostkey"
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                            "-f", str(keyfile)], check=True)
            pub = keyfile.with_suffix(".pub").read_text().strip()
            run_kh.parent.mkdir(parents=True, exist_ok=True)
            run_kh.write_text("[198.51.100.7]:22 %s\n" % pub)
            prints = t2.host_key_fingerprints()
            check("K5 recorded host key yields a SHA256 fingerprint",
                  prints and any("SHA256:" in line for line in prints), prints)
        else:
            print("  SKIP  K5 (ssh-keygen not on PATH)")

    print()
    if FAILED:
        print("selftest_sshbase: %d FAILED" % len(FAILED))
        return 1
    print("selftest_sshbase: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
