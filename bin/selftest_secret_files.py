#!/usr/bin/env python3
"""Secret files are 0600 from the first instant, at both ends, and cleaned up.

WHY THIS EXISTS
---------------
Peer review 2026-08-31, security chapter (High): the controller wrote the
local token file and only THEN chmod'd it to 0600 -- on a permissive umask
that is a window another local user can read, and the project's own
concurrency test once captured a full token through it.  The remote path
created `.secrets`, uploaded the token, and tightened modes only afterwards.
The container entrypoint opened the destination with a plain `open("w")`
(follows a planted symlink on the persistent bind mount) and never removed
the token when the run ended.

  S1  write_secret_file: 0600 file inside a 0700 directory, atomically.
  S2  a pre-existing loose-mode file is replaced, not inherited.
  S3  a planted symlink is removed, never written through.
  S4  shred_secret_file removes the file; a missing file is a no-op.
  S5  remote transport ORDER: refuse a pre-existing secret directory, create
      it as 0700 BEFORE upload, upload uniquely, chmod 600, atomic rename.
  S6  the token value never appears on any remote command line.
  S7  the local copy is shredded even when the upload raises.
  S8  container write_token uses the same exclusive/no-follow creation.
  S9  the container entrypoint removes the token in a finally -- a FAILED
      stage still leaves no token behind on the bind mount.

No network, no provider, no real token.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.common import shred_secret_file, write_secret_file  # noqa: E402

FAILED = []
TOKEN = "hf_selftest_not_a_real_token_11111"


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:8]:
            print("        %s" % line)


def mode(path):
    return oct(os.lstat(str(path)).st_mode & 0o777)


def load_measure_cloud():
    spec = importlib.util.spec_from_file_location(
        "measure_cloud", str(ROOT / "bin" / "measure_cloud.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubJL:
    """Records every remote operation, in order."""

    def __init__(self, fail_upload=False, events=None):
        self.ops = []
        self.fail_upload = fail_upload
        self.events = events if events is not None else []

    def exec(self, machine_id, command, **kw):
        self.ops.append(("exec", command))
        self.events.append(("exec", command))
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    def upload(self, machine_id, local, remote):
        self.ops.append(("upload", local, remote))
        self.events.append(("upload", remote))
        if self.fail_upload:
            raise RuntimeError("upload failed (stub)")
        return {"ok": True}


class StubTD:
    fs_root = "/fs"
    machine_id = 7


def main():
    old_umask = os.umask(0o000)   # the hostile umask the review names
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # S1
            dest = td / "sec" / "hf_token"
            write_secret_file(str(dest), TOKEN)
            check("S1 file is 0600 in a 0700 directory (umask 000)",
                  mode(dest) == "0o600" and mode(td / "sec") == "0o700",
                  "file=%s dir=%s" % (mode(dest), mode(td / "sec")))
            check("S1b content intact", dest.read_text() == TOKEN)

            # S2
            loose = td / "sec" / "loose"
            loose.write_text("old")
            os.chmod(str(loose), 0o644)
            write_secret_file(str(loose), TOKEN)
            check("S2 pre-existing 0644 file replaced with a fresh 0600 one",
                  mode(loose) == "0o600" and loose.read_text() == TOKEN)

            # S3
            victim = td / "victim"
            victim.write_text("victim-bytes")
            link = td / "sec" / "planted"
            link.symlink_to(victim)
            write_secret_file(str(link), TOKEN)
            check("S3 planted symlink is removed, never written through",
                  victim.read_text() == "victim-bytes"
                  and not link.is_symlink() and mode(link) == "0o600")

            # S4
            shred_secret_file(str(loose))

            # S4c: paid RunPod accepts only an explicit owned 0600 token file.
            mc = load_measure_cloud()
            download_token = td / "download-token"
            write_secret_file(str(download_token), TOKEN)
            check("S4c required download token reads the exact 0600 file",
                  mc._load_required_hf_download_token(
                      str(download_token)) == TOKEN)
            download_token.chmod(0o644)
            bad_mode_refused = False
            try:
                mc._load_required_hf_download_token(str(download_token))
            except mc.Refusal:
                bad_mode_refused = True
            missing_refused = False
            try:
                mc._load_required_hf_download_token(
                    str(td / "missing-download-token"))
            except mc.Refusal:
                missing_refused = True
            check("S4d loose or missing download token refuses before transport",
                  bad_mode_refused and missing_refused)

            check("S4 shred removes the file", not loose.exists())
            shred_secret_file(str(td / "never-existed"))

            # S5/S6: the controller's remote transport, against a stub.
            jl = StubJL()
            outdir = td / "out"
            outdir.mkdir()
            mc._transport_hf_token(
                jl, StubTD.machine_id, StubTD.fs_root, outdir, TOKEN)
            kinds = [op[0] for op in jl.ops]
            check("S5a exactly exec, upload, exec", kinds == ["exec", "upload", "exec"],
                  jl.ops)
            first = jl.ops[0][1]
            # `mkdir -p -m 700` after `test ! -e` is still an exclusive create
            # (the existence test refuses first), and the command now ALSO
            # reads the mode back (`stat -c %a` = 700) before anything is
            # uploaded; the rung asserts that whole sequence, in order.
            check("S5b the directory is exclusively created 0700 before upload",
                  "test ! -e /fs/.secrets" in first
                  and "test ! -L /fs/.secrets" in first
                  and ("mkdir -m 700 -- /fs/.secrets" in first
                       or "mkdir -p -m 700 -- /fs/.secrets" in first)
                  and 'test "$(stat -c %a -- /fs/.secrets)" = 700' in first
                  and first.index("test ! -e /fs/.secrets") < first.index("mkdir")
                  < first.index("stat -c %a -- /fs/.secrets"),
                  first)
            up_remote = jl.ops[1][2]
            check("S5c the upload lands on a unique temporary name, not the "
                  "final path",
                  up_remote.startswith("/fs/.secrets/") and
                  up_remote != "/fs/.secrets/hf_token", up_remote)
            last = jl.ops[2][1]
            check("S5d chmod 600 the temp, then rename it into place",
                  ("chmod 600 -- %s" % up_remote) in last
                  and ("mv -- %s /fs/.secrets/hf_token" % up_remote) in last
                  and last.index("chmod 600") < last.index("mv --"), last)
            check("S5e the local staging copy is gone afterwards",
                  not (outdir / ".secrets-local" / "hf_token").exists())
            check("S5f the local staging directory was 0700",
                  mode(outdir / ".secrets-local") == "0o700")
            check("S6 the token value never appears on a remote command line",
                  all(TOKEN not in " ".join(str(part) for part in op)
                      for op in jl.ops), jl.ops)

            # S7: shredded even when the upload raises.
            jl2 = StubJL(fail_upload=True)
            outdir2 = td / "out2"
            outdir2.mkdir()
            raised = False
            try:
                mc._transport_hf_token(
                    jl2, StubTD.machine_id, StubTD.fs_root, outdir2, TOKEN)
            except RuntimeError:
                raised = True
            check("S7 a failed upload still shreds the local copy (and "
                  "propagates)",
                  raised and not (outdir2 / ".secrets-local" / "hf_token").exists())

            # S7a/S7b: the RunPod target-fetch wrapper scopes credential use
            # to that one stage and cleans up on both success and failure.
            events = []
            runpod = StubJL(events=events)
            original_stage = mc._runpod_stage
            try:
                mc._runpod_stage = lambda *_a, **_kw: events.append(
                    ("stage", "fetch_target"))
                mc._transport_hf_token(
                    runpod, StubTD.machine_id, StubTD.fs_root,
                    td / "runpod-success", TOKEN)
                cleanup = mc._runpod_fetch_target_and_remove_token(
                    runpod, StubTD.machine_id, StubTD.fs_root, "/engine",
                    1.0, "image@sha256:" + "a" * 64)
                check("S7a RunPod authenticates only fetch_target, then confirms "
                      "remote cleanup",
                      cleanup.get("confirmed") is True
                      and [kind for kind, _value in events]
                          == ["exec", "upload", "exec", "stage", "exec"]
                      and "shred -u" in events[-1][1],
                      events)

                events.clear()
                def fail_stage(*_args, **_kwargs):
                    events.append(("stage", "fetch_target"))
                    raise RuntimeError("fetch failed")
                mc._runpod_stage = fail_stage
                failed = False
                try:
                    mc._transport_hf_token(
                        runpod, StubTD.machine_id, StubTD.fs_root,
                        td / "runpod-failure", TOKEN)
                    mc._runpod_fetch_target_and_remove_token(
                        runpod, StubTD.machine_id, StubTD.fs_root, "/engine",
                        1.0, "image@sha256:" + "a" * 64)
                except RuntimeError as exc:
                    failed = str(exc) == "fetch failed"
                check("S7b failed RunPod fetch still removes the remote token",
                      failed and events[-1][0] == "exec"
                      and "shred -u" in events[-1][1],
                      events)
            finally:
                mc._runpod_stage = original_stage

            # S8: the container entrypoint's writer.
            spec = importlib.util.spec_from_file_location(
                "container_entry", str(ROOT / "bin" / "container_entry.py"))
            CE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(CE)
            fs = td / "fs"
            fs.mkdir()
            victim2 = td / "victim2"
            victim2.write_text("victim-bytes")
            (fs / ".secrets").mkdir(mode=0o700)
            (fs / ".secrets" / "hf_token").symlink_to(victim2)
            src = td / "tokfile"
            src.write_text(TOKEN + "\n")
            wrote = CE.write_token(fs, str(src), lambda *_a, **_k: None)
            tokpath = fs / ".secrets" / "hf_token"
            check("S8 container write_token: symlink removed, 0600, 0700 dir, "
                  "no write-through",
                  wrote and victim2.read_text() == "victim-bytes"
                  and not tokpath.is_symlink() and mode(tokpath) == "0o600"
                  and mode(fs / ".secrets") == "0o700")

            # S9: drive the entrypoint's stage lifecycle with its science
            # validators stubbed at this unit boundary. Other selftests own the
            # strict job.v2 contract; this case owns the token-finally invariant.
            fs9 = td / "fs9"
            fs9.mkdir()
            fs9.joinpath("job.json").write_text("{}\n", encoding="utf-8")
            os.environ["HF_TOKEN"] = TOKEN
            os.environ.pop("FIDELITY_SUITE_ROOT", None)
            try:
                CE._prevalidate_stage_job = lambda *_a, **_k: {
                    "role": "quant",
                    "capture": {},
                    "target": {"surface": "native-bf16"},
                }
                CE.sync_suite = lambda *_a, **_k: 0
                CE.validate_job_document = lambda *_a, **_k: None
                CE.require_accelerator = lambda *_a, **_k: None
                CE.run_stage = lambda *_a, **_k: 1
                CE.RS.parse_sinks = lambda *_a, **_k: []
                CE.RS.build_summary = lambda *_a, **_k: {}
                CE.RS.deliver = lambda *_a, **_k: []
                rc = CE.main(["stage", "fetch_target", "--fs-root", str(fs9),
                              "--engine-root", str(td / "noengine")])
                token_left = (fs9 / ".secrets" / "hf_token").exists()
                check("S9 a failed stage leaves no token on the run root "
                      "(rc=%s)" % rc,
                      rc == CE.EXIT_FAILED and not token_left)
            finally:
                os.environ.pop("HF_TOKEN", None)
    finally:
        os.umask(old_umask)

    print()
    if FAILED:
        print("selftest_secret_files: %d FAILED" % len(FAILED))
        return 1
    print("selftest_secret_files: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
