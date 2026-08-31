#!/usr/bin/env python3
"""ROOT-1: a sealed root gets published, and teardown refuses to eat one.

WHY THIS EXISTS
---------------
The controller destroyed a sealed, twice-validated MiniMax-M3 root dataset at
teardown -- $6.59 of GPU time and the only copy of the evidence -- because
nothing published or preserved a root (REVIEW-DEFERRED ROOT-1, fee collected
2026-08-31).

  RP1  stage_sequence: --publish-root-to appends `publish_root` after
       `verify`, on both the plain and the race root paths, never for quant.
  RP2  teardown REFUSES to destroy a box holding a sealed+VERIFIED but
       unpublished root: instance kept, lease kept, banner names the way out.
  RP3  --allow-unpublished-root is the explicit override: destroy proceeds.
  RP4  a PUBLISHED root tears down normally.
  RP5  a quant run is untouched by the guard.
  RP6  the container entrypoint plumbs --publish-root-to into job.json's
       capture block (C3c parity) and its stage list gains publish_root.
  RP7  runpodapi.create composes a container-native launch: our image, our
       dockerArgs command, env-carried configuration -- and the token never
       appears in dockerArgs.
  RP8  the controller refuses --publish-root-to on a quant run and refuses a
       malformed repo id, both before any spend.
  RP9  the publish receipt pins the uploaded revision, is sealed, and is
       written only after the fetched-back copy verifies.
  RP10 race composition: a PREVIEW aimed at the FINAL repo name is refused
       at plan time (identity separation, docs/RACE-MODE.md).
  RP11 acceptance: a container dry-run of `capture --race --preview-of ...
       --publish-root-to ...` composes without refusal and its stage list is
       the race sequence ending in publish_root.

Stub provider, no network, $0.00.  The stage-level behavior of publish_root
itself (refuses before verify, correct publish argv, marker binding) lives in
bin/selftest_stage_measure.py, which executes the real script.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.stages import stage_sequence  # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:10]:
            print("        %s" % line)


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = load("measure_cloud", "bin/measure_cloud.py")
CE = load("container_entry", "bin/container_entry.py")


class Con:
    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        def log(*a, **kw):
            self.lines.append(" ".join(str(x) for x in a))
        return log

    def text(self):
        return "\n".join(self.lines)


class StubJL:
    dry = False

    def __init__(self):
        self.destroyed = []

    def destroy(self, mid):
        self.destroyed.append(mid)

    def list_instances(self):
        return [] if self.destroyed else [
            type("I", (), {"machine_id": 77, "status": "running"})()]

    def exec(self, *a, **kw):
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    def exec_stdout(self, *a, **kw):
        return ""

    def download(self, *a, **kw):
        return {"ok": True}

    def fs_delete(self, fsid):
        return {}


def teardown(tmp, **flags):
    jl, con = StubJL(), Con()
    Path(tmp).mkdir(parents=True, exist_ok=True)
    td = MC.Teardown(jl, con, Path(tmp))
    td.machine_id = 77
    lease = Path(tmp) / "lease.json"
    lease.write_text(json.dumps({"job_id": "x", "machine_id": 77,
                                 "deadline_epoch": 0}))
    td.lease_path = lease
    for key, value in flags.items():
        setattr(td, key, value)
    td.run("test")
    return td, jl, con, lease


def main():
    # RP1
    check("RP1 publish_root is appended after verify, root paths only",
          stage_sequence("root", publish_root=True)
          == ["setup", "fetch_target", "capture", "verify", "publish_root"]
          and stage_sequence("root", race=True, publish_root=True)[-2:]
          == ["verify", "publish_root"]
          and "publish_root" not in stage_sequence("quant", publish_root=True)
          and "publish_root" not in stage_sequence("root"))

    with tempfile.TemporaryDirectory() as tmp:
        # RP2: verified + unpublished -> HELD.
        td, jl, con, lease = teardown(
            tmp + "/a", root_publish_expected=True, root_verified=True,
            root_published=False)
        check("RP2 teardown HOLDS a verified, unpublished root "
              "(no destroy, lease kept)",
              jl.destroyed == [] and td.held_for_unpublished_root
              and lease.exists()
              and "NEVER PUBLISHED" in con.text(),
              "destroyed=%s held=%s\n%s" % (jl.destroyed,
                                            td.held_for_unpublished_root,
                                            con.text()[-500:]))

        # RP3: the explicit override destroys.
        td, jl, con, lease = teardown(
            tmp + "/b", root_publish_expected=True, root_verified=True,
            root_published=False, allow_unpublished_root=True)
        check("RP3 --allow-unpublished-root destroys",
              jl.destroyed == [77] and not td.held_for_unpublished_root,
              (jl.destroyed, con.text()[-300:]))

        # RP4: published -> normal teardown.
        td, jl, con, lease = teardown(
            tmp + "/c", root_publish_expected=True, root_verified=True,
            root_published=True)
        check("RP4 a published root tears down normally",
              jl.destroyed == [77] and not td.held_for_unpublished_root)

        # RP5: quant runs unaffected.
        td, jl, con, lease = teardown(tmp + "/d")
        check("RP5 a quant run is untouched by the guard",
              jl.destroyed == [77] and not td.held_for_unpublished_root)

        # RP6: the container entrypoint's job document + stage list.
        panel = Path(tmp) / "panel"
        (panel / "arrays").mkdir(parents=True)
        (panel / "panel.json").write_text(json.dumps({"panel_id": "panel--t"}))
        fs = Path(tmp) / "fs"
        fs.mkdir()
        args = CE.build_parser().parse_args([
            "capture", "--model", "MiniMaxAI/MiniMax-M3",
            "--revision", "r" * 40, "--lane", "streaming",
            "--panel-dir", str(panel), "--dataset-id", "minimaxm3-root-v1",
            "--publish-root-to", "malaiwah/minimaxm3-fidelity-root-v1",
            "--fs-root", str(fs)])
        doc = CE.job_document(args, ROOT, fs, lambda *a, **kw: None)
        check("RP6 container job.json carries capture.publish_root_to",
              doc["capture"].get("publish_root_to")
              == "malaiwah/minimaxm3-fidelity-root-v1",
              doc["capture"])
        stages = stage_sequence(
            doc["role"], race=bool(doc["capture"].get("race")),
            surface=(doc.get("target") or {}).get("surface"),
            publish_root=bool(doc["capture"].get("publish_root_to")))
        check("RP6b ...and its stage list ends with publish_root",
              stages[-1] == "publish_root", stages)
        # C3c parity: the controller's capture block carries the same key.
        import inspect
        src = inspect.getsource(MC._job_document)
        check("RP6c the SSH controller's capture block carries the same key",
              '"publish_root_to"' in src)

    # RP7: container-native runpod launch.
    from fidelity.runpodapi import RunPod
    rp = RunPod.__new__(RunPod)
    rp.dry = False
    rp.ssh_key = "/nonexistent"
    captured = {}

    def fake_gql(q, timeout=180):
        captured["q"] = q
        return {"podFindAndDeployOnDemand": {"id": "pod1", "name": "n",
                                             "costPerHr": 1.0}}

    rp._gql = fake_gql
    token = "hf_selftest_not_a_real_token_22222"
    out = rp.create(
        gpu_type="NVIDIA H100 80GB HBM3", name="fidcloud-ab12cd34-x1",
        image="ghcr.io/malaiwah/quant-fidelity-measure:latest",
        docker_args=('capture --model MiniMaxAI/MiniMax-M3 --revision %s '
                     '--dataset-id minimaxm3-root-v1 '
                     '--publish-root-to "malaiwah/minimaxm3-fidelity-root-v1"'
                     % ("r" * 40)),
        env={"HF_TOKEN": token})
    q = captured["q"]
    da = q.split("dockerArgs:")[1].split(", ports:")[0]
    check("RP7 the pod runs OUR image with OUR command",
          'imageName:"ghcr.io/malaiwah/quant-fidelity-measure:latest"' in q
          and "dockerArgs:" in q and "--publish-root-to" in q
          and out["machine_id"] == "pod1", q)
    check("RP7b the token travels as env, never inside dockerArgs",
          token not in da and ('{key:"HF_TOKEN", value:"%s"}' % token) in q,
          da)
    # ...and an image-less create still composes the legacy SSH-pod query.
    rp.create(gpu_type="NVIDIA H100 80GB HBM3", name="fidcloud-x")
    check("RP7c a plain create has no dockerArgs and keeps ssh",
          "dockerArgs" not in captured["q"] and 'ports:"22/tcp"' in captured["q"])

    # RP9: the publish receipt is written only after the fetched-back copy
    # verifies, and it pins the uploaded revision.
    FD = load("fidelity_dataset", "bin/fidelity_dataset.py")
    from fidelity import dshub as real_dshub
    import argparse as _ap
    with tempfile.TemporaryDirectory() as tmp:
        receipt = Path(tmp) / "publish-root.json"
        orig = (real_dshub.publish_dataset, real_dshub.fetch_dataset,
                FD.dsvalidate.validate_dataset)
        try:
            real_dshub.publish_dataset = lambda *a, **kw: {
                "repository": "malaiwah/mm3-root-v1",
                "dataset_sha256": "d" * 64, "private": False,
                "revision": "c" * 40}
            real_dshub.fetch_dataset = lambda *a, **kw: tmp
            FD.dsvalidate.validate_dataset = lambda *a, **kw: type(
                "R", (), {"errors": [], "passed": True})()
            rc = FD.cmd_publish(_ap.Namespace(
                dataset=tmp, repo="malaiwah/mm3-root-v1", private=False,
                revision_message="m", token_file=None, receipt=str(receipt)))
            doc = json.loads(receipt.read_text()) if receipt.is_file() else {}
            check("RP9 the publish receipt pins the uploaded revision and "
                  "is sealed",
                  rc == FD.OK and doc.get("revision") == "c" * 40
                  and doc.get("repository") == "malaiwah/mm3-root-v1"
                  and doc.get("verified_after_publish") is True
                  and doc.get("receipt_sha256"), doc)

            # ...and a fetched-back copy that does NOT verify writes nothing.
            receipt2 = Path(tmp) / "r2.json"
            FD.dsvalidate.validate_dataset = lambda *a, **kw: type(
                "R", (), {"errors": [{"code": "X", "rule": "X-1", "message": "bad",
                             "where": "here"}],
                 "warnings": [], "passed": False})()
            rc = FD.cmd_publish(_ap.Namespace(
                dataset=tmp, repo="malaiwah/mm3-root-v1", private=False,
                revision_message="m", token_file=None, receipt=str(receipt2)))
            check("RP9b no receipt when the published copy fails to verify",
                  rc != FD.OK and not receipt2.exists(), rc)
        finally:
            (real_dshub.publish_dataset, real_dshub.fetch_dataset,
             FD.dsvalidate.validate_dataset) = orig

    # RP8: refusals, before any spend.
    rc = MC.main(["--model", "x/y", "--panel", "o/p", "--lane", "streaming",
                  "--publish-root-to", "a/b"])
    check("RP8 --publish-root-to on a quant run is refused",
          rc == MC.EXIT_REFUSED, rc)
    rc = MC.main(["--role", "root", "--model", "x/y", "--panel", "o/p",
                  "--lane", "streaming", "--dataset-id", "d",
                  "--publish-root-to", "not-a-repo"])
    check("RP8b a malformed repo id is refused", rc == MC.EXIT_REFUSED, rc)

    # RP10: plan-time identity separation -- a preview aimed at the FINAL
    # repo name is refused for $0.00, mirroring the on-box stage guard.
    rc = MC.main(["--role", "root", "--race", "--model", "MiniMaxAI/MiniMax-M3",
                  "--panel", "o/p", "--lane", "streaming",
                  "--dataset-id", "mm3-root-v1.preview",
                  "--preview-of", "mm3-root-v1",
                  "--publish-root-to", "malaiwah/mm3-root-v1"])
    check("RP10 a preview aimed at the FINAL repo name refuses at plan time",
          rc == MC.EXIT_REFUSED, rc)

    # RP11 (race-composition acceptance): a container dry-run of
    # `capture --race --preview-of ... --publish-root-to ...` composes
    # without refusal and prints the race sequence ending in publish_root.
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        panel = Path(tmp) / "panel"
        (panel / "arrays").mkdir(parents=True)
        (panel / "panel.json").write_text(json.dumps({"panel_id": "panel--t"}))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CE.main([
                "capture", "--model", "MiniMaxAI/MiniMax-M3",
                "--revision", "r" * 40, "--lane", "streaming",
                "--panel-dir", str(panel),
                "--dataset-id", "minimaxm3-fidelity-root-v1.preview",
                "--preview-of", "minimaxm3-fidelity-root-v1",
                "--race", "--race-workers", "12",
                "--publish-root-to", "malaiwah/minimaxm3-fidelity-root-v1-preview",
                "--fs-root", str(Path(tmp) / "fs"), "--dry-run"])
        out = buf.getvalue()
        check("RP11 container dry-run: race + preview + publish compose "
              "(rc 0, race sequence ends in publish_root)",
              rc == CE.EXIT_OK
              and "stages: setup race_bootstrap race_capture verify "
                  "publish_root" in out,
              "rc=%s\n%s" % (rc, out[-800:]))
        doc = json.loads(out[:out.rindex("}") + 1][out.index("{"):]) \
            if "{" in out else {}
        check("RP11b ...and job.json carries race + preview + publish "
              "together (C3c parity fields)",
              doc.get("capture", {}).get("race") is True
              and doc["capture"].get("preview_of")
              == "minimaxm3-fidelity-root-v1"
              and doc["capture"].get("publish_root_to")
              == "malaiwah/minimaxm3-fidelity-root-v1-preview",
              doc.get("capture"))

    print()
    if FAILED:
        print("selftest_root_publish: %d FAILED" % len(FAILED))
        return 1
    print("selftest_root_publish: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
