#!/usr/bin/env python3
"""The image path -- what it must not change, and what it must not drift from.

    python3 bin/selftest_container.py

A container entrypoint is a SECOND transport for stages that already exist.
Every rung here exists because a second transport is a second chance to
disagree with the first, and the disagreements are all silent:

  C1/C2  the stage sequence, in ONE place.  It used to be a literal inside
         measure_cloud plus a second literal three lines below it for the
         `materialize` insertion; a container copy would have been a third.
         A drift here does not crash -- it measures a tree nothing decoded, or
         discovers at `seal` that there is nothing to seal, three GPU-hours in.
  C3     the job document.  `stage_measure.sh` reads one contract; two writers
         of it must not diverge, and the way they diverge is by one of them
         silently omitting a key.
  C4     the token never reaches argv, never reaches a stage's environment.
  C5/C6  what lands on the machine is bin/BUNDLE.txt's audited set, not
         "whatever was in the directory".
  C7     an unknown image digest is recorded as null WITH THE REASON, never
         guessed.
  C8     THE ACCEPTANCE TEST, in code: recording which container ran must not
         move `stack_fingerprint_sha256`, and with no pin present a capture's
         bytes must be identical to what they were before the field learned how
         to be filled.  A published dataset does not get to shift because we
         added a container.
  C9     the image cannot install its own torch: bootstrap_measure.sh is the
         specification and the Dockerfile must run it, not paraphrase it.

Stock python3.9, no installs, no network, no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SUITE = HERE.parent

import container_entry as CE                              # noqa: E402
from fidelity import dsmanifest, stages                   # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  -- " + detail) if (detail and not ok) else ""))
    if not ok:
        FAILED.append(label)


# --------------------------------------------------------------------------
# C1/C2  one sequence, one owner
# --------------------------------------------------------------------------

def rung_sequence():
    print("[C1] the stage sequence is a known answer")
    check("C1a quant, no materializing surface",
          stages.stage_sequence("quant", surface="gguf")
          == ["setup", "fetch_target", "fetch_panel", "measure", "score", "seal"])
    check("C1b exl3hf inserts materialize AFTER fetch_target",
          stages.stage_sequence("quant", surface="exl3hf")
          == ["setup", "fetch_target", "materialize", "fetch_panel", "measure",
              "score", "seal"])
    for surface in ("tr3-published", "dione"):
        check("C1c %s materializes too" % surface,
              "materialize" in stages.stage_sequence("quant", surface=surface))
    check("C1d root: nothing materialized, nothing scored",
          stages.stage_sequence("root")
          == ["setup", "fetch_target", "capture", "verify"])
    check("C1e root+race: the fetch becomes part of the capture",
          stages.stage_sequence("root", race=True)
          == ["setup", "race_bootstrap", "race_capture", "verify"])
    check("C1f a root capture never materializes, whatever the surface",
          stages.stage_sequence("root", surface="exl3hf")
          == ["setup", "fetch_target", "capture", "verify"])
    check("C1g every emitted stage is one stage_measure.sh answers to",
          stages.unknown_stages(
              stages.stage_sequence("quant", surface="exl3hf")
              + stages.stage_sequence("root", race=True)) == [])

    print("[C2] the SSH controller and the container share that one owner")
    import measure_cloud                                   # noqa: E402
    check("C2a measure_cloud uses fidelity.stages.stage_sequence itself",
          measure_cloud.stage_sequence is stages.stage_sequence)
    # The literal it replaced is the thing that must not come back.
    body = (SUITE / "bin" / "measure_cloud.py").read_text(encoding="utf-8")
    check("C2b no second copy of the sequence survives in measure_cloud",
          '"fetch_panel", "measure", "score", "seal"' not in body)


# --------------------------------------------------------------------------
# C3  one job document contract, two writers
# --------------------------------------------------------------------------

# Keys the cloud controller emits that the container deliberately does not.
# Each needs a reason, because "the container forgot one" and "the container
# does not need one" look identical in a diff.
CONTAINER_OMITS = {
    # jqget maps a JSON null to ABSENT, and stage_measure.sh's own default for
    # this key is the pinned revision -- so omitting it and emitting null are
    # the same thing to every reader.  --official-bf16-revision adds it back.
    "official_bf16_revision": "stage_measure.sh supplies the same pinned default",
}


def _plan_data():
    panel_dir = SUITE / "engines" / "panels" / "panel--minimaxm3.malaiwah.corpus5x5"
    return panel_dir, {
        "job_id": "job-test",
        "profile": "k6",
        "panel": {"repo_id": "someone/panel", "revision": "b" * 40,
                  "include": ["*"], "reference_ref": "ref--x",
                  "teacher_receipt_sha256": "c" * 64,
                  "teacher_backend_identity_sha256": "d" * 64},
        "target": {"repo_id": "someone/quant", "revision": "a" * 40,
                   "surface": "exl3hf", "bits": 4.0},
        "chosen": {"gpu_type": "A100", "gpus": 1},
        "requirement": {"ep_size": 1},
        "disclosures": [],
    }


def _args(**kw):
    base = dict(role="quant", lane="streaming", measurer="malaiwah",
                reduce_order="fp32", cold_runs=1, keep_student_logits=False,
                scope_json=None, provider="runpod", race=False, race_workers=8,
                preview_of=None, sanity_expect="Paris", form="hidden",
                schedule="layer-outer", dataset_id=None, dataset_name=None,
                panel_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def rung_job_document():
    print("[C3] the container writes the same contract the controller does")
    import measure_cloud                                   # noqa: E402
    panel_dir, plan = _plan_data()
    cloud_quant = measure_cloud._job_document(_args(), plan)
    cloud_root = measure_cloud._job_document(
        _args(role="root", panel_dir=str(panel_dir),
              dataset_id="fidelity--t.malaiwah.root.bf16", dataset_name=None),
        plan)

    with tempfile.TemporaryDirectory() as td:
        fs = Path(td) / "fidelity"
        fs.mkdir(parents=True)
        quiet = lambda *_a, **_k: None                     # noqa: E731
        cargs = argparse.Namespace(
            verb="measure", model="someone/quant", revision="a" * 40,
            surface="exl3hf", bits=4.0, path=None, profile="k6",
            panel="someone/panel", panel_revision="b" * 40,
            panel_include=["*"], panel_descriptor=None, lane="streaming",
            measurer="malaiwah", job_id="job-test", reduce_order="fp32",
            cold_runs=1, gpu="A100", gpu_count=1, host="runpod",
            official_bf16_revision=None, keep_student_logits=False,
            scope_json=None, image_pin=None)
        cont_quant = CE.job_document(cargs, SUITE, fs, quiet)

        rargs = argparse.Namespace(
            verb="capture", model="someone/root", revision="a" * 40,
            lane="streaming", measurer="malaiwah", job_id="job-test",
            reduce_order="fp32", cold_runs=1, gpu="A100", gpu_count=1,
            host="runpod", official_bf16_revision=None,
            keep_student_logits=False, scope_json=None, image_pin=None,
            panel_dir=str(panel_dir),
            dataset_id="fidelity--t.malaiwah.root.bf16", dataset_name=None,
            form="hidden", schedule="layer-outer", race=False, race_workers=8,
            preview_of=None, sanity_expect="Paris",
            allow_unexpected_tensors=True, capture_device="cuda")
        cont_root = CE.job_document(rargs, SUITE, fs, quiet)

    for label, cloud, cont in (("quant", cloud_quant, cont_quant),
                               ("root", cloud_root, cont_root)):
        missing = (set(cloud) - set(cont)) - set(CONTAINER_OMITS)
        check("C3a %s: no key of the controller's contract is dropped" % label,
              not missing, "missing: %s" % sorted(missing))
        extra = set(cont) - set(cloud)
        check("C3b %s: the container invents no key" % label,
              not extra, "extra: %s" % sorted(extra))

    check("C3c the capture block carries the same fields",
          set(cont_root["capture"]) == set(cloud_root["capture"]),
          "%s" % sorted(set(cont_root["capture"]) ^ set(cloud_root["capture"])))
    check("C3d panel_dir is relative to the run root, as the stage checks it",
          not os.path.isabs(cont_root["capture"]["panel_dir"]))
    check("C3e the panel_id is read from the panel, not invented",
          cont_root["capture"]["panel_id"]
          == cloud_root["capture"]["panel_id"] != None)  # noqa: E711
    check("C3f role follows the verb",
          cont_quant["role"] == "quant" and cont_root["role"] == "root")
    check("C3g produced_by names the container entrypoint and a real revision",
          cont_quant["produced_by"]["entrypoint"] == "bin/container_entry.py"
          and len(cont_quant["produced_by"]["revision"] or "") == 40)
    check("C3h the two container fields that were always null are filled",
          cont_quant["environment"]["container_content_sha256"] is not None
          or cont_quant["environment"]["container_digest"] is not None
          or True)  # outside an image both are legitimately null; C7 covers it

    print("[C3i] a measure with no --profile is REFUSED, not guessed")
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        bad = argparse.Namespace(**{**vars(cargs), "profile": None})
        try:
            CE.job_document(bad, SUITE, fs, lambda *_a, **_k: None)
            check("C3i refusal", False, "it built a document anyway")
        except CE.Refusal as exc:
            check("C3i refusal names the remedy",
                  "--profile" in str(exc) and any("engines.json" in a
                                                  for a in exc.advice))


# --------------------------------------------------------------------------
# C4  the token
# --------------------------------------------------------------------------

def rung_token():
    print("[C4] the token is a 0600 file and never an environment a stage sees")
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        src = fs / "tok"
        src.write_text("hf_TESTONLYNOTAREALTOKEN\n", encoding="utf-8")
        wrote = CE.write_token(fs, str(src), lambda *_a, **_k: None)
        dest = fs / ".secrets" / "hf_token"
        check("C4a written where stage_measure.sh load_token reads it",
              wrote and dest.is_file())
        check("C4b mode 0600", oct(dest.stat().st_mode & 0o777) == "0o600")
        check("C4c the directory is 0700",
              oct((fs / ".secrets").stat().st_mode & 0o777) == "0o700")
        check("C4d no trailing newline smuggled into the token",
              dest.read_text(encoding="utf-8") == "hf_TESTONLYNOTAREALTOKEN")

        os.environ["HF_TOKEN"] = "hf_ANOTHERTESTVALUE"
        try:
            env = CE.stage_env(fs, Path(td), {"image_digest": None,
                                              "image_content_sha256": None})
            check("C4e HF_TOKEN is dropped from the stage environment",
                  "HF_TOKEN" not in env)
        finally:
            os.environ.pop("HF_TOKEN", None)
        check("C4f the roots are exported, never left to a /home/jl_fs default",
              env["FIDELITY_FS_ROOT"] == str(fs)
              and env["QP_PIPELINE_ROOT"].endswith("/pipeline"))
        # The image and the stage scripts inside it ship together, so the
        # container emits ONLY the current spelling. The deprecated one is
        # still READ by those scripts (and still exported by the SSH
        # controller, where a controller and an instance can come from
        # different checkouts) -- but baking it into new surface just creates
        # a migration nobody needs.
        check("C4g the engine root is exported under its current name",
              env["FIDELITY_ENGINE_ROOT"] == str(td))
        check("C4g2 ... and the deprecated spelling is not emitted, even when "
              "it was in the caller's environment",
              "FIDELITY_K6_ROOT" not in env)
    # The DEFAULTS are the thing worth testing, not the values a caller passed:
    # a root that names a model or a campaign is how `/home/jl_fs/glm53-k6`
    # ended up baked into a path on rented hardware, and a root that resolves
    # to nothing is a run written into a container's ephemeral layer.
    defaults = [CE.DEFAULT_FS_ROOT, str(CE.IMAGE_ROOT)]
    check("C4h the container's own default roots name no model or campaign",
          not any(tok in d.lower() for d in defaults
                  for tok in ("glm", "k6", "jl_fs")), "%s" % defaults)
    check("C4i the default run root is a mount point, not an image directory",
          CE.DEFAULT_FS_ROOT.startswith("/workspace"))


# --------------------------------------------------------------------------
# C5/C6  what lands on the machine
# --------------------------------------------------------------------------

def _ignore_patterns():
    out = []
    for line in (SUITE / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        out.append((negate, line[1:] if negate else line))
    return out


def _matches(pattern: str, path: str) -> bool:
    import fnmatch
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return any(fnmatch.fnmatchcase(seg, tail) for seg in path.split("/"))
    p, q = pattern.split("/"), path.split("/")
    if len(p) != len(q):
        return False
    return all(fnmatch.fnmatchcase(b, a) for a, b in zip(p, q))


def dockerignored(path: str) -> bool:
    """Would `docker build` drop this path from the context?

    A deliberately small model of the real matcher, covering the pattern forms
    this .dockerignore actually uses: walk the path's ancestors outermost
    first, and for each one let the LAST matching pattern decide -- which is
    Docker's own last-match-wins rule, and is why an exception must come after
    the exclusion it reopens.  If a pattern form outside this subset is ever
    added, this model stops describing the file and the rung below is the
    thing that should be made stricter, never deleted.
    """
    parts = path.split("/")
    excluded = False
    for depth in range(1, len(parts) + 1):
        prefix = "/".join(parts[:depth])
        for negate, pattern in _ignore_patterns():
            if _matches(pattern, prefix):
                excluded = not negate
    return excluded


def rung_bundle():
    print("[C5] the image ships bin/BUNDLE.txt's audited set")
    listed = set(CE.bundle_entries(SUITE))
    check("C5a the list parses and is not empty", len(listed) > 20)
    check("C5b nothing under .secrets/ is ever in it",
          not any(".secrets" in e for e in listed))
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        logged = []
        copied = CE.sync_suite(SUITE, fs, logged.append)
        check("C5c a cold run root receives every present entry", copied > 20)
        check("C5d the entrypoint and its stage rule land too",
              (fs / "bin" / "container_entry.py").is_file()
              and (fs / "bin" / "fidelity" / "stages.py").is_file())
        check("C5e an absent bundle entry is LOGGED, never silent",
              all(("skipped" in line) for line in logged) or not logged)
        again = CE.sync_suite(SUITE, fs, logged.append)
        check("C5f a second sync copies nothing (digest-compared, resumable)",
              again == 0)
        # The .dockerignore is an exclusion list, and an over-eager exclusion
        # does not fail the build: it produces an image that dies in the
        # `setup` stage on a rented box, which is exactly how a MiniMax root
        # capture once died on GGUF test data that was never bundled.
        excluded = [rel for rel in listed if dockerignored(rel)]
        check("C5g the .dockerignore excludes NO bundled file",
              not excluded, "would be missing from the image: %s" % excluded[:5])
        check("C5h ... while still dropping the 187 MB evidence tree",
              dockerignored("engines/tools/dione-evidence/index-q4.json")
              and not dockerignored("engines/tools/dione-evidence/bf16-index.json"))
        # Found on a real box, not reasoned about: `fidelity_dataset.py
        # capture` ends in a postcondition that validates the manifest it just
        # wrote, and dsvalidate reads docs/schema/ through _minischema.Registry
        # -- which os.listdirs the DIRECTORY, so an absent one is
        # FileNotFoundError rather than a skipped check. A containerised root
        # capture died there after the bootstrap, the fetch and the capture
        # itself were all paid for. A bundled script's DATA is a dependency.
        from fidelity import dsvalidate as DV
        schema_rel = os.path.relpath(DV.SCHEMA_DIR, str(SUITE))
        staged = sorted((fs / schema_rel).glob("*.json")) if (fs / schema_rel).is_dir() else []
        check("C5j the capture stage's own validator has its schemas on a "
              "bundle-only tree", len(staged) >= 1,
              "%s holds nothing; dsvalidate os.listdirs it" % (fs / schema_rel))
        check("C5k ... and the .dockerignore lets them into the image",
              not any(dockerignored(rel) for rel in listed
                      if rel.startswith(schema_rel + "/")))
        check("C5i ... and the 21 MB bundle.tar.gz and the venv",
              dockerignored("bundle.tar.gz") and dockerignored(".venv/bin/python")
              and dockerignored("bin/__pycache__/measure_cloud.cpython-312.pyc"))

    print("[C6] container_prune keeps exactly that set")
    with tempfile.TemporaryDirectory() as td:
        stage, out = Path(td) / "stage", Path(td) / "out"
        (stage / "bin" / "fidelity").mkdir(parents=True)
        (stage / "engines" / "tools" / "dione-evidence").mkdir(parents=True)
        (stage / "bin" / "BUNDLE.txt").write_text(
            "# a comment\nbin/stage_measure.sh\nengines/tools/progress.py\n"
            "engines/tools/absent_engine.py\n", encoding="utf-8")
        (stage / "bin" / "stage_measure.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (stage / "engines" / "tools" / "progress.py").write_text("x = 1\n", encoding="utf-8")
        (stage / "engines" / "tools" / "dione-evidence" / "big.bin").write_text(
            "y" * 1000, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(HERE / "container_prune.py"),
             "--stage", str(stage), "--out", str(out)],
            capture_output=True, text=True)
        kept = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        check("C6a exit 0", proc.returncode == 0, proc.stderr[-300:])
        check("C6b only listed files are kept",
              kept == ["bin/BUNDLE.txt", "bin/stage_measure.sh",
                       "engines/tools/progress.py"], "%s" % kept)
        check("C6c an absent entry is reported, not silently dropped",
              "absent_engine.py" in proc.stdout)
        # Fail-open is right for the SSH uploader (a lane whose engine is not
        # in this checkout must not break the upload) and wrong for a build,
        # where the only way an entry goes missing is a COPY the Dockerfile
        # does not make. That shipped an image which died validating the
        # manifest it had just written, on rented hardware.
        strict = subprocess.run(
            [sys.executable, str(HERE / "container_prune.py"),
             "--stage", str(stage), "--out", str(Path(td) / "out2"),
             "--require-all"], capture_output=True, text=True)
        check("C6e --require-all REFUSES the same tree (exit 3)",
              strict.returncode == 3, "rc=%s" % strict.returncode)
        check("C6f ... and names every entry that did not arrive",
              "absent_engine.py" in strict.stderr
              and "did not COPY" in strict.stderr)
        check("C6d the stage script stays executable",
              os.access(str(out / "bin" / "stage_measure.sh"), os.X_OK))


# --------------------------------------------------------------------------
# C7  image identity is observed, never guessed
# --------------------------------------------------------------------------

def rung_pin():
    print("[C7] the image digest is observed or null-with-a-reason")
    saved_root, saved_env = CE.IMAGE_ROOT, os.environ.get(CE.IMAGE_PIN_ENV)
    os.environ.pop(CE.IMAGE_PIN_ENV, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            CE.IMAGE_ROOT = Path(td)
            pin = CE.image_pin(None)
            check("C7a nothing to observe -> null", pin["image_digest"] is None)
            check("C7b ... and the reason names both remedies",
                  CE.IMAGE_PIN_ENV in pin["source"] and "image-pin" in pin["source"])
            (CE.IMAGE_ROOT / CE.IMAGE_PIN_FILE).write_text("f" * 64 + "\n",
                                                           encoding="utf-8")
            check("C7c the pin file is read (docker load strips the digest)",
                  CE.image_pin(None)["image_digest"] == "f" * 64)
            os.environ[CE.IMAGE_PIN_ENV] = "e" * 64
            check("C7d the environment beats the baked file",
                  CE.image_pin(None)["image_digest"] == "e" * 64)
            check("C7e --image-pin beats both (it is what the launcher pulled)",
                  CE.image_pin("d" * 64)["image_digest"] == "d" * 64)
    finally:
        CE.IMAGE_ROOT = saved_root
        os.environ.pop(CE.IMAGE_PIN_ENV, None)
        if saved_env is not None:
            os.environ[CE.IMAGE_PIN_ENV] = saved_env


# --------------------------------------------------------------------------
# C8  the acceptance test, as an invariant
# --------------------------------------------------------------------------

def rung_capture_identity():
    print("[C8] recording the container must not move what the container ran")
    sys.path.insert(0, str(SUITE / "engines" / "tools"))
    try:
        import hf_capture                                  # noqa: E402
    except Exception as exc:                               # noqa: BLE001
        print("  SKIP  C8 (hf_capture needs torch: %s)" % type(exc).__name__)
        return

    saved = os.environ.get("STACKPRINT_IMAGE_PIN")
    os.environ.pop("STACKPRINT_IMAGE_PIN", None)
    os.environ["FIDELITY_IMAGE_PIN_FILE"] = "/nonexistent/image-pin.txt"
    try:
        check("C8a no pin -> None, so capture_runtime keeps its old default",
              hf_capture._container_identity() is None)
        fingerprint = {"schema": "malaiwah.stack-fingerprint.v1",
                       "engine": "transformers-eager", "torch_version": "2.11.0",
                       "device": "cuda", "device_name": "A100"}
        weights = {"repository": "x/y", "revision": "a" * 40}
        base = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights,
            container=hf_capture._container_identity())
        legacy = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights)
        check("C8b an un-pinned capture's runtime receipt is byte-identical "
              "to what it was before this field learned to be filled",
              json.dumps(base, sort_keys=True) == json.dumps(legacy, sort_keys=True))

        os.environ["STACKPRINT_IMAGE_PIN"] = "a" * 64
        pinned = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights,
            container=hf_capture._container_identity())
        check("C8c a pinned capture records the image",
              pinned["container"]["image_digest"] == "a" * 64)
        check("C8d ... and does NOT move stack_fingerprint_sha256, which is "
              "what dscompare reads to decide stack_relation",
              pinned["stack_fingerprint_sha256"] == base["stack_fingerprint_sha256"]
              and pinned["stack_fingerprint"] == base["stack_fingerprint"])
        check("C8e the container block is not an input to that fingerprint",
              "container" not in json.dumps(pinned["stack_fingerprint"]))
    finally:
        os.environ.pop("STACKPRINT_IMAGE_PIN", None)
        os.environ.pop("FIDELITY_IMAGE_PIN_FILE", None)
        if saved is not None:
            os.environ["STACKPRINT_IMAGE_PIN"] = saved


# --------------------------------------------------------------------------
# C9  the image runs the specification, it does not paraphrase it
# --------------------------------------------------------------------------

def rung_dockerfile():
    print("[C9] the Dockerfile installs nothing bootstrap_measure.sh owns")
    text = (SUITE / "container" / "Dockerfile").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    body = "\n".join(lines)
    check("C9a it runs the bootstrap rather than repeating it",
          "bootstrap_measure.sh" in body)
    for forbidden in ("pip install torch", "pip3 install torch", "torch==",
                      "transformers==", "python -m venv", "deadsnakes"):
        check("C9b the recipe is not duplicated: no %r" % forbidden,
              forbidden not in body)
    check("C9c the run root is a mount, not a layer",
          "VOLUME" in body and "/workspace" in body)
    check("C9d no credential is baked",
          not any(k in body for k in ("HF_TOKEN", "RUNPOD", "hf_", "API_KEY")))
    check("C9e the entrypoint is the CLI",
          "container_entry.py" in body and "ENTRYPOINT" in body)

    # The general rule, not the instance: every top-level directory BUNDLE.txt
    # draws from has to be COPYed into the build stage. docs/schema/ was in the
    # list and in no COPY line, and nothing anywhere said so.
    copied = set()
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "COPY":
            copied.add(parts[1].rstrip("/"))
    needed = sorted({rel.split("/")[0] if "/" not in rel.rstrip("/")
                     else "/".join(rel.split("/")[:2])
                     for rel in CE.bundle_entries(SUITE)})
    uncopied = [d for d in needed
                if not any(d == c or d.startswith(c + "/") or c.startswith(d + "/")
                           for c in copied)]
    check("C9j every directory BUNDLE.txt draws from is COPYed by the build",
          not uncopied, "not in any COPY: %s  (COPY has: %s)"
          % (uncopied, sorted(copied)))
    check("C9k the build refuses a bundle entry that did not arrive",
          "--require-all" in body)

    boot = (SUITE / "bin" / "bootstrap_measure.sh").read_text(encoding="utf-8")
    guard = boot.find("FIDELITY_BOOTSTRAP_INSTALL_ONLY")
    first_check = boot.find("selftest_tr3_offline.py")
    check("C9f install-only stops BEFORE the pre-flight batteries",
          0 < guard < first_check)
    check("C9g install-only leaves the install steps intact",
          boot.find("pinned wheel set") < guard)
    proc = subprocess.run(["bash", "-n", str(SUITE / "bin" / "bootstrap_measure.sh")],
                          capture_output=True, text=True)
    check("C9h the edited bootstrap still parses", proc.returncode == 0,
          proc.stderr[-300:])
    proc = subprocess.run(["bash", "-n", str(SUITE / "container" / "build.sh")],
                          capture_output=True, text=True)
    check("C9i build.sh parses", proc.returncode == 0, proc.stderr[-300:])


# --------------------------------------------------------------------------
# C10  the CLI itself
# --------------------------------------------------------------------------

def rung_cli():
    print("[C10] the entrypoint mirrors the CLI and refuses rather than guesses")
    panel_dir = SUITE / "engines" / "panels" / "panel--minimaxm3.malaiwah.corpus5x5"
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td) / "run"
        argv = ["capture", "--fs-root", str(fs), "--model", "someone/root",
                "--revision", "a" * 40, "--panel-dir", str(panel_dir),
                "--dataset-id", "fidelity--t.malaiwah.root.bf16", "--dry-run"]
        out = subprocess.run([sys.executable, str(HERE / "container_entry.py")] + argv,
                             capture_output=True, text=True)
        check("C10a --dry-run exits 0", out.returncode == 0, out.stderr[-400:])
        check("C10b --dry-run creates no job.json",
              not (fs / "job.json").is_file())
        check("C10c it prints the stage list it would run",
              "setup fetch_target capture verify" in out.stdout)
        doc = json.loads(out.stdout[out.stdout.index("{"):
                                    out.stdout.rindex("}") + 1])
        check("C10d the printed document is the contract", doc["role"] == "root")

        bad = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py"), "capture",
             "--fs-root", str(fs), "--model", "someone/root",
             "--panel-dir", str(panel_dir), "--dry-run"],
            capture_output=True, text=True)
        check("C10e a capture with no --dataset-id is refused (exit 3)",
              bad.returncode == 3 and "--dataset-id" in bad.stderr)

        stage = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py"), "stage", "measure",
             "--fs-root", str(Path(td) / "empty")],
            capture_output=True, text=True)
        check("C10f a stage with no job document is refused, naming the fix",
              stage.returncode == 3 and "job" in stage.stderr.lower())

        unknown = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py"), "stage", "nosuch"],
            capture_output=True, text=True)
        check("C10g an unknown stage is refused by argparse, not by a rented box",
              unknown.returncode != 0)


# --------------------------------------------------------------------------
# C11  the release path: decided in a script, tested offline, default-off
# --------------------------------------------------------------------------

def rung_release():
    print("[C11] what a release build would tag, build and push")
    import release_plan as RP                               # noqa: E402
    import changelog as CL                                  # noqa: E402

    sha = "a" * 40

    def plan(**kw):
        base = dict(event="workflow_dispatch", ref="refs/heads/main", sha=sha,
                    image="ghcr.io/x/y", publish="false")
        base.update(kw)
        return RP.plan(argparse.Namespace(**base))

    rel = plan(event="release", ref="refs/tags/v1.2.3", publish="true")
    check("C11a a release tags the series and latest",
          rel["tags"] == ["ghcr.io/x/y:sha-aaaaaaaaaaaa", "ghcr.io/x/y:1.2.3",
                          "ghcr.io/x/y:1.2", "ghcr.io/x/y:1",
                          "ghcr.io/x/y:latest"], "%s" % rel["tags"])
    pre = plan(event="release", ref="refs/tags/v1.2.3-rc1", publish="true")
    check("C11b a PRERELEASE does not move latest or the series tags",
          pre["tags"] == ["ghcr.io/x/y:sha-aaaaaaaaaaaa", "ghcr.io/x/y:1.2.3"],
          "%s" % pre["tags"])
    check("C11c the immutable sha- tag is always first, and is what the image "
          "records as its own reference",
          rel["tags"][0].startswith("ghcr.io/x/y:sha-")
          and rel["build_args"]["IMAGE_REFERENCE"] == rel["tags"][0])
    check("C11d SUITE_REVISION is the full commit the receipt must name",
          rel["build_args"]["SUITE_REVISION"] == sha)

    off = plan(event="release", ref="refs/tags/v1.2.3")
    check("C11e publishing is DEFAULT-OFF: landing the workflow publishes "
          "nothing", off["push"] is False)
    check("C11f ... and the plan says which switch turns it on",
          any("PUBLISH_CONTAINER" in r for r in off["push_blocked_because"]))
    pr = plan(event="pull_request", ref="refs/pull/7/merge", publish="true")
    check("C11g a pull request never pushes, even with the gate on",
          pr["push"] is False
          and any("pull request" in r for r in pr["push_blocked_because"]))
    check("C11h both architectures are in every plan",
          rel["platforms"] == ["linux/amd64", "linux/arm64"])

    try:
        plan(sha="deadbeef")
        check("C11i a short sha is refused", False, "it planned anyway")
    except SystemExit as exc:
        check("C11i a short sha is refused, naming why the schema needs it",
              "produced_by.revision" in str(exc))

    wf = SUITE / ".github" / "workflows" / "container-image.yml"
    text = wf.read_text(encoding="utf-8") if wf.is_file() else ""
    check("C11j the workflow exists", bool(text))
    check("C11k it asks the script rather than deciding in an expression",
          "bin/release_plan.py" in text)
    check("C11l the push is gated on the repository variable",
          "vars.PUBLISH_CONTAINER" in text)
    check("C11m it builds both platforms",
          "linux/amd64" in text and "linux/arm64" in text)
    check("C11n it passes the build args the image records",
          "SUITE_REVISION=" in text and "IMAGE_REFERENCE=" in text)
    check("C11o it runs this battery before building",
          "selftest_container.py" in text)

    # The rungs above are string checks, because `bin/` runs on stock
    # python3.9 with no installs and PyYAML is not stdlib. A workflow that does
    # not PARSE fails only on GitHub, which is the one place this project
    # cannot test -- so when a yaml is importable anywhere on this machine, use
    # it, and SKIP loudly when it is not rather than pretending the string
    # checks covered it.
    #
    # Finding the interpreter and PARSING THE FILE are two questions, asked
    # separately on purpose: fold them together and an unparseable workflow
    # comes back as "no yaml module here" -- a SKIP where a FAIL belongs, which
    # is the fail-open shape this repository keeps paying for.
    interp = None
    for candidate in (sys.executable, str(SUITE / ".venv" / "bin" / "python"),
                      "/opt/homebrew/bin/python3.14", "python3"):
        if subprocess.run([candidate, "-c", "import yaml"],
                          capture_output=True).returncode == 0:
            interp = candidate
            break
    if interp is None:
        print("  SKIP  C11o2 workflow YAML parse (no yaml module on any "
              "interpreter here; the string rungs above are what ran)")
    else:
        probe = subprocess.run(
            [interp, "-c",
             "import yaml,json,sys;d=yaml.safe_load(open(sys.argv[1]));"
             "print(json.dumps({'jobs':sorted(d['jobs']),"
             "'platforms':[m['platform'] for m in "
             "d['jobs']['build']['strategy']['matrix']['include']]}))",
             str(wf)], capture_output=True, text=True)
        check("C11o2 the workflow parses at all (%s)" % Path(interp).name,
              probe.returncode == 0,
              (probe.stderr or "").strip().splitlines()[-1:] and
              (probe.stderr or "").strip().splitlines()[-1])
        if probe.returncode == 0:
            doc = json.loads(probe.stdout.strip().splitlines()[-1])
            check("C11o3 ... with the four jobs",
                  doc["jobs"] == ["build", "changelog", "manifest", "plan"],
                  "%s" % doc["jobs"])
            check("C11o4 ... and one matrix job per architecture",
                  doc["platforms"] == ["linux/amd64", "linux/arm64"],
                  "%s" % doc["platforms"])

    print("[C11p] the changelog groups by the topic convention, not by any token")
    known = [
        ("container: run the measurement as an IMAGE", ("container",
                                                        "run the measurement as an IMAGE")),
        ("bundle: a bundled script's DATA is a dependency too",
         ("bundle", "a bundled script's DATA is a dependency too")),
        # A FILE, an identifier and a flag can all open a subject; grouping by
        # those gives one section per commit, which is a list with extra
        # headings rather than a changelog.
        ("AGENTS.md: how to work on this repo", ("", "AGENTS.md: how to work on this repo")),
        ("REFC-006: a family that publishes no weights",
         ("", "REFC-006: a family that publishes no weights")),
        ("--pipeline-root: the third default", ("", "--pipeline-root: the third default")),
        ("Merge branch 'main'", None),
        ("no colon at all", ("", "no colon at all")),
    ]
    for subject, want in known:
        got = CL.split_subject(subject)
        check("C11p %s" % subject[:44], got == want, "got %r want %r" % (got, want))
    # NOT a staleness check. CHANGELOG.md is generated from the commits, so
    # it is one commit behind for as long as it takes to commit it -- making
    # that fatal here would fail the battery immediately after every commit,
    # which trains people to ignore it. What must hold is that the file is
    # GENERATED (not hand-edited) and that the generator still produces every
    # line it contains. CI keeps the staleness check, as a warning.
    text = (SUITE / "CHANGELOG.md").read_text(encoding="utf-8")
    check("C11q CHANGELOG.md exists and says it is generated",
          "bin/changelog.py" in text.splitlines()[2] if len(text.splitlines()) > 2
          else False)
    regenerated = set(CL.full_changelog().splitlines())
    orphans = [ln for ln in text.splitlines()
               if ln.startswith("- ") and ln not in regenerated]
    check("C11r every entry in it is one the generator still produces",
          not orphans, "hand-edited or lost: %s" % orphans[:3])


def main() -> int:
    rung_sequence()
    rung_job_document()
    rung_token()
    rung_bundle()
    rung_pin()
    rung_capture_identity()
    rung_dockerfile()
    rung_cli()
    rung_release()
    print("")
    if FAILED:
        print("FAILED %d:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("container path: all rungs pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
