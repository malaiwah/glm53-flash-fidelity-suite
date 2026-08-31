#!/usr/bin/env python3
"""The image entrypoint: run the measurement, do not orchestrate a machine.

    docker run --gpus all -v /data:/workspace <image> \
        measure --model <repo> --revision <40-hex> --surface exl3hf --bits 4.0 \
                --panel-descriptor /panel.json --lane streaming
    docker run --gpus all -v /data:/workspace <image> \
        capture --model <repo> --revision <40-hex> --panel-dir /panel \
                --dataset-id fidelity--x.y.root.bf16 --lane streaming
    docker run <image> stage setup --job /job.json
    docker run <image> doctor

WHY THIS EXISTS
---------------
Porting this suite to three clouds produced five defects and not one of them
was about the measurement:

  * machine ids that are not integers (crashed AFTER the pod existed, twice,
    leaving a billing instance unadopted);
  * a running state spelled "Running" on one provider and "RUNNING" on
    another (two healthy polls read as a preemption, and a box was torn down
    mid-bootstrap);
  * those same ids compared as ints inside a set, in the "is it really gone?"
    check -- a type mismatch there reports a LIVE instance as destroyed;
  * a 100 GB disk, because one provider has a filesystem that outlives its
    instance and nobody else does (`No space left on device`, 45 minutes into
    a paid run);
  * three hardcoded `/home/jl_fs` roots, the worst of which left an A100
    sitting at 0% GPU for two hours at $1.59/h.

Every one of them is an artefact of orchestrating a MACHINE -- create it, poll
its state, size its disk, ssh into it, guess where its filesystem is mounted --
instead of running an IMAGE.  A container entrypoint deletes the category:
there is no id to parse, no state to poll, no disk to size, and the filesystem
root is a mount the caller chose.  Providers that take a custom image (RunPod,
Vast, Lambda) run this directly; the SSH path stays, because JarvisLabs is
driven through its own CLI and because a new transport proves itself before it
replaces a working one.

WHAT IT DOES NOT DO
-------------------
It does not reimplement a stage.  `bin/stage_measure.sh` owns what a stage
does, `bin/fidelity/stages.py` owns which stages run, and both are used here
verbatim -- so a containerised run is the same stage sequence, receipt-
resumable in the same way, writing the same markers.  This file's whole job is
to turn a command line into the `job.json` contract those stages already read,
and to make the environment they assume true inside a container.

It contains no credential.  The HF token arrives at runtime as `--token-file`
or `HF_TOKEN`, is written to the 0600 file the stages already read, and never
reaches argv -- the property `bin/measure_cloud.py` has and must keep.

Stdlib only, python3.9-clean: this runs before any venv is on PATH, and it is
exercised by `bin/selftest_container.py` on a laptop with no torch.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import shred_secret_file, write_secret_file  # noqa: E402
from fidelity.stages import KNOWN_STAGES, stage_sequence  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 3

# Where the Dockerfile bakes things.  Both are overridable so this file can be
# exercised outside a container (the selftest does exactly that).
IMAGE_ROOT = Path(os.environ.get("FIDELITY_IMAGE_ROOT", "/opt/fidelity"))
BUILD_MANIFEST = "BUILD.json"
IMAGE_PIN_FILE = "image-pin.txt"

# `stackprint._container_block` already reads this env var, and the serving
# pipeline already writes an image-pin file, because `docker load` strips the
# digest and the file is then the only trustworthy source.  Reuse the
# convention rather than inventing a second one.
IMAGE_PIN_ENV = "STACKPRINT_IMAGE_PIN"

DEFAULT_FS_ROOT = "/workspace/fidelity"


class Refusal(RuntimeError):
    """A refusal names its remedy; it is not a stack trace."""

    def __init__(self, reason: str, advice=()) -> None:
        super().__init__(reason)
        self.advice = list(advice)


# --------------------------------------------------------------------------
# the baked image
# --------------------------------------------------------------------------


def build_manifest() -> dict:
    """What the image was built from, or an empty dict outside one.

    Written by the Dockerfile.  It is the reason a containerised run can emit
    a real `produced_by` block: on an SSH-driven instance there is no git
    checkout, so the controller has to compute that block on the caller's
    laptop and ship it in job.json.  In an image the revision is baked, so the
    run can name its own code without anyone shipping it a promise.
    """
    path = IMAGE_ROOT / BUILD_MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}


def image_pin(explicit=None) -> dict:
    """The identity of the image we are running inside.

    A container cannot ask Docker for its own digest, so the digest is what
    the LAUNCHER knew: `--image-pin` (what measure_cloud passed, or what the
    operator pulled), else the env var, else the file the build wrote.  Never
    guessed: an unknown digest is recorded as null with the reason, exactly as
    `stackprint._container_block` does it.
    """
    build = build_manifest()
    ref = build.get("image_reference")
    content = build.get("image_content_sha256")
    digest = explicit or os.environ.get(IMAGE_PIN_ENV) or ""
    source = "argv:--image-pin" if explicit else (
        "env:%s" % IMAGE_PIN_ENV if digest else None)
    if not digest:
        try:
            digest = (IMAGE_ROOT / IMAGE_PIN_FILE).read_text(
                encoding="utf-8").strip().split()[0]
            source = "image-pin-file:%s" % (IMAGE_ROOT / IMAGE_PIN_FILE)
        except Exception:                                  # noqa: BLE001
            digest = ""
    return {
        "image_digest": digest.strip() or None,
        "image_reference": ref,
        "image_repository_digest": None,
        # NOT a substitute for the registry digest and never presented as one:
        # this is the digest of the manifest the build wrote over its own
        # inputs (pip freeze, the pipeline pin, the patch series, every bundled
        # file).  It answers "is this the same stack?" on a box where the
        # registry digest was stripped by `docker load`.
        "image_content_sha256": content,
        "source": source or (
            "undetected (docker load strips digests; pass --image-pin, set %s, "
            "or write %s)" % (IMAGE_PIN_ENV, IMAGE_ROOT / IMAGE_PIN_FILE)),
    }


# --------------------------------------------------------------------------
# filesystem
# --------------------------------------------------------------------------


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_entries(suite: Path):
    """The upload set, read from the ONE list that already defines it.

    `bin/BUNDLE.txt` exists so that what lands on rented hardware is auditable
    rather than "whatever was in the directory".  The image is a second
    transport for the same set, so it reads the same list instead of keeping a
    parallel one that can drift.
    """
    text = (suite / "bin" / "BUNDLE.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.startswith("#")]


def sync_suite(suite: Path, fs_root: Path, con) -> int:
    """Put the baked code where the stage scripts look for it.

    `stage_measure.sh` addresses everything as `$FS/bin/...`, `$FS/engines/...`,
    `$FS/registry/...`, and the same `$FS` also holds the models, the panel and
    the receipts.  In a container the code is immutable (it is an image layer)
    and the data is a mount, so the code is copied into the mount once, by
    digest, and re-copied only when it differs.  Copying is a few megabytes and
    makes a restarted container self-heal; a symlink farm would not survive a
    provider that wipes the mount.
    """
    if suite.resolve() == fs_root.resolve():
        con("suite already at the run root; nothing to sync")
        return 0
    copied = 0
    wanted = list(bundle_entries(suite))
    # The entrypoint and the stage-sequence rule it imports are not in
    # BUNDLE.txt's SSH-era set; they must still land, or a stage run from
    # inside $FS would import a file that is not there.
    for extra in ("bin/container_entry.py", "bin/fidelity/stages.py"):
        if extra not in wanted:
            wanted.append(extra)
    for rel in wanted:
        src = suite / rel
        if not src.is_file():
            # BUNDLE.txt's own policy: missing files are skipped and the
            # skipping is LOGGED, so a lane whose engine is absent from this
            # checkout does not break the transport.
            con("bundle entry not baked into the image, skipped: %s" % rel)
            continue
        dst = fs_root / rel
        if dst.is_file() and sha256_file(str(dst)) == sha256_file(str(src)):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        copied += 1
    return copied


def stage_panel(panel_dir, fs_root: Path, con) -> str:
    """Return `capture.panel_dir` as the stages want it: relative to $FS.

    `stage_measure.sh capture` checks `[ -d "$FS/$PANEL_REL" ]`.  A panel
    handed to a container is normally a bind mount at an arbitrary path, so it
    is copied under the run root and the RELATIVE path is what job.json
    carries.  A panel already inside $FS is used where it lies.
    """
    src = Path(panel_dir).resolve()
    if not (src / "panel.json").is_file():
        raise Refusal(
            "--panel-dir %s has no panel.json" % src,
            ["A panel directory is panel.json + arrays/.",
             "Build one with engines/tools/build_token_panel.py, then bind-mount it."])
    try:
        return str(src.relative_to(fs_root.resolve()))
    except ValueError:
        pass
    dest = fs_root / "panel-src" / src.name
    if dest.exists():
        shutil.rmtree(str(dest))
    shutil.copytree(str(src), str(dest))
    con("panel staged under the run root: %s" % dest)
    return str(dest.relative_to(fs_root))


def write_token(fs_root: Path, token_file, con) -> bool:
    """The token as a 0600 file, never as argv and never in a log.

    Same contract the SSH controller transports: `stage_measure.sh load_token`
    reads `$FS/.secrets/hf_token`.  `HF_TOKEN` in the environment is accepted
    because that is how every container runtime passes a secret, but it is
    written to the file and the file is what the stages read.
    """
    token = ""
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    elif os.environ.get("HF_TOKEN"):
        token = os.environ["HF_TOKEN"].strip()
    if not token:
        return False
    # Exclusive, no-follow, 0600 from the first instant, inside a directory
    # that is 0700 before the file exists.  The run root is a persistent bind
    # mount, so a pre-planted symlink or a stale loose-mode file at this path
    # must be impossible to write through or inherit (peer review 2026-08-31,
    # "secret creation follows a pre-existing path").
    write_secret_file(str(fs_root / ".secrets" / "hf_token"), token)
    con("HF token installed  0600 file, never argv, removed when this run ends")
    return True


# --------------------------------------------------------------------------
# the job document
# --------------------------------------------------------------------------


def produced_by(suite: Path, build: dict, pin: dict) -> dict:
    """Name the code that produced the number -- from the image, not a promise.

    `fidelity.receipt.produced_by_block` refuses to emit this without a git
    revision, which is why the SSH path computes it on the caller's laptop and
    ships it in job.json.  An image HAS the revision: the build recorded it.
    So a containerised run fills the two container fields that have been null
    on every receipt this repo has ever sealed.
    """
    entry = "bin/container_entry.py"
    revision = build.get("suite_revision")
    if not revision:
        # Outside an image (a developer checkout, the selftest) there is still
        # a git tree to ask.  Inside one there is not, which is exactly why the
        # build bakes the answer.
        try:
            proc = subprocess.run(["git", "-C", str(suite), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=30)
            revision = (proc.stdout or "").strip() or None
        except Exception:                                  # noqa: BLE001
            revision = None
    if not revision:
        raise Refusal(
            "the image records no suite_revision, so this run cannot name the "
            "code that produced it",
            ["A receipt whose producing code cannot be named is not "
             "reproducible, and the schema requires the field.",
             "Rebuild the image from a git checkout (the Dockerfile records "
             "the revision into %s), or pass a complete --job document whose "
             "produced_by block was computed where the checkout lives."
             % (IMAGE_ROOT / BUILD_MANIFEST)])
    return {
        "tool": "quant-fidelity-suite/bin",
        "repository": "malaiwah/quant-fidelity-suite",
        "revision": revision,
        "entrypoint": entry,
        "entrypoint_sha256": sha256_file(str(suite / entry)),
        "runtime_reader_sha256": None,
        "container_image": pin.get("image_reference"),
        "container_digest": pin.get("image_digest") or pin.get("image_content_sha256"),
        "dependencies": {k: str(v) for k, v in
                         (build.get("pins") or {}).items() if v is not None},
    }


def job_document(args, suite: Path, fs_root: Path, con) -> dict:
    """The same contract `stage_measure.sh` already reads, built from flags.

    Deliberately NOT a second planner.  Everything the cloud planner resolves
    by talking to Hugging Face and the registry -- the panel's reference_ref
    and teacher digests, the sniffed surface, the engine profile -- is either
    passed explicitly or refused by name.  A container that guessed a profile
    would fail on the engine's argparse an hour into a rental, which is the
    exact failure this project already paid for once.
    """
    build = build_manifest()
    pin = image_pin(args.image_pin)
    role = "root" if args.verb == "capture" else "quant"

    panel = {}
    if getattr(args, "panel_descriptor", None):
        panel = json.loads(Path(args.panel_descriptor).read_text(encoding="utf-8"))
    elif getattr(args, "panel", None):
        panel = {"repo_id": args.panel, "revision": args.panel_revision,
                 "include": list(args.panel_include or ["*"])}

    capture = {}
    if role == "root":
        if not args.dataset_id:
            raise Refusal(
                "capture needs --dataset-id",
                ["A capture with no identity cannot be published or cited.",
                 "Example: --dataset-id fidelity--<model>.<author>.root.bf16"])
        if not args.panel_dir:
            raise Refusal(
                "capture needs --panel-dir",
                ["Bind-mount the panel directory and point --panel-dir at it."])
        panel_rel = stage_panel(args.panel_dir, fs_root, con)
        panel_id = json.loads(
            (Path(args.panel_dir) / "panel.json").read_text(encoding="utf-8")
        ).get("panel_id")
        capture = {
            "role": "root",
            "form": args.form,
            "schedule": args.schedule,
            "panel_dir": panel_rel,
            "panel_id": panel_id,
            # Present on both transports, filled only when the caller passes
            # --designated-reference through the cloud controller; the container
            # form does not take the flag yet (the plan-time release check that
            # justifies it lives in measure_cloud), so it is honestly None here
            # rather than absent -- C3c holds the two blocks to the same fields.
            "designated_reference": None,
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name or args.dataset_id,
            "author": args.measurer,
            "race": bool(args.race),
            "race_workers": int(args.race_workers),
            "preview_of": args.preview_of or None,
            "sanity_expect": args.sanity_expect,
            # Same field, same spelling, same default as the controller's --
            # C3c asserts the two capture blocks carry identical keys, because
            # a knob that exists on one transport and not the other is a run
            # that behaves differently depending on how it was launched.
            "allow_unexpected_tensors": bool(
                getattr(args, "allow_unexpected_tensors", False)),
            "device": getattr(args, "capture_device", None) or "cuda",
        }
    elif not getattr(args, "profile", None):
        raise Refusal(
            "measure needs --profile (the engine's own profile name)",
            ["The cloud planner resolves it from bin/engines.json's "
             "profile_map_by_surface for the sniffed surface and bit rate.",
             "In a container nothing sniffs: pass --profile explicitly, or "
             "pass the planner's own document with --job.",
             "Guessing here is not a smaller failure -- 'k6' is a real profile "
             "naming a real receipt family, so a wrong guess publishes a wrong "
             "label instead of crashing."])

    target = {"repo_id": args.model, "revision": args.revision}
    if getattr(args, "surface", None):
        target["surface"] = args.surface
    if getattr(args, "bits", None) is not None:
        target["bits"] = args.bits
    if getattr(args, "path", None):
        target["path"] = args.path

    doc = {
        "role": role,
        "capture": capture,
        # seal_receipt only reads this to pick a fallback runner name, and we
        # never take that fallback: produced_by is always supplied below.
        "recipe": "container",
        "job_id": args.job_id or ("job-%d" % int(time.time())),
        "lane": args.lane,
        "measurer": {
            "name": args.measurer, "handle": args.measurer,
            "url": "https://huggingface.co/%s" % args.measurer,
            "is_artifact_author": False,
        },
        "reduce_order": args.reduce_order,
        "cold_runs": args.cold_runs,
        # A root capture reads no engine profile: there is no quantized surface
        # to decode and no reference to diverge from.
        "profile": getattr(args, "profile", None),
        "target": target,
        "panel": panel,
        "reference": {
            "reference_ref": panel.get("reference_ref"),
            "teacher_receipt_sha256": panel.get("teacher_receipt_sha256"),
            "teacher_backend_identity_sha256":
                panel.get("teacher_backend_identity_sha256"),
        },
        "environment": {
            "gpu": args.gpu,
            "gpu_count": args.gpu_count,
            "tensor_parallel": 1,
            # The provider actually rented from, not a constant: a receipt that
            # names the wrong host is a false provenance claim in the one block
            # whose job is provenance.
            "host": args.host,
            # The two fields that were null on every receipt this repo sealed,
            # because nothing until now knew what image it was in.
            "container_image": pin.get("image_reference"),
            "container_digest": pin.get("image_digest"),
            "container_content_sha256": pin.get("image_content_sha256"),
        },
        "keep_student_logits": bool(args.keep_student_logits),
        "disclosures": [],
        "scope": (json.loads(Path(args.scope_json).read_text(encoding="utf-8"))
                  if args.scope_json else None),
        "produced_by": produced_by(suite, build, pin),
    }
    if args.official_bf16_revision:
        doc["official_bf16_revision"] = args.official_bf16_revision
    return doc


# --------------------------------------------------------------------------
# running the stages
# --------------------------------------------------------------------------


def stage_env(fs_root: Path, engine_root: Path, pin: dict) -> dict:
    """The roots the stage scripts read, made true instead of assumed.

    Both scripts default to a JarvisLabs path, which is correct there and
    silently wrong everywhere else: the run root, and then the pipeline root,
    each defaulting to `/home/jl_fs/...` that nothing exported is what left an
    A100 running at 0% GPU for two hours.  A container has no excuse for
    guessing -- the mount is an argument.
    """
    env = dict(os.environ)
    env["FIDELITY_FS_ROOT"] = str(fs_root)
    # FIDELITY_ENGINE_ROOT only. The pre-2026-08-31 spelling FIDELITY_K6_ROOT
    # is still ACCEPTED by the stage scripts as a fallback, and the SSH
    # controller still exports both for one release, because a controller and
    # an instance can come from different checkouts. This transport has no such
    # history: the image and the stage scripts inside it ship together, so
    # emitting the deprecated name here would bake a migration into new
    # surface for no compatibility anyone needs.
    env["FIDELITY_ENGINE_ROOT"] = str(engine_root)
    env.pop("FIDELITY_K6_ROOT", None)
    env["QP_PIPELINE_ROOT"] = str(engine_root / "pipeline")
    # Read by hf_capture (through the same convention stackprint uses) so the
    # capture's own runtime receipt records which image produced it.
    if pin.get("image_digest") or pin.get("image_content_sha256"):
        env[IMAGE_PIN_ENV] = str(pin.get("image_digest")
                                 or pin.get("image_content_sha256"))
    # The token must never be visible to a stage as an environment variable it
    # could echo; the 0600 file is the contract.  Drop it after it is written.
    env.pop("HF_TOKEN", None)
    return env


def run_stage(name: str, fs_root: Path, env: dict, con) -> int:
    script = fs_root / "bin" / "stage_measure.sh"
    if not script.is_file():
        raise Refusal(
            "%s is missing: the image did not bake bin/stage_measure.sh" % script,
            ["The entrypoint does not reimplement a stage; it runs that script."])
    con("stage %s starting" % name)
    started = time.time()
    proc = subprocess.Popen(["bash", str(script), name], env=env)
    code = proc.wait()
    con("stage %s %s  %.0fs" % (name, "ok" if code == 0 else "FAILED (%d)" % code,
                                time.time() - started))
    return code


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common(p) -> None:
    p.add_argument("--fs-root", default=os.environ.get("FIDELITY_FS_ROOT")
                   or DEFAULT_FS_ROOT,
                   help="the run root: models, panel, receipts, logs, job.json. "
                        "Bind-mount it (default %s)." % DEFAULT_FS_ROOT)
    p.add_argument("--engine-root",
                   default=os.environ.get("FIDELITY_ENGINE_ROOT") or str(IMAGE_ROOT),
                   help="where the baked venv and patched pipeline live. The "
                        "default is the image's own %s, and that is not the run "
                        "root on purpose: the venv and the patched pipeline are "
                        "immutable image content, while /workspace is a mount." % IMAGE_ROOT)
    p.add_argument("--job", help="use this job document verbatim instead of "
                                 "building one from the flags")
    p.add_argument("--token-file", help="0600 file holding the HF token "
                                        "(HF_TOKEN is also accepted)")
    p.add_argument("--image-pin", help="the registry digest of this image, as "
                                       "known to whoever pulled it")
    p.add_argument("--dry-run", action="store_true",
                   help="print the job document and the stage list; run nothing")
    p.add_argument("--only", action="append", default=[],
                   help="run only these stages (repeatable)")
    p.add_argument("--stop-after", help="run through this stage and stop")


def add_job_flags(p, *, root: bool) -> None:
    p.add_argument("--model", required=True, help="the checkpoint repo id")
    p.add_argument("--revision", help="40-hex commit; unpinned is a disclosure")
    p.add_argument("--lane", default="streaming")
    p.add_argument("--measurer", default="malaiwah")
    p.add_argument("--job-id")
    p.add_argument("--reduce-order", default="fp32")
    p.add_argument("--cold-runs", type=int, default=1)
    p.add_argument("--gpu", help="the GPU model, recorded in the receipt")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--host", default=os.environ.get("FIDELITY_HOST", "container"),
                   help="the provider actually rented from")
    p.add_argument("--official-bf16-revision")
    p.add_argument("--keep-student-logits", action="store_true")
    p.add_argument("--scope-json")
    if root:
        p.add_argument("--panel-dir", help="a panel directory (panel.json + arrays/)")
        p.add_argument("--dataset-id", help="the identity of the dataset this writes")
        p.add_argument("--dataset-name")
        p.add_argument("--form", default="hidden")
        p.add_argument("--schedule", default="layer-outer")
        p.add_argument("--race", action="store_true")
        p.add_argument("--race-workers", type=int, default=8)
        p.add_argument("--preview-of")
        p.add_argument("--sanity-expect", default="Paris")
        p.add_argument("--allow-unexpected-tensors", action="store_true")
        p.add_argument("--capture-device", default="cuda")
    else:
        p.add_argument("--surface")
        p.add_argument("--bits", type=float)
        p.add_argument("--path", help="subpath inside a multi-artifact repo")
        p.add_argument("--profile", help="the engine profile for this surface")
        p.add_argument("--panel", help="the panel repo id")
        p.add_argument("--panel-revision")
        p.add_argument("--panel-include", action="append", default=[])
        p.add_argument("--panel-descriptor",
                       help="a panel descriptor JSON (carries reference_ref and "
                            "the teacher digests the flags cannot)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fidelity", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    m = sub.add_parser("measure", help="measure a quantized artifact against a panel")
    add_common(m)
    add_job_flags(m, root=False)

    c = sub.add_parser("capture", help="capture a reference (root) dataset")
    add_common(c)
    add_job_flags(c, root=True)

    s = sub.add_parser("stage", help="run one named stage against an existing job.json")
    s.add_argument("name", choices=list(KNOWN_STAGES))
    add_common(s)

    sub.add_parser("doctor", help="print what this image is and what it can see")
    sub.add_parser("version", help="print the baked pins")
    return ap


def cmd_doctor(con) -> int:
    build = build_manifest()
    pin = image_pin(None)
    con("image reference        %s" % pin.get("image_reference"))
    con("image digest           %s" % (pin.get("image_digest") or "(undetected)"))
    con("image content sha256   %s" % pin.get("image_content_sha256"))
    con("suite revision         %s" % build.get("suite_revision"))
    con("built                  %s" % build.get("built_utc"))
    for key, value in sorted((build.get("pins") or {}).items()):
        con("  pin %-18s %s" % (key, value))
    engine = Path(os.environ.get("FIDELITY_ENGINE_ROOT") or str(IMAGE_ROOT))
    py = engine / "venv" / "bin" / "python"
    if not py.is_file():
        con("venv                   MISSING at %s" % py)
        return EXIT_FAILED
    probe = ("import torch, transformers, safetensors, numpy, hf_transfer;"
             "print('torch', torch.__version__, 'cuda', torch.version.cuda,"
             "'| transformers', transformers.__version__);"
             "print('cuda_available', torch.cuda.is_available(),"
             "'| device', torch.cuda.get_device_name(0)"
             " if torch.cuda.is_available() else None)")
    proc = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    con((proc.stdout or "").rstrip() or (proc.stderr or "").rstrip())
    return EXIT_OK if proc.returncode == 0 else EXIT_FAILED


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def con(text: str) -> None:
        sys.stdout.write("[%s] %s\n"
                         % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), text))
        sys.stdout.flush()

    if args.verb == "version":
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
        return EXIT_OK
    if args.verb == "doctor":
        return cmd_doctor(con)

    suite = Path(os.environ.get("FIDELITY_SUITE_ROOT")
                 or str(IMAGE_ROOT / "suite"))
    if not (suite / "bin" / "stage_measure.sh").is_file():
        # Outside an image (the selftest, a developer checkout) the suite is
        # simply this file's own parent.
        suite = Path(__file__).resolve().parent.parent
    fs_root = Path(args.fs_root)
    engine_root = Path(args.engine_root)

    try:
        fs_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EROFS):
            raise SystemExit(
                "cannot create the run root %s (%s).\n"
                "  Bind-mount a writable directory there, or pass --fs-root."
                % (fs_root, exc.strerror))
        raise

    try:
        copied = sync_suite(suite, fs_root, con)
        con("suite synced into the run root  %d file(s) changed" % copied)

        if args.job:
            doc = json.loads(Path(args.job).read_text(encoding="utf-8"))
        elif args.verb == "stage":
            existing = fs_root / "job.json"
            if not existing.is_file():
                raise Refusal(
                    "stage %s needs a job document and %s does not exist"
                    % (args.name, existing),
                    ["Pass --job, or run `measure`/`capture` which writes it."])
            doc = json.loads(existing.read_text(encoding="utf-8"))
        else:
            doc = job_document(args, suite, fs_root, con)

        pin = image_pin(args.image_pin)
        if args.verb == "stage":
            stages = [args.name]
        else:
            stages = stage_sequence(
                doc.get("role", "quant"),
                race=bool((doc.get("capture") or {}).get("race")),
                surface=(doc.get("target") or {}).get("surface"))
        if args.only:
            unknown = [s for s in args.only if s not in stages]
            if unknown:
                raise Refusal(
                    "--only names %s, which this job does not run" % ", ".join(unknown),
                    ["This job's stages: %s" % " ".join(stages)])
            stages = [s for s in stages if s in args.only]
        if args.stop_after:
            if args.stop_after not in stages:
                raise Refusal(
                    "--stop-after %s is not one of this job's stages" % args.stop_after,
                    ["This job's stages: %s" % " ".join(stages)])
            stages = stages[:stages.index(args.stop_after) + 1]

        if args.dry_run:
            print(json.dumps(doc, indent=2, sort_keys=True))
            con("stages: %s" % " ".join(stages))
            con("dry run: nothing was created, nothing was fetched")
            return EXIT_OK

        job_path = fs_root / "job.json"
        job_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        con("job.json written  %d bytes" % job_path.stat().st_size)
        wrote_token = write_token(fs_root, args.token_file, con)

        # The run root is a persistent bind mount: without the finally, the
        # token outlived the run on the host filesystem, forever, on success
        # AND on failure (peer review 2026-08-31).  Success, a failed stage,
        # an exception and ^C all pass through here.
        try:
            env = stage_env(fs_root, engine_root, pin)
            for name in stages:
                code = run_stage(name, fs_root, env, con)
                if code != 0:
                    con("run failed at stage %s" % name)
                    return EXIT_FAILED
            con("all stages complete; receipts under %s/receipts" % fs_root)
            return EXIT_OK
        finally:
            if wrote_token:
                shred_secret_file(str(fs_root / ".secrets" / "hf_token"))
                con("HF token shredded from the run root")
    except Refusal as exc:
        sys.stderr.write("REFUSED: %s\n" % exc)
        for line in exc.advice:
            sys.stderr.write("  %s\n" % line)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
