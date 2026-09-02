#!/usr/bin/env python3
"""Local/on-box image stage driver; it never creates or manages cloud resources.

This entrypoint turns explicit local inputs into the same finalized ``job.json``
read by ``bin/stage_measure.sh``, runs the sequence owned by
``fidelity.stages``, and emits the deterministic result archive owned by
``fidelity.resultsink``.  It supports CPU and CUDA execution in an already
running container.

It is not a RunPod API client or a paid-cloud orchestrator: it does not select
offers, create, recover, pause, or destroy pods.  The approved initial paid
RunPod route is the repository's SSH controller against one fresh on-demand
pod, not a provider-native container launch.

The HF token arrives at runtime as ``--token-file`` or ``HF_TOKEN``, is written
to the 0600 file the stages already read, and never reaches stage argv.

Stdlib only and Python 3.9-clean: this runs before any venv is on PATH.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import secrets
import subprocess
import sys
import time
import tarfile
import tempfile
from pathlib import PurePosixPath
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import panel as PANEL
from fidelity import resultsink as RS

from fidelity.common import shred_secret_file, write_secret_file  # noqa: E402
from fidelity.jobcontract import (  # noqa: E402
    finalize_bundle_manifest,
    finalize_job,
    parse_job_bytes,
    verify_bundle_manifest,
    verify_job,
)
from fidelity.engines import (  # noqa: E402
    EngineProfileRefused, EngineTimingUnavailable, RootTimingUnavailable,
    load_engines, require_supported_profile, resolve_profile_timing,
    resolve_root_timing,
)
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
DIRECT_ENTRY_FILES = (
    "bin/container_entry.py",
    "bin/fidelity/jobcontract.py",
    "bin/fidelity/panel.py",
    "bin/fidelity/resultsink.py",
    "bin/fidelity/stages.py",
)


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



def exact_bundle_manifest(suite: Path, build: dict) -> dict:
    """Bind the exact regular suite bytes used by this local container."""
    built_files = build.get("bundle_sha256") if isinstance(build, dict) else None
    source = "BUILD.json"
    if isinstance(built_files, dict) and built_files:
        candidate_map = dict(built_files)
    else:
        source = "BUNDLE.txt"
        candidate_map = {rel: None for rel in bundle_entries(suite)}
    # An older BUILD.json may predate a new direct import while the running
    # entrypoint necessarily contains it. Bind those bytes explicitly rather
    # than trusting an incomplete historical subset.
    for rel in DIRECT_ENTRY_FILES:
        candidate_map.setdefault(rel, None)
    candidates = sorted(candidate_map.items())
    rows = []
    for rel, built_digest in candidates:
        pure = PurePosixPath(rel) if isinstance(rel, str) else PurePosixPath("")
        if (not rel or pure.is_absolute() or ".." in pure.parts
                or "\\" in rel or pure.as_posix() != rel):
            raise Refusal("bundle manifest has an unsafe path: %r" % rel)
        path = suite / rel
        if path.is_symlink():
            raise Refusal("bundle entry is a symlink: %s" % rel)
        if not path.is_file():
            raise Refusal("bundle manifest entry is missing: %s" % rel)
        observed_digest = sha256_file(str(path))
        if built_digest is not None and observed_digest != _hex_sha256(
                built_digest, "BUILD.json bundle file SHA-256"):
            raise Refusal("BUILD.json bundle digest differs on disk: %s" % rel)
        rows.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": observed_digest,
        })
    try:
        return finalize_bundle_manifest(rows, source)
    except ValueError as exc:
        raise Refusal("cannot finalize exact bundle manifest: %s" % exc)

def _local_contract_manifests(suite: Path, bundle: dict):
    registry_path = suite / "bin" / "BUNDLE.txt"
    if registry_path.is_symlink() or not registry_path.is_file():
        raise Refusal("bin/BUNDLE.txt must be a regular exact suite file")
    registry = {
        "path": "bin/BUNDLE.txt",
        "bytes": registry_path.stat().st_size,
        "sha256": sha256_file(str(registry_path)),
    }
    control_paths = list(DIRECT_ENTRY_FILES) + ["bin/stage_measure.sh"]
    rows = []
    for rel in sorted(set(control_paths)):
        path = suite / rel
        if path.is_symlink() or not path.is_file():
            raise Refusal("local control-plane file is missing: %s" % rel)
        rows.append({
            "path": rel, "bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        })
    control = finalize_bundle_manifest(rows, "local-container-control")
    control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    binding_raw = json.dumps(
        {"bundle": bundle, "registry": registry},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return registry, control, hashlib.sha256(binding_raw).hexdigest()


def prepare_fs_root(fs_root: Path) -> None:
    """Create a run root only when no existing path component is a symlink."""
    cursor = Path(fs_root.anchor) if fs_root.is_absolute() else Path(".")
    parts = fs_root.parts[1:] if fs_root.is_absolute() else fs_root.parts
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Refusal("--fs-root crosses a symlink: %s" % cursor)
        if cursor.exists() and not cursor.is_dir():
            raise Refusal("--fs-root crosses a non-directory: %s" % cursor)
    fs_root.mkdir(parents=True, exist_ok=True)
    if fs_root.is_symlink() or not fs_root.is_dir():
        raise Refusal("--fs-root must be a regular, non-symlink directory")


def require_fresh_job_root(fs_root: Path) -> None:
    """New measure/capture attempts never adopt outputs from an older run."""
    entries = []
    for entry in fs_root.iterdir():
        if (entry.name == ".secrets" and not entry.is_symlink()
                and entry.is_dir() and not any(entry.iterdir())):
            continue
        entries.append(entry.name)
    if entries:
        raise Refusal(
            "measure/capture requires a fresh empty --fs-root",
            ["Found pre-existing entries: %s" % ", ".join(sorted(entries)[:8]),
             "Only explicit `stage --job` may resume an existing run root."])




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
    if fs_root.is_symlink() or not fs_root.is_dir():
        raise Refusal("--fs-root must be a regular, non-symlink directory")
    if suite.resolve() == fs_root.resolve():
        con("suite already at the run root; nothing to sync")
        return 0
    copied = 0
    wanted = list(bundle_entries(suite))
    for extra in DIRECT_ENTRY_FILES:
        if extra not in wanted:
            wanted.append(extra)
    for rel in wanted:
        src = suite / rel
        if src.is_symlink():
            raise Refusal("bundle source is a symlink: %s" % rel)
        if not src.is_file():
            con("bundle entry not baked into the image, skipped: %s" % rel)
            continue
        dst = _safe_sync_destination(fs_root, rel)
        source_sha = sha256_file(str(src))
        if dst.is_file() and sha256_file(str(dst)) == source_sha:
            continue
        body = src.read_bytes()
        if hashlib.sha256(body).hexdigest() != source_sha:
            raise Refusal("bundle source changed while copying: %s" % rel)
        _write_atomic(dst, body)
        os.chmod(str(dst), src.stat().st_mode & 0o777)
        copied += 1
    return copied


def _safe_sync_destination(fs_root: Path, rel: str) -> Path:
    root = fs_root
    if root.is_symlink() or not root.is_dir():
        raise Refusal("--fs-root must be a regular, non-symlink directory")
    pure = PurePosixPath(rel)
    if (pure.is_absolute() or ".." in pure.parts or "\\" in rel
            or pure.as_posix() != rel):
        raise Refusal("bundle destination path is unsafe: %r" % rel)
    cursor = root
    for part in pure.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Refusal("bundle destination ancestor is a symlink: %s" % cursor)
        if cursor.exists():
            if not cursor.is_dir():
                raise Refusal(
                    "bundle destination ancestor is not a directory: %s" % cursor)
        else:
            cursor.mkdir(mode=0o755)
    destination = cursor / pure.name
    if destination.is_symlink():
        raise Refusal("bundle destination is a symlink: %s" % destination)
    if destination.exists() and not destination.is_file():
        raise Refusal("bundle destination is not a regular file: %s" % destination)
    return destination


def _hex_sha256(value, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise Refusal("%s must be 64 lowercase hexadecimal characters" % label)
    return value


def _relative_input(path, fs_root: Path, label: str, *, directory: bool):
    """Resolve one job path beneath fs_root without accepting a symlink."""
    if not isinstance(path, str) or not path:
        raise Refusal("%s is required" % label)
    rel = PurePosixPath(path)
    if rel.is_absolute() or ".." in rel.parts or "\\" in path:
        raise Refusal("%s must be relative and traversal-free" % label)
    root = fs_root.resolve()
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Refusal("%s crosses a symlink beneath --fs-root" % label)
    target = root.joinpath(*rel.parts)
    try:
        target.resolve().relative_to(root)
    except ValueError:
        raise Refusal("%s resolves outside --fs-root" % label)
    if target.is_symlink():
        raise Refusal("%s must not be a symlink" % label)
    if directory and not target.is_dir():
        raise Refusal("%s is not a directory beneath --fs-root" % label)
    if not directory and not target.is_file():
        raise Refusal("%s is not a regular file beneath --fs-root" % label)
    return target


def _write_atomic(path: Path, body: bytes) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise Refusal("atomic-write parent must be a regular directory: %s"
                      % path.parent)
    if path.is_symlink():
        raise Refusal("atomic-write destination must not be a symlink: %s" % path)
    if path.exists() and not path.is_file():
        raise Refusal("atomic-write destination must be a regular file: %s" % path)
    fd, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _extract_panel_archive(archive_path: Path, destination: Path) -> None:
    """Extract the panel archive as regular files only, then publish atomically."""
    staging = Path(tempfile.mkdtemp(
        prefix=".panel-extract-", dir=str(destination.parent)))
    try:
        with tarfile.open(str(archive_path), "r:") as archive:
            seen = set()
            for member in archive.getmembers():
                rel = PurePosixPath(member.name)
                if (not member.isfile() or rel.is_absolute() or ".." in rel.parts
                        or "\\" in member.name or member.name in seen):
                    raise Refusal(
                        "validated panel archive contains an unsafe member: %r"
                        % member.name)
                seen.add(member.name)
                output = staging.joinpath(*rel.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise Refusal("cannot read panel archive member %s" % member.name)
                with output.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
        os.replace(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def stage_panel(panel_dir, fs_root: Path, con, *, binding_file=None,
                binding_sha256=None, tokenizer_root=None) -> dict:
    """Validate, archive and safely stage one exact panel and its binding."""
    if not panel_dir:
        raise Refusal("capture needs --panel-dir")
    if not binding_file or not binding_sha256:
        raise Refusal(
            "root capture needs --panel-binding and --panel-binding-sha256",
            ["Pass the resolved binding file and its exact raw-file SHA-256."])
    expected_binding_sha = _hex_sha256(
        binding_sha256, "--panel-binding-sha256")
    supplied_path = Path(binding_file)
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise Refusal("--panel-binding must name a regular, non-symlink file")
    binding_raw = supplied_path.read_bytes()
    if hashlib.sha256(binding_raw).hexdigest() != expected_binding_sha:
        raise Refusal("--panel-binding-sha256 does not match --panel-binding")
    try:
        supplied_binding = json.loads(binding_raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise Refusal("--panel-binding is not valid UTF-8 JSON")
    if not isinstance(supplied_binding, dict):
        raise Refusal("--panel-binding must contain a JSON object")

    src = Path(panel_dir)
    if src.is_symlink() or not src.is_dir():
        raise Refusal("--panel-dir must name a regular, non-symlink directory")
    fs_root = fs_root.resolve()
    inputs = fs_root / "inputs"
    if inputs.is_symlink():
        raise Refusal("the run root inputs path must not be a symlink")
    if inputs.exists() and not inputs.is_dir():
        raise Refusal("the run root inputs path must be a directory")
    inputs.mkdir(parents=True, exist_ok=True)
    validation_dir = Path(tempfile.mkdtemp(
        prefix=".panel-validation-", dir=str(inputs)))
    archive_path = validation_dir / "panel.tar"
    try:
        try:
            written = PANEL.write_panel_archive(
                src, archive_path, tokenizer_root=tokenizer_root)
        except (OSError, PANEL.PanelError, ValueError) as exc:
            raise Refusal("panel validation failed: %s" % exc)
        tokenizer = written["binding"].get("tokenizer") or {}
        if tokenizer.get("files_verified") is not True:
            raise Refusal(
                "resolved panel tokenizer files are not locally verified",
                ["Pass --panel-tokenizer-root containing the exact tokenizer "
                 "artifacts named by the panel binding."])
        if written["binding"] != supplied_binding:
            raise Refusal(
                "--panel-binding does not describe the exact --panel-dir tree")
        destination = inputs / "panel"
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise Refusal("staged panel destination is not a regular directory")
            observed = PANEL.resolve_panel(
                destination, tokenizer_root=tokenizer_root).to_dict()
            if observed != supplied_binding:
                raise Refusal(
                    "the run root already contains a different staged panel")
        else:
            _extract_panel_archive(archive_path, destination)
            observed = PANEL.resolve_panel(
                destination, tokenizer_root=tokenizer_root).to_dict()
            if observed != supplied_binding:
                raise Refusal("extracted panel does not match its resolved binding")
    finally:
        shutil.rmtree(str(validation_dir), ignore_errors=True)

    binding_destination = inputs / "panel.binding.json"
    if binding_destination.exists():
        if (binding_destination.is_symlink()
                or binding_destination.read_bytes() != binding_raw):
            raise Refusal(
                "the run root already contains a different panel binding")
    else:
        _write_atomic(binding_destination, binding_raw)
    con("validated panel staged under the run root: %s" % destination)
    return {
        "panel_dir": destination.relative_to(fs_root).as_posix(),
        "resolved_binding": supplied_binding,
        "binding_path": binding_destination.relative_to(fs_root).as_posix(),
        "binding_file_sha256": expected_binding_sha,
    }


def stage_allowlist(path, artifact_sha256, names_sha256,
                    fs_root: Path) -> dict:
    """Validate and copy the optional exact unexpected-tensor set."""
    values = (path, artifact_sha256, names_sha256)
    if not any(value is not None for value in values):
        return {}
    if not all(value is not None for value in values):
        raise Refusal(
            "unexpected-tensor allowlist file and both SHA-256 values are all-or-none")
    raw_sha = _hex_sha256(
        artifact_sha256, "--unexpected-tensors-allowlist-sha256")
    canonical_sha = _hex_sha256(
        names_sha256, "--unexpected-tensors-name-sha256")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise Refusal(
            "--unexpected-tensors-allowlist must name a regular, non-symlink file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha:
        raise Refusal("unexpected-tensor allowlist artifact SHA-256 mismatch")
    try:
        names = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise Refusal("unexpected-tensor allowlist is not valid UTF-8 JSON")
    if (not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise Refusal(
            "unexpected-tensor allowlist must be a duplicate-free JSON string array")
    canonical = json.dumps(
        sorted(names), separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != canonical_sha:
        raise Refusal("unexpected-tensor canonical sorted-name SHA-256 mismatch")
    inputs = fs_root.resolve() / "inputs"
    if inputs.is_symlink():
        raise Refusal("the run root inputs path must not be a symlink")
    if inputs.exists() and not inputs.is_dir():
        raise Refusal("the run root inputs path must be a directory")
    inputs.mkdir(parents=True, exist_ok=True)
    destination = inputs / "unexpected-tensors.json"
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != raw:
            raise Refusal(
                "the run root already contains a different unexpected-tensor allowlist")
    else:
        _write_atomic(destination, raw)
    return {
        "path": destination.relative_to(fs_root.resolve()).as_posix(),
        "artifact_sha256": raw_sha,
        "canonical_sorted_names_sha256": canonical_sha,
    }


def clear_stale_token(fs_root: Path, con) -> None:
    """Remove the sole stage token path without following planted links."""
    secret_dir = fs_root / ".secrets"
    if secret_dir.is_symlink():
        secret_dir.unlink()
        con("removed stale symlink at the secret directory")
        return
    if secret_dir.exists() and not secret_dir.is_dir():
        secret_dir.unlink()
        con("removed stale non-directory at the secret path")
        return
    token_path = secret_dir / "hf_token"
    if token_path.is_dir() and not token_path.is_symlink():
        shutil.rmtree(str(token_path))
        con("removed stale directory at the HF token path")
        return
    if token_path.exists() or token_path.is_symlink():
        shred_secret_file(str(token_path))
        con("removed stale HF token before this run")


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
    clear_stale_token(fs_root, con)
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


def produced_by(suite: Path, build: dict, pin: dict,
                dependencies: dict) -> dict:
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
    dependency_block = {
        k: str(v) for k, v in (build.get("pins") or {}).items()
        if v is not None
    }
    dependency_block.update({
        key: str(value) for key, value in dependencies.items()
    })
    return {
        "tool": "quant-fidelity-suite/bin",
        "repository": "malaiwah/quant-fidelity-suite",
        "revision": revision,
        "entrypoint": entry,
        "entrypoint_sha256": sha256_file(str(suite / entry)),
        "runtime_reader_sha256": None,
        "container_image": pin.get("image_reference"),
        "container_digest": pin.get("image_digest") or pin.get("image_content_sha256"),
        "dependencies": dependency_block,
    }


def _load_scope(path, *, required: bool):
    if not path:
        if required:
            raise Refusal("measure needs --scope-json with an exact scope object")
        return None
    try:
        scope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise Refusal("cannot read --scope-json: %s" % exc)
    if not isinstance(scope, dict):
        raise Refusal("--scope-json must contain a JSON object")
    return scope


def _load_target_descriptor(path, model: str, revision: str) -> dict:
    if not path:
        raise Refusal(
            "new jobs need --target-descriptor with exact artifact identity")
    descriptor_path = Path(path)
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise Refusal("--target-descriptor must be a regular, non-symlink file")
    try:
        target = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise Refusal("cannot read --target-descriptor: %s" % exc)
    if not isinstance(target, dict):
        raise Refusal("--target-descriptor must contain a JSON object")
    if target.get("repo_id") != model or target.get("revision") != revision:
        raise Refusal(
            "--target-descriptor repo_id/revision disagree with --model/--revision")
    required_strings = (
        "repo_id", "revision", "requested_revision", "surface", "codec",
        "config_sha256", "index_sha256", "shard_manifest_sha256")
    for field in required_strings:
        if not isinstance(target.get(field), str) or not target.get(field):
            raise Refusal("target descriptor %s is required" % field)
    for field in ("config_sha256", "index_sha256", "shard_manifest_sha256"):
        _hex_sha256(target[field], "target descriptor %s" % field)
    bits = target.get("bits")
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise Refusal("target descriptor bits must be numeric")
    if ("path" not in target
            or (target["path"] is not None
                and (not isinstance(target["path"], str) or not target["path"]))):
        raise Refusal("target descriptor path must be a string or explicit null")
    model_bytes = target.get("model_bytes")
    if (not isinstance(model_bytes, int) or isinstance(model_bytes, bool)
            or model_bytes <= 0):
        raise Refusal("target descriptor model_bytes must be a positive integer")
    shards = target.get("shards")
    if not isinstance(shards, list) or not shards:
        raise Refusal("target descriptor shards must be a nonempty array")
    total = 0
    canonical_shards = []
    seen = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {"path", "bytes"}:
            raise Refusal(
                "target descriptor shard rows must be exact path/bytes objects")
        filename = shard.get("path")
        pure = (PurePosixPath(filename) if isinstance(filename, str)
                else PurePosixPath(""))
        if (not filename or pure.is_absolute() or ".." in pure.parts
                or "\\" in filename or filename in seen):
            raise Refusal("target descriptor has an unsafe/duplicate shard path")
        seen.add(filename)
        size = shard.get("bytes")
        if (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            raise Refusal("target descriptor shard bytes must be positive integers")
        total += size
        canonical_shards.append({"path": filename, "bytes": size})
    canonical_shards.sort(key=lambda row: row["path"])
    if shards != canonical_shards:
        raise Refusal("target descriptor shards must be sorted by path")
    canonical_raw = json.dumps(
        canonical_shards, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    if hashlib.sha256(canonical_raw).hexdigest() != target[
            "shard_manifest_sha256"]:
        raise Refusal("target descriptor shard_manifest_sha256 mismatch")
    if total != model_bytes:
        raise Refusal("target descriptor model_bytes disagrees with shard bytes")
    return target




def _resolve_profile_and_timing(args, target: dict, role: str):
    try:
        if role == "root":
            timing = resolve_root_timing(
                target_repo=target["repo_id"],
                target_revision=target["revision"],
                gpu=args.gpu,
                form=args.form,
                schedule="two-fresh-process-qualification")
            model_identity = timing.get("model_identity") or {}
            for field in ("model_bytes", "config_sha256", "index_sha256"):
                if model_identity.get(field) != target.get(field):
                    raise Refusal(
                        "root timing evidence differs from target %s" % field)
            profile = {
                "profile_id": "root-hf-transformers-bf16",
                "lane": "root",
                "source": "native",
                "surface": target["surface"],
                "form": args.form,
                "engine": "hf-transformers",
                "compute_dtype": "bfloat16",
                "device": "cuda",
                "schedule": "two-fresh-process-qualification",
            }
            return profile, timing
        engines = load_engines()
        engine = engines.get(args.lane)
        if engine is None:
            raise Refusal("unknown engine lane: %s" % args.lane)
        require_supported_profile(
            engine, surface=target["surface"], bits=target["bits"])
        profile_id = args.profile
        timing = resolve_profile_timing(
            engine, profile=profile_id, surface=target["surface"],
            bits=target["bits"], target_repo=target["repo_id"],
            target_revision=target["revision"], gpu=args.gpu)
        source = {
            "tr3-6bpw": "tr3",
            "native-bf16": "native",
        }.get(profile_id)
        if source is None:
            raise Refusal(
                "local initial quant path permits tr3-6bpw or native-bf16")
        profile = {
            "profile_id": profile_id,
            "lane": args.lane,
            "source": source,
            "surface": target["surface"],
            "bits": target["bits"],
        }
        return profile, timing
    except Refusal:
        raise
    except (EngineProfileRefused, EngineTimingUnavailable,
            RootTimingUnavailable, KeyError, TypeError, ValueError) as exc:
        raise Refusal("engine profile/timing is unavailable: %s" % exc)


def job_document(args, suite: Path, fs_root: Path, con) -> dict:
    """Build and canonically finalize the complete local stage contract."""
    build = build_manifest()
    pin = image_pin(args.image_pin)
    role = "root" if args.verb == "capture" else "quant"
    revision = getattr(args, "revision", None)
    if (not isinstance(revision, str) or len(revision) != 40
            or any(ch not in "0123456789abcdef" for ch in revision)):
        raise Refusal("--revision must be an exact lowercase 40-hex commit")
    target = _load_target_descriptor(
        getattr(args, "target_descriptor", None), args.model, revision)
    profile, timing = _resolve_profile_and_timing(args, target, role)

    panel = {}
    capture = {}
    if role == "root":
        if getattr(args, "race", False) or getattr(args, "preview_of", None):
            raise Refusal(
                "race/preview root capture is unsupported",
                ["This local driver does not create or manage RunPod resources.",
                 "The initial paid RunPod route is one SSH controller against "
                 "one fresh on-demand pod."])
        if args.cold_runs != 2:
            raise Refusal(
                "root capture requires --cold-runs 2",
                ["The root protocol is exactly two fresh capture processes, "
                 "two verifies, exact self-comparison, and qualification."])
        if (getattr(args, "replay_device", None) != "numpy"
                or getattr(args, "replay_dtype", None) != "float32"
                or getattr(args, "vocab_chunk", None) != 8192):
            raise Refusal(
                "root capture needs --replay-device numpy, "
                "--replay-dtype float32, and --vocab-chunk 8192")
        if not args.dataset_id:
            raise Refusal("capture needs --dataset-id")
        if not getattr(args, "dataset_repository", None):
            raise Refusal(
                "capture needs --dataset-repository",
                ["This is the immutable repository identity written into both "
                 "fresh captures even when no outward publication is requested."])
        if getattr(args, "publish_root_to", None) is not None:
            raise Refusal(
                "--publish-root-to is unsupported by this local container",
                ["Root capture stops after qualification. Remote publication "
                 "is outside the initial safe execution path."])
        staged = stage_panel(
            args.panel_dir, fs_root, con,
            binding_file=getattr(args, "panel_binding", None),
            binding_sha256=getattr(args, "panel_binding_sha256", None),
            tokenizer_root=getattr(args, "panel_tokenizer_root", None))
        panel = {
            "resolved_binding": staged["resolved_binding"],
            "binding_path": staged["binding_path"],
            "binding_file_sha256": staged["binding_file_sha256"],
        }
        allowlist = stage_allowlist(
            getattr(args, "unexpected_tensors_allowlist", None),
            getattr(args, "unexpected_tensors_allowlist_sha256", None),
            getattr(args, "unexpected_tensors_name_sha256", None),
            fs_root)
        if not allowlist:
            raise Refusal(
                "initial root capture requires the exact unexpected-tensor "
                "allowlist artifact and both digests")
        capture = {
            "role": "root",
            "form": args.form,
            "replay": {
                "device": args.replay_device,
                "dtype": args.replay_dtype,
                "vocab_chunk": args.vocab_chunk,
            },
            "root_protocol": {
                "schedule": "two-fresh-process-qualification",
                "fresh_processes": 2,
                "run_count_per_process": 1,
                "exact_self_comparison": True,
                "qualification_required": True,
                "canonical_publication_required":
                    bool(getattr(args, "publish_root_to", None)),
                "publication_mode": (
                    "canonical-public"
                    if getattr(args, "publish_root_to", None)
                    else "qualified-unpublished"),
            },
            "schedule": args.schedule,
            "panel_dir": staged["panel_dir"],
            "panel_id": staged["resolved_binding"]["panel"]["id"],
            "designated_reference": None,
            "dataset_id": args.dataset_id,
            "dataset_repository": args.dataset_repository,
            "dataset_name": args.dataset_name or args.dataset_id,
            "author": args.measurer,
            "race": False,
            "preview_of": None,
            "sanity_expect": args.sanity_expect,
            "publish_root_to": args.publish_root_to,
            "dataset_license": "mit",
            "weights_license": None,
            "engine": "hf-transformers",
            "dtype": "bfloat16",
            "device": "cuda",
            "replay_device": args.replay_device,
            "replay_dtype": args.replay_dtype,
            "vocab_chunk": args.vocab_chunk,
        }
        if allowlist:
            capture["unexpected_tensor_allowlist"] = allowlist
    else:
        descriptor_path = getattr(args, "panel_descriptor", None)
        if not descriptor_path:
            raise Refusal(
                "measure needs --panel-descriptor",
                ["A quant result must bind the exact panel token/receipt and "
                 "teacher backend receipts before any stage runs."])
        try:
            panel = json.loads(
                Path(descriptor_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise Refusal("cannot read --panel-descriptor: %s" % exc)
        if not isinstance(panel, dict):
            raise Refusal("--panel-descriptor must contain a JSON object")
        if not getattr(args, "profile", None):
            raise Refusal(
                "measure needs --profile (the engine's own profile name)",
                ["Resolve it from bin/engines.json for the exact surface and "
                 "bit rate; this local driver never guesses."])

    if role == "quant":
        runtime_profile = timing.get("runtime_profile") or {}
        if runtime_profile.get("window_count") != panel.get("contexts"):
            raise Refusal(
                "timing window_count differs from panel contexts")
        expected_cache = (
            "none" if profile["profile_id"] == "tr3-6bpw" else "ram")
        if (runtime_profile.get("decode_cache") != expected_cache
                or runtime_profile.get("decode_threads") != 28
                or runtime_profile.get("reader_threads") != 28):
            raise Refusal(
                "timing evidence lacks exact cache/thread runtime identity")
        runtime = {
            "min_vcpu": 28,
            "min_ram_gb": 300,
            "decode_cache": expected_cache,
            "decode_threads": 28,
            "reader_threads": 28,
            "device": "cuda",
            "expert_parallel": False,
            "reduce_order": "fp32",
        }
    else:
        runtime = {
            "min_vcpu": 8,
            "min_ram_gb": 64,
            "device": "cuda",
            "reduce_order": "fp32",
        }
    # The exact target census is supplied before fetch and is the same shape
    # used by the SSH controller. No stage may infer or backfill identity.
    bundle = exact_bundle_manifest(suite, build)
    registry, control, bundle_contract_sha256 = _local_contract_manifests(
        suite, bundle)
    produced = produced_by(
        suite, build, pin, {
            "profile": profile["profile_id"],
            "lane": args.lane,
            "provider": "local-container",
        })
    doc = {
        "schema": "fidelity-suite/job.v2",
        "role": role,
        "capture": capture,
        "recipe": "local-container",
        "execution_attempt": {
            "number": 1, "kind": "local-container",
            "attempt_id": secrets.token_hex(12),
        },
        "bundle": bundle,
        "bundle_registry": registry,
        "bundle_contract_sha256": bundle_contract_sha256,
        "control_plane": control,
        "lane": args.lane,
        "measurer": {
            "name": args.measurer, "handle": args.measurer,
            "url": "https://huggingface.co/%s" % args.measurer,
            "is_artifact_author": False,
        },
        "reduce_order": args.reduce_order,
        "cold_runs": args.cold_runs,
        "profile": profile,
        "timing": timing,
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
            "host": args.host,
            "execution_mode": "local-container",
            "container_image": pin.get("image_reference"),
            "container_digest": pin.get("image_digest"),
            "container_content_sha256": pin.get("image_content_sha256"),
        },
        "runtime": runtime,
        "keep_student_logits": bool(args.keep_student_logits),
        "resource_requirements": {
            "workspace_available_bytes_minimum":
                args.workspace_available_bytes_minimum,
            "container_available_bytes_minimum":
                args.container_available_bytes_minimum,
            "min_vcpu_count": runtime["min_vcpu"],
            "min_memory_gb": runtime["min_ram_gb"],
            "expected_vram_bytes": args.expected_vram_bytes,
        },
        "disclosures": [],
        "scope": (_load_scope(args.scope_json, required=True)
                  if role == "quant" else {
                      "kind": "root-capture",
                      "engine": "hf-transformers",
                      "dtype": "bfloat16",
                      "form": args.form,
                  }),
        "produced_by": produced,
    }
    if role == "quant":
        doc["scoring"] = {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda",
            "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        }
    if args.official_bf16_revision:
        doc["official_bf16_revision"] = args.official_bf16_revision
    try:
        finalized = finalize_job(doc)
        validate_job_document(finalized, fs_root)
        return finalized
    except Refusal:
        raise
    except (TypeError, ValueError) as exc:
        raise Refusal("job document cannot be finalized: %s" % exc)


def _validate_bound_panel_tree(panel_dir: Path, binding: dict) -> None:
    manifest = ((binding.get("content") or {}).get("manifest")
                if isinstance(binding, dict) else None)
    if not isinstance(manifest, list) or not manifest:
        raise Refusal("panel.resolved_binding has no content manifest")
    expected = {}
    for row in manifest:
        if (not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256"}):
            raise Refusal("panel resolved-binding manifest is malformed")
        rel = row["path"]
        pure = PurePosixPath(rel) if isinstance(rel, str) else PurePosixPath("")
        if (not rel or pure.is_absolute() or ".." in pure.parts
                or "\\" in rel or rel in expected):
            raise Refusal("panel resolved-binding manifest has an unsafe path")
        size = row["bytes"]
        if (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise Refusal("panel resolved-binding manifest has invalid bytes")
        expected[rel] = (
            row["bytes"], _hex_sha256(row["sha256"], "panel manifest SHA-256"))
    observed = {}
    for path in panel_dir.rglob("*"):
        if path.is_symlink():
            raise Refusal("staged panel contains a symlink: %s" % path)
        if path.is_file():
            rel = path.relative_to(panel_dir).as_posix()
            observed[rel] = (path.stat().st_size, sha256_file(str(path)))
        elif not path.is_dir():
            raise Refusal("staged panel contains a non-regular entry: %s" % path)
    if observed != expected:
        raise Refusal(
            "staged panel tree does not exactly match panel.resolved_binding")


def validate_job_document(doc: dict, fs_root: Path, expected_bundle=None) -> str:
    """Fail closed on identity and every local stage prerequisite."""
    try:
        identity = verify_job(doc)
    except (TypeError, ValueError) as exc:
        raise Refusal("job.json identity is invalid: %s" % exc)
    if doc.get("schema") != "fidelity-suite/job.v2":
        raise Refusal("job.json schema must be fidelity-suite/job.v2")
    attempt = doc.get("execution_attempt")
    if not isinstance(attempt, dict):
        raise Refusal("job.json execution_attempt must be an object")
    attempt_kind = attempt.get("kind")
    if attempt_kind == "local-container":
        if (set(attempt) != {"number", "kind", "attempt_id"}
                or attempt.get("number") != 1
                or not isinstance(attempt.get("attempt_id"), str)
                or len(attempt["attempt_id"]) != 24
                or any(ch not in "0123456789abcdef"
                       for ch in attempt["attempt_id"])):
            raise Refusal("local-container execution_attempt is not exact")
    elif attempt_kind == "runpod-ssh":
        required_attempt = {
            "attempt_id", "kind", "lease_path", "workload_deadline_utc",
            "provider_terminate_after", "planned_at"}
        if set(attempt) != required_attempt:
            raise Refusal("runpod-ssh execution_attempt has unknown/missing fields")
    else:
        raise Refusal("job.json execution_attempt.kind is unsupported")
    bundle = doc.get("bundle")
    try:
        verify_bundle_manifest(bundle)
    except (TypeError, ValueError) as exc:
        raise Refusal("job.json bundle manifest is invalid: %s" % exc)
    if expected_bundle is not None and bundle != expected_bundle:
        raise Refusal("job.json bundle differs from this container's exact suite bytes")
    role = doc.get("role")
    if role not in ("quant", "root"):
        raise Refusal("job.json role must be quant or root")
    target = doc.get("target")
    panel = doc.get("panel")
    if not isinstance(target, dict):
        raise Refusal("job.json target must be an object")
    if not isinstance(target.get("repo_id"), str) or not target.get("repo_id"):
        raise Refusal("job.json target.repo_id is required")
    revision = target.get("revision")
    if (not isinstance(revision, str) or len(revision) != 40
            or any(ch not in "0123456789abcdef" for ch in revision)):
        raise Refusal("job.json target.revision must be exact lowercase 40-hex")
    if not isinstance(panel, dict):
        raise Refusal("job.json panel must be an object")
    if role == "quant":
        if not isinstance(doc.get("profile"), dict) or not doc.get("profile"):
            raise Refusal("quant job profile object is required")
        if not isinstance(doc.get("timing"), dict) or not doc.get("timing"):
            raise Refusal("quant job timing object is required")
        expected_scoring = {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda",
            "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        }
        if doc.get("scoring") != expected_scoring:
            raise Refusal("quant job scoring policy is not exact")
        if not isinstance(target.get("surface"), str) or not target.get("surface"):
            raise Refusal("quant job target.surface is required")
        if ("path" not in target
                or (target["path"] is not None
                    and (not isinstance(target["path"], str)
                         or not target["path"]))):
            raise Refusal("quant job target.path must be a string or explicit null")
        if not isinstance(target.get("codec"), str) or not target.get("codec"):
            raise Refusal("quant job target.codec is required")
        bits = target.get("bits")
        if isinstance(bits, bool) or not isinstance(bits, (int, float)):
            raise Refusal("quant job target.bits must be numeric")
        _hex_sha256(target.get("config_sha256"), "quant target.config_sha256")
        _hex_sha256(target.get("index_sha256"), "quant target.index_sha256")
        if (not isinstance(panel.get("repo_id"), str) or not panel.get("repo_id")
                or not isinstance(panel.get("revision"), str)
                or len(panel.get("revision")) != 40
                or any(ch not in "0123456789abcdef"
                       for ch in panel.get("revision"))):
            raise Refusal(
                "quant job panel must pin repo_id and lowercase 40-hex revision")
        if not isinstance(panel.get("panel_ref"), str) or not panel.get("panel_ref"):
            raise Refusal("quant job panel.panel_ref is required")
        _hex_sha256(panel.get("panel_token_sha256"),
                    "quant panel.panel_token_sha256")
        _hex_sha256(panel.get("panel_receipt_sha256"),
                    "quant panel.panel_receipt_sha256")
        for field in ("contexts", "scored_positions"):
            value = panel.get(field)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value <= 0):
                raise Refusal("quant job panel.%s must be a positive integer"
                              % field)
        if panel.get("roles") != "final":
            raise Refusal("quant job panel.roles must be final")
        reference = doc.get("reference")
        if not isinstance(reference, dict):
            raise Refusal("quant job reference must be an object")
        if (not isinstance(reference.get("reference_ref"), str)
                or not reference.get("reference_ref")):
            raise Refusal("quant job reference.reference_ref is required")
        _hex_sha256(reference.get("teacher_receipt_sha256"),
                    "quant reference.teacher_receipt_sha256")
        _hex_sha256(reference.get("teacher_backend_identity_sha256"),
                    "quant reference.teacher_backend_identity_sha256")
        if not isinstance(doc.get("scope"), dict):
            raise Refusal("quant job scope must be an exact object")
        if not isinstance(doc.get("lane"), str) or not doc.get("lane"):
            raise Refusal("quant job lane is required")
        measurer = doc.get("measurer")
        if (not isinstance(measurer, dict)
                or not isinstance(measurer.get("name"), str)
                or not measurer.get("name")
                or not isinstance(measurer.get("handle"), str)
                or not measurer.get("handle")):
            raise Refusal("quant job measurer name and handle are required")
        produced = doc.get("produced_by")
        for field in ("tool", "repository", "revision", "entrypoint",
                      "entrypoint_sha256"):
            if (not isinstance(produced, dict)
                    or not isinstance(produced.get(field), str)
                    or not produced.get(field)):
                raise Refusal("quant job produced_by.%s is required" % field)
        _hex_sha256(produced.get("entrypoint_sha256"),
                    "quant produced_by.entrypoint_sha256")
        if (not isinstance(doc.get("cold_runs"), int)
                or isinstance(doc.get("cold_runs"), bool)
                or doc.get("cold_runs") < 2):
            raise Refusal("quant job cold_runs must be at least 2")
        return identity

    capture = doc.get("capture")
    if not isinstance(capture, dict):
        raise Refusal("root job needs a capture object")
    if "allow_unexpected_tensors" in capture:
        raise Refusal(
            "capture.allow_unexpected_tensors is obsolete; broad acceptance refuses")
    required_keys = {
        "role", "form", "schedule", "panel_dir", "panel_id",
        "designated_reference", "dataset_id", "dataset_name",
        "dataset_repository", "dataset_license", "weights_license",
        "author", "race", "preview_of", "sanity_expect", "publish_root_to",
        "engine", "dtype", "device", "replay_device", "replay_dtype",
        "vocab_chunk", "replay", "root_protocol"}
    absent = sorted(required_keys - set(capture))
    if absent:
        raise Refusal(
            "root job is missing capture fields: %s" % ", ".join(absent))
    if capture.get("role") != "root":
        raise Refusal("root job capture.role must be root")
    if capture.get("race") is not False or capture.get("preview_of") is not None:
        raise Refusal(
            "race/preview root capture is unsupported",
            ["This local driver does not create or manage RunPod resources.",
             "Use the SSH controller for the initial paid RunPod route."])
    if doc.get("cold_runs") != 2:
        raise Refusal(
            "root job cold_runs must be exactly 2 for fresh-process qualification")
    value_required = (
        "form", "schedule", "panel_dir", "panel_id", "dataset_id",
        "dataset_name", "dataset_repository", "author", "engine", "dtype",
        "device", "replay_device", "replay_dtype", "vocab_chunk")
    missing = [key for key in value_required
               if capture.get(key) is None or capture.get(key) == ""]
    if missing:
        raise Refusal(
            "root job has empty exact capture fields: %s" % ", ".join(missing))
    if (capture["engine"], capture["dtype"], capture["device"]) != (
            "hf-transformers", "bfloat16", "cuda"):
        raise Refusal(
            "root capture engine/dtype/device must be "
            "hf-transformers/bfloat16/cuda")
    if capture["replay_device"] != "numpy":
        raise Refusal("capture.replay_device must be numpy")
    if capture["replay_dtype"] != "float32":
        raise Refusal("capture.replay_dtype must be float32")
    if capture["vocab_chunk"] != 8192:
        raise Refusal("capture.vocab_chunk must be exactly 8192")
    if capture["replay"] != {
            "device": capture["replay_device"],
            "dtype": capture["replay_dtype"],
            "vocab_chunk": capture["vocab_chunk"]}:
        raise Refusal("capture.replay disagrees with explicit replay fields")
    expected_protocol = {
        "schedule": "two-fresh-process-qualification",
        "fresh_processes": 2,
        "run_count_per_process": 1,
        "exact_self_comparison": True,
        "qualification_required": True,
        "canonical_publication_required": bool(capture["publish_root_to"]),
        "publication_mode": (
            "canonical-public" if capture["publish_root_to"]
            else "qualified-unpublished"),
    }
    if capture["root_protocol"] != expected_protocol:
        raise Refusal("capture.root_protocol is not exact two-process qualification")
    target_repo = (doc.get("target") or {}).get("repo_id")
    target_revision = (doc.get("target") or {}).get("revision")
    dataset_repository = capture["dataset_repository"]
    destination = capture["publish_root_to"]
    if destination is not None:
        raise Refusal(
            "root publication is unsupported by this local container",
            ["A local root ends as qualified-unpublished; no publish stage "
             "or remote mutation is available."])
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    for label, repository in (
            ("capture.dataset_repository", dataset_repository),
            ("capture.publish_root_to", destination)):
        if repository is None and label == "capture.publish_root_to":
            continue
        parts = repository.split("/") if isinstance(repository, str) else []
        if (len(parts) != 2 or not all(parts)
                or any(any(ch not in allowed for ch in part) for part in parts)):
            raise Refusal("%s must be an owner/name repository id" % label)
    if dataset_repository == target_repo:
        raise Refusal("target weights and intended dataset repository must differ")
    if not isinstance(target_repo, str) or not target_repo:
        raise Refusal("root job target.repo_id is required")
    if not isinstance(target_revision, str) or not target_revision:
        raise Refusal("root job target.revision is required")
    dataset_license = capture.get("dataset_license")
    weights_license = capture.get("weights_license")
    if dataset_license not in ("mit", "other"):
        raise Refusal("capture.dataset_license must be mit or other")
    if dataset_license == "mit":
        if weights_license is not None:
            raise Refusal(
                "capture.weights_license requires capture.dataset_license=other")
    else:
        if (not isinstance(weights_license, dict)
                or set(weights_license) != {
                    "source_path", "dataset_path", "bytes", "sha256"}
                or weights_license.get("source_path") != "LICENSE"
                or weights_license.get("dataset_path") != "LICENSE"
                or isinstance(weights_license.get("bytes"), bool)
                or not isinstance(weights_license.get("bytes"), int)
                or not 0 < weights_license["bytes"] <= 1024 * 1024):
            raise Refusal(
                "capture.weights_license is not an exact bounded LICENSE identity")
        _hex_sha256(
            weights_license.get("sha256"), "capture.weights_license.sha256")
    if (doc.get("target") or {}).get("weights_license") != weights_license:
        raise Refusal(
            "capture.weights_license differs from target weights-license identity")

    binding = panel.get("resolved_binding")
    binding_path = panel.get("binding_path")
    binding_sha = _hex_sha256(
        panel.get("binding_file_sha256"), "panel.binding_file_sha256")
    if not isinstance(binding, dict):
        raise Refusal("root job panel.resolved_binding must be an object")
    if binding.get("schema") != PANEL.RESOLVED_SCHEMA:
        raise Refusal("panel.resolved_binding has an unsupported schema")
    tokenizer = binding.get("tokenizer") or {}
    if tokenizer.get("files_verified") is not True:
        raise Refusal(
            "panel.resolved_binding tokenizer files_verified must be true")
    binding_file = _relative_input(
        binding_path, fs_root, "panel.binding_path", directory=False)
    binding_raw = binding_file.read_bytes()
    if hashlib.sha256(binding_raw).hexdigest() != binding_sha:
        raise Refusal("panel binding raw-file SHA-256 mismatch")
    try:
        on_disk_binding = json.loads(binding_raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise Refusal("panel binding file is not valid UTF-8 JSON")
    if on_disk_binding != binding:
        raise Refusal("panel.resolved_binding differs from panel.binding_path")
    panel_dir = _relative_input(
        capture["panel_dir"], fs_root, "capture.panel_dir", directory=True)
    _validate_bound_panel_tree(panel_dir, binding)
    if capture["panel_id"] != (binding.get("panel") or {}).get("id"):
        raise Refusal("capture.panel_id differs from panel.resolved_binding")

    allowlist = capture.get("unexpected_tensor_allowlist")
    if allowlist is not None:
        if (not isinstance(allowlist, dict)
                or set(allowlist) != {
                    "path", "artifact_sha256",
                    "canonical_sorted_names_sha256"}):
            raise Refusal(
                "capture.unexpected_tensor_allowlist needs exactly path and both SHA-256 identities")
        allow_path = _relative_input(
            allowlist["path"], fs_root,
            "capture.unexpected_tensor_allowlist.path", directory=False)
        raw = allow_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != _hex_sha256(
                allowlist["artifact_sha256"],
                "unexpected_tensor_allowlist.artifact_sha256"):
            raise Refusal("unexpected-tensor allowlist artifact SHA-256 mismatch")
        try:
            names = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise Refusal("unexpected-tensor allowlist is not valid UTF-8 JSON")
        if (not isinstance(names, list)
                or any(not isinstance(name, str) or not name for name in names)
                or len(names) != len(set(names))):
            raise Refusal(
                "unexpected-tensor allowlist must be a duplicate-free JSON string array")
        canonical = json.dumps(
            sorted(names), separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != _hex_sha256(
                allowlist["canonical_sorted_names_sha256"],
                "unexpected_tensor_allowlist.canonical_sorted_names_sha256"):
            raise Refusal("unexpected-tensor canonical sorted-name SHA-256 mismatch")
    return identity


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


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a positive integer")
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
    p.add_argument("--job", help="use this already-finalized job document "
                                 "instead of building one from flags; identity "
                                 "and local root prerequisites are still verified")
    p.add_argument("--token-file", help="0600 file holding the HF token "
                                        "(HF_TOKEN is also accepted)")
    p.add_argument("--image-pin", help="the registry digest of this image, as "
                                       "known to whoever pulled it")
    p.add_argument("--result-sink", action="append", default=[], metavar="URI",
                   help="where the deterministic result archive goes: "
                        "file:PATH or https://URL (PUT), repeatable. stdout is "
                        "always emitted. Every explicitly requested sink must "
                        "succeed for a completed run.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the job document and the stage list; run nothing")
    p.add_argument("--only", action="append", default=[],
                   help="run only these stages (repeatable)")
    p.add_argument("--stop-after", help="run through this stage and stop")


def add_job_flags(p, *, root: bool) -> None:
    p.add_argument("--model", required=True, help="the checkpoint repo id")
    p.add_argument("--revision", help="exact checkpoint revision")
    p.add_argument(
        "--target-descriptor", required=True,
        help="exact pre-fetched target census JSON; must bind config, index, "
             "model bytes, sorted shards and immutable revision")
    p.add_argument("--lane", default="streaming")
    p.add_argument("--measurer", default="malaiwah")
    p.add_argument("--reduce-order", default="fp32")
    p.add_argument("--cold-runs", type=int, default=2)
    p.add_argument("--gpu", required=True,
                   help="exact observed GPU model used for profile/timing admission")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--host", default=os.environ.get("FIDELITY_HOST", "local"),
                   help="local execution-host provenance label; this flag does "
                        "not select or create cloud resources")
    p.add_argument("--official-bf16-revision")
    p.add_argument("--keep-student-logits", action="store_true")
    p.add_argument(
        "--workspace-available-bytes-minimum", type=_positive_int,
        required=True,
        help="precomputed minimum free bytes on the run workspace filesystem")
    p.add_argument(
        "--container-available-bytes-minimum", type=_positive_int,
        required=True,
        help="precomputed minimum free bytes on the container filesystem")
    p.add_argument(
        "--expected-vram-bytes", type=_positive_int, required=True,
        help="exact minimum GPU VRAM bytes bound by the local plan")
    p.add_argument("--scope-json")
    if root:
        p.add_argument("--panel-dir",
                       help="validated panel source tree (panel.json, receipt, arrays)")
        p.add_argument("--panel-binding",
                       help="resolved panel binding JSON for the exact source tree")
        p.add_argument("--panel-binding-sha256",
                       help="SHA-256 of the exact --panel-binding file bytes")
        p.add_argument("--panel-tokenizer-root",
                       help="required exact local tokenizer tree used to verify "
                            "the resolved panel binding before stages")
        p.add_argument("--dataset-id", help="the identity of the dataset this writes")
        p.add_argument("--dataset-name")
        p.add_argument("--form", default="hidden")
        p.add_argument("--schedule", default="layer-outer")
        p.add_argument("--race", action="store_true",
                       help="always refused: race roots are outside this local driver")
        p.add_argument("--preview-of",
                       help="always refused: preview roots are outside this local driver")
        p.add_argument("--sanity-expect", default="Paris")
        p.add_argument("--unexpected-tensors-allowlist",
                       help="exact JSON string-array artifact; optional, but "
                            "requires both digest flags")
        p.add_argument("--unexpected-tensors-allowlist-sha256",
                       help="SHA-256 of exact allowlist artifact bytes")
        p.add_argument("--unexpected-tensors-name-sha256",
                       help="SHA-256 of canonical sorted allowlist names")
        p.add_argument("--replay-device", choices=("numpy",),
                       help="required exact qualification replay backend")
        p.add_argument("--replay-dtype", choices=("float32",),
                       help="required exact qualification replay dtype")
        p.add_argument("--vocab-chunk", type=int,
                       help="explicit positive qualification vocabulary chunk")
        p.add_argument(
            "--dataset-repository", metavar="HF_DATASET_REPO",
            help="required immutable owner/name repository identity recorded "
                 "by both captures; this does not publish")
        p.add_argument(
            "--publish-root-to", metavar="HF_DATASET_REPO",
            help="always refused: local roots end qualified-unpublished and "
                 "this driver exposes no remote publication stage")
    else:
        p.add_argument("--profile", required=True,
                       help="the engine profile for target descriptor surface/bits")
        p.add_argument("--panel-descriptor", required=True,
                       help="exact panel/reference descriptor JSON, including "
                            "panel token/receipt and teacher identity digests")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fidelity", description=__doc__.splitlines()[0],
        epilog=("LOCAL DRIVER ONLY: this command does not select offers or "
                "create/manage RunPod resources. The initial paid RunPod route "
                "is the SSH controller against one fresh on-demand pod; native "
                "container launch is not an approved paid route."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    m = sub.add_parser(
        "measure", help="measure locally against a panel",
        description=("Run quant stages inside this already-running local/on-box "
                     "container; no cloud resources are created or managed."))
    add_common(m)
    add_job_flags(m, root=False)

    c = sub.add_parser(
        "capture", help="capture and qualify a local reference root",
        description=("Run the exact two-process root protocol inside this "
                     "already-running container. This is not the paid RunPod "
                     "orchestrator; the approved paid path uses the SSH controller."))
    add_common(c)
    add_job_flags(c, root=True)

    s = sub.add_parser(
        "stage", help="run one local stage against existing job.json",
        description=("Run one stage on-box; this command never creates, "
                     "recovers, pauses, or destroys a pod."))
    s.add_argument("name", choices=list(KNOWN_STAGES))
    add_common(s)

    d = sub.add_parser(
        "doctor", help="inspect this already-running local/on-box image")
    add_common(d)
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
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    proc = subprocess.run(
        [str(py), "-c", probe], capture_output=True, text=True, env=env)
    con((proc.stdout or "").rstrip() or (proc.stderr or "").rstrip())
    return EXIT_OK if proc.returncode == 0 else EXIT_FAILED


def require_accelerator(doc, engine_root: Path, con) -> None:
    """Refuse an unavailable requested CUDA device before fetching artifacts."""
    device = str((doc.get("capture") or {}).get("device")
                 or (doc.get("environment") or {}).get("device")
                 or "cuda").lower()
    if device != "cuda":
        return
    py = engine_root / "venv" / "bin" / "python"
    if not py.is_file():
        return                      # no venv yet; the bootstrap speaks first
    probe = ("import json, torch;"
             "ok = torch.cuda.is_available();"
             "print(json.dumps({'ok': ok, 'torch': torch.__version__,"
             " 'built': torch.version.cuda,"
             " 'name': torch.cuda.get_device_name(0) if ok else None}))")
    proc = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    try:
        info = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:
        raise Refusal(
            "could not determine whether this container has a usable CUDA device",
            ["torch probe exited %d" % proc.returncode,
             (proc.stderr or "").strip()[-300:] or "(no stderr)",
             "This local job asks for CUDA; no stage or artifact fetch started."])
    if info.get("ok"):
        con("accelerator              ok  %s (torch %s, built for CUDA %s)"
            % (info.get("name"), info.get("torch"), info.get("built")))
        return
    detail = (proc.stderr or "").strip().splitlines()
    why = next((ln for ln in detail if "driver" in ln.lower()), "")
    raise Refusal(
        "this container has no usable CUDA device, and the job asks for one",
        [why[:300] or "torch.cuda.is_available() is False",
         "torch %s is built for CUDA %s."
         % (info.get("torch"), info.get("built")),
         "Nothing was fetched. Choose a compatible already-running host/image.",
         "This driver does not create or replace RunPod resources.",
         "Initial root capture is fixed to CUDA; CPU capture is unsupported."])


def _record_container_failure(fs_root: Path, failed_stage: str) -> None:
    """Guarantee the failed-result archive has a local diagnostic log."""
    path = _safe_sync_destination(fs_root, "logs/container-entry.log")
    existing = path.read_bytes() if path.exists() else b""
    line = ("container stage driver failed at %s\n"
            % failed_stage).encode("utf-8")
    _write_atomic(path, existing + line)


def _prevalidate_stage_job(args, fs_root: Path, suite: Path) -> dict:
    """Authenticate a resume job against this suite before mutating its root."""
    existing_path = fs_root / "job.json"
    supplied = Path(args.job) if args.job else existing_path
    if supplied.is_symlink() or not supplied.is_file():
        raise Refusal(
            "stage %s needs a regular existing job document" % args.name,
            ["Pass --job, or restore %s exactly." % existing_path])
    try:
        raw = supplied.read_bytes()
    except OSError as exc:
        raise Refusal("cannot read stage job document: %s" % exc)
    if args.job and (existing_path.exists() or existing_path.is_symlink()):
        if existing_path.is_symlink() or not existing_path.is_file():
            raise Refusal("existing run-root job.json is not a regular file")
        try:
            existing_raw = existing_path.read_bytes()
        except OSError as exc:
            raise Refusal("cannot read existing run-root job.json: %s" % exc)
        if raw != existing_raw:
            raise Refusal(
                "supplied --job bytes differ from existing run-root job.json",
                ["A resume never replaces or adopts another attempt's job."])
    try:
        document = parse_job_bytes(raw)
    except (TypeError, ValueError) as exc:
        raise Refusal("stage job document is invalid: %s" % exc)
    validate_job_document(
        document, fs_root,
        expected_bundle=exact_bundle_manifest(suite, build_manifest()))
    return document


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
        lines = []

        def tee(text):
            lines.append(text)
            con(text)

        requested_sink = bool(
            getattr(args, "result_sink", [])
            or os.environ.get("FIDELITY_RESULT_SINK"))
        try:
            sinks = RS.parse_sinks(getattr(args, "result_sink", []))
            requested_nonstdout = [
                sink for sink in sinks if sink.scheme != "stdout"]
            fs_root = Path(getattr(args, "fs_root", DEFAULT_FS_ROOT))
            prepare_fs_root(fs_root)
            clear_stale_token(fs_root, tee)
        except Exception as exc:
            con("doctor preparation refused: %s" % exc.__class__.__name__)
            return EXIT_FAILED if requested_sink else EXIT_REFUSED
        code = cmd_doctor(tee)
        try:
            doctor_path = _safe_sync_destination(
                fs_root, "receipts/doctor.json")
            _write_atomic(
                doctor_path,
                (json.dumps({
                    "schema": "malaiwah.fidelity-doctor.v1",
                    "status": "ok" if code == EXIT_OK else "failed",
                    "report": lines,
                }, indent=2) + "\n").encode("utf-8"))
            summary = RS.build_summary(
                fs_root, "doctor", "ok" if code == EXIT_OK else "failed",
                [], image_pin(getattr(args, "image_pin", None)))
            deliveries = RS.deliver(fs_root, sinks, summary, con)
            delivered_nonstdout = [
                result for result in deliveries
                if result.get("scheme") != "stdout"]
            if (len(delivered_nonstdout) != len(requested_nonstdout)
                    or any(not result.get("ok")
                           for result in delivered_nonstdout)):
                con("doctor requested result delivery failed")
                code = EXIT_FAILED
        except Exception as exc:
            con("doctor report not delivered: %s" % exc.__class__.__name__)
            if requested_sink:
                code = EXIT_FAILED
        return code

    suite = Path(os.environ.get("FIDELITY_SUITE_ROOT")
                 or str(IMAGE_ROOT / "suite"))
    if not (suite / "bin" / "stage_measure.sh").is_file():
        # Outside an image (the selftest, a developer checkout) the suite is
        # simply this file's own parent.
        suite = Path(__file__).resolve().parent.parent
    fs_root = Path(args.fs_root)
    engine_root = Path(args.engine_root)

    try:
        prepare_fs_root(fs_root)
    except Refusal as exc:
        sys.stderr.write("REFUSED: %s\n" % exc)
        return EXIT_REFUSED
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EROFS):
            raise SystemExit(
                "cannot create the run root %s (%s).\n"
                "  Bind-mount a writable directory there, or pass --fs-root."
                % (fs_root, exc.strerror))
        raise

    try:
        doc = (_prevalidate_stage_job(args, fs_root, suite)
               if args.verb == "stage" else None)
        clear_stale_token(fs_root, con)
        if args.verb in ("measure", "capture"):
            if args.job:
                raise Refusal(
                    "measure/capture does not resume --job",
                    ["Use explicit `stage NAME --job FILE` against its existing "
                     "run root; new measure/capture always creates a fresh attempt."])
            require_fresh_job_root(fs_root)
        copied = sync_suite(suite, fs_root, con)
        con("suite synced into the run root  %d file(s) changed" % copied)

        if args.verb != "stage":
            doc = job_document(args, suite, fs_root, con)
        validate_job_document(
            doc, fs_root,
            expected_bundle=exact_bundle_manifest(suite, build_manifest()))
        if (doc.get("role") == "root" and args.verb != "stage"
                and (args.only or args.stop_after)):
            raise Refusal(
                "root capture cannot use --only or --stop-after",
                ["The root protocol always runs two fresh captures, both "
                 "verifications, exact self-comparison and qualification, "
                 "plus canonical publication only when explicitly requested."])

        pin = image_pin(args.image_pin)
        # Parse every sink before the first stage. A typo must not turn a
        # completed local run into evidence stranded inside its container.
        try:
            sinks = RS.parse_sinks(getattr(args, "result_sink", []))
        except RS.SinkError as exc:
            raise Refusal(str(exc), ["Known schemes: file:PATH, https://URL."])
        if args.verb == "stage":
            stages = [args.name]
        else:
            try:
                stages = stage_sequence(
                    doc.get("role", "quant"),
                    race=bool((doc.get("capture") or {}).get("race")),
                    surface=(doc.get("target") or {}).get("surface"),
                    publish_root=bool((doc.get("capture") or {})
                                      .get("publish_root_to")))
            except ValueError as exc:
                raise Refusal(str(exc))
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
            con("dry run: no stage ran and nothing was fetched or published")
            return EXIT_OK

        # BEFORE the stage list, so a dead box costs nothing.
        require_accelerator(doc, engine_root, con)

        job_path = fs_root / "job.json"
        if args.verb == "stage":
            if not job_path.exists():
                _write_atomic(job_path, Path(args.job).read_bytes())
                con("job.json installed from the prevalidated exact bytes")
            else:
                con("job.json verified unchanged for explicit stage resume")
        else:
            _write_atomic(
                job_path,
                (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            con("job.json written  %d bytes" % job_path.stat().st_size)
        try:
            write_token(fs_root, args.token_file, con)
        except BaseException:
            clear_stale_token(fs_root, con)
            raise

        # The run root is a persistent bind mount. Success, a failed stage, an
        # exception and ^C all shred the token before any result leaves the box.
        failed = None
        delivery_failed = False
        try:
            env = stage_env(fs_root, engine_root, pin)
            for name in stages:
                code = run_stage(name, fs_root, env, con)
                if code != 0:
                    con("run failed at stage %s" % name)
                    failed = name
                    _record_container_failure(fs_root, failed)
                    break
        except BaseException:
            failed = failed or "exception"
            _record_container_failure(fs_root, failed)
            raise
        finally:
            clear_stale_token(fs_root, con)
            try:
                summary = RS.build_summary(
                    fs_root, args.verb, "failed" if failed else "ok",
                    stages, pin, failed)
                deliveries = RS.deliver(fs_root, sinks, summary, con)
                requested_nonstdout = [
                    sink for sink in sinks if sink.scheme != "stdout"]
                delivered_nonstdout = [
                    result for result in deliveries
                    if result.get("scheme") != "stdout"]
                unsuccessful_requested = [
                    result for result in delivered_nonstdout
                    if not result.get("ok")]
                if (len(delivered_nonstdout) != len(requested_nonstdout)
                        or unsuccessful_requested):
                    delivery_failed = True
                    con("one or more explicitly requested result sinks failed")
            except Exception as exc:
                delivery_failed = True
                con("result archive/delivery failed: %s" % exc.__class__.__name__)
        if failed:
            return EXIT_FAILED
        if delivery_failed:
            con("run stages completed, but verified result delivery did not")
            return EXIT_FAILED
        con("all stages and requested result deliveries complete; receipts under "
            "%s/receipts" % fs_root)
        return EXIT_OK
    except Refusal as exc:
        sys.stderr.write("REFUSED: %s\n" % exc)
        for line in exc.advice:
            sys.stderr.write("  %s\n" % line)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
