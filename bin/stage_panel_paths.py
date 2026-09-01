#!/usr/bin/env python3
"""Safely stage the one pinned token panel at its legacy producer paths.

  stage_panel_paths.py --panel <dir> [--receipt <path>] [--check-only]

Only the small token-panel artifacts are copied.  Teacher logits are resolved
portably by the scorer and are deliberately outside this tool's authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fidelity import panel as panel_contract


LEGACY_PREFIX = PurePosixPath("/workspace/artifacts/dataset/calibration/panel-v1")
DESTINATION_PREFIX = Path(str(LEGACY_PREFIX))
DESTINATION_ANCHOR = Path("/workspace")
PINNED_RECEIPT_SHA256 = (
    "0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5")
PINNED_PANEL_SHA256 = (
    "6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff")
PINNED_ARTIFACT_COUNT = 667
PINNED_FINAL_WINDOWS = 25
PINNED_FINAL_PREDICTION_POSITIONS = 51175
MAX_RECEIPT_BYTES = 8 << 20
RECEIPT_KEYS = {
    "schema", "receipt_sha256", "token_panel_artifact_sha256", "artifacts",
    "corpus_receipt_sha256", "domains", "final_windows",
    "final_prediction_positions", "roles", "tokenizer_receipt_sha256",
}
ROW_KEYS = {"path", "bytes", "sha256"}
PINNED_DOMAINS = (
    "axis1_general", "axis2_legal", "axis3_code_agentic",
    "axis4_reasoning_termination",
)
PINNED_ROLES = ("fit", "conditional-fit", "selection", "confirmation", "final")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class StageError(ValueError):
    """The requested staging operation is unsafe or cannot verify."""


class ReceiptError(StageError):
    """The token-panel receipt is malformed, unsealed, or not the pinned one."""


@dataclass(frozen=True)
class Artifact:
    path: str
    relative_parts: Tuple[str, ...]
    size: int
    digest: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key %r" % key)
        result[key] = value
    return result


def _read_fd(fd: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _hash_fd(fd: int) -> Tuple[int, str]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return size, digest.hexdigest()


def _path_parts(path: os.PathLike) -> Tuple[str, ...]:
    absolute = os.path.abspath(os.fspath(path))
    return tuple(part for part in absolute.split("/") if part)


def _open_directory(path: os.PathLike, label: str) -> int:
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in _path_parts(path):
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except OSError as exc:
        os.close(fd)
        raise StageError("%s is not a symlink-safe real directory: %s" %
                         (label, exc)) from exc


def _open_regular_at(base_fd: int, parts: Sequence[str], label: str) -> int:
    if not parts:
        raise StageError("%s has no file component" % label)
    parent = os.dup(base_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=parent)
            os.close(parent)
            parent = next_fd
        fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise StageError("%s is not a regular file" % label)
        return fd
    except OSError as exc:
        raise StageError("%s is not a symlink-safe regular file: %s" %
                         (label, exc)) from exc
    finally:
        os.close(parent)


def _open_regular(path: os.PathLike, label: str) -> int:
    parts = _path_parts(path)
    if not parts:
        raise StageError("%s has no file component" % label)
    parent = _open_directory("/" + "/".join(parts[:-1]), "%s parent" % label)
    try:
        return _open_regular_at(parent, (parts[-1],), label)
    finally:
        os.close(parent)


def _load_strict_json(path: os.PathLike) -> Mapping[str, Any]:
    try:
        fd = _open_regular(path, "token-panel receipt")
        try:
            info = os.fstat(fd)
            if info.st_size > MAX_RECEIPT_BYTES:
                raise ReceiptError("token-panel receipt exceeds the safe size limit")
            raw = _read_fd(fd)
        finally:
            os.close(fd)
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON token %s" % token)))
    except (OSError, UnicodeDecodeError, ValueError, StageError) as exc:
        raise ReceiptError("cannot read strict token-panel receipt %s: %s" %
                           (path, exc)) from exc
    if not isinstance(value, dict):
        raise ReceiptError("token-panel receipt must be a JSON object")
    return value


def _exact_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptError("%s must be an exact nonnegative integer" % label)
    return value


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ReceiptError("%s must be a lowercase SHA-256 hex string" % label)
    return value


def _artifact_path(value: Any, index: int) -> Tuple[str, Tuple[str, ...]]:
    label = "receipt.artifacts[%d].path" % index
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ReceiptError("%s must be a canonical absolute POSIX path" % label)
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value:
        raise ReceiptError("%s is not canonical: %r" % (label, value))
    try:
        relative = path.relative_to(LEGACY_PREFIX)
    except ValueError as exc:
        raise ReceiptError("%s is outside %s" % (label, LEGACY_PREFIX)) from exc
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ReceiptError("%s does not name a file beneath %s" %
                           (label, LEGACY_PREFIX))
    canonical = str(LEGACY_PREFIX) + "/" + "/".join(parts)
    if canonical != value:
        raise ReceiptError("%s is not canonical: %r" % (label, value))
    return value, tuple(parts)


def validate_receipt(
    receipt_path: os.PathLike,
    *,
    expected_receipt_sha256: str = PINNED_RECEIPT_SHA256,
    expected_panel_sha256: str = PINNED_PANEL_SHA256,
    expected_artifact_count: Optional[int] = PINNED_ARTIFACT_COUNT,
    expected_final_windows: int = PINNED_FINAL_WINDOWS,
    expected_final_prediction_positions: int = PINNED_FINAL_PREDICTION_POSITIONS,
) -> Tuple[Artifact, ...]:
    """Parse and fully validate the legacy sealed artifact receipt."""
    receipt = _load_strict_json(receipt_path)
    if set(receipt) != RECEIPT_KEYS:
        raise ReceiptError("token-panel receipt must contain exactly %s" %
                           sorted(RECEIPT_KEYS))
    if receipt.get("schema") != panel_contract.ARTIFACT_RECEIPT_SCHEMA:
        raise ReceiptError("unsupported token-panel receipt schema %r" %
                           receipt.get("schema"))
    claimed = _hex(receipt.get("receipt_sha256"), "receipt.receipt_sha256")
    if claimed != expected_receipt_sha256:
        raise ReceiptError("receipt is not the pinned Brandon final25 panel")
    body = dict(receipt)
    del body["receipt_sha256"]
    computed = _sha256_bytes(panel_contract._canonical(body, newline=True))
    if computed != claimed:
        raise ReceiptError(
            "receipt seal does not reproduce legacy field-absent canonical JSON-with-newline")
    panel_digest = _hex(receipt.get("token_panel_artifact_sha256"),
                        "receipt.token_panel_artifact_sha256")
    if panel_digest != expected_panel_sha256:
        raise ReceiptError("receipt binds the wrong token-panel artifact")
    _hex(receipt.get("tokenizer_receipt_sha256"),
         "receipt.tokenizer_receipt_sha256")
    _hex(receipt.get("corpus_receipt_sha256"),
         "receipt.corpus_receipt_sha256")
    domains = receipt.get("domains")
    if not isinstance(domains, list) or tuple(domains) != PINNED_DOMAINS:
        raise ReceiptError("receipt.domains is not the exact pinned domain list")
    roles = receipt.get("roles")
    if not isinstance(roles, list) or tuple(roles) != PINNED_ROLES:
        raise ReceiptError("receipt.roles is not the exact pinned role list")
    if _exact_integer(receipt.get("final_windows"), "receipt.final_windows") \
            != expected_final_windows:
        raise ReceiptError("receipt.final_windows is not the pinned panel count")
    if _exact_integer(receipt.get("final_prediction_positions"),
                      "receipt.final_prediction_positions") \
            != expected_final_prediction_positions:
        raise ReceiptError(
            "receipt.final_prediction_positions is not the pinned panel count")
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ReceiptError("receipt.artifacts must be a non-empty array")
    if expected_artifact_count is not None and len(rows) != expected_artifact_count:
        raise ReceiptError("receipt.artifacts has %d rows, expected %d" %
                           (len(rows), expected_artifact_count))
    artifacts: List[Artifact] = []
    paths = set()
    digests: Dict[str, str] = {}
    panel_rows = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ReceiptError(
                "receipt.artifacts[%d] must contain exactly path, bytes, sha256" % index)
        logical_path, relative_parts = _artifact_path(row["path"], index)
        size = _exact_integer(row["bytes"], "receipt.artifacts[%d].bytes" % index)
        digest = _hex(row["sha256"], "receipt.artifacts[%d].sha256" % index)
        if logical_path in paths:
            raise ReceiptError("duplicate destination path %s" % logical_path)
        paths.add(logical_path)
        if digest in digests:
            raise ReceiptError("digest %s ambiguously maps to both %s and %s" %
                               (digest, digests[digest], logical_path))
        digests[digest] = logical_path
        if relative_parts == ("panel.json",):
            panel_rows += 1
            if digest != panel_digest:
                raise ReceiptError("panel.json does not match token_panel_artifact_sha256")
        artifacts.append(Artifact(logical_path, relative_parts, size, digest))
    if panel_rows != 1:
        raise ReceiptError("receipt must bind exactly one panel.json")
    return tuple(artifacts)


def _validate_source_files(panel_root: os.PathLike,
                           artifacts: Sequence[Artifact]) -> Dict[str, int]:
    panel_fd = _open_directory(panel_root, "fetched panel root")
    opened: Dict[str, int] = {}
    try:
        for artifact in artifacts:
            source_parts = ("calibration", "panel-v1") + artifact.relative_parts
            fd = _open_regular_at(
                panel_fd, source_parts,
                "fetched artifact calibration/panel-v1/%s" %
                "/".join(artifact.relative_parts))
            try:
                size, digest = _hash_fd(fd)
                if size != artifact.size or digest != artifact.digest:
                    raise StageError(
                        "fetched artifact %s fails its listed size/SHA-256" %
                        artifact.path)
                opened[artifact.path] = fd
            except BaseException:
                os.close(fd)
                raise
        return opened
    except Exception:
        for fd in opened.values():
            os.close(fd)
        raise
    finally:
        os.close(panel_fd)


def _require_destination_anchor(destination_prefix: Path,
                                destination_anchor: Path) -> None:
    prefix = PurePosixPath(os.path.abspath(os.fspath(destination_prefix)))
    anchor = PurePosixPath(os.path.abspath(os.fspath(destination_anchor)))
    try:
        prefix.relative_to(anchor)
    except ValueError as exc:
        raise StageError("destination prefix is outside its required anchor") from exc
    fd = _open_directory(str(anchor), "destination anchor")
    os.close(fd)


def _destination_state(destination_prefix: Path,
                       artifact: Artifact) -> str:
    parts = _path_parts(destination_prefix) + artifact.relative_parts
    current = os.open("/", _DIR_FLAGS)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            except FileNotFoundError:
                return "missing"
            os.close(current)
            current = next_fd
        try:
            info = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            return "missing"
        if not stat.S_ISREG(info.st_mode):
            raise StageError("destination %s is not an exact regular file" %
                             artifact.path)
        fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=current)
        try:
            size, digest = _hash_fd(fd)
        finally:
            os.close(fd)
        if size != artifact.size or digest != artifact.digest:
            raise StageError(
                "destination %s exists with different bytes; refusing overwrite" %
                artifact.path)
        return "present"
    except OSError as exc:
        raise StageError("destination %s has an unsafe ancestor: %s" %
                         (artifact.path, exc)) from exc
    finally:
        os.close(current)


def _ensure_directory(path: Path) -> int:
    current = os.open("/", _DIR_FLAGS)
    try:
        for part in _path_parts(path):
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except OSError as exc:
        os.close(current)
        raise StageError("cannot create symlink-safe destination directory %s: %s" %
                         (path, exc)) from exc


def _write_all(fd: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(fd, value[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _copy_new(destination_prefix: Path, artifact: Artifact, source_fd: int) -> None:
    parent_path = destination_prefix.joinpath(*artifact.relative_parts[:-1])
    parent_fd = _ensure_directory(parent_path)
    name = artifact.relative_parts[-1]
    destination_fd = -1
    created = False
    try:
        destination_fd = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
            os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        created = True
        os.lseek(source_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            _write_all(destination_fd, chunk)
            size += len(chunk)
            digest.update(chunk)
        if size != artifact.size or digest.hexdigest() != artifact.digest:
            raise StageError("source changed while copying %s" % artifact.path)
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o644)
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = -1
        verify_fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        try:
            final_size, final_digest = _hash_fd(verify_fd)
        finally:
            os.close(verify_fd)
        if final_size != artifact.size or final_digest != artifact.digest:
            raise StageError("new destination %s failed final verification" %
                             artifact.path)
        os.fsync(parent_fd)
    except BaseException:
        if destination_fd >= 0:
            os.close(destination_fd)
        if created:
            try:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


def stage_panel(
    panel_root: os.PathLike,
    receipt_path: os.PathLike,
    *,
    check_only: bool = False,
    destination_prefix: os.PathLike = DESTINATION_PREFIX,
    destination_anchor: os.PathLike = DESTINATION_ANCHOR,
    expected_receipt_sha256: str = PINNED_RECEIPT_SHA256,
    expected_panel_sha256: str = PINNED_PANEL_SHA256,
    expected_artifact_count: Optional[int] = PINNED_ARTIFACT_COUNT,
    expected_final_windows: int = PINNED_FINAL_WINDOWS,
    expected_final_prediction_positions: int = PINNED_FINAL_PREDICTION_POSITIONS,
) -> Dict[str, int]:
    """Validate the complete operation, then stage missing artifacts securely."""
    artifacts = validate_receipt(
        receipt_path, expected_receipt_sha256=expected_receipt_sha256,
        expected_panel_sha256=expected_panel_sha256,
        expected_artifact_count=expected_artifact_count,
        expected_final_windows=expected_final_windows,
        expected_final_prediction_positions=expected_final_prediction_positions)
    sources = _validate_source_files(panel_root, artifacts)
    prefix = Path(os.path.abspath(os.fspath(destination_prefix)))
    anchor = Path(os.path.abspath(os.fspath(destination_anchor)))
    try:
        _require_destination_anchor(prefix, anchor)
        states = {artifact.path: _destination_state(prefix, artifact)
                  for artifact in artifacts}
        already = sum(state == "present" for state in states.values())
        missing = [artifact for artifact in artifacts
                   if states[artifact.path] == "missing"]
        if not check_only:
            for artifact in missing:
                _copy_new(prefix, artifact, sources[artifact.path])
            for artifact in artifacts:
                if _destination_state(prefix, artifact) != "present":
                    raise StageError("final destination verification failed for %s" %
                                     artifact.path)
        return {
            "artifacts": len(artifacts),
            "already_present": already,
            "staged": 0 if check_only else len(missing),
            "would_stage": len(missing) if check_only else 0,
            "unresolved": 0,
        }
    finally:
        for fd in sources.values():
            os.close(fd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True, help="the fetched panel tree")
    parser.add_argument(
        "--receipt",
        help="token-panel receipt (default <panel>/token-panel-receipt.json)")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    panel_root = Path(os.path.abspath(args.panel))
    receipt_path = Path(args.receipt) if args.receipt else \
        panel_root / "token-panel-receipt.json"
    try:
        summary = stage_panel(panel_root, receipt_path, check_only=args.check_only)
    except ReceiptError as exc:
        print(json.dumps({"artifacts": 0, "already_present": 0, "staged": 0,
                          "would_stage": 0, "unresolved": 1}, sort_keys=True))
        print("stage_panel_paths: %s" % exc, file=sys.stderr)
        return 2
    except (StageError, OSError) as exc:
        print(json.dumps({"artifacts": 0, "already_present": 0, "staged": 0,
                          "would_stage": 0, "unresolved": 1}, sort_keys=True))
        print("stage_panel_paths: %s" % exc, file=sys.stderr)
        return 3
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
