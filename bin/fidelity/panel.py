"""Fail-closed resolution of a local sealed token panel.

A receipt's declared self-seal and its file SHA-256 are different identities.
This module preserves both and reduces a mutable panel directory to one frozen,
JSON-serializable binding suitable for inclusion in a canonical job contract.
"""
from __future__ import annotations

import ast
import hashlib
import fnmatch
import json
import math
import os
import re
import struct
import tempfile
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PANEL_SCHEMA = "quant-pipeline.glm53-token-panel.v1"
BUILD_RECEIPT_SCHEMA = "malaiwah.token-panel-build-receipt.v1"
ARTIFACT_RECEIPT_SCHEMA = "quant-pipeline.glm53-token-panel-receipt.v1"
TOKENIZER_RECEIPT_SCHEMA = "quant-pipeline.glm53-tokenizer-receipt.v1"
RESOLVED_SCHEMA = "malaiwah.resolved-panel.v1"
ARCHIVE_ALGORITHM = "sha256(ustar: sorted regular files; mode=0644; uid=gid=mtime=0)"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DTYPE = re.compile(r"^([<>=|])([iub])(1|2|4|8)$")
_HF_REPO = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION40 = re.compile(r"^[0-9a-f]{40}$")

BRANDON_REFERENCE_REPO = "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
BRANDON_REFERENCE_REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
BRANDON_REFERENCE_REF = "reference--brandonmusic.glm53-bf16-fp32-logits.final25"
BRANDON_TEACHER_RECEIPT_SHA256 = (
    "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5")
BRANDON_PANEL_RECEIPT_SHA256 = (
    "0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5")
BRANDON_BACKEND_SHA256 = (
    "85b11599c6b36a83fa8099a09a298a386a0c603d1f18d3702e7fb1c470962ce4")
BRANDON_REFERENCE_INCLUDE = (
    "*.json",
    "calibration/panel-v1/arrays/*.npy",
    "logits/window-*.safetensors",
)
BRANDON_LOGIT_BYTES = 1268157840
BRANDON_INCLUDED_FILE_COUNT = 743
BRANDON_INCLUDED_BYTES = 31732372989
BRANDON_INCLUDED_MANIFEST_SHA256 = (
    "a0f5c65ba8a082edd6246547ce872b0d2ecc537984a81f6d58a49f179e8cdcac")
M2_TARGET_REPO = "zai-org/GLM-5.3-Flash-BF16"
M2_TARGET_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
M2_PANEL_FILE_SHA256 = (
    "6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff")
M2_PANEL_SUITE_TOKEN_SHA256 = (
    "186b6923582ba59334262178f445440070bd428a862e2e5c9459aaa15b4475fe")
M2_PANEL_MANIFEST_SHA256 = (
    "3e96238bb14cd97b5dab2e87315d8006dc88e4a5314b6fd28eae90e35cc0d0af")
M2_TOKENIZER_IDENTITY_SHA256 = (
    "4de3937ae77b0908990b28ef7b64a6517b5a005bc51205cd071746fd3f60b09d")
GLM53_TARGET_REPO = "zai-org/GLM-5.3-BF16"
GLM53_TARGET_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
GLM53_PANEL_MANIFEST_SHA256 = (
    "4ffa985400a98db57ffcf81b20fee395fe40276e495a1b9ae65e11754897b843")
GLM53_TOKENIZER_IDENTITY_SHA256 = (
    "b18e4d378be7e75fd2b323f7c81cc65640f6a08e1842d3bf4e025d88a9d78bf9")
GLM53_TOKENIZER_FILES = (
    ("LICENSE", 4263,
     "96e1622099fc9d6b70c9760f007d99e66d7497eec636b63c60fe208401e9170c"),
    ("chat_template.jinja", 10465,
     "69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705"),
    ("config.json", 3732,
     "ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca"),
    ("generation_config.json", 194,
     "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55"),
    ("tokenizer.json", 20217442,
     "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"),
    ("tokenizer_config.json", 761,
     "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"),
)
_BRANDON_METADATA = {
    "capture-receipt.json": (
        13831, "af682a8e9f7afd38172565614804f68d570199eed427a5ee25ba151752cab7ab"),
    "dataset-manifest.json": (
        12431, "1c6cba530a60af71ff62e5c2180f2edd26090d8d5845747c85f2e0b53aafc736"),
    "backend.json": (
        3082, "43dd699afc05792ac8cfa9202c5917d90113c92c99ecebaf800f6dd2cd5d411d"),
    "calibration/panel-v1/panel.receipt.json": (
        144018, "fd7416886a9c6f3183b024686e884b19d2b15841a34376b330ed615d015b4086"),
}


class PanelError(ValueError):
    """A local panel cannot support the identity it declares."""


def _canonical(value: Any, newline: bool = False) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return (text + ("\n" if newline else "")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PanelError("%s must be a lowercase SHA-256 hex string" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PanelError("%s must be an integer >= %d" % (label, minimum))
    return value



def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key %r" % key)
        value[key] = item
    return value


def _json_file(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON token %s" % token)))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PanelError("cannot read strict JSON %s %s: %s" % (label, path, exc))
    if not isinstance(value, dict):
        raise PanelError("%s must be a JSON object" % label)
    return value, raw

def _json_bytes(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON token %s" % token)))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PanelError("cannot read strict JSON %s: %s" % (label, exc))
    if not isinstance(value, dict):
        raise PanelError("%s must be a JSON object" % label)
    return value


def _verify_legacy_named_seal(doc: Mapping[str, Any], field: str,
                              expected: str, label: str) -> str:
    claimed = _hex(doc.get(field), "%s.%s" % (label, field))
    if claimed != expected:
        raise PanelError("%s has unexpected %s" % (label, field))
    body = dict(doc)
    del body[field]
    if _sha(_canonical(body, newline=True)) != claimed:
        raise PanelError("%s has an invalid legacy field-absent seal" % label)
    return claimed


def _verify_seal(doc: Mapping[str, Any], label: str) -> Tuple[str, str]:
    """Accept the modern blank-field seal and the public legacy absent-field seal."""
    claimed = _hex(doc.get("receipt_sha256"), "%s.receipt_sha256" % label)
    modern = dict(doc)
    modern["receipt_sha256"] = ""
    if _sha(_canonical(modern)) == claimed:
        return claimed, "self-blank"
    legacy = dict(doc)
    del legacy["receipt_sha256"]
    if _sha(_canonical(legacy, newline=True)) == claimed:
        return claimed, "legacy-field-absent"
    raise PanelError("%s receipt seal is neither modern self-blank nor legacy "
                     "field-absent canonical JSON-with-newline" % label)

def verify_bound_panel_receipt_bytes(
        receipt_binding: Mapping[str, Any], raw: bytes,
        label: str = "panel receipt") -> Dict[str, Any]:
    """Verify exact raw bytes and semantic seal against a resolved binding."""
    if not isinstance(receipt_binding, Mapping):
        raise PanelError("%s binding must be an object" % label)
    if not isinstance(raw, bytes):
        raise PanelError("%s bytes are unavailable" % label)
    expected_bytes = _integer(
        receipt_binding.get("bytes"), "%s binding.bytes" % label, minimum=1)
    expected_file_sha = _hex(
        receipt_binding.get("receipt_file_sha256"),
        "%s binding.receipt_file_sha256" % label)
    expected_declared = _hex(
        receipt_binding.get("declared_receipt_sha256"),
        "%s binding.declared_receipt_sha256" % label)
    expected_mode = receipt_binding.get("receipt_seal_mode")
    if expected_mode not in ("self-blank", "legacy-field-absent"):
        raise PanelError("%s binding seal mode is unsupported" % label)
    if len(raw) != expected_bytes or _sha(raw) != expected_file_sha:
        raise PanelError("%s raw bytes differ from resolved binding" % label)
    receipt = _json_bytes(raw, label)
    declared, mode = _verify_seal(receipt, label)
    if declared != expected_declared or mode != expected_mode:
        raise PanelError("%s semantic seal differs from resolved binding" % label)
    return receipt


def _pinned_tokenizer_identity(repository: Any, revision: Any,
                               label: str) -> Tuple[str, str]:
    if not isinstance(repository, str) or _HF_REPO.fullmatch(repository) is None:
        raise PanelError("%s repository must have immutable owner/name form" % label)
    if not isinstance(revision, str) or _REVISION40.fullmatch(revision) is None:
        raise PanelError("%s revision must be a lowercase 40-hex commit" % label)
    return repository, revision


def _safe_rel(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        raise PanelError("%s must be a non-empty canonical POSIX path" % label)
    path = PurePosixPath(value)
    canonical = path.as_posix()
    if (canonical != value or path.is_absolute()
            or any(part in ("", ".", "..") for part in path.parts)):
        raise PanelError("unsafe or non-canonical %s: %r" % (label, value))
    return canonical


def _artifact_contract(receipt: Mapping[str, Any]) -> Optional[Tuple[Tuple[str, int, str], ...]]:
    if receipt.get("schema") != ARTIFACT_RECEIPT_SCHEMA:
        return None
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise PanelError("artifact panel receipt has no artifacts")
    panel_sha = _hex(receipt.get("token_panel_artifact_sha256"),
                     "receipt.token_panel_artifact_sha256")
    parsed = []
    panel_paths = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PanelError("receipt.artifacts[%d] must contain exactly path, bytes, sha256" % index)
        raw_path = row["path"]
        if (not isinstance(raw_path, str) or "\\" in raw_path or "//" in raw_path):
            raise PanelError("receipt.artifacts[%d].path is not canonical POSIX" % index)
        path = PurePosixPath(raw_path)
        parts = path.parts[1:] if path.is_absolute() else path.parts
        if path.as_posix() != raw_path or any(part in ("", ".", "..") for part in parts):
            raise PanelError("receipt.artifacts[%d].path is unsafe or non-canonical" % index)
        size = _integer(row["bytes"], "receipt.artifacts[%d].bytes" % index)
        digest = _hex(row["sha256"], "receipt.artifacts[%d].sha256" % index)
        parsed.append((path, size, digest))
        if path.name == "panel.json" and digest == panel_sha:
            panel_paths.append(path)
    if len(panel_paths) != 1:
        raise PanelError("receipt must bind exactly one panel.json with token_panel_artifact_sha256")
    prefix = panel_paths[0].parent
    out = []
    seen = set()
    for path, size, digest in parsed:
        try:
            rel = _safe_rel(str(path.relative_to(prefix)), "artifact path")
        except ValueError:
            raise PanelError("listed artifact %s is outside panel root %s" % (path, prefix))
        if rel in seen:
            raise PanelError("duplicate listed artifact path %s" % rel)
        seen.add(rel)
        out.append((rel, size, digest))
    return tuple(sorted(out))


def _npy(path: Path) -> Tuple[str, Tuple[int, ...], int, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PanelError("cannot read panel array %s: %s" % (path, exc))
    if len(raw) < 10 or raw[:6] != b"\x93NUMPY":
        raise PanelError("%s is not a .npy file" % path)
    major = raw[6]
    if major == 1:
        start, count, encoding = 10, struct.unpack("<H", raw[8:10])[0], "latin1"
    elif major in (2, 3) and len(raw) >= 12:
        start, count = 12, struct.unpack("<I", raw[8:12])[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise PanelError("%s has unsupported or truncated .npy version" % path)
    end = start + count
    try:
        header = ast.literal_eval(raw[start:end].decode(encoding).strip())
    except (UnicodeDecodeError, ValueError, SyntaxError) as exc:
        raise PanelError("invalid .npy header in %s: %s" % (path, exc))
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise PanelError("unsupported .npy header in %s" % path)
    dtype, shape = header["descr"], header["shape"]
    if header["fortran_order"] is not False or not isinstance(shape, tuple) or not shape:
        raise PanelError("%s must be a non-empty C-order array" % path)
    match = _DTYPE.fullmatch(dtype) if isinstance(dtype, str) else None
    if match is None:
        raise PanelError("unsupported dtype %r in %s" % (dtype, path))
    dims = tuple(_integer(value, "%s shape" % path) for value in shape)
    needed = math.prod(dims) * int(match.group(3))
    if end > len(raw) or len(raw) - end != needed:
        raise PanelError("%s .npy payload size disagrees with dtype/shape" % path)
    return dtype, dims, end, raw


def _values(dtype: str, raw: bytes, offset: int) -> Iterable[int]:
    match = _DTYPE.fullmatch(dtype)
    assert match is not None
    endian, kind, width_text = match.groups()
    if kind not in ("i", "u"):
        raise PanelError("token dtype must be integer, got %s" % dtype)
    width = int(width_text)
    code = {("i", 1): "b", ("u", 1): "B", ("i", 2): "h", ("u", 2): "H",
            ("i", 4): "i", ("u", 4): "I", ("i", 8): "q", ("u", 8): "Q"}[(kind, width)]
    prefix = ">" if endian == ">" else "<"
    return (item[0] for item in struct.iter_unpack(prefix + code, raw[offset:]))

def _mask_values(dtype: str, raw: bytes, offset: int) -> Iterable[int]:
    match = _DTYPE.fullmatch(dtype)
    assert match is not None
    endian, kind, width_text = match.groups()
    width = int(width_text)
    if kind == "b":
        if width != 1:
            raise PanelError("boolean mask dtype must have width 1, got %s" % dtype)
        code = "?"
    else:
        code = {("i", 1): "b", ("u", 1): "B", ("i", 2): "h", ("u", 2): "H",
                ("i", 4): "i", ("u", 4): "I", ("i", 8): "q",
                ("u", 8): "Q"}[(kind, width)]
    prefix = ">" if endian == ">" else "<"
    return (int(item[0]) for item in struct.iter_unpack(prefix + code, raw[offset:]))


def _listed_tokenizer_files(rows: Any, label: str) -> Tuple[Tuple[str, int, str], ...]:
    if not isinstance(rows, list) or not rows:
        raise PanelError("%s must be a non-empty artifact array" % label)
    out = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise PanelError("%s[%d] must contain exactly path, bytes, sha256" % (label, index))
        raw_path = row["path"]
        if (not isinstance(raw_path, str) or "\\" in raw_path or "//" in raw_path):
            raise PanelError("%s[%d].path is not canonical POSIX" % (label, index))
        path = PurePosixPath(raw_path)
        if path.as_posix() != raw_path or not path.name:
            raise PanelError("%s[%d].path is non-canonical" % (label, index))
        name = path.name
        if name in seen:
            raise PanelError("%s has duplicate canonical basename %s" % (label, name))
        seen.add(name)
        out.append((name, _integer(row["bytes"], "%s[%d].bytes" % (label, index)),
                    _hex(row["sha256"], "%s[%d].sha256" % (label, index))))
    return tuple(sorted(out))


def _verify_tokenizer_files(rows: Sequence[Tuple[str, int, str]], root: Optional[os.PathLike]) -> bool:
    if root is None:
        return False
    base = Path(root).resolve()
    if not base.is_dir():
        raise PanelError("tokenizer_root is not a directory: %s" % base)
    for name, size, digest in rows:
        path = base / name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise PanelError("cannot read tokenizer artifact %s: %s" % (path, exc))
        if len(raw) != size or _sha(raw) != digest:
            raise PanelError("tokenizer artifact %s fails its listed size/SHA-256" % name)
    return True


def _tokenizer_binding(panel_root: Path, receipt: Mapping[str, Any],
                       tokenizer_root: Optional[os.PathLike]) -> Dict[str, Any]:
    token_receipt_path = panel_root / "tokenizer.receipt.json"
    if token_receipt_path.is_file():
        token_receipt, raw = _json_file(token_receipt_path, "tokenizer receipt")
        if token_receipt.get("schema") != TOKENIZER_RECEIPT_SCHEMA:
            raise PanelError("unsupported tokenizer receipt schema %r" % token_receipt.get("schema"))
        declared, mode = _verify_seal(token_receipt, "tokenizer")
        identity = token_receipt.get("tokenizer_identity")
        if not isinstance(identity, dict):
            raise PanelError("tokenizer_identity must be an object")
        identity_sha = _hex(token_receipt.get("tokenizer_identity_sha256"),
                            "tokenizer_identity_sha256")
        if _sha(_canonical(identity, newline=True)) != identity_sha:
            raise PanelError("tokenizer_identity_sha256 does not verify")
        repository, revision = _pinned_tokenizer_identity(
            identity.get("model_id"), identity.get("model_revision"),
            "tokenizer identity")
        files = _listed_tokenizer_files(identity.get("files"), "tokenizer_identity.files")
        vocab = _integer(token_receipt.get("vocab_size"), "tokenizer.vocab_size", 1)
        if _integer(token_receipt.get("minimum_token_id"), "tokenizer.minimum_token_id") != 0:
            raise PanelError("tokenizer minimum_token_id must be zero")
        maximum = _integer(token_receipt.get("maximum_token_id_exclusive"),
                           "tokenizer.maximum_token_id_exclusive", 1)
        if maximum > vocab:
            raise PanelError("tokenizer maximum_token_id_exclusive exceeds vocab_size")
        return {"repository": repository, "revision": revision, "vocab_size": vocab,
                "maximum_token_id_exclusive": maximum, "files": files,
                "files_verified": _verify_tokenizer_files(files, tokenizer_root),
                "identity_sha256": identity_sha, "receipt": {
                    "declared_receipt_sha256": declared, "receipt_seal_mode": mode,
                    "receipt_file_sha256": _sha(raw), "receipt_file_bytes": len(raw)}}
    source = receipt.get("tokenizer")
    if not isinstance(source, dict):
        raise PanelError("panel has no tokenizer.receipt.json or receipt.tokenizer identity")
    repository, revision = _pinned_tokenizer_identity(
        source.get("repository"), source.get("revision"), "receipt.tokenizer")
    file_map = source.get("files_sha256")
    if not isinstance(file_map, dict) or not file_map:
        raise PanelError("receipt.tokenizer.files_sha256 must be non-empty")
    files_list = []
    seen_names = set()
    for source_name, digest in file_map.items():
        safe_name = _safe_rel(source_name, "receipt.tokenizer.files_sha256 key")
        name = PurePosixPath(safe_name).name
        if name in seen_names:
            raise PanelError("receipt.tokenizer files have duplicate basename %s" % name)
        seen_names.add(name)
        files_list.append((name, -1, _hex(digest, "tokenizer file sha256")))
    files = tuple(sorted(files_list))
    verified = False
    if tokenizer_root is not None:
        observed = []
        base = Path(tokenizer_root).resolve()
        for name, _size, digest in files:
            raw = (base / name).read_bytes()
            if _sha(raw) != digest:
                raise PanelError("tokenizer artifact %s fails its SHA-256" % name)
            observed.append((name, len(raw), digest))
        files, verified = tuple(observed), True
    return {"repository": repository, "revision": revision,
            "vocab_size": _integer(source.get("vocab_size"), "tokenizer.vocab_size", 1),
            "maximum_token_id_exclusive": _integer(source.get("vocab_size"),
                                                     "tokenizer.vocab_size", 1),
            "files": files, "files_verified": verified,
            "identity_sha256": _sha(_canonical({"repository": repository,
                                                 "revision": revision,
                                                 "files_sha256": dict(sorted(file_map.items())),
                                                 "vocab_size": source.get("vocab_size")},
                                                newline=True)), "receipt": None}


class _HashWriter:
    def __init__(self, output=None) -> None:
        self.digest, self.size, self.output = hashlib.sha256(), 0, output

    def write(self, data: bytes) -> int:
        if self.output is not None:
            self.output.write(data)
        self.digest.update(data)
        self.size += len(data)
        return len(data)


class _CheckedReader:
    def __init__(self, source, expected_size: int, expected_sha256: str) -> None:
        self.source = source
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def verify(self, rel: str) -> None:
        if self.size != self.expected_size or self.digest.hexdigest() != self.expected_sha256:
            raise PanelError("panel file changed while archiving: %s" % rel)


def _emit_archive(root: Path, manifest: Sequence[Tuple[str, int, str]],
                  output=None) -> Tuple[str, int]:
    sink = _HashWriter(output)
    with tarfile.open(fileobj=sink, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for rel, size, digest in manifest:
            info = tarfile.TarInfo(rel)
            info.size, info.mode = size, 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with (root / rel).open("rb") as source:
                checked = _CheckedReader(source, size, digest)
                archive.addfile(info, checked)
                checked.verify(rel)
    return sink.digest.hexdigest(), sink.size


def _archive(root: Path, manifest: Sequence[Tuple[str, int, str]]) -> Tuple[str, int]:
    return _emit_archive(root, manifest)


@dataclass(frozen=True)
class ResolvedPanel:
    _serialized: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a fresh JSON tree; no mutable object is retained internally."""
        return json.loads(self._serialized)


def resolve_panel(root: os.PathLike, role: str = "final",
                  tokenizer_root: Optional[os.PathLike] = None) -> ResolvedPanel:
    panel_root = Path(root).resolve()
    if not panel_root.is_dir() or not isinstance(role, str) or not role:
        raise PanelError("panel root must be a directory and role must be non-empty")
    panel, panel_raw = _json_file(panel_root / "panel.json", "panel")
    receipt, receipt_raw = _json_file(panel_root / "panel.receipt.json", "panel receipt")
    if panel.get("schema") != PANEL_SCHEMA:
        raise PanelError("unsupported panel schema %r" % panel.get("schema"))
    if receipt.get("schema") not in (BUILD_RECEIPT_SCHEMA, ARTIFACT_RECEIPT_SCHEMA):
        raise PanelError("unsupported panel receipt schema %r" % receipt.get("schema"))
    declared_receipt, seal_mode = _verify_seal(receipt, "panel")
    contract = _artifact_contract(receipt)

    files = {}
    for path in panel_root.rglob("*"):
        if path.is_symlink():
            raise PanelError("panel contains symlink %s" % path)
        if path.is_file():
            rel = path.relative_to(panel_root).as_posix()
            raw = path.read_bytes()
            files[rel] = (len(raw), _sha(raw))
        elif not path.is_dir():
            raise PanelError("panel contains non-regular entry %s" % path)
    for rel, size, digest in contract or ():
        if rel not in files:
            raise PanelError("listed panel artifact is missing: %s" % rel)
        if files[rel] != (size, digest):
            raise PanelError("listed panel artifact %s fails size/SHA-256" % rel)
    if contract is not None and files.get("panel.json", (None, None))[1] != receipt.get(
            "token_panel_artifact_sha256"):
        raise PanelError("panel.json does not match token_panel_artifact_sha256")

    rows = panel.get("windows")
    if not isinstance(rows, list) or not rows:
        raise PanelError("panel.windows must be a non-empty array")
    artifact_by_digest: Dict[str, List[str]] = {}
    for rel, (_size, digest) in files.items():
        artifact_by_digest.setdefault(digest, []).append(rel)
    selected = []
    all_digests = []
    seen = set()
    used_window_files = set()
    tokenizer = _tokenizer_binding(panel_root, receipt, tokenizer_root)
    token_limit = tokenizer["maximum_token_id_exclusive"]
    if receipt.get("tokenizer_receipt_sha256") is not None:
        token_receipt = tokenizer.get("receipt")
        if token_receipt is None or _hex(
                receipt["tokenizer_receipt_sha256"],
                "receipt.tokenizer_receipt_sha256") != token_receipt[
                    "declared_receipt_sha256"]:
            raise PanelError("panel receipt binds a different tokenizer receipt")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PanelError("panel.windows[%d] must be an object" % index)
        window = row.get("window_id")
        if not isinstance(window, str) or not window or window in seen:
            raise PanelError("panel window_id is absent or duplicated")
        seen.add(window)
        token_sha = _hex(row.get("token_ids_sha256"), "window token_ids_sha256")
        mask_sha = _hex(row.get("attention_mask_sha256"), "window attention_mask_sha256")
        preferred_token = "arrays/%s.tokens.npy" % window
        preferred_mask = "arrays/%s.mask.npy" % window
        token_candidates = [p for p in artifact_by_digest.get(token_sha, ())
                            if p.endswith(".tokens.npy")]
        mask_candidates = [p for p in artifact_by_digest.get(mask_sha, ()) if p.endswith(".npy")]
        token_rel = preferred_token if preferred_token in token_candidates else (
            token_candidates[0] if len(token_candidates) == 1 else None)
        mask_rel = preferred_mask if preferred_mask in mask_candidates else (
            sorted(mask_candidates)[0] if mask_candidates else None)
        if token_rel is None or mask_rel is None:
            raise PanelError("window %s cannot resolve its exact token/mask artifacts" % window)
        used_window_files.update((token_rel, mask_rel))
        token_dtype, token_shape, offset, token_raw = _npy(panel_root / token_rel)
        mask_dtype, mask_shape, mask_offset, mask_raw = _npy(panel_root / mask_rel)
        if _sha(token_raw) != token_sha or _sha(mask_raw) != mask_sha:
            raise PanelError("window %s artifact digest mismatch" % window)
        if len(token_shape) != 1 or token_shape != mask_shape:
            raise PanelError("window %s token/mask shapes differ or are not vectors" % window)
        mask_match = _DTYPE.fullmatch(mask_dtype)
        if mask_match is None or mask_match.group(2) not in ("i", "u", "b"):
            raise PanelError("window %s mask dtype is unsupported" % window)
        values = list(_values(token_dtype, token_raw, offset))
        if any(value < 0 or value >= token_limit for value in values):
            raise PanelError("window %s has token outside pinned tokenizer range" % window)
        token_json_sha = _sha(_canonical(values))
        if "token_ids_json_sha256" in row and token_json_sha != _hex(
                row["token_ids_json_sha256"], "window token_ids_json_sha256"):
            raise PanelError("window %s token JSON digest mismatch" % window)
        if "num_tokens" in row and _integer(row["num_tokens"], "window num_tokens", 1) != token_shape[0]:
            raise PanelError("window %s num_tokens disagrees with artifact shape" % window)
        positions = _integer(row.get("prediction_positions"), "window prediction_positions", 1)
        if positions > token_shape[0]:
            raise PanelError("window %s scores more positions than tokens" % window)
        mask_values = list(_mask_values(mask_dtype, mask_raw, mask_offset))
        if any(value not in (0, 1) for value in mask_values):
            raise PanelError("window %s mask contains a value other than 0 or 1" % window)
        shifted_valid = sum(
            1 for current, following in zip(mask_values[:-1], mask_values[1:])
            if current == 1 and following == 1)
        if shifted_valid != positions:
            raise PanelError(
                "window %s mask has %d valid shifted next-token pairs but "
                "prediction_positions is %d" % (window, shifted_valid, positions))
        window_role = row.get("role")
        if not isinstance(window_role, str) or not window_role:
            raise PanelError("window %s role is absent" % window)
        all_digests.append((window, token_json_sha))
        if window_role == role:
            selected.append((window, token_shape[0], positions, token_json_sha))
    expected_files = ({rel for rel, _size, _digest in contract}
                      if contract is not None
                      else {"panel.json", "panel.receipt.json"})
    expected_files.add("panel.receipt.json")
    expected_files.update(used_window_files)
    if receipt.get("tokenizer_receipt_sha256") is not None:
        expected_files.add("tokenizer.receipt.json")
    actual_files = set(files)
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    if missing_files or extra_files:
        raise PanelError("panel root is not the exact sealed file closure; missing=%r extra=%r"
                         % (missing_files, extra_files))
    if not selected:
        raise PanelError("panel has no windows with role %r" % role)
    all_suite = _sha("\n".join(digest for _window, digest in sorted(all_digests)).encode("ascii"))
    for label, value in (("panel", panel.get("suite_token_hash_sha256")),
                         ("receipt", receipt.get("suite_token_hash_sha256"))):
        if value is not None and _hex(value, "%s suite token hash" % label) != all_suite:
            raise PanelError("%s suite token hash does not match sorted token artifacts" % label)
    selected.sort()
    lengths = {row[1] for row in selected}
    positions = {row[2] for row in selected}
    if len(lengths) != 1 or len(positions) != 1:
        raise PanelError("selected role is not shape-homogeneous")
    context_length, per_context = next(iter(lengths)), next(iter(positions))
    scored_total = sum(row[2] for row in selected)
    selected_suite = _sha("\n".join(row[3] for row in selected).encode("ascii"))
    params = receipt.get("parameters")
    if role == "final" and isinstance(params, dict):
        for key, actual in (("context_length", context_length),
                            ("prediction_positions_per_window", per_context),
                            ("scored_positions_total", scored_total),
                            ("windows_total", len(selected))):
            if key in params and _integer(params[key], "receipt.parameters.%s" % key) != actual:
                raise PanelError("receipt.parameters.%s disagrees with selected role" % key)
    if role == "final":
        for key, actual in (("final_windows", len(selected)),
                            ("final_prediction_positions", scored_total)):
            if key in receipt and _integer(receipt[key], "receipt.%s" % key) != actual:
                raise PanelError("receipt.%s disagrees with selected role" % key)

    manifest = tuple((rel, files[rel][0], files[rel][1]) for rel in sorted(files))
    manifest_doc = [{"path": rel, "bytes": size, "sha256": digest}
                    for rel, size, digest in manifest]
    archive_sha, archive_bytes = _archive(panel_root, manifest)
    panel_id = panel.get("panel_id") or receipt.get("panel_id") or (
        "panel-artifact-sha256:%s" % _sha(panel_raw))
    panel_name = panel.get("name") or receipt.get("panel_name") or panel_id
    binding = {
        "schema": RESOLVED_SCHEMA,
        "panel": {"id": panel_id, "name": panel_name, "role": role,
                  "contexts": len(selected), "context_length": context_length,
                  "positions_per_context": per_context,
                  "scored_positions_total": scored_total,
                  "suite_token_hash_sha256": selected_suite,
                  "file": "panel.json", "bytes": len(panel_raw), "sha256": _sha(panel_raw)},
        "receipt": {"file": "panel.receipt.json", "bytes": len(receipt_raw),
                    "declared_receipt_sha256": declared_receipt,
                    "receipt_seal_mode": seal_mode,
                    "receipt_file_sha256": _sha(receipt_raw)},
        "tokenizer": {"id": tokenizer["repository"],
                      "repository": tokenizer["repository"],
                      "revision": tokenizer["revision"],
                      "vocab_size": tokenizer["vocab_size"],
                      "maximum_token_id_exclusive": token_limit,
                      "identity_sha256": tokenizer["identity_sha256"],
                      "files": [{"name": name, "bytes": size, "sha256": digest}
                                for name, size, digest in tokenizer["files"]],
                      "files_verified": tokenizer["files_verified"],
                      "receipt": tokenizer["receipt"]},
        "content": {"manifest": manifest_doc,
                    "manifest_sha256": _sha(_canonical(manifest_doc)),
                    "archive": {"format": "ustar", "compression": "none",
                                "algorithm": ARCHIVE_ALGORITHM,
                                "bytes": archive_bytes, "sha256": archive_sha}}}
    return ResolvedPanel(_canonical(binding).decode("utf-8"))


def write_panel_archive(
        root: os.PathLike, destination: os.PathLike, role: str = "final",
        tokenizer_root: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Atomically write the exact deterministic archive named by a panel binding.

    Only files in the resolved panel root manifest are included. ``tokenizer_root``
    is verification input and is never copied into the archive.
    """
    panel_root = Path(root).resolve()
    resolved = resolve_panel(panel_root, role=role, tokenizer_root=tokenizer_root)
    binding = resolved.to_dict()
    target = Path(destination).resolve()
    try:
        inside_panel = os.path.commonpath((str(panel_root), str(target))) == str(panel_root)
    except ValueError:
        inside_panel = False
    if inside_panel:
        raise PanelError("panel archive destination must be outside the panel root")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = tuple(
        (row["path"], _integer(row["bytes"], "content manifest bytes"),
         _hex(row["sha256"], "content manifest sha256"))
        for row in binding["content"]["manifest"])
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name,
                                     suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as output:
            digest, size = _emit_archive(panel_root, manifest, output)
            output.flush()
            os.fsync(output.fileno())
        expected = binding["content"]["archive"]
        if digest != expected["sha256"] or size != expected["bytes"]:
            raise PanelError("written panel archive differs from the resolved binding")
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent),
                               os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    binding_sha = _sha(_canonical(binding))
    return {"path": str(target), "bytes": size, "sha256": digest,
            "binding": binding, "binding_sha256": binding_sha}


def _validate_glm53_root_panel(resolved: Mapping[str, Any]) -> None:
    if set(resolved) != {"schema", "panel", "receipt", "tokenizer", "content"}:
        raise PanelError("full GLM-5.3 resolved panel keys differ")
    expected_panel = {
        "id": "panel--glm53.malaiwah.corpus5x5-v1",
        "name": "GLM-5.3 corpus 5-stratum x 5-window panel",
        "role": "final",
        "contexts": 25,
        "context_length": 2048,
        "positions_per_context": 2047,
        "scored_positions_total": 51175,
        "suite_token_hash_sha256":
            "f09ee395f635225a077695ed193d61f7d1e70650cebe1b68f82b16f59d399f86",
        "file": "panel.json",
        "bytes": 26504,
        "sha256":
            "f2df810b44b840568a28e89abd6fe0252bb8ce6204f926acec9ab7b7d3aa649c",
    }
    expected_receipt = {
        "file": "panel.receipt.json",
        "bytes": 13969,
        "declared_receipt_sha256":
            "9c3bc4f59e8825ac78b366ca0e2988a48ecdbcaf26fdaeaedd3251edb6f9a828",
        "receipt_seal_mode": "self-blank",
        "receipt_file_sha256":
            "abaf095a0887b35fe5f0b0fcd34de4f6448bd314955bc870a3ceb93d43c726d0",
    }
    if (resolved.get("schema") != RESOLVED_SCHEMA
            or resolved.get("panel") != expected_panel
            or resolved.get("receipt") != expected_receipt):
        raise PanelError(
            "full GLM-5.3 root is not the exact corpus5x5 panel/receipt")

    expected_tokenizer_files = [
        {"name": name, "bytes": size, "sha256": digest}
        for name, size, digest in GLM53_TOKENIZER_FILES
    ]
    expected_tokenizer = {
        "id": GLM53_TARGET_REPO,
        "repository": GLM53_TARGET_REPO,
        "revision": GLM53_TARGET_REVISION,
        "vocab_size": 154820,
        "maximum_token_id_exclusive": 154820,
        "identity_sha256": GLM53_TOKENIZER_IDENTITY_SHA256,
        "files": expected_tokenizer_files,
        "files_verified": True,
        "receipt": None,
    }
    if resolved.get("tokenizer") != expected_tokenizer:
        raise PanelError(
            "full GLM-5.3 root tokenizer pin or verified files differ")

    content = resolved.get("content")
    if not isinstance(content, dict) or set(content) != {
            "manifest", "manifest_sha256", "archive"}:
        raise PanelError("full GLM-5.3 root panel content binding differs")
    rows = content.get("manifest")
    if (not isinstance(rows, list) or len(rows) != 28
            or rows != sorted(rows, key=lambda row: row.get("path", ""))
            or len({row.get("path") for row in rows}) != len(rows)
            or any(not isinstance(row, dict)
                   or set(row) != {"path", "bytes", "sha256"}
                   for row in rows)
            or _sha(_canonical(rows)) != GLM53_PANEL_MANIFEST_SHA256
            or content.get("manifest_sha256")
                != GLM53_PANEL_MANIFEST_SHA256
            or content.get("archive") != {
                "algorithm": ARCHIVE_ALGORITHM,
                "bytes": 276480,
                "compression": "none",
                "format": "ustar",
                "sha256":
                    "34450e4b7db23d0dd52f5ef9b1b427547ea640f3dcc4de4a12259c31e4c4d2ea",
            }):
        raise PanelError(
            "full GLM-5.3 root panel content identity differs")


def validate_root_panel_binding(
        binding: Mapping[str, Any], target_repo: str,
        revision: str) -> Dict[str, Any]:
    """Refuse any scientifically valid panel that is not exact for the target."""
    try:
        resolved = json.loads(_canonical(binding).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise PanelError("root panel binding is not finite canonical JSON: %s" % exc)
    if not isinstance(resolved, dict):
        raise PanelError("root panel binding must be an object")

    if (target_repo, revision) == (
            GLM53_TARGET_REPO, GLM53_TARGET_REVISION):
        _validate_glm53_root_panel(resolved)
        return resolved
    if (target_repo, revision) == (
            "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
            "ef68013aa6e16453cf52b5b77647f72fbe258c3c"):
        from fidelity.runpodsafety import (
            SafetyProofError, _validate_fruit_panel_binding)
        try:
            _validate_fruit_panel_binding(resolved)
        except SafetyProofError as exc:
            raise PanelError("target-specific Fruit panel refuses: %s" % exc)
        return resolved
    if (target_repo, revision) != (M2_TARGET_REPO, M2_TARGET_REVISION):
        raise PanelError("no exact paid root-panel contract for %s@%s" %
                         (target_repo, revision))

    if set(resolved) != {"schema", "panel", "receipt", "tokenizer", "content"}:
        raise PanelError("M2 resolved panel keys differ")
    expected_panel_id = "panel-artifact-sha256:%s" % M2_PANEL_FILE_SHA256
    expected_panel = {
        "id": expected_panel_id,
        "name": expected_panel_id,
        "role": "final",
        "contexts": 25,
        "context_length": 2048,
        "positions_per_context": 2047,
        "scored_positions_total": 51175,
        "suite_token_hash_sha256": M2_PANEL_SUITE_TOKEN_SHA256,
        "file": "panel.json",
        "bytes": 266208,
        "sha256": M2_PANEL_FILE_SHA256,
    }
    expected_receipt = {
        "file": "panel.receipt.json",
        "bytes": 144018,
        "declared_receipt_sha256": BRANDON_PANEL_RECEIPT_SHA256,
        "receipt_seal_mode": "legacy-field-absent",
        "receipt_file_sha256":
            "fd7416886a9c6f3183b024686e884b19d2b15841a34376b330ed615d015b4086",
    }
    if (resolved.get("schema") != RESOLVED_SCHEMA
            or resolved.get("panel") != expected_panel
            or resolved.get("receipt") != expected_receipt):
        raise PanelError("M2 root is not the exact Brandon final25 panel/receipt")

    tokenizer = resolved.get("tokenizer")
    expected_tokenizer_files = [
        {"name": "chat_template.jinja", "bytes": 8617,
         "sha256": "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3"},
        {"name": "tokenizer.json", "bytes": 20217442,
         "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"},
        {"name": "tokenizer_config.json", "bytes": 761,
         "sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
    ]
    expected_tokenizer_receipt = {
        "declared_receipt_sha256":
            "d1522ca96b57e1e60e781d217557d5ed4b9669537cddfeefe346e568ccce8656",
        "receipt_seal_mode": "legacy-field-absent",
        "receipt_file_sha256":
            "fd6e407903e7c787f84df361c44d0af945193ade27e953a02dd613ecf9a4c3b2",
        "receipt_file_bytes": 1778,
    }
    expected_tokenizer = {
        "id": M2_TARGET_REPO,
        "repository": M2_TARGET_REPO,
        "revision": M2_TARGET_REVISION,
        "vocab_size": 154856,
        "maximum_token_id_exclusive": 154856,
        "identity_sha256": M2_TOKENIZER_IDENTITY_SHA256,
        "files": expected_tokenizer_files,
        "files_verified": True,
        "receipt": expected_tokenizer_receipt,
    }
    if tokenizer != expected_tokenizer:
        raise PanelError("M2 root tokenizer pin or verified receipt differs")

    content = resolved.get("content")
    if not isinstance(content, dict) or set(content) != {
            "manifest", "manifest_sha256", "archive"}:
        raise PanelError("M2 root panel content binding differs")
    rows = content.get("manifest")
    if not isinstance(rows, list) or len(rows) != 669:
        raise PanelError("M2 root panel must bind exactly 669 content files")
    paths = set()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict)
                or set(row) != {"path", "bytes", "sha256"}):
            raise PanelError("M2 content row %d has unexpected fields" % index)
        path = _safe_rel(row["path"], "M2 content row %d path" % index)
        if path in paths:
            raise PanelError("duplicate M2 panel content path %s" % path)
        paths.add(path)
        _integer(row["bytes"], "M2 content row %s bytes" % path, minimum=1)
        _hex(row["sha256"], "M2 content row %s sha256" % path)
    observed_manifest_sha = _sha(_canonical(rows))
    if (content.get("manifest_sha256") != M2_PANEL_MANIFEST_SHA256
            or observed_manifest_sha != M2_PANEL_MANIFEST_SHA256):
        raise PanelError("M2 root content is not the exact Brandon manifest")
    by_path = {row["path"]: row for row in rows}
    if (by_path.get("panel.json") != {
            "path": "panel.json", "bytes": 266208,
            "sha256": M2_PANEL_FILE_SHA256}
            or by_path.get("panel.receipt.json") != {
                "path": "panel.receipt.json", "bytes": 144018,
                "sha256":
                    "fd7416886a9c6f3183b024686e884b19d2b15841a34376b330ed615d015b4086"}
            or by_path.get("tokenizer.receipt.json") != {
                "path": "tokenizer.receipt.json", "bytes": 1778,
                "sha256":
                    "fd6e407903e7c787f84df361c44d0af945193ade27e953a02dd613ecf9a4c3b2"}):
        raise PanelError("M2 root panel metadata content identities differ")

    archive = content.get("archive")
    if (not isinstance(archive, dict)
            or set(archive) != {
                "format", "compression", "algorithm", "bytes", "sha256"}
            or archive.get("format") != "ustar"
            or archive.get("compression") != "none"
            or archive.get("algorithm") != ARCHIVE_ALGORITHM
            or archive.get("bytes") != 6553600):
        raise PanelError("M2 root deterministic archive shape differs")
    _hex(archive.get("sha256"), "M2 root archive sha256")
    return resolved


def validate_reference_manifest(
        repo_id: str, revision: str, repo_meta: Any = None,
        fetch_file_func: Any = None) -> Dict[str, Any]:
    """Resolve the one admitted Brandon final25 teacher reference without logits.

    The result binds the immutable RepoMeta closure and the exact content
    declarations in the four small, pinned public metadata documents.  It does
    not fetch a logit tensor.
    """
    if repo_id != BRANDON_REFERENCE_REPO:
        raise PanelError("unsupported teacher reference repository %r" % repo_id)
    if revision != BRANDON_REFERENCE_REVISION:
        raise PanelError("unsupported teacher reference revision %r" % revision)
    try:
        from fidelity import hfmeta
    except ImportError as exc:
        raise PanelError("cannot load Hugging Face metadata client: %s" % exc)
    if repo_meta is None:
        repo_meta = hfmeta.repo_meta(repo_id, "dataset", revision)
    if (getattr(repo_meta, "repo_id", None) != repo_id
            or getattr(repo_meta, "repo_type", None) != "dataset"
            or getattr(repo_meta, "revision", None) != revision):
        raise PanelError("RepoMeta identity differs from the pinned reference")

    meta_files: Dict[str, int] = {}
    try:
        source_files = repo_meta.files
    except AttributeError:
        raise PanelError("RepoMeta has no files inventory")
    for index, entry in enumerate(source_files):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise PanelError("RepoMeta file %d is not a (path, size) pair" % index)
        path = _safe_rel(entry[0], "RepoMeta file %d path" % index)
        size = _integer(entry[1], "RepoMeta file %s bytes" % path, minimum=1)
        if path in meta_files:
            raise PanelError("duplicate RepoMeta path %s" % path)
        meta_files[path] = size

    included = [
        {"path": path, "bytes": size}
        for path, size in sorted(meta_files.items())
        if any(fnmatch.fnmatchcase(path, pattern)
               for pattern in BRANDON_REFERENCE_INCLUDE)
    ]
    included_bytes = sum(row["bytes"] for row in included)
    included_sha = _sha(_canonical(included))
    if (len(included) != BRANDON_INCLUDED_FILE_COUNT
            or included_bytes != BRANDON_INCLUDED_BYTES
            or included_sha != BRANDON_INCLUDED_MANIFEST_SHA256):
        raise PanelError(
            "pinned reference include closure differs from the published RepoMeta "
            "(files=%d bytes=%d sha256=%s)" %
            (len(included), included_bytes, included_sha))

    if fetch_file_func is None:
        def fetch_file_func(path):
            return hfmeta.fetch_file(
                repo_id, path, repo_type="dataset", revision=revision)

    documents: Dict[str, Dict[str, Any]] = {}
    metadata_rows: List[Dict[str, Any]] = []
    for path, expected in sorted(_BRANDON_METADATA.items()):
        expected_size, expected_sha = expected
        if meta_files.get(path) != expected_size:
            raise PanelError("RepoMeta has unexpected bytes for %s" % path)
        try:
            raw = fetch_file_func(path)
        except Exception as exc:
            raise PanelError("cannot fetch pinned metadata %s: %s" % (path, exc))
        if not isinstance(raw, bytes):
            raise PanelError("metadata fetch for %s did not return bytes" % path)
        actual_sha = _sha(raw)
        if len(raw) != expected_size or actual_sha != expected_sha:
            raise PanelError("pinned metadata bytes differ for %s" % path)
        documents[path] = _json_bytes(raw, path)
        metadata_rows.append(
            {"path": path, "bytes": len(raw), "sha256": actual_sha})

    capture = documents["capture-receipt.json"]
    teacher_seal, teacher_convention = _verify_seal(
        capture, "Brandon teacher capture")
    if (teacher_seal != BRANDON_TEACHER_RECEIPT_SHA256
            or capture.get("schema") != "quant-pipeline.glm53-logit-capture.v1"
            or capture.get("capture_role") != "bf16_teacher"):
        raise PanelError("teacher capture identity, schema, or role is not pinned")

    dataset = documents["dataset-manifest.json"]
    dataset_seal = _verify_legacy_named_seal(
        dataset, "dataset_sha256",
        "61faf80c9a8c7bb60bcefbfd6208c7f63609ddc4089798c2f317bfcabc8569a4",
        "Brandon dataset manifest")
    if (dataset.get("schema")
            != "quant-pipeline.glm53-bf16-teacher-logits-dataset.v1"
            or dataset.get("repo_id") != repo_id
            or dataset.get("teacher_capture_receipt_sha256") != teacher_seal):
        raise PanelError("teacher dataset manifest identity is not pinned")

    backend = documents["backend.json"]
    backend_seal = _verify_legacy_named_seal(
        backend, "backend_identity_sha256", BRANDON_BACKEND_SHA256,
        "Brandon backend identity")
    if backend.get("schema") != "quant-pipeline.glm53-teacher-backend-identity.v1":
        raise PanelError("teacher backend schema is not pinned")

    panel_receipt = documents["calibration/panel-v1/panel.receipt.json"]
    panel_seal, panel_convention = _verify_seal(
        panel_receipt, "Brandon token panel")
    if (panel_seal != BRANDON_PANEL_RECEIPT_SHA256
            or panel_receipt.get("schema") != ARTIFACT_RECEIPT_SCHEMA
            or panel_receipt.get("final_windows") != 25
            or panel_receipt.get("final_prediction_positions") != 51175):
        raise PanelError("teacher token-panel receipt is not the pinned final25 panel")

    shared = (
        ("backend_identity_sha256", backend_seal),
        ("token_panel_receipt_sha256", panel_seal),
        ("prediction_positions", 51175),
        ("logits_dtype", "float32"),
        ("vocab_size", 154880),
    )
    for field, expected in shared:
        if capture.get(field) != expected or dataset.get(field) != expected:
            raise PanelError("capture and dataset disagree on %s" % field)
    if (capture.get("model_revision") != backend.get("model_revision")
            or dataset.get("model_revision") != backend.get("model_revision")):
        raise PanelError("capture, dataset, and backend model revisions disagree")

    row_keys = {
        "attention_mask_sha256", "bytes", "document_id", "domain", "path",
        "prediction_positions", "role", "sha256", "token_ids_sha256",
        "window_id",
    }
    capture_rows = capture.get("logit_files")
    dataset_rows = dataset.get("logit_files")
    if not isinstance(capture_rows, list) or not isinstance(dataset_rows, list):
        raise PanelError("teacher logit_files must be arrays")
    normalized_capture: List[Dict[str, Any]] = []
    capture_prefix = (
        "/workspace/artifacts/evaluation/glm53-teacher-final-ep4/logits/")
    for index, source in enumerate(capture_rows):
        if not isinstance(source, dict) or set(source) != row_keys:
            raise PanelError("capture logit row %d has unexpected fields" % index)
        row = dict(source)
        if not isinstance(row["path"], str) or not row["path"].startswith(capture_prefix):
            raise PanelError("capture logit row %d has unexpected path" % index)
        basename = row["path"][len(capture_prefix):]
        if "/" in basename or not basename:
            raise PanelError("capture logit row %d path is not a single file" % index)
        row["path"] = "logits/" + basename
        normalized_capture.append(row)

    normalized_dataset: List[Dict[str, Any]] = []
    for index, source in enumerate(dataset_rows):
        if not isinstance(source, dict) or set(source) != row_keys:
            raise PanelError("dataset logit row %d has unexpected fields" % index)
        row = dict(source)
        row["path"] = _safe_rel(row["path"], "dataset logit row %d path" % index)
        normalized_dataset.append(row)
    if normalized_capture != normalized_dataset:
        raise PanelError("capture and dataset logit declarations differ")
    if len(normalized_dataset) != 25:
        raise PanelError("teacher reference must contain exactly 25 final windows")

    logits: List[Dict[str, Any]] = []
    for index, row in enumerate(normalized_dataset):
        window_id = "final-%04d" % index
        path = "logits/window-%04d.safetensors" % index
        if (row["window_id"] != window_id or row["path"] != path
                or row["role"] != "final"
                or row["prediction_positions"] != 2047
                or row["bytes"] != BRANDON_LOGIT_BYTES):
            raise PanelError("logit row %d is not the pinned final25 contract" % index)
        for field in ("sha256", "token_ids_sha256", "attention_mask_sha256"):
            _hex(row[field], "logit row %d %s" % (index, field))
        if meta_files.get(path) != row["bytes"]:
            raise PanelError("RepoMeta size differs for %s" % path)
        logits.append(row)

    artifacts = panel_receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 667:
        raise PanelError("token-panel receipt must declare exactly 667 artifacts")
    artifact_paths = set()
    artifact_prefix = "/workspace/artifacts/dataset/"
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict) or set(row) != {"bytes", "path", "sha256"}:
            raise PanelError("token-panel artifact %d has unexpected fields" % index)
        source_path = row["path"]
        if not isinstance(source_path, str) or not source_path.startswith(artifact_prefix):
            raise PanelError("token-panel artifact %d has unexpected path" % index)
        path = _safe_rel(
            source_path[len(artifact_prefix):],
            "token-panel artifact %d path" % index)
        if path in artifact_paths:
            raise PanelError("duplicate token-panel artifact path %s" % path)
        artifact_paths.add(path)
        size = _integer(row["bytes"], "token-panel artifact %s bytes" % path,
                        minimum=1)
        _hex(row["sha256"], "token-panel artifact %s sha256" % path)
        if meta_files.get(path) != size:
            raise PanelError("RepoMeta size differs for token-panel artifact %s" % path)

    total_logits_bytes = sum(row["bytes"] for row in logits)
    result = {
        "schema": "malaiwah.brandon-reference-manifest.v1",
        "repo_id": repo_id,
        "revision": revision,
        "reference_ref": BRANDON_REFERENCE_REF,
        "role": "bf16_teacher",
        "capture_receipt_sha256": teacher_seal,
        "capture_receipt_seal_convention": teacher_convention,
        "dataset_manifest_sha256": dataset_seal,
        "backend_identity_sha256": backend_seal,
        "token_panel_receipt_sha256": panel_seal,
        "token_panel_receipt_seal_convention": panel_convention,
        "model_revision": backend["model_revision"],
        "contexts": len(logits),
        "positions_per_context": 2047,
        "prediction_positions": 51175,
        "logits_dtype": "float32",
        "logits": logits,
        "total_declared_logits_bytes": total_logits_bytes,
        "include": list(BRANDON_REFERENCE_INCLUDE),
        "included_repo_files": included,
        "included_repo_file_count": len(included),
        "included_repo_bytes": included_bytes,
        "included_repo_manifest_sha256": included_sha,
        "metadata_files": metadata_rows,
        "manifest_sha256": "",
    }
    if total_logits_bytes != 31703946000:
        raise PanelError("teacher logit liability differs from the pinned total")
    result["manifest_sha256"] = _sha(_canonical(result))
    return result
