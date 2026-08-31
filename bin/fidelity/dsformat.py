"""On-disk format primitives for `malaiwah.fidelity-dataset.v1`.

Everything in this module is pure, deterministic, OFFLINE and stdlib-only, and
imports nothing heavier than `hashlib`.  It is the single implementation of the
five frozen digest preimages (spec section 5.1), the `capture_content_digest`
(5.2), the self-blanked seal (5.3), the `checksums.txt` chain (5.4) and the path
rules (2.1).  Every other module in this package -- and the comparator, which
needs torch -- derives its digests from here, because two implementations of a
hash function is two chances to disagree.

The seal recipe is imported verbatim from `fidelity.common`, which is the same
four-line recipe `registry/tools/registry_lib.py` documents to contributors.  A
stranger verifies our datasets with `python3 -c` and no imports from us.

Reference: docs/FIDELITY-DATASET-SPEC.md
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .common import canonical_json, seal, sha256_file, sha256_hex, verify_seal  # noqa: F401

# ---------------------------------------------------------------------------
# Format identity
# ---------------------------------------------------------------------------

DATASET_SCHEMA = "malaiwah.fidelity-dataset.v1"
CAPTURE_MANIFEST_SCHEMA = "malaiwah.fidelity-capture-manifest.v1"
PANEL_SCHEMA = "malaiwah.fidelity-panel-binding.v1"
HEAD_SCHEMA = "malaiwah.fidelity-head-identity.v1"
RUNTIME_SCHEMA = "malaiwah.fidelity-capture-runtime.v1"
REMAP_SCHEMA = "malaiwah.fidelity-path-remap.v1"
QUALIFICATION_SCHEMA = "malaiwah.fidelity-replay-qualification.v1"
RECEIPT_SCHEMA = "malaiwah.fidelity-comparison-receipt.v1"
VALIDATION_SCHEMA = "malaiwah.fidelity-structural-validation.v1"

FORMAT_VERSION = 1

MANIFEST_NAME = "fidelity-dataset.json"
CHECKSUMS_NAME = "checksums.txt"
SEAL_FIELD = "dataset_sha256"
SEAL_METHOD = "self_blanked_canonical_json_sha256"

#: The two files `checksums.txt` cannot cover.  This exclusion is what breaks
#: the cycle: the manifest names checksums.txt by digest, checksums.txt covers
#: every other file, and the manifest covers itself by self-blanking.
SEAL_EXCLUDES = (CHECKSUMS_NAME, MANIFEST_NAME)

ROLES = ("root", "quant", "derived")
FORMS = ("hidden", "logit")
STRUCTURAL_STATUS = ("draft", "structural", "sealed")

#: registry submission.schema.json's enum, unchanged.  Adding a lane is a
#: registry schema change on purpose (spec section 9.3).
LANES = ("sealed-ep8", "streaming", "local-mps", "local-cuda-budget", "other")

SEMANTIC_POINTS = (
    "after_final_rmsnorm_before_lm_head",
    "lm_head_output_before_sampling",
    "live_lm_head_output_before_sampling",
)

#: v1 normative tensor keys.  `hidden` is accepted from a pre-v1 artifact and
#: rewritten on ingest with a disclosure (REC-2).
TENSOR_KEY_HIDDEN = "hidden_states"
TENSOR_KEY_LOGIT = "logits"
LEGACY_HIDDEN_KEYS = ("hidden",)

HEAD_SOURCES = ("native", "artifact_dequantized", "shared_reference_head", "unknown")

#: DET-D1/DET-D2.  Container digests are absent on purpose: stream_score writes
#: `cold_run` into the safetensors __metadata__, so a whole-file digest differs
#: between bitwise-identical runs.
CONTENT_EVIDENCE_KINDS = (
    "hidden_state_tensor_sha256",
    "logits_tensor_sha256",
    "tokenwise_kld_sha256",
    "sealed_tokenwise_digest",
)
WEAK_EVIDENCE_KINDS = (
    "receipt_file_sha256",
    "container_or_archive_sha256",
    "run_mean_equality_only",
    "none",
)

#: Festr's exact bucket edges (spec section 10.3).
CONTEXT_DEPTH_BUCKETS = (
    ("0000-0255", 0, 256),
    ("0256-0511", 256, 512),
    ("0512-1023", 512, 1024),
    ("1024-1535", 1024, 1536),
    ("1536-2046", 1536, 2047),
)

#: The T1 constant: np.save of 51,175 float64 zeros (spec section 10.4).
ZERO_TOKENWISE_BYTES_51175 = 409528
ZERO_TOKENWISE_SHA256_51175 = (
    "3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17"
)


class FormatError(Exception):
    """A refusal.  Carries a machine-readable reason code (spec section 10.1)."""

    def __init__(self, code: str, message: str, path: str = ""):
        self.code = code
        self.message = message
        self.path = path
        super().__init__("%s: %s%s" % (code, message, (" [%s]" % path) if path else ""))


# ---------------------------------------------------------------------------
# The five frozen digest preimages (spec section 5.1)
# ---------------------------------------------------------------------------


def file_sha256(path: str) -> str:
    """Preimage 1: the whole file bytes.  The container digest.

    This is what `checksums.txt` carries.  It is NEVER determinism evidence
    (DET-D2) and never a head identity (HEAD-IDENT).
    """
    return sha256_file(path)


def read_safetensors_header(path: str) -> Tuple[int, Dict[str, Any]]:
    """Return (header_length, header_dict) for a safetensors file."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise FormatError("bad_tensor_file", "file shorter than a safetensors header", path)
        header_len = struct.unpack("<Q", raw)[0]
        blob = handle.read(header_len)
        if len(blob) != header_len:
            raise FormatError("bad_tensor_file", "truncated safetensors header", path)
    try:
        header = json.loads(blob.decode("utf-8"))
    except ValueError as exc:
        raise FormatError("bad_tensor_file", "unparseable safetensors header: %s" % exc, path)
    return header_len, header


def payload_sha256(path: str) -> str:
    """Preimage 2: the safetensors DATA REGION, header skipped.

    Byte-identical to `engines/tools/hidden_replay.py::payload_sha256` and to
    `engines/stage_campaign.sh` L4's payload-shas.json: read the `<Q` header length at
    offset 0, seek past `8 + header_len`, hash the rest.  Survives
    `__metadata__` churn.
    """
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_len)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_content_sha256(path: str, key: str) -> str:
    """Preimage 3: the raw little-endian bytes of the NAMED TENSOR only.

    Container-independent.  Equivalent to
    `hidden_replay.tensor_content_sha256(tensor)` (which views bf16 as uint16
    and hashes `numpy.tobytes()`) without needing torch: safetensors already
    stores exactly those bytes, contiguously, at `data_offsets`.
    """
    header_len, header = read_safetensors_header(path)
    if key not in header:
        keys = sorted(k for k in header if k != "__metadata__")
        raise FormatError(
            "bad_tensor_file",
            "tensor key %r absent; file carries %s" % (key, keys),
            path,
        )
    start, stop = header[key]["data_offsets"]
    base = 8 + header_len
    digest = hashlib.sha256()
    remaining = stop - start
    with open(path, "rb") as handle:
        handle.seek(base + start)
        while remaining > 0:
            block = handle.read(min(remaining, 1 << 22))
            if not block:
                raise FormatError("bad_tensor_file", "truncated tensor payload", path)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def token_ids_json_sha256(ids: Sequence[int]) -> str:
    """Preimage 4 (NORMATIVE): compact separators -- kimi-k3's preimage, adopted.

    `sha256(json.dumps(ids, separators=(",",":")).encode("utf-8"))`
    """
    return hashlib.sha256(
        json.dumps(list(ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def token_ids_json_sha256_legacy(ids: Sequence[int]) -> str:
    """Our historical preimage: `json.dumps` defaults (`", "` separators).

    Carried as `token_ids_sha256_legacy` so sealed pre-v1 receipts still
    cross-check.  Never normative.
    """
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


def suite_token_hash_sha256(per_record_hex: Sequence[str]) -> str:
    """Preimage 5 (NORMATIVE): newline join -- kimi-k3's preimage, adopted.

    `sha256("\\n".join(per_record_hex).encode("ascii"))`, records in ascending
    index order.
    """
    return hashlib.sha256("\n".join(per_record_hex).encode("ascii")).hexdigest()


def suite_token_hash_sha256_legacy(per_record_hex: Sequence[str]) -> str:
    """Our historical aggregate: empty-string join.  Never normative."""
    return hashlib.sha256("".join(per_record_hex).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# capture_content_digest (spec section 5.2)
# ---------------------------------------------------------------------------


def capture_content_digest(records: Iterable[Dict[str, Any]]) -> str:
    """The manifest-INDEPENDENT identity of what was captured.

        sha256("\\n".join("%d:%s" % (index, tensor_content_sha256)))

    over index-sorted records.  Independent of every container, every manifest
    serialization and every piece of metadata.  This is the value compared for
    the A == B self-compare short-circuit, the value that appears in
    `determinism.evidence_hashes`, and the value a card and a registry row cite.
    """
    rows = sorted(records, key=lambda record: int(record["index"]))
    seen = set()
    parts = []
    for record in rows:
        index = int(record["index"])
        if index in seen:
            raise FormatError("digest_mismatch", "duplicate record index %d" % index)
        seen.add(index)
        content = record.get("tensor_content_sha256")
        if not content:
            raise FormatError(
                "digest_mismatch",
                "record %d has no tensor_content_sha256; a container digest is "
                "not an identity (spec 5.2)" % index,
            )
        parts.append("%d:%s" % (index, content))
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Path rules (spec section 2.1)
# ---------------------------------------------------------------------------

#: PATH-3: `..` is permitted only inside compat/, and only where it still
#: resolves inside the root.
PARENT_ALLOWED_PREFIXES = ("compat/",)


def check_relpath(value: str, owner: str = "", allow_parent: bool = False) -> str:
    """PATH-1/PATH-3: relative, normalizes to inside the root, no drive letters.

    Returns the normalized path.  Raises FormatError('path_escape') otherwise.
    """
    if not isinstance(value, str) or not value:
        raise FormatError("path_escape", "path must be a non-empty string", owner)
    if value.startswith("/") or value.startswith("\\") or ":" in value.split("/")[0]:
        raise FormatError("path_escape", "absolute path %r" % value, owner)
    if "://" in value:
        raise FormatError("path_escape", "URI %r is not a dataset path" % value, owner)
    if ".." in value.split("/") and not allow_parent:
        raise FormatError(
            "path_escape",
            "%r uses '..'; permitted only under %s or on the k3-adopted "
            "runtime_manifest field (PATH-3)"
            % (value, "/".join(PARENT_ALLOWED_PREFIXES)),
            owner,
        )
    normalized = os.path.normpath(value)
    if normalized == ".":
        raise FormatError("path_escape", "%r is not a file path" % value, owner)
    if normalized.startswith("..") and not allow_parent:
        raise FormatError("path_escape", "%r escapes the dataset root" % value, owner)
    return normalized.replace(os.sep, "/")


def resolve_inside(root: str, relpath: str, owner: str = "") -> str:
    """Join and prove the result stays inside `root`.  PATH-1."""
    root_abs = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_abs, relpath))
    if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
        raise FormatError("path_escape", "%r resolves outside the root" % relpath, owner)
    return candidate


#: Field names known to carry host-local directories upstream.  PATH-2 requires
#: them to be stripped when a receipt is copied into `upstream/`.
HOST_PATH_FIELDS = (
    "packed_root",
    "output_root",
    "checkpoint_root",
    "capture_chunk_dir",
    "corpus",
    "path",
    "model_path",
    "weight_source",
)


def strip_host_paths(doc: Any, stripped: Optional[List[str]] = None, prefix: str = "") -> Tuple[Any, List[str]]:
    """PATH-2: remove host-local directory fields, recording what was removed.

    Returns (stripped_copy, names).  A receipt copied into `upstream/` MUST go
    through this and record `stripped_fields[]`.  This is the rule our data loss
    taught -- `packed_root: /home/jl_fs/glm53-k6/out-k6` -- made mechanical.
    """
    if stripped is None:
        stripped = []
    if isinstance(doc, dict):
        out = {}
        for key, value in doc.items():
            here = "%s%s" % (prefix, key)
            if key in HOST_PATH_FIELDS and isinstance(value, str) and (
                value.startswith("/") or value.startswith("~") or "\\" in value
            ):
                stripped.append(here)
                out[key] = None
                continue
            child, _ = strip_host_paths(value, stripped, here + ".")
            out[key] = child
        return out, stripped
    if isinstance(doc, list):
        out_list = []
        for item in doc:
            child, _ = strip_host_paths(item, stripped, prefix.rstrip(".") + "[].")
            out_list.append(child)
        return out_list, stripped
    return doc, stripped


# ---------------------------------------------------------------------------
# checksums.txt (spec section 5.4) -- adopted verbatim from kimi-k3
# ---------------------------------------------------------------------------


#: Files that must never be sealed into a dataset or uploaded with one. This is a
#: REFUSAL list, not a filter: `iter_dataset_files` hashes whatever it walks, so a stray
#: credential under a dataset root was hashed into the published `checksums.txt` and
#: sealed into the manifest before `upload_folder` -- which passed no ignore_patterns --
#: sent the file itself. `measure_cloud.py` writes `.hf_token` into a run directory and
#: the repo's own .gitignore documents the crash window where it survives, so "the token
#: is a sibling of the dataset dir, not inside it" is a property of one directory layout
#: rather than a property of the publisher.
#:
#: Deliberately NOT here: anything matching `*token*`. `panel/tokens/context-0000.json`
#: is required dataset payload and `tokenizer.json` is required model payload; a pattern
#: that broad would have silently stripped them and published an unloadable artifact.
CREDENTIAL_FILE_PATTERNS = (
    ".hf_token", "hf_token", ".env", ".netrc", ".npmrc", ".git-credentials",
    "id_rsa", "id_ed25519", "credentials.json", "service-account.json",
)
CREDENTIAL_DIR_NAMES = (".secrets", ".ssh", ".aws", ".gnupg", ".git")
CREDENTIAL_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def looks_like_a_credential(relpath: str) -> bool:
    """Would publishing this file disclose a secret? Basename and directory, both."""
    parts = relpath.split("/")
    if any(part in CREDENTIAL_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1]
    if name in CREDENTIAL_FILE_PATTERNS or name in CREDENTIAL_DIR_NAMES:
        return True
    return any(name.endswith(sfx) for sfx in CREDENTIAL_SUFFIXES)


def iter_dataset_files(root: str, exclude: Sequence[str] = SEAL_EXCLUDES) -> List[str]:
    """Every regular file under `root`, as sorted relative POSIX paths.

    PATH-4: a symlink anywhere in the tree is a hard error.
    """
    excluded = set(exclude)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(dirnames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                raise FormatError("symlink", "symlinked directory %r (PATH-4)" % rel)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                raise FormatError("symlink", "symlink %r (PATH-4)" % rel)
            if rel in excluded:
                continue
            if looks_like_a_credential(rel):
                raise FormatError(
                    "credential_in_tree",
                    "%r is under the dataset root. Sealing it would hash a secret into the "
                    "published checksums.txt and publishing would disclose it. Refusing rather "
                    "than filtering: a file quietly dropped from the upload but still listed in "
                    "checksums.txt makes the published dataset unverifiable." % rel)
            found.append(rel)
    return sorted(found)


def format_checksums(entries: Sequence[Tuple[str, str]]) -> str:
    """`<64-hex><space><space><relpath>`, sorted by path, LF endings.

    `sha256sum --check`-compatible: a reviewer with no tooling of ours verifies
    the payload with one coreutils command.
    """
    lines = ["%s  %s" % (digest, path) for path, digest in sorted(entries)]
    return "\n".join(lines) + ("\n" if lines else "")


def parse_checksums(text: str) -> Dict[str, str]:
    out = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(line) < 67 or line[64:66] != "  ":
            raise FormatError(
                "seal_failed",
                "checksums.txt line %d is not `<sha256>  <path>`" % lineno,
            )
        digest, path = line[:64], line[66:]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise FormatError("seal_failed", "checksums.txt line %d: bad digest" % lineno)
        if path in out:
            raise FormatError("seal_failed", "checksums.txt lists %r twice" % path)
        # The digest was validated and the PATH was not, so a remote checksums.txt could
        # name `../../engines/tools/stream_score.py` or an absolute path and the fetcher joined
        # it onto the download directory. `write_checksums` derives every entry from
        # `iter_dataset_files`, which is an os.walk + relpath and hard-errors on symlinks,
        # so a legitimately generated file can never contain one: rejecting the whole list
        # here costs nothing and closes the class for every caller at once.
        try:
            check_relpath(path, owner="checksums.txt line %d" % lineno)
        except FormatError as exc:
            raise FormatError("seal_failed",
                              "checksums.txt line %d names %r, which does not stay inside the "
                              "dataset root (%s)" % (lineno, path, exc.message))
        out[path] = digest
    return out


def write_checksums(root: str) -> str:
    """Hash every file except the two exclusions; write checksums.txt.

    Returns the sha256 of `checksums.txt` itself -- the value the manifest
    carries as `seal.checksums_sha256`.
    """
    entries = []
    for rel in iter_dataset_files(root):
        entries.append((rel, sha256_file(os.path.join(root, rel))))
    text = format_checksums(entries)
    path = os.path.join(root, CHECKSUMS_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return sha256_file(path)


def verify_checksums(root: str, allow_partial: bool = False,
                     partial_ok_prefixes: Sequence[str] = ("capture/",)) -> Dict[str, Any]:
    """SEAL-1(c) + SEAL-2.  Returns a report; raises on a hard refusal."""
    path = os.path.join(root, CHECKSUMS_NAME)
    if not os.path.isfile(path):
        raise FormatError("seal_failed", "checksums.txt is missing")
    with open(path, "r", encoding="utf-8") as handle:
        listed = parse_checksums(handle.read())
    present = set(iter_dataset_files(root))
    unlisted = sorted(present - set(listed))
    missing = sorted(set(listed) - present)
    if unlisted:
        raise FormatError(
            "unlisted_file",
            "%d file(s) present but not in checksums.txt: %s"
            % (len(unlisted), unlisted[:5]),
        )
    if missing:
        blocking = [m for m in missing
                    if not (allow_partial and m.startswith(tuple(partial_ok_prefixes)))]
        if blocking:
            raise FormatError(
                "missing_file",
                "%d listed file(s) absent: %s%s"
                % (len(blocking), blocking[:5],
                   "" if allow_partial else " (--allow-partial covers capture tensors only)"),
            )
    bad = []
    for rel, want in sorted(listed.items()):
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        got = sha256_file(full)
        if got != want:
            bad.append(rel)
    if bad:
        raise FormatError(
            "tensor_mismatch" if any(b.startswith("capture/") for b in bad) else "seal_failed",
            "%d file(s) do not match checksums.txt: %s" % (len(bad), bad[:5]),
        )
    return {
        "listed": len(listed),
        "verified": len(listed) - len(missing),
        "missing": missing,
        "unlisted": unlisted,
    }


# ---------------------------------------------------------------------------
# Seal helpers
# ---------------------------------------------------------------------------


def seal_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Self-blank `dataset_sha256` and seal.  Spec section 5.3."""
    return seal(manifest, SEAL_FIELD)


def verify_manifest_seal(manifest: Dict[str, Any]) -> bool:
    return verify_seal(manifest, SEAL_FIELD)


def seal_receipt(doc: Dict[str, Any], field: str = "receipt_sha256") -> Dict[str, Any]:
    return seal(doc, field)


def recompute_seal(doc: Dict[str, Any], field: str) -> str:
    body = dict(doc)
    body[field] = ""
    return sha256_hex(canonical_json(body))


def write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(root: str) -> Dict[str, Any]:
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FormatError("bad_schema", "%s is missing" % MANIFEST_NAME, root)
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise FormatError("bad_schema", "%s is not an object" % MANIFEST_NAME, root)
    schema = manifest.get("schema")
    if schema != DATASET_SCHEMA:
        # Dispatch on the exact string and refuse unknown ones rather than
        # guess -- the registry_add.py rule (spec section 1.3).
        raise FormatError(
            "bad_schema",
            "schema is %r; this tool reads only %r" % (schema, DATASET_SCHEMA),
            root,
        )
    if manifest.get("format_version") != FORMAT_VERSION:
        raise FormatError(
            "bad_schema",
            "format_version is %r, expected %d"
            % (manifest.get("format_version"), FORMAT_VERSION),
            root,
        )
    return manifest


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def bucket_for_position(row: int) -> Optional[str]:
    """Festr's exact context-depth buckets (spec section 10.3)."""
    for name, low, high in CONTEXT_DEPTH_BUCKETS:
        if low <= row < high:
            return name
    return None


def divisors_hint(vocab_size: int, limit: int = 6) -> List[int]:
    """Working `--vocab-chunk` values, largest first.

    154,880 is NOT divisible by kimi-k3's default 10,240; 9,680 is the value to
    ship in a k3-compat README (spec section 10.2).
    """
    out = []
    for candidate in range(min(vocab_size, 32768), 0, -1):
        if vocab_size % candidate == 0:
            out.append(candidate)
            if len(out) >= limit:
                break
    return out


def stats_block(values) -> Dict[str, float]:
    """`{mean, median, p95, p99, p99_9, max}` -- Festr's block, field for field."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise FormatError("geometry_mismatch", "empty tokenwise array")
    return {
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "p99_9": float(np.quantile(array, 0.999)),
        "max": float(array.max()),
    }
