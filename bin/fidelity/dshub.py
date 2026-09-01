"""Fetch / publish a fidelity dataset on the Hugging Face Hub.

Stdlib by default (urllib against the public resolve endpoints), with
`huggingface_hub` used when it is importable and a token is needed.  The token
is registered with `fidelity.common.register_secret()` the moment it is read,
BEFORE anything can print, so a traceback or a debug dump cannot leak it.

Fetch is digest-driven: the manifest names `checksums.txt` by digest,
`checksums.txt` names every other file, and `verify` refuses anything that does
not match.  A partial fetch is therefore a *stated* condition (`--allow-partial`,
capture tensors only), never a silent one.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import common
from . import dsformat as F

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
USER_AGENT = "malaiwah-fidelity-dataset/1.0"


class HubError(Exception):
    pass


def read_token(path_or_env: Optional[str] = None) -> Optional[str]:
    """Read an explicit protected token file, or standard client sources.

    An explicit path is authoritative: it is never silently ignored in favor
    of ambient credentials.
    """
    token = None
    if path_or_env is not None:
        if not hasattr(os, "O_NOFOLLOW"):
            raise HubError(
                "explicit token file REFUSED: O_NOFOLLOW is unavailable")
        flags = os.O_RDONLY | os.O_NOFOLLOW
        try:
            fd = os.open(path_or_env, flags)
        except OSError as exc:
            raise HubError("explicit token file REFUSED: %s" % exc)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise HubError("explicit token file is not regular")
            if info.st_uid != os.getuid():
                raise HubError("explicit token file owner differs from current uid")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise HubError("explicit token file mode must be exactly 0600")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                token = handle.read().strip()
        finally:
            if fd >= 0:
                os.close(fd)
        if not token:
            raise HubError("explicit token file is empty")
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")
        if not token:
            cached = os.path.expanduser("~/.cache/huggingface/token")
            if os.path.isfile(cached):
                with open(cached, "r", encoding="utf-8") as handle:
                    token = handle.read().strip()
    common.register_secret(token)
    return token or None


def parse_ref(ref: str) -> Tuple[str, str]:
    """`hf://repo[@rev]` or `repo[@rev]` -> (repo, revision)."""
    text = ref[len("hf://"):] if ref.startswith("hf://") else ref
    if "@" in text:
        repo, revision = text.rsplit("@", 1)
        return repo, revision
    return text, "main"


# Drop `Authorization` when a redirect leaves the original origin.  The class
# used to live here alone; the 2026-08-31 peer review found hfmeta and the
# truncation fetcher still on urllib's default handler (which forwards the
# bearer across hosts, and HF `/resolve/` 302s to CDN hosts), so the one
# correct implementation moved to `common` and every client shares it.
_NoCrossHostAuth = common.make_no_cross_origin_auth_handler()

_OPENER = urllib.request.build_opener(_NoCrossHostAuth())


def _host(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _endpoint_host() -> str:
    return _host(HF_ENDPOINT)


def _get(url: str, token: Optional[str] = None, binary: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if token:
        # Only ever to the configured endpoint. HF_ENDPOINT is an environment variable, so
        # a stale export or a proxy silently redirected the write-scoped token off-Hub;
        # and the token was attached to every GET including reads of PUBLIC datasets,
        # which registry_client is explicit about never doing.
        if _host(url) == _endpoint_host():
            request.add_header("Authorization", "Bearer %s" % token)
        else:
            raise HubError("refusing to send the Hugging Face token to %s (the configured "
                           "endpoint is %s)" % (_host(url) or "an unparseable host",
                                                _endpoint_host()))
    try:
        with _OPENER.open(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        err = HubError("HTTP %s for %s" % (exc.code, common.redact(url)))
        err.status = exc.code
        raise err
    except urllib.error.URLError as exc:
        raise HubError("network error for %s: %s" % (common.redact(url), exc.reason))
    return payload if binary else payload.decode("utf-8")

def _read_remote_exact(
        url: str, expected_bytes: int, expected_sha256: str, *,
        token: Optional[str] = None, capture: bool = False,
        capture_limit: int = 16 * 1024 * 1024) -> Optional[bytes]:
    if (isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int) or expected_bytes < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None):
        raise HubError("remote byte identity is malformed")
    if capture and expected_bytes > capture_limit:
        raise HubError("remote evidence exceeds bounded capture size")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if token:
        if _host(url) != _endpoint_host():
            raise HubError(
                "refusing to send the Hugging Face token across origins")
        request.add_header("Authorization", "Bearer %s" % token)
    digest = hashlib.sha256()
    total = 0
    body = bytearray() if capture else None
    try:
        with _OPENER.open(request, timeout=60) as response:
            status = response.getcode()
            if status != 200:
                raise HubError(
                    "HTTP %s while streaming immutable public evidence" % status)
            while True:
                chunk = response.read(
                    min(1024 * 1024, expected_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_bytes:
                    raise HubError(
                        "immutable public member exceeds its exact byte bound")
                digest.update(chunk)
                if body is not None:
                    body.extend(chunk)
    except HubError:
        raise
    except urllib.error.HTTPError as exc:
        raise HubError(
            "HTTP %s while streaming immutable public evidence" % exc.code)
    except urllib.error.URLError as exc:
        raise HubError(
            "network error while streaming immutable public evidence: %s"
            % exc.reason)
    if total != expected_bytes or digest.hexdigest() != expected_sha256:
        raise HubError(
            "immutable public member differs from verified source archive")
    return bytes(body) if body is not None else None


def fetch_exact_bytes(
        url: str, expected_bytes: int, expected_sha256: str, *,
        token: Optional[str] = None,
        max_bytes: int = 16 * 1024 * 1024) -> bytes:
    """Fetch small evidence with exact byte/hash caps and no unbounded read."""
    body = _read_remote_exact(
        url, expected_bytes, expected_sha256, token=token, capture=True,
        capture_limit=max_bytes)
    assert body is not None
    return body


def verify_remote_dataset_exact(
        repo: str, revision: str, records: Mapping[str, Mapping[str, Any]], *,
        expected_dataset_sha256: str, max_total_bytes: int,
        repo_type: str = "datasets") -> Dict[str, Any]:
    """Stream and discard one immutable public dataset, exact archive bytes only."""
    if (re.fullmatch(r"[0-9a-f]{40}", str(revision)) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(expected_dataset_sha256)) is None
            or isinstance(max_total_bytes, bool)
            or not isinstance(max_total_bytes, int)
            or max_total_bytes <= 0
            or not isinstance(records, Mapping)
            or not records):
        raise HubError("remote dataset verification contract is malformed")
    normalized = {}
    total = 0
    for relpath, record in records.items():
        try:
            relpath = F.check_relpath(
                relpath, owner="verify_remote_dataset_exact")
        except F.FormatError as exc:
            raise HubError(str(exc))
        if not isinstance(record, Mapping):
            raise HubError("remote dataset member identity is malformed")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None):
            raise HubError("remote dataset member identity is malformed")
        normalized[relpath] = (size, digest)
        total += size
        if total > max_total_bytes:
            raise HubError("remote dataset exceeds its aggregate byte bound")
    if F.MANIFEST_NAME not in normalized or F.CHECKSUMS_NAME not in normalized:
        raise HubError("remote dataset proof lacks manifest or checksums")
    priority = [F.MANIFEST_NAME, F.CHECKSUMS_NAME]
    ordered = priority + sorted(set(normalized) - set(priority))
    for relpath in ordered:
        size, digest = normalized[relpath]
        _read_remote_exact(
            resolve_url(repo, revision, relpath, repo_type),
            size, digest, token=None)
    return {
        "dataset_sha256": expected_dataset_sha256,
        "files_verified": len(normalized),
        "bytes_verified": total,
        "bounded_streaming": True,
    }


def list_files(repo: str, revision: str = "main", token: Optional[str] = None,
               repo_type: str = "datasets") -> List[Dict[str, Any]]:
    url = "%s/api/%s/%s/tree/%s?recursive=1" % (
        HF_ENDPOINT, repo_type, urllib.parse.quote(repo, safe="/"),
        urllib.parse.quote(revision, safe=""))
    rows = json.loads(_get(url, token))
    return [row for row in rows if row.get("type") == "file"]


def resolve_url(repo: str, revision: str, path: str, repo_type: str = "datasets") -> str:
    prefix = "" if repo_type == "models" else "%s/" % repo_type
    return "%s/%s%s/resolve/%s/%s" % (
        HF_ENDPOINT, prefix, repo, urllib.parse.quote(revision, safe=""),
        urllib.parse.quote(path, safe="/"))


def fetch_dataset(ref: str, dest: str, *, token: Optional[str] = None,
                  allow_partial: bool = False, manifest_only: bool = False,
                  repo_type: str = "datasets") -> str:
    """Download a dataset to `dest`.  Returns the local root.

    Manifest and `checksums.txt` first, and the download really is digest-driven:
    every payload is hashed and compared to the listed digest BEFORE it lands, and
    every path is proved to stay inside `dest` before a byte is written.

    Neither was true.  `checksums.txt` comes from the remote repo and its paths were
    joined onto `dest` unchecked, so a line reading `<64 hex>  ../../../../engines/tools/
    stream_score.py` wrote there -- `os.path.join` also lets an ABSOLUTE entry win
    outright -- and `seal.checksums_file` from the remote manifest was a second such
    sink that fired even on the error path.  `validate`, `verify`, `compare` and the
    post-publish re-verify all reach this from a plain `hf://` argument, which is the
    documented way to look at somebody else's dataset.  The digests, meanwhile, were
    parsed and never used: bytes were written first and checked never, so the
    "digest-driven" claim in this docstring was decoration.
    """
    repo, revision = parse_ref(ref)
    os.makedirs(dest, exist_ok=True)

    manifest_bytes = _get(resolve_url(repo, revision, F.MANIFEST_NAME, repo_type),
                          token, binary=True)
    with open(os.path.join(dest, F.MANIFEST_NAME), "wb") as handle:
        handle.write(manifest_bytes)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema") != F.DATASET_SCHEMA:
        raise HubError("%s is not a %s (schema %r)"
                       % (ref, F.DATASET_SCHEMA, manifest.get("schema")))
    if manifest_only:
        return dest

    checksums_name = (manifest.get("seal") or {}).get("checksums_file") or F.CHECKSUMS_NAME
    try:
        checksums_name = F.check_relpath(checksums_name, owner="fetch_dataset")
    except F.FormatError as exc:
        raise HubError("%s: seal.checksums_file %r does not stay inside the dataset (%s)"
                       % (ref, checksums_name, exc))
    checksums_bytes = _get(resolve_url(repo, revision, checksums_name, repo_type),
                           token, binary=True)
    listed = F.parse_checksums(checksums_bytes.decode("utf-8"))

    # Refuse the WHOLE list before any I/O. One hostile entry condemns the fetch; it
    # must not be able to land the entries that precede it in sort order.
    for relpath in sorted(listed):
        try:
            F.check_relpath(relpath, owner="fetch_dataset/checksums")
            F.resolve_inside(dest, relpath, owner="fetch_dataset")
        except F.FormatError as exc:
            raise HubError("%s: checksums.txt lists %r, which does not stay inside the "
                           "download directory (%s). Nothing was written."
                           % (ref, relpath, exc))

    with open(F.resolve_inside(dest, checksums_name, owner="fetch_dataset"), "wb") as handle:
        handle.write(checksums_bytes)

    for relpath in sorted(listed):
        if allow_partial and relpath.startswith("capture/") \
                and relpath != "capture/manifest.json":
            continue
        payload = _get(resolve_url(repo, revision, relpath, repo_type), token, binary=True)
        want = listed[relpath]
        got = hashlib.sha256(payload).hexdigest()
        if want and got != want:
            raise HubError("%s: %s does not match the digest its own checksums.txt lists "
                           "(listed %s, downloaded %s). Nothing was written for this file."
                           % (ref, relpath, want[:16] + "...", got[:16] + "..."))
        target = F.resolve_inside(dest, relpath, owner="fetch_dataset")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(payload)
    return dest

def _strict_object(text: str, owner: str) -> Dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HubError("%s contains duplicate key %r" % (owner, key))
            result[key] = value
        return result

    try:
        value = json.loads(
            text, object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                HubError("%s contains non-finite JSON %s" % (owner, value))))
    except (TypeError, ValueError) as exc:
        raise HubError("%s is not strict JSON: %s" % (owner, exc))
    if not isinstance(value, dict):
        raise HubError("%s must be a JSON object" % owner)
    return value


def _expect_absent(url: str, token: Optional[str], owner: str) -> int:
    try:
        _get(url, token=token)
    except HubError as exc:
        if getattr(exc, "status", None) == 404:
            return 404
        raise HubError("%s absence is ambiguous: %s" % (owner, exc))
    raise HubError("%s already exists or collides with the destination" % owner)


def preflight_create(repo: str, token_file: str) -> Dict[str, Any]:
    """Prove a public dataset destination is absent without mutating the Hub."""
    if HF_ENDPOINT != "https://huggingface.co":
        raise HubError(
            "publication preflight requires exact https://huggingface.co endpoint")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*",
                    repo) is None:
        raise HubError("publication repository must be exact owner/name")
    token = read_token(token_file)
    whoami = _strict_object(
        _get(HF_ENDPOINT + "/api/whoami-v2", token=token),
        "authenticated principal response")
    principal = whoami.get("name")
    namespace = repo.split("/", 1)[0]
    authorization = None
    if isinstance(principal, str) and principal == namespace:
        authorization = {
            "basis": "user", "namespace": namespace, "role": "owner"}
    else:
        for org in whoami.get("orgs") or []:
            if not isinstance(org, dict) or org.get("name") != namespace:
                continue
            role = org.get("roleInOrg") or org.get("role")
            if role in ("write", "admin"):
                authorization = {
                    "basis": "organization", "namespace": namespace,
                    "role": role}
                break
    if not isinstance(principal, str) or not principal or authorization is None:
        raise HubError(
            "authenticated principal lacks exact write/admin namespace authority")

    probes = {}
    quoted = urllib.parse.quote(repo, safe="/")
    for kind in ("datasets", "models", "spaces"):
        url = "%s/api/%s/%s" % (HF_ENDPOINT, kind, quoted)
        probes[kind] = {
            "authenticated_status":
                _expect_absent(url, token, "authenticated %s/%s" % (kind, repo)),
            "anonymous_status":
                _expect_absent(url, None, "anonymous %s/%s" % (kind, repo)),
        }
    return common.seal({
        "schema": "fidelity.hf-publish-create-preflight.v1",
        "checked_at": common.utcnow(),
        "endpoint": "https://huggingface.co",
        "repository": repo,
        "repo_type": "dataset",
        "expected_destination_state": "absent",
        "authenticated_principal": principal,
        "authorization": authorization,
        "probes": probes,
        "mutation_performed": False,
    })



def _write_namespace_allowed(identity: Dict[str, Any], namespace: str) -> bool:
    """Fail-closed namespace authorization from the authenticated principal."""
    if identity.get("name") == namespace:
        return True
    for org in identity.get("orgs") or []:
        if not isinstance(org, dict) or org.get("name") != namespace:
            continue
        role = org.get("roleInOrg") or org.get("role")
        if role in ("admin", "write"):
            return True
    return False


_OBVIOUS_HF_TOKEN = re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{16,}(?![A-Za-z0-9])")
_PRIVATE_ABSOLUTE_PATHS = (
    b"/home/", b"/root/", b"/Users/", b"/private/", b"/tmp/",
    b"/var/tmp/", b"/workspace/", b"/mnt/c/Users/", b"file:///",
    b":\\Users\\", b":\\\\Users\\\\",
    b"\\/home\\/", b"\\/root\\/", b"\\/Users\\/", b"\\/private\\/",
    b"\\/tmp\\/", b"\\/workspace\\/",
)
_TEXT_FILE_SUFFIXES = (
    ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv", ".tsv",
    ".toml", ".ini", ".cfg", ".receipt",
)


def _textual_publish_member(relpath: str) -> bool:
    name = relpath.rsplit("/", 1)[-1].lower()
    return (name in ("checksums.txt", "manifest.json")
            or name.endswith(_TEXT_FILE_SUFFIXES))


def _scan_publish_member(path: str, relpath: str, token: str, *,
                         textual: bool) -> None:
    """Stream a prospective upload with overlap so secrets cannot straddle chunks."""
    token_bytes = token.encode("utf-8")
    overlap = max(len(token_bytes) - 1, 255)
    flags = os.O_RDONLY
    if not hasattr(os, "O_NOFOLLOW"):
        raise HubError("REFUSED to scan upload: O_NOFOLLOW is unavailable")
    flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise HubError("REFUSED to scan upload member %r: %s" % (relpath, exc))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise HubError(
                "REFUSED to publish non-regular member %r" % relpath)
        carry = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            window = carry + chunk
            if token_bytes and token_bytes in window:
                raise HubError(
                    "REFUSED to publish: exact credential bytes occur in %r"
                    % relpath)
            if _OBVIOUS_HF_TOKEN.search(window):
                raise HubError(
                    "REFUSED to publish: apparent Hugging Face token occurs in %r"
                    % relpath)
            if textual and any(pattern in window
                               for pattern in _PRIVATE_ABSOLUTE_PATHS):
                raise HubError(
                    "REFUSED to publish: private absolute path occurs in %r"
                    % relpath)
            carry = window[-overlap:]
    finally:
        os.close(fd)




def _repository_absent(exc: Exception) -> bool:
    """Recognize only the Hub's authenticated 404, never a generic API failure."""
    response = getattr(exc, "response", None)
    return (exc.__class__.__name__ == "RepositoryNotFoundError"
            and response is not None
            and getattr(response, "status_code", None) == 404)


def publish_dataset(root: str, repo: str, qualification_path: str, *,
                    expected_head: Optional[str], token: Optional[str] = None,
                    private: bool = False,
                    message: str = "publish qualified fidelity dataset",
                    repo_type: str = "dataset") -> Dict[str, Any]:
    """Publish one qualified root dataset with an optimistic one-commit cutover."""
    from . import dsvalidate

    report = dsvalidate.validate_dataset(root, verify_tensors=True)
    if not report.passed:
        raise HubError("REFUSED to publish: %s did not verify (%d errors, first: %s)"
                       % (root, len(report.errors), report.errors[0]["message"]))
    manifest = F.load_manifest(root)
    if (manifest.get("dataset") or {}).get("structural_status") == "draft":
        raise HubError("REFUSED to publish: structural_status is 'draft'")
    if not isinstance(token, str) or not token:
        raise HubError("REFUSED to publish: no token (--token-file is required)")
    if expected_head is not None \
            and (not isinstance(expected_head, str)
                 or len(expected_head) != 40
                 or any(ch not in "0123456789abcdef" for ch in expected_head)):
        raise HubError("REFUSED to publish: expected HEAD must be absent or exact 40-hex")
    if repo_type != "dataset" or repo.count("/") != 1:
        raise HubError("REFUSED to publish: destination must be owner/name dataset repo")
    if private:
        raise HubError(
            "REFUSED to publish: canonical root publication must be public")

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise HubError("publishing needs huggingface_hub; `pip install huggingface_hub`")
    evidence_rel = "receipts/root-qualification.json"
    files = sorted(set(F.iter_dataset_files(root))
                   | {F.CHECKSUMS_NAME, F.MANIFEST_NAME})
    if evidence_rel in files:
        raise HubError(
            "REFUSED to publish: sealed dataset already occupies qualification path")
    operations = []
    for relpath in files:
        if F.looks_like_a_credential(relpath):
            raise HubError(
                "REFUSED to publish credential/private file %r" % relpath)
        source = F.resolve_inside(root, relpath, owner="publish_dataset")
        source_absolute = os.path.realpath(source).lstrip(os.sep).replace(os.sep, "/")
        if F.looks_like_a_credential(source_absolute):
            raise HubError(
                "REFUSED to publish credential/private absolute path for %r"
                % relpath)
        if os.path.islink(source) or not os.path.isfile(source):
            raise HubError(
                "REFUSED to publish non-regular dataset member %r" % relpath)
        _scan_publish_member(
            source, relpath, token,
            textual=_textual_publish_member(relpath))
        operations.append(CommitOperationAdd(
            path_in_repo=relpath, path_or_fileobj=source))
    qualification_absolute = os.path.realpath(qualification_path)
    qualification_private = qualification_absolute.lstrip(os.sep).replace(os.sep, "/")
    if F.looks_like_a_credential(qualification_private):
        raise HubError("REFUSED to publish qualification from credential/private path")
    if os.path.islink(qualification_path) or not os.path.isfile(qualification_path):
        raise HubError("REFUSED to publish non-regular qualification receipt")
    _scan_publish_member(
        qualification_absolute, evidence_rel, token, textual=True)
    operations.append(CommitOperationAdd(
        path_in_repo=evidence_rel, path_or_fileobj=qualification_absolute))
    api = HfApi(token=token, endpoint=HF_ENDPOINT)
    try:
        identity = api.whoami(token=token)
    except Exception as exc:
        raise HubError("REFUSED to publish: authenticated principal preflight failed: %s"
                       % exc)
    namespace = repo.split("/", 1)[0]
    if not isinstance(identity, dict) or not identity.get("name") \
            or not _write_namespace_allowed(identity, namespace):
        raise HubError(
            "REFUSED to publish: authenticated principal lacks declared write "
            "authority for namespace %r" % namespace)

    absent = False
    try:
        info = api.repo_info(repo_id=repo, repo_type=repo_type, token=token)
    except Exception as exc:
        if not _repository_absent(exc):
            raise HubError("REFUSED to publish: destination preflight failed: %s" % exc)
        absent = True
        info = None
    for collision_type in ("model", "space"):
        try:
            api.repo_info(
                repo_id=repo, repo_type=collision_type, token=token)
        except Exception as exc:
            if _repository_absent(exc):
                continue
            raise HubError(
                "REFUSED to publish: %s repo-type collision preflight is "
                "ambiguous: %s" % (collision_type, exc))
        raise HubError(
            "REFUSED to publish: destination collides with existing %s repo"
            % collision_type)
    observed_head = getattr(info, "sha", None)
    observed_private = getattr(info, "private", None)
    if absent:
        if expected_head is not None:
            raise HubError(
                "REFUSED to publish: destination is absent, not expected HEAD %s"
                % expected_head)
        try:
            api.create_repo(repo_id=repo, repo_type=repo_type, private=private,
                            exist_ok=False, token=token)
            info = api.repo_info(repo_id=repo, repo_type=repo_type, token=token)
        except Exception as exc:
            raise HubError(
                "REFUSED to publish: exclusive destination creation failed: %s" % exc)
        observed_head = getattr(info, "sha", None)
        observed_private = getattr(info, "private", None)
    else:
        if expected_head is None:
            raise HubError(
                "REFUSED to publish: destination already exists; exact expected "
                "immutable HEAD was not authorized")
        if observed_head != expected_head:
            raise HubError(
                "REFUSED to publish: destination HEAD changed or was not authorized "
                "(expected %s, observed %r)" % (expected_head, observed_head))
    if observed_private is not False:
        raise HubError(
            "REFUSED to publish: destination is not confirmed public")
    if (not isinstance(observed_head, str) or len(observed_head) != 40
            or any(ch not in "0123456789abcdef" for ch in observed_head)):
        raise HubError(
            "REFUSED to publish: destination has no exact immutable 40-hex parent")

    try:
        commit = api.create_commit(
            repo_id=repo, repo_type=repo_type, operations=operations,
            commit_message=message, parent_commit=observed_head, token=token)
    except Exception as exc:
        raise HubError(
            "REFUSED to publish: optimistic one-commit publication failed: %s" % exc)
    revision = getattr(commit, "oid", None)
    if (not isinstance(revision, str) or len(revision) != 40
            or any(ch not in "0123456789abcdef" for ch in revision)):
        raise HubError(
            "publication did not return an immutable 40-hex commit revision")
    return {
        "repository": repo,
        "dataset_sha256": manifest[F.SEAL_FIELD],
        "private": bool(private),
        "revision": revision,
        "parent_revision": observed_head,
        "authenticated_principal": identity["name"],
        "qualification_path_in_repo": evidence_rel,
    }
