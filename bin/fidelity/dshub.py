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
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import common
from . import dsformat as F

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
USER_AGENT = "malaiwah-fidelity-dataset/1.0"


class HubError(Exception):
    pass


def read_token(path_or_env: Optional[str] = None) -> Optional[str]:
    """Read a token from an explicit path, then HF_TOKEN, then the CLI cache.

    Registered as a secret immediately; never returned into a log line.
    """
    token = None
    if path_or_env and os.path.isfile(path_or_env):
        with open(path_or_env, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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


class _NoCrossHostAuth(urllib.request.HTTPRedirectHandler):
    """Drop `Authorization` when a redirect crosses to another host.

    urllib copies every header except content-length/content-type across a redirect, and
    HF `/resolve/` URLs 302 to a pre-signed CDN host -- so the bearer token was handed to
    whatever host the endpoint redirected to. `requests`, and therefore huggingface_hub,
    strips it; this hand-rolled client did not, making it strictly looser than the library
    it stands in for."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl)
        if new is not None and _host(newurl) != _host(req.full_url):
            new.headers = {k: v for k, v in new.headers.items()
                           if k.lower() != "authorization"}
            new.unredirected_hdrs = {k: v for k, v in getattr(new, "unredirected_hdrs", {}).items()
                                     if k.lower() != "authorization"}
        return new


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
    joined onto `dest` unchecked, so a line reading `<64 hex>  ../../../../k6/tools/
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


def publish_dataset(root: str, repo: str, *, token: Optional[str] = None,
                    private: bool = False, message: str = "publish fidelity dataset",
                    repo_type: str = "dataset") -> Dict[str, Any]:
    """Upload a verified dataset.  Refuses everything the spec says to refuse.

    Three refusals before a byte moves: the dataset must verify with tensors,
    its `structural_status` must not be `draft`, and a token must be present.
    """
    from . import dsvalidate

    report = dsvalidate.validate_dataset(root, verify_tensors=True)
    if not report.passed:
        raise HubError("REFUSED to publish: %s did not verify (%d errors, first: %s)"
                       % (root, len(report.errors), report.errors[0]["message"]))
    manifest = F.load_manifest(root)
    if (manifest.get("dataset") or {}).get("structural_status") == "draft":
        raise HubError("REFUSED to publish: structural_status is 'draft'")
    if not token:
        raise HubError("REFUSED to publish: no token (HF_TOKEN, or --token-file)")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HubError("publishing needs huggingface_hub; `pip install huggingface_hub`")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type=repo_type, private=private, exist_ok=True)
    # Belt to dsformat's braces. `iter_dataset_files` now REFUSES a credential under the
    # dataset root, so a sealed dataset cannot contain one; this catches a file that
    # appeared between the seal and the upload, and huggingface_hub's own default ignore
    # list rescues only `.git`. Each pattern needs both a bare and a `**/` form, because
    # filter_repo_objects matches with fnmatch on the relative path and `**/x` still
    # requires a literal slash -- so `**/.hf_token` alone would have uploaded a
    # root-level `.hf_token`, which is exactly where measure_cloud.py writes one.
    ignore = []
    for pat in F.CREDENTIAL_FILE_PATTERNS:
        ignore += [pat, "**/" + pat]
    for d in F.CREDENTIAL_DIR_NAMES:
        ignore += [d + "/*", d + "/**", "**/" + d + "/**"]
    for sfx in F.CREDENTIAL_SUFFIXES:
        ignore += ["*" + sfx, "**/*" + sfx]
    api.upload_folder(folder_path=root, repo_id=repo, repo_type=repo_type,
                      commit_message=message, ignore_patterns=ignore)
    return {"repository": repo, "dataset_sha256": manifest[F.SEAL_FIELD],
            "private": bool(private)}
