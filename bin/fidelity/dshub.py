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


def _get(url: str, token: Optional[str] = None, binary: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if token:
        request.add_header("Authorization", "Bearer %s" % token)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise HubError("HTTP %s for %s" % (exc.code, common.redact(url)))
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

    Manifest and `checksums.txt` first, so the download is digest-driven from
    the second file onward.
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
    checksums_bytes = _get(resolve_url(repo, revision, checksums_name, repo_type),
                           token, binary=True)
    with open(os.path.join(dest, checksums_name), "wb") as handle:
        handle.write(checksums_bytes)
    listed = F.parse_checksums(checksums_bytes.decode("utf-8"))

    for relpath in sorted(listed):
        if allow_partial and relpath.startswith("capture/") \
                and relpath != "capture/manifest.json":
            continue
        payload = _get(resolve_url(repo, revision, relpath, repo_type), token, binary=True)
        target = os.path.join(dest, relpath)
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
    api.upload_folder(folder_path=root, repo_id=repo, repo_type=repo_type,
                      commit_message=message)
    return {"repository": repo, "dataset_sha256": manifest[F.SEAL_FIELD],
            "private": bool(private)}
