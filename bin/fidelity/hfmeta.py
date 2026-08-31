"""Hugging Face metadata: revision pinning, blob sizes, surface sniffing.

Everything here costs a few megabytes at most.  That is the point: the whole
fit estimate has to be answerable BEFORE a 200 GB download, so a refusal costs
seconds instead of an hour and a rental.

Auth: the token is read from the environment or the standard cache file and is
registered for redaction the moment it is read.  It is never passed on a
command line and never written to a receipt.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import register_secret, safe_urlopen

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _endpoint_host() -> str:
    try:
        return (urllib.parse.urlsplit(HF_ENDPOINT).hostname or "").lower()
    except ValueError:
        return ""


def _url_host(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
SHA40 = re.compile(r"^[0-9a-f]{40}$")

# Filenames that identify a checkpoint's packing surface.  Sniffing beats
# asking the user, because the user usually does not know either.
SURFACE_MARKERS = {
    "tr3-published": ("materialization-receipt.json", "exl3-mcg-storage-abi.json"),
    # 0xSero publishes the manifest as EXL3_MANIFEST.json on newer repos; the
    # sniffer matches the name case-insensitively with _ and - equivalent.
    "dione": ("exl3-manifest.json", "EXL3_MANIFEST.json"),
    "packed": ("materialization-receipt.json",),
    # stock exllamav3 HF-sharded release: no marker FILE at all -- identified
    # by config.json's inline quantization_config.quant_method == "exl3" plus
    # a canonical model.safetensors.index.json.
    "exl3hf": ("config.json (inline quantization_config, quant_method exl3)",),
    # llama.cpp container: no config.json, no index, no marker file at all --
    # identified by the .gguf extension itself.  A GGUF *repo* is a shelf of
    # independent builds, not one artifact (see `gguf_builds`).
    "gguf": ("*.gguf",),
}


# --------------------------------------------------------------------------
# GGUF: a repo is a shelf, not an artifact
# --------------------------------------------------------------------------

#: ggml type tokens as they appear in a build's directory or file name, longest
#: first so `IQ4_XS` is not read as `Q4_...`.  The nominal rate is what the NAME
#: claims; unsloth's "UD" (Unsloth Dynamic) recipes mix several types across
#: tensor classes, so this is a family label and never a measured bpw.  What the
#: build actually contains is read from its own headers by the readability gate.
_GGML_NAME_TOKENS = (
    ("IQ2_XXS", 2.0, "gguf-i-quant"), ("IQ3_XXS", 3.0, "gguf-i-quant"),
    ("IQ4_XS", 4.0, "gguf-i-quant"), ("IQ4_NL", 4.0, "gguf-i-quant"),
    ("IQ2_XS", 2.0, "gguf-i-quant"), ("IQ3_S", 3.0, "gguf-i-quant"),
    ("IQ2_S", 2.0, "gguf-i-quant"), ("IQ1_M", 1.0, "gguf-i-quant"),
    ("IQ1_S", 1.0, "gguf-i-quant"),
    ("Q8_0", 8.0, "gguf-k-quant"), ("Q6_K", 6.0, "gguf-k-quant"),
    ("Q5_K", 5.0, "gguf-k-quant"), ("Q4_K", 4.0, "gguf-k-quant"),
    ("Q3_K", 3.0, "gguf-k-quant"), ("Q2_K", 2.0, "gguf-k-quant"),
    ("Q5_1", 5.0, "gguf-k-quant"), ("Q5_0", 5.0, "gguf-k-quant"),
    ("Q4_1", 4.0, "gguf-k-quant"), ("Q4_0", 4.0, "gguf-k-quant"),
    ("TQ1_0", 1.0, "gguf-k-quant"), ("TQ2_0", 2.0, "gguf-k-quant"),
    ("MXFP4", 4.0, "mxfp4"),
    ("BF16", 16.0, "bf16"), ("F16", 16.0, "fp16"), ("F32", 32.0, "fp32"),
)

#: llama.cpp's split suffix, e.g. `-00002-of-00006.gguf`.
_GGUF_SPLIT = re.compile(r"-\d{5}-of-\d{5}$")


def gguf_build_key(path: str) -> str:
    """The build a .gguf file belongs to.

    Two published layouts, both in the wild for the same publisher:
    `unsloth/GLM-5.3-Flash-GGUF` puts each build in its own directory
    (`UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00003-of-00006.gguf`), while
    `unsloth/Qwen3.8-27B-GGUF` publishes flat files at the repo root
    (`Qwen3.8-27B-Q6_K.gguf`).  Directory wins where there is one; otherwise
    the file stem with llama.cpp's split suffix removed, which is what makes
    the parts of one split build group together instead of reading as six
    separate artifacts.
    """
    head, _, tail = path.rpartition("/")
    if head:
        return head
    return _GGUF_SPLIT.sub("", tail[:-len(".gguf")])


def gguf_builds(meta: "RepoMeta") -> Dict[str, List[Tuple[str, int]]]:
    """Every selectable model build in a GGUF repo, keyed by `gguf_build_key`.

    `mmproj-*` files are the vision projector, not a model build: llama.cpp
    ships the vision tower separately and the text-only sealed panel never
    executes it.  They are excluded from selection rather than offered as a
    build nobody can score.  `.gguf_file` (unsloth's shard-rewrite sidecars and
    its published imatrix) is deliberately NOT `.gguf` and never matches.
    """
    builds: Dict[str, List[Tuple[str, int]]] = {}
    for path, size in meta.files:
        if not path.endswith(".gguf"):
            continue
        if path.rpartition("/")[2].startswith("mmproj"):
            continue
        builds.setdefault(gguf_build_key(path), []).append((path, size))
    for files in builds.values():
        files.sort()
    return builds


def gguf_nominal_rate(build: str) -> Tuple[Optional[float], str]:
    """(nominal bits, registry codec) claimed by a build's NAME."""
    upper = build.upper()
    for token, bits, codec in _GGML_NAME_TOKENS:
        if token in upper:
            return bits, codec
    return None, "unknown"


class HFError(RuntimeError):
    pass


# SECURITY NOTE (SEC-01).  `repo_meta` and `resolve_revision` below are not only
# metadata helpers: `bin/stage_measure.sh`'s fetch_panel stage interpolates
# `panel.repo_id` and `panel.revision` from job.json into a shell command on a
# rented box that holds a live HF token.  The plan path calls `repo_meta` and
# overwrites the revision with `resolve_revision`'s 40-hex result, which is why
# an injecting value cannot survive a live run.  That guarantee is LOAD-BEARING;
# do not make either call optional, cached-only or best-effort without first
# re-reading docs/REVIEW-DEFERRED.md SEC-01.  `load_panel_descriptor` validates
# the same two fields at ingestion as a second, independent backstop.


def hf_token() -> Optional[str]:
    """Read the token from env or the standard cache, and register it."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        path = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        ) / "token"
        if path.is_file():
            try:
                token = path.read_text(encoding="utf-8").strip()
            except OSError:
                token = None
    token = (token or "").strip() or None
    register_secret(token)
    return token


def _get(url: str, *, timeout: float = 30.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "fidelity-suite/0.1"})
    token = hf_token()
    if token:
        # Only ever to the configured endpoint: a caller-built URL that names
        # any other host must not carry the token at all -- and the redirect
        # handler in `safe_urlopen` strips it again if the endpoint 302s to a
        # CDN host, which HF `/resolve/` URLs routinely do.
        if _url_host(url) == _endpoint_host():
            req.add_header("Authorization", "Bearer " + token)
        else:
            raise HFError(
                "refusing to send the Hugging Face token to %s (the configured "
                "endpoint is %s)" % (_url_host(url) or "an unparseable host",
                                     _endpoint_host()))
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        hint = ""
        if exc.code in (401, 403):
            hint = " (private or gated? export HF_TOKEN)"
        elif exc.code == 404:
            hint = " (no such repo/revision, or it is private)"
        raise HFError("HTTP %d for %s%s" % (exc.code, url, hint)) from None
    except urllib.error.URLError as exc:
        raise HFError("network error for %s: %s" % (url, exc.reason)) from None


@dataclass
class RepoMeta:
    repo_id: str
    repo_type: str                  # "model" | "dataset"
    revision: str                   # resolved 40-hex
    requested_revision: str
    last_modified: Optional[str]
    files: List[Tuple[str, int]] = field(default_factory=list)  # (path, size)
    author: Optional[str] = None
    private: bool = False

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files)

    def matching(self, patterns: List[str]) -> List[Tuple[str, int]]:
        import fnmatch

        out = []
        for path, size in self.files:
            if any(fnmatch.fnmatch(path, pat) for pat in patterns):
                out.append((path, size))
        return out

    def bytes_matching(self, patterns: List[str]) -> int:
        return sum(size for _, size in self.matching(patterns))

    def has(self, name: str) -> bool:
        return any(p == name or p.endswith("/" + name) for p, _ in self.files)

    @property
    def weight_bytes(self) -> int:
        return self.bytes_matching(["*.safetensors"])

    @property
    def url(self) -> str:
        seg = "datasets/" if self.repo_type == "dataset" else ""
        return "%s/%s%s" % (HF_ENDPOINT, seg, self.repo_id)


def _api_path(repo_id: str, repo_type: str) -> str:
    kind = "datasets" if repo_type == "dataset" else "models"
    return "%s/api/%s/%s" % (HF_ENDPOINT, kind, repo_id)


def hf_unavailable_text(repo_id: str, exc: Exception) -> str:
    """The three-way-honest failure text for an unauthenticated repo lookup.

    HF returns 401 ("Invalid username or password.") for a NONEXISTENT repo on
    unauthenticated requests, and errors for gated/private repos the same way,
    so "gone", "private" and "gated" are indistinguishable without auth.  Say
    exactly that instead of guessing one of the three.
    """
    return (
        "HF returned an error for %s (%s): the repo does not exist, or is "
        "private/gated (unauthenticated requests cannot distinguish these). "
        "The registry lookup continues by repo string regardless -- the "
        "registry records artifacts whose repos have since vanished."
        % (repo_id, exc)
    )


# --------------------------------------------------------------------------
# Lineage metadata (base_model chains)
# --------------------------------------------------------------------------


@dataclass
class ModelLineageMeta:
    """The slice of /api/models/<repo> that lineage resolution needs.

    `base_models` is a list of (relation_or_None, base_repo) pairs.  Tags of
    the form "base_model:<relation>:<repo>" are preferred over
    cardData.base_model because the relation lives in the tag and cardData's
    base_model_relation is unreliably present (verified live: 0xSero publishes
    the list form with no relation field; malaiwah the string form with one).
    """

    repo_id: str                      # canonical case, from the API's own `id`
    sha: Optional[str]                # current main commit
    last_modified: Optional[str]
    tags: List[str] = field(default_factory=list)
    base_models: List[Tuple[Optional[str], str]] = field(default_factory=list)
    gated: Any = None
    private: bool = False


def model_lineage_meta(repo_id: str) -> ModelLineageMeta:
    """GET /api/models/<repo> and extract lineage-relevant fields.

    Follows redirects (a wrong-cased repo 307s to the canonical one); the
    returned `id` is adopted as the canonical spelling.  Raises HFError on
    401/404/network -- callers wrap it with hf_unavailable_text().
    """
    data = _get(_api_path(repo_id, "model"))
    tags = [t for t in (data.get("tags") or []) if isinstance(t, str)]
    bases: List[Tuple[Optional[str], str]] = []
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        parts = tag.split(":", 2)
        if len(parts) == 3:
            bases.append((parts[1] or None, parts[2]))
        elif len(parts) == 2 and "/" in parts[1]:
            bases.append((None, parts[1]))
    if not bases:
        card = data.get("cardData") or {}
        raw = card.get("base_model")
        listed = raw if isinstance(raw, list) else ([raw] if raw else [])
        relation = card.get("base_model_relation")
        for base in listed:
            if isinstance(base, str) and "/" in base:
                bases.append((relation, base))
    # dedupe, preserving first-seen order
    seen = set()
    unique: List[Tuple[Optional[str], str]] = []
    for rel, repo in bases:
        key = (rel, repo.lower())
        if key not in seen:
            seen.add(key)
            unique.append((rel, repo))
    return ModelLineageMeta(
        repo_id=data.get("id") or repo_id,
        sha=data.get("sha"),
        last_modified=data.get("lastModified"),
        tags=tags,
        base_models=unique,
        gated=data.get("gated"),
        private=bool(data.get("private")),
    )


def resolve_commit(repo_id: str, revision: str, repo_type: str = "model") -> str:
    """Resolve a branch / tag / short sha to the full 40-hex commit.

    Uses /api/<kind>/<repo>/revision/<rev>, which answers for all three forms;
    a full 40-hex revision is still round-tripped through the API so a typo'd
    hash fails HERE, not after a download.
    """
    url = "%s/revision/%s" % (_api_path(repo_id, repo_type),
                              urllib.parse.quote(revision, safe=""))
    data = _get(url)
    sha = data.get("sha")
    if not (isinstance(sha, str) and SHA40.match(sha)):
        raise HFError("revision %r of %s did not resolve to a 40-hex commit"
                      % (revision, repo_id))
    return sha


def resolve_revision(repo_id: str, repo_type: str = "model",
                     revision: str = "main") -> str:
    """Turn a branch name into an immutable 40-hex commit, on the caller's machine.

    This happens BEFORE any money is spent, and the resolved pin is echoed in
    the confirmation prompt.  A recipe that fetches `main` measures whatever
    the author happened to have pushed that morning and cannot be reproduced.
    """
    if SHA40.match(revision or ""):
        return revision
    data = _get(_api_path(repo_id, repo_type) + "/refs")
    for group in ("branches", "tags"):
        for ref in data.get(group, []) or []:
            if ref.get("name") == revision or ref.get("ref") == "refs/heads/" + revision:
                target = ref.get("targetCommit") or ref.get("target_commit")
                if target and SHA40.match(target):
                    return target
    raise HFError(
        "cannot resolve %r to a commit in %s; pass --revision <40-hex>"
        % (revision, repo_id)
    )


def repo_meta(repo_id: str, repo_type: str = "model",
              revision: str = "main") -> RepoMeta:
    pinned = resolve_revision(repo_id, repo_type, revision)
    url = "%s/revision/%s?blobs=true" % (
        _api_path(repo_id, repo_type), urllib.parse.quote(pinned, safe="")
    )
    data = _get(url)
    files: List[Tuple[str, int]] = []
    for sib in data.get("siblings", []) or []:
        name = sib.get("rfilename")
        if not name:
            continue
        size = sib.get("size")
        if size is None:
            size = (sib.get("lfs") or {}).get("size", 0)
        files.append((name, int(size or 0)))
    return RepoMeta(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=pinned,
        requested_revision=revision,
        last_modified=data.get("lastModified"),
        files=sorted(files),
        author=data.get("author"),
        private=bool(data.get("private")),
    )


def fetch_file(repo_id: str, path: str, *, repo_type: str = "model",
               revision: str = "main", timeout: float = 60.0,
               byte_range: Optional[Tuple[int, int]] = None) -> bytes:
    kind = "datasets/" if repo_type == "dataset" else ""
    url = "%s/%s%s/resolve/%s/%s" % (
        HF_ENDPOINT, kind, repo_id, revision, urllib.parse.quote(path)
    )
    req = urllib.request.Request(url, headers={"User-Agent": "fidelity-suite/0.1"})
    if byte_range is not None:
        req.add_header("Range", "bytes=%d-%d" % byte_range)
    token = hf_token()
    if token:
        # Same rule as _get: token only to the configured endpoint, and
        # `safe_urlopen` strips it when the resolve URL redirects off-origin.
        if _url_host(url) == _endpoint_host():
            req.add_header("Authorization", "Bearer " + token)
        else:
            raise HFError(
                "refusing to send the Hugging Face token to %s (the configured "
                "endpoint is %s)" % (_url_host(url) or "an unparseable host",
                                     _endpoint_host()))
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise HFError("HTTP %d fetching %s from %s" % (exc.code, path, repo_id)) from None


def fetch_json(repo_id: str, path: str, **kw) -> Any:
    return json.loads(fetch_file(repo_id, path, **kw).decode("utf-8"))


def safetensors_header(repo_id: str, path: str, **kw) -> Optional[Dict[str, Any]]:
    """The tensor headers of one safetensors file, by RANGE request.

    A safetensors file begins with an 8-byte little-endian header length and
    then that many bytes of JSON, so its full tensor inventory is readable in
    two small requests -- no matter that the file itself is gigabytes. Used to
    see inside sidecars an index does not cover (mtp.safetensors). Returns None
    when the file is absent.
    """
    import struct

    try:
        raw = fetch_file(repo_id, path, byte_range=(0, 7), **kw)
        if len(raw) < 8:
            return None
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length < (64 << 20):
            return None
        body = fetch_file(repo_id, path, byte_range=(8, 8 + length - 1), **kw)
        header = json.loads(body.decode("utf-8"))
    except (HFError, ValueError, struct.error):
        return None
    return {k: v for k, v in header.items() if k != "__metadata__"}


@dataclass
class SurfaceInfo:
    surface: str                # tr3-published | dione | packed | unknown
    codec_family: Optional[str] = None
    bits: Optional[float] = None
    codebook: Optional[str] = None
    exllamav3_pin: Optional[str] = None
    nonrouted_native: Optional[bool] = None
    shard_count: int = 0
    tp_sliced: bool = False
    tp_world_size: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    #: For a repo that publishes MANY artifacts at one revision (a GGUF shelf),
    #: the build this SurfaceInfo describes, and exactly the files it is made
    #: of.  Every other surface leaves these None/[]: the repo IS the artifact.
    path: Optional[str] = None
    artifact_files: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def artifact_bytes(self) -> Optional[int]:
        return sum(size for _, size in self.artifact_files) or None

    @property
    def usable(self) -> bool:
        return self.surface != "unknown" and not self.problems


# The registry's codec vocabulary is closed (artifact.schema.json), and a
# checkpoint's own `quant_method` is not written in it.  Mapping is a real step,
# not a pass-through: a TR3 repo says `quant_method: "exl3"` with
# `codebook: "mcg"`, which the registry calls `exl3-mcg`.  Emitting the raw
# string produces a receipt that fails schema validation at submission time --
# which is exactly where you least want to discover it.
_CODEC_VOCABULARY = {
    "fp64", "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "nvfp4", "mxfp4",
    "int8", "int4", "exl3-mcg", "exl3-mul1", "exl3-trellis", "gguf-k-quant",
    "gguf-i-quant",
    "awq", "gptq", "mlx-affine", "hqq", "mixed", "unknown",
}


def normalize_codec(quant_method: Optional[str],
                    codebook: Optional[str] = None) -> str:
    raw = (quant_method or "").strip().lower()
    book = (codebook or "").strip().lower()
    if raw in _CODEC_VOCABULARY:
        return raw
    if raw in ("exl3", "exllamav3"):
        if book == "mcg":
            return "exl3-mcg"
        if book == "mul1":
            # exllamav3 >= 1.4 default codebook; a DIFFERENT decode map than
            # MCG (multiplier 0x83DCD12D, dp4a byte-sum).  Labeling it
            # exl3-mcg would write a false codec family on artifact records.
            return "exl3-mul1"
        if book in ("trellis", "3inst", ""):
            return "exl3-trellis"
        return "exl3-%s" % book
    if raw == "exl3_selective_tp4":
        # the Dione conversion: standard EXL3/MCG payloads, TP4-sliced storage
        return "exl3-mcg"
    aliases = {
        "gptq": "gptq", "awq": "awq", "hqq": "hqq",
        "compressed-tensors": "mixed", "fp8": "fp8_e4m3",
        "bitsandbytes_4bit": "int4", "bitsandbytes_8bit": "int8",
        "mlx": "mlx-affine",
    }
    return aliases.get(raw, "unknown")


def sniff_surface(meta: RepoMeta, path: Optional[str] = None) -> SurfaceInfo:
    """Decide how a published checkpoint must be read, from its own files.

    The distinction that matters most: a `packed` checkpoint's
    materialization-receipt names a `packed_root` payload store which lives on
    the PRODUCER's machine and is not published.  A `tr3-published` checkpoint
    carries its payloads inline in the shards.  They look nearly identical in a
    file listing and behave completely differently, so we check the receipt's
    contents rather than its name.
    """
    names = {p for p, _ in meta.files}
    info = SurfaceInfo(surface="unknown")
    info.shard_count = len([p for p in names if p.endswith(".safetensors")])

    if any(re.search(r"\.rank\d+\.", p) for p in names):
        info.tp_sliced = True
        ranks = {int(m.group(1)) for p in names
                 for m in [re.search(r"\.rank(\d+)\.", p)] if m}
        info.tp_world_size = max(ranks) + 1 if ranks else None

    # GGUF first, and by extension alone.  A llama.cpp container carries no
    # config.json, no safetensors index and no marker file, so every marker the
    # other branches look for is absent -- which is why an unsloth GGUF repo
    # used to reach the "no recognised surface marker" refusal.  It is also the
    # only surface here whose REPO is not the artifact: unsloth publishes
    # twelve independent builds of GLM-5.3-Flash at one revision, so a
    # measurement must name one.
    builds = gguf_builds(meta)
    if builds:
        info.surface = "gguf"
        info.evidence["gguf_builds"] = {
            key: {"files": len(files), "bytes": sum(s for _, s in files)}
            for key, files in sorted(builds.items())
        }
        want = (path or "").strip().strip("/")
        chosen = None
        if want:
            for key, files in builds.items():
                if want == key or want in {p for p, _ in files} or \
                        want == key.rpartition("/")[2]:
                    chosen = key
                    break
            if chosen is None:
                info.problems.append(
                    "--path %r names no build in %s; it publishes: %s"
                    % (want, meta.repo_id, ", ".join(sorted(builds))))
        elif len(builds) == 1:
            chosen = next(iter(builds))
        else:
            info.problems.append(
                "%s publishes %d GGUF builds at this revision (%s) and a "
                "measurement describes ONE of them. Pass --path <build>."
                % (meta.repo_id, len(builds), ", ".join(sorted(builds))))
        if chosen is not None:
            info.path = chosen
            info.artifact_files = list(builds[chosen])
            info.shard_count = len(info.artifact_files)
            info.bits, info.codec_family = gguf_nominal_rate(chosen)
            info.evidence["gguf_rate_source"] = (
                "nominal, read from the build NAME. unsloth's UD (Unsloth "
                "Dynamic) recipes mix ggml types across tensor classes, so this "
                "is a family label, not a measured bits-per-weight.")
            # A GGUF quantizes the whole forward, so nothing in it is retained
            # at source precision -- the opposite of the routed-experts-only
            # releases this suite mostly measures. Recorded here so the plan
            # and the receipt say it rather than implying the usual scope.
            info.nonrouted_native = False
        return info

    if "exl3-mcg-storage-abi.json" in names:
        info.surface = "tr3-published"
        try:
            abi = fetch_json(meta.repo_id, "exl3-mcg-storage-abi.json",
                             revision=meta.revision)
            info.exllamav3_pin = abi.get("git_commit")
            info.evidence["packed_reader_abi_sha256"] = abi.get("packed_reader_abi_sha256")
        except HFError as exc:
            info.problems.append("cannot read exl3-mcg-storage-abi.json: %s" % exc)
    elif any(n.lower().replace("_", "-") == "exl3-manifest.json" for n in names):
        info.surface = "dione"
    elif "materialization-receipt.json" in names:
        info.surface = "packed"

    def _apply_quant_config(qc):
        info.codebook = qc.get("codebook")
        info.codec_family = normalize_codec(qc.get("quant_method"), info.codebook)
        # stock exllamav3 writes `bits`; the Dione conversion writes
        # `bits_per_weight` (and `target_expert_bpw`).  Read whichever exists.
        bits = qc.get("bits")
        if bits is None:
            bits = qc.get("bits_per_weight")
        if bits is None:
            bits = qc.get("target_expert_bpw")
        info.bits = float(bits) if bits is not None else None
        if qc.get("head_bits") is not None:
            info.evidence["head_bits"] = qc.get("head_bits")
        if qc.get("version"):
            info.evidence["quantizer_version"] = qc.get("version")

    # Where the quantization block lives is a PUBLISHER's choice, not a format
    # property: exllamav3 inlines it in config.json AND (turboderp's releases)
    # also ships a standalone quantization_config.json carrying the full
    # per-module bit map -- 47.9 MB on GLM-5.3-Flash-exl3, for three fields we
    # actually need.  Prefer the small inline block; fall back to the file.
    # Classification is done AFTER, on whichever block was parsed, so a repo
    # that ships both is not misclassified by which arm ran (that bug refused
    # turboderp/GLM-5.3-Flash-exl3 as "unreadable" while its codec parsed fine).
    quant_config = None
    if "config.json" in names:
        try:
            cfg = fetch_json(meta.repo_id, "config.json", revision=meta.revision)
            inline = cfg.get("quantization_config") or \
                (cfg.get("text_config") or {}).get("quantization_config")
            if isinstance(inline, dict) and inline:
                quant_config = inline
                info.evidence["quantization_config_source"] = "config.json (inline)"
        except HFError:
            pass
    if quant_config is None and "quantization_config.json" in names:
        try:
            quant_config = fetch_json(meta.repo_id, "quantization_config.json",
                                      revision=meta.revision)
            info.evidence["quantization_config_source"] = "quantization_config.json"
        except HFError:
            pass
    if isinstance(quant_config, dict) and quant_config:
        _apply_quant_config(quant_config)
        if info.surface == "unknown" and \
                str(quant_config.get("quant_method", "")).lower() == "exl3" and \
                "model.safetensors.index.json" in names and not info.tp_sliced:
            # stock exllamav3 HF-sharded release (turboderp layout):
            # canonical index, per-module {trellis,suh,svh,<codebook>}
            # payloads, full-scope quant.  Read by the exl3hf surface.
            info.surface = "exl3hf"
        if quant_config.get("original_quantization_config") is not None:
            # quantized FROM another quant (e.g. the FP8 release): lineage
            # that the artifact record must disclose
            oqc = quant_config["original_quantization_config"]
            info.evidence["original_quantization_config_fmt"] = \
                str(oqc.get("fmt") or oqc.get("quant_method") or "unknown")

    if "materialization-receipt.json" in names:
        try:
            mr = fetch_json(meta.repo_id, "materialization-receipt.json",
                            revision=meta.revision)
            info.nonrouted_native = mr.get("nonrouted_native_exact")
            info.evidence["receipt_sha256"] = mr.get("receipt_sha256")
            info.evidence["native_tensor_count"] = mr.get("native_tensor_count")
            packed_root = mr.get("packed_root")
            if packed_root:
                info.evidence["packed_root"] = packed_root
                # THE trap.  A `packed` surface dereferences this path at
                # capture time; if the payload store is not in the repo, the
                # run dies before it touches a GPU.  Detect it here, where it
                # costs nothing, instead of there, where it costs a rental.
                # CC-07.  The old predicate was
                #   any(p.startswith(".materialization/") or p.startswith("payload"))
                # and it disarmed the trap two ways.  `.materialization/shards/
                # *.json` are shard RECEIPTS, not a payload store, and our own
                # published K6/K8 repos ship 120 of them each; and
                # `startswith("payload")` is a bare string prefix, so a file
                # merely named `payload_notes.txt` disarmed it too.  What the
                # consumer actually dereferences is named here instead
                # (stream_score.py requires exactly these five things).
                #
                # NOTE, so nobody reads more into this than it does: inside
                # `if info.surface == "packed"` this changes nothing observable
                # today, because every publisher that ships `.materialization/`
                # also ships exl3-mcg-storage-abi.json and is classified
                # `tr3-published` above.  Widening that outer guard to every
                # surface that dereferences a packed_root is a separate change
                # with live blast radius; see docs/REVIEW-DEFERRED.md CC-07.
                store_published = (
                    any(p.startswith("payload-store/objects/") for p in names)
                    and any(p.startswith("payload-store/choices/") for p in names)
                    and {"contract.json", "inventory.json",
                         "mtp-adapter-receipt.json"} <= set(names)
                )
                if info.surface == "packed" and not store_published:
                    info.problems.append(
                        "materialization-receipt.json points packed_root at %r, "
                        "which is a path on the PRODUCER's machine, and this repo "
                        "does not publish the payload store. The `packed` surface "
                        "cannot read this checkpoint."
                        % packed_root
                    )
        except HFError:
            pass

    if info.surface == "unknown" and "config.json" in names and \
            info.shard_count > 0 and "quantization_config.json" not in names:
        # A plain full-precision release tree: config + safetensors shards and
        # no quant markers anywhere.  This is the `native-bf16` surface the
        # bf16-floor lane reads (--source native needs only this tree + a
        # sealed inventory), so classify it rather than shrugging "unknown".
        try:
            cfg = fetch_json(meta.repo_id, "config.json", revision=meta.revision)
            # dtype location probed against the real release: GLM-5.3-Flash
            # nests it as text_config.dtype; older HF configs use top-level
            # torch_dtype; transformers >= 5 writes a TOP-LEVEL `dtype` on a
            # single-modality config, which is the spelling every model saved
            # by a current transformers has.  All three are read, and each one
            # was added because a real release used it and was refused without
            # it: `malaiwah/GLM-5.2-SIQ-Fruit-bf16` (transformers_version
            # 5.12.0) declares only top-level `dtype: bfloat16` and reached
            # "no recognised surface marker" -- a plain bf16 tree refused as
            # unreadable.  The old comment said "check both, never guess a
            # third"; the third was not a guess, it was the current default.
            nested = cfg.get("text_config") or {}
            dtype = str(cfg.get("torch_dtype") or cfg.get("dtype")
                        or nested.get("dtype")
                        or nested.get("torch_dtype") or "").lower()
            if "quantization_config" not in cfg and \
                    "quantization_config" not in nested and dtype in (
                    "bfloat16", "float16", "float32"):
                info.surface = "native-bf16"
                info.codec_family = {"bfloat16": "bf16", "float16": "fp16",
                                     "float32": "fp32"}[dtype]
                info.bits = 16.0 if dtype in ("bfloat16", "float16") else 32.0
                info.evidence["torch_dtype"] = dtype
        except HFError:
            pass

    if info.surface == "unknown":
        info.problems.append(
            "no recognised surface marker in %s (looked for %s, or a plain "
            "full-precision tree: config.json + shards with no "
            "quantization_config)"
            % (meta.repo_id,
               ", ".join(sorted({n for group in SURFACE_MARKERS.values() for n in group})))
        )
    return info


# --------------------------------------------------------------------------
# Panel descriptors
# --------------------------------------------------------------------------


@dataclass
class PanelDescriptor:
    """What to fetch from a teacher/panel dataset, and what it contains.

    Panel identity is first-class in the registry, so it is a parameter here,
    never a constant.  The include globs are part of the descriptor because
    getting them wrong is a 42x overspend: the default panel's repo is 1,318 GB
    and the 25 sealed final windows are 31.7 GB of it.
    """

    panel_ref: str
    repo_id: str
    revision: str
    include: List[str]
    contexts: int
    positions_per_context: int
    scored_positions: int
    roles: str = "final"
    note: str = ""
    # Identity, so a receipt can bind the panel it actually scored against
    # rather than naming it. A panel_ref alone is a label; these are the hashes
    # the registry checks.
    panel_token_sha256: Optional[str] = None
    panel_receipt_sha256: Optional[str] = None
    reference_ref: Optional[str] = None
    teacher_receipt_sha256: Optional[str] = None
    teacher_backend_identity_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_ref": self.panel_ref,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "include": list(self.include),
            "contexts": self.contexts,
            "positions_per_context": self.positions_per_context,
            "scored_positions": self.scored_positions,
            "roles": self.roles,
            "note": self.note,
            "panel_token_sha256": self.panel_token_sha256,
            "panel_receipt_sha256": self.panel_receipt_sha256,
            "reference_ref": self.reference_ref,
            "teacher_receipt_sha256": self.teacher_receipt_sha256,
            "teacher_backend_identity_sha256": self.teacher_backend_identity_sha256,
        }


# The default panel, pinned so `--dry-run` works offline.  `--panel` overrides
# every field of this; nothing downstream assumes GLM-5.3-Flash.
DEFAULT_PANEL = PanelDescriptor(
    panel_ref="panel--glm53.brandonmusic.final25",
    repo_id="brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
    revision="main",
    # The token-panel receipt names 667 artifacts by ABSOLUTE path and verifies
    # each by size and sha256 -- panel.json plus 666 .npy token/mask arrays,
    # 5.8 MB in total. They are not JSON, so the first two globs miss them, and
    # the capture then dies at load_panel_windows with "artifact identity
    # mismatch" AFTER the fetch, the materialize and the model load. Cheap to
    # fetch, fatal to omit.
    include=["logits/window-*.safetensors", "*.json",
             "calibration/panel-v1/arrays/*.npy"],
    contexts=25,
    positions_per_context=2047,
    scored_positions=51175,
    roles="final",
    panel_token_sha256="6bafe3283c54bc9342d0f30aa3199d36032d103feb92c31715be8545362790ff",
    panel_receipt_sha256="0beec5770e5107547731b084f1bc5f9fb8ba79d67af56ddb70d919da367737d5",
    reference_ref="reference--brandonmusic.glm53-bf16-fp32-logits.final25",
    teacher_receipt_sha256="2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5",
    teacher_backend_identity_sha256="85b11599c6b36a83fa8099a09a298a386a0c603d1f18d3702e7fb1c470962ce4",
    note=(
        "25 sealed 'final' windows, fp32 teacher logits, plus the 5.8 MB of "
        "token-panel arrays the panel receipt verifies by digest. The repo "
        "also holds the calibration trees (475 GB) and non-final logits; the "
        "include set fetches ~2.4% of it."
    ),
)


def load_panel_descriptor(spec: Optional[str]) -> PanelDescriptor:
    """A descriptor is a JSON file path, or a repo id, or None for the default."""
    if not spec:
        return DEFAULT_PANEL
    path = Path(spec)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        # SEC-01 (companion).  These two strings travel verbatim into job.json
        # and from there into a shell command on a rented box that holds a live
        # HF token.  Validate them where they ENTER the tree, so a hostile value
        # never reaches the shell at all.  This is a BACKSTOP, not the control:
        # the control is that stage_measure.sh no longer `eval`s them, and
        # resolve_revision's 40-hex guarantee is stronger still on the path that
        # goes through the Hub.  Note the pattern deliberately excludes `/` in a
        # revision -- a branch name with a slash is refused here rather than
        # widened, because nothing in this suite pins a panel by branch.
        repo_id = str(raw["repo_id"])
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$",
                        repo_id):
            raise HFError("panel descriptor repo_id %r is not an owner/name pair"
                          % repo_id)
        revision = str(raw.get("revision", "main"))
        if not re.match(r"^[A-Za-z0-9._-]+$", revision):
            raise HFError("panel descriptor revision %r is not a revision" % revision)
        return PanelDescriptor(
            panel_ref=raw["panel_ref"],
            repo_id=repo_id,
            revision=revision,
            include=list(raw.get("include") or ["*"]),
            contexts=int(raw["contexts"]),
            positions_per_context=int(raw["positions_per_context"]),
            scored_positions=int(raw["scored_positions"]),
            roles=raw.get("roles", "final"),
            note=raw.get("note", ""),
            panel_token_sha256=raw.get("panel_token_sha256"),
            panel_receipt_sha256=raw.get("panel_receipt_sha256"),
            reference_ref=raw.get("reference_ref"),
            teacher_receipt_sha256=raw.get("teacher_receipt_sha256"),
            teacher_backend_identity_sha256=raw.get("teacher_backend_identity_sha256"),
        )
    if spec == DEFAULT_PANEL.repo_id:
        return DEFAULT_PANEL
    raise HFError(
        "panel %r is not a known descriptor. Pass --panel-descriptor with a JSON "
        "file naming its include globs, contexts, positions_per_context and "
        "scored_positions -- the runner will not guess a panel's shape, because "
        "a wrong guess silently measures a different thing." % spec
    )
