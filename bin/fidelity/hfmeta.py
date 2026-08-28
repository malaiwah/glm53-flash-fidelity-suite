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

from .common import register_secret

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
SHA40 = re.compile(r"^[0-9a-f]{40}$")

# Filenames that identify a checkpoint's packing surface.  Sniffing beats
# asking the user, because the user usually does not know either.
SURFACE_MARKERS = {
    "tr3-published": ("materialization-receipt.json", "exl3-mcg-storage-abi.json"),
    "dione": ("exl3-manifest.json",),
    "packed": ("materialization-receipt.json",),
}


class HFError(RuntimeError):
    pass


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
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
               revision: str = "main", timeout: float = 60.0) -> bytes:
    kind = "datasets/" if repo_type == "dataset" else ""
    url = "%s/%s%s/resolve/%s/%s" % (
        HF_ENDPOINT, kind, repo_id, revision, urllib.parse.quote(path)
    )
    req = urllib.request.Request(url, headers={"User-Agent": "fidelity-suite/0.1"})
    token = hf_token()
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise HFError("HTTP %d fetching %s from %s" % (exc.code, path, repo_id)) from None


def fetch_json(repo_id: str, path: str, **kw) -> Any:
    return json.loads(fetch_file(repo_id, path, **kw).decode("utf-8"))


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
    "int8", "int4", "exl3-mcg", "exl3-trellis", "gguf-k-quant", "gguf-i-quant",
    "awq", "gptq", "mlx-affine", "hqq", "mixed", "unknown",
}


def normalize_codec(quant_method: Optional[str],
                    codebook: Optional[str] = None) -> str:
    raw = (quant_method or "").strip().lower()
    book = (codebook or "").strip().lower()
    if raw in _CODEC_VOCABULARY:
        return raw
    if raw in ("exl3", "exllamav3"):
        if book in ("mcg", ""):
            return "exl3-mcg" if book == "mcg" else "exl3-trellis"
        if book == "trellis":
            return "exl3-trellis"
        return "exl3-mcg"
    aliases = {
        "gptq": "gptq", "awq": "awq", "hqq": "hqq",
        "compressed-tensors": "mixed", "fp8": "fp8_e4m3",
        "bitsandbytes_4bit": "int4", "bitsandbytes_8bit": "int8",
        "mlx": "mlx-affine",
    }
    return aliases.get(raw, "unknown")


def sniff_surface(meta: RepoMeta) -> SurfaceInfo:
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

    if "exl3-mcg-storage-abi.json" in names:
        info.surface = "tr3-published"
        try:
            abi = fetch_json(meta.repo_id, "exl3-mcg-storage-abi.json",
                             revision=meta.revision)
            info.exllamav3_pin = abi.get("git_commit")
            info.evidence["packed_reader_abi_sha256"] = abi.get("packed_reader_abi_sha256")
        except HFError as exc:
            info.problems.append("cannot read exl3-mcg-storage-abi.json: %s" % exc)
    elif "exl3-manifest.json" in names:
        info.surface = "dione"
    elif "materialization-receipt.json" in names:
        info.surface = "packed"

    if "quantization_config.json" in names:
        try:
            qc = fetch_json(meta.repo_id, "quantization_config.json",
                            revision=meta.revision)
            info.codebook = qc.get("codebook")
            info.codec_family = normalize_codec(qc.get("quant_method"), info.codebook)
            bits = qc.get("bits")
            info.bits = float(bits) if bits is not None else None
        except HFError:
            pass

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
                store_published = any(
                    p.startswith(".materialization/") or p.startswith("payload")
                    for p in names
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

    if info.surface == "unknown":
        info.problems.append(
            "no recognised surface marker in %s (looked for %s)"
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
    include=["logits/window-*.safetensors", "*.json"],
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
        "25 sealed 'final' windows, fp32 teacher logits. The repo also holds "
        "the calibration trees (475 GB) and non-final logits; the include set "
        "fetches ~2.4% of it."
    ),
)


def load_panel_descriptor(spec: Optional[str]) -> PanelDescriptor:
    """A descriptor is a JSON file path, or a repo id, or None for the default."""
    if not spec:
        return DEFAULT_PANEL
    path = Path(spec)
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PanelDescriptor(
            panel_ref=raw["panel_ref"],
            repo_id=raw["repo_id"],
            revision=raw.get("revision", "main"),
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
