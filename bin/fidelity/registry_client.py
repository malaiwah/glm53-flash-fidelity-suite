"""Registry client: the public quant-fidelity dataset, from bin/ tools.

Two backends, one loader, identical in-memory shape:

  * the LOCAL clone     -- registry/data/*.jsonl in this checkout (offline);
  * the PUBLIC mirror   -- huggingface.co/datasets/malaiwah/quant-fidelity-registry,
                           fetched UNAUTHENTICATED (it is public by design; no
                           token is ever attached to these requests).

Everything derived (the comparability key above all) is computed by the
registry's OWN code, imported read-only via common.load_registry_lib -- never
reimplemented, and never trusted from a row's stored block.

Rendering rules are reproduced from registry/tools/registry_render.py, the
normative reference:
  * rows are grouped by RECOMPUTED comparability key; one table per key;
    a filter may HIDE groups but never MERGE them;
  * within a group, rows whose pipeline declares a lane are tabled apart from
    rows with no declared lane (None means "no declared lane", NOT "sealed");
  * sorting by value happens only within a lane sub-table.

Stock python3.9, stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import Console, load_registry_lib, run

SUITE_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "malaiwah/quant-fidelity-registry"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
COLLECTION_NAMES = ("models", "artifacts", "panels", "references",
                    "pipelines", "measurements")
SHA40 = re.compile(r"^[0-9a-f]{40}$")

# Tiers for the already-measured lookup.  Names are printed verbatim.
TIER_EXACT = "EXACT"          # artifact pinned at exactly the resolved revision
TIER_UNPINNED = "UNPINNED"    # artifact record has revision null
TIER_STALE = "STALE"          # artifact pinned at a DIFFERENT revision
TIER_UNVERIFIED = "PINNED-UNVERIFIED"  # artifact pinned; target revision unknown


class RegistryUnavailable(RuntimeError):
    """No data source could be loaded.  Message names both remedies."""


def cache_dir() -> Path:
    root = os.environ.get("FIDELITY_CACHE_DIR")
    # Not `.cache/glm53-fidelity`: this cache holds registry snapshots and a
    # fixture for whatever model is being measured. Renaming it orphans an
    # old cache, which costs one re-download and nothing else.
    return Path(root) if root else (Path.home() / ".cache" / "quant-fidelity")


class RegistrySnapshot:
    """One loaded registry: {collection: {id: record}} plus its identity."""

    def __init__(self, collections: Dict[str, Dict[str, dict]], snapshot_id: str,
                 origin: str, index: Optional[dict] = None,
                 notes: Optional[List[str]] = None) -> None:
        self.collections = collections
        self.snapshot_id = snapshot_id
        self.origin = origin
        self.index = index
        self.notes = list(notes or [])
        self.lib = load_registry_lib(SUITE_ROOT)

    # -- joins reproduced from registry_render.py ---------------------------

    def lane_of(self, m: dict) -> Optional[str]:
        """The lane a row's pipeline declares.  None = no declared lane
        (which is NOT the same as "the sealed lane" -- render's own rule)."""
        pl = self.collections.get("pipelines", {}).get(m.get("pipeline_ref")) or {}
        return (pl.get("lane") or {}).get("name")

    def recomputed_key(self, m: dict) -> str:
        """comparability key recomputed from the row's authoritative fields --
        deliberately NOT read from the stored comparability block."""
        return self.lib.comparability_key(self.lib.key_inputs_from_measurement(m))

    def key_label(self, key: str) -> Optional[str]:
        for entry in (self.index or {}).get("comparability_keys", []) or []:
            if entry.get("key") == key:
                return entry.get("label")
        return None

    def footer(self) -> str:
        note = ("; ".join(self.notes) + "; ") if self.notes else ""
        return "registry snapshot %s (%s%s)" % (self.snapshot_id, note, self.origin)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    """Unauthenticated GET.  The dataset is public; no token, ever."""
    req = urllib.request.Request(url, headers={"User-Agent": "fidelity-suite/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_jsonl(text: str, name: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise RegistryUnavailable("%s.jsonl:%d is not valid JSON: %s"
                                      % (name, lineno, exc))
        out[rec["id"]] = rec
    return out


def load_hf(quiet: bool = True) -> RegistrySnapshot:
    """Fetch the public mirror, cached under FIDELITY_CACHE_DIR keyed by the
    dataset's live commit sha (a cache hit at the same sha skips the refetch)."""
    try:
        meta = json.loads(_http_get(
            "%s/api/datasets/%s" % (HF_ENDPOINT, DATASET_ID)))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RegistryUnavailable(
            "cannot reach the public registry dataset (%s/datasets/%s): %s"
            % (HF_ENDPOINT, DATASET_ID, exc))
    # CLI-15. The cache is keyed on the commit sha the API just reported, and the
    # footer reports `snapshot_id = sha[:12]` -- but every file used to be fetched
    # from `raw/main/`. A push landing between the API call and the file fetches
    # therefore returned bytes from a DIFFERENT commit, cached permanently under, and
    # labelled with, the older sha; and a fetch loop spanning a push could mix two
    # commits inside one snapshot. The index.json cross-check below cannot see it,
    # because index.json came from main too. This is not hypothetical while a
    # measurement campaign is pushing rows into this dataset.
    #
    # Pin every fetch to the sha the cache is named after. HF serves
    # `/datasets/<id>/raw/<40-hex>/<path>` (verified 200), so the snapshot is now
    # genuinely a snapshot.
    sha = meta.get("sha")
    if not isinstance(sha, str) or not SHA40.match(sha):
        # The old code fell back to the literal "unknown", which made the FIRST fetch
        # the cache entry for every future run: a registry frozen at whatever it
        # happened to be, with a footer that named no commit.
        raise RegistryUnavailable(
            "the public registry dataset (%s/datasets/%s) did not report a commit sha "
            "(got %r), so a snapshot cannot be pinned or cached honestly"
            % (HF_ENDPOINT, DATASET_ID, meta.get("sha")))
    cdir = cache_dir() / "registry" / sha
    cdir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    def _cached_fetch(rel: str) -> bytes:
        local = cdir / rel.replace("/", "__")
        if local.is_file():
            return local.read_bytes()
        url = "%s/datasets/%s/raw/%s/%s" % (HF_ENDPOINT, DATASET_ID, sha, rel)
        blob = _http_get(url)
        handle, tmp = tempfile.mkstemp(dir=str(cdir), prefix=".fetch-", suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, local)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return blob

    try:
        index = json.loads(_cached_fetch("index.json").decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        index = None
        notes.append("index.json unavailable; group labels will be derived")

    collections: Dict[str, Dict[str, dict]] = {}
    for name in COLLECTION_NAMES:
        try:
            blob = _cached_fetch("data/%s.jsonl" % name)
        except (urllib.error.URLError, OSError) as exc:
            raise RegistryUnavailable(
                "cannot fetch data/%s.jsonl from the public dataset: %s"
                % (name, exc))
        declared = (((index or {}).get("collections") or {}).get(name) or {}).get("sha256")
        if declared:
            got = hashlib.sha256(blob).hexdigest()
            if got != declared:
                # A mid-push mirror is the innocent explanation; say so and
                # carry on with what was fetched rather than inventing data.
                notes.append(
                    "data/%s.jsonl sha256 %s… does not match index.json's %s… "
                    "(mirror possibly mid-update)" % (name, got[:8], declared[:8]))
        collections[name] = _parse_jsonl(blob.decode("utf-8"), name)
    return RegistrySnapshot(collections, sha[:12], "fetched from the public HF "
                            "dataset " + DATASET_ID, index=index, notes=notes)


def load_local(data_dir: Optional[Path] = None) -> RegistrySnapshot:
    data_dir = Path(data_dir) if data_dir else (SUITE_ROOT / "registry" / "data")
    if not data_dir.is_dir():
        raise RegistryUnavailable("no local registry clone at %s" % data_dir)
    lib = load_registry_lib(SUITE_ROOT)
    # registry_lib.load_registry re-reads the files on every call -- exactly
    # what a concurrently-edited registry needs; nothing is cached by path.
    collections = lib.load_registry(str(data_dir))
    for name in COLLECTION_NAMES:
        if name not in collections:
            raise RegistryUnavailable("local clone at %s is missing %s.jsonl"
                                      % (data_dir, name))
    head = "no-git"
    try:
        proc = run(["git", "-C", str(data_dir), "rev-parse", "--short", "HEAD"],
                   check=False)
        head = (proc.stdout or "").strip() or "no-git"
        dirty = run(["git", "-C", str(data_dir), "status", "--porcelain", "."],
                    check=False).stdout.strip()
        if dirty:
            head += "+dirty"
    except Exception:                                   # noqa: BLE001
        pass
    index_path = data_dir.parent / "index.json"
    index = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except ValueError:
            index = None
    return RegistrySnapshot(collections, "git " + head,
                            "LOCAL clone at %s" % data_dir, index=index)


def load(source: str = "auto", *, purpose: str = "rows",
         con: Optional[Console] = None) -> RegistrySnapshot:
    """source: auto | hf | local | local:PATH.

    auto policy (fixed): for purpose "check" (the already-measured gate) the
    HF mirror is the published truth and is tried FIRST, with a disclosed
    fallback to the local clone; for browsing ("rows"/"lineage") the local
    clone answers offline when present, else the mirror.
    """
    con = con or Console()
    source = source or "auto"
    if source == "hf":
        return load_hf()
    if source == "local":
        return load_local()
    if source.startswith("local:"):
        return load_local(Path(source.split(":", 1)[1]))
    if source != "auto":
        raise RegistryUnavailable("unknown --registry source %r (auto | hf | "
                                  "local | local:PATH)" % source)
    order = (("hf", "local") if purpose == "check" else ("local", "hf"))
    errors: List[str] = []
    for backend in order:
        try:
            if backend == "hf":
                return load_hf()
            snap = load_local()
            if purpose == "check":
                snap.notes.append(
                    "using LOCAL registry clone (%s); HF mirror unreachable -- "
                    "the local clone may be ahead of or behind published truth"
                    % snap.snapshot_id)
            return snap
        except RegistryUnavailable as exc:
            errors.append(str(exc))
    raise RegistryUnavailable(
        "no registry data source available.\n  - " + "\n  - ".join(errors) +
        "\n  Remedies: restore network access (the dataset is public, no auth "
        "needed), or pass --registry local:PATH pointing at a clone's data/ dir.")


# --------------------------------------------------------------------------
# Target parsing (URLs and repo ids)
# --------------------------------------------------------------------------


def parse_hf_target(text: str) -> Dict[str, Optional[str]]:
    """Accepts a full HF URL, org/name, @rev, /tree/rev, and a trailing subpath.

      zai-org/GLM-5.3-Flash
      zai-org/GLM-5.3-Flash@3f1971b7...
      https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
      orcarouter/GLM-5.3-Flash-MLX/4-bit         (subpath -> path hint)

    Returns {"repo", "revision", "path"} (revision/path may be None).
    """
    t = (text or "").strip()
    for prefix in ("https://huggingface.co/", "http://huggingface.co/",
                   "https://hf.co/", "http://hf.co/", "huggingface.co/", "hf.co/"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
            break
    t = t.strip("/")
    if t.lower().startswith("models/"):
        t = t[len("models/"):]
    revision: Optional[str] = None
    segments = [s for s in t.split("/") if s]
    if len(segments) < 2:
        raise ValueError(
            "%r does not name an HF repo (need org/name, optionally @rev or "
            "/tree/rev and a subpath)" % text)
    repo = "/".join(segments[:2])
    rest = segments[2:]
    if "@" in repo:
        repo, revision = repo.rsplit("@", 1)
    if rest and rest[0] == "tree":
        if len(rest) < 2:
            raise ValueError("%r has /tree/ with no revision" % text)
        revision = rest[1]
        rest = rest[2:]
    if rest and rest[0] in ("blob", "resolve", "raw"):
        if len(rest) < 2:
            raise ValueError("%r has /%s/ with no revision" % (text, rest[0]))
        revision = rest[1]
        rest = rest[2:]
    path = "/".join(rest) or None
    if revision == "main":
        revision = None                    # "main" means "the live head" = default
    return {"repo": repo, "revision": revision, "path": path}


# --------------------------------------------------------------------------
# Already-measured matching
# --------------------------------------------------------------------------


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return p.strip().strip("/") or None


def match_artifacts(reg: RegistrySnapshot, repo: str, rev: Optional[str],
                    path_hint: Optional[str] = None) -> Dict[str, Any]:
    """All artifact records for a repo, tiered against the target revision.

    repo->artifact is 1:N (the MLX repo holds 5 artifacts split by path; a
    GGUF repo holds 4 split by codec/name; zai BF16 appears at two revisions),
    so this returns ALL candidates and never guesses.  `ambiguous` is set when
    a MEASUREMENT-time caller would have to pick one and cannot: multiple
    candidates distinguished only by huggingface.path with no --path given.
    """
    repo_l = (repo or "").lower()
    hint = _norm_path(path_hint)
    cands: List[Tuple[dict, str, str]] = []
    paths = set()
    for art in reg.collections.get("artifacts", {}).values():
        hf = art.get("huggingface") or {}
        if (hf.get("repository") or "").lower() != repo_l:
            continue
        apath = _norm_path(hf.get("path"))
        if hint is not None and apath != hint:
            continue
        paths.add(apath)
        arev = hf.get("revision")
        if arev is None:
            tier = TIER_UNPINNED
            note = ("measured with revision UNPINNED (revision_source=%s); "
                    "identity rests on the content hashes in the artifact "
                    "record, not on a commit"
                    % (hf.get("revision_source") or "none"))
            for d in art.get("disclosures") or []:
                if d.get("code") == "revision_unpinned":
                    note += " -- " + str(d.get("detail", "")).strip()
                    break
        elif rev is None:
            tier = TIER_UNVERIFIED
            note = ("measured at %s (%s); your target revision is unknown "
                    "(HF unreachable), so EXACT-vs-STALE cannot be decided"
                    % (arev[:10], hf.get("revision_source") or "?"))
        elif arev == rev:
            tier = TIER_EXACT
            note = "measured at exactly this revision (%s)" % arev[:10]
        else:
            tier = TIER_STALE
            note = ("this repo was measured at %s (%s); you asked about %s"
                    % (arev[:10], hf.get("revision_source") or "?", rev[:10]))
        cands.append((art, tier, note))
    distinct_paths = {p for p in paths if p is not None}
    ambiguous = (hint is None and len(cands) > 1 and len(distinct_paths) > 1)
    return {"candidates": cands, "ambiguous": ambiguous,
            "paths": sorted(distinct_paths)}


def rows_for(reg: RegistrySnapshot, artifact_ids: List[str]) -> List[dict]:
    wanted = set(artifact_ids)
    out = []
    for m in reg.collections.get("measurements", {}).values():
        if m.get("artifact_ref") in wanted and m.get("status") == "published":
            out.append(m)
    return out


# --------------------------------------------------------------------------
# Rendering (never merges comparability groups)
# --------------------------------------------------------------------------


def _receipt_link(m: dict) -> str:
    for src in (m.get("provenance") or {}).get("sources") or []:
        if src.get("kind") in ("hf_file", "github_file") and src.get("uri"):
            return str(src["uri"])
    return "receipt not publicly fetchable (local path)"


def _artifact_name(reg: RegistrySnapshot, m: dict) -> str:
    art = reg.collections.get("artifacts", {}).get(m.get("artifact_ref")) or {}
    hf = art.get("huggingface") or {}
    repo = hf.get("repository")
    path = _norm_path(hf.get("path"))
    base = art.get("name") or m.get("artifact_ref") or "?"
    if repo:
        base += "  [%s%s]" % (repo, ("/" + path) if path else "")
    return base


def _row_lines(reg: RegistrySnapshot, m: dict) -> List[str]:
    metric = m.get("metric") or {}
    comp = m.get("comparability") or {}
    prov = m.get("provenance") or {}
    unc = m.get("uncertainty") or {}
    lines = ["%-12s %s" % (("%.6f %s" % (metric.get("value"), metric.get("units") or "")).strip(),
                           _artifact_name(reg, m))]
    badge = prov.get("measured_by") or "?"
    handle = ((prov.get("measurer") or {}).get("handle"))
    if handle and handle != badge:
        badge += ":" + str(handle)
    lines.append("    id %s  panel %s  class %s  measured_by %s"
                 % (m.get("id"), m.get("panel_ref"), comp.get("class"), badge))
    if unc.get("ci95_low") is not None and unc.get("ci95_high") is not None:
        lines.append("    ci95 [%s, %s]  method %s"
                     % (unc["ci95_low"], unc["ci95_high"], unc.get("method")))
    bias = comp.get("bias")
    if bias:
        detail = str(bias.get("detail", "")).strip()
        if len(detail) > 220:
            detail = detail[:217] + "..."
        lines.append("    BIAS (%s, %s): %s"
                     % (bias.get("kind"), bias.get("direction"), detail))
        if bias.get("floor_measurement_ref") is None and \
                bias.get("kind") == "cross_stack_capture_replay":
            lines.append("    (measurement floor)")
    lines.append("    receipt: %s" % _receipt_link(m))
    return lines


def _subset_caveat(reg: RegistrySnapshot, panel_ref: str) -> Optional[str]:
    panel = reg.collections.get("panels", {}).get(panel_ref) or {}
    for d in panel.get("disclosures") or []:
        if d.get("code") == "subset_of_panel":
            return ("CAVEAT: %s is a SUBSET panel (%s) -- its rows must never "
                    "be read beside a parent-panel table as if they were the "
                    "same estimand" % (panel_ref, str(d.get("detail", "")).strip()[:160]))
    return None


def _undisclosed_caveat(reg: RegistrySnapshot, panel_ref: str) -> Optional[str]:
    """Sibling of _subset_caveat: an UNDISCLOSED panel's low numbers are the
    most seductive on the page and mean the least outside their own table."""
    panel = reg.collections.get("panels", {}).get(panel_ref)
    if panel is None:
        return ("CAVEAT: panel %s has no panel record in this registry "
                "snapshot -- its rows are comparable only to each other, and "
                "their values must not be read against any other table"
                % panel_ref)
    for d in panel.get("disclosures") or []:
        if d.get("code") == "undisclosed_panel":
            return ("CAVEAT: %s is an UNDISCLOSED panel (%s) -- its values "
                    "(however low) are comparable only within this group and "
                    "must never be read beside any disclosed-panel table"
                    % (panel_ref, str(d.get("detail", "")).strip()[:160]))
    return None


def _full_group_sizes(reg: RegistrySnapshot) -> Dict[str, int]:
    """Group sizes over the WHOLE snapshot (published rows), so a filtered
    view can say '1 of N matched your filters' instead of falsely claiming
    'nothing to rank against' (usability review, 2026-08-28)."""
    sizes: Dict[str, int] = {}
    for m in reg.collections.get("measurements", {}).values():
        if m.get("status") != "published":
            continue
        try:
            key = reg.recomputed_key(m)
        except Exception as exc:                        # noqa: BLE001
            key = "cmp--UNCOMPUTABLE(%s)" % type(exc).__name__
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def group_rows(reg: RegistrySnapshot, rows: List[dict]) -> "Dict[str, Dict[str, Any]]":
    """{recomputed_key: {"label":..., "lanes": {lane_or_"": [rows]}}}.

    The grouping is STRUCTURAL: callers render one table per key and cannot
    merge them, because the merge would need code that does not exist here.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for m in rows:
        try:
            key = reg.recomputed_key(m)
        except Exception as exc:                        # noqa: BLE001
            key = "cmp--UNCOMPUTABLE(%s)" % type(exc).__name__
        g = groups.setdefault(key, {"label": reg.key_label(key), "lanes": {}})
        lane = reg.lane_of(m) or ""
        g["lanes"].setdefault(lane, []).append(m)
    for g in groups.values():
        for lane, ms in g["lanes"].items():
            # sorting by value is allowed only WITHIN a lane sub-table
            ms.sort(key=lambda m: ((m.get("metric") or {}).get("value") is None,
                                   (m.get("metric") or {}).get("value")))
    return groups


def render_rows(reg: RegistrySnapshot, rows: List[dict], con: Console) -> None:
    if not rows:
        con.say("  (no measurement rows)")
        return
    groups = group_rows(reg, rows)
    full_sizes = _full_group_sizes(reg)
    for key in sorted(groups):
        g = groups[key]
        label = g["label"] or "(no index label; key inputs recomputed from the rows)"
        con.say("")
        con.say("COMPARABILITY GROUP %s" % key)
        con.say("  %s" % label)
        total = sum(len(v) for v in g["lanes"].values())
        full = full_sizes.get(key, total)
        if total < full:
            con.say("  (%d of %d rows in this comparability group shown -- "
                    "the registry holds %d; the whole group ranks together "
                    "via bin/registry-view rows with no filters)"
                    % (total, full, full))
        elif total == 1:
            con.say("  (single row -- nothing to rank against in this group)")
        caveats = set()
        for lane in sorted(g["lanes"], key=lambda x: (x != "", x)):
            ms = g["lanes"][lane]
            if lane:
                con.say("  -- lane: %s (tabled apart; lane offsets are real) --" % lane)
            else:
                if len(g["lanes"]) > 1:
                    sealed = any((m.get("comparability") or {}).get("class")
                                 == "strict" for m in ms)
                    con.say("  -- no declared lane --%s"
                            % (" (sealed rows land here: class strict is the "
                               "sealed number)" if sealed else ""))
            for m in ms:
                for line in _row_lines(reg, m):
                    con.say("  " + line)
                for cav in (_subset_caveat(reg, m.get("panel_ref")),
                            _undisclosed_caveat(reg, m.get("panel_ref"))):
                    if cav:
                        caveats.add(cav)
        for cav in sorted(caveats):
            con.say("  " + cav)
    if len(groups) > 1:
        con.say("")
        con.say("NOTE: %d comparability groups above. Values in different groups "
                "are NOT comparable and are never ranked together." % len(groups))


def render_check(reg: RegistrySnapshot, repo: str, rev: Optional[str],
                 match: Dict[str, Any], con: Console) -> List[dict]:
    """Print the tiered candidates + their rows; return the rows."""
    cands = match["candidates"]
    if not cands:
        con.say("  no artifact record for %s in the registry" % repo)
        con.say("  " + reg.footer())
        return []
    con.say("  %d artifact record(s) for %s:" % (len(cands), repo))
    ids = []
    for art, tier, note in cands:
        ids.append(art["id"])
        con.say("    [%s] %s" % (tier, art["id"]))
        con.say("        %s" % note)
    if match["ambiguous"]:
        con.say("    AMBIGUOUS for measurement: this repo holds %d artifacts "
                "distinguished only by path (%s). Pass --path to pick one; "
                "the rows below cover all of them."
                % (len(cands), ", ".join(match["paths"])))
    rows = rows_for(reg, ids)
    if rows:
        con.say("  %d published measurement row(s):" % len(rows))
        render_rows(reg, rows, con)
    else:
        con.say("  artifact known, but no published measurement rows")
    con.say("")
    con.say("  " + reg.footer())
    return rows


# --------------------------------------------------------------------------
# The front gate shared by measure-local / measure-cloud / measure (one-cmd)
# --------------------------------------------------------------------------


def stale_scope_hint(match: Dict[str, Any]) -> List[str]:
    """Say so when the STALE rows are scoped to a *part* of a multi-artifact repo.

    "Revision drift" is the right words for a repo whose head moved under a
    measurement. It is the WRONG words for a repo that publishes several
    artifacts on several branches: turboderp/GLM-5.3-Flash-exl3 ships 4.05bpw,
    3.05bpw and 2.05bpw as three branches, the registry models them as three
    artifacts (`huggingface.path: "branch 4.05bpw"`), and the gate keys on the
    repo id alone -- so asking about 2.05bpw reports the 4.05bpw row and calls
    the difference drift. The remedy flag is the same (--force) but the reason
    is not, and a reader told "drift" reasonably concludes the artifact is
    already measured and stops. The registry already knows the scope; this
    prints it.
    """
    scopes = []
    for artifact, tier, _ in match.get("candidates", []):
        if tier != TIER_STALE:
            continue
        path = ((artifact.get("huggingface") or {}).get("path") or "").strip()
        if path and path not in scopes:
            scopes.append(path)
    if not scopes:
        return []
    return ["",
            "        NOTE: the measured row(s) above are scoped to %s of this "
            "repo." % ", ".join(repr(x) for x in scopes),
            "        If you are asking about a DIFFERENT branch or subpath, "
            "this is not drift --",
            "        it is a separate artifact that has never been measured, "
            "and --force is",
            "        the right flag: it records a new artifact rather than "
            "restating an old one."]


def front_gate(*, repo: str, revision: Optional[str], path_hint: Optional[str],
               source: str, force: bool, accept_measured_revision: bool,
               con: Console, already_measured_advice: Optional[str] =
               "Pass --force to measure anyway (e.g. to reproduce)."
               ) -> Dict[str, Any]:
    """The scorer checks the registry BEFORE planning or spending anything.

    Returns {"status": ...}:
      already-measured  -> caller reports and exits 0 (unless --force was given,
                           in which case status is proceed-forced)
      stale-refused     -> caller refuses (needs --force or
                           --accept-measured-revision)
      proceed / proceed-forced / proceed-stale-accepted / unavailable
    """
    con.say("REGISTRY CHECK (before anything is planned or spent)")
    try:
        reg = load(source, purpose="check", con=con)
    except RegistryUnavailable as exc:
        con.warn("registry check unavailable: %s" % exc)
        con.warn("continuing to plan; re-run once a data source is reachable "
                 "to avoid re-measuring something already published")
        return {"status": "unavailable"}

    resolved = revision
    if resolved is not None and not SHA40.match(resolved):
        try:
            from .hfmeta import resolve_commit
            resolved = resolve_commit(repo, resolved)
        except Exception as exc:                        # noqa: BLE001
            # An EXPLICITLY requested revision that HF cannot resolve is
            # never silently replaced with live main; the tier degrades to
            # PINNED-UNVERIFIED where a pin exists (same rule as cmd_check).
            con.warn("cannot resolve revision %r via HF (%s); tier will be "
                     "PINNED-UNVERIFIED where a pin exists" % (revision, exc))
            resolved = None
    if resolved is None and revision is None:
        try:
            from .hfmeta import model_lineage_meta
            meta = model_lineage_meta(repo)
            resolved = meta.sha
            if resolved:
                con.say("  target revision: %s (live main -- what a download "
                        "today would fetch)" % resolved[:12])
            if meta.repo_id and meta.repo_id.lower() != repo.lower():
                # HF 307-redirects renamed repos; the registry records the
                # canonical name, so matching the typed alias would be a
                # false "never measured".
                con.say("  HF redirects %s to %s; matching the registry "
                        "under the canonical name" % (repo, meta.repo_id))
                repo = meta.repo_id
        except Exception as exc:                        # noqa: BLE001
            from .hfmeta import hf_unavailable_text
            con.warn(hf_unavailable_text(repo, exc))
            resolved = None

    match = match_artifacts(reg, repo, resolved, path_hint)
    if not match["candidates"] and revision is not None:
        # The explicit-revision paths above make no HF metadata call, so a
        # RENAMED repo (HF 307-redirects old names) would false-negative
        # under its typed alias and the gate would proceed to a duplicate
        # paid measurement.  One tolerant canonicalization attempt closes
        # that hole; failure (offline, pinned-sha flows) stays silent --
        # the alias lookup already returned nothing either way.
        try:
            from .hfmeta import model_lineage_meta
            meta = model_lineage_meta(repo)
            if meta.repo_id and meta.repo_id.lower() != repo.lower():
                con.say("  HF redirects %s to %s; matching the registry "
                        "under the canonical name" % (repo, meta.repo_id))
                repo = meta.repo_id
                match = match_artifacts(reg, repo, resolved, path_hint)
        except Exception:                               # noqa: BLE001
            pass
    rows = render_check(reg, repo, resolved, match, con)
    tiers = {tier for _, tier, _ in match["candidates"]}
    result = {"status": "proceed", "rows": len(rows), "tiers": sorted(tiers),
              "snapshot": reg.snapshot_id, "resolved_revision": resolved}
    if not rows:
        return result
    if tiers & {TIER_EXACT, TIER_UNPINNED, TIER_UNVERIFIED}:
        if force:
            con.warn("--force: measuring despite %d published row(s) above"
                     % len(rows))
            result["status"] = "proceed-forced"
            return result
        con.say("")
        message = "ALREADY MEASURED: the rows above answer this request."
        if already_measured_advice:
            message += " " + already_measured_advice
        con.say(message)
        result["status"] = "already-measured"
        return result
    # STALE only
    if force:
        con.warn("--force: revision drift accepted; measuring at %s will "
                 "create a NEW artifact record at the new revision"
                 % (resolved or "?")[:12])
        result["status"] = "proceed-forced"
        return result
    if accept_measured_revision:
        pinned = [(a.get("huggingface") or {}).get("revision")
                  for a, t, _ in match["candidates"] if t == TIER_STALE]
        result["status"] = "proceed-stale-accepted"
        result["measured_revision"] = pinned[0] if pinned else None
        con.warn("--accept-measured-revision: treating the measured revision "
                 "%s as the target" % (result["measured_revision"] or "?")[:12])
        return result
    con.say("")
    con.say("REVISION DRIFT: this repo was measured at a different commit "
            "than the one you asked about (rows above). Either pass "
            "--accept-measured-revision to target the measured commit, or "
            "--force to measure the new commit as a NEW artifact record.")
    for line in stale_scope_hint(match):
        con.say(line)
    result["status"] = "stale-refused"
    return result
