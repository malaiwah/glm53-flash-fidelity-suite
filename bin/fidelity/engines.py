"""Engine adapters: how each lane actually invokes a scorer.

WHY THIS INDIRECTION EXISTS.  The runners are orchestration -- fit, cost,
teardown, receipts.  The measurement itself is done by the engines that already
exist under `k6/tools/`.  Hard-coding their flags into the runners would mean
that every engine change silently breaks a recipe that strangers are pasting.
Instead, each lane names an engine in `bin/engines.json`, and a lane whose
engine is not PINNED refuses to plan.

That refusal is the point.  At the time of writing, `k6/tools/stream_score.py`
is not present in this checkout -- it exists only on the validation box -- so
the `streaming`, `local-mps` and `local-cuda-budget` lanes are declared
`pinned: false` with the exact contract they need.  `--dry-run` reports that
as an unresolved engine and names the file to fill in.  It does not invent
plausible flags, because a plausible-looking wrong flag is how you spend an
hour of H200 time discovering that `--reduce-order` was actually spelled
`--reduce_order`.

When the file lands: run `bin/measure-local --probe-engines`, which scrapes
`--help` from every engine it can find and reports, per lane, which required
flags are present and which are missing.  Then set `pinned: true`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import run, which

ENGINES_FILE = Path(__file__).resolve().parent.parent / "engines.json"


@dataclass
class Engine:
    lane: str
    name: str
    entrypoint: str
    pinned: bool
    launcher: List[str]
    required_flags: List[str]
    flag_map: Dict[str, str]
    scorer: Optional[Dict[str, Any]]
    notes: str
    unpinned_reason: str = ""
    contract: List[str] = field(default_factory=list)
    surfaces: List[str] = field(default_factory=list)
    surfaces_note: str = ""
    timing: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    fixed_flags: Dict[str, str] = field(default_factory=dict)
    profile_map: Dict[str, str] = field(default_factory=dict)
    profile_map_by_surface: Dict[str, Dict[str, str]] = field(default_factory=dict)
    receipt_class: str = ""
    pinned_note: str = ""

    def resolve(self, suite_root: Path) -> Optional[Path]:
        p = (suite_root / self.entrypoint).resolve()
        return p if p.is_file() else None

    def probe(self, suite_root: Path) -> Dict[str, Any]:
        """Scrape --help and report which required flags actually exist.

        The whole value of this function is that it answers "will my
        invocation work?" without a GPU, a download, or a rental.
        """
        path = self.resolve(suite_root)
        result: Dict[str, Any] = {
            "lane": self.lane,
            "entrypoint": self.entrypoint,
            "present": path is not None,
            "pinned": self.pinned,
            "missing_flags": [],
            "found_flags": [],
            "help_ok": False,
            "problems": [],
        }
        if path is None:
            result["problems"].append(
                "engine file not present at %s" % self.entrypoint)
            return result
        proc = run(["python3", str(path), "--help"], check=False, timeout=120)
        text = (proc.stdout or "") + (proc.stderr or "")
        # A missing heavy import (torch, quant_pipeline) makes --help fail on a
        # laptop.  That is not the engine's fault and not a reason to refuse;
        # fall back to reading the argparse calls out of the source.
        if proc.returncode == 0 and "--" in text:
            result["help_ok"] = True
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            result["problems"].append(
                "--help did not run (likely a missing heavy import); read the "
                "argparse declarations from source instead")
        declared = set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', text))
        declared |= set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]+)", text))
        for flag in self.required_flags:
            (result["found_flags"] if flag in declared
             else result["missing_flags"]).append(flag)
        if result["missing_flags"]:
            result["problems"].append(
                "required flags absent: " + ", ".join(result["missing_flags"]))
        return result


def load_engines(path: Optional[Path] = None) -> Dict[str, Engine]:
    raw = json.loads((path or ENGINES_FILE).read_text(encoding="utf-8"))
    out: Dict[str, Engine] = {}
    for lane, spec in raw["lanes"].items():
        out[lane] = Engine(
            lane=lane,
            name=spec["name"],
            entrypoint=spec["entrypoint"],
            pinned=bool(spec.get("pinned")),
            launcher=list(spec.get("launcher") or []),
            required_flags=list(spec.get("required_flags") or []),
            flag_map=dict(spec.get("flag_map") or {}),
            scorer=spec.get("scorer"),
            notes=spec.get("notes", ""),
            unpinned_reason=spec.get("unpinned_reason", ""),
            contract=list(spec.get("contract") or []),
            surfaces=list(spec.get("surfaces") or []),
            surfaces_note=spec.get("surfaces_note", ""),
            timing=dict(spec.get("timing") or {}),
            env=dict(spec.get("env") or {}),
            fixed_flags=dict(spec.get("fixed_flags") or {}),
            profile_map=dict(spec.get("profile_map") or {}),
            profile_map_by_surface={k: dict(v) for k, v in
                                    (spec.get("profile_map_by_surface") or {}).items()},
            receipt_class=spec.get("receipt_class", ""),
            pinned_note=spec.get("pinned_note", ""),
        )
    return out


class EngineUnpinned(RuntimeError):
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        lines = [
            "lane %r has no pinned engine invocation." % engine.lane,
            "  engine:    %s (%s)" % (engine.name, engine.entrypoint),
            "  reason:    %s" % (engine.unpinned_reason or "not pinned"),
        ]
        if engine.contract:
            lines.append("  it must accept, at minimum:")
            lines.extend("    %s" % c for c in engine.contract)
        lines += [
            "",
            "  Nothing was created and nothing was spent.",
            "  Fix: put the engine in place, run `bin/measure-local --probe-engines`",
            "       to confirm its real flags, then set pinned:true and the",
            "       flag_map for this lane in bin/engines.json.",
            "  Meanwhile `--lane sealed-ep8` IS pinned and works.",
        ]
        super().__init__("\n".join(lines))


def build_invocation(
    engine: Engine,
    *,
    suite_root: Path,
    checkpoint: str,
    panel_dir: str,
    out_dir: str,
    surface: str,
    profile: str,
    cold_run: int,
    reduce_order: str,
    roles: str,
    extra: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Turn lane-neutral intent into that engine's actual argv.

    `flag_map` maps our vocabulary onto the engine's spelling, so a rename in
    an engine is a one-line JSON edit rather than a code change in two runners.
    """
    if not engine.pinned:
        raise EngineUnpinned(engine)
    values = {
        "checkpoint": checkpoint,
        "panel": panel_dir,
        "out": out_dir,
        "surface": surface,
        "profile": profile,
        "cold_run": str(cold_run),
        "reduce_order": reduce_order,
        "roles": roles,
    }
    values.update(extra or {})
    argv = list(engine.launcher) + [str((suite_root / engine.entrypoint).resolve())]
    for key, flag in engine.flag_map.items():
        value = values.get(key)
        if value in (None, ""):
            continue
        if flag.endswith("="):            # bare switch, no value
            argv.append(flag[:-1])
        else:
            argv.extend([flag, str(value)])
    return argv


# --------------------------------------------------------------------------
# Preflight: everything --execute needs, checked BEFORE anything is spent
# --------------------------------------------------------------------------

FIDELITY_PYTHON_DEFAULT = "/opt/homebrew/bin/python3.14"


def fidelity_python() -> str:
    """The torch-capable interpreter engines run under.

    FIDELITY_PYTHON env wins; the documented default is the homebrew 3.14
    (torch 2.13) on the operator's Mac; plain python3 is the last resort so a
    box with system-wide torch still works."""
    import shutil

    env = os.environ.get("FIDELITY_PYTHON")
    if env:
        return env
    if Path(FIDELITY_PYTHON_DEFAULT).exists():
        return FIDELITY_PYTHON_DEFAULT
    return shutil.which("python3") or "python3"


def _can_import(python: str, module: str, version_attr: str = "__version__"):
    proc = run([python, "-c",
                "import %s; print(getattr(%s, %r, 'unknown'))"
                % (module, module, version_attr)], check=False, timeout=120)
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or "unknown"


def preflight(engine: Engine, *, suite_root: Path,
              pipeline_root: Optional[str] = None,
              teacher_dir: Optional[Path] = None,
              need_disk_bytes: Optional[float] = None,
              workdir: Optional[Path] = None) -> List[Dict[str, str]]:
    """Return [] when an --execute could actually start, else every missing
    prerequisite WITH its remedy.  Never raises for a missing dependency --
    the caller turns the list into one refusal naming all of them, because
    discovering prerequisites one at a time is five refusals where one would
    do."""
    # PEP 668: Homebrew/distro Pythons refuse bare `pip install` -- every
    # printed pip remedy must carry the escape or it fails verbatim
    # (usability review, 2026-08-28).
    PEP668_NOTE = (" (on a Homebrew/distro Python add --break-system-packages,"
                   " or use a venv and export FIDELITY_PYTHON to its python)")
    problems: List[Dict[str, str]] = []
    python = fidelity_python()
    if not Path(python).exists() and not which(python):
        problems.append({
            "missing": "FIDELITY_PYTHON interpreter (%s)" % python,
            "remedy": "export FIDELITY_PYTHON=/path/to/python3.1x with torch "
                      "installed (the documented default is %s)"
                      % FIDELITY_PYTHON_DEFAULT})
        return problems                      # nothing else is checkable
    torch_version = _can_import(python, "torch")
    if torch_version is None:
        problems.append({
            "missing": "torch under %s" % python,
            "remedy": '"%s" -m pip install torch' % python + PEP668_NOTE})
    tf_version = _can_import(python, "transformers")
    if tf_version is None:
        problems.append({
            "missing": "transformers under %s (engine needs >= 5.16)" % python,
            "remedy": '"%s" -m pip install "transformers>=5.16"' % python + PEP668_NOTE})
    else:
        try:
            major, minor = (int(x) for x in tf_version.split(".")[:2])
            if (major, minor) < (5, 16):
                problems.append({
                    "missing": "transformers>=5.16 (found %s)" % tf_version,
                    "remedy": '"%s" -m pip install -U "transformers>=5.16"' % python + PEP668_NOTE})
        except ValueError:
            pass
    qp_env = {"QP_PIPELINE_ROOT": pipeline_root} if pipeline_root else None
    if pipeline_root:
        src_ok = any((Path(pipeline_root) / c / "quant_pipeline" / "__init__.py").is_file()
                     for c in ("runtime/src", "src", "."))
        if not src_ok:
            problems.append({
                "missing": "quant_pipeline package under --pipeline-root %s" % pipeline_root,
                "remedy": "point --pipeline-root at the patched tree (clone "
                          "PIPE_REPO per k6/stage_k6.sh + apply patches-v2)"})
    elif _can_import(python, "quant_pipeline") is None:
        problems.append({
            "missing": "quant_pipeline (the engine's reader package)",
            "remedy": "clone PIPE_REPO per k6/stage_k6.sh + apply patches-v2, "
                      "then pass --pipeline-root PATH (or pip-install it into "
                      "FIDELITY_PYTHON)"})
    _ = qp_env
    if teacher_dir is not None:
        receipt = Path(teacher_dir) / "capture-receipt.json"
        found = None
        if receipt.is_file():
            found = receipt
        elif Path(teacher_dir).is_dir():
            for candidate in sorted(Path(teacher_dir).glob("**/capture-receipt.json")):
                found = candidate
                break
        if found is None:
            problems.append({
                "missing": "teacher tree with a sealed capture receipt at %s" % teacher_dir,
                "remedy": "fetch the panel's teacher logits (the default panel's "
                          "include globs pull logits/window-*.safetensors + *.json, "
                          "31.7 GB) into that directory"})
        else:
            try:
                doc = json.loads(found.read_text(encoding="utf-8"))
                if doc.get("capture_role") != "bf16_teacher" or \
                        "-preview." in str(doc.get("schema", "")):
                    problems.append({
                        "missing": "a bf16_teacher capture receipt (found role %r, "
                                   "schema %r)" % (doc.get("capture_role"),
                                                   doc.get("schema")),
                        "remedy": "point --teacher-tree at a real teacher (previews "
                                  "and student captures cannot be teachers)"})
            except (OSError, ValueError):
                problems.append({
                    "missing": "readable capture-receipt.json under %s" % teacher_dir,
                    "remedy": "re-fetch the teacher tree; the receipt is corrupt"})
    if need_disk_bytes and workdir is not None:
        import shutil as _shutil

        probe = workdir if workdir.exists() else workdir.parent
        try:
            free = _shutil.disk_usage(probe).free
        except OSError:
            free = 0
        if free < need_disk_bytes:
            problems.append({
                "missing": "disk: need %.0f GB free at %s, have %.0f GB"
                           % (need_disk_bytes / 1e9, workdir, free / 1e9),
                "remedy": "free %.0f GB or pass --work on a bigger volume"
                          % ((need_disk_bytes - free) / 1e9)})
    return problems
