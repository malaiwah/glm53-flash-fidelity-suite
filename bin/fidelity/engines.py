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

from .common import run

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
    timing: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)

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
            timing=dict(spec.get("timing") or {}),
            env=dict(spec.get("env") or {}),
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
