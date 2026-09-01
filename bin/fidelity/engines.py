"""Engine adapters: how each lane actually invokes a scorer.

WHY THIS INDIRECTION EXISTS.  The runners are orchestration -- fit, cost,
teardown, receipts.  The measurement itself is done by the engines that already
exist under `engines/tools/`.  Hard-coding their flags into the runners would mean
that every engine change silently breaks a recipe that strangers are pasting.
Instead, each lane names an engine in `bin/engines.json`, and a lane whose
engine is not PINNED refuses to plan.

That refusal is the point.  At the time of writing, `engines/tools/stream_score.py`
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

class EngineConfigError(ValueError):
    """The authored engine/timing registry is not strict finite JSON."""


def _load_engine_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse the one scientific engine registry without last-key-wins JSON."""
    selected = path or ENGINES_FILE

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise EngineConfigError(
                    "engine config contains duplicate key %r" % key)
            value[key] = item
        return value

    def reject_constant(value):
        raise EngineConfigError(
            "engine config contains non-finite JSON constant %s" % value)

    try:
        raw = selected.read_bytes()
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=reject_constant)
    except EngineConfigError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineConfigError(
            "engine config is not strict UTF-8 JSON: %s" % exc) from exc
    if not isinstance(document, dict):
        raise EngineConfigError("engine config root must be an object")
    return document


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
    profile_refusals_by_surface: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict)
    receipt_class: str = ""
    pinned_note: str = ""

    def resolve(self, suite_root: Path) -> Optional[Path]:
        p = (suite_root / self.entrypoint).resolve()
        return p if p.is_file() else None

    def probe(self, suite_root: Path, *, paid: bool = False,
              python: str = "python3",
              env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Probe authored argv flags; paid mode accepts live ``--help`` only."""
        def normalized(flags):
            return sorted({
                flag[:-1] if flag.endswith("=") else flag
                for flag in flags if isinstance(flag, str) and flag.startswith("--")
            })

        def one(entrypoint: str, expected: List[str]) -> Dict[str, Any]:
            path = (suite_root / entrypoint).resolve()
            result = {
                "entrypoint": entrypoint,
                "present": path.is_file(),
                "help_ok": False,
                "expected_flags": normalized(expected),
                "found_flags": [],
                "missing_flags": [],
                "problems": [],
            }
            if not result["present"]:
                result["missing_flags"] = list(result["expected_flags"])
                result["problems"].append(
                    "engine file not present at %s" % entrypoint)
                return result
            proc = run(
                [python, str(path), "--help"], check=False, timeout=120,
                env=env)
            text = (proc.stdout or "") + (proc.stderr or "")
            live_ok = proc.returncode == 0 and "--" in text
            if live_ok:
                result["help_ok"] = True
            elif paid:
                result["missing_flags"] = list(result["expected_flags"])
                result["problems"].append(
                    "live --help failed with exit %d; paid admission never "
                    "falls back to source regex" % proc.returncode)
                return result
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                result["problems"].append(
                    "--help did not run; non-paid probe read argparse source")
            declared = set(re.findall(
                r"(?<![\w-])(--[a-z0-9][a-z0-9-]+)", text))
            result["found_flags"] = sorted(
                set(result["expected_flags"]) & declared)
            result["missing_flags"] = sorted(
                set(result["expected_flags"]) - declared)
            if result["missing_flags"]:
                result["problems"].append(
                    "authored invocation flags absent: "
                    + ", ".join(result["missing_flags"]))
            return result

        engine_expected = (
            list((self.flag_map or {}).values())
            + list((self.fixed_flags or {}).keys())
            if paid else list(self.required_flags))
        engine_probe = one(self.entrypoint, engine_expected)
        scorer_probe = None
        if paid:
            scorer = self.scorer or {}
            scorer_entrypoint = scorer.get("entrypoint")
            scorer_flags = (scorer.get("flag_map") or {}).values()
            if not isinstance(scorer_entrypoint, str) or not scorer_entrypoint:
                scorer_probe = {
                    "entrypoint": None, "present": False, "help_ok": False,
                    "expected_flags": normalized(scorer_flags),
                    "found_flags": [], "missing_flags": normalized(scorer_flags),
                    "problems": ["paid lane pins no scorer entrypoint"],
                }
            else:
                scorer_probe = one(scorer_entrypoint, list(scorer_flags))
        result = {
            "schema": "fidelity-suite/engine-probe.v2",
            "lane": self.lane,
            "pinned": self.pinned,
            "mode": "paid-live-help" if paid else "diagnostic",
            "engine": engine_probe,
            "scorer": scorer_probe,
            "help_ok": (
                engine_probe["help_ok"]
                and not engine_probe["missing_flags"]
                and (not paid or (
                    scorer_probe is not None
                    and scorer_probe["help_ok"]
                    and not scorer_probe["missing_flags"]))),
            # Compatibility fields used by the local diagnostic caller.
            "entrypoint": self.entrypoint,
            "present": engine_probe["present"],
            "found_flags": list(engine_probe["found_flags"]),
            "missing_flags": list(engine_probe["missing_flags"]),
            "problems": (
                list(engine_probe["problems"])
                + (list(scorer_probe["problems"]) if scorer_probe else [])),
        }
        return result


def load_engines(path: Optional[Path] = None) -> Dict[str, Engine]:
    raw = _load_engine_config(path)
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
            profile_refusals_by_surface={
                surface: {rate: dict(reason) for rate, reason in rows.items()}
                for surface, rows in
                (spec.get("profile_refusals_by_surface") or {}).items()
            },
            receipt_class=spec.get("receipt_class", ""),
            pinned_note=spec.get("pinned_note", ""),
        )
    return out


def _rate_keys(bits: Optional[float]) -> List[str]:
    if bits is None:
        return []
    keys = []
    for candidate in (bits, round(float(bits), 4)):
        for text in ("%g" % float(candidate), str(candidate)):
            if text not in keys:
                keys.append(text)
    return keys


class EngineProfileRefused(RuntimeError):
    """A known surface/rate that has evidence for a refusal, not an omission."""


class EngineTimingUnavailable(RuntimeError):
    """A paid profile has no truthful target-specific timing basis."""


class RootTimingUnavailable(RuntimeError):
    """No exact root target/GPU/form/schedule timing evidence exists."""


def profile_refusal(engine: Engine, *, surface: str,
                    bits: Optional[float]) -> Optional[Dict[str, Any]]:
    rows = engine.profile_refusals_by_surface.get(surface) or {}
    for key in _rate_keys(bits):
        if key in rows:
            return dict(rows[key])
    return None


def require_supported_profile(engine: Engine, *, surface: str,
                              bits: Optional[float]) -> None:
    refusal = profile_refusal(engine, surface=surface, bits=bits)
    if refusal is None:
        return
    raise EngineProfileRefused(
        "REFUSED before spend [%s]: %s"
        % (refusal.get("code", "unsupported_profile"),
           refusal.get("detail", "profile is unsupported"))
    )


def resolve_profile_timing(engine: Engine, *, profile: str, surface: str,
                           bits: Optional[float] = None,
                           target_repo: Optional[str] = None,
                           target_revision: Optional[str] = None,
                           gpu: Optional[str] = None) -> Dict[str, Any]:
    """Return serializable, profile-specific timing or refuse.

    A lane-wide fallback is intentionally not accepted here: different TR3
    rates and native BF16 move different byte counts, so substituting a generic
    minutes/window figure is not a scientific estimate.
    """
    require_supported_profile(engine, surface=surface, bits=bits)
    row = ((engine.timing.get("profiles") or {}).get(profile))
    if not isinstance(row, dict):
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_evidence_absent]: lane %s profile %s "
            "on surface %s has no exact timing profile"
            % (engine.lane, profile, surface)
        )
    minutes = row.get("minutes_per_window")
    if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes <= 0:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_value_invalid]: profile %s has no "
            "positive minutes_per_window" % profile
        )
    if not isinstance(row.get("runtime_profile"), dict) or not row["runtime_profile"]:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_runtime_profile_absent]: profile %s "
            "does not name the measured runtime profile" % profile
        )
    runtime_profile = row["runtime_profile"]
    if row.get("resource_admission_required") is True:
        missing_resources = [
            field for field in ("gpu_count", "decode_threads",
                                "min_vcpu_count", "min_memory_gb")
            if not isinstance(runtime_profile.get(field), int)
            or isinstance(runtime_profile.get(field), bool)
            or runtime_profile[field] <= 0
        ]
        if runtime_profile.get("decode_cache") == "ram":
            reader_threads = runtime_profile.get("reader_threads")
            if (
                not isinstance(reader_threads, int)
                or isinstance(reader_threads, bool)
                or reader_threads <= 0
            ):
                missing_resources.append("reader_threads")
        if missing_resources:
            raise EngineTimingUnavailable(
                "REFUSED before spend [timing_resources_absent]: profile %s "
                "has no safe admission value for %s"
                % (profile, ", ".join(sorted(missing_resources)))
            )
    if not isinstance(row.get("evidence"), dict) or not row["evidence"]:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_provenance_absent]: profile %s has "
            "no timing evidence" % profile
        )
    expected_repo = row.get("target_repo")
    expected_revision = row.get("target_revision")
    expected_gpu = (row.get("runtime_profile") or {}).get("gpu")
    if expected_repo is not None and target_repo != expected_repo:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_target_mismatch]: profile %s timing "
            "is for %s, got %r" % (profile, expected_repo, target_repo)
        )
    if expected_revision is not None and target_revision != expected_revision:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_revision_mismatch]: profile %s timing "
            "is for revision %s, got %r"
            % (profile, expected_revision, target_revision)
        )
    if expected_gpu is not None and gpu != expected_gpu:
        raise EngineTimingUnavailable(
            "REFUSED before spend [timing_gpu_mismatch]: profile %s timing is "
            "for %s, got %r" % (profile, expected_gpu, gpu)
        )
    # JSON round-trip both copies the record and proves it is safe to bind into
    # the integration owner's canonical job content.
    return json.loads(json.dumps(row, sort_keys=True, allow_nan=False))


def resolve_root_timing(*, target_repo: str, target_revision: str, gpu: str,
                        form: str, schedule: str,
                        path: Optional[Path] = None) -> Dict[str, Any]:
    """Exact root timing lookup; every dimension is admission-critical."""
    raw = _load_engine_config(path)
    for row in raw.get("root_timing_profiles") or []:
        if (
            row.get("target_repo") == target_repo
            and row.get("target_revision") == target_revision
            and row.get("gpu") == gpu
            and row.get("form") == form
            and row.get("schedule") == schedule
        ):
            hours = row.get("conservative_upper_hours")
            if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0:
                break
            admission = row.get("resource_admission")
            if (
                not isinstance(admission, dict)
                or admission.get("required") is not True
                or admission.get("mode") != "controller_explicit_safe_resources"
            ):
                break
            if not isinstance(row.get("evidence"), dict) or not row["evidence"]:
                break
            identity = row.get("model_identity")
            if not isinstance(identity, dict):
                break
            if (
                not isinstance(identity.get("model_bytes"), int)
                or isinstance(identity.get("model_bytes"), bool)
                or identity["model_bytes"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(identity.get("config_sha256", ""))) is None
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(identity.get("index_sha256", ""))) is None
            ):
                break
            return json.loads(json.dumps(row, sort_keys=True, allow_nan=False))
    raise RootTimingUnavailable(
        "REFUSED before spend [root_timing_evidence_absent]: no exact timing "
        "for target %s@%s on %s, form=%s, schedule=%s"
        % (target_repo, target_revision, gpu, form, schedule)
    )


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
        if value is None or (isinstance(value, str) and value == ""):
            continue
        if isinstance(value, (list, tuple)):
            # A REPEATED flag: argparse `action="append"`.  stream_score's
            # --gguf-file is the first -- a llama.cpp build is n files and the
            # engine wants every one of them, because the container's tensor
            # table is per-part and a missing part is a missing layer rather
            # than a short read.  Joining them into one comma-separated value
            # would reach argparse as a single path that does not exist, an
            # hour into a rental.
            if not value:
                continue
            for item in value:
                argv.extend([flag, str(item)])
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
                          "PIPE_REPO per engines/stage_campaign.sh + apply patches-v2)"})
    elif _can_import(python, "quant_pipeline") is None:
        problems.append({
            "missing": "quant_pipeline (the engine's reader package)",
            "remedy": "clone PIPE_REPO per engines/stage_campaign.sh + apply patches-v2, "
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
