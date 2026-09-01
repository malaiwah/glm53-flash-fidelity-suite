#!/usr/bin/env python3
"""measure -- one command: measure KLD on a quant given only an HF link.

    bin/measure zai-org/GLM-5.3-Flash-BF16
    bin/measure https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX/tree/main/4-bit
    bin/measure malaiwah/GLM-5.3-Flash-TR3-6bpw --plan-only

What it resolves FOR you, printing one status line per step:
  1. the target (URL or org/name, @rev or /tree/rev, subpath -> --path hint)
  2. the registry (public HF dataset first; local clone fallback)
  3. the live revision (what a download today would fetch)
  4. ALREADY MEASURED? -> prints the rows + receipt links and exits 0
     (--force to measure anyway; revision drift needs --force or
     --accept-measured-revision)
  5. the model, by walking base_model lineage to a root the registry knows
  6. the panel + teacher prior measurements of that model used (overridable
     with --panel/--teacher; alternatives are printed)
  7. the artifact's surface -- a repo no lane can read is refused HERE, for
     $0.00, with the remedy named
  8. the lane for THIS machine (local-mps on Apple Silicon; --lane overrides)
  9. hands off to measure-local WITH --execute (use --plan-only to stop at
     the plan; preflight still verifies torch/transformers/quant_pipeline/
     teacher/disk first and refuses with remedies)

Exit codes: 0 = already-measured report OR a completed measurement/preview;
3 = refusal (every refusal states its arithmetic or remedy); 4 = no data
source. Never a stack trace.

This is a THIN front-end: all real logic lives in fidelity/* and
measure_local.py, so measure-local/measure-cloud users get identical
behaviour from the same code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import Console                       # noqa: E402
from fidelity.engines import load_engines                 # noqa: E402
from fidelity import registry_client as RC                # noqa: E402
from fidelity import lineage as LIN                       # noqa: E402
from fidelity.hfmeta import (                             # noqa: E402
    DEFAULT_PANEL, HFError, hf_unavailable_text, model_lineage_meta,
    repo_meta, sniff_surface,
)

EXIT_OK, EXIT_REFUSED, EXIT_NO_SOURCE = 0, 3, 4


class Refusal(RuntimeError):
    def __init__(self, reason: str, advice: Optional[List[str]] = None) -> None:
        self.reason, self.advice = reason, list(advice or [])
        super().__init__(reason)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="measure", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="HF url or org/name[@rev][/subpath]")
    p.add_argument("--path", help="subpath for multi-artifact repos (e.g. 4-bit/)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--accept-measured-revision", action="store_true")
    p.add_argument("--base", help="override the lineage walk with this base repo")
    p.add_argument("--panel", help="override the panel pick (registry panel id)")
    p.add_argument("--teacher", help="override the teacher pick (reference id)")
    p.add_argument("--lane", default="auto",
                   choices=("auto", "local-mps", "local-cuda-budget", "streaming"))
    p.add_argument("--registry", default="auto")
    p.add_argument("--budget", type=float, metavar="GB",
                   help="forwarded as measure-local --vram-budget")
    p.add_argument("--out", help="receipt output directory")
    p.add_argument("--plan-only", action="store_true",
                   help="stop after the plan; execute nothing")
    p.add_argument("--json", action="store_true")
    return p


def run(args: argparse.Namespace, con: Console) -> int:
    # 1. target ------------------------------------------------------------
    try:
        target = RC.parse_hf_target(args.target)
    except ValueError as exc:
        raise Refusal(str(exc), ["examples: org/name, org/name@<sha>, "
                                 "https://huggingface.co/org/name/tree/main"])
    path_hint = args.path or target["path"]
    con.say("[1/9] target: %s%s%s" % (
        target["repo"],
        ("@" + target["revision"][:12]) if target["revision"] else " (live head)",
        (" path=" + path_hint) if path_hint else ""))

    # 2. registry ----------------------------------------------------------
    try:
        reg = RC.load(args.registry, purpose="check", con=con)
    except RC.RegistryUnavailable as exc:
        con.err(str(exc))
        return EXIT_NO_SOURCE
    con.say("[2/9] registry: %s" % reg.footer())

    # 3. live revision -----------------------------------------------------
    resolved = target["revision"]
    hf_ok = True
    try:
        if resolved is None:
            resolved = model_lineage_meta(target["repo"]).sha
        elif not RC.SHA40.match(resolved):
            from fidelity.hfmeta import resolve_commit
            resolved = resolve_commit(target["repo"], resolved)
        con.say("[3/9] revision: %s" % (resolved or "?")[:12])
    except HFError as exc:
        hf_ok = False
        con.warn("[3/9] " + hf_unavailable_text(target["repo"], exc))

    # 4. already measured? -------------------------------------------------
    # Tier only against a RESOLVED 40-hex sha: an unresolvable branch string
    # must not tier a pinned artifact as STALE ("measured at a different
    # commit") when the truth is "your revision could not be verified".
    match_rev = resolved if (resolved and RC.SHA40.match(resolved)) else None
    match = RC.match_artifacts(reg, target["repo"], match_rev, path_hint)
    rows = RC.render_check(reg, target["repo"], match_rev, match, con)
    tiers = {tier for _, tier, _ in match["candidates"]}
    if rows:
        if tiers & {RC.TIER_EXACT, RC.TIER_UNPINNED, RC.TIER_UNVERIFIED}:
            if not args.force:
                con.say("")
                con.say("[4/9] ALREADY MEASURED -- the rows above answer this "
                        "request; nothing else runs. --force to measure anyway.")
                return EXIT_OK
            con.warn("[4/9] --force: measuring despite the rows above")
        elif tiers == {RC.TIER_STALE}:
            if not (args.force or args.accept_measured_revision):
                raise Refusal(
                    "revision drift: this repo was measured at a different "
                    "commit (rows above).",
                    ["--accept-measured-revision targets the measured commit",
                     "--force measures the new commit as a NEW artifact record"]
                    + [line.strip() for line in RC.stale_scope_hint(match)
                       if line.strip()])
            if args.accept_measured_revision and not args.force:
                pinned = [(a.get("huggingface") or {}).get("revision")
                          for a, t, _ in match["candidates"] if t == RC.TIER_STALE]
                if pinned and pinned[0]:
                    resolved = pinned[0]
                    con.warn("[4/9] targeting the measured revision %s"
                             % resolved[:12])
                    # Retargeting makes the request "the commit that WAS
                    # measured" -- re-run the tier match against that sha and,
                    # when it answers (it is EXACT by construction), report
                    # the rows and stop.  Measuring anyway stays behind
                    # --force; this path must never dead-end in an --execute
                    # preflight refusal (usability review, 2026-08-28).
                    rematch = RC.match_artifacts(reg, target["repo"],
                                                 resolved, path_hint)
                    retiers = {t for _, t, _ in rematch["candidates"]}
                    if retiers & {RC.TIER_EXACT, RC.TIER_UNPINNED,
                                  RC.TIER_UNVERIFIED}:
                        con.say("")
                        con.say("[4/9] at the accepted (measured) revision "
                                "these rows tier %s:"
                                % "/".join(sorted(retiers)))
                        RC.render_check(reg, target["repo"], resolved,
                                        rematch, con)
                        con.say("")
                        con.say("[4/9] ALREADY MEASURED -- the rows above "
                                "answer this request at the measured "
                                "revision; nothing else runs. --force to "
                                "measure anyway.")
                        return EXIT_OK
                    con.warn("[4/9] the accepted revision still does not "
                             "tier EXACT (unexpected); proceeding to "
                             "measure it")
    else:
        con.say("[4/9] not yet measured -- proceeding")
    if match["ambiguous"] and not path_hint:
        raise Refusal(
            "this repo holds %d artifacts distinguished only by path (%s); "
            "measuring needs exactly one" % (len(match["candidates"]),
                                             ", ".join(match["paths"])),
            ["pass --path %s/ (for example)" % match["paths"][0]])

    # 5. lineage -----------------------------------------------------------
    try:
        walk = LIN.resolve_base(target["repo"], base_override=args.base)
        mapped = LIN.map_to_registry_model(walk["chain"], reg)
    except LIN.LineageError as exc:
        raise Refusal(exc.reason, exc.advice)
    con.say("[5/9] lineage: %s -> model %s"
            % (" -> ".join(walk["chain"]), mapped["model_ref"]))

    # 6. panel + teacher ---------------------------------------------------
    lane_intent = None if args.lane == "auto" else args.lane
    try:
        pick = LIN.pick_panel_and_teacher(mapped["model_ref"], lane_intent, reg)
    except LIN.LineageError as exc:
        raise Refusal(exc.reason, exc.advice)
    panel_ref = args.panel or pick["panel_ref"]
    reference_ref = args.teacher or pick["reference_ref"]
    con.say("[6/9] panel %s + teacher %s (%d prior rows join this "
            "comparability group)" % (panel_ref, reference_ref, pick["rows"]))
    for alt in pick["alternatives"]:
        con.say("      alternative: --panel %s --teacher %s (%d rows)"
                % (alt["panel_ref"], alt["reference_ref"], alt["rows"]))
    if panel_ref != DEFAULT_PANEL.panel_ref:
        raise Refusal(
            "panel %s has no local fetch descriptor in this checkout (only %s "
            "ships one)" % (panel_ref, DEFAULT_PANEL.panel_ref),
            ["pass measure-local --panel-descriptor with a JSON naming its "
             "include globs, contexts and positions -- the tool will not "
             "guess a panel's shape"])

    # 7. surface -----------------------------------------------------------
    surface_name = None
    if hf_ok:
        try:
            meta = repo_meta(target["repo"], "model", resolved or "main")
            # path_hint, not None: this is the same repo-is-a-shelf case the
            # two runners have. Without it a GGUF repo sniffs as "12 builds,
            # pick one" even when the caller already did, and step 7 prints a
            # null codec for an artifact it has in hand.
            surface = sniff_surface(meta, path_hint)
            surface_name = surface.surface
            con.say("[7/9] surface: %s (codec %s @ %s bpw)"
                    % (surface.surface, surface.codec_family, surface.bits))
            # WHICH LANE. `bin/measure` never rents (step 8 refuses
            # --lane streaming), so it always plans a LOCAL lane, and the local
            # lanes read strictly less than the streaming one. Saying "no lane
            # can read it" when the cloud lane reads it fine sends the reader
            # away from the recipe that would have worked -- which is what
            # happened to the first stranger who tried to measure a stock
            # exllamav3 release with this tool.
            readable_here = {"packed", "native-bf16"}
            # READ from engines.json, not retyped. This set was a literal and
            # went stale twice: `dione` and `gguf` readers both landed on the
            # streaming lane while this list still said they did not exist, so
            # the one tool a newcomer is told to start with sent them away from
            # a recipe that would have worked.
            readable_streaming = set(
                (load_engines().get("streaming").surfaces if
                 load_engines().get("streaming") else [])
                or ["packed", "native-bf16", "exl3hf", "tr3-published"])
            # A repo that publishes SEVERAL artifacts at one revision is not
            # unreadable; it is unchosen, and those are different verdicts with
            # different remedies. Answer the second one before the first.
            builds = surface.evidence.get("gguf_builds") or {}
            if builds and surface.path is None:
                reason = ("%s publishes %d artifacts at this revision and a "
                          "measurement describes ONE of them"
                          % (target["repo"], len(builds)))
                lines = ["pass --path <build>, for example:",
                         "  bin/measure %s --path %s"
                         % (target["repo"], sorted(builds)[0]),
                         "",
                         "builds: " + ", ".join(sorted(builds))]
                if not args.plan_only:
                    raise Refusal(reason, lines)
                con.warn(reason + " -- planning without one")
            elif surface.problems or surface.surface not in readable_here:
                elsewhere = (surface.surface in readable_streaming
                             and not surface.problems)
                lines = ["registry rows above stand; measured receipts exist "
                         "for what IS measurable" if rows else
                         "no registry rows exist for it either"]
                lines += surface.problems
                if elsewhere:
                    lines = [
                        "The streaming engine declares this surface, but the "
                        "paid controller admits only exact authored targets.",
                        "There is no generic cloud handoff for this repository; "
                        "see docs/THIRD-PARTY-QUICKSTART.md for the current "
                        "RunPod pins and prerequisites.",
                    ] + lines
                    reason = (
                        "%s publishes surface '%s'. The local lanes read only "
                        "%s; a streaming engine can decode the surface, but "
                        "bin/measure never rents and paid admission is a "
                        "separate exact-target contract."
                        % (target["repo"], surface.surface,
                           " and ".join(sorted(readable_here))))
                else:
                    reason = ("%s publishes surface '%s'; no lane can read it "
                              "(engines.json). This tool can (a) report "
                              "existing rows [%s], (b) plan (--plan-only). "
                              "Measuring third-party surfaces needs a reader "
                              "for THIS surface; the streaming lane reads %s "
                              "today."
                              % (target["repo"], surface.surface,
                                 "listed above" if rows else "none",
                                 ", ".join(sorted(readable_streaming))))
                if not args.plan_only:
                    raise Refusal(reason, lines)
                con.warn("surface '%s' has no reader on a LOCAL lane -- "
                         "planning only, an --execute would refuse%s"
                         % (surface.surface,
                            " (streaming engine capability does not imply "
                            "paid admission)" if elsewhere else ""))
        except HFError as exc:
            con.warn("[7/9] cannot sniff the surface: %s" % exc)
    else:
        con.warn("[7/9] surface unknown (HF unreachable); a real run would "
                 "re-check")

    # 8. lane --------------------------------------------------------------
    lane = args.lane
    if lane == "streaming":
        raise Refusal(
            "lane 'streaming' is cloud-engine capability; bin/measure never "
            "rents hardware and cannot authorize a generic paid handoff.",
            ["current paid execution is exact-target-only:",
             "  docs/THIRD-PARTY-QUICKSTART.md §5"])
    if lane == "auto":
        import platform
        lane = ("local-mps" if platform.machine() == "arm64" and
                platform.system() == "Darwin" else "local-cuda-budget")
    con.say("[8/9] lane: %s%s" % (lane,
            "" if args.lane != "auto" else " (auto for this machine)"))

    # 9. hand off to measure-local ----------------------------------------
    import measure_local

    argv: List[str] = [
        "--artifact", target["repo"],
        "--panel", DEFAULT_PANEL.repo_id,
        "--lane", lane,
        "--skip-registry-check",          # the gate above already answered
    ]
    if resolved:
        argv += ["--revision", resolved]
    if path_hint:
        argv += ["--path", path_hint]
    if args.budget:
        argv += ["--vram-budget", str(args.budget)]
    if args.out:
        argv += ["--out", args.out]
    if args.plan_only:
        argv += ["--estimate-only"]
    else:
        argv += ["--execute"]
    con.say("[9/9] handing off: measure-local %s" % " ".join(argv))
    con.say("")
    return measure_local.main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()
    try:
        return run(args, con)
    except Refusal as exc:
        con.say("")
        con.say("REFUSE: %s" % exc.reason)
        for line in exc.advice:
            if line:
                con.say("        %s" % line)
        return EXIT_REFUSED
    except Exception as exc:                              # noqa: BLE001
        # the contract is "never a traceback": name the failure and where the
        # detail lives, then exit non-zero
        con.err("unexpected %s: %s (this is a bug in bin/measure_one.py; "
                "re-run the underlying step directly for the full detail)"
                % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
