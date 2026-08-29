#!/usr/bin/env python3
"""registry-view -- browse the quant-fidelity registry from the CLI.

    bin/registry-view check  <hf-url-or-repo>[@rev]  [--path SUB]
    bin/registry-view rows   [--model X] [--artifact X] [--panel ID] [--lane L]
                             [--measured-by WHO] [--metric NAME] [--codec FAM]
                             [--bpw N] [--class strict|advisory]
    bin/registry-view lineage <repo> [--base OVERRIDE] [--lane streaming]

Data sources (--registry auto|hf|local[:PATH]):
  * `check` prefers the PUBLIC HF dataset (published truth), falling back to
    the local clone with a disclosure;
  * `rows`/`lineage` prefer the local clone (offline-friendly), else the
    mirror.  The footer always names the snapshot that answered.

Rendering never merges comparability groups: one table per recomputed
comparability key, lane sub-tables inside, sorting only within a lane.
A filter may HIDE groups but never merge them.

Exit codes: 0 rows found and printed; 1 no rows matched; 3 refusal
(ambiguity, lineage failure); 4 no data source available.

Stock python3.9 stdlib; no installs, no tokens, nothing written anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import Console                       # noqa: E402
from fidelity import registry_client as RC                # noqa: E402
from fidelity import lineage as LIN                       # noqa: E402

EXIT_OK, EXIT_NO_ROWS, EXIT_REFUSED, EXIT_NO_SOURCE = 0, 1, 3, 4


def _groups_to_json(reg: RC.RegistrySnapshot, rows: List[dict]) -> Dict[str, Any]:
    groups = RC.group_rows(reg, rows)
    out = []
    for key in sorted(groups):
        g = groups[key]
        out.append({
            "key": key,
            "label": g["label"],
            "lanes": {
                lane or "none": [
                    {"id": m.get("id"),
                     "value": (m.get("metric") or {}).get("value"),
                     "units": (m.get("metric") or {}).get("units"),
                     "artifact_ref": m.get("artifact_ref"),
                     "panel_ref": m.get("panel_ref"),
                     "class": (m.get("comparability") or {}).get("class"),
                     "measured_by": (m.get("provenance") or {}).get("measured_by")}
                    for m in ms
                ]
                for lane, ms in g["lanes"].items()
            },
        })
    return {"snapshot": reg.snapshot_id, "origin": reg.origin, "groups": out}


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace, con: Console) -> int:
    try:
        target = RC.parse_hf_target(args.target)
    except ValueError as exc:
        con.err(str(exc))
        return EXIT_REFUSED
    try:
        reg = RC.load(args.registry, purpose="check", con=con)
    except RC.RegistryUnavailable as exc:
        con.err(str(exc))
        return EXIT_NO_SOURCE
    repo = target["repo"]
    revision = target["revision"]
    resolved: Optional[str] = revision
    if resolved is not None and not RC.SHA40.match(resolved):
        try:
            from fidelity.hfmeta import resolve_commit
            resolved = resolve_commit(repo, resolved)
        except Exception as exc:                          # noqa: BLE001
            con.warn("cannot resolve revision %r via HF (%s); tier will be "
                     "PINNED-UNVERIFIED where a pin exists" % (revision, exc))
            resolved = None
    if resolved is None and revision is None:
        # default: the live head -- what a user downloading today would get.
        try:
            from fidelity.hfmeta import model_lineage_meta
            meta = model_lineage_meta(repo)
            resolved = meta.sha
            if resolved:
                con.say("target revision defaulted to the LIVE main commit "
                        "%s (what a download today fetches)" % resolved[:12])
            if meta.repo_id and meta.repo_id.lower() != repo.lower():
                # HF 307-redirects renamed repos; the registry records the
                # canonical name, so matching the typed alias would be a
                # false "never measured".
                con.say("HF redirects %s to %s; matching the registry under "
                        "the canonical name" % (repo, meta.repo_id))
                repo = meta.repo_id
        except Exception as exc:                          # noqa: BLE001
            from fidelity.hfmeta import hf_unavailable_text
            con.warn(hf_unavailable_text(repo, exc))
    match = RC.match_artifacts(reg, repo, resolved, args.path or target["path"])
    rows = RC.render_check(reg, repo, resolved, match, con)
    if args.json:
        doc = _groups_to_json(reg, rows)
        doc["tiers"] = [{"artifact": a["id"], "tier": t, "note": n}
                        for a, t, n in match["candidates"]]
        doc["ambiguous_paths"] = match["paths"] if match["ambiguous"] else []
        print(json.dumps(doc, indent=2, sort_keys=True))
    return EXIT_OK if rows else EXIT_NO_ROWS


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------


def _filter_rows(reg: RC.RegistrySnapshot, args: argparse.Namespace) -> List[dict]:
    artifacts = reg.collections.get("artifacts", {})
    models = reg.collections.get("models", {})
    out = []
    for m in reg.collections.get("measurements", {}).values():
        if m.get("status") != "published":
            continue
        if args.model:
            mid = m.get("model_ref") or ""
            fam = (models.get(mid) or {}).get("family") or ""
            if args.model.lower() not in mid.lower() and \
                    args.model.lower() != fam.lower():
                continue
        art = artifacts.get(m.get("artifact_ref")) or {}
        if args.artifact:
            needle = args.artifact.lower()
            hay = " ".join(str(x) for x in (
                m.get("artifact_ref"),
                (art.get("huggingface") or {}).get("repository"),
                (art.get("producer") or {}).get("handle"))).lower()
            if needle not in hay:
                continue
        if args.panel and args.panel.lower() not in (m.get("panel_ref") or "").lower():
            continue
        if args.lane:
            lane = reg.lane_of(m)
            if args.lane == "none":
                if lane is not None:
                    continue
            elif lane != args.lane:
                continue
        if args.measured_by:
            prov = m.get("provenance") or {}
            handle = (prov.get("measurer") or {}).get("handle") or ""
            if args.measured_by not in (prov.get("measured_by"), handle):
                continue
        if args.metric and (m.get("metric") or {}).get("name") != args.metric:
            continue
        if args.codec and (art.get("codec") or {}).get("family") != args.codec:
            continue
        if args.bpw is not None:
            nominal = (art.get("codec") or {}).get("bits_per_weight_nominal")
            if nominal is None or float(nominal) != float(args.bpw):
                continue
        if getattr(args, "cls", None) and \
                (m.get("comparability") or {}).get("class") != args.cls:
            continue
        out.append(m)
    return out


def cmd_rows(args: argparse.Namespace, con: Console) -> int:
    try:
        reg = RC.load(args.registry, purpose="rows", con=con)
    except RC.RegistryUnavailable as exc:
        con.err(str(exc))
        return EXIT_NO_SOURCE
    rows = _filter_rows(reg, args)
    if args.json:
        print(json.dumps(_groups_to_json(reg, rows), indent=2, sort_keys=True))
    else:
        if rows:
            con.say("%d row(s) matched (filters hide groups; they never merge "
                    "them)" % len(rows))
            RC.render_rows(reg, rows, con)
        else:
            con.say("no measurement rows matched the filters")
        con.say("")
        con.say(reg.footer())
    return EXIT_OK if rows else EXIT_NO_ROWS


# --------------------------------------------------------------------------
# lineage
# --------------------------------------------------------------------------


def cmd_lineage(args: argparse.Namespace, con: Console) -> int:
    try:
        reg = RC.load(args.registry, purpose="lineage", con=con)
    except RC.RegistryUnavailable as exc:
        con.err(str(exc))
        return EXIT_NO_SOURCE
    try:
        target = RC.parse_hf_target(args.repo)
        walk = LIN.resolve_base(target["repo"], base_override=args.base)
        con.say("lineage walk (%s):" % walk["status"])
        for i, hop in enumerate(walk["chain"]):
            con.say("  %s %s" % ("->" if i else "  ", hop))
        for hop in walk["hops"]:
            if hop.get("error"):
                con.warn(hop["error"])
        mapped = LIN.map_to_registry_model(walk["chain"], reg)
        con.say("registry model: %s (via %s record %s, hop %s)"
                % (mapped["model_ref"], mapped["via"], mapped["matched"],
                   mapped["hop"]))
        pick = LIN.pick_panel_and_teacher(mapped["model_ref"], args.lane, reg)
    except LIN.LineageError as exc:
        con.say("")
        con.say("REFUSE: %s" % exc.reason)
        for line in exc.advice:
            con.say("        %s" % line)
        return EXIT_REFUSED
    ref = reg.collections.get("references", {}).get(pick["reference_ref"]) or {}
    key_inputs = {
        "panel_id": pick["panel_ref"], "reference_id": pick["reference_ref"],
        "metric_name": "mean_of_run_means_tokenwise_kld",
        "direction": "reference_to_candidate",
        "accumulation_dtype": "float64", "stack_relation": "same_stack",
        "head_policy": "native_head",
    }
    con.say("suggested panel:   %s" % pick["panel_ref"])
    con.say("suggested teacher: %s (%s)" % (pick["reference_ref"],
                                            pick["reference_kind"]))
    con.say("  %d prior row(s) used this pair; a new row would join their "
            "comparability group" % pick["rows"])
    # Floor hint.  The reference's self_consistency names ONE floor row, but
    # floors are lane-specific -- steering a streaming measurer toward the
    # cross-stack floor is exactly the forbidden mix (adversarial review,
    # 2026-08-28).  With a --lane intent, prefer the floor row whose PIPELINE
    # declares that lane; always print the lane beside whatever is shown.
    floor_ref = (ref.get("self_consistency") or {}).get("floor_measurement_ref")
    lane_floor = None
    if args.lane:
        for m in reg.collections.get("measurements", {}).values():
            if m.get("status") != "published":
                continue
            if m.get("reference_ref") != pick["reference_ref"]:
                continue
            if reg.lane_of(m) != args.lane:
                continue
            detail = str(((m.get("comparability") or {}).get("bias") or {})
                         .get("detail", ""))
            if "THIS ROW IS THE FLOOR" in detail or "floor" in str(m.get("id", "")):
                lane_floor = m
                break
    if lane_floor is not None:
        con.say("  floor for lane %r: %s" % (args.lane, lane_floor["id"]))
        if floor_ref and floor_ref != lane_floor["id"]:
            con.say("  (the reference's self_consistency names %s, lane %s -- "
                    "a DIFFERENT lane's floor; never subtract it from a %s-"
                    "lane mean)" % (floor_ref,
                                    reg.lane_of(reg.collections.get(
                                        "measurements", {}).get(floor_ref)
                                        or {}) or "none declared",
                                    args.lane))
    elif floor_ref:
        floor_row = reg.collections.get("measurements", {}).get(floor_ref) or {}
        floor_lane = reg.lane_of(floor_row) or "none declared"
        con.say("  reference names its own floor measurement: %s (lane: %s)"
                % (floor_ref, floor_lane))
        if args.lane and floor_lane != args.lane:
            con.warn("that floor's lane (%s) does not match your --lane %s -- "
                     "floors are (panel, teacher, lane)-specific; do not "
                     "subtract it from a %s-lane mean"
                     % (floor_lane, args.lane, args.lane))
    for alt in pick["alternatives"]:
        con.say("  alternative: --panel %s --teacher %s (%d prior rows)"
                % (alt["panel_ref"], alt["reference_ref"], alt["rows"]))
    _ = key_inputs   # informational; the measuring tools recompute for real
    con.say("")
    con.say(reg.footer())
    if args.json:
        print(json.dumps({"chain": walk["chain"], "model_ref": mapped["model_ref"],
                          "panel_ref": pick["panel_ref"],
                          "reference_ref": pick["reference_ref"],
                          "alternatives": pick["alternatives"]},
                         indent=2, sort_keys=True))
    return EXIT_OK


# --------------------------------------------------------------------------
# live selftest (T8)
# --------------------------------------------------------------------------

# Published values that must NEVER change (tripwire; a published value moving
# is a registry integrity failure, not a drift to accommodate).
TRIPWIRE = {
    "measurement--glm53.k6-6bpw-stream.brandonmusic-final25": 0.013714888822596553,
    "measurement--glm53.k8-8bpw-stream.brandonmusic-final25": 0.012384191023436866,
}


def selftest_live(con: Console) -> int:
    con.say("registry-view --selftest-live: fetching the PUBLIC dataset "
            "unauthenticated")
    try:
        reg = RC.load_hf()
    except RC.RegistryUnavailable as exc:
        con.err(str(exc))
        return EXIT_NO_SOURCE
    ok = True
    n = len(reg.collections.get("measurements", {}))
    con.say("  snapshot %s  (%s)" % (reg.snapshot_id, reg.origin))
    for name in RC.COLLECTION_NAMES:
        con.say("  %-13s %d rows" % (name, len(reg.collections.get(name, {}))))
    if n < 59:
        con.err("measurement count %d < 59 -- the mirror shrank" % n)
        ok = False
    mismatched = 0
    for m in reg.collections["measurements"].values():
        stored = (m.get("comparability") or {}).get("key")
        try:
            recomputed = reg.recomputed_key(m)
        except Exception as exc:                          # noqa: BLE001
            con.err("cannot recompute key for %s: %s" % (m.get("id"), exc))
            ok = False
            continue
        if stored != recomputed:
            con.err("cmp key mismatch on %s: stored %s recomputed %s"
                    % (m.get("id"), stored, recomputed))
            mismatched += 1
            ok = False
    con.say("  recomputed comparability keys: %d/%d match stored" % (n - mismatched, n))
    for mid, expected in TRIPWIRE.items():
        row = reg.collections["measurements"].get(mid)
        if row is None:
            con.say("  TRIPWIRE %s: not yet in the mirror (local-only rows are "
                    "expected mid-push); tolerated" % mid)
            continue
        got = (row.get("metric") or {}).get("value")
        if got == expected:
            con.say("  TRIPWIRE %s == %.18g  ok" % (mid, expected))
        else:
            con.err("TRIPWIRE FAILED: %s is %r, published value was %.18g -- "
                    "published values must never change" % (mid, got, expected))
            ok = False
    for note in reg.notes:
        con.warn(note)
    con.say("selftest-live: %s" % ("PASS" if ok else "FAIL"))
    return EXIT_OK if ok else 1


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="registry-view", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest-live", action="store_true", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="is this artifact already measured?")
    c.add_argument("target", help="HF url or org/name[@rev][/subpath]")
    c.add_argument("--path", help="subpath for multi-artifact repos (e.g. 4-bit/)")
    c.add_argument("--registry", default="auto")
    c.add_argument("--json", action="store_true")

    r = sub.add_parser("rows", help="filter and render measurement rows")
    r.add_argument("--model")
    r.add_argument("--artifact")
    r.add_argument("--panel")
    r.add_argument("--lane", help="streaming | <name> | none")
    r.add_argument("--measured-by", help="self-measured | author-reported | <handle>")
    r.add_argument("--metric")
    r.add_argument("--codec")
    r.add_argument("--bpw", type=float)
    r.add_argument("--class", dest="cls", choices=("strict", "advisory"))
    r.add_argument("--registry", default="auto")
    r.add_argument("--json", action="store_true")

    ln = sub.add_parser("lineage", help="resolve a repo's base model and the "
                                        "panel/teacher precedent")
    ln.add_argument("repo")
    ln.add_argument("--base", help="override the lineage walk with this base repo")
    ln.add_argument("--lane", help="intended lane (streaming biases the teacher pick)")
    ln.add_argument("--registry", default="auto")
    ln.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()
    if args.selftest_live:
        return selftest_live(con)
    if args.cmd == "check":
        return cmd_check(args, con)
    if args.cmd == "rows":
        return cmd_rows(args, con)
    if args.cmd == "lineage":
        return cmd_lineage(args, con)
    build_parser().print_help()
    return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
