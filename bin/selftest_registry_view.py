#!/usr/bin/env python3
"""Offline known-answer tests for the registry client, matcher and renderer.

    python3 bin/selftest_registry_view.py

Stock python3.9, stdlib, no network, no GPU.  Assertions are INVARIANTS, not
exact counts, because another agent may be adding rows to registry/ while this
runs: recomputed==stored for every row, counts >= the published floor of 59.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.common import Console                       # noqa: E402
from fidelity import registry_client as RC                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def main() -> int:
    print("[1] LOCAL clone loads, and every derived key matches its stored copy")
    reg = RC.load_local()
    for name in RC.COLLECTION_NAMES:
        check("collection %s loads" % name, name in reg.collections and
              len(reg.collections[name]) > 0,
              "%d rows" % len(reg.collections.get(name, {})))
    n = len(reg.collections["measurements"])
    check("measurement count >= 59 (never asserts an exact count; the "
          "registry is concurrently edited)", n >= 59, "%d rows" % n)
    mismatches = []
    for m in reg.collections["measurements"].values():
        stored = (m.get("comparability") or {}).get("key")
        recomputed = reg.recomputed_key(m)
        if stored != recomputed:
            mismatches.append((m.get("id"), stored, recomputed))
    check("recomputed comparability key == stored key for ALL %d rows" % n,
          not mismatches, mismatches[:2])

    print("\n[2] target parsing")
    t = RC.parse_hf_target("https://huggingface.co/zai-org/GLM-5.3-Flash/tree/abc123/sub/dir")
    check("URL with /tree/rev/subpath",
          t == {"repo": "zai-org/GLM-5.3-Flash", "revision": "abc123",
                "path": "sub/dir"}, t)
    t = RC.parse_hf_target("org/name@deadbeef")
    check("org/name@rev", t["repo"] == "org/name" and t["revision"] == "deadbeef")
    t = RC.parse_hf_target("orcarouter/GLM-5.3-Flash-MLX/4-bit")
    check("bare subpath becomes the path hint", t["path"] == "4-bit")
    t = RC.parse_hf_target("org/name/tree/main")
    check("'main' normalizes to None (the live head is the default)",
          t["revision"] is None)
    try:
        RC.parse_hf_target("justonename")
        check("one-segment input refused", False)
    except ValueError:
        check("one-segment input refused", True)

    print("\n[3] tier logic on synthetic artifacts (deterministic fixtures)")
    fake = RC.RegistrySnapshot({
        "artifacts": {
            "artifact--x.pinned": {"id": "artifact--x.pinned", "huggingface": {
                "repository": "Org/Repo", "revision": "a" * 40,
                "revision_source": "hf_api", "path": None}},
            "artifact--x.unpinned": {"id": "artifact--x.unpinned", "huggingface": {
                "repository": "org/unpinned", "revision": None,
                "revision_source": "none", "path": None},
                "disclosures": [{"code": "revision_unpinned",
                                 "detail": "identity rests on content hashes"}]},
            "artifact--x.mlx4": {"id": "artifact--x.mlx4", "huggingface": {
                "repository": "org/mlx", "revision": "b" * 40,
                "revision_source": "hf_api", "path": "4-bit/"}},
            "artifact--x.mlx6": {"id": "artifact--x.mlx6", "huggingface": {
                "repository": "org/mlx", "revision": "b" * 40,
                "revision_source": "hf_api", "path": "6-bit/"}},
        },
        "models": {}, "panels": {}, "references": {}, "pipelines": {},
        "measurements": {},
    }, "synthetic", "in-memory fixture")

    m = RC.match_artifacts(fake, "org/repo", "a" * 40)      # case-insensitive
    check("EXACT tier at the pinned revision (case-insensitive repo match)",
          [t for _, t, _ in m["candidates"]] == [RC.TIER_EXACT])
    m = RC.match_artifacts(fake, "org/repo", "f" * 40)
    check("STALE tier at a different revision",
          [t for _, t, _ in m["candidates"]] == [RC.TIER_STALE])
    note = m["candidates"][0][2]
    check("STALE note quotes both revisions",
          "aaaaaaaaaa" in note and "ffffffffff" in note, note)
    # A repo that publishes several artifacts on several branches
    # (turboderp/GLM-5.3-Flash-exl3 ships 4.05/3.05/2.05bpw that way) hits the
    # STALE path when you ask about a branch other than the measured one -- and
    # "revision drift" is the wrong story for it: nothing drifted, you named a
    # different artifact. The registry knows the scope (huggingface.path); the
    # refusal has to say it, or the reader concludes it is already measured.
    hint = RC.stale_scope_hint(
        {"candidates": [({"huggingface": {"path": "branch 4.05bpw"}},
                         RC.TIER_STALE, "")]})
    check("a branch-scoped STALE row says WHICH scope was measured",
          any("branch 4.05bpw" in line for line in hint)
          and any("not drift" in line for line in hint), repr(hint))
    check("an unscoped STALE row adds no scope note",
          RC.stale_scope_hint(
              {"candidates": [({"huggingface": {}}, RC.TIER_STALE, "")]}) == [])
    check("a non-STALE candidate never contributes a scope note",
          RC.stale_scope_hint(
              {"candidates": [({"huggingface": {"path": "branch x"}},
                               RC.TIER_EXACT, "")]}) == [])
    m = RC.match_artifacts(fake, "org/unpinned", "f" * 40)
    check("UNPINNED tier when the record has revision null",
          [t for _, t, _ in m["candidates"]] == [RC.TIER_UNPINNED])
    check("UNPINNED note surfaces the revision_unpinned disclosure",
          "content hashes" in m["candidates"][0][2])
    m = RC.match_artifacts(fake, "org/repo", None)
    check("PINNED-UNVERIFIED when the target revision is unknowable",
          [t for _, t, _ in m["candidates"]] == [RC.TIER_UNVERIFIED])
    m = RC.match_artifacts(fake, "org/mlx", "b" * 40)
    check("multi-path repo with no --path is AMBIGUOUS, candidates listed",
          m["ambiguous"] and len(m["candidates"]) == 2 and
          m["paths"] == ["4-bit", "6-bit"])
    m = RC.match_artifacts(fake, "org/mlx", "b" * 40, path_hint="4-bit/")
    check("--path 4-bit/ disambiguates to one candidate",
          not m["ambiguous"] and len(m["candidates"]) == 1)

    print("\n[4] AMBIGUOUS on the real MLX rows (5 artifacts, one repo)")
    mlx = RC.match_artifacts(reg, "orcarouter/GLM-5.3-Flash-MLX", None)
    check("real MLX repo matches >= 5 artifacts and is ambiguous without --path",
          len(mlx["candidates"]) >= 5 and mlx["ambiguous"],
          "%d candidates, paths %s" % (len(mlx["candidates"]), mlx["paths"]))

    print("\n[5] renderer NEVER merges comparability groups")
    rows = list(reg.collections["measurements"].values())
    by_key = {}
    for m in rows:
        by_key.setdefault(reg.recomputed_key(m), []).append(m)
    keys = sorted(by_key, key=lambda k: -len(by_key[k]))[:2]
    check("the registry has >= 2 comparability groups to test with",
          len(keys) == 2, "%d groups total" % len(by_key))
    mixed = by_key[keys[0]] + by_key[keys[1]]
    groups = RC.group_rows(reg, mixed)
    check("two keys in -> two group tables out (structural, not stylistic)",
          set(groups) == set(keys))
    import registry_view as RV
    doc = RV._groups_to_json(reg, mixed)
    check("--json output carries exactly the two groups", len(doc["groups"]) == 2)
    ids_in = {m["id"] for m in mixed}
    ids_out = {r["id"] for g in doc["groups"]
               for ms in g["lanes"].values() for r in ms}
    check("no row lost or invented by the renderer", ids_in == ids_out)

    print("\n[6] lane join reproduces registry_render's rule (None != sealed)")
    lanes = {reg.lane_of(m) for m in rows}
    check("both None-lane and named-lane rows exist", None in lanes and
          any(x for x in lanes if x), sorted(x for x in lanes if x))
    streaming = [m for m in rows if reg.lane_of(m) == "streaming"]
    check("streaming lane rows found via the pipeline join (>= 2)",
          len(streaming) >= 2, "%d" % len(streaming))

    print("\n" + "-" * 72)
    print("selftest_registry_view: %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
