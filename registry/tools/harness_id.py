"""Harness identity: which code produced this number.

WHY THIS EXISTS
---------------
Every measured value in this registry is a function of some code.  Until this
module there was no field that said *which* code, so a defect found in the
estimator put every published row equally under suspicion and no row could be
cleared.  A peer review left ~130 findings open with the honest note that none
is known to move a published number and none has been individually cleared
either; without a code stamp that liability floats over every row forever.

With a stamp, "which rows predate the STAT-01 fix" is a field test instead of an
archaeology project.

THE BOUNDARY, AND WHY IT IS DRAWN HERE
--------------------------------------
The digest must answer: *if this set of bytes is identical, is the number
necessarily identical?*  Two boundaries are obviously wrong.

  * **The whole repository.**  Useless.  A typo fix in `README.md` changes the
    digest, so every row's identity churns for reasons that cannot affect any
    number, and the field stops carrying information.
  * **Only the estimator file.**  Unsafe.  `bin/jointstd/stats.py` calls
    `chi2.norm_ppf` for every BCa endpoint; a one-ULP change in `chi2.py` moves
    published endpoints while `stats.py` is byte-identical.  The stamp would say
    "same code" about two different numbers, which is worse than no stamp.

So the boundary is the **computational closure**: every file on the path from
the published inputs to the published number, enumerated by role, and nothing
else.  For the joint-standard derivation that is the estimator, its numerical
support, the protocol stamper, the enrichment layer that fixes B, the seed and
the rounding -- and the coverage simulator, because `coverage_measured` is a
published number and that is the code that produced it.  It is NOT
`seed_registry.py` (it assembles rows and changes whenever an unrelated row is
added), NOT the validator, NOT the renderer, NOT docs.

The boundary errs deliberately toward **over-sensitivity**.  `joint_enrich.py`
carries a `SERIES` table, so adding an unrelated measurement series changes the
id of rows whose numbers did not move.  That is the safe direction: a changed id
costs a reader one diff of `code_digests`, which name their roles precisely so
the diff is legible, while a missed change costs them a wrong comparison.  The
guarantee is one-way and is stated that way everywhere:

    equal harness_id  =>  identical code
    differing id      =>  look at code_digests; the number may or may not move

WHAT IS *NOT* IN THE ID, AND WHY
--------------------------------
`repository.commit` is recorded but does NOT enter `harness_id`.  A commit sha
changes on a docs edit, which is exactly the churn the boundary exists to avoid,
and a commit cannot be recorded by the change that introduces it -- so the
commit is a human pointer and the digests are the identity.  When a row is
produced by a working tree that became the commit introducing it, that is stated
as `commit_role: "parent"` with `dirty: true` rather than back-dated.

`tool_versions` DOES enter the id.  This project has already been bitten by an
interpreter difference moving a published number: CPython 3.12 switched builtin
`sum()` to Neumaier summation and the same data reduced by 3.9 and 3.12 differed
in the last ULP.  An interpreter is part of the estimator.

Dependency-free, stock python3.9, no network.  Importable by the validator, by
`seed_registry.py` and by `registry_add.py`.
"""

import hashlib
import json
import os

BOUNDARY = "estimator_closure/v1"

# The digest set for a row whose `uncertainty` and `by_domain` blocks are derived
# locally from published per-window means by the joint standard.  role -> path,
# repo-relative.  Order is irrelevant (the id sorts by role); the roles are the
# contract.
JOINT_DERIVATION_CLOSURE = (
    ("estimator", "bin/jointstd/stats.py"),
    ("estimator_numerics", "bin/jointstd/chi2.py"),
    ("protocol_stamp", "bin/jointstd/protocol.py"),
    ("enrichment", "registry/tools/joint_enrich.py"),
    # `coverage_measured` is a published number too, and this is the code that
    # produced it. Leaving it out would let the simulator change while every row
    # quoting its output kept the same identity -- which is precisely the failure
    # the block exists to prevent, and it is easy to talk yourself out of because
    # the simulator does not compute the endpoints.
    ("coverage_simulation", "registry/tools/coverage_sim.py"),
)

# The digest set for a row assembled from a measurement receipt by registry_add.
# It attests the INGEST code only: registry_add did not compute `metric.value`,
# the measuring run did, and saying otherwise in a provenance field would be the
# exact failure this module exists to prevent.
INGEST_CLOSURE = (
    ("ingest", "registry/tools/registry_add.py"),
    ("ingest_support", "registry/tools/registry_lib.py"),
)

# Field paths a harness block may claim to cover.  Closed set: a typo in `covers`
# must be an error, not a silently uncovered field.
COVERABLE = (
    "metric.value",
    "auxiliary_metrics",
    "uncertainty",
    "by_domain",
    "protocol",
    "determinism",
    "row_assembly",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digests(repo_root, closure, alias_root=None):
    """[{role, path, sha256}] for a closure, read from the tree at `repo_root`.

    Reads the BYTES.  A digest transcribed beside the file it names is a
    constant, not a provenance record -- the same lesson `_receipt_sha` in
    seed_registry.py was written for.

    `alias_root` exists because this registry lives in TWO shapes.  In the suite
    repo the tools are at `registry/tools/`; in the published dataset repo --
    the one CONTRIBUTING §1 tells a contributor to clone -- the SAME files are at
    `tools/`, with no `registry/` above them.  Resolving only against
    `repo_root` therefore raised IOError out of `registry_validate.py
    --submission`, which is the single command the contributor path documents,
    so every outside submission check died on a stack trace instead of printing
    ACCEPTED or a named failure.

    The RECORDED `path` stays the suite-relative one in both shapes, and that is
    the point: `harness_id` is a function of the code, not of where somebody
    cloned it.  Only the place the bytes are read from changes.
    """
    out = []
    for role, rel in closure:
        full = os.path.join(repo_root, rel)
        if not os.path.exists(full) and alias_root:
            # strip the leading path segment: registry/tools/x.py -> tools/x.py
            head, sep, tail = rel.partition("/")
            if sep:
                candidate = os.path.join(alias_root, tail)
                if os.path.exists(candidate):
                    full = candidate
        if not os.path.exists(full):
            raise IOError("harness closure names %s, which exists under neither "
                          "%s nor %s (as %s). This registry ships in two shapes "
                          "-- the suite repo, where the tools are at "
                          "registry/tools/, and the published dataset repo, "
                          "where they are at tools/ -- and the caller must pass "
                          "alias_root for the second."
                          % (rel, repo_root, alias_root or "(no alias root)",
                             rel.partition("/")[2] or rel))
        out.append({"role": role, "path": rel, "sha256": sha256_file(full)})
    return sorted(out, key=lambda d: d["role"])


def compute_id(code_digests, tool_versions, boundary=BOUNDARY):
    """harness_id from the digest set + the tool versions + the boundary name.

    Deliberately excludes `repository`, `covers` and `note`: see the module
    docstring.  Recomputed by the validator (HARN-003), never trusted.
    """
    payload = {
        "boundary": boundary,
        "code_digests": sorted(
            [{"role": d["role"], "path": d["path"], "sha256": d["sha256"]}
             for d in code_digests], key=lambda d: d["role"]),
        "tool_versions": {k: v for k, v in sorted((tool_versions or {}).items())
                          if v is not None},
    }
    return "harness--" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


def block(repo_root, closure, covers, tool_versions, repository,
          boundary=BOUNDARY, note=None):
    """A complete, self-consistent harness block."""
    cds = digests(repo_root, closure)
    bad = [c for c in covers if c not in COVERABLE]
    if bad:
        raise ValueError("harness covers %r, which is not in the coverable set %r"
                         % (bad, list(COVERABLE)))
    b = {
        "harness_id": compute_id(cds, tool_versions, boundary),
        "recorded": True,
        "boundary": boundary,
        "covers": sorted(covers),
        "repository": repository,
        "code_digests": cds,
        "tool_versions": dict(sorted((tool_versions or {}).items())),
    }
    if note:
        b["note"] = note
    return b


def unrecorded_block(covers, note):
    """The honest shape for a row whose producing code was never stamped.

    Never invent digests for a historical row: the files in today's checkout are
    not the files that produced it, and a plausible-looking digest set would be
    a fabricated provenance record.
    """
    return {"harness_id": None, "recorded": False, "covers": sorted(covers),
            "note": note}


def covering(row, field):
    """The harness block covering `field` on `row`, or None.

    The field test the mechanism exists for: two rows share code for `field`
    when `covering(a, f)` and `covering(b, f)` both exist and their
    `harness_id` values are equal.
    """
    h = row.get("harness")
    if not h or not h.get("recorded"):
        return None
    return h if field in (h.get("covers") or []) else None
