#!/usr/bin/env python3
"""What the reseed moved, and -- as importantly -- what it did not.

`git diff` on a JSONL file answers "these lines changed".  The question an
operator actually has before approving a change to published numbers is "did any
HEADLINE move", and a line diff cannot answer it: every one of these rows changes
on a reseed because the note text changes.

So this diffs the OLD data against the NEW one FIELD BY FIELD, and separates:

  * headline `metric.value`                       -- must not move
  * top-level `uncertainty` (SE, deff, endpoints) -- must not move
  * per-domain `mean` and `se_clustered`          -- must not move
  * per-domain CI endpoints                       -- the change being made

A "no headline moved" claim that is not a recomputation is an assertion. This is
the recomputation.

Usage:
    git show HEAD:registry/data/measurements.jsonl > /tmp/old.jsonl
    python3 tools/reseed_delta.py /tmp/old.jsonl data/measurements.jsonl
"""

import argparse
import json
import sys

# Every field of `uncertainty` that is a NUMBER a reader could quote. `note` is
# prose and is expected to change; listing the numbers explicitly rather than
# diffing the whole block is what makes "no top-level uncertainty figure moved" a
# checkable claim instead of a vague one.
UNC_NUMERIC = ("ci95_low", "ci95_high", "se_clustered", "se_naive", "deff",
               "sigma_run", "sigma_run_runs", "se_total", "clusters", "samples",
               "bootstrap_b", "bootstrap_seed")
DOMAIN_INVARIANT = ("mean", "se_clustered", "scored_positions", "windows")
DOMAIN_MOVED = ("ci95_low", "ci95_high")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}


def rel(a, b):
    if a == b:
        return 0.0
    if a in (None, 0) or b is None:
        return float("inf")
    return abs(b - a) / abs(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)

    report = {"rows_old": len(old), "rows_new": len(new),
              "rows_added": sorted(set(new) - set(old)),
              "rows_removed": sorted(set(old) - set(new)),
              "headline_changed": [], "uncertainty_changed": [],
              "domain_invariant_changed": [], "domain_ci_changed": []}

    for mid in sorted(set(old) & set(new)):
        o, n = old[mid], new[mid]
        if o["metric"]["value"] != n["metric"]["value"]:
            report["headline_changed"].append(
                {"id": mid, "old": o["metric"]["value"], "new": n["metric"]["value"]})
        uo, un = o.get("uncertainty") or {}, n.get("uncertainty") or {}
        for k in UNC_NUMERIC:
            if uo.get(k) != un.get(k):
                report["uncertainty_changed"].append(
                    {"id": mid, "field": k, "old": uo.get(k), "new": un.get(k)})
        bo = {c["domain"]: c for c in (o.get("by_domain") or [])}
        bn = {c["domain"]: c for c in (n.get("by_domain") or [])}
        if set(bo) != set(bn):
            report["domain_invariant_changed"].append(
                {"id": mid, "field": "domain set", "old": sorted(bo), "new": sorted(bn)})
        for dom in sorted(set(bo) & set(bn)):
            for k in DOMAIN_INVARIANT:
                if bo[dom].get(k) != bn[dom].get(k):
                    report["domain_invariant_changed"].append(
                        {"id": mid, "domain": dom, "field": k,
                         "old": bo[dom].get(k), "new": bn[dom].get(k)})
            for k in DOMAIN_MOVED:
                a, b = bo[dom].get(k), bn[dom].get(k)
                if a != b:
                    report["domain_ci_changed"].append(
                        {"id": mid, "domain": dom, "endpoint": k, "old": a, "new": b,
                         "rel": rel(a, b)})

    moved = report["domain_ci_changed"]
    report["summary"] = {
        "headline metric.value changed": len(report["headline_changed"]),
        "top-level uncertainty numbers changed": len(report["uncertainty_changed"]),
        "by_domain mean / se_clustered / positions changed":
            len(report["domain_invariant_changed"]),
        "by_domain CI endpoints changed": len(moved),
        "worst relative move": max([m["rel"] for m in moved], default=0.0),
    }

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        for k, v in report["summary"].items():
            print("%-52s: %s" % (k, ("%.4f%%" % (100 * v)) if k == "worst relative move" else v))
        for name in ("headline_changed", "uncertainty_changed", "domain_invariant_changed"):
            for item in report[name]:
                print("  MOVED-THAT-MUST-NOT %s %r" % (name, item))
        if moved:
            print("\n%-56s %-28s %-10s %-22s %-22s %9s"
                  % ("row", "domain", "endpoint", "old", "new", "rel"))
            for m in sorted(moved, key=lambda x: -x["rel"]):
                print("%-56s %-28s %-10s %-22.15g %-22.15g %8.4f%%"
                      % (m["id"].replace("measurement--", ""), m["domain"],
                         m["endpoint"].replace("ci95_", ""), m["old"], m["new"],
                         100 * m["rel"]))
    bad = (report["headline_changed"] + report["uncertainty_changed"]
           + report["domain_invariant_changed"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
