#!/usr/bin/env python3
"""What a release build would tag, build and push -- decided here, not in YAML.

    release_plan.py --event release --ref refs/tags/v0.3.1 --sha <40-hex>
    release_plan.py --event workflow_dispatch --sha <40-hex> --publish false

Prints a JSON plan and, with --github-output, the `name=value` lines the
workflow consumes.

WHY A SCRIPT AND NOT WORKFLOW EXPRESSIONS.  A GitHub Actions expression is
untestable anywhere except GitHub: you find out it was wrong by pushing a tag
and reading a red run. This repository's whole discipline is that a rule which
decides what gets published is exercised offline first -- so the tag rule, the
platform list and the publish gate live in a script the workflow calls, and
`bin/selftest_container.py` drives that script with known inputs and known
answers. The workflow becomes plumbing: checkout, qemu, buildx, call this,
build what it said.

THE PUBLISH GATE IS DEFAULT-OFF, ON PURPOSE.  Pushing an image to a registry
is a maintainer's decision, not a side effect of landing a workflow file.
`--publish` comes from a repository VARIABLE (`vars.PUBLISH_CONTAINER`), so the
files can be reviewed and merged with the registry step inert, and enabling it
later is one switch rather than one commit. Every plan says, in words, whether
it would push and why not.

Stdlib only, python3.9-clean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# The two architectures this image targets, and why aarch64 is not decoration:
# the cheapest measurement hardware measured anywhere is a Grace-Hopper GH200,
# which is aarch64.  See container/Dockerfile's MULTI-ARCH note for the wheel
# audit that says the pins hold on both.
PLATFORMS = ("linux/amd64", "linux/arm64")

SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


def parse_ref(ref: str):
    """(kind, name) for a git ref, without guessing.

    refs/tags/v1.2.3 -> ("tag", "v1.2.3");  refs/heads/main -> ("branch", "main")
    """
    if ref.startswith("refs/tags/"):
        return "tag", ref[len("refs/tags/"):]
    if ref.startswith("refs/heads/"):
        return "branch", ref[len("refs/heads/"):]
    return "other", ref


def tags_for(image: str, *, event: str, ref: str, sha: str):
    """Every tag this build should carry, most specific first.

    The immutable one is always present: a receipt cites a digest, but a human
    reading a receipt needs a tag that still means the same bytes tomorrow, and
    `latest` never does.
    """
    short = sha[:12]
    tags = ["%s:sha-%s" % (image, short)]
    kind, name = parse_ref(ref)
    match = SEMVER.match(name) if kind == "tag" else None
    if match:
        major, minor, patch, pre = match.groups()
        tags.append("%s:%s.%s.%s" % (image, major, minor, patch))
        if not pre:
            # A prerelease must not move the series tags: v1.2.3-rc1 is not
            # what `1.2` should resolve to.
            tags.append("%s:%s.%s" % (image, major, minor))
            tags.append("%s:%s" % (image, major))
            tags.append("%s:latest" % image)
    elif kind == "branch" and name in ("main", "master"):
        tags.append("%s:main" % image)
    elif event == "workflow_dispatch":
        tags.append("%s:dev" % image)
    return tags


def plan(args) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha or ""):
        raise SystemExit(
            "release_plan: --sha must be the full 40-hex commit.\n"
            "  The image records it as produced_by.revision, which the "
            "submission schema requires and has no 'unknown' value for.")
    kind, name = parse_ref(args.ref)
    publish = str(args.publish).strip().lower() in ("1", "true", "yes", "on")
    reasons = []
    if not publish:
        reasons.append(
            "publish gate is off: set the repository variable "
            "PUBLISH_CONTAINER=true to enable pushing to the registry. "
            "Landing this workflow does not start publishing anything.")
    if args.event == "pull_request":
        publish = False
        reasons.append("a pull request never pushes an image")
    return {
        "schema": "malaiwah.fidelity-release-plan.v1",
        "event": args.event,
        "ref": args.ref,
        "ref_kind": kind,
        "ref_name": name,
        "sha": args.sha,
        "image": args.image,
        "platforms": list(PLATFORMS),
        "tags": tags_for(args.image, event=args.event, ref=args.ref, sha=args.sha),
        "push": publish,
        "push_blocked_because": reasons,
        "build_args": {
            "SUITE_REVISION": args.sha,
            # The first tag is the immutable sha- one, so the image's own
            # BUILD.json records a reference that still means these bytes.
            "IMAGE_REFERENCE": tags_for(args.image, event=args.event,
                                        ref=args.ref, sha=args.sha)[0],
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="workflow_dispatch")
    ap.add_argument("--ref", default="refs/heads/main")
    ap.add_argument("--sha", required=True)
    ap.add_argument("--image", default="ghcr.io/malaiwah/quant-fidelity-measure")
    ap.add_argument("--publish", default="false",
                    help="the repository variable PUBLISH_CONTAINER")
    ap.add_argument("--github-output", action="store_true",
                    help="also print name=value lines for $GITHUB_OUTPUT")
    args = ap.parse_args(argv)

    doc = plan(args)
    print(json.dumps(doc, indent=2, sort_keys=True))
    if args.github_output:
        out = sys.stderr
        out.write("tags=%s\n" % ",".join(doc["tags"]))
        out.write("platforms=%s\n" % ",".join(doc["platforms"]))
        out.write("push=%s\n" % ("true" if doc["push"] else "false"))
        out.write("suite_revision=%s\n" % doc["build_args"]["SUITE_REVISION"])
        out.write("image_reference=%s\n" % doc["build_args"]["IMAGE_REFERENCE"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
