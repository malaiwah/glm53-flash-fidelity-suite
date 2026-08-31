#!/usr/bin/env python3
"""Guards for the class of defect a NAMING SWEEP creates.

This tree was renamed `glm53-fidelity-suite` -> `quant-fidelity-suite`, and
the code is being swept to follow.  A sweep like that is a global rewrite of
identifiers, which is exactly the operation that can silently destroy a
published number:

  N1  a registry id, a receipt schema string, or a provenance PATH is a
      HASHED, PUBLISHED identity.  `COMPARABILITY_KEY_FIELDS` hashes
      `panel_id` and `reference_id`, so renaming one regroups every
      measurement that referenced it -- with no error anywhere.  `harness_id`
      is a sha256 over {boundary, [{role, PATH, sha256}], tool_versions}, so a
      code path a published row names is inside that hash and must keep the
      spelling the tree had WHEN THE NUMBER RAN, however the tree is arranged
      today.  `bin/published-identity.json` freezes all of them; this file
      refuses a tree where one has vanished.

  N2  a filesystem ROOT on rented hardware with a MODEL or CAMPAIGN name
      baked into it.  This is the same defect as a root with a PROVIDER name
      baked in, one axis over.  Three `/home/jl_fs` roots were each found by a
      paid run -- one of them stalled an A100 at 0% GPU for two hours -- and
      `bin/selftest_provider_portability.py` is the rule that finds the fourth
      without renting anything.  `/home/jl_fs/glm53-k6` was BOTH at once.

  N3  a two-file agreement broken on one side.  An env var the controller
      exports but no on-instance script reads; an entrypoint `engines.json`
      names that the bundle does not ship; a convention path one file writes
      and another reads.  Each is silent until a rented box is already
      running.

  N4  a helper duplicated in two places, free to drift.  The pattern is
      `bin/selftest_canonical_json.py`.  This rung is the general form.

Offline.  Stock python3.9, no installs.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import _identity_scan as scan                                    # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        if detail:
            for line in str(detail).splitlines()[:20]:
                print("        %s" % line)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# N1  published identity is frozen
# ---------------------------------------------------------------------------
print("== N1: nothing hashed, sealed or published may be renamed ==")

frozen = json.loads(_read(os.path.join(HERE, "published-identity.json")))
live = scan.snapshot(ROOT)

for key, what in (
        ("registry_ids",
         "registry ids (hashed into comparability.key)"),
        ("receipt_schemas",
         "schema strings inside sealed receipts"),
        ("code_schema_literals",
         "schema literals the code emits into receipts"),
        ("provenance_paths",
         "code paths a published row names as its provenance"),
):
    gone = sorted(set(frozen[key]) - set(live[key]))
    check("every frozen %s still exists (%d frozen, %d live)"
          % (what, len(frozen[key]), len(live[key])),
          not gone, "vanished: " + ", ".join(gone))

# The freeze is only worth something if it is not trivially empty.
check("the freeze covers a real corpus (>= 150 ids, >= 200 schema literals)",
      len(frozen["registry_ids"]) >= 150
      and len(frozen["code_schema_literals"]) >= 200)

# ...and it must actually contain the GLM-era names, since those are the ones
# a sweep is tempted to "fix".
glm_frozen = [s for s in frozen["registry_ids"] + frozen["code_schema_literals"]
              if "glm" in s.lower()]
check("the freeze includes the GLM-era names a sweep would be tempted to fix "
      "(%d of them)" % len(glm_frozen), len(glm_frozen) >= 150)


# ---------------------------------------------------------------------------
# N3  two-file agreements
# ---------------------------------------------------------------------------
# Files bin/BUNDLE.txt uploads that RUN on rented hardware and construct paths.
# Kept in step with selftest_provider_portability.py's list of the same name.
ON_INSTANCE = ("invoke_engine.py", "invoke_scorer.py", "stage_measure.sh",
               "bootstrap_measure.sh", "watchdog.sh", "seal_receipt.py",
               "stage_panel_paths.py")


# ---------------------------------------------------------------------------
# N2  no model or campaign name inside an on-instance filesystem ROOT
# ---------------------------------------------------------------------------
print("\n== N2: a root on rented hardware names neither provider nor model ==")

# A model, an architecture family, or a campaign profile.  None of these
# belongs in a directory name on a box that may be measuring something else.
MODEL_TOKENS = ("glm", "qwen", "minimax", "deepseek", "fruit", "kimi",
                "k6", "k8", "tr3", "exl3", "gguf", "mlx", "nvfp4")

# An absolute path rooted where a provider mounts storage.  Scoped to those
# prefixes on purpose: `$FS/engines/patches-v2` is a path INSIDE the upload
# tree and is named by the suite, while `/home/jl_fs/<x>` and `/workspace/<x>`
# are the roots the controller has to choose and export.
_ROOT_PATH = re.compile(
    r'(?<![:\w/])(/(?:home|workspace|mnt|data|root|opt|srv)'
    r'(?:/[A-Za-z0-9._$~{}*-]+)+)')

offenders = []
for fname in ON_INSTANCE:
    path = os.path.join(HERE, fname)
    if not os.path.isfile(path):
        continue
    for n, line in enumerate(_read(path).splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue          # a comment may name the history it explains
        for hit in _ROOT_PATH.findall(line):
            low = hit.lower()
            if any(tok in low for tok in MODEL_TOKENS):
                offenders.append("%s:%d %s" % (fname, n, hit))
for o in offenders:
    print("      %s" % o)
check("no instance root in an on-instance tool names a model or a campaign",
      not offenders,
      "a run measuring MiniMax must not live in a directory called glm53-k6")

# The rule is worthless if it never looks at anything, and worthless again if
# it never looks at a path -- both of which a rename could quietly cause.
scanned = [f for f in ON_INSTANCE if os.path.isfile(os.path.join(HERE, f))]
check("...and it read %d on-instance files" % len(scanned), len(scanned) >= 6,
      "found: %s" % ", ".join(scanned))
seen = sum(len(_ROOT_PATH.findall(ln))
           for f in scanned
           for ln in _read(os.path.join(HERE, f)).splitlines()
           if not ln.strip().startswith("#"))
check("...and found %d instance-root literals to judge" % seen, seen >= 3)


print("\n== N3: both sides of every two-file agreement ==")

sys.path.insert(0, HERE)
import measure_cloud as mc                                       # noqa: E402


class _Prov:
    provider = "runpod"


class _Con:
    def __getattr__(self, _):
        return lambda *a, **k: None


_td = mc.Teardown(_Prov(), _Con(), mc.Path("."))
stage_env = mc._stage_env(_td)

# (a) every FIDELITY_*_ROOT the controller EXPORTS must be READ by an
#     on-instance script, and every one they read must be exported.  The old
#     spelling of one of these (FIDELITY_K6_ROOT -> FIDELITY_ENGINE_ROOT) is
#     deliberately still
#     accepted as a fallback, so a fallback read does not count as the
#     agreement -- the exported name has to appear.
exported = sorted(set(re.findall(r"\b(FIDELITY_[A-Z0-9_]*ROOT)=", stage_env)))
check("the controller exports at least two FIDELITY_*_ROOT names",
      len(exported) >= 2, "exported: %s" % exported)

on_instance_text = "\n".join(
    _read(os.path.join(HERE, f)) for f in ON_INSTANCE
    if os.path.isfile(os.path.join(HERE, f)))
for name in exported:
    check("%s is exported AND read on the instance" % name,
          name in on_instance_text,
          "the controller sets it and nothing on the box looks at it")

# (b) the container image pin: stackprint READS the path remote/vm_setup.sh
#     WRITES.  `docker load` strips tags, so this file is the only trustworthy
#     record of what actually ran -- and the two sides live in different trees.
from fidelity import stackprint                                  # noqa: E402

vm_setup = os.path.join(ROOT, "remote", "vm_setup.sh")
if os.path.isfile(vm_setup):
    written = _read(vm_setup)
    conv = stackprint.IMAGE_PIN_CONVENTION_PATH
    tail = conv.split("/", 2)[-1] if conv.count("/") >= 2 else conv
    check("stackprint's image-pin path is the one remote/vm_setup.sh writes "
          "(%s)" % conv,
          tail in written,
          "vm_setup.sh writes a different file; the digest would be lost")

# (c) every engines.json entrypoint exists on disk and travels in the bundle.
engines = json.loads(_read(os.path.join(HERE, "engines.json")))
bundle_lines = [ln.strip() for ln
                in _read(os.path.join(HERE, "BUNDLE.txt")).splitlines()
                if ln.strip() and not ln.startswith("#")]
bundle = set(bundle_lines)


def _entrypoints(node, out):
    if isinstance(node, dict):
        for key, val in node.items():
            if key == "entrypoint" and isinstance(val, str) and val.endswith(".py"):
                out.add(val)
            _entrypoints(val, out)
    elif isinstance(node, list):
        for item in node:
            _entrypoints(item, out)


eps = set()
_entrypoints(engines, eps)
check("engines.json declares entrypoints at all", len(eps) >= 3, sorted(eps))
missing_disk = sorted(e for e in eps if not os.path.isfile(os.path.join(ROOT, e)))
check("every engines.json entrypoint exists on disk", not missing_disk,
      missing_disk)

# Only the lanes that RUN ON RENTED HARDWARE have to travel.  `local-*` lanes
# execute on the caller's own machine out of the checkout, which is why
# bin/kld_preview.py is their scorer and is deliberately not in the bundle.
remote_eps = set()
for lane_name, lane in engines.get("lanes", {}).items():
    if lane_name.startswith("local-"):
        continue
    _entrypoints(lane, remote_eps)
missing_bundle = sorted(e for e in remote_eps
                        if os.path.isfile(os.path.join(ROOT, e)) and e not in bundle)
check("every non-local engines.json entrypoint is in BUNDLE.txt (%d checked)"
      % len(remote_eps),
      not missing_bundle,
      "a renamed engine that BUNDLE.txt still spells the old way uploads "
      "nothing and dies at hour zero: %s" % missing_bundle)
check("...and the local lanes' scorer is deliberately NOT bundled",
      "bin/kld_preview.py" in eps and "bin/kld_preview.py" not in remote_eps)

# (d) BUNDLE.txt lists no path that does not exist.  selftest_root_capture.py
#     checks this too; it is repeated here because THIS file is the one a
#     rename sweep is run against, and a missing entry is skipped silently by
#     the uploader.
absent = sorted(p for p in bundle if not os.path.exists(os.path.join(ROOT, p)))
check("every BUNDLE.txt entry exists", not absent, absent)


# ---------------------------------------------------------------------------
# N4  no helper duplicated byte-for-byte in two places
# ---------------------------------------------------------------------------
print("\n== N4: one helper, one copy ==")

import hashlib                                                   # noqa: E402

# Two byte-identical executables in one checkout are a fork waiting to happen:
# a fix lands in the copy that is invoked and the other rots, or worse, the
# other one is what gets uploaded.  `k6_publish.py` (now publish_release.py)
# existed twice, identically, at engines/ and engines/tools/, and only the engines/tools/
# copy was ever run.
by_digest = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if d not in scan._SKIP_DIRS and not d.endswith("-evidence")]
    for name in filenames:
        if not name.endswith((".py", ".sh")):
            continue
        full = os.path.join(dirpath, name)
        if os.path.islink(full):
            continue
        with open(full, "rb") as fh:
            body = fh.read()
        if len(body) < 512:
            continue          # a two-line shim is not a forked helper
        by_digest.setdefault(hashlib.sha256(body).hexdigest(), []).append(
            os.path.relpath(full, ROOT))

# `.patchwork/a` and `.patchwork/b` are two pinned snapshots of the SAME
# upstream tree, kept side by side on purpose so the decode-parity selftest can
# diff them. They are evidence, not helpers.
def _exempt(paths):
    return all("/.patchwork/" in "/" + p for p in paths)


dups = sorted(v for v in by_digest.values() if len(v) > 1 and not _exempt(v))
for group in dups:
    print("      identical: %s" % " == ".join(sorted(group)))
check("no script exists byte-identically in two places", not dups,
      "delete the copy nothing invokes, or make one import the other")


print()
if FAILED:
    print("selftest_naming_sweep: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_naming_sweep: all passed")
