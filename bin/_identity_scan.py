#!/usr/bin/env python3
"""The scanner behind `bin/selftest_naming_sweep.py`'s frozen-identity rung.

It is a MODULE, not a test, for one reason: the same code has to produce the
frozen snapshot (`bin/published-identity.json`) and to re-read the tree when
the test runs.  Two implementations of one scan is two chances to disagree,
which is the defect `bin/selftest_canonical_json.py` exists for.

Stock python3.9, no installs (AGENTS.md dependency rule for `bin/`).
"""
import json
import os
import re

# Every schema namespace this project has ever sealed a receipt under.  A
# receipt's `schema` is covered by its own `receipt_sha256`, so these strings
# are content, not naming.
_SCHEMA_RE = re.compile(
    r'"((?:malaiwah|quant-pipeline|glm53flash)[A-Za-z0-9._/+-]*)"')

# Files whose *bytes* carry identity.  They are scanned for identity, never
# rewritten by a naming sweep.
_TEXT_EXT = (".py", ".sh", ".json", ".txt", ".jsonl")

_SKIP_DIRS = {".git", ".venv", "__pycache__", "fidelity-runs", "fidelity-local",
              "bundle_stage", "corpus_dl", "tokenizer", "cal_data", "deliverables"}


def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(_TEXT_EXT):
                yield os.path.join(dirpath, name)


def registry_ids(root):
    """Every published record id in `registry/data/*.jsonl`.

    These are the strings `COMPARABILITY_KEY_FIELDS` hashes.  Renaming one
    regroups every measurement that pointed at it.
    """
    out = set()
    data = os.path.join(root, "registry", "data")
    if not os.path.isdir(data):
        return out
    for name in sorted(os.listdir(data)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(data, name), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rid = json.loads(line).get("id")
                if isinstance(rid, str) and "--" in rid:
                    out.add(rid)
    return out


def receipt_schemas(root):
    """Every `schema` value inside a sealed receipt under `registry/receipts/`."""
    out = set()

    def visit(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "schema" and isinstance(val, str):
                    out.add(val)
                visit(val)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    rec = os.path.join(root, "registry", "receipts")
    if not os.path.isdir(rec):
        return out
    for path in _walk(rec):
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as fh:
                visit(json.load(fh))
    return out


def provenance_paths(root):
    """Every repo-relative code path a PUBLISHED row names as provenance.

    Two kinds, and both are identity rather than configuration:

    * `harness.code_digests[].path` -- `harness_id` is a sha256 over
      {boundary, [{role, path, sha256}], tool_versions}, so the PATH is inside
      the hash. Two rows share an id exactly when they share this list.
    * `implementation.entrypoint` -- the published statement of which tool
      computed the row.

    They name the tree AS IT WAS when the number was produced. A rename sweep
    that "fixes" them retro-stamps rows with the identity of code that did not
    run, which is the exact failure `registry/tools/harness_id.py` exists to
    prevent -- and `seed_registry.py` transcribes these digests rather than
    re-reading today's checkout for the same reason.
    """
    out = set()
    data = os.path.join(root, "registry", "data")
    if not os.path.isdir(data):
        return out
    for name in ("measurements.jsonl", "pipelines.jsonl"):
        path = os.path.join(data, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for cd in ((rec.get("harness") or {}).get("code_digests") or []):
                    if isinstance(cd, dict) and isinstance(cd.get("path"), str):
                        out.add(cd["path"])
                ep = (rec.get("implementation") or {}).get("entrypoint")
                if isinstance(ep, str) and ep:
                    out.add(ep)
    return out


def code_schema_literals(root):
    """Schema-namespace string literals anywhere in the tree's code and data.

    Scanned by CONTENT across the whole checkout rather than from a list of
    paths, so a file that moves during a rename sweep does not look like a
    string that vanished.  Markdown is excluded on purpose: a schema mentioned
    only in prose must not be able to keep a deleted code literal "present".
    """
    out = set()
    for path in _walk(root):
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        out.update(_SCHEMA_RE.findall(text))
    return out


def snapshot(root):
    return {
        "registry_ids": sorted(registry_ids(root)),
        "receipt_schemas": sorted(receipt_schemas(root)),
        "code_schema_literals": sorted(code_schema_literals(root)),
        "provenance_paths": sorted(provenance_paths(root)),
    }


if __name__ == "__main__":
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    print(json.dumps(snapshot(os.path.dirname(here)), indent=1, sort_keys=True),
          file=sys.stdout)
