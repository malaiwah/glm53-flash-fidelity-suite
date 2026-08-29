"""`compat/` -- a kimi-k3 view of a v1 fidelity dataset.

Spec section 12.3.  With this tree present, Festr's UNMODIFIED
`compare-kimi-k3-hidden-replay.py` reads our dataset:

    python3 compare-kimi-k3-hidden-replay.py \\
        --suite-manifest  <root>/compat/suite-manifest.json \\
        --reference-dir   <root>/compat/reference-hidden \\
        --candidate-dir   <root>/compat/candidate-hidden \\
        --lm-head         <root>/head/weight.safetensors

Four incompatibilities, all of them naming and none of them semantic, make a
translation view necessary rather than optional:

  1. `panel.json.contexts` is an integer RECORD COUNT in v1 and the record LIST
     in his format -- the same key at the same nesting level with incompatible
     types (PANEL-D5).  His reader does `for context in contexts`, so ours
     raises `TypeError: 'int' object is not iterable`.  One file cannot spell
     both, which is why this is a separate tree and not an alias.
  2. our record list is `records`; he reads `contexts` or `windows`.
  3. `suite_token_hash_sha256` lives in `panel/panel.json` for us and is read
     from the CAPTURE manifest by him.
  4. `token_file` is dataset-root-relative for us; he resolves it against the
     suite manifest's OWN directory.

What this does NOT do is duplicate bytes.  The reference shim written during
review hardlinked every tensor and token into `compat/`, which is correct but
doubles what `checksums.txt` covers and what a `publish` uploads -- 86 GB
becomes 172 GB.  His loader resolves `directory / record["file"]` with pathlib
and calls `.is_file()`, so a relative alias such as
`../../capture/hidden_0000.safetensors` resolves to the one real tensor.  The
compat tree is therefore pure metadata: four small JSON files, whatever the
panel size.

SEAL ORDER: every file here is written BEFORE `DatasetWriter.finish`, so it is
listed in `checksums.txt` and covered by the seal.  Writing it afterwards makes
the dataset refuse itself with `unlisted_file` / SEAL-1(c) -- and carving
`compat/` OUT of the seal instead would be a hole aimed straight at the largest
files in the dataset.
"""

from __future__ import annotations

import json
import os
import posixpath
from typing import Any, Dict, List, Optional

from . import dsformat as F

COMPAT_ID = "kimi-k3-distribution-fidelity/1"

#: His tensor-key expectation is hard: `record.get("key") != expected_key` is a
#: RuntimeError, where expected_key is "hidden_states" or "logits". Ours are the
#: same two strings (adopted from him deliberately), so nothing is rewritten.
KIND_FOR_FORM = {"hidden": "hidden", "logit": "logits"}


def _compat_record(record: Dict[str, Any], depth: int, capture_rel: str) -> Dict[str, Any]:
    """One capture record, re-pathed relative to `compat/<dir>/`."""
    out = dict(record)
    out["context_index"] = int(record["index"])
    out["window_index"] = int(record["index"])
    out["file"] = posixpath.join("../" * depth, capture_rel, record["file"])
    return out


def emit(writer, *, panel_doc: Dict[str, Any], capture_doc: Dict[str, Any],
         manifest_capture: Dict[str, Any], head_relpath: Optional[str],
         dataset_name: str) -> Dict[str, Any]:
    """Write `compat/` through `writer`, before the seal.

    `writer` is a `dsmanifest.DatasetWriter` whose `add_file` already permits
    the `compat/` prefix.  Returns the interop summary the manifest records.
    """
    form = capture_doc.get("form") or manifest_capture.get("form") or "hidden"
    kind = KIND_FOR_FORM.get(form, "hidden")
    capture_rel = posixpath.dirname(manifest_capture["manifest_file"]) or "capture"
    subdir = "reference-hidden" if form == "hidden" else "reference-logits"

    contexts: List[Dict[str, Any]] = []
    for record in sorted(panel_doc["records"], key=lambda r: int(r["index"])):
        row = dict(record)
        row["context_index"] = int(record["index"])
        row["window_index"] = int(record["index"])
        # He resolves token_file against the suite manifest's own directory,
        # which here is compat/.
        row["token_file"] = posixpath.join("..", record["token_file"])
        contexts.append(row)

    suite = {
        "kind": "malaiwah fidelity panel, kimi-k3 compatibility view",
        "format_version": 1,
        "x_malaiwah_note": (
            "A DERIVED VIEW. The normative panel is ../panel/panel.json; this file exists "
            "because `contexts` is an integer record count there and a record list here "
            "(PANEL-D5), which one file cannot spell both ways. token_file entries are "
            "relative to THIS directory and point at the one real copy of each token file."),
        "x_malaiwah_panel_id": panel_doc.get("panel_id"),
        "x_malaiwah_normative_panel": "../panel/panel.json",
        "context_count": len(contexts),
        "context_length": panel_doc["context_length"],
        "scored_positions_per_context": panel_doc["positions_per_context"],
        "total_scored_positions": panel_doc["scored_positions_total"],
        "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
        "tokenizer": panel_doc.get("tokenizer"),
        "contexts": contexts,
    }
    writer.add_file("compat/suite-manifest.json",
                    (json.dumps(suite, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    source = {
        "kind": "malaiwah fidelity capture, kimi-k3 compatibility view",
        "format_version": 1,
        "x_malaiwah_note": ("A DERIVED VIEW of ../../%s/manifest.json, which is the normative "
                            "capture manifest and the one the seal covers. `file` entries are "
                            "relative aliases: there is exactly one copy of each tensor."
                            % capture_rel),
        "run_name": capture_doc.get("run_name") or dataset_name,
        "semantic_point": capture_doc.get("semantic_point"),
        "tensor_key": capture_doc.get("tensor_key"),
        "context_length": capture_doc.get("context_length"),
        "hidden_width": capture_doc.get("hidden_width"),
        "scored_rows_per_context": (capture_doc["records"][0]["scored_rows"]
                                    if capture_doc.get("records") else None),
        # (3) he reads the suite token hash from the CAPTURE manifest.
        "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
        "total_size_bytes": capture_doc.get("total_size_bytes"),
        # (2) `contexts`, not `records`.
        "contexts": [_compat_record(record, 2, capture_rel)
                     for record in sorted(capture_doc["records"],
                                          key=lambda r: int(r["index"]))],
    }
    writer.add_file("compat/%s/manifest.json" % subdir,
                    (json.dumps(source, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    readme = _README % {
        "name": dataset_name, "subdir": subdir, "kind": kind,
        "head": head_relpath or "head/weight.safetensors",
        "count": len(contexts),
    }
    writer.add_file("compat/README.md", readme.encode("utf-8"))
    return {
        "k3_compat_emitted": True,
        "k3_compat_files": ["compat/suite-manifest.json",
                            "compat/%s/manifest.json" % subdir,
                            "compat/README.md"],
        "k3_compat_tensor_bytes_duplicated": 0,
        "k3_compat_note": ("metadata only: `file` and `token_file` are relative aliases onto "
                           "the one real copy of each tensor and token file, so the view "
                           "costs three small JSON files at any panel size."),
    }


_README = """# kimi-k3 compatibility view

Three JSON files that let the kimi-k3 comparator read **%(name)s** without a
patch. They are inside the seal: `checksums.txt` lists them, and editing one
makes the dataset refuse itself.

    python3 compare-kimi-k3-hidden-replay.py \\
        --suite-manifest  ../compat/suite-manifest.json \\
        --reference-dir   ../compat/%(subdir)s \\
        --candidate-dir   <the other dataset>/compat/%(subdir)s \\
        --lm-head         ../%(head)s

Nothing here is a second copy of the data. `file` and `token_file` are relative
aliases onto the one real tensor and the one real token file. %(count)d contexts.

The normative documents are `../panel/panel.json` and the capture manifest; if
the two ever disagree, those win and this view is a bug.
"""


def verify(root: str) -> List[str]:
    """Check a `compat/` tree against the dataset it claims to describe.

    Returns a list of problems, empty when the view is faithful.  This is the
    check that stops the compat tree drifting into a second, quietly different
    description of the same bytes.
    """
    problems: List[str] = []
    compat = os.path.join(root, "compat")
    suite_path = os.path.join(compat, "suite-manifest.json")
    if not os.path.isfile(suite_path):
        return ["compat/suite-manifest.json is absent"]
    manifest = F.load_manifest(root)
    panel = F.read_json(os.path.join(root, manifest["panel"]["panel_file"]))
    suite = F.read_json(suite_path)

    if suite.get("suite_token_hash_sha256") != panel["suite_token_hash_sha256"]:
        problems.append("compat/suite-manifest.json carries a different suite token hash "
                        "than panel/panel.json")
    if not isinstance(suite.get("contexts"), list):
        problems.append("compat/suite-manifest.json `contexts` must be a LIST (that is the "
                        "whole reason this view exists)")
        return problems
    if len(suite["contexts"]) != len(panel["records"]):
        problems.append("compat lists %d contexts, the panel has %d"
                        % (len(suite["contexts"]), len(panel["records"])))
    by_index = {int(r["index"]): r for r in panel["records"]}
    for row in suite["contexts"]:
        index = int(row.get("context_index", row.get("index")))
        real = by_index.get(index)
        if real is None:
            problems.append("compat context %d is not in the panel" % index)
            continue
        if row.get("token_ids_json_sha256") != real["token_ids_json_sha256"]:
            problems.append("compat context %d carries a different token digest" % index)
        resolved = os.path.normpath(os.path.join(compat, row["token_file"]))
        if not os.path.isfile(resolved):
            problems.append("compat context %d token_file %r does not resolve (he joins it "
                            "onto the suite manifest's own directory)" % (index, row["token_file"]))

    for name in sorted(os.listdir(compat)):
        source_path = os.path.join(compat, name, "manifest.json")
        if not os.path.isfile(source_path):
            continue
        source = F.read_json(source_path)
        if source.get("suite_token_hash_sha256") != panel["suite_token_hash_sha256"]:
            problems.append("compat/%s/manifest.json carries a different suite token hash"
                            % name)
        if not isinstance(source.get("contexts"), list):
            problems.append("compat/%s/manifest.json `contexts` must be a LIST" % name)
            continue
        capture = F.read_json(os.path.join(root, manifest["capture"]["manifest_file"]))
        real_records = {int(r["index"]): r for r in capture["records"]}
        for row in source["contexts"]:
            index = int(row.get("context_index", row.get("index")))
            real = real_records.get(index)
            if real is None:
                problems.append("compat/%s record %d is not in the capture manifest"
                                % (name, index))
                continue
            if row.get("sha256") != real["sha256"]:
                problems.append("compat/%s record %d carries a different file digest"
                                % (name, index))
            resolved = os.path.normpath(os.path.join(compat, name, row["file"]))
            if not os.path.isfile(resolved):
                problems.append("compat/%s record %d file %r does not resolve"
                                % (name, index, row["file"]))
    return problems
