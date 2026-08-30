#!/usr/bin/env python3
"""Build a sealed token panel from a published plain-text corpus tree.

Why this file exists
--------------------
``k6/tools/hf_capture.py`` consumes a panel directory in the upstream
``quant-pipeline.glm53-token-panel.v1`` layout (``panel.json`` + ``arrays/``),
but every such panel we had was built for GLM-5.3-Flash by somebody else's
pipeline.  Reusing one of those against a *different* model is exactly the
cross-model comparison our own rules forbid: the token ids may be numerically
valid (a shared vocabulary is common), yet the panel was selected for another
model's corpus and another model's calibration separation, and a number
measured on it invites being ranked against numbers it has no business being
ranked against.

So: a model that needs its own yardstick gets its own panel, with its own
``panel_id``, built by a rule anyone can re-run.

The rule (deterministic, no RNG)
--------------------------------
1. Take a corpus tree of plain-text documents, one subdirectory per stratum,
   pinned by a published per-file sha256 list.
2. Within each requested stratum, sort documents by file name, ascending.
3. Tokenize each document whole, ``add_special_tokens=False``.
4. Keep a document only if it yields at least ``skip + context_length`` tokens;
   its window is ``tokens[skip : skip + context_length]``.  The skip exists
   because the first few hundred tokens of a real document are title pages,
   headers and boilerplate, which are the least representative text in it.
5. Take the first ``--windows-per-stratum`` surviving documents per stratum.
6. Emit windows ordered by (stratum, document name), numbered from zero.

Every one of those steps is a sorted traversal or a fixed slice, so two runs on
the same corpus revision and the same tokenizer produce byte-identical arrays.
``panel.receipt.json`` records the corpus revision, every source document's
sha256, the byte-exact selection rule, and the tokenizer file digests, so the
panel can be rebuilt from public inputs alone.

Contamination
-------------
This tool does **not** run a contamination scan.  It records which strata were
used and what the caller declared about them (``--separation-note``), and the
declaration is copied verbatim into the receipt.  Claiming "held out" is the
caller's claim to defend; the tool only makes it visible and pinned.
"""

import argparse
import hashlib
import json
import os
import sys
import time


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(doc):
    """The repository's four-line self-blanked seal recipe."""
    body = dict(doc)
    body["receipt_sha256"] = ""
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def token_ids_json_sha256(ids):
    """Spec 5.1 per-record preimage: compact separators."""
    return sha256_bytes(json.dumps([int(v) for v in ids],
                                   separators=(",", ":")).encode("utf-8"))


def suite_token_hash_sha256(per_record_hex):
    """Spec 5.1 aggregate preimage: newline join, ascending record order."""
    return sha256_bytes("\n".join(per_record_hex).encode("ascii"))


def die(msg):
    sys.stderr.write("build_token_panel: %s\n" % msg)
    raise SystemExit(4)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="build_token_panel",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="root of the text tree; one subdirectory per stratum")
    ap.add_argument("--stratum", action="append", required=True, dest="strata",
                    help="stratum subdirectory to draw from; repeatable, order irrelevant")
    ap.add_argument("--corpus-repository", default=None,
                    help="where the corpus tree is published, for the receipt")
    ap.add_argument("--corpus-revision", default=None)
    ap.add_argument("--corpus-path-prefix", default="",
                    help="prefix that turns a stratum-relative path into a repository path")
    ap.add_argument("--tokenizer", required=True,
                    help="HF repo id or a local directory holding tokenizer.json")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--tokenizer-id", default=None,
                    help="the name the tokenizer is known by, for the receipt")
    ap.add_argument("--tokenizer-repository", default=None,
                    help="the repository the tokenizer files came from. Recorded INSTEAD of "
                         "--tokenizer when that is a local directory: a published receipt "
                         "must never name a host-local path (spec PATH-2).")
    ap.add_argument("--windows-per-stratum", type=int, default=8)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--skip-tokens", type=int, default=2048,
                    help="tokens dropped from the head of each document before the window")
    ap.add_argument("--panel-id", required=True)
    ap.add_argument("--panel-name", required=True)
    ap.add_argument("--separation-note", default=None,
                    help="what the caller claims about this panel's separation from training "
                         "or calibration data; copied verbatim, never verified")
    ap.add_argument("--out", required=True, help="panel directory to write")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.context_length < 2:
        die("--context-length must be at least 2")
    if args.skip_tokens < 0:
        die("--skip-tokens must not be negative")
    if os.path.exists(args.out) and not args.force:
        die("%s exists (use --force)" % args.out)

    import numpy as np
    from transformers import AutoTokenizer

    tok_kwargs = {}
    if args.tokenizer_revision:
        tok_kwargs["revision"] = args.tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **tok_kwargs)

    tokenizer_files = {}
    if os.path.isdir(args.tokenizer):
        for name in sorted(os.listdir(args.tokenizer)):
            path = os.path.join(args.tokenizer, name)
            if os.path.isfile(path):
                tokenizer_files[name] = sha256_file(path)

    arrays = os.path.join(args.out, "arrays")
    os.makedirs(arrays, exist_ok=True)

    mask = np.ones(args.context_length, dtype=np.uint8)
    mask_path = os.path.join(arrays, "causal-mask-%d.npy" % args.context_length)
    np.save(mask_path, mask, allow_pickle=False)
    mask_sha = sha256_file(mask_path)

    windows = []
    sources = []
    considered = 0
    for stratum in sorted(args.strata):
        directory = os.path.join(args.corpus_dir, stratum)
        if not os.path.isdir(directory):
            die("no stratum directory %s" % directory)
        taken = 0
        for name in sorted(os.listdir(directory)):
            if taken >= args.windows_per_stratum:
                break
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            considered += 1
            raw = open(path, "rb").read()
            text = raw.decode("utf-8")
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            need = args.skip_tokens + args.context_length
            if len(ids) < need:
                sources.append({"stratum": stratum, "document": name,
                                "document_sha256": sha256_bytes(raw), "bytes": len(raw),
                                "tokens_total": len(ids), "selected": False,
                                "reason": "fewer than %d tokens" % need})
                continue
            window_ids = [int(v) for v in ids[args.skip_tokens:need]]
            index = len(windows)
            window_id = "final-%04d" % index
            token_path = os.path.join(arrays, "%s.tokens.npy" % window_id)
            np.save(token_path, np.asarray(window_ids, dtype=np.int32), allow_pickle=False)
            repo_path = (args.corpus_path_prefix + stratum + "/" + name)
            windows.append({
                "window_id": window_id,
                "role": "final",
                "domain": stratum,
                "document_id": repo_path,
                "prediction_positions": args.context_length - 1,
                "token_ids_sha256": sha256_file(token_path),
                "attention_mask_sha256": mask_sha,
                # spec 5.1 preimages, carried so the sealed panel and the
                # dataset written from it agree without recomputation
                "token_ids_json_sha256": token_ids_json_sha256(window_ids),
                "token_ids_first16": window_ids[:16],
                "token_ids_last16": window_ids[-16:],
                "num_tokens": len(window_ids),
            })
            sources.append({"stratum": stratum, "document": name,
                            "document_sha256": sha256_bytes(raw), "bytes": len(raw),
                            "tokens_total": len(ids), "selected": True,
                            "window_id": window_id,
                            "token_slice": [args.skip_tokens, need]})
            taken += 1
        if taken < args.windows_per_stratum:
            die("stratum %s yielded only %d of %d requested windows"
                % (stratum, taken, args.windows_per_stratum))

    if not windows:
        die("no windows produced")

    aggregate = suite_token_hash_sha256([w["token_ids_json_sha256"] for w in windows])

    panel = {
        "schema": "quant-pipeline.glm53-token-panel.v1",
        "panel_id": args.panel_id,
        "name": args.panel_name,
        "sealed_corpus_sha256": None,
        "suite_token_hash_sha256": aggregate,
        "windows": windows,
    }
    with open(os.path.join(args.out, "panel.json"), "w", encoding="utf-8") as handle:
        json.dump(panel, handle, indent=2, sort_keys=False)
        handle.write("\n")

    receipt = {
        "schema": "malaiwah.token-panel-build-receipt.v1",
        "format_version": 1,
        "receipt_sha256": "",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "k6/tools/build_token_panel.py",
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "panel_id": args.panel_id,
        "panel_name": args.panel_name,
        "suite_token_hash_sha256": aggregate,
        "selection_rule": (
            "strata sorted ascending; within each stratum documents sorted by file name "
            "ascending; each document tokenized whole with add_special_tokens=False; a "
            "document is eligible only if it yields at least skip_tokens+context_length "
            "tokens; its window is tokens[skip_tokens : skip_tokens+context_length]; the "
            "first windows_per_stratum eligible documents are taken; windows are numbered "
            "from zero in (stratum, document) order. No RNG is used anywhere."),
        "parameters": {
            "strata": sorted(args.strata),
            "windows_per_stratum": args.windows_per_stratum,
            "context_length": args.context_length,
            "skip_tokens": args.skip_tokens,
            "prediction_positions_per_window": args.context_length - 1,
            "windows_total": len(windows),
            "scored_positions_total": len(windows) * (args.context_length - 1),
        },
        "corpus": {
            "repository": args.corpus_repository,
            "revision": args.corpus_revision,
            "path_prefix": args.corpus_path_prefix,
            "documents_considered": considered,
            "documents_selected": len(windows),
        },
        "tokenizer": {
            # PATH-2: never the local --tokenizer directory. A published receipt
            # that names /private/tmp/... on the builder's laptop is unreadable
            # provenance and a privacy leak; the repository is the real identity.
            "id": args.tokenizer_id or args.tokenizer_repository,
            "repository": args.tokenizer_repository,
            "revision": args.tokenizer_revision,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)) or None,
            "add_special_tokens": False,
            "chat_template_applied": False,
            "files_sha256": tokenizer_files or None,
        },
        "separation": {
            "checked": False,
            "method": "declaration only; this tool runs no lexical or n-gram scan",
            "note": args.separation_note,
        },
        "documents": sources,
    }
    receipt["receipt_sha256"] = seal(receipt)
    with open(os.path.join(args.out, "panel.receipt.json"), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "panel_id": args.panel_id,
        "out": args.out,
        "windows": len(windows),
        "context_length": args.context_length,
        "scored_positions_total": len(windows) * (args.context_length - 1),
        "suite_token_hash_sha256": aggregate,
        "panel_receipt_sha256": receipt["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
