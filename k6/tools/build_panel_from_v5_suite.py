#!/usr/bin/env python3
"""Rebuild the sealed malaiwah Qwen3.8 suite-v5 shard-0 token panel as a
``quant-pipeline.glm53-token-panel.v1`` tree that ``k6/tools/hf_capture.py``
can consume.

Why this exists
---------------
The 37 existing Qwen3.8-27B registry rows were scored on
``panel--qwen38.malaiwah.suite-v5-shard0-1m``.  That panel's registry record
carries ``availability.status = "private"`` and ``uri = null``, and its only
recorded source is a receipt file on the author's laptop -- so on its face the
panel looked unreusable and a fresh panel looked mandatory.

It is in fact fully recoverable and byte-verifiable.  The public dataset
``malaiwah/qwen38-27b-fidelity-suite-v5`` carries every context's token ids
under ``suite/tokens/context-NNNN.json``.  This tool fetches the 512 shard-0
contexts at a pinned revision and verifies TWO independent seals before it
writes anything:

  1. each context file's sha256 equals the ``token_sha256`` recorded for it in
     the sealed suite manifest; and
  2. the sha256 of the 512 digests concatenated in ``context_index`` order --
     the suite's own aggregate rule, see ``tools/kld_aggregate.py`` in
     qwen38-27b-exl3 -- equals the registry panel's sealed
     ``identity.panel_token_sha256``.

Only if both hold is this the same panel.  Reusing it rather than minting a
fresh one keeps the token content fixed, so a new-lane row and an old-lane row
differ by the lane ALONE.  That does not make them rankable against each other
(the comparability key binds the reference, and the references differ), but it
does make the difference between them interpretable.

No RNG. No tokenizer is loaded: the token ids are transported, never re-derived.
"""

import argparse
import hashlib
import json
import os
import sys


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_ids_json_sha256(ids):
    """Spec 5.1 per-record preimage: compact separators."""
    return sha256_bytes(json.dumps([int(v) for v in ids],
                                   separators=(",", ":")).encode("utf-8"))


def suite_token_hash_sha256(per_record_hex):
    """Spec 5.1 aggregate preimage: newline join, ascending record order."""
    return sha256_bytes("\n".join(per_record_hex).encode("ascii"))


def seal(doc):
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def die(msg):
    sys.stderr.write("build_panel_from_v5_suite: %s\n" % msg)
    raise SystemExit(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", required=True,
                    help="directory of fetched context-NNNN.json token files")
    ap.add_argument("--suite-manifest", required=True,
                    help="the shard's sealed suite-manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel-id", required=True)
    ap.add_argument("--panel-name", required=True)
    ap.add_argument("--expect-panel-token-sha256", required=True,
                    help="the registry panel's sealed identity.panel_token_sha256")
    ap.add_argument("--corpus-repository",
                    default="malaiwah/qwen38-27b-fidelity-suite-v5")
    ap.add_argument("--corpus-revision", required=True)
    ap.add_argument("--tokenizer-repository", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--tokenizer-revision", default=None)
    args = ap.parse_args()

    import numpy as np

    manifest = json.loads(open(args.suite_manifest, "r", encoding="utf-8").read())
    index = manifest.get("context_index")
    if not isinstance(index, list) or not index:
        die("suite manifest has no context_index")

    arrays = os.path.join(args.out, "arrays")
    os.makedirs(arrays, exist_ok=True)

    context_length = int(manifest["context_length"])
    mask = np.ones(context_length, dtype=np.uint8)
    mask_path = os.path.join(arrays, "causal-mask-%d.npy" % context_length)
    np.save(mask_path, mask, allow_pickle=False)
    mask_sha = sha256_file(mask_path)

    windows = []
    file_digests = []
    for position, row in enumerate(index):
        name = os.path.basename(row["file"])
        src = os.path.join(args.tokens_dir, name)
        if not os.path.isfile(src):
            die("missing fetched token file %s" % src)
        raw = open(src, "rb").read()

        # SEAL 1: this file is the file the suite manifest sealed.
        got = sha256_bytes(raw)
        if got != row["token_sha256"]:
            die("context %s: fetched digest %s != sealed %s"
                % (name, got[:12], row["token_sha256"][:12]))
        file_digests.append(got)

        ids = json.loads(raw)
        if not isinstance(ids, list) or len(ids) != int(row["tokens"]):
            die("context %s: expected %s ids, got %s"
                % (name, row["tokens"], len(ids)))
        if len(ids) != context_length:
            die("context %s: length %d != context_length %d"
                % (name, len(ids), context_length))

        window_id = "final-%04d" % position
        token_path = os.path.join(arrays, "%s.tokens.npy" % window_id)
        np.save(token_path, np.asarray(ids, dtype=np.int32), allow_pickle=False)

        windows.append({
            "window_id": window_id,
            "role": "final",
            "domain": row["stratum"],
            "document_id": row["source_cluster"],
            "prediction_positions": context_length - 1,
            "token_ids_sha256": sha256_file(token_path),
            "attention_mask_sha256": mask_sha,
            "token_ids_json_sha256": token_ids_json_sha256(ids),
            "token_ids_first16": [int(v) for v in ids[:16]],
            "token_ids_last16": [int(v) for v in ids[-16:]],
            "num_tokens": len(ids),
            # provenance back to the sealed suite
            "suite_context_index": int(row["index"]),
            "suite_token_sha256": got,
            "suite_partition": row.get("partition"),
            "source_window_index": row.get("source_window_index"),
            "source_char_start": row.get("source_char_start"),
            "source_char_end": row.get("source_char_end"),
        })

    # SEAL 2: the suite's own aggregate rule must reproduce the registry
    # panel's sealed digest, or this is not that panel.
    recomputed = sha256_bytes("".join(file_digests).encode())
    if recomputed != args.expect_panel_token_sha256:
        die("shard token digest %s != sealed panel_token_sha256 %s -- "
            "this is NOT the registry panel"
            % (recomputed, args.expect_panel_token_sha256))

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
        "schema": "malaiwah.token-panel-rebuild-receipt.v1",
        "format_version": 1,
        "receipt_sha256": "",
        "tool": "k6/tools/build_panel_from_v5_suite.py",
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "panel_id": args.panel_id,
        "panel_name": args.panel_name,
        "suite_token_hash_sha256": aggregate,
        "derivation": "transport, not reconstruction",
        "selection_rule": (
            "every context of the sealed shard, in the shard suite-manifest's "
            "context_index order; window N carries context_index[N]'s token ids "
            "verbatim. No RNG, no tokenizer, no re-derivation from text."),
        "seals_verified": {
            "per_context_file_sha256": {
                "rule": "sha256(context-NNNN.json bytes) == context_index[i].token_sha256",
                "contexts_checked": len(windows),
                "mismatches": 0,
            },
            "shard_aggregate": {
                "rule": ("sha256(concat of per-context token_sha256 hex in "
                         "context_index order) == registry panel "
                         "identity.panel_token_sha256"),
                "expected": args.expect_panel_token_sha256,
                "recomputed": recomputed,
                "match": True,
            },
        },
        "parameters": {
            "context_length": context_length,
            "prediction_positions_per_window": context_length - 1,
            "windows_total": len(windows),
            "scored_positions_total": len(windows) * (context_length - 1),
        },
        "corpus": {
            "repository": args.corpus_repository,
            "revision": args.corpus_revision,
            "path_prefix": "suite/tokens/",
            "suite_manifest_sha256": sha256_file(args.suite_manifest),
            "suite_schema": manifest.get("schema"),
            "suite_token_sha256_parent": manifest.get("suite_token_sha256"),
        },
        "tokenizer": {
            "id": "qwen3.8",
            "repository": args.tokenizer_repository,
            "revision": args.tokenizer_revision,
            "vocab_size": int(manifest.get("vocab_size")),
            "add_special_tokens": False,
            "chat_template_applied": False,
            "note": ("no tokenizer was loaded by this tool; token ids are "
                     "transported from the sealed suite, so tokenizer identity "
                     "is inherited from the suite manifest, not re-asserted."),
        },
        "separation": {
            "checked": False,
            "method": "inherited from the parent suite manifest",
            "note": manifest.get("corpus_note"),
        },
    }
    receipt["receipt_sha256"] = seal(receipt)
    with open(os.path.join(args.out, "panel.receipt.json"), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "panel_id": args.panel_id,
        "out": args.out,
        "windows": len(windows),
        "context_length": context_length,
        "scored_positions_total": len(windows) * (context_length - 1),
        "suite_token_hash_sha256": aggregate,
        "shard_token_sha256_verified": recomputed,
        "panel_receipt_sha256": receipt["receipt_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
