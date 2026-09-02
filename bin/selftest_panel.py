#!/usr/bin/env python3
"""Focused selftest for fidelity.panel's immutable local-panel contract."""
from __future__ import annotations

import hashlib
import os
import json
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

BIN = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN))
from fidelity import panel as P  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print("  %s  %s%s" % ("PASS" if condition else "FAIL", name,
                           (" -- " + detail) if detail else ""))


def canonical(value, newline=False):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + ("\n" if newline else "")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def npy_i8(values):
    shape = (len(values),)
    header = repr({"descr": "<i8", "fortran_order": False, "shape": shape})
    pad = 16 - ((10 + len(header) + 1) % 16)
    encoded = (header + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + struct.pack(
        "<%dq" % len(values), *values)


def npy_u1(values):
    header = repr({"descr": "|u1", "fortran_order": False, "shape": (len(values),)})
    pad = 16 - ((10 + len(header) + 1) % 16)
    encoded = (header + " " * pad + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded + bytes(values)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tokenizer_files(root):
    root.mkdir(parents=True)
    values = {"tokenizer.json": b'{"version":"1"}\n',
              "tokenizer_config.json": b'{"model_max_length":16}\n',
              "chat_template.jinja": b"{{ messages }}\n"}
    for name, raw in values.items():
        (root / name).write_bytes(raw)
    return values


def panel_doc(root):
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    rows = []
    for index, values in enumerate(([1, 2, 3, 4], [5, 6, 7, 8])):
        window = "final-%04d" % index
        tokens, mask = npy_i8(values), npy_u1([1] * len(values))
        (arrays / (window + ".tokens.npy")).write_bytes(tokens)
        (arrays / (window + ".mask.npy")).write_bytes(mask)
        rows.append({"window_id": window, "role": "final", "num_tokens": len(values),
                     "prediction_positions": len(values) - 1,
                     "token_ids_sha256": sha(tokens), "attention_mask_sha256": sha(mask),
                     "token_ids_json_sha256": sha(canonical(values))})
    suite = sha("\n".join(row["token_ids_json_sha256"] for row in rows).encode("ascii"))
    doc = {"schema": P.PANEL_SCHEMA, "panel_id": "panel--selftest", "name": "selftest",
           "suite_token_hash_sha256": suite, "windows": rows}
    write_json(root / "panel.json", doc)
    return doc


def modern_fixture(root, token_root):
    doc = panel_doc(root)
    files = tokenizer_files(token_root)
    receipt = {"schema": P.BUILD_RECEIPT_SCHEMA, "receipt_sha256": "",
               "panel_id": doc["panel_id"], "panel_name": doc["name"],
               "suite_token_hash_sha256": doc["suite_token_hash_sha256"],
               "parameters": {"context_length": 4, "prediction_positions_per_window": 3,
                              "scored_positions_total": 6, "windows_total": 2},
               "tokenizer": {"repository": "example/tokenizer",
                             "revision": "a" * 40, "vocab_size": 100,
                             "files_sha256": {name: sha(raw) for name, raw in files.items()}}}
    receipt["receipt_sha256"] = sha(canonical(receipt))
    write_json(root / "panel.receipt.json", receipt)
    return receipt


def legacy_fixture(root, token_root):
    doc = panel_doc(root)
    values = tokenizer_files(token_root)
    identity_files = [{"path": "/producer/model/" + name, "bytes": len(raw), "sha256": sha(raw)}
                      for name, raw in sorted(values.items())]
    identity = {"class": "TokenizersBackend", "model_id": "example/tokenizer",
                "model_revision": "b" * 40, "files": identity_files}
    token_receipt = {"schema": P.TOKENIZER_RECEIPT_SCHEMA,
                     "tokenizer_identity": identity,
                     "tokenizer_identity_sha256": sha(canonical(identity, newline=True)),
                     "artifacts": identity_files, "vocab_size": 100,
                     "minimum_token_id": 0, "maximum_token_id_exclusive": 100}
    token_receipt["receipt_sha256"] = sha(canonical(token_receipt, newline=True))
    write_json(root / "tokenizer.receipt.json", token_receipt)
    artifacts = []
    for path in sorted([root / "panel.json"] + list((root / "arrays").iterdir())):
        raw = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        artifacts.append({"path": "/producer/panel/" + rel,
                          "bytes": len(raw), "sha256": sha(raw)})
    receipt = {"schema": P.ARTIFACT_RECEIPT_SCHEMA,
               "token_panel_artifact_sha256": sha((root / "panel.json").read_bytes()),
               "artifacts": artifacts, "final_windows": 2, "final_prediction_positions": 6,
               "tokenizer_receipt_sha256": token_receipt["receipt_sha256"]}
    receipt["receipt_sha256"] = sha(canonical(receipt, newline=True))
    write_json(root / "panel.receipt.json", receipt)
    return receipt, token_receipt


def refused(root, token_root):
    try:
        P.resolve_panel(root, tokenizer_root=token_root)
    except P.PanelError:
        return True
    return False

def target_refused(binding, repo, revision):
    try:
        P.validate_root_panel_binding(binding, repo, revision)
    except P.PanelError:
        return True
    return False

def replace_first_mask(root, values):
    doc = json.loads((root / "panel.json").read_text())
    raw = npy_u1(values)
    window = doc["windows"][0]
    (root / "arrays" / (window["window_id"] + ".mask.npy")).write_bytes(raw)
    window["attention_mask_sha256"] = sha(raw)
    write_json(root / "panel.json", doc)


def main():
    work = Path(tempfile.mkdtemp(prefix="panel-contract-selftest-"))
    try:
        modern, modern_tokens = work / "modern", work / "modern-tokenizer"
        modern.mkdir()
        modern_receipt = modern_fixture(modern, modern_tokens)
        resolved = P.resolve_panel(modern, tokenizer_root=modern_tokens)
        binding = resolved.to_dict()
        check("modern self-blank receipt resolves", binding["receipt"]["receipt_seal_mode"] == "self-blank")
        check("declared receipt identity is distinct from raw receipt-file SHA",
              binding["receipt"]["declared_receipt_sha256"] == modern_receipt["receipt_sha256"]
              and binding["receipt"]["receipt_file_sha256"] != modern_receipt["receipt_sha256"])
        check("role/shape/token aggregate is bound", binding["panel"]["contexts"] == 2
              and binding["panel"]["context_length"] == 4
              and binding["panel"]["scored_positions_total"] == 6)
        check("tokenizer repository/revision/files/vocab are verified",
              binding["tokenizer"]["id"] == "example/tokenizer"
              and binding["tokenizer"]["files_verified"] is True
              and len(binding["tokenizer"]["files"]) == 3
              and binding["tokenizer"]["vocab_size"] == 100)

        legacy, legacy_tokens = work / "legacy", work / "legacy-tokenizer"
        legacy.mkdir()
        legacy_receipt, _ = legacy_fixture(legacy, legacy_tokens)
        legacy_binding = P.resolve_panel(legacy, tokenizer_root=legacy_tokens).to_dict()
        check("legacy field-absent canonical receipt resolves",
              legacy_binding["receipt"]["receipt_seal_mode"] == "legacy-field-absent"
              and legacy_binding["receipt"]["declared_receipt_sha256"]
              == legacy_receipt["receipt_sha256"])

        fruit_root = (BIN.parent / "engines" / "panels"
                      / "panel--fruit.malaiwah.heldout-v1")
        fruit_binding = P.resolve_panel(fruit_root).to_dict()
        check("committed Fruit panel resolves all-ones attention masks as shifted pairs",
              fruit_binding["panel"]["contexts"] == 16
              and fruit_binding["panel"]["context_length"] == 2048
              and fruit_binding["panel"]["positions_per_context"] == 2047)
        fruit_repo = "malaiwah/GLM-5.2-SIQ-Fruit-bf16"
        fruit_revision = "ef68013aa6e16453cf52b5b77647f72fbe258c3c"
        check("unprefetched Fruit tokenizer closure cannot authorize spend",
              fruit_binding["tokenizer"]["files_verified"] is False
              and target_refused(fruit_binding, fruit_repo, fruit_revision))
        check("Fruit16 panel swap cannot authorize M2 spend",
              target_refused(
                  fruit_binding, P.M2_TARGET_REPO, P.M2_TARGET_REVISION))
        check("arbitrary valid sealed panel cannot authorize any paid root",
              target_refused(binding, fruit_repo, fruit_revision)
              and target_refused(
                  binding, P.M2_TARGET_REPO, P.M2_TARGET_REVISION)
              and target_refused(
                  binding, P.GLM53_TARGET_REPO, P.GLM53_TARGET_REVISION))
        mutated_fruit = json.loads(json.dumps(fruit_binding))
        mutated_fruit["panel"]["id"] = "panel--swapped"
        check("mutated Fruit panel identity refuses before spend",
              target_refused(mutated_fruit, fruit_repo, fruit_revision))

        glm53_root = (
            BIN.parent / "engines" / "panels" /
            "panel--glm53.malaiwah.corpus5x5-v1")
        glm53_binding = P.resolve_panel(glm53_root).to_dict()
        check("committed full GLM53 panel resolves exact corpus5x5 geometry",
              glm53_binding["panel"]["contexts"] == 25
              and glm53_binding["panel"]["context_length"] == 2048
              and glm53_binding["panel"]["scored_positions_total"] == 51175)
        check("unprefetched full GLM53 tokenizer cannot authorize spend",
              glm53_binding["tokenizer"]["files_verified"] is False
              and target_refused(
                  glm53_binding, P.GLM53_TARGET_REPO,
                  P.GLM53_TARGET_REVISION))
        verified_glm53 = json.loads(json.dumps(glm53_binding))
        expected_files = {
            name: (size, digest)
            for name, size, digest in P.GLM53_TOKENIZER_FILES}
        for row in verified_glm53["tokenizer"]["files"]:
            row["bytes"] = expected_files[row["name"]][0]
        verified_glm53["tokenizer"]["files_verified"] = True
        check("exact verified full GLM53 panel authorizes only its target pin",
              P.validate_root_panel_binding(
                  verified_glm53, P.GLM53_TARGET_REPO,
                  P.GLM53_TARGET_REVISION) == verified_glm53
              and target_refused(
                  verified_glm53, fruit_repo, fruit_revision))
        mutated_glm53 = json.loads(json.dumps(verified_glm53))
        mutated_glm53["content"]["manifest"][0]["bytes"] += 1
        check("mutated full GLM53 panel content refuses before spend",
              target_refused(
                  mutated_glm53, P.GLM53_TARGET_REPO,
                  P.GLM53_TARGET_REVISION))

        brandon_root = Path(os.environ.get(
            "QFS_BRANDON_PANEL_ROOT", "/tmp/qfs-brandon-panel/calibration/panel-v1"))
        if brandon_root.is_dir():
            brandon_closure = work / "brandon-closure"
            brandon_receipt = json.loads((brandon_root / "panel.receipt.json").read_text())
            panel_artifact = next(row for row in brandon_receipt["artifacts"]
                                  if row["path"].endswith("/panel.json"))
            producer_root = Path(panel_artifact["path"]).parent
            for artifact in brandon_receipt["artifacts"]:
                rel = Path(artifact["path"]).relative_to(producer_root)
                (brandon_closure / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(brandon_root / rel, brandon_closure / rel)
            shutil.copy2(brandon_root / "panel.receipt.json",
                         brandon_closure / "panel.receipt.json")
            shutil.copy2(brandon_root / "tokenizer.receipt.json",
                         brandon_closure / "tokenizer.receipt.json")
            brandon_binding = P.resolve_panel(brandon_closure).to_dict()
            check("real Brandon sealed closure resolves shifted next-token mask geometry",
                  brandon_binding["panel"]["contexts"] == 25
                  and brandon_binding["panel"]["context_length"] == 2048
                  and brandon_binding["panel"]["positions_per_context"] == 2047)
            check("unprefetched Brandon tokenizer closure cannot authorize M2 spend",
                  brandon_binding["tokenizer"]["files_verified"] is False
                  and target_refused(
                      brandon_binding, P.M2_TARGET_REPO,
                      P.M2_TARGET_REVISION))
            check("Brandon final25 panel swap cannot authorize Fruit spend",
                  target_refused(brandon_binding, fruit_repo, fruit_revision))
            mutated_brandon = json.loads(json.dumps(brandon_binding))
            mutated_brandon["receipt"]["receipt_file_sha256"] = "0" * 64
            check("mutated Brandon receipt identity refuses before M2 spend",
                  target_refused(
                      mutated_brandon, P.M2_TARGET_REPO,
                      P.M2_TARGET_REVISION))
        else:
            check("real Brandon sealed closure resolves shifted next-token mask geometry",
                  True, "SKIPPED: set QFS_BRANDON_PANEL_ROOT to the downloaded panel")

        duplicate_window_key = work / "duplicate-window-key"
        shutil.copytree(modern, duplicate_window_key)
        duplicate_window_text = (duplicate_window_key / "panel.json").read_text()
        duplicate_window_text = duplicate_window_text.replace(
            '      "role": "final",',
            '      "role": "final",\\n      "role": "final",', 1)
        (duplicate_window_key / "panel.json").write_text(duplicate_window_text)
        check("duplicate panel window field refuses before last-wins parsing",
              refused(duplicate_window_key, modern_tokens))

        duplicate_tokenizer_field = work / "duplicate-tokenizer-field"
        shutil.copytree(legacy, duplicate_tokenizer_field)
        duplicate_tokenizer_text = (
            duplicate_tokenizer_field / "tokenizer.receipt.json").read_text()
        tokenizer_needle = '  "vocab_size": 100'
        duplicate_tokenizer_text = duplicate_tokenizer_text.replace(
            tokenizer_needle, tokenizer_needle + ',\\n' + tokenizer_needle, 1)
        (duplicate_tokenizer_field / "tokenizer.receipt.json").write_text(
            duplicate_tokenizer_text)
        check("duplicate tokenizer receipt field refuses before last-wins parsing",
              refused(duplicate_tokenizer_field, legacy_tokens))

        duplicate_seal_field = work / "duplicate-receipt-seal-field"
        shutil.copytree(modern, duplicate_seal_field)
        duplicate_seal_text = (duplicate_seal_field / "panel.receipt.json").read_text()
        seal_needle = '  "receipt_sha256": "%s",' % modern_receipt["receipt_sha256"]
        duplicate_seal_text = duplicate_seal_text.replace(
            seal_needle, seal_needle + '\\n' + seal_needle, 1)
        (duplicate_seal_field / "panel.receipt.json").write_text(duplicate_seal_text)
        check("duplicate receipt seal field refuses before last-wins parsing",
              refused(duplicate_seal_field, modern_tokens))

        extra_file = work / "extra-panel-file"
        shutil.copytree(modern, extra_file)
        (extra_file / ".env").write_text("HF_TOKEN=must-not-upload\\n")
        check("unlisted extra panel-root file refuses instead of entering archive",
              refused(extra_file, modern_tokens))

        archive_one = work / "panel-one.tar"
        archive_two = work / "panel-two.tar"
        written_one = P.write_panel_archive(
            legacy, archive_one, tokenizer_root=legacy_tokens)
        written_two = P.write_panel_archive(
            legacy, archive_two, tokenizer_root=legacy_tokens)
        archive_raw = archive_one.read_bytes()
        check("archive writer emits the exact bytes named by the resolved binding",
              written_one["bytes"] == len(archive_raw)
              and written_one["sha256"] == sha(archive_raw)
              and written_one["sha256"]
              == legacy_binding["content"]["archive"]["sha256"]
              and written_one["binding"] == legacy_binding
              and written_one["binding_sha256"] == sha(canonical(legacy_binding)))
        check("two atomic archive writes are byte-for-byte deterministic",
              archive_one.read_bytes() == archive_two.read_bytes()
              and not list(work.glob(".*.tmp")))
        with tarfile.open(archive_one, "r:") as archive:
            archived_names = archive.getnames()
        manifest_names = [row["path"] for row in legacy_binding["content"]["manifest"]]
        check("archive contains only the sorted panel manifest, never tokenizer-root files",
              archived_names == manifest_names
              and "tokenizer.json" not in archived_names
              and "tokenizer_config.json" not in archived_names
              and "chat_template.jinja" not in archived_names)

        artifact_bad = work / "artifact-bad"
        shutil.copytree(legacy, artifact_bad)
        target = artifact_bad / "arrays" / "final-0000.tokens.npy"
        target.write_bytes(target.read_bytes() + b"tamper")
        check("listed artifact byte/size tamper refuses", refused(artifact_bad, legacy_tokens))

        tokenizer_bad = work / "tokenizer-bad"
        shutil.copytree(legacy_tokens, tokenizer_bad)
        (tokenizer_bad / "tokenizer.json").write_bytes(b"tamper")
        check("tokenizer artifact tamper refuses", refused(legacy, tokenizer_bad))

        receipt_bad = work / "receipt-bad"
        shutil.copytree(modern, receipt_bad)
        receipt_doc = json.loads((receipt_bad / "panel.receipt.json").read_text())
        receipt_doc["tokenizer"]["revision"] = "c" * 40
        write_json(receipt_bad / "panel.receipt.json", receipt_doc)
        check("tokenizer identity tamper breaks the receipt seal", refused(receipt_bad, modern_tokens))

        mutable_revision = work / "mutable-tokenizer-revision"
        shutil.copytree(modern, mutable_revision)
        mutable_doc = json.loads((mutable_revision / "panel.receipt.json").read_text())
        mutable_doc["tokenizer"]["revision"] = "main"
        mutable_doc["receipt_sha256"] = ""
        mutable_doc["receipt_sha256"] = sha(canonical(mutable_doc))
        write_json(mutable_revision / "panel.receipt.json", mutable_doc)
        check("mutable tokenizer revision refuses despite a valid receipt seal",
              refused(mutable_revision, modern_tokens))

        malformed_repository = work / "malformed-tokenizer-repository"
        shutil.copytree(modern, malformed_repository)
        malformed_doc = json.loads((malformed_repository / "panel.receipt.json").read_text())
        malformed_doc["tokenizer"]["repository"] = "ownerless-tokenizer"
        malformed_doc["receipt_sha256"] = ""
        malformed_doc["receipt_sha256"] = sha(canonical(malformed_doc))
        write_json(malformed_repository / "panel.receipt.json", malformed_doc)
        check("tokenizer repository without owner/name shape refuses",
              refused(malformed_repository, modern_tokens))

        alias_tokenizer_path = work / "alias-tokenizer-path"
        shutil.copytree(modern, alias_tokenizer_path)
        alias_doc = json.loads((alias_tokenizer_path / "panel.receipt.json").read_text())
        tokenizer_digest = alias_doc["tokenizer"]["files_sha256"].pop("tokenizer.json")
        alias_doc["tokenizer"]["files_sha256"]["./tokenizer.json"] = tokenizer_digest
        alias_doc["receipt_sha256"] = ""
        alias_doc["receipt_sha256"] = sha(canonical(alias_doc))
        write_json(alias_tokenizer_path / "panel.receipt.json", alias_doc)
        check("non-canonical tokenizer path alias refuses",
              refused(alias_tokenizer_path, modern_tokens))

        duplicate_tokenizer_path = work / "duplicate-tokenizer-path"
        shutil.copytree(modern, duplicate_tokenizer_path)
        duplicate_doc = json.loads(
            (duplicate_tokenizer_path / "panel.receipt.json").read_text())
        duplicate_doc["tokenizer"]["files_sha256"]["nested/tokenizer.json"] = (
            duplicate_doc["tokenizer"]["files_sha256"]["tokenizer.json"])
        duplicate_doc["receipt_sha256"] = ""
        duplicate_doc["receipt_sha256"] = sha(canonical(duplicate_doc))
        write_json(duplicate_tokenizer_path / "panel.receipt.json", duplicate_doc)
        check("duplicate canonical tokenizer basename refuses",
              refused(duplicate_tokenizer_path, modern_tokens))

        alias_panel_path = work / "alias-panel-artifact-path"
        shutil.copytree(legacy, alias_panel_path)
        alias_panel_doc = json.loads((alias_panel_path / "panel.receipt.json").read_text())
        array_row = next(row for row in alias_panel_doc["artifacts"]
                         if "/arrays/" in row["path"])
        array_row["path"] = array_row["path"].replace("/arrays/", "/arrays//", 1)
        alias_panel_doc.pop("receipt_sha256")
        alias_panel_doc["receipt_sha256"] = sha(canonical(alias_panel_doc, newline=True))
        write_json(alias_panel_path / "panel.receipt.json", alias_panel_doc)
        check("non-canonical panel artifact path alias refuses",
              refused(alias_panel_path, legacy_tokens))

        duplicate_panel_path = work / "duplicate-panel-artifact-path"
        shutil.copytree(legacy, duplicate_panel_path)
        duplicate_panel_doc = json.loads(
            (duplicate_panel_path / "panel.receipt.json").read_text())
        duplicate_panel_doc["artifacts"].append(dict(duplicate_panel_doc["artifacts"][1]))
        duplicate_panel_doc.pop("receipt_sha256")
        duplicate_panel_doc["receipt_sha256"] = sha(
            canonical(duplicate_panel_doc, newline=True))
        write_json(duplicate_panel_path / "panel.receipt.json", duplicate_panel_doc)
        check("duplicate canonical panel artifact path refuses",
              refused(duplicate_panel_path, legacy_tokens))

        mask_value_bad = work / "mask-value-bad"
        shutil.copytree(modern, mask_value_bad)
        replace_first_mask(mask_value_bad, [0, 1, 2, 0])
        check("non-binary mask value refuses even with matching artifact digest",
              refused(mask_value_bad, modern_tokens))

        mask_count_bad = work / "mask-count-bad"
        shutil.copytree(modern, mask_count_bad)
        replace_first_mask(mask_count_bad, [1, 1, 0, 1])
        check("mask shifted next-token validity disagreeing with prediction_positions refuses",
              refused(mask_count_bad, modern_tokens))

        script = ("import json,sys;sys.path.insert(0,%r);"
                  "from fidelity.panel import resolve_panel;"
                  "print(json.dumps(resolve_panel(%r,tokenizer_root=%r).to_dict(),"
                  "sort_keys=True,separators=(',',':')))" %
                  (str(BIN), str(legacy), str(legacy_tokens)))
        first = subprocess.run([sys.executable, "-c", script], check=True,
                               capture_output=True, text=True).stdout
        second = subprocess.run([sys.executable, "-c", script], check=True,
                                capture_output=True, text=True).stdout
        check("fresh-process manifest/archive evidence is exactly equal", first == second
              and json.loads(first)["content"]["archive"]["sha256"]
              == legacy_binding["content"]["archive"]["sha256"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
