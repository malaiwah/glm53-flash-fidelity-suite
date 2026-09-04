"""Re-seal a verified dataset whose ONLY publication defect is a private path
inside the validator's own sealed verdict.

Why this exists
---------------
Every dataset sealed before 2026-09-04 carries the absolute output directory
as the `subject` of `validation/structural-validation.json` -- on a pod,
`/workspace/fidelity/<job>/<attempt>/dataset`.  The member is inside the seal
(spec section 3 requires it) and the publisher refuses any private absolute
path, so such a dataset is scientifically valid, fully verifiable, and
unpublishable.  The GLM-5.3 root capture salvaged from the 2026-09-03 H200
run (capture_content_digest 02963bc5...) is exactly that case, and re-running
it costs a rented H200 cold run.

What a reseal is, and is not
----------------------------
A reseal rewrites exactly one string in exactly one member, appends a
disclosure and a `dataset.resealed` block to the manifest, writes a receipt
INSIDE the tree, and re-runs `finalize` (checksums.txt + the self-blanked
seal).  Every tensor byte, the capture manifest, the runtime manifest, the
panel and the head are untouched, so `capture.capture_content_digest` --
the scientific identity -- is unchanged, and the tool refuses if it is not.

`dataset_sha256` DOES change: it is the tree's identity, and the tree
changed.  The block and the receipt name the original seal, the original
member digest and the digest of the string that was replaced (never the
string itself: it is the private path).  Anyone holding the original tree
can confirm the correspondence member by member.

The tool refuses anything broader: a source that does not verify, a source
already re-sealed, a source that is already publishable, or any offending
text outside that one field.  Fixing scientific content is not a reseal.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from . import common
from . import dsformat as F
from . import dsmanifest
from . import dshub
from . import dsvalidate

RESEAL_SCHEMA = "fidelity-dataset.reseal.v1"
RESEAL_RECEIPT_SCHEMA = "fidelity-dataset.reseal-receipt.v1"
RESEAL_RECEIPT_NAME = "validation/reseal-receipt.json"
VALIDATION_RECORD = "validation/structural-validation.json"
REASON = "validation_subject_private_path"
DISCLOSURE_CODE = "resealed_validation_subject"


class ResealError(RuntimeError):
    """Refusal: the source is not the one narrow case a reseal may repair."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _textual_members_with_private_paths(root: str) -> List[Tuple[str, List[str]]]:
    """Every sealed text member the publisher would refuse, with the patterns hit."""
    offenders = []
    for relpath in F.iter_dataset_files(root):
        if not dshub._textual_publish_member(relpath):
            continue
        with open(os.path.join(root, relpath), "rb") as handle:
            body = handle.read()
        hits = [pattern.decode("ascii", "replace")
                for pattern in dshub._PRIVATE_ABSOLUTE_PATHS if pattern in body]
        if hits:
            offenders.append((relpath, hits))
    return offenders


def _verified(root: str, label: str) -> Dict[str, Any]:
    report = dsvalidate.validate_dataset(root, verify_tensors=True)
    if not report.passed:
        first = report.errors[0]
        raise ResealError("%s does not verify: %s (%s)"
                          % (label, first["message"], first["rule"]))
    manifest = F.load_manifest(root)
    if (manifest.get("dataset") or {}).get("structural_status") != "sealed":
        raise ResealError("%s is not sealed" % label)
    return manifest


def _record_without_private_subject(record: Dict[str, Any], dataset_id: str
                                    ) -> Tuple[Dict[str, Any], str]:
    """The same verdict with the dataset's own identity as its subject."""
    if record.get("schema") != F.VALIDATION_SCHEMA:
        raise ResealError("validation record schema is %r" % record.get("schema"))
    if not common.verify_seal(record, "receipt_sha256"):
        raise ResealError("validation record seal does not verify")
    subject = record.get("subject")
    if not isinstance(subject, str) or not subject:
        raise ResealError("validation record has no subject")
    if not any(pattern.decode("ascii", "replace") in subject
               for pattern in dshub._PRIVATE_ABSOLUTE_PATHS):
        raise ResealError("validation record subject %r is not a private path; "
                          "nothing to reseal" % subject)
    rewritten = dict(record)
    rewritten["subject"] = "dataset:%s" % dataset_id
    rewritten["receipt_sha256"] = ""
    return F.seal_receipt(rewritten), subject


def reseal_dataset(source: str, out: str, *, tool_path: Optional[str] = None
                   ) -> Dict[str, Any]:
    """Copy `source` to `out` with the validation subject repaired; return the receipt.

    Refuses (ResealError) unless the source verifies with tensors recomputed,
    has never been re-sealed, and its only publisher-refused text is the
    validation record's subject.  `out` must not exist.
    """
    source = os.path.abspath(source)
    out = os.path.abspath(out)
    if os.path.exists(out):
        raise ResealError("output %s already exists" % out)
    manifest = _verified(source, "source dataset")
    dataset = manifest["dataset"]
    if dataset.get("resealed") is not None:
        raise ResealError("source was already re-sealed from %s; a reseal is not "
                          "repeatable" % dataset["resealed"].get("from_dataset_sha256"))
    offenders = _textual_members_with_private_paths(source)
    if [relpath for relpath, _ in offenders] != [VALIDATION_RECORD]:
        raise ResealError(
            "a reseal repairs exactly %s; the source's publisher-refused members "
            "are %r" % (VALIDATION_RECORD, [relpath for relpath, _ in offenders]))
    record_path = os.path.join(source, VALIDATION_RECORD)
    original_record = F.read_json(record_path)
    rewritten_record, original_subject = _record_without_private_subject(
        original_record, dataset["id"])
    # The subject must be the ONLY private-path text in that member.
    probe_bytes = F.canonical_json(rewritten_record).encode("utf-8")
    if any(pattern in probe_bytes for pattern in dshub._PRIVATE_ABSOLUTE_PATHS):
        raise ResealError("the validation record carries a private path outside "
                          "its subject; a reseal does not repair that")

    tool_path = tool_path or os.path.abspath(__file__)
    from_checksums = F.file_sha256(os.path.join(source, F.CHECKSUMS_NAME))
    original_member_sha = F.file_sha256(record_path)
    capture_digest = manifest["capture"]["capture_content_digest"]

    shutil.copytree(source, out, symlinks=False)
    try:
        F.write_json(os.path.join(out, VALIDATION_RECORD), rewritten_record)
        resealed_member_sha = F.file_sha256(os.path.join(out, VALIDATION_RECORD))
        receipt = F.seal_receipt({
            "schema": RESEAL_RECEIPT_SCHEMA,
            "receipt_sha256": "",
            "resealed_utc": common.utcnow(),
            "reason": REASON,
            "dataset_id": dataset["id"],
            "from_dataset_sha256": manifest[F.SEAL_FIELD],
            "from_checksums_sha256": from_checksums,
            "from_created_utc": manifest.get("created_utc"),
            "capture_content_digest": capture_digest,
            "capture_manifest_file_sha256": manifest["capture"]["manifest_file_sha256"],
            "members_rewritten": [{
                "path": VALIDATION_RECORD,
                "field": "subject",
                "original_sha256": original_member_sha,
                "resealed_sha256": resealed_member_sha,
                "original_value_sha256": _sha256_text(original_subject),
                "resealed_value": rewritten_record["subject"],
            }],
            "members_added": [RESEAL_RECEIPT_NAME],
            "tool": {
                "file": "bin/fidelity/dsreseal.py",
                "sha256": F.file_sha256(tool_path),
            },
        })
        F.write_json(os.path.join(out, RESEAL_RECEIPT_NAME), receipt)
        receipt_file_sha = F.file_sha256(os.path.join(out, RESEAL_RECEIPT_NAME))

        new_manifest = dict(manifest)
        new_dataset = dict(dataset)
        new_dataset["resealed"] = {
            "schema": RESEAL_SCHEMA,
            "reason": REASON,
            "from_dataset_sha256": manifest[F.SEAL_FIELD],
            "resealed_utc": receipt["resealed_utc"],
            "receipt": RESEAL_RECEIPT_NAME,
            "receipt_sha256": receipt_file_sha,
            "members_rewritten": [VALIDATION_RECORD],
        }
        new_manifest["dataset"] = new_dataset
        new_manifest["disclosures"] = list(manifest["disclosures"]) + [{
            "code": DISCLOSURE_CODE,
            "severity": "info",
            "affects_comparability": False,
            "detail": (
                "re-sealed from dataset_sha256 %s: the validator's sealed verdict "
                "(%s) named the capture's output directory, a private absolute path "
                "the publisher refuses; its subject is now 'dataset:%s'. No tensor, "
                "capture, runtime, panel or head member changed; "
                "capture_content_digest %s is unchanged. Receipt: %s."
                % (manifest[F.SEAL_FIELD], VALIDATION_RECORD, dataset["id"],
                   capture_digest, RESEAL_RECEIPT_NAME)),
        }]
        _append_card_note(out, new_dataset["resealed"], capture_digest)
        sealed = dsmanifest.finalize(
            out, new_manifest,
            (manifest.get("seal") or {}).get("external_anchor"))
        verified = _verified(out, "re-sealed dataset")
        if verified["capture"]["capture_content_digest"] != capture_digest:
            raise ResealError("capture_content_digest changed across the reseal")
        if verified[F.SEAL_FIELD] != sealed[F.SEAL_FIELD]:
            raise ResealError("re-sealed manifest does not match the tree")
        remaining = _textual_members_with_private_paths(out)
        if remaining:
            raise ResealError("re-sealed tree still carries private paths in %r"
                              % [relpath for relpath, _ in remaining])
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    outer = dict(receipt)
    outer["resealed_dataset_sha256"] = sealed[F.SEAL_FIELD]
    outer["resealed_checksums_sha256"] = sealed["seal"]["checksums_sha256"]
    outer["output"] = None  # never a host path
    return outer


def _append_card_note(root: str, resealed: Dict[str, Any], capture_digest: str) -> None:
    """The card is a rendered view of the manifest; keep it honest without
    re-rendering it (the renderer needs the capture's own arguments)."""
    path = os.path.join(root, "README.md")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        body = handle.read()
    note = (
        "\n\n## Re-sealed\n\n"
        "This tree was re-sealed from `dataset_sha256` `%s` on %s: the validator's "
        "sealed verdict (`%s`) named the capture's output directory, a private "
        "absolute path the publisher refuses. Only that one field changed. Every "
        "tensor, the capture, runtime, panel and head members are byte-identical "
        "to the original; `capture_content_digest` `%s` is unchanged. The receipt "
        "is `%s` (`%s`).\n"
        % (resealed["from_dataset_sha256"], resealed["resealed_utc"],
           VALIDATION_RECORD, capture_digest, RESEAL_RECEIPT_NAME,
           resealed["receipt_sha256"]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body.rstrip("\n") + note)
