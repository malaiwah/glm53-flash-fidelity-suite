"""Structural + seal validator for `malaiwah.fidelity-dataset.v1`.

Two layers, on purpose:

  1. JSON Schema, checked with the registry's own vendored `_minischema` so the
     dataset validates on a stock py3.9 interpreter with no pip install -- the
     same guarantee `registry/Makefile` makes.
  2. The rules a JSON Schema cannot express: the seal chain, the digest
     recomputations, the panel <-> capture binding, coverage honesty, the head
     trap, and the determinism-evidence rules.

Every rule is reported with the spec id it enforces, so a refusal is traceable
to a sentence in docs/FIDELITY-DATASET-SPEC.md rather than to an opinion.

`validate()` reports EVERY failure; `verify()` (bin/fidelity-dataset verify)
stops at the first.  There is no `--force`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import dsformat as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
SCHEMA_DIR = os.path.join(_REPO, "docs", "schema")
_REGISTRY_TOOLS = os.path.join(_REPO, "registry", "tools")


def _minischema():
    if _REGISTRY_TOOLS not in sys.path:
        sys.path.insert(0, _REGISTRY_TOOLS)
    import _minischema  # noqa: WPS433 -- the registry's own vendored validator

    return _minischema


class Report(object):
    """Accumulates findings.  A finding is (code, rule, message, where)."""

    def __init__(self, subject: str):
        self.subject = subject
        self.errors: List[Dict[str, str]] = []
        self.warnings: List[Dict[str, str]] = []
        self.checks: List[str] = []

    def error(self, code: str, rule: str, message: str, where: str = "") -> None:
        self.errors.append({"code": code, "rule": rule, "message": message, "where": where})

    def warn(self, code: str, rule: str, message: str, where: str = "") -> None:
        self.warnings.append({"code": code, "rule": rule, "message": message, "where": where})

    def ok(self, name: str) -> None:
        self.checks.append(name)

    @property
    def passed(self) -> bool:
        return not self.errors

    def first_error_code(self) -> Optional[str]:
        return self.errors[0]["code"] if self.errors else None

    def to_dict(self, **extra: Any) -> Dict[str, Any]:
        doc = {
            "schema": F.VALIDATION_SCHEMA,
            "format_version": F.FORMAT_VERSION,
            "receipt_sha256": "",
            "subject": self.subject,
            "structural_status": "sealed" if self.passed else "invalid",
            "checks_run": sorted(self.checks),
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }
        doc.update(extra)
        return F.seal_receipt(doc)


# ---------------------------------------------------------------------------
# JSON Schema layer
# ---------------------------------------------------------------------------


def schema_errors(instance: Any, schema_file: str) -> List[str]:
    mini = _minischema()
    registry = mini.Registry(SCHEMA_DIR)
    return [str(e) for e in registry.validate(instance, schema_file)]


def validate_manifest_schema(manifest: Dict[str, Any], report: Report) -> None:
    for message in schema_errors(manifest, "fidelity-dataset.schema.json"):
        report.error("schema_invalid", "SCHEMA", message, F.MANIFEST_NAME)
    report.ok("schema")


def validate_receipt(receipt: Dict[str, Any], subject: str = "receipt") -> Report:
    """Validate a comparison receipt: schema + the SC-1/SC-2/BIAS-001 rules."""
    report = Report(subject)
    if receipt.get("schema") != F.RECEIPT_SCHEMA:
        report.error("bad_schema", "SCHEMA",
                     "schema is %r; expected %r" % (receipt.get("schema"), F.RECEIPT_SCHEMA))
        return report
    for message in schema_errors(receipt, "fidelity-comparison-receipt.schema.json"):
        report.error("schema_invalid", "SCHEMA", message)
    if F.recompute_seal(receipt, "receipt_sha256") != receipt.get("receipt_sha256"):
        report.error("seal_failed", "SEAL-1", "receipt_sha256 does not recompute")
    report.ok("schema")
    report.ok("seal")

    kind = receipt.get("comparison_kind")
    metric = receipt.get("metric") or {}
    sc = receipt.get("self_compare") or {}
    if kind == "reproduction_confirmation":
        # SC-1: exactly zero, not epsilon.
        for field, value in (("metric.value", metric.get("value")),
                             ("top1_agreement", receipt.get("top1_agreement"))):
            expect = 0.0 if field == "metric.value" else 1.0
            if value != expect:
                report.error("self_compare_invalid", "SC-1",
                             "%s is %r; a reproduction confirmation asserts exactly %r"
                             % (field, value, expect))
        for key, value in sorted((receipt.get("kl") or {}).items()):
            if value != 0.0:
                report.error("self_compare_invalid", "SC-1", "kl.%s is %r, not 0.0" % (key, value))
            elif str(value)[0] == "-":
                report.error("self_compare_invalid", "SC-1", "kl.%s is -0.0" % key)
        if not sc.get("capture_content_digest_equal"):
            report.error("self_compare_invalid", "SC-1",
                         "reproduction_confirmation without capture_content_digest_equal")
    if kind == "run_to_run_floor" and sc.get("capture_content_digest_equal"):
        report.error("self_compare_invalid", "SC-2",
                     "run_to_run_floor with equal capture digests is a reproduction, not a floor")
    if kind != "measurement" and (receipt.get("submission") or {}).get("emitted"):
        report.error("not_submittable", "SC-3",
                     "only comparison_kind == 'measurement' may be handed to registry_add")
    comparability = receipt.get("comparability") or {}
    if comparability.get("same_lane") is False and comparability.get("usable_as_floor"):
        report.error("bias_invalid", "BIAS-006",
                     "a cross-lane comparison may never be cited as a floor")
    if (receipt.get("estimator") or {}).get("stack_relation") == "cross_stack" \
            and not comparability.get("bias"):
        report.error("bias_invalid", "BIAS-001", "cross_stack requires a bias block")
    report.ok("self_compare")
    report.ok("bias")
    return report


# ---------------------------------------------------------------------------
# The dataset rules
# ---------------------------------------------------------------------------


def _load_sub(root: str, relpath: str, report: Report, rule: str) -> Optional[Dict[str, Any]]:
    try:
        F.check_relpath(relpath, owner=rule)
    except F.FormatError as exc:
        report.error(exc.code, rule, exc.message, relpath)
        return None
    full = os.path.join(root, relpath)
    if not os.path.isfile(full):
        report.error("missing_file", rule, "%s is missing" % relpath, relpath)
        return None
    try:
        doc = F.read_json(full)
    except ValueError as exc:
        report.error("bad_schema", rule, "unparseable JSON: %s" % exc, relpath)
        return None
    if not isinstance(doc, dict):
        report.error("bad_schema", rule, "not an object", relpath)
        return None
    return doc


def _check_sub_seal(doc: Dict[str, Any], relpath: str, report: Report) -> None:
    claimed = doc.get("receipt_sha256")
    if claimed is None:
        report.error("seal_failed", "SEAL-1(g)", "no receipt_sha256", relpath)
        return
    if F.recompute_seal(doc, "receipt_sha256") != claimed:
        report.error("seal_failed", "SEAL-1(g)", "receipt_sha256 does not recompute", relpath)


PATH_KEYS = ("file", "token_file", "attention_mask_file", "panel_file",
             "manifest_file", "checksums_file", "head_json", "dir",
             "remap_file", "runtime_manifest", "receipt")

#: PATH-3 permits `..` inside compat/.  `runtime_manifest` is the one field
#: outside compat/ that carries it, because it is adopted VERBATIM from
#: kimi-k3, whose `reference-hidden/manifest.json` names
#: `../capture-runtime.json` and whose validator refuses only an ABSOLUTE path.
#: Spec section 6.1 shows exactly that value.  Containment inside the dataset
#: root is still enforced, which is what PATH-3 is actually protecting.
PARENT_ALLOWED_KEYS = ("runtime_manifest",)


def _walk_paths(node: Any, report: Report, where: str, root: str = "") -> None:
    """PATH-1/PATH-3 over every string that looks like a dataset path.

    A path is resolved relative to the directory of the file that carries it,
    then proven to stay inside the dataset root.
    """
    base = os.path.dirname(where)
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and key in PATH_KEYS:
                if value.startswith("http://") or value.startswith("https://"):
                    continue
                allow_parent = where.startswith("compat/") or key in PARENT_ALLOWED_KEYS
                try:
                    F.check_relpath(value, owner=where, allow_parent=allow_parent)
                    joined = os.path.normpath(os.path.join(base, value))
                    if joined.startswith(".."):
                        raise F.FormatError(
                            "path_escape",
                            "%r resolves outside the dataset root" % value, where)
                except F.FormatError as exc:
                    report.error(exc.code, "PATH-1/PATH-3", exc.message, where + ":" + key)
            _walk_paths(value, report, where, root)
    elif isinstance(node, list):
        for item in node:
            _walk_paths(item, report, where, root)


def validate_dataset(
    root: str,
    verify_tensors: bool = False,
    allow_partial: bool = False,
    manifest_only: bool = False,
    strict: bool = False,
) -> Report:
    report = Report(os.path.abspath(root))

    # 1-2. manifest parses, schema exact, JSON Schema clean.
    try:
        manifest = F.load_manifest(root)
    except F.FormatError as exc:
        report.error(exc.code, "SCHEMA", exc.message, F.MANIFEST_NAME)
        return report
    validate_manifest_schema(manifest, report)

    # 3. self-seal.
    if F.recompute_seal(manifest, F.SEAL_FIELD) != manifest.get(F.SEAL_FIELD):
        report.error("seal_failed", "SEAL-1(a)",
                     "dataset_sha256 does not recompute over the self-blanked manifest",
                     F.MANIFEST_NAME)
    report.ok("seal.manifest")

    _walk_paths(manifest, report, F.MANIFEST_NAME)

    if manifest_only:
        return report

    # 4. checksums.txt digest, then 5. every line.
    seal_block = manifest.get("seal") or {}
    checksums_path = os.path.join(root, seal_block.get("checksums_file") or F.CHECKSUMS_NAME)
    if not os.path.isfile(checksums_path):
        report.error("seal_failed", "SEAL-1(b)", "checksums.txt is missing")
    else:
        if F.sha256_file(checksums_path) != seal_block.get("checksums_sha256"):
            report.error("seal_failed", "SEAL-1(b)",
                         "sha256(checksums.txt) != seal.checksums_sha256")
        try:
            F.verify_checksums(root, allow_partial=allow_partial)
            report.ok("seal.checksums")
        except F.FormatError as exc:
            report.error(exc.code, "SEAL-1(c)/SEAL-2", exc.message)

    # 7. sub-manifest seals + their own rules.
    capture = manifest.get("capture") or {}
    panel = manifest.get("panel") or {}
    head = manifest.get("head") or {}
    runtime = manifest.get("runtime") or {}
    coverage = manifest.get("coverage") or {}
    determinism = manifest.get("determinism") or {}
    dataset = manifest.get("dataset") or {}
    scope = manifest.get("scope") or {}

    capture_manifest = _load_sub(root, capture.get("manifest_file") or "capture/manifest.json",
                                 report, "CAPTURE")
    panel_doc = _load_sub(root, panel.get("panel_file") or "panel/panel.json", report, "PANEL")
    head_doc = None
    if head.get("head_json"):
        head_doc = _load_sub(root, head["head_json"], report, "HEAD")
    runtime_doc = _load_sub(root, runtime.get("file") or "runtime/capture-runtime.json",
                            report, "RUNTIME")
    for doc, rel in ((capture_manifest, capture.get("manifest_file")),
                     (panel_doc, panel.get("panel_file")),
                     (head_doc, head.get("head_json")),
                     (runtime_doc, runtime.get("file"))):
        if doc is not None and rel:
            _check_sub_seal(doc, rel, report)
            _walk_paths(doc, report, rel)
    report.ok("seal.submanifests")

    # File digests of the sub-manifests, as the top-level manifest claims them.
    for label, rel, want in (
        ("capture.manifest_file_sha256", capture.get("manifest_file"),
         capture.get("manifest_file_sha256")),
        ("panel.panel_file_sha256", panel.get("panel_file"), panel.get("panel_file_sha256")),
        ("runtime.file_sha256", runtime.get("file"), runtime.get("file_sha256")),
    ):
        if rel and want and os.path.isfile(os.path.join(root, rel)):
            got = F.sha256_file(os.path.join(root, rel))
            if got != want:
                report.error("digest_mismatch", "SEAL-1", "%s: %s != %s" % (label, got, want), rel)

    # 8. capture_content_digest recomputes from the record list.
    records = (capture_manifest or {}).get("records") or []
    if capture_manifest is not None:
        try:
            derived = F.capture_content_digest(records)
            if derived != capture.get("capture_content_digest"):
                report.error("digest_mismatch", "5.2",
                             "capture_content_digest %s != manifest %s"
                             % (derived, capture.get("capture_content_digest")))
            if derived != (capture_manifest.get("capture_content_digest") or derived):
                report.error("digest_mismatch", "5.2",
                             "capture/manifest.json disagrees with its own records")
        except F.FormatError as exc:
            report.error(exc.code, "5.2", exc.message)
        report.ok("digest.capture_content")

        # REC-1/2/3.
        seen = set()
        tensor_key = capture_manifest.get("tensor_key")
        for record in records:
            index = record.get("index")
            if index in seen:
                report.error("digest_mismatch", "REC-1", "duplicate record index %r" % index)
            seen.add(index)
            if not isinstance(index, int) or index < 0:
                report.error("schema_invalid", "REC-1", "record index %r is not >= 0" % index)
            if record.get("key") != tensor_key:
                report.error("schema_invalid", "REC-2",
                             "record %r key %r != header tensor_key %r"
                             % (index, record.get("key"), tensor_key))
            shape = record.get("shape") or []
            if len(shape) == 2:
                if shape[0] != record.get("scored_rows"):
                    report.error("schema_invalid", "REC-3",
                                 "record %r shape[0] %r != scored_rows %r"
                                 % (index, shape[0], record.get("scored_rows")))
                want_width = (capture_manifest.get("hidden_width")
                              if capture.get("form") == "hidden"
                              else capture_manifest.get("vocab_size"))
                if want_width is not None and shape[1] != want_width:
                    report.error("schema_invalid", "REC-3",
                                 "record %r shape[1] %r != %r" % (index, shape[1], want_width))
            if record.get("raw_chunks_retained") is True:
                report.error("path_escape", "REC-4",
                             "record %r retains host-local chunk keys (PATH-2)" % index)
        report.ok("records")

    # 9-10. panel digests and BIND-1..6.
    if panel_doc is not None:
        _validate_panel(root, manifest, panel_doc, capture_manifest, report, allow_partial)

    # 11. coverage COV-1..3.
    present = len([r for r in records])
    declared = coverage.get("declared_records")
    if declared is not None:
        indices = sorted(int(r["index"]) for r in records if isinstance(r.get("index"), int))
        complete = (present == declared and indices == list(range(declared)))
        if bool(coverage.get("complete")) != complete:
            report.error("incomplete", "COV-1",
                         "coverage.complete is %r but present=%d declared=%d contiguous=%s"
                         % (coverage.get("complete"), present, declared,
                            indices == list(range(declared))))
        if coverage.get("present_records") != present:
            report.error("incomplete", "COV-1",
                         "coverage.present_records %r != %d records in the capture manifest"
                         % (coverage.get("present_records"), present))
        if not complete and not coverage.get("shard_of") and not coverage.get("subset_detail"):
            report.error("incomplete", "COV-2",
                         "an incomplete capture needs shard_of or subset_detail")
        if not complete and not allow_partial:
            report.error("incomplete", "COV-3",
                         "incomplete dataset; pass --allow-partial to accept it as a shard")
    report.ok("coverage")

    # Head: HEAD-4..7 and ROOT-1/HEAD-6.
    _validate_head(manifest, head, capture, dataset, report)

    # Determinism: DET-D1..D4, PANEL-D2.
    _validate_determinism(manifest, determinism, panel, capture, report)

    # Role matrix (spec section 3).
    _validate_role(root, manifest, dataset, scope, head, capture, report)

    # Scope vocabulary: the dataset and the registry must speak the same one.
    _validate_scope_vocabulary(scope, report)

    # Runtime.
    if runtime.get("lane") not in F.LANES:
        report.error("schema_invalid", "9.3", "lane %r is not a registry lane" % runtime.get("lane"))

    # Remap REMAP-1..3.
    if panel.get("remap_file"):
        _validate_remap(root, panel, report)

    # 12. optional tensor verification.
    if verify_tensors and capture_manifest is not None:
        _verify_tensors(root, capture, capture_manifest, head, report)

    if strict:
        report.errors.extend(report.warnings)
        report.warnings = []
    return report


def _validate_panel(root, manifest, panel_doc, capture_manifest, report, allow_partial):
    panel = manifest.get("panel") or {}
    panel_records = panel_doc.get("records") or []

    # BIND-6: the aggregate recomputes from the per-record digests, and each of
    # those recomputes from panel/tokens/context-NNNN.json.
    per_record = [r.get("token_ids_json_sha256") or "" for r in
                  sorted(panel_records, key=lambda r: int(r.get("index", 0)))]
    if per_record and all(per_record):
        aggregate = F.suite_token_hash_sha256(per_record)
        if aggregate != panel_doc.get("suite_token_hash_sha256"):
            report.error("panel_digest_mismatch", "BIND-6",
                         "suite_token_hash_sha256 does not recompute from the record digests")
        if aggregate != panel.get("suite_token_hash_sha256"):
            report.error("panel_digest_mismatch", "BIND-6",
                         "manifest panel.suite_token_hash_sha256 disagrees with panel.json")
        legacy = panel_doc.get("panel_token_sha256_legacy")
        if legacy and F.suite_token_hash_sha256_legacy(per_record) != legacy \
                and panel_doc.get("token_digest_algorithm"):
            report.warn("panel_digest_mismatch", "5.1",
                        "panel_token_sha256_legacy does not recompute under the legacy join; "
                        "it is a historical value carried for cross-check only")
    report.ok("panel.aggregate")

    checked = 0
    for record in panel_records:
        token_file = record.get("token_file")
        if not token_file:
            continue
        full = os.path.join(root, token_file)
        if not os.path.isfile(full):
            if not allow_partial:
                report.error("missing_file", "3", "panel token file %s is missing" % token_file)
            continue
        try:
            ids = F.read_json(full)
        except ValueError as exc:
            report.error("bad_schema", "BIND-6", "unparseable token file: %s" % exc, token_file)
            continue
        got = F.token_ids_json_sha256(ids)
        if got != record.get("token_ids_json_sha256"):
            report.error("panel_digest_mismatch", "BIND-6",
                         "%s hashes to %s, record says %s"
                         % (token_file, got, record.get("token_ids_json_sha256")))
        legacy = record.get("token_ids_sha256_legacy")
        if legacy and F.token_ids_json_sha256_legacy(ids) != legacy:
            report.error("panel_digest_mismatch", "5.1/P8",
                         "%s legacy digest does not recompute" % token_file)
        if record.get("num_tokens") is not None and len(ids) != record["num_tokens"]:
            report.error("panel_binding", "BIND-6",
                         "%s has %d tokens, record says %r"
                         % (token_file, len(ids), record["num_tokens"]))
        checked += 1
    report.ok("panel.tokens(%d)" % checked)

    # PANEL-D2: the panel receipt digest is traceability only.
    receipt_digest = panel.get("panel_receipt_sha256")
    if receipt_digest and receipt_digest == panel.get("suite_token_hash_sha256"):
        report.error("schema_invalid", "PANEL-D2",
                     "panel_receipt_sha256 reused as the token identity")

    # BIND-1..5.
    if capture_manifest is None:
        return
    panel_by_index = {}
    for record in panel_records:
        try:
            panel_by_index[int(record["index"])] = record
        except (KeyError, TypeError, ValueError):
            report.error("schema_invalid", "REC-1", "panel record without an integer index")
    total_positions = 0
    for record in capture_manifest.get("records") or []:
        index = record.get("index")
        counterpart = panel_by_index.get(index)
        if counterpart is None:
            report.error("panel_binding", "BIND-1",
                         "capture record %r has no panel record" % index)
            continue
        if record.get("token_ids_json_sha256") != counterpart.get("token_ids_json_sha256"):
            report.error("panel_binding", "BIND-2",
                         "record %r token digest differs between capture and panel" % index)
        a, b = record.get("attention_mask_sha256"), counterpart.get("attention_mask_sha256")
        if a is not None and b is not None and a != b:
            report.error("panel_binding", "BIND-3",
                         "record %r attention_mask_sha256 differs" % index)
        if record.get("scored_rows") != counterpart.get("prediction_positions"):
            report.error("panel_binding", "BIND-4",
                         "record %r scored_rows %r != panel prediction_positions %r"
                         % (index, record.get("scored_rows"),
                            counterpart.get("prediction_positions")))
        total_positions += int(record.get("scored_rows") or 0)
    if (manifest.get("coverage") or {}).get("complete"):
        if total_positions != panel.get("scored_positions_total"):
            report.error("panel_binding", "BIND-5",
                         "sum(scored_rows) %d != panel.scored_positions_total %r"
                         % (total_positions, panel.get("scored_positions_total")))
    report.ok("panel.binding")


def _validate_head(manifest, head, capture, dataset, report):
    form = capture.get("form")
    if form == "hidden":
        if head.get("applied_in_capture"):
            report.error("schema_invalid", "HEAD-5",
                         "hidden form declares the cut BEFORE the head, but "
                         "head.applied_in_capture is true")
        if not head.get("tensor_content_sha256"):
            report.error("head_mismatch", "HEAD-4",
                         "a hidden-form dataset with a null head content digest cannot be "
                         "scored through anyone's head; there is no override")
    if form == "logit" and head.get("present") and head.get("applied_in_capture") is False:
        report.error("schema_invalid", "4.2",
                     "logit form must declare head.applied_in_capture true")
    if head.get("raw_tensor_sha256") and head.get("tensor_content_sha256") \
            and head["raw_tensor_sha256"] != head["tensor_content_sha256"]:
        report.error("schema_invalid", "HEAD-IDENT",
                     "raw_tensor_sha256 is a REQUIRED byte-equal alias of "
                     "tensor_content_sha256; they differ")
    if head.get("file_sha256") and head.get("tensor_content_sha256") \
            and head["file_sha256"] == head["tensor_content_sha256"]:
        report.warn("schema_invalid", "HEAD-IDENT/O-6",
                    "head file_sha256 equals tensor_content_sha256; a container digest and a "
                    "content digest coinciding is almost certainly one convention pasted twice")
    final_norm = head.get("final_norm") or {}
    if capture.get("semantic_point") == "after_final_rmsnorm_before_lm_head" \
            and final_norm.get("applied_at_replay"):
        report.error("schema_invalid", "HEAD-7",
                     "the capture is already after the final norm; replay applies the head only")
    if dataset.get("role") == "root" and not head.get("present"):
        report.error("schema_invalid", "HEAD-6",
                     "a root that ships no head cannot be replayed against")
    report.ok("head")


def _validate_determinism(manifest, determinism, panel, capture, report):
    kind = determinism.get("evidence_kind")
    hashes = determinism.get("evidence_hashes") or []
    distinct = len(set(hashes))
    if determinism.get("distinct_evidence_hash_count") not in (None, distinct):
        report.error("schema_invalid", "DET-D1",
                     "distinct_evidence_hash_count %r != %d distinct hashes"
                     % (determinism.get("distinct_evidence_hash_count"), distinct))
    if determinism.get("identical_across_runs"):
        if kind not in F.CONTENT_EVIDENCE_KINDS:
            report.error("schema_invalid", "DET-D1",
                         "identical_across_runs requires content-level evidence, not %r" % kind)
        if int(determinism.get("run_count") or 0) < 2:
            report.error("schema_invalid", "DET-D1",
                         "identical_across_runs requires run_count >= 2")
        if distinct != 1:
            report.error("schema_invalid", "DET-D1",
                         "identical_across_runs with %d distinct evidence hashes" % distinct)
    # DET-D2: a container digest can never be evidence.
    containers = set()
    _collect_container_digests(manifest, containers)
    for value in hashes:
        if value in containers:
            report.error("schema_invalid", "DET-D2",
                         "evidence hash %s is a container (file) digest; stream_score writes "
                         "cold_run into __metadata__ so file digests differ between "
                         "bitwise-identical runs" % value[:12])
    # DET-D3 / PANEL-002.
    if panel.get("panel_receipt_sha256") in hashes:
        report.error("schema_invalid", "DET-D3",
                     "panel_receipt_sha256 may never be determinism evidence")
    if int(determinism.get("run_count") or 0) < 5:
        # DET-D4 asks for a disclosure; it used to warn even when the dataset
        # carried one, so the warning could never be cleared and every honest
        # single-run capture verified at exit 2 forever.  Warn only when the
        # disclosure is actually absent.
        declared = any((d or {}).get("code") == "reduced_run_count"
                       for d in (manifest.get("disclosures") or []))
        if not declared:
            report.warn("reduced_run_count", "DET-D4",
                        "run_count %r < 5 and no reduced_run_count disclosure is present; "
                        "a published dataset carries one"
                        % determinism.get("run_count"))
    report.ok("determinism")


def _collect_container_digests(node, out, key_hint=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("file_sha256", "sha256", "manifest_file_sha256", "panel_file_sha256",
                       "checksums_sha256", "receipt_sha256") and isinstance(value, str):
                out.add(value)
            _collect_container_digests(value, out, key)
    elif isinstance(node, list):
        for item in node:
            _collect_container_digests(item, out, key_hint)



_NUMERIC_FORMATS = None


def registry_numeric_formats():
    """The registry's `numeric_format` enum, READ from its schema, never copied.

    Returns None when the registry tree is not next to us, in which case the
    check is skipped rather than guessed at.
    """
    global _NUMERIC_FORMATS
    if _NUMERIC_FORMATS is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "registry", "schema", "common.schema.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                schema = json.load(handle)
            defs = schema.get("$defs") or schema.get("definitions") or {}
            _NUMERIC_FORMATS = frozenset(defs["numeric_format"]["enum"])
        except Exception:
            _NUMERIC_FORMATS = frozenset()
    return _NUMERIC_FORMATS or None


def _validate_scope_vocabulary(scope, report, strict=False):
    """SCOPE-VOCAB: a scope the registry will reject, caught at capture time.

    `scope.assignments[].format` is checked against the registry's own
    `numeric_format` enum.  Without this a capture can be run, sealed and
    published with a format string of the author's invention, and the first
    thing that notices is `registry_validate.py --submission` -- after the GPU
    time has been spent and the artifact is on the Hub.  A warning, not an
    error, because the dataset itself is still internally consistent and
    already-published datasets must keep verifying.

    `strict=True` is the controller's pre-spend gate on a scope FILE, where the
    same findings are refusals: scope_digest is sealed into the dataset, so the
    only fix after a capture is a re-capture. On a sealed dataset they stay
    warnings for the same reason the format check does -- the registry's own
    validator decides what it admits, and a validator that starts refusing
    datasets it verified yesterday is a wire-format break (2026-09-05: the
    published GLM-5.3 FP8 and K4 datasets carry the pre-SCOPE-004 two-rows-per-
    class scope and must keep verifying).
    """
    finding = report.error if strict else report.warn
    # SCOPE-004 and the closed assignment schema: the registry refuses a
    # duplicate (tensor_class, layer_range) and any key outside
    # {tensor_class, treatment, format, bits_per_weight, layer_range, note},
    # and both published rows of 2026-09-04 carried duplicates because nothing
    # here looked.
    assignments = scope.get("assignments") or []
    schema_keys = {"tensor_class", "treatment", "format", "bits_per_weight",
                   "layer_range", "note"}
    pairs = [(a.get("tensor_class"), a.get("layer_range")) for a in assignments]
    for pair in sorted({p for p in pairs if pairs.count(p) > 1}, key=str):
        finding("scope_duplicate_assignment", "SCOPE-004",
                "scope.assignments has %d rows for (tensor_class=%r, layer_range=%r); the "
                "registry admits one row per class and layer_range. Split by disjoint "
                "layer ranges, or write ONE row with format 'mixed' and the census in "
                "its note." % (pairs.count(pair), pair[0], pair[1]))
    for assignment in assignments:
        extra = sorted(set(assignment) - schema_keys)
        if extra:
            finding("scope_assignment_keys", "SCOPE-004",
                    "scope assignment %r carries keys the registry schema does not admit: %s"
                    % (assignment.get("tensor_class"), ", ".join(extra)))
    classes = {a.get("tensor_class") for a in assignments}
    for needed in ("embed_tokens", "lm_head"):
        if needed not in classes:
            finding("scope_coverage", "SCOPE-004",
                    "scope.assignments does not cover %s" % needed)
    if not any(str(c).startswith("attn.") for c in classes):
        finding("scope_coverage", "SCOPE-004", "scope.assignments covers no attention class")
    if not any(str(c).startswith(("mlp.", "moe.")) for c in classes):
        finding("scope_coverage", "SCOPE-004", "scope.assignments covers no mlp/moe class")
    allowed = registry_numeric_formats()
    if not allowed:
        return
    for assignment in assignments:
        value = assignment.get("format")
        if value is None or value in allowed:
            continue
        report.warn("scope_format_unknown", "SCOPE-VOCAB",
                    "scope assignment %r declares format %r, which is not in the registry's "
                    "numeric_format enum; a submission carrying this scope is REJECTED by "
                    "registry_validate.py --submission. Use one of: %s (put the exact "
                    "scheme in a disclosure instead)."
                    % (assignment.get("tensor_class"), value,
                       ", ".join(sorted(allowed))))
    report.ok("scope_vocabulary")

def _validate_role(root, manifest, dataset, scope, head, capture, report):
    role = dataset.get("role")
    # Spec section 3: both files are REQUIRED in a published dataset.  A draft
    # is allowed to be missing them; a dataset that claims `sealed` is not.
    sealed = dataset.get("structural_status") == "sealed"
    for relpath, why in (
        ("validation/structural-validation.json",
         "the validator's own verdict, sealed with the dataset"),
        ("README.md", "the dataset card carrying x_fidelity"),
    ):
        if not os.path.isfile(os.path.join(root, relpath)):
            (report.error if sealed else report.warn)(
                "missing_file", "3", "%s is missing (%s)" % (relpath, why), relpath)
    if role == "root":
        if scope.get("policy") != "native" or any(
            a.get("treatment") != "native" for a in scope.get("assignments") or []
        ):
            report.error("schema_invalid", "ROOT-1",
                         "a root declares scope.policy native with every tensor class native; "
                         "a quantized-weight capture is a quant dataset even if its author "
                         "calls it a reference")
        if head.get("quantized"):
            report.error("schema_invalid", "ROOT-1", "a root's head is not quantized")
        if dataset.get("base_capture") is not None:
            report.error("schema_invalid", "3", "a root's base_capture must be null")
        if capture.get("lossy_codec") is not None:
            report.error("schema_invalid", "3", "a root's lossy_codec must be null")
        if head.get("file") and not os.path.isfile(os.path.join(root, head["file"])):
            report.warn("missing_file", "3",
                        "root declares head.file %s but it is not present" % head["file"])
    if role == "derived" and dataset.get("base_capture") is None:
        report.error("schema_invalid", "3", "a derived dataset must name its base_capture")
    report.ok("role")


def _validate_remap(root, panel, report):
    remap = _load_sub(root, panel["remap_file"], report, "REMAP")
    if remap is None:
        return
    _check_sub_seal(remap, panel["remap_file"], report)
    entries = remap.get("entries") or {}
    for digest, target in sorted(entries.items()):
        try:
            F.check_relpath(target, owner="REMAP-2")
        except F.FormatError as exc:
            report.error("remap_invalid", "REMAP-2", exc.message, target)
            continue
        full = os.path.join(root, target)
        if not os.path.isfile(full):
            report.error("remap_invalid", "REMAP-2", "remap target %s is missing" % target)
            continue
        if F.sha256_file(full) != digest:
            report.error("remap_invalid", "REMAP-2",
                         "remap target %s does not hash to its key %s" % (target, digest[:12]))
    # REMAP-1/3: the copied sealed receipt still verifies, and every artifact
    # digest appears exactly once as a key.
    for_file = remap.get("for_receipt_file")
    if for_file:
        sealed = _load_sub(root, os.path.join(os.path.dirname(panel["remap_file"]), for_file)
                           if not for_file.startswith("panel/") else for_file,
                           report, "REMAP-3")
        if sealed is not None:
            if F.recompute_seal(sealed, "receipt_sha256") != sealed.get("receipt_sha256"):
                report.error("remap_invalid", "REMAP-3",
                             "the copied sealed receipt no longer verifies; it must be verbatim")
            wanted = {a.get("sha256") for a in sealed.get("artifacts") or []}
            missing = sorted(d for d in wanted if d and d not in entries)
            if missing:
                report.error("remap_invalid", "REMAP-1",
                             "%d sealed artifact digest(s) have no remap entry" % len(missing))
    report.ok("remap")


def _verify_tensors(root, capture, capture_manifest, head, report):
    checked = 0
    for record in capture_manifest.get("records") or []:
        rel = os.path.join(os.path.dirname(capture["manifest_file"]), record["file"])
        full = os.path.join(root, rel)
        if not os.path.isfile(full):
            continue
        try:
            content = F.tensor_content_sha256(full, record["key"])
            payload = F.payload_sha256(full)
            _, header = F.read_safetensors_header(full)
        except F.FormatError as exc:
            report.error(exc.code, "SEAL-1(d)", exc.message, rel)
            continue
        # The declared dtype/shape are what a consumer reads instead of opening
        # the tensor; hashing the bytes does not check that the DECLARATION
        # matches them.  A manifest that says F32 over BF16 bytes hashes clean.
        meta = header.get(record["key"]) or {}
        if record.get("dtype") and meta.get("dtype") \
                and str(meta["dtype"]).upper() != str(record["dtype"]).upper():
            report.error("tensor_mismatch", "SEAL-1(d)",
                         "%s declares dtype %r but the safetensors header says %r"
                         % (rel, record["dtype"], meta["dtype"]), rel)
        if record.get("shape") and meta.get("shape") \
                and [int(d) for d in meta["shape"]] != [int(d) for d in record["shape"]]:
            report.error("tensor_mismatch", "SEAL-1(d)",
                         "%s declares shape %s but the safetensors header says %s"
                         % (rel, list(record["shape"]), list(meta["shape"])), rel)
        if content != record.get("tensor_content_sha256"):
            report.error("tensor_mismatch", "SEAL-1(d)",
                         "%s tensor content %s != manifest %s"
                         % (rel, content[:12], (record.get("tensor_content_sha256") or "")[:12]))
        if record.get("payload_sha256") and payload != record["payload_sha256"]:
            report.error("tensor_mismatch", "SEAL-1(d)", "%s payload digest differs" % rel)
        if record.get("sha256") and F.sha256_file(full) != record["sha256"]:
            report.error("tensor_mismatch", "SEAL-1(d)", "%s file digest differs" % rel)
        checked += 1
    if head.get("file"):
        full = os.path.join(root, head["file"])
        if os.path.isfile(full):
            got = F.tensor_content_sha256(full, head["tensor_key"])
            if got != head.get("tensor_content_sha256"):
                report.error("tensor_mismatch", "HEAD-IDENT",
                             "head tensor content %s != manifest %s"
                             % (got[:12], (head.get("tensor_content_sha256") or "")[:12]))
    report.ok("tensors(%d)" % checked)
