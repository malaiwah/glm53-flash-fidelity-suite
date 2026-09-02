#!/usr/bin/env python3
"""The three-step fidelity tool: capture, verify, compare.

    step 1  capture   reference weights + panel  -> fidelity dataset A   [publish: REQUIRED for a root]
    step 2  capture   quantized weights + panel  -> fidelity dataset B   [publish: OPTIONAL]
    step 3  compare   A, B                       -> KLD + determinism + a registry receipt
    root proof  capture A1, capture A2            -> reproduction confirmation, exactly 0.0

Capture and comparison used to be fused, so every measurement re-paid for
capture, teachers were non-portable, and a lost capture killed reproducibility.
Splitting them makes a root capture a public good, lets a quant author
contribute a capture with no access to our infrastructure, and collapses the
same-lane floor toward zero.

Exit codes: 0 ok, 2 warnings only, 3 refused, 4 bad usage.

Full specification: docs/FIDELITY-DATASET-SPEC.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import stat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fidelity import common, jobcontract, panel, resultsink  # noqa: E402
from fidelity import dsformat as F  # noqa: E402
from fidelity import dsadapt, dsmanifest, dsvalidate  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OK, WARN, REFUSED, USAGE = 0, 2, 3, 4


def emit(text=""):
    print(common.redact(str(text)))


def refuse(code, message, remedy=None):
    emit("REFUSED [%s]: %s" % (code, message))
    if remedy:
        emit("  remedy: %s" % remedy)
    return REFUSED


def cache_path(base, repo, revision):
    """Where an `hf://repo@rev` fetch lands.  One directory per repo AND revision,
    ALWAYS, including under an explicit --cache."""
    root = base or os.path.join(REPO, "fidelity-runs", "datasets")
    return os.path.join(root, repo.replace("/", "__"), revision or "main")


def _resolve(ref, args, allow_partial=False, manifest_only=False):
    """Local dir, or `hf://repo[@rev]` fetched into a per-repo dir under --cache.

    The nesting is NOT cosmetic.  `--cache DIR` used to be the fetch
    destination itself, so `compare --reference hf://A --candidate hf://B
    --cache DIR` downloaded A into DIR, then downloaded B on top of it, and
    compared B against B: exit 0, "REPRODUCTION CONFIRMATION", 0.0 nats,
    class=strict, usable_as_floor=true, with BOTH sides of the receipt naming
    the same dataset_sha256.  Observed on the first two real published
    datasets.  A cache is a cache of many things; it is never one dataset's
    directory.
    """
    if not ref.startswith("hf://"):
        return ref
    from fidelity import dshub

    token = dshub.read_token(getattr(args, "token_file", None))
    repo, revision = dshub.parse_ref(ref)
    cache = cache_path(getattr(args, "cache", None), repo, revision)
    emit("fetching %s@%s -> %s" % (repo, revision, cache))
    return dshub.fetch_dataset(ref, cache, token=token, allow_partial=allow_partial,
                               manifest_only=manifest_only)


# ---------------------------------------------------------------------------
# validate / verify
# ---------------------------------------------------------------------------


def _print_report(report, verbose=True):
    for error in report.errors:
        emit("  ERROR   [%s/%s] %s%s" % (error["code"], error["rule"], error["message"],
                                         ("  (%s)" % error["where"]) if error["where"] else ""))
    for warning in report.warnings:
        emit("  warning [%s/%s] %s" % (warning["code"], warning["rule"], warning["message"]))
    if verbose:
        emit("  checks: %s" % ", ".join(sorted(report.checks)))


def cmd_validate(args):
    if args.receipt:
        receipt = F.read_json(args.receipt)
        report = dsvalidate.validate_receipt(receipt, args.receipt)
    else:
        root = _resolve(args.dataset, args, allow_partial=args.allow_partial)
        report = dsvalidate.validate_dataset(
            root, verify_tensors=args.verify_tensors, allow_partial=args.allow_partial,
            manifest_only=args.manifest_only, strict=args.strict)
    emit("%s: %d error(s), %d warning(s)"
         % (args.receipt or args.dataset, len(report.errors), len(report.warnings)))
    _print_report(report)
    if args.json:
        F.write_json(args.json, report.to_dict())
        emit("report -> %s" % args.json)
    if report.errors:
        return REFUSED
    return WARN if report.warnings else OK


def cmd_verify(args):
    """Same engine as validate, but stops at the first refusal.  No --force."""
    root = _resolve(args.dataset, args, allow_partial=args.allow_partial,
                    manifest_only=args.manifest_only)
    report = dsvalidate.validate_dataset(
        root, verify_tensors=args.verify_tensors, allow_partial=args.allow_partial,
        manifest_only=args.manifest_only)
    if report.errors:
        first = report.errors[0]
        if args.json:
            F.write_json(args.json, report.to_dict())
        return refuse(first["code"], "%s (%s)" % (first["message"], first["rule"]),
                      "there is no --force; fix the dataset or fetch it again")
    manifest = F.load_manifest(root)
    emit("VERIFIED %s" % root)
    emit("  dataset_sha256          %s" % manifest[F.SEAL_FIELD])
    emit("  capture_content_digest  %s" % manifest["capture"]["capture_content_digest"])
    emit("  head tensor content     %s" % manifest["head"].get("tensor_content_sha256"))
    emit("  panel suite token hash  %s" % manifest["panel"]["suite_token_hash_sha256"])
    emit("  tensors recomputed      %s" % ("yes" if args.verify_tensors else
                                           "NO (--no-verify-tensors was passed: a byte "
                                           "flipped inside a tensor whose checksums were "
                                           "refreshed is not caught)"))
    for warning in report.warnings:
        emit("  warning [%s] %s" % (warning["rule"], warning["message"]))
    if args.json:
        F.write_json(args.json, report.to_dict())
    return WARN if report.warnings else OK


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def cmd_describe(args):
    root = _resolve(args.dataset, args, manifest_only=True)
    manifest = F.load_manifest(root)
    if args.format == "json":
        emit(json.dumps(manifest, indent=2, sort_keys=True))
        return OK
    dataset, capture, panel = manifest["dataset"], manifest["capture"], manifest["panel"]
    head, runtime, coverage = manifest["head"], manifest["runtime"], manifest["coverage"]
    determinism = manifest["determinism"]
    lines = [
        "%s  (%s)" % (dataset["name"], dataset["id"]),
        "  role / status      %s / %s" % (dataset["role"], dataset["structural_status"]),
        "  form               %s at %s" % (capture["form"], capture["semantic_point"]),
        "  tensor key / dtype %s / %s (lossless=%s)"
        % (capture["tensor_key"], capture["dtype"], capture["dtype_lossless"]),
        "  geometry           vocab %s, hidden %s"
        % (capture["vocab_size"], capture.get("hidden_width")),
        "  panel              %s  %s contexts x %s"
        % (panel.get("panel_id"), panel["contexts"], panel["context_length"]),
        "  panel token hash   %s" % panel["suite_token_hash_sha256"],
        "  scoring window     score_from=%s windowed=%s"
        % (panel["scoring_window"]["score_from"], panel["scoring_window"]["windowed"]),
        "  head               %s, quantized=%s, source=%s"
        % (head["tensor_key"], head.get("quantized"), head.get("source")),
        "  head content       %s" % head.get("tensor_content_sha256"),
        "  lane               %s (inferred=%s)" % (runtime["lane"], runtime.get("lane_inferred")),
        "  stack fingerprint  %s" % runtime.get("stack_fingerprint_sha256"),
        "  coverage           %s/%s records, complete=%s"
        % (coverage["present_records"], coverage["declared_records"], coverage["complete"]),
        "  determinism        run_count=%s evidence=%s identical=%s"
        % (determinism["run_count"], determinism["evidence_kind"],
           determinism.get("identical_across_runs")),
        "  capture digest     %s" % capture["capture_content_digest"],
        "  dataset_sha256     %s" % manifest[F.SEAL_FIELD],
        "  lossy codec        %s" % (capture.get("lossy_codec") or "null"),
    ]
    divergences = (manifest.get("interop") or {}).get("divergences") or []
    if divergences:
        lines.append("  divergences        %s" % ", ".join(d["id"] for d in divergences))
    for disclosure in manifest.get("disclosures") or []:
        lines.append("  disclosure         %s (%s)" % (disclosure["code"], disclosure["severity"]))
    if args.format == "markdown":
        emit("| field | value |\n|---|---|")
        for line in lines[1:]:
            key, _, value = line.strip().partition("  ")
            emit("| %s | `%s` |" % (key.strip(), value.strip()))
    else:
        for line in lines:
            emit(line)
    return OK


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


PROVENANCE_TEMPLATE = {
    "_comment": [
        "Everything a registry submission needs that a fidelity dataset cannot know.",
        "artifact: the quant on the Hub. revision MUST be the immutable 40-hex commit",
        "  (IDENT-001). scope says what is quantized and what is native; it is what",
        "  scope_digest is computed over, so it is identity, not description.",
        "panel_ref / reference_ref: registry ids that must ALREADY exist -- a",
        "  measurement may not introduce a panel (CONTRIBUTING.md section 6).",
        "See registry/schema/submission.schema.json for every field and its meaning.",
    ],
    "measurer": {"name": "", "handle": "", "url": None, "is_artifact_author": False},
    "artifact": {
        "repository": "owner/repo", "revision": "0" * 40, "url": None,
        "container": "safetensors", "precision_label": None, "size_bytes": None,
        "index_sha256": None, "config_sha256": None, "shard_hash_verification": "none",
        "codec": {"family": "exl3", "bits_per_weight_nominal": None,
                  "bits_per_weight_effective": None, "group_size": None,
                  "quantizer_tool": None, "quantizer_version": None},
        "scope": {"policy": "uniform", "head_policy": "native", "kv_cache_dtype": "bf16",
                  "assignments": [{"tensor_class": "routed_expert_mlp",
                                   "treatment": "quantized", "format": "exl3",
                                   "bits_per_weight": None, "layer_range": None}]},
        "producer": {"name": "", "handle": None, "url": None},
    },
    "panel": {"panel_ref": "panel--", "panel_token_sha256": None,
              "panel_receipt_sha256": None, "contexts": None,
              "scored_positions_total": None},
    "reference": {"reference_ref": "reference--", "teacher_receipt_sha256": None,
                  "teacher_backend_identity_sha256": None},
    "environment": {"gpu": None, "gpu_count": None, "host": None,
                    "wall_clock_hours": None},
}


def _validate_submission(path, handle=None):
    """Run `registry/tools/registry_validate.py --submission` on our own output.

    The validator also enforces that the file lives in a directory named after
    the measurer's handle -- a rule about where you COMMIT it in the registry,
    not about where this tool happened to write it. So the check runs on a copy
    filed the way a contributor would file it, and the caller is told the path
    the registry expects.
    """
    validator = os.path.join(REPO, "registry", "tools", "registry_validate.py")
    if not os.path.isfile(validator):
        return {"accepted": None, "summary": "registry/ not present; not checked",
                "detail": []}
    target = path
    if handle:
        import shutil
        import tempfile

        staged = os.path.join(tempfile.mkdtemp(prefix="fidelity-submit-"), handle)
        os.makedirs(staged, exist_ok=True)
        target = os.path.join(staged, os.path.basename(path))
        shutil.copyfile(path, target)
    proc = subprocess.run([sys.executable, validator, "--submission", target],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    accepted = proc.returncode == 0
    return {"accepted": accepted,
            "summary": ("ACCEPTED by registry_validate.py --submission"
                        if accepted else
                        "REJECTED by registry_validate.py --submission (exit %d)"
                        % proc.returncode),
            "detail": lines}


def cmd_verify_k3_compat(args):
    from fidelity import k3compat

    problems = k3compat.verify(args.dataset)
    if problems:
        for problem in problems:
            emit("  %s" % problem)
        return refuse("k3_compat_invalid", "%d problem(s) in compat/" % len(problems),
                      "regenerate it: the tree is a VIEW, never hand-edited "
                      "(and it is inside the seal, so an edit refuses the dataset too)")
    emit("compat/ is faithful to %s" % args.dataset)
    emit("  the kimi-k3 comparator reads this dataset unmodified:")
    emit("    --suite-manifest %s/compat/suite-manifest.json" % args.dataset)
    return OK


def cmd_provenance_template(args):
    text = json.dumps(PROVENANCE_TEMPLATE, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        emit("wrote %s" % args.out)
        emit("fill it in, then: fidelity-dataset compare ... --emit-submission "
             "--submission-provenance %s" % args.out)
        return OK
    emit(text)
    return OK


def cmd_compare(args):
    from fidelity import dscompare

    reference = _resolve(args.reference, args, allow_partial=args.allow_partial)
    candidate = _resolve(args.candidate, args, allow_partial=args.allow_partial)
    # Defence in depth behind the cache-nesting fix in _resolve(): if the two
    # sides ever resolve to ONE directory again -- a symlink, a typo, a future
    # caching change -- the comparison is a directory against itself and every
    # gate passes. That is the one wrong answer this tool must never return
    # quietly, so it is a refusal even under --self-compare, where the operator
    # is asking to compare two SEPARATE captures of the same weights.
    if os.path.realpath(reference) == os.path.realpath(candidate):
        return refuse("same_root",
                      "reference and candidate resolve to the same directory (%s); the "
                      "comparison would be that directory against itself and would report "
                      "0.0 nats through every gate"
                      % os.path.realpath(reference),
                      "point --reference and --candidate at two different roots; a "
                      "self-compare needs two SEPARATE cold captures, not one path twice")
    options = {
        "device": args.device,
        "replay_device": args.replay_device,
        "replay_dtype": args.replay_dtype,
        "vocab_chunk": args.vocab_chunk,
        "position_block": args.chunk_positions,
        "head_path": args.head,
        "self_compare": args.self_compare,
        "force_compute": args.force_compute,
        "allow_cross_lane": args.allow_cross_lane,
        "allow_partial": args.allow_partial,
        "disclose_head_substitution": args.disclose_head_substitution,
        "verify_tensors": args.verify_tensors,
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
    }
    try:
        receipt = dscompare.compare(reference, candidate, args.out, options)
    except dscompare.Refusal as exc:
        return refuse(exc.code, "gate %s: %s" % (exc.gate, exc.message),
                      exc.override or exc.remedy)
    except F.FormatError as exc:
        return refuse(exc.code, exc.message)

    emit("%s" % receipt["comparison_kind"].upper().replace("_", " "))
    emit("  metric              %s = %r %s"
         % (receipt["metric"]["name"], receipt["metric"]["value"], receipt["metric"]["units"]))
    emit("  direction           %s" % receipt["metric"]["direction_label"])
    emit("  top-1 agreement     %r" % receipt["top1_agreement"])
    emit("  kl                  %s" % json.dumps(receipt["kl"]))
    emit("  scored positions    %s over %s contexts"
         % (receipt["measurement_scope"]["scored_positions"],
            receipt["measurement_scope"]["contexts"]))
    emit("  estimator           full vocabulary, %s, head_policy=%s, stack=%s"
         % (receipt["estimator"]["accumulation_dtype"], receipt["estimator"]["head_policy"],
            receipt["estimator"]["stack_relation"]))
    emit("  backend             %s (replay %s)"
         % (receipt["comparator"].get("estimator_backend"),
            receipt["comparator"].get("replay_backend")))
    emit("  comparability       class=%s same_lane=%s usable_as_floor=%s"
         % (receipt["comparability"]["class"], receipt["comparability"]["same_lane"],
            receipt["comparability"]["usable_as_floor"]))
    for disclosure in receipt["disclosures"]:
        emit("  disclosure          %s (%s)" % (disclosure["code"], disclosure["severity"]))
    emit("  receipt             %s" % os.path.join(args.out, "comparison-receipt.json"))
    emit("  tokenwise           %s (%s bytes, %s)"
         % (receipt["tokenwise"]["path"], receipt["tokenwise"]["bytes"],
            receipt["tokenwise"]["sha256"][:16]))

    report = dsvalidate.validate_receipt(receipt, args.out)
    if report.errors:
        _print_report(report, verbose=False)
        return refuse("schema_invalid", "the emitted receipt does not validate")

    if args.emit_submission:
        provenance = {}
        if args.submission_provenance:
            try:
                provenance = F.read_json(args.submission_provenance)
            except Exception as exc:
                return refuse("bad_provenance", "cannot read %s: %s"
                              % (args.submission_provenance, exc),
                              "write one with --print-provenance-template")
        measurer = provenance.get("measurer") or {
            "name": args.measurer or "unknown", "handle": args.measurer,
            "url": None, "is_artifact_author": False}
        submission_path = os.path.join(args.out, "submission-receipt.json")
        try:
            dscompare.emit_submission(
                receipt, submission_path, measurer=measurer,
                artifact=provenance.get("artifact") or {},
                panel=provenance.get("panel") or {},
                reference=provenance.get("reference") or {},
                environment=provenance.get("environment"))
        except dscompare.NotAMeasurement as exc:
            emit("  submission          %s" % exc)
            return WARN
        except dscompare.MissingProvenance as exc:
            emit("  submission          %s" % exc)
            return WARN
        except Exception as exc:  # the builder's own refusals carry their reason
            emit("  submission          REFUSED: %s" % exc)
            return WARN
        emit("  submission          %s" % submission_path)
        # The registry's OWN gate, on the file we just wrote, before anyone is
        # told it is submittable. `--emit-submission` used to print a path and
        # exit 0 for a file that `registry-submit` rejected with twenty errors.
        verdict = _validate_submission(submission_path, measurer.get("handle"))
        emit("  submission gate     %s" % verdict["summary"])
        for line in verdict["detail"][:8]:
            emit("    %s" % line)
        if not verdict["accepted"]:
            return WARN
        emit("  to submit           copy it to registry/receipts/%s/ and open a PR "
             "(registry/CONTRIBUTING.md)" % (measurer.get("handle") or "<your-handle>"))
    if receipt["comparability"]["class"] != "strict":
        return WARN
    return OK


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------

CAPTURE_REFUSALS = (
    ("--sweep", "a sweep runs extra forwards that interleave the hidden-state tap"),
    ("--store-positions", "only `all` is a capture; a sampled store is a preview"),
)


def _preflight(passthrough):
    """The pre-flight refusals inherited from `hidden_replay.run_capture`."""
    problems = []
    for index, item in enumerate(passthrough):
        if item == "--sweep" or item.startswith("--sweep="):
            problems.append("--sweep: %s" % CAPTURE_REFUSALS[0][1])
        if item == "--store-positions" and index + 1 < len(passthrough):
            if passthrough[index + 1] != "all":
                problems.append("--store-positions %s: %s"
                                % (passthrough[index + 1], CAPTURE_REFUSALS[1][1]))
        if item.startswith("--store-positions=") and item.split("=", 1)[1] != "all":
            problems.append("%s: %s" % (item, CAPTURE_REFUSALS[1][1]))
    if not any(item == "--token-panel" or item.startswith("--token-panel=")
               for item in passthrough):
        problems.append("--token-panel is REQUIRED: the wrapper needs the mask .npy paths, "
                        "which capture-receipt.json does not carry")
    return problems


def _postcondition(out):
    """`capture --out X` must leave a dataset at X.  It used to exit 0 having left
    a capture WORK TREE and no dataset at all, which is the worst possible
    failure: a green exit code over a missing artifact."""
    manifest = os.path.join(out, F.MANIFEST_NAME)
    if not os.path.isfile(manifest):
        return refuse("dataset_not_assembled",
                      "the capture finished but %s does not exist, so --out %s holds no "
                      "dataset" % (manifest, out),
                      "the sealed-lane engine writes a capture WORK TREE, not a dataset; "
                      "either assemble it (bin/fidelity/dsmanifest.py, bin/fidelity/"
                      "dsadapt.py) or use --engine hf-transformers, which writes the "
                      "dataset itself")
    report = dsvalidate.validate_dataset(out, verify_tensors=False)
    if report.errors:
        first = report.errors[0]
        _print_report(report, verbose=False)
        return refuse("dataset_invalid",
                      "the capture wrote %s but it does not validate: %s (%s)"
                      % (out, first["message"], first["rule"]))
    emit("SEALED DATASET -> %s" % out)
    manifest_doc = F.load_manifest(out)
    emit("  dataset_sha256          %s" % manifest_doc[F.SEAL_FIELD])
    emit("  capture_content_digest  %s" % manifest_doc["capture"]["capture_content_digest"])
    emit("  records / scored rows   %s / %s" % (manifest_doc["capture"]["records_count"],
                                                manifest_doc["capture"]["scored_rows_total"]))
    return WARN if report.warnings else OK


def cmd_capture(args):
    passthrough = list(args.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if args.engine == "hf-transformers":
        if args.form != "hidden":
            return refuse("bad_capture_argv",
                          "--engine hf-transformers captures hidden form only "
                          "(a logit-form capture of a 154,880-token vocabulary is ~1,200x "
                          "the bytes; use the streaming lane if you really want it)")
        if os.path.exists(args.out) and not args.force:
            return refuse("destination_exists", "%s exists" % args.out, "pass --force")
        tool = os.path.join(REPO, "engines", "tools", "hf_capture.py")
        python = os.environ.get("FIDELITY_PYTHON", sys.executable)
        command = ([python, tool, "--out", args.out, "--role", args.role,
                    "--lane", args.lane] + (["--force"] if args.force else [])
                   + passthrough)
        emit("capture plan")
        emit("  engine          hf-transformers (engines/tools/hf_capture.py)")
        emit("  form / role     %s / %s   lane %s" % (args.form, args.role, args.lane))
        emit("  dataset root    %s" % args.out)
        emit("  command         %s" % " ".join(command))
        if args.dry_run:
            emit("--dry-run is not supported by this engine: it has no plan phase "
                 "separate from the forward pass. Run it on a small --windows instead.")
            return USAGE
        result = subprocess.call(command)
        if result != 0:
            return refuse("capture_failed",
                          "the capture exited %d; no dataset written" % result)
        return _postcondition(args.out)

    problems = _preflight(passthrough)
    if problems:
        for problem in problems:
            emit("  " + problem)
        return refuse("bad_capture_argv", "%d pre-flight refusal(s)" % len(problems),
                      "these are inherited from engines/tools/hidden_replay.py::run_capture")

    if os.path.exists(args.out) and not args.force:
        return refuse("destination_exists", "%s exists" % args.out, "pass --force")

    tool = os.path.join(REPO, "engines", "tools",
                        "hidden_replay.py" if args.form == "hidden" else "stream_score.py")
    if not os.path.isfile(tool):
        return refuse("engine_missing",
                      "%s does not exist in this checkout" % os.path.relpath(tool, REPO),
                      "engines/tools/hidden_replay.py is campaign-internal and is not part of the "
                      "published repository; from a public clone use "
                      "`--engine hf-transformers`")
    python = os.environ.get("FIDELITY_PYTHON", sys.executable)
    work = args.work or (args.out + ".capture")
    if args.form == "hidden":
        command = [python, tool, "capture", "--out", work, "--"] + passthrough
    else:
        command = [python, tool, "--out", work] + passthrough
    if args.dry_run:
        command.append("--dry-run")

    emit("capture plan")
    emit("  engine          %s" % args.engine)
    emit("  form            %s" % args.form)
    emit("  role / lane     %s / %s" % (args.role, args.lane))
    emit("  wraps           %s (never edited; stream_score's own path is byte-identical "
         "to a plain run)" % os.path.relpath(tool, REPO))
    emit("  work dir        %s" % work)
    emit("  dataset root    %s" % args.out)
    emit("  command         %s" % " ".join(command))
    if args.dry_run:
        emit("")
        emit("--dry-run: stream_score validates every input, seal and layout and exits 0")
        emit("without touching weights or a GPU. This is the CI conformance hook.")
        result = subprocess.call(command)
        if result != 0:
            return refuse("capture_failed", "the pass-through --dry-run exited %d" % result)
        emit("dry run OK; no dataset was written")
        return OK

    result = subprocess.call(command)
    if result != 0:
        return refuse("capture_failed", "the capture exited %d; no dataset written" % result)
    emit("capture finished; assembling the dataset from %s" % work)
    return _postcondition(args.out)


# ---------------------------------------------------------------------------
# adapt
# ---------------------------------------------------------------------------


def cmd_adapt(args):
    # `--role` defaults to root, which is right for our own capture and WRONG for
    # a kimi-k3 translation (ROOT-1 asserts a head quantization status that
    # artifact never records). Distinguish "the default" from "the operator asked
    # for root", so the refusal only fires on a real request.
    args.role_explicit = any(item == "--role" or item.startswith("--role=")
                             for item in sys.argv[1:])
    try:
        if args.source in ("k3v1", "k3v0-window"):
            report = dsadapt.adapt_k3(
                args.input, args.out, source=args.source, tokens_dir=args.tokens,
                recompute_content_digests=args.recompute_content_digests,
                emit_dataset=args.emit_dataset, emit_k3_compat=args.emit_k3_compat,
                dataset_id=args.dataset_id or "fidelity--adapted.kimi-k3",
                name=args.name or "adapted kimi-k3 capture",
                role=(args.role if args.role_explicit else "derived"),
                lane=args.lane, limit=args.limit, link=not args.copy)
            emit("translated %s -> %s" % (args.source, args.out))
            emit("  panel aggregate agrees with the source manifest: %s"
                 % report["panel"]["aggregate_agrees"])
            emit("  declared / present records: %s / %s"
                 % (report["coverage"]["declared_records"],
                    report["coverage"]["present_records"]))
            emit("  inferred fields (each forces advisory at compare time):")
            for field in report["inferred_fields"]:
                emit("    - %s" % field)
            for item in report.get("outstanding") or []:
                emit("  outstanding: %s" % item)
            emitted = report.get("emitted")
            if emitted and emitted.get("written"):
                emit("  SEALED DATASET -> %s" % args.out)
                emit("    dataset_sha256          %s" % emitted["dataset_sha256"])
                emit("    capture_content_digest  %s" % emitted["capture_content_digest"])
                emit("    records                 %s" % emitted["records"])
                emit("    head payload present    %s" % emitted["head_payload_present"])
                check = dsvalidate.validate_dataset(args.out, verify_tensors=True,
                                                    allow_partial=True)
                emit("    validate                %d error(s), %d warning(s)"
                     % (len(check.errors), len(check.warnings)))
                _print_report(check, verbose=False)
                if check.errors:
                    return REFUSED
            elif emitted:
                emit("  NOT SEALED: %s" % emitted["reason"])
                return WARN
            return WARN if report["coverage"]["present_records"] == 0 else OK
        if args.source == "llamacpp-kld":
            report = dsadapt.adapt_llamacpp_kld(args.input, args.out)
            emit("translated llamacpp-kld -> %s" % args.out)
            emit("  lossy codec: %s" % json.dumps(report["capture"]["lossy_codec"]))
            emit("  scoring window: %s" % json.dumps(report["panel"]["scoring_window"]))
            for item in report.get("outstanding") or []:
                emit("  outstanding: %s" % item)
            return WARN
        if args.source == "malaiwah-serving-v2":
            if not args.suite:
                return refuse("bad_usage", "--suite is required for malaiwah-serving-v2")
            manifest = dsadapt.adapt_serving_v2(
                args.input, args.out, suite_dir=args.suite, head_dir=args.head_dir,
                dataset_id=args.dataset_id or "fidelity--adapted.serving-v2",
                name=args.name or "adapted serving-v2 capture",
                role=args.role, lane=args.lane, limit=args.limit, link=not args.copy,
                emit_k3_compat=args.emit_k3_compat)
            emit("adapted -> %s" % args.out)
            emit("  dataset_sha256          %s" % manifest[F.SEAL_FIELD])
            emit("  capture_content_digest  %s" % manifest["capture"]["capture_content_digest"])
            emit("  head tensor content     %s" % manifest["head"]["tensor_content_sha256"])
            emit("  coverage                %s/%s complete=%s"
                 % (manifest["coverage"]["present_records"],
                    manifest["coverage"]["declared_records"],
                    manifest["coverage"]["complete"]))
            report = dsvalidate.validate_dataset(args.out, verify_tensors=True,
                                                 allow_partial=True)
            emit("  validate                %d error(s), %d warning(s)"
                 % (len(report.errors), len(report.warnings)))
            _print_report(report, verbose=False)
            return REFUSED if report.errors else OK
    except dsadapt.AdapterError as exc:
        return refuse("adapter_refused", str(exc))
    return USAGE



# ---------------------------------------------------------------------------
# root qualification
# ---------------------------------------------------------------------------


class RootQualificationError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_QUALIFICATION_SCHEMA = "fidelity.root-qualification-receipt.v1"
_QUALIFICATION_KEYS = {
    "schema", "receipt_sha256", "qualified_at", "canonical_job_sha256",
    "job_file_sha256", "dataset_repository", "destination_repository",
    "job_contract", "captures", "comparison", "comparator", "verification",
    "reproduction_confirmation",
}
_QUALIFICATION_JOB_CONTRACT_KEYS = (
    jobcontract.ROOT_QUALIFICATION_CONTRACT_KEYS)

def _read_json_file(path, label):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return common.parse_json(handle.read())
    except (OSError, ValueError, TypeError) as exc:
        raise RootQualificationError("%s is not readable strict JSON: %s" % (label, exc))


def _positive_zero(value):
    return isinstance(value, float) and value == 0.0 \
        and math.copysign(1.0, value) == 1.0


def _verified_report(path, dataset_root, label):
    doc = _read_json_file(path, "%s verification receipt" % label)
    if not common.verify_seal(doc):
        raise RootQualificationError("%s verification receipt seal does not recompute" % label)
    if doc.get("schema") != F.VALIDATION_SCHEMA or doc.get("structural_status") != "sealed" \
            or doc.get("error_count") != 0 or doc.get("errors") != []:
        raise RootQualificationError("%s verification receipt is not a successful full verify"
                                     % label)
    subject = doc.get("subject")
    if not isinstance(subject, str) or os.path.realpath(subject) != os.path.realpath(dataset_root):
        raise RootQualificationError(
            "%s verification receipt names %r, not dataset root %s"
            % (label, subject, dataset_root))
    return {
        "receipt_sha256": doc["receipt_sha256"],
        "file_sha256": common.sha256_file(path),
    }


def _capture_identity(root, expected_label, label):
    report = dsvalidate.validate_dataset(root, verify_tensors=True)
    if report.errors:
        raise RootQualificationError(
            "%s dataset does not independently verify (%s)"
            % (label, report.errors[0]["message"]))
    manifest_path = os.path.join(root, F.MANIFEST_NAME)
    manifest = F.load_manifest(root)
    dataset = manifest.get("dataset") or {}
    determinism = manifest.get("determinism") or {}
    capture = manifest.get("capture") or {}
    runtime = manifest.get("runtime") or {}
    if dataset.get("role") != "root":
        raise RootQualificationError("%s dataset role is %r, not root"
                                     % (label, dataset.get("role")))
    if determinism.get("run_count") != 1:
        raise RootQualificationError(
            "%s dataset determinism.run_count is %r; each public manifest must remain "
            "one independent capture" % (label, determinism.get("run_count")))
    capture_rel = capture.get("manifest_file")
    runtime_rel = runtime.get("file")
    if not capture_rel or not runtime_rel:
        raise RootQualificationError("%s dataset omits its capture/runtime manifest" % label)
    capture_path = F.resolve_inside(root, capture_rel, owner="root qualification/capture")
    runtime_path = F.resolve_inside(root, runtime_rel, owner="root qualification/runtime")
    capture_doc = _read_json_file(capture_path, "%s capture manifest" % label)
    runtime_doc = _read_json_file(runtime_path, "%s runtime manifest" % label)
    run_name = capture_doc.get("run_name")
    cold_run = (runtime_doc.get("runtime_environment") or {}).get("cold_run")
    if run_name != expected_label or cold_run != expected_label:
        raise RootQualificationError(
            "%s process label mismatch: expected %r, capture run_name=%r, runtime cold_run=%r"
            % (label, expected_label, run_name, cold_run))
    stack_fingerprint = runtime_doc.get("stack_fingerprint") or {}
    capture_tool = runtime_doc.get("capture_tool") or {}
    runtime_container = runtime_doc.get("container")
    if not isinstance(runtime_container, dict):
        raise RootQualificationError(
            "%s runtime manifest container identity is not an object" % label)
    manifest_panel = manifest.get("panel")
    binding_evidence = capture_tool.get("resolved_panel_binding")
    resolved_binding = (
        binding_evidence.get("binding")
        if isinstance(binding_evidence, dict) else None)
    receipt_binding = (
        resolved_binding.get("receipt")
        if isinstance(resolved_binding, dict) else None)
    receipt_rel = (
        manifest_panel.get("panel_receipt_file")
        if isinstance(manifest_panel, dict) else None)
    if receipt_rel != "panel/panel-receipt.json":
        raise RootQualificationError(
            "%s manifest does not name the canonical bound panel receipt" % label)
    if (manifest_panel.get("panel_receipt_sha256")
            != (receipt_binding or {}).get("declared_receipt_sha256")):
        raise RootQualificationError(
            "%s manifest panel receipt identity differs from resolved binding"
            % label)
    expected_receipt_bytes = (
        receipt_binding.get("bytes")
        if isinstance(receipt_binding, dict) else None)
    if (isinstance(expected_receipt_bytes, bool)
            or not isinstance(expected_receipt_bytes, int)
            or not 0 < expected_receipt_bytes <= 16 * 1024 * 1024):
        raise RootQualificationError(
            "%s resolved panel receipt byte count is invalid" % label)
    try:
        receipt_path = F.resolve_inside(root, receipt_rel)
        with open(receipt_path, "rb") as handle:
            raw_receipt = handle.read(expected_receipt_bytes + 1)
        panel.verify_bound_panel_receipt_bytes(
            receipt_binding, raw_receipt, "%s panel receipt" % label)
    except (OSError, F.FormatError, panel.PanelError) as exc:
        raise RootQualificationError(
            "%s bound panel receipt is invalid: %s" % (label, exc)) from exc
    dataset_license = dataset.get("license")
    license_file_sha256 = None
    license_file_bytes = None
    if dataset_license == "other":
        try:
            license_path = F.resolve_inside(
                root, "LICENSE", owner="%s weights license" % label)
            license_file_sha256 = common.sha256_file(license_path)
            license_file_bytes = os.path.getsize(license_path)
        except (OSError, F.FormatError) as exc:
            raise RootQualificationError(
                "%s dataset license file is invalid: %s" % (label, exc)) from exc
    return {
        "process_label": expected_label,
        "dataset_id": dataset.get("id"),
        "dataset_name": dataset.get("name"),
        "dataset_author": (dataset.get("author") or {}).get("name"),
        "dataset_repository": dataset.get("repository"),
        "dataset_license": dataset_license,
        "weights_license_file_sha256": license_file_sha256,
        "weights_license_file_bytes": license_file_bytes,
        "dataset_sha256": manifest.get(F.SEAL_FIELD),
        "dataset_manifest_file_sha256": common.sha256_file(manifest_path),
        "capture_manifest": capture_rel,
        "capture_manifest_sha256": common.sha256_file(capture_path),
        "capture_content_digest": capture.get("capture_content_digest"),
        "capture_form": capture.get("form"),
        "capture_dtype": capture.get("dtype"),
        "runtime_manifest": runtime_rel,
        "runtime_manifest_sha256": common.sha256_file(runtime_path),
        "runtime_lane": runtime.get("lane"),
        "runtime_device": stack_fingerprint.get("device"),
        "runtime_engine": stack_fingerprint.get("engine"),
        "runtime_container": runtime_container,
        "capture_tool_file": capture_tool.get("file"),
        "capture_schedule": capture_tool.get("schedule"),
        "panel": {
            "panel_id": (manifest.get("panel") or {}).get("panel_id"),
            "suite_token_hash_sha256":
                (manifest.get("panel") or {}).get("suite_token_hash_sha256"),
            "panel_receipt_sha256":
                (manifest.get("panel") or {}).get("panel_receipt_sha256"),
            "tokenizer": (manifest.get("panel") or {}).get("tokenizer"),
            "resolved_binding_evidence": capture_tool.get("resolved_panel_binding"),
        },
        "unexpected_tensor_allowlist":
            capture_tool.get("unexpected_tensor_allowlist"),
        "weights_license": capture_tool.get("weights_license"),
        "stack_fingerprint_sha256": runtime.get("stack_fingerprint_sha256"),
        "lane_identity_sha256": runtime.get("lane_identity_sha256"),
        "weights_repository": (manifest.get("weights") or {}).get("repository"),
        "weights_revision": (manifest.get("weights") or {}).get("revision"),
        "determinism_run_count": 1,
    }



def _check_capture_job_contract(job, identity, label):
    capture = job.get("capture") or {}
    panel_job = job.get("panel") or {}
    binding = panel_job.get("resolved_binding")
    if not isinstance(binding, dict):
        raise RootQualificationError("job panel.resolved_binding is absent")
    panel_binding = binding.get("panel") or {}
    receipt_binding = binding.get("receipt") or {}
    tokenizer_binding = binding.get("tokenizer") or {}
    if tokenizer_binding.get("files_verified") is not True:
        raise RootQualificationError("job panel tokenizer files are not verified")
    def hex64(value):
        return (isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value))

    if (not panel_binding.get("id")
            or not hex64(panel_binding.get("suite_token_hash_sha256"))
            or not hex64(receipt_binding.get("declared_receipt_sha256"))
            or not hex64(panel_job.get("binding_file_sha256"))):
        raise RootQualificationError(
            "job panel binding lacks a sealed panel/receipt identity")
    if (not tokenizer_binding.get("repository")
            or not tokenizer_binding.get("revision")
            or not isinstance(tokenizer_binding.get("vocab_size"), int)
            or tokenizer_binding.get("vocab_size") <= 0
            or not isinstance(tokenizer_binding.get("files"), list)
            or not tokenizer_binding.get("files")
            or not hex64(tokenizer_binding.get("identity_sha256"))):
        raise RootQualificationError(
            "job panel binding lacks a complete tokenizer identity")

    expected_dtype = {"bfloat16": "BF16", "bf16": "BF16"}.get(
        str(capture.get("dtype")).lower())
    represented = (
        ("dataset_id", identity["dataset_id"], capture.get("dataset_id")),
        ("dataset_name", identity["dataset_name"], capture.get("dataset_name")),
        ("dataset_author", identity["dataset_author"], capture.get("author")),
        ("dataset_repository", identity["dataset_repository"],
         capture.get("dataset_repository")),
        ("dataset_license", identity["dataset_license"],
         capture.get("dataset_license")),
        ("lane", identity["runtime_lane"], job.get("lane")),
        ("form", identity["capture_form"], capture.get("form")),
        ("schedule", identity["capture_schedule"], capture.get("schedule")),
        ("device", identity["runtime_device"], capture.get("device")),
        ("dtype", identity["capture_dtype"], expected_dtype),
    )
    for field, observed, expected in represented:
        if expected is None or expected == "" or observed != expected:
            raise RootQualificationError(
                "%s capture %s=%r does not match job value %r"
                % (label, field, observed, expected))
    if capture.get("panel_id") != panel_binding.get("id"):
        raise RootQualificationError(
            "job capture.panel_id does not match panel.resolved_binding")
    if capture.get("engine") != "hf-transformers" \
            or identity["runtime_engine"] != "transformers-eager" \
            or identity["capture_tool_file"] != "engines/tools/hf_capture.py":
        raise RootQualificationError(
            "%s capture engine evidence does not match job engine %r"
            % (label, capture.get("engine")))

    panel_identity = identity["panel"]
    evidence = panel_identity.get("resolved_binding_evidence") or {}
    if evidence.get("binding_file_sha256") != panel_job.get("binding_file_sha256") \
            or evidence.get("binding") != binding:
        raise RootQualificationError(
            "%s capture lacks the exact job panel binding evidence" % label)
    if panel_identity.get("panel_id") != panel_binding.get("id") \
            or panel_identity.get("suite_token_hash_sha256") \
            != panel_binding.get("suite_token_hash_sha256") \
            or panel_identity.get("panel_receipt_sha256") \
            != receipt_binding.get("declared_receipt_sha256"):
        raise RootQualificationError(
            "%s dataset panel identity does not match the resolved job panel" % label)
    manifest_tokenizer = panel_identity.get("tokenizer") or {}
    for field in ("repository", "revision", "vocab_size", "files",
                  "identity_sha256"):
        if manifest_tokenizer.get(field) != tokenizer_binding.get(field):
            raise RootQualificationError(
                "%s dataset tokenizer %s does not match the resolved job tokenizer"
                % (label, field))
    expected_license = capture.get("weights_license")
    observed_license = identity.get("weights_license")
    if expected_license is None:
        if (observed_license is not None
                or identity.get("weights_license_file_sha256") is not None
                or identity.get("weights_license_file_bytes") is not None):
            raise RootQualificationError(
                "%s capture carries source-license bytes absent from job.json"
                % label)
    else:
        if (not isinstance(expected_license, dict)
                or set(expected_license) != {
                    "source_path", "dataset_path", "bytes", "sha256"}
                or expected_license.get("source_path") != "LICENSE"
                or expected_license.get("dataset_path") != "LICENSE"
                or not isinstance(observed_license, dict)
                or observed_license.get("source_file") != "LICENSE"
                or observed_license.get("dataset_path") != "LICENSE"
                or observed_license.get("bytes") != expected_license.get("bytes")
                or observed_license.get("sha256") != expected_license.get("sha256")
                or identity.get("weights_license_file_bytes")
                    != expected_license.get("bytes")
                or identity.get("weights_license_file_sha256")
                    != expected_license.get("sha256")):
            raise RootQualificationError(
                "%s source-license bytes do not match the exact job identity"
                % label)

    expected_allowlist = capture.get("unexpected_tensor_allowlist")
    observed_allowlist = identity.get("unexpected_tensor_allowlist")
    if expected_allowlist is None:
        if observed_allowlist is not None:
            raise RootQualificationError(
                "%s capture records unexpected tensors absent from job.json" % label)
        return
    if not isinstance(expected_allowlist, dict) \
            or not isinstance(observed_allowlist, dict):
        raise RootQualificationError(
            "%s capture lacks exact unexpected-tensor allowlist evidence" % label)
    if (not expected_allowlist.get("path")
            or not hex64(expected_allowlist.get("artifact_sha256"))
            or not hex64(
                expected_allowlist.get("canonical_sorted_names_sha256"))):
        raise RootQualificationError(
            "job unexpected-tensor allowlist identity is incomplete")
    expected_keys = observed_allowlist.get("expected_keys")
    if (not isinstance(expected_keys, list)
            or not expected_keys
            or any(not isinstance(name, str) or not name for name in expected_keys)
            or len(expected_keys) != len(set(expected_keys))):
        raise RootQualificationError(
            "%s unexpected-tensor evidence has invalid expected names" % label)
    names_sha256 = common.sha256_hex(json.dumps(
        sorted(expected_keys), separators=(",", ":"),
        ensure_ascii=False, allow_nan=False))
    if names_sha256 != expected_allowlist.get("canonical_sorted_names_sha256"):
        raise RootQualificationError(
            "%s unexpected-tensor names do not match the job name identity" % label)
    if observed_allowlist.get("artifact_sha256") \
            != expected_allowlist.get("artifact_sha256") \
            or observed_allowlist.get("canonical_sorted_names_sha256") \
            != expected_allowlist.get("canonical_sorted_names_sha256") \
            or observed_allowlist.get("exact_match") is not True \
            or not observed_allowlist.get("expected_keys") \
            or observed_allowlist.get("observed_keys") \
            != observed_allowlist.get("expected_keys") \
            or observed_allowlist.get("duplicate_observed_keys") != [] \
            or observed_allowlist.get("missing_keys") != [] \
            or observed_allowlist.get("extra_keys") != []:
        raise RootQualificationError(
            "%s unexpected-tensor evidence is not an exact job-bound match" % label)

def _load_qualification(
        path, *, job_path=None, dataset=None, repository=None):
    doc = _read_json_file(path, "root qualification receipt")
    if not isinstance(doc, dict):
        raise RootQualificationError("root qualification receipt must be an object")
    if set(doc) != _QUALIFICATION_KEYS:
        raise RootQualificationError(
            "root qualification receipt keys differ from the v1 contract (missing=%s, "
            "unexpected=%s)"
            % (sorted(_QUALIFICATION_KEYS - set(doc)), sorted(set(doc) - _QUALIFICATION_KEYS)))
    if doc.get("schema") != _QUALIFICATION_SCHEMA or not common.verify_seal(doc):
        raise RootQualificationError("root qualification receipt schema/seal is invalid")
    if not job_path:
        raise RootQualificationError(
            "root qualification validation requires the exact job.json")
    job = _read_json_file(job_path, "job.json")
    contract = doc.get("job_contract") or {}
    captures = doc.get("captures") or {}
    if not isinstance(contract, dict) or not isinstance(captures, dict):
        raise RootQualificationError(
            "root qualification job/capture identities must be objects")
    try:
        expected_job_sha = jobcontract.verify_job(job)
        expected_contract = jobcontract.root_qualification_contract(job)
    except jobcontract.JobContractError as exc:
        raise RootQualificationError(
            "root qualification job.json is invalid: %s" % exc)
    expected_job_file_sha = common.sha256_file(job_path)
    if (doc.get("canonical_job_sha256") != expected_job_sha
            or doc.get("job_file_sha256") != expected_job_file_sha
            or contract != expected_contract):
        raise RootQualificationError(
            "root qualification is not bound to the exact job.json")
    canonical = captures.get("canonical") or {}
    repeat = captures.get("repeat") or {}
    if not isinstance(canonical, dict) or not isinstance(repeat, dict):
        raise RootQualificationError(
            "root qualification capture identities must be objects")
    if (_HEX64.fullmatch(str(doc.get("canonical_job_sha256", ""))) is None
            or _HEX64.fullmatch(str(doc.get("job_file_sha256", ""))) is None):
        raise RootQualificationError(
            "root qualification job identities are invalid")
    for label, identity in (
            ("canonical", canonical), ("repeat", repeat)):
        _check_capture_job_contract(job, identity, label)
    if (not contract.get("dataset_id")
            or not doc.get("dataset_repository")
            or canonical.get("dataset_repository")
            != doc.get("dataset_repository")
            or repeat.get("dataset_repository")
            != doc.get("dataset_repository")
            or contract.get("dataset_repository") != doc.get("dataset_repository")
            or contract.get("publish_root_to")
            != doc.get("destination_repository")
            or canonical.get("dataset_id") != contract.get("dataset_id")
            or repeat.get("dataset_id") != contract.get("dataset_id")
            or _HEX64.fullmatch(
                str(canonical.get("capture_content_digest", ""))) is None
            or canonical.get("capture_content_digest")
            != repeat.get("capture_content_digest")
            or not canonical.get("process_label")
            or not repeat.get("process_label")
            or canonical.get("process_label") == repeat.get("process_label")):
        raise RootQualificationError(
            "root qualification has inconsistent job/capture identities")
    execution_kind = contract.get("execution_kind")
    image_reference = contract.get("container_image_reference")
    image_digest = contract.get("container_image_digest")
    if execution_kind == "runpod-ssh":
        if (not isinstance(image_reference, str)
                or re.fullmatch(
                    r".+@sha256:[0-9a-f]{64}", image_reference) is None
                or image_digest != image_reference.rsplit("@", 1)[1]):
            raise RootQualificationError(
                "root qualification container job contract is invalid")
        for label, identity in (("canonical", canonical), ("repeat", repeat)):
            runtime_container = identity.get("runtime_container")
            if (not isinstance(runtime_container, dict)
                    or runtime_container.get("image_digest") != image_digest
                    or runtime_container.get("image_reference")
                    != image_reference):
                raise RootQualificationError(
                    "%s qualification capture container differs from job contract"
                    % label)
    if doc.get("destination_repository") is not None \
            and doc.get("destination_repository") != doc.get("dataset_repository"):
        raise RootQualificationError(
            "root qualification publication destination differs from dataset identity")
    first = (doc.get("captures") or {}).get("canonical") or {}
    if dataset is not None:
        observed = _capture_identity(
            dataset, first.get("process_label"), "publish")
        if observed != first:
            raise RootQualificationError(
                "qualification canonical capture identity differs from the "
                "dataset selected for publication")
    if repository is not None:
        if doc.get("destination_repository") != repository \
                or doc.get("dataset_repository") != repository:
            raise RootQualificationError(
                "qualification dataset/destination repositories do not match publish "
                "repository %r" % repository)
    confirmation = doc.get("reproduction_confirmation") or {}
    if not (confirmation.get("two_fresh_processes") is True
            and confirmation.get("distinct_dataset_roots") is True
            and confirmation.get("both_independently_verified") is True
            and confirmation.get("exact_zero_comparison") is True
            and confirmation.get("canonical_dataset_only") is True):
        raise RootQualificationError("qualification lacks the required reproduction semantics")
    return doc


def cmd_qualify_root(args):
    if os.path.realpath(args.first) == os.path.realpath(args.repeat):
        return refuse("same_root", "root qualification needs two distinct dataset paths")
    if args.first_label == args.repeat_label:
        return refuse("same_process_label", "the two cold capture process labels must differ")
    try:
        job = _read_json_file(args.job, "job.json")
        try:
            canonical_job_sha256 = jobcontract.verify_job(job)
        except jobcontract.JobContractError as exc:
            raise RootQualificationError("job.json self-identity is invalid: %s" % exc)
        capture_job = job.get("capture") or {}
        if capture_job.get("preview_of") is not None \
                or capture_job.get("race") is not False:
            raise RootQualificationError(
                "preview/race roots are unsupported by the first safe paid path")
        dataset_repository = capture_job.get("dataset_repository")
        destination = capture_job.get("publish_root_to")
        weights_repo = (job.get("target") or {}).get("repo_id")
        weights_revision = (job.get("target") or {}).get("revision")
        if not dataset_repository:
            raise RootQualificationError("job.json has no capture.dataset_repository")
        if not weights_repo or dataset_repository == weights_repo:
            raise RootQualificationError(
                "target weights repository and intended dataset repository must be distinct")
        if destination is not None and destination != dataset_repository:
            raise RootQualificationError(
                "capture.publish_root_to must equal capture.dataset_repository when set")

        first = _capture_identity(args.first, args.first_label, "canonical")
        repeat = _capture_identity(args.repeat, args.repeat_label, "repeat")
        execution_kind = (job.get("execution_attempt") or {}).get("kind")
        environment = job.get("environment") or {}
        image_reference = environment.get("image")
        if execution_kind == "runpod-ssh":
            if (not isinstance(image_reference, str)
                    or re.fullmatch(
                        r".+@sha256:[0-9a-f]{64}", image_reference) is None):
                raise RootQualificationError(
                    "RunPod job environment image is not an immutable "
                    "container reference")
            image_digest = image_reference.rsplit("@", 1)[1]
        else:
            image_reference = environment.get("container_image")
            image_digest = environment.get("container_digest")
        replay_device = capture_job.get("replay_device")
        replay_dtype = capture_job.get("replay_dtype")
        vocab_chunk = capture_job.get("vocab_chunk")
        if replay_device is None or replay_dtype not in ("float32", "float64") \
                or not isinstance(vocab_chunk, int) or isinstance(vocab_chunk, bool) \
                or vocab_chunk <= 0:
            raise RootQualificationError(
                "job capture must explicitly bind replay_device, replay_dtype, "
                "and a positive integer vocab_chunk")
        for label, identity in (("canonical", first), ("repeat", repeat)):
            if identity["weights_repository"] != weights_repo \
                    or identity["weights_revision"] != weights_revision:
                raise RootQualificationError(
                    "%s capture weights identity does not match job target %s@%s"
                    % (label, weights_repo, weights_revision))
            _check_capture_job_contract(job, identity, label)
            runtime_container = identity["runtime_container"]
            if (execution_kind == "runpod-ssh"
                    and (runtime_container.get("image_digest") != image_digest
                         or runtime_container.get("image_reference")
                         != image_reference)):
                raise RootQualificationError(
                    "%s capture runtime container differs from job image"
                    % label)
        if first["dataset_id"] != repeat["dataset_id"] \
                or first["dataset_id"] != capture_job.get("dataset_id"):
            raise RootQualificationError(
                "both captures must carry job capture.dataset_id exactly")
        if first["capture_content_digest"] != repeat["capture_content_digest"]:
            raise RootQualificationError("the two independently captured content digests differ")
        if first["stack_fingerprint_sha256"] \
                != repeat["stack_fingerprint_sha256"] \
                or first["lane_identity_sha256"] != repeat["lane_identity_sha256"]:
            raise RootQualificationError(
                "the two cold captures do not share one runtime stack/lane identity")

        first_verify = _verified_report(args.first_verify, args.first, "canonical")
        repeat_verify = _verified_report(args.repeat_verify, args.repeat, "repeat")
        comparison = _read_json_file(args.comparison, "comparison receipt")
        comparison_report = dsvalidate.validate_receipt(comparison, args.comparison)
        if comparison_report.errors:
            raise RootQualificationError(
                "comparison receipt does not validate (%s)"
                % comparison_report.errors[0]["message"])
        if comparison.get("comparison_kind") != "reproduction_confirmation":
            raise RootQualificationError("comparison is not a reproduction confirmation")
        metric = comparison.get("metric") or {}
        if metric.get("name") != "mean_tokenwise_kld" \
                or not _positive_zero(metric.get("value")) \
                or not _positive_zero((comparison.get("kl") or {}).get("max")) \
                or comparison.get("top1_agreement") != 1.0:
            raise RootQualificationError(
                "comparison must have positive mean_kld=0.0, max_kld=0.0, "
                "and top-1 agreement=1.0")
        if (comparison.get("reference") or {}).get("dataset_sha256") \
                != first["dataset_sha256"] \
                or (comparison.get("candidate") or {}).get("dataset_sha256") \
                != repeat["dataset_sha256"]:
            raise RootQualificationError(
                "comparison sides do not bind canonical then repeat dataset")
        if (comparison.get("reference") or {}).get("label") != args.first_label \
                or (comparison.get("candidate") or {}).get("label") != args.repeat_label:
            raise RootQualificationError("comparison process labels do not match the captures")
        self_compare = comparison.get("self_compare") or {}
        comparator = comparison.get("comparator") or {}
        expected_replay_backend = (
            "numpy:cpu:float32" if replay_device == "numpy"
            else "torch:%s:%s" % (replay_device, replay_dtype))
        expected_device = "cpu" if replay_device == "numpy" else replay_device
        if comparator.get("replay_backend") != expected_replay_backend \
                or comparator.get("device") != expected_device \
                or comparator.get("vocab_chunk") != vocab_chunk \
                or not comparator.get("estimator_backend"):
            raise RootQualificationError(
                "comparison backend does not match the explicit job replay profile")
        if not (self_compare.get("capture_content_digest_equal") is True
                and self_compare.get("weights_identity_equal") is True
                and self_compare.get("asserted_exact_zero") is True
                and self_compare.get("force_compute_agreed") is True):
            raise RootQualificationError(
                "comparison lacks forced exact reproduction-confirmation semantics")

        receipt = common.seal({
            "schema": _QUALIFICATION_SCHEMA,
            "qualified_at": common.utcnow(),
            "canonical_job_sha256": canonical_job_sha256,
            "job_file_sha256": common.sha256_file(args.job),
            "dataset_repository": dataset_repository,
            "destination_repository": destination,
            "job_contract": jobcontract.root_qualification_contract(job),
            "captures": {"canonical": first, "repeat": repeat},
            "comparison": {
                "path": os.path.basename(args.comparison),
                "file_sha256": common.sha256_file(args.comparison),
                "receipt_sha256": comparison.get("receipt_sha256"),
                "comparison_kind": "reproduction_confirmation",
                "mean_kld": 0.0,
                "max_kld": 0.0,
                "top1_agreement": 1.0,
            },
            "comparator": {
                "requested_replay_device": replay_device,
                "requested_replay_dtype": replay_dtype,
                "requested_vocab_chunk": vocab_chunk,
                "device": comparator.get("device"),
                "replay_backend": comparator.get("replay_backend"),
                "estimator_backend": comparator.get("estimator_backend"),
                "accumulation_dtype": comparator.get("accumulation_dtype"),
                "vocab_chunk": comparator.get("vocab_chunk"),
                "force_compute_agreed": True,
            },
            "verification": {
                "canonical": first_verify,
                "repeat": repeat_verify,
            },
            "reproduction_confirmation": {
                "two_fresh_processes": True,
                "distinct_dataset_roots": True,
                "both_independently_verified": True,
                "exact_zero_comparison": True,
                "canonical_dataset_only": True,
            },
        })
        common.write_json(args.out, receipt)
        _load_qualification(args.out, job_path=args.job)
    except (RootQualificationError, F.FormatError) as exc:
        return refuse("root_qualification_refused", str(exc))
    emit("ROOT QUALIFIED %s" % args.first)
    emit("  repeat              %s" % args.repeat)
    emit("  comparison          exact +0.0 mean/max, top-1 1.0")
    emit("  receipt             %s" % args.out)
    return OK


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------
def _verify_publish_source_archive(
        archive_path, expected_sha256, expected_bytes,
        dataset_path, qualification_path, job_path):
    if not isinstance(expected_sha256, str) or not _HEX64.fullmatch(
            expected_sha256):
        raise RootQualificationError(
            "expected result archive SHA-256 must be exact lowercase 64-hex")
    if (isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int)
            or expected_bytes <= 0):
        raise RootQualificationError(
            "expected result archive byte count must be a positive integer")
    try:
        verified = resultsink.verify_archive(
            archive_path, expected_sha256=expected_sha256,
            expected_bytes=expected_bytes)
        archive_manifest = verified.get("manifest") or {}
        records = archive_manifest.get("files")
        if (archive_manifest.get("role") != "root"
                or archive_manifest.get("verb") != "capture"
                or not isinstance(records, list)):
            raise RootQualificationError(
                "result archive is not a completed root-capture proof")
        by_name = {
            record.get("path"): record
            for record in records if isinstance(record, dict)
        }

        def exact_file(member_name, local_path, label):
            record = by_name.get(member_name)
            if (not isinstance(record, dict)
                    or os.path.getsize(local_path) != record.get("bytes")
                    or common.sha256_file(local_path) != record.get("sha256")):
                raise RootQualificationError(
                    "%s bytes differ from the verified result archive" % label)

        exact_file("job.json", job_path, "job.json")
        exact_file(
            "receipts/root-qualification.json", qualification_path,
            "root qualification receipt")
        local_names = set(F.iter_dataset_files(dataset_path, exclude=()))
        archive_names = {
            name[len("dataset/"):]
            for name in by_name
            if isinstance(name, str) and name.startswith("dataset/")
        }
        if local_names != archive_names:
            raise RootQualificationError(
                "canonical dataset file set differs from verified result archive")
        exact_file(
            "dataset/" + F.MANIFEST_NAME,
            os.path.join(dataset_path, F.MANIFEST_NAME),
            "canonical dataset manifest")
        exact_file(
            "dataset/" + F.CHECKSUMS_NAME,
            os.path.join(dataset_path, F.CHECKSUMS_NAME),
            "canonical dataset checksums")
        if not any(
                isinstance(name, str) and name.startswith("dataset-repeat/")
                for name in by_name):
            raise RootQualificationError(
                "verified result archive lacks the independent repeat dataset")
    except RootQualificationError:
        raise
    except (resultsink.ArchiveError, F.FormatError, OSError) as exc:
        raise RootQualificationError(
            "verified result archive is invalid: %s" % exc) from exc
    canonical_records = {
        name[len("dataset/"):]: {
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        for name, record in by_name.items()
        if isinstance(name, str) and name.startswith("dataset/")
    }
    qualification_record = by_name["receipts/root-qualification.json"]
    verified = dict(verified)
    verified["canonical_dataset_records"] = canonical_records
    verified["canonical_dataset_bytes"] = sum(
        record["bytes"] for record in canonical_records.values())
    verified["qualification_record"] = {
        "bytes": qualification_record["bytes"],
        "sha256": qualification_record["sha256"],
    }
    return verified




def _private_publish_inputs(dataset_path, qualification_path, job_path):
    """Bind publication to one private extraction; same euid is the trust boundary."""
    if (os.path.islink(dataset_path)
            or os.path.islink(qualification_path)
            or os.path.islink(job_path)):
        raise RootQualificationError(
            "publication inputs must not be symlinks")
    dataset_real = os.path.realpath(dataset_path)
    extraction_root = os.path.dirname(dataset_real)
    expected_dataset = os.path.join(extraction_root, "dataset")
    expected_qualification = os.path.join(
        extraction_root, "receipts", "root-qualification.json")
    expected_job = os.path.join(extraction_root, "job.json")
    if (dataset_real != expected_dataset
            or os.path.realpath(qualification_path) != expected_qualification
            or os.path.realpath(job_path) != expected_job):
        raise RootQualificationError(
            "publication inputs must use one canonical verified extraction: "
            "root/dataset, root/receipts/root-qualification.json, root/job.json")
    root_info = os.lstat(extraction_root)
    if (stat.S_ISLNK(root_info.st_mode)
            or not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.geteuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700):
        raise RootQualificationError(
            "verified extraction root must be an owned, non-symlink mode-0700 "
            "directory")
    chain_error = common.private_directory_chain_error(
        extraction_root, owner_uid=os.geteuid())
    if chain_error:
        raise RootQualificationError(chain_error)
    for source, label in (
            (dataset_real, "canonical dataset"),
            (expected_qualification, "root qualification receipt"),
            (expected_job, "job.json")):
        if os.path.islink(source) or not os.path.exists(source):
            raise RootQualificationError(
                "%s must be a non-symlink member of the private extraction"
                % label)
    return dataset_real, expected_qualification, expected_job


def cmd_publish(args):
    qualification_path = getattr(args, "qualification", None)
    if not qualification_path:
        return refuse(
            "qualification_required",
            "publishing a root requires --qualification; an independently "
            "verified second capture and exact-zero comparison are mandatory")
    if args.private:
        return refuse(
            "public_publication_required",
            "canonical root publication must be anonymously readable; "
            "--private is refused")
    try:
        dataset_path, qualification_path, job_path = _private_publish_inputs(
            args.dataset, qualification_path, args.job)
    except (OSError, RootQualificationError) as exc:
        return refuse("publication_source_invalid", str(exc))
    return _cmd_publish_private_extraction(
        args, dataset_path, qualification_path, job_path)


def _cmd_publish_private_extraction(
        args, dataset_path, qualification_path, job_path):
    from fidelity import dshub

    try:
        qualification = _load_qualification(
            qualification_path, job_path=job_path,
            dataset=dataset_path, repository=args.repo)
    except (RootQualificationError, F.FormatError) as exc:
        return refuse("qualification_invalid", str(exc))

    local_manifest = F.load_manifest(dataset_path)
    local_dataset_sha256 = local_manifest.get(F.SEAL_FIELD)
    try:
        token = dshub.read_token(args.token_file)
    except (dshub.HubError, OSError) as exc:
        return refuse("publish_refused", str(exc))
    expected_head = getattr(args, "expected_head", None)
    try:
        source_archive = _verify_publish_source_archive(
            args.result_archive, args.expected_archive_sha256,
            args.expected_archive_bytes, dataset_path,
            qualification_path, job_path)
    except RootQualificationError as exc:
        return refuse("source_archive_invalid", str(exc))
    try:
        result = dshub.publish_dataset(
            dataset_path, args.repo, qualification_path,
            expected_head=expected_head, token=token, private=args.private,
            message=args.revision_message)
        if (result.get("repository") != args.repo
                or result.get("dataset_sha256") != local_dataset_sha256
                or result.get("private") is not False):
            raise dshub.HubError(
                "publisher result does not bind the public requested repository "
                "and local dataset")
        revision = result.get("revision")
        if not isinstance(revision, str) or not _HEX40.fullmatch(revision):
            raise dshub.HubError(
                "one-commit publication did not return an immutable 40-hex revision")
        evidence_rel = result.get("qualification_path_in_repo")
        if evidence_rel != "receipts/root-qualification.json":
            raise dshub.HubError(
                "publisher result does not bind the canonical qualification path")
    except (dshub.HubError, F.FormatError, ImportError, OSError) as exc:
        return refuse("publish_refused", str(exc))

    emit("published %s -> %s@%s (dataset_sha256 %s)"
         % (args.dataset, result["repository"], revision, local_dataset_sha256))
    emit("stream-verifying exact public bytes at the immutable revision...")
    try:
        published_dataset = dshub.verify_remote_dataset_exact(
            args.repo, revision,
            source_archive["canonical_dataset_records"],
            expected_dataset_sha256=local_dataset_sha256,
            max_total_bytes=source_archive["canonical_dataset_bytes"])
        qualification_record = source_archive["qualification_record"]
        evidence_bytes = dshub.fetch_exact_bytes(
            dshub.resolve_url(args.repo, revision, evidence_rel),
            qualification_record["bytes"],
            qualification_record["sha256"],
            token=None,
            max_bytes=resultsink.MAX_RETAINED_MEMBER_BYTES)
        published_qualification_file_sha256 = hashlib.sha256(
            evidence_bytes).hexdigest()
        with open(qualification_path, "rb") as handle:
            local_qualification_bytes = handle.read(
                qualification_record["bytes"] + 1)
        if evidence_bytes != local_qualification_bytes:
            return refuse(
                "publish_qualification_mismatch",
                "published qualification bytes differ from the verified "
                "source archive")
    except (dshub.HubError, OSError) as exc:
        return refuse("publish_verify_failed", str(exc))

    emit("the immutable published dataset and qualification verify")
    if getattr(args, "receipt", None):
        doc = common.seal({
            "schema": "fidelity.publish-root-receipt.v2",
            "repository": result["repository"],
            "revision": revision,
            "revision_immutable": True,
            "private": False,
            "dataset_sha256": local_dataset_sha256,
            "published_dataset_sha256": published_dataset["dataset_sha256"],
            "qualification_receipt_sha256": qualification.get("receipt_sha256"),
            "qualification_file_sha256": common.sha256_file(qualification_path),
            "published_qualification_file_sha256":
                published_qualification_file_sha256,
            "published_at": common.utcnow(),
            "verified_after_publish": True,
            "verified_anonymously": True,
            "verified_revision": revision,
            "result_archive_sha256": source_archive["archive_sha256"],
            "result_archive_bytes": source_archive["archive_bytes"],
        })
        common.write_json(args.receipt, doc)
        emit("publish receipt written to %s (immutable revision %s)"
             % (args.receipt, revision))
    return OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expected_head(text):
    if text == "absent":
        return None
    if _HEX40.fullmatch(text):
        return text
    raise argparse.ArgumentTypeError(
        "expected HEAD must be 'absent' or an exact lowercase 40-hex revision")


def _positive_int(text):
    """Require a positive integer for chunk sizing."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fidelity-dataset", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def common_dataset_flags(p):
        p.add_argument("--cache", help="where hf:// datasets are fetched")
        p.add_argument("--token-file", help="path to a file holding an HF token "
                                            "(never echoed, never committed)")

    p = sub.add_parser("capture", help="step 1/2: produce a fidelity dataset from weights")
    p.add_argument("--out", required=True)
    p.add_argument("--form", choices=F.FORMS, default="hidden")
    p.add_argument("--role", choices=F.ROLES, required=True)
    p.add_argument("--lane", choices=F.LANES, required=True)
    p.add_argument("--engine", choices=["sealed-lane", "hf-transformers"],
                   default="sealed-lane",
                   help="sealed-lane wraps engines/tools/hidden_replay.py + stream_score.py "
                        "(campaign-internal, GLM-5.3-Flash geometry only, and it writes a "
                        "capture work tree that something else must assemble); "
                        "hf-transformers wraps engines/tools/hf_capture.py, which runs any HF "
                        "causal LM and writes the SEALED DATASET at --out itself")
    p.add_argument("--work", help="capture working directory (default: <out>.capture)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate every input and the plan, exit 0 without a GPU")
    p.add_argument("--force", action="store_true")
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="everything after `--` is passed to the scorer verbatim")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("verify", help="seal + digest verification; stops at the first refusal")
    p.add_argument("dataset")
    p.add_argument("--verify-tensors", dest="verify_tensors", action="store_true",
                   default=True, help="(default) recompute every tensor_content_sha256")
    p.add_argument("--no-verify-tensors", dest="verify_tensors", action="store_false",
                   help="skip re-reading the tensors. The seal and checksums.txt still "
                        "verify, but a byte flipped inside a tensor whose checksums were "
                        "refreshed is NOT caught. Only for suites too large to re-read.")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--json")
    common_dataset_flags(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("validate", help="report EVERY failure; also validates a receipt")
    p.add_argument("dataset", nargs="?")
    p.add_argument("--receipt", help="validate a comparison receipt instead")
    p.add_argument("--verify-tensors", dest="verify_tensors", action="store_true",
                   default=True, help="(default) recompute every tensor_content_sha256")
    p.add_argument("--no-verify-tensors", dest="verify_tensors", action="store_false",
                   help="skip re-reading the tensors; see `verify --no-verify-tensors`")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--strict", action="store_true", help="warnings become errors")
    p.add_argument("--json")
    common_dataset_flags(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compare", help="step 3: compare two datasets")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--replay-device", default="numpy",
                   help="where the hidden->logit head matmul runs. 'numpy' (default) is "
                        "the published path: fp32 BLAS on the CPU. Any other value is a "
                        "torch device (e.g. 'cuda') and must equal --device. A GPU replay "
                        "is ~an order of magnitude faster and its fp32 GEMM accumulates in "
                        "a DIFFERENT ORDER, so it produces different last digits: rows "
                        "measured under different --replay-device values are not rankable "
                        "against each other. comparator.replay_backend records which ran.")
    p.add_argument("--replay-dtype", choices=("float32", "float64"), default="float32",
                   help="accumulation dtype for the replay matmul on a torch device. "
                        "float32 matches what the numpy path accumulates in; float64 is "
                        "more accurate AND more reproducible across backends, and is a "
                        "different measurement from either.")
    p.add_argument("--vocab-chunk", type=_positive_int,
                   help="positive output-column block size; final block may be partial")
    p.add_argument("--chunk-positions", type=_positive_int, default=128)
    p.add_argument("--head", help="head payload; only with --disclose-head-substitution")
    p.add_argument("--self-compare", action="store_true",
                   help="assert A and B are the same capture")
    p.add_argument("--force-compute", action="store_true",
                   help="run the math even when the hash proof answers, and assert agreement")
    p.add_argument("--allow-cross-lane", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--verify-tensors", dest="verify_tensors", action="store_true",
                   default=True, help="(default) recompute every tensor_content_sha256")
    p.add_argument("--no-verify-tensors", dest="verify_tensors", action="store_false",
                   help="skip re-reading the tensors; see `verify --no-verify-tensors`. "
                        "The receipt records which of the two ran.")
    p.add_argument("--disclose-head-substitution", action="store_true",
                   help="HEAD-1b override: advisory, downward bias, BLOCKING disclosure")
    p.add_argument("--emit-submission", action="store_true",
                   help="also write a registry submission; needs --submission-provenance")
    p.add_argument("--submission-provenance", metavar="FILE",
                   help="JSON with the artifact/panel/reference/measurer blocks a submission "
                        "needs and a dataset cannot know; skeleton: "
                        "`fidelity-dataset provenance-template`")
    p.add_argument("--measurer")
    p.add_argument("--reference-label")
    p.add_argument("--candidate-label")
    p.add_argument("--json")
    common_dataset_flags(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("adapt", help="translate a foreign capture artifact")
    p.add_argument("--source", choices=dsadapt.SOURCES, required=True)
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", help="token directory when the source ships them elsewhere")
    p.add_argument("--suite", help="suite directory (malaiwah-serving-v2)")
    p.add_argument("--head-dir", help="head directory (malaiwah-serving-v2)")
    p.add_argument("--dataset-id")
    p.add_argument("--name")
    p.add_argument("--role", choices=F.ROLES, default="root",
                   help="malaiwah-serving-v2 defaults to root; a k3 translation defaults to "
                        "derived, because ROOT-1 asserts things a k3 artifact never says")
    p.add_argument("--lane", choices=F.LANES, default="other")
    p.add_argument("--limit", type=int, help="adapt only the first N records")
    p.add_argument("--copy", action="store_true", help="copy tensors instead of hardlinking")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--recompute-content-digests", action="store_true",
                   help="read tensors to upgrade container digests to content digests")
    p.add_argument("--emit-k3-compat", action="store_true",
                   help="also write compat/, so the kimi-k3 comparator reads this dataset "
                        "UNMODIFIED. Metadata only -- three JSON files of relative aliases, "
                        "no tensor is duplicated. Written before the seal, so it is covered "
                        "by checksums.txt.")
    p.add_argument("--emit-dataset", action="store_true",
                   help="k3v1/k3v0-window: also WRITE a sealed v1 dataset, not just the "
                        "translation report. Needs the capture tensors present locally -- a "
                        "seal is computed over bytes, never fabricated.")
    p.set_defaults(func=cmd_adapt)

    p = sub.add_parser("verify-k3-compat",
                       help="check a compat/ tree against the dataset it describes")
    p.add_argument("dataset")
    p.set_defaults(func=cmd_verify_k3_compat)

    p = sub.add_parser("provenance-template",
                       help="skeleton for --submission-provenance")
    p.add_argument("--out", help="write it here instead of stdout")
    p.set_defaults(func=cmd_provenance_template)

    p = sub.add_parser("describe", help="print the identity card")
    p.add_argument("dataset")
    p.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    common_dataset_flags(p)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser(
        "qualify-root",
        help="bind two independently verified root captures and their exact-zero comparison")
    p.add_argument("--job", required=True)
    p.add_argument("--first", required=True)
    p.add_argument("--repeat", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--first-verify", required=True)
    p.add_argument("--repeat-verify", required=True)
    p.add_argument("--first-label", required=True)
    p.add_argument("--repeat-label", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_qualify_root)

    p = sub.add_parser("publish", help="upload a verified dataset to the Hub")
    p.add_argument("dataset")
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--expected-head", required=True, type=_expected_head,
        metavar="absent|40_HEX",
        help="optimistic publication authorization: destination must be absent "
             "or exactly this immutable HEAD")
    p.add_argument("--revision-message", default="publish fidelity dataset")
    p.add_argument("--qualification", required=True,
                   help="self-sealed root qualification receipt; published and refetched "
                        "at the returned immutable revision")
    p.add_argument(
        "--job", required=True,
        help="exact job.json whose canonical identity, file digest, target, "
             "profile and panel contract the qualification must match")
    p.add_argument(
        "--result-archive", required=True,
        help="original retrieved result.tar.gz containing the exact job, both "
             "verified captures, comparison, and qualification")
    p.add_argument(
        "--expected-archive-sha256", required=True,
        help="exact on-box archive SHA-256 reported before transfer")
    p.add_argument(
        "--expected-archive-bytes", required=True, type=_positive_int,
        help="exact on-box archive byte count reported before transfer")
    p.add_argument(
        "--receipt",
        help="write a sealed publish receipt here only after every immutable "
             "public member stream-verifies against the source archive")
    common_dataset_flags(p)
    p.set_defaults(func=cmd_publish)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return USAGE
    if args.command == "validate" and not args.dataset and not args.receipt:
        emit("validate needs a DATASET or --receipt")
        return USAGE
    try:
        return args.func(args)
    except Exception as exc:                                    # noqa: BLE001
        # A hub failure is a REFUSAL with a reason, not a traceback. Every
        # hf:// path -- verify, validate, compare, describe, adapt, publish --
        # used to exit 1 with twenty lines of stack above the useful line.
        from fidelity import dshub
        if not isinstance(exc, dshub.HubError):
            raise
        return refuse("hub_error", str(exc),
                      "a 404 on fidelity-dataset.json means the repo is not a fidelity "
                      "dataset -- translate it with `adapt` first. A 401/403 means the "
                      "token is missing or lacks access: pass --token-file.")


if __name__ == "__main__":
    sys.exit(main())
