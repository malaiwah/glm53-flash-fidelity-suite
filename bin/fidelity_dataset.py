#!/usr/bin/env python3
"""The three-step fidelity tool: capture, verify, compare.

    step 1  capture   reference weights + panel  -> fidelity dataset A   [publish: REQUIRED for a root]
    step 2  capture   quantized weights + panel  -> fidelity dataset B   [publish: OPTIONAL]
    step 3  compare   A, B                       -> KLD + determinism + a registry receipt
                      A, A                       -> reproduction confirmation, exactly 0.0

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
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fidelity import common  # noqa: E402
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
    emit("  backend             %s" % receipt["comparator"].get("estimator_backend"))
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
        tool = os.path.join(REPO, "k6", "tools", "hf_capture.py")
        python = os.environ.get("FIDELITY_PYTHON", sys.executable)
        command = ([python, tool, "--out", args.out, "--role", args.role,
                    "--lane", args.lane] + (["--force"] if args.force else [])
                   + passthrough)
        emit("capture plan")
        emit("  engine          hf-transformers (k6/tools/hf_capture.py)")
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
                      "these are inherited from k6/tools/hidden_replay.py::run_capture")

    if os.path.exists(args.out) and not args.force:
        return refuse("destination_exists", "%s exists" % args.out, "pass --force")

    tool = os.path.join(REPO, "k6", "tools",
                        "hidden_replay.py" if args.form == "hidden" else "stream_score.py")
    if not os.path.isfile(tool):
        return refuse("engine_missing",
                      "%s does not exist in this checkout" % os.path.relpath(tool, REPO),
                      "k6/tools/hidden_replay.py is campaign-internal and is not part of the "
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
# publish
# ---------------------------------------------------------------------------


def cmd_publish(args):
    from fidelity import dshub

    token = dshub.read_token(args.token_file)
    try:
        result = dshub.publish_dataset(args.dataset, args.repo, token=token,
                                       private=args.private,
                                       message=args.revision_message)
    except dshub.HubError as exc:
        return refuse("publish_refused", str(exc))
    emit("published %s -> %s (dataset_sha256 %s)"
         % (args.dataset, result["repository"], result["dataset_sha256"]))
    emit("re-verifying the published copy...")
    cache = os.path.join(REPO, "fidelity-runs", "datasets", "verify-after-publish")
    dshub.fetch_dataset("hf://%s" % args.repo, cache, token=token)
    report = dsvalidate.validate_dataset(cache, verify_tensors=True)
    if report.errors:
        _print_report(report, verbose=False)
        return refuse("publish_verify_failed", "the fetched copy does not verify")
    emit("the fetched copy verifies")
    return OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
                   help="sealed-lane wraps k6/tools/hidden_replay.py + stream_score.py "
                        "(campaign-internal, GLM-5.3-Flash geometry only, and it writes a "
                        "capture work tree that something else must assemble); "
                        "hf-transformers wraps k6/tools/hf_capture.py, which runs any HF "
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
    p.add_argument("--vocab-chunk", type=int,
                   help="must divide vocab_size exactly (9680 for GLM-5.3-Flash)")
    p.add_argument("--chunk-positions", type=int, default=128)
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

    p = sub.add_parser("publish", help="upload a verified dataset to the Hub")
    p.add_argument("dataset")
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--revision-message", default="publish fidelity dataset")
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
