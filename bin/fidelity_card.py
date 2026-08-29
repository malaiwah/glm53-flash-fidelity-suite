#!/usr/bin/env python3
"""Generate and validate the HF card fidelity-provenance annotation.

    bin/fidelity-card annotate --card README.md --role quant \
        --measurement-id measurement--glm53.k6-6bpw.brandonmusic-final25 ... \
        --out README.annotated.md --diff

    bin/fidelity-card validate --card README.md [--offline]

Two layers go in: one conformant `model-index` entry (so every HF tool sees the
number) and one `x_fidelity:` block (for what model-index structurally cannot
express).  Three axes come out: the live Hub validator, a `huggingface_hub`
round-trip, and our own XC-1..XC-5 cross-checks against the registry.

`annotate` never rewrites the card BODY and never invents a head digest.
Publishing is a separate, permissioned act: this tool writes files.

Spec: docs/CARD-ANNOTATION-SPEC.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fidelity import cardmeta, common  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK, WARN, REFUSED, USAGE = 0, 2, 3, 4


def emit(text=""):
    print(common.redact(str(text)))


def _read_card(ref):
    if ref.startswith("hf://"):
        from fidelity import dshub

        repo, revision = dshub.parse_ref(ref)
        url = dshub.resolve_url(repo, revision, "README.md", repo_type="models")
        return dshub._get(url)
    with open(ref, "r", encoding="utf-8") as handle:
        return handle.read()


def cmd_annotate(args):
    registry = cardmeta.load_registry(args.registry)
    measurement_ids = list(args.measurement_id or [])
    if not measurement_ids and args.artifact_id:
        measurement_ids = sorted(
            mid for mid, row in registry["measurements"].items()
            if row.get("artifact_ref") == args.artifact_id and row.get("status") == "published")
        emit("resolved %d published measurement(s) for %s"
             % (len(measurement_ids), args.artifact_id))
    if args.role == "quant" and not measurement_ids:
        emit("REFUSED: a quant card needs at least one registry measurement")
        emit("  remedy: the three steps run BEFORE the card. capture -> compare "
             "--emit-submission --submission-provenance FILE -> get the row merged into "
             "the registry (registry/CONTRIBUTING.md) -> then annotate. There is no "
             "receipt-to-card path: a card cites REGISTRY ids, not local receipts.")
        return REFUSED

    text = _read_card(args.card)
    try:
        if args.role == "fidelity-dataset":
            if not args.fidelity_dataset_root:
                emit("REFUSED: a fidelity-dataset card is built FROM the dataset")
                emit("  remedy: pass --fidelity-dataset-root DIR (the directory holding "
                     "fidelity-dataset.json). Every value the card needs -- captured_model, "
                     "form, panel, head, lane, seal, scope_digest -- is in that manifest, and "
                     "none of them is derivable from the registry.")
                return REFUSED
            repository, _, revision = (args.fidelity_dataset or "").partition("@")
            x_fidelity = cardmeta.build_dataset_x_fidelity(
                args.fidelity_dataset_root, repository=repository or None,
                revision=revision or None)
            merged = cardmeta.merge_card(
                text, model_index=None, x_fidelity=x_fidelity,
                datasets=list(args.dataset or []),
                metrics=(), tags=("fidelity", "fidelity-provenance", "fidelity-dataset"))
            return _finish_annotate(args, text, merged, registry, "dataset")

        model_index = None
        if args.role in ("quant", "root") and measurement_ids:
            model_index = cardmeta.build_model_index(registry, measurement_ids, args.model_name)
        fidelity_dataset = None
        if args.fidelity_dataset:
            repo, _, revision = args.fidelity_dataset.partition("@")
            fidelity_dataset = {
                "repository": repo,
                "revision": str(revision) if revision else None,
                "dataset_sha256": args.dataset_sha256,
                "capture_content_digest": args.capture_content_digest,
                "form": args.form,
                "role": args.role,
            }
        # XC/GEN: derive what the registry already knows rather than requiring the
        # operator to retype it -- and when a hop does not resolve, WARN with the
        # flag that would supply it instead of writing a silent null.
        reference_model, reference_revision = args.reference_model, args.reference_revision
        if not reference_model and measurement_ids:
            derived_model, derived_revision, notes = cardmeta.reference_identity(
                registry, measurement_ids)
            reference_model = derived_model
            if not args.reference_revision:
                reference_revision = derived_revision
            if derived_model:
                emit("derived reference_model=%s revision=%s from the registry"
                     % (derived_model, derived_revision))
            for note in notes:
                emit("  warn  %s" % note)
        for label, value, flag in (("reference_model", reference_model, "--reference-model"),
                                   ("reference_revision", reference_revision,
                                    "--reference-revision"),
                                   ("head.lm_head_tensor_content_sha256",
                                    args.head_content_sha256, "--head-content-sha256")):
            if not value:
                emit("  warn  x_fidelity.%s will be null; nothing in the registry supplies it. "
                     "Pass %s if you have it." % (label, flag))
        x_fidelity = cardmeta.build_x_fidelity(
            registry, role=args.role, measurement_ids=measurement_ids,
            artifact_id=args.artifact_id,
            reference_model=reference_model,
            reference_revision=reference_revision,
            fidelity_dataset=fidelity_dataset,
            head_content_sha256=args.head_content_sha256,
            head_file_sha256=args.head_file_sha256,
            final_norm_file_sha256=args.final_norm_file_sha256,
            equality_receipt=args.equality_receipt)
        datasets = list(args.dataset or [])
        if fidelity_dataset and fidelity_dataset["repository"] not in datasets:
            datasets.append(fidelity_dataset["repository"])
        merged = cardmeta.merge_card(
            text, model_index=model_index, x_fidelity=x_fidelity,
            datasets=datasets,
            metrics=("kl_divergence", "top1_agreement"),
            tags=("fidelity", "kl-divergence", "fidelity-provenance"),
            base_model=args.base_model,
            base_model_relation=("quantized" if args.role == "quant" and args.base_model
                                 else None))
    except cardmeta.CardError as exc:
        emit("REFUSED: %s" % exc)
        return REFUSED

    return _finish_annotate(args, text, merged, registry, "model", measurement_ids)


def _finish_annotate(args, text, merged, registry, repo_type, measurement_ids=()):
    if args.diff:
        for line in difflib.unified_diff(text.splitlines(True), merged.splitlines(True),
                                         fromfile=args.card, tofile=(args.out or args.card),
                                         n=2):
            emit(line.rstrip("\n"))
    destination = args.out or (args.card if args.in_place else None)
    if destination:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(merged)
        emit("wrote %s" % destination)
    if args.eval_results_v2:
        rows = cardmeta.build_eval_results_v2(registry, measurement_ids,
                                              args.base_model or args.model_name)
        path = args.eval_results_v2
        import yaml

        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(rows, handle, sort_keys=False, allow_unicode=True)
        emit("wrote %s (OFF BY DEFAULT: the format cannot express units or direction, so a "
             "leaderboard would sort KLD backwards)" % path)
    # GEN-9: annotate ALWAYS checks its own output. A generator that writes an
    # invalid card and exits 0 is worse than one that refuses -- the caller only
    # finds out when the Hub, or a reader, does. `--validate` now controls how
    # LOUD the check is, not whether it runs.
    our_axis = cardmeta.validate_card(merged, registry, offline=True, repo_type=repo_type)
    self_errors = [e for axis in our_axis["axes"] if axis["axis"] == "ours"
                   for e in (axis.get("errors") or [])]
    if self_errors:
        emit("REFUSED: the card this run produced does not satisfy its own validator")
        for error in self_errors:
            emit("          %s" % error)
        emit("  remedy: nothing was published. Fix the inputs above and re-run; "
             "`fidelity-card validate --card %s` re-checks all three axes."
             % (args.out or args.card))
        return REFUSED
    if args.validate:
        return _validate_text(merged, registry, args.offline, repo_type)
    emit("self-check        PASS (ours; run `validate` for the Hub and round-trip axes)")
    return OK


def _validate_text(text, registry, offline, repo_type):
    report = cardmeta.validate_card(text, registry, offline=offline, repo_type=repo_type)
    for axis in report["axes"]:
        if axis.get("ran") is False:
            emit("  SKIP  %-10s %s" % (axis["axis"], axis.get("skipped")))
        elif axis["ok"]:
            detail = ""
            if axis["axis"] == "roundtrip":
                info = axis.get("detail") or {}
                detail = ("  (%s eval results, model_name=%s, x_fidelity preserved=%s)"
                          % (info.get("eval_results"), info.get("model_name"),
                             info.get("x_fidelity_present")))
            emit("  PASS  %-10s clean%s" % (axis["axis"], detail))
        else:
            emit("  FAIL  %-10s" % axis["axis"])
            for error in axis.get("errors") or []:
                emit("          %s" % error)
    for warning in report["warnings"]:
        emit("  warn  %s" % warning)
    if report["errors"]:
        return REFUSED
    return WARN if report["skipped_axes"] else OK


def cmd_validate(args):
    registry = cardmeta.load_registry(args.registry)
    text = _read_card(args.card)
    emit("validating %s (%s)" % (args.card, args.repo_type))
    result = _validate_text(text, registry, args.offline, args.repo_type)
    if args.json:
        report = cardmeta.validate_card(text, registry, offline=args.offline,
                                        repo_type=args.repo_type)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        emit("report -> %s" % args.json)
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="fidelity-card", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("annotate", help="merge the two layers into an existing card")
    p.add_argument("--card", required=True, help="README.md path or hf://<repo>[@rev]")
    p.add_argument("--role", choices=cardmeta.ROLES, required=True)
    p.add_argument("--measurement-id", action="append")
    p.add_argument("--artifact-id")
    p.add_argument("--model-name", help="the model-index entry name")
    p.add_argument("--registry", help="registry clone (default: ./registry)")
    p.add_argument("--base-model")
    p.add_argument("--reference-model")
    p.add_argument("--reference-revision")
    p.add_argument("--dataset", action="append", help="add to top-level datasets:")
    p.add_argument("--fidelity-dataset", help="REPO[@REV] for x_fidelity.fidelity_dataset")
    p.add_argument("--fidelity-dataset-root", metavar="DIR",
                   help="a local fidelity dataset directory; REQUIRED for "
                        "--role fidelity-dataset, which is built entirely from its manifest")
    p.add_argument("--dataset-sha256")
    p.add_argument("--capture-content-digest")
    p.add_argument("--form", choices=("hidden", "logit"))
    p.add_argument("--head-content-sha256",
                   help="NEVER invented; omitted -> replay_permitted false (GEN-8)")
    p.add_argument("--head-file-sha256")
    p.add_argument("--final-norm-file-sha256")
    p.add_argument("--equality-receipt")
    p.add_argument("--eval-results-v2", metavar="PATH",
                   help="ALSO emit .eval_results/fidelity.yaml (off by default)")
    p.add_argument("--out")
    p.add_argument("--in-place", action="store_true")
    p.add_argument("--diff", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("validate", help="three axes: Hub, round-trip, ours")
    p.add_argument("--card", required=True)
    p.add_argument("--registry")
    p.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    p.add_argument("--offline", action="store_true",
                   help="skip the live Hub axis; the report SAYS it was skipped")
    p.add_argument("--json")
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return USAGE
    if args.command == "annotate" and not args.model_name:
        args.model_name = (args.base_model or args.card).rsplit("/", 1)[-1].replace(".md", "")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
