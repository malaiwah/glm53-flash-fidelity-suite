#!/usr/bin/env python3
"""Selftest: the candidate route as a human reaches it -- `--role candidate`,
the codec vocabulary, the bits cross-check and the two refusals that must
print the route.

The defects (cloud usability review, 2026-09-05): the route that produced
every GLM-5.3 quant row was spelled `--role root --candidate-scope ...` under
the root-capture help group (S1-1); `--candidate-codec` was free text and had
drifted three ways across real jobs, one of them (`exl3`) outside the
registry's numeric_format enum (S2-5); `--candidate-bits` re-declared what the
target gate had already read from the release without a cross-check (S2-5);
and the quantized-root refusal said "Point --model at the unquantized release"
to the human who was one flag away from the candidate route (S1-1).

Rungs (offline, $0.00; the Hub is never contacted -- config reads are stubbed):
  R1  --role candidate is accepted by the parser; --role root + the four flags
      still parses (the m-* scripts keep running)
  R2  main(): --role candidate without all four flags refuses (exit 3) and
      names the missing ones; with all four it is normalised to the root path
  R3  _candidate_block: a codec outside registry numeric_format refuses and
      lists the vocabulary; an in-enum codec passes that gate
  R4  _candidate_block: --candidate-bits disagreeing with the release's
      declaration refuses naming both; agreeing passes; fp8 e4m3 = 8
  R5  _refuse_quantized_root: the refusal carries the candidate command with
      --model/--revision, the reference dataset (when passed) and the codec
      and bits read from the config
  R6  --help files the four flags under the candidate group and the epilog
      shows the candidate example and the authored root numbers
"""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import measure_cloud as MC  # noqa: E402
from fidelity.common import Console  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


def main_code(argv):
    """MC.main's exit code; an argparse rejection is reported as exit 2."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = MC.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue()


def refusal_of(fn):
    try:
        fn()
    except MC.Refusal as exc:
        return exc
    return None


def refusal_text(exc):
    return "\n".join([exc.reason] + list(getattr(exc, "advice", None) or []))


BASE = ["--provider", "runpod", "--model", "o/q", "--revision", "a" * 40,
        "--panel-dir", "engines/panels/panel--glm53.malaiwah.corpus5x5-v1",
        "--dataset-id", "fidelity--glm53.h.quant.q", "--measurer", "h",
        "--max-cost", "45", "--max-runtime", "3h30m", "--out", "/tmp/x", "--dry-run"]
FOUR = ["--candidate-scope", "s.json", "--candidate-codec", "exl3-mcg",
        "--candidate-bits", "3.25", "--reference-dataset", "o/r@" + "b" * 40]


class Target:
    repo_id, revision = "o/q", "a" * 40


class Surface:
    surface = "exl3hf"


def main():
    parser = MC.build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            cand = parser.parse_args(["--role", "candidate"] + BASE + FOUR)
            root = parser.parse_args(["--role", "root"] + BASE + FOUR)
            ok = cand.role == "candidate" and root.role == "root"
        except SystemExit:
            ok = False
    check("R1: --role candidate and --role root + four flags both parse", ok)

    code, text = main_code(["--role", "candidate"] + BASE + FOUR[:2])
    check("R2: --role candidate without the four flags refuses with exit 3",
          code == MC.EXIT_REFUSED and "--candidate-codec" in text
          and "--reference-dataset" in text and "--candidate-scope" not in text.split("requires")[-1],
          text[-300:])

    # R2b: with the four flags the alias is normalised to the root path: the
    # next refusal main() reaches is root's own (--out must not exist yet).
    code, text = main_code(["--role", "candidate"] + BASE[:-3] + ["--out", "/", "--dry-run"] + FOUR)
    check("R2: with the four flags --role candidate reaches the root path",
          code == MC.EXIT_REFUSED and "--role candidate requires" not in text,
          text[-300:])

    # R3/R4 through _candidate_block with a stub plan carrying the decode plan
    # the target gate would have bound, and a real scope file.
    scope = ROOT / "engines" / "scopes" / "scope--dy325-exl3.json"
    check("R3: committed scope fixture present", scope.is_file())
    con = Console()

    def block(codec, bits, decode):
        import argparse
        args = argparse.Namespace(
            candidate_scope=str(scope), candidate_codec=codec, candidate_bits=bits,
            reference_dataset="o/r@" + "b" * 40, dataset_repository="o/q-ds")
        plan = {"_candidate_decode": decode}
        return MC._candidate_block(args, plan, con, binding_panel={})

    trellis = {"method": MC.CANDIDATE_DECODE_METHOD_TRELLIS,
               "quantization_config": {"quant_method": "exl3", "codebook": "mcg", "bits": 3.25}}
    with contextlib.redirect_stdout(io.StringIO()):
        exc = refusal_of(lambda: block("exl3", 3.25, trellis))
    check("R3: codec outside numeric_format refuses and lists the vocabulary",
          exc is not None and "numeric_format" in exc.reason
          and "exl3-mcg" in refusal_text(exc) and "fp8_e4m3" in refusal_text(exc),
          refusal_text(exc) if exc else "accepted")

    with contextlib.redirect_stdout(io.StringIO()):
        exc = refusal_of(lambda: block("exl3-mcg", 3.0, trellis))
    check("R4: bits disagreeing with the declaration refuse naming both",
          exc is not None and "3.25" in exc.reason and "--candidate-bits 3" in refusal_text(exc)
          and "disagrees" in exc.reason, refusal_text(exc) if exc else "accepted")

    # Agreeing codec and bits pass BOTH gates: the next thing the block does
    # is fetch the reference manifest (the first Hub touch), stubbed to raise
    # a marker so the rung proves the gates were passed, not skipped.
    class Passed(Exception):
        pass

    def marker(*a, **k):
        raise Passed()
    original_manifest = MC._candidate_reference_manifest
    MC._candidate_reference_manifest = marker
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                block("exl3-mcg", 3.25, trellis)
                passed_gates = False
            except Passed:
                passed_gates = True
            except MC.Refusal as exc:
                passed_gates = False
                print(refusal_text(exc))
    finally:
        MC._candidate_reference_manifest = original_manifest
    check("R4: agreeing codec and bits pass the two gates", passed_gates)
    fp8 = {"method": MC.CANDIDATE_DECODE_METHOD,
           "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                   "weight_block_size": [128, 128]}}
    with contextlib.redirect_stdout(io.StringIO()):
        exc = refusal_of(lambda: block("fp8_e4m3", 4.0, fp8))
    check("R4: fp8 e4m3 declares 8; --candidate-bits 4 refuses",
          exc is not None and "declaration 8" in exc.reason,
          refusal_text(exc) if exc else "accepted")

    # R5: the quantized-root refusal prints the candidate command.
    import argparse
    args = argparse.Namespace(
        candidate_scope=None, designated_reference=False,
        reference_dataset="malaiwah/glm53-fidelity-root-v1@" + "9" * 40,
        panel_dir="engines/panels/panel--glm53.malaiwah.corpus5x5-v1",
        dataset_id="fidelity--glm53.h.quant.q", measurer="h")
    original = MC.fetch_json
    MC.fetch_json = lambda *a, **k: {"quantization_config": {"quant_method": "exl3", "bits": 4,
                                                             "codebook": "mcg"}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exc = refusal_of(lambda: MC._refuse_quantized_root(
                con, Target(), Surface(), {}, args=args))
    finally:
        MC.fetch_json = original
    text = refusal_text(exc) if exc else ""
    check("R5: quantized-root refusal names the candidate route",
          exc is not None and "--role candidate" in text and "3b" in text)
    for needle in ("--model o/q --revision " + "a" * 40,
                   "--reference-dataset malaiwah/glm53-fidelity-root-v1@" + "9" * 40,
                   "--candidate-codec exl3-mcg --candidate-bits 4",
                   "--panel-dir engines/panels/panel--glm53.malaiwah.corpus5x5-v1",
                   "--measurer h"):
        check("R5: command carries %s" % needle[:44], needle in text)
    check("R5: still tells a root author to point at the unquantized release",
          "unquantized release" in text)

    help_text = parser.format_help()
    cand_start = help_text.find("candidate measurement (--role candidate")
    root_start = help_text.find("root capture (--role root)")
    quant_start = help_text.find("quant measurement (--role quant)")
    cand_section = help_text[cand_start:quant_start] if cand_start >= 0 else ""
    check("R6: the four flags are filed under the candidate group",
          cand_start >= 0
          and all(f in cand_section for f in ("--candidate-scope", "--candidate-codec",
                                                "--candidate-bits", "--reference-dataset"))
          and "--candidate-scope" not in help_text[root_start:cand_start])
    check("R6: the epilog shows the candidate example and the authored root numbers",
          "--role candidate" in help_text and "--max-cost 65 --max-runtime 7h30m" in help_text
          and "block-scaled FP8" not in help_text[root_start:cand_start])

    print()
    if failures:
        print("selftest_measure_cloud_candidate: %d FAILED" % len(failures))
        return 1
    print("selftest_measure_cloud_candidate: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
