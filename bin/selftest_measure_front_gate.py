#!/usr/bin/env python3
"""Selftest: `bin/measure`'s front gate hands a quant of a family with a
published root to the candidate route instead of `--panel-descriptor`.

The defect (cloud usability review, 2026-09-05, S1-1): for every GLM-5.3 quant
`bin/measure <url>` resolved the right panel and reference, then refused with
"pass measure-local --panel-descriptor" -- the wrong tool for a family whose
panel is in this checkout and whose reference is a published root dataset.
The route that produced every GLM-5.3 row (`measure-cloud --role root
--candidate-scope ... --reference-dataset OWNER/REPO@40HEX`) was named
nowhere the front gate pointed.

Rungs (offline; the registry is the local clone, no Hub call is made):
  R1  candidate_handoff: in-tree panel + hf dataset uri + reference sha -> handoff
  R2  candidate_handoff: no in-tree panel / no hf uri / no sha -> None (old refusal stands)
  R3  handoff_refusal: the command carries the resolved values (model, revision,
      panel dir, reference repo, codec and bits from the sniff) and names 3b
  R4  handoff_refusal without Hub access leaves the dataset head as a marked
      placeholder rather than inventing one
  R5  the local registry clone resolves the GLM-5.3 family to
      malaiwah/glm53-fidelity-root-v1 with the published dataset_sha256

Stock python3, stdlib only, $0.00.
"""

import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUITE_ROOT / "bin"))

import measure_one  # noqa: E402
from fidelity import registry_client as RC  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


class StubRegistry:
    def __init__(self, panels, references):
        self.collections = {"panels": panels, "references": references}


class StubSurface:
    codec_family = "exl3-mcg"
    bits = 3.25


PANEL = "panel--glm53.malaiwah.corpus5x5-v1"
REF = "reference--malaiwah.glm-5.3-bf16-hf.corpus5x5-v1"
SHA = "6b8d3a7bdf934f18fc819cc72d85c5385c3351fa50a8c9c2612dd9a29172a4a4"
URI = "https://huggingface.co/datasets/malaiwah/glm53-fidelity-root-v1"


def main():
    good = StubRegistry({PANEL: {"availability": {"uri": URI}}},
                        {REF: {"capture": {"capture_receipt_sha256": SHA}}})
    h = measure_one.candidate_handoff(good, PANEL, REF)
    check("R1: in-tree panel + hf dataset uri + reference sha -> handoff",
          h is not None and h["repo"] == "malaiwah/glm53-fidelity-root-v1"
          and h["dataset_sha256"] == SHA and (h["panel_dir"] / "panel.json").is_file(),
          repr(h))

    check("R2: panel not in this checkout -> None",
          measure_one.candidate_handoff(good, "panel--nowhere.x.y", REF) is None)
    no_uri = StubRegistry({PANEL: {"availability": {"uri": "https://example.org/panel"}}},
                          good.collections["references"])
    check("R2: panel availability not an hf dataset -> None",
          measure_one.candidate_handoff(no_uri, PANEL, REF) is None)
    no_sha = StubRegistry(good.collections["panels"], {REF: {"capture": {}}})
    check("R2: reference without capture_receipt_sha256 -> None",
          measure_one.candidate_handoff(no_sha, PANEL, REF) is None)

    target = {"repo": "davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "revision": None, "path": None}
    ref = measure_one.handoff_refusal(h, target, "6d" * 20, surface=StubSurface(), hf_ok=False)
    text = "\n".join([ref.reason] + ref.advice)
    check("R3: refusal names the candidate route and never rents",
          "candidate route" in ref.reason and "never rents" in ref.reason)
    for needle in ("bin/measure-cloud --provider runpod --role root",
                   "--model davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw --revision " + "6d" * 20,
                   "--panel-dir engines/panels/" + PANEL,
                   "--reference-dataset malaiwah/glm53-fidelity-root-v1@",
                   "--candidate-codec exl3-mcg --candidate-bits 3.25",
                   "--candidate-scope", "--dry-run", "3b"):
        check("R3: command carries %s" % needle[:48], needle in text)
    check("R3: no --panel-descriptor remedy", "--panel-descriptor" not in text)

    check("R4: without Hub access the dataset head is a marked placeholder",
          "@<40-hex head of malaiwah/glm53-fidelity-root-v1>" in text and SHA[:16] in text)
    bare = measure_one.handoff_refusal(h, target, None, surface=None, hf_ok=False)
    bare_text = "\n".join(bare.advice)
    check("R4: without a sniff, codec and bits are placeholders, revision too",
          "--candidate-codec <registry numeric_format> --candidate-bits <declared bpw>" in bare_text
          and "--revision <40-hex>" in bare_text)

    try:
        reg = RC.load("local")
    except RC.RegistryUnavailable as exc:
        check("R5: local registry clone loads", False, str(exc))
    else:
        real = measure_one.candidate_handoff(reg, PANEL, REF)
        check("R5: the GLM-5.3 family resolves to the published root dataset",
              real is not None and real["repo"] == "malaiwah/glm53-fidelity-root-v1"
              and real["dataset_sha256"] == SHA, repr(real))

    print()
    if failures:
        print("selftest_measure_front_gate: %d FAILED" % len(failures))
        return 1
    print("selftest_measure_front_gate: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
