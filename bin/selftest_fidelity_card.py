#!/usr/bin/env python3
"""T7 -- the HF card fidelity-provenance annotation.

    K1   the shipped example cards: schema clean, round-trip identical
    K2   XC-1..XC-5 against registry/data/measurements.jsonl
    K3   two model-index entries are refused
    K4   two results sharing the 5-tuple merge key are refused
    K5   lane only in dataset.args is refused
    K6   an all-digit unquoted revision is refused
    K7   replay_permitted: true with a null head content digest is refused
    K8   quantization_attributable whose floor_lane != lane is refused
    K9   a result for a measurement the registry does not have is refused
    K10  base_model_relation: fidelity-reference is refused (the enum has 4 values)
    K11  pre-existing unknown top-level keys survive annotate
    K12  a hand-set `verified: true` is stripped
    K13  the live Hub validate-yaml axis (SKIPPED under --offline, and the skip is REPORTED)

`--offline` skips exactly one axis and says so.
"""

from __future__ import annotations

import json
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml  # noqa: E402

from fidelity import cardmeta  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL, SKIP = [], [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def skip(name, reason):
    SKIP.append(name)
    print("  SKIP  %s (%s)" % (name, reason))


def card_text(front, body="\n# selftest card\n\nbody text.\n"):
    return "---\n%s---\n%s" % (yaml.dump(front, sort_keys=False, allow_unicode=True), body)


def expect_refusal(name, fn, needle=None):
    try:
        fn()
    except cardmeta.CardError as exc:
        check(name, needle is None or needle in str(exc), str(exc)[:120])
        return
    check(name, False, "no refusal raised")


def main(argv):
    offline = "--offline" in argv
    registry = cardmeta.load_registry()
    committed = os.environ.get("FIDELITY_REGISTRY_HEAD")
    if committed and os.path.isdir(committed):
        registry = cardmeta.load_registry(committed)

    print("== K: card annotation ==")

    # -- K1 / K13: the four shipped examples + the generated K6/K8 cards ------
    examples = []
    for name, repo_type in (("card-k6.yaml", "model"), ("card-k8.yaml", "model"),
                            ("card-root-bf16.yaml", "model"),
                            ("card-dataset-suite-v1.yaml", "dataset")):
        path = os.path.join(REPO, "docs", "examples", name)
        if os.path.isfile(path):
            examples.append((path, repo_type))
    for name in ("GLM-5.3-Flash-TR3-6bpw.README.md", "GLM-5.3-Flash-TR3-8bpw.README.md"):
        path = os.path.join(REPO, "docs", "cards", name)
        if os.path.isfile(path):
            examples.append((path, "model"))
    all_ok = True
    hub_ran = False
    for path, repo_type in examples:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if not text.startswith("---"):
            text = "---\n%s---\n\n# %s\n" % (text, os.path.basename(path))
        report = cardmeta.validate_card(text, registry, offline=offline, repo_type=repo_type)
        for axis in report["axes"]:
            if axis["axis"] == "hub" and axis.get("ran"):
                hub_ran = True
        if report["errors"]:
            all_ok = False
            print("        %s: %s" % (os.path.basename(path), report["errors"][:2]))
    check("K1  %d shipped/generated cards validate on every runnable axis" % len(examples),
          all_ok and bool(examples))
    if hub_ran:
        check("K13 live Hub validate-yaml axis ran and was clean", all_ok)
    else:
        skip("K13 live Hub validate-yaml", "offline" if offline else "network unavailable")

    # -- K2: XC-1..XC-5 on live registry data --------------------------------
    ids = [mid for mid, row in registry["measurements"].items()
           if row.get("artifact_ref") == "artifact--malaiwah.glm-5.3-flash-tr3-6bpw"
           and row.get("status") == "published"]
    if ids:
        model_index = cardmeta.build_model_index(registry, sorted(ids), "GLM-5.3-Flash-TR3-6bpw")
        fidelity = cardmeta.build_x_fidelity(
            registry, role="quant", measurement_ids=sorted(ids),
            reference_model="zai-org/GLM-5.3-Flash-BF16",
            head_file_sha256="47eaf729c93346a2394a72a83da2ae4126dadc51155be477d212a3f0fe3085d0")
        text = card_text({"license": "mit", "base_model": "zai-org/GLM-5.3-Flash-BF16",
                          "base_model_relation": "quantized",
                          "model-index": model_index, "x_fidelity": fidelity})
        axis = cardmeta._our_axis(text, registry)
        check("K2  XC-1..XC-5 on live registry rows for K6", axis["ok"],
              json.dumps(axis["errors"][:3]))
    else:
        skip("K2  XC-1..XC-5", "no K6 rows in this registry clone")

    base_front = {"license": "mit", "base_model": "zai-org/GLM-5.3-Flash-BF16",
                  "base_model_relation": "quantized"}
    minimal_fidelity = {
        "spec": cardmeta.SPEC_URL, "spec_version": cardmeta.SPEC_VERSION, "role": "quant",
        "scope_digest": "x=native:bf16@16|head=native|kv=bf16",
        "registry": {"measurement_ids": ["measurement--x"]},
        "head": {"replay_permitted": False, "lm_head_tensor_content_sha256": None},
        "measurements": [],
    }

    # -- K3: two model-index entries -----------------------------------------
    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": []}, {"name": "B", "results": []}]
    front["x_fidelity"] = minimal_fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K3  two model-index entries are refused (GEN-2)",
          any("ONE model-index" in e for e in axis["errors"]), json.dumps(axis["errors"][:2]))

    # -- K4: colliding merge key ---------------------------------------------
    def one_result(args_lane, split="streaming", revision="a" * 40, config="final25"):
        return {"task": {"type": "text-generation", "name": "x"},
                "dataset": {"type": "d/s", "name": "n", "config": config, "split": split,
                            "revision": revision, "args": {"lane": args_lane}},
                "metrics": [{"type": "kl_divergence", "value": 1.0,
                             "args": {"lane": args_lane}}]}

    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": [one_result("streaming"),
                                                      one_result("streaming")]}]
    front["x_fidelity"] = minimal_fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K4  two results sharing the 5-tuple merge key are refused (GEN-4)",
          any("merge key" in e for e in axis["errors"]), json.dumps(axis["errors"][:2]))

    # -- K5: lane only in args -----------------------------------------------
    result = one_result("serving")
    result["dataset"]["split"] = "streaming"
    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": [result]}]
    front["x_fidelity"] = minimal_fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K5  a lane in args that disagrees with dataset.split is refused (GEN-3)",
          any("lane" in e for e in axis["errors"]), json.dumps(axis["errors"][:2]))

    # -- K6: unquoted all-digit revision -------------------------------------
    result = one_result("streaming", revision=1234567890123456789)
    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": [result]}]
    front["x_fidelity"] = minimal_fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K6  an unquoted all-digit revision is refused (the YAML integer trap)",
          any("quoted string" in e for e in axis["errors"]), json.dumps(axis["errors"][:2]))

    # -- K7: replay_permitted with a null content digest ----------------------
    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": [one_result("streaming")]}]
    fidelity = json.loads(json.dumps(minimal_fidelity))
    fidelity["head"] = {"replay_permitted": True, "lm_head_tensor_content_sha256": None}
    front["x_fidelity"] = fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K7  replay_permitted true with a null head content digest is refused (XC-5)",
          any("XC-5" in e for e in axis["errors"]), json.dumps(axis["errors"][:2]))

    # -- K8: floor_lane != lane ----------------------------------------------
    result = one_result("streaming")
    result["metrics"][0]["args"]["measurement_id"] = "measurement--x"
    result["metrics"].append({
        "type": "kl_divergence_quantization_attributable", "value": 0.5,
        "args": {"floor_lane": "sealed-ep8", "floor_value": 0.5,
                 "floor_measurement_id": "measurement--floor"}})
    front = dict(base_front)
    front["model-index"] = [{"name": "A", "results": [result]}]
    fidelity = json.loads(json.dumps(minimal_fidelity))
    fidelity["measurements"] = [{"id": "measurement--x", "lane": "streaming", "value": 1.0,
                                 "quantization_attributable": 0.5,
                                 "comparability_key": None, "determinism": {}}]
    front["x_fidelity"] = fidelity
    axis = cardmeta._our_axis(card_text(front), registry)
    check("K8  a floor from a different lane is refused (XC-4 / BIAS-006)",
          any("BIAS-006" in e for e in axis["errors"]), json.dumps(axis["errors"][:3]))

    # -- K8b: same lane, DIFFERENT SCOPE -------------------------------------
    if registry["measurements"]:
        fake_row = {
            "measurement_scope": {"scope_name": "clean17"},
            "comparability": {"bias": {"floor_measurement_ref": "measurement--fakefloor"}},
            "metric": {"value": 0.02}, "pipeline_ref": None,
        }
        fake_registry = dict(registry)
        fake_registry["measurements"] = dict(registry["measurements"])
        fake_registry["measurements"]["measurement--fakefloor"] = {
            "measurement_scope": {"scope_name": "panel25"},
            "metric": {"value": 0.01}, "pipeline_ref": None,
            "comparability": {},
        }
        reason = cardmeta.attributable_refusal(fake_registry, fake_row, "sealed-ep8")
        check("K8b a floor over a DIFFERENT SCOPE is withheld (a 25-window floor is not a "
              "17-window row's zero-point)",
              bool(reason) and "scope" in reason, str(reason)[:110])

    # -- K9: a measurement id the registry does not have ---------------------
    expect_refusal("K9  a model-index result for an unknown measurement is refused (XC-3)",
                   lambda: cardmeta.build_model_index(registry, ["measurement--nope"], "A"),
                   needle="not in the registry")

    # -- K10: base_model_relation enum ---------------------------------------
    expect_refusal("K10 base_model_relation: fidelity-reference is refused (4-value enum)",
                   lambda: cardmeta.merge_card(
                       card_text({"license": "mit"}), model_index=None,
                       x_fidelity=minimal_fidelity,
                       base_model="a/b", base_model_relation="fidelity-reference"),
                   needle="four values")

    # -- K11: unknown top-level keys survive ---------------------------------
    original = card_text({"license": "mit", "model_creator": "someone",
                          "prompt_template": "{prompt}", "quantized_by": "malaiwah"})
    merged = cardmeta.merge_card(original, model_index=None, x_fidelity=minimal_fidelity)
    front, body = cardmeta.split_card(merged)
    check("K11 pre-existing unknown top-level keys survive annotate (GEN-5)",
          front.get("model_creator") == "someone"
          and front.get("prompt_template") == "{prompt}"
          and front.get("quantized_by") == "malaiwah"
          and body.strip().startswith("# selftest card"))

    # -- K12: verified is stripped -------------------------------------------
    result = one_result("streaming")
    result["verified"] = True
    result["verifyToken"] = "abc"
    merged = cardmeta.merge_card(
        card_text({"license": "mit"}),
        model_index=[{"name": "A", "results": [result]}], x_fidelity=minimal_fidelity)
    front, _ = cardmeta.split_card(merged)
    emitted = front["model-index"][0]["results"][0]
    check("K12 a hand-set verified/verifyToken is stripped (GEN-7)",
          "verified" not in emitted and "verifyToken" not in emitted)

    # -- K14: an absolute host path in front matter is refused ----------------
    # Regression guard.  The generator used to copy `os.path.abspath(registry)`
    # into `x_fidelity.registry.snapshot.root`, which shipped
    # `/Users/<name>/...` into two cards queued for publication.  The Hub's own
    # validate-yaml accepts that happily -- it is only ever OUR check -- and it
    # is the same class of defect as the dead `packed_root` that motivated the
    # whole format.  Both the leak and the silence are covered here.
    leaky = dict(minimal_fidelity)
    leaky["registry"] = dict(leaky.get("registry") or {})
    leaky["registry"]["snapshot"] = {
        "root": "/Users/someone/Projects/glm53-fidelity-suite/registry",
        "data_sha256": {"measurements": "00" * 32},
    }
    merged = cardmeta.merge_card(
        card_text({"license": "mit"}),
        model_index=[{"name": "A", "results": [one_result("streaming")]}],
        x_fidelity=leaky)
    axis = cardmeta._our_axis(merged, cardmeta.load_registry())
    check("K14 an absolute host path in front matter is refused (HOSTPATH-1)",
          not axis["ok"] and any("HOSTPATH-1" in e for e in axis["errors"]),
          "errors=%r" % (axis["errors"],))

    # ... and the check must not fire on the legitimately rooted strings that
    # live in front matter: URL paths, and our own `/api/...`-style fragments.
    ok_fidelity = dict(minimal_fidelity)
    ok_fidelity["registry"] = dict(ok_fidelity.get("registry") or {})
    ok_fidelity["registry"]["snapshot"] = {"data_sha256": {"measurements": "00" * 32}}
    merged = cardmeta.merge_card(
        card_text({"license": "mit"}),
        model_index=[{"name": "A", "results": [one_result("streaming")]}],
        x_fidelity=ok_fidelity)
    axis = cardmeta._our_axis(merged, cardmeta.load_registry())
    check("K15 HOSTPATH-1 does not fire on URLs or a snapshot without a root",
          not any("HOSTPATH-1" in e for e in axis["errors"]),
          "errors=%r" % (axis["errors"],))

    # ---- CLI-06 / CLI-12: the annotate DRIVER, not just the merge helpers --------
    import subprocess, tempfile, shutil
    cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fidelity_card.py")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "docs/cards/GLM-5.3-Flash-TR3-6bpw.README.md")
    if os.path.exists(cli) and os.path.exists(src):
        tmp = tempfile.mkdtemp(prefix="card-cli-")

        def run(argv):
            r = subprocess.run([sys.executable, cli] + argv, capture_output=True, text=True)
            return r.returncode, (r.stdout or "") + (r.stderr or "")

        # CLI-12: the documented quickstart passes --base-model and NO --model-name.
        # Pre-fix that named the quant's model-index entry after the BASE model, so the
        # quant's KLD was attributed to the unquantized reference.
        c1 = os.path.join(tmp, "q.md"); shutil.copyfile(src, c1)
        out1 = os.path.join(tmp, "q.out.md")
        rc, txt = run(["annotate", "--card", c1, "--role", "quant",
                       "--artifact-id", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                       "--base-model", "zai-org/GLM-5.3-Flash-BF16",
                       "--out", out1, "--offline"])
        name = None
        if os.path.exists(out1):
            for i, line in enumerate(io.open(out1, encoding="utf-8").read().splitlines()):
                if line.strip().startswith("- name:"):
                    name = line.split(":", 1)[1].strip(); break
        check("CLI-12 model-index names the MEASURED artifact, not --base-model",
              rc == 0 and name == "GLM-5.3-Flash-TR3-6bpw",
              "rc=%d name=%r (pre-fix: GLM-5.3-Flash-BF16)" % (rc, name))

        # and with neither flag it must not fall back to the card's FILENAME
        c2 = os.path.join(tmp, "README.md"); shutil.copyfile(src, c2)
        out2 = os.path.join(tmp, "r.out.md")
        rc2, _ = run(["annotate", "--card", c2, "--role", "quant",
                      "--artifact-id", "artifact--malaiwah.glm-5.3-flash-tr3-6bpw",
                      "--out", out2, "--offline"])
        name2 = None
        if os.path.exists(out2):
            for line in io.open(out2, encoding="utf-8").read().splitlines():
                if line.strip().startswith("- name:"):
                    name2 = line.split(":", 1)[1].strip(); break
        check("CLI-12 a card called README.md is not named \"README\"",
              rc2 == 0 and name2 == "GLM-5.3-Flash-TR3-6bpw",
              "rc=%d name=%r (pre-fix: README)" % (rc2, name2))

        # CLI-06: a refused annotate must leave the card BYTE-IDENTICAL. A quant card
        # with no base_model fails its own validator; pre-fix the file was already
        # overwritten by the time "nothing was published" printed.
        c3 = os.path.join(tmp, "inplace.md")
        io.open(c3, "w", encoding="utf-8").write(
            "---\nlibrary_name: transformers\n---\n\n# Hello\n\nbody text\n")
        before = io.open(c3, encoding="utf-8").read()
        rc3, txt3 = run(["annotate", "--card", c3, "--role", "quant",
                         "--measurement-id",
                         "measurement--glm53.k6-6bpw-stream.brandonmusic-final25",
                         "--in-place", "--offline"])
        after = io.open(c3, encoding="utf-8").read()
        check("CLI-06 a REFUSED annotate leaves the card untouched",
              rc3 != 0 and after == before,
              "rc=%d, card %s (pre-fix: rewritten, and the refusal said "
              "'nothing was published')" % (rc3, "unchanged" if after == before else "REWRITTEN"))

    print("\nselftest_fidelity_card: %d passed, %d failed, %d skipped"
          % (len(PASS), len(FAIL), len(SKIP)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
