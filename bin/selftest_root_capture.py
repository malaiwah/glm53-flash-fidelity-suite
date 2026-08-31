#!/usr/bin/env python3
"""`--role root` -- capturing the thing every other measurement is a distance from.

The measure path answers "how far is this quant from the reference?". A root
capture answers nothing: it PRODUCES the reference side, sealed and
publishable, so that later measurements read it instead of re-deriving it. It
has no candidate, no divergence and no engine profile, and the one thing it
must never do is capture a quantized checkpoint and call it a floor.

Offline: the release's config is injected, so this proves the DECISIONS, not
the network.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_cloud as mc                                # noqa: E402

SUITE = Path(mc.SUITE_ROOT)
FAILED = []


def check(label, ok):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)


class Con:
    def ok(self, *a):
        pass

    def warn(self, *a):
        pass

    def err(self, *a):
        pass


def surface(kind="native-bf16", codec="bf16"):
    return type("S", (), {"surface": kind, "codec_family": codec,
                          "evidence": {}, "problems": []})()


def target(repo="x/y", rev="a" * 40):
    return type("T", (), {"repo_id": repo, "revision": rev})()


def guard(config):
    """Returns None if accepted, the refusal text if refused."""
    real, mc.fetch_json = mc.fetch_json, lambda *a, **k: config
    try:
        mc._refuse_quantized_root(Con(), target(), surface(), {})
        return None
    except mc.Refusal as exc:
        return "\n".join([str(exc)] + [str(a) for a in (exc.advice or [])])
    finally:
        mc.fetch_json = real


print("== a root must be the unquantized thing, or it is not a reference ==")

check("a plain BF16 checkpoint is accepted",
      guard({"model_type": "minimax_m3_vl"}) is None)

msg = guard({"quantization_config": {"quant_method": "fp8"}})
check("a checkpoint with a quantization_config is REFUSED", bool(msg))
check("...naming the method", bool(msg) and "fp8" in msg)
check("...and saying why it would not fail loudly",
      bool(msg) and "block scale" in msg)

check("a quantization_config nested in text_config is also REFUSED",
      guard({"text_config": {"quantization_config": {"quant_method": "awq"}}})
      is not None)

# `sniff_surface` returns "unknown" for plenty of unquantized roots --
# zai-org/GLM-5.3-BF16 and zai-org/GLM-5.2 both do -- and an earlier version of
# this gate refused on that, which would have blocked the exact captures the
# mode exists for.
real, mc.fetch_json = mc.fetch_json, lambda *a, **k: {"model_type": "glm_moe_dsa"}
try:
    plan = {}
    mc._refuse_quantized_root(Con(), target(), surface("unknown", None), plan)
    check("an UNQUANTIZED root whose surface sniffs 'unknown' is accepted",
          plan.get("target", {}).get("root_unquantized") is True)
finally:
    mc.fetch_json = real

print("\n== the root path takes different inputs, and refuses without them ==")


def cli(*argv):
    p = subprocess.run([sys.executable, str(SUITE / "bin" / "measure_cloud.py")]
                       + list(argv), capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


rc, out = cli("--role", "root", "--model", "a/b", "--lane", "streaming")
check("--role root with no panel is refused", rc == mc.EXIT_REFUSED)
check("...naming both accepted forms",
      "--panel" in out and "--panel-dir" in out)

rc, out = cli("--role", "root", "--model", "a/b",
              "--panel-dir", "k6/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--lane", "streaming")
check("--role root with no --dataset-id is refused", rc == mc.EXIT_REFUSED)
check("...because a capture with no identity cannot be cited",
      "cannot be" in out and "published" in out)

rc, out = cli("--role", "root", "--model", "a/b", "--panel", "some/panel",
              "--panel-dir", "k6/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--dataset-id", "d", "--lane", "streaming")
check("--panel and --panel-dir together are refused", rc == mc.EXIT_REFUSED)

print("\n== the stage sequence has no second side ==")
src = (SUITE / "bin" / "measure_cloud.py").read_text()
i = src.index('stages = ["setup", "fetch_target", "capture", "verify"]')
check("a root runs setup/fetch_target/capture/verify", i > 0)
check("...and does NOT score (there is nothing to diverge from)",
      '"capture", "verify"' in src and
      'stages = ["setup", "fetch_target", "capture", "verify"]' in src)

stage_sh = (SUITE / "bin" / "stage_measure.sh").read_text()
for st in ("capture)", "verify)"):
    check("stage_measure.sh implements %s" % st.rstrip(")"), st in stage_sh)
check("the capture stage refuses a non-root job",
      "capture stage is --role root only" in stage_sh)
check("the capture stage is receipt-resumable",
      "already written at $OUT -- skipping" in stage_sh)

print("\n== race mode: a different stage sequence, and a different identity ==")

# The whole point of race mode is that the fetch stops being a barrier. If the
# stage list still contained fetch_target the overlap could not happen at all,
# so the sequence itself is the assertion.
check("a race root runs setup/race_bootstrap/race_capture/verify",
      'stages = ["setup", "race_bootstrap", "race_capture", "verify"]' in src)
check("...and race mode has no fetch_target stage (the fetch is IN the capture)",
      'stages = ["setup", "race_bootstrap", "race_capture", "verify"]' in src
      and 'stages = ["setup", "fetch_target", "capture", "verify"]' in src)
for st in ("race_bootstrap)", "race_capture)"):
    check("stage_measure.sh implements %s" % st.rstrip(")"), st in stage_sh)
check("race_bootstrap fetches the index and NO shards",
      "--include model.safetensors.index.json" in stage_sh
      and "no shards" in stage_sh)
check("race_bootstrap refuses a checkpoint with no shard index",
      "race mode needs $DEST/model.safetensors.index.json" in stage_sh)
check("race_capture passes --race-repo to the capture engine",
      "--race-repo" in stage_sh)
# Split defensively: against a tree WITHOUT this change there is no
# `race_capture)` stanza, and an IndexError here would abort the file before the
# remaining cases reported -- which is the one shape of evidence that cannot be
# read as "these cases fail without the fix".
_race_parts = stage_sh.split("race_capture)")
_race_block = _race_parts[1].split("\nverify)")[0] if len(_race_parts) > 1 else ""
check("race_capture verifies the published seal AFTER the tree is complete",
      "verify_published_sums.py" in _race_block)
# SEC-01: the values come from job.json, so the shell must not be PARSING data.
# The check looks for an eval INVOCATION, not the word -- the comment beside the
# array in the stage script says "never an eval" and would match a naive scan.
_evals = [ln for ln in _race_block.splitlines()
          if ln.split("#")[0].strip().startswith("eval ")
          or "$(eval" in ln.split("#")[0]]
check("race_capture builds its extra flags as an array, never an eval (SEC-01)",
      "EXTRA=()" in _race_block and not _evals)

rc, out = cli("--role", "root", "--model", "a/b",
              "--panel-dir", "k6/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--dataset-id", "d", "--lane", "streaming",
              "--race", "--schedule", "window-outer")
check("--race with --schedule window-outer is refused for $0.00",
      rc == mc.EXIT_REFUSED and "no not-yet-arrived layer" in out)

rc, out = cli("--role", "root", "--model", "a/b",
              "--panel-dir", "k6/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--dataset-id", "d", "--preview-of", "d", "--lane", "streaming")
check("--preview-of equal to --dataset-id is refused before any spend",
      rc == mc.EXIT_REFUSED and "two DATASETS" in out)
check("...naming reference_id as the reason it would corrupt the registry",
      "reference_id is a comparability-key field" in out)

check("the job document carries preview_of, race and the sanity expectation",
      '"preview_of": getattr(args, "preview_of", None) or None' in src
      and '"race": bool(getattr(args, "race", False))' in src
      and '"sanity_expect"' in src)

# jqget is how every stage reads job.json, and a JSON null used to come back as
# the four-letter string "None" -- so `[ -n "$PREVIEW_OF" ]` was TRUE for a job
# that set no preview, and the capture would have been handed a dataset id
# spelled None. Driven by actually sourcing the function, not by reading it.
with tempfile.TemporaryDirectory() as tmp:
    conf = Path(tmp) / "job.json"
    conf.write_text(json.dumps({"capture": {"preview_of": None, "sanity_expect": "",
                                            "panel_dir": None},
                                "lane": None, "role": "root"}))
    fn = Path(tmp) / "jqget.sh"
    text = (SUITE / "bin" / "stage_measure.sh").read_text()
    start = text.index("jqget() {")
    fn.write_text(text[start:text.index("\n}\n", start) + 3])
    script = ('CONF=%s\n. %s\nprintf "[%%s][%%s][%%s][%%s]" '
              '"$(jqget capture.preview_of)" "$(jqget lane streaming)" '
              '"$(jqget capture.sanity_expect Paris)" "$(jqget capture.nope fb)"'
              % (conf, fn))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout
    check("jqget reads a JSON null as ABSENT, not as the string 'None'",
          out == "[][streaming][][fb]")

print("\n== the panel travels with the bundle ==")
bundle = (SUITE / "bin" / "BUNDLE.txt").read_text()
for need in ("bin/fidelity_dataset.py", "k6/tools/hf_capture.py",
             "k6/tools/layer_outer.py", "bin/fidelity/dsmanifest.py",
             "k6/tools/race_fetch.py", "k6/tools/generation_probe.py"):
    check("bundle ships %s" % need, need in bundle)
missing = [ln.strip() for ln in bundle.splitlines()
           if ln.strip() and not ln.startswith("#")
           and not (SUITE / ln.strip()).is_file()]
check("every bundle entry exists on disk", not missing)

panel = SUITE / "k6" / "panels" / "panel--minimaxm3.malaiwah.corpus5x5"
check("the committed MiniMax panel is a panel directory",
      (panel / "panel.json").is_file() and (panel / "arrays").is_dir())

# A panel outside the suite has no remote path, because the uploader addresses
# files RELATIVE to the suite root.
with tempfile.TemporaryDirectory() as tmp:
    outside = Path(tmp) / "panel"
    (outside / "arrays").mkdir(parents=True)
    (outside / "panel.json").write_text("{}")
    rc, out = cli("--role", "root", "--model", "a/b", "--panel-dir", str(outside),
                  "--dataset-id", "d", "--lane", "streaming", "--dry-run")
    check("a --panel-dir outside the suite checkout is refused",
          "must live inside the suite checkout" in out)

print()
if FAILED:
    print("selftest_root_capture: %d FAILED" % len(FAILED))
    sys.exit(1)
print("selftest_root_capture: all passed")
