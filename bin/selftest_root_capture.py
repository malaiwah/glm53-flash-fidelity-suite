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

print("\n== the panel travels with the bundle ==")
bundle = (SUITE / "bin" / "BUNDLE.txt").read_text()
for need in ("bin/fidelity_dataset.py", "k6/tools/hf_capture.py",
             "k6/tools/layer_outer.py", "bin/fidelity/dsmanifest.py"):
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
