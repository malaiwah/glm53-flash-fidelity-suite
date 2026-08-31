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
import re
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

print("\n== a designated reference is a door, not a hole in the wall ==")


def guard_flagged(config, designated):
    real, mc.fetch_json = mc.fetch_json, lambda *a, **k: config
    ns = type("A", (), {"designated_reference": designated})()
    plan = {}
    try:
        mc._refuse_quantized_root(Con(), target(), surface("unknown", "fp8_e4m3"),
                                  plan, args=ns)
        return None, plan
    except mc.Refusal as exc:
        return "\n".join([str(exc)] + [str(a) for a in (exc.advice or [])]), plan
    finally:
        mc.fetch_json = real


FP8 = {"quantization_config": {"quant_method": "fp8"}}
BF16 = {"model_type": "deepseek_v4"}

# Without the flag, a quantized root is still refused -- the wall stands.
msg, _ = guard_flagged(FP8, False)
check("a quantized root is still REFUSED without the flag", msg is not None)

# With the flag, it proceeds AND the designation is recorded in the plan,
# which is what carries it into job.json and the sealed dataset.
msg, plan = guard_flagged(FP8, True)
check("--designated-reference admits a quantized root", msg is None)
dr = (plan.get("target") or {}).get("designated_reference") or {}
check("...and records the designation with its quant method",
      dr.get("quant_method") == "fp8")
check("...and root_unquantized is honestly False",
      (plan.get("target") or {}).get("root_unquantized") is False)

# The contradiction case: the flag on a TRUE root is refused, because minting a
# proxy for a family that has a real root would turn advisory-by-necessity
# into advisory-by-convenience.
msg, _ = guard_flagged(BF16, True)
check("the flag on an UNQUANTIZED root is refused", msg is not None)
check("...telling the caller to capture it as a plain root",
      bool(msg) and "plain root" in msg)

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
              "--panel-dir", "engines/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--lane", "streaming")
check("--role root with no --dataset-id is refused", rc == mc.EXIT_REFUSED)
check("...because a capture with no identity cannot be cited",
      "cannot be" in out and "published" in out)

rc, out = cli("--role", "root", "--model", "a/b", "--panel", "some/panel",
              "--panel-dir", "engines/panels/panel--minimaxm3.malaiwah.corpus5x5",
              "--dataset-id", "d", "--lane", "streaming")
check("--panel and --panel-dir together are refused", rc == mc.EXIT_REFUSED)

print("\n== the stage sequence has no second side ==")
# Asserted as a SEQUENCE, not as a source literal. These checks used to grep
# measure_cloud.py for `stages = [...]`, and broke the moment the lists moved
# into fidelity/stages.py -- a test that fails on a refactor which changed no
# behaviour is testing the text, not the tool.
def stage_seq(name):
    try:
        from fidelity import stages as _st
        return tuple(getattr(_st, name))
    except Exception:                                     # noqa: BLE001
        src = (SUITE / "bin" / "measure_cloud.py").read_text(encoding="utf-8")
        import ast as _ast
        for node in _ast.walk(_ast.parse(src)):
            if isinstance(node, _ast.Assign) and any(
                    getattr(t, "id", "") == "stages" for t in node.targets):
                try:
                    val = _ast.literal_eval(node.value)
                except Exception:                         # noqa: BLE001
                    continue
                if name == "ROOT_STAGES" and "capture" in val and "score" not in val:
                    return tuple(val)
                if name == "ROOT_RACE_STAGES" and "race_capture" in val:
                    return tuple(val)
        return ()

root = stage_seq("ROOT_STAGES")
race = stage_seq("ROOT_RACE_STAGES")
check("a root runs setup/fetch_target/capture/verify",
      root == ("setup", "fetch_target", "capture", "verify"))
check("...and does NOT score (there is nothing to diverge from)",
      "score" not in root and "materialize" not in root)

stage_sh = (SUITE / "bin" / "stage_measure.sh").read_text()
for st in ("capture)", "verify)"):
    check("stage_measure.sh implements %s" % st.rstrip(")"), st in stage_sh)
check("the capture stage refuses a non-root job",
      "capture stage is --role root only" in stage_sh)
check("the capture stage is receipt-resumable",
      "already written at $OUT -- skipping" in stage_sh)

print("\n== what the controller writes into job.json, the stage must FORWARD ==")
# A knob the controller records and the stage ignores is worse than a missing
# knob: the operator is told a thing happened that did not. Both of these were
# real. `--sanity-expect Paris` reached job.json and only `race_capture` read
# it, so on the DEFAULT capture path the generation probe ran unenforced -- the
# one check that distinguishes "captured" from "captured nonsense".
# `--allow-unexpected-tensors` had no controller flag at all, so a root capture
# of any checkpoint carrying an MTP/draft block (this suite's own Fruit fixture;
# GLM-5.3-Flash; GLM-5.3) died at the capture stage with the rental already paid
# for.
CAPTURE_STAGES = ("capture", "race_capture")


def stage_body(name):
    """The text of one `case` arm of stage_measure.sh."""
    start = stage_sh.index("\n%s)\n" % name)
    end = stage_sh.index("\n  ;;", start)
    return stage_sh[start:end]


for _st in CAPTURE_STAGES:
    body = stage_body(_st)
    check("%s reads capture.sanity_expect" % _st,
          "capture.sanity_expect" in body)
    check("...and forwards --sanity-expect to the engine" ,
          "--sanity-expect" in body)
    check("%s reads capture.allow_unexpected_tensors" % _st,
          "capture.allow_unexpected_tensors" in body)
    check("...and forwards --allow-unexpected-tensors when it is true",
          "--allow-unexpected-tensors" in body)
    # Data from job.json is passed as an ARRAY, never through an eval (SEC-01).
    check("...via the EXTRA array, expanded into the %s invocation" % _st,
          "EXTRA+=" in body and '"${EXTRA[@]}"' in body)
    # PROVENANCE, not cosmetics. hf_capture resolves the weights' identity as
    # `--weights-repository or --model`, and --model is the LOCAL tree the
    # fetch wrote. Without the first flag a published root records the RENTED
    # BOX'S ABSOLUTE PATH as the checkpoint repository, the panel's tokenizer
    # id and the card's provenance line -- pointing at a filesystem that no
    # longer exists, which is the exact defect AGENTS.md tells you to grep a
    # published artifact for. It also makes the capture non-comparable: the
    # tokenizer id is panel IDENTITY (PANEL-D6), so two roots of one model
    # captured on two boxes declare two tokenizers and `compare` refuses them.
    check("%s names the HF repo as the weights repository, not the local path"
          % _st, "--weights-repository" in body)
    # THE GPU. hf_capture's --device defaults to "cpu"; the stage script never
    # set it, so a root capture ran the forward on the CPU of a box rented for
    # its GPU -- at 0% utilisation, for the full hourly rate, on every provider.
    # The `materialize` and `measure` stages have always passed `--device
    # cuda`; only this path was missed.
    check("%s reads capture.device (default cuda)" % _st,
          "capture.device cuda" in body)
    check("...and forwards --device to the engine",
          '--device "$DEVICE"' in body)

check("the controller has a --capture-device flag, defaulting to cuda",
      '"--capture-device", default="cuda"' in
      (SUITE / "bin" / "measure_cloud.py").read_text(encoding="utf-8"))

check("the controller has an --allow-unexpected-tensors flag to set it with",
      "--allow-unexpected-tensors" in
      (SUITE / "bin" / "measure_cloud.py").read_text(encoding="utf-8"))

print("\n== race mode: a different stage sequence, and a different identity ==")
# The whole point of race mode is that the fetch stops being a barrier. If the
# sequence still contained fetch_target the overlap could not happen at all.
check("a race root runs setup/race_bootstrap/race_capture/verify",
      race == ("setup", "race_bootstrap", "race_capture", "verify"))
check("...and race mode has no fetch_target stage (the fetch is IN the capture)",
      "fetch_target" not in race and "fetch_target" in root)

print("\n== a plain full-precision tree must SNIFF as native-bf16, whatever "
      "spelling its config uses for the dtype ==")
# The surface check runs BEFORE the root guard above and refuses for $0.00,
# which is right -- but it read the dtype from only two of the three places a
# real config puts it.  `malaiwah/GLM-5.2-SIQ-Fruit-bf16` (transformers 5.12)
# writes a TOP-LEVEL `dtype`, the current transformers default for a
# single-modality config, and was refused as "no recognised surface marker":
# a plain bf16 checkpoint, the one thing a root capture exists to read,
# declared unreadable by every adapter.  One dict key, and the run that found
# it was a paid rental away.
from fidelity import hfmeta as HM                          # noqa: E402


def sniff_plain(config):
    meta = HM.RepoMeta(
        repo_id="x/y", repo_type="model", revision="a" * 40,
        requested_revision="main", last_modified=None,
        files=[("config.json", 1846), ("model-layer-000.safetensors", 1 << 20),
               ("model.safetensors.index.json", 4096)])
    real, HM.fetch_json = HM.fetch_json, lambda *a, **k: config
    try:
        return HM.sniff_surface(meta)
    finally:
        HM.fetch_json = real


for label, cfg in (
        ("top-level `dtype` (transformers >= 5; the Fruit release)",
         {"model_type": "glm_moe_dsa", "dtype": "bfloat16"}),
        ("top-level `torch_dtype` (older configs)",
         {"model_type": "llama", "torch_dtype": "bfloat16"}),
        ("nested `text_config.dtype` (GLM-5.3-Flash)",
         {"model_type": "glm4v_moe", "text_config": {"dtype": "bfloat16"}})):
    got = sniff_plain(cfg)
    check("%s sniffs as native-bf16" % label,
          got.surface == "native-bf16" and got.codec_family == "bf16"
          and got.bits == 16.0 and not got.problems)

check("a config with NO dtype anywhere is still 'unknown' (not guessed)",
      sniff_plain({"model_type": "llama"}).surface == "unknown")
check("a quantized config is not promoted to native-bf16 by its dtype",
      sniff_plain({"model_type": "llama", "dtype": "bfloat16",
                   "quantization_config": {"quant_method": "fp8"}}
                  ).surface == "unknown")

print("\n== the panel travels with the bundle ==")
bundle = (SUITE / "bin" / "BUNDLE.txt").read_text()
for need in ("bin/fidelity_dataset.py", "engines/tools/hf_capture.py",
             "engines/tools/layer_outer.py", "bin/fidelity/dsmanifest.py",
             "engines/tools/race_fetch.py", "engines/tools/generation_probe.py"):
    check("bundle ships %s" % need, need in bundle)
missing = [ln.strip() for ln in bundle.splitlines()
           if ln.strip() and not ln.startswith("#")
           and not (SUITE / ln.strip()).is_file()]
check("every bundle entry exists on disk", not missing)

# A bundled file's DATA is a dependency too, and nothing checked that.
# `bootstrap_measure.sh` runs `selftest_gguf_offline.py` at setup, fail-closed;
# that selftest reads `engines/tools/gguf-evidence/`, which was never bundled. So a
# MiniMax ROOT capture died in its setup stage on GGUF test fixtures, with the
# controller showing nothing but "stage setup" while the instance billed. The
# existing import check (P11) could not see it: this is data, not an import.
bundled = {ln.strip() for ln in bundle.splitlines()
           if ln.strip() and not ln.startswith("#")}
bundled_dirs = {str(Path(b).parent) for b in bundled}
# The rule is "bundle the data, OR the reader must tolerate its absence" --
# not "bundle everything". dione-evidence is 187 MB of fixtures for a surface
# most runs never touch, and shipping it on every rental would cost more than
# the bug it prevents. Its selftest already skips cleanly when it is missing,
# which is why exl3 runs have always passed setup while the gguf one -- which
# does NOT skip -- killed a MiniMax capture. Anything listed here is a
# deliberate exception with a stated reason, and the list is the review surface.
TOO_BIG_TO_BUNDLE = {
    "engines/tools/dione-evidence": "187 MB; its selftest skips when absent",
}
data_gaps = []
for entry in sorted(bundled):
    path = SUITE / entry
    if path.suffix not in (".py", ".sh") or not path.is_file():
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    # sibling data directories the file names, e.g. `TOOLS / "gguf-evidence"`
    # or a literal "engines/tools/gguf-evidence/..."
    # any sibling data directory the file names, however it spells the path
    for sib in set(re.findall(r"([A-Za-z0-9_.-]+-evidence)", text)):
        d = SUITE / Path(entry).parent / sib
        if not d.is_dir():
            continue
        rel = str(d.relative_to(SUITE))
        if any(b.startswith(rel + "/") for b in bundled):
            continue
        if rel in TOO_BIG_TO_BUNDLE:
            continue
        data_gaps.append("%s reads %s/, which is not bundled" % (entry, rel))
for g in sorted(set(data_gaps)):
    print("      %s" % g)
check("a bundled file's data directories are bundled too", not data_gaps)

panel = SUITE / "engines" / "panels" / "panel--minimaxm3.malaiwah.corpus5x5"
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
