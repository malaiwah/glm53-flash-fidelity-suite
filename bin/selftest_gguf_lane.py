#!/usr/bin/env python3
"""T13 -- the GGUF LANE wiring: shelf -> plan -> argv -> fetch -> receipt.

`k6/tools/gguf_surface.py` could already read a llama.cpp container, and
`stream_score.py` could already score one, and `k6_kld_report.py` and
`registry_add.py` already knew the family.  None of that was reachable: the
lane did not classify a GGUF repo as anything, so `bin/measure-cloud` refused
`unsloth/GLM-5.3-Flash-GGUF` -- the largest quant audience this model has --
with "this artifact cannot be read by any available surface adapter", which was
false.  A capability nothing can invoke is indistinguishable from a missing one.

Every rung here is a link in that chain, and each fails without its fix:

  1  a GGUF repo is a SHELF: builds are grouped, an unnamed choice is refused
     by NAME, and `--path` selects one.  (hfmeta.sniff_surface)
  2  the profile is RATE-INDEPENDENT: one reader, one receipt family, one
     format-wide student label.  (engines.json "*" + resolve_profile)
  3  a repeated flag survives the flag_map: `--gguf-file` is `action="append"`
     and a build is n files.  (engines.build_invocation)
  4  the composed on-instance argv names every part, the sealed inventory and
     `--source gguf`.  (invoke_engine)
  5  the structural invariant that stops the NEXT surface half-landing: every
     surface a lane declares has a `--source` spelling.
  6  the fetch is scoped to the chosen build -- 200 GB, not the shelf's 2.55
     TB -- and reaches `hf` as literal arguments.  (stage_measure.sh)
  7  the sealed receipt says gguf: container, path, effective rate, and a scope
     that does NOT record "unknown" for tensors the artifact declares it
     quantized.  Validated by the registry's own validator.

Run:  python3 bin/selftest_gguf_lane.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.engines import build_invocation, load_engines      # noqa: E402
from fidelity.hfmeta import (                                    # noqa: E402
    RepoMeta, gguf_build_key, gguf_builds, gguf_nominal_rate, sniff_surface,
)

REPO = "unsloth/GLM-5.3-Flash-GGUF"
REV = "2975ab414d30340466d8c51533c6e91f0cca64c1"
BUILD = "UD-Q4_K_XL"
PARTS = ["%s/GLM-5.3-Flash-%s-%05d-of-00006.gguf" % (BUILD, BUILD, i)
         for i in range(1, 7)]

_pass, _fail = 0, 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print("  PASS  %s" % msg)


def no(msg: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    print("  FAIL  %s%s" % (msg, ("\n        " + detail) if detail else ""))


def check(cond, msg: str, detail: str = "") -> None:
    ok(msg) if cond else no(msg, detail)


def shelf_meta() -> RepoMeta:
    """The real repo's file list, in miniature: two builds plus the noise.

    The noise is not decoration. `mmproj-*.gguf` is the vision projector, which
    is not a model build; `*.gguf_file` is unsloth's shard-rewrite sidecar and
    its published imatrix, deliberately not named `.gguf`. Both were live traps:
    either one offered as a "build" is a choice a measurement cannot make.
    """
    files = [(name, 1 << 30) for name in PARTS]
    files += [("Q8_0/GLM-5.3-Flash-Q8_0-%05d-of-00008.gguf" % i, 1 << 30)
              for i in range(1, 9)]
    files += [("mmproj-BF16.gguf", 1164010080),
              ("imatrix_unsloth.gguf_file", 512687616),
              ("Shard_Rewrite/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf_file", 9429984),
              ("README.md", 8377)]
    return RepoMeta(repo_id=REPO, repo_type="model", revision=REV,
                    requested_revision="main", last_modified=None, files=files)


# ---------------------------------------------------------------- rung 1
def rung_shelf() -> None:
    meta = shelf_meta()
    builds = gguf_builds(meta)
    check(sorted(builds) == ["Q8_0", BUILD],
          "1a a GGUF repo is a shelf: builds group, mmproj and .gguf_file are not builds",
          "got %s" % sorted(builds))
    check(gguf_build_key("Qwen3.8-27B-Q6_K.gguf") == "Qwen3.8-27B-Q6_K"
          and gguf_build_key("Qwen3.8-27B-BF16-00001-of-00002.gguf")
          == "Qwen3.8-27B-BF16",
          "1b the FLAT layout groups too (unsloth publishes both; the split "
          "suffix must not read as four separate artifacts)")

    unnamed = sniff_surface(meta)
    check(unnamed.surface == "gguf" and unnamed.problems
          and BUILD in unnamed.problems[0] and "--path" in unnamed.problems[0],
          "1c an unnamed choice is refused and LISTS the builds",
          "; ".join(unnamed.problems) or "no problem raised")

    chosen = sniff_surface(meta, BUILD)
    check(chosen.surface == "gguf" and not chosen.problems
          and chosen.path == BUILD
          and [n for n, _ in chosen.artifact_files] == PARTS,
          "1d --path selects one build and carries exactly its parts")
    check(chosen.artifact_bytes == 6 << 30 and chosen.artifact_bytes < meta.total_bytes,
          "1e the artifact is the BUILD's bytes, not the shelf's "
          "(pricing the shelf refuses a run that fits)",
          "%s vs %s" % (chosen.artifact_bytes, meta.total_bytes))
    check(chosen.bits == 4.0 and chosen.codec_family == "gguf-k-quant"
          and chosen.nonrouted_native is False,
          "1f the nominal rate and codec come from the name, and non-routed "
          "native is False: a GGUF quantizes the whole forward")
    check(gguf_nominal_rate("UD-IQ4_XS") == (4.0, "gguf-i-quant"),
          "1g IQ4_XS is not read as Q4_ (token order matters)")
    check(sniff_surface(meta, "UD-Q9_NOPE").problems,
          "1h a --path that names no build is refused")


# ---------------------------------------------------------------- rung 2
def rung_profile() -> None:
    from measure_cloud import PROFILE_TABLE_NAMES, resolve_profile

    lane = load_engines()["streaming"]
    check("gguf" in lane.surfaces,
          "2a the streaming lane declares the gguf surface")
    check(lane.profile_map_by_surface.get("gguf") == {"*": "gguf"},
          "2b the gguf map is rate-INDEPENDENT ('*'), not a bits table",
          repr(lane.profile_map_by_surface.get("gguf")))
    check(resolve_profile(lane, "gguf", 4.0) == "gguf"
          and resolve_profile(lane, "gguf", 2.0) == "gguf"
          and resolve_profile(lane, "gguf", None) == "gguf",
          "2c every rate -- and an UNKNOWN rate -- resolves to one profile, "
          "because the receipt family and the student label are format-wide")
    check(resolve_profile(lane, "dione", 9.0) is None
          and resolve_profile(lane, "exl3hf", 4.05) == "turbo-4.05bpw",
          "2d the wildcard did not leak: a rate-keyed surface still refuses an "
          "unmapped rate and still resolves a mapped one")
    src = (ROOT / "k6" / "tools" / "stream_score.py").read_text(encoding="utf-8")
    check(("\n%s =" % PROFILE_TABLE_NAMES["gguf"]) in src,
          "2e the refusal's advice names a constant that exists in stream_score",
          PROFILE_TABLE_NAMES["gguf"])
    check("gguf" in load_engines()["streaming"].flag_map
          or "gguf_files" in load_engines()["streaming"].flag_map,
          "2f the lane can spell the gguf flags at all")


# ---------------------------------------------------------------- rung 3
def rung_repeated_flag() -> None:
    lane = load_engines()["streaming"]
    argv = build_invocation(
        lane, suite_root=ROOT, checkpoint="/c", panel_dir="/p", out_dir="/o",
        surface="gguf", profile="gguf", cold_run=1, reduce_order="fp32",
        roles="final",
        extra={"source": "gguf", "gguf_files": ["/a.gguf", "/b.gguf"],
               "gguf_repo": REPO, "gguf_revision": REV})
    check(argv.count("--gguf-file") == 2
          and argv[argv.index("--gguf-file") + 1] == "/a.gguf",
          "3a a list value REPEATS its flag (argparse action=append)",
          " ".join(argv))
    joined = " ".join(argv)
    check("/a.gguf,/b.gguf" not in joined and "['/a.gguf'" not in joined,
          "3b it is not joined or str()'d into one path that does not exist")
    empty = build_invocation(
        lane, suite_root=ROOT, checkpoint="/c", panel_dir="/p", out_dir="/o",
        surface="gguf", profile="gguf", cold_run=1, reduce_order="fp32",
        roles="final", extra={"source": "gguf", "gguf_files": []})
    check("--gguf-file" not in empty,
          "3c an empty list drops the flag rather than emitting a bare one")


# ---------------------------------------------------------------- rung 4
def rung_argv(tmp: Path) -> None:
    job = {
        "lane": "streaming", "cold_runs": 2, "profile": "gguf",
        "reduce_order": "fp32", "panel": {"roles": "final"},
        "target": {"repo_id": REPO, "revision": REV, "surface": "gguf",
                   "path": BUILD,
                   "artifact_files": [{"name": n, "bytes": 1} for n in PARTS]},
    }
    (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
    env = dict(os.environ, FIDELITY_FS_ROOT="/fsroot",
               FIDELITY_SUITE_ROOT=str(ROOT),
               FIDELITY_ENGINE_PYTHON=sys.executable)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "invoke_engine.py"),
         "--job", str(tmp / "job.json"), "--lane", "streaming",
         "--cold-run", "1", "--out", "/out", "--print-only"],
        capture_output=True, text=True, env=env)
    argv = proc.stdout.strip()
    check(proc.returncode == 0, "4a invoke_engine composes a gguf capture",
          proc.stdout + proc.stderr)
    check("--source gguf" in argv, "4b it spells --source gguf",
          argv[:200])
    check(all(("/fsroot/models/target/" + n) in argv for n in PARTS),
          "4c every part of the build is on the argv, by local path "
          "(a container's tensor table is per-part; a missing part is a "
          "missing layer, not a short read)")
    check("--profile gguf" in argv, "4d the profile reaches the engine")
    check("--inventory /fsroot/models/bf16-inventory.json" in argv,
          "4e the sealed BF16 inventory is passed: --source gguf REFUSES "
          "without one, an hour into a rental")
    check("--bf16 /fsroot/models/bf16" in argv,
          "4f --bf16 is the OFFICIAL skeleton beside the stage markers -- not a "
          "materialized tree (stream_score materializes the gguf view itself), "
          "and not the container's ephemeral layer (a restarted pod would skip "
          "a `setup` whose 4.2 GB vision shard had evaporated)",
          argv)
    check("--gguf-revision %s" % REV in argv and "--gguf-repo %s" % REPO in argv,
          "4g the artifact's identity is pinned on the argv")

    # and the refusal when the plan forgot to name a build
    job["target"]["artifact_files"] = []
    (tmp / "job-noshelf.json").write_text(json.dumps(job), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "invoke_engine.py"),
         "--job", str(tmp / "job-noshelf.json"), "--lane", "streaming",
         "--cold-run", "1", "--out", "/out", "--print-only"],
        capture_output=True, text=True, env=env)
    check(proc.returncode == 3 and "shelf" in (proc.stdout + proc.stderr),
          "4h a job with no artifact_files is REFUSED here, not run with a "
          "default source that dies after the fetch",
          proc.stdout + proc.stderr)


# ---------------------------------------------------------------- rung 5
def rung_structural() -> None:
    import re

    src = (ROOT / "bin" / "invoke_engine.py").read_text(encoding="utf-8")
    body = src.split("source_by_surface = {", 1)[1].split("}", 1)[0]
    spelled = set(re.findall(r'"([a-z0-9-]+)":\s*"[a-z0-9-]+"', body))
    for lane_name, lane in load_engines().items():
        if "source" not in (lane.flag_map or {}):
            continue
        missing = sorted(set(lane.surfaces) - spelled)
        check(not missing,
              "5 every surface lane %r declares has a --source spelling "
              "(a surface with none reaches the GPU and dies on argparse)"
              % lane_name,
              "unspelled: %s" % ", ".join(missing))


# ---------------------------------------------------------------- rung 6
def modern_bash():
    for cand in (shutil.which("bash"), "/opt/homebrew/bin/bash",
                 "/usr/local/bin/bash"):
        if not cand or not Path(cand).exists():
            continue
        probe = subprocess.run(
            [cand, "-c", '[ "${BASH_VERSINFO[0]}" -gt 4 ] || '
                         '{ [ "${BASH_VERSINFO[0]}" -eq 4 ] && '
                         '[ "${BASH_VERSINFO[1]}" -ge 4 ]; }'],
            capture_output=True)
        if probe.returncode == 0:
            return cand
    return None


def rung_fetch_scope(tmp: Path) -> None:
    """Drive the REAL fetch_target stage with a stub `hf`.

    The property under test is a money property: a whole-repo `hf download` of
    unsloth/GLM-5.3-Flash-GGUF is 2.55 TB for the 200 GB one build needs, and
    the failure mode is a full disk three stages into a paid run.
    """
    bash = modern_bash()
    if bash is None:
        print("  SKIP  6 fetch scoping needs bash 4.4+ (mapfile -d); none found")
        return
    fs, k6 = tmp / "fs", tmp / "k6"
    (fs / ".secrets").mkdir(parents=True)
    (fs / "receipts" / "done").mkdir(parents=True)
    (fs / "logs").mkdir(parents=True)
    (k6 / "venv" / "bin").mkdir(parents=True)
    argv_log = tmp / "hf-argv.txt"
    (k6 / "venv" / "bin" / "hf").write_text(
        '#!/usr/bin/env bash\nprintf "%%s\\n" "$@" > %s\nexit 0\n' % argv_log,
        encoding="utf-8")
    (k6 / "venv" / "bin" / "hf").chmod(0o755)
    # gguf_surface's two post-fetch passes need a python; a stub keeps this
    # rung about ARGV rather than about numpy being installed.
    (k6 / "venv" / "bin" / "python").write_text(
        '#!/usr/bin/env bash\necho "{}"\nexit 0\n', encoding="utf-8")
    (k6 / "venv" / "bin" / "python").chmod(0o755)
    (fs / "job.json").write_text(json.dumps({
        "target": {"repo_id": REPO, "revision": REV, "surface": "gguf",
                   "path": BUILD,
                   "artifact_files": [{"name": n, "bytes": 1} for n in PARTS]},
    }), encoding="utf-8")
    stage = tmp / "bin"
    shutil.copytree(ROOT / "bin", stage, dirs_exist_ok=True)
    proc = subprocess.run(
        [bash, str(stage / "stage_measure.sh"), "fetch_target"],
        capture_output=True, text=True,
        env=dict(os.environ, FIDELITY_FS_ROOT=str(fs), FIDELITY_K6_ROOT=str(k6)))
    got = argv_log.read_text(encoding="utf-8").splitlines() if argv_log.exists() else []
    check(got, "6a the fetch stage ran and called hf",
          (proc.stdout + proc.stderr)[-1200:])
    check(got.count("--include") == len(PARTS),
          "6b the download is scoped to the chosen build's parts, by name "
          "(not the 2.55 TB shelf, and not a %s/* glob that would sweep in a "
          "sidecar the publisher adds tomorrow)" % BUILD,
          " ".join(got))
    check(all(part in got for part in PARTS),
          "6c every part is one literal argument",
          " ".join(got))


# ---------------------------------------------------------------- rung 7
def rung_receipt(tmp: Path) -> None:
    """Seal a GGUF measurement and put it through the registry's validator.

    The fixture's metric is fake and says so; what is real is every field the
    LANE fills in -- and each of them used to be wrong for a GGUF, because the
    defaults were written for the four EXL3 surfaces.
    """
    from fidelity.hfmeta import DEFAULT_PANEL
    from fidelity.receipt import produced_by_block

    scope_src = ROOT / "k6" / "tools" / "gguf-evidence" / "udq4kxl-scope.json"
    scope = json.loads(scope_src.read_text(encoding="utf-8"))["scope"]
    panel = dict(DEFAULT_PANEL.to_dict(), revision="0" * 40)
    job = {
        "recipe": "cloud", "lane": "streaming", "reduce_order": "fp32",
        "cold_runs": 2, "profile": "gguf",
        "target": {
            "repo_id": REPO, "revision": REV, "surface": "gguf", "path": BUILD,
            "size_bytes": 199707324347, "codec": "gguf-k-quant", "bits": 4.0,
            "container": "gguf", "precision_label": BUILD,
            "shard_hash_verification": "full",
            "quantizer_tool": "llama.cpp (quantized_by: Unsloth)",
            "quantizer_version": "gguf quantization_version 2",
            "bits_per_weight_effective": 4.98062529958249,
            "artifact_files": [{"name": n, "bytes": 1} for n in PARTS],
        },
        "scope": scope,
        "panel": panel,
        "reference": {"reference_ref": panel["reference_ref"],
                      "teacher_receipt_sha256": panel["teacher_receipt_sha256"],
                      "teacher_backend_identity_sha256":
                          panel["teacher_backend_identity_sha256"]},
        "measurer": {"name": "selftest", "handle": "selftest", "url": None,
                     "is_artifact_author": False},
        "producer": {"name": "unsloth", "handle": "unsloth",
                     "url": "https://huggingface.co/unsloth"},
        "environment": {"gpu": "NVIDIA A100-SXM4-80GB", "gpu_count": 1,
                        "tensor_parallel": 1, "host": "selftest"},
        "disclosures": [{
            "code": "imatrix_calibrated", "severity": "info",
            "affects_comparability": False, "asserts_provenance": True,
            "detail": "the build declares importance-matrix calibration in its "
                      "own GGUF metadata: entries_count=809.",
            "sources": [{"kind": "hf_file",
                         "uri": "https://huggingface.co/%s/resolve/%s/%s"
                                % (REPO, REV, PARTS[0])}],
        }],
        "produced_by": produced_by_block(ROOT, "bin/measure_cloud.py",
                                         {"lane": "streaming"}),
    }
    (tmp / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    (tmp / "metrics.json").write_text(json.dumps({
        "metric_name": "mean_of_run_means_tokenwise_kld",
        "value": 0.0311111, "run_means": [0.0311111, 0.0311111],
        "evidence_hashes": ["4b2f0c19aa7e5d1188f3c0a94e6b7d2215ac9f83e0d47b6c1a9e2f5083c17e4d"],
        "per_run_report_sha256": [
            "c19a4b2f0c19aa7e5d1188f3c0a94e6b7d2215ac9f83e0d47b6c1a9e2f5083c1",
            "7d2215ac9f83e0d47b6c1a9e2f5083c1c19a4b2f0c19aa7e5d1188f3c0a94e6b"],
        "determinism_note": "selftest fixture; not a real measurement.",
    }, indent=2), encoding="utf-8")
    (tmp / "receipts").mkdir(exist_ok=True)
    out = tmp / "receipt.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "bin" / "seal_receipt.py"),
         "--job", str(tmp / "job.json"), "--receipts", str(tmp / "receipts"),
         "--metrics-json", str(tmp / "metrics.json"), "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT))
    check(proc.returncode == 0,
          "7a a GGUF measurement seals AND passes the registry's own validator",
          (proc.stdout + proc.stderr)[-2500:])
    if not out.is_file():
        return
    doc = json.loads(out.read_text(encoding="utf-8"))
    art = doc["artifact"]
    check(art["container"] == "gguf",
          "7b container is gguf, not the exl3 default four of five surfaces share",
          art["container"])
    check(art.get("path") == BUILD,
          "7c the receipt names WHICH build: repo+revision alone identifies "
          "twelve different artifacts here", repr(art.get("path")))
    check(art["codec"]["family"] == "gguf-k-quant"
          and art["codec"]["quantizer_tool"].startswith("llama.cpp"),
          "7d the codec is a llama.cpp K-quant, not exllamav3 EXL3",
          json.dumps(art["codec"]))
    check(abs((art["codec"]["bits_per_weight_effective"] or 0) - 4.98062529958249) < 1e-9,
          "7e the EFFECTIVE rate is recorded: the name says 4, the bytes say 4.98",
          repr(art["codec"]["bits_per_weight_effective"]))
    treatments = {a["tensor_class"]: a["treatment"] for a in art["scope"]["assignments"]}
    check(treatments.get("lm_head") == "quantized"
          and treatments.get("embed_tokens") == "quantized"
          and treatments.get("attn.qkv") == "quantized",
          "7f the scope says the artifact quantized the head, the embeddings "
          "and attention -- the default would have recorded `unknown` for all "
          "three, which is the M1 lesson in reverse: the artifact DECLARES them",
          json.dumps(treatments))
    check(art["scope"]["head_policy"] == "quantized",
          "7g head_policy is quantized (a routed-experts-only row's is native)")
    check(any(d["code"] == "imatrix_calibrated" for d in doc["disclosures"]),
          "7h the calibration the build declares survives into the receipt")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gguf-lane-"))
    try:
        # the seal directory is named for the measurer handle on purpose: the
        # registry validator refuses a receipt filed under someone else's name,
        # and that check is worth exercising rather than routing around.
        for name in ("argv", "fetch", "selftest"):
            (tmp / name).mkdir(parents=True, exist_ok=True)
        rung_shelf()
        rung_profile()
        rung_repeated_flag()
        rung_argv(tmp / "argv")
        rung_structural()
        rung_fetch_scope(tmp / "fetch")
        rung_receipt(tmp / "selftest")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("selftest_gguf_lane: %d passed, %d failed" % (_pass, _fail))
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
