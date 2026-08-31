#!/usr/bin/env python3
"""EXECUTE every stage of `bin/stage_measure.sh`, offline, with stubbed tools.

WHY THIS EXISTS
---------------
Before this file, exactly two of the eleven stages were ever run by a test:
`fetch_target` (bin/selftest_gguf_lane.py rung 6) and `fetch_panel`
(bin/selftest_shell_guards.sh SEC-01).  Every other stage -- `setup`,
`materialize`, `measure`, `score`, `seal`, and `capture`/`race_capture`/
`verify` for `--role root` -- was "covered" by grepping the file for a
substring:

    check("stage_measure.sh implements capture", "capture)" in stage_sh)

That is the shape of test that passes happily through all four of the
expensive bugs this project actually hit:

  H1  `QP_PIPELINE_ROOT` hard-coded to a JarvisLabs path in the `measure`
      stage's engine argv.  Stalled an A100 at 0% GPU for two hours at
      $1.59/h -- after the bootstrap, a 200 GB fetch and the panel were all
      paid for.
  H2  the same bug again in `score`, found only when a second run got that
      far.
  H3  `FIDELITY_FS_ROOT` / `FIDELITY_K6_ROOT` never exported by the
      controller, so a whole run would have been written into a container's
      ephemeral layer.
  H4  `jqget` printing a JSON null as the four-letter string "None", so every
      `[ -n "$X" ]` guard read an absent key as present: `--preview-of None`,
      a dataset id spelled None, and "panel not uploaded: .../None" instead of
      a message naming the missing key.

So this harness runs the REAL script, under a REAL bash, with the heavy tools
replaced by argv-logging stubs -- the shape bin/selftest_gguf_lane.py already
proved works.  `invoke_engine.py`, `invoke_scorer.py` and the surface adapters
are executed for real where they are pure argv composition, because H1 and H2
lived inside that composition and a stub there would have hidden them.

Four properties are asserted for every stage that has them:

  S-ROOT   it resolves its roots from FIDELITY_FS_ROOT / FIDELITY_K6_ROOT /
           QP_PIPELINE_ROOT and nothing it says, writes or hands onward names
           a provider path (`/home/jl_fs`, `/workspace`).
  S-CLOSED it fails closed on a missing input rather than proceeding.
  S-MARK   `$DONE/<stage>.done` appears on success and NOT on failure.
  S-ARGV   every absolute path on the argv it hands a tool came from the
           environment, not from a literal in the source.

Offline: no network, no GPU, no torch.  Needs bash 4.4+ (`mapfile -d`); macOS
ships bash 3.2 as /bin/bash, so a modern one is located and the whole file
SKIPs loudly if there is none, rather than passing on a shell that cannot run
the code under test.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILED = []
SKIPPED = []

REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40

# The two provider roots.  Neither may appear in anything a stage emits when
# the environment named somewhere else.  H1, H2 and H3 were all this.
PROVIDER_ROOTS = ("/home/jl_fs", "/workspace/")


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:14]:
            print("        %s" % line)


def skip(label, why):
    print("  SKIP  %s (%s)" % (label, why))
    SKIPPED.append(label)


def modern_bash():
    """bash 4.4+, which `mapfile -d` needs.  /bin/bash on macOS is 3.2."""
    for cand in (shutil.which("bash"), "/opt/homebrew/bin/bash",
                 "/usr/local/bin/bash"):
        if not cand or not os.access(cand, os.X_OK):
            continue
        probe = subprocess.run(
            [cand, "-c", '[ "${BASH_VERSINFO[0]}" -gt 4 ] || '
                         '{ [ "${BASH_VERSINFO[0]}" -eq 4 ] && '
                         '[ "${BASH_VERSINFO[1]}" -ge 4 ]; }'],
            capture_output=True)
        if probe.returncode == 0:
            return cand
    return None


STUB_PY = r"""#!/usr/bin/env bash
# Argv-logging stand-in for the venv interpreter.  Scripts named in
# STAGE_REAL_SCRIPTS are EXECUTED, under the real interpreter, because their
# argv composition is the thing under test.
printf 'PY' >> "$STAGE_ARGV_LOG"
for a in "$@"; do printf '\t%s' "$a" >> "$STAGE_ARGV_LOG"; done
printf '\n' >> "$STAGE_ARGV_LOG"
for real in $STAGE_REAL_SCRIPTS; do
  case "${1:-}" in
    *"/$real") exec "$STAGE_REAL_PY" "$@" ;;
  esac
done
# The dataset writer creates its own --out tree; the stage then `du -sh`s it
# under `set -e`. Reproduce just that side effect, so a stubbed capture does
# not fail the stage for a reason the real one never would.
case "${1:-}" in
  *fidelity_dataset.py)
    prev=""
    for a in "$@"; do
      if [ "$prev" = "--out" ]; then mkdir -p "$a"; fi
      prev="$a"
    done
    ;;
esac
exit 0
"""

STUB_HF = r"""#!/usr/bin/env bash
printf 'HF' >> "$STAGE_ARGV_LOG"
for a in "$@"; do printf '\t%s' "$a" >> "$STAGE_ARGV_LOG"; done
printf '\n' >> "$STAGE_ARGV_LOG"
exit 0
"""

STUB_BOOTSTRAP = r"""#!/usr/bin/env bash
printf 'BOOTSTRAP\t%s\t%s\n' "${FIDELITY_FS_ROOT:-UNSET}" \
    "${FIDELITY_ENGINE_ROOT:-${FIDELITY_K6_ROOT:-UNSET}}" >> "$STAGE_ARGV_LOG"
exit 0
"""


class Sandbox:
    """One instance-shaped filesystem, plus the stubs, plus a runner."""

    def __init__(self, tmp, job, *, real_scripts=(), engine_root_env=True,
                 pipeline_root=None):
        self.tmp = Path(tmp)
        self.fs = self.tmp / "fs"
        self.engine = self.tmp / "engine"
        self.pipeline = Path(pipeline_root) if pipeline_root else None
        self.argv_log = self.tmp / "argv.log"
        self.real_scripts = list(real_scripts)
        self.engine_root_env = engine_root_env

        for d in (".secrets", "receipts/done", "logs", "models/bf16", "panel",
                  "models/target"):
            (self.fs / d).mkdir(parents=True, exist_ok=True)
        (self.engine / "venv" / "bin").mkdir(parents=True, exist_ok=True)

        # The on-instance layout: the upload lands at $FS/bin and $FS/<engines>.
        shutil.copytree(ROOT / "bin", self.fs / "bin", dirs_exist_ok=True)
        # bootstrap_measure.sh installs apt packages and clones two repos.  It
        # has its own reasons to exist; what THIS file tests is that `setup`
        # arranges the layout and calls it with the roots it was given.
        (self.fs / "bin" / "bootstrap_measure.sh").write_text(STUB_BOOTSTRAP,
                                                              encoding="utf-8")
        (self.fs / "bin" / "bootstrap_measure.sh").chmod(0o755)
        # stage_panel_paths.py rewrites a sealed 667-artifact panel receipt in
        # place; there is no panel here to rewrite.
        (self.fs / "bin" / "stage_panel_paths.py").write_text(
            "import sys\nprint('stage_panel_paths: stub')\n", encoding="utf-8")

        for name, body in (("python", STUB_PY), ("hf", STUB_HF)):
            p = self.engine / "venv" / "bin" / name
            p.write_text(body, encoding="utf-8")
            p.chmod(0o755)

        # The official metadata skeleton `setup` would otherwise fetch over the
        # network.  Present => the stage's fetch block is a no-op and this file
        # stays offline.
        (self.fs / "models" / "bf16" / "config.json").write_text("{}")
        (self.fs / "models" / "bf16" / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.visual.x": "s-00001.safetensors"}}))

        (self.fs / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (self.fs / ".secrets" / "hf_token").write_text("not-a-real-token")

    # -- helpers ----------------------------------------------------------
    def marker(self, stage):
        return self.fs / "receipts" / "done" / ("%s.done" % stage)

    def env(self, **extra):
        env = dict(os.environ)
        env["FIDELITY_FS_ROOT"] = str(self.fs)
        if self.engine_root_env:
            env["FIDELITY_K6_ROOT"] = str(self.engine)
        env["STAGE_ARGV_LOG"] = str(self.argv_log)
        env["STAGE_REAL_PY"] = sys.executable
        env["STAGE_REAL_SCRIPTS"] = " ".join(self.real_scripts)
        env.pop("QP_PIPELINE_ROOT", None)
        env.pop("FIDELITY_ENGINE_PYTHON", None)
        env.pop("BF16", None)
        env.pop("VENV", None)
        if self.pipeline:
            env["QP_PIPELINE_ROOT"] = str(self.pipeline)
        env.update(extra)
        return env

    def run(self, stage, bash, **extra):
        if self.argv_log.exists():
            self.argv_log.unlink()
        proc = subprocess.run(
            [bash, str(self.fs / "bin" / "stage_measure.sh"), stage],
            capture_output=True, text=True, env=self.env(**extra), cwd=str(self.tmp))
        calls = []
        if self.argv_log.exists():
            for line in self.argv_log.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                calls.append((parts[0], parts[1:]))
        return proc, calls

    # -- the four properties ---------------------------------------------
    def sandbox_roots(self):
        roots = [str(self.fs), str(self.engine), str(self.tmp)]
        if self.pipeline:
            roots.append(str(self.pipeline))
        return roots

    def foreign_paths(self, calls):
        """Absolute path arguments that no root in this environment explains."""
        allowed = tuple(self.sandbox_roots()) + (
            "/usr/", "/bin/", "/sbin/", "/opt/", "/Library/", "/System/",
            "/private/", "/var/", "/tmp/", "/etc/", "/dev/", "/Applications/",
            os.path.dirname(sys.executable) + "/")
        bad = []
        for _, argv in calls:
            for tok in argv:
                if not tok.startswith("/"):
                    continue
                if tok.startswith(allowed):
                    continue
                bad.append(tok)
        return sorted(set(bad))


def provider_leak(text):
    return sorted({r for r in PROVIDER_ROOTS if r in text})


# ---------------------------------------------------------------------------


def job_quant(surface="tr3-published", **over):
    job = {
        "lane": "streaming", "cold_runs": 2, "profile": "tr3-4bpw",
        "reduce_order": "fp32", "role": "quant",
        "keep_student_logits": False,
        "official_bf16_revision": REV_C,
        "panel": {"repo_id": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
                  "revision": REV_B, "roles": "final",
                  "include": ["logits/window-*.safetensors", "*.json"]},
        "target": {"repo_id": "malaiwah/GLM-5.3-Flash-TR3-6bpw",
                   "revision": REV_A, "surface": surface},
    }
    job.update(over)
    return job


def job_root(**over):
    job = {
        "lane": "streaming", "cold_runs": 1, "role": "root",
        "profile": "native-bf16", "official_bf16_revision": REV_C,
        "panel": {"repo_id": "malaiwah/panel", "revision": REV_B},
        "target": {"repo_id": "MiniMaxAI/MiniMax-M3", "revision": REV_A,
                   "surface": "native-bf16"},
        "capture": {"form": "hidden", "schedule": "layer-outer",
                    "panel_dir": "panel-src", "panel_id": "panel--x.y.z",
                    "dataset_id": "malaiwah/ds", "dataset_name": "ds",
                    "author": "malaiwah", "race_workers": 4,
                    "preview_of": None, "sanity_expect": "Paris"},
    }
    job.update(over)
    return job


def main():
    bash = modern_bash()
    if bash is None:
        skip("every rung", "needs bash 4.4+ for `mapfile -d`; none found")
        print("\nselftest_stage_measure: %d skipped" % len(SKIPPED))
        return 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---------------------------------------------------------------
        print("== setup: arranges the layout and calls the bootstrap ==")
        sb = Sandbox(td / "setup", job_quant())
        (sb.fs / "k6" / "patches-v2").mkdir(parents=True)
        (sb.fs / "k6" / "patches-v2" / "0001-x.patch").write_text("patch\n")
        proc, calls = sb.run("setup", bash)
        out = proc.stdout + proc.stderr
        check("setup exits 0 offline", proc.returncode == 0, out[-900:])
        check("S-MARK setup writes its marker", sb.marker("setup").is_file())
        boot = [c for c in calls if c[0] == "BOOTSTRAP"]
        check("setup calls the bootstrap exactly once", len(boot) == 1, calls)
        if boot:
            check("S-ROOT the bootstrap is handed BOTH roots from the "
                  "environment, not defaults",
                  boot[0][1][0] == str(sb.fs) and boot[0][1][1] == str(sb.engine),
                  boot[0][1])
        check("setup stages the patch series under the engine root",
              (sb.engine / "patches-v2" / "0001-x.patch").is_file(),
              sorted(p.name for p in (sb.engine).glob("*")))
        check("S-ROOT setup names no provider path", not provider_leak(out),
              out[-600:])

        # ---------------------------------------------------------------
        print("\n== fetch_target: scoped download + the artifact's own seal ==")
        sb = Sandbox(td / "ft", job_quant())
        proc, calls = sb.run("fetch_target", bash)
        out = proc.stdout + proc.stderr
        hf = [c for c in calls if c[0] == "HF"]
        check("fetch_target exits 0 and calls hf once", proc.returncode == 0
              and len(hf) == 1, out[-900:])
        if hf:
            argv = hf[0][1]
            check("S-ARGV the download lands under FIDELITY_FS_ROOT",
                  "--local-dir" in argv
                  and argv[argv.index("--local-dir") + 1]
                  == str(sb.fs / "models" / "target"), argv)
            check("the pinned revision reaches hf",
                  "--revision" in argv and REV_A in argv, argv)
        seal_calls = [c for c in calls if c[0] == "PY"
                      and any("tr3_surface.py" in a for a in c[1])]
        check("a tr3 release has its published seal verified AND its scope "
              "written, right after the bytes land", len(seal_calls) == 2,
              [c[1][:3] for c in calls])
        check("S-MARK fetch_target writes its marker",
              sb.marker("fetch_target").is_file())
        check("S-ARGV no argument names a path the environment did not supply",
              not sb.foreign_paths(calls), sb.foreign_paths(calls))
        check("S-ROOT fetch_target names no provider path", not provider_leak(out),
              out[-600:])

        # S-CLOSED
        sb2 = Sandbox(td / "ft2", job_quant(target={"revision": REV_A,
                                                    "surface": "tr3-published"}))
        proc2, _ = sb2.run("fetch_target", bash)
        check("S-CLOSED a job with no target.repo_id is refused (exit 2)",
              proc2.returncode == 2, proc2.stdout + proc2.stderr)
        check("S-MARK ...and no marker is left behind",
              not sb2.marker("fetch_target").is_file())

        # ---------------------------------------------------------------
        print("\n== fetch_panel: include-scoped, data never parsed by the shell ==")
        sb = Sandbox(td / "fp", job_quant())
        proc, calls = sb.run("fetch_panel", bash)
        out = proc.stdout + proc.stderr
        hf = [c for c in calls if c[0] == "HF"]
        check("fetch_panel exits 0 and calls hf once",
              proc.returncode == 0 and len(hf) == 1, out[-900:])
        if hf:
            argv = hf[0][1]
            check("the panel is fetched as a DATASET repo",
                  "--repo-type" in argv and argv[argv.index("--repo-type") + 1]
                  == "dataset", argv)
            check("both include patterns arrive, one literal argument each",
                  argv.count("--include") == 2
                  and "logits/window-*.safetensors" in argv, argv)
            check("S-ARGV the panel lands under FIDELITY_FS_ROOT",
                  argv[argv.index("--local-dir") + 1] == str(sb.fs / "panel"),
                  argv)
        check("S-MARK fetch_panel writes its marker",
              sb.marker("fetch_panel").is_file())
        check("S-ROOT fetch_panel names no provider path", not provider_leak(out),
              out[-600:])

        # ---------------------------------------------------------------
        print("\n== materialize: only the surfaces that need it, and skip-with-marker ==")
        sb = Sandbox(td / "mat", job_quant("tr3-published"))
        proc, calls = sb.run("materialize", bash)
        out = proc.stdout + proc.stderr
        mat = [c for c in calls if c[0] == "PY"
               and any("exl3hf_surface.py" in a for a in c[1])]
        check("a tr3 release IS materialized (its natives share shards with "
              "the routed payloads)", proc.returncode == 0 and len(mat) == 1,
              out[-900:])
        if mat:
            argv = mat[0][1]
            check("S-ARGV materialize reads and writes under the fs root",
                  argv[argv.index("--root") + 1] == str(sb.fs / "models/target")
                  and argv[argv.index("--out") + 1]
                  == str(sb.fs / "models/target-bf16-materialized"), argv)
            check("S-ARGV the official index it checks against is the fs one",
                  argv[argv.index("--official-index") + 1]
                  == str(sb.fs / "models/bf16/model.safetensors.index.json"), argv)
        check("S-MARK materialize writes its marker",
              sb.marker("materialize").is_file())
        check("S-ARGV no foreign path", not sb.foreign_paths(calls),
              sb.foreign_paths(calls))

        sb = Sandbox(td / "mat2", job_quant("native-bf16"))
        proc, calls = sb.run("materialize", bash)
        check("a surface that needs no materialization skips AND marks done "
              "(so a resume does not re-enter it)",
              proc.returncode == 0 and sb.marker("materialize").is_file()
              and not [c for c in calls if c[0] == "PY"],
              proc.stdout + proc.stderr)

        # ---------------------------------------------------------------
        # H1: the measure stage's engine argv.  invoke_engine.py runs FOR REAL
        # here; a stub would have hidden the hard-coded QP_PIPELINE_ROOT that
        # stalled an A100 at 0% GPU for two hours.
        print("\n== measure: the engine argv, composed by the real invoke_engine ==")
        sb = Sandbox(td / "meas", job_quant(), real_scripts=["invoke_engine.py"])
        proc, calls = sb.run("measure", bash)
        out = proc.stdout + proc.stderr
        engine_calls = [c for c in calls if c[0] == "PY"
                        and any("stream_score.py" in a for a in c[1])]
        check("measure runs one capture per cold run (cold_runs=2)",
              proc.returncode == 0 and len(engine_calls) == 2, out[-1200:])
        if engine_calls:
            argv = engine_calls[0][1]
            pr = argv[argv.index("--pipeline-root") + 1] \
                if "--pipeline-root" in argv else ""
            check("H1 --pipeline-root is DERIVED from the engine root the "
                  "controller exported, with QP_PIPELINE_ROOT unset",
                  pr == str(sb.engine / "pipeline"),
                  "got %r; a literal here stalls a paid box at 0%% GPU" % pr)
            check("S-ARGV the capture writes into the run directory it was given",
                  any(a == str(sb.fs / "receipts" / "run-1") for a in argv), argv)
        check("S-ARGV no argument names a path the environment did not supply",
              not sb.foreign_paths(calls), sb.foreign_paths(calls))
        check("S-ROOT measure names no provider path anywhere in its output",
              not provider_leak(out), out[-800:])
        check("S-MARK measure writes its marker", sb.marker("measure").is_file())

        # ...and an explicit QP_PIPELINE_ROOT still wins.
        sb = Sandbox(td / "meas2", job_quant(), real_scripts=["invoke_engine.py"],
                     pipeline_root=str(td / "meas2" / "explicit-pipe"))
        _, calls = sb.run("measure", bash)
        ec = [c for c in calls if c[0] == "PY"
              and any("stream_score.py" in a for a in c[1])]
        check("an explicit QP_PIPELINE_ROOT overrides the derivation",
              ec and ec[0][1][ec[0][1].index("--pipeline-root") + 1]
              == str(td / "meas2" / "explicit-pipe"), ec[0][1] if ec else calls)

        # S-CLOSED: the stage refuses before it runs anything when the venv is
        # absent.  A bare `exit 127` used to be the only signal.
        sb = Sandbox(td / "meas3", job_quant())
        (sb.engine / "venv" / "bin" / "python").unlink()
        proc, calls = sb.run("measure", bash)
        check("S-CLOSED measure refuses with a named remedy when the venv "
              "interpreter is missing (exit 3, not 127)",
              proc.returncode == 3 and "setup" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-500:])
        check("S-MARK ...and leaves no marker",
              not sb.marker("measure").is_file())

        # ---------------------------------------------------------------
        # H2: the same defect, one stage later.  It was found only when a
        # second paid run got this far.
        print("\n== score: the scorer argv, composed by the real invoke_scorer ==")
        sb = Sandbox(td / "score", job_quant(), real_scripts=["invoke_scorer.py"])
        for n in (1, 2):
            d = sb.fs / "receipts" / ("run-%d" % n)
            d.mkdir(parents=True)
            (d / "capture-receipt.json").write_text("{}")
            (d / "logits").mkdir()
            (d / "logits" / "w.safetensors").write_bytes(b"x" * 16)
        proc, calls = sb.run("score", bash)
        out = proc.stdout + proc.stderr
        sc = [c for c in calls if c[0] == "PY"
              and any("kld_report" in a for a in c[1])]
        check("score runs the lane's pinned scorer once",
              proc.returncode == 0 and len(sc) == 1, out[-1200:])
        if sc:
            argv = sc[0][1]
            pr = argv[argv.index("--pipeline-root") + 1] \
                if "--pipeline-root" in argv else ""
            check("H2 the scorer's --pipeline-root is derived from the engine "
                  "root too", pr == str(sb.engine / "pipeline"), argv)
            # `.resolve()` in invoke_scorer follows macOS's /var -> /private/var
            # symlink, so the comparison is between realpaths.
            real_rcpt = os.path.realpath(str(sb.fs / "receipts"))
            check("S-ARGV both run directories are passed, by fs-root path",
                  os.path.join(real_rcpt, "run-1") in argv
                  and os.path.join(real_rcpt, "run-2") in argv, argv)
            check("S-ARGV the teacher panel is the fetched one",
                  "%s/panel" % sb.fs in argv, argv)
        check("the transient fp32 student logits are deleted "
              "(63 GB otherwise times out the receipts pull)",
              not (sb.fs / "receipts" / "run-1" / "logits").exists(),
              "keep_student_logits was false")
        check("S-ARGV no foreign path", not sb.foreign_paths(calls),
              sb.foreign_paths(calls))
        check("S-ROOT score names no provider path", not provider_leak(out),
              out[-800:])
        check("S-MARK score writes its marker", sb.marker("score").is_file())

        # S-CLOSED: a missing capture receipt must stop the stage, not produce
        # an empty aggregate.
        sb = Sandbox(td / "score2", job_quant(), real_scripts=["invoke_scorer.py"])
        d = sb.fs / "receipts" / "run-1"
        d.mkdir(parents=True)
        (d / "capture-receipt.json").write_text("{}")
        proc, _ = sb.run("score", bash)
        check("S-CLOSED score refuses when a cold run has no capture receipt",
              proc.returncode != 0
              and "no capture receipt" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-500:])
        check("S-MARK ...and leaves no marker", not sb.marker("score").is_file())

        # ---------------------------------------------------------------
        print("\n== seal ==")
        sb = Sandbox(td / "seal", job_quant())
        proc, calls = sb.run("seal", bash)
        sealer = [c for c in calls if c[0] == "PY"
                  and any("seal_receipt.py" in a for a in c[1])]
        check("seal invokes the sealer with the job and the receipts tree",
              proc.returncode == 0 and len(sealer) == 1
              and str(sb.fs / "receipts") in sealer[0][1], calls)
        check("S-MARK seal writes its marker", sb.marker("seal").is_file())

        sb = Sandbox(td / "seal2", job_quant(), real_scripts=["seal_receipt.py"])
        proc, _ = sb.run("seal", bash)
        check("S-CLOSED a sealer that fails does not leave a done marker "
              "(a resume would otherwise skip straight to teardown)",
              proc.returncode != 0 and not sb.marker("seal").is_file(),
              (proc.stdout + proc.stderr)[-500:])

        # ---------------------------------------------------------------
        print("\n== capture / verify (--role root) ==")
        sb = Sandbox(td / "cap", job_root())
        (sb.fs / "panel-src").mkdir()
        proc, calls = sb.run("capture", bash)
        out = proc.stdout + proc.stderr
        cap = [c for c in calls if c[0] == "PY"
               and any("fidelity_dataset.py" in a for a in c[1])]
        check("capture runs the dataset writer once",
              proc.returncode == 0 and len(cap) == 1, out[-1000:])
        if cap:
            argv = cap[0][1]
            check("S-ARGV the panel is the uploaded one, resolved under the fs root",
                  str(sb.fs / "panel-src") in argv, argv)
            check("S-ARGV the model is the LOCAL tree fetch_target wrote",
                  str(sb.fs / "models" / "target") in argv, argv)
            check("the recorded identity stays the PUBLISHED repo, not a path "
                  "on a machine that will not exist",
                  "--repository" in argv
                  and argv[argv.index("--repository") + 1] == "MiniMaxAI/MiniMax-M3",
                  argv)
        check("S-MARK capture writes its marker", sb.marker("capture").is_file())
        check("S-ARGV no foreign path", not sb.foreign_paths(calls),
              sb.foreign_paths(calls))

        # H4: a JSON null must read as ABSENT.
        j = job_root()
        j["capture"] = dict(j["capture"], panel_dir=None)
        sb = Sandbox(td / "cap2", j)
        proc, _ = sb.run("capture", bash)
        msg = proc.stdout + proc.stderr
        check("H4 a null capture.panel_dir is refused BY NAME, not chased to "
              "a directory literally called None",
              proc.returncode == 2 and "no capture.panel_dir" in msg
              and "None" not in msg, msg[-500:])
        check("S-MARK ...and leaves no marker", not sb.marker("capture").is_file())

        j = job_root()
        j["capture"] = dict(j["capture"], dataset_id=None)
        sb = Sandbox(td / "cap3", j)
        (sb.fs / "panel-src").mkdir()
        proc, _ = sb.run("capture", bash)
        check("H4 a null capture.dataset_id is refused before anything runs",
              proc.returncode == 2
              and "no capture.dataset_id" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-400:])

        sb = Sandbox(td / "cap4", job_quant())      # role=quant
        proc, _ = sb.run("capture", bash)
        check("S-CLOSED the capture stage refuses a --role quant job",
              proc.returncode == 2
              and "role=quant" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-400:])

        sb = Sandbox(td / "ver", job_root())
        (sb.fs / "dataset").mkdir()
        proc, calls = sb.run("verify", bash)
        ver = [c for c in calls if c[0] == "PY"
               and any("fidelity_dataset.py" in a for a in c[1])]
        check("verify recomputes the seal AND describes the dataset, before "
              "the box is destroyed",
              proc.returncode == 0 and len(ver) == 2
              and "verify" in ver[0][1] and "describe" in ver[1][1], calls)
        check("S-MARK verify writes its marker", sb.marker("verify").is_file())

        # ---------------------------------------------------------------
        print("\n== race mode (--role root --race) ==")
        sb = Sandbox(td / "rb", job_root())
        proc, calls = sb.run("race_bootstrap", bash)
        check("S-CLOSED race_bootstrap refuses when the fetch produced no "
              "weight_map: without it there is no fetch ORDER, only a download",
              proc.returncode == 3
              and "map from" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-500:])
        check("S-MARK ...and leaves no marker",
              not sb.marker("race_bootstrap").is_file())

        sb = Sandbox(td / "rb2", job_root())
        (sb.fs / "models" / "target" / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1", "b": "s2"}}))
        proc, calls = sb.run("race_bootstrap", bash)
        out = proc.stdout + proc.stderr
        check("race_bootstrap succeeds once the index is there, and reports "
              "the shard count it will order by",
              proc.returncode == 0 and "2 tensors over 2 shards" in out,
              out[-600:])
        check("S-MARK race_bootstrap writes its marker",
              sb.marker("race_bootstrap").is_file())
        hf = [c for c in calls if c[0] == "HF"]
        check("race_bootstrap fetches NO shard (--include per name only)",
              hf and not any(a.endswith(".safetensors") for a in hf[0][1]),
              hf[0][1] if hf else calls)

        sb = Sandbox(td / "rc", job_root())
        (sb.fs / "panel-src").mkdir()
        proc, calls = sb.run("race_capture", bash)
        out = proc.stdout + proc.stderr
        cap = [c for c in calls if c[0] == "PY"
               and any("fidelity_dataset.py" in a for a in c[1])]
        check("race_capture runs the fused fetch+capture once",
              proc.returncode == 0 and len(cap) == 1, out[-1000:])
        if cap:
            argv = cap[0][1]
            check("H4 a null capture.preview_of drops the flag entirely -- it "
                  "does not become `--preview-of None`",
                  "--preview-of" not in argv and "None" not in argv, argv)
            check("the race report is written under the receipts tree",
                  str(sb.fs / "receipts" / "race-fetch-report.json") in argv, argv)
            check("race mode streams layers rather than resident-loading them",
                  "--layer-residency" in argv
                  and argv[argv.index("--layer-residency") + 1] == "stream", argv)
        check("S-ARGV no foreign path", not sb.foreign_paths(calls),
              sb.foreign_paths(calls))
        check("S-MARK race_capture writes its marker",
              sb.marker("race_capture").is_file())

        j = job_root()
        j["capture"] = dict(j["capture"], schedule="window-outer")
        sb = Sandbox(td / "rc2", j)
        (sb.fs / "panel-src").mkdir()
        proc, _ = sb.run("race_capture", bash)
        check("S-CLOSED race mode refuses a window-outer schedule (it would "
              "read the tree once per window)",
              proc.returncode == 2
              and "layer-outer" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-400:])

        # ---------------------------------------------------------------
        print("\n== cross-cutting ==")
        sb = Sandbox(td / "xx", job_quant())
        proc, _ = sb.run("no_such_stage", bash)
        check("S-CLOSED an unknown stage is refused and the usage names every "
              "stage this file implements", proc.returncode == 2
              and all(s in proc.stderr for s in
                      ("setup", "fetch_target", "fetch_panel", "measure",
                       "score", "seal", "capture", "verify",
                       "race_bootstrap", "race_capture")),
              proc.stderr)

        # Every stage in that usage line must actually be reachable, or the
        # driver advertises a stage the case statement does not implement.
        sb = Sandbox(td / "xy", job_quant())
        unknown = []
        for stage in ("setup", "fetch_target", "fetch_panel", "materialize",
                      "measure", "score", "seal", "capture", "verify",
                      "race_bootstrap", "race_capture"):
            p, _ = Sandbox(td / ("xy-" + stage), job_quant()).run(stage, bash)
            if "unknown stage" in p.stderr:
                unknown.append(stage)
        check("every stage the controller can ask for is implemented",
              not unknown, unknown)

        # A finished stage is a no-op: this is what makes a spot preemption
        # cost one stage instead of the whole run.
        sb = Sandbox(td / "resume", job_quant())
        sb.marker("fetch_panel").parent.mkdir(parents=True, exist_ok=True)
        sb.marker("fetch_panel").write_text("")
        proc, calls = sb.run("fetch_panel", bash)
        check("a stage whose marker exists is skipped without re-running it",
              proc.returncode == 0 and not calls
              and "already done" in proc.stdout, proc.stdout)

    print()
    if FAILED:
        print("selftest_stage_measure: %d FAILED" % len(FAILED))
        return 1
    print("selftest_stage_measure: all passed%s"
          % (" (%d skipped)" % len(SKIPPED) if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
