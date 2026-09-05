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
  H3  `FIDELITY_FS_ROOT` / `FIDELITY_ENGINE_ROOT` never exported by the
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

  S-ROOT   it resolves its roots from FIDELITY_FS_ROOT / FIDELITY_ENGINE_ROOT /
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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
from fidelity import common, jobcontract
FAILED = []
SKIPPED = []

REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40
PANEL_BINDING = {"schema": "fidelity.resolved-panel.v1", "panel_id": "panel--x.y.z"}
PANEL_BINDING_BYTES = json.dumps(
    PANEL_BINDING, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False, allow_nan=False).encode("utf-8")
PANEL_BINDING_SHA = hashlib.sha256(PANEL_BINDING_BYTES).hexdigest()

def execution_attempt(attempt_id="1" * 24):
    return {"number": 1, "kind": "local-container", "attempt_id": attempt_id}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


SELFTEST_BUNDLE = jobcontract.finalize_bundle_manifest(
    [{"path": "bin/stage_measure.sh", "bytes": 1, "sha256": "d" * 64}],
    "stage-selftest")
SELFTEST_CONTROL = jobcontract.finalize_bundle_manifest(
    [{"path": "bin/fidelity/jobcontract.py", "bytes": 1,
      "sha256": "e" * 64}], "stage-selftest-control")
SELFTEST_CONTROL["schema"] = "fidelity-suite/control-plane-manifest.v1"
SELFTEST_REGISTRY = {
    "path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "f" * 64}
SELFTEST_BUNDLE_CONTRACT = hashlib.sha256(canonical({
    "bundle": SELFTEST_BUNDLE, "registry": SELFTEST_REGISTRY})).hexdigest()
SELFTEST_SHARDS = [{"path": "model-00001-of-00001.safetensors", "bytes": 17}]
SELFTEST_ALLOWLIST_BYTES = b'["model.unused"]'
SELFTEST_ALLOWLIST_SHA = hashlib.sha256(SELFTEST_ALLOWLIST_BYTES).hexdigest()
SELFTEST_CONFIG_BYTES = b"{}\n"
SELFTEST_INDEX_BYTES = canonical({
    "weight_map": {"model.x": SELFTEST_SHARDS[0]["path"]}})
SELFTEST_SHARD_BYTES = b"x" * 17
SELFTEST_BF16_CONFIG_BYTES = b"{}"
SELFTEST_BF16_INDEX_BYTES = json.dumps({
    "weight_map": {"model.visual.x": "s-00001.safetensors"}
}).encode("utf-8")


def target_contract(repo, revision, surface):
    download_manifest = [
        {"path": "config.json", "bytes": len(SELFTEST_CONFIG_BYTES)},
        {"path": SELFTEST_SHARDS[0]["path"],
         "bytes": SELFTEST_SHARDS[0]["bytes"]},
        {"path": "model.safetensors.index.json",
         "bytes": len(SELFTEST_INDEX_BYTES)},
    ]
    target = {
        "repo_id": repo,
        "revision": revision,
        "path": None,
        "surface": surface,
        "codec": "exl3-mcg" if surface == "tr3-published" else "bf16",
        "config_sha256": hashlib.sha256(SELFTEST_CONFIG_BYTES).hexdigest(),
        "index_sha256": hashlib.sha256(SELFTEST_INDEX_BYTES).hexdigest(),
        "shard_manifest_sha256":
            hashlib.sha256(canonical(SELFTEST_SHARDS)).hexdigest(),
        "model_bytes": 17,
        "download_manifest": download_manifest,
        "download_bytes_total":
            sum(row["bytes"] for row in download_manifest),
        "download_manifest_sha256":
            hashlib.sha256(canonical(download_manifest)).hexdigest(),
        "bits": 6 if surface == "tr3-published" else 16,
        "shards": SELFTEST_SHARDS,
    }
    if surface == "tr3-published":
        target["official_bf16_identity"] = {
            "config_sha256":
                hashlib.sha256(SELFTEST_BF16_CONFIG_BYTES).hexdigest(),
            "index_sha256":
                hashlib.sha256(SELFTEST_BF16_INDEX_BYTES).hexdigest(),
            "config_bytes": len(SELFTEST_BF16_CONFIG_BYTES),
            "index_bytes": len(SELFTEST_BF16_INDEX_BYTES),
        }
    return target

# The two provider roots.  Neither may appear in anything a stage emits when
# the environment named somewhere else.  H1, H2 and H3 were all this.
PROVIDER_ROOTS = ("/home/jl_fs", "/workspace/")



def self_consistent_job(document):
    job = json.loads(json.dumps(document))
    job.pop("job_id", None)
    job.pop("job_id_full", None)
    digest = hashlib.sha256(canonical(
        jobcontract.job_identity_projection(job))).hexdigest()
    job["job_id_full"] = digest
    job["job_id"] = digest[:16]
    return job
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
if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  printf 'HF_TOKEN_ENV_LEAK\n' >> "$STAGE_ARGV_LOG"
  exit 89
fi
public=0
previous=
for argument in "$@"; do
  if [ "$previous" = "--repo-type" ] && [ "$argument" = "dataset" ]; then
    public=1
  fi
  previous="$argument"
done
if [ "$public" = 1 ]; then
  expected_home="$FIDELITY_FS_ROOT/.hf-public-panel"
  if [ "${HF_ENDPOINT:-}" != "https://huggingface.co" ] \
      || [ "${HF_HUB_DISABLE_IMPLICIT_TOKEN:-}" != 1 ] \
      || [ "${HF_HOME:-}" != "$expected_home" ] \
      || [ "${HF_HUB_CACHE:-}" != "$expected_home/hub" ] \
      || [ "${HF_TOKEN_PATH:-}" != "$expected_home/no-token" ]; then
    printf 'HF_PUBLIC_ENV_WRONG\t%s\t%s\t%s\t%s\t%s\n' \
      "${HF_ENDPOINT:-UNSET}" "${HF_HUB_DISABLE_IMPLICIT_TOKEN:-UNSET}" \
      "${HF_HOME:-UNSET}" "${HF_HUB_CACHE:-UNSET}" \
      "${HF_TOKEN_PATH:-UNSET}" >> "$STAGE_ARGV_LOG"
    exit 91
  fi
  printf 'HF_PUBLIC_ENV\tanonymous\tofficial\tisolated\n' >> "$STAGE_ARGV_LOG"
elif [ "${HF_TOKEN_PATH:-}" != "$FIDELITY_FS_ROOT/.secrets/hf_token" ]; then
  printf 'HF_TOKEN_PATH_WRONG\t%s\n' "${HF_TOKEN_PATH:-UNSET}" >> "$STAGE_ARGV_LOG"
  exit 90
fi
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
                 pipeline_root=None, finalize_job_doc=True):
        self.tmp = Path(tmp)
        self.fs = self.tmp / "fs"
        self.engine = self.tmp / "engine"
        self.pipeline = Path(pipeline_root) if pipeline_root else None
        self.argv_log = self.tmp / "argv.log"
        self.real_scripts = list(real_scripts)
        self.engine_root_env = engine_root_env
        if finalize_job_doc:
            job = jobcontract.finalize_job(job)
        for d in ("receipts", "logs", "models/target", "models/bf16",
                  ".secrets"):
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
        (self.fs / "models" / "target" / "config.json").write_bytes(
            SELFTEST_CONFIG_BYTES)
        (self.fs / "models" / "target" /
         "model.safetensors.index.json").write_bytes(SELFTEST_INDEX_BYTES)
        (self.fs / "models" / "target" /
         SELFTEST_SHARDS[0]["path"]).write_bytes(SELFTEST_SHARD_BYTES)
        (self.fs / "models" / "bf16" / "config.json").write_bytes(
            SELFTEST_BF16_CONFIG_BYTES)
        (self.fs / "models" / "bf16" /
         "model.safetensors.index.json").write_bytes(
             SELFTEST_BF16_INDEX_BYTES)

        (self.fs / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (self.fs / ".secrets" / "hf_token").write_text("not-a-real-token")
        (self.fs / ".secrets" / "hf_token").chmod(0o600)
        binding_rel = (job.get("panel") or {}).get("binding_path")
        if binding_rel == "panel-binding.json":
            (self.fs / binding_rel).write_bytes(PANEL_BINDING_BYTES)
        allowlist_rel = ((job.get("capture") or {}).get(
            "unexpected_tensor_allowlist") or {}).get("path")
        if allowlist_rel == "allowlist.json":
            (self.fs / allowlist_rel).write_bytes(SELFTEST_ALLOWLIST_BYTES)

    # -- helpers ----------------------------------------------------------
    def marker(self, stage):
        return self.fs / "receipts" / "done" / ("%s.done" % stage)

    def write_bound_marker(self, stage):
        raw = (self.fs / "job.json").read_bytes()
        job = json.loads(raw.decode("utf-8"))
        self.marker(stage).parent.mkdir(parents=True, exist_ok=True)
        self.marker(stage).write_text(
            "job_id_full=%s\njob_sha256=%s\nstage=%s\n"
            "completed_at=2026-01-02T03:04:05Z\n"
            % (job["job_id_full"], hashlib.sha256(raw).hexdigest(), stage),
            encoding="utf-8")

    def write_target_census(self):
        raw = (self.fs / "job.json").read_bytes()
        job = jobcontract.parse_job_bytes(raw)
        target = job["target"]
        common.write_json(
            self.fs / "receipts" / "fetch-target-census.json",
            common.seal({
                "schema": "fidelity.fetch-target-census.v1",
                "verified_at": "2026-01-02T03:04:05Z",
                "job_id_full": job["job_id_full"],
                "job_file_sha256": hashlib.sha256(raw).hexdigest(),
                "repository": target["repo_id"],
                "revision": target["revision"],
                "config_sha256": target["config_sha256"],
                "index_sha256": target["index_sha256"],
                "shard_manifest_sha256": target["shard_manifest_sha256"],
                "model_bytes": target["model_bytes"],
                "shards": target["shards"],
                "index_shards": [row["path"] for row in target["shards"]],
            }))
        self.write_bound_marker("fetch_target")

    def env(self, **extra):
        env = dict(os.environ)
        env["FIDELITY_FS_ROOT"] = str(self.fs)
        if self.engine_root_env:
            env["FIDELITY_ENGINE_ROOT"] = str(self.engine)
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

    def run(self, stage, bash, provision_target=True, **extra):
        if provision_target and stage in (
                "materialize", "measure", "capture", "capture_repeat"):
            self.write_target_census()
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
        roots = tuple(self.sandbox_roots())
        bad = []
        for _kind, argv in calls:
            for token in argv:
                if not token.startswith("/"):
                    continue
                if token.startswith(roots):
                    continue
                bad.append(token)
        return sorted(set(bad))


def provider_leak(text):
    return sorted({r for r in PROVIDER_ROOTS if r in text})


# ---------------------------------------------------------------------------


def job_quant(surface="tr3-published", **over):
    if surface == "tr3-published":
        profile_id, source, bits = "tr3-6bpw", "tr3", 6
    elif surface == "native-bf16":
        profile_id, source, bits = "native-bf16", "native", 16
    else:
        profile_id, source, bits = "fixture-%s" % surface, "native", 16
    job = {
        "schema": "fidelity-suite/job.v2",
        "execution_attempt": execution_attempt(),
        "bundle": SELFTEST_BUNDLE,
        "control_plane": SELFTEST_CONTROL,
        "bundle_registry": SELFTEST_REGISTRY,
        "bundle_contract_sha256": SELFTEST_BUNDLE_CONTRACT,
        "scope": {"kind": "stage-selftest"},
        "timing": {
            "kind": "stage-selftest",
            "runtime_profile": {
                "decode_cache": "none",
                "decode_threads": 28,
                "reader_threads": 28,
            },
        },
        "lane": "streaming", "cold_runs": 2,
        "profile": {
            "profile_id": profile_id, "lane": "streaming",
            "source": source, "surface": surface, "bits": bits,
        },
        "reduce_order": "fp32", "role": "quant",
        "recipe": "cloud",
        "runtime": {
            "decode_cache": "none",
            "decode_threads": 28,
            "reader_threads": 28,
            "device": "cuda",
            "reduce_order": "fp32",
        },
        "environment": {}, "measurer": {},
        "produced_by": {
            "dependencies": {
                "profile": profile_id,
                "lane": "streaming", "provider": "runpod"}},
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 1, "min_memory_gb": 1,
            "expected_vram_bytes": 1,
        },
        "keep_student_logits": False,
        "official_bf16_revision": REV_C,
        "reference": {
            "reference_ref": "hf://selftest/reference@" + REV_B,
            "teacher_receipt_sha256": "4" * 64,
            "teacher_backend_identity_sha256": "5" * 64,
        },
        "scoring": {
            "schema": "fidelity-suite/kld-scoring.v1",
            "device": "cuda", "chunk_positions": 512,
            "compute_dtype": "float64",
            "direction": "reference_to_candidate",
            "vocabulary": "full",
            "reduction": "mean_of_run_means_tokenwise_kld",
        },
        "panel": {
            "repo_id": "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits",
            "revision": REV_B, "role": "final", "roles": "final",
            "include": ["logits/window-*.safetensors", "*.json"],
            "panel_receipt_sha256": "6" * 64,
            "reference_ref": "hf://selftest/reference@" + REV_B,
            "teacher_receipt_sha256": "4" * 64,
            "teacher_backend_identity_sha256": "5" * 64,
        },
        "target": target_contract(
            "malaiwah/GLM-5.3-Flash-TR3-6bpw", REV_A, surface),
    }
    job.update(over)
    job.pop("job_id", None)
    job.pop("job_id_full", None)
    return jobcontract.finalize_job(job)


def job_root(**over):
    job = {
        "schema": "fidelity-suite/job.v2",
        "execution_attempt": execution_attempt(),
        "bundle": SELFTEST_BUNDLE,
        "control_plane": SELFTEST_CONTROL,
        "bundle_registry": SELFTEST_REGISTRY,
        "bundle_contract_sha256": SELFTEST_BUNDLE_CONTRACT,
        "scope": {"kind": "stage-selftest"},
        "timing": {"kind": "stage-selftest"},
        "lane": "streaming", "cold_runs": 2, "role": "root",
        "recipe": "cloud",
        "runtime": {}, "environment": {}, "measurer": {},
        "produced_by": {
            "dependencies": {
                "profile": "root-hf-transformers-bf16",
                "lane": "streaming", "provider": "local-container"}},
        "resource_requirements": {
            "workspace_available_bytes_minimum": 1,
            "container_available_bytes_minimum": 1,
            "min_vcpu_count": 1, "min_memory_gb": 1,
            "expected_vram_bytes": 1,
        },
        "profile": {
            "profile_id": "root-hf-transformers-bf16",
            "lane": "root", "source": "native",
            "surface": "native-bf16", "form": "hidden",
            "engine": "hf-transformers",
            "compute_dtype": "bfloat16",
            "device": "cuda",
            "schedule": "two-fresh-process-qualification",
        },
        "official_bf16_revision": REV_C,
        "panel": {"repo_id": "malaiwah/panel", "revision": REV_B,
                  "binding_path": "panel-binding.json",
                  "binding_file_sha256": PANEL_BINDING_SHA,
                  "resolved_binding": PANEL_BINDING},
        "target": target_contract(
            "MiniMaxAI/MiniMax-M3", REV_A, "native-bf16"),
        "capture": {"form": "hidden", "schedule": "layer-outer",
                    "engine": "hf-transformers", "dtype": "bfloat16",
                    "device": "cuda",
                    "panel_dir": "panel-src", "panel_id": "panel--x.y.z",
                    "dataset_id": "malaiwah/ds", "dataset_name": "ds",
                    "dataset_repository": "malaiwah/mm3-root-v1",
                    "author": "malaiwah", "race_workers": 4,
                    "publish_root_to": "malaiwah/mm3-root-v1",
                    "dataset_license": "mit", "weights_license": None,
                    "preview_of": None, "race": False,
                    "replay_device": "numpy", "replay_dtype": "float32",
                    "vocab_chunk": 8192,
                    "replay": {
                        "device": "numpy", "dtype": "float32",
                        "vocab_chunk": 8192},
                    "root_protocol": {
                        "schedule": "two-fresh-process-qualification",
                        "fresh_processes": 2, "run_count_per_process": 1,
                        "exact_self_comparison": True,
                        "qualification_required": True,
                        "canonical_publication_required": True,
                        "publication_mode": "canonical-public"},
                    "unexpected_tensor_allowlist": {
                        "path": "allowlist.json",
                        "artifact_sha256": SELFTEST_ALLOWLIST_SHA,
                        "canonical_sorted_names_sha256":
                            SELFTEST_ALLOWLIST_SHA}},
    }
    job.update(over)
    job.pop("job_id", None)
    job.pop("job_id_full", None)
    return jobcontract.finalize_job(job)


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
        (sb.fs / "engines" / "patches-v2").mkdir(parents=True)
        (sb.fs / "engines" / "patches-v2" / "0001-x.patch").write_text("patch\n")
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
        wrong_official = job_quant()
        wrong_official["target"]["official_bf16_identity"][
            "config_sha256"] = "0" * 64
        refused_setup = Sandbox(td / "setup-wrong-official", wrong_official)
        proc, calls = refused_setup.run("setup", bash)
        check("setup refuses official BF16 metadata outside the sealed job",
              proc.returncode != 0
              and not [call for call in calls if call[0] == "BOOTSTRAP"]
              and not refused_setup.marker("setup").exists()
              and "differs from job identity" in proc.stderr,
              proc.stdout + proc.stderr)

        # ---------------------------------------------------------------
        print("\n== fetch_target: scoped download + the artifact's own seal ==")
        sb = Sandbox(td / "ft", job_quant())
        proc, calls = sb.run(
            "fetch_target", bash, HF_TOKEN="ambient-secret",
            HUGGING_FACE_HUB_TOKEN="ambient-secret-two",
            HF_TOKEN_PATH="/hostile/token")
        out = proc.stdout + proc.stderr
        hf = [c for c in calls if c[0] == "HF"]
        check("fetch_target exits 0, uses file-only token path, and calls hf once",
              proc.returncode == 0 and len(hf) == 1
              and not [c for c in calls if c[0].startswith("HF_TOKEN_")],
              proc.stdout + proc.stderr)
        if hf:
            argv = hf[0][1]
            check("S-ARGV the download lands under FIDELITY_FS_ROOT",
                  "--local-dir" in argv
                  and argv[argv.index("--local-dir") + 1]
                  == str(sb.fs / "models" / "target"), argv)
            check("the pinned revision reaches hf",
                  "--revision" in argv and REV_A in argv, argv)
            included = [
                argv[index + 1] for index, value in enumerate(argv)
                if value == "--include"]
            check("target fetch names every sealed download path and no others",
                  included == [
                      row["path"]
                      for row in job_quant()["target"]["download_manifest"]],
                  included)
        seal_calls = [c for c in calls if c[0] == "PY"
                      and any("tr3_surface.py" in a for a in c[1])]
        check("a tr3 release has its published seal verified AND its scope "
              "written, right after the bytes land", len(seal_calls) == 2,
              [c[1][:3] for c in calls])
        check("S-MARK fetch_target writes its marker",
              sb.marker("fetch_target").is_file())
        census_path = sb.fs / "receipts" / "fetch-target-census.json"
        census = json.loads(census_path.read_text()) if census_path.is_file() else {}
        check("fetch_target writes a self-sealed exact job/target census",
              census.get("schema") == "fidelity.fetch-target-census.v1"
              and common.verify_seal(census)
              and census.get("job_id_full") == job_quant()["job_id_full"]
              and census.get("config_sha256")
              == hashlib.sha256(SELFTEST_CONFIG_BYTES).hexdigest()
              and census.get("shards") == SELFTEST_SHARDS,
              census)
        check("S-ARGV no argument names a path the environment did not supply",
              not sb.foreign_paths(calls), sb.foreign_paths(calls))
        check("S-ROOT fetch_target names no provider path", not provider_leak(out),
              out[-600:])

        # S-CLOSED
        sb2 = Sandbox(td / "ft2", job_quant())
        invalid = json.loads((sb2.fs / "job.json").read_text())
        del invalid["target"]["repo_id"]
        (sb2.fs / "job.json").write_text(json.dumps(invalid))
        proc2, calls2 = sb2.run("fetch_target", bash)
        check("S-CLOSED a job with no target.repo_id is refused before download",
              proc2.returncode != 0 and not calls2,
              proc2.stdout + proc2.stderr)
        check("S-MARK ...and no marker is left behind",
              not sb2.marker("fetch_target").is_file())

        duplicate_index_raw = (
            b'{"weight_map":{"a":"model-00001-of-00001.safetensors"},'
            b'"weight_map":{"b":"model-00001-of-00001.safetensors"}}')
        duplicate_index_job = job_quant()
        duplicate_index_job["target"] = dict(
            duplicate_index_job["target"],
            index_sha256=hashlib.sha256(duplicate_index_raw).hexdigest())
        duplicate_index = Sandbox(
            td / "ft-duplicate-index",
            jobcontract.finalize_job(duplicate_index_job))
        (duplicate_index.fs / "models" / "target" /
         "model.safetensors.index.json").write_bytes(duplicate_index_raw)
        proc2, calls2 = duplicate_index.run("fetch_target", bash)
        check("duplicate index keys refuse before the fetch marker",
              proc2.returncode != 0
              and not duplicate_index.marker("fetch_target").exists()
              and "duplicate key" in (proc2.stdout + proc2.stderr),
              proc2.stdout + proc2.stderr)

        wrong_size = Sandbox(td / "ft-wrong-shard-size", job_quant())
        (wrong_size.fs / "models" / "target" /
         SELFTEST_SHARDS[0]["path"]).write_bytes(b"x" * 16)
        proc2, calls2 = wrong_size.run("fetch_target", bash)
        check("shard size/model census drift refuses before the fetch marker",
              proc2.returncode != 0
              and not wrong_size.marker("fetch_target").exists()
              and "shard size differs" in (proc2.stdout + proc2.stderr),
              proc2.stdout + proc2.stderr)

        # ---------------------------------------------------------------
        print("\n== fetch_panel: include-scoped, data never parsed by the shell ==")
        sb = Sandbox(td / "fp", job_quant())
        secret = sb.fs / ".secrets" / "hf_token"
        secret.unlink()
        secret.mkdir()
        proc, calls = sb.run(
            "fetch_panel", bash, HF_TOKEN="ambient-secret",
            HUGGING_FACE_HUB_TOKEN="ambient-secret-two",
            HUGGINGFACE_HUB_TOKEN="ambient-secret-three",
            HF_TOKEN_PATH="/hostile/token",
            HF_HOME="/hostile/cache",
            HF_ENDPOINT="https://hostile.invalid")
        out = proc.stdout + proc.stderr
        hf = [c for c in calls if c[0] == "HF"]
        extra_index_raw = canonical({
            "weight_map": {
                "model.a": SELFTEST_SHARDS[0]["path"],
                "model.b": "extra.safetensors",
            }})
        extra_index_job = job_quant()
        extra_index_job["target"] = dict(
            extra_index_job["target"],
            index_sha256=hashlib.sha256(extra_index_raw).hexdigest())
        extra_index = Sandbox(
            td / "ft-extra-index",
            jobcontract.finalize_job(extra_index_job))
        (extra_index.fs / "models" / "target" /
         "model.safetensors.index.json").write_bytes(extra_index_raw)
        proc2, _ = extra_index.run("fetch_target", bash)
        check("missing/extra index shard names refuse before the fetch marker",
              proc2.returncode != 0
              and not extra_index.marker("fetch_target").exists()
              and "missing/extra indexed shards" in
              (proc2.stdout + proc2.stderr),
              proc2.stdout + proc2.stderr)
        check("fetch_panel exits 0 and calls hf once",
              proc.returncode == 0 and len(hf) == 1, out[-900:])
        public_env = [c for c in calls if c[0] == "HF_PUBLIC_ENV"]
        check("fetch_panel never loads .secrets/hf_token and emits one "
              "anonymous official-endpoint isolated-cache command",
              proc.returncode == 0
              and public_env == [
                  ("HF_PUBLIC_ENV", ["anonymous", "official", "isolated"])]
              and not [c for c in calls if c[0].startswith(
                  ("HF_TOKEN_", "HF_PUBLIC_ENV_WRONG"))],
              calls)
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

        # Ambient pipeline overrides are overwritten by the exact staged engine root.
        sb = Sandbox(td / "meas2", job_quant(), real_scripts=["invoke_engine.py"],
                     pipeline_root=str(td / "meas2" / "hostile-pipe"))
        _, calls = sb.run(
            "measure", bash, BF16="/hostile/bf16",
            TR3_BF16="/hostile/tr3", FIDELITY_ENGINE_PYTHON="/hostile/python")
        ec = [c for c in calls if c[0] == "PY"
              and any("stream_score.py" in a for a in c[1])]
        check("ambient root/interpreter overrides cannot alter paid invoker",
              ec and ec[0][1][ec[0][1].index("--pipeline-root") + 1]
              == str(sb.engine / "pipeline")
              and "/hostile" not in "\n".join(ec[0][1]),
              ec[0][1] if ec else calls)

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
        proc, calls = sb.run("score", bash, KLD_DEVICE="cpu")
        out = proc.stdout + proc.stderr
        sc = [c for c in calls if c[0] == "PY"
              and any("kld_report" in a for a in c[1])]
        check("score runs the lane's pinned scorer once",
              proc.returncode == 0 and len(sc) == 1, out[-1200:])
        wrapper = [c[1] for c in calls if c[0] == "PY"
                   and any("invoke_scorer.py" in a for a in c[1])]
        check("paid score argv leaves device/chunk solely to sealed job.json",
              len(wrapper) == 1
              and "--device" not in wrapper[0]
              and "--vocab-chunk" not in wrapper[0]
              and "--chunk-positions" not in wrapper[0],
              wrapper)
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
        print("\n== capture (--role root, GGUF candidate: the build dir gets the "
              "reference root's config + tokenizer files) ==")
        # The GGUF lane's paid attempt 3 (2026-09-05) built the whole streamed
        # model and then died in hf_capture's fail-closed generation probe,
        # which loads the tokenizer from --model itself: a GGUF ships no
        # tokenizer and no config.json. The capture stage therefore COPIES the
        # reference root's model-class files beside the build (regular files,
        # config byte-checked against target.config_sha256) and links the
        # tokenizer files into the tokenizer root. This rung drives the REAL
        # stage over a GGUF-shaped target and asserts exactly that arrangement.
        gguf_build = "UD-Q4_K_XL"
        gguf_shards = [{"path": "%s/mini-00001-of-00001.gguf" % gguf_build, "bytes": 17}]
        gguf_manifest = sorted(gguf_shards + [{"path": "LICENSE", "bytes": 3}],
                               key=lambda row: row["path"])
        gguf_config = b'{"num_hidden_layers": 78, "indexer_types": ["full"]}\n'
        gguf_scope = json.dumps({"policy": "mixed", "head_policy": "quantized",
                                 "assignments": []}).encode("utf-8")
        gguf_binding = {"schema": "fidelity.resolved-panel.v1", "panel_id": "panel--x.y.z",
                        "panel": {"id": "panel--x.y.z"}}
        gguf_binding_bytes = canonical(gguf_binding)
        gguf_job = job_root()
        gguf_job["panel"] = dict(gguf_job["panel"], resolved_binding=gguf_binding,
                                 binding_file_sha256=hashlib.sha256(gguf_binding_bytes).hexdigest())
        gguf_job["target"] = {
            "repo_id": "unsloth/GLM-5.3-GGUF", "revision": REV_A, "path": gguf_build,
            "surface": "gguf", "codec": "gguf-k-quant", "bits": 4.0,
            "config_sha256": hashlib.sha256(gguf_config).hexdigest(),
            "index_sha256": "9" * 64, "index_source": "sha256 of the canonical GGUF tensor table",
            "shard_manifest_sha256": hashlib.sha256(canonical(gguf_shards)).hexdigest(),
            "model_bytes": 17, "shards": gguf_shards,
            "download_manifest": gguf_manifest,
            "download_bytes_total": sum(r["bytes"] for r in gguf_manifest),
            "download_manifest_sha256": hashlib.sha256(canonical(gguf_manifest)).hexdigest(),
        }
        gguf_job["profile"] = dict(gguf_job["profile"], surface="gguf")
        gguf_job["capture"] = dict(gguf_job["capture"], own_heads=True, candidate={
            "scope": {"path": "candidate/scope.json",
                      "sha256": hashlib.sha256(gguf_scope).hexdigest(),
                      "scope_digest": "lm_head=quantized:gguf-k-quant@8.5"},
            "codec": "gguf-k-quant", "declared_bits": 4.0,
            "weights_decode": {"method": "gguf-dequant-to-bf16",
                               "quantization_config": {"container": "gguf", "build": gguf_build}},
            "reference": {"repository": "malaiwah/root-v1", "revision": REV_B,
                          "dataset_sha256": "1" * 64, "capture_content_digest": "2" * 64,
                          "dataset_id": "fidelity--root", "panel_id": "panel--x.y.z",
                          "suite_token_hash_sha256": "3" * 64}})
        gguf_job.pop("job_id", None)
        gguf_job.pop("job_id_full", None)
        gguf_job = jobcontract.finalize_job(gguf_job)
        gsb = Sandbox(td / "cap-gguf", gguf_job, finalize_job_doc=False)
        (gsb.fs / "panel-binding.json").write_bytes(gguf_binding_bytes)
        (gsb.fs / "panel-src").mkdir()
        for stale in ("config.json", "model.safetensors.index.json", SELFTEST_SHARDS[0]["path"]):
            (gsb.fs / "models" / "target" / stale).unlink()
        (gsb.fs / "models" / "target" / gguf_build).mkdir()
        (gsb.fs / "models" / "target" / gguf_build / "mini-00001-of-00001.gguf").write_bytes(b"GGUF" + b"\0" * 13)
        (gsb.fs / "models" / "target" / "LICENSE").write_bytes(b"MIT")
        (gsb.fs / "candidate").mkdir()
        (gsb.fs / "candidate" / "scope.json").write_bytes(gguf_scope)
        ref_model = gsb.fs / "reference-model"
        ref_model.mkdir()
        # a REAL (tiny) tokenizers-format tokenizer, so the probe's loader can be
        # exercised on the arranged build dir under a torch interpreter
        tiny_tokenizer = json.dumps({
            "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
            "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
            "decoder": None,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "The": 1, "capital": 2,
                                                     "of": 3, "France": 4, "is": 5, "Paris": 6},
                      "unk_token": "[UNK]"}}).encode("utf-8")
        ref_files = {"config.json": gguf_config, "tokenizer.json": tiny_tokenizer,
                     "tokenizer_config.json": b'{"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}\n',
                     "generation_config.json": b'{"do_sample": false}\n',
                     "LICENSE": b"MIT", "chat_template.jinja": b"{{ messages }}"}
        for name, body in ref_files.items():
            (ref_model / name).write_bytes(body)
        gsb.write_bound_marker("fetch_reference")
        proc, calls = gsb.run("capture", bash)
        out = proc.stdout + proc.stderr
        target_dir = gsb.fs / "models" / "target"
        copied = {name: (target_dir / name).is_file() and not (target_dir / name).is_symlink()
                  and (target_dir / name).read_bytes() == ref_files[name]
                  for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                               "generation_config.json")}
        tok_root = gsb.fs / "inputs" / "tokenizer-root"
        check("gguf candidate: the capture stage copies the reference root's config.json, "
              "tokenizer.json, tokenizer_config.json and generation_config.json beside the "
              "build as REGULAR files, byte-identical, and the stage runs the writer",
              proc.returncode == 0 and all(copied.values())
              and any(c[0] == "PY" and any("fidelity_dataset.py" in a for a in c[1]) for c in calls),
              "%s\n%s" % (copied, out[-1200:]))
        check("gguf candidate: the tokenizer root links tokenizer.json and tokenizer_config.json "
              "from the reference root (the panel's byte gate) and the build itself is untouched",
              (tok_root / "tokenizer.json").is_symlink() and (tok_root / "tokenizer_config.json").is_symlink()
              and (tok_root / "tokenizer.json").resolve() == (ref_model / "tokenizer.json").resolve()
              and (target_dir / gguf_build / "mini-00001-of-00001.gguf").read_bytes() == b"GGUF" + b"\0" * 13,
              out[-600:])
        gguf_cap = [c for c in calls if c[0] == "PY" and any("fidelity_dataset.py" in a for a in c[1])]
        if gguf_cap:
            argv = gguf_cap[0][1]
            check("gguf candidate: the capture is invoked as role quant with the bound scope, "
                  "codec gguf-k-quant, 4.0 bits and --model = the build's parent tree",
                  argv[argv.index("--codec") + 1] == "gguf-k-quant"
                  and argv[argv.index("--declared-bits") + 1] == "4.0"
                  and argv[argv.index("--scope-file") + 1] == str(gsb.fs / "candidate" / "scope.json")
                  and argv[argv.index("--model") + 1] == str(target_dir), argv)
        try:
            sys.path.insert(0, str(ROOT / "engines" / "tools"))
            import generation_probe
            import transformers  # noqa: F401
            tok, why = generation_probe.load_tokenizer(str(target_dir))
            check("gguf candidate: hf_capture's generation-probe tokenizer load SUCCEEDS on the "
                  "arranged build dir (the exact call paid attempt 3 died in)",
                  tok is not None and tok.encode("The capital of France is", add_special_tokens=True),
                  str(why))
        except ImportError as exc:
            skip("gguf candidate: generation-probe tokenizer load on the arranged build dir",
                 "transformers not importable here (%s); runs under the torch interpreter" % exc)
        # ...and a mismatching reference config refuses before any capture
        bad = Sandbox(td / "cap-gguf-badcfg", gguf_job, finalize_job_doc=False)
        (bad.fs / "panel-binding.json").write_bytes(gguf_binding_bytes)
        (bad.fs / "panel-src").mkdir()
        (bad.fs / "candidate").mkdir()
        (bad.fs / "candidate" / "scope.json").write_bytes(gguf_scope)
        (bad.fs / "reference-model").mkdir()
        for name, body in ref_files.items():
            (bad.fs / "reference-model" / name).write_bytes(body if name != "config.json" else b'{"other": 1}\n')
        bad.write_bound_marker("fetch_reference")
        proc, calls = bad.run("capture", bash)
        check("gguf candidate: a reference config.json that does not carry target.config_sha256 "
              "is REFUSED before the writer runs",
              proc.returncode != 0 and "target.config_sha256" in (proc.stdout + proc.stderr)
              and not [c for c in calls if any("fidelity_dataset.py" in a for a in c[1])],
              (proc.stdout + proc.stderr)[-600:])

        # ---------------------------------------------------------------
        print("\n== capture / verify (--role root) ==")
        sb = Sandbox(td / "cap", job_root())
        (sb.fs / "panel-src").mkdir()
        proc, calls = sb.run("capture", bash)
        capture_returncode = proc.returncode
        capture_calls = calls
        out = proc.stdout + proc.stderr
        no_census = Sandbox(td / "cap-no-target-census", job_root())
        (no_census.fs / "panel-src").mkdir()
        proc, calls = no_census.run(
            "capture", bash, provision_target=False)
        check("capture refuses before compute without the bound target census",
              proc.returncode != 0 and not calls
              and "fetch_target" in (proc.stdout + proc.stderr),
              proc.stdout + proc.stderr)

        cap = [c for c in capture_calls if c[0] == "PY"
               and any("fidelity_dataset.py" in a for a in c[1])]
        check("capture runs the dataset writer once",
              capture_returncode == 0 and len(cap) == 1, out[-1000:])
        if cap:
            argv = cap[0][1]
            check("S-ARGV the panel is the uploaded one, resolved under the fs root",
                  str(sb.fs / "panel-src") in argv, argv)
            check("S-ARGV the model is the LOCAL tree fetch_target wrote",
                  str(sb.fs / "models" / "target") in argv, argv)
            check("dataset destination and executed weights repositories stay distinct",
                  "--repository" in argv
                  and argv[argv.index("--repository") + 1] == "malaiwah/mm3-root-v1"
                  and argv[argv.index("--weights-repository") + 1]
                  == "MiniMaxAI/MiniMax-M3",
                  argv)
            check("first capture has a stable cold-process label",
                  argv[argv.index("--run-name") + 1] == "root-cold-1"
                  and argv[argv.index("--cold-run") + 1] == "root-cold-1", argv)
            check("capture binds the exact uploaded panel contract and tokenizer root",
                  argv[argv.index("--panel-binding") + 1]
                  == str(sb.fs / "panel-binding.json")
                  and argv[argv.index("--panel-binding-sha256") + 1]
                  == PANEL_BINDING_SHA
                  and argv[argv.index("--panel-tokenizer-root") + 1]
                  == str(sb.fs / "models" / "target"), argv)
        check("S-MARK capture writes its marker", sb.marker("capture").is_file())
        check("S-ARGV no foreign path", not sb.foreign_paths(capture_calls),
              sb.foreign_paths(capture_calls))
        unpublished = job_root()
        unpublished_capture = dict(
            unpublished["capture"], publish_root_to=None)
        unpublished_capture["root_protocol"] = dict(
            unpublished_capture["root_protocol"],
            canonical_publication_required=False,
            publication_mode="qualified-unpublished")
        unpublished["capture"] = unpublished_capture
        no_publish = Sandbox(td / "cap-qualified-unpublished", unpublished)
        (no_publish.fs / "panel-src").mkdir()
        proc, calls = no_publish.run("capture", bash)
        capture_calls = [c[1] for c in calls if c[0] == "PY"
                         and any("fidelity_dataset.py" in a for a in c[1])]
        argv = capture_calls[0] if capture_calls else []
        check("capture allows a qualified-unpublished intended dataset identity",
              proc.returncode == 0 and argv
              and argv[argv.index("--repository") + 1]
              == "malaiwah/mm3-root-v1",
              proc.stdout + proc.stderr)
        missing_binding = job_root()
        missing_binding["panel"] = dict(
            missing_binding["panel"], binding_path="absent-panel-binding.json")
        bad = Sandbox(td / "cap-missing-binding", missing_binding)
        (bad.fs / "panel-src").mkdir()
        proc, calls = bad.run("capture", bash)
        check("missing panel binding refuses before capture",
              proc.returncode != 0 and not calls and "panel binding file is absent" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        mismatched_binding = job_root()
        mismatched_binding["panel"] = dict(
            mismatched_binding["panel"], binding_file_sha256="0" * 64)
        bad = Sandbox(td / "cap-binding-mismatch", mismatched_binding)
        (bad.fs / "panel-src").mkdir()
        proc, calls = bad.run("capture", bash)
        check("panel binding raw-hash mismatch refuses before capture",
              proc.returncode != 0 and not calls and "binding_file_sha256 mismatch" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        legacy = job_root()
        legacy["capture"] = dict(legacy["capture"], allow_unexpected_tensors=False)
        bad = Sandbox(td / "cap-legacy-unexpected", legacy)
        (bad.fs / "panel-src").mkdir()
        proc, calls = bad.run("capture", bash)
        check("even a false legacy broad unexpected-tensor boolean refuses",
              proc.returncode != 0 and not calls and "obsolete" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        partial = job_root()
        partial["capture"] = dict(
            partial["capture"],
            unexpected_tensor_allowlist={"path": "allowlist.json"})
        partial = self_consistent_job(partial)
        bad = Sandbox(
            td / "cap-partial-allowlist", partial,
            finalize_job_doc=False)
        (bad.fs / "panel-src").mkdir()
        (bad.fs / "allowlist.json").write_text('["model.unused"]')
        proc, calls = bad.run("capture", bash)
        check("partial exact unexpected-tensor allowlist refuses in job validation",
              proc.returncode != 0 and not calls
              and "root capture contract is incomplete" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        allowlist_raw = b'["model.unused"]'
        allowlist_names_sha = hashlib.sha256(allowlist_raw).hexdigest()
        mismatched = job_root()
        mismatched["capture"] = dict(
            mismatched["capture"],
            unexpected_tensor_allowlist={
                "path": "allowlist.json", "artifact_sha256": "0" * 64,
                "canonical_sorted_names_sha256": allowlist_names_sha})
        bad = Sandbox(td / "cap-allowlist-mismatch", mismatched)
        (bad.fs / "panel-src").mkdir()
        (bad.fs / "allowlist.json").write_bytes(allowlist_raw)
        proc, calls = bad.run("capture", bash)
        check("unexpected-tensor allowlist raw-hash mismatch refuses",
              proc.returncode != 0 and not calls and "raw SHA-256 mismatch" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        exact = job_root()
        allowlist_sha = hashlib.sha256(allowlist_raw).hexdigest()
        exact["capture"] = dict(
            exact["capture"],
            unexpected_tensor_allowlist={
                "path": "allowlist.json", "artifact_sha256": allowlist_sha,
                "canonical_sorted_names_sha256": allowlist_names_sha})
        allowed = Sandbox(td / "cap-exact-allowlist", exact)
        (allowed.fs / "panel-src").mkdir()
        (allowed.fs / "allowlist.json").write_bytes(allowlist_raw)
        proc, calls = allowed.run("capture", bash)
        exact_calls = [c[1] for c in calls if c[0] == "PY"
                       and any("fidelity_dataset.py" in a for a in c[1])]
        argv = exact_calls[0] if exact_calls else []
        check("complete, correctly hashed allowlist passes all exact flags",
              proc.returncode == 0 and argv
              and argv[argv.index("--unexpected-tensors-allowlist") + 1]
              == str(allowed.fs / "allowlist.json")
              and argv[argv.index("--unexpected-tensors-allowlist-sha256") + 1]
              == allowlist_sha
              and argv[argv.index("--unexpected-tensors-name-sha256") + 1]
              == allowlist_names_sha,
              argv)

        for case_name, raw, refusal in (
                ("empty", b"[]", "non-empty"),
                ("duplicate", b'["model.unused","model.unused"]', "duplicate")):
            names = json.loads(raw.decode("utf-8"))
            canonical_names = json.dumps(
                sorted(names), separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")
            guarded = job_root()
            guarded["capture"] = dict(
                guarded["capture"],
                unexpected_tensor_allowlist={
                    "path": "allowlist.json",
                    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                    "canonical_sorted_names_sha256":
                        hashlib.sha256(canonical_names).hexdigest()})
            bad = Sandbox(td / ("cap-allowlist-" + case_name), guarded)
            (bad.fs / "panel-src").mkdir()
            (bad.fs / "allowlist.json").write_bytes(raw)
            proc, calls = bad.run("capture", bash)
            check("%s unexpected-tensor allowlist refuses" % case_name,
                  proc.returncode != 0 and not calls and refusal in
                  (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        escaped_panel = job_root()
        escaped_panel["capture"] = dict(
            escaped_panel["capture"], panel_dir="../outside-panel")
        bad = Sandbox(td / "cap-panel-traversal", escaped_panel)
        proc, calls = bad.run("capture", bash)
        check("capture.panel_dir traversal refuses before capture",
              proc.returncode != 0 and not calls and "canonical relative path" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        symlink_panel = job_root()
        symlink_panel["capture"] = dict(
            symlink_panel["capture"], panel_dir="panel-link")
        bad = Sandbox(td / "cap-panel-symlink", symlink_panel)
        outside_panel = bad.tmp / "outside-panel"
        outside_panel.mkdir()
        (bad.fs / "panel-link").symlink_to(outside_panel, target_is_directory=True)
        proc, calls = bad.run("capture", bash)
        check("capture.panel_dir symlink escape refuses before capture",
              proc.returncode != 0 and not calls and "may not traverse a symlink" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        for case_name, bad_path in (
                ("dot", "./panel-src"),
                ("double-slash", "panel//src"),
                ("backslash", "panel\\src")):
            noncanonical = job_root()
            noncanonical["capture"] = dict(
                noncanonical["capture"], panel_dir=bad_path)
            bad = Sandbox(td / ("cap-panel-" + case_name), noncanonical)
            proc, calls = bad.run("capture", bash)
            check("capture.panel_dir %s normalization refuses" % case_name,
                  proc.returncode != 0 and not calls and "canonical relative" in
                  (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        for case_name, bad_path in (
                ("dot", "./panel-binding.json"),
                ("double-slash", "panel//binding.json"),
                ("backslash", "panel\\binding.json")):
            noncanonical = job_root()
            noncanonical["panel"] = dict(
                noncanonical["panel"], binding_path=bad_path)
            bad = Sandbox(td / ("cap-binding-" + case_name), noncanonical)
            (bad.fs / "panel-src").mkdir()
            proc, calls = bad.run("capture", bash)
            check("panel.binding_path %s normalization refuses" % case_name,
                  proc.returncode != 0 and not calls and "canonical relative" in
                  (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        duplicate_binding_raw = (
            b'{"schema":"fidelity.resolved-panel.v1",'
            b'"schema":"fidelity.resolved-panel.v1","panel_id":"panel--x.y.z"}')
        duplicate_binding = job_root()
        duplicate_binding["panel"] = dict(
            duplicate_binding["panel"],
            binding_file_sha256=hashlib.sha256(
                duplicate_binding_raw).hexdigest())
        bad = Sandbox(td / "cap-binding-duplicate-json", duplicate_binding)
        (bad.fs / "panel-src").mkdir()
        (bad.fs / "panel-binding.json").write_bytes(duplicate_binding_raw)
        proc, calls = bad.run("capture", bash)
        check("duplicate keys in the bound panel JSON refuse before capture",
              proc.returncode != 0 and not calls and "duplicate key" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

        for case_name, bad_path in (
                ("dot", "./allowlist.json"),
                ("double-slash", "tensor//allowlist.json"),
                ("backslash", "tensor\\allowlist.json")):
            noncanonical = job_root()
            allowlist = dict(
                noncanonical["capture"]["unexpected_tensor_allowlist"],
                path=bad_path)
            noncanonical["capture"] = dict(
                noncanonical["capture"],
                unexpected_tensor_allowlist=allowlist)
            bad = Sandbox(td / ("cap-allowlist-path-" + case_name), noncanonical)
            (bad.fs / "panel-src").mkdir()
            proc, calls = bad.run("capture", bash)
            check("allowlist %s path normalization refuses" % case_name,
                  proc.returncode != 0 and not calls and "canonical relative" in
                  (proc.stdout + proc.stderr), proc.stdout + proc.stderr)

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
        (sb.fs / "panel-src").mkdir()
        sb.run("capture", bash)
        proc, calls = sb.run("verify", bash)
        ver = [c for c in calls if c[0] == "PY"
               and any("fidelity_dataset.py" in a for a in c[1])]
        check("verify recomputes the first seal, writes its receipt, and describes",
              proc.returncode == 0 and len(ver) == 2
              and "verify" in ver[0][1] and "describe" in ver[1][1]
              and str(sb.fs / "receipts" / "dataset-verify.json") in ver[0][1],
              calls)
        check("S-MARK verify writes its marker", sb.marker("verify").is_file())

        proc, calls = sb.run("capture_repeat", bash)
        repeat = [c for c in calls if c[0] == "PY"
                  and any("fidelity_dataset.py" in a for a in c[1])]
        repeat_argv = repeat[0][1] if repeat else []
        check("second root capture invokes a fresh process into a distinct tree",
              proc.returncode == 0 and len(repeat) == 1
              and str(sb.fs / "dataset-repeat") in repeat_argv
              and repeat_argv[repeat_argv.index("--run-name") + 1] == "root-cold-2"
              and repeat_argv[repeat_argv.index("--cold-run") + 1] == "root-cold-2",
              repeat_argv)
        sb.write_bound_marker("verify_repeat")
        proc, calls = sb.run("compare_root", bash)
        compare_calls = [c for c in calls if c[0] == "PY"
                         and any("fidelity_dataset.py" in a for a in c[1])]
        compare_argv = compare_calls[0][1] if compare_calls else []
        check("root comparison is forced and binds the job's explicit replay profile",
              proc.returncode == 0 and len(compare_calls) == 1
              and "--self-compare" in compare_argv
              and "--force-compute" in compare_argv
              and compare_argv[compare_argv.index("--replay-device") + 1] == "numpy"
              and compare_argv[compare_argv.index("--replay-dtype") + 1] == "float32"
              and compare_argv[compare_argv.index("--vocab-chunk") + 1] == "8192",
              compare_argv)

        no_backend = job_root()
        del no_backend["capture"]["replay_device"]
        no_backend = self_consistent_job(no_backend)
        bad = Sandbox(
            td / "compare-no-backend", no_backend,
            finalize_job_doc=False)
        (bad.fs / "dataset").mkdir()
        for stage in ("verify", "verify_repeat"):
            bad.write_bound_marker(stage)
        proc, calls = bad.run("compare_root", bash)
        check("root compare refuses a missing explicit replay backend",
              proc.returncode != 0 and not calls
              and "root capture contract is incomplete" in
              (proc.stdout + proc.stderr), proc.stdout + proc.stderr)


        # ---------------------------------------------------------------
        print("\n== unsupported preview/race paid roots ==")
        for stage in ("race_bootstrap", "race_capture"):
            bad = Sandbox(td / ("unsupported-" + stage), job_root())
            proc, calls = bad.run(stage, bash)
            check("%s explicitly refuses before any command" % stage,
                  proc.returncode == 3 and not calls
                  and "unsupported" in (proc.stdout + proc.stderr),
                  proc.stdout + proc.stderr)
        preview_job = job_root()
        preview_job["capture"] = dict(preview_job["capture"], preview_of="final-root")
        bad = Sandbox(td / "unsupported-preview", preview_job)
        (bad.fs / "panel-src").mkdir()
        proc, calls = bad.run("capture", bash)
        check("preview capture refuses before the first paid capture process",
              proc.returncode == 3 and not calls
              and "preview/race" in (proc.stdout + proc.stderr),
              proc.stdout + proc.stderr)


        # ---------------------------------------------------------------
        print("\n== cross-cutting ==")
        sb = Sandbox(td / "xx", job_quant())
        proc, _ = sb.run("no_such_stage", bash)
        check("S-CLOSED an unknown stage is refused and the usage names every "
              "stage this file implements", proc.returncode == 2
              and all(s in proc.stderr for s in
                      ("setup", "fetch_target", "fetch_panel", "measure",
                       "score", "seal", "capture", "verify", "capture_repeat",
                       "verify_repeat", "compare_root", "qualify_root",
                       "publish_root", "race_bootstrap", "race_capture")),
              proc.stderr)

        # Every stage in that usage line must actually be reachable, or the
        # driver advertises a stage the case statement does not implement.
        sb = Sandbox(td / "xy", job_quant())
        unknown = []
        for stage in ("setup", "fetch_target", "fetch_panel", "materialize",
                      "measure", "score", "seal", "capture", "verify",
                      "capture_repeat", "verify_repeat", "compare_root",
                      "qualify_root", "publish_root", "race_bootstrap", "race_capture"):
            p, _ = Sandbox(td / ("xy-" + stage), job_quant()).run(stage, bash)
            if "unknown stage" in p.stderr:
                unknown.append(stage)
        check("every stage the controller can ask for is implemented",
              not unknown, unknown)

        # A valid marker for the exact uploaded job attempt is the sole resume
        # shape.  Legacy/unbound/torn markers are never adopted.
        sb = Sandbox(td / "resume", job_quant())
        sb.write_bound_marker("fetch_panel")
        proc, calls = sb.run("fetch_panel", bash)
        check("an exact job+attempt-bound marker skips without re-running",
              proc.returncode == 0 and not calls
              and "already done" in proc.stdout, proc.stdout)

        print("\n== markers bind the full job file and execution attempt ==")
        job_a = job_quant(execution_attempt=execution_attempt("a" * 24))
        full_a = job_a["job_id_full"]
        sb = Sandbox(td / "bind1", job_a)
        proc, calls = sb.run("fetch_panel", bash)
        bound = sb.marker("fetch_panel").read_text()
        check("M1 a completed stage atomically writes all four bound fields",
              proc.returncode == 0
              and ("job_id_full=%s" % full_a) in bound
              and "job_sha256=" in bound
              and "stage=fetch_panel" in bound
              and "completed_at=" in bound, bound)
        proc, calls = sb.run("fetch_panel", bash)
        check("M1b the identical job attempt skips on its own marker",
              proc.returncode == 0 and not calls
              and "already done" in proc.stdout, proc.stdout)

        job_b = job_quant("gguf", execution_attempt=execution_attempt("b" * 24))
        (sb.fs / "job.json").write_text(json.dumps(job_b), encoding="utf-8")
        proc, calls = sb.run("fetch_panel", bash)
        check("M2 a different scientific job refuses the stale marker",
              proc.returncode == 7 and not calls
              and "invalid/stale marker" in proc.stderr,
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        # execution_attempt is intentionally excluded from job_id_full.  The
        # raw job-file binding must still prevent recovery/adoption.
        attempt_b = jobcontract.finalize_job(
            dict(job_a, execution_attempt=execution_attempt("b" * 24)))
        check("M3 test setup preserves scientific id across attempts",
              attempt_b["job_id_full"] == full_a)
        (sb.fs / "job.json").write_text(
            json.dumps(attempt_b), encoding="utf-8")
        proc, calls = sb.run("fetch_panel", bash)
        check("M3 same science but a different attempt refuses old outputs",
              proc.returncode == 7 and not calls
              and "job_sha256 mismatch" in proc.stderr,
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        torn = Sandbox(td / "bind-torn", job_quant())
        torn.marker("fetch_panel").parent.mkdir(parents=True, exist_ok=True)
        torn.marker("fetch_panel").write_text(
            "job_id_full=%s\n" % job_quant()["job_id_full"])
        proc, calls = torn.run(
            "fetch_panel", bash, FIDELITY_ADOPT_MARKERS="1")
        check("M4 a torn first-line-only marker refuses without an adoption escape",
              proc.returncode == 7 and not calls
              and "torn/legacy" in proc.stderr,
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        tampered = Sandbox(td / "job-tamper", job_quant())
        bad_job = json.loads((tampered.fs / "job.json").read_text())
        bad_job["target"]["revision"] = REV_B
        (tampered.fs / "job.json").write_text(json.dumps(bad_job))
        proc, calls = tampered.run("fetch_panel", bash)
        check("job self-identity is verified before any stage action",
              proc.returncode != 0 and not calls
              and "self-identity REFUSED" in proc.stderr
              and not tampered.marker("fetch_panel").exists(),
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        duplicate = Sandbox(td / "job-duplicate-key", job_quant())
        original = (duplicate.fs / "job.json").read_text(encoding="utf-8")
        (duplicate.fs / "job.json").write_text(
            '{"role":"root",' + original[1:], encoding="utf-8")
        for rel in ("receipts", "logs", "models", "panel", ".secrets"):
            shutil.rmtree(duplicate.fs / rel, ignore_errors=True)
        proc, calls = duplicate.run("fetch_panel", bash)
        check("duplicate job keys refuse before mkdir or any stage action",
              proc.returncode != 0 and not calls
              and "duplicate key" in proc.stderr
              and not (duplicate.fs / "receipts").exists()
              and not (duplicate.fs / "logs").exists(),
              "rc=%s\n%s" % (proc.returncode, proc.stderr))


        incomplete_execution = job_quant()
        incomplete_execution["execution_attempt"] = {
            "kind": "runpod-ssh",
            "attempt_id": None, "cost_quote": None, "engine_root": None,
            "execution_contract_sha256": None, "lease_path": None,
            "planned_at": None, "pre_create_safety": None,
            "remote_root": None, "provider_terminate_after": None,
            "storage_layout": None, "workload_deadline_utc": None,
        }
        incomplete_execution["execution_attempt"]["execution_contract_sha256"] = (
            jobcontract.execution_contract_sha256(incomplete_execution))
        incomplete_execution = self_consistent_job(incomplete_execution)
        incomplete = Sandbox(
            td / "job-incomplete-execution", incomplete_execution,
            finalize_job_doc=False)
        for rel in ("receipts", "logs", "models"):
            shutil.rmtree(incomplete.fs / rel)
        proc, calls = incomplete.run("fetch_panel", bash)
        check("incomplete execution contract refuses before mkdir or stage action",
              proc.returncode != 0 and not calls
              and "runpod-ssh execution_attempt fields differ" in proc.stderr
              and not (incomplete.fs / "receipts").exists()
              and not (incomplete.fs / "logs").exists(),
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        unknown_profile = job_quant()
        unknown_profile["profile"]["ambient_override"] = "forbidden"
        unknown_profile = self_consistent_job(unknown_profile)
        malformed = Sandbox(
            td / "job-unknown-profile", unknown_profile,
            finalize_job_doc=False)
        for rel in ("receipts", "logs", "models"):
            shutil.rmtree(malformed.fs / rel)
        proc, calls = malformed.run("fetch_panel", bash)
        check("unknown profile field refuses before mkdir or stage action",
              proc.returncode != 0 and not calls
              and "profile contract is incomplete" in proc.stderr
              and not (malformed.fs / "receipts").exists()
              and not (malformed.fs / "logs").exists(),
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        bad_token_mode = Sandbox(td / "token-mode", job_quant())
        (bad_token_mode.fs / ".secrets" / "hf_token").chmod(0o644)
        proc, calls = bad_token_mode.run("fetch_target", bash)
        check("fetch refuses non-0600 token file before any Hub command",
              proc.returncode != 0 and not calls
              and "mode must be exactly 0600" in proc.stderr,
              proc.stdout + proc.stderr)

        token_symlink = Sandbox(td / "token-symlink", job_quant())
        token_path = token_symlink.fs / ".secrets" / "hf_token"
        outside_token = token_symlink.tmp / "outside-token"
        outside_token.write_text("secret")
        outside_token.chmod(0o600)
        token_path.unlink()
        token_path.symlink_to(outside_token)
        proc, calls = token_symlink.run("fetch_target", bash)
        check("fetch refuses symlinked token file before any Hub command",
              proc.returncode != 0 and not calls
              and "HF token file REFUSED" in proc.stderr,
              proc.stdout + proc.stderr)

        replay_mutations = (
            ("missing replay", lambda capture: capture.pop("replay")),
            ("mismatched replay chunk",
             lambda capture: capture["replay"].update(vocab_chunk=4096)),
            ("unexpected replay field",
             lambda capture: capture["replay"].update(ambient=True)),
        )
        for index, (label, mutate) in enumerate(replay_mutations):
            replay_job = job_root()
            mutate(replay_job["capture"])
            replay_job = self_consistent_job(replay_job)
            replay_box = Sandbox(
                td / ("replay-contract-%d" % index), replay_job,
                finalize_job_doc=False)
            for rel in ("receipts", "logs", "models"):
                shutil.rmtree(replay_box.fs / rel)
            proc, calls = replay_box.run("fetch_panel", bash)
            check("%s refuses before mkdir or stage action" % label,
                  proc.returncode != 0 and not calls
                  and "root capture contract is incomplete" in proc.stderr
                  and not (replay_box.fs / "receipts").exists(),
                  proc.stdout + proc.stderr)

        provenance_mutations = (
            ("missing producer profile",
             lambda deps: deps.pop("profile")),
            ("mismatched producer profile",
             lambda deps: deps.update(profile="some-other-profile")),
            ("mismatched producer lane",
             lambda deps: deps.update(lane="other-lane")),
            ("mismatched producer provider",
             lambda deps: deps.update(provider="other-provider")),
        )
        for index, (label, mutate) in enumerate(provenance_mutations):
            provenance_job = job_root()
            mutate(provenance_job["produced_by"]["dependencies"])
            provenance_job = self_consistent_job(provenance_job)
            provenance_box = Sandbox(
                td / ("producer-contract-%d" % index), provenance_job,
                finalize_job_doc=False)
            for rel in ("receipts", "logs", "models"):
                shutil.rmtree(provenance_box.fs / rel)
            proc, calls = provenance_box.run("fetch_panel", bash)
            check("%s refuses before mkdir or stage action" % label,
                  proc.returncode != 0 and not calls
                  and "producing-code" in proc.stderr
                  and not (provenance_box.fs / "receipts").exists(),
                  proc.stdout + proc.stderr)
        # ---------------------------------------------------------------
        # P1-14: the atomic per-stage lock.  An unknown liveness probe must
        # never authorize a second writer; the lock is the on-box guarantee.
        print("\n== the per-stage lock refuses a second writer (P1-14) ==")
        sb = Sandbox(td / "lock1", job_a)
        lock = sb.fs / "receipts" / "locks" / "fetch_panel.lock"
        lock.mkdir(parents=True)
        (lock / "owner").write_text(
            "job_id_full=%s\npid=%d\nhost=x\nstarted=t\n"
            % (full_a, os.getpid()))    # a LIVE pid: this process
        proc, calls = sb.run("fetch_panel", bash)
        check("L1 a lock held by a LIVE process refuses (exit 8), nothing runs",
              proc.returncode == 8 and not calls
              and "second writer" in proc.stderr,
              "rc=%s\n%s" % (proc.returncode, proc.stderr))

        # L2: a lock whose owner is dead is stale -> taken over, stage runs.
        (lock / "owner").write_text(
            "job_id_full=%s\npid=99999999\nhost=x\nstarted=t\n" % full_a)
        proc, calls = sb.run("fetch_panel", bash)
        check("L2 a stale lock (dead owner) is taken over and the stage runs",
              proc.returncode == 0 and len(calls) >= 1
              and "taking over" in proc.stdout,
              "rc=%s\n%s" % (proc.returncode, proc.stdout + proc.stderr))
        check("L2b the lock is released when the stage exits",
              not lock.exists())

        # ---------------------------------------------------------------
        print("\n== publish_root: controller-local only, never on the pod ==")
        sb = Sandbox(td / "pub-qualified", job_root())
        (sb.fs / "dataset").mkdir()
        for stage in ("verify", "verify_repeat", "compare_root", "qualify_root"):
            sb.write_bound_marker(stage)
        (sb.fs / "receipts" / "root-qualification.json").write_text("{}")
        secret = sb.fs / ".secrets" / "hf_token"
        secret.unlink()
        secret.mkdir()
        proc, calls = sb.run(
            "publish_root", bash, HF_TOKEN="ambient-secret",
            HUGGING_FACE_HUB_TOKEN="ambient-secret-two",
            HF_TOKEN_PATH="/hostile/token")
        refusal = proc.stdout + proc.stderr
        check("even a qualified root refuses on-pod publication before any "
              "command or token read",
              proc.returncode == 3 and not calls
              and not sb.marker("publish_root").exists()
              and "controller-local only" in refusal
              and "verified retrieval" in refusal
              and "provider-confirmed pod absence" in refusal
              and "billing reconciliation" in refusal,
              refusal)

        preview = job_root()
        preview["capture"] = dict(preview["capture"], preview_of="mm3-root-v1")
        sb = Sandbox(td / "pub-preview", preview)
        proc, calls = sb.run("publish_root", bash)
        check("preview publication reaches the same unconditional controller "
              "refusal without a command",
              proc.returncode == 3 and not calls
              and "controller-local only" in (proc.stdout + proc.stderr),
              proc.stdout + proc.stderr)

        sb = Sandbox(td / "pub-quant", job_quant())
        proc, calls = sb.run("publish_root", bash)
        check("quant publication reaches the same unconditional controller "
              "refusal before role/job parsing",
              proc.returncode == 3 and not calls
              and "controller-local only" in (proc.stdout + proc.stderr),
              proc.stdout + proc.stderr)

    # The capture stage accepts exit 2 -- "sealed, with warnings" (M1 learning
    # 20) -- exactly as the comparator does, and refuses anything else. A
    # sealed 78-layer trellis capture (wrldsuksgo2mars, 438 s, 51,175 scored
    # rows, sanity probe " Paris" PASS) was destroyed by treating 2 as failure.
    driver = Path("bin/stage_measure.sh").read_text(encoding="utf-8")
    block = driver[driver.index("capturing fresh process"):]
    block = block[:block.index("write_marker")]
    check("capture stage captures its own exit status",
          'CAPTURE_STATUS="${PIPESTATUS[0]}"' in block, block[-400:])
    check("capture stage accepts 0 and 2 and refuses anything else",
          '"$CAPTURE_STATUS" != 0 ] && [ "$CAPTURE_STATUS" != 2 ' in block
          and 'exit "$CAPTURE_STATUS"' in block, block[-400:])
    check("a sealed-with-caveats capture is logged, not hidden",
          "sealed WITH CAVEATS" in block, block[-400:])

    print()
    if FAILED:
        print("selftest_stage_measure: %d FAILED" % len(FAILED))
        return 1
    print("selftest_stage_measure: all passed%s"
          % (" (%d skipped)" % len(SKIPPED) if SKIPPED else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
