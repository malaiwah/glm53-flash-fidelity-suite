#!/usr/bin/env python3
"""The contract path, end to end, over FAKE sealed datasets, at $0.

WHY THIS EXISTS. On 2026-09-04/05 the GLM-5.3 lane paid for seven pods that
died in the CONTROLLER/CONTRACT layer after the science had already passed:
a capture exit code the driver treated as failure, a scope vocabulary the
registry refuses, a `weights_decode` the capture never recorded, a
qualification target surface hardcoded to `fp8-block`, and a head rule
(HEAD-1b) that refused two bitwise-identical cold captures. The Fruit fixture
caught none of them, because it exercises the DECODE layer. This file drives
the layer the pods die in:

    job contract -> verify -> verify_repeat -> compare_root -> qualify_root
                 -> compare_reference (own heads) -> result archive -> post

through the REAL `bin/stage_measure.sh`, the REAL comparator / qualifier /
archiver, over tiny datasets sealed by the same writer the pod uses, for the
three candidate surfaces the streaming loader decodes (block-scaled FP8,
stock exllamav3 trellis, TP-rank-composed trellis) and for a candidate whose
head differs from the root's (the exllamav3 head_bits=16 fp16 round trip).

Nothing here is a stub except the provider: the interpreter the stage calls
is the one running this file, `hf` is the argv-logging stand-in of
selftest_stage_measure (no network), and the two fetch stages are marked done
because the datasets are already on disk -- which is exactly the state a pod
is in when the contract layer fails.

Every rung is a property a paid run would have to satisfy; a red rung here is
a pod that would not have qualified.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import selftest_fidelity_dataset as fixtures  # noqa: E402
import selftest_stage_measure as stages  # noqa: E402
from fidelity import common  # noqa: E402
from fidelity import dsformat as F  # noqa: E402
from fidelity import jobcontract  # noqa: E402
from fidelity import resultsink as RS  # noqa: E402

PASS, FAIL = [], []


def has_code(receipt, code):
    return receipt is not None and any(d.get("code") == code for d in receipt.get("disclosures") or [])


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print("  %s  %s" % ("PASS" if condition else "FAIL", name))
    if not condition and detail:
        for line in str(detail).splitlines()[-12:]:
            print("        %s" % line[:220])


# ---------------------------------------------------------------------------
# The three surfaces the streaming loader decodes, as the CANDIDATE block a
# controller writes and the identity a capture seals. One table, both sides:
# a mismatch between the two columns is the defect qualify_root exists to
# refuse, so the harness builds the sealed side FROM the job side.
# ---------------------------------------------------------------------------
SURFACES = {
    "fp8-block": {
        "codec": "fp8_e4m3", "declared_bits": 8.0, "target_surface": "fp8-block",
        "weights_decode": {
            "method": "fp8-block-dequant-to-bf16",
            "quantization_config": {
                "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
                "activation_scheme": "dynamic", "modules_to_not_convert": ["lm_head"]}},
    },
    "exl3-trellis": {
        "codec": "exl3-trellis", "declared_bits": 4.0, "target_surface": "exl3hf",
        "weights_decode": {
            "method": "exl3-trellis-decode-to-bf16",
            "quantization_config": {
                "quant_method": "exl3", "bits": 4, "codebook": None,
                "head_bits": None, "modules_to_not_convert": []}},
    },
    "exl3-tp-compose": {
        "codec": "exl3-mcg", "declared_bits": 3.25, "target_surface": "exl3hf",
        "weights_decode": {
            "method": "exl3-trellis-tp-compose-to-bf16",
            "quantization_config": {
                "quant_method": "exl3", "bits": 3.25, "codebook": "mcg",
                "head_bits": None, "modules_to_not_convert": []}},
    },
    "nvfp4": {
        "codec": "nvfp4", "declared_bits": 4.0, "target_surface": "nvfp4",
        "weights_decode": {
            "method": "nvfp4-modelopt-dequant-to-bf16",
            "quantization_config": {
                "quant_method": "modelopt", "quant_algo": "NVFP4", "num_bits": 4,
                "group_size": 16, "weights_declared_by": "config_groups.group_0.weights",
                "activation_scheme": "static-nvfp4-not-applied",
                "producer": {"name": "modelopt", "version": "0.47.0"},
                "ignore_count": 231, "ignore_sha256": "5" * 64}},
    },
}


# The panel's tokenizer identity names the ROOT release (PANEL-D6): a
# candidate shares the tokenizer files byte for byte and is captured against
# the root's binding, so every fixture here declares the root's tokenizer.
ROOT_TOKENIZER = {
    "id": "selftest-root-tokenizer",
    "repository": "selftest/root-weights",
    "revision": "a" * 40,
    "vocab_size": fixtures.VOCAB,
    "files": [{"path": "tokenizer.json", "bytes": 17, "sha256": "4" * 64}],
    "identity_sha256": "3" * 64,
    "add_special_tokens": False,
    "chat_template_applied": False,
}


RESOURCES = {"workspace_available_bytes_minimum": 4096,
             "container_available_bytes_minimum": 2048,
             "min_vcpu_count": 1, "min_memory_gb": 1, "expected_vram_bytes": 1}


def write_attestation(sb):
    """The controller's live attestation, as it lands on the box before any stage.

    The archive contract (resultsink._check_local_runpod_attestation) requires it
    for every job that carries resource_requirements, and binds it to the job's
    resource block and gpu; the harness writes the same shape the controller does.
    """
    doc = {
        "schema": RS.RUNPOD_ATTESTATION_SCHEMA,
        "provider": "runpod", "provider_id": "pod-selftest",
        "observed_at_utc": "2026-01-01T00:00:01Z",
        "clock": {"controller_send_epoch": 1767225600.0,
                  "controller_send_utc": "2026-01-01T00:00:00Z",
                  "controller_receive_epoch": 1767225601.0,
                  "controller_receive_utc": "2026-01-01T00:00:01Z",
                  "round_trip_seconds": 1.0, "remote_time_epoch": 1767225601,
                  "remote_time_utc": "2026-01-01T00:00:01Z", "clock_skew_seconds": 0.5,
                  "allowed_skew_seconds": 31.0, "within_bound": True},
        "expected": {"expected_vram_bytes": RESOURCES["expected_vram_bytes"],
                     "min_vcpu": RESOURCES["min_vcpu_count"],
                     "min_ram_gb": RESOURCES["min_memory_gb"],
                     "volume_gb": 100, "container_disk_gb": 20,
                     "workspace_available_bytes_minimum": RESOURCES["workspace_available_bytes_minimum"],
                     "container_available_bytes_minimum": RESOURCES["container_available_bytes_minimum"],
                     "gpu_model": "selftest-gpu"},
        "observed": {"remote_time_epoch": 1767225601, "remote_time_utc": "2026-01-01T00:00:01Z",
                     "filesystems": {"workspace": {"available_bytes": 8192},
                                     "container": {"available_bytes": 4096}}},
        "transport_error": None,
        "checks": {"container_available_bytes": True, "gpu_model": True, "remote_clock": True,
                   "storage": True, "workspace_available_bytes": True},
        "failures": [], "ok": True,
    }
    doc["attestation_sha256"] = hashlib.sha256(json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()
    (sb.fs / "receipts" / RS.RUNPOD_ATTESTATION_PATH.split("/", 1)[1]).write_text(
        json.dumps(doc, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def build_root(tmp, seed=71, head_seed=7):
    root = os.path.join(tmp, "reference-root")
    fixtures.build_dataset(root, seed=seed, head_seed=head_seed, run_name="root-cold-1",
                           cold_run="root-cold-1", dataset_repository="selftest/root-v1",
                           weights_repository="selftest/root-weights",
                           tokenizer=dict(ROOT_TOKENIZER), qualification_contract=True)
    return root


def build_candidate(where, surface, *, seed, head_seed, label, model_revision):
    spec = SURFACES[surface]
    fixtures.build_dataset(where, seed=seed, head_seed=head_seed, role="quant",
                           quantized=True, run_name=label, cold_run=label,
                           dataset_repository="selftest/candidate-v1",
                           weights_repository="selftest/candidate-weights",
                           model_revision=model_revision,
                           tokenizer=dict(ROOT_TOKENIZER), qualification_contract=True,
                           panel_binding_file="panel-binding.json",
                           codec=spec["codec"], declared_bits=spec["declared_bits"],
                           weights_decode=spec["weights_decode"])


def candidate_job(surface, root_manifest, candidate_manifest, binding, *, own_heads=True,
                  publish=True):
    """A root-protocol job on a quantized target, the shape measure-cloud writes."""
    spec = SURFACES[surface]
    weights = "selftest/candidate-weights"
    revision = candidate_manifest["weights"]["revision"]
    destination = "selftest/candidate-v1"
    q_bundle = jobcontract.finalize_bundle_manifest(
        [{"path": "bin/fidelity_dataset.py", "bytes": 1, "sha256": "6" * 64}],
        "contract-harness")
    q_control = jobcontract.finalize_bundle_manifest(
        [{"path": "bin/fidelity/jobcontract.py", "bytes": 1, "sha256": "7" * 64}],
        "contract-harness-control")
    q_control["schema"] = "fidelity-suite/control-plane-manifest.v1"
    q_registry = {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "8" * 64}
    q_contract_sha = common.sha256_hex(json.dumps(
        {"bundle": q_bundle, "registry": q_registry},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    shards = [{"path": "model.safetensors", "bytes": 17}]
    download_manifest = [{"path": "config.json", "bytes": 1}, shards[0],
                         {"path": "model.safetensors.index.json", "bytes": 1}]
    names_sha = common.sha256_hex(json.dumps(["model.unused"], separators=(",", ":")))
    root_panel = root_manifest["panel"]
    # The controller's storage contract for the two hidden-form dataset trees;
    # the archive enforces its member and byte caps by the exact formula.
    positions = int(candidate_manifest["capture"]["scored_rows_total"])
    duplicate = 8 * 1024 * 1024          # a generous bound over two tiny trees
    uncompressed = duplicate + RS.ARCHIVE_MARGIN_BYTES
    root_capture_storage = {
        "form": "hidden", "storage_dtype": "bfloat16",
        "selected_prediction_positions": positions,
        "vocab_size": fixtures.VOCAB, "hidden_size": fixtures.HIDDEN, "bytes_per_element": 2,
        "fresh_processes": 2,
        "hidden_bytes_per_process": duplicate // 4,
        "shared_head_bytes_per_process": duplicate // 4,
        "bytes_per_process": duplicate // 2,
        "capture_bytes_total": duplicate,
        "capture_archive_duplicate_upper_bound_bytes": duplicate,
        "required_dataset_trees": 2,
        "result_archive_max_members": 2 * positions + 128,
        "result_archive_max_uncompressed_bytes": uncompressed,
        "result_archive_max_transfer_bytes": (
            uncompressed + ((uncompressed + 16382) // 16383) * 5 + 64),
    }
    candidate_block = {
        "scope": {"path": "candidate/scope.json", "sha256": "a" * 64,
                  "scope_digest": candidate_manifest["scope"]["scope_digest"]},
        "codec": spec["codec"], "declared_bits": spec["declared_bits"],
        "weights_decode": spec["weights_decode"],
        "reference": {
            "repository": root_manifest["dataset"]["repository"],
            "revision": "5" * 40,
            "dataset_sha256": root_manifest["dataset_sha256"],
            "capture_content_digest": root_manifest["capture"]["capture_content_digest"],
            "dataset_id": root_manifest["dataset"]["id"],
            "panel_id": root_panel["panel_id"],
            "suite_token_hash_sha256": root_panel["suite_token_hash_sha256"]},
    }
    return jobcontract.finalize_job({
        "schema": "fidelity-suite/job.v2",
        "execution_attempt": {"number": 1, "kind": "local-container", "attempt_id": "9" * 24},
        "bundle": q_bundle, "control_plane": q_control, "bundle_registry": q_registry,
        "bundle_contract_sha256": q_contract_sha,
        "role": "root", "lane": "sealed-ep8", "cold_runs": 2, "recipe": "local-container",
        "runtime": {}, "environment": {"provider": "local-container", "gpu": "selftest-gpu"},
        "measurer": {},
        "resource_requirements": RESOURCES,
        "produced_by": {"dependencies": {"profile": "root-hf-transformers-bf16",
                                         "lane": "sealed-ep8", "provider": "local-container"}},
        "profile": {"profile_id": "root-hf-transformers-bf16", "lane": "root",
                    "source": "native", "surface": spec["target_surface"], "form": "hidden",
                    "engine": "hf-transformers", "compute_dtype": "bfloat16",
                    "device": "cuda", "schedule": "two-fresh-process-qualification"},
        "timing": {"kind": "contract-harness"},
        "scope": {"kind": "contract-harness"},
        "target": {"repo_id": weights, "revision": revision, "path": None,
                   "surface": spec["target_surface"], "codec": spec["codec"],
                   "bits": spec["declared_bits"],
                   "config_sha256": "a" * 64, "index_sha256": "b" * 64,
                   "shard_manifest_sha256": common.sha256_hex(json.dumps(
                       shards, sort_keys=True, separators=(",", ":"))),
                   "model_bytes": 17, "shards": shards,
                   "download_manifest": download_manifest, "download_bytes_total": 19,
                   "download_manifest_sha256": common.sha256_hex(json.dumps(
                       download_manifest, sort_keys=True, separators=(",", ":"))),
                   "weights_license": None,
                   "root_capture_storage": root_capture_storage},
        "panel": {"binding_file_sha256": "2" * 64, "binding_path": "panel-binding.json",
                  "resolved_binding": binding},
        "capture": {
            "dataset_id": candidate_manifest["dataset"]["id"],
            "panel_id": binding["panel"]["id"],
            "dataset_name": candidate_manifest["dataset"]["name"],
            "author": "selftest",
            "dataset_repository": destination,
            "publish_root_to": destination if publish else None,
            "dataset_license": "mit", "weights_license": None,
            "form": "hidden", "schedule": "layer-outer", "device": "cuda",
            "dtype": "bfloat16", "engine": "hf-transformers",
            "preview_of": None, "race": False,
            "replay_device": "numpy", "replay_dtype": "float32", "vocab_chunk": 8192,
            "replay": {"device": "numpy", "dtype": "float32", "vocab_chunk": 8192},
            "own_heads": own_heads,
            "root_protocol": {"schedule": "two-fresh-process-qualification",
                              "fresh_processes": 2, "run_count_per_process": 1,
                              "exact_self_comparison": True, "qualification_required": True,
                              "canonical_publication_required": publish,
                              "publication_mode": ("canonical-public" if publish
                                                   else "qualified-unpublished")},
            "unexpected_tensor_allowlist": {"path": "allowlist.json",
                                            "artifact_sha256": "5" * 64,
                                            "canonical_sorted_names_sha256": names_sha},
            "candidate": candidate_block,
        },
    })


def stage(sb, name, bash, **extra):
    proc, calls = sb.run(name, bash, provision_target=False, **extra)
    return proc, calls, proc.stdout + proc.stderr


def drive(tmp, bash, surface, *, head_seed=7, label=None):
    """Build the fake sealed world, then run the real contract path over it."""
    label = label or surface
    case = Path(tmp) / label
    case.mkdir()
    root = build_root(str(case))
    root_manifest = F.load_manifest(root)
    revision = "c" * 40
    first = str(case / "cand-1")
    repeat = str(case / "cand-2")
    build_candidate(first, surface, seed=93, head_seed=head_seed, label="root-cold-1",
                    model_revision=revision)
    build_candidate(repeat, surface, seed=93, head_seed=head_seed, label="root-cold-2",
                    model_revision=revision)
    cand_manifest = F.load_manifest(first)
    runtime = F.read_json(os.path.join(first, cand_manifest["runtime"]["file"]))
    binding = runtime["capture_tool"]["resolved_panel_binding"]["binding"]
    job = candidate_job(surface, root_manifest, cand_manifest, binding)

    sb = stages.Sandbox(case / "sb", job, real_scripts=("fidelity_dataset.py",),
                        finalize_job_doc=False)
    # The upload lands the whole bundle at $FS, not bin/ alone: the verifier
    # reaches the registry's vendored schema validator and the dataset schemas
    # by path, exactly as it does on the pod (bin/BUNDLE.txt ships both).
    for rel in ("registry/tools", "registry/schema", "docs/schema", "engines/tools"):
        src = ROOT / rel
        if src.is_dir():
            shutil.copytree(src, sb.fs / rel, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*-evidence", "*.pt"))
    # The datasets are already on disk, as they are on a pod whose captures
    # sealed: move them into the run root and mark the stages that put them
    # there. The reference is the verified published root the pod symlinked.
    shutil.move(first, sb.fs / "dataset")
    shutil.move(repeat, sb.fs / "dataset-repeat")
    os.symlink(root, sb.fs / "reference")
    # fetch_reference verifies the published root anonymously and writes its
    # receipt; the fetch is the only network act, so the harness runs the same
    # verifier over the local root and marks the stage done.
    verified = subprocess.run(
        [sys.executable, str(sb.fs / "bin" / "fidelity_dataset.py"), "verify", root,
         "--json", str(sb.fs / "receipts" / "reference-verify.json")],
        capture_output=True, text=True)
    if verified.returncode not in (0, 2):
        raise SystemExit("reference verify failed: %s" % (verified.stdout + verified.stderr)[-800:])
    sb.write_target_census()          # fetch_target's sealed census + its marker
    write_attestation(sb)
    for done in ("setup", "fetch_reference", "capture", "capture_repeat"):
        sb.write_bound_marker(done)
    return sb, job, root, root_manifest, cand_manifest


def receipts_of(sb):
    r = sb.fs / "receipts"
    out = {}
    for rel in ("dataset-verify.json", "dataset-repeat-verify.json",
                "root-comparison/comparison-receipt.json", "root-qualification.json",
                "reference-comparison/comparison-receipt.json"):
        p = r / rel
        out[rel] = F.read_json(str(p)) if p.is_file() else None
    return out


def main() -> int:
    bash = stages.modern_bash()
    if not bash:
        print("  FAIL  no bash >= 4.4 on this host; the stage driver needs one")
        return 1
    with tempfile.TemporaryDirectory(prefix="qfs-contract-") as tmp:
        print("== C1-C5: the contract path on a block-scaled FP8 candidate ==")
        sb, job, root, root_manifest, cand_manifest = drive(tmp, bash, "fp8-block")
        p, calls, out = stage(sb, "verify", bash)
        check("C1  verify runs the real verifier over the sealed candidate and writes its receipt",
              p.returncode == 0 and (sb.fs / "receipts" / "dataset-verify.json").is_file()
              and sb.marker("verify").is_file(), out)
        p, calls, out = stage(sb, "verify_repeat", bash)
        check("C1b verify_repeat likewise", p.returncode == 0 and sb.marker("verify_repeat").is_file(), out)
        p, calls, out = stage(sb, "compare_root", bash)
        rc = receipts_of(sb)["root-comparison/comparison-receipt.json"]
        check("C2  compare_root is a forced, exact-zero SC-1 between the two cold captures, "
              "strict and free of weights-decode caveats (the same artifact on both sides)",
              p.returncode == 0 and rc is not None
              and rc["comparison_kind"] == "reproduction_confirmation"
              and rc["metric"]["value"] == 0.0 and rc["self_compare"]["force_compute_agreed"] is True
              and rc["comparability"]["class"] == "strict"
              and not has_code(rc, "activation_quantization_not_captured")
              and not has_code(rc, "weights_reconstructed"),
              out)
        p, calls, out = stage(sb, "qualify_root", bash)
        q = receipts_of(sb)["root-qualification.json"]
        check("C3  qualify_root binds the sealed candidate identity (codec, bits, scope, "
              "weights_decode) to the job's candidate block and the fp8-block target",
              p.returncode == 0 and q is not None and common.verify_seal(q)
              and q["job_contract"]["candidate"]["codec"] == "fp8_e4m3"
              and q["job_contract"]["target"]["surface"] == "fp8-block"
              and q["captures"]["canonical"]["candidate"]["weights_decode"]["method"]
              == "fp8-block-dequant-to-bf16", out)
        p, calls, out = stage(sb, "compare_reference", bash)
        ref = receipts_of(sb)["reference-comparison/comparison-receipt.json"]
        argv = next((c[1] for c in calls if any("fidelity_dataset.py" in a for a in c[1])), [])
        check("C4  compare_reference scores the candidate against the reference with the "
              "job's replay contract and --own-heads, and seals an ADVISORY measurement "
              "carrying activation_quantization_not_captured (activation_scheme dynamic)",
              p.returncode in (0, 2) and ref is not None
              and ref["comparison_kind"] == "measurement"
              and ref["estimator"]["head_policy"] == "native_head"
              and ref["comparability"]["class"] == "advisory"
              and has_code(ref, "activation_quantization_not_captured")
              and sum(1 for d in ref["disclosures"]
                      if d["code"] == "activation_quantization_not_captured") == 1
              and "--own-heads" in argv
              and argv[argv.index("--replay-device") + 1] == "numpy"
              and ref["reference"]["dataset_sha256"] == root_manifest["dataset_sha256"]
              and ref["candidate"]["dataset_sha256"] == cand_manifest["dataset_sha256"],
              out + "\nargv=%r" % argv)
        check("C4b the number is a real KL(root || candidate) on the fixture, not a short-circuit",
              ref is not None and ref["metric"]["value"] > 0.0
              and ref["comparator"]["short_circuited"] is False, repr(ref and ref["metric"]))

        # -- the archive and the post, over the tree the stages left ----------
        archive = Path(tmp) / "fp8-result.tar.gz"
        done = ["setup", "fetch_target", "fetch_reference", "capture", "verify",
                "capture_repeat", "verify_repeat", "compare_root", "qualify_root",
                "compare_reference"]
        # A pod never publishes: its sink bundle says qualified-unpublished, and
        # the controller publishes from the retrieved, verified archive.
        built = subprocess.run(
            [sys.executable, str(sb.fs / "bin" / "result_archive.py"), "--fs-root", str(sb.fs),
             "--verb", "capture", "--status", "qualified-unpublished",
             "--stages", ",".join(done), "--out", str(archive)],
            capture_output=True, text=True)
        verified = None
        if built.returncode == 0:
            try:
                verified = RS.verify_archive(str(archive))
            except Exception as exc:  # noqa: BLE001
                verified = {"error": repr(exc)}
        check("C5  result_archive builds the sink bundle from that tree and it verifies "
              "with the reference comparison bound to the job",
              built.returncode == 0 and isinstance(verified, dict)
              and (verified.get("manifest") or {}).get("status") == "qualified-unpublished",
              (built.stdout + built.stderr)[-1500:] + "\n%r" % (verified,))
        post = Path(tmp) / "fp8-post.md"
        rendered = subprocess.run(
            [sys.executable, str(ROOT / "bin" / "fidelity_post.py"), "render",
             "--result", str(sb.fs), "--out", str(post)], capture_output=True, text=True)
        body = post.read_text(encoding="utf-8") if post.is_file() else ""
        check("C5b the discussion post renders from those receipts and asks for --own-heads",
              rendered.returncode == 0 and repr(ref["metric"]["value"]) in body
              and "--own-heads" in body and "dequantize-and-run" in body,
              rendered.stdout + rendered.stderr)

        # -- the two trellis surfaces --------------------------------------
        print("\n== C6-C7: the two trellis surfaces through the same path ==")
        for surface, method in (("exl3-trellis", "exl3-trellis-decode-to-bf16"),
                                ("exl3-tp-compose", "exl3-trellis-tp-compose-to-bf16")):
            sb2, job2, root2, rm2, cm2 = drive(tmp, bash, surface)
            outs = []
            ok = True
            for name in ("verify", "verify_repeat", "compare_root", "qualify_root",
                         "compare_reference"):
                p, calls, out = stage(sb2, name, bash)
                outs.append("[%s rc=%d]\n%s" % (name, p.returncode, out[-600:]))
                ok = ok and p.returncode == 0
            q2 = receipts_of(sb2)["root-qualification.json"]
            ref2 = receipts_of(sb2)["reference-comparison/comparison-receipt.json"]
            check("C6  %s: qualify_root maps the decode to the exl3hf target and "
                  "compare_reference seals an ADVISORY own-head measurement carrying "
                  "weights_reconstructed (once)" % surface,
                  ok and q2 is not None and ref2 is not None
                  and q2["job_contract"]["target"]["surface"] == "exl3hf"
                  and q2["captures"]["canonical"]["candidate"]["weights_decode"]["method"] == method
                  and ref2["estimator"]["head_policy"] == "native_head"
                  and ref2["comparability"]["class"] == "advisory"
                  and sum(1 for d in ref2["disclosures"] if d["code"] == "weights_reconstructed") == 1,
                  "\n".join(outs))
            post2 = Path(tmp) / ("%s-post.md" % surface)
            rendered = subprocess.run(
                [sys.executable, str(ROOT / "bin" / "fidelity_post.py"), "render",
                 "--result", str(sb2.fs), "--out", str(post2)], capture_output=True, text=True)
            body2 = post2.read_text(encoding="utf-8") if post2.is_file() else ""
            check("C7  %s: the post names its decode, never the generic fallback sentence" % surface,
                  rendered.returncode == 0 and "decode-and-run" in body2
                  and "decode recorded in the sealed runtime receipt" not in body2,
                  rendered.stdout + rendered.stderr)

        # -- a candidate whose head is NOT the root's (HEAD-1d) --------------
        print("\n== C8: a candidate head that differs from the root's ==")
        sb3, job3, root3, rm3, cm3 = drive(tmp, bash, "exl3-trellis", head_seed=99,
                                           label="exl3-other-head")
        outs = []
        ok = True
        for name in ("verify", "verify_repeat", "compare_root", "qualify_root",
                     "compare_reference"):
            p, calls, out = stage(sb3, name, bash)
            outs.append("[%s rc=%d]\n%s" % (name, p.returncode, out[-600:]))
            ok = ok and p.returncode == 0
        ref3 = receipts_of(sb3)["reference-comparison/comparison-receipt.json"]
        check("C8  differing heads: the whole path still qualifies and scores under HEAD-1d, "
              "each side through its own sealed head (advisory: the trellis decode caveat)",
              ok and ref3 is not None and ref3["estimator"]["head_policy"] == "native_head"
              and ref3["comparability"]["class"] == "advisory"
              and has_code(ref3, "weights_reconstructed")
              and ref3["comparator"]["head_applied_reference_tensor_content_sha256"]
              != ref3["comparator"]["head_applied_candidate_tensor_content_sha256"]
              and any(d["code"] == "native_head_replay" for d in ref3["disclosures"]),
              "\n".join(outs))
        # The same world with the job NOT asking for own heads is the 2026-09-05
        # drowzeys pod: two sealed captures and a HEAD-1b refusal at the last stage.
        sb4, job4, root4, rm4, cm4 = drive(tmp, bash, "exl3-trellis", head_seed=99,
                                           label="exl3-other-head-shared")
        job4_shared = json.loads((sb4.fs / "job.json").read_text(encoding="utf-8"))
        job4_shared["capture"]["own_heads"] = False
        (sb4.fs / "job.json").write_text(json.dumps(jobcontract.finalize_job(
            {k: v for k, v in job4_shared.items() if k not in ("job_id", "job_id_full")})),
            encoding="utf-8")
        for done in ("setup", "fetch_target", "fetch_reference", "capture", "capture_repeat"):
            sb4.write_bound_marker(done)
        for name in ("verify", "verify_repeat", "compare_root", "qualify_root"):
            stage(sb4, name, bash)
        p, calls, out = stage(sb4, "compare_reference", bash)
        check("C8b ...and without own_heads on the job the SAME world refuses at "
              "compare_reference with HEAD-1b, exactly the paid failure",
              p.returncode == 3 and "HEAD-1b" in out and "--own-heads" in out, out[-800:])

        # -- the contract refusals a wrong controller must hit BEFORE a rental --
        print("\n== C9-C10: contract mismatches refuse at qualify_root ==")
        sb5, job5, root5, rm5, cm5 = drive(tmp, bash, "fp8-block", label="fp8-wrong-decode")
        wrong = json.loads((sb5.fs / "job.json").read_text(encoding="utf-8"))
        wrong["capture"]["candidate"]["weights_decode"] = dict(
            wrong["capture"]["candidate"]["weights_decode"],
            quantization_config=dict(
                wrong["capture"]["candidate"]["weights_decode"]["quantization_config"],
                weight_block_size=[64, 64]))
        (sb5.fs / "job.json").write_text(json.dumps(jobcontract.finalize_job(
            {k: v for k, v in wrong.items() if k not in ("job_id", "job_id_full")})),
            encoding="utf-8")
        for done in ("setup", "fetch_target", "fetch_reference", "capture", "capture_repeat"):
            sb5.write_bound_marker(done)
        for name in ("verify", "verify_repeat", "compare_root"):
            stage(sb5, name, bash)
        p, calls, out = stage(sb5, "qualify_root", bash)
        check("C9  a job whose candidate decode differs from the decode the capture sealed "
              "is refused at qualify_root by name (the 2026-09-04 weights_decode defect)",
              p.returncode != 0 and "candidate identity differs" in out, out[-800:])

        sb6, job6, root6, rm6, cm6 = drive(tmp, bash, "exl3-trellis", label="exl3-wrong-surface")
        wrong = json.loads((sb6.fs / "job.json").read_text(encoding="utf-8"))
        wrong["target"]["surface"] = "fp8-block"
        wrong["profile"]["surface"] = "fp8-block"
        (sb6.fs / "job.json").write_text(json.dumps(jobcontract.finalize_job(
            {k: v for k, v in wrong.items() if k not in ("job_id", "job_id_full")})),
            encoding="utf-8")
        for done in ("setup", "fetch_target", "fetch_reference", "capture", "capture_repeat"):
            sb6.write_bound_marker(done)
        for name in ("verify", "verify_repeat", "compare_root"):
            stage(sb6, name, bash)
        p, calls, out = stage(sb6, "qualify_root", bash)
        check("C10 a trellis candidate whose target says fp8-block is refused at qualify_root "
              "(the 2026-09-04 hardcoded-surface defect, from the other side)",
              p.returncode != 0 and "target contract differs" in out, out[-800:])

        # -- the modelopt NVFP4 surface through the same path -----------------
        print("\n== C11-C12: the modelopt NVFP4 surface (job -> qualify -> compare -> "
              "archive -> post) ==")
        sb7, job7, root7, rm7, cm7 = drive(tmp, bash, "nvfp4")
        outs = []
        ok = True
        for name in ("verify", "verify_repeat", "compare_root", "qualify_root",
                     "compare_reference"):
            p, calls, out = stage(sb7, name, bash)
            outs.append("[%s rc=%d]\n%s" % (name, p.returncode, out[-600:]))
            ok = ok and p.returncode == 0
        q7 = receipts_of(sb7)["root-qualification.json"]
        ref7 = receipts_of(sb7)["reference-comparison/comparison-receipt.json"]
        archive7 = Path(tmp) / "nvfp4-result.tar.gz"
        built7 = subprocess.run(
            [sys.executable, str(sb7.fs / "bin" / "result_archive.py"), "--fs-root", str(sb7.fs),
             "--verb", "capture", "--status", "qualified-unpublished",
             "--stages", ",".join(done), "--out", str(archive7)],
            capture_output=True, text=True)
        verified7 = None
        if built7.returncode == 0:
            try:
                verified7 = RS.verify_archive(str(archive7))
            except Exception as exc:  # noqa: BLE001
                verified7 = {"error": repr(exc)}
        outs.append("[result_archive rc=%d]\n%s\n%r" % (
            built7.returncode, (built7.stdout + built7.stderr)[-600:], verified7))
        check("C11 nvfp4: qualify_root binds the modelopt decode to the nvfp4 target "
              "(codec nvfp4 @ 4 bits, activation_scheme static-nvfp4-not-applied), "
              "compare_reference seals an ADVISORY own-head measurement carrying "
              "activation_quantization_not_captured (input scales not applied) and the archive builds",
              ok and q7 is not None and ref7 is not None
              and q7["job_contract"]["target"]["surface"] == "nvfp4"
              and q7["job_contract"]["candidate"]["codec"] == "nvfp4"
              and q7["job_contract"]["candidate"]["declared_bits"] == 4.0
              and q7["captures"]["canonical"]["candidate"]["weights_decode"]["method"]
              == "nvfp4-modelopt-dequant-to-bf16"
              and q7["captures"]["canonical"]["candidate"]["weights_decode"]
              ["quantization_config"]["activation_scheme"] == "static-nvfp4-not-applied"
              and ref7["estimator"]["head_policy"] == "native_head"
              and ref7["comparability"]["class"] == "advisory"
              and has_code(ref7, "activation_quantization_not_captured")
              and built7.returncode == 0 and isinstance(verified7, dict)
              and (verified7.get("manifest") or {}).get("status") == "qualified-unpublished",
              "\n".join(outs))
        post7 = Path(tmp) / "nvfp4-post.md"
        rendered = subprocess.run(
            [sys.executable, str(ROOT / "bin" / "fidelity_post.py"), "render",
             "--result", str(sb7.fs), "--out", str(post7)], capture_output=True, text=True)
        body7 = post7.read_text(encoding="utf-8") if post7.is_file() else ""
        check("C12 nvfp4: the post names the modelopt dialect, group 16, e2m1, the scale "
              "product, routed-experts-only and the unapplied input_scale -- never the "
              "generic fallback sentence",
              rendered.returncode == 0 and "ModelOpt NVFP4" in body7 and "group 16" in body7
              and "e2m1" in body7 and "weight_scale.f32 x weight_scale_2" in body7
              and "routed experts only" in body7 and "input_scale" in body7
              and "NOT applied" in body7
              and "decode recorded in the sealed runtime receipt" not in body7,
              (rendered.stdout + rendered.stderr + body7)[-900:])
        # ... and the mirror-side refusal: an nvfp4 job whose decode block was
        # written by a controller that disagrees with the pod (a different
        # ignore hash) is refused at qualify_root by name.
        sb8, job8, root8, rm8, cm8 = drive(tmp, bash, "nvfp4", label="nvfp4-wrong-decode")
        wrong = json.loads((sb8.fs / "job.json").read_text(encoding="utf-8"))
        wrong["capture"]["candidate"]["weights_decode"] = dict(
            wrong["capture"]["candidate"]["weights_decode"],
            quantization_config=dict(
                wrong["capture"]["candidate"]["weights_decode"]["quantization_config"],
                ignore_sha256="6" * 64))
        (sb8.fs / "job.json").write_text(json.dumps(jobcontract.finalize_job(
            {k: v for k, v in wrong.items() if k not in ("job_id", "job_id_full")})),
            encoding="utf-8")
        for done in ("setup", "fetch_target", "fetch_reference", "capture", "capture_repeat"):
            sb8.write_bound_marker(done)
        for name in ("verify", "verify_repeat", "compare_root"):
            stage(sb8, name, bash)
        p, calls, out = stage(sb8, "qualify_root", bash)
        check("C12b nvfp4: a job whose contract block differs from the sealed decode by one "
              "field (ignore_sha256) is refused at qualify_root by name",
              p.returncode != 0 and "candidate identity differs" in out, out[-800:])

    print("\nselftest_contract_harness: %d passed, %d failed" % (len(PASS), len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
