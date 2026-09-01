#!/usr/bin/env python3
"""The image path -- what it must not change, and what it must not drift from.

    python3 bin/selftest_container.py

A container entrypoint is a SECOND transport for stages that already exist.
Every rung here exists because a second transport is a second chance to
disagree with the first, and the disagreements are all silent:

  C1/C2  the stage sequence, in ONE place.  It used to be a literal inside
         measure_cloud plus a second literal three lines below it for the
         `materialize` insertion; a container copy would have been a third.
         A drift here does not crash -- it measures a tree nothing decoded, or
         discovers at `seal` that there is nothing to seal, three GPU-hours in.
  C3     the job document.  `stage_measure.sh` reads one contract; two writers
         of it must not diverge, and the way they diverge is by one of them
         silently omitting a key.
  C4     the token never reaches argv, never reaches a stage's environment.
  C5/C6  what lands on the machine is bin/BUNDLE.txt's audited set, not
         "whatever was in the directory".
  C7     an unknown image digest is recorded as null WITH THE REASON, never
         guessed.
  C8     THE ACCEPTANCE TEST, in code: recording which container ran must not
         move `stack_fingerprint_sha256`, and with no pin present a capture's
         bytes must be identical to what they were before the field learned how
         to be filled.  A published dataset does not get to shift because we
         added a container.
  C9     the image cannot install its own torch: bootstrap_measure.sh is the
         specification and the Dockerfile must run it, not paraphrase it.

Stock python3.9, no installs, no network, no GPU.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import urllib.parse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SUITE = HERE.parent

import container_entry as CE                              # noqa: E402
import selftest_panel as PANEL_TEST                         # noqa: E402
from fidelity import dsmanifest, stages                   # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  -- " + detail) if (detail and not ok) else ""))
    if not ok:
        FAILED.append(label)


# --------------------------------------------------------------------------
# C1/C2  one sequence, one owner
# --------------------------------------------------------------------------

def rung_sequence():
    print("[C1] the stage sequence is a known answer")
    check("C1a quant, no materializing surface",
          stages.stage_sequence("quant", surface="gguf")
          == ["setup", "fetch_target", "fetch_panel", "measure", "score", "seal"])
    check("C1b exl3hf inserts materialize AFTER fetch_target",
          stages.stage_sequence("quant", surface="exl3hf")
          == ["setup", "fetch_target", "materialize", "fetch_panel", "measure",
              "score", "seal"])
    for surface in ("tr3-published", "dione"):
        check("C1c %s materializes too" % surface,
              "materialize" in stages.stage_sequence("quant", surface=surface))
    root_stages = [
        "setup", "fetch_target", "capture", "verify",
        "capture_repeat", "verify_repeat", "compare_root", "qualify_root"]
    check("C1d root is two fresh processes plus exact qualification",
          stages.stage_sequence("root") == root_stages)
    try:
        stages.stage_sequence("root", race=True)
        check("C1e root+race refuses before composing stages", False)
    except ValueError:
        check("C1e root+race refuses before composing stages", True)
    check("C1f a root capture never materializes, whatever the surface",
          stages.stage_sequence("root", surface="exl3hf") == root_stages)
    check("C1g every emitted stage is one stage_measure.sh answers to",
          stages.unknown_stages(
              stages.stage_sequence("quant", surface="exl3hf")
              + stages.stage_sequence("root")) == [])

    print("[C2] the SSH controller and the container share that one owner")
    import measure_cloud                                   # noqa: E402
    check("C2a measure_cloud uses fidelity.stages.stage_sequence itself",
          measure_cloud.stage_sequence is stages.stage_sequence)
    # The literal it replaced is the thing that must not come back.
    body = (SUITE / "bin" / "measure_cloud.py").read_text(encoding="utf-8")
    check("C2b no second copy of the sequence survives in measure_cloud",
          '"fetch_panel", "measure", "score", "seal"' not in body)


# --------------------------------------------------------------------------
# C3  finalized local quant/root job contracts
# --------------------------------------------------------------------------


def _root_fixture(work: Path):
    panel_dir = work / "panel-source"
    tokenizer_root = work / "tokenizer"
    panel_dir.mkdir()
    PANEL_TEST.modern_fixture(panel_dir, tokenizer_root)
    written = CE.PANEL.write_panel_archive(
        panel_dir, work / "panel.tar", tokenizer_root=tokenizer_root)
    binding_raw = (
        json.dumps(written["binding"], indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    binding_file = work / "panel.binding.json"
    binding_file.write_bytes(binding_raw)
    allow_raw = b'["model.layers.13.mtp.weight"]\n'
    allow_file = work / "unexpected.json"
    allow_file.write_bytes(allow_raw)
    canonical_names = json.dumps(
        ["model.layers.13.mtp.weight"], separators=(",", ":")
    ).encode("utf-8")
    return {
        "panel_dir": panel_dir,
        "tokenizer_root": tokenizer_root,
        "binding_file": binding_file,
        "binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "allow_file": allow_file,
        "allow_sha256": hashlib.sha256(allow_raw).hexdigest(),
        "names_sha256": hashlib.sha256(canonical_names).hexdigest(),
    }
def _target_descriptor(work: Path, name: str, *, repo_id: str, revision: str,
                       surface: str, codec: str, bits: float,
                       model_bytes: int = 1234,
                       config_sha256: str = "5" * 64,
                       index_sha256: str = "6" * 64):
    shards = [{
        "path": "model-00001-of-00001.safetensors",
        "bytes": model_bytes,
    }]
    shard_raw = json.dumps(
        shards, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    download_manifest = [
        {"path": "config.json", "bytes": 1},
        shards[0],
        {"path": "model.safetensors.index.json", "bytes": 1},
    ]
    download_raw = json.dumps(
        download_manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    doc = {
        "repo_id": repo_id,
        "revision": revision,
        "requested_revision": revision,
        "surface": surface,
        "codec": codec,
        "bits": bits,
        "path": None,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "model_bytes": model_bytes,
        "shards": shards,
        "shard_manifest_sha256": hashlib.sha256(shard_raw).hexdigest(),
        "download_manifest": download_manifest,
        "download_bytes_total": model_bytes + 2,
        "download_manifest_sha256": hashlib.sha256(download_raw).hexdigest(),
    }
    path = work / ("%s-target.json" % name)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc

def _refuses_job(doc, fs):
    try:
        CE.validate_job_document(CE.finalize_job(doc), fs)
    except (CE.Refusal, TypeError, ValueError):
        return True
    return False


def rung_job_document():
    print("[C3] local jobs use the one canonical finalized identity contract")
    quiet = lambda *_a, **_k: None                         # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        fixture = _root_fixture(work)
        quant_fs = work / "quant"
        quant_fs.mkdir()
        quant_panel = {
            "repo_id": "someone/panel",
            "revision": "b" * 40,
            "include": ["*"],
            "roles": "final",
            "panel_ref": "someone/panel@" + "b" * 40,
            "panel_token_sha256": "1" * 64,
            "panel_receipt_sha256": "2" * 64,
            "contexts": 25,
            "scored_positions": 512,
            "reference_ref": "someone/root@" + "c" * 40,
            "teacher_receipt_sha256": "3" * 64,
            "teacher_backend_identity_sha256": "4" * 64,
        }
        quant_panel_path = work / "quant-panel.json"
        quant_panel_path.write_text(
            json.dumps(quant_panel), encoding="utf-8")
        scope_path = work / "scope.json"
        scope_path.write_text(
            json.dumps({"layers": "all", "metric": "kl"}),
            encoding="utf-8")
        quant_target_path, quant_target = _target_descriptor(
            work, "quant",
            repo_id="malaiwah/GLM-5.3-Flash-TR3-6bpw",
            revision="9ab94105a71708a19c6d960d24b4aa6d459f5623",
            surface="tr3-published", codec="tr3", bits=6.0)
        quant_target["official_bf16_identity"] = {
            "config_sha256": "c" * 64,
            "config_bytes": 1,
            "index_sha256": "d" * 64,
            "index_bytes": 1,
        }
        quant_target_path.write_text(
            json.dumps(quant_target), encoding="utf-8")
        quant_args = argparse.Namespace(
            verb="measure", model=quant_target["repo_id"],
            revision=quant_target["revision"],
            target_descriptor=str(quant_target_path), profile="tr3-6bpw",
            panel_descriptor=str(quant_panel_path), lane="streaming",
            measurer="malaiwah", reduce_order="fp32", cold_runs=2,
            gpu="H200", gpu_count=1, host="local",
            scope_json=str(scope_path), image_pin=None,
            keep_student_logits=False,
            official_bf16_revision=
                "a6c167b62691b2bac901344b65cb651a70f53e43",
            workspace_available_bytes_minimum=1234,
            container_available_bytes_minimum=1,
            expected_vram_bytes=141 * 1024 ** 3)
        quant = CE.job_document(quant_args, SUITE, quant_fs, quiet)

        root_fs = work / "root"
        root_fs.mkdir()
        root_target_path, root_target = _target_descriptor(
            work, "root",
            repo_id="malaiwah/GLM-5.2-SIQ-Fruit-bf16",
            revision="ef68013aa6e16453cf52b5b77647f72fbe258c3c",
            surface="native-bf16", codec="bf16", bits=16.0,
            model_bytes=10102776813,
            config_sha256="5a19697e555fff140d1b089b852c3ef227114b196f8d76796560feeeb34dc44a",
            index_sha256="86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56")
        root_args = argparse.Namespace(
            verb="capture", model=root_target["repo_id"],
            revision=root_target["revision"],
            target_descriptor=str(root_target_path),
            lane="streaming", measurer="malaiwah", reduce_order="fp32",
            cold_runs=2, gpu="L4", gpu_count=1, host="local",
            scope_json=None, image_pin=None, keep_student_logits=False,
            official_bf16_revision=None,
            workspace_available_bytes_minimum=20 * 1024 ** 3,
            container_available_bytes_minimum=1024 ** 3,
            expected_vram_bytes=24 * 1024 ** 3,
            panel_dir=str(fixture["panel_dir"]),
            panel_binding=str(fixture["binding_file"]),
            panel_binding_sha256=fixture["binding_sha256"],
            panel_tokenizer_root=str(fixture["tokenizer_root"]),
            dataset_id="fidelity--t.malaiwah.root.bf16", dataset_name=None,
            form="hidden", schedule="layer-outer", race=False,
            preview_of=None, sanity_expect="Paris",
            replay_device="numpy", replay_dtype="float32", vocab_chunk=8192,
            dataset_repository="someone/root-dataset",
            publish_root_to=None,
            unexpected_tensors_allowlist=str(fixture["allow_file"]),
            unexpected_tensors_allowlist_sha256=fixture["allow_sha256"],
            unexpected_tensors_name_sha256=fixture["names_sha256"])
        root = CE.job_document(root_args, SUITE, root_fs, quiet)

        check("C3a quant is finalized by fidelity.jobcontract",
              CE.verify_job(quant) == quant["job_id_full"]
              and quant["job_id"] == quant["job_id_full"][:16])
        check("C3a2 quant pre-binds every result-archive identity block",
              quant["target"] == quant_target
              and quant["target"]["path"] is None
              and quant["panel"]["panel_ref"] == quant_panel["panel_ref"]
              and quant["reference"]["reference_ref"]
              == quant_panel["reference_ref"]
              and quant["scope"] == {"layers": "all", "metric": "kl"}
              and quant["panel"]["roles"] == "final"
              and quant["runtime"]["decode_threads"] == 28
              and quant["runtime"]["reader_threads"] == 28
              and quant["scoring"] == {
                  "schema": "fidelity-suite/kld-scoring.v1",
                  "device": "cuda",
                  "chunk_positions": 512,
                  "compute_dtype": "float64",
                  "direction": "reference_to_candidate",
                  "vocabulary": "full",
                  "reduction": "mean_of_run_means_tokenwise_kld",
              })
        check("C3b root is finalized and accepted by the exact stage contract",
              CE.validate_job_document(root, root_fs) == root["job_id_full"]
              and root["target"] == root_target
              and (root["capture"]["engine"], root["capture"]["dtype"],
                   root["capture"]["device"])
              == ("hf-transformers", "bfloat16", "cuda"))
        second_quant_fs = work / "quant-second"
        second_quant_fs.mkdir()
        second_quant = CE.job_document(
            quant_args, SUITE, second_quant_fs, quiet)
        check("C3a3 equal science has a distinct raw local attempt identity",
              second_quant["job_id_full"] == quant["job_id_full"]
              and second_quant["execution_attempt"]["attempt_id"]
              != quant["execution_attempt"]["attempt_id"]
              and second_quant != quant)
        existing_job_raw = (
            json.dumps(root, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (root_fs / "job.json").write_bytes(existing_job_raw)
        other_attempt = json.loads(json.dumps(root))
        other_attempt["execution_attempt"]["attempt_id"] = "f" * 24
        other_attempt = CE.finalize_job(other_attempt)
        supplied_job = work / "other-attempt.json"
        supplied_job.write_text(
            json.dumps(other_attempt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        before_resume_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        try:
            CE._prevalidate_stage_job(
                argparse.Namespace(job=str(supplied_job), name="setup"),
                root_fs, SUITE)
        except CE.Refusal:
            mismatch_refused = True
        else:
            mismatch_refused = False
        after_resume_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        check("C3a4 mismatched resume job refuses before mutating run root",
              mismatch_refused
              and after_resume_refusal == before_resume_refusal)
        image_mismatch = json.loads(json.dumps(root))
        mismatch_rows = image_mismatch["bundle"]["files"]
        mismatch_rows[0]["sha256"] = (
            "f" * 64 if mismatch_rows[0]["sha256"] != "f" * 64 else "e" * 64)
        image_mismatch["bundle"] = CE.finalize_bundle_manifest(
            mismatch_rows, image_mismatch["bundle"]["source"])
        binding_raw = json.dumps(
            {"bundle": image_mismatch["bundle"],
             "registry": image_mismatch["bundle_registry"]},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
        image_mismatch["bundle_contract_sha256"] = hashlib.sha256(
            binding_raw).hexdigest()
        image_mismatch = CE.finalize_job(image_mismatch)
        (root_fs / "job.json").write_text(
            json.dumps(image_mismatch, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        before_image_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        try:
            CE._prevalidate_stage_job(
                argparse.Namespace(job=None, name="setup"), root_fs, SUITE)
        except CE.Refusal:
            image_refused = True
        else:
            image_refused = False
        after_image_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        check("C3a5 image/job bundle mismatch leaves resume root byte-exact",
              image_refused and after_image_refusal == before_image_refusal)
        duplicate_job_raw = (
            b'{"execution_attempt":null,' + existing_job_raw[1:])
        (root_fs / "job.json").write_bytes(duplicate_job_raw)
        before_duplicate_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        try:
            CE._prevalidate_stage_job(
                argparse.Namespace(job=None, name="setup"), root_fs, SUITE)
        except CE.Refusal:
            duplicate_refused = True
        else:
            duplicate_refused = False
        after_duplicate_refusal = {
            path.relative_to(root_fs).as_posix(): path.read_bytes()
            for path in root_fs.rglob("*") if path.is_file()
        }
        check("C3a6 duplicate job keys refuse before run-root mutation",
              duplicate_refused
              and after_duplicate_refusal == before_duplicate_refusal)
        check("C3b2 qualification accepts an intended repository without publish",
              root["capture"]["publish_root_to"] is None
              and root["capture"]["root_protocol"]["publication_mode"]
              == "qualified-unpublished"
              and "publish_root" not in stages.stage_sequence(
                  "root", publish_root=False))
        attempted_publish = json.loads(json.dumps(root))
        attempted_publish["capture"]["publish_root_to"] = (
            attempted_publish["capture"]["dataset_repository"])
        attempted_publish["capture"]["root_protocol"][
            "canonical_publication_required"] = True
        attempted_publish["capture"]["root_protocol"][
            "publication_mode"] = "canonical-public"
        check("C3b3 local root publication always refuses",
              _refuses_job(attempted_publish, root_fs))
        check("C3c root binds the exact panel file/tree identity",
              set(root["panel"]) == {
                  "resolved_binding", "binding_path", "binding_file_sha256"}
              and not os.path.isabs(root["panel"]["binding_path"])
              and not os.path.isabs(root["capture"]["panel_dir"]))
        check("C3d root binds the optional exact unexpected-tensor set",
              root["capture"]["unexpected_tensor_allowlist"] == {
                  "path": "inputs/unexpected-tensors.json",
                  "artifact_sha256": fixture["allow_sha256"],
                  "canonical_sorted_names_sha256": fixture["names_sha256"]})
        check("C3e broad unexpected-tensor acceptance is absent",
              "allow_unexpected_tensors" not in root["capture"])
        check("C3f root fixes the two-process qualification profile",
              root["cold_runs"] == 2
              and root["capture"]["replay_device"] == "numpy"
              and root["capture"]["replay_dtype"] == "float32"
              and root["capture"]["vocab_chunk"] == 8192
              and root["capture"]["root_protocol"]["fresh_processes"] == 2
              and root["capture"]["root_protocol"]["run_count_per_process"] == 1)
        check("C3f2 version, attempt object, verified tokenizer and bundle bind",
              root["schema"] == "fidelity-suite/job.v2"
              and isinstance(root["execution_attempt"], dict)
              and root["panel"]["resolved_binding"]["tokenizer"][
                  "files_verified"] is True
              and root["bundle"]["manifest_sha256"]
              and root["bundle_registry"]["path"] == "bin/BUNDLE.txt"
              and root["control_plane"]["schema"]
              == "fidelity-suite/control-plane-manifest.v1"
              and isinstance(root["profile"], dict)
              and isinstance(root["timing"], dict)
              and root["produced_by"]["dependencies"]["profile"]
              == root["profile"]["profile_id"]
              and root["produced_by"]["dependencies"]["lane"] == root["lane"]
              and root["produced_by"]["dependencies"]["provider"]
              == "local-container")
        unsafe_chunk = json.loads(json.dumps(root))
        unsafe_chunk["capture"]["vocab_chunk"] = 4096
        unsafe_chunk["capture"]["replay"]["vocab_chunk"] = 4096
        check("C3g non-safe replay chunk refuses",
              _refuses_job(unsafe_chunk, root_fs))
        unsafe_replay = json.loads(json.dumps(root))
        unsafe_replay["capture"]["replay_dtype"] = "float64"
        unsafe_replay["capture"]["replay"]["dtype"] = "float64"
        check("C3g2 non-safe replay dtype refuses",
              _refuses_job(unsafe_replay, root_fs))
        broad = json.loads(json.dumps(root))
        broad["capture"]["allow_unexpected_tensors"] = False
        check("C3h even a false legacy broad flag refuses",
              _refuses_job(broad, root_fs))
        partial_allow = json.loads(json.dumps(root))
        partial_allow["capture"]["unexpected_tensor_allowlist"].pop(
            "canonical_sorted_names_sha256")
        check("C3i partial exact allowlist identity refuses",
              _refuses_job(partial_allow, root_fs))
        partial_panel = json.loads(json.dumps(root))
        partial_panel["panel"].pop("binding_file_sha256")
        check("C3j partial panel identity refuses",
              _refuses_job(partial_panel, root_fs))
        escaped_panel = json.loads(json.dumps(root))
        escaped_panel["capture"]["panel_dir"] = "../panel"
        check("C3j2 panel traversal path refuses",
              _refuses_job(escaped_panel, root_fs))
        absolute_binding = json.loads(json.dumps(root))
        absolute_binding["panel"]["binding_path"] = "/tmp/panel.binding.json"
        check("C3j3 absolute panel binding path refuses",
              _refuses_job(absolute_binding, root_fs))
        raced = json.loads(json.dumps(root))
        raced["capture"]["race"] = True
        check("C3k race root refuses before stages", _refuses_job(raced, root_fs))
        unverified_tokenizer = json.loads(json.dumps(root))
        unverified_tokenizer["panel"]["resolved_binding"]["tokenizer"][
            "files_verified"] = False
        check("C3l unverified tokenizer binding refuses",
              _refuses_job(unverified_tokenizer, root_fs))
        bad_revision = json.loads(json.dumps(root))
        bad_revision["target"]["revision"] = "A" * 40
        check("C3m target revision must be exact lowercase 40-hex",
              _refuses_job(bad_revision, root_fs))
        incomplete_target = json.loads(json.dumps(root))
        incomplete_target["target"].pop("model_bytes")
        check("C3m2 incomplete target census refuses",
              _refuses_job(incomplete_target, root_fs))
        cpu_root = json.loads(json.dumps(root))
        cpu_root["capture"]["device"] = "cpu"
        check("C3m3 CPU root refuses before stages",
              _refuses_job(cpu_root, root_fs))
        bad_schema = json.loads(json.dumps(root))
        bad_schema["schema"] = "fidelity-suite/job.v1"
        check("C3n wrong job schema refuses", _refuses_job(bad_schema, root_fs))
        scalar_attempt = json.loads(json.dumps(root))
        scalar_attempt["execution_attempt"] = 1
        check("C3o scalar execution_attempt refuses",
              _refuses_job(scalar_attempt, root_fs))
        broken_bundle = json.loads(json.dumps(root))
        broken_bundle["bundle"]["manifest_sha256"] = "0" * 64
        check("C3p bundle manifest digest must verify",
              _refuses_job(broken_bundle, root_fs))
        changed_bundle = json.loads(json.dumps(root))
        changed_files = changed_bundle["bundle"]["files"]
        changed_files[0]["sha256"] = "f" * 64
        changed_bundle["bundle"] = CE.finalize_bundle_manifest(
            changed_files, changed_bundle["bundle"]["source"])
        changed_bundle["bundle_contract_sha256"] = hashlib.sha256(json.dumps(
            {"bundle": changed_bundle["bundle"],
             "registry": changed_bundle["bundle_registry"]},
            sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")).hexdigest()
        changed_bundle = CE.finalize_job(changed_bundle)
        check("C3q exact bundle bytes participate in job identity",
              changed_bundle["job_id_full"] != root["job_id_full"])
        import measure_cloud                               # noqa: E402
        cloud_bundle = measure_cloud._bundle_manifest()
        container_bundle = CE.exact_bundle_manifest(SUITE, {})
        cloud_shaped = json.loads(json.dumps(root))
        cloud_shaped["recipe"] = "cloud"
        cloud_shaped["bundle"] = cloud_bundle
        cloud_shaped["bundle_contract_sha256"] = hashlib.sha256(json.dumps(
            {"bundle": cloud_shaped["bundle"],
             "registry": cloud_shaped["bundle_registry"]},
            sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf-8")).hexdigest()
        cloud_shaped = CE.finalize_job(cloud_shaped)
        check("C3q2 cloud/container bundle builders share one exact-file verifier",
              CE.verify_bundle_manifest(cloud_bundle)
              == cloud_bundle["manifest_sha256"]
              and CE.verify_bundle_manifest(container_bundle)
              == container_bundle["manifest_sha256"]
              and all(set(row) == {"path", "bytes", "sha256"}
                      for row in cloud_bundle["files"] + container_bundle["files"]))
        check("C3q3 cloud-shaped and container-built jobs use one job verifier",
              CE.verify_job(cloud_shaped) == cloud_shaped["job_id_full"]
              and CE.verify_job(root) == root["job_id_full"])

        bad_quant = argparse.Namespace(**{
            **vars(quant_args), "profile": None})
        try:
            CE.job_document(bad_quant, SUITE, quant_fs, quiet)
            check("C3r a quant profile is explicit, never guessed", False)
        except CE.Refusal:
            check("C3r a quant profile is explicit, never guessed", True)
        missing_quant_scope = argparse.Namespace(**{
            **vars(quant_args), "scope_json": None})
        try:
            CE.job_document(missing_quant_scope, SUITE, quant_fs, quiet)
            check("C3s quant scope must resolve before stages", False)
        except CE.Refusal:
            check("C3s quant scope must resolve before stages", True)


# --------------------------------------------------------------------------
# C4  the token
# --------------------------------------------------------------------------

def rung_token():
    print("[C4] the token is a 0600 file and never an environment a stage sees")
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        src = fs / "tok"
        src.write_text("hf_TESTONLYNOTAREALTOKEN\n", encoding="utf-8")
        wrote = CE.write_token(fs, str(src), lambda *_a, **_k: None)
        dest = fs / ".secrets" / "hf_token"
        check("C4a written where stage_measure.sh load_token reads it",
              wrote and dest.is_file())
        check("C4b mode 0600", oct(dest.stat().st_mode & 0o777) == "0o600")
        check("C4c the directory is 0700",
              oct((fs / ".secrets").stat().st_mode & 0o777) == "0o700")
        check("C4d no trailing newline smuggled into the token",
              dest.read_text(encoding="utf-8") == "hf_TESTONLYNOTAREALTOKEN")

        os.environ["HF_TOKEN"] = "hf_ANOTHERTESTVALUE"
        try:
            env = CE.stage_env(fs, Path(td), {"image_digest": None,
                                              "image_content_sha256": None})
            check("C4e HF_TOKEN is dropped from the stage environment",
                  "HF_TOKEN" not in env)
        finally:
            os.environ.pop("HF_TOKEN", None)
        check("C4f the roots are exported, never left to a /home/jl_fs default",
              env["FIDELITY_FS_ROOT"] == str(fs)
              and env["QP_PIPELINE_ROOT"].endswith("/pipeline"))
        # The image and the stage scripts inside it ship together, so the
        # container emits ONLY the current spelling. The deprecated one is
        # still READ by those scripts (and still exported by the SSH
        # controller, where a controller and an instance can come from
        # different checkouts) -- but baking it into new surface just creates
        # a migration nobody needs.
        check("C4g the engine root is exported under its current name",
              env["FIDELITY_ENGINE_ROOT"] == str(td))
        check("C4g2 ... and the deprecated spelling is not emitted, even when "
              "it was in the caller's environment",
              "FIDELITY_K6_ROOT" not in env)
        check("C4g3 no-token invocation shreds a stale persisted token",
              not CE.write_token(fs, None, lambda *_a, **_k: None)
              and not dest.exists())
        (fs / ".secrets").rmdir()
        outside = fs / "outside-secret"
        outside.mkdir()
        sentinel = outside / "hf_token"
        sentinel.write_text("do-not-touch", encoding="utf-8")
        (fs / ".secrets").symlink_to(outside, target_is_directory=True)
        CE.clear_stale_token(fs, lambda *_a, **_k: None)
        check("C4g4 stale secret-directory symlink is unlinked, not followed",
              not (fs / ".secrets").exists()
              and sentinel.read_text(encoding="utf-8") == "do-not-touch")
    # The DEFAULTS are the thing worth testing, not the values a caller passed:
    # a root that names a model or a campaign is how `/home/jl_fs/glm53-k6`
    # ended up baked into a path on rented hardware, and a root that resolves
    # to nothing is a run written into a container's ephemeral layer.
    defaults = [CE.DEFAULT_FS_ROOT, str(CE.IMAGE_ROOT)]
    check("C4h the container's own default roots name no model or campaign",
          not any(tok in d.lower() for d in defaults
                  for tok in ("glm", "k6", "jl_fs")), "%s" % defaults)
    check("C4i the default run root is a mount point, not an image directory",
          CE.DEFAULT_FS_ROOT.startswith("/workspace"))


# --------------------------------------------------------------------------
# C5/C6  what lands on the machine
# --------------------------------------------------------------------------

def _ignore_patterns():
    out = []
    for line in (SUITE / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        out.append((negate, line[1:] if negate else line))
    return out


def _matches(pattern: str, path: str) -> bool:
    import fnmatch
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return any(fnmatch.fnmatchcase(seg, tail) for seg in path.split("/"))
    p, q = pattern.split("/"), path.split("/")
    if len(p) != len(q):
        return False
    return all(fnmatch.fnmatchcase(b, a) for a, b in zip(p, q))


def dockerignored(path: str) -> bool:
    """Would `docker build` drop this path from the context?

    A deliberately small model of the real matcher, covering the pattern forms
    this .dockerignore actually uses: walk the path's ancestors outermost
    first, and for each one let the LAST matching pattern decide -- which is
    Docker's own last-match-wins rule, and is why an exception must come after
    the exclusion it reopens.  If a pattern form outside this subset is ever
    added, this model stops describing the file and the rung below is the
    thing that should be made stricter, never deleted.
    """
    parts = path.split("/")
    excluded = False
    for depth in range(1, len(parts) + 1):
        prefix = "/".join(parts[:depth])
        for negate, pattern in _ignore_patterns():
            if _matches(pattern, prefix):
                excluded = not negate
    return excluded


def rung_bundle():
    print("[C5] the image ships bin/BUNDLE.txt's audited set")
    listed = set(CE.bundle_entries(SUITE))
    check("C5a the list parses and is not empty", len(listed) > 20)
    check("C5b nothing under .secrets/ is ever in it",
          not any(".secrets" in e for e in listed))
    with tempfile.TemporaryDirectory() as td:
        fs = Path(td)
        logged = []
        copied = CE.sync_suite(SUITE, fs, logged.append)
        check("C5c a cold run root receives every present entry", copied > 20)
        check("C5d the entrypoint and every direct contract module land too",
              all((fs / rel).is_file() for rel in (
                  "bin/container_entry.py", "bin/fidelity/jobcontract.py",
                  "bin/fidelity/panel.py", "bin/fidelity/resultsink.py",
                  "bin/fidelity/stages.py")))
        check("C5d2 strict image prune keeps the entrypoint and stage owner",
              {"bin/container_entry.py", "bin/fidelity/stages.py"}.issubset(listed))
        check("C5e an absent bundle entry is LOGGED, never silent",
              all(("skipped" in line) for line in logged) or not logged)
        again = CE.sync_suite(SUITE, fs, logged.append)
        check("C5f a second sync copies nothing (digest-compared, resumable)",
              again == 0)
        # The .dockerignore is an exclusion list, and an over-eager exclusion
        # does not fail the build: it produces an image that dies in the
        # `setup` stage on a rented box, which is exactly how a MiniMax root
        # capture once died on GGUF test data that was never bundled.
        excluded = [rel for rel in listed if dockerignored(rel)]
        check("C5g the .dockerignore excludes NO bundled file",
              not excluded, "would be missing from the image: %s" % excluded[:5])
        check("C5h ... while still dropping the 187 MB evidence tree",
              dockerignored("engines/tools/dione-evidence/index-q4.json")
              and not dockerignored("engines/tools/dione-evidence/bf16-index.json"))
        # Found on a real box, not reasoned about: `fidelity_dataset.py
        # capture` ends in a postcondition that validates the manifest it just
        # wrote, and dsvalidate reads docs/schema/ through _minischema.Registry
        # -- which os.listdirs the DIRECTORY, so an absent one is
        # FileNotFoundError rather than a skipped check. A containerised root
        # capture died there after the bootstrap, the fetch and the capture
        # itself were all paid for. A bundled script's DATA is a dependency.
        from fidelity import dsvalidate as DV
        schema_rel = os.path.relpath(DV.SCHEMA_DIR, str(SUITE))
        staged = sorted((fs / schema_rel).glob("*.json")) if (fs / schema_rel).is_dir() else []
        check("C5j the capture stage's own validator has its schemas on a "
              "bundle-only tree", len(staged) >= 1,
              "%s holds nothing; dsvalidate os.listdirs it" % (fs / schema_rel))
        check("C5k ... and the .dockerignore lets them into the image",
              not any(dockerignored(rel) for rel in listed
                      if rel.startswith(schema_rel + "/")))
        # C5l  A COMMITTED PANEL IS DATA THE ENTRYPOINT IS POINTED AT.
        # --require-all proves every BUNDLE.txt entry arrived in the stage. It
        # cannot prove the converse -- that what a container-native capture
        # NEEDS is listed -- and the two are not the same check. The committed
        # panels landed under engines/panels/ in one commit and in BUNDLE.txt
        # two commits later, so container_prune correctly stripped them from
        # the image built in between, and that image refused its own committed
        # panel on a rented L4: "--panel-dir ... has no panel.json". Four
        # minutes and $0.003, but only because the refusal is early; the same
        # omission behind a fetch stage is an hour of GPU. Found by renting;
        # this is the static form of the same question.
        panels_dir = SUITE / "engines" / "panels"
        panels = sorted(d for d in panels_dir.iterdir() if d.is_dir()) \
            if panels_dir.is_dir() else []
        check("C5l there is at least one committed panel to check", bool(panels))
        for panel in panels:
            staged_panel = fs / "engines" / "panels" / panel.name
            arrays = sorted((staged_panel / "arrays").glob("*")) \
                if (staged_panel / "arrays").is_dir() else []
            check("C5l %s survives the prune with panel.json + arrays/"
                  % panel.name,
                  (staged_panel / "panel.json").is_file() and bool(arrays),
                  "a container-native --panel-dir would refuse it; "
                  "add it to bin/BUNDLE.txt")
        check("C5i ... and the 21 MB bundle.tar.gz and the venv",
              dockerignored("bundle.tar.gz") and dockerignored(".venv/bin/python")
              and dockerignored("bin/__pycache__/measure_cloud.cpython-312.pyc"))
    with tempfile.TemporaryDirectory() as td, \
            tempfile.TemporaryDirectory() as outside_td:
        fs = Path(td)
        outside = Path(outside_td)
        (fs / "bin").symlink_to(outside, target_is_directory=True)
        try:
            CE.sync_suite(SUITE, fs, lambda *_a, **_k: None)
        except CE.Refusal:
            refused = True
        else:
            refused = False
        check("C5f2 suite sync refuses a planted destination parent symlink",
              refused and not any(outside.iterdir()))

    print("[C6] container_prune keeps exactly that set")
    with tempfile.TemporaryDirectory() as td:
        stage, out = Path(td) / "stage", Path(td) / "out"
        (stage / "bin" / "fidelity").mkdir(parents=True)
        (stage / "engines" / "tools" / "dione-evidence").mkdir(parents=True)
        (stage / "bin" / "BUNDLE.txt").write_text(
            "# a comment\nbin/stage_measure.sh\nengines/tools/progress.py\n"
            "engines/tools/absent_engine.py\n", encoding="utf-8")
        (stage / "bin" / "stage_measure.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (stage / "engines" / "tools" / "progress.py").write_text("x = 1\n", encoding="utf-8")
        (stage / "engines" / "tools" / "dione-evidence" / "big.bin").write_text(
            "y" * 1000, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(HERE / "container_prune.py"),
             "--stage", str(stage), "--out", str(out)],
            capture_output=True, text=True)
        kept = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        check("C6a exit 0", proc.returncode == 0, proc.stderr[-300:])
        check("C6b only listed files are kept",
              kept == ["bin/BUNDLE.txt", "bin/stage_measure.sh",
                       "engines/tools/progress.py"], "%s" % kept)
        check("C6c an absent entry is reported, not silently dropped",
              "absent_engine.py" in proc.stdout)
        # Fail-open is right for the SSH uploader (a lane whose engine is not
        # in this checkout must not break the upload) and wrong for a build,
        # where the only way an entry goes missing is a COPY the Dockerfile
        # does not make. That shipped an image which died validating the
        # manifest it had just written, on rented hardware.
        strict = subprocess.run(
            [sys.executable, str(HERE / "container_prune.py"),
             "--stage", str(stage), "--out", str(Path(td) / "out2"),
             "--require-all"], capture_output=True, text=True)
        check("C6e --require-all REFUSES the same tree (exit 3)",
              strict.returncode == 3, "rc=%s" % strict.returncode)
        check("C6f ... and names every entry that did not arrive",
              "absent_engine.py" in strict.stderr
              and "did not COPY" in strict.stderr)
        check("C6d the stage script stays executable",
              os.access(str(out / "bin" / "stage_measure.sh"), os.X_OK))


# --------------------------------------------------------------------------
# C7  image identity is observed, never guessed
# --------------------------------------------------------------------------

def rung_pin():
    print("[C7] the image digest is observed or null-with-a-reason")
    saved_root, saved_env = CE.IMAGE_ROOT, os.environ.get(CE.IMAGE_PIN_ENV)
    os.environ.pop(CE.IMAGE_PIN_ENV, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            CE.IMAGE_ROOT = Path(td)
            pin = CE.image_pin(None)
            check("C7a nothing to observe -> null", pin["image_digest"] is None)
            check("C7b ... and the reason names both remedies",
                  CE.IMAGE_PIN_ENV in pin["source"] and "image-pin" in pin["source"])
            (CE.IMAGE_ROOT / CE.IMAGE_PIN_FILE).write_text("f" * 64 + "\n",
                                                           encoding="utf-8")
            check("C7c the pin file is read (docker load strips the digest)",
                  CE.image_pin(None)["image_digest"] == "f" * 64)
            os.environ[CE.IMAGE_PIN_ENV] = "e" * 64
            check("C7d the environment beats the baked file",
                  CE.image_pin(None)["image_digest"] == "e" * 64)
            check("C7e --image-pin beats both (it is what the launcher pulled)",
                  CE.image_pin("d" * 64)["image_digest"] == "d" * 64)
    finally:
        CE.IMAGE_ROOT = saved_root
        os.environ.pop(CE.IMAGE_PIN_ENV, None)
        if saved_env is not None:
            os.environ[CE.IMAGE_PIN_ENV] = saved_env


# --------------------------------------------------------------------------
# C8  the acceptance test, as an invariant
# --------------------------------------------------------------------------

def rung_capture_identity():
    print("[C8] recording the container must not move what the container ran")
    sys.path.insert(0, str(SUITE / "engines" / "tools"))
    try:
        import hf_capture                                  # noqa: E402
    except Exception as exc:                               # noqa: BLE001
        print("  SKIP  C8 (hf_capture needs torch: %s)" % type(exc).__name__)
        return

    saved = os.environ.get("STACKPRINT_IMAGE_PIN")
    os.environ.pop("STACKPRINT_IMAGE_PIN", None)
    os.environ["FIDELITY_IMAGE_PIN_FILE"] = "/nonexistent/image-pin.txt"
    try:
        check("C8a no pin -> None, so capture_runtime keeps its old default",
              hf_capture._container_identity() is None)
        fingerprint = {"schema": "malaiwah.stack-fingerprint.v1",
                       "engine": "transformers-eager", "torch_version": "2.11.0",
                       "device": "cuda", "device_name": "A100"}
        weights = {"repository": "x/y", "revision": "a" * 40}
        base = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights,
            container=hf_capture._container_identity())
        legacy = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights)
        check("C8b an un-pinned capture's runtime receipt is byte-identical "
              "to what it was before this field learned to be filled",
              json.dumps(base, sort_keys=True) == json.dumps(legacy, sort_keys=True))

        os.environ["STACKPRINT_IMAGE_PIN"] = "a" * 64
        pinned = dsmanifest.capture_runtime(
            lane="streaming", stack_fingerprint=fingerprint,
            stack_fingerprint_sha256="s" * 64, weights=weights,
            container=hf_capture._container_identity())
        check("C8c a pinned capture records the image",
              pinned["container"]["image_digest"] == "a" * 64)
        check("C8d ... and does NOT move stack_fingerprint_sha256, which is "
              "what dscompare reads to decide stack_relation",
              pinned["stack_fingerprint_sha256"] == base["stack_fingerprint_sha256"]
              and pinned["stack_fingerprint"] == base["stack_fingerprint"])
        check("C8e the container block is not an input to that fingerprint",
              "container" not in json.dumps(pinned["stack_fingerprint"]))
    finally:
        os.environ.pop("STACKPRINT_IMAGE_PIN", None)
        os.environ.pop("FIDELITY_IMAGE_PIN_FILE", None)
        if saved is not None:
            os.environ["STACKPRINT_IMAGE_PIN"] = saved


# --------------------------------------------------------------------------
# C9  the image runs the specification, it does not paraphrase it
# --------------------------------------------------------------------------

def rung_dockerfile():
    print("[C9] the Dockerfile installs nothing bootstrap_measure.sh owns")
    text = (SUITE / "container" / "Dockerfile").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    body = "\n".join(lines)
    check("C9a it runs the bootstrap rather than repeating it",
          "bootstrap_measure.sh" in body)
    for forbidden in ("pip install torch", "pip3 install torch", "torch==",
                      "transformers==", "python -m venv", "deadsnakes"):
        check("C9b the recipe is not duplicated: no %r" % forbidden,
              forbidden not in body)
    check("C9c the run root is a mount, not a layer",
          "VOLUME" in body and "/workspace" in body)
    check("C9d no credential is baked",
          not any(k in body for k in ("HF_TOKEN", "RUNPOD", "hf_", "API_KEY")))
    check("C9e the entrypoint is the CLI",
          "container_entry.py" in body and "ENTRYPOINT" in body)

    # The general rule, not the instance: every top-level directory BUNDLE.txt
    # draws from has to be COPYed into the build stage. docs/schema/ was in the
    # list and in no COPY line, and nothing anywhere said so.
    copied = set()
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "COPY":
            copied.add(parts[1].rstrip("/"))
    needed = sorted({rel.split("/")[0] if "/" not in rel.rstrip("/")
                     else "/".join(rel.split("/")[:2])
                     for rel in CE.bundle_entries(SUITE)})
    uncopied = [d for d in needed
                if not any(d == c or d.startswith(c + "/") or c.startswith(d + "/")
                           for c in copied)]
    check("C9j every directory BUNDLE.txt draws from is COPYed by the build",
          not uncopied, "not in any COPY: %s  (COPY has: %s)"
          % (uncopied, sorted(copied)))
    check("C9k the build refuses a bundle entry that did not arrive",
          "--require-all" in body)

    boot = (SUITE / "bin" / "bootstrap_measure.sh").read_text(encoding="utf-8")
    lock_text = (
        SUITE / "bin" / "requirements-cu130-py312.lock"
    ).read_text(encoding="utf-8")
    lock_lines = lock_text.splitlines()
    locked = {}
    malformed_lock_lines = []
    consumed = set()
    for index, line in enumerate(lock_lines):
        if not line or line.startswith("#"):
            consumed.add(index)
            continue
        if " @ " in line and line.endswith(" \\") and index + 1 < len(lock_lines):
            name, url = line[:-2].split(" @ ", 1)
            hash_line = lock_lines[index + 1]
            match = re.fullmatch(r"    --hash=sha256:([0-9a-f]{64})", hash_line)
            if (re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                    and url.startswith("https://") and match
                    and name.lower().replace("_", "-") not in locked):
                locked[name.lower().replace("_", "-")] = (
                    url, match.group(1))
                consumed.update((index, index + 1))
                continue
        if index not in consumed:
            malformed_lock_lines.append((index + 1, line))
    malformed_lock_lines.extend(
        (index + 1, line) for index, line in enumerate(lock_lines)
        if index not in consumed
        and not any(row[0] == index + 1 for row in malformed_lock_lines))
    guard = boot.find("FIDELITY_BOOTSTRAP_INSTALL_ONLY")
    first_check = boot.find("selftest_tr3_offline.py")
    check("C9f install-only stops BEFORE the pre-flight batteries",
          0 < guard < first_check)
    check("C9g install-only leaves the exact hashed wheel closure intact",
          0 < boot.find("exact hashed wheel closure") < guard)
    proc = subprocess.run(["bash", "-n", str(SUITE / "bin" / "bootstrap_measure.sh")],
                          capture_output=True, text=True)
    check("C9h the edited bootstrap still parses", proc.returncode == 0,
          proc.stderr[-300:])
    proc = subprocess.run(["bash", "-n", str(SUITE / "container" / "build.sh")],
                          capture_output=True, text=True)
    check("C9i build.sh parses", proc.returncode == 0, proc.stderr[-300:])

    # Exercise the exact validator with importable fake distributions.  This is
    # deliberately not a text-only "there is a == somewhere" assertion: the
    # regression was a wrong template wheel satisfying an import-only guard.
    begin = boot.index("# DIRECT_WHEEL_VALIDATOR_BEGIN")
    end = boot.index("# DIRECT_WHEEL_VALIDATOR_END")
    validator = boot[begin:end].split("\n", 1)[1]
    validator_tree = ast.parse(validator)
    expected = None
    for node in validator_tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "expected"
                        for target in node.targets)):
            expected = ast.literal_eval(node.value)
            break
    guarded = {
        "pip", "setuptools", "wheel", "ninja", "packaging",
        "torch", "transformers", "safetensors", "numpy",
        "huggingface-hub", "hf-transfer", "accelerate", "rich",
        "tokenizers", "Pillow", "pydantic", "formatron", "kbnf",
    }
    check("C9l the exact validator names every direct distribution",
          expected is not None and set(expected) == guarded)
    check("C9l2 every guarded direct distribution is in the same exact lock",
          expected is not None
          and all(
              name.lower().replace("_", "-") in locked
              and version in urllib.parse.unquote(
                  locked[name.lower().replace("_", "-")][0])
              for name, version in expected.items()),
          locked)
    check("C9l3 the wheel closure is closed, HTTPS-only, and fully hashed",
          len(locked) == 72 and not malformed_lock_lines
          and all(len(digest) == 64 for _url, digest in locked.values()),
          (len(locked), malformed_lock_lines))
    check("C9l4 bootstrap permits no resolver-selected or unhashed wheel",
          "--no-deps --require-hashes --only-binary=:all:" in boot
          and '-r "$WHEEL_LOCK"' in boot
          and '"$FLASH_ATTN_WHL#sha256=$FLASH_ATTN_SHA256"' in boot
          and 'cuda-keyring_1.1-1_all.deb | sha256sum -c -' in boot
          and boot.count("validate_direct_wheels | tee") == 1
          and '"$PY" -m pip check' in boot)
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td)
        (fake / "sitecustomize.py").write_text(
            "import importlib.metadata as metadata, json, os\n"
            "_versions = json.loads(os.environ['FAKE_VERSIONS'])\n"
            "metadata.version = lambda name: _versions[name]\n",
            encoding="utf-8")
        (fake / "torch.py").write_text(
            "import os\n"
            "class _Version:\n"
            "    cuda = os.environ['FAKE_CUDA']\n"
            "version = _Version()\n",
            encoding="utf-8")
        validator_env = dict(
            os.environ,
            PYTHONPATH=str(fake),
            FAKE_VERSIONS=json.dumps(expected or {}),
            FAKE_CUDA="13.0")
        exact = subprocess.run(
            [sys.executable, "-c", validator],
            capture_output=True, text=True, env=validator_env)
        check("C9m the named exact direct set and CUDA 13.0 are accepted",
              exact.returncode == 0, exact.stderr[-300:])

        wrong_versions = dict(expected or {})
        wrong_versions["transformers"] = "0.0.0"
        validator_env["FAKE_VERSIONS"] = json.dumps(wrong_versions)
        wrong = subprocess.run(
            [sys.executable, "-c", validator],
            capture_output=True, text=True, env=validator_env)
        check("C9n an importable wrong direct distribution is refused by name",
              wrong.returncode != 0
              and "transformers" in wrong.stderr
              and "5.16.1" in wrong.stderr,
              wrong.stderr[-300:])

        validator_env["FAKE_VERSIONS"] = json.dumps(expected or {})
        validator_env["FAKE_CUDA"] = "12.8"
        wrong_cuda = subprocess.run(
            [sys.executable, "-c", validator],
            capture_output=True, text=True, env=validator_env)
        check("C9o a torch wheel built for the wrong CUDA is refused",
              wrong_cuda.returncode != 0
              and "CUDA" in wrong_cuda.stderr
              and "13.0" in wrong_cuda.stderr,
              wrong_cuda.stderr[-300:])

    pipeline = boot[boot.index("# ---- 3."):
                    boot.index("# ---- 4.")]
    checkout = pipeline.find('checkout -q --detach -f "$pin"')
    reset = pipeline.find('reset -q --hard "$pin"')
    clean = pipeline.find("clean -q -ffdx")
    apply = pipeline.find("patch -p1 -s --fuzz=0")
    check("C9p every pipeline run resets the exact pin and removes ignored residue "
          "before applying patches",
          0 < checkout < reset < clean < apply)
    check("C9p2 reconstruction never cleans a symlink or nested/outer worktree",
          '[ ! -L "$destination" ]' in pipeline
          and "rev-parse --show-toplevel" in pipeline
          and 'top_real="$(realpath "$top"' in pipeline
          and '[ "$top_real" = "$destination_real" ]' in pipeline
          and '"$PIPE"|"$EXL3"' in pipeline)
    check("C9q a marker can never bypass pipeline reconstruction",
          "_STORED_BITS" not in pipeline
          and ('"$PIPE_REPO" "$PIPE_PIN" "$PIPE_TREE" '
               '"$PIPE_ARCHIVE_SHA256"') in pipeline)
    check("C9r the complete measurement patch selection is cardinality-checked",
          '${#_series[@]}" -eq 7' in pipeline
          and '${#_files[@]}" -eq "${#_series[@]}' in pipeline)

    exl3 = boot[boot.index("# ---- 4."):
                boot.index("# ---- INSTALL/CHECK SPLIT")]
    check("C9s an imported exllamav3 must be the exact reconstructed checkout",
          ('"$EXL3_REPO" "$EXL3_PIN" "$EXL3_TREE" '
           '"$EXL3_ARCHIVE_SHA256"') in exl3
          and 'source != expected' in exl3
          and 'resolved.relative_to(expected)' in exl3
          and "git -C \"$EXL3\" diff --quiet" in exl3)
    check("C9t reinstalling direct wheels forces the editable extension rebuild",
          '_direct_wheels_reinstalled" -eq 1' in exl3
          and "--force-reinstall --no-build-isolation --no-deps -e ." in exl3)
    check("C9u the torch2.10-tagged flash wheel proves torch2.11 compatibility "
          "with a runtime CUDA kernel outside install-only builds",
          "flash_attn_func(q, q, q" in exl3
          and "torch.cuda.synchronize()" in exl3
          and "FLASH_ATTN_INSTALL_ONLY" in exl3)


# --------------------------------------------------------------------------
# C10  the CLI itself
# --------------------------------------------------------------------------

def rung_cli():
    print("[C10] exact root argv/refusals and stubbed stage behavior")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        fixture = _root_fixture(work)
        root_target_path, root_target = _target_descriptor(
            work, "root",
            repo_id="malaiwah/GLM-5.2-SIQ-Fruit-bf16",
            revision="ef68013aa6e16453cf52b5b77647f72fbe258c3c",
            surface="native-bf16", codec="bf16", bits=16.0,
            model_bytes=10102776813,
            config_sha256="5a19697e555fff140d1b089b852c3ef227114b196f8d76796560feeeb34dc44a",
            index_sha256="86e6cc1d8548c7bdbbc117e93b85b8ae249f446de9b48d2195e51f358674ba56")
        fs = work / "dry"
        root_argv = [
            "capture", "--fs-root", str(fs),
            "--engine-root", str(work / "engine"),
            "--model", root_target["repo_id"],
            "--revision", root_target["revision"],
            "--target-descriptor", str(root_target_path),
            "--panel-dir", str(fixture["panel_dir"]),
            "--gpu", "L4",
            "--panel-binding", str(fixture["binding_file"]),
            "--panel-binding-sha256", fixture["binding_sha256"],
            "--panel-tokenizer-root", str(fixture["tokenizer_root"]),
            "--dataset-id", "fidelity--t.malaiwah.root.bf16",
            "--dataset-repository", "someone/root-dataset",
            "--replay-device", "numpy",
            "--replay-dtype", "float32", "--vocab-chunk", "8192",
            "--unexpected-tensors-allowlist", str(fixture["allow_file"]),
            "--unexpected-tensors-allowlist-sha256", fixture["allow_sha256"],
            "--unexpected-tensors-name-sha256", fixture["names_sha256"],
            "--workspace-available-bytes-minimum", str(20 * 1024 ** 3),
            "--container-available-bytes-minimum", str(1024 ** 3),
            "--expected-vram-bytes", str(24 * 1024 ** 3),
        ]
        out = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + root_argv + ["--dry-run"],
            capture_output=True, text=True)
        check("C10a exact root --dry-run exits 0",
              out.returncode == 0, out.stderr[-400:])
        check("C10b --dry-run creates no job.json",
              not (fs / "job.json").is_file())
        expected = (
            "setup fetch_target capture verify capture_repeat verify_repeat "
            "compare_root qualify_root")
        check("C10c exact root prints the complete two-process sequence",
              expected in out.stdout)
        doc = json.loads(out.stdout[out.stdout.index("{"):
                                    out.stdout.rindex("}") + 1])
        check("C10d printed job is finalized and exact",
              CE.verify_job(doc) == doc["job_id_full"]
              and doc["cold_runs"] == 2
              and doc["capture"]["dataset_repository"] == "someone/root-dataset"
              and doc["capture"]["publish_root_to"] is None
              and doc["capture"]["device"] == "cuda")
        publish_argv = list(root_argv)
        publish_argv[publish_argv.index(str(fs))] = str(work / "publish-refused")
        publish_argv += [
            "--publish-root-to", "someone/root-dataset", "--dry-run"]
        publish_out = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")] + publish_argv,
            capture_output=True, text=True)
        check("C10d2 local publication explicitly refuses before stages",
              publish_out.returncode == CE.EXIT_REFUSED
              and "unsupported" in publish_out.stderr
              and "stages:" not in publish_out.stdout)

        no_tokenizer = list(root_argv)
        no_tokenizer[no_tokenizer.index(str(fs))] = str(work / "no-tokenizer")
        token_index = no_tokenizer.index("--panel-tokenizer-root")
        del no_tokenizer[token_index:token_index + 2]
        unverified = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + no_tokenizer + ["--dry-run"],
            capture_output=True, text=True)
        check("C10e2 absent tokenizer root refuses unverified binding",
              unverified.returncode == 3
              and "tokenizer" in unverified.stderr.lower())
        partial_argv = list(root_argv)
        partial_argv[partial_argv.index(str(fs))] = str(work / "partial")
        digest_index = partial_argv.index(
            "--unexpected-tensors-name-sha256")
        del partial_argv[digest_index:digest_index + 2]
        partial = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + partial_argv + ["--dry-run"],
            capture_output=True, text=True)
        check("C10e partial allowlist identity refuses before stages",
              partial.returncode == 3 and "all-or-none" in partial.stderr)
        raced_argv = list(root_argv)
        raced_argv[raced_argv.index(str(fs))] = str(work / "raced")
        raced = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + raced_argv + ["--race", "--dry-run"],
            capture_output=True, text=True)
        check("C10f race refuses and names the non-orchestrating boundary",
              raced.returncode == 3
              and "does not create or manage RunPod" in raced.stderr)
        broad_argv = list(root_argv)
        broad_argv[broad_argv.index(str(fs))] = str(work / "broad")
        broad = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + broad_argv + ["--allow-unexpected-tensors"],
            capture_output=True, text=True)
        check("C10g obsolete broad allow flag is not a CLI surface",
              broad.returncode != 0
              and "unrecognized arguments" in broad.stderr)
        help_out = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py"), "--help"],
            capture_output=True, text=True)
        check("C10h help refuses any paid native-container implication",
              "LOCAL DRIVER ONLY" in help_out.stdout
              and "RunPod resources" in help_out.stdout
              and "not an approved paid route" in help_out.stdout)
        entry_body = (HERE / "container_entry.py").read_text(encoding="utf-8")
        check("C10h2 local driver has no provider API client",
              "import runpod" not in entry_body.lower()
              and "from runpod" not in entry_body.lower())

        stage = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py"), "stage", "measure",
             "--fs-root", str(work / "empty")],
            capture_output=True, text=True)
        check("C10i a stage with no job document refuses locally",
              stage.returncode == 3 and "job" in stage.stderr.lower())
        outside_root = work / "outside-root"
        outside_root.mkdir()
        linked_root = work / "linked-root"
        linked_root.symlink_to(outside_root, target_is_directory=True)
        symlink_argv = list(root_argv)
        symlink_argv[symlink_argv.index(str(fs))] = str(linked_root)
        linked = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + symlink_argv + ["--dry-run"],
            capture_output=True, text=True)
        check("C10i2 a symlink run root refuses before writing outside it",
              linked.returncode == CE.EXIT_REFUSED
              and not any(outside_root.iterdir()))
        stale_root = work / "stale-root"
        (stale_root / "receipts" / "done").mkdir(parents=True)
        stale_marker = stale_root / "receipts" / "done" / "setup.done"
        stale_marker.write_text("old-attempt", encoding="utf-8")
        stale_argv = list(root_argv)
        stale_argv[stale_argv.index(str(fs))] = str(stale_root)
        stale = subprocess.run(
            [sys.executable, str(HERE / "container_entry.py")]
            + stale_argv + ["--dry-run"],
            capture_output=True, text=True)
        check("C10i3 new jobs refuse stale attempt markers before sync/stage",
              stale.returncode == CE.EXIT_REFUSED
              and stale_marker.read_text(encoding="utf-8") == "old-attempt"
              and not (stale_root / "bin").exists())

        original_run = CE.run_stage
        original_summary = CE.RS.build_summary
        original_deliver = CE.RS.deliver
        original_doctor = CE.cmd_doctor
        original_accelerator = CE.require_accelerator
        seen_stages, seen_status = [], []
        try:
            def stub_run(name, _fs, _env, _con):
                seen_stages.append(name)
                return 0

            def stub_summary(_fs, _verb, status, stage_names, _pin,
                             failed_stage=None):
                seen_status.append((status, failed_stage))
                return {"status": status, "stages": list(stage_names)}

            def stub_deliver(_fs, sinks, _summary, _con):
                return [{"scheme": sink.scheme, "ok": True} for sink in sinks]

            CE.run_stage = stub_run
            CE.RS.build_summary = stub_summary
            CE.require_accelerator = lambda *_a, **_k: None
            CE.RS.deliver = stub_deliver
            execute_argv = list(root_argv)
            execute_argv[execute_argv.index(str(fs))] = str(work / "execute")
            code = CE.main(execute_argv)
            check("C10j behavioral stub runs the exact root sequence once",
                  code == CE.EXIT_OK and " ".join(seen_stages) == expected)
            check("C10k successful stub builds/delivers an ok result",
                  seen_status[-1] == ("ok", None))

            CE.run_stage = lambda name, _fs, _env, _con: 9
            failed_argv = list(root_argv)
            failed_argv[failed_argv.index(str(fs))] = str(work / "failed")
            failed_code = CE.main(failed_argv)
            check("C10l failed stub still builds/delivers a failed result",
                  failed_code == CE.EXIT_FAILED
                  and seen_status[-1] == ("failed", "setup"))

            CE.run_stage = stub_run
            CE.RS.deliver = lambda _fs, sinks, _summary, _con: [
                {"scheme": sink.scheme,
                 "ok": sink.scheme == "stdout"} for sink in sinks]
            sink_argv = list(root_argv)
            sink_argv[sink_argv.index(str(fs))] = str(work / "sink-failed")
            sink_argv += ["--result-sink", "file:%s" % (work / "answer.tar.gz")]
            sink_code = CE.main(sink_argv)
            check("C10m an explicitly requested failed sink gates success",
                  sink_code == CE.EXIT_FAILED)
            CE.cmd_doctor = lambda _con: CE.EXIT_OK
            doctor_code = CE.main([
                "doctor", "--fs-root", str(work / "doctor-sink-failed"),
                "--result-sink", "file:%s" % (work / "doctor.tar.gz")])
            check("C10m2 requested doctor result-sink failure gates success",
                  doctor_code == CE.EXIT_FAILED)
        finally:
            CE.run_stage = original_run
            CE.RS.build_summary = original_summary
            CE.RS.deliver = original_deliver
            CE.cmd_doctor = original_doctor
            CE.require_accelerator = original_accelerator


# --------------------------------------------------------------------------
# C11  the release path: decided in a script, tested offline, default-off
# --------------------------------------------------------------------------

def rung_accelerator():
    """C12  a rented box whose CUDA does not work refuses BEFORE the fetch."""
    import container_entry as CE
    print("[C12] a dead accelerator is refused before anything is fetched")

    def venv(tmp, body):
        root = Path(tmp)
        (root / "venv" / "bin").mkdir(parents=True)
        py = root / "venv" / "bin" / "python"
        py.write_text("#!/bin/sh\n" + body + "\n")
        py.chmod(0o755)
        return root

    def quiet(_text):
        return None

    with tempfile.TemporaryDirectory() as tmp:
        root = venv(tmp, "echo '{\"ok\": true, \"torch\": \"2.11.0+cu130\","
                         " \"built\": \"13.0\", \"name\": \"NVIDIA L4\"}'")
        CE.require_accelerator({"capture": {"device": "cuda"}}, root, quiet)
        check("C12a a box with a usable CUDA device proceeds", True)

    with tempfile.TemporaryDirectory() as tmp:
        # Verbatim from the field: RunPod L4, driver 12040, image pinned cu130.
        root = venv(tmp, "echo 'UserWarning: The NVIDIA driver on your system"
                         " is too old (found version 12040).' 1>&2\n"
                         "echo '{\"ok\": false, \"torch\": \"2.11.0+cu130\","
                         " \"built\": \"13.0\", \"name\": null}'")
        try:
            CE.require_accelerator({"capture": {"device": "cuda"}}, root, quiet)
            check("C12b a CUDA-less box REFUSES before any stage", False,
                  "it proceeded, which is how 10 GB got fetched onto a dead box")
        except CE.Refusal as exc:
            check("C12b a CUDA-less box REFUSES before any stage", True)
            check("C12c ...naming the driver version, because the remedy "
                  "depends on which it is",
                  any("12040" in a for a in exc.advice))
            check("C12d ...and stating that nothing was fetched",
                  any("Nothing was fetched" in a for a in exc.advice))

    with tempfile.TemporaryDirectory() as tmp:
        root = venv(tmp, "echo '{\"ok\": false, \"torch\": \"x\","
                         " \"built\": \"13.0\", \"name\": null}'")
        CE.require_accelerator({"capture": {"device": "cpu"}}, root, quiet)
        check("C12e a job that asked for cpu is not gated", True)

    with tempfile.TemporaryDirectory() as tmp:
        CE.require_accelerator({"capture": {"device": "cuda"}}, Path(tmp), quiet)
        check("C12f before the venv exists the bootstrap speaks first", True)

    with tempfile.TemporaryDirectory() as tmp:
        root = venv(tmp, "echo boom 1>&2; exit 3")
        try:
            CE.require_accelerator({"capture": {"device": "cuda"}}, root, quiet)
            check("C12g a box that cannot ANSWER is refused, not assumed good",
                  False, "proceeded")
        except CE.Refusal:
            check("C12g a box that cannot ANSWER is refused, not assumed good",
                  True)


def rung_release():
    print("[C11] what a release build would tag, build and push")
    import release_plan as RP                               # noqa: E402
    import changelog as CL                                  # noqa: E402

    sha = "a" * 40

    def plan(**kw):
        base = dict(event="workflow_dispatch", ref="refs/heads/main", sha=sha,
                    image="ghcr.io/x/y", publish="false")
        base.update(kw)
        return RP.plan(argparse.Namespace(**base))

    rel = plan(event="release", ref="refs/tags/v1.2.3", publish="true")
    check("C11a a release tags the series and latest",
          rel["tags"] == ["ghcr.io/x/y:sha-aaaaaaaaaaaa", "ghcr.io/x/y:1.2.3",
                          "ghcr.io/x/y:1.2", "ghcr.io/x/y:1",
                          "ghcr.io/x/y:latest"], "%s" % rel["tags"])
    pre = plan(event="release", ref="refs/tags/v1.2.3-rc1", publish="true")
    check("C11b a PRERELEASE does not move latest or the series tags",
          pre["tags"] == ["ghcr.io/x/y:sha-aaaaaaaaaaaa", "ghcr.io/x/y:1.2.3"],
          "%s" % pre["tags"])
    check("C11c the immutable sha- tag is always first, and is what the image "
          "records as its own reference",
          rel["tags"][0].startswith("ghcr.io/x/y:sha-")
          and rel["build_args"]["IMAGE_REFERENCE"] == rel["tags"][0])
    check("C11d SUITE_REVISION is the full commit the receipt must name",
          rel["build_args"]["SUITE_REVISION"] == sha)

    off = plan(event="release", ref="refs/tags/v1.2.3")
    check("C11e publishing is DEFAULT-OFF: landing the workflow publishes "
          "nothing", off["push"] is False)
    check("C11f ... and the plan says which switch turns it on",
          any("PUBLISH_CONTAINER" in r for r in off["push_blocked_because"]))
    pr = plan(event="pull_request", ref="refs/pull/7/merge", publish="true")
    check("C11g a pull request never pushes, even with the gate on",
          pr["push"] is False
          and any("pull request" in r for r in pr["push_blocked_because"]))
    check("C11h both architectures are in every plan",
          rel["platforms"] == ["linux/amd64", "linux/arm64"])

    try:
        plan(sha="deadbeef")
        check("C11i a short sha is refused", False, "it planned anyway")
    except SystemExit as exc:
        check("C11i a short sha is refused, naming why the schema needs it",
              "produced_by.revision" in str(exc))

    wf = SUITE / ".github" / "workflows" / "container-image.yml"
    text = wf.read_text(encoding="utf-8") if wf.is_file() else ""
    check("C11j the workflow exists", bool(text))
    check("C11k it asks the script rather than deciding in an expression",
          "bin/release_plan.py" in text)
    check("C11l the push is gated on the repository variable",
          "vars.PUBLISH_CONTAINER" in text)
    check("C11m it builds both platforms",
          "linux/amd64" in text and "linux/arm64" in text)
    check("C11n it passes the build args the image records",
          "SUITE_REVISION=" in text and "IMAGE_REFERENCE=" in text)
    check("C11o it runs this battery before building",
          "selftest_container.py" in text)

    # The rungs above are string checks, because `bin/` runs on stock
    # python3.9 with no installs and PyYAML is not stdlib. A workflow that does
    # not PARSE fails only on GitHub, which is the one place this project
    # cannot test -- so when a yaml is importable anywhere on this machine, use
    # it, and SKIP loudly when it is not rather than pretending the string
    # checks covered it.
    #
    # Finding the interpreter and PARSING THE FILE are two questions, asked
    # separately on purpose: fold them together and an unparseable workflow
    # comes back as "no yaml module here" -- a SKIP where a FAIL belongs, which
    # is the fail-open shape this repository keeps paying for.
    interp = None
    for candidate in (sys.executable, str(SUITE / ".venv" / "bin" / "python"),
                      "/opt/homebrew/bin/python3.14", "python3"):
        if subprocess.run([candidate, "-c", "import yaml"],
                          capture_output=True).returncode == 0:
            interp = candidate
            break
    if interp is None:
        print("  SKIP  C11o2 workflow YAML parse (no yaml module on any "
              "interpreter here; the string rungs above are what ran)")
    else:
        probe = subprocess.run(
            [interp, "-c",
             "import yaml,json,sys;d=yaml.safe_load(open(sys.argv[1]));"
             "print(json.dumps({'jobs':sorted(d['jobs']),"
             "'platforms':[m['platform'] for m in "
             "d['jobs']['build']['strategy']['matrix']['include']]}))",
             str(wf)], capture_output=True, text=True)
        check("C11o2 the workflow parses at all (%s)" % Path(interp).name,
              probe.returncode == 0,
              (probe.stderr or "").strip().splitlines()[-1:] and
              (probe.stderr or "").strip().splitlines()[-1])
        if probe.returncode == 0:
            doc = json.loads(probe.stdout.strip().splitlines()[-1])
            check("C11o3 ... with the four jobs",
                  doc["jobs"] == ["build", "changelog", "manifest", "plan"],
                  "%s" % doc["jobs"])
            check("C11o4 ... and one matrix job per architecture",
                  doc["platforms"] == ["linux/amd64", "linux/arm64"],
                  "%s" % doc["platforms"])

    print("[C11p] the changelog groups by the topic convention, not by any token")
    known = [
        ("container: run the measurement as an IMAGE", ("container",
                                                        "run the measurement as an IMAGE")),
        ("bundle: a bundled script's DATA is a dependency too",
         ("bundle", "a bundled script's DATA is a dependency too")),
        # A FILE, an identifier and a flag can all open a subject; grouping by
        # those gives one section per commit, which is a list with extra
        # headings rather than a changelog.
        ("AGENTS.md: how to work on this repo", ("", "AGENTS.md: how to work on this repo")),
        ("REFC-006: a family that publishes no weights",
         ("", "REFC-006: a family that publishes no weights")),
        ("--pipeline-root: the third default", ("", "--pipeline-root: the third default")),
        ("Merge branch 'main'", None),
        ("no colon at all", ("", "no colon at all")),
    ]
    for subject, want in known:
        got = CL.split_subject(subject)
        check("C11p %s" % subject[:44], got == want, "got %r want %r" % (got, want))
    # NOT a staleness check. CHANGELOG.md is generated from the commits, so
    # it is one commit behind for as long as it takes to commit it -- making
    # that fatal here would fail the battery immediately after every commit,
    # which trains people to ignore it. What must hold is that the file is
    # GENERATED (not hand-edited) and that the generator still produces every
    # line it contains. CI keeps the staleness check, as a warning.
    text = (SUITE / "CHANGELOG.md").read_text(encoding="utf-8")
    check("C11q CHANGELOG.md exists and says it is generated",
          "bin/changelog.py" in text.splitlines()[2] if len(text.splitlines()) > 2
          else False)
    regenerated = set(CL.full_changelog().splitlines())
    orphans = [ln for ln in text.splitlines()
               if ln.startswith("- ") and ln not in regenerated]
    check("C11r every entry in it is one the generator still produces",
          not orphans, "hand-edited or lost: %s" % orphans[:3])


def main() -> int:
    rung_sequence()
    rung_job_document()
    rung_token()
    rung_bundle()
    rung_pin()
    rung_capture_identity()
    rung_dockerfile()
    rung_cli()
    rung_accelerator()
    rung_release()
    rung_github_output()
    print("")
    if FAILED:
        print("FAILED %d:" % len(FAILED))
        for name in FAILED:
            print("  - %s" % name)
        return 1
    print("container path: all rungs pass")
    return 0



def rung_github_output():
    """C11s -- the plan step's outputs actually reach GITHUB_OUTPUT.

    P1-17 (independent peer review): the workflow caught release_plan's
    name=value lines with `| tee summary 2> "$GITHUB_OUTPUT"` -- a redirect
    that binds to TEE's stderr, which is silent. GITHUB_OUTPUT stayed empty,
    `push` evaluated false, and the first ARMED publish run built both
    architectures and then skipped GHCR with every job green. release_plan now
    writes the runner's file itself, so a caller's plumbing cannot lose it.
    """
    import tempfile

    print("[C11s] the plan step owns its outputs")
    fh = tempfile.NamedTemporaryFile("r", delete=False)
    gh_out = fh.name
    fh.close()
    env = dict(os.environ, GITHUB_OUTPUT=gh_out)
    p = subprocess.run(
        [sys.executable, str(SUITE / "bin" / "release_plan.py"),
         "--event", "workflow_dispatch", "--ref", "refs/heads/main",
         "--sha", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
         "--publish", "true", "--github-output"],
        capture_output=True, text=True, env=env)
    body = open(gh_out, encoding="utf-8").read()
    os.unlink(gh_out)
    check("C11s release_plan exits 0", p.returncode == 0)
    check("C11s GITHUB_OUTPUT is non-empty", bool(body.strip()))
    check("C11s ...and carries push=", "push=" in body)
    check("C11s ...and the tags", "tags=ghcr.io/" in body)
    check("C11s nothing to stderr for a caller's redirect to lose",
          "push=" not in (p.stderr or ""))

    print("[C11t] a digest never becomes a filename with a colon in it")
    # upload-artifact@v4 refuses ':' in filenames; buildx digests are
    # `sha256:<hex>`. The workflow must strip the prefix when it writes the
    # digest AS a file, and the manifest job re-adds it. Run two died on this.
    wf = (SUITE / ".github" / "workflows" / "container-image.yml").read_text(
        encoding="utf-8")
    check("C11t the digest file strips the sha256: prefix",
          '${digest#sha256:}' in wf)
    check("C11t no raw digest expression is used as a path",
          'digests/${{ steps.build.outputs.digest }}' not in wf)
    check("C11t the manifest job re-adds the prefix when reading",
          "@sha256:%s" in wf)


if __name__ == "__main__":
    sys.exit(main())
