#!/usr/bin/env python3
"""ROOT-1: a sealed root gets published, and teardown refuses to eat one.

WHY THIS EXISTS
---------------
The controller destroyed a sealed, twice-validated MiniMax-M3 root dataset at
teardown -- $6.59 of GPU time and the only copy of the evidence -- because
nothing published or preserved a root (REVIEW-DEFERRED ROOT-1, fee collected
2026-08-31).

  RP1  remote execution ends at qualification; race and remote publication
       both refuse.
  RP2  legacy teardown holds verified but unpublished root evidence.
  RP3  the explicit legacy override permits that teardown.
  RP4  an already-published legacy root tears down normally.
  RP5  a quant run is unaffected by the legacy root hold.
  RP6  container and safe SSH composition bind intended dataset identity while
       refusing remote publication.
  RP7  RunPod refuses container-native execution and provider-carried HF
       credentials before mutation.
  RP8  malformed or quant-only publication arguments refuse before spend.
  RP9  local publication requires the exact verified two-capture archive,
       qualification, immutable refetch, and fetched-byte binding.
  RP10 SSH controller refuses preview/race paid roots before provider access.
  RP11 container composition refuses the same unsupported path.

Stub provider, no network, $0.00.  Stage-level publication ordering and argv
composition live in bin/selftest_stage_measure.py, which executes the real
script.
"""
import importlib.util
import json
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))

from fidelity.stages import stage_sequence  # noqa: E402

FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        FAILED.append(label)
        for line in str(detail).splitlines()[:10]:
            print("        %s" % line)


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC = load("measure_cloud", "bin/measure_cloud.py")
CE = load("container_entry", "bin/container_entry.py")


class Con:
    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        def log(*a, **kw):
            self.lines.append(" ".join(str(x) for x in a))
        return log

    def text(self):
        return "\n".join(self.lines)


class StubJL:
    dry = False

    def __init__(self):
        self.destroyed = []

    def destroy(self, mid):
        self.destroyed.append(mid)

    def list_instances(self):
        return [] if self.destroyed else [
            type("I", (), {"machine_id": 77, "status": "running"})()]

    def exec(self, *a, **kw):
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    def exec_stdout(self, *a, **kw):
        return ""

    def download(self, *a, **kw):
        return {"ok": True}

    def fs_delete(self, fsid):
        return {}


def teardown(tmp, **flags):
    jl, con = StubJL(), Con()
    Path(tmp).mkdir(parents=True, exist_ok=True)
    td = MC.Teardown(jl, con, Path(tmp))
    td.machine_id = 77
    lease = Path(tmp) / "lease.json"
    lease.write_text(json.dumps({"job_id": "x", "machine_id": 77,
                                 "deadline_epoch": 0}))
    td.lease_path = lease
    for key, value in flags.items():
        setattr(td, key, value)
    td.run("test")
    return td, jl, con, lease

def fruit_target_descriptor(root):
    # The descriptor must carry the identity the authored timing row binds
    # (shard-file bytes, config and index digests), or job_document refuses
    # for the wrong reason: read the row rather than restate it.
    revision = "ef68013aa6e16453cf52b5b77647f72fbe258c3c"
    row = next(
        row for row in json.load(
            open(ROOT / "bin" / "engines.json", encoding="utf-8"))["root_timing_profiles"]
        if row["target_repo"] == "malaiwah/GLM-5.2-SIQ-Fruit-bf16"
        and row["target_revision"] == revision)
    model_bytes = row["model_identity"]["model_bytes"]
    shards = [{
        "path": "model-00001-of-00001.safetensors",
        "bytes": model_bytes,
    }]
    shard_raw = json.dumps(
        shards, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    document = {
        "repo_id": "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
        "revision": revision,
        "requested_revision": revision,
        "surface": "native-bf16",
        "codec": "bf16",
        "bits": 16.0,
        "path": None,
        "config_sha256": row["model_identity"]["config_sha256"],
        "index_sha256": row["model_identity"]["index_sha256"],
        "model_bytes": model_bytes,
        "shards": shards,
        "shard_manifest_sha256": hashlib.sha256(shard_raw).hexdigest(),
    }
    path = Path(root) / "fruit-target.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def main():
    # RP1: remote execution ends at qualification. Publication is controller-local.
    safe_root = [
        "setup", "fetch_target", "capture", "verify",
        "capture_repeat", "verify_repeat", "compare_root", "qualify_root",
    ]
    race_refused = False
    publication_refused = False
    publication_message = ""
    try:
        stage_sequence("root", race=True)
    except ValueError:
        race_refused = True
    try:
        stage_sequence("root", publish_root=True)
    except ValueError as exc:
        publication_refused = True
        publication_message = str(exc)
    check("RP1 remote root ends at qualification; race and remote publication refuse",
          stage_sequence("root") == safe_root
          and "publish_root" not in stage_sequence("quant", publish_root=True)
          and race_refused and publication_refused
          and "controller-local" in publication_message)

    with tempfile.TemporaryDirectory() as tmp:
        # RP2: verified + unpublished -> HELD.
        td, jl, con, lease = teardown(
            tmp + "/a", root_publish_expected=True, root_verified=True,
            root_published=False)
        check("RP2 teardown HOLDS a verified, unpublished root "
              "(no destroy, lease kept)",
              jl.destroyed == [] and td.held_for_unpublished_root
              and lease.exists()
              and "NEVER PUBLISHED" in con.text(),
              "destroyed=%s held=%s\n%s" % (jl.destroyed,
                                            td.held_for_unpublished_root,
                                            con.text()[-500:]))

        # RP3: the explicit override destroys.
        td, jl, con, lease = teardown(
            tmp + "/b", root_publish_expected=True, root_verified=True,
            root_published=False, allow_unpublished_root=True)
        check("RP3 --allow-unpublished-root destroys",
              jl.destroyed == [77] and not td.held_for_unpublished_root,
              (jl.destroyed, con.text()[-300:]))

        # RP4: published -> normal teardown.
        td, jl, con, lease = teardown(
            tmp + "/c", root_publish_expected=True, root_verified=True,
            root_published=True)
        check("RP4 a published root tears down normally",
              jl.destroyed == [77] and not td.held_for_unpublished_root)

        # RP5: quant runs unaffected.
        td, jl, con, lease = teardown(tmp + "/d")
        check("RP5 a quant run is untouched by the guard",
              jl.destroyed == [77] and not td.held_for_unpublished_root)

        # RP6: publication is rejected before panel staging or job creation.
        panel = Path(tmp) / "absent-panel"
        fs = Path(tmp) / "fs"
        fs.mkdir()
        target_path, target = fruit_target_descriptor(tmp)
        args = CE.build_parser().parse_args([
            "capture", "--model", target["repo_id"],
            "--revision", target["revision"], "--lane", "streaming",
            "--target-descriptor", str(target_path), "--gpu", "L4",
            "--panel-dir", str(panel), "--dataset-id", "fruit-root-v1",
            "--dataset-repository", "malaiwah/fruit-root-v1",
            "--publish-root-to", "malaiwah/fruit-root-v1",
            "--replay-device", "numpy", "--replay-dtype", "float32",
            "--vocab-chunk", "8192",
            "--workspace-available-bytes-minimum", "1",
            "--container-available-bytes-minimum", "1",
            "--expected-vram-bytes", "1",
            "--fs-root", str(fs)])
        remote_publication_refused = False
        remote_publication_message = ""
        try:
            CE.job_document(args, ROOT, fs, lambda *a, **kw: None)
        except CE.Refusal as exc:
            remote_publication_refused = True
            remote_publication_message = str(exc)
        check("RP6 container remote publication is refused before panel staging",
              remote_publication_refused
              and "unsupported" in remote_publication_message,
              remote_publication_message)
        import inspect
        src = inspect.getsource(MC._plan_runpod_anonymous)
        check("RP6c the safe SSH controller binds intended dataset repository",
              '"dataset_repository"' in src)

    # RP7: the safe RunPod path is SSH-only and never transports HF credentials.
    from fidelity.runpodapi import RunPod, RunPodError
    rp = RunPod.__new__(RunPod)
    submitted = []
    rp.submit_prepared_create = lambda prepared: submitted.append(prepared)
    native_refused = False
    env_refused = False
    token = "hf_selftest_not_a_real_token_22222"
    try:
        rp.create(docker_args="capture --publish-root-to malaiwah/root",
                  env={"HF_TOKEN": token})
    except RunPodError as exc:
        native_refused = "native docker" in str(exc)
    try:
        rp.create(env={"HF_TOKEN": token})
    except RunPodError as exc:
        env_refused = "provider env" in str(exc)
    check("RP7 RunPod refuses container-native execution before mutation",
          native_refused and submitted == [])
    check("RP7b RunPod refuses provider-carried HF credentials before mutation",
          env_refused and submitted == [])

    # RP9: qualification is mandatory, the exact returned commit is refetched
    # (never mutable main), and the receipt binds the bytes actually refetched.
    FD = load("fidelity_dataset", "bin/fidelity_dataset.py")
    from fidelity import dshub as real_dshub
    import argparse as _ap
    import types as _types
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        receipt = root / "publish-root.json"
        extraction = root / "result"
        extraction.mkdir(mode=0o700)
        dataset = extraction / "dataset"
        dataset.mkdir()
        receipts = extraction / "receipts"
        receipts.mkdir()
        qualification = receipts / "root-qualification.json"
        qualification.write_bytes(b"sealed qualification bytes\n")
        job_path = extraction / "job.json"
        job_path.write_bytes(b"sealed job bytes\n")
        qualification_sha = hashlib.sha256(
            qualification.read_bytes()).hexdigest()
        missing_rc = FD.cmd_publish(_ap.Namespace(
            dataset=str(dataset), repo="malaiwah/mm3-root-v1", private=False,
            revision_message="m", token_file=None, receipt=str(receipt),
            qualification=None))
        check("RP9 missing qualification refuses before any upload",
              missing_rc == FD.REFUSED and not receipt.exists(), missing_rc)

        fetched_refs = []
        fetched_tokens = []
        evidence_tokens = []
        local_hash_calls = []
        old_hf = sys.modules.get("huggingface_hub")
        orig = (
            real_dshub.publish_dataset,
            real_dshub.verify_remote_dataset_exact,
            real_dshub.fetch_exact_bytes,
            real_dshub.read_token,
            FD.dsvalidate.validate_dataset, FD.F.load_manifest,
            FD._load_qualification, FD._verify_publish_source_archive,
            FD.common.sha256_file,
        )
        try:
            real_dshub.publish_dataset = lambda *a, **kw: {
                "repository": "malaiwah/mm3-root-v1",
                "dataset_sha256": "d" * 64, "private": False,
                "revision": "b" * 40,
                "qualification_path_in_repo":
                    "receipts/root-qualification.json"}

            def fake_verify_remote(repo, revision, records, **kwargs):
                fetched_refs.append("hf://%s@%s" % (repo, revision))
                fetched_tokens.append(None)
                return {
                    "dataset_sha256": kwargs["expected_dataset_sha256"],
                    "files_verified": len(records),
                    "bytes_verified": kwargs["max_total_bytes"],
                    "bounded_streaming": True,
                }

            def fake_file_hash(path):
                path = str(path)
                if path.endswith("root-qualification.json"):
                    local_hash_calls.append(path)
                    return qualification_sha
                return "d" * 64

            # cmd_publish must not perform a second Hub commit.  The dshub
            # publisher owns the single commit containing dataset + evidence.
            real_dshub.verify_remote_dataset_exact = fake_verify_remote
            real_dshub.read_token = lambda *a, **kw: "token"

            def fake_exact_bytes(_url, expected_bytes, expected_sha256, **kwargs):
                evidence_tokens.append(kwargs.get("token"))
                assert expected_bytes == len(qualification.read_bytes())
                assert expected_sha256 == qualification_sha
                return qualification.read_bytes()

            real_dshub.fetch_exact_bytes = fake_exact_bytes
            FD.dsvalidate.validate_dataset = lambda *a, **kw: type(
                "R", (), {"errors": [], "warnings": [], "passed": True})()
            FD.F.load_manifest = lambda *a, **kw: {FD.F.SEAL_FIELD: "d" * 64}
            FD._load_qualification = lambda *a, **kw: {
                "receipt_sha256": "q" * 64}
            FD._verify_publish_source_archive = lambda *a, **kw: {
                "archive_sha256": "f" * 64,
                "archive_bytes": 123,
                "canonical_dataset_records": {
                    "fidelity-dataset.json": {
                        "bytes": 1, "sha256": "d" * 64}},
                "canonical_dataset_bytes": 1,
                "qualification_record": {
                    "bytes": len(qualification.read_bytes()),
                    "sha256": qualification_sha,
                },
            }
            FD.common.sha256_file = fake_file_hash
            sys.modules["huggingface_hub"] = _types.SimpleNamespace()
            private_rc = FD.cmd_publish(_ap.Namespace(
                dataset=str(dataset), repo="malaiwah/mm3-root-v1", private=True,
                revision_message="m", token_file=None, receipt=str(receipt),
                qualification=str(qualification), job=str(job_path),
                result_archive=str(root / "result.tar.gz"),
                expected_archive_sha256="f" * 64,
                expected_archive_bytes=123, expected_head=None))
            check("RP9 private/token-only root publication refuses before upload",
                  private_rc == FD.REFUSED and not receipt.exists(), private_rc)


            rc = FD.cmd_publish(_ap.Namespace(
                dataset=str(dataset), repo="malaiwah/mm3-root-v1", private=False,
                revision_message="m", token_file=None, receipt=str(receipt),
                qualification=str(qualification), job=str(job_path),
                result_archive=str(root / "result.tar.gz"),
                expected_archive_sha256="f" * 64,
                expected_archive_bytes=123, expected_head=None))
            doc = json.loads(receipt.read_text()) if receipt.is_file() else {}
            check("RP9 immutable publication proof pins and refetches exact revision",
                  rc == FD.OK and doc.get("schema")
                  == "fidelity.publish-root-receipt.v2"
                  and doc.get("revision") == "b" * 40
                  and doc.get("verified_revision") == "b" * 40
                  and fetched_refs == [
                      "hf://malaiwah/mm3-root-v1@%s" % ("b" * 40)]
                  and doc.get("revision_immutable") is True
                  and fetched_tokens == [None]
                  and evidence_tokens == [None]
                  and doc.get("private") is False
                  and doc.get("verified_anonymously") is True
                  and doc.get("result_archive_sha256") == "f" * 64
                  and doc.get("result_archive_bytes") == 123,
                  (doc, fetched_refs, fetched_tokens, evidence_tokens))
            check("RP9b published qualification hash comes from fetched evidence",
                  doc.get("published_qualification_file_sha256")
                  == qualification_sha
                  and doc.get("qualification_file_sha256")
                  == qualification_sha,
                  doc)
        finally:
            (real_dshub.publish_dataset,
             real_dshub.verify_remote_dataset_exact,
             real_dshub.fetch_exact_bytes,
             real_dshub.read_token,
             FD.dsvalidate.validate_dataset, FD.F.load_manifest,
             FD._load_qualification, FD._verify_publish_source_archive,
             FD.common.sha256_file) = orig
            if old_hf is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = old_hf

        class StreamResponse:
            def __init__(self, body, reads):
                self.body = body
                self.offset = 0
                self.reads = reads

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getcode(self):
                return 200

            def read(self, size):
                self.reads.append(size)
                chunk = self.body[self.offset:self.offset + size]
                self.offset += len(chunk)
                return chunk

        manifest_body = b"sealed-manifest"
        checksums_body = b"sealed-checksums"
        payload_body = b"x" * (2 * 1024 * 1024 + 17)
        remote_records = {
            FD.F.MANIFEST_NAME: {
                "bytes": len(manifest_body),
                "sha256": hashlib.sha256(manifest_body).hexdigest(),
            },
            FD.F.CHECKSUMS_NAME: {
                "bytes": len(checksums_body),
                "sha256": hashlib.sha256(checksums_body).hexdigest(),
            },
            "capture/payload.bin": {
                "bytes": len(payload_body),
                "sha256": hashlib.sha256(payload_body).hexdigest(),
            },
        }
        bodies = {
            real_dshub.resolve_url(
                "malaiwah/mm3-root-v1", "b" * 40, relpath): body
            for relpath, body in (
                (FD.F.MANIFEST_NAME, manifest_body),
                (FD.F.CHECKSUMS_NAME, checksums_body),
                ("capture/payload.bin", payload_body))
        }
        stream_reads = []
        stream_opens = []

        class StreamOpener:
            @staticmethod
            def open(request, timeout):
                stream_opens.append((request.full_url, timeout))
                return StreamResponse(
                    bodies[request.full_url], stream_reads)

        original_opener = real_dshub._OPENER
        try:
            real_dshub._OPENER = StreamOpener()
            total_remote_bytes = sum(
                record["bytes"] for record in remote_records.values())
            streamed = real_dshub.verify_remote_dataset_exact(
                "malaiwah/mm3-root-v1", "b" * 40, remote_records,
                expected_dataset_sha256="d" * 64,
                max_total_bytes=total_remote_bytes)
            check("RP9l immutable refetch streams within one-MiB reads",
                  streamed["bounded_streaming"] is True
                  and streamed["bytes_verified"] == total_remote_bytes
                  and max(stream_reads) <= 1024 * 1024
                  and len(stream_opens) == len(remote_records),
                  (streamed, stream_reads, stream_opens))
            opens_before_bound = len(stream_opens)
            try:
                real_dshub.verify_remote_dataset_exact(
                    "malaiwah/mm3-root-v1", "b" * 40, remote_records,
                    expected_dataset_sha256="d" * 64,
                    max_total_bytes=total_remote_bytes - 1)
            except real_dshub.HubError:
                aggregate_refused = True
            else:
                aggregate_refused = False
            check("RP9m aggregate public-byte overflow refuses before I/O",
                  aggregate_refused
                  and len(stream_opens) == opens_before_bound)
            payload_url = real_dshub.resolve_url(
                "malaiwah/mm3-root-v1", "b" * 40,
                "capture/payload.bin")
            bodies[payload_url] = payload_body + b"!"
            try:
                real_dshub.verify_remote_dataset_exact(
                    "malaiwah/mm3-root-v1", "b" * 40, remote_records,
                    expected_dataset_sha256="d" * 64,
                    max_total_bytes=total_remote_bytes)
            except real_dshub.HubError:
                excess_member_refused = True
            else:
                excess_member_refused = False
            finally:
                bodies[payload_url] = payload_body
            check("RP9n expected-plus-one public member is refused",
                  excess_member_refused
                  and max(stream_reads) <= 1024 * 1024)
        finally:
            real_dshub._OPENER = original_opener
        root.chmod(0o770)
        try:
            FD._private_publish_inputs(
                str(dataset), str(qualification), str(job_path))
        except FD.RootQualificationError:
            unsafe_ancestor_refused = True
        else:
            unsafe_ancestor_refused = False
        finally:
            root.chmod(0o700)
        check("RP9o group-writable extraction ancestor is refused",
              unsafe_ancestor_refused)

    # The Hub mutation itself is one optimistic commit.  It preflights the
    # authenticated namespace and exact closure before creating or committing.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        data_file = root / real_dshub.F.MANIFEST_NAME
        data_file.write_bytes(b"manifest\n")
        (root / real_dshub.F.CHECKSUMS_NAME).write_bytes(b"checksums\n")
        qualification = Path(tmp) / "qualification.json"
        qualification.write_bytes(b"qualification\n")

        class Add:
            def __init__(self, **kwargs):
                self.path_in_repo = kwargs["path_in_repo"]
                self.path_or_fileobj = kwargs["path_or_fileobj"]

        class RepositoryNotFoundError(Exception):
            def __init__(self, message, *, exact=True):
                super().__init__(message)
                if exact:
                    self.response = type("Response", (), {"status_code": 404})()

        class Api:
            def __init__(self, *, head="a" * 40, owner="malaiwah",
                         absent=False, ambiguous_absent=False,
                         commit_error=None):
                self.head = head
                self.owner = owner
                self.absent = absent
                self.ambiguous_absent = ambiguous_absent
                self.commit_error = commit_error
                self.created = []
                self.commits = []

            def whoami(self, **kwargs):
                return {"name": self.owner, "orgs": []}

            def repo_info(self, **kwargs):
                if kwargs.get("repo_type") != "dataset":
                    raise RepositoryNotFoundError("absent repo type")
                if self.absent:
                    raise RepositoryNotFoundError(
                        "absent", exact=not self.ambiguous_absent)
                return type(
                    "Info", (), {"sha": self.head, "private": False})()

            def create_repo(self, **kwargs):
                self.created.append(kwargs)
                self.absent = False

            def create_commit(self, **kwargs):
                self.commits.append(kwargs)
                if self.commit_error:
                    raise self.commit_error
                return type("Commit", (), {"oid": "b" * 40})()

        original_hf = sys.modules.get("huggingface_hub")
        original_atomic = (
            real_dshub.dsvalidate.validate_dataset
            if hasattr(real_dshub, "dsvalidate") else None,
            real_dshub.F.load_manifest,
            real_dshub.F.iter_dataset_files,
        )
        # publish_dataset imports dsvalidate locally, so patch the real module.
        from fidelity import dsvalidate as atomic_validate
        original_validate = atomic_validate.validate_dataset
        try:
            atomic_validate.validate_dataset = lambda *a, **kw: type(
                "R", (), {"passed": True, "errors": []})()
            real_dshub.F.load_manifest = lambda *a, **kw: {
                real_dshub.F.SEAL_FIELD: "d" * 64,
                "dataset": {"structural_status": "sealed"}}
            real_dshub.F.iter_dataset_files = lambda *a, **kw: [
                real_dshub.F.MANIFEST_NAME]

            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            atomic = real_dshub.publish_dataset(
                str(root), "malaiwah/mm3-root-v1", str(qualification),
                expected_head="a" * 40, token="token")
            operation_names = [
                op.path_in_repo for op in api.commits[0]["operations"]]
            check("RP9c dataset and qualification land in one parent-conditioned commit",
                  atomic["revision"] == "b" * 40
                  and len(api.commits) == 1
                  and api.commits[0]["parent_commit"] == "a" * 40
                  and operation_names == [
                      real_dshub.F.CHECKSUMS_NAME,
                      real_dshub.F.MANIFEST_NAME,
                      "receipts/root-qualification.json"],
                  (atomic, operation_names, api.commits))

            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            refused_unknown = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head=None, token="token")
            except real_dshub.HubError as exc:
                refused_unknown = "already exists" in str(exc)
            check("RP9d unknown existing history refuses without a commit",
                  refused_unknown and not api.commits and not api.created)

            api = Api(absent=True, head="c" * 40)
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            real_dshub.publish_dataset(
                str(root), "malaiwah/mm3-root-v1", str(qualification),
                expected_head=None, token="token")
            check("RP9e absent destination uses exclusive create and new exact parent",
                  len(api.created) == 1
                  and api.created[0]["exist_ok"] is False
                  and api.commits[0]["parent_commit"] == "c" * 40,
                  (api.created, api.commits))


            api = Api(absent=True, ambiguous_absent=True)
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            ambiguous_refused = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head=None, token="token")
            except real_dshub.HubError as exc:
                ambiguous_refused = "destination preflight failed" in str(exc)
            check("RP9e2 missing HTTP status cannot authorize repository creation",
                  ambiguous_refused and not api.created and not api.commits)
            api = Api(owner="other")
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            refused_permission = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token="token")
            except real_dshub.HubError as exc:
                refused_permission = "lacks declared write authority" in str(exc)
            check("RP9f principal permission refusal performs no mutation",
                  refused_permission and not api.commits and not api.created)

            api = Api(commit_error=RuntimeError("parent changed"))
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            refused_race = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token="token")
            except real_dshub.HubError as exc:
                refused_race = "optimistic one-commit" in str(exc)
            check("RP9g concurrent HEAD change has no fallback or partial commit",
                  refused_race and len(api.commits) == 1 and not api.created)

            credential = root / ".hf_token"
            credential.write_text("secret")
            real_dshub.F.iter_dataset_files = lambda *a, **kw: [".hf_token"]
            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            refused_credential = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token="token")
            except (real_dshub.HubError, real_dshub.F.FormatError) as exc:
                refused_credential = "credential" in str(exc)
            check("RP9h credential in exact closure refuses before Hub preflight",
                  refused_credential and not api.commits and not api.created)

            real_dshub.F.iter_dataset_files = lambda *a, **kw: [
                real_dshub.F.MANIFEST_NAME]
            credential.unlink()
            exact_token = "hf_exact_boundary_token_1234567890"
            prefix = b"x" * ((1024 * 1024) - 5)
            data_file.write_bytes(prefix + exact_token.encode("ascii"))
            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            boundary_refused = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token=exact_token)
            except real_dshub.HubError as exc:
                boundary_refused = "exact credential bytes" in str(exc)
            check("RP9h2 exact token spanning scan chunks refuses before mutation",
                  boundary_refused and not api.commits and not api.created)

            data_file.write_bytes(b"manifest\n")
            qualification.write_bytes(
                b'{"private_path":"/home/controller/frozen/job.json"}\n')
            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            path_refused = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token="different-secret")
            except real_dshub.HubError as exc:
                path_refused = "private absolute path" in str(exc)
            check("RP9h3 private absolute path in qualification refuses before mutation",
                  path_refused and not api.commits and not api.created)

            qualification.write_bytes(
                b'{"note":"hf_apparentcredential1234567890"}\n')
            api = Api()
            sys.modules["huggingface_hub"] = _types.SimpleNamespace(
                HfApi=lambda token=None, endpoint=None: api, CommitOperationAdd=Add)
            apparent_refused = False
            try:
                real_dshub.publish_dataset(
                    str(root), "malaiwah/mm3-root-v1", str(qualification),
                    expected_head="a" * 40, token="different-secret")
            except real_dshub.HubError as exc:
                apparent_refused = "apparent Hugging Face token" in str(exc)
            check("RP9h4 apparent HF token in evidence refuses before mutation",
                  apparent_refused and not api.commits and not api.created)
        finally:
            atomic_validate.validate_dataset = original_validate
            real_dshub.F.load_manifest = original_atomic[1]
            real_dshub.F.iter_dataset_files = original_atomic[2]
            if original_hf is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original_hf

    # RP8: refusals, before any spend.
    rc = MC.main(["--model", "x/y", "--panel", "o/p", "--lane", "streaming",
                  "--publish-root-to", "a/b"])
    check("RP8 --publish-root-to on a quant run is refused",
          rc == MC.EXIT_REFUSED, rc)
    rc = MC.main(["--role", "root", "--model", "x/y", "--panel", "o/p",
                  "--lane", "streaming", "--dataset-id", "d",
                  "--publish-root-to", "not-a-repo"])
    check("RP8b a malformed repo id is refused", rc == MC.EXIT_REFUSED, rc)
    # The Hub's own rule, enforced before spend: a paid Fruit capture on
    # 2026-09-04 reached its final step before the publisher learned that
    # 'fidelity--fruit.malaiwah.root.bf16' is not a legal Hub name.
    forbidden_ids = ("malaiwah/fidelity--fruit.malaiwah.root.bf16",
                     "malaiwah/a..b", "mal--aiwah/ok", "malaiwah/x" * 50,
                     "malaiwah/name.git")
    refused_ids = [
        repo for repo in forbidden_ids
        if MC.main(["--role", "root", "--model", "x/y", "--panel", "o/p",
                    "--lane", "streaming", "--dataset-id", "d",
                    "--publish-root-to", repo]) == MC.EXIT_REFUSED]
    check("RP8c the Hub's forbidden repo-id forms are refused before spend",
          refused_ids == list(forbidden_ids), refused_ids)
    from fidelity import dshub as rule_dshub

    def rule_refuses(repo):
        try:
            rule_dshub.validate_repo_id(repo)
        except rule_dshub.HubError:
            return True
        return False
    check("RP8d legal ids pass the rule and forbidden ids raise in the publisher",
          rule_dshub.validate_repo_id("malaiwah/glm53-fidelity-root-v1")
          == "malaiwah/glm53-fidelity-root-v1"
          and not rule_refuses("malaiwah/fruit-fidelity-root-container-v1")
          and all(rule_refuses(repo) for repo in forbidden_ids))

    # RP10/RP11: the first safe paid path has no preview/race branch, under
    # either SSH controller or container composition.
    race_args = MC.build_parser().parse_args([
        "--provider", "runpod", "--role", "root", "--race",
        "--model", "malaiwah/GLM-5.2-SIQ-Fruit-bf16",
        "--panel", "o/p", "--lane", "streaming",
        "--dataset-id", "fruit-root-v1.preview",
        "--dataset-repository", "malaiwah/fruit-root-v1-preview",
        "--preview-of", "fruit-root-v1",
        "--publish-root-to", "malaiwah/fruit-root-v1-preview"])
    forbidden = MC._runpod_forbidden(race_args)
    check("RP10 safe controller rejects preview/race before provider access",
          any(item.startswith("--race") for item in forbidden)
          and any(item.startswith("--preview-of") for item in forbidden),
          forbidden)

    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        panel = Path(tmp) / "panel"
        (panel / "arrays").mkdir(parents=True)
        (panel / "panel.json").write_text(json.dumps({"panel_id": "panel--t"}))
        target_path, target = fruit_target_descriptor(tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CE.main([
                "capture", "--model", target["repo_id"],
                "--revision", target["revision"], "--lane", "streaming",
                "--target-descriptor", str(target_path), "--gpu", "L4",
                "--panel-dir", str(panel),
                "--dataset-id", "fruit-root-v1.preview",
                "--dataset-repository", "malaiwah/fruit-root-v1-preview",
                "--preview-of", "fruit-root-v1",
                "--race",
                "--publish-root-to", "malaiwah/fruit-root-v1-preview",
                "--workspace-available-bytes-minimum", "1",
                "--container-available-bytes-minimum", "1",
                "--expected-vram-bytes", "1",
                "--fs-root", str(Path(tmp) / "fs"), "--dry-run"])
        check("RP11 container composition also refuses preview/race roots",
              rc == CE.EXIT_REFUSED, "rc=%s\n%s" % (rc, buf.getvalue()[-800:]))
    # Controller-side create preflight is stdlib-only and has no mutation path.
    with tempfile.TemporaryDirectory() as tmp:
        token_file = Path(tmp) / "hf_token"
        original_endpoint = real_dshub.HF_ENDPOINT
        token_file.write_text("hf_selftest_token")
        token_file.chmod(0o600)
        preflight_calls = []
        original_get = real_dshub._get

        def absent_get(url, token=None, binary=False):
            preflight_calls.append((url, token))
            if url.endswith("/api/whoami-v2"):
                return json.dumps({"name": "malaiwah", "orgs": []})
            exc = real_dshub.HubError(
                "HTTP %d" % (401 if token is None else 404))
            exc.status = 401 if token is None else 404
            raise exc

        try:
            real_dshub.HF_ENDPOINT = "https://huggingface.co"
            real_dshub._get = absent_get
            evidence = real_dshub.preflight_create(
                "malaiwah/mm3-root-v1", str(token_file))
            destination_calls = [
                row for row in preflight_calls
                if not row[0].endswith("/api/whoami-v2")]
            check("RP9i stdlib create preflight seals principal and dual-view absence",
                  evidence.get("schema")
                  == "fidelity.hf-publish-create-preflight.v1"
                  and real_dshub.common.verify_seal(evidence)
                  and evidence.get("mutation_performed") is False
                  and evidence.get("authenticated_principal") == "malaiwah"
                  and len(destination_calls) == 6
                  and sum(token is None for _, token in destination_calls) == 3
                  and all(
                      probe == {
                          "authenticated_status": 404,
                          "anonymous_status": 401,
                      }
                      for probe in evidence.get("probes", {}).values()),
                  (evidence, destination_calls))

            def collision_get(url, token=None, binary=False):
                if url.endswith("/api/whoami-v2"):
                    return json.dumps({"name": "malaiwah", "orgs": []})
                if "/api/models/" in url and token is not None:
                    return "{}"
                exc = real_dshub.HubError("HTTP 404")
                exc.status = 404
                raise exc

            real_dshub._get = collision_get
            collision_refused = False
            try:
                real_dshub.preflight_create(
                    "malaiwah/mm3-root-v1", str(token_file))
            except real_dshub.HubError as exc:
                collision_refused = "collides" in str(exc)
            check("RP9j repo-type collision freezes create admission",
                  collision_refused)

            token_file.chmod(0o644)
            insecure_refused = False
            try:
                real_dshub.preflight_create(
                    "malaiwah/mm3-root-v1", str(token_file))
            except real_dshub.HubError as exc:
                insecure_refused = "0600" in str(exc)
            check("RP9k preflight refuses insecure token file before HTTP",
                  insecure_refused)
        finally:
            real_dshub._get = original_get
            real_dshub.HF_ENDPOINT = original_endpoint

    print()
    if FAILED:
        print("selftest_root_publish: %d FAILED" % len(FAILED))
        return 1
    print("selftest_root_publish: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
