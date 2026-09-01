"""Getting the answer off the box, for every verb -- not just published roots.

WHY THIS EXISTS
---------------
`container_entry` ended a successful run by saying

    all stages complete; receipts under /workspace/fidelity/receipts

which names a path on a filesystem the caller may have no way to read.  On the
SSH transport that was fine: the controller owned the box and pulled
`receipts.tar.gz` back over the same connection it opened.  A container has no
such connection, and the providers differ:

  * RunPod  -- pod-scoped volume, no sshd in our image, and the REST API
    (`/v1/pods`, `/v1/pods/{id}`, `.../billing`) exposes no logs and no files.
    The pod's stdout is visible in the web console and NOWHERE ELSE.
  * Vast    -- custom image, logs retrievable.
  * Lambda  -- a real VM: `docker run -v` and the results are already local.
  * laptop / k8s / CI -- a bind mount, or `docker logs`.

ROOT-1 solved exactly one case: a root CAPTURE, whose product is a multi-GB
dataset, uploaded to the Hub.  It solved nothing for the verb this project
exists to serve.  `measure` produces `receipts/measurement-receipt.json` -- a
few KB, and THE submission object the registry ingests -- and had no way home
at all.  Nor did `stage`, nor `doctor`, nor a FAILED run, whose receipts and
logs are the evidence you most want and least often can reach.

So the product is not "a dataset" or "a receipt"; it is "whatever this run
sealed, small or large", and the sink is chosen by the caller because only the
caller knows what they can read.

THE SCHEMES
-----------
    stdout                 frame the small artifacts into the container log
    file:PATH              copy the bundle to a path (a second mount, /workspace)
    https://... http://... PUT (or POST) the bundle to a URL the caller owns

`stdout` is ALWAYS delivered and cannot be switched off.  It is the only
channel that exists on every platform without configuration, it is what makes
a RunPod run legible at all, and it costs nothing.  Large payloads are not
dumped: over the cap the frame carries the summary and the digests, and says
what it withheld.

`https` is what makes this automatable without giving the box a credential
that can do anything else: a presigned S3/R2/GCS PUT, a collector endpoint, an
ntfy topic.  The URL is frequently ITSELF the secret, so it is registered for
redaction and read from the environment by preference -- never from argv,
which providers echo back in their consoles and API listings.

A dataset still publishes with `--publish-root-to`: multi-GB does not belong in
a log frame or a PUT body, and that path already re-verifies what it uploaded.

Stdlib only, python3.9-clean: this runs inside the entrypoint, which runs
before any venv is on PATH, and it is exercised on a laptop with no torch.
"""
from __future__ import annotations

import calendar
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

try:                                        # inside the suite
    from fidelity.common import (
        canonical_json, register_secret, safe_urlopen, seal, sha256_file,
        verify_seal,
    )
except Exception:                           # pragma: no cover - standalone
    def register_secret(value): return None

    def safe_urlopen(request, *, timeout=60.0):
        return urllib.request.urlopen(request, timeout=timeout)

    def canonical_json(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    def seal(doc, field="receipt_sha256"):
        body = dict(doc)
        body[field] = ""
        out = dict(doc)
        out[field] = hashlib.sha256(
            canonical_json(body).encode("utf-8")).hexdigest()
        return out

    def verify_seal(doc, field="receipt_sha256"):
        body = dict(doc)
        claimed = body.get(field, "")
        body[field] = ""
        return hashlib.sha256(
            canonical_json(body).encode("utf-8")).hexdigest() == claimed

    def sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
from fidelity import jobcontract, panel


#: stdout frame markers. Deliberately greppable and unlikely to occur in a log:
#: a scraper reads between them without parsing the surrounding chatter.
BEGIN = "===== FIDELITY-RESULT BEGIN ====="
END = "===== FIDELITY-RESULT END ====="

#: Above this, the frame carries digests instead of bytes. A measurement
#: receipt is ~4-40 KB; a per-window breakdown can be a few hundred. Providers
#: truncate long log lines, so the cap protects the SUMMARY from being pushed
#: out of the buffer by a payload nobody can use in that form anyway.
STDOUT_CAP_BYTES = 256 * 1024

#: Never leaves the box, on any sink.
EXCLUDE_DIRS = (".secrets", ".stream-work", ".cache", "__pycache__")
RESULT_MANIFEST_SCHEMA = "result-manifest-v1"
RESULT_MANIFEST_NAME = "result-manifest.json"
RESULT_SUMMARY_NAME = "result-summary.json"
RUN_STATE_NAME = "run-state.json"
ARCHIVE_NAME = "result-bundle.tar.gz"
QUANT_RECEIPT_SCHEMA = "quant-fidelity-registry/submission-receipt.v1"
TARGET_CENSUS_SCHEMA = "fidelity.fetch-target-census.v1"
TARGET_CENSUS_PATH = "receipts/fetch-target-census.json"
TARGET_CENSUS_KEYS = frozenset((
    "schema", "receipt_sha256", "verified_at", "job_id_full",
    "job_file_sha256", "repository", "revision", "config_sha256",
    "index_sha256", "shard_manifest_sha256", "model_bytes", "shards",
    "index_shards",
))
ARCHIVE_MARGIN_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 131072
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024
ROOT_QUALIFICATION_SCHEMA = "fidelity.root-qualification-receipt.v1"
RUNPOD_ATTESTATION_SCHEMA = "fidelity-suite/runpod-live-attestation.v2"
RUNPOD_ATTESTATION_PATH = "receipts/runpod-live-attestation.json"
RUNPOD_HOST_KEY_PROOF_SCHEMA = (
    "fidelity-suite/runpod-ssh-host-key-proof.v1")
RUNPOD_HOST_KEY_PROOF_PATH = "receipts/runpod-ssh-host-key-proof.json"
RUNPOD_HOST_KEY_PROOF_KEYS = frozenset((
    "schema", "provider", "provider_id", "verified_at_utc",
    "verification_source", "algorithm", "fingerprint", "host", "port",
    "known_hosts_sha256", "proof_sha256",
))
RUNPOD_ATTESTATION_KEYS = frozenset((
    "schema", "provider", "provider_id", "observed_at_utc", "clock",
    "expected", "observed", "transport_error", "checks", "failures", "ok",
    "attestation_sha256",
))
RUNPOD_CLOCK_KEYS = frozenset((
    "controller_send_epoch", "controller_send_utc",
    "controller_receive_epoch", "controller_receive_utc",
    "round_trip_seconds", "remote_time_epoch", "remote_time_utc",
    "clock_skew_seconds", "allowed_skew_seconds", "within_bound",
))
RUNPOD_EXPECTED_KEYS = frozenset((
    "expected_vram_bytes", "min_vcpu", "min_ram_gb", "volume_gb",
    "container_disk_gb", "workspace_available_bytes_minimum",
    "container_available_bytes_minimum", "gpu_model",
))
MAX_RETAINED_MEMBER_BYTES = 32 * 1024 * 1024
MAX_RETAINED_METADATA_BYTES = 128 * 1024 * 1024
RESOURCE_REQUIREMENT_KEYS = frozenset((
    "workspace_available_bytes_minimum",
    "container_available_bytes_minimum",
    "min_vcpu_count", "min_memory_gb", "expected_vram_bytes",
))
QUANT_RECEIPT_KEYS = frozenset((
    "submission_schema", "receipt_sha256", "produced_by", "measured_at",
    "lane", "measurer", "artifact", "panel", "reference",
    "metric", "auxiliary_metrics", "estimator", "determinism",
    "measurement_scope", "environment", "cost", "evidence", "disclosures",
))
ROOT_PUBLICATION_SCHEMA = "fidelity.publish-root-receipt.v2"

ROOT_CAPTURE_IDENTITY_FIELDS = frozenset((
    "process_label", "dataset_id", "dataset_name", "dataset_author",
    "dataset_repository", "dataset_sha256",
    "dataset_manifest_file_sha256", "capture_manifest",
    "capture_manifest_sha256", "capture_content_digest", "capture_form",
    "capture_dtype", "runtime_manifest", "runtime_manifest_sha256",
    "runtime_lane", "runtime_device", "runtime_engine", "runtime_container",
    "capture_tool_file", "capture_schedule", "panel",
    "unexpected_tensor_allowlist", "stack_fingerprint_sha256",
    "lane_identity_sha256", "weights_repository", "weights_revision",
    "determinism_run_count",
))
MAX_IN_MEMORY_ARCHIVE_BYTES = 256 * 1024 * 1024

class ArchiveError(RuntimeError):
    """A result archive is incomplete, unsafe, or fails its cryptographic identity."""




class SinkError(RuntimeError):
    """A sink could not be parsed or could not be delivered."""


class Sink(object):
    __slots__ = ("scheme", "target", "raw")

    def __init__(self, scheme, target, raw):
        self.scheme, self.target, self.raw = scheme, target, raw

    def __repr__(self):                     # never prints a presigned URL
        return "Sink(%s)" % self.scheme


def parse_sinks(values, env=None):
    """Build the sink list from --result-sink values plus the environment.

    FIDELITY_RESULT_SINK is comma-separated and is the PREFERRED channel for a
    URL that carries its own credential: `runpodapi.create` puts env in `env`
    and the command in `dockerArgs`, and only the latter is echoed back by the
    provider's API.
    """
    env = os.environ if env is None else env
    raw = list(values or [])
    from_env = (env.get("FIDELITY_RESULT_SINK") or "").strip()
    if from_env:
        raw.extend(part.strip() for part in from_env.split(",") if part.strip())

    sinks, seen = [], set()
    for item in raw:
        if item in seen:
            continue
        seen.add(item)
        low = item.lower()
        if low in ("stdout", "stdout:", "-"):
            continue                        # always present; see below
        if low.startswith("file:"):
            sinks.append(Sink("file", item[len("file:"):], item))
        elif low.startswith("https://") or low.startswith("http://"):
            register_secret(item)           # a presigned URL IS the credential
            sinks.append(Sink("http", item, item))
        elif low.startswith("hf://"):
            raise SinkError(
                "hf:// is not a result sink. A sealed DATASET publishes with "
                "--publish-root-to, which re-verifies the uploaded copy; a "
                "receipt is a file, so use file: or an https: endpoint.")
        else:
            raise SinkError(
                "unknown result sink %r. Known: stdout, file:PATH, "
                "https://URL (PUT)." % item)
    # stdout is unconditional and first: if a later sink raises, the answer has
    # already been printed. That ordering is the whole point.
    return [Sink("stdout", "", "stdout")] + sinks

#: Per-log tail, in bytes. A stage log is mostly a progress meter; the part
#: that says why a run died is at the end. Whole logs would put a 200 MB
#: `Loading weights:` bar through a PUT body.
LOG_TAIL_BYTES = 64 * 1024


def _relevant(fs_root, include_datasets=False):
    """Return every deliverable regular file without following links."""
    root = Path(fs_root)
    out = []
    trees = ["receipts", "reports", "control", "logs"]
    if include_datasets:
        trees.extend(("dataset", "dataset-repeat"))
    for sub in trees:
        base = root / sub
        if not base.exists():
            if sub in ("dataset", "dataset-repeat"):
                raise ArchiveError("completed root result requires %s/" % sub)
            continue
        if base.is_symlink() or not base.is_dir():
            raise ArchiveError("%s must be a real directory" % sub)
        before = len(out)
        for directory, names, files in os.walk(str(base), followlinks=False):
            names[:] = sorted(name for name in names if name not in EXCLUDE_DIRS)
            for name in list(names):
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    raise ArchiveError("refusing linked result directory %s"
                                       % candidate.relative_to(root).as_posix())
            for name in sorted(files):
                path = Path(directory) / name
                rel = path.relative_to(root)
                if any(part in EXCLUDE_DIRS for part in rel.parts):
                    continue
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ArchiveError("refusing non-regular result member %s"
                                       % rel.as_posix())
                out.append(path)
        if sub in ("dataset", "dataset-repeat") and len(out) == before:
            raise ArchiveError("completed root result has empty %s/" % sub)
    for name in ("job.json", "ABANDONED.json", ".done"):
        path = root / name
        if not path.exists():
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ArchiveError("refusing non-regular result member %s" % name)
        out.append(path)
    if len(out) > MAX_ARCHIVE_MEMBERS:
        raise ArchiveError("result exceeds archive member safety limit")
    source_bytes = sum(path.lstat().st_size for path in out)
    if source_bytes > MAX_ARCHIVE_BYTES:
        raise ArchiveError("result exceeds archive source-byte safety limit")
    return sorted(out, key=lambda path: path.relative_to(root).as_posix())


def _payload_record(path, fs_root, retain=True):
    """Hash exact delivered bytes without following links or trusting one stat."""
    rel = path.relative_to(fs_root).as_posix()
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    chunks = [] if retain else None
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_size != before.st_size):
            raise ArchiveError("result source changed before read: %s" % rel)
        source_bytes = opened.st_size
        omitted = 0
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            if rel.startswith("logs/") and source_bytes > LOG_TAIL_BYTES:
                omitted = source_bytes - LOG_TAIL_BYTES
                stream.seek(omitted)
                tail = stream.read(LOG_TAIL_BYTES)
                head = (
                    b"[... %d earlier bytes omitted; this is the last %d ...]\n"
                    % (omitted, LOG_TAIL_BYTES))
                data = head + tail
                digest = hashlib.sha256(data).hexdigest()
                delivered_bytes = len(data)
                if not retain:
                    data = None
            else:
                hasher = hashlib.sha256()
                delivered_bytes = 0
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(chunk)
                    delivered_bytes += len(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                digest = hasher.hexdigest()
                data = b"".join(chunks) if chunks is not None else None
        after = path.lstat()
        if (after.st_dev != before.st_dev or after.st_ino != before.st_ino
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns):
            raise ArchiveError("result source changed during read: %s" % rel)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    record = {
        "path": rel,
        "bytes": delivered_bytes,
        "sha256": digest,
        "source_bytes": source_bytes,
        "delivery": "tail" if omitted else "whole",
    }
    if omitted:
        record["omitted_prefix_bytes"] = omitted
    return data, record


def _payload(path, fs_root):
    """Compatibility helper returning the exact bytes put in the archive."""
    return _payload_record(path, Path(fs_root))[0]


def build_summary(fs_root, verb, status, stages, pin=None, failed_stage=None):
    """Build the stdout summary over delivered bytes, not pre-tail source bytes."""
    fs_root = Path(fs_root)
    include_datasets = (
        verb == "capture"
        and str(status).lower() in (
            "ok", "complete", "completed", "success",
            "qualified-unpublished", "completed-operational-failure")
        and _load_job(fs_root).get("role") == "root")
    files = []
    for path in _relevant(fs_root, include_datasets=include_datasets):
        _body, record = _payload_record(path, fs_root, retain=False)
        files.append(record)
    return {
        "schema": "malaiwah.fidelity-result-summary.v1",
        "verb": verb,
        "status": status,
        "failed_stage": failed_stage,
        "stages": list(stages or []),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": dict(pin or {}),
        "files": files,
    }


def _json_bytes(doc):
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _verify_job_contract(job):
    try:
        return jobcontract.verify_job(job)
    except (jobcontract.JobContractError, TypeError, ValueError) as exc:
        raise ArchiveError("job.json identity is invalid: %s" % exc)

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key %r" % key)
        result[key] = value
    return result


def _strict_json_loads(raw, label):
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(
            raw, object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("non-finite JSON token %s" % token)))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ArchiveError("%s is not strict UTF-8 JSON: %s"
                           % (label, exc))



def _load_job(fs_root):
    path = Path(fs_root) / "job.json"
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("result archive requires a regular job.json")
    try:
        job = _strict_json_loads(
            path.read_text(encoding="utf-8"), "job.json")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError("job.json is not valid UTF-8 JSON: %s"
                           % exc.__class__.__name__)
    if not isinstance(job, dict):
        raise ArchiveError("job.json must contain an object")
    _verify_job_contract(job)
    return job


def _require_sealed_receipt(fs_root, relative, purpose):
    path = Path(fs_root) / relative
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("%s requires %s" % (purpose, relative))
    try:
        doc = _strict_json_loads(
            path.read_text(encoding="utf-8"), relative)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError("%s is not valid UTF-8 JSON: %s"
                           % (relative, exc.__class__.__name__))
    if not isinstance(doc, dict) or not verify_seal(doc):
        raise ArchiveError("%s does not carry a valid receipt_sha256 self-seal"
                           % relative)
    return doc


def _require_abandoned_state(fs_root):
    path = Path(fs_root) / "ABANDONED.json"
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("abandoned result requires ABANDONED.json")
    try:
        doc = _strict_json_loads(
            path.read_text(encoding="utf-8"), "ABANDONED.json")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError("ABANDONED.json is not valid UTF-8 JSON: %s"
                           % exc.__class__.__name__)
    if (not isinstance(doc, dict)
            or doc.get("schema") != "fidelity-suite/abandoned.v2"):
        raise ArchiveError("ABANDONED.json has the wrong schema")
    return doc


def _status_kind(summary):
    status = str(summary.get("status") or "").lower()
    if status in ("ok", "complete", "completed", "success", "succeeded"):
        return status, True, False
    if status == "completed-operational-failure":
        return status, True, True
    if status == "qualified-unpublished":
        return status, False, True
    if status in ("failed", "failure", "abandoned"):
        return status, False, True
    raise ArchiveError(
        "result status %r is neither completed nor failed/abandoned"
        % summary.get("status"))


def _doctor_contract(fs_root, status):
    path = Path(fs_root) / "receipts" / "doctor.json"
    if not path.is_file() or path.is_symlink():
        raise ArchiveError("doctor result requires receipts/doctor.json")
    try:
        doc = _strict_json_loads(
            path.read_text(encoding="utf-8"), "receipts/doctor.json")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArchiveError("receipts/doctor.json is not valid UTF-8 JSON: %s"
                           % exc.__class__.__name__)
    expected = "ok" if status == "ok" else "failed"
    if (not isinstance(doc, dict)
            or doc.get("schema") != "malaiwah.fidelity-doctor.v1"
            or doc.get("status") != expected):
        raise ArchiveError("receipts/doctor.json disagrees with doctor status")


def _stage_contract(fs_root, summary, complete, failed):
    stages = list(summary.get("stages") or [])
    if (len(stages) != 1 or not isinstance(stages[0], str)
            or not stages[0] or "/" in stages[0] or "\\" in stages[0]):
        raise ArchiveError("stage result must name exactly one safe stage")
    stage = stages[0]
    log = Path(fs_root) / "logs" / ("%s.log" % stage)
    if not log.is_file() or log.is_symlink():
        raise ArchiveError("stage result requires logs/%s.log" % stage)
    if complete:
        done = Path(fs_root) / "receipts" / "done" / ("%s.done" % stage)
        if not done.is_file() or done.is_symlink():
            raise ArchiveError(
                "completed stage requires receipts/done/%s.done" % stage)
    if failed and summary.get("failed_stage") != stage:
        raise ArchiveError("failed stage result must name its failed_stage")


def _role_contract(fs_root, summary, source_paths):
    """Fail closed on the verb/status/role-specific evidence contract."""
    verb = str(summary.get("verb") or "")
    status, complete, failed = _status_kind(summary)
    qualified_unpublished = status == "qualified-unpublished"
    if verb == "doctor":
        if status not in ("ok", "failed"):
            raise ArchiveError("doctor status must be ok or failed")
        _doctor_contract(fs_root, status)
        return "doctor", status, False

    job = _load_job(fs_root)
    role = job.get("role")
    if role not in ("quant", "root"):
        raise ArchiveError("job.json role must be quant or root")
    publication_requested = (
        isinstance(job.get("capture"), dict)
        and job["capture"].get("publish_root_to") is not None)
    _check_local_runpod_attestation(fs_root, job)
    _check_local_target_census(fs_root, summary, job, complete)
    if verb == "stage":
        _stage_contract(fs_root, summary, complete, failed)
        if status == "abandoned":
            _require_abandoned_state(fs_root)
        return role, status, publication_requested
    if verb not in ("measure", "capture"):
        raise ArchiveError("unsupported result verb %r" % verb)
    if verb == "measure" and role != "quant":
        raise ArchiveError("measure result requires quant role")
    if verb == "capture" and role != "root":
        raise ArchiveError("capture result requires root role")

    if complete and verb == "measure":
        measurement = _require_sealed_receipt(
            fs_root, "receipts/measurement-receipt.json",
            "completed quant measurement")
        if measurement.get("submission_schema") != QUANT_RECEIPT_SCHEMA:
            raise ArchiveError(
                "measurement receipt has the wrong submission_schema")
        _validate_quant_evidence(job, measurement)
        _validate_local_quant_reports(source_paths, measurement)
    if (complete or qualified_unpublished) and verb == "capture":
        qualification = _require_sealed_receipt(
            fs_root, "receipts/root-qualification.json",
            "qualified root capture")
        if qualification.get("schema") != ROOT_QUALIFICATION_SCHEMA:
            raise ArchiveError("root qualification receipt has the wrong schema")
        published = None
        publication_path = Path(fs_root) / "receipts" / "publish-root.json"
        if qualified_unpublished and publication_path.exists():
            raise ArchiveError(
                "qualified-unpublished result cannot carry publication receipt")
        if complete and publication_requested:
            published = _require_sealed_receipt(
                fs_root, "receipts/publish-root.json",
                "published completed root result")
            if published.get("schema") != ROOT_PUBLICATION_SCHEMA:
                raise ArchiveError("root publication receipt has the wrong schema")
        qualification_file_sha = sha256_file(
            str(Path(fs_root) / "receipts" / "root-qualification.json"))
        _validate_root_evidence(
            job, qualification, published, qualification_file_sha,
            sha256_file(str(Path(fs_root) / "job.json")))
    if failed:
        if not any(path.relative_to(Path(fs_root)).as_posix().startswith("logs/")
                   for path in source_paths):
            raise ArchiveError("failed/abandoned result requires a stage log")
        if status == "abandoned":
            _require_abandoned_state(fs_root)
    return role, status, publication_requested


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info
class _HashingReader:
    def __init__(self, source):
        self.source = source
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        body = self.source.read(size)
        self.hasher.update(body)
        self.bytes_read += len(body)
        return body

    def hexdigest(self):
        return self.hasher.hexdigest()

def _archive_parts(fs_root, summary, stream=False):
    root = Path(fs_root)
    verb = str(summary.get("verb") or "")
    _normalized_status, complete, _failed = _status_kind(summary)
    qualified_unpublished = (
        str(summary.get("status") or "").lower() == "qualified-unpublished")
    include_datasets = (
        verb == "capture" and (complete or qualified_unpublished)
        and _load_job(root).get("role") == "root")
    source_paths = _relevant(root, include_datasets=include_datasets)
    role, status, publication_requested = _role_contract(
        root, summary, source_paths)
    parts = []
    records = []
    bodies = {}
    digests = {}
    summary_records = summary.get("files")
    if not isinstance(summary_records, list):
        raise ArchiveError("result summary files must be an array")
    summary_by_name = {
        record.get("path"): record for record in summary_records
        if isinstance(record, dict)
    }
    if len(summary_by_name) != len(summary_records):
        raise ArchiveError("result summary has malformed/duplicate file paths")
    retained_bytes = 0
    for path in source_paths:
        relative = path.relative_to(root).as_posix()
        retain = (not stream or relative.endswith(".json")
                  or relative.endswith("/checksums.txt")
                  or relative.startswith("logs/"))
        if stream and not retain:
            record = summary_by_name.get(relative)
            current = path.lstat()
            if (not isinstance(record, dict)
                    or set(record) != {
                        "path", "bytes", "sha256", "source_bytes", "delivery"}
                    or record.get("path") != relative
                    or record.get("delivery") != "whole"
                    or record.get("bytes") != current.st_size
                    or record.get("source_bytes") != current.st_size
                    or not _valid_hex(record.get("sha256"), 64)
                    or not stat.S_ISREG(current.st_mode)):
                raise ArchiveError(
                    "result source differs from sealed summary: %s" % relative)
            body = None
        else:
            body, record = _payload_record(path, root, retain=retain)
            if stream:
                retained_bytes += len(body)
                if (len(body) > MAX_RETAINED_MEMBER_BYTES
                        or retained_bytes > MAX_RETAINED_METADATA_BYTES):
                    raise ArchiveError(
                        "retained archive metadata exceeds memory safety cap")
        parts.append((record["path"], body, path if body is None else None))
        records.append(record)
        if (not stream
                and sum(item["bytes"] for item in records)
                > MAX_IN_MEMORY_ARCHIVE_BYTES):
            raise ArchiveError(
                "build_archive refuses large in-memory bundles; use write_archive")
        bodies[record["path"]] = body
        digests[record["path"]] = record["sha256"]
    for name, body in bodies.items():
        if name.endswith(".json") and isinstance(body, bytes):
            _strict_json_loads(body, name)
    if summary.get("files") != records:
        raise ArchiveError(
            "result sources changed after summary construction; rebuild the "
            "summary before archiving")
    archive_job = None if role == "doctor" else _load_job(root)
    if role == "root" and (complete or qualified_unpublished):
        qualification = _require_sealed_receipt(
            root, "receipts/root-qualification.json",
            "completed root capture")
        _validate_qualified_dataset_bodies(
            qualification, bodies, digests, job=archive_job)
    job_id_full = None
    measurement_receipt_sha256 = None
    if archive_job is not None:
        job_id_full = archive_job["job_id_full"]
    if verb == "measure" and complete:
        measurement_receipt_sha256 = _require_sealed_receipt(
            root, "receipts/measurement-receipt.json",
            "completed quant measurement")["receipt_sha256"]
    state = {
        "schema": "result-state-v1",
        "verb": str(summary.get("verb") or ""),
        "job_id_full": job_id_full,
        "measurement_receipt_sha256": measurement_receipt_sha256,
        "role": role,
        "status": status,
        "failed_stage": summary.get("failed_stage"),
        "stages": list(summary.get("stages") or []),
        "publication_requested": publication_requested,
    }
    for name, body in (
            (RESULT_SUMMARY_NAME, _json_bytes(summary)),
            (RUN_STATE_NAME, _json_bytes(state))):
        parts.append((name, body, None))
        records.append({
            "path": name, "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "source_bytes": len(body), "delivery": "whole",
        })
    records.sort(key=lambda item: item["path"])
    manifest = seal({
        "schema": RESULT_MANIFEST_SCHEMA,
        "manifest_sha256": "",
        "verb": str(summary.get("verb") or ""),
        "job_id_full": job_id_full,
        "measurement_receipt_sha256": measurement_receipt_sha256,
        "role": role,
        "status": status,
        "publication_requested": publication_requested,
        "files": records,
    }, field="manifest_sha256")
    manifest_body = _json_bytes(manifest)
    parts.sort(key=lambda item: item[0])
    parts.append((RESULT_MANIFEST_NAME, manifest_body, None))
    if role == "root" and (complete or qualified_unpublished):
        _enforce_root_archive_caps(
            _load_job(root), len(parts),
            sum(record["bytes"] for record in records) + len(manifest_body))
    if role == "quant" and complete:
        _enforce_quant_archive_caps(
            _load_job(root), len(parts),
            sum(record["bytes"] for record in records) + len(manifest_body))
    return parts, manifest


def build_archive(fs_root, summary):
    """Return a small deterministic gzip tar; large bundles must stream."""
    if sum(int(record.get("bytes", 0))
           for record in (summary.get("files") or [])) > MAX_IN_MEMORY_ARCHIVE_BYTES:
        raise ArchiveError(
            "build_archive refuses large in-memory bundles; use write_archive")
    parts, _manifest = _archive_parts(fs_root, summary)
    compressed = io.BytesIO()
    with gzip.GzipFile(
            filename="", mode="wb", fileobj=compressed, mtime=0,
            compresslevel=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, body, _path in parts:
                tar.addfile(_tar_info(name, len(body)), io.BytesIO(body))
    archive_body = compressed.getvalue()
    if (summary.get("verb") == "capture"
            and (_status_kind(summary)[1]
                 or summary.get("status") == "qualified-unpublished")
            and _load_job(fs_root).get("role") == "root"):
        _enforce_root_archive_caps(
            _load_job(fs_root), len(parts),
            sum(len(body) for _name, body, _path in parts),
            len(archive_body))
    if (summary.get("verb") == "measure"
            and _status_kind(summary)[1]
            and _load_job(fs_root).get("role") == "quant"):
        _enforce_quant_archive_caps(
            _load_job(fs_root), len(parts),
            sum(len(body) for _name, body, _path in parts),
            len(archive_body))
    return archive_body


def _bundle(fs_root, summary):
    """Backward-compatible private name for the canonical result archive."""
    return build_archive(fs_root, summary)


def _fsync_directory(path):
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ArchiveError("cannot open result directory safely: %s"
                           % exc.__class__.__name__)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path):
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ArchiveError("result destination parent is not a safe directory")
    for directory in reversed(missing):
        try:
            os.mkdir(str(directory), 0o700)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise ArchiveError(
                    "result destination directory was replaced unsafely")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    _fsync_directory(path)


def _write_exclusive_durable(path, body):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(Path(path).parent)
def _stream_exclusive_durable(path, source):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            for chunk in iter(lambda: source.read(1 << 20), b""):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(Path(path).parent)




def write_archive(fs_root, summary, output_path):
    """Stream a deterministic archive to disk and atomically publish it."""
    root = Path(fs_root)
    parts, manifest = _archive_parts(root, summary, stream=True)
    records = {record["path"]: record for record in manifest["files"]}
    destination = Path(output_path)
    _ensure_durable_directory(destination.parent)
    handle, temporary = tempfile.mkstemp(
        dir=str(destination.parent), prefix=".result-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as output:
            with gzip.GzipFile(
                    filename="", mode="wb", fileobj=output, mtime=0,
                    compresslevel=0) as zipped:
                with tarfile.open(
                        fileobj=zipped, mode="w",
                        format=tarfile.PAX_FORMAT) as archive:
                    for name, body, source_path in parts:
                        if body is not None:
                            archive.addfile(
                                _tar_info(name, len(body)), io.BytesIO(body))
                            continue
                        record = records[name]
                        before = source_path.lstat()
                        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                        descriptor = os.open(str(source_path), flags)
                        try:
                            opened = os.fstat(descriptor)
                            if (not stat.S_ISREG(opened.st_mode)
                                    or opened.st_dev != before.st_dev
                                    or opened.st_ino != before.st_ino
                                    or opened.st_size != record["bytes"]):
                                raise ArchiveError(
                                    "result source changed before archive: %s"
                                    % name)
                            with os.fdopen(descriptor, "rb") as source_file:
                                descriptor = -1
                                reader = _HashingReader(source_file)
                                archive.addfile(
                                    _tar_info(name, record["bytes"]), reader)
                            after = source_path.lstat()
                            if (reader.bytes_read != record["bytes"]
                                    or reader.hexdigest() != record["sha256"]
                                    or after.st_dev != before.st_dev
                                    or after.st_ino != before.st_ino
                                    or after.st_size != before.st_size
                                    or after.st_mtime_ns != before.st_mtime_ns):
                                raise ArchiveError(
                                    "result source changed during archive: %s"
                                    % name)
                        finally:
                            if descriptor >= 0:
                                os.close(descriptor)
            output.flush()
            os.fsync(output.fileno())
        archive_bytes = os.stat(temporary).st_size
        archive_sha256 = sha256_file(temporary)
        if (summary.get("verb") == "capture"
                and (_status_kind(summary)[1]
                     or summary.get("status") == "qualified-unpublished")
                and _load_job(root).get("role") == "root"):
            _enforce_root_archive_caps(
                _load_job(root), len(parts),
                sum(len(body) if body is not None else records[name]["bytes"]
                    for name, body, _path in parts),
                archive_bytes)
        if (summary.get("verb") == "measure"
                and _status_kind(summary)[1]
                and _load_job(root).get("role") == "quant"):
            _enforce_quant_archive_caps(
                _load_job(root), len(parts),
                sum(len(body) if body is not None else records[name]["bytes"]
                    for name, body, _path in parts),
                archive_bytes)
        os.replace(temporary, str(destination))
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "path": str(destination),
        "bytes": archive_bytes,
        "sha256": archive_sha256,
    }
def _archive_source(source):
    if isinstance(source, (bytes, bytearray)):
        body = bytes(source)
        if len(body) > MAX_ARCHIVE_BYTES:
            raise ArchiveError("result archive exceeds transfer-byte safety limit")
        return body, len(body), hashlib.sha256(body).hexdigest()
    path = Path(source)
    try:
        size = path.stat().st_size
        if size > MAX_ARCHIVE_BYTES:
            raise ArchiveError("result archive exceeds transfer-byte safety limit")
        digest = sha256_file(str(path))
        return path, size, digest
    except OSError as exc:
        raise ArchiveError("cannot read result archive: %s"
                           % exc.__class__.__name__)


def _safe_member_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ArchiveError("unsafe archive member name %r" % name)
    path = PurePosixPath(name)
    canonical = path.as_posix()
    if (path.is_absolute() or canonical != name
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ArchiveError("unsafe archive member name %r" % name)
    return canonical


def _parse_json_member(name, body):
    doc = _strict_json_loads(body, name)
    if not isinstance(doc, dict):
        raise ArchiveError("%s must contain an object" % name)
    return doc


def _valid_hex(value, length):
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def _member_sha256(name, bodies, digests=None):
    body = bodies.get(name)
    if isinstance(body, bytes):
        return hashlib.sha256(body).hexdigest()
    if digests is not None and name in digests:
        return digests[name]
    raise ArchiveError("cannot resolve delivered member digest for %s" % name)


def _dataset_path(prefix, relative):
    relative = _safe_member_name(relative)
    return "%s/%s" % (prefix, relative)


def _validate_bound_panel_receipt(prefix, manifest_panel, binding,
                                  bodies, digests=None):
    receipt_binding = (
        binding.get("receipt") if isinstance(binding, dict) else None)
    if not isinstance(receipt_binding, dict):
        raise ArchiveError(
            "%s resolved panel receipt binding is missing" % prefix)
    relative = "panel/panel-receipt.json"
    if (not isinstance(manifest_panel, dict)
            or manifest_panel.get("panel_receipt_file") != relative
            or manifest_panel.get("panel_receipt_sha256")
                != receipt_binding.get("declared_receipt_sha256")):
        raise ArchiveError(
            "%s manifest does not name the exact bound panel receipt" % prefix)
    name = _dataset_path(prefix, relative)
    body = bodies.get(name)
    if not isinstance(body, bytes):
        raise ArchiveError("%s bound panel receipt bytes are missing" % prefix)
    try:
        panel.verify_bound_panel_receipt_bytes(
            receipt_binding, body, "%s panel receipt" % prefix)
    except panel.PanelError as exc:
        raise ArchiveError(str(exc)) from exc


def _validate_dataset_tree(prefix, identity, bodies, digests=None,
                           contract=None):
    if not isinstance(identity, dict):
        raise ArchiveError("qualification %s capture identity is missing" % prefix)
    if set(identity) != ROOT_CAPTURE_IDENTITY_FIELDS:
        raise ArchiveError(
            "qualification %s capture identity fields differ" % prefix)
    expected_process = (
        "root-cold-1" if prefix == "dataset" else "root-cold-2")
    if (identity.get("process_label") != expected_process
            or identity.get("capture_form") != "hidden"
            or identity.get("determinism_run_count") != 1
            or any(identity.get(field) is None for field in (
                "dataset_id", "dataset_name", "dataset_author",
                "dataset_repository", "capture_dtype", "runtime_lane",
                "runtime_device", "runtime_engine", "runtime_container",
                "capture_tool_file", "capture_schedule", "panel",
                "unexpected_tensor_allowlist", "weights_repository",
                "weights_revision"))):
        raise ArchiveError(
            "qualification %s capture identity is incomplete" % prefix)
    if not isinstance(contract, dict):
        raise ArchiveError(
            "%s root qualification job contract is missing" % prefix)
    target = contract.get("target")
    profile = contract.get("profile")
    binding = contract.get("panel_resolved_binding")
    panel_identity = identity.get("panel")
    expected_dtype = {
        "bfloat16": "BF16", "bf16": "BF16",
    }.get(str(contract.get("dtype")).lower())
    expected_binding_evidence = {
        "binding_file": PurePosixPath(
            contract.get("panel_binding_path", "")).name,
        "binding_file_sha256": contract.get("panel_binding_file_sha256"),
        "binding": binding,
    }
    expected_allowlist = contract.get("unexpected_tensor_allowlist")
    observed_allowlist = identity.get("unexpected_tensor_allowlist")
    bound_panel = binding.get("panel") if isinstance(binding, dict) else None
    bound_receipt = (
        binding.get("receipt") if isinstance(binding, dict) else None)
    bound_tokenizer = (
        binding.get("tokenizer") if isinstance(binding, dict) else None)
    if (not isinstance(target, dict) or not isinstance(profile, dict)
            or not isinstance(panel_identity, dict)
            or identity.get("dataset_id") != contract.get("dataset_id")
            or identity.get("dataset_name") != contract.get("dataset_name")
            or identity.get("dataset_author") != contract.get("author")
            or identity.get("dataset_repository")
                != contract.get("dataset_repository")
            or identity.get("weights_repository") != target.get("repo_id")
            or identity.get("weights_revision") != target.get("revision")
            or identity.get("runtime_lane") != contract.get("lane")
            or identity.get("runtime_device") != contract.get("device")
            or identity.get("runtime_engine") != "transformers-eager"
            or identity.get("capture_form") != contract.get("form")
            or identity.get("capture_dtype") != expected_dtype
            or identity.get("capture_schedule") != contract.get("schedule")
            or identity.get("capture_tool_file")
                != "engines/tools/hf_capture.py"
            or not isinstance(bound_panel, dict)
            or not isinstance(bound_receipt, dict)
            or not isinstance(bound_tokenizer, dict)
            or panel_identity.get("panel_id") != bound_panel.get("id")
            or panel_identity.get("suite_token_hash_sha256")
                != bound_panel.get("suite_token_hash_sha256")
            or panel_identity.get("panel_receipt_sha256")
                != bound_receipt.get("declared_receipt_sha256")
            or panel_identity.get("tokenizer") != bound_tokenizer
            or panel_identity.get("resolved_binding_evidence")
                != expected_binding_evidence):
        raise ArchiveError(
            "%s capture identity differs from the exact root job" % prefix)
    expected_keys = (
        observed_allowlist.get("expected_keys")
        if isinstance(observed_allowlist, dict) else None)
    if (not isinstance(expected_allowlist, dict)
            or not isinstance(expected_keys, list)
            or not expected_keys
            or any(not isinstance(name, str) or not name
                   for name in expected_keys)
            or len(expected_keys) != len(set(expected_keys))
            or hashlib.sha256(
                canonical_json(sorted(expected_keys)).encode("utf-8")
            ).hexdigest()
                != expected_allowlist.get(
                    "canonical_sorted_names_sha256")):
        raise ArchiveError(
            "%s capture allowlist names differ from the exact root job"
            % prefix)
    if (not isinstance(expected_allowlist, dict)
            or not isinstance(observed_allowlist, dict)
            or observed_allowlist.get("artifact_sha256")
                != expected_allowlist.get("artifact_sha256")
            or observed_allowlist.get("canonical_sorted_names_sha256")
                != expected_allowlist.get("canonical_sorted_names_sha256")
            or observed_allowlist.get("exact_match") is not True
            or observed_allowlist.get("observed_keys")
                != observed_allowlist.get("expected_keys")
            or observed_allowlist.get("duplicate_observed_keys") != []
            or observed_allowlist.get("missing_keys") != []
            or observed_allowlist.get("extra_keys") != []):
        raise ArchiveError(
            "%s capture allowlist differs from the exact root job" % prefix)
    manifest_name = "%s/fidelity-dataset.json" % prefix
    manifest_body = bodies.get(manifest_name)
    if not isinstance(manifest_body, bytes):
        raise ArchiveError("%s lacks fidelity-dataset.json" % prefix)
    manifest = _parse_json_member(manifest_name, manifest_body)
    declared_dataset_sha = identity.get("dataset_sha256")
    raw_manifest_sha = hashlib.sha256(manifest_body).hexdigest()
    blanked = dict(manifest)
    blanked["dataset_sha256"] = ""
    if (not _valid_hex(declared_dataset_sha, 64)
            or manifest.get("dataset_sha256") != declared_dataset_sha
            or hashlib.sha256(
                canonical_json(blanked).encode("utf-8")).hexdigest()
            != declared_dataset_sha
            or identity.get("dataset_manifest_file_sha256")
            != raw_manifest_sha):
        raise ArchiveError("%s dataset manifest seal/hash differs from qualification"
                           % prefix)
    manifest_dataset = manifest.get("dataset")
    manifest_weights = manifest.get("weights")
    manifest_panel = manifest.get("panel")
    manifest_capture_block = manifest.get("capture")
    manifest_runtime_block = manifest.get("runtime")
    if (not isinstance(manifest_dataset, dict)
            or not isinstance(manifest_weights, dict)
            or not isinstance(manifest_panel, dict)
            or not isinstance(manifest_capture_block, dict)
            or not isinstance(manifest_runtime_block, dict)
            or manifest_dataset.get("id") != contract.get("dataset_id")
            or manifest_dataset.get("name") != contract.get("dataset_name")
            or (manifest_dataset.get("author") or {}).get("name")
                != contract.get("author")
            or manifest_dataset.get("repository")
                != contract.get("dataset_repository")
            or manifest_dataset.get("role") != "root"
            or manifest_weights.get("repository") != target.get("repo_id")
            or manifest_weights.get("revision") != target.get("revision")
            or manifest_weights.get("model_revision") != target.get("revision")
            or manifest_weights.get("quantized") is not False
            or manifest_panel.get("panel_id") != bound_panel.get("id")
            or manifest_panel.get("suite_token_hash_sha256")
                != bound_panel.get("suite_token_hash_sha256")
            or manifest_panel.get("panel_receipt_sha256")
                != bound_receipt.get("declared_receipt_sha256")
            or manifest_panel.get("tokenizer") != bound_tokenizer
            or manifest_capture_block.get("form") != contract.get("form")
            or manifest_capture_block.get("dtype") != expected_dtype
            or manifest_runtime_block.get("lane") != contract.get("lane")
            or manifest_runtime_block.get("source") != "native"):
        raise ArchiveError(
            "%s top manifest inputs differ from the exact root job" % prefix)
    checksums_name = "%s/checksums.txt" % prefix
    checksums_body = bodies.get(checksums_name)
    seal_block = manifest.get("seal")
    if (not isinstance(checksums_body, bytes)
            or not isinstance(seal_block, dict)
            or seal_block.get("checksums_file", "checksums.txt") != "checksums.txt"
            or seal_block.get("checksums_sha256")
            != hashlib.sha256(checksums_body).hexdigest()):
        raise ArchiveError("%s checksums proof is missing or mismatched" % prefix)
    checked = set()
    try:
        lines = checksums_body.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ArchiveError("%s checksums.txt is not UTF-8" % prefix) from exc
    for line in lines:
        if not line:
            continue
        fields = line.split("  ", 1)
        if len(fields) != 2 or not _valid_hex(fields[0], 64):
            raise ArchiveError("%s checksums.txt has malformed row" % prefix)
        member_name = _dataset_path(prefix, fields[1])
        if member_name in checked or member_name not in bodies:
            raise ArchiveError("%s checksums.txt has duplicate/missing member" % prefix)
        if _member_sha256(member_name, bodies, digests) != fields[0]:
            raise ArchiveError("%s checksummed member digest differs" % member_name)
        checked.add(member_name)
    actual = {
        name for name in bodies
        if name.startswith(prefix + "/")
        and name not in (manifest_name, checksums_name)
    }
    if checked != actual:
        raise ArchiveError("%s checksums do not cover exact dataset closure" % prefix)
    _validate_bound_panel_receipt(
        prefix, manifest_panel, binding, bodies, digests)
    capture_name = _dataset_path(prefix, identity.get("capture_manifest"))
    runtime_name = _dataset_path(prefix, identity.get("runtime_manifest"))
    manifest_capture = manifest.get("capture")
    manifest_runtime = manifest.get("runtime")
    if (not isinstance(manifest_capture, dict)
            or not isinstance(manifest_runtime, dict)
            or manifest_capture.get("manifest_file")
            != identity.get("capture_manifest")
            or manifest_capture.get("manifest_file_sha256")
            != identity.get("capture_manifest_sha256")
            or manifest_capture.get("capture_content_digest")
            != identity.get("capture_content_digest")
            or manifest_runtime.get("file")
            != identity.get("runtime_manifest")
            or manifest_runtime.get("file_sha256")
            != identity.get("runtime_manifest_sha256")
            or manifest_runtime.get("stack_fingerprint_sha256")
            != identity.get("stack_fingerprint_sha256")
            or manifest_runtime.get("lane_identity_sha256")
            != identity.get("lane_identity_sha256")):
        raise ArchiveError("%s dataset manifest differs from qualification"
                           % prefix)
    capture_body = bodies.get(capture_name)
    runtime_body = bodies.get(runtime_name)
    if not isinstance(capture_body, bytes) or not isinstance(runtime_body, bytes):
        raise ArchiveError("%s qualification names missing manifests" % prefix)
    capture = _parse_json_member(capture_name, capture_body)
    runtime = _parse_json_member(runtime_name, runtime_body)
    if (identity.get("capture_manifest_sha256")
            != _member_sha256(capture_name, bodies, digests)
            or identity.get("runtime_manifest_sha256")
            != _member_sha256(runtime_name, bodies, digests)
            or not _valid_hex(identity.get("capture_content_digest"), 64)
            or capture.get("capture_content_digest")
            != identity.get("capture_content_digest")
            or runtime.get("stack_fingerprint_sha256")
            != identity.get("stack_fingerprint_sha256")
            or runtime.get("lane_identity_sha256")
            != identity.get("lane_identity_sha256")):
        raise ArchiveError("%s manifests differ from qualification identities"
                           % prefix)
    runtime_weights = runtime.get("weights")
    runtime_environment = runtime.get("runtime_environment")
    capture_tool = runtime.get("capture_tool")
    stack_fingerprint = runtime.get("stack_fingerprint")
    if (capture.get("run_name") != expected_process
            or capture.get("form") != contract.get("form")
            or capture.get("dtype") != expected_dtype
            or not isinstance(runtime_weights, dict)
            or runtime_weights.get("repository") != target.get("repo_id")
            or runtime_weights.get("revision") != target.get("revision")
            or runtime_weights.get("model_revision") != target.get("revision")
            or runtime.get("lane") != contract.get("lane")
            or runtime.get("container") != identity.get("runtime_container")
            or not isinstance(runtime_environment, dict)
            or runtime_environment.get("cold_run") != expected_process
            or not isinstance(stack_fingerprint, dict)
            or stack_fingerprint.get("device") != contract.get("device")
            or stack_fingerprint.get("engine") != "transformers-eager"
            or not isinstance(capture_tool, dict)
            or capture_tool.get("file") != "engines/tools/hf_capture.py"
            or capture_tool.get("schedule") != contract.get("schedule")
            or capture_tool.get("resolved_panel_binding")
                != expected_binding_evidence
            or capture_tool.get("unexpected_tensor_allowlist")
                != observed_allowlist):
        raise ArchiveError(
            "%s capture/runtime inputs differ from the exact root job" % prefix)


def _positive_json_zero(value):
    return (isinstance(value, float) and value == 0.0
            and math.copysign(1.0, value) == 1.0)


def _validate_root_qualification_semantics(qualification):
    required = {
        "schema", "receipt_sha256", "qualified_at", "canonical_job_sha256",
        "job_file_sha256", "dataset_repository", "destination_repository",
        "job_contract", "captures", "comparison", "comparator",
        "verification", "reproduction_confirmation",
    }
    if (not isinstance(qualification, dict)
            or set(qualification) != required
            or qualification.get("schema") != ROOT_QUALIFICATION_SCHEMA
            or not _valid_hex(qualification.get("receipt_sha256"), 64)
            or not _valid_hex(qualification.get("canonical_job_sha256"), 64)
            or not _valid_hex(qualification.get("job_file_sha256"), 64)):
        raise ArchiveError("root qualification closed schema differs")
    captures = qualification.get("captures")
    if not isinstance(captures, dict) or set(captures) != {
            "canonical", "repeat"}:
        raise ArchiveError(
            "root qualification must bind canonical and repeat captures")
    canonical = captures["canonical"]
    repeat = captures["repeat"]
    if (not isinstance(canonical, dict) or not isinstance(repeat, dict)
            or canonical.get("process_label") != "root-cold-1"
            or repeat.get("process_label") != "root-cold-2"
            or canonical.get("dataset_id") != repeat.get("dataset_id")
            or canonical.get("dataset_repository")
                != qualification.get("dataset_repository")
            or repeat.get("dataset_repository")
                != qualification.get("dataset_repository")
            or canonical.get("capture_content_digest")
                != repeat.get("capture_content_digest")
            or canonical.get("stack_fingerprint_sha256")
                != repeat.get("stack_fingerprint_sha256")
            or canonical.get("lane_identity_sha256")
                != repeat.get("lane_identity_sha256")):
        raise ArchiveError(
            "root qualification fresh-process capture identities differ")
    comparison = qualification.get("comparison")
    if (not isinstance(comparison, dict)
            or set(comparison) != {
                "path", "file_sha256", "receipt_sha256", "comparison_kind",
                "mean_kld", "max_kld", "top1_agreement"}
            or comparison.get("comparison_kind")
                != "reproduction_confirmation"
            or not _valid_hex(comparison.get("file_sha256"), 64)
            or not _valid_hex(comparison.get("receipt_sha256"), 64)
            or not _positive_json_zero(comparison.get("mean_kld"))
            or not _positive_json_zero(comparison.get("max_kld"))
            or comparison.get("top1_agreement") != 1.0):
        raise ArchiveError(
            "root qualification exact-zero comparison semantics differ")
    comparator = qualification.get("comparator")
    if (not isinstance(comparator, dict)
            or set(comparator) != {
                "requested_replay_device", "requested_replay_dtype",
                "requested_vocab_chunk", "device", "replay_backend",
                "estimator_backend", "accumulation_dtype", "vocab_chunk",
                "force_compute_agreed"}
            or comparator.get("requested_replay_device") != "numpy"
            or comparator.get("requested_replay_dtype") != "float32"
            or comparator.get("requested_vocab_chunk") != 8192
            or comparator.get("device") != "cpu"
            or comparator.get("replay_backend") != "numpy:cpu:float32"
            or not isinstance(comparator.get("estimator_backend"), str)
            or not comparator["estimator_backend"]
            or comparator.get("accumulation_dtype") != "float64"
            or comparator.get("vocab_chunk") != 8192
            or comparator.get("force_compute_agreed") is not True):
        raise ArchiveError("root qualification comparator contract differs")
    verification = qualification.get("verification")
    if (not isinstance(verification, dict)
            or set(verification) != {"canonical", "repeat"}):
        raise ArchiveError("root qualification verification binding differs")
    for label in ("canonical", "repeat"):
        row = verification.get(label)
        if (not isinstance(row, dict)
                or set(row) != {"receipt_sha256", "file_sha256"}
                or not _valid_hex(row.get("receipt_sha256"), 64)
                or not _valid_hex(row.get("file_sha256"), 64)):
            raise ArchiveError(
                "root qualification %s verification receipt differs" % label)
    confirmation = qualification.get("reproduction_confirmation")
    expected_confirmation = {
        "two_fresh_processes": True,
        "distinct_dataset_roots": True,
        "both_independently_verified": True,
        "exact_zero_comparison": True,
        "canonical_dataset_only": True,
    }
    if confirmation != expected_confirmation:
        raise ArchiveError(
            "root qualification reproduction semantics differ")


def _validate_qualified_dataset_bodies(
        qualification, bodies, digests=None, *, job=None):
    _validate_root_qualification_semantics(qualification)
    if not isinstance(job, dict):
        raise ArchiveError(
            "qualified dataset validation requires the exact root job")
    captures = qualification.get("captures")
    if not isinstance(captures, dict) or set(captures) != {
            "canonical", "repeat"}:
        raise ArchiveError(
            "root qualification must bind canonical and repeat captures")
    contract = qualification.get("job_contract")
    _validate_dataset_tree(
        "dataset", captures["canonical"], bodies, digests, contract=contract)
    _validate_dataset_tree(
        "dataset-repeat", captures["repeat"], bodies, digests,
        contract=contract)




def _validate_publication_receipt(doc):
    revision = doc.get("revision")
    dataset_sha = doc.get("dataset_sha256")
    published_dataset_sha = doc.get("published_dataset_sha256")
    qualification_sha = doc.get("qualification_file_sha256")
    published_qualification_sha = doc.get(
        "published_qualification_file_sha256")
    result_archive_sha = doc.get("result_archive_sha256")
    result_archive_bytes = doc.get("result_archive_bytes")
    if (not _valid_hex(revision, 40)
            or doc.get("revision_immutable") is not True
            or doc.get("private") is not False
            or doc.get("verified_anonymously") is not True
            or doc.get("verified_after_publish") is not True
            or doc.get("verified_revision") != revision


            or not _valid_hex(dataset_sha, 64)
            or published_dataset_sha != dataset_sha
            or not _valid_hex(qualification_sha, 64)
            or published_qualification_sha != qualification_sha
            or not _valid_hex(result_archive_sha, 64)
            or isinstance(result_archive_bytes, bool)
            or not isinstance(result_archive_bytes, int)
            or result_archive_bytes <= 0):
        raise ArchiveError(
            "published root receipt lacks immutable revision, dataset refetch, "
            "or qualification refetch evidence")
def _validate_runpod_attestation(job, attestation):
    if not isinstance(attestation, dict):
        raise ArchiveError("RunPod live attestation must be an object")
    seal_body = dict(attestation)
    claimed_seal = seal_body.pop("attestation_sha256", None)
    computed_seal = hashlib.sha256(json.dumps(
        seal_body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    if (set(attestation) != RUNPOD_ATTESTATION_KEYS
            or attestation.get("schema") != RUNPOD_ATTESTATION_SCHEMA
            or not _valid_hex(claimed_seal, 64)
            or claimed_seal != computed_seal
            or attestation.get("provider") != "runpod"
            or not isinstance(attestation.get("provider_id"), str)
            or not attestation["provider_id"]
            or not isinstance(attestation.get("observed_at_utc"), str)
            or not attestation["observed_at_utc"]
            or attestation.get("transport_error") is not None
            or attestation.get("failures") != []
            or attestation.get("ok") is not True):
        raise ArchiveError("RunPod live attestation schema/seal/outcome is invalid")
    checks = attestation.get("checks")
    if (not isinstance(checks, dict) or not checks
            or any(value is not True for value in checks.values())):
        raise ArchiveError("RunPod live attestation checks are not all true")
    expected = attestation.get("expected")
    observed = attestation.get("observed")
    if not isinstance(observed, dict):
        raise ArchiveError("RunPod live attestation observation is invalid")
    clock = attestation.get("clock")
    if not isinstance(clock, dict) or set(clock) != RUNPOD_CLOCK_KEYS:
        raise ArchiveError("RunPod live attestation clock keys differ")
    numeric_clock_fields = (
        "controller_send_epoch", "controller_receive_epoch",
        "round_trip_seconds", "clock_skew_seconds",
        "allowed_skew_seconds")
    if any(isinstance(clock.get(field), bool)
           or not isinstance(clock.get(field), (int, float))
           or not math.isfinite(float(clock[field]))
           for field in numeric_clock_fields):
        raise ArchiveError("RunPod live attestation clock values are invalid")
    remote_epoch = clock.get("remote_time_epoch")
    if isinstance(remote_epoch, bool) or not isinstance(remote_epoch, int):
        raise ArchiveError("RunPod live attestation remote epoch is invalid")
    try:
        parsed_times = {
            field: calendar.timegm(time.strptime(
                clock[field], "%Y-%m-%dT%H:%M:%SZ"))
            for field in (
                "controller_send_utc", "controller_receive_utc",
                "remote_time_utc")
        }
    except (KeyError, TypeError, ValueError):
        raise ArchiveError("RunPod live attestation UTC values are invalid")
    if any(time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(parsed_times[field]))
           != clock[field] for field in parsed_times):
        raise ArchiveError("RunPod live attestation UTC values are not exact")
    sent = float(clock["controller_send_epoch"])
    received = float(clock["controller_receive_epoch"])
    round_trip = float(clock["round_trip_seconds"])
    midpoint = sent + round_trip / 2.0
    expected_skew = abs(remote_epoch - midpoint)
    if (received < sent
            or not math.isclose(
                round_trip, received - sent, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                float(clock["allowed_skew_seconds"]), 30.0 + round_trip,
                rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(
                float(clock["clock_skew_seconds"]), expected_skew,
                rel_tol=0.0, abs_tol=1e-6)
            or parsed_times["controller_send_utc"] != int(sent)
            or parsed_times["controller_receive_utc"] != int(received)
            or parsed_times["remote_time_utc"] != remote_epoch
            or attestation["observed_at_utc"]
            != clock["controller_receive_utc"]
            or clock.get("within_bound") is not True
            or expected_skew > float(clock["allowed_skew_seconds"])
            or checks.get("remote_clock") is not True
            or observed.get("remote_time_epoch") != remote_epoch
            or observed.get("remote_time_utc") != clock["remote_time_utc"]):
        raise ArchiveError("RunPod live attestation clock proof is inconsistent")
    requirements = job.get("resource_requirements")
    environment = job.get("environment")
    if (not isinstance(expected, dict) or set(expected) != RUNPOD_EXPECTED_KEYS
            or not isinstance(observed, dict)
            or not isinstance(requirements, dict)
            or set(requirements) != RESOURCE_REQUIREMENT_KEYS
            or not isinstance(environment, dict)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in requirements.values())):
        raise ArchiveError("RunPod live attestation resource blocks are invalid")
    if (expected.get("expected_vram_bytes")
            != requirements["expected_vram_bytes"]
            or expected.get("min_vcpu")
            != requirements["min_vcpu_count"]
            or expected.get("min_ram_gb")
            != requirements["min_memory_gb"]
            or expected.get("workspace_available_bytes_minimum")
            != requirements["workspace_available_bytes_minimum"]
            or expected.get("container_available_bytes_minimum")
            != requirements["container_available_bytes_minimum"]
            or expected.get("gpu_model") != environment.get("gpu")
            or isinstance(expected.get("volume_gb"), bool)
            or not isinstance(expected.get("volume_gb"), int)
            or expected["volume_gb"] <= 0
            or isinstance(expected.get("container_disk_gb"), bool)
            or not isinstance(expected.get("container_disk_gb"), int)
            or expected["volume_gb"] * 1024 ** 3
            < requirements["workspace_available_bytes_minimum"]
            or expected["container_disk_gb"] * 1024 ** 3
            < requirements["container_available_bytes_minimum"]):
        raise ArchiveError("RunPod live attestation differs from job resources")
    filesystems = observed.get("filesystems")
    workspace = ((filesystems.get("workspace") or {}).get("available_bytes")
                 if isinstance(filesystems, dict) else None)
    container = ((filesystems.get("container") or {}).get("available_bytes")
                 if isinstance(filesystems, dict) else None)
    if (isinstance(workspace, bool) or not isinstance(workspace, int)
            or workspace
            < requirements["workspace_available_bytes_minimum"]
            or isinstance(container, bool) or not isinstance(container, int)
            or container
            < requirements["container_available_bytes_minimum"]):
        raise ArchiveError("RunPod available storage is below job requirements")


def _validate_runpod_host_key_proof(job, proof, attestation):
    required = (
        (job.get("execution_attempt") or {}).get("kind") == "runpod-ssh")
    if not required:
        return
    if (not isinstance(proof, dict)
            or set(proof) != RUNPOD_HOST_KEY_PROOF_KEYS
            or proof.get("schema") != RUNPOD_HOST_KEY_PROOF_SCHEMA
            or not verify_seal(proof, field="proof_sha256")
            or proof.get("provider") != "runpod"
            or proof.get("provider_id") != attestation.get("provider_id")
            or proof.get("verification_source")
                != "operator-authenticated-runpod-web-terminal"
            or proof.get("algorithm") != "ssh-ed25519"
            or re.fullmatch(
                r"SHA256:[A-Za-z0-9+/]{43}",
                str(proof.get("fingerprint") or "")) is None
            or not isinstance(proof.get("host"), str)
            or not proof["host"]
            or isinstance(proof.get("port"), bool)
            or not isinstance(proof.get("port"), int)
            or not 1 <= proof["port"] <= 65535
            or not _valid_hex(proof.get("known_hosts_sha256"), 64)
            or not _valid_hex(proof.get("proof_sha256"), 64)):
        raise ArchiveError(
            "RunPod SSH host-key proof is absent, unsealed, or unauthenticated")
    try:
        parsed = time.strptime(proof.get("verified_at_utc"), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        raise ArchiveError("RunPod SSH host-key proof time is invalid")
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != proof["verified_at_utc"]:
        raise ArchiveError("RunPod SSH host-key proof time is noncanonical")


def _check_local_runpod_attestation(fs_root, job):
    path = Path(fs_root) / RUNPOD_ATTESTATION_PATH
    required = isinstance(job.get("resource_requirements"), dict)
    if not path.is_file() or path.is_symlink():
        if required:
            raise ArchiveError("RunPod result requires %s" % RUNPOD_ATTESTATION_PATH)
        return
    attestation = _strict_json_loads(
        path.read_text(encoding="utf-8"), RUNPOD_ATTESTATION_PATH)
    if not isinstance(attestation, dict):
        raise ArchiveError("RunPod live attestation must be an object")
    _validate_runpod_attestation(job, attestation)
    if ((job.get("execution_attempt") or {}).get("kind") == "runpod-ssh"):
        proof_path = Path(fs_root) / RUNPOD_HOST_KEY_PROOF_PATH
        if not proof_path.is_file() or proof_path.is_symlink():
            raise ArchiveError(
                "RunPod result requires %s" % RUNPOD_HOST_KEY_PROOF_PATH)
        proof = _strict_json_loads(
            proof_path.read_text(encoding="utf-8"),
            RUNPOD_HOST_KEY_PROOF_PATH)
        _validate_runpod_host_key_proof(job, proof, attestation)


def _check_archive_runpod_attestation(job, bodies):
    body = bodies.get(RUNPOD_ATTESTATION_PATH)
    required = isinstance(job.get("resource_requirements"), dict)
    if body is None:
        if required:
            raise ArchiveError("RunPod archive requires %s"
                               % RUNPOD_ATTESTATION_PATH)
        return
    attestation = _parse_json_member(RUNPOD_ATTESTATION_PATH, body)
    _validate_runpod_attestation(job, attestation)
    if ((job.get("execution_attempt") or {}).get("kind") == "runpod-ssh"):
        proof_body = bodies.get(RUNPOD_HOST_KEY_PROOF_PATH)
        if proof_body is None:
            raise ArchiveError(
                "RunPod archive requires %s" % RUNPOD_HOST_KEY_PROOF_PATH)
        proof = _parse_json_member(
            RUNPOD_HOST_KEY_PROOF_PATH, proof_body)
        _validate_runpod_host_key_proof(job, proof, attestation)


def _required_string(value, label):
    if not isinstance(value, str) or not value:
        raise ArchiveError("%s is missing or unrepresentable" % label)
    return value
def _validate_target_census(job, census, job_file_sha256):
    if set(census) != TARGET_CENSUS_KEYS:
        raise ArchiveError("fetch-target census keys differ from v1 contract")
    if census.get("schema") != TARGET_CENSUS_SCHEMA or not verify_seal(census):
        raise ArchiveError("fetch-target census schema or self-seal is invalid")
    _required_string(census.get("verified_at"), "fetch-target verified_at")
    target = job.get("target")
    if not isinstance(target, dict):
        raise ArchiveError("job target contract is missing")
    for field in ("config_sha256", "index_sha256", "shard_manifest_sha256"):
        if (not _valid_hex(target.get(field), 64)
                or census.get(field) != target.get(field)):
            raise ArchiveError("fetch-target census %s differs from job" % field)
    if (census.get("job_id_full") != job.get("job_id_full")
            or not _valid_hex(job_file_sha256, 64)
            or census.get("job_file_sha256") != job_file_sha256
            or census.get("repository") != target.get("repo_id")
            or census.get("revision") != target.get("revision")):
        raise ArchiveError("fetch-target census is not bound to exact job/target")
    model_bytes = target.get("model_bytes")
    shards = target.get("shards")
    if (isinstance(model_bytes, bool) or not isinstance(model_bytes, int)
            or model_bytes <= 0 or census.get("model_bytes") != model_bytes
            or not isinstance(shards, list) or not shards):
        raise ArchiveError("fetch-target census size/shards differ from job")
    canonical = []
    for row in shards:
        if (not isinstance(row, dict) or set(row) != {"path", "bytes"}
                or not isinstance(row.get("path"), str) or not row["path"]
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int)
                or row["bytes"] < 0):
            raise ArchiveError("job target shard contract is invalid")
        canonical.append({"path": row["path"], "bytes": row["bytes"]})
    canonical.sort(key=lambda row: row["path"])
    paths = [row["path"] for row in canonical]
    if (shards != canonical or census.get("shards") != canonical
            or census.get("index_shards") != sorted(set(paths))
            or len(paths) != len(set(paths))
            or sum(row["bytes"] for row in canonical) != model_bytes):
        raise ArchiveError("fetch-target census shard map differs from job/index")


def _target_census_required(fs_root, summary, complete):
    marker = Path(fs_root) / "receipts" / "done" / "fetch_target.done"
    stages = list(summary.get("stages") or [])
    return (marker.is_file()
            or (complete and summary.get("verb") in ("measure", "capture"))
            or (summary.get("status") == "qualified-unpublished"
                and summary.get("verb") == "capture")
            or (complete and summary.get("verb") == "stage"
                and stages == ["fetch_target"]))

def _check_local_target_census(fs_root, summary, job, complete):
    path = Path(fs_root) / TARGET_CENSUS_PATH
    required = _target_census_required(fs_root, summary, complete)
    if not path.is_file():
        if required:
            raise ArchiveError("result requires %s" % TARGET_CENSUS_PATH)
        return
    census = _require_sealed_receipt(
        fs_root, TARGET_CENSUS_PATH, "post-fetch result")
    _validate_target_census(
        job, census, sha256_file(str(Path(fs_root) / "job.json")))
def _check_archive_target_census(job, bodies, verb, complete, state):
    stages = state.get("stages")
    required = (
        "receipts/done/fetch_target.done" in bodies
        or (complete and verb in ("measure", "capture"))
        or (state.get("status") == "qualified-unpublished"
            and verb == "capture")
        or (complete and verb == "stage" and stages == ["fetch_target"]))
    body = bodies.get(TARGET_CENSUS_PATH)
    if body is None:
        if required:
            raise ArchiveError("archive requires %s" % TARGET_CENSUS_PATH)
        return
    census = _parse_json_member(TARGET_CENSUS_PATH, body)
    _validate_target_census(
        job, census, hashlib.sha256(bodies["job.json"]).hexdigest())






def _validate_quant_evidence(job, receipt):
    receipt_keys = set(receipt)
    if receipt_keys not in (
            QUANT_RECEIPT_KEYS, QUANT_RECEIPT_KEYS | {"comparability"}):
        raise ArchiveError("measurement receipt keys differ from exact contract")
    if (not isinstance(receipt.get("measured_at"), str)
            or not receipt["measured_at"]
            or not isinstance(receipt.get("auxiliary_metrics"), dict)
            or not isinstance(receipt.get("environment"), dict)
            or not isinstance(receipt.get("cost"), dict)
            or not isinstance(receipt.get("evidence"), list)
            or not isinstance(receipt.get("disclosures"), list)
            or not receipt["disclosures"]):
        raise ArchiveError("measurement receipt block types are invalid")
    target = job.get("target")
    panel = job.get("panel")
    reference = job.get("reference")
    artifact = receipt.get("artifact")
    receipt_panel = receipt.get("panel")
    receipt_reference = receipt.get("reference")
    if not all(isinstance(value, dict) for value in (
            target, panel, reference, artifact, receipt_panel,
            receipt_reference)):
        raise ArchiveError("quant job/receipt identity blocks must be objects")

    repository = _required_string(target.get("repo_id"), "job target repository")
    revision = _required_string(target.get("revision"), "job target revision")
    _required_string(job.get("lane"), "job lane")
    if (artifact.get("repository") != repository
            or artifact.get("revision") != revision
            or "path" not in artifact
            or artifact.get("path") != target.get("path")):
        raise ArchiveError("measurement artifact does not match job target")
    model_bytes = target.get("model_bytes")
    if (isinstance(model_bytes, bool) or not isinstance(model_bytes, int)
            or model_bytes <= 0 or target.get("size_bytes") != model_bytes
            or artifact.get("size_bytes") != model_bytes):
        raise ArchiveError("measurement artifact size differs from job model bytes")
    precision = _required_string(
        target.get("precision_label"), "job target precision_label")
    container = _required_string(
        target.get("container"), "job target container")
    if (artifact.get("precision_label") != precision
            or artifact.get("container") != container
            or artifact.get("url")
            != "https://huggingface.co/%s" % repository):
        raise ArchiveError(
            "measurement artifact container/precision/url differs from job")

    codec = artifact.get("codec")
    if not isinstance(codec, dict):
        raise ArchiveError("measurement artifact codec is missing")
    codec_family = _required_string(target.get("codec"), "job target codec")
    bits = target.get("bits")
    if (isinstance(bits, bool) or not isinstance(bits, (int, float))
            or codec.get("family") != codec_family
            or codec.get("bits_per_weight_nominal") != bits):
        raise ArchiveError("measurement codec does not match job target")
    quantizer_tool = target.get("quantizer_tool")
    quantizer_version = (
        target.get("exllamav3_pin") or target.get("quantizer_version"))
    if (codec.get("quantizer_tool") != quantizer_tool
            or codec.get("quantizer_version") != quantizer_version
            or codec.get("bits_per_weight_effective")
            != target.get("bits_per_weight_effective")
            or codec.get("group_size") != target.get("group_size")):
        raise ArchiveError(
            "measurement quantizer provenance does not match job target")
    if (artifact.get("shard_hash_verification")
            != target.get("shard_hash_verification")):
        raise ArchiveError(
            "measurement shard verification differs from job target")
    for field in ("config_sha256", "index_sha256"):
        if (not _valid_hex(target.get(field), 64)
                or artifact.get(field) != target.get(field)):
            raise ArchiveError(
                "measurement artifact %s does not match job target" % field)
    if not isinstance(job.get("scope"), dict) or artifact.get("scope") != job["scope"]:
        raise ArchiveError("measurement artifact scope does not match job scope")

    for field in ("panel_token_sha256", "panel_receipt_sha256"):
        if (not _valid_hex(panel.get(field), 64)
                or receipt_panel.get(field) != panel.get(field)):
            raise ArchiveError("measurement panel %s does not match job" % field)
    if (_required_string(panel.get("panel_ref"), "job panel_ref")
            != receipt_panel.get("panel_ref")
            or receipt_panel.get("contexts") != panel.get("contexts")
            or receipt_panel.get("scored_positions_total")
            != panel.get("scored_positions")):
        raise ArchiveError("measurement panel does not match job panel contract")

    for field in ("teacher_receipt_sha256",
                  "teacher_backend_identity_sha256"):
        if (not _valid_hex(reference.get(field), 64)
                or receipt_reference.get(field) != reference.get(field)):
            raise ArchiveError(
                "measurement reference %s does not match job" % field)
    if (_required_string(reference.get("reference_ref"), "job reference_ref")
            != receipt_reference.get("reference_ref")):
        raise ArchiveError("measurement reference does not match job")
    if receipt.get("lane") != job.get("lane"):
        raise ArchiveError("measurement lane does not match job")
    measurer = job.get("measurer")
    if not isinstance(measurer, dict):
        raise ArchiveError("job measurer is missing")
    _required_string(measurer.get("name"), "job measurer.name")
    _required_string(measurer.get("handle"), "job measurer.handle")
    if receipt.get("measurer") != measurer:
        raise ArchiveError("measurement measurer does not match job")
    profile = job.get("profile")
    profile_id = (profile.get("profile_id")
                  if isinstance(profile, dict) else None)
    produced_by = job.get("produced_by")
    dependencies = (produced_by.get("dependencies")
                    if isinstance(produced_by, dict) else None)
    if (not isinstance(profile_id, str) or not profile_id
            or not isinstance(dependencies, dict)
            or dependencies.get("profile") != profile_id
            or receipt.get("produced_by") != produced_by):
        raise ArchiveError(
            "measurement pipeline/profile provenance does not match job")
    scoring = job.get("scoring")
    metric = receipt.get("metric")
    estimator = receipt.get("estimator")
    measurement_scope = receipt.get("measurement_scope")
    determinism = receipt.get("determinism")
    cold_runs = job.get("cold_runs")
    if (not isinstance(scoring, dict)
            or scoring.get("direction") != "reference_to_candidate"
            or scoring.get("vocabulary") != "full"
            or scoring.get("compute_dtype") != "float64"
            or scoring.get("reduction")
            != "mean_of_run_means_tokenwise_kld"
            or not isinstance(metric, dict)
            or metric.get("name") != scoring["reduction"]
            or metric.get("direction") != scoring["direction"]
            or metric.get("units") != "nats"
            or isinstance(metric.get("value"), bool)
            or not isinstance(metric.get("value"), (int, float))
            or not math.isfinite(metric["value"]) or metric["value"] < 0
            or not isinstance(estimator, dict)
            or estimator.get("accumulation_dtype")
            != scoring["compute_dtype"]
            or not isinstance(measurement_scope, dict)
            or measurement_scope.get("contexts") != 25
            or measurement_scope.get("scored_positions") != 51175
            or measurement_scope.get("covers_full_panel") is not True
            or measurement_scope.get("position_filter") != "all"
            or measurement_scope.get("subset_detail") is not None
            or panel.get("contexts") != 25
            or panel.get("scored_positions") != 51175):
        raise ArchiveError(
            "measurement scoring is not exact full-vocabulary fp64 KLD")
    run_means = determinism.get("run_means") if isinstance(determinism, dict) else None
    evidence_hashes = (
        determinism.get("evidence_hashes")
        if isinstance(determinism, dict) else None)
    report_hashes = (
        determinism.get("per_run_report_sha256")
        if isinstance(determinism, dict) else None)
    if (not isinstance(determinism, dict)
            or cold_runs != 2
            or determinism.get("run_count") != 2
            or determinism.get("cold_start_per_run") is not True
            or determinism.get("identical_across_runs") is not True
            or determinism.get("evidence_kind") != "tokenwise_kld_sha256"
            or determinism.get("distinct_evidence_hash_count") != 1
            or not isinstance(run_means, list) or len(run_means) != 2
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0
                   for value in run_means)
            or run_means[0] != run_means[1]
            or not isinstance(evidence_hashes, list)
            or len(evidence_hashes) != 1
            or not _valid_hex(evidence_hashes[0], 64)
            or not isinstance(report_hashes, list) or len(report_hashes) != 2
            or len(set(report_hashes)) != 2
            or any(not _valid_hex(value, 64) for value in report_hashes)):
        raise ArchiveError(
            "measurement determinism lacks two distinct cold-run proofs")


def _quant_report_hashes(receipt):
    determinism = receipt.get("determinism")
    hashes = (determinism.get("per_run_report_sha256")
              if isinstance(determinism, dict) else None)
    if (not isinstance(hashes, list) or len(hashes) != 2
            or len(set(hashes)) != 2
            or any(not _valid_hex(value, 64) for value in hashes)):
        raise ArchiveError(
            "measurement receipt lacks two distinct per-run report identities")
    return set(hashes)


def _validate_local_quant_reports(source_paths, receipt):
    expected = _quant_report_hashes(receipt)
    matched = []
    for path in source_paths:
        relative = path.as_posix()
        if (not relative.endswith(".json")
                or relative.endswith("receipts/measurement-receipt.json")
                or ("/receipts/" not in relative
                    and "/reports/" not in relative)):
            continue
        digest = sha256_file(str(path))
        if digest in expected:
            matched.append((digest, path))
    if len(matched) != 2 or {item[0] for item in matched} != expected:
        raise ArchiveError(
            "delivered evidence lacks the two exact per-run reports")


def _validate_archive_quant_reports(receipt, digests):
    expected = _quant_report_hashes(receipt)
    matched = [
        (digest, name) for name, digest in digests.items()
        if digest in expected
        and name.endswith(".json")
        and name != "receipts/measurement-receipt.json"
        and (name.startswith("receipts/") or name.startswith("reports/"))
    ]
    if len(matched) != 2 or {item[0] for item in matched} != expected:
        raise ArchiveError(
            "archive lacks the two exact per-run reports")

def _root_archive_caps(job):
    target = job.get("target")
    storage = (target.get("root_capture_storage")
               if isinstance(target, dict) else None)
    if not isinstance(storage, dict) or storage.get("form") != "hidden":
        raise ArchiveError("root job lacks hidden-capture archive storage contract")
    positions = storage.get("selected_prediction_positions")
    duplicate = storage.get(
        "capture_archive_duplicate_upper_bound_bytes")
    members = storage.get("result_archive_max_members")
    uncompressed = storage.get("result_archive_max_uncompressed_bytes")
    transfer = storage.get("result_archive_max_transfer_bytes")
    values = (positions, duplicate, members, uncompressed, transfer)
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value <= 0 for value in values):
        raise ArchiveError("root archive storage bounds must be positive integers")
    if (storage.get("required_dataset_trees") != 2
            or storage.get("fresh_processes") != 2
            or storage.get("capture_bytes_total") != duplicate
            or members != 2 * positions + 128
            or uncompressed != duplicate + ARCHIVE_MARGIN_BYTES
            or transfer != (uncompressed
                            + ((uncompressed + 16382) // 16383) * 5 + 64)):
        raise ArchiveError("root archive storage bounds violate exact formula")
    return {
        "members": members,
        "uncompressed": uncompressed,
        "transfer": transfer,
    }


def _enforce_root_archive_caps(job, member_count, uncompressed, transfer=None):
    caps = _root_archive_caps(job)

    if member_count > caps["members"]:
        raise ArchiveError("root result exceeds job archive member bound")
    if uncompressed > caps["uncompressed"]:
        raise ArchiveError("root result exceeds job archive byte bound")
    if transfer is not None and transfer > caps["transfer"]:
        raise ArchiveError("root result exceeds job archive transfer bound")
def _quant_archive_caps(job):
    target = job.get("target")
    contract = (target.get("result_archive_contract")
                if isinstance(target, dict) else None)
    required_keys = {
        "retained_content", "result_archive_max_members",
        "result_archive_max_uncompressed_bytes",
        "result_archive_max_transfer_bytes",
    }
    retained = ["receipts", "reports", "bounded-log-tails", "control"]
    uncompressed = 2 * 1024 ** 3
    transfer = uncompressed + ((uncompressed + 16382) // 16383) * 5 + 64
    if (not isinstance(contract, dict) or set(contract) != required_keys
            or contract.get("retained_content") != retained
            or contract.get("result_archive_max_members") != 2048
            or contract.get("result_archive_max_uncompressed_bytes")
            != uncompressed
            or contract.get("result_archive_max_transfer_bytes") != transfer):
        raise ArchiveError("quant result archive contract is not the exact bound")
    return {
        "members": 2048,
        "uncompressed": uncompressed,
        "transfer": transfer,
    }


def _enforce_quant_archive_caps(job, member_count, uncompressed, transfer=None):
    caps = _quant_archive_caps(job)
    if member_count > caps["members"]:
        raise ArchiveError("quant result exceeds job archive member bound")
    if uncompressed > caps["uncompressed"]:
        raise ArchiveError("quant result exceeds job archive byte bound")
    if transfer is not None and transfer > caps["transfer"]:
        raise ArchiveError("quant result exceeds job archive transfer bound")


def _validate_root_evidence(job, qualification, publication=None,
                            qualification_file_sha256=None,
                            job_file_sha256=None):
    job_sha = job.get("job_id_full")
    capture = job.get("capture")
    if not isinstance(capture, dict):
        raise ArchiveError("root job capture contract is missing")
    dataset_repository = _required_string(
        capture.get("dataset_repository"), "root dataset_repository")
    publish_destination = capture.get("publish_root_to")
    canonical_capture = ((qualification.get("captures") or {}).get("canonical")
                         if isinstance(qualification.get("captures"), dict)
                         else None)
    canonical_dataset_sha = (
        canonical_capture.get("dataset_sha256")
        if isinstance(canonical_capture, dict) else None)
    try:
        expected_contract = jobcontract.root_qualification_contract(job)
    except jobcontract.JobContractError as exc:
        raise ArchiveError(
            "root job cannot produce a public qualification contract: %s"
            % exc) from exc
    if qualification.get("job_contract") != expected_contract:
        raise ArchiveError(
            "root qualification job_contract differs from exact job.json")
    if (not _valid_hex(job_sha, 64)
            or qualification.get("canonical_job_sha256") != job_sha
            or qualification.get("dataset_repository")
            != dataset_repository
            or qualification.get("destination_repository")
            != publish_destination
            or not _valid_hex(job_file_sha256, 64)
            or qualification.get("job_file_sha256") != job_file_sha256
            or not _valid_hex(canonical_dataset_sha, 64)):
        raise ArchiveError(
            "root qualification is not bound to job.json, dataset repository, "
            "and canonical dataset")
    execution_kind = (job.get("execution_attempt") or {}).get("kind")
    if execution_kind == "runpod-ssh":
        image_reference = ((job.get("environment") or {}).get("image")
                           if isinstance(job.get("environment"), dict) else None)
        image_parts = (image_reference.rsplit("@", 1)
                       if isinstance(image_reference, str) else [])
        if (len(image_parts) != 2
                or not image_parts[0]
                or not image_parts[1].startswith("sha256:")
                or not _valid_hex(image_parts[1][len("sha256:"):], 64)):
            raise ArchiveError("root job image is not an immutable reference")
        image_digest = image_parts[1]
        qualification_contract = qualification.get("job_contract")
        qualification_captures = qualification.get("captures")
        if (not isinstance(qualification_contract, dict)
                or qualification_contract.get("execution_kind")
                != execution_kind
                or qualification_contract.get("container_image_reference")
                != image_reference
                or qualification_contract.get("container_image_digest")
                != image_digest
                or not isinstance(qualification_captures, dict)):
            raise ArchiveError(
                "root qualification container contract differs from job image")
        for label in ("canonical", "repeat"):
            identity = qualification_captures.get(label)
            runtime_container = (
                identity.get("runtime_container")
                if isinstance(identity, dict) else None)
            if (not isinstance(runtime_container, dict)
                    or runtime_container.get("image_reference")
                    != image_reference
                    or runtime_container.get("image_digest") != image_digest):
                raise ArchiveError(
                    "%s root capture container differs from job image" % label)
    if publication is None:
        return
    if publish_destination != dataset_repository:
        raise ArchiveError(
            "published root destination differs from dataset_repository")
    if publication.get("repository") != dataset_repository:
        raise ArchiveError(
            "root publication repository differs from dataset_repository")
    if (publication.get("qualification_receipt_sha256")
            != qualification.get("receipt_sha256")
            or not _valid_hex(qualification_file_sha256, 64)
            or publication.get("qualification_file_sha256")
            != qualification_file_sha256
            or publication.get("dataset_sha256") != canonical_dataset_sha
            or publication.get("published_dataset_sha256")
            != canonical_dataset_sha):
        raise ArchiveError(
            "root publication is not bound to the qualification and canonical "
            "dataset")
    _validate_publication_receipt(publication)

def _verify_role_members(manifest, bodies, digests=None):
    verb = str(manifest.get("verb") or "")
    status, complete, failed = _status_kind(manifest)
    qualified_unpublished = status == "qualified-unpublished"
    state = _parse_json_member(RUN_STATE_NAME, bodies.get(RUN_STATE_NAME, b""))
    if (state.get("verb") != verb or state.get("status") != status
            or state.get("role") != manifest.get("role")
            or state.get("job_id_full") != manifest.get("job_id_full")
            or state.get("measurement_receipt_sha256")
            != manifest.get("measurement_receipt_sha256")
            or bool(state.get("publication_requested"))
            != bool(manifest.get("publication_requested"))):
        raise ArchiveError("run-state.json disagrees with the manifest")

    if verb == "doctor":
        if status not in ("ok", "failed"):
            raise ArchiveError("doctor status must be ok or failed")
        if (manifest.get("role") != "doctor"
                or manifest.get("job_id_full") is not None
                or manifest.get("measurement_receipt_sha256") is not None
                or bool(manifest.get("publication_requested"))):
            raise ArchiveError(
                "doctor manifest has an invalid role or identity")
        doctor = _parse_json_member(
            "receipts/doctor.json", bodies.get("receipts/doctor.json", b""))
        expected = "ok" if status == "ok" else "failed"
        if (doctor.get("schema") != "malaiwah.fidelity-doctor.v1"
                or doctor.get("status") != expected):
            raise ArchiveError("receipts/doctor.json disagrees with doctor status")
        return

    job = _parse_json_member("job.json", bodies.get("job.json", b""))
    verified_job_id = _verify_job_contract(job)
    if (manifest.get("job_id_full") != verified_job_id
            or state.get("job_id_full") != verified_job_id):
        raise ArchiveError("manifest job identity does not match job.json")
    role = job.get("role")
    if role not in ("quant", "root") or role != manifest.get("role"):
        raise ArchiveError("manifest role does not match job.json role")
    publication_requested = (
        isinstance(job.get("capture"), dict)
        and job["capture"].get("publish_root_to") is not None)
    if publication_requested != bool(manifest.get("publication_requested")):
        raise ArchiveError("manifest publication request disagrees with job.json")
    _check_archive_runpod_attestation(job, bodies)
    _check_archive_target_census(job, bodies, verb, complete, state)

    if verb == "stage":
        stages = state.get("stages")
        if (not isinstance(stages, list) or len(stages) != 1
                or not isinstance(stages[0], str) or not stages[0]
                or "/" in stages[0] or "\\" in stages[0]):
            raise ArchiveError("stage archive must name exactly one safe stage")
        stage = stages[0]
        if "logs/%s.log" % stage not in bodies:
            raise ArchiveError("stage archive lacks logs/%s.log" % stage)
        if complete and "receipts/done/%s.done" % stage not in bodies:
            raise ArchiveError(
                "completed stage archive lacks receipts/done/%s.done" % stage)
        if failed and state.get("failed_stage") != stage:
            raise ArchiveError("failed stage archive does not name its failed_stage")
        if status == "abandoned":
            abandoned = _parse_json_member(
                "ABANDONED.json", bodies.get("ABANDONED.json", b""))
            if abandoned.get("schema") != "fidelity-suite/abandoned.v2":
                raise ArchiveError("ABANDONED.json has the wrong schema")
        return
    if verb not in ("measure", "capture"):
        raise ArchiveError("archive has unsupported result verb %r" % verb)
    if verb == "measure" and role != "quant":
        raise ArchiveError("measure archive requires quant role")
    if verb == "capture" and role != "root":
        raise ArchiveError("capture archive requires root role")

    def sealed(name):
        doc = _parse_json_member(name, bodies.get(name, b""))
        if not verify_seal(doc):
            raise ArchiveError("%s has an invalid receipt_sha256 self-seal" % name)
        return doc

    if complete and verb == "measure":
        measurement = sealed("receipts/measurement-receipt.json")
        if measurement.get("submission_schema") != QUANT_RECEIPT_SCHEMA:
            raise ArchiveError("measurement receipt has the wrong submission_schema")
        if (manifest.get("measurement_receipt_sha256")
                != measurement.get("receipt_sha256")):
            raise ArchiveError(
                "manifest measurement identity does not match receipt")
        _validate_quant_evidence(job, measurement)
        _validate_archive_quant_reports(measurement, digests)
    elif manifest.get("measurement_receipt_sha256") is not None:
        raise ArchiveError("non-measurement archive binds a measurement identity")

    if (complete or qualified_unpublished) and verb == "capture":
        qualification = sealed("receipts/root-qualification.json")
        if qualification.get("schema") != ROOT_QUALIFICATION_SCHEMA:
            raise ArchiveError("root qualification receipt has the wrong schema")
        publication = None
        if (qualified_unpublished
                and "receipts/publish-root.json" in bodies):
            raise ArchiveError(
                "qualified-unpublished archive carries publication receipt")
        if complete and publication_requested:
            publication = sealed("receipts/publish-root.json")
            if publication.get("schema") != ROOT_PUBLICATION_SCHEMA:
                raise ArchiveError("root publication receipt has the wrong schema")
        _validate_root_evidence(
            job, qualification, publication,
            hashlib.sha256(
                bodies["receipts/root-qualification.json"]).hexdigest(),
            hashlib.sha256(bodies["job.json"]).hexdigest())
        _validate_qualified_dataset_bodies(
            qualification, bodies, digests, job=job)
    if failed:
        if not any(name.startswith("logs/") for name in bodies):
            raise ArchiveError("failed/abandoned archive has no stage log")
        if status == "abandoned":
            abandoned = _parse_json_member(
                "ABANDONED.json", bodies.get("ABANDONED.json", b""))
            if abandoned.get("schema") != "fidelity-suite/abandoned.v2":
                raise ArchiveError("ABANDONED.json has the wrong schema")


def _verified_archive(source, expected_sha256=None, expected_bytes=None):
    archive_source, archive_bytes, archive_sha256 = _archive_source(source)
    if expected_bytes is not None and archive_bytes != expected_bytes:
        raise ArchiveError("transferred archive byte count mismatch: expected "
                           "%d, got %d" % (expected_bytes, archive_bytes))
    if expected_sha256 is not None:
        if not _valid_hex(expected_sha256, 64):
            raise ArchiveError("expected archive SHA-256 is malformed")
        if archive_sha256 != expected_sha256:
            raise ArchiveError("transferred archive SHA-256 mismatch")

    stream = (io.BytesIO(archive_source)
              if isinstance(archive_source, bytes) else None)
    try:
        archive = tarfile.open(
            name=None if stream is not None else str(archive_source),
            fileobj=stream, mode="r:gz")
        with archive:
            seen = set()
            by_name = {}
            uncompressed_bytes = 0
            while True:
                member = archive.next()
                if member is None:
                    break
                if len(by_name) >= MAX_ARCHIVE_MEMBERS + 3:
                    raise ArchiveError("archive exceeds member safety limit")
                name = _safe_member_name(member.name)
                if name in seen:
                    raise ArchiveError("duplicate archive member %s" % name)
                seen.add(name)
                if not member.isfile():
                    raise ArchiveError(
                        "archive member %s is not a regular file" % name)
                if member.size < 0:
                    raise ArchiveError(
                        "archive member %s has negative size" % name)
                uncompressed_bytes += member.size
                if uncompressed_bytes > MAX_ARCHIVE_BYTES:
                    raise ArchiveError(
                        "archive exceeds uncompressed-byte safety limit")
                by_name[name] = member
            retained_declared_bytes = 0
            for name, member in by_name.items():
                if (name == RESULT_MANIFEST_NAME
                        or name.endswith(".json")
                        or name.endswith("/checksums.txt")):
                    if member.size > MAX_RETAINED_MEMBER_BYTES:
                        raise ArchiveError(
                            "retained archive member exceeds memory safety cap: "
                            + name)
                    retained_declared_bytes += member.size
                    if retained_declared_bytes > MAX_RETAINED_METADATA_BYTES:
                        raise ArchiveError(
                            "retained archive metadata exceeds memory safety cap")
            if RESULT_MANIFEST_NAME not in by_name:
                raise ArchiveError("archive lacks %s" % RESULT_MANIFEST_NAME)
            manifest_file = archive.extractfile(by_name[RESULT_MANIFEST_NAME])
            if manifest_file is None:
                raise ArchiveError("cannot read %s" % RESULT_MANIFEST_NAME)
            manifest_body = manifest_file.read()
            manifest = _parse_json_member(RESULT_MANIFEST_NAME, manifest_body)
            if (manifest.get("schema") != RESULT_MANIFEST_SCHEMA
                    or not verify_seal(manifest, field="manifest_sha256")):
                raise ArchiveError("result manifest schema or self-seal is invalid")
            records = manifest.get("files")
            if not isinstance(records, list):
                raise ArchiveError("result manifest files must be an array")
            expected_names = set()
            record_by_name = {}
            for record in records:
                if not isinstance(record, dict):
                    raise ArchiveError("result manifest file entry is not an object")
                name = _safe_member_name(record.get("path"))
                if name == RESULT_MANIFEST_NAME or name in expected_names:
                    raise ArchiveError("duplicate/reserved manifest path %s" % name)
                if (not isinstance(record.get("bytes"), int)
                        or record["bytes"] < 0
                        or not _valid_hex(record.get("sha256"), 64)):
                    raise ArchiveError("invalid size or SHA-256 for %s" % name)
                expected_names.add(name)
                record_by_name[name] = record
            actual_payload = set(by_name) - {RESULT_MANIFEST_NAME}
            if actual_payload != expected_names:
                missing = sorted(expected_names - actual_payload)
                extra = sorted(actual_payload - expected_names)
                raise ArchiveError("archive member set mismatch (missing=%r extra=%r)"
                                   % (missing, extra))
            science_status = str(manifest.get("status")).lower()
            early_root_caps = (
                manifest.get("role") == "root"
                and manifest.get("verb") == "capture"
                and science_status in (
                    "ok", "complete", "completed", "success",
                    "qualified-unpublished",
                    "completed-operational-failure"))
            early_quant_caps = (
                manifest.get("role") == "quant"
                and manifest.get("verb") == "measure"
                and science_status in (
                    "ok", "complete", "completed", "success",
                    "completed-operational-failure"))
            if early_root_caps or early_quant_caps:
                job_member = by_name.get("job.json")
                job_record = record_by_name.get("job.json")
                if (job_member is None or job_record is None
                        or job_member.size != job_record["bytes"]):
                    raise ArchiveError(
                        "completed science archive lacks exact job.json")
                job_stream = archive.extractfile(job_member)
                if job_stream is None:
                    raise ArchiveError("cannot read job.json for archive caps")
                job_body_for_caps = job_stream.read()
                if (len(job_body_for_caps) != job_record["bytes"]
                        or hashlib.sha256(job_body_for_caps).hexdigest()
                        != job_record["sha256"]):
                    raise ArchiveError(
                        "job.json differs before archive-cap enforcement")
                job_for_caps = _parse_json_member(
                    "job.json", job_body_for_caps)
                _verify_job_contract(job_for_caps)
                if early_root_caps:
                    _enforce_root_archive_caps(
                        job_for_caps, len(by_name), uncompressed_bytes,
                        archive_bytes)
                else:
                    _enforce_quant_archive_caps(
                        job_for_caps, len(by_name), uncompressed_bytes,
                        archive_bytes)
            bodies = {}
            digests = {
                name: record["sha256"]
                for name, record in record_by_name.items()
            }
            for name in sorted(expected_names):
                member = by_name[name]
                record = record_by_name[name]
                if member.size != record["bytes"]:
                    raise ArchiveError("archive member size mismatch for %s" % name)
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise ArchiveError("cannot read archive member %s" % name)
                retain = (name.endswith(".json")
                          or name.endswith("/checksums.txt"))
                if retain:
                    body = source_file.read()
                    actual_bytes = len(body)
                    actual_sha = hashlib.sha256(body).hexdigest()
                    bodies[name] = body
                else:
                    digest = hashlib.sha256()
                    actual_bytes = 0
                    for chunk in iter(lambda: source_file.read(1 << 20), b""):
                        actual_bytes += len(chunk)
                        digest.update(chunk)
                    actual_sha = digest.hexdigest()
                    bodies[name] = None
                if (actual_bytes != record["bytes"]
                        or actual_sha != record["sha256"]):
                    raise ArchiveError("archive member digest mismatch for %s" % name)
    except ArchiveError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ArchiveError("result archive is truncated or unreadable: %s"
                           % exc.__class__.__name__)
    _strict_json_loads(manifest_body, RESULT_MANIFEST_NAME)
    for name, body in bodies.items():
        if name.endswith(".json") and isinstance(body, bytes):
            _strict_json_loads(body, name)
    if not isinstance(archive_source, bytes):
        _same_source, after_bytes, after_sha = _archive_source(source)
        if after_bytes != archive_bytes or after_sha != archive_sha256:
            raise ArchiveError("result archive changed during verification")
    _verify_role_members(manifest, bodies, digests)
    # Job-specific count/size/transfer caps were enforced from headers and the
    # exact job bytes before streaming potentially large payload members.
    bodies[RESULT_MANIFEST_NAME] = manifest_body
    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "manifest": manifest,
    }, bodies


def verify_archive(source, expected_sha256=None, expected_bytes=None):
    """Verify transfer identity, every member, self-seal, and role contract.

    `expected_sha256` and `expected_bytes` are the on-pod values reported before
    SSH transfer. Any mismatch raises ArchiveError; there is no partial result.
    """
    result, _bodies = _verified_archive(
        source, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    return result

def extract_verified_archive(source, destination, expected_sha256=None,
                             expected_bytes=None):
    """Verify completely, then atomically publish a link-free extraction."""
    result, _retained_json = _verified_archive(
        source, expected_sha256=expected_sha256, expected_bytes=expected_bytes)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("extraction destination already exists: %s"
                           % destination)
    _ensure_durable_directory(destination.parent)
    staging = Path(tempfile.mkdtemp(
        dir=str(destination.parent), prefix=".result-extract-"))
    _fsync_directory(staging)
    _fsync_directory(destination.parent)
    try:
        archive_source, current_bytes, current_sha = _archive_source(source)
        if (current_bytes != result["archive_bytes"]
                or current_sha != result["archive_sha256"]):
            raise ArchiveError("result archive changed before extraction")
        stream = (io.BytesIO(archive_source)
                  if isinstance(archive_source, bytes) else None)
        with tarfile.open(
                name=None if stream is not None else str(archive_source),
                fileobj=stream, mode="r:gz") as archive:
            for member in sorted(archive.getmembers(),
                                 key=lambda item: item.name):
                name = _safe_member_name(member.name)
                if not member.isfile():
                    raise ArchiveError(
                        "archive member %s changed type before extraction" % name)
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise ArchiveError("cannot extract archive member %s" % name)
                output = staging.joinpath(*PurePosixPath(name).parts)
                _ensure_durable_directory(output.parent)
                _stream_exclusive_durable(output, source_file)
        _after_source, after_bytes, after_sha = _archive_source(source)
        if (after_bytes != result["archive_bytes"]
                or after_sha != result["archive_sha256"]):
            raise ArchiveError("result archive changed during extraction")
        _fsync_directory(staging)
        os.replace(str(staging), str(destination))
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    result = dict(result)
    result["extracted_to"] = str(destination)
    return result


def _delivery_identity(archive):
    if isinstance(archive, (bytes, bytearray)):
        body = bytes(archive)
        return len(body), hashlib.sha256(body).hexdigest()
    path = Path(archive)
    return path.stat().st_size, sha256_file(str(path))


class _FileChunks:
    def __init__(self, path):
        self.path = Path(path)

    def __iter__(self):
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                yield chunk


def _deliver_stdout(fs_root, summary, con, archive_body=None):
    """Frame the answer and the transferable archive identity into stdout."""
    if archive_body is None:
        archive_body = build_archive(fs_root, summary)
    archive_bytes, archive_sha256 = _delivery_identity(archive_body)
    transfer = {
        "name": ARCHIVE_NAME,
        "bytes": archive_bytes,
        "sha256": archive_sha256,
    }
    framed = dict(summary)
    framed["archive"] = transfer
    print(BEGIN, flush=True)
    print(json.dumps(framed, indent=2, sort_keys=True), flush=True)
    failed = summary.get("failed_stage")
    if failed:
        log = Path(fs_root) / "logs" / ("%s.log" % failed)
        if log.is_file():
            body = _payload(log, Path(fs_root))
            print("----- logs/%s.log (the stage that failed) -----" % failed,
                  flush=True)
            print(body.decode("utf-8", "replace"), flush=True)
    receipt = Path(fs_root) / "receipts" / "measurement-receipt.json"
    if receipt.is_file():
        size = receipt.stat().st_size
        if size <= STDOUT_CAP_BYTES:
            print("----- measurement-receipt.json -----", flush=True)
            print(receipt.read_text(encoding="utf-8"), flush=True)
        else:
            print("----- measurement-receipt.json WITHHELD: %d bytes > %d cap; "
                  "sha256 is in the summary above; use file: or https: -----"
                  % (size, STDOUT_CAP_BYTES), flush=True)
    print(END, flush=True)
    return {
        "scheme": "stdout", "ok": True, "files": len(summary["files"]),
        "bytes": transfer["bytes"], "sha256": transfer["sha256"],
    }


def _deliver_file(fs_root, summary, target, con, archive_body=None):
    if archive_body is None:
        archive_body = build_archive(fs_root, summary)
    archive_bytes, digest = _delivery_identity(archive_body)
    dest = Path(target)
    _ensure_durable_directory(dest)
    output = dest / ARCHIVE_NAME
    handle, temporary = tempfile.mkstemp(
        dir=str(dest), prefix=".result-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            if isinstance(archive_body, (bytes, bytearray)):
                stream.write(archive_body)
            else:
                with Path(archive_body).open("rb") as source:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if (os.stat(temporary).st_size != archive_bytes
                or sha256_file(temporary) != digest):
            raise ArchiveError("result archive changed during file delivery")
        os.replace(temporary, str(output))
        _fsync_directory(dest)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    con("result sink file: %d bytes -> %s" % (archive_bytes, output))
    return {
        "scheme": "file", "ok": True, "bytes": archive_bytes,
        "sha256": digest, "target": str(output),
    }


def _deliver_http(fs_root, summary, url, con, *, method=None, timeout=120.0,
                  archive_body=None):
    """PUT the canonical archive. A presigned upload URL is normally a PUT."""
    body = archive_body if archive_body is not None else build_archive(
        fs_root, summary)
    archive_bytes, digest = _delivery_identity(body)
    request_data = (bytes(body)
                    if isinstance(body, (bytes, bytearray))
                    else _FileChunks(body))
    method = (method or os.environ.get("FIDELITY_RESULT_SINK_METHOD") or "PUT").upper()
    req = urllib.request.Request(url, data=request_data, method=method)
    req.add_header("Content-Type", "application/gzip")
    req.add_header("Content-Length", str(archive_bytes))
    req.add_header("X-Fidelity-Result-SHA256", digest)
    req.add_header("X-Fidelity-Status", str(summary.get("status")))
    req.add_header("X-Fidelity-Verb", str(summary.get("verb")))
    auth = os.environ.get("FIDELITY_RESULT_SINK_AUTH")
    if auth:
        register_secret(auth)
        req.add_header("Authorization", auth)
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
    except urllib.error.HTTPError as exc:
        raise SinkError("result sink https: %s returned HTTP %s"
                        % (_host(url), exc.code))
    except Exception as exc:
        raise SinkError("result sink https: %s failed: %s"
                        % (_host(url), exc.__class__.__name__))
    con("result sink https: %d bytes -> %s (HTTP %s)"
        % (archive_bytes, _host(url), code))
    return {
        "scheme": "http", "ok": True, "bytes": archive_bytes,
        "sha256": digest, "code": code,
    }


def _host(url):
    """Never echo a presigned URL: the query string is the credential."""
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        return "%s://%s%s" % (parts.scheme, parts.netloc, parts.path)
    except Exception:
        return "(url)"


def deliver(fs_root, sinks, summary, con):
    """Build once and deliver byte-identical archives to every binary sink.

    Archive construction and role validation happen before any sink. Invalid
    scientific evidence is a hard ArchiveError, not a successful partial
    delivery. Transport failures remain per-sink so one endpoint cannot hide
    another endpoint's outcome.
    """
    payload_bytes = sum(
        int(record.get("bytes", 0)) for record in (summary.get("files") or []))
    spool = None
    if payload_bytes > MAX_IN_MEMORY_ARCHIVE_BYTES:
        spool = Path(tempfile.mkdtemp(
            dir=str(Path(fs_root)), prefix=".result-delivery-"))
        archive_body = spool / ARCHIVE_NAME
        write_archive(fs_root, summary, archive_body)
    else:
        archive_body = build_archive(fs_root, summary)
    results = []
    try:
        for sink in sinks:
            try:
                if sink.scheme == "stdout":
                    results.append(_deliver_stdout(
                        fs_root, summary, con, archive_body=archive_body))
                elif sink.scheme == "file":
                    results.append(_deliver_file(
                        fs_root, summary, sink.target, con,
                        archive_body=archive_body))
                elif sink.scheme == "http":
                    results.append(_deliver_http(
                        fs_root, summary, sink.target, con,
                        archive_body=archive_body))
            except SinkError as exc:
                con("RESULT SINK FAILED (%s): %s" % (sink.scheme, exc))
                results.append(
                    {"scheme": sink.scheme, "ok": False, "error": str(exc)})
            except Exception as exc:
                con("RESULT SINK FAILED (%s): %s"
                    % (sink.scheme, exc.__class__.__name__))
                results.append({"scheme": sink.scheme, "ok": False,
                                "error": exc.__class__.__name__})
    finally:
        if spool is not None:
            shutil.rmtree(str(spool), ignore_errors=True)
    return results
