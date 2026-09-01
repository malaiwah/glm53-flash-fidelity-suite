#!/usr/bin/env python3
"""RunPod backend, duck-typed to `fidelity.jlapi.JL`.

Why duck-typed rather than a refactor
-------------------------------------
`measure_cloud.py` touches a provider through exactly eighteen methods, and
everything else it does -- the fit check, the cost band, the lease, all four
teardown layers, every stage in `stage_measure.sh` -- is written against that
surface rather than against JarvisLabs. So a second provider is a second class
with the same eighteen methods, not a rewrite. This file is that class.

What is genuinely different from JarvisLabs, and matters
--------------------------------------------------------
* **There is no CLI.** JarvisLabs is driven through `jl`; RunPod is a GraphQL
  endpoint plus SSH. The API key therefore never reaches argv here -- it is
  read from a 0600 file into a request header, in-process.
* **There is no managed-run concept.** `jl run` starts a tracked background
  job; RunPod has nothing like it, so `run_job` is `nohup` plus three files
  (pid, exit code, log) and `run_status` reads them. That is the same contract
  the stage runner already expects, and it is honest about the one thing that
  matters: a stage that died is distinguishable from a stage still running.
* **Storage is not separable AS THIS BACKEND DRIVES IT.** JarvisLabs
  filesystems outlive their instance; the POD volume this backend creates is
  made and destroyed with the pod. `fs_create` therefore records the request
  and returns the pod's own volume, and `fs_delete` is a no-op that says so
  rather than pretending to have deleted something.

  RunPod offers separately billed network volumes and native container command
  launch.  The first supported paid profile here deliberately uses neither:
  it is one SSH-driven controller process on one pod-scoped volume.  Passing a
  network-volume/mount option or a native docker command is therefore refused
  before any provider call rather than silently changing teardown or recovery
  semantics.

Everything the controller does with the returned objects -- `Instance`,
`GpuOffer` -- uses the dataclasses from `jlapi`, imported rather than re-typed,
so a field added there cannot silently diverge here.
"""
from __future__ import annotations

import base64
import binascii
import calendar
import email.utils
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import ipaddress
import json
import math
import re
import secrets
import os
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .common import register_secret, safe_urlopen
from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

GQL = "https://api.runpod.io/graphql"
V2 = "https://api.runpod.io/v2"
REST_V1 = "https://rest.runpod.io/v1"
MAX_JSON_RESPONSE_BYTES = 8 * 1024 * 1024
# Sole stable default, expanded once to an absolute path; key bytes stay in-file.
DEFAULT_KEY_FILE = os.path.abspath(
    os.path.expanduser("~/.config/runpod/api_key"))
# CUDA 13.0 / Ubuntu 24.04 supplies the admitted driver and Python 3.12 base.
# Bootstrap replaces Python packages from the exact URL+SHA256 lock.
DEFAULT_IMAGE = (
    "runpod/pytorch@sha256:"
    "ab2addc2916ffc72989288bd5048933c69ba6531f1d679c25afbd9eadc5a5fd5")
MIN_CREATE_SETUP_SECONDS = 300
_BILLING_ROUNDING_TOLERANCE_USD = Decimal("0.000000000000000001")


def _billing_total_matches_record_sum(
        total: Decimal, record_sum: Decimal) -> bool:
    """Accept only sub-attodollar provider aggregation roundoff."""
    return abs(total - record_sum) <= _BILLING_ROUNDING_TOLERANCE_USD




class RunPodError(JLError):
    """Same exception family as the JarvisLabs backend, so callers need no branch."""

class RunPodCreateResponseError(RunPodError):
    """Create committed with an exact id but returned unqualified metadata."""

    def __init__(self, message: str, provider_id: str,
                 response: Dict[str, Any]) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.response = dict(response)


class _NoMutationRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

@dataclass(frozen=True)
class PreparedRunPodCreate:
    http_request: Any
    http_opener: Any
    graphql_body: bytes
    request_identity_json: bytes
    name: str
    terminate_after: str
    storage_gb: int
    container_disk_gb: int
    image_name: str
    dry_run: bool

    def to_dict(self) -> Dict[str, Any]:
        identity = _strict_json_loads(
            self.request_identity_json.decode("utf-8"))
        return {
            "schema": "fidelity-suite/runpod-prepared-create.v1",
            "request_identity": identity,
            "graphql_body_sha256":
                hashlib.sha256(self.graphql_body).hexdigest(),
            "graphql_body_bytes": len(self.graphql_body),
            "graphql_body_base64":
                base64.b64encode(self.graphql_body).decode("ascii"),
        }


def _load_key(path: Optional[str] = None) -> str:
    selected = path or os.environ.get("RUNPOD_KEY_FILE") or DEFAULT_KEY_FILE
    selected = os.path.expanduser(str(selected))
    if not os.path.isabs(selected):
        raise RunPodError("RunPod key file path must be absolute")
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(selected, flags)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RunPodError("RunPod key file must be a regular file, not a symlink")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise RunPodError("RunPod key file must have mode 0600: %s" % selected)
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise RunPodError("RunPod key file must be owned by the current user")
        with os.fdopen(fd, encoding="utf-8") as fh:
            fd = None
            key = fh.read().strip()
    except RunPodError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RunPodError("RunPod key file is unavailable or invalid at %s: %s"
                          % (selected, exc))
    finally:
        if fd is not None:
            os.close(fd)
    if not key:
        raise RunPodError("RunPod key file is empty: %s" % selected)
    register_secret(key)
    return key

def _exact_utc(value: str, field: str) -> str:
    text = str(value)
    try:
        parsed = time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise RunPodError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != text:
        raise RunPodError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    return text


def _terminate_after(kw: Dict[str, Any]) -> Optional[str]:
    text = kw.get("terminate_after")
    epoch = kw.get("terminate_after_epoch")
    if text is not None and epoch is not None:
        raise RunPodError("pass only one of terminate_after and terminate_after_epoch")
    if epoch is not None:
        try:
            value = float(epoch)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("non-positive or non-finite")
            text = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
        except (TypeError, ValueError, OverflowError, OSError):
            raise RunPodError("terminate_after_epoch must be a positive finite epoch")
    return _exact_utc(text, "terminate_after") if text is not None else None

def _canonical_public_key(value: str) -> str:
    lines = str(value).splitlines()
    if len(lines) != 1:
        raise RunPodError("SSH public key must be exactly one line")
    fields = lines[0].split()
    accepted = (
        "ssh-ed25519", "ssh-rsa",
        "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521", "sk-ssh-ed25519@openssh.com",
        "sk-ecdsa-sha2-nistp256@openssh.com",
    )
    if len(fields) < 2 or fields[0] not in accepted:
        raise RunPodError("SSH public key has an unsupported key type")
    try:
        base64.b64decode(fields[1].encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeError):
        raise RunPodError("SSH public key payload is not valid base64")
    return "%s %s" % (fields[0], fields[1])

def _strict_json_loads(raw: str) -> Any:
    def _pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise RunPodError("RunPod JSON contains duplicate key %r" % key)
            out[key] = value
        return out

    def _constant(value):
        raise RunPodError("RunPod JSON contains non-finite number %s" % value)

    try:
        return json.loads(
            raw, object_pairs_hook=_pairs, parse_float=str,
            parse_int=int, parse_constant=_constant)
    except RunPodError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunPodError("RunPod returned invalid JSON: %s" % exc)


def _finite_decimal(value: Any, field: str, *, positive: bool = False,
                    nonnegative: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RunPodError("%s is not an exact decimal" % field)
    if (not parsed.is_finite()
            or (positive and parsed <= 0)
            or (nonnegative and parsed < 0)):
        raise RunPodError("%s is not a valid finite amount" % field)
    return parsed

def _decimal_float(value: Decimal, field: str) -> float:
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        raise RunPodError("%s cannot be represented as a finite float" % field)
    if not math.isfinite(parsed):
        raise RunPodError("%s cannot be represented as a finite float" % field)
    return parsed

_PROVIDER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_GPU_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:+()/+-]{0,255}\Z")


def _provider_id(value: Any, field: str = "RunPod pod id") -> str:
    if not isinstance(value, str) or _PROVIDER_ID_RE.fullmatch(value) is None:
        raise RunPodError("%s has invalid characters or length" % field)
    return value


def _gpu_id(value: Any, field: str = "RunPod GPU id") -> str:
    if not isinstance(value, str) or _GPU_ID_RE.fullmatch(value) is None:
        raise RunPodError("%s has invalid characters or length" % field)
    return value


def _gql_string(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise RunPodError("%s must be a string" % field)
    return json.dumps(value, ensure_ascii=True, allow_nan=False)

def _response_json(response: Any, expected_url: str, label: str) -> Any:
    expected = urllib.parse.urlsplit(expected_url)
    final_url = response.geturl()
    final = urllib.parse.urlsplit(final_url)
    expected_port = expected.port or 443
    final_port = final.port or (443 if final.scheme == "https" else None)
    if (expected.scheme != "https" or final.scheme != "https"
            or final.hostname != expected.hostname
            or final_port != expected_port):
        raise RunPodError("RunPod %s response crossed its HTTPS origin" % label)
    status = getattr(response, "status", None) or response.getcode()
    if status != 200:
        raise RunPodError("RunPod %s returned unexpected HTTP status %s"
                          % (label, status))
    content_type = response.headers.get_content_type()
    if content_type != "application/json":
        raise RunPodError("RunPod %s returned non-JSON content type" % label)
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            parsed_length = int(length)
        except ValueError:
            raise RunPodError("RunPod %s returned invalid Content-Length" % label)
        if parsed_length < 0 or parsed_length > MAX_JSON_RESPONSE_BYTES:
            raise RunPodError("RunPod %s response is too large" % label)
    raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
    if len(raw) > MAX_JSON_RESPONSE_BYTES:
        raise RunPodError("RunPod %s response is too large" % label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise RunPodError("RunPod %s response is not UTF-8" % label)
    return _strict_json_loads(text)

_LIVE_ATTEST_SCRIPT = r'''
import json
import os
import subprocess
import time

def mount(path):
    resolved = os.path.realpath(path)
    stats = os.statvfs(resolved)
    best = None
    with open("/proc/self/mountinfo", "r", encoding="utf-8") as stream:
        for line in stream:
            left, right = line.rstrip("\n").split(" - ", 1)
            fields = left.split()
            point = fields[4].replace("\\040", " ")
            if resolved == point or resolved.startswith(point.rstrip("/") + "/"):
                if best is None or len(point) > len(best[0]):
                    tail = right.split()
                    best = (point, tail[0], tail[1])
    if best is None:
        raise RuntimeError("mountpoint not found for " + resolved)
    return {
        "path": resolved, "mount_point": best[0], "fs_type": best[1],
        "source": best[2], "device": int(os.stat(resolved).st_dev),
        "total_bytes": int(stats.f_blocks * stats.f_frsize),
        "available_bytes": int(stats.f_bavail * stats.f_frsize),
    }

mem_kib = None
with open("/proc/meminfo", "r", encoding="ascii") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
if mem_kib is None:
    raise RuntimeError("MemTotal missing")
limits = [mem_kib * 1024]
for path in ("/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        value = open(path, "r", encoding="ascii").read().strip()
        if value != "max":
            parsed = int(value)
            if 0 < parsed < (1 << 60):
                limits.append(parsed)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
smi = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,driver_version",
     "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
gpus = []
if smi.returncode == 0:
    for line in smi.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            gpus.append({
                "index": int(fields[0]), "name": fields[1],
                "vram_bytes": int(fields[2]) * 1024 * 1024,
                "driver_version": fields[3],
            })
cuda = {"usable": False, "count": 0, "name": None,
        "vram_bytes": None, "error": None}
try:
    import torch
    cuda["usable"] = bool(torch.cuda.is_available())
    cuda["count"] = int(torch.cuda.device_count())
    if cuda["usable"] and cuda["count"]:
        probe = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        del probe
        props = torch.cuda.get_device_properties(0)
        cuda["name"] = str(props.name)
        cuda["vram_bytes"] = int(props.total_memory)
except Exception as exc:
    cuda["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
remote_time_epoch = int(time.time())
remote_time_utc = time.strftime(
    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(remote_time_epoch))
print(json.dumps({
    "remote_time_epoch": remote_time_epoch,
    "remote_time_utc": remote_time_utc,
    "logical_cpus": len(os.sched_getaffinity(0)),
    "memtotal_bytes": mem_kib * 1024,
    "effective_memory_bytes": min(limits),
    "nvidia_smi_exit_code": smi.returncode,
    "nvidia_smi_error": smi.stderr[:300],
    "gpus": gpus, "cuda": cuda,
    "filesystems": {"container": mount("/"), "workspace": mount("/workspace")},
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


def _attestation_seal(document: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(document)
    raw = json.dumps(
        out, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    out["attestation_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


class RunPod(SSHTransport):
    """Thin, auditable wrapper. `dry` short-circuits every mutating call."""
    # False because THIS BACKEND creates a pod-scoped volume, which is made
    # and destroyed with the pod -- so the whole run must fit on the
    # instance's own disk. RunPod itself does offer network volumes that
    # outlive a pod (`networkVolumeId` on REST pod creation); this code does
    # not attach one. See the module docstring: the flag describes our
    # driving of the provider, not the provider.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False

    provider = "runpod"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self._server_time: Optional[Dict[str, Any]] = None
        self._ssh_cache: Dict[int, tuple] = {}

    # -- transport ---------------------------------------------------------
    def _capture_server_time(self, response: Any, endpoint: str) -> None:
        raw = response.headers.get("Date")
        if not isinstance(raw, str) or not raw:
            raise RunPodError("RunPod authenticated response lacks HTTP Date")
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            raise RunPodError("RunPod authenticated response Date is invalid")
        if (parsed is None or parsed.utcoffset() is None
                or parsed.utcoffset().total_seconds() != 0
                or email.utils.format_datetime(parsed, usegmt=True) != raw):
            raise RunPodError(
                "RunPod authenticated response Date is not strict GMT")
        received = time.time()
        server_epoch = parsed.timestamp()
        self._server_time = {
            "schema": "fidelity-suite/runpod-server-time.v1",
            "endpoint_origin": (
                urllib.parse.urlsplit(endpoint).scheme + "://"
                + str(urllib.parse.urlsplit(endpoint).hostname)),
            "date_header": raw,
            "server_epoch": server_epoch,
            "local_received_epoch": received,
            "local_minus_server_seconds": received - server_epoch,
        }

    def server_time_evidence(
            self, *, max_clock_delta_seconds: float = 30,
            max_evidence_age_seconds: float = 30) -> Dict[str, Any]:
        evidence = self._server_time
        if evidence is None:
            raise RunPodError(
                "RunPod server time is unavailable; authenticated status or "
                "inventory must succeed before create")
        now = time.time()
        age = now - evidence["local_received_epoch"]
        if not math.isfinite(age) or age < -1 or age > max_evidence_age_seconds:
            raise RunPodError("RunPod server-time evidence is stale")
        delta = evidence["local_minus_server_seconds"]
        if abs(delta) > max_clock_delta_seconds:
            raise RunPodError(
                "local UTC differs from RunPod server UTC by more than %.0fs"
                % max_clock_delta_seconds)
        out = dict(evidence)
        out.update({
            "checked_at_epoch": now,
            "evidence_age_seconds": age,
            "max_clock_delta_seconds": float(max_clock_delta_seconds),
            "max_evidence_age_seconds": float(max_evidence_age_seconds),
        })
        return out

    # -- transport ---------------------------------------------------------
    def _gql(self, query: str, *, timeout: float = 60) -> Dict[str, Any]:
        if self._key is None:
            self._key = _load_key(self._key_file)
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            GQL, data=body,
            headers={"Content-Type": "application/json",
                     # Cloudflare fronts api.runpod.io and answers urllib's
                     # default User-Agent with HTTP 403 "error code: 1010"
                     # (browser integrity check). curl works only because it
                     # sends one. Not optional.
                     "User-Agent": "quant-fidelity-suite/0.1",
                     # header, never argv: `ps` on a shared box would show it
                     "Authorization": "Bearer " + self._key})
        try:
            with safe_urlopen(req, timeout=timeout) as resp:
                doc = _response_json(resp, GQL, "GraphQL")
                if query.lstrip().startswith("query"):
                    self._capture_server_time(resp, GQL)
        except urllib.error.HTTPError as exc:
            raise RunPodError("RunPod HTTP %d: %s"
                              % (exc.code, redact(
                                  exc.read(300).decode("utf-8", "replace"))))
        except RunPodError:
            raise
        except Exception as exc:                          # noqa: BLE001
            raise RunPodError("RunPod request failed: %s" % redact(str(exc)))
        if not isinstance(doc, dict):
            raise RunPodError("RunPod GraphQL returned non-object JSON")
        if doc.get("errors"):
            raise RunPodError("RunPod GraphQL: %s"
                              % redact(json.dumps(doc["errors"])[:300]))
        data = doc.get("data")
        if not isinstance(data, dict):
            raise RunPodError("RunPod GraphQL response lacks object data")
        return data

    def _get_readonly(self, base: str, path: str, query: Dict[str, str], *,
                      label: str, timeout: float = 60) -> Any:
        """Read a REST endpoint without placing the key in URL or argv."""
        if self._key is None:
            self._key = _load_key(self._key_file)
        suffix = ("?" + urllib.parse.urlencode(query)) if query else ""
        url = base + path + suffix
        req = urllib.request.Request(
            url, method="GET",
            headers={"Accept": "application/json",
                     "User-Agent": "quant-fidelity-suite/0.1",
                     "Authorization": "Bearer " + self._key})
        try:
            with safe_urlopen(req, timeout=timeout) as resp:
                doc = _response_json(resp, url, label)
                self._capture_server_time(resp, url)
                return doc
        except urllib.error.HTTPError as exc:
            raise RunPodError(
                "RunPod %s GET %s -> HTTP %d: %s"
                % (label, path, exc.code,
                   redact(exc.read(300).decode("utf-8", "replace"))))
        except RunPodError:
            raise
        except Exception as exc:                          # noqa: BLE001
            raise RunPodError("RunPod %s request failed: %s"
                              % (label, redact(str(exc))))

    def _get_v2(self, path: str, query: Dict[str, str], *,
                timeout: float = 60) -> Dict[str, Any]:
        doc = self._get_readonly(V2, path, query, label="v2", timeout=timeout)
        if not isinstance(doc, dict):
            raise RunPodError("RunPod v2 GET %s returned non-object JSON" % path)
        return doc

    def _get_v1(self, path: str, query: Optional[Dict[str, str]] = None, *,
                timeout: float = 60) -> Any:
        return self._get_readonly(
            REST_V1, path, query or {}, label="REST v1", timeout=timeout)

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            _load_key(self._key_file)
            return True
        except RunPodError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise RunPodError("RunPod credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "runpod-graphql"

    def status(self) -> Dict[str, Any]:
        data = self._gql(
            "query { myself { id clientBalance currentSpendPerHr } }")
        myself = data.get("myself")
        if (not isinstance(myself, dict)
                or not isinstance(myself.get("id"), str)
                or not myself["id"]
                or myself["id"] != myself["id"].strip()):
            raise RunPodError("RunPod status lacks exact string myself.id")
        out = dict(myself)
        out["observed_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for field in ("clientBalance", "currentSpendPerHr"):
            if out.get(field) is not None:
                out[field] = format(
                    _finite_decimal(
                        out[field], "RunPod %s" % field, nonnegative=True),
                    "f")
        return out

    def balance(self) -> Decimal:
        value = self.status().get("clientBalance")
        if value is None:
            raise RunPodError("RunPod status lacks clientBalance")
        return _finite_decimal(
            value, "RunPod clientBalance", nonnegative=True)

    # -- catalogue ---------------------------------------------------------
    def gpus(self, *, gpu_type: Optional[str] = None,
             secure_only: bool = False) -> List[GpuOffer]:
        """Return validated offers, optionally querying one exact secure GPU."""
        requested = (
            None if gpu_type is None
            else _gpu_id(gpu_type, "requested RunPod GPU inventory id"))
        if not isinstance(secure_only, bool):
            raise RunPodError("secure_only must be a boolean")
        offers: List[GpuOffer] = []
        base = self._gql(
            "query { gpuTypes { id displayName memoryInGb communityCloud secureCloud } }")
        rows = base.get("gpuTypes")
        if not isinstance(rows, list):
            raise RunPodError("RunPod GPU inventory lacks gpuTypes list")
        seen = set()
        for gpu in rows:
            if not isinstance(gpu, dict):
                raise RunPodError("RunPod GPU inventory contains a non-object row")
            gpu_id = _gpu_id(gpu.get("id"), "RunPod GPU inventory id")
            if gpu_id in seen:
                raise RunPodError(
                    "RunPod GPU inventory contains duplicate id %s" % gpu_id)
            seen.add(gpu_id)
            if requested is not None and gpu_id != requested:
                continue
            for secure in ((True,) if secure_only else (True, False)):
                if secure and gpu.get("secureCloud") is not True:
                    continue
                if not secure and gpu.get("communityCloud") is not True:
                    continue
                query = (
                    'query { gpuTypes(input:{id:%s}) { lowestPrice'
                    '(input:{gpuCount:1,secureCloud:%s}) '
                    '{ minimumBidPrice uninterruptablePrice stockStatus } } }'
                    % (_gql_string(gpu_id, "RunPod GPU inventory id"),
                       "true" if secure else "false"))
                priced = self._gql(query).get("gpuTypes")
                if (not isinstance(priced, list) or len(priced) != 1
                        or not isinstance(priced[0], dict)):
                    raise RunPodError(
                        "RunPod exact GPU price query is structurally ambiguous")
                lowest = priced[0].get("lowestPrice")
                if lowest is None:
                    continue
                if not isinstance(lowest, dict):
                    raise RunPodError("RunPod lowestPrice is not an object")
                stock = lowest.get("stockStatus")
                if stock in (None, "", False):
                    continue
                if stock not in ("High", "Medium", "Low"):
                    raise RunPodError(
                        "RunPod stockStatus is not a recognized exact value")
                raw_price = lowest.get("uninterruptablePrice")
                if raw_price is None:
                    continue
                price_decimal = _finite_decimal(
                    raw_price, "RunPod uninterruptablePrice", positive=True)
                memory_decimal = _finite_decimal(
                    gpu.get("memoryInGb"), "RunPod memoryInGb", positive=True)
                bid_raw = lowest.get("minimumBidPrice")
                bid_decimal = (
                    None if bid_raw is None else _finite_decimal(
                        bid_raw, "RunPod minimumBidPrice", nonnegative=True))
                offers.append(GpuOffer(
                    gpu_type=gpu_id,
                    region="secure" if secure else "community",
                    vram_bytes=_decimal_float(
                        memory_decimal, "RunPod memoryInGb") * (1024 ** 3),
                    price=_decimal_float(
                        price_decimal, "RunPod uninterruptablePrice"),
                    spot=False,
                    free_devices={"High": 8, "Medium": 3, "Low": 1}[stock],
                    workload_type="container",
                    raw={
                        "displayName": gpu.get("displayName"),
                        "stockStatus": stock,
                        "secureCloud": secure,
                        "uninterruptablePriceDecimal":
                            format(price_decimal, "f"),
                        "bid": (None if bid_decimal is None
                                else format(bid_decimal, "f")),
                    }))
        return offers
    # -- instances ---------------------------------------------------------
    _POD_FIELDS = (
        "id name desiredStatus costPerHr imageName gpuCount "
        "containerDiskInGb volumeInGb networkVolumeId "
        "machine { id gpuTypeId gpuDisplayName secureCloud "
        "currentPricePerGpu podHostId } "
        "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } }"
    )

    def _pods(self) -> List[Dict[str, Any]]:
        data = self._gql(
            "query { myself { pods { %s } } }" % self._POD_FIELDS)
        myself = data.get("myself")
        if not isinstance(myself, dict) or not isinstance(myself.get("pods"), list):
            raise RunPodError(
                "RunPod pod listing lacks myself.pods list; inventory is unknown")
        pods = myself["pods"]
        seen = set()

        def nullable_decimal(raw: Any, label: str) -> Optional[str]:
            if raw is None:
                return None
            try:
                return format(
                    _finite_decimal(raw, label, nonnegative=True), "f")
            except RunPodError:
                return None

        def nullable_integral(
                raw: Any, label: str, minimum: int) -> Optional[int]:
            if raw is None or isinstance(raw, bool):
                return None
            try:
                parsed = _finite_decimal(raw, label, nonnegative=True)
            except RunPodError:
                return None
            integral = parsed.to_integral_value()
            if parsed != integral or integral < minimum:
                return None
            return int(integral)

        normalized = []
        for source in pods:
            if not isinstance(source, dict):
                raise RunPodError(
                    "RunPod pod listing contains a non-object row")
            pod = dict(source)
            pod_id = _provider_id(
                pod.get("id"), "RunPod pod listing id")
            if pod_id in seen:
                raise RunPodError(
                    "RunPod pod listing contains duplicate exact id %s" % pod_id)
            seen.add(pod_id)
            pod["id"] = pod_id
            pod["costPerHr"] = nullable_decimal(
                pod.get("costPerHr"), "RunPod pod costPerHr")
            pod["gpuCount"] = nullable_integral(
                pod.get("gpuCount"), "RunPod pod gpuCount", 1)
            pod["volumeInGb"] = nullable_integral(
                pod.get("volumeInGb"), "RunPod pod volumeInGb", 0)
            pod["containerDiskInGb"] = nullable_integral(
                pod.get("containerDiskInGb"),
                "RunPod pod containerDiskInGb", 0)
            runtime = pod.get("runtime")
            if isinstance(runtime, dict):
                runtime = dict(runtime)
                runtime["uptimeInSeconds"] = nullable_decimal(
                    runtime.get("uptimeInSeconds"),
                    "RunPod uptimeInSeconds")
                pod["runtime"] = runtime
            else:
                pod["runtime"] = None
            machine = pod.get("machine")
            if isinstance(machine, dict):
                machine = dict(machine)
                machine["currentPricePerGpu"] = nullable_decimal(
                    machine.get("currentPricePerGpu"),
                    "RunPod currentPricePerGpu")
                pod["machine"] = machine
            else:
                pod["machine"] = None
            normalized.append(pod)
        return normalized

    @staticmethod
    def _to_instance(p: Dict[str, Any]) -> Instance:
        runtime = p.get("runtime") or {}
        uptime_decimal = _finite_decimal(
            runtime.get("uptimeInSeconds") or 0,
            "RunPod uptimeInSeconds", nonnegative=True)
        uptime = _decimal_float(uptime_decimal, "RunPod uptimeInSeconds")
        rate_decimal = (
            Decimal("0") if p.get("costPerHr") is None else
            _finite_decimal(
                p["costPerHr"], "RunPod pod costPerHr", nonnegative=True))
        rate = _decimal_float(rate_decimal, "RunPod pod costPerHr")
        cost = rate * uptime / 3600.0
        if not math.isfinite(cost):
            raise RunPodError("RunPod accrued pod cost is not finite")
        machine = p.get("machine") or {}
        secure_cloud = machine.get("secureCloud")
        inst = Instance.from_json({
            "machine_id": 0,
            "status": p.get("desiredStatus") or "",
            "gpu_type": machine.get("gpuTypeId"),
            "num_gpus": p.get("gpuCount") or 1,
            "region": ("secure" if secure_cloud is True else
                       "community" if secure_cloud is False else None),
            "is_spot": False,
            "cost": cost,
            "runtime": uptime,
            "fs_id": None,
            "storage_gb": p.get("volumeInGb"),
            "name": p.get("name"),
            "pod_id": p.get("id"),
            "cost_per_hr": rate,
            "raw_pod": p,
        })
        inst.machine_id = p.get("id")
        return inst

    def list_instances(self) -> List[Instance]:
        return [self._to_instance(p) for p in self._pods()]

    def list_lifecycle_resources(self) -> List[Dict[str, Any]]:
        """Complete exact-id rows; every listed status remains live."""
        resources = []
        for pod in self._pods():
            machine = pod.get("machine") or {}
            resources.append({
                "id": str(pod["id"]),
                "name": pod.get("name"),
                "status": pod.get("desiredStatus"),
                "listed": True,
                "cost_per_hr": pod.get("costPerHr"),
                "runtime": pod.get("runtime"),
                "gpu_count": pod.get("gpuCount"),
                "gpu_type_id": machine.get("gpuTypeId"),
                "gpu_display_name": machine.get("gpuDisplayName"),
                "secure_cloud": machine.get("secureCloud"),
                "current_price_per_gpu": machine.get("currentPricePerGpu"),
                "provider_machine_id": machine.get("id"),
                "pod_host_id": machine.get("podHostId"),
                "volume_gb": pod.get("volumeInGb"),
                "container_disk_gb": pod.get("containerDiskInGb"),
                "network_volume_id": pod.get("networkVolumeId"),
                "image_name": pod.get("imageName"),
                # Some provider deployments expose this although it is not in
                # the stable documented Pod selection. Validate when present.
                "terminate_after": pod.get("terminateAfter"),
                "raw": pod,
            })
        return resources

    def get_lifecycle_resource(self, provider_id: Any) -> Optional[Dict[str, Any]]:
        """Exact-id detail; names are deliberately not accepted as ids."""
        wanted = _provider_id(provider_id)
        return next((row for row in self.list_lifecycle_resources()
                     if row["id"] == wanted), None)

    def validate_safe_resource_binding(
            self, provider_id: Any, *, expected_name: str,
            gpu_type_id: str, secure_cloud: bool, gpu_count: int,
            volume_gb: int, container_disk_gb: int, image_name: str,
            terminate_after: str) -> Dict[str, Any]:
        """Fail unless the live exact-id Pod is the resource requested."""
        observed = self.get_lifecycle_resource(provider_id)
        if observed is None:
            raise RunPodError(
                "created RunPod id %s is absent from the complete listing"
                % provider_id)
        expected_deadline = _exact_utc(terminate_after, "terminate_after")
        if not isinstance(secure_cloud, bool):
            raise RunPodError("secure_cloud expectation must be an exact bool")
        expected = {
            "name": str(expected_name),
            "gpu_type_id": str(gpu_type_id),
            "secure_cloud": secure_cloud,
            "gpu_count": int(gpu_count),
            "volume_gb": int(volume_gb),
            "container_disk_gb": int(container_disk_gb),
            "image_name": str(image_name),
        }
        problems = []
        for key, value in expected.items():
            actual = observed.get(key)
            if key == "secure_cloud":
                if not isinstance(actual, bool) or actual != value:
                    problems.append("%s expected %r, observed %r"
                                    % (key, value, observed.get(key)))
                continue
            if key in ("gpu_count", "volume_gb", "container_disk_gb"):
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    pass
            if actual != value:
                problems.append("%s expected %r, observed %r"
                                % (key, value, observed.get(key)))
        try:
            live_rate = _finite_decimal(
                observed.get("cost_per_hr"),
                "RunPod live pod costPerHr", positive=True)
        except RunPodError:
            problems.append(
                "cost_per_hr must be a known positive exact decimal")
        else:
            observed["cost_per_hr"] = format(live_rate, "f")
        if observed.get("network_volume_id") not in (None, ""):
            problems.append("network_volume_id must be absent, observed %r"
                            % observed.get("network_volume_id"))
        observed_deadline = observed.get("terminate_after")
        if (observed_deadline is not None
                and str(observed_deadline) != expected_deadline):
            problems.append("terminate_after expected %r, observed %r"
                            % (expected_deadline, observed_deadline))
        if problems:
            raise RunPodError("RunPod post-create identity mismatch: %s"
                              % "; ".join(problems))
        return {
            "provider_id": str(provider_id),
            "passed": True,
            "expected": dict(expected, terminate_after=expected_deadline,
                             network_volume_id=None),
            "observed": observed,
            "terminate_after_observable": observed_deadline is not None,
        }

    def attest_live_resource(
            self, provider_id: Any, *, expected_gpu_model: str,
            expected_vram_bytes: int, min_vcpu: int, min_ram_gb: int,
            volume_gb: int, container_disk_gb: int,
            workspace_available_bytes_minimum: int,
            container_available_bytes_minimum: int) -> Dict[str, Any]:
        """Read-only SSH qualification before upload or campaign RUNNING."""
        pod_id = _provider_id(provider_id)
        model = _gpu_id(expected_gpu_model, "expected GPU model")
        expected_numbers = {
            "expected_vram_bytes": expected_vram_bytes,
            "min_vcpu": min_vcpu,
            "min_ram_gb": min_ram_gb,
            "volume_gb": volume_gb,
            "container_disk_gb": container_disk_gb,
            "workspace_available_bytes_minimum":
                workspace_available_bytes_minimum,
            "container_available_bytes_minimum":
                container_available_bytes_minimum,
        }
        for key, value in expected_numbers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RunPodError("%s must be a positive integer" % key)
        expected = dict(expected_numbers, gpu_model=model)
        command = (
            "python3 -c 'import base64;"
            "exec(base64.b64decode(\"%s\").decode(\"utf-8\"))'"
            % base64.b64encode(
                _LIVE_ATTEST_SCRIPT.encode("utf-8")).decode("ascii"))
        observed = None
        transport_error = None
        controller_send_epoch = time.time()
        controller_receive_epoch = controller_send_epoch
        if self.dry:
            transport_error = "dry mode cannot attest a live resource"
            controller_receive_epoch = time.time()
        else:
            try:
                raw = self.exec_stdout(pod_id, command, timeout=180)
                observed = _strict_json_loads(raw)
            except Exception as exc:                          # noqa: BLE001
                transport_error = redact(str(exc))[:500]
            finally:
                controller_receive_epoch = time.time()
        round_trip_seconds = max(
            0.0, controller_receive_epoch - controller_send_epoch)
        remote_epoch = (
            observed.get("remote_time_epoch")
            if isinstance(observed, dict) else None)
        remote_utc = (
            observed.get("remote_time_utc")
            if isinstance(observed, dict) else None)
        remote_utc_epoch = None
        if isinstance(remote_utc, str):
            try:
                remote_utc_epoch = calendar.timegm(time.strptime(
                    _exact_utc(remote_utc, "remote attestation time"),
                    "%Y-%m-%dT%H:%M:%SZ"))
            except RunPodError:
                pass
        midpoint_epoch = (
            controller_send_epoch + round_trip_seconds / 2.0)
        allowed_skew_seconds = 30.0 + round_trip_seconds
        clock_skew_seconds = (
            abs(float(remote_epoch) - midpoint_epoch)
            if isinstance(remote_epoch, int)
            and not isinstance(remote_epoch, bool) else None)
        clock_ok = bool(
            clock_skew_seconds is not None
            and remote_utc_epoch == remote_epoch
            and clock_skew_seconds <= allowed_skew_seconds)
        clock = {
            "controller_send_epoch": controller_send_epoch,
            "controller_send_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(controller_send_epoch)),
            "controller_receive_epoch": controller_receive_epoch,
            "controller_receive_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(controller_receive_epoch)),
            "round_trip_seconds": round_trip_seconds,
            "remote_time_epoch": remote_epoch,
            "remote_time_utc": remote_utc,
            "clock_skew_seconds": clock_skew_seconds,
            "allowed_skew_seconds": allowed_skew_seconds,
            "within_bound": clock_ok,
        }
        failures = []
        checks: Dict[str, bool] = {"remote_clock": clock_ok}
        if not isinstance(observed, dict):
            failures.append("live SSH attestation unavailable")
        else:
            exact_observed = {
                "remote_time_epoch", "remote_time_utc",
                "logical_cpus", "memtotal_bytes", "effective_memory_bytes",
                "nvidia_smi_exit_code", "nvidia_smi_error", "gpus", "cuda",
                "filesystems",
            }
            if set(observed) != exact_observed:
                failures.append("live attestation keys differ")
            for key in (
                    "logical_cpus", "memtotal_bytes", "effective_memory_bytes",
                    "nvidia_smi_exit_code"):
                if (isinstance(observed.get(key), bool)
                        or not isinstance(observed.get(key), int)):
                    failures.append("%s is not an exact integer" % key)
            checks["logical_cpu_floor"] = (
                isinstance(observed.get("logical_cpus"), int)
                and not isinstance(observed.get("logical_cpus"), bool)
                and observed["logical_cpus"] >= min_vcpu)
            checks["memory_floor"] = (
                isinstance(observed.get("effective_memory_bytes"), int)
                and not isinstance(observed.get("effective_memory_bytes"), bool)
                and observed["effective_memory_bytes"] >= min_ram_gb * 10 ** 9)
            gpus = observed.get("gpus")
            checks["one_nvidia_gpu"] = (
                observed.get("nvidia_smi_exit_code") == 0
                and isinstance(gpus, list) and len(gpus) == 1)
            gpu = gpus[0] if checks["one_nvidia_gpu"] else {}
            if gpu and set(gpu) != {
                    "index", "name", "vram_bytes", "driver_version"}:
                failures.append("nvidia-smi GPU keys differ")
            observed_name = gpu.get("name") if isinstance(gpu, dict) else None
            observed_vram = (
                gpu.get("vram_bytes") if isinstance(gpu, dict) else None)
            checks["gpu_model"] = (
                isinstance(observed_name, str)
                and observed_name.strip().casefold() == model.casefold())
            vram_floor = expected_vram_bytes * 9 // 10
            vram_ceiling = expected_vram_bytes * 11 // 10
            checks["gpu_vram"] = (
                isinstance(observed_vram, int)
                and not isinstance(observed_vram, bool)
                and vram_floor <= observed_vram <= vram_ceiling)
            cuda = observed.get("cuda")
            if not isinstance(cuda, dict) or set(cuda) != {
                    "usable", "count", "name", "vram_bytes", "error"}:
                failures.append("CUDA attestation keys differ")
                cuda = {}
            checks["cuda_usable"] = (
                cuda.get("usable") is True and cuda.get("count") == 1
                and isinstance(cuda.get("name"), str)
                and cuda["name"].strip().casefold() == model.casefold()
                and isinstance(cuda.get("vram_bytes"), int)
                and vram_floor <= cuda["vram_bytes"] <= vram_ceiling)
            filesystems = observed.get("filesystems")
            if not isinstance(filesystems, dict) or set(filesystems) != {
                    "container", "workspace"}:
                failures.append("filesystem attestation keys differ")
                filesystems = {}
            filesystem_keys = {
                "path", "mount_point", "fs_type", "source", "device",
                "total_bytes", "available_bytes",
            }
            for role in ("container", "workspace"):
                row = filesystems.get(role)
                if not isinstance(row, dict) or set(row) != filesystem_keys:
                    failures.append("%s filesystem keys differ" % role)
            container = filesystems.get("container", {})
            workspace = filesystems.get("workspace", {})
            checks["container_disk_size"] = (
                isinstance(container.get("total_bytes"), int)
                and container["total_bytes"]
                >= container_disk_gb * 900_000_000)
            checks["workspace_volume_size"] = (
                isinstance(workspace.get("total_bytes"), int)
                and workspace["total_bytes"] >= volume_gb * 900_000_000)
            checks["workspace_mount"] = (
                workspace.get("path") == "/workspace"
                and workspace.get("mount_point") == "/workspace"
                and container.get("path") == "/"
                and (workspace.get("device"), workspace.get("source"))
                != (container.get("device"), container.get("source")))
            checks["container_available_bytes"] = (
                isinstance(container.get("available_bytes"), int)
                and not isinstance(container.get("available_bytes"), bool)
                and container["available_bytes"]
                >= container_available_bytes_minimum)
            checks["workspace_available_bytes"] = (
                isinstance(workspace.get("available_bytes"), int)
                and not isinstance(workspace.get("available_bytes"), bool)
                and workspace["available_bytes"]
                >= workspace_available_bytes_minimum)
            failures.extend(
                name for name, passed in sorted(checks.items()) if not passed)
        document = {
            "schema": "fidelity-suite/runpod-live-attestation.v2",
            "provider": "runpod", "provider_id": pod_id,
            "observed_at_utc": clock["controller_receive_utc"],
            "clock": clock,
            "expected": expected, "observed": observed,
            "transport_error": transport_error,
            "checks": checks, "failures": sorted(set(failures)),
            "ok": bool(not failures and transport_error is None
                       and checks and all(checks.values())),
        }
        return _attestation_seal(document)

    def list_network_volumes(self) -> List[Dict[str, Any]]:
        """Enumerate persistent chargeable volumes through the official REST API."""
        doc = self._get_v1("/networkvolumes")
        if not isinstance(doc, list):
            raise RunPodError(
                "RunPod network-volume listing returned non-list JSON")
        resources = []
        seen = set()
        for row in doc:
            if not isinstance(row, dict):
                raise RunPodError(
                    "RunPod network-volume listing contains a non-object row")
            volume_id = _provider_id(
                row.get("id"), "RunPod network-volume id")
            if not volume_id or volume_id in seen:
                raise RunPodError(
                    "RunPod network-volume listing has missing or duplicate id")
            seen.add(volume_id)
            size = row.get("size")
            if (isinstance(size, bool) or not str(size).isdigit()
                    or int(size) <= 0):
                raise RunPodError("RunPod network-volume size is invalid")
            normalized = dict(row)
            normalized["id"] = volume_id
            normalized["size"] = int(size)
            for field in (
                    "costPerHr", "costPerGbMonth", "pricePerGbMonth",
                    "monthlyCost"):
                if normalized.get(field) is not None:
                    normalized[field] = format(
                        _finite_decimal(
                            normalized[field], "RunPod network-volume %s" % field,
                            nonnegative=True), "f")
            resources.append({
                "id": volume_id,
                "name": normalized.get("name"),
                "size_gb": normalized["size"],
                "data_center_id": normalized.get("dataCenterId"),
                "raw": normalized,
            })
        return resources

    def chargeable_inventory(self) -> Dict[str, Any]:
        """Return pod plus network-volume inventory with explicit completeness.

        RunPod currently documents resource enumeration under REST v1; API v2
        exposes billing history but no documented network-volume list.  The
        source endpoints are named so a caller never mistakes an unavailable
        family for an empty one.
        """
        families: Dict[str, Dict[str, Any]] = {}
        try:
            pod_doc = self._get_v1(
                "/pods",
                {"includeNetworkVolume": "true", "includeWorkers": "true"})
            if not isinstance(pod_doc, list):
                raise RunPodError("RunPod pod inventory returned non-list JSON")
            pods = []
            seen_pods = set()
            for row in pod_doc:
                if not isinstance(row, dict):
                    raise RunPodError(
                        "RunPod pod inventory contains a non-object row")
                pod_id = _provider_id(
                    row.get("id"), "RunPod pod inventory id")
                if not pod_id or pod_id in seen_pods:
                    raise RunPodError(
                        "RunPod pod inventory has missing or duplicate id")
                seen_pods.add(pod_id)
                rate = _finite_decimal(
                    row.get("costPerHr"), "RunPod inventory costPerHr",
                    positive=True)
                adjusted_raw = row.get("adjustedCostPerHr")
                adjusted = (
                    None if adjusted_raw is None else _finite_decimal(
                        adjusted_raw, "RunPod inventory adjustedCostPerHr",
                        nonnegative=True))
                volume = row.get("networkVolume")
                if volume is not None and not isinstance(volume, dict):
                    raise RunPodError(
                        "RunPod attached network volume is not an object")
                attached = (volume.get("id") if isinstance(volume, dict)
                            else row.get("networkVolumeId"))
                if isinstance(volume, dict):
                    attached = str(attached or "").strip()
                    size = volume.get("size")
                    if (not attached or isinstance(size, bool)
                            or not str(size).isdigit() or int(size) <= 0):
                        raise RunPodError(
                            "RunPod attached network volume identity is invalid")
                pods.append({
                    "id": pod_id,
                    "name": row.get("name"),
                    "status": row.get("desiredStatus"),
                    "cost_per_hr": format(rate, "f"),
                    "adjusted_cost_per_hr": (
                        None if adjusted is None else format(adjusted, "f")),
                    "network_volume_id": (
                        str(attached) if attached is not None else None),
                    "network_volume": volume,
                    "raw": row,
                })
            families["pods"] = {
                "complete": True,
                "source": "GET https://rest.runpod.io/v1/pods"
                          "?includeNetworkVolume=true&includeWorkers=true",
                "resources": pods,
            }
        except RunPodError as exc:
            families["pods"] = {
                "complete": False,
                "source": "GET https://rest.runpod.io/v1/pods",
                "resources": [],
                "unknown": redact(str(exc)),
            }
        try:
            volumes = self.list_network_volumes()
            families["network_volumes"] = {
                "complete": True,
                "source": "GET https://rest.runpod.io/v1/networkvolumes",
                "resources": volumes,
            }
        except RunPodError as exc:
            families["network_volumes"] = {
                "complete": False,
                "source": "GET https://rest.runpod.io/v1/networkvolumes",
                "resources": [],
                "unknown": redact(str(exc)),
            }
        unknown = sorted(name for name, family in families.items()
                         if not family["complete"])
        return {
            "schema": "fidelity-suite/runpod-chargeable-inventory.v1",
            "provider": "runpod",
            "observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": not unknown,
            "unknown_families": unknown,
            "families": families,
        }

    def billing_history(self, pod_id: Any, *, start_time: str, end_time: str,
                        bucket_size: str = "hour") -> Dict[str, Any]:
        """Validate the exact official v2 pod-billing response."""
        wanted = _provider_id(pod_id)
        start = _exact_utc(start_time, "start_time")
        end = _exact_utc(end_time, "end_time")
        start_epoch = calendar.timegm(time.strptime(
            start, "%Y-%m-%dT%H:%M:%SZ"))
        end_epoch = calendar.timegm(time.strptime(
            end, "%Y-%m-%dT%H:%M:%SZ"))
        if end_epoch <= start_epoch:
            raise RunPodError("billing end_time must follow start_time")
        if bucket_size != "hour":
            raise RunPodError("billing bucket_size must be hour")
        query = {"podId": wanted, "startTime": start,
                 "endTime": end, "bucketSize": bucket_size}
        doc = self._get_v2("/billing/pods", query)
        if not isinstance(doc, dict) or set(doc) != {"records", "metadata"}:
            raise RunPodError(
                "RunPod billing response keys differ from official schema")
        records = doc["records"]
        metadata = doc["metadata"]
        if not isinstance(records, list) or not records:
            raise RunPodError(
                "RunPod billing has no record for pod %s yet; reconciliation "
                "remains unresolved" % wanted)
        if not isinstance(metadata, dict) or set(metadata) != {
                "query", "recordCount", "uniquePodCount", "totals"}:
            raise RunPodError(
                "RunPod billing metadata keys differ from official schema")
        observed_query = metadata["query"]
        totals = metadata["totals"]
        if (not isinstance(observed_query, dict)
                or set(observed_query) != {
                    "startTime", "endTime", "bucketSize", "podId"}
                or not isinstance(totals, dict)):
            raise RunPodError(
                "RunPod billing metadata query or totals keys differ")
        required_amounts = ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount")
        if set(totals) != set(required_amounts):
            raise RunPodError("RunPod billing totals keys differ")
        resolved_start = _exact_utc(
            observed_query["startTime"],
            "RunPod billing metadata startTime")
        resolved_end = _exact_utc(
            observed_query["endTime"],
            "RunPod billing metadata endTime")
        resolved_start_epoch = calendar.timegm(time.strptime(
            resolved_start, "%Y-%m-%dT%H:%M:%SZ"))
        resolved_end_epoch = calendar.timegm(time.strptime(
            resolved_end, "%Y-%m-%dT%H:%M:%SZ"))
        requested_start_offset = start_epoch - resolved_start_epoch
        requested_end_offset = resolved_end_epoch - end_epoch
        if (observed_query.get("podId") != wanted
                or observed_query.get("bucketSize") != bucket_size
                or resolved_start_epoch % 3600
                or resolved_end_epoch % 3600
                or not 0 <= requested_start_offset < 3600
                or not 0 <= requested_end_offset < 3600
                or resolved_end_epoch <= resolved_start_epoch):
            raise RunPodError(
                "RunPod billing metadata resolved query is inconsistent")
        for field, expected in (
                ("recordCount", len(records)), ("uniquePodCount", 1)):
            value = metadata[field]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value != expected):
                raise RunPodError(
                    "RunPod billing metadata %s is inconsistent" % field)

        def amount(value: Any, label: str) -> Decimal:
            try:
                parsed = Decimal(str(value))
            except (InvalidOperation, ValueError):
                raise RunPodError("%s is not decimal" % label)
            if not parsed.is_finite() or parsed < 0:
                raise RunPodError("%s is invalid" % label)
            return parsed

        sums = {key: Decimal("0") for key in required_amounts}
        bucket_ranges = []
        prior_end = None
        for index, row in enumerate(records):
            if not isinstance(row, dict) or set(row) != {
                    "startTime", "endTime", "podId",
                    "totalAmount", "gpuAmount", "cpuAmount", "diskAmount"}:
                raise RunPodError(
                    "RunPod billing record keys differ from official schema")
            if row["podId"] != wanted:
                raise RunPodError(
                    "RunPod billing returned a non-matching pod record")
            bucket_start = _exact_utc(
                row["startTime"], "RunPod billing record startTime")
            bucket_end = _exact_utc(
                row["endTime"], "RunPod billing record endTime")
            bucket_start_epoch = calendar.timegm(time.strptime(
                bucket_start, "%Y-%m-%dT%H:%M:%SZ"))
            bucket_end_epoch = calendar.timegm(time.strptime(
                bucket_end, "%Y-%m-%dT%H:%M:%SZ"))
            if (bucket_start_epoch % 3600 or bucket_end_epoch % 3600
                    or bucket_end_epoch - bucket_start_epoch != 3600
                    or bucket_start_epoch < resolved_start_epoch
                    or bucket_end_epoch > resolved_end_epoch
                    or (index == 0
                        and bucket_start_epoch != resolved_start_epoch)
                    or (prior_end is not None
                        and bucket_start_epoch != prior_end)):
                raise RunPodError(
                    "RunPod billing buckets do not provide a complete, "
                    "contiguous resolved window")
            prior_end = bucket_end_epoch
            bucket_ranges.append({
                "startTime": bucket_start, "endTime": bucket_end})
            for key in required_amounts:
                sums[key] += amount(
                    row[key], "RunPod billing record %s" % key)
        if prior_end != resolved_end_epoch:
            raise RunPodError(
                "RunPod billing records omit the end of the resolved window")
        for key in required_amounts:
            total = amount(
                totals[key], "RunPod billing total %s" % key)
            if not _billing_total_matches_record_sum(total, sums[key]):
                raise RunPodError(
                    "RunPod billing total %s does not equal record sum "
                    "within the bounded provider rounding tolerance" % key)
        return {
            "schema": "fidelity-suite/runpod-billing-evidence.v2",
            "provider": "runpod",
            "pod_id": wanted,
            "query": query,
            "records": records,
            "metadata": metadata,
            "validated_record_sums": {
                key: format(sums[key], "f") for key in required_amounts},
            "validated_bucket_ranges": bucket_ranges,
            "retrieved_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def reconcile_billing(self, lease: Dict[str, Any]) -> Dict[str, Any]:
        """Return only a post-absence, independently stable billing closure."""
        ids = sorted({str(value) for value
                      in lease.get("provider_resource_ids") or []
                      if str(value).strip()})
        if not ids:
            raise RunPodError(
                "RunPod billing reconciliation needs at least one exact pod id")
        create = lease.get("create") or {}
        start = _exact_utc(create.get("pre_create_observed_at"),
                           "lease pre_create_observed_at")
        absence_events = [
            item for item in lease.get("history") or []
            if item.get("to") == "ABSENCE_CONFIRMED"
        ]
        if not absence_events:
            raise RunPodError("lease has no provider-absence event")
        end = _exact_utc(absence_events[-1].get("at"), "lease absence time")
        absence_epoch = calendar.timegm(time.strptime(
            end, "%Y-%m-%dT%H:%M:%SZ"))
        stabilization_seconds = 300
        if time.time() - absence_epoch < stabilization_seconds:
            raise RunPodError(
                "RunPod billing remains inside the 300-second "
                "post-absence stabilization window")

        def retrieve() -> Dict[str, Any]:
            histories = []
            total = Decimal("0")
            for pod_id in ids:
                history = self.billing_history(
                    pod_id, start_time=start, end_time=end)
                raw_total = history["metadata"]["totals"]["totalAmount"]
                try:
                    amount = Decimal(str(raw_total))
                except (InvalidOperation, ValueError):
                    raise RunPodError(
                        "RunPod billing totalAmount is not an exact decimal "
                        "for %s" % pod_id)
                if not amount.is_finite() or amount < 0:
                    raise RunPodError(
                        "RunPod billing totalAmount is invalid for %s" % pod_id)
                total += amount
                histories.append(history)
            return {
                "reconciled": True,
                "provider": "runpod",
                "provider_resource_ids": ids,
                "billing_histories": histories,
                "total_amount": format(total, "f"),
                "evidence": {
                    "schema":
                        "fidelity-suite/runpod-billing-retrieval.v1",
                    "retrieval_id": secrets.token_hex(12),
                    "retrieved_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            }

        def closure(evidence: Dict[str, Any]) -> Dict[str, Any]:
            result = json.loads(json.dumps(
                evidence, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False))
            result.pop("evidence", None)
            for history in result["billing_histories"]:
                history.pop("retrieved_at_utc", None)
            return result

        first = retrieve()
        second = retrieve()
        first_closure = closure(first)
        second_closure = closure(second)
        if first_closure != second_closure:
            raise RunPodError(
                "RunPod billing changed between independent retrievals")
        result = dict(second)
        result["evidence"] = {
            "schema": "fidelity-suite/runpod-billing-stabilization.v1",
            "absence_confirmed_at": end,
            "minimum_stabilization_seconds": stabilization_seconds,
            "closure_sha256": hashlib.sha256(json.dumps(
                second_closure, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest(),
            "first_retrieval": first["evidence"],
            "second_retrieval": second["evidence"],
        }
        return result

    def get(self, machine_id: Any) -> Optional[Instance]:
        provider_id = _provider_id(machine_id)
        for pod in self._pods():
            if str(pod.get("id")) == provider_id:
                return self._to_instance(pod)
        return None

    def _validated_ssh_public_key(self) -> str:
        """Validate the exact unattended SSH identity before any paid POST."""
        private_path = os.path.abspath(os.path.expanduser(self.ssh_key))
        public_path = private_path + ".pub"
        private_fd = None
        public_fd = None
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            private_fd = os.open(private_path, os.O_RDONLY | nofollow)
            public_fd = os.open(public_path, os.O_RDONLY | nofollow)
            private = os.fstat(private_fd)
            public = os.fstat(public_fd)
            if not stat.S_ISREG(private.st_mode):
                raise RunPodError(
                    "SSH private key must be a regular non-symlink file")
            if stat.S_IMODE(private.st_mode) & 0o077:
                raise RunPodError(
                    "SSH private key must not grant group/other access")
            if not stat.S_IMODE(private.st_mode) & 0o400:
                raise RunPodError("SSH private key must be owner-readable")
            if hasattr(os, "getuid") and private.st_uid != os.getuid():
                raise RunPodError(
                    "SSH private key must be owned by the current user")
            if not stat.S_ISREG(public.st_mode):
                raise RunPodError(
                    "SSH public key must be a regular non-symlink file")
            if hasattr(os, "getuid") and public.st_uid != os.getuid():
                raise RunPodError(
                    "SSH public key must be owned by the current user")
            with os.fdopen(public_fd, encoding="utf-8") as fh:
                public_fd = None
                lines = fh.read().splitlines()
            if len(lines) != 1 or not lines[0].strip():
                raise RunPodError(
                    "SSH public key must be exactly one nonempty line")
            supplied = _canonical_public_key(lines[0])
            fields = supplied.split()
            try:
                derived = subprocess.run(
                    ["ssh-keygen", "-y",
                     "-f", "/proc/self/fd/%d" % private_fd],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=15,
                    pass_fds=(private_fd,))
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RunPodError(
                    "cannot verify SSH private/public key match: %s" % exc)
            derived_fields = derived.stdout.strip().split()
            if (derived.returncode != 0 or len(derived_fields) < 2
                    or derived_fields[:2] != fields[:2]):
                raise RunPodError("SSH .pub does not match the private key")
            return supplied
        except RunPodError:
            raise
        except (OSError, UnicodeError) as exc:
            raise RunPodError(
                "SSH safe profile key pair is unavailable or invalid: %s" % exc)
        finally:
            if public_fd is not None:
                os.close(public_fd)
            if private_fd is not None:
                os.close(private_fd)

    def prepare_safe_create(self, **kw) -> PreparedRunPodCreate:
        forbidden = (
            "network_volume_id", "networkVolumeId", "network_volume",
            "network_mounts", "mounts",
        )
        used = [key for key in forbidden if kw.get(key) is not None]
        mount_path = kw.get("volume_mount_path")
        if mount_path is not None and str(mount_path) != "/workspace":
            used.append("volume_mount_path")
        if used:
            raise RunPodError(
                "safe RunPod profile refuses network/custom mounts: %s"
                % ", ".join(sorted(used)))
        native = [key for key in ("docker_cmd", "docker_entrypoint", "docker_args")
                  if kw.get(key) is not None and kw.get(key) != ""]
        if native:
            raise RunPodError(
                "safe RunPod profile is SSH-driven and refuses native docker launch: %s"
                % ", ".join(sorted(native)))
        if kw.get("env"):
            raise RunPodError(
                "safe RunPod SSH profile refuses caller-supplied provider env")
        if kw.get("spot", False) is not False:
            raise RunPodError(
                "safe RunPod profile requires spot exactly false")
        if kw.get("offer", "on-demand") != "on-demand":
            raise RunPodError(
                "safe RunPod profile requires offer exactly on-demand")

        def _positive_int(key, default=None):
            value = kw.get(key, default)
            if isinstance(value, bool):
                raise RunPodError("%s must be a positive integer" % key)
            if isinstance(value, int):
                parsed = value
            elif isinstance(value, str) and value.isdigit():
                parsed = int(value)
            else:
                raise RunPodError("%s must be a positive integer" % key)
            if parsed <= 0:
                raise RunPodError("%s must be a positive integer" % key)
            return parsed

        terminate_after = _terminate_after(kw)
        if terminate_after is None:
            raise RunPodError(
                "safe RunPod create requires terminate_after or "
                "terminate_after_epoch")
        termination_epoch = calendar.timegm(
            time.strptime(terminate_after, "%Y-%m-%dT%H:%M:%SZ"))
        if termination_epoch - time.time() < MIN_CREATE_SETUP_SECONDS:
            raise RunPodError(
                "RunPod terminate_after must be at least %d seconds in the future"
                % MIN_CREATE_SETUP_SECONDS)
        gpu = _gpu_id(
            kw.get("gpu_type") or kw.get("gpu"), "RunPod create gpu_type")
        name = str(kw.get("name") or "").strip()
        if not name:
            raise RunPodError("safe RunPod create requires an exact lease name")
        region = kw.get("region")
        if region != "secure":
            raise RunPodError("safe RunPod region must be exactly secure")
        volume_gb = _positive_int("storage_gb")
        container_disk_gb = _positive_int("container_disk_gb")
        gpu_count = _positive_int("num_gpus", 1)
        min_vcpu = _positive_int("min_vcpu", 4)
        min_ram_gb = _positive_int("min_ram_gb", 16)
        image = str(kw.get("image") or DEFAULT_IMAGE).strip()
        if not image:
            raise RunPodError("RunPod image must be nonempty")
        public_key = _canonical_public_key(
            self._validated_ssh_public_key())

        request_identity = {
            "cloud_type": "SECURE",
            "is_spot": False,
            "offer": "on-demand",
            "gpu_type_id": gpu,
            "gpu_count": gpu_count,
            "volume_gb": volume_gb,
            "container_disk_gb": container_disk_gb,
            "min_vcpu": min_vcpu,
            "min_ram_gb": min_ram_gb,
            "name": name,
            "image_name": image,
            "terminate_after": terminate_after,
            "ports": "22/tcp",
            "volume_mount_path": "/workspace",
            "network_volume_id": None,
            "public_key_sha256": hashlib.sha256(
                public_key.encode("utf-8")).hexdigest(),
        }

        env_gql = '{key:"PUBLIC_KEY", value:%s}' % _gql_string(
            public_key, "RunPod SSH public key")
        query = ('mutation { podFindAndDeployOnDemand(input:{'
                 'cloudType:%s, gpuCount:%d, volumeInGb:%d, '
                 'containerDiskInGb:%d, minVcpuCount:%d, minMemoryInGb:%d, '
                 'gpuTypeId:%s, name:%s, imageName:%s, '
                 'terminateAfter:%s, ports:"22/tcp", '
                 'volumeMountPath:"/workspace", env:[%s] '
                 '}) { id name costPerHr } }'
                 % (request_identity["cloud_type"], gpu_count, volume_gb,
                    container_disk_gb, min_vcpu, min_ram_gb,
                    _gql_string(gpu, "RunPod create gpu_type"),
                    _gql_string(name, "RunPod create name"),
                    _gql_string(image, "RunPod create image"),
                    _gql_string(terminate_after, "RunPod terminate_after"),
                    env_gql))
        if self._key is None:
            self._key = _load_key(self._key_file)
        body = json.dumps({"query": query}).encode("utf-8")
        http_request = urllib.request.Request(
            GQL, data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "quant-fidelity-suite/0.1",
                "Authorization": "Bearer " + self._key,
            })
        identity_json = json.dumps(
            request_identity, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
        return PreparedRunPodCreate(
            http_request=http_request,
            http_opener=urllib.request.build_opener(_NoMutationRedirect()),
            graphql_body=body, request_identity_json=identity_json, name=name,
            terminate_after=terminate_after, storage_gb=volume_gb,
            container_disk_gb=container_disk_gb, image_name=image,
            dry_run=self.dry)

    def submit_prepared_create(
            self, prepared: PreparedRunPodCreate) -> Dict[str, Any]:
        if prepared.dry_run:
            return {
                "dry_run": True,
                "request": prepared.to_dict()["request_identity"],
                "prepared_create": prepared.to_dict(),
            }
        try:
            with prepared.http_opener.open(
                    prepared.http_request, timeout=180) as response:
                document = _response_json(response, GQL, "GraphQL create")
        except urllib.error.HTTPError as exc:
            raise RunPodError("RunPod HTTP %d: %s"
                              % (exc.code, redact(
                                  exc.read(300).decode("utf-8", "replace"))))
        except RunPodError:
            raise
        except Exception as exc:                          # noqa: BLE001
            raise RunPodError(
                "RunPod prepared create request failed: %s"
                % redact(str(exc)))
        if not isinstance(document, dict):
            raise RunPodError("RunPod GraphQL create returned non-object JSON")
        if document.get("errors"):
            raise RunPodError("RunPod GraphQL create: %s"
                              % redact(json.dumps(document["errors"])[:300]))
        data = document.get("data")
        if not isinstance(data, dict):
            raise RunPodError("RunPod GraphQL create lacks object data")
        pod = data.get("podFindAndDeployOnDemand")
        if not isinstance(pod, dict):
            raise RunPodError("RunPod create response is not an exact pod object")
        pod_id = _provider_id(
            pod.get("id"), "RunPod create response pod id")
        response_name = pod.get("name")
        response_cost = pod.get("costPerHr")
        response_evidence = {
            "id": pod_id, "name": response_name,
            "cost_per_hr": response_cost,
        }
        if (response_name not in (None, "")
                and (not isinstance(response_name, str)
                     or response_name != prepared.name)):
            raise RunPodCreateResponseError(
                "RunPod create response name does not match exact lease name",
                pod_id, response_evidence)
        return {
            "machine_id": pod_id, "pod_id": pod_id,
            "name": response_name, "cost_per_hr": response_cost,
            "request": prepared.to_dict()["request_identity"],
            "prepared_create": prepared.to_dict(),
            "requested_terminate_after": prepared.terminate_after,
            "storage_gb": prepared.storage_gb,
            "container_disk_gb": prepared.container_disk_gb,
            "image_name": prepared.image_name,
        }

    def create(self, **kw) -> Dict[str, Any]:
        return self.submit_prepared_create(self.prepare_safe_create(**kw))

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        pod_id = _provider_id(machine_id)
        if self.dry:
            return {"dry_run": True}
        self._gql(
            "mutation { podTerminate(input:{podId:%s}) }"
            % _gql_string(pod_id, "RunPod pod id"))
        return {"terminated": pod_id}
    def pause(self, machine_id: Any) -> Dict[str, Any]:
        raise RunPodError(
            "safe RunPod profile refuses pause/hold; destroy at teardown")

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        raise RunPodError(
            "safe RunPod profile refuses resume/recovery; create retries are "
            "not authorized")

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        pid = _provider_id(machine_id)
        if pid in self._ssh_cache:
            return self._ssh_cache[pid]
        deadline = time.time() + wait
        while time.time() < deadline:
            for p in self._pods():
                if p.get("id") != pid:
                    continue
                for port in ((p.get("runtime") or {}).get("ports") or []):
                    if port.get("privatePort") == 22 and port.get("isIpPublic"):
                        try:
                            endpoint_ip = str(ipaddress.ip_address(port.get("ip")))
                            endpoint_port = port.get("publicPort")
                            if (isinstance(endpoint_port, bool)
                                    or not isinstance(endpoint_port, int)
                                    or not 1 <= endpoint_port <= 65535):
                                raise ValueError("port outside 1..65535")
                        except (TypeError, ValueError) as exc:
                            raise RunPodError(
                                "pod %s exposed a malformed public SSH endpoint"
                                % pid) from exc
                        ep = (endpoint_ip, endpoint_port)
                        self._await_ssh(ep[0], ep[1],
                                        wait=max(60.0, deadline - time.time()))
                        self._ssh_cache[pid] = ep
                        return ep
            time.sleep(10)
        raise RunPodError("pod %s exposed no public SSH port within %ds "
                          "(it may still be provisioning)" % (pid, int(wait)))

    # exec / exec_stdout / upload / download / run_job / run_status /
    # run_logs all come from SSHTransport. They were written here first and
    # then needed again for Vast and Lambda; keeping three copies of a
    # transport whose two subtle bugs (a detached job that never records its
    # exit code, and liveness read from a pid that has already forked away)
    # cost real money to find is how the third copy reintroduces them.

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        """RunPod storage is not separable from the pod.

        A JarvisLabs filesystem outlives its instance, which is what makes a
        preempted spot box cheap to resume. A RunPod volume is created with the
        pod and dies with it. Rather than pretend, this records the request and
        the controller creates the pod with `volumeInGb` set to the same size.
        """
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "runpod volumes are pod-scoped; created with the pod"}

    def fs_delete(self, fs_id: Any) -> Any:
        return {"deleted": False,
                "note": "no separable filesystem on runpod; the volume went "
                        "with the pod at podTerminate"}
