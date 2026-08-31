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

import io
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:                                        # inside the suite
    from fidelity.common import register_secret, safe_urlopen, sha256_file
except Exception:                           # pragma: no cover - standalone
    def register_secret(value): return None

    def safe_urlopen(request, *, timeout=60.0):
        return urllib.request.urlopen(request, timeout=timeout)

    def sha256_file(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

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
EXCLUDE_DIRS = (".secrets", ".stream-work")


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


def _relevant(fs_root):
    """The files a caller actually wants, newest-first by directory."""
    out = []
    receipts = Path(fs_root) / "receipts"
    if receipts.is_dir():
        for path in sorted(receipts.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(fs_root)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            out.append(path)
    job = Path(fs_root) / "job.json"
    if job.is_file():
        out.append(job)
    return out


def build_summary(fs_root, verb, status, stages, pin=None, failed_stage=None):
    """The small JSON that every sink carries, whatever else it carries.

    Digests, not bytes: a log line that survives truncation still identifies
    the artifact well enough to match it against one fetched another way.
    """
    fs_root = Path(fs_root)
    files = []
    for path in _relevant(fs_root):
        try:
            files.append({"path": str(path.relative_to(fs_root)),
                          "bytes": path.stat().st_size,
                          "sha256": sha256_file(str(path))})
        except OSError:
            continue
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


def _bundle(fs_root, summary):
    """receipts + job.json + the summary, as tar.gz bytes. No secrets."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in _relevant(fs_root):
            tar.add(str(path), arcname=str(path.relative_to(fs_root)))
        blob = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo("result-summary.json")
        info.size = len(blob)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


def _deliver_stdout(fs_root, summary, con):
    """The universal channel: frame the answer into the container log."""
    print(BEGIN, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
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
    return {"scheme": "stdout", "ok": True, "files": len(summary["files"])}


def _deliver_file(fs_root, summary, target, con):
    dest = Path(target)
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in _relevant(fs_root):
        out = dest / path.relative_to(fs_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(out))
        copied += 1
    (dest / "result-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    con("result sink file: %d file(s) -> %s" % (copied, dest))
    return {"scheme": "file", "ok": True, "files": copied, "target": str(dest)}


def _deliver_http(fs_root, summary, url, con, *, method=None, timeout=120.0):
    """PUT the bundle. PUT because a presigned upload URL is a PUT.

    An `Authorization` value is taken from FIDELITY_RESULT_SINK_AUTH when the
    endpoint needs a header instead of a signature in the URL -- from the
    environment, never argv.
    """
    body = _bundle(fs_root, summary)
    method = (method or os.environ.get("FIDELITY_RESULT_SINK_METHOD") or "PUT").upper()
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/gzip")
    req.add_header("Content-Length", str(len(body)))
    # ntfy renders these; every other endpoint ignores them.
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
    con("result sink https: %d bytes -> %s (HTTP %s)" % (len(body), _host(url), code))
    return {"scheme": "http", "ok": True, "bytes": len(body), "code": code}


def _host(url):
    """Never echo a presigned URL: the query string is the credential."""
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        return "%s://%s%s" % (parts.scheme, parts.netloc, parts.path)
    except Exception:
        return "(url)"


def deliver(fs_root, sinks, summary, con):
    """Run every sink. stdout first, and one failure never eats the rest.

    A sink that cannot deliver is reported and the run's own exit code is NOT
    changed by it: the measurement either happened or it did not, and a
    collector being down is not a measurement result.
    """
    results = []
    for sink in sinks:
        try:
            if sink.scheme == "stdout":
                results.append(_deliver_stdout(fs_root, summary, con))
            elif sink.scheme == "file":
                results.append(_deliver_file(fs_root, summary, sink.target, con))
            elif sink.scheme == "http":
                results.append(_deliver_http(fs_root, summary, sink.target, con))
        except SinkError as exc:
            con("RESULT SINK FAILED (%s): %s" % (sink.scheme, exc))
            results.append({"scheme": sink.scheme, "ok": False, "error": str(exc)})
        except Exception as exc:            # a sink must never mask the run
            con("RESULT SINK FAILED (%s): %s" % (sink.scheme, exc.__class__.__name__))
            results.append({"scheme": sink.scheme, "ok": False,
                            "error": exc.__class__.__name__})
    return results
