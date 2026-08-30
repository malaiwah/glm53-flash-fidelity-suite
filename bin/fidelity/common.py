"""Shared plumbing: secret redaction, canonical JSON, subprocess, console.

Stdlib only, on purpose.  Both runners are meant to be copy-pasted onto a
stock machine and run with the system `python3`; a dependency here would turn
a one-paste recipe into a virtualenv tutorial.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

# Shapes we redact even when we were never told the value: HF user tokens, HF
# org tokens, and anything the caller registered.  This is belt-and-braces --
# the runners never put a token on a command line in the first place -- but a
# stray `env` in a log, or a library that echoes its own auth header, should
# not be able to leak one through us.
_TOKEN_SHAPES = [
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bapi_org_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    # Every `jl list/get/create --json` record carries a Jupyter URL with a
    # live 64-char access token in the query string. We never serialize those
    # records, but a debug dump or a pasted traceback would, and that token is
    # a working credential for the instance.
    re.compile(r"(?i)\btoken=[A-Za-z0-9._\-]{24,}"),
    re.compile(r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._\-]{20,}"),
]

_REGISTERED: List[str] = []


def register_secret(value: Optional[str]) -> None:
    """Add a literal secret value to the redaction set.

    Call this the moment a token is read, before it can reach any stream.
    """
    if value and len(value) >= 8 and value not in _REGISTERED:
        _REGISTERED.append(value)


def redact(text: str) -> str:
    if not text:
        return text
    for secret in _REGISTERED:
        text = text.replace(secret, "***REDACTED***")
    for pattern in _TOKEN_SHAPES:
        text = pattern.sub("***REDACTED***", text)
    return text


# --------------------------------------------------------------------------
# Canonical JSON + hashing (must match registry/tools/registry_lib.py exactly)
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(doc: Dict[str, Any], field: str = "receipt_sha256") -> Dict[str, Any]:
    """Self-seal a receipt: sha256 over the canonical form with the seal blanked.

    Deliberately the same four-line recipe the registry documents to
    contributors, so a stranger can verify our receipts with `python3 -c` and
    no imports from us.
    """
    body = dict(doc)
    body[field] = ""
    doc = dict(doc)
    doc[field] = sha256_hex(canonical_json(body))
    return doc


def verify_seal(doc: Dict[str, Any], field: str = "receipt_sha256") -> bool:
    body = dict(doc)
    claimed = body.get(field, "")
    body[field] = ""
    return sha256_hex(canonical_json(body)) == claimed


def write_json(path: str, obj: Any) -> None:
    """Write a receipt atomically.

    The temp file used to be the fixed name `path + ".tmp"`, so two processes writing
    the same output path interleaved into ONE staging file and the survivor's
    `os.replace` published a mixture; and when the destination was a directory, the
    replace failed AFTER the temp file had been written, leaving it behind forever.
    A unique name per writer, removed on failure, fixes both. The fsync matters here
    specifically: these are receipts, and the machine that writes one is often
    destroyed minutes later."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, prefix=".receipt-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------

_T0 = time.time()


def _stamp() -> str:
    el = int(time.time() - _T0)
    return "%02d:%02d" % (el // 3600, (el % 3600) // 60)


class Console:
    def __init__(self, quiet: bool = False, stream=None) -> None:
        self.quiet = quiet
        self.stream = stream or sys.stdout

    def _w(self, text: str) -> None:
        self.stream.write(redact(text) + "\n")
        self.stream.flush()

    def rule(self, width: int = 78) -> None:
        self._w("-" * width)

    def say(self, text: str = "") -> None:
        self._w(text)

    def step(self, text: str) -> None:
        self._w("  %s  %s" % (_stamp(), text))

    def ok(self, label: str, detail: str = "") -> None:
        self._w("  %-38s ok%s" % (label, ("  " + detail) if detail else ""))

    def warn(self, text: str) -> None:
        self._w("  WARNING  " + text)

    def err(self, text: str) -> None:
        sys.stderr.write(redact("  ERROR  " + text) + "\n")
        sys.stderr.flush()

    def kv(self, key: str, value: Any, indent: int = 2) -> None:
        self._w("%s%-22s %s" % (" " * indent, key, value))


# --------------------------------------------------------------------------
# Subprocess
# --------------------------------------------------------------------------


class CommandError(RuntimeError):
    def __init__(self, argv: Sequence[str], code: int, out: str, err: str) -> None:
        self.argv, self.code, self.out, self.err = list(argv), code, out, err
        super().__init__(
            "command failed (%d): %s\n%s" % (code, redact(" ".join(argv)), redact(err or out))
        )


def run(
    argv: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    stdin_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a command, capture both streams, redact before anything is shown.

    Never `shell=True`: everything here takes a real argv, so a repo id with a
    shell metacharacter in it cannot become a command.
    """
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
        input=stdin_text,
    )
    if check and proc.returncode != 0:
        raise CommandError(argv, proc.returncode, proc.stdout, proc.stderr)
    return proc


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000.0:
            return "%.2f %s" % (n, unit)
        n /= 1000.0
    return "%.2f PB" % n


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return "%dh %02dm" % (h, m)
    if m:
        return "%dm %02ds" % (m, s)
    return "%ds" % s


def parse_duration(text: str) -> float:
    """Accept 6h, 90m, 3600, 1h30m."""
    text = str(text).strip().lower()
    if not text:
        raise ValueError("empty duration")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    total, matched = 0.0, False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        matched = True
    if not matched:
        raise ValueError("cannot parse duration %r (try 6h, 90m, 5400)" % text)
    return total


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# Registry library loader (READ-ONLY import of the registry's own code)
# --------------------------------------------------------------------------


def load_registry_lib(suite_root) -> Any:
    """Load registry/tools/registry_lib.py READ-ONLY, by file path.

    The derived values the registry's comparability guarantee rests on
    (comparability.key, scope_digest) must be computed by the registry's OWN
    code, imported, never reimplemented: two implementations of a hash function
    is two chances to disagree, and the disagreement would surface as a
    rejected submission months later.  Nothing else under registry/tools may be
    imported from bin/ -- registry_add/registry_validate are heavyweight and
    may be edited concurrently.
    """
    import importlib.util
    from pathlib import Path

    path = Path(suite_root) / "registry" / "tools" / "registry_lib.py"
    if not path.is_file():
        raise RuntimeError(
            "registry/tools/registry_lib.py not found under %s; the derived "
            "fields (scope_digest, comparability key) must be computed by the "
            "registry's own code, not reimplemented here" % suite_root
        )
    spec = importlib.util.spec_from_file_location("_registry_lib", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
