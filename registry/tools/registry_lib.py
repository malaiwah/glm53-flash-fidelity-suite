"""Shared primitives for the quant-fidelity-registry tools.

Everything in this module is pure, deterministic and OFFLINE. No module in this
package may import a networking library; tools/registry_validate.py --offline-selftest
asserts that over the whole transitive import graph.

The two derived values that the registry's comparability guarantee rests on --
comparability.key and scope_digest -- are computed HERE and nowhere else, so
registry_add.py (which writes them) and registry_validate.py (which recomputes
and rejects mismatches) can never drift apart.

Python 3.8+ / stdlib only.
"""

import hashlib
import json
import os
import re

SCHEMA_VERSION = "quant-fidelity-registry/v1"
REGISTRY_ID = "malaiwah/quant-fidelity-registry"
MAINTAINER = "malaiwah"

COLLECTIONS = (
    ("models", "model", "model.schema.json"),
    ("artifacts", "artifact", "artifact.schema.json"),
    ("panels", "panel", "panel.schema.json"),
    ("references", "reference", "reference.schema.json"),
    ("pipelines", "pipeline", "pipeline.schema.json"),
    ("measurements", "measurement", "measurement.schema.json"),
)

ID_PREFIX_TO_COLLECTION = {tag: name for name, tag, _ in COLLECTIONS}
COLLECTION_TO_ID_PREFIX = {name: tag for name, tag, _ in COLLECTIONS}

ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*(?:--[a-z0-9][a-z0-9.-]*)*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------
# One rule, used for every hash and every line written to a .jsonl file:
#   json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
# Floats are emitted by repr(), which round-trips IEEE-754 double exactly, so a
# value never loses precision on the way into or out of the registry.


def canonical_json(obj):
    """The registry's single canonical JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# comparability.key
# ---------------------------------------------------------------------------
# The whole registry hangs off this function. Two measurement values may be
# compared IF AND ONLY IF their comparability.key values are equal. The key is
# a hash over the seven things that must be identical for two fidelity numbers
# to mean the same thing.

COMPARABILITY_KEY_FIELDS = (
    "panel_id",
    "reference_id",
    "metric_name",
    "direction",
    "accumulation_dtype",
    "stack_relation",
    "head_policy",
)


def comparability_key(key_inputs):
    """cmp-- + first 16 hex of sha256 over the '|'-joined key inputs.

    key_inputs: dict with exactly the COMPARABILITY_KEY_FIELDS keys.
    Serialization: each value is str()'d as-is (they are all strings), joined
    with a single '|', encoded UTF-8. No JSON, no padding, no normalization --
    the values come from closed enums and id fields, so there is nothing to
    normalize and nothing that can contain a '|'.
    """
    missing = [f for f in COMPARABILITY_KEY_FIELDS if f not in key_inputs]
    if missing:
        raise ValueError("comparability_key: missing inputs %s" % missing)
    joined = "|".join(str(key_inputs[f]) for f in COMPARABILITY_KEY_FIELDS)
    if any("|" in str(key_inputs[f]) for f in COMPARABILITY_KEY_FIELDS):
        raise ValueError("comparability_key: a key input contains the '|' separator")
    return "cmp--" + sha256_hex(joined)[:16]


def key_inputs_from_measurement(m):
    """Derive the key inputs from a measurement row's own authoritative fields.

    Deliberately does NOT read m['comparability']['key_inputs'] -- the validator
    compares this against that, so a hand-edited key_inputs block is caught.
    """
    return {
        "panel_id": m["panel_ref"],
        "reference_id": m["reference_ref"],
        "metric_name": m["metric"]["name"],
        "direction": m["metric"]["direction"],
        "accumulation_dtype": m["estimator"]["accumulation_dtype"],
        "stack_relation": m["estimator"]["stack_relation"],
        "head_policy": m["estimator"]["head_policy"],
    }


# ---------------------------------------------------------------------------
# scope_digest
# ---------------------------------------------------------------------------
# A one-line canonical summary of what was actually quantized, so a measurement
# row is readable in a table without a join, and so a scope edit nobody restated
# cannot slip through.
#
#   segment := <tensor_class>=<treatment>:<format>[@<bits_per_weight>]
#   join    := "|", segments sorted lexicographically
#   suffix  := "|head=<scope.head_policy>|kv=<scope.kv_cache_dtype>"
#
# bits_per_weight is omitted (with its '@') when null. Numbers are formatted by
# format_bpw() below so 6 and 6.0 produce the same digest.


def format_bpw(value):
    """Canonical bits-per-weight rendering: integral values lose the '.0'."""
    if value is None:
        return None
    f = float(value)
    if f == int(f):
        return str(int(f))
    return repr(f)


def scope_digest(scope):
    segments = []
    for a in scope["assignments"]:
        seg = "%s=%s:%s" % (a["tensor_class"], a["treatment"], a["format"])
        bpw = format_bpw(a.get("bits_per_weight"))
        if bpw is not None:
            seg += "@" + bpw
        segments.append(seg)
    segments.sort()
    return "|".join(segments) + "|head=%s|kv=%s" % (
        scope["head_policy"],
        scope["kv_cache_dtype"],
    )


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rows.append((lineno, json.loads(line), line))
            except ValueError as exc:
                raise ValueError("%s:%d: not valid JSON: %s" % (path, lineno, exc))
    return rows


def load_collection(data_dir, name):
    """Return a list of records (dicts) for a collection."""
    return [obj for _, obj, _ in read_jsonl(os.path.join(data_dir, name + ".jsonl"))]


def write_jsonl(path, records):
    """Write records canonically, sorted by id, one per line."""
    records = sorted(records, key=lambda r: r["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(canonical_json(r) + "\n")
    return len(records)


def load_registry(data_dir):
    """Load every collection into a dict of {collection_name: {id: record}}."""
    out = {}
    for name, _, _ in COLLECTIONS:
        out[name] = {}
        for rec in load_collection(data_dir, name):
            out[name][rec["id"]] = rec
    return out


# ---------------------------------------------------------------------------
# small helpers shared by validator and renderer
# ---------------------------------------------------------------------------


def collection_of_id(rid):
    """Which collection an id belongs to, from its first '--' segment."""
    if not isinstance(rid, str):
        return None
    head = rid.split("--", 1)[0]
    return ID_PREFIX_TO_COLLECTION.get(head)


def disclosure_codes(record):
    return [d.get("code") for d in record.get("disclosures", [])]


def has_disclosure(record, code, affects=None):
    for d in record.get("disclosures", []):
        if d.get("code") != code:
            continue
        if affects is None or bool(d.get("affects_comparability", False)) == affects:
            return True
    return False


def population_stddev(values):
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


def close(a, b, rel=1e-12, abs_=1e-15):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def repo_root(start=None):
    """The registry root (the directory containing schema/ and data/)."""
    here = os.path.dirname(os.path.abspath(start or __file__))
    return os.path.dirname(here)
