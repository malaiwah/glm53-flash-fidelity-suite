"""The ONE frozen protocol file, its two hashes, and the stamp every receipt carries.

brandonmusic's rule is: one frozen protocol file, whose hash is embedded in every
output.  We adopt it, and we fix the failure his own campaign hit.

THE FAILURE.  ``kld_eval.protocol.load_verified()`` hashes the RAW BYTES of
protocol.yaml.  Three different ``protocol_sha256`` values appear across his
published receipts (53e165dd..., 8e80e8e1..., 4d1d91ad...) because the file was
hand-edited twice after generation -- once to add the governing-document hash,
once to add a ``student_nvfp4`` identity block.  Neither edit changed a single
scoring rule, but every receipt written before them now carries a hash that no
longer resolves, and his EXL3 headline and the NVFP4 headline it is compared
against were produced under different protocol hashes.  That is exactly the
failure the "one frozen file" rule exists to prevent.

THE FIX.  We publish two hashes:

  protocol_file_sha256     sha256 of the raw bytes -- byte-level provenance,
                           his rule, unchanged.
  protocol_scoring_sha256  sha256 of a CANONICAL JSON serialization of only the
                           scoring-relevant subset {scoring, selection,
                           uncertainty, determinism, canary_r0, lane,
                           reporting}.  Invariant to comments, whitespace,
                           key order and identity-block edits.

Two receipts are comparable when their scoring hashes match.  A file hash that
moved while the scoring hash held is a provenance note, not an incomparability.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

PROTOCOL_SCHEMA = "malaiwah.glm53-joint-kld-protocol.v1"

# The subset that defines comparability. Order is fixed here, not by the file.
SCORING_SUBSET_KEYS = (
    "scoring",
    "selection",
    "uncertainty",
    "determinism",
    "canary_r0",
    "lane",
    "reporting",
)

# .../<repo>/bin/jointstd/protocol.py -> <repo>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROTOCOL_PATH = os.path.join(
    _REPO_ROOT, "registry", "protocol", "glm53-joint-kld-protocol.v1.json"
)


class ProtocolError(RuntimeError):
    """The protocol file is missing, malformed, or does not match a pin."""


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


class Protocol:
    """A loaded protocol file plus both of its hashes."""

    def __init__(self, path: str, raw: bytes, doc: Dict[str, Any]) -> None:
        self.path = path
        self.raw = raw
        self.doc = doc
        self.file_sha256 = hashlib.sha256(raw).hexdigest()
        missing = [k for k in SCORING_SUBSET_KEYS if k not in doc]
        if missing:
            raise ProtocolError(
                "protocol file %s is missing scoring-relevant blocks: %s"
                % (path, ", ".join(missing))
            )
        subset = {k: doc[k] for k in SCORING_SUBSET_KEYS}
        self.scoring_canonical = canonical_json(subset)
        self.scoring_sha256 = hashlib.sha256(self.scoring_canonical).hexdigest()
        self.schema = doc.get("schema", "")
        if self.schema != PROTOCOL_SCHEMA:
            raise ProtocolError(
                "protocol schema %r is not %r" % (self.schema, PROTOCOL_SCHEMA)
            )

    # -- accessors used by the rest of the package ------------------------
    @property
    def vocab_limit(self) -> int:
        return int(self.doc["scoring"]["tokenizer_vocab"])

    @property
    def stored_vocab(self) -> int:
        return int(self.doc["scoring"]["stored_vocab"])

    @property
    def mask_padded(self) -> bool:
        return self.doc["scoring"]["padded_column_policy"] == "mask_both_sides"

    @property
    def ngram_n(self) -> int:
        return int(self.doc["selection"]["calibration_overlap_scan"]["ngram_n"])

    @property
    def ngram_threshold(self) -> float:
        return float(
            self.doc["selection"]["calibration_overlap_scan"]["ngram_exclusion_threshold"]
        )

    @property
    def min_exceedances(self) -> int:
        return int(self.doc["uncertainty"]["percentile_min_exceedances"])

    @property
    def bootstrap_b(self) -> int:
        return 5000

    @property
    def bootstrap_seed(self) -> int:
        return 20260829

    @property
    def sigma_run_gate(self) -> float:
        return float(self.doc["uncertainty"]["sigma_run_gate"])

    @property
    def shift_ratio_min(self) -> float:
        return float(self.doc["canary_r0"]["shift_ratio_min"])

    @property
    def alignment_band(self) -> Tuple[float, float]:
        lo, hi = self.doc["canary_r0"]["teacher_top1_realized_agreement_band"]
        return float(lo), float(hi)

    # -- the stamp --------------------------------------------------------
    def stamp(self) -> Dict[str, Any]:
        """The block that goes into every receipt this repository emits."""
        return {
            "protocol_schema": self.schema,
            "protocol_file": os.path.relpath(self.path, _REPO_ROOT),
            "protocol_file_sha256": self.file_sha256,
            "protocol_scoring_sha256": self.scoring_sha256,
        }

    def stamp_into(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc.update(self.stamp())
        return doc


def load(path: Optional[str] = None) -> Protocol:
    path = path or os.environ.get("JOINTSTD_PROTOCOL") or DEFAULT_PROTOCOL_PATH
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise ProtocolError("cannot read protocol file %s: %s" % (path, exc))
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtocolError("protocol file %s is not valid JSON: %s" % (path, exc))
    return Protocol(path, raw, doc)


def require_stamp(doc: Dict[str, Any], proto: Optional[Protocol] = None) -> None:
    """Refuse a receipt that does not carry the stamp (or carries a stale one).

    This is the mechanical half of "the hash is embedded in every output": it
    is not enough to write the stamp, something has to refuse output without it.
    """
    for key in ("protocol_schema", "protocol_file_sha256", "protocol_scoring_sha256"):
        if not doc.get(key):
            raise ProtocolError("receipt is missing %s" % key)
    if doc["protocol_schema"] != PROTOCOL_SCHEMA:
        raise ProtocolError("receipt carries foreign protocol schema %r" % doc["protocol_schema"])
    if proto is not None and doc["protocol_scoring_sha256"] != proto.scoring_sha256:
        raise ProtocolError(
            "receipt scoring hash %s does not match the loaded protocol %s"
            % (doc["protocol_scoring_sha256"][:16], proto.scoring_sha256[:16])
        )
