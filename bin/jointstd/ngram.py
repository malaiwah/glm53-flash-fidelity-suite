"""Calibration-overlap scanner: document-id check plus token n-gram shingling.

brandonmusic's finding is the reason this exists: DOCUMENT-level separation of
the sealed panel was clean -- ``document_id_in_calibration`` is false for all 25
final windows -- and yet six of them share 37-39% of their 13-grams with
calibration-role windows.  A document-hash dedup alone does not catch it.

Algorithm, matched exactly to his ``kld_eval.teacher.token_ngrams`` so the two
implementations can be compared on the real panel:

  * cast token ids to int64, take every contiguous n-token slice,
  * digest with blake2b(digest_size=12) over the raw little-endian bytes,
  * the window's gram set is the DEDUPLICATED set of those digests,
  * fraction = |grams AND calibration_grams| / |grams|,
  * exclude when document_id appears in a calibration window, or fraction >
    threshold.

The deduplicated denominator matters and is easy to get wrong: an axis4 window
in this panel has only ~710 distinct 13-grams out of 2036 slices, because that
corpus repeats itself.  Using 2036 would report 13% instead of 38%.

Stdlib only.  numpy is used if present purely to read .npy files; a tiny
built-in .npy reader covers the stock interpreter.
"""

from __future__ import annotations

import ast
import hashlib
import struct
from typing import Any, Dict, Iterable, List, Sequence, Set

DIGEST_SIZE = 12


# ---------------------------------------------------------------- .npy read
def read_npy_1d(path: str) -> List[int]:
    """Minimal .npy reader for 1-D int arrays -- no numpy required."""
    with open(path, "rb") as fh:
        magic = fh.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("%s is not a .npy file" % path)
        major = fh.read(1)[0]
        fh.read(1)  # minor
        if major == 1:
            (hlen,) = struct.unpack("<H", fh.read(2))
        else:
            (hlen,) = struct.unpack("<I", fh.read(4))
        header = fh.read(hlen).decode("latin1")
        body = fh.read()
    meta = ast.literal_eval(header.strip())
    descr = str(meta["descr"])
    if meta.get("fortran_order"):
        raise ValueError("%s: fortran order unsupported" % path)
    if len(meta.get("shape", ())) != 1:
        raise ValueError("%s: expected a 1-D token array, got %r" % (path, meta.get("shape")))
    if descr not in ("<i4", "<i8", "|i1", "<i2", "<u4", "<u2"):
        raise ValueError("%s: unsupported dtype %s" % (path, descr))
    fmt = {"<i4": "<i", "<i8": "<q", "|i1": "<b", "<i2": "<h",
           "<u4": "<I", "<u2": "<H"}[descr]
    size = struct.calcsize(fmt)
    n = len(body) // size
    return [v[0] for v in struct.iter_unpack(fmt, body[: n * size])]


def load_tokens(path: str) -> List[int]:
    try:
        import numpy as _np  # noqa: F401
    except Exception:
        return read_npy_1d(path)
    import numpy as np

    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 1:
        raise ValueError("%s: expected a 1-D token array, got %r" % (path, arr.shape))
    return [int(v) for v in arr]


# ----------------------------------------------------------------- shingles
def token_ngrams(tokens: Sequence[int], n: int = 13) -> Set[bytes]:
    """Deduplicated blake2b-12 digests of every contiguous n-token slice."""
    if n <= 0:
        raise ValueError("n must be positive")
    ids = [int(t) for t in tokens]
    grams: Set[bytes] = set()
    if len(ids) < n:
        return grams
    packer = struct.Struct("<%dq" % n).pack
    for i in range(len(ids) - n + 1):
        grams.add(hashlib.blake2b(packer(*ids[i : i + n]), digest_size=DIGEST_SIZE).digest())
    return grams


def build_calibration_grams(
    token_arrays: Iterable[Sequence[int]], n: int = 13
) -> Set[bytes]:
    out: Set[bytes] = set()
    for toks in token_arrays:
        out |= token_ngrams(toks, n)
    return out


# --------------------------------------------------------------------- scan
def scan(
    final_windows: Sequence[Dict[str, Any]],
    calibration_grams: Set[bytes],
    calibration_document_ids: Set[str],
    n: int = 13,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Scan sealed windows for calibration bleed.

    ``final_windows`` items need ``window_id``, ``document_id``, ``domain`` and
    ``tokens`` (a sequence of ids).
    """
    per_window: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for w in final_windows:
        grams = token_ngrams(w["tokens"], n)
        hits = len(grams & calibration_grams)
        # STAT-13. `hits / max(1, len(grams))` turned an EMPTY gram set into 0.0, so a
        # window shorter than the n-gram width was reported as perfectly clean and could
        # never be excluded -- a clean verdict for a window the scanner could not scan at
        # all. Verified: a 12-token window that is a VERBATIM PREFIX of the calibration
        # corpus scored 0.0000 and was SELECTED. That is the silent-zero class, in the
        # scanner whose whole job is to certify the published clean scope.
        scannable = len(grams) > 0
        frac = (hits / len(grams)) if scannable else None
        doc_overlap = w.get("document_id") in calibration_document_ids
        row = {
            "window_id": w["window_id"],
            "domain": w.get("domain"),
            "document_id": w.get("document_id"),
            "document_id_in_calibration": bool(doc_overlap),
            "scannable": scannable,
            "distinct_ngrams": len(grams),
            "shared_ngram_count": hits,
            "shared_ngram_fraction": (round(frac, 6) if scannable else None),
        }
        if not scannable:
            row["unscannable_reason"] = (
                "window holds %d tokens, fewer than the %d-gram width: overlap is not "
                "measurable" % (len(w["tokens"]), n))
        for extra in ("token_ids_sha256", "prediction_positions"):
            if w.get(extra) is not None:
                row[extra] = w[extra]
        per_window.append(row)
        # Refusing is the right answer for an unscannable window, for the same reason
        # guard_pooled_percentiles refuses a percentile it cannot derive.
        if doc_overlap or (not scannable) or frac > threshold:
            excluded.append(
                {
                    "window_id": w["window_id"],
                    "domain": w.get("domain"),
                    "reason": (
                        "document overlaps calibration corpus"
                        if doc_overlap
                        else row["unscannable_reason"] if not scannable
                        else "%d-gram overlap %.1f%% > %.0f%%" % (n, frac * 100.0, threshold * 100.0)
                    ),
                    "shared_ngram_fraction": (round(frac, 6) if scannable else None),
                }
            )
    excluded_ids = {e["window_id"] for e in excluded}
    selected = [w["window_id"] for w in final_windows if w["window_id"] not in excluded_ids]
    return {
        "method": "document-id check + %d-token-gram overlap (blake2b-%d digests, "
                  "deduplicated denominator)" % (n, DIGEST_SIZE),
        "ngram_n": n,
        "threshold": threshold,
        "calibration_grams": len(calibration_grams),
        "selected_windows": selected,
        "excluded_windows": excluded,
        "per_window": per_window,
    }


def threshold_sensitivity(
    per_window: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10, 0.20),
) -> List[Dict[str, Any]]:
    """How many windows survive at each threshold.

    His 0.05 is a bare literal in cli.py with no published sensitivity analysis.
    The threshold is worth a joint decision, and a decision needs this table.
    """
    fracs = [(w["window_id"], float(w["shared_ngram_fraction"])) for w in per_window]
    rows = []
    for t in thresholds:
        kept = [wid for wid, f in fracs if f <= t]
        rows.append(
            {
                "threshold": t,
                "kept": len(kept),
                "dropped": len(fracs) - len(kept),
                "dropped_windows": [wid for wid, f in fracs if f > t],
            }
        )
    return rows
