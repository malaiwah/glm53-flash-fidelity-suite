"""R0: the alignment canary.

Two halves, and the second is the one nobody implements.

  R0-a  SELF-KLD.  Score the teacher against itself.  Every scored position must
        be EXACTLY 0.0 -- not 1e-15, not "close".  In fp64 log-softmax the
        identical-input case is bit-exact, so any non-zero value means the two
        sides are not actually the same tensor: a transposed slice, an off-by-one
        row, a dtype round-trip, a silently truncated vocabulary.

  R0-b  ONE-POSITION SHIFT.  Re-score with the student rows shifted by one and
        assert the mean EXPLODES to entropy scale.  This is the half that
        catches the failure R0-a cannot: a harness that is consistently
        misaligned by one position scores teacher-vs-teacher at exactly 0.0 and
        looks perfect, because both sides are shifted together.  R0-a proves the
        two tensors are identical; R0-b proves the alignment convention is the
        one the protocol declares.

brandonmusic's proposed standard names both.  His harness implements R0-a as a
session gate (``cmd_canary_loader``, with a teacher-top1-vs-realized-token band)
but R0-b only as a synthetic unit test on V=512 random logits
(``tests/test_kld.py::test_one_position_shift_is_entropy_scale``) -- it never
runs against the real teacher inside a session.  This module runs both against
whatever logits it is handed, which is the concrete piece we can hand back.

Needs numpy (and torch if you want the fp64 path on GPU); this module is the
one place in jointstd that is allowed a dependency, because there is no logits
tensor without one.  Every caller degrades to SKIP, never to PASS.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

MASK_BOTH_SIDES = "mask_both_sides"


class CanaryFailure(RuntimeError):
    """R0 did not pass.  The protocol says abort the session; never shift."""


def _log_softmax(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kld_rows(teacher, student, vocab_limit: Optional[int] = None):
    """KL(teacher||student) per row, fp64, optionally masking padded columns."""
    import numpy as np

    t = np.asarray(teacher, dtype=np.float64)
    s = np.asarray(student, dtype=np.float64)
    if t.shape != s.shape:
        raise ValueError("teacher %r and student %r shapes differ" % (t.shape, s.shape))
    if vocab_limit is not None:
        if t.shape[-1] < vocab_limit:
            raise ValueError("stored vocab %d narrower than tokenizer vocab %d"
                             % (t.shape[-1], vocab_limit))
        t = t[..., :vocab_limit]
        s = s[..., :vocab_limit]
    tl = _log_softmax(t)
    sl = _log_softmax(s)
    p = np.exp(tl)
    return (p * (tl - sl)).sum(axis=-1)


def entropy_rows(teacher, vocab_limit: Optional[int] = None):
    import numpy as np

    t = np.asarray(teacher, dtype=np.float64)
    if vocab_limit is not None:
        t = t[..., :vocab_limit]
    tl = _log_softmax(t)
    return -(np.exp(tl) * tl).sum(axis=-1)


def run_r0(
    teacher,
    vocab_limit: Optional[int] = None,
    stored_vocab: Optional[int] = None,
    shift_ratio_min: float = 3.0,
    tag: str = "start",
    realized_tokens: Optional[Sequence[int]] = None,
    alignment_band: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Run both halves of R0 on one real teacher window.

    Returns a verdict dict; raises ``CanaryFailure`` on any failed gate.
    """
    import numpy as np

    t = np.asarray(teacher, dtype=np.float32)
    rows, vocab = t.shape
    out: Dict[str, Any] = {
        "tag": tag,
        "rows": int(rows),
        "stored_vocab": int(vocab),
        "tokenizer_vocab": vocab_limit,
        "padded_columns": (int(vocab - vocab_limit) if vocab_limit else None),
    }

    # ---- R0-a: self-KLD in both scopes ---------------------------------
    scopes: Dict[str, Any] = {}
    for name, limit in (("unmasked", None), ("masked", vocab_limit)):
        if name == "masked" and vocab_limit is None:
            continue
        d = kld_rows(t, t, limit)
        mx = float(np.max(np.abs(d)))
        scopes[name] = {
            "max_abs_self_kld": mx,
            "all_positions_exactly_zero": bool(np.all(d == 0.0)),
            "nonzero_positions": int(np.count_nonzero(d)),
        }
    out["self_kld"] = scopes

    # ---- R0-b: one-position shift must explode -------------------------
    limit = vocab_limit if vocab_limit else None
    ent = entropy_rows(t, limit)
    mean_entropy = float(np.mean(ent))
    shifted = kld_rows(t[1:], t[:-1], limit)
    mean_shift = float(np.mean(shifted))
    ratio = mean_shift / mean_entropy if mean_entropy > 0 else float("inf")
    out["teacher_mean_entropy_nats"] = mean_entropy
    out["shift"] = {
        "offset": 1,
        "mean_kld_nats": mean_shift,
        "ratio_to_entropy": ratio,
        "ratio_min": shift_ratio_min,
        "exploded": ratio >= shift_ratio_min,
    }

    # ---- optional: teacher top-1 vs realized-token agreement band -------
    if realized_tokens is not None:
        top1 = np.argmax(t[:, :limit] if limit else t, axis=-1)
        realized = np.asarray(realized_tokens, dtype=np.int64)
        m = min(len(top1), len(realized))
        agree = float(np.mean(top1[:m] == realized[:m]))
        lo, hi = (alignment_band or (0.20, 0.995))
        out["teacher_top1_realized_agreement"] = agree
        out["alignment_band"] = [lo, hi]
        out["alignment_band_ok"] = bool(lo <= agree <= hi)

    failures = []
    for name, sc in scopes.items():
        if not sc["all_positions_exactly_zero"]:
            failures.append(
                "R0-a %s scope: %d of %d positions have non-zero self-KLD "
                "(max |d| = %.6g)" % (name, sc["nonzero_positions"], rows,
                                      sc["max_abs_self_kld"])
            )
    if not out["shift"]["exploded"]:
        failures.append(
            "R0-b: one-position shift gave mean %.6g nats, only %.2fx the "
            "teacher entropy %.6g -- the alignment convention is not the one "
            "the protocol declares" % (mean_shift, ratio, mean_entropy)
        )
    if realized_tokens is not None and not out["alignment_band_ok"]:
        failures.append(
            "R0-c: teacher top-1 vs realized-token agreement %.4f is outside "
            "the band %s" % (out["teacher_top1_realized_agreement"], out["alignment_band"])
        )
    out["failures"] = failures
    out["verdict"] = "PASS" if not failures else "FAIL"
    if failures:
        raise CanaryFailure("; ".join(failures))
    return out


def run_r0_on_pair(teacher, student, vocab_limit: Optional[int] = None,
                   tag: str = "pair") -> Dict[str, Any]:
    """The negative control: R0-a on a pair that is NOT the same tensor.

    A canary that has never been seen to fire is not a canary.  This entry point
    exists so the selftest can hand it a deliberately misaligned pair and prove
    the gate rejects it.
    """
    import numpy as np

    d = kld_rows(teacher, student, vocab_limit)
    nz = int(np.count_nonzero(d))
    ok = nz == 0
    res = {
        "tag": tag,
        "all_positions_exactly_zero": ok,
        "nonzero_positions": nz,
        "max_abs_self_kld": float(np.max(np.abs(d))),
        "mean_kld": float(np.mean(d)),
        "verdict": "PASS" if ok else "FAIL",
    }
    if not ok:
        raise CanaryFailure(
            "R0-a: %d of %d positions have non-zero KLD (max |d| = %.6g); the "
            "two sides are not the same tensor"
            % (nz, len(d), res["max_abs_self_kld"])
        )
    return res
