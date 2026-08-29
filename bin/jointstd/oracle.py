"""Call brandonmusic's harness instead of reimplementing it -- where we can.

THE EXPLICIT EVALUATION the mission asked for, verb by verb, from reading his
published source and running it on this Mac:

  kld-eval inspect / select / run / card
      NOT CALLABLE BY US.  Every verb starts with
      ``kld_eval.protocol.load_verified()``, which refuses to proceed unless
      five files hash to values recorded in HIS protocol.yaml, at absolute
      paths inside his machine::

          $ python -m kld_eval.cli inspect
          kld_eval.protocol.ProtocolMismatch: student config.json: missing at
            /home/brandonmusic/models/GLM-5.3-Flash-EXL3-4bpw/config.json

      Reusing the CLI means forking protocol.yaml with our own identity block.
      Under his licence that is authoring a Derivative, not calling his tool.

  kld_eval.analysis.stats            CALLABLE, AND WE CALL IT.  Pure
  kld_eval.kld.core.score_window     numpy/scipy/pandas/torch, no GPU, no
  kld_eval.teacher.token_ngrams      engine, no HF token, no access to his
                                     checkpoint.  His 16 unit tests pass
                                     unmodified on a fresh macOS venv with
                                     libraries far newer than his pins.

So this module imports the analysis layer when it is importable and uses it as
the ORACLE our own implementation is pinned against.  Two reasons we still keep
our own:

  1. DEPENDENCY.  ``registry/ make check`` must run on a stock interpreter with
     no pip install.  scipy and pandas are not available there, and the whole
     point of the registry is that a contributor can validate a submission
     without a virtualenv tutorial.
  2. LICENCE.  His repository carries the SHAPLEYMCG LICENSE 1.0 -- source
     available, attribution-as-a-condition, with a named Excluded Party.  We are
     not the Excluded Party (his THIRD_PARTY_NOTICES.md credits malaiwah), but
     his section 1.2 defines a Derivative broadly enough to include
     "re-implementations made with reference to the Work", and section 5.1
     constrains onward distribution.  Vendoring his source into our public
     repository is a decision for the operator and for him, not a default.
     IMPORTING an installed package at runtime is use, not distribution, and is
     what this module does.

Nothing here copies his code.  If ``kld_eval`` is not installed, every function
returns ``None`` and the caller falls back to ``jointstd.stats``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

_STATUS: Optional[Dict[str, Any]] = None


def probe() -> Dict[str, Any]:
    """Is his analysis layer importable here, and at what versions?"""
    global _STATUS
    if _STATUS is not None:
        return _STATUS
    out: Dict[str, Any] = {"available": False, "modules": {}, "reason": None}
    try:
        import kld_eval  # noqa: F401
        from kld_eval.analysis import stats as _s  # noqa: F401
        out["available"] = True
        out["entry"] = "kld_eval.analysis.stats"
        for mod in ("numpy", "scipy", "pandas"):
            try:
                m = __import__(mod)
                out["modules"][mod] = getattr(m, "__version__", "?")
            except Exception as exc:
                out["modules"][mod] = "MISSING (%s)" % type(exc).__name__
        out["constants"] = {
            "MIN_EXCEEDANCES": getattr(_s, "MIN_EXCEEDANCES", None),
            "BOOTSTRAP_B": getattr(_s, "BOOTSTRAP_B", None),
            "BOOTSTRAP_SEED": getattr(_s, "BOOTSTRAP_SEED", None),
        }
    except Exception as exc:
        out["reason"] = "%s: %s" % (type(exc).__name__, exc)
    _STATUS = out
    return out


def _frame(window_means: Dict[str, float], positions_per_window: int = 2047):
    """Materialise the per-token frame his API expects from per-window means.

    Legal because every window is exactly ``positions_per_window`` positions:
    a constant column with that mean reproduces the window's contribution to
    every statistic his bootstrap computes from ``kld``.  Statistics that need
    the WITHIN-window distribution (percentiles, rms_delta_p) are not requested
    and would be wrong -- ``jointstd.stats.guard_pooled_percentiles`` refuses
    them separately.
    """
    import pandas as pd

    rows = []
    for wid in sorted(window_means):
        m = float(window_means[wid])
        for pos in range(positions_per_window):
            rows.append((wid, pos, m))
    df = pd.DataFrame(rows, columns=["window_id", "pos", "kld"])
    df["top1_agree"] = 0.0
    df["delta_p_realized"] = 0.0
    df["teacher_logp_realized"] = 0.0
    df["student_logp_realized"] = 0.0
    return df


def block_bootstrap_via_kld_eval(
    window_means: Dict[str, float],
    b: int = 5000,
    seed: int = 20260829,
    positions_per_window: int = 2047,
) -> Optional[Dict[str, Any]]:
    """Delegate the bootstrap to HIS implementation.  None when unavailable."""
    st = probe()
    if not st["available"]:
        return None
    try:
        from kld_eval.analysis.stats import block_bootstrap
    except Exception:
        return None
    df = _frame(window_means, positions_per_window)
    res = block_bootstrap(df, b=b, seed=seed)
    return {
        "backend": "kld_eval",
        "observed": float(res.observed["mean_kld"]),
        "ci95_percentile": [float(x) for x in res.ci_percentile["mean_kld"]],
        "ci95_bca": [float(x) for x in res.ci_bca["mean_kld"]],
        "b": res.b,
        "seed": res.seed,
        "n_windows": res.n_windows,
    }


def clustered_se_via_kld_eval(
    values: Sequence[float], clusters: Sequence[Any]
) -> Optional[Dict[str, Any]]:
    st = probe()
    if not st["available"]:
        return None
    try:
        import numpy as np
        from kld_eval.analysis.stats import clustered_se
    except Exception:
        return None
    se, deff, g = clustered_se(np.asarray(values, dtype=float), np.asarray(clusters))
    return {"backend": "kld_eval", "se": float(se), "deff": float(deff),
            "n_clusters": int(g)}


def token_ngrams_via_kld_eval(tokens: Sequence[int], n: int = 13):
    st = probe()
    if not st["available"]:
        return None
    try:
        import numpy as np
        from kld_eval.teacher import token_ngrams
    except Exception:
        return None
    return token_ngrams(np.asarray(tokens, dtype=np.int64), n)


def score_window_via_kld_eval(teacher, student, token_ids, vocab_limit: int):
    """His per-token KLD kernel, imported.  Used by the selftest as the oracle
    our fp64 canary kernel is pinned against."""
    st = probe()
    if not st["available"]:
        return None
    try:
        from kld_eval.kld.core import score_window
    except Exception:
        return None
    return score_window(teacher, student, token_ids, vocab_limit)
