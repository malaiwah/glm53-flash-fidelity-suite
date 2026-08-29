"""jointstd -- the joint fidelity standard: protocol stamping, R0 canary,
calibration-overlap scanning, and the inference layer (clustered SE, window
block bootstrap with BCa, sigma_run, McNemar, percentile guards).

WHY THIS EXISTS AS A SEPARATE PACKAGE.  brandonmusic proposed a community
standard and published a working harness (``kld_eval``).  Most of it is better
than what we had wired into our GLM-5.3 publications, so we adopt it.  Three
constraints shaped how:

  1. His CLI cannot be run by us.  ``kld_eval.protocol.load_verified()`` runs
     before every verb and hard-fails on absolute paths inside his machine
     (``/home/brandonmusic/models/...``).  Reusing the CLI means forking his
     protocol.yaml, which is authoring a Derivative, not calling his tool.
  2. His analysis LAYER is portable and excellent, and we DO call it -- see
     ``jointstd.oracle``.  When numpy+scipy+pandas are importable we delegate
     the bootstrap to his ``kld_eval.analysis.stats.block_bootstrap`` and
     reproduce his published endpoints exactly.  Our own implementation is the
     fallback, and the selftest pins the two against each other.
  3. Everything in this repository that ``make check`` touches must run on a
     stock interpreter with no pip install.  So the fallback here is stdlib
     only (json, math, random, statistics, hashlib).  numpy is used when
     present solely to reproduce his RNG stream bit-for-bit.

Nothing here copies his source.  The statistics are textbook (Efron & Tibshirani
BCa; Liang-Zeger cluster-robust SE; McNemar with continuity correction) and the
protocol design is cited, not vendored.  See docs/PROTOCOL-ALIGNMENT.md.
"""

from __future__ import annotations

__all__ = ["protocol", "canary", "ngram", "stats", "oracle", "chi2"]

VERSION = "1.0.0"
