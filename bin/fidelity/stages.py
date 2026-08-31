#!/usr/bin/env python3
"""Which stages run, in which order -- in ONE place, for both transports.

`stage_measure.sh` owns what a stage DOES.  Nothing owned which stages run:
the sequence was a literal inside `measure_cloud._bootstrap_and_run`, and the
`materialize` insertion was a second literal three lines below it.  That was
fine while SSH was the only transport.  It stopped being fine the moment a
container entrypoint had to drive the same stages, because a second copy of a
sequence is a second chance to drift -- and a drift here is not a crash, it is
a run that skips `materialize` and then measures a tree nothing decoded, or
one that skips `score` and discovers at `seal` that there is nothing to seal,
three GPU-hours in.

So the rule lives here and both callers ask for it.

Stdlib only, python3.9-clean: `bin/` and `registry/` must run on stock
python3.9 with no installs (AGENTS.md), and this module is imported by the
on-instance entrypoint before any venv exists.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

# The surfaces whose `--bf16` side must be written out before anything can be
# measured.  exl3hf because those tensors are QUANTIZED and must be decoded;
# tr3-published and dione because they share shards with the routed payloads
# and transformers derives its checkpoint key set from the shard FILES, so a
# symlink view reports tens of thousands of tensors as unloaded and the load
# gate refuses.  (There the materializer decodes nothing; it re-shards the
# natives verbatim.)
MATERIALIZING_SURFACES = ("exl3hf", "tr3-published", "dione")

QUANT_STAGES = ("setup", "fetch_target", "fetch_panel", "measure", "score", "seal")
ROOT_STAGES = ("setup", "fetch_target", "capture", "verify")
ROOT_RACE_STAGES = ("setup", "race_bootstrap", "race_capture", "verify")

# Every stage name `stage_measure.sh` answers to.  Kept here so a caller can
# refuse an unknown --stage locally instead of paying for a box to print
# "unknown stage" and exit 2.
KNOWN_STAGES = ("setup", "fetch_target", "fetch_panel", "materialize", "measure",
                "score", "seal", "capture", "verify", "race_bootstrap",
                "race_capture", "publish_root")


def stage_sequence(role: str = "quant", *, race: bool = False,
                   surface: Optional[str] = None,
                   publish_root: bool = False) -> List[str]:
    """The ordered stage list for one job.

    role="root"  -- there is no candidate and no divergence: the reference IS
                    the checkpoint, so nothing is materialized and nothing is
                    scored.  `verify` recomputes the dataset's digest chain
                    while the box still exists, which is the last moment a bad
                    capture is free to throw away.
    race=True    -- root only: the fetch stops being a stage and becomes a
                    thread inside the capture, so `fetch_target` is replaced by
                    `race_bootstrap` (the kilobytes that make the rest
                    plannable) plus `race_capture`.
    publish_root -- root only: append `publish_root`, which uploads the sealed
                    dataset to the Hub AFTER `verify` passes and records the
                    published revision (ROOT-1).
    surface      -- only consulted for role="quant"; see MATERIALIZING_SURFACES.
    """
    if role == "root":
        stages = list(ROOT_RACE_STAGES if race else ROOT_STAGES)
        if publish_root:
            # ROOT-1: a sealed, twice-validated root was destroyed at teardown
            # because nothing published it -- $6.59 of GPU time and the only
            # copy of the evidence. When the job names a destination repo the
            # publish is a STAGE, after `verify`, on the instance, where the
            # dataset and the (networked-phase) token both already are.
            stages.append("publish_root")
        return stages
    stages = list(QUANT_STAGES)
    if surface in MATERIALIZING_SURFACES:
        # After fetch_target, before fetch_panel: the tree it writes is what the
        # engine loads as --bf16, and it is the artifact's own bytes, so it
        # belongs beside the fetch rather than beside the measurement.
        stages.insert(2, "materialize")
    return stages


def unknown_stages(names: Sequence[str]) -> List[str]:
    """The members of `names` `stage_measure.sh` has no case for."""
    return [n for n in names if n not in KNOWN_STAGES]
