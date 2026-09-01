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
ROOT_STAGES = (
    "setup", "fetch_target",
    "capture", "verify",
    "capture_repeat", "verify_repeat",
    "compare_root", "qualify_root",
)

# Every stage name `stage_measure.sh` answers to.  Kept here so a caller can
# refuse an unknown --stage locally instead of paying for a box to print
# "unknown stage" and exit 2.
KNOWN_STAGES = ("setup", "fetch_target", "fetch_panel", "materialize", "measure",
                "score", "seal", "capture", "verify", "capture_repeat",
                "verify_repeat", "compare_root", "qualify_root",
                "race_bootstrap", "race_capture", "publish_root")


def stage_sequence(role: str = "quant", *, race: bool = False,
                   surface: Optional[str] = None,
                   publish_root: bool = False) -> List[str]:
    """The ordered stage list for one job.

    role="root"  -- two fresh processes capture the same reference into
                    distinct roots.  Each is verified independently, then a
                    forced self-compare and outer qualification bind the proof.
                    Neither public manifest is relabeled as a multi-run capture.
    race=True    -- explicitly unsupported for a paid root in the first safe
                    SSH-driven path.  There is no recovery proof for a
                    fetch/capture race, so this refuses before composing stages.
    publish_root -- refused here. Publication is controller-local only after
                    qualified results are retrieved, the paid pod is absent,
                    and the extracted evidence verifies.
    surface      -- only consulted for role="quant"; see MATERIALIZING_SURFACES.
    """
    if role == "root":
        if race:
            raise ValueError(
                "race/preview root capture is unsupported by the first safe paid path")
        if publish_root:
            raise ValueError(
                "root publication is controller-local after qualified result "
                "retrieval and provider-confirmed teardown; no remote stage may publish")
        return list(ROOT_STAGES)
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
