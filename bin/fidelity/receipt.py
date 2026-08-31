"""Build and seal a `quant-fidelity-registry/submission-receipt.v1`.

Both runners produce the SAME receipt schema.  That is the whole point of
having two recipes: a number measured on a rented H200 and a number measured
on someone's desk must be the same kind of object, so the registry can rank
them and a reader can see which lane produced which.

The derived fields (`scope_digest`, and the comparability key the validator
recomputes) are produced by `registry/tools/registry_lib.py` -- the registry's
OWN code, imported, not reimplemented.  Two implementations of a hash function
is two chances to disagree, and the disagreement would surface as a rejected
submission months later.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import canonical_json, sha256_hex, seal, utcnow
from .common import load_registry_lib as _load_registry_lib  # moved; kept for back-compat


LANE_DISCLOSURES: Dict[str, Dict[str, Any]] = {
    "streaming": {
        "code": "streaming_lane_offset",
        "severity": "caveat",
        "affects_comparability": True,
        "detail": (
            "Measured on the streaming lane. Routed-expert combine order differs "
            "from the sealed EP8 lane, moving logits rms 0.26-0.28; with "
            "--reduce-order fp32 the window-0 KLD delta against the sealed lane "
            "is +1.5076e-5 at 99.80% argmax agreement. Streaming runs are bitwise "
            "identical to each other, so determinism holds within the lane; they "
            "are NOT bitwise identical to sealed-lane runs, and bitwise "
            "cross-topology parity is not achievable at all."
        ),
    },
    "local-mps": {
        "code": "local_device_reduction_order",
        "severity": "caveat",
        "affects_comparability": True,
        "detail": (
            "Measured on Apple Silicon via MPS. Weight decode carries no device "
            "offset -- the EXL3/TR3 decode was verified BITWISE IDENTICAL between "
            "MPS and CPU at 4 and 6 bits over full-size matrices -- so any offset "
            "against the sealed EP8 lane comes from the forward pass reduction "
            "order alone. KLD accumulation ran on CPU in float64 because MPS "
            "cannot represent float64 at all."
        ),
    },
    "local-cuda-budget": {
        "code": "local_device_reduction_order",
        "severity": "caveat",
        "affects_comparability": True,
        "detail": (
            "Measured on a single consumer CUDA device under a VRAM budget, using "
            "the panel-batched layer-outer schedule with CPU-resident non-routed "
            "weights and streamed expert chunks. Weight decode is bit-exact; any "
            "offset against the sealed EP8 lane comes from forward-pass reduction "
            "order."
        ),
    },
}

BUDGET_DISCLOSURE = {
    "code": "memory_budget_schedule",
    "severity": "info",
    "affects_comparability": False,
    "detail": (
        "Run under an explicit memory budget. expert_chunk and window_batch are "
        "numerics-invariant: experts are visited in strictly ascending order and "
        "accumulated sequentially into an fp32 accumulator, so the reported value "
        "does not depend on either knob."
    ),
}


class NotSubmittable(RuntimeError):
    """A receipt that is structurally barred from becoming a registry row."""


def _scan_for_unsubmittable(node: Any, path: str = "$") -> Optional[str]:
    """Depth-first scan of any input block for preview/teacher markers.

    Returns the reason string, or None when the subtree is clean.  This is the
    bin-side refusal axis for preview receipts; the registry's own const gate
    (submission_schema must equal quant-fidelity-registry/submission-receipt.v1)
    is the independent second axis, enforced by code this repo may not edit.
    """
    if isinstance(node, dict):
        schema = node.get("schema")
        if isinstance(schema, str) and "-preview." in schema:
            return ("%s carries schema %r -- a PREVIEW receipt. Preview receipts "
                    "are structurally unsubmittable: they are labeled estimates "
                    "(sampled, or measured on an unpinned local lane), not sealed "
                    "measurements. Run the sealed lane (engines.json lane "
                    "'streaming') to produce a submittable row." % (path, schema))
        if node.get("not_submittable") is True:
            return ("%s declares not_submittable: true. The tool that wrote it "
                    "said so on purpose; nothing downstream may launder it into "
                    "a registry row." % path)
        if node.get("capture_role") == "bf16_teacher":
            return ("%s is a bf16_teacher CAPTURE receipt, not a measurement. A "
                    "teacher becomes a registry REFERENCE record (see "
                    "engines/SAME-LANE-TEACHER.md), never a measurement row: it has "
                    "no measured_mean_kld because it IS the thing students are "
                    "measured against." % path)
        for key, value in node.items():
            reason = _scan_for_unsubmittable(value, "%s.%s" % (path, key))
            if reason:
                return reason
    elif isinstance(node, list):
        for i, value in enumerate(node):
            reason = _scan_for_unsubmittable(value, "%s[%d]" % (path, i))
            if reason:
                return reason
    return None


def assert_submittable(blocks: Dict[str, Any]) -> None:
    """Refuse to build a submission from preview/teacher material.

    `blocks` is a {label: block} dict of everything about to enter a
    submission receipt.  Raises NotSubmittable with the full reason.
    """
    for label, block in blocks.items():
        reason = _scan_for_unsubmittable(block, label)
        if reason:
            raise NotSubmittable("REFUSED to build a submission receipt: " + reason)


def build_submission(
    *,
    suite_root: Path,
    lane: str,
    measurer: Dict[str, Any],
    artifact: Dict[str, Any],
    panel: Dict[str, Any],
    reference: Dict[str, Any],
    metric: Dict[str, Any],
    estimator: Dict[str, Any],
    determinism: Dict[str, Any],
    measurement_scope: Dict[str, Any],
    produced_by: Dict[str, Any],
    environment: Optional[Dict[str, Any]] = None,
    cost: Optional[Dict[str, Any]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    auxiliary_metrics: Optional[Dict[str, Any]] = None,
    extra_disclosures: Optional[List[Dict[str, Any]]] = None,
    measured_at: Optional[str] = None,
    comparability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a receipt and self-seal it.

    `artifact["scope"]` must already be populated; its digest is computed here
    with the registry's own function so the validator's recomputation agrees.
    """
    assert_submittable({
        "artifact": artifact, "panel": panel, "reference": reference,
        "metric": metric, "estimator": estimator, "determinism": determinism,
        "measurement_scope": measurement_scope, "environment": environment,
        "evidence": evidence, "auxiliary_metrics": auxiliary_metrics,
    })
    lib = _load_registry_lib(suite_root)

    artifact = dict(artifact)
    if "scope" in artifact:
        artifact["scope_digest"] = lib.scope_digest(artifact["scope"])

    disclosures: List[Dict[str, Any]] = []
    if lane in LANE_DISCLOSURES:
        disclosures.append(dict(LANE_DISCLOSURES[lane]))
    if (environment or {}).get("vram_budget_gb"):
        disclosures.append(dict(BUDGET_DISCLOSURE))
    for d in extra_disclosures or []:
        disclosures.append(dict(d))
    if not disclosures:
        disclosures = [{
            "code": "no_known_deviations",
            "severity": "info",
            "affects_comparability": False,
            "detail": "No deviations from the pipeline's declared protocol.",
        }]
    else:
        # DISC-002: no_known_deviations never coexists with anything else.
        disclosures = [d for d in disclosures if d["code"] != "no_known_deviations"]

    doc: Dict[str, Any] = {
        "submission_schema": "quant-fidelity-registry/submission-receipt.v1",
        "receipt_sha256": "",
        "produced_by": produced_by,
        "measured_at": measured_at or utcnow(),
        "lane": lane,
        "measurer": measurer,
        "artifact": artifact,
        "panel": panel,
        "reference": reference,
        "metric": metric,
        "auxiliary_metrics": auxiliary_metrics or {"top1_agreement": None},
        "estimator": estimator,
        "determinism": determinism,
        "measurement_scope": measurement_scope,
        "environment": environment or {},
        "cost": cost or {"usd": None, "basis": None},
        "evidence": evidence or [],
        "disclosures": disclosures,
    }
    # How the number may be READ, when the producing tool has a verdict on it.
    # Optional and additive: a runner that has nothing to say omits it entirely,
    # and `registry_add` falls back to deriving the bias from stack_relation
    # exactly as it did before this block existed.
    if comparability and any(v is not None for v in comparability.values()):
        doc["comparability"] = {key: value for key, value in comparability.items()
                                if value is not None}
    return seal(doc)


def produced_by_block(suite_root: Path, entrypoint: str,
                      dependencies: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Name the exact code that produced a number, or refuse to pretend.

    `revision` and `entrypoint_sha256` are REQUIRED by the schema and must be
    real hashes -- there is no "unknown" value for them, deliberately: a
    receipt that cannot say which code produced it is not reproducible, and
    accepting one would quietly hollow out the whole registry.

    This is why the cloud controller computes this block on the CALLER's
    machine, where the git checkout lives, and ships it in job.json. On the
    instance there is no checkout and the controller's own entrypoint was never
    uploaded, so computing it there is impossible -- and we say so at seal time
    rather than emitting a receipt that gets rejected days later in review.
    """
    from .common import run, sha256_file

    path = suite_root / entrypoint
    revision = None
    try:
        proc = run(["git", "-C", str(suite_root), "rev-parse", "HEAD"], check=False)
        revision = (proc.stdout or "").strip() or None
    except Exception:                                  # noqa: BLE001
        revision = None

    missing = []
    if not revision:
        missing.append("revision (no git checkout at %s)" % suite_root)
    if not path.is_file():
        missing.append("entrypoint_sha256 (%s is not present)" % entrypoint)
    if missing:
        raise RuntimeError(
            "cannot build produced_by: " + "; ".join(missing) + ".\n"
            "  The schema requires both, because a number whose producing code "
            "cannot be named is not reproducible.\n"
            "  Fix: have the controller compute this block where the checkout "
            "lives and pass it through job.json[\"produced_by\"]."
        )

    deps = {str(k): str(v) for k, v in (dependencies or {}).items()
            if v is not None}
    return {
        # The GitHub repo was RENAMED glm53-fidelity-suite -> quant-fidelity-suite
        # when it stopped being one model's campaign. This block is copied
        # verbatim onto every published row's `harness.repository.url`, and the
        # old name 404s, so every receipt sealed between the rename and this fix
        # cites a repository that cannot be fetched -- in the one field whose
        # entire job is to let a reader find the code that produced the number.
        # The local checkout directory is still the old name; that is cosmetic,
        # this was not.
        "tool": "quant-fidelity-suite/bin",
        "repository": "malaiwah/quant-fidelity-suite",
        "revision": revision,
        "entrypoint": entrypoint,
        "entrypoint_sha256": sha256_file(str(path)),
        "runtime_reader_sha256": None,
        "container_image": None,
        "container_digest": None,
        "dependencies": deps,
    }


def host_environment(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "gpu": None,
        "gpu_count": None,
        "tensor_parallel": None,
        "vram_budget_gb": None,
        "peak_vram_gb": None,
        "host": "%s %s" % (platform.system(), platform.machine()),
        "wall_clock_hours": None,
    }
    env.update(extra or {})
    return env


def validate_locally(suite_root: Path, receipt_path: Path) -> Any:
    """Run the registry's own `--submission` check, if the tools are present."""
    from .common import run

    validator = suite_root / "registry" / "tools" / "registry_validate.py"
    if not validator.is_file():
        return None
    return run([sys.executable, str(validator), "--submission", str(receipt_path)],
               check=False, timeout=300)
