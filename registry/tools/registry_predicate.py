"""The comparability PREDICATE: what an equal key does NOT certify, made machine-readable.

The seven-field ``comparability.key`` is a NECESSARY partition key: rows under
different keys are never comparable, full stop. It was never a SUFFICIENT
certificate, and this registry's own data is the proof:

  * the key carries no measurement LANE, and group ``cmp--202b717f3219c414``
    holds one artifact measured on two lanes, differing in the fourth decimal;
  * the key carries no candidate PIPELINE, and the same checkpoint/panel/teacher
    measured through two pipelines has differed by ~24% (0.030480 vs 0.024555);
  * the key carries no HARDWARE class, and an A100-vs-H200 term of 2.97e-4 nats
    is ~13x the gap this registry publishes between two 4-bit quantizers
    (docs/ARCHITECTURE-DETERMINISM.md);
  * the key carries no artifact SCOPE, and a routed-experts-only quant and a
    full-forward GGUF at "the same" bpw are different interventions
    (docs/GGUF-MEASUREMENT.md).

This module computes, per comparability group, the secondary-dimension check
that must ALSO pass before rows are ranked as like-for-like:

  comparable = "true"     every secondary dimension is recorded and homogeneous
                          across the group's live members;
  comparable = "false"    a RECORDED secondary dimension differs -- ranking
                          across it attributes a lane/pipeline/hardware/scope
                          effect to quantization quality;
  comparable = "unknown"  nothing recorded differs, but at least one dimension
                          is unrecorded for at least one member, so homogeneity
                          cannot be certified.

The key deliberately stays UNVERSIONED AND UNCHANGED: rehashing it would regroup
every published row and orphan every key third parties have already cited. The
predicate is additive metadata beside the key, not a new key.

Computed here and nowhere else, so tools/registry_render.py (which writes it
into index.json) and tools/registry_validate.py (which recomputes it and rejects
a hand-edited index, CMP-007) can never drift apart.

Stdlib only, offline, python 3.8+.
"""

PREDICATE_VERSION = "comparability-predicate/v1"

WITHDRAWN_STATUS = ("superseded", "retracted")

_UNRECORDED = "unrecorded"


def _live_members(C, member_ids):
    return [m for m in member_ids
            if C["measurements"][m].get("status") not in WITHDRAWN_STATUS]


def _lane_of(C, m):
    """The declared lane name, or the documented default, or 'unrecorded'.

    A pipeline with no ``lane`` object is the sealed-ep8 lane for a SAME-STACK
    row -- that is BIAS-006's own documented reading, and it is what makes the
    flagship group's mix legible: sealed rows declare nothing, streaming rows
    declare 'streaming', and the two are genuinely different lanes. A
    CROSS-STACK row with no lane declaration stays 'unrecorded': the sealed
    default was never defined for replay pipelines and this function does not
    invent it.
    """
    pl = C["pipelines"].get(m.get("pipeline_ref")) or {}
    name = (pl.get("lane") or {}).get("name")
    if name:
        return name
    if ((m.get("estimator") or {}).get("stack_relation")) == "same_stack":
        return "sealed-ep8"
    return _UNRECORDED


def _hardware_of(C, m):
    pl = C["pipelines"].get(m.get("pipeline_ref")) or {}
    hw = pl.get("hardware") or {}
    gpu = hw.get("gpu")
    if not gpu:
        return _UNRECORDED
    count = hw.get("gpu_count")
    return "%sx %s" % (count, gpu) if count else str(gpu)


def _scope_class_of(C, m):
    """What the artifact's quantization actually TOUCHES, as a coverage class.

    Derived from the artifact's scope assignments: the sorted set of tensor
    classes whose treatment is not 'native', plus the scope head policy and KV
    dtype. bits_per_weight and format are deliberately EXCLUDED -- comparing a
    6-bit against a 4-bit of the same coverage is the registry's purpose, while
    comparing a routed-experts-only quant against a full-forward one is a scope
    mix (docs/GGUF-MEASUREMENT.md).

    Returns 'unrecorded' when the artifact declares no scope or any treatment is
    'unknown', and 'unquantized' for an artifact that quantizes nothing (a base
    / control / floor artifact) -- the caller EXEMPTS those from the mix check,
    because a floor is printed as context, never ranked as a quant.
    """
    art = C["artifacts"].get(m.get("artifact_ref")) or {}
    scope = art.get("scope") or {}
    assignments = scope.get("assignments")
    if not assignments:
        return _UNRECORDED
    quantized = set()
    for a in assignments:
        treatment = a.get("treatment")
        if treatment == "unknown":
            return _UNRECORDED
        if treatment != "native":
            quantized.add(a.get("tensor_class") or "?")
    if not quantized:
        return "unquantized"
    return "+".join(sorted(quantized)) + "|head=%s|kv=%s" % (
        scope.get("head_policy"), scope.get("kv_cache_dtype"))


def _dimension(values, label, fail_note=""):
    """One secondary dimension's verdict from its per-member values.

    Two or more distinct RECORDED values is a fail: the group demonstrably
    mixes the dimension. Any unrecorded value (with no recorded divergence) is
    unknown: nothing proves a difference, and nothing certifies homogeneity
    either.
    """
    recorded = sorted({v for v in values if v != _UNRECORDED})
    unrecorded = sum(1 for v in values if v == _UNRECORDED)
    if len(recorded) > 1:
        seen = recorded + ([_UNRECORDED] if unrecorded else [])
        return {"status": "fail", "values": seen}, (
            "%s: members span {%s}%s" % (label, ", ".join(seen), fail_note))
    if unrecorded:
        return {"status": "unknown", "values": recorded + [_UNRECORDED]}, (
            "%s: unrecorded for %d member%s; homogeneity cannot be certified"
            % (label, unrecorded, "" if unrecorded == 1 else "s"))
    return {"status": "pass", "values": recorded}, None


def group_predicate(C, member_ids):
    """The predicate for one comparability group. Pure function of the data."""
    live = _live_members(C, member_ids)
    ms = [C["measurements"][mid] for mid in live]
    secondary = {}
    reasons = []

    dim, reason = _dimension(
        [_lane_of(C, m) for m in ms], "lane",
        " -- lanes are not interchangeable: where one artifact appears on two "
        "lanes it is one set of weights measured twice, not two quants, and "
        "the renderer tables non-sealed lanes apart (a same-stack pipeline "
        "with no lane declaration is the sealed-ep8 lane, per BIAS-006)")
    secondary["lane"] = dim
    if reason:
        reasons.append(reason)

    dim, reason = _dimension(
        [m.get("pipeline_ref") or _UNRECORDED for m in ms], "pipeline",
        " -- the candidate pipeline is not a key input, and a measured pipeline "
        "effect of ~24% exists on this registry's own data")
    secondary["pipeline"] = dim
    if reason:
        reasons.append(reason)

    scope_values = [_scope_class_of(C, m) for m in ms]
    # A control/floor artifact quantizes nothing and is exempt from the scope
    # mix: it is printed as the measurement floor, not ranked as a quant.
    ranked = [v for v in scope_values if v != "unquantized"]
    dim, reason = _dimension(
        ranked or ["unquantized"], "scope",
        " -- these are different interventions; equal nominal bpw does not make "
        "a routed-experts-only quant and a full-forward quant the same thing "
        "(docs/GGUF-MEASUREMENT.md)")
    secondary["scope"] = dim
    if reason:
        reasons.append(reason)

    dim, reason = _dimension(
        [_hardware_of(C, m) for m in ms], "hardware",
        " -- a measured A100-vs-H200 term (2.97e-4 nats) exceeds fine rank "
        "differences (docs/ARCHITECTURE-DETERMINISM.md)")
    secondary["hardware"] = dim
    if reason:
        reasons.append(reason)

    statuses = [d["status"] for d in secondary.values()]
    if "fail" in statuses:
        comparable = "false"
    elif "unknown" in statuses:
        comparable = "unknown"
    else:
        comparable = "true"
    return {
        "predicate_version": PREDICATE_VERSION,
        "comparable": comparable,
        "reasons": reasons,
        "secondary": secondary,
        "live_member_count": len(live),
    }


def registry_predicates(C, groups):
    """{key: predicate} over {key: [measurement ids]}."""
    return {key: group_predicate(C, members) for key, members in groups.items()}


def pair_predicate(C, a_id, b_id):
    """The predicate over exactly two rows -- what submit-time and --explain print.

    A synthetic row not yet in C["measurements"] may be passed by inserting it
    first; this function only reads.
    """
    return group_predicate(C, [a_id, b_id])


def pair_label(pred):
    """One-line verdict for a peer listing. Field evidence: bin/registry-submit
    once printed eleven same-key rows flatly as "comparable", mixed lanes
    included, directly contradicting the README's promise that lanes are tabled
    apart. The first thing a contributor sees must apply the full predicate."""
    if pred["comparable"] == "true":
        return "comparable (like-for-like: lane, pipeline, scope, hardware match)"
    dims_failed = sorted(d for d, v in pred["secondary"].items() if v["status"] == "fail")
    dims_unknown = sorted(d for d, v in pred["secondary"].items() if v["status"] == "unknown")
    if pred["comparable"] == "false":
        return ("same-key-but-%s-differ%s (NOT rankable without a measured bridge)"
                % ("/".join(dims_failed), "s" if len(dims_failed) == 1 else ""))
    return ("same key; %s unrecorded -- like-for-like not certified"
            % "/".join(dims_unknown))
