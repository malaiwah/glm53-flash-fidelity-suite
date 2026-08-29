"""HF lineage resolution: from any quant repo to its registry model + panel/teacher.

The chain is walked over Hugging Face metadata (tags of the form
"base_model:<relation>:<repo>" first -- verified live to be more complete than
cardData -- then cardData.base_model normalized to a list), and mapped onto the
registry leaf-first: an artifact whose huggingface.repository matches a hop
wins before a model record does.  Both published GLM roots (zai-org
GLM-5.3-Flash = FP8 and GLM-5.3-Flash-BF16) land on the SAME registry model,
because the registry records the FP8 repo on the model row and the BF16 repo on
the base artifacts.

Stock python3.9, stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .hfmeta import HFError, hf_unavailable_text, model_lineage_meta
from .registry_client import RegistrySnapshot

MAX_DEPTH = 5


class LineageError(RuntimeError):
    """A refusal with remedies; never a stack trace at the CLI."""

    def __init__(self, reason: str, advice: Optional[List[str]] = None) -> None:
        self.reason, self.advice = reason, list(advice or [])
        super().__init__(reason)


def resolve_base(repo: str, *, base_override: Optional[str] = None) -> Dict[str, Any]:
    """Walk base_model metadata to the root.  Returns
    {"chain": [repos...], "hops": [{repo, sha, bases}], "status": ...}."""
    if base_override:
        return {"chain": [repo, base_override], "hops": [],
                "status": "base-overridden"}
    chain: List[str] = []
    hops: List[Dict[str, Any]] = []
    seen = set()
    current = repo
    status = "root-reached"
    for _ in range(MAX_DEPTH):
        if current.lower() in seen:
            status = "cycle"
            break
        seen.add(current.lower())
        try:
            meta = model_lineage_meta(current)
        except HFError as exc:
            if not chain:
                # even the leaf is unreadable; the caller continues with the
                # registry by repo string and reports HF status separately
                chain.append(current)
            hops.append({"repo": current, "error": hf_unavailable_text(current, exc)})
            status = "hf-unreachable"
            break
        current = meta.repo_id            # adopt canonical case
        if not chain or chain[-1].lower() != current.lower():
            chain.append(current)
        hops.append({"repo": current, "sha": meta.sha,
                     "bases": list(meta.base_models)})
        bases = meta.base_models
        if not bases:
            break                          # this repo IS a root
        preferred = [b for b in bases if b[0] == "quantized"] or \
                    [b for b in bases if b[0] is None] or bases
        distinct = sorted({b[1].lower() for b in preferred})
        if len(distinct) > 1:
            raise LineageError(
                "AMBIGUOUS lineage: %s declares %d distinct base models (%s)."
                % (current, len(distinct),
                   ", ".join(sorted({b[1] for b in preferred}))),
                ["pass --base <repo> to pick the lineage to follow"])
        current = preferred[0][1]
    else:
        status = "max-depth"
    return {"chain": chain, "hops": hops, "status": status}


def map_to_registry_model(chain: List[str], reg: RegistrySnapshot) -> Dict[str, Any]:
    """Leaf-first mapping of a lineage chain onto the registry.

    An artifact match yields its model_ref; failing that, a model record whose
    huggingface.repository matches.  No hop matching is a refusal that lists
    both the chain walked and the models the registry knows."""
    artifacts = reg.collections.get("artifacts", {})
    models = reg.collections.get("models", {})
    for hop in chain:
        hop_l = hop.lower()
        for art in artifacts.values():
            if ((art.get("huggingface") or {}).get("repository") or "").lower() == hop_l:
                return {"model_ref": art.get("model_ref"), "via": "artifact",
                        "matched": art["id"], "hop": hop}
        for mod in models.values():
            if ((mod.get("huggingface") or {}).get("repository") or "").lower() == hop_l:
                return {"model_ref": mod["id"], "via": "model",
                        "matched": mod["id"], "hop": hop}
    raise LineageError(
        "model not in registry; walked: %s; the registry knows models: %s"
        % (" -> ".join(chain) or "(nothing)", ", ".join(sorted(models)) or "(none)"),
        ["pass --base <repo> to override the lineage walk",
         "or add the model/artifact to the registry first (registry/CONTRIBUTING.md)"])


def pick_panel_and_teacher(model_ref: str, lane_intent: Optional[str],
                           reg: RegistrySnapshot) -> Dict[str, Any]:
    """Choose the (panel, reference) pair prior measurements of this model used.

    Hard gates: the panel must be sealed; the reference must have logits
    available; subset panels are excluded unless the parent has no eligible
    rows; native_bf16 references outrank dequantized ones.  Pick order:
    (1) for a streaming intent, prefer a reference whose self_consistency
    names a floor measurement; (2) the pair with the most prior rows (the new
    row lands in the biggest comparable group); (3) deterministic panel-id
    tiebreak.  The choice is RETURNED WITH its alternatives so the caller can
    print the override flags."""
    panels = reg.collections.get("panels", {})
    references = reg.collections.get("references", {})
    prior = [m for m in reg.collections.get("measurements", {}).values()
             if m.get("model_ref") == model_ref and m.get("status") == "published"]
    if not prior:
        raise LineageError(
            "no published measurements for %s -- no precedent to pick a panel/"
            "teacher from" % model_ref,
            ["pass --panel and --teacher explicitly"])

    def subset_of(panel_ref: str) -> bool:
        p = panels.get(panel_ref) or {}
        return any(d.get("code") == "subset_of_panel"
                   for d in p.get("disclosures") or [])

    counts: Dict[Tuple[str, str], int] = {}
    for m in prior:
        pair = (m.get("panel_ref"), m.get("reference_ref"))
        if None in pair:
            continue
        counts[pair] = counts.get(pair, 0) + 1

    def eligible(pair: Tuple[str, str], allow_subset: bool) -> bool:
        panel_ref, ref_ref = pair
        panel = panels.get(panel_ref) or {}
        ref = references.get(ref_ref) or {}
        if not panel.get("sealed"):
            return False
        if not ref.get("logits_available"):
            return False
        if subset_of(panel_ref) and not allow_subset:
            return False
        return True

    pool = [p for p in counts if eligible(p, allow_subset=False)]
    if not pool:
        pool = [p for p in counts if eligible(p, allow_subset=True)]
    if not pool:
        raise LineageError(
            "no eligible (sealed panel, logits-available reference) pair among "
            "%d prior rows for %s" % (len(prior), model_ref),
            ["pass --panel and --teacher explicitly"])

    def score(pair: Tuple[str, str]) -> Tuple[int, int, int, str]:
        panel_ref, ref_ref = pair
        ref = references.get(ref_ref) or {}
        native = 1 if ref.get("reference_kind") == "native_bf16" else 0
        floor = 0
        if lane_intent == "streaming":
            if (ref.get("self_consistency") or {}).get("floor_measurement_ref"):
                floor = 1
        # sort descending on (floor, native, count), ascending panel id
        return (-floor, -native, -counts[pair], panel_ref)

    ranked = sorted(pool, key=score)
    chosen = ranked[0]
    alternatives = [{"panel_ref": p, "reference_ref": r, "rows": counts[(p, r)]}
                    for p, r in ranked[1:]]
    return {
        "panel_ref": chosen[0],
        "reference_ref": chosen[1],
        "rows": counts[chosen],
        "reference_kind": (references.get(chosen[1]) or {}).get("reference_kind"),
        "alternatives": alternatives,
    }
