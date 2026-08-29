"""HF model-card fidelity-provenance annotation: generator and validator.

Two layers (docs/CARD-ANNOTATION-SPEC.md):

  Layer 1  one conformant `model-index` entry, so HF leaderboards and every
           existing card reader see the measurement.  LANE lives in
           `dataset.split` because `huggingface_hub` merges results on
           `(task.type, dataset.type, dataset.config, dataset.split,
           dataset.revision)` and would silently discard a lane carried only in
           `dataset.args` -- the exact lane mixing BIAS-006 forbids.

  Layer 2  one top-level, namespaced, additive `x_fidelity:` block for what
           model-index structurally cannot express: the dataset pointer, head
           identity, determinism evidence and the registry ids.

Neither layer is trusted alone: XC-1..XC-5 cross-check them against each other
and against `registry/data/measurements.jsonl`, which is what stops them
drifting.

Stdlib + PyYAML.  The round-trip axis shells out to an interpreter that has
`huggingface_hub`; when none does, the axis is SKIPPED and the report says so
rather than silently passing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import common

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))

SPEC_URL = ("https://github.com/malaiwah/glm53-flash-fidelity-suite/blob/main/docs/"
            "FIDELITY-DATASET-SPEC.md")
SPEC_VERSION = "fidelity-provenance/v1"
REGISTRY_DATASET = "malaiwah/quant-fidelity-registry"
REGISTRY_SCHEMA_VERSION = "quant-fidelity-registry/v1"
VIEWER = ("https://huggingface.co/datasets/malaiwah/quant-fidelity-registry/viewer/"
          "measurements?q=%s")
HUB_VALIDATE = "https://huggingface.co/api/validate-yaml"

TASK_TYPE = "text-generation"
TASK_NAME = "Distribution fidelity (KL divergence vs BF16 reference)"

#: base_model_relation is a HARD enum with exactly these four values; none of
#: them means "fidelity reference" (card spec section 3).
BASE_MODEL_RELATIONS = ("adapter", "merge", "quantized", "finetune")

ROLES = ("root", "quant", "fidelity-dataset")


class CardError(Exception):
    pass


def _yaml():
    try:
        import yaml
    except ImportError:
        raise CardError("PyYAML is required: python3 -m pip install PyYAML")
    return yaml


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def load_registry(registry_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    root = registry_dir or os.path.join(REPO, "registry")
    data = os.path.join(root, "data")
    out = {}
    digests = {}
    for name in ("models", "artifacts", "panels", "references", "pipelines", "measurements"):
        path = os.path.join(data, "%s.jsonl" % name)
        rows = {}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                payload = handle.read()
            digests[name] = common.sha256_hex(payload)
            for line in payload.splitlines():
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[row["id"]] = row
        out[name] = rows
    # NEVER put the local checkout path here.  This dict is copied verbatim into
    # `x_fidelity.registry.snapshot` on a card that gets PUBLISHED, and an
    # absolute host path is precisely the defect this whole format exists to
    # prevent (the dead `packed_root: /home/jl_fs/glm53-k6/out-k6`, and Festr's
    # validator-enforced `raw_chunks_retained: false` rule we adopt in
    # FIDELITY-DATASET-SPEC 5.A).  The registry is identified by
    # `x_fidelity.registry.dataset` plus these content digests; the reader's own
    # filesystem layout is not part of that identity.
    out["_snapshot"] = {"data_sha256": digests}
    return out


def lane_of(registry: Dict[str, Any], measurement: Dict[str, Any]) -> str:
    """A pipeline with no `lane` object is the sealed-ep8 lane (invariant BIAS-006)."""
    pipeline = registry["pipelines"].get(measurement.get("pipeline_ref")) or {}
    lane = pipeline.get("lane") or {}
    return lane.get("name") or "sealed-ep8"


def _panel_args(panel: Dict[str, Any], measurement: Dict[str, Any]) -> Dict[str, Any]:
    structure = panel.get("structure") or {}
    identity = panel.get("identity") or {}
    tokenizer = panel.get("tokenizer") or {}
    return _drop_nulls({
        "panel_id": panel["id"],
        "panel_token_sha256": identity.get("panel_token_sha256"),
        "contexts": structure.get("contexts"),
        "scored_positions": structure.get("scored_positions_total"),
        "context_length": structure.get("context_length"),
        "tokenizer": "%s@%s" % (tokenizer.get("repository"), tokenizer.get("revision")),
        "vocab_size": tokenizer.get("vocab_size"),
        "reference_id": measurement.get("reference_ref"),
    })


def _panel_repo(panel: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Derive the panel's HF repo + revision from its sealed source URIs."""
    for source in panel.get("sources") or []:
        uri = source.get("uri") or ""
        if "/datasets/" in uri and "/resolve/" in uri:
            tail = uri.split("/datasets/", 1)[1]
            repo, _, rest = tail.partition("/resolve/")
            revision = rest.split("/", 1)[0]
            return repo, revision
    availability = (panel.get("availability") or {}).get("uri") or ""
    if "/datasets/" in availability:
        return availability.split("/datasets/", 1)[1], None
    return None, None


# ---------------------------------------------------------------------------
# Layer 1: model-index
# ---------------------------------------------------------------------------


def _drop_nulls(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """GEN-9: never emit a null inside `args`; omit the key.

    Measured, not assumed: `huggingface_hub`'s `EvalResult` DROPS null-valued
    keys inside `dataset.args` / `metrics[].args` on a round-trip, so a card
    that emits `population_stddev_of_run_means: null` is not the card the Hub
    re-serves.  A missing key and a null key mean the same thing here, and only
    the missing form survives.
    """
    return {key: value for key, value in mapping.items() if value is not None}


def _metric_args(registry, measurement, lane) -> Dict[str, Any]:
    determinism = measurement.get("determinism") or {}
    estimator = measurement.get("estimator") or {}
    out = {
        "units": measurement["metric"]["units"],
        "higher_is_better": measurement["metric"]["higher_is_better"],
        "direction": measurement["metric"]["direction"],
        "estimator": "full_vocabulary_fp64",
        "accumulation_dtype": estimator.get("accumulation_dtype"),
        "logits_dtype": estimator.get("logits_dtype"),
        "head_policy": estimator.get("head_policy"),
        "stack_relation": estimator.get("stack_relation"),
        "lane": lane,
        "run_count": determinism.get("run_count"),
        "population_stddev_of_run_means": determinism.get("population_stddev_of_run_means"),
        "determinism": ("bitwise_identical_across_runs"
                        if determinism.get("identical_across_runs") else "not_established"),
        "measurement_id": measurement["id"],
        "comparability_key": (measurement.get("comparability") or {}).get("key"),
    }
    return _drop_nulls(out)


def attributable_refusal(registry, measurement, lane) -> Optional[str]:
    """Why the floor-subtracted number may NOT be published, or None.

    Three conditions, each a reason to omit the number rather than print an
    unverifiable one:

      * the floor row does not resolve (a dangling reference);
      * the floor was measured on a different LANE (BIAS-006);
      * the floor was measured over a different SCOPE.  A 25-window floor is
        not the zero-point for a 17-window measurement, for the same reason a
        streaming floor is not the zero-point for a sealed-lane row.  The
        registry has an invariant for the lane case and none for the scope
        case; this is the card-level guard, and the finding is written up in
        the JOURNAL for the registry owner.
    """
    bias = ((measurement.get("comparability") or {}).get("bias") or {})
    floor_ref = bias.get("floor_measurement_ref")
    if not floor_ref:
        return None
    floor = registry["measurements"].get(floor_ref)
    if not floor:
        return ("floor %s does not resolve in this registry clone; no attributable number "
                "is published" % floor_ref)
    floor_lane = lane_of(registry, floor)
    if floor_lane != lane:
        return ("floor %s was measured on lane %r but this row is lane %r; BIAS-006 forbids "
                "subtracting a floor across lanes" % (floor_ref, floor_lane, lane))
    row_scope = (measurement.get("measurement_scope") or {}).get("scope_name")
    floor_scope = (floor.get("measurement_scope") or {}).get("scope_name")
    if row_scope != floor_scope:
        return ("floor %s was measured over scope %r but this row is scope %r; a floor over a "
                "different set of scored positions is not this row's zero-point"
                % (floor_ref, floor_scope, row_scope))
    return None


def _attributable(registry, measurement, lane) -> Optional[Dict[str, Any]]:
    """The floor-subtracted number, SAME-LANE AND SAME-SCOPE ONLY (XC-4 / BIAS-006)."""
    bias = ((measurement.get("comparability") or {}).get("bias") or {})
    floor_ref = bias.get("floor_measurement_ref")
    if not floor_ref or attributable_refusal(registry, measurement, lane):
        return None
    floor = registry["measurements"][floor_ref]
    floor_lane = lane_of(registry, floor)
    value = measurement["metric"]["value"] - floor["metric"]["value"]
    return {
        "type": "kl_divergence_quantization_attributable",
        "name": "KLD attributable to quantization (same-lane floor removed), nats",
        "value": value,
        "args": {
            "units": "nats",
            "higher_is_better": False,
            "derived": True,
            "derivation": "candidate_minus_same_lane_floor",
            "floor_value": floor["metric"]["value"],
            "floor_measurement_id": floor_ref,
            "floor_lane": floor_lane,
            "caveat": "KL is not additive; this is an estimate, valid only against a floor "
                      "measured on the same lane, same panel and same reference.",
        },
    }


def build_model_index(registry: Dict[str, Any], measurement_ids: Sequence[str],
                      model_name: str) -> List[Dict[str, Any]]:
    """GEN-2/3/4: exactly one entry, lane in `split`, no two results sharing the key."""
    results = []
    seen_keys = set()
    for measurement_id in measurement_ids:
        measurement = registry["measurements"].get(measurement_id)
        if not measurement:
            raise CardError("%s is not in the registry" % measurement_id)
        panel = registry["panels"].get(measurement["panel_ref"])
        if not panel:
            raise CardError("panel %s is not in the registry" % measurement["panel_ref"])
        lane = lane_of(registry, measurement)
        repo, revision = _panel_repo(panel)
        # GEN-1: quote every digest and revision.  An all-digit 40-char revision
        # parses as an integer and the Hub rejects it.
        # `config` carries the panel id AND the measurement scope.  Discovered
        # on real data: the registry has `.clean17` rows -- the 17 of 25 windows
        # that survive a 13-gram calibration-overlap scan -- which share
        # panel_ref, lane and revision with the full-panel rows and would
        # therefore collide on huggingface_hub's 5-tuple merge key, silently
        # discarding one row's args.  A different scored SCOPE is a different
        # dataset config, which is exactly what `config` means, and it keeps the
        # discriminator inside the merge key where GEN-3 requires it.
        scope_name = (measurement.get("measurement_scope") or {}).get("scope_name")
        config = panel["id"].rsplit(".", 1)[-1]
        if scope_name and not config.endswith(scope_name):
            config = "%s-%s" % (config, scope_name)
        panel_args = _panel_args(panel, measurement)
        scope = measurement.get("measurement_scope") or {}
        if scope_name:
            panel_args["scope_name"] = scope_name
            panel_args["scored_positions"] = scope.get("scored_positions")
            panel_args["contexts"] = scope.get("contexts")
            panel_args["covers_full_panel"] = scope.get("covers_full_panel")
            panel_args["scope_selection_sha256"] = scope.get("scope_selection_sha256")
            panel_args = _drop_nulls(panel_args)
        dataset_block = {
            "type": repo,
            "name": panel["name"] + (" -- %s subset" % scope_name if scope_name else ""),
            "config": config,
            "split": lane,
            "revision": str(revision) if revision else None,
            "args": panel_args,
        }
        key = (TASK_TYPE, dataset_block["type"], dataset_block["config"],
               dataset_block["split"], dataset_block["revision"])
        if key in seen_keys:
            raise CardError(
                "REFUSED: two results would share the huggingface_hub merge key %s. The "
                "second result's args would be silently discarded (GEN-4)." % (key,))
        seen_keys.add(key)

        metrics = [{
            "type": "kl_divergence",
            "name": "Mean tokenwise KLD (reference || candidate), nats",
            "value": measurement["metric"]["value"],
            "args": _metric_args(registry, measurement, lane),
        }]
        attributable = _attributable(registry, measurement, lane)
        if attributable:
            metrics.append(attributable)
        top1 = (measurement.get("auxiliary_metrics") or {}).get("top1_agreement")
        if top1 is not None:
            metrics.append({
                "type": "top1_agreement",
                "name": "Top-1 agreement with reference",
                "value": top1,
                "args": {"units": "fraction", "higher_is_better": True, "lane": lane},
            })
        for metric in metrics:
            metric["args"] = _drop_nulls(metric.get("args") or {})
        results.append({
            "task": {"type": TASK_TYPE, "name": TASK_NAME},
            "dataset": dataset_block,
            "metrics": metrics,
            "source": {"name": "quant-fidelity-registry",
                       "url": VIEWER % measurement_id},
        })
    # GEN-2: exactly ONE entry.  `model_index_to_eval_results` flattens every
    # entry into one list and keeps the LAST entry's name.
    return [{"name": model_name, "results": results}]


# ---------------------------------------------------------------------------
# Layer 2: x_fidelity
# ---------------------------------------------------------------------------


def build_x_fidelity(registry: Dict[str, Any], *, role: str,
                     measurement_ids: Sequence[str] = (),
                     artifact_id: Optional[str] = None,
                     reference_model: Optional[str] = None,
                     reference_revision: Optional[str] = None,
                     fidelity_dataset: Optional[Dict[str, Any]] = None,
                     head_content_sha256: Optional[str] = None,
                     head_file_sha256: Optional[str] = None,
                     final_norm_file_sha256: Optional[str] = None,
                     head_quantized: bool = False,
                     head_bits: Optional[int] = 16,
                     equality_receipt: Optional[str] = None,
                     scope_digest: Optional[str] = None,
                     captured_model: Optional[Dict[str, Any]] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if role not in ROLES:
        raise CardError("role must be one of %s" % (ROLES,))
    measurements = []
    for measurement_id in measurement_ids:
        measurement = registry["measurements"].get(measurement_id)
        if not measurement:
            raise CardError("%s is not in the registry" % measurement_id)
        lane = lane_of(registry, measurement)
        determinism = measurement.get("determinism") or {}
        bias = ((measurement.get("comparability") or {}).get("bias") or {})
        floor_ref = bias.get("floor_measurement_ref")
        attributable = None
        refusal = attributable_refusal(registry, measurement, lane) if floor_ref else None
        if floor_ref and not refusal:
            floor = registry["measurements"][floor_ref]
            attributable = measurement["metric"]["value"] - floor["metric"]["value"]
        measurements.append({
            "id": measurement_id,
            "lane": lane,
            "status": measurement["status"],
            "comparability_key": (measurement.get("comparability") or {}).get("key"),
            "panel_id": measurement["panel_ref"],
            "reference_id": measurement["reference_ref"],
            "pipeline_id": measurement["pipeline_ref"],
            "metric_name": measurement["metric"]["name"],
            "value": measurement["metric"]["value"],
            "units": measurement["metric"]["units"],
            "direction": measurement["metric"]["direction"],
            "run_count": determinism.get("run_count"),
            "determinism": {
                "evidence_kind": determinism.get("evidence_kind"),
                "distinct_evidence_hash_count": determinism.get(
                    "distinct_evidence_hash_count"),
                "identical_across_runs": determinism.get("identical_across_runs"),
                "evidence_hashes": [str(h) for h in determinism.get("evidence_hashes") or []],
            },
            "floor_measurement_id": floor_ref,
            "quantization_attributable": attributable,
            "quantization_attributable_withheld": refusal,
            "measured_by": (measurement.get("provenance") or {}).get("measured_by"),
            "disclosures": [d["code"] for d in measurement.get("disclosures") or []],
        })
        if scope_digest is None:
            scope_digest = measurement.get("scope_digest")
        if artifact_id is None:
            artifact_id = measurement.get("artifact_ref")

    # GEN-8: never invent a head digest.  A null content digest FORBIDS
    # cross-artifact hidden replay (HEAD-4 / XC-5).
    replay_permitted = bool(head_content_sha256)
    head = {
        "policy": "quantized" if head_quantized else "native",
        "quantized": bool(head_quantized),
        "bits": head_bits,
        "lm_head_tensor_content_sha256": head_content_sha256,
        "lm_head_file_sha256": head_file_sha256,
        "final_norm_tensor_content_sha256": None,
        "final_norm_file_sha256": final_norm_file_sha256,
        "equality_receipt": equality_receipt,
        "replay_permitted": replay_permitted,
    }
    if not replay_permitted:
        head["note"] = (
            "lm_head_tensor_content_sha256 is null because no head-identity receipt has been "
            "published for this artifact yet. A comparator MUST refuse to replay this "
            "artifact's hidden states through any other artifact's head until it is filled in "
            "(FIDELITY-DATASET-SPEC HEAD-4). The published receipts record only the FILE "
            "digest, which is a container digest and never an identity (O-6).")

    block = {
        "spec": SPEC_URL,
        "spec_version": SPEC_VERSION,
        "role": role,
        "reference_model": reference_model,
        "reference_revision": str(reference_revision) if reference_revision else None,
        "fidelity_dataset": fidelity_dataset,
        "registry": {
            "dataset": REGISTRY_DATASET,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "measurement_ids": list(measurement_ids),
            # Which registry state produced these numbers.  A registry clone is
            # a moving target; without this, a card and the rows it cites can
            # drift with nothing on either side to say so.
            "snapshot": registry.get("_snapshot"),
        },
        "scope_digest": scope_digest,
        "head": head,
        "measurements": measurements,
    }
    if captured_model:
        block["captured_model"] = captured_model
    if extra:
        block.update(extra)
    return block


def reference_identity(registry: Dict[str, Any], measurement_ids: Sequence[str]
                       ) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Derive (reference_model, reference_revision) from the registry.

    The measurement names its `reference_ref`, the reference row names the
    artifact it captured, and the artifact row carries the HF repository and
    revision.  Every hop already exists in the data, so requiring the operator
    to retype the answer on the command line -- and silently writing nulls when
    they do not -- is the generator failing to reproduce its own output.

    Returns the pair plus a list of notes explaining any hop that did not
    resolve, so a null is always accompanied by a reason.
    """
    notes: List[str] = []
    for measurement_id in measurement_ids:
        measurement = registry["measurements"].get(measurement_id) or {}
        reference_id = measurement.get("reference_ref")
        reference = (registry.get("references") or {}).get(reference_id) or {}
        if not reference:
            notes.append("%s names reference %s, which is not in this registry clone"
                         % (measurement_id, reference_id))
            continue
        artifact_id = reference.get("artifact_ref")
        artifact = (registry.get("artifacts") or {}).get(artifact_id) or {}
        hub = artifact.get("huggingface") or {}
        repository, revision = hub.get("repository"), hub.get("revision")
        if repository:
            return repository, (str(revision) if revision else None), notes
        notes.append("reference %s names artifact %s, which carries no huggingface.repository"
                     % (reference_id, artifact_id))
    return None, None, notes


def build_dataset_x_fidelity(root: str, *, repository: Optional[str] = None,
                             revision: Optional[str] = None,
                             extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The `role: fidelity-dataset` block, read from a real dataset.

    Every value here is in `fidelity-dataset.json` already.  The generator used
    to expose no way to reach them, so the one card a standalone capture
    publisher needs -- step 2 of the three-step architecture, the whole point of
    a capture being publishable on its own -- was the one card it could not
    emit.  Nothing is invented: a field the manifest does not carry stays null
    and the validator says which.
    """
    sys.path.insert(0, os.path.join(REPO, "bin"))
    from fidelity import dsformat as manifest_format               # noqa: WPS433

    manifest = manifest_format.load_manifest(root)
    dataset, capture = manifest["dataset"], manifest["capture"]
    panel, head, runtime = manifest["panel"], manifest["head"], manifest["runtime"]
    weights = manifest.get("weights") or {}
    seal = manifest.get("seal") or {}
    block = {
        "spec": SPEC_URL,
        "spec_version": SPEC_VERSION,
        "role": "fidelity-dataset",
        "captured_model": {
            "repository": weights.get("repository"),
            "revision": (str(weights.get("model_revision") or weights.get("revision"))
                         if (weights.get("model_revision") or weights.get("revision"))
                         else None),
            "role": dataset.get("role"),
        },
        "form": capture["form"],
        "lane": runtime["lane"],
        "panel": {
            "panel_id": panel.get("panel_id"),
            "suite_token_hash_sha256": panel["suite_token_hash_sha256"],
            "contexts": panel.get("contexts"),
            "scored_positions": capture.get("scored_rows_total"),
            "repository": panel.get("repository"),
            "revision": (str(panel.get("revision")) if panel.get("revision") else None),
        },
        "head": {
            "policy": "quantized" if head.get("quantized") else "native",
            "quantized": bool(head.get("quantized")),
            "bits": head.get("bits"),
            "lm_head_tensor_content_sha256": head.get("tensor_content_sha256"),
            "lm_head_file_sha256": head.get("file_sha256"),
            "final_norm_tensor_content_sha256": (head.get("final_norm") or {})
            .get("tensor_content_sha256"),
            "final_norm_file_sha256": (head.get("final_norm") or {}).get("file_sha256"),
            "equality_receipt": head.get("equality_receipt"),
            "replay_permitted": bool(head.get("tensor_content_sha256")),
        },
        "scope_digest": (manifest.get("scope") or {}).get("scope_digest"),
        "seal": {
            "manifest": "fidelity-dataset.json",
            "dataset_sha256": manifest[manifest_format.SEAL_FIELD],
            "checksums": "checksums.txt",
            "checksums_sha256": seal.get("checksums_sha256"),
        },
        "interop": {
            "compatible_with": ((manifest.get("interop") or {}).get("compatible_with") or []),
            "k3_compat_emitted": bool((manifest.get("interop") or {})
                                      .get("k3_compat_emitted")),
        },
        "registry": {
            "dataset": REGISTRY_DATASET,
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "artifact_id": weights.get("artifact_ref"),
            "measurement_ids": [],
            "snapshot": None,
        },
        "fidelity_dataset": {
            "repository": repository or dataset.get("repository"),
            "revision": (str(revision or dataset.get("revision"))
                         if (revision or dataset.get("revision")) else None),
            "dataset_sha256": manifest[manifest_format.SEAL_FIELD],
            "capture_content_digest": capture["capture_content_digest"],
            "form": capture["form"],
            "role": dataset.get("role"),
        },
        "measurements": [],
    }
    if extra:
        block.update(extra)
    return block


# ---------------------------------------------------------------------------
# Card merge
# ---------------------------------------------------------------------------


def split_card(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a README into (frontmatter dict, body).  GEN-5: the body is never rewritten."""
    yaml = _yaml()
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n")
    if parts[0].strip() != "---":
        return {}, text
    for index in range(1, len(parts)):
        if parts[index].strip() == "---":
            front = "\n".join(parts[1:index])
            body = "\n".join(parts[index + 1:])
            return (yaml.safe_load(front) or {}), body
    return {}, text


def render_card(front: Dict[str, Any], body: str) -> str:
    yaml = _yaml()
    dumped = yaml.dump(front, sort_keys=False, allow_unicode=True, width=100,
                       default_flow_style=False)
    return "---\n%s---\n%s" % (dumped, body)


def merge_card(text: str, *, model_index: Optional[List[Dict[str, Any]]],
               x_fidelity: Dict[str, Any], datasets: Sequence[str] = (),
               metrics: Sequence[str] = (), tags: Sequence[str] = (),
               base_model: Optional[str] = None,
               base_model_relation: Optional[str] = None) -> str:
    """GEN-5/6/7: merge into card.data, preserve unknown keys, never set `verified`."""
    front, body = split_card(text)
    front = dict(front)
    if base_model:
        front["base_model"] = base_model
    if base_model_relation:
        if base_model_relation not in BASE_MODEL_RELATIONS:
            raise CardError("base_model_relation must be one of %s; the Hub enum has exactly "
                            "four values and none of them means 'fidelity reference'"
                            % (BASE_MODEL_RELATIONS,))
        front["base_model_relation"] = base_model_relation
    if datasets:
        existing = list(front.get("datasets") or [])
        for item in datasets:
            if item not in existing:
                existing.append(item)
        front["datasets"] = existing
    if metrics:
        existing = list(front.get("metrics") or [])
        for item in metrics:
            if item not in existing:
                existing.append(item)
        front["metrics"] = existing
    if tags:
        existing = list(front.get("tags") or [])
        for item in tags:
            if item not in existing:
                existing.append(item)
        front["tags"] = existing
    if model_index is not None:
        front["model-index"] = model_index
    front["x_fidelity"] = x_fidelity
    # GEN-7: never set `verified` or `verifyToken`; those are HF-controlled.
    for entry in front.get("model-index") or []:
        for result in entry.get("results") or []:
            result.pop("verified", None)
            result.pop("verifyToken", None)
    return render_card(front, body)


# ---------------------------------------------------------------------------
# Validation: three axes
# ---------------------------------------------------------------------------


def _hub_axis(text: str, repo_type: str = "model") -> Dict[str, Any]:
    import urllib.error
    import urllib.request

    payload = json.dumps({"content": text, "repoType": repo_type}).encode("utf-8")
    request = urllib.request.Request(
        HUB_VALIDATE, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "malaiwah-fidelity-card/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {"axis": "hub", "ran": True, "ok": not body.get("errors"),
                "errors": body.get("errors") or [], "warnings": body.get("warnings") or []}
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"errors": ["HTTP %s" % exc.code]}
        return {"axis": "hub", "ran": True, "ok": False,
                "errors": body.get("errors") or ["HTTP %s" % exc.code],
                "warnings": body.get("warnings") or []}
    except Exception as exc:
        return {"axis": "hub", "ran": False, "ok": None,
                "skipped": "network unavailable: %s" % exc}


ROUNDTRIP_SNIPPET = r'''
import json, sys
import yaml
from huggingface_hub import ModelCard
text = open(sys.argv[1], encoding="utf-8").read()
card = ModelCard(text)
again = ModelCard(str(card))

def frontmatter(raw):
    # Split on lines that are EXACTLY `---`.  A naive raw.split("---") breaks on
    # any card whose YAML contains a `---` inside a comment or a string, which
    # our own worked examples do.
    lines = raw.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return yaml.safe_load("\n".join(lines[1:i])) or {}
    return {}

before = frontmatter(text)
after = card.data.to_dict()
def norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))
lost = sorted(set(before) - set(after))
added = sorted(set(after) - set(before))
changed = sorted(k for k in set(before) & set(after) if norm(before[k]) != norm(after[k]))
print(json.dumps({"lost": lost, "added": added, "changed": changed,
                  "eval_results": len(card.data.eval_results or []),
                  "model_name": card.data.model_name,
                  "x_fidelity_present": "x_fidelity" in after,
                  "reparsed_ok": bool(again)}))
'''


def _roundtrip_axis(text: str) -> Dict[str, Any]:
    """YAML -> ModelCardData -> YAML must be structurally identical.

    This catches the multi-entry collapse and the args-drop merge automatically,
    without hard-coding either.
    """
    candidates = [os.environ.get("FIDELITY_CARD_PYTHON"),
                  os.path.join(REPO, ".venv", "bin", "python"),
                  "/opt/homebrew/bin/python3.14",
                  sys.executable]
    for python in candidates:
        if not python or not os.path.exists(python):
            continue
        probe = subprocess.run([python, "-c", "import huggingface_hub, yaml"],
                               capture_output=True)
        if probe.returncode != 0:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(text)
            path = handle.name
        script = os.path.join(tempfile.gettempdir(), "fidelity_card_roundtrip.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(ROUNDTRIP_SNIPPET)
        result = subprocess.run([python, script, path], capture_output=True, text=True)
        os.unlink(path)
        if result.returncode != 0:
            return {"axis": "roundtrip", "ran": True, "ok": False,
                    "errors": [result.stderr.strip().splitlines()[-1]
                               if result.stderr.strip() else "parse failed"],
                    "interpreter": python}
        info = json.loads(result.stdout.strip().splitlines()[-1])
        ok = not (info["lost"] or info["added"] or info["changed"])
        return {"axis": "roundtrip", "ran": True, "ok": ok, "detail": info,
                "errors": ([] if ok else
                           ["lost=%s added=%s changed=%s"
                            % (info["lost"], info["added"], info["changed"])]),
                "interpreter": python}
    return {"axis": "roundtrip", "ran": False, "ok": None,
            "skipped": "no interpreter with huggingface_hub (tried %s); set "
                       "FIDELITY_CARD_PYTHON" % [c for c in candidates if c]}


#: An absolute POSIX path into a user/host filesystem.  Deliberately NOT a bare
#: "starts with /" test: `/api/...` fragments and rooted URL paths are fine, and
#: a *prose* mention of a dead path (documenting the defect) lives in the body,
#: never in front matter.
_HOST_PATH_RE = re.compile(r"^/(Users|home|root|private|var/folders|tmp|mnt|media|opt/homebrew)(/|$)")


def _walk_scalars(node: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Every (dotted-path, scalar) pair in a nested mapping/sequence."""
    out: List[Tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(_walk_scalars(value, "%s.%s" % (path, key) if path else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_walk_scalars(value, "%s[%d]" % (path, i)))
    else:
        out.append((path, node))
    return out


def _our_axis(text: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    """Role-conditional required fields plus XC-1..XC-5."""
    errors: List[str] = []
    warnings: List[str] = []
    front, _ = split_card(text)
    fidelity = front.get("x_fidelity")
    if not isinstance(fidelity, dict):
        return {"axis": "ours", "ran": True, "ok": False,
                "errors": ["no x_fidelity block"], "warnings": []}
    # GEN-10 / HOSTPATH-1.  A card is a PUBLISHED artifact.  An absolute host
    # path in it leaks the author's filesystem and, worse, encodes a location
    # nobody else can resolve -- the exact defect that made the K6
    # materialization receipt useless once its box died.  Festr's validator
    # enforces the same rule from the other side (`raw_chunks_retained: false`
    # requires the host-local capture-chunk keys to be stripped before
    # publication); FIDELITY-DATASET-SPEC 5.A adopts it verbatim, so the card
    # surface must honour it too.  Checked over the whole front matter, not just
    # x_fidelity, because any generator field can regress into one.
    for path, value in _walk_scalars(front):
        if isinstance(value, str) and _HOST_PATH_RE.match(value):
            errors.append(
                "HOSTPATH-1: %s carries an absolute host path %r; a published card "
                "must never encode the author's filesystem layout" % (path, value))

    role = fidelity.get("role")
    if role not in ROLES:
        errors.append("x_fidelity.role %r is not one of %s" % (role, ROLES))
    for field in ("spec", "spec_version"):
        if not fidelity.get(field):
            errors.append("x_fidelity.%s is required for every role" % field)
    if not fidelity.get("scope_digest"):
        errors.append("x_fidelity.scope_digest is required for every role")

    if role == "root" and not fidelity.get("fidelity_dataset"):
        errors.append("a root MUST publish its fidelity dataset "
                      "(x_fidelity.fidelity_dataset non-null)")
    if role == "quant":
        if not (fidelity.get("registry") or {}).get("measurement_ids"):
            errors.append("a quant MUST carry at least one registry measurement id")
        if not front.get("base_model") or front.get("base_model_relation") != "quantized":
            errors.append("a quant card MUST carry base_model + "
                          "base_model_relation: quantized")
    if role == "fidelity-dataset":
        for field in ("captured_model", "form", "panel", "head", "lane", "seal"):
            if fidelity.get(field) is None:
                errors.append("a fidelity-dataset card MUST carry x_fidelity.%s" % field)
        if front.get("model-index"):
            errors.append("dataset cards have no model-index")

    entries = front.get("model-index") or []
    if len(entries) > 1:
        errors.append("exactly ONE model-index entry is safe: "
                      "model_index_to_eval_results keeps the LAST entry's name (GEN-2)")
    results = entries[0].get("results", []) if entries else []

    # GEN-4 / lane-in-key
    seen = set()
    for result in results:
        dataset = result.get("dataset") or {}
        key = (result.get("task", {}).get("type"), dataset.get("type"),
               dataset.get("config"), dataset.get("split"), dataset.get("revision"))
        if key in seen:
            errors.append("two results share the merge key %s; the second's args are "
                          "silently discarded (GEN-4)" % (key,))
        seen.add(key)
        for value in (dataset.get("revision"),):
            if value is not None and not isinstance(value, str):
                errors.append("dataset.revision %r must be a quoted string; an all-digit "
                              "revision parses as an integer and the Hub rejects it" % value)
        lane_in_args = (dataset.get("args") or {}).get("lane")
        for metric in result.get("metrics") or []:
            metric_lane = (metric.get("args") or {}).get("lane")
            if metric_lane and metric_lane != dataset.get("split"):
                errors.append("metric args lane %r != dataset.split %r (XC-2 / GEN-3)"
                              % (metric_lane, dataset.get("split")))
            if metric_lane and not dataset.get("split"):
                errors.append("lane lives only in args; it must be in dataset.split or "
                              "huggingface_hub merges lanes (GEN-3)")
        if lane_in_args and lane_in_args != dataset.get("split"):
            errors.append("dataset.args.lane %r != dataset.split %r" % (lane_in_args,
                                                                        dataset.get("split")))
        if "verified" in result or "verifyToken" in result:
            errors.append("verified/verifyToken are HF-controlled; never set them (GEN-7)")
        # GEN-9: a null inside args is silently dropped by EvalResult, so the
        # card that ships is not the card the Hub re-serves.
        for label, block in [("dataset.args", dataset.get("args") or {})] + [
                ("metrics[%d].args" % i, m.get("args") or {})
                for i, m in enumerate(result.get("metrics") or [])]:
            nulls = sorted(k for k, v in block.items() if v is None)
            if nulls:
                errors.append("GEN-9: %s carries null value(s) %s; huggingface_hub drops "
                              "them on round-trip, so omit the key instead" % (label, nulls))

    by_id = {}
    for result in results:
        for metric in result.get("metrics") or []:
            measurement_id = (metric.get("args") or {}).get("measurement_id")
            if measurement_id:
                by_id[measurement_id] = (result, metric)

    for entry in fidelity.get("measurements") or []:
        measurement_id = entry.get("id")
        # XC-1 value agreement, XC-2 lane == split
        if measurement_id not in by_id:
            if role == "quant":
                errors.append("XC-3: %s has no model-index result" % measurement_id)
            continue
        result, metric = by_id[measurement_id]
        if metric.get("value") != entry.get("value"):
            errors.append("XC-1: %s value %r in model-index != %r in x_fidelity"
                          % (measurement_id, metric.get("value"), entry.get("value")))
        if (result.get("dataset") or {}).get("split") != entry.get("lane"):
            errors.append("XC-2: %s lane %r != dataset.split %r"
                          % (measurement_id, entry.get("lane"),
                             (result.get("dataset") or {}).get("split")))
        # XC-4 floor_lane == lane
        if entry.get("quantization_attributable") is not None:
            attributable = [m for m in result.get("metrics") or []
                            if m.get("type") == "kl_divergence_quantization_attributable"]
            if not attributable:
                errors.append("XC-4: %s declares quantization_attributable but the result "
                              "carries no attributable metric" % measurement_id)
            else:
                args = attributable[0].get("args") or {}
                if args.get("floor_lane") != entry.get("lane"):
                    errors.append("XC-4/BIAS-006: floor_lane %r != lane %r"
                                  % (args.get("floor_lane"), entry.get("lane")))
                expected = entry["value"] - (args.get("floor_value") or 0.0)
                if abs(attributable[0]["value"] - expected) > 1e-15:
                    errors.append("XC-4: attributable %r != value - floor_value (%r)"
                                  % (attributable[0]["value"], expected))
        # registry agreement
        row = registry["measurements"].get(measurement_id) if registry else None
        if row is None:
            warnings.append("%s does not resolve in the local registry clone" % measurement_id)
        else:
            if row["metric"]["value"] != entry.get("value"):
                errors.append("%s value %r != registry %r"
                              % (measurement_id, entry.get("value"), row["metric"]["value"]))
            if lane_of(registry, row) != entry.get("lane"):
                errors.append("%s lane %r != registry lane %r"
                              % (measurement_id, entry.get("lane"), lane_of(registry, row)))
            key = (row.get("comparability") or {}).get("key")
            if entry.get("comparability_key") != key:
                errors.append("%s comparability_key %r != registry %r"
                              % (measurement_id, entry.get("comparability_key"), key))
            declared = [str(h) for h in (row.get("determinism") or {}).get(
                "evidence_hashes") or []]
            carried = (entry.get("determinism") or {}).get("evidence_hashes") or []
            if declared != list(carried):
                errors.append("%s determinism evidence hashes differ from the registry"
                              % measurement_id)

    # XC-3 the other direction
    declared_ids = {m.get("id") for m in fidelity.get("measurements") or []}
    for measurement_id in by_id:
        if measurement_id not in declared_ids:
            errors.append("XC-3: model-index cites %s, x_fidelity does not" % measurement_id)

    # XC-5 replay_permitted requires a non-null content digest
    head = fidelity.get("head") or {}
    if head.get("replay_permitted") and not head.get("lm_head_tensor_content_sha256"):
        errors.append("XC-5: replay_permitted requires a non-null "
                      "lm_head_tensor_content_sha256 (HEAD-4)")
    if head.get("lm_head_file_sha256") and head.get("lm_head_tensor_content_sha256") \
            and head["lm_head_file_sha256"] == head["lm_head_tensor_content_sha256"]:
        errors.append("head file digest equals the content digest; one convention was "
                      "pasted twice (O-6)")
    return {"axis": "ours", "ran": True, "ok": not errors,
            "errors": errors, "warnings": warnings}


def validate_card(text: str, registry: Optional[Dict[str, Any]] = None,
                  offline: bool = False, repo_type: str = "model") -> Dict[str, Any]:
    axes = []
    if offline:
        axes.append({"axis": "hub", "ran": False, "ok": None,
                     "skipped": "--offline: the live Hub validate-yaml axis was NOT run"})
    else:
        axes.append(_hub_axis(text, repo_type))
    axes.append(_roundtrip_axis(text))
    axes.append(_our_axis(text, registry or {"measurements": {}, "pipelines": {}}))
    errors = [e for axis in axes for e in (axis.get("errors") or [])]
    warnings = [w for axis in axes for w in (axis.get("warnings") or [])]
    skipped = [axis["axis"] for axis in axes if axis.get("ran") is False]
    return {"axes": axes, "errors": errors, "warnings": warnings,
            "skipped_axes": skipped, "ok": not errors}


# ---------------------------------------------------------------------------
# eval-results v2 (off by default; card spec section 6)
# ---------------------------------------------------------------------------


def build_eval_results_v2(registry: Dict[str, Any], measurement_ids: Sequence[str],
                          model_repo: str) -> List[Dict[str, Any]]:
    """`.eval_results/fidelity.yaml` rows.

    Emitted only behind an off-by-default flag: the format's bare `value:`
    cannot express units or direction, so a KLD leaderboard would sort
    backwards, and `evaluation_framework` is a closed upstream enum with no
    entry that fits.  See CARD-ANNOTATION-SPEC section 6.
    """
    rows = []
    for measurement_id in measurement_ids:
        measurement = registry["measurements"][measurement_id]
        panel = registry["panels"][measurement["panel_ref"]]
        repo, revision = _panel_repo(panel)
        rows.append({
            "dataset": {"id": repo, "task_id": "distribution-fidelity",
                        "revision": str(revision) if revision else None},
            "value": measurement["metric"]["value"],
            "source": {"url": "https://huggingface.co/%s" % model_repo, "name": "Model Card"},
            "notes": "Mean tokenwise KL(reference || candidate) in nats on lane %s; LOWER IS "
                     "BETTER. This format cannot express units or direction, so read the "
                     "model card's x_fidelity block, not this number alone."
                     % lane_of(registry, measurement),
        })
    return rows
