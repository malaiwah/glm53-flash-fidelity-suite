#!/usr/bin/env python3
"""Portable hidden-form capture: an HF checkpoint + a token panel -> a SEALED fidelity dataset.

Why this file exists
--------------------
`bin/fidelity-dataset capture` documented a three-step architecture whose step 1
could not be run by anybody outside our own campaign:

  * it delegates to ``k6/tools/hidden_replay.py``, which was never committed to
    the public repository, so a fresh clone cannot capture at all;
  * that wrapper hard-codes ``HIDDEN_WIDTH = 4096`` and
    ``EXPECTED_VOCAB = 154880``, i.e. GLM-5.3-Flash's exact geometry, so it
    refuses every other checkpoint including the public 0.1B architectural
    fixture;
  * it drives ``stream_score.py``, our sealed streaming engine, which needs the
    campaign's sealed-corpus machinery; and
  * having run, it wrote a *capture tree*, never the dataset the ``--out`` flag
    promised.

This module is the missing counterpart: one forward pass per panel window
through a plain ``transformers`` model, the lm_head input tapped with a forward
pre-hook, and the result assembled into a dataset that
``docs/FIDELITY-DATASET-SPEC.md`` accepts.  It has no GLM-5.3-Flash constants
and no dependency on ``stream_score.py``; it works for any causal LM whose
``get_output_embeddings()`` is a bias-free ``nn.Linear``-shaped head.

THE CUT (state it, never assume it)
-----------------------------------
The captured tensor is the model's final hidden state as handed to ``lm_head``
-- i.e. after the text model's final norm and immediately before the head
matmul -- taken as the head module's INPUT via ``register_forward_pre_hook``.
Replay applies the head ONLY.  This is the same cut as
``k6/tools/hidden_replay.py`` and as Festr's kimi-k3 hidden-replay convention,
which is what makes the two artifact families comparable.

Storage consequence: a hidden-form record costs ``hidden_size * 2`` bytes per
scored position, a logit-form record ``vocab_size * 4``.  The manifest records
both so the ratio is arithmetic, not a claim.

Determinism
-----------
This tool captures ONE cold run.  Cross-run determinism is not asserted here --
it is established by capturing twice in two separate processes and running
``fidelity-dataset compare --self-compare``, which is the spec's SC-1
reproduction confirmation.  The ``determinism`` block therefore declares
``run_count: 1`` and names the self-compare as the outstanding evidence, rather
than claiming an identity this process never observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "bin"))

from fidelity import dsformat as F  # noqa: E402
from fidelity import dsmanifest, dsvalidate  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layer_outer  # noqa: E402
import race_fetch  # noqa: E402
# "The capital of France is" -> " Paris", asked of every capture, as one extra
# window through the schedule that is already running. It is the only guard here
# that sees a shard which loaded as ZEROS: the names, the shapes and the tensor
# count are all correct in that case, and only the model's own output is wrong.
import generation_probe  # noqa: E402
# The capture loop is the other multi-hour silence in a stage log: a root
# capture is 25 windows of full-vocabulary forward and, under the layer-outer
# schedule, N layers x 25 windows inside a single `run_panel` call.  Same meter,
# same file-vs-TTY rule -- see k6/tools/progress.py.
import progress as progress_meter  # noqa: E402

TOOL_VERSION = "hf_capture/1"

# Recorded in `runtime.capture_tool.mechanism`, which is NOT an input to
# `stack_fingerprint_sha256` -- deliberately.  The fingerprint is what
# `dscompare` reads to decide `stack_relation`, and a cross-stack verdict
# stamps `usable_as_floor: false` and attaches a 1e-2-class bias block.  The
# layer-outer schedule is proven bit-identical to the window-outer one on two
# architectures (see docs/GLM53-LAYER-OUTER.md), so charging a capture a
# comparability penalty for it would be asserting a difference the digests say
# is not there.  It is still written down, in the sealed receipt, where a
# reader can see which loop produced their tensors.
SCHEDULE_MECHANISM = {
    layer_outer.SCHEDULE_WINDOW_OUTER:
        "transformers forward pass; forward pre-hook on model.get_output_embeddings()",
    layer_outer.SCHEDULE_LAYER_OUTER:
        "transformers forward pass, layer-outer/window-inner schedule: for each decoder "
        "layer the model's own forward is run once per window with the layers below "
        "replaying their memoised output and the layers above suspending, so each "
        "layer's weights are materialised once for the whole panel. Windows are pushed "
        "through sequentially, never batched. Forward pre-hook on "
        "model.get_output_embeddings(), fired by the model's own epilogue after the "
        "last layer",
}

CUT_POINT = "after_final_rmsnorm_before_lm_head"
CUT_STATEMENT = (
    "the final hidden state handed to lm_head -- after the text model's final "
    "norm and immediately before the head matmul -- captured as the head "
    "module's input via torch.nn.Module.register_forward_pre_hook; replay "
    "applies the head ONLY (no final norm at replay time: the capture already "
    "sits after it). Same cut as k6/tools/hidden_replay.py and as Festr's "
    "kimi-k3 hidden-replay qualification."
)

# The checkpoint identity algorithm.  The spec REQUIRES
# weights.checkpoint_identity_sha256 but ships no portable way to compute one
# for a generic HF checkpoint, so this file defines one and names it in the
# dataset instead of emitting an unexplained hex string.
CHECKPOINT_IDENTITY_ALGORITHM = (
    "sha256 over the canonical JSON {\"algorithm\": <this string>, \"files\": "
    "[{\"name\", \"size\", \"sha256\"}, ...]} of every *.safetensors shard plus "
    "config.json in the checkpoint directory, sorted by name"
)


def fail(message: str, code: int = 1) -> "SystemExit":
    print("hf_capture: ERROR: %s" % message, file=sys.stderr, flush=True)
    return SystemExit(code)


def log(**fields: Any) -> None:
    print(json.dumps(fields, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# checkpoint identity
# ---------------------------------------------------------------------------


def checkpoint_identity(model_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    names = sorted(name for name in os.listdir(model_dir)
                   if name.endswith(".safetensors") or name == "config.json")
    files = []
    for name in names:
        full = os.path.join(model_dir, name)
        files.append({"name": name, "size": os.path.getsize(full),
                      "sha256": F.sha256_file(full)})
    doc = {"algorithm": CHECKPOINT_IDENTITY_ALGORITHM, "files": files}
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), files


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------


class Panel(object):
    """A sealed token panel: window metadata plus the token/mask .npy arrays.

    The upstream panel (brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits,
    calibration/panel-v1) digests its arrays by FILE sha256, while the dataset
    spec digests token ids by compact-JSON sha256.  Both are recorded: the
    per-record ``attention_mask_sha256`` we emit is the file digest of the mask
    bytes we write, which are byte-identical to the upstream .npy, so the
    upstream anchor survives into the dataset and can be re-checked.
    """

    def __init__(self, root: str, panel_id: str, source: str, receipt_sha256: Optional[str],
                 windows: Sequence[Dict[str, Any]], synthetic: bool = False,
                 tokenizer: Optional[Dict[str, Any]] = None):
        self.root = root
        self.panel_id = panel_id
        self.source = source
        self.receipt_sha256 = receipt_sha256
        self.windows = list(windows)
        self.synthetic = synthetic
        self.tokenizer = tokenizer or {}


def load_panel(panel_dir: str, role: str, limit: Optional[int],
               tokenizer_id: Optional[str], vocab_size: int) -> Panel:
    """Read a quant-pipeline.glm53-token-panel.v1 tree from disk."""
    import numpy as np

    panel_json = os.path.join(panel_dir, "panel.json")
    if not os.path.isfile(panel_json):
        raise fail("no panel.json under %s" % panel_dir)
    doc = json.loads(open(panel_json, "r", encoding="utf-8").read())
    schema = doc.get("schema")
    if schema != "quant-pipeline.glm53-token-panel.v1":
        raise fail("unexpected panel schema %r" % schema)
    rows = [w for w in doc.get("windows", []) if role in (None, "", w.get("role"))]
    rows.sort(key=lambda w: w["window_id"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise fail("no windows with role=%r in %s" % (role, panel_json))

    arrays = os.path.join(panel_dir, "arrays")
    windows = []
    for row in rows:
        token_path = os.path.join(arrays, "%s.tokens.npy" % row["window_id"])
        if not os.path.isfile(token_path):
            raise fail("panel window %s: missing %s" % (row["window_id"], token_path))
        got = F.sha256_file(token_path)
        if got != row["token_ids_sha256"]:
            raise fail("panel window %s: token file digest %s != sealed %s"
                       % (row["window_id"], got[:12], row["token_ids_sha256"][:12]))
        ids = np.load(token_path, allow_pickle=False)
        mask_path = _find_mask(arrays, row, ids.shape[0])
        got_mask = F.sha256_file(mask_path)
        if got_mask != row["attention_mask_sha256"]:
            raise fail("panel window %s: mask digest %s != sealed %s"
                       % (row["window_id"], got_mask[:12], row["attention_mask_sha256"][:12]))
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape[0] != ids.shape[0]:
            raise fail("panel window %s: mask len %d != tokens len %d"
                       % (row["window_id"], mask.shape[0], ids.shape[0]))
        if int(ids.max()) >= vocab_size:
            raise fail("panel window %s: token id %d >= vocab_size %d -- this panel was "
                       "built for a different tokenizer"
                       % (row["window_id"], int(ids.max()), vocab_size))
        windows.append({
            "window_id": row["window_id"],
            "role": row.get("role"),
            "domain": row.get("domain"),
            "document_id": row.get("document_id"),
            "prediction_positions": int(row["prediction_positions"]),
            "token_ids": [int(v) for v in ids],
            "token_file_sha256": got,
            "mask_bytes": open(mask_path, "rb").read(),
            "mask_sha256": got_mask,
            "mask": mask,
        })

    receipt = os.path.join(panel_dir, "panel.receipt.json")
    return Panel(
        root=panel_dir, panel_id="panel--%s" % os.path.basename(panel_dir.rstrip("/")),
        source=panel_json, receipt_sha256=F.sha256_file(receipt) if os.path.isfile(receipt) else None,
        windows=windows,
        tokenizer={"id": tokenizer_id, "repository": tokenizer_id, "revision": None,
                   "vocab_size": vocab_size, "add_special_tokens": False,
                   "chat_template_applied": False})


def _find_mask(arrays: str, row: Dict[str, Any], length: int) -> str:
    """The upstream panel shares one causal mask across windows of equal length."""
    named = os.path.join(arrays, "%s.mask.npy" % row["window_id"])
    if os.path.isfile(named):
        return named
    shared = os.path.join(arrays, "causal-mask-%d.npy" % length)
    if os.path.isfile(shared):
        return shared
    for name in sorted(os.listdir(arrays)):
        if name.startswith("causal-mask-") and name.endswith(".npy"):
            if F.sha256_file(os.path.join(arrays, name)) == row["attention_mask_sha256"]:
                return os.path.join(arrays, name)
    raise fail("panel window %s: no attention mask array found in %s"
               % (row["window_id"], arrays))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


REPORT_OBSERVED = "_load_report_observed"
REPORT_AUGMENTED = "_load_report_has_conversion_errors"


class _FullLoadingReport:
    """Make `from_pretrained`'s load report tell the whole truth.

    `output_loading_info=True` does not hand back `transformers`'
    `LoadStateDictInfo`; it hands back `LoadStateDictInfo.to_dict()`, and that
    method says, in its own source:

        # Does not include the `conversion_errors` to be coherent with legacy
        # reporting in the tests

    `conversion_errors` is where a failure of the MoE `WeightConverter` lands --
    the converter that, for `zai-org/GLM-5.3-BF16`, is responsible for 57,600 of
    59,585 checkpoint tensors (96.7%).  A guard that reads only the dict is
    blind to the failure of 96.7% of the checkpoint by construction.

    So for the duration of the load we wrap `to_dict` to also emit
    `conversion_errors` (and `skipped_pp_keys`), plus a flag saying the wrap
    took effect.  If this build of `transformers` has no such class or method,
    the flag stays false and `load_report` refuses to call the load verified
    rather than reporting a clean bill of health it never checked.
    """

    def __init__(self) -> None:
        self.target = None
        self.original = None

    def __enter__(self) -> "_FullLoadingReport":
        try:
            from transformers.utils.loading_report import LoadStateDictInfo
        except Exception:  # pragma: no cover - older/newer transformers
            return self
        original = getattr(LoadStateDictInfo, "to_dict", None)
        if original is None:  # pragma: no cover
            return self

        def to_dict(self_inner):
            doc = dict(original(self_inner))
            doc["conversion_errors"] = dict(getattr(self_inner, "conversion_errors", {}) or {})
            doc["skipped_pp_keys"] = set(getattr(self_inner, "skipped_pp_keys", set()) or set())
            doc[REPORT_AUGMENTED] = True
            return doc

        self.target, self.original = LoadStateDictInfo, original
        LoadStateDictInfo.to_dict = to_dict
        return self

    def __exit__(self, *exc) -> bool:
        if self.target is not None:
            self.target.to_dict = self.original
        return False


def _from_pretrained(cls, model_dir: str, torch_dtype, **extra):
    """`from_pretrained` plus the load report, across the dtype-kwarg rename.

    The report is what tells us whether the checkpoint actually populated the
    model.  It is requested here rather than reconstructed later because only
    `from_pretrained` knows the checkpoint-name -> parameter-name mapping (a
    MoE checkpoint may ship 256 per-expert matrices that the model holds as one
    fused tensor, so comparing key sets by hand is wrong).

    Every return marks whether a report was actually OBSERVED.  It used to
    return a bare ``{}`` on the no-report path, which downstream read as "no
    missing keys" -- i.e. an unexamined load and a clean load were the same
    value.  They are not the same fact and must not be the same value.
    """
    with _FullLoadingReport():
        for kwargs in ({"dtype": torch_dtype}, {"torch_dtype": torch_dtype}):
            try:
                out = cls.from_pretrained(model_dir, output_loading_info=True,
                                          **kwargs, **extra)
            except TypeError as exc:
                if "dtype" not in str(exc):
                    # A TypeError from INSIDE the load is not a signature
                    # mismatch. Retrying under a different kwarg name would
                    # hide it, and the final fallback would then return a
                    # model with no report at all.
                    raise
                continue
            if isinstance(out, tuple):
                info = dict(out[1] or {})
                info[REPORT_OBSERVED] = out[1] is not None
                return out[0], info
            return out, {REPORT_OBSERVED: False}
        # Very old / unusual classes: take the model without a report and say so.
        return (cls.from_pretrained(model_dir, torch_dtype=torch_dtype, **extra),
                {REPORT_OBSERVED: False})


# `transformers` 5.16.1 `quantizers/quantizer_finegrained_fp8.py:195`:
#
#     layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
#     ...
#     updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
#
# `_impl_tp_layer_overrides` has exactly one key, `deepgemm_megamoe`, and
# `config._experts_implementation` is still None when `get_hf_quantizer` runs,
# so `layer_overrides` is None and the comprehension raises -- but only for a
# config whose parallel plan is NON-EMPTY.  Every FP8 `deepseek_v4` repo is in
# that set (`DeepseekV4Config.base_model_ep_plan` has 7 entries), which is why
# `deepseek-ai/DeepSeek-V4-Flash-0731` -- 4.6M downloads -- does not load at
# all on this build, on ANY device.
_FP8_TP_PLAN_BUG = "'NoneType' object has no attribute 'get'"


def _is_fp8_tp_plan_bug(exc: BaseException) -> bool:
    import traceback as _tb

    if not isinstance(exc, AttributeError) or _FP8_TP_PLAN_BUG not in str(exc):
        return False
    return any("quantizer_finegrained_fp8.py" in frame.filename
               and frame.name == "update_tp_plan"
               for frame in _tb.extract_tb(exc.__traceback__))


def neutralize_parallel_plan(config) -> List[str]:
    """Empty the tensor/expert parallel plans on a config and its sub-configs.

    Those plans are a MAP FROM MODULE PATH TO SHARDING KIND.  They are consulted
    only when the model is being split across ranks (`tp_plan=`, a device mesh,
    `accelerate`'s parallel loader); a single-process load -- which is every
    load this tool performs -- never reads them.  Emptying them is therefore
    inert for the forward pass, and it is the SMALLEST edit that walks around
    the FP8 quantizer defect above.

    The alternative that also "works" is a trap and is deliberately not offered:
    `from_pretrained(..., experts_implementation="deepgemm_megamoe")` makes the
    load succeed, because that is the one key `_impl_tp_layer_overrides` has --
    and then the first forward pass dies in
    `integrations/moe.py:get_interface` with

        KeyError: `deepgemm_megamoe` is not a valid experts implementation
                  registered in the `ExpertsInterface`

    i.e. it buys a model that loads and cannot run.  Measured, not assumed:
    `docs/NEW-ARCHITECTURES-FEASIBILITY.md`.

    Returns the dotted names of the plans it actually emptied, for the receipt.
    """
    emptied: List[str] = []

    def strip(node, prefix: str) -> None:
        for attr in ("base_model_tp_plan", "base_model_ep_plan"):
            if getattr(node, attr, None):
                setattr(node, attr, {})
                emptied.append(prefix + attr)
        for name in ("text_config", "vision_config", "audio_config"):
            child = getattr(node, name, None)
            if child is not None and hasattr(child, "__dict__"):
                strip(child, prefix + name + ".")

    strip(config, "")
    return emptied


def load_model(model_dir: str, device: str, dtype_name: str,
               device_map: Any = None, max_memory: Optional[Dict[Any, str]] = None,
               offload_folder: Optional[str] = None,
               drop_parallel_plan: bool = False):
    """Instantiate the checkpoint, optionally without ever materialising it whole.

    `device_map` exists for the reason `docs/GLM53-ROOT-FEASIBILITY.md` R2
    gives: with `device_map=None` `transformers` materialises the entire model
    in CPU RAM and this function then calls `.to(device)`.  At
    `zai-org/GLM-5.3-BF16`'s 1,486.8 GB that exceeds the largest rentable RAM
    and the whole VRAM of an 8x H200 node, so the default path cannot load the
    root model on any machine we can rent.  When a `device_map` is passed the
    model is dispatched by `accelerate` instead, and `.to(device)` is SKIPPED --
    calling it on a dispatched model raises.
    """
    import torch
    import transformers

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype_name]
    config = transformers.AutoConfig.from_pretrained(model_dir)
    architectures = list(getattr(config, "architectures", None) or [])
    extra: Dict[str, Any] = {}
    if drop_parallel_plan:
        emptied = neutralize_parallel_plan(config)
        # Pass the edited config explicitly; otherwise `from_pretrained` reads
        # config.json again and the edit never reaches the quantizer.
        extra["config"] = config
        log(stage="parallel_plan_dropped", plans=emptied)
    if device_map is not None:
        extra["device_map"] = device_map
        if max_memory:
            extra["max_memory"] = max_memory
        if offload_folder:
            extra["offload_folder"] = offload_folder
    if device_map is not None:
        try:
            import accelerate  # noqa: F401
        except Exception:
            raise fail(
                "--device-map needs `accelerate`, which is not installed. Without it "
                "transformers materialises the WHOLE model in CPU RAM and this tool then "
                "calls .to(--device); for zai-org/GLM-5.3-BF16 that is 1,486.8 GB and it "
                "cannot succeed anywhere. Install accelerate, or drop --device-map and "
                "accept the whole-model-resident path.")
    model = None
    info: Dict[str, Any] = {}
    errors = []
    for name in architectures:
        cls = getattr(transformers, name, None)
        if cls is None:
            errors.append("transformers has no %s" % name)
            continue
        try:
            model, info = _from_pretrained(cls, model_dir, torch_dtype, **extra)
            break
        except Exception as exc:  # pragma: no cover - depends on the checkpoint
            if _is_fp8_tp_plan_bug(exc) and not drop_parallel_plan:
                raise fail(
                    "REFUSED: this transformers build cannot even BEGIN to load this FP8 "
                    "checkpoint. `FineGrainedFP8HfQuantizer.update_tp_plan` rewrites the "
                    "config's tensor/expert parallel plan through "
                    "`FP8Experts._impl_tp_layer_overrides.get(config._experts_implementation)`, "
                    "which is None for every implementation except `deepgemm_megamoe` and "
                    "is ALWAYS None at this point in the load, so the rewrite raises "
                    "AttributeError on any FP8 config whose parallel plan is non-empty "
                    "(deepseek_v4 ships 7 entries). This is an upstream defect, not a "
                    "property of the artifact. The safe walk-around is "
                    "--drop-parallel-plan, which empties those plans -- they are read "
                    "only when the model is split across ranks, which this tool never "
                    "does. Do NOT instead pass experts_implementation=deepgemm_megamoe: "
                    "that loads and then dies on the first forward pass. (%s: %s: %s)"
                    % (name, type(exc).__name__, exc))
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
    if model is None:
        try:
            model, info = _from_pretrained(transformers.AutoModelForCausalLM,
                                           model_dir, torch_dtype, **extra)
        except Exception as exc:
            if _is_fp8_tp_plan_bug(exc) and not drop_parallel_plan:
                raise fail(
                    "REFUSED: transformers' FP8 quantizer crashed rewriting this config's "
                    "parallel plan before any weight was read (upstream defect; see "
                    "--drop-parallel-plan). AutoModelForCausalLM: %s: %s"
                    % (type(exc).__name__, exc))
            raise fail("could not instantiate the model (%s); AutoModelForCausalLM: %s: %s"
                       % ("; ".join(errors) or "no architectures declared",
                          type(exc).__name__, exc))
    model.eval()
    if device_map is None:
        model.to(device)
    return model, config, info


def _base_capture(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """`--base-capture` -> the schema's object, not the raw string.

    `dataset.base_capture` is `{dataset_sha256, capture_content_digest,
    repository, revision, note}` or null.  The flag used to be written straight
    through, so ANY capture that named its intended root was refused by the
    validator the capture itself runs -- after the forward pass had been paid
    for.  A bare `repo` or `repo@revision` is the common case and is accepted;
    a JSON object is passed through so the digests can be pinned.
    """
    if value in (None, ""):
        return None
    text = value.strip()
    if text.startswith("{"):
        doc = json.loads(text)
        if not isinstance(doc, dict):
            raise fail("--base-capture JSON must be an object, got %s" % type(doc).__name__)
        doc.setdefault("dataset_sha256", None)
        return doc
    if text.startswith("hf://"):
        text = text[len("hf://"):]
    repository, _, revision = text.partition("@")
    return {"dataset_sha256": None, "capture_content_digest": None,
            "repository": repository or None, "revision": revision or None,
            "note": "named by --base-capture; digests not pinned because the "
                    "comparison partner was not read at capture time"}


def _key_list(value: Any) -> List[str]:
    if not value:
        return []
    try:
        return sorted(str(k) for k in value)
    except TypeError:  # pragma: no cover - a report shape we do not know
        return [str(value)]


def missing_weight_keys(info: Dict[str, Any]) -> List[str]:
    """Parameters `transformers` had to INVENT because the checkpoint lacked them.

    `missing_keys` is not the whole set.  `mismatched_keys` names parameters
    that were present in the checkpoint at the wrong shape, and the loading
    report `transformers` prints for them says, verbatim,

        Reinit due to size mismatch - ckpt: ... vs model: ...

    -- a randomly initialised tensor under a different heading.  `transformers`
    itself unions the two in `LoadStateDictInfo.missing_and_mismatched()`; this
    function reads only `missing_keys` no longer.
    """
    keys = set(_key_list(info.get("missing_keys")))
    for entry in (info.get("mismatched_keys") or []):
        # (key, checkpoint_shape, model_shape)
        keys.add(str(entry[0]) if isinstance(entry, (tuple, list)) and entry else str(entry))
    return sorted(keys)


def load_report(info: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the load told us, including what `to_dict()` drops.

    `observed` is false when no report was produced at all.  That is NOT the
    same as a clean report and this tool must never treat it as one: an
    unexamined load is exactly the state in which a randomly-initialised model
    gets measured and published.
    """
    mismatched = []
    for entry in (info.get("mismatched_keys") or []):
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            mismatched.append({"key": str(entry[0]),
                               "checkpoint_shape": list(entry[1]) if entry[1] else None,
                               "model_shape": list(entry[2]) if entry[2] else None})
        else:
            mismatched.append({"key": str(entry), "checkpoint_shape": None,
                               "model_shape": None})
    conversion = info.get("conversion_errors")
    return {
        "observed": bool(info.get(REPORT_OBSERVED)),
        "conversion_errors_visible": bool(info.get(REPORT_AUGMENTED)),
        "missing_keys": _key_list(info.get("missing_keys")),
        "unexpected_keys": _key_list(info.get("unexpected_keys")),
        "mismatched_keys": mismatched,
        "error_msgs": [str(m) for m in (info.get("error_msgs") or [])],
        "conversion_errors": {str(k): str(v) for k, v in (conversion or {}).items()},
    }


def refuse_on_load_report(report: Dict[str, Any], allow_missing: bool,
                          allow_unexpected: bool = False) -> List[str]:
    """CAPTURE-03. Returns the reinitialised keys when the caller forced through.

    Five independent ways a `transformers` load can hand back a model whose
    forward pass is not the artifact's, in increasing order of subtlety:

      1. `missing_keys`   -- the classic. Observed on Fruit: routed experts
         shipped as exl3-trellis atoms, reported missing, randomly initialised,
         mean ~0 std 0.0199, model runs.
      2. `mismatched_keys` -- present but the wrong shape. "Reinit due to size
         mismatch". Same harm, different heading, and this guard used to ignore
         it entirely.
      3. `error_msgs`     -- a state-dict copy that threw.
      4. `conversion_errors` -- a `WeightConverter` that threw. The parameters
         it was building are skipped, and `LoadStateDictInfo.to_dict()` -- the
         dict this tool is handed -- deliberately omits the field. For a model
         routed through the `qwen2_moe` fusion pattern this is 96.7% of the
         checkpoint hiding behind a field we were not shown.
      5. `unexpected_keys` -- the SILENT one, and the reason this list grew a
         fifth entry.  Every parameter the model builds was filled, so nothing
         above fires; the checkpoint simply carried tensors the loaded model
         had no home for.  That is sometimes benign (GLM-5.3-BF16 ships an MTP
         layer `GlmMoeDsaForCausalLM` does not build, 791 tensors of it) and
         sometimes it is the whole defect.  On `Qwen/Qwen3.8-27B-FP8` (M1,
         learning 6/7) the line read `unexpected: 64`: the producer's
         `modules_to_not_convert` listed `...mlp.gate`, `should_convert_module`
         matches it with a start-anchored `re.match`, so `...mlp.gate_proj`
         matched too and 65 of 65 gate_proj modules skipped FP8 conversion.
         Their 64 block-scale tensors then had nowhere to go and fell out as
         "unexpected", the fp8 payload was read into a bf16 Linear with the
         scale never applied, and the model produced confidently wrong values
         in that projection.  Nothing raised.  That one log line was the only
         signal.  It is now a refusal.

    1, 2 and 5 are overridable -- by `--allow-missing-weights` and
    `--allow-unexpected-tensors` respectively -- and each override forces a
    BLOCKING disclosure. 3 and 4 are not: an exception during loading means we
    cannot say what the parameter now holds, so there is no disclosure that
    would make a number measured on it readable.
    """
    if not report["observed"]:
        raise fail(
            "REFUSED: this load produced NO loading report, so nothing here has checked "
            "whether transformers randomly initialised parameters the checkpoint did not "
            "provide. An unexamined load is not a clean load. Use a transformers build "
            "whose from_pretrained honours output_loading_info=True.")
    if not report["conversion_errors_visible"]:
        raise fail(
            "REFUSED: this transformers build did not expose `conversion_errors`, the "
            "field that records a failure of the on-the-fly WeightConverter. For a fused-"
            "expert MoE checkpoint that converter owns almost the entire checkpoint, so a "
            "guard that cannot read the field cannot claim the weights are the artifact's.")
    if report["conversion_errors"]:
        keys = sorted(report["conversion_errors"])
        raise fail(
            "REFUSED: %d weight CONVERSION error(s) during loading -- a WeightConverter "
            "raised, so the parameters it was assembling were skipped and hold whatever "
            "initialisation was left behind: %s%s. This is not overridable: an exception "
            "mid-conversion means the parameter's contents are unknown, and there is no "
            "disclosure that makes an unknown weight measurable. First error: %s"
            % (len(keys), ", ".join(keys[:6]),
               " (+%d more)" % (len(keys) - 6) if len(keys) > 6 else "",
               report["conversion_errors"][keys[0]].strip().splitlines()[-1][:400]))
    if report["error_msgs"]:
        raise fail(
            "REFUSED: %d error(s) were raised while copying the state dict into the model: "
            "%s. Not overridable." % (len(report["error_msgs"]),
                                      " | ".join(report["error_msgs"])[:600]))
    reinitialised = sorted({m["key"] for m in report["mismatched_keys"]}
                           | set(report["missing_keys"]))
    if reinitialised and not allow_missing:
        mismatched = len(report["mismatched_keys"])
        raise fail(
            "REFUSED: %d parameter(s) were NOT usable from the checkpoint and were randomly "
            "initialised by transformers, so this model's forward pass is not the "
            "artifact's: %s%s.%s Either this build of transformers cannot read the "
            "checkpoint's storage format, or the checkpoint is incomplete. Pass "
            "--allow-missing-weights only if you can defend a number measured on a "
            "partially random model; it forces a BLOCKING disclosure."
            % (len(reinitialised), ", ".join(reinitialised[:6]),
               " (+%d more)" % (len(reinitialised) - 6) if len(reinitialised) > 6 else "",
               (" %d of them were present at the WRONG SHAPE ('Reinit due to size "
                "mismatch')." % mismatched) if mismatched else ""))
    unexpected = list(report.get("unexpected_keys") or [])
    if unexpected and not allow_unexpected:
        raise fail(
            "REFUSED: %d checkpoint tensor(s) were loaded from the artifact but this "
            "architecture has no home for them, so they took no part in the forward pass: "
            "%s%s. What that USUALLY means: a quantization path silently did not engage. "
            "The producer's exclusion list, the quantizer's module matcher or this "
            "`transformers` build disagreed about which modules carry quant state, the "
            "affected modules were left in their unquantized form, and their scale/zero "
            "tensors then had nowhere to go and landed here. A capture taken in that state "
            "is a confident number for a projection nobody quantized. (It is also how a "
            "legitimately unused block looks -- an MTP or draft layer the architecture does "
            "not build -- which is why there is an override.) Cross-check the converted/"
            "excluded module split against the checkpoint's real tensor names before "
            "trusting anything captured here. Pass --allow-unexpected-tensors only if you "
            "have done that; it forces a BLOCKING disclosure."
            % (len(unexpected), ", ".join(unexpected[:6]),
               " (+%d more)" % (len(unexpected) - 6) if len(unexpected) > 6 else ""))
    return reinitialised


def head_module(model):
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise fail("model.get_output_embeddings() did not return a weight-bearing head; "
                   "a tied-embedding model must still expose one")
    if getattr(head, "bias", None) is not None:
        raise fail("the head carries a bias; the hidden-replay contract assumes none "
                   "(HEAD gate). Capture in logit form instead.")
    return head


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def st_bytes(name: str, dtype: str, shape: Sequence[int], raw: bytes,
             metadata: Optional[Dict[str, str]] = None) -> bytes:
    header: Dict[str, Any] = {name: {"dtype": dtype, "shape": [int(v) for v in shape],
                                     "data_offsets": [0, len(raw)]}}
    if metadata:
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + raw


def _bf16_raw(tensor) -> bytes:
    import torch

    flat = tensor.detach().contiguous().cpu()
    if flat.dtype != torch.bfloat16:
        raise fail("refusing to store a %s tensor as BF16: that would be a lossy cast the "
                   "manifest claims is lossless" % flat.dtype)
    return flat.view(torch.uint16).numpy().tobytes()


def _source_files(args: argparse.Namespace) -> Dict[str, str]:
    """Every file that decided the arithmetic, hashed.

    `layer_outer.py` is listed only when it ran: a window-outer capture is not
    made by that file and must not claim to be.
    """
    files = {"k6/tools/hf_capture.py": F.sha256_file(os.path.abspath(__file__))}
    if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER:
        files["k6/tools/layer_outer.py"] = F.sha256_file(
            os.path.abspath(layer_outer.__file__))
    return files


def _run_layer_outer(args, model, streamer, panel, tap, forward_window, layer_seconds,
                     window_count=None):
    """Drive `layer_outer.run_panel` and hand back one tapped hidden state per window.

    The saving is in weight LOADING, not compute.  `on_layer_start` /
    `on_layer_end` are where it is banked: with `--layer-residency stream` each
    layer's weights are materialised once here for the whole panel and freed
    before the next, so a run reads the checkpoint tree once instead of once
    per window.
    """
    if streamer is not None:
        layers = streamer.layers
    else:
        try:
            _, layers = layer_outer.find_decoder_layers(model)
        except layer_outer.LayerOuterError as exc:
            raise fail(str(exc))
    # The sanity probe rides along as ONE EXTRA WINDOW, so it costs one more
    # forward per layer and no additional weight loading at all -- the layers are
    # already being loaded for the panel. It is never a separate pass over the
    # checkpoint, which for a 1.5 TB model would be the whole capture again.
    if window_count is None:
        window_count = len(panel.windows)
    log(stage="schedule", schedule=args.schedule, residency=args.layer_residency,
        layers=len(layers), windows=len(panel.windows),
        forward_windows=window_count)

    # The window-outer loop discovers a ragged panel on the window that differs,
    # having paid only for the windows before it. This schedule runs every
    # window's forward before the record loop is reached, so the same refusal
    # would arrive after the whole panel had been paid for. Check it first.
    lengths = {len(window["token_ids"]) for window in panel.windows}
    if len(lengths) > 1:
        raise fail("a ragged panel is not supported: context lengths %s"
                   % sorted(lengths))

    watcher = args.resident_watcher

    def on_layer_start(index: int) -> None:
        if streamer is not None:
            started = time.monotonic()
            try:
                streamer.load_layer(index)
            except layer_outer.LayerOuterError as exc:
                raise fail(str(exc))
            # Sampled with the layer loaded -- i.e. at the moment the schedule
            # holds the most it ever holds.
            resident = watcher.sample()
            log(stage="layer_load", index=index, seconds=round(time.monotonic() - started, 3),
                checkpoint_tensors=streamer.layer_counts.get(index),
                resident_weight_bytes=resident)

    def on_layer_end(index: int) -> None:
        if streamer is not None:
            streamer.free_layer(index)
            watcher.sample()

    # The layer-outer schedule runs len(layers) x len(windows) forwards inside
    # ONE `run_panel` call, and only the per-layer boundary logged anything --
    # so the longest layer looked like a hang.  The meter counts the real inner
    # unit.  `layer_seconds` is already being accumulated here; the meter reads
    # nothing else and adds one integer increment per forward.
    inner_meter = progress_meter.Progress(
        len(layers) * window_count, label="layer-outer forwards",
        interval=getattr(args, "progress_seconds", progress_meter.DEFAULT_INTERVAL_SECONDS),
    )

    def timed_forward(window_index: int) -> None:
        started = time.monotonic()
        forward_window(window_index)
        layer_seconds[window_index] += time.monotonic() - started
        inner_meter.update(1)

    def collect(window_index: int):
        window_id = (panel.windows[window_index]["window_id"]
                     if window_index < len(panel.windows) else "sanity-probe")
        if len(tap) != 1:
            raise fail("window %s: the head hook fired %d times, expected exactly 1 -- an "
                       "extra forward would misalign the capture"
                       % (window_id, len(tap)))
        return tap[0]

    try:
        hidden = layer_outer.run_panel(model, layers, timed_forward, window_count,
                                       log, on_layer_start=on_layer_start,
                                       on_layer_end=on_layer_end, collect=collect)
    except layer_outer.LayerOuterError as exc:
        raise fail(str(exc))
    finally:
        inner_meter.close()
    del tap[:]
    if streamer is not None:
        streamer.close()
    for index, value in enumerate(hidden):
        if value is None:  # pragma: no cover - run_panel guarantees this
            raise fail("window %d produced no hidden state under the layer-outer schedule"
                       % index)
    return hidden


def _prepare_probe(args: argparse.Namespace, model_dir: str) -> Dict[str, Any]:
    """Tokenize the sanity prompt, or say why the probe cannot run.

    A refusal here rather than a silent skip whenever the caller asked for
    ENFORCEMENT: a fail-closed check that could not run has not passed.
    """
    prompt = getattr(args, "sanity_prompt", generation_probe.DEFAULT_PROMPT)
    expect = getattr(args, "sanity_expect", None)
    if expect is not None and not expect.strip():
        expect = None
    if not getattr(args, "sanity_check", True):
        return {"stub": generation_probe.skipped(
            "--no-sanity-check was passed; the probe did not run",
            prompt=prompt, expect=expect, enforced=False)}
    tokenizer, reason = generation_probe.load_tokenizer(model_dir)
    if tokenizer is None:
        if expect is not None:
            raise fail(
                "REFUSED: --sanity-expect %r asks for a FAIL-CLOSED generation check and "
                "the tokenizer could not be loaded from %s (%s). A check that could not "
                "run has not passed. Ship the tokenizer with the checkpoint, or pass "
                "--sanity-expect '' to record the probe without enforcing it."
                % (expect, model_dir, reason))
        return {"stub": generation_probe.skipped(
            "tokenizer unavailable: %s" % reason,
            prompt=prompt, expect=expect, enforced=False)}
    try:
        token_ids = generation_probe.encode_prompt(tokenizer, prompt)
    except generation_probe.ProbeRefusal as exc:
        raise fail(str(exc))
    log(stage="sanity_probe_planned", prompt=prompt, expect=expect,
        prompt_tokens=len(token_ids))
    return {"token_ids": token_ids, "tokenizer": tokenizer,
            "prompt": prompt, "expect": expect}


def _resolve_probe(args: argparse.Namespace, plan: Dict[str, Any],
                   logits: List[Any]) -> Dict[str, Any]:
    if not plan:
        return None
    if "stub" in plan:
        log(stage="sanity_probe", **{k: v for k, v in plan["stub"].items()
                                     if k != "schema"})
        return plan["stub"]
    if not logits:
        if plan["expect"] is not None:
            raise fail(
                "REFUSED: the generation sanity probe was enforced but the model "
                "returned no logits for its window. The head is where this capture "
                "taps its hidden states, so a forward that produced none is a "
                "malfunction of the capture itself, not of the probe.")
        return generation_probe.skipped(
            "the model returned no logits for the probe window",
            prompt=plan["prompt"], expect=plan["expect"], enforced=False)
    try:
        verdict = generation_probe.evaluate(logits[0], plan["tokenizer"],
                                            prompt=plan["prompt"],
                                            expect=plan["expect"])
    except generation_probe.ProbeRefusal as exc:
        raise fail(str(exc))
    log(stage="sanity_probe", status=verdict["status"], top1=verdict["top1_text"],
        probability=round(verdict["top1_probability"], 6),
        entropy_nats=round(verdict["entropy_nats"], 4),
        uniform_entropy_nats=round(verdict["uniform_entropy_nats"], 4),
        enforced=verdict["enforced"])
    return verdict


def _race_trailing_files(args: argparse.Namespace, model_dir: str, plan) -> List[str]:
    """Every published file that is NOT a shard the plan already covers.

    Best effort by design: if the repo listing cannot be obtained the capture
    still runs -- the shards are what the arithmetic needs -- and the log says
    the sidecars were not enumerated rather than pretending they were absent.
    """
    shards = set(plan.needed_at)
    try:
        if getattr(args, "race_simulate_source", None):
            names = sorted(os.listdir(args.race_simulate_source))
        else:
            from huggingface_hub import list_repo_files

            names = list_repo_files(
                args.race_repo, revision=args.race_revision or args.model_revision,
                token=os.environ.get("HF_TOKEN") or None)
    except Exception as exc:  # pragma: no cover - network/listing dependent
        log(stage="race_sidecars_unlisted", reason="%s: %s" % (type(exc).__name__, exc))
        return []
    # A file `race_bootstrap` already pinned at this revision is not re-fetched:
    # it is on disk, at the same revision, and in the simulate harness a
    # re-fetch would also charge it the injected delay -- which would make the
    # A/B compare two different file sets.
    return [name for name in sorted(names)
            if name not in shards and not name.startswith(".")
            and not os.path.exists(os.path.join(model_dir, name))]


def _start_race_fetch(args: argparse.Namespace, model_dir: str):
    """Start the overlapped fetch, or return None when --race-repo was not given.

    Refuses rather than degrading: race mode only means anything under the
    layer-outer/stream schedule (that is the only loop that reads layer N after
    layer N-1 has already been used), and it needs the checkpoint index, which is
    the file that says which shard holds which layer.
    """
    if not getattr(args, "race_repo", None):
        return None
    if args.schedule != layer_outer.SCHEDULE_LAYER_OUTER \
            or args.layer_residency != layer_outer.RESIDENCY_STREAM:
        raise fail(
            "--race-repo needs --schedule layer-outer --layer-residency stream. Every "
            "other schedule materialises the whole model before the first window, so "
            "there is no point at which a not-yet-arrived layer could be waited for; "
            "overlapping the fetch with it would just be downloading during a load.")
    if not os.path.isdir(model_dir):
        raise fail("--race-repo needs --model to be a LOCAL directory holding the "
                   "bootstrap files (config.json, the tokenizer, "
                   "model.safetensors.index.json); the shards land into it. Got %r"
                   % model_dir)
    try:
        weight_map = race_fetch.read_index(model_dir)
        plan = race_fetch.plan_shards(weight_map, args.race_layer_key_regex)
    except race_fetch.RaceFetchError as exc:
        raise fail(str(exc))
    revision = args.race_revision or args.model_revision
    if getattr(args, "race_simulate_source", None):
        download = race_fetch.simulated_downloader(
            args.race_simulate_source, model_dir, args.race_simulate_seconds)
        log(stage="race_simulated_downloader", source=args.race_simulate_source,
            seconds_per_file=args.race_simulate_seconds,
            warning="TEST HARNESS: this capture is not a measurement and is "
                    "stamped with a blocking disclosure")
    else:
        download = race_fetch.hf_downloader(args.race_repo, revision, model_dir,
                                            token=os.environ.get("HF_TOKEN") or None)
    # The whole repo, not just the shards: `fetch_target` pulls everything, and
    # a race-mode tree that quietly lacked the release's own SHA256SUMS would
    # skip a verification the ordinary path performs. They are queued AFTER every
    # layer, so nothing ever waits on them.
    trailing = _race_trailing_files(args, model_dir, plan)
    fetcher = race_fetch.RaceFetcher(plan, download, workers=args.race_workers,
                                     log=log, timeout=args.race_timeout_seconds,
                                     trailing_files=trailing)
    # A file already on disk (the bootstrap fetch put config/index/tokenizer
    # there) is still enqueued: hf_hub_download is a no-op on an unchanged file
    # and returns immediately, and enqueueing it keeps the gate's record the
    # single source of truth about what has landed.
    log(stage="race_plan", repo=args.race_repo, revision=revision,
        shards=len(plan.needed_at), resident_shards=len(plan.resident_shards),
        layers=plan.layer_count, unmatched_keys=plan.unmatched_keys,
        layer_key_regex=plan.layer_key_regex)
    fetcher.start()
    return fetcher


def run_capture(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    started = time.monotonic()
    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(args.model, revision=args.model_revision,
                                      local_dir=args.model_cache)
        log(stage="snapshot", repo=args.model, dir=model_dir)

    # RACE MODE.  Start the background fetch before anything else touches the
    # tree: from here on the checkpoint is arriving, not present.
    fetcher = _start_race_fetch(args, model_dir)

    # The checkpoint identity is a sha256 over EVERY shard, so in race mode it
    # cannot be computed here -- most of the tree is still on the wire, and a
    # digest over what happens to have landed would be an identity for a
    # checkpoint that does not exist. It is computed at assembly time instead,
    # once the fetch has joined, over the complete tree. Same preimage, same
    # value; only the moment moves.
    identity = identity_files = None
    if fetcher is None:
        identity, identity_files = checkpoint_identity(model_dir)
        log(stage="checkpoint_identity", sha256=identity, files=len(identity_files))

    layer_outer.reset_peak_memory(args.device)
    max_memory = json.loads(args.max_memory) if args.max_memory else None
    streamer = None
    if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER \
            and args.layer_residency == layer_outer.RESIDENCY_STREAM:
        # The whole point: never materialise the model, materialise one layer at
        # a time.  `load_model`'s path is not reachable from here -- it would
        # allocate the 1,486.8 GB this schedule exists to avoid.
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir)
        architectures = list(getattr(config, "architectures", None) or [])
        cls = None
        for name in architectures:
            cls = getattr(transformers, name, None)
            if cls is not None:
                break
        if cls is None:
            # No AutoModelForCausalLM fallback here, unlike `load_model`: the
            # meta-device build calls `cls(config)` directly and an auto class
            # cannot be instantiated that way. A refusal naming the config's own
            # architectures is more useful than a TypeError from inside torch.
            raise fail(
                "the layer-outer streaming loader needs a concrete architecture class; "
                "config.architectures is %r and transformers exposes none of them. "
                "AutoModelForCausalLM cannot be built on the meta device without one; "
                "use --schedule window-outer, which can fall back to it."
                % (architectures,))

        def layer_guard(index: int, info: Dict[str, Any]) -> None:
            # CAPTURE-03, per streamed layer.  The window-outer path gets one
            # load and one guard; this path gets one load per layer, so the
            # guard runs per layer -- otherwise the streamed weights (97.5% of
            # GLM-5.3 by bytes) would be the only unexamined part of the model.
            report = load_report(info)
            reinit = refuse_on_load_report(report, args.allow_missing_weights,
                                           args.allow_unexpected_tensors)
            if reinit:
                log(stage="layer_missing_weights", index=index, count=len(reinit),
                    keys=reinit[:8])

        try:
            streamer = layer_outer.build_streamed_model(
                model_dir, cls, config, args.dtype, args.device, log,
                layer_guard=layer_guard, gate=fetcher)
        except layer_outer.LayerOuterError as exc:
            raise fail(str(exc))
        except race_fetch.RaceFetchError as exc:
            raise fail(str(exc))
        model = streamer.model
        loading_info = layer_outer.streamed_loading_info(streamer)
    else:
        model, config, loading_info = load_model(
            model_dir, args.device, args.dtype, device_map=args.device_map,
            max_memory=max_memory, offload_folder=args.offload_folder,
            drop_parallel_plan=args.drop_parallel_plan)

    # CAPTURE-03.  A checkpoint whose tensors this `transformers` build cannot
    # name does not fail to load: `from_pretrained` RANDOMLY INITIALISES the
    # parameters it could not find, logs a table, and returns a model that
    # runs.  Captured, that produces a confident number for weights nobody ever
    # measured -- the single most dangerous outcome this tool can have.
    # Observed on malaiwah/GLM-5.2-SIQ-Fruit, whose routed experts ship as
    # exl3-trellis atoms (`.trellis`/`.suh`/`.svh`/`.mcg`): stock transformers
    # reported `model.layers.{3..12}.mlp.experts.{gate_up,down}_proj` MISSING
    # and handed back a model with random experts, mean ~0, std 0.0199.
    #
    # `refuse_on_load_report` widens that to the three neighbouring ways the
    # same harm arrives -- mismatched shapes, state-dict errors, and converter
    # errors -- and to the case where no report was produced at all.
    report = load_report(loading_info)
    log(stage="load_report", observed=report["observed"],
        conversion_errors_visible=report["conversion_errors_visible"],
        missing=len(report["missing_keys"]), mismatched=len(report["mismatched_keys"]),
        unexpected=len(report["unexpected_keys"]),
        conversion_errors=len(report["conversion_errors"]),
        error_msgs=len(report["error_msgs"]),
        unexpected_sample=report["unexpected_keys"][:6])
    missing = refuse_on_load_report(report, args.allow_missing_weights,
                                    args.allow_unexpected_tensors)
    if missing:
        log(stage="missing_weights", count=len(missing), keys=missing[:12])

    # The high-water mark of MATERIALISED weight bytes. RSS cannot answer this
    # on the CPU path (safetensors mmaps the shards, so the page cache lands in
    # ru_maxrss whether or not the schedule ever held those bytes as its own),
    # and it is the figure a "does GLM-5.3 fit" projection actually needs.
    args.resident_watcher = layer_outer.ResidentWeightPeak(model)
    args.resident_watcher.sample()

    head = head_module(model)
    vocab_size = int(head.weight.shape[0])
    hidden_size = int(head.weight.shape[1])
    if int(getattr(config, "vocab_size", vocab_size)) != vocab_size:
        log(stage="warning", message="config.vocab_size %s != head rows %d; the head wins"
            % (getattr(config, "vocab_size", None), vocab_size))
    log(stage="model", vocab_size=vocab_size, hidden_size=hidden_size,
        dtype=str(head.weight.dtype), device=args.device,
        architectures=list(getattr(config, "architectures", None) or []))

    # The panel's tokenizer identity is part of panel IDENTITY (PANEL-D6) and the
    # schema requires a string.  Default it to the checkpoint whose tokenizer
    # actually produced these ids rather than emitting null and failing `verify`
    # only after the whole capture has been paid for.
    tokenizer_id = args.tokenizer_id or args.weights_repository or args.model
    panel = load_panel(args.panel, args.panel_role, args.windows, tokenizer_id, vocab_size)
    log(stage="panel", windows=len(panel.windows), panel_json=panel.source,
        receipt_sha256=panel.receipt_sha256)

    tap: List[Any] = []

    def pre_hook(module, inputs):
        hidden = inputs[0]
        if not torch.is_tensor(hidden):
            raise fail("the head pre-hook received a non-tensor input")
        if hidden.dtype != torch.bfloat16:
            raise fail("head input dtype %s != bfloat16 -- the 'bf16 capture is lossless' "
                       "claim would be false; refusing" % hidden.dtype)
        if hidden.ndim != 3 or hidden.shape[0] != 1 or int(hidden.shape[-1]) != hidden_size:
            raise fail("head input shape %s is not [1, seq, %d]"
                       % (tuple(hidden.shape), hidden_size))
        tap.append(hidden.detach().squeeze(0).to("cpu", copy=True))

    handle = head.register_forward_pre_hook(pre_hook)

    writer = dsmanifest.DatasetWriter(args.out)
    panel_records: List[Dict[str, Any]] = []
    capture_records: List[Dict[str, Any]] = []
    context_length = None

    # THE ONE FORWARD CALL, shared by both schedules.  It is written once so
    # that the layer-outer path cannot drift from the window-outer path in the
    # inputs it builds: same dtypes, same devices, same kwargs, same
    # `use_cache=False`.  What differs between the schedules is only WHEN this
    # is called and which layers are resident when it is.
    # -- the generation sanity probe's window -------------------------------
    # Prepared BEFORE the panel loop because under the layer-outer schedule it
    # must be pushed through the same `run_panel` call: a probe run afterwards
    # would need every layer a second time, i.e. a second read of the whole
    # checkpoint.
    probe_plan = _prepare_probe(args, model_dir)
    probe_index = len(panel.windows) if probe_plan.get("token_ids") else None
    probe_logits: List[Any] = []

    def forward_window(index: int) -> None:
        if probe_index is not None and index == probe_index:
            token_ids = probe_plan["token_ids"]
            mask = [1] * len(token_ids)
        else:
            window = panel.windows[index]
            token_ids = window["token_ids"]
            mask = window["mask"]
        del tap[:]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        attention_mask = torch.from_numpy(
            np.asarray(mask, dtype=np.int64).reshape(1, -1)).to(args.device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False)
        if probe_index is not None and index == probe_index:
            logits = getattr(out, "logits", None)
            if logits is None and isinstance(out, (tuple, list)) and out:
                logits = out[0]
            if logits is not None:
                del probe_logits[:]
                probe_logits.append(logits[0, -1].detach().to("cpu", copy=True))

    precomputed: Optional[List[Any]] = None
    forward_count = len(panel.windows) + (1 if probe_index is not None else 0)
    layer_seconds: List[float] = [0.0] * forward_count
    if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER:
        precomputed = _run_layer_outer(args, model, streamer, panel, tap, forward_window,
                                       layer_seconds, window_count=forward_count)
    elif probe_index is not None:
        # Window-outer: the model is fully resident, so the probe is one extra
        # forward and costs nothing but its own compute.
        forward_window(probe_index)
    probe = _resolve_probe(args, probe_plan, probe_logits)

    # The outer meter: how far through the panel, and when this capture ends.
    # `every=1` because a window is minutes long -- every completed one earns a
    # line even when the 30 s throttle has not elapsed.
    window_meter = progress_meter.Progress(
        len(panel.windows), label="windows", every=1,
        interval=getattr(args, "progress_seconds", progress_meter.DEFAULT_INTERVAL_SECONDS),
    )
    try:
        for index, window in enumerate(panel.windows):
            ids = window["token_ids"]
            mask = window["mask"]
            if context_length is None:
                context_length = len(ids)
            elif context_length != len(ids):
                raise fail("window %s has context length %d, expected %d; a ragged panel is "
                           "not supported" % (window["window_id"], len(ids), context_length))
            token_rel = writer.add_token_file(index, ids)
            mask_rel, mask_sha = writer.add_mask_file(index, window["mask_bytes"])
            if mask_sha != window["mask_sha256"]:
                raise fail("window %s: the mask bytes we wrote hash to %s, not the sealed %s"
                           % (window["window_id"], mask_sha[:12], window["mask_sha256"][:12]))

            if precomputed is not None:
                # layer-outer: this window's forward already happened, spread
                # across the layer loop.  Everything below -- the row
                # selection, the BF16 payload, the records -- is the same code
                # on the same tensor.
                hidden_full = precomputed[index]
                elapsed = layer_seconds[index]
            else:
                elapsed = time.monotonic()
                forward_window(index)
                elapsed = time.monotonic() - elapsed
                if len(tap) != 1:
                    raise fail("window %s: the head hook fired %d times, expected exactly 1 "
                               "-- an extra forward would misalign the capture"
                               % (window["window_id"], len(tap)))
                hidden_full = tap[0]
            if hidden_full.shape[0] != len(ids):
                raise fail("window %s: hidden seq %d != tokens %d"
                           % (window["window_id"], hidden_full.shape[0], len(ids)))
            causal = (np.asarray(mask[:-1], dtype=np.bool_)
                      & np.asarray(mask[1:], dtype=np.bool_))
            selected = hidden_full[:-1][torch.from_numpy(causal)].contiguous()
            if int(selected.shape[0]) != window["prediction_positions"]:
                raise fail("window %s: %d scored rows != the panel's declared %d"
                           % (window["window_id"], int(selected.shape[0]),
                              window["prediction_positions"]))

            payload = st_bytes(F.TENSOR_KEY_HIDDEN, "BF16",
                               [int(selected.shape[0]), hidden_size], _bf16_raw(selected),
                               metadata={"capture_role": "hidden_states_pre_lm_head",
                                         "cut_point": CUT_POINT,
                                         "window_id": window["window_id"],
                                         "cold_run": args.cold_run,
                                         "dtype": "bfloat16"})
            rel = writer.add_capture_tensor(index, payload, "hidden")
            full = os.path.join(args.out, rel)

            panel_records.append(dsmanifest.panel_record(
                index=index, token_file=token_rel, token_ids=ids,
                prediction_positions=window["prediction_positions"],
                window_id=window["window_id"], attention_mask_file=mask_rel,
                attention_mask_sha256=mask_sha, role=window["role"],
                domain=window["domain"], document_id=window["document_id"]))
            capture_records.append(dsmanifest.tensor_record(
                index=index, filename=os.path.basename(rel), abs_path=full,
                key=F.TENSOR_KEY_HIDDEN, dtype="BF16",
                shape=[int(selected.shape[0]), hidden_size],
                scored_rows=int(selected.shape[0]),
                token_ids_json_sha256=F.token_ids_json_sha256(ids),
                token_ids_sha256_legacy=F.token_ids_json_sha256_legacy(ids),
                attention_mask_sha256=mask_sha, window_id=window["window_id"],
                role=window["role"], domain=window["domain"],
                document_id=window["document_id"],
                elapsed_seconds=round(elapsed, 3)))
            log(stage="window", index=index, window_id=window["window_id"],
                rows=int(selected.shape[0]), bytes=os.path.getsize(full),
                seconds=round(elapsed, 3))
            window_meter.update(1)
            del tap[:]
    finally:
        window_meter.close(suffix="capture complete")
        handle.remove()

    # -- head ---------------------------------------------------------------
    head_weight = head.weight
    head_rel = writer.add_head_payload(
        st_bytes("lm_head.weight", "BF16", list(head_weight.shape), _bf16_raw(head_weight)))
    head_full = os.path.join(args.out, head_rel)
    head_content = F.tensor_content_sha256(head_full, "lm_head.weight")
    log(stage="head", file=head_rel, tensor_content_sha256=head_content,
        bytes=os.path.getsize(head_full))

    # -- measured, not predicted -------------------------------------------
    # docs/GLM53-ROOT-FEASIBILITY.md §2 projects a peak from the census. A
    # projection is not a measurement, so every run now reports what it
    # actually used and the projection can be rebuilt on top of numbers.
    args.resident_watcher.sample()
    memory = layer_outer.peak_memory(args.device)
    memory.update({"peak_resident_weight_bytes": args.resident_watcher.peak,
                   "peak_resident_weight_gb": round(args.resident_watcher.peak / 1e9, 3),
                   "peak_resident_weight_detail": args.resident_watcher.detail,
                   "resident_weight_note":
                       "the maximum, over the run, of the materialised parameter+buffer "
                       "bytes. On the CPU path peak_rss_bytes ALSO counts the safetensors "
                       "mmap page cache, which is evictable and is not held by the "
                       "schedule; this figure is not confounded by it. On CUDA, "
                       "peak_cuda_allocated_bytes is the authoritative number because it "
                       "includes activations and workspace and has no page cache.",
                   "schedule": args.schedule, "device": args.device,
                   "layer_residency": (args.layer_residency
                                       if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER
                                       else None),
                   "windows": len(panel.windows), "context_length": context_length,
                   "hidden_size": hidden_size, "vocab_size": vocab_size,
                   "model": args.weights_repository or args.model,
                   "decoder_layers": (len(streamer.layers) if streamer is not None
                                      else None)})
    log(stage="peak_memory", **memory)
    if args.memory_report:
        directory = os.path.dirname(os.path.abspath(args.memory_report))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.memory_report, "w", encoding="utf-8") as handle:
            json.dump(memory, handle, indent=2, sort_keys=True)
            handle.write("\n")

    # RACE MODE: the tail of the fetch. Every layer has been read by now, so
    # what is left is whatever the plan did not need -- and the checkpoint
    # IDENTITY is a digest over the complete tree, so it cannot be taken until
    # this returns. The wait is measured and reported rather than hidden: it is
    # the part of the fetch the overlap did NOT hide.
    race_report = None
    if fetcher is not None:
        join_started = time.monotonic()
        fetcher.join()
        tail = time.monotonic() - join_started
        race_report = fetcher.report()
        race_report["tail_join_seconds"] = round(tail, 3)
        log(stage="race_fetch_joined", tail_seconds=round(tail, 3),
            blocked_seconds=race_report["blocked_seconds"],
            files=race_report["files"])
        identity, identity_files = checkpoint_identity(model_dir)
        log(stage="checkpoint_identity", sha256=identity, files=len(identity_files))
        if getattr(args, "race_report", None):
            directory = os.path.dirname(os.path.abspath(args.race_report))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(args.race_report, "w", encoding="utf-8") as handle:
                json.dump({"plan": fetcher.plan.to_dict(), "fetch": race_report},
                          handle, indent=2, sort_keys=True)
                handle.write("\n")

    return _assemble(args, writer, panel, panel_records, capture_records,
                     context_length=context_length, vocab_size=vocab_size,
                     hidden_size=hidden_size, head_rel=head_rel, head_full=head_full,
                     head_content=head_content, head_shape=list(head_weight.shape),
                     model_dir=model_dir, identity=identity, identity_files=identity_files,
                     config=config, started=started, missing_weights=missing,
                     load_report=report, probe=probe, race_report=race_report)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _stack_fingerprint(device: str) -> Dict[str, Any]:
    import torch

    fingerprint = {
        "schema": "malaiwah.stack-fingerprint.v1",
        "engine": "transformers-eager",
        "torch_version": torch.__version__,
        "device": device,
        "device_name": None,
        "cuda_runtime_version": None,
        "numeric_policy": "default torch matmul precision; no TF32 override applied",
        "attention_backend": os.environ.get("ATTN_IMPLEMENTATION", "model default"),
    }
    try:
        import transformers

        fingerprint["transformers_version"] = transformers.__version__
    except Exception:
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        fingerprint["device_name"] = torch.cuda.get_device_name(0)
        fingerprint["cuda_runtime_version"] = torch.version.cuda
    return fingerprint


def _scope(args: argparse.Namespace) -> Dict[str, Any]:
    if args.scope_file:
        doc = json.loads(open(args.scope_file, "r", encoding="utf-8").read())
        return dsmanifest.scope_block(doc["assignments"], doc["head_policy"],
                                      doc.get("kv_cache_dtype", "bf16"), doc["policy"])
    return dsmanifest.native_scope()



def preview_banner(manifest) -> str:
    """The preview warning, at the TOP of the card, in the reader's first screen.

    Prepended in `render_card` rather than in `default_card_body` on purpose: a
    capture run with `--readme` supplies its own body, and the one thing that
    must never be lost by supplying your own body is the statement that this
    dataset is preliminary.
    """
    block = manifest.get("preview") or {}
    if not block:
        return ""
    return "\n".join([
        "> ## ⚠ PRELIMINARY — NOT THE FINAL ROOT",
        "> ",
        "> " + block["statement"].replace("\n", "\n> "),
        "> ",
        "> | | |",
        "> |---|---|",
        "> | cold runs backing this dataset | **%s** |" % block.get("run_count"),
        "> | cross-run determinism demonstrated | **no** |",
        "> | will ever be updated in place | **no** |",
        "> | superseded by | **`%s`** |" % block.get("superseded_by"),
        "> | usable as a registry reference | **no** (`not_submittable`) |",
        "",
        "",
    ])


def render_card(args, manifest, body: str) -> str:
    """A dataset card whose frontmatter carries the required x_fidelity block."""
    sys.path.insert(0, os.path.join(REPO, "bin"))
    from fidelity import cardmeta

    body = preview_banner(manifest) + body

    front = {
        "license": "mit",
        "tags": ["fidelity", "kl-divergence", "fidelity-provenance",
                 "fidelity-dataset", "fidelity-%s" % args.role],
        "x_fidelity": cardmeta.build_dataset_x_fidelity(
            None, repository=args.repository, manifest=manifest),
    }
    front, _ = cardmeta.split_card(cardmeta.render_card(front, body))
    return cardmeta.render_card(front, body)


def _probe_card_line(manifest) -> str:
    """One sentence a reader can check without opening the manifest."""
    probe = manifest.get("generation_sanity_probe") or {}
    status = probe.get("status")
    if status in ("pass", "recorded"):
        return ("`%s` -> `%s` (p=%.4f, entropy %.3f of a possible %.3f nats). %s "
                "One extra window through the same schedule; its hidden state was "
                "discarded and is not part of this dataset."
                % (probe.get("prompt"), probe.get("top1_text"),
                   probe.get("top1_probability", float("nan")),
                   probe.get("entropy_nats", float("nan")),
                   probe.get("uniform_entropy_nats", float("nan")),
                   ("Enforced against %r." % probe.get("expect")
                    if probe.get("enforced") else "Recorded, not enforced.")))
    return ("**The generation sanity probe did not run** (%s). Tensor counts, "
            "shapes and the load report can all be clean while a shard loaded as "
            "zeros; this dataset carries no evidence that the model generates "
            "sensibly." % probe.get("reason", "no reason recorded"))


def default_card_body(args, manifest, scope) -> str:
    capture = manifest["capture"]
    panel = manifest["panel"]
    rows = capture["scored_rows_total"]
    hidden_bytes = capture["total_size_bytes"]
    logit_bytes = rows * capture["vocab_size"] * 4
    lines = [
        "# %s" % args.dataset_name,
        "",
        "A **%s** fidelity dataset in **hidden form**, produced by "
        "`k6/tools/hf_capture.py` from `%s`." % (args.role, args.weights_repository or args.model),
        "",
        "## The cut",
        "",
        CUT_STATEMENT,
        "",
        "## What is here",
        "",
        "| | |",
        "|---|---|",
        "| records | %d |" % capture["records_count"],
        "| scored positions | %d |" % rows,
        "| context length | %d |" % panel["context_length"],
        "| hidden width | %d |" % capture["hidden_width"],
        "| vocab size | %d |" % capture["vocab_size"],
        "| capture bytes (hidden form) | %d |" % hidden_bytes,
        "| the same capture in logit form would be | %d bytes (%.0fx) |"
        % (logit_bytes, logit_bytes / float(hidden_bytes) if hidden_bytes else 0),
        "| scope policy | %s |" % scope["policy"],
        "| lane | %s |" % args.lane,
        "",
        "## Does the model still generate sensibly",
        "",
        _probe_card_line(manifest),
        "",
        "## How to check it",
        "",
        "```",
        "bin/fidelity-dataset verify <this directory>",
        "```",
        "",
        "Full specification: <https://github.com/malaiwah/quant-fidelity-suite/"
        "blob/main/docs/FIDELITY-DATASET-SPEC.md>",
        "",
    ]
    return "\n".join(lines)


PREVIEW_SCHEMA = "malaiwah.fidelity-dataset-preview.v1"

PREVIEW_STATEMENT = (
    "THIS IS A PRELIMINARY CAPTURE. It is backed by ONE cold run. Cross-run "
    "determinism is NOT demonstrated: a second cold capture agreeing digest-for-"
    "digest, plus the exactly-0.0 self-compare, is what would establish it, and "
    "neither has happened here. It is sealed and immutable and will NEVER be "
    "updated in place -- the complete-evidence capture is a SEPARATE dataset with "
    "a separate id, named below. Numbers measured against this dataset are true "
    "statements about THIS dataset; they do not become statements about the final "
    "one, and the comparability key makes that mechanical rather than a matter of "
    "care: reference_id is one of its seven fields."
)


def _apply_preview_identity(args, manifest) -> None:
    """Seal this capture as a PREVIEW: a different identity, not an earlier version.

    The property that must hold is that a measurement made against the preview
    can never be silently read as a measurement against the final. Two
    mechanisms, both structural:

      * IDENTITY -- the preview's `dataset.id` differs from the final's, and
        `bin/fidelity/dscompare.py` uses exactly that field as the `reference_id`
        input to `registry_lib.comparability_key`. Different reference_id means a
        different comparability group, which the renderer draws as a different
        table and the validator refuses to merge. This is the same mechanism
        `docs/DESIGNATED-REFERENCE.md` relies on, for the same reason.
      * PUBLISHABILITY -- `not_submittable: true` is a marker
        `bin/fidelity/receipt.py::_scan_for_unsubmittable` refuses at any depth,
        and the blocking disclosure below is what `emit_submission`'s SC-5 check
        refuses on. Neither is new machinery.

    Updating a published root in place is the failure this exists to prevent:
    the identity would stay the same while the CONTENT changed, so rows measured
    against the old bytes and rows measured against the new bytes would land in
    ONE comparability group and be quietly incomparable.
    """
    final_id = args.preview_of.strip()
    if not final_id:
        raise fail("--preview-of needs the final dataset's id, not an empty string")
    if final_id == args.dataset_id:
        raise fail(
            "REFUSED: --preview-of %r is the same id as --dataset-id. A preview and a "
            "final are two DATASETS, not two versions of one. Sharing the id is exactly "
            "the corruption this flag exists to prevent: `reference_id` is a "
            "comparability-key field, so rows measured against the one-run bytes and "
            "rows measured against the full-evidence bytes would land in the same group "
            "and be silently incomparable. Give the preview its own id (convention: the "
            "final id with a `.preview` suffix)." % final_id)
    manifest["not_submittable"] = True
    manifest["preview"] = {
        "schema": PREVIEW_SCHEMA,
        "preliminary": True,
        "superseded_by": final_id,
        "run_count": manifest["determinism"]["run_count"],
        "determinism_demonstrated": False,
        "updated_in_place": False,
        "immutable": True,
        "statement": PREVIEW_STATEMENT,
    }
    manifest["disclosures"].append({
        "code": "preview_capture", "severity": "blocking",
        "affects_comparability": True,
        "detail": "%s Final dataset id: %s." % (PREVIEW_STATEMENT, final_id),
    })
    log(stage="preview_identity", dataset_id=args.dataset_id, superseded_by=final_id,
        not_submittable=True)


def _assemble(args, writer, panel, panel_records, capture_records, *, context_length,
              vocab_size, hidden_size, head_rel, head_full, head_content, head_shape,
              model_dir, identity, identity_files, config, started,
              missing_weights=(), load_report=None, probe=None,
              race_report=None) -> int:
    scope = _scope(args)
    quantized = scope["policy"] != "native"

    panel_doc = dsmanifest.panel_binding(
        panel_id=args.panel_id or panel.panel_id,
        name=args.panel_name or ("token panel %s" % (args.panel_id or panel.panel_id)),
        records=panel_records, context_length=context_length,
        tokenizer=panel.tokenizer,
        repository=args.panel_repository, revision=args.panel_revision,
        panel_receipt_sha256=panel.receipt_sha256,
        scoring_window={"score_from": 0, "windowed": False,
                        "min_left_context_tokens": 1, "dropped_positions_total": 0,
                        "policy": "every causal prediction position of every window is "
                                  "scored; nothing is dropped"})

    coverage = dsmanifest.coverage_block(
        capture_records, len(panel_records),
        subset_detail=(args.subset_detail or
                       ("the first %d role=%s windows of the upstream panel, taken in "
                        "window_id order to keep the run cheap"
                        % (len(panel_records), args.panel_role))
                       if args.windows else None))

    capture_doc = dsmanifest.capture_manifest(
        run_name=args.run_name, form="hidden", semantic_point=CUT_POINT,
        tensor_key=F.TENSOR_KEY_HIDDEN, dtype="BF16", dtype_lossless=True,
        vocab_size=vocab_size, context_length=context_length, records=capture_records,
        hidden_width=hidden_size, coverage=coverage)

    head_doc = dsmanifest.head_identity(
        present=True, tensor_key="lm_head.weight", shape=head_shape, dtype="BF16",
        file_sha256=F.sha256_file(head_full), tensor_content_sha256=head_content,
        quantized=False, source="native", applied_in_capture=False, file=head_rel, bits=16,
        final_norm={"file": None, "tensor_key": None, "shape": None, "dtype": None,
                    "file_sha256": None, "tensor_content_sha256": None,
                    "applied_in_capture": True, "applied_at_replay": False},
        note="the head is shipped verbatim from the checkpoint so a third party can "
             "replay logits = hidden @ head^T without the weights")

    fingerprint = _stack_fingerprint(args.device)
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    runtime_doc = dsmanifest.capture_runtime(
        lane=args.lane,
        stack_fingerprint=fingerprint,
        stack_fingerprint_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        lane_identity_sha256=hashlib.sha256(
            ("%s|%s|%s" % (args.lane, fingerprint.get("torch_version"),
                           fingerprint.get("device_name"))).encode("utf-8")).hexdigest(),
        weights={"repository": args.weights_repository or args.model,
                 "revision": args.model_revision,
                 "model_revision": args.model_revision,
                 "checkpoint_identity_sha256": identity,
                 "checkpoint_identity_algorithm": CHECKPOINT_IDENTITY_ALGORITHM,
                 "checkpoint_files": identity_files},
        runtime_environment={"python": sys.version.split()[0],
                             "cold_run": args.cold_run},
        source_files=_source_files(args),
        capture_tool={"file": "k6/tools/hf_capture.py",
                      "sha256": F.sha256_file(os.path.abspath(__file__)),
                      "version": TOOL_VERSION, "wraps": [],
                      "schedule": args.schedule,
                      # Recorded, never inferred: a reader must be able to see
                      # that the parallel plan was emptied at load time without
                      # re-deriving it from the flags.
                      "parallel_plan_dropped": bool(getattr(args, "drop_parallel_plan", False)),
                      "layer_residency": (args.layer_residency
                                          if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER
                                          else None),
                      "mechanism": SCHEDULE_MECHANISM[args.schedule]})

    evidence = capture_doc["capture_content_digest"]
    determinism = {
        "run_count": 1,
        "cold_start_per_run": True,
        "evidence_kind": "hidden_state_tensor_sha256",
        "evidence_hashes": [evidence],
        "distinct_evidence_hash_count": 1,
        # One run cannot observe cross-run identity.  Say so instead of
        # asserting it: the SC-1 self-compare is what establishes it.
        "identical_across_runs": None,
        "repeats": [],
        "repeat_noise": None,
        "note": "a single cold capture; cross-run determinism is NOT asserted here. "
                "Capture a second cold run and run `fidelity-dataset compare "
                "--self-compare` -- an exactly 0.0 result is the SC-1 reproduction "
                "confirmation.",
    }

    disclosures = [dict(d) for d in (json.loads(args.disclosures) if args.disclosures else [])]
    if not disclosures:
        disclosures = [{"code": "no_known_deviations", "severity": "info",
                        "affects_comparability": False,
                        "detail": "captured by k6/tools/hf_capture.py"}]
    # --allow-missing-weights was used: the number in this dataset is partly a
    # measurement of randomly initialised parameters. Say so, loudly, forever.
    if missing_weights:
        disclosures.append({
            "code": "randomly_initialised_weights", "severity": "blocking",
            "affects_comparability": True,
            "detail": "%d parameter(s) were absent from the checkpoint and were randomly "
                      "initialised by transformers before this capture ran, so this is NOT "
                      "a measurement of the published artifact. First keys: %s."
                      % (len(missing_weights), ", ".join(list(missing_weights)[:6]))})

    # `unexpected_keys` are checkpoint tensors this build of transformers does
    # not place anywhere.  Reaching here at all means --allow-unexpected-tensors
    # was passed, because `refuse_on_load_report` now REFUSES otherwise: the
    # benign reading (GLM-5.3-BF16's MTP layer, `model.layers.78.*`, 791
    # tensors that `GlmMoeDsaForCausalLM` does not build) and the fatal one
    # (Qwen3.8-27B-FP8's 64 orphaned `gate_proj` block scales, M1 learning 6/7)
    # look IDENTICAL from here, and only one of them is safe to publish.  The
    # factual record stays a caveat; the override itself is blocking, exactly
    # as --allow-missing-weights is.
    unexpected = list((load_report or {}).get("unexpected_keys") or [])
    if unexpected:
        disclosures.append({
            "code": "checkpoint_tensors_not_loaded", "severity": "caveat",
            "affects_comparability": False,
            "detail": "%d checkpoint tensor(s) were present but not used by this "
                      "architecture (transformers `unexpected_keys`); they took no part in "
                      "the forward pass. First keys: %s."
                      % (len(unexpected), ", ".join(sorted(unexpected)[:6]))})
        disclosures.append({
            "code": "unexpected_tensors_overridden", "severity": "blocking",
            "affects_comparability": True,
            "detail": "--allow-unexpected-tensors was passed to capture over %d checkpoint "
                      "tensor(s) this architecture has no home for. The usual cause is a "
                      "quantization path that silently did not engage, leaving its scale "
                      "tensors orphaned and the affected modules unquantized; the benign "
                      "cause is an unused MTP/draft block. This capture does not say which. "
                      "First keys: %s."
                      % (len(unexpected), ", ".join(sorted(unexpected)[:6]))})

    # DET-D4: `verify` warns when run_count < 5 and asks for this disclosure, but
    # nothing in the tooling emitted it, so every capture this engine writes
    # would carry a warning nobody could clear.  A single cold run is a real
    # choice; declare it.
    if determinism["run_count"] < 5 and not any(
            d.get("code") == "reduced_run_count" for d in disclosures):
        disclosures.append({
            "code": "reduced_run_count", "severity": "caveat",
            "affects_comparability": False,
            "detail": "this dataset is ONE cold capture (run_count 1, the DET-D4 floor is 5). "
                      "Cross-run determinism is not asserted by the dataset; it is "
                      "established by a second cold capture plus `fidelity-dataset compare "
                      "--self-compare`, whose exactly-0.0 result is the SC-1 reproduction "
                      "confirmation."})

    # The generation sanity probe's verdict. Recorded on EVERY capture, including
    # when it was skipped and when it was disabled, because a check nobody ran is
    # not a check that passed and a reader must be able to tell the difference.
    if probe is None:
        probe = generation_probe.skipped(
            "the capture engine did not run the probe",
            prompt=getattr(args, "sanity_prompt", None),
            expect=getattr(args, "sanity_expect", None), enforced=False)
    if probe.get("status") == "skipped":
        disclosures.append({
            "code": "generation_sanity_probe_skipped", "severity": "caveat",
            "affects_comparability": False,
            "detail": "the generation sanity probe did not run (%s). Tensor counts, shapes "
                      "and the load report can all be clean while a shard loaded as zeros "
                      "or as randomly initialised weights; this capture carries no "
                      "evidence that the model generates sensibly."
                      % probe.get("reason", "no reason recorded")})

    manifest = dsmanifest.top_manifest(
        dataset={"id": args.dataset_id, "name": args.dataset_name, "role": args.role,
                 "structural_status": "sealed", "qualification": None,
                 "author": {"name": args.author, "role": "capture-author", "handle": None,
                            "url": None, "is_registry_maintainer": False},
                 "license": "mit", "repository": args.repository, "revision": None,
                 "base_capture": _base_capture(args.base_capture)},
        weights={"repository": args.weights_repository or args.model,
                 "revision": args.model_revision, "model_revision": args.model_revision,
                 "quantized": quantized, "checkpoint_identity_sha256": identity,
                 "config_sha256": F.sha256_file(os.path.join(model_dir, "config.json"))
                 if os.path.isfile(os.path.join(model_dir, "config.json")) else None,
                 "index_sha256": None, "artifact_ref": args.artifact_ref,
                 # `model_ref` is a REGISTRY model id (`model--...`), not an HF
                 # repository. A capture that has not been registered has none,
                 # and inventing one out of the repo name fails the schema.
                 "model_ref": args.model_ref,
                 "codec": args.codec, "declared_bits": args.declared_bits,
                 "declared_head_bits": 16},
        scope=scope,
        panel={"panel_id": panel_doc["panel_id"], "panel_file": "panel/panel.json",
               "panel_file_sha256": "0" * 64,
               "suite_token_hash_sha256": panel_doc["suite_token_hash_sha256"],
               "panel_token_sha256_legacy": panel_doc["panel_token_sha256_legacy"],
               "panel_receipt_sha256": panel.receipt_sha256,
               "repository": args.panel_repository, "revision": args.panel_revision,
               "contexts": len(panel_records), "context_length": context_length,
               "scored_positions_total": sum(int(r["prediction_positions"])
                                             for r in panel_records),
               "scoring_window": panel_doc["scoring_window"],
               "tokenizer": panel_doc["tokenizer"], "remap_file": None,
               "contamination": panel_doc["contamination"]},
        capture={"manifest_file": "capture/manifest.json", "manifest_file_sha256": "0" * 64,
                 "capture_content_digest": evidence, "form": "hidden",
                 "semantic_point": CUT_POINT, "tensor_key": F.TENSOR_KEY_HIDDEN,
                 "dtype": "BF16", "dtype_lossless": True, "hidden_width": hidden_size,
                 "vocab_size": vocab_size, "head_separable": True,
                 "head_not_separable_reason": None,
                 "records_count": len(capture_records),
                 "scored_rows_total": capture_doc["total_scored_rows"],
                 "total_size_bytes": capture_doc["total_size_bytes"],
                 "lossy_codec": None},
        head={"present": True, "file": head_rel, "head_json": "head/head.json",
              "tensor_key": "lm_head.weight", "compat_tensor_key": "weight",
              "shape": head_shape, "dtype": "BF16", "bias": None,
              "file_sha256": head_doc["file_sha256"], "raw_tensor_sha256": head_content,
              "tensor_content_sha256": head_content, "quantized": False, "bits": 16,
              "source": "native", "applied_in_capture": False,
              "final_norm": head_doc["final_norm"], "equality_receipt": None},
        runtime={"file": "runtime/capture-runtime.json", "file_sha256": "0" * 64,
                 "lane": args.lane, "lane_inferred": False,
                 "lane_identity_sha256": runtime_doc["lane_identity_sha256"],
                 "stack_fingerprint_sha256": runtime_doc["stack_fingerprint_sha256"],
                 "backend_identity_sha256": None, "runtime_reader_sha256": None,
                 "source": "native"},
        determinism=determinism,
        coverage=coverage,
        disclosures=disclosures)

    # Additive top-level blocks (the dataset schema is additionalProperties: true
    # and section 1.3 requires v1 readers to ignore unknown keys). They are set
    # BEFORE `writer.finish`, so the self-blanked `dataset_sha256` covers them --
    # a preview marker outside the seal would be a label anyone could strip.
    manifest["generation_sanity_probe"] = probe
    if race_report is not None:
        simulated = bool(getattr(args, "race_simulate_source", None))
        manifest["race_mode"] = {
            "schema": "malaiwah.race-mode-capture.v1",
            "enabled": True,
            "downloader": "simulated" if simulated else "huggingface_hub",
            "repository": getattr(args, "race_repo", None),
            "revision": getattr(args, "race_revision", None) or args.model_revision,
            "fetch": race_report,
            "statement": "the checkpoint was fetched WHILE this capture ran, in the order "
                         "the layer-outer schedule needed it. Scheduling only: the same "
                         "bytes went to the same converter in the same order, and the "
                         "capture_content_digest is the one a fetch-then-capture run "
                         "produces.",
        }
        if simulated:
            manifest["not_submittable"] = True
            manifest["disclosures"].append({
                "code": "simulated_fetch", "severity": "blocking",
                "affects_comparability": True,
                "detail": "--race-simulate-source was used: the 'fetch' copied from a "
                          "local directory with an injected delay. That is the offline "
                          "harness for measuring the SCHEDULE, not a fetch of a published "
                          "checkpoint. This dataset is not a measurement of anything."})
    if getattr(args, "preview_of", None):
        _apply_preview_identity(args, manifest)

    # The panel's own build receipt, byte-verbatim (spec section 2,
    # `panel/panel-receipt.json`).  Recording `panel_receipt_sha256` while
    # shipping no preimage leaves a reader holding a digest of something they
    # cannot obtain: for a ROOT dataset -- the yardstick everything else is
    # measured against -- that is the one piece of provenance most worth having.
    # It must be written BEFORE `finish()`, or `checksums.txt` will not cover it
    # and `verify` refuses the tree with `unlisted_file`.
    if panel.receipt_sha256:
        receipt_src = os.path.join(panel.root, "panel.receipt.json")
        raw_receipt = open(receipt_src, "rb").read()
        writer.add_file("panel/panel-receipt.json", raw_receipt)
        manifest["panel"]["panel_receipt_file"] = "panel/panel-receipt.json"
        log(stage="panel_receipt", file="panel/panel-receipt.json",
            sha256=panel.receipt_sha256, bytes=len(raw_receipt))

    # README.md is REQUIRED by the spec, is covered by checksums.txt, and so has
    # to exist BEFORE the seal.  A capture that does not write one produces a
    # tree that `verify` refuses -- which is exactly what happened the first
    # time this engine ran.
    body = (open(args.readme, "r", encoding="utf-8").read() if args.readme
            else default_card_body(args, manifest, scope))
    writer.add_readme(render_card(args, manifest, body))

    report = dsvalidate.Report(args.out)
    report.ok("pre-seal")
    manifest = writer.finish(manifest, panel_doc, capture_doc, head_doc, runtime_doc,
                             validation_report=report.to_dict())

    logit_bytes = capture_doc["total_scored_rows"] * vocab_size * 4
    hidden_bytes = capture_doc["total_size_bytes"]
    log(stage="sealed", out=args.out, dataset_sha256=manifest[F.SEAL_FIELD],
        capture_content_digest=evidence, records=len(capture_records),
        scored_rows=capture_doc["total_scored_rows"],
        hidden_form_bytes=hidden_bytes, logit_form_bytes_equivalent=logit_bytes,
        storage_ratio=round(logit_bytes / float(hidden_bytes), 1) if hidden_bytes else None,
        elapsed_seconds=round(time.monotonic() - started, 1))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hf_capture", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="HF repo id or a local checkpoint directory")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--panel", required=True, help="directory holding panel.json + arrays/")
    parser.add_argument("--panel-role", default="final")
    parser.add_argument("--panel-id", default=None)
    parser.add_argument("--panel-name", default=None)
    parser.add_argument("--panel-repository", default=None)
    parser.add_argument("--panel-revision", default=None)
    parser.add_argument("--windows", type=int, default=None,
                        help="cap the window count (a subset is disclosed in coverage)")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--progress-seconds", type=float, default=progress_meter.interval_from_env(),
        help="how often to print a progress line when stdout is a FILE (default 30; "
             "0 disables). On a TTY the meter updates in place instead and this is "
             "ignored. Env override: FIDELITY_PROGRESS_SECONDS.")
    parser.add_argument("--out", required=True, help="dataset root to WRITE (it is created)")
    parser.add_argument("--role", choices=["root", "quant", "derived"], required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--schedule", default=layer_outer.SCHEDULE_WINDOW_OUTER,
                        choices=[layer_outer.SCHEDULE_WINDOW_OUTER,
                                 layer_outer.SCHEDULE_LAYER_OUTER],
                        help="window-outer (default) loads the model and pushes one window "
                             "at a time through the whole stack -- for a model that does "
                             "not fit, that pays for the weights once PER WINDOW. "
                             "layer-outer inverts the loop: for each layer { load it once; "
                             "for each window: push that window through it; free it }, so "
                             "the checkpoint tree is read once per RUN. Windows are still "
                             "pushed sequentially and the per-window arithmetic is "
                             "bit-identical; see docs/GLM53-LAYER-OUTER.md for the proofs.")
    parser.add_argument("--layer-residency", default=layer_outer.RESIDENCY_STREAM,
                        choices=[layer_outer.RESIDENCY_STREAM, layer_outer.RESIDENCY_RESIDENT],
                        help="layer-outer only. `stream` (default) builds the model on the "
                             "meta device and materialises one layer at a time -- the point "
                             "of the schedule. `resident` reorders the loop over a fully "
                             "loaded model; it saves nothing and exists so a digest "
                             "mismatch can be attributed to the loop or to the loader.")
    parser.add_argument("--memory-report", default=None,
                        help="write measured peak memory for this run to this JSON path. "
                             "It is written OUTSIDE the dataset so it cannot disturb the "
                             "seal; the same figures are also logged as stage=peak_memory.")
    parser.add_argument("--device-map", default=None,
                        help="transformers device_map ('auto', 'balanced', a JSON object, "
                             "...). When set, the model is DISPATCHED by accelerate and the "
                             "post-load `.to(--device)` is skipped -- the only way to load a "
                             "checkpoint larger than one device's memory.")
    parser.add_argument("--drop-parallel-plan", action="store_true",
                        help="empty the config's base_model_tp_plan / base_model_ep_plan "
                             "before loading. Those plans are read only when the model is "
                             "split across ranks, which this tool never does, so this is "
                             "inert for the forward pass -- it exists because transformers "
                             "5.16.1's FP8 quantizer CRASHES rewriting a non-empty plan "
                             "(every deepseek_v4 FP8 repo). Recorded in the receipt.")
    parser.add_argument("--max-memory", default=None,
                        help="JSON object passed to from_pretrained as max_memory, e.g. "
                             '\'{"0": "130GiB", "cpu": "250GiB"}\'')
    parser.add_argument("--offload-folder", default=None,
                        help="accelerate disk-offload directory. NOTE: this writes a SECOND "
                             "full copy of the weights.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--run-name", default="hf-capture")
    parser.add_argument("--cold-run", default=None,
                        help="a label for THIS process, recorded in every tensor's metadata")
    parser.add_argument("--author", default="malaiwah")
    parser.add_argument("--repository", default=None)
    parser.add_argument("--base-capture", default=None)
    parser.add_argument("--weights-repository", default=None)
    parser.add_argument("--artifact-ref", default=None,
                        help="registry artifact id (artifact--...), if one exists")
    parser.add_argument("--model-ref", default=None,
                        help="registry model id (model--...), if one exists")
    parser.add_argument("--codec", default=None)
    parser.add_argument("--declared-bits", type=float, default=None)
    parser.add_argument("--scope-file", default=None,
                        help="JSON {policy, head_policy, kv_cache_dtype, assignments[]} -- "
                             "REQUIRED for a quantized candidate: the scope is the honest "
                             "description of what was changed")
    parser.add_argument("--subset-detail", default=None)
    parser.add_argument("--disclosures", default=None, help="JSON list")
    parser.add_argument("--readme", default=None)
    parser.add_argument("--allow-missing-weights", action="store_true",
                        help="capture even though transformers had to randomly initialise "
                             "parameters the checkpoint did not provide. Forces a BLOCKING "
                             "disclosure. Almost never the right flag.")
    parser.add_argument("--allow-unexpected-tensors", action="store_true",
                        help="capture even though the checkpoint carried tensors this "
                             "architecture has no home for. Usually means a quantization "
                             "path silently did not engage; sometimes means a legitimately "
                             "unused MTP/draft block. Forces a BLOCKING disclosure.")
    parser.add_argument("--force", action="store_true")

    race = parser.add_argument_group(
        "race mode -- overlap the fetch with the capture (k6/tools/race_fetch.py)")
    race.add_argument("--race-repo", default=None,
                      help="fetch this HF repo IN THE BACKGROUND, in the order the "
                           "layer-outer schedule needs it, while the capture runs. "
                           "--model must be a directory already holding config.json, "
                           "the tokenizer and model.safetensors.index.json; the shards "
                           "land into it. Changes WHEN bytes arrive, never which.")
    race.add_argument("--race-revision", default=None,
                      help="the pinned revision to fetch (defaults to --model-revision)")
    race.add_argument("--race-workers", type=int, default=8,
                      help="parallel downloads (default 8). Ordering is by priority, so "
                           "raising this does not reorder the queue, it widens it.")
    race.add_argument("--race-timeout-seconds", type=float, default=7200.0,
                      help="how long the capture will block for a layer's shards before "
                           "REFUSING (default 7200). A timeout is never a proceed.")
    race.add_argument("--race-layer-key-regex",
                      default=race_fetch.DEFAULT_LAYER_KEY_REGEX,
                      help="how a checkpoint key names its decoder layer index. The "
                           "default matches model.layers.N., language_model.model."
                           "layers.N. and a bare layers.N.")
    race.add_argument("--race-report", default=None,
                      help="write the measured fetch/overlap report to this JSON path")
    race.add_argument("--race-simulate-source", default=None, metavar="DIR",
                      help="TEST HARNESS ONLY: fetch from this local directory instead "
                           "of the Hub, so the SCHEDULE can be measured offline and as a "
                           "controlled A/B. Stamps a BLOCKING disclosure -- a capture "
                           "made this way is not a measurement of anything.")
    race.add_argument("--race-simulate-seconds", type=float, default=0.0,
                      help="TEST HARNESS ONLY: per-file delay for --race-simulate-source")

    sanity = parser.add_argument_group(
        "generation sanity probe (k6/tools/generation_probe.py) -- on by default")
    sanity.add_argument("--sanity-prompt", default=generation_probe.DEFAULT_PROMPT,
                        help="the prompt whose next token is checked. One extra window "
                             "through the schedule already running: ~1/N of an N-window "
                             "capture, and no extra weight loading at all.")
    sanity.add_argument("--sanity-expect", default=None,
                        help="the expected continuation, e.g. Paris. Given, the probe is "
                             "FAIL-CLOSED on it; omitted, the probe still runs and is "
                             "still recorded, and only a degenerate (all-logits-equal) "
                             "distribution refuses.")
    preview = parser.add_argument_group(
        "preview identity (docs/RACE-MODE.md) -- a preview is a DIFFERENT dataset, "
        "never an earlier version of the final one")
    preview.add_argument("--preview-of", default=None, metavar="FINAL_DATASET_ID",
                         help="seal this capture as a PRELIMINARY dataset superseded by "
                              "FINAL_DATASET_ID. It gets its own dataset id (which is the "
                              "reference_id half of every comparability key computed "
                              "against it), a blocking `preview_capture` disclosure, and "
                              "not_submittable: true. It is sealed and immutable like any "
                              "other capture and is NEVER updated in place.")
    sanity.add_argument("--no-sanity-check", dest="sanity_check", action="store_false",
                        default=True,
                        help="do not run the probe at all. The manifest then records that "
                             "it was disabled, because a check nobody ran is not a check "
                             "that passed.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device_map and args.device_map.strip().startswith("{"):
        args.device_map = json.loads(args.device_map)
    if args.cold_run is None:
        args.cold_run = "%s-%d" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), os.getpid())
    if os.path.exists(args.out):
        if not args.force:
            print("hf_capture: REFUSED: %s exists (pass --force)" % args.out, file=sys.stderr)
            return 3
        import shutil

        shutil.rmtree(args.out)
    if args.schedule == layer_outer.SCHEDULE_LAYER_OUTER and args.device_map:
        # These are two different answers to the same question and they fight.
        # `--device-map` hands the model to `accelerate`, which attaches
        # `AlignDevicesHook`s that move weights around per module call; the
        # layer-outer streamer owns residency itself and would be racing those
        # hooks for the same parameters. They are also not complementary: the
        # reason `--device-map` exists (docs/GLM53-ROOT-FEASIBILITY.md R2) is
        # that the window-outer loop cannot fit the model, and layer-outer
        # removes that need on a single device. Composing them is future work,
        # not a silently-broken flag combination.
        print("hf_capture: REFUSED: --schedule layer-outer with --device-map. The "
              "layer-outer streamer manages residency itself (meta model, one layer "
              "materialised at a time); accelerate's dispatch hooks move the same "
              "parameters on their own schedule and the two would fight over them. "
              "Drop --device-map: on a single device layer-outer is what --device-map "
              "was working around. Multi-device layer-outer is not implemented.",
              file=sys.stderr)
        return 3
    if args.role != "root" and not args.scope_file:
        print("hf_capture: REFUSED: --role %s without --scope-file. A candidate capture that "
              "does not describe what was quantized is unreadable evidence." % args.role,
              file=sys.stderr)
        return 3
    try:
        return run_capture(args)
    except SystemExit as exc:
        return int(exc.code or 1)


if __name__ == "__main__":
    sys.exit(main())
