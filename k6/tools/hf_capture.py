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

TOOL_VERSION = "hf_capture/1"

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


def load_model(model_dir: str, device: str, dtype_name: str,
               device_map: Any = None, max_memory: Optional[Dict[Any, str]] = None,
               offload_folder: Optional[str] = None):
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
            errors.append("%s: %s: %s" % (name, type(exc).__name__, exc))
    if model is None:
        try:
            model, info = _from_pretrained(transformers.AutoModelForCausalLM,
                                           model_dir, torch_dtype, **extra)
        except Exception as exc:
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


def refuse_on_load_report(report: Dict[str, Any], allow_missing: bool) -> List[str]:
    """CAPTURE-03. Returns the reinitialised keys when the caller forced through.

    Four independent ways a `transformers` load can hand back a model whose
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

    1 and 2 are overridable by `--allow-missing-weights`, which forces a
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

    identity, identity_files = checkpoint_identity(model_dir)
    log(stage="checkpoint_identity", sha256=identity, files=len(identity_files))

    max_memory = json.loads(args.max_memory) if args.max_memory else None
    model, config, loading_info = load_model(
        model_dir, args.device, args.dtype, device_map=args.device_map,
        max_memory=max_memory, offload_folder=args.offload_folder)

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
    missing = refuse_on_load_report(report, args.allow_missing_weights)
    if missing:
        log(stage="missing_weights", count=len(missing), keys=missing[:12])

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

            del tap[:]
            input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
            attention_mask = torch.from_numpy(
                np.asarray(mask, dtype=np.int64).reshape(1, -1)).to(args.device)
            elapsed = time.monotonic()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            elapsed = time.monotonic() - elapsed
            if len(tap) != 1:
                raise fail("window %s: the head hook fired %d times, expected exactly 1 -- an "
                           "extra forward would misalign the capture"
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
            del tap[:]
    finally:
        handle.remove()

    # -- head ---------------------------------------------------------------
    head_weight = head.weight
    head_rel = writer.add_head_payload(
        st_bytes("lm_head.weight", "BF16", list(head_weight.shape), _bf16_raw(head_weight)))
    head_full = os.path.join(args.out, head_rel)
    head_content = F.tensor_content_sha256(head_full, "lm_head.weight")
    log(stage="head", file=head_rel, tensor_content_sha256=head_content,
        bytes=os.path.getsize(head_full))

    return _assemble(args, writer, panel, panel_records, capture_records,
                     context_length=context_length, vocab_size=vocab_size,
                     hidden_size=hidden_size, head_rel=head_rel, head_full=head_full,
                     head_content=head_content, head_shape=list(head_weight.shape),
                     model_dir=model_dir, identity=identity, identity_files=identity_files,
                     config=config, started=started, missing_weights=missing,
                     load_report=report)


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



def render_card(args, manifest, body: str) -> str:
    """A dataset card whose frontmatter carries the required x_fidelity block."""
    sys.path.insert(0, os.path.join(REPO, "bin"))
    from fidelity import cardmeta

    front = {
        "license": "mit",
        "tags": ["fidelity", "kl-divergence", "fidelity-provenance",
                 "fidelity-dataset", "fidelity-%s" % args.role],
        "x_fidelity": cardmeta.build_dataset_x_fidelity(
            None, repository=args.repository, manifest=manifest),
    }
    front, _ = cardmeta.split_card(cardmeta.render_card(front, body))
    return cardmeta.render_card(front, body)


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


def _assemble(args, writer, panel, panel_records, capture_records, *, context_length,
              vocab_size, hidden_size, head_rel, head_full, head_content, head_shape,
              model_dir, identity, identity_files, config, started,
              missing_weights=(), load_report=None) -> int:
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
        source_files={"k6/tools/hf_capture.py": F.sha256_file(os.path.abspath(__file__))},
        capture_tool={"file": "k6/tools/hf_capture.py",
                      "sha256": F.sha256_file(os.path.abspath(__file__)),
                      "version": TOOL_VERSION, "wraps": [],
                      "mechanism": "transformers forward pass; forward pre-hook on "
                                   "model.get_output_embeddings()"})

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
    # not place anywhere.  They are often legitimate -- GLM-5.3-BF16 ships an
    # MTP layer (`model.layers.78.*`, 791 tensors) that `GlmMoeDsaForCausalLM`
    # does not build -- so this is a DISCLOSURE, not a refusal.  But a reader
    # comparing two datasets deserves to know that some of the checkpoint was
    # never used, and until now nothing recorded it at all.
    unexpected = list((load_report or {}).get("unexpected_keys") or [])
    if unexpected:
        disclosures.append({
            "code": "checkpoint_tensors_not_loaded", "severity": "caveat",
            "affects_comparability": False,
            "detail": "%d checkpoint tensor(s) were present but not used by this "
                      "architecture (transformers `unexpected_keys`); they took no part in "
                      "the forward pass. First keys: %s."
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
    parser.add_argument("--out", required=True, help="dataset root to WRITE (it is created)")
    parser.add_argument("--role", choices=["root", "quant", "derived"], required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--device-map", default=None,
                        help="transformers device_map ('auto', 'balanced', a JSON object, "
                             "...). When set, the model is DISPATCHED by accelerate and the "
                             "post-load `.to(--device)` is skipped -- the only way to load a "
                             "checkpoint larger than one device's memory.")
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
    parser.add_argument("--force", action="store_true")
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
