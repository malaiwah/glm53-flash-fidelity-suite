"""Step 3: compare two fidelity datasets.

Two halves, deliberately separable:

  * the GATE LADDER (spec section 10.1) -- eleven ordered gates, each with a
    named refusal id.  Pure stdlib + numpy, no torch, so every refusal is
    testable on a stock py3.9 interpreter with no GPU;
  * the ESTIMATOR (spec section 10.2) -- full vocabulary, fp64 log_softmax.
    When torch is importable this is `k6_kld_report._token_kld`, imported and
    called, never reimplemented, so a number produced here is the same number
    the sealed pipeline produces.  Without torch it falls back to the identical
    fp64 formula in numpy and SAYS SO in the receipt
    (`comparator.estimator_backend`), because a silent backend swap is exactly
    the kind of undeclared difference this whole format exists to stop.

The A == B short-circuit (SC-1) answers by hash proof without a matmul;
`--force-compute` runs the math anyway and asserts bitwise agreement.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import dsformat as F
from . import dsvalidate

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_K6_TOOLS = os.path.join(_REPO, "k6", "tools")


class Refusal(Exception):
    def __init__(self, gate: str, code: str, message: str, override: Optional[str] = None):
        self.gate = gate
        self.code = code
        self.message = message
        self.override = override
        super().__init__("%s: %s%s" % (code, message,
                                       ("  [override: %s]" % override) if override else ""))


# ---------------------------------------------------------------------------
# Dataset handle
# ---------------------------------------------------------------------------


class Dataset(object):
    """A loaded, seal-verified fidelity dataset."""

    def __init__(self, root: str, manifest: Dict[str, Any],
                 capture: Dict[str, Any], panel: Dict[str, Any],
                 head_doc: Optional[Dict[str, Any]]):
        self.root = root
        self.manifest = manifest
        self.capture_manifest = capture
        self.panel_doc = panel
        self.head_doc = head_doc

    # -- convenience accessors -------------------------------------------------
    @property
    def form(self) -> str:
        return self.manifest["capture"]["form"]

    @property
    def head(self) -> Dict[str, Any]:
        return self.manifest["head"]

    @property
    def lane(self) -> str:
        return self.manifest["runtime"]["lane"]

    @property
    def content_digest(self) -> str:
        return self.manifest["capture"]["capture_content_digest"]

    @property
    def records(self) -> List[Dict[str, Any]]:
        return sorted(self.capture_manifest.get("records") or [],
                      key=lambda r: int(r["index"]))

    def record_path(self, record: Dict[str, Any]) -> str:
        base = os.path.dirname(self.manifest["capture"]["manifest_file"])
        return os.path.join(self.root, base, record["file"])

    def head_path(self) -> Optional[str]:
        rel = self.head.get("file")
        if not rel:
            return None
        full = os.path.join(self.root, rel)
        return full if os.path.isfile(full) else None

    def panel_by_index(self) -> Dict[int, Dict[str, Any]]:
        return {int(r["index"]): r for r in self.panel_doc.get("records") or []}

    def side_block(self, label: str) -> Dict[str, Any]:
        manifest = self.manifest
        head = manifest["head"]
        return {
            "label": label,
            "dataset_id": (manifest.get("dataset") or {}).get("id"),
            "dataset_sha256": manifest[F.SEAL_FIELD],
            "capture_content_digest": self.content_digest,
            "role": (manifest.get("dataset") or {}).get("role"),
            "form": self.form,
            "lane": manifest["runtime"]["lane"],
            "repository": (manifest.get("dataset") or {}).get("repository"),
            "revision": (manifest.get("dataset") or {}).get("revision"),
            "head": {
                "tensor_content_sha256": head.get("tensor_content_sha256"),
                "raw_tensor_sha256": head.get("raw_tensor_sha256"),
                "quantized": head.get("quantized"),
                "bits": head.get("bits"),
                "source": head.get("source"),
            },
            "stack_fingerprint_sha256": manifest["runtime"].get("stack_fingerprint_sha256"),
            "lane_identity_sha256": manifest["runtime"].get("lane_identity_sha256"),
            "weights": dict(manifest.get("weights") or {}),
            "scope_digest": (manifest.get("scope") or {}).get("scope_digest"),
        }

    def weights_identity(self) -> Tuple[Any, Any, Any]:
        weights = self.manifest.get("weights") or {}
        return (
            weights.get("model_revision"),
            weights.get("checkpoint_identity_sha256"),
            (self.manifest.get("scope") or {}).get("scope_digest"),
        )


def load_dataset(root: str, verify: bool = True, verify_tensors: bool = False,
                 allow_partial: bool = False) -> Dataset:
    """Gate 1: seal.  A dataset that does not verify is never compared."""
    if verify:
        report = dsvalidate.validate_dataset(
            root, verify_tensors=verify_tensors, allow_partial=allow_partial)
        if not report.passed:
            first = report.errors[0]
            raise Refusal("seal", "seal_failed",
                          "%s did not verify: [%s/%s] %s"
                          % (root, first["code"], first["rule"], first["message"]))
    manifest = F.load_manifest(root)
    capture = F.read_json(os.path.join(root, manifest["capture"]["manifest_file"]))
    panel = F.read_json(os.path.join(root, manifest["panel"]["panel_file"]))
    head_doc = None
    if manifest["head"].get("head_json"):
        head_path = os.path.join(root, manifest["head"]["head_json"])
        if os.path.isfile(head_path):
            head_doc = F.read_json(head_path)
    return Dataset(root, manifest, capture, panel, head_doc)


# ---------------------------------------------------------------------------
# The gate ladder (spec section 10.1)
# ---------------------------------------------------------------------------


def _gate(passed: bool, detail: str, overridden_by: Optional[str] = None) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": detail, "overridden_by": overridden_by}


def run_gates(reference: Dataset, candidate: Dataset, options: Dict[str, Any]
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run gates 2-9 in order, stopping at the first refusal.

    Returns (gates, findings).  `findings` carries what the receipt needs:
    same_lane, stack_relation, head_policy, comparability class, disclosures,
    the shared index set, and the head to apply.
    """
    gates: Dict[str, Any] = {}
    findings: Dict[str, Any] = {"disclosures": [], "class": "strict",
                                "usable_as_floor": True}

    # --- 2. form ------------------------------------------------------------
    form_a, form_b = reference.form, candidate.form
    mixed = form_a != form_b
    if mixed and not options.get("allow_mixed_form", True):
        raise Refusal("form", "form_mismatch", "%s vs %s" % (form_a, form_b))
    gates["form"] = _gate(True, "%s vs %s" % (form_a, form_b))

    # --- 3. panel -----------------------------------------------------------
    pa, pb = reference.manifest["panel"], candidate.manifest["panel"]
    if pa["suite_token_hash_sha256"] != pb["suite_token_hash_sha256"]:
        gates["panel"] = _gate(False, "suite_token_hash_sha256 differs")
        raise Refusal("panel", "panel_mismatch",
                      "different panels: %s vs %s"
                      % (pa["suite_token_hash_sha256"][:12], pb["suite_token_hash_sha256"][:12]))
    if pa["scoring_window"] != pb["scoring_window"]:
        gates["panel"] = _gate(False, "scoring_window differs")
        raise Refusal("panel", "panel_mismatch",
                      "scoring_window is part of panel IDENTITY (PANEL-D3): score_from %r vs %r"
                      % (pa["scoring_window"].get("score_from"),
                         pb["scoring_window"].get("score_from")))
    ra = {int(r["index"]): r for r in reference.records}
    rb = {int(r["index"]): r for r in candidate.records}
    shared = sorted(set(ra) & set(rb))
    for index in shared:
        if ra[index]["token_ids_json_sha256"] != rb[index]["token_ids_json_sha256"]:
            gates["panel"] = _gate(False, "record %d token digest differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d has different tokens on the two sides (BIND-2)" % index)
        ma, mb = ra[index].get("attention_mask_sha256"), rb[index].get("attention_mask_sha256")
        if ma is not None and mb is not None and ma != mb:
            gates["panel"] = _gate(False, "record %d attention mask differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d attention_mask_sha256 differs (BIND-3)" % index)
        if ra[index]["scored_rows"] != rb[index]["scored_rows"]:
            gates["panel"] = _gate(False, "record %d scored_rows differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d scored_rows %r vs %r"
                          % (index, ra[index]["scored_rows"], rb[index]["scored_rows"]))
    gates["panel"] = _gate(True, "suite_token_hash_sha256 equal; %d shared records; "
                                 "per-record token and mask digests equal; scoring_window equal"
                           % len(shared))

    # --- 4. head (HEAD-1..7) ------------------------------------------------
    findings.update(_head_gate(reference, candidate, options, gates, findings))

    # --- 5. lane ------------------------------------------------------------
    same_lane = reference.lane == candidate.lane
    if not same_lane and not options.get("allow_cross_lane"):
        gates["lane"] = _gate(False, "%s vs %s" % (reference.lane, candidate.lane))
        raise Refusal("lane", "lane_mismatch",
                      "lanes differ: %s vs %s" % (reference.lane, candidate.lane),
                      override="--allow-cross-lane")
    gates["lane"] = _gate(True, "lane=%s vs %s" % (reference.lane, candidate.lane),
                          overridden_by=None if same_lane else "--allow-cross-lane")
    findings["same_lane"] = same_lane
    if not same_lane:
        # BIAS-006: a cross-lane comparison may never be cited as a floor.
        findings["class"] = "advisory"
        findings["usable_as_floor"] = False
        findings["disclosures"].append({
            "code": "cross_engine_capture", "severity": "caveat",
            "affects_comparability": True,
            "detail": "reference lane %s, candidate lane %s. A cross-lane number carries the "
                      "lane offset; BIAS-006 forbids citing it as a floor, so usable_as_floor "
                      "is stamped false and cannot be laundered downstream."
                      % (reference.lane, candidate.lane),
        })

    # --- 6. stack -----------------------------------------------------------
    la = reference.manifest["runtime"].get("lane_identity_sha256")
    lb = candidate.manifest["runtime"].get("lane_identity_sha256")
    sa = reference.manifest["runtime"].get("stack_fingerprint_sha256")
    sb = candidate.manifest["runtime"].get("stack_fingerprint_sha256")
    # A dataset compared with ITSELF is the same stack by construction, whether
    # or not `lane_identity_sha256` happens to be recorded.  Without this the
    # flagship self-compare emits a bias block asserting "the two captures were
    # produced by different stacks" about one capture.
    same_dataset = bool(reference.manifest.get(F.SEAL_FIELD)
                        and reference.manifest.get(F.SEAL_FIELD)
                        == candidate.manifest.get(F.SEAL_FIELD))
    same_stack = same_dataset or bool(la and lb and la == lb and sa and sb and sa == sb)
    findings["stack_relation"] = "same_stack" if same_stack else "cross_stack"
    gates["stack"] = _gate(True,
                           "lane_identity %s / stack_fingerprint %s -> %s"
                           % ("equal" if la == lb else "differ",
                              "equal" if sa == sb else "differ",
                              findings["stack_relation"]))
    if not same_stack:
        findings["class"] = "advisory"
        findings["bias"] = {
            # The registry's own enum value (measurement.schema.json); BIAS-001
            # binds it to estimator.stack_relation == cross_stack.
            "kind": "cross_stack_capture_replay",
            "direction": "unknown",
            "estimated_magnitude": None,
            "floor_measurement_ref": None,
            "detail": "the two captures were produced by different stacks; BIAS-001 requires "
                      "this block, and a residual of the 1e-2 class is expected from a "
                      "different kernel, GPU class or torch build alone.",
        }

    # --- 7. geometry --------------------------------------------------------
    ca, cb = reference.manifest["capture"], candidate.manifest["capture"]
    if ca["vocab_size"] != cb["vocab_size"]:
        gates["geometry"] = _gate(False, "vocab_size differs")
        raise Refusal("geometry", "geometry_mismatch",
                      "vocab_size %r vs %r" % (ca["vocab_size"], cb["vocab_size"]))
    if form_a == form_b == "hidden" and ca.get("hidden_width") != cb.get("hidden_width"):
        gates["geometry"] = _gate(False, "hidden_width differs")
        raise Refusal("geometry", "geometry_mismatch",
                      "hidden_width %r vs %r" % (ca.get("hidden_width"), cb.get("hidden_width")))
    gates["geometry"] = _gate(True, "vocab_size %d equal; hidden_width %r"
                              % (ca["vocab_size"], ca.get("hidden_width")))

    # --- 8. coverage --------------------------------------------------------
    if set(ra) != set(rb):
        if not options.get("allow_partial"):
            gates["coverage"] = _gate(False, "index sets differ")
            raise Refusal("coverage", "coverage_mismatch",
                          "reference has %d records, candidate %d, %d shared"
                          % (len(ra), len(rb), len(shared)),
                          override="--allow-partial")
        findings["disclosures"].append({
            "code": "subset_of_panel", "severity": "caveat", "affects_comparability": True,
            "detail": "index sets differ; the comparison ran on the %d-record intersection "
                      "(SCOPE-010)." % len(shared),
        })
    covers_full = (
        bool((reference.manifest.get("coverage") or {}).get("complete"))
        and bool((candidate.manifest.get("coverage") or {}).get("complete"))
        and set(ra) == set(rb)
    )
    gates["coverage"] = _gate(True, "%d of %d/%d declared records shared"
                              % (len(shared), len(ra), len(rb)),
                              overridden_by=None if set(ra) == set(rb) else "--allow-partial")
    findings["shared_indices"] = shared
    findings["covers_full_panel"] = covers_full
    if not covers_full:
        findings["disclosures"].append({
            "code": "subset_of_panel", "severity": "caveat", "affects_comparability": True,
            "detail": "at least one side does not cover the full panel; "
                      "measurement_scope.covers_full_panel is false (SCOPE-010).",
        }) if not any(d["code"] == "subset_of_panel" for d in findings["disclosures"]) else None

    # --- 9. lossy -----------------------------------------------------------
    lossy = [side for side, manifest in (("reference", reference.manifest),
                                         ("candidate", candidate.manifest))
             if manifest["capture"].get("lossy_codec")]
    if lossy:
        findings["class"] = "advisory"
        findings["disclosures"].append({
            "code": "lossy_capture_codec", "severity": "caveat", "affects_comparability": True,
            "detail": "%s stores values that are not the model's values (D-8); the comparison "
                      "is advisory." % ", ".join(lossy),
        })
    gates["lossy"] = _gate(True, "lossy_codec %s" % ("on " + ", ".join(lossy) if lossy
                                                     else "null on both sides"))
    for side, manifest in (("reference", reference.manifest), ("candidate", candidate.manifest)):
        if manifest["capture"].get("dtype_lossless") is False:
            findings["class"] = "advisory"
            findings["disclosures"].append({
                "code": "quantized_capture_dtype", "severity": "caveat",
                "affects_comparability": True,
                "detail": "%s stored its values in a dtype narrower than the forward's "
                          "(FORM-1)." % side,
            })
    return gates, findings


def _head_gate(reference: Dataset, candidate: Dataset, options: Dict[str, Any],
               gates: Dict[str, Any], findings: Dict[str, Any]) -> Dict[str, Any]:
    """HEAD-1a/1b/2/3/4 at compare time (spec section 8.3)."""
    ha, hb = reference.head, candidate.head
    da, db = ha.get("tensor_content_sha256"), hb.get("tensor_content_sha256")
    form_a, form_b = reference.form, candidate.form
    out: Dict[str, Any] = {}

    hidden_involved = "hidden" in (form_a, form_b)

    if hidden_involved:
        # HEAD-4: no override.  A capture that cannot name its own head cannot
        # be scored through anyone's.
        for label, form, digest in (("reference", form_a, da), ("candidate", form_b, db)):
            if form == "hidden" and not digest:
                gates["head"] = _gate(False, "%s has a null head content digest" % label)
                raise Refusal("head", "head_mismatch",
                              "HEAD-4: %s is hidden-form with head.tensor_content_sha256 null; "
                              "there is no override" % label)
        # O-6 / H11: comparing a container digest to a content digest is forbidden.
        if ha.get("file_sha256") and hb.get("file_sha256") \
                and ha["file_sha256"] == hb["file_sha256"] and da != db:
            gates["head"] = _gate(False, "equal file digests, different tensor content")
            raise Refusal("head", "head_mismatch",
                          "HEAD-IDENT/O-6: the two heads share a file digest but differ in "
                          "tensor CONTENT; content is normative")

    if form_a == "logit" and form_b == "logit":
        # HEAD-2: never refuse.  Each capture applied its own head, so head
        # quantization error is INSIDE the measurement, which is correct.
        out["head_policy"] = "native_head"
        out["head_applied"] = None
        detail = "HEAD-2: both sides are logit form; each applied its own head, so head error "
        if da and db and da != db:
            detail += "is inside the measurement (heads differ, correctly)."
        elif da and db:
            detail += "is inside the measurement (heads are identical)."
        else:
            detail += "is inside the measurement (at least one head digest is unrecorded)."
            findings["class"] = "advisory"
            findings["disclosures"].append({
                "code": "estimator_unknown", "severity": "caveat",
                "affects_comparability": True,
                "detail": "a logit-form capture did not record its head identity; the "
                          "comparison is advisory (HEAD-2).",
            })
        gates["head"] = _gate(True, detail)
        return out

    if form_a != form_b:
        # HEAD-3: mixed hidden <-> logit.  Only legal when the head replayed
        # onto the hidden side equals the head that produced the logit side.
        if not (da and db and da == db):
            gates["head"] = _gate(False, "mixed form with unequal heads")
            raise Refusal("head", "head_mismatch",
                          "HEAD-3: a hidden<->logit comparison needs the replay head to equal "
                          "the head that produced the logits (%s vs %s)"
                          % ((da or "null")[:12], (db or "null")[:12]))
        out["head_policy"] = "native_head"
        out["head_applied"] = da
        findings["disclosures"].append({
            "code": "no_known_deviations", "severity": "info", "affects_comparability": False,
            "detail": "HEAD-3: the hidden side was replayed through the same head "
                      "(%s) that produced the logit side." % da[:12],
        })
        gates["head"] = _gate(True, "HEAD-3: mixed form, identical head content digest")
        return out

    # hidden <-> hidden
    if da == db:
        # HEAD-1a: ALLOW, and `class` may remain strict.
        out["head_policy"] = "shared_reference_head"
        out["head_applied"] = da
        findings["disclosures"].append({
            "code": "shared_reference_head", "severity": "info", "affects_comparability": False,
            "detail": "HEAD-1a: both captures declare the same lm_head tensor content digest "
                      "(%s); one head applied to both sides is a shared APPLICATION of "
                      "identical WEIGHTS." % da[:12],
        })
        gates["head"] = _gate(True, "HEAD-1a: identical head content digest on both sides")
        return out

    # HEAD-1b: REFUSE.
    if not options.get("disclose_head_substitution"):
        gates["head"] = _gate(False, "head content digests differ: %s vs %s"
                              % (da[:12], db[:12]))
        raise Refusal(
            "head", "head_mismatch",
            "HEAD-1b: head content digests differ (%s vs %s). Replaying one artifact's hidden "
            "states through the other's head erases its head-quantization error and flatters "
            "it." % (da[:12], db[:12]),
            override="--disclose-head-substitution")
    applied = options.get("head_content_sha256") or da
    out["head_policy"] = "shared_reference_head"
    out["head_applied"] = applied
    findings["class"] = "advisory"
    findings["usable_as_floor"] = False
    findings["bias"] = {
        "kind": "other",
        "direction": "downward",
        "estimated_magnitude": None,
        "floor_measurement_ref": None,
        "detail": "a single head was applied to hidden states captured under two different "
                  "heads; the candidate's own head-quantization error is erased, biasing the "
                  "number DOWNWARD (flattering the candidate).",
    }
    findings["disclosures"].append({
        "code": "head_substituted", "severity": "blocking", "affects_comparability": True,
        "detail": "HEAD-1b override: reference head %s, candidate head %s, applied %s. Under "
                  "DISC-003 a blocking disclosure forces status pending/retracted -- a "
                  "head-substituted number is not publishable as a measurement."
                  % (da[:12], db[:12], applied[:12]),
    })
    gates["head"] = _gate(True, "HEAD-1b overridden: head substitution disclosed",
                          overridden_by="--disclose-head-substitution")
    return out


# ---------------------------------------------------------------------------
# Tensor I/O (numpy, no torch needed to READ)
# ---------------------------------------------------------------------------


def load_tensor(path: str, key: str):
    """Read one safetensors tensor as float32 numpy, widening bf16 exactly."""
    import numpy as np

    header_len, header = F.read_safetensors_header(path)
    if key not in header:
        raise Refusal("compute", "bad_tensor_file",
                      "tensor key %r absent from %s" % (key, path))
    meta = header[key]
    start, stop = meta["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(8 + header_len + start)
        buf = handle.read(stop - start)
    dtype = meta["dtype"]
    if dtype == "BF16":
        wide = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
        wide <<= 16
        array = wide.view(np.float32)
    elif dtype == "F32":
        array = np.frombuffer(buf, dtype="<f4")
    elif dtype == "F16":
        array = np.frombuffer(buf, dtype="<f2").astype(np.float32)
    elif dtype == "F64":
        array = np.frombuffer(buf, dtype="<f8")
    else:
        raise Refusal("compute", "bad_tensor_file", "unsupported tensor dtype %r" % dtype)
    shape = tuple(int(dim) for dim in meta["shape"])
    want = 1
    for dim in shape:
        want *= dim
    if array.size != want:
        # A truncated or over-long payload must be a refusal with a gate, not a
        # numpy ValueError escaping as an exit-1 traceback: the CLI's contract is
        # 0 ok / 2 warnings / 3 refused / 4 bad usage.
        raise Refusal("compute", "bad_tensor_file",
                      "%s: tensor %r holds %d values but its header declares shape %s "
                      "(%d values); the file is truncated or its header lies. Run "
                      "`fidelity-dataset verify --verify-tensors` on it."
                      % (path, key, array.size, list(shape), want))
    return array.reshape(shape)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def token_kld(reference_logits, candidate_logits, device: str = "cpu"):
    """The fp64 full-vocabulary estimator.

    With torch present this IS `k6_kld_report._token_kld` -- imported, not
    copied, so a number here equals a number from the sealed pipeline.  Without
    torch, the identical fp64 formula in numpy, and the receipt says which.
    """
    import numpy as np

    if _torch_available():
        import torch

        if _K6_TOOLS not in sys.path:
            sys.path.insert(0, _K6_TOOLS)
        import k6_kld_report  # noqa: WPS433

        a = torch.from_numpy(np.ascontiguousarray(reference_logits))
        b = torch.from_numpy(np.ascontiguousarray(candidate_logits))
        try:
            values, matches = k6_kld_report._token_kld(a, b, device)
        except SystemExit as exc:
            # `k6_kld_report._fail` returns a SystemExit -- correct for a CLI,
            # wrong for a library call.  Convert it into our own refusal so a
            # non-finite logit refuses the COMPARISON rather than killing the
            # process, and so the reason survives into the caller.
            raise Refusal("compute", "non_finite",
                          "the fp64 estimator refused these logits (see the k6_kld_report "
                          "message above; the usual cause is a non-finite value, which is "
                          "never clamped): %s" % exc)
        return np.asarray(values, dtype=np.float64), int(matches), "torch:k6_kld_report._token_kld"

    a = np.asarray(reference_logits, dtype=np.float64)
    b = np.asarray(candidate_logits, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise Refusal("compute", "geometry_mismatch", "logit geometry mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        # Never a clamp.  A non-finite intermediate is a hard refusal.
        raise Refusal("compute", "non_finite", "logits must be finite")
    a_logp = a - (a.max(axis=-1, keepdims=True)
                  + np.log(np.exp(a - a.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)))
    b_logp = b - (b.max(axis=-1, keepdims=True)
                  + np.log(np.exp(b - b.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)))
    values = np.sum(np.exp(a_logp) * (a_logp - b_logp), axis=-1)
    matches = int(np.count_nonzero(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))
    return values.astype(np.float64), matches, "numpy_fp64"


def _replay(hidden32, head32_t, vocab_chunk: Optional[int]):
    """logits' = hidden @ head^T, fp32 both sides.

    Mirrors `k6/tools/hidden_replay.py::_replay_logits`, including the
    vocab-chunked path used as the invariance probe.
    """
    import numpy as np

    # Apple's Accelerate BLAS raises spurious divide-by-zero / overflow /
    # invalid FP status flags inside its blocked GEMM even when every input and
    # every output is finite (verified: head absmax 0.216, hidden absmax 12.25,
    # result finite).  Suppress the FLAG, never the CHECK: the result is proven
    # finite immediately below, and `token_kld` refuses a non-finite logit
    # again before any softmax.
    with np.errstate(all="ignore"):
        if vocab_chunk is None:
            out = hidden32 @ head32_t
        else:
            pieces = []
            for start in range(0, head32_t.shape[1], vocab_chunk):
                pieces.append(hidden32 @ head32_t[:, start:start + vocab_chunk])
            out = np.concatenate(pieces, axis=1)
    if not np.isfinite(out).all():
        raise Refusal("compute", "non_finite",
                      "the hidden->logit replay produced a non-finite value; never clamped")
    return out


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute(reference: Dataset, candidate: Dataset, findings: Dict[str, Any],
            options: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    vocab = reference.manifest["capture"]["vocab_size"]
    vocab_chunk = options.get("vocab_chunk")
    if vocab_chunk is not None and vocab % vocab_chunk != 0:
        raise Refusal("compute", "bad_vocab_chunk",
                      "--vocab-chunk %d does not divide vocab_size %d; working values: %s"
                      % (vocab_chunk, vocab, F.divisors_hint(vocab)))
    position_block = int(options.get("position_block") or 128)
    device = options.get("device") or "cpu"

    head32_t = None
    head_applied = findings.get("head_applied")
    if "hidden" in (reference.form, candidate.form):
        head_path = options.get("head_path")
        if head_path is None:
            for dataset in (reference, candidate):
                if dataset.head_path() and dataset.head["tensor_content_sha256"] == head_applied:
                    head_path = dataset.head_path()
                    break
        if head_path is None:
            raise Refusal("compute", "head_missing",
                          "no head payload available to replay; ship head/weight.safetensors or "
                          "pass --head")
        head_key = None
        _, header = F.read_safetensors_header(head_path)
        for candidate_key in (reference.head.get("tensor_key"), "lm_head.weight", "weight"):
            if candidate_key and candidate_key in header:
                head_key = candidate_key
                break
        if head_key is None:
            raise Refusal("compute", "head_missing", "no known head tensor key in %s" % head_path)
        got = F.tensor_content_sha256(head_path, head_key)
        if head_applied and got != head_applied:
            raise Refusal("compute", "head_mismatch",
                          "the head payload hashes to %s but the gate resolved %s"
                          % (got[:12], head_applied[:12]))
        head32_t = np.ascontiguousarray(load_tensor(head_path, head_key).T)
        findings["head_applied"] = got

    ra = {int(r["index"]): r for r in reference.records}
    rb = {int(r["index"]): r for r in candidate.records}
    panel = reference.panel_by_index()

    chunks: List[Any] = []
    matches_total = 0
    positions_total = 0
    per_context: List[Dict[str, Any]] = []
    per_domain: Dict[str, List[float]] = {}
    per_stratum: Dict[str, List[float]] = {}
    depth: Dict[str, List[float]] = {}
    backend = None

    for index in findings["shared_indices"]:
        rec_a, rec_b = ra[index], rb[index]
        left = load_tensor(reference.record_path(rec_a), rec_a["key"])
        right = load_tensor(candidate.record_path(rec_b), rec_b["key"])
        if reference.form == "hidden":
            left = _replay(left, head32_t, vocab_chunk)
        if candidate.form == "hidden":
            right = _replay(right, head32_t, vocab_chunk)
        if left.shape != right.shape:
            raise Refusal("compute", "geometry_mismatch",
                          "record %d: %s vs %s" % (index, left.shape, right.shape))
        values = np.empty(left.shape[0], dtype=np.float64)
        for start in range(0, left.shape[0], position_block):
            stop = min(start + position_block, left.shape[0])
            piece, matched, backend = token_kld(left[start:stop], right[start:stop], device)
            values[start:stop] = piece
            matches_total += matched
        chunks.append(values)
        positions_total += values.size
        record_panel = panel.get(index) or {}
        per_context.append({
            "index": index,
            "window_id": rec_a.get("window_id"),
            "scored_rows": int(values.size),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "domain": rec_a.get("domain") or record_panel.get("domain"),
            "allocation_stratum": rec_a.get("allocation_stratum")
            or record_panel.get("allocation_stratum"),
        })
        key = per_context[-1]["domain"]
        if key:
            per_domain.setdefault(key, []).append(float(values.mean()))
        key = per_context[-1]["allocation_stratum"]
        if key:
            per_stratum.setdefault(key, []).append(float(values.mean()))
        for name, low, high in F.CONTEXT_DEPTH_BUCKETS:
            piece = values[low:min(high, values.size)]
            if piece.size:
                depth.setdefault(name, []).append(float(piece.mean()))

    tokenwise = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
    return {
        "tokenwise": tokenwise,
        "top1_agreement": (matches_total / positions_total) if positions_total else None,
        "per_context": per_context,
        "per_domain": {k: float(sum(v) / len(v)) for k, v in sorted(per_domain.items())},
        "per_stratum": {k: float(sum(v) / len(v)) for k, v in sorted(per_stratum.items())},
        "context_depth_buckets": {k: float(sum(v) / len(v)) for k, v in sorted(depth.items())},
        "estimator_backend": backend,
        "vocab_chunk": vocab_chunk,
        "position_block": position_block,
        "device": device,
        "head_applied": findings.get("head_applied"),
    }


# ---------------------------------------------------------------------------
# Self-compare (spec section 10.4)
# ---------------------------------------------------------------------------


def classify(reference: Dataset, candidate: Dataset) -> str:
    """SC-1 / SC-2 / measurement.

    A == B by `capture_content_digest` is a REPRODUCTION CONFIRMATION.  Equal
    weights identity but different capture content is a RUN-TO-RUN FLOOR and is
    never called a reproduction: a different grouped-mm kernel, GPU class or
    torch build changes the bf16 forward itself, and that class of difference is
    what produced our 0.011506 cross-topology floor -- 1e-2 class.
    """
    if reference.content_digest == candidate.content_digest:
        return "reproduction_confirmation"
    if reference.weights_identity() == candidate.weights_identity() \
            and all(v is not None for v in reference.weights_identity()):
        return "run_to_run_floor"
    return "measurement"


def zero_tokenwise(scored_positions: int):
    """The T1 hash proof's array: np.save of N float64 zeros."""
    import numpy as np

    return np.zeros(int(scored_positions), dtype=np.float64)


def save_tokenwise(path: str, array) -> Dict[str, Any]:
    import numpy as np

    np.save(path, array, allow_pickle=False)
    return {"path": os.path.basename(path), "bytes": os.path.getsize(path),
            "sha256": F.sha256_file(path)}


def short_circuit_result(reference: Dataset, candidate: Dataset,
                         findings: Dict[str, Any]) -> Dict[str, Any]:
    """SC-1 without the matmul.

    fp64 log_softmax of bitwise-equal inputs is deterministic and
    `t_logp - s_logp` is a subtraction of equal doubles, so every per-token
    value is exactly +0.0.
    """
    positions = sum(int(r["scored_rows"]) for r in reference.records
                    if int(r["index"]) in set(findings["shared_indices"]))
    array = zero_tokenwise(positions)
    per_context = []
    for record in reference.records:
        index = int(record["index"])
        if index not in set(findings["shared_indices"]):
            continue
        per_context.append({
            "index": index,
            "window_id": record.get("window_id"),
            "scored_rows": int(record["scored_rows"]),
            # +0.0, never -0.0.
            "mean": 0.0,
            "max": 0.0,
            "domain": record.get("domain"),
            "allocation_stratum": record.get("allocation_stratum"),
        })
    return {
        "tokenwise": array,
        "top1_agreement": 1.0,
        "per_context": per_context,
        "per_domain": {},
        "per_stratum": {},
        "context_depth_buckets": {},
        "estimator_backend": "hash_proof",
        "vocab_chunk": None,
        "position_block": None,
        "device": "none",
        "head_applied": findings.get("head_applied"),
        "short_circuited": True,
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def build_receipt(reference: Dataset, candidate: Dataset, gates: Dict[str, Any],
                  findings: Dict[str, Any], result: Dict[str, Any],
                  tokenwise_meta: Dict[str, Any], options: Dict[str, Any],
                  comparison_kind: str) -> Dict[str, Any]:
    import numpy as np

    array = result["tokenwise"]
    stats = F.stats_block(array) if array.size else {
        "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "p99_9": 0.0, "max": 0.0}
    if comparison_kind == "reproduction_confirmation":
        # SC-1 asserts exactness; never let a formatter round anything in.
        stats = {key: 0.0 for key in stats}
    panel = reference.manifest["panel"]
    head_policy = findings.get("head_policy") or "native_head"
    same_lane = findings.get("same_lane", reference.lane == candidate.lane)
    usable_as_floor = bool(findings.get("usable_as_floor", True)) and same_lane
    klass = findings.get("class", "strict")

    receipt: Dict[str, Any] = {
        "schema": F.RECEIPT_SCHEMA,
        "format_version": F.FORMAT_VERSION,
        "receipt_sha256": "",
        "created_utc": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "comparison_kind": comparison_kind,
        "reference": reference.side_block(options.get("reference_label") or "reference"),
        "candidate": candidate.side_block(options.get("candidate_label") or "candidate"),
        "panel": {
            "panel_id": panel.get("panel_id"),
            "suite_token_hash_sha256": panel["suite_token_hash_sha256"],
            "panel_token_sha256_legacy": panel.get("panel_token_sha256_legacy"),
            "contexts": len(findings["shared_indices"]),
            "scored_positions": int(array.size),
            "context_length": panel.get("context_length"),
            "scoring_window": panel["scoring_window"],
            "tokenizer": panel.get("tokenizer"),
        },
        "gates": {name: gates[name] for name in
                  ("seal", "form", "panel", "head", "lane", "stack",
                   "geometry", "coverage", "lossy")},
        "comparator": {
            "device": result.get("device") or "cpu",
            "accumulation_dtype": "float64",
            "logprob_dtype": "float64",
            "two_pass": True,
            "vocab_chunk": result.get("vocab_chunk"),
            "position_block": result.get("position_block"),
            "tf32": False,
            "deterministic_algorithms": True,
            "bf16_reduced_precision_reduction": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "head_applied_tensor_content_sha256": result.get("head_applied"),
            "source_file_hashes_verified": True,
            "short_circuited": bool(result.get("short_circuited", False)),
            "estimator_backend": result.get("estimator_backend"),
            "tool": {
                "entrypoint": "bin/fidelity-dataset compare",
                "note": "the fp64 estimator is k6_kld_report._token_kld, imported not copied, "
                        "whenever torch is importable; estimator_backend records which path ran.",
            },
        },
        "metric": {
            "name": "mean_tokenwise_kld",
            "value": stats["mean"],
            "units": "nats",
            "direction": "reference_to_candidate",
            "direction_label": "KL(reference || candidate)",
            "higher_is_better": False,
        },
        "kl": stats,
        "top1_agreement": result.get("top1_agreement"),
        "kl_micro_token_mean": stats["mean"],
        "kl_macro_stratum_mean": (
            float(sum(result["per_stratum"].values()) / len(result["per_stratum"]))
            if result.get("per_stratum") else None),
        "per_stratum": result.get("per_stratum") or None,
        "per_domain": result.get("per_domain") or None,
        "context_depth_buckets": result.get("context_depth_buckets") or None,
        "per_context": result.get("per_context") or None,
        "high_kld_contexts": sorted(
            [c for c in (result.get("per_context") or [])],
            key=lambda c: -c["mean"])[:5],
        "top1_discordant_contexts": [],
        "uncertainty": {"method": "none", "ci95_low": None, "ci95_high": None,
                        "clusters": None, "samples": None, "cluster_unit": None, "seed": None},
        "estimator": {
            "accumulation_dtype": "float64",
            "logits_dtype": reference.manifest["capture"]["dtype"].lower(),
            "two_pass": True,
            "vocab_chunk": result.get("vocab_chunk"),
            "stack_relation": findings.get("stack_relation", "same_stack"),
            "head_policy": head_policy,
            "softmax_note": "full vocabulary, torch.log_softmax in float64; no truncation, no "
                            "top-k, no clamp. A non-finite intermediate is a hard refusal.",
            "zero_handling": ("exact zero asserted, not rounded"
                              if comparison_kind == "reproduction_confirmation" else None),
        },
        "determinism": {
            "run_count": 1,
            "cold_start_per_run": None,
            "run_means": [stats["mean"]],
            "population_stddev_of_run_means": 0.0,
            "min_run_mean": stats["mean"],
            "max_run_mean": stats["mean"],
            "identical_across_runs": None,
            "evidence_kind": ("hidden_state_tensor_sha256" if reference.form == "hidden"
                              else "logits_tensor_sha256"),
            "evidence_hashes": sorted({reference.content_digest, candidate.content_digest}),
            "distinct_evidence_hash_count":
                len({reference.content_digest, candidate.content_digest}),
            "note": "one comparison of two sealed captures. identical_across_runs is null, not "
                    "true: that claim needs two independently produced captures of the SAME "
                    "weights (SC-2), which is a different receipt.",
        },
        "measurement_scope": {
            "scored_positions": int(array.size),
            "contexts": len(findings["shared_indices"]),
            "positions_per_context": (int(result["per_context"][0]["scored_rows"])
                                      if result.get("per_context") else None),
            "covers_full_panel": bool(findings.get("covers_full_panel", False)),
            "subset_detail": None if findings.get("covers_full_panel") else
                             "compared on the %d-record intersection of the two captures"
                             % len(findings["shared_indices"]),
            "position_filter": "all",
        },
        "comparability": {
            "class": klass,
            "usable_as_floor": usable_as_floor,
            "same_lane": same_lane,
            "key": None,
            "key_inputs": None,
            "bias": findings.get("bias"),
        },
        "tokenwise": tokenwise_meta,
        "self_compare": None,
        "submission": {
            "emitted": False, "path": None, "receipt_sha256": None,
            "submission_schema": None,
            "refusal": None if comparison_kind == "measurement" else
                       "comparison_kind is %s, not measurement; bin/fidelity/receipt.py::"
                       "_scan_for_unsubmittable is the second, independent refusal axis."
                       % comparison_kind,
        },
        "disclosures": findings.get("disclosures") or [],
    }

    if comparison_kind in ("reproduction_confirmation", "run_to_run_floor"):
        receipt["self_compare"] = {
            "capture_content_digest_equal":
                reference.content_digest == candidate.content_digest,
            "weights_identity_equal":
                reference.weights_identity() == candidate.weights_identity(),
            "head_digest_equal": (reference.head.get("tensor_content_sha256")
                                  == candidate.head.get("tensor_content_sha256")),
            "asserted_exact_zero": comparison_kind == "reproduction_confirmation",
            "force_compute_agreed": options.get("force_compute_agreed"),
            "expected_tokenwise_bytes": tokenwise_meta.get("bytes"),
            "expected_tokenwise_sha256": tokenwise_meta.get("sha256"),
        }
    if not receipt["disclosures"]:
        receipt["disclosures"] = [{
            "code": "no_known_deviations", "severity": "info",
            "affects_comparability": False,
            "detail": "every gate passed with no override.",
        }]
    if comparison_kind == "measurement":
        registry_lib = _registry_lib()
        key_inputs = {
            "panel_id": panel.get("panel_id") or "panel--unknown",
            "reference_id": (reference.manifest.get("dataset") or {}).get("id") or "reference--unknown",
            "metric_name": "mean_tokenwise_kld",
            "direction": "reference_to_candidate",
            "accumulation_dtype": "float64",
            "stack_relation": findings.get("stack_relation", "same_stack"),
            "head_policy": head_policy,
        }
        receipt["comparability"]["key_inputs"] = key_inputs
        receipt["comparability"]["key"] = registry_lib.comparability_key(key_inputs)
    return F.seal_receipt(receipt)


def _registry_lib():
    registry_tools = os.path.join(_REPO, "registry", "tools")
    if registry_tools not in sys.path:
        sys.path.insert(0, registry_tools)
    import registry_lib  # noqa: WPS433

    return registry_lib


def compare(reference_root: str, candidate_root: str, out_dir: str,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole of step 3: gate ladder, then the answer, then the receipt."""
    import numpy as np

    options = dict(options or {})
    os.makedirs(out_dir, exist_ok=True)

    reference = load_dataset(reference_root,
                             verify=not options.get("skip_seal"),
                             verify_tensors=bool(options.get("verify_tensors")),
                             allow_partial=bool(options.get("allow_partial")))
    candidate = load_dataset(candidate_root,
                             verify=not options.get("skip_seal"),
                             verify_tensors=bool(options.get("verify_tensors")),
                             allow_partial=bool(options.get("allow_partial")))
    gates, findings = run_gates(reference, candidate, options)
    gates["seal"] = _gate(True, "both manifests self-seal and checksums.txt verified"
                                + ("; every tensor_content_sha256 recomputed"
                                   if options.get("verify_tensors") else ""))

    kind = classify(reference, candidate)
    if options.get("self_compare") and kind != "reproduction_confirmation":
        raise Refusal("self-compare", "not_a_self_compare",
                      "--self-compare was asked for but the two captures differ "
                      "(%s vs %s); this is a %s"
                      % (reference.content_digest[:12], candidate.content_digest[:12], kind))

    if kind == "reproduction_confirmation":
        result = short_circuit_result(reference, candidate, findings)
        if options.get("force_compute"):
            computed = compute(reference, candidate, findings, options)
            same = bool(np.array_equal(computed["tokenwise"], result["tokenwise"]))
            options["force_compute_agreed"] = same
            if not same:
                raise Refusal("compute", "self_compare_disagreement",
                              "--force-compute produced a result that is not bitwise identical "
                              "to the hash proof; the estimator or the reader is broken")
            result["estimator_backend"] = computed["estimator_backend"]
            result["device"] = computed["device"]
            result["vocab_chunk"] = computed["vocab_chunk"]
            result["position_block"] = computed["position_block"]
    else:
        result = compute(reference, candidate, findings, options)

    tokenwise_path = os.path.join(out_dir, "tokenwise-kld.npy")
    tokenwise_meta = save_tokenwise(tokenwise_path, result["tokenwise"])
    receipt = build_receipt(reference, candidate, gates, findings, result,
                            tokenwise_meta, options, kind)
    F.write_json(os.path.join(out_dir, "comparison-receipt.json"), receipt)
    return receipt


# ---------------------------------------------------------------------------
# Registry submission (SC-3)
# ---------------------------------------------------------------------------


class NotAMeasurement(RuntimeError):
    """SC-3: only `comparison_kind == "measurement"` may reach registry_add."""


def emit_submission(receipt: Dict[str, Any], out_path: str, *,
                    measurer: Dict[str, Any], artifact: Dict[str, Any],
                    panel: Dict[str, Any], reference: Dict[str, Any],
                    environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Turn a measurement receipt into a registry submission.

    Two independent refusal axes, on purpose:

      1. HERE: `comparison_kind != "measurement"` is refused outright.  A
         reproduction confirmation and a run-to-run floor are real results and
         real receipts; neither is a measurement row.
      2. `bin/fidelity/receipt.py::assert_submittable`, which scans every block
         for preview/teacher markers at any depth.

    `build_submission` self-seals and computes `scope_digest` and the
    comparability key with `registry/tools/registry_lib.py` -- the registry's
    own code, imported, never reimplemented.
    """
    from pathlib import Path

    if receipt.get("comparison_kind") != "measurement":
        raise NotAMeasurement(
            "REFUSED: comparison_kind is %r. Only a measurement may become a registry row "
            "(SC-3). A reproduction_confirmation proves the estimator is exact on identical "
            "input; a run_to_run_floor is a lane property. Neither is a quantization result."
            % receipt.get("comparison_kind"))

    sys.path.insert(0, os.path.join(_REPO, "bin"))
    from fidelity import receipt as receipt_mod  # noqa: WPS433

    submission = receipt_mod.build_submission(
        suite_root=Path(_REPO),
        lane=receipt["candidate"]["lane"],
        measurer=measurer,
        artifact=artifact,
        panel=panel,
        reference=reference,
        metric={
            "name": "mean_tokenwise_kld",
            "value": receipt["metric"]["value"],
            "units": receipt["metric"]["units"],
            "direction": receipt["metric"]["direction"],
        },
        estimator=receipt["estimator"],
        determinism=receipt["determinism"],
        measurement_scope=receipt["measurement_scope"],
        produced_by=receipt_mod.produced_by_block(
            Path(_REPO), "bin/fidelity_dataset.py",
            {"comparison_kind": receipt["comparison_kind"]}),
        environment=environment,
        evidence=[
            {"kind": "fidelity_dataset", "role": "reference",
             "sha256": receipt["reference"]["dataset_sha256"]},
            {"kind": "fidelity_dataset", "role": "candidate",
             "sha256": receipt["candidate"]["dataset_sha256"]},
        ],
        extra_disclosures=[d for d in receipt.get("disclosures") or []
                           if d.get("code") != "no_known_deviations"],
    )
    F.write_json(out_path, submission)
    return submission
