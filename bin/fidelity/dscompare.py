"""Step 3: compare two fidelity datasets.

Two halves, deliberately separable:

  * the GATE LADDER (spec section 10.1) -- eleven ordered gates, each with a
    named refusal id.  Pure stdlib + numpy, no torch, so every refusal is
    testable on a stock py3.9 interpreter with no GPU;
  * the ESTIMATOR (spec section 10.2) -- full vocabulary, fp64 log_softmax.
    When torch is importable this is `kld_report._token_kld`, imported and
    called, never reimplemented, so a number produced here is the same number
    the sealed pipeline produces.  Without torch it falls back to the identical
    fp64 formula in numpy and SAYS SO in the receipt
    (`comparator.estimator_backend`), because a silent backend swap is exactly
    the kind of undeclared difference this whole format exists to stop.

  * the REPLAY (`_replay` / `_TorchReplay`) -- `logits' = hidden @ head.T` for
    a hidden-form capture.  This is where the wall-clock is: M1 measured one
    512-window comparison at 60m19s against a 335s capture of the same panel,
    because the head matmul ran in numpy on the CPU while the GPU already
    holding that head for the estimator sat at 0%.  `--replay-device cuda`
    moves it, and because an fp32 GEMM's accumulation order is the BLAS's
    choice, that is a DIFFERENT NUMBER in the last digits, so it is opt-in and
    named on the receipt (`comparator.replay_backend`).

The A == B short-circuit (SC-1) answers by hash proof without a matmul;
`--force-compute` runs the math anyway and asserts bitwise agreement.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import dsformat as F
from . import dsvalidate

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_K6_TOOLS = os.path.join(_REPO, "engines", "tools")


class Refusal(Exception):
    """A gate said no.

    `override` names the flag that would have allowed it; `remedy` is prose for
    the refusals that have NO override, so a caller can tell "you passed the
    wrong path" apart from "this is refused by design".
    """

    def __init__(self, gate: str, code: str, message: str, override: Optional[str] = None,
                 remedy: Optional[str] = None):
        self.gate = gate
        self.code = code
        self.message = message
        self.override = override
        self.remedy = remedy
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

    @functools.cached_property
    def weights_decode(self) -> Optional[Dict[str, Any]]:
        """The decode the capture applied to the checkpoint's bytes before the
        forward, from the sealed runtime receipt (`capture_tool.weights_decode`,
        written by `layer_outer.weights_decode_evidence`).  None for a native
        checkpoint, for a runtime receipt that predates the field, and for a
        dataset whose runtime file is absent.  The runtime file is inside
        checksums.txt, so after gate 1 its content is as trusted as the manifest.
        """
        rel = (self.manifest.get("runtime") or {}).get("file")
        if not rel:
            return None
        full = os.path.join(self.root, rel)
        if not os.path.isfile(full):
            return None
        tool = (F.read_json(full) or {}).get("capture_tool") or {}
        decode = tool.get("weights_decode")
        return decode if isinstance(decode, dict) else None


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


PANEL_REMEDY = (
    "none by design (PANEL-D3): a comparison is only meaningful between two captures of the "
    "SAME panel, so there is no override flag. Check you passed the paths you meant; "
    "otherwise recapture the candidate on the reference's panel.")


def _gate(passed: bool, detail: str, overridden_by: Optional[str] = None) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": detail, "overridden_by": overridden_by}


# The tokenizer fields that are panel IDENTITY.  `vocab_size` is here because a
# different vocabulary is a different distribution; `add_special_tokens` and
# `chat_template_applied` because either one changes what text the ids stand for.
TOKENIZER_IDENTITY_FIELDS = ("id", "repository", "revision", "vocab_size",
                             "add_special_tokens", "chat_template_applied")


def _tokenizer_divergence(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]
                          ) -> Dict[str, Tuple[Any, Any]]:
    """Which identity fields two tokenizer blocks disagree on.

    A field that is absent or null on EITHER side is unknown, not different: an
    adapted dataset legitimately cannot name a revision.  Only a genuine
    disagreement between two stated values is a refusal.
    """
    left, right = (a or {}), (b or {})
    out: Dict[str, Tuple[Any, Any]] = {}
    for field in TOKENIZER_IDENTITY_FIELDS:
        av, bv = left.get(field), right.get(field)
        if av is None or bv is None:
            continue
        if av != bv:
            out[field] = (av, bv)
    return out


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

    # --- 2b. provenance: is either side a PREVIEW capture? ------------------
    #
    # A race-mode preview root is a real, sealed, verifiable dataset backed by
    # ONE cold run, with cross-run determinism NOT demonstrated. It is published
    # under its own dataset id -- which is the `reference_id` half of the
    # comparability key computed a few hundred lines below -- so a row measured
    # against it can never share a table with a row measured against the final.
    # That is the identity half.
    #
    # This is the publishability half. A comparison against a preview is a real
    # result and gets a real receipt; what it must never do is become a registry
    # row, because the registry's whole contract is that a published number's
    # reference is settled. Rather than invent a new refusal, this raises the
    # marker the comparator ALREADY refuses on: SC-5 turns any blocking
    # disclosure into a `NotAMeasurement` inside `emit_submission`, and DISC-003
    # forces status pending/retracted downstream. One mechanism, already tested.
    preview_sides = []
    for label, dataset in (("reference", reference), ("candidate", candidate)):
        block = dataset.manifest.get("preview") or {}
        if dataset.manifest.get("not_submittable") is True or block:
            preview_sides.append((label, dataset, block))
    if preview_sides:
        detail = "; ".join(
            "%s %s is a PREVIEW capture (%s cold run(s), determinism not "
            "demonstrated, superseded by %s)"
            % (label, (ds.manifest.get("dataset") or {}).get("id"),
               (ds.manifest.get("determinism") or {}).get("run_count"),
               block.get("superseded_by") or "an unnamed final capture")
            for label, ds, block in preview_sides)
        findings["class"] = "advisory"
        findings["usable_as_floor"] = False
        findings["disclosures"].append({
            "code": "preview_capture", "severity": "blocking",
            "affects_comparability": True,
            "detail": "%s. The number is a true statement about that PREVIEW dataset and "
                      "about nothing else: it is not a distance from the final root, and "
                      "the preview will never be updated in place to become one. Re-run "
                      "the comparison against the final capture to obtain a publishable "
                      "row." % detail,
        })
        gates["provenance"] = _gate(True, "preview capture on %d side(s); the receipt "
                                          "stands, a registry row does not"
                                    % len(preview_sides))
    else:
        gates["provenance"] = _gate(True, "neither side is a preview capture")

    # --- 3. panel -----------------------------------------------------------
    pa, pb = reference.manifest["panel"], candidate.manifest["panel"]
    if pa["suite_token_hash_sha256"] != pb["suite_token_hash_sha256"]:
        gates["panel"] = _gate(False, "suite_token_hash_sha256 differs")
        raise Refusal("panel", "panel_mismatch",
                      "different panels: %s vs %s"
                      % (pa["suite_token_hash_sha256"][:12], pb["suite_token_hash_sha256"][:12]),
                      remedy=PANEL_REMEDY)
    if pa["scoring_window"] != pb["scoring_window"]:
        gates["panel"] = _gate(False, "scoring_window differs")
        raise Refusal("panel", "panel_mismatch",
                      "scoring_window is part of panel IDENTITY (PANEL-D3): score_from %r vs %r"
                      % (pa["scoring_window"].get("score_from"),
                         pb["scoring_window"].get("score_from")), remedy=PANEL_REMEDY)
    # PANEL-D6: the tokenizer is panel identity too, and `suite_token_hash_sha256`
    # cannot see it -- that digest hashes token IDS, which are integers.  Two
    # tokenizers can emit the same ids from different text, and one that applies a
    # chat template or special tokens has scored a different corpus.
    tok_diff = _tokenizer_divergence(pa.get("tokenizer"), pb.get("tokenizer"))
    if tok_diff:
        gates["panel"] = _gate(False, "tokenizer identity differs: %s"
                               % ", ".join(sorted(tok_diff)))
        raise Refusal("panel", "panel_mismatch",
                      "the two captures declare different tokenizers (PANEL-D6); the token id "
                      "digest cannot see this because it hashes integers. Differing field(s): %s"
                      % "; ".join("%s %r vs %r" % (field, a, b)
                                  for field, (a, b) in sorted(tok_diff.items())),
                      remedy=PANEL_REMEDY)
    ra = {int(r["index"]): r for r in reference.records}
    rb = {int(r["index"]): r for r in candidate.records}
    shared = sorted(set(ra) & set(rb))
    # P1-09: an EMPTY intersection is refused before any computation, with no
    # override. --allow-partial covers a partial overlap; two datasets that share
    # no context index have nothing to score, and letting them through turned
    # empty reductions into a plausible metric=0.0 / positions=0 / contexts=0
    # receipt -- a perfect-fidelity artifact about nothing.
    if not shared:
        gates["coverage"] = _gate(False, "no shared context indices")
        raise Refusal("coverage", "empty_intersection",
                      "the two datasets share NO context indices (reference %d records, "
                      "candidate %d); there is nothing to score. --allow-partial covers a "
                      "partial overlap, never a disjoint pair." % (len(ra), len(rb)),
                      remedy="check that the two captures were taken over the same panel "
                             "(or overlapping shards of it)")
    for index in shared:
        if ra[index]["token_ids_json_sha256"] != rb[index]["token_ids_json_sha256"]:
            gates["panel"] = _gate(False, "record %d token digest differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d has different tokens on the two sides (BIND-2)" % index,
                          remedy=PANEL_REMEDY)
        ma, mb = ra[index].get("attention_mask_sha256"), rb[index].get("attention_mask_sha256")
        if ma is not None and mb is not None and ma != mb:
            gates["panel"] = _gate(False, "record %d attention mask differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d attention_mask_sha256 differs (BIND-3)" % index,
                          remedy=PANEL_REMEDY)
        if ra[index]["scored_rows"] != rb[index]["scored_rows"]:
            gates["panel"] = _gate(False, "record %d scored_rows differs" % index)
            raise Refusal("panel", "panel_mismatch",
                          "record %d scored_rows %r vs %r"
                          % (index, ra[index]["scored_rows"], rb[index]["scored_rows"]),
                          remedy=PANEL_REMEDY)
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
        # Symmetric with BIAS-006.  The bias block below declares a residual of
        # the 1e-2 class with direction `unknown`; a number carrying an unknown
        # bias of that size is not a zero-point for anything, for exactly the
        # reason a cross-lane number is not.
        findings["usable_as_floor"] = False
        findings["bias"] = {
            # The registry's own enum value (measurement.schema.json); BIAS-001
            # binds it to estimator.stack_relation == cross_stack.
            "kind": "cross_stack_capture_replay",
            "direction": "unknown",
            "estimated_magnitude": None,
            "floor_measurement_ref": None,
            "detail": "the two captures were produced by different stacks; BIAS-001 requires "
                      "this block, and a residual of the 1e-2 class is expected from a "
                      "different kernel, GPU class or torch build alone. usable_as_floor is "
                      "stamped false for the same reason BIAS-006 stamps it false across "
                      "lanes: an unknown-direction residual of that size is not a zero-point.",
        }
        # measurement.schema.json rule 4: a cross-stack row needs the bias block
        # AND a comparability-affecting disclosure naming it. The bias block alone
        # makes the row schema-invalid the moment it reaches the registry.
        findings["disclosures"].append({
            "code": "cross_stack_capture", "severity": "caveat",
            "affects_comparability": True,
            "detail": "the two captures were produced by different stacks "
                      "(lane_identity %s, stack_fingerprint %s); the number carries a "
                      "cross-stack residual of the 1e-2 class in an unknown direction."
                      % ("differs" if la != lb else "equal on both sides, or unrecorded",
                         "differs" if sa != sb else "equal on both sides, or unrecorded"),
        })

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

    # --- 9b. decode ---------------------------------------------------------
    #
    # Gate 9 reads `capture.lossy_codec`, which describes the CAPTURE (a
    # llama.cpp .kld's uint16 log-probs).  It says nothing about the WEIGHTS:
    # a trellis or FP8 candidate is captured from a bf16 reconstruction of its
    # payloads under the same transformers forward as the reference, and the
    # sealed runtime receipt records that decode.  Six published GLM-5.3
    # receipts said `class: strict` with only a head disclosure while the
    # registry filed the same numbers as advisory (review-science S1-2); the
    # receipt is the more visible statement and it was the weaker one.
    _decode_gate(reference, candidate, gates, findings)
    return gates, findings


#: Decoder parity evidence against exllamav3 itself
#: (engines/tools/exl3_decoder_parity_vs_exllamav3.py writes it).  Absent, or
#: neither `all_bitwise` nor `all_bitwise_pre_hadamard` exactly true, means
#: "no parity": the caveat then says the decoder is this repository's
#: transcription, proven only against in-house routes.
DECODER_PARITY_EVIDENCE = os.path.join(
    "engines", "tools", "layer-outer-evidence", "exl3-decoder-parity-vs-exllamav3.json")


def _decoder_parity() -> Optional[Dict[str, Any]]:
    path = os.path.join(_REPO, DECODER_PARITY_EVIDENCE)
    if not os.path.isfile(path):
        return None
    try:
        doc = F.read_json(path)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("all_bitwise") is True or doc.get("all_bitwise_pre_hadamard") is True:
        return doc
    return None


def _decode_gate(reference: Dataset, candidate: Dataset, gates: Dict[str, Any],
                 findings: Dict[str, Any]) -> None:
    """Weights-only reconstruction is a comparability caveat, never strict.

    Either way the number is advisory: a weights-only reconstruction under a
    transformers bf16 forward exercises the STORED WEIGHTS, and the served
    kernel's own numerics (exllamav3's fp16 activations and on-the-fly dequant,
    an FP8 stack's per-token activation quantization) are not in it.  Decoder
    parity evidence against exllamav3 changes what the caveat can SAY about
    the decoder (bitwise vs. transcribed), not the class: the activation term
    is unmeasured with or without it.
    """
    summary = []
    parity = _decoder_parity()
    for side, dataset in (("reference", reference), ("candidate", candidate)):
        decode = dataset.weights_decode
        if not decode:
            summary.append("%s none" % side)
            continue
        method = str(decode.get("method") or "")
        summary.append("%s %s" % (side, method or "unnamed"))
        if method.startswith("exl3-trellis-"):
            findings["class"] = "advisory"
            findings["disclosures"].append({
                "code": "weights_reconstructed", "severity": "caveat",
                "affects_comparability": True,
                "detail": _reconstruction_detail(side, decode, parity),
            })
        schemes = {
            (block.get("quantization_config") or {}).get("activation_scheme")
            for block in (decode, decode.get("mixed_fp8") or {})
            if isinstance(block, dict)}
        if method == "fp8-block-dequant-to-bf16" and "dynamic" in schemes:
            findings["class"] = "advisory"
            findings["disclosures"].append({
                "code": "activation_quantization_not_captured", "severity": "caveat",
                "affects_comparability": True,
                "detail": "%s was captured from a bf16 materialisation of its block-scaled "
                          "FP8 weights (%s); the checkpoint declares activation_scheme: "
                          "dynamic, so a served W8A8 deployment also quantizes activations "
                          "per token at runtime. That term is not in this number, which is "
                          "expected to understate the served divergence; it is not a "
                          "mathematical bound. The comparison is advisory."
                          % (side, method),
            })
    gates["decode"] = _gate(True, "weights_decode %s" % "; ".join(summary))


def _reconstruction_detail(side: str, decode: Dict[str, Any],
                           parity: Optional[Dict[str, Any]]) -> str:
    method = str(decode.get("method"))
    modules = decode.get("modules_decoded")
    histogram = decode.get("k_histogram") or {}
    parts = ["%s was captured from a WEIGHTS-ONLY RECONSTRUCTION: %s trellis payload "
             "group(s) were decoded to bf16 by engines/tools/exl3hf_surface.py:"
             "decode_payload_hf, this repository's transcription of exllamav3's "
             "mul1/mcg codebooks (exllamav3_ext/quant/codebook.cuh, pack.cu), before "
             "the transformers forward (method %s%s)"
             % (side, modules if modules is not None else "the artifact's", method,
                ("; K histogram %s" % ", ".join("K%s x %s" % kv for kv in sorted(histogram.items())))
                if histogram else "")]
    compose = decode.get("tp_rank_composition") or {}
    if compose:
        parts.append("%s module(s) stored as tp=%s rank shards were composed in ascending "
                     "rank order" % (compose.get("modules_composed"), compose.get("tp")))
    padded = decode.get("zero_padded_rows_truncated") or {}
    if padded:
        parts.append("%s trailing zero row(s) were truncated on %s tensor(s) (%s)"
                     % (padded.get("rows"), padded.get("count"), padded.get("method")))
    if parity:
        version = parity.get("exllamav3_version") or "unversioned"
        count = parity.get("modules_compared") or len(parity.get("modules") or [])
        if parity.get("all_bitwise") is True:
            parts.append("decoder bitwise vs exllamav3 %s on %s real module(s) (%s); "
                         "served-kernel activations unmeasured"
                         % (version, count, DECODER_PARITY_EVIDENCE))
        else:
            # exllamav3's get_weight_tensor rounds to fp16 four times through
            # its Hadamard path where this decoder rounds once; the stage the
            # served GEMM consumes -- unpack, codebook, tile layout -- is the
            # one proven bitwise, and the fp16 weight differs by at most the
            # recorded max_abs_diff.
            worst = max((float(m.get("max_abs_diff") or 0.0)
                         for m in parity.get("modules") or []), default=None)
            parts.append("trellis unpack + codebook + tile layout bitwise vs exllamav3 %s "
                         "exllamav3_ext.reconstruct on %s real module(s) (%s); the fp16 weight "
                         "after exllamav3's own four-rounding Hadamard path differs by "
                         "max_abs_diff <= %s; served-kernel activations unmeasured"
                         % (version, count, DECODER_PARITY_EVIDENCE,
                            "%.3g" % worst if worst is not None else "unrecorded"))
    else:
        parts.append("the decoder has NOT been proven bitwise against exllamav3 itself "
                     "(no %s evidence); it is proven against in-house fp64 routes and real "
                     "payloads only" % DECODER_PARITY_EVIDENCE)
    parts.append("the served exllamav3 kernel's fp16 activations and on-the-fly dequant are "
                 "not in this number. The comparison is advisory")
    return ". ".join(parts) + "."


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
    if options.get("own_heads"):
        # HEAD-1d: each side is replayed through ITS OWN sealed head, which is
        # HEAD-2 (logit form, native heads) computed offline from the shipped
        # head payloads instead of in the capture. Nothing is substituted, so
        # the candidate's head error is INSIDE the number, exactly as under
        # HEAD-2; and nothing is erased when the heads happen to be equal, so
        # the rule is the same procedure whether da == db or not. HEAD-1c is
        # checked in compare(): bitwise-equal hiddens under differing heads
        # would make classify() call this a reproduction, which own-head
        # replay cannot honour, so it still refuses there.
        for label, dataset in (("reference", reference), ("candidate", candidate)):
            if dataset.form == "hidden" and not dataset.head_path():
                gates["head"] = _gate(False, "%s ships no head payload to replay" % label)
                raise Refusal("head", "head_missing",
                              "HEAD-1d: --own-heads replays each side through its own head, "
                              "but the %s dataset ships no head/weight.safetensors" % label)
        out["head_policy"] = "native_head"
        out["head_applied"] = None
        out["head_applied_reference"] = da
        out["head_applied_candidate"] = db
        findings["disclosures"].append({
            "code": "native_head_replay", "severity": "info", "affects_comparability": False,
            "detail": "HEAD-1d: each side replayed through its own sealed head (reference %s, "
                      "candidate %s); head error is inside the measurement, as under HEAD-2, "
                      "and nothing is substituted. The heads %s."
                      % (da[:12], db[:12],
                         "are content-identical" if da == db else "differ in content"),
        })
        gates["head"] = _gate(True, "HEAD-1d: own heads on both sides (%s vs %s)"
                              % (da[:12], db[:12]))
        return out

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
            "it. Both datasets ship their own head: --own-heads replays each side through its "
            "own (HEAD-1d, native_head, strict)." % (da[:12], db[:12]),
            override="--own-heads (each side through its own sealed head) or "
                     "--disclose-head-substitution (one head, BLOCKING disclosure)")
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


def _as_tensor(array):
    """numpy -> torch without the read-only warning, and without a needless copy.

    `load_tensor` hands back `np.frombuffer` views, which are read-only;
    `torch.from_numpy` accepts them but warns once per process. Copying only
    the read-only case keeps the hot path allocation-free.
    """
    import numpy as np
    import torch

    block = np.ascontiguousarray(array)
    if not block.flags.writeable:
        block = block.copy()
    return torch.from_numpy(block)


def token_kld(reference_logits, candidate_logits, device: str = "cpu"):
    """The fp64 full-vocabulary estimator.

    With torch present this IS `kld_report._token_kld` -- imported, not
    copied, so a number here equals a number from the sealed pipeline.  Without
    torch, the identical fp64 formula in numpy, and the receipt says which.
    """
    import numpy as np

    if _torch_available():
        import torch

        if _K6_TOOLS not in sys.path:
            sys.path.insert(0, _K6_TOOLS)
        import kld_report  # noqa: WPS433

        # Already-resident tensors pass straight through: the GPU replay path
        # hands this function logits that are ALREADY on the device, and a
        # round trip through numpy would undo the whole point of the fix (and
        # would also be the one place a host copy could silently re-enter).
        a = reference_logits if torch.is_tensor(reference_logits) else _as_tensor(reference_logits)
        b = candidate_logits if torch.is_tensor(candidate_logits) else _as_tensor(candidate_logits)
        try:
            values, matches = kld_report._token_kld(a, b, device)
        except SystemExit as exc:
            # `kld_report._fail` returns a SystemExit -- correct for a CLI,
            # wrong for a library call.  Convert it into our own refusal so a
            # non-finite logit refuses the COMPARISON rather than killing the
            # process, and so the reason survives into the caller.
            raise Refusal("compute", "non_finite",
                          "the fp64 estimator refused these logits (see the kld_report "
                          "message above; the usual cause is a non-finite value, which is "
                          "never clamped): %s" % exc)
        # PUBLISHED IDENTITY, not a stale name. This string is the
        # `estimator_backend` value inside every sealed comparison receipt this
        # tool has written -- registry/protocol/*/comparison.*.json carry it
        # verbatim, covered by their receipt_sha256. The MODULE was renamed
        # k6_kld_report.py -> kld_report.py on 2026-08-31; the backend id was
        # not, because renaming it would say a different estimator produced
        # numbers that are already out in the world.
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

    Mirrors `engines/tools/hidden_replay.py::_replay_logits`, including the
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
# Replay backends
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS.  M1 measured it: one 512-window comparison took 60m19s
# against a 335s capture of the same panel on the same box -- 10.8x the capture
# it consumes -- because `_replay` does the language-model head matmul in numpy
# on the CPU while the GPU, which is ALREADY HOLDING THE HEAD for the fp64 KLD
# step, sits at 0%.  M2 is a 642.7 GB re-capture with several candidates and M3
# is 1.5 TB; at those sizes the capture is a rounding error and the candidate
# comparisons are the entire bill.
#
# WHY IT IS NOT THE DEFAULT.  An fp32 GEMM is not one function.  BLAS
# accumulates in fp32 in an order chosen by the implementation's blocking, so
# `hidden @ head.T` has a different last bit on OpenBLAS, on Accelerate, and on
# cuBLAS.  MEASURED on the real published Qwen3.8 root, 32,752 positions,
# numpy-fp32-on-OpenBLAS against cuBLAS-fp32 with TF32 off: max absolute logit
# delta 3.624e-05 (1.360e-06 relative), and
#
#     KLD(numpy replay || cuda replay) = 5.237e-12 nats mean, 1.791e-10 max,
#     top-1 agreement 1.000000 -- not one argmax flipped.
#
# That is 1.75e-9 relative to the smallest published Qwen3.8 row (FP8,
# 0.002989850396847924) and 2.2e9 times below the streaming lane floor of
# 0.011506, so a row agrees to roughly nine significant figures and differs
# past the tenth.  Small, and NOT ZERO: the published values are 16-digit, so
# they would not reproduce.  A comparison that produces a different value after
# an optimisation is a broken comparison, not a faster one, so the numpy path
# stays the default and the receipt now names which backend ran
# (`comparator.replay_backend`), because a silent replay-backend swap is
# exactly the class of drift this suite exists to make impossible.
#
# WHAT IS BACKEND-INDEPENDENT, AND IT IS THE ONE THAT MATTERS MOST.  The floor.
# A self-compare replays bitwise-equal hidden states through one head on ONE
# backend, so both sides get bitwise-equal logits and the KLD is exactly 0.0
# on any deterministic backend -- the published `tokenwise-kld.npy` digest
# 8be5dcca... included, since it is the digest of 1,048,064 float64 zeros.
# `bin/selftest_replay_device.py` asserts that, on both backends, as a gate.


def _replay_backend_name(replay_device: Optional[str], replay_dtype: str) -> str:
    if not replay_device or replay_device == "numpy":
        return "numpy:cpu:float32"
    return "torch:%s:%s" % (replay_device, replay_dtype)


def _cpu_model() -> Optional[str]:
    """The CPU the BLAS ran on.  `platform.processor()` is empty on Linux."""
    import platform

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        import subprocess

        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5,
                                  check=False).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or None


def _blas_thread_count(blas_name: Optional[str]) -> Tuple[Optional[int], str]:
    """How many threads the BLAS used, and how that was learned.

    threadpoolctl asks the loaded library; without it, the OpenBLAS entry point
    is looked up on the libraries already mapped into this process (Linux);
    failing both, the answer is null, never a guess from os.cpu_count().
    """
    try:
        import threadpoolctl  # noqa: WPS433

        for info in threadpoolctl.threadpool_info():
            if info.get("user_api") == "blas":
                return int(info["num_threads"]), "threadpoolctl:%s" % info.get("internal_api")
    except ImportError:
        pass
    except Exception:
        pass
    if blas_name and "openblas" in blas_name.lower():
        import ctypes

        try:
            with open("/proc/self/maps", "r", encoding="utf-8") as handle:
                mapped = sorted({line.split()[-1] for line in handle
                                 if "openblas" in line.lower() and "/" in line})
        except OSError:
            mapped = []
        for path in mapped:
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for symbol in ("openblas_get_num_threads", "scipy_openblas_get_num_threads64_",
                           "openblas_get_num_threads64_", "scipy_openblas_get_num_threads"):
                fn = getattr(lib, symbol, None)
                if fn is not None:
                    fn.restype = ctypes.c_int
                    return int(fn()), "ctypes:%s" % symbol
    return None, "unresolved"


def _numpy_replay_env() -> Dict[str, Any]:
    """Name the BLAS and CPU behind `numpy:cpu:float32`, observed not asserted.

    The host term between two numpy replays of the same sealed tensors
    (1.8e-10 ... 3.8e-9 nats on the GLM-5.3 rows, workstation vs pod) is fp32
    GEMM accumulation order, which is the BLAS's blocking on this CPU with this
    many threads.  A receipt that says only `numpy:cpu:float32` leaves that
    attributable through prose alone.
    """
    import numpy as np

    blas: Dict[str, Any] = {}
    try:
        config = np.show_config(mode="dicts")
        blas = dict((config.get("Build Dependencies") or {}).get("blas") or {})
    except (TypeError, AttributeError, ValueError):
        blas = {}
    blas_name = blas.get("name")
    threads, source = _blas_thread_count(blas_name)
    return {
        "library": "numpy",
        "numpy_version": np.__version__,
        "blas_name": blas_name,
        "blas_version": blas.get("version"),
        "blas_configuration": blas.get("openblas configuration"),
        "blas_threads": threads,
        "blas_threads_source": source,
        "cpu_model": _cpu_model(),
        "replay_dtype": "float32",
    }


def _torch_replay_env(replay_device: str) -> Dict[str, Any]:
    """Pin every knob the receipt CLAIMS, and report what was actually set.

    `comparator.tf32` and friends were hardcoded constants in the receipt:
    vacuously honest while every matmul ran in numpy, a lie the moment one runs
    on a GPU.  TF32 truncates the fp32 mantissa to 10 bits, which would put the
    replay error two orders of magnitude above the fp32-accumulation term and
    below the KLD values being measured.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    for flag in ("allow_bf16_reduced_precision_reduction",
                 "allow_fp16_reduced_precision_reduction"):
        try:
            setattr(torch.backends.cuda.matmul, flag, False)
        except Exception:
            pass
    observed = {
        "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision",
                                            lambda: None)(),
        "bf16_reduced_precision_reduction": bool(
            getattr(torch.backends.cuda.matmul,
                    "allow_bf16_reduced_precision_reduction", False)),
    }
    if replay_device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        observed["device_name"] = torch.cuda.get_device_name(index)
        observed["device_capability"] = "%d.%d" % torch.cuda.get_device_capability(index)
    return observed


class _TorchReplay(object):
    """The head, resident, plus the chunked matmul that consumes it.

    Chunking is value-preserving BY CONSTRUCTION and that is the whole reason
    it is allowed here: every output element of `hidden @ head.T` is an
    independent dot product over the hidden axis, so splitting the POSITION
    axis or the VOCABULARY axis partitions the output without touching any
    reduction.  Splitting the hidden axis would not be, and is never done.
    """

    def __init__(self, head32_t, replay_device: str, replay_dtype: str,
                 vocab_chunk: Optional[int]):
        import torch

        self.env = _torch_replay_env(replay_device)
        self.device = torch.device(replay_device)
        self.dtype = torch.float64 if replay_dtype == "float64" else torch.float32
        self.vocab_chunk = vocab_chunk
        self.head = _as_tensor(head32_t).to(device=self.device, dtype=self.dtype)
        self.head_bytes = int(self.head.element_size() * self.head.nelement())

    def to_device(self, block):
        return _as_tensor(block).to(device=self.device, dtype=self.dtype)

    def replay(self, hidden_block):
        """logits' = hidden @ head^T for one block of positions."""
        import torch

        hidden = self.to_device(hidden_block)
        if self.vocab_chunk is None:
            out = hidden @ self.head
        else:
            out = torch.cat(
                [hidden @ self.head[:, start:start + self.vocab_chunk]
                 for start in range(0, self.head.shape[1], self.vocab_chunk)], dim=1)
        # Same contract as the numpy path: proven finite here, refused -- never
        # clamped -- and refused again by the estimator before any softmax.
        if not torch.isfinite(out).all():
            raise Refusal("compute", "non_finite",
                          "the hidden->logit replay produced a non-finite value on %s; "
                          "never clamped" % self.device)
        return out

    def peak_bytes(self) -> Optional[int]:
        import torch

        if self.device.type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(self.device))

    def reset_peak(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute(reference: Dataset, candidate: Dataset, findings: Dict[str, Any],
            options: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    vocab = reference.manifest["capture"]["vocab_size"]
    vocab_chunk = options.get("vocab_chunk")
    if (vocab_chunk is not None
            and (isinstance(vocab_chunk, bool)
                 or not isinstance(vocab_chunk, int)
                 or vocab_chunk < 1)):
        raise Refusal(
            "compute", "bad_vocab_chunk",
            "--vocab-chunk must be a positive integer; got %r" % vocab_chunk)
    # CLI-05. `or 128` also swallowed 0, and nothing bounded the value. A negative
    # --chunk-positions made range() empty, the loop below never ran, and `values` --
    # allocated with np.empty -- was published as the headline metric, the per_context
    # means, the kl percentiles AND tokenwise-kld.npy, straight from uninitialized heap.
    # Measured on a real fixture: metric 2.0 nats against a true 3.688, tokenwise
    # [3,2,1,3,2,1], backend null, and both artifacts written to disk before the schema's
    # `minimum: 1` caught it. A plausible-looking wrong number from a typo in a flag.
    raw_block = options.get("position_block")
    position_block = 128 if raw_block is None else int(raw_block)
    if position_block < 1:
        raise Refusal("compute", "bad_position_block",
                      "--chunk-positions must be >= 1; got %d" % position_block)
    device = options.get("device") or "cpu"
    replay_device = options.get("replay_device") or "numpy"
    replay_dtype = options.get("replay_dtype") or "float32"
    if replay_dtype not in ("float32", "float64"):
        raise Refusal("compute", "bad_replay_dtype",
                      "--replay-dtype must be float32 or float64; got %r" % replay_dtype)
    if replay_device != "numpy" and not _torch_available():
        raise Refusal("compute", "replay_backend_unavailable",
                      "--replay-device %s needs torch, which is not importable here. The "
                      "default (numpy) is the published path and needs nothing."
                      % replay_device)
    if replay_device != "numpy" and replay_device != device:
        # The estimator and the replay on two different devices would move the
        # logits across PCIe once per position block -- slower than the numpy
        # path it replaces -- and the receipt has one `device` field to name.
        raise Refusal("compute", "replay_device_mismatch",
                      "--replay-device %s and --device %s disagree; the replay must run on "
                      "the device the estimator consumes, or the logits cross the bus "
                      "twice per block" % (replay_device, device))

    def _load_head(head_path: str, want: Optional[str], tensor_key_hint: Optional[str]):
        head_key = None
        _, header = F.read_safetensors_header(head_path)
        for candidate_key in (tensor_key_hint, "lm_head.weight", "weight"):
            if candidate_key and candidate_key in header:
                head_key = candidate_key
                break
        if head_key is None:
            raise Refusal("compute", "head_missing", "no known head tensor key in %s" % head_path)
        got = F.tensor_content_sha256(head_path, head_key)
        if want and got != want:
            raise Refusal("compute", "head_mismatch",
                          "the head payload hashes to %s but the gate resolved %s"
                          % (got[:12], want[:12]))
        return np.ascontiguousarray(load_tensor(head_path, head_key).T), got

    # One head per SIDE. Under every rule but HEAD-1d both names point at the
    # same array; under HEAD-1d each hidden-form side is replayed through the
    # head its own dataset sealed, loaded once when the two digests coincide.
    head32_t = None
    head32_t_a = head32_t_b = None
    head_applied = findings.get("head_applied")
    if findings.get("head_policy") == "native_head" and "hidden" in (reference.form, candidate.form) \
            and findings.get("head_applied_reference") is not None:
        if options.get("head_path"):
            raise Refusal("compute", "head_mismatch",
                          "--head names one head to apply to both sides; --own-heads replays "
                          "each side through its own sealed head. Pass one or the other.")
        loaded: Dict[str, Any] = {}
        for label, dataset, want in (("reference", reference, findings["head_applied_reference"]),
                                     ("candidate", candidate, findings["head_applied_candidate"])):
            if dataset.form != "hidden":
                continue
            if want not in loaded:
                loaded[want] = _load_head(dataset.head_path(), want,
                                          dataset.head.get("tensor_key"))[0]
            if label == "reference":
                head32_t_a = loaded[want]
            else:
                head32_t_b = loaded[want]
        head32_t = head32_t_a if head32_t_a is not None else head32_t_b
    elif "hidden" in (reference.form, candidate.form):
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
        head32_t, got = _load_head(head_path, head_applied, reference.head.get("tensor_key"))
        head32_t_a = head32_t_b = head32_t
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

    replayer = replayer_a = replayer_b = None
    if head32_t is not None and replay_device != "numpy":
        replayer_a = _TorchReplay(head32_t_a, replay_device, replay_dtype, vocab_chunk) \
            if head32_t_a is not None else None
        # The same array is the same resident head: never upload it twice.
        if head32_t_b is head32_t_a:
            replayer_b = replayer_a
        elif head32_t_b is not None:
            replayer_b = _TorchReplay(head32_t_b, replay_device, replay_dtype, vocab_chunk)
        replayer = replayer_a if replayer_a is not None else replayer_b
        replayer.reset_peak()

    for index in findings["shared_indices"]:
        rec_a, rec_b = ra[index], rb[index]
        left = load_tensor(reference.record_path(rec_a), rec_a["key"])
        right = load_tensor(candidate.record_path(rec_b), rec_b["key"])
        if replayer is None:
            if reference.form == "hidden":
                left = _replay(left, head32_t_a, vocab_chunk)
            if candidate.form == "hidden":
                right = _replay(right, head32_t_b, vocab_chunk)
        rows_a = left.shape[0]
        rows_b = right.shape[0]
        width_a = vocab if reference.form == "hidden" and replayer is not None else left.shape[1]
        width_b = vocab if candidate.form == "hidden" and replayer is not None else right.shape[1]
        if (rows_a, width_a) != (rows_b, width_b):
            raise Refusal("compute", "geometry_mismatch",
                          "record %d: %s vs %s" % (index, (rows_a, width_a), (rows_b, width_b)))
        # np.full(nan), not np.empty: an unwritten slot must be detectable rather than
        # being whatever the allocator last held. The check after the loop is a Refusal,
        # not an assert -- asserts vanish under `python -O`.
        values = np.full(rows_a, np.nan, dtype=np.float64)
        for start in range(0, rows_a, position_block):
            stop = min(start + position_block, rows_a)
            if replayer is None:
                block_a, block_b = left[start:stop], right[start:stop]
            else:
                # The fix, in four lines: the head is already resident for the
                # fp64 estimator, so the replay happens THERE, one position
                # block at a time, and the full [positions x vocab] fp32 logit
                # array is never materialised on the host at all.
                block_a = (replayer_a.replay(left[start:stop]) if reference.form == "hidden"
                           else replayer.to_device(left[start:stop]))
                block_b = (replayer_b.replay(right[start:stop]) if candidate.form == "hidden"
                           else replayer.to_device(right[start:stop]))
            piece, matched, backend = token_kld(block_a, block_b, device)
            values[start:stop] = piece
            matches_total += matched
        # Belt and braces on the poisoned buffer: token_kld already refuses non-finite
        # logits on both the torch and numpy paths, and an fp64 log-softmax difference
        # cannot make a non-finite KLD from finite inputs, so this can only fire on a
        # genuinely unwritten slot.
        if values.size and not np.isfinite(values).all():
            raise Refusal("compute", "incomplete_scan",
                          "position scan left %d of %d positions unwritten"
                          % (int((~np.isfinite(values)).sum()), values.size))
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
        "head_applied_reference": findings.get("head_applied_reference",
                                               findings.get("head_applied")),
        "head_applied_candidate": findings.get("head_applied_candidate",
                                               findings.get("head_applied")),
        # Named on EVERY receipt, including the numpy default. Two rows are
        # only rankable against each other if their replay ran on the same
        # backend, and a field that appears only when the non-default path ran
        # would make the default look like "unknown" instead of "numpy".
        "replay_backend": _replay_backend_name(replay_device, replay_dtype)
        if head32_t is not None else None,
        # The numpy path names its BLAS, CPU and thread count for the same
        # reason the torch path names its device: the host term lives there.
        "replay_env": (replayer.env if replayer is not None
                       else _numpy_replay_env() if head32_t is not None else None),
        "replay_head_bytes": replayer.head_bytes if replayer is not None else None,
        "replay_peak_device_bytes": replayer.peak_bytes() if replayer is not None else None,
    }


# ---------------------------------------------------------------------------
# Self-compare (spec section 10.4)
# ---------------------------------------------------------------------------


def classify(reference: Dataset, candidate: Dataset,
             options: Optional[Dict[str, Any]] = None,
             findings: Optional[Dict[str, Any]] = None) -> str:
    """SC-1 / SC-2 / measurement.

    A == B by `capture_content_digest` is a REPRODUCTION CONFIRMATION.  Equal
    weights identity but different capture content is a RUN-TO-RUN FLOOR and is
    never called a reproduction: a different grouped-mm kernel, GPU class or
    torch build changes the bf16 forward itself, and that class of difference is
    what produced our 0.011506 cross-topology floor -- 1e-2 class.

    HEAD-1c is checked HERE and not in the head gate, because it is the one head
    condition that only exists once the capture digests are known to be equal.
    Under HEAD-1d (`own_heads`) the same condition is a MEASUREMENT: each side
    goes through its own head, so identical hiddens under different heads yield
    exactly the head-quantization KL -- the quantity a head-only quant (stock
    EXL3 head_bits 6-8) needs, and the one case own-head replay exists to
    measure.  It is recorded on `findings` as `head_only_difference`.
    """
    if reference.content_digest == candidate.content_digest:
        da = reference.head.get("tensor_content_sha256")
        db = candidate.head.get("tensor_content_sha256")
        if "hidden" in (reference.form, candidate.form) and da != db:
            if (options or {}).get("own_heads") and findings is not None:
                findings["disclosures"].append({
                    "code": "head_only_difference", "severity": "info",
                    "affects_comparability": False,
                    "detail": "the two captures share bitwise-identical hidden states "
                              "(capture_content_digest %s); the whole of this number is the "
                              "head difference (%s vs %s), each side replayed through its "
                              "own sealed head (HEAD-1d)."
                              % (reference.content_digest[:12], (da or "null")[:12],
                                 (db or "null")[:12]),
                })
                return "measurement"
            # HEAD-1c: bitwise-equal hiddens under DIFFERENT heads.  A head-only
            # quant (stock EXL3 head_bits 6-8 is exactly this) changes nothing
            # before the final norm, so its post-norm hiddens are bitwise
            # identical to the reference's and the capture content digests
            # match.  Replaying both sides through ONE head then subtracts a
            # quantity from itself: the answer is exactly 0.0 and it measures
            # nothing.  There is no override on the shared-head path, because
            # there is no reading of that comparison under which the number
            # means anything; --own-heads is the other procedure, not an override.
            raise Refusal(
                "head", "head_substitution_vacuous",
                "HEAD-1c: the two captures are bitwise identical "
                "(capture_content_digest %s) but declare DIFFERENT heads (%s vs %s). "
                "The head is therefore the whole of the difference between these two "
                "artifacts, and hidden replay through a single head erases exactly that "
                "-- the result would be 0.0 nats and would read as an exact reproduction. "
                "A head-only quantization is measured with --own-heads (HEAD-1d: each side "
                "replayed through its own sealed head), or from logit-form captures, where "
                "each side applies its own head (HEAD-2)."
                % (reference.content_digest[:12], (da or "null")[:12], (db or "null")[:12]))
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
        "replay_backend": "none",
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
    capture_dtype = reference.manifest["capture"]["dtype"].lower()
    replay_backend = result.get("replay_backend")
    replayed = bool(replay_backend) and replay_backend != "none"
    logits_dtype = replay_backend.rsplit(":", 1)[-1] if replayed else capture_dtype
    hidden_dtype = next((ds.manifest["capture"]["dtype"].lower()
                         for ds in (reference, candidate) if ds.form == "hidden"), None)

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
            # Both sides, always.  The gate refuses a genuine disagreement, but a
            # receipt that prints only ONE side's block is asserting something it
            # did not check whenever a field is null on the other side.
            "tokenizer_reference": panel.get("tokenizer"),
            "tokenizer_candidate": candidate.manifest["panel"].get("tokenizer"),
            "tokenizer_identity_equal": not _tokenizer_divergence(
                panel.get("tokenizer"), candidate.manifest["panel"].get("tokenizer")),
        },
        "gates": {name: gates[name] for name in
                  ("seal", "form", "panel", "head", "lane", "stack",
                   "geometry", "coverage", "lossy", "decode") if name in gates},
        "comparator": {
            "device": result.get("device") or "cpu",
            "accumulation_dtype": "float64",
            "logprob_dtype": "float64",
            "two_pass": True,
            "vocab_chunk": result.get("vocab_chunk"),
            "position_block": result.get("position_block"),
            # The replay backend is part of the number, not part of the
            # plumbing: an fp32 GEMM accumulates in an order the BLAS chooses,
            # so numpy-on-Accelerate, numpy-on-OpenBLAS and cuBLAS give
            # different last digits from the same head and the same hidden
            # states. Two rows are comparable only if this field matches.
            "replay_backend": result.get("replay_backend"),
            "replay_env": result.get("replay_env"),
            "replay_peak_device_bytes": result.get("replay_peak_device_bytes"),
            # Observed, not asserted, whenever a torch device did the replay.
            "tf32": (result.get("replay_env") or {}).get("tf32", False),
            "deterministic_algorithms": True,
            "bf16_reduced_precision_reduction": (
                result.get("replay_env") or {}).get("bf16_reduced_precision_reduction", False),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "head_applied_tensor_content_sha256": result.get("head_applied"),
            # Additive (2026-09-05): under HEAD-1d each hidden-form side is
            # replayed through its OWN sealed head, so the single field above is
            # null and these two name what each side actually saw. Under every
            # other rule they both equal the single field.
            "head_applied_reference_tensor_content_sha256": result.get(
                "head_applied_reference", result.get("head_applied")),
            "head_applied_candidate_tensor_content_sha256": result.get(
                "head_applied_candidate", result.get("head_applied")),
            # What was ACTUALLY recomputed, never a constant.  The seal covers the
            # manifest and checksums.txt on every run; the per-tensor
            # `tensor_content_sha256` values are only re-derived when the caller
            # asked, and a receipt has to say which of the two it is.
            "tensor_content_digests_verified": bool(options.get("verify_tensors")),
            "source_file_hashes_verified": bool(options.get("verify_tensors")),
            "short_circuited": bool(result.get("short_circuited", False)),
            "estimator_backend": result.get("estimator_backend"),
            "tool": {
                "entrypoint": "bin/fidelity-dataset compare",
                # Same reason as the estimator_backend literal above: this
                # note is inside sealed receipts, so it keeps the module's
                # 2026-08 spelling. Today the file is engines/tools/kld_report.py.
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
            # The dtype the estimator CONSUMED.  A hidden-form side is replayed
            # (`logits' = hidden @ head.T`) in the replay backend's dtype, so the
            # logits are float32 on the published numpy path even though the
            # sealed capture is bf16; seven published receipts said "bf16" here
            # while their own `replay_backend` said numpy:cpu:float32 (S2-1).
            # The capture dtype survives only where no replay happened: a
            # logit-form capture, or a hash-proof short circuit.
            "logits_dtype": logits_dtype,
            "hidden_dtype": hidden_dtype,
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
        receipt["comparability"]["key"] = registry_lib.comparability_key(key_inputs)
        # The registry hashes its OWN reference record id (reference--...), not
        # the dataset id this receipt knows, so the two keys never match and a
        # reader cross-checking sees a silent mismatch. CMP-001 recomputes the
        # key at ingest; this one is a preview and says so.
        key_inputs["note"] = ("provisional; the registry recomputes with its reference "
                              "record id (registry CMP-001)")
        receipt["comparability"]["key_inputs"] = key_inputs
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

    # Tensor verification is ON unless the caller says otherwise, in the library
    # as well as the CLI. Recomputing checksums.txt and the seal does NOT catch a
    # byte flipped inside a tensor whose checksums were refreshed afterwards --
    # and refreshing them is what re-running finalize after an edit does. The
    # receipt records which of the two ran.
    options["verify_tensors"] = bool(options.get("verify_tensors", True))
    reference = load_dataset(reference_root,
                             verify=not options.get("skip_seal"),
                             verify_tensors=options["verify_tensors"],
                             allow_partial=bool(options.get("allow_partial")))
    candidate = load_dataset(candidate_root,
                             verify=not options.get("skip_seal"),
                             verify_tensors=options["verify_tensors"],
                             allow_partial=bool(options.get("allow_partial")))
    gates, findings = run_gates(reference, candidate, options)
    gates["seal"] = _gate(True, "both manifests self-seal and checksums.txt verified"
                                + ("; every tensor_content_sha256 recomputed"
                                   if options.get("verify_tensors") else ""))

    kind = classify(reference, candidate, options, findings)
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
            for field in ("replay_backend", "replay_env", "replay_peak_device_bytes"):
                result[field] = computed.get(field)
    else:
        result = compute(reference, candidate, findings, options)

    # P1-09: outputs are staged in a private temporary directory and renamed
    # into out_dir only after the receipt passes its own schema/seal validation.
    # The old order wrote the array and the receipt first and validated later
    # (in the CLI), so a refused comparison still left a plausible receipt on
    # disk for any library caller or shell script to pick up.
    import shutil
    import tempfile

    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".compare-staging-", dir=parent)
    try:
        tokenwise_meta = save_tokenwise(os.path.join(tmp_dir, "tokenwise-kld.npy"),
                                        result["tokenwise"])
        receipt = build_receipt(reference, candidate, gates, findings, result,
                                tokenwise_meta, options, kind)
        report = dsvalidate.validate_receipt(receipt, out_dir)
        if report.errors:
            raise Refusal(
                "publish", "receipt_invalid",
                "the computed receipt fails its own schema/seal validation and is NOT "
                "written: %s"
                % "; ".join("%(rule)s %(code)s: %(message)s" % e for e in report.errors[:5]))
        F.write_json(os.path.join(tmp_dir, "comparison-receipt.json"), receipt)
        os.makedirs(out_dir, exist_ok=True)
        # The array lands first, the receipt last: at no instant does a receipt
        # exist whose tokenwise file is missing.
        os.replace(os.path.join(tmp_dir, "tokenwise-kld.npy"),
                   os.path.join(out_dir, "tokenwise-kld.npy"))
        os.replace(os.path.join(tmp_dir, "comparison-receipt.json"),
                   os.path.join(out_dir, "comparison-receipt.json"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return receipt


# ---------------------------------------------------------------------------
# Registry submission (SC-3)
# ---------------------------------------------------------------------------


class NotAMeasurement(RuntimeError):
    """SC-3: only `comparison_kind == "measurement"` may reach registry_add."""


class MissingProvenance(RuntimeError):
    """SC-4: a submission names a registered artifact, panel and reference.

    None of the three can be derived from a fidelity dataset: `artifact` is the
    quant's HF repository at a 40-hex revision with its codec and scope,
    `panel_ref`/`reference_ref` are registry ids that must already exist.  The
    comparator refuses to write a file with those blocks empty rather than
    emitting something that fails `registry_validate.py --submission` with
    twenty schema errors an hour later.
    """


# `submission.schema.json#/properties/determinism` sets additionalProperties:false.
# The comparison receipt's determinism block is deliberately RICHER -- it carries
# min/max/stddev of the run means, which the receipt schema wants and the
# submission schema forbids -- so the crossing is a projection, listed here rather
# than done by deletion so a new receipt field cannot silently leak through.
_SUBMISSION_DETERMINISM_FIELDS = (
    "run_count", "cold_start_per_run", "run_means", "identical_across_runs",
    "evidence_kind", "evidence_hashes", "distinct_evidence_hash_count",
    "per_run_report_sha256", "note",
)
_SUBMISSION_ESTIMATOR_FIELDS = (
    "accumulation_dtype", "head_policy", "logits_dtype", "padded_columns_masked",
    "softmax_note", "stack_relation", "two_pass", "vocab_chunk",
    "vocab_masking_policy", "zero_handling",
)
_SUBMISSION_SCOPE_FIELDS = (
    "calibration_overlap_scan", "contexts", "covers_full_panel", "position_filter",
    "positions_per_context", "scope_name", "scope_selection_file",
    "scope_selection_sha256", "scored_positions", "subset_detail",
)


def _project(block: Optional[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    block = block or {}
    return {key: block[key] for key in fields if key in block}


#: The receipt names dtypes the way numpy/torch do (`float32`, the replay
#: dtype); the registry's `common.schema.json#/$defs/numeric_format` enum spells
#: them `fp32`.  Mapped at the crossing, never inside the sealed receipt.
_REGISTRY_NUMERIC_FORMAT = {"float32": "fp32", "float64": "fp64", "f32": "fp32",
                            "f16": "fp16", "float16": "fp16", "bfloat16": "bf16"}


def _submission_estimator(estimator: Dict[str, Any]) -> Dict[str, Any]:
    block = _project(estimator, _SUBMISSION_ESTIMATOR_FIELDS)
    dtype = block.get("logits_dtype")
    if isinstance(dtype, str):
        block["logits_dtype"] = _REGISTRY_NUMERIC_FORMAT.get(dtype.lower(), dtype)
    return block


def _evidence_source(side: Dict[str, Any], role: str) -> Dict[str, Any]:
    """A `common.schema.json#/$defs/source` pointing at one fidelity dataset.

    `kind` must come from the registry's enum and `uri` is required; there is no
    `fidelity_dataset` kind and no `role` property, so the role travels in the
    note.  The digest is the dataset SEAL, which is what makes the pointer
    checkable (DS-001).
    """
    repository, revision = side.get("repository"), side.get("revision")
    note = ("%s fidelity dataset %s (malaiwah.fidelity-dataset.v1, form %s, lane %s)"
            % (role, side.get("dataset_id") or "unnamed", side.get("form"), side.get("lane")))
    if repository:
        uri = "https://huggingface.co/datasets/%s/blob/%s/fidelity-dataset.json" % (
            repository, revision or "main")
        return {"kind": "hf_file", "uri": uri, "sha256": side["dataset_sha256"], "note": note}
    # Not published: name it by its own id, never by a path on the measuring box
    # -- that is the defect (`packed_root: /home/jl_fs/...`) this format exists
    # to stop shipping.
    return {"kind": "filesystem_path",
            "uri": "%s/fidelity-dataset.json" % (side.get("dataset_id") or "fidelity-dataset"),
            "sha256": side["dataset_sha256"],
            "note": note + " -- not published; the digest is the only pointer"}


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

    # SC-5.  A blocking disclosure means the comparator itself said the number is
    # not publishable as a measurement.  The registry's DISC-003 says the same,
    # but ONLY at row-ingest time -- `registry_validate.py --submission` runs
    # `check_submission`, which never calls `check_disclosures`.  So the tool that
    # minted the number is the one that has to refuse it.
    blocking = [d for d in receipt.get("disclosures") or []
                if d.get("severity") == "blocking"]
    if blocking:
        raise NotAMeasurement(
            "REFUSED: the comparison carries %d blocking disclosure(s) -- %s. A blocking "
            "disclosure is the comparator saying this number is not publishable as a "
            "measurement (DISC-003 forces status pending/retracted downstream). The receipt "
            "stands as a result; it does not become a registry row."
            % (len(blocking), ", ".join(sorted(d.get("code", "?") for d in blocking))))

    missing = []
    for label, block, required in (("artifact", artifact,
                                    ("repository", "revision", "container", "codec",
                                     "scope", "producer")),
                                   ("panel", panel, ("panel_ref", "panel_token_sha256")),
                                   ("reference", reference,
                                    ("reference_ref", "teacher_receipt_sha256"))):
        for key in required:
            if not (block or {}).get(key):
                missing.append("%s.%s" % (label, key))
    if missing:
        raise MissingProvenance(
            "REFUSED to write a submission: %d required field(s) have no value -- %s.\n"
            "  These name registry records and cannot be derived from a fidelity dataset: "
            "the artifact is an HF repository at a 40-hex revision with its codec and "
            "quantization scope, and panel_ref/reference_ref must already exist in the "
            "registry (a measurement may not introduce a panel).\n"
            "  Fix: pass --submission-provenance FILE. `bin/fidelity-dataset compare "
            "--print-provenance-template` writes a skeleton with every required key."
            % (len(missing), ", ".join(missing)))

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
        estimator=_submission_estimator(receipt["estimator"]),
        determinism=_project(receipt["determinism"], _SUBMISSION_DETERMINISM_FIELDS),
        measurement_scope=_project(receipt["measurement_scope"], _SUBMISSION_SCOPE_FIELDS),
        produced_by=receipt_mod.produced_by_block(
            Path(_REPO), "bin/fidelity_dataset.py",
            {"comparison_kind": receipt["comparison_kind"]}),
        environment=environment,
        evidence=[
            _evidence_source(receipt["reference"], "reference"),
            _evidence_source(receipt["candidate"], "candidate"),
        ],
        auxiliary_metrics={"top1_agreement": receipt.get("top1_agreement"),
                           "median_kld": receipt["kl"].get("median"),
                           "p95_kld": receipt["kl"].get("p95"),
                           "p99_kld": receipt["kl"].get("p99"),
                           "p999_kld": receipt["kl"].get("p99_9"),
                           "max_kld": receipt["kl"].get("max"),
                           "context_macro_mean_kld": receipt.get("kl_macro_stratum_mean")},
        # The bias block and usable_as_floor are the comparator's own verdict on
        # how the number may be READ.  Dropping them here is how a row ends up
        # carrying `bias: null` for a comparison its own receipt declared biased.
        comparability={"bias": (receipt.get("comparability") or {}).get("bias"),
                       "usable_as_floor": (receipt.get("comparability") or {})
                       .get("usable_as_floor")},
        extra_disclosures=[d for d in receipt.get("disclosures") or []
                           if d.get("code") != "no_known_deviations"],
    )
    F.write_json(out_path, submission)
    return submission
