#!/usr/bin/env python3
"""T8 -- the comparator's numerics: known answers and exactness.

Runs on any interpreter with numpy.  When torch is importable the estimator
under test IS `kld_report._token_kld` (imported, not copied); otherwise the
numpy fp64 fallback runs and the receipt says so.  Both paths are asserted
against the SAME analytic answers, which is the point: a backend swap must not
move a number.

    N1   known-answer KLD, analytic, 1e-15
    N2   KL(x||x) on a random capture is all-zero, exactly
    N3   self-compare A == B by digest: exactly 0.0, top-1 exactly 1.0, +0.0 maxima
    N4   N3 with --force-compute: the computed array is bitwise identical
    N5   the T1 constant: 51,175 float64 zeros -> 409,528 bytes, 3ffddc61...be17
    N6   same weights identity, different capture content -> run_to_run_floor
    N7   vocab-chunk invariance
    N8   a --vocab-chunk that does not divide vocab_size is refused with the hint
    N9   a NaN in one capture -> hard refusal, never a clamp
    N10  a permuted head applied at replay -> large KLD (the estimator has teeth)
    N11  a reproduction-confirmation receipt fed to the submission builder -> refused
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from fidelity import dsformat as F  # noqa: E402
from fidelity import dscompare, dsvalidate  # noqa: E402

import selftest_fidelity_dataset as fixtures  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []

#: A real registry panel, reference and artifact. N16 is worthless against a made-up
#: triple: `registry_add.submission_to_records` resolves every id, cross-checks the
#: panel token digest and the teacher receipt digest against the rows, and refuses a
#: measurement that would introduce a panel. These three exist in registry/data/.
SUBMISSION_PROVENANCE = {
    "measurer": {"name": "selftest", "handle": "selftest", "url": None,
                 "is_artifact_author": False},
    "artifact": {
        "repository": "malaiwah/GLM-5.3-Flash-TR3-6bpw",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "url": None, "container": "exl3", "precision_label": "6bpw",
        "size_bytes": None, "index_sha256": None, "config_sha256": None,
        "shard_hash_verification": "none",
        "codec": {"family": "exl3-mcg", "bits_per_weight_nominal": 6.0,
                  "bits_per_weight_effective": None, "group_size": None,
                  "quantizer_tool": "exllamav3", "quantizer_version": None},
        "scope": {"policy": "mixed", "head_policy": "native", "kv_cache_dtype": "bf16",
                  "assignments": [{"tensor_class": "moe.experts",
                                   "treatment": "quantized", "format": "exl3-mcg",
                                   "bits_per_weight": 6.0, "layer_range": None}]},
        "producer": {"name": "selftest", "handle": None, "url": None},
    },
    "panel": {"panel_ref": "panel--glm53.brandonmusic.final-0000",
              "panel_token_sha256":
                  "338027e62f41540f73e38c6f9b4b9a06a50196cbd38cd9c69f11886af9d3cf9f",
              "panel_receipt_sha256": None, "contexts": None,
              "scored_positions_total": None},
    "reference": {"reference_ref": "reference--brandonmusic.glm53-bf16-fp32-logits.final-0000",
                  "teacher_receipt_sha256":
                      "2ae08117c3d4247f747b2a9a889b68e1a06387b788d56a0bf23bb950c77bc5a5",
                  "teacher_backend_identity_sha256": None},
}


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def analytic_kl(p_logits, q_logits):
    """KL(P||Q) from logits, in plain python floats -- an independent oracle."""
    out = []
    for row_p, row_q in zip(p_logits, q_logits):
        mp = max(row_p)
        mq = max(row_q)
        zp = sum(math.exp(v - mp) for v in row_p)
        zq = sum(math.exp(v - mq) for v in row_q)
        total = 0.0
        for vp, vq in zip(row_p, row_q):
            p = math.exp(vp - mp) / zp
            lp = (vp - mp) - math.log(zp)
            lq = (vq - mq) - math.log(zq)
            total += p * (lp - lq)
        out.append(total)
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="fidelity-compare-selftest-")
    backend_note = "torch" if dscompare._torch_available() else "numpy fp64 fallback"
    print("== N: comparator numerics (estimator backend: %s) ==" % backend_note)
    try:
        # -- N1 known-answer -------------------------------------------------
        rng = np.random.RandomState(11)
        p = rng.normal(size=(2, 8)).astype(np.float32) * 3.0
        q = rng.normal(size=(2, 8)).astype(np.float32) * 3.0
        values, matches, backend = dscompare.token_kld(p, q, "cpu")
        want = analytic_kl(p.astype(np.float64).tolist(), q.astype(np.float64).tolist())
        worst = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(values.tolist(), want))
        check("N1  known-answer KLD (2 x 8, analytic) agrees to fp64 epsilon (<1e-15 rel)",
              worst < 1e-15, "worst relative delta = %.3e, backend=%s" % (worst, backend))

        # -- N2 KL(x||x) -----------------------------------------------------
        values, matches, _ = dscompare.token_kld(p, p, "cpu")
        check("N2  KL(x||x) is exactly zero everywhere and top-1 agrees",
              bool(np.all(values == 0.0)) and matches == p.shape[0],
              "max=%r" % float(values.max()))

        # -- N3/N4 self-compare ---------------------------------------------
        a = os.path.join(tmp, "a")
        b = os.path.join(tmp, "b")
        fixtures.build_dataset(a, seed=1)
        shutil.copytree(a, b)
        out = os.path.join(tmp, "sc")
        receipt = dscompare.compare(a, b, out, {"self_compare": True, "vocab_chunk": 8})
        zeros_ok = (
            receipt["comparison_kind"] == "reproduction_confirmation"
            and receipt["metric"]["value"] == 0.0
            and receipt["top1_agreement"] == 1.0
            and all(v == 0.0 for v in receipt["kl"].values())
            and all(not math.copysign(1, c["max"]) < 0 for c in receipt["per_context"])
            and receipt["comparator"]["short_circuited"] is True
        )
        check("N3  A == B by digest -> exactly 0.0, top-1 1.0, every per-window max +0.0",
              zeros_ok, json.dumps(receipt["kl"]))
        report = dsvalidate.validate_receipt(receipt)
        check("N3b the reproduction receipt passes its own schema and SC-1 rules",
              report.passed, json.dumps(report.errors[:3]))

        out2 = os.path.join(tmp, "sc-forced")
        forced = dscompare.compare(a, b, out2, {"self_compare": True, "force_compute": True,
                                                "vocab_chunk": 8})
        check("N4  --force-compute agrees bitwise with the hash proof",
              forced["self_compare"]["force_compute_agreed"] is True
              and forced["metric"]["value"] == 0.0
              and forced["comparator"]["estimator_backend"] is not None,
              json.dumps(forced["self_compare"]))

        # -- N5 the T1 constant ---------------------------------------------
        path = os.path.join(tmp, "tokenwise-51175.npy")
        meta = dscompare.save_tokenwise(path, dscompare.zero_tokenwise(51175))
        check("N5  51,175 float64 zeros -> 409,528 bytes and the published sha256",
              meta["bytes"] == F.ZERO_TOKENWISE_BYTES_51175
              and meta["sha256"] == F.ZERO_TOKENWISE_SHA256_51175,
              "%d bytes, %s" % (meta["bytes"], meta["sha256"][:16]))

        # -- N6 run-to-run floor --------------------------------------------
        c = os.path.join(tmp, "c")
        fixtures.build_dataset(c, seed=2)  # same weights identity, different content
        out3 = os.path.join(tmp, "floor")
        floor = dscompare.compare(a, c, out3, {"vocab_chunk": 8})
        check("N6  same weights, different capture content -> run_to_run_floor, never a reproduction",
              floor["comparison_kind"] == "run_to_run_floor"
              and floor["metric"]["value"] > 0.0
              and floor["self_compare"]["capture_content_digest_equal"] is False,
              floor["comparison_kind"])

        # -- N6b every emitted receipt validates, including the cross_stack and
        #    cross_lane paths whose bias block is only reachable with a flag.
        d = os.path.join(tmp, "d")
        fixtures.build_dataset(d, seed=3, role="quant", quantized=True,
                               stack="stack-b", lane_identity="lane-b", lane="streaming",
                               model_revision="c" * 40, checkpoint_identity="d" * 64)
        cross = dscompare.compare(a, d, os.path.join(tmp, "cross"),
                                  {"vocab_chunk": 8, "allow_cross_lane": True})
        # A REAL head substitution needs a different head, not just a different
        # capture: `d` shares `a`'s head_seed, so HEAD-1a passes and the override
        # does nothing. This one quantizes its own head, which is the case the
        # override exists for.
        e = os.path.join(tmp, "e")
        fixtures.build_dataset(e, seed=5, head_seed=42, role="quant", quantized=True,
                               stack="stack-b", lane_identity="lane-b", lane="streaming",
                               model_revision="9" * 40, checkpoint_identity="8" * 64)
        head_sub = dscompare.compare(
            a, e, os.path.join(tmp, "cross2"),
            {"vocab_chunk": 8, "allow_cross_lane": True,
             "disclose_head_substitution": True})
        problems = []
        for label, candidate_receipt in (("same_stack", receipt), ("floor", floor),
                                         ("cross_stack+cross_lane", cross),
                                         ("head_substituted", head_sub)):
            report = dsvalidate.validate_receipt(candidate_receipt, label)
            if report.errors:
                problems.append("%s: %s" % (label, report.errors[0]["message"][:70]))
        check("N6b every emitted receipt validates against its own schema, including the "
              "cross_stack / cross_lane / head-substituted paths",
              not problems, "; ".join(problems))
        check("N6c a cross-lane receipt is stamped usable_as_floor false (BIAS-006) and "
              "carries the registry's own bias kind",
              cross["comparability"]["usable_as_floor"] is False
              and cross["comparability"]["bias"]["kind"] == "cross_stack_capture_replay",
              json.dumps(cross["comparability"]["bias"])[:120])

        # -- N7 vocab-chunk invariance --------------------------------------
        out4 = os.path.join(tmp, "chunk4")
        out16 = os.path.join(tmp, "chunk16")
        r4 = dscompare.compare(a, c, out4, {"vocab_chunk": 4})
        r16 = dscompare.compare(a, c, out16, {"vocab_chunk": 16})
        delta = abs(r4["metric"]["value"] - r16["metric"]["value"])
        check("N7  vocab-chunk invariance: two chunk sizes agree to < 1e-12",
              delta < 1e-12, "delta = %.3e" % delta)

        # -- N7b a non-positive --chunk-positions (CLI-05) -------------------
        # Pre-fix this did NOT refuse: range(0, n, -5) is empty, the per-position loop
        # never ran, and the np.empty buffer under it was published as the headline
        # metric, the per_context means and tokenwise-kld.npy -- straight from
        # uninitialized heap, with both artifacts already written to disk.
        for bad_block in (-5, 0):
            outb = os.path.join(tmp, "posblock%s" % bad_block)
            try:
                r = dscompare.compare(a, c, outb, {"position_block": bad_block})
                check("N7b --chunk-positions %d is refused, not published from "
                      "uninitialized memory" % bad_block, False,
                      "returned metric %r" % r["metric"]["value"])
            except dscompare.Refusal as exc:
                wrote = os.path.isdir(outb) and sorted(os.listdir(outb))
                check("N7b --chunk-positions %d is refused before anything is written"
                      % bad_block,
                      exc.code == "bad_position_block" and not wrote,
                      "%s; output dir %s" % (exc.code, wrote or "not created"))
        # a positive block still agrees with the default: the guard is a bound, not a
        # change to the estimator.
        outp = os.path.join(tmp, "posblock_ok")
        rp = dscompare.compare(a, c, outp, {"position_block": 3})
        rdef = dscompare.compare(a, c, os.path.join(tmp, "posblock_def"), {})
        check("N7c a valid --chunk-positions is unchanged by the bound",
              abs(rp["metric"]["value"] - rdef["metric"]["value"]) < 1e-12,
              "%.12f vs %.12f" % (rp["metric"]["value"], rdef["metric"]["value"]))

        # -- N8 bad vocab chunk ---------------------------------------------
        try:
            dscompare.compare(a, c, os.path.join(tmp, "bad"), {"vocab_chunk": 7})
            check("N8  a --vocab-chunk that does not divide vocab_size is refused", False,
                  "no refusal")
        except dscompare.Refusal as exc:
            check("N8  a --vocab-chunk that does not divide vocab_size is refused, with a hint",
                  exc.code == "bad_vocab_chunk" and "working values" in exc.message,
                  exc.message[:90])
        check("N8b the divisor hint for GLM-5.3-Flash names 9680, not kimi-k3's 10240",
              154880 % 10240 != 0 and 9680 in F.divisors_hint(154880, limit=12))

        # -- N9 NaN -> hard refusal -----------------------------------------
        bad_p = p.copy()
        bad_p[0, 0] = np.nan
        try:
            dscompare.token_kld(bad_p, q, "cpu")
            check("N9  a NaN in a capture is a hard refusal, never a clamp", False,
                  "no refusal")
        except (dscompare.Refusal, Exception) as exc:
            check("N9  a NaN in a capture is a hard refusal, never a clamp",
                  "finite" in str(exc).lower(), str(exc)[:110])

        # -- N10 permuted head ----------------------------------------------
        ref = dscompare.load_dataset(a)
        cand = dscompare.load_dataset(b)
        gates, findings = dscompare.run_gates(ref, cand, {})
        head_path = ref.head_path()
        head = dscompare.load_tensor(head_path, "lm_head.weight")
        permuted = head[::-1].copy()
        hidden = dscompare.load_tensor(ref.record_path(ref.records[0]), "hidden_states")
        straight = hidden @ np.ascontiguousarray(head.T)
        crooked = hidden @ np.ascontiguousarray(permuted.T)
        big, _, _ = dscompare.token_kld(straight, crooked, "cpu")
        small, _, _ = dscompare.token_kld(straight, straight, "cpu")
        check("N10 a permuted head at replay produces a large KLD (the estimator has teeth)",
              float(big.mean()) > 1.0 and float(small.mean()) == 0.0,
              "permuted mean = %.4f" % float(big.mean()))

        # -- N11 SC-3 --------------------------------------------------------
        try:
            dscompare.emit_submission(
                receipt, os.path.join(tmp, "submission.json"),
                measurer={"name": "selftest", "handle": "selftest", "url": None,
                          "is_artifact_author": False},
                artifact={}, panel={}, reference={})
            check("N11 a reproduction-confirmation receipt is refused a submission (SC-3)",
                  False, "no refusal")
        except dscompare.NotAMeasurement as exc:
            check("N11 a reproduction-confirmation receipt is refused a submission (SC-3)",
                  "reproduction_confirmation" in str(exc), str(exc)[:80])
        try:
            dscompare.emit_submission(
                floor, os.path.join(tmp, "submission2.json"),
                measurer={}, artifact={}, panel={}, reference={})
            check("N11b a run_to_run_floor receipt is refused a submission too", False,
                  "no refusal")
        except dscompare.NotAMeasurement:
            check("N11b a run_to_run_floor receipt is refused a submission too", True)

        # -- N12 HEAD-1c: the head trap, closed -------------------------------
        # Identical hiddens, different head. A head-only quant (stock EXL3
        # head_bits 6-8) changes nothing before the final norm, so its capture
        # content digest MATCHES the reference's -- and replaying both through
        # one head would erase the only difference there is and report 0.0.
        head_only = os.path.join(tmp, "head-only")
        fixtures.build_dataset(head_only, seed=1, head_seed=99, role="quant",
                               quantized=True, model_revision="e" * 40,
                               checkpoint_identity="f" * 64)
        same_content = (dscompare.load_dataset(a).content_digest
                        == dscompare.load_dataset(head_only).content_digest)
        try:
            dscompare.compare(a, head_only, os.path.join(tmp, "headonly"),
                              {"vocab_chunk": 8, "disclose_head_substitution": True})
            check("N12 a head-only quant is REFUSED, not scored 0.0 (HEAD-1c)", False,
                  "no refusal: it would have reported an exact reproduction")
        except dscompare.Refusal as exc:
            check("N12 a head-only quant is REFUSED, not scored 0.0 (HEAD-1c)",
                  same_content and exc.code == "head_substitution_vacuous"
                  and exc.override is None,
                  "content_equal=%s code=%s" % (same_content, exc.code))

        # -- N13 PANEL-D6: the tokenizer is panel identity --------------------
        other_tok = os.path.join(tmp, "other-tokenizer")
        fixtures.build_dataset(
            other_tok, seed=3, role="quant", quantized=True,
            model_revision="c" * 40, checkpoint_identity="d" * 64,
            tokenizer={"id": "a-completely-different-tokenizer",
                       "repository": "evil/tok", "revision": "9" * 40,
                       "vocab_size": fixtures.VOCAB, "add_special_tokens": True,
                       "chat_template_applied": True})
        try:
            dscompare.compare(a, other_tok, os.path.join(tmp, "tok"), {"vocab_chunk": 8})
            check("N13 a different tokenizer is refused; token ids cannot see it (PANEL-D6)",
                  False, "no refusal")
        except dscompare.Refusal as exc:
            check("N13 a different tokenizer is refused; token ids cannot see it (PANEL-D6)",
                  exc.code == "panel_mismatch" and "PANEL-D6" in exc.message
                  and exc.remedy, exc.message[:90])

        # -- N14/N15 the submission's two structural refusals -----------------
        # A realistic pair: both sides declare a panel far larger than the shard
        # they captured, so the row is a SUBSET of the registry panel it names --
        # which is what a quant author scoring a shard actually produces, and what
        # the registry's own scope check (`covers_full_panel is true but N of M
        # positions were scored`) exists to police.
        sa, sd = os.path.join(tmp, "sa"), os.path.join(tmp, "sd")
        fixtures.build_dataset(sa, seed=1, declared_records=64,
                               subset_detail="6 of 2047 panel positions (selftest shard)")
        fixtures.build_dataset(sd, seed=3, role="quant", quantized=True, stack="stack-b",
                               lane_identity="lane-b", declared_records=64,
                               subset_detail="6 of 2047 panel positions (selftest shard)",
                               model_revision="c" * 40, checkpoint_identity="d" * 64)
        measurement = dscompare.compare(sa, sd, os.path.join(tmp, "meas"),
                                        {"vocab_chunk": 8, "allow_partial": True})
        try:
            dscompare.emit_submission(
                measurement, os.path.join(tmp, "s-empty.json"),
                measurer={"name": "selftest", "handle": "selftest", "url": None,
                          "is_artifact_author": False},
                artifact={}, panel={}, reference={})
            check("N14 --emit-submission with empty provenance REFUSES (SC-4)", False,
                  "wrote a file the registry gate would reject")
        except dscompare.MissingProvenance as exc:
            check("N14 --emit-submission with empty provenance REFUSES (SC-4)",
                  "artifact.repository" in str(exc) and "panel.panel_ref" in str(exc),
                  str(exc)[:90])
        try:
            dscompare.emit_submission(
                head_sub, os.path.join(tmp, "s-blocking.json"),
                measurer=SUBMISSION_PROVENANCE["measurer"],
                artifact=SUBMISSION_PROVENANCE["artifact"],
                panel=SUBMISSION_PROVENANCE["panel"],
                reference=SUBMISSION_PROVENANCE["reference"])
            check("N15 a BLOCKING disclosure refuses a submission bin-side (SC-5)", False,
                  "the registry's DISC-003 never runs on a submission, so nothing would stop it")
        except dscompare.NotAMeasurement as exc:
            check("N15 a BLOCKING disclosure refuses a submission bin-side (SC-5)",
                  "head_substituted" in str(exc), str(exc)[:90])

        # -- N16 the registry's OWN gate, on our own output -------------------
        # The registry requires the file to sit in a directory named after the
        # measurer's handle, so credit cannot be misfiled -- so the selftest
        # files it the way a contributor would.
        os.makedirs(os.path.join(tmp, "selftest"), exist_ok=True)
        submission_path = os.path.join(tmp, "selftest", "submission-receipt.json")
        submission = dscompare.emit_submission(
            measurement, submission_path,
            measurer=SUBMISSION_PROVENANCE["measurer"],
            artifact=SUBMISSION_PROVENANCE["artifact"],
            panel=SUBMISSION_PROVENANCE["panel"],
            reference=SUBMISSION_PROVENANCE["reference"])
        validator = os.path.join(REPO, "registry", "tools", "registry_validate.py")
        proc = subprocess.run([sys.executable, validator, "--submission", submission_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        check("N16 the emitted submission is ACCEPTED by registry_validate.py --submission",
              proc.returncode == 0,
              (proc.stdout or "").strip().splitlines()[0] if proc.stdout else "no output")

        # -- N16b the bias verdict survives the crossing ----------------------
        bias = (submission.get("comparability") or {}).get("bias")
        check("N16b comparability.bias and usable_as_floor reach the submission",
              bias is not None and bias["kind"] == "cross_stack_capture_replay"
              and (submission["comparability"]["usable_as_floor"] is False),
              json.dumps(submission.get("comparability"))[:110])
        check("N16c the submission's determinism block carries no receipt-only keys",
              not ({"min_run_mean", "max_run_mean", "population_stddev_of_run_means"}
                   & set(submission["determinism"])),
              ", ".join(sorted(submission["determinism"])))
        check("N16d evidence names each dataset by its SEAL, with a legal source kind",
              all(item["kind"] in ("hf_file", "filesystem_path") and item.get("uri")
                  and item.get("sha256") for item in submission["evidence"])
              and {item["sha256"] for item in submission["evidence"]}
              == {measurement["reference"]["dataset_sha256"],
                  measurement["candidate"]["dataset_sha256"]},
              json.dumps(submission["evidence"])[:120])

        # -- N17 a tampered tensor, resealed honestly, is caught by DEFAULT ----
        tampered = os.path.join(tmp, "tampered")
        shutil.copytree(a, tampered)
        victim = os.path.join(tampered, "capture", "hidden_0000.safetensors")
        with open(victim, "r+b") as handle:
            handle.seek(os.path.getsize(victim) - 2)
            byte = handle.read(1)
            handle.seek(os.path.getsize(victim) - 2)
            handle.write(bytes([byte[0] ^ 0x01]))
        F.write_checksums(tampered)                       # refreshed, as an author would
        manifest = F.load_manifest(tampered)
        manifest["seal"]["checksums_sha256"] = F.sha256_file(
            os.path.join(tampered, "checksums.txt"))
        F.write_json(os.path.join(tampered, "fidelity-dataset.json"),
                     F.seal_manifest(dict(manifest, dataset_sha256="")))
        F.write_checksums(tampered)
        try:
            dscompare.compare(a, tampered, os.path.join(tmp, "tamper"), {"vocab_chunk": 8})
            check("N17 a tampered tensor with refreshed checksums is refused by default",
                  False, "scored it anyway")
        except dscompare.Refusal as exc:
            check("N17 a tampered tensor with refreshed checksums is refused by default",
                  exc.code == "seal_failed" and "tensor" in exc.message.lower(),
                  exc.message[:100])
        # -- N18/N19 the cache collision that returned a silent, wrong 0.0 ----
        # `compare --reference hf://A --candidate hf://B --cache DIR` fetched A
        # into DIR, fetched B on top of it, and compared B with B: exit 0,
        # "REPRODUCTION CONFIRMATION", 0.0 nats, class=strict,
        # usable_as_floor=true, both sides of the receipt naming ONE
        # dataset_sha256. Found on the first two real published datasets.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fidelity_dataset                                     # noqa: WPS433

        left = fidelity_dataset.cache_path("/tmp/cache", "owner/root-v1", "main")
        right = fidelity_dataset.cache_path("/tmp/cache", "owner/quant-v1", "main")
        same_repo_other_rev = fidelity_dataset.cache_path("/tmp/cache", "owner/root-v1", "abc")
        check("N18 an explicit --cache still nests per repo AND revision",
              left != right and left != same_repo_other_rev
              and left.startswith("/tmp/cache" + os.sep),
              "%s vs %s vs %s" % (left, right, same_repo_other_rev))

        class _Args(object):
            reference = candidate = out = None
            allow_partial = self_compare = force_compute = False
            allow_cross_lane = disclose_head_substitution = False
            verify_tensors = True
            device = "cpu"
            vocab_chunk = chunk_positions = None
            head = json = cache = token_file = None
            emit_submission = False
            submission_provenance = measurer = None
            reference_label = candidate_label = None

        args = _Args()
        args.reference = args.candidate = a
        args.out = os.path.join(tmp, "same-root")
        check("N19 one directory passed as BOTH sides is refused, not scored as 0.0",
              fidelity_dataset.cmd_compare(args) == 3)

        # -- N20 P1-09: a DISJOINT pair is refused and leaves NO files --------
        # With --allow-partial, two captures sharing no context index used to
        # pass the differing-set coverage gate, reduce over nothing, and WRITE a
        # metric=0.0 / positions=0 / contexts=0 receipt before the CLI's later
        # validation rejected it -- a perfect-fidelity artifact about nothing,
        # left on disk for any library caller to pick up.
        d1, d2 = os.path.join(tmp, "dj-a"), os.path.join(tmp, "dj-b")
        fixtures.build_dataset(d1, seed=1, records=6, capture_indices=[0, 1, 2],
                               declared_records=6, subset_detail="shard 0-2")
        fixtures.build_dataset(d2, seed=3, role="quant", quantized=True, stack="stack-b",
                               lane_identity="lane-b", model_revision="c" * 40,
                               checkpoint_identity="d" * 64,
                               records=6, capture_indices=[3, 4, 5],
                               declared_records=6, subset_detail="shard 3-5")
        dj_out = os.path.join(tmp, "dj-out")
        try:
            dscompare.compare(d1, d2, dj_out, {"vocab_chunk": 8, "allow_partial": True})
            check("N20 disjoint index sets are refused, never scored (P1-09)", False,
                  "no refusal: an empty comparison went through")
        except dscompare.Refusal as exc:
            check("N20 disjoint index sets are refused, never scored (P1-09)",
                  exc.code == "empty_intersection" and exc.override is None,
                  "code=%s override=%r" % (exc.code, exc.override))
        check("N20b the refusal leaves NO files behind",
              not os.path.exists(dj_out) or not os.listdir(dj_out),
              repr(os.listdir(dj_out) if os.path.exists(dj_out) else []))
        check("N20c no staging directory survives the refusal",
              not [n for n in os.listdir(tmp) if n.startswith(".compare-staging-")])

        # -- N21 P1-09: outputs are staged and only published after validation.
        # A good comparison must still land both files atomically in out_dir.
        good_out = os.path.join(tmp, "publish-out")
        dscompare.compare(sa, sd, good_out, {"vocab_chunk": 8, "allow_partial": True})
        check("N21 a valid comparison publishes receipt + array and no staging leftovers",
              sorted(os.listdir(good_out)) == ["comparison-receipt.json", "tokenwise-kld.npy"]
              and not [n for n in os.listdir(tmp) if n.startswith(".compare-staging-")],
              repr(sorted(os.listdir(good_out))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nselftest_fidelity_compare: %d passed, %d failed" % (len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s  %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
