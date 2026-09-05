#!/usr/bin/env python3
"""T12 -- the replay backend: where the head matmul runs, and what that costs.

M1 measured the reason this file exists.  One 512-window Qwen3.8-27B comparison
took 60m19s against a 335s capture of the same panel on the same box -- 10.8x
the capture it consumes -- because `dscompare._replay` did the language-model
head matmul in numpy on the CPU while the GPU, already holding that head for
the fp64 KLD step, sat at 0%.  M2 is a 642.7 GB re-capture and M3 is 1.5 TB; at
those sizes the capture is a rounding error and the comparisons are the bill.

The fix is `--replay-device`.  The hazard is that an fp32 GEMM is not one
function: BLAS accumulates in fp32 in an order the implementation's blocking
chooses, so the same head and the same hidden states give different last bits
on Accelerate, on OpenBLAS and on cuBLAS.  So the numpy path stays the default,
the backend is named on every receipt, and these rungs are what stops a silent
swap.

    R1  the default is numpy, and the receipt says so on every comparison
    R2  a torch replay device is accepted and reaches the receipt by name
    R3  THE FLOOR IS BACKEND-INDEPENDENT: a --force-compute self-compare is
        exactly 0.0 through BOTH backends, and both write the SAME
        tokenwise-kld.npy digest.  This is the anchor the published Qwen3.8
        floor (0.0, digest 8be5dcca...) rests on, and it is the one property a
        backend change cannot be allowed to move.
    R4  a real A-vs-B measurement through both backends: the delta is REPORTED,
        not assumed, and is bounded by the fp32-accumulation term rather than
        by faith.  On a backend pair that happens to agree bitwise this is 0.0;
        on one that does not, the test still passes and prints the delta,
        because the point of the rung is that the number is measured.
    R5  chunking is value-preserving on the torch path too: --vocab-chunk and
        --chunk-positions partition the OUTPUT, never a reduction.
    R6  --replay-device on a device the estimator does not use is refused: the
        logits would cross the bus twice per block, which is slower than the
        numpy path it replaces.
    R7  --replay-dtype is validated, and float64 is offered as a DIFFERENT
        measurement rather than a more correct spelling of the same one.
    R8  mixed hidden<->logit form works on the torch path (only one side is
        replayed; the other is moved, not recomputed).

numpy-only interpreters run R1 and R6-R7 and SKIP the rest, loudly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

BIN = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BIN)
sys.path.insert(0, BIN)

import numpy as np  # noqa: E402

from fidelity import dscompare  # noqa: E402
from fidelity import dsformat as F  # noqa: E402

import selftest_fidelity_dataset as fixtures  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print("  PASS  %s" % name)
    else:
        FAIL.append((name, detail))
        print("  FAIL  %s%s" % (name, ("  -- " + detail) if detail else ""))


def skip(name, why):
    SKIP.append(name)
    print("  SKIP  %s (%s)" % (name, why))


def torch_devices():
    """Every torch device this box can actually replay on, best first."""
    try:
        import torch
    except Exception:
        return []
    found = []
    if getattr(torch.backends, "cuda", None) and torch.cuda.is_available():
        found.append("cuda")
    # MPS is deliberately NOT offered: the estimator accumulates in float64 and
    # Metal has no float64, so a replay there could only be consumed by a
    # different estimator, which is a different measurement.
    found.append("cpu")
    return found


def digest_of(out_dir):
    return F.sha256_file(os.path.join(out_dir, "tokenwise-kld.npy"))


def body(tmp):
    a = os.path.join(tmp, "a")
    b = os.path.join(tmp, "b")
    c = os.path.join(tmp, "c")
    logits = os.path.join(tmp, "logit")
    fixtures.build_dataset(a, seed=1)
    fixtures.build_dataset(b, seed=1)                    # same content -> SC-1
    fixtures.build_dataset(c, seed=2)                    # different -> measurement
    fixtures.build_dataset(logits, seed=2, form="logit")

    # -- R1 -----------------------------------------------------------------
    out = os.path.join(tmp, "default")
    receipt = dscompare.compare(a, c, out, {})
    named = receipt["comparator"].get("replay_backend")
    check("R1  the default replay backend is numpy and the receipt names it",
          named == "numpy:cpu:float32",
          "comparator.replay_backend = %r" % named)

    devices = torch_devices()
    if not devices:
        skip("R2-R5, R8, R10", "torch not importable")
    else:
        device = devices[0]
        # -- R2 -------------------------------------------------------------
        out2 = os.path.join(tmp, "torch")
        r2 = dscompare.compare(a, c, out2,
                               {"device": device, "replay_device": device})
        want = "torch:%s:float32" % device
        check("R2  --replay-device %s reaches the receipt by name" % device,
              r2["comparator"].get("replay_backend") == want,
              "replay_backend = %r" % r2["comparator"].get("replay_backend"))

        # -- R3 -- THE ANCHOR ------------------------------------------------
        # A self-compare replays bitwise-equal hidden states through one head on
        # ONE backend, so both sides get bitwise-equal logits and the KLD is
        # exactly 0.0 whatever the backend rounds to.  The published Qwen3.8
        # floor is exactly this, at 1,048,064 positions, with tokenwise digest
        # 8be5dcca...; if a backend change could move THAT, the fix would be
        # unshippable at any speed.
        sc_np = os.path.join(tmp, "sc-numpy")
        sc_t = os.path.join(tmp, "sc-torch")
        f_np = dscompare.compare(a, b, sc_np, {"self_compare": True,
                                               "force_compute": True})
        f_t = dscompare.compare(a, b, sc_t, {"self_compare": True,
                                             "force_compute": True,
                                             "device": device,
                                             "replay_device": device})
        check("R3  --force-compute self-compare is EXACTLY 0.0 through both backends, "
              "and both write the same tokenwise digest",
              f_np["metric"]["value"] == 0.0 and f_t["metric"]["value"] == 0.0
              and max(f_np["kl"].values()) == 0.0 and max(f_t["kl"].values()) == 0.0
              and f_np["top1_agreement"] == 1.0 and f_t["top1_agreement"] == 1.0
              and digest_of(sc_np) == digest_of(sc_t)
              and f_t["comparator"].get("replay_backend") == want,
              "numpy %r / torch %r; digests %s vs %s"
              % (f_np["metric"]["value"], f_t["metric"]["value"],
                 digest_of(sc_np)[:12], digest_of(sc_t)[:12]))

        # -- R4 -- the delta, measured -----------------------------------------
        # Not "assert equal": asserting bit-equality across two GEMM
        # implementations would be asserting a property neither BLAS promises,
        # and a green test that says something false is worse than a number.
        delta = abs(r2["metric"]["value"] - receipt["metric"]["value"])
        rel = delta / abs(receipt["metric"]["value"]) if receipt["metric"]["value"] else 0.0
        agree = 17 if delta == 0.0 else max(0, int(-np.log10(rel))) if rel else 17
        print("        replay-backend delta on a real measurement: "
              "numpy %.17g vs %s %.17g -> abs %.3e, rel %.3e (~%d significant digits agree)"
              % (receipt["metric"]["value"], want, r2["metric"]["value"], delta, rel, agree))
        # The bound: fp32 accumulation noise on the logits is ~1e-5 absolute and
        # enters the KLD linearly with random sign over ~1e4 positions here, so
        # anything above 1e-4 RELATIVE is not rounding, it is a broken replay.
        check("R4  the two backends agree to within the fp32-accumulation term "
              "(and the delta is printed, not assumed)",
              rel < 1e-4 and r2["top1_agreement"] == receipt["top1_agreement"],
              "rel=%.3e top1 %r vs %r"
              % (rel, r2["top1_agreement"], receipt["top1_agreement"]))

        # -- R5 -------------------------------------------------------------
        # Chunking splits the OUTPUT (positions, vocabulary); every output
        # element is an independent dot product over the hidden axis, so no
        # reduction is touched and the values must be identical, not close.
        vc = dscompare.compare(a, c, os.path.join(tmp, "torch-vc"),
                               {"device": device, "replay_device": device,
                                "vocab_chunk": 4})
        pb = dscompare.compare(a, c, os.path.join(tmp, "torch-pb"),
                               {"device": device, "replay_device": device,
                                "position_block": 3})
        check("R5  on the torch path, --vocab-chunk and --chunk-positions leave the "
              "value bitwise unchanged",
              vc["metric"]["value"] == r2["metric"]["value"]
              and pb["metric"]["value"] == r2["metric"]["value"],
              "base %.17g vocab_chunk %.17g position_block %.17g"
              % (r2["metric"]["value"], vc["metric"]["value"], pb["metric"]["value"]))

        # -- R8 -------------------------------------------------------------
        # HEAD-3: one side hidden, one side logit. Only the hidden side is
        # replayed; the logit side is moved to the device, never recomputed.
        mixed_np = dscompare.compare(a, logits, os.path.join(tmp, "mixed-np"), {})
        mixed_t = dscompare.compare(a, logits, os.path.join(tmp, "mixed-torch"),
                                    {"device": device, "replay_device": device})
        mrel = (abs(mixed_t["metric"]["value"] - mixed_np["metric"]["value"])
                / abs(mixed_np["metric"]["value"]))
        check("R8  mixed hidden<->logit replays only the hidden side on the device",
              mixed_t["comparator"].get("replay_backend") == want and mrel < 1e-4,
              "numpy %.17g torch %.17g rel %.3e"
              % (mixed_np["metric"]["value"], mixed_t["metric"]["value"], mrel))

        # -- R10 -- HEAD-1d on the device path --------------------------------
        # Each hidden-form side through ITS OWN sealed head, replayed on the
        # torch device: two resident heads, one per side, and the answer must
        # be the numpy own-head answer to within the fp32-accumulation term.
        other = os.path.join(tmp, "other-head")
        fixtures.build_dataset(other, seed=2, head_seed=99, role="quant", quantized=True)
        own_np = dscompare.compare(a, other, os.path.join(tmp, "own-np"), {"own_heads": True})
        own_t = dscompare.compare(a, other, os.path.join(tmp, "own-torch"),
                                  {"own_heads": True, "device": device, "replay_device": device})
        orel = (abs(own_t["metric"]["value"] - own_np["metric"]["value"])
                / abs(own_np["metric"]["value"]))
        ca, ct = own_np["comparator"], own_t["comparator"]
        check("R10 --own-heads on the device path replays each side through its own head",
              ct.get("replay_backend") == want and orel < 1e-4
              and own_t["estimator"]["head_policy"] == "native_head"
              and ct["head_applied_tensor_content_sha256"] is None
              and ct["head_applied_reference_tensor_content_sha256"]
              == ca["head_applied_reference_tensor_content_sha256"]
              and ct["head_applied_candidate_tensor_content_sha256"]
              == ca["head_applied_candidate_tensor_content_sha256"]
              and ct["head_applied_reference_tensor_content_sha256"]
              != ct["head_applied_candidate_tensor_content_sha256"],
              "numpy %.17g torch %.17g rel %.3e"
              % (own_np["metric"]["value"], own_t["metric"]["value"], orel))

    # -- R6 -----------------------------------------------------------------
    try:
        dscompare.compare(a, c, os.path.join(tmp, "split"),
                          {"device": "cpu", "replay_device": "cuda"})
        check("R6  a replay device the estimator does not use is refused", False,
              "no refusal")
    except dscompare.Refusal as exc:
        check("R6  a replay device the estimator does not use is refused",
              exc.code in ("replay_device_mismatch", "replay_backend_unavailable"),
              "%s: %s" % (exc.code, exc.message[:80]))

    # -- R7 -----------------------------------------------------------------
    try:
        dscompare.compare(a, c, os.path.join(tmp, "dtype"),
                          {"device": "cpu", "replay_device": "cpu",
                           "replay_dtype": "bfloat16"})
        check("R7  --replay-dtype is validated", False, "no refusal")
    except dscompare.Refusal as exc:
        check("R7  --replay-dtype is validated",
              exc.code in ("bad_replay_dtype", "replay_backend_unavailable"),
              "%s" % exc.code)

    if devices:
        f64 = dscompare.compare(a, c, os.path.join(tmp, "f64"),
                                {"device": devices[0], "replay_device": devices[0],
                                 "replay_dtype": "float64"})
        check("R7b --replay-dtype float64 is a DIFFERENT measurement, and says so",
              f64["comparator"].get("replay_backend") == "torch:%s:float64" % devices[0],
              "replay_backend = %r" % f64["comparator"].get("replay_backend"))
        print("        float64 replay: %.17g (fp32 replay %.17g)"
              % (f64["metric"]["value"], receipt["metric"]["value"]))

    # The CLI surface, not just the library: a flag nobody can reach is not a
    # fix. Driven through the real entrypoint on the default backend.
    import subprocess

    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "bin", "fidelity_dataset.py"), "compare",
         "--reference", a, "--candidate", c, "--out", os.path.join(tmp, "cli"),
         "--replay-device", "numpy", "--json", os.path.join(tmp, "cli.json")],
        capture_output=True, text=True)
    doc = {}
    if os.path.isfile(os.path.join(tmp, "cli", "comparison-receipt.json")):
        doc = json.load(open(os.path.join(tmp, "cli", "comparison-receipt.json")))
    check("R9  `fidelity-dataset compare --replay-device` exists and lands in the receipt",
          proc.returncode in (0, 2)
          and doc.get("comparator", {}).get("replay_backend") == "numpy:cpu:float32",
          "rc=%s backend=%r err=%s"
          % (proc.returncode, doc.get("comparator", {}).get("replay_backend"),
             (proc.stderr or "")[-200:]))


def main():
    tmp = tempfile.mkdtemp(prefix="replaydev-")
    try:
        body(tmp)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIP)))
    for name, detail in FAIL:
        print("  FAILED %s: %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
