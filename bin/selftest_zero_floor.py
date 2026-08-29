#!/usr/bin/env python3
"""T4 -- the zero-floor identity: identical logits score EXACTLY 0.0.

    python3 bin/selftest_zero_floor.py            # numpy half
    FIDELITY_PYTHON=... bin/selftest_zero_floor.py  # + torch half

Two rungs, each SKIPping (not failing) when its dependency is absent:

  [1] numpy:  np.save of 51,175 float64 zeros has the FIXED sha256
      3ffddc61...be17 (409,528 bytes).  This is the tokenwise-kld.npy any
      T1-identical native run produces, so the constant is the cheapest
      possible future check that a claimed zero floor is bitwise real.
  [2] torch:  a tiny identical-logits teacher/student capture pair pushed
      through bin/kld_preview census mode yields tokenwise +0.0 at EVERY
      position and a panel mean of exactly 0.0 -- not epsilon: the fp32 store
      rounding cancels bit-for-bit and fp64 log_softmax of bitwise-equal
      inputs is deterministic, so (teacher_logp - student_logp) is a
      subtraction of equal doubles.

Exit 0 on PASS or SKIP (reason printed); 1 on FAIL.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ZEROS_SHA256 = "3ffddc61af8350782afd24c7a69de1f37c260bf5489c4e0f6e3ad89b0ab9be17"
ZEROS_BYTES = 409528

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def skip(name, reason):
    SKIP.append((name, reason))
    print("  skip " + name + "  -- " + reason)


def rung_numpy() -> None:
    try:
        import numpy as np
    except ImportError:
        skip("zeros-npy identity", "numpy not installed for this interpreter")
        return
    buf = io.BytesIO()
    np.save(buf, np.zeros(51175, dtype=np.float64), allow_pickle=False)
    blob = buf.getvalue()
    check("np.save of 51,175 float64 zeros is %d bytes" % ZEROS_BYTES,
          len(blob) == ZEROS_BYTES, len(blob))
    check("...and hashes to the fixed constant 3ffddc61...be17",
          hashlib.sha256(blob).hexdigest() == ZEROS_SHA256,
          hashlib.sha256(blob).hexdigest()[:16])


def _write_capture(root: Path, role: str, schema: str, logits, panel_sha: str,
                   label: str) -> None:
    import torch
    from safetensors.torch import save_file

    from fidelity.common import canonical_json, sha256_hex, sha256_file

    (root / "logits").mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(logits.shape[0]):
        path = (root / "logits" / ("window-%04d.safetensors" % index)).resolve()
        save_file({"logits": logits[index].clone().contiguous()}, str(path))
        rows.append({
            "window_id": "final-%04d" % index,
            "document_id": "doc-%02d" % index,
            "domain": "prose",
            "role": "final",
            "token_ids_sha256": "2" * 64,
            "attention_mask_sha256": "3" * 64,
            "prediction_positions": int(logits.shape[1]),
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        })
    receipt = {
        "schema": schema,
        "capture_role": role,
        "model_revision": "a" * 40,
        "token_panel_receipt_sha256": panel_sha,
        "backend_identity_sha256": sha256_hex(role),
        "logits_dtype": "float32",
        "kld_direction": "teacher_to_student",
        "prediction_positions": int(logits.shape[0] * logits.shape[1]),
        "vocab_size": int(logits.shape[2]),
        "student_label": label,
        "checkpoint_identity_sha256": "4" * 64,
        "runtime_reader_sha256": "5" * 64,
        "logit_files": rows,
    }
    receipt["receipt_sha256"] = sha256_hex(canonical_json(receipt))
    (root / "capture-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def rung_torch() -> None:
    try:
        import torch
        import safetensors                               # noqa: F401
    except ImportError as exc:
        skip("identical-logits census -> exact 0.0",
             "torch/safetensors not importable here (%s); run under "
             "FIDELITY_PYTHON" % exc)
        return
    import kld_preview as KP

    torch.manual_seed(0)
    windows, positions, vocab = 25, 5, 11
    logits = (torch.randn(windows, positions, vocab) * 3.0).float()
    with tempfile.TemporaryDirectory() as work:
        base = Path(work)
        _write_capture(base / "teacher", "bf16_teacher",
                       KP.SEALED_CAPTURE_SCHEMA, logits, "e" * 64, "native-bf16")
        _write_capture(base / "student", "native_bf16_student",
                       KP.SEALED_CAPTURE_SCHEMA, logits.clone(), "e" * 64,
                       "native-bf16")
        out = base / "preview.json"
        rc = KP.main(["--teacher", str(base / "teacher"),
                      "--student", str(base / "student"),
                      "--out", str(out)])
        check("kld_preview census on the identical pair exits 0", rc == 0)
        doc = json.loads(out.read_text(encoding="utf-8"))
        check("panel mean is EXACTLY 0.0 (not epsilon)",
              doc.get("preview_panel_mean_estimate") == 0.0,
              repr(doc.get("preview_panel_mean_estimate")))
        check("every per-window mean is exactly 0.0",
              all(w["mean"] == 0.0 for w in doc["per_window"].values()))
        check("every per-window max is +0.0 (no negative-zero surprises)",
              all(w["max"] == 0.0 for w in doc["per_window"].values()))
        check("top-1 agreement is exactly 1.0", doc.get("top1_agreement") == 1.0)
        check("the receipt is a census PREVIEW (lane differs from teacher's), "
              "unsubmittable by shape",
              doc["schema"] == "malaiwah.glm53-census-kld-preview.v1"
              and doc["not_submittable"] is True
              and "measured_mean_kld" not in doc)


def main() -> int:
    print("[1] the fixed zeros constant (numpy)")
    rung_numpy()
    print("\n[2] identical logits through kld_preview census (torch)")
    rung_torch()
    print("\n" + "-" * 72)
    verdict = "FAIL" if FAIL else ("PASS" if PASS else "SKIP")
    print("selftest_zero_floor: %s (%d passed, %d failed, %d skipped)"
          % (verdict, len(PASS), len(FAIL), len(SKIP)))
    for name, detail in FAIL:
        print("  FAILED: %s %s" % (name, detail))
    for name, reason in SKIP:
        print("  SKIPPED: %s -- %s" % (name, reason))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
