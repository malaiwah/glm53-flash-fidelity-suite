#!/usr/bin/env python3
"""T9 -- the stack fingerprint: deterministic, engine-absent-safe, never a guess.

    python3 bin/selftest_stackprint.py

Runs entirely offline on a stock interpreter.  What it proves, on THIS
machine (a Mac without vllm and possibly without torch is exactly the hostile
environment the probes must survive):

  [1] importing stackprint pulls in NO engine: torch/vllm stay unimported.
  [2] collect() is deterministic: two collections hash identically, volatile
      keys (collected_utc, paths, the seal itself) do not participate, and
      any NON-volatile change does.
  [3] engine-absent handling: collect("vllm") on a vllm-less host records a
      probe_error and unqueryable sources instead of raising or guessing.
  [4] declared facts are labeled harness_arg -- never presented as a query.
  [5] CUDA/MPS-absent handling: the gpu block answers with booleans or a
      probe_error, devices list stays consistent, nothing raises.
  [6] write() round trip: the seal in the file verifies against the file's
      own canonical subset, and pip-freeze.txt is the digest's preimage.
  [7] from_backend_json maps a checkpoint-lane backend.json without reading
      THIS process's environment (a finished run must not inherit today's env).
  [8] the whole fingerprint is json-stable (round-trips through json).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import stackprint  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name +
          (("  -- " + str(detail)) if detail else ""))


def main() -> int:
    # [1] import purity: a fresh interpreter that imports stackprint must not
    # have imported torch or vllm as a side effect.
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import stackprint; "
         "print(sorted(m for m in ('torch', 'vllm') if m in sys.modules))"
         % str(Path(__file__).resolve().parent / "fidelity")],
        capture_output=True, text=True)
    check("[1] import touches no engine",
          probe.returncode == 0 and probe.stdout.strip() == "[]",
          probe.stdout.strip() or probe.stderr.strip()[-200:])

    # [2] determinism.
    a = stackprint.collect("none")
    b = stackprint.collect("none")
    sha_a, sha_b = stackprint.fingerprint_sha256(a), stackprint.fingerprint_sha256(b)
    check("[2a] two collections hash identically", sha_a == sha_b,
          "%s vs %s" % (sha_a[:16], sha_b[:16]))
    mutated = dict(stackprint.public_dict(a))
    mutated["collected_utc"] = "1970-01-01T00:00:00Z"
    mutated["paths"] = {"out_dir": "/somewhere/else"}
    mutated["stack_fingerprint_sha256"] = "0" * 64
    check("[2b] volatile keys do not participate in the hash",
          stackprint.fingerprint_sha256(mutated) == sha_a)
    mutated2 = dict(stackprint.public_dict(a))
    mutated2["python"] = "0.0.0"
    check("[2c] a non-volatile change changes the hash",
          stackprint.fingerprint_sha256(mutated2) != sha_a)
    check("[2d] env pin keys are complete and ordered",
          list(a["env_pins"])[:len(stackprint.ENV_PIN_KEYS)] ==
          list(stackprint.ENV_PIN_KEYS))

    # [3] engine-absent vllm: on this Mac vllm is not installed; even if it
    # were, there is no live handle, so nothing may be asserted.
    v = stackprint.collect("vllm")
    have_vllm = v["engine"]["version"] is not None
    check("[3a] engine-absent collect does not raise", True)
    if have_vllm:
        check("[3b] vllm version recorded when importable", True,
              v["engine"]["version"])
    else:
        check("[3b] absent engine records probe_error, never a version",
              v["engine"]["probe_error"] is not None
              and v["engine"]["version"] is None,
              v["engine"]["probe_error"])
    check("[3c] no handle => enforce_eager unknown-with-reason, not a guess",
          v["execution"]["enforce_eager"] is None
          and "unqueryable" in (v["execution"]["enforce_eager_source"] or ""),
          v["execution"]["enforce_eager_source"])
    check("[3d] no handle => attention backend unqueryable, not defaulted",
          v["execution"]["attention_backend"]["selected"] is None
          and "unqueryable" in v["execution"]["attention_backend"]["selected_source"])

    # [4] declared facts carry the harness_arg label.
    d = stackprint.collect("vllm", declared={"enforce_eager": True,
                                             "attention_backend_requested": None})
    check("[4] declared enforce_eager is True/harness_arg",
          d["execution"]["enforce_eager"] is True
          and d["execution"]["enforce_eager_source"] == "harness_arg")

    # [5] CUDA/MPS-absent handling on this machine.
    g = a["gpus"]
    if g["probe_error"] is not None:
        check("[5] torchless host: gpu probe records the error and no devices",
              g["devices"] == [] and g["cuda_available"] is None,
              g["probe_error"])
    else:
        consistent = (isinstance(g["cuda_available"], bool)
                      and isinstance(g["mps_available"], bool)
                      and (g["cuda_available"] or g["devices"] == []))
        check("[5] gpu probe answers with booleans; no CUDA => no devices",
              consistent, json.dumps({k: g[k] for k in
                                      ("cuda_available", "mps_available")}))

    # [6] write() round trip.
    with tempfile.TemporaryDirectory() as tmp:
        public, sha = stackprint.write(stackprint.collect("none", out_dir=tmp), tmp)
        on_disk = json.loads((Path(tmp) / "stack-fingerprint.json").read_text())
        check("[6a] embedded seal verifies against the file itself",
              on_disk["stack_fingerprint_sha256"] == sha
              == stackprint.fingerprint_sha256(on_disk))
        freeze = Path(tmp) / "pip-freeze.txt"
        if on_disk["pip_freeze_sha256"] is None:
            check("[6b] no freeze => digest and file both absent, with reason",
                  not freeze.exists() and on_disk["pip_freeze_error"] is not None)
        else:
            check("[6b] pip-freeze.txt is the digest's preimage",
                  freeze.is_file() and hashlib.sha256(
                      freeze.read_bytes()).hexdigest() == on_disk["pip_freeze_sha256"])
        check("[6c] write() seal equals a fresh collection's seal",
              sha == sha_a)

    # [7] checkpoint-lane adapter on a fixture modeled on the published
    # teacher backend.json (brandonmusic dataset, file 43dd699a...).
    fixture = {
        "schema": "malaiwah.glm53-teacher-backend-identity.v1",
        "model_revision": "b1967181a3917ae70a437f4884748f6b8e3a1f4d",
        "transformers_version": "5.16.1",
        "torch_version": "2.11.0+cu130",
        "cuda_runtime_version": "13.0",
        "device_name": "NVIDIA H200",
        "attention_backend": "eager",
        "experts_implementation": "grouped_mm",
        "grouped_mm_kernel": {"probe": "transformers.integrations.moe._can_use_grouped_mm",
                              "can_use_native_grouped_mm": True,
                              "dispatched_kernel": "torch._grouped_mm"},
        "numeric_policy": {"allow_tf32": False},
        "host": {"platform": "Linux", "machine": "x86_64", "python": "3.12.3"},
    }
    fb1 = stackprint.from_backend_json(fixture)
    fb2 = stackprint.from_backend_json(json.loads(json.dumps(fixture)))
    check("[7a] adapter kind/attention/eager semantics",
          fb1["engine"]["kind"] == "transformers-reference"
          and fb1["execution"]["enforce_eager"] is None
          and fb1["execution"]["enforce_eager_source"] == "not_applicable_reference_lane"
          and fb1["execution"]["attention_backend"]["selected"] == "eager"
          and fb1["execution"]["attention_backend"]["selected_source"].startswith("receipt_field"))
    check("[7b] adapter is deterministic on equal input",
          fb1["stack_fingerprint_sha256"] == fb2["stack_fingerprint_sha256"])
    check("[7c] adapter reads the receipt, not THIS process's env",
          not any(k in fb1["env_pins"] for k in stackprint.ENV_PIN_KEYS)
          and fb1["kernels"]["grouped_mm"] == fixture["grouped_mm_kernel"])

    # [7d] the REAL published teacher backend.json shape (brandonmusic dataset,
    # file 43dd699a...), which carries neither host/device_name nor a
    # numeric_policy block -- it states allow_tf32 alone, and it seals itself
    # as backend_identity_sha256.  The adapter must recover the policy from
    # whichever field holds it and must carry the digest link, otherwise a
    # derived fingerprint floats free of the published chain.
    teacher = {
        "schema": "quant-pipeline.glm53-teacher-backend-identity.v1",
        "architecture": "Glm53FlashForCausalLM",
        "model_revision": "b1967181a3917ae70a437f4884748f6b8e3a1f4d",
        "transformers_version": "5.16.1",
        "torch_version": "2.11.0+cu130",
        "cuda_runtime_version": "13.0",
        "attention_backend": "eager",
        "allow_tf32": False,
        "world_size": 4,
        "parallelism": "expert_parallel_world_size_4_with_replicated_protected_weights",
        "nccl_version": "2.28.9",
        "backend_identity_sha256":
            "85b11599c6b36a83fa8099a09a298a386a0c603d1f18d3702e7fb1c470962ce4",
    }
    tf = stackprint.from_backend_json(teacher)
    check("[7d] real teacher shape: tf32 policy recovered from allow_tf32, "
          "named, and the backend digest is carried",
          tf["kernels"]["numeric_policy"]["allow_tf32"] is False
          and tf["kernels"]["numeric_policy"]["source_field"] == "backend.allow_tf32"
          and tf["kernels"]["parallelism"] == teacher["parallelism"]
          and tf["kernels"]["world_size"] == 4
          and tf["kernels"]["nccl_version"] == "2.28.9"
          and (tf["source_receipt"]["backend_identity_sha256"]
               == teacher["backend_identity_sha256"])
          and tf["execution"]["attention_backend"]["selected"] == "eager",
          tf["kernels"]["numeric_policy"]["source_field"])
    check("[7e] a receipt with no policy field at all yields null, not a guess",
          stackprint.from_backend_json({"schema": "x"})["kernels"]["numeric_policy"]
          is None)

    # [8] json stability of everything produced above.
    for label, doc in (("collect", stackprint.public_dict(a)),
                       ("vllm-absent", stackprint.public_dict(v)),
                       ("adapter", fb1)):
        rt = json.loads(json.dumps(doc, sort_keys=True))
        check("[8] json round trip: %s" % label,
              stackprint.fingerprint_sha256(rt) == stackprint.fingerprint_sha256(doc))

    # invalid engine kind refuses.
    try:
        stackprint.collect("triton-server")
        check("[9] unknown engine kind refuses", False)
    except ValueError as exc:
        check("[9] unknown engine kind refuses", "engine_kind" in str(exc))

    # [10] the retro-disclosure receipt we PUBLISH must verify against its own
    # seal.  It was once written, edited, and left with a stale digest -- a
    # self-sealed transparency receipt that does not verify is worse than none,
    # so the seal is re-checked here on every run.
    retro = Path(__file__).resolve().parent.parent / "reports" / "stack-provenance-retro.json"
    if retro.exists():
        doc = json.loads(retro.read_text(encoding="utf-8"))
        claimed = doc.get("receipt_sha256")
        body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
        recomputed = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest()
        check("[10] published retro receipt seals itself (method stated in-file)",
              claimed == recomputed
              and doc.get("seal", {}).get("method", "").startswith("sha256"),
              "%s vs %s" % (str(claimed)[:16], recomputed[:16]))
    else:
        check("[10] published retro receipt seals itself", True, "(absent: skipped)")

    print("selftest_stackprint: %s (%d passed, %d failed)"
          % ("FAIL" if FAIL else "PASS", len(PASS), len(FAIL)))
    for name, detail in FAIL:
        print("  FAILED: %s %s" % (name, detail))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
