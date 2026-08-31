#!/usr/bin/env python3
"""Stack fingerprint: WHAT ran, recorded as a receipt field instead of folklore.

A community reviewer asked three questions of every number we publish: what
vLLM runner produced it, which kernels actually ran, and was enforce-eager on
(CUDA graphs off)?  The campaign could answer only by convention — the facts
lived in environment.json, launch logs, and code defaults at a pinned commit,
none of them linked from the measurement receipts by digest.  This module makes
the answer a first-class, hashable block that every capture receipt embeds
verbatim, so "which stack produced this number" is a digest lookup, not an
archaeology project.

Design rules (the same ones the rest of this repo lives by):

* NEVER GUESS.  Every fact is queried from the live process; a fact that
  cannot be queried is recorded as null WITH THE REASON, mirroring
  `stream_score.probe_grouped_mm_kernel` ("a receipt must never assert which
  one ran, so this probes it").
* Dependency-light.  Importing this module touches only the stdlib.  torch and
  vllm are imported lazily inside probes, each probe wrapped so an absent or
  broken engine records a `probe_error` instead of killing the capture.
* Deterministic and json-stable.  Field order is fixed; the canonical hash is
  computed over the dict with volatile keys removed (`collected_utc`, `paths`,
  and the embedded seal itself), so two collections on an identical stack hash
  identically — the lane-identity pattern from the streaming lane.

API:

    fp = collect("vllm", llm=llm, declared={"enforce_eager": True})
    sha = fingerprint_sha256(fp)
    fp, sha = write(fp, out_dir)      # stack-fingerprint.json + pip-freeze.txt

Engine kinds:

    "vllm"                   serving lane; execution facts queried from the
                             live LLM handle's vllm_config when given.
    "transformers-reference" checkpoint lane (the reference forward).  CUDA
                             graphs and torch.compile are structurally absent
                             there, and the block says so instead of "false".
    "none"                   engineless processes (replay comparator,
                             cross-check compare): the process computes with
                             torch but serves nothing.

Checkpoint-lane integration: `from_backend_json` (bottom of this file) maps an
existing stream_score/k6 `backend.json` dict onto this schema.  It is shipped
as the documented adapter so the k6 tools can embed a fingerprint WITHOUT this
module needing torch at import time — but `engines/tools/stream_score.py` itself is
NOT wired yet: a concurrent format-adapters merge owns that file (JOURNAL
2026-08-29, "stack fingerprint" entry).  Wire it there after their merge
lands, by calling `from_backend_json(backend)` right after backend.json is
assembled.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

SCHEMA = "malaiwah.stack-fingerprint.v1"

# Keys excluded from the canonical hash: they vary between two collections on
# an identical stack (timestamps, absolute output paths, and the seal itself).
VOLATILE_KEYS = ("collected_utc", "paths", "stack_fingerprint_sha256")

# The exact env pins the campaign's determinism work established as
# behavior-relevant (JOURNAL: autotune pinning, symm-mem, NCCL shape, cuBLAS
# workspace).  Fixed tuple => fixed key order => stable hashes.  Raw value or
# null; never defaulted.
ENV_PIN_KEYS = (
    "NVIDIA_TF32_OVERRIDE",
    "TRITON_CACHE_AUTOTUNING",
    "TRITON_CACHE_DIR",
    "TRITON_PRINT_AUTOTUNING",
    "VLLM_USE_DEEP_GEMM",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_NVLS_ENABLE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "HF_HUB_OFFLINE",
    "PYTHONPATH",
)

ENGINE_KINDS = ("vllm", "transformers-reference", "none")

# Where the serving pipeline drops the container digest on the VM
# (remote/stage.sh writes it; docker load strips tags, so the file is the only
# trustworthy source — JOURNAL lesson).
IMAGE_PIN_CONVENTION_PATH = "/glm53/out/image-pin.txt"
IMAGE_PIN_ENV = "STACKPRINT_IMAGE_PIN"


def _err(exc) -> str:
    return "%s: %s" % (type(exc).__name__, exc)


def canonical_json(obj) -> str:
    # Must match registry/tools/registry_lib.py and bin/fidelity/common.py.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- probes
# Each probe returns a complete sub-dict with every key present.  A probe that
# cannot answer records nulls plus a `probe_error` or `reason` — the receipt
# then SAYS the fact is unknown instead of omitting it.


def _freeze_lines() -> "tuple[list, object]":
    """Installed distributions as sorted `name==version` lines, stdlib-only.

    importlib.metadata is used instead of a `pip freeze` subprocess so the
    fingerprint works inside stripped containers and costs milliseconds.  The
    line set is deterministic for a given environment.
    """
    try:
        from importlib import metadata

        seen = {}
        for dist in metadata.distributions():
            name = (dist.metadata.get("Name") or "").strip()
            if name:
                seen[name.lower()] = "%s==%s" % (name, dist.version)
        return sorted(seen.values()), None
    except Exception as exc:  # noqa: BLE001 - an unprobeable freeze is a fact
        return [], _err(exc)


def _package_origin(package: str) -> object:
    """Where a package came from (PEP 610 direct_url.json), or null."""
    try:
        from importlib import metadata

        raw = metadata.distribution(package).read_text("direct_url.json")
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 - absence of origin metadata is normal
        return None


def _engine_block(kind: str, freeze_lines) -> dict:
    block = {"kind": kind, "version": None, "git_sha": None,
             "git_sha_source": None, "package_origin": None, "probe_error": None}
    package = {"vllm": "vllm", "transformers-reference": "transformers"}.get(kind)
    if package is None:
        block["probe_error"] = None
        block["version"] = None
        block["git_sha_source"] = "not_applicable_no_engine"
        return block
    try:
        from importlib import metadata

        block["version"] = metadata.version(package)
    except Exception as exc:  # noqa: BLE001
        block["probe_error"] = _err(exc)
    if block["version"]:
        # vLLM dev builds carry `+gXXXXXXXXXX` (truncated commit) in the local
        # version segment; record it as such, never padded to 40 hex.
        local = block["version"].partition("+")[2]
        for part in local.split("."):
            if part.startswith("g") and len(part) > 4 and all(
                    c in "0123456789abcdef" for c in part[1:]):
                block["git_sha"] = part[1:]
                block["git_sha_source"] = "truncated_sha_from_version_string"
                break
        else:
            block["git_sha_source"] = "no_local_version_segment"
    block["package_origin"] = _package_origin(package) if block["version"] else None
    if block["package_origin"] is None and block["version"]:
        for line in freeze_lines:
            if line.lower().startswith(package + "=="):
                block["package_origin"] = {"freeze_line": line}
                break
    return block


def _torch_block() -> dict:
    try:
        import torch

        return {"version": str(torch.__version__),
                "cuda": torch.version.cuda,
                "cudnn_version": (torch.backends.cudnn.version()
                                  if torch.backends.cudnn.is_available() else None),
                "git_version": getattr(torch.version, "git_version", None),
                "probe_error": None}
    except Exception as exc:  # noqa: BLE001 - a torchless host is a fact
        return {"version": None, "cuda": None, "cudnn_version": None,
                "git_version": None, "probe_error": _err(exc)}


def _attr_chain(root, *paths):
    """First reachable attribute path; returns (value, 'a.b.c') or (None, None)."""
    for path in paths:
        node = root
        ok = True
        for name in path.split("."):
            node = getattr(node, name, None)
            if node is None:
                ok = False
                break
        if ok:
            return node, path
    return None, None


def _execution_block(kind: str, llm, declared: dict) -> dict:
    na_reason = ("transformers reference forward: no vLLM, torch.compile never "
                 "called, CUDA graphs never captured")
    if kind == "transformers-reference":
        return {
            "enforce_eager": None,
            "enforce_eager_source": "not_applicable_reference_lane",
            "compilation_mode": {"value": None, "reason": na_reason},
            "cudagraph_mode": {"value": None, "reason": na_reason},
            "max_cudagraph_capture_size": None,
            "attention_backend": {
                "requested": declared.get("attention_backend_requested"),
                "selected": declared.get("attention_backend_requested"),
                "selected_source": ("harness_arg"
                                    if "attention_backend_requested" in declared
                                    else "unqueryable"),
            },
        }
    if kind == "none":
        reason = "no inference engine in this process"
        return {
            "enforce_eager": None,
            "enforce_eager_source": "not_applicable_no_engine",
            "compilation_mode": {"value": None, "reason": reason},
            "cudagraph_mode": {"value": None, "reason": reason},
            "max_cudagraph_capture_size": None,
            "attention_backend": {"requested": None, "selected": None,
                                  "selected_source": "not_applicable_no_engine"},
        }

    # kind == "vllm"
    block = {
        "enforce_eager": None,
        "enforce_eager_source": None,
        "compilation_mode": None,
        "cudagraph_mode": None,
        "max_cudagraph_capture_size": None,
        "attention_backend": {
            "requested": declared.get("attention_backend_requested"),
            "selected": None,
            "selected_source": None,
        },
    }
    config = None
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", llm)
            config, _ = _attr_chain(engine, "vllm_config")
        except Exception:  # noqa: BLE001
            config = None
    if config is not None:
        value, path = _attr_chain(config, "model_config.enforce_eager")
        if isinstance(value, bool):
            block["enforce_eager"] = value
            block["enforce_eager_source"] = "vllm_config:%s" % path
        comp, _ = _attr_chain(config, "compilation_config")
        if comp is not None:
            mode, mpath = _attr_chain(comp, "mode", "level")
            cg, cgpath = _attr_chain(comp, "cudagraph_mode")
            size, _ = _attr_chain(comp, "max_cudagraph_capture_size")
            block["compilation_mode"] = ({"value": str(mode),
                                          "source": "vllm_config:compilation_config.%s" % mpath}
                                         if mode is not None else
                                         {"value": None,
                                          "reason": "compilation_config exposes no mode attr on this build"})
            block["cudagraph_mode"] = ({"value": str(cg),
                                        "source": "vllm_config:compilation_config.%s" % cgpath}
                                       if cg is not None else
                                       {"value": None,
                                        "reason": "compilation_config exposes no cudagraph_mode on this build"})
            block["max_cudagraph_capture_size"] = (int(size)
                                                   if isinstance(size, int) else None)
        selected, spath = _attr_chain(
            config,
            "model_config.attention_backend",
            "attention_config.backend",
        )
        if selected is not None:
            block["attention_backend"]["selected"] = str(selected)
            block["attention_backend"]["selected_source"] = "engine_query:%s" % spath
    if block["enforce_eager"] is None and "enforce_eager" in declared:
        block["enforce_eager"] = bool(declared["enforce_eager"])
        block["enforce_eager_source"] = "harness_arg"
    if block["enforce_eager"] is None:
        block["enforce_eager_source"] = (
            "unqueryable: no live engine handle and no declared value")
    if block["compilation_mode"] is None:
        block["compilation_mode"] = {
            "value": None,
            "reason": "not queryable on this vllm build/handle; see the launch log config dump"}
    if block["cudagraph_mode"] is None:
        block["cudagraph_mode"] = {
            "value": None,
            "reason": "not queryable on this vllm build/handle; see the launch log config dump"}
    if block["attention_backend"]["selected"] is None:
        block["attention_backend"]["selected_source"] = (
            "unqueryable: this vllm build exposes no selected-backend attr; "
            "see the launch log line 'Using ... attention backend'")
    return block


def _kernels_block(kind: str, llm) -> dict:
    if kind == "vllm":
        block = {"probed": False, "kernel_config": None, "quant_method_kernel": None,
                 "note": None}
        config = None
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", llm)
                config, _ = _attr_chain(engine, "vllm_config")
            except Exception:  # noqa: BLE001
                config = None
        kc, _ = _attr_chain(config, "kernel_config") if config is not None else (None, None)
        if kc is not None:
            echo = {}
            for name in ("enable_flashinfer_autotune", "moe_backend", "linear_backend"):
                value = getattr(kc, name, None)
                echo[name] = value if isinstance(value, (bool, int, str, type(None))) else str(value)
            prio = getattr(kc, "ir_op_priority", None)
            echo["ir_op_priority"] = str(prio) if prio is not None else None
            block["kernel_config"] = echo
            block["probed"] = True
        else:
            block["note"] = ("kernel_config not queryable from this handle; the "
                             "selected attention/MoE/GEMM kernels print in the "
                             "launch log ('Using ... backend' lines)")
        return block
    if kind == "transformers-reference":
        # The checkpoint lane's kernel truth is the grouped-MM dispatch probe;
        # same contract as stream_score.probe_grouped_mm_kernel ("a receipt
        # must never assert which one ran, so this probes it").
        block = {"probed": False,
                 "grouped_mm": {"probe": "transformers.integrations.moe._can_use_grouped_mm",
                                "can_use_native_grouped_mm": None,
                                "dispatched_kernel": None}}
        try:
            import torch
            from transformers.integrations import moe as moe_module

            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available() else "cpu")
            weight = torch.zeros(2, 8, 4, dtype=torch.bfloat16, device=device)
            hidden = torch.zeros(4, 4, dtype=torch.bfloat16, device=device)
            offs = torch.tensor([2, 4], dtype=torch.int32, device=device)
            can_use = bool(moe_module._can_use_grouped_mm(
                hidden, weight.transpose(-2, -1), offs))
            if can_use and hasattr(torch.nn.functional, "grouped_mm"):
                kernel = "torch.nn.functional.grouped_mm"
            elif can_use and hasattr(torch, "_grouped_mm"):
                kernel = "torch._grouped_mm"
            else:
                kernel = "torch.ops.transformers.grouped_mm_fallback"
            block["grouped_mm"].update({"can_use_native_grouped_mm": can_use,
                                        "dispatched_kernel": kernel,
                                        "device": str(device)})
            block["probed"] = True
        except Exception as exc:  # noqa: BLE001
            block["grouped_mm"]["probe_error"] = _err(exc)
        return block
    return {"probed": False,
            "note": "no inference engine in this process; nothing to probe"}


def _env_pins(extra_env=()) -> dict:
    keys = list(ENV_PIN_KEYS) + [k for k in extra_env if k not in ENV_PIN_KEYS]
    pins = {key: os.environ.get(key) for key in keys}
    pins["pin_shim_active"] = "pin_shim" in (os.environ.get("PYTHONPATH") or "")
    return pins


def _container_block() -> dict:
    env_pin = os.environ.get(IMAGE_PIN_ENV)
    if env_pin:
        return {"image_digest": env_pin.strip().split()[0],
                "source": "env:%s" % IMAGE_PIN_ENV}
    try:
        with open(IMAGE_PIN_CONVENTION_PATH, "r") as fh:
            first = fh.read().strip().split()[0]
        if first:
            return {"image_digest": first,
                    "source": "image-pin-file:%s" % IMAGE_PIN_CONVENTION_PATH}
    except Exception:  # noqa: BLE001 - not running in the pinned container
        pass
    return {"image_digest": None,
            "source": ("undetected (docker load strips digests; write %s or set "
                       "%s)" % (IMAGE_PIN_CONVENTION_PATH, IMAGE_PIN_ENV))}


def _gpu_block() -> dict:
    block = {"cuda_available": None, "mps_available": None, "devices": [],
             "driver_version": None, "probe_error": None}
    try:
        import torch

        block["cuda_available"] = bool(torch.cuda.is_available())
        mps = getattr(torch.backends, "mps", None)
        block["mps_available"] = bool(mps.is_available()) if mps is not None else False
        if block["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                block["devices"].append({
                    "name": props.name,
                    "total_memory_mib": int(props.total_memory // (1024 * 1024)),
                    "compute_capability": "%d.%d" % (props.major, props.minor),
                })
    except Exception as exc:  # noqa: BLE001 - a torchless host is a fact
        block["probe_error"] = _err(exc)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        versions = sorted({line.strip() for line in out.stdout.splitlines()
                           if line.strip()})
        block["driver_version"] = (versions[0] if len(versions) == 1
                                   else (versions or None))
    except Exception:  # noqa: BLE001
        block["driver_version"] = None
        if block["probe_error"] is None and block["cuda_available"] in (None, False):
            pass  # no nvidia-smi on a non-NVIDIA host: null is the honest answer
    return block


# ----------------------------------------------------------------- public API


def collect(engine_kind, llm=None, model=None, out_dir=None, declared=None,
            extra_env=()) -> dict:
    """Collect the stack fingerprint of THIS process.

    engine_kind: "vllm" | "transformers-reference" | "none".
    llm:         live vllm.LLM handle (or None) — execution facts are queried
                 from its vllm_config when reachable.
    model:       optional model path/name, recorded verbatim (identity proper
                 belongs to candidate_identity, not here).
    declared:    facts the harness itself set and can therefore attest —
                 {"enforce_eager": bool, "attention_backend_requested": str|None}.
                 Used only when the engine cannot be queried; recorded with
                 source "harness_arg", never presented as an engine query.
    extra_env:   extra environment keys to pin beyond ENV_PIN_KEYS.
    """
    if engine_kind not in ENGINE_KINDS:
        raise ValueError("engine_kind must be one of %r, got %r"
                         % (ENGINE_KINDS, engine_kind))
    declared = dict(declared or {})
    freeze_lines, freeze_error = _freeze_lines()
    freeze_text = "\n".join(freeze_lines) + ("\n" if freeze_lines else "")
    fp = {
        "schema": SCHEMA,
        "engine": _engine_block(engine_kind, freeze_lines),
        "model": model,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host_machine": platform.machine(),
        "torch": _torch_block(),
        "execution": _execution_block(engine_kind, llm, declared),
        "kernels": _kernels_block(engine_kind, llm),
        "env_pins": _env_pins(extra_env),
        "container": _container_block(),
        "gpus": _gpu_block(),
        "pip_freeze_sha256": (sha256_text(freeze_text) if freeze_lines else None),
        "pip_freeze_file": "pip-freeze.txt" if freeze_lines else None,
        "pip_freeze_error": freeze_error,
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": {"out_dir": str(out_dir) if out_dir else None},
    }
    fp["stack_fingerprint_sha256"] = fingerprint_sha256(fp)
    # Stash the freeze text for write() without making it part of the dict.
    fp["_pip_freeze_text"] = freeze_text
    return fp


def fingerprint_sha256(fp: dict) -> str:
    """Canonical hash of the fingerprint MINUS volatile keys.

    Two collections on an identical stack produce the same hash even though
    their timestamps differ — the streaming lane's lane-identity pattern.
    """
    stable = {k: v for k, v in fp.items()
              if k not in VOLATILE_KEYS and not k.startswith("_")}
    return sha256_text(canonical_json(stable))


def public_dict(fp: dict) -> dict:
    """The fingerprint as it is embedded in receipts (private stash removed)."""
    return {k: v for k, v in fp.items() if not k.startswith("_")}


def write(fp: dict, out_dir) -> "tuple[dict, str]":
    """Write stack-fingerprint.json (+ pip-freeze.txt) atomically into out_dir.

    Returns (public_fingerprint_dict, fingerprint_sha256).  The freeze file is
    the preimage of pip_freeze_sha256; the fingerprint carries only the digest
    so receipts stay small.
    """
    import pathlib

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    freeze_text = fp.get("_pip_freeze_text") or ""
    public = public_dict(fp)
    public["paths"] = {"out_dir": str(out)}
    sha = fingerprint_sha256(public)
    public["stack_fingerprint_sha256"] = sha
    if freeze_text:
        tmp = out / "pip-freeze.txt.tmp"
        tmp.write_text(freeze_text)
        tmp.replace(out / "pip-freeze.txt")
    tmp = out / "stack-fingerprint.json.tmp"
    tmp.write_text(json.dumps(public, indent=2))
    tmp.replace(out / "stack-fingerprint.json")
    return public, sha


# ---------------------------------------------- checkpoint-lane adapter (stub)


def _numeric_policy_from_backend(backend: dict) -> object:
    """Numeric policy as the receipt states it, from whichever field carries it.

    The streaming/student backend.json records a whole `numeric_policy` block;
    the published teacher backend.json instead records the single `allow_tf32`
    flag.  Take the richer one when present and otherwise say WHICH field the
    answer came from, so a null never has to stand in for "the receipt did say
    something, we just did not look at that key".
    """
    policy = backend.get("numeric_policy")
    if policy is not None:
        return policy
    if "allow_tf32" in backend:
        return {"allow_tf32": backend.get("allow_tf32"),
                "source_field": "backend.allow_tf32",
                "note": "this receipt records only the TF32 flag, not a full "
                        "numeric-policy block"}
    return None


def from_backend_json(backend: dict) -> dict:
    """Map a stream_score/k6 backend.json dict onto the fingerprint schema.

    INTEGRATION STUB — not called by any pipeline yet.  engines/tools/stream_score.py
    is owned by the in-flight format-adapters merge; once that lands on
    origin/main (rebase first), call this right after `backend` is assembled
    and store the result as backend["stack_fingerprint"] plus
    backend["stack_fingerprint_sha256"] (JOURNAL 2026-08-29 TODO).

    The mapping records only what backend.json actually states; env pins were
    not recorded by that lane, and the fingerprint says so instead of reading
    the CURRENT process env, which would attribute today's environment to a
    finished run.  All sources are receipt fields, so the result's
    `enforce_eager` block is the lane's structural answer: a transformers
    reference forward has no CUDA graphs to disable.
    """
    host = backend.get("host") or {}
    fp = {
        "schema": SCHEMA,
        "engine": {
            "kind": "transformers-reference",
            "version": backend.get("transformers_version"),
            "git_sha": None,
            "git_sha_source": "not_recorded_in_backend_json",
            "package_origin": None,
            "probe_error": None,
        },
        "model": backend.get("model_revision"),
        "python": host.get("python"),
        "platform": host.get("platform") or backend.get("device_name"),
        "host_machine": host.get("machine"),
        "torch": {"version": backend.get("torch_version"),
                  "cuda": backend.get("cuda_runtime_version"),
                  "cudnn_version": None, "git_version": None, "probe_error": None},
        "execution": {
            "enforce_eager": None,
            "enforce_eager_source": "not_applicable_reference_lane",
            "compilation_mode": {"value": None,
                                 "reason": "transformers reference forward: no vLLM, "
                                           "torch.compile never called, CUDA graphs "
                                           "never captured"},
            "cudagraph_mode": {"value": None,
                               "reason": "transformers reference forward: no vLLM, "
                                         "torch.compile never called, CUDA graphs "
                                         "never captured"},
            "max_cudagraph_capture_size": None,
            "attention_backend": {
                "requested": backend.get("attention_backend"),
                "selected": backend.get("attention_backend"),
                "selected_source": "receipt_field:backend.attention_backend",
            },
        },
        "kernels": {
            "probed": backend.get("grouped_mm_kernel") is not None,
            "grouped_mm": backend.get("grouped_mm_kernel"),
            "experts_implementation": backend.get("experts_implementation"),
            "numeric_policy": _numeric_policy_from_backend(backend),
            "parallelism": backend.get("parallelism"),
            "world_size": backend.get("world_size"),
            "nccl_version": backend.get("nccl_version"),
        },
        "env_pins": {"note": "not recorded by the checkpoint lane's backend.json; "
                             "numeric policy (TF32 off, fp32-matmul precision) is "
                             "recorded under kernels.numeric_policy instead"},
        "container": {"image_digest": None,
                      "source": "checkpoint lane runs on bare hosts; no image pin"},
        "gpus": {"cuda_available": None, "mps_available": None,
                 "devices": [{"name": backend.get("device_name"),
                              "total_memory_mib": None,
                              "compute_capability": None}]
                 if backend.get("device_name") else [],
                 "driver_version": None,
                 "probe_error": None},
        "pip_freeze_sha256": None,
        "pip_freeze_file": None,
        "pip_freeze_error": "not recorded in backend.json",
        "collected_utc": None,
        "paths": {"out_dir": None},
        # The link back to the receipt this was derived from.  backend.json
        # seals itself as backend_identity_sha256, and the kld receipts pin
        # that same digest — so a fingerprint derived here stays joinable to
        # the published chain instead of floating free.
        "source_receipt": {
            "schema": backend.get("schema"),
            "backend_identity_sha256": backend.get("backend_identity_sha256"),
            "checkpoint_identity_sha256": backend.get("checkpoint_identity_sha256"),
            "runtime_reader_sha256": backend.get("runtime_reader_sha256"),
            "packed_reader_abi_sha256": backend.get("packed_reader_abi_sha256"),
        },
        "source_note": "derived from a %s receipt by stackprint.from_backend_json"
                       % (backend.get("schema") or "backend.json"),
    }
    fp["stack_fingerprint_sha256"] = fingerprint_sha256(fp)
    return fp


if __name__ == "__main__":
    # `python3 stackprint.py [vllm|transformers-reference|none] [out_dir]`
    kind = sys.argv[1] if len(sys.argv) > 1 else "none"
    fp = collect(kind)
    if len(sys.argv) > 2:
        public, sha = write(fp, sys.argv[2])
        print(sha)
    else:
        print(json.dumps(public_dict(fp), indent=2))
