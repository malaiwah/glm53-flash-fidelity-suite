#!/usr/bin/env python3
"""Capture per-layer linear-input activations over a calibration suite (MoE-aware).

For every decoder layer this stores, per context, the INPUT tensors of the two
sub-blocks as they enter their linear projections:

  * ``attn_in``  - the argument of the attention module's forward (post
    input-norm hidden states): the q/k/v (or KDA in-proj / DSA q,kv) input;
  * ``mlp_in``   - the argument of the MLP/MoE module's forward (post
    post-attention-norm hidden states): the router input AND the gate/up input
    of every routed/shared expert.

From ``mlp_in`` plus the public router and expert weights, everything MoE
calibration needs is recomputable offline: expert routing/top-k statistics,
per-expert token subsets, per-expert gate/up Hessians E[xx^T], and (by running
gate/up + activation) each expert's down-proj inputs and Hessians. Storing the
block inputs instead of derived Hessians keeps the artifact universal and
byte-auditable.

Same engine discipline as fidelity.py capture: eager, max_num_seqs=1, one
context at a time, resumable via an atomic manifest with per-file sha256.

    activation_capture.py --model DIR --suite SUITE_DIR --out OUT_DIR \
        [--tensor-parallel N] [--engine-kwargs JSON] [--contexts N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def load_stackprint():
    """bin/fidelity/stackprint.py by path (repo checkout or VM bundle); a
    receipt without a stack fingerprint is refusable, so failure refuses."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "bin" / "fidelity" / "stackprint.py"
    try:
        spec = importlib.util.spec_from_file_location("glm53_stackprint", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise SystemExit(
            f"stack fingerprint module unavailable ({exc}) at {path}; "
            "re-run make_bundle.sh so bin/fidelity/stackprint.py ships next to tools/"
        )


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


ATTN_NAMES = ("self_attn", "attn", "linear_attn", "kda", "attention")
MLP_NAMES = ("mlp", "block_sparse_moe", "moe", "feed_forward")


def _rpc_install_act_hooks(self):
    """Runs in each worker: pre-hooks on every layer's attention + MLP modules."""
    import re
    import torch

    model = self.model_runner.model
    layer_re = re.compile(r"^(.*\.layers\.(\d+))\.([A-Za-z_0-9]+)$")
    hooked = {}
    store = {"parts": {}, "enabled": False}

    def make_hook(key):
        def hook(_m, args, kwargs=None):
            if not store["enabled"]:
                return
            t = None
            if args:
                for a in args:
                    if hasattr(a, "dim") and a.dim() in (2, 3) and a.shape[-1] > 8:
                        t = a
                        break
            if t is None and kwargs:
                for a in kwargs.values():
                    if hasattr(a, "dim") and a.dim() in (2, 3) and a.shape[-1] > 8:
                        t = a
                        break
            if t is None:
                return
            cpu = t.detach().reshape(-1, t.shape[-1]).to("cpu", torch.bfloat16, copy=True)
            store["parts"].setdefault(key, []).append(cpu)
        return hook

    for name, module in model.named_modules():
        m = layer_re.match(name)
        if not m:
            continue
        child = m.group(3)
        layer = int(m.group(2))
        if child in ATTN_NAMES:
            kind = "attn_in"
        elif child in MLP_NAMES:
            kind = "mlp_in"
        else:
            continue
        key = f"layer_{layer:03d}.{kind}"
        if key in hooked:
            raise RuntimeError(f"duplicate hook target {key}: {name} vs {hooked[key]}")
        module.register_forward_pre_hook(make_hook(key), with_kwargs=True)
        hooked[key] = name
    def make_gate_hook(key):
        def hook(_m, _args, output):
            if not store["enabled"]:
                return
            t = output[0] if isinstance(output, tuple) else output
            if hasattr(t, "dim") and t.dim() in (2, 3):
                cpu = t.detach().reshape(-1, t.shape[-1]).to("cpu", torch.float32, copy=True)
                store["parts"].setdefault(key, []).append(cpu)
        return hook

    gate_re = re.compile(r"^.*\.layers\.(\d+)\.(?:mlp|block_sparse_moe|moe|feed_forward)\.(gate|router)$")
    gates = 0
    for name, module in model.named_modules():
        g = gate_re.match(name)
        if not g:
            continue
        key = f"layer_{int(g.group(1)):03d}.router_logits"
        module.register_forward_hook(make_gate_hook(key))
        hooked[key] = name
        gates += 1
    if not gates:
        print("WARNING: no router gate modules matched; routing stays recomputable from mlp_in", flush=True)

    if not hooked:
        raise RuntimeError("no attention/MLP modules matched for activation hooks")
    self._act_store = store
    return sorted(hooked.items())


def _rpc_act_start(self):
    store = getattr(self, "_act_store", None)
    if store is not None:
        store["parts"] = {}
        store["enabled"] = True
    return True


def _rpc_act_pop(self):
    import torch

    store = getattr(self, "_act_store", None)
    if store is None:
        return None
    store["enabled"] = False
    if getattr(self, "rank", 0) != 0:
        store["parts"] = {}
        return None
    out = {}
    for key, parts in store["parts"].items():
        out[key] = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    store["parts"] = {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--engine-kwargs", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--contexts", type=int, default=0, help="0 = every suite context")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from safetensors.torch import save_file

    suite = Path(args.suite)
    manifest_suite = json.loads((suite / "suite-manifest.json").read_text())
    ctx_len = manifest_suite["context_length"]
    selected = manifest_suite["context_index"]
    if args.contexts:
        selected = selected[: args.contexts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "activation-manifest.json"

    kwargs = dict(model=args.model, trust_remote_code=args.trust_remote_code,
                  tensor_parallel_size=args.tensor_parallel,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  dtype="bfloat16", load_format="safetensors",
                  max_model_len=ctx_len + 64, max_num_batched_tokens=ctx_len,
                  max_num_seqs=1, enable_prefix_caching=False, disable_log_stats=True,
                  enforce_eager=True)
    if args.engine_kwargs:
        extra = json.loads(args.engine_kwargs)
        kwargs.update(extra)
        print("engine_kwargs " + json.dumps(extra), flush=True)
    if not kwargs.get("enforce_eager"):
        raise SystemExit("activation capture requires enforce_eager")
    llm = LLM(**kwargs)
    stackprint = load_stackprint()
    stack_fp, stack_fp_sha = stackprint.write(
        stackprint.collect("vllm", llm=llm, model=args.model,
                           declared={"enforce_eager": bool(kwargs.get("enforce_eager")),
                                     "attention_backend_requested": None}),
        out)
    print("stack_fingerprint " + json.dumps({
        "sha256": stack_fp_sha,
        "enforce_eager": stack_fp["execution"]["enforce_eager"],
        "enforce_eager_source": stack_fp["execution"]["enforce_eager_source"],
    }), flush=True)

    hooked = llm.collective_rpc(_rpc_install_act_hooks)[0]
    print(f"hooked {len(hooked)} modules; first/last: {hooked[0]} .. {hooked[-1]}", flush=True)

    rev = Path(args.model, "revision.txt")
    records = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        records = {r["index"]: r for r in prior.get("captures", [])}
        for index, record in list(records.items()):
            f = out / record["file"]
            if not f.is_file() or sha256_file(f) != record["sha256"]:
                raise SystemExit(f"resume mismatch on {f}")

    def write_manifest(complete: bool) -> None:
        payload = {
            "schema": "glm53flash-activation-capture/1",
            "model": args.model,
            "model_revision": rev.read_text().strip() if rev.is_file() else None,
            "suite_token_sha256": manifest_suite["suite_token_sha256"],
            "context_length": ctx_len,
            "stack_fingerprint": stack_fp,
            "stack_fingerprint_sha256": stack_fp_sha,
            "hooked_modules": hooked,
            "dtype": "bfloat16",
            "semantics": {
                "attn_in": "forward_pre_hook input of the attention module (post input-norm)",
                "mlp_in": "forward_pre_hook input of the MLP/MoE module (post post-attention-norm); router and expert gate/up input",
                "router_logits": "forward hook output of the MoE gate module, float32 (natural routing ground truth)",
            },
            "contexts": len(records),
            "expected_contexts": len(selected),
            "captures": [records[i] for i in sorted(records)],
            "complete": complete,
        }
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(manifest_path)

    params = SamplingParams(max_tokens=1, temperature=0, detokenize=False)
    started = time.time()
    write_manifest(complete=False)
    for ctx in selected:
        index = ctx["index"]
        if index in records:
            continue
        ids = json.loads((suite / ctx["file"]).read_text())
        if sha256_bytes(json.dumps(ids).encode()) != ctx["token_sha256"]:
            raise SystemExit(f"token hash drift on context {index}")
        llm.collective_rpc(_rpc_act_pop)
        llm.collective_rpc(_rpc_act_start)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params=params,
                     use_tqdm=False)
        got = llm.collective_rpc(_rpc_act_pop)[0]
        if not got:
            raise SystemExit(f"no activations captured for context {index}")
        bad = {k: tuple(v.shape) for k, v in got.items()
               if not k.endswith("router_logits") and v.shape[0] != ctx_len}
        if bad:
            raise SystemExit(f"unexpected activation row counts for context {index}: {bad}")
        dst = out / f"acts_{index:04d}.safetensors"
        tmp = dst.with_name(dst.name + ".tmp")
        save_file({k: v.contiguous() for k, v in got.items()}, str(tmp))
        tmp.replace(dst)
        records[index] = {"index": index, "file": dst.name, "sha256": sha256_file(dst),
                          "tensors": len(got)}
        write_manifest(complete=False)
        if len(records) % 8 == 0:
            print(f"{len(records)}/{len(selected)} contexts "
                  f"({time.time() - started:.0f}s)", flush=True)
    write_manifest(complete=True)
    print("activation_capture_done " + json.dumps(
        {"contexts": len(records), "elapsed_sec": time.time() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
