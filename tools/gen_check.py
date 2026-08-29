#!/usr/bin/env python3
"""Generation sanity gate: prove the engine produces coherent text before any
hidden-state extraction. Greedy completions for two raw prompts and one
chat-templated prompt; hard-fails on empty or degenerate (looping) output.

    gen_check.py --model DIR --out receipt.json [--tensor-parallel N] [--engine-kwargs JSON]
"""
from __future__ import annotations

import argparse
import json
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


def degenerate(text: str) -> bool:
    words = text.split()
    if len(text.strip()) < 20 or len(words) < 8:
        return True
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) < max(1, len(grams)) * 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-parallel", type=int, default=8)
    ap.add_argument("--engine-kwargs", default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=640)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, tensor_parallel_size=args.tensor_parallel,
                  dtype="bfloat16", load_format="safetensors",
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=2112, max_num_seqs=4, enforce_eager=True,
                  enable_prefix_caching=False, disable_log_stats=True)
    if args.engine_kwargs:
        kwargs.update(json.loads(args.engine_kwargs))
    llm = LLM(**kwargs)
    stackprint = load_stackprint()
    stack_fp, stack_fp_sha = stackprint.write(
        stackprint.collect("vllm", llm=llm, model=args.model,
                           declared={"enforce_eager": bool(kwargs.get("enforce_eager")),
                                     "attention_backend_requested": None}),
        Path(args.out).resolve().parent)
    print("stack_fingerprint " + json.dumps({
        "sha256": stack_fp_sha,
        "enforce_eager": stack_fp["execution"]["enforce_eager"],
        "enforce_eager_source": stack_fp["execution"]["enforce_eager_source"],
    }), flush=True)
    tok = llm.get_tokenizer()

    raw_prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    ]
    chat = [{"role": "user",
             "content": "In one short paragraph, explain what KL divergence measures."}]
    prompts = list(raw_prompts)
    chat_rendered = None
    try:
        chat_rendered = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(chat_rendered)
    except Exception as exc:
        print(f"chat template unavailable ({exc}); raw prompts only", flush=True)

    outs = llm.generate(prompts, SamplingParams(max_tokens=args.max_tokens, temperature=0))
    results = []
    for prompt, out in zip(prompts, outs):
        text = out.outputs[0].text
        think, sep, answer = text.partition("</think>")
        has_think = bool(sep) or think.lstrip().startswith("<think>")
        visible = answer.strip() if sep else text.strip()
        results.append({"prompt": prompt[:120], "completion": text,
                        "answer_excerpt": visible[:500],
                        "has_think_block": has_think,
                        "think_chars": len(think) if sep else 0,
                        "answer_chars": len(visible),
                        "tokens": len(out.outputs[0].token_ids),
                        "degenerate": degenerate(text)})
        print("=" * 70)
        print("PROMPT:", prompt[:120].replace("\n", " "))
        if sep:
            print("THINK (first 300):", think[:300].replace("\n", " "))
            print("ANSWER:", visible[:500])
        else:
            print("COMPLETION:", text[:600])

    receipt = {
        "schema": "glm53flash-gen-check/1",
        "model": args.model,
        "stack_fingerprint": stack_fp,
        "stack_fingerprint_sha256": stack_fp_sha,
        "chat_template_used": chat_rendered is not None,
        "results": results,
        "pass": all(not r["degenerate"] for r in results),
        "paris_mentioned": "Paris" in results[0]["completion"],
        "chat_answered_after_think": (results[-1]["answer_chars"] > 30
                                      if chat_rendered is not None else None),
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2))
    print("GEN_CHECK " + json.dumps({"pass": receipt["pass"],
                                     "paris": receipt["paris_mentioned"]}), flush=True)
    if not receipt["pass"]:
        raise SystemExit("gen_check FAILED: degenerate or empty completions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
