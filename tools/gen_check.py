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
    ap.add_argument("--max-tokens", type=int, default=220)
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
        results.append({"prompt": prompt[:120], "completion": text,
                        "tokens": len(out.outputs[0].token_ids),
                        "degenerate": degenerate(text)})
        print("=" * 70)
        print("PROMPT:", prompt[:120].replace("\n", " "))
        print("COMPLETION:", text[:600])

    receipt = {
        "schema": "glm53flash-gen-check/1",
        "model": args.model,
        "chat_template_used": chat_rendered is not None,
        "results": results,
        "pass": all(not r["degenerate"] for r in results),
        "paris_mentioned": "Paris" in results[0]["completion"],
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2))
    print("GEN_CHECK " + json.dumps({"pass": receipt["pass"],
                                     "paris": receipt["paris_mentioned"]}), flush=True)
    if not receipt["pass"]:
        raise SystemExit("gen_check FAILED: degenerate or empty completions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
