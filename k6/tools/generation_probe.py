#!/usr/bin/env python3
"""Does this model still generate sensibly?  One extra window, asked of every capture.

The failure this closes
-----------------------
`hf_capture.refuse_on_load_report` and `layer_outer.audit_checkpoint_tree` are
guards over NAMES, SHAPES and COUNTS: they catch a tensor that is missing, a
tensor whose shape disagrees, a shard shorter than its own header, a key set that
disagrees with the index.  There is a class of catastrophe that passes all of
them -- a shard whose bytes are the right length and the wrong content:

  * a sparse-file fetch that left a hole the right size, so the tensor reads as
    ZEROS (`audit_checkpoint_tree` says so itself: "what it does NOT catch");
  * `--allow-missing-weights` over a real absence, where `transformers`
    randomly initialises the parameters and hands back a model that runs
    (observed on `malaiwah/GLM-5.2-SIQ-Fruit`: random experts, mean ~0,
    std 0.0199, correct names, correct shapes, correct count);
  * a plain FP8 payload cast into a bf16 parameter with its block scale never
    applied -- identical shapes, and only `unexpected_keys` to show for it.

Every one of those produces a confident capture of weights nobody measured.
None of them survives asking the model a question a language model can answer.

    "The capital of France is"  ->  " Paris"

Why it is nearly free, and therefore belongs in EVERY capture
------------------------------------------------------------
Under the layer-outer schedule the probe is not a second pass over the
checkpoint.  It is ONE MORE WINDOW pushed through the layers the schedule is
already loading, so it costs one extra forward per layer -- 1/N of a capture on
an N-window panel, ~4% at N=25, and ZERO extra weight loading, which is the part
that actually costs money.  Under window-outer it is one extra forward.  That is
why this is proposed as unconditional rather than as a race-mode extra.

What it does NOT do
-------------------
It does not touch the captured tensors.  The probe is a separate window whose
hidden state is discarded; it is never written to the dataset, never enters
`capture_content_digest`, and its presence must not move a single captured byte
(`bin/selftest_race_mode.py` asserts the digests are equal with and without it).
Windows are pushed through the schedule one at a time and carry no shared state,
so there is no mechanism by which it could.

The verdict, and how much of it is fail-closed
----------------------------------------------
Always RECORDED, never inferred: top-1 id, its detokenisation, its probability,
the distribution's entropy, and the uniform entropy to compare it against.

Fail-closed unconditionally on a DEGENERATE distribution -- every logit exactly
equal.  That is what an all-zeros head or an all-zeros final hidden state
produces, it is model-agnostic, and no trained model at any scale emits it.

Fail-closed on the CONTENT only when the caller declares an expectation
(`expect=`).  A declared expectation is checked against the tokenizer's own
encoding of the expected continuation rather than against a string compare, so
it works the same on SentencePiece and BPE; the detokenised form is accepted too,
so a tokenizer that splits differently still passes when it is right.  Without an
expectation the probe still runs and still records -- an unlabelled model (the
0.1B CI fixture is randomly initialised and answers nothing) gets a recorded
observation rather than a false alarm.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

PROBE_SCHEMA = "malaiwah.generation-sanity-probe.v1"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_EXPECT = "Paris"


class ProbeRefusal(Exception):
    """The model does not generate sensibly, and the capture must not be published."""


def load_tokenizer(model_dir: str):
    """The tokenizer, or None with the reason -- never a crash mid-capture."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir), None
    except Exception as exc:  # pragma: no cover - depends on the checkpoint
        return None, "%s: %s" % (type(exc).__name__, exc)


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    if not ids:
        raise ProbeRefusal("the sanity prompt %r encodes to zero tokens" % prompt)
    return [int(i) for i in ids]


def expected_first_ids(tokenizer, expect: str) -> List[int]:
    """Every plausible FIRST token id of `expect` as a continuation.

    A leading space is part of the token on BPE and part of the piece marker on
    SentencePiece, and which spelling a given tokenizer wants is not knowable
    from here -- so all of them are offered and any match counts.  This is the
    difference between a check that works on one tokenizer and a check that
    works on the next model to land.
    """
    ids: List[int] = []
    for spelling in (" " + expect, expect, " " + expect.lower(), expect.lower()):
        try:
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
        except Exception:  # pragma: no cover - tokenizer-specific
            continue
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def evaluate(logits, tokenizer, *, prompt: str, expect: Optional[str],
             top_k: int = 5) -> Dict[str, Any]:
    """Turn one last-position logit row into a recorded, possibly refusing, verdict.

    `logits` is a 1-D torch tensor over the full vocabulary. It is read in fp64
    for the same reason every estimator here does: a softmax in bf16 over 150k
    logits is not a number anybody should quote.
    """
    import torch

    row = logits.detach().to("cpu", torch.float64).reshape(-1)
    vocab = int(row.shape[0])
    probs = torch.softmax(row, dim=0)
    top = torch.topk(probs, k=min(top_k, vocab))
    top_ids = [int(i) for i in top.indices.tolist()]
    top_probs = [float(p) for p in top.values.tolist()]
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-300))).sum())
    uniform_entropy = math.log(vocab)
    spread = float(row.max() - row.min())

    def detok(index: int) -> str:
        try:
            return tokenizer.decode([index])
        except Exception:  # pragma: no cover - tokenizer-specific
            return "<undecodable:%d>" % index

    verdict: Dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "prompt": prompt,
        "expect": expect,
        "vocab_size": vocab,
        "top1_id": top_ids[0],
        "top1_text": detok(top_ids[0]),
        "top1_probability": top_probs[0],
        "top_k": [{"id": i, "text": detok(i), "probability": p}
                  for i, p in zip(top_ids, top_probs)],
        "entropy_nats": entropy,
        "uniform_entropy_nats": uniform_entropy,
        "logit_spread": spread,
        "degenerate": spread == 0.0,
        "status": "recorded",
        "enforced": expect is not None,
        "note": "one extra window through the same schedule; its hidden state is "
                "discarded and never enters the dataset.",
    }

    # Fail-closed, always, model-agnostically. Every logit exactly equal is what
    # an all-zeros tensor produces and what no trained model produces.
    if verdict["degenerate"]:
        verdict["status"] = "fail"
        raise ProbeRefusal(
            "REFUSED: the generation sanity probe found a DEGENERATE output "
            "distribution -- all %d logits are exactly equal (entropy %.6f nats == "
            "ln(vocab)). That is the signature of a tensor that loaded as zeros: the "
            "shapes are right, the tensor count is right, and the model computes "
            "nothing. Prompt was %r. Nothing about this capture is a measurement of "
            "the published weights." % (vocab, entropy, prompt))

    if expect is None:
        return verdict

    wanted = expected_first_ids(tokenizer, expect)
    verdict["expected_first_ids"] = wanted
    text_match = verdict["top1_text"].strip().lower() == expect.strip().lower()
    if verdict["top1_id"] in wanted or text_match:
        verdict["status"] = "pass"
        return verdict
    verdict["status"] = "fail"
    raise ProbeRefusal(
        "REFUSED: the generation sanity probe failed. %r continued with %r "
        "(p=%.4f, entropy %.3f of a possible %.3f nats), and %r was expected. "
        "Tensor counts, shapes and the load report can ALL be clean while a shard "
        "loaded as zeros or as randomly initialised weights -- this is the check "
        "that sees it. Top-%d: %s. Pass --sanity-expect '' to record the probe "
        "without enforcing it, if this model is genuinely expected to answer "
        "otherwise."
        % (prompt, verdict["top1_text"], verdict["top1_probability"],
           verdict["entropy_nats"], verdict["uniform_entropy_nats"], expect,
           len(top_ids),
           ", ".join("%r p=%.4f" % (e["text"], e["probability"])
                     for e in verdict["top_k"])))


def skipped(reason: str, *, prompt: str, expect: Optional[str],
            enforced: bool) -> Dict[str, Any]:
    """A probe that could not run says so, with the reason. SKIP is a verdict."""
    return {
        "schema": PROBE_SCHEMA,
        "prompt": prompt,
        "expect": expect,
        "status": "skipped",
        "enforced": bool(enforced),
        "reason": reason,
    }
