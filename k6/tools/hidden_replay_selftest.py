#!/usr/bin/env python3
"""Offline selftest ladder for hidden_replay.py (no GPU, no model, no panel).

Rungs (all required; a rung that cannot run is a FAIL, not a SKIP):
  [a] fp64 KL estimator: KL(x||x) == 0.0 exactly; direction and value against
      an independent numpy reference on a hand-checkable case.
  [b] mask-selection alignment: the hidden selection (hidden[:-1][causal])
      lands on exactly the rows whose logits the streaming loop stores
      (logits[0, :-1, :][causal]), including a mask with a padded tail.
  [c] replay equivalence has teeth: "live" = bf16 matmul upcast to fp32,
      replay = fp32 matmul of the same bf16 inputs -> small KLD and high
      top-1; a row-permuted head -> KLD orders of magnitude larger.
  [d] vocab-chunk invariance on CPU fp32: chunked vs monolithic GEMM deltas
      are ~0 and the per-token KLD delta is below 1e-9 nats.
  [e] the forward pre-hook mechanism: a hook on a Linear head observes the
      exact post-norm bf16 tensor the module consumes, and the tap sees one
      entry per forward.
  [f] payload_sha256 is metadata-independent: two safetensors files with the
      same tensor and different __metadata__ share the payload hash and
      differ in file hash.

Run:  python3 hidden_replay_selftest.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hidden_replay  # noqa: E402


def kl_reference(p_logits, q_logits):
    """Independent fp64 KL(p||q) reference in plain python/numpy."""
    import numpy as np

    p_logits = np.asarray(p_logits, dtype=np.float64)
    q_logits = np.asarray(q_logits, dtype=np.float64)
    out = []
    for row_p, row_q in zip(p_logits, q_logits):
        lse_p = row_p.max() + math.log(np.exp(row_p - row_p.max()).sum())
        lse_q = row_q.max() + math.log(np.exp(row_q - row_q.max()).sum())
        logp = row_p - lse_p
        logq = row_q - lse_q
        out.append(float((np.exp(logp) * (logp - logq)).sum()))
    return np.asarray(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kld_report

    results = []

    def record(rung: str, ok: bool, detail: str) -> None:
        results.append({"rung": rung, "ok": bool(ok), "detail": detail})
        print(f"[{'ok' if ok else 'FAIL'}] {rung}: {detail}", flush=True)

    torch.manual_seed(20260829)

    # ---- [a] estimator identity + direction ------------------------------
    x = torch.randn(5, 33, dtype=torch.float64) * 3
    same, matches = kld_report._token_kld(x.clone(), x.clone(), "cpu")
    y = x.clone()
    y[:, 0] += 1.5  # boost one logit in the STUDENT -> teacher-weighted penalty
    diff, _ = kld_report._token_kld(x, y, "cpu")
    ref = kl_reference(x.numpy(), y.numpy())
    a_ok = (
        bool((same == 0.0).all())
        and matches == 5
        and bool(np.allclose(diff, ref, rtol=0, atol=1e-12))
        and float(diff.max()) > 0
    )
    record("a-estimator", a_ok,
           f"KL(x||x) all-zero={bool((same == 0.0).all())}, vs-reference max|d|="
           f"{float(np.abs(diff - ref).max()):.3e}")

    # ---- [b] mask-selection alignment ------------------------------------
    seq, hid, vocab = 12, 8, 17
    hidden_full = torch.randn(1, seq, hid, dtype=torch.bfloat16)
    head = torch.randn(vocab, hid, dtype=torch.bfloat16)
    logits_full = (hidden_full.float() @ head.float().t()).unsqueeze(0).squeeze(0)  # [1,seq,vocab]
    mask = np.ones(seq, dtype=np.int64)
    mask[-3:] = 0  # padded tail
    causal = np.asarray(mask[:-1], dtype=np.bool_) & np.asarray(mask[1:], dtype=np.bool_)
    stored_logits = logits_full[0, :-1, :][torch.from_numpy(causal)]
    selected_hidden = hidden_full.squeeze(0)[:-1][torch.from_numpy(causal)]
    replayed = selected_hidden.float() @ head.float().t()
    b_ok = (
        stored_logits.shape[0] == int(causal.sum())
        and torch.equal(stored_logits, replayed)
    )
    record("b-mask-alignment", b_ok,
           f"positions={int(causal.sum())}/{seq - 1}, stored==replayed(bitwise)={bool(torch.equal(stored_logits, replayed))}")

    # ---- [c] replay equivalence has teeth --------------------------------
    positions, hid_c, vocab_c = 64, 128, 512
    hidden_c = torch.randn(positions, hid_c, dtype=torch.bfloat16)
    head_c = (torch.randn(vocab_c, hid_c) / math.sqrt(hid_c)).to(torch.bfloat16)
    live = (hidden_c @ head_c.t()).float() * 4.0        # bf16 matmul, upcast (the serving-ish path)
    replay = (hidden_c.float() @ head_c.float().t()) * 4.0  # fp32 replay of the same bf16 inputs
    kld_close, match_close = kld_report._token_kld(live, replay, "cpu")
    perm = torch.randperm(vocab_c)
    kld_perm, _ = kld_report._token_kld(live, replay[:, perm], "cpu")
    c_ok = (
        float(kld_close.mean()) < 5e-3
        and float(kld_perm.mean()) > 100 * float(kld_close.mean())
        and match_close >= int(0.9 * positions)
    )
    record("c-replay-teeth", c_ok,
           f"bf16-vs-fp32 mean KLD={float(kld_close.mean()):.3e} (top1 {match_close}/{positions}); "
           f"permuted-head mean KLD={float(kld_perm.mean()):.3e}")

    # ---- [d] vocab-chunk invariance --------------------------------------
    mono = hidden_replay._replay_logits(hidden_c, head_c.float().t().contiguous(), "cpu")
    chunked = hidden_replay._replay_logits(hidden_c, head_c.float().t().contiguous(), "cpu",
                                           vocab_chunk=100)
    kld_mono, _ = kld_report._token_kld(live, mono * 4.0, "cpu")
    kld_chunk, _ = kld_report._token_kld(live, chunked * 4.0, "cpu")
    delta_means = abs(float(kld_mono.mean()) - float(kld_chunk.mean()))
    bitwise_fraction = float((mono == chunked).float().mean())
    d_ok = delta_means < 1e-9 and float(np.abs(kld_mono - kld_chunk).max()) < 1e-9
    record("d-chunk-invariance", d_ok,
           f"delta-of-means={delta_means:.3e}, max-token-delta={float(np.abs(kld_mono - kld_chunk).max()):.3e}, "
           f"bitwise-equal-fraction={bitwise_fraction:.6f}")

    # ---- [e] pre-hook mechanism ------------------------------------------
    class TinyHeaded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hid, dtype=torch.bfloat16)
            self.lm_head = torch.nn.Linear(hid, vocab, bias=False, dtype=torch.bfloat16)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, hidden_states):
            normed = self.norm(hidden_states)
            self._last_normed = normed.detach()
            return self.lm_head(normed)

    tiny = TinyHeaded().eval()
    tap = []

    def pre_hook(module, hook_args):
        value = hook_args[0]
        assert value.dtype == torch.bfloat16
        tap.append(value.detach().squeeze(0).to("cpu", copy=True))

    tiny.get_output_embeddings().register_forward_pre_hook(pre_hook)
    with torch.inference_mode():
        for _ in range(3):
            tiny(torch.randn(1, seq, hid, dtype=torch.bfloat16))
    e_ok = len(tap) == 3 and torch.equal(tap[-1], tiny._last_normed.squeeze(0))
    record("e-hook-mechanism", e_ok,
           f"tap fired {len(tap)}/3; captured==post-norm(bitwise)={bool(torch.equal(tap[-1], tiny._last_normed.squeeze(0)))}")

    # ---- [f] payload sha is metadata-independent -------------------------
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp:
        tensor = torch.randn(7, 5, dtype=torch.bfloat16)
        path1, path2 = Path(tmp) / "one.safetensors", Path(tmp) / "two.safetensors"
        save_file({"hidden": tensor}, path1, metadata={"cold_run": "1"})
        save_file({"hidden": tensor}, path2, metadata={"cold_run": "2"})
        f_ok = (
            hidden_replay.payload_sha256(path1) == hidden_replay.payload_sha256(path2)
            and hidden_replay.sha256_file(path1) != hidden_replay.sha256_file(path2)
            and hidden_replay.tensor_content_sha256(tensor)
            == hidden_replay.tensor_content_sha256(tensor.clone())
        )
        record("f-payload-sha", f_ok,
               "payload hashes equal across metadata variants; file hashes differ")

    passed = sum(1 for row in results if row["ok"])
    verdict = {"schema": "malaiwah.glm53-hidden-replay-selftest.v1",
               "passed": passed, "failed": len(results) - passed, "rungs": results}
    if args.json:
        args.json.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "failed": len(results) - passed}, sort_keys=True))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
