#!/usr/bin/env python3
"""Place the drafted docs into the repo working tree, filling every number from
the receipt.

Deliberate asymmetry, because another workflow has uncommitted changes in this
tree:

  k6/HIDDEN-REPLAY.md      NEW file        -> written directly (no conflict possible)
  k6/SAME-LANE-TEACHER.md  clean in tree   -> edited directly (safe to stage alone)
  WHAT-WE-MEASURE.md       DIRTY (theirs)  -> NOT touched; the one-line pointer is
                                              emitted to the scratchpad for the
                                              publish phase to apply after a rebase,
                                              so their uncommitted work is never
                                              swept into our commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    args = ap.parse_args()

    r = json.loads(args.receipt.read_text(encoding="utf-8"))
    r1 = r["metrics_per_run"][0]
    st = r["storage"]
    inv = r["vocab_chunk_invariance"]
    cross = r["cross_run_determinism"]

    subs = {
        "{{HIDDEN_MB}}": f"{st['hiddens_mib']:.0f}",
        "{{HIDDEN_BYTES}}": f"{st['hiddens_bytes_per_run']:,}",
        "{{LOGITS_BYTES}}": f"{st['logits_bytes_per_run']:,}",
        "{{LOGITS_GIB}}": f"{st['logits_gib']:.2f}",
        "{{SHRINK}}": f"{st['shrink_factor']}",
        "{{VOCAB}}": f"{r['vocab_size']:,}",
        "{{RECEIPT_SHA}}": r["receipt_sha256"],
        "{{RUNS}}": str(len(r["metrics_per_run"])),
        "{{POSITIONS}}": f"{r1['positions']:,}",
        "{{MEAN}}": repr(r1["replay_kld_mean"]),
        "{{MAX}}": repr(r1["replay_kld_max"]),
        "{{P99}}": repr(r1["replay_kld_p99"]),
        "{{P999}}": repr(r1["replay_kld_p99_9"]),
        "{{TOP1}}": repr(r1["top1_agreement_live_vs_replayed"]),
        "{{PANEL_DELTA}}": repr(r1["panel_delta_replayed_minus_live"]),
        "{{INV_DELTA}}": repr(inv["delta_of_means"]),
        "{{HEAD_SHA}}": r["head"]["tensor_content_sha256"],
        "{{HIDDEN_DIGESTS}}": str(len(cross["distinct_hiddens_payload_digests"])),
        "{{REPRO}}": ("bitwise" if r["path_a_reproduction"]["mean_reproduced_exactly"]
                      else "NOT bitwise (see receipt)"),
        "{{PARFETCH_SHA}}": r["stack"]["on_box_only_helpers"]["files"]["par_fetch.sh"],
        "{{FETCHFAST_SHA}}": r["stack"]["on_box_only_helpers"]["files"]["fetch_fast.sh"],
    }

    def fill(text: str) -> str:
        for key, value in subs.items():
            text = text.replace(key, value)
        left = [k for k in subs if k in text]
        assert not left, f"unfilled placeholders: {left}"
        return text

    # ---- 1. SAME-LANE-TEACHER.md: insert before "## Open items" -----------
    slt = args.repo / "k6" / "SAME-LANE-TEACHER.md"
    body = slt.read_text(encoding="utf-8")
    addition = fill((args.scratch / "same_lane_addition.md.tmpl").read_text(encoding="utf-8"))
    marker = "## Open items"
    if "## The hidden-form teacher" in body:
        print("SAME-LANE-TEACHER.md already carries the section; leaving as is")
    else:
        assert marker in body, "could not find '## Open items' anchor"
        body = body.replace(marker, addition.rstrip() + "\n\n" + marker, 1)
        slt.write_text(body, encoding="utf-8")
        print(f"updated {slt}")

    # ---- 2. the WHAT-WE-MEASURE pointer: scratchpad only -------------------
    pointer = fill((args.scratch / "wwm_pointer.tmpl").read_text(encoding="utf-8"))
    out = args.scratch / "WHAT-WE-MEASURE-section2-pointer.md"
    out.write_text(pointer, encoding="utf-8")
    print(f"wrote {out} (apply to WHAT-WE-MEASURE.md section 2 AFTER rebase -- "
          "that file has another workflow's uncommitted changes)")

    # ---- 3. journal entry draft ------------------------------------------
    journal = fill((args.scratch / "journal_entry.md.tmpl").read_text(encoding="utf-8"))
    jout = args.scratch / "JOURNAL-entry.md"
    jout.write_text(journal, encoding="utf-8")
    print(f"wrote {jout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
