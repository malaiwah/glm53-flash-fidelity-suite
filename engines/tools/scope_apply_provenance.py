#!/usr/bin/env python3
"""Apply a non-routed provenance verdict to an authored scope file.

    engines/tools/scope_apply_provenance.py --scope engines/scopes/scope--X.json \
        --evidence engines/tools/layer-outer-evidence/X-nonrouted-provenance.json \
        [--class attn.qkv ...] [--out path | --in-place]

`exl3_scope.py` / `fp8_scope.py` read a checkpoint's bytes and label a class by
how it is STORED: a class kept at fp16 is `native:fp16@16`. That is the right
default and it is wrong for one common construction -- an EXL3 release built
from a block-FP8 release keeps the FP8 release's dequantized values at 16 bits,
so the class carries an 8-bit quantization it does not declare. Only a byte
comparison against both possible sources can tell, and
`engines/tools/nonrouted_provenance.py` writes that comparison down as an
evidence file with an explicit `covers_classes` list.

This tool rewrites exactly the covered classes of an authored scope to
`treatment quantized, format fp8_e4m3, bits 8`, keeps the original census in
the note beside the evidence path, and refuses everything else: a class the
evidence does not cover, an evidence file whose verdict is not the FP8 one, a
covered class with no sampled tensor of its own, an assignment that is not a
16-bit native row, or a scope authored from a different repository/revision
than the evidence names. It prints the old and new `scope_digest` (the
registry's canonical form) so the correction can be disclosed with both.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "registry" / "tools"))

import fp8_scope  # noqa: E402
import registry_lib as L  # noqa: E402

EVIDENCE_SCHEMA = "fidelity.nonrouted-provenance.v1"
FP8_VERDICT = "stored_16bit_of_fp8_release_dequantized"
EQ_FP8 = "eq_stored(dequantize_block_fp8(fp8_release, fp32))"
EQ_ROOT = "eq_stored(bf16_root)"
NATIVE_16 = {("fp16", 16), ("bf16", 16)}


class ApplyError(Exception):
    pass


def _fail(msg: str) -> None:
    raise ApplyError("scope_apply_provenance: REFUSED: %s" % msg)


def _check_evidence(ev: dict) -> None:
    if ev.get("schema") != EVIDENCE_SCHEMA:
        _fail("evidence schema %r is not %s" % (ev.get("schema"), EVIDENCE_SCHEMA))
    if ev.get("verdict") != FP8_VERDICT:
        _fail("evidence verdict %r is not %r; only the FP8-provenance verdict rewrites a scope"
              % (ev.get("verdict"), FP8_VERDICT))
    if not ev.get("covers_classes"):
        _fail("evidence covers no classes")
    tensors = ev.get("tensors") or {}
    if not tensors:
        _fail("evidence samples no tensors")
    for name, t in tensors.items():
        if not (t.get(EQ_FP8) is True and t.get(EQ_ROOT) is False and t.get("n_diff_vs_fp8_dequant") == 0):
            _fail("sampled tensor %s does not carry the FP8 verdict (%s=%r, %s=%r, n_diff_vs_fp8_dequant=%r)"
                  % (name, EQ_FP8, t.get(EQ_FP8), EQ_ROOT, t.get(EQ_ROOT), t.get("n_diff_vs_fp8_dequant")))
    for k in ("candidate", "fp8_release"):
        if not ((ev.get(k) or {}).get("repo") and len((ev.get(k) or {}).get("revision", "")) == 40):
            _fail("evidence %s block lacks repo/40-hex revision" % k)


def _classes_sampled(ev: dict, num_layers: int) -> dict:
    out: dict = {}
    for name in ev["tensors"]:
        out.setdefault(fp8_scope.classify(name, num_layers), []).append(name)
    return out


def _decoder_layers_from_notes(scope: dict) -> int:
    """The scope carries no config; the largest layer index below the mtp row
    bounds the decoder. Only the classifier's mtp cut-off needs it."""
    top = -1
    for a in scope["assignments"]:
        rng = a.get("layer_range") or ""
        for m in re.finditer(r"\d+", rng):
            if a.get("tensor_class") != "mtp":
                top = max(top, int(m.group(0)))
    return top + 1 if top >= 0 else 10 ** 6


def apply(scope: dict, ev: dict, evidence_path: str, only: list[str] | None = None) -> dict:
    """Return a NEW scope dict with the covered classes rewritten. Raises ApplyError."""
    _check_evidence(ev)
    covers = list(ev["covers_classes"])
    wanted = list(only) if only else covers
    for c in wanted:
        if c not in covers:
            _fail("class %s is not covered by the evidence (covers: %s)" % (c, ", ".join(covers)))
    cand = ev["candidate"]
    origin = "%s@%s" % (cand["repo"], cand["revision"])
    sampled = _classes_sampled(ev, _decoder_layers_from_notes(scope))
    new = json.loads(json.dumps(scope))
    fp8 = ev["fp8_release"]
    rows_each = sorted({t["rows_compared"] for t in ev["tensors"].values()})
    for cls in wanted:
        if cls not in sampled:
            _fail("class %s is covered by the evidence but none of its sampled tensors classify into it"
                  % cls)
        hits = [a for a in new["assignments"] if a.get("tensor_class") == cls]
        if not hits:
            _fail("scope has no assignment for covered class %s" % cls)
        for a in hits:
            if origin not in (a.get("note") or ""):
                _fail("assignment %s was not authored from %s (note does not name it)" % (cls, origin))
            if a.get("treatment") != "native" or (a.get("format"), a.get("bits_per_weight")) not in NATIVE_16:
                _fail("assignment %s[%s] is %s:%s@%s, not a 16-bit native row; the evidence speaks only "
                      "to 16-bit-stored natives" % (cls, a.get("layer_range"), a.get("treatment"),
                                                    a.get("format"), a.get("bits_per_weight")))
            stored = a["format"]
            census = a.get("note") or ""
            a["treatment"] = "quantized"
            a["format"] = "fp8_e4m3"
            a["bits_per_weight"] = 8
            a["note"] = (
                "stored %s; values bitwise equal to %s(dequantize_block_fp8(%s@%s)) on the sampled rows "
                "(%s; %d rows each; 0 differing elements; sampled tensors of this class: %s), "
                "evidence %s. Original census: %s"
                % (stored, stored, fp8["repo"], fp8["revision"][:8],
                   "%d tensors" % len(ev["tensors"]), rows_each[0] if len(rows_each) == 1 else max(rows_each),
                   ", ".join(sorted(sampled[cls])), evidence_path, census))
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scope", required=True, help="authored scope file (exl3_scope.py / fp8_scope.py output)")
    ap.add_argument("--evidence", required=True, help="nonrouted_provenance.py evidence file")
    ap.add_argument("--class", dest="classes", action="append", default=None,
                    help="restrict to these covered classes (default: every class the evidence covers)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out")
    g.add_argument("--in-place", action="store_true")
    args = ap.parse_args(argv)

    scope = json.loads(Path(args.scope).read_text(encoding="utf-8"))
    ev = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    repo_root = HERE.parent.parent
    try:
        ev_rel = str(Path(args.evidence).resolve().relative_to(repo_root))
    except ValueError:
        ev_rel = args.evidence
    try:
        new = apply(scope, ev, ev_rel, args.classes)
    except ApplyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    old_digest, new_digest = L.scope_digest(scope), L.scope_digest(new)
    out = Path(args.scope) if args.in_place else Path(args.out)
    out.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    changed = sorted({a["tensor_class"] for a, b in zip(new["assignments"], scope["assignments"]) if a != b})
    print("rewrote %s: classes %s" % (out, ", ".join(changed)))
    print("scope_digest old: %s" % old_digest)
    print("scope_digest new: %s" % new_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
