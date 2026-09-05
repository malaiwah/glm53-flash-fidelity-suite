#!/usr/bin/env python3
"""Where do a re-quantized checkpoint's fp16/bf16 NON-ROUTED tensors come from?

    .venv/bin/python engines/tools/nonrouted_provenance.py \
        --candidate drowzeys/keys-GLM-5.3-EXL3@ebf3c8bb... \
        --bf16-root zai-org/GLM-5.3-BF16@304b8051... \
        --fp8-release zai-org/GLM-5.3@187fb9ff... \
        --tensor model.layers.0.self_attn.q_a_proj.weight [--tensor ...] \
        --covers-class attn.qkv [--covers-class ...] \
        --rows 576 --out engines/tools/layer-outer-evidence/<name>.json

An EXL3 release built from the FP8 release carries, for every tensor it did not
trellis-quantize, the FP8 release's DEQUANTIZED values stored at 16 bits -- not
the BF16 release's values. A scope tool that reads only the stored dtype labels
such a class `native:fp16`, which is what the bytes say about STORAGE and wrong
about TREATMENT: the class carries an 8-bit quantization. This tool settles the
question for named tensors, spend-free, by range-reading the first `--rows` rows
of each tensor from all three repositories (safetensors header offsets + HTTP
Range; rows are contiguous in row-major storage, so no shard is downloaded) and
testing bitwise equality of the candidate rows against

    fp16(bf16_root_rows)                              -- the BF16 release, and
    fp16(dequantize_block_fp8(fp8_rows, scale_rows, fp32))
                                                      -- the FP8 release,

with `layer_outer.dequantize_block_fp8` (the same reference arithmetic the
capture path uses). Exactly one of the two must hold for the evidence to be
usable; `--covers-class` names the registry tensor classes the sampled tensors
stand for, and `engines/tools/scope_apply_provenance.py` rewrites ONLY those
classes of an authored scope. The output records repos, revisions, shards,
tensor names, rows compared, dtypes, counts, the dequant function and the
commit of the file that defines it, the torch version, the date, and this
script's own digest.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EVIDENCE_SCHEMA = "fidelity.nonrouted-provenance.v1"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
DTYPES = {"BF16": "bfloat16", "F16": "float16", "F8_E4M3": "float8_e4m3fn", "F32": "float32"}
ITEMSIZE = {"BF16": 2, "F16": 2, "F8_E4M3": 1, "F32": 4}


def _fail(msg: str) -> None:
    raise SystemExit("nonrouted_provenance: REFUSED: %s" % msg)


def _get(url: str, rng: str | None = None) -> bytes:
    headers = {"User-Agent": "quant-fidelity-suite/nonrouted_provenance"}
    if rng is not None:
        headers["Range"] = "bytes=%s" % rng
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    if rng is not None and resp.status != 206:
        _fail("%s did not honour Range %s (HTTP %s)" % (url, rng, resp.status))
    return data


class _Repo:
    """Ranged reads against one pinned repository: index once, headers per shard."""

    def __init__(self, spec: str) -> None:
        if "@" not in spec:
            _fail("repository spec %r must be owner/name@<40-hex revision>" % spec)
        self.repo, self.revision = spec.split("@", 1)
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            _fail("revision %r is not a 40-hex commit" % self.revision)
        self.index = json.loads(_get(self._url("model.safetensors.index.json")))
        self._headers: dict[str, tuple[int, dict]] = {}

    def _url(self, rel: str) -> str:
        return "%s/%s/resolve/%s/%s" % (HF, self.repo, self.revision, rel)

    def header(self, shard: str) -> tuple[int, dict]:
        if shard not in self._headers:
            url = self._url(shard)
            n = struct.unpack("<Q", _get(url, "0-7"))[0]
            self._headers[shard] = (n, json.loads(_get(url, "8-%d" % (7 + n))))
        return self._headers[shard]

    def has(self, name: str) -> bool:
        return name in self.index["weight_map"]

    def rows(self, name: str, rows: int):
        """The first `rows` rows of a 2-D tensor as a torch tensor, plus its record."""
        import torch

        if not self.has(name):
            _fail("%s@%s has no tensor %s" % (self.repo, self.revision[:8], name))
        shard = self.index["weight_map"][name]
        n, hdr = self.header(shard)
        meta = hdr[name]
        shape = list(meta["shape"])
        if len(shape) != 2:
            _fail("%s in %s is %s-D; this tool compares 2-D weights" % (name, self.repo, len(shape)))
        take = min(rows, shape[0])
        a, b = meta["data_offsets"]
        nbytes = take * shape[1] * ITEMSIZE[meta["dtype"]]
        if a + nbytes > b:
            _fail("%s: %d rows exceed the tensor's byte extent" % (name, take))
        raw = _get(self._url(shard), "%d-%d" % (8 + n + a, 8 + n + a + nbytes - 1))
        if len(raw) != nbytes:
            _fail("%s: Range read returned %d bytes, wanted %d" % (name, len(raw), nbytes))
        t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, DTYPES[meta["dtype"]]))
        return t.reshape(take, shape[1]), {"dtype": meta["dtype"], "shape": shape, "shard": shard,
                                            "rows_read": take, "bytes_read": nbytes}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(HERE), text=True).strip()


def compare_tensor(name: str, rows: int, cand: _Repo, root: _Repo, fp8: _Repo) -> dict:
    import torch

    import layer_outer as LO

    c, cm = cand.rows(name, rows)
    r, rm = root.rows(name, rows)
    q, qm = fp8.rows(name, rows)
    if qm["dtype"] != "F8_E4M3":
        _fail("%s in the FP8 release is stored %s, not F8_E4M3; the provenance question "
              "does not arise for it" % (name, qm["dtype"]))
    if cm["dtype"] not in ("F16", "BF16"):
        _fail("%s in the candidate is stored %s; this tool compares 16-bit storage" % (name, cm["dtype"]))
    if rm["shape"] != qm["shape"]:
        _fail("%s: the BF16 and FP8 releases disagree on shape: %s vs %s" % (name, rm["shape"], qm["shape"]))
    if cm["shape"][1] != rm["shape"][1] or cm["shape"][0] < rm["shape"][0]:
        _fail("%s: candidate shape %s is not the source shape %s (or a row-padded copy of it)"
              % (name, cm["shape"], rm["shape"]))
    if cm["rows_read"] != rm["rows_read"]:
        # The candidate has more rows than the source (drowzeys zero-pads
        # kv_a_proj_with_mqa 576 -> 640; the tail is proven zero elsewhere).
        # Compare the rows the source has.
        c = c[: rm["rows_read"]]
        cm = dict(cm, rows_read=rm["rows_read"])
    scale_name = name + "_scale_inv"
    if not fp8.has(scale_name):
        _fail("%s has no %s in the FP8 release" % (name, scale_name))
    # The full scale grid is small (ceil(rows/128) x ceil(cols/128) fp32); the
    # first ceil(take/128) scale rows govern exactly the rows read.
    take = cm["rows_read"]
    s_rows = -(-take // 128)
    s, sm = fp8.rows(scale_name, s_rows)
    if sm["dtype"] != "F32":
        _fail("%s is stored %s, expected F32" % (scale_name, sm["dtype"]))
    stored = getattr(torch, DTYPES[cm["dtype"]])
    deq32 = LO.dequantize_block_fp8(q, s, torch.float32)
    deq_stored = deq32.to(stored)
    root_stored = r.to(stored)
    eq_root = bool(torch.equal(c, root_stored))
    eq_fp8 = bool(torch.equal(c, deq_stored))
    return {
        "candidate": cm, "bf16_root": rm, "fp8_release": dict(qm, scale_inv=scale_name,
                                                             scale_rows_read=int(sm["rows_read"])),
        "rows_compared": take, "elements_compared": int(c.numel()),
        "stored_dtype": cm["dtype"],
        "eq_stored(bf16_root)": eq_root,
        "eq_stored(dequantize_block_fp8(fp8_release, fp32))": eq_fp8,
        "n_diff_vs_bf16_root": int((c != root_stored).sum()),
        "n_diff_vs_fp8_dequant": int((c != deq_stored).sum()),
        "max_abs_diff_vs_bf16_root": float((c.float() - r.float()).abs().max()),
        "max_abs_diff_vs_fp8_dequant_fp32": float((c.float() - deq32).abs().max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True, help="owner/name@40hex of the re-quantized release")
    ap.add_argument("--bf16-root", required=True, help="owner/name@40hex of the BF16 release")
    ap.add_argument("--fp8-release", required=True, help="owner/name@40hex of the block-FP8 release")
    ap.add_argument("--tensor", action="append", required=True, help="2-D weight name; repeatable")
    ap.add_argument("--covers-class", action="append", default=[],
                    help="registry tensor class the sampled tensors stand for; repeatable")
    ap.add_argument("--rows", type=int, default=576, help="leading rows compared per tensor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.rows < 1:
        _fail("--rows must be >= 1")

    import torch

    cand, root, fp8 = _Repo(args.candidate), _Repo(args.bf16_root), _Repo(args.fp8_release)
    tensors: dict[str, dict] = {}
    for name in args.tensor:
        res = compare_tensor(name, args.rows, cand, root, fp8)
        tensors[name] = res
        print("%s: eq_root=%s eq_fp8_dequant=%s (n_diff root %d / fp8 %d of %d)"
              % (name, res["eq_stored(bf16_root)"],
                 res["eq_stored(dequantize_block_fp8(fp8_release, fp32))"],
                 res["n_diff_vs_bf16_root"], res["n_diff_vs_fp8_dequant"], res["elements_compared"]))
    verdicts = {(t["eq_stored(bf16_root)"], t["eq_stored(dequantize_block_fp8(fp8_release, fp32))"])
                for t in tensors.values()}
    if len(verdicts) != 1:
        _fail("the sampled tensors disagree on their source: %s; no single verdict can cover a class"
              % sorted(verdicts))
    eq_root, eq_fp8 = verdicts.pop()
    if eq_root == eq_fp8:
        _fail("every sampled tensor is %s to BOTH candidates; the evidence decides nothing"
              % ("equal" if eq_root else "unequal"))
    verdict = ("stored_16bit_of_fp8_release_dequantized" if eq_fp8
               else "stored_16bit_of_bf16_root")
    script = Path(__file__).resolve()
    lo = HERE / "layer_outer.py"
    doc = {
        "schema": EVIDENCE_SCHEMA,
        "question": "Do the candidate's 16-bit non-routed tensors carry the BF16 release's values "
                    "or the FP8 release's dequantized values?",
        "verdict": verdict,
        "verdict_text": {
            "stored_16bit_of_fp8_release_dequantized":
                "bitwise equal to fp16(dequantize_block_fp8(FP8 release, fp32 output)) and NOT equal to "
                "fp16(BF16 release) on every sampled tensor: the class carries the FP8 release's 8-bit "
                "block quantization stored at 16 bits (treatment quantized, format fp8_e4m3, 8 bits).",
            "stored_16bit_of_bf16_root":
                "bitwise equal to the BF16 release cast to the stored dtype and NOT equal to the FP8 "
                "release dequantized: the class is native (a 16-bit round trip of the source).",
        }[verdict],
        "covers_classes": sorted(set(args.covers_class)),
        "candidate": {"repo": cand.repo, "revision": cand.revision},
        "bf16_root": {"repo": root.repo, "revision": root.revision},
        "fp8_release": {"repo": fp8.repo, "revision": fp8.revision},
        "rows_requested": args.rows,
        "tensors": tensors,
        "method": {
            "reads": "safetensors header (8-byte length + JSON) and the leading rows of each tensor by "
                     "HTTP Range against huggingface.co/<repo>/resolve/<revision>/<shard>; no shard "
                     "downloaded; rows are contiguous in row-major storage.",
            "dequantize": "engines/tools/layer_outer.py:dequantize_block_fp8(q, scale_inv, torch.float32) "
                          "-- fp8 -> fp32, one multiply by the fp32 128x128 block scale, one RNE cast; the "
                          "first ceil(rows/128) scale rows govern exactly the rows read.",
            "equality": "torch.equal on the candidate's stored dtype after casting each reference to it "
                        "(RNE); counts and max |diff| computed in fp32.",
            "layer_outer_commit": _git("log", "-1", "--format=%H", "--", str(lo)),
            "layer_outer_sha256": hashlib.sha256(lo.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
        "produced_by": {
            "script": str(script.relative_to(HERE.parent.parent)),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "repo_commit": _git("rev-parse", "HEAD"),
            "argv": [a for a in (argv if argv is not None else sys.argv[1:])],
            "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print("wrote %s: verdict %s over %d tensors, classes %s"
          % (out, verdict, len(tensors), ", ".join(doc["covers_classes"]) or "(none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
