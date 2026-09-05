#!/usr/bin/env python3
"""Offline checks for `index_census_allowlist.py`, the index-census allowlist tool.

Stock Python, no network: a synthetic index/config pair exercises the census
rule (layer index >= declared layers, on the flat and the nested VL stack),
the committed file shape (JSON array, indent 1, sorted, unique, trailing
newline) and the three digests the `_ALLOWLISTS` table records; then every
committed allowlist whose provenance sidecar says "index census" is re-hashed
against its own sidecar so the shape contract and the evidence agree.

Rungs:
  R1  flat stack: keys of layers >= num_hidden_layers, sorted, unique; per-layer counts
  R2  nested VL stack: text_config.num_hidden_layers places the boundary
  R3  a checkpoint with nothing past the boundary is a refusal, not an empty list
  R4  artifact bytes are the committed shape; artifact/canonical digests match
      the file and the runpodsafety canonical-bytes rule
  R5  every committed index-census allowlist re-hashes to its sidecar
  R6  CLI: --out written, sidecar written, digests printed, --force required to overwrite
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "bin"))

import index_census_allowlist as ica  # noqa: E402
from fidelity.runpodsafety import canonical_bytes  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  (%s)" % detail) if (detail and not ok) else ""))
    if not ok:
        failures.append(name)


def _index(names):
    return json.dumps({"metadata": {}, "weight_map": {n: "model-00001.safetensors" for n in names}}).encode()


def _config(arch, layers, nested=False):
    doc = {"architectures": [arch]}
    if nested:
        doc["text_config"] = {"num_hidden_layers": layers}
    else:
        doc["num_hidden_layers"] = layers
    return json.dumps(doc).encode()


def main():
    flat = ["model.layers.1.mlp.down_proj.weight", "model.layers.2.b", "model.layers.2.a",
            "model.layers.3.x", "model.layers.10.y", "model.embed_tokens.weight",
            "lm_head.weight", "model.layers.2.a"]
    names, per_layer = ica.census(flat, 2)
    check("R1: flat census picks layers >= 2, sorted, unique",
          names == ["model.layers.10.y", "model.layers.2.a", "model.layers.2.b",
                    "model.layers.3.x"], repr(names))
    check("R1: per-layer counts in numeric order",
          list(per_layer.items()) == [("2", 2), ("3", 1), ("10", 1)], repr(per_layer))

    nested = ["model.language_model.layers.44.a", "model.language_model.layers.45.b",
              "model.visual.blocks.0.w"]
    artifact, prov = ica.build(_index(nested), _config("Glm5NextForConditionalGeneration", 45, nested=True),
                               repo="o/n", revision="0" * 40)
    check("R2: nested VL stack uses text_config.num_hidden_layers",
          json.loads(artifact) == ["model.language_model.layers.45.b"]
          and prov["decoder_layers"] == 45
          and "model.language_model.layers.45" in prov["derived_by"], prov["derived_by"])

    try:
        ica.build(_index(["model.layers.0.a"]), _config("X", 1), repo="o/n", revision="0" * 40)
        check("R3: nothing past the boundary refuses", False, "returned")
    except SystemExit as exc:
        check("R3: nothing past the boundary refuses", "nothing to allowlist" in str(exc), str(exc))

    artifact, prov = ica.build(_index(flat), _config("GlmMoeDsaForCausalLM", 2), repo="o/n", revision="1" * 40)
    expect = ["model.layers.10.y", "model.layers.2.a", "model.layers.2.b", "model.layers.3.x"]
    check("R4: artifact bytes are the committed shape (indent 1, trailing newline)",
          artifact == (json.dumps(expect, indent=1) + "\n").encode())
    check("R4: artifact_sha256 is the file digest",
          prov["artifact_sha256"] == hashlib.sha256(artifact).hexdigest())
    check("R4: canonical digest equals runpodsafety.canonical_bytes of the names",
          prov["canonical_sorted_names_sha256"] == hashlib.sha256(canonical_bytes(expect)).hexdigest())
    check("R4: count and derived_from", prov["count"] == 4 and prov["derived_from"].startswith("o/n@1111"))

    evidence = ROOT / "engines" / "tools" / "layer-outer-evidence"
    seen = 0
    for sidecar in sorted(evidence.glob("*.json.provenance.json")):
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if not str(prov.get("derived_by", "")).startswith("index census"):
            continue
        seen += 1
        raw = sidecar.with_name(sidecar.name[:-len(".provenance.json")]).read_bytes()
        names = json.loads(raw)
        ok = (hashlib.sha256(raw).hexdigest() == prov["artifact_sha256"]
              and hashlib.sha256(canonical_bytes(names)).hexdigest() == prov["canonical_sorted_names_sha256"]
              and len(names) == prov["count"] and names == sorted(set(names))
              and raw == (json.dumps(names, indent=1) + "\n").encode("utf-8"))
        check("R5: %s re-hashes to its sidecar" % sidecar.name[:-len(".json.provenance.json")], ok)
    check("R5: found committed index-census allowlists", seen >= 5, "%d found" % seen)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "index.json").write_bytes(_index(flat))
        (tmp / "config.json").write_bytes(_config("GlmMoeDsaForCausalLM", 2))
        cmd = [sys.executable, str(HERE / "index_census_allowlist.py"), "--repo", "o/n",
               "--revision", "2" * 40, "--index", str(tmp / "index.json"),
               "--config", str(tmp / "config.json"), "--out", str(tmp / "x.json")]
        run = subprocess.run(cmd, capture_output=True, text=True)
        check("R6: CLI exit 0", run.returncode == 0, run.stderr.strip()[-200:])
        check("R6: CLI writes the allowlist and sidecar",
              (tmp / "x.json").is_file() and (tmp / "x.json.provenance.json").is_file())
        sha = hashlib.sha256((tmp / "x.json").read_bytes()).hexdigest()
        check("R6: CLI prints the artifact sha256 and count", sha in run.stdout and "count" in run.stdout)
        again = subprocess.run(cmd, capture_output=True, text=True)
        check("R6: CLI refuses to overwrite without --force",
              again.returncode != 0 and "--force" in again.stderr)
        forced = subprocess.run(cmd + ["--force"], capture_output=True, text=True)
        check("R6: --force overwrites", forced.returncode == 0)

    print()
    if failures:
        print("selftest_index_census_allowlist: %d FAILED" % len(failures))
        return 1
    print("selftest_index_census_allowlist: all rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
