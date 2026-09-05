#!/usr/bin/env python3
"""Author an exact unexpected-tensor allowlist by INDEX CENSUS -- $0, seconds.

This is the method that produced every GLM-5.3 allowlist committed under
`engines/tools/layer-outer-evidence/` (their `.provenance.json` sidecars say
`derived_by: "index census: every key of model.layers.78 ..."`), written down
as a tool. `derive_unexpected_allowlist.py` is the loader-exact derivation and
needs the whole checkpoint on disk; this one needs two small files.

The census: `model.safetensors.index.json` lists every tensor the checkpoint
carries; `config.json` declares how many decoder layers the architecture
builds (`num_hidden_layers`, or `text_config.num_hidden_layers` for the nested
VL stack). Every key whose layer index is at or past that count belongs to a
block `transformers` never instantiates -- the MTP block on GLM-5.3 -- and the
streamed loader (`hf_capture.py --schedule layer-outer`) reports exactly those
keys as unexpected, per layer, straight from the index. The allowlist the
capture binds by SHA-256 is therefore the sorted set of those keys, and
nothing else: a tensor the architecture DOES declare is never allowlisted
here, because an unconsumed declared tensor is a broken load, not a census.

    engines/tools/index_census_allowlist.py \\
        --index model.safetensors.index.json --config config.json \\
        --repo owner/name --revision 40hex \\
        --out engines/tools/layer-outer-evidence/X-unexpected-keys.json

Without `--index`/`--config` the two files are fetched anonymously from
huggingface.co at the pinned revision (they are public metadata). The tool
writes `X.json` in the committed shape (a JSON array, indent 1, sorted, unique)
plus `X.json.provenance.json`, and prints the three digests
`bin/fidelity/runpodsafety.py`'s `_ALLOWLISTS` row records: `artifact_sha256`
(the file bytes), `canonical_sorted_names_sha256` (the canonical JSON of the
names, the digest the capture binds) and `count`.

Only safetensors indexes are read today. A GGUF carries the same census in
its header key names; feed those names through `census()` once
`gguf_surface.py` exposes them -- the rule (layer index >= declared layers)
does not change.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from fp8_scope import STACK, decoder_layers

LAYER_RE = re.compile(STACK + r"layers\.(\d+)\.")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
HF_RESOLVE = "https://huggingface.co/%s/resolve/%s/%s"


def _fail(message: str) -> SystemExit:
    return SystemExit("index_census_allowlist: ERROR: %s" % message)


def canonical_names_sha256(names: List[str]) -> str:
    return hashlib.sha256(
        json.dumps(names, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def census(names, layers: int) -> Tuple[List[str], Dict[str, int]]:
    """Sorted unique keys whose layer index >= `layers`, plus per-layer counts.

    Pure over a name iterable: a safetensors weight_map today, a GGUF header's
    tensor names when that surface exposes them.
    """
    picked = set()
    for name in names:
        match = LAYER_RE.match(name)
        if match is not None and int(match.group(1)) >= layers:
            picked.add(name)
    per_layer: Dict[str, int] = {}
    for name in picked:
        layer = LAYER_RE.match(name).group(1)
        per_layer[layer] = per_layer.get(layer, 0) + 1
    return sorted(picked), dict(sorted(per_layer.items(), key=lambda kv: int(kv[0])))


def _load(path_or_none, repo, revision, filename) -> Tuple[bytes, str]:
    if path_or_none is not None:
        raw = Path(path_or_none).read_bytes()
        return raw, str(path_or_none)
    url = HF_RESOLVE % (repo, revision, filename)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - pinned https URL
            return resp.read(), url
    except OSError as exc:
        raise _fail("cannot fetch %s: %s" % (url, exc))


def _strict_json(raw: bytes, what: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail("%s is not JSON: %s" % (what, exc))


def build(index_raw: bytes, config_raw: bytes, *, repo: str, revision: str,
          relation: str = None) -> Tuple[bytes, dict]:
    index = _strict_json(index_raw, "index")
    config = _strict_json(config_raw, "config")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise _fail("index carries no weight_map")
    layers = decoder_layers(config)
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else None
    if not isinstance(architecture, str):
        raise _fail("config declares no architectures[0]")
    names, per_layer = census(weight_map.keys(), layers)
    if not names:
        raise _fail("index has no key at or past layer %d; nothing to allowlist "
                    "(a checkpoint with no never-built block needs no allowlist)"
                    % layers)
    stack_prefix = "model.language_model.layers" if any(
        n.startswith("model.language_model.layers.") for n in names) else "model.layers"
    blocks = ", ".join("%s.%s" % (stack_prefix, k) for k in per_layer)
    artifact = (json.dumps(names, indent=1) + "\n").encode("utf-8")
    provenance = {
        "architecture": architecture,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "canonical_sorted_names_sha256": canonical_names_sha256(names),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "count": len(names),
        "decoder_layers": layers,
        "derived_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "derived_by": ("index census: every key of %s (the block %s never builds; "
                       "num_hidden_layers is %d); engines/tools/index_census_allowlist.py"
                       % (blocks, architecture, layers)),
        "derived_from": "%s@%s model.safetensors.index.json" % (repo, revision),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "per_layer_unexpected_counts": per_layer,
    }
    if relation:
        provenance["relation_to_bf16_allowlist"] = relation
    return artifact, provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="owner/name of the checkpoint")
    parser.add_argument("--revision", required=True, help="40-hex commit the index was read at")
    parser.add_argument("--index", help="model.safetensors.index.json (fetched anonymously when omitted)")
    parser.add_argument("--config", help="config.json (fetched anonymously when omitted)")
    parser.add_argument("--out", required=True,
                        help="allowlist path; the provenance sidecar is <out>.provenance.json")
    parser.add_argument("--relation", default=None,
                        help="optional prose for relation_to_bf16_allowlist in the sidecar")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args(argv)

    if not _HEX40.match(args.revision):
        raise _fail("--revision must be the 40-hex commit, got %r" % args.revision)
    if (args.index is None) != (args.config is None):
        raise _fail("pass both --index and --config, or neither (both fetched)")
    out = Path(args.out)
    sidecar = out.with_name(out.name + ".provenance.json")
    if not args.force and (out.exists() or sidecar.exists()):
        raise _fail("%s or its sidecar exists; pass --force to overwrite" % out)

    index_raw, index_src = _load(args.index, args.repo, args.revision,
                                 "model.safetensors.index.json")
    config_raw, config_src = _load(args.config, args.repo, args.revision, "config.json")
    artifact, provenance = build(index_raw, config_raw, repo=args.repo,
                                 revision=args.revision, relation=args.relation)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(artifact)
    sidecar.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
    print("index_census_allowlist: read %s and %s" % (index_src, config_src))
    print("  architecture %s, decoder layers %d, unexpected keys per block %s"
          % (provenance["architecture"], provenance["decoder_layers"],
             json.dumps(provenance["per_layer_unexpected_counts"])))
    print("  wrote %s (+ %s)" % (out, sidecar.name))
    print("  bin/fidelity/runpodsafety.py _ALLOWLISTS row for (%r, %r):"
          % (args.repo, args.revision))
    for key in ("artifact_sha256", "canonical_sorted_names_sha256", "count"):
        print("    %-30s %s" % (key, json.dumps(provenance[key])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
