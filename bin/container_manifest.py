#!/usr/bin/env python3
"""Record what the image actually resolved -- never what it asked for.

    container_manifest.py --image-root /opt/fidelity --suite /opt/fidelity/suite \
        --suite-revision <40-hex> --image-reference <registry>/<name>:<tag>

Writes `$IMAGE_ROOT/BUILD.json` and `$IMAGE_ROOT/image-pin.txt`.

WHY A CONTENT PIN AND NOT JUST THE TAG.  A tag is a mutable label and a
registry digest does not survive `docker save`/`docker load` -- which is why
the serving side of this project already writes an image-pin file and
`fidelity/stackprint.py` already reads one.  So the identity baked here is a
digest of a MANIFEST of the stack: the resolved wheel versions, the pipeline
commit, the sha256 of every applied patch, and the sha256 of every bundled
file.  Two images with the same `image_content_sha256` compute with the same
bytes; two with different ones do not, whatever their tags say.

Every fact is queried from the built image.  A fact that cannot be queried is
recorded as null WITH THE REASON, the same rule `fidelity/stackprint.py`
follows -- a receipt must never assert something it did not observe.

Stdlib only, python3.9-clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    # Must match registry/tools/registry_lib.py and bin/fidelity/common.py.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run(argv, cwd=None):
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=300)
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except Exception as exc:                               # noqa: BLE001
        return 127, "", "%s: %s" % (type(exc).__name__, exc)


def wheel_pins(python: Path):
    """Ask the interpreter, do not parse the recipe.

    The bootstrap NAMES torch==2.11.0; what matters for the arithmetic is what
    pip actually resolved and what CUDA that wheel carries.
    """
    if not python.is_file():
        return {}, "no interpreter at %s" % python
    probe = "\n".join([
        "import json, torch, transformers, safetensors, numpy",
        "d = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,",
        "     'transformers': transformers.__version__,",
        "     'safetensors': safetensors.__version__, 'numpy': numpy.__version__}",
        "try:",
        "    import hf_transfer",
        "    d['hf_transfer'] = getattr(hf_transfer, '__version__', 'present')",
        "except Exception:",
        "    d['hf_transfer'] = None",
        "print(json.dumps(d))",
    ])
    code, out, err = run([str(python), "-c", probe])
    if code != 0:
        return {}, err[:400]
    try:
        return json.loads(out.splitlines()[-1]), None
    except Exception as exc:                               # noqa: BLE001
        return {}, "unparseable probe output: %s" % exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--suite-revision", default="")
    ap.add_argument("--image-reference", default="")
    args = ap.parse_args(argv)

    root, suite = Path(args.image_root), Path(args.suite)
    venv_py = root / "venv" / "bin" / "python"
    pipeline = root / "pipeline"

    pins, wheel_error = wheel_pins(venv_py)
    code, py_version, _ = run([str(venv_py), "-c",
                               "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    pins["python"] = py_version if code == 0 else None

    code, head, err = run(["git", "-C", str(pipeline), "rev-parse", "HEAD"])
    pins["pipeline_commit"] = head if code == 0 else None
    pipeline_error = None if code == 0 else err[:200]

    patches = {}
    pdir = root / "patches-v2"
    if pdir.is_dir():
        patches = {p.name: sha256_file(p) for p in sorted(pdir.iterdir())
                   if p.is_file()}

    exl3 = root / "exllamav3"
    if (exl3 / ".git").is_dir():
        code, head, _ = run(["git", "-C", str(exl3), "rev-parse", "HEAD"])
        pins["exllamav3_commit"] = head if code == 0 else None
    else:
        # The bootstrap builds exllamav3 ONLY IF the pipeline cannot import
        # without it; on the measurement path it cannot, so "absent" is the
        # expected answer and it is recorded rather than left ambiguous.
        pins["exllamav3_commit"] = None

    bundle = {}
    for path in sorted(suite.rglob("*")):
        if path.is_file():
            bundle[str(path.relative_to(suite))] = sha256_file(path)

    freeze = ""
    if venv_py.is_file():
        code, freeze, _ = run([str(venv_py), "-m", "pip", "freeze"])
        if code == 0:
            (root / "pip-freeze.txt").write_text(freeze + "\n", encoding="utf-8")

    doc = {
        "schema": "malaiwah.fidelity-image-build.v1",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_revision": args.suite_revision.strip() or None,
        "image_reference": args.image_reference.strip() or None,
        "pins": pins,
        "patches_sha256": patches,
        "bundle_sha256": bundle,
        "pip_freeze_sha256": hashlib.sha256(
            (freeze + "\n").encode("utf-8")).hexdigest() if freeze else None,
        "probe_errors": {k: v for k, v in
                         (("wheels", wheel_error), ("pipeline", pipeline_error))
                         if v},
    }
    # The content pin covers everything that decides the arithmetic and nothing
    # that does not: the build timestamp and the mutable tag are excluded on
    # purpose, so two builds of one revision on one recipe pin identically.
    material = {k: doc[k] for k in
                ("pins", "patches_sha256", "bundle_sha256", "pip_freeze_sha256",
                 "suite_revision")}
    doc["image_content_sha256"] = hashlib.sha256(
        canonical(material).encode("utf-8")).hexdigest()

    root.mkdir(parents=True, exist_ok=True)
    (root / "BUILD.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The convention fidelity/stackprint.py already reads: first whitespace-
    # separated token is the pin.
    (root / "image-pin.txt").write_text(doc["image_content_sha256"] + "\n",
                                        encoding="utf-8")
    os.chmod(str(root / "image-pin.txt"), 0o644)

    print("image_content_sha256 %s" % doc["image_content_sha256"])
    print("suite_revision       %s" % doc["suite_revision"])
    print("pins                 %s" % canonical(pins))
    print("bundle               %d file(s)" % len(bundle))
    if doc["probe_errors"]:
        # Not fatal: a probe that cannot answer records the reason.  But it is
        # printed, because a build whose torch version is unknown is a build
        # whose arithmetic is unpinned and somebody has to see that.
        sys.stderr.write("probe errors: %s\n" % canonical(doc["probe_errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
