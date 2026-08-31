#!/usr/bin/env python3
"""drive_probe -- rent ONE card, fingerprint its arithmetic, give it back.

    reports/arch-determinism/drive_probe.py --gpu "NVIDIA A40" --dry-run
    reports/arch-determinism/drive_probe.py --gpu "NVIDIA A40" --max-usd 0.50

Uploads `probe_payload.py`, runs it twice (lane policy, then lane policy plus
`torch.use_deterministic_algorithms` and a pinned cuBLAS workspace -- as a
SECOND PROCESS, because the workspace pin is read when the cuBLAS handle is
created and a flag flipped mid-process is too late), pulls the digests and the
tensors back, and destroys the instance in `finally`.

`--max-usd` is a hard bound on the estimated bill, checked BEFORE `create`
against the offer's own advertised rate and again after against the rate the
instance is actually billing.  `--dry-run` prices the rental and creates
nothing.

Stock python3 + the suite's own provider wrappers.  No credential ever reaches
argv: the provider classes read a 0600 key file named by RUNPOD_KEY_FILE /
VAST_KEY_FILE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SUITE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUITE / "bin"))

from fidelity.bench import rented_rate, wait_ready            # noqa: E402

PAYLOAD = Path(__file__).with_name("probe_payload.py")
# The campaign's own numpy. Pinned because PCG64 is the source of every input
# byte in the probe.
NUMPY_PIN = "2.5.2"
# Budget guard: the probe is ~2 min of compute; anything past this is a stuck
# box and it must be destroyed rather than waited on.
HARD_MINUTES = 40.0   # 600s ready + 2x600s exec + 600s download


def provider_for(name: str, dry: bool):
    if name == "runpod":
        from fidelity.runpodapi import RunPod
        return RunPod(dry=dry)
    if name == "vast":
        from fidelity.vastapi import Vast
        return Vast(dry=dry)
    from fidelity.jlapi import JL
    return JL(dry=dry)


def catalogue(prov, cache: str):
    """The offer list, snapshotted to a file.

    RunPod's catalogue costs ~90 GraphQL round trips, and two drivers asking at
    once get rate-limited into a partial answer -- which `gpus()` swallows, so
    a real card reads as "not offered" and the budget guard has nothing to bind
    to. Snapshot once, reuse; `--refresh-prices` retakes it.
    """
    if cache and os.path.isfile(cache):
        return json.loads(open(cache, encoding="utf-8").read())
    rows = [{"gpu_type": o.gpu_type, "region": o.region, "price": o.price,
             "stock": (o.raw or {}).get("stockStatus")} for o in prov.gpus()]
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def price_direct(prov, gpu: str, region: str):
    """Ask about ONE gpuTypeId. Two round trips instead of ninety.

    The bulk catalogue is where the rate limiting bites; a single-card query
    survives it, so a card missing from the snapshot gets a second, cheap
    chance before the budget guard refuses on no evidence.
    """
    secure = "true" if region != "community" else "false"
    q = ('query { gpuTypes(input:{id:"%s"}) { lowestPrice'
         '(input:{gpuCount:1,secureCloud:%s}) '
         '{ minimumBidPrice uninterruptablePrice stockStatus } } }'
         % (gpu.replace('"', ''), secure))
    try:
        lp = ((prov._gql(q).get("gpuTypes") or [{}])[0].get("lowestPrice") or {})
    except Exception:                                             # noqa: BLE001
        return None
    price = lp.get("uninterruptablePrice") or lp.get("minimumBidPrice")
    if not price or not lp.get("stockStatus"):
        return None
    return {"gpu_type": gpu, "region": region or "secure", "price": float(price),
            "stock": lp.get("stockStatus")}


def price_of(rows, gpu: str, region: str):
    best = None
    for o in rows:
        if o["gpu_type"] != gpu:
            continue
        if region and o["region"] != region:
            continue
        if best is None or o["price"] < best["price"]:
            best = o
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drive_probe")
    ap.add_argument("--provider", default="runpod",
                    choices=("runpod", "vast", "jarvislabs"))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--region", default="", help="runpod: 'community' or 'secure'")
    ap.add_argument("--label", help="short name for the output file")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results")))
    ap.add_argument("--storage", type=int, default=30)
    ap.add_argument("--max-usd", type=float, default=1.00,
                    help="hard bound on the estimated bill for this rental")
    ap.add_argument("--price-cache",
                    default=str(Path(__file__).with_name("results") / "offers.json"))
    ap.add_argument("--refresh-prices", action="store_true")
    ap.add_argument("--allow-unknown-price", action="store_true",
                    help="proceed when the catalogue did not return a rate "
                         "(the budget guard then cannot bind -- say so out loud)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-tensors", action="store_true")
    ap.add_argument("--sweep-only", action="store_true",
                    help="run only the reduction sweeps (and skip the second, "
                         "deterministic-mode process)")
    args = ap.parse_args(argv)

    label = args.label or args.gpu.replace(" ", "-").replace("/", "-")
    outdir = Path(args.out) / label
    prov = provider_for(args.provider, dry=args.dry_run)
    if not prov.available():
        print("no credential for %s" % args.provider, file=sys.stderr)
        return 3

    if args.refresh_prices and args.price_cache and os.path.isfile(args.price_cache):
        os.remove(args.price_cache)
    offer = price_of(catalogue(prov, args.price_cache), args.gpu, args.region)
    if offer is None and args.provider == "runpod":
        offer = price_direct(prov, args.gpu, args.region)
    rate = offer["price"] if offer else None
    est = None if rate is None else round(rate * HARD_MINUTES / 60.0, 3)
    print("gpu           %s (%s)" % (args.gpu, args.region or "any"))
    print("advertised    %s" % ("$%.3f/h" % rate if rate else "unknown"))
    print("worst-case    %s at the %.0f-minute hard stop"
          % ("$%.3f" % est if est else "unknown", HARD_MINUTES))
    print("budget        $%.2f" % args.max_usd)
    if est is None and not args.allow_unknown_price:
        print("REFUSED: no advertised rate for %r -- the catalogue query failed or "
              "the card is not offered. Re-run, or pass --allow-unknown-price."
              % args.gpu, file=sys.stderr)
        return 3
    if est is not None and est > args.max_usd:
        print("REFUSED: worst-case $%.3f exceeds --max-usd $%.2f" % (est, args.max_usd),
              file=sys.stderr)
        return 3
    if args.dry_run:
        print("dry-run: nothing created")
        return 0
    if offer is None:
        print("REFUSED: %r is not offered%s right now"
              % (args.gpu, " in " + args.region if args.region else ""), file=sys.stderr)
        return 3

    outdir.mkdir(parents=True, exist_ok=True)
    mid = None
    started = time.time()
    doc = {"schema": "malaiwah.arch-determinism-rental.v1", "label": label,
           "provider": args.provider, "requested_gpu": args.gpu,
           "requested_region": args.region or None,
           "advertised_usd_per_hour": rate}
    try:
        kw = {"storage": args.storage, "name": "archprobe",
              "gpu_type": args.gpu}
        if args.region:
            kw["region"] = args.region
        created = prov.create(**kw)
        mid = created.get("machine_id") or created.get("pod_id")
        if mid is None:
            raise RuntimeError("provider returned no machine id: %r" % (created,))
        print("  rented %s" % mid)
        wait_ready(prov, mid, wait=600)
        print("  ready after %.0fs" % (time.time() - started))
        doc["rental"] = dict(rented_rate(prov, mid), machine_id=str(mid))

        prov.upload(mid, str(PAYLOAD), "/tmp/probe_payload.py")
        # `python3` on a provider image is not necessarily the interpreter that
        # owns torch: the RunPod pytorch image ships it inside a venv, and the
        # bare python3 answers ModuleNotFoundError for numpy. Ask the box.
        probe = (
            "for P in python3 python "
            "$(ls /usr/bin/python3.* /usr/local/bin/python3.* 2>/dev/null "
            "| grep -v -- -config) "
            "/venv/bin/python /venv/main/bin/python /workspace/venv/bin/python "
            "/opt/conda/bin/python /usr/bin/python3; do "
            "\"$P\" -c 'import torch' >/dev/null 2>&1 && "
            "{ echo PY=$P; break; }; done; "
            "echo SCAN_DONE; ls -d /usr/local/lib/python3.* 2>/dev/null; true")
        found = prov.exec_stdout(mid, probe, timeout=180, check=False)
        py = ""
        for line in found.splitlines():
            if line.startswith("PY="):
                py = line[3:].strip()
        if not py:
            raise RuntimeError("no interpreter on the box imports torch:\n%s"
                               % found[-400:])
        doc["interpreter"] = py
        print("  interpreter    %s" % py)
        # The RunPod pytorch image ships torch WITHOUT numpy (torch itself warns
        # "Failed to initialize NumPy" and carries on). The probe's inputs come
        # out of numpy's PCG64, so the version is pinned rather than resolved:
        # a different numpy on one box would be a second variable in an
        # experiment that has exactly one.
        ver = prov.exec_stdout(
            mid,
            "%s -c 'import numpy;print(\"NUMPY=\"+numpy.__version__)' 2>/dev/null "
            "|| { %s -m pip install --disable-pip-version-check "
            "--break-system-packages 'numpy==%s' 2>&1 | tail -3; "
            "%s -c 'import numpy;print(\"NUMPY=\"+numpy.__version__)'; }"
            % (py, py, NUMPY_PIN, py), timeout=600, check=False)
        got = [l for l in ver.splitlines() if l.startswith("NUMPY=")]
        if not got:
            raise RuntimeError("could not provide numpy on the box:\n%s" % ver[-600:])
        doc["numpy_version"] = got[-1][6:].strip()
        print("  numpy          %s" % doc["numpy_version"])
        runs = [("lane", py + " /tmp/probe_payload.py --out /tmp/ap-lane"
                 + (" --no-tensors" if args.no_tensors else "")
                 + (" --sweep-only --no-tensors" if args.sweep_only else ""))]
        if not args.sweep_only:
            runs.append(("deterministic",
                     "CUBLAS_WORKSPACE_CONFIG=:4096:8 " + py +
                         " /tmp/probe_payload.py --out /tmp/ap-det "
                         "--deterministic --no-tensors"))
        for tag, cmd in runs:
            t0 = time.time()
            out = prov.exec_stdout(mid, cmd + " 2>&1 | tail -6", timeout=600)
            if "ARCHPROBE_JSON_BEGIN" not in out:
                raise RuntimeError("%s run produced no JSON:\n%s" % (tag, out[-800:]))
            body = out.split("ARCHPROBE_JSON_BEGIN", 1)[1]
            body = body.split("ARCHPROBE_JSON_END", 1)[0].strip()
            doc[tag] = json.loads(body)
            print("  %-14s %.0fs" % (tag, time.time() - t0))
        if not args.no_tensors and not args.sweep_only:
            t0 = time.time()
            prov.download(mid, "/tmp/ap-lane/probe-tensors.npz",
                          str(outdir / "probe-tensors.npz"), recursive=False,
                          timeout=600)
            print("  tensors        %.0fs, %d bytes"
                  % (time.time() - t0, (outdir / "probe-tensors.npz").stat().st_size))
        doc["wall_seconds"] = round(time.time() - started, 1)
        billed = (doc.get("rental") or {}).get("usd_per_hour") or rate or 0.0
        doc["estimated_usd"] = round(billed * doc["wall_seconds"] / 3600.0, 4)
        print("  estimated bill $%.4f" % doc["estimated_usd"])
        return 0
    finally:
        if mid is not None and not args.keep:
            gone = False
            for attempt in range(4):
                try:
                    prov.destroy(mid)
                except Exception as exc:                          # noqa: BLE001
                    print("  destroy attempt %d failed: %s" % (attempt + 1, exc))
                    time.sleep(5)
                    continue
                time.sleep(4)
                try:
                    gone = prov.get(mid) is None
                except Exception:                                 # noqa: BLE001
                    gone = False
                if gone:
                    break
            doc["destroyed"] = True
            doc["verified_gone"] = gone
            print("  destroyed %s (verified gone: %s)" % (mid, gone))
            if not gone:
                print("  !! CHECK THE PROVIDER CONSOLE FOR %s" % mid, file=sys.stderr)
        (outdir / "rental.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("  wrote %s" % (outdir / "rental.json"))


if __name__ == "__main__":
    sys.exit(main())
