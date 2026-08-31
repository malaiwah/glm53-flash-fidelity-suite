#!/usr/bin/env python3
"""T11 -- the money chokepoint: `JL.list_instances` must never invent an empty account.

Four call sites in `measure_cloud.py` read an empty instance list as a FACT, and each
one spends or leaks money on it:

  * `reaper_sweep` retires the lease of any machine that is "gone" -- the last-resort
    backstop then never looks at that box again;
  * the adopt loop reads "no instance for this job" as licence to CREATE one, the
    double-spend its own comment says the loop exists to prevent;
  * `_find_by_name` is last-resort id recovery for an instance that is ALREADY billing;
  * the name-deadline sweep silently degrades to leases only.

`JL._call` answers `{}` for an empty body on a zero exit, and returns a parsed object
unchanged on a NON-zero exit as long as the JSON carries no `error` key. The old
`data.get("instances", [])` fallback collapsed all of those into "the account is empty".

Everything here runs against a stub `jl` on PATH. No network, no account, no rental.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fidelity.jlapi import JL, JLError            # noqa: E402

STUB = """#!/bin/sh
case "$1" in
  --version) echo "jl, version 0.2.17"; exit 0;;
  list) printf '%s' "$JL_LIST_OUT"; exit ${JL_LIST_RC:-0};;
  gpus) printf '%s' "$JL_GPUS_OUT"; exit 0;;
  *) echo '{}'; exit 0;;
esac
"""

# Verbatim rows from `jl 0.2.17 gpus --json` on a live account, 2026-08-31.
# The first is spot+on-demand, the second is on-demand ONLY (spot_price null),
# which is how the whole EU1 region is shaped.
REAL_GPUS = (
    '[{"gpu_type":"H200","region":"IN2","num_free_devices":6,'
    '"effective_num_free_devices":8,"price_per_hour":3.99,"spot_price":1.99,'
    '"vram":"141","cpus_per_gpu":28,"ram_per_gpu":300,'
    '"workload_type":"container"},'
    '{"gpu_type":"H200","region":"EU1","num_free_devices":1,'
    '"effective_num_free_devices":27,"price_per_hour":3.99,"spot_price":null,'
    '"vram":"141","cpus_per_gpu":16,"ram_per_gpu":200,"workload_type":null}]'
)

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  PASS  %s" % name)
    else:
        FAIL += 1
        print("  FAIL  %s  -- %s" % (name, detail))


def listing(stub_dir, out, rc=0):
    """Run JL.list_instances against the stub. Returns (rows, error_or_None)."""
    os.environ["JL_LIST_OUT"] = out
    os.environ["JL_LIST_RC"] = str(rc)
    os.environ["PATH"] = stub_dir + os.pathsep + os.environ["PATH"]
    jl = JL()
    try:
        return jl.list_instances(), None
    except JLError as exc:
        return None, exc


def main():
    tmp = tempfile.mkdtemp()
    stub = os.path.join(tmp, "jl")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write(STUB)
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print("== T11 jl list: an unreadable answer is not an empty account ==")

    rows, err = listing(tmp, '[{"machine_id":999001,"status":"Running","name":"fidcloud-a1b2c3d4-x1abcde"}]')
    check("J1 the real shape (a top-level JSON array) is read as instances",
          err is None and rows is not None and len(rows) == 1
          and rows[0].machine_id == 999001, "%s %s" % (err, rows))

    rows, err = listing(tmp, "[]")
    check("J2 a genuinely empty account is still an empty account, not an error",
          err is None and rows == [], "%s %s" % (err, rows))

    rows, err = listing(tmp, '{"instances":[{"machine_id":7}]}')
    check("J3 the documented {\"instances\": [...]} envelope still works",
          err is None and rows is not None and len(rows) == 1, "%s %s" % (err, rows))

    rows, err = listing(tmp, "")
    check("J4 an EMPTY BODY on a zero exit raises rather than reporting no instances",
          err is not None and "empty body" in str(err), "%s %s" % (err, rows))

    rows, err = listing(tmp, '{"data":[{"machine_id":999001}]}')
    check("J5 an unrecognised envelope (a vendor key rename) raises",
          err is not None and "'data'" in str(err), "%s %s" % (err, rows))

    rows, err = listing(tmp, '{"detail":"authentication failed"}', rc=2)
    check("J6 a non-zero exit whose JSON carries no `error` key raises "
          "(_call passes that body straight through)",
          err is not None and "'detail'" in str(err), "%s %s" % (err, rows))

    rows, err = listing(tmp, '["999001","483634"]')
    check("J7 rows that are not objects raise -- liveness cannot be read from them",
          err is not None and "not objects" in str(err), "%s %s" % (err, rows))

    print("\n== T11b jl gpus: an on-demand row must not vanish ==")
    # `gpus()` looked for `price` / `on_demand_price`; the CLI writes
    # `price_per_hour`. Every JarvisLabs on-demand offer was therefore missing
    # from the catalogue, `select_offer(..., spot=False)` had nothing to pick,
    # and `--on-demand` refused "no available instance fits this lane" on the
    # reference provider. A region with no spot price at all was invisible in
    # both modes.
    from fidelity.jlapi import select_offer                # noqa: E402
    os.environ["JL_GPUS_OUT"] = REAL_GPUS
    os.environ["PATH"] = tmp + os.pathsep + os.environ["PATH"]
    offers = JL().gpus()
    spot = [o for o in offers if o.spot]
    od = [o for o in offers if not o.spot]
    check("J8 the spot rate is read (it always was)",
          len(spot) == 1 and spot[0].price == 1.99, [(o.price, o.spot) for o in offers])
    check("J9 the on-demand rate is read from `price_per_hour`",
          len(od) == 2 and all(o.price == 3.99 for o in od),
          [(o.gpu_type, o.region, o.price, o.spot) for o in offers])
    check("J10 an on-demand-ONLY row still produces an offer",
          any(not o.spot and o.region == "EU1" for o in offers),
          [(o.region, o.spot) for o in offers])
    picked, _ = select_offer(offers, required_vram_bytes=63e9, gpus=1, spot=False)
    check("J11 --on-demand can therefore be planned at all",
          picked is not None and picked.price == 3.99, picked)

    print("\nselftest_jlapi: %d passed, %d failed" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
