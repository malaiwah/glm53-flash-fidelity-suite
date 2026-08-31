#!/usr/bin/env python3
"""Turn a directory of fidelity-bench receipts into the provider comparison table.

The table in `docs/CLOUD-COMPARISON.md` is GENERATED, for the same reason the
registry's tables are: a number typed into prose cannot be checked, and this
one is the number that decides where someone spends their money.

    python3 bin/provider_bench_table.py reports/provider-bench
    python3 bin/provider_bench_table.py reports/provider-bench --per-rental

Each input is one `fidelity-bench --json` receipt. Everything is derived from
it -- including the hourly rate, which the receipt reads back off the instance
that was actually billing rather than off the catalogue that advertised it. A
receipt with no rate prints its timing and an empty cost cell; it is never
given a guessed price.

**Rentals of the same card are aggregated, not averaged into one number.** The
whole finding this table exists to carry is that two rentals of one SKU at one
price are not the same machine: three RunPod "secure" H100 80GB HBM3 rentals at
a flat $3.29/h measured 0.451, 0.513 and 1.139 ms per matrix. A mean would hide
that; best/median/worst is what you are actually buying, and `n` says how much
to trust it. Rule 2 of `llms.txt` applies to machines as well as to windows.

Stdlib only, python3.9, no installs -- like everything else under `bin/`.
"""
import argparse
import json
import os
import sys

# GLM-5.3-Flash: 42 MoE layers x 288 experts x 3 projections, the same default
# fidelity-bench estimates against, so the two agree by construction.
MATRICES = 42 * 288 * 3


def load(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["_file"] = os.path.basename(path)
    return doc


def per_rental(docs):
    """One row per receipt: what that ONE machine did, and what it cost."""
    out = []
    for d in docs:
        ms = d.get("stream_matrix_ms")
        if not ms:
            continue
        rent = d.get("rental") or {}
        rate = rent.get("usd_per_hour")
        win = ms * MATRICES / 1000.0 / 60.0
        out.append({
            "file": d["_file"],
            "provider": d.get("provider") or "?",
            "gpu": d.get("gpu") or "?",
            "link": (d.get("pcie_load") or {}).get("text") or "?",
            "h2d_GBps": d.get("h2d_GBps"),
            "expert_gemm_TFLOPs": d.get("expert_gemm_TFLOPs"),
            "stream_matrix_ms": ms,
            "min_per_window": round(win, 3),
            "usd_per_hour": rate,
            "usd_per_window": round(win / 60.0 * float(rate), 6) if rate else None,
            "loop_seconds": d.get("wall_seconds"),
        })
    out.sort(key=lambda r: (r["provider"], r["gpu"], r["stream_matrix_ms"]))
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def aggregate(rentals):
    """Group by (provider, card). Report the spread, never a mean.

    Grouping deliberately does NOT include the rate. On a marketplace every
    rental of one card is a different seller at a different price, so keying on
    the rate shatters exactly the rows whose spread the reader most needs to
    see: four Vast H100 SXM rentals would become four groups of one. The rate
    column becomes a range where the rentals disagreed, and $/window is derived
    per rental before it is aggregated, so each cell keeps its own price.
    """
    groups = {}
    for r in rentals:
        key = (r["provider"], r["gpu"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (prov, gpu), rs in groups.items():
        ms = [r["stream_matrix_ms"] for r in rs]
        wins = [r["min_per_window"] for r in rs]
        rates = [r["usd_per_hour"] for r in rs if r["usd_per_hour"]]
        usd = sorted(r["usd_per_window"] for r in rs
                     if r["usd_per_window"] is not None)
        rate = min(rates) if rates else None
        rows.append({
            "provider": prov, "gpu": gpu, "usd_per_hour": rate,
            "usd_per_hour_max": max(rates) if rates else None,
            "n": len(rs),
            "link": rs[0]["link"],
            "h2d_best": max(r["h2d_GBps"] for r in rs if r["h2d_GBps"]) if any(
                r["h2d_GBps"] for r in rs) else None,
            "ms_best": min(ms), "ms_median": _median(ms), "ms_worst": max(ms),
            "win_best": min(wins), "win_median": _median(wins),
            "win_worst": max(wins),
            "usd_win_best": min(usd) if usd else None,
            "usd_win_median": round(_median(usd), 6) if usd else None,
            "usd_win_worst": max(usd) if usd else None,
            "spread_x": round(max(ms) / min(ms), 2),
        })
    rows.sort(key=lambda r: (r["usd_win_median"] is None,
                             r["usd_win_median"] if r["usd_win_median"]
                             is not None else r["ms_median"]))
    return rows


def _rate_cell(r):
    lo, hi = r["usd_per_hour"], r["usd_per_hour_max"]
    if lo is None:
        return "n/r"
    return "%.3f" % lo if (hi is None or abs(hi - lo) < 5e-4) \
        else "%.3f-%.3f" % (lo, hi)


def render(rows):
    best = min([r["usd_win_median"] for r in rows if r["usd_win_median"]] or [0])
    lines = [
        "| provider | card, as `nvidia-smi` reports it | $/h | n | h2d GB/s | "
        "min/window best-median-worst | $/window (median) | vs best | host spread |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rel = ("%.1fx" % (r["usd_win_median"] / best)
               if (best and r["usd_win_median"]) else "-")
        lines.append("| %s | %s | %s | %d | %s | %.2f - %.2f - %.2f | %s | %s | %.2fx |"
                     % (r["provider"], r["gpu"],
                        _rate_cell(r), r["n"],
                        "%.0f" % r["h2d_best"] if r["h2d_best"] else "-",
                        r["win_best"], r["win_median"], r["win_worst"],
                        "%.5f" % r["usd_win_median"] if r["usd_win_median"] else "-",
                        rel, r["spread_x"]))
    return "\n".join(lines)


def render_rentals(rentals):
    lines = ["| receipt | provider | card | $/h | PCIe link, loaded | h2d GB/s | "
             "expert GEMM TF | ms/matrix | min/window | $/window | rent->destroy s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rentals:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %.3f | %.2f | %s | %s |"
                     % (r["file"], r["provider"], r["gpu"],
                        "%.4f" % r["usd_per_hour"] if r["usd_per_hour"] else "n/r",
                        r["link"],
                        "%.1f" % r["h2d_GBps"] if r["h2d_GBps"] else "-",
                        "%.0f" % r["expert_gemm_TFLOPs"]
                        if r["expert_gemm_TFLOPs"] else "-",
                        r["stream_matrix_ms"], r["min_per_window"],
                        "%.5f" % r["usd_per_window"] if r["usd_per_window"] else "-",
                        "%.0f" % r["loop_seconds"] if r["loop_seconds"] else "-"))
    return "\n".join(lines)


BEGIN_AGG = "<!-- BEGIN GENERATED:"
END_AGG = "<!-- END GENERATED -->"
BEGIN_ONE = "<!-- BEGIN GENERATED RENTALS:"
END_ONE = "<!-- END GENERATED RENTALS -->"


def _splice(text, begin_tag, end_tag, table):
    i = text.find(begin_tag)
    if i < 0:
        raise SystemExit("marker %r not found in the document" % begin_tag)
    head_end = text.index("-->", i) + 3
    j = text.find(end_tag, head_end)
    if j < 0:
        raise SystemExit("marker %r not found in the document" % end_tag)
    return text[:head_end] + "\n" + table + "\n" + text[j:]


def inject(path, agg_table, rental_table):
    """Rewrite the tables in a markdown document, in place.

    The alternative is a human pasting a table, which is how five wrong numbers
    reached two published documents before `bin/check_doc_numbers.py` existed.
    Order matters: the per-rental block is spliced first because its markers sit
    below the aggregate's, and splicing the aggregate first would move them.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = _splice(text, BEGIN_ONE, END_ONE, rental_table)
    text = _splice(text, BEGIN_AGG, END_AGG, agg_table)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("injected both tables into %s" % path)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="provider_bench_table",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="directory of fidelity-bench --json receipts")
    p.add_argument("--per-rental", action="store_true",
                   help="one row per receipt instead of one per card")
    p.add_argument("--json", action="store_true",
                   help="emit the derived rows as JSON instead of markdown")
    p.add_argument("--inject", metavar="DOC.md",
                   help="rewrite both tables in place between the BEGIN/END "
                        "GENERATED markers, so the doc cannot drift from the "
                        "receipts")
    a = p.parse_args(argv)

    paths = sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir)
                   if f.endswith(".json"))
    rentals = per_rental([load(x) for x in paths])
    if not rentals:
        sys.stderr.write("no benchmark receipts with a stream_matrix_ms in %s\n"
                         % a.dir)
        return 1
    if a.inject:
        return inject(a.inject, render(aggregate(rentals)), render_rentals(rentals))
    if a.per_rental:
        print(json.dumps(rentals, indent=1) if a.json else render_rentals(rentals))
    else:
        rows = aggregate(rentals)
        print(json.dumps(rows, indent=1) if a.json else render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
