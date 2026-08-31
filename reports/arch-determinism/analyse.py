#!/usr/bin/env python3
"""analyse -- group the probed GPUs by the arithmetic they produced.

    reports/arch-determinism/analyse.py [--results DIR] [--replicate DIR] [--json OUT]

Reads every `results/<label>/rental.json` written by `drive_probe.py` and
answers the questions the raw digests do not:

 1. WHICH LAYER first disagrees.  Each probe digest is a pure function of
    (fixed input bytes, torch build, GPU), so a key with one value across all
    boxes is architecture-INVARIANT and a key with several is the opposite.

 2. WHAT THE GROUPING IMPLICATES.  Every variant key's partition is tested
    against the two candidate explanations -- compute capability, and SM count
    -- so "groups by architecture", "groups by SM count" and "every card its
    own group" are distinguishable rather than a matter of opinion.

 3. WHETHER FORCING DETERMINISM HELPS.  The second probe run per box had
    `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`
    set before the cuBLAS handle existed.  Two numbers matter: how many keys
    it changed WITHIN a box, and how many still differ ACROSS boxes.

 4. HOW BIG IT IS, in the units the registry publishes.  With one card's fp32
    logits nominated as the reference panel, KLD(reference || each box's bf16
    logits) is computed here, on the CPU, in fp64 -- identical arithmetic for
    every box -- so the spread across cards is the same kind of number as a
    published panel mean.

 5. WHETHER THE FINGERPRINT IS THE CARD OR THE HOST.  `--replicate DIR` points
    at an earlier round of rentals; each label is compared key by key.  A
    second rental of the same GPU model, on a different physical machine in a
    different datacenter, must reproduce the digests exactly or the whole
    experiment is measuring hosts.

Pure stdlib + numpy.  It rents nothing and needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(results: Path):
    boxes = {}
    for rental in sorted(results.glob("*/rental.json")):
        doc = json.loads(rental.read_text())
        if "lane" not in doc:
            continue
        boxes[doc["label"]] = {
            "rental": doc,
            "lane": doc["lane"],
            "det": doc.get("deterministic"),
            "npz": rental.parent / "probe-tensors.npz",
        }
    return boxes


def ident(b):
    d = b["lane"]["device"]
    return "%-46s %-7s %3d SM" % (d["name"], d["capability"],
                                  d["multi_processor_count"])


def partition(boxes, key, mode="lane"):
    groups = {}
    for label, b in boxes.items():
        doc = b[mode]
        if not doc:
            continue
        groups.setdefault(doc["digests"].get(key), []).append(label)
    return groups


def explained_by(groups, attr):
    """Is this partition a function of `attr` alone?

    True iff no two boxes with the same attribute value landed in different
    groups AND no group mixes attribute values -- i.e. the partition IS the
    partition by that attribute.
    """
    owner = {}
    for gid, labels in enumerate(groups.values()):
        for lab in labels:
            owner[lab] = gid
    by_attr = {}
    for lab, gid in owner.items():
        by_attr.setdefault(attr[lab], set()).add(gid)
    if any(len(v) > 1 for v in by_attr.values()):
        return False                      # same attribute, different results
    seen = {}
    for a, gids in by_attr.items():
        gid = next(iter(gids))
        if gid in seen:
            return False                  # same result, different attribute
        seen[gid] = a
    return True


def ulp_f32(a, b):
    """ULP distance between two float32 arrays, monotone-int encoded.

    IEEE-754 float32 bit patterns are already ordered for non-negatives, but
    the negatives run backwards, so the map for i < 0 is INT32_MIN - i -- which
    sends -0.0 (0x80000000, i.e. INT32_MIN) to 0, the same key as +0.0. Getting
    the sign of that constant wrong scores +0.0 against -0.0 as 2^32 ULPs and
    every negative/positive pair as nonsense; selftest cases [13] and [14]
    exist because this implementation did exactly that.
    """
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-0x80000000) - ia, ia)
    ib = np.where(ib < 0, np.int64(-0x80000000) - ib, ib)
    return np.abs(ia - ib)


def kld_fp64(teacher32, student32):
    """The scorer's own estimator, on the CPU, in fp64: KLD(teacher||student)."""
    t = teacher32.astype(np.float64)
    s = student32.astype(np.float64)

    def logsoftmax(x):
        z = x - x.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    tl, sl = logsoftmax(t), logsoftmax(s)
    return (np.exp(tl) * (tl - sl)).sum(axis=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analyse")
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--replicate", help="an earlier results dir to compare against")
    ap.add_argument("--json", help="also write the findings here")
    ap.add_argument("--reference", help="label whose fp32 logits act as the panel")
    ap.add_argument("--sweep", help="the results dir from --sweep-only rentals")
    ap.add_argument("--show-partitions", action="store_true")
    args = ap.parse_args(argv)

    boxes = load(Path(args.results))
    if not boxes:
        print("no completed probes under %s" % args.results, file=sys.stderr)
        return 4
    labels = sorted(boxes)
    cap = {l: boxes[l]["lane"]["device"]["capability"] for l in labels}
    smc = {l: boxes[l]["lane"]["device"]["multi_processor_count"] for l in labels}
    F = {"boxes": {}, "invariant": [], "variant": {}}

    print("=" * 78)
    print("BOXES")
    print("=" * 78)
    stacks = set()
    for lab in labels:
        b = boxes[lab]
        st = b["lane"]["stack"]
        stacks.add((st["torch"], st["torch_cuda"], st["numpy"], str(st["cudnn"])))
        print("  %-24s %s  driver %s"
              % (lab, ident(b), b["lane"]["device"]["driver"]))
        F["boxes"][lab] = dict(b["lane"]["device"], torch=st["torch"],
                               torch_cuda=st["torch_cuda"], numpy=st["numpy"])
    print()
    print("  software stack%s identical across boxes"
          % ("" if len(stacks) == 1 else " NOT"))
    for s in sorted(stacks):
        print("    torch %s / cuda %s / numpy %s / cudnn %s" % s)
    F["stack_identical"] = len(stacks) == 1
    F["stacks"] = sorted(stacks)

    # ---- 1 & 2 -----------------------------------------------------------
    keys = sorted({k for b in boxes.values() for k in b["lane"]["digests"]})
    real = [k for k in keys if not k.endswith(".in") and ".in_" not in k]
    print()
    print("=" * 78)
    print("WHICH LAYERS DIVERGE  (lane policy: TF32 off, matmul precision highest)")
    print("=" * 78)
    print("  %-10s %8s %8s   %s" % ("layer", "keys", "variant", "example"))
    for layer in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"):
        ks = [k for k in real if k.startswith(layer + ".")]
        if not ks:
            continue
        var = [k for k in ks if len(partition(boxes, k)) > 1]
        print("  %-10s %8d %8d   %s"
              % (layer, len(ks), len(var), var[0] if var else "-"))
        F.setdefault("layer_summary", {})[layer] = {"keys": len(ks),
                                                    "variant": len(var)}
    for k in real:
        g = partition(boxes, k)
        if len(g) == 1:
            F["invariant"].append(k)
            continue
        F["variant"][k] = {
            "groups": [sorted(v) for v in g.values()],
            "n_groups": len(g),
            "explained_by_capability": explained_by(g, cap),
            "explained_by_sm_count": explained_by(g, smc),
        }
    print()
    print("  %d of %d result keys are architecture-INVARIANT"
          % (len(F["invariant"]), len(real)))

    print()
    print("=" * 78)
    print("WHAT EXPLAINS THE GROUPING")
    print("=" * 78)
    bycap = sum(1 for v in F["variant"].values() if v["explained_by_capability"])
    bysm = sum(1 for v in F["variant"].values() if v["explained_by_sm_count"])
    allsep = sum(1 for v in F["variant"].values() if v["n_groups"] == len(labels))
    print("  variant keys                                    %d" % len(F["variant"]))
    print("  ... a function of compute capability alone      %d" % bycap)
    print("  ... a function of SM count alone                %d" % bysm)
    print("  ... every card in its own group                 %d" % allsep)
    F["explained_by_capability_count"] = bycap
    F["explained_by_sm_count_count"] = bysm
    F["all_separate_count"] = allsep
    print()
    print("  headline keys:")
    for k in ("L7.fp32.logits", "L7.bf16.logits", "L2.sq2048.bf16",
              "L2.expert.bf16", "L2.expert.fp64", "L3.grouped_mm",
              "L5.log_softmax.fp64", "L6.tokenwise_kld",
              "L4.wide.sum_lastdim.fp64"):
        if k not in {kk for kk in real}:
            continue
        g = partition(boxes, k)
        tag = ("INVARIANT" if len(g) == 1 else
               "%d groups" % len(g))
        print("    %-28s %-11s cap-explained=%-5s sm-explained=%s"
              % (k, tag, explained_by(g, cap), explained_by(g, smc)))
        if len(g) > 1:
            for v in sorted(g.values(), key=lambda x: sorted(x)):
                print("        %s" % "  ".join(
                    "%s(%s,%dSM)" % (x, cap[x], smc[x]) for x in sorted(v)))

    if args.show_partitions:
        print()
        for k in sorted(F["variant"]):
            print("  %-34s %d groups" % (k, F["variant"][k]["n_groups"]))
            for v in F["variant"][k]["groups"]:
                print("      %s" % ",".join(v))

    # ---- 3 ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("DOES FORCING DETERMINISM HELP")
    print("=" * 78)
    det_boxes = [l for l in labels if boxes[l]["det"]]
    if det_boxes:
        changed = {}
        for lab in det_boxes:
            d, l = boxes[lab]["det"]["digests"], boxes[lab]["lane"]["digests"]
            changed[lab] = sorted(k for k in real if k in d and l[k] != d[k])
            print("  %-24s changed %d of %d result keys within the box"
                  % (lab, len(changed[lab]), len(real)))
        still = [k for k in real
                 if len(partition({l: boxes[l] for l in det_boxes}, k, "det")) > 1]
        print("  keys still differing ACROSS boxes under deterministic mode: %d"
              % len(still))
        F["deterministic_changed_within_box"] = {k: len(v)
                                                 for k, v in changed.items()}
        F["deterministic_still_variant"] = len(still)
    else:
        print("  (no deterministic-mode runs collected)")

    # ---- 4 ---------------------------------------------------------------
    have = [l for l in labels if boxes[l]["npz"].is_file()]
    if len(have) >= 2:
        print()
        print("=" * 78)
        print("MAGNITUDE  (toy 16-layer decoder; fp64 CPU estimator, same for all)")
        print("=" * 78)
        data = {l: np.load(boxes[l]["npz"]) for l in have}
        ref = args.reference if args.reference in have else sorted(have)[0]
        print("  reference panel: %s -- its fp32 logits stand in for the teacher."
              % ref)
        print("  (the choice is arbitrary; only the SPREAD is meaningful)")
        teacher = data[ref]["logits_fp32"]
        rows = {l: float(kld_fp64(teacher, data[l]["logits_bf16"]).mean())
                for l in have}
        base = min(rows.values())
        print()
        print("  %-24s %-24s %s" % ("box", "mean KLD (nats)", "delta vs lowest"))
        for l in sorted(rows, key=lambda x: rows[x]):
            print("  %-24s %-24.17g %+.4e" % (l, rows[l], rows[l] - base))
        spread = max(rows.values()) - min(rows.values())
        mean = sum(rows.values()) / len(rows)
        print()
        print("  spread across cards: %.4e nats = %.3g%% of the mean"
              % (spread, 100 * spread / mean))
        F["toy_panel_kld"] = rows
        F["toy_panel_kld_reference"] = ref
        F["toy_panel_kld_spread"] = spread
        F["toy_panel_kld_relative_spread"] = spread / mean

        print()
        print("  bf16 logits vs %s -- bitwise, max-abs, ULP over DIFFERING elements" % ref)
        print("  (max ULP is not quoted: two near-zero values a few bf16 steps")
        print("   apart span many float32 exponents, which inflates it to ~2e9)")
        for l in sorted(have):
            a, b = data[ref]["logits_bf16"], data[l]["logits_bf16"]
            same = bool(np.array_equal(a, b))
            d = ulp_f32(a, b)
            d = d[d > 0]
            print("  %-24s bitwise=%-5s max_abs=%.3e differing=%5.2f%% "
                  "ulp p50=%s p99=%s"
                  % (l, same, float(np.abs(a - b).max()),
                     100.0 * (a != b).mean(),
                     "-" if d.size == 0 else int(np.median(d)),
                     "-" if d.size == 0 else int(np.percentile(d, 99))))

        print()
        print("  hidden-state max-abs difference vs %s, by depth" % ref)
        depths = sorted(k for k in data[ref].files if k.startswith("h_depth"))
        print("  %-24s" % "box" + "".join("%9s" % d.replace("h_depth", "L")
                                          for d in depths))
        growth = {}
        for l in sorted(have):
            growth[l] = [float(np.abs(data[ref][d] - data[l][d]).max())
                         for d in depths]
            print("  %-24s" % l + "".join("%9.1e" % v for v in growth[l]))
        F["depth_growth"] = {"depths": depths, "max_abs": growth}

        print()
        print("  L6 tokenwise KLD -- IDENTICAL synthetic logits fed to every box")
        for l in sorted(have):
            v, r = data[l]["tokenwise_kld_L6"], data[ref]["tokenwise_kld_L6"]
            print("  %-24s mean=%.17g bitwise_vs_ref=%-5s max_abs=%.3e"
                  % (l, float(v.mean()), bool(np.array_equal(v, r)),
                     float(np.abs(v - r).max())))
        F["L6_means"] = {l: float(data[l]["tokenwise_kld_L6"].mean())
                         for l in have}

    # ---- the reduction-split sweep ---------------------------------------
    if args.sweep:
        sw = load(Path(args.sweep))
        if sw:
            print()
            print("=" * 78)
            print("WHERE THE REDUCTION SPLITS  (%d cards)" % len(sw))
            print("=" * 78)
            ssm = {l: sw[l]["lane"]["device"]["multi_processor_count"] for l in sw}
            print("  " + ", ".join("%s(%dSM)" % (l, ssm[l]) for l in sorted(sw)))
            F["sweep"] = {}
            for tag, cols, rows_list in (
                    ("L8", 16384, (16, 32, 64, 96, 128, 192, 256, 384, 512,
                                   768, 1024)),
                    ("L9", 154880, (32, 64, 128, 256, 384, 448, 512, 640,
                                    768, 1024))):
                print()
                print("  %s: fp64 sum over the last dim, %d columns" % (tag, cols))
                for r in rows_list:
                    g = partition(sw, "%s.rows%04d.sum_lastdim.fp64" % (tag, r))
                    print("    rows=%5d  %d group(s)%s"
                          % (r, len(g), "" if len(g) == 1 else "   " + " | ".join(
                              ",".join(sorted(v)) for v in
                              sorted(g.values(), key=lambda x: -len(x)))))
                    F["sweep"]["%s.rows%d" % (tag, r)] = len(g)
            print()
            print("  L10: k6_kld_report._token_kld VERBATIM, %d-wide vocabulary,"
                  % 154880)
            print("       by position block -- the lane runs 512 today")
            for r in (256, 512, 640, 1024):
                g = partition(sw, "L10.chunk%04d.tokenwise_kld" % r)
                print("    chunk=%5d  %d group(s)%s"
                      % (r, len(g), "" if len(g) == 1 else "   " + " | ".join(
                          ",".join(sorted(v)) for v in
                          sorted(g.values(), key=lambda x: -len(x)))))
                F["sweep"]["L10.chunk%d" % r] = len(g)

    # ---- 5 ---------------------------------------------------------------
    if args.replicate:
        print()
        print("=" * 78)
        print("REPLICATION  (a second rental of the same GPU model, other host)")
        print("=" * 78)
        prev = load(Path(args.replicate))
        rep = {}
        for lab in sorted(set(labels) & set(prev)):
            a = boxes[lab]["lane"]["digests"]
            b = prev[lab]["lane"]["digests"]
            common = [k for k in real if k in a and k in b]
            diff = [k for k in common if a[k] != b[k]]
            rep[lab] = {"compared": len(common), "differing": len(diff),
                        "examples": diff[:5]}
            print("  %-24s %4d keys compared, %d differ%s"
                  % (lab, len(common), len(diff),
                     "" if not diff else "  " + ",".join(diff[:3])))
        F["replication"] = rep

    if args.json:
        Path(args.json).write_text(
            json.dumps(F, indent=1, sort_keys=True, default=str) + "\n")
        print("\n  wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
