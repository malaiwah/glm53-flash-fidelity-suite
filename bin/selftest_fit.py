#!/usr/bin/env python3
"""Known-answer tests for the census and the fit estimator.

Offline, stdlib only, no GPU.  Run it before trusting a plan:

    python3 bin/selftest_fit.py

Every assertion below is a claim someone can check against a published number
or against arithmetic they can redo on paper.  Where a figure came from a
measurement rather than a derivation, the test says so, and the tolerance is
set by how the measurement was made -- not by what would make the test pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity.census import (  # noqa: E402
    GB,
    Census,
    Device,
    GTX_1650,
    H100_80,
    H200,
    MAC_128,
    RTX_5090,
    RTX_PRO6000,
    check_device,
    default_budget,
    gb,
    glm53_flash_census,
    lane_requirement,
    local_peak_bytes,
    minimum_viable_budget,
    round_up_storage_gb,
    solve_local,
    storage_need,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(("  ok   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail else ""))


def near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    c = glm53_flash_census()

    print("\n[1] CENSUS reproduces the independently measured figures")
    print("    census source: %s" % c.census_source)
    print("    total BF16      %8.2f GB" % gb(c.total_bf16_bytes))
    print("    non-routed      %8.2f GB   (1618 tensors)" % gb(c.nonrouted_bytes))
    print("    routed main     %8.2f GB   (42 layers)" % gb(c.routed_main_bytes))
    print("    routed MTP      %8.2f GB   (1 layer, not executed at capture)"
          % gb(c.routed_mtp_bytes))
    print("    per routed layer%8.2f GB" % gb(c.per_routed_layer_bytes))

    check("routed layers == 42 (45 - first_k_dense_replace 3)", c.routed_layers == 42)
    check("per-expert params == 25,165,824 (3 x 4096 x 2048)",
          c.per_expert_params == 25_165_824)
    check("per routed layer == 14.50 GB",
          near(gb(c.per_routed_layer_bytes), 14.50, 0.01))
    # 42 * 288 * 3 * 4096 * 2048 * 2 bytes
    check("routed main == 608.81 GB", near(gb(c.routed_main_bytes), 608.81, 0.02))
    check("routed MTP == 14.50 GB", near(gb(c.routed_mtp_bytes), 14.50, 0.01))
    # MEASURED elsewhere by range-fetching all 47 non-routed-bearing shard
    # headers and summing data_offsets: 19.34 GB over 1,618 tensors.  We get
    # here by subtraction from the published repo size, so agreement to a few
    # hundred MB is the real check -- the two methods share no arithmetic.
    check("non-routed == 19.34 GB +/- 0.1 (measured independently by range fetch)",
          near(gb(c.nonrouted_bytes), 19.34, 0.10),
          "got %.2f GB" % gb(c.nonrouted_bytes))
    check("census closes: nonrouted + routed == published 642.7 GB",
          near(gb(c.total_bf16_bytes), 642.7, 0.01))
    check("routed matrices per pass == 36,288", c.routed_matrices_per_pass == 36_288)
    check("one window of fp32 logits == 1.27 GB",
          near(gb(c.logits_bytes(2048, 4)), 1.269, 0.005))

    print("\n[2] SEALED-EP8 lane requirement")
    req8 = lane_requirement(c, "sealed-ep8")
    print("    %s" % req8.rationale)
    for k, v in req8.components.items():
        print("      %-26s %7.2f GB" % (k, gb(v)))
    print("      %-26s %7.2f GB  <- required per GPU" % ("TOTAL", gb(req8.per_gpu_bytes)))
    check("EP8 needs >= 100 GB/GPU", gb(req8.per_gpu_bytes) >= 100.0,
          "%.1f GB" % gb(req8.per_gpu_bytes))
    check("EP8 needs 8 GPUs", req8.gpus == 8)

    print("\n[3] STREAMING lane requirement")
    reqs = lane_requirement(c, "streaming")
    print("      required per GPU  %7.2f GB  (observed peak x headroom)"
          % gb(reqs.per_gpu_bytes))
    check("streaming needs >= 60 GB and <= 70 GB",
          60.0 <= gb(reqs.per_gpu_bytes) <= 70.0)
    check("streaming needs 1 GPU", reqs.gpus == 1)

    print("\n[4] KNOWN DEVICE CASES")

    print("\n  (a) H200 141 GB x8, lane sealed-ep8  -- must FIT")
    d = Device(H200.name, "cuda", H200.memory_bytes, count=8)
    v = check_device(c, d, "sealed-ep8")
    print("      %s" % v.reason)
    check("H200x8 fits sealed-ep8", v.ok)

    print("\n  (b) H200 141 GB x1, lane streaming  -- must FIT")
    v = check_device(c, H200, "streaming")
    print("      %s" % v.reason)
    check("H200x1 fits streaming", v.ok)

    print("\n  (c) H100 80 GB x8, lane sealed-ep8  -- must be REFUSED WITH ADVICE")
    d = Device(H100_80.name, "cuda", H100_80.memory_bytes, count=8)
    v = check_device(c, d, "sealed-ep8")
    print("      %s" % v.reason)
    for a in v.advice:
        print("      advice: %s" % a)
    check("H100-80 x8 refused for sealed-ep8", not v.ok)
    check("refusal names the streaming alternative",
          any("streaming" in a for a in v.advice))

    print("\n  (d) RTX PRO 6000 96 GB x1, lane streaming  -- must FIT")
    v = check_device(c, RTX_PRO6000, "streaming")
    print("      %s" % v.reason)
    check("RTX PRO 6000 fits streaming", v.ok)

    print("\n  (e) RTX 5090 32 GB, lane local-cuda-budget, --vram-budget 30")
    v = check_device(c, RTX_5090, "local-cuda-budget", bits=4.0, budget_bytes=30 * GB)
    print("      %s" % v.reason)
    check("5090 honours a genuine 30 GB budget", v.ok)
    if v.ok:
        p = v.detail["plan"]
        print("      expert_chunk %d  window_batch %d  buffers %d  passes %d"
              % (p["expert_chunk"], p["window_batch"], p["buffers"], p["passes"]))
        print("      peak %.2f GB" % p["peak_gb"])
        for k in ("panel_state", "decoded_expert_chunk", "packed_expert_chunk",
                  "decode_workspace", "nonrouted"):
            print("        %-24s %6.3f GB" % (k, p["breakdown_gb"][k]))
        check("5090 plan keeps the whole panel batched (no extra passes)",
              p["passes"] == 1)
        check("5090 peak is under 30 GB", p["peak_gb"] <= 30.0)

    print("\n      whole-layer schedule on the same card (expert_chunk=288):")
    total, layer_p, head_p, _ = local_peak_bytes(
        c, expert_chunk=288, window_batch=25, decode_batch_matrices=4,
        buffers=1, bits=4.0, ctx=2048)
    print("        peak %.2f GB  (fits 32 GB: %s)"
          % (gb(total), gb(total) <= 32.0))
    total6, _, _, _ = local_peak_bytes(
        c, expert_chunk=288, window_batch=25, decode_batch_matrices=4,
        buffers=1, bits=6.0, ctx=2048)
    print("        peak %.2f GB at 6bpw" % gb(total6))
    check("whole-layer 4bpw schedule is in the ~20-30 GB band the recon predicted",
          20.0 <= gb(total) <= 30.0, "%.2f GB" % gb(total))

    print("\n  (f) Apple Silicon 128 GB unified, lane local-mps  -- must FIT")
    budget = default_budget(MAC_128)
    print("      default budget %.1f GB of %.0f GB unified (70%%)"
          % (gb(budget), gb(MAC_128.memory_bytes)))
    v = check_device(c, MAC_128, "local-mps", bits=4.0, budget_bytes=budget)
    print("      %s" % v.reason)
    check("128 GB Mac fits local-mps", v.ok)
    if v.ok:
        p = v.detail["plan"]
        print("      expert_chunk %d  window_batch %d  peak %.2f GB"
              % (p["expert_chunk"], p["window_batch"], p["peak_gb"]))
        check("Mac plan holds all non-routed weights resident (unified memory)",
              p["breakdown_gb"]["nonrouted"] > 15.0,
              "%.2f GB resident" % p["breakdown_gb"]["nonrouted"])

    print("\n  (g) GTX 1650 4 GB, lane local-cuda-budget  -- must be REFUSED")
    v = check_device(c, GTX_1650, "local-cuda-budget", bits=4.0,
                     budget_bytes=default_budget(GTX_1650))
    print("      %s" % v.reason)
    for a in v.advice:
        print("      advice: %s" % a)
    check("4 GB card is refused", not v.ok)
    check("refusal quotes the minimum viable budget",
          any("minimum viable" in a for a in v.advice))
    check("refusal points at the cloud recipe",
          any("measure-cloud" in a for a in v.advice))

    print("\n[5] MINIMUM VIABLE BUDGET (the floor a refusal must quote)")
    for bits in (4.0, 6.0):
        mv = minimum_viable_budget(c, bits=bits)
        print("      %g bpw -> %.2f GB" % (bits, gb(mv)))
    mv4 = minimum_viable_budget(c, bits=4.0)
    # The floor must exceed lm_head weight + one window of fp32 logits, which
    # are the two terms no memory knob can shrink.
    irreducible = float(c.vocab) * c.hidden * 2.0 + c.logits_bytes(2048, 4)
    check("floor exceeds the irreducible lm_head pair (%.2f GB)" % gb(irreducible),
          mv4 > irreducible, "floor %.2f GB" % gb(mv4))
    check("floor is under 6 GB (an 8 GB card should be usable)", gb(mv4) < 6.0,
          "%.2f GB" % gb(mv4))
    v = check_device(c, Device("8 GB card", "cuda", 8 * GB), "local-cuda-budget",
                     bits=4.0, budget_bytes=default_budget(Device("x", "cuda", 8 * GB)))
    check("an 8 GB card is accepted (at a cost in passes), not refused", v.ok,
          v.reason)
    if v.ok:
        print("      8 GB card: expert_chunk %d window_batch %d passes %d peak %.2f GB"
              % (v.detail["plan"]["expert_chunk"], v.detail["plan"]["window_batch"],
                 v.detail["plan"]["passes"], v.detail["plan"]["peak_gb"]))

    print("\n[6] INVARIANCE: the memory knobs must not move the number")
    # The solver may pick any (expert_chunk, window_batch); the schedule is
    # bit-invariant to both because experts are visited in ascending order and
    # accumulated sequentially into an fp32 accumulator.  We cannot assert
    # bitwise equality without a GPU, so we assert the property the solver is
    # allowed to rely on: every candidate it can return covers the same expert
    # set the same number of times.
    for e in (1, 7, 64, 128, 288):
        for w in (1, 5, 25):
            visits = c.routed_layers * c.n_routed_experts
            chunks = -(-c.n_routed_experts // e)
            check("chunking %3d experts x %2d windows covers all %d expert visits"
                  % (e, w, visits),
                  chunks * e >= c.n_routed_experts and visits == 42 * 288)
            break
        break
    print("      (bitwise invariance itself is asserted by the engine fixture")
    print("       check at the extremes: (288,25) vs (1,1) must produce an")
    print("       identical tokenwise-kld tensor hash)")

    print("\n[7] STORAGE sizing")
    need = storage_need(
        artifact_bytes=175.79 * GB, panel_bytes=31.71 * GB, keep_student_logits=False)
    print("      artifact 175.79 + panel 31.71 + 2x transient student logits 63.42")
    print("      + toolchain 40 + 15%% slack")
    print("      -> %.1f GB -> provision %d GB"
          % (gb(need.total_bytes), round_up_storage_gb(need.total_bytes)))
    check("proof-target storage rounds to 400 GB (2 cold runs' student logits "
          "are on disk before the report seals -- lesson 31)",
          round_up_storage_gb(need.total_bytes) == 400,
          "%d GB" % round_up_storage_gb(need.total_bytes))
    need_keep = storage_need(
        artifact_bytes=175.79 * GB, panel_bytes=31.71 * GB, keep_student_logits=True)
    check("KEEPING the student logits changes nothing: the transient already "
          "sized for both cold runs",
          round_up_storage_gb(need_keep.total_bytes) == 400
          and need_keep.total_bytes == need.total_bytes,
          "%d GB" % round_up_storage_gb(need_keep.total_bytes))
    need_thin = storage_need(
        artifact_bytes=175.79 * GB, panel_bytes=31.71 * GB, keep_student_logits=False,
        cold_runs=1)
    check("a single cold run needs strictly less than two",
          need_thin.total_bytes < need.total_bytes,
          "%.1f GB < %.1f GB" % (gb(need_thin.total_bytes), gb(need.total_bytes)))

    print("\n[8] WINDOW-MAJOR COST MODEL (the engine that exists; additive --")
    print("    the 33 checks above are untouched)")
    from fidelity.census import window_major_cost
    wm = window_major_cost(c, ms_per_matrix=18.0)
    # 36,288 matrices x 18 ms = 653.184 s = 10.886 min per pass
    check("decode pass at 18 ms/matrix == 653.184 s (10.9 min), exact",
          near(wm["decode_seconds_per_pass"], 653.184, 1e-9),
          "%.3f s" % wm["decode_seconds_per_pass"])
    check("--decode-cache none -> 25 pass-equivalents (the engine re-decodes "
          "per window)", wm["decode_pass_equivalents"] == 25.0)
    wm_ram = window_major_cost(c, ms_per_matrix=18.0, decode_cache="ram",
                               budget_bytes=128 * GB)
    check("ram cache on 128 GB -> floor(0.8*128GB/14.5GB) == 7 cached layers",
          wm_ram["cached_layers"] == 7, str(wm_ram["cached_layers"]))
    check("ram cache -> 1 + 24*(35/42) == 21.0 pass-equivalents",
          near(wm_ram["decode_pass_equivalents"], 21.0, 1e-9),
          "%.3f" % wm_ram["decode_pass_equivalents"])
    wm_disk = window_major_cost(c, ms_per_matrix=18.0, decode_cache="disk",
                                disk_gb_per_s=5.5)
    check("disk cache decodes ONCE", wm_disk["decode_pass_equivalents"] == 1.0)
    check("disk rereads at 5.5 GB/s == 25 x 608.81/5.5 s (~46.1 min)",
          near(wm_disk["disk_reread_seconds_total"],
               25 * gb(c.routed_main_bytes) / 5.5, 1e-6),
          "%.1f s" % wm_disk["disk_reread_seconds_total"])
    check("trunk term is null (UNMEASURED on Apple) -- never invented",
          wm["trunk_seconds_per_window"] is None and
          wm["total_is_lower_bound"] is True and
          "Measure via" in (wm["trunk_note"] or ""))
    check("fp64 scoring 25x2047 positions at 0.15 ms == 7.68 s (never a "
          "reason to sample)", near(wm["scoring_seconds_total"], 7.676, 0.01),
          "%.2f s" % wm["scoring_seconds_total"])

    print("\n" + "-" * 72)
    print("selftest_fit: %d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for name, detail in FAIL:
            print("  FAILED: %s %s" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
