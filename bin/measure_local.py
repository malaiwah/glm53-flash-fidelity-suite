#!/usr/bin/env python3
"""measure-local -- measure a quant's fidelity on hardware you already own.

    bin/measure-local --artifact <hf-repo> --panel <hf-dataset> --vram-budget 30

Targets, both first-class:
  * a 128 GB Apple-Silicon Mac via MPS;
  * a 32 GB consumer CUDA card (RTX 5090 and friends) under a HARD VRAM budget.

It produces the SAME sealed receipt schema as the cloud recipe, so a number
measured on a desk and a number measured on a rented H200 are the same kind of
object and the registry can rank them against each other.

THE SCHEDULE.  The obvious way to stream a 600 GB model through a small card is
per-window: for each of the 25 panel windows, stream all 42 routed layers.  That
re-reads the entire checkpoint 25 times and is why the cloud streaming lane
costs ~9 minutes per window.  The windows are independent teacher-forced
prefills with no state carried between them, so the loop inverts: for each
layer, decode once, then push all 25 windows through it.  Decode and weight I/O
then happen EXACTLY ONCE for the whole panel.  The price is holding the panel's
inter-layer state resident, which is 2.94 GB.

Two knobs shrink the peak further and NEITHER moves the number: experts are
visited in strictly ascending order and accumulated sequentially into an fp32
accumulator, so the result is bit-identical for any `--expert-chunk` and
`--window-batch`.  That invariance is a promise the engine must keep -- an
atomicAdd-based scatter would break it -- and it is what lets you tune memory
without tuning your result.

WHAT IT TELLS YOU BEFORE IT STARTS.  Disk, RAM, VRAM plan and hours, with each
number's provenance, and a refusal-with-advice if your device cannot do it.
The hours estimate comes from a ~5 second micro-benchmark run on YOUR machine,
not from a hardcoded per-GPU table.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SUITE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fidelity import census as C                       # noqa: E402
from fidelity.common import (                          # noqa: E402
    Console, human_bytes, human_duration, read_json, write_json,
)
from fidelity.engines import EngineUnpinned, load_engines  # noqa: E402
from fidelity.hfmeta import (                          # noqa: E402
    HFError, hf_token, load_panel_descriptor, repo_meta, sniff_surface,
)

VERSION = "0.1.0"
GB = C.GB
EXIT_OK, EXIT_REFUSED = 0, 3

# Matrices decoded in one full routed pass: 42 layers x 288 experts x 3.
MATRICES_PER_PASS = 36288


# ==========================================================================
# Device discovery
# ==========================================================================


def parse_simulated(spec: str) -> C.Device:
    """`--simulate-device NAME:VRAM_GB[:count][:unified]`.

    Planning for hardware you do not have in front of you is a real need --
    "will this fit on a 5090 before I buy one", "will it fit on the box in the
    other office" -- and it is also the only way to TEST the fit estimator
    against devices nobody here owns. It plans and refuses exactly as the real
    thing would; it just cannot micro-benchmark, and says so.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError("--simulate-device wants NAME:VRAM_GB[:count][:unified]")
    name, vram = parts[0], float(parts[1])
    count = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    unified = len(parts) > 3 and parts[3].lower() in ("unified", "1", "true")
    return C.Device(name, "mps" if unified else "cuda", vram * GB, count=count,
                    unified=unified, host_ram_bytes=(vram * GB if unified else None),
                    note="SIMULATED")


def detect_device(requested: str, con: Console) -> C.Device:
    """Find the accelerator, WITHOUT importing torch unless we have to."""
    system = platform.system()
    machine = platform.machine()

    if requested in ("auto", "mps") and system == "Darwin" and machine == "arm64":
        total = _mac_memory_bytes()
        return C.Device("Apple Silicon (%s)" % _mac_chip(), "mps", total,
                        unified=True, host_ram_bytes=total)
    if requested in ("auto", "cuda"):
        info = _cuda_info()
        if info:
            name, mem, count = info
            return C.Device(name, "cuda", mem, count=count,
                            host_ram_bytes=_host_ram_bytes())
        if requested == "cuda":
            raise RuntimeError("--device cuda requested but no CUDA device was found")
    if requested in ("auto", "cpu"):
        ram = _host_ram_bytes()
        return C.Device("CPU only", "cpu", ram, unified=True, host_ram_bytes=ram,
                        note="no accelerator found")
    raise RuntimeError("cannot satisfy --device %s" % requested)


def _mac_chip() -> str:
    try:
        from fidelity.common import run
        return run(["sysctl", "-n", "machdep.cpu.brand_string"],
                   check=False).stdout.strip() or "arm64"
    except Exception:                                   # noqa: BLE001
        return "arm64"


def _mac_memory_bytes() -> float:
    try:
        from fidelity.common import run
        return float(run(["sysctl", "-n", "hw.memsize"], check=False).stdout.strip())
    except Exception:                                   # noqa: BLE001
        return 16 * GB


def _host_ram_bytes() -> float:
    if hasattr(os, "sysconf"):
        try:
            return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError):
            pass
    return _mac_memory_bytes()


def _cuda_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        from fidelity.common import run
        out = run([exe, "--query-gpu=name,memory.total",
                   "--format=csv,noheader,nounits"], check=False, timeout=30).stdout
    except Exception:                                   # noqa: BLE001
        return None
    rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
    if not rows:
        return None
    name, mib = rows[0].split(",")
    return name.strip(), float(mib) * 1024 * 1024, len(rows)


# ==========================================================================
# Micro-benchmark: the honesty mechanism
# ==========================================================================


def microbench(device: C.Device, bits: float, con: Console,
               reps: int = 3) -> Optional[Dict[str, Any]]:
    """Time the real decode on THIS machine, in about five seconds.

    A hardcoded per-GPU performance table is a lie on any machine that is not
    in it, and this recipe is aimed at machines nobody has tested.  So we run
    the actual decode kernel once here and project from that.  If torch or the
    reader source is unavailable we say the estimate is unavailable rather than
    inventing one.
    """
    selftest = SUITE_ROOT / "bin" / "selftest_decode_parity.py"
    if not selftest.is_file():
        return None
    try:
        import torch                                    # noqa: F401
    except ImportError:
        con.warn("torch is not installed, so the timing estimate is unavailable. "
                 "Install it (`pip install torch`) for a projection measured on "
                 "this machine rather than a guess.")
        return None
    sys.path.insert(0, str(SUITE_ROOT / "bin"))
    try:
        from selftest_decode_parity import (            # noqa: WPS433
            READER_CANDIDATES, load_reader_functions, make_payload,
        )
        import torch
        reader = next((p for p in READER_CANDIDATES if p.is_file()), None)
        if reader is None:
            return None
        ns = load_reader_functions(reader)
        decode = ns["decode_choice_hf"]
        dev = {"mps": "mps", "cuda": "cuda", "cpu": "cpu"}[device.kind]
        b = int(bits) if int(bits) in (4, 6) else 4
        trellis, suh, svh = make_payload(torch, bits=b, k_tiles=256, n_tiles=128)
        t_, s_, v_ = trellis.to(dev), suh.to(dev), svh.to(dev)
        decode(t_, s_, v_, bits=b)
        _sync(torch, dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            decode(t_, s_, v_, bits=b)
        _sync(torch, dev)
        per = (time.perf_counter() - t0) / reps
        return {
            "device": dev, "bits": b,
            "seconds_per_matrix": per,
            "matrices_per_pass": MATRICES_PER_PASS,
            "routed_pass_seconds": per * MATRICES_PER_PASS,
            "provenance": "measured on this machine just now, %d reps of the real "
                          "decode at the artifact's own bit width" % reps,
        }
    except Exception as exc:                            # noqa: BLE001
        con.warn("micro-benchmark failed (%s); timing estimate unavailable"
                 % type(exc).__name__)
        return None


def _sync(torch, dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


# ==========================================================================
# Planning
# ==========================================================================


class Refusal(RuntimeError):
    def __init__(self, reason: str, advice: List[str]) -> None:
        self.reason, self.advice = reason, advice
        super().__init__(reason)


def plan(args: argparse.Namespace, con: Console) -> Dict[str, Any]:
    out: Dict[str, Any] = {"would_refuse": []}

    def problem(reason: str, advice: List[str]) -> None:
        r = Refusal(reason, advice)
        if not (args.dry_run or args.estimate_only):
            raise r
        con.warn("WOULD REFUSE (real run): %s" % reason)
        for line in advice[:4]:
            con.say("           %s" % line)
        out["would_refuse"].append(reason)

    # -- device ------------------------------------------------------------
    simulated = bool(args.simulate_device)
    device = (parse_simulated(args.simulate_device) if simulated
              else detect_device(args.device, con))
    con.say("MACHINE" + ("   (SIMULATED -- planning only, nothing measured on it)"
                         if simulated else ""))
    con.kv("device", "%s (%s)" % (device.name, device.kind))
    con.kv("memory", "%s%s" % (human_bytes(device.memory_bytes),
                               " unified" if device.unified else " VRAM"))
    if device.host_ram_bytes and not device.unified:
        con.kv("host RAM", human_bytes(device.host_ram_bytes))
    lane = args.lane or ("local-mps" if device.kind == "mps" else "local-cuda-budget")
    con.kv("lane", lane)
    out["device"] = {"name": device.name, "kind": device.kind,
                     "memory_bytes": device.memory_bytes, "unified": device.unified}
    out["lane"] = lane

    # -- target ------------------------------------------------------------
    con.say("")
    con.say("TARGET")
    artifact_bytes, bits, offline = 176.0 * GB, 4.0, False
    try:
        meta = repo_meta(args.artifact, "model", args.revision or "main")
        surface = sniff_surface(meta)
        artifact_bytes = float(meta.total_bytes)
        bits = float(surface.bits or 4.0)
        con.kv("artifact", meta.repo_id)
        con.kv("revision", "%s  (from %s)" % (meta.revision, meta.requested_revision))
        con.kv("size", "%s over %d files" % (human_bytes(artifact_bytes), len(meta.files)))
        con.kv("surface", "%s  codec %s @ %g bpw"
               % (surface.surface, surface.codec_family or "?", bits))
        out["artifact"] = {"repo_id": meta.repo_id, "revision": meta.revision,
                           "size_bytes": artifact_bytes, "surface": surface.surface,
                           "bits": bits}
        if surface.problems:
            problem("this artifact cannot be read by any available surface adapter",
                    surface.problems)
    except HFError as exc:
        offline = True
        con.warn("cannot reach Hugging Face (%s); estimating with pinned sizes" % exc)
        out["artifact"] = {"repo_id": args.artifact, "offline": True}

    # -- panel -------------------------------------------------------------
    con.say("")
    con.say("PANEL")
    descriptor = load_panel_descriptor(args.panel_descriptor or args.panel)
    panel_bytes = 31.71 * GB
    if not offline:
        try:
            pmeta = repo_meta(descriptor.repo_id, "dataset",
                              args.panel_revision or descriptor.revision)
            panel_bytes = float(pmeta.bytes_matching(descriptor.include))
            con.kv("panel", "%s @ %s" % (pmeta.repo_id, pmeta.revision[:12]))
            con.kv("fetches", "%s of %s" % (human_bytes(panel_bytes),
                                            human_bytes(pmeta.total_bytes)))
        except HFError as exc:
            con.warn("panel metadata unavailable (%s)" % exc)
    con.kv("shape", "%d contexts x %d positions = %d scored"
           % (descriptor.contexts, descriptor.positions_per_context,
              descriptor.scored_positions))
    out["panel"] = dict(descriptor.to_dict(), fetch_bytes=panel_bytes)

    # -- memory plan -------------------------------------------------------
    con.say("")
    con.say("MEMORY PLAN")
    cen = C.glm53_flash_census()
    budget = (args.vram_budget * GB) if args.vram_budget else C.default_budget(device)
    con.kv("budget", "%s%s" % (human_bytes(budget),
                               "  (--vram-budget)" if args.vram_budget
                               else "  (default: %d%% of device memory)"
                                    % (70 if device.unified else 90)))
    memplan = C.solve_local(
        cen, device, budget_bytes=budget, bits=bits,
        ctx=descriptor.positions_per_context + 1, windows=descriptor.contexts,
        decode_batch_matrices=args.decode_batch_matrices,
        buffers=args.prefetch_depth,
        nonrouted_resident=(None if args.nonrouted_residency == "auto"
                            else args.nonrouted_residency == "gpu"))
    if memplan is None:
        mv = C.minimum_viable_budget(cen, bits=bits)
        problem("no schedule fits a %s budget" % human_bytes(budget), [
            "minimum viable budget for this model at %g bpw is %s"
            % (bits, human_bytes(mv)),
            "that floor is set by the lm_head step -- the lm_head weight (%s) "
            "and one window of fp32 logits (%s) must be resident together, and "
            "neither shrinks with --expert-chunk or --window-batch"
            % (human_bytes(float(cen.vocab) * cen.hidden * 2.0),
               human_bytes(cen.logits_bytes(descriptor.positions_per_context + 1, 4))),
            "run the cloud recipe instead:  bin/measure-cloud --lane streaming",
        ])
    else:
        con.kv("expert_chunk", "%d of %d experts%s"
               % (memplan.expert_chunk, cen.n_routed_experts,
                  "  (numerics-invariant)" if memplan.expert_chunk else ""))
        con.kv("window_batch", "%d of %d windows -> %d pass(es) over the checkpoint"
               % (memplan.window_batch, descriptor.contexts, memplan.passes))
        con.kv("peak VRAM", "%s of %s budget (%.0f%%)"
               % (human_bytes(memplan.peak_bytes), human_bytes(budget),
                  100.0 * memplan.peak_bytes / budget))
        for key in ("panel_state", "decoded_expert_chunk", "packed_expert_chunk",
                    "decode_workspace", "nonrouted", "lm_head_weight",
                    "lm_head_logits_fp32"):
            v = memplan.breakdown.get(key, 0.0)
            if v:
                con.say("      %-24s %s" % (key, human_bytes(v)))
        out["memory_plan"] = memplan.to_dict()

    # -- disk and RAM ------------------------------------------------------
    con.say("")
    con.say("WHAT THIS NEEDS FROM YOUR MACHINE")
    need = C.storage_need(artifact_bytes=artifact_bytes, panel_bytes=panel_bytes,
                          keep_student_logits=args.keep_student_logits,
                          toolchain_bytes=5 * GB)
    workdir = Path(args.work or "./fidelity-local").resolve()
    workdir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir.parent if not workdir.exists() else workdir).free
    con.kv("disk needed", "%s  (artifact %s + panel %s%s + slack)"
           % (human_bytes(need.total_bytes), human_bytes(artifact_bytes),
              human_bytes(panel_bytes),
              " + student logits %s" % human_bytes(need.student_logits_bytes)
              if args.keep_student_logits else ""))
    con.kv("disk free", "%s at %s" % (human_bytes(free), workdir))
    if free < need.total_bytes:
        problem("not enough disk: short %s" % human_bytes(need.total_bytes - free), [
            "free %s at %s, or pass --work with a bigger volume"
            % (human_bytes(need.total_bytes - free), workdir),
            "" if args.keep_student_logits else
            "student logits are already being discarded after scoring (saves %s)"
            % human_bytes(panel_bytes),
        ])

    # The full 643 GB BF16 checkpoint is NOT needed: both the TR3 and Dione
    # surfaces ship non-routed tensors natively in-repo. Say so, because
    # assuming otherwise turns a 208 GB recipe into an 850 GB one.
    con.kv("BF16 base", "NOT required -- non-routed tensors ship in the artifact "
                        "(pass --bf16 only for full inventory verification)")

    ram_need = cen.nonrouted_bytes + 4 * GB + 6 * GB
    if device.unified:
        con.kv("RAM", "unified with VRAM; the budget above covers it")
    else:
        con.kv("RAM needed", "%s  (non-routed %s CPU-resident + staging + runtime)"
               % (human_bytes(ram_need), human_bytes(cen.nonrouted_bytes)))
        host = device.host_ram_bytes or 0
        con.kv("RAM present",
               human_bytes(host) if host else
               "unknown (simulated device -- check this yourself before running)")
        if host and host < ram_need:
            problem("not enough host RAM: short %s" % human_bytes(ram_need - host), [
                "the non-routed weights (%s) are held CPU-resident and streamed "
                "per layer" % human_bytes(cen.nonrouted_bytes),
                "use --nonrouted-residency mmap to trade RAM for disk reads",
            ])
    out["storage_need"] = need.to_dict()
    out["disk_free_bytes"] = free

    # -- time --------------------------------------------------------------
    con.say("")
    con.say("HOW LONG")
    bench = (None if (args.no_bench or simulated)
             else microbench(device, bits, con))
    if simulated and not args.no_bench:
        con.warn("device is simulated, so no micro-benchmark was run and there is "
                 "no timing estimate. The memory plan above IS valid: it is "
                 "arithmetic over the model census and your stated budget.")
    if bench:
        passes = memplan.passes if memplan else 1
        decode_s = bench["routed_pass_seconds"] * passes
        io_s = artifact_bytes / (2.0 * 1e9) * passes       # ~2 GB/s conservative
        compute_s = 200.0
        score_s = 60.0
        per_run = decode_s + io_s + compute_s + score_s
        total = per_run * args.runs
        con.kv("measured just now", "%.1f ms per decoded matrix on %s"
               % (bench["seconds_per_matrix"] * 1000, bench["device"]))
        con.say("      routed decode  %s  (%s matrices x %d pass)"
                % (human_duration(decode_s), "{:,}".format(MATRICES_PER_PASS), passes))
        con.say("      weight I/O     %s  (%s at ~2 GB/s)"
                % (human_duration(io_s), human_bytes(artifact_bytes)))
        con.say("      forward+score  %s  (ESTIMATE -- see the caveat below)"
                % human_duration(compute_s + score_s))
        con.kv("per cold run", human_duration(per_run))
        con.kv("total (%d run%s)" % (args.runs, "" if args.runs == 1 else "s"),
               human_duration(total))
        out["timing"] = {"bench": bench, "per_run_seconds": per_run,
                         "total_seconds": total, "runs": args.runs}
    else:
        con.kv("estimate", "unavailable (no micro-benchmark could be run)")

    if args.runs < 2:
        con.warn("--runs %d produces a receipt the registry will REJECT: a "
                 "published row needs run_count >= 2, because one run cannot "
                 "show determinism. Use --runs 2 to submit." % args.runs)
        out["submittable"] = False

    con.warn("The forward-pass term is an ESTIMATE, not a measurement. 34 of this "
             "model's 45 layers are Kimi-Delta linear attention, whose only known "
             "fast paths are Triton/CUDA or a fused MLX kernel; whether a "
             "torch/MPS path exists at usable speed is UNVERIFIED. If it falls "
             "back to the reference implementation the run is much slower.")

    # -- engine ------------------------------------------------------------
    con.say("")
    engines = load_engines()
    engine = engines.get(lane)
    if engine is None:
        problem("no engine configured for lane %r" % lane,
                ["known lanes: " + ", ".join(sorted(engines))])
    else:
        probe = engine.probe(SUITE_ROOT)
        out["engine"] = {"lane": lane, "entrypoint": engine.entrypoint,
                         "pinned": engine.pinned, "probe": probe}
        if engine.pinned and not probe["missing_flags"]:
            con.ok("engine %s" % engine.entrypoint, "pinned")
        elif args.dry_run or args.estimate_only:
            con.warn("WOULD REFUSE (real run): lane %r has no pinned engine" % lane)
            con.say("           %s" % engine.unpinned_reason)
            out["would_refuse"].append("lane %s engine unpinned" % lane)
        else:
            raise EngineUnpinned(engine)

    out["census"] = cen.to_dict()
    return out


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure-local", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", help="HF repo id of the quant to measure")
    p.add_argument("--revision")
    p.add_argument("--panel", help="HF dataset id of the panel/teacher")
    p.add_argument("--panel-revision")
    p.add_argument("--panel-descriptor")
    p.add_argument("--bf16", help="optional: base BF16 repo, for full inventory "
                                  "verification only (weights are NOT needed)")

    d = p.add_argument_group("device and memory")
    d.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    d.add_argument("--lane", choices=("local-mps", "local-cuda-budget"))
    d.add_argument("--vram-budget", type=float, metavar="GB",
                   help="HARD bound on accelerator memory. Honoured, not approximated.")
    d.add_argument("--expert-chunk", default="auto",
                   help="experts decoded at once (numerics-invariant)")
    d.add_argument("--window-batch", type=int, default=None,
                   help="windows carried together (numerics-invariant)")
    d.add_argument("--decode-batch-matrices", type=int, default=4)
    d.add_argument("--prefetch-depth", type=int, default=2)
    d.add_argument("--nonrouted-residency", default="auto",
                   choices=("auto", "pinned", "mmap", "gpu"))
    d.add_argument("--kld-device", default="cpu", choices=("cpu", "cuda", "mps"))
    d.add_argument("--simulate-device", metavar="NAME:VRAM_GB[:count][:unified]",
                   help="plan for hardware you do not have in front of you, e.g. "
                        "'RTX 5090:32' or 'Mac Studio:128::unified'. Refuses "
                        "exactly as the real device would; cannot benchmark.")

    r = p.add_argument_group("run")
    r.add_argument("--runs", type=int, default=2,
                   help="cold runs (2 is the registry minimum for a "
                        "submittable row; 1 measures but cannot be submitted)")
    r.add_argument("--reduce-order", default="fp32", choices=("fp32", "native"))
    r.add_argument("--keep-student-logits", action="store_true")
    r.add_argument("--work", help="working directory (default ./fidelity-local)")
    r.add_argument("--out", help="receipt output directory")
    r.add_argument("--max-runtime", default=None)
    r.add_argument("--fixture", help="smoke on a tiny model before the real one")

    m = p.add_argument_group("modes")
    m.add_argument("--estimate-only", action="store_true",
                   help="print the plan and exit; download nothing")
    m.add_argument("--dry-run", action="store_true", help="alias for --estimate-only")
    m.add_argument("--probe-engines", action="store_true",
                   help="report each lane's engine and whether its flags resolve")
    m.add_argument("--selftest", action="store_true",
                   help="run the fit and decode-parity selftests")
    m.add_argument("--no-bench", action="store_true",
                   help="skip the micro-benchmark (timing estimate unavailable)")
    m.add_argument("--yes", action="store_true")
    return p


def probe_engines(con: Console) -> int:
    engines = load_engines()
    worst = EXIT_OK
    for lane, engine in sorted(engines.items()):
        info = engine.probe(SUITE_ROOT)
        state = ("PINNED" if engine.pinned else "unpinned")
        con.say("%-20s %-9s %s" % (lane, state, engine.entrypoint))
        con.say("    present: %s" % info["present"])
        if info["found_flags"]:
            con.say("    found:   %s" % " ".join(info["found_flags"]))
        if info["missing_flags"]:
            con.say("    MISSING: %s" % " ".join(info["missing_flags"]))
        for prob in info["problems"]:
            con.say("    note:    %s" % prob)
        if engine.pinned and info["missing_flags"]:
            worst = EXIT_REFUSED
        con.say("")
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    con = Console()

    if args.probe_engines:
        return probe_engines(con)

    if args.selftest:
        from fidelity.common import run as _run
        rc = 0
        for name in ("selftest_fit.py", "selftest_decode_parity.py"):
            path = SUITE_ROOT / "bin" / name
            con.say("=== %s ===" % name)
            proc = _run([sys.executable, str(path)], check=False, timeout=1800)
            sys.stdout.write(proc.stdout)
            rc = rc or proc.returncode
        return rc

    if not args.artifact or not args.panel:
        con.err("--artifact and --panel are required "
                "(or use --probe-engines / --selftest)")
        return EXIT_REFUSED

    if args.kld_device == "mps":
        con.err(
            "--kld-device mps is refused. MPS cannot represent float64 at all, "
            "and estimator.accumulation_dtype is a comparability key input: "
            "silently scoring in fp32 would move your row into a different "
            "comparability group without anyone noticing. Score on CPU "
            "(--kld-device cpu); for this panel that costs about 10 seconds.")
        return EXIT_REFUSED

    con.say("fidelity-local %s" % VERSION)
    con.rule()
    try:
        result = plan(args, con)
    except Refusal as exc:
        con.say("")
        con.say("REFUSE: %s" % exc.reason)
        for line in exc.advice:
            if line:
                con.say("        %s" % line)
        return EXIT_REFUSED
    except EngineUnpinned as exc:
        con.say("")
        con.say("REFUSE: %s" % exc)
        return EXIT_REFUSED
    except HFError as exc:
        con.err(str(exc))
        return 1

    outdir = Path(args.out or "./fidelity-local").resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(str(outdir / "local-plan.json"), result)

    con.say("")
    con.rule()
    blockers = result.get("would_refuse") or []
    if args.estimate_only or args.dry_run:
        con.say("ESTIMATE ONLY -- nothing downloaded, nothing measured.")
        con.say("plan written to %s" % (outdir / "local-plan.json"))
        if blockers:
            con.say("")
            con.say("%d check(s) would REFUSE a real run:" % len(blockers))
            for b in blockers:
                con.say("  - %s" % b)
            return EXIT_REFUSED
        con.say("all checks passed.")
        return EXIT_OK

    if blockers:
        con.say("REFUSE: %d check(s) failed" % len(blockers))
        for b in blockers:
            con.say("  - %s" % b)
        return EXIT_REFUSED

    con.say("Plan accepted, but the measurement engine for this lane is not "
            "pinned in this checkout, so there is nothing to execute yet.")
    con.say("See bin/engines.json and `bin/measure-local --probe-engines`.")
    return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
