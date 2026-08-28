# 10 tips & wishes for the next campaign (GLM-5.3-full and beyond)

Distilled from the GLM-5.3-Flash K6/K8 campaign (2026-08-27/28). The full
record: `JOURNAL.md` (27+ lessons), `k6/DECISIONS.md` (9 operator decisions),
`k6/RUNBOOK.md`, patch series `k6/patches-v2/0001-0010`.

1. **Fixture before real weights, always.** The 0.1B random fixture
   (`inference-optimization/GLM-5.3-Flash-0.1B-A0.1B`) validated the entire
   chain in minutes and its per-matrix bench set the fleet plan. For a new
   model, find or build the tiny-random fixture FIRST and run
   `k6_driver.py rehearse` end-to-end on a $2 GPU-hour before renting anything
   big. If no fixture exists, make one (tiny dims, full architecture, stock-
   transformers-loadable) — it pays for itself the same day.

2. **Bench the production path, not the kernel.** Our kernel bench said
   0.84 s/matrix; production ran 2× slower because prep/seal/receipt CPU work
   dominates. Run ONE full work unit (prep → encode → seal) and read the
   seal fraction before sizing the fleet. Corollary: **encode is CPU-bound —
   size boxes by cores and worker count, not GPU class.** 4×H100 ≈ 4×H200
   for this workload at 60% of the price; the GPUs idle ~85% either way.
   Qualification is the opposite: it's a MEMORY problem (decoded-BF16 student
   ÷ EP ranks must fit VRAM — do that division before picking the box).

3. **Re-derive every hardcoded census from the new config.** The pipeline and
   our driver embed Flash's geometry everywhere: 42 routed layers, 288
   experts, 36,288 main + 864 MTP matrices, 120 shards, byte counts. GLM-5.3-
   full changes all of them. `grep -nE "36288|864|42|288|253536370680"` the
   bundle and re-derive from `config.json` before the first run — a wrong
   census fails at the LAST verify, not the first.

4. **Expect never-executed paths to stop you; make stops cost minutes.** K6
   took five launches (bridge doc, ALLOWED_BITS, sealed-import ×2, arch-list
   env). The architecture that made that cheap: fail-fast guards before spend,
   all state on the shared fs, per-expert receipt resume, atomic `.new`→`mv`
   script syncs, receipts-over-exit-codes. Never fix forward on a box; fix in
   the bundle, sync, relaunch — the resume machinery eats the retry.

5. **Prep and encode want different parallelism.** The contract's prep loop is
   single-GPU serial: fan `k6_driver.py prepare --layers A-B` range workers
   across idle GPUs (3× faster) — but DRAIN them before rerunning the
   contract sweep. We hit a staging-dir race at layer 20 because cache-warm
   prep closed a "comfortable" margin. Wish: make the prep loop claim-based
   like encode so the barrier disappears.

6. **Check `free -g` before assuming IO-bound.** The 3 TB box had the whole
   1.1 TB working set in page cache; our warmup pass "finished" in 76 s
   because there was nothing cold. Burst-idle GPUs + pegged CPU = compute in
   the seal/conditioning path, not IO. The `--overlap-seal` flag (verified
   byte-transparent) recovers part of it; ceiling is the seal fraction
   (~15%), measured per campaign by `overlap_smoke.sh`.

7. **Respect the sealed-pipeline design; extend it honestly.** Bit-width
   admissions live in ~6 places (`SUPPORTED_BITS`, codec adapter, v31
   ALLOWED_BITS, reader trellis-words, materializer schema strings, qualify
   argparse) — patches 0007-0010 are the map. Gated modules (the K4-KL gate)
   get DISCLOSED bridge docs carrying the author's real published hashes,
   never fabrications. Every deviation (hardware attestation, reconstructed
   codec, admissions) goes in receipts and cards. Disclosure is what lets you
   move fast without burning trust.

8. **Uniform-K campaigns + one transform seed = the parts bin.** Don't build
   mixed-precision campaigns; encode uniform K6 and K8 with the SAME seed and
   calibration, publish both payload stores, and mixed builds become offline
   CPU assembly forever. Prep is K-specific (GSS targets the codebook);
   moments/calibration are shared. Known second-order caveat: down_proj
   conditioning assumes its own K's gate/up decode — measure the mix on the
   panel before claiming numbers.

9. **Naming and release discipline.** TR3 (codec family), never "EXL3" —
   these are not stock-exllamav3-loadable (turboderp's explicit ask). Cards
   state codec-vs-runtime plainly and credit the whole lineage (zai, Brandon,
   turboderp). Weights upload PRIVATE the moment they materialize (parallel
   with qualification), flip public only on a green five-cold-run panel
   receipt. The first-X claim is worthless without the number attached.

10. **Supervise like money is burning, because it is.** 10-minute watchdog on
    every rental (run footer + GPU util + log growth + disk free; wedge = 2
    quiet ticks); pause/destroy boxes the second their role ends; ntfy every
    stage transition; log the balance at every phase boundary (the `jl get`
    cost field is a rate, not a total). Budget is SHARED across sessions —
    announce your rentals. And keep a captain's log as you go: half of this
    file existed in JOURNAL.md before anyone asked for it.

**Standing wishes for whoever touches the stack next:** upstream the K8/K3
admissions + `--overlap-seal` + the claim-based prep loop to brandonmusic's
repo so campaigns start patch-free; make capture contracts embed upstream
repo+revision (not container paths); add the seal fraction to the rehearsal
receipt so fleet sizing is automatic; and get the tiny-fixture recipe into CI
of whatever serving runtime hosts these quants.
