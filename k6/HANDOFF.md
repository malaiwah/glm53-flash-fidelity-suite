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
    stage transition; log the balance at every phase boundary. Budget is
    SHARED across sessions — announce your rentals. And keep a captain's log as
    you go: half of this file existed in JOURNAL.md before anyone asked for it.

    **Correction (2026-08-28):** this lesson used to say the `jl get` cost
    field is a rate, not a total. That is wrong, and a cost model built on it
    is off by a factor of the elapsed hours. It is a running USD **total**.
    Verified by reconciling three live instances against their published
    rates: an 8×H200 spot box at 2h33m reported `cost` 40.897 → $16.04/h
    against a list rate of 8 × $1.99 = $15.92/h; a 1×H200 spot box at 2h28m
    reported 5.081 → $2.06/h against $1.99; a CPU VM at 8h56m reported 1.901.
    All three reconcile as accumulated totals within rounding, and none
    reconciles as a rate. `bin/measure_cloud.py` treats it as a total and
    cross-checks it against the account balance delta, which is the only
    figure that also catches filesystem charges.

**Standing wishes for whoever touches the stack next:** upstream the K8/K3
admissions + `--overlap-seal` + the claim-based prep loop to brandonmusic's
repo so campaigns start patch-free; make capture contracts embed upstream
repo+revision (not container paths); add the seal fraction to the rehearsal
receipt so fleet sizing is automatic; and get the tiny-fixture recipe into CI
of whatever serving runtime hosts these quants.

---

# Part 2 — measurement, publication and community (added 2026-08-29)

The ten tips above are about *making* a quant. These are about *measuring*,
*publishing* and *comparing* one — the half of the campaign that produced the
most surprises.

11. **Budget disk for the MEASUREMENT phase, not just the encode.** Each
    streaming panel run writes `positions x vocab x 4` bytes of fp32 student
    logits — for GLM-5.3-Flash that is 51,175 x 154,880 x 4 = **31.7 GB per
    cold run per model**, and runs are KEPT for the determinism check. Two
    models x two runs = ~127 GB that no encode-era ledger predicts. We hit
    "Disk quota exceeded" twice; the second time cost 36 idle minutes because
    the watcher was counting output files rather than run state.

12. **A failed `jl run` leaves the box IDLE but RUNNING.** Window-count
    watchers cannot see it — a stalled counter looks exactly like slow
    progress. Poll `jl run status --json` and alert on `failed`. Better still,
    compute a pace (min/window) and an ETA from file mtimes: a real stall
    moves "minutes since last write" past one window's pace, while GPU
    utilisation at any instant proves nothing (these workloads are bursty and
    frequently sample at 0%).

13. **Hash CONTENT, never CONTAINERS, when testing determinism.** Capture
    receipts embed `elapsed_seconds`; safetensors embed `__metadata__`
    (cold_run, backend identity). Both differ between runs of a bit-exact
    computation. We raised two false "nondeterminism" alarms in one hour
    before comparing tensor bytes and finding max_abs_diff exactly 0.0. The
    only valid artifacts are raw tensor bytes or a sealed tokenwise hash.

14. **Never quote a single-window KLD as a rate comparison.** Per-window
    scatter (sd 1.73e-3 here) exceeds the effect between adjacent bit-widths
    (K6 vs K8 = 1.22e-3). One unlucky window made our K8 look WORSE than K6;
    the full 25-window panel showed it winning decisively. Previews are fine
    for "is the pipeline alive" — label them, and never let one stand in for
    the panel. A corollary for any registry: single-window panels belong in
    their own comparability group.

15. **`cos(|a|,|b|) ~ 2/pi` with matching sorted spectra means PERMUTED, not
    broken.** A weight-space audit of this pipeline showed decoded payloads
    uncorrelated with the BF16 tensors their provenance named. The cause was
    an intermediate-channel permutation (the expert-MLP symmetry), recovered
    empirically as a perfect bijection and identical across K6/K8 because they
    share a transform seed. Undo it before comparing to source weights, or
    lose a day.

16. **Two lanes need a measured bridge, not an assumption.** Bitwise
    cross-topology parity is IMPOSSIBLE — expert-combine order alone moves
    logits by rms 0.26-0.28, because bf16 addition is not associative. What IS
    achievable: NCCL's bf16 all_reduce behaves like an fp32 accumulate, so
    `--reduce-order fp32` brought the single-GPU lane to within -8.5e-6 (0.06%)
    of the 8xH200 sealed lane on the full panel. Publish the offset as a field;
    mark `publishable_as_reproduction: false`; never silently rank rows from
    different lanes.

17. **Right-size by the phase's bottleneck.** Encode is CPU-bound (GPUs idle
    ~85%). Sealed qualification is a MEMORY problem (decoded-BF16 experts /
    EP ranks must fit VRAM). Streaming qualification is decode-bound and fits
    ONE GPU — ~$6/model against ~$50 for the 8xGPU protocol, because the
    schedule goes layer-outer and decodes each expert once for the whole panel
    instead of once per window (a 25x difference on identical hardware).

18. **Sync is a two-way street.** The repo copy and the box copy of
    `stage_k6.sh` diverged for hours — the box gained an entire stage the repo
    never received, so a downstream agent "verified" a CLI that did not exist
    and wrote a runner against a guessed contract. After every on-box fix,
    pull it back to git the same way you push. Pin engine CLIs by PROBING the
    real file, never by reading its docs.

19. **HF lineage is a discovery decision with an honesty cost.** Z.ai
    published two sibling roots (`GLM-5.3-Flash` FP8 with ~1,484 likes and
    `GLM-5.3-Flash-BF16` with 41) and neither declares the other. Quants that
    declare the FP8 as `base_model` appear on the busy page; ours declare BF16
    because that is what we quantized from. Do not "fix" visibility by
    declaring a base you did not use — use collections, cross-links, an
    explicit lineage section, and discovery tags instead.

20. **Engage the ecosystem with numbers, not claims.** Publishing a measured
    comparison on someone else's artifact (with their method, their panel, and
    a receipt link) got the sealed numeric core we were missing published
    within a day, turned a "disclosed reconstruction" into a **verified-
    equivalent** codec (120/120 encodes byte-identical, 0 differing bytes of
    624 MiB), and produced an independent base measurement nobody had. Lead
    with what the other person did well, state your deviations before anyone
    asks, and close the loop when they deliver.
