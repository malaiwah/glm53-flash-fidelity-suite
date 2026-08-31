# Review findings that were NOT applied, and the exact patch for each

**Written:** 2026-08-29, during a full peer review of this suite.
**Why this file exists:** a sequential measurement campaign (M2/M3/M4) was running against
rented GPUs while the review ran, and it owns five files. A second session was concurrently
rewriting three more. Editing a file another process is mid-write on is how a campaign
loses a measurement and how a reviewer loses a fix; so every finding that lands in one of
those files is written down here instead, with enough detail to apply without re-deriving
it.

Nothing in this file is speculative. Every claim below was reproduced. Where a proposed fix
was tested and found to be wrong, that is said explicitly, because a patch that looks right
and regresses something is worse than no patch.

This file now covers two different reasons to hold a fix back:

1. **A live workflow owns the file.** Editing it races another process.
2. **The fix would change a number that is already published.** Those need an operator
   decision and a disclosure, not a quiet edit. They are in
   [Published-number changes](#published-number-changes-operator-decision-required),
   each with the delta measured rather than estimated.

## The files this covers

| File | Owner while the review ran |
|---|---|
| `bin/measure_cloud.py` | M2/M3/M4 campaign |
| `bin/stage_measure.sh` | M2/M3/M4 campaign |
| `bin/fidelity/hfmeta.py` | M2/M3/M4 campaign |
| `bin/engines.json` | M2/M3/M4 campaign |
| `bin/invoke_engine.py` | M2/M3/M4 campaign |
| `bin/fidelity_dataset.py` | a second session (hf-transformers capture engine, uncommitted) |
| `bin/fidelity/cardmeta.py` | a second session (uncommitted) |

## Line numbers in this file DRIFT — find the code by these anchors instead

The campaign edits the locked files while this document sits still. Between the first
pass and the second-pass review, `f57f9ae` moved `bin/stage_measure.sh` by ~16 lines and
`bin/measure_cloud.py` by ~100, so every `file:line` below is a hint, not an address.
Each finding was re-verified against the tree at commit `a50d0f4`; locate the code with
these greps, which are stable across the moves observed so far.

| Finding | `grep -n` anchor |
|---|---|
| SEC-01 | `eval HF_HUB_ENABLE_HF_TRANSFER` in `bin/stage_measure.sh` (was :199, now :215) |
| SEC-01 companion | `def load_panel_descriptor` in `bin/fidelity/hfmeta.py` |
| CLI-01 | `def _destroy_instance` in `bin/measure_cloud.py` |
| CLI-02(b) | `def run(self, reason` in `bin/measure_cloud.py` (`self.done = True` is the 4th line) |
| CLI-11 / SEC-08 | `noqa: S202` in `bin/measure_cloud.py` |
| SH-05 | `arming on-instance watchdog` in `bin/measure_cloud.py` (was :1469, now :1544) |
| SH-10 | `get("include", \["\*"\])` in `bin/stage_measure.sh` |
| SH-22 | `capture-receipt.json" \]; then` in `bin/stage_measure.sh` (was :250, now :278) |
| CC-07 | `store_published = any(` in `bin/fidelity/hfmeta.py` |
| CLI-16 | `scored_positions` in `bin/fidelity/hfmeta.py` |
| CLI-25 | `def fetch_file` in `bin/fidelity/hfmeta.py` |
| SEC-09 | `os.chmod(tmp, 0o600)` in `bin/measure_cloud.py` (was :1459, now :1534) |
| NUM-16 | `bf16-floor` in `bin/engines.json`; `extra = {` in `bin/invoke_engine.py` |
| REAP-1 | `reaper: destroying` in `bin/measure_cloud.py` |
| REAP-2 | `def parse_deadline_name` in `bin/measure_cloud.py` |
| REAP-3 | `retiring lease` in `bin/measure_cloud.py` |

Re-verified at `a50d0f4`: every finding below still reproduces, and no proposed patch
has been overtaken by a campaign edit.

---

# CRITICAL / HIGH

## SEC-01 — command injection into a rented GPU box that holds a live HF token

> **APPLIED 2026-08-30 (M4), after the campaign that locked these files ended.** The
> `eval` is gone (NUL-delimited bash array, `mapfile -d`), `load_panel_descriptor`
> validates `repo_id` and `revision` at ingestion, and `hfmeta` carries a comment
> recording that `repo_meta`/`resolve_revision` are load-bearing for shell safety.
> Regression: `bin/selftest_shell_guards.sh` drives the REAL `fetch_panel` stage with
> a hostile `panel.repo_id` and a stub `hf`. Against the unpatched tree the
> substitution executes and the recorded argv shows only `org/panel`, exactly as filed;
> against the patched tree nothing runs and the hostile string arrives as ONE literal
> argument. The rung SKIPS loudly where bash is older than 4.4 rather than passing on a
> shell that cannot run the code under test.

**File:** `bin/stage_measure.sh:199-201` (`fetch_panel`)
**Severity:** medium as the code stands (see reachability), but it is an unsafe `eval` one
refactor away from RCE on a token-bearing paid box.

**Claim.** The include globs are correctly `shlex.quote`d, which makes the line LOOK safe,
but `$REPO` and `$REV` sit inside the same `eval` and get a second round of shell parsing:

```sh
eval HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
  "$VENV/bin/hf" download "$REPO" --repo-type dataset --revision "$REV" \
    --local-dir "$PANEL" $INCLUDES
```

`panel.repo_id` reaches `$REPO` from `job.json`, which `hfmeta.load_panel_descriptor` reads
verbatim out of an operator-supplied `--panel-descriptor` file with no validation.

**Repro.** With a stub `hf` on PATH and a job.json whose `panel.repo_id` is
`org/panel$(id -un > /tmp/PWNED.txt)`, the file is created and the *logged* argv shows only
`download org/panel ...` — the substitution is stripped from the log, so the injection is
invisible in `fetch_panel.log`.

**Reachability — this is why it is not critical today.** `bin/measure_cloud.py:934` calls
`repo_meta(descriptor.repo_id, ...)` and re-raises on failure unless `--dry-run`, so an
injecting repo id cannot resolve on the Hub and a live run aborts. `measure_cloud.py:936`
overwrites the revision with `pmeta.revision`, which `hfmeta.resolve_revision` guarantees is
40-hex. So the guard is real but **incidental, undocumented, and in another file and
language** — nothing marks `repo_meta` as a security control, and an air-gapped or cached
plan path that writes `job.json` without the HF round-trip re-arms it immediately.

**Patch (bin/stage_measure.sh:199-201).** Delete the `eval`; it exists only to word-split
`$INCLUDES`. Use a NUL-delimited bash array, which also fixes a newline in a pattern being
silently split into two argv entries:

```sh
  mapfile -d '' -t INCLUDES < <(python3 - "$CONF" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for p in doc.get("panel", {}).get("include", ["*"]):
    sys.stdout.write("--include\0" + p + "\0")
PY
  )
  HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME="$FS/hf" \
    "$VENV/bin/hf" download "$REPO" --repo-type dataset --revision "$REV" \
      --local-dir "$PANEL" "${INCLUDES[@]}" >>"$LOGS/fetch_panel.log" 2>&1
```

Drop `shlex.quote` from the emitter — array elements must NOT be pre-quoted or the literal
quotes become part of the glob. Requires bash 4.4+ (the instance is Ubuntu bash 5; do not
port this idiom to a macOS-local script, bash 3.2 has no `mapfile`).

**Companion patch (`bin/fidelity/hfmeta.py:636-658`, also locked).** Validate at ingestion so
a hostile value never reaches job.json:

```python
    repo_id = str(raw["repo_id"])
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", repo_id):
        raise HFError("panel descriptor repo_id %r is not an owner/name pair" % repo_id)
    revision = str(raw["revision"])
    if not re.match(r"^[A-Za-z0-9._-]+$", revision):
        raise HFError("panel descriptor revision %r is not a revision" % revision)
```

Note the reviewer's suggested revision pattern `^[A-Za-z0-9._/-]+$` is WRONG — drop the `/`,
and note this is weaker than `resolve_revision`'s existing 40-hex guarantee, so it is a
backstop, not the control.

**Also add a comment** at `hfmeta.repo_meta` / `resolve_revision` recording that
`stage_measure.sh`'s `fetch_panel` depends on them for shell safety, so the incidental guard
is not removed by a future refactor.

---

## CLI-01 — the teardown reads "the API call failed" as "the instance was destroyed"

> **APPLIED 2026-08-30 (M4).** `_confirm_gone` (via `list_instances`, which propagates
> `JLError`) replaces `jl.get(mid) is None`; destruction is accepted only on a positive
> `True`, unexpected exceptions in the destroy loop fall through to the next attempt,
> and an exception escaping a destroy STEP now sets `leaked = True` so `_drop_lease`
> keeps the lease for the reaper. The reviewer's `get_strict()` was NOT used, for the
> reason recorded below. Regression: `bin/selftest_teardown.py` (new; there was no test
> for `Teardown` at all) covers healthy destroy, total outage, outage on the confirm
> only, and a raising destroy step. Against the unpatched tree it reproduces the filed
> repro: destruction declared on attempt 1 of 5 with zero successful API interaction.

**File:** `bin/measure_cloud.py:291-311` (`Teardown._destroy_instance`), with
`bin/fidelity/jlapi.py:230-237`

**Claim.** `jl.get(mid)` returning `None` is treated as proof of destruction, but
`JLApi.get()` swallows every `JLError` and returns `None`. A transient API failure during
teardown is read as "destroyed": `machine_id` is set to None, `leaked` stays False, and
`_drop_lease()` deletes the lease, so the reaper never looks at the box again.

**Repro (no rental).** A fake `jl` on PATH that exits 1 with empty stdout, then the real
`Teardown` with `machine_id` set:

```
WARNING  destroy attempt 1: jl destroy 486969 exited 1: ...Max retries exceeded
instance destroyed                     ok  486969
-> machine_id=None  leaked=False  lease deleted  exit=0
```

The `jl.destroy()` on that same attempt RAISED and was merely warned; the next line still
printed "instance destroyed ok". Destruction is declared on attempt 1 of 5 with zero
successful API interaction.

**Blast radius is narrower than "unbounded".** In the default config (`fs_id` present,
`keep_fs` false) `_destroy_fs` fails on the same outage, so `leaked=True`, the lease is
KEPT, and exit is 90 (EXIT_LEAK) — the instance leak is still mis-reported as "ok" and the
banner names only the filesystem, but the reaper still owns the box. Fully silent
lease-dropping needs `--keep-fs`, or `fs_id` None. L3 (the name-encoded deadline sweep) also
still reaps it, if a reaper is installed at all.

**Patch — do NOT use the reviewer's proposed `get_strict()`.** On a HEALTHY API a destroyed
instance is a 404: vendored `jl 0.2.17` raises `NotFoundError("Instance N not found...")`,
`cli/app.py` renders it as `{"error": ...}` and exits 1, which `_call` turns into `JLError`.
So `get() -> None` IS the normal, load-bearing signal for "successfully destroyed" — making
`get()` strict would disarm the leak detector on EVERY run, not after 5 attempts. (Verified:
the strict variant escapes `_destroy_instance` entirely, because the `try` wraps only
`jl.destroy`, not `jl.get`.)

Confirm with `jl list`, whose `list_instances()` already propagates `JLError`:

```python
    def _confirm_gone(self, mid):
        """True gone, False alive, None unknown. `jl get` cannot answer this: a 404 for a
        destroyed instance and a 404 from an outage are the same JLError, and JLApi.get()
        turns both into None."""
        try:
            alive = {i.machine_id for i in self.jl.list_instances()}
        except JLError:
            return None
        return mid not in alive
```

In `_destroy_instance`, accept destruction only on `True`; on `None` or `False` fall through
to the next attempt, and after 5 attempts set `self.leaked = True`. Never accept
confirmation from an attempt whose `jl.destroy()` raised, unless the confirmation is a
positive "gone". Wrap the whole loop body so any unexpected exception falls through to the
retry rather than escaping with `leaked=False`.

**Independent hole in the same function.** Any exception escaping `_destroy_instance` or
`_destroy_fs` leaves `leaked=False` and `_drop_lease` deletes the lease anyway. In
`Teardown.run`, treat an escaping exception as `leaked=True`.

**Test it without a rental.** A fake `jl` shell script plus the real `Teardown` class covers
all four cases (healthy destroy; total outage; outage on the confirm only; `--keep-fs` under
outage). There is no selftest for `Teardown` at all today.

---

## CLI-02 (part b) — the teardown marks itself done before it does anything

> **APPLIED 2026-08-30 (M4)**, and it stopped being defence-in-depth on the way: when
> the harness reaped M4's controller mid-run, the teardown reached `pulling receipts`
> and stopped, so `_shred_secrets` never ran and the 0600 HF token sat on a rented box
> for an hour with nothing able to retry it. `_running` is now a separate re-entrancy
> flag cleared in a `finally`; `done` is set only after the steps loop was attempted, so
> a second `run()` RETRIES; and the SIG_IGN restore moved inside a try that opens before
> the handlers are installed, so a raise cannot leave the process immune to `kill`.

**File:** `bin/measure_cloud.py:160-190` (`Teardown.run`)

**Claim.** `self.done = True` is set at line 165, then `con.say("")` / `con.step("teardown
...")` at 184-186, and only at 192 does the `try` that runs the destroy steps begin. A
console write that raises therefore skips `_destroy_instance` / `_destroy_fs` /
`_drop_lease` with `done` already True, so the atexit hook and the outer `finally` both
no-op and the instance is never destroyed.

**Part (a) — `common.Console._w` swallowing the error — IS FIXED** on the unlocked side
(see the commit for CLI-02). With that in place `Teardown.run` reaches the destroy steps
regardless, so what remains here is defence in depth.

**Repro.** With a closed stdout, `Console.say("")` raises `BrokenPipeError`; under a real pty
whose master is closed (the SIGHUP case) it raises `OSError: [Errno 5]`, so a fix that
catches only `BrokenPipeError` leaves the flagship scenario broken. Recorded output with the
pre-fix Console:

```
[pipeline] issuing kill; [stage EXIT trap] rc=0; ps: child orphaned (PPID=1) and still alive
run-RAISED:BrokenPipeError  done_flag=True  leaked_flag=False  finally-run  atexit-run
```

**Patch.** Keep the re-entrancy guard, but as a SEPARATE flag, and clear it in a `finally`
or an exception mid-teardown reproduces this bug in a new costume:

```python
        with self._lock:
            if self.done or self._running:
                return
            self._running = True
        try:
            ...                      # announce, install SIG_IGN, run the steps
        finally:
            self._running = False
            self.done = True
```

Set `self.done = True` only after the steps loop has been attempted. This makes a second
`td.run()` RETRY the destroy rather than no-op, which is desired and safe:
`_destroy_instance` sets `self.machine_id = None` on confirmed destruction and
`_destroy_fs` early-returns on `fs_id is None`.

**While in there:** the SIG_IGN restore lives in the `finally` of the try at 192, so any
raise between 180 and 192 leaves SIGINT/SIGTERM/SIGHUP ignored for the life of the process —
the process that just leaked the GPU is immune to ^C and to `kill`. Move the restore out, or
install the handlers after the announcement.

---

## CLI-11 / SEC-08 — unfiltered `tar.extractall` of an archive built on a rented box

> **APPLIED 2026-08-30 (M4)**, using the explicit member pass as the load-bearing
> control with `filter="data"` added only where `tarfile.data_filter` exists, and with
> the link rejection BEFORE the `resolve_inside` check for the reason filed. Regression:
> `bin/selftest_teardown.py` builds a `receipts.tar.gz` with an absolute member, a `..`
> member, and a symlink-plus-write-through, and asserts the victim outside `outdir` is
> untouched while the legitimate member still extracts. Against the unpatched tree the
> victim's contents become `evil`.

**File:** `bin/measure_cloud.py:256-257` (`_pull_receipts`)

**Claim.** `tf.extractall(self.outdir)` with no member filter, annotated
`# noqa: S202 - our own archive`. The archive is built by `cd $FS && tar czf ... receipts` on
a rented instance and arrives through the vendor `jl` control plane — it is not "our own" in
any security-relevant sense. On Python 3.9, the documented stock target, `extractall`
applies no filter and emits no warning.

**Repro.** A synthetic `receipts.tar.gz` plus the literal two lines from the file: members
with an absolute path, a `../../` component, and a symlink-plus-write-through all escaped
`outdir`, silently. `outdir` defaults to `./fidelity-runs/<job_id>` resolved relative to CWD
and the README invokes `bin/measure-cloud` from the repo root, so two `..` reach the suite's
own source.

**Patch — the reviewer's `filter='data'` alone BREAKS THIS on the project's own interpreter.**
`hasattr(tarfile, 'data_filter')` is False on Python 3.9.6 (PEP 706 landed in 3.9.17), and
`tf.extractall(out, filter="data")` raises `TypeError`. Guard it, and make the explicit pass
the load-bearing one:

```python
    _MAP = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
    with tarfile.open(local) as tf:
        safe = []
        for m in tf.getmembers():
            if m.issym() or m.islnk() or m.type not in _MAP:
                self.con.warn("receipts.tar.gz: refusing %s member %s"
                              % ("link" if (m.issym() or m.islnk()) else "special",
                                 redact(m.name)))
                continue
            if os.path.isabs(m.name) or ".." in Path(m.name).parts:
                self.con.warn("receipts.tar.gz: refusing escaping member %s" % redact(m.name))
                continue
            resolve_inside(str(self.outdir.resolve()), m.name, "receipts.tar.gz")
            safe.append(m)
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(self.outdir, members=safe, **kw)
```

Order matters: the symlink rejection MUST come first. `resolve_inside` calls
`os.path.realpath`, and before extraction the symlink does not exist yet, so both
`receipts/link` and `receipts/link/keep.txt` resolve inside the root — I ran exactly that
pre-check and both PASSED, then `extractall` overwrote the victim file.

Skip-with-a-warning, not raise: raising lands in the `except Exception` at line 262 and falls
back to `jl download -r`, which the docstring records as having "blew the 300-second timeout
and the whole measurement came home with nothing -- twice, observed."

`resolve_inside` is importable — `measure_cloud.py:42` already puts `bin/` on `sys.path` and
imports from `fidelity`.

**Also:** nothing verifies the archive before extraction; the digest the docstring advertises
is computed locally after download. `sha256sum receipts.tar.gz` on the instance, compared
before extract, closes the transport half.

---

## SH-05 — the L1 watchdog is armed and never verified

**File:** `bin/measure_cloud.py:1469-1473`

**Claim.** `jl.exec(mid, "nohup bash %s/bin/watchdog.sh ... &")` backgrounds the process, so
`sh -lc` exits 0 whether or not the watchdog started, and `jl.exec`'s exit-code check always
passes. The controller prints "arming on-instance watchdog" unconditionally. Nothing reads
`$FS/logs/watchdog.log`, and nothing ever checks that the process exists.

**Corroboration.** Commit 945255b records that on M2 a remote process launched WITHOUT
`setsid` was killed when the local session died — "the remote process group simply went away
with the session that started it", 65 min of a rented H200 rebought. That commit fixed the
STAGE launch and left the watchdog arming in the old shape.

**Patch.** Add `setsid` and `</dev/null` to the arming itself, then verify:

```python
    _arm = ("nohup setsid bash %s/bin/watchdog.sh %d %d %s >%s/logs/watchdog.log 2>&1 "
            "</dev/null &" % (td.fs_root, int(plan_data["deadline_epoch"]),
                              int(args.heartbeat_timeout), td.fs_root, td.fs_root))
    armed = False
    for _attempt in (1, 2):
        jl.exec(td.machine_id, _arm)
        for _ in range(5):
            time.sleep(2)
            # The bracket keeps the probe from matching its OWN `sh -lc` cmdline under
            # procps-ng pgrep, which excludes only pgrep's pid, not the parent shell.
            out = jl.exec_stdout(td.machine_id,
                                 "grep -q 'watchdog armed pid=' %s/logs/watchdog.log "
                                 "&& pgrep -f 'bin/watchdo[g]\\.sh' >/dev/null "
                                 "&& echo WD_OK || echo WD_NO" % td.fs_root,
                                 timeout=120, check=False)
            if out.strip().splitlines()[-1:] == ["WD_OK"]:
                armed = True
                break
        if armed:
            break
    plan_data["watchdog_armed"] = armed
```

Require BOTH the log line and a live process: grep alone passes for a watchdog that armed and
then died; pgrep alone is the self-match risk.

**Do NOT hard-abort a created, billing instance when it fails.** L1 cannot destroy anything
(`bin/watchdog.sh:12-17` says so: an on-instance script must never carry a JarvisLabs
credential), so its absence costs no rental money — L2/L3 still reap. Warn loudly, persist
`watchdog_armed: false` into the plan and the seal receipt so the run is self-describing, and
gate a hard refusal only on the existing leak-risk condition at `measure_cloud.py:103-104`.

**Separately, verify `pgrep -f 'stage_measure.sh %s'` at `measure_cloud.py:1624`.** If that
self-matches on the instance, `_await_stage` can never return "failed" for a dead stage and
burns the whole `--max-runtime`. That one does cost real money.

---

## SH-10 — `panel.include: []` fetches the entire 1.3 TB dataset

**File:** `bin/stage_measure.sh:192-201` (`fetch_panel`), `bin/fidelity/hfmeta.py:647`

**Claim.** The heredoc falls back to `["*"]` only when the KEY IS ABSENT, not when the list
is empty, so `panel.include: []` produces an EMPTY `$INCLUDES` and `hf download` runs
unscoped. The comment two lines above says include-scoping "is the difference between 32 GB
and 1.3 TB."

**Repro (the heredoc, extracted verbatim):**

```
{"panel":{"include":["windows/*","panel/*"]}} -> --include 'windows/*' --include 'panel/*'
{"panel":{"include":[]}}                      -> (EMPTY)
{"panel":{}}                                  -> --include '*'
{"panel":{"include":null}}                    -> TypeError -> stage aborts (fail-loud, safe)
```

**Correction to the finding as filed.** `--include '*'` is NOT the protective fallback: it
fetches the same 1,318 GB. The correct statement is "there is no guard on this path at all,
in either branch". And `[]` is not currently producible by this codebase —
`hfmeta.load_panel_descriptor:647` does `list(raw.get("include") or ["*"])`, which turns `[]`
into `["*"]`, and `measure_cloud._job_document` is the only writer of job.json. The trigger
is a hand-edited `$FS/job.json`.

**Patch.** Fold into the SEC-01 array rewrite above, and add to the emitter:

```python
patterns = doc.get("panel", {}).get("include")
if not isinstance(patterns, list) or not patterns or patterns == ["*"]:
    sys.stderr.write("panel.include is empty or ['*'] -- refusing an unscoped fetch "
                     "(the panel repo is ~1.3 TB)\n")
    raise SystemExit(2)
```

**Companion (`hfmeta.py:647`, locked).** `list(raw.get("include") or ["*"])` also accepts a
STRING: `"logits/window-*.safetensors"` becomes per-character globs `['l','o','g',...]`,
which matches nothing — the silent-zero class of known defect 5, and it would also make
`bytes_matching` under-report the panel size at plan time. Reject a non-list.

**Stronger, and worth more than either:** job.json already carries `panel.fetch_bytes`. After
the fetch, compare `du -sb "$PANEL"` against it and fail on a large discrepancy. That catches
unscoped fetch, wrong globs and revision drift together.

---

# MEDIUM

## CC-07 — the packed_root pre-flight trap is disarmed by any `.materialization/` file

> **APPLIED 2026-08-30 (M4)** — the predicate only. `store_published` now names the five
> things `stream_score` actually dereferences. Decision 2 in this entry stands and was
> NOT taken: the outer `if info.surface == "packed"` guard is unchanged, so this fixes
> the predicate without the live blast radius, and the code says so. Regression:
> `bin/selftest_teardown.py` runs four repo shapes through the real `sniff_surface` and
> asserts the trap fires on a bare packed repo, on shard receipts, on a stray
> `payload_notes.txt` and on a half-store, and stays silent on a complete store.

**File:** `bin/fidelity/hfmeta.py:501-504`

**Claim.** `store_published = any(p.startswith(".materialization/") or p.startswith("payload")
for p in names)`. `.materialization/shards/*.json` are shard RECEIPTS, not a payload store,
and our own published K6/K8 repos ship 120 of them each. The consumers require something
else entirely: `stream_score.py:2190-2196` needs `contract.json`, `inventory.json`,
`mtp-adapter-receipt.json`, `payload-store/objects/` AND `payload-store/choices/`.

**Repro (four file lists through the real `sniff_surface`, `fetch_json` stubbed):**

```
receipt only, no store                 surface=packed  trap_fires=True
+ 120 .materialization/shards/*.json   surface=packed  trap_fires=False  <-- DISARMED
+ a file named "payload_notes.txt"     surface=packed  trap_fires=False  <-- DISARMED
+ real store (objects+choices+3 JSONs) surface=packed  trap_fires=False
```

The second bypass is not in the finding as filed: `p.startswith("payload")` is a bare STRING
prefix, so any top-level file merely beginning with "payload" disarms the trap.

**Patch.**

```python
    store_published = (
        any(p.startswith("payload-store/objects/") for p in names)
        and any(p.startswith("payload-store/choices/") for p in names)
        and {"contract.json", "inventory.json", "mtp-adapter-receipt.json"} <= set(names))
```

Verified against four repo shapes: it fires on a bare packed repo, on the real TR3
shard-receipt layout, on a stray `payload_notes.txt`, and on a half-store (objects without
choices); it stays silent on a complete store. No regression.

**Two things to decide deliberately before applying.**

1. The suite's own docs (`docs/FIDELITY-DATASET-SPEC.md:1497`, O-5) say the payload store is
   never published, so the strict predicate is essentially always False for a public repo.
   That is the intended semantics of the trap, but `packed` is readable by the LOCAL lanes,
   and `measure_local.py:314` hard-refuses on `surface.problems`. Let the local path proceed
   when an explicit `--packed-root` is supplied, or downgrade to a warning there.
2. **The fix changes nothing observable until the outer guard is widened.** Every observed
   `.materialization/` publisher also ships `exl3-mcg-storage-abi.json`, which classifies it
   `tr3-published` at `hfmeta.py:415`, so `if info.surface == "packed"` (the sibling bypass
   already documented at `bin/selftest_all.sh:136-141`) shadows this one. Widening that guard
   to every surface that resolves a packed_root out of the materialization receipt is the
   change with live blast radius.

---

## CC-08 — MLX, GGUF and NVFP4 have bitwise-verified readers the front door cannot reach

**Files:** `bin/fidelity/hfmeta.py:31-42` (`SURFACE_MARKERS`), `395-549` (`sniff_surface`),
`bin/engines.json` (`lanes.*.surfaces`)

**Claim.** `sniff_surface` has no branch for MLX, GGUF or NVFP4, so those artifacts resolve to
`unknown` and the one-command front ends refuse them with "no recognised surface marker" —
while `engines/tools/{mlx,gguf,nvfp4}_surface.py` exist, are bitwise-verified against
mlx.core / gguf-py / compressed-tensors, `stream_score.py --source` accepts all three, and
`README.md:221` advertises them. `README.md:14` shows an MLX example that only works because
that revision is already in the registry and the front gate answers before the sniff.

**Repro.** Driving the real `sniff_surface` with the repo's own committed evidence configs:

```
MLX      (engines/tools/mlx-evidence/orcarouter-config.json)  -> unknown
NVFP4/RH (engines/tools/nvfp4-evidence/redhat-config.json)    -> unknown
NVFP4/LB (engines/tools/nvfp4-evidence/libertai-config.json)  -> unknown
GGUF     (.gguf siblings, no config)                     -> unknown
```

Also: `SurfaceInfo.surface`'s docstring at line 335 lists four values while the function
emits six (`exl3hf` at 479 and `native-bf16` at 533 are missing).

**Patch — sniffer branches, but NOT with the detection rule as proposed.**

```python
    # gguf: any .gguf sibling AND no safetensors shards (a repo holding both bf16
    # shards and .gguf quants currently sniffs as native-bf16 and PASSES every gate).
    # mlx: a top-level `quantization` dict carrying group_size and bits. Verified true
    # for BOTH orcarouter-config.json and pipenetwork-4bit-config.json. Set the codec
    # explicitly -- normalize_codec has the alias but the mirrored quantization_config
    # block carries no quant_method, so it would publish codec "unknown" with bits 4.0.
    # nvfp4: EITHER quant_method == "compressed-tensors" AND any
    #        config_groups[*].format contains "nvfp4"   (redhat: group_0.format ==
    #        "nvfp4-pack-quantized"; its TOP-LEVEL format is "mixed-precision" and must
    #        NOT be the discriminator)
    #     OR quant_method == "modelopt" AND quant_algo == "NVFP4"   (libertai, which has
    #        no top-level format key at all).
```

The reviewer's proposed rule ("quant_method is compressed-tensors with an nvfp4/e2m1
format") **misses both real NVFP4 releases**. Test any branch against both evidence configs;
a rule that satisfies one is not done.

**Do NOT add mlx/gguf/nvfp4 to `lanes.streaming.surfaces` on its own.** The lane's `flag_map`
has no `mlx_root`/`gguf_file`/`nvfp4_root` entries, so the artifact location can never be
passed; and `profile` is keyed by BITS ONLY (`measure_cloud.py:1345-1352`), so an MLX 4-bit
artifact maps to `tr3-4bpw` and `stream_score.py:1866-1883` hard-refuses "--source mlx and
--profile mlx must be used together" — AFTER the rental. Worse,
`measure_cloud.py:1352-1353` is `if not profile: profile = "k6"`, so an unmapped bits value
silently selects the WRONG profile rather than refusing. Gate all three changes as one
atomic commit and land the surfaces-list line LAST.

**Blind spot the fix does not close.** `sniff_surface` keys on exact ROOT names, and
orcarouter ships five builds as subfolders (`README.md:14` targets `/tree/main/4-bit`).
`path_hint` is threaded to the registry matcher and to `measure-local --path` but never to
`sniff_surface`, so even with the branches added it will not fire on the flagship MLX repo.
Track that separately.

**Unlocked half, applied now:** `bin/measure_one.py:221` hardcodes a THIRD, staler allowlist
`readable = {"packed", "native-bf16"}`, which refuses `exl3hf` and `tr3-published` — surfaces
that are fully wired and published — while its own refusal text says the streaming lane reads
them. That is fixed in the CLI commit.

---

## NUM-16 — engines.json advertises knobs no runner fills

**Files:** `bin/engines.json` (`lanes.bf16-floor.flag_map`), `bin/invoke_engine.py:103-111`

**Claim.** Every other streaming lane maps both `decode_cache` and `decode_cache_dir`; the
`bf16-floor` lane maps only `decode_cache`, and `stream_score.ExpertStreamer.__init__:777-779`
refuses `--decode-cache disk` without a directory. Separately, `invoke_engine.py` populates
`extra` with only source/bf16/pipeline_root plus artifact identity, so `ep_emulate`,
`decode_cache`, `decode_cache_dir`, `unpack_device`, `device` and — crucially — `inventory`
are advertised in engines.json and never filled from a job.json.

**Repro.**

```
$ python3 -c "import json;d=json.load(open('bin/engines.json'));[print(l,'decode_cache' in c.get('flag_map',{}),'decode_cache_dir' in c.get('flag_map',{})) for l,c in d['lanes'].items()]"
sealed-ep8 False False | streaming True True | local-mps True True
local-cuda-budget True True | bf16-floor True False
```

```
$ python3 bin/invoke_engine.py --job <job with all six keys set> --lane bf16-floor --print-only
engine argv: ... --source native --roles final --reduce-order fp32 --pipeline-root ...
```

All six job keys absent.

**Correction to the finding as filed.** The headline scenario is unreachable *because* of the
second defect: no front end can select `bf16-floor` (`measure_local.py:581` restricts to
local lanes, `measure_cloud.py:1740` to streaming/sealed-ep8), and the only runner that fills
`decode_cache*` is `measure_local.py`, which hardcodes `surface="packed"`. It is a latent
schema gap, and even if reached it is a loud fail-fast raise before any measurement.

**The bigger miss:** `inventory` is also unmapped-from-job, and it is the bf16-floor lane's
entire documented input contract (`--inventory` is in its `required_flags`, and
`stream_score.py:2088-2092` hard-fails `--source native requires --inventory`). So the
bf16-floor cloud path is non-functional end to end, which subsumes the `decode_cache_dir`
gap.

**Patch.**
1. `bin/engines.json`: add `"decode_cache_dir": "--decode-cache-dir"` to the bf16-floor
   flag_map. Safe and behaviour-neutral today.
2. `bin/invoke_engine.py`: forward the advertised job keys into `extra` — and `inventory`
   must be in that list, or the lane still dies at `stream_score.py:2088`. Behaviour-neutral
   today because `measure_cloud._job_document` writes none of them. Pair it with a guard that
   REFUSES unknown top-level job.json keys: once keys are forwarded, a typo becomes a
   silently-defaulted paid run, which is the same silent-drop failure one level up.
3. Do NOT delete these keys from the local lanes' flag_maps — `measure_local.py:924-929`
   depends on `decode_cache`, `decode_cache_dir` and `vram_budget_gb` being present.
4. Add the missing invariant to `--probe-engines`: every flag_map key must be either
   populated by some runner or listed as intentionally planner-only. `--probe-engines` today
   only checks that `required_flags` are DECLARED, so this whole class is invisible to it.

---

## CLI-17 — the engine's output is buffered for hours, so a wedged capture looks healthy

**File:** `bin/invoke_engine.py:185-188`

**Claim.** The engine is launched through `fidelity.common.run`, which hardcodes
`capture_output=True, text=True`, so a multi-hour GPU capture emits nothing until it exits.

**Measured.** Piping through `tee` exactly as `stage_measure.sh:256-258` does: the log stayed
at 0 bytes at t=1s, 2s, 3s, 4s; all 230 bytes landed at exit. `measure_cloud.py:214` tells
the operator that the inspection path IS `jl exec <id> 'tail -50 <fs>/logs/*.log'` — that
command returns an empty file for the whole capture.

**Corrections to the finding as filed.** The OOM claim does not hold: the pinned engines are
not chatty (~0.21 MB per cold run at the real window/layer counts, ~1 MB of controller RSS).
And stdout/stderr interleaving is destroyed by the sequential replay, which detaches every
warning from the window it occurred in — that is the part worth fixing. `text=True` also
means binary noise raises AFTER a successful child, discarding the whole log.

**Patch.**

```python
    sys.stdout.flush(); sys.stderr.flush()
    return subprocess.call(argv, env=env)
```

Verified: live streaming, binary passthrough, flat RSS, correct interleaving, and IDENTICAL
exit-code semantics including signals (SIGKILL -> 247 both ways). Nothing parses this stdout
— the only caller is `stage_measure.sh:256`, which pipes to `tee -a`; `--print-only` returns
before the call. Drop `run` from the `fidelity.common` import if it becomes unused.

**Apply the same change to `bin/invoke_scorer.py:100-102`** — byte-identical pattern, equally
long-running child, and NOT a locked file. (Done in the CLI commit.)

---

## SH-22 — resume on existence, and a swallowed digest

**File:** `bin/stage_measure.sh:250` and `:297`

**Claim.** Line 250 resumes on file EXISTENCE (`[ -f "$RCPT/run-$run/capture-receipt.json" ]`),
not validity, while the watchdog kills captures mid-flight by design; line 297 wraps the
receipt digest in `|| true`, so a failed `sha256sum` leaves an empty `RECEIPT.sha256`.

**Both consequences as filed are REFUTED — read this before deciding it is worth doing.**

- `stream_score.py` writes `capture-receipt.json` EXACTLY ONCE (line 3652), as the last write
  in `main()`, after sealing. It is never written incrementally, so a valid-JSON receipt
  implies the capture ran to completion; the only torn state reachable is invalid JSON.
- The `score` stage runs before `seal` and `kld_report.py:262` calls
  `load_capture_receipt`, which raises on invalid JSON. Under `set -euo pipefail` that aborts
  the stage, the marker is never touched, and `seal` is never reached.
- `RECEIPT.sha256` has ZERO readers: `grep -rn 'RECEIPT.sha256'` over the whole repo returns
  only the line that writes it. The actual byte-binding is `measurement-receipt.json`'s own
  `receipt_sha256`, which `registry_add.verify_seal` recomputes and refuses on.
- The redirect truncates BEFORE `sha256sum` runs, so the outcome is always EMPTY, never
  stale.

**Patch (low priority, hygiene).** Make the resume predicate structural — parse the receipt
and require `schema` and `receipt_sha256` to be non-empty — so a torn receipt costs one cheap
re-run of one cold run instead of a confusing `KeyError` at the `score` stage after the whole
`measure` stage has been paid for. Use a PERMISSIVE predicate: requiring surface-specific
keys risks a false "not captured", which re-runs a cold run at 3.1-8.3 min/window x 25
windows, i.e. 1.3-3.5 GPU-hours per false negative.

For line 297: either drop `|| true` and assert 64 hex, or **delete the line** and note in the
header that the receipt self-seals. Fixing the write without adding a reader leaves a
write-only file that can only mislead — the same class as known defect 4.

---

## SEC-09 — the HF token file exists world-readable for ~20 microseconds

**File:** `bin/measure_cloud.py:1458-1459`

**Claim.** `tmp.write_text(token)` then `os.chmod(tmp, 0o600)`: the file is created at the
umask default (0644) and narrowed afterwards. The comment says "Written to a 0600 file".

**Measured.** A concurrent reader captured the FULL token during the window; `outdir` is
`drwxr-xr-x`, so it is reachable by any local user. Window: 20.5 us.

**Patch.**

```python
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    os.chmod(tmp, 0o600)   # O_CREAT's mode is ignored for an EXISTING inode; this covers a
                           # stale 0644 file left by a run killed between write and unlink.
```

**Keep the trailing chmod.** The reviewer's patch without it silently regresses: I tested a
pre-existing 0644 `.hf_token` and `O_CREAT|O_TRUNC` left the mode at 644.

**The larger exposure is remote,** at `measure_cloud.py:1462-1464`: `mkdir -p .secrets`,
then a separate `jl upload` round trip (documented at ~10-15 s), then `chmod 600` — seconds,
on the rented box. Narrow the DIRECTORY before the upload, which works regardless of what
mode the CLI creates the file with:

```python
    jl.exec(machine_id, "mkdir -p %s/.secrets && chmod 700 %s/.secrets" % (fs_root, fs_root))
```

**Also:** `:1456` says "shredded at teardown", which is an overstatement for the LOCAL copy —
`tmp.unlink()` is an unlink, not a shred.

---

## CLI-16 — a panel descriptor's `scored_positions` is never checked against its own arithmetic

**File:** `bin/fidelity/hfmeta.py:636-658` (`load_panel_descriptor`)

**Claim.** `scored_positions` is read verbatim and never checked against
`contexts * positions_per_context`; missing keys raise `KeyError` instead of the `HFError`
the function otherwise uses.

**Repro.** A descriptor claiming 25 x 2047 = 999999 prints
`shape  25 contexts x 2047 positions = 999999 scored`. `{"panel_ref":"p"}` gives
`KeyError: 'repo_id'` at line 645.

**Corrections.** `scored_positions` feeds NO cost or memory term (every one uses `contexts`
and `positions_per_context`), and `seal_receipt`'s inline registry check REFUSES the
resulting receipt with SCOPE-007 ("covers_full_panel is true but 999999 of 51175 positions
were scored"). So this is defence in depth, not a live path to a published number. Severity
low.

**Patch.** Raise `HFError` naming the missing/unparseable field for KeyError, ValueError and
`json.JSONDecodeError`; and mirror `registry_validate.py:730`'s exemption rather than
asserting unconditionally:

```python
    windowed = bool(raw.get("windowed")) or bool((raw.get("scoring_window") or {}).get("windowed"))
    if not windowed and contexts * positions_per_context != scored_positions:
        raise HFError("panel descriptor says %d x %d = %d, but declares scored_positions %d"
                      % (contexts, positions_per_context, contexts * positions_per_context,
                         scored_positions))
```

Verified safe against every panel in `registry/data/panels.jsonl` (including all three
windowed ones) and against `DEFAULT_PANEL`. No selftest or fixture uses `--panel-descriptor`.

The unlocked half — a duplicate guard at `measure_local.py`'s call site so `measure-local`
stops printing a false identity — is applied in the CLI commit.

---

## CLI-25 — a ranged fetch that the server ignores is read as a header

**File:** `bin/fidelity/hfmeta.py:284-330` (`fetch_file`, `safetensors_header`)

**Claim.** `fetch_file` sends a `Range` header and never verifies the response was 206, then
returns `resp.read()` unbounded; `safetensors_header` swallows `HFError`, `ValueError` and
`struct.error` alike and returns `None`, so a network failure is indistinguishable from "the
file has no such tensor".

**Measured.** Against a Range-ignoring endpoint: `fetch_file(byte_range=(0,7))` returned
41,943,166 bytes instead of 8. Against 404/401/429/503 stubs, `safetensors_header` returned
`None` for all four — a gated-repo token expiry and a transient 503 are byte-identical to
"absent".

**Corrections.** `sniff_surface` does NOT use `safetensors_header` (grep: its only caller is
`measure_cloud.py:621`), so the "silently changes the sniffed surface" claim is wrong — and
that path FAILS CLOSED: a swallowed `None` shrinks `planned`, grows `missing`, and raises a
Refusal before any rental. The OOM ceiling is single-GB (the probe hard-codes
`mtp.safetensors`), not 165 GB. A third leg the finding missed: `fetch_file` catches only
`HTTPError`, while the sibling `_get` at :65-81 also catches `URLError` — so transport
failures escape raw, past every `except HFError` guard.

**Patch.**

```python
    span = None if byte_range is None else byte_range[1] - byte_range[0] + 1
    ...
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if span is None:
            return resp.read()
        payload = resp.read(span)            # the cap: a post-hoc len() check does not
                                             # prevent the multi-GB buffer, the body is
                                             # already resident by then
        if len(payload) != span:
            raise HFError("ranged fetch of %s from %s returned %d of %d bytes (HTTP %s) -- "
                          "endpoint ignored Range"
                          % (path, repo_id, len(payload), span, resp.status))
        return payload
    except urllib.error.URLError as exc:     # mirror _get; every caller already guards HFError
        raise HFError("network error fetching %s from %s: %s" % (path, repo_id, exc.reason))
```

Gate the length check on `byte_range is not None` — `fetch_json` wraps `fetch_file` with no
range and is the backbone of `sniff_surface`; an unqualified 206 assert breaks all of it.

Give `HFError` a `status` attribute so `safetensors_header` can distinguish a genuine 404
(absent -> return None) from 401/403/429/5xx/transport (raise). Then have
`_refuse_incomplete_exl3hf` raise a Refusal that NAMES the transport failure, rather than
either a false completeness accusation or the silently-skipped gate that `except HFError` at
`measure_cloud.py:624` would otherwise produce.

Priority: the 401/429/503-collapsed-to-None leg can fire against production HF today; the
Range cap is latent behind an `HF_ENDPOINT` override.

---

# CONTENDED BY A SECOND SESSION (not the campaign)

`bin/fidelity_dataset.py` and `bin/fidelity/cardmeta.py` were being rewritten by another
session while this review ran (uncommitted `hf-transformers` capture engine, `cache_path`,
`_postcondition`, plus new `engines/tools/hf_capture.py`). These patches were written and TESTED
against that working tree, then reverted so that another agent's in-flight work is not
committed under this review's name.

## CLI-22 / SEC-03 (caller half) — reads should be unauthenticated by default

**File:** `bin/fidelity_dataset.py::_resolve`

The `_get` host-scoping half IS applied (see the CLI-03 commit), so the token can no longer
leave the configured endpoint. What remains is not attaching it at all to a public read.

```python
    token = (dshub.read_token(getattr(args, "token_file", None))
             if getattr(args, "token_file", None) else None)
    ...
    try:
        return dshub.fetch_dataset(ref, cache, token=token, allow_partial=allow_partial,
                                   manifest_only=manifest_only)
    except dshub.HubError as exc:
        if token is None and getattr(exc, "status", None) in (401, 403):
            token = dshub.read_token(None)
            if token:
                emit("  HTTP %s from the Hub; retrying once with the ambient HF_TOKEN"
                     % exc.status)
                return dshub.fetch_dataset(ref, cache, token=token,
                                           allow_partial=allow_partial,
                                           manifest_only=manifest_only)
        raise
```

The retry must NOT be an interactive prompt — these run from shell stages. `dshub._get` now
sets `.status` on `HubError` for exactly this.

Do NOT extend a host allow-list to `registry_client._http_get`: its URLs are built internally
from `HF_ENDPOINT` plus a constant DATASET_ID, and a hard allow-list there breaks every
mirror and enterprise-proxy user.

## CLI-28 — six error paths print a traceback instead of a diagnosis

**File:** `bin/fidelity_dataset.py::main` catch-all

```python
        from fidelity import dsformat as _F
        if isinstance(exc, _F.FormatError):
            return refuse(exc.code, exc.message,
                          "this dataset does not satisfy the v1 format; nothing was written")
        if isinstance(exc, (IOError, OSError)) and not isinstance(exc, dshub.HubError):
            return refuse("unreadable", str(exc), "check the path and its permissions")
```

Verified: `fidelity-dataset validate --receipt /no/such.json` goes from a `FileNotFoundError`
traceback (exit 1) to `REFUSED [unreadable] ... exit 3`.

**Do NOT map every `ValueError` to REFUSED(3).** Exit 3 means a principled refusal — a
publishable outcome. Mapping an internal arithmetic bug to it would make a real defect look
like a considered decision, which is what `dscompare.py:577` was written to avoid.

## CLI-21 — the round-trip axis writes an executable to a fixed temp path

**File:** `bin/fidelity/cardmeta.py:726-734`

`/tmp/fidelity_card_roundtrip.py` is a fixed, predictable path; `open()` follows symlinks;
the file is never removed. **Verified: the RCE framing does not survive** — Linux
`fs.protected_symlinks`/`protected_regular` defaults block the escalation, and the file is
mode 0644 containing only our own static snippet, so the practical failure is an uncaught
`PermissionError` when another user's copy already exists (reproduced with `chmod 444`).

```python
    result = subprocess.run([python, "-", path], input=ROUNDTRIP_SNIPPET,
                            capture_output=True, text=True)
```

Delete the two `script = ...` / `open(script, "w")` lines. Verified byte-identical output to
the file-based path on the real published card (`python - <path>` gives
`sys.argv == ["-", path]`, so the snippet's contract holds). This removes the fixed path, the
symlink write, the leftover file and the TOCTOU window in one edit, with no cleanup code to
get wrong — prefer it to `mkdtemp`, whose `finally: rmtree` can itself leak on a signal.

**Also, adjacent and real:** `os.unlink(path)` for the 0600 `.md` temp file is unguarded, so
it leaks whenever `subprocess.run` or the later `json.loads` raises. Wrap in try/finally.

Do NOT justify this as closing a privilege-escalation hole — that claim is unproven.


---

# PUBLISHED-NUMBER CHANGES (operator decision required)

> **STAT-01 + STAT-17 are CLOSED as of 2026-08-30.** The operator approved the
> reseed. What shipped is not exactly what this section recommended, and the
> difference is recorded rather than quietly absorbed — see the CLOSED note at
> the head of that item below and `docs/PUBLISHED-CORRECTIONS.md` §3.

These are not blocked by a file lock. They are blocked because applying them rewrites
numbers that are already public — in `registry/data/measurements.jsonl`, mirrored to the
`quant-fidelity-registry` HF dataset. The code fix is written out; the reseed is not run.

Deltas below were measured by running the fix and diffing the regenerated rows against the
committed ones, not estimated.

---

## STAT-01 + STAT-17 — the 42 per-domain intervals claim 95% and deliver 78-83%

> **CLOSED 2026-08-30 — reseeded, with two departures from the recommendation.**
>
> 1. **Not bootstrap-t on log(mean).** Recommendation 3's "re-label" option was
>    implemented and measured, and it is not publishable at g=5. On the real
>    `k8-8bpw-stream/clean17/axis2_legal` cell — five ordinary windows, cv 0.47,
>    nothing pathological — it returns an upper endpoint of **0.187 nats around a
>    mean of 0.0103**, eighteen times the estimate, driven by resamples that draw
>    four copies of one window and collapse the studentizing denominator. Three of
>    the 42 cells exceed 10x. Replacing an interval that is too narrow with one
>    that is absurd is not a correction. What shipped is the **analytic** Student-t
>    interval on log(mean) with the delta-method SE, exponentiated
>    (`interval_method: delta_t_log`): same coverage to within Monte-Carlo error,
>    bounded, and with **no resample stream at all** — which retires STAT-17 rather
>    than mitigating it, and also retires the 6.10%-at-B=1000 seed noise this
>    section measured. `DOMAIN_BOOTSTRAP_B` was raised to 20000 anyway, for the
>    `interval="bca"` path that regenerates the old numbers.
> 2. **Recommendation 1's premise was checked and is wrong for coverage.** B does
>    not move it: measured 81.3% at B=1000 and **81.5% at B=20000**. Raising B
>    buys stability against seed choice, not coverage. Both numbers are in
>    `registry/protocol/coverage/domain-interval-coverage.v1.json`.
>
> Coverage was **measured, not asserted**: 4000 replications per cell against a
> lognormal fitted to each cell's own windows, old code and new code through the
> same simulated panels. **81.3% before (77.2–84.5), 92.0% after (89.0–95.0).**
> Not 95% — at five windows nothing is — so every cell now publishes
> `coverage_measured`, which is the actual content of the STAT-01 fix. The panel
> block keeps its BCa endpoints unchanged as recommendation 4 asks, and now states
> its own measured coverage (90.2%). Recommendation 6's assertion is
> `registry/tools/selftest_stat01_reseed.py`.

**Files:** `registry/tools/joint_enrich.py` (`DOMAIN_BOOTSTRAP_B`, `_by_domain`),
`bin/jointstd/stats.py` (`domain_table`).

**What is wrong.** Two coupled defects in the same block of numbers.

*STAT-01.* 54 published intervals are emitted as `ci95_low`/`ci95_high` with
`interval_kind: "bca"` and schema text calling them "the joint standard's interval …
B=5000", but their measured coverage is not 95%. Simulating each published cell at its own
`g` and `B` against a lognormal fitted to that cell's own windows, 4000 reps each:

| block | cells | measured coverage |
|---|---|---|
| panel (`g`=25 / 17, B=5000) | 12 | mean **90.4%** (89.1–91.6) |
| per-domain (`g`=5..7, B=1000) | 42 | mean **81.4%** (78.3–84.7) |

The implementation is not at fault — it reproduces `scipy.stats.bootstrap(method='BCa')`
coverage to within Monte-Carlo error and reproduces brandonmusic's published endpoints
bit-for-bit. Small-`g` BCa simply does not deliver nominal coverage, and it undercovers in
the *bad* direction: truth falls **above** the interval 12–18% of the time and below it
2–4%, so the per-domain intervals systematically understate divergence. A normal
population undercovers too (g=7 → 87.2%), so this is not a skew artifact that a different
population shape would rescue.

*STAT-17.* Inside `domain_table`, every domain is bootstrapped with the **same** seed, so
at equal window counts the resample index streams are identical across domains and these
intervals share their Monte-Carlo error. Measured on the real panel, the replicate-mean
correlation reaches |r| = 0.57 — and it is arbitrary, because it pairs domain A's k-th
window with domain B's k-th, which are unrelated windows.

**Why they are one decision.** Fixing STAT-17 alone moves **79 published `by_domain` CI
endpoints by up to 24.15%** (worst:
`measurement--glm53.k6-6bpw-stream.brandonmusic-final25.clean17` / `axis3_code_agentic` /
`ci95_low`). That movement is *not* the seed — it is the raw MC instability of a BCa
bootstrap at B=1000 over 5–7 windows, i.e. the same instability that produces STAT-01's
undercoverage. Re-rolling the dice with a better seed would replace one wrong interval with
a differently wrong one. Measured seed noise on the worst cell: **11.5%** relative sd of
the low endpoint at B=1000 across 40 seeds.

**Measured blast radius of the seed fix alone** (`seed_registry.py --out` vs committed):

```
headline metric.value changed : 0
top-level uncertainty changed : 0
by_domain CI endpoints changed: 79
worst relative move           : 24.1519%
```

No headline KLD, no `se_clustered`, no `deff`, no domain mean moves. Only the per-domain
interval endpoints.

**Recommended fix, as one reseed.**

1. Raise `DOMAIN_BOOTSTRAP_B` from 1000 to **20000**, not 5000. Measured seed noise on the
   worst cell: 6.10% at B=1000, 3.71% at B=5000, 1.19% at B=20000. It is 42 cells over
   ≤7 values; the cost is trivial.
2. Derive the per-domain seed so the intervals stop sharing MC error, and record it so the
   value stays reproducible. In `bin/jointstd/stats.py::domain_table`, inside
   `if len(ws) >= 2:`:

   ```python
   dseed = seed ^ (int.from_bytes(hashlib.sha256(dom.encode("utf-8")).digest()[:8],
                                  "big") & 0x7FFFFFFF)
   row["bootstrap_seed_derived"] = dseed
   bs = window_block_bootstrap(..., b=b, seed=dseed, backend=backend)
   ```

   (`import hashlib` at the top. This was written, measured and then backed out — the
   comment at that line records why.)
3. Decide what the per-domain block *claims*. Two honest options:
   - **Refuse.** Drop `ci95_low`/`ci95_high` from `by_domain` at `g < 10` and put the
     window count and the measured coverage in `note`, keeping `se_clustered` so a reader
     can build their own. Verified schema-legal as-is: `by_domain`'s `anyOf` does not
     require the endpoints, and `JOINT-006` / `invariants.json:183` are both conditional on
     `ci95_low` being present.
   - **Re-label.** Switch to bootstrap-t on **log(mean)** (studentize the log of the
     resampled mean with the delta-method SE, exponentiate). Measured coverage recovers to
     90.7/92.4/91.3/93.8% on the four k6 domains and 95.5% on the panel, and it is
     non-negative by construction. Use `interval_kind: "t"`, which the schema enum already
     permits. **Do not** use bootstrap-t on the raw mean: it produces a **negative** lower
     endpoint on 5 of the 42 cells (e.g. `fp8-crossstack`/`clean17`/`axis3_code_agentic` =
     −0.009475). A negative lower bound on a KL divergence is a worse publication defect
     than the one being fixed.
4. Whichever is chosen, the panel block should keep its BCa endpoints unchanged — they are
   the joint standard's interop surface and
   `bin/jointstd/fixtures/brandonmusic-known-answer.json` pins four external panels'
   endpoints as a known-answer test. Add the measured coverage as a stated caveat
   (`uncertainty.coverage_measured` plus a sentence in `note`) rather than switching the
   method. Note `uncertainty` and `by_domain.items` are both
   `"additionalProperties": false`, so a new field needs edits to
   `registry/schema/measurement.schema.json` **and** `registry/schema/submission.schema.json`.
5. `bin/joint_standard.py:419` calls the same `domain_table`, so whatever guard or caveat
   lands must land in `bin/jointstd/stats.py` too, or every future contributor running the
   public CLI gets the same uncalibrated intervals.
6. Add the assertion to `bin/selftest_joint_standard.py`: either the committed
   `coverage_measured` values reproduce from a seeded simulation, or no `by_domain` entry
   with `windows < 10` carries `ci95_low`.

**Do not** repeat the framing "readers will accept overlaps that are not really overlaps".
Undercoverage makes these intervals too narrow and biased low, so the errors are false
**separations** (measured 4.2–5.1% at domain level against the ~0.5–1% a reader infers) and
systematic **understatement** of per-domain divergence.

---

# SECOND-PASS ADDITIONS (independent review, 2026-08-30)

A second reviewer re-tested this file's claims and hunted the areas the first pass
under-covered: concurrency between the live campaign and the tooling, degraded-network
behaviour, the rented-instance lifecycle, and the published artifacts. The first pass's
CLI-01 analysis was re-checked and **stands** — in particular its refusal to make
`JLApi.get()` strict, which is correct: on a healthy API a 404 for a destroyed instance
and a 404 from an outage are the same `JLError`, so `get() -> None` is the load-bearing
"successfully destroyed" signal and making it raise would disarm the leak detector on
every run. `list_instances()` propagating `JLError` was verified, so the proposed
`_confirm_gone` helper is sound.

The findings below are new. All four are in `bin/measure_cloud.py`, which the campaign
owns; the fifth is a published artifact.

The root cause shared by REAP-1 and REAP-2 — `JLApi.list_instances()` reporting an empty
account when it could not read the answer — **has been fixed** in `bin/fidelity/jlapi.py`,
which is not locked, with `bin/selftest_jlapi.py` (T11) covering it. The items below are
what remains inside the locked file.

---

## REAP-1 — the reaper reports success when every destroy failed

**File:** `bin/measure_cloud.py:517-524` (`reaper_sweep`, the destroy loop)

**Claim.** The reaper is the last-resort backstop, meant to be run from cron or a
systemd timer on a machine that may never have seen the job. Its destroy loop is:

```python
    for mid, why in sorted(targets.items()):
        con.say("reaper: destroying %s (%s)" % (mid, why))
        if not dry:
            try:
                jl.destroy(mid)
            except JLError as exc:
                con.err("reaper could not destroy %s: %s" % (mid, redact(str(exc))))
    return EXIT_OK
```

Three problems, all in five lines:

1. **`EXIT_OK` regardless.** Every destroy can fail and the process still exits 0. Any
   wrapper, cron mailer or monitor that keys on exit status sees a healthy backstop while
   an 8×H200 keeps billing.
2. **One attempt, no retry.** `Teardown._destroy_instance` retries five times; the
   backstop tries once and gives up.
3. **No confirmation.** `jl.destroy` returning without raising is not proof the instance
   is gone — the same gap CLI-01 documents for the primary teardown, here in the thing
   that is supposed to catch CLI-01.

**Patch.**

```python
    failed = []
    for mid, why in sorted(targets.items()):
        con.say("reaper: destroying %s (%s)" % (mid, why))
        if dry:
            continue
        gone = False
        for attempt in (1, 2, 3):
            try:
                jl.destroy(mid)
            except JLError as exc:
                con.err("reaper destroy %s attempt %d: %s" % (mid, attempt, redact(str(exc))))
            try:                       # `jl list` propagates JLError; `jl get` does not
                gone = mid not in {i.machine_id for i in jl.list_instances()}
            except JLError as exc:
                con.err("reaper cannot confirm %s: %s" % (mid, redact(str(exc))))
                gone = False
            if gone:
                break
            time.sleep(5)
        if not gone:
            failed.append(mid)
    if failed:
        con.err("reaper could NOT confirm destruction of %s -- these are still billing"
                % ", ".join(str(m) for m in failed))
        return EXIT_LEAK
    return EXIT_OK
```

`EXIT_LEAK` (90) already exists and already means exactly this.

---

## REAP-2 — the name-encoded deadline overrides a live lease, with no plausibility bound

**File:** `bin/measure_cloud.py:440-453` (`parse_deadline_name`) and `478-487`

**Claim.** `parse_deadline_name` accepts ANY base36-parsable tail after the last `-x`
and returns it as an epoch, with no sanity bound. A `fidcloud-`-prefixed instance whose
name was not produced by `deadline_name()` therefore resolves to a deadline in 1970 and
is destroyed on the next sweep. Reproduced: `parse_deadline_name("fidcloud-x9zz")`
returns **12959** (1970-01-01), and `reaper_sweep` printed
`reaper: destroying 999001 (name deadline 12959 passed)` for an instance whose own lease
said it had 24 hours left. The lease is not consulted: `targets.setdefault` only stops
the name path from OVERWRITING a lease-derived reason, never from ADDING a target.

Names this tool creates are safe today — `fidcloud-<8 hex>-x<base36 epoch>`, and hex
contains no `-` — so this is a trap for anything else wearing the prefix, including a
hand-named box and a future naming scheme. The cost of the trap is an unrecoverable
destroyed instance; the cost of the guard is four lines.

**Patch.**

```python
#: A parsed deadline outside this window is not a deadline, it is a coincidence.
#: `int("tra", 36)` is 38566 -- epoch 1970 -- and every "expired" test then passes.
_DEADLINE_FLOOR = 1_750_000_000        # 2025-06; before this project existed
_DEADLINE_CEIL_SECONDS = 90 * 86400    # no run this tool plans plausibly outlives

def parse_deadline_name(name, now=None):
    now = time.time() if now is None else now
    for sep, base in (("-x", 36), ("-exp", 10)):
        head, found, tail = name.rpartition(sep)
        if found and tail:
            try:
                value = int(tail, base)
            except ValueError:
                continue
            if _DEADLINE_FLOOR <= value <= now + _DEADLINE_CEIL_SECONDS:
                return value
            return None                # parsed, implausible: NOT a deadline
    return None
```

And in the sweep, do not let a name-derived deadline contradict a live lease:

```python
        leased = {int(l["machine_id"]): float(l.get("deadline_epoch", 0))
                  for l in _read_leases() if l.get("machine_id")}
        ...
            if deadline is not None and deadline < now:
                if leased.get(inst.machine_id, 0) > now:
                    con.warn("reaper: %s's NAME says expired but its lease says %s "
                             "remain; not destroying, and the mismatch is the bug"
                             % (inst.machine_id,
                                human_duration(leased[inst.machine_id] - now)))
                    continue
                targets.setdefault(inst.machine_id, "name deadline %d passed" % deadline)
```

---

## REAP-3 — `reaper --sweep --dry-run` under-reports what the real sweep does

**File:** `bin/measure_cloud.py:496-513`

**Claim.** The whole phantom-lease retirement block is guarded by `if not dry:`, so the
dry run never says which leases it would delete. An operator who runs the documented
preview sees `reaper: nothing expired` and then the real run silently removes leases.
A preview that omits a destructive action is worse than no preview.

**Patch.** Hoist the retirement scan out of `if not dry:` and guard only the `unlink`:

```python
        if alive is not None:
            for path in sorted(LEASE_DIR.glob("*.json")) if LEASE_DIR.is_dir() else []:
                ...
                if mid and int(mid) not in alive and int(mid) not in targets:
                    con.say("reaper: %sretiring lease %s (machine %s is gone)"
                            % ("would be " if dry else "", path.name, mid))
                    if not dry:
                        path.unlink(missing_ok=True)
```

---

## REAP-4 — a `jl list` this client cannot read is no longer "an empty account" (fixed in jlapi)

**File:** `bin/fidelity/jlapi.py` — **FIXED, not deferred.** Recorded here because the
consequences all land in `measure_cloud.py`.

`JL._call` returns `{}` for an empty body on a zero exit, and returns a parsed object
unchanged on a NON-zero exit as long as the JSON carries no `error` key. The old
`data.get("instances", [])` fallback turned all of that into "the account is empty", and
four call sites spend or leak money on that answer:

| site | what an empty list means there |
|---|---|
| `reaper_sweep:499` | every lease is a phantom; retire them all, never look again |
| adopt loop `:1329` | no instance for this job — create a second one (double-spend) |
| `_find_by_name:571` | give up recovering the id of an instance that is already billing |
| `reaper_sweep:480-487` | the L3 name sweep silently degrades to leases only |

Reproduced against a stub `jl`, three ways: an empty body, `{"data": [...]}` (a vendor
key rename), and `exit 2` with `{"detail": "authentication failed"}` — each retired a
live lease and exited 0. The live `jl 0.2.17 list --json` answers with a top-level JSON
array, so the `{"instances": [...]}` branch had never run.

`list_instances()` now raises rather than reporting an empty account, and a genuinely
empty `[]` still reads as empty. `bin/selftest_jlapi.py` (T11, wired into
`bin/selftest_all.sh`) covers all seven cases with no network and no account. **Nothing
in `measure_cloud.py` needs to change for this fix to take effect** — but REAP-1..3 are
still open inside it.

---

## CC-01 (published) — the K8 model card's power arithmetic is wrong on both numbers

**Artifact:** `https://huggingface.co/malaiwah/GLM-5.3-Flash-TR3-8bpw` → `README.md`,
lines 324-325. **Not corrected here: publishing to the Hub needs the operator.**

The published card says:

> per-window KLD scatter has sd 1.73e-3 against a K6-vs-K8 effect of 1.22e-3

Both values come from `engines/K8-ANOMALY.json`, where they are the per-window **delta** sd
(`per_window_delta_stdev` = 0.0017334539428769534) and the pooled delta
(`pooled_delta_k8_minus_k6` = -0.0012176728196882456) over an **eleven-window subset**.
They are correctly labelled in that document and were mis-scoped everywhere else.

Recomputed from the committed per-window series (`registry/protocol/per-window/`, n=25):

| quantity | published | actual |
|---|---|---|
| per-window KLD sd, K6 sealed | 1.73e-3 | **7.198e-3** |
| per-window KLD sd, K8 streaming | — | **6.935e-3** |
| paired per-window K6−K8 delta sd | (mislabelled as the above) | **2.027e-3** |
| K6-vs-K8 effect | 1.22e-3 | **1.331e-3** |

The card's *conclusion* — a single window has no power to compare quants — survives, and
is in fact stronger. Only the numbers are wrong.

The repo copy (`docs/cards/GLM-5.3-Flash-TR3-8bpw.README.md`) and the five code/doc sites
that repeated it are corrected, and `bin/check_doc_numbers.py` section 12 now re-derives
all three values from the per-window series and fails if any document quotes the retracted
pair as a live claim. **The Hub card still carries the wrong sentence.** Replacement text
is the corrected paragraph in the repo card; the 6bpw card does not carry this sentence.

---

# DEPENDENCY-AUDIT ADDITIONS (2026-08-31)

From the reinvented-wheel audit in [`DEPENDENCIES.md`](DEPENDENCIES.md). Every finding
below lands in a **provider backend or the shared SSH transport**, which a live
measurement campaign owns (a run was on RunPod while this was written), so none was
applied. Each is a docstring or a three-flag change; none needs a redesign.

Owners for this batch:

| File | Owner while the audit ran |
|---|---|
| `bin/fidelity/runpodapi.py` | live RunPod measurement |
| `bin/fidelity/vastapi.py` | live RunPod measurement (shared controller) |
| `bin/fidelity/lambdaapi.py` | live RunPod measurement (shared controller) |
| `bin/fidelity/sshbase.py` | live RunPod measurement |
| `bin/fidelity/bench.py`, `bin/selftest_provider_portability.py` | a second session (uncommitted at audit time) |

## DEP-01 — `vastapi._req` retries 429 and nothing else, which is the incident it was written to stop

**Anchor:** `bin/fidelity/vastapi.py`, `def _req`, the `except urllib.error.HTTPError` arm
(`if exc.code == 429 and _tries > 1`).

The docstring records the original incident correctly: Vast rate-limits to ~1 req/s, the
banded catalogue search tripped it *"INSIDE the run, after the lease was written, so a rate
limit read as a failed run."* The fix paces at `_MIN_INTERVAL = 1.1` and retries 429.

**The gap:** `status_forcelist` is effectively `[429]`. A 502/503/504 — and every
`ConnectTimeout`/`ReadTimeout`/connection reset — falls through to `except Exception` and
raises hard, mid-run, after the lease is written. That is the *same shape* as the failure
the 429 handling exists to prevent, on a marketplace host the journal already describes as
flaky.

Two halves are genuinely custom and must survive any rewrite: `retry_after` arrives **in the
JSON body**, not the `Retry-After` header (`urllib3`'s `respect_retry_after_header` would
not find it), and the 1.1 s client-side pacing is not something `requests` provides.

**Patch (stdlib, no dependency):** widen the retried set and add the network-error arm.

```python
_RETRY_STATUS = (429, 500, 502, 503, 504)
...
    except urllib.error.HTTPError as exc:
        payload = exc.read()[:300].decode("utf-8", "replace")
        if exc.code in self._RETRY_STATUS and _tries > 1:
            wait = 2.0
            if exc.code == 429:
                try:
                    wait = max(1.0, float(json.loads(payload).get("retry_after") or 1)) + 1.0
                except Exception:                       # noqa: BLE001
                    pass
            else:
                wait = min(30.0, 2.0 ** (5 - _tries))   # 5xx: back off, body carries no hint
            time.sleep(wait)
            return self._req(method, path, body, timeout=timeout, _tries=_tries - 1)
        raise VastError(...)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if _tries > 1:
            time.sleep(min(30.0, 2.0 ** (5 - _tries)))
            return self._req(method, path, body, timeout=timeout, _tries=_tries - 1)
        raise VastError("Vast request failed: %s" % redact(str(exc)))
```

**Also note, not urgent:** `Vast._last_call` is a mutable **class** attribute assigned via
`Vast._last_call = ...`, so the pacing state is process-global and not thread-safe. Fine
for the single-threaded controller; a trap if the catalogue search is ever parallelised.

**Test:** none exists. `selftest_provider_portability.py` reaches `Vast` only to read
`separable_storage`; nothing in the repo patches `urllib.request.urlopen`. A rung that
monkeypatches `urlopen` to raise `HTTPError(502)` twice then succeed would fail without the
patch above and pass with it — worth adding **with** the fix, since its absence is why the
429 case had to be found on a live lease.

## DEP-02 — the Cloudflare User-Agent workaround should say it was a `urllib` tax

**Anchor:** `bin/fidelity/runpodapi.py`, `def _gql`, the `"User-Agent"` header.

The inline comment is already excellent and should not change. What is missing is at the
**module docstring** level: this file's transport choice has a measured cost, and the next
person deciding "urllib or requests" should see it without reading commit archaeology.

Add to the docstring, after the "There is no CLI" paragraph:

```
WHAT urllib COSTS HERE, so the trade is visible.  Cloudflare fronts
api.runpod.io and answers urllib's DEFAULT User-Agent with HTTP 403 "error
code: 1010".  requests sends its own UA and would never have hit it; this was
found by smoking it on a $1.59/h pod for six minutes, and the header at _gql is
not optional.  Two further costs are unpaid rather than fixed: there is NO
retry (a Cloudflare 502 raises hard, mid-run, after the pod is billing), and
there is no connection pooling -- gpus() issues 100+ sequential urlopen calls,
each a fresh TCP+TLS handshake, and _endpoint() polls every 10 s for up to
900 s.  The surface used is one verb against one endpoint, so urllib remains
defensible; it is a trade, not a free choice.
```

**Cargo-cult note worth a one-liner in each:** the same `"User-Agent":
"quant-fidelity-suite/0.1"` was copied into `vastapi.py` and `lambdaapi.py`, neither of
which is behind Cloudflare. Harmless, but it reads as required when it is not.

## DEP-03 — `sshbase` opens a fresh handshake per exec and per scp; ControlMaster is three flags

**Anchor:** `bin/fidelity/sshbase.py`, `def _ssh_opts`.

`JOURNAL.md` already carries *"ControlMaster from minute one"* as a lesson, and records it
being configured by hand in `~/.ssh/config` for the JarvisLabs box. It never reached the
shared transport, where it would serve RunPod, Vast **and** Lambda. Combined with the
per-file `scp` delta uploader, that is one full SSH handshake per file.

**Patch:**

```python
def _ssh_opts(self) -> List[str]:
    # ControlMaster: every exec and every scp otherwise pays a full handshake,
    # and the delta uploader sends one file per scp.  JOURNAL: "ControlMaster
    # from minute one".  ControlPath lives in a per-process temp dir so two
    # concurrent controllers cannot collide on one socket.
    return ["-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto",
            "-o", "ControlPath=%s/cm-%%C" % self._control_dir,
            "-o", "ControlPersist=120"]
```

with `self._control_dir = tempfile.mkdtemp(prefix="fidelity-ssh-")` in `__init__` and an
`atexit`/teardown `shutil.rmtree`. **Verify before adopting** that the socket path stays
under the 104-byte `sun_path` limit on macOS — `%C` is a hash, which is why it is used here
rather than `%h-%p-%r`.

**Also undocumented, same file:** `StrictHostKeyChecking=no` +
`UserKnownHostsFile=/dev/null` on the channel that carries the HF token. TOFU on ephemeral
marketplace instances is genuinely hard and the endpoint is discovered over an authenticated
TLS API, so the choice is defensible — but in a file this heavily commented the silence is
conspicuous. One sentence naming the threat model would settle it.

## DEP-04 — `sshbase.run_status` uses the naive `pgrep -f` this project has already paid for twice

**Anchor:** `bin/fidelity/sshbase.py`, `def run_status`, the
`elif pgrep -f %s >/dev/null 2>&1; then echo RUNNING` branch.

`JOURNAL.md` lesson 44 records this exact bug: *"the probe reported a stage that has never
existed as running, because `pgrep -f` matches full command lines and the probe's own shell
carries the pattern in its own."* `measure_cloud._stage_is_alive` carries the fix and
explains it at length — `[s]tage_measure` rather than `stage_measure`, verified against a
live instance.

`sshbase.run_status` did not get it. The pattern is `shlex.quote(run_id)`, and the run id
appears in the probe's own command line twice over (the `%s/exit_code` path is built from
it), inside `sh -lc '<script>'`.

**Effect if it triggers:** the `GONE` branch becomes unreachable, so a stage killed without
writing `exit_code` — OOM-kill, `kill -9`, host reboot — reads `RUNNING` until
`--max-runtime`. That burns a billing instance, which is precisely the cost lesson 44
records. Note this is the RunPod/Vast/Lambda path; JarvisLabs uses `jl` and is unaffected.

**Patch** (bracket-class the first character, as `measure_cloud` does):

```python
# [r]un-id and NOT run-id: pgrep -f matches full command lines and this probe's
# own shell carries the id in ITS command line (twice -- the exit_code path is
# built from it).  JOURNAL lesson 36 / 44, and the same fix measure_cloud.py's
# _stage_is_alive carries.
_pat = shlex.quote("[%s]%s" % (run_id[0], run_id[1:]))
```

**EVIDENCE IS PARTIAL — verify on Linux before treating this as confirmed.** On macOS/BSD
`pgrep` the probe did **not** self-match in testing (`sh -lc 'pgrep -f r_9999'` → rc=1,
while `ps` confirmed the pattern was in the parent's args). The remote is Linux/procps,
where this project's own live verification says it *does*. The asymmetry — fixed in
`measure_cloud.py`, not in `sshbase.py`, same author, same lesson — is the finding
regardless of which `pgrep` is in play.

**Test:** none. No selftest imports `sshbase`, and the two bugs its docstring exists to
memorialize (the detached-job exit code, the pid-based liveness) have no offline regression
test either — they were verified live on rented hardware only. `selftest_teardown.py` tests
`measure_cloud`'s *other* pgrep probe. A rung driving `run_status` against a fake
`exec_stdout` would cover all three cheaply.

## DEP-05 — `lambdaapi.create()` fetches `/instance-types` twice in one call

**Anchor:** `bin/fidelity/lambdaapi.py`, `def create`, the two `_req("GET", "/instance-types")`
call sites.

Two full TLS handshakes for a catalogue already held in a local variable, on the critical
path before launch. Not a correctness bug and not a library problem — hoist the first
result. Mentioned here only so it is not rediscovered.

**Related, and worth a comment rather than a fix:** VRAM is parsed out of Lambda's free-text
`gpu_description` by `.split("(")[-1].split("GB")[0]`. The vendor publishes no structured
field, so no library helps; but it is fragile and untested, and this file's one historical
bug was also a data-shape bug (`_KNOWN_DISK_GB` guessed 200 GB for a box whose `df -h /`
said 1.4T).

## PANEL-D6 — a capture records its tokenizer as a filesystem PATH, so two
## captures of one panel on two mount roots are refused as different tokenizers

**Anchor:** `engines/tools/hf_capture.py`, the panel block's tokenizer identity — it takes
`--model`, which is the local checkpoint tree, rather than `--repository`, which is the
published repo id the same invocation already passes.

**Found by** the container transport's acceptance test, 2026-08-31: two captures of the
*same* checkpoint at the *same* revision over the *same* panel on the *same* GPU, whose
tensors are bit-identical (`capture_content_digest`
`b42ffe8f1d1dfcfdd78452339cdcd913c8be9ceae13f88f6348f19b43a960549` on both). Comparing them
is refused:

```
REFUSED [panel_mismatch]: gate panel: the two captures declare different tokenizers
(PANEL-D6); the token id digest cannot see this because it hashes integers.
Differing field(s): id '/home/ubuntu/armA/fidelity/models/target'
                 vs '/workspace/fidelity/models/target'
```

The gate is right to exist — a token id digest hashes integers and cannot see a tokenizer
change — and it is right to have no override (PANEL-D3). What is wrong is the *value* it
compares: a path on a machine that will not exist, where a published repo id was available
in the same argv. On the SSH path every arm shares a root (`/home/jl_fs/...`), so this could
not surface; a container has a different mount root by construction, and so does any second
machine.

**Why it is deferred rather than fixed here:** the tokenizer id is a field of a published
manifest. Changing what it holds changes the bytes of every future capture and makes new
captures compare unequal to already-published ones on that field — a comparability decision,
not a bug fix, and it belongs to whoever owns the dataset format.

**Suggested shape:** record `repository` (already passed as `--repository`) as the identity
and keep the local path as a separate, non-compared `source_path`; treat a legacy
path-valued id as "unknown" rather than as a mismatch, so old and new captures compare on
what they actually share.

**Test:** a rung that captures the same fixture twice under two different `--out`/model-tree
roots and asserts `compare --self-compare` succeeds. It fails today.

**Caller-side half LANDED, 2026-08-31 (GH200 qualification):** `bin/stage_measure.sh` now
passes `--weights-repository "$REPO"` in both capture stages, so `tokenizer_id =
args.tokenizer_id or args.weights_repository or args.model` resolves to the published repo
id instead of the local tree. That is the same value this entry says should be recorded,
placed at the caller so the dataset FORMAT is untouched. It cannot make anything compare
worse — today's cloud captures record a path that already compares unequal to every other
capture, including to a second run of themselves — and it removes an on-instance absolute
path from four other published fields the same expression feeds (`runtime.weights.
repository`, the card's provenance line, and both `weights=` blocks). Regression:
`selftest_root_capture.py`, "names the HF repo as the weights repository, not the local
path". **The engine-side half of this entry is still open**: reading a legacy path-valued
`id` as *unknown* rather than as a mismatch, so already-published captures stay comparable.

## ROOT-1 — a root capture's sealed dataset is never brought home, and dies with the box

**Anchor:** `bin/measure_cloud.py`, `def _pull_receipts`; `fidelity/stages.py`,
`ROOT_STAGES`.

`--role root` writes the sealed dataset to `$FS/dataset`, `verify` recomputes its digest
chain there, and then the instance is destroyed. Teardown pulls **`$FS/receipts` and
nothing else**. There is no `$FS/dataset` pull anywhere in the controller.

On JarvisLabs that is survivable by accident: `$FS` is a separable filesystem, so
`--keep-fs` leaves the dataset behind for a later box to publish from. On the other three
backends `separable_storage = False` — RunPod's volume, Vast's disk and Lambda's local NVMe
all die with the instance — so **the artifact the whole rental existed to produce is
deleted at teardown**, and the run reports success. Confirmed by reading; worked around
during the GH200 qualification by a sidecar that polled for
`$FS/dataset/fidelity-dataset.json` and `scp`'d the tree down before the controller
finished.

This is the one defect in this file that destroys the *product* rather than costing money.

**Suggested shape:** a `_pull_dataset` teardown step, ordered before `_pull_receipts` and
gated on `role == "root"`, tarring `$FS/dataset` exactly as receipts are tarred (one
transfer, not a per-file walk — that lesson is already paid for). Its timeout must scale
with the dataset: a Fruit root is 385 MB, a GLM-5.3-Flash hidden-form root is ~30 GB. If a
run's dataset is too large to pull, the honest answer is to publish it from the instance
before teardown, not to leave it there.

**Test:** the container/stage battery already builds a fake `$FS`; a rung that runs the
teardown against one containing `dataset/` and asserts the tree arrives under `--out`
fails today.

## ROOT-2 — a root capture is fit-checked against GLM-5.3-Flash, whatever it is capturing

**Anchor:** the `fit` block printed by `bin/measure-cloud --dry-run`; `base decoded BF16
642.70 GB` and `required VRAM 63 GB/GPU` for **any** target.

`bin/measure-cloud --role root --model malaiwah/GLM-5.2-SIQ-Fruit-bf16` — a 10.10 GB
checkpoint whose capture peaked well inside a single 40 GB card — plans against
GLM-5.3-Flash's 642.70 GB census and refuses every instance type under 63 GB:

```
gpu_1x_a100_sxm4 us-east-1   43 GB  $1.99   free=1   too small (43 < 63 GB)
REFUSE: no available instance fits this lane
        lane streaming needs >=63 GB/GPU x1
```

The disk line is computed from the real target (`10.10 GB artifact`), so the census is
reachable; the VRAM arithmetic simply is not wired to it for `role == "root"`. Two costs,
both paid during the GH200 qualification: the cheap x86 control arm had to be rented from
another provider because no Lambda type both fit the phantom 63 GB and had capacity, and a
run is refused outright whenever the only free cards are ones that would have worked.

It also mis-prices every root: the cost estimate quotes `25 windows @ ~2.82 min` and a
31.73 GB panel fetch for a job that has 16 windows, no `fetch_panel` stage at all (see
`ROOT_STAGES`), and a capture that finished in minutes.

**Suggested shape:** for `role == "root"`, size the fit from the target's own census — the
hidden-form capture's working set is the resident non-layer parameters plus one streamed
layer under `--schedule layer-outer`, not the whole decoded checkpoint — and take the window
count from the panel directory that was passed, which the planner has already read to
extract `panel_id`.

**Test:** a fit rung asserting that a root plan for a 10 GB checkpoint does not demand 63 GB
of VRAM, and that its window count equals the panel's. Both fail today.
