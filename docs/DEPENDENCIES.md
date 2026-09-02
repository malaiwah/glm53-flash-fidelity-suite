# Dependency audit — what we hand-wrote, and whether we should have

**Written:** 2026-08-31, prompted by a single question: *we told an agent to hand-write a
progress meter instead of using `tqdm` — where else are we reinventing standard libraries,
and why?*

The stated reason at the time was "do not add a dependency, the bundle is uploaded
file-by-file." **That reason was wrong**, and this document exists because it was wrong.
`bin/bootstrap_measure.sh` pip-installs torch, transformers, accelerate and `rich` onto the
rented instance; one more wheel is close to free. The audit below re-derives every
hand-rolled component from evidence instead of from that slogan.

Some reinvention here is load-bearing and must not be undone. The job of this file is to
tell the two apart, **one line at a time, so a reviewer can disagree with a line rather than
with the whole document.**

## Reasons that count, and reasons that do not

A KEEP is only valid if it survives all four questions:

1. Why does it exist? (docstring + `git log -S`, not assumption)
2. Would the stdlib or a well-known package do it **concretely**, not in principle?
3. What would adopting it cost? A new dependency on the measurement instance is a new thing
   that can fail at hour zero of a paid run, and a new supply-chain surface for a project
   whose entire product is *trustworthy numbers*.
4. Verdict, in one or two sentences.

These are **not** valid reasons, and any KEEP resting on one of them is marked as such:

- "we already wrote it"
- "no dependencies" as a slogan, where the same process already installs a stack
- avoiding a dependency that is **already transitively present**

## The governing constraint, stated accurately

`AGENTS.md` says `bin/` and `registry/` must run on **stock python3.9 with no installs**.
That is real and it is verified: the system interpreter here is Python 3.9.6, and
`registry/` validates clean under it.

But the constraint is narrower than it is usually invoked, and the distinction decides
several rows below:

| tree | runs where | third-party available? |
|---|---|---|
| `registry/tools/` | contributor's stock interpreter | **no** — this is the point |
| `bin/` (controller: `measure_cloud.py`, the provider backends, `sshbase.py`) | the operator's laptop, stock `python3` | **no by policy**; the admitted RunPod path is stdlib-only |
| `bin/fidelity/` files listed in `bin/BUNDLE.txt` | *both* laptop and rented instance | laptop is the binding constraint |
| `engines/tools/` engines | the rented instance **after** `bootstrap_measure.sh` | **yes** — torch, transformers, `rich`, and `tqdm` are all installed |

Two facts complicate the policy and are recorded here rather than argued away:

- Historical `jlapi.py` can call an externally installed JarvisLabs CLI, but
  JarvisLabs is no longer an admitted paid provider. The current RunPod
  controller uses stock-stdlib HTTP and OpenSSH subprocesses.
- Provider backends are controller-side only and are not uploaded in
  `bin/BUNDLE.txt`. Their stdlib-only style is still load-bearing: planning,
  leasing, reaping, retrieval and teardown run on the operator's stock Python.

---

## Verdict table

| # | Component | Alternative | Verdict |
|---|---|---|---|
| 1 | `engines/tools/progress.py` | `tqdm`, `rich.progress` | **KEEP — for a different reason than the file gave** |
| 2 | `registry/tools/_minischema.py` | `jsonschema` | **KEEP** |
| 3 | `bin/fidelity/jlapi.py` (CLI wrapper) | JarvisLabs REST | **KEEP** |
| 4 | `bin/fidelity/runpodapi.py` (`urllib`) | `requests`/`httpx` | **KEEP-BUT-DOCUMENT** — urllib demonstrably cost one live incident |
| 5 | `bin/fidelity/vastapi.py` 429 backoff | `urllib3.Retry` | **KEEP-BUT-DOCUMENT** — the hand-rolled retry is incomplete in a way that matters |
| 6 | `bin/fidelity/lambdaapi.py` Basic auth | `requests(auth=…)` | **KEEP-BUT-DOCUMENT** — textbook reinvention, zero observed cost |
| 7 | `bin/fidelity/sshbase.py` | `paramiko`/`fabric` | **KEEP** — strict per-attempt `known_hosts` plus authenticated provider-log ED25519 evidence |
| 8 | `bin/fidelity/common.py` `Console` | `logging`, `rich` | **KEEP** |
| 9 | `bin/fidelity_stats.py` statistics | `scipy.stats` | **KEEP** — the strongest keep in the repo |
| 10 | `canonical_json` duplicated per tree | one shared copy | **KEEP** |
| 11 | `sha256_file` × 15 | `hashlib.file_digest` | **ADOPT (proposed)** — genuine duplication, no rationale anywhere |
| 12 | ISO-8601 hand-spelled × 13, incl. deprecated `utcnow()` | `common.utcnow()` / `datetime.now(timezone.utc)` | **ADOPT (proposed)** |
| 13 | `bin/kld_preview.py` `sum()/len()` | `statistics.fmean` | **ADOPT (proposed)** — measurement path, so proposed not applied |
| 14 | The selftest harness, re-declared ~24× | `unittest` | **KEEP the design, ADOPT the shared module (proposed)** |
| 15 | Atomic-write helper × 8 | nothing in stdlib | **KEEP** |
| 16 | Two checksum-file parsers | one parser | **KEEP — deliberately two** |
| 17 | Two retry curves in `engines/tools/fetch_*.py` | one curve | **KEEP, reconcile the constants (proposed)** |
| 18 | `argparse`, `pathlib`, `dataclasses`, `enum`, tables, colors | — | **not reinvented; no finding** |

---

## 1. `engines/tools/progress.py` — the meter that started this

**KEEP.** The conclusion is unchanged; **the reason in the file was false and has been
corrected.**

### What the file claimed

> It is also not `tqdm` because `tqdm` is not in the bundle. `bin/BUNDLE.txt` is an
> explicit, auditable upload list; adding a third-party package to a rented instance to
> print a percentage is not a trade this project makes.

### Why that is false

`bin/bootstrap_measure.sh` installs `transformers==5.16.1` and `huggingface_hub` on the
instance. Checked against the installed metadata, not from memory:

```
transformers 5.16.1     HARD: tqdm>=4.60
huggingface_hub 1.28.0  HARD: tqdm>=4.42.1
```

`tqdm` 4.70.0 is present in this repo's own `.venv` with **no `REQUESTED` marker** — i.e.
nothing asked for it; it arrived as a hard transitive dependency. There is no such thing as
"adding tqdm to the instance": it is already there, unconditionally, before the measure
stage starts. **The incremental supply-chain surface of using it is exactly zero.**
`rich` is installed too, explicitly, on the same `pip install` line.

So the dependency argument is dead. It was the "avoiding a dependency that is already
transitively present" fallacy, and it should not have been written.

### Why the verdict is still KEEP

The real reason is one the file understated, and it holds up under test. Every stage runs
`nohup … > logs/stage-<name>.log`, so the process needing a progress bar is exactly the one
whose stdout is never a TTY. **`tqdm` has no newline mode.** From `tqdm/std.py`:

- line 457: `fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))` — the `\r` is
  **unconditional**.
- line 979: `if disable is None and hasattr(file, "isatty") and not file.isatty():` — the
  only `isatty` branch in the class, and it decides *whether to disable*, never *how to
  render*.

Measured, not assumed. Writing `tqdm` to a file:

```
$ python tq.py > out.log          # default
$ wc -l out.log
       2 out.log                  # 40 iterations, ONE line, 7 embedded \r
^Mfill L003/g0:   0%|          | 0/40 …^Mfill L003/g0:  20%|██  …^Mfill L003/g0:  40%|████ …
```

Configured the way "just configure it" implies — `ascii=True, mininterval=1.0` — fixes the
block characters and the update count and **does not fix the `\r`**:

```
^Mfill:   0%|          | 0/40 …^Mfill:  45%|####5     | 18/40 …^Mfill: 100%|##########| 40/40 …
```

And `disable=None`, tqdm's own non-TTY affordance, produces **complete silence** — which is
the two-to-three-hour void the meter was written to end.

So the honest menu with `tqdm` is: a megabyte-long single line, or nothing. Getting
newline-terminated throttled lines requires passing a custom `file=` object that rewrites
`\r` to `\n` — i.e. writing code anyway, and then owning a shim wrapped around a dependency
instead of owning sixty readable lines.

Two further properties are contract, not taste:

- **The output is machine-read.** `bin/measure_cloud.py:_progress_counter` parses
  `progress: <label> <n>/<total>` out of the log tail `_await_stage` already fetches, so a
  hung run can be *named* as hung — `_stage_is_alive`'s `pgrep` says "alive" for a hung
  process forever. The `progress:` prefix and the `n/total` token are an interface.
- **`bin/` cannot import it.** `measure_cloud.py` duplicates the seven-character prefix
  rather than importing `progress.py`, because the controller runs on stock python3.9 with
  no torch and no `engines/tools` on `sys.path`. `bin/selftest_progress.py` asserts the two
  agree.

**Verdict: KEEP.** Not because tqdm is a dependency — it is already installed — but because
tqdm cannot produce newline-terminated throttled output at all, and this meter's output is
a parsed contract. *The docstring and `bin/selftest_progress.py` have been corrected to say
this; they previously said the false thing.*

**Reviewer, disagree here if:** you think a ~10-line `file=` shim around `tqdm` is cheaper
to own than 60 lines of meter. That is a real position. It loses the `every`-N-items
throttle (a window takes minutes; `mininterval` alone is the wrong knob) and it still leaves
`bin/` parsing a format it does not control.

## 2. `registry/tools/_minischema.py` — 290-line JSON Schema validator

**KEEP.** This is the model case, and its docstring already argued it correctly.

The claim in the brief was that a vendored validator with no transitive deps *may be* the
point. It is, and it is enforced three ways:

- **The interpreter is real.** macOS ships Python 3.9.6 with no `jsonschema`.
  `registry/Makefile` states `validate`, `render-check`, `selftest` and `joint` "run on a
  stock interpreter with no pip install". Verified: `registry_validate.py` runs clean under
  `/usr/bin/python3` (3.9.6), 0 errors over 157 records.
- **Networking is asserted absent.** `registry_validate.py:27` defines
  `FORBIDDEN_NET_MODULES`, `check_offline()` raises `OFFLINE-002` if one is loaded, and
  `registry_selftest.py:1300` runs section *"E. the tools import no networking library"*
  against both tools. Tellingly, `registry_validate.py:1553` has to carve out an exception —
  *"jsonschema's optional deps may drag in networking modules; only our own graph matters"* —
  which is evidence **for** the vendoring: the real library's dependency closure is exactly
  what the offline assertion is designed to exclude.
- **It is differentially tested against the real thing.** `_external_validator()`
  (`registry_validate.py:123`) builds a `jsonschema.Draft202012Validator` over the same
  schema set, and `--jsonschema-lib both` runs both and compares. It degrades gracefully
  (returns `None`) when the library is absent.

**I ran that cross-check rather than trusting it.** Under the venv with `jsonschema` 4.26.0:
`0 error(s)` across all 157 records. The vendored validator and the reference library agree.

It also refuses to be a silent subset: it raises on any keyword it does not implement, so a
schema growing a new keyword fails loudly instead of validating nothing.

**One honest correction.** The docstring said `registry_validate` "runs this by default and
cross-checks against the real library whenever it is importable". True of the *tool*
(`--jsonschema-lib` defaults to `both`) but **not of the gate**: `make check` → `validate`
passes `--jsonschema-lib mini` explicitly, and `validate-both` is a separate target wired
into neither `make check` nor `bin/selftest_all.sh`. The cross-check is available and
passing, but nothing runs it automatically. The docstring has been corrected to say exactly
that and to name the command.

**Not changed, deliberately:** I did not wire `validate-both` into `make check`. `check`'s
purity — one stock interpreter, nothing installed — is a designed property, and on the
default interpreter the flag would be a no-op anyway. Wiring it into a *richer* CI lane is
the right home for it, and is left as a proposal.

## 3–6. The provider backends (`urllib` + a vendor CLI)

These files are **owned by a live measurement campaign** (a run is on RunPod as this is
written). Per `AGENTS.md`, they were reviewed read-only; nothing here was edited. The
docstring additions each verdict calls for are written into
[`REVIEW-DEFERRED.md`](REVIEW-DEFERRED.md).

### 3. `jlapi.py` — **KEEP**, and the framing does not apply

This is not an HTTP client at all. It is a `subprocess` wrapper around the vendor's `jl`
CLI, and the docstring's reason is sound: *"The REST surface behind it is not publicly
documented — the vendor documents the CLI — so reimplementing it would make this recipe a
maintenance liability."* Using a documented CLI instead of reverse-engineering an
undocumented REST API is the opposite of reinventing a wheel.

Every bug in its history is CLI-contract drift, which no library prevents: argv order,
`--yes` rejected by `upload`, `--json` landing after a bare `--`, and the probe that matched
its own command text because `jl exec --json` echoes `command` back.

The real cost of the CLI is not correctness but **latency**: each `jl` call is one API round
trip of 10–15 s, which blew a 300 s download timeout twice on a 34-file receipts directory,
forcing a remote `tar czf` and a `sha256sum`-delta uploader — i.e. a hand-rolled `rsync`.
That cost is real, and it is caused by the CLI choice, but it lands in `measure_cloud.py`.

### 4. `runpodapi.py` — **KEEP-BUT-DOCUMENT**. This is where `urllib` actually cost money.

The brief asked whether `urllib` is causing real bugs. It is, and here is the receipt
(`runpodapi.py:92`):

```python
# Cloudflare fronts api.runpod.io and answers urllib's
# default User-Agent with HTTP 403 "error code: 1010"
# (browser integrity check). curl works only because it
# sends one. Not optional.
"User-Agent": "quant-fidelity-suite/0.1",
```

Found, per the commit, *"BY SMOKING IT ON A $1.59/h POD FOR SIX MINUTES"*. `requests` sends
`python-requests/x.y.z` and this 403 class could not have happened. The header was then
copy-pasted into `vastapi.py:91` and `lambdaapi.py:92`, neither of which is behind
Cloudflare — cargo-culting a workaround is itself a symptom.

Other costs, all small individually: `HTTPError`-as-exception plus `exc.read()` to recover a
body; manual `json.dumps`/`loads` on both sides; **no connection pooling** (`gpus()` issues
100+ sequential `urlopen` calls, each a fresh TCP+TLS handshake through Cloudflare;
`_endpoint()` polls every 10 s for up to 900 s); and **no retry of any kind** — a Cloudflare
502 raises hard, mid-run, after the pod is billing.

**Why still KEEP:** the used surface is genuinely tiny (one verb, one endpoint, bearer
token, JSON in/out; no streaming, multipart, cookies, sessions or auth refresh), the file is
never uploaded to an instance, and swapping transports on a controller mid-campaign is not a
trade worth making while a paid run is live. **But the docstring should say `urllib` cost a
live incident, so the next person weighs it with that on the table.** Recorded in
`REVIEW-DEFERRED.md`.

**Reviewer, disagree here if:** you weigh "already burned one paid pod, and has zero retry
on 5xx" above "controller-side, tiny surface". That is a defensible ADOPT for `requests`,
and it is the single strongest ADOPT case in the repo. It is not applied only because the
file is campaign-owned.

### 5. `vastapi.py` — **KEEP-BUT-DOCUMENT**. The hand-rolled retry is a *partial* `urllib3.Retry`.

The docstring records the incident honestly: Vast rate-limits to ~1 req/s and answers 429
with `retry_after`; the banded catalogue search tripped it *"INSIDE the run, after the lease
was written, so a rate limit read as a failed run."*

The half that genuinely had to be custom: `retry_after` arrives **in the JSON body**, not in
the standard `Retry-After` header, so `respect_retry_after_header` would not have found it;
and the `_MIN_INTERVAL = 1.1` client-side pacing is not something `requests` provides either
(that is `pyrate-limiter` territory).

The half that did not, and the gaps that matter:

| `urllib3.Retry` gives | `_req` does |
|---|---|
| `status_forcelist=[429,500,502,503,504]` | **429 only** — a 502/503 raises hard mid-run, after the lease is written: the exact failure mode the 429 fix was written to stop |
| retry on connect/read timeout, connection reset | **none** — falls to `except Exception` and raises |
| exponential backoff with jitter | fixed sleep, no jitter |
| retry budget as `Retry(total=…)` | `_tries: int = 4` threaded through the method signature |

Also: process-global mutable `Vast._last_call` as the rate-limiter's state — set on the
class, not the instance, and not thread-safe.

**Verdict KEEP** on the same grounds as RunPod (controller-side, small surface, campaign-owned),
**but the 5xx and network-error gap is a real latent repeat of an incident already paid for**
and belongs in the docstring. Recorded in `REVIEW-DEFERRED.md`.

### 6. `lambdaapi.py` — **KEEP-BUT-DOCUMENT**, and be honest that this one is textbook

`lambdaapi.py:87` hand-builds HTTP Basic auth:

```python
token = base64.b64encode((self._load_key() + ":").encode()).decode()
```

That is `requests.get(url, auth=(key, ""))`. The `import base64` exists solely for it.
Correctly implemented, and it has never cost anything — every bug in this file's history is
about Lambda's JSON shape (`_KNOWN_DISK_GB` guessed 200 GB for a box whose `df -h /` said
1.4T), not about transport. It also fetches `/instance-types` **twice in one `create()`
call** — two full TLS handshakes for a catalogue already in a local variable, which pooling
would make free.

**Verdict KEEP** by consistency with its siblings and because the cost is zero, **but it
should not pretend to be load-bearing.** It is the weakest KEEP in this document and the
first thing to revisit if the provider backends are ever allowed a dependency.

## 7. `sshbase.py` — **KEEP**, and the missing piece is not a library

`paramiko`/`fabric` would be a large dependency for a very small surface: exec, scp-up,
scp-down. No SFTP, port forwarding, agent forwarding, jump hosts or PTY. And it would
*lose* something — `scp`'s recursive tree copy, which `paramiko` does not implement (you
would hand-roll an `SFTPClient` walk).

What is done right and needs no library: `shlex.quote` at every remote-command boundary,
never `shell=True`; `socket.create_connection` as the readiness probe; and the two
non-obvious bugs the file exists to memorialize (a detached job must record its own exit
code because `wait` does not know a subshell's children; liveness cannot use the recorded
pid because the launching shell forks and exits immediately). The DRY refactor paid off
visibly — the commit notes *"Vast worked on the first try because of that shared base."*

What remains missing is an optional performance optimization, not a dependency:
`ControlMaster` / `ControlPersist` / `ControlPath`. `_ssh_opts()` opens a full
SSH handshake for every exec and every scp. The first safe RunPod path favors
an independently authenticated, attempt-local connection over multiplexing;
revisit only with a proof that one control socket cannot cross attempt or
endpoint identity.

The load-bearing host-authentication gap is closed. Before the first SSH byte,
the controller reads the fresh pod's ED25519 fingerprint from RunPod's
authenticated v2 container-log API. `ssh-keyscan` is treated as untrusted input
and must match that independently retrieved fingerprint exactly. The resulting
owner-only per-attempt `known_hosts` file is then used with
`StrictHostKeyChecking=yes`; ambient agents, password/interactive
authentication and forwarding are disabled. The explicit read-scoped target
token is transferred only as a mode-0600 file and erased after target fetch.

## 8. `bin/fidelity/common.py` — **KEEP**

The docstring's reason is correct and holds: *"Both runners are meant to be copy-pasted onto
a stock machine and run with the system `python3`; a dependency here would turn a one-paste
recipe into a virtualenv tutorial."* `common.py` **is** in `BUNDLE.txt`, so it runs on both
the laptop and the instance — the laptop's stock 3.9 is the binding constraint, and `rich`
being present on the instance does not help.

`Console` is not a reimplementation of `logging`. It has one property `logging` does not
give by default: **every write is redacted.** `_w()`, `err()` and `CommandError.__str__` all
route through `redact()`, which strips registered secrets plus five token shapes — including
the 64-char Jupyter access token that every `jl … --json` record carries in a query string.
`logging` could do this with a `Filter`, but the contract here is stronger: there is no way
to write to the stream *without* redaction. For a project whose rule is "never echo a token
into a log, a receipt, or git", that inversion is the point.

`sha256_file` is the one thing here that *is* stdlib now — `hashlib.file_digest`, 3.11+ —
and is blocked purely by the 3.9 floor. See row 11.

## 9. `bin/fidelity_stats.py` — **KEEP**, the strongest keep in the repo

~170 lines: `_betacf` (Lentz continued fraction), `_betai`, `t_two_sided_p`,
`t_quantile_975`, `bca_interval` (BCa bootstrap with jackknife acceleration),
`sign_test_two_sided`, `wilcoxon_signed_rank` with tie correction.

This is squarely in the measurement path — it is the CI machinery behind published paired
per-window deltas — and it is **not naive reinvention**. It reaches for stdlib wherever
stdlib has the piece: `statistics.NormalDist().inv_cdf/cdf` for the normal quantile,
`statistics.fmean` inside the bootstrap resampler, `math.comb` for the exact sign test. Only
the pieces genuinely absent from stdlib are hand-written.

`t_quantile_975` carries a *correctness* argument, not a convenience one: inverting the same
`betai` the p-value uses "is exact for ANY df and cannot disagree with the p-value printed
next to it." Two independent implementations of the t distribution can print a CI and a
p-value that contradict each other. `scipy` would also make the numbers depend on a library
whose reduction order and dtype can change between versions — the determinism argument the
brief flagged as a strong KEEP, and it applies here more than anywhere else.

## 10. `canonical_json` spelled out in each tree — **KEEP**

`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` appears in 22
files. This is deliberate and documented in three places, and the reasoning is sound: it is
a **published wire format**, and `common.py:seal()` exists so *"a stranger can verify our
receipts with `python3 -c` and no imports from us."* `BUNDLE.txt` uploads
`registry/tools/registry_lib.py` to the instance precisely so the derived fields are
computed by the registry's own code — *"Two implementations of one hash is two chances to
disagree."*

That argues for **one copy per independently-shippable tree**, which is what exists.

**One gap worth a reviewer's attention:** `common.py:66` says the copy there "must match
`registry/tools/registry_lib.py` exactly", and **no test asserts that.** The two are
byte-identical today (verified by reading both), but the invariant is enforced by a comment.
A three-line rung asserting the two functions agree on a nested fixture would close it.
Proposed, not applied — it belongs in whichever selftest the reviewer prefers.

## 11. `sha256_file` × 15 — **ADOPT (proposed)**

Fifteen distinct definitions of the same chunked sha256 loop, with chunk sizes that drift
arbitrarily between copies (`1024*1024`, `1<<20`, `1<<22`, `8<<20`). Unlike `canonical_json`,
**this duplication is justified nowhere.** It is not a wire format; it is a loop.

- `engines/tools/` (7 copies) runs on the instance under 3.12+, so those can become
  `hashlib.file_digest(fh, "sha256").hexdigest()` outright.
- `bin/` and `registry/` are blocked by the 3.9 floor, but the three copies inside
  `bin/fidelity/` alone should collapse to one.

Proposed, not applied: several of the files involved are campaign-owned or bundled, and this
touches digest computation that binds published receipts. Low risk, but not zero, and it
earns a test rather than a sweep.

## 12. ISO-8601 hand-spelled × 13, two of them deprecated — **ADOPT (proposed)**

`time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())` in 9 files, 13 times — while
`common.utcnow()` already exists and does exactly this.

Two are worse than duplication:

- `bin/fidelity/dsmanifest.py:42` — `datetime.datetime.utcnow().strftime(...)`
- `bin/fidelity/dscompare.py:1158` — `__import__("datetime").datetime.utcnow().strftime(...)`,
  inline inside a dict literal

**`datetime.utcnow()` is deprecated as of 3.12** and is the classic naive-datetime footgun.
`registry/tools/registry_render.py:657` already does it correctly with
`datetime.now(timezone.utc)`. These are receipt timestamps — provenance, not numbers — so
the blast radius is small, but one API is scheduled for removal and the instance runs 3.12.

Proposed, not applied: `dscompare.py`/`dsmanifest.py` are in `BUNDLE.txt` and `cardmeta.py`'s
neighbourhood was recently held by a second session. Cheap and zero-risk once the tree is quiet.

## 13. `bin/kld_preview.py:251,354` — `sum()/len()` in the measurement path — **ADOPT (proposed)**

These are the **only** two places in the repo that compute a mean with a bare `sum()/len()`.
Every sibling is deliberate about it: `previewstats.py:88` uses `statistics.fmean`;
`joint_standard.py`, `emit_clean_scope_report.py` and `check_doc_numbers.py` use
`math.fsum(...)/len(...)`; and `registry/tools/coverage_sim.py:85` carries an explicit note
that `fmean` and a naive mean *"differ in the last ULP"*.

In fairness the blast radius is bounded: these are the per-window *display* mean and the
`per_window` dict; the panel mean that reaches the receipt routes through
`previewstats.stratified_mean`, which does use `fmean`. But this repo's premise is that the
last ULP is the product, and the surrounding code argues against these two lines.

**Proposed, not applied**, per the instruction to prefer proposing over doing on anything
touching the measurement path. A reviewer should decide whether changing a printed
diagnostic counts as changing a number.

## 14. The selftest harness, re-declared ~24 times — **KEEP the design, ADOPT the shared module (proposed)**

Zero `import unittest`, zero `import pytest` across 39 selftest files. Each re-declares a
`check()` closure plus its own summary block — ~300 lines of pure harness, in four
gratuitously different spellings (`ok`/`FAIL` vs `PASS`/`FAIL`, `(ok, name)` vs
`(name, cond)` arg order). `bin/selftest_all.sh` adds a third layer in bash.

**No docstring anywhere explains why `unittest` was rejected**, and `unittest` is stdlib —
the dependency rule does not block it. So the usual defense is unavailable.

Even so I lean KEEP on the *design*, because three real arguments are visible in the code
even though nobody wrote them down: **SKIP is a first-class verdict** here, printed with the
missing dependency and never silently dropped, and these skips are runtime-discovered
(`import numpy` failing, `mlx` absent, no GPU) where `unittest.skipIf` is decorator-time;
these run as bare scripts on rented boxes under a `FIDELITY_PYTHON` that may not be the
system interpreter; and **the output is the artifact** — rung-by-rung human-readable
evidence, not a dot-progress bar.

The defensible fix is not "switch to pytest" but "put `check`/`skip`/`summary` in one module
and import it 24 times." `bin/fidelity/common.py` is already exactly such a shared home.
Proposed; it touches 24 files and every gate, so it wants its own change, not this one.

## 15. The atomic-write helper × 8 — **KEEP**

There is no stdlib atomic write. `tempfile` + `os.replace` **is** the correct recipe and
that is what all eight copies do. The rationale is documented at three separate sites from a
real incident — `registry_lib.py:199` records a watchdog kill between the truncate and the
last line leaving `data/measurements.jsonl` damaged, and `common.py:write_json` records two
more (a fixed `.tmp` name letting two writers interleave into one staging file; a failed
replace onto a directory leaving the temp behind forever). `common.py` also `fsync`s,
because "the machine that writes one is often destroyed minutes later."

Correctly hand-rolled. The only criticism is eight copies where two — one per tree — would
do, and that is the least urgent item in this document.

## 16. Two checksum-file parsers — **KEEP, deliberately two**

`dsformat.py:422` is strict and positional, validates hex, rejects duplicate paths, and runs
`check_relpath` on every entry against path traversal in a remote `checksums.txt`.
`verify_published_sums.py:53` is lenient, tolerates the `*name` binary marker, and silently
skips malformed lines.

They answer different questions at different trust boundaries: one parses **our own sealed**
manifest (a security boundary — must be strict), the other parses **a third party's**
`SHA256SUMS` off HuggingFace (you do not control the producer's format — must be lenient).
That is a legitimate reason for two parsers, and it deserves a cross-reference comment so
nobody "fixes" the lenient one into strictness.

## 17. Two retry curves in `engines/tools/fetch_*.py` — **KEEP, reconcile the constants**

`fetch_truncated_ckpt.py` sleeps `min(30.0, 1.5 * 2**attempt)`; `fetch_nonrouted_sparse.py`
sleeps `min(2**attempt, 30)`. Same intent, silently different behaviour, same directory.

These live in `engines/tools/`, so a library is not blocked by policy — but the retry wraps a
**Range** request and also catches the module's own `IOError("short read")`, i.e. it retries
partial-content responses, which is application-level logic `urllib3.Retry` does not
express. The defect is that there are two curves, not that they exist.

## 18. Checked and found correctly done — stated explicitly

Being even-handed means naming what is *not* a finding. Each of these was searched for and
came back clean:

- **`argparse`: not reinvented.** 65 of ~68 scoped scripts use it, with real subparsers and
  `set_defaults(func=…)`. No hand-rolled `--help`, no manual `sys.argv` walking outside two
  40-line internal helpers. `typer` being installed on the instance is irrelevant — it would
  violate the `bin/` rule.
- **Terminal color / ANSI: zero occurrences.** No `\033[`, no hand-rolled color constants.
  `rich`/`colorama` have nothing to replace.
- **Table formatting: not a wheel.** The renderers emit **Markdown pipe tables**, which by
  definition need no column-width computation. `rich.table` would be the wrong tool — the
  output is committed Markdown, not a terminal render.
- **`pathlib`, `glob`, `fnmatch`: used properly.** No `os.path` string surgery of note.
- **JSONL: correctly hand-parsed.** JSONL has no stdlib module; per-line `json.loads` with
  line numbers for error messages is exactly right.
- **CSV: none hand-rolled.** Every `.split(",")` is `--flag=a,b,c` argument splitting.
- **`dataclasses`: used, not reimplemented.** 11 files use `@dataclass`; no hand-written
  `__init__`/`__eq__`/`__repr__` triples.
- **`enum`: zero imports, and that is correct.** The enum-shaped constants are membership
  tuples validating strings that arrive from and return to JSON verbatim. `StrEnum` is
  3.11+ (blocked), and a plain `Enum` would add `.value` ceremony at every serialization
  boundary for no gain. *Caveat:* `LANES` is defined twice with different contents
  (`dsformat.py:60` has a trailing `"other"`, `census.py:273` does not), as is `ROLES`. A
  drift risk worth a comment, not a refactor.
- **Caching: absent, and nothing obviously needs it.**
- **Subprocess: no wrapper reimplements `subprocess.run`.** `common.run()` adds redaction,
  which is the point.
- **Semver:** `tuple(int(p) for p in version.split(".")[:2]) < (5, 16)` in 4 places.
  `packaging.version` would be correct but is overkill and not guaranteed present; it will
  raise on a dev suffix like `5.16.0.dev0`. Low priority.

---

## What this audit changed

- `engines/tools/progress.py` and `bin/selftest_progress.py`: the **false** "tqdm is not in the
  bundle / a rented instance gets no pip install" rationale replaced with the verified one
  (tqdm is already installed transitively; it has no newline mode; the output is a parsed
  contract).
- `bin/BUNDLE.txt`: same correction to the meter's entry comment, plus a stale reference to
  `bin/selftest_bundle_imports.py` — a file that does not exist and never did. The check it
  names is real and lives in `bin/selftest_progress.py` (rung P11).
- `registry/tools/_minischema.py`: docstring corrected to distinguish the tool's default
  (`both`) from the gate's behaviour (`mini`), and to name `make validate-both`.
- `docs/REVIEW-DEFERRED.md`: the campaign-owned findings, with their patches.

## What this audit deliberately did not change

No ADOPT was applied. Rows 11–14 are all proposals, for two reasons that are worth stating
plainly rather than dressing up: the highest-value ADOPT (`requests` in `runpodapi.py`) sits
in a file a **live paid measurement currently owns**, and the next two touch the measurement
path or files a second session was recently holding. Nothing here is urgent enough to race
another process for.

The honest summary is that **the audit's product is corrected reasoning, not a diff.** The
codebase reinvents far less than its size suggests, and where it does hand-roll it usually
says why. The one place it said why and was *wrong* was the meter that prompted the
question — and the correct reason turned out to be better than the one it gave.
