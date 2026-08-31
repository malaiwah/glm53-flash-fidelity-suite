#!/usr/bin/env python3
"""T14 -- the progress meter, and the two ways it used to be useless.

A measure stage prints one line when it starts and one when it ends.  On the
streaming lane that is a two-to-three hour silence in `logs/stage-measure.log`,
and it looks exactly like a hang: a first-time contributor checked `ps` twice to
convince themselves the run was alive, and the first GGUF run sat at 0% GPU for
two hours with nothing saying so.

The two failure modes this battery holds the line on:

  P1-P4  **a file is not a terminal.**  Every stage runs under
         `nohup ... > logs/stage-<name>.log`, so the process that needs a
         progress bar is precisely the one whose stdout is never a TTY.  A
         carriage-return spinner into a file is one unreadable megabyte-long
         line.  The meter must emit newline-terminated lines when
         `isatty()` is false and the live in-place form only when it is true.
  P5-P7  **it must not become log spam, or a dependency.**  Throttled by time
         and by item count; a unit that finishes faster than one tick stays
         silent (the exl3 lane fills a layer in 4.5 s and already logs a JSON
         record per fill); stdlib only, no tqdm, because `bin/BUNDLE.txt` is an
         explicit upload list and a rented instance gets no pip install.
  P8-P10 **it has to be WIRED, and it has to ship.**  A meter nothing calls is
         indistinguishable from no meter (the same lesson the GGUF lane taught:
         "a capability nothing can invoke is indistinguishable from a missing
         one").  So: both engines tick it in their real loops, and every
         k6/tools module they import at module scope is in `bin/BUNDLE.txt` --
         an omission there is an ImportError at the START of the measure stage,
         after the bootstrap, the 200 GB fetch and the panel are all paid for.
  P11-P13 **liveness is not progress.**  `_stage_is_alive`'s pgrep says yes for
         a hung process forever.  `measure_cloud._progress_counter` reads the
         meter's item count out of a log tail so the controller can SAY a
         stalled run is stalled.  It reports; it never returns a verdict and
         never touches teardown.

Every rung fails without its fix: `git stash` removes k6/tools/progress.py and
the wiring, and P1-P13 fail (P1-P7 on the missing import, P8-P13 on the source
and bundle checks).

Run:  python3 bin/selftest_progress.py
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "k6" / "tools"))

_pass, _fail = 0, 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print("  PASS  %s" % msg)


def no(msg: str, detail: str = "") -> None:
    global _fail
    _fail += 1
    print("  FAIL  %s%s" % (msg, ("\n        " + detail) if detail else ""))


def check(cond, msg: str, detail: str = "") -> None:
    ok(msg) if cond else no(msg, detail)


class _Tty(io.StringIO):
    """A stream that claims to be a terminal.  StringIO's isatty() is False."""

    def isatty(self) -> bool:  # noqa: D401
        return True


# ---------------------------------------------------------------------------
# P1-P4: a file is not a terminal
# ---------------------------------------------------------------------------
def rung_file_mode(P) -> None:
    buf = io.StringIO()
    meter = P.Progress(864, label="fill L003/g0 matrices", stream=buf,
                       interval=0.02, check_every=1)
    for _ in range(288):
        meter.update(3)
        time.sleep(0.0004)
    meter.close()
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.strip()]

    check("\r" not in out,
          "P1  file mode emits NO carriage returns (a \\r spinner in a "
          "nohup log is one megabyte-long line)",
          "found %d" % out.count("\r"))
    check(out.endswith("\n") and len(lines) >= 2,
          "P2  file mode emits several newline-terminated lines",
          "lines=%d, endswith-newline=%s" % (len(lines), out.endswith("\n")))
    # the tqdm shape: n/total, percent, [elapsed<remaining, rate]
    shape = re.compile(
        r"^progress: fill L003/g0 matrices \d+/864 \d+% "
        r"\[[0-9:]+<[0-9:-]+, [0-9.]+ (it/s|s/it)\]$")
    bad = [ln for ln in lines if not shape.match(ln)]
    check(not bad,
          "P3  every line carries n/total, percent, elapsed<remaining and a rate",
          "e.g. %r" % (bad[0] if bad else ""))
    check(lines[-1].startswith("progress: fill L003/g0 matrices 864/864 100%"),
          "P4  the last line closes the unit at 100%%", lines[-1] if lines else "(none)")


def rung_tty_mode(P) -> None:
    tty = _Tty()
    meter = P.Progress(100, label="fill", stream=tty, interval=30, check_every=1)
    for _ in range(100):
        meter.update(1)
        time.sleep(0.004)
    meter.close()
    out = tty.getvalue()
    check("\r" in out and out.count("\n") == 1,
          "P5  TTY mode rewrites ONE line in place (\\r, single trailing newline)",
          "carriage returns=%d newlines=%d" % (out.count("\r"), out.count("\n")))
    # interval=30 was passed and ignored: a TTY refreshes on its own cadence.
    check(out.count("\r") > 1,
          "P6  TTY mode refreshes live rather than obeying the file throttle",
          "only %d updates" % out.count("\r"))


# ---------------------------------------------------------------------------
# P7: throttling -- a fast unit stays silent
# ---------------------------------------------------------------------------
def rung_throttle(P) -> None:
    quiet = io.StringIO()
    fast = P.Progress(864, label="fill L003/g0 matrices", stream=quiet,
                      interval=30, check_every=8)
    for _ in range(288):
        fast.update(3)
    fast.close()
    loud = io.StringIO()
    windows = P.Progress(25, label="windows", stream=loud, interval=30, every=1)
    for _ in range(25):
        windows.update(1)
    windows.close(suffix="capture complete")
    lines = loud.getvalue().splitlines()
    # 25 item lines + one closing line that says something the last one did not.
    check(quiet.getvalue() == "" and len(lines) == 26
          and lines[-1].endswith("capture complete"),
          "P7  a fill faster than one tick prints nothing; an `every=1` meter "
          "prints every item and closes once",
          "fast=%d bytes, windows=%d lines, last=%r"
          % (len(quiet.getvalue()), len(lines), lines[-1] if lines else ""))

    # The same meter closed with nothing new to say stays silent rather than
    # repeating its own last line.
    mute = io.StringIO()
    again = P.Progress(3, label="windows", stream=mute, interval=30, every=1)
    for _ in range(3):
        again.update(1)
    before = mute.getvalue()
    again.close()
    check(mute.getvalue() == before,
          "P7b close() does not repeat a line identical to the one it just "
          "printed")


# ---------------------------------------------------------------------------
# P8-P9: the meter is actually wired into both engines
# ---------------------------------------------------------------------------
def rung_wiring() -> None:
    stream_src = (ROOT / "k6" / "tools" / "stream_score.py").read_text(encoding="utf-8")
    capture_src = (ROOT / "k6" / "tools" / "hf_capture.py").read_text(encoding="utf-8")

    # The fill loops are the 2-3 hours.  There are three of them (packed/native/
    # mlx/gguf/nvfp4 share one, dione has its own, exl3hf/tr3 have a third), and
    # each one increments `decoded_matrices`.  Every one must tick.
    ticks = stream_src.count("self._tick(3)")
    bumps = stream_src.count("self.decoded_matrices += 3")
    check(ticks == bumps and ticks == 3,
          "P8  all %d stream_score fill loops tick the meter (not just the one "
          "the author was looking at)" % bumps,
          "ticks=%d decoded_matrices bumps=%d" % (ticks, bumps))
    check("window_meter.update(1)" in stream_src
          and "window_meter.update(1)" in capture_src,
          "P9  both engines meter the OUTER window loop -- the loop that "
          "answers 'will this finish inside --max-runtime'")
    check("--progress-seconds" in stream_src and "--progress-seconds" in capture_src,
          "P10 both engines expose --progress-seconds (0 disables)")

    # P8 is a source check because constructing a real ExpertStreamer allocates
    # a 9.6 GB expert slab.  The METHODS the fill loops call have no such
    # dependency, so run them for real against a stub -- a source grep can prove
    # a call site exists but not that the thing it calls works.
    try:
        import stream_score as S
    except Exception as exc:  # noqa: BLE001 - numpy/torch absent on this interpreter
        print("  SKIP  P8b live _begin_fill/_tick/_end_fill (%s)" % exc)
        return
    sink = io.StringIO()

    class _Stub(object):
        progress = True
        progress_interval = 0.02
        _fill_meter = None
        _FILL_CHECK_EVERY = S.ExpertStreamer._FILL_CHECK_EVERY

    stub = _Stub()
    S.ExpertStreamer._begin_fill(stub, 3, 0, 288)
    stub._fill_meter.stream = sink
    stub._fill_meter.tty = False
    for _ in range(288):
        S.ExpertStreamer._tick(stub, 3)
        time.sleep(0.0002)
    S.ExpertStreamer._end_fill(stub)
    lines = [ln for ln in sink.getvalue().splitlines() if ln.strip()]
    check(lines and lines[-1].startswith("progress: fill L003/g0 matrices 864/864")
          and stub._fill_meter is None,
          "P8b the streamer's own _begin_fill/_tick/_end_fill count 288 experts "
          "as 864 matrices and release the meter",
          "last=%r meter=%r" % (lines[-1] if lines else "", stub._fill_meter))


# ---------------------------------------------------------------------------
# P11: bundle completeness -- the ImportError that would land at hour zero of
# the measure stage, after everything has been paid for
# ---------------------------------------------------------------------------
_IMPORT = re.compile(r"^\s*(?:import (\w+)|from (\w+) import)", re.M)


def rung_bundle() -> None:
    listed = {ln.strip() for ln in
              (ROOT / "bin" / "BUNDLE.txt").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")}
    tools = ROOT / "k6" / "tools"
    engines = [rel for rel in listed
               if rel.startswith("k6/tools/") and rel.endswith(".py")]
    missing = []
    for rel in engines:
        src_path = ROOT / rel
        if not src_path.is_file():
            continue
        for m in _IMPORT.finditer(src_path.read_text(encoding="utf-8")):
            name = m.group(1) or m.group(2)
            sibling = "k6/tools/%s.py" % name
            if (tools / ("%s.py" % name)).is_file() and sibling not in listed:
                missing.append("%s imports %s" % (rel, sibling))
    check(not missing,
          "P11 every k6/tools module a BUNDLE.txt engine imports is itself in "
          "BUNDLE.txt (a missing line is an ImportError after the 200 GB fetch)",
          "; ".join(sorted(set(missing))))


# ---------------------------------------------------------------------------
# P12-P13: liveness is not progress
# ---------------------------------------------------------------------------
def rung_counter(P) -> None:
    import measure_cloud as mc

    if not hasattr(mc, "_progress_counter") or not hasattr(mc, "progress_meter_PREFIX"):
        no("P12 measure_cloud exposes _progress_counter / progress_meter_PREFIX",
           "the controller cannot tell a stalled stage from a slow one without them")
        no("P13 the counter reads the newest meter line", "not implemented")
        return
    check(mc.progress_meter_PREFIX == P.PREFIX,
          "P12 measure_cloud's copy of the meter prefix matches progress.py's "
          "(bin/ may not import k6 tools: stock python3.9, no torch)",
          "%r vs %r" % (mc.progress_meter_PREFIX, P.PREFIX))

    tail = "\n".join([
        '{"fill": "L003/g0", "seconds": 26.74}',
        "progress: fill L004/g0 matrices 432/864 50% [00:13<00:13, 32.1 it/s]",
        "progress: fill L004/g0 matrices 720/864 83% [00:22<00:04, 32.0 it/s]",
    ])
    got = mc._progress_counter(tail)
    none_case = mc._progress_counter('{"stage": "setup"}\nno meter here\n')
    unknown_total = mc._progress_counter("progress: layer fills 7 [01:30, 12.9 s/it]")
    check(got == 720 and none_case is None and unknown_total == 7,
          "P13 the counter reads the NEWEST meter line, tolerates the JSON "
          "records beside it, and returns None (never a stall) when the tail "
          "has no meter at all",
          "newest=%r none=%r unknown-total=%r" % (got, none_case, unknown_total))


def main() -> int:
    print("== T14: progress meter ==")
    try:
        import progress as P
    except ImportError as exc:
        no("P1-P7 k6/tools/progress.py is importable", str(exc))
        print("\nselftest_progress: %d passed, %d failed" % (_pass, _fail))
        return 1
    rung_file_mode(P)
    rung_tty_mode(P)
    rung_throttle(P)
    rung_wiring()
    rung_bundle()
    rung_counter(P)
    print()
    print("selftest_progress: %d passed, %d failed" % (_pass, _fail))
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
