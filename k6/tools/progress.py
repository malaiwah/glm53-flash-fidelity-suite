"""A progress meter that stays readable when stdout is a FILE.

Why this exists rather than `tqdm`
----------------------------------
Every measurement stage runs as ``nohup ... > logs/stage-<name>.log``, so the
process that needs a progress bar is exactly the process whose stdout is never
a terminal.  A carriage-return spinner written to a file produces one line
several megabytes long that no pager will open -- which is worse than the
silence it replaced.  So this module has two rendering modes and picks by
``isatty()``:

* **TTY** -- one line rewritten in place with ``\\r``, refreshed a few times a
  second, the familiar live bar.
* **NOT a TTY** -- newline-terminated lines, throttled (default one every 30 s
  *or* every N items, whichever comes first), each one a complete standalone
  status.  ``tail -f`` on the stage log reads like a heartbeat, and the whole
  file stays a few hundred lines.

Why NOT `tqdm`, for the right reason
------------------------------------
This docstring used to say "because `tqdm` is not in the bundle" and that a
rented instance gets no pip install.  **That was false**, and the audit in
``docs/DEPENDENCIES.md`` retired it: ``bin/bootstrap_measure.sh`` installs
``transformers==5.16.1`` and ``huggingface_hub``, and BOTH hard-require ``tqdm``
(``tqdm>=4.60`` and ``tqdm>=4.42.1`` respectively).  tqdm is already on every
measurement instance, unconditionally, before the measure stage starts; ``rich``
is installed explicitly on the same pip line.  The incremental dependency cost
of using either is exactly zero, and "avoid a dependency that is already
transitively present" is not a reason.

The real reason is the one above, and it is structural: **tqdm has no newline
mode.**  In ``tqdm/std.py`` the write is ``fp_write('\\r' + s + ...)`` --
unconditional -- and the class's only ``isatty()`` branch decides whether to
*disable*, never how to *render*.  So into a file tqdm offers exactly two
behaviours, both measured rather than assumed:

* default (also with ``ascii=True, mininterval=...``): 40 iterations become ONE
  line with 7 embedded ``\\r``.  Configuring it fixes the block glyphs and the
  update count and does not touch the ``\\r``.
* ``disable=None``, tqdm's own non-TTY affordance: complete silence -- which is
  the two-to-three-hour void this meter was written to end.

Getting throttled, newline-terminated lines out of tqdm means passing a custom
``file=`` object that rewrites ``\\r`` to ``\\n`` -- writing code anyway, and
then owning a shim wrapped around a dependency instead of sixty readable lines.
It would also still lack the ``every``-N-items throttle, which is the knob that
matters when one item is a multi-minute window.

And the output here is a CONTRACT, not decoration: ``measure_cloud``'s
``_progress_counter`` parses ``progress: <label> <n>/<total>`` out of the log
tail (see the note at the end of this docstring).  ``bin/`` cannot even import
this module -- it runs on stock python3.9 with no torch and no ``k6/tools`` on
``sys.path`` -- so the prefix is duplicated there and
``bin/selftest_progress.py`` asserts the two agree.

What it must not do
-------------------
This meter is inside the measurement loop, so it may not perturb the
measurement.  It therefore:

* never touches a tensor, never calls ``.item()``, never synchronizes a device
  -- the only thing it reads is an integer the caller already incremented;
* calls ``time.monotonic()`` at most once per ``check_every`` updates (the
  default check is cheap arithmetic on the counter), so an inner loop that
  advances by 3 does not pay for a clock read per matrix;
* writes nothing at all when throttled, and holds no lock (a single writer is
  assumed -- the fill loops are one consumer thread each).

The rendered shape is the familiar one, with a ``progress:`` prefix so a reader
(human or ``grep``) can tell it from the JSON records the same streams carry:

    progress: fill L003/g0 552/864 63% [00:17<00:09, 32.1 it/s]
    progress: windows 7/25 28% [21:43<56:00, 0.005 it/s] eta 2026-08-31T04:12:09Z

The item counter in that line is what makes a stalled run distinguishable from
a slow one: the count is monotonic, so two consecutive polls that read the same
``n`` mean no forward progress, which ``bin/measure_cloud.py``'s stage waiter
reports (see ``_progress_counter``).  It stalls nothing and kills nothing --
teardown semantics are unchanged -- it only stops a 0%-GPU hang from looking
exactly like healthy slow work.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, TextIO

#: How often a non-TTY stream gets a line, in seconds.  Thirty seconds is
#: chosen against the two loops that use it: a GGUF layer fill takes ~27 s and a
#: streaming window ~3-24 min, so every fill produces at least one line and a
#: window produces a handful -- enough to see a rate, few enough to read.
DEFAULT_INTERVAL_SECONDS = 30.0

#: TTY refresh cadence.  Fast enough to look live, slow enough that the write
#: never shows up in a profile.
TTY_INTERVAL_SECONDS = 0.25

PREFIX = "progress:"


def _hms(seconds: float) -> str:
    """MM:SS, or H:MM:SS past an hour.  Unknown -> ``--:--``."""
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


def _rate(n: int, elapsed: float) -> str:
    if elapsed <= 0 or n <= 0:
        return "?"
    per_second = n / elapsed
    if per_second >= 1.0:
        return "%.2f it/s" % per_second
    # Below one item per second the reciprocal is the number a human wants:
    # "23.7 min/window" is the unit this project actually plans in.
    return "%.1f s/it" % (elapsed / n)


class Progress:
    """Count items, render ``n/total [elapsed<remaining, rate]``.

    ``total`` may be ``None`` (unknown), in which case the remaining-time and
    percentage fields are omitted rather than guessed.
    """

    def __init__(
        self,
        total: Optional[int],
        *,
        label: str = "",
        stream: Optional[TextIO] = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        every: Optional[int] = None,
        enabled: bool = True,
        check_every: int = 1,
    ) -> None:
        self.total = int(total) if total else None
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self.every = int(every) if every else None
        self.n = 0
        self.enabled = bool(enabled) and interval > 0
        self.check_every = max(1, int(check_every))
        try:
            self.tty = bool(self.stream.isatty())
        except Exception:  # noqa: BLE001 - a stream with no isatty is not a tty
            self.tty = False
        self.interval = TTY_INTERVAL_SECONDS if self.tty else float(interval)
        self.started = time.monotonic()
        self._last_emit = self.started
        self._last_n = 0
        self._since_check = 0
        self._dirty = False

    # -- rendering --------------------------------------------------------
    def render(self, elapsed: Optional[float] = None) -> str:
        if elapsed is None:
            elapsed = time.monotonic() - self.started
        parts = [PREFIX]
        if self.label:
            parts.append(self.label)
        if self.total:
            parts.append("%d/%d" % (self.n, self.total))
            parts.append("%d%%" % (100 * self.n // self.total))
            remaining = (
                (self.total - self.n) * elapsed / self.n if self.n > 0 else None
            )
            parts.append("[%s<%s, %s]" % (_hms(elapsed), _hms(remaining), _rate(self.n, elapsed)))
        else:
            parts.append("%d" % self.n)
            parts.append("[%s, %s]" % (_hms(elapsed), _rate(self.n, elapsed)))
        return " ".join(parts)

    def _emit(self, elapsed: float, final: bool) -> None:
        line = self.render(elapsed)
        try:
            if self.tty and not final:
                self.stream.write("\r" + line + "\x1b[K")
            elif self.tty:
                self.stream.write("\r" + line + "\x1b[K\n")
            else:
                self.stream.write(line + "\n")
            self.stream.flush()
        except Exception:  # noqa: BLE001 - a broken stage log must not kill a run
            self.enabled = False
            return
        self._last_emit = time.monotonic()
        self._last_n = self.n
        self._dirty = not final

    # -- driving ----------------------------------------------------------
    def update(self, step: int = 1) -> None:
        """Advance by ``step`` and emit if the throttle allows.

        Cheap path first: integer add plus one comparison.  ``time.monotonic``
        is only consulted once every ``check_every`` updates, and only then can
        a write happen.
        """
        self.n += step
        if not self.enabled:
            return
        self._since_check += 1
        if self._since_check < self.check_every:
            return
        self._since_check = 0
        if self.every and (self.n - self._last_n) >= self.every:
            self._emit(time.monotonic() - self.started, final=False)
            return
        now = time.monotonic()
        if now - self._last_emit >= self.interval:
            self._emit(now - self.started, final=False)

    def close(self, *, suffix: str = "", force: bool = False) -> None:
        """Emit the final line -- unless this unit finished faster than a tick.

        A closing line for a unit that never needed a mid-line is pure log
        volume: the exl3 lane fills a layer in ~4.5 s and already prints a JSON
        record per fill, so an unconditional close would double the size of a
        25-window stage log to say nothing new.  The GGUF lane's 27 s fills and
        the multi-minute window meter (which sets ``every``) both still get
        their line, because both satisfy one of the conditions below.
        """
        wanted = (
            force
            or self.every is not None
            or self._last_n > 0
            or (time.monotonic() - self.started) >= self.interval
        )
        # ...but never repeat a line that would say exactly what the last one
        # said.  An `every=1` meter has already reported the final item by the
        # time close() runs; the only reason to speak again is a suffix, which
        # is new information ("capture complete").
        if self._last_n == self.n and not suffix:
            wanted = False
        if not self.enabled or self.n == 0 or not wanted:
            if self.tty and self._dirty:
                try:
                    self.stream.write("\n")
                    self.stream.flush()
                except Exception:  # noqa: BLE001
                    pass
            return
        elapsed = time.monotonic() - self.started
        line = self.render(elapsed)
        if suffix:
            line = line + " " + suffix
        try:
            if self.tty:
                self.stream.write("\r" + line + "\x1b[K\n")
            else:
                self.stream.write(line + "\n")
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass
        self._dirty = False

    # context manager sugar -- `with Progress(...) as p:` closes on exception
    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def interval_from_env(default: float = DEFAULT_INTERVAL_SECONDS) -> float:
    """``FIDELITY_PROGRESS_SECONDS`` override; ``0`` disables the meter.

    An env var rather than only a flag because the stage driver composes the
    engine argv from ``bin/engines.json`` and a debugging operator on the box
    should be able to turn the cadence up without editing a pinned invocation.
    """
    raw = os.environ.get("FIDELITY_PROGRESS_SECONDS")
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default
