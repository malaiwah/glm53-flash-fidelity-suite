#!/usr/bin/env python3
"""Priority-ordered, overlapped checkpoint fetch for the layer-outer schedule.

Why this file exists
--------------------
`k6/tools/layer_outer.py` runs

    for each layer:  load it once;  for each window: push that window through it;  free it

Layer N's weights are not read until the capture reaches layer N.  The default
pipeline nevertheless waits for the WHOLE checkpoint to land before the first
window is pushed, because `stage_measure.sh fetch_target` is a separate stage
that must complete before `capture` starts.  For a model that lands on the Hub
at 09:00 and gets quantized by four people by 11:00, those serialized hours are
the entire race.

This module makes the fetch a PRIORITY QUEUE instead of a barrier:

  * the tiny files first -- `config.json`, the tokenizer, and above all
    `model.safetensors.index.json`, whose `weight_map` is what makes everything
    below plannable at all;
  * then the RESIDENT set: every shard holding a tensor that is not a decoder
    layer (embeddings, the final norm, `lm_head`, router biases, rotary tables).
    `build_streamed_model` loads all of those before the first layer;
  * then the shards holding layer 0, layer 1, ... in that order;
  * everything else keeps downloading in the background at full speed while the
    capture runs.

WHAT THIS DOES NOT DO
---------------------
It does not change a single number.  It changes WHEN bytes arrive, never which
bytes, never the order of any arithmetic, never the schedule the windows are
pushed through.  A race-mode capture and a fetch-then-capture capture of the
same checkpoint produce the same `capture_content_digest`; `bin/selftest_race_mode.py`
asserts exactly that rather than asserting it here.

THE HEAD IS NEEDED FIRST, NOT LAST
----------------------------------
The obvious guess -- "the final norm and `lm_head` are needed last, so fetch
them last" -- is WRONG for this capture path, and getting it wrong would
deadlock the run at layer 0.  `build_streamed_model` performs one resident load
of everything outside the decoder stack BEFORE any layer is streamed, and
refuses if any non-layer parameter is still on the meta device; `hf_capture`
then reads `head.weight.shape` for the vocab/hidden sizes and registers its tap
as a pre-hook ON the head.  And a hidden-form dataset must publish the head's
own tensor bytes, because `compare` refuses a hidden-form capture with a null
head content digest (HEAD-4, no override).  So the head is a priority-0 file on
every form we publish.  Priority is derived from the checkpoint's own
`weight_map`, never from a guess about which tensors "come last".

MIS-BUCKETING IS CONSERVATIVE BY CONSTRUCTION
---------------------------------------------
A shard's priority is the MINIMUM layer index over every decoder tensor it
carries, and any key the layer regex does not match falls into the resident
bucket.  So the two ways the regex can be wrong on an unfamiliar architecture
both fetch a shard EARLIER than strictly needed (a vision tower's
`...encoder.layers.N.` keys pull their shard forward; an unmatched decoder
spelling pulls its shard into priority 0).  Neither can make the gate release a
layer whose bytes have not landed -- and if the plan were wrong in that
direction anyway, `layer_outer.audit_checkpoint_tree` refuses the short or
absent shard before a window is pushed through it.  The gate is an
optimisation; the audit is the guard.

Stdlib only, except for the DEFAULT downloader (`huggingface_hub`), which the
caller may replace -- and the tests do, which is what lets the scheduling be
measured offline in seconds instead of rented.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# The priority bucket for everything a forward pass needs at EVERY layer:
# embeddings, the final norm, the head, buffers.  Sorts before layer 0.
RESIDENT = -1

# `model.layers.N.`, `language_model.model.layers.N.`, and DeepSeek's bare
# `layers.N.` all match.  A checkpoint that spells its stack some other way
# passes --layer-key-regex; the failure mode of not matching is a slower fetch,
# never a wrong one (see the module docstring).
DEFAULT_LAYER_KEY_REGEX = r"(?:^|\.)layers\.(\d+)\."

# Files that are not shards but are needed before anything can be planned or
# loaded.  Fetched first, and they are kilobytes.
BOOTSTRAP_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
)


class RaceFetchError(Exception):
    """Something about this checkpoint or fetch the race scheduler will not guess at."""


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


class FetchPlan(object):
    """Which shard is needed when, derived from the checkpoint's own weight_map."""

    def __init__(self, needed_at: Dict[str, int], layer_shards: Dict[int, Set[str]],
                 layer_count: int, unmatched_keys: int, total_keys: int,
                 layer_key_regex: str):
        self.needed_at = dict(needed_at)
        self._layer_shards = {k: set(v) for k, v in layer_shards.items()}
        self.layer_count = int(layer_count)
        self.unmatched_keys = int(unmatched_keys)
        self.total_keys = int(total_keys)
        self.layer_key_regex = layer_key_regex

    # -- what the fetcher orders by ----------------------------------------

    def order(self) -> List[Tuple[int, str]]:
        """(priority, shard) in fetch order: resident first, then layer 0, 1, ..."""
        return sorted(((p, name) for name, p in self.needed_at.items()),
                      key=lambda row: (row[0], row[1]))

    # -- what the gate blocks on -------------------------------------------

    @property
    def resident_shards(self) -> Set[str]:
        return {name for name, p in self.needed_at.items() if p == RESIDENT}

    def shards_for_layer(self, index: int) -> Set[str]:
        """EVERY shard carrying a tensor of layer `index`.

        Not the same as "shards whose priority is `index`": a shard holding
        layers 7 and 8 has priority 7 and is needed again at 8.
        """
        return set(self._layer_shards.get(int(index), ()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "malaiwah.race-fetch-plan.v1",
            "layer_key_regex": self.layer_key_regex,
            "layer_count": self.layer_count,
            "shards": len(self.needed_at),
            "resident_shards": sorted(self.resident_shards),
            "unmatched_keys": self.unmatched_keys,
            "total_keys": self.total_keys,
            "order": [{"shard": name, "needed_at": p} for p, name in self.order()],
        }


def plan_shards(weight_map: Dict[str, str],
                layer_key_regex: str = DEFAULT_LAYER_KEY_REGEX) -> FetchPlan:
    """Bucket every shard by the FIRST layer that needs it.

    `weight_map` is `model.safetensors.index.json`'s own tensor -> shard map.
    A shard carrying any non-layer tensor is RESIDENT regardless of what else
    it carries, because the resident load happens before layer 0.
    """
    try:
        pattern = re.compile(layer_key_regex)
    except re.error as exc:
        raise RaceFetchError("--layer-key-regex %r does not compile: %s"
                             % (layer_key_regex, exc))
    if not weight_map:
        raise RaceFetchError(
            "the checkpoint index has an empty weight_map, so nothing can be "
            "planned. Race mode needs model.safetensors.index.json; a "
            "single-shard checkpoint has nothing to overlap and should run the "
            "ordinary fetch-then-capture path.")

    needed_at: Dict[str, int] = {}
    layer_shards: Dict[int, Set[str]] = {}
    unmatched = 0
    max_layer = -1
    for key, shard in weight_map.items():
        match = pattern.search(key)
        if match is None:
            unmatched += 1
            needed_at[shard] = RESIDENT
            continue
        index = int(match.group(1))
        max_layer = max(max_layer, index)
        layer_shards.setdefault(index, set()).add(shard)
        current = needed_at.get(shard)
        if current is None:
            needed_at[shard] = index
        elif current != RESIDENT:
            needed_at[shard] = min(current, index)
    # A shard that carries BOTH a resident tensor and layer tensors must stay
    # resident: the second pass below re-applies that, because dict order is
    # not priority order.
    for key, shard in weight_map.items():
        if pattern.search(key) is None:
            needed_at[shard] = RESIDENT
    return FetchPlan(needed_at, layer_shards, max_layer + 1, unmatched,
                     len(weight_map), layer_key_regex)


def read_index(model_dir: str) -> Dict[str, str]:
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(path):
        raise RaceFetchError(
            "race mode needs %s, and it is not present. It is the file that says "
            "which shard holds which layer; without it the fetch cannot be "
            "ordered and the ordinary fetch-then-capture path is the honest "
            "answer." % path)
    with open(path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    return index.get("weight_map") or {}


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


class ShardGate(object):
    """Block until named shards have LANDED, and account for the time spent.

    "Landed" is decided by the fetcher reporting a completed download, never by
    polling for a filename: a partially written file is exactly the failure this
    whole suite is built to refuse, and a poller cannot tell it from a finished
    one.  The fetcher writes through a temporary name and renames, so the two
    agree -- but the in-process record is the authority.
    """

    def __init__(self, log: Optional[Callable[..., None]] = None):
        self._lock = threading.Condition()
        self._landed: Set[str] = set()
        self._error: Optional[BaseException] = None
        self._log = log
        self.blocked_seconds = 0.0
        self.blocks: List[Dict[str, Any]] = []

    # -- fetcher side -------------------------------------------------------

    def mark(self, name: str) -> None:
        with self._lock:
            self._landed.add(name)
            self._lock.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc
            self._lock.notify_all()

    # -- loader side --------------------------------------------------------

    def landed(self, name: str) -> bool:
        with self._lock:
            return name in self._landed

    def wait_for(self, names: Iterable[str], timeout: float,
                 what: str = "shards") -> float:
        """Block until every name has landed. Returns the seconds actually waited.

        A timeout is a REFUSAL, not a proceed-anyway: continuing would hand the
        loader a shard that is absent or short, and a short safetensors shard
        reads as zeros rather than raising.
        """
        wanted = set(names)
        if not wanted:
            return 0.0
        started = time.monotonic()
        with self._lock:
            if self._error is not None:
                raise RaceFetchError("the background fetch failed: %s" % self._error)
            if wanted <= self._landed:
                return 0.0
            deadline = started + float(timeout)
            while True:
                missing = wanted - self._landed
                if self._error is not None:
                    raise RaceFetchError("the background fetch failed: %s" % self._error)
                if not missing:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RaceFetchError(
                        "race mode: waited %.0f s for %d %s that never landed (%s%s). "
                        "Refusing rather than reading a shard that is absent or short "
                        "-- a short safetensors shard does not raise, it reads as zeros."
                        % (time.monotonic() - started, len(missing), what,
                           ", ".join(sorted(missing)[:4]),
                           " +%d more" % (len(missing) - 4) if len(missing) > 4 else ""))
                self._lock.wait(min(remaining, 5.0))
        waited = time.monotonic() - started
        self.blocked_seconds += waited
        if waited > 0.0:
            self.blocks.append({"what": what, "seconds": round(waited, 3),
                                "shards": sorted(wanted)})
            if self._log is not None:
                self._log(stage="race_block", what=what, seconds=round(waited, 3),
                          shards=len(wanted))
        return waited


# ---------------------------------------------------------------------------
# the fetcher
# ---------------------------------------------------------------------------


def hf_downloader(repo_id: str, revision: Optional[str], dest: str,
                  token: Optional[str] = None) -> Callable[[str], str]:
    """The default file downloader: one `hf_hub_download` per file.

    Per-file rather than a whole-repo `hf download` for the obvious reason --
    a whole-repo download has no notion of "this shard first" -- and it keeps
    hf_transfer, which honours HF_HUB_ENABLE_HF_TRANSFER on this path too, so
    ordering is not bought with throughput.
    """
    from huggingface_hub import hf_hub_download

    def download(name: str) -> str:
        return hf_hub_download(repo_id=repo_id, filename=name, revision=revision,
                               local_dir=dest, token=token)

    return download


def simulated_downloader(source_dir: str, dest: str, seconds_per_file: float = 0.0
                         ) -> Callable[[str], str]:
    """THE OFFLINE TEST HARNESS. Copy from a local directory after a fixed delay.

    It exists because the thing race mode changes is a SCHEDULE, and a schedule
    cannot be A/B-tested on a real 1.5 TB fetch -- you get one arm, once, at
    whatever the link happened to be doing. With an injected per-file delay both
    arms run in seconds and the comparison is controlled.

    A run that used it is NOT a measurement of anything: the capture records
    `downloader: "simulated"` and stamps a blocking disclosure, so a dataset
    produced this way can never become a registry row.
    """
    def download(name: str) -> str:
        if seconds_per_file:
            time.sleep(seconds_per_file)
        src = os.path.join(source_dir, name)
        dst = os.path.join(dest, name)
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            tmp = dst + ".part"
            with open(src, "rb") as fh_in, open(tmp, "wb") as fh_out:
                while True:
                    block = fh_in.read(1 << 20)
                    if not block:
                        break
                    fh_out.write(block)
            # Rename, never write-in-place: a file that EXISTS must be a file
            # that is COMPLETE, or the audit is auditing a moving target.
            os.replace(tmp, dst)
        return dst

    return download


class RaceFetcher(object):
    """A priority-ordered background fetch with a gate the capture blocks on."""

    def __init__(self, plan: FetchPlan, download: Callable[[str], Any],
                 extra_files: Sequence[str] = (), workers: int = 8,
                 log: Optional[Callable[..., None]] = None,
                 sizes: Optional[Dict[str, int]] = None,
                 timeout: float = 7200.0,
                 trailing_files: Sequence[str] = ()):
        self.plan = plan
        self._download = download
        self._workers = max(1, int(workers))
        # The default every gate call uses, so the loader -- which knows nothing
        # about this run's flags -- cannot silently wait forever.
        self.timeout = float(timeout)
        self._log = log or (lambda **kw: None)
        self._sizes = dict(sizes or {})
        self.gate = ShardGate(log=self._log)
        self._queue: List[Tuple[int, int, str]] = []
        self._seq = 0
        self._qlock = threading.Lock()
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self.completed: List[Dict[str, Any]] = []
        self.started_monotonic: Optional[float] = None
        self.finished_monotonic: Optional[float] = None
        # Ahead of the resident shards: a file the loader needs before anything
        # else (a bootstrap file the caller wants re-confirmed). Re-queuing a
        # file already on disk costs one metadata request, and it keeps the
        # gate's record the single source of truth about what has landed.
        for name in extra_files:
            self._push(RESIDENT - 1, name)
        for priority, name in plan.order():
            self._push(priority, name)
        # AFTER every layer: the release's own sidecars -- SHA256SUMS, the
        # licence, the card. `fetch_target` pulls the whole repo, so race mode
        # must too or a race-mode tree would be missing the published seal the
        # verification step reads. Nothing waits on them, so they go last.
        for name in trailing_files:
            self._push(plan.layer_count + 1, name)

    # -- queue --------------------------------------------------------------

    def _push(self, priority: int, name: str) -> None:
        with self._qlock:
            self._seq += 1
            heapq.heappush(self._queue, (priority, self._seq, name))

    def _pop(self) -> Optional[Tuple[int, str]]:
        with self._qlock:
            if not self._queue:
                return None
            priority, _, name = heapq.heappop(self._queue)
            return priority, name

    @property
    def pending(self) -> int:
        with self._qlock:
            return len(self._queue)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "RaceFetcher":
        self.started_monotonic = time.monotonic()
        for index in range(self._workers):
            thread = threading.Thread(target=self._run, name="race-fetch-%d" % index,
                                      daemon=True)
            thread.start()
            self._threads.append(thread)
        self._log(stage="race_fetch_start", workers=self._workers,
                  files=len(self._queue))
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._pop()
            if item is None:
                return
            priority, name = item
            started = time.monotonic()
            try:
                self._download(name)
            except BaseException as exc:  # noqa: BLE001 - reported through the gate
                self.gate.fail(RaceFetchError("%s: %s" % (name, exc)))
                return
            seconds = time.monotonic() - started
            self.completed.append({"file": name, "needed_at": priority,
                                   "seconds": round(seconds, 3),
                                   "bytes": self._sizes.get(name)})
            self.gate.mark(name)
            self._log(stage="race_fetched", file=name, needed_at=priority,
                      seconds=round(seconds, 3), pending=self.pending)

    def join(self, timeout: Optional[float] = None) -> None:
        for thread in self._threads:
            thread.join(timeout)
        self.finished_monotonic = time.monotonic()

    def stop(self) -> None:
        self._stop.set()

    # -- what the capture calls --------------------------------------------

    def wait_for_shards(self, names: Iterable[str],
                        timeout: Optional[float] = None) -> float:
        """Block for an EXPLICIT shard set.

        The loader uses this rather than `wait_for_resident` because it knows
        something the plan does not: which shards the resident load will
        actually open, computed from the model's own stack prefix and the
        conversion mapping's renames. `plan.resident_shards` decides the fetch
        ORDER; this decides what is waited for.
        """
        return self.gate.wait_for(names,
                                  self.timeout if timeout is None else timeout,
                                  what="resident-set shards")

    def wait_for_resident(self, timeout: Optional[float] = None) -> float:
        return self.gate.wait_for(self.plan.resident_shards,
                                  self.timeout if timeout is None else timeout,
                                  what="resident-set shards")

    def wait_for_layer(self, index: int, timeout: Optional[float] = None) -> float:
        return self.gate.wait_for(self.plan.shards_for_layer(index),
                                  self.timeout if timeout is None else timeout,
                                  what="layer %d shards" % index)

    # -- the honest accounting ---------------------------------------------

    def report(self) -> Dict[str, Any]:
        """What the overlap actually bought, measured, never asserted.

        `serial_fetch_seconds` is the sum of every file's own download wall
        clock divided by the worker count -- the fetch this run would have paid
        as a barrier, at the same parallelism -- and is therefore an ESTIMATE
        from this run's own timings, labelled as one.  `blocked_seconds` is
        measured directly and is the part the overlap failed to hide.
        """
        total = sum(row["seconds"] for row in self.completed)
        wall = None
        if self.started_monotonic is not None:
            end = self.finished_monotonic or time.monotonic()
            wall = round(end - self.started_monotonic, 3)
        return {
            "schema": "malaiwah.race-fetch-report.v1",
            "files": len(self.completed),
            "workers": self._workers,
            "fetch_thread_seconds_total": round(total, 3),
            "fetch_wall_seconds": wall,
            "blocked_seconds": round(self.gate.blocked_seconds, 3),
            "blocks": self.gate.blocks,
            "serial_fetch_seconds_estimate": round(total / self._workers, 3),
            "note": "blocked_seconds is measured: the time the capture spent "
                    "waiting for bytes it needed next. fetch_wall_seconds is the "
                    "background fetch's own wall clock. The saving claimed by a "
                    "race run is (fetch_wall + capture_wall) - race_wall, and the "
                    "control arm that supplies the first term must be RUN, not "
                    "assumed.",
        }
