"""The single chokepoint for every JarvisLabs call.

WHY THE CLI AND NOT THE REST API.  `jl` owns auth, region selection, the
spot-vs-container rules, ssh-key plumbing and upload/download/exec over SSH.
The REST surface behind it is not publicly documented -- the vendor documents
the CLI -- so reimplementing it would make this recipe a maintenance liability,
which is the opposite of "a standard anyone can run".

The cost of that choice is CLI drift.  It is paid down here rather than spread
through the runner: EVERY invocation goes through `JL._call`, which appends
`--json`, parses stdout, and normalises the vendor's `{"error": ...}` shape
into an exception.  A future `--transport api` is a one-function swap, and a
`jl` version bump has exactly one place to break.

Install:  uv tool install jarvislabs        (or pipx install jarvislabs)
Auth:     jl setup --token <token> --yes    (or export JL_API_KEY=...)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .common import CommandError, redact, register_secret, run

MIN_VERSION = (0, 2, 17)


class JLError(RuntimeError):
    pass


class JLNotInstalled(JLError):
    pass


def _parse_version(text: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)


@dataclass
class Instance:
    machine_id: int
    status: str
    gpu_type: Optional[str]
    num_gpus: int
    region: Optional[str]
    is_spot: bool
    cost: float          # RUNNING USD TOTAL, not a rate -- see `billed_usd`
    runtime: Any
    fs_id: Optional[int]
    storage_gb: Optional[int]
    name: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Instance":
        return cls(
            machine_id=int(d.get("machine_id", 0)),
            status=str(d.get("status", "")),
            gpu_type=d.get("gpu_type"),
            num_gpus=int(d.get("num_gpus") or 0),
            region=d.get("region"),
            is_spot=bool(d.get("is_spot")),
            cost=float(d.get("cost") or 0.0),
            runtime=d.get("runtime"),
            fs_id=d.get("fs_id"),
            storage_gb=d.get("storage_gb"),
            name=d.get("name"),
            raw=d,
        )

    @property
    def billed_usd(self) -> float:
        """The accumulated dollar total for this instance so far.

        NOTE, and this contradicts an earlier note in k6/HANDOFF.md: `jl get`'s
        `cost` field is a running TOTAL in USD, not an hourly rate.  Verified
        by reconciling live instances against the published rates -- e.g. an
        8x H200 spot box at 2h33m reported 40.897, which is $16.04/h against a
        list rate of 8 x $1.99 = $15.92/h.  A cost model built on the "rate"
        reading would be wrong by a factor of the elapsed hours.
        """
        return self.cost


@dataclass
class GpuOffer:
    gpu_type: str
    region: Optional[str]
    vram_bytes: float
    price: float
    spot: bool
    free_devices: int
    workload_type: Optional[str]
    raw: Dict[str, Any]


class JL:
    """Thin, auditable wrapper.  `dry` short-circuits every mutating call."""

    def __init__(self, *, dry: bool = False, binary: str = "jl",
                 timeout: float = 300.0) -> None:
        self.binary = binary
        self.dry = dry
        self.timeout = timeout
        self._version: Optional[tuple] = None
        register_secret(os.environ.get("JL_API_KEY"))

    # ---- plumbing ---------------------------------------------------------

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def require(self) -> tuple:
        if not self.available():
            raise JLNotInstalled(
                "the `jl` CLI is not on PATH.\n"
                "  install:  uv tool install jarvislabs\n"
                "            (or: pipx install jarvislabs)\n"
                "  auth:     jl setup --token <your-token> --yes\n"
                "            (or: export JL_API_KEY=...)"
            )
        if self._version is None:
            proc = run([self.binary, "--version"], timeout=30, check=False)
            self._version = _parse_version(proc.stdout or proc.stderr)
        if self._version < MIN_VERSION:
            raise JLError(
                "jl %s is older than the pinned minimum %s; upgrade with "
                "`uv tool upgrade jarvislabs`"
                % (".".join(map(str, self._version)), ".".join(map(str, MIN_VERSION)))
            )
        return self._version

    @property
    def version(self) -> str:
        return ".".join(map(str, self._version or (0, 0, 0)))

    def _call(self, argv: Sequence[str], *, mutating: bool = False,
              timeout: Optional[float] = None, check: bool = True) -> Any:
        """Every jl invocation lands here.  Nothing else may shell out to jl."""
        if mutating and self.dry:
            return {"dry_run": True, "argv": list(argv)}
        cmd = [self.binary] + list(argv)
        if "--json" not in cmd:
            cmd.append("--json")
        if mutating and "--yes" not in cmd:
            cmd.append("--yes")
        try:
            proc = run(cmd, timeout=timeout or self.timeout, check=False)
        except Exception as exc:                      # noqa: BLE001
            raise JLError("jl invocation failed: %s" % redact(str(exc))) from None
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 and not out:
            raise JLError(
                "jl %s exited %d: %s"
                % (" ".join(argv[:2]), proc.returncode, redact(proc.stderr or "")[:400])
            )
        try:
            data = json.loads(out) if out else {}
        except json.JSONDecodeError:
            if check and proc.returncode != 0:
                raise JLError(
                    "jl %s exited %d with non-JSON output: %s"
                    % (" ".join(argv[:2]), proc.returncode, redact(out)[:400])
                ) from None
            return out
        if isinstance(data, dict) and data.get("error"):
            raise JLError("jl %s: %s" % (" ".join(argv[:2]), redact(str(data["error"]))))
        return data

    # ---- read-only --------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return self._call(["status"])

    def balance(self) -> Optional[float]:
        try:
            data = self.status()
        except JLError:
            return None
        bal = data.get("balance")
        if isinstance(bal, dict):
            bal = bal.get("balance")
        try:
            return float(bal)
        except (TypeError, ValueError):
            return None

    def list_instances(self) -> List[Instance]:
        data = self._call(["list"])
        rows = data if isinstance(data, list) else data.get("instances", [])
        return [Instance.from_json(r) for r in rows or []]

    def get(self, machine_id: int) -> Optional[Instance]:
        try:
            data = self._call(["get", str(machine_id)])
        except JLError:
            return None
        if isinstance(data, list):
            data = data[0] if data else {}
        return Instance.from_json(data) if data else None

    def gpus(self) -> List[GpuOffer]:
        data = self._call(["gpus"])
        rows = data if isinstance(data, list) else data.get("gpus", [])
        offers: List[GpuOffer] = []
        for r in rows or []:
            vram = r.get("vram") or r.get("gpu_ram") or 0
            spot_price = r.get("spot_price")
            free = r.get("num_free_devices")
            if free is None:
                free = r.get("effective_num_free_devices") or 0
            base = {
                "gpu_type": r.get("gpu_type") or r.get("name") or "?",
                "region": r.get("region"),
                "vram_bytes": float(vram) * 1e9,
                "free_devices": int(free or 0),
                "workload_type": r.get("workload_type"),
                "raw": r,
            }
            on_demand = r.get("price") or r.get("on_demand_price")
            if on_demand is not None:
                offers.append(GpuOffer(price=float(on_demand), spot=False, **base))
            if spot_price is not None:
                offers.append(GpuOffer(price=float(spot_price), spot=True, **base))
        return offers

    # ---- mutating ---------------------------------------------------------

    def create(self, **kw) -> Dict[str, Any]:
        argv = ["create"]
        for key, value in kw.items():
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            argv.append(flag) if value is True else argv.extend([flag, str(value)])
        return self._call(argv, mutating=True, timeout=900)

    def destroy(self, machine_id: int) -> Dict[str, Any]:
        return self._call(["destroy", str(machine_id)], mutating=True, timeout=600)

    def pause(self, machine_id: int) -> Dict[str, Any]:
        return self._call(["pause", str(machine_id)], mutating=True, timeout=600)

    def resume(self, machine_id: int, *, spot: bool = False) -> Dict[str, Any]:
        argv = ["resume", str(machine_id)]
        if spot:
            argv.append("--spot")
        return self._call(argv, mutating=True, timeout=900)

    def exec(self, machine_id: int, command: str, *,
             timeout: float = 600) -> Any:
        return self._call(["exec", str(machine_id), command],
                          mutating=True, timeout=timeout)

    def upload(self, machine_id: int, local: str, remote: str) -> Any:
        return self._call(["upload", str(machine_id), local, remote],
                          mutating=True, timeout=1800)

    def download(self, machine_id: int, remote: str, local: str,
                 *, recursive: bool = True, timeout: float = 900) -> Any:
        argv = ["download", str(machine_id), remote, local]
        if recursive:
            argv.append("-r")
        return self._call(argv, mutating=True, timeout=timeout)

    def run_job(self, machine_id: int, command: str) -> Any:
        return self._call(["run", command, "--on", str(machine_id)],
                          mutating=True, timeout=600)

    def run_logs(self, run_id: str, *, tail: int = 50) -> Any:
        return self._call(["run", "logs", str(run_id), "--tail", str(tail)],
                          timeout=120, check=False)

    def fs_create(self, *, storage: int, region: str, name: Optional[str] = None) -> Any:
        argv = ["filesystem", "create", "--storage", str(storage), "--region", region]
        if name:
            argv.extend(["--name", name])
        return self._call(argv, mutating=True, timeout=600)

    def fs_delete(self, fs_id: int) -> Any:
        return self._call(["filesystem", "delete", str(fs_id)],
                          mutating=True, timeout=600)

    def fs_list(self) -> Any:
        return self._call(["filesystem", "list"])


def select_offer(
    offers: Sequence[GpuOffer],
    *,
    required_vram_bytes: float,
    gpus: int,
    spot: bool,
    gpu_type: Optional[str] = None,
    region: Optional[str] = None,
) -> tuple:
    """Cheapest offer that actually fits, plus the full audited candidate table.

    Filter order is deliberate.  VRAM first, because a row that does not fit is
    not a candidate at any price; then the spot rule (spot is GPU CONTAINERS
    only, never VMs); then capacity; then price.  Every row and its verdict
    goes into the receipt, so the choice is auditable after the fact rather
    than a number that appeared from nowhere.
    """
    table = []
    viable = []
    for o in offers:
        verdict = "ok"
        if gpu_type and o.gpu_type.lower() != gpu_type.lower():
            verdict = "not requested"
        elif region and (o.region or "") != region:
            verdict = "wrong region"
        elif o.vram_bytes < required_vram_bytes:
            verdict = "too small (%.0f < %.0f GB)" % (
                o.vram_bytes / 1e9, required_vram_bytes / 1e9)
        elif spot and not o.spot:
            verdict = "on-demand row, --spot requested"
        elif not spot and o.spot:
            verdict = "spot row, --on-demand requested"
        elif spot and (o.workload_type not in (None, "", "container")):
            verdict = "spot is containers only (workload_type=%s)" % o.workload_type
        elif o.free_devices < gpus:
            verdict = "no capacity (%d free, need %d)" % (o.free_devices, gpus)
        table.append({
            "gpu_type": o.gpu_type, "region": o.region,
            "vram_gb": round(o.vram_bytes / 1e9, 0), "price": o.price,
            "spot": o.spot, "free": o.free_devices, "verdict": verdict,
        })
        if verdict == "ok":
            viable.append(o)
    if not viable:
        return None, table
    viable.sort(key=lambda o: (o.price * gpus, -o.vram_bytes))
    return viable[0], table
