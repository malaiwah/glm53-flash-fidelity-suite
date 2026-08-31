#!/usr/bin/env python3
"""Lambda Cloud backend, duck-typed to `fidelity.jlapi.JL`.

Lambda is the simplest of the three and the least flexible, and both halves of
that matter to this suite:

* **No spot, no bidding.** One published on-demand price per instance type. It
  is the most expensive per GPU-hour of the providers wired up here, and the
  most predictable -- which makes it the right place to run something that must
  not be interrupted, and the wrong place to run something cheap.
* **Storage is NOT selectable.** Every other backend takes the disk size the
  plan computed. A Lambda instance type comes with whatever local disk it comes
  with, so `fs_create` cannot honour a request; it records what was asked for
  and the caller must check the type actually fits. A measurement that needs
  300 GB and lands on a type with less will fail during fetch, after the money
  starts -- so `create` refuses up front when the requested size exceeds what
  the type is known to provide.
* **Instances are region-pinned and capacity is bursty.** `instance-types`
  reports `regions_with_capacity_available`, and launching into a region that
  is not in that list fails. It is queried per launch rather than cached.

SSH keys are per-account and must already be registered by NAME; Lambda does
not accept an inline public key at launch. The user is `ubuntu`, not `root`.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

API = "https://cloud.lambdalabs.com/api/v1"

# Lambda publishes each type's real local disk as `specs.storage_gib`, so it is
# READ, never guessed. It was guessed once, from a hardcoded table that had
# `gpu_1x_a10` at 200 GB by confusing storage_gib with memory_gib -- and a plan
# needing 400 GB was refused on a machine whose root filesystem measured 1.4 TB.
# A false refusal is cheap to notice and expensive to trust, so the table is
# gone and the only fallback is "unknown", which does not refuse.


class LambdaError(JLError):
    pass


class LambdaCloud(SSHTransport):
    # This provider has NO filesystem that outlives its instance, so the
    # whole run must fit on the instance's own disk: lambda disk is fixed per instance type.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False
    provider = "lambda"
    ssh_user = "ubuntu"
    RUNS = "/home/ubuntu/.fidruns"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None,
                 ssh_key_names: Optional[List[str]] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self.ssh_key_names = ssh_key_names
        self._ep: Dict[str, tuple] = {}

    def _load_key(self) -> str:
        if self._key:
            return self._key
        path = self._key_file or os.environ.get("LAMBDA_KEY_FILE") or ""
        if path and os.path.isfile(path):
            self._key = open(path, encoding="utf-8").read().strip()
        else:
            self._key = os.environ.get("LAMBDA_API_KEY", "").strip()
        if not self._key:
            raise LambdaError("no Lambda credential: set LAMBDA_KEY_FILE to a "
                              "0600 file, or LAMBDA_API_KEY")
        return self._key

    def _req(self, method: str, path: str, body: Any = None,
             *, timeout: float = 90) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        # HTTP Basic with the key as the username and an empty password.
        token = base64.b64encode((self._load_key() + ":").encode()).decode()
        req = urllib.request.Request(
            API + path, data=data, method=method,
            headers={"Content-Type": "application/json",
                     "User-Agent": "quant-fidelity-suite/0.1",
                     "Authorization": "Basic " + token})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LambdaError("Lambda HTTP %d on %s: %s"
                              % (exc.code, path,
                                 redact(exc.read()[:300].decode("utf-8", "replace"))))
        except Exception as exc:                          # noqa: BLE001
            raise LambdaError("Lambda request failed: %s" % redact(str(exc)))
        return json.loads(raw) if raw.strip() else {}

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            self._load_key()
            return True
        except LambdaError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise LambdaError("Lambda credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "lambda-api-v1"

    def status(self) -> Dict[str, Any]:
        return {"instance_types": len(self._req("GET", "/instance-types")
                                      .get("data", {}))}

    def balance(self) -> Optional[float]:
        # Lambda publishes no balance endpoint: it is pay-as-you-go and bills
        # after the fact. Returning None is honest; inventing a number here
        # would make the controller's "can this account pay?" check a lie.
        return None

    def ssh_key_names_available(self) -> List[str]:
        return [k.get("name") for k in
                self._req("GET", "/ssh-keys").get("data", [])]

    # -- catalogue ---------------------------------------------------------
    def gpus(self) -> List[GpuOffer]:
        data = self._req("GET", "/instance-types").get("data", {})
        offers = []
        for name, v in data.items():
            it = v.get("instance_type") or {}
            regions = v.get("regions_with_capacity_available") or []
            specs = it.get("specs") or {}
            gpus = int(specs.get("gpus") or 1)
            vram = float(it.get("gpu_description", "0").split("(")[-1]
                         .split("GB")[0].strip() or 0) if "GB" in (
                             it.get("gpu_description") or "") else 0.0
            for r in (regions or [None]):
                offers.append(GpuOffer(
                    gpu_type=name,
                    region=(r or {}).get("name") if isinstance(r, dict) else r,
                    vram_bytes=vram * (1024 ** 3),
                    price=float(it.get("price_cents_per_hour") or 0) / 100.0,
                    spot=False,
                    free_devices=1 if regions else 0,
                    workload_type="vm",
                    raw={"gpus": gpus,
                         "disk_gb": (specs.get("storage_gib")),
                         "description": it.get("gpu_description"),
                         "available": bool(regions)}))
        return offers

    # -- instances ---------------------------------------------------------
    @staticmethod
    def _to_instance(d: Dict[str, Any]) -> Instance:
        it = d.get("instance_type") or {}
        inst = Instance.from_json({
            "machine_id": 0, "status": d.get("status") or "",
            "gpu_type": it.get("name"),
            "num_gpus": int(((it.get("specs") or {}).get("gpus")) or 1),
            "region": (d.get("region") or {}).get("name"),
            "is_spot": False, "cost": 0.0, "runtime": None, "fs_id": None,
            "storage_gb": ((it.get("specs") or {}).get("storage_gib")),
            "name": d.get("name"),
        })
        inst.machine_id = d.get("id")
        inst.raw["ip"] = d.get("ip")
        inst.raw["price_cents_per_hour"] = it.get("price_cents_per_hour")
        return inst

    def list_instances(self) -> List[Instance]:
        return [self._to_instance(d)
                for d in self._req("GET", "/instances").get("data", [])]

    def get(self, machine_id: Any) -> Optional[Instance]:
        for i in self.list_instances():
            if str(i.machine_id) == str(machine_id):
                return i
        return None

    def create(self, **kw) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True, **kw}
        itype = kw.get("gpu_type") or kw.get("instance_type")
        if not itype:
            raise LambdaError("create requires gpu_type (a Lambda instance type)")
        want_disk = int(kw.get("storage") or kw.get("storage_gb") or 0)
        types_now = self._req("GET", "/instance-types").get("data", {})
        have = (((types_now.get(itype) or {}).get("instance_type") or {})
                .get("specs") or {}).get("storage_gib")
        if want_disk and have and want_disk > have:
            raise LambdaError(
                "instance type %s provides ~%d GB of local disk and this plan "
                "needs %d GB. Lambda disk is fixed per type and cannot be "
                "grown, so this would fail during fetch, after billing starts."
                % (itype, have, want_disk))
        names = self.ssh_key_names or self.ssh_key_names_available()
        if not names:
            raise LambdaError(
                "no SSH key registered on the Lambda account. Lambda attaches "
                "keys BY NAME at launch and accepts no inline public key, so "
                "one must be added in the console first.")
        types = self._req("GET", "/instance-types").get("data", {})
        regions = (types.get(itype) or {}).get(
            "regions_with_capacity_available") or []
        if not regions:
            raise LambdaError("instance type %s has no region with capacity "
                              "right now" % itype)
        region = kw.get("region") or regions[0].get("name")
        got = self._req("POST", "/instance-operations/launch", {
            "region_name": region, "instance_type_name": itype,
            "ssh_key_names": names[:1], "quantity": 1,
            "name": kw.get("name") or "fidcloud"}, timeout=180)
        ids = (got.get("data") or {}).get("instance_ids") or []
        if not ids:
            raise LambdaError("Lambda returned no instance id: %s"
                              % redact(json.dumps(got)[:300]))
        return {"machine_id": ids[0], "region": region}

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._req("POST", "/instance-operations/terminate",
                  {"instance_ids": [str(machine_id)]}, timeout=180)
        return {"terminated": str(machine_id)}

    def pause(self, machine_id: Any) -> Dict[str, Any]:
        raise LambdaError("Lambda has no pause: an instance is running or "
                          "terminated. Use destroy().")

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        raise LambdaError("Lambda has no resume; a terminated instance is gone.")

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        key = str(machine_id)
        if key in self._ep:
            return self._ep[key]
        deadline = time.time() + wait
        while time.time() < deadline:
            inst = self.get(machine_id)
            if inst is not None and inst.raw.get("ip") \
                    and str(inst.status).lower() in ("active", "running"):
                self._ep[key] = (inst.raw["ip"], 22)
                return self._ep[key]
            time.sleep(10)
        raise LambdaError("instance %s never became reachable within %ds"
                          % (machine_id, int(wait)))

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "lambda disk is fixed per instance type and cannot be "
                        "requested; create() refuses a type too small for it"}

    def fs_delete(self, fs_id: Any) -> Any:
        return {"deleted": False, "note": "no separable filesystem on lambda"}
