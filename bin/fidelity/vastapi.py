#!/usr/bin/env python3
"""Vast.ai backend, duck-typed to `fidelity.jlapi.JL`.

Vast is a MARKETPLACE, not a fleet, and that is the whole character of this
backend. You do not ask for "an A100"; you bid on one specific machine some
specific person owns, with its own disk, its own uplink and its own driver
stack. Two consequences the rest of the suite has to know about:

* **Offers are per-host, and an offer id is not a GPU model.** `create` takes
  the `ask_id` of a bundle that was searched for, so the search and the create
  are one transaction. An offer that vanishes between them is normal.
* **Bitwise determinism claims do not travel here.** docs/CLOUD-PROVIDERS.md
  §3 says it plainly: the cheapness comes from renting whatever a host happens
  to own -- different drivers, different host CPUs, sometimes different silicon
  under one GPU name. That is fine for work whose output is content-digested
  and verified (`verify` recomputes the whole chain before teardown) and is the
  wrong place to ESTABLISH a determinism result.

Disk is chosen at rent time and cannot grow, so `create` asks for the size the
plan computed and refuses an offer that cannot hold it.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

API = "https://console.vast.ai/api/v0"
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"


class VastError(JLError):
    pass


class Vast(SSHTransport):
    # This provider has NO filesystem that outlives its instance, so the
    # whole run must fit on the instance's own disk: vast disk is chosen at rent time and dies with the instance.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False
    provider = "vast"
    ssh_user = "root"
    RUNS = "/workspace/.fidruns"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self._ep: Dict[str, tuple] = {}

    # -- transport ---------------------------------------------------------
    def _load_key(self) -> str:
        if self._key:
            return self._key
        path = self._key_file or os.environ.get("VAST_KEY_FILE") or ""
        if path and os.path.isfile(path):
            self._key = open(path, encoding="utf-8").read().strip()
        else:
            self._key = os.environ.get("VAST_API_KEY", "").strip()
        if not self._key:
            raise VastError("no Vast credential: set VAST_KEY_FILE to a 0600 "
                            "file, or VAST_API_KEY")
        return self._key

    # Vast enforces roughly one API request per second and answers HTTP 429
    # with a `retry_after`. The banded catalogue search fires five queries back
    # to back, which tripped it immediately -- and it tripped INSIDE the run,
    # after the lease was written, so a rate limit read as a failed run.
    _MIN_INTERVAL = 1.1
    _last_call = 0.0

    def _req(self, method: str, path: str, body: Any = None,
             *, timeout: float = 90, _tries: int = 4) -> Any:
        gap = time.time() - Vast._last_call
        if gap < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - gap)
        Vast._last_call = time.time()
        url = API + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "quant-fidelity-suite/0.1",
                                              "Authorization": "Bearer " + self._load_key()})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            payload = exc.read()[:300].decode("utf-8", "replace")
            if exc.code == 429 and _tries > 1:
                wait = 2.0
                try:
                    wait = max(1.0, float(json.loads(payload).get("retry_after") or 1)) + 1.0
                except Exception:                         # noqa: BLE001
                    pass
                time.sleep(wait)
                return self._req(method, path, body, timeout=timeout,
                                 _tries=_tries - 1)
            raise VastError("Vast HTTP %d on %s: %s"
                            % (exc.code, path, redact(payload)))
        except Exception as exc:                          # noqa: BLE001
            raise VastError("Vast request failed: %s" % redact(str(exc)))
        return json.loads(raw) if raw.strip() else {}

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            self._load_key()
            return True
        except VastError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise VastError("Vast credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "vast-rest-v0"

    def status(self) -> Dict[str, Any]:
        return self._req("GET", "/users/current/") or {}

    def balance(self) -> Optional[float]:
        try:
            return float(self.status().get("credit"))
        except Exception:                                 # noqa: BLE001
            return None

    # -- catalogue ---------------------------------------------------------
    # Vast has tens of thousands of live offers and the API returns them
    # cheapest-first. A single unconstrained search therefore returns nothing
    # but 6-13 GB consumer cards, and the controller -- which does its own
    # VRAM filtering on whatever list it is handed -- concluded that Vast had
    # no instance able to hold the model. The catalogue is assembled from
    # SEVERAL banded searches so every tier is represented.
    _VRAM_BANDS = (0, 24, 48, 63, 80)

    def gpus(self, *, min_vram_gb: int = 0, min_disk_gb: int = 300,
             limit: int = 40) -> List[GpuOffer]:
        if not min_vram_gb:
            seen, merged = set(), []
            for band in self._VRAM_BANDS:
                for o in self._search(band, min_disk_gb, max(8, limit // 4)):
                    if o.raw["ask_id"] in seen:
                        continue
                    seen.add(o.raw["ask_id"])
                    merged.append(o)
            return merged
        return self._search(min_vram_gb, min_disk_gb, limit)

    def _search(self, min_vram_gb: int, min_disk_gb: int, limit: int,
                gpu_name: Optional[str] = None) -> List[GpuOffer]:
        q = {"rentable": {"eq": True}, "num_gpus": {"eq": 1},
             "disk_space": {"gte": int(min_disk_gb)},
             "order": [["dph_total", "asc"]], "limit": int(limit),
             "type": "on-demand"}
        if min_vram_gb:
            q["gpu_ram"] = {"gte": int(min_vram_gb) * 1024}
        if gpu_name:
            # Ask Vast for the card BY NAME rather than filtering a generic
            # list. The banded catalogue returns the cheapest few per VRAM
            # band, so a specifically-requested 20 GB card routinely is not in
            # it and "no offer for RTX A4500" was reported while dozens were
            # rentable.
            q["gpu_name"] = {"eq": gpu_name}
        got = self._req("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
        offers = []
        for o in (got or {}).get("offers", []):
            offers.append(GpuOffer(
                gpu_type=o.get("gpu_name") or "?",
                region=(o.get("geolocation") or "").strip() or None,
                vram_bytes=float(o.get("gpu_ram") or 0) * (1024 ** 2),
                price=float(o.get("dph_total") or 0), spot=False,
                free_devices=1, workload_type="container",
                # the ask id is the only thing `create` can act on
                raw={"ask_id": o.get("id"), "disk_space": o.get("disk_space"),
                     "cuda": o.get("cuda_max_good"),
                     "inet_down": o.get("inet_down"),
                     "reliability": o.get("reliability2")}))
        return offers

    # -- instances ---------------------------------------------------------
    @staticmethod
    def _to_instance(d: Dict[str, Any]) -> Instance:
        inst = Instance.from_json({
            "machine_id": 0,
            "status": d.get("actual_status") or d.get("cur_state") or "",
            "gpu_type": d.get("gpu_name"), "num_gpus": d.get("num_gpus") or 1,
            "region": d.get("geolocation"), "is_spot": False,
            "cost": float(d.get("dph_total") or 0)
            * float(d.get("duration") or 0) / 3600.0,
            "runtime": d.get("duration"), "fs_id": None,
            "storage_gb": d.get("disk_space"), "name": d.get("label"),
        })
        inst.machine_id = d.get("id")
        inst.raw["ssh_host"] = d.get("ssh_host")
        inst.raw["ssh_port"] = d.get("ssh_port")
        # The CONTRACT rate, not the ask's. On a marketplace those are two
        # different objects: the ask you searched can be gone by the time the
        # rental lands, and an ask id is not a durable name for one machine --
        # one that advertised a B200 handed back an H100. Anything that prices
        # a run must read what is billing, not what was listed.
        inst.raw["dph_total"] = d.get("dph_total")
        inst.raw["gpu_name"] = d.get("gpu_name")
        return inst

    def list_instances(self) -> List[Instance]:
        got = self._req("GET", "/instances/") or {}
        return [self._to_instance(d) for d in got.get("instances", [])]

    def get(self, machine_id: Any) -> Optional[Instance]:
        for i in self.list_instances():
            if str(i.machine_id) == str(machine_id):
                return i
        return None

    def create(self, **kw) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True, **kw}
        ask = kw.get("ask_id") or kw.get("offer_id")
        disk = int(kw.get("storage") or kw.get("storage_gb") or 100)
        if not ask:
            # No ask id supplied: search now for the cheapest bundle that fits.
            # Searching and renting must be one transaction on a marketplace --
            # an offer that vanishes in between is ordinary, not an error.
            want = (kw.get("gpu_type") or kw.get("gpu") or "").strip()
            fits = []
            if want:
                # exact name first, then a substring pass over the catalogue
                fits = self._search(int(kw.get("min_vram_gb") or 0), disk, 20,
                                    gpu_name=want)
            if not fits:
                fits = self.gpus(min_vram_gb=int(kw.get("min_vram_gb") or 0),
                                 min_disk_gb=disk)
            if want:
                # HONOUR the requested GPU. Without this the "cheapest that
                # fits" is whatever the marketplace is dumping -- on this
                # account that was a CMP 170HX, a 64 GB MINING card that
                # satisfies a >=63 GB VRAM filter and is useless for this work.
                # The controller already chose a model; renting a different one
                # silently would make `on_validated_hardware` a lie.
                fits = [o for o in fits
                        if want.lower() in (o.gpu_type or "").lower()]
            if not fits:
                raise VastError(
                    "no rentable Vast offer for %s with >=%d GB VRAM and >=%d GB "
                    "disk" % (want or "any GPU",
                              int(kw.get("min_vram_gb") or 0), disk))
        # Container-native mode.  Vast's `runtype: "args"` preserves the
        # image ENTRYPOINT and passes `args` as CMD -- but has no post-start
        # hook, so preparation (target.json, tokenizer, panel binding) cannot
        # run before the capture.  `runtype: "ssh"` replaces the entrypoint
        # with sshd and runs `onstart` AFTER init -- so the full command
        # (prep + entrypoint) goes in `onstart` as a shell script.  The
        # container stays alive for SSH after the script exits; we destroy
        # it when the result arrives.  Secrets travel in `env`, never in
        # onstart text: a provider may echo the command back, but environment
        # variables it does not.  Triggered by `docker_cmd`; when absent the
        # SSH path below is byte-identical.
        docker_cmd = kw.get("docker_cmd")
        if docker_cmd is not None:
            onstart = kw.get("onstart") or ""
            # If onstart is supplied it is a prep script; the docker_cmd
            # (the capture argv) is appended after it so both run in one
            # shell.  If onstart is empty, docker_cmd runs alone.
            if onstart and docker_cmd:
                exec_line = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
                full = onstart + "\n" + exec_line
            elif docker_cmd:
                full = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
            else:
                full = onstart
            # Vast limits onstart to 4048 chars.  gzip+base64 the prep
            # script and decode it at runtime when the combined text is
            # too long (Vast's own documented workaround).
            if len(full) > 4048 and onstart and docker_cmd:
                import gzip as _gz
                compressed = _gz.compress(onstart.encode("utf-8"))
                encoded = base64.b64encode(compressed).decode("ascii")
                exec_line = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
                full = (
                    "echo '%s' | base64 -d | gunzip > /workspace/prep.sh "
                    "&& bash /workspace/prep.sh\n" % encoded) + exec_line
            # Vast's REST API takes `env` as a string in Docker flag
            # format (e.g. "-e KEY=VAL -p 8000:8000"), not a plain dict.
            # The CLI's parse_env converts this to a dict internally, but
            # the PUT body expects the string form.  Secrets stay under env,
            # never in onstart or args.
            env_dict = kw.get("env") or {}
            env_str = " ".join("-e %s=%s" % (k, v) for k, v in env_dict.items())
            body = {"client_id": "me",
                    "image": kw.get("image") or DEFAULT_IMAGE,
                    "disk": disk,
                    "label": kw.get("name") or "fidcloud",
                    "runtype": "ssh",
                    "onstart": full,
                    "env": env_str}
            got = self._req("PUT", "/asks/%s/" % ask, body, timeout=180)
            if not got.get("success"):
                raise VastError("Vast refused the rental: %s"
                                % redact(json.dumps(got)[:300]))
            return {"machine_id": got.get("new_contract"), "ask_id": ask}
        pub = ""
        kp = self.ssh_key + ".pub"
        if os.path.isfile(kp):
            pub = open(kp, encoding="utf-8").read().strip()
        body = {"client_id": "me", "image": kw.get("image") or DEFAULT_IMAGE,
                "disk": disk, "label": kw.get("name") or "fidcloud",
                "runtype": "ssh", "onstart": "", "env": {}}
        if pub:
            body["extra_env"] = {"PUBLIC_KEY": pub}
        got = self._req("PUT", "/asks/%s/" % ask, body, timeout=180)
        if not got.get("success"):
            raise VastError("Vast refused the rental: %s"
                            % redact(json.dumps(got)[:300]))
        cid = got.get("new_contract")
        if pub:
            # Vast attaches keys per-instance, not per-account.
            try:
                self._req("POST", "/instances/%s/ssh/" % cid, {"ssh_key": pub})
            except VastError:
                pass
        return {"machine_id": cid, "ask_id": ask}

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._req("DELETE", "/instances/%s/" % machine_id, {})
        return {"terminated": str(machine_id)}

    def pause(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        return self._req("PUT", "/instances/%s/" % machine_id, {"state": "stopped"})

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        return self._req("PUT", "/instances/%s/" % machine_id, {"state": "running"})

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        key = str(machine_id)
        if key in self._ep:
            return self._ep[key]
        deadline = time.time() + wait
        while time.time() < deadline:
            inst = self.get(machine_id)
            if inst is not None:
                host = inst.raw.get("ssh_host")
                port = inst.raw.get("ssh_port")
                if host and port and str(inst.status).lower().startswith("run"):
                    # `running` is the CONTRACT's state, not sshd's.
                    self._await_ssh(host, int(port),
                                    wait=max(60.0, deadline - time.time()))
                    self._ep[key] = (host, int(port))
                    return self._ep[key]
            time.sleep(10)
        raise VastError("instance %s never reported a running SSH endpoint "
                        "within %ds" % (machine_id, int(wait)))

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "vast disk is chosen at rent time and dies with the "
                        "instance; requested via create(disk=)"}

    def fs_delete(self, fs_id: Any) -> Any:
        return {"deleted": False, "note": "no separable filesystem on vast"}
