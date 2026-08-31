#!/usr/bin/env python3
"""RunPod backend, duck-typed to `fidelity.jlapi.JL`.

Why duck-typed rather than a refactor
-------------------------------------
`measure_cloud.py` touches a provider through exactly eighteen methods, and
everything else it does -- the fit check, the cost band, the lease, all four
teardown layers, every stage in `stage_measure.sh` -- is written against that
surface rather than against JarvisLabs. So a second provider is a second class
with the same eighteen methods, not a rewrite. This file is that class.

What is genuinely different from JarvisLabs, and matters
--------------------------------------------------------
* **There is no CLI.** JarvisLabs is driven through `jl`; RunPod is a GraphQL
  endpoint plus SSH. The API key therefore never reaches argv here -- it is
  read from a 0600 file into a request header, in-process.
* **There is no managed-run concept.** `jl run` starts a tracked background
  job; RunPod has nothing like it, so `run_job` is `nohup` plus three files
  (pid, exit code, log) and `run_status` reads them. That is the same contract
  the stage runner already expects, and it is honest about the one thing that
  matters: a stage that died is distinguishable from a stage still running.
* **Storage is not separable.** JarvisLabs filesystems outlive their instance;
  a RunPod volume is created and destroyed with the pod. `fs_create` therefore
  records the request and returns the pod's own volume, and `fs_delete` is a
  no-op that says so rather than pretending to have deleted something.

Everything the controller does with the returned objects -- `Instance`,
`GpuOffer` -- uses the dataclasses from `jlapi`, imported rather than re-typed,
so a field added there cannot silently diverge here.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

GQL = "https://api.runpod.io/graphql"
# A REAL tag, read from the account's own template list rather than guessed.
# CUDA 13.0 / torch 2.9.1 / Ubuntu 24.04 matches what bootstrap_measure.sh
# builds exllamav3 against on JarvisLabs, and 24.04 ships python3.12 natively.
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster"


class RunPodError(JLError):
    """Same exception family as the JarvisLabs backend, so callers need no branch."""


def _load_key(path: Optional[str] = None) -> str:
    path = path or os.environ.get("RUNPOD_KEY_FILE") or ""
    if path and os.path.isfile(path):
        return open(path, encoding="utf-8").read().strip()
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError(
            "no RunPod credential: set RUNPOD_KEY_FILE to a 0600 file "
            "containing the key, or RUNPOD_API_KEY")
    return key


class RunPod(SSHTransport):
    """Thin, auditable wrapper. `dry` short-circuits every mutating call."""
    # This provider has NO filesystem that outlives its instance, so the
    # whole run must fit on the instance's own disk: a RunPod volume is created with the pod and dies with it.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False

    provider = "runpod"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self._ssh_cache: Dict[int, tuple] = {}

    # -- transport ---------------------------------------------------------
    def _gql(self, query: str, *, timeout: float = 60) -> Dict[str, Any]:
        if self._key is None:
            self._key = _load_key(self._key_file)
        body = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            GQL, data=body,
            headers={"Content-Type": "application/json",
                     # Cloudflare fronts api.runpod.io and answers urllib's
                     # default User-Agent with HTTP 403 "error code: 1010"
                     # (browser integrity check). curl works only because it
                     # sends one. Not optional.
                     "User-Agent": "quant-fidelity-suite/0.1",
                     # header, never argv: `ps` on a shared box would show it
                     "Authorization": "Bearer " + self._key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RunPodError("RunPod HTTP %d: %s"
                              % (exc.code, redact(exc.read()[:300].decode("utf-8", "replace"))))
        except Exception as exc:                          # noqa: BLE001
            raise RunPodError("RunPod request failed: %s" % redact(str(exc)))
        if doc.get("errors"):
            raise RunPodError("RunPod GraphQL: %s"
                              % redact(json.dumps(doc["errors"])[:300]))
        return doc.get("data") or {}

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            _load_key(self._key_file)
            return True
        except RunPodError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise RunPodError("RunPod credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "runpod-graphql"

    def status(self) -> Dict[str, Any]:
        d = self._gql("query { myself { id clientBalance currentSpendPerHr } }")
        return d.get("myself") or {}

    def balance(self) -> Optional[float]:
        try:
            return float(self.status().get("clientBalance"))
        except Exception:                                 # noqa: BLE001
            return None

    # -- catalogue ---------------------------------------------------------
    def gpus(self) -> List[GpuOffer]:
        """Offers, with the honest price.

        `lowestPrice` WITHOUT a cloud filter aggregates across clouds and
        reports a number no one can actually rent -- it reported $0.50/h for an
        H200 NVL that costs $3.79 secure. Both clouds are queried explicitly
        and each becomes its own offer.
        """
        offers: List[GpuOffer] = []
        base = self._gql(
            "query { gpuTypes { id displayName memoryInGb communityCloud secureCloud } }")
        for g in base.get("gpuTypes") or []:
            for secure in (True, False):
                if secure and not g.get("secureCloud"):
                    continue
                if not secure and not g.get("communityCloud"):
                    continue
                q = ('query { gpuTypes(input:{id:"%s"}) { lowestPrice'
                     '(input:{gpuCount:1,secureCloud:%s}) '
                     '{ minimumBidPrice uninterruptablePrice stockStatus } } }'
                     % (g["id"].replace('"', ''), "true" if secure else "false"))
                try:
                    lp = ((self._gql(q).get("gpuTypes") or [{}])[0]
                          .get("lowestPrice") or {})
                except RunPodError:
                    continue
                if not lp.get("stockStatus"):
                    continue
                price = lp.get("uninterruptablePrice") or lp.get("minimumBidPrice")
                if not price:
                    continue
                offers.append(GpuOffer(
                    gpu_type=g["id"], region="secure" if secure else "community",
                    vram_bytes=float(g.get("memoryInGb") or 0) * (1024 ** 3),
                    price=float(price), spot=False,
                    free_devices={"High": 8, "Medium": 3, "Low": 1}.get(
                        lp.get("stockStatus"), 0),
                    workload_type="container",
                    raw={"displayName": g.get("displayName"),
                         "stockStatus": lp.get("stockStatus"),
                         "secureCloud": secure,
                         "bid": lp.get("minimumBidPrice")}))
        return offers

    # -- instances ---------------------------------------------------------
    _POD_FIELDS = ("id name desiredStatus costPerHr machine { podHostId } "
                   "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } "
                   "gpuCount volumeInGb machineId imageName")

    def _pods(self) -> List[Dict[str, Any]]:
        d = self._gql("query { myself { pods { %s } } }" % self._POD_FIELDS)
        return (d.get("myself") or {}).get("pods") or []

    @staticmethod
    def _to_instance(p: Dict[str, Any]) -> Instance:
        rt = p.get("runtime") or {}
        up = float(rt.get("uptimeInSeconds") or 0)
        rate = float(p.get("costPerHr") or 0)
        inst = Instance.from_json({
            # RunPod ids are opaque strings; Instance.machine_id is an int, so
            # the string id is kept in `raw` and used for every API call.
            "machine_id": 0, "status": p.get("desiredStatus") or "",
            "gpu_type": p.get("machineId"), "num_gpus": p.get("gpuCount") or 1,
            "region": None, "is_spot": False,
            # a running TOTAL in USD, matching jlapi's contract
            "cost": rate * up / 3600.0,
            "runtime": up, "fs_id": None,
            "storage_gb": p.get("volumeInGb"), "name": p.get("name"),
            "pod_id": p.get("id"), "cost_per_hr": rate, "raw_pod": p,
        })
        # `Instance.machine_id` is typed int for JarvisLabs, and every consumer
        # -- teardown, the name-based id recovery, the lease reaper -- just
        # passes it back to the provider. Carrying the opaque RunPod id here,
        # after construction, is what makes those paths work unchanged; leaving
        # it 0 made _find_by_name return a machine that does not exist.
        inst.machine_id = p.get("id")
        return inst

    def list_instances(self) -> List[Instance]:
        return [self._to_instance(p) for p in self._pods()]

    def get(self, machine_id: Any) -> Optional[Instance]:
        pid = str(machine_id)
        for p in self._pods():
            if p.get("id") == pid or p.get("name") == pid:
                return self._to_instance(p)
        return None

    def create(self, **kw) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True, **kw}
        gpu = kw.get("gpu_type") or kw.get("gpu")
        if not gpu:
            raise RunPodError("create requires gpu_type")
        name = kw.get("name") or "fidcloud"
        disk = int(kw.get("storage") or kw.get("storage_gb") or 100)
        secure = kw.get("region") != "community"
        pubkey = ""
        kp = self.ssh_key + ".pub"
        if os.path.isfile(kp):
            pubkey = open(kp, encoding="utf-8").read().strip()
        q = ('mutation { podFindAndDeployOnDemand(input:{'
             'cloudType:%s, gpuCount:%d, volumeInGb:%d, containerDiskInGb:%d, '
             'minVcpuCount:%d, minMemoryInGb:%d, gpuTypeId:"%s", name:"%s", '
             'imageName:"%s", ports:"22/tcp", volumeMountPath:"/workspace", '
             'env:[{key:"PUBLIC_KEY", value:"%s"}] '
             '}) { id name costPerHr } }'
             % ("SECURE" if secure else "COMMUNITY", int(kw.get("num_gpus") or 1),
                disk, min(disk, 200),
                # Asking for more vCPU/RAM than the host happens to pair with a
                # GPU is answered SUPPLY_CONSTRAINT, which reads like "no GPUs"
                # and is really "no GPUs with that much CPU". Kept low and
                # overridable rather than hard-coded at 8/32.
                int(kw.get("min_vcpu") or 4), int(kw.get("min_ram_gb") or 16),
                gpu, name,
                kw.get("image") or DEFAULT_IMAGE, pubkey.replace('"', '')))
        d = self._gql(q, timeout=180)
        pod = d.get("podFindAndDeployOnDemand")
        if not pod:
            raise RunPodError("RunPod returned no pod for gpuTypeId=%r (usually "
                              "means no capacity in that cloud)" % gpu)
        return {"machine_id": pod["id"], "pod_id": pod["id"],
                "name": pod.get("name"), "cost_per_hr": pod.get("costPerHr")}

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._gql('mutation { podTerminate(input:{podId:"%s"}) }' % str(machine_id))
        return {"terminated": str(machine_id)}

    def pause(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._gql('mutation { podStop(input:{podId:"%s"}) { id } }' % str(machine_id))
        return {"stopped": str(machine_id)}

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._gql('mutation { podResume(input:{podId:"%s", gpuCount:1}) { id } }'
                  % str(machine_id))
        return {"resumed": str(machine_id)}

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        pid = str(machine_id)
        if pid in self._ssh_cache:
            return self._ssh_cache[pid]
        deadline = time.time() + wait
        while time.time() < deadline:
            for p in self._pods():
                if p.get("id") != pid:
                    continue
                for port in ((p.get("runtime") or {}).get("ports") or []):
                    if port.get("privatePort") == 22 and port.get("isIpPublic"):
                        ep = (port["ip"], int(port["publicPort"]))
                        # The port being PUBLISHED is not sshd accepting on it.
                        self._await_ssh(ep[0], ep[1],
                                        wait=max(60.0, deadline - time.time()))
                        self._ssh_cache[pid] = ep
                        return ep
            time.sleep(10)
        raise RunPodError("pod %s exposed no public SSH port within %ds "
                          "(it may still be provisioning)" % (pid, int(wait)))

    # exec / exec_stdout / upload / download / run_job / run_status /
    # run_logs all come from SSHTransport. They were written here first and
    # then needed again for Vast and Lambda; keeping three copies of a
    # transport whose two subtle bugs (a detached job that never records its
    # exit code, and liveness read from a pid that has already forked away)
    # cost real money to find is how the third copy reintroduces them.

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        """RunPod storage is not separable from the pod.

        A JarvisLabs filesystem outlives its instance, which is what makes a
        preempted spot box cheap to resume. A RunPod volume is created with the
        pod and dies with it. Rather than pretend, this records the request and
        the controller creates the pod with `volumeInGb` set to the same size.
        """
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "runpod volumes are pod-scoped; created with the pod"}

    def fs_delete(self, fs_id: Any) -> Any:
        return {"deleted": False,
                "note": "no separable filesystem on runpod; the volume went "
                        "with the pod at podTerminate"}
