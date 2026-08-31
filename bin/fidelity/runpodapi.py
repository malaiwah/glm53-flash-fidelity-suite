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


class RunPod:
    """Thin, auditable wrapper. `dry` short-circuits every mutating call."""

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
                        self._ssh_cache[pid] = ep
                        return ep
            time.sleep(10)
        raise RunPodError("pod %s exposed no public SSH port within %ds "
                          "(it may still be provisioning)" % (pid, int(wait)))

    def _ssh_argv(self, machine_id: Any) -> List[str]:
        ip, port = self._endpoint(machine_id)
        return ["ssh", "-i", self.ssh_key, "-p", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "root@%s" % ip]

    def exec(self, machine_id: Any, command: str, *,
             timeout: float = 600, check: bool = True) -> Any:
        """Run a shell command over SSH, and CHECK that it worked.

        Returns the same {exit_code, stdout, stderr} shape the JarvisLabs
        backend returns, because the controller reads `exit_code` out of the
        payload rather than trusting the transport's own exit status.
        """
        if self.dry:
            return {"exit_code": 0, "stdout": "", "stderr": "", "dry_run": True}
        argv = self._ssh_argv(machine_id) + ["sh -lc " + shlex.quote(command)]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunPodError("remote command timed out after %ss" % timeout)
        res = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        if check and p.returncode != 0:
            raise RunPodError("remote command exited %s: %s"
                              % (p.returncode, redact((p.stderr or p.stdout)[:400])))
        return res

    def exec_stdout(self, machine_id: Any, command: str, *,
                    timeout: float = 600, check: bool = True) -> str:
        res = self.exec(machine_id, command, timeout=timeout, check=check)
        return str(res.get("stdout") or "")

    def upload(self, machine_id: Any, local: str, remote: str) -> Any:
        if self.dry:
            return {"dry_run": True}
        ip, port = self._endpoint(machine_id)
        argv = ["scp", "-i", self.ssh_key, "-P", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                "-r", local, "root@%s:%s" % (ip, remote)]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise RunPodError("upload failed: %s" % redact(p.stderr[:300]))
        return {"uploaded": remote}

    def download(self, machine_id: Any, remote: str, local: str,
                 *, recursive: bool = True, timeout: float = 900) -> Any:
        if self.dry:
            return {"dry_run": True}
        ip, port = self._endpoint(machine_id)
        argv = ["scp", "-i", self.ssh_key, "-P", str(port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
        if recursive:
            argv.append("-r")
        argv += ["root@%s:%s" % (ip, remote), local]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RunPodError("download failed: %s" % redact(p.stderr[:300]))
        return {"downloaded": local}

    # -- detached jobs -----------------------------------------------------
    # RunPod has no managed-run concept, so this is nohup plus three files.
    # The exit-code file is what makes "died" distinguishable from "still
    # running" -- without it a polling controller cannot tell them apart, which
    # is the failure mode jlapi's exec_stdout docstring describes.
    RUNS = "/workspace/.fidruns"

    def run_job(self, machine_id: Any, command: str) -> Any:
        """Start a detached job that records its OWN exit code.

        The obvious spelling -- `nohup cmd & ( wait $!; echo $? > exit_code )`
        -- does not work and fails in the worst direction: `wait` only knows
        children of the shell that spawned them, and the subshell is not that
        shell, so exit_code was never written. `run_status` then saw no
        exit_code and no live pid and reported a perfectly healthy job as
        FAILED. A controller believing that would tear down a run mid-measure.

        So the command is written to a file and wrapped: the wrapper runs it,
        then writes its own status. setsid detaches it from the SSH session, so
        closing the connection does not kill the stage.
        """
        run_id = "r_%d" % int(time.time() * 1000)
        d = "%s/%s" % (self.RUNS, run_id)
        payload = command.replace("'\\''", "'\\''")
        launcher = (
            "mkdir -p {d} && printf '%s' {cmd} > {d}/run.sh && "
            "setsid sh -c 'sh {d}/run.sh > {d}/output.log 2>&1; "
            "echo $? > {d}/exit_code' </dev/null >/dev/null 2>&1 & "
            "echo $! > {d}/pid; sleep 1; echo launched {rid}"
        ).format(d=d, cmd=shlex.quote(payload), rid=run_id)
        self.exec(machine_id, launcher, timeout=180)
        return {"run_id": run_id, "machine_id": str(machine_id)}

    def run_status(self, run_id: str, machine_id: Any = None) -> Any:
        """Done, running, or dead -- and never "dead" for a healthy job.

        Liveness is decided by `pgrep -f <run_id>`, NOT by the recorded pid.
        `echo $!` captures the backgrounded shell, which forks and exits almost
        immediately, so a pid check reported a perfectly healthy job as dead on
        the very first poll -- and the controller polls right after launch, so
        every stage would have been torn down seconds after starting. The run
        id is in the wrapper's own command line, which is what actually tracks
        the work.
        """
        if machine_id is None:
            raise RunPodError("run_status needs machine_id on this backend")
        d = "%s/%s" % (self.RUNS, run_id)
        out = self.exec_stdout(
            machine_id,
            "if [ -f %s/exit_code ]; then echo DONE $(cat %s/exit_code); "
            "elif pgrep -f %s >/dev/null 2>&1; then echo RUNNING; "
            "else echo GONE; fi" % (d, d, shlex.quote(run_id)),
            timeout=120).strip().split()
        if not out:
            return {"state": "unknown", "run_id": run_id}
        if out[0] == "DONE":
            code = int(out[1]) if len(out) > 1 and out[1].lstrip("-").isdigit() else 1
            return {"state": "succeeded" if code == 0 else "failed",
                    "exit_code": code, "run_id": run_id}
        if out[0] == "RUNNING":
            return {"state": "running", "run_id": run_id}
        return {"state": "failed", "run_id": run_id,
                "note": "no exit_code file and no process matching the run id"}

    def run_logs(self, run_id: str, *, tail: int = 50, machine_id: Any = None) -> Any:
        if machine_id is None:
            raise RunPodError("run_logs needs machine_id on this backend")
        return self.exec_stdout(
            machine_id, "tail -n %d %s/%s/output.log 2>/dev/null || true"
            % (int(tail), self.RUNS, run_id), timeout=120)

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
